# Copyright (c) Panocular AI
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
HeLoCo — Heterogeneity-aware Low-Communication Training
=========================================================
Extends AsyncDiLoCo with two server-side modifications:

  1. Look-ahead worker initialization (Eq. 5):
     Workers receive θ̄ = θ − η·μ·m instead of θ, so they fine-tune
     from the predicted future outer-model position.

  2. Tensor-block directional correction (Algorithm 2):
     Each incoming pseudo-gradient block is compared against the current
     outer momentum. Aligned blocks pass through; anti-aligned blocks are
     shrunk; weakly-aligned blocks are rotated toward momentum while
     preserving the original block magnitude.

The worker class (HeLoCoWorker) is AsyncDiLoCo unchanged — both HeLoCo
modifications live entirely on the server.

Reference: HeLoCo paper https://arxiv.org/pdf/2606.00271.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.profiler
from torch import nn, optim

from panoengine.decentralized.async_diloco import AsyncDiLoCo, AsyncDiLoCoServer

logger: logging.Logger = logging.getLogger(__name__)


class HeLoCoOptimizer(optim.Optimizer):
    """
    Outer optimizer implementing HeLoCo's MLA update rule (Eqs. 18-19):

      m_{t+1} = μ·m_t + (1−μ)·G_t
      θ_{t+1} = θ_t − η·(G_t + μ·m_{t+1})

    Block correction is applied by HeLoCoServer *before* setting p.grad,
    so this optimizer receives the already-corrected gradient G_t and only
    applies the plain momentum-lookahead update.

    Momentum buffers are stored as float32 regardless of parameter dtype
    to avoid precision loss during accumulation.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 0.7,
        momentum: float = 0.9,
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        """
        Apply one outer step using the corrected gradient already in p.grad.

        Eqs. 18-19:
          m_{t+1} = μ·m_t + (1−μ)·G_t
          θ_{t+1} = θ_t − η·(G_t + μ·m_{t+1})
        """
        for group in self.param_groups:
            lr: float = group["lr"]
            mu: float = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                G = p.grad.detach().float()
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                m: torch.Tensor = state["m"]
                m.mul_(mu).add_(G, alpha=1.0 - mu)       # Eq. 18
                p.add_(-(G + mu * m), alpha=lr)           # Eq. 19


@torch.profiler.record_function("heloco.block_correct")
def block_correct(
    pseudo_grads: Dict[str, torch.Tensor],
    momentum_buffers: Dict[str, Optional[torch.Tensor]],
    rho: float = 1.0,
    c_ok: float = 0.2,
    k_s: float = 0.5,
    k_d: float = 1.0,
    kappa: float = 3.0,
    beta_max: float = 0.5,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Algorithm 2: Tensor-block directional correction (paper Eqs. 9-15).

    Three cases per block b:
      cos_b ≥ c_ok  → pass through (Eq. 9)
      cos_b < 0     → shrink: Δ̂_b = Δ_b − β_b·cos_b·‖Δ_b‖·v̂_b, β_b = clamp(k_s·(-cos_b)·conf_b, β_max) (Eqs. 10-11)
      otherwise     → rotate: Δ̂_b = ‖Δ_b‖·ũ_mix/‖ũ_mix‖, ũ_mix=(1−λ_b)û_b+λ_b·v̂_b (Eqs. 12-14)
      conf_b = ‖Δ_b‖/(‖Δ_b‖+κ‖m_b‖+ε) (Eq. 15)

    No .item() calls — all math stays in tensor land, enabling PyTorch to fuse
    ops and avoid per-scalar Python/C++ round-trips.
    """
    corrected: Dict[str, torch.Tensor] = {}

    for name, delta in pseudo_grads.items():
        m = momentum_buffers.get(name)
        if m is None:
            corrected[name] = (rho * delta).to(delta.dtype)
            continue

        delta_f = delta.float()
        m_f     = m.float()
        norm_d  = delta_f.norm()
        norm_m  = m_f.norm()
        safe_d  = norm_d.clamp(min=eps)
        safe_m  = norm_m.clamp(min=eps)

        cos_b  = torch.dot(delta_f.flatten(), m_f.flatten()) / (safe_d * safe_m)  # Eqs. 7-8
        conf_b = norm_d / (norm_d + kappa * norm_m + eps)                          # Eq. 15

        u_hat = delta_f / safe_d
        v_hat = m_f     / safe_m

        # Anti-aligned case (Eqs. 10-11)
        beta_b     = torch.clamp(k_s * (-cos_b) * conf_b, max=beta_max)
        block_anti = delta_f - beta_b * cos_b * norm_d * v_hat

        # Weakly-aligned case (Eqs. 12-14)
        lambda_b   = torch.clamp(k_d * (1.0 - cos_b) * conf_b, max=1.0)
        u_mix      = (1.0 - lambda_b) * u_hat + lambda_b * v_hat
        norm_mix   = u_mix.norm()
        block_weak = torch.where(
            norm_mix > eps,
            norm_d * u_mix / norm_mix.clamp(min=eps),
            delta_f,
        )

        degen = (norm_d < eps) | (norm_m < eps)
        corrected_block = torch.where(
            degen | (cos_b >= c_ok), delta_f,
            torch.where(cos_b < 0.0, block_anti, block_weak),
        )
        corrected[name] = (rho * corrected_block).to(delta.dtype)

    return corrected


class HeLoCoServer(AsyncDiLoCoServer):
    """
    Parameter server for HeLoCo distributed training.

    Extends AsyncDiLoCoServer with:
      1. Look-ahead initialization: both pull-only and post-full-sync
         responses send θ̄ = θ − η·μ·m so workers train from the
         predicted future outer position (Algorithm 1, line 3).
      2. Block correction: before each outer step, the incoming
         pseudo-gradient is passed through block_correct() to align
         it with the outer momentum (Algorithm 2).

    Both modifications hook into the base class (:meth:`_apply_one` and
    :meth:`_build_snapshot_locked`); the wire protocol is exactly
    :meth:`AsyncDiLoCoServer.forward`, so no changes on the worker side
    are required (HeLoCoWorker = AsyncDiLoCo). The look-ahead snapshot is
    computed at most once per revision via the base snapshot cache.

    DyLU is inherited and continues to work when dylu_H > 0.
    """

    def __init__(
        self,
        model: nn.Module,
        outer_optimizer: HeLoCoOptimizer,
        rho: float = 1.0,
        c_ok: float = 0.2,
        k_s: float = 0.5,
        k_d: float = 1.0,
        kappa: float = 3.0,
        beta_max: float = 0.5,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model: Global (outer) model on CPU.
            outer_optimizer: HeLoCoOptimizer bound to model.parameters().
            rho: Arrival weight ρ applied after block correction.
                 Paper recommends 1/√K for K concurrent workers.
            c_ok: Alignment threshold (default 0.2).
            k_s: Anti-aligned shrinkage strength (default 0.5).
            k_d: Weakly-aligned rotation strength (default 1.0).
            kappa: Confidence factor momentum scale κ (default 3.0).
            beta_max: Shrinkage coefficient cap (default 0.5).
            eps: Numerical floor (default 1e-8).
            **kwargs: All :class:`AsyncDiLoCoServer` options (ports, hosts,
                auth, DyLU, quantization, grace period, checkpointing, …).
        """
        # Set HeLoCo attrs before super().__init__ launches the server thread
        self._rho = rho
        self._c_ok = c_ok
        self._k_s = k_s
        self._k_d = k_d
        self._kappa = kappa
        self._beta_max = beta_max
        self._eps = eps
        super().__init__(model=model, outer_optimizer=outer_optimizer, **kwargs)

    @torch.profiler.record_function("heloco.lookahead_snapshot")
    def _lookahead_snapshot(self, names: List[str]) -> Dict[str, torch.Tensor]:
        """Compute θ̄ = θ − η·μ·m (Eq. 5) for the parameters in ``names`` —
        one fragment's slice, or all of them. Must hold self._lock."""
        param_to_hyper: Dict[int, Tuple[float, float]] = {}
        for group in self._outer_optimizer.param_groups:
            lr: float = group["lr"]
            mu: float = group["momentum"]
            for p in group["params"]:
                param_to_hyper[id(p)] = (lr, mu)

        snapshot: Dict[str, torch.Tensor] = {}
        # Group by lr*mu so _foreach_sub can process all params in one C++ call.
        scale_groups: Dict[float, List[Tuple[str, torch.nn.Parameter, torch.Tensor]]] = defaultdict(list)

        for name in names:
            p = self._params_by_name[name]
            state = self._outer_optimizer.state.get(p)
            m = state["m"] if (state and "m" in state) else None
            lr, mu = param_to_hyper.get(id(p), (0.0, 0.0))
            if m is not None and mu > 0.0:
                scale_groups[lr * mu].append((name, p, m))
            else:
                snapshot[name] = p.data.clone().detach()  # momentum not yet seeded — send raw θ

        for scale, items in scale_groups.items():
            ps_f = [p.data.float() for _, p, _ in items]
            ms   = [m for _, _, m in items]
            for (name, p, _), la in zip(items, torch._foreach_sub(ps_f, ms, alpha=scale)):
                snapshot[name] = la.to(p.dtype).detach()

        return snapshot

    def _build_snapshot_locked(self, names: List[str]) -> Dict[str, torch.Tensor]:
        # Workers always receive the look-ahead position θ̄, never raw θ.
        return self._lookahead_snapshot(names)

    @torch.profiler.record_function("heloco.apply")
    def _apply_one(
        self, pseudo_grads: Dict[str, torch.Tensor], fragment: int = 0
    ) -> None:
        """Block-correct one worker's pseudo-gradient, then commit the outer step.

        Clone momentum (brief lock) → block_correct (no lock) → commit (lock).
        Sequential grace-batch workers therefore each correct against the
        momentum updated by the previous worker's step (paper Algorithm 2
        ordering).

        Momentum is cloned only for the parameters this push covers
        (``pseudo_grads`` keys — one fragment's slice under fragment-wise
        sync, which also shrinks the per-push clone to model/P). Block
        correction is per-parameter, so a fragment push corrects and commits
        bitwise the same values a whole-model push of the same deltas would.
        """
        with self._lock:
            mom_bufs: Dict[str, Optional[torch.Tensor]] = {}
            for name in pseudo_grads:
                p = self._params_by_name[name]
                state = self._outer_optimizer.state.get(p)
                m = state["m"] if (state and "m" in state) else None
                mom_bufs[name] = m.clone() if m is not None else None

        # Block correction outside the lock
        corrected = block_correct(
            pseudo_grads,
            mom_bufs,
            rho=self._rho,
            c_ok=self._c_ok,
            k_s=self._k_s,
            k_d=self._k_d,
            kappa=self._kappa,
            beta_max=self._beta_max,
            eps=self._eps,
        )

        with self._lock:
            self._commit_step_locked(corrected, fragment)


# Workers are standard AsyncDiLoCo — all HeLoCo logic lives on the server.
HeLoCoWorker = AsyncDiLoCo
