# Copyright (c) Panocular AI
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional
from unittest import TestCase

import torch
from torch import nn, optim

from panoengine.decentralized.heloco import HeLoCoOptimizer, HeLoCoServer, HeLoCoWorker, block_correct
from panoengine.decentralized.async_diloco import AsyncDiLoCo, AsyncDiLoCoServer
from panoengine.decentralized.tests.async_diloco_test import push_pull


# Sequential reference for parity checks — paper Eqs. 9-15 exactly.
def _block_correct_ref(
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
    corrected: Dict[str, torch.Tensor] = {}
    for name, delta in pseudo_grads.items():
        m = momentum_buffers.get(name)
        if m is None:
            corrected[name] = (rho * delta).to(delta.dtype)
            continue
        delta_f = delta.float()
        m_f = m.float()
        norm_d = delta_f.norm().item()
        norm_m = m_f.norm().item()
        if norm_d < eps or norm_m < eps:
            corrected[name] = (rho * delta_f).to(delta.dtype)
            continue
        cos_b  = torch.dot(delta_f.flatten(), m_f.flatten()).item() / (norm_d * norm_m)
        conf_b = norm_d / (norm_d + kappa * norm_m + eps)
        if cos_b >= c_ok:
            corrected_block = delta_f
        elif cos_b < 0:
            v_hat  = m_f / norm_m
            beta_b = min(k_s * (-cos_b) * conf_b, beta_max)
            corrected_block = delta_f - beta_b * cos_b * norm_d * v_hat
        else:
            u_hat    = delta_f / norm_d
            v_hat    = m_f / norm_m
            lambda_b = min(k_d * (1.0 - cos_b) * conf_b, 1.0)
            u_mix    = (1.0 - lambda_b) * u_hat + lambda_b * v_hat
            norm_mix = u_mix.norm().item()
            corrected_block = norm_d * u_mix / norm_mix if norm_mix > eps else delta_f
        corrected[name] = (rho * corrected_block).to(delta.dtype)
    return corrected


def _make_model(d: int = 4) -> nn.Module:
    return nn.Sequential(nn.Linear(d, d))


class TestHeLoCoOptimizer(TestCase):
    def test_update_rule_eqs_18_19(self) -> None:
        """Verify Eqs. 18-19: m_{t+1}=μm+(1-μ)G, θ_{t+1}=θ-η(G+μm_{t+1})."""
        lr, mu = 0.1, 0.9
        G = torch.tensor([2.0, -1.0])
        p = torch.nn.Parameter(torch.tensor([3.0, 5.0]))

        opt = HeLoCoOptimizer([p], lr=lr, momentum=mu)
        p.grad = G.clone()
        opt.step()

        m_expected = (1.0 - mu) * G
        p_expected = torch.tensor([3.0, 5.0]) - lr * (G + mu * m_expected)

        torch.testing.assert_close(opt.state[p]["m"], m_expected)
        torch.testing.assert_close(p.data, p_expected)

    def test_second_step_uses_updated_momentum(self) -> None:
        lr, mu = 0.1, 0.9
        G1 = torch.tensor([1.0])
        G2 = torch.tensor([3.0])
        p = torch.nn.Parameter(torch.tensor([0.0]))

        opt = HeLoCoOptimizer([p], lr=lr, momentum=mu)
        p.grad = G1.clone()
        opt.step()
        m1 = (1.0 - mu) * G1

        p.grad = G2.clone()
        opt.step()
        m2 = mu * m1 + (1.0 - mu) * G2

        torch.testing.assert_close(opt.state[p]["m"], m2, atol=1e-6, rtol=0)


class TestBlockCorrect(TestCase):
    def _correct(self, delta, m, **kwargs) -> torch.Tensor:
        return block_correct({"p": delta}, {"p": m}, **kwargs)["p"]

    def test_aligned_block_unchanged(self) -> None:
        delta = torch.tensor([2.0, 0.0])
        m = torch.tensor([1.0, 0.0])
        torch.testing.assert_close(self._correct(delta, m, rho=1.0, c_ok=0.2), delta)

    def test_anti_aligned_exact_math(self) -> None:
        """β_b = clamp(k_s·(-cos)·conf, β_max) — paper Eq. 11."""
        # cos=-1, k_s=0.5, kappa=0 → conf≈1, β=clamp(0.5·1·1, 0.5)=0.5
        # corrected = [-1,0] - 0.5·(-1)·1·[1,0] = [-0.5, 0]
        out = self._correct(
            torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 0.0]),
            rho=1.0, c_ok=0.2, k_s=0.5, beta_max=0.5, kappa=0.0,
        )
        torch.testing.assert_close(out, torch.tensor([-0.5, 0.0]), atol=1e-5, rtol=0)

        # conf < 1, clamp doesn't bind: β = clamp(1·1·(1/3), 0.5) = 1/3
        delta2, m2 = torch.tensor([-1.0, 0.0]), torch.tensor([2.0, 0.0])
        conf2 = 1.0 / (1.0 + 1.0 * 2.0)
        beta2 = min(1.0 * 1.0 * conf2, 0.5)
        out2 = self._correct(delta2, m2, rho=1.0, c_ok=0.2, k_s=1.0, beta_max=0.5, kappa=1.0)
        torch.testing.assert_close(out2, torch.tensor([-1.0 + beta2, 0.0]), atol=1e-4, rtol=0)

    def test_weakly_aligned_rotated_toward_momentum(self) -> None:
        delta = torch.tensor([0.0, 1.0])
        m = torch.tensor([1.0, 0.0])
        out = self._correct(delta, m, rho=1.0, c_ok=0.5, k_d=1.0, kappa=0.0)
        self.assertGreater(out[0].item(), 0.0)

    def test_weakly_aligned_norm_preserved(self) -> None:
        delta = torch.tensor([0.0, 2.0])
        m = torch.tensor([1.0, 0.0])
        out = self._correct(delta, m, rho=1.0, c_ok=0.5, k_d=1.0, kappa=0.0)
        torch.testing.assert_close(out.norm(), torch.tensor(2.0), atol=1e-5, rtol=0)

    def test_none_momentum_passthrough(self) -> None:
        delta = torch.tensor([-1.0, 0.0])
        result = block_correct({"p": delta}, {"p": None}, rho=0.7)
        torch.testing.assert_close(result["p"], 0.7 * delta)

    def test_tiny_norm_passthrough(self) -> None:
        """Near-zero delta or momentum → pass through unchanged."""
        m = torch.tensor([1.0, 0.0])
        torch.testing.assert_close(
            self._correct(torch.tensor([1e-10, 0.0]), m, rho=1.0, eps=1e-8),
            torch.tensor([1e-10, 0.0]),
        )
        torch.testing.assert_close(
            self._correct(torch.tensor([1.0, 0.0]), torch.tensor([1e-10, 0.0]), rho=1.0, eps=1e-8),
            torch.tensor([1.0, 0.0]),
        )

    def test_output_dtype_matches_input(self) -> None:
        delta = torch.tensor([1.0, 0.0], dtype=torch.float16)
        m = torch.tensor([1.0, 0.0], dtype=torch.float32)
        self.assertEqual(block_correct({"p": delta}, {"p": m}, rho=1.0)["p"].dtype, torch.float16)


class TestBlockCorrectParity(TestCase):
    """block_correct must match the sequential paper reference on all cases."""

    _KW = dict(rho=0.8, c_ok=0.2, k_s=0.5, k_d=1.0, kappa=3.0, beta_max=0.5, eps=1e-8)

    def _check(self, grads, moms, **kw):
        kw = {**self._KW, **kw}
        ref = _block_correct_ref(grads, moms, **kw)
        out = block_correct(grads, moms, **kw)
        for name in grads:
            torch.testing.assert_close(out[name], ref[name], atol=1e-5, rtol=1e-5,
                                       msg=f"parity failure for '{name}'")

    def test_parity_all_cases_mixed(self) -> None:
        """Multiple params hitting different correction branches simultaneously."""
        self._check(
            {
                "aligned": torch.tensor([2.0, 0.0]),
                "anti":    torch.tensor([-1.0, 0.0]),
                "weak":    torch.tensor([0.0, 1.0]),
                "no_mom":  torch.randn(5, 3),
            },
            {
                "aligned": torch.tensor([1.0, 0.0]),
                "anti":    torch.tensor([1.0, 0.0]),
                "weak":    torch.tensor([1.0, 0.0]),
                "no_mom":  None,
            },
        )

    def test_parity_random_large(self) -> None:
        """Random multi-dim params of varying shapes."""
        torch.manual_seed(42)
        grads = {f"p{i}": torch.randn(*s) for i, s in enumerate([(64, 64), (128,), (32, 8, 4)])}
        moms  = {name: torch.randn_like(g) for name, g in grads.items()}
        self._check(grads, moms)


class TestHeLoCoServer(TestCase):
    def _push_pull(self, server, model, full_sync=True, pseudo_grad_value=1.0, speed=1.0):
        received, new_steps, _, _ = push_pull(
            server.address(),
            model,
            full_sync=full_sync,
            speed=speed,
            grad_value=pseudo_grad_value,
        )
        received["__new_steps__"] = torch.tensor([float(new_steps)])
        return received

    def test_pull_only_sends_lookahead(self) -> None:
        """Pull-only sends θ̄ = θ − η·μ·m when momentum is seeded."""
        model = _make_model()
        lr, mu = 0.1, 0.9
        outer_opt = HeLoCoOptimizer(model.parameters(), lr=lr, momentum=mu)
        server = HeLoCoServer(model, outer_opt, port=0)
        m_val = 2.0
        for p in model.parameters():
            outer_opt.state[p] = {"m": torch.full_like(p, m_val, dtype=torch.float32)}
        theta = {n: p.data.clone() for n, p in model.named_parameters()}
        received = self._push_pull(server, model, full_sync=False)
        for name in theta:
            torch.testing.assert_close(received[name], theta[name] - lr * mu * m_val,
                                       atol=1e-5, rtol=0)

    def test_full_sync_sends_lookahead_after_step(self) -> None:
        """After full sync, worker receives θ̄_{t+1} = θ_{t+1} − η·μ·m_{t+1}."""
        model = _make_model()
        lr, mu = 0.5, 0.8
        outer_opt = HeLoCoOptimizer(model.parameters(), lr=lr, momentum=mu)
        server = HeLoCoServer(model, outer_opt, port=0)
        received = self._push_pull(server, model, full_sync=True, pseudo_grad_value=1.0)
        for name, p in model.named_parameters():
            m = outer_opt.state[p]["m"]
            expected = (p.data.float() - lr * mu * m).to(p.dtype)
            torch.testing.assert_close(received[name], expected, atol=1e-5, rtol=0)

    def test_block_correction_applied_on_full_sync(self) -> None:
        """Anti-aligned pseudo-grad is shrunk more by HeLoCo than uncorrected baseline."""
        d = 1
        model_corr = nn.Sequential(nn.Linear(d, d, bias=False))
        model_no   = nn.Sequential(nn.Linear(d, d, bias=False))
        with torch.no_grad():
            for pc, pn in zip(model_corr.parameters(), model_no.parameters()):
                pn.data.copy_(pc.data)

        lr = 0.1
        outer_opt_corr = HeLoCoOptimizer(model_corr.parameters(), lr=lr, momentum=0.0)
        for p in model_corr.parameters():
            outer_opt_corr.state[p] = {"m": torch.full_like(p, 1.0, dtype=torch.float32)}
        server_corr = HeLoCoServer(model_corr, outer_opt_corr, port=0,
                                   k_s=1.0, beta_max=0.9, kappa=0.0)
        server_no = AsyncDiLoCoServer(model_no, optim.SGD(model_no.parameters(), lr=lr), port=0)

        initial = {n: p.data.clone() for n, p in model_corr.named_parameters()}
        received_corr = self._push_pull(server_corr, model_corr, pseudo_grad_value=-1.0)
        received_no   = self._push_pull(server_no,   model_no,   pseudo_grad_value=-1.0)

        for name in initial:
            diff_corr = (received_corr[name] - initial[name]).abs().mean().item()
            diff_no   = (received_no[name]   - initial[name]).abs().mean().item()
            self.assertLess(diff_corr, diff_no)

    def test_full_training_loop(self) -> None:
        """End-to-end: HeLoCoWorker trains and syncs with HeLoCoServer."""
        d = 4
        global_model = _make_model(d)
        outer_opt = HeLoCoOptimizer(global_model.parameters(), lr=0.1, momentum=0.9)
        server = HeLoCoServer(global_model, outer_opt, port=0)

        worker_model = _make_model(d)
        worker_model.load_state_dict(global_model.state_dict())
        inner_opt = optim.SGD(worker_model.parameters(), lr=0.01)
        sync_every = 3

        with HeLoCoWorker(server.address(), worker_model, inner_opt, sync_every=sync_every):
            x, y = torch.randn(2, d), torch.randint(0, d, (2,))
            for _ in range(sync_every):
                inner_opt.zero_grad()
                nn.CrossEntropyLoss()(worker_model(x), y).backward()
                inner_opt.step()

        self.assertTrue(any("m" in s for s in outer_opt.state.values()))


class TestHeLoCoWorkerRejoin(TestCase):
    def test_rejoined_worker_gets_lookahead_params(self) -> None:
        """Worker rejoining after a sync receives θ̄, not raw θ."""
        d = 4
        global_model = _make_model(d)
        lr, mu = 0.5, 0.8
        outer_opt = HeLoCoOptimizer(global_model.parameters(), lr=lr, momentum=mu)
        server = HeLoCoServer(global_model, outer_opt, port=0)
        x, y = torch.randn(2, d), torch.randint(0, d, (2,))

        w1 = _make_model(d)
        w1.load_state_dict(global_model.state_dict())
        o1 = optim.SGD(w1.parameters(), lr=0.01)
        with HeLoCoWorker(server.address(), w1, o1, sync_every=2):
            for _ in range(2):
                o1.zero_grad()
                nn.CrossEntropyLoss()(w1(x), y).backward()
                o1.step()

        expected = {}
        for name, p in global_model.named_parameters():
            state = outer_opt.state.get(p)
            m = state["m"] if state and "m" in state else None
            expected[name] = (p.data.float() - lr * mu * m).to(p.dtype) if m is not None else p.data.clone()

        w2, o2 = _make_model(d), optim.SGD(_make_model(d).parameters(), lr=0.01)
        with HeLoCoWorker(server.address(), w2, o2, sync_every=100):
            for name, p in w2.named_parameters():
                torch.testing.assert_close(p.data.cpu(), expected[name], atol=1e-5, rtol=0)

    def test_training_continues_after_rejoin(self) -> None:
        """Params and momentum keep updating through a leave-rejoin cycle."""
        d = 4
        global_model = _make_model(d)
        outer_opt = HeLoCoOptimizer(global_model.parameters(), lr=0.5, momentum=0.9)
        server = HeLoCoServer(global_model, outer_opt, port=0)
        x, y = torch.randn(2, d), torch.randint(0, d, (2,))
        initial = {n: p.data.clone() for n, p in global_model.named_parameters()}

        for _ in range(2):
            wm = _make_model(d)
            wo = optim.SGD(wm.parameters(), lr=0.01)
            with HeLoCoWorker(server.address(), wm, wo, sync_every=2):
                for _ in range(2):
                    wo.zero_grad()
                    nn.CrossEntropyLoss()(wm(x), y).backward()
                    wo.step()

        self.assertTrue(any(not torch.equal(p.data, initial[n]) for n, p in global_model.named_parameters()))
        self.assertTrue(any("m" in s for s in outer_opt.state.values()))
