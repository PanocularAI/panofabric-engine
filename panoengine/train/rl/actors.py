# Copyright (c) Panocular AI.
#
# On-GPU trainer actors for the decentralized_rl coordination strategies. Each class
# extends torchtitan.experiments.rl's PolicyTrainer (the base
# forward_backward/optim_step -- stock token-level GRPO loss -- is reused
# unchanged) with only its strategy's weight-exchange endpoints:
#
#   - DiLoCoManagerTrainer: wraps the model + inner optimizer in torchft's
#     ``local_sgd.DiLoCo`` (stock DiLoCo, Douillard et al., 2311.08105), which
#     hooks the inner optimizer's post-step and drives the pseudo-gradient
#     all-reduce + outer Nesterov step automatically every ``sync_every``
#     steps -- no parameter server, no custom outer optimizer.
#   - HeLoCoPolicyTrainer: full-parameter snapshot/restore keyed by canonical
#     named_parameters() FQNs, for the HeLoCo parameter-server exchange.
#   - SnapshotPolicyTrainer: full-parameter snapshot/restore keyed by
#     state_dict() keys, for relay publishing to vLLM generator workers.
#
# HeLoCoPolicyTrainer and SnapshotPolicyTrainer are analogous snapshot/restore
# pairs kept as separate classes: the two exchanges' shapes differ enough not
# to share code (native fused param names for the server wire vs split
# state_dict keys for the generator load path).

import logging
from datetime import timedelta

import torch
import torch.distributed as dist
from monarch.actor import concurrent_endpoint
from torch.distributed.tensor import distribute_tensor, DTensor

from torchft.checkpointing.http_transport import HTTPTransport
from torchft.local_sgd import DiLoCo
from torchft.manager import Manager
from torchft.process_group import ProcessGroupGloo

from torchtitan.components.checkpoint_utils import canonical_fqn
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer

logger = logging.getLogger(__name__)


class DiLoCoManagerTrainer(PolicyTrainer):
    """PolicyTrainer that syncs across replicas via torchft ``local_sgd.DiLoCo``.

    Each replica is a single-GPU trainer (TP=1, world_size=1): the model params
    are unsharded on this rank, so DiLoCo's pseudo-gradient all-reduce moves whole
    tensors across replicas (the small-model transport assumption). The base
    ``forward_backward`` / ``optim_step`` are reused unchanged -- the DiLoCo sync
    is driven entirely by the inner-optimizer post-step hook.
    """

    def _diloco_state_dict(self) -> dict:
        # Consulted only by the Manager's recovery path (a replica that falls
        # behind after a failed all-reduce). Not exercised in a clean run.
        return {
            "model": self.model.state_dict(),
            "inner_optim": self.optimizers.optimizers[0].state_dict(),
        }

    def _diloco_load_state_dict(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.optimizers.optimizers[0].load_state_dict(state_dict["inner_optim"])

    @concurrent_endpoint
    async def setup_diloco(
        self,
        *,
        lighthouse_address: str,
        replica_id: int,
        num_replicas: int,
        sync_every: int,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
    ) -> None:
        """Build the torchft Manager and enter the DiLoCo context.

        ``min_replica_size = num_replicas`` and ``use_async_quorum=False`` make the
        quorum a synchronous barrier over all workers. ``init_sync=False`` because
        every replica loads the SAME HF checkpoint deterministically, so weights
        already agree at step 0 (skips a fragile DTensor-over-HTTP broadcast).
        """
        # Each replica owns a private TCPStore (single rank; the Manager needs one
        # even at world_size=1). port=0 -> OS-assigned free port.
        self._diloco_store = dist.TCPStore(
            host_name="localhost", port=0, is_master=True, wait_for_workers=False
        )
        pg = ProcessGroupGloo(timeout=timedelta(seconds=60))
        transport = HTTPTransport(timeout=timedelta(seconds=300), num_chunks=0)

        self._diloco_manager = Manager(
            pg=pg,
            min_replica_size=num_replicas,
            use_async_quorum=False,
            load_state_dict=self._diloco_load_state_dict,
            state_dict=self._diloco_state_dict,
            replica_id=f"diloco_{replica_id}",
            store_addr="localhost",
            store_port=self._diloco_store.port,
            rank=0,
            world_size=1,
            lighthouse_addr=lighthouse_address,
            port=19530 + replica_id,
            connect_timeout=timedelta(seconds=120),
            quorum_timeout=timedelta(seconds=900),
            timeout=timedelta(seconds=900),
            checkpoint_transport=transport,
            init_sync=False,
        )

        inner_optimizer = self.optimizers.optimizers[0]
        outer_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=outer_lr,
            momentum=outer_momentum,
            nesterov=True,
        )
        self._diloco = DiLoCo(
            self._diloco_manager,
            [self.model_parts[0]],
            inner_optimizer,
            outer_optimizer,
            backup_device=self.device,
            sync_every=sync_every,
        )
        self._diloco.__enter__()
        logger.info(
            "DiLoCoManagerTrainer replica=%d connected to lighthouse=%s "
            "(sync_every=%d, outer_lr=%.3g, momentum=%.3g)",
            replica_id,
            lighthouse_address,
            sync_every,
            outer_lr,
            outer_momentum,
        )

    @concurrent_endpoint
    async def diloco_step_info(self) -> dict:
        """Expose the Manager's current step / participant count for logging."""
        return {
            "current_step": int(self._diloco_manager.current_step()),
            "num_participants": int(self._diloco_manager.num_participants()),
        }

    @concurrent_endpoint
    async def close_diloco(self) -> None:
        if getattr(self, "_diloco", None) is not None:
            self._diloco.__exit__(None, None, None)
            self._diloco = None
        if getattr(self, "_diloco_manager", None) is not None:
            self._diloco_manager.shutdown(wait=False)
            self._diloco_manager = None


class HeLoCoPolicyTrainer(PolicyTrainer):
    """PolicyTrainer + full-parameter snapshot/restore for the HeLoCo sync.

    The base forward_backward/optim_step are reused unchanged (stock token-level
    GRPO loss); only the parameter-server exchange endpoints are added. The
    controller drives one window like::

        theta_local = trainer.get_full_state_dict_cpu()   # native names, CPU, fp32
        new_global  = client.push(theta_local, speed)     # server outer step
        trainer.load_full_state_dict_cpu(new_global)      # theta <- global
        trainer.push_model_state_dict()                   # refresh generator
        generator.pull_model_state_dict(v)
    """

    @concurrent_endpoint
    async def get_full_state_dict_cpu(
        self, names: list[str] | None = None
    ) -> dict[str, torch.Tensor]:
        """Return full unsharded *parameters* as native-named fp32 CPU tensors.

        Mirrors ``push_model_state_dict``'s unshard but returns to the controller
        and filters out buffers. ``cpu_offload=True`` keeps gathered tensors off
        the GPU; fp32 because the pseudo-gradient subtraction and outer optimizer
        run in fp32 on the server.

        ``names`` restricts the gather to those canonical parameter names (a
        fragment's slice under fragment-wise sync) — the unshard collectives
        run only for the requested subset, so a fragment window pays
        model/num_fragments of device->CPU traffic, not the whole model. The
        controller ``.call()``s every rank with the same ``names``, so the
        collective order stays aligned.
        """
        # Iterate named_parameters() (the same source param_metadata keys the
        # wire buffers on) and unshard each DTensor with full_tensor(). Do NOT
        # use DCP get_model_state_dict: it triggers per-module state_dict hooks
        # (e.g. FusedQKVLinear's wq/wk/wv split, now default) whose synthetic
        # keys have no real submodule, so DCP's FQN resolution raises
        # AttributeError. named_parameters() yields the real (fused) params, so
        # names stay aligned with the server/client ordering.
        # canonical_fqn strips compile/AC/FSDP wrapper segments (e.g. _orig_mod)
        # so keys match the uncompiled meta-model names the server/client are
        # built from (see param_metadata).
        wanted = None if names is None else set(names)
        sd = {}
        for name, param in self.model.named_parameters():
            canonical = canonical_fqn(name)
            if wanted is not None and canonical not in wanted:
                continue
            tensor = param.detach()
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()
            sd[canonical] = tensor.to(device="cpu", dtype=torch.float32)
        return sd

    @concurrent_endpoint
    async def load_full_state_dict_cpu(
        self, global_sd: dict[str, torch.Tensor], clear_optimizer: bool = True
    ) -> None:
        """Load global theta (native-named, fp32, CPU) back into the FSDP/TP model.

        Every rank receives the full dict (the controller `.call()`s the whole
        trainer mesh), so each param is distributed onto its own mesh/placement
        here rather than via ``full_state_dict=True, broadcast_from_rank0=True``:
        that path discovers shardings by walking ``named_children()``, which the
        HF transformers backend overrides to the titan-alias view
        (tok_embeddings/layers/...) whose FQNs never match these canonical
        names — full CPU tensors would then reach DTensor params unconverted
        and fail the copy. Keying off the model's own state dict is
        backend-neutral. ``strict=False`` because we send parameters only
        (buffers are not part of the DiLoCo exchange). After the discontinuous
        theta jump we clear the inner optimizer state (stale momentum would
        bias the next window) but leave ``policy_version`` monotone so
        staleness tracking stays correct.

        ``global_sd`` may be a SUBSET of the parameters (one fragment's merged
        slice under fragment-wise sync) — only matching keys are copied.
        ``clear_optimizer=False`` skips the inner-state clear for those
        mid-cycle fragment adopts, keeping the once-per-full-cycle clearing
        cadence the whole-model path has.
        """
        train_dtype = TORCH_DTYPE_MAP[self.config.training.dtype]
        # Copy the incoming global theta straight into named_parameters() (same
        # keying as param_metadata / get_full_state_dict_cpu), redistributing
        # each full CPU tensor onto its param's mesh/placement. Avoids DCP
        # set_model_state_dict, which trips on FusedQKVLinear's hook-synthesized
        # wq/wk/wv keys (no real submodule -> FQN AttributeError). Parameters
        # only (buffers are not part of the DiLoCo exchange).
        local_params = {canonical_fqn(n): p for n, p in self.model.named_parameters()}
        with torch.no_grad():
            for k, v in global_sd.items():
                ref = local_params.get(k)
                if ref is None:
                    continue
                v = v.to(dtype=train_dtype, device=ref.device)
                if isinstance(ref, DTensor):
                    v = distribute_tensor(v, ref.device_mesh, ref.placements)
                ref.copy_(v)

        if clear_optimizer:
            self.optimizers.zero_grad(set_to_none=True)
            for opt in self.optimizers:
                opt.state.clear()
        logger.info(
            "HeLoCoPolicyTrainer loaded global theta (policy_version=%d)",
            self.policy_version,
        )


class SnapshotPolicyTrainer(PolicyTrainer):
    """PolicyTrainer + full-parameter snapshot/restore endpoints.

    Used by AsyncInferenceReplica to read the whole model out for relay
    publishing (see replicas.py's AsyncInferenceReplica.setup_async). The base
    forward_backward/optim_step are reused unchanged (stock token-level
    GRPO loss); only the full-state-dict exchange endpoints are added::

        theta = trainer.get_full_state_dict_cpu()   # native names, CPU, fp32
        trainer.load_full_state_dict_cpu(theta)      # theta -> model
    """

    @concurrent_endpoint
    async def get_full_state_dict_cpu(self) -> dict[str, torch.Tensor]:
        """Return the full unsharded model state dict as fp32 CPU tensors.

        Uses ``self.model.state_dict()`` — the SAME source ``push_model_state_dict``
        stages for the generators — so the relay-published weights carry the exact
        keys the worker's vLLM engine consumes on pull. In particular FusedQKVLinear's
        state_dict hooks split the fused ``wqkv`` back into ``wq``/``wk``/``wv``,
        the layout the generator load path expects; a ``named_parameters()`` dump
        would instead emit the fused key and the worker would load mismatched
        attention weights. DCP ``get_model_state_dict`` can't be used here — it
        trips on FusedQKVLinear's synthetic keys during FQN resolution — so unshard
        each DTensor with ``full_tensor()`` directly (relay shards are torch.save'd,
        so full CPU tensors are required)."""
        sd = {}
        for name, tensor in self.model.state_dict().items():
            tensor = tensor.detach()
            if isinstance(tensor, DTensor):
                tensor = tensor.full_tensor()
            sd[name] = tensor.to(device="cpu", dtype=torch.float32)
        return sd

    @concurrent_endpoint
    async def load_full_state_dict_cpu(
        self, global_sd: dict[str, torch.Tensor]
    ) -> None:
        """Load a full state dict (state_dict()-keyed, fp32, CPU) back into the
        FSDP/TP model — the inverse of ``get_full_state_dict_cpu``.

        Distributes each incoming full tensor onto the placement of the model's
        current state_dict entry, then loads via the model's own
        ``load_state_dict`` (whose FusedQKVLinear hooks re-fuse wq/wk/wv), which
        is fused-QKV-safe unlike DCP ``set_model_state_dict``. ``strict=False``
        tolerates any absent buffers. After the discontinuous theta jump we clear
        the inner optimizer state (stale momentum would bias the next window) but
        leave ``policy_version`` monotone so staleness tracking stays correct.
        """
        train_dtype = TORCH_DTYPE_MAP[self.config.training.dtype]
        ref_sd = self.model.state_dict()
        to_load = {}
        for k, v in global_sd.items():
            ref = ref_sd.get(k)
            if ref is None:
                continue
            v = v.to(dtype=train_dtype, device=ref.device)
            if isinstance(ref, DTensor):
                v = distribute_tensor(v, ref.device_mesh, ref.placements)
            to_load[k] = v
        self.model.load_state_dict(to_load, strict=False)

        self.optimizers.zero_grad(set_to_none=True)
        for opt in self.optimizers:
            opt.state.clear()
        logger.info(
            "SnapshotPolicyTrainer loaded full theta (policy_version=%d)",
            self.policy_version,
        )
