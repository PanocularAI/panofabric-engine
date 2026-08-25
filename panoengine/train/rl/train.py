# Copyright (c) Panocular AI.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Panocular AI.
#
# Worker launch entrypoint for every decentralized_rl coordination strategy (the
# --config picks the strategy; mirrors torchtitan.experiments.rl.train):
#
#   python -m panoengine.train.rl.train \
#       --module decentralized_rl --config rl_heloco_qwen3_0_6b
#
# and likewise for rl_diloco_* / rl_async_inference_* configs, or a
# benchmark config via --module panoengine.train.rl.___benchmark
# --config bench_local_qwen3_0_6b. Launch plumbing (server/lighthouse/relay
# addresses, replica ids) comes from the environment -- exported by the
# ___benchmark/launch_*.sh scripts -- rather than CLI flags, so one launch
# script serves any config; see _ENV_OVERRIDES below. The async_inference
# inference-worker role has its own entrypoint (async_inference/worker.py):
# it spawns no trainer, so it doesn't fit this replica lifecycle.

import os

# Must be set before torch is imported (which the monarch/torchtitan imports
# below do transitively) so the allocator config applies process-wide.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import asyncio  # noqa: E402
import itertools  # noqa: E402
import logging  # noqa: E402
import socket  # noqa: E402

from monarch.actor import ProcMesh, this_host  # noqa: E402
from monarch.spmd import setup_torch_elastic_env_async  # noqa: E402

from torchtitan.config import ConfigManager  # noqa: E402
from torchtitan.tools.logging import init_logger  # noqa: E402
from torchtitan.experiments.rl.train import (  # noqa: E402
    _compute_generator_world_size,
    _compute_trainer_world_size,
    PerHostProvisioner as _UpstreamProvisioner,
)

logger = logging.getLogger(__name__)

#: config field -> (env-var candidates, cast). Applied after CLI parsing, to
#: fields the parsed config actually has (each strategy's Config declares a
#: different subset). First set (non-empty) env var wins.
_ENV_OVERRIDES = {
    "server_address": (("DILOCO_SERVER_ADDR",), str),
    "heartbeat_address": (("DILOCO_HB_ADDR",), str),
    "lighthouse_address": (("DILOCO_LIGHTHOUSE_ADDR",), str),
    "num_replicas": (("DILOCO_NUM_REPLICAS",), int),
    "relay_addresses": (("ASYNC_INFERENCE_RELAY_ADDRS",), str),
    "replica_id": (("DILOCO_REPLICA_ID", "ASYNC_INFERENCE_REPLICA_ID"), int),
    "rollout_queue_address": (("ROLLOUT_QUEUE_ADDR",), str),
}


# Below the OS ephemeral range (32768+), so dynamic allocations by unrelated
# processes can never take these; distinct from the launchers' 29500+ rdzv
# ports and the coordinator servers' 295xx/87xx defaults.
_ELASTIC_PORT_BASE = 29800
_mesh_counter = itertools.count()


async def setup_mesh_elastic_env(mesh: ProcMesh) -> None:
    """``setup_torch_elastic_env_async`` with a deterministic, host-unique port.

    Monarch's default probes for a free MASTER_PORT (bind-0 then close; rank
    0's TCPStore binds it for real later) — a check-then-use race: sibling
    replicas on one host probing concurrently can be handed the SAME port,
    and the loser dies with EADDRINUSE deep in vLLM/torch.distributed init.

    Launchers partition a host's replicas by disjoint, contiguous
    CUDA_VISIBLE_DEVICES ranges, and a replica spawns at most one mesh per
    GPU it owns, so (first visible device + this process's mesh counter) is
    unique across every mesh on the host. Without CUDA_VISIBLE_DEVICES there
    is no partition (sole tenant / remote host meshes) — keep Monarch's pick.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        await setup_torch_elastic_env_async(mesh)
        return
    port = _ELASTIC_PORT_BASE + int(visible.split(",")[0]) + next(_mesh_counter)
    await setup_torch_elastic_env_async(
        mesh, master_addr=socket.gethostname(), master_port=port
    )


class PerHostProvisioner(_UpstreamProvisioner):
    """Upstream PerHostProvisioner, but allocating out of the devices this
    process is already restricted to instead of absolute indices.

    Our launchers partition a node across replicas via CUDA_VISIBLE_DEVICES
    (e.g. replica 1 gets "2,3"). Upstream's allocator emits absolute "0"/"1"
    bootstraps, which would silently move every replica onto the same
    physical GPUs (each replica's vLLM then profiles a GPU another replica
    already filled, and whoever loses the race gets a starved KV cache).
    Here the bootstrap ids are drawn from the parent's visible-device pool,
    so children stay inside their replica's slice.

    Lives here (not in torchtitan.experiments.rl) because we treat that base
    RL experiment strictly as upstream: this subclass carries the one
    behavior our launchers need that its PerHostProvisioner lacks.
    """

    def __init__(self, total_gpus: int = 8):
        super().__init__(total_gpus=total_gpus)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            self.device_pool = [d.strip() for d in visible.split(",") if d.strip()]
        else:
            self.device_pool = [str(i) for i in range(total_gpus)]

    def allocate(self, num_gpus: int):
        if num_gpus > self.available:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but only {self.available} "
                f"available (total={self.total_gpus}, allocated={self.next_gpu})"
            )
        if self.next_gpu + num_gpus > len(self.device_pool):
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but CUDA_VISIBLE_DEVICES exposes "
                f"only {len(self.device_pool)} device(s) "
                f"({','.join(self.device_pool)}), {self.next_gpu} already allocated"
            )
        gpu_ids = self.device_pool[self.next_gpu : self.next_gpu + num_gpus]
        self.next_gpu += num_gpus

        def _bootstrap():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
            # TODO: remove once Monarch/PyTorch fixes concurrent import during
            # unpickling (mirrors upstream's bootstrap).
            import torch  # noqa: F401

        return _bootstrap


def _ensure_cuda_toolchain() -> None:
    """Point the CUDA toolchain at a >=12 toolkit before spawning actors.

    vLLM's FlashInfer kernels JIT-compile with nvcc; the box's default
    /usr/bin/nvcc may be <12. Set CUDA_HOME/PATH so the Monarch-spawned
    generator subprocess inherits a working nvcc.
    """
    for home in ("/usr/local/cuda-12.8", "/usr/local/cuda-12.3", "/usr/local/cuda-12"):
        if os.path.isfile(os.path.join(home, "bin", "nvcc")):
            os.environ["CUDA_HOME"] = home
            os.environ["PATH"] = (
                os.path.join(home, "bin") + os.pathsep + os.environ.get("PATH", "")
            )
            logger.info("CUDA toolchain set to %s", home)
            return
    logger.warning("no CUDA >=12 toolkit found; FlashInfer JIT may fail")


async def main() -> None:
    _ensure_cuda_toolchain()
    # Give THIS process a stdout handler. init_logger() is called inside the
    # actor processes (rl/actors/trainer.py, .../generator.py) and by pretrain's
    # torchtitan/train.py, but never by the replica/worker mains -- so their root
    # logger had no handler and Python's `lastResort` fallback (level WARNING)
    # silently dropped every logger.info.
    #
    # That is not cosmetic: `_run_window` logs "[replica %d] step: %d" as the
    # documented progress contract an external supervisor greps for, and
    # controld's SkyPilot handle greps exactly that to drive its readiness gate.
    # With the message discarded, progress() always returned 0, quorum never
    # advanced past "ready 0/1", and healthy 4B decoupled runs were killed at the
    # quorum deadline -- six of them, over ~10,700 captured log lines with not a
    # single step line between them, while the trainer was in fact training
    # (verified locally: reward_mean=0.094, finite loss, PS applied_pushes=1).
    init_logger()
    config = ConfigManager().parse_args()
    for field, (env_names, cast) in _ENV_OVERRIDES.items():
        if not hasattr(config, field):
            continue
        for env_name in env_names:
            if os.environ.get(env_name):
                setattr(config, field, cast(os.environ[env_name]))
                break

    replica = config.build()
    trainer_ws = _compute_trainer_world_size(config.trainer.parallelism)
    generator_ws = _compute_generator_world_size(config.generator.parallelism)
    # Spawn config.num_generators independent engines, each on its own GPU
    # slice of this replica's CUDA_VISIBLE_DEVICES pool; setup_async's
    # GeneratorRouter round-robins requests across them and fans weight
    # refreshes out to all of them.
    provisioner = PerHostProvisioner(
        total_gpus=trainer_ws + generator_ws * config.num_generators
    )
    trainer_mesh = this_host().spawn_procs(
        per_host={"gpus": trainer_ws}, bootstrap=provisioner.allocate(trainer_ws)
    )
    generator_meshes = [
        this_host().spawn_procs(
            per_host={"gpus": generator_ws},
            bootstrap=provisioner.allocate(generator_ws),
        )
        for _ in range(config.num_generators)
    ]
    try:
        await replica.setup_async(
            trainer_mesh=trainer_mesh, generator_meshes=generator_meshes
        )
        await replica.train()
    finally:
        await replica.close()


if __name__ == "__main__":
    asyncio.run(main())
