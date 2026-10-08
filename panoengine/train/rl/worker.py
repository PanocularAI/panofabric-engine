# Copyright (c) Panocular AI.
#
# Standalone inference-worker process for the async-inference relay swarm.
#
# This role has NO trainer actor at all. `torchtitan.rl.controller
# .Controller.setup_async` always spawns a `TrainerActor` and binds
# TorchStore's storage volumes to the trainer mesh -- a coupling this role
# deliberately doesn't have, so it can't reuse that setup path. This spawns
# just a generator actor and binds TorchStore to ITS OWN mesh instead; weight
# updates arrive exclusively through the relay tier (panoengine.train.rl.relay),
# never through a local trainer push.
#
# This worker fetches weights via the relay tier (panoengine.train.rl.relay)
# and pushes its generated rollouts to the standalone rollout-queue process
# (rollout_queue.py) via RolloutQueuePushClient -- workers are trusted here,
# so this is a plain push/queue.


import asyncio
import itertools
import logging
import os
import time
from dataclasses import dataclass, field, replace
from typing import Annotated

import tyro

# Must be set before torch is imported (transitively, below).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torchstore as ts  # noqa: E402

from torchtitan.observability.logging import init_logger  # noqa: E402

from torchtitan.config import CompileConfig  # noqa: E402
from panoengine.decentralized.relay import RelayClient  # noqa: E402
from panoengine.decentralized.rollout_queue import (
    RolloutQueuePushClient,
)  # noqa: E402

from panoengine.train.rl.train import (
    _bootstrap_generator,
    _ensure_cuda_toolchain,
    PerHostProvisioner,
    setup_mesh_elastic_env,
    spawn_gpu_procs,
)  # noqa: E402
from renderers import Qwen3RendererConfig  # noqa: E402
from torchtitan.components.renderer import from_renderers, RendererConfig  # noqa: E402
from torchtitan.components.tokenizer import HuggingFaceTokenizer  # noqa: E402
from torchtitan.models.common.decoder import Decoder  # noqa: E402
from torchtitan.rl.distributed.actors.generator import VLLMGeneratorActor  # noqa: E402
from torchtitan.rl.generator import VLLMGenerator  # noqa: E402
from torchtitan.rl.rollout.rollouter import Rollouter  # noqa: E402
from torchtitan.rl.train import (
    _compute_generator_world_size as _compute_world_size,
)  # noqa: E402

logger = logging.getLogger(__name__)


def _site_relay_cache() -> str | None:
    """Where this site's generators share relay shards (RelayClient.site_cache_dir):
    the shared cache root `make ensure` uses -- $PF_ENV_CACHE_DIR, else the real
    home behind a Slurm job's HOME. None (download
    directly) on hosts without one, e.g. a cloud VM; PF_RELAY_SITE_CACHE=0 opts out."""
    if os.environ.get("PF_RELAY_SITE_CACHE") == "0":
        return None
    root = os.environ.get("PF_ENV_CACHE_DIR")
    home = os.path.expanduser("~")
    if not root and "/.sky_clusters/" in home:
        root = home.split("/.sky_clusters/")[0] + "/.panofabric-cache"
    return os.path.join(root, "relay") if root else None


class AsyncInferenceWorker:
    """Inference-only node in the async-inference relay swarm (see module
    docstring for the trainer-less design and rollout-feedback scope
    boundary)."""

    @dataclass(kw_only=True, slots=True)
    class Config:
        # Set by the presets, not the CLI -- the same as Controller.Config.model.
        # The concrete `X | None` annotations matter: with a bare `object`, tyro
        # narrows to the default instance's type and recurses anyway, Suppress
        # notwithstanding.
        model: Annotated[Decoder.Config | None, tyro.conf.Suppress] = None
        hf_assets_path: str = ""
        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        rollouter: Annotated[Rollouter.Config | None, tyro.conf.Suppress] = None
        tokenizer: HuggingFaceTokenizer.Config = field(
            default_factory=HuggingFaceTokenizer.Config
        )
        renderer: RendererConfig = field(
            default_factory=lambda: from_renderers(
                Qwen3RendererConfig(enable_thinking=False)
            )
        )
        compile: Annotated[CompileConfig | None, tyro.conf.AvoidSubcommands] = None
        group_size: int = 8
        """Rollouts per group (mirrors the trainer configs' group_size)."""
        groups_per_round: int = 2
        """Rollout groups generated (and pushed to the trainer) per
        checkpoint load."""
        relay_addresses: str = ""
        """Comma-separated relay server base URLs. Required -- launch
        plumbing (usually from $ASYNC_INFERENCE_RELAY_ADDRS), so -- like
        AsyncInferenceReplica.relay_addresses -- it's checked at construction time
        (RelayClient itself raises on an empty list) rather than here:
        ConfigManager calls the --config function with zero args before
        overlaying CLI flags, so validating a required-with-empty-default
        field in __post_init__ would break the CLI path."""
        rollout_queue_address: str = ""
        """The standalone rollout-queue process's base URL (e.g.
        "http://localhost:8767"), usually from
        $ASYNC_INFERENCE_ROLLOUT_QUEUE_ADDR. Required -- same launch-plumbing
        reasoning as relay_addresses; checked by RolloutQueuePushClient's own
        constructor."""
        worker_id: int = 0
        poll_interval_s: float = 2.0
        """Seconds between relay polls before the first checkpoint has been
        loaded (once loaded, the worker free-runs and only polls between
        rounds)."""
        num_rounds: int = 0
        """Stop after this many rollout rounds (0 = run until killed)."""
        round_slowdown_factor: float = 1.0
        """Heterogeneous-hardware emulation: stretch each generation round to
        this factor x its measured duration (>1 = a slower inference GPU).
        This is the benchmark's hetero knob -- slow generators simply
        contribute fewer (and staler) rollouts to the shared pool while the
        trainer never waits on them. 1.0 = no slowdown."""
        dump_folder: str = ""

        def __post_init__(self):
            if self.group_size < 1:
                raise ValueError(f"group_size must be >= 1, got {self.group_size}")
            if self.groups_per_round < 1:
                raise ValueError(
                    f"groups_per_round must be >= 1, got {self.groups_per_round}"
                )
            if self.round_slowdown_factor < 1.0:
                raise ValueError(
                    "round_slowdown_factor must be >= 1.0, got "
                    f"{self.round_slowdown_factor}"
                )

    def __init__(self, config: "AsyncInferenceWorker.Config"):
        self.config = config
        self._relay_client = RelayClient(
            [u.strip() for u in config.relay_addresses.split(",") if u.strip()],
            site_cache_dir=_site_relay_cache(),
        )
        self._rollout_queue_client = RolloutQueuePushClient(
            config.rollout_queue_address
        )
        self._version = 0
        self.generator = None
        self._proc_mesh = None
        self._rollouter = config.rollouter.build()

    async def setup_async(self, *, generator_mesh) -> None:
        cfg = self.config
        self._proc_mesh = generator_mesh
        await setup_mesh_elastic_env(generator_mesh)

        self.generator = generator_mesh.spawn(
            "generator",
            VLLMGeneratorActor,
            cfg.generator,
            model_config=cfg.model,
            model_path=cfg.hf_assets_path,
            compile_config=cfg.compile,
            max_num_seqs=cfg.groups_per_round * cfg.group_size,
            output_dir=cfg.dump_folder,
        )
        # No trainer mesh exists to host TorchStore's storage volumes (there
        # is no trainer here); this worker's own generator mesh hosts them.
        # LocalRankStrategy resolves ITS CALLER's client id from $RANK/
        # $LOCAL_RANK (torchstore/strategy.py); every other TorchStore caller
        # in this package puts/pulls through a Monarch actor endpoint, whose
        # process gets those env vars from setup_mesh_elastic_env
        # above. This driver calls ts.put_state_dict directly (below, in
        # _load_checkpoint -- there's no trainer actor to delegate to), so it
        # needs its OWN client id: a lone, unreplicated coordinator is rank 0
        # of 1.
        os.environ.setdefault("RANK", "0")
        await ts.initialize(mesh=generator_mesh, strategy=ts.LocalRankStrategy())

        tokenizer = cfg.tokenizer.build(tokenizer_path=cfg.hf_assets_path)
        self.renderer = cfg.renderer.build(tokenizer=tokenizer)
        # The rollouter drives its envs in its own CPU worker pool.
        await self._rollouter.setup_async(
            tokenizer_config=cfg.tokenizer,
            renderer_config=cfg.renderer,
            hf_assets_path=cfg.hf_assets_path,
        )
        self._sampling = replace(
            cfg.generator.sampling,
            stop_token_ids=list(self.renderer.get_stop_token_ids()),
        )
        # Start the vLLM engine loop before any pull_model_state_dict: the
        # generator now guards weight pulls on a running engine loop (the
        # controller starts it via generator_router.start_engine_loop;
        # this worker has a single generator actor and must do the same).
        await self.generator.start_engine_loop.call()

    async def _load_checkpoint(self, version: int, state_dict: dict) -> None:
        """Push the relay-fetched state dict into TorchStore under the same
        key `Trainer.push_model_state_dict` uses, then pull it into the
        local engine through the generator's existing, unmodified endpoint.

        Takes ownership of ``state_dict`` and empties it once TorchStore holds
        the weights: the put copies them into shared memory before returning,
        and kept, this dict was another full 18 GB at 9B. TorchStore's own copy stays:
        deleting its keys made every next put allocate fresh shared-memory
        segments while the old ones stayed mapped and pinned in each process's
        segment cache, where a repeated put otherwise reuses them in place."""
        await ts.put_state_dict(state_dict, "model_state_dict")
        state_dict.clear()
        await self.generator.pull_model_state_dict.call(version)
        self._version = version
        logger.info(
            "[worker %d] loaded checkpoint v%d via relay",
            self.config.worker_id,
            version,
        )

    async def _rollout_one_group(self, index: int):
        """Generate ONE rollout group and return ``(group, version)``, where
        version is the checkpoint the group was SPAWNED under -- the send must
        tag it with that, not with whatever the worker holds at send time (a
        checkpoint may swap while the group is in flight). Mirrors the
        controller's _make_generate_fn wiring (same rollouter.run_group_rollouts
        contract), minus the training-batch plumbing this role has no use for --
        the trainer assembles training batches from these groups itself once
        they land in its buffer.

        round_slowdown_factor (heterogeneous-hardware emulation) stretches the
        group's own duration, keeping its concurrency slot occupied the way a
        slower inference GPU would."""

        # The rollouter ships generate_fn to its CPU rollout-worker process, so
        # the closure must capture the generator handle, not `self`: pickling
        # the whole worker (datasets, relay and queue clients) took ~0.8 s of
        # event-loop time per group, which starved the relay download -- a
        # 1.5 GB checkpoint took minutes and generation ran 3-4 versions stale.
        generator = self.generator

        async def generate_fn(
            prompt_token_ids,
            *,
            request_id,
            routing_session_id=None,
            sampling_config=None,
        ):
            result = await generator.generate.call(
                prompt_token_ids,
                request_id=request_id,
                # VLLMGenerator.generate requires this for its intra-mesh DP
                # routing (the rollouter now passes it through GenerateFn).
                routing_session_id=routing_session_id,
                sampling_config=sampling_config,
                metrics_prefix="generator",
            )
            return result.get(0)

        version = self._version
        t0 = time.perf_counter()
        group = await self._rollouter.run_group_rollouts(
            generate_fn=generate_fn,
            sample=self._rollouter.get_training_sample(),
            # An int (the batcher sorts on it), unique per worker; the version
            # travels alongside the group, not in its id.
            group_id=self.config.worker_id * 1_000_000_000 + index,
            group_size=self.config.group_size,
            sampling=self._sampling,
        )
        factor = self.config.round_slowdown_factor
        if factor > 1.0:
            await asyncio.sleep((factor - 1.0) * (time.perf_counter() - t0))
        return group, version

    async def run(self) -> None:
        """Free-run generation as a CONTINUOUS pipeline: keep groups_per_round
        rollout groups in flight and spawn a replacement the moment one
        completes, sending each finished group to the trainer's rollout queue
        immediately. The engine therefore never drains to a round's last
        straggler (the old round barrier left it at 1-2 running sequences for
        most of each round -- measured 0.385 vs ~1.5 rollouts/s/engine against
        the co-located loop).

        A newer checkpoint downloads from the relay in the background and swaps
        in with groups still in flight -- the same in-flight weight update the
        co-located async loop performs on its engines every optimizer step; a
        group keeps the version it was spawned under (see _rollout_one_group).
        The worker never waits for a newer checkpoint before generating (that
        would deadlock the trainer; its max_staleness bound tolerates the
        skew). Before the first checkpoint lands, poll every poll_interval_s.
        Counts a round per groups_per_round groups sent; stops after
        config.num_rounds rounds (0 = run until cancelled).

        The mean reward of the rollouts generated here is the benchmark's
        learning-curve signal: the trainer logs it per window as it consumes
        them (RLControllerMixin.train's ``reward`` field), so a decoupled swarm
        needs no separate greedy validator to measure progress."""
        rounds = 0
        sent = 0
        group_index = itertools.count()
        inflight: set[asyncio.Task] = set()
        fetch = asyncio.ensure_future(
            self._relay_client.fetch_latest(min_version=self._version)
        )
        try:
            while self.config.num_rounds == 0 or rounds < self.config.num_rounds:
                if fetch.done():
                    result = fetch.result()  # fetch_latest never raises: None on fail
                    if result is not None:
                        version, state_dict = result
                        await self._load_checkpoint(version, state_dict)
                    fetch = asyncio.ensure_future(
                        self._relay_client.fetch_latest(min_version=self._version)
                    )
                if self._version == 0:
                    # No weights loaded yet -- nothing to generate from.
                    await asyncio.sleep(self.config.poll_interval_s)
                    continue
                while len(inflight) < self.config.groups_per_round:
                    inflight.add(
                        asyncio.create_task(
                            self._rollout_one_group(next(group_index))
                        )
                    )
                # The timeout keeps checkpoint swaps from being gated on a
                # slow group's completion.
                done, inflight = await asyncio.wait(
                    inflight,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=self.config.poll_interval_s or 1.0,
                )
                for task in done:
                    group, version = task.result()
                    accepted = await self._rollout_queue_client.send(
                        self.config.worker_id, version, [group]
                    )
                    if not accepted:
                        logger.warning(
                            "[worker %d] trainer rejected/unreachable; dropped "
                            "a rollout group",
                            self.config.worker_id,
                        )
                    sent += 1
                    if sent % self.config.groups_per_round == 0:
                        rounds += 1
        finally:
            for task in (fetch, *inflight):
                if not task.done():
                    task.cancel()
            await asyncio.gather(fetch, *inflight, return_exceptions=True)

    async def close(self) -> None:
        await self._rollouter.close()
        if self.generator is not None:
            await self.generator.close.call()
        if self._proc_mesh is not None:
            await self._proc_mesh.stop()


async def _main() -> None:
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
    from torchtitan.config import ConfigManager

    config = ConfigManager().parse_args()
    for field_name, env_name, cast in (
        ("relay_addresses", "ASYNC_INFERENCE_RELAY_ADDRS", str),
        ("rollout_queue_address", "ASYNC_INFERENCE_ROLLOUT_QUEUE_ADDR", str),
        ("worker_id", "ASYNC_INFERENCE_WORKER_ID", int),
        ("round_slowdown_factor", "ASYNC_INFERENCE_ROUND_SLOWDOWN", float),
    ):
        if os.environ.get(env_name):
            setattr(config, field_name, cast(os.environ[env_name]))

    worker = AsyncInferenceWorker(config)
    generator_ws = _compute_world_size(config.generator.parallelism)
    provisioner = PerHostProvisioner(total_gpus=generator_ws)
    generator_mesh = spawn_gpu_procs(
        generator_ws,
        provisioner.allocate(generator_ws),
        bootstrap=_bootstrap_generator,
    )
    try:
        await worker.setup_async(generator_mesh=generator_mesh)
        await worker.run()
    finally:
        await worker.close()


def run_worker() -> None:
    """Entrypoint body for `python -m panoengine.train.rl.worker`."""
    asyncio.run(_main())


if __name__ == "__main__":
    run_worker()
