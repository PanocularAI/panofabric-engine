# Copyright (c) Panocular AI.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Panocular AI.
#
# The four coordination strategies' replica classes, one per strategy, built
# on the shared bases in controller.py (RLTrainer + RLControllerMixin for the
# windowed strategies; PureLearnerReplica for the decoupled-generation ones):
#
#   DiLoCoRLReplica              -- synchronous DiLoCo via torchft
#                                   Manager/Lighthouse quorum.
#   HeLoCoRLReplica              -- asynchronous DiLoCo through the parameter
#                                   server (parameter_server.py).
#   AsyncInferenceReplica        -- one pure learner + remote workers;
#                                   publishes weights to the relay tier.
#   HeLoCoAsyncInferenceReplica  -- N pure learners sharing one rollout queue,
#                                   coordinating through the parameter server.

import asyncio
import logging
import time
from dataclasses import dataclass

import torch

from panoengine.train.rl.actors import (
    DiLoCoManagerTrainer,
    HeLoCoPolicyTrainer,
    SnapshotPolicyTrainer,
)
from panoengine.train.rl.controller import (
    PureLearnerReplica,
    RLControllerMixin,
    RLTrainer,
)
from panoengine.decentralized.parameter_server import (
    HeLoCoRLClient,
    param_metadata,
)
from panoengine.decentralized.relay import (
    build_manifest,
    RelayClient,
    shard_state_dict,
)
from panoengine.decentralized.rollout_queue import (
    RolloutQueuePopClient,
)
from torchtitan.experiments.rl import controller as _rl_controller_mod

logger = logging.getLogger(__name__)


class DiLoCoRLReplica(RLControllerMixin, RLTrainer):
    """A single synchronous-DiLoCo RL replica (worker).

    N of these, coordinated by a torchft Lighthouse, run identical flat RL
    loops and sync weights every ``sync_every`` steps via stock DiLoCo.

    Unlike HeLoCoRLReplica (which drives the parameter-server sync manually at
    the window boundary), this controller runs a flat per-step RL loop: the
    DiLoCo sync happens automatically inside ``optim_step``, via the hook
    registered in DiLoCoManagerTrainer.setup_diloco, synchronized across
    replicas through the torchft Manager/Lighthouse quorum.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RLTrainer.Config):
        # Launch plumbing -- filled in by train.py from environment variables.
        lighthouse_address: str = ""
        """torchft Lighthouse address (usually from $DILOCO_LIGHTHOUSE_ADDR)."""
        replica_id: int = 0
        """This worker's index (usually from $DILOCO_REPLICA_ID)."""
        num_replicas: int = 2
        """Total workers in the quorum (= min_replica_size for the synchronous barrier)."""

        # DiLoCo hyperparameters.
        sync_every: int = 4
        """Local RL steps between DiLoCo syncs (the inner-loop length H)."""
        num_outer_steps: int = 0
        """Fixed number of sync windows to run (step-bound run). Mutually
        exclusive with train_seconds: set exactly one."""
        train_seconds: float = 0.0
        """Wall-clock training budget in seconds (time-bound run): keep running
        sync_every-step windows until the deadline (checked at window
        boundaries, so a run overshoots by at most one window). Slow or
        heterogeneous workers naturally complete fewer steps in the same time.
        Mutually exclusive with num_outer_steps: set exactly one.
        Replicas share the budget so they stop within one window of each
        other; torchft's fault-tolerant quorum absorbs the skew."""
        outer_lr: float = 0.7
        """Outer Nesterov-SGD learning rate (stock DiLoCo default)."""
        outer_momentum: float = 0.9
        """Outer Nesterov-SGD momentum (stock DiLoCo default)."""
        num_generators: int = 1
        """Number of independent vLLM engines this replica spawns (each of
        generator.parallelism's world size, on its own GPU slice out of the
        replica's CUDA_VISIBLE_DEVICES pool). Requests are spread round-robin
        by the GeneratorRouter; weight refreshes fan out to all engines. The
        replica needs trainer_ws + num_generators * generator_ws visible GPUs
        (launch_diloco.sh's GPUS_PER_REPLICA)."""

        def __post_init__(self):
            RLTrainer.Config.__post_init__(self)
            if self.sync_every < 1:
                raise ValueError(f"sync_every must be >= 1, got {self.sync_every}")
            if (self.num_outer_steps > 0) == (self.train_seconds > 0):
                raise ValueError(
                    "Set exactly one of num_outer_steps (step-bound run) or "
                    "train_seconds (time-bound run); got num_outer_steps="
                    f"{self.num_outer_steps}, train_seconds={self.train_seconds}"
                )

    async def setup_async(self, *, trainer_mesh, generator_meshes):
        """Spawn DiLoCoManagerTrainer (not the base PolicyTrainer) and enter the
        DiLoCo context so subsequent ``optim_step`` calls sync automatically."""
        cfg = self.config
        if not cfg.lighthouse_address:
            raise ValueError(
                "lighthouse_address is required (set $DILOCO_LIGHTHOUSE_ADDR)"
            )

        # The inherited Controller.setup_async resolves the trainer actor class
        # from the rl.controller module globals at spawn time; scope-patch that
        # symbol so the base spawns DiLoCoManagerTrainer.
        orig = _rl_controller_mod.PolicyTrainer
        _rl_controller_mod.PolicyTrainer = DiLoCoManagerTrainer
        try:
            await super().setup_async(
                trainer_mesh=trainer_mesh, generator_meshes=generator_meshes
            )
        finally:
            _rl_controller_mod.PolicyTrainer = orig

        await self.trainer.setup_diloco.call(
            lighthouse_address=cfg.lighthouse_address,
            replica_id=cfg.replica_id,
            num_replicas=cfg.num_replicas,
            sync_every=cfg.sync_every,
            outer_lr=cfg.outer_lr,
            outer_momentum=cfg.outer_momentum,
        )
        logger.info(
            "[replica %d] DiLoCo trainer ready (lighthouse=%s)",
            cfg.replica_id,
            cfg.lighthouse_address,
        )

    async def _window_sync(self, t0: float) -> str:
        """The DiLoCo sync for this window already happened inside the
        window's last optim_step (step % sync_every == 0, via the
        inner-optimizer post-step hook); just report the Manager's view as an
        extra detail line (kept off the primary window line so analyze.py's
        WINDOW_RE -- shared verbatim across all controllers -- still matches)."""
        info = self._get_rank_0_value(await self.trainer.diloco_step_info.call())
        return (
            f"diloco_step={info['current_step']} "
            f"participants={info['num_participants']}"
        )

    async def close(self):
        try:
            await self.trainer.close_diloco.call()
        except Exception:
            logger.exception(
                "[replica %d] error closing DiLoCo", self.config.replica_id
            )
        await super().close()


class HeLoCoRLReplica(RLControllerMixin, RLTrainer):
    """A single HeLoCo RL replica (worker).

    Subclasses the base RLTrainer to reuse its actor setup, rollout
    collection, episode/advantage building, batching, forward_backward,
    optim_step, and weight sync; the mixin's train loop runs windows of H
    local RL steps, and _window_sync adds the DiLoCo outer step::

        pull global theta  (client holds it as theta_0, done in setup_async)
        for each window:                        # mixin train loop
            H local RL steps                    # _run_window
            theta_local = trainer.get_full_state_dict_cpu()
            new_global  = client.push(theta_0 - theta_local)  # server outer step
            trainer.load_full_state_dict_cpu(new_global); refresh generators
    """

    @dataclass(kw_only=True, slots=True)
    class Config(RLTrainer.Config):
        server_address: str = ""
        """torchft server /sync URL (usually from $DILOCO_SERVER_ADDR)."""
        heartbeat_address: str = ""
        """torchft server /heartbeat URL (usually from $DILOCO_HB_ADDR)."""
        replica_id: int = 0
        """Identifier for logging (usually from $DILOCO_REPLICA_ID)."""
        sync_every: int = 4
        """Local RL steps between parameter-server syncs (DiLoCo inner-loop length)."""
        num_outer_steps: int = 0
        """Fixed number of sync windows to run (step-bound run). Mutually
        exclusive with train_seconds: set exactly one."""
        train_seconds: float = 0.0
        """Wall-clock training budget in seconds (time-bound run): keep running
        sync_every-step windows until the deadline (checked at window
        boundaries, so a run overshoots by at most one window). Slow or
        heterogeneous workers naturally complete fewer steps in the same time.
        Mutually exclusive with num_outer_steps: set exactly one."""
        should_quantize: bool = False
        """fp16 wire transfer (must match the server)."""
        num_fragments: int = 1
        """Fragment-wise sync with communication overlap (Decoupled DiLoCo,
        arXiv 2604.21428). The model is split into this many contiguous,
        numel-balanced fragments; ONE fragment (rotating) syncs every
        sync_every/num_fragments inner steps, its push/merge roundtrip
        overlapped with the next fragment-window's training and adopted at
        the following boundary. The cycle's LAST fragment is pushed
        synchronously at the window end so each generator refresh sees a
        fully merged model — the refresh/validation/logging cadence stays
        once per sync_every steps, exactly like whole-model sync. Requires
        sync_every % num_fragments == 0 and must match the server's
        --num_fragments. 1 (default) is legacy whole-model sync."""
        num_generators: int = 1
        """Number of independent vLLM engines this replica spawns (each of
        generator.parallelism's world size, on its own GPU slice out of the
        replica's CUDA_VISIBLE_DEVICES pool). Requests are spread round-robin
        by the GeneratorRouter; weight refreshes at the window boundary fan out
        to all engines. The replica needs trainer_ws + num_generators *
        generator_ws visible GPUs (launch_decentralized.sh's
        GPUS_PER_REPLICA)."""

        def __post_init__(self):
            RLTrainer.Config.__post_init__(self)
            if self.sync_every < 1:
                raise ValueError(f"sync_every must be >= 1, got {self.sync_every}")
            if self.num_fragments < 1:
                raise ValueError(
                    f"num_fragments must be >= 1, got {self.num_fragments}"
                )
            if self.sync_every % self.num_fragments != 0:
                raise ValueError(
                    f"sync_every ({self.sync_every}) must be divisible by "
                    f"num_fragments ({self.num_fragments})"
                )
            if (self.num_outer_steps > 0) == (self.train_seconds > 0):
                raise ValueError(
                    "Set exactly one of num_outer_steps (step-bound run) or "
                    "train_seconds (time-bound run); got num_outer_steps="
                    f"{self.num_outer_steps}, train_seconds={self.train_seconds}"
                )

    def __init__(self, config: "HeLoCoRLReplica.Config"):
        super().__init__(config)
        self.client: HeLoCoRLClient | None = None
        # Fragment rotation state (num_fragments > 1): the rotating fragment
        # index, the window-step counter, the in-flight background push
        # (asyncio task + its fragment), and the last boundary's timestamp
        # for per-fragment-window DyLU speed.
        self._frag_idx: int = 0
        self._win_step: int = 0
        self._frag_inflight: "asyncio.Task | None" = None
        self._frag_t0: float = 0.0

    async def setup_async(self, *, trainer_mesh, generator_meshes):
        """Spawn actors (HeLoCoPolicyTrainer) and connect the torchft client.

        Reuses the base setup verbatim but (1) swaps in the HeLoCo trainer
        subclass via a scoped monkeypatch of the symbol setup_async spawns, and
        (2) after setup, builds the torchft client and adopts the server's shared
        theta so all replicas start from the same global weights.
        """
        cfg = self.config
        if not cfg.server_address:
            raise ValueError("server_address is required (set $DILOCO_SERVER_ADDR)")

        # Spawn HeLoCoPolicyTrainer instead of the base PolicyTrainer. The
        # inherited Controller.setup_async resolves the trainer actor class from
        # the rl.controller module globals at spawn time, so scope-patch that
        # symbol for the duration of the base setup.
        orig = _rl_controller_mod.PolicyTrainer
        _rl_controller_mod.PolicyTrainer = HeLoCoPolicyTrainer
        try:
            await super().setup_async(
                trainer_mesh=trainer_mesh, generator_meshes=generator_meshes
            )
        finally:
            _rl_controller_mod.PolicyTrainer = orig

        # Build the torchft client. Parameter names/shapes come from a throwaway
        # meta model built from the SAME model_spec as the server's global model,
        # so both agree on name ordering without a runtime handshake.
        with torch.device("meta"):
            meta_model = cfg.model_spec.model.build()
        names, shapes, dtypes = param_metadata(meta_model)
        del meta_model
        self.client = HeLoCoRLClient(
            cfg.server_address,
            names,
            shapes,
            dtypes,
            heartbeat_address=cfg.heartbeat_address or None,
            should_quantize=cfg.should_quantize,
            num_fragments=cfg.num_fragments,
        )
        self.client.start_heartbeat()

        # Adopt the server's shared theta, then refresh the generator from it.
        global_sd = self.client.pull()
        await self.trainer.load_full_state_dict_cpu.call(global_sd)
        await self.trainer.push_model_state_dict.call()
        await self.generator_router.pull_model_state_dict.call_one(0)
        self._frag_t0 = time.perf_counter()
        logger.info(
            "[replica %d] connected to server; adopted global theta", cfg.replica_id
        )

    # ------------------------------------------------------------------ #
    # Fragment rotation (num_fragments > 1): one fragment exchange per
    # sync_every/P steps, overlapped with the next fragment-window's
    # training (Decoupled DiLoCo). Intermediate boundaries live in
    # _after_inner_step; the cycle's final boundary is _window_sync.
    # ------------------------------------------------------------------ #

    async def _adopt_inflight_fragment(self, clear_optimizer: bool) -> None:
        """Await the in-flight fragment exchange (backpressure: blocks only
        if the roundtrip was slower than one fragment-window of training)
        and load the merged fragment into the trainer. Generators are NOT
        refreshed here — they refresh once per full cycle, at _window_sync."""
        task, self._frag_inflight = self._frag_inflight, None
        if task is None:
            return
        merged = await task  # push errors propagate, like the legacy path
        await self.trainer.load_full_state_dict_cpu.call(
            merged, clear_optimizer=clear_optimizer
        )

    async def _push_fragment(self, background: bool):
        """Extract the rotating fragment's local params (fragment-scoped
        gather — model/P of device->CPU traffic) and push its pseudo-gradient.
        ``background=True`` launches the roundtrip as a task and returns; the
        merge is adopted at the next boundary. ``background=False`` awaits
        and returns the merged fragment."""
        fragment = self._frag_idx
        self._frag_idx = (fragment + 1) % self.config.num_fragments
        names = self.client.frag_names(fragment)
        theta_frag = self._get_rank_0_value(
            await self.trainer.get_full_state_dict_cpu.call(names=names)
        )
        now = time.perf_counter()
        frag_window = self._sync_every // self.config.num_fragments
        speed = frag_window / max(now - self._frag_t0, 1e-6)
        self._frag_t0 = now
        if background:
            self._frag_inflight = asyncio.create_task(
                asyncio.to_thread(
                    self.client.push, theta_frag, speed=speed, fragment=fragment
                )
            )
            return None
        return await asyncio.to_thread(
            self.client.push, theta_frag, speed=speed, fragment=fragment
        )

    async def _after_inner_step(self, iter_t0: float) -> None:
        """Intermediate fragment boundaries: every sync_every/P steps (except
        the window end, which _window_sync owns), adopt the previous
        fragment's merge and launch this fragment's exchange in the
        background. Mid-cycle adopts keep the inner optimizer state — the
        once-per-cycle clear stays at _window_sync, matching whole-model
        cadence."""
        if self.config.num_fragments <= 1 or self.client is None:
            return
        self._win_step += 1
        frag_window = self._sync_every // self.config.num_fragments
        if self._win_step % frag_window == 0 and self._win_step < self._sync_every:
            await self._adopt_inflight_fragment(clear_optimizer=False)
            await self._push_fragment(background=True)

    async def _window_sync(self, t0: float) -> None:
        """The sync boundary closing a full cycle: with fragments, drain the
        pipeline and push the LAST fragment synchronously so the generator
        refresh sees a fully merged model (whole-model cadence for refresh /
        validation / logging); without, the legacy whole-model push. Applies
        any DyLU window-length recommendation to the next window (rounded to
        a multiple of num_fragments so the boundary cadence stays integral)."""
        self._win_step = 0
        num_fragments = self.config.num_fragments
        if num_fragments <= 1:
            theta_local = self._get_rank_0_value(
                await self.trainer.get_full_state_dict_cpu.call()
            )
            speed = self._sync_every / max(time.perf_counter() - t0, 1e-6)
            new_global = self.client.push(theta_local, speed=speed)
            await self.trainer.load_full_state_dict_cpu.call(new_global)
        else:
            await self._adopt_inflight_fragment(clear_optimizer=False)
            merged = await self._push_fragment(background=False)
            await self.trainer.load_full_state_dict_cpu.call(
                merged, clear_optimizer=True
            )
        await self.trainer.push_model_state_dict.call()
        await self.generator_router.pull_model_state_dict.call_one(
            self._policy_version
        )
        if self.client.last_dylu_steps > 0:
            steps = self.client.last_dylu_steps
            self._sync_every = max(
                num_fragments, steps // num_fragments * num_fragments
            )
        return None

    async def close(self):
        if self._frag_inflight is not None:
            # Drain (never adopt) a leftover exchange, e.g. after divergence.
            try:
                await self._frag_inflight
            except Exception:
                pass
            self._frag_inflight = None
        if self.client is not None:
            self.client.stop_heartbeat()
        await super().close()


class AsyncInferenceReplica(PureLearnerReplica):
    """One pure-learner trainer + a pool of remote generator workers.

    A PURE-LEARNER trainer (one GPU, no local generation of any kind) whose
    training rollouts come entirely from a pool of REMOTE generator workers on
    separate machines, pushed into a standalone rollout-queue process
    (rollout_queue) this trainer pops from. The trainer consumes with a
    max_staleness bound, trains, and publishes its weights to a relay tier
    (SHARDCAST-style) so the workers can pull the next policy version.
    Generation on the same node as the trainer would just be the single-node
    "local" baseline with an async buffer, so there is deliberately no vLLM
    here -- inference is fully delocalized.

    Inherits the pure-learner consumer/buffer/staleness machinery and the
    generator-less setup from PureLearnerReplica. Adds:
      - a pop client on the standalone rollout-queue process the remote
        workers push into (drained by the inherited
        ``_consume_remote_rollouts`` into the staleness-bounded buffer);
      - relay publishing of THIS trainer's own weights (its weights are the
        policy -- there is a single trainer, no parameter server): an initial
        publish at setup so the workers can bootstrap, then a publish every
        ``publish_every`` windows.

    Staleness is bounded against ``self._checkpoint_version`` (the latest
    version this trainer published): workers stamp each rollout batch with the
    checkpoint version they loaded off the relay, so the two live in the same
    version space.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(PureLearnerReplica.Config):
        relay_addresses: str = ""
        """Comma-separated relay server base URLs (e.g.
        "http://localhost:8765,http://localhost:8766"). Required -- every
        published checkpoint version is uploaded to all of them. Launch
        plumbing (usually from $ASYNC_INFERENCE_RELAY_ADDRS via train.py), so
        it's checked in setup_async rather than __post_init__ (ConfigManager
        calls the --config function with zero args before overlaying CLI
        flags, so a required-with-empty-default field can't be validated at
        __post_init__ time without breaking the CLI path)."""
        num_shards: int = 4
        """Shards per published checkpoint (SHARDCAST-style checkpoint-
        transfer sharding; unrelated to model/optimizer sharding)."""
        publish_every: int = 1
        """Windows between relay publishes (1 = publish every window)."""
        rollout_queue_address: str = ""
        """Base URL of the standalone rollout-queue process (rollout_queue)
        this trainer pops from, e.g. "http://localhost:8767". Required --
        launch plumbing (usually from $ROLLOUT_QUEUE_ADDR via train.py), so
        it's checked in setup_async rather than __post_init__ (same reasoning
        as relay_addresses)."""

        def __post_init__(self):
            PureLearnerReplica.Config.__post_init__(self)
            if self.num_shards < 1:
                raise ValueError(f"num_shards must be >= 1, got {self.num_shards}")
            if self.publish_every < 1:
                raise ValueError(
                    f"publish_every must be >= 1, got {self.publish_every}"
                )

    def __init__(self, config: "AsyncInferenceReplica.Config"):
        super().__init__(config)
        self._relay_client: RelayClient | None = None
        self._checkpoint_version = 0
        self._window_count = 0
        self._publish_task: asyncio.Task | None = None
        self._queue_client: RolloutQueuePopClient | None = None

    async def setup_async(self, *, trainer_mesh, generator_meshes):
        """Spawn only the SnapshotPolicyTrainer actor (one learner GPU), build
        the relay client and the rollout-queue pop client, and publish the
        initial checkpoint so the worker pool can bootstrap before the first
        window."""
        cfg = self.config
        if not cfg.relay_addresses:
            raise ValueError(
                "relay_addresses is required (set $ASYNC_INFERENCE_RELAY_ADDRS)"
            )
        if not cfg.rollout_queue_address:
            raise ValueError(
                "rollout_queue_address is required (set $ROLLOUT_QUEUE_ADDR)"
            )
        self._relay_client = RelayClient(
            [url.strip() for url in cfg.relay_addresses.split(",") if url.strip()]
        )
        self._queue_client = RolloutQueuePopClient(cfg.rollout_queue_address)

        await self._spawn_trainer_only(
            trainer_mesh=trainer_mesh,
            generator_meshes=generator_meshes,
            policy_trainer_cls=SnapshotPolicyTrainer,
        )

        # Publish the initial (HF-initialized) weights so the remote workers
        # have a checkpoint to load and start generating from before the first
        # window -- otherwise the trainer would starve at window 0 (nothing to
        # consume) waiting for rollouts the workers can't produce yet.
        note = await self._publish_checkpoint()
        logger.info("[replica %d] %s (initial)", cfg.replica_id, note)

    # ------------------------------------------------------------------ #
    # PureLearnerReplica hooks: rollout source + staleness reference.
    # ------------------------------------------------------------------ #

    async def _next_rollout_batch(self):
        """Pop one batch from the standalone queue (None when empty; the base
        consumer then waits queue_poll_interval_s and retries)."""
        return await self._queue_client.pop()

    def _staleness_reference(self) -> int:
        return self._checkpoint_version

    # ------------------------------------------------------------------ #
    # Relay publish.
    # ------------------------------------------------------------------ #

    async def _publish_checkpoint(self) -> str:
        theta = self._get_rank_0_value(
            await self.trainer.get_full_state_dict_cpu.call()
        )
        # bf16 over the wire: halves every publish and every worker download.
        # Safe because relay checkpoints are a dead end -- workers use them
        # only as the generation behavior policy (the engine runs bf16 anyway)
        # and never train on or push back these weights, so unlike the
        # trainer<->server DiLoCo channel the cast can't compound. The IS-
        # corrected loss stays consistent: the recorded behavior logprobs come
        # from the bf16 model that actually generated.
        theta = {k: v.to(dtype=torch.bfloat16) for k, v in theta.items()}
        self._checkpoint_version += 1
        # Shard in a thread: torch.save of ~GB blobs would otherwise block
        # the event loop (and with it the embedded rollout queue).
        shards = await asyncio.to_thread(
            shard_state_dict, theta, self.config.num_shards
        )
        manifest = build_manifest(self._checkpoint_version, shards)
        await self._relay_client.publish(self._checkpoint_version, shards, manifest)
        total_bytes = sum(manifest.shard_sizes)
        return (
            f"relay: published v{self._checkpoint_version} "
            f"({manifest.num_shards} shards, {total_bytes}B)"
        )

    # ------------------------------------------------------------------ #
    # Coordination boundary.
    # ------------------------------------------------------------------ #

    async def _window_sync(self, t0: float) -> str:
        """Window boundary: publish the new weights to the relay tier on
        publish_every boundaries so the worker pool advances to the next
        policy version. The publish runs as a BACKGROUND task -- snapshotting,
        sharding, and POSTing ~GBs must not stall the next window (the actor
        mailbox serializes the snapshot against optim steps, so the background
        task still reads a consistent theta). If the previous publish is
        somehow still in flight, this boundary's publish is skipped -- workers
        just keep the last version a little longer, which max_staleness
        already tolerates. No generator refresh and no producer pausing --
        there is no local generation."""
        del t0
        stats = f"buffer: depth={self._buffer.qsize()} dropped={self._num_dropped}"
        self._num_dropped = 0
        self._window_count += 1
        if self._window_count % self.config.publish_every == 0:
            prev = self._publish_task
            if prev is not None and not prev.done():
                stats = f"{stats} | relay: publish skipped (previous in flight)"
            else:
                if prev is not None and (exc := prev.exception()) is not None:
                    logger.warning(
                        "[replica %d] previous relay publish failed: %s",
                        self.config.replica_id,
                        exc,
                    )
                self._publish_task = asyncio.create_task(self._publish_checkpoint())
                stats = f"{stats} | relay: publish started (background)"
        return stats

    async def close(self):
        task = self._publish_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await super().close()


class HeLoCoAsyncInferenceReplica(PureLearnerReplica):
    """A pure-learner HeLoCo trainer whose training rollouts come entirely
    from the shared rollout-queue process (fed by a remote generator pool).

    Decoupled generation scaled to MULTIPLE trainers: N
    pure-learner HeLoCo trainer replicas (1 GPU each, no local generation)
    draw ALL training rollouts from one shared rollout-queue process
    (rollout_queue) fed by a pool of remote generator workers, and coordinate
    no-barrier through the HeLoCo parameter server (parameter_server.py run with
    --relay_addr), which publishes the CURRENT global theta to a relay
    process (relay) for the generator pool.

    Inherits the pure-learner consumer/buffer/staleness machinery and the
    generator-less setup from PureLearnerReplica; adds the HeLoCo outer step:
      - ``setup_async``: spawn only the HeLoCoPolicyTrainer actor, build the
        torchft client, and adopt the server's shared theta.
      - ``_window_sync``: push theta_local (the server applies its outer step
        to the pseudo-gradient), adopt the returned global theta -- no
        generator refresh.

    Staleness is bounded against the shared parameter-server revision, not the
    replica's own ``policy_version``: generator workers stamp each rollout
    batch with the checkpoint version they loaded (the hub publishes checkpoint
    version = server revision + 1), and this replica gates on
    ``self._last_known_revision`` (refreshed every ``_window_sync`` from
    ``self.client.revision + 1``). Server revision is shared and advances once
    per applied push; ``policy_version`` advances per local optim step,
    independently per replica -- so comparing a worker's revision tag against a
    local ``policy_version`` would be meaningless.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(PureLearnerReplica.Config):
        server_address: str = ""
        """torchft server /sync URL (usually from $DILOCO_SERVER_ADDR)."""
        heartbeat_address: str = ""
        """torchft server /heartbeat URL (usually from $DILOCO_HB_ADDR)."""
        should_quantize: bool = False
        """fp16/int8 pseudo-gradient wire transfer (must match the server)."""
        rollout_queue_address: str = ""
        """The shared rollout-queue process's base URL (e.g.
        "http://localhost:8767"), usually from $ROLLOUT_QUEUE_ADDR. Required
        -- launch plumbing, so it's checked in setup_async rather than
        __post_init__ (ConfigManager calls the --config function with zero
        args before overlaying CLI flags, so a required-with-empty-default
        field can't be validated at __post_init__ time without breaking the
        CLI path)."""

    def __init__(self, config: "HeLoCoAsyncInferenceReplica.Config"):
        super().__init__(config)
        self.client: HeLoCoRLClient | None = None
        self._queue_client: RolloutQueuePopClient | None = None
        self._last_known_revision = 0

    async def setup_async(self, *, trainer_mesh, generator_meshes):
        """Spawn only the HeLoCoPolicyTrainer actor (one learner GPU), connect
        the torchft client, and adopt the server's shared theta."""
        cfg = self.config
        if not cfg.server_address:
            raise ValueError("server_address is required (set $DILOCO_SERVER_ADDR)")
        if not cfg.rollout_queue_address:
            raise ValueError(
                "rollout_queue_address is required (set $ROLLOUT_QUEUE_ADDR)"
            )

        await self._spawn_trainer_only(
            trainer_mesh=trainer_mesh,
            generator_meshes=generator_meshes,
            policy_trainer_cls=HeLoCoPolicyTrainer,
        )

        # Build the torchft client from a throwaway meta model built from the
        # SAME model_spec as the server's global model, so both agree on
        # parameter name ordering without a runtime handshake.
        with torch.device("meta"):
            meta_model = cfg.model_spec.model.build()
        names, shapes, dtypes = param_metadata(meta_model)
        del meta_model
        self.client = HeLoCoRLClient(
            cfg.server_address,
            names,
            shapes,
            dtypes,
            heartbeat_address=cfg.heartbeat_address or None,
            should_quantize=cfg.should_quantize,
        )
        self.client.start_heartbeat()

        # Adopt the server's shared theta into the trainer model. No generator
        # to refresh and no TorchStore push -- the hub publishes the global
        # model to the relay for the worker pool.
        global_sd = self.client.pull()
        await self.trainer.load_full_state_dict_cpu.call(global_sd)
        self._queue_client = RolloutQueuePopClient(cfg.rollout_queue_address)
        # +1: the hub publishes checkpoint versions as server revision + 1
        # (see parameter_server.py::_watch_and_publish for why), so this reference must
        # be shifted the same way to stay comparable to the revision workers
        # stamp on their rollout batches.
        self._last_known_revision = self.client.revision + 1
        logger.info(
            "[replica %d] connected to server; adopted global theta "
            "(pure learner, no local generation)",
            cfg.replica_id,
        )

    # ------------------------------------------------------------------ #
    # PureLearnerReplica hooks: rollout source + staleness reference.
    # ------------------------------------------------------------------ #

    async def _next_rollout_batch(self):
        """Pop one batch from the hub's shared queue (None when empty; the
        base consumer then waits queue_poll_interval_s and retries)."""
        return await self._queue_client.pop()

    def _staleness_reference(self) -> int:
        return self._last_known_revision

    # ------------------------------------------------------------------ #
    # HeLoCo outer step.
    # ------------------------------------------------------------------ #

    async def _window_sync(self, t0: float) -> str:
        """The HeLoCo outer step, with NO generator refresh: push theta_local
        (the server applies its outer step to the pseudo-gradient), adopt the
        returned global theta, apply any DyLU recommendation, and refresh this
        replica's revision freshness reference for the next window's staleness
        gate.

        ``client.push`` is torchft's SYNCHRONOUS client -- a multi-GB
        quantize/upload/merge/download roundtrip. Run it in a thread so the
        event loop stays live: the rollout consumer keeps draining the shared
        queue into the buffer throughout the sync, and the next window starts
        data-rich instead of stalled. (The trainer GPU still waits for
        adoption -- the pseudo-gradient/baseline contract -- only the event
        loop is freed.)"""
        theta_local = self._get_rank_0_value(
            await self.trainer.get_full_state_dict_cpu.call()
        )
        speed = self._sync_every / max(time.perf_counter() - t0, 1e-6)
        new_global = await asyncio.to_thread(self.client.push, theta_local, speed=speed)
        await self.trainer.load_full_state_dict_cpu.call(new_global)
        if self.client.last_dylu_steps > 0:
            self._sync_every = self.client.last_dylu_steps
        self._last_known_revision = self.client.revision + 1
        stats = f"buffer: depth={self._buffer.qsize()} dropped={self._num_dropped}"
        self._num_dropped = 0
        return stats

    async def close(self):
        if self.client is not None:
            self.client.stop_heartbeat()
        await super().close()
