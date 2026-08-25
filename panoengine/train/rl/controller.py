# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Panocular AI.
#
# The shared controller-side machinery of decentralized_rl, top-down (mirrors
# rl/controller.py: controllers orchestrate; the GPU actors they spawn live in
# actors.py). The four concrete strategy replicas built on these bases live in
# replicas.py.
#
#   RLTrainer            -- synchronous orchestration base. Subclasses the
#                           async rl/ Controller to reuse its actor setup,
#                           validation, and teardown, but replaces the 4-loop
#                           run() with synchronous single-step primitives
#                           (_collect_training_batch / _apply_training_batch)
#                           for the windowed outer loops below. Trainer-actor
#                           injection seam: Controller.setup_async resolves
#                           the name ``PolicyTrainer`` from rl.controller's
#                           module globals at spawn time, so replicas rebind
#                           it (scoped monkeypatch) to spawn their subclass.
#   RLControllerMixin    -- the shared windowed train loop: LlamaRL-style
#                           one-step-ahead generation/training overlap
#                           (arXiv:2505.24034; safe under the IS-corrected
#                           GRPO loss), divergence detection, fixed-set
#                           validation, and the per-window log line
#                           ___benchmark/analyze.py parses. Coordinators plug
#                           into its hooks (_window_sync and friends).
#   PureLearnerReplica   -- base for the decoupled-generation strategies:
#                           no local vLLM; all rollouts arrive from a remote
#                           generator pool via a queue, consumed under a staleness
#                           bound.

import asyncio
import itertools
import json
import logging
import math
import time
from dataclasses import dataclass

from panoengine.train.rl.train import setup_mesh_elastic_env
from torchtitan.experiments.rl.components.weight_sync import WeightSyncManager
from torchtitan.experiments.rl.controller import Controller
from torchtitan.experiments.rl.controller_metrics import compute_rollout_metrics

# Re-exported so config_registry keeps a single import site for the loss it
# used to get from ``rl.trainer``; it now lives in the shared ``rl/losses``.
from torchtitan.experiments.rl.losses import GRPOLoss  # noqa: F401  re-export: presets
# and stored specs named this path back when the losses lived in the fork's RL tree.
from torchtitan.experiments.rl.rollout import RolloutGroup

logger = logging.getLogger(__name__)


class RLTrainer(Controller):
    """Synchronous RL orchestration base for the decentralized_rl coordinators.

    Reuses ``Controller.__init__`` (metrics/renderer/sampling/rollouter/recorder),
    ``Controller.setup_async`` (actor spawn + initial weight sync),
    ``Controller.validate``, and ``Controller.close``. Adds synchronous
    single-step primitives the windowed ``RLControllerMixin`` loop drives; the
    async ``Controller.run`` is intentionally unused by decentralized_rl.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Controller.Config):
        """decentralized_rl replicas subclass this; it adds no fields of its own."""

    def _build_sync_pipeline(self) -> None:
        """Build the rollout→sample→batch pipeline components used by the
        synchronous loop. Mirrors what ``Controller.run`` builds inline for the
        async loops, but stores them on ``self`` so the windowed loop can call
        them one step at a time. Must run after ``setup_async`` (needs
        ``self.trainer_dp_degree``)."""
        async_loop = self.config.async_loop
        self._training_sample_builder = async_loop.training_sample_builder.build()
        self._batcher = async_loop.batcher.build(
            num_prompts_per_train_step=async_loop.num_prompts_per_train_step,
            dp_degree=self.trainer_dp_degree,
            pad_id=self.renderer._tokenizer.eos_token_id,
        )
        self._generate_fn = self._make_generate_fn(metrics_prefix="generator")
        self._group_counter = itertools.count()
        # Local-generation replicas source groups from the upstream async
        # producer pipeline (RolloutGroupWorkBuffer + one rollout worker per
        # slot + data input) instead of the serial one-group-at-a-time path:
        # serial collection left the vLLM engines at group_size sequences in
        # flight (~2% of the sized rollout_concurrency budget) and paid
        # sum(group latencies) per step where the pipeline pays max().
        # Pure learners (num_generators == 0) have no local generation to
        # pipeline; they keep consuming their remote queue.
        self._producers_started = False

    def _start_producers_if_local_generation(self) -> None:
        """Start the rollout producer pipeline (after pre-training validation,
        so validation runs on idle engines, matching Controller.run's order)."""
        # getattr: pure-learner fakes in the tests carry a bare namespace config.
        if getattr(self.config, "num_generators", 0) > 0:
            self._start_rollout_producers()
            # Overlaps each step's weight handoff (push -> pull -> slot
            # release) with the next step's fwd/bwd, as upstream's
            # _trainer_loop does.
            self._weight_sync = WeightSyncManager(
                trainer=self.trainer,
                generator_router=self.generator_router,
                group_buffer=self._group_buffer,
                num_prompts_per_train_step=(
                    self.config.async_loop.num_prompts_per_train_step
                ),
            )
            self._producers_started = True

    async def _stop_producers_if_started(self) -> None:
        # getattr: pure learners override _build_sync_pipeline and never set
        # the flag, and there is nothing to stop for them.
        if getattr(self, "_producers_started", False):
            self._producers_started = False
            # Land the last step's overlapped handoff first: its release
            # touches the buffer this is about to close.
            await self._weight_sync.wait_inflight_push_pull()
            await self._stop_rollout_producers()

    async def _rollout_one_group(self, *, sampling, step: int) -> RolloutGroup:
        """Generate + score one rollout group synchronously (the synchronous
        analogue of ``Controller._rollout_loop``'s body). Records the raw group
        for inspection before any downstream drop."""
        async_loop = self.config.async_loop
        group = await self._rollouter.run_group_rollouts(
            generate_fn=self._generate_fn,
            sample=self._rollouter.get_training_sample(),
            group_id=next(self._group_counter),
            group_size=async_loop.num_samples_per_prompt,
            sampling=sampling,
            renderer=self.renderer,
        )
        group.metrics = compute_rollout_metrics(
            prefix="rollout", rollouts=group.rollouts
        )
        self.rollout_recorder.record(is_validation=False, rollout_groups=[group])
        return group

    async def _collect_training_batch(self, step: int):
        """Collect rollout groups and pack exactly one ``TrainingBatch``.

        Feeds each group through ``TrainingSampleBuilder`` and the packing
        ``Batcher`` (main's pipeline) until the batcher yields a batch — that
        happens once ``num_prompts_per_train_step`` trainable groups have
        accumulated. Returns ``(packed_batch, rollout_groups)``; the groups are
        kept for the reward metric in the per-window log line.

        With the producer pipeline running, groups come out of the shared
        ``RolloutGroupWorkBuffer`` (windowed FIFO, generated concurrently by
        the rollout workers); the serial ``_rollout_one_group`` path remains
        only as the no-pipeline fallback. Untrainable groups release their
        buffer slot immediately (upstream ``_batcher_loop`` contract); trained
        slots are released after the post-step generator refresh, preserving
        the born-fresh invariant.
        """
        rollout_groups: list[RolloutGroup] = []
        packed = None
        while packed is None:
            if self._producers_started:
                group = await self._group_buffer.take_finalized()
                if group is None:
                    raise RuntimeError(
                        "rollout group buffer closed while a training batch "
                        "was still being collected"
                    )
            else:
                group = await self._rollout_one_group(
                    sampling=self._sampling, step=step
                )
            rollout_groups.append(group)
            training_sample_group = self._training_sample_builder.build_from_group(
                rollout_group=group
            )
            if self._producers_started and not training_sample_group.training_samples:
                await self._group_buffer.release_active_groups(
                    1, reason="untrainable_group"
                )
            packed = self._batcher.add_training_samples(
                training_sample_group=training_sample_group
            )
        return packed, rollout_groups

    async def _apply_training_batch(self, packed):
        """Run fwd/bwd on every microbatch, then the optimizer step. Returns the
        rank-0 optim result (``.policy_version``, ``.metrics``) plus the last
        microbatch's mean loss for the divergence guard / log line."""
        last_loss = 0.0
        for microbatch in packed.microbatches:
            mb = self._get_rank_0_value(
                await self.trainer.forward_backward.call(
                    microbatch, packed.num_global_valid_tokens
                )
            )
            last_loss = mb.get("loss/mean", mb.get("loss", last_loss))
        if getattr(self, "_producers_started", False):
            # The previous step's overlapped weight push must land before this
            # optimizer step mutates the weights (upstream _trainer_loop order).
            await self._weight_sync.wait_prev_push()
        optim_output = self._get_rank_0_value(await self.trainer.optim_step.call())
        return optim_output, last_loss


class RLControllerMixin:
    """The shared controller loop for RLTrainer-based coordinators.

    Requires the host class to be (or subclass) torchtitan's RLTrainer, with a
    `Config` exposing `sync_every: int`, `num_outer_steps: int`,
    `train_seconds: float`, and `replica_id: int`.
    Provides `__init__` (initializes `_policy_version`) and `train`; subclasses
    override the `_window_sync` / `_train_*` hooks for their coordination work.
    """

    def __init__(self, config):
        super().__init__(config)
        self._policy_version = 0

    # ------------------------------------------------------------------ #
    # Generation/training split (the pipelining building blocks).
    # ------------------------------------------------------------------ #

    async def _collect_and_build(self, step: int):
        """Rollout collection + training-batch packing (generator-mesh + CPU
        work only; no trainer-mesh fwd/bwd). Launched one step ahead of the
        previous step's trainer-mesh work by _run_window's pipeline.

        Uses the upstream rollout→sample→batch pipeline (RolloutGroup ->
        TrainingSampleBuilder -> Batcher.add_training_samples) provided by the
        RLTrainer base as ``_collect_training_batch``. Returns
        ``(packed_batch, rollout_groups)``; the groups are kept for the reward
        metric in the per-window log line."""
        await self.trainer.sync_log_step.call(step)
        # InterGeneratorRouter is a monarch Actor: reach it through its
        # @concurrent_endpoint (a single-actor mesh, hence call_one), never by
        # calling the implementation directly. Upstream privatized those to
        # _fanout/_pull_model_state_dict when it actorized the router, and an
        # ActorEndpoint has no __call__ -- a direct call raises
        # "ActorEndpoint object is not callable" at the first co-located step.
        # The endpoint also sets the router process's own step counter, which
        # _fanout alone skipped.
        await self.generator_router.sync_log_step.call_one(step)
        return await self._collect_training_batch(step)

    async def _train_on(self, packed, rollout_groups) -> dict:
        """forward_backward loop + optim_step + generator weight refresh
        (trainer-mesh work)."""
        pre_optim_policy_version = self._policy_version

        optim_output, last_loss = await self._apply_training_batch(packed)
        self._policy_version = optim_output.policy_version
        if getattr(self, "_producers_started", False):
            # Overlap this step's push -> pull -> buffer-slot release with the
            # next step's fwd/bwd (upstream _trainer_loop's WeightSyncManager
            # pattern; the manager's release-after-pull keeps born-fresh).
            await self._weight_sync.wait_prev_pull()
            self._weight_sync.start_async_push_pull(version=self._policy_version)
        else:
            await self._refresh_generators()

        rewards = [
            r.reward for g in rollout_groups for r in g.rollouts if r.reward is not None
        ]
        reward_mean = sum(rewards) / len(rewards) if rewards else float("nan")
        staleness = self._batch_staleness(
            pre_optim_policy_version, packed.min_policy_versions
        )
        return {
            "loss": last_loss,
            "reward_mean": reward_mean,
            "policy_version": self._policy_version,
            "num_rollouts": len(rewards),
            "staleness": staleness,
        }

    def _batch_staleness(
        self, pre_optim_policy_version: int, min_policy_versions
    ) -> int:
        """The staleness (in versions) of the just-consumed batch, for the
        per-window log line. The packed batch carries ``min_policy_versions``
        (the versions its samples' GENERATORS held), so the two operands are in
        the same version space: generators pull per optim step tagged with the
        trainer's ``policy_version``, so the gap to the pre-optim policy_version
        is the steps of lag the IS-corrected loss absorbed. Always >= 0; grows
        if generation lags training (expected under the always-on one-step
        pipeline). Pure learners override this: their samples carry relay/hub
        checkpoint versions, which a local optim-step counter is not comparable
        to."""
        return (
            pre_optim_policy_version - min(min_policy_versions)
            if min_policy_versions
            else 0
        )

    async def _refresh_generators(self) -> None:
        """Bring this replica's generators up to the just-updated local policy:
        stage the trainer weights, then drain-and-pull every engine. Overridden
        by controllers that decouple the pull from the train step
        (AsyncInferenceReplica stages only; its producers pull per-round)."""
        await self.trainer.push_model_state_dict.call()
        await self.generator_router.pull_model_state_dict.call_one(
            self._policy_version
        )

    # ------------------------------------------------------------------ #
    # Window runner.
    # ------------------------------------------------------------------ #

    async def _run_window(self, sync_every: int, start_step: int):
        """Run one window of ``sync_every`` inner steps, always
        pipelined: overlap ``_collect_and_build(h+1)`` with ``_train_on(h)``
        (LlamaRL-style one-step-ahead generation/training overlap). Collection
        is never launched on the window's last inner step, so the pipeline
        can't straddle the caller's outer sync boundary.

        Returns ``(window_rewards, last, global_step, diverged)``. The caller
        (train) applies the coordinator's post-window sync via _window_sync
        and stops on ``diverged``.
        """
        window_rewards: list[float] = []
        last = None
        global_step = start_step
        # Trainer idle: wall time the trainer mesh spent blocked on the
        # collection task instead of doing fwd/bwd. High fraction => generation
        # is the bottleneck (add generators); ~0 => trainer-bound.
        self._window_wait_s = 0.0

        global_step += 1
        pending = asyncio.create_task(self._collect_and_build(global_step))
        try:
            for _h in range(sync_every):
                iter_t0 = time.perf_counter()
                packed, rollout_groups = await pending
                self._window_wait_s += time.perf_counter() - iter_t0
                pending = None
                if _h != sync_every - 1:
                    global_step += 1
                    pending = asyncio.create_task(self._collect_and_build(global_step))
                last = await self._train_on(packed, rollout_groups)
                window_rewards.append(last["reward_mean"])
                # "step: N" is the progress contract an external supervisor greps
                # for (same line pretrain emits) -- the step trained in iteration
                # _h is start_step + _h + 1 (the pipelined `global_step` already
                # points at the NEXT step's collection).
                logger.info(
                    "[replica %d] step: %d",
                    self.config.replica_id,
                    start_step + _h + 1,
                )
                if not math.isfinite(float(last["loss"])):
                    return window_rewards, last, global_step, True
                await self._after_inner_step(iter_t0)
        finally:
            # Divergence return or a raising _train_on must not leak the
            # in-flight collection task.
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

        return window_rewards, last, global_step, False

    # ------------------------------------------------------------------ #
    # Validation on a fixed eval set.
    # ------------------------------------------------------------------ #

    def _reset_validation_set(self) -> None:
        """Rebuild the validation dataset so every pass scores the same fixed
        prompts. Without this, the iterator draws fresh random samples each
        call, making the validation curve reflect prompt difficulty rather
        than learning progress."""
        self._rollouter._validation_dataset = (
            self.config.rollouter.validation_dataset.build()
        )

    async def _validate_fixed(self, step: int):
        """validate() on the fixed eval set. Greedy reward on this set is the
        metric that should trend up; train rollout reward is noisy/exploratory."""
        self._reset_validation_set()
        return await self.validate(step=step)

    def _aggregate_validation(self, metrics) -> dict:
        from torchtitan.experiments.rl.observability import metrics as m

        return m.MetricsProcessor._aggregate_metrics(metrics)

    # ------------------------------------------------------------------ #
    # Coordination hooks (all no-ops for a standalone worker).
    # ------------------------------------------------------------------ #

    async def _train_setup(self) -> None:
        """Runs once, after pre-training validation and before the first
        window (e.g. start rollout producers)."""

    async def _after_inner_step(self, iter_t0: float) -> None:
        """Runs after each inner step of a window (not after a divergence
        stop). ``iter_t0`` is the step's start time (time.perf_counter) —
        subclass-side per-step instrumentation or pacing (e.g. the
        benchmark's slow-worker emulation) hooks in here."""

    def _window_yield_snapshot(self) -> dict | None:
        """This window's rollout-accounting counters ({groups_consumed,
        dropped_stale, dropped_zero_std}), or None on paths that don't track
        them (the co-located collector drops zero-std groups inside the
        upstream builder, invisibly from here). Read-and-reset."""
        return None

    #: Consecutive failed outer syncs before train() gives up. A replica that
    #: keeps training locally but never merges looks healthy and produces
    #: nothing, so an unreachable hub must eventually be fatal -- just not on
    #: the first socket stall.
    _MAX_SYNC_FAILURES = 3

    async def _window_sync(self, t0: float) -> str | None:
        """The coordinator's window-boundary work, between the window's last
        optim_step and its validation pass (e.g. HeLoCo's push/pull). ``t0``
        is the window's start time (time.perf_counter). May adjust
        ``self._sync_every`` for the next window (DyLU). Returns an extra
        detail line to log after the standard window line, or None."""
        return None

    async def _after_validation(self) -> None:
        """Runs after each window's validation pass (e.g. resume producers)."""

    async def _train_teardown(self) -> None:
        """Runs once, after the last window and before post-training
        validation (e.g. quiesce producers so post-val measures the final
        policy)."""

    async def _train_cleanup(self) -> None:
        """Always runs on the way out of train(), including on divergence or
        error (e.g. cancel producer tasks)."""

    # ------------------------------------------------------------------ #
    # The outer training loop.
    # ------------------------------------------------------------------ #

    async def train(self):
        """Run windows until the step or time bound, validating and emitting
        the standard per-window log line (the format ___benchmark/analyze.py's
        WINDOW_RE parses -- change them together) after each."""
        cfg = self.config
        rid = cfg.replica_id
        self._sync_every = cfg.sync_every

        # Build the synchronous rollout->sample->batch pipeline now that
        # setup_async has run (it needs self.trainer_dp_degree).
        self._build_sync_pipeline()

        logger.info("[replica %d] pre-training validation", rid)
        pre_acc = self._aggregate_validation(await self._validate_fixed(0))

        try:
            await self._train_setup()
            # After pre-training validation (idle engines for it) and after
            # _train_setup (the coordinator connection precedes any rollout
            # work, same order as the serial loop).
            # getattr: the mixin's documented host is RLTrainer, but the unit
            # tests drive train() on bare fakes; producer hooks are optional there.
            start_producers = getattr(
                self, "_start_producers_if_local_generation", None
            )
            if start_producers is not None:
                start_producers()

            global_step = 0
            outer = 0
            sync_failures = 0
            deadline = (
                time.monotonic() + cfg.train_seconds
                if cfg.train_seconds > 0
                else float("inf")
            )
            while time.monotonic() < deadline and (
                cfg.num_outer_steps == 0 or outer < cfg.num_outer_steps
            ):
                t0 = time.perf_counter()
                window_rewards, last, global_step, diverged = await self._run_window(
                    self._sync_every, global_step
                )
                if diverged:
                    logger.error("[replica %d] loss diverged; stopping", rid)
                    return

                # A failed outer sync must not kill the run. torchft's contract
                # is "drop the push, keep training on the current params,
                # retry at the next window boundary" (AsyncDiLoCo class
                # docstring), but that catch-all lives in its
                # _step_post_hook -- the decentralized_rl replicas call
                # client.push directly and inherited none of it. One 60 s
                # socket stall on the hub therefore ended run 947642665102
                # four steps in, with the PS holding perfectly good weights.
                try:
                    # Pause admission during the merge so no rollout is BORN
                    # under weights the sync is about to replace ("keep the
                    # deepest stale sample inside a window" — the preset's
                    # target_offpolicy_steps < sync_every contract). In-flight
                    # groups keep generating and are IS-priced by version.
                    if getattr(self, "_producers_started", False):
                        await self._group_buffer.pause()
                        # The merge reads/replaces trainer weights: the last
                        # step's overlapped push/pull must settle first.
                        await self._weight_sync.wait_inflight_push_pull()
                    extra = await self._window_sync(t0)
                    sync_failures = 0
                except Exception:
                    sync_failures += 1
                    logger.warning(
                        "[replica %d] window %d: outer sync failed (%d in a "
                        "row); dropping this window's pseudo-gradient and "
                        "retrying at the next boundary",
                        rid,
                        outer,
                        sync_failures,
                        exc_info=True,
                    )
                    if sync_failures >= self._MAX_SYNC_FAILURES:
                        raise
                    extra = None
                if getattr(self, "_producers_started", False):
                    await self._group_buffer.resume()
                val_agg = self._aggregate_validation(
                    await self._validate_fixed(global_step)
                )
                val_reward = float(val_agg.get("validation_reward/_mean", float("nan")))
                await self._after_validation()

                reward_mean = sum(window_rewards) / len(window_rewards)
                logger.info(
                    "[replica %d] window %2d | loss %+.4f | reward %+.3f | "
                    "val %+.3f | rollouts %d | sync_every=%d | staleness=%d | %.1fs",
                    rid,
                    outer,
                    last["loss"],
                    reward_mean,
                    val_reward,
                    last["num_rollouts"],
                    self._sync_every,
                    last.get("staleness", 0),
                    time.perf_counter() - t0,
                )
                if extra:
                    logger.info("[replica %d] %s", rid, extra)
                # One machine-readable line per window, same wire format as
                # components/metrics.py::StdoutJsonLogger ("PFMETRICS " +
                # JSON with a "step" key) so the platform's telemetry sink
                # parses pretrain and RL with one grammar. This is what the
                # PanoFabric run page charts; keys are additive.
                pf: dict = {
                    "step": global_step,
                    "loss": float(last["loss"]),
                    "reward_mean": reward_mean,
                    "rollouts": last["num_rollouts"],
                    "sync_every": self._sync_every,
                    "staleness": last.get("staleness", 0),
                    "window_s": round(time.perf_counter() - t0, 3),
                }
                if not math.isnan(val_reward):
                    pf["val_reward"] = val_reward
                # getattr: tests drive train() with a stubbed _run_window.
                pf["trainer_idle_frac"] = round(
                    getattr(self, "_window_wait_s", 0.0)
                    / max(pf["window_s"], 1e-9),
                    3,
                )
                window_yield = self._window_yield_snapshot()
                if window_yield:
                    pf.update(window_yield)
                logger.info("PFMETRICS %s", json.dumps(pf, sort_keys=True))
                outer += 1

            # Idle the engines before the held-out pass (mirrors run()'s
            # producers-then-validate teardown order); the finally's stop is
            # then a no-op on this path.
            stop_producers = getattr(self, "_stop_producers_if_started", None)
            if stop_producers is not None:
                await stop_producers()
            await self._train_teardown()
            logger.info("[replica %d] post-training validation", rid)
            post_acc = self._aggregate_validation(
                await self._validate_fixed(global_step)
            )
            logger.info("[replica %d] pre=%s -> post=%s", rid, pre_acc, post_acc)
        finally:
            stop_producers = getattr(self, "_stop_producers_if_started", None)
            if stop_producers is not None:
                await stop_producers()
            await self._train_cleanup()


class PureLearnerReplica(RLControllerMixin, RLTrainer):
    """Base for a trainer that runs no local generation and consumes all
    training rollouts from a remote generator pool via a queue. Subclasses
    implement ``_next_rollout_batch`` (the queue source), ``_staleness_reference``
    (the version this replica compares consumed rollouts against), and
    ``_window_sync`` (their coordination), and call ``_spawn_trainer_only`` from
    their ``setup_async``."""

    @dataclass(kw_only=True, slots=True)
    class Config(RLTrainer.Config):
        sync_every: int = 4
        """Local training steps per logged window."""
        num_outer_steps: int = 0
        """Fixed number of windows to run (step-bound). Mutually exclusive with
        train_seconds: set exactly one."""
        train_seconds: float = 0.0
        """Wall-clock training budget (time-bound). Mutually exclusive with
        num_outer_steps: set exactly one."""
        replica_id: int = 0
        """Identifier for logging."""
        num_generators: int = 0
        """Always 0: a pure learner runs no local vLLM (generation is fully
        decoupled onto the remote worker pool). Kept as an explicit field so
        ``decentralized_rl/train.py`` provisions only the trainer GPU and spawns no
        generator meshes."""
        max_staleness: int = 4
        """Maximum version lag (in the subclass's staleness-reference space) of
        a consumed rollout batch; staler batches are dropped at consume time
        (async_level, enforced by dropping)."""
        buffer_groups: int = 0
        """Rollout-buffer capacity in groups; a full buffer back-pressures the
        consumer. 0 = two steps' worth
        (2 * async_loop.num_prompts_per_train_step)."""
        queue_poll_interval_s: float = 0.5
        """Seconds to wait before re-polling the rollout source when it yields
        nothing (only relevant for a non-blocking source, e.g. a remote hub)."""
        rollout_stall_timeout_s: float = 900.0
        """Fail the run if no rollout arrives for this many seconds while the
        trainer is waiting on the buffer. The remote generator pool is this
        trainer's ONLY rollout source, and a worker crash is invisible from
        here (the queue/hub stays reachable, just empty forever) -- without
        this bound the trainer polls silently for the rest of the run. Must
        comfortably exceed a worker cold start (vLLM engine init + first
        checkpoint pull + first round, ~3-5 min). 0 disables the bound."""

        def __post_init__(self):
            # A pure learner runs no local generator, so the base Controller
            # config's generator-centric validations (num_generators>=1,
            # generator checkpoint/hot_swap/cudagraph/batch_invariant-generator)
            # don't apply. Do the trainer-side essentials the base would
            # otherwise do: the SP-divisibility check and mirroring the batcher
            # width into trainer.training.seq_len for the model build.
            if self.trainer.parallelism.enable_sequence_parallel:
                sp_degree = self.trainer.parallelism.tensor_parallel_degree
                seq_len = self.async_loop.batcher.batch.seq_len
                if sp_degree > 1 and seq_len % sp_degree != 0:
                    raise ValueError(
                        f"RL batcher sequence length ({seq_len}) must be "
                        f"divisible by sequence parallel degree ({sp_degree})."
                    )
            self.trainer.training.seq_len = self.async_loop.batcher.batch.seq_len

            if self.sync_every < 1:
                raise ValueError(f"sync_every must be >= 1, got {self.sync_every}")
            if (self.num_outer_steps > 0) == (self.train_seconds > 0):
                raise ValueError(
                    "Set exactly one of num_outer_steps (step-bound run) or "
                    "train_seconds (time-bound run); got num_outer_steps="
                    f"{self.num_outer_steps}, train_seconds={self.train_seconds}"
                )
            if self.max_staleness < 1:
                raise ValueError(
                    f"max_staleness must be >= 1, got {self.max_staleness}"
                )
            if self.num_generators != 0:
                raise ValueError(
                    "a pure learner runs no local generators (generation is "
                    "fully decoupled onto the remote worker pool); "
                    f"num_generators must be 0, got {self.num_generators}"
                )

    # ------------------------------------------------------------------ #
    # Generator-less actor spawn.
    # ------------------------------------------------------------------ #

    async def _spawn_trainer_only(
        self, *, trainer_mesh, generator_meshes, policy_trainer_cls
    ):
        """Spawn ONLY the trainer actor (one learner GPU, ``policy_trainer_cls``)
        -- no generator mesh, no GeneratorRouter, no TorchStore. ``generator_meshes``
        must be empty (train.py spawns none when num_generators=0); it's accepted
        only to match the launcher's setup_async signature. ``generator_router``
        stays None (from RLTrainer.__init__), which close() tolerates."""
        cfg = self.config
        if generator_meshes:
            raise ValueError(
                "a pure learner spawns no generators, but "
                f"{len(generator_meshes)} generator mesh(es) were provisioned; "
                "set num_generators=0"
            )
        trainer_parallelism = cfg.trainer.parallelism
        dp_shard = max(trainer_parallelism.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = (
            trainer_parallelism.data_parallel_replicate_degree * dp_shard
        )
        self._proc_meshes = [trainer_mesh]
        await setup_mesh_elastic_env(trainer_mesh)
        self.trainer = trainer_mesh.spawn(
            "trainer",
            policy_trainer_cls,
            cfg.trainer,
            model_spec=cfg.model_spec,
            hf_assets_path=cfg.hf_assets_path,
            generator_dtype=cfg.generator.model_dtype,
            compile_config=cfg.compile,
            output_dir=cfg.dump_folder,
        )

    # ------------------------------------------------------------------ #
    # Rollout source + staleness reference (subclass-provided).
    # ------------------------------------------------------------------ #

    async def _next_rollout_batch(self):
        """Return one ``(worker_id, version, groups)`` batch from the remote
        generator pool, or ``None`` if none is available right now (the base
        consumer then waits ``queue_poll_interval_s`` and retries)."""
        raise NotImplementedError

    def _staleness_reference(self) -> int:
        """The current version a consumed rollout's stamped version is compared
        against for the ``max_staleness`` bound (e.g. the latest published
        checkpoint version, or the shared parameter-server revision)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Consumer side.
    # ------------------------------------------------------------------ #

    async def _consume_remote_rollouts(self) -> None:
        """Drain the remote generator pool into the local buffer, tagging each
        group with the batch's generation version. Runs until cancelled."""
        while True:
            batch = await self._next_rollout_batch()
            if batch is None:
                await asyncio.sleep(self.config.queue_poll_interval_s)
                continue
            _worker_id, version, groups = batch
            for group in groups:
                await self._buffer.put((group, version))

    async def _buffer_get_checked(self):
        """buffer.get that fails fast when the feed is gone, instead of
        waiting forever (there are no local producers):
          - the single remote-consumer task died or exited; or
          - nothing has arrived for rollout_stall_timeout_s. A crashed remote
            WORKER is invisible from here -- the queue/hub stays reachable,
            just empty forever -- so an empty buffer past any plausible worker
            cold start means the generator pool is dead."""
        timeout = self.config.rollout_stall_timeout_s
        deadline = asyncio.get_running_loop().time() + timeout if timeout > 0 else None
        get_task = asyncio.ensure_future(self._buffer.get())
        try:
            while not get_task.done():
                consumer = self._remote_consumer_task
                if consumer.done():
                    if not consumer.cancelled() and consumer.exception() is not None:
                        raise RuntimeError(
                            "remote rollout consumer died"
                        ) from consumer.exception()
                    raise RuntimeError(
                        "remote rollout consumer exited while the trainer "
                        "still needs data"
                    )
                remaining = None
                if deadline is not None:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise RuntimeError(
                            f"no rollout arrived for {timeout:.0f}s -- the "
                            "remote generator pool is likely dead (worker "
                            "crash?) or unable to reach this trainer's queue; "
                            "check the worker logs. (rollout_stall_timeout_s "
                            "bounds this wait; 0 disables it.)"
                        )
                await asyncio.wait(
                    {get_task, consumer},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=remaining,
                )
            return get_task.result()
        finally:
            if not get_task.done():
                get_task.cancel()

    async def _collect_and_build(self, step: int):
        """Assemble one training step's batch from the buffer, dropping groups
        staler than max_staleness (in the subclass's staleness-reference
        space). This overrides the mixin's generator-fanout collector, so
        _run_window's always-on pipeline overlaps this pop+build+batch with
        training instead of driving local generation.

        Uses the upstream pipeline (TrainingSampleBuilder -> Batcher) provided
        by the RLTrainer base: each surviving remote group is built into a
        training-sample group and fed to the packing batcher until it yields a
        batch. Returns ``(packed_batch, rollout_groups)``."""
        await self.trainer.sync_log_step.call(step)
        rollout_groups = []
        packed = None
        # Silent-hang guard (training_sample_builder's own TODO(robustness)): if
        # every group is dropped -- too stale, zero-std reward, or untrainable --
        # this loop consumes the queue forever, trains nothing, and says nothing,
        # because the drop counters live in metrics that only flush AT a train
        # step. Diagnosing that once cost an hour of a paid 4B cluster's time, so
        # the loop now reports itself. Counted per call; a batch that legitimately
        # needs several groups is normal and stays quiet.
        seen = stale = untrainable = 0
        while packed is None:
            group, version = await self._buffer_get_checked()
            seen += 1
            if self._staleness_reference() - version > self.config.max_staleness:
                self._num_dropped += 1
                stale += 1
                self._warn_if_starved(step, seen, stale, untrainable)
                continue
            rollout_groups.append(group)
            training_sample_group = self._training_sample_builder.build_from_group(
                rollout_group=group
            )
            # getattr: this counter is diagnostics only and must never be the
            # thing that breaks a train step -- test fakes and any future
            # builder return shape are tolerated.
            if not getattr(training_sample_group, "training_samples", None):
                untrainable += 1
            packed = self._batcher.add_training_samples(
                training_sample_group=training_sample_group
            )
            if packed is None:
                self._warn_if_starved(step, seen, stale, untrainable)
        # Window-level accounting for the PFMETRICS line: trainable yield is
        # THE decoupled health number (a 4B DAPO run measured 84% of groups
        # discarded as zero-std -- every throughput and $ metric is off ~6x
        # unless read against it). Lazy init tolerates the object.__new__
        # test fakes.
        y = getattr(self, "_win_yield", None)
        if y is None:
            y = self._win_yield = {
                "groups_consumed": 0, "dropped_stale": 0, "dropped_zero_std": 0,
            }
        y["groups_consumed"] += seen
        y["dropped_stale"] += stale
        y["dropped_zero_std"] += untrainable
        return packed, rollout_groups

    def _window_yield_snapshot(self) -> dict | None:
        y = getattr(self, "_win_yield", None)
        self._win_yield = None
        return y

    # Warn once per power-of-two so a long starvation escalates in the log
    # without flooding it.
    _STARVE_WARN_AT = 32

    def _warn_if_starved(
        self, step: int, seen: int, stale: int, untrainable: int
    ) -> None:
        """Say something when a train step cannot be assembled.

        `_collect_and_build` is unbounded by design (rollouts arrive
        asynchronously), so the failure mode is silence, not an exception. Emit
        at 32 groups and then at every doubling, naming WHY groups are going
        away -- the reason is otherwise unrecoverable, since the builder's
        drop counters flush only at a completed train step.
        """
        if seen < self._STARVE_WARN_AT or seen & (seen - 1):
            return
        logger.warning(
            "[replica %d] step %d: consumed %d rollout group(s) without "
            "assembling a batch (%d dropped as too stale > max_staleness=%d, "
            "%d dropped as untrainable/zero-std reward). Generation is feeding "
            "the queue but nothing is trainable -- check the reward function "
            "and rollout_reward/group_zero_std_frac.",
            self.config.replica_id, step, seen, stale,
            self.config.max_staleness, untrainable,
        )

    def _batch_staleness(
        self, pre_optim_policy_version: int, min_policy_versions
    ) -> int:
        """A pure learner's samples are stamped with the relay/hub checkpoint
        version their remote generator loaded -- a SHARED counter that advances
        per publish, not per local optim step -- so the mixin's default
        (local ``policy_version`` minus sample version) would subtract across
        two unrelated spaces and grow without bound. Measure against the same
        reference the ``max_staleness`` consume gate uses instead: the logged
        number is then directly comparable to ``config.max_staleness``."""
        del pre_optim_policy_version  # local optim-step space; not comparable
        return (
            self._staleness_reference() - min(min_policy_versions)
            if min_policy_versions
            else 0
        )

    async def _refresh_generators(self) -> None:
        """No-op: there is no local generator to refresh (the remote workers
        pull weights from the relay tier, out of band from the train step)."""

    # ------------------------------------------------------------------ #
    # No trainer-side validation (a pure learner runs no generator).
    # ------------------------------------------------------------------ #

    async def _validate_fixed(self, step: int):
        """Validation needs greedy generation, which a pure learner can't do
        (no vLLM here). Skip it: the learning curve is instead the per-window
        mean reward of the remote workers' rollouts, which train() already logs
        as it consumes them (it tracks a greedy held-out eval closely -- corr
        ~0.77, gap ~0.03 across the overnight runs). Returns no metrics so the
        mixin's train() logs val=nan without touching a generator."""
        return []

    # ------------------------------------------------------------------ #
    # Controller-loop hooks shared by both strategies.
    # ------------------------------------------------------------------ #

    async def _train_setup(self) -> None:
        cfg = self.config
        self._buffer = asyncio.Queue(
            maxsize=cfg.buffer_groups or 2 * cfg.async_loop.num_prompts_per_train_step
        )
        self._num_dropped = 0
        self._remote_consumer_task = asyncio.create_task(
            self._consume_remote_rollouts(), name="remote_rollout_consumer"
        )

    async def _train_cleanup(self) -> None:
        task = getattr(self, "_remote_consumer_task", None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

