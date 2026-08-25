# Copyright (c) Panocular AI.
#
# Tests for the heloco_async_inference coordination plane (the revision-watch
# publish loop, over the real HTTP relay wire) and the
# HeLoCoAsyncInferenceReplica controller (config validation, the HeLoCo
# window-sync push/pull with no generator refresh, and an end-to-end train
# loop that proves zero local generation). The pure-learner consumer/buffer/
# staleness machinery is inherited from PureLearnerReplica, and the shared
# rollout queue (push/pop wire protocol, timeout robustness) lives in
# the rollout_queue module -- both covered by test_async_inference.
# CPU-only: no GPU, no vLLM, no Monarch actors -- everything above the wire
# is faked.

import asyncio
import json
import logging
import itertools
from types import SimpleNamespace

import pytest
import torch
from aiohttp.test_utils import TestServer

from panoengine.train.rl.config_registry import base_rl_config, wrap_replica

from panoengine.decentralized.relay import RelayClient, RelayServer
from panoengine.decentralized.parameter_server import _watch_and_publish
from panoengine.train.rl.replicas import HeLoCoAsyncInferenceReplica


def _ep(fn):
    """A Monarch-endpoint stand-in: exposes .call(...)."""
    return SimpleNamespace(call=fn)


async def _start_relay(retain_last=5):
    relay = RelayServer(retain_last=retain_last)
    server = TestServer(relay.app())
    await server.start_server()
    base_url = str(server.make_url("")).rstrip("/")
    return relay, server, base_url


# --------------------------------------------------------------------- #
# _consistent_snapshot / _watch_and_publish (parameter_server.py's publish loop).
# --------------------------------------------------------------------- #


def test_watch_and_publish_publishes_at_startup_and_on_cadence():
    """The publish loop runs in the parameter-server process and publishes to
    the SEPARATE relay/queue process over HTTP (RelayClient against a real
    loopback relay here, the split-process wire path). Revision is stable
    between reads (a FakeServer backed by mutable state, not a one-shot
    iterator -- _consistent_snapshot itself re-reads the revision). Published
    checkpoint versions are server revision + 1 (never 0 -- see
    _watch_and_publish's docstring: AsyncInferenceWorker starts at version 0
    and RelayClient.fetch_latest requires a strictly newer version, so a raw
    revision-0 publish could never be fetched by a fresh worker). Publishes
    are bf16 over the wire."""
    model = torch.nn.Linear(2, 2)
    state = SimpleNamespace(revision=0)

    class FakeServer:
        def status(self):
            return {"revision": state.revision}

    async def scenario():
        relay, server, base_url = await _start_relay()
        watcher = asyncio.create_task(
            _watch_and_publish(
                FakeServer(),
                model,
                [n for n, _ in model.named_parameters()],
                RelayClient([base_url]),
                num_shards=1,
                publish_every_revisions=1,
                poll_interval_s=0.01,
            )
        )
        try:
            await asyncio.wait_for(_poll_until(lambda: relay.latest_version() == 1), 2)
            assert (
                relay.get_manifest(1) is not None
            )  # startup publish (revision 0 -> v1)

            state.revision = 1
            await asyncio.wait_for(_poll_until(lambda: relay.latest_version() == 2), 2)

            # The wire checkpoint is bf16 (halved publishes/downloads) and
            # round-trips through the worker's fetch path.
            fetched = await RelayClient([base_url]).fetch_latest()
            assert fetched is not None
            _version, sd = fetched
            assert all(t.dtype == torch.bfloat16 for t in sd.values())
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            await server.close()

    asyncio.run(scenario())


async def _poll_until(condition, interval=0.01):
    while not condition():
        await asyncio.sleep(interval)


# --------------------------------------------------------------------- #
# HeLoCoAsyncInferenceReplica.Config validation.
# --------------------------------------------------------------------- #


def test_config_pure_learner_defaults_and_validation():
    base = base_rl_config()
    cfg = wrap_replica(
        HeLoCoAsyncInferenceReplica,
        base,
        num_outer_steps=1,
    )
    # Pure learner: num_generators defaults to 0 (no local vLLM) and any
    # non-zero value is rejected.
    assert cfg.num_generators == 0
    with pytest.raises(ValueError, match="max_staleness must be >= 1"):
        wrap_replica(
            HeLoCoAsyncInferenceReplica,
            base,
            num_outer_steps=1,
            max_staleness=0,
        )
    with pytest.raises(ValueError, match="num_generators must be 0"):
        wrap_replica(
            HeLoCoAsyncInferenceReplica,
            base,
            num_outer_steps=1,
            num_generators=2,
        )


# --------------------------------------------------------------------- #
# HeLoCo outer step (window sync). The consumer/buffer/staleness machinery is
# inherited from PureLearnerReplica and covered by the async_inference tests.
# --------------------------------------------------------------------- #


def make_replica():
    """A HeLoCoAsyncInferenceReplica with only the window-sync state, skipping
    RLTrainer.__init__ (no actors, no torchft client)."""
    r = object.__new__(HeLoCoAsyncInferenceReplica)
    r.config = SimpleNamespace(
        queue_poll_interval_s=0,
        rollout_stall_timeout_s=0,
        max_staleness=4,
        async_loop=SimpleNamespace(num_prompts_per_train_step=2),
    )
    r._buffer = asyncio.Queue()
    r._num_dropped = 0
    r._last_known_revision = 10
    return r


def test_window_sync_pushes_pseudograd_and_updates_revision_without_touching_generators():
    """The pure-learner outer step: pull theta_local, client.push (server
    outer step), load the returned global theta back -- and NO generator
    refresh (there is no local generator). Refreshes the revision freshness
    reference (client.revision + 1) and reports buffer stats."""

    async def scenario():
        r = make_replica()
        r._num_dropped = 3
        r._sync_every = 4

        loaded = []
        theta_local = {"w": torch.zeros(2)}
        new_global = {"w": torch.ones(2)}

        async def get_full():
            return theta_local

        async def load_full(sd):
            loaded.append(sd)

        r.trainer = SimpleNamespace(
            get_full_state_dict_cpu=_ep(get_full),
            load_full_state_dict_cpu=_ep(load_full),
        )
        r._get_rank_0_value = lambda x: x

        pushed = []

        class _FakeClient:
            revision = 42
            last_dylu_steps = 0

            def push(self, local_sd, speed):
                pushed.append((local_sd, speed))
                return new_global

        r.client = _FakeClient()

        stats = await r._window_sync(0.0)

        # Pushed the local theta and adopted the returned global theta.
        assert pushed and pushed[0][0] is theta_local
        assert loaded == [new_global]
        # Revision reference shifted by +1 (see parameter_server.py's version-shift note);
        # no generator_router call anywhere in the pure-learner window sync.
        assert r._last_known_revision == 43
        assert "dropped=3" in stats
        assert r._num_dropped == 0

    asyncio.run(scenario())


def test_window_sync_does_not_block_the_event_loop():
    """client.push is torchft's SYNCHRONOUS multi-GB roundtrip; _window_sync
    must run it in a thread so the rollout consumer keeps draining the queue
    during the sync -- a push executed on the event loop would freeze every
    coroutine for the duration (the old behavior: the buffer stopped filling
    at every window boundary)."""

    async def scenario():
        r = make_replica()
        r._sync_every = 4

        async def get_full():
            return {"w": torch.zeros(2)}

        async def load_full(sd):
            del sd

        r.trainer = SimpleNamespace(
            get_full_state_dict_cpu=_ep(get_full),
            load_full_state_dict_cpu=_ep(load_full),
        )
        r._get_rank_0_value = lambda x: x

        class _SlowClient:
            revision = 0
            last_dylu_steps = 0

            def push(self, local_sd, speed):
                import time as _time

                _time.sleep(0.2)  # thread-blocking, like the real HTTP client
                return {"w": torch.ones(2)}

        r.client = _SlowClient()

        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)

        ticker_task = asyncio.create_task(ticker())
        try:
            await asyncio.wait_for(r._window_sync(0.0), 5)
        finally:
            ticker_task.cancel()
            await asyncio.gather(ticker_task, return_exceptions=True)
        # The loop ran throughout the 0.2s push (>= ~10 ticks; allow slack).
        assert ticks >= 5, f"event loop was blocked during push (ticks={ticks})"

    asyncio.run(scenario())


# --------------------------------------------------------------------- #
# End-to-end controller loop on fakes (mirrors the GPU smoke's shape) for
# the PURE-LEARNER trainer: no generator, no generator_router, no
# push_model_state_dict. Those attributes are deliberately never set, so if
# any code path tried to touch a generator the test would AttributeError --
# i.e. the test proves the trainer runs zero generation.
# --------------------------------------------------------------------- #


def _e2e_group(num_tokens=5, reward=0.5):
    turn = SimpleNamespace(
        prompt_token_ids=[0, 0], completion_token_ids=[0] * (num_tokens - 1)
    )
    return SimpleNamespace(rollouts=[SimpleNamespace(turns=[turn], reward=reward)])


class _InfiniteQueueClient:
    """Always returns a batch tagged at ``version`` -- the generator pool
    feeding the trainer, never empty."""

    def __init__(self, *, version=1, groups_per_batch=4):
        self.version = version
        self.groups_per_batch = groups_per_batch
        self.pops = 0

    async def pop(self):
        self.pops += 1
        return (0, self.version, [_e2e_group() for _ in range(self.groups_per_batch)])


def test_train_end_to_end_pure_learner_on_fakes(caplog):
    async def scenario():
        r = object.__new__(HeLoCoAsyncInferenceReplica)
        r.config = SimpleNamespace(
            replica_id=0,
            sync_every=2,
            num_outer_steps=2,
            train_seconds=0.0,
            async_loop=SimpleNamespace(num_prompts_per_train_step=2),
            buffer_groups=0,
            max_staleness=4,
            queue_poll_interval_s=0,
            rollout_stall_timeout_s=0,
        )
        r._policy_version = 0
        # Rollouts arrive tagged v1; the trainer's freshness reference is v1
        # too, so nothing is dropped for staleness.
        r._last_known_revision = 1
        r._queue_client = _InfiniteQueueClient(version=1)

        versions = itertools.count(1)
        loaded = []
        pushed = []

        async def get_full():
            return {"w": 0}

        async def load_full(sd):
            loaded.append(sd)

        r.trainer = SimpleNamespace(
            sync_log_step=_ep(lambda step: _areturn(None)),
            forward_backward=_ep(lambda mb, n: _areturn({"loss": 0.25})),
            optim_step=_ep(
                lambda: _areturn(SimpleNamespace(policy_version=next(versions)))
            ),
            get_full_state_dict_cpu=_ep(get_full),
            load_full_state_dict_cpu=_ep(load_full),
        )
        r._get_rank_0_value = lambda x: x
        r.trainer_dp_degree = 1
        # The current pipeline builds the training-sample builder + batcher from
        # config in _build_sync_pipeline; this fake replica has no real config to
        # build from, so no-op it and wire passthrough fakes: the builder passes
        # each RolloutGroup through and the batcher returns one packed batch per
        # group (one microbatch, valid-token count, generator policy versions).
        r._build_sync_pipeline = lambda: None
        r._training_sample_builder = SimpleNamespace(
            build_from_group=lambda *, rollout_group: rollout_group
        )
        r._batcher = SimpleNamespace(
            add_training_samples=lambda *, training_sample_group: SimpleNamespace(
                microbatches=["mb"],
                num_global_valid_tokens=4,
                min_policy_versions=[0],
            )
        )

        class _FakeClient:
            revision = 0
            last_dylu_steps = 0

            def push(self, local_sd, speed):
                pushed.append(speed)
                return {"w": 1}

        r.client = _FakeClient()
        r._aggregate_validation = lambda metrics: {}

        with caplog.at_level(logging.INFO):
            await asyncio.wait_for(r.train(), 15)

        # The pure-learner path tracks per-window rollout accounting and
        # ships it on the PFMETRICS line (trainable yield -- see
        # _collect_and_build). 2 windows x 2 steps x 1 group per packed
        # batch = 2 consumed per window; the passthrough builder fake has
        # no .training_samples, so every group also counts untrainable.
        pf_lines = [rec.getMessage() for rec in caplog.records
                    if rec.getMessage().startswith("PFMETRICS ")]
        assert len(pf_lines) == 2
        for line in pf_lines:
            payload = json.loads(line.removeprefix("PFMETRICS "))
            assert payload["groups_consumed"] == 2
            assert payload["dropped_stale"] == 0
            assert payload["dropped_zero_std"] == 2
        # read-and-reset: nothing carries over past the last window
        assert r._window_yield_snapshot() is None

        # 2 windows x sync_every=2 optim steps.
        assert r._policy_version == 4
        # One HeLoCo push per window boundary.
        assert len(pushed) == 2
        assert len(loaded) == 2  # adopted global theta each window
        # The remote-rollout consumer ran and was cleaned up.
        assert r._queue_client.pops > 0
        assert r._remote_consumer_task.done()
        # Never spawned a generator: the attribute was never set, and nothing
        # touched it (else AttributeError would have failed train()).
        assert not hasattr(r, "generator_router")

    asyncio.run(scenario())


async def _areturn(value):
    return value
