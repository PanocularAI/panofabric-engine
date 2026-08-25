# Copyright (c) Panocular AI.
#
# Tests for the shared RL coordination infrastructure: config_registry.py (model/
# flavor/task/GPU-count resolution and Config validation), controller.py
# (the window runner and the template train() loop), train.py
# (PerHostProvisioner), and the package's CPU-light import guarantee.
# Per-strategy tests live next to their package (heloco/test_heloco.py etc).

import asyncio
import json
import logging
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

# Straight from the canonical module, not through the top-level
# `torchtitan.experiments.decentralized_rl.config_registry` shim of old: it was an
# `import *`
# re-export, and a star-import does not carry the leading-underscore names this
# file asserts on.
from panoengine.train.rl.config_registry import (
    _DEFAULT_HF_ASSETS_PATH,
    _MODEL_REGISTRY_BY_MODEL,
    _RENDERER_NAME_BY_MODEL,
    base_rl_config,
    rl_diloco_llama3_8b,
    rl_heloco_llama3_8b,
    rl_heloco_qwen3_1_7b,
    wrap_replica,
)
from panoengine.train.rl.controller import RLControllerMixin
from panoengine.train.rl.replicas import (
    DiLoCoRLReplica,
    HeLoCoRLReplica,
)
from panoengine.train.rl.train import PerHostProvisioner

#: The CPU-light guard runs a child interpreter. It used to rely on cwd=<torchtitan
#: repo root> to make the packages importable; from here that path no longer exists,
#: and handing the child our own sys.path works whether we are installed or on a
#: PYTHONPATH from two sibling checkouts.
CHILD_PYTHONPATH = os.pathsep.join(p for p in sys.path if p)


# === config_registry.py ====================================================


def test_model_and_flavor_resolution():
    """flavor selects the model spec and the default checkpoint (an explicit
    hf_assets_path still wins); llama3 resolves as a second model family
    through the SAME builder functions; unknown models fail clearly."""
    cfg_default = base_rl_config()
    cfg_large = base_rl_config(flavor="1.7B")
    assert cfg_default.model_spec.flavor == "0.6B"
    assert cfg_large.model_spec.flavor == "1.7B"
    assert cfg_large.hf_assets_path == _DEFAULT_HF_ASSETS_PATH[("qwen3", "1.7B")]
    assert cfg_large.hf_assets_path != cfg_default.hf_assets_path
    cfg_override = base_rl_config(hf_assets_path="/tmp/custom", flavor="1.7B")
    assert cfg_override.hf_assets_path == "/tmp/custom"
    # The named 1.7B preset routes through the same flavor resolution.
    assert rl_heloco_qwen3_1_7b().hf_assets_path == cfg_large.hf_assets_path

    # A second model family: no coordinator code is model-specific.
    for fn, cls in (
        (rl_heloco_llama3_8b, HeLoCoRLReplica),
        (rl_diloco_llama3_8b, DiLoCoRLReplica),
    ):
        cfg = fn()
        assert isinstance(cfg, cls.Config)
        assert cfg.model_spec.name == "llama3" and cfg.model_spec.flavor == "8B"
        # llama3 has no dedicated renderer entry -- resolved via the "default" key.
        assert cfg.renderer.name == "default"
        assert cfg.hf_assets_path.endswith("Llama-3.1-8B")

    with pytest.raises(ValueError, match="unknown RL model 'bogus_model'"):
        base_rl_config(model="bogus_model")


def test_new_model_needs_only_registry_dict_entries():
    """The extension contract in practice: register a fake model purely at
    runtime (no filesystem changes) by adding entries to the three module-
    level dicts, and confirm base_rl_config resolves it with no other
    RL coordination code touched. Without the hf_assets_path entry, resolution
    fails with a clear error rather than a silent bogus default."""
    from torchtitan.models.qwen3 import model_registry as qwen3_model_registry

    _MODEL_REGISTRY_BY_MODEL[
        "_fake_for_test"
    ] = lambda flavor, *, attn_backend, hf_assets_path: qwen3_model_registry(
        "0.6B", attn_backend=attn_backend
    )
    _RENDERER_NAME_BY_MODEL["_fake_for_test"] = "auto"
    try:
        with pytest.raises(ValueError, match="no default hf_assets_path"):
            base_rl_config(model="_fake_for_test", flavor="tiny")

        _DEFAULT_HF_ASSETS_PATH[("_fake_for_test", "tiny")] = "/fake/checkpoints/tiny"
        try:
            cfg = base_rl_config(model="_fake_for_test", flavor="tiny")
            assert cfg.hf_assets_path == "/fake/checkpoints/tiny"
            assert cfg.renderer.name == "auto"
        finally:
            del _DEFAULT_HF_ASSETS_PATH[("_fake_for_test", "tiny")]
    finally:
        del _MODEL_REGISTRY_BY_MODEL["_fake_for_test"]
        del _RENDERER_NAME_BY_MODEL["_fake_for_test"]


def test_tensor_parallel_degree_is_a_real_gpu_count_knob():
    """trainer/generator tensor_parallel_degree must not be hardcoded -- the
    number of GPUs a role needs is a function of these, so a model too big
    for one GPU per role must be expressible without editing the RL package: as a
    Python kwarg AND as a CLI overlay on a named --config preset (the real
    ConfigManager path)."""
    from torchtitan.config import ConfigManager

    default_cfg = base_rl_config()
    assert default_cfg.trainer.parallelism.tensor_parallel_degree == 1
    assert default_cfg.generator.parallelism.tensor_parallel_degree == 1

    scaled_cfg = base_rl_config(
        trainer_tensor_parallel_degree=4, generator_tensor_parallel_degree=2
    )
    assert scaled_cfg.trainer.parallelism.tensor_parallel_degree == 4
    assert scaled_cfg.generator.parallelism.tensor_parallel_degree == 2

    cfg = ConfigManager().parse_args(
        [
            "--module",
            # the canonical path. `decentralized_rl` is a control-plane spec
            # value, translated to this by controld's engine_module(); the
            # engine ships no package under that name.
            "panoengine.train.rl",
            "--config",
            "rl_heloco_qwen3_0_6b",
            "--trainer.parallelism.tensor_parallel_degree",
            "2",
            "--generator.parallelism.tensor_parallel_degree",
            "8",
        ]
    )
    assert cfg.trainer.parallelism.tensor_parallel_degree == 2
    assert cfg.generator.parallelism.tensor_parallel_degree == 8


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(sync_every=0, train_seconds=60.0), "sync_every"),
        (dict(train_seconds=0.0), "exactly one"),  # neither bound
        (dict(train_seconds=60.0, num_outer_steps=4), "exactly one"),  # both
    ],
)
def test_config_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        wrap_replica(HeLoCoRLReplica, base_rl_config(), **kwargs)


def test_rollouter_is_a_swappable_task():
    """The task bundle (dataset + reward rubric + environment) must be
    injectable without touching any coordinator code: a different
    Rollouter.Config passed to base_rl_config flows through wrap_replica into
    every strategy's Config unchanged."""
    from torchtitan.experiments.rl.examples.alphabet_sort import AlphabetSortRollouter

    custom = AlphabetSortRollouter.Config()  # stands in for any other task's Config
    base = base_rl_config(rollouter=custom)
    assert base.rollouter is custom
    cfg = wrap_replica(HeLoCoRLReplica, base, train_seconds=60.0)
    assert cfg.rollouter is custom


def test_dapo_math_preset():
    """The DAPO-Math preset swaps the task bundle AND the reference recipe's
    training knobs onto the heloco stack: Math-Verify rollouter, DAPO
    clip-higher loss under the chunked wrapper, thinking on, temp/top-p 1.0,
    and math-sized token budgets (alphabet-sort's 700-token budget would
    truncate every solution). math_verify is an example-only dep, so skip
    where it isn't installed -- and its absence must not break the registry's
    OTHER presets (the lazy-import contract)."""
    base_rl_config()  # registry import + alphabet-sort preset work regardless

    pytest.importorskip("math_verify")
    from torchtitan.components.loss import ChunkedLossWrapper
    from panoengine.train.rl.config_registry import (
        rl_heloco_dapo_math_qwen3_0_6b,
        rl_heloco_dapo_math_qwen3_4b,
    )
    from torchtitan.experiments.rl.examples.dapo_math import DapoMathRollouter
    from torchtitan.experiments.rl.losses import DAPOLoss

    cfg = rl_heloco_dapo_math_qwen3_0_6b()
    assert isinstance(cfg, HeLoCoRLReplica.Config)
    assert isinstance(cfg.rollouter, DapoMathRollouter.Config)
    assert cfg.rollouter.token_env.max_rollout_tokens == 10240
    assert isinstance(cfg.trainer.loss, ChunkedLossWrapper.Config)
    assert isinstance(cfg.trainer.loss.loss_fn, DAPOLoss.Config)
    assert cfg.trainer.loss.loss_fn.ratio_clip_high == 0.28
    assert cfg.renderer.enable_thinking is True
    assert cfg.generator.sampling.temperature == 1.0
    assert cfg.generator.sampling.max_tokens == 8192
    assert cfg.async_loop.batcher.batch.seq_len == 10240
    # One packed sequence per rank: 2 OOMs a 140 GiB H200 at this seq_len.
    assert cfg.async_loop.batcher.batch.local_batch_size == 1
    assert cfg.async_loop.validation.num_samples == 30

    # The reference LOOP, not just the reference task: without these the
    # preset ran alphabet-sort's 5x8 step at lr 2e-6 with warmup+decay, and
    # reproducing upstream needed an override pile at the call site.
    assert cfg.async_loop.num_prompts_per_train_step == 8
    assert cfg.async_loop.num_samples_per_prompt == 16
    # The reference's rollout lag. Upstream (fd2776584) made
    # max_offpolicy_steps a DERIVED bound and moved the knob to
    # target_offpolicy_steps + window_fraction, so assert the knobs we set --
    # matching upstream's own dapo_math preset -- not the derived value.
    assert cfg.async_loop.target_offpolicy_steps == 4
    assert cfg.async_loop.window_fraction == 0.3
    assert cfg.trainer.optimizer.param_groups[0].optimizer_kwargs["lr"] == 1e-6
    assert cfg.trainer.lr_scheduler.warmup_steps == 0
    assert cfg.trainer.lr_scheduler.min_lr_factor == 1.0  # constant LR

    large = rl_heloco_dapo_math_qwen3_4b()
    assert large.model_spec.flavor == "4B"
    assert large.hf_assets_path.endswith("Qwen3-4B-Base")
    # Default preserved when not passed.
    from torchtitan.experiments.rl.examples.alphabet_sort import AlphabetSortRollouter

    assert type(base_rl_config().rollouter) is AlphabetSortRollouter.Config


def test_dapo_math_decoupled_presets():
    """The decoupled (pure-learner + worker-pool) DAPO-Math pair. The launcher
    derives the worker preset name from the trainer's by inserting the engine's
    _worker_ segment, so both names must exist -- and the worker's group_size
    must match the trainer's num_samples_per_prompt or its groups are unusable."""
    pytest.importorskip("math_verify")
    from panoengine.train.rl.config_registry import (
        rl_heloco_async_inference_dapo_math_qwen3_0_6b,
        rl_heloco_async_inference_worker_dapo_math_qwen3_0_6b,
    )
    from panoengine.train.rl.replicas import (
        HeLoCoAsyncInferenceReplica,
    )
    from torchtitan.experiments.rl.examples.dapo_math import DapoMathRollouter

    trainer = rl_heloco_async_inference_dapo_math_qwen3_0_6b(num_outer_steps=2)
    assert isinstance(trainer, HeLoCoAsyncInferenceReplica.Config)
    assert trainer.num_generators == 0        # pure learner, still
    assert isinstance(trainer.rollouter, DapoMathRollouter.Config)
    assert trainer.async_loop.num_samples_per_prompt == 16
    assert trainer.trainer.optimizer.param_groups[0].optimizer_kwargs["lr"] == 1e-6

    worker = rl_heloco_async_inference_worker_dapo_math_qwen3_0_6b()
    assert isinstance(worker.rollouter, DapoMathRollouter.Config)
    assert worker.group_size == trainer.async_loop.num_samples_per_prompt
    assert worker.renderer.enable_thinking is True
    assert worker.generator.sampling.max_tokens == 8192


# === controller.py =========================================================


class _WindowHost(RLControllerMixin):
    """Mixin host with the generator/trainer work replaced by event recorders."""

    def __init__(self, *, losses=None, collect_s=0.002, train_s=0.004):
        self.config = SimpleNamespace(replica_id=0)
        self._policy_version = 0
        self.events = []
        self._losses = iter(losses or [])
        self._collect_s = collect_s
        self._train_s = train_s

    async def _collect_and_build(self, step):
        self.events.append(("collect_start", step))
        await asyncio.sleep(self._collect_s)
        self.events.append(("collect_end", step))
        return [f"mb{step}"], []

    async def _train_on(self, packed, rollout_groups):
        step = int(packed[0][2:])
        self.events.append(("train_start", step))
        await asyncio.sleep(self._train_s)
        self.events.append(("train_end", step))
        loss = next(self._losses, 0.1)
        return {
            "loss": loss,
            "reward_mean": 0.5,
            "policy_version": step,
            "num_rollouts": 8,
            "staleness": 0,
        }


def test_run_window_pipelined_overlap_and_divergence_cancel():
    # _run_window is always pipelined (one-step-ahead generation/training
    # overlap); there is no sequential path.
    host = _WindowHost()
    rewards, last, global_step, diverged = asyncio.run(host._run_window(3, 0))
    assert not diverged
    assert global_step == 3 and len(rewards) == 3
    ev = host.events
    # Collection for step h+1 starts before training on step h finishes.
    assert ev.index(("collect_start", 2)) < ev.index(("train_end", 1))
    assert ev.index(("collect_start", 3)) < ev.index(("train_end", 2))
    # No collection for a step beyond the window (nothing straddles the boundary).
    assert ("collect_start", 4) not in ev

    # A non-finite loss stops the window early (divergence).
    host = _WindowHost(losses=[0.1, math.inf])
    rewards, last, global_step, diverged = asyncio.run(host._run_window(4, 0))
    assert diverged and len(rewards) == 2

    # Regression: divergence must CANCEL the already-launched collection for
    # the next step, not leak it to run during shutdown.
    host = _WindowHost(losses=[math.nan], collect_s=0.05)

    async def run():
        result = await host._run_window(3, 0)
        # Give a leaked task time to run if one existed.
        await asyncio.sleep(0.1)
        return result

    rewards, last, global_step, diverged = asyncio.run(run())
    assert diverged and len(rewards) == 1
    # Step 2's collection was launched but cancelled before completing.
    assert ("collect_start", 2) in host.events
    assert ("collect_end", 2) not in host.events


class _LoopHost(RLControllerMixin):
    def __init__(
        self,
        *,
        num_outer_steps=0,
        train_seconds=0.0,
        sync_every=2,
        window_s=0.0,
        diverge_after=None,
        sync_fails=(),
    ):
        self.config = SimpleNamespace(
            replica_id=3,
            sync_every=sync_every,
            num_outer_steps=num_outer_steps,
            train_seconds=train_seconds,
        )
        self._policy_version = 0
        self.calls = []
        self._window_s = window_s
        self._diverge_after = diverge_after
        # 1-based _window_sync attempt numbers that raise.
        self._sync_fails = set(sync_fails)
        self.sync_attempts = 0
        self.windows_run = 0

    def _build_sync_pipeline(self):
        # Provided by RLTrainer on real hosts; the loop tests fake the window
        # runner entirely, so the pipeline is never used. Not recorded in
        # self.calls to keep the asserted hook sequences focused on the loop.
        pass

    async def _run_window(self, sync_every, start_step):
        self.calls.append(("window", sync_every, start_step))
        self.windows_run += 1
        if self._window_s:
            await asyncio.sleep(self._window_s)
        diverged = (
            self._diverge_after is not None and self.windows_run > self._diverge_after
        )
        last = {
            "loss": math.nan if diverged else 0.1234,
            "reward_mean": 0.6,
            "policy_version": 1,
            "num_rollouts": 8,
            "staleness": 1,
        }
        return [0.5, 0.7], last, start_step + sync_every, diverged

    async def _validate_fixed(self, step):
        self.calls.append(("validate", step))
        return None

    def _aggregate_validation(self, metrics):
        return {"validation_reward/_mean": 0.25}

    async def _train_setup(self):
        self.calls.append(("setup",))

    async def _window_sync(self, t0):
        self.calls.append(("sync",))
        self.sync_attempts += 1
        if self.sync_attempts in self._sync_fails:
            raise TimeoutError("timed out")
        return "detail line"

    async def _after_validation(self):
        self.calls.append(("resume",))

    async def _train_teardown(self):
        self.calls.append(("teardown",))

    async def _train_cleanup(self):
        self.calls.append(("cleanup",))


def test_train_step_bound_hook_order():
    host = _LoopHost(num_outer_steps=2, sync_every=2)
    asyncio.run(host.train())
    assert host.calls == [
        ("validate", 0),  # pre-training validation before setup
        ("setup",),
        ("window", 2, 0),
        ("sync",),
        ("validate", 2),
        ("resume",),
        ("window", 2, 2),
        ("sync",),
        ("validate", 4),
        ("resume",),
        ("teardown",),
        ("validate", 4),  # post-training validation
        ("cleanup",),
    ]


_MONARCH_VERBS = frozenset(
    {"call", "call_one", "broadcast", "choose", "stream", "rref"}
)


def test_router_calls_name_public_endpoints_and_use_a_monarch_verb():
    """InterGeneratorRouter is upstream's, and it is a monarch Actor: every call
    must name an @endpoint and go through a monarch verb.

    Upstream actorized the router and privatized the implementations
    (fanout -> _fanout, pull_model_state_dict -> _pull_model_state_dict behind
    an endpoint of the same name). Our stack kept calling the old plain-object
    API, which a rebase cannot catch because it only sweeps upstream's files --
    an ActorEndpoint has no __call__, so A-4b died 12 GPU-minutes in with
    "ActorEndpoint object is not callable". Scanning the source is the cheap
    version of that discovery."""
    import re
    import types

    from torchtitan.experiments.rl.routing.inter_generator_router import (
        InterGeneratorRouter,
    )

    here = os.path.dirname(os.path.dirname(__file__))  # the package, not tests/
    pattern = re.compile(r"generator_router\.(\w+)(?:\.(\w+))?\s*\(")
    checked = 0
    for fname in ("controller.py", "replicas.py"):
        src = open(os.path.join(here, fname)).read()
        for name, verb in pattern.findall(src):
            checked += 1
            attr = getattr(InterGeneratorRouter, name, None)
            assert attr is not None, f"{fname}: router has no attribute {name!r}"
            assert not isinstance(attr, types.FunctionType), (
                f"{fname}: {name!r} is a plain method, not an endpoint -- "
                "upstream privatized it or never exposed it"
            )
            assert verb in _MONARCH_VERBS, (
                f"{fname}: {name!r} is called as {verb or 'a direct call'}; "
                f"an ActorEndpoint needs one of {sorted(_MONARCH_VERBS)}"
            )
    assert checked, "regex matched nothing -- the scan silently covered zero calls"


def test_train_emits_one_pfmetrics_line_per_window(caplog):
    """The platform's metrics sink parses `PFMETRICS {json}` lines (same
    grammar as components/metrics.py::StdoutJsonLogger). One line per window,
    keyed by global step; no yield keys on hosts that don't track them."""
    host = _LoopHost(num_outer_steps=2, sync_every=2)
    with caplog.at_level(logging.INFO):
        asyncio.run(host.train())
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("PFMETRICS ")]
    assert len(lines) == 2
    first = json.loads(lines[0].removeprefix("PFMETRICS "))
    assert first["step"] == 2 and first["rollouts"] == 8
    assert first["loss"] == pytest.approx(0.1234)
    assert first["reward_mean"] == pytest.approx(0.6)
    assert first["val_reward"] == pytest.approx(0.25)
    assert first["sync_every"] == 2 and first["staleness"] == 1
    assert first["window_s"] >= 0
    # _LoopHost inherits the mixin default (no yield tracking) -> keys absent.
    assert "groups_consumed" not in first
    second = json.loads(lines[1].removeprefix("PFMETRICS "))
    assert second["step"] == 4


def test_train_survives_transient_outer_sync_failures():
    """A dropped push must not end the run -- the hub still holds good weights
    and the next window boundary retries. Regression for run 947642665102,
    killed four steps in when one 60 s socket stall on the parameter server let
    a TimeoutError out of client.push. Runs of failures shorter than
    _MAX_SYNC_FAILURES must also RESET the counter, so a run that loses one
    push every few windows never accumulates its way to fatal."""
    host = _LoopHost(num_outer_steps=6, sync_every=2, sync_fails={1, 2, 4, 5})
    asyncio.run(host.train())
    assert host.windows_run == 6
    assert host.sync_attempts == 6
    assert ("teardown",) in host.calls


def test_train_gives_up_after_consecutive_outer_sync_failures():
    """A replica that keeps training but never merges looks healthy and
    produces nothing, so a permanently unreachable hub still has to be fatal.
    num_outer_steps bounds the loop so a broken guard fails instead of hanging."""
    host = _LoopHost(num_outer_steps=10, sync_every=2, sync_fails=range(1, 100))
    with pytest.raises(TimeoutError):
        asyncio.run(host.train())
    assert host.sync_attempts == RLControllerMixin._MAX_SYNC_FAILURES
    assert ("cleanup",) in host.calls  # the finally: block still ran


def test_train_divergence_skips_teardown_but_cleans_up():
    host = _LoopHost(num_outer_steps=5, diverge_after=1)
    asyncio.run(host.train())
    assert host.windows_run == 2
    assert ("teardown",) not in host.calls
    assert host.calls[-1] == ("cleanup",)
    # Only the first (healthy) window was synced and validated.
    assert host.calls.count(("sync",)) == 1
    assert [c for c in host.calls if c[0] == "validate"] == [
        ("validate", 0),
        ("validate", 2),
    ]


def test_train_time_bound_and_sync_retarget():
    # Time bound: the loop stops at the deadline, still tears down + cleans up.
    host = _LoopHost(train_seconds=0.08, window_s=0.05)
    asyncio.run(host.train())
    assert 1 <= host.windows_run <= 3
    assert host.calls[-1] == ("cleanup",)
    assert ("teardown",) in host.calls

    # _window_sync may retarget sync_every (a DyLU recommendation): the next
    # window adopts it.
    class _DyLUHost(_LoopHost):
        async def _window_sync(self, t0):
            self._sync_every = 5
            return None

    host = _DyLUHost(num_outer_steps=2, sync_every=2)
    asyncio.run(host.train())
    windows = [c for c in host.calls if c[0] == "window"]
    assert windows == [("window", 2, 0), ("window", 5, 2)]


# === train.py (PerHostProvisioner) =========================================


def _bootstrap_devices(bootstrap, monkeypatch):
    """Run a bootstrap callable and return the CUDA_VISIBLE_DEVICES it set."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    bootstrap()
    return os.environ["CUDA_VISIBLE_DEVICES"]


def test_provisioner_slices_pool_and_rejects_over_allocation(monkeypatch):
    # Whitespace-tolerant parse, then multi- and single-GPU slices carved out
    # of the pool in order.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " 1 , 4 ,6,7")
    prov = PerHostProvisioner(total_gpus=4)
    assert _bootstrap_devices(prov.allocate(2), monkeypatch) == "1,4"
    assert _bootstrap_devices(prov.allocate(1), monkeypatch) == "6"
    assert _bootstrap_devices(prov.allocate(1), monkeypatch) == "7"
    # The pool is exhausted.
    with pytest.raises(RuntimeError, match="only 0 available"):
        prov.allocate(1)

    # A replica asked to spawn more meshes than its CUDA_VISIBLE_DEVICES slice
    # can hold (e.g. 1 trainer + 2 engines on a 2-GPU slice).
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    prov = PerHostProvisioner(total_gpus=3)
    prov.allocate(1)
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES exposes"):
        prov.allocate(2)


# === __init__.py ============================================================


def test_package_import_stays_cpu_light():
    """The heloco parameter server and async_inference relay server are
    CPU-only processes that import their parent packages; a bare package
    import must not pull vLLM/monarch or the RL actor stack (which is why
    the __init__.py files re-export nothing)."""
    # The three modules the control plane launches on CPU nodes.
    # panoengine.train.rl itself is NOT in this list -- importing any coordinator
    # pulls torchtitan's RL stack and vLLM with it. No CPU-only process imports
    # it, and that is the point.
    code = (
        "import sys; "
        "import panoengine.decentralized.parameter_server, "
        "panoengine.decentralized.relay, "
        "panoengine.decentralized.rollout_queue; "
        "heavy = [m for m in sys.modules if m == 'vllm' or m == 'monarch' "
        "or m.startswith('torchtitan.experiments.rl.actors')]; "
        "assert not heavy, heavy; print('light')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": CHILD_PYTHONPATH},
    )
    assert out.returncode == 0, out.stderr
    assert "light" in out.stdout


def test_hf_backend_registry_resolution():
    """The HF transformers backend resolves through the same table-driven
    contract: dims come from the checkpoint's config.json (not derived), the
    titan-shaped layers view satisfies the trainer/generator assertion
    expression, and the near-identity state-dict adapter is wired."""
    from torchtitan.models.common.attention import FlexAttention

    cfg = base_rl_config(model="hf", flavor="Qwen3-0.6B")
    spec = cfg.model_spec
    # the exact expression asserted by rl/actors/trainer.py and generator.py
    # (the HF backend routes attention through its flex path)
    inner = spec.model.layers[0].attention.inner_attention
    assert isinstance(inner, FlexAttention.Config)
    attn = spec.model.layers[0].attention
    assert attn.head_dim == 128, "must come from config.json, not dim/n_heads"
    assert attn.n_kv_heads == 8
    assert spec.state_dict_adapter is not None
    assert cfg.renderer.name == "auto"
    assert cfg.hf_assets_path.endswith("Qwen3-0.6B")
    # trained untied (FSDP); the adapter aliases embeddings into lm_head at load
    assert spec.model.tie_word_embeddings is False


def test_hf_backend_covers_every_strategy():
    """Every coordination strategy (and both decoupled worker roles) has an HF
    preset that is a pure model/flavor redirect of its native counterpart —
    the strategies themselves are model-agnostic, so the redirect plus the
    registry tables is the whole integration surface. Resolve each preset and
    check the HF markers that distinguish it from a native config."""
    from panoengine.train.rl.config_registry import (
        rl_async_inference_hf_qwen3_0_6b,
        rl_async_inference_worker_hf_qwen3_0_6b,
        rl_diloco_hf_qwen3_0_6b,
        rl_heloco_async_inference_hf_qwen3_0_6b,
        rl_heloco_async_inference_worker_hf_qwen3_0_6b,
        rl_heloco_hf_qwen3_0_6b,
    )

    for preset in (
        rl_diloco_hf_qwen3_0_6b,
        rl_heloco_hf_qwen3_0_6b,
        rl_async_inference_hf_qwen3_0_6b,
        rl_heloco_async_inference_hf_qwen3_0_6b,
        rl_async_inference_worker_hf_qwen3_0_6b,
        rl_heloco_async_inference_worker_hf_qwen3_0_6b,
    ):
        cfg = preset()
        spec = cfg.model_spec
        assert spec.name == "hf_transformers_rl", preset.__name__
        assert spec.state_dict_adapter is not None, preset.__name__
        assert cfg.renderer.name == "auto", preset.__name__
        assert cfg.hf_assets_path.endswith("Qwen3-0.6B"), preset.__name__
        assert spec.model.tie_word_embeddings is False, preset.__name__


def test_hf_backend_1_7b_flavor_registered():
    """The Qwen3-1.7B HF flavor resolves dims from ITS checkpoint config (not
    0.6B's): 28 layers, hidden 2048, and the shared example_checkpoint dir."""
    cfg = base_rl_config(model="hf", flavor="Qwen3-1.7B")
    assert cfg.hf_assets_path.endswith("Qwen3-1.7B")
    assert cfg.model_spec.model.num_hidden_layers == 28
    assert cfg.model_spec.model.hidden_size == 2048
    assert ("hf", "Qwen3-1.7B") in _DEFAULT_HF_ASSETS_PATH


def test_gpu_memory_limit_is_not_pinned_by_the_presets():
    """The KV budget must stay at the engine's own default, and removing the
    old pin must not shift `hf_assets_path`.

    These presets used to force `gpu_memory_limit=0.35` for shared dev boxes.
    Every generator actor already owns a disjoint GPU slice, so on a
    provisioned island that just threw the card away (measured on a decoupled
    H100 generator: 9.4 GiB used of 79, vLLM capped at ~3.2 concurrent
    full-length requests, ~68 GiB idle) -- and because it was a preset KWARG,
    ConfigManager could not reach it from argv, so no spec could raise it.

    The removed pin was also `base_rl_config`'s FIRST POSITIONAL parameter and
    four presets forwarded it positionally, so dropping it in one place only
    would have silently bound the next argument to `hf_assets_path`. Both
    halves are asserted here.
    """
    from panoengine.train.rl.config_registry import (
        rl_heloco_async_inference_dapo_math_qwen3_0_6b,
        rl_heloco_async_inference_qwen3_0_6b,
        rl_heloco_async_inference_worker_dapo_math_qwen3_0_6b,
        rl_heloco_dapo_math_qwen3_0_6b,
        rl_heloco_qwen3_0_6b,
    )
    from torchtitan.experiments.rl.actors.generator import VLLMGenerator

    engine_default = VLLMGenerator.Config().gpu_memory_limit
    for fn in (
        base_rl_config,
        rl_heloco_qwen3_0_6b,
        rl_heloco_async_inference_qwen3_0_6b,
        rl_heloco_dapo_math_qwen3_0_6b,
        rl_heloco_async_inference_dapo_math_qwen3_0_6b,
        rl_heloco_async_inference_worker_dapo_math_qwen3_0_6b,
    ):
        cfg = fn()
        assert cfg.generator.gpu_memory_limit == engine_default, (
            f"{fn.__name__} pins gpu_memory_limit to "
            f"{cfg.generator.gpu_memory_limit}; leave it at the engine default "
            "and override per run with --generator.gpu_memory_limit"
        )
        # the positional-shift guard
        sentinel = "/tmp/sentinel-assets"
        assert fn(hf_assets_path=sentinel).hf_assets_path == sentinel, fn.__name__

    # base_rl_config's first positional arg must be hf_assets_path now
    assert base_rl_config("/tmp/sentinel-assets").hf_assets_path == "/tmp/sentinel-assets"
