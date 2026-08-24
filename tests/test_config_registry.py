# Copyright (c) Panocular AI.
#
# Registry smoke test — cheap insurance that the next torchtitan/torchft fork
# rebase (via the `update-submodules` skill) cannot silently re-break every
# recipe's `config_registry.py`, the exact failure mode that produced the
# original `experiments.ft.*` / `build_loss_fn=` drift.
#
# Recipes live in `panoengine.train.*`; `models.<name>` is a compatibility shim
# kept alive because stored run specs name it. Discovery runs against the
# CANONICAL modules (a shim's re-exported functions carry the canonical
# `__module__`, so filtering by it would find nothing), and
# test_compat_shims_resolve_presets covers the shim layer separately.
#
# For every config factory in every supported model it asserts:
#   1. the canonical `config_registry` module imports            (catches dead imports)
#   2. every zero-arg config factory constructs without raising  (catches API drift)
#   3. the result is a buildable `*.Config`                      (right return type)
#   4. it carries the fields the drift used to drop (`loss`, `model_spec`)
#   5. the FT topology invariants from the roadmap (§1.1 / §4.0) hold:
#        - data_parallel_replicate_degree == 1 when fault_tolerance.enable
#        - sync_steps % num_fragments == 0 for DiLoCo
#
# It deliberately does NOT call `.build()` — that instantiates the trainer and needs
# torch.distributed + GPUs + a lighthouse, which CI doesn't have. Construction +
# `.build` being callable is what regresses on an API change.
#
# Run:  uv run pytest tests/test_config_registry.py

from __future__ import annotations

import importlib
import inspect

import pytest

import models

REPLICATE_DEGREE_FIELD = "data_parallel_replicate_degree"

SUPPORTED_MODELS = sorted(models._supported_models)


def _config_module_name(model: str) -> str:
    """The canonical (post-move) config_registry module for a recipe."""
    return f"{models._canonical_modules[model]}.config_registry"


def _shim_module_name(model: str) -> str:
    """The legacy `models.*` path that stored run specs still name."""
    return f"models.{model}.config_registry"


def discover_config_factories(model: str):
    """Yield (fn_name, fn) for every zero-arg config factory defined in a model's
    config_registry. Imported helpers (e.g. `model_registry`, whose __module__ is
    the package, not the config_registry module) are skipped."""
    mod_name = _config_module_name(model)
    mod = importlib.import_module(mod_name)
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("_"):
            continue
        if fn.__module__ != mod_name:  # skip re-exported helpers
            continue
        required = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if required:  # config factories take no required args
            continue
        yield name, fn


def _collect_cases() -> list:
    """Parametrize cases (model, fn_name, fn). Models that fail to import are skipped
    here — their failure is reported by test_config_registry_imports instead of
    erroring collection of the whole module."""
    cases = []
    for model in SUPPORTED_MODELS:
        try:
            factories = list(discover_config_factories(model))
        except Exception:
            continue
        for fn_name, fn in factories:
            cases.append(pytest.param(model, fn_name, fn, id=f"{model}.{fn_name}"))
    return cases


CASES = _collect_cases()


@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_config_registry_imports(model: str) -> None:
    """Every supported model's config_registry imports (no dead torchtitan imports)."""
    importlib.import_module(_config_module_name(model))


@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_every_model_has_a_config(model: str) -> None:
    """Guard against a registry that lost all its factory functions."""
    assert list(discover_config_factories(model)), f"{model}: no config factories found"


@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_compat_shims_resolve_presets(model: str) -> None:
    """`models.<pkg>.config_registry` still resolves every preset, and to the SAME
    object as the canonical module.

    This is the whole contract the shims exist for: stored run specs and the
    `RunSpec.model.module` default name `models.<pkg>`, and torchtitan's
    ConfigManager reaches a preset with getattr(module, config_name). A shim that
    silently stopped re-exporting would strand every stored spec.
    """
    shim = importlib.import_module(_shim_module_name(model))
    canonical = importlib.import_module(_config_module_name(model))

    factories = dict(discover_config_factories(model))
    assert factories, f"{model}: no config factories to check the shim against"

    for fn_name, fn in factories.items():
        via_shim = getattr(shim, fn_name, None)
        assert via_shim is not None, (
            f"{_shim_module_name(model)}.{fn_name} does not resolve — the shim "
            f"stopped re-exporting it and every stored run spec naming it breaks"
        )
        assert via_shim is fn is getattr(canonical, fn_name), (
            f"{model}.{fn_name}: shim and canonical module disagree on the object"
        )


@pytest.mark.parametrize("model,fn_name,fn", CASES)
def test_config_constructs(model: str, fn_name: str, fn) -> None:
    """Every config factory constructs a well-formed, buildable Config."""
    cfg = fn()
    where = f"{model}.{fn_name}"

    assert cfg is not None, f"{where}: returned None"
    assert callable(getattr(cfg, "build", None)), (
        f"{where}: result has no callable .build() — not a *.Config"
    )

    # Fields the drift used to drop entirely.
    assert getattr(cfg, "loss", None) is not None, f"{where}: loss is unset (drift regression)"
    assert getattr(cfg, "model_spec", None) is not None, f"{where}: model_spec is unset"


@pytest.mark.parametrize("model,fn_name,fn", CASES)
def test_ft_topology_invariants(model: str, fn_name: str, fn) -> None:
    """FT topology invariants (roadmap §1.1 / §4.0) hold for every config with FT enabled."""
    cfg = fn()
    where = f"{model}.{fn_name}"

    ft = getattr(cfg, "fault_tolerance", None)
    if ft is None or not getattr(ft, "enable", False):
        pytest.skip(f"{where}: fault tolerance not enabled")

    par = getattr(cfg, "parallelism", None)
    assert par is not None, f"{where}: fault_tolerance.enable but no parallelism config"
    rep = getattr(par, REPLICATE_DEGREE_FIELD, None)
    assert rep == 1, (
        f"{where}: {REPLICATE_DEGREE_FIELD}={rep!r} with FT enabled; must be 1 "
        f"(torchft drives the replicate dim — roadmap §1.1)"
    )

    if getattr(ft, "semi_sync_method", None) == "diloco":
        sync_steps = getattr(ft, "sync_steps", None)
        num_fragments = getattr(ft, "num_fragments", None)
        assert sync_steps and num_fragments and sync_steps % num_fragments == 0, (
            f"{where}: sync_steps={sync_steps!r} not divisible by "
            f"num_fragments={num_fragments!r} (DiLoCo invariant)"
        )

