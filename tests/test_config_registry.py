# Copyright (c) Panocular AI.
#
# Registry smoke test — cheap insurance that the next torchtitan/torchft fork
# rebase (via the `update-submodules` skill) cannot silently re-break every
# recipe's `config_registry.py`, the exact failure mode that produced the
# original `experiments.ft.*` / `build_loss_fn=` drift.
#
# Recipes live in `models.<name>` — the path the control plane stores and
# torchtitan's ConfigManager resolves. (They used to live in `panoengine.train.*`
# with `models/` re-exporting them; that indirection is gone.)
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
    """The config_registry module for a recipe."""
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
def test_config_manager_resolves_the_stored_module_path(model: str) -> None:
    """`--module models.<pkg>` resolves every preset through torchtitan's own
    ConfigManager.

    This is the contract that matters: `models.<pkg>` is the `RunSpec.model.module`
    default and the value baked into every run spec already in the control plane's
    database, and ConfigManager reaches a preset with getattr(module, config_name).
    A recipe that moved without this path following it strands every stored spec.
    """
    from torchtitan.config.manager import _import_registry

    registry = _import_registry(f"models.{model}.config_registry")
    assert registry is not None, (
        f"models.{model}.config_registry does not import — every stored run spec "
        f"naming it breaks"
    )

    factories = dict(discover_config_factories(model))
    assert factories, f"{model}: no config factories to resolve"
    for fn_name, fn in factories.items():
        assert getattr(registry, fn_name, None) is fn, (
            f"models.{model}.{fn_name} is not reachable via getattr on the module "
            f"ConfigManager imports"
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



# --- panoengine.train.strategies.semi_sync ----------------------------------
#
# The one place a recipe states its decentralized strategy. It used to be named
# `diloco()`, which read as though the preset picked DiLoCo; `method` is a real
# parameter, so these pin that it stays one.


def test_semi_sync_defaults_to_diloco_and_accepts_the_others() -> None:
    from panoengine.train.strategies import SEMI_SYNC_METHODS, semi_sync

    assert semi_sync().semi_sync_method == "diloco"
    for method in SEMI_SYNC_METHODS:
        assert semi_sync(method=method).semi_sync_method == method

    # the whole point of the rename: a preset CAN pin heloco, and the rest of
    # the shared block comes along unchanged.
    ft = semi_sync(method="heloco", num_fragments=1)
    assert ft.enable and ft.num_fragments == 1 and ft.process_group == "gloo"

    # arbitrary FaultTolerance fields still ride through **overrides
    assert semi_sync(sync_steps=40).sync_steps == 40


def test_semi_sync_rejects_an_unknown_method() -> None:
    """A typo must fail here, not silently reach `maybe_semi_sync_training`
    and raise "Unknown training method" after the cluster is already up."""
    from panoengine.train.strategies import semi_sync

    with pytest.raises(ValueError, match="semi_sync_method must be one of"):
        semi_sync(method="dilcoo")
    # controld's strategy axis has "solo", but it means "no FT block at all",
    # not a semi_sync_method value.
    with pytest.raises(ValueError, match="semi_sync_method must be one of"):
        semi_sync(method="solo")


def test_semi_sync_method_has_exactly_one_spelling() -> None:
    """`semi_sync_method=` would otherwise ride through **overrides and win the
    dict update, silently running a different algorithm than `method=` asked for."""
    from panoengine.train.strategies import semi_sync

    with pytest.raises(TypeError, match="named parameter"):
        semi_sync(semi_sync_method="heloco")
    with pytest.raises(TypeError, match="named parameter"):
        semi_sync(method="diloco", semi_sync_method="heloco")
