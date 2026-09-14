# Copyright (c) Panocular AI.
#
# Guards the bug that made Qwen3.5 untrainable on the platform.
#
# Qwen3.5 interleaves GatedDeltaNet linear-attention layers with full attention.
# Run through `models.hf_transformers`, its `A_log` / `dt_bias` fall through every
# branch of that backend's class-name-keyed `_init_weights`, so after `to_empty()`
# they hold uninitialised memory and `-exp(A_log)` overflows to inf on step 1 --
# a NaN with no log line, which is how a customer lost days to it.
#
# The native `models.qwen3_5` path inits them explicitly. This asserts that, the
# same way the trainer does it: build on meta, `to_empty()` (so anything init
# forgets keeps garbage), then `init_weights`.
#
# Run:  uv run pytest tests/test_qwen3_5_init.py

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fla", reason="Qwen3.5 needs flash-linear-attention")


def test_gated_deltanet_params_are_initialised():
    from models.qwen3_5.config_registry import qwen35_debugmodel

    cfg = qwen35_debugmodel()
    with torch.device("meta"):
        model = cfg.model_spec.model.build()
    model.to_empty(device="cpu")
    with torch.no_grad():
        model.init_weights(buffer_device=torch.device("cpu"))

    gdn = {n: p for n, p in model.named_parameters()
           if n.endswith(("A_log", "dt_bias"))}
    assert gdn, "no GatedDeltaNet params found -- is this still the hybrid model?"

    for name, p in gdn.items():
        assert torch.isfinite(p).all(), f"{name} left uninitialised after to_empty()"
        if name.endswith("A_log"):
            # The actual crash: the gate computes -exp(A_log).
            assert torch.isfinite(-torch.exp(p.float())).all(), f"-exp({name}) overflowed"


def test_preset_is_text_only_and_fault_tolerant():
    """The presets must feed text (not torchtitan's cc12m image-text loader) and
    carry fragment_fn, or cross-site HeLoCo has nothing to fragment."""
    from models.qwen3_5.config_registry import qwen35_9b

    cfg = qwen35_9b()
    assert "Text" in type(cfg.dataloader).__qualname__, type(cfg.dataloader).__qualname__
    assert cfg.model_spec.fragment_fn is not None
    assert cfg.model_spec.state_dict_adapter is not None
