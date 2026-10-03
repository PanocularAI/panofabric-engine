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


def test_gated_deltanet_params_are_initialised():
    from models.qwen3_5.config_registry import qwen35_debugmodel

    cfg = qwen35_debugmodel()
    with torch.device("meta"):
        model = cfg.model.build()
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
    carry the _fragment hook, or cross-site HeLoCo has nothing to fragment."""
    from models.qwen3_5 import VerifyingQwen35StateDictAdapter
    from models.qwen3_5.config_registry import qwen35_9b

    cfg = qwen35_9b()
    processor = type(cfg.dataloader.dataset.dataset.processor).__qualname__
    assert "TextProcessor" in processor, processor
    model_cls = type(cfg.model)._owner
    assert model_cls._fragment is not None
    assert model_cls.state_dict_adapter_cls is VerifyingQwen35StateDictAdapter


# --------------------------------------------------------------------------- #
# Partial-load guard. The base adapter drops unrecognised checkpoint keys with a
# bare `continue`, so a mapping gap leaves parameters at random init and the run
# still looks healthy. VerifyingQwen35StateDictAdapter turns that into an error.
# --------------------------------------------------------------------------- #
def _adapter_and_roundtrip():
    """A debugmodel adapter plus an HF state dict that should load completely."""
    from models.qwen3_5 import VerifyingQwen35StateDictAdapter
    from models.qwen3_5.config_registry import qwen35_debugmodel

    config = qwen35_debugmodel().model
    adapter = VerifyingQwen35StateDictAdapter(config, hf_assets_path=None)
    with torch.device("meta"):
        model = config.build()
    # to_hf gives us exactly the checkpoint this model would produce, which is
    # the checkpoint it must be able to read back.
    hf = adapter.to_hf({n: p for n, p in model.state_dict().items()})
    return adapter, hf


def test_complete_checkpoint_loads_without_complaint():
    adapter, hf = _adapter_and_roundtrip()
    out = adapter.from_hf(dict(hf))
    assert out, "round-trip produced nothing"


def test_dropped_decoder_weight_is_caught():
    """Simulate the mapping gap: drop a GatedDeltaNet tensor the model needs.

    A linear_attn weight is the apt victim -- those hybrid layers are exactly
    what the lookup table is most likely to miss on a checkpoint revision."""
    adapter, hf = _adapter_and_roundtrip()
    victim = next(k for k in hf if "linear_attn.out_proj" in k)
    broken = {k: v for k, v in hf.items() if k != victim}
    with pytest.raises(ValueError, match="would train from random init"):
        adapter.from_hf(broken)


def test_generator_side_load_skips_the_meta_rebuild():
    """The RL generator's config has vLLM GatedDeltaNet layers, which register
    themselves with vLLM when built: the guard's meta rebuild would die on
    "Duplicate GDN layer name". It must not rebuild there (those layers add no
    parameters, and the load the guard protects is the trainer's)."""
    pytest.importorskip("vllm")
    from torchtitan.rl.model.vllm_wrapper import _replace_vllm_layer_configs

    from models.qwen3_5 import VerifyingQwen35StateDictAdapter, _is_vllm_side
    from models.qwen3_5.config_registry import qwen35_debugmodel

    config = qwen35_debugmodel().model
    assert not _is_vllm_side(config)
    vllm_config = _replace_vllm_layer_configs(config)
    assert _is_vllm_side(vllm_config)
    _, hf = _adapter_and_roundtrip()
    adapter = VerifyingQwen35StateDictAdapter(vllm_config, hf_assets_path=None)
    assert adapter.from_hf(dict(hf))
    assert adapter._expected is None  # never built


def test_missing_vision_weights_only_warn(caplog):
    """On a MULTIMODAL build, an absent tower warns rather than raises.

    Needs text_only=False explicitly: the presets are text-only now, so the
    default config has no tower and there would be nothing to report. In a real
    run DCP raises before this branch is reached (it plans from the model's
    params and finds the checkpoint short); this keeps the backstop honest.
    """
    from models.qwen3_5 import VerifyingQwen35StateDictAdapter, model_registry

    config = model_registry("debugmodel", text_only=False)
    adapter = VerifyingQwen35StateDictAdapter(config, hf_assets_path=None)
    with torch.device("meta"):
        model = config.build()
    hf = adapter.to_hf(dict(model.state_dict()))
    stripped = {k: v for k, v in hf.items() if ".visual." not in k}
    with caplog.at_level("WARNING"):
        adapter.from_hf(stripped)           # must not raise
    assert "vision-tower" in caplog.text


def test_text_only_drops_the_vision_tower_by_default():
    """A text workload should not carry the tower: it is ~0.45B of the 9B in
    params and optimizer state, and bytes on the wire at every HeLoCo boundary,
    for a module that never sees an image."""
    from models.qwen3_5 import model_registry

    with torch.device("meta"):
        model = model_registry("debugmodel").build()
    names = [n for n, _ in model.named_parameters()]
    assert not any(n.startswith("vision_encoder.") for n in names)
    assert model.vision_encoder is None


def test_text_only_false_keeps_it_for_multimodal():
    from models.qwen3_5 import model_registry

    with torch.device("meta"):
        model = model_registry("debugmodel", text_only=False).build()
    assert model.vision_encoder is not None
    assert any(n.startswith("vision_encoder.") for n, _ in model.named_parameters())


def test_text_only_model_reads_a_multimodal_checkpoint(tmp_path):
    """Every published Qwen3.5 checkpoint keys its decoder model.language_model.*
    (it ships a vision tower). Upstream keys a tower-less model as a text-only
    checkpoint (model.*), which asks the published 9B for 426 of its 427 decoder
    keys under names that do not exist. The verifying adapter keys it the way the
    checkpoint on disk does."""
    import json

    from models.qwen3_5 import model_registry, VerifyingQwen35StateDictAdapter

    config = model_registry("debugmodel")  # text-only
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"model.language_model.norm.weight": "model-00001-of-00002.safetensors"}}
    ))
    with torch.device("meta"):
        model = config.build()
    sd = dict(model.state_dict())
    hf = VerifyingQwen35StateDictAdapter(config, hf_assets_path=str(tmp_path)).to_hf(sd)
    decoder = [k for k in hf if k != "lm_head.weight"]
    assert decoder and all(k.startswith("model.language_model.") for k in decoder)
    # no index / a text-only checkpoint keeps upstream's model.* keys
    hf = VerifyingQwen35StateDictAdapter(config, hf_assets_path=None).to_hf(sd)
    assert not any(k.startswith("model.language_model.") for k in hf)
