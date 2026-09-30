"""FT glue for torchtitan's transformers_modeling_backend (HF-architecture
pretraining and full-parameter fine-tuning).

The backend builds any dense Llama-like HF architecture from a repo id. The
hf_debugmodel/hf_full presets train it from scratch (random init); hf_finetune
loads the repo's pretrained safetensors through the backend's near-identity
HFTransformerStateDictAdapter (native keys == "model." + HF keys) — genuine
full-parameter fine-tuning of any dense CausalLM repo. This package marries
the backend to the engine's fault-tolerant trainer, mirroring the
models.llama3 glue, and is selected from a RunSpec via
model.module: models.hf_transformers (a shim for this package).

The model has no ``_fragment`` hook, so DiLoCo syncs it whole
(``num_fragments`` must be 1). Its state-dict adapter -- near-identity HF<->DCP
keys, plus aliasing a tied checkpoint's embed_tokens into our untied lm_head at
load -- is the backend's ``HFTransformerModel.state_dict_adapter_cls``.
"""

import copy

from torchtitan.experiments.transformers_modeling_backend import (
    HFTransformerModel,
    TitanModelConfig,
)


def _untied_flavor(model_config: TitanModelConfig) -> HFTransformerModel.Config:
    """Backend Config with tied word embeddings dropped after the HF config load.

    FSDP shards tok_embeddings and lm_head into separate groups; a tied weight
    then spans two groups and torchtitan rejects it at the first forward.
    Untying makes tied-embedding repos (most small models) launchable at all;
    for fine-tuning, the state-dict adapter aliases the tied checkpoint's
    embed_tokens into our untied lm_head at load. `_titan_injected_model_args`
    is the backend's own "these values win over the repo's config.json"
    mechanism, re-applied inside update_from_config.

    REAL-DIMS vs OVERRIDE is now per-FIELD, not a flag: the backend's
    _initialize_attributes records only the fields EXPLICITLY set on
    `model_config` as injected args, so a bare TitanModelConfig() lets the
    repo's config.json govern the architecture entirely (what
    inject_titan_dims=False used to mean), while a config that sets dims
    overrides exactly those.
    """
    cfg = HFTransformerModel.Config(model_config)
    cfg._titan_injected_model_args["tie_word_embeddings"] = False
    return cfg


flavors = {
    # debugmodel: tiny dims that intentionally OVERRIDE the repo (smoke tests,
    # random init — the whole point is a 2-layer stand-in for any architecture).
    "debugmodel": _untied_flavor(
        TitanModelConfig(
            dim=256,
            n_layers=2,
            n_heads=16,
            n_kv_heads=16,
        ),
    ),
    # full: the repo's OWN dims from config.json (from-scratch pretraining, and
    # the base for hf_finetune's pretrained-weight loading). Setting NO fields
    # is what makes the repo's config govern.
    "full": _untied_flavor(TitanModelConfig()),
}


def model_registry(flavor: str, *, seq_len: int) -> HFTransformerModel.Config:
    """flavor: "debugmodel" (tiny override dims, smoke test) | "full" (the repo's
    real dims from config.json — from-scratch pretraining or, with a
    weight-loading preset, fine-tuning)."""
    config = copy.deepcopy(flavors[flavor])
    config.max_seq_len = seq_len
    return config
