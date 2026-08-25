# Copyright (c) Panocular AI.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# RL glue for the HF transformers modeling backend: a ModelSpec factory whose
# result satisfies the async-RL contract (HF weight loading via the near-identity
# state-dict adapter, titan-shaped ``layers`` view) so the RL trainer, the vLLM
# generator wrapper, and torchft consume it like a native model spec. Mirrors
# the pretraining glue pattern: compose at the ModelSpec seam, no RL-consumer
# edits.
#
# Lived in the torchtitan fork as experiments/transformers_modeling_backend/rl/.
# It moved here because it only COMPOSES that backend's public pieces --
# HFTransformerModel, the parallelize/pipeline fns, TitanModelConfig and the
# state-dict adapter -- which is the engine's side of the line (see FORK-DELTA.md).
# Nothing in the fork imported it; only panoengine.train.rl.config_registry does.
#
# TODO(async-rl, unverified): this subpackage was re-based from the old
# transformers-backend design onto the current one. Two behavioral changes are
# NOT yet runtime-validated (no GPU/checkpoint available at port time):
#   1. Attention is now the backend's flex path (document/block-causal
#      BlockMask via ``get_attention_masks``), NOT the old varlen kernel. The RL
#      trainer drives it through ``get_attention_masks`` + ``positions`` exactly
#      as the native decoder path does.
#   2. The architecture dims are injected explicitly from the checkpoint's
#      ``config.json`` (via AutoConfig) into ``TitanModelConfig`` so the config
#      is architecture-complete at registry time (the generator wrapper and the
#      ``layers`` shim read head/kv/dim off it before any trainer build). Final
#      HF-config resolution + weight loading happen when the trainer builds the
#      model. The old registry-time ``update_from_config`` call is dropped: the
#      current backend's ``update_from_config`` requires a full runtime config
#      (training/parallelism/debug), which isn't available at registry time.

from dataclasses import dataclass

from transformers import AutoConfig

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.protocols.model_spec import ModelSpec

from torchtitan.experiments.transformers_modeling_backend import (
    HFTransformerModel,
    parallelize_hf_transformers,
    pipeline_hf_transformers,
    TitanModelConfig,
)
from torchtitan.experiments.transformers_modeling_backend.state_dict_adapter import (
    HFTransformerStateDictAdapter,
)


# The RL trainer and generator read
# ``model_spec.model.layers[0].attention.{inner_attention,n_heads,n_kv_heads,
# head_dim}`` off every model spec and retarget ``attention_cfg.rope``, but
# the HF backend Config has no native layer tree. ``_titan_layers_view``
# serves those reads: attention reports the backend's flex path, dims come
# from the checkpoint's config.json (head_dim is not derivable from
# hidden_size/num_heads for e.g. Qwen3), and the rope write is absorbed (the
# HF model resolves its real rope from the checkpoint config at build).


@dataclass
class _RopeConfig:
    max_seq_len: int


@dataclass
class _AttentionConfig:
    n_heads: int
    n_kv_heads: int
    head_dim: int
    inner_attention: object
    rope: _RopeConfig


@dataclass
class _LayerConfig:
    attention: _AttentionConfig
    moe: None = None


def _titan_layers_view(hf_cfg, max_seq_len: int) -> list[_LayerConfig]:
    from torchtitan.models.common.attention import FlexAttention

    n_heads = hf_cfg.num_attention_heads
    n_kv_heads = getattr(hf_cfg, "num_key_value_heads", None) or n_heads
    head_dim = getattr(hf_cfg, "head_dim", None) or hf_cfg.hidden_size // n_heads
    # One instance per index (not [cfg] * n, which would alias every layer).
    return [
        _LayerConfig(
            attention=_AttentionConfig(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                inner_attention=FlexAttention.Config(),
                rope=_RopeConfig(max_seq_len=max_seq_len),
            )
        )
        for _ in range(hf_cfg.num_hidden_layers)
    ]


def model_registry(
    flavor: str,
    *,
    attn_backend: str = "flex",
    hf_assets_path: str = "",
    max_seq_len: int = 2048,
) -> ModelSpec:
    """RL-capable ModelSpec for an HF-architecture policy.

    The architecture comes from ``hf_assets_path`` (a local HF checkpoint dir
    with config.json + safetensors — the same dir the trainer initial-loads
    from and vLLM reads tokenizer assets from). ``flavor`` is informational.

    ``attn_backend`` is accepted for signature compatibility with the native
    registries; the HF backend always routes attention through its flex path
    (see the module TODO), so only "flex" (or the historical "varlen" alias) is
    honored.
    """
    if attn_backend not in ("flex", "varlen"):
        raise ValueError(
            f"hf backend RL uses the flex attention path; got attn_backend="
            f"{attn_backend!r}"
        )
    if not hf_assets_path:
        raise ValueError("hf backend needs hf_assets_path (local dir with config.json)")

    hf_cfg = AutoConfig.from_pretrained(hf_assets_path)
    # Inject the checkpoint's real dims so the config is architecture-complete
    # at registry time (generator wrapper + layers shim read off it) and so the
    # explicit-override re-application inside the trainer's update_from_config is
    # a no-op instead of clobbering real values.
    model_config = TitanModelConfig(
        dim=hf_cfg.hidden_size,
        n_layers=hf_cfg.num_hidden_layers,
        n_heads=hf_cfg.num_attention_heads,
        n_kv_heads=getattr(hf_cfg, "num_key_value_heads", None)
        or hf_cfg.num_attention_heads,
        vocab_size=hf_cfg.vocab_size,
        norm_eps=getattr(hf_cfg, "rms_norm_eps", 1e-6),
        rope_theta=getattr(hf_cfg, "rope_theta", 10000.0),
        max_seq_len=max_seq_len,
        # Packed RL sequences need same-document masking (causal AND same-doc).
        attn_mask_type="block_causal",
    )
    cfg = HFTransformerModel.Config(model_config=model_config)
    # AutoConfig source for the trainer-side resolve + weight load.
    cfg.hf_model = hf_assets_path
    # FSDP cannot shard one weight into two groups; both RL roles host the same
    # untied model and the adapter aliases embeddings into lm_head at load, so
    # step-0 logprobs match the tied checkpoint.
    cfg.tie_word_embeddings = False
    cfg._titan_injected_model_args["hf_model"] = hf_assets_path
    cfg._titan_injected_model_args["tie_word_embeddings"] = False
    # Titan-shaped per-layer view: the RL trainer/generator read
    # ``model.layers[0].attention.*`` off every spec (PretrainedConfig
    # instances are attribute bags, so dynamic assignment matches ``hf_model``
    # above; the backend Config's ``_replace``-based build preserves it).
    cfg.layers = _titan_layers_view(hf_cfg, max_seq_len)

    return ModelSpec(
        name="hf_transformers_rl",
        flavor=flavor,
        model=cfg,
        parallelize_fn=parallelize_hf_transformers,
        pipelining_fn=pipeline_hf_transformers,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=HFTransformerStateDictAdapter,
    )
