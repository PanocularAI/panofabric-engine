# Copyright (c) Panocular AI
# All rights reserved.
#

"""Config-as-code preset: tool-calling SFT of a Qwen3 model.

The engine's contract is one zero-arg function per preset returning a fully
formed trainer Config. This one starts from the engine's `hf_finetune` (real
pretrained safetensors, full-parameter, HF-architecture backend) and changes
exactly three things:

  * the dataloader        -> ToolChatDataLoader (multi-span assistant masking)
  * the tokenizer         -> HFBackendTokenizer, which renders the repo's own
                             chat template (the base tokenizer leaves the
                             template's bos_token/eos_token variables empty)
  * attn_mask_type        -> "block_causal", so packed conversations cannot
                             attend across each other

Everything else -- FSDP, checkpointing, the torchft trainer, the state-dict
adapter that loads a tied checkpoint into the untied model -- is inherited.
"""

from pathlib import Path

from models.hf_transformers import _untied_flavor, model_registry
from models.hf_transformers.config_registry import HFFTConfig, hf_finetune
from panoengine.train.strategies import adamw
from torchtitan.experiments.transformers_modeling_backend import TitanModelConfig
from torchtitan.experiments.transformers_modeling_backend.tokenizer import (
    HFBackendTokenizer,
)

from .tool_chat import ToolChatDataLoader, conversation_processor, sharegpt_processor

MODEL_REPO = "Qwen/Qwen3-4B"
DATA_FILE = Path(__file__).with_name("data.json")


def _sft_model_spec():
    """The repo's real dims (from config.json) with a packed-sequence mask.

    `_untied_flavor` is the engine's own fix for tied word embeddings: FSDP
    shards tok_embeddings and lm_head separately, so a tied weight spanning
    both groups is rejected at the first forward. Qwen3-4B is tied, so reuse
    it rather than rediscovering the failure.
    """
    spec = model_registry("full")
    spec.flavor = "sft_full"
    spec.model = _untied_flavor(TitanModelConfig(attn_mask_type="block_causal"))
    return spec


def sft_tool_qwen() -> HFFTConfig:
    config = hf_finetune()
    config.hf_model = MODEL_REPO
    config.model_spec = _sft_model_spec()
    config.tokenizer = HFBackendTokenizer.Config()
    # The HF transformers backend only supports this spmd backend today; the
    # trainer aborts at parallelize_fn otherwise (the engine's own hf_* presets
    # leave it at the torchtitan default and hit the same wall).
    config.parallelism.spmd_backend = "partial_dtensor"
    config.dataloader = ToolChatDataLoader.Config(
        # The overlay is installed next to the engine's own model packages, so
        # the data travels with the code and needs no path in the spec. Point
        # data.dataset_path at a shared filesystem path or an HF dataset id for
        # anything bigger than a demo (the upload cap is 10 MiB compressed).
        dataset_path=str(DATA_FILE),
        sample_processor=conversation_processor,
        infinite=True,
    )
    # Tool schemas cost ~130 tokens each in the system prompt; a 15-tool
    # trajectory does not fit in hf_full's 8192 either, but 4096 is enough for
    # this demo set and halves activation memory.
    config.training.seq_len = 4096
    config.training.local_batch_size = 4
    config.training.steps = 200
    # SFT on a small set: below hf_finetune's 2e-5, and warm up over ~5% of the
    # run rather than its 20 steps.
    config.optimizer = adamw(lr=1e-5, eps=1e-8)
    config.lr_scheduler.warmup_steps = 10
    config.metrics.log_freq = 5
    config.checkpoint.interval = 200
    return config


def sft_tool_qwen_debug() -> HFFTConfig:
    """Same pipeline, 2-layer random-init stand-in: shape/mask smoke test.

    The dims are built fresh rather than patched onto model_registry("debugmodel"):
    the backend snapshots which fields were EXPLICITLY set as its
    "these override config.json" list at construction, so an attribute assigned
    afterwards would not be injected.
    """
    config = sft_tool_qwen()
    spec = model_registry("debugmodel")
    spec.flavor = "sft_debugmodel"
    spec.model = _untied_flavor(
        TitanModelConfig(
            dim=256,
            n_layers=2,
            n_heads=16,
            n_kv_heads=16,
            attn_mask_type="block_causal",
        )
    )
    config.model_spec = spec
    config.checkpoint.enable = False           # nothing to load into a stand-in
    config.checkpoint.initial_load_in_hf = False
    config.training.steps = 10
    return config


def sft_tool_qwen_hermes() -> HFFTConfig:
    """Same masking, streamed from a real hub dataset instead of the demo file.

    Nothing is downloaded up front: `streaming=True` yields an IterableDataset
    that ChatDataset shards per node and buffer-shuffles. The tradeoff is
    resume -- ChatDataset can only .skip() a map-style dataset, so a restart
    replays the shard from its start rather than the exact sample.
    """
    config = sft_tool_qwen()
    config.dataloader = ToolChatDataLoader.Config(
        dataset_path="NousResearch/hermes-function-calling-v1",
        load_dataset_kwargs={"split": "train", "streaming": True},
        sample_processor=sharegpt_processor,
        infinite=True,
    )
    # Hermes rows carry the tool schemas inline in a long system prompt, so
    # they run longer than the demo set.
    config.training.seq_len = 8192
    return config
