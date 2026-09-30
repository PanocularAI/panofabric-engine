# Config-as-code presets for LoRA fine-tuning, mirroring the
# panoengine.train.* FT glue. Selected by torchtitan's ConfigManager:
# --module models.lora --config <fn> (models.lora shims this package).
#
# What makes these FINE-TUNING presets rather than pretraining:
#   * checkpointer.initial_load_in_hf=True loads the repo's pretrained
#     safetensors from hf_assets_path at startup (enable_checkpoint must be
#     True for that load path to run at all). The launcher gives each run its
#     own checkpoint folder so a finished run's checkpoint never shadows the
#     next run's pretrained-weight load (see spec/launch.py).
#   * LoRAFTConfig.build() applies torchtitan's LoRATransform AFTER tyro CLI
#     overrides land, making lora_rank/lora_alpha/lora_target_modules runtime
#     knobs — converters are otherwise baked in at preset-construction time,
#     unreachable from argv.
#   * Embedding configs are frozen too: the converter only freezes Linears,
#     and with weight tying (qwen3) the embedding IS the output head, so a
#     trainable embedding would silently unfreeze the head.
#
# The HF checkpoint has no lora_a/lora_b keys, and none are needed: the
# state-dict adapter skips unmapped keys and the model load is strict=False,
# so adapters keep their init (lora_b == 0 => the first forward is exactly
# the pretrained model).

from dataclasses import dataclass

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import LinearLoRAHandler, LoRATransform
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.protocols.module import Module

from panoengine.train.config import EngineTrainer
from panoengine.train.strategies import adamw, semi_sync

# All per-layer linears: attention wqkv/wo + MLP w13/w2 (upstream fuses q/k/v
# into one wqkv and gate/up into one w13). Deliberately excludes the output
# head and embeddings: adapters there produce non-layer state-dict keys the HF
# adapters can't map, and freezing them is standard LoRA practice anyway.
DEFAULT_TARGET_MODULES = "wqkv,wo,w13,w2"

#: Names that stopped existing when upstream fused the projections.
_FUSED_INTO = {"wq": "wqkv", "wk": "wqkv", "wv": "wqkv", "wkv": "wqkv", "w1": "w13", "w3": "w13"}


def apply_lora(config: "LoRAFTConfig") -> None:
    """Rewrite config.model: LoRA-adapt the target Linears and freeze
    everything else (other Linears, embeddings, norms) via torchtitan's
    LoRATransform, so ONLY the adapters train (PEFT semantics).
    Module-level so tests can exercise the conversion without building a trainer."""
    targets = [t.strip() for t in config.lora_target_modules.split(",") if t.strip()]
    if not targets:
        raise ValueError(
            "lora_target_modules must name at least one Linear config "
            f"(comma-separated), e.g. {DEFAULT_TARGET_MODULES!r}"
        )
    # No freeze_* arguments: the converter now freezes by CONSTRUCTION, rewriting
    # every non-target module config into a frozen subclass, so only adapter
    # parameters ever train. The old freeze_embeddings/freeze_norms flags are gone
    # (passing them raises TypeError), and their removal is a strengthening rather
    # than a loss — the tied-weights hazard they guarded, where a trainable
    # embedding silently unfreezes the output head, is now impossible by default.
    handler = LinearLoRAHandler()
    present = {
        fqn.rsplit(".", 1)[-1]
        for fqn, cfg, *_ in config.model.traverse(Module.Config, recurse=True)
        if isinstance(cfg, handler.config_type)
    }
    # Upstream only WARNS about a target that matches nothing; a fine-tune that
    # silently adapts fewer projections than it was asked to is not worth running.
    unknown = [t for t in targets if t not in present]
    if unknown:
        hints = {t: _FUSED_INTO[t] for t in unknown if t in _FUSED_INTO}
        raise ValueError(
            f"lora_target_modules {unknown} match no Linear in this model "
            f"(it has {sorted(present)})"
            + (f"; renamed upstream: {hints}" if hints else "")
        )
    config.model = LoRATransform(
        handlers=(handler,),
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        target_modules=targets,
    ).transform(config.model)


@dataclass(kw_only=True, slots=True)
class LoRAFTConfig(EngineTrainer.Config):
    lora_rank: int = 16
    """Rank of the low-rank adapters (8-64 typical; higher = more capacity + cost)."""

    lora_alpha: float = 32.0
    """Adapter scaling: output is scaled by alpha/rank. ~2x rank is the common default."""

    lora_target_modules: str = DEFAULT_TARGET_MODULES
    """Comma-separated Linear config names (FQN last segment) to adapt; every
    other Linear is frozen. A string (not a list) so it stays settable via a
    single --lora_target_modules=a,b,c tyro flag."""

    def build(self, **kwargs):
        apply_lora(self)
        # Explicit super(): @dataclass(slots=True) replaces the class object,
        # so zero-arg super()'s __class__ cell points at the discarded
        # pre-dataclass class and raises TypeError at runtime.
        return super(LoRAFTConfig, self).build(**kwargs)


def _qwen3_model(flavor: str, seq_len: int):
    from models.qwen3 import model_registry

    return model_registry(flavor, seq_len=seq_len)


def _llama3_model(flavor: str, seq_len: int):
    from models.llama3 import model_registry

    return model_registry(flavor, seq_len=seq_len)


def _lora_preset(
    model_fn,
    flavor: str,
    *,
    hf_assets_path: str,
    local_batch_size: int,
    seq_len: int,
    warmup_steps: int,
    log_freq: int,
) -> LoRAFTConfig:
    """One fine-tuning preset. Shared fine-tune-appropriate defaults; the
    per-model knobs (batch/seq/warmup) come from the preset table below.
    hf_assets_path matches the spec layer's derived assets dir
    (assets/hf/<repo with / -> -->); the launcher passes --hf_assets_path with
    the same value."""
    model = model_fn(flavor, seq_len)
    return LoRAFTConfig(
        loss=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model)),
        hf_assets_path=hf_assets_path,
        dump_folder="./outputs",
        profiler=Profiler.Config(
            enable_profiling=False,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=log_freq,
            enable_tensorboard=False,
            save_tb_folder="tb",
        ),
        model=model,
        # LoRA fine-tunes run hotter than full fine-tuning (only the adapters
        # move): 1e-4 vs the pretrain presets' 3e-4-per-token-budget schedules.
        optimizer=adamw(lr=1e-4, eps=1e-8),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=warmup_steps,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=local_batch_size * seq_len,
            max_context_length=seq_len,
            max_norm=1.0,
            steps=1000,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
        ),
        enable_checkpoint=True,        # required: the initial HF-weight load is gated on it
        checkpointer=TorchFTCheckpointManager.Config(
            initial_load_in_hf=True,   # pretrained safetensors from hf_assets_path
            enable_ft_dataloader_checkpoints=False,
            folder="checkpoint",       # launcher scopes to checkpoint/<run_id>/replica-<i>
            interval=500,
            last_save_model_only=True,
            export_dtype="float32",
        ),
        activation_checkpoint=SelectiveAC.Config(),
        fault_tolerance=semi_sync(),
    )


# Preset functions must match LORA_PRESET_REPOS (the panofabric control plane spec layer)
# 1:1 — the spec layer validates submissions against that map and fetches the
# repo's weights into the derived assets dir before launch.

def lora_qwen3_0_6b() -> LoRAFTConfig:
    """Fine-tune Qwen/Qwen3-0.6B (ungated; small enough for a 1-GPU island)."""
    return _lora_preset(
        _qwen3_model, "0.6B",
        hf_assets_path="assets/hf/Qwen--Qwen3-0.6B",
        local_batch_size=4, seq_len=2048, warmup_steps=10, log_freq=1,
    )


def lora_qwen3_1_7b() -> LoRAFTConfig:
    """Fine-tune Qwen/Qwen3-1.7B (ungated)."""
    return _lora_preset(
        _qwen3_model, "1.7B",
        hf_assets_path="assets/hf/Qwen--Qwen3-1.7B",
        local_batch_size=4, seq_len=4096, warmup_steps=20, log_freq=10,
    )


def lora_qwen3_4b() -> LoRAFTConfig:
    """Fine-tune Qwen/Qwen3-4B (ungated)."""
    return _lora_preset(
        _qwen3_model, "4B",
        hf_assets_path="assets/hf/Qwen--Qwen3-4B",
        local_batch_size=2, seq_len=4096, warmup_steps=20, log_freq=10,
    )


def lora_qwen3_8b() -> LoRAFTConfig:
    """Fine-tune Qwen/Qwen3-8B (ungated)."""
    return _lora_preset(
        _qwen3_model, "8B",
        hf_assets_path="assets/hf/Qwen--Qwen3-8B",
        local_batch_size=1, seq_len=4096, warmup_steps=50, log_freq=10,
    )


def lora_llama3_1b() -> LoRAFTConfig:
    """Fine-tune meta-llama/Llama-3.2-1B (gated repo: the daemon needs HF_TOKEN)."""
    return _lora_preset(
        _llama3_model, "1B",
        hf_assets_path="assets/hf/meta-llama--Llama-3.2-1B",
        local_batch_size=4, seq_len=4096, warmup_steps=10, log_freq=10,
    )


def lora_llama3_3b() -> LoRAFTConfig:
    """Fine-tune meta-llama/Llama-3.2-3B (gated repo: the daemon needs HF_TOKEN)."""
    return _lora_preset(
        _llama3_model, "3B",
        hf_assets_path="assets/hf/meta-llama--Llama-3.2-3B",
        local_batch_size=2, seq_len=4096, warmup_steps=20, log_freq=10,
    )


def lora_llama3_8b() -> LoRAFTConfig:
    """Fine-tune meta-llama/Llama-3.1-8B (gated repo: the daemon needs HF_TOKEN)."""
    return _lora_preset(
        _llama3_model, "8B",
        hf_assets_path="assets/hf/meta-llama--Llama-3.1-8B",
        local_batch_size=1, seq_len=8192, warmup_steps=50, log_freq=10,
    )
