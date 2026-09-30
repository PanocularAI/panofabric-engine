# Config-as-code presets for FT training of HF-architecture models, mirroring
# models/llama3/config_registry.py. Selected by torchtitan's
# ConfigManager: --module models.hf_transformers --config <fn>.
# The HF repo id is NOT baked into presets — it arrives via the --hf_model
# CLI overlay (RunSpec.model.hf_model), so two presets cover all models.

from dataclasses import dataclass

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.config import CommConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from panoengine.train.config import EngineTrainer
from panoengine.train.strategies import adamw, semi_sync
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.observability.profiler import Profiler

from . import model_registry


@dataclass(kw_only=True, slots=True)
class HFFTConfig(EngineTrainer.Config):
    hf_model: str = ""
    """HuggingFace repo id (e.g. 'Qwen/Qwen2.5-7B'); architecture only, random init."""


def hf_debugmodel() -> HFFTConfig:
    return HFFTConfig(
        # Architecture donor for smoke tests; the debugmodel flavor re-applies
        # tiny dims (dim=256, 2 layers) after the HF config load.
        hf_model="Qwen/Qwen3-0.6B",
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="tests/assets/tokenizer",
        dump_folder="./outputs",
        profiler=Profiler.Config(
            enable_profiling=False,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=False,
            save_tb_folder="tb",
            enable_wandb=False,
        ),
        model=model_registry("debugmodel", seq_len=2048),
        optimizer=adamw(lr=8e-4, eps=1e-8),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * 2048,
            max_context_length=2048,
            max_norm=1.0,
            steps=100,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpointer=TorchFTCheckpointManager.Config(
            enable_ft_dataloader_checkpoints=False,
            folder="checkpoint",
            interval=10,
            last_save_model_only=False,
            export_dtype="float32",
        ),
        activation_checkpoint=SelectiveAC.Config(),
        comm=CommConfig(train_timeout_seconds=15),
        fault_tolerance=semi_sync(num_fragments=1),  # no fragment_fn: whole-model DiLoCo
    )


def hf_full() -> HFFTConfig:
    return HFFTConfig(
        hf_model="",  # required: set via --hf_model (RunSpec.model.hf_model)
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="",  # required: set via --hf_assets_path (auto-derived by the launcher)
        dump_folder="./outputs",
        profiler=Profiler.Config(
            enable_profiling=True,
            save_traces_folder="profile_trace",
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=10,
            enable_tensorboard=False,
            save_tb_folder="tb",
            enable_wandb=False,
        ),
        model=model_registry("full", seq_len=8192),
        optimizer=adamw(lr=3e-4, eps=1e-8),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * 8192,
            max_context_length=8192,
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
        checkpointer=TorchFTCheckpointManager.Config(
            enable_ft_dataloader_checkpoints=False,
            folder="checkpoint",
            interval=500,
            last_save_model_only=True,
            export_dtype="float32",
        ),
        activation_checkpoint=SelectiveAC.Config(),
        fault_tolerance=semi_sync(num_fragments=1),  # no fragment_fn: whole-model DiLoCo
    )


def hf_finetune() -> HFFTConfig:
    """Full-parameter FINE-TUNING of any dense CausalLM repo: hf_full's
    architecture (the repo's real dims from config.json) plus the repo's
    pretrained safetensors loaded at startup via the backend's state-dict
    adapter. The scheduler fetches the weights into hf_assets_path before
    launch (see panofabric.spec.runspec.hf_fetch_plan); the launcher scopes
    the checkpoint folder per run/replica like the LoRA presets."""
    # hf_full already uses the "full" flavor = the repo's real dims from
    # config.json, so fine-tuning just adds pretrained-weight loading on top.
    config = hf_full()
    # Full fine-tuning moves every weight: ~10x cooler than the pretrain
    # schedule (3e-4) or gradients shear the pretrained features off.
    config.optimizer = adamw(lr=2e-5, eps=1e-8)
    config.lr_scheduler.warmup_steps = 20
    config.enable_checkpoint = True                  # the initial HF-weight load is gated on it
    config.checkpointer.initial_load_in_hf = True   # pretrained safetensors from hf_assets_path
    return config
