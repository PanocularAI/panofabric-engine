# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.models.common.config_utils import decoder_vocab_size
from panoengine.train.config import EngineTrainer
from panoengine.train.strategies import adamw, semi_sync
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.observability.profiler import Profiler

from . import model_registry


def qwen3_0_6b() -> EngineTrainer.Config:
    model = model_registry("0.6B", seq_len=4096)
    return EngineTrainer.Config(
        loss=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model)),
        hf_assets_path="./assets/hf/Qwen3-0.6B",
        dump_folder="./outputs",
        profiler=Profiler.Config(
            enable_profiling=False,
            save_traces_folder="profile_trace",
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=False,
            save_tb_folder="tb",
        ),
        model=model,
        optimizer=adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * 4096,
            max_context_length=4096,
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
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
        fault_tolerance=semi_sync(),
    )


def qwen3_1_7b() -> EngineTrainer.Config:
    model = model_registry("1.7B", seq_len=4096)
    return EngineTrainer.Config(
        loss=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model)),
        hf_assets_path="./assets/hf/Qwen3-1.7B",
        dump_folder="./outputs",
        profiler=Profiler.Config(
            enable_profiling=False,
            save_traces_folder="profile_trace",
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=10,
            enable_tensorboard=False,
            save_tb_folder="tb",
        ),
        model=model,
        optimizer=adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=20,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * 4096,
            max_context_length=4096,
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
            interval=50,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
        fault_tolerance=semi_sync(),
    )


def qwen3_32b() -> EngineTrainer.Config:
    # Preset for 8 H100 GPUs with 96 GiB memory
    model = model_registry("32B", seq_len=4096)
    return EngineTrainer.Config(
        loss=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model)),
        hf_assets_path="./assets/hf/Qwen3-32B",
        dump_folder="./outputs",
        profiler=Profiler.Config(
            enable_profiling=False,
            save_traces_folder="profile_trace",
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=10,
            enable_tensorboard=False,
            save_tb_folder="tb",
        ),
        model=model,
        optimizer=adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=600,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=2 * 4096,
            max_context_length=4096,
            max_norm=1.0,
            steps=3000,
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
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=FullAC.Config(),
        fault_tolerance=semi_sync(),
    )


def qwen3_moe_debug() -> EngineTrainer.Config:
    model = model_registry("debugmodel_moe", seq_len=4096)
    return EngineTrainer.Config(
        loss=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model)),
        hf_assets_path="./tests/assets/tokenizer",
        dump_folder="./outputs",
        profiler=Profiler.Config(
            enable_profiling=False,
            save_traces_folder="profile_trace",
            profile_freq=100,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=False,
            save_tb_folder="tb",
        ),
        model=model,
        optimizer=adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * 4096,
            max_context_length=4096,
            max_norm=1.0,
            steps=10,
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
            expert_parallel_degree=1,
        ),
        checkpointer=TorchFTCheckpointManager.Config(
            enable_ft_dataloader_checkpoints=False,
            folder="checkpoint",
            interval=10,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
        fault_tolerance=semi_sync(),
    )
