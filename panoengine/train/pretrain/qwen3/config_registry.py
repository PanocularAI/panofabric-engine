# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.validate import Validator
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from panoengine.train.strategies import diloco
from torchtitan.experiments.torchft.optimizer import default_ft_adamw
from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.profiler import Profiler

from . import model_registry


def qwen3_0_6b() -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        loss=CrossEntropyLoss.Config(),
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
        model_spec=model_registry("0.6B"),
        optimizer=default_ft_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            max_norm=1.0,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=TorchFTCheckpointManager.Config(
            enable=False,
            enable_ft_dataloader_checkpoints=False,
            folder="checkpoint",
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
        fault_tolerance=diloco(),
        validator=Validator.Config(
            enable=False,
        ),
    )


def qwen3_1_7b() -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        loss=CrossEntropyLoss.Config(),
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
        model_spec=model_registry("1.7B"),
        optimizer=default_ft_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=20,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            max_norm=1.0,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=TorchFTCheckpointManager.Config(
            enable=False,
            enable_ft_dataloader_checkpoints=False,
            folder="checkpoint",
            interval=50,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
        fault_tolerance=diloco(),
        validator=Validator.Config(
            enable=False,
        ),
    )


def qwen3_32b() -> FaultTolerantTrainer.Config:
    # Preset for 8 H100 GPUs with 96 GiB memory
    return FaultTolerantTrainer.Config(
        loss=CrossEntropyLoss.Config(),
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
        model_spec=model_registry("32B"),
        optimizer=default_ft_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=600,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            max_norm=1.0,
            steps=3000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=TorchFTCheckpointManager.Config(
            enable=False,
            enable_ft_dataloader_checkpoints=False,
            folder="checkpoint",
            interval=500,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=FullAC.Config(),
        fault_tolerance=diloco(),
        validator=Validator.Config(
            enable=False,
        ),
    )


def qwen3_moe_debug() -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
        loss=CrossEntropyLoss.Config(),
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
        model_spec=model_registry("debugmodel_moe"),
        optimizer=default_ft_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
        ),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            max_norm=1.0,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
            expert_parallel_degree=1,
        ),
        checkpoint=TorchFTCheckpointManager.Config(
            enable=False,
            enable_ft_dataloader_checkpoints=False,
            folder="checkpoint",
            interval=10,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=SelectiveAC.Config(),
        fault_tolerance=diloco(),
        validator=Validator.Config(
            enable=False,
        ),
    )
