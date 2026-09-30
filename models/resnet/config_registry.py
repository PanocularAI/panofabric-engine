# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler

from panoengine.train.config import EngineTrainer
from panoengine.train.strategies import adamw, semi_sync

from .datasets.cifar10 import cifar10_dataloader
from .model.loss import ResNetCrossEntropyLoss

from . import model_registry


def resnet18_cifar10() -> EngineTrainer.Config:
    return EngineTrainer.Config(
        loss=ResNetCrossEntropyLoss.Config(),
        hf_assets_path="",
        tokenizer=None,
        dump_folder="./outputs",
        profiler=Profiler.Config(
            enable_profiling=False,
            save_traces_folder="profile_trace",
            profile_freq=10,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=False,
            save_tb_folder="tb",
            enable_wandb=False,
        ),
        model=model_registry("18"),
        optimizer=adamw(lr=0.01),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=64,  # one token per image
            max_context_length=32,  # CIFAR-10 image side; FLOPs are per 32x32 image
            max_norm=1.0,
            steps=10000,
            mixed_precision_param="float32",
        ),
        dataloader=cifar10_dataloader(),
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
        fault_tolerance=semi_sync(num_fragments=1),
    )


def resnet34_cifar10() -> EngineTrainer.Config:
    return EngineTrainer.Config(
        loss=ResNetCrossEntropyLoss.Config(),
        hf_assets_path="",
        tokenizer=None,
        dump_folder="./outputs",
        profiler=Profiler.Config(
            enable_profiling=False,
            save_traces_folder="profile_trace",
            profile_freq=10,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=False,
            save_tb_folder="tb",
            enable_wandb=False,
        ),
        model=model_registry("34"),
        optimizer=adamw(lr=0.01),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=64,  # one token per image
            max_context_length=32,  # CIFAR-10 image side; FLOPs are per 32x32 image
            max_norm=1.0,
            steps=10000,
            mixed_precision_param="float32",
        ),
        dataloader=cifar10_dataloader(),
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
        fault_tolerance=semi_sync(num_fragments=1),
    )


def resnet50_cifar10() -> EngineTrainer.Config:
    return EngineTrainer.Config(
        loss=ResNetCrossEntropyLoss.Config(),
        hf_assets_path="",
        tokenizer=None,
        dump_folder="./outputs",
        profiler=Profiler.Config(
            enable_profiling=False,
            save_traces_folder="profile_trace",
            profile_freq=10,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=False,
            save_tb_folder="tb",
            enable_wandb=False,
        ),
        model=model_registry("50"),
        optimizer=adamw(lr=0.01),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=64,  # one token per image
            max_context_length=32,  # CIFAR-10 image side; FLOPs are per 32x32 image
            max_norm=1.0,
            steps=10000,
            mixed_precision_param="float32",
        ),
        dataloader=cifar10_dataloader(),
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
        fault_tolerance=semi_sync(num_fragments=1),
    )


def resnet152_cifar10() -> EngineTrainer.Config:
    return EngineTrainer.Config(
        loss=ResNetCrossEntropyLoss.Config(),
        hf_assets_path="",
        tokenizer=None,
        dump_folder="./outputs",
        profiler=Profiler.Config(
            enable_profiling=False,
            save_traces_folder="profile_trace",
            profile_freq=10,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=False,
            save_tb_folder="tb",
            enable_wandb=False,
        ),
        model=model_registry("152"),
        optimizer=adamw(lr=0.01),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=64,  # one token per image
            max_context_length=32,  # CIFAR-10 image side; FLOPs are per 32x32 image
            max_norm=1.0,
            steps=10000,
            mixed_precision_param="float32",
        ),
        dataloader=cifar10_dataloader(),
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
        activation_checkpoint=None,
        fault_tolerance=semi_sync(num_fragments=1),
    )
