# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.validate import Validator
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from panoengine.train.strategies import diloco
from torchtitan.experiments.torchft.optimizer import default_ft_adamw
from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer
from torchtitan.tools.profiler import Profiler

from .datasets.cifar10 import CifarDataLoader
from .model.loss import ResNetCrossEntropyLoss

from . import model_registry


def resnet18_cifar10() -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
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
        model_spec=model_registry("18"),
        optimizer=default_ft_adamw(lr=0.01),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=64,
            max_norm=1.0,
            steps=10000,
            mixed_precision_param="float32",
        ),
        dataloader=CifarDataLoader.Config(
            dataset="uoft-cs/cifar10",
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
            interval=10,
            last_save_model_only=False,
            export_dtype="float32",
        ),
        activation_checkpoint=SelectiveAC.Config(),
        fault_tolerance=diloco(num_fragments=1),
        validator=Validator.Config(
            enable=False,
            freq=5,
            steps=10,
        ),
    )


def resnet34_cifar10() -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
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
        model_spec=model_registry("34"),
        optimizer=default_ft_adamw(lr=0.01),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=64,
            max_norm=1.0,
            steps=10000,
            mixed_precision_param="float32",
        ),
        dataloader=CifarDataLoader.Config(
            dataset="uoft-cs/cifar10",
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
            interval=10,
            last_save_model_only=False,
            export_dtype="float32",
        ),
        activation_checkpoint=SelectiveAC.Config(),
        fault_tolerance=diloco(num_fragments=1),
        validator=Validator.Config(
            enable=False,
            freq=5,
            steps=10,
        ),
    )


def resnet50_cifar10() -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
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
        model_spec=model_registry("50"),
        optimizer=default_ft_adamw(lr=0.01),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=64,
            max_norm=1.0,
            steps=10000,
            mixed_precision_param="float32",
        ),
        dataloader=CifarDataLoader.Config(
            dataset="uoft-cs/cifar10",
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
            interval=10,
            last_save_model_only=False,
            export_dtype="float32",
        ),
        activation_checkpoint=SelectiveAC.Config(),
        fault_tolerance=diloco(num_fragments=1),
        validator=Validator.Config(
            enable=False,
            freq=5,
            steps=10,
        ),
    )


def resnet152_cifar10() -> FaultTolerantTrainer.Config:
    return FaultTolerantTrainer.Config(
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
        model_spec=model_registry("152"),
        optimizer=default_ft_adamw(lr=0.01),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=64,
            max_norm=1.0,
            steps=10000,
            mixed_precision_param="float32",
        ),
        dataloader=CifarDataLoader.Config(
            dataset="uoft-cs/cifar10",
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
            interval=10,
            last_save_model_only=False,
            export_dtype="float32",
        ),
        activation_checkpoint=None,
        fault_tolerance=diloco(num_fragments=1),
        validator=Validator.Config(
            enable=False,
            freq=5,
            steps=10,
        ),
    )
