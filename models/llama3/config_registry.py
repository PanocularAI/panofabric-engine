# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.config import CommConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler

from panoengine.train.config import EngineTrainer
from panoengine.train.strategies import adamw, semi_sync

from . import model_registry


def llama3_8b() -> EngineTrainer.Config:
    model = model_registry("8B", seq_len=8192)
    return EngineTrainer.Config(
        loss=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model)),
        hf_assets_path="./assets/hf/Llama-3.1-8B",
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
        model=model,
        optimizer=adamw(lr=3e-4),
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
        fault_tolerance=semi_sync(),
    )


def llama3_debugmodel() -> EngineTrainer.Config:
    model = model_registry("debugmodel", seq_len=2048)
    return EngineTrainer.Config(
        loss=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model)),
        # The forks are SIBLINGS now, not submodules (no ./torchtitan here).
        # Only a bare local run uses this; the launcher always passes
        # --hf_assets_path explicitly.
        hf_assets_path="../torchtitan/tests/assets/tokenizer",
        dump_folder="./outputs",
        profiler=Profiler.Config(
            enable_profiling=True,
            save_traces_folder="profile_trace",
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=False,
            save_tb_folder="tb",
            enable_wandb=False,
        ),
        model=model,
        optimizer=adamw(lr=8e-4),
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
        fault_tolerance=semi_sync(),
    )
