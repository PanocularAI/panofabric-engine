# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Text-only Qwen3.5 presets.

torchtitan's own ``models/qwen3_5/config_registry.py`` is vision-first: it wires
an ``MMDataLoader`` over cc12m and a ``MultiModalTokenizer``, so importing it
drags in torchvision and trains on image-text pairs. These presets keep the same
model (vision tower included -- it is part of the published checkpoint) but feed
it text, which the decoder's forward supports directly: ``pixel_values`` is
optional and RoPE falls back to plain ``positions`` when no MRoPE positions
arrive.

Run as:  --module models.qwen3_5 --config qwen35_9b
"""

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.components.validate import Validator
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.tools.profiler import Profiler

from panoengine.train.strategies import adamw, semi_sync

from . import model_registry


def qwen35_debugmodel() -> FaultTolerantTrainer.Config:
    """Tiny Qwen3.5 for smoke tests: exercises the hybrid GatedDeltaNet +
    full-attention stack (and therefore the FLA kernels) without a real GPU
    budget. Run this before a 9B to prove the stack imports and steps."""
    model_spec = model_registry("debugmodel")
    return FaultTolerantTrainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
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
        model_spec=model_spec,
        optimizer=adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=1024,
            max_norm=1.0,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
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
        activation_checkpoint=FullAC.Config(),
        fault_tolerance=semi_sync(),
        validator=Validator.Config(enable=False),
    )


def qwen35_9b() -> FaultTolerantTrainer.Config:
    """Qwen3.5-9B (dense decoder + vision tower) on text.

    Parallelism and activation checkpointing mirror torchtitan's own
    ``qwen35_9b``: TP=2 with full AC.

    Trains from scratch as written. To fine-tune from published weights, set
    ``model.init_from`` on the run spec (e.g. ``Qwen/Qwen3.5-9B``): the control
    plane fetches that repo's safetensors and emits the ``--checkpoint`` flags
    that load them, so this one preset serves both.
    """
    model_spec = model_registry("9B")
    return FaultTolerantTrainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./assets/hf/Qwen3.5-9B",
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
        model_spec=model_spec,
        optimizer=adamw(lr=5e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            max_norm=1.0,
            steps=1000,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=2,
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
        fault_tolerance=semi_sync(),
        validator=Validator.Config(enable=False),
    )
