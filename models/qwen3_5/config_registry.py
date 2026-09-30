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

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler

from panoengine.train.config import EngineTrainer
from panoengine.train.strategies import adamw, semi_sync

from . import model_registry


def qwen35_debugmodel() -> EngineTrainer.Config:
    """Tiny Qwen3.5 for smoke tests: exercises the hybrid GatedDeltaNet +
    full-attention stack (and therefore the FLA kernels) without a real GPU
    budget. Run this before a 9B to prove the stack imports and steps."""
    model = model_registry("debugmodel", seq_len=1024)
    return EngineTrainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model),
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
        model=model,
        optimizer=adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * 1024,
            max_context_length=1024,
            max_norm=1.0,
            steps=10,
            # Required, not a preference: GatedDeltaNet consumes per-batch
            # document offsets (cu_seqlens), so the forward's auxiliary inputs
            # change shape from step to step and graph capture aborts with
            # "CUDA graph auxiliary input structure must remain constant".
            disable_cuda_graphs=True,
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
            interval=50,
            last_save_model_only=False,
            export_dtype="float16",
        ),
        activation_checkpoint=FullAC.Config(),
        fault_tolerance=semi_sync(),
    )


def qwen35_9b() -> EngineTrainer.Config:
    """Qwen3.5-9B (dense decoder + vision tower) on text.

    Parallelism and activation checkpointing mirror torchtitan's own
    ``qwen35_9b``: TP=2 with full AC.

    Trains from scratch as written. To fine-tune from published weights, set
    ``model.init_from`` on the run spec (e.g. ``Qwen/Qwen3.5-9B``): the control
    plane fetches that repo's safetensors and emits the ``--enable_checkpoint``
    and ``--checkpointer.*`` flags that load them, so this one preset serves both.
    """
    model = model_registry("9B", seq_len=4096)
    return EngineTrainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model),
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
        model=model,
        optimizer=adamw(lr=5e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * 4096,
            max_context_length=4096,
            max_norm=1.0,
            steps=1000,
            disable_cuda_graphs=True,   # see qwen35_debugmodel
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4"]),
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=2,
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
