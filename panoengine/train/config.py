"""The engine's trainer: torchtitan's FaultTolerantTrainer plus the knobs controld sets.

Upstream torchtitan configures data and optimizers in Python only. GrainDataLoader's
``dataset`` and OptimizersContainer's ``optimizers`` are ``tyro.conf.Suppress``-ed, and a
checkpointer is on exactly when it is not None. The control plane still has to choose a
dataset, a learning rate and checkpointing per run from the command line, so the engine
owns those four flags:

    --dataset=<DATASETS key>   --dataset_path=<dir | HF repo>
    --lr=<float>               --enable_checkpoint

It also keeps the control plane's batch shape -- sequences of ``seq_len`` tokens -- which
upstream replaced with token counts:

    --local_batch_size=<seqs>  --global_batch_size=<seqs>  --seq_len=<tokens>

Only the engine can translate those, because a batch in tokens is ``batch x seq_len`` and
the control plane does not know each preset's own ``seq_len``.

They are applied in ``EngineTrainer.__init__``, i.e. after tyro has parsed the CLI --
not in ``Config.__post_init__``, which also runs on the preset itself: nulling the
checkpointer there would remove every ``--checkpointer.*`` flag from the CLI.

Every recipe under ``models/`` returns an ``EngineTrainer.Config`` (or a subclass).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Annotated

import tyro

from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.protocols.module import Module

__all__ = ["EngineTrainer", "apply_cli_knobs", "fault_tolerant"]


class EngineTrainer(FaultTolerantTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(FaultTolerantTrainer.Config):
        dataset: str | None = None
        """A torchtitan ``DATASETS`` key (``c4``, ``c4_test``, ...) replacing the
        preset's dataset. Its packing is kept."""

        dataset_path: str | None = None
        """Where the dataset lives: a local directory or an HF dataset repo id.
        Replaces the dataset source's ``path`` (and its ``data_files``)."""

        lr: float | None = None
        """Learning rate for every optimizer the preset builds."""

        enable_checkpoint: bool = False
        """Checkpointing is on only when set; the preset's ``checkpointer`` block
        supplies everything else (folder, interval, HF load)."""

        local_batch_size: int | None = None
        """Sequences per microbatch per DP rank. Sets
        ``training.num_tokens_per_microbatch_per_dp_rank`` to this x ``seq_len``."""

        global_batch_size: int | None = None
        """Sequences per optimizer step over all DP ranks. Sets
        ``training.num_tokens_per_train_step`` to this x ``seq_len``."""

        seq_len: int | None = None
        """Tokens per sequence: ``training.max_context_length``. Given alone, the
        preset's batch (in sequences) is kept, so its token budget scales with it
        -- what ``training.seq_len`` used to do."""

        tokens_per_sample: Annotated[int | None, tyro.conf.Suppress] = None
        """Tokens one sample counts as, for the batch knobs above. None means
        ``seq_len`` (a text sequence); ResNet sets 1 (one image)."""

    def __init__(self, config: Config):
        super().__init__(apply_cli_knobs(config))


def apply_cli_knobs(config: EngineTrainer.Config) -> EngineTrainer.Config:
    """``config`` with the engine-owned knobs folded into the upstream fields."""
    changes = {}
    if config.dataset is not None or config.dataset_path is not None:
        changes["dataloader"] = _with_dataset(
            config.dataloader, config.dataset, config.dataset_path
        )
    if config.lr is not None:
        changes["optimizer"] = dataclasses.replace(
            config.optimizer,
            optimizers=[
                dataclasses.replace(o, lr=config.lr) for o in config.optimizer.optimizers
            ],
        )
    if not config.enable_checkpoint:
        changes["checkpointer"] = None
    training = _with_batch_shape(config)
    if training is not config.training:
        changes["training"] = training
    # A preset binds its model's context (the RoPE cache) to its own seq_len; a
    # longer run must not index past it.
    _cover_context(config.model, training.max_context_length)
    return dataclasses.replace(config, **changes) if changes else config


def _with_batch_shape(config: EngineTrainer.Config):
    """``config.training`` with the batch-of-sequences knobs turned into tokens."""
    training = config.training
    if config.local_batch_size is None and config.global_batch_size is None and (
        config.seq_len is None
    ):
        return training
    preset_len = training.max_context_length
    seq_len = config.seq_len or preset_len
    per_sample = config.tokens_per_sample or seq_len
    # The preset's own batch in samples, so --seq_len alone scales it like before.
    local = config.local_batch_size or (
        training.num_tokens_per_microbatch_per_dp_rank
        // (config.tokens_per_sample or preset_len)
    )
    changes = {
        "max_context_length": seq_len,
        "num_tokens_per_microbatch_per_dp_rank": local * per_sample,
    }
    if config.global_batch_size is not None:
        changes["num_tokens_per_train_step"] = config.global_batch_size * per_sample
    elif training.num_tokens_per_train_step != -1:
        changes["num_tokens_per_train_step"] = (
            training.num_tokens_per_train_step
            // (config.tokens_per_sample or preset_len)
            * per_sample
        )
    return dataclasses.replace(training, **changes)


def _cover_context(model_config, max_context_length: int) -> None:
    """Raise every ``max_context_length`` in the model config tree (the decoder's
    and each RoPE's) to at least ``max_context_length``, in place."""
    configs = [model_config] + [
        cfg for _, cfg, *_ in model_config.traverse(Module.Config, recurse=True)
    ] if hasattr(model_config, "traverse") else [model_config]
    for cfg in configs:
        current = getattr(cfg, "max_context_length", None)
        if isinstance(current, int) and current < max_context_length:
            cfg.max_context_length = max_context_length


def _with_dataset(dataloader, name: str | None, path: str | None):
    """Swap the single dataset inside the preset's packing config.

    Presets wrap exactly one ``SingleDatasetConfig`` in a packing config
    (``ConcatThenSplitPackingConfig(dataset=...)``); a preset built any other way
    has no single dataset to swap, and saying so beats guessing.
    """
    packing = dataloader.dataset
    single = getattr(packing, "dataset", None)
    if single is None or not hasattr(single, "source"):
        raise ValueError(
            f"--dataset/--dataset_path need a preset whose dataloader packs one "
            f"dataset; this one has {type(packing).__name__}"
        )
    if name is not None:
        if name not in DATASETS:
            raise ValueError(f"--dataset={name!r}: known datasets are {sorted(DATASETS)}")
        single = DATASETS[name]
    if path is not None:
        source = single.source
        # data_files are relative to the source path they came with; a new path
        # means the files are wherever that path points.
        kwargs = {
            k: v for k, v in source.load_dataset_kwargs.items() if k != "data_files"
        }
        single = dataclasses.replace(
            single,
            source=dataclasses.replace(source, path=path, load_dataset_kwargs=kwargs),
        )
    return dataclasses.replace(
        dataloader, dataset=dataclasses.replace(packing, dataset=single)
    )


def fault_tolerant(model_cls, config):
    """``config`` rebuilt as ``model_cls.Config``.

    ``model_cls`` is a model's fault-tolerant subclass: same model, plus the
    ``_fragment`` hook the FT trainer hands to DiLoCo for fragment-wise sync.
    """
    return model_cls.Config(
        **{f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    )
