"""CIFAR-10 for image classification on torchtitan's Grain dataloader.

``GrainDataLoader`` already does what the old ``ParallelAwareDataloader`` did
here and more: a disjoint shard per data-parallel rank, repeat forever,
prefetch, and checkpointable position (shuffled per epoch too). All CIFAR-10
needs is a collator that stacks images instead of packing tokens.

One "token" is one image: the collator takes
``training.num_tokens_per_microbatch_per_dp_rank`` rows per microbatch, and
``num_valid_tokens`` is the image count the summed loss is divided by.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from torchtitan.components.data import (
    Collator,
    DatasetBuildContext,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
    TrainingMicrobatch,
)


@dataclass(kw_only=True, slots=True)
class ImageMicrobatch(TrainingMicrobatch):
    """``input`` is ``[batch, 3, H, W]`` floats in [0, 1]; ``labels`` is ``[batch]``."""

    input: torch.Tensor
    labels: torch.Tensor
    num_valid_tokens: int

    def as_input_dict(self) -> dict[str, Any]:
        return {"input": self.input, "labels": self.labels}


class ImageCollator(Collator):
    """Stacks HF image rows (``img``: PIL image, ``label``: int) into a microbatch."""

    @dataclass(kw_only=True, slots=True)
    class Config(Collator.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        del config
        self._batch_size = context.num_tokens_per_microbatch

    def num_rows_per_microbatch(self) -> int:
        return self._batch_size

    def __call__(self, rows: Sequence[dict[str, Any]]) -> ImageMicrobatch:
        images = np.stack([np.asarray(row["img"]) for row in rows])  # (B, H, W, C)
        return ImageMicrobatch(
            input=torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().float() / 255.0,
            labels=torch.tensor([row["label"] for row in rows]),
            num_valid_tokens=len(rows),
        )


def cifar10_dataloader(path: str = "uoft-cs/cifar10") -> GrainDataLoader.Config:
    """The CIFAR-10 train split (a local dir or HF repo id), one image per token."""
    return GrainDataLoader.Config(
        dataset=SingleDatasetConfig(
            source=HuggingFaceRandomAccessSource.Config(path=path, split="train"),
        ),
        collator=ImageCollator.Config(),
    )
