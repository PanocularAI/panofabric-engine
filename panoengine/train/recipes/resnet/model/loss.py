from dataclasses import dataclass

import torch

from torchtitan.components.loss import BaseLoss, LossFunction
from torchtitan.config import CompileConfig


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss for image classification.

    Uses sum reduction (not mean) so the trainer can normalize by the global
    sample count (``global_valid_tokens``) across data-parallel ranks, matching
    how the LLM losses are accounted for. ``pred`` is ``[batch, num_classes]``
    and ``labels`` is ``[batch]``, so ``global_valid_tokens`` equals the batch
    size and ``sum / batch`` recovers the mean cross-entropy.
    """
    return torch.nn.functional.cross_entropy(pred.float(), labels, reduction="sum")


class ResNetCrossEntropyLoss(BaseLoss):
    """Cross-entropy loss for ResNet image classification."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    def __init__(
        self,
        config: "ResNetCrossEntropyLoss.Config",
        *,
        compile_config: CompileConfig | None = None,
    ):
        self.fn: LossFunction = cross_entropy_loss
        self._maybe_compile(compile_config)
