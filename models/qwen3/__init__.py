# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.experiments.torchft.diloco import fragment_llm
from torchtitan.models.qwen3 import model_registry as _qwen3_registry, Qwen3Model

from panoengine.train.config import fault_tolerant


class FaultTolerantQwen3Model(Qwen3Model):
    """Qwen3 plus the `_fragment` hook DiLoCo splits the model with."""

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen3Model.Config):
        pass

    _fragment = staticmethod(fragment_llm)


def model_registry(flavor: str, **kwargs) -> FaultTolerantQwen3Model.Config:
    """A qwen3 flavor, fault-tolerant. ``kwargs`` go to torchtitan's qwen3
    ``model_registry`` (``seq_len``, ``attn_backend``, ``moe_comm_backend``)."""
    return fault_tolerant(FaultTolerantQwen3Model, _qwen3_registry(flavor, **kwargs))
