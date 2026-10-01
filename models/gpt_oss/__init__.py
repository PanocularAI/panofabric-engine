# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.experiments.torchft.diloco import fragment_llm
from torchtitan.models.gpt_oss import GptOssModel, model_registry as _gptoss_registry

from panoengine.train.config import fault_tolerant


class FaultTolerantGptOssModel(GptOssModel):
    """GPT-OSS plus the `_fragment` hook DiLoCo splits the model with."""

    @dataclass(kw_only=True, slots=True)
    class Config(GptOssModel.Config):
        pass

    _fragment = staticmethod(fragment_llm)


def model_registry(flavor: str, **kwargs) -> FaultTolerantGptOssModel.Config:
    """A GPT-OSS flavor, fault-tolerant. ``kwargs`` go to torchtitan's gpt_oss
    ``model_registry`` (``seq_len``, ``moe_comm_backend``, ``attn_backend``)."""
    return fault_tolerant(FaultTolerantGptOssModel, _gptoss_registry(flavor, **kwargs))
