# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.torchft.config.job_config import FaultTolerantModelSpec
from torchtitan.experiments.torchft.diloco import fragment_llm
from torchtitan.models.qwen3 import (
    Qwen3StateDictAdapter,
    parallelize_qwen3,
    pipeline_llm,
    qwen3_configs,
)


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
) -> FaultTolerantModelSpec:
    kwargs = dict(attn_backend=attn_backend)
    if moe_comm_backend is not None:
        kwargs["moe_comm_backend"] = moe_comm_backend
    config = qwen3_configs[flavor](**kwargs)
    return FaultTolerantModelSpec(
        name="ft/qwen3",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
        fragment_fn=fragment_llm,
    )
