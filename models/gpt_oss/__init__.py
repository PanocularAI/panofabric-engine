# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.torchft.config.job_config import FaultTolerantModelSpec
from torchtitan.experiments.torchft.diloco import fragment_llm
from torchtitan.models.gpt_oss import (
    GptOssStateDictAdapter,
    gptoss_configs,
    parallelize_gptoss,
    register_moe_load_balancing_hook,
)


def model_registry(
    flavor: str,
    moe_comm_backend: str = "standard",
) -> FaultTolerantModelSpec:
    config = gptoss_configs[flavor](moe_comm_backend=moe_comm_backend)
    return FaultTolerantModelSpec(
        name="ft/gpt_oss",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_gptoss,
        pipelining_fn=None,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=GptOssStateDictAdapter,
        fragment_fn=fragment_llm,
    )
