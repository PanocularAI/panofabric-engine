# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 — torchtitan's NATIVE hybrid model, wrapped for fault tolerance.

Qwen3.5 interleaves GatedDeltaNet linear-attention layers with full attention.
That is why it must run through this module and NOT through
``models.hf_transformers``: the HF backend applies one causal BlockMask to every
layer and initializes weights from a class-name lookup that a GatedDeltaNet
matches nowhere, leaving ``A_log``/``dt_bias`` as uninitialized memory --
``-exp(A_log)`` then yields NaN on the first step. torchtitan's native
``models/qwen3_5`` builds the per-consumer mask dict and inits every parameter
explicitly.

Requires ``flash-linear-attention`` (declared in this repo's [train] extra):
torchtitan imports it at module scope for the GatedDeltaNet kernels but declares
it only in a VLM-CI requirements file that nothing installs.
"""

from torchtitan.experiments.torchft.config.job_config import FaultTolerantModelSpec
from torchtitan.experiments.torchft.diloco import fragment_llm
from torchtitan.models.qwen3_5 import (
    parallelize_qwen3_5,
    Qwen35StateDictAdapter,
    qwen3_5_configs,
)
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_vlm


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
) -> FaultTolerantModelSpec:
    """A Qwen3.5 flavor as a fault-tolerant spec (adds ``fragment_fn``).

    Mirrors torchtitan's own ``qwen3_5.model_registry`` field for field --
    including ``pipeline_vlm`` and the MoE load-balancing hook -- so a flavor
    behaves here exactly as it does upstream. The only additions are the
    FT wrapper and ``fragment_llm``, which is what lets HeLoCo/DiLoCo fragment
    the decoder for cross-site sync.

    Every Qwen3.5 flavor carries a vision encoder (it is part of the published
    checkpoint), but the decoder's forward takes ``pixel_values=None`` and falls
    back to plain ``positions`` when no MRoPE positions arrive, so a text-only
    dataloader trains the decoder correctly. The vision tower is then dead
    weight: ~0.4B of the 9B, which also syncs on every HeLoCo boundary.
    """
    kwargs = dict(attn_backend=attn_backend)
    if moe_comm_backend is not None:
        kwargs["moe_comm_backend"] = moe_comm_backend
    config = qwen3_5_configs[flavor](**kwargs)
    return FaultTolerantModelSpec(
        name="ft/qwen3_5",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3_5,
        pipelining_fn=pipeline_vlm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=Qwen35StateDictAdapter,
        fragment_fn=fragment_llm,
    )
