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

import logging

import torch

from torchtitan.experiments.torchft.config.job_config import FaultTolerantModelSpec
from torchtitan.experiments.torchft.diloco import fragment_llm
from torchtitan.models.qwen3_5 import (
    parallelize_qwen3_5,
    Qwen35StateDictAdapter,
    qwen3_5_configs,
)
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_vlm

logger = logging.getLogger(__name__)


class VerifyingQwen35StateDictAdapter(Qwen35StateDictAdapter):
    """Qwen3.5's HF adapter, made to fail loudly on a partial load.

    The base adapter translates HF checkpoint keys to torchtitan ones through a
    lookup table and skips anything it does not recognise::

        if hf_abstract_key not in self.from_hf_map:
            continue        # no warning, no error

    A checkpoint tensor whose name is missing from that table is therefore
    dropped in silence, and the parameter keeps the random value `init_weights`
    gave it. Nothing downstream notices: the run does not crash, does not NaN,
    and its loss still falls -- it is just quietly worse than the model it was
    supposed to start from, which is the most expensive kind of bug to find.

    So after translating, check that every parameter the model will ask for
    actually received a tensor.

    Decoder parameters raise: a fine-tune that silently starts part of the
    network from noise is not worth running. Vision-tower parameters only warn
    -- our presets feed text, so the tower is unused, and a text-only Qwen3.5
    checkpoint legitimately carries no vision weights.
    """

    def __init__(self, model_config, hf_assets_path):
        super().__init__(model_config, hf_assets_path)
        self._expected: frozenset[str] | None = None

    def _expected_params(self) -> frozenset[str]:
        """Parameter names of this config, from a meta build (allocates nothing)."""
        if self._expected is None:
            with torch.device("meta"):
                model = self.model_config.build()
            self._expected = frozenset(n for n, _ in model.named_parameters())
        return self._expected

    def from_hf(self, hf_state_dict):
        tt_state_dict = super().from_hf(hf_state_dict)

        missing = self._expected_params() - set(tt_state_dict)
        vision = {n for n in missing if n.startswith("vision_encoder.")}
        decoder = sorted(missing - vision)

        if vision:
            logger.warning(
                "%d vision-tower parameters were not in the checkpoint and keep "
                "their initial values. Harmless for a text run (the tower is "
                "unused); a sign of the wrong checkpoint if you meant to train "
                "multimodally.", len(vision),
            )
        if decoder:
            shown = ", ".join(decoder[:8])
            more = f" (+{len(decoder) - 8} more)" if len(decoder) > 8 else ""
            raise ValueError(
                f"{len(decoder)} decoder parameters got no weights from the HF "
                f"checkpoint and would train from random init: {shown}{more}. "
                "Either the checkpoint does not match this model flavor, or "
                "Qwen35StateDictAdapter.from_hf_map is missing an entry for a "
                "renamed tensor. Fix the mapping rather than removing this check "
                "-- the load would otherwise succeed silently."
            )

        logger.info(
            "Loaded %d tensors from the HF checkpoint into %d model parameters.",
            len(tt_state_dict), len(self._expected_params()),
        )
        return tt_state_dict


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
        state_dict_adapter=VerifyingQwen35StateDictAdapter,
        fragment_fn=fragment_llm,
    )
