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

The GatedDeltaNet kernels come from ``attn-gym[linear]`` (a required torchtitan
dependency since upstream #4389); ``flash-linear-attention`` is no longer needed.
"""

import dataclasses
import json
import logging
import os
from dataclasses import dataclass

import torch

from torchtitan.experiments.torchft.diloco import fragment_llm
from torchtitan.models.qwen3_5 import model_registry as _qwen35_registry, Qwen35Model
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter

from panoengine.train.config import fault_tolerant

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
    network from noise is not worth running.

    This is NOT redundant with DCP's own "Missing key in checkpoint state_dict"
    check, which fires when the checkpoint lacks a tensor the mapping knows to
    ask for. The gap this closes is the symmetric one: a key missing from
    `from_hf_map` is never asked for either, so DCP is satisfied and `from_hf`
    drops it -- the parameter keeps its init value and nothing says a word.

    Vision-tower parameters only warn rather than raise, since our presets feed
    text and the tower is then unused. In practice DCP raises first for a
    checkpoint with no vision weights at all (observed:
    "Missing key in checkpoint state_dict: model.visual.blocks.0.attn.proj.bias"),
    so this branch is a backstop, not the thing that makes a vision-free
    checkpoint work. The published Qwen3.5 checkpoints all ship the tower.
    """

    def __init__(self, model_config, hf_assets_path):
        super().__init__(model_config, hf_assets_path)
        self._expected: frozenset[str] | None = None
        # Upstream keys a model WITHOUT a vision tower as a text-only checkpoint
        # (``model.*``). Every published Qwen3.5 checkpoint is multimodal
        # (``model.language_model.*`` + ``model.visual.*``), and our presets drop
        # the tower, so loading one would ask for keys that do not exist. Key the
        # decoder the way the checkpoint on disk does.
        if model_config.vision_encoder is None and _checkpoint_is_multimodal(
            hf_assets_path
        ):
            self.hf_language_model_prefix = "model.language_model"
            self.from_hf_map = {
                (f"model.language_model.{k[len('model.'):]}" if k.startswith("model.") else k): v
                for k, v in self.from_hf_map.items()
            }

    def _expected_params(self) -> frozenset[str]:
        """Parameter names of this config, from a meta build (allocates nothing)."""
        if self._expected is None:
            with torch.device("meta"):
                model = self.model_config.build()
            self._expected = frozenset(n for n, _ in model.named_parameters())
        return self._expected

    def from_hf(self, hf_state_dict):
        tt_state_dict = super().from_hf(hf_state_dict)
        if _is_vllm_side(self.model_config):
            # The RL generator's copy: torchtitan's vLLM wrapper swaps in layers
            # that register themselves with vLLM when built, so a second (meta)
            # build dies on "Duplicate GDN layer name". They add no parameters,
            # and the load this guard exists for is the trainer's.
            return tt_state_dict

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


def _is_vllm_side(model_config) -> bool:
    """Whether ``model_config`` is torchtitan's vLLM-generator rewrite of a
    Qwen3.5 config (its GatedDeltaNet inner modules are vLLM layers)."""
    return any(
        type(getattr(getattr(layer, "delta_net", None), "inner_gated_delta_net", None))
        .__module__.startswith("torchtitan.rl.")
        for layer in model_config.layers
    )


def _checkpoint_is_multimodal(hf_assets_path: str | None) -> bool:
    """Whether the HF checkpoint at ``hf_assets_path`` keys its decoder under
    ``model.language_model.`` -- read from the safetensors index, or from the
    single file's header. No checkpoint (random init) means nothing to match."""
    if not hf_assets_path:
        return False
    index = os.path.join(hf_assets_path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            keys = json.load(f)["weight_map"]
    else:
        single = os.path.join(hf_assets_path, "model.safetensors")
        if not os.path.exists(single):
            return False
        from safetensors import safe_open

        with safe_open(single, framework="pt") as f:
            keys = f.keys()
    return any(k.startswith("model.language_model.") for k in keys)


class FaultTolerantQwen35Model(Qwen35Model):
    """Qwen3.5 plus the `_fragment` hook DiLoCo splits the decoder with, loading
    through the verifying adapter above."""

    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35Model.Config):
        pass

    _fragment = staticmethod(fragment_llm)
    state_dict_adapter_cls = VerifyingQwen35StateDictAdapter


def model_registry(
    flavor: str,
    *,
    text_only: bool = True,
    enable_sp: bool = True,
    **kwargs,
) -> FaultTolerantQwen35Model.Config:
    """A Qwen3.5 flavor, fault-tolerant. ``kwargs`` go to torchtitan's qwen3_5
    ``model_registry`` (``seq_len``, ``attn_backend``, ``moe_comm_backend``).

    ``enable_sp`` must match ``parallelism.enable_sequence_parallel`` (default
    True): it picks the projection classes the sharding plan expects.

    ``text_only`` (the default) drops the vision tower. Every published Qwen3.5
    flavor carries one, and for a text workload it is pure cost: ~0.4B of the 9B
    in parameters and optimizer state, and bytes on the wire at every HeLoCo
    boundary, for a tower that never sees an image. Dropping it also avoids a
    hard blocker -- ``apply_fsdp_to_vision_encoder`` cannot shard the tower on
    this stack ("When dp_mesh_dims is provided, all parameters must be DTensors
    on the full SPMD mesh"), which makes ANY multi-GPU Qwen3.5 run fail while
    the tower is attached.

    The checkpoint's 57 vision tensors are then simply unused: the loader plans
    from the model's own parameters, so it never asks for them.

    Pass ``text_only=False`` for a multimodal run, which also needs the
    multimodal dataloader and tokenizer (see config_registry).
    """
    config = _qwen35_registry(flavor, enable_sp=enable_sp, **kwargs)
    if text_only:
        config = dataclasses.replace(config, vision_encoder=None)
    return fault_tolerant(FaultTolerantQwen35Model, config)
