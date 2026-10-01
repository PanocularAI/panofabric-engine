import logging
from typing import Any

import torch.nn as nn
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    get_fsdp_reshard_after_forward_policy,
    resolve_fsdp_mesh,
)
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import annotate_replicated_parameters

logger = logging.getLogger(__name__)


def parallelize_resnet(
    model: nn.Module,
    *,
    parallelism_context: ParallelismContext,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
):
    """FSDP2 over each stage in ``model.layers``, then the root.

    Applied even on one GPU and for replicate-only data parallelism (HSDP with a
    shard degree of 1 is DDP): the trainer drives the model as an FSDP module,
    and dp_shard always exists in the storage mesh.
    """
    # Nothing here has a ShardingConfig: every parameter is replicated, which is
    # what FSDP needs to see to shard it over the data-parallel axes.
    annotate_replicated_parameters(model, parallelism_context)

    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
    fsdp_config: dict[str, Any] = {
        "mesh": dp_mesh,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        ),
    }
    if dp_mesh_dims is not None:
        fsdp_config["dp_mesh_dims"] = dp_mesh_dims
    if training.enable_cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward, parallelism_context.pp_enabled
    )
    for block in model.layers.values():
        fully_shard(block, **fsdp_config, reshard_after_forward=reshard_after_forward)
    fully_shard(model, **fsdp_config)

    # The loss is divided by the global sample count; FSDP must not divide again.
    disable_fsdp_gradient_division(model)

    if parallelism_context.dp_replicate_enabled:
        logger.info("Applied HSDP to the model")
    else:
        logger.info("Applied FSDP to the model")
    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")
