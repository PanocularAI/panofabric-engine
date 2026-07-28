from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch.distributed._composable.replicate import replicate

from torch.distributed.device_mesh import DeviceMesh
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.config.configs import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


def parallelize_resnet(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
    **kwargs,
):
    if parallel_dims.fsdp_enabled:
        names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        dp_mesh = parallel_dims.get_mesh(names)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh(["dp_replicate"])
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=False,
        )

    return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    for layer_id, block in model.layers.items():
        fully_shard(
            block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    fully_shard(model, **fsdp_config)


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
):
    if enable_compile:
        torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")
