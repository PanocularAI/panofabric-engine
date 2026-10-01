# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .infra.parallelize import parallelize_resnet
from .model.args import ResNetModelArgs
from .model.model import ResNetModel

__all__ = [
    "parallelize_resnet",
    "ResNetModelArgs",
    "ResNetModel",
    "resnet_configs",
    # Listed so the `models.resnet` compatibility shim's star-import re-exports
    # it — config_registry.py does `from . import model_registry`.
    "model_registry",
]


resnet_configs = {
    "18": ResNetModelArgs(
        num_classes=10,
        block="ResidualBlock",
        layers=[2, 2, 2, 2]
    ),
    "34": ResNetModelArgs(
        num_classes=10,
        block="ResidualBlock",
        layers=[3, 4, 6, 3]
    ),
    "50": ResNetModelArgs(
        num_classes=10,
        block="BottleNeckBlock",
        layers=[3, 4, 6, 3]
    ),
    "152": ResNetModelArgs(
        num_classes=10,
        block="BottleNeckBlock",
        layers=[3, 8, 36, 3]
    ),
}


# No fault-tolerant subclass: ResNet has no `_fragment` hook, so DiLoCo syncs the
# whole model and the presets pin semi_sync(num_fragments=1).
def model_registry(flavor: str) -> ResNetModel.Config:
    return resnet_configs[flavor]
