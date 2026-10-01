# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torchtitan's own fault-tolerant llama3: the native model plus the `_fragment`
# hook DiLoCo splits it with. Re-exported so every recipe is `models.<pkg>`.
from torchtitan.experiments.torchft.llama3 import model_registry

__all__ = ["model_registry"]
