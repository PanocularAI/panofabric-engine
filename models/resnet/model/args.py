from dataclasses import dataclass
from typing import Literal, Optional

import math
from torch import nn

import dataclasses


@dataclass
class ResNetModelArgs:
    """
    Configuration for building a ResNet-50 backbone.
    """
    num_classes: int = 10
    n_layers: int = 4  # number of layers to include (max 4)
    block: Literal["ResidualBlock", "BottleNeckBlock"] = "ResidualBlock"
    layers: Optional[list[int]] = None  # number of blocks in each layer

    def update_from_config(self, **kwargs) -> None:
        pass

    def build(self):
        from .model import ResNetModel
        return ResNetModel(self)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        nparams = sum(p.numel() for p in model.parameters())
        
        nflops = 0.0

        in_channels = 3
        h, w = seq_len, seq_len

        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                k_h, k_w = layer.kernel_size
                s_h, s_w = layer.stride
                p_h, p_w = layer.padding
                g = layer.groups
                out_channels = layer.out_channels

                # compute output spatial dimensions
                h_out = math.floor((h + 2*p_h - k_h) / s_h + 1)
                w_out = math.floor((w + 2*p_w - k_w) / s_w + 1)

                # FLOPs for this conv: 2 * Cout * Hout * Wout * (Cin/group) * Kh * Kw
                nflops += 2 * out_channels * h_out * w_out * (in_channels / g) * k_h * k_w

                # output of this layer will be input to next
                in_channels = out_channels
                h, w = h_out, w_out

            elif isinstance(layer, nn.Linear):
                nflops += 2 * layer.in_features * layer.out_features

        return nparams, nflops
