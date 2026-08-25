import torch
from torch import nn

from .args import ResNetModelArgs

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu(out)
        return out
    

class BottleNeckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super().__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv3 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size = 1),
                        nn.BatchNorm2d(out_channels * self.expansion),
                        nn.ReLU())
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class ResNetModel(nn.Module):
    def __init__(self, model_args: ResNetModelArgs):
        super().__init__()

        block = ResidualBlock if model_args.block == "ResidualBlock" else BottleNeckBlock

        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layers = torch.nn.ModuleDict()
        self.layers["0"] = self._make_layer(block, 64, model_args.layers[0], stride=1)
        self.layers["1"] = self._make_layer(block, 128, model_args.layers[1], stride=2)
        self.layers["2"] = self._make_layer(block, 256, model_args.layers[2], stride=2)
        self.layers["3"] = self._make_layer(block, 512, model_args.layers[3], stride=2)

        # for layer_id in range(model_args.n_layers):
        #     self.layers[str(layer_id)] = self._make_layer(ResidualBlock, layer_conf[layer_id][0], layer_conf[layer_id][1], stride = layer_conf[layer_id][2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, model_args.num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                if module.affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zero-initialize the last BN in each residual block to improve stability.
        for layer in self.layers.values():
            for block in layer:
                final_bn = getattr(block, "bn3", None)
                if final_bn is None:
                    final_bn = getattr(block, "bn2", None)

                if isinstance(final_bn, nn.BatchNorm2d) and final_bn.affine:
                    nn.init.zeros_(final_bn.weight)

    def init_states(self, *, buffer_device=None) -> None:
        self.init_weights(buffer_device=buffer_device)

    def verify_module_protocol(self) -> None:
        pass

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = self.conv1(x)
        x = self.maxpool(x)
        for layer in self.layers.values():
            x = layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x