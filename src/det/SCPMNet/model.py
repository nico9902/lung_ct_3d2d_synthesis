from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Norm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.normal = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normal(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3x3(in_channels, out_channels, stride=stride),
            Norm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = Norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = Norm(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class SAC(nn.Module):
    """Scale-aware convolution from the released SCPM-Net backbone."""

    def __init__(self, input_channel: int, out_channel: int):
        super().__init__()
        self.conv_1 = nn.Conv3d(input_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv3d(input_channel, out_channel, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv_5 = nn.Conv3d(input_channel, out_channel, kernel_size=3, stride=1, padding=3, dilation=3)
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.weights, dim=0)
        return self.conv_1(inputs) * weights[0] + self.conv_3(inputs) * weights[1] + self.conv_5(inputs) * weights[2]


class Pyramid3D(nn.Module):
    def __init__(self, c2: int, c3: int, c4: int, c5: int, feature_size: int = 64, using_sac: bool = False):
        super().__init__()
        conv = SAC if using_sac else nn.Conv3d
        self.p5_1 = nn.Conv3d(c5, feature_size, kernel_size=1)
        self.p5_2 = conv(feature_size, feature_size, kernel_size=3, stride=1, padding=1) if not using_sac else conv(feature_size, feature_size)
        self.p4_1 = nn.Conv3d(c4, feature_size, kernel_size=1)
        self.p4_2 = conv(feature_size, feature_size, kernel_size=3, stride=1, padding=1) if not using_sac else conv(feature_size, feature_size)
        self.p3_1 = nn.Conv3d(c3, feature_size, kernel_size=1)
        self.p3_2 = conv(feature_size, feature_size, kernel_size=3, stride=1, padding=1) if not using_sac else conv(feature_size, feature_size)
        self.p2_1 = nn.Conv3d(c2, feature_size, kernel_size=1)
        self.p2_2 = conv(feature_size, feature_size, kernel_size=3, stride=1, padding=1) if not using_sac else conv(feature_size, feature_size)

    def forward(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        c2, c3, c4, c5 = inputs
        p5 = self.p5_1(c5)
        p4 = self.p4_1(c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p3 = self.p3_1(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p2 = self.p2_1(c2) + F.interpolate(p3, size=c2.shape[2:], mode="nearest")
        return [self.p2_2(p2), self.p3_2(p3), self.p4_2(p4), self.p5_2(p5)]


class AttentionSECA(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Sequential(nn.Linear(channel, channel), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(channel, channel), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.fc2(self.fc1(self.pool(x).flatten(1))).view(x.size(0), x.size(1), 1, 1, 1)
        return gate * x


class SCPMNet(nn.Module):
    """SCPM-Net backbone and detection heads.

    The network predicts center logits, sphere radius, and local center offset on
    two stride-2 feature maps, matching the public paper backbone.
    """

    def __init__(self, layers: tuple[int, int, int, int] = (2, 2, 3, 3), using_sac: bool = False):
        super().__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = Norm(32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = Norm(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(32, layers[0])
        self.layer2 = self._make_layer(64, layers[1], stride=2)
        self.layer3 = self._make_layer(64, layers[2], stride=2)
        self.layer4 = self._make_layer(64, layers[3], stride=2)
        self.attention1 = AttentionSECA(32)
        self.attention2 = AttentionSECA(32)
        self.attention3 = AttentionSECA(64)
        self.attention4 = AttentionSECA(64)
        self.conv_1 = ConvBlock(67, 64)
        self.conv_2 = ConvBlock(67, 64)
        self.conv_3 = ConvBlock(67, 64)
        self.conv_4 = ConvBlock(67, 64)
        self.conv_8x = ConvBlock(64, 64)
        self.conv_4x = ConvBlock(64, 64)
        self.conv_2x = ConvBlock(64, 64)
        self.convc = nn.Conv3d(64, 1, kernel_size=1)
        self.convr = nn.Conv3d(64, 1, kernel_size=1)
        self.convo = nn.Conv3d(64, 3, kernel_size=1)
        self.fpn = Pyramid3D(32, 64, 64, 64, feature_size=64, using_sac=using_sac)
        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False), Norm(planes))
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def coordinate_map(batch: int, shape: tuple[int, int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        z = torch.linspace(-1.0, 1.0, shape[0], device=device, dtype=dtype)
        y = torch.linspace(-1.0, 1.0, shape[1], device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, shape[2], device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack([zz, yy, xx], dim=0).unsqueeze(0).repeat(batch, 1, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.attention1(self.maxpool(x))
        x1 = self.attention2(self.layer1(x))
        x2 = self.attention3(self.layer2(x1))
        x3 = self.attention4(self.layer3(x2))
        x4 = self.layer4(x3)
        feats = self.fpn([x1, x2, x3, x4])
        for idx, feat in enumerate(feats):
            coords = self.coordinate_map(feat.size(0), tuple(feat.shape[2:]), feat.device, feat.dtype)
            feats[idx] = getattr(self, f"conv_{idx + 1}")(torch.cat([feat, coords], dim=1))
        feat_8x = self.conv_8x(F.interpolate(feats[3], size=feats[2].shape[2:], mode="nearest") + feats[2])
        feat_4x = self.conv_4x(F.interpolate(feat_8x, size=feats[1].shape[2:], mode="nearest") + feats[1])
        feat_2x = self.conv_2x(F.interpolate(feat_4x, size=feats[0].shape[2:], mode="nearest"))
        return {
            "Cls1": self.convc(feats[0]),
            "Reg1": F.softplus(self.convr(feats[0])) + 1e-3,
            "Off1": torch.tanh(self.convo(feats[0])),
            "Cls2": self.convc(feat_2x),
            "Reg2": F.softplus(self.convr(feat_2x)) + 1e-3,
            "Off2": torch.tanh(self.convo(feat_2x)),
        }


def scpmnet18(**kwargs) -> SCPMNet:
    return SCPMNet(layers=(2, 2, 3, 3), **kwargs)
