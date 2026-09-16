from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


SUPPORTED_BACKBONES = ("efficientnet_v2_s", "efficientnet_b0", "resnet18", "resnet50")


def _torchvision_models():
    from torchvision import models

    return models


def _weights_enum(backbone: str, pretrained: bool):
    if not pretrained:
        return None
    models = _torchvision_models()
    mapping = {
        "efficientnet_v2_s": models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
    }
    return mapping[backbone]


def _first_conv2d(model: nn.Module) -> tuple[nn.Module, str, nn.Conv2d] | None:
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            return model, name, child
        found = _first_conv2d(child)
        if found is not None:
            return found
    return None


def _model_rgb2gray(model: nn.Module) -> nn.Module:
    found = _first_conv2d(model)
    if found is None:
        raise ValueError("Could not find a Conv2d layer to convert to grayscale input.")
    parent, name, conv = found
    if conv.in_channels == 1:
        return model
    if conv.in_channels != 3:
        raise ValueError(f"Expected first Conv2d to have 3 input channels, got {conv.in_channels}.")

    gray_conv = nn.Conv2d(
        in_channels=1,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        gray_conv.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        if conv.bias is not None:
            gray_conv.bias.copy_(conv.bias)
    setattr(parent, name, gray_conv)
    return model


def _build_feature_extractor(backbone: str, pretrained: bool) -> tuple[nn.Module, int]:
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose from: {', '.join(SUPPORTED_BACKBONES)}")

    models = _torchvision_models()
    model = getattr(models, backbone)(weights=_weights_enum(backbone, pretrained))
    model = _model_rgb2gray(model)

    if backbone.startswith("efficientnet"):
        feature_dim = model.classifier[-1].in_features
        extractor = nn.Sequential(model.features, model.avgpool, nn.Flatten(1))
    elif backbone.startswith("resnet"):
        feature_dim = model.fc.in_features
        extractor = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
            nn.Flatten(1),
        )
    else:
        raise AssertionError(f"Backbone handling missing for {backbone}")
    return extractor, feature_dim


def _freeze_half_backbone_parameters(encoder: nn.Module) -> tuple[int, int]:
    """Freeze approximately half of the encoder weights, starting from its input side."""
    parameters = list(encoder.parameters())
    total_weights = sum(parameter.numel() for parameter in parameters)
    target = total_weights // 2
    frozen_weights = 0

    for parameter in parameters:
        if frozen_weights + parameter.numel() <= target:
            parameter.requires_grad = False
            frozen_weights += parameter.numel()

    print(
        f"Frozen {frozen_weights:,}/{total_weights:,} backbone weights "
        f"({100 * frozen_weights / total_weights:.1f}%; target 50.0%)"
    )
    return frozen_weights, total_weights


class GatedAttention(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.v = nn.Sequential(nn.Linear(feature_dim, attention_dim), nn.Tanh())
        self.u = nn.Sequential(nn.Linear(feature_dim, attention_dim), nn.Sigmoid())
        self.score = nn.Linear(attention_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gated = self.dropout(self.v(features) * self.u(features))
        logits = self.score(gated).squeeze(-1).float()
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask, -torch.inf)
        weights = torch.softmax(logits, dim=1)
        if valid_mask is not None:
            weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
        pooled = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class SliceAttentionClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_v2_s",
        pretrained: bool = True,
        num_classes: int = 2,
        attention_dim: int = 256,
        dropout: float = 0.2,
        slice_chunk_size: int = 16,
        freeze_backbone: bool = False,
        freeze_half_backbone: bool = False,
    ) -> None:
        super().__init__()
        if freeze_backbone and freeze_half_backbone:
            raise ValueError("Choose either full or half backbone freezing, not both.")
        self.slice_chunk_size = slice_chunk_size
        self.encoder, feature_dim = _build_feature_extractor(backbone, pretrained)
        if freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        elif freeze_half_backbone:
            _freeze_half_backbone_parameters(self.encoder)
        self.attention = GatedAttention(feature_dim, attention_dim=attention_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(
        self,
        slices: torch.Tensor,
        slice_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if slices.ndim == 5:
            batch_size, num_slices, channels, height, width = slices.shape
            flat_slices = slices.reshape(batch_size * num_slices, channels, height, width)
            counts = [num_slices] * batch_size
        elif slices.ndim == 4 and slice_counts is not None:
            flat_slices = slices
            counts = [int(count) for count in slice_counts.tolist()]
            if not counts or any(count <= 0 for count in counts):
                raise ValueError(f"slice_counts must contain positive values, got {counts}")
            if sum(counts) != slices.shape[0]:
                raise ValueError(
                    f"slice_counts sum to {sum(counts)}, but packed input has {slices.shape[0]} slices"
                )
        else:
            raise ValueError(
                "Expected BxSxCxHxW slices, or packed NxCxHxW slices with slice_counts; "
                f"got {tuple(slices.shape)}"
            )

        encoded_chunks = []
        for chunk in flat_slices.split(self.slice_chunk_size, dim=0):
            encoded_chunks.append(self.encoder(chunk))
        packed_features = torch.cat(encoded_chunks, dim=0)
        feature_sequences = packed_features.split(counts)
        features = pad_sequence(feature_sequences, batch_first=True)
        count_tensor = torch.tensor(counts, device=features.device)
        valid_mask = torch.arange(features.shape[1], device=features.device).unsqueeze(0) < count_tensor.unsqueeze(1)
        pooled, attention_weights = self.attention(features, valid_mask)
        logits = self.classifier(pooled)
        return logits, attention_weights
