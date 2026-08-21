from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from src.luna16_synthetic_2d.models import SUPPORTED_BACKBONES, _weights_enum


class SliceAttentionClassifier(nn.Module):
    """Encode all axial slices with a shared 2D backbone and aggregate via gated attention."""

    def __init__(
        self,
        backbone: str = "efficientnet_v2_s",
        pretrained: bool = True,
        dropout: float = 0.2,
        encoder_chunk_size: int = 32,
    ) -> None:
        super().__init__()
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone '{backbone}'")
        self.backbone_name = backbone
        self.encoder_chunk_size = int(encoder_chunk_size)
        self.encoder, feature_dim = self._build_encoder(backbone, pretrained)
        self.attention_v = nn.Sequential(nn.Linear(feature_dim, 256), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(feature_dim, 256), nn.Sigmoid())
        self.attention_w = nn.Linear(256, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, 1))

    @staticmethod
    def _build_encoder(backbone: str, pretrained: bool) -> tuple[nn.Module, int]:
        model = getattr(models, backbone)(weights=_weights_enum(backbone, pretrained))
        if backbone.startswith("efficientnet"):
            feature_dim = model.classifier[-1].in_features
            model.classifier = nn.Identity()
            return model, feature_dim
        if backbone.startswith("resnet"):
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, feature_dim
        if backbone.startswith("densenet"):
            feature_dim = model.classifier.in_features
            model.classifier = nn.Identity()
            return model, feature_dim
        if backbone.startswith("vgg"):
            feature_dim = model.classifier[-1].in_features
            model.classifier[-1] = nn.Identity()
            return model, feature_dim
        raise AssertionError(f"Backbone handling missing for {backbone}")

    def _encode_slices(self, slices: torch.Tensor) -> torch.Tensor:
        chunks = []
        for start in range(0, slices.shape[0], self.encoder_chunk_size):
            chunks.append(self.encoder(slices[start : start + self.encoder_chunk_size]))
        return torch.cat(chunks, dim=0)

    def forward(self, bags: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_slices, channels, height, width = bags.shape
        flat = bags.view(batch_size * max_slices, channels, height, width)
        flat_mask = valid_mask.view(-1)
        valid_flat = flat[flat_mask]

        feature_dim = self.classifier[-1].in_features
        feature_dtype = bags.dtype
        if valid_flat.numel() > 0:
            encoded = self._encode_slices(valid_flat)
            feature_dtype = encoded.dtype
        flat_features = torch.zeros(flat.shape[0], feature_dim, device=bags.device, dtype=feature_dtype)
        if valid_flat.numel() > 0:
            flat_features[flat_mask] = encoded
        features = flat_features.view(batch_size, max_slices, feature_dim)

        gated = self.attention_v(features) * self.attention_u(features)
        attention_logits = self.attention_w(gated).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~valid_mask, -torch.finfo(attention_logits.dtype).max)
        attention_weights = torch.softmax(attention_logits, dim=1)
        attention_weights = torch.where(valid_mask, attention_weights, torch.zeros_like(attention_weights))
        pooled = (features * attention_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(pooled).squeeze(1)
        return logits, attention_weights
