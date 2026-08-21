from __future__ import annotations

import torch
import torch.nn as nn

from src.luna16_synthetic_2d.models import SUPPORTED_BACKBONES, _weights_enum
from torchvision import models


class DetectionMILClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_v2_s",
        pooling: str = "attention",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone '{backbone}'")
        if pooling not in {"mean", "max", "attention"}:
            raise ValueError("pooling must be one of: mean, max, attention")

        self.backbone_name = backbone
        self.pooling = pooling
        self.encoder, feature_dim = self._build_encoder(backbone, pretrained)
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
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

    def _pool(self, features: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = valid_mask.bool()
        if self.pooling == "mean":
            weights = mask.float()
            pooled = (features * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            return pooled, weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)

        if self.pooling == "max":
            masked = features.masked_fill(~mask.unsqueeze(-1), -torch.finfo(features.dtype).max)
            pooled = masked.max(dim=1).values
            has_instances = mask.any(dim=1, keepdim=True)
            pooled = torch.where(has_instances, pooled, torch.zeros_like(pooled))
            weights = torch.zeros(mask.shape, device=features.device, dtype=features.dtype)
            return pooled, weights

        logits = self.attention(features).squeeze(-1)
        logits = logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)
        weights = torch.softmax(logits, dim=1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        pooled = (features * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights

    def forward(self, bags: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, top_k, channels, height, width = bags.shape
        flat = bags.view(batch_size * top_k, channels, height, width)
        features = self.encoder(flat).view(batch_size, top_k, -1)
        pooled, attention_weights = self._pool(features, valid_mask)
        logits = self.classifier(pooled).squeeze(1)
        return logits, attention_weights
