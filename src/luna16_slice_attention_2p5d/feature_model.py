from __future__ import annotations

import torch
import torch.nn as nn


class FeatureAttentionClassifier(nn.Module):
    def __init__(self, feature_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.attention_v = nn.Sequential(nn.Linear(feature_dim, 256), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(feature_dim, 256), nn.Sigmoid())
        self.attention_w = nn.Linear(256, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, 1))

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gated = self.attention_v(features) * self.attention_u(features)
        attention_logits = self.attention_w(gated).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~valid_mask, -torch.finfo(attention_logits.dtype).max)
        attention_weights = torch.softmax(attention_logits, dim=1)
        attention_weights = torch.where(valid_mask, attention_weights, torch.zeros_like(attention_weights))
        pooled = (features * attention_weights.unsqueeze(-1)).sum(dim=1)
        logits = self.classifier(pooled).squeeze(1)
        return logits, attention_weights
