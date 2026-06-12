from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.functional import softmax


class SpaceAttentionV5(nn.Module):

    def __init__(self,
                 num_features: int,
                 feature_map_shape: Tuple[int, int],
                 hidden_dim: int,
                 window_size: int):
        """
        __init__ method: run one when instantiating the object

        :param num_features: Number of features from the backbone model.
        :param feature_map_shape: Shape of the feature map (Height, Width).
        :param hidden_dim: Dimension for attention mechanism.
        :param window_size: Number of slices in the attention window.
        """

        super(SpaceAttentionV5, self).__init__()

        self.num_features = num_features
        self.feature_map_shape = feature_map_shape
        self.hidden_dim = hidden_dim
        self.window_size = window_size

        # projection
        self.projection = nn.Linear(self.num_features * self.feature_map_shape[0] * self.feature_map_shape[1], self.hidden_dim)

        # positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(self.window_size, self.hidden_dim))  # window_size x hidden_dim

        # soft attention
        self.soft_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1)
        )

        # inverse projection layer
        self.inverse_projection = nn.Linear(self.hidden_dim, self.num_features * self.feature_map_shape[0] * self.feature_map_shape[1])

        # Xavier initialization
        torch.manual_seed(seed=0)
        self._initialize_weights()
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.positional_encoding)
        nn.init.xavier_uniform_(self.inverse_projection.weight)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


    def forward(self,
                window_features: torch.Tensor,
                current_slice_index: int,
                B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward method: directly call a method in the class when an instance name is called

        :param window_features: window features B x K x F x H_FM x W_FM
        :param current_slice_index: Index of the current slice
        :param B: Batch size
        :return: Tuple of attention outputs B x F x H_FM x W_FM and weights B x K x 1 x 1 x 1
        """

        B, K, F, H_FM, W_FM = window_features.shape  # B x K x F x H_FM x W_FM

        # combine F, H_FM, W_FM to create the full token dimension of (F * H_FM * W_FM)
        window_features = window_features.reshape(B, self.window_size, -1)  # B x K x (F * H_FM * W_FM)

        # tokens
        tokens = self.projection(window_features) # B x K x hidden_dim

        # apply positional encoding
        tokens += self.positional_encoding.unsqueeze(0).expand(B, -1, -1).to(window_features.device)  # adding B dimension in first place and add positional encoding to the tokens

        # pass through soft attention layer
        attention_logits = self.soft_attention(tokens.view(B * K, F))  # from B x K x hidden_dim to B * K x hidden_dim to B * K x 1

        # reshape to B x K for a softmax across the K slices
        attention_logits = attention_logits.view(B, K)  # B x K
        attention_weights = softmax(attention_logits, dim=1)  # softmax along K

        # expand weights for broadcasting
        attention_weights = attention_weights.view(B, K, 1)  # from B x K to B x K x 1

        # weighted sum across K
        attention_output = (tokens * attention_weights).sum(dim=1)  # B x K x hidden_dim multiplied B x K x 1, summed along K: B x hidden_dim

        # reshape the attention output: from B x hidden_dim to B x (F * H_FM * W_FM) to B x F x H_FM x W_FM
        attention_output = self.inverse_projection(attention_output).reshape(B, self.num_features, self.feature_map_shape[0], self.feature_map_shape[1])

        return attention_output, attention_weights