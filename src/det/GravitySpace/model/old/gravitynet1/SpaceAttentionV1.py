from typing import Tuple

import torch
from torch import nn

class SpaceAttentionV1(nn.Module):
    """
    SpaceAttentionV1
    """

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
        super(SpaceAttentionV1, self).__init__()

        self.num_features = num_features
        self.feature_map_shape = feature_map_shape
        self.hidden_dim = hidden_dim
        self.window_size = window_size

        # attention layer
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)

        # projection layer
        self.query_projection = nn.Linear(self.num_features * self.feature_map_shape[0] * self.feature_map_shape[1], self.hidden_dim)
        self.key_projection = nn.Linear(self.num_features * self.feature_map_shape[0] * self.feature_map_shape[1], self.hidden_dim)
        self.value_projection = nn.Linear(self.num_features * self.feature_map_shape[0] * self.feature_map_shape[1], self.hidden_dim)

        # inverse projection layer
        self.inverse_projection = nn.Linear(self.hidden_dim, self.num_features * self.feature_map_shape[0] * self.feature_map_shape[1])

        # positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(self.window_size, self.hidden_dim))  # window_size x hidden_dim

        # Xavier initialization
        torch.manual_seed(seed=0)
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.xavier_uniform_(self.inverse_projection.weight)
        nn.init.xavier_uniform_(self.positional_encoding)


    def forward(self, window_features: torch.Tensor,
                current_slice_index: int,
                B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward method: directly call a method in the class when an instance name is called

        :param window_features: Input features of shape B x K x F x H_FM x W_FM
        :param current_slice_index: Index of the current slice
        :param B: Batch size
        :return: Tuple of attention outputs B x F x H_FM x W_FM and weights B x K x K
        """

        # ---------- #
        # Projection #
        # ---------- #
        # combine F, H_FM, W_FM to create the full token dimension of (F * H_FM * W_FM)
        window_features = window_features.reshape(B, self.window_size, -1)  # B x K x (F * H_FM * W_FM)

        # query, key, value
        Q = self.query_projection(window_features)
        K = self.key_projection(window_features)
        V = self.value_projection(window_features)

        # apply positional encoding
        Q += self.positional_encoding.unsqueeze(0).expand(B, -1, -1).to(window_features.device)  # adding B dimension in first place and add positional encoding to the Q
        K += self.positional_encoding.unsqueeze(0).expand(B, -1, -1).to(window_features.device)  # adding B dimension in first place and add positional encoding to the K
        V += self.positional_encoding.unsqueeze(0).expand(B, -1, -1).to(window_features.device)  # adding B dimension in first place and add positional encoding to the V

        # --------- #
        # Attention #
        # --------- #
        # attention
        attention_output, attention_weights = self.attention(Q, K, V)  # B x K x hidden_dim

        # inverse projection to the original token dimension
        attention_output = self.inverse_projection(attention_output)  # B x K x (F * H_FM * W_FM)

        # reshape to original window feature dimensions
        attention_output = attention_output.reshape(B, self.window_size, self.num_features, self.feature_map_shape[0], self.feature_map_shape[1])  # B x K x F x H_FM x W_FM

        # take the output corresponding to the current slice
        attention_output = attention_output[:, current_slice_index, :, :, :]  # B x F x H_FM x W_FM

        return attention_output, attention_weights
