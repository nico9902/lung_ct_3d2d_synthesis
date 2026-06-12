from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.functional import softmax


class SpaceAttentionV5(nn.Module):

    def __init__(self,
                 num_features: int,
                 hidden_dim: int):
        """
        __init__ method: run one when instantiating the object

        :param num_features: Number of features from the backbone model.
        :param feature_map_shape: Shape of the feature map (Height, Width).
        :param hidden_dim: Dimension for attention mechanism.
        :param window_size: Number of slices in the attention window.
        """

        super(SpaceAttentionV5, self).__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim

        # soft attention
        self.soft_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1)
        )

        # Xavier initialization
        torch.manual_seed(seed=0)
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


    def forward(self,
                window_features_tokens: torch.Tensor,
                current_slice_index: int,
                B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward method: directly call a method in the class when an instance name is called

        :param window_features_tokens: window features B x K x F
        :param current_slice_index: Index of the current slice
        :param B: Batch size
        :return: Tuple of attention outputs B x F x H_FM x W_FM and weights B x K x 1 x 1 x 1
        """

        B, K, hidden_dim = window_features_tokens.shape  # B x K x hidden_dim

        # pass through soft attention layer
        attention_logits = self.soft_attention(window_features_tokens.reshape(B * K, hidden_dim))  # from B x K x hidden_dim to B * K x hidden_dim to B * K x 1

        # reshape to B x K for a softmax across the K slices
        attention_logits = attention_logits.view(B, K)  # B x K
        attention_weights = softmax(attention_logits, dim=1)  # softmax along K

        # expand weights for broadcasting
        attention_weights = attention_weights.unsqueeze(2)  # from B x K to B x K x 1

        # weighted sum across K
        attention_output = (window_features_tokens * attention_weights).sum(dim=1)  # B x K x hidden_dim multiplied B x K x 1, summed along K: B x hidden_dim

        # reshape the attention output: from B x hidden_dim to B x 1 x hidden_dim
        attention_output = attention_output.unsqueeze(1)


        return attention_output, attention_weights