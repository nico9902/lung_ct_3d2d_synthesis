from typing import Tuple

import torch
from torch import nn

class SpaceAttentionV4(nn.Module):
    """
    SpaceAttentionV4
    """

    def __init__(self,
                 hidden_dim: int):
        """
        __init__ method: run one when instantiating the object

        :param hidden_dim: Dimension for attention mechanism.
        """
        super(SpaceAttentionV4, self).__init__()

        self.hidden_dim = hidden_dim

        # attention layer
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)


    def forward(self,
                window_tokens_Q: torch.Tensor,
                window_tokens_K: torch.Tensor,
                window_tokens_V: torch.Tensor,
                current_slice_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward method: directly call a method in the class when an instance name is called

        :param window_tokens_Q: Q Input features of shape B x K x hidden_dim
        :param window_tokens_K: K Input features of shape B x K x hidden_dim
        :param window_tokens_V: V Input features of shape B x K x hidden_dim
        :param current_slice_index: Index of the current slice
        :return: Tuple of attention outputs and weights
        """

        # --------- #
        # Attention #
        # --------- #
        # query as the central window slice
        Q = window_tokens_Q[:, current_slice_index, :].unsqueeze(1)  # B x 1 x hidden_dim
        # key and value are the entire window
        K = window_tokens_K  # B x K x hidden_dim
        V = window_tokens_V  # B x K x hidden_dim

        # attention
        attention_output, attention_weights = self.attention(Q, K, V)  # B x 1 x hidden_dim

        return attention_output, attention_weights
