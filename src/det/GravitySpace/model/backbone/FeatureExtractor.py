import torch
from torch import nn

class FeatureExtractor(nn.Module):
    """
    Extracts features for each batch using the backbone model.
    Supports optional chunked processing to reduce peak memory usage.
    """

    def __init__(self, backbone_model: nn.Module, chunk_size: int = None):
        super(FeatureExtractor, self).__init__()
        self.backbone_model = backbone_model
        self.chunk_size = chunk_size  # None = disable chunking, process all at once

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts features for a batch of images with optional chunking.

        :param images: Tensor of shape B x S x C x H x W
        :return: Extracted features of shape B x S x F x H_FM x W_FM
        """
        B, S = images.shape[0:2]

        # Reshape all images to process together: (B*S) x C x H x W
        B_S = B * S
        images_reshaped = images.reshape(B_S, *images.shape[2:])
        
        # Process all at once or in chunks
        if self.chunk_size is None:
            features_reshaped = self.backbone_model(images_reshaped)  # (B*S) x F x H_FM x W_FM
        else:
            features_list = []
            for chunk_start in range(0, B_S, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, B_S)
                chunk = images_reshaped[chunk_start:chunk_end]
                chunk_features = self.backbone_model(chunk)
                features_list.append(chunk_features)
            features_reshaped = torch.cat(features_list, dim=0)
        
        # Reshape back to batch format: B x S x F x H_FM x W_FM
        return features_reshaped.reshape(B, S, *features_reshaped.shape[1:])
