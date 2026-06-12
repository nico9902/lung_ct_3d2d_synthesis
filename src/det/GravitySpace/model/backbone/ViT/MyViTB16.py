import torch
from torchvision.models.vision_transformer import VisionTransformer


class MyViTB16(VisionTransformer):
    """
    My ViTB16
    """

    def __init__(self):

        super(MyViTB16, self).__init__(patch_size=16,
                                      num_layers=12,
                                      num_heads=12,
                                      hidden_dim=768,
                                      mlp_dim=3072)

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x
