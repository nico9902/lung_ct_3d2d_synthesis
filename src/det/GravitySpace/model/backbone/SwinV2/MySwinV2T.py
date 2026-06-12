from torchvision.models.swin_transformer import SwinTransformer, SwinTransformerBlockV2, PatchMergingV2


class MySwinV2T(SwinTransformer):
    """
    My SwinV2-T
    """

    def __init__(self):

        super(MySwinV2T, self).__init__(patch_size=[4, 4],
                                        embed_dim=96,
                                        depths=[2, 2, 6, 2],
                                        num_heads=[3, 6, 12, 24],
                                        window_size=[8, 8],
                                        stochastic_depth_prob=0.2,
                                        block=SwinTransformerBlockV2,
                                        downsample_layer=PatchMergingV2)

    def forward(self, x):

        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # permute from B x H_FM x W_FM x F to B x F x H_FM x W_FM

        return x
