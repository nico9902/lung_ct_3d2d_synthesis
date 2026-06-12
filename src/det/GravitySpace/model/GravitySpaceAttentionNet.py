from typing import Tuple

import torch
from torch import nn

from src.det.GravitySpace.model.attention.EnhancedMSA import EnhancedMSA
from src.det.GravitySpace.model.attention.StandardMSA import StandardMSA
from src.det.GravitySpace.model.attention.SummedMSA import SummedMSA
from src.det.GravitySpace.model.backbone.FeatureExtractor import FeatureExtractor
from src.det.GravitySpace.model.backbone.MyDenseNet_models import MyDenseNet_models
from src.det.GravitySpace.model.backbone.MyEfficientNetV2_models import MyEfficientNetV2_models
from src.det.GravitySpace.model.backbone.MyEfficientNet_models import MyEfficientNet_models
from src.det.GravitySpace.model.backbone.MyResNeXt_models import MyResNeXt_models
from src.det.GravitySpace.model.backbone.MyResNet_models import MyResNet_models
from src.det.GravitySpace.model.backbone.MySwin_models import MySwin_models
from src.det.GravitySpace.model.gravitynet.ClassificationSubNet import ClassificationModel
from src.det.GravitySpace.model.gravitynet.RegressionSubNet import RegressionModel


class GravitySpaceAttentionNet(nn.Module):
    """
    Gravity Space Attention Network
    """

    def __init__(self,
                 backbone: str,
                 pretrained: bool,
                 attention: str,
                 num_gravity_points_feature_map: int,
                 feature_map_shape: Tuple[int, int],
                 window_size: int,
                 sampling: int,
                 hidden_dim: int = 512,
                 chunk_size: int = None,
                 input_channels: int = 3):
        """
        __init__ method: run one when instantiating the object

        :param backbone: backbone
        :param pretrained: pretrained flag
        :param attention: attention
        :param num_gravity_points_feature_map: num gravity points for feature map
        :param feature_map_shape: shape of feature map based on the image size
        :param window_size: number of slices of the window
        :param sampling: sampling step
        :param hidden_dim: hidden dim of the attention
        :param chunk_size: number of slices to process in each chunk for memory efficiency
        """

        super(GravitySpaceAttentionNet, self).__init__()

        # PreTrained (True/False)
        self.pretrained = pretrained

        # num gravity points in feature map (reference window)
        self.num_gravity_points_feature_map = num_gravity_points_feature_map

        # feature map shape
        self.feature_map_shape = feature_map_shape

        # number of consecutive slices for cross-attention
        self.window_size = window_size

        # sampling step
        self.sampling = sampling

        # dimension for attention mechanism
        self.hidden_dim = hidden_dim

        # chunk size for memory-efficient feature extraction
        self.chunk_size = chunk_size

        if input_channels != 3:
            raise ValueError("Backbone input is fixed to 3 channels: repeated 2D slice or 2.5D [i-1, i, i+1].")

        # channels consumed by the 2D backbone
        self.input_channels = input_channels

        # -------------- #
        # Backbone Model #
        # -------------- #
        # - ResNet
        if backbone.split('-')[0] == 'ResNet':

            # ResNet [18, 34, 50, 101, 152]
            resnet = int(backbone.split('-')[1])

            # ResNet Model
            self.backboneModel, self.num_features = MyResNet_models(resnet=resnet,
                                                                    pretrained=self.pretrained)

        # - ResNeXt
        elif backbone.split('-')[0] == 'ResNeXt':

            # ResNeXt [50_32x4d, 101_32x8d, 101_64x4d]
            resnext = str(backbone.split('-')[1])

            # ResNeXt Model
            self.backboneModel, self.num_features = MyResNeXt_models(resnext=resnext,
                                                                     pretrained=self.pretrained)

        # - DenseNet
        elif backbone.split('-')[0] == 'DenseNet':

            # DenseNet [121, 161, 169, 201]
            densenet = int(backbone.split('-')[1])

            # DenseNet Model
            self.backboneModel, self.num_features = MyDenseNet_models(densenet=densenet,
                                                                      pretrained=self.pretrained)

        # - EfficientNet
        elif backbone.split('-')[0] == 'EfficientNet':

            # EfficientNet [B0, B1, B2, B3, B4, B5, B6, B7]
            efficientnet = str(backbone.split('-')[1])

            # EfficientNet Model
            self.backboneModel, self.num_features = MyEfficientNet_models(efficientnet=efficientnet,
                                                                          pretrained=self.pretrained)

        # - EfficientNetV2
        elif backbone.split('-')[0] == 'EfficientNetV2':

            # EfficientNet [S, M, L]
            efficientnetv2 = str(backbone.split('-')[1])

            # EfficientNetV2 Model
            self.backboneModel, self.num_features = MyEfficientNetV2_models(efficientnetv2=efficientnetv2,
                                                                            pretrained=self.pretrained)

        # - Swin
        elif backbone.split('-')[0] == 'Swin':

            # Swin [T, S, B]
            swin = str(backbone.split('-')[1])

            # Swin Model
            self.backboneModel, self.num_features = MySwin_models(swin=swin,
                                                                  pretrained=self.pretrained)

        else:
            ValueError("ERROR: wrong Backbone")

        # ----------------- #
        # Feature Extractor #
        # ----------------- #
        self.feature_extractor = FeatureExtractor(self.backboneModel, chunk_size=self.chunk_size)

        # --------------------- #
        # Cross-Slice Attention #
        # --------------------- #
        if attention == "standard":
            self.space_attention = StandardMSA(num_features=self.num_features,
                                               feature_map_shape=self.feature_map_shape,
                                               hidden_dim=self.hidden_dim,
                                               window_size=self.window_size)

        elif attention == "summed":
            self.space_attention = SummedMSA(num_features=self.num_features,
                                             feature_map_shape=self.feature_map_shape,
                                             hidden_dim=self.hidden_dim,
                                             window_size=self.window_size)

        elif attention == "enhanced":
            self.space_attention = EnhancedMSA(num_features=self.num_features,
                                               feature_map_shape=self.feature_map_shape,
                                               hidden_dim=self.hidden_dim,
                                               window_size=self.window_size)

        else:
            ValueError("ERROR: wrong Cross-Slice Attention")

        # ----------------- #
        # Regression SubNet #
        # ----------------- #
        self.regressionModel = RegressionModel(num_features_in=self.num_features,
                                               num_gravity_points_feature_map=self.num_gravity_points_feature_map)

        # --------------------- #
        # Classification SubNet #
        # --------------------- #
        self.classificationModel = ClassificationModel(num_features_in=self.num_features,
                                                       num_classes=2,
                                                       num_gravity_points_feature_map=self.num_gravity_points_feature_map)

    def forward(self,
                images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward method: directly call a method in the class when an instance name is called

        :param images: images stack
        :return: classification subnet output,
                 regression subnet output
        """

        B, S, C, H, W = images.shape  # Batch, Number of Slices, Channels, Height, Width

        # ------------------------- #
        # Slice Features Extraction #
        # ------------------------- #
        # feature extraction
        features = self.feature_extractor(images)  # B x S x F x H_FM x W_FM

        # --------------------- #
        # Cross-Slice Attention #
        # --------------------- #
        # initialize lists to store outputs
        classification_outputs = []
        regression_outputs = []

        # iterate over each slice to calculate attention and make predictions
        for i in range(0, S, self.sampling):  # from 0 to S with sampling step

            # --------- #
            # Windowing #
            # --------- #
            # window range cannot go under 0 or over S
            start = max(0, min(i - self.window_size // 2, S - self.window_size))
            end = start + self.window_size

            # slice the features for the current window
            window_features = features[:, start:end]  # B x K x F x H_FM x W_FM

            # --------------- #
            # Space Attention #
            # --------------- #
            attention_output, attention_weights = self.space_attention(window_features, i - start, B)  # B x F x H_FM x W_FM,  B x K x K or B x K x 1 x 1 x 1 in V5

            # ----------------- #
            # Regression SubNet #
            # ----------------- #
            regression_output = self.regressionModel(attention_output)  # regression shape: B x A x 2

            # --------------------- #
            # Classification SubNet #
            # --------------------- #
            classification_output = self.classificationModel(attention_output)  # classification shape: B x A x 2

            # append to the list
            classification_outputs.append(classification_output)
            regression_outputs.append(regression_output)

        # stack the outputs over slices
        classification_outputs = torch.stack(classification_outputs, dim=1)  # B x S x A x 2
        regression_outputs = torch.stack(regression_outputs, dim=1)  # B x S x A x 2

        return classification_outputs, regression_outputs
