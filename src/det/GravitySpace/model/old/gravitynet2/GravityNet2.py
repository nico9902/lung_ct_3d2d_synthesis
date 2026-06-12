from typing import Tuple

import torch
import torch.nn as nn

from src.det.GravitySpace.model.backbone.MyDenseNet_models import MyDenseNet_models
from src.det.GravitySpace.model.backbone.MyEfficientNetV2_models import MyEfficientNetV2_models
from src.det.GravitySpace.model.backbone.MyEfficientNet_models import MyEfficientNet_models
from src.det.GravitySpace.model.backbone.MyResNeXt_models import MyResNeXt_models
from src.det.GravitySpace.model.backbone.MyResNet_models import MyResNet_models
from src.det.GravitySpace.model.backbone.MySwin_models import MySwin_models
from src.det.GravitySpace.model.backbone.MySwinV2_models import MySwinV2_models
from src.det.GravitySpace.model.backbone.FeatureExtractor import FeatureExtractor
from src.det.GravitySpace.model.old.gravitynet2.ClassificationSubNet import ClassificationModel
from src.det.GravitySpace.model.old.gravitynet2.RegressionSubNet import RegressionModel
from src.det.GravitySpace.model.old.gravitynet2.SpaceAttentionV1 import SpaceAttentionV1
from src.det.GravitySpace.model.old.gravitynet2.SpaceAttentionV2 import SpaceAttentionV2
from src.det.GravitySpace.model.old.gravitynet2.SpaceAttentionV3 import SpaceAttentionV3
from src.det.GravitySpace.model.old.gravitynet2.SpaceAttentionV4 import SpaceAttentionV4
from src.det.GravitySpace.model.old.gravitynet2.SpaceAttentionV5 import SpaceAttentionV5


class GravityNet2(nn.Module):
    """
    Gravity Network for positional encoding on volume
    """

    def __init__(self,
                 backbone: str,
                 version: str,
                 pretrained: bool,
                 num_gravity_points_feature_map: int,
                 feature_map_shape: Tuple[int, int],
                 num_slices: int,
                 window_size: int,
                 sampling: int,
                 hidden_dim: int = 512):
        """
        __init__ method: run one when instantiating the object

        :param backbone: backbone
        :param pretrained: pretrained flag
        :param num_gravity_points_feature_map: num gravity points for feature map
        :param feature_map_shape: shape of feature map based on the image size
        :param window_size: number of slices of the window
        :param hidden_dim: hidden dim of the attention
        """

        super(GravityNet2, self).__init__()

        # PreTrained (True/False)
        self.pretrained = pretrained

        # num gravity points in feature map (reference window)
        self.num_gravity_points_feature_map = num_gravity_points_feature_map

        # feature map shape
        self.feature_map_shape = feature_map_shape

        # number of slices
        self.num_slices = num_slices

        # number of consecutive slices for cross-attention
        self.window_size = window_size

        # sampling step
        self.sampling = sampling

        # dimension for attention mechanism
        self.hidden_dim = hidden_dim

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

        # - SwinV2
        elif backbone.split('-')[0] == 'SwinV2':

            # Swin [T, S, B]
            swinv2 = str(backbone.split('-')[1])

            # SwinV2 Model
            self.backboneModel, self.num_features = MySwinV2_models(swinv2=swinv2,
                                                                    pretrained=self.pretrained)

        # ----------------- #
        # Feature Extractor #
        # ----------------- #
        self.feature_extractor = FeatureExtractor(self.backboneModel)

        # ----------------- #
        # Token projections #
        # ----------------- #
        # projection layer
        self.query_projection = nn.Linear(self.num_features * self.feature_map_shape[0] * self.feature_map_shape[1], self.hidden_dim)
        self.key_projection = nn.Linear(self.num_features * self.feature_map_shape[0] * self.feature_map_shape[1], self.hidden_dim)
        self.value_projection = nn.Linear(self.num_features * self.feature_map_shape[0] * self.feature_map_shape[1], self.hidden_dim)

        # inverse projection layer
        self.inverse_projection = nn.Linear(self.hidden_dim, self.num_features * self.feature_map_shape[0] * self.feature_map_shape[1])

        # positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(self.num_slices, self.hidden_dim))  # num_slices x hidden_dim

        # Xavier initialization
        torch.manual_seed(seed=0)
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.key_projection.weight)
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.xavier_uniform_(self.inverse_projection.weight)
        nn.init.xavier_uniform_(self.positional_encoding)

        # --------------- #
        # Space Attention #
        # --------------- #
        self.version = version

        if self.version == 'V1' or self.version == 'V1_1':

            self.space_attention = SpaceAttentionV1(hidden_dim=self.hidden_dim)

        elif self.version == 'V2' or self.version == 'V2_1':

            self.space_attention = SpaceAttentionV2(hidden_dim=self.hidden_dim)

        elif self.version == 'V3':

            self.space_attention = SpaceAttentionV3(hidden_dim=self.hidden_dim)

        elif self.version == 'V4':

            self.space_attention = SpaceAttentionV4(hidden_dim=self.hidden_dim)

        elif self.version == 'V5':

            self.space_attention = SpaceAttentionV5(hidden_dim=self.hidden_dim,
                                                    num_features=self.num_features)



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

        # ---------------------- #
        # Slice Token Projection #
        # ---------------------- #
        # projection to token on the entire volume
        tokens_Q = self.query_projection(features.reshape(B, S, -1))  # B x S x hidden_dim
        tokens_K = self.key_projection(features.reshape(B, S, -1))  # B x S x hidden_dim
        tokens_V = self.value_projection(features.reshape(B, S, -1))  # B x S x hidden_dim

        # ------------------- #
        # Positional Encoding #
        # ------------------- #
        # add positional encoding on the entire volume
        tokens_Q += self.positional_encoding.unsqueeze(0).expand(B, -1, -1)
        tokens_K += self.positional_encoding.unsqueeze(0).expand(B, -1, -1)
        tokens_V += self.positional_encoding.unsqueeze(0).expand(B, -1, -1)

        # --------------------- #
        # Cross-Slice Attention #
        # --------------------- #
        # initialize lists to store outputs
        classification_outputs = []
        regression_outputs = []

        # iterate over each slice to calculate attention and make predictions
        for i in range(0, S, self.sampling): # from 0 to S with sampling step

            # --------- #
            # Windowing #
            # --------- #
            # window range cannot go under 0 or over S
            start = max(0, min(i - self.window_size // 2, S - self.window_size))
            end = start + self.window_size

            # slice the features for the current window
            window_tokens_Q = tokens_Q[:, start:end]  # B x K x hidden_dim
            window_tokens_K = tokens_K[:, start:end]  # B x K x hidden_dim
            window_tokens_V = tokens_V[:, start:end]  # B x K x hidden_dim

            # --------------- #
            # Space Attention #
            # --------------- #
            if self.version == 'V5':
                attention_output, attention_weights = self.space_attention(window_tokens_Q, i - start, B)  # B x K x hidden_dim, B x K x K
            else:
                attention_output, attention_weights = self.space_attention(window_tokens_Q, window_tokens_K, window_tokens_V, i - start)  # B x K x hidden_dim, B x K x K

            # note: K = 1 for versions V4 and V5

            # inverse projection to the original token dimension
            attention_output = self.inverse_projection(attention_output)  # from B x K x hidden_dim to B x K x (F * H_FM * W_FM)

            # reshape to original slice feature dimensions B x F x H_FM x W_FM

            # note: since the encoding is done "outside the space attention modules" the attention output is managed here based on the different versions
            # summary:
            # - V4 and V5 output B x 1 x hidden_dim instead of B x K x hidden_dim, thus the reshape do not include the window size
            # - V1_1 and V2_1 apply sum along the sequence on the attention output

            # handle different reshape for B x 1 x hidden_dim and B x K x hidden_dim
            if self.version == 'V4' or self.version == 'V5':  # V4 and V5 output only 1 token instead of the window size
                attention_output = attention_output.reshape(B, 1, self.num_features, self.feature_map_shape[0], self.feature_map_shape[1])  # B x 1 x F x H_FM x W_FM

                # take the output corresponding to the current slice
                attention_output = attention_output.squeeze(1)  # from # B x 1 x F x H_FM x W_FM to B x F x H_FM x W_FM
            else:
                attention_output = attention_output.reshape(B, self.window_size, self.num_features, self.feature_map_shape[0], self.feature_map_shape[1])  # B x K x F x H_FM x W_FM

                # handle different attention output managing (sum or central window slice extraction)
                if self.version == 'V1_1' or self.version == 'V2_1':
                    # take the output sum across sequence
                    attention_output = attention_output.sum(dim=1).squeeze(1)  # from B x K x F x H_FM x W_FM to B x F x H_FM x W_FM
                else:
                    # take the output corresponding to the current slice
                    attention_output = attention_output[:, i - start, :, :, :]  # from B x K x F x H_FM x W_FM to B x F x H_FM x W_FM

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
