import sys
from typing import Tuple, Union

from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights

from src.det.GravitySpace.model.backbone.EfficientNetV2.MyEfficientNetV2L import MyEfficientNetV2L
from src.det.GravitySpace.model.backbone.EfficientNetV2.MyEfficientNetV2M import MyEfficientNetV2M
from src.det.GravitySpace.model.backbone.EfficientNetV2.MyEfficientNetV2S import MyEfficientNetV2S
from src.det.GravitySpace.utility.msg.msg_error import msg_error


def MyEfficientNetV2_models(efficientnetv2: str,
                            pretrained: bool,
                            in_channels: int = 3) -> Tuple[Union[MyEfficientNetV2L, MyEfficientNetV2M, MyEfficientNetV2S], int]:
    """
    Get EfficientNetV2 models

    :param efficientnetv2: EfficientNetV2 [S, M, L]
    :param pretrained: pretrained flag
    :return: EfficientNetV2 model,
             num features
    """

    # ---------------- #
    # EfficientNetV2-S #
    # ---------------- #
    if efficientnetv2 == 'S':
        EfficientNet_model = MyEfficientNetV2S()  # MyEfficientNetV2-S model
        if pretrained:
            EfficientNet_model.load_state_dict(models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1).state_dict())
        num_features = 1280  # num features out

    # ---------------- #
    # EfficientNetV2-M #
    # ---------------- #
    elif efficientnetv2 == 'M':
        EfficientNet_model = MyEfficientNetV2M()  # MyEfficientNetV2-M model
        if pretrained:
            EfficientNet_model.load_state_dict(models.efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1).state_dict())
        num_features = 1280  # num features out

    # ---------------- #
    # EfficientNetV2-L #
    # ---------------- #
    elif efficientnetv2 == 'L':
        EfficientNet_model = MyEfficientNetV2L()  # MyEfficientNetV2-L model
        if pretrained:
            EfficientNet_model.load_state_dict(models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1).state_dict())
        num_features = 1280  # num features out

    else:
        str_err = msg_error(file=__file__,
                            variable=efficientnetv2,
                            type_variable="EfficientNetV2",
                            choices="[S, M, L]")
        sys.exit(str_err)

    if in_channels != 3:
        raise ValueError("EfficientNetV2 backbones only support 3 input channels.")

    return EfficientNet_model, num_features
