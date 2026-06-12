import sys
from typing import Tuple, Union

from torchvision import models
from torchvision.models import Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights

from src.det.GravitySpace.model.backbone.SwinV2.MySwinV2B import MySwinV2B
from src.det.GravitySpace.model.backbone.SwinV2.MySwinV2S import MySwinV2S
from src.det.GravitySpace.model.backbone.SwinV2.MySwinV2T import MySwinV2T
from src.det.GravitySpace.utility.msg.msg_error import msg_error


def MySwinV2_models(swinv2: str,
                    pretrained: bool) -> Tuple[Union[MySwinV2T, MySwinV2S, MySwinV2B], int]:
    """
    Get SwinV2 models

    :param swinv2: SwinV2 [T, S, B]
    :param pretrained: pretrained flag
    :return: SwinV2 model,
             num features
    """

    # ------ #
    # Swin-T #
    # ------ #
    if swinv2 == 'T':
        SwinV2_model = MySwinV2T()  # MySwinT model
        if pretrained:
            SwinV2_model.load_state_dict(models.swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1).state_dict())
        num_features = 768

    # ------ #
    # Swin-S #
    # ------ #
    elif swinv2 == 'S':
        SwinV2_model = MySwinV2S()  # MySwinS model
        if pretrained:
            SwinV2_model.load_state_dict(models.swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1).state_dict())
        num_features = 768

    # ------ #
    # Swin-S #
    # ------ #
    elif swinv2 == 'B':
        SwinV2_model = MySwinV2B()  # MySwinB model
        if pretrained:
            SwinV2_model.load_state_dict(models.swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1).state_dict())
        num_features = 1024

    else:
        str_err = msg_error(file=__file__,
                            variable=swinv2,
                            type_variable="SwinV2",
                            choices="[T, S, B]")
        sys.exit(str_err)

    return SwinV2_model, num_features
