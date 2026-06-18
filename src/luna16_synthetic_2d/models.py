from __future__ import annotations

import torch.nn as nn
from torchvision import models


SUPPORTED_BACKBONES = (
    "vgg16",
    "vgg19",
    "resnet18",
    "resnet34",
    "resnet50",
    "densenet121",
    "densenet169",
    "densenet201",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_v2_s",
)


def _weights_enum(backbone: str, pretrained: bool):
    if not pretrained:
        return None
    mapping = {
        "vgg16": models.VGG16_Weights.IMAGENET1K_V1,
        "vgg19": models.VGG19_Weights.IMAGENET1K_V1,
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V2,
        "densenet121": models.DenseNet121_Weights.IMAGENET1K_V1,
        "densenet169": models.DenseNet169_Weights.IMAGENET1K_V1,
        "densenet201": models.DenseNet201_Weights.IMAGENET1K_V1,
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "efficientnet_b1": models.EfficientNet_B1_Weights.IMAGENET1K_V2,
        "efficientnet_b2": models.EfficientNet_B2_Weights.IMAGENET1K_V1,
        "efficientnet_v2_s": models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
    }
    return mapping[backbone]


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def build_model(
    backbone: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Choose from: {', '.join(SUPPORTED_BACKBONES)}"
        )

    weights = _weights_enum(backbone, pretrained)
    model = getattr(models, backbone)(weights=weights)

    if freeze_backbone:
        _set_requires_grad(model, False)

    if backbone.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone.startswith("densenet"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif backbone.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone.startswith("vgg"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise AssertionError(f"Backbone handling missing for {backbone}")

    return model

