from __future__ import annotations

import torch
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


def _head_prefixes(backbone: str) -> tuple[str, ...]:
    if backbone.startswith("resnet"):
        return ("fc.",)
    if backbone.startswith("densenet"):
        return ("classifier.",)
    if backbone.startswith("efficientnet"):
        return ("classifier.1.",)
    if backbone.startswith("vgg"):
        return ("classifier.6.",)
    return ()


def _is_head_parameter(name: str, backbone: str) -> bool:
    return name.startswith(_head_prefixes(backbone))


def _first_conv2d(model: nn.Module) -> tuple[nn.Module, str, nn.Conv2d] | None:
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            return model, name, child
        found = _first_conv2d(child)
        if found is not None:
            return found
    return None


def model_rgb2gray(model: nn.Module) -> nn.Module:
    """Convert the first convolution of a torchvision RGB backbone to grayscale."""
    found = _first_conv2d(model)
    if found is None:
        raise ValueError("Could not find a Conv2d layer to convert to grayscale input.")

    parent, name, conv = found
    if conv.in_channels == 1:
        return model
    if conv.in_channels != 3:
        raise ValueError(f"Expected first Conv2d to have 3 input channels, got {conv.in_channels}.")

    gray_conv = nn.Conv2d(
        in_channels=1,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        gray_conv.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        if conv.bias is not None:
            gray_conv.bias.copy_(conv.bias)
    gray_conv.weight.requires_grad = conv.weight.requires_grad
    if gray_conv.bias is not None and conv.bias is not None:
        gray_conv.bias.requires_grad = conv.bias.requires_grad
    setattr(parent, name, gray_conv)
    return model


def _freeze_backbone_parameters(model: nn.Module, backbone: str) -> None:
    for name, parameter in model.named_parameters():
        if not _is_head_parameter(name, backbone):
            parameter.requires_grad = False


def _freeze_first_parameter_groups(model: nn.Module, backbone: str, n_groups: int) -> None:
    if n_groups <= 0:
        return
    frozen = 0
    for name, parameter in model.named_parameters():
        if _is_head_parameter(name, backbone):
            continue
        parameter.requires_grad = False
        frozen += 1
        if frozen >= n_groups:
            break


def _freeze_backbone_unfreeze_last_parameter_groups(model: nn.Module, backbone: str, n_groups: int) -> None:
    if n_groups <= 0:
        return
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not _is_head_parameter(name, backbone)
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if _is_head_parameter(name, backbone)
    ]

    for parameter in backbone_parameters:
        parameter.requires_grad = False
    for parameter in backbone_parameters[-n_groups:]:
        parameter.requires_grad = True
    for parameter in head_parameters:
        parameter.requires_grad = True

    frozen = sum(parameter.numel() for parameter in backbone_parameters if not parameter.requires_grad)
    total = sum(parameter.numel() for parameter in backbone_parameters)
    print(
        f"Unfroze last {min(n_groups, len(backbone_parameters))} backbone parameter groups; "
        f"frozen {frozen:,}/{total:,} backbone weights ({100 * frozen / total:.1f}%)"
    )


def _freeze_half_backbone_parameters(model: nn.Module, backbone: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True

    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not _is_head_parameter(name, backbone)
    ]

    total_weights = sum(parameter.numel() for parameter in backbone_parameters)
    target = total_weights // 2
    frozen = 0

    for parameter in backbone_parameters:
        if frozen + parameter.numel() <= target:
            parameter.requires_grad = False
            frozen += parameter.numel()

    print(
        f"Frozen {frozen:,}/{total_weights:,} backbone weights "
        f"({100 * frozen / total_weights:.1f}%; target 50.0%)"
    )


def build_model(
    backbone: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    freeze_half_backbone: bool = False,
    freeze_first_layers: int = 0,
    unfreeze_last_layers: int = 0,
) -> nn.Module:
    if backbone not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Choose from: {', '.join(SUPPORTED_BACKBONES)}"
        )

    weights = _weights_enum(backbone, pretrained)
    model = getattr(models, backbone)(weights=weights)

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

    if freeze_backbone:
        _freeze_backbone_parameters(model, backbone)
    if freeze_half_backbone:
        _freeze_half_backbone_parameters(model, backbone)
    if freeze_first_layers > 0:
        _freeze_first_parameter_groups(model, backbone, freeze_first_layers)
    if unfreeze_last_layers > 0:
        _freeze_backbone_unfreeze_last_parameter_groups(model, backbone, unfreeze_last_layers)

    return model
