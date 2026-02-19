"""U-Net++ model with EfficientNet backbone for building segmentation."""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(config: dict) -> nn.Module:
    """Build U-Net++ segmentation model."""
    model = smp.UnetPlusPlus(
        encoder_name=config["model"]["encoder"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["model"]["in_channels"],
        classes=config["data"]["num_classes"],
        activation=None,  # Raw logits
    )
    return model
