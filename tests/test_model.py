"""Basic tests for building segmentation model."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from building_seg.models.model import build_model
from building_seg.models.components import CombinedLoss


def test_model_forward():
    config = {
        "model": {"architecture": "unetplusplus", "encoder": "efficientnet-b4",
                   "encoder_weights": "imagenet", "in_channels": 3},
        "data": {"num_classes": 1},
    }
    model = build_model(config)
    model.eval()
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1, 512, 512)


def test_combined_loss():
    criterion = CombinedLoss()
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    boundary = torch.rand(2, 1, 64, 64)
    loss = criterion(logits, targets, boundary)
    assert loss.shape == ()
    assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
