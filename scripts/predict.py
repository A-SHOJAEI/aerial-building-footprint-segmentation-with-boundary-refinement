"""Run inference on a single aerial image."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from building_seg.utils.config import load_config
from building_seg.models.model import build_model

import albumentations as A
from albumentations.pytorch import ToTensorV2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to aerial image")
    parser.add_argument("--output", default="prediction.png", help="Output path")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config)
    ckpt = args.checkpoint or str(project_root / config["output"]["checkpoint_dir"] / "best_model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tensor = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        pred = torch.sigmoid(logit).squeeze().cpu().numpy()

    pred_binary = (pred > 0.5).astype(np.uint8) * 255
    cv2.imwrite(args.output, pred_binary)
    print(f"Prediction saved to {args.output}")


if __name__ == "__main__":
    main()
