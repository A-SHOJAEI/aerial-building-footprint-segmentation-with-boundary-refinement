"""Evaluate segmentation model on test set with TTA."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from building_seg.utils.config import load_config
from building_seg.data.preprocessing import get_file_pairs
from building_seg.models.model import build_model
from building_seg.evaluation.metrics import compute_all_metrics, save_results
from building_seg.evaluation.analysis import plot_prediction_grid, plot_training_curves, plot_metrics_bar

import albumentations as A
from albumentations.pytorch import ToTensorV2


def sliding_window_inference(
    model, image: torch.Tensor, tile_size: int = 512,
    overlap: int = 256, device: str = "cuda"
) -> np.ndarray:
    """Run inference with sliding window and Gaussian stitching."""
    _, H, W = image.shape
    stride = tile_size - overlap

    # Gaussian weight for blending
    sigma = tile_size / 6
    y_grid, x_grid = np.mgrid[0:tile_size, 0:tile_size]
    center = tile_size / 2
    gaussian = np.exp(-((x_grid - center) ** 2 + (y_grid - center) ** 2) / (2 * sigma ** 2))
    gaussian = torch.tensor(gaussian, dtype=torch.float32).to(device)

    pred_sum = torch.zeros(1, H, W, device=device)
    weight_sum = torch.zeros(1, H, W, device=device)

    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            tile = image[:, y:y + tile_size, x:x + tile_size].unsqueeze(0).to(device)
            with autocast("cuda"):
                logit = model(tile)
            prob = torch.sigmoid(logit).squeeze(0)
            pred_sum[:, y:y + tile_size, x:x + tile_size] += prob * gaussian
            weight_sum[:, y:y + tile_size, x:x + tile_size] += gaussian

    # Handle edges
    weight_sum = torch.clamp(weight_sum, min=1e-6)
    return (pred_sum / weight_sum).squeeze(0).cpu().numpy()


def tta_inference(model, image: torch.Tensor, **kwargs) -> np.ndarray:
    """Test-time augmentation with 8 transforms."""
    preds = []

    for k in range(4):  # 4 rotations
        rotated = torch.rot90(image, k, dims=[1, 2])
        pred = sliding_window_inference(model, rotated, **kwargs)
        pred = np.rot90(pred, -k)
        preds.append(pred)

        # Flipped version
        flipped = torch.flip(rotated, dims=[2])
        pred_f = sliding_window_inference(model, flipped, **kwargs)
        pred_f = np.flip(np.rot90(pred_f, -k), axis=1)
        preds.append(pred_f)

    return np.mean(preds, axis=0)


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model(config)
    ckpt_path = project_root / config["output"]["checkpoint_dir"] / "best_model.pt"
    if not ckpt_path.exists():
        print("No checkpoint found. Run scripts/train.py first.")
        sys.exit(1)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Load test data
    data_dir = project_root / config["data"]["data_dir"]
    test_pairs = get_file_pairs(
        str(data_dir / "test" / "sat"),
        str(data_dir / "test" / "map"),
    )

    if not test_pairs:
        print("No test data found.")
        sys.exit(1)

    print(f"Evaluating on {len(test_pairs)} test tiles...")

    normalize = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    all_metrics = []
    images_for_plot = []
    masks_for_plot = []
    preds_for_plot = []

    for img_path, mask_path in test_pairs:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        # Normalize
        transformed = normalize(image=image)
        image_tensor = transformed["image"]

        # Inference with TTA
        with torch.no_grad():
            if config["inference"]["tta"]:
                pred = tta_inference(
                    model, image_tensor,
                    tile_size=config["inference"]["tile_size"],
                    overlap=config["inference"]["overlap"],
                    device=str(device),
                )
            else:
                pred = sliding_window_inference(
                    model, image_tensor,
                    tile_size=config["inference"]["tile_size"],
                    overlap=config["inference"]["overlap"],
                    device=str(device),
                )

        pred_binary = (pred > 0.5).astype(np.float32)
        metrics = compute_all_metrics(pred_binary, mask)
        all_metrics.append(metrics)

        if len(images_for_plot) < 4:
            images_for_plot.append(image)
            masks_for_plot.append(mask)
            preds_for_plot.append(pred_binary)

        print(f"  {Path(img_path).name}: IoU={metrics['iou']:.4f}, F1={metrics['f1']:.4f}, "
              f"Boundary F1@2px={metrics['boundary_f1_2px']:.4f}")

    # Aggregate
    avg_metrics = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }

    print(f"\n{'='*60}")
    print("Average Test Metrics:")
    for key, val in avg_metrics.items():
        print(f"  {key}: {val['mean']:.4f} (+/- {val['std']:.4f})")

    # Save results
    results_dir = Path(config["output"]["results_dir"])
    save_results({"test_metrics": avg_metrics, "per_tile": all_metrics}, str(results_dir / "test_results.json"))

    # Plots
    if images_for_plot:
        plot_prediction_grid(images_for_plot, masks_for_plot, preds_for_plot,
                           str(results_dir / "prediction_grid.png"))

    plot_metrics_bar(
        {k: v["mean"] for k, v in avg_metrics.items()},
        str(results_dir / "metrics_bar.png"),
    )

    # Training curves
    history_path = results_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, str(results_dir / "training_curves.png"))


if __name__ == "__main__":
    main()
