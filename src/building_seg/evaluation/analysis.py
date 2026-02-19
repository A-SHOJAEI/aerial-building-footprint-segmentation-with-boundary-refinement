"""Visualization utilities for building segmentation."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def plot_prediction_grid(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    predictions: List[np.ndarray],
    output_path: str,
    n_samples: int = 4,
):
    """Plot grid of image, ground truth, prediction, and overlay."""
    n = min(n_samples, len(images))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        # Original image
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title("Aerial Image")
        axes[i, 0].axis("off")

        # Ground truth
        axes[i, 1].imshow(masks[i], cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        # Prediction
        axes[i, 2].imshow(predictions[i], cmap="gray")
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

        # Overlay
        overlay = images[i].copy()
        overlay[predictions[i] > 0.5] = [255, 0, 0]
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title("Overlay")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_curves(history: Dict, output_path: str):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_iou"], color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("Validation IoU")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_bar(metrics: Dict, output_path: str):
    """Plot metrics as bar chart."""
    names = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=plt.cm.Set2(range(len(names))))
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Segmentation Metrics on Test Set")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=10)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
