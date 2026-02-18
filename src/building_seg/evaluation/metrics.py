"""Evaluation metrics for building segmentation."""

import json
from pathlib import Path
from typing import Dict

import numpy as np


def compute_iou(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Intersection over Union."""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return float((intersection + smooth) / (union + smooth))


def compute_f1(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute pixel-wise F1 score (Dice coefficient)."""
    tp = (pred * target).sum()
    fp = pred.sum() - tp
    fn = target.sum() - tp
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return float(2 * precision * recall / (precision + recall + smooth))


def compute_boundary_f1(
    pred: np.ndarray, target: np.ndarray, tolerance: int = 2
) -> float:
    """Compute boundary F1 score at given pixel tolerance.

    Evaluates how well predicted boundaries match ground truth boundaries.
    """
    import cv2

    # Extract boundaries
    pred_uint8 = (pred * 255).astype(np.uint8)
    target_uint8 = (target * 255).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred_boundary = cv2.morphologyEx(pred_uint8, cv2.MORPH_GRADIENT, kernel) > 0
    target_boundary = cv2.morphologyEx(target_uint8, cv2.MORPH_GRADIENT, kernel) > 0

    if pred_boundary.sum() == 0 and target_boundary.sum() == 0:
        return 1.0
    if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
        return 0.0

    # Distance transform
    pred_dist = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, 5)
    target_dist = cv2.distanceTransform((~target_boundary).astype(np.uint8), cv2.DIST_L2, 5)

    # Precision: fraction of predicted boundary pixels close to target boundary
    precision = np.mean(target_dist[pred_boundary] <= tolerance)
    # Recall: fraction of target boundary pixels close to predicted boundary
    recall = np.mean(pred_dist[target_boundary] <= tolerance)

    if precision + recall == 0:
        return 0.0

    return float(2 * precision * recall / (precision + recall))


def compute_all_metrics(pred: np.ndarray, target: np.ndarray) -> Dict:
    """Compute all segmentation metrics."""
    return {
        "iou": compute_iou(pred, target),
        "f1": compute_f1(pred, target),
        "boundary_f1_2px": compute_boundary_f1(pred, target, tolerance=2),
        "boundary_f1_3px": compute_boundary_f1(pred, target, tolerance=3),
        "precision": float((pred * target).sum() / max(pred.sum(), 1e-6)),
        "recall": float((pred * target).sum() / max(target.sum(), 1e-6)),
    }


def save_results(results: Dict, output_path: str):
    """Save results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
