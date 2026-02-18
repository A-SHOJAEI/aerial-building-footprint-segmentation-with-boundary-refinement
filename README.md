# Aerial Building Footprint Segmentation with Boundary Refinement

A deep learning pipeline for segmenting building footprints from aerial/satellite imagery using **U-Net++** with an **EfficientNet-B4** encoder and a novel **boundary-weighted loss** that sharpens predictions along building edges.

## Architecture

```
Input (3×512×512)
    │
    ▼
┌──────────────────────────┐
│  EfficientNet-B4 Encoder │  (ImageNet pre-trained)
│  ─────────────────────── │
│  U-Net++ Nested Decoder  │  (Dense skip connections)
└──────────────────────────┘
    │
    ▼
Output (1×512×512)  →  Sigmoid  →  Binary Mask
```

**Key design decisions:**

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Encoder | EfficientNet-B4 | Strong accuracy/efficiency trade-off with compound scaling |
| Decoder | U-Net++ (nested) | Dense skip connections recover fine spatial detail |
| Loss | BCE (0.5) + Dice (0.3) + Boundary (0.2) | Multi-task loss: pixel accuracy + region overlap + edge sharpness |
| Boundary map | Distance-transform weight map | 5x loss amplification at building edges forces crisp boundaries |
| Inference | Sliding window + 8x TTA | Handles full 1500×1500 tiles via 512-crop overlap-blend + rotations/flips |

## Dataset

**Massachusetts Buildings Dataset** (Mnih, 2013) — aerial imagery of the Boston metro area.

| Split | Tiles | Resolution |
|-------|-------|------------|
| Train | 137 | 1500 × 1500 px (1 m/px) |
| Valid | 4 | 1500 × 1500 px |
| Test | 10 | 1500 × 1500 px |

Source: [University of Toronto](https://www.cs.toronto.edu/~vmnih/data/mass_buildings/)

## Results

Trained for **100 epochs** on a single NVIDIA RTX 4090. Best validation IoU: **0.665**.

**Test set performance** (10 tiles, sliding window + 8x TTA):

| Metric | Mean | Std |
|--------|------|-----|
| IoU | 0.351 | 0.050 |
| F1 Score | 0.517 | 0.056 |
| Boundary F1 @2px | 0.523 | 0.140 |
| Boundary F1 @3px | 0.580 | 0.145 |
| Precision | 0.868 | 0.031 |
| Recall | 0.371 | 0.054 |

> **Note:** The gap between validation IoU (0.665) and test IoU (0.351) reflects the Massachusetts Buildings test set's known difficulty — test tiles contain denser urban areas with more complex building layouts than the training distribution.

## Installation

```bash
git clone https://github.com/A-SHOJAEI/aerial-building-footprint-segmentation-with-boundary-refinement.git
cd aerial-building-footprint-segmentation-with-boundary-refinement

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Download data

```bash
python scripts/download_data.py
```

Downloads the Massachusetts Buildings dataset (~2 GB) from the University of Toronto servers.

### Train

```bash
python scripts/train.py
```

Configuration lives in `configs/default.yaml`. Key training settings:

- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: Cosine annealing with 5-epoch linear warmup
- Mixed precision (AMP) enabled
- Gradient clipping at 1.0
- 100 epochs, batch size 8

### Evaluate

```bash
python scripts/evaluate.py
```

Runs sliding-window inference with 8x test-time augmentation on the test split and produces:
- `results/test_results.json` — per-tile and aggregate metrics
- `results/prediction_grid.png` — visual comparison (image / ground truth / prediction / overlay)
- `results/training_curves.png` — loss and IoU over training
- `results/metrics_bar.png` — bar chart of test metrics

### Predict on a single image

```bash
python scripts/predict.py --image path/to/image.tiff --checkpoint checkpoints/best_model.pt
```

## Project Structure

```
├── configs/
│   └── default.yaml            # All hyperparameters
├── scripts/
│   ├── download_data.py        # Dataset download
│   ├── train.py                # Training entry point
│   ├── evaluate.py             # Evaluation with TTA
│   └── predict.py              # Single-image inference
├── src/building_seg/
│   ├── data/
│   │   ├── loader.py           # Dataset + DataLoader with boundary maps
│   │   └── preprocessing.py    # File pairing + distance-transform boundaries
│   ├── models/
│   │   ├── model.py            # U-Net++ builder (segmentation_models_pytorch)
│   │   └── components.py       # DiceLoss, BoundaryLoss, CombinedLoss
│   ├── training/
│   │   └── trainer.py          # Training loop (AMP, warmup, cosine schedule)
│   ├── evaluation/
│   │   ├── metrics.py          # IoU, F1, Boundary F1 @ tolerance
│   │   └── analysis.py         # Visualization utilities
│   └── utils/
│       └── config.py           # YAML config loader
├── tests/
│   └── test_model.py           # Unit tests
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## References

- Mnih, V. (2013). *Machine Learning for Aerial Image Labeling.* University of Toronto.
- Zhou, Z., et al. (2018). *UNet++: A Nested U-Net Architecture for Medical Image Segmentation.* DLMIA.
- Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for CNNs.* ICML.

## License

MIT
