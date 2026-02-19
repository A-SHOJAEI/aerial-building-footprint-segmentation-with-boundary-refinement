"""Train building segmentation model."""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from building_seg.utils.config import load_config
from building_seg.data.preprocessing import get_file_pairs
from building_seg.data.loader import create_dataloaders
from building_seg.models.model import build_model
from building_seg.models.components import CombinedLoss
from building_seg.training.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    config = load_config()
    set_seed(config["training"]["seed"])

    data_dir = project_root / config["data"]["data_dir"]

    # Get file pairs
    train_pairs = get_file_pairs(
        str(data_dir / "train" / "sat"),
        str(data_dir / "train" / "map"),
    )

    if len(train_pairs) == 0:
        print("No training data found. Run scripts/download_data.py first.")
        sys.exit(1)

    # Use official valid split if available, else split from train
    valid_sat = data_dir / "valid" / "sat"
    valid_map = data_dir / "valid" / "map"
    if valid_sat.exists() and valid_map.exists():
        val_pairs = get_file_pairs(str(valid_sat), str(valid_map))
    else:
        n_val = max(1, len(train_pairs) // 10)
        val_pairs = train_pairs[-n_val:]
        train_pairs = train_pairs[:-n_val]

    print(f"Training: {len(train_pairs)} tiles, Validation: {len(val_pairs)} tiles")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_pairs, val_pairs, config)

    # Build model
    model = build_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: U-Net++ with {config['model']['encoder']}")
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss
    criterion = CombinedLoss(
        bce_weight=config["loss"]["bce_weight"],
        dice_weight=config["loss"]["dice_weight"],
        boundary_weight=config["loss"]["boundary_weight"],
    )

    # Train
    trainer = Trainer(config)
    start_time = time.time()
    result = trainer.train(model, train_loader, val_loader, criterion)
    elapsed = time.time() - start_time

    print(f"\nTraining completed in {elapsed / 3600:.1f} hours")
    print(f"Best validation IoU: {result['best_iou']:.4f}")

    # Save checkpoint
    ckpt_dir = Path(config["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "best_model.pt")

    # Save training history
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(result["history"], f, indent=2)

    print("Training complete!")


if __name__ == "__main__":
    main()
