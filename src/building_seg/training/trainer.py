"""Training loop for building segmentation."""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader


class Trainer:
    """Training manager for segmentation."""

    def __init__(self, config: dict, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler(enabled=config["training"]["amp"])

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict:
        """Full training loop with warmup + cosine schedule."""
        model = model.to(self.device)
        cfg = self.config["training"]

        optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        warmup_steps = cfg["warmup_epochs"] * len(train_loader)
        total_steps = cfg["epochs"] * len(train_loader)
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

        best_iou = 0.0
        best_state = None
        history = {"train_loss": [], "val_loss": [], "val_iou": []}

        for epoch in range(cfg["epochs"]):
            # Train
            model.train()
            train_loss = 0.0
            n = 0
            for images, masks, boundary_weights in train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                boundary_weights = boundary_weights.to(self.device)

                optimizer.zero_grad()
                with autocast("cuda", enabled=cfg["amp"]):
                    logits = model(images)
                    loss = criterion(logits, masks, boundary_weights)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()

                train_loss += loss.item()
                n += 1

            train_loss /= max(n, 1)

            # Validate
            val_loss, val_iou = self._validate(model, val_loader, criterion)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_iou"].append(val_iou)

            if epoch % 10 == 0 or epoch == cfg["epochs"] - 1:
                print(
                    f"  Epoch {epoch+1}/{cfg['epochs']} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val IoU: {val_iou:.4f}"
                )

            if val_iou > best_iou:
                best_iou = val_iou
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)

        return {"history": history, "best_iou": best_iou}

    @torch.no_grad()
    def _validate(self, model, loader, criterion):
        model.eval()
        total_loss = 0.0
        total_iou = 0.0
        n = 0

        for images, masks, boundary_weights in loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            boundary_weights = boundary_weights.to(self.device)

            with autocast("cuda", enabled=self.config["training"]["amp"]):
                logits = model(images)
                loss = criterion(logits, masks, boundary_weights)

            preds = (torch.sigmoid(logits) > 0.5).float()
            iou = compute_iou(preds, masks)

            total_loss += loss.item()
            total_iou += iou
            n += 1

        return total_loss / max(n, 1), total_iou / max(n, 1)


def compute_iou(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute IoU for binary segmentation."""
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()
