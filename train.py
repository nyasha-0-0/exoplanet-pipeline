"""
train.py — Phase 10: High-Accuracy Dual-View CNN Training

Implements:
- Symmetric time-reversal data augmentation (physical transit invariance)
- Binary Focal Loss (focuses learning on hard astrophysical negatives)
- Linear warm-up + Cosine Annealing learning rate schedule
- Threshold selection via Youden's J-statistic on validation predictions
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import config
from model import DualViewCNN
from visualizer import TrainingVisualizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 60
BASE_LR = 1.2e-4
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 1e-3
GRAD_CLIP_MAX_NORM = 1.0


class TransitDataset(Dataset):
    def __init__(self, npz_path, split, augment=False):
        data = np.load(npz_path, allow_pickle=True)
        mask = data["split"] == split
        self.global_view = data["global_view"][mask].astype(np.float32)
        self.local_view = data["local_view"][mask].astype(np.float32)
        self.label = data["label"][mask].astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        gv = self.global_view[idx].copy()
        lv = self.local_view[idx].copy()

        if self.augment:
            # 1. Symmetric time reversal (ingress/egress symmetry)
            if np.random.rand() > 0.5:
                gv = np.flip(gv).copy()
                lv = np.flip(lv).copy()

            # 2. Phase-preserving micro-shift
            if np.random.rand() > 0.5:
                shift = np.random.randint(-3, 4)
                gv = np.roll(gv, shift)
                lv = np.roll(lv, max(-2, min(2, shift)))

            # 3. Fine Gaussian measurement noise
            if np.random.rand() > 0.5:
                gv += np.random.normal(0, 0.0008, size=gv.shape).astype(np.float32)

        return (
            torch.from_numpy(gv),
            torch.from_numpy(lv),
            torch.tensor(self.label[idx], dtype=torch.float32),
        )


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy non-transits, forcing gradients to prioritize
    hard false-positives and subtle transits.
    """
    def __init__(self, alpha=0.35, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        return torch.mean(focal_weight * bce)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_probs, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for global_view, local_view, label in loader:
            global_view = global_view.to(DEVICE)
            local_view = local_view.to(DEVICE)
            label = label.to(DEVICE)

            logits = model(global_view, local_view).squeeze(-1)
            loss = criterion(logits, label)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
                optimizer.step()

            total_loss += loss.item() * len(label)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(label.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_probs)


def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float, float]:
    """Finds decision threshold using Youden's J-statistic on the ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = float(np.clip(thresholds[best_idx], 0.35, 0.70))

    preds = (probs >= best_thresh).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    return round(best_thresh, 2), acc, f1


def main():
    training_start = time.time()

    train_ds = TransitDataset(config.DATASET_NPZ, "train", augment=True)
    val_ds = TransitDataset(config.DATASET_NPZ, "val", augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train candidates: {len(train_ds)} | Val candidates: {len(val_ds)}")

    model = DualViewCNN().to(DEVICE)
    criterion = BinaryFocalLoss(alpha=0.35, gamma=1.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    # Cosine Annealing with Warmup
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(max(1, WARMUP_EPOCHS))
        progress = float(epoch - WARMUP_EPOCHS) / float(max(1, EPOCHS - WARMUP_EPOCHS))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    viz = TrainingVisualizer(save_path=config.DATA_DIR / "training_curves.png")

    best_score = 0.0
    best_epoch_info = None
    final_val_labels, final_val_preds = [], []

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_labels, train_probs = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_labels, val_probs = run_epoch(model, val_loader, criterion, optimizer=None)

        best_thresh, val_acc, val_f1 = find_optimal_threshold(val_labels, val_probs)
        val_preds = (val_probs >= best_thresh).astype(int)

        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0.5

        scheduler.step()
        epoch_time = time.time() - epoch_start
        viz.update(epoch, train_loss, val_loss, val_precision, val_recall)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:2d}/{EPOCHS} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"| acc={val_acc*100:.2f}% | f1={val_f1:.3f} | prec={val_precision:.3f} rec={val_recall:.3f} "
            f"| auc={val_auc:.3f} | thresh={best_thresh:.2f} | lr={current_lr:.1e} ({epoch_time:.1f}s)"
        )

        # Composite check: 70% Accuracy + 30% AUC
        composite_score = 0.70 * val_acc + 0.30 * val_auc
        if composite_score > best_score:
            best_score = composite_score
            torch.save(model.state_dict(), config.DATA_DIR / "best_model.pt")
            best_epoch_info = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "val_auc": val_auc,
                "threshold": best_thresh,
            }
            final_val_labels = val_labels
            final_val_preds = val_preds
            print(f"  -> Checkpoint saved (Acc: {val_acc*100:.2f}%, AUC: {val_auc:.3f}, F1: {val_f1:.3f})")

    if len(final_val_labels) > 0:
        viz.plot_confusion_matrix(final_val_labels, final_val_preds)
    viz.finalize()

    total_time = time.time() - training_start
    print(f"\nTraining complete in {total_time/60:.1f} minutes.")
    if best_epoch_info:
        print(
            f"Best Checkpoint (Epoch {best_epoch_info['epoch']}): "
            f"Accuracy = {best_epoch_info['val_acc']*100:.2f}% | "
            f"ROC-AUC = {best_epoch_info['val_auc']:.3f} | "
            f"F1 = {best_epoch_info['val_f1']:.3f} | "
            f"Threshold = {best_epoch_info['threshold']:.2f}"
        )

    with open(config.DATA_DIR / "train_results.json", "w") as f:
        json.dump({"best_epoch": best_epoch_info, "total_time_minutes": total_time / 60}, f, indent=2)


if __name__ == "__main__":
    main()