"""
test.py — Comprehensive Evaluation & Visual Dashboard for Dual-View CNN
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

import config
from model import DualViewCNN
from train import TransitDataset, run_epoch, DEVICE

OPTIMAL_THRESH = 0.45


def plot_metrics_dashboard(labels, probs, preds, thresh, save_path):
    """Generates a 2x2 dashboard showing how the classifier makes decisions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EXOSCOPE — Unseen Test Set Evaluation Dashboard", fontsize=15, fontweight="bold")

    # --- Panel 1: Confusion Matrix ---
    cm = confusion_matrix(labels, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    ax = axes[0, 0]
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    quadrant_names = [["True Neg (FP)", "False Pos (Alarm)"],
                      ["False Neg (Missed)", "True Pos (Planet)"]]
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(
                j, i,
                f"{quadrant_names[i][j]}\n{cm[i, j]} ({cm_pct[i, j]:.1f}%)",
                ha="center", va="center", color=text_color, fontsize=10, fontweight="bold"
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred: False Positive", "Pred: Planet"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True: False Positive", "True: Planet"])
    ax.set_title(f"Confusion Matrix (Threshold = {thresh:.2f})")

    # --- Panel 2: ROC Curve ---
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    ax = axes[0, 1]
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Guess")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # --- Panel 3: Precision-Recall Curve ---
    precisions, recalls, pr_thresholds = precision_recall_curve(labels, probs)
    ax = axes[1, 0]
    ax.plot(recalls, precisions, color="purple", lw=2, label="PR Curve")
    
    # Mark current operating threshold
    curr_prec = precision_score(labels, preds, zero_division=0)
    curr_rec = recall_score(labels, preds, zero_division=0)
    ax.scatter([curr_rec], [curr_prec], color="red", s=80, zorder=5,
               label=f"Chosen Thresh ({thresh:.2f})\nP={curr_prec:.2f}, R={curr_rec:.2f}")
    ax.set_xlabel("Recall (Fraction of Real Planets Caught)")
    ax.set_ylabel("Precision (Fraction of Flagged Candidates that are Real)")
    ax.set_title("Precision-Recall Trade-off Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    # --- Panel 4: Probability Distribution ---
    ax = axes[1, 1]
    fp_probs = probs[labels == 0]
    planet_probs = probs[labels == 1]
    ax.hist(fp_probs, bins=30, alpha=0.6, color="steelblue", label="True False Positives (Label=0)", density=True)
    ax.hist(planet_probs, bins=30, alpha=0.6, color="crimson", label="True Planets (Label=1)", density=True)
    ax.axvline(thresh, color="black", linestyle="--", lw=2, label=f"Decision Boundary ({thresh:.2f})")
    ax.set_xlabel("Model Predicted Probability (Sigmoid Output)")
    ax.set_ylabel("Density")
    ax.set_title("Model Confidence Distribution")
    ax.legend(loc="upper center")
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    print(f"Saved metrics dashboard to: {save_path}")


def plot_sample_predictions(dataset, labels, probs, preds, save_path):
    """
    Plots Global View (2001 pts) and Local View (201 pts) for 1 example each of:
    True Positive, True Negative, False Positive, and False Negative.
    """
    categories = [
        ("True Positive (Correctly Caught Planet)", (labels == 1) & (preds == 1), "forestgreen"),
        ("True Negative (Correctly Filtered Noise)", (labels == 0) & (preds == 0), "royalblue"),
        ("False Positive (Noise Flagged as Planet)", (labels == 0) & (preds == 1), "darkorange"),
        ("False Negative (Missed Real Planet)", (labels == 1) & (preds == 0), "crimson"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle("How DualViewCNN Sees the Data: Global vs. Local Views", fontsize=15, fontweight="bold")

    for row_idx, (title, mask, color) in enumerate(categories):
        indices = np.where(mask)[0]
        ax_global = axes[row_idx, 0]
        ax_local = axes[row_idx, 1]

        if len(indices) == 0:
            ax_global.text(0.5, 0.5, "No examples in this category", ha="center", va="center")
            ax_local.text(0.5, 0.5, "No examples in this category", ha="center", va="center")
            continue

        # Pick the most confident example in each category
        if "Positive" in title.split("(")[0]:
            chosen_idx = indices[np.argmax(probs[indices])]
        else:
            chosen_idx = indices[np.argmin(probs[indices])]

        gv = dataset.global_view[chosen_idx]
        lv = dataset.local_view[chosen_idx]
        prob = probs[chosen_idx]

        # Global View (2001 bins)
        ax_global.plot(np.linspace(-0.5, 0.5, len(gv)), gv, color=color, lw=0.8)
        ax_global.set_title(f"{title} — Global View (2001 bins) | Prob={prob:.3f}", fontsize=10, fontweight="bold")
        ax_global.set_xlabel("Orbital Phase")
        ax_global.set_ylabel("Normalized Flux")
        ax_global.grid(alpha=0.3)

        # Local View (201 bins)
        ax_local.plot(np.linspace(-1.0, 1.0, len(lv)), lv, color=color, lw=1.5)
        ax_local.set_title(f"{title} — Local Transit Zoom (201 bins) | Prob={prob:.3f}", fontsize=10, fontweight="bold")
        ax_local.set_xlabel("Normalized Transit Window")
        ax_local.set_ylabel("Normalized Flux")
        ax_local.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path, dpi=150)
    print(f"Saved sample light-curve predictions to: {save_path}")


def main():
    print("Loading test dataset...")
    test_ds = TransitDataset(config.DATASET_NPZ, "test", augment=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    print("Loading best model weights...")
    model = DualViewCNN().to(DEVICE)
    model.load_state_dict(torch.load(config.DATA_DIR / "best_model.pt", map_location=DEVICE))

    criterion = torch.nn.BCEWithLogitsLoss()

    print("Running evaluation on the unseen test set...")
    test_loss, test_labels, test_probs = run_epoch(model, test_loader, criterion, optimizer=None)
    test_preds = (test_probs > OPTIMAL_THRESH).astype(int)

    # Calculate final metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)

    print("\n" + "=" * 40)
    print("FINAL TEST SET RESULTS")
    print("=" * 40)
    print(f"Total Examples : {len(test_ds)}")
    print(f"Accuracy       : {test_acc * 100:.2f}%")
    print(f"F1 Score       : {test_f1:.3f}")
    print(f"Precision      : {test_precision:.3f}")
    print(f"Recall         : {test_recall:.3f}")
    print("\nConfusion Matrix:")
    print("True: FP       True: Planet")
    print(confusion_matrix(test_labels, test_preds))
    print("=" * 40 + "\n")

    # Generate and save both visual dashboards
    dashboard_path = config.DATA_DIR / "test_metrics_dashboard.png"
    samples_path = config.DATA_DIR / "test_lightcurve_predictions.png"

    plot_metrics_dashboard(test_labels, test_probs, test_preds, OPTIMAL_THRESH, dashboard_path)
    plot_sample_predictions(test_ds, test_labels, test_probs, test_preds, samples_path)

    plt.show()


if __name__ == "__main__":
    main()