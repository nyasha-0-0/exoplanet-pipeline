"""
visualizer.py — Live Training Visualizer

WHY a live plot instead of just printing numbers each epoch:
Numbers scrolling in a terminal are hard to spot trends in — is loss
actually decreasing, or did it plateau 10 epochs ago? A live-updating
plot lets you SEE overfitting (train loss down, val loss up) the moment
it starts, so you can stop training early instead of wasting time.

HOW: matplotlib interactive mode (plt.ion()) redraws the figure after
every epoch without blocking the training loop.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


class TrainingVisualizer:
    def __init__(self, save_path=None):
        self.save_path = save_path
        self.history = {
            "train_loss": [], "val_loss": [],
            "val_precision": [], "val_recall": [],
        }

        plt.ion()  # WHY: interactive mode lets the plot update without pausing execution
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 4))
        self.fig.suptitle("EXOSCOPE Training Progress")

    def update(self, epoch, train_loss, val_loss, val_precision, val_recall):
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_precision"].append(val_precision)
        self.history["val_recall"].append(val_recall)

        epochs = range(1, len(self.history["train_loss"]) + 1)

        # --- Panel 1: Loss curves ---
        ax = self.axes[0]
        ax.clear()
        ax.plot(epochs, self.history["train_loss"], label="train loss")
        ax.plot(epochs, self.history["val_loss"], label="val loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("BCE loss")
        ax.set_title("Loss (watch for val loss rising = overfitting)")
        ax.legend()

        # --- Panel 2: Precision/Recall curves ---
        ax = self.axes[1]
        ax.clear()
        ax.plot(epochs, self.history["val_precision"], label="val precision")
        ax.plot(epochs, self.history["val_recall"], label="val recall")
        ax.axhline(0.98, color="gray", linestyle="--", linewidth=0.8, label="target precision")
        ax.axhline(0.95, color="gray", linestyle=":", linewidth=0.8, label="target recall")
        ax.set_xlabel("epoch")
        ax.set_ylim(0, 1.05)
        ax.set_title("Precision / Recall vs target benchmarks")
        ax.legend(fontsize=7)

        # --- Panel 3: placeholder, filled in at the end by plot_confusion_matrix ---
        self.axes[2].set_title("Confusion matrix (shown after training)")

        plt.tight_layout()
        plt.pause(0.01)  # WHY: yields control briefly so the GUI event loop can redraw

    def plot_confusion_matrix(self, y_true, y_pred):
        """
        WHY this matters more than a single accuracy number: on a ~2-3%
        positive dataset, the confusion matrix immediately shows whether
        the model is actually catching real planets or just always
        predicting 'not planet' (which would still show high accuracy).
        """
        cm = confusion_matrix(y_true, y_pred)
        ax = self.axes[2]
        ax.clear()
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred: FP", "Pred: Planet"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True: FP", "True: Planet"])
        ax.set_title("Confusion Matrix (validation set)")
        plt.tight_layout()
        plt.pause(0.01)

    def finalize(self):
        plt.ioff()
        if self.save_path:
            self.fig.savefig(self.save_path, dpi=120)
            print(f"Saved training plot to {self.save_path}")
        plt.show()