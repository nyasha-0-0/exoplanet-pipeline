"""
calibrate.py — Phase 11: Probability Calibration (Platt Scaling) with Metrics & Plots

1. Loads the trained DualViewCNN from best_model.pt.
2. Evaluates raw CNN scores vs. Platt-calibrated probabilities on the validation set.
3. Computes Accuracy, F1, Precision, Recall, AUC, and Confusion Matrices.
4. Generates data/calibration_dashboard.png and automatically opens the interactive plot window.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from model import DualViewCNN
from train import DEVICE, TransitDataset

logger = logging.getLogger(__name__)

CALIBRATOR_PATH = config.DATA_DIR / "calibrator.joblib"
CALIBRATION_REPORT_PATH = config.DATA_DIR / "calibration_report.json"
CALIBRATION_PLOT_PATH = config.DATA_DIR / "calibration_dashboard.png"


def load_trained_model() -> DualViewCNN:
    model = DualViewCNN().to(DEVICE)
    state_dict = torch.load(config.DATA_DIR / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def collect_raw_scores(model: DualViewCNN, dataset: TransitDataset) -> tuple[np.ndarray, np.ndarray]:
    scores, labels = [], []
    with torch.no_grad():
        for i in range(len(dataset)):
            global_view, local_view, label = dataset[i]
            global_view = global_view.unsqueeze(0).to(DEVICE)
            local_view = local_view.unsqueeze(0).to(DEVICE)

            logit = model(global_view, local_view)
            score = torch.sigmoid(logit).item()

            scores.append(score)
            labels.append(label.item())

    return np.array(scores), np.array(labels)


def fit_calibrator(raw_scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression()
    calibrator.fit(raw_scores[:, None], labels)
    return calibrator


def apply_calibration(calibrator: LogisticRegression, raw_scores: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(raw_scores[:, None])[:, 1]


def compute_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (y_probs >= threshold).astype(int)
    acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    cm = confusion_matrix(y_true, preds, labels=[0, 1]).tolist()
    auc = float(roc_auc_score(y_true, y_probs)) if len(set(y_true)) > 1 else float("nan")
    brier = float(brier_score_loss(y_true, y_probs))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "brier_score": brier,
        "confusion_matrix": cm,
    }


def plot_calibration_dashboard(
    labels: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    raw_metrics: dict,
    cal_metrics: dict,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dual-View CNN: Calibration & Validation Metrics (Threshold = 0.50)", fontsize=14, fontweight="bold")

    # 1. Confusion Matrix (Raw CNN)
    raw_cm = np.array(raw_metrics["confusion_matrix"])
    disp_raw = ConfusionMatrixDisplay(confusion_matrix=raw_cm, display_labels=["FP", "Planet"])
    disp_raw.plot(ax=axes[0, 0], cmap="Blues", colorbar=False)
    axes[0, 0].set_title(
        f"Raw CNN (Val Set)\nAcc: {raw_metrics['accuracy']*100:.1f}% | Prec: {raw_metrics['precision']:.3f} | Rec: {raw_metrics['recall']:.3f}"
    )

    # 2. Confusion Matrix (Calibrated CNN)
    cal_cm = np.array(cal_metrics["confusion_matrix"])
    disp_cal = ConfusionMatrixDisplay(confusion_matrix=cal_cm, display_labels=["FP", "Planet"])
    disp_cal.plot(ax=axes[0, 1], cmap="Greens", colorbar=False)
    axes[0, 1].set_title(
        f"Calibrated CNN (Val Set)\nAcc: {cal_metrics['accuracy']*100:.1f}% | Prec: {cal_metrics['precision']:.3f} | Rec: {cal_metrics['recall']:.3f}"
    )

    # 3. Reliability Diagram (Calibration Curve)
    CalibrationDisplay.from_predictions(labels, raw_scores, n_bins=5, strategy="quantile", name="Raw CNN", color="#e74c3c", ax=axes[1, 0])
    CalibrationDisplay.from_predictions(labels, calibrated_scores, n_bins=5, strategy="quantile", name="Calibrated CNN", color="#27ae60", ax=axes[1, 0])
    axes[1, 0].set_title(f"Reliability Curve\nBrier: {raw_metrics['brier_score']:.4f} -> {cal_metrics['brier_score']:.4f}")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 4. Confidence Score Distribution Histogram
    axes[1, 1].hist(raw_scores, bins=25, alpha=0.5, label="Raw CNN", color="#e74c3c", edgecolor="black")
    axes[1, 1].hist(calibrated_scores, bins=25, alpha=0.5, label="Calibrated", color="#27ae60", edgecolor="black")
    axes[1, 1].set_xlabel("Predicted Probability")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Probability Distribution Shift")
    axes[1, 1].legend(loc="upper center")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(CALIBRATION_PLOT_PATH, dpi=200)
    logger.info("Saved calibration dashboard -> %s", CALIBRATION_PLOT_PATH)

    # Pop-up interactive window immediately
    plt.show()
    plt.close()


def evaluate_calibration(raw_scores: np.ndarray, calibrated_scores: np.ndarray, labels: np.ndarray) -> dict:
    frac_pos_before, mean_pred_before = calibration_curve(labels, raw_scores, n_bins=5, strategy="quantile")
    frac_pos_after, mean_pred_after = calibration_curve(labels, calibrated_scores, n_bins=5, strategy="quantile")

    raw_metrics = compute_metrics(labels, raw_scores)
    cal_metrics = compute_metrics(labels, calibrated_scores)

    return {
        "raw_model_metrics": raw_metrics,
        "calibrated_model_metrics": cal_metrics,
        "improved": bool(cal_metrics["brier_score"] < raw_metrics["brier_score"]),
        "reliability_curve_before": {
            "mean_predicted": mean_pred_before.tolist(),
            "fraction_actually_positive": frac_pos_before.tolist(),
        },
        "reliability_curve_after": {
            "mean_predicted": mean_pred_after.tolist(),
            "fraction_actually_positive": frac_pos_after.tolist(),
        },
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Loading trained CNN from best_model.pt ...")
    model = load_trained_model()

    val_ds = TransitDataset(config.DATASET_NPZ, "val")
    logger.info("Collecting raw scores on %d validation examples ...", len(val_ds))
    raw_scores, labels = collect_raw_scores(model, val_ds)

    logger.info("Fitting Platt-scaling calibrator on validation scores ...")
    calibrator = fit_calibrator(raw_scores, labels)
    calibrated_scores = apply_calibration(calibrator, raw_scores)

    report = evaluate_calibration(raw_scores, calibrated_scores, labels)
    raw_m = report["raw_model_metrics"]
    cal_m = report["calibrated_model_metrics"]

    logger.info("--- VALIDATION EVALUATION COMPARISON ---")
    logger.info(
        "Raw CNN       : acc=%.3f f1=%.3f prec=%.3f rec=%.3f auc=%.3f brier=%.4f | cm=%s",
        raw_m["accuracy"], raw_m["f1"], raw_m["precision"], raw_m["recall"], raw_m["auc"], raw_m["brier_score"], raw_m["confusion_matrix"],
    )
    logger.info(
        "Calibrated CNN: acc=%.3f f1=%.3f prec=%.3f rec=%.3f auc=%.3f brier=%.4f | cm=%s",
        cal_m["accuracy"], cal_m["f1"], cal_m["precision"], cal_m["recall"], cal_m["auc"], cal_m["brier_score"], cal_m["confusion_matrix"],
    )

    joblib.dump(calibrator, CALIBRATOR_PATH)
    logger.info("Saved calibrator -> %s", CALIBRATOR_PATH)

    with open(CALIBRATION_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved calibration report -> %s", CALIBRATION_REPORT_PATH)

    plot_calibration_dashboard(labels, raw_scores, calibrated_scores, raw_m, cal_m)


def calibrated_probability(raw_sigmoid_score: float) -> float:
    calibrator = joblib.load(CALIBRATOR_PATH)
    return float(calibrator.predict_proba(np.array([[raw_sigmoid_score]]))[0, 1])


if __name__ == "__main__":
    main()