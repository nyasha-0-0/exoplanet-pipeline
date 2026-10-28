"""
baseline_model.py — Phase 9: High-Accuracy Baseline Model

Optimized XGBoost baseline with domain feature engineering,
validation threshold optimization for maximum classification accuracy,
and automatic interactive plot display.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
import features

logger = logging.getLogger(__name__)

RAW_FEATURE_COLUMNS = [
    "depth",
    "duration_hours",
    "period_days",
    "snr",
    "odd_even_depth_diff",
    "secondary_eclipse_depth",
]

BASELINE_RESULTS_PATH = config.DATA_DIR / "baseline_results.json"
BASELINE_MODEL_PATH = config.DATA_DIR / "baseline_model.joblib"
BASELINE_PLOT_PATH = config.DATA_DIR / "baseline_metrics_dashboard.png"


def ensure_features_exist() -> pd.DataFrame:
    path = config.DATA_DIR / "features.csv"
    if not path.exists():
        logger.info("features.csv not found. Running feature extraction...")
        catalog_df = pd.read_csv(config.CATALOG_CSV)
        features_df = features.build_feature_table(catalog_df)
        features_df.to_csv(path, index=False)
        logger.info("Saved feature table -> %s", path)
    else:
        logger.info("Loaded existing features from %s", path)

    df = pd.read_csv(path)
    before = len(df)
    df = df.dropna(subset=RAW_FEATURE_COLUMNS)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d/%d rows with NaN features.", dropped, before)
    return df


def engineer_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Expands the 6 basic metrics with physical ratio features and
    log transforms to help the gradient booster achieve higher accuracy.
    """
    data = df[RAW_FEATURE_COLUMNS].copy()

    # Domain interaction features
    # 1. Duty cycle (fraction of orbit spent in transit)
    period_hours = np.maximum(data["period_days"].values * 24.0, 1e-4)
    data["duty_cycle"] = data["duration_hours"].values / period_hours

    # 2. Normalized odd/even difference (detects secondary eclipses from binary stars)
    depth_safe = np.maximum(data["depth"].values, 1.0)
    data["norm_odd_even_diff"] = data["odd_even_depth_diff"].values / depth_safe

    # 3. Log-compressed physical scales
    data["log_period"] = np.log1p(np.maximum(data["period_days"].values, 0))
    data["log_depth"] = np.log1p(np.maximum(data["depth"].values, 0))
    data["log_snr"] = np.log1p(np.maximum(data["snr"].values, 0))

    feature_names = list(data.columns)
    return data.to_numpy(), feature_names


def build_model(pos_weight: float):
    try:
        import xgboost as xgb

        logger.info("Using tuned regularized XGBoost for maximum accuracy.")
        return xgb.XGBClassifier(
            n_estimators=320,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.80,
            min_child_weight=3,
            reg_lambda=3.0,
            reg_alpha=0.2,
            # Dampened weight avoids flooding false positives, raising overall accuracy
            scale_pos_weight=max(1.0, np.sqrt(pos_weight)),
            eval_metric="logloss",
            random_state=config.RANDOM_SEED,
        )
    except ImportError:
        logger.warning("xgboost not installed — falling back to LogisticRegression.")
        return LogisticRegression(
            class_weight={0: 1.0, 1: pos_weight},
            max_iter=1000,
            random_state=config.RANDOM_SEED,
        )


def find_optimal_accuracy_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Finds the decision threshold on validation set that directly maximizes total accuracy."""
    best_thresh = 0.50
    best_acc = 0.0

    for thresh in np.arange(0.35, 0.75, 0.01):
        preds = (y_probs >= thresh).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = float(thresh)

    return round(best_thresh, 2)


def evaluate(y_true: np.ndarray, y_probs: np.ndarray, split_name: str, threshold: float = 0.5) -> dict:
    preds = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=[0, 1]).tolist()
    auc = roc_auc_score(y_true, y_probs) if len(set(y_true)) > 1 else float("nan")

    logger.info(
        "  %-5s (thresh=%.2f): accuracy=%.2f%% | f1=%.3f prec=%.3f rec=%.3f auc=%.3f | cm=%s",
        split_name,
        threshold,
        acc * 100,
        f1,
        prec,
        rec,
        auc,
        cm,
    )

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "auc": auc,
        "confusion_matrix": cm,
        "threshold": threshold,
    }


def plot_metrics(model, y_test: np.ndarray, test_probs: np.ndarray, threshold: float, feature_names: list[str]):
    test_preds = (test_probs >= threshold).astype(int)
    test_acc = accuracy_score(y_test, test_preds) * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Baseline Model (XGBoost) — Test Accuracy: {test_acc:.2f}% (Threshold = {threshold:.2f})",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        test_preds,
        display_labels=["False Positive", "Candidate/Confirmed"],
        cmap="Blues",
        ax=axes[0],
        colorbar=False,
    )
    axes[0].set_title(f"Test Confusion Matrix\nAccuracy: {test_acc:.2f}%")

    # 2. ROC Curve
    disp = RocCurveDisplay.from_predictions(
        y_test,
        test_probs,
        name="XGBoost Baseline",
        ax=axes[1],
    )
    if disp.line_ is not None:
        disp.line_.set_color("#2b5c8f")

    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC = 0.50)")
    axes[1].set_title("Test ROC Curve")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right")

    # 3. Feature Importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)
        axes[2].barh(range(len(sorted_idx)), importances[sorted_idx], color="#4682b4", align="center")
        axes[2].set_yticks(range(len(sorted_idx)))
        axes[2].set_yticklabels([feature_names[i] for i in sorted_idx])
        axes[2].set_xlabel("Relative Importance")
        axes[2].set_title("Feature Importances")
        axes[2].grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(BASELINE_PLOT_PATH, dpi=200)
    logger.info("Saved evaluation dashboard -> %s", BASELINE_PLOT_PATH)

    # Pop up interactive window immediately
    plt.show()
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = ensure_features_exist()
    logger.info("Loaded %d usable feature rows.", len(df))

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    X_train, feature_names = engineer_features(train_df)
    y_train = train_df["label"].to_numpy()

    X_val, _ = engineer_features(val_df)
    y_val = val_df["label"].to_numpy()

    X_test, _ = engineer_features(test_df)
    y_test = test_df["label"].to_numpy()

    logger.info("Train: %d | Val: %d | Test: %d", len(X_train), len(X_val), len(X_test))

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    logger.info("Class balance (train): %d pos / %d neg -> raw pos_weight=%.3f", n_pos, n_neg, pos_weight)

    model = build_model(pos_weight)
    model.fit(X_train, y_train)

    train_probs = model.predict_proba(X_train)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    best_thresh = find_optimal_accuracy_threshold(y_val, val_probs)
    logger.info("Optimal accuracy threshold tuned on validation: %.2f", best_thresh)

    results = {
        "train": evaluate(y_train, train_probs, "train", threshold=best_thresh),
        "val": evaluate(y_val, val_probs, "val", threshold=best_thresh),
        "test": evaluate(y_test, test_probs, "test", threshold=best_thresh),
        "feature_columns": feature_names,
        "model_type": type(model).__name__,
        "optimal_threshold": best_thresh,
    }

    if hasattr(model, "feature_importances_"):
        importances = dict(zip(feature_names, model.feature_importances_.tolist()))
        results["feature_importances"] = importances
        logger.info("Feature importances: %s", importances)

    with open(BASELINE_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results -> %s", BASELINE_RESULTS_PATH)

    joblib.dump(model, BASELINE_MODEL_PATH)
    logger.info("Saved model -> %s", BASELINE_MODEL_PATH)

    plot_metrics(model, y_test, test_probs, best_thresh, feature_names)


if __name__ == "__main__":
    main()