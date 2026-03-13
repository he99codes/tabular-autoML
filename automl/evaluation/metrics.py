"""
Evaluation System: Computes metrics for classification and regression.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_classes: int = 2,
) -> Dict[str, float]:
    avg = "binary" if n_classes == 2 else "macro"
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    if y_proba is not None:
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            metrics["roc_auc"] = float(auc)
        except Exception:
            metrics["roc_auc"] = float("nan")
    return metrics


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def primary_metric(task_type: str, metrics: Dict[str, float]) -> float:
    """Return the single metric used for ranking (higher=better)."""
    if task_type == "classification":
        return metrics.get("roc_auc", metrics.get("f1", 0.0))
    else:
        # Negate RMSE so that higher = better
        return -metrics.get("rmse", float("inf"))


def format_metrics(metrics: Dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
