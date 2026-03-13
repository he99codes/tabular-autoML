"""Explainability: SHAP feature importance (optional)."""

import numpy as np
from typing import Any, Dict, List, Optional

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def compute_shap_importance(model, X, feature_names=None, model_type="tree",
                             task_type="classification", n_background=100):
    if not HAS_SHAP:
        print("  [SHAP not installed – using model feature_importances_ if available]")
        # Fallback: use sklearn feature_importances_ if available
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            names = feature_names or [f"f{i}" for i in range(len(imp))]
            d = dict(zip(names, imp.tolist()))
            return dict(sorted(d.items(), key=lambda x: -x[1]))
        if hasattr(model, "coef_"):
            imp = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
            names = feature_names or [f"f{i}" for i in range(len(imp))]
            d = dict(zip(names, imp.tolist()))
            return dict(sorted(d.items(), key=lambda x: -x[1]))
        return {}

    try:
        shap_values = None
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = np.mean([np.abs(s) for s in shap_values], axis=0)
        elif model_type == "linear":
            bg = X[:min(n_background, len(X))]
            explainer = shap.LinearExplainer(model, bg)
            shap_values = explainer.shap_values(X)
        else:
            bg = X[:min(n_background, len(X))]
            def pred_fn(x):
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(x)
                return model.predict(x)
            explainer = shap.KernelExplainer(pred_fn, bg)
            shap_values = explainer.shap_values(X[:min(50, len(X))], nsamples=50)
            if isinstance(shap_values, list):
                shap_values = np.mean([np.abs(s) for s in shap_values], axis=0)

        if shap_values is None:
            return {}
        mean_abs = np.abs(shap_values).mean(axis=0)
        names = feature_names or [f"f{i}" for i in range(len(mean_abs))]
        d = {n: float(v) for n, v in zip(names, mean_abs)}
        return dict(sorted(d.items(), key=lambda x: -x[1]))
    except Exception as e:
        print(f"  [SHAP error: {e}]")
        return {}


def print_feature_importance(importance: Dict[str, float], top_k: int = 15):
    if not importance:
        print("  No feature importance available.")
        return
    top = list(importance.items())[:top_k]
    max_val = max(v for _, v in top) or 1e-9
    print(f"\n  Top {len(top)} Feature Importances (SHAP / model-based):")
    for i, (name, val) in enumerate(top, 1):
        bar = "█" * max(1, int(val / max_val * 30))
        print(f"  {i:2d}. {name:<30s} {val:.4f} {bar}")
