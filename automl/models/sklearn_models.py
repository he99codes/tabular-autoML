"""
Classical sklearn model candidates.
"""

from typing import Any, Dict
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def get_sklearn_candidates(task_type: str) -> Dict[str, Any]:
    candidates = {}
    if task_type == "classification":
        candidates["RandomForest"] = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        candidates["GradientBoosting"] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        candidates["LogisticRegression"] = LogisticRegression(max_iter=1000, random_state=42)
        if HAS_XGB:
            candidates["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)
        if HAS_LGBM:
            candidates["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    else:
        candidates["RandomForest"] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        candidates["GradientBoosting"] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        candidates["Ridge"] = Ridge()
        candidates["LinearRegression"] = LinearRegression()
        if HAS_XGB:
            candidates["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        if HAS_LGBM:
            candidates["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    return candidates


SKLEARN_SEARCH_SPACES = {
    "RandomForest": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 3, 20),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
    },
    "GradientBoosting": {
        "n_estimators": ("int", 50, 300),
        "max_depth": ("int", 2, 8),
        "learning_rate": ("float_log", 1e-3, 0.3),
        "subsample": ("float", 0.5, 1.0),
    },
    "LogisticRegression": {"C": ("float_log", 1e-4, 10.0)},
    "Ridge": {"alpha": ("float_log", 1e-3, 100.0)},
    "LinearRegression": {},
    "XGBoost": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 2, 10),
        "learning_rate": ("float_log", 1e-3, 0.3),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
    },
    "LightGBM": {
        "n_estimators": ("int", 50, 500),
        "max_depth": ("int", 2, 10),
        "learning_rate": ("float_log", 1e-3, 0.3),
        "num_leaves": ("int", 16, 256),
        "subsample": ("float", 0.5, 1.0),
    },
}
