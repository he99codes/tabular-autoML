"""
Feature Engineering: Auto-generates polynomial, interaction, log/sqrt features.
Optional feature selection via SelectKBest.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple

from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, f_regression


class FeatureEngineer:
    """Generates and optionally selects engineered features."""

    def __init__(
        self,
        task_type: str,
        use_polynomial: bool = True,
        use_log: bool = True,
        use_sqrt: bool = True,
        use_interactions: bool = True,
        select_k: Optional[int] = None,
        poly_degree: int = 2,
    ):
        self.task_type = task_type
        self.use_polynomial = use_polynomial
        self.use_log = use_log
        self.use_sqrt = use_sqrt
        self.use_interactions = use_interactions
        self.select_k = select_k
        self.poly_degree = poly_degree

        self._poly: Optional[PolynomialFeatures] = None
        self._selector: Optional[SelectKBest] = None
        self._n_original: int = 0
        self._fitted = False

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._n_original = X.shape[1]
        X_out = self._generate(X, fit=True)

        if self.select_k and self.select_k < X_out.shape[1]:
            score_func = f_classif if self.task_type == "classification" else f_regression
            self._selector = SelectKBest(score_func=score_func, k=self.select_k)
            X_out = self._selector.fit_transform(X_out, y)

        self._fitted = True
        return X_out.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_out = self._generate(X, fit=False)
        if self._selector is not None:
            X_out = self._selector.transform(X_out)
        return X_out.astype(np.float32)

    def _generate(self, X: np.ndarray, fit: bool) -> np.ndarray:
        parts = [X]
        eps = 1e-6

        # Log transform (only positive values)
        if self.use_log:
            # Shift to positive before log
            X_shifted = X - X.min(axis=0) + eps
            log_feats = np.log1p(X_shifted)
            parts.append(log_feats)

        # Sqrt transform
        if self.use_sqrt:
            X_shifted = X - X.min(axis=0)
            sqrt_feats = np.sqrt(X_shifted)
            parts.append(sqrt_feats)

        # Polynomial features (degree 2 without bias)
        if self.use_polynomial or self.use_interactions:
            # Limit to first 20 features to avoid explosion
            X_sub = X[:, :min(20, X.shape[1])]
            if fit:
                self._poly = PolynomialFeatures(
                    degree=self.poly_degree,
                    interaction_only=not self.use_polynomial,
                    include_bias=False,
                )
                poly_feats = self._poly.fit_transform(X_sub)
            else:
                poly_feats = self._poly.transform(X_sub)

            # Remove original features (already in parts[0])
            n_orig_sub = X_sub.shape[1]
            poly_feats = poly_feats[:, n_orig_sub:]
            parts.append(poly_feats)

        X_out = np.hstack(parts)
        # Replace any NaN/inf
        X_out = np.nan_to_num(X_out, nan=0.0, posinf=0.0, neginf=0.0)
        return X_out

    @property
    def n_features_out(self) -> int:
        if not self._fitted:
            return 0
        # Approximate; actual is computed during fit
        return -1  # Will be set after fit
