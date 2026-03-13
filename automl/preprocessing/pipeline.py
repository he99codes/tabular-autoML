"""
Preprocessing Pipeline: Handles numeric, categorical, and text features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import scipy.sparse as sp

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from category_encoders import TargetEncoder
    HAS_TARGET_ENC = True
except ImportError:
    HAS_TARGET_ENC = False


class PreprocessingPipeline:
    """Auto-build preprocessing pipeline based on feature types."""

    def __init__(
        self,
        numeric_cols: List[str],
        categorical_cols: List[str],
        text_cols: List[str],
        task_type: str,
        cat_threshold: int = 15,
    ):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.text_cols = text_cols
        self.task_type = task_type
        self.cat_threshold = cat_threshold
        self.pipeline: Optional[ColumnTransformer] = None
        self.target_encoder: Optional[LabelEncoder] = None
        self._fitted = False

    def build(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        transformers = []

        if self.numeric_cols:
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("numeric", numeric_pipeline, self.numeric_cols))

        ohe_cols, te_cols = [], []
        for col in self.categorical_cols:
            n_unique = df[col].nunique()
            if n_unique <= self.cat_threshold or not HAS_TARGET_ENC:
                ohe_cols.append(col)
            else:
                te_cols.append(col)

        if ohe_cols:
            ohe_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("ohe_cat", ohe_pipeline, ohe_cols))

        if te_cols and HAS_TARGET_ENC:
            from category_encoders import TargetEncoder as TE
            te_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", TE()),
            ])
            transformers.append(("te_cat", te_pipeline, te_cols))
        elif te_cols:
            # Fallback: ordinal encode high-cardinality cats
            ord_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ])
            transformers.append(("ord_cat", ord_pipeline, te_cols))

        if self.text_cols:
            for col in self.text_cols:
                transformers.append((
                    f"text_{col}",
                    TfidfVectorizer(max_features=50, ngram_range=(1, 2)),
                    col,
                ))

        if not transformers:
            # Fallback: passthrough all columns as numeric
            transformers.append(("passthrough", "passthrough", list(df.columns)))

        self.pipeline = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.0,
        )
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        Xt = self.pipeline.fit_transform(X, y)
        if sp.issparse(Xt):
            Xt = Xt.toarray()
        self._fitted = True
        return np.nan_to_num(Xt.astype(np.float32))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.pipeline.transform(X)
        if sp.issparse(Xt):
            Xt = Xt.toarray()
        return np.nan_to_num(Xt.astype(np.float32))

    def fit_transform_target(self, y: pd.Series) -> np.ndarray:
        if self.task_type == "classification":
            self.target_encoder = LabelEncoder()
            return self.target_encoder.fit_transform(y)
        return y.values.astype(np.float32)

    def transform_target(self, y: pd.Series) -> np.ndarray:
        if self.task_type == "classification" and self.target_encoder:
            return self.target_encoder.transform(y)
        return y.values.astype(np.float32)

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        if self.task_type == "classification" and self.target_encoder:
            return self.target_encoder.inverse_transform(y)
        return y

    def get_categorical_dims(self, df: pd.DataFrame) -> Dict[str, int]:
        return {col: int(df[col].nunique()) + 1 for col in self.categorical_cols}
