"""
Dataset Analyzer: Inspects CSV datasets and produces a summary report.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


class DatasetAnalyzer:
    """Automatically inspect a tabular dataset and produce a summary report."""

    def __init__(self, df: pd.DataFrame, target_col: str, task_type: str):
        self.df = df.copy()
        self.target_col = target_col
        self.task_type = task_type
        self.report: Dict[str, Any] = {}
        self.feature_types: Dict[str, str] = {}

    def analyze(self) -> Dict[str, Any]:
        """Run full analysis and return report dict."""
        self._basic_stats()
        self._detect_feature_types()
        self._missing_value_stats()
        self._class_imbalance()
        self._print_report()
        return self.report

    def _basic_stats(self):
        self.report["n_rows"] = len(self.df)
        self.report["n_cols"] = len(self.df.columns)
        self.report["target_col"] = self.target_col
        self.report["task_type"] = self.task_type
        feature_cols = [c for c in self.df.columns if c != self.target_col]
        self.report["feature_cols"] = feature_cols

    def _detect_feature_types(self):
        feature_cols = self.report["feature_cols"]
        numeric_cols, categorical_cols, text_cols = [], [], []

        for col in feature_cols:
            col_data = self.df[col].dropna()
            if col_data.empty:
                categorical_cols.append(col)
                continue

            dtype = self.df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                numeric_cols.append(col)
                self.feature_types[col] = "numeric"
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                # Heuristic: if avg string length > 30 → text
                avg_len = col_data.astype(str).str.len().mean()
                n_unique = col_data.nunique()
                if avg_len > 30 and n_unique > 50:
                    text_cols.append(col)
                    self.feature_types[col] = "text"
                else:
                    categorical_cols.append(col)
                    self.feature_types[col] = "categorical"
            else:
                categorical_cols.append(col)
                self.feature_types[col] = "categorical"

        self.report["numeric_cols"] = numeric_cols
        self.report["categorical_cols"] = categorical_cols
        self.report["text_cols"] = text_cols
        self.report["feature_types"] = self.feature_types

    def _missing_value_stats(self):
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        missing_report = {
            col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
            for col in self.df.columns
            if missing[col] > 0
        }
        self.report["missing_values"] = missing_report
        self.report["total_missing_cells"] = int(missing.sum())
        self.report["missing_pct_overall"] = round(
            missing.sum() / (len(self.df) * len(self.df.columns)) * 100, 2
        )

    def _class_imbalance(self):
        target = self.df[self.target_col]
        if self.task_type == "classification":
            counts = target.value_counts()
            ratios = (counts / len(target) * 100).round(2)
            imbalance_ratio = counts.max() / max(counts.min(), 1)
            self.report["class_distribution"] = ratios.to_dict()
            self.report["class_imbalance_ratio"] = round(float(imbalance_ratio), 2)
            self.report["is_imbalanced"] = imbalance_ratio > 3.0
            self.report["n_classes"] = int(target.nunique())
        else:
            self.report["target_stats"] = {
                "mean": float(target.mean()),
                "std": float(target.std()),
                "min": float(target.min()),
                "max": float(target.max()),
            }

    def _print_report(self):
        print("\n" + "=" * 60)
        print("         DATASET ANALYSIS REPORT")
        print("=" * 60)
        print(f"  Rows          : {self.report['n_rows']}")
        print(f"  Columns       : {self.report['n_cols']}")
        print(f"  Target        : {self.report['target_col']}")
        print(f"  Task          : {self.report['task_type']}")
        print(f"  Numeric cols  : {len(self.report['numeric_cols'])}")
        print(f"  Categorical   : {len(self.report['categorical_cols'])}")
        print(f"  Text cols     : {len(self.report['text_cols'])}")
        print(f"  Missing cells : {self.report['total_missing_cells']} ({self.report['missing_pct_overall']}%)")

        if self.task_type == "classification":
            print(f"  Classes       : {self.report['n_classes']}")
            print(f"  Imbalance ratio: {self.report['class_imbalance_ratio']}")
            if self.report["is_imbalanced"]:
                print("  ⚠ Dataset appears imbalanced!")
        else:
            stats = self.report["target_stats"]
            print(f"  Target mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        print("=" * 60 + "\n")
