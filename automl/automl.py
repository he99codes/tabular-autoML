"""
AutoML Core Orchestrator.
"""

import warnings
import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.base import clone

from .dataset_analyzer import DatasetAnalyzer
from .preprocessing import PreprocessingPipeline
from .feature_engineering import FeatureEngineer
from .models import (
    get_sklearn_candidates, SKLEARN_SEARCH_SPACES,
    build_pytorch_model, PYTORCH_SEARCH_SPACES,
)
from .optimization import OptunaOptimizer
from .evaluation import (
    evaluate_classification, evaluate_regression,
    primary_metric, format_metrics, Leaderboard,
)
from .explainability import compute_shap_importance, print_feature_importance
from .utils import save_model, save_pipeline, Timer, set_seed

try:
    from .models.pytorch_models import TabularDataset, HAS_TORCH
    from .training import PyTorchTrainer
except ImportError:
    HAS_TORCH = False
    TabularDataset = None
    PyTorchTrainer = None

warnings.filterwarnings("ignore")


class AutoML:
    """
    Modular Tabular AutoML framework combining classical ML and PyTorch models.

    Usage:
        automl = AutoML(task_type="classification", time_budget=300)
        automl.fit(df, target_col="churn")
        automl.report()
    """

    PYTORCH_MODELS = ["FeedforwardNN", "ResidualMLP"]

    def __init__(
        self,
        task_type: str = "classification",
        time_budget: Optional[float] = None,
        n_optuna_trials: int = 15,
        max_epochs: int = 80,
        output_dir: str = "./automl_output",
        seed: int = 42,
        val_size: float = 0.15,
        test_size: float = 0.15,
        use_feature_engineering: bool = True,
        verbose: bool = True,
    ):
        assert task_type in ("classification", "regression")
        self.task_type = task_type
        self.time_budget = time_budget or float("inf")
        self.n_optuna_trials = n_optuna_trials
        self.max_epochs = max_epochs
        self.output_dir = output_dir
        self.seed = seed
        self.val_size = val_size
        self.test_size = test_size
        self.use_feature_engineering = use_feature_engineering
        self.verbose = verbose

        self.leaderboard = Leaderboard()
        self.best_model: Any = None
        self.best_model_name: str = ""
        self.best_model_type: str = ""
        self.best_metrics: Dict[str, float] = {}
        self.feature_importance: Dict[str, float] = {}

        self._prep: Optional[PreprocessingPipeline] = None
        self._feat_eng: Optional[FeatureEngineer] = None
        self._timer = Timer()
        self._n_classes: int = 2
        self._feature_names: List[str] = []

        if HAS_TORCH:
            import torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = None

        set_seed(seed)

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, target_col: str) -> "AutoML":
        print(f"\n{'='*60}")
        print("  🚀 AutoML Training Started")
        print(f"  Device: {'CPU' if not HAS_TORCH else self._device}  |  Task: {self.task_type}")
        if self.time_budget < float("inf"):
            print(f"  Time budget: {self.time_budget}s")
        if not HAS_TORCH:
            print("  ℹ  PyTorch not found – skipping neural network models")
        print(f"{'='*60}\n")

        self._timer = Timer()

        # 1. Analyze
        analyzer = DatasetAnalyzer(df, target_col, self.task_type)
        report = analyzer.analyze()
        feature_cols = report["feature_cols"]
        self._n_classes = report.get("n_classes", 2)

        # 2. Split
        X, y = df[feature_cols], df[target_col]
        stratify = y if self.task_type == "classification" else None
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=stratify)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.seed,
            stratify=y_tmp if self.task_type == "classification" else None)
        print(f"  Split → train={len(X_train)}  val={len(X_val)}  test={len(X_test)}\n")

        # 3. Preprocessing
        self._prep = PreprocessingPipeline(
            numeric_cols=report["numeric_cols"],
            categorical_cols=report["categorical_cols"],
            text_cols=report["text_cols"],
            task_type=self.task_type,
        ).build(X_train)
        X_train_pp = self._prep.fit_transform(X_train, y_train)
        X_val_pp   = self._prep.transform(X_val)
        X_test_pp  = self._prep.transform(X_test)
        y_train_e  = self._prep.fit_transform_target(y_train)
        y_val_e    = self._prep.transform_target(y_val)
        y_test_e   = self._prep.transform_target(y_test)

        # 4. Feature engineering
        if self.use_feature_engineering:
            k = min(60, X_train_pp.shape[1] * 3)
            self._feat_eng = FeatureEngineer(
                task_type=self.task_type, select_k=k)
            X_train_fe = self._feat_eng.fit_transform(X_train_pp, y_train_e)
            X_val_fe   = self._feat_eng.transform(X_val_pp)
            X_test_fe  = self._feat_eng.transform(X_test_pp)
        else:
            X_train_fe, X_val_fe, X_test_fe = X_train_pp, X_val_pp, X_test_pp

        n_features = X_train_fe.shape[1]
        self._feature_names = [f"feature_{i}" for i in range(n_features)]
        print(f"  Feature matrix: {X_train_fe.shape}\n")

        # 5. Train sklearn models
        for name, base_model in get_sklearn_candidates(self.task_type).items():
            if self._timer.elapsed() >= self.time_budget:
                break
            self._train_sklearn(
                name, base_model, SKLEARN_SEARCH_SPACES.get(name, {}),
                X_train_fe, y_train_e, X_val_fe, y_val_e, X_test_fe, y_test_e)

        # 6. Train PyTorch models
        if HAS_TORCH:
            output_dim = self._n_classes if self.task_type == "classification" else 1
            for pt_name in self.PYTORCH_MODELS:
                if self._timer.elapsed() >= self.time_budget:
                    break
                self._train_pytorch(
                    pt_name, PYTORCH_SEARCH_SPACES[pt_name],
                    n_features, output_dim,
                    X_train_fe, y_train_e, X_val_fe, y_val_e, X_test_fe, y_test_e)

        # 7. Leaderboard
        self.leaderboard.rank()
        self.leaderboard.print()
        best = self.leaderboard.best()
        if best:
            self.best_model = best["_model"]
            self.best_model_name = best["model_name"]
            self.best_model_type = best.get("_type", "sklearn")
            self.best_metrics = {k: v for k, v in best.items()
                                 if k not in ("_model", "_type", "model_name", "primary_score")}
            print(f"  🏆 Best: {self.best_model_name}  {format_metrics(self.best_metrics)}\n")

        # 8. Explainability
        self._run_explainability(X_test_fe)

        # 9. Save
        self._save_artifacts()

        print(f"\n  ✅ Done in {self._timer.elapsed():.1f}s\n")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        feature_cols = (list(self._prep.pipeline.feature_names_in_)
                        if hasattr(self._prep.pipeline, "feature_names_in_") else df.columns.tolist())
        X_pp = self._prep.transform(df[feature_cols])
        if self._feat_eng:
            X_pp = self._feat_eng.transform(X_pp)
        if self.best_model_type == "pytorch":
            trainer = self.best_model
            if self.task_type == "classification":
                return trainer.predict_classes(X_pp)
            return trainer.predict(X_pp).squeeze()
        return self.best_model.predict(X_pp)

    def report(self):
        self.leaderboard.print()
        print(f"  Best model : {self.best_model_name}")
        print(f"  Metrics    : {format_metrics(self.best_metrics)}")
        print_feature_importance(self.feature_importance)

    # ─────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────

    def _time_rem(self, frac=1.0):
        return max(0.0, (self.time_budget - self._timer.elapsed()) * frac)

    def _eval(self, name, y_true, y_pred, y_proba):
        if self.task_type == "classification":
            return evaluate_classification(y_true, y_pred, y_proba, self._n_classes)
        return evaluate_regression(y_true, y_pred)

    # ── sklearn ──────────────────────────────────────────────────────

    def _train_sklearn(self, name, base_model, search_space,
                       X_tr, y_tr, X_va, y_va, X_te, y_te):
        print(f"  ▶ {name}")
        t0 = time.time()
        budget = min(self._time_rem(0.2), 60)

        if search_space and budget > 5:
            def objective(params):
                try:
                    mdl = clone(base_model).set_params(**params)
                    mdl.fit(X_tr, y_tr)
                    yp = mdl.predict(X_va)
                    ypr = mdl.predict_proba(X_va) if hasattr(mdl, "predict_proba") else None
                    m = self._eval(name, y_va, yp, ypr)
                    return primary_metric(self.task_type, m)
                except Exception:
                    return float("-inf")

            opt = OptunaOptimizer(objective, search_space,
                                  n_trials=self.n_optuna_trials,
                                  direction="maximize",
                                  time_budget=budget)
            best_p = opt.optimize()
            try:
                model = clone(base_model).set_params(**best_p)
            except Exception:
                model = clone(base_model)
        else:
            model = clone(base_model)

        model.fit(X_tr, y_tr)
        yp  = model.predict(X_te)
        ypr = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None
        m   = self._eval(name, y_te, yp, ypr)
        s   = primary_metric(self.task_type, m)
        print(f"    {format_metrics(m)}  [{time.time()-t0:.1f}s]")
        self.leaderboard.add(name, m, s, model, {"_type": "sklearn"})

    # ── pytorch ──────────────────────────────────────────────────────

    def _train_pytorch(self, name, search_space, input_dim, output_dim,
                       X_tr, y_tr, X_va, y_va, X_te, y_te):
        print(f"  ▶ {name} (PyTorch)")
        t0 = time.time()
        budget = self._time_rem(0.3)

        train_ds = TabularDataset(X_tr, y_tr, self.task_type)
        val_ds   = TabularDataset(X_va, y_va, self.task_type)

        def objective(params):
            try:
                mdl = build_pytorch_model(name, input_dim, output_dim, self.task_type, params)
                trainer = PyTorchTrainer(
                    mdl, self.task_type, self._n_classes,
                    lr=params.get("lr", 1e-3),
                    weight_decay=params.get("weight_decay", 1e-4),
                    batch_size=params.get("batch_size", 64),
                    max_epochs=min(self.max_epochs, 30),
                    patience=6, device=self._device)
                trainer.fit(train_ds, val_ds, time_budget=min(20, budget / 3))
                yp, ypr = self._pt_preds(trainer, X_va)
                m = self._eval(name, y_va, yp, ypr)
                return primary_metric(self.task_type, m)
            except Exception:
                return float("-inf")

        opt = OptunaOptimizer(objective, search_space,
                              n_trials=max(5, self.n_optuna_trials // 2),
                              direction="maximize",
                              time_budget=min(budget * 0.5, 90))
        best_p = opt.optimize()

        mdl = build_pytorch_model(name, input_dim, output_dim, self.task_type, best_p)
        trainer = PyTorchTrainer(
            mdl, self.task_type, self._n_classes,
            lr=best_p.get("lr", 1e-3),
            weight_decay=best_p.get("weight_decay", 1e-4),
            batch_size=best_p.get("batch_size", 64),
            max_epochs=self.max_epochs,
            patience=10, device=self._device)
        trainer.fit(train_ds, val_ds, time_budget=self._time_rem(0.4))

        yp, ypr = self._pt_preds(trainer, X_te)
        m  = self._eval(name, y_te, yp, ypr)
        s  = primary_metric(self.task_type, m)
        print(f"    {format_metrics(m)}  [{time.time()-t0:.1f}s]")
        self.leaderboard.add(name, m, s, trainer, {"_type": "pytorch"})

    def _pt_preds(self, trainer, X):
        if self.task_type == "classification":
            ypr = trainer.predict_proba(X)
            return np.argmax(ypr, axis=1), ypr
        yp = trainer.predict(X).squeeze()
        return yp, None

    def _run_explainability(self, X_test):
        print("  🔍 Computing SHAP feature importance...")
        if self.best_model is None:
            return
        from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
        shap_type = "pytorch" if self.best_model_type == "pytorch" else "tree"
        if self.best_model_type == "sklearn" and isinstance(
                self.best_model, (LinearRegression, LogisticRegression, Ridge)):
            shap_type = "linear"
        X_sample = X_test[:min(300, len(X_test))]
        self.feature_importance = compute_shap_importance(
            self.best_model, X_sample, self._feature_names, shap_type, self.task_type)
        print_feature_importance(self.feature_importance)

    def _save_artifacts(self):
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"\n  💾 Saving artifacts → {self.output_dir}/")
        if self.best_model is None:
            return
        if self.best_model_type == "pytorch":
            save_model(self.best_model.model,
                       os.path.join(self.output_dir, "best_model.pt"), "pytorch")
        else:
            save_model(self.best_model,
                       os.path.join(self.output_dir, "best_model.joblib"), "sklearn")
        save_pipeline(self._prep, os.path.join(self.output_dir, "preprocessing.joblib"))
        if self._feat_eng:
            save_pipeline(self._feat_eng, os.path.join(self.output_dir, "feature_engineering.joblib"))
        lb = os.path.join(self.output_dir, "leaderboard.csv")
        self.leaderboard.to_dataframe().to_csv(lb, index=False)
        print(f"  [Leaderboard → {lb}]")
