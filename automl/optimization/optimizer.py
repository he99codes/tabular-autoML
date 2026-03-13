"""
Hyperparameter Optimization - uses Optuna if available, else random search.
"""

import time
import warnings
import numpy as np
from typing import Any, Dict, Optional, Callable
import random

warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def _random_suggest(search_space: Dict[str, tuple]) -> Dict[str, Any]:
    params = {}
    for name, spec in search_space.items():
        kind = spec[0]
        if kind == "int":
            params[name] = random.randint(spec[1], spec[2])
        elif kind == "int_none":
            params[name] = None if random.random() < 0.3 else random.randint(spec[1], spec[2])
        elif kind == "float":
            params[name] = random.uniform(spec[1], spec[2])
        elif kind == "float_log":
            log_val = random.uniform(np.log(spec[1]), np.log(spec[2]))
            params[name] = float(np.exp(log_val))
        elif kind == "categorical":
            params[name] = random.choice(spec[1])
    return params


def suggest_params(trial, search_space: Dict[str, tuple]) -> Dict[str, Any]:
    params = {}
    for name, spec in search_space.items():
        kind = spec[0]
        if kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "int_none":
            use_none = trial.suggest_categorical(f"{name}_none", [True, False])
            params[name] = None if use_none else trial.suggest_int(name, spec[1], spec[2])
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "float_log":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


class OptunaOptimizer:
    def __init__(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        search_space: Dict[str, tuple],
        n_trials: int = 20,
        direction: str = "maximize",
        time_budget: Optional[float] = None,
    ):
        self.objective_fn = objective_fn
        self.search_space = search_space
        self.n_trials = n_trials
        self.direction = direction
        self.time_budget = time_budget
        self._best_params: Dict[str, Any] = {}
        self._best_score = float("-inf") if direction == "maximize" else float("inf")

    def optimize(self) -> Dict[str, Any]:
        start = time.time()

        if HAS_OPTUNA and self.search_space:
            study = optuna.create_study(direction=self.direction)

            def wrapped(trial):
                if self.time_budget and (time.time() - start) > self.time_budget:
                    raise optuna.exceptions.OptunaError("budget")
                params = suggest_params(trial, self.search_space)
                return self.objective_fn(params)

            try:
                study.optimize(wrapped, n_trials=self.n_trials, catch=(Exception,))
                if study.best_trials:
                    self._best_params = study.best_params
                    self._best_score = study.best_value
            except Exception:
                pass
        elif self.search_space:
            # Random search fallback
            for _ in range(self.n_trials):
                if self.time_budget and (time.time() - start) > self.time_budget:
                    break
                params = _random_suggest(self.search_space)
                try:
                    score = self.objective_fn(params)
                    if score > self._best_score:
                        self._best_score = score
                        self._best_params = params
                except Exception:
                    pass

        return self._best_params

    @property
    def best_score(self) -> float:
        return self._best_score
