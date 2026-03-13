import os, time, joblib
import numpy as np
from typing import Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def save_model(model: Any, path: str, model_type: str = "sklearn"):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if model_type == "pytorch" and HAS_TORCH:
        torch.save(model, path)
        print(f"  [Saved PyTorch model → {path}]")
    else:
        joblib.dump(model, path)
        print(f"  [Saved model → {path}]")


def load_model(path: str, model_type: str = "sklearn") -> Any:
    if model_type == "pytorch" and HAS_TORCH:
        return torch.load(path, map_location="cpu")
    return joblib.load(path)


def save_pipeline(pipeline: Any, path: str):
    joblib.dump(pipeline, path)
    print(f"  [Saved pipeline → {path}]")


class Timer:
    def __init__(self): self._start = time.time()
    def elapsed(self) -> float: return time.time() - self._start
    def remaining(self, budget: float) -> float: return max(0.0, budget - self.elapsed())
    def is_expired(self, budget: float) -> bool: return self.elapsed() >= budget


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
