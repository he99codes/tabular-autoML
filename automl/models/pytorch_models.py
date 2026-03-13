"""
PyTorch neural network models for tabular data (optional - requires torch).
"""

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np
from typing import List, Optional


# ── Dataset ────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class TabularDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray, task_type: str):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = (torch.tensor(y, dtype=torch.long)
                      if task_type == "classification"
                      else torch.tensor(y, dtype=torch.float32))

        def __len__(self): return len(self.X)
        def __getitem__(self, idx): return self.X[idx], self.y[idx]

    # ── FeedforwardNN ──────────────────────────────────────────────────────

    class FeedforwardNN(nn.Module):
        def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.3, task_type="classification"):
            super().__init__()
            self.task_type = task_type
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x): return self.net(x)

    # ── ResidualMLP ────────────────────────────────────────────────────────

    class ResidualBlock(nn.Module):
        def __init__(self, dim, dropout):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(dim, dim), nn.BatchNorm1d(dim),
            )
            self.relu = nn.ReLU()

        def forward(self, x): return self.relu(x + self.block(x))

    class ResidualMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_blocks, output_dim, dropout=0.3, task_type="classification"):
            super().__init__()
            self.task_type = task_type
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
            )
            self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
            self.head = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            return self.head(self.blocks(self.input_proj(x)))

else:
    # Stubs when torch is not available
    class TabularDataset:
        def __init__(self, *a, **kw): raise ImportError("PyTorch not installed")

    class FeedforwardNN:
        def __init__(self, *a, **kw): raise ImportError("PyTorch not installed")

    class ResidualMLP:
        def __init__(self, *a, **kw): raise ImportError("PyTorch not installed")


def build_pytorch_model(model_name, input_dim, output_dim, task_type, config):
    if not HAS_TORCH:
        raise ImportError("PyTorch not installed")
    dropout = config.get("dropout", 0.3)
    if model_name == "FeedforwardNN":
        n_layers = config.get("n_layers", 3)
        hidden_dim = config.get("hidden_dim", 128)
        return FeedforwardNN(input_dim, [hidden_dim] * n_layers, output_dim, dropout, task_type)
    elif model_name == "ResidualMLP":
        return ResidualMLP(input_dim, config.get("hidden_dim", 128),
                           config.get("n_blocks", 3), output_dim, dropout, task_type)
    raise ValueError(f"Unknown model: {model_name}")


PYTORCH_SEARCH_SPACES = {
    "FeedforwardNN": {
        "lr": ("float_log", 1e-4, 1e-2),
        "hidden_dim": ("categorical", [64, 128, 256]),
        "n_layers": ("int", 2, 4),
        "dropout": ("float", 0.1, 0.5),
        "batch_size": ("categorical", [32, 64, 128]),
        "weight_decay": ("float_log", 1e-6, 1e-3),
    },
    "ResidualMLP": {
        "lr": ("float_log", 1e-4, 1e-2),
        "hidden_dim": ("categorical", [64, 128, 256]),
        "n_blocks": ("int", 2, 5),
        "dropout": ("float", 0.1, 0.5),
        "batch_size": ("categorical", [32, 64, 128]),
        "weight_decay": ("float_log", 1e-6, 1e-3),
    },
}
