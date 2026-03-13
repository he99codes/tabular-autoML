"""
Training Engine for PyTorch models (requires torch).
"""

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import time
import numpy as np
from typing import Optional, Dict


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False
        self.best_weights = None

    def step(self, loss, model):
        if not HAS_TORCH:
            return False
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore_best(self, model):
        if HAS_TORCH and self.best_weights:
            model.load_state_dict(self.best_weights)


class PyTorchTrainer:
    def __init__(self, model, task_type, n_classes=2, lr=1e-3, weight_decay=1e-4,
                 batch_size=64, max_epochs=100, patience=10, device=None):
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed.")
        self.model = model
        self.task_type = task_type
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)
        self.history = {"train_loss": [], "val_loss": []}

    def _loss_fn(self):
        return nn.CrossEntropyLoss() if self.task_type == "classification" else nn.MSELoss()

    def fit(self, train_dataset, val_dataset, time_budget=None):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_dataset,   batch_size=self.batch_size * 2, shuffle=False, num_workers=0)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler  = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion  = self._loss_fn()
        stopper    = EarlyStopping(patience=self.patience)
        start      = time.time()

        for epoch in range(self.max_epochs):
            if time_budget and (time.time() - start) > time_budget:
                break
            self.model.train()
            tl = []
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(Xb)
                if self.task_type == "regression":
                    out = out.squeeze(-1)
                loss = criterion(out, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                tl.append(loss.item())

            vl = self._eval_loss(val_loader, criterion)
            self.history["train_loss"].append(np.mean(tl))
            self.history["val_loss"].append(vl)
            scheduler.step(vl)
            if stopper.step(vl, self.model):
                break

        stopper.restore_best(self.model)
        return self

    def _eval_loss(self, loader, criterion):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                out = self.model(Xb)
                if self.task_type == "regression":
                    out = out.squeeze(-1)
                losses.append(criterion(out, yb).item())
        return float(np.mean(losses))

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        parts = []
        with torch.no_grad():
            for i in range(0, len(Xt), 512):
                parts.append(self.model(Xt[i:i+512]).cpu().numpy())
        return np.vstack(parts)

    def predict_proba(self, X):
        logits = self.predict(X)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_classes(self, X):
        return np.argmax(self.predict(X), axis=1)
