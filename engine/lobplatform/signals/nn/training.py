"""Training loop ported from research/lob/train.py.

Same design: windowed dataset over precomputed features, AdamW + cosine LR,
focal loss with inverse-frequency weights, early stopping on validation
macro-F1, grad clipping. Adapted: walk-forward day splits, TorchScript
export, metrics JSON — per §5.2.1.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .labels import INVALID
from .losses import FocalLoss, class_weights
from .models import build_model

SEQ_LEN = 64


class WindowDataset(Dataset):  # type: ignore[type-arg]
    """(window [T, F], label) at positions with valid labels + full windows."""

    def __init__(self, x: np.ndarray, y: np.ndarray, positions: list[int],
                 window: int = SEQ_LEN) -> None:
        self.x, self.y, self.window = x, y, window
        self.positions = [p for p in positions if p >= window - 1 and y[p] != INVALID]

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        t = self.positions[i]
        return torch.from_numpy(self.x[t - self.window + 1: t + 1]), int(self.y[t])


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        d = 2 * tp + fp + fn
        f1s.append(2 * tp / d if d else 0.0)
    return float(np.mean(f1s))


@dataclass
class TrainResult:
    model: torch.nn.Module
    best_val_f1: float
    metrics: dict[str, Any] = field(default_factory=dict)


def train(
    x: np.ndarray, y: np.ndarray, train_idx: list[int], val_idx: list[int],
    arch: str = "tcn", window: int = SEQ_LEN, max_epochs: int = 30,
    patience: int = 5, lr: float = 1e-3, batch_size: int = 256, seed: int = 0,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    ds_tr = WindowDataset(x, y, train_idx, window)
    ds_va = WindowDataset(x, y, val_idx, window)
    if len(ds_tr) < 64 or len(ds_va) < 16:
        raise ValueError(f"insufficient samples: train={len(ds_tr)} val={len(ds_va)}")
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size * 2)

    model = build_model(arch, x.shape[1])
    w = class_weights(y[ds_tr.positions])
    crit = FocalLoss(alpha=w, gamma=2.0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    best_f1, best_state, bad = 0.0, None, 0
    for _epoch in range(max_epochs):
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                preds.append(model(xb).argmax(1).numpy())
                trues.append(yb.numpy())
        f1 = macro_f1(np.concatenate(trues), np.concatenate(preds))
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # final validation metrics
    model.eval()
    preds_l, trues_l = [], []
    with torch.no_grad():
        for xb, yb in dl_va:
            preds_l.append(model(xb).argmax(1).numpy())
            trues_l.append(yb.numpy())
    yp, yt = np.concatenate(preds_l), np.concatenate(trues_l)
    nonflat = yt != 1
    dir_acc = float((yp[nonflat] == yt[nonflat]).mean()) if nonflat.any() else 0.0
    cm = [[int(np.sum((yt == i) & (yp == j))) for j in range(3)] for i in range(3)]
    metrics = {
        "val_macro_f1": round(best_f1, 4),
        "directional_accuracy": round(dir_acc, 4),
        "confusion_matrix": cm,
        "n_train": len(ds_tr), "n_val": len(ds_va), "arch": arch,
    }
    return TrainResult(model=model, best_val_f1=best_f1, metrics=metrics)


def export(result: TrainResult, models_dir: str, window: int = SEQ_LEN,
           n_features: int = 14) -> Path:
    """TorchScript the model + write metrics JSON; returns the .pt path."""
    out = Path(models_dir)
    out.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, window, n_features)
    scripted = torch.jit.trace(result.model.eval(), example)  # type: ignore[no-untyped-call]
    pt = out / "lob_flow.pt"
    scripted.save(str(pt))
    (out / "lob_flow_metrics.json").write_text(json.dumps(result.metrics, indent=2))
    return pt
