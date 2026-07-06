"""Training pipeline: temporal splits, windowed dataset, training loop.

Leakage discipline — the three rules everything here enforces:

1. Splits are contiguous in time, ordered train < val < test, with an
   embargo gap of (window + horizon) samples between them so no label's
   future window or input window straddles a boundary.
2. The feature Normalizer is fitted on the train slice only.
3. DataLoader shuffling happens only WITHIN the train slice (sample order
   within an already-causal training set is free to permute; the windows
   themselves remain strictly causal).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .features import Normalizer
from .labels import INVALID
from .losses import FocalLoss, class_weights
from .models import build_model


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def temporal_split(
    n: int,
    window: int,
    horizon: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> SplitIndices:
    """Contiguous time split with embargo gaps.

    Returned indices are *label positions* t; the model input is the window
    ending at t. Valid positions start at window-1 (need a full input
    window) and the embargo between segments is window + horizon.
    """
    embargo = window + horizon
    t_end = int(n * train_frac)
    v_end = int(n * (train_frac + val_frac))

    train = np.arange(window - 1, t_end)
    val = np.arange(t_end + embargo, v_end)
    test = np.arange(v_end + embargo, n - horizon)
    if len(val) < 100 or len(test) < 100:
        raise ValueError("series too short for requested splits")
    return SplitIndices(train, val, test)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class WindowDataset(Dataset):
    """Yields (window [T, F] float32, label int64) at requested positions."""

    def __init__(
        self,
        features: np.ndarray,     # [N, F] already normalized, float32
        labels: np.ndarray,       # [N] int64 with INVALID markers
        positions: np.ndarray,    # candidate label positions
        window: int,
    ):
        self.x = features
        self.y = labels
        self.window = window
        keep = labels[positions] != INVALID
        self.positions = positions[keep]

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, i: int):
        t = self.positions[i]
        win = self.x[t - self.window + 1 : t + 1]
        return torch.from_numpy(win), int(self.y[t])


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    model: str = "tcn"
    window: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    patience: int = 5            # early stopping on val macro-F1
    gamma: float = 2.0           # focal loss focusing parameter
    num_workers: int = 0
    seed: int = 0


@dataclass
class TrainResult:
    model: torch.nn.Module
    normalizer: Normalizer
    history: list[dict] = field(default_factory=list)
    best_val_f1: float = 0.0


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


@torch.no_grad()
def _evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        out = model(xb.to(device))
        preds.append(out.argmax(dim=1).cpu().numpy())
        trues.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    return macro_f1(y_true, y_pred), y_true, y_pred


def train_model(
    features: np.ndarray,        # [N, F] raw (unnormalized)
    labels: np.ndarray,          # [N] with INVALID
    cfg: TrainConfig,
    splits: SplitIndices | None = None,
    horizon: int = 20,
    verbose: bool = True,
) -> TrainResult:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device()

    n = len(features)
    if splits is None:
        splits = temporal_split(n, cfg.window, horizon)

    # Normalize using train statistics only
    normalizer = Normalizer().fit(features[: splits.train[-1] + 1])
    x = normalizer.transform(features)

    ds_train = WindowDataset(x, labels, splits.train, cfg.window)
    ds_val = WindowDataset(x, labels, splits.val, cfg.window)

    train_loader = DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, drop_last=True,
    )
    val_loader = DataLoader(ds_val, batch_size=cfg.batch_size * 2, shuffle=False)

    model = build_model(cfg.model, x.shape[1]).to(device)
    w = class_weights(labels[ds_train.positions]).to(device)
    criterion = FocalLoss(alpha=w, gamma=cfg.gamma)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)

    best_f1, best_state, bad_epochs = 0.0, None, 0
    history = []

    for epoch in range(cfg.max_epochs):
        model.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()

        val_f1, _, _ = _evaluate(model, val_loader, device)
        rec = {
            "epoch": epoch,
            "train_loss": total_loss / max(n_batches, 1),
            "val_macro_f1": val_f1,
            "seconds": time.time() - t0,
        }
        history.append(rec)
        if verbose:
            print(f"  epoch {epoch:2d}  loss {rec['train_loss']:.4f}  "
                  f"val macro-F1 {val_f1:.4f}  ({rec['seconds']:.1f}s)")

        if val_f1 > best_f1:
            best_f1, bad_epochs = val_f1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best val F1 {best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model=model, normalizer=normalizer,
                       history=history, best_val_f1=best_f1)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    features_norm: np.ndarray,
    positions: np.ndarray,
    window: int,
    batch_size: int = 512,
) -> np.ndarray:
    """Class probabilities [len(positions), 3] at the given label positions."""
    device = next(model.parameters()).device
    model.eval()
    ds = WindowDataset(
        features_norm,
        np.zeros(len(features_norm), dtype=np.int64),   # dummy labels, all valid
        positions, window,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = []
    for xb, _ in loader:
        probs = torch.softmax(model(xb.to(device)), dim=1)
        out.append(probs.cpu().numpy())
    return np.concatenate(out)
