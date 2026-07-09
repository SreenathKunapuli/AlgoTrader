"""Rankers for the cross-sectional tournament: GBT, MLP, ensemble, distilled.

Every model maps a feature-rank vector to a predicted return-rank. They are
trained per OOS year on identical expanding windows and judged only by the
same cost-aware portfolio — no model gets to pick its own scoreboard.

Distillation: the teacher (GBT+MLP average) is compressed into a ridge
student trained on the TEACHER'S predictions (not the labels). The student
is a linear read-out you can inspect coefficient-by-coefficient and run
anywhere; the experiment measures how much of the teacher's OOS alpha
survives compression. This is the same teacher→student pattern used to
distill LLMs, applied at quant scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from torch import nn

MLP_SEEDS = (0, 1, 2)


class _MLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_one_mlp(x: np.ndarray, y: np.ndarray, seed: int,
                   epochs: int = 25, batch: int = 4096, lr: float = 1e-3) -> _MLP:
    torch.manual_seed(seed)
    m = _MLP(x.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    xt = torch.from_numpy(x.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    n = len(xt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i: i + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(m(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    m.eval()
    return m


class MLPRanker:
    """Seed-ensembled MLP: mean prediction of MLP_SEEDS independently trained nets."""

    def __init__(self) -> None:
        self.nets: list[_MLP] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPRanker":
        self.nets = [_train_one_mlp(x, y, s) for s in MLP_SEEDS]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
        with torch.no_grad():
            preds = [n(xt).numpy() for n in self.nets]
        return np.mean(preds, axis=0)


class GBTRanker:
    def __init__(self, seed: int = 0) -> None:
        self.m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=False, random_state=seed)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GBTRanker":
        self.m.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.m.predict(x)


class TeacherEnsemble:
    """Average of GBT and MLP predicted ranks (rank-normalized per call)."""

    def __init__(self) -> None:
        self.gbt = GBTRanker()
        self.mlp = MLPRanker()

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TeacherEnsemble":
        self.gbt.fit(x, y)
        self.mlp.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        a = pd.Series(self.gbt.predict(x)).rank(pct=True).values
        b = pd.Series(self.mlp.predict(x)).rank(pct=True).values
        return (a + b) / 2.0


class DistilledStudent:
    """Ridge regression trained to imitate a fitted teacher's predictions."""

    def __init__(self, teacher: TeacherEnsemble, alpha: float = 1.0) -> None:
        self.teacher = teacher
        self.m = Ridge(alpha=alpha)

    def fit(self, x: np.ndarray, y_unused: np.ndarray | None = None) -> "DistilledStudent":
        self.m.fit(x, self.teacher.predict(x))  # soft targets, not labels
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.m.predict(x)

    def coefficients(self, feature_names: list[str]) -> pd.Series:
        return pd.Series(self.m.coef_, index=feature_names).sort_values()
