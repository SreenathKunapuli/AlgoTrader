"""lob_flow — the ported LOB neural model as a live signal (§5.2.1).

Loads the TorchScript export of MY TCN (or the transformer variant when it
wins the walk-forward comparison). Inference mapping per spec:
score = p_up − p_down; confidence = max(p_up, p_down), gated to 0 when
FLAT is the argmax. If validation directional accuracy ≤ 51%, confidence
is scaled down proportionally — weak signals ship, the ensemble copes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..data.bar_builder import Bar
from .base import Signal, SignalOutput
from .nn.features import N_FEATURES, build_features
from .nn.training import SEQ_LEN


class LobFlowSignal(Signal):
    name = "lob_flow"

    def __init__(self, models_dir: str = "models") -> None:
        self._model: torch.jit.ScriptModule | None = None
        self._conf_scale = 1.0
        pt = Path(models_dir) / "lob_flow.pt"
        metrics = Path(models_dir) / "lob_flow_metrics.json"
        if pt.exists():
            self._model = torch.jit.load(str(pt), map_location="cpu")  # type: ignore[no-untyped-call]
            self._model.eval()
            if metrics.exists():
                m = json.loads(metrics.read_text())
                da = float(m.get("directional_accuracy", 0.0))
                if da <= 0.51:
                    # proportional shrink: 50% acc -> 0 confidence, 51% -> ~0.2
                    self._conf_scale = max(0.0, (da - 0.50) / 0.05)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def compute(self, symbol: str, bars: list[Bar]) -> SignalOutput:
        if self._model is None or len(bars) < SEQ_LEN + 12:
            return SignalOutput(0.0, 0.0)
        feats = build_features(bars)[-SEQ_LEN:]
        x = torch.from_numpy(feats[np.newaxis].astype(np.float32))
        with torch.no_grad():
            probs = torch.softmax(self._model(x), dim=1)[0].numpy()
        p_down, p_flat, p_up = float(probs[0]), float(probs[1]), float(probs[2])
        score = max(-1.0, min(1.0, p_up - p_down))
        conf = 0.0 if p_flat >= max(p_up, p_down) else max(p_up, p_down)
        return SignalOutput(score, min(1.0, conf * self._conf_scale))


__all__ = ["LobFlowSignal", "N_FEATURES"]
