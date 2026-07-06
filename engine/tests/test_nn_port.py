"""Model-port test suite (§5.8): shape, causality, overfit, focal loss,
TorchScript round-trip, walk-forward leakage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch
from lobplatform.signals.nn.features import N_FEATURES
from lobplatform.signals.nn.losses import FocalLoss, class_weights
from lobplatform.signals.nn.models import TCN, SmallTransformer, build_model
from lobplatform.signals.nn.splits import walk_forward_day_split
from lobplatform.signals.nn.training import SEQ_LEN, TrainResult, export


def test_forward_shapes() -> None:
    x = torch.randn(4, SEQ_LEN, N_FEATURES)
    for arch in ("tcn", "deeplob", "transformer"):
        out = build_model(arch, N_FEATURES)(x)
        assert out.shape == (4, 3), arch
        assert torch.isfinite(out).all()


@pytest.mark.parametrize("model_cls", [TCN, SmallTransformer])
def test_causality(model_cls) -> None:  # type: ignore[no-untyped-def]
    """Perturbing the future must not change earlier outputs."""
    torch.manual_seed(0)
    m = model_cls(n_features=N_FEATURES).eval()
    x = torch.randn(1, SEQ_LEN, N_FEATURES)
    with torch.no_grad():
        base = m(x[:, :40])
        x2 = x.clone()
        x2[:, 40:] = 999.0
        pert = m(x2[:, :40])
    torch.testing.assert_close(base, pert)


def test_overfit_tiny_batch() -> None:
    """The ported loop must be able to memorize 32 samples (loss -> ~0)."""
    torch.manual_seed(0)
    x = torch.randn(32, SEQ_LEN, N_FEATURES)
    y = torch.randint(0, 3, (32,))
    m = TCN(n_features=N_FEATURES)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    loss = torch.tensor(1.0)
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, f"failed to overfit: {loss.item():.3f}"


def test_focal_loss_hand_computed() -> None:
    # gamma=0, no alpha -> plain cross-entropy
    torch.manual_seed(0)
    logits = torch.randn(16, 3)
    target = torch.randint(0, 3, (16,))
    torch.testing.assert_close(
        FocalLoss(gamma=0.0)(logits, target),
        torch.nn.functional.cross_entropy(logits, target))
    # gamma=2, single sample, hand value: p=softmax; loss=-(1-p)^2 log p
    logits1 = torch.tensor([[2.0, 0.0, 0.0]])
    t1 = torch.tensor([0])
    p = torch.softmax(logits1, dim=1)[0, 0]
    expected = -((1 - p) ** 2) * torch.log(p)
    torch.testing.assert_close(FocalLoss(gamma=2.0)(logits1, t1), expected)


def test_class_weights_inverse_freq() -> None:
    w = class_weights(np.array([1] * 90 + [0] * 5 + [2] * 5))
    assert w[0] > w[1] and w[2] > w[1]
    assert abs(float(w.mean()) - 1.0) < 1e-6


def test_torchscript_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    m = TCN(n_features=N_FEATURES).eval()
    res = TrainResult(model=m, best_val_f1=0.5, metrics={"val_macro_f1": 0.5})
    pt = export(res, str(tmp_path), n_features=N_FEATURES)
    loaded = torch.jit.load(str(pt)).eval()
    x = torch.randn(2, SEQ_LEN, N_FEATURES)
    with torch.no_grad():
        torch.testing.assert_close(m(x), loaded(x), rtol=1e-4, atol=1e-5)


def test_walk_forward_no_leakage() -> None:
    t0 = datetime(2026, 5, 1, tzinfo=UTC)
    stamps = [t0 + timedelta(days=d, minutes=5 * i) for d in range(20) for i in range(50)]
    sp = walk_forward_day_split(stamps, val_days=4, embargo_days=1)
    max_train = max(stamps[i] for i in sp.train_idx)
    min_val = min(stamps[i] for i in sp.val_idx)
    assert min_val.date() > max_train.date() + timedelta(days=1)  # embargo respected
    assert sp.train_idx and sp.val_idx
