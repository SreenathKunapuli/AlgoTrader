import numpy as np
import pytest
import torch

from lob.losses import FocalLoss, class_weights
from lob.models import DeepLOB, TCN, build_model


def test_deeplob_forward():
    m = DeepLOB()
    x = torch.randn(8, 100, 40)
    out = m(x)
    assert out.shape == (8, 3)
    assert torch.isfinite(out).all()


def test_tcn_forward():
    m = TCN(n_features=62)
    x = torch.randn(8, 100, 62)
    out = m(x)
    assert out.shape == (8, 3)
    assert torch.isfinite(out).all()


def test_tcn_causality():
    """Future inputs must not change current output."""
    torch.manual_seed(0)
    m = TCN(n_features=8, channels=(16, 16, 16)).eval()
    x = torch.randn(1, 50, 8)
    with torch.no_grad():
        base = m(x[:, :30])
        x2 = x.clone()
        x2[:, 30:] = 999.0          # perturb the future
        pert = m(x2[:, :30])        # same first-30 window
    torch.testing.assert_close(base, pert)


def test_build_model_guards():
    with pytest.raises(ValueError):
        build_model("deeplob", n_features=62)
    assert isinstance(build_model("deeplob", 40), DeepLOB)
    assert isinstance(build_model("tcn", 62), TCN)


def test_focal_loss_matches_ce_at_gamma0():
    torch.manual_seed(0)
    logits = torch.randn(64, 3)
    target = torch.randint(0, 3, (64,))
    fl = FocalLoss(gamma=0.0)(logits, target)
    ce = torch.nn.functional.cross_entropy(logits, target)
    torch.testing.assert_close(fl, ce)


def test_class_weights_inverse_frequency():
    labels = np.array([1] * 90 + [0] * 5 + [2] * 5)
    w = class_weights(labels)
    assert w[0] > w[1] and w[2] > w[1]
    assert abs(w.mean().item() - 1.0) < 1e-6


def test_models_overfit_tiny_batch():
    """Sanity: both models can drive loss near zero on 32 samples."""
    torch.manual_seed(0)
    for name, f in (("deeplob", 40), ("tcn", 62)):
        x = torch.randn(32, 100, f)
        y = torch.randint(0, 3, (32,))
        m = build_model(name, f)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(150):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
        assert loss.item() < 0.1, f"{name} failed to overfit: {loss.item():.3f}"
