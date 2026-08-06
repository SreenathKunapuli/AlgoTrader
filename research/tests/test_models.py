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
    """Perturbing future timesteps must not change earlier outputs."""
    torch.manual_seed(0)
    m = TCN(n_features=8, channels=(16, 16, 16)).eval()
    x = torch.randn(1, 50, 8)
    x2 = x.clone()
    x2[:, 30:] = 999.0                        # perturb the future
    with torch.no_grad():
        feats = m.tcn(x.permute(0, 2, 1))     # [B, C, T]
        feats2 = m.tcn(x2.permute(0, 2, 1))
        prefix = m(x[:, :30])                 # prediction seeing data up to t=30
        full_at_30 = m.head(feats2[:, :, 29]) # same timestep, perturbed future
    # The perturbation must actually reach the network (guards against a
    # vacuous test) ...
    assert not torch.allclose(feats[:, :, 30:], feats2[:, :, 30:])
    # ... yet features before the perturbation point are untouched, and the
    # t=30 prediction is identical whether the perturbed future exists or not.
    torch.testing.assert_close(feats[:, :, :30], feats2[:, :, :30])
    torch.testing.assert_close(prefix, full_at_30)


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
