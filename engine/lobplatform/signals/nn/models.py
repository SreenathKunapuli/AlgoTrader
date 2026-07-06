"""Model architectures ported from research/lob/models.py (see PLAN.md map).

The TCN and DeepLOB are copied faithfully: identical layer structure,
kernel sizes, dilations, and activation choices. Only the input feature
dimension is parameterized (live L1 features F=14 vs research 40/62).
The optional transformer (LOB_FLOW_ARCH=transformer) is the §5.2.1
experiment: 2 layers, 4 heads, d_model 64, same 3-class head.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn


# --- DeepLOB (ported; kept for research parity — live default is TCN) --- #
class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, fuse_kernel: tuple[int, int],
                 fuse_stride: tuple[int, int]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, fuse_kernel, stride=fuse_stride),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, (4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, (4, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


class _Inception(nn.Module):
    def __init__(self, in_ch: int, branch_ch: int = 64) -> None:
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, (1, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(branch_ch),
            nn.Conv2d(branch_ch, branch_ch, (3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(branch_ch),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, (1, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(branch_ch),
            nn.Conv2d(branch_ch, branch_ch, (5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(branch_ch),
        )
        self.b3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_ch, branch_ch, (1, 1)), nn.LeakyReLU(0.01), nn.BatchNorm2d(branch_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)


class DeepLOB(nn.Module):
    """CNN-LSTM. NOTE: pairwise-fusing convs assume an even feature count."""

    def __init__(self, n_features: int, lstm_hidden: int = 64, n_classes: int = 3) -> None:
        super().__init__()
        assert n_features % 2 == 0, "DeepLOB fuse conv needs even feature count"
        half = n_features // 2
        self.block1 = _ConvBlock(1, 32, (1, 2), (1, 2))
        self.block2 = _ConvBlock(32, 32, (1, half), (1, 1))
        self.inception = _Inception(32, branch_ch=64)
        self.lstm = nn.LSTM(192, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.unsqueeze(1)
        y = self.block1(y)
        y = self.block2(y)
        y = self.inception(y)
        y = y.squeeze(3).permute(0, 2, 1)
        out, _ = self.lstm(y)
        return cast(torch.Tensor, self.head(out[:, -1]))


# --- TCN (ported verbatim; the canonical live model) --- #
class _CausalConv1d(nn.Module):
    """Left-padded conv so output at t sees only inputs <= t."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv(nn.functional.pad(x, (self.pad, 0))))


class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int,
                 dropout: float) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop(self.act(self.conv1(x)))
        y = self.drop(self.act(self.conv2(y)))
        return cast(torch.Tensor, self.act(y + self.downsample(x)))


class TCN(nn.Module):
    def __init__(self, n_features: int = 14,
                 channels: tuple[int, ...] = (64, 64, 64, 64, 64, 64),
                 kernel: int = 3, dropout: float = 0.1, n_classes: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = n_features
        for i, ch in enumerate(channels):
            layers.append(_TCNBlock(in_ch, ch, kernel, dilation=2**i, dropout=dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.tcn(x.permute(0, 2, 1))
        return cast(torch.Tensor, self.head(y[:, :, -1]))


# --- Transformer experiment (§5.2.1 optional flag) --- #
class SmallTransformer(nn.Module):
    def __init__(self, n_features: int = 14, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, n_classes: int = 3, max_len: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128,
                                         dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        y = self.proj(x) + self.pos[:, :t]
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        y = self.encoder(y, mask=mask)
        return cast(torch.Tensor, self.head(y[:, -1]))


def build_model(arch: str, n_features: int) -> nn.Module:
    if arch == "tcn":
        return TCN(n_features=n_features)
    if arch == "deeplob":
        return DeepLOB(n_features=n_features)
    if arch == "transformer":
        return SmallTransformer(n_features=n_features)
    raise ValueError(f"unknown arch: {arch}")
