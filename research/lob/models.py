"""Model architectures for LOB mid-price prediction.

DeepLOB  (Zhang, Zohren, Roberts 2019, arXiv:1808.03668)
    Treats a window of snapshots as a [T x 40] image. Three conv blocks
    exploit the spatial structure of the columns: stride-2 convs first fuse
    each (price, volume) pair, then each (ask, bid) pair, then a full-width
    conv fuses the 10 levels. An inception module extracts multi-scale
    temporal patterns; an LSTM aggregates over time.
    Input must be raw40-ordered features — column order is load-bearing.

TCN
    Dilated causal convolutions with residual blocks (Bai et al. 2018).
    Feature-order agnostic, so it accepts the extended 62-feature set.
    Receptive field with kernel 3 and dilations (1,2,4,8,16,32) is 253,
    comfortably above the standard 100-step window.

Both take [B, T, F] float32 and return [B, 3] logits (down/flat/up).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# DeepLOB
# --------------------------------------------------------------------------- #
class _ConvBlock(nn.Module):
    """One DeepLOB conv block: a spatial-fusing conv + two temporal convs."""

    def __init__(self, in_ch: int, out_ch: int, fuse_kernel: tuple, fuse_stride: tuple):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, fuse_kernel, stride=fuse_stride),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, (4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, (4, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.net(x)


class _Inception(nn.Module):
    def __init__(self, in_ch: int, branch_ch: int = 64):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, (1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(branch_ch),
            nn.Conv2d(branch_ch, branch_ch, (3, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(branch_ch),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, branch_ch, (1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(branch_ch),
            nn.Conv2d(branch_ch, branch_ch, (5, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(branch_ch),
        )
        self.b3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_ch, branch_ch, (1, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(branch_ch),
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)


class DeepLOB(nn.Module):
    def __init__(self, n_levels: int = 10, lstm_hidden: int = 64, n_classes: int = 3):
        super().__init__()
        self.block1 = _ConvBlock(1, 32, (1, 2), (1, 2))    # 40 -> 20: fuse (p,v)
        self.block2 = _ConvBlock(32, 32, (1, 2), (1, 2))   # 20 -> 10: fuse (ask,bid)
        self.block3 = _ConvBlock(32, 32, (1, n_levels), (1, 1))  # 10 -> 1: fuse levels
        self.inception = _Inception(32, branch_ch=64)
        self.lstm = nn.LSTM(192, lstm_hidden, batch_first=True)
        self.head = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, 40] -> [B, 1, T, 40]
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)              # [B, 32, T', 1]
        x = self.inception(x)           # [B, 192, T', 1]
        x = x.squeeze(3).permute(0, 2, 1)   # [B, T', 192]
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


# --------------------------------------------------------------------------- #
# TCN
# --------------------------------------------------------------------------- #
class _CausalConv1d(nn.Module):
    """Left-padded conv so output at t sees only inputs <= t."""

    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
        )

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class _TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel, dilation)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel, dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        y = self.drop(self.act(self.conv1(x)))
        y = self.drop(self.act(self.conv2(y)))
        return self.act(y + self.downsample(x))


class TCN(nn.Module):
    def __init__(
        self,
        n_features: int = 62,
        channels: tuple = (64, 64, 64, 64, 64, 64),
        kernel: int = 3,
        dropout: float = 0.1,
        n_classes: int = 3,
    ):
        super().__init__()
        layers = []
        in_ch = n_features
        for i, ch in enumerate(channels):
            layers.append(_TCNBlock(in_ch, ch, kernel, dilation=2**i, dropout=dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, F] -> [B, F, T]
        y = self.tcn(x.permute(0, 2, 1))
        return self.head(y[:, :, -1])


def build_model(name: str, n_features: int) -> nn.Module:
    if name == "deeplob":
        if n_features != 40:
            raise ValueError("DeepLOB requires raw40 features (got "
                             f"{n_features}); its conv strides assume the "
                             "(p,v)/(ask,bid) column layout")
        return DeepLOB()
    if name == "tcn":
        return TCN(n_features=n_features)
    raise ValueError(f"unknown model: {name}")
