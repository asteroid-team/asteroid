import torch
from torch import nn


class HailoIdentityNorm2D(nn.Module):
    """No-op normalization for Hailo-safe export paths."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        # Keep explicit Conv op for stable parser behavior in module tests.
        self.passthrough = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False)
        with torch.no_grad():
            self.passthrough.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.passthrough(x)


class HailoChannelAffineNorm2D(nn.Module):
    """Per-channel affine transform on [N, C, 1, T] tensors."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        # Use grouped 1x1 conv so Hailo parser treats this as conv, not normalization algebra.
        self.affine = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=True)
        with torch.no_grad():
            self.affine.weight.fill_(1.0)
            self.affine.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"HailoChannelAffineNorm2D expects 4D input, got {tuple(x.shape)}")
        return self.affine(x)


def get_hailo_norm(mode: str, channels: int) -> nn.Module:
    mode = mode.lower()
    if mode == "identity":
        return HailoIdentityNorm2D(channels)
    if mode in {"affine", "channel_affine"}:
        return HailoChannelAffineNorm2D(channels)
    raise ValueError(f"Unsupported Hailo norm mode: {mode}")
