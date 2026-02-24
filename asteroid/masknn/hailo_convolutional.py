from typing import Optional, Tuple

import torch
from torch import nn

from .hailo_activations import get_hailo_activation
from .hailo_norms import get_hailo_norm


class HailoConv1DBlock2D(nn.Module):
    """Hailo-oriented 1D block implemented with horizontal Conv2d kernels."""

    def __init__(
        self,
        in_chan: int,
        hid_chan: int,
        skip_out_chan: int,
        kernel_size: int,
        padding: int,
        dilation: int,
        norm_mode: str = "affine",
    ):
        super().__init__()
        self.skip_out_chan = int(skip_out_chan)

        self.in_conv = nn.Conv2d(in_chan, hid_chan, kernel_size=1, bias=True)
        self.in_act = nn.ReLU(inplace=False)
        self.in_norm = get_hailo_norm(norm_mode, hid_chan)

        self.depth_conv = nn.Conv2d(
            hid_chan,
            hid_chan,
            kernel_size=(1, kernel_size),
            stride=(1, 1),
            padding=(0, padding),
            dilation=(1, dilation),
            groups=hid_chan,
            bias=True,
        )
        self.depth_act = nn.ReLU(inplace=False)
        self.depth_norm = get_hailo_norm(norm_mode, hid_chan)

        self.res_conv = nn.Conv2d(hid_chan, in_chan, kernel_size=1, bias=True)
        self.skip_conv = nn.Conv2d(hid_chan, self.skip_out_chan, kernel_size=1, bias=True) if self.skip_out_chan else None

    def forward(self, x: torch.Tensor):
        h = self.in_conv(x)
        h = self.in_act(h)
        h = self.in_norm(h)
        h = self.depth_conv(h)
        h = self.depth_act(h)
        h = self.depth_norm(h)

        residual = self.res_conv(h)
        if self.skip_conv is None:
            return residual
        skip = self.skip_conv(h)
        return residual, skip


class HailoTDConvNet2D(nn.Module):
    """Hailo masker stack with static 4D tensors [N, C, 1, T]."""

    def __init__(
        self,
        in_chan: int,
        n_src: int,
        out_chan: Optional[int] = None,
        n_blocks: int = 1,
        n_repeats: int = 1,
        bn_chan: int = 128,
        hid_chan: int = 256,
        skip_chan: int = 128,
        conv_kernel_size: int = 3,
        mask_act: str = "sigmoid",
        norm_mode: str = "affine",
        disable_skip: bool = False,
        truncate_k_blocks: int = 0,
    ):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.out_chan = out_chan or in_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.conv_kernel_size = conv_kernel_size
        self.disable_skip = disable_skip
        self.truncate_k_blocks = max(int(truncate_k_blocks), 0)

        self.bottleneck_norm = get_hailo_norm(norm_mode, in_chan)
        self.bottleneck_conv = nn.Conv2d(in_chan, bn_chan, kernel_size=1, bias=True)

        layers = []
        for _r in range(n_repeats):
            for x in range(n_blocks):
                dilation = 2**x
                padding = ((conv_kernel_size - 1) * dilation) // 2
                layers.append(
                    HailoConv1DBlock2D(
                        in_chan=bn_chan,
                        hid_chan=hid_chan,
                        skip_out_chan=skip_chan,
                        kernel_size=conv_kernel_size,
                        padding=padding,
                        dilation=dilation,
                        norm_mode=norm_mode,
                    )
                )
        self.tcn = nn.ModuleList(layers)

        mask_inp_chan = bn_chan if (skip_chan == 0 or disable_skip) else skip_chan
        self.mask_head = nn.Conv2d(mask_inp_chan, n_src * self.out_chan, kernel_size=1, bias=True)
        self.output_act = get_hailo_activation(mask_act)

    def forward(self, mixture_w: torch.Tensor) -> torch.Tensor:
        if mixture_w.dim() != 4:
            raise ValueError(f"HailoTDConvNet2D expects [N,C,1,T], got {tuple(mixture_w.shape)}")

        output = self.bottleneck_conv(self.bottleneck_norm(mixture_w))
        skip_connection = None

        for idx, layer in enumerate(self.tcn):
            if self.truncate_k_blocks > 0 and idx >= self.truncate_k_blocks:
                break
            tcn_out = layer(output)
            if isinstance(tcn_out, tuple):
                residual, skip = tcn_out
            else:
                residual, skip = tcn_out, None
            output = output + residual
            if not self.disable_skip and skip is not None:
                skip_connection = skip if skip_connection is None else (skip_connection + skip)

        mask_input = output if (self.disable_skip or skip_connection is None) else skip_connection
        score = self.mask_head(mask_input)
        est_mask = self.output_act(score)
        return est_mask


class HailoConv1DBlockAsTensor(nn.Module):
    """Export helper to convert (residual, skip) tuple into one tensor."""

    def __init__(self, block: HailoConv1DBlock2D):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if isinstance(out, tuple):
            residual, skip = out
            return torch.cat([residual, skip], dim=1)
        return out
