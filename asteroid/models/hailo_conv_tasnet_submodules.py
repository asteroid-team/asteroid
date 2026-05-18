from __future__ import annotations

import torch
from torch import nn

from ..masknn.hailo_convolutional import HailoConv1DBlockAsTensor, HailoTDConvNet2D


class HailoMaskerBottleneckOnly(nn.Module):
    """Export wrapper for masker bottleneck norm + 1x1 conv."""

    def __init__(self, masker: HailoTDConvNet2D):
        super().__init__()
        self.bottleneck_norm = masker.bottleneck_norm
        self.bottleneck_conv = masker.bottleneck_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck_conv(self.bottleneck_norm(x))


class HailoMaskerFirstTCNBlockAsTensor(nn.Module):
    """Export wrapper for first TCN block as one tensor output."""

    def __init__(self, masker: HailoTDConvNet2D):
        super().__init__()
        if len(masker.tcn) == 0:
            raise ValueError("masker has no TCN layers")
        self.block0 = HailoConv1DBlockAsTensor(masker.tcn[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block0(x)


class HailoMaskerHeadOnly(nn.Module):
    """Export wrapper for masker mask head + output activation."""

    def __init__(self, masker: HailoTDConvNet2D):
        super().__init__()
        self.mask_head = masker.mask_head
        self.output_act = masker.output_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_act(self.mask_head(x))


class HailoDecoderPreConvOrIdentity1x1(nn.Module):
    """
    Export-safe decoder_pre wrapper.
    If decoder_pre is Identity, replace it with an explicit identity 1x1 Conv2d.
    """

    def __init__(self, decoder_pre: nn.Module, in_chan: int):
        super().__init__()
        if isinstance(decoder_pre, nn.Identity):
            conv = nn.Conv2d(in_chan, in_chan, kernel_size=1, bias=False)
            with torch.no_grad():
                conv.weight.zero_()
                for c in range(in_chan):
                    conv.weight[c, c, 0, 0] = 1.0
            self.pre = conv
        else:
            self.pre = decoder_pre

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pre(x)


class HailoSourceProjectorSlice(nn.Module):
    """Export wrapper for one 256-channel slice of source projector output."""

    def __init__(self, source_projector: nn.Conv2d, out_index: int, slice_width: int = 256):
        super().__init__()
        if source_projector is None:
            raise ValueError("source_projector is required")
        if not isinstance(source_projector, nn.Conv2d):
            raise TypeError("source_projector must be nn.Conv2d")
        start = int(out_index) * int(slice_width)
        end = start + int(slice_width)
        if end > source_projector.out_channels:
            raise ValueError("slice out of range for source_projector")

        conv = nn.Conv2d(
            in_channels=source_projector.in_channels,
            out_channels=slice_width,
            kernel_size=1,
            bias=(source_projector.bias is not None),
        )
        with torch.no_grad():
            conv.weight.copy_(source_projector.weight[start:end, :, :, :])
            if source_projector.bias is not None:
                conv.bias.copy_(source_projector.bias[start:end])
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class HailoDecoderPreSlice(nn.Module):
    """Export wrapper for one 256-channel slice of decoder_pre output."""

    def __init__(self, decoder_pre: nn.Module, in_chan: int, out_index: int, slice_width: int = 256):
        super().__init__()
        pre = HailoDecoderPreConvOrIdentity1x1(decoder_pre, in_chan).pre
        if not isinstance(pre, nn.Conv2d):
            raise TypeError("decoder_pre slice expects Conv2d-compatible pre module")
        start = int(out_index) * int(slice_width)
        end = start + int(slice_width)
        if end > pre.out_channels:
            raise ValueError("slice out of range for decoder_pre")

        conv = nn.Conv2d(
            in_channels=pre.in_channels,
            out_channels=slice_width,
            kernel_size=1,
            bias=(pre.bias is not None),
        )
        with torch.no_grad():
            conv.weight.copy_(pre.weight[start:end, :, :, :])
            if pre.bias is not None:
                conv.bias.copy_(pre.bias[start:end])
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class HailoDecoderHeadSingleSrc(nn.Module):
    """Export wrapper for one source head from Conv1x1 decoder (512->2 split to 512->1)."""

    def __init__(self, decoder: nn.Module, src_index: int):
        super().__init__()
        if not isinstance(decoder, nn.Conv2d):
            raise TypeError("decoder head split expects conv1x1 decoder (nn.Conv2d)")
        if decoder.out_channels < (src_index + 1):
            raise ValueError("src_index out of range for decoder out channels")
        conv = nn.Conv2d(
            in_channels=decoder.in_channels,
            out_channels=1,
            kernel_size=1,
            bias=(decoder.bias is not None),
        )
        with torch.no_grad():
            conv.weight.copy_(decoder.weight[src_index : src_index + 1, :, :, :])
            if decoder.bias is not None:
                conv.bias.copy_(decoder.bias[src_index : src_index + 1])
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class HailoConv1x1PartialBlock(nn.Module):
    """
    Export wrapper for a partial 1x1 conv block:
    output = Conv1x1(input_block) where weights are copied from a submatrix
    of a larger source conv. This keeps module graph minimal (single Conv2d).
    """

    def __init__(
        self,
        source_conv: nn.Conv2d,
        in_start: int,
        in_len: int,
        out_start: int,
        out_len: int,
        include_bias: bool,
    ):
        super().__init__()
        if not isinstance(source_conv, nn.Conv2d):
            raise TypeError("source_conv must be nn.Conv2d")
        in_end = in_start + in_len
        out_end = out_start + out_len
        if in_start < 0 or out_start < 0 or in_end > source_conv.in_channels or out_end > source_conv.out_channels:
            raise ValueError("partial block range is out of source conv bounds")

        conv = nn.Conv2d(
            in_channels=in_len,
            out_channels=out_len,
            kernel_size=1,
            bias=include_bias and (source_conv.bias is not None),
        )
        with torch.no_grad():
            conv.weight.copy_(source_conv.weight[out_start:out_end, in_start:in_end, :, :])
            if conv.bias is not None:
                conv.bias.copy_(source_conv.bias[out_start:out_end])
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class HailoDepthwisePartialBlock(nn.Module):
    """Export wrapper for a channel-sliced depthwise Conv2d block."""

    def __init__(self, depth_conv: nn.Conv2d, block_index: int, block_chan: int = 64):
        super().__init__()
        if not isinstance(depth_conv, nn.Conv2d):
            raise TypeError("depth_conv must be nn.Conv2d")
        if depth_conv.groups != depth_conv.in_channels or depth_conv.groups != depth_conv.out_channels:
            raise ValueError("depth_conv must be depthwise (groups==in_channels==out_channels)")
        start = int(block_index) * int(block_chan)
        end = start + int(block_chan)
        if start < 0 or end > depth_conv.in_channels:
            raise ValueError("depthwise block range out of bounds")

        conv = nn.Conv2d(
            in_channels=block_chan,
            out_channels=block_chan,
            kernel_size=depth_conv.kernel_size,
            stride=depth_conv.stride,
            padding=depth_conv.padding,
            dilation=depth_conv.dilation,
            groups=block_chan,
            bias=(depth_conv.bias is not None),
        )
        with torch.no_grad():
            conv.weight.copy_(depth_conv.weight[start:end, :, :, :])
            if depth_conv.bias is not None:
                conv.bias.copy_(depth_conv.bias[start:end])
        self.conv = conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class HailoMaskerBottleneckBlock(nn.Module):
    """Masker bottleneck partial 1x1 block (linear contribution only)."""

    def __init__(self, masker: HailoTDConvNet2D, out_block_idx: int, in_block_idx: int, block_chan: int = 64):
        super().__init__()
        self.partial = HailoConv1x1PartialBlock(
            source_conv=masker.bottleneck_conv,
            in_start=in_block_idx * block_chan,
            in_len=block_chan,
            out_start=out_block_idx * block_chan,
            out_len=block_chan,
            include_bias=(in_block_idx == 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.partial(x)


class HailoMaskerTCN0InConvBlock(nn.Module):
    """First TCN block in_conv partial (linear contribution only)."""

    def __init__(self, masker: HailoTDConvNet2D, out_block_idx: int, in_block_idx: int, block_chan: int = 64):
        super().__init__()
        if len(masker.tcn) == 0:
            raise ValueError("masker has no TCN blocks")
        block0 = masker.tcn[0]
        self.partial = HailoConv1x1PartialBlock(
            source_conv=block0.in_conv,
            in_start=in_block_idx * block_chan,
            in_len=block_chan,
            out_start=out_block_idx * block_chan,
            out_len=block_chan,
            include_bias=(in_block_idx == 0),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.partial(x)


class HailoMaskerTCN0DepthBlock(nn.Module):
    """First TCN block depthwise partial (linear contribution only)."""

    def __init__(self, masker: HailoTDConvNet2D, depth_block_idx: int, block_chan: int = 64):
        super().__init__()
        if len(masker.tcn) == 0:
            raise ValueError("masker has no TCN blocks")
        block0 = masker.tcn[0]
        self.partial = HailoDepthwisePartialBlock(block0.depth_conv, block_index=depth_block_idx, block_chan=block_chan)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.partial(x)


class HailoMaskerTCN0ResBlock(nn.Module):
    """First TCN block residual 1x1, partial by in/out block."""

    def __init__(self, masker: HailoTDConvNet2D, out_block_idx: int, in_block_idx: int, block_chan: int = 64):
        super().__init__()
        if len(masker.tcn) == 0:
            raise ValueError("masker has no TCN blocks")
        block0 = masker.tcn[0]
        self.partial = HailoConv1x1PartialBlock(
            source_conv=block0.res_conv,
            in_start=in_block_idx * block_chan,
            in_len=block_chan,
            out_start=out_block_idx * block_chan,
            out_len=block_chan,
            include_bias=(in_block_idx == 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.partial(x)


class HailoMaskerTCN0SkipBlock(nn.Module):
    """First TCN block skip 1x1, partial by in/out block."""

    def __init__(self, masker: HailoTDConvNet2D, out_block_idx: int, in_block_idx: int, block_chan: int = 64):
        super().__init__()
        if len(masker.tcn) == 0:
            raise ValueError("masker has no TCN blocks")
        block0 = masker.tcn[0]
        if block0.skip_conv is None:
            raise ValueError("masker block0 has no skip conv")
        self.partial = HailoConv1x1PartialBlock(
            source_conv=block0.skip_conv,
            in_start=in_block_idx * block_chan,
            in_len=block_chan,
            out_start=out_block_idx * block_chan,
            out_len=block_chan,
            include_bias=(in_block_idx == 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.partial(x)


class HailoMaskerHeadBlock(nn.Module):
    """Masker head partial conv block (linear score contribution before activation)."""

    def __init__(self, masker: HailoTDConvNet2D, out_block_idx: int, in_block_idx: int, block_chan: int = 64):
        super().__init__()
        self.partial = HailoConv1x1PartialBlock(
            source_conv=masker.mask_head,
            in_start=in_block_idx * block_chan,
            in_len=block_chan,
            out_start=out_block_idx * block_chan,
            out_len=block_chan,
            include_bias=(in_block_idx == 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.partial(x)
