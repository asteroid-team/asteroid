"""Copied and adapted from: https://github.com/etzinis/sudo_rm_rf

University of Illinois Open Source License

Copyright © 2020, University of Illinois at Urbana Champaign. All rights reserved.

Developed by: Efthymios Tzinis 1, Zhepei Wang 1 and Paris Smaragdis 1,2

1: University of Illinois at Urbana-Champaign

2: Adobe Research

This work was supported by NSF grant 1453104.

Paper link: arxiv.org/abs/2007.06833

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the “Software”), to deal with
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions: Redistributions of source code must
retain the above copyright notice, this list of conditions and the following
disclaimers. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimers in the
documentation and/or other materials provided with the distribution. Neither
the names of Computational Audio Group, University of Illinois at
Urbana-Champaign, nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission. THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS WITH THE SOFTWARE.
"""
import torch
import torch.nn as nn
import math

from ..filterbanks import make_enc_dec
from ..masknn.norms import GlobLN


class _BaseUBlock(nn.Module):
    def __init__(self, out_chan=128, in_chan=512, upsampling_depth=4, use_globln=False):
        super().__init__()
        self.proj_1x1 = _ConvNormAct(
            out_chan, in_chan, 1, stride=1, groups=1, use_globln=use_globln
        )
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            _DilatedConvNorm(
                in_chan, in_chan, kSize=5, stride=1, groups=in_chan, d=1, use_globln=use_globln,
            )
        )

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(
                _DilatedConvNorm(
                    in_chan,
                    in_chan,
                    kSize=2 * stride + 1,
                    stride=stride,
                    groups=in_chan,
                    d=1,
                    use_globln=use_globln,
                )
            )
        if upsampling_depth > 1:
            self.upsampler = torch.nn.Upsample(
                scale_factor=2,
                # align_corners=True,
                # mode='bicubic'
            )


class UBlock(_BaseUBlock):
    """ Upsampling block.

    Based on the following principle:
        ``REDUCE ---> SPLIT ---> TRANSFORM --> MERGE``
    """

    def __init__(self, out_chan=128, in_chan=512, upsampling_depth=4):
        super().__init__(out_chan, in_chan, upsampling_depth, use_globln=False)
        self.conv_1x1_exp = _ConvNorm(in_chan, out_chan, 1, 1, groups=1)
        self.final_norm = _NormAct(in_chan)
        self.module_act = _NormAct(out_chan)

    def forward(self, x):
        """
        Args:
            x: input feature map

        Returns:
            transformed feature map
        """

        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # Gather them now in reverse order
        for _ in range(self.depth - 1):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k

        expanded = self.conv_1x1_exp(self.final_norm(output[-1]))

        return self.module_act(expanded + x)


class UConvBlock(_BaseUBlock):
    """ Block which performs successive downsampling and upsampling
    in order to be able to analyze the input features in multiple resolutions.
    """

    def __init__(self, out_chan=128, in_chan=512, upsampling_depth=4):
        super().__init__(out_chan, in_chan, upsampling_depth, use_globln=True)
        self.final_norm = _NormAct(in_chan, use_globln=True)
        self.res_conv = nn.Conv1d(in_chan, out_chan, 1)

    def forward(self, x):
        """
        Args
            x: input feature map

        Returns:
            transformed feature map
        """
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # Gather them now in reverse order
        for _ in range(self.depth - 1):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k

        expanded = self.final_norm(output[-1])

        return self.res_conv(expanded) + residual


class _SuDORMRFBase(nn.Module):
    def __init__(
        self,
        out_chan=128,
        in_chan=512,
        num_blocks=16,
        upsampling_depth=4,
        kernel_size=21,
        n_filters=512,
        n_src=2,
    ):
        super().__init__()

        # Number of sources to produce
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.n_src = n_src

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(self.kernel_size // 2 * 2 ** self.upsampling_depth) // math.gcd(
            self.kernel_size // 2, 2 ** self.upsampling_depth
        )

    # Forward pass
    def forward(self, input_wav):
        x, s = self.frontend(input_wav)
        x = self.separation(x)
        return self.backend(x, s)

    def separation(self, x):
        raise NotImplementedError("Must be implemented in subclass")

    def frontend(self, input_wav):
        self.input_wav_shape = input_wav.shape
        if input_wav.ndim == 2:
            input_wav = input_wav.unsqueeze(1)

        x = self.pad_to_appropriate_length(input_wav)
        x = self.encoder(x)

        # Split paths
        s = x.clone().unsqueeze(1)

        return x, s

    def backend(self, x, s):
        estimated_waveforms = self.decoder((x * s).view(x.shape[0], self.n_src, -1, x.shape[-1]))
        # Remove padding
        estimated_waveforms = estimated_waveforms[..., : self.input_wav_shape[-1]]
        return estimated_waveforms

    def pad_to_appropriate_length(self, x):
        values_to_pad = int(x.shape[-1]) % self.lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padded_x = torch.zeros(
                list(appropriate_shape[:-1]) + [appropriate_shape[-1] + self.lcm - values_to_pad],
                dtype=torch.float32,
            )
            padded_x[..., : x.shape[-1]] = x
            return padded_x
        return x


class SuDORMRF(_SuDORMRFBase):
    """ SuDORMRF separation model, as described in [1].

    Args:
        n_src (int): Number of sources in the input mixtures.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        num_blocks (int): Number of of UBlocks
        upsampling_depth (int): Depth of upsampling
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References:
        [1] : "Sudo rm -rf: Efficient Networks for Universal Audio Source Separation",
            Tzinis et al. MLSP 2020.
    """

    def __init__(
        self,
        n_src,
        out_chan=128,
        in_chan=None,
        num_blocks=16,
        upsampling_depth=4,
        fb_name="free",
        kernel_size=21,
        n_filters=512,
        **fb_kwargs,
    ):
        # Need the encoder to determine the number of input channels
        enc, dec = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=kernel_size // 2,
            padding=kernel_size // 2,
            output_padding=(kernel_size // 2) - 1,
            **fb_kwargs,
        )
        n_feats = enc.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )
        super().__init__(
            out_chan, n_feats, num_blocks, upsampling_depth, kernel_size, n_feats, n_src,
        )

        # Front end
        self.encoder = nn.Sequential(enc, nn.ReLU())

        # Norm before the rest, and apply one more dense layer
        self.ln = nn.GroupNorm(1, n_feats, eps=1e-08)
        self.l1 = nn.Conv1d(n_feats, out_chan, kernel_size=1)

        # Separation module
        self.sm = nn.Sequential(
            *[
                UBlock(out_chan=out_chan, in_chan=in_chan, upsampling_depth=upsampling_depth,)
                for _ in range(num_blocks)
            ]
        )

        if out_chan != n_feats:
            self.reshape_before_masks = nn.Conv1d(out_chan, n_feats, kernel_size=1)

        # Masks layer
        self.m = nn.Conv2d(
            1, n_src, kernel_size=(n_feats + 1, 1), padding=(n_feats - n_feats // 2, 0),
        )

        # Back end
        self.decoder = dec
        self.ln_mask_in = nn.GroupNorm(1, n_feats, eps=1e-08)

    def separation(self, x):
        x = self.ln(x)
        x = self.l1(x)
        x = self.sm(x)

        if self.out_chan != self.n_filters:
            # x = self.ln_bef_out_reshape(x)
            x = self.reshape_before_masks(x)

        # Get masks and apply them
        x = self.m(x.unsqueeze(1))
        if self.n_src == 1:
            x = torch.sigmoid(x)
        else:
            x = nn.functional.softmax(x, dim=1)

        return x


class SuDORMRFImproved(_SuDORMRFBase):
    """ Improved SuDORMRF separation model, as described in [1].

    Args:
        n_src (int): Number of sources in the input mixtures.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        in_chan (int, optional): Number of input channels, should be equal to
            n_filters.
        num_blocks (int): Number of of UBlocks
        upsampling_depth (int): Depth of upsampling
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``].
        n_filters (int): Number of filters / Input dimension of the masker net.
        kernel_size (int): Length of the filters.
        **fb_kwargs (dict): Additional kwards to pass to the filterbank
            creation.

    References:
        [1] : "Sudo rm -rf: Efficient Networks for Universal Audio Source Separation",
            Tzinis et al. MLSP 2020.
    """

    def __init__(
        self,
        n_src,
        out_chan=128,
        in_chan=512,
        num_blocks=16,
        upsampling_depth=4,
        fb_name="free",
        kernel_size=21,
        n_filters=512,
        **fb_kwargs,
    ):
        # Need the encoder to determine the number of input channels
        enc, dec = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=kernel_size // 2,
            padding=kernel_size // 2,
            output_padding=(kernel_size // 2) - 1,
            **fb_kwargs,
        )
        n_feats = enc.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )
        super().__init__(
            out_chan, n_feats, num_blocks, upsampling_depth, kernel_size, n_feats, n_src,
        )
        # Front end
        self.encoder = enc
        if fb_name in ["free", "analytic_free"]:
            torch.nn.init.xavier_uniform_(self.encoder.filterbank._filters)

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobLN(n_feats)
        self.bottleneck = nn.Conv1d(n_feats, out_chan, kernel_size=1)

        # Separation module
        self.sm = nn.Sequential(
            *[
                UConvBlock(out_chan=out_chan, in_chan=in_chan, upsampling_depth=upsampling_depth,)
                for _ in range(num_blocks)
            ]
        )

        mask_conv = nn.Conv1d(out_chan, n_src * n_feats, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Back end
        self.decoder = dec
        if fb_name in ["free", "analytic_free"]:
            torch.nn.init.xavier_uniform_(self.decoder.filterbank._filters)
        self.mask_nl_class = nn.ReLU()

    def separation(self, x):
        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x)

        x = self.mask_net(x)
        x = x.view(x.shape[0], self.n_src, self.n_filters, -1)
        x = self.mask_nl_class(x)
        return x


class _ConvNormAct(nn.Module):
    """ Convolution layer with normalization and a PReLU activation.

    Args
        nIn: number of input channels
        nOut: number of output channels
        kSize: kernel size
        stride: stride rate for down-sampling. Default is 1
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, use_globln=False):

        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        if use_globln:
            self.norm = GlobLN(nOut)
            self.act = nn.PReLU()
        else:
            self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
            self.act = nn.PReLU(nOut)

    def forward(self, inp):
        output = self.conv(inp)
        output = self.norm(output)
        return self.act(output)


class _ConvNorm(nn.Module):
    """ Convolution layer with normalization without activation.

    Args:
        nIn: number of input channels
        nOut: number of output channels
        kSize: kernel size
        stride: stride rate for down-sampling. Default is 1
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):

        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        self.norm = nn.GroupNorm(1, nOut, eps=1e-08)

    def forward(self, inp):
        output = self.conv(inp)
        return self.norm(output)


class _NormAct(nn.Module):
    """ Normalization and PReLU activation.

    Args:
         nOut: number of output channels
    """

    def __init__(self, nOut, use_globln=False):
        super().__init__()
        if use_globln:
            self.norm = GlobLN(nOut)
        else:
            self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.act = nn.PReLU(nOut)

    def forward(self, inp):
        output = self.norm(inp)
        return self.act(output)


class _DilatedConvNorm(nn.Module):
    """ Dilated convolution with normalized output.

    Args:
        nIn: number of input channels
        nOut: number of output channels
        kSize: kernel size
        stride: optional stride rate for down-sampling
        d: optional dilation rate
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1, use_globln=False):
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )
        if use_globln:
            self.norm = GlobLN(nOut)
        else:
            self.norm = nn.GroupNorm(1, nOut, eps=1e-08)

    def forward(self, inp):
        output = self.conv(inp)
        return self.norm(output)
