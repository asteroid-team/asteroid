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
from asteroid.masknn.norms import GlobLN


class ConvNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation

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

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    """
    This class defines the convolution layer with normalization and PReLU activation

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

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    """
    This class defines a normalization and PReLU activation
    Args
         nOut: number of output channels
    """

    def __init__(self, nOut, use_globln=False):
        super().__init__()
        if use_globln:
            self.norm = GlobLN(nOut)
        else:
            self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConvNorm(nn.Module):
    """
    This class defines the dilated convolution with normalized output.
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

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class BaseUBlock(nn.Module):
    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, use_globln=False):
        super().__init__()
        self.proj_1x1 = ConvNormAct(
            out_channels, in_channels, 1, stride=1, groups=1, use_globln=use_globln
        )
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DilatedConvNorm(
                in_channels,
                in_channels,
                kSize=5,
                stride=1,
                groups=in_channels,
                d=1,
                use_globln=use_globln,
            )
        )

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(
                DilatedConvNorm(
                    in_channels,
                    in_channels,
                    kSize=2 * stride + 1,
                    stride=stride,
                    groups=in_channels,
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


class UBlock(BaseUBlock):
    """
    This class defines the Upsampling block, which is based on the following
    principle:
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    """

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4):
        super().__init__(out_channels, in_channels, upsampling_depth, use_globln=False)
        self.conv_1x1_exp = ConvNorm(in_channels, out_channels, 1, 1, groups=1)
        self.final_norm = NormAct(in_channels)
        self.module_act = NormAct(out_channels)

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


class UConvBlock(BaseUBlock):
    """
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    """

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4):
        super().__init__(out_channels, in_channels, upsampling_depth, use_globln=True)
        self.final_norm = NormAct(in_channels, use_globln=True)
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)

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


class SuDORMRFBase(nn.Module):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=21,
        enc_num_basis=512,
        num_sources=2,
    ):
        super().__init__()

        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(self.enc_kernel_size // 2 * 2 ** self.upsampling_depth) // math.gcd(
            self.enc_kernel_size // 2, 2 ** self.upsampling_depth
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
        estimated_waveforms = self.decoder((x * s).view(x.shape[0], -1, x.shape[-1]))
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


class SuDORMRF(SuDORMRFBase):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=21,
        enc_num_basis=512,
        num_sources=2,
    ):
        super().__init__(
            out_channels,
            in_channels,
            num_blocks,
            upsampling_depth,
            enc_kernel_size,
            enc_num_basis,
            num_sources,
        )
        # Front end
        self.encoder = nn.Sequential(
            *[
                nn.Conv1d(
                    in_channels=1,
                    out_channels=enc_num_basis,
                    kernel_size=enc_kernel_size,
                    stride=enc_kernel_size // 2,
                    padding=enc_kernel_size // 2,
                ),
                nn.ReLU(),
            ]
        )

        # Norm before the rest, and apply one more dense layer
        self.ln = nn.GroupNorm(1, enc_num_basis, eps=1e-08)
        self.l1 = nn.Conv1d(in_channels=enc_num_basis, out_channels=out_channels, kernel_size=1)

        # Separation module
        self.sm = nn.Sequential(
            *[
                UBlock(
                    out_channels=out_channels,
                    in_channels=in_channels,
                    upsampling_depth=upsampling_depth,
                )
                for r in range(num_blocks)
            ]
        )

        if out_channels != enc_num_basis:
            self.reshape_before_masks = nn.Conv1d(
                in_channels=out_channels, out_channels=enc_num_basis, kernel_size=1
            )

        # Masks layer
        self.m = nn.Conv2d(
            in_channels=1,
            out_channels=num_sources,
            kernel_size=(enc_num_basis + 1, 1),
            padding=(enc_num_basis - enc_num_basis // 2, 0),
        )

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_num_basis * num_sources,
            out_channels=num_sources,
            output_padding=(enc_kernel_size // 2) - 1,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2,
            groups=num_sources,
        )
        self.ln_mask_in = nn.GroupNorm(1, enc_num_basis, eps=1e-08)

    def separation(self, x):
        x = self.ln(x)
        x = self.l1(x)
        x = self.sm(x)

        if self.out_channels != self.enc_num_basis:
            # x = self.ln_bef_out_reshape(x)
            x = self.reshape_before_masks(x)

        # Get masks and apply them
        x = self.m(x.unsqueeze(1))
        if self.num_sources == 1:
            x = torch.sigmoid(x)
        else:
            x = nn.functional.softmax(x, dim=1)

        return x


class SuDORMRFImproved(SuDORMRFBase):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=21,
        enc_num_basis=512,
        num_sources=2,
    ):
        super().__init__(
            out_channels,
            in_channels,
            num_blocks,
            upsampling_depth,
            enc_kernel_size,
            enc_num_basis,
            num_sources,
        )
        # Front end
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=enc_num_basis,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2,
            bias=False,
        )
        torch.nn.init.xavier_uniform(self.encoder.weight)

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobLN(enc_num_basis)
        self.bottleneck = nn.Conv1d(
            in_channels=enc_num_basis, out_channels=out_channels, kernel_size=1
        )

        # Separation module
        self.sm = nn.Sequential(
            *[
                UConvBlock(
                    out_channels=out_channels,
                    in_channels=in_channels,
                    upsampling_depth=upsampling_depth,
                )
                for _ in range(num_blocks)
            ]
        )

        mask_conv = nn.Conv1d(out_channels, num_sources * enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_num_basis * num_sources,
            out_channels=num_sources,
            output_padding=(enc_kernel_size // 2) - 1,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2,
            groups=1,
            bias=False,
        )
        torch.nn.init.xavier_uniform(self.decoder.weight)
        self.mask_nl_class = nn.ReLU()

    def separation(self, x):
        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x)

        x = self.mask_net(x)
        x = x.view(x.shape[0], self.num_sources, self.enc_num_basis, -1)
        x = self.mask_nl_class(x)

        return x
