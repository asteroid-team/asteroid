from torch import nn
from .norms import GlobLN


class _ConvNormAct(nn.Module):
    """Convolution layer with normalization and a PReLU activation.

    See license and copyright notices here
        https://github.com/etzinis/sudo_rm_rf#copyright-and-license
        https://github.com/etzinis/sudo_rm_rf/blob/master/LICENSE

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
    """Convolution layer with normalization without activation.

    See license and copyright notices here
        https://github.com/etzinis/sudo_rm_rf#copyright-and-license
        https://github.com/etzinis/sudo_rm_rf/blob/master/LICENSE


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
    """Normalization and PReLU activation.

    See license and copyright notices here
        https://github.com/etzinis/sudo_rm_rf#copyright-and-license
        https://github.com/etzinis/sudo_rm_rf/blob/master/LICENSE

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
    """Dilated convolution with normalized output.

    See license and copyright notices here
        https://github.com/etzinis/sudo_rm_rf#copyright-and-license
        https://github.com/etzinis/sudo_rm_rf/blob/master/LICENSE

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
