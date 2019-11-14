"""
NN blocks for separators.
@author : Manuel Pariente, Inria-Nancy
"""

from torch import nn

from . import norms, activations
from ..utils import has_arg
from ..engine.sub_module import SubModule, NoLayer



class Conv1DBlock(nn.Module):
    """One dimensional convolutional block, as proposed in [1].

    Args
        in_chan: int. Number of input channels.
        hid_chan: int. Number of hidden channels in the depth-wise convolution.
        skip_out_chan: int. Number of channels in the skip convolution.
        kernel_size: int. Size of the depth-wise convolutional kernel.
        padding: int. Padding of the depth-wise convolution.
        dilation: int. Dilation of the depth-wise convolution.
        norm_type: string. Type of normalization to use.
            Among `gLN` (global Layernorm), `cLN` (channelwise Layernorm) and
            `cgLN` (cumulative global Layernorm).

    References :
    [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking for
    speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
    https://arxiv.org/abs/1809.07454
    """
    def __init__(self, in_chan, hid_chan, skip_out_chan, kernel_size, padding,
                 dilation, norm_type="gLN"):
        super(Conv1DBlock, self).__init__()
        conv_norm = norms.get(norm_type)
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(hid_chan, hid_chan, kernel_size,
                                 padding=padding, dilation=dilation,
                                 groups=hid_chan)
        self.shared_block = nn.Sequential(in_conv1d, nn.PReLU(),
                                          conv_norm(hid_chan), depth_conv1d,
                                          nn.PReLU(), conv_norm(hid_chan))
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)
        self.skip_conv = nn.Conv1d(hid_chan, skip_out_chan, 1)

    def forward(self, x):
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        skip_out = self.skip_conv(shared_out)
        return res_out, skip_out


class TDConvNet(SubModule):
    """ Temporal Convolutional network used in ConvTasnet.

    Args
        in_chan: int > 0. Number of input filters.
        n_src: int > 0. Number of masks to estimate.
        out_chan : int > 0. Number of bins in the estimated masks.
            If None, `out_chan = in_chan`.
        n_blocks: int > 0. Number of convolutional blocks in each repeat.
            Defaults to 8
        n_repeats: int > 0. Number of repeats. Defaults to 3.
        bn_chan: int > 0. Number of channels after the bottleneck.
        hid_chan: int > 0. Number of channels in the convolutional blocks.
        skip_chan: int > 0. Number of channels in the skip connections.
        kernel_size: int > 0. Kernel size in convolutional blocks.
        norm_type: string. Among [BN, gLN, cLN]
        mask_act: string. Which non-linear function to generate mask
    """
    def __init__(self, in_chan, n_src, out_chan=None, n_blocks=8, n_repeats=3,
                 bn_chan=128, hid_chan=512, skip_chan=128, kernel_size=3,
                 norm_type="gLN", mask_act='relu'):
        super(TDConvNet, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.out_chan = out_chan if out_chan else in_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2**x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan, skip_chan,
                                            kernel_size, padding=padding,
                                            dilation=2**x, norm_type=norm_type))
        mask_conv = nn.Conv1d(bn_chan, n_src*out_chan, 1)
        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

    def forward(self, mixture_w):
        """
        Args:
            mixture_w: torch.Tensor of shape [batch, n_filters, n_frames]
        Returns:
            est_mask: torch.Tensor of shape [batch, n_src, n_filters, n_frames]
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = 0.
        for i in range(len(self.TCN)):
            residual, skip = self.TCN[i](output)
            output = output + residual
            skip_connection = skip_connection + skip
        score = self.mask_net(skip_connection)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        return est_mask

    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'out_chan': self.out_chan,
            'bn_chan': self.bn_chan,
            'hid_chan': self.hid_chan,
            'skip_chan': self.skip_chan,
            'kernel_size': self.kernel_size,
            'n_blocks': self.n_blocks,
            'n_repeats': self.n_repeats,
            'n_src': self.n_src,
            'norm_type': self.norm_type,
            'mask_act': self.mask_act
        }
        return config
