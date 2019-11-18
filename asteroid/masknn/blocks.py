"""
NN blocks for separators.
@author : Manuel Pariente, Inria-Nancy
"""

from torch import nn
from torch.nn.functional import fold, unfold

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

class SingleRNN(nn.Module):
    """
    From https://github.com/yluo42/TAC/blob/master/utility/models.py
    Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM', 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        n_layers: int > 0. number of layers used in RNN.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, n_layers=1, dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()

        assert rnn_type in ["RNN", "LSTM", "GRU"]
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = int(bidirectional) + 1

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers=n_layers,
                                             dropout=dropout, batch_first=True, bidirectional=bool(bidirectional))

        # linear projection layer
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        rnn_output, _ = self.rnn(output)
        rnn_output = self.proj(rnn_output)
        return rnn_output


class DPRNNBlock(nn.Module):
    """Dual-Path RNN Block as proposed in [1].

    Args
        in_chan: int. Number of input channels.
        hid_size: int. Number of hidden neurons in the RNNs.
        norm_type: string. Type of normalization to use.
            Among `LN` (2D Layernorm).
        bidirectional: bool. True for bidirectional Inter-Chunk RNN.
        rnn_type: string. Type of RNN used.
            Choose between 'RNN', 'LSTM' and 'GRU'.
        num_layers: int>0. Number of layers used in each RNN.
        dropout: int in (0,1).

    References :
        [1] : "Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation",
        Luo, Yi, Zhuo Chen, and Takuya Yoshioka.
        https://arxiv.org/abs/1910.06379
    """
    def __init__(self, in_chan, hid_size, norm_type="LN",
                 bidirectional=True, rnn_type="LSTM", num_layers=1, dropout=0):
        super(DPRNNBlock, self).__init__()
        self.intra_RNN=SingleRNN(rnn_type, in_chan, hid_size, num_layers, dropout=dropout, bidirectional=True)
        self.intra_norm=norms.get(norm_type)(in_chan)
        self.inter_RNN=SingleRNN(rnn_type, in_chan, hid_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.inter_norm = norms.get(norm_type)(in_chan)


    def forward(self, x):
        # x is [batch, num_features, chunk_size, num_chunks]
        B, N, K, L = x.size()
        output = x # for skip connection
        # intra-chunk processing
        x = x.transpose(1, -1).reshape(B * L, K, N)
        x = self.intra_RNN(x)
        x = x.reshape(B, L, K, N).transpose(1, -1)
        x = self.intra_norm(x)
        output = output + x
        # inter-chunk processing
        output = output.transpose(1, 2).transpose(2, -1).reshape(B * K, L, N)
        output = self.inter_RNN(output)
        output = output.reshape(B, K, L, N).transpose(1, -1).transpose(2, -1)
        output = self.inter_norm(output)
        return output + x


class DPRNN(SubModule):
    """ Dual-path RNN Network for Single-Channel Source Separation introduced in [1].
    Args
        in_chan: int > 0. Number of input filters.
        out_chan : int > 0. Number of bins in the estimated masks.
        bn_chan: int > 0. Number of channels after the bottleneck.
        hid_size: int > 0. Number of neurons in the RNNs.
        kernel_size: int > 0. Kernel size in convolutional blocks.
        n_blocks: int > 0. Number of convolutional blocks in each repeat.
        n_repeats: int > 0. Number of repeats.
        n_src: int > 0. Number of masks to estimate.
        norm_type: string. Among [BN, gLN, cLN]
        mask_act: string. Which non-linear function to generate mask.
        bidirectional: bool: True for bidirectional Inter-Chunk RNN (Intra-Chunk is always bidirectional).
        rnn_type: string. Type of RNN used. Choose between 'RNN', 'LSTM' and 'GRU'.
        num_layers: number of layers in each RNN.
        dropout: int in (0,1).

    References :
        [1] : "Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation",
        Luo, Yi, Zhuo Chen, and Takuya Yoshioka.
        https://arxiv.org/abs/1910.06379
    """
    def __init__(self, in_chan, out_chan, bn_chan, hid_size,
                 chunk_size, hop_size, n_repeats, n_src, norm_type="LN",
                 mask_act='sigmoid', bidirectional=True, rnn_type="LSTM", num_layers=1, dropout=0):
        super(DPRNN, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.bn_chan = bn_chan
        self.hid_size = hid_size
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.n_src = n_src
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout=dropout

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        net = [] # Succession of DPRNNBlocks.
        for x in range(self.n_repeats):
            net += [DPRNNBlock(bn_chan, hid_size, norm_type, bidirectional, rnn_type, num_layers, dropout)]
        self.net = nn.Sequential(*net)

        mask_conv = nn.Conv2d(bn_chan, n_src*out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Get activation function. For softmax, feed the source dimension.
        if mask_act.lower() == 'linear':
            mask_nl_class = NoLayer
        else:
            mask_nl_class = getattr(nn, mask_act)
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        """
        Args:
            mixture_w: torch.Tensor of shape [batch, n_filters, n_frames]
        Returns:
            est_mask: torch.Tensor of shape [batch, n_src, n_filters, n_frames]
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w) # [batch x bn_chan x n_frames]
        output = unfold(output.unsqueeze(-1), kernel_size=(self.chunk_size, 1), padding=(self.chunk_size, 0),
                        stride=(self.hop_size, 1))
        S = output.size(-1)
        output = output.reshape(batch, self.bn_chan, self.chunk_size, S) # [batch x bn_chan x chunk_size x n_chunks]
        output = self.net(output) # apply stacked DPRNN Blocks sequentially
        output = self.mask_net(output) # apply mask
        output = output.reshape(batch * self.n_src, n_filters, self.chunk_size, S)
        # overlap and add: [batch x bn_chan x chunk_size x n_chunks] -> [batch x bn_chan x n_frames]
        output = fold(output.reshape(batch, self.bn_chan * self.chunk_size, S),
                                 (n_frames, 1), kernel_size=(self.chunk_size, 1),
                                 padding=(self.chunk_size, 0),
                                 stride=(self.hop_size, 1))
        output = output.squeeze(-1) / (self.K / self.P) # normalization
        score = output.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        return est_mask

    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'out_chan': self.out_chan,
            'bn_chan': self.bn_chan,
            'hid_size': self.hid_size,
            'chunk_size': self.kernel_size,
            'hop_size': self.hop_size,
            'n_repeats': self.n_repeats,
            'n_src': self.n_src,
            'norm_type': self.norm_type,
            'mask_act': self.mask_act,
            'bidirectional': self.bidirectional,
            'rnn_type': self.rnn_type,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }
        return config
