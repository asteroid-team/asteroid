"""
| NN blocks for separators.
| @author : Manuel Pariente, Inria-Nancy
"""

from torch import nn
from torch.nn.functional import fold, unfold

from . import norms, activations
from ..utils import has_arg


class Conv1DBlock(nn.Module):
    """One dimensional convolutional block, as proposed in [1].

    Args:
        in_chan (int): Number of input channels.
        hid_chan (int): Number of hidden channels in the depth-wise
            convolution.
        skip_out_chan (int): Number of channels in the skip convolution.
        kernel_size (int): Size of the depth-wise convolutional kernel.
        padding (int): Padding of the depth-wise convolution.
        dilation (int): Dilation of the depth-wise convolution.
        norm_type (str, optional): Type of normalization to use. To choose from

            -  ``'gLN'``: global Layernorm
            -  ``'cLN'``: channelwise Layernorm
            -  ``'cgLN'``: cumulative global Layernorm

    References:
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
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
        """ Input shape [batch, feats, seq]"""
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        skip_out = self.skip_conv(shared_out)
        return res_out, skip_out


class TDConvNet(nn.Module):
    """ Temporal Convolutional network used in ConvTasnet.

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
        kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.
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
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2**x // 2
                self.TCN.append(Conv1DBlock(bn_chan, hid_chan, skip_chan,
                                            kernel_size, padding=padding,
                                            dilation=2**x, norm_type=norm_type))
        mask_conv = nn.Conv1d(bn_chan, n_src*out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        """

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape
                [batch, n_filters, n_frames]

        Returns:
            :class:`torch.Tensor`:
                estimated mask of shape [batch, n_src, n_filters, n_frames]
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
    """ Module for a RNN block.

    Inspired from https://github.com/yluo42/TAC/blob/master/utility/models.py
    Licensed under CC BY-NC-SA 3.0 US.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        input_size (int): Dimension of the input feature. The input should have
            shape [batch, seq_len, input_size].
        hidden_size (int): Dimension of the hidden state.
        n_layers (int, optional): Number of layers used in RNN. Default is 1.
        dropout (float, optional): Dropout ratio. Default is 0.
        bidirectional (bool): Whether the RNN layers are bidirectional.
            Default is ``False``.
    """

    def __init__(self, rnn_type, input_size, hidden_size, n_layers=1,
                 dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()
        assert rnn_type.upper() in ["RNN", "LSTM", "GRU"]
        self.rnn_type = rnn_type.upper()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size,
                                         num_layers=n_layers,
                                         dropout=dropout,
                                         batch_first=True,
                                         bidirectional=bool(bidirectional))

    def forward(self, inp):
        """ Input shape [batch, seq, feats] """
        output = inp
        rnn_output, _ = self.rnn(output)
        return rnn_output


class DPRNNBlock(nn.Module):
    """ Dual-Path RNN Block as proposed in [1].

    Args:
        in_chan (int): Number of input channels.
        hid_size (int): Number of hidden neurons in the RNNs.
        norm_type (str, optional): Type of normalization to use. To choose from
            - ``'gLN'``: global Layernorm
            - ``'cLN'``: channelwise Layernorm
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN.
        rnn_type (str, optional): Type of RNN used. Choose from ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        num_layers (int, optional): Number of layers used in each RNN.
        dropout (float, optional): Dropout ratio. Must be in [0, 1].

    References:
        [1] "Dual-path RNN: efficient long sequence modeling for
        time-domain single-channel speech separation", Yi Luo, Zhuo Chen
        and Takuya Yoshioka. https://arxiv.org/abs/1910.06379
    """
    def __init__(self, in_chan, hid_size, norm_type="gLN", bidirectional=True,
                 rnn_type="LSTM", num_layers=1, dropout=0):
        super(DPRNNBlock, self).__init__()
        # IntraRNN and linear projection layer (always bi-directional)
        self.intra_RNN = SingleRNN(rnn_type, in_chan, hid_size, num_layers,
                                   dropout=dropout, bidirectional=True)
        self.intra_linear = nn.Linear(hid_size * 2, in_chan)
        self.intra_norm = norms.get(norm_type)(in_chan)
        # InterRNN block and linear projection layer (uni or bi-directional)
        self.inter_RNN = SingleRNN(rnn_type, in_chan, hid_size, num_layers,
                                   dropout=dropout, bidirectional=bidirectional)
        num_direction = int(bidirectional) + 1
        self.inter_linear = nn.Linear(hid_size * num_direction, in_chan)
        self.inter_norm = norms.get(norm_type)(in_chan)

    def forward(self, x):
        """ Input shape : [batch, feats, chunk_size, num_chunks] """
        B, N, K, L = x.size()
        output = x  # for skip connection
        # Intra-chunk processing
        x = x.transpose(1, -1).reshape(B * L, K, N)
        x = self.intra_RNN(x)
        x = self.intra_linear(x)
        x = x.reshape(B, L, K, N).transpose(1, -1)
        x = self.intra_norm(x)
        output = output + x
        # Inter-chunk processing
        output = output.transpose(1, 2).transpose(2, -1).reshape(B * K, L, N)
        output = self.inter_RNN(output)
        output = self.inter_linear(output)
        output = output.reshape(B, K, L, N).transpose(1, -1).transpose(2, -1)
        output = self.inter_norm(output)
        return output + x


class DPRNN(nn.Module):
    """ Dual-path RNN Network for Single-Channel Source Separation

        Method introduced in [1].

    Args:
        in_chan (int): Number of input filters.
        out_chan  (int): Number of bins in the estimated masks.
        bn_chan (int): Number of channels after the bottleneck.
        hid_size (int): Number of neurons in the RNNs cell state.
        chunk_size (int): window size of overlap and add processing.
        hop_size (int): hop size (stride) of overlap and add processing.
        n_repeats (int): Number of repeats.
        n_src (int): Number of masks to estimate.
        norm_type (str, optional): Type of normalization to use. To choose from

            - ``'gLN'``: global Layernorm
            - ``'cLN'``: channelwise Layernorm
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        rnn_type (str, optional): Type of RNN used. Choose between ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        num_layers (int, optional): Number of layers in each RNN.
        dropout (float, optional): Dropout ratio, must be in [0,1].

    References:
        [1] "Dual-path RNN: efficient long sequence modeling for
        time-domain single-channel speech separation", Yi Luo, Zhuo Chen
        and Takuya Yoshioka. https://arxiv.org/abs/1910.06379
    """
    def __init__(self, in_chan, out_chan, bn_chan, hid_size, chunk_size,
                 hop_size, n_repeats, n_src, norm_type="gLN",
                 mask_act='sigmoid', bidirectional=True, rnn_type="LSTM",
                 num_layers=1, dropout=0):
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
        self.dropout = dropout

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        # Succession of DPRNNBlocks.
        net = []
        for x in range(self.n_repeats):
            net += [DPRNNBlock(bn_chan, hid_size, norm_type, bidirectional,
                               rnn_type, num_layers, dropout)]
        self.net = nn.Sequential(*net)

        mask_conv = nn.Conv2d(bn_chan, n_src*out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, 'dim'):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        """
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape
                [batch, n_filters, n_frames]
        Returns:
            :class:`torch.Tensor`
                estimated mask of shape [batch, n_src, n_filters, n_frames]
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)  # [batch, bn_chan, n_frames]
        output = unfold(output.unsqueeze(-1), kernel_size=(self.chunk_size, 1),
                        padding=(self.chunk_size, 0), stride=(self.hop_size, 1))
        n_chunks = output.size(-1)
        output = output.reshape(batch, self.bn_chan, self.chunk_size, n_chunks)
        # Apply stacked DPRNN Blocks sequentially
        output = self.net(output)
        output = self.mask_net(output)
        output = output.reshape(batch * self.n_src, n_filters, self.chunk_size,
                                n_chunks)
        # Overlap and add:
        # [batch, bn_chan, chunk_size, n_chunks] -> [batch, bn_chan, n_frames]
        to_unfold = self.bn_chan * self.chunk_size
        output = fold(output.reshape(batch, to_unfold, n_chunks),
                      (n_frames, 1), kernel_size=(self.chunk_size, 1),
                      padding=(self.chunk_size, 0),
                      stride=(self.hop_size, 1))
        # Normalization
        output = output.squeeze(-1) / (self.chunk_size / self.hop_size)
        score = output.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        return est_mask

    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'out_chan': self.out_chan,
            'bn_chan': self.bn_chan,
            'hid_size': self.hid_size,
            'chunk_size': self.chunk_size,
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
