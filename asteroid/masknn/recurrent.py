import torch
from torch import nn
from torch.nn.functional import fold, unfold

from . import norms, activations
from .norms import GlobLN, CumLN
from ..utils import has_arg


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
        bidirectional (bool, optional): Whether the RNN layers are
            bidirectional. Default is ``False``.
    """

    def __init__(
        self, rnn_type, input_size, hidden_size, n_layers=1, dropout=0, bidirectional=False
    ):
        super(SingleRNN, self).__init__()
        assert rnn_type.upper() in ["RNN", "LSTM", "GRU"]
        rnn_type = rnn_type.upper()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bool(bidirectional),
        )

    def forward(self, inp):
        """ Input shape [batch, seq, feats] """
        self.rnn.flatten_parameters()  # Enables faster multi-GPU training.
        output = inp
        rnn_output, _ = self.rnn(output)
        return rnn_output


class StackedResidualRNN(nn.Module):
    """ Stacked RNN with builtin residual connection.
    Only supports forward RNNs.
    See StackedResidualBiRNN for bidirectional ones.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        n_units (int): Number of units in recurrent layers. This will also be
            the expected input size.
        n_layers (int): Number of recurrent layers.
        dropout (float): Dropout value, between 0. and 1. (Default: 0.)
        bidirectional (bool): If True, use bidirectional RNN, else
            unidirectional. (Default: False)
    """

    def __init__(self, rnn_type, n_units, n_layers=4, dropout=0.0, bidirectional=False):
        super(StackedResidualRNN, self).__init__()
        self.rnn_type = rnn_type
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout = dropout
        assert bidirectional is False, "Bidirectional not supported yet"
        self.bidirectional = bidirectional

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                SingleRNN(
                    rnn_type, input_size=n_units, hidden_size=n_units, bidirectional=bidirectional
                )
            )
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        """ Builtin residual connections + dropout applied before residual.
            Input shape : [batch, time_axis, feat_axis]
        """
        for rnn in self.layers:
            rnn_out = rnn(x)
            dropped_out = self.dropout_layer(rnn_out)
            x = x + dropped_out
        return x


class StackedResidualBiRNN(nn.Module):
    """ Stacked Bidirectional RNN with builtin residual connection.
    Residual connections are applied on both RNN directions.
    Only supports bidiriectional RNNs.
    See StackedResidualRNN for unidirectional ones.

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        n_units (int): Number of units in recurrent layers. This will also be
            the expected input size.
        n_layers (int): Number of recurrent layers.
        dropout (float): Dropout value, between 0. and 1. (Default: 0.)
        bidirectional (bool): If True, use bidirectional RNN, else
            unidirectional. (Default: False)
    """

    def __init__(self, rnn_type, n_units, n_layers=4, dropout=0.0, bidirectional=True):
        super().__init__()
        self.rnn_type = rnn_type
        self.n_units = n_units
        self.n_layers = n_layers
        self.dropout = dropout
        assert bidirectional is True, "Only bidirectional not supported yet"
        self.bidirectional = bidirectional

        # The first layer has as many units as input size
        self.first_layer = SingleRNN(
            rnn_type, input_size=n_units, hidden_size=n_units, bidirectional=bidirectional
        )
        # As the first layer outputs 2*n_units, the following layers need
        # 2*n_units as input size
        self.layers = nn.ModuleList()
        for i in range(n_layers - 1):
            input_size = 2 * n_units
            self.layers.append(
                SingleRNN(
                    rnn_type,
                    input_size=input_size,
                    hidden_size=n_units,
                    bidirectional=bidirectional,
                )
            )
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        """ Builtin residual connections + dropout applied before residual.
            Input shape : [batch, time_axis, feat_axis]
        """
        # First layer
        rnn_out = self.first_layer(x)
        dropped_out = self.dropout_layer(rnn_out)
        x = torch.cat([x, x], dim=-1) + dropped_out
        # Rest of the layers
        for rnn in self.layers:
            rnn_out = rnn(x)
            dropped_out = self.dropout_layer(rnn_out)
            x = x + dropped_out
        return x


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

    def __init__(
        self,
        in_chan,
        hid_size,
        norm_type="gLN",
        bidirectional=True,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
    ):
        super(DPRNNBlock, self).__init__()
        # IntraRNN and linear projection layer (always bi-directional)
        self.intra_RNN = SingleRNN(
            rnn_type, in_chan, hid_size, num_layers, dropout=dropout, bidirectional=True
        )
        self.intra_linear = nn.Linear(hid_size * 2, in_chan)
        self.intra_norm = norms.get(norm_type)(in_chan)
        # InterRNN block and linear projection layer (uni or bi-directional)
        self.inter_RNN = SingleRNN(
            rnn_type, in_chan, hid_size, num_layers, dropout=dropout, bidirectional=bidirectional
        )
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
        x = output.transpose(1, 2).transpose(2, -1).reshape(B * K, L, N)
        x = self.inter_RNN(x)
        x = self.inter_linear(x)
        x = x.reshape(B, K, L, N).transpose(1, -1).transpose(2, -1)
        x = self.inter_norm(x)
        return output + x


class DPRNN(nn.Module):
    """ Dual-path RNN Network for Single-Channel Source Separation
        introduced in [1].

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan  (int or None): Number of bins in the estimated masks.
            Defaults to `in_chan`.
        bn_chan (int): Number of channels after the bottleneck.
            Defaults to 128.
        hid_size (int): Number of neurons in the RNNs cell state.
            Defaults to 128.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
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

    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        bn_chan=128,
        hid_size=128,
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        norm_type="gLN",
        mask_act="relu",
        bidirectional=True,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
    ):
        super(DPRNN, self).__init__()
        self.in_chan = in_chan
        out_chan = out_chan if out_chan is not None else in_chan
        self.out_chan = out_chan
        self.bn_chan = bn_chan
        self.hid_size = hid_size
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
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
            net += [
                DPRNNBlock(
                    bn_chan,
                    hid_size,
                    norm_type=norm_type,
                    bidirectional=bidirectional,
                    rnn_type=rnn_type,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            ]
        self.net = nn.Sequential(*net)
        # Masking in 3D space
        net_out_conv = nn.Conv2d(bn_chan, n_src * bn_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        # Gating and masking in 2D space (after fold)
        self.net_out = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Sigmoid())
        self.mask_net = nn.Conv1d(bn_chan, out_chan, 1, bias=False)

        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
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
        output = unfold(
            output.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        n_chunks = output.size(-1)
        output = output.reshape(batch, self.bn_chan, self.chunk_size, n_chunks)
        # Apply stacked DPRNN Blocks sequentially
        output = self.net(output)
        # Map to sources with kind of 2D masks
        output = self.first_out(output)
        output = output.reshape(batch * self.n_src, self.bn_chan, self.chunk_size, n_chunks)
        # Overlap and add:
        # [batch, out_chan, chunk_size, n_chunks] -> [batch, out_chan, n_frames]
        to_unfold = self.bn_chan * self.chunk_size
        output = fold(
            output.reshape(batch * self.n_src, to_unfold, n_chunks),
            (n_frames, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        # Apply gating
        output = output.reshape(batch * self.n_src, self.bn_chan, -1)
        output = self.net_out(output) * self.net_gate(output)
        # Compute mask
        score = self.mask_net(output)
        est_mask = self.output_act(score)
        est_mask = est_mask.view(batch, self.n_src, self.out_chan, n_frames)
        return est_mask

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "out_chan": self.out_chan,
            "bn_chan": self.bn_chan,
            "hid_size": self.hid_size,
            "chunk_size": self.chunk_size,
            "hop_size": self.hop_size,
            "n_repeats": self.n_repeats,
            "n_src": self.n_src,
            "norm_type": self.norm_type,
            "mask_act": self.mask_act,
            "bidirectional": self.bidirectional,
            "rnn_type": self.rnn_type,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
        return config


class LSTMMasker(nn.Module):
    """ LSTM mask network introduced in [1], without skip connections.

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan  (int or None): Number of bins in the estimated masks.
            Defaults to `in_chan`.
        rnn_type (str, optional): Type of RNN used. Choose between ``'RNN'``,
            ``'LSTM'`` and ``'GRU'``.
        n_layers (int, optional): Number of layers in each RNN.
        hid_size (int): Number of neurons in the RNNs cell state.
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): Whether to use BiLSTM
        dropout (float, optional): Dropout ratio, must be in [0,1].

    References:
        [1]: Yi Luo et al. "Real-time Single-channel Dereverberation and Separation
            with Time-domain Audio Separation Network", Interspeech 2018
    """

    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        rnn_type="lstm",
        n_layers=4,
        hid_size=512,
        dropout=0.3,
        mask_act="sigmoid",
        bidirectional=True,
    ):
        super().__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan is not None else in_chan
        self.out_chan = out_chan
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.hid_size = hid_size
        self.dropout = dropout
        self.mask_act = mask_act
        self.bidirectional = bidirectional

        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

        # Create TasNet masker
        out_size = hid_size * (int(bidirectional) + 1)
        if bidirectional:
            self.bn_layer = GlobLN(in_chan)
        else:
            self.bn_layer = CumLN(in_chan)
        self.masker = nn.Sequential(
            SingleRNN(
                "lstm",
                in_chan,
                hidden_size=hid_size,
                n_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
            ),
            nn.Linear(out_size, self.n_src * out_chan),
            self.output_act,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        to_sep = self.bn_layer(x)
        est_masks = self.masker(to_sep.transpose(-1, -2)).transpose(-1, -2)
        est_masks = est_masks.view(batch_size, self.n_src, self.out_chan, -1)
        return est_masks

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "n_src": self.n_src,
            "out_chan": self.out_chan,
            "rnn_type": self.rnn_type,
            "n_layers": self.n_layers,
            "hid_size": self.hid_size,
            "dropout": self.dropout,
            "mask_act": self.mask_act,
            "bidirectional": self.bidirectional,
        }
        return config
