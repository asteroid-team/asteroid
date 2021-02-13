import json
import os
import torch
from torch import nn
from torch.nn.functional import fold, unfold


from asteroid_filterbanks import make_enc_dec
from asteroid.engine.optimizers import make_optimizer
from asteroid.masknn import activations, norms
from asteroid.masknn.recurrent import DPRNNBlock
from asteroid.utils import has_arg


class DPRNN_Multistage(nn.Module):
    """Implementation of the Dual-Path-RNN model,
        with multi-stage output, without Conv2D projection

    Args:
        in_chan: The number of expected features in the input x
        out_channels: The number of features in the hidden state h
        rnn_type: RNN, LSTM, GRU
        norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
        dropout: If non-zero, introduces a Dropout layer on the outputs
                    of each LSTM layer except the last layer,
                    with dropout probability equal to dropout. Default: 0
        bidirectional: If True, becomes a bidirectional LSTM. Default: False
        num_layers: number of Dual-Path-Block
        K: the length of chunk
        num_spks: the number of speakers
    """

    def __init__(
        self,
        in_chan,
        bn_chan,
        hid_size,
        chunk_size,
        hop_size=None,
        n_repeats=6,
        norm_type="gLN",
        bidirectional=True,
        rnn_type="LSTM",
        use_mulcat=True,
        num_layers=1,
        dropout=0,
    ):
        super(DPRNN_Multistage, self).__init__()
        self.in_chan = in_chan
        self.bn_chan = bn_chan
        self.hid_size = hid_size
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.norm_type = norm_type
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_mulcat = use_mulcat
        self.num_layers = num_layers

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        # Succession of DPRNNBlocks.
        self.net = nn.ModuleList([])
        for i in range(self.n_repeats):
            self.net.append(
                DPRNNBlock(
                    bn_chan,
                    hid_size,
                    norm_type=norm_type,
                    bidirectional=bidirectional,
                    rnn_type=rnn_type,
                    use_mulcat=use_mulcat,
                    num_layers=num_layers,
                    dropout=dropout,
                )
            )

    def forward(self, mixture_w):
        r"""Forward.

        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape $(batch, nfilters, nframes)$

        Returns:
            :class:`torch.Tensor`: estimated mask of shape $(batch, nsrc, nfilters, nframes)$
        """
        batch, n_filters, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)  # [batch, bn_chan, n_frames]
        output = unfold(
            output.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        n_chunks = output.shape[-1]
        output = output.reshape(batch, self.bn_chan, self.chunk_size, n_chunks)
        # Apply stacked DPRNN Blocks sequentially
        output_list = []
        for i in range(self.num_layers):
            output = self.dual_rnn[i](output)
            output_list.append(output)
        return output_list


class SingleDecoder(nn.Module):
    def __init__(
        self, kernel_size, in_chan, n_src, bn_chan, chunk_size, hop_size=None, mask_act="relu"
    ):
        super(SingleDecoder, self).__init__()
        self.kernel_size = kernel_size
        self.in_chan = in_chan
        self.bn_chan = bn_chan
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_src = n_src
        self.mask_act = mask_act

        # Masking in 3D space
        net_out_conv = nn.Conv2d(bn_chan, n_src * bn_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        # Gating and masking in 2D space (after fold)
        self.net_out = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Sigmoid())
        self.mask_net = nn.Conv1d(bn_chan, in_chan, 1, bias=False)

        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

        _, self.trans_conv = make_enc_dec("free", kernel_size=kernel_size, n_filters=in_chan)

    def forward(self, x, e, gap):
        """
        args:
            x: [num_stages, out_channels, K, S]
            e: [in_channels, L]
        outputs:
            x: [num_stages, num_spks, T]
        """
        x = self.prelu(x)
        # [num_stages, num_spks * out_channels, K, S]
        x = self.conv2d(x)
        num_stages, _, K, S = x.shape
        # [num_stages * num_spks, out_channels, K, S]
        x = x.view(num_stages * self.num_spks, self.out_channels, K, S)
        # [num_stages * num_spks, out_channels, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)
        # [num_stages * num_spks, in_channels, L]
        x = self.end_conv1x1(x)
        _, N, L = x.shape
        # [num_stages, num_spks, in_channels, L]
        x = x.view(num_stages, self.num_spks, N, L)
        x = self.activation(x)
        # [1, 1, in_channels, L]
        e = e.unsqueeze(0).unsqueeze(1)
        x = x * e
        # [num_stages * num_spks, N, L]
        x = x.view(num_stages * self.num_spks, N, L)
        # [num_stages, num_spks, T]
        x = self.decoder(x).view(num_stages, self.num_spks, -1)

        return x