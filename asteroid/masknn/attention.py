import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from asteroid.masknn import activations, norms
import torch
from asteroid.utils import has_arg

class ImprovedTransformedLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout=0., activation="relu", bidirectional=True, norm="gLN"):
        super(ImprovedTransformedLayer, self).__init__()

        self.mha = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.recurrent = nn.LSTM(d_model, dim_ff, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        ff_inner_dim = 2*dim_ff if bidirectional else dim_ff
        self.linear = nn.Linear(ff_inner_dim, d_model)
        self.activation = activations.get(activation)()
        self.norm_mha = norms.get(norm)(d_model)
        self.norm_ff = norms.get(norm)(d_model)

    def forward(self, x):
        x = x.transpose(1, -1)
        # x is batch, seq_len, channels
        # self-attention is applied
        out = self.mha(x, x, x)[0]
        x = self.dropout(out) + x
        x = self.norm_mha(x.transpose(1, -1)).transpose(1, -1)

        # lstm is applied
        out = self.linear(self.dropout(self.activation(self.recurrent(x)[0])))
        x = self.dropout(out) + x
        return self.norm_ff(x.transpose(1, -1))


class Oladd(nn.Module):

    def __init__(self, chunk_size, hop_size):
        super(Oladd, self).__init__()
        self.chunk_size = chunk_size
        self.hop_size = hop_size
        self.n_orig_frames = None

    def unfold(self, x):
        # x is (batch, chan, frames)
        batch, chan, frames = x.size()
        assert x.ndim == 3
        self.n_orig_frames = x.shape[-1]
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )

        return unfolded.reshape(batch, chan, self.chunk_size, -1) # (batch, chan, chunk_size, n_chunks)

    def fold(self, x):
        # x is (batch, chan, chunk_size, n_chunks)
        batch, chan, chunk_size, n_chunks = x.size()
        to_unfold = x.reshape(batch, chan * self.chunk_size, n_chunks)
        x = torch.nn.functional.fold(
            to_unfold,
            (self.n_orig_frames, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )

        x /= (self.chunk_size / self.hop_size)

        return x.reshape(batch, chan, self.n_orig_frames)

    def intra_process(self, x, module):

        # x is (batch, channels, chunk_size, n_chunks)
        batch, channels, chunk_size, n_chunks = x.size()
        # we reshape to batch*chunk_size, channels, n_chunks
        x = x.transpose(1, -1).reshape(batch*n_chunks, chunk_size, channels).transpose(1, -1)
        x = module(x)
        x = x.reshape(batch, n_chunks, channels, chunk_size).transpose(1, -1).transpose(1, 2)
        return x


    def inter_process(self, x, module):
        batch, channels, chunk_size, n_chunks = x.size()
        x = x.transpose(1, 2).reshape(batch*chunk_size, channels, n_chunks)
        x = module(x)
        x = x.reshape(batch, chunk_size, channels, n_chunks).transpose(1, 2)
        return x


class DPTNet(nn.Module):
    """ Dual-path Transformer
        introduced in [1].

    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan  (int or None): Number of bins in the estimated masks.
            Defaults to `in_chan`.
        hid_ff (int): Number of neurons in the RNNs cell state.
            Defaults to 128.
        chunk_size (int): window size of overlap and add processing.
            Defaults to 100.
        hop_size (int or None): hop size (stride) of overlap and add processing.
            Default to `chunk_size // 2` (50% overlap).
        n_repeats (int): Number of repeats. Defaults to 6.
        norm_type (str, optional): Type of normalization to use.
        mask_act (str, optional): Which non-linear function to generate mask.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        dropout (float, optional): Dropout ratio, must be in [0,1].

    References:
        [1]
    """

    def __init__(
        self,
        in_chan,
        n_src,
        n_heads=4,
        ff_hid=256,
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        norm_type="gLN",
        ff_activation="relu",
        mask_act='relu',
        bidirectional=True,
        dropout=0,
    ):
        super(DPTNet, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.n_heads = n_heads
        self.ff_hid = ff_hid
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.n_src = n_src
        self.norm_type = norm_type
        self.ff_activation = ff_activation
        self.mask_act = mask_act
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.in_norm = norms.get(norm_type)(in_chan)

        # Succession of DPRNNBlocks.
        self.layers = nn.ModuleList([])
        for x in range(self.n_repeats):
            self.layers.append(nn.ModuleList([ImprovedTransformedLayer(self.in_chan, self.n_heads, self.ff_hid,
                                                                   self.dropout, self.ff_activation, True, self.norm_type),
                                          ImprovedTransformedLayer(self.in_chan, self.n_heads, self.ff_hid,
                                                                   self.dropout, self.ff_activation, self.bidirectional,
                                                                   self.norm_type)
                            ]))
        net_out_conv = nn.Conv2d(self.in_chan, n_src * self.in_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        # Gating and masking in 2D space (after fold)
        self.net_out = nn.Sequential(nn.Conv1d(self.in_chan, self.in_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(self.in_chan, self.in_chan, 1), nn.Sigmoid())


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
        #batch, n_filters, n_frames = mixture_w.size()
        mixture_w = self.in_norm(mixture_w)  # [batch, bn_chan, n_frames]

        ola = Oladd(self.chunk_size, self.hop_size)
        mixture_w = ola.unfold(mixture_w)
        batch, n_filters, self.chunk_size, n_chunks = mixture_w.size()

        for layer_idx in range(len(self.layers)):
            intra, inter = self.layers[layer_idx]
            mixture_w = ola.intra_process(mixture_w, intra)
            mixture_w = ola.inter_process(mixture_w, inter)

        output = self.first_out(mixture_w)
        output = output.reshape(batch * self.n_src, self.in_chan, self.chunk_size, n_chunks)
        output = ola.fold(output)

        output = self.net_out(output) * self.net_gate(output)
        # Compute mask
        output = output.reshape(batch, self.n_src, self.in_chan, -1)
        est_mask = self.output_act(output)
        return est_mask


    def get_config(self):
        config = {
            'in_chan': self.in_chan,
            'ff_hid': self.hid_size,
            'n_heads': self.n_heads,
            'chunk_size': self.chunk_size,
            'hop_size': self.hop_size,
            'n_repeats': self.n_repeats,
            'n_src': self.n_src,
            'norm_type': self.norm_type,
            'ff_activation': self.ff_activation,
            'mask_act': self.mask_act,
            'bidirectional': self.bidirectional,
            'dropout': self.dropout,
        }
        return config


if __name__ == "__main__":
    import torch
    a = torch.rand((2, 64, 16000))
    l = DPTNet(64, 2, n_repeats=1)

