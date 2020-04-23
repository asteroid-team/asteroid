import torch
from torch import nn

import warnings

from numpy import VisibleDeprecationWarning
warnings.warn("`blocks` has been splited between `convolutional` and "
              "`recurrent` since asteroid v0.1.2 and will be removed "
              "in v0.2.0", VisibleDeprecationWarning)
from .convolutional import *
from .recurrent import *
# Waiting to finalize ChimeraPP's arch and comply coding style to move it.


class ChimeraPP(nn.Module):
    """ Chimera plus plus model used in deep clustering.

    Args:
        in_chan: input dimension to the network a.k.a number of frequency bins
        n_src: int > 0. Number of masks to estimate.
        rnn_type: string, select from `'RNN'`, `'LSTM'`, `'GRU'`. Can
            also be passed in lowercase letters.
        embedding_dim: int. Dimension of the embedding vector
        n_layers: int. Number of rnn layers
        hidden_size: int. Number of non-linear elements in the hidden layer
        dropout: float. In range [0-1]
        bidirectional: Boolen. Is your rnn bi-directional
    """
    def __init__(self, in_chan, n_src, rnn_type = 'lstm',
            embedding_dim=20, n_layers=2, hidden_size=600,
            dropout=0, bidirectional=True):
        super(ChimeraPP, self).__init__()
        self.input_dim = in_chan
        self.n_src = n_src
        self.embedding_dim = embedding_dim
        self.rnn = SingleRNN(rnn_type, in_chan, hidden_size, n_layers, \
            dropout, bidirectional)
        self.dropout = nn.Dropout(dropout)
        rnn_out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.embedding_layer = nn.Linear(rnn_out_dim, \
                in_chan * embedding_dim)
        self.mask_layer = nn.Linear(rnn_out_dim, in_chan * n_src)
        self.non_linearity = nn.Sigmoid()

    def forward(self, input_data):
        batches, freq_dim, seq_cnt = input_data.shape
        out = self.rnn(input_data.permute(0,2,1))
        out = self.dropout(out)
        projection = self.embedding_layer(out)
        projection = self.non_linearity(projection)
        projection = projection.view(batches, -1, self.embedding_dim)
        proj_norm = torch.norm(projection, p=2, dim=-1, keepdim=True) + \
                torch.finfo(torch.float32).eps
        projection_final =  projection/proj_norm
        mask_out = self.mask_layer(out)
        mask_out = mask_out.view(batches, self.n_src, self.input_dim, seq_cnt)
        return projection_final, mask_out
