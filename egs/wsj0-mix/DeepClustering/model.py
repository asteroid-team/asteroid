import torch as th
from torch import nn
from ipdb import set_trace

import asteroid.filterbanks as fb
from asteroid.masknn import TDConvNet
from asteroid.engine.optimizers import make_optimizer
from asteroid.filterbanks.inputs_and_masks import take_mag

class Model(nn.Module):
    def __init__(self, encoder, masker):
        super().__init__()
        self.encoder = encoder
        self.masker = masker

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = take_mag(self.encoder(x))
        embedding = self.masker(tf_rep)
        return embedding 


class SingleRNN(nn.Module):
    """ Module for a RNN block.

    taken from github.com/yluo42/TAC/blob/master/utility/models.py
    Licensed under CC BY-NC-SA 3.0 US.

    Args:
        rnn_type: string, select from `'RNN'`, `'LSTM'`, `'GRU'`. Can
            also be passed in lowercase letters.
        input_size: int, dimension of the input feature. The input should have
            shape (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        n_layers: int > 0. Number of layers used in RNN. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional.
            Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, n_layers=1,
                 dropout=0, bidirectional=False):
        super(SingleRNN, self).__init__()
        rnn_type = rnn_type.upper()
        assert rnn_type.upper() in ["RNN", "LSTM", "GRU"]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size,
                                         num_layers=n_layers,
                                         dropout=dropout,
                                         batch_first=True,
                                         bidirectional=bool(bidirectional))
        self.rnn_type = rnn_type

    def forward(self, inp):
        """ Input shape [batch, seq, feats] """
        output = inp
        rnn_output, _ = self.rnn(output)
        return rnn_output


class ChimeraPP(nn.Module):
    """ Chimera plus plus model used in deep clustering.

    Args:
        rnn_type: string, select from `'RNN'`, `'LSTM'`, `'GRU'`. Can
            also be passed in lowercase letters.
        n_src: int > 0. Number of masks to estimate.
        input_size: input dimension to the network a.k.a number of frequency bins
        embedding_dim: int. Dimension of the embedding vector
        n_layers: int. Number of rnn layers
        hidden_size: int. Number of non-linear elements in the hidden layer
        dropout: float. In range [0-1]
        bidirectional: Boolen. Is your rnn bi-directional
    """
    def __init__(self, rnn_type, n_src, input_size, \
            embedding_dim=20, n_layers=2, hidden_size=600, \
            dropout=0, bidirectional=True):
        super(ChimeraPP, self).__init__()
        self.input_dim = input_size
        self.n_src = n_src
        self.embedding_dim = embedding_dim
        self.rnn = SingleRNN(rnn_type, input_size, hidden_size, n_layers, \
            dropout, bidirectional)
        self.dropout = nn.Dropout(dropout)
        rnn_out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.embedding_layer = nn.Linear(rnn_out_dim, \
                input_size * embedding_dim)
        self.mask_layer = nn.Linear(rnn_out_dim, input_size * n_src)
        self.non_linearity = nn.Sigmoid()

    def forward(self, input_data):
        batches, freq_dim, seq_cnt = input_data.shape
        out = self.rnn(input_data.permute(0,2,1))
        out = self.dropout(out)
        projection = self.embedding_layer(out)
        projection = self.non_linearity(projection)
        projection = projection.view(batches, -1, self.embedding_dim)
        proj_norm = th.norm(projection, p=2, dim=-1, keepdim=True) + \
                th.finfo(th.float32).eps
        projection_final =  projection/proj_norm
        mask_out = self.mask_layer(out)
        mask_out = mask_out.view(batches, self.n_src, self.input_dim, seq_cnt)
        return projection_final, mask_out

def make_model_and_optimizer(conf):
    """ Function to define the model and optimizer for a config dictionary.
    Args:
        conf: Dictionary containing the output of hierachical argparse.
    Returns:
        model, optimizer.
    The main goal of this function is to make reloading for resuming
    and evaluation very simple.
    """
    enc, _ = fb.make_enc_dec('stft', **conf['filterbank'])
    masker = ChimeraPP( 'lstm', 2 , int(enc.filterbank.n_feats_out/2), \
            embedding_dim=20, n_layers=2, hidden_size=600, \
            dropout=0, bidirectional=True)
    model = Model(enc, masker)
    optimizer = make_optimizer(model.parameters(), **conf['optim'])
    return model, optimizer

