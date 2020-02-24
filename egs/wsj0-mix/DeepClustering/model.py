import torch as th
from torch import nn

import asteroid.filterbanks as fb
from asteroid.masknn import TDConvNet
from asteroid.engine.optimizers import make_optimizer
from asteroid.filterbanks.inputs_and_masks import take_mag
from asteroid.masknn.blocks import ChimeraPP 

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


def make_model_and_optimizer(conf):
    """ Function to define the model and optimizer for a config dictionary.
    Args:
        conf: Dictionary containing the output of hierachical argparse.
    Returns:
        model, optimizer.
    The main goal of this function is to make reloading for resuming
    and evaluation very simple.
    """
    enc = fb.Encoder(fb.STFTFB(**conf['filterbank']))
    masker = ChimeraPP(int(enc.filterbank.n_feats_out/2), 2,
                       embedding_dim=20, n_layers=2, hidden_size=600, \
                       dropout=0, bidirectional=True)
    model = Model(enc, masker)
    optimizer = make_optimizer(model.parameters(), **conf['optim'])
    return model, optimizer

