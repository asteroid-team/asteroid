import os
import torch as th
from torch import nn
from asteroid import torch_utils

import asteroid.filterbanks as fb
from asteroid.masknn import TDConvNet
from asteroid.engine.optimizers import make_optimizer
from asteroid.filterbanks.transforms import take_mag
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
                       dropout=0.5, bidirectional=True, \
                       log=conf['filterbank']['log'])
    model = Model(enc, masker)
    optimizer = make_optimizer(model.parameters(), **conf['optim'])
    return model, optimizer

def load_best_model(train_conf, best_model_path):
    """ Load best model after training.

    Args:
        train_conf (dict): dictionary as expected by `make_model_and_optimizer`
        exp_dir(str): Experiment directory. Expects to find
            `'best_k_models.json'` there.

    Returns:
        nn.Module the best pretrained model according to the val_loss.
    """
    # Create the model from recipe-local function
    model, _ = make_model_and_optimizer(train_conf)
    # Last best model summary
    checkpoint = th.load(best_model_path, map_location='cpu')
    # Removing additional saved info 
    checkpoint['state_dict'].pop('enc.filterbank._filters')
    # Load state_dict into model.
    model = torch_utils.load_state_dict_in(checkpoint['state_dict'],
                                           model)
    model.eval()
    return model
