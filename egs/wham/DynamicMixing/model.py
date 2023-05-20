import json
import os
import torch
from torch import nn

import asteroid_filterbanks as fb
from asteroid import torch_utils
from asteroid.masknn import DPRNN
from asteroid.engine.optimizers import make_optimizer


class Model(nn.Module):
    def __init__(self, encoder, masker, decoder):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)
        est_masks = self.masker(tf_rep)
        masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        return self.pad_output_to_inp(self.decoder(masked_tf_rep), x)

    @staticmethod
    def pad_output_to_inp(output, inp):
        """Pad first argument to have same size as second argument"""
        inp_len = inp.size(-1)
        output_len = output.size(-1)
        return nn.functional.pad(output, [0, inp_len - output_len])


def make_model_and_optimizer(conf):
    """Function to define the model and optimizer for a config dictionary.
    Args:
        conf: Dictionary containing the output of hierachical argparse.
    Returns:
        model, optimizer.
    The main goal of this function is to make reloading for resuming
    and evaluation very simple.
    """
    # Define building blocks for local model
    enc, dec = fb.make_enc_dec("free", **conf["filterbank"])
    masker = DPRNN(**conf["masknet"])
    model = Model(enc, masker, dec)
    # Define optimizer of this model
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    return model, optimizer


def load_best_model(train_conf, exp_dir):
    """Load best model after training.

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
    with open(os.path.join(exp_dir, "best_k_models.json"), "r") as f:
        best_k = json.load(f)
    best_model_path = min(best_k, key=best_k.get)
    # Load checkpoint
    checkpoint = torch.load(best_model_path, map_location="cpu")
    # Load state_dict into model.
    model = torch_utils.load_state_dict_in(checkpoint["state_dict"], model)
    model.eval()
    return model
