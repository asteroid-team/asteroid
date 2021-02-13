import json
import os
import torch
from torch import nn

from asteroid import torch_utils
from asteroid import torch_utils
from asteroid_filterbanks import Encoder, Decoder, FreeFB
from asteroid.masknn.recurrent import SingleRNN
from asteroid.engine.optimizers import make_optimizer
from asteroid.masknn.norms import GlobLN


class TasNet(nn.Module):
    """Some kind of TasNet, but not the original one
    Differences:
        - Overlap-add support (strided convolutions)
        - No frame-wise normalization on the wavs
        - GlobLN as bottleneck layer.
        - No skip connection.

    Args:
        fb_conf (dict): see local/conf.yml
        mask_conf (dict): see local/conf.yml
    """

    def __init__(self, fb_conf, mask_conf):
        super().__init__()
        self.n_src = mask_conf["n_src"]
        self.n_filters = fb_conf["n_filters"]
        # Create TasNet encoders and decoders (could use nn.Conv1D as well)
        self.encoder_sig = Encoder(FreeFB(**fb_conf))
        self.encoder_relu = Encoder(FreeFB(**fb_conf))
        self.decoder = Decoder(FreeFB(**fb_conf))
        self.bn_layer = GlobLN(fb_conf["n_filters"])

        # Create TasNet masker
        self.masker = nn.Sequential(
            SingleRNN(
                "lstm",
                fb_conf["n_filters"],
                hidden_size=mask_conf["n_units"],
                n_layers=mask_conf["n_layers"],
                bidirectional=True,
                dropout=mask_conf["dropout"],
            ),
            nn.Linear(2 * mask_conf["n_units"], self.n_src * self.n_filters),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encode(x)
        to_sep = self.bn_layer(tf_rep)
        est_masks = self.masker(to_sep.transpose(-1, -2)).transpose(-1, -2)
        est_masks = est_masks.view(batch_size, self.n_src, self.n_filters, -1)
        masked_tf_rep = tf_rep.unsqueeze(1) * est_masks
        return torch_utils.pad_x_to_y(self.decoder(masked_tf_rep), x)

    def encode(self, x):
        relu_out = torch.relu(self.encoder_relu(x))
        sig_out = torch.sigmoid(self.encoder_sig(x))
        return sig_out * relu_out


def make_model_and_optimizer(conf):
    """Function to define the model and optimizer for a config dictionary.
    Args:
        conf: Dictionary containing the output of hierachical argparse.
    Returns:
        model, optimizer.
    The main goal of this function is to make reloading for resuming
    and evaluation very simple.
    """
    model = TasNet(conf["filterbank"], conf["masknet"])
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
