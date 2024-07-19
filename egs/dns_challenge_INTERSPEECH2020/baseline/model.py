import json
import os

import torch
from torch import nn

from asteroid.engine.system import System
from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import magreim, mag
from asteroid_filterbanks.transforms import apply_real_mask
from asteroid_filterbanks.transforms import apply_mag_mask
from asteroid.masknn.recurrent import StackedResidualRNN
from asteroid.engine.optimizers import make_optimizer
from asteroid import torch_utils


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
    stft, istft = make_enc_dec("stft", **conf["filterbank"])
    # Because we concatenate (re, im, mag) as input and compute a complex mask.
    if conf["main_args"]["is_complex"]:
        inp_size = int(stft.n_feats_out * 3 / 2)
        output_size = stft.n_feats_out
    else:
        inp_size = output_size = int(stft.n_feats_out / 2)
    # Add these fields to the mask model dict
    conf["masknet"].update(dict(input_size=inp_size, output_size=output_size))
    masker = SimpleModel(**conf["masknet"])
    # Make the complete model
    model = Model(stft, masker, istft, is_complex=conf["main_args"]["is_complex"])
    # Define optimizer of this model
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    return model, optimizer


class Model(nn.Module):
    """Speech enhancement model.

    Args:
        encoder (~.Encoder): instance of a complex filterbank encoder
            `Encoder(STFTBFB(**))`.
        masker (nn.Module): Mask estimator network.
        decoder (~.Decoder): instance of a complex filterbank decoder
            `Decoder(STFTBFB(**))`.
        is_complex (bool): If the network works on the complex domain.

    If `is_complex` is `True`, the input to the network are complex features,
    the network estimates a complex mask and returns a complex speech estimate.
    Else, the input is the magnitude, the network estimates a magnitude mask
    and the returns a **complex** speech estimate.
    The loss function needs to be adapted to complex representations.
    """

    def __init__(self, encoder, masker, decoder, is_complex=True):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        # Decoder is not used for training but eventually, we want to invert
        # the encoder. Might as well include it in the model.
        self.decoder = decoder
        self.is_complex = is_complex

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        # Compute STFT
        tf_rep = self.encoder(x)
        # Estimate TF mask from STFT features : cat([re, im, mag])
        if self.is_complex:
            to_masker = magreim(tf_rep)
        else:
            to_masker = mag(tf_rep)
        # LSTM masker expects a feature dimension last (not like 1D conv)
        est_masks = self.masker(to_masker.transpose(1, 2)).transpose(1, 2)
        # Apply TF mask
        if self.is_complex:
            masked_tf_rep = apply_real_mask(tf_rep, est_masks)
        else:
            masked_tf_rep = apply_mag_mask(tf_rep, est_masks)
        return masked_tf_rep

    def denoise(self, x):
        estimate_stft = self(x)
        wav = self.decoder(estimate_stft)
        return torch_utils.pad_x_to_y(wav, x)


class SimpleModel(nn.Module):
    """Simple recurrent model for the DNS challenge.

    Args:
        input_size (int): input size along the features dimension
        hidden_size (int): hidden size in the recurrent net
        output_size (int): output size, defaults to `:attr:` input_size
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can also
            be passed in lowercase letters.
        n_layers (int): Number of recurrent layers.
        dropout (float): dropout value between recurrent layers.
    """

    def __init__(
        self, input_size, hidden_size, output_size=None, rnn_type="gru", n_layers=3, dropout=0.3
    ):
        super(SimpleModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        output_size = input_size if output_size is None else output_size
        self.output_size = output_size
        self.in_proj_layer = nn.Linear(input_size, hidden_size)
        self.residual_rec = StackedResidualRNN(
            rnn_type, hidden_size, n_layers=n_layers, dropout=dropout
        )
        self.out_proj_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Mask estimator's forward pass. Expects [batch, time, input_size]"""
        # Non negative features from input
        out_rec = self.residual_rec(torch.relu(self.in_proj_layer(x)))
        # Activation is relu on the mask (better gradients allegedly)
        return torch.relu(self.out_proj_layer(out_rec))


class SimpleSystem(System):
    def common_step(self, batch, batch_nb):
        mixture, speech, noise = batch
        estimate = self(mixture.unsqueeze(1))
        speech_stft = self.model.encoder(speech.unsqueeze(1))
        # The loss function can be something like
        # loss_func = partial(distance, is_complex=some_bool)
        loss = self.loss_func(estimate, speech_stft)
        return loss


def distance(estimate, target, is_complex=True):
    """Compute the average distance in the complex plane. Makes more sense
    when the network computes a complex mask.

    Args:
        estimate (torch.Tensor): Estimate complex spectrogram.
        target (torch.Tensor): Speech target complex spectrogram.
        is_complex (bool): Whether to compute the distance in the complex or
            the magnitude space.

    Returns:
        torch.Tensor the loss value, in a tensor of size 1.
    """
    if is_complex:
        # Take the difference in the complex plane and compute the squared norm
        # of the remaining vector.
        return mag(estimate - target).pow(2).mean()
    else:
        # Compute the mean difference between magnitudes.
        return (mag(estimate) - mag(target)).pow(2).mean()


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
