from torch import nn

import asteroid_filterbanks as fb
from asteroid.masknn import TDConvNet
from asteroid.engine.optimizers import make_optimizer


class Model(nn.Module):
    def __init__(self, encoder, masker, decoder):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder
        self.inp_mode = encoder.inp_mode
        self.mask_mode = encoder.mask_mode

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        # Encode the waveform
        tf_rep = self.encoder(x)
        # Post process TF representation (take magnitude or keep [Re, Im] etc)
        # FIXME, that's removed
        # masker_input = self.encoder.post_process_inputs(tf_rep)
        masker_input = tf_rep
        # Estimate masks (Size [batch, n_scr, bins, time])
        est_masks = self.masker(masker_input)
        # Apply mask to TF representation
        masked_tf_reps = self.encoder.apply_mask(tf_rep.unsqueeze(1), est_masks, dim=2)
        # Map back TF representation to time domain
        return self.pad_output_to_inp(self.decoder(masked_tf_reps), x)

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
    # The encoder and decoder can directly be made from the dictionary.
    encoder, decoder = fb.make_enc_dec(**conf["filterbank"])

    # The input post-processing changes the dimensions of input features to
    # the mask network. Different type of masks impose different output
    # dimensions to the mask network's output. We correct for these here.
    nn_in = int(encoder.n_feats_out * encoder.in_chan_mul)
    nn_out = int(encoder.n_feats_out * encoder.out_chan_mul)
    masker = TDConvNet(in_chan=nn_in, out_chan=nn_out, **conf["masknet"])
    # Another possibility is to correct for these effects inside of Model,
    # but then instantiation of masker should also be done inside.
    model = Model(encoder, masker, decoder)

    # The model is defined in Container, which is passed to DataParallel.

    # Define optimizer : can be instantiate from dictonary as well.
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    return model, optimizer
