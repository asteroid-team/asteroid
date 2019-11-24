from torch import nn

import asteroid.filterbanks as fb
from asteroid.masknn import TDConvNet
from asteroid.engine.optimizers import make_optimizer


class Model(nn.Module):
    def __init__(self, encoder, masker, decoder):
        super().__init__()
        self.encoder = encoder
        self.masker = masker
        self.decoder = decoder

    def forward(self, x):
        tf_rep = self.encoder(x)
        est_masks = self.masker(tf_rep)
        masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        return self.pad_output_to_inp(self.decoder(masked_tf_rep), x)

    @staticmethod
    def pad_output_to_inp(output, inp):
        """ Pad first argument to have same size as second argument"""
        inp_len = inp.size(-1)
        output_len = output.size(-1)
        return nn.functional.pad(output, [0, inp_len - output_len])


def make_model_and_optimizer(conf):
    """ This will be very practical to re-instantiate the model
    We save the conf in a yaml, json or pickle file and save the state_dict.
    For eval, load args, call this function and load state_dict.
    Super simple, no need for structure."""
    # Define building blocks for local model
    enc, dec = fb.make_enc_dec('free', **conf['filterbank'])
    masker = TDConvNet(in_chan=enc.filterbank.n_feats_out,
                       out_chan=enc.filterbank.n_feats_out,
                       **conf['masknet'])
    model = nn.DataParallel(Model(enc, masker, dec))
    # Define optimizer of this model
    optimizer = make_optimizer(model.parameters(), **conf['optim'])
    return model, optimizer
