"""
Container for encoding/masking/decoding networks
@author : Manuel Pariente, Inria-Nancy
"""

import torch
from torch import nn
from .sub_module import NoLayer


class Container(nn.Module):
    """ Model container for encoder-masker-decoder architectures.
    Args:
        encoder: SubModule instance. The encoder of the network.
        masker: SubModule instance. The mask network.
        decoder: SubModule instance. The decoder of the network.

    If either of `encoder`, `masker` or `decoder` is None (default), they will
    be ignored.
    """
    def __init__(self, encoder=None, masker=None, decoder=None):
        super(Container, self).__init__()
        self.encoder = encoder if encoder is not None else NoLayer()
        self.masker = masker if masker is not None else NoLayer()
        self.decoder = decoder if decoder is not None else NoLayer()

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        tf_rep = self.encoder(x)

        # NoLayer doesn't have post_process_inputs so this will break if
        # encoder is None. Same for apply_mask in the next lines.
        # We could use defaults in self.__init__ or in NoLayer.
        # Try to think about better design for flexible forward without to
        # much code from the user.
        # And, importantly, we don't loose the serialize possibility..
        masker_input = self.encoder.post_process_inputs(tf_rep)
        # est_masks : [batch, n_scr, bins, time]
        est_masks = self.masker(masker_input)

        # The apply_mask method should also belong to the encoder as the
        # mask is applied to its output, the decoder's dimensions can be
        # deduced by it but the decoder doesnt dictate the masking mode.
        masked_tf_reps = self.encoder.apply_mask(tf_rep.unsqueeze(1),
                                                 est_masks, dim=2)
        output = self.decoder(masked_tf_reps)

        # This is desirable only for waveform models, or where the input output
        # are the same type of representation.
        output = self.pad_output_to_inp(output, x)
        return output

    def apply_mask(self, x, mask):
        # To be removed most probably.
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        return x * mask

    def pad_output_to_inp(self, output, inp):
        inp_len = inp.size(-1)
        output_len = output.size(-1)
        return nn.functional.pad(output, [0, inp_len - output_len])

    def serialize(self, optimizer=None, **kwargs):
        """ Serialization method for a Container.
        Args:
            optimizer: torch.optim.Optimizer instance.
            **kwargs:

        Returns:
            A dictionary containing all Container info.
        """
        pack = {'model': {
            'encoder': self.encoder.serialize(),
            'masker': self.masker.serialize(),
            'decoder': self.decoder.serialize()},
                'optimizer': {},
                'infos': kwargs}
        if optimizer is not None:
            pack['optimizer'] = {'args': optimizer.defaults,
                                 'state_dict': optimizer.state_dict()}
        return pack

    def load_encoder(self, pack):
        pack = self.get_subpack(pack, 'encoder')
        return self.load_pack(self.encoder, pack)

    def load_masker(self, pack):
        pack = self.get_subpack(pack, 'masker')
        return self.load_pack(self.masker, pack)

    def load_decoder(self, pack):
        pack = self.get_subpack(pack, 'decoder')
        return self.load_pack(self.decoder, pack)

    def load_model(self, pack):
        """ Load model from a package
        Intended usage : instantiate Container with empty objects
        model = Container(FreeFB, TDConvNet, FreeFB)
        model.load_model(pack)
        pack being the output of
        pack = previous_model.serialize()

        The pack contains the arguments to reinstantiate the encoder, masker
        and decoder classes and their state_dict

        """
        self.load_encoder(pack)
        self.load_masker(pack)
        self.load_decoder(pack)
        # If some checks are performed in self.__init__, the first instiation
        # didn't trigger them for sure, we might want to reinstantiate the
        # self with the right components
        # otherwise, making a class method would probably work better.
        # self = self.reinstantiate()  # Rerun the init

    def reinstantiate(self):
        """ Call the class on encoder, masker and decoder class instances"""
        return self.__class__(self.encoder, self.masker, self.decoder)

    @staticmethod
    def load_pack(obj, pack):
        """
        Loads config and state_dict in `obj` from `pack`
        Args:
            obj: SubModule instance. Needs a `from_pack method`.
            pack: A dictionary containing the keys `args` and `state_dict`
                to instantiate the `obj`
        Returns:
            An instance of `obj`.
        """
        if isinstance(obj, nn.Module):
            # The object has already been instantiated : load state_dict
            obj.load_state_dict(pack['state_dict'])
        else:
            # The class was passed to Container, instantiate and load.
            obj.load_from_pack(pack)
        return obj

    @staticmethod
    def get_subpack(pack, name):
        """ Get subpack from key `name` if it exists. """
        if name in pack.keys():
            return pack[name]
        else:
            return pack
