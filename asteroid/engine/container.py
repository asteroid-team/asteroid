"""
Container for encoding/masking/decoding networks
@author : Manuel Pariente, Inria-Nancy
"""

import torch
from torch import nn
from .sub_module import NoLayer
from ..filterbanks import NoEncoder


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
        self.encoder = encoder if encoder is not None else NoEncoder()
        self.masker = masker if masker is not None else NoLayer()
        self.decoder = decoder if decoder is not None else NoLayer()

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        # Encode the waveform
        tf_rep = self.encoder(x)
        # Post process TF representation (take magnitude or keep [Re, Im] etc)
        masker_input = self.encoder.post_process_inputs(tf_rep)
        # Estimate masks (Size [batch, n_scr, bins, time])
        est_masks = self.masker(masker_input)
        # Apply mask to TF representation
        masked_tf_reps = self.encoder.apply_mask(tf_rep.unsqueeze(1),
                                                 est_masks, dim=2)
        # Map back TF representation to time domain
        output = self.decoder(masked_tf_reps)
        # Pad back the waveform to the input length
        output = self.pad_output_to_inp(output, x)
        return output

    def pad_output_to_inp(self, output, inp):
        """ Pad first argument to have same size as second argument"""
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
        # If some checks are performed in self.__init__, the first instance
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
