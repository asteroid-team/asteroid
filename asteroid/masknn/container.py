"""
Container for encoding/masking/decoding networks
@author : Manuel Pariente, Inria-Nancy
"""

import torch
from torch import nn
from ..sub_module import NoLayer

"""
What we want here. 
If encoder is None, feed features directly.
If decoder is None, return the masked representation
Findout deep clustering case..
Handle lists of decoders for several heads
"""


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
        est_masks = self.masker(tf_rep)
        masked_tf_reps = self.apply_mask(tf_rep, est_masks)
        output = self.decoder(masked_tf_reps)
        return output

    def apply_mask(self, x, mask):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        return x * mask

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
        """ Load model from a package """
        self.load_encoder(pack)
        self.load_masker(pack)
        self.load_decoder(pack)

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
            obj.load_from_pack(obj, pack)
        return obj

    @staticmethod
    def get_subpack(pack, name):
        """ Get subpack from key `name` if it exists. """
        if name in pack.keys():
            return pack[name]
        else:
            return pack
