"""
Base class for all save/load-able modules.
@author : Manuel Pariente, Inria-Nancy
"""
from torch import nn


class SubModule(nn.Module):
    """Base class for all save/load-able modules."""
    def get_config(self):
        """ Returns dictionary of arguments to re-instantiate the class."""
        raise NotImplementedError

    def serialize(self):
        return {'args': self.get_config(),
                'state_dict': self.state_dict()}

    @classmethod
    def from_config(cls, config):
        """Create class instance from argument dictionary.
        Args:
            config: a dictionary containing the arguments to instantiate
                the class.
        Returns:
            A class instance.
        """
        return cls(**config)

    @classmethod
    def load_from_pack(cls, pack):
        """ Instantiate the class and load the state_dict.
        Args:
            pack: a dictionary containing the arguments to instantiate
                the class and the state_dict to be loaded.
        Returns:
            A class instance with loaded state_dict.
        """
        model = cls.from_config(pack['args'])
        model.load_state_dict(pack['state_dict'])
        return model


class NoLayer(nn.Module):
    """Class for linear activation layer.
    Can be used instead of an encoder to feed features directly to a mask
    network.
        Args:
        input_feats: int. Number of input features to the layer. Optional.
    """

    def __init__(self, input_feats=None):
        super(NoLayer, self).__init__()
        self.input_shape = input_feats
        self.n_feats_out = input_feats

    def forward(self, x):
        return x

    def get_config(self):
        return {'input_feats': self.input_feats}

    def serialize(self):
        return {'args': self.get_config()}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def load_from_pack(cls, pack):
        return cls.from_config(pack['args'])

