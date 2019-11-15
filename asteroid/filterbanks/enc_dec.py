"""
EncoderDecoder base class.
@author : Manuel Pariente, Inria-Nancy
"""

from torch.nn import functional as F

from ..engine.sub_module import SubModule


class EncoderDecoder(SubModule):
    """Base Encoder-Decoder class.
    Each subclass has to implement a `filters` property, and a
    `get_config` method.

    # Args
        n_filters: Positive int. Number of filters.
        kernel_size: Positive int. Length of the filters.
        stride: Positive int. Stide of the conv or transposed conv. (Hop size).
            If None (default), set to `kernel_size // 2`.
        enc_or_dec: String. `enc` or `dec`. Controls if filterbank is used as
            an encoder of a decoder. Based on `enc_or_dec`, the class defines
            the filterbank call `fb_call` which will be called in the `forward`
            so that `forward` can be overwritten in child classes.
    """
    def __init__(self, n_filters, kernel_size, stride=None,
                 enc_or_dec='encoder'):
        super(EncoderDecoder, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride if stride else self.kernel_size // 2
        self.enc_or_dec = enc_or_dec

        if enc_or_dec in ['enc', 'encoder']:
            self.fb_call = self.encode  # Filterbank call
        elif enc_or_dec in ['dec', 'decoder']:
            self.fb_call = self.decode  # Filterbank call
        else:
            raise ValueError('Expected `enc_or_dec` in [`enc`, `dec`], ' +
                             'received {}'.format(enc_or_dec))

    @property
    def filters(self):
        raise NotImplementedError

    def encode(self, waveform):
        return F.conv1d(waveform, self.filters, stride=self.stride)

    def decode(self, spec):
        if len(spec.shape) == 3:
            return F.conv_transpose1d(spec, self.filters, stride=self.stride)
        elif len(spec.shape) == 4:
            batch, n_src, chan, spec_len = spec.shape
            out = F.conv_transpose1d(spec.view(batch * n_src, chan, spec_len),
                                     self.filters, stride=self.stride)
            return out.view(batch, n_src, -1)

    def forward(self, *inputs):
        return self.fb_call(*inputs)

    def get_config(self):
        """ Returns dictionary of arguments to re-instantiate the class."""
        config = {
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'enc_or_dec': self.enc_or_dec
        }
        return config
