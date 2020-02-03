"""
| Base classes for filterbanks, encoders and decoders.
| @author : Manuel Pariente, Inria-Nancy
"""

import torch
from torch import nn
from torch.nn import functional as F

from . inputs_and_masks import _inputs, _masks


class Filterbank(nn.Module):
    """Base Filterbank class.

    Each subclass has to implement a `filters` property.

    Args:
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the conv or transposed conv. (Hop size).
            If None (default), set to ``kernel_size // 2``.

    Attributes:
        n_filters (int)
        kernel_size (int)
        stride (int)
    """
    def __init__(self, n_filters, kernel_size, stride=None):
        super(Filterbank, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride if stride else self.kernel_size // 2
        # If not specified otherwise in the filterbank's init, output
        # number of features is equal to number of required filters.
        self.n_feats_out = n_filters

    @property
    def filters(self):
        """ Abstract method for filters. """
        raise NotImplementedError

    def get_config(self):
        """ Returns dictionary of arguments to re-instantiate the class. """
        config = {
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride
        }
        return config


class _EncDec(nn.Module):
    """ Base private class for Encoder and Decoder.

    Common parameters and methods.

    Args:
        filterbank (:class:`Filterbank`): Filterbank instance. The filterbank
            to use as an encoder or a decoder.
        is_pinv (bool): Whether to be the pseudo inverse of filterbank.

    Attributes:
        filterbank (:class:`Filterbank`)
        stride (int)
        is_pinv (bool)
    """
    def __init__(self, filterbank, is_pinv=False):
        super(_EncDec, self).__init__()
        self.filterbank = filterbank
        self.stride = self.filterbank.stride
        self.is_pinv = is_pinv

    @property
    def filters(self):
        return self.filterbank.filters

    def compute_filter_pinv(self, filters):
        """ Computes pseudo inverse filterbank of given filters."""
        scale = self.filterbank.stride / self.filterbank.kernel_size
        shape = filters.shape
        ifilt = torch.pinverse(filters.squeeze()).transpose(-1, -2).view(shape)
        # Compensate for the overlap-add.
        return ifilt * scale

    def get_filters(self):
        """ Returns filters or pinv filters depending on `is_pinv` attribute """
        if self.is_pinv:
            return self.compute_filter_pinv(self.filters)
        else:
            return self.filters

    def get_config(self):
        """ Returns dictionary of arguments to re-instantiate the class."""
        config = {'is_pinv': self.is_pinv}
        base_config = self.filterbank.get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Encoder(_EncDec):
    """ Encoder class.

    Add encoding methods to Filterbank classes.
    Not intended to be subclassed.

    Args:
        filterbank (:class:`Filterbank`): The filterbank to use
            as an encoder.
        is_pinv (bool): Whether to be the pseudo inverse of filterbank.
        inp_mode (str): One of [``'reim'``, ``'mag'``, ``'cat'``]. Controls
            `post_processing_inputs` method which can be applied after
            the forward.
                -  ``'reim'`` or ``'real'`` corresponds to the concatenation of
                   real and imaginary parts. (filterbank seen as real-valued).
                -  ``'mag'`` or ``'mod'`` corresponds to the magnitude (or
                   modulus) of the complex filterbank output.
                -  ``'cat'`` or ``'concat'`` corresponds to the concatenation of
                   both previous representations.
        mask_mode (str): One of [``'reim'``, ``'mag'``, ``'comp'``]. Controls
            the way the time-frequency mask is applied to the input
            representation in the :func:`apply_mask` method.
                -  ``'reim'`` or ``'real'`` corresponds to a real-valued
                   filterbank where both the input and the mask consists in the
                   concatenation of real and imaginary parts.
                -  ``'mag'`` or ``'mod'`` corresponds to a magnitude (or
                   modulus) mask applied to the complex filterbank output.
                -  ``'comp'`` or ``'complex'`` corresponds to a complex mask:
                   the input and the mask are point-wise multiplied in the
                   complex sense.

    Attributes:
        inp_mode (str)
        mask_mode (str)
        inp_func (callable)
        in_chan_mul
        mask_fun (callable)
        n_feats_out (int)
    """
    def __init__(self, filterbank, inp_mode='reim', mask_mode='reim',
                 is_pinv=False):
        super(Encoder, self).__init__(filterbank, is_pinv=is_pinv)
        self.inp_mode = inp_mode
        self.mask_mode = mask_mode

        self.inp_func, self.in_chan_mul = _inputs[self.inp_mode]
        self.mask_func, self.out_chan_mul = _masks[self.mask_mode]
        self.n_feats_out = self.filterbank.n_feats_out

    @classmethod
    def pinv_of(cls, filterbank, **kwargs):
        """ Returns an Encoder, pseudo inverse of a filterbank or Decoder."""
        if isinstance(filterbank, Filterbank):
            return cls(filterbank, is_pinv=True, **kwargs)
        elif isinstance(filterbank, Decoder):
            return cls(filterbank.filterbank, is_pinv=True, **kwargs)

    def forward(self, waveform):
        """ Convolve 1D torch.Tensor with filterbank."""
        filters = self.get_filters()
        return F.conv1d(waveform, filters, stride=self.stride)

    def post_process_inputs(self, x):
        """ Computes real or complex representation from `forward` output."""
        return self.inp_func(x)

    def apply_mask(self, tf_rep, mask, dim=1):
        """ Applies real of complex masks `forward` output. """
        return self.mask_func(tf_rep, mask, dim=dim)

    def get_config(self):
        """ Returns dictionary of arguments to re-instantiate the class."""
        config = {
            'inp_mode': self.inp_mode,
            'mask_mode': self.mask_mode
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Decoder(_EncDec):
    """Decoder class.
    
    Add decoding methods to Filterbank classes.
    Not intended to be subclassed.

    Args:
        filterbank (:class:`Filterbank`): The filterbank to use as an decoder.
        is_pinv (bool): Whether to be the pseudo inverse of filterbank.
    """
    @classmethod
    def pinv_of(cls, filterbank):
        """ Returns an Decoder, pseudo inverse of a filterbank or Encoder."""
        if isinstance(filterbank, Filterbank):
            return cls(filterbank, is_pinv=True)
        elif isinstance(filterbank, Encoder):
            return cls(filterbank.filterbank, is_pinv=True)

    def forward(self, spec):
        """ Applies transposed convolution to a TF representation.

        This is equivalent to overlap-add.

        Args:
            spec (:class:`torch.Tensor`): 3D or 4D Tensor. The TF
                representation. (Output of :func:`Encoder.forward`).
        Returns:
            :class:`torch.Tensor`: The corresponding time domain signal.
        """
        filters = self.get_filters()

        if len(spec.shape) == 3:
            return F.conv_transpose1d(spec, filters, stride=self.stride)
        elif len(spec.shape) == 4:
            batch, n_src, chan, spec_len = spec.shape
            out = F.conv_transpose1d(spec.view(batch * n_src, chan, spec_len),
                                     filters, stride=self.stride)
            return out.view(batch, n_src, -1)


class NoEncoder(nn.Module):  # pragma: no cover
    """ Class to use for no neural encoder.
    
    This is a placeholder for precomputed features.
    The features can be complex with real and imaginary parts concatenated
    into the same axis. The same post processing and masking strategies are
    available as for the :class:`Encoder` class.

    Args:
        inp_mode (str): One of [``'reim'``, ``'mag'``, ``'cat'``]. Controls
            `post_processing_inputs` method which can be applied after
            the forward.
                -  ``'reim'`` or ``'real'`` corresponds to the concatenation of
                   real and imaginary parts. (filterbank seen as real-valued).
                -  ``'mag'`` or ``'mod'`` corresponds to the magnitude (or
                   modulus) of the complex filterbank output.
                -  ``'cat'`` or ``'concat'`` corresponds to the concatenation
                   of both previous representations.
        mask_mode (str): One of [``'reim'``, ``'mag'``, ``'comp'``]. Controls
            the way the time-frequency mask is applied to the input
            representation in the :func:`apply_mask` method.
                -  ``'reim'`` or ``'real'`` corresponds to a real-valued
                   filterbank where both the input and the mask consists in the
                   concatenation of real and imaginary parts.
                -  ``'mag'`` or ``'mod'`` corresponds to a magnitude (or
                   modulus) mask applied to the complex filterbank output.
                -  ``'comp'`` or ``'complex'`` corresponds to a complex mask:
                   the input and the mask are point-wise multiplied in the
                   complex sense.

    By default:
        - The forward returns the input.
        - The input post-processing is the identity.
        - The mask is applied to the input features.
    """
    def __init__(self, inp_mode='reim', mask_mode='reim'):

        super(NoEncoder, self).__init__()
        self.inp_mode = inp_mode
        self.mask_mode = mask_mode

        self.inp_func, self.in_chan_mul = _inputs[self.inp_mode]
        self.mask_func, self.out_chan_mul = _masks[self.mask_mode]

    def forward(self, features):
        return features

    def post_process_inputs(self, x):
        return self.inp_func(x)

    def apply_mask(self, tf_rep, mask, dim=1):
        return self.mask_func(tf_rep, mask, dim=dim)

    def get_config(self):
        """ Returns dictionary of arguments to re-instantiate the class."""
        config = {
            'inp_mode': self.inp_mode,
            'mask_mode': self.mask_mode
        }
        return config
