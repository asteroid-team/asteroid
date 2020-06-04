from .analytic_free_fb import AnalyticFreeFB
from .free_fb import FreeFB
from .param_sinc_fb import ParamSincFB
from .stft_fb import STFTFB
from .enc_dec import Filterbank, Encoder, Decoder
from .griffin_lim import griffin_lim, misi
from .multiphase_gammatone_fb import MultiphaseGammatoneFB


def make_enc_dec(fb_name, n_filters, kernel_size, stride=None,
                 who_is_pinv=None, **kwargs):
    """ Creates congruent encoder and decoder from the same filterbank family.

    Args:
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``]. Can also be a class defined in a
            submodule in this subpackade (e.g. :class:`~.FreeFB`).
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        who_is_pinv (str, optional): If `None`, no pseudo-inverse filters will
            be used. If string (among [``'encoder'``, ``'decoder'``]), decides
            which of ``Encoder`` or ``Decoder`` will be the pseudo inverse of
            the other one.
        **kwargs: Arguments which will be passed to the filterbank class
            additionally to the usual `n_filters`, `kernel_size` and `stride`.
            Depends on the filterbank family.
    Returns:
        :class:`.Encoder`, :class:`.Decoder`
    """
    fb_class = get(fb_name)

    if who_is_pinv in ['dec', 'decoder']:
        fb = fb_class(n_filters, kernel_size, stride=stride, **kwargs)
        enc = Encoder(fb)
        # Decoder filterbank is pseudo inverse of encoder filterbank.
        dec = Decoder.pinv_of(fb)
    elif who_is_pinv in ['enc', 'encoder']:
        fb = fb_class(n_filters, kernel_size, stride=stride, **kwargs)
        dec = Decoder(fb)
        # Encoder filterbank is pseudo inverse of decoder filterbank.
        enc = Encoder.pinv_of(fb)
    else:
        fb = fb_class(n_filters, kernel_size, stride=stride, **kwargs)
        enc = Encoder(fb)
        # Filters between encoder and decoder should not be shared.
        fb = fb_class(n_filters, kernel_size, stride=stride, **kwargs)
        dec = Decoder(fb)
    return enc, dec


def get(identifier):
    """ Returns a filterbank class from a string. Returns its input if it
    is callable (already a :class:`.Filterbank` for example).

    Args:
        identifier (str or Callable or None): the filterbank identifier.

    Returns:
        :class:`.Filterbank` or None
    """
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)
        if cls is None:
            raise ValueError('Could not interpret filterbank identifier: ' +
                             str(identifier))
        return cls
    else:
        raise ValueError('Could not interpret filterbank identifier: ' +
                         str(identifier))


# Aliases.
free = FreeFB
analytic_free = AnalyticFreeFB
param_sinc = ParamSincFB
stft = STFTFB
multiphase_gammatone = mpgtf = MultiphaseGammatoneFB

# For the docs
__all__ = ['Filterbank', 'Encoder', 'Decoder', 'FreeFB', 'STFTFB',
           'AnalyticFreeFB', 'ParamSincFB', 'MultiphaseGammatoneFB']
