from .analytic_free_fb import AnalyticFreeFB
from .free_fb import FreeFB
from .param_sinc_fb import ParamSincFB
from .stft_fb import STFTFB
from .enc_dec import Filterbank, Encoder, Decoder, NoEncoder


def make_enc_dec(fb_name, mask_mode='reim', inp_mode='reim',
                 who_is_pinv=None, **kwargs):
    """ Creates congruent encoder and decoder from the same filterbank family.

    Args:
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``]. Can also be a class defined in a
            submodule in this subpackade (e.g. :class:`~.FreeFB`).
        mask_mode (str, optional): One of [``'reim'``, ``'mag'``, ``'comp'``].
            Controls the way the time-frequency mask is applied to the input
            representation in the `apply_mask` method.
                -  ``'reim'`` or ``'real'`` corresponds to a real-valued
                   filterbank where both the input and the mask consists in the
                   concatenation of real and imaginary parts.
                -  ``'mag'`` or ``'mod'`` corresponds to a magnitude (or
                   modulus) mask applied to the complex filterbank output.
                -  ``'comp'`` or ``'complex'`` corresponds to a complex mask:
                   the input and the mask are point-wise multiplied in the
                   complex sense.
        inp_mode (str, optional): One of [``'reim'``, ``'mag'``, ``'cat'``].
            Controls `post_processing_inputs` method which can be applied after
            the forward.
                -  ``'reim'`` or ``'real'`` corresponds to the concatenation
                   of real and imaginary parts. (filterbank seen as
                   real-valued).
                -  ``'mag'`` or ``'mod'`` corresponds to the magnitude (or
                   modulus)
                   of the complex filterbank output.
                -  ``'cat'`` or ``'concat'`` corresponds to the concatenation
                   of both previous representations.
        who_is_pinv (str, optional): If `None`, no pseudo-inverse filters will
            be used. If string (among [``'encoder'``, ``'decoder'``]), decides
            which of ``Encoder`` or ``Decoder`` will be the pseudo inverse of
            the other one.
        **kwargs: Arguments which will be passed to the filterbank class.
            Usual argument include `n_filters`, `kernel_size` and `stride`.
            Depends on the filterbank family.
    Returns:
        :class:`.Encoder`, :class:`.Decoder`
    """
    fb_class = get(fb_name)

    if who_is_pinv in ['dec', 'decoder']:
        fb = fb_class(**kwargs)
        enc = Encoder(fb, mask_mode=mask_mode, inp_mode=inp_mode)
        # Decoder filterbank is pseudo inverse of encoder filterbank.
        dec = Decoder.pinv_of(fb)
    elif who_is_pinv in ['enc', 'encoder']:
        fb = fb_class(**kwargs)
        dec = Decoder(fb)
        # Encoder filterbank is pseudo inverse of decoder filterbank.
        enc = Encoder.pinv_of(fb, mask_mode=mask_mode, inp_mode=inp_mode)
    else:
        fb = fb_class(**kwargs)
        enc = Encoder(fb, mask_mode=mask_mode, inp_mode=inp_mode)
        # Filters between encoder and decoder should not be shared.
        fb = fb_class(**kwargs)
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

# For the docs
__all__ = ['Filterbank', 'Encoder', 'Decoder', 'FreeFB', 'STFTFB',
           'AnalyticFreeFB', 'ParamSincFB']
