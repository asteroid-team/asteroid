from .analytic_free_fb import AnalyticFreeFB
from .free_fb import FreeFB
from .param_sinc_fb import ParamSincFB
from .stft_fb import STFTFB
from .enc_dec import Encoder, Decoder, NoEncoder


def make_enc_dec(fb_name='free', mask_mode='reim', inp_mode='reim', **kwargs):
    """
    Creates congruent encoder and decoder from the same filterbank family.
    Args:
        fb_name: String or className. Filterbank family from which to make
            encoder and decoder. Among [`'free'`, `'analytic_free'`,
            `'param_sinc'`, `'stft'`] and many more to come.
        inp_mode: String. One of [`'reim'`, `'mag'`, `'cat'`]. Controls
            `post_processing_inputs` method which can be applied after
            the forward.
            - `'reim'` or `'real'` corresponds to the concatenation of
                real and imaginary parts. (filterbank seen as real-valued).
            - `'mag'` or `'mod'`corresponds to the magnitude (or modulus)of the
                complex filterbank output.
            - `'cat'` or `'concat'` corresponds to the concatenation of both
                previous representations.
        mask_mode: String. One of [`'reim'`, `'mag'`, `'comp'`]. Controls
            the way the time-frequency mask is applied to the input
            representation in the `apply_mask` method.
            - `'reim'` or `'real'` corresponds to a real-valued filterbank
                where both the input and the mask consists in the
                concatenation of real and imaginary parts.
            - `'mag'` or `'mod'`corresponds to a magnitude (or modulus) mask
                applied to the complex filterbank output.
            - `'comp'` or `'complex'` corresponds to a complex mask : the
                input and the mask are point-wise multiplied in the complex
                sense.
        **kwargs: Arguments which will be passed to the filterbank class.
            Usual argument include `n_filters`, `kernel_size` and `stride` but
            this will depend on the filterbank family.
    Returns:
        Encoder instance, Decoder instance.
    """
    fb = get(fb_name)(**kwargs)
    enc = Encoder(fb, mask_mode=mask_mode, inp_mode=inp_mode)
    # Filters between encoder and decoder should not be shared.
    fb = get(fb_name)(**kwargs)
    dec = Decoder(fb)
    return enc, dec


def get(identifier):
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
