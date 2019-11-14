from .analytic_free_fb import AnalyticFreeFB
from .free_fb import FreeFB
from .param_sinc_fb import ParamSincFB
from .stft_fb import STFTFB


def make_fb(filterbank='free', **kwargs):
    """ Instantiate filterbank class from name and config dictionary. """
    return get(filterbank)(**kwargs)


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
