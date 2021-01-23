from asteroid_filterbanks.analytic_free_fb import AnalyticFreeFB
from asteroid_filterbanks.free_fb import FreeFB
from asteroid_filterbanks.param_sinc_fb import ParamSincFB
from asteroid_filterbanks.stft_fb import STFTFB
from asteroid_filterbanks.enc_dec import Filterbank, Encoder, Decoder
from asteroid_filterbanks.griffin_lim import griffin_lim, misi
from asteroid_filterbanks.multiphase_gammatone_fb import MultiphaseGammatoneFB
from asteroid_filterbanks.melgram_fb import MelGramFB
from asteroid_filterbanks import (
    make_enc_dec,
    register_filterbank,
    get,
    free,
    analytic_free,
    param_sinc,
    stft,
    multiphase_gammatone,
    mpgtf,
)

import warnings
from ..utils.deprecation_utils import VisibleDeprecationWarning

warnings.warn(
    "asteroid.filterbanks has been replaced by `asteroid_filterbanks` and will be totally "
    "remove in a future release. Please use `asteroid_filterbanks` instead.",
    VisibleDeprecationWarning,
)
