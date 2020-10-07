import pathlib

from .models import ConvTasNet, DCCRNet, DCUNet, DPRNNTasNet, DPTNet, LSTMTasNet, DeMask
from .utils import deprecation_utils, torch_utils  # noqa

project_root = str(pathlib.Path(__file__).expanduser().absolute().parent.parent)
__version__ = "0.3.4rc0"


def show_available_models():
    from .utils.hub_utils import MODELS_URLS_HASHTABLE

    print(" \n".join(list(MODELS_URLS_HASHTABLE.keys())))


__all__ = [
    "ConvTasNet",
    "DPRNNTasNet",
    "DPTNet",
    "LSTMTasNet",
    "DeMask",
    "DCUNet",
    "DCCRNet",
    "show_available_models",
]
