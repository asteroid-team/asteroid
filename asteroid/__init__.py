import pathlib
from .utils import deprecation_utils, torch_utils  # noqa
from .models import ConvTasNet, DPRNNTasNet, DPTNet, LSTMTasNet

project_root = str(pathlib.Path(__file__).expanduser().absolute().parent.parent)
__version__ = "0.3.1"


def show_available_models():
    from .utils.hub_utils import MODELS_URLS_HASHTABLE

    print(" \n".join(list(MODELS_URLS_HASHTABLE.keys())))


__all__ = [
    "ConvTasNet",
    "DPRNNTasNet",
    "DPTNet",
    "show_available_models",
]
