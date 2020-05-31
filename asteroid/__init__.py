import pathlib
from .utils import deprecation_utils, torch_utils
from .models import ConvTasNet, DPRNNTasNet

project_root = str(pathlib.Path(__file__).expanduser().absolute().parent.parent)
__version__ = '0.2.1'
