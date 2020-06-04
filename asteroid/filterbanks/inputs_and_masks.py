import warnings

from numpy import VisibleDeprecationWarning
warnings.warn("`inputs_and_masks` has been renamed `transforms` since asteroid "
              "v0.1.0 and will be removed in v0.2.0", VisibleDeprecationWarning)
from .transforms import *
