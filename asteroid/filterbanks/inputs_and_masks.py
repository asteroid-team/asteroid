import warnings

warnings.warn("`inputs_and_masks` has been renamed `transforms` since asteroid "
              "v0.1.0 and will be removed in v0.2.0", FutureWarning)
from .transforms import *
