import warnings
from numpy import VisibleDeprecationWarning
warnings.warn("`blocks` has been splited between `convolutional` and "
              "`recurrent` since asteroid v0.2.0 and will be removed "
              "in v0.3.0", VisibleDeprecationWarning)
from .convolutional import *
from .recurrent import *
