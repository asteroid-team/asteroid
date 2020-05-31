import os
from asteroid.utils import hub_utils


def test_download():
    # We download
    path1 = hub_utils.cached_download('mpariente/ConvTasNet_WHAM!_sepclean')
    assert os.path.isfile(path1)
    # We use cache
    path2 = hub_utils.cached_download('mpariente/ConvTasNet_WHAM!_sepclean')
    assert path1 == path2
