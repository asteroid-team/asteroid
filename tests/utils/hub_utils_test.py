import os
from asteroid.utils import hub_utils


HF_EXAMPLE_MODEL_IDENTIFER = "julien-c/DPRNNTasNet-ks16_WHAM_sepclean"


def test_download():
    # We download
    path1 = hub_utils.cached_download("mpariente/ConvTasNet_WHAM!_sepclean")
    assert os.path.isfile(path1)
    # We use cache
    path2 = hub_utils.cached_download("mpariente/ConvTasNet_WHAM!_sepclean")
    assert path1 == path2


def test_hf_download():
    # We download
    path1 = hub_utils.cached_download(HF_EXAMPLE_MODEL_IDENTIFER)
    assert os.path.isfile(path1)
    # We use cache
    path2 = hub_utils.cached_download(HF_EXAMPLE_MODEL_IDENTIFER)
    assert path1 == path2
