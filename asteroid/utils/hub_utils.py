import os
from torch import hub
from hashlib import sha256


CACHE_DIR = os.getenv("ASTEROID_CACHE", os.path.expanduser("~/.cache/torch/asteroid"),)
MODELS_URLS_HASHTABLE = {
    "mpariente/ConvTasNet_WHAM!_sepclean": "https://zenodo.org/record/3862942/files/model.pth?download=1",
    "mpariente/DPRNNTasNet_WHAM!_sepclean": "https://zenodo.org/record/3873670/files/model.pth?download=1",
    "mpariente/DPRNNTasNet(ks=16)_WHAM!_sepclean": "https://zenodo.org/record/3903795/files/model.pth?download=1",
    "Cosentino/ConvTasNet_LibriMix_sep_clean": "https://zenodo.org/record/3873572/files/model.pth?download=1",
    "Cosentino/ConvTasNet_LibriMix_sep_noisy": "https://zenodo.org/record/3874420/files/model.pth?download=1",
    "brijmohan/ConvTasNet_Libri1Mix_enhsingle": "https://zenodo.org/record/3970768/files/model.pth?download=1",
}


def cached_download(filename_or_url):
    """ Download from URL with torch.hub and cache the result in ASTEROID_CACHE.

    Args:
        filename_or_url (str): Name of a model as named on the Zenodo Community
            page (ex: mpariente/ConvTasNet_WHAM!_sepclean), or an URL to a model
            file (ex: https://zenodo.org/.../model.pth), or a filename
            that exists locally (ex: local/tmp_model.pth)

    Returns:
        str, normalized path to the downloaded (or not) model
    """
    if os.path.isfile(filename_or_url):
        return filename_or_url

    if filename_or_url in MODELS_URLS_HASHTABLE:
        url = MODELS_URLS_HASHTABLE[filename_or_url]
    else:
        # Give a chance to direct URL, torch.hub will handle exceptions
        url = filename_or_url
    cached_filename = url_to_filename(url)
    cached_dir = os.path.join(get_cache_dir(), cached_filename)
    cached_path = os.path.join(cached_dir, "model.pth")

    os.makedirs(cached_dir, exist_ok=True)
    if not os.path.isfile(cached_path):
        hub.download_url_to_file(url, cached_path)
        return cached_path
    # It was already downloaded
    print(f"Using cached model `{filename_or_url}`")
    return cached_path


def url_to_filename(url):
    """ Consistently convert `url` into a filename. """
    _bytes = url.encode("utf-8")
    _hash = sha256(_bytes)
    filename = _hash.hexdigest()
    return filename


def get_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR
