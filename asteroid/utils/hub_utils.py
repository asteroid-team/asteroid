import os
from functools import lru_cache
from hashlib import sha256
from typing import Union, Dict, List

import requests
from torch import hub
import huggingface_hub


CACHE_DIR = os.getenv(
    "ASTEROID_CACHE",
    os.path.expanduser("~/.cache/torch/asteroid"),
)
MODELS_URLS_HASHTABLE = {
    "mpariente/ConvTasNet_WHAM!_sepclean": "https://zenodo.org/record/3862942/files/model.pth?download=1",
    "mpariente/DPRNNTasNet_WHAM!_sepclean": "https://zenodo.org/record/3873670/files/model.pth?download=1",
    "mpariente/DPRNNTasNet(ks=16)_WHAM!_sepclean": "https://zenodo.org/record/3903795/files/model.pth?download=1",
    "Cosentino/ConvTasNet_LibriMix_sep_clean": "https://zenodo.org/record/3873572/files/model.pth?download=1",
    "Cosentino/ConvTasNet_LibriMix_sep_noisy": "https://zenodo.org/record/3874420/files/model.pth?download=1",
    "brijmohan/ConvTasNet_Libri1Mix_enhsingle": "https://zenodo.org/record/3970768/files/model.pth?download=1",
    "groadabike/ConvTasNet_DAMP-VSEP_enhboth": "https://zenodo.org/record/3994193/files/model.pth?download=1",
    "popcornell/DeMask_Surgical_mask_speech_enhancement_v1": "https://zenodo.org/record/3997047/files/model.pth?download=1",
    "popcornell/DPRNNTasNet_WHAM_enhancesingle": "https://zenodo.org/record/3998647/files/model.pth?download=1",
    "tmirzaev-dotcom/ConvTasNet_Libri3Mix_sepnoisy": "https://zenodo.org/record/4020529/files/model.pth?download=1",
    "mhu-coder/ConvTasNet_Libri1Mix_enhsingle": "https://zenodo.org/record/4301955/files/model.pth?download=1",
    "r-sawata/XUMX_MUSDB18_music_separation": "https://zenodo.org/record/4704231/files/pretrained_xumx.pth?download=1",
    "r-sawata/XUMXL_MUSDB18_music_separation": "https://zenodo.org/record/7128659/files/pretrained_xumxl.pth?download=1",
}

SR_HASHTABLE = {k: 8000.0 if not "DeMask" in k else 16000.0 for k in MODELS_URLS_HASHTABLE}


def cached_download(filename_or_url):
    """Download from URL and cache the result in ASTEROID_CACHE.

    Args:
        filename_or_url (str): Name of a model as named on the Zenodo Community
            page (ex: ``"mpariente/ConvTasNet_WHAM!_sepclean"``), or model id from
            the Hugging Face model hub (ex: ``"julien-c/DPRNNTasNet-ks16_WHAM_sepclean"``),
            or a URL to a model file (ex: ``"https://zenodo.org/.../model.pth"``), or a filename
            that exists locally (ex: ``"local/tmp_model.pth"``)

    Returns:
        str, normalized path to the downloaded (or not) model
    """
    from .. import __version__ as asteroid_version  # Avoid circular imports

    if os.path.isfile(filename_or_url):
        return filename_or_url

    if filename_or_url.startswith(huggingface_hub.HUGGINGFACE_CO_URL_HOME):
        filename_or_url = filename_or_url[len(huggingface_hub.HUGGINGFACE_CO_URL_HOME) :]

    if filename_or_url.startswith(("http://", "https://")):
        url = filename_or_url
    elif filename_or_url in MODELS_URLS_HASHTABLE:
        url = MODELS_URLS_HASHTABLE[filename_or_url]
    else:
        # Finally, let's try to find it on Hugging Face model hub
        # e.g. julien-c/DPRNNTasNet-ks16_WHAM_sepclean is a valid model id
        # and  julien-c/DPRNNTasNet-ks16_WHAM_sepclean@main supports specifying a commit/branch/tag.
        if "@" in filename_or_url:
            model_id = filename_or_url.split("@")[0]
            revision = filename_or_url.split("@")[1]
        else:
            model_id = filename_or_url
            revision = None
        return huggingface_hub.hf_hub_download(
            repo_id=model_id,
            filename=huggingface_hub.PYTORCH_WEIGHTS_NAME,
            cache_dir=get_cache_dir(),
            revision=revision,
            library_name="asteroid",
            library_version=asteroid_version,
        )

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
    """Consistently convert ``url`` into a filename."""
    _bytes = url.encode("utf-8")
    _hash = sha256(_bytes)
    filename = _hash.hexdigest()
    return filename


def get_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


@lru_cache()
def model_list(
    endpoint=huggingface_hub.HUGGINGFACE_CO_URL_HOME, name_only=False
) -> Union[str, List[Dict]]:
    """Get the public list of all the models on huggingface with an 'asteroid' tag."""
    path = "{}api/models?full=true&filter=asteroid".format(endpoint)
    r = requests.get(path)
    r.raise_for_status()
    all_models = r.json()
    if name_only:
        return [x["modelId"] for x in all_models]
    return all_models
