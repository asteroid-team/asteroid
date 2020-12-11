import fnmatch
import io
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from functools import partial, lru_cache
from hashlib import sha256
from typing import BinaryIO, Dict, Optional, Union, List
from urllib.parse import urlparse

import requests
import torch
from filelock import FileLock
from torch import hub


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
}

SR_HASHTABLE = {k: 8000.0 if not "DeMask" in k else 16000.0 for k in MODELS_URLS_HASHTABLE}

HF_WEIGHTS_NAME = "pytorch_model.bin"
HUGGINGFACE_CO_PREFIX = "https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
ENDPOINT = "https://huggingface.co"


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
    if os.path.isfile(filename_or_url):
        return filename_or_url

    if urlparse(filename_or_url).scheme in ("http", "https"):
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
        url = hf_bucket_url(model_id=model_id, filename=HF_WEIGHTS_NAME, revision=revision)
        return hf_get_from_cache(url, cache_dir=get_cache_dir())
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


def hf_bucket_url(
    model_id: str, filename: str, subfolder: Optional[str] = None, revision: Optional[str] = None
) -> str:
    """
    Resolve a model identifier, a file name, and an optional revision id, to a huggingface.co-hosted url, redirecting
    to Cloudfront (a Content Delivery Network, or CDN) for large files.

    Cloudfront is replicated over the globe so downloads are way faster for the end user (and it also lowers our
    bandwidth costs).

    Cloudfront aggressively caches files by default (default TTL is 24 hours), however this is not an issue here
    because we migrated to a git-based versioning system on huggingface.co, so we now store the files on S3/Cloudfront
    in a content-addressable way (i.e., the file name is its hash). Using content-addressable filenames means cache
    can't ever be stale.

    In terms of client-side caching from this library, we base our caching on the objects' ETag. An object' ETag is:
    its sha1 if stored in git, or its sha256 if stored in git-lfs.
    """
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    if revision is None:
        revision = "main"
    return HUGGINGFACE_CO_PREFIX.format(model_id=model_id, revision=revision, filename=filename)


def hf_url_to_filename(url: str, etag: Optional[str] = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way. If `etag` is specified, append its hash to the url's,
    delimited by a period.
    """
    url_bytes = url.encode("utf-8")
    filename = sha256(url_bytes).hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        filename += "." + sha256(etag_bytes).hexdigest()

    return filename


def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    from .. import __version__ as asteroid_version  # Avoid circular imports

    ua = "asteroid/{}; python/{}".format(asteroid_version, sys.version.split()[0])
    ua += "; torch/{}".format(torch.__version__)
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join("{}/{}".format(k, v) for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def http_get(
    url: str,
    temp_file: BinaryIO,
    proxies=None,
    resume_size=0,
    user_agent: Union[Dict, str, None] = None,
):
    """
    Donwload remote file. Do not gobble up errors.
    """
    headers = {"user-agent": http_user_agent(user_agent)}
    if resume_size > 0:
        headers["Range"] = "bytes=%d-" % (resume_size,)
    r = requests.get(url, stream=True, proxies=proxies, headers=headers)
    r.raise_for_status()
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            temp_file.write(chunk)


def hf_get_from_cache(
    url: str,
    cache_dir: str,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    local_files_only=False,
) -> Optional[str]:  # pragma: no cover
    """
    Given a URL, look for the corresponding file in the local cache. If it's not there, download it. Then return the
    path to the cached file.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """

    os.makedirs(cache_dir, exist_ok=True)

    url_to_download = url
    etag = None
    if not local_files_only:
        try:
            headers = {"user-agent": http_user_agent(user_agent)}
            r = requests.head(
                url, headers=headers, allow_redirects=False, proxies=proxies, timeout=etag_timeout
            )
            r.raise_for_status()
            etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
            # We favor a custom header indicating the etag of the linked resource, and
            # we fallback to the regular etag header.
            # If we don't have any of those, raise an error.
            if etag is None:
                raise OSError(
                    "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
                )
            # In case of a redirect,
            # save an extra redirect on the request.get call,
            # and ensure we download the exact atomic version even if it changed
            # between the HEAD and the GET (unlikely, but hey).
            if 300 <= r.status_code <= 399:
                url_to_download = r.headers["Location"]
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # etag is already None
            pass

    filename = hf_url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # etag is None == we don't have a connection or we passed local_files_only.
    # try to get the last downloaded one
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [
                file
                for file in fnmatch.filter(os.listdir(cache_dir), filename.split(".")[0] + ".*")
                if not file.endswith(".json") and not file.endswith(".lock")
            ]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            else:
                # If files cannot be found and local_files_only=True,
                # the models might've been found if local_files_only=False
                # Notify the user about that
                if local_files_only:
                    raise ValueError(
                        "Cannot find the requested files in the cached path and outgoing traffic has been"
                        " disabled. To enable model look-ups and downloads online, set 'local_files_only'"
                        " to False."
                    )
                else:
                    raise ValueError(
                        "Connection error, and we cannot find the requested files in the cached path."
                        " Please try again or make sure your Internet connection is on."
                    )

    # From now on, etag is not None.
    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):

        # If the download just completed while the lock was activated.
        if os.path.exists(cache_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return cache_path

        if resume_download:
            incomplete_path = cache_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager() -> "io.BufferedWriter":
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(
                tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False
            )
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            http_get(
                url_to_download,
                temp_file,
                proxies=proxies,
                resume_size=resume_size,
                user_agent=user_agent,
            )

        os.replace(temp_file.name, cache_path)

        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path


@lru_cache()
def model_list(endpoint=ENDPOINT, name_only=False) -> List[Dict]:
    """Get the public list of all the models on huggingface with an 'asteroid' tag."""
    path = "{}/api/models?full=true&filter=asteroid".format(endpoint)
    r = requests.get(path)
    r.raise_for_status()
    all_models = r.json()
    if name_only:
        return [x["modelId"] for x in all_models]
    return all_models
