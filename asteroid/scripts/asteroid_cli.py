import os
import argparse
import torch
import yaml
import itertools
import glob
import warnings
from typing import List

import asteroid
from asteroid.separate import separate
from asteroid.dsp import LambdaOverlapAdd
from asteroid.models.publisher import upload_publishable
from asteroid.models.base_models import BaseModel


SUPPORTED_EXTENSIONS = [
    ".wav",
    ".flac",
    ".ogg",
]


def validate_window_length(n):
    try:
        n = int(n)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be integer")
    if n < 10:
        # Note: This doesn't allow for hop < 10.
        raise argparse.ArgumentTypeError("Must be given in samples, not seconds")
    return n


def upload():
    """CLI function to upload pretrained models."""
    parser = argparse.ArgumentParser()
    parser.add_argument("publish_dir", type=str, help="Path to the publish dir.")
    parser.add_argument(
        "--uploader", default=None, type=str, help="Name of the uploader. Ex: `Manuel Pariente`"
    )
    parser.add_argument(
        "--affiliation", default=None, type=str, help="Affiliation of the uploader. Ex `INRIA` "
    )
    parser.add_argument("--git_username", default=None, type=str, help="Username in GitHub")
    parser.add_argument(
        "--token", default=None, type=str, help="Access token for Zenodo (or sandbox)"
    )
    parser.add_argument(
        "--force_publish",
        default=False,
        action="store_true",
        help="Whether to  without asking confirmation",
    )
    parser.add_argument(
        "--use_sandbox", default=False, action="store_true", help="Whether to use Zenodo sandbox."
    )
    args = parser.parse_args()
    args_as_dict = dict(vars(args))
    # Load uploader info if present
    info_file = os.path.join(asteroid.project_root, "uploader_info.yml")
    if os.path.exists(info_file):
        uploader_info = yaml.safe_load(open(info_file, "r"))
        # Replace fields that where not specified (CLI dominates)
        for k, v in uploader_info.items():
            if args_as_dict[k] == parser.get_default(k):
                args_as_dict[k] = v

    upload_publishable(**args_as_dict)
    # Suggest creating uploader_infos.yml
    if not os.path.exists(info_file):
        example = """
        ```asteroid/uploader_infos.yml
        uploader: Manuel Pariente
        affiliation: Universite Lorraine, CNRS, Inria, LORIA, France
        git_username: mpariente
        token: XXX
        ```
        """
        print(
            "You can create a `uploader_infos.yml` file in `Asteroid` root"
            f"to stop passing your name, affiliation etc. to the CLI. "
            f"Here is an example {example}"
        )
        print(
            "Thanks a lot for sharing your model! Don't forget to create"
            "a model card in the repo! "
        )


def infer(argv=None):
    """CLI function to run pretrained model inference on wav files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("url_or_path", type=str, help="Path to the pretrained model.")
    parser.add_argument(
        "--files",
        default=None,
        required=True,
        type=str,
        help="Path to the wav files to separate. Also supports list of filenames, "
        "directory names and globs.",
        nargs="+",
    )
    parser.add_argument(
        "-f",
        "--force-overwrite",
        action="store_true",
        help="Whether to overwrite output wav files.",
    )
    parser.add_argument(
        "-r",
        "--resample",
        action="store_true",
        help="Whether to resample wrong sample rate input files.",
    )
    parser.add_argument(
        "-w",
        "--ola-window",
        type=validate_window_length,
        default=None,
        help="Overlap-add window to use. If not set (default), overlap-add is not used.",
    )
    parser.add_argument(
        "--ola-hop",
        type=validate_window_length,
        default=None,
        help="Overlap-add hop length in samples. Defaults to ola-window // 2. Only used if --ola-window is set.",
    )
    parser.add_argument(
        "--ola-window-type",
        type=str,
        default="hanning",
        help="Type of overlap-add window to use. Only used if --ola-window is set.",
    )
    parser.add_argument(
        "--ola-no-reorder",
        action="store_true",
        help="Disable automatic reordering of overlap-add chunk. See asteroid.dsp.LambdaOverlapAdd for details. "
        "Only used if --ola-window is set.",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None, type=str, help="Output directory to save files."
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="Device to run the model on, eg. 'cuda:0'."
        "Defaults to 'cuda' if CUDA is available, else 'cpu'.",
    )
    args = parser.parse_args(argv)

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model = BaseModel.from_pretrained(pretrained_model_conf_or_path=args.url_or_path)
    if args.ola_window is not None:
        model = LambdaOverlapAdd(
            model,
            n_src=None,
            window_size=args.ola_window,
            hop_size=args.ola_hop,
            window=args.ola_window_type,
            reorder_chunks=not args.ola_no_reorder,
        )
    model = model.to(device)

    file_list = _process_files_as_list(args.files)
    for f in file_list:
        separate(
            model,
            f,
            force_overwrite=args.force_overwrite,
            output_dir=args.output_dir,
            resample=args.resample,
        )


def register_sample_rate():
    """CLI to register sample rate to an Asteroid model saved without `sample_rate`,  before 0.4.0."""

    def _register_sample_rate(filename, sample_rate):
        import torch

        conf = torch.load(filename, map_location="cpu")
        conf["model_args"]["sample_rate"] = sample_rate
        torch.save(conf, filename)

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Model file to edit.")
    parser.add_argument("sample_rate", type=float, help="Sampling rate to add to the model.")
    args = parser.parse_args()

    _register_sample_rate(filename=args.filename, sample_rate=args.sample_rate)


def _process_files_as_list(files_str: List[str]) -> List[str]:
    """Support filename, folder name, and globs. Returns list of filenames."""
    all_files = []
    for f in files_str:
        # Existing file
        if os.path.isfile(f):
            all_files.append(f)
        # Glob folder and append.
        elif os.path.isdir(f):
            all_files.extend(glob_dir(f))
        else:
            local_list = glob.glob(f)
            if not local_list:
                warnings.warn(f"Could find any file that matched {f}", UserWarning)
            all_files.extend(local_list)
    return all_files


def glob_dir(d):
    """Return all filenames in directory that match the supported extensions."""
    return list(
        itertools.chain(
            *[
                glob.glob(os.path.join(d, "**/*" + ext), recursive=True)
                for ext in SUPPORTED_EXTENSIONS
            ]
        )
    )
