import os
import yaml
import itertools
import glob
import warnings
from typing import List

import asteroid
from asteroid.models.publisher import upload_publishable
from asteroid.models.base_models import BaseModel


SUPPORTED_EXTENSIONS = [
    ".wav",
    ".flac",
    ".ogg",
]


def upload():
    """ CLI function to upload pretrained models.

    Args:
        publish_dir (str): Path to the publishing directory.
            Usually under exp/exp_name/publish_dir
        uploader (str): Full name of the uploader (Ex: Manuel Pariente)
        affiliation (str, optional): Affiliation (no accent).
        git_username (str, optional): GitHub username.
        token (str): Access token generated to upload depositions.
        force_publish (bool): Whether to directly publish without
            asking confirmation before. Defaults to False.
        use_sandbox (bool): Whether to use Zenodo's sandbox instead of
            the official Zenodo.

    """
    import argparse

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


def infer():
    """ CLI function to run pretrained model inference on wav files.

    Args:
        url_or_path(str): Path to the pretrained model.
        files (List(str)): Path to the wav files to separate. Also support list
            of filenames, directory names and globs.
        force_overwrite (bool): Whether to overwrite output wav files.
        output_dir (str): Output directory to save files.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("url_or_path", type=str, help="Path to the pretrained model.")
    parser.add_argument(
        "--files",
        default=None,
        type=str,
        help="Path to the wav files to separate. Also support list of filenames, "
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
        "-o", "--output-dir", default=None, type=str, help="Output directory to save files."
    )
    args = parser.parse_args()

    model = BaseModel.from_pretrained(pretrained_model_conf_or_path=args.url_or_path)
    file_list = _process_files_as_list(args.files)

    for f in file_list:
        model.separate(f, force_overwrite=args.force_overwrite, output_dir=args.output_dir)


def _process_files_as_list(files_str: List) -> List:
    """ Support filename, folder name, and globs. Returns list of filenames."""
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
    """ Return all filenames in directory that match the supported extensions."""
    return list(
        itertools.chain(
            *[
                glob.glob(os.path.join(d, "**/*" + ext), recursive=True)
                for ext in SUPPORTED_EXTENSIONS
            ]
        )
    )
