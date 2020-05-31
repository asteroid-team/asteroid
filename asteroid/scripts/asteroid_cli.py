import os
import yaml

import asteroid
from asteroid.models.publisher import upload_publishable


def upload():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('publish_dir', type=str,
                        help='Path to the publish dir.')
    parser.add_argument('--uploader', default=None, type=str,
                        help='Name of the uploader. Ex: `Manuel Pariente`')
    parser.add_argument('--affiliation', default=None, type=str,
                        help='Affiliation of the uploader. Ex `INRIA` ')
    parser.add_argument('--git_username', default=None, type=str,
                        help='Username in GitHub')
    parser.add_argument('--token', default=None, type=str,
                        help='Access token for Zenodo (or sandbox)')
    parser.add_argument('--force_publish', default=False, action='store_true',
                        help='Whether to  without asking confirmation')
    parser.add_argument('--use_sandbox', default=False, action='store_true',
                        help='Whether to use Zenodo sandbox.')
    args = parser.parse_args()
    args_as_dict = dict(vars(args))
    # Load uploader info if present
    info_file = os.path.join(asteroid.project_root, 'uploader_info.yml')
    if os.path.exists(info_file):
        uploader_info = yaml.safe_load(open(info_file, 'r'))
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
        force_upload: n
        ```
        """
        print('You can create a `uploader_infos.yml` file in `Asteroid` root'
              f'to stop passing your name, affiliation etc. to the CLI. '
              f'Here is an example {example}')
