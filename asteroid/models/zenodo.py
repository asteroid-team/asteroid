import os
import json
import requests
from io import BufferedReader, BytesIO


class Zenodo(object):
    """Faciliate Zenodo's REST API.

    Args:
        api_key (str): Access token generated to upload depositions.
        use_sandbox (bool): Whether to use the sandbox (default: True)
            Note that `api_key` are different in sandbox.

    All methods return the requests response.

    .. note:: A Zenodo record is something that is public and cannot be deleted.
        A Zenodo deposit has not yet been published, is private and can be
        deleted.
    """

    def __init__(self, api_key=None, use_sandbox=True):
        if api_key is None:
            api_key = os.getenv("ACCESS_TOKEN", None)
        if api_key is None:
            raise ValueError(
                "Need to set `api_key` somehow. Either through the class"
                "arguments or by setting ACCESS_TOKEN env variable in bash."
            )
        self.use_sandbox = use_sandbox
        if use_sandbox is True:
            self.zenodo_address = "https://sandbox.zenodo.org"
        else:
            self.zenodo_address = "https://zenodo.org"

        self.api_key = api_key
        self.auth_header = {"Authorization": f"Bearer {self.api_key}"}
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def create_new_deposition(self, metadata=None):
        """Creates a new deposition.

        Args:
            metadata (dict, optional): Metadata dict to upload on the new
                deposition.
        """
        r = requests.post(
            f"{self.zenodo_address}/api/deposit/depositions", json={}, headers=self.headers
        )

        if r.status_code != 201:
            print("Creation failed (status code: {})".format(r.status_code))
            return r

        if metadata is not None and isinstance(metadata, dict):
            return self.change_metadata_in_deposition(r.json()["id"], metadata)
        else:
            print(f"Could not interpret metadata type ({type(metadata)}), " "expected dict")
        return r

    def change_metadata_in_deposition(self, dep_id, metadata):
        """Set or replace metadata in given deposition

        Args:
            dep_id (int): deposition id. You cna get it with
                `r = create_new_deposition(); dep_id = r.json()['id']`
            metadata (dict): Metadata dict.

        Examples
            >>> metadata = {
            ...     'title': 'My first upload',
            ...     'upload_type': 'poster',
            ...     'description': 'This is my first upload',
            ...     'creators': [{'name': 'Doe, John',
            ...                   'affiliation': 'Zenodo'}]
            ... }
        """
        data = {"metadata": metadata}
        r = requests.put(
            f"{self.zenodo_address}/api/deposit/depositions/{dep_id}",
            data=json.dumps(data),
            headers=self.headers,
        )
        return r

    def upload_new_file_to_deposition(self, dep_id, file, name=None):
        """Upload one file to existing deposition.

        Args:
            dep_id (int): deposition id. You cna get it with
                `r = create_new_deposition(); dep_id = r.json()['id']`
            file (str or io.BufferedReader): path to a file, or already opened
                file (path prefered).
            name (str, optional): name given to the uploaded file.
                Defaults to the path.

        (More: https://developers.zenodo.org/#deposition-files)
        """
        if isinstance(file, BufferedReader):
            files = {"file": file}
            filename = name if name else "Unknown"
        elif isinstance(file, str):
            if os.path.isfile(file):
                # This is a file, read it
                files = {"file": open(os.path.expanduser(file), "rb")}
                filename = name if name else os.path.basename(file)
            else:
                # This is a string, convert to BytesIO
                files = {"file": BytesIO(bytes(file, "utf-8"))}
                filename = name if name else "Unknown"
        else:
            raise ValueError("Unknown file format , expected str or Bytes ")
        data = {"name": filename}
        print("Submitting Data: {} and Files: {}".format(data, files))

        r = requests.post(
            f"{self.zenodo_address}/api/deposit/depositions/{dep_id}/files",
            headers=self.auth_header,
            data=data,
            files=files,
        )
        print("Zenodo received : {}".format(r.content))
        return r

    def publish_deposition(self, dep_id):  # pragma: no cover (Cannot publish)
        """Publish given deposition (Cannot be deleted)!

        Args:
            dep_id (int): deposition id. You cna get it with
                `r = create_new_deposition(); dep_id = r.json()['id']`
        """
        r = requests.post(
            f"{self.zenodo_address}/api/deposit/depositions/{dep_id}/actions/publish",
            headers=self.headers,
        )
        return r

    def get_deposition(self, dep_id=-1):
        """ Get deposition by deposition id. Get all dep_id is -1 (default)."""
        if dep_id > -1:
            print(f"Get deposition {dep_id} from Zenodo")
            r = requests.get(
                f"{self.zenodo_address}/api/deposit/depositions/{dep_id}", headers=self.headers
            )
        else:
            print("Get all depositions from Zenodo")
            r = requests.get(f"{self.zenodo_address}/api/deposit/depositions", headers=self.headers)
        print("Get Depositions: Status Code: {}".format(r.status_code))
        return r

    def remove_deposition(self, dep_id):
        """ Remove deposition with deposition id `dep_id`"""
        print(f"Delete deposition number {dep_id}")
        r = requests.delete(
            f"{self.zenodo_address}/api/deposit/depositions/{dep_id}", headers=self.auth_header
        )
        return r

    def remove_all_depositions(self):
        """ Removes all unpublished deposition (not records)."""
        all_depositions = self.get_deposition()
        for dep in all_depositions.json():
            self.remove_deposition(dep["id"])
