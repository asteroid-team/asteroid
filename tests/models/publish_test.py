import os
import json
import shutil
import pytest

from asteroid.models import save_publishable, upload_publishable, ConvTasNet
from asteroid.data.wham_dataset import WhamDataset


def populate_wham_dir(path):
    wham_files = ["s1", "s2", "noise", "mix_single", "mix_clean", "mix_both"]
    os.makedirs(path, exist_ok=True)
    for source in wham_files:
        json_file = os.path.join(path, source + ".json")
        with open(os.path.join(json_file), "w") as f:
            json.dump(dict(), f)


@pytest.mark.skipif(os.getenv("ACCESS_TOKEN", None) is None, reason="Require private key")
def test_upload():
    # Make dirs
    os.makedirs("tmp/publish_dir", exist_ok=True)
    populate_wham_dir("tmp/wham")

    # Dataset and NN
    train_set = WhamDataset("tmp/wham", task="sep_clean")
    model = ConvTasNet(
        n_src=2, n_repeats=2, n_blocks=2, bn_chan=16, hid_chan=4, skip_chan=8, n_filters=32
    )

    # Save publishable
    model_conf = model.serialize()
    model_conf.update(train_set.get_infos())
    save_publishable("tmp/publish_dir", model_conf, metrics={}, train_conf={})

    # Upload
    token = os.getenv("ACCESS_TOKEN")
    if token:  # ACESS_TOKEN is not available on forks.
        zen, current = upload_publishable(
            "tmp/publish_dir",
            uploader="Manuel Pariente",
            affiliation="INRIA",
            use_sandbox=True,
            unit_test=True,  # Remove this argument and monkeypatch `input()`
            git_username="mpariente",
        )

        # Assert metadata is correct
        meta = current.json()["metadata"]
        assert meta["creators"][0]["name"] == "Manuel Pariente"
        assert meta["creators"][0]["affiliation"] == "INRIA"
        assert "asteroid-models" in [d["identifier"] for d in meta["communities"]]

        # Clean up
        zen.remove_deposition(current.json()["id"])
        shutil.rmtree("tmp/wham")


if __name__ == "__main__":
    test_upload()
