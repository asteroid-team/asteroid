import json
import os

import pytest

from asteroid.data.wham_dataset import WhamDataset
from asteroid.data.whamr_dataset import WhamRDataset
from asteroid.data.wsj0_mix import Wsj0mixDataset

SAMPLE_RATE = 8000
SEGMENT = 7200.0  # seconds
# Two utterances of one hour each are shorter than the segment and so are
# dropped, which is two hours in total.
DROPPED_HOURS = 2.0


def _make_json_dir(tmp_path, names):
    lengths = [3600 * SAMPLE_RATE, 3600 * SAMPLE_RATE, int(SEGMENT * SAMPLE_RATE)]
    entries = [["/does/not/exist/{}.wav".format(i), n] for i, n in enumerate(lengths)]
    for name in names:
        with open(os.path.join(str(tmp_path), name + ".json"), "w") as f:
            json.dump(entries, f)
    return str(tmp_path)


def _reported_hours(out):
    # "Drop 2 utts(2.00 h) from 3 (shorter than 57600000 samples)"
    return float(out.split("(")[1].split(" h)")[0])


@pytest.mark.parametrize(
    "klass, names, kwargs",
    [
        (WhamDataset, ["mix_both", "s1", "s2"], dict(task="sep_noisy")),
        (
            WhamRDataset,
            ["mix_both_reverb", "s1_anechoic", "s2_anechoic"],
            dict(task="sep_reverb_noisy"),
        ),
        (Wsj0mixDataset, ["mix", "s1", "s2"], dict(n_src=2)),
    ],
)
def test_dropped_duration_is_reported_in_hours(tmp_path, capsys, klass, names, kwargs):
    json_dir = _make_json_dir(tmp_path, names)
    klass(json_dir, sample_rate=SAMPLE_RATE, segment=SEGMENT, **kwargs)
    assert _reported_hours(capsys.readouterr().out) == pytest.approx(DROPPED_HOURS)
