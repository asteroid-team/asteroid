import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
from torch.testing import assert_close

from asteroid.data import LibriMix


SAMPLE_RATE = 8000
MIX_BOTH = [0.125, 0.25, 0.375]
MIX_CLEAN = [0.5, 0.625, 0.75]
SOURCES = [[0.03125, 0.15625], [0.0625, 0.1875], [0.09375, 0.21875]]


def _write_metadata(tmp_path, lengths):
    """Write aligned metadata and distinct, exactly representable audio per row."""
    audio_root = (tmp_path / "audio").resolve()
    audio_root.mkdir()
    for folder in ["mix_both", "mix_clean", "s1", "s2"]:
        (audio_root / folder).mkdir()
    rows = {"mix_both": [], "mix_clean": []}
    for index, length in enumerate(lengths):
        filename = f"1000-1-{index:04d}_2000-1-{index:04d}.wav"
        amplitudes = [MIX_BOTH[index], MIX_CLEAN[index], *SOURCES[index]]
        paths = {}
        for folder, amplitude in zip(["mix_both", "mix_clean", "s1", "s2"], amplitudes):
            path = audio_root / folder / filename
            sf.write(
                path, np.full(length, amplitude, dtype=np.float32), SAMPLE_RATE, subtype="FLOAT"
            )
            paths[folder] = str(path)
        for mixture in rows:
            rows[mixture].append(
                {
                    "mixture_path": paths[mixture],
                    "source_1_path": paths["s1"],
                    "source_2_path": paths["s2"],
                    "length": length,
                }
            )
    for mixture, entries in rows.items():
        pd.DataFrame(entries).to_csv(tmp_path / f"metadata_{mixture}.csv", index=False)


@pytest.mark.parametrize("task", ["enh_both", "sep_clean", "sep_noisy"])
@pytest.mark.parametrize("return_id", [False, True])
@pytest.mark.parametrize(
    "lengths,segment,expected_indices",
    [
        pytest.param([4000, 8000, 8000], 1.0, [1, 2], id="short-first"),
        pytest.param([8000, 4000, 8000], 1.0, [0, 2], id="short-middle"),
        pytest.param([8000, 8000, 4000], 1.0, [0, 1], id="short-last"),
        pytest.param([7999, 8000, 8001], 1.0, [1, 2], id="boundary"),
        pytest.param([8000, 12000, 16000], 1.0, [0, 1, 2], id="no-filtering"),
        pytest.param([4000, 8000, 8000], None, [0, 1, 2], id="full-utterances"),
        pytest.param([4000, 4000, 4000], 1.0, [], id="all-short"),
    ],
)
def test_librimix_filtered_audio(tmp_path, task, return_id, lengths, segment, expected_indices):
    _write_metadata(tmp_path, lengths)
    dataset = LibriMix(
        str(tmp_path), task=task, sample_rate=SAMPLE_RATE, segment=segment, return_id=return_id
    )

    assert len(dataset) == len(expected_indices)
    for position, input_index in enumerate(expected_indices):
        item = dataset[position]
        assert len(item) == (3 if return_id else 2)
        mixture, sources = item[:2]
        length = lengths[input_index] if segment is None else SAMPLE_RATE
        mixture_value = MIX_CLEAN[input_index] if task == "sep_clean" else MIX_BOTH[input_index]
        source_values = [MIX_CLEAN[input_index]] if task == "enh_both" else SOURCES[input_index]
        assert mixture.shape == (length,)
        assert sources.shape == (len(source_values), length)
        assert_close(mixture, torch.full((length,), mixture_value), rtol=0, atol=0)
        expected_sources = torch.tensor(source_values).unsqueeze(1).expand(-1, length)
        assert_close(sources, expected_sources, rtol=0, atol=0)
        if return_id:
            assert item[2] == [f"1000-1-{input_index:04d}", f"2000-1-{input_index:04d}"]


def test_librimix_enh_both_same_length_target(tmp_path):
    # The second retained row has the right shape even when its clean audio is wrong.
    _write_metadata(tmp_path, [4000, 8000, 8000])
    dataset = LibriMix(str(tmp_path), task="enh_both", sample_rate=SAMPLE_RATE, segment=1.0)
    mixture, sources = dataset[1]
    assert mixture.shape == (8000,)
    assert sources.shape == (1, 8000)
    assert_close(mixture, torch.full((8000,), MIX_BOTH[2]), rtol=0, atol=0)
    assert_close(sources, torch.full((1, 8000), MIX_CLEAN[2]), rtol=0, atol=0)
