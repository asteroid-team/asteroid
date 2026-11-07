"""FUSS file lists preserve paths and pad missing foreground sources."""

import numpy as np
import pytest
import soundfile as sf
import torch


@pytest.fixture
def fuss_dataset():
    from asteroid.data.fuss_dataset import FUSSDataset

    return FUSSDataset


def check_file_list(
    fuss_dataset, tmp_path, slots_by_row, return_bg, explicit_empty, missing_marker=""
):
    """Read real audio and verify every source slot without changing input files."""
    rows = []
    expected = []
    for row_index, slots in enumerate(slots_by_row):
        ramp = np.arange(32, dtype=np.float32) / 1024
        background = ramp + (row_index + 1) / 16
        sources = np.zeros((3, 32), dtype=np.float32)
        paths = [""] * 3
        for slot in slots:
            sources[slot] = ramp + (row_index + 1) / 8 + (slot + 1) / 32
            path = tmp_path / f"row{row_index}_fg{slot}.wav"
            sf.write(str(path), sources[slot], 16000, subtype="FLOAT")
            paths[slot] = str(path)
        mixture = background + sources.sum(axis=0)
        mix_path = tmp_path / f"row{row_index}_mix.wav"
        bg_path = tmp_path / f"row{row_index}_bg.wav"
        sf.write(str(mix_path), mixture, 16000, subtype="FLOAT")
        sf.write(str(bg_path), background, 16000, subtype="FLOAT")
        fields = [str(mix_path), str(bg_path)] + [path or missing_marker for path in paths]
        if not explicit_empty:
            while fields[-1] == "":
                fields.pop()
        rows.append("\t".join(fields))
        expected.append((mixture, sources, background, paths))
    file_list = tmp_path / "examples.tsv"
    file_list.write_text("\n".join(rows) + "\n", encoding="utf-8")
    original = {path: path.read_bytes() for path in tmp_path.iterdir()}

    dataset = fuss_dataset(str(file_list), return_bg=return_bg)
    assert len(dataset) == len(slots_by_row)
    assert dataset.sample_rate == 16000
    assert dataset.num_samples == 160000
    assert dataset.max_n_fg == dataset.n_src == 3
    assert dataset.fg_names == ["fg0", "fg1", "fg2"]
    infos = dataset.get_infos()
    assert infos["dataset"] == "FUSS"
    assert infos["task"] == "sep_noisy"
    assert len(infos["licenses"]) == 1
    assert infos["licenses"][0]["title"] == "Free Universal Sound Separation Dataset"
    assert infos["licenses"][0]["non_commercial"] is False

    for index, (mixture, sources, background, paths) in enumerate(expected):
        assert list(dataset.mix_df.iloc[index][dataset.fg_names]) == paths
        for _ in range(2):
            result = dataset[index]
            assert len(result) == (3 if return_bg else 2)
            assert all(isinstance(value, torch.Tensor) for value in result)
            assert all(value.dtype == torch.float32 for value in result)
            assert result[0].shape == (32,)
            assert result[1].shape == (3, 32)
            assert torch.equal(result[0], torch.from_numpy(mixture))
            assert torch.equal(result[1], torch.from_numpy(sources))
            if return_bg:
                assert result[2].shape == (32,)
                assert torch.equal(result[2], torch.from_numpy(background))
    assert set(tmp_path.iterdir()) == set(original)
    assert all(path.read_bytes() == contents for path, contents in original.items())


@pytest.mark.parametrize("n_fg", [0, 1, 2, 3])
@pytest.mark.parametrize("return_bg", [False, True])
@pytest.mark.parametrize("explicit_empty", [False, True])
def test_exclusive_foregrounds(fuss_dataset, tmp_path, n_fg, return_bg, explicit_empty):
    slots = tuple(range(n_fg))
    check_file_list(fuss_dataset, tmp_path, [slots, slots], return_bg, explicit_empty)


@pytest.mark.parametrize("counts", [(0, 1, 2, 3), (0, 1), (0, 2)])
@pytest.mark.parametrize("return_bg", [False, True])
def test_mixed_foregrounds(fuss_dataset, tmp_path, counts, return_bg):
    slots_by_row = [tuple(range(count)) for count in counts]
    check_file_list(fuss_dataset, tmp_path, slots_by_row, return_bg, False)


@pytest.mark.parametrize("return_bg", [False, True])
def test_missing_interior_slot(fuss_dataset, tmp_path, return_bg):
    check_file_list(fuss_dataset, tmp_path, [(0, 2), (2,)], return_bg, True)


@pytest.mark.parametrize("return_bg", [False, True])
def test_default_na_marker(fuss_dataset, tmp_path, return_bg):
    check_file_list(fuss_dataset, tmp_path, [(0, 2), (0, 1, 2)], return_bg, True, "NA")
