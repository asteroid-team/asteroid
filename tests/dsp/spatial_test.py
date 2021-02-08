import torch
import pytest
import numpy as np
from asteroid.dsp.spatial import xcorr


@pytest.mark.parametrize("seq_len_input", [1390])
@pytest.mark.parametrize("seq_len_ref", [1390, 1290])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_mics_input", [1])
@pytest.mark.parametrize("n_mics_ref", [1, 2])
@pytest.mark.parametrize("normalized", [False, True])
def test_xcorr(seq_len_input, seq_len_ref, batch_size, n_mics_input, n_mics_ref, normalized):
    target = torch.rand((batch_size, n_mics_input, seq_len_input))
    ref = torch.rand((batch_size, n_mics_ref, seq_len_ref))
    result = xcorr(target, ref, normalized)
    assert result.shape[-1] == (seq_len_input - seq_len_ref) + 1

    if normalized == False:
        for b in range(batch_size):
            for m in range(n_mics_input):
                npy_result = np.correlate(target[b, m].numpy(), ref[b, m].numpy())
                np.testing.assert_array_almost_equal(
                    result[b, m, : len(npy_result)].numpy(), npy_result, decimal=2
                )
