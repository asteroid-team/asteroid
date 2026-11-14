import torch
import pytest
from asteroid.dsp.vad import ebased_vad


def test_ebased_vad():
    mag_spec = torch.abs(torch.randn(10, 2, 65, 16))  # Need positive inputs
    batch_src_mask = ebased_vad(mag_spec)

    assert isinstance(batch_src_mask, torch.BoolTensor)
    batch_1_mask = ebased_vad(mag_spec[:, 0])
    # Assert independence of VAD output
    assert (batch_src_mask[:, 0] == batch_1_mask).all()


@pytest.mark.parametrize("leading_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize("traced", [False, True])
def test_ebased_vad_noncontiguous(leading_shape, traced):
    # Native STFT magnitudes have noncontiguous frequency/time dimensions.
    waveform = torch.randn(6, 256, generator=torch.Generator().manual_seed(0))
    mag_spec = torch.stft(
        waveform, n_fft=32, window=torch.hann_window(32), return_complex=True
    ).abs()
    count = 1
    for size in leading_shape:
        count *= size
    mag_spec = mag_spec[:count].reshape(*leading_shape, *mag_spec.shape[-2:])
    assert not mag_spec.is_contiguous()
    vad = torch.jit.trace(ebased_vad, mag_spec.contiguous()) if traced else ebased_vad
    # Independent amplitude-domain reference, including each leading item.
    threshold = mag_spec.amax(dim=(-2, -1), keepdim=True) * 0.01
    expected = mag_spec > threshold
    assert torch.equal(vad(mag_spec), expected)
    assert torch.equal(vad(mag_spec.transpose(-1, -2)), expected.transpose(-1, -2))
