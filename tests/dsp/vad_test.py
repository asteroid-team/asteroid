import torch
from asteroid.dsp.vad import ebased_vad


def test_ebased_vad():
    mag_spec = torch.abs(torch.randn(10, 2, 65, 16))  # Need positive inputs
    batch_src_mask = ebased_vad(mag_spec)

    assert isinstance(batch_src_mask, torch.BoolTensor)
    batch_1_mask = ebased_vad(mag_spec[:, 0])
    # Assert independence of VAD output
    assert (batch_src_mask[:, 0] == batch_1_mask).all()
