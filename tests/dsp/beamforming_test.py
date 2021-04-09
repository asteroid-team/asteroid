import torch
import pytest
from asteroid_filterbanks import make_enc_dec, transforms as tr

from asteroid.dsp.beamforming import (
    BeamFormer,
    SCM,
    MvdrBeamformer,
    SdwMwfBeamformer,
    GEVBeamformer,
)


torch_has_complex_support = tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 8)

_stft, _istft = make_enc_dec("stft", kernel_size=512, n_filters=512, stride=128)
stft = lambda x: tr.to_torch_complex(_stft(x))
istft = lambda x: _istft(tr.from_torch_complex(x))


@pytest.mark.skipif(not torch_has_complex_support, "No complex support ")
def _default_beamformer_test(beamformer: BeamFormer, n_mics=4, *args, **kwargs):
    scm = SCM()

    speech = torch.randn(1, n_mics, 16000 * 6)
    noise = torch.randn(1, n_mics, 16000 * 6)
    mix = speech + noise
    # GeV Beamforming
    mix_stft = stft(mix)
    speech_stft = stft(speech)
    noise_stft = stft(noise)
    sigma_ss = scm(speech_stft)
    sigma_nn = scm(noise_stft)

    Ys_gev = beamformer.forward(mix=mix_stft, target_scm=sigma_ss, noise_scm=sigma_nn)
    ys_gev = istft(Ys_gev)


@pytest.mark.skipif(not torch_has_complex_support, reason="No complex support ")
@pytest.mark.parametrize("n_mics", [2, 3, 4])
def test_gev(n_mics):
    _default_beamformer_test(GEVBeamformer(), n_mics=n_mics)


@pytest.mark.skipif(not torch_has_complex_support, reason="No complex support ")
@pytest.mark.parametrize("n_mics", [2, 3, 4])
def test_mvdr(n_mics):
    _default_beamformer_test(MvdrBeamformer(), n_mics=n_mics)


@pytest.mark.skipif(not torch_has_complex_support, reason="No complex support ")
@pytest.mark.parametrize("n_mics", [2, 3, 4])
@pytest.mark.parametrize("mu", [1.0, 2.0, 0])
def test_mwf(n_mics, mu):
    _default_beamformer_test(SdwMwfBeamformer(mu=mu), n_mics=n_mics)
