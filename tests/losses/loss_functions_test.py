import pytest
import torch
from torch.testing import assert_close
import warnings

from asteroid_filterbanks import STFTFB, Encoder, transforms
from asteroid.losses import PITLossWrapper
from asteroid.losses import sdr, mse
from asteroid.losses import deep_clustering_loss, SingleSrcPMSQE
from asteroid.losses import SingleSrcNegSTOI
from asteroid.losses.multi_scale_spectral import SingleSrcMultiScaleSpectral


def assert_loss_checks_shape(loss_func, shape, arbitrary_last_dim=False, no_batch_ok=False):
    """Test that `loss_func` raises a TypeError if you are passing anything that isn't of the expected shape.

    Args:
        loss_func (callable): The loss to check, signature: `loss_func(x, y) -> Any`
        shape (tuple): Shape that the loss is expected to accept (without batch dimension).
        arbitrary_last_dim (bool, optional): Whether the last dimension may be replaced by any number of dimensions.
        no_batch_ok (bool, optional): Whether having no batch dimension is acceptable.
    """

    def _test(shape):
        loss_func(torch.randn(shape), torch.randn(shape))

    batch_size = 5

    # Check that given shape works.
    _test((batch_size, *shape))

    if not no_batch_ok:
        # Should fail without batch dim.
        with pytest.raises(TypeError):
            _test(shape)

    if arbitrary_last_dim:
        # Last dim can be arbitrary
        _test((batch_size, *shape, 4))
        _test((batch_size, *shape, 4, 5))
    else:
        # Random unsqueezes should fail.
        for dim in range(len(shape)):
            with pytest.raises(TypeError):
                _test((batch_size, *shape[:dim], 1, *shape[dim:]))


loss_properties = [
    # Pairwise loss, singlesrc loss, multisrc loss, arbitrary_last_dim?
    (sdr.pairwise_neg_sisdr, sdr.singlesrc_neg_sisdr, sdr.multisrc_neg_sisdr, False),
    (sdr.pairwise_neg_sdsdr, sdr.singlesrc_neg_sdsdr, sdr.multisrc_neg_sdsdr, False),
    (sdr.pairwise_neg_snr, sdr.singlesrc_neg_snr, sdr.multisrc_neg_snr, False),
    (mse.pairwise_mse, mse.singlesrc_mse, mse.multisrc_mse, True),
]


@pytest.mark.parametrize("n_src", [2, 3, 4])
@pytest.mark.parametrize("loss", loss_properties)
def test_sisdr_and_mse(n_src, loss):
    # Unpack the triplet
    pairwise, singlesrc, multisrc, _ = loss
    # Fake targets and estimates
    targets = torch.randn(2, n_src, 10000)
    est_targets = torch.randn(2, n_src, 10000)
    # Create the 3 PIT wrappers
    pw_wrapper = PITLossWrapper(pairwise, pit_from="pw_mtx")
    wo_src_wrapper = PITLossWrapper(singlesrc, pit_from="pw_pt")
    w_src_wrapper = PITLossWrapper(multisrc, pit_from="perm_avg")

    # Circular tests on value
    assert_close(pw_wrapper(est_targets, targets), wo_src_wrapper(est_targets, targets))
    assert_close(wo_src_wrapper(est_targets, targets), w_src_wrapper(est_targets, targets))

    # Circular tests on returned estimates
    assert_close(
        pw_wrapper(est_targets, targets, return_est=True)[1],
        wo_src_wrapper(est_targets, targets, return_est=True)[1],
    )
    assert_close(
        wo_src_wrapper(est_targets, targets, return_est=True)[1],
        w_src_wrapper(est_targets, targets, return_est=True)[1],
    )


@pytest.mark.parametrize("loss", loss_properties)
def test_sisdr_and_mse_shape_checks(loss):
    pairwise, singlesrc, multisrc, arbitrary_last_dim = loss
    assert_loss_checks_shape(pairwise, (3, 1000), arbitrary_last_dim)
    assert_loss_checks_shape(singlesrc, (1000,), arbitrary_last_dim)
    # Special case for multisrc_mse that shares the same code with
    # singlesrc_mse and thus accepts 1-dim tensors.
    no_batch_ok = multisrc == mse.multisrc_mse
    assert_loss_checks_shape(multisrc, (3, 1000), arbitrary_last_dim, no_batch_ok)


@pytest.mark.parametrize("spk_cnt", [2, 3, 4])
def test_dc(spk_cnt):
    embedding = torch.randn(10, 5 * 400, 20)
    targets = torch.zeros(10, 400, 5).random_(0, spk_cnt).long()
    loss = deep_clustering_loss(embedding, targets)
    assert loss.shape[0] == 10


@pytest.mark.parametrize("n_src", [2, 3])
def test_multi_scale_spectral_PIT(n_src):
    # Test in with reduced number of STFT scales.
    filt_list = [512, 256, 32]
    # Fake targets and estimates
    targets = torch.randn(2, n_src, 8000)
    est_targets = torch.randn(2, n_src, 8000)
    # Create PITLossWrapper in 'pw_pt' mode
    pt_loss = SingleSrcMultiScaleSpectral(
        windows_size=filt_list, n_filters=filt_list, hops_size=filt_list
    )

    loss_func = PITLossWrapper(pt_loss, pit_from="pw_pt")
    # Compute the loss
    loss_func(targets, est_targets)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_multi_scale_spectral_shape(batch_size):
    # Test in with reduced number of STFT scales.
    filt_list = [512, 256, 32]
    # Fake targets and estimates
    targets = torch.randn(batch_size, 8000)
    est_targets = torch.randn(batch_size, 8000)
    # Create PITLossWrapper in 'pw_pt' mode
    loss_func = SingleSrcMultiScaleSpectral(
        windows_size=filt_list, n_filters=filt_list, hops_size=filt_list
    )
    # Compute the loss
    loss = loss_func(targets, est_targets)
    assert loss.shape[0] == batch_size


@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_pmsqe(sample_rate):
    # Define supported STFT
    if sample_rate == 16000:
        stft = Encoder(STFTFB(kernel_size=512, n_filters=512, stride=256))
    else:
        stft = Encoder(STFTFB(kernel_size=256, n_filters=256, stride=128))
    # Usage by itself
    ref, est = torch.randn(2, 1, 16000), torch.randn(2, 1, 16000)
    ref_spec = transforms.mag(stft(ref))
    est_spec = transforms.mag(stft(est))
    loss_func = SingleSrcPMSQE(sample_rate=sample_rate)
    loss_value = loss_func(est_spec, ref_spec)
    # Assert output has shape (batch,)
    assert loss_value.shape[0] == ref.shape[0]
    # Assert support for transposed inputs.
    tr_loss_value = loss_func(est_spec.transpose(1, 2), ref_spec.transpose(1, 2))
    assert_close(loss_value, tr_loss_value)


@pytest.mark.parametrize("n_src", [2, 3])
@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_pmsqe_pit(n_src, sample_rate):
    # Define supported STFT
    if sample_rate == 16000:
        stft = Encoder(STFTFB(kernel_size=512, n_filters=512, stride=256))
    else:
        stft = Encoder(STFTFB(kernel_size=256, n_filters=256, stride=128))
    # Usage by itself
    ref, est = torch.randn(2, n_src, 16000), torch.randn(2, n_src, 16000)
    ref_spec = transforms.mag(stft(ref))
    est_spec = transforms.mag(stft(est))
    loss_func = PITLossWrapper(SingleSrcPMSQE(sample_rate=sample_rate), pit_from="pw_pt")
    # Assert forward ok.
    loss_func(est_spec, ref_spec)


@pytest.mark.parametrize("n_src", [2, 3])
@pytest.mark.parametrize("sample_rate", [8000, 16000])
@pytest.mark.parametrize("use_vad", [True, False])
@pytest.mark.parametrize("extended", [True, False])
def test_negstoi_pit(n_src, sample_rate, use_vad, extended):
    ref, est = torch.randn(2, n_src, 8000), torch.randn(2, n_src, 8000)
    singlesrc_negstoi = SingleSrcNegSTOI(
        sample_rate=sample_rate, use_vad=use_vad, extended=extended
    )
    loss_func = PITLossWrapper(singlesrc_negstoi, pit_from="pw_pt")
    # Assert forward ok.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loss_func(est, ref)
