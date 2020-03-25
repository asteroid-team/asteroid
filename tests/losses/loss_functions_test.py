import pytest
import torch
from torch.testing import assert_allclose

from asteroid.filterbanks import STFTFB, Encoder, transforms
from asteroid.losses import PITLossWrapper
from asteroid.losses import sdr
from asteroid.losses import singlesrc_mse, pairwise_mse, multisrc_mse
from asteroid.losses import deep_clustering_loss, SingleSrcPMSQE
from asteroid.losses.multi_scale_spectral import SingleSrcMultiScaleSpectral

@pytest.mark.parametrize("n_src", [2, 3, 4])
@pytest.mark.parametrize("function_triplet", [
    [sdr.pairwise_neg_sisdr, sdr.singlesrc_neg_sisdr, sdr.multisrc_neg_sisdr],
    [sdr.pairwise_neg_sdsdr, sdr.singlesrc_neg_sdsdr, sdr.multisrc_neg_sdsdr],
    [sdr.pairwise_neg_snr, sdr.singlesrc_neg_snr, sdr.multisrc_neg_snr],
    [pairwise_mse, singlesrc_mse, multisrc_mse],
])
def test_sisdr(n_src, function_triplet):
    # Unpack the triplet
    pairwise, nosrc, nonpit = function_triplet
    # Fake targets and estimates
    targets = torch.randn(2, n_src, 10000)
    est_targets = torch.randn(2, n_src, 10000)
    # Create the 3 PIT wrappers
    pw_wrapper = PITLossWrapper(pairwise, pit_from='pw_mtx')
    wo_src_wrapper = PITLossWrapper(nosrc, pit_from='pw_pt')
    w_src_wrapper = PITLossWrapper(nonpit, pit_from='perm_avg')

    # Circular tests on value
    assert_allclose(pw_wrapper(est_targets, targets),
                    wo_src_wrapper(est_targets, targets))
    assert_allclose(wo_src_wrapper(est_targets, targets),
                    w_src_wrapper(est_targets, targets))

    # Circular tests on returned estimates
    assert_allclose(pw_wrapper(est_targets, targets, return_est=True)[1],
                    wo_src_wrapper(est_targets, targets, return_est=True)[1])
    assert_allclose(wo_src_wrapper(est_targets, targets, return_est=True)[1],
                    w_src_wrapper(est_targets, targets, return_est=True)[1])


@pytest.mark.parametrize("spk_cnt", [2, 3, 4])
def test_dc(spk_cnt):
    embedding = torch.randn(10, 5*400, 20)
    targets = torch.LongTensor(10, 400, 5).random_(0, spk_cnt)
    loss = deep_clustering_loss(embedding, targets, spk_cnt)
    assert loss.shape[0] == 10


@pytest.mark.parametrize("n_src", [2, 3])
def test_multi_scale_spectral_PIT(n_src):
    # Test in with reduced number of STFT scales.
    filt_list = [512, 256, 32]
    # Fake targets and estimates
    targets = torch.randn(2, n_src, 8000)
    est_targets = torch.randn(2, n_src, 8000)
    # Create PITLossWrapper in 'pw_pt' mode
    pt_loss = SingleSrcMultiScaleSpectral(windows_size=filt_list,
                                          n_filters=filt_list,
                                          hops_size=filt_list)
    loss_func = PITLossWrapper(pt_loss, pit_from='pw_pt')
    # Compute the loss
    loss = loss_func(targets, est_targets)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_multi_scale_spectral_shape(batch_size):
    # Test in with reduced number of STFT scales.
    filt_list = [512, 256, 32]
    # Fake targets and estimates
    targets = torch.randn(batch_size, 8000)
    est_targets = torch.randn(batch_size, 8000)
    # Create PITLossWrapper in 'pw_pt' mode
    loss_func = SingleSrcMultiScaleSpectral(windows_size=filt_list,
                                            n_filters=filt_list,
                                            hops_size=filt_list)
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
    ref_spec = transforms.take_mag(stft(ref))
    est_spec = transforms.take_mag(stft(est))
    loss_func = SingleSrcPMSQE(sample_rate=sample_rate)
    loss_value = loss_func(est_spec, ref_spec)
    # Assert output has shape (batch,)
    assert loss_value.shape[0] == ref.shape[0]
    # Assert support for transposed inputs.
    tr_loss_value = loss_func(est_spec.transpose(1, 2),
                              ref_spec.transpose(1, 2))
    assert_allclose(loss_value, tr_loss_value)


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
    ref_spec = transforms.take_mag(stft(ref))
    est_spec = transforms.take_mag(stft(est))
    loss_func = PITLossWrapper(SingleSrcPMSQE(sample_rate=sample_rate),
                               pit_from='pw_pt')
    # Assert forward ok.
    loss_value = loss_func(ref_spec, est_spec)
