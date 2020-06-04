import random
import pytest
import torch
from torch.testing import assert_allclose
import numpy as np

from asteroid import filterbanks as fb
from asteroid.filterbanks import transforms


COMPLEX_FBS = [
    fb.STFTFB,
    fb.ParamSincFB,
    fb.AnalyticFreeFB
]


@pytest.fixture(scope="module")
def fb_config_list():
    keys = ['n_filters', 'kernel_size', 'stride']
    param_list = [
        [256, 256, 128],
        [256, 256, 64],
        [512, 32, None],
        [512, 16, 8],
    ]
    return [dict(zip(keys, values)) for values in param_list]


@pytest.fixture(scope="module")
def encoder_list(fb_config_list):
    enc_list = []
    for fb_class in COMPLEX_FBS:
        for fb_config in fb_config_list:
            enc_list.append(make_encoder_from(fb_class, fb_config))
    return enc_list


def make_encoder_from(fb_class, config):
    enc = fb.Encoder(fb_class(**config))
    fb_dim = enc.filterbank.n_feats_out
    return enc, fb_dim


def test_mag_mask(encoder_list):
    """ Assert identity mask works. """
    for (enc, fb_dim) in encoder_list:
        tf_rep = enc(torch.randn(2, 1, 8000))  # [batch, freq, time]
        id_mag_mask = torch.ones((1, fb_dim//2, 1))
        masked = transforms.apply_mag_mask(tf_rep, id_mag_mask, dim=1)
        assert_allclose(masked, tf_rep)


def test_reim_mask(encoder_list):
    """ Assert identity mask works. """
    for (enc, fb_dim) in encoder_list:
        tf_rep = enc(torch.randn(2, 1, 8000))  # [batch, freq, time]
        id_reim_mask = torch.ones((1, fb_dim, 1))
        masked = transforms.apply_real_mask(tf_rep, id_reim_mask, dim=1)
        assert_allclose(masked, tf_rep)


def test_comp_mask(encoder_list):
    """ Assert identity mask works. """
    for (enc, fb_dim) in encoder_list:
        tf_rep = enc(torch.randn(2, 1, 8000))  # [batch, freq, time]
        id_complex_mask = torch.cat((torch.ones((1, fb_dim // 2, 1)),
                                     torch.zeros((1, fb_dim // 2, 1))),
                                    dim=1)
        masked = transforms.apply_complex_mask(tf_rep, id_complex_mask,
                                               dim=1)
        assert_allclose(masked, tf_rep)


def test_reim(encoder_list):
    for (enc, fb_dim) in encoder_list:
        tf_rep = enc(torch.randn(2, 1, 16000))  # [batch, freq, time]
        assert_allclose(tf_rep, transforms.take_reim(tf_rep))


def test_mag(encoder_list):
    for (enc, fb_dim) in encoder_list:
        tf_rep = enc(torch.randn(2, 1, 16000))  # [batch, freq, time]
        batch, freq, time = tf_rep.shape
        mag = transforms.take_mag(tf_rep, dim=1)
        assert mag.shape == (batch, freq // 2, time)


def test_cat(encoder_list):
    for (enc, fb_dim) in encoder_list:
        tf_rep = enc(torch.randn(2, 1, 16000))  # [batch, freq, time]
        batch, freq, time = tf_rep.shape
        mag = transforms.take_cat(tf_rep, dim=1)
        assert mag.shape == (batch, 3 * (freq // 2), time)


@pytest.mark.parametrize("np_torch_tuple", [
    ([0], [0, 0]),
    ([1j], [0, 1]),
    ([-1], [-1, 0]),
    ([-1j], [0, -1]),
    ([1 + 1j], [1, 1]),
])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_to_numpy(np_torch_tuple, dim):
    """ Test torch --> np conversion (right angles)"""
    from_np, from_torch = np_torch_tuple
    if dim == 0:
        np_array = np.array(from_np)
        torch_tensor = torch.tensor(from_torch)
    elif dim == 1:
        np_array = np.array([from_np])
        torch_tensor = torch.tensor([from_torch])
    elif dim == 2:
        np_array = np.array([[from_np]])
        torch_tensor = torch.tensor([[from_torch]])
    else:
        return
    np_from_torch = transforms.to_numpy(torch_tensor, dim=dim)
    np.testing.assert_allclose(np_array, np_from_torch)


@pytest.mark.parametrize("np_torch_tuple", [
    ([0], [0, 0]),
    ([1j], [0, 1]),
    ([-1], [-1, 0]),
    ([-1j], [0, -1]),
    ([1 + 1j], [1, 1]),
])
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_from_numpy(np_torch_tuple, dim):
    """ Test np --> torch conversion (right angles)"""
    from_np, from_torch = np_torch_tuple
    if dim == 0:
        np_array = np.array(from_np)
        torch_tensor = torch.tensor(from_torch)
    elif dim == 1:
        np_array = np.array([from_np])
        torch_tensor = torch.tensor([from_torch])
    elif dim == 2:
        np_array = np.array([[from_np]])
        torch_tensor = torch.tensor([[from_torch]])
    else:
        return
    torch_from_np = transforms.from_numpy(np_array, dim=dim)
    np.testing.assert_allclose(torch_tensor, torch_from_np)


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_return_ticket_np_torch(dim):
    """ Test torch --> np --> torch --> np conversion"""
    max_tested_ndim = 4
    # Random tensor shape
    tensor_shape = [random.randint(1, 10) for _ in range(max_tested_ndim)]
    # Make sure complex dimension has even shape
    tensor_shape[dim] = 2 * tensor_shape[dim]
    complex_tensor = torch.randn(tensor_shape)
    np_array = transforms.to_numpy(complex_tensor, dim=dim)
    tensor_back = transforms.from_numpy(np_array, dim=dim)
    np_back = transforms.to_numpy(tensor_back, dim=dim)
    # Check torch --> np --> torch
    assert_allclose(complex_tensor, tensor_back)
    # Check np --> torch --> np
    np.testing.assert_allclose(np_array, np_back)


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_angle_mag_recompostion(dim):
    """ Test complex --> (mag, angle) --> complex conversions"""
    max_tested_ndim = 4
    # Random tensor shape
    tensor_shape = [random.randint(1, 10) for _ in range(max_tested_ndim)]
    # Make sure complex dimension has even shape
    tensor_shape[dim] = 2 * tensor_shape[dim]
    complex_tensor = torch.randn(tensor_shape)
    phase = transforms.angle(complex_tensor, dim=dim)
    mag = transforms.take_mag(complex_tensor, dim=dim)
    tensor_back = transforms.from_mag_and_phase(mag, phase, dim=dim)
    assert_allclose(complex_tensor, tensor_back)


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_check_complex_error(dim):
    """ Test error in angle """
    not_complex = torch.randn(3, 5, 7, 9, 15)
    with pytest.raises(AssertionError):
        phase = transforms.check_complex(not_complex, dim=dim)


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_torchaudio_format(dim):
    max_tested_ndim = 4
    # Random tensor shape
    tensor_shape = [random.randint(1, 10) for _ in range(max_tested_ndim)]
    # Make sure complex dimension has even shape
    tensor_shape[dim] = 2 * tensor_shape[dim]
    complex_tensor = torch.randn(tensor_shape)
    ta_tensor = transforms.to_torchaudio(complex_tensor, dim=dim)
    tensor_back = transforms.from_torchaudio(ta_tensor, dim=dim)
    assert_allclose(complex_tensor, tensor_back)
    assert ta_tensor.shape[-1] == 2
