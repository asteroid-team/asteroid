from glob import glob
import os.path as osp
import re

import pytest
import torch
from torch.testing import assert_close

from asteroid.models import DeMask


# def get_input_files(dirname, prefix=''):
#     input_files = []
#     for file in glob(osp.join(dirname, '*.pth')):
#         basename = osp.basename(file)
#         if basename.startswith(prefix):
#             input_files.append(file)
#     input_files.sort()
#     return input_files
#
#
# def get_output_files(input_files, prefix='', prefix_sub=''):
#     pattern = re.compile(f'^{prefix}')
#     output_files = []
#     for in_file in input_files:
#         dirname = osp.dirname(in_file)
#         basename = osp.basename(in_file)
#         output_basename = pattern.sub(prefix_sub, basename)
#         output_file = osp.join(dirname, output_basename)
#         output_files.append(output_file)
#     return output_files
#
#
# def compatibility_data():
#     model_dir = 'demask_rsc'
#     io_data_dir = osp.join('demask_rsc', 'io_data')
#     input_files = get_input_files(io_data_dir, prefix='input')
#     output_files = get_output_files(
#         input_files,
#         prefix="input",
#         prefix_sub="output",
#     )
#     return list(zip(input_files, output_files))
#
#
# @pytest.fixture(scope='module')
# def model():
#     model_path = osp.join('demask_rsc', 'model.pth')
#     return DeMask.from_pretrained(model_path, sample_rate=16000).eval()
#
#
# @pytest.mark.parametrize('in_file, ref_file', (*compatibility_data(),))
# def test_demask_compat(model, in_file, ref_file):
#     in_data = torch.load(in_file)
#     expected_data = torch.load(ref_file)
#     with torch.no_grad():
#         output = model(in_data)
#         assert_close(output, expected_data)
#
#
# def test_get_model_args(model):
#     expected = {
#         'activation': 'relu',
#         'dropout': 0,
#         'fb_kwargs': {},
#         'fb_name': 'stft',
#         'hidden_dims': [1024],
#         'input_type': 'mag',
#         'kernel_size': 512,
#         'mask_act': 'relu',
#         'n_filters': 512,
#         'norm_type': 'gLN',
#         'output_type': 'mag',
#         'sample_rate': 16000,
#         'stride': 256,
#     }
#     assert model.get_model_args() == expected


@pytest.mark.parametrize("input_type", ("mag", "cat", "reim"))
@pytest.mark.parametrize("fb_name", ("stft", "free"))
@pytest.mark.parametrize("output_type", ("mag", "reim"))
@pytest.mark.parametrize(
    "data",
    (
        (torch.rand(130, requires_grad=False) - 0.5) * 2,
        (torch.rand(1, 100, requires_grad=False) - 0.5) * 2,
        (torch.rand(3, 50, requires_grad=False) - 0.5) * 2,
        (torch.rand(1, 1, 50, requires_grad=False) - 0.5) * 2,
        (torch.rand(2, 1, 50, requires_grad=False) - 0.5) * 2,
    ),
)
def test_forward(input_type, output_type, fb_name, data):
    demask = DeMask(
        input_type=input_type,
        output_type=output_type,
        fb_name=fb_name,
        hidden_dims=(16,),
        kernel_size=8,
        n_filters=8,
        stride=4,
    )
    demask = demask.eval()
    with torch.no_grad():
        demask(data)


def test_sample_rate():
    demask = DeMask(hidden_dims=(16,), kernel_size=8, n_filters=8, stride=4, sample_rate=9704)
    assert demask.sample_rate == 9704


def test_get_model_args():
    demask = DeMask()
    expected = {
        "activation": "relu",
        "dropout": 0,
        "fb_name": "STFTFB",
        "hidden_dims": (1024,),
        "input_type": "mag",
        "kernel_size": 512,
        "mask_act": "relu",
        "n_filters": 512,
        "norm_type": "gLN",
        "output_type": "mag",
        "sample_rate": 16000,
        "stride": 256,
    }
    assert demask.get_model_args() == expected
