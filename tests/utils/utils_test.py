import argparse
import torch
from torch.testing import assert_allclose
import pytest

from asteroid import utils
from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict


@pytest.fixture(scope="module")
def parser():
    # Create dictionary as from .yml file
    def_conf = dict(
        top1=dict(key1=2),
        top2=dict(key2=None, key3=True)
    )
    # Create empty parser and add top level keys
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_key', default='')
    # Populate parser from def_conf
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    return parser


def test_namespace_dic(parser):
    fake_args = ['--key2', 'hey', '--key3', '0']
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True,
                                             args=fake_args)
    assert arg_dic['main_args']['main_key'] == plain_args.main_key
    assert arg_dic['top2']['key3'] == plain_args.key3


@pytest.mark.parametrize("inp", ['one_string', 3, 3.14])
def test_none_default(parser, inp):
    # If the default is None, convert the input string into an int, a float
    # or string.
    fake_args = ['--key2', str(inp)]  # Note : inp is converted to string
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True,
                                             args=fake_args)
    assert type(plain_args.key2) == type(inp)


def test_boolean(parser):
    fake_args = ['--key3', 'y']
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True,
                                             args=fake_args)
    assert plain_args.key3 is True

    fake_args = ['--key3', 'n']
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True,
                                             args=fake_args)
    assert plain_args.key3 is False


@pytest.mark.parametrize("tensors", [
    torch.randn(10, 10),  # tensor
    dict(tensor_a=torch.randn(10, 10), tensor_b=torch.randn(12, 12)),  # dict
    [torch.randn(10, 10), torch.randn(12, 12)],  # list
    dict(
        tensor_a=torch.randn(10, 10),
        tensor_list=[torch.randn(12, 12), torch.randn(14, 14)],
        tensor_dict=dict(u=torch.randn(8, 10), v=torch.randn(10, 8))
    ),
    [dict(u=torch.randn(8, 10), v=torch.randn(10, 8)), torch.randn(10, 10)]
])
def test_transfer(tensors):
    if isinstance(tensors, torch.Tensor):
        assert_allclose(utils.tensors_to_device(tensors, 'cpu'), tensors)
    if isinstance(tensors, list):
        assert list(utils.tensors_to_device(tensors, 'cpu')) == list(tensors)
    if isinstance(tensors, dict):
        assert dict(utils.tensors_to_device(tensors, 'cpu')) == dict(tensors)

