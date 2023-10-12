import itertools
import argparse
from collections.abc import MutableMapping
import torch
from torch.testing import assert_close
import pytest
import numpy as np

from asteroid import utils
from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict


@pytest.fixture(scope="module")
def parser():
    # Create dictionary as from .yml file
    def_conf = dict(top1=dict(key1=2), top2=dict(key2=None, key3=True))
    # Create empty parser and add top level keys
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_key", default="")
    # Populate parser from def_conf
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    return parser


def test_namespace_dic(parser):
    fake_args = ["--key2", "hey", "--key3", "0"]
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True, args=fake_args)
    assert arg_dic["main_args"]["main_key"] == plain_args.main_key
    assert arg_dic["top2"]["key3"] == plain_args.key3


@pytest.mark.parametrize("inp", ["one_string", 3, 3.14])
def test_none_default(parser, inp):
    # If the default is None, convert the input string into an int, a float
    # or string.
    fake_args = ["--key2", str(inp)]  # Note : inp is converted to string
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True, args=fake_args)
    assert type(plain_args.key2) == type(inp)


def test_boolean(parser):
    fake_args = ["--key3", "y"]
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True, args=fake_args)
    assert plain_args.key3 is True

    fake_args = ["--key3", "n"]
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True, args=fake_args)
    assert plain_args.key3 is False


@pytest.mark.parametrize(
    "tensors",
    [
        torch.randn(10, 10),  # tensor
        dict(tensor_a=torch.randn(10, 10), tensor_b=torch.randn(12, 12)),  # dict
        [torch.randn(10, 10), torch.randn(12, 12)],  # list
        dict(
            tensor_a=torch.randn(10, 10),
            tensor_list=[torch.randn(12, 12), torch.randn(14, 14)],
            tensor_dict=dict(u=torch.randn(8, 10), v=torch.randn(10, 8)),
        ),
        [dict(u=torch.randn(8, 10), v=torch.randn(10, 8)), torch.randn(10, 10)],
    ],
)
def test_transfer(tensors):
    if isinstance(tensors, torch.Tensor):
        assert_close(utils.tensors_to_device(tensors, "cpu"), tensors)
    if isinstance(tensors, list):
        assert list(utils.tensors_to_device(tensors, "cpu")) == list(tensors)
    if isinstance(tensors, dict):
        assert dict(utils.tensors_to_device(tensors, "cpu")) == dict(tensors)


def test_flatten_dict():
    to_test = dict(
        top1=[1, 2],
        top2=dict(
            sub1="hey", sub2=dict(subsub1=True, subsub2=["This", "is", "a", "list"]), sub3=False
        ),
    )
    flat_dic = utils.flatten_dict(to_test)
    for k, v in flat_dic.items():
        assert not isinstance(v, MutableMapping)


def test_average_array_in_dic():
    d = dict(a="hey", b=np.array([1.0, 3.0]), c=2)
    av_d = utils.average_arrays_in_dic(d)
    d_should_be = dict(a="hey", b=2.0, c=2)
    # We need the arrays to be averaged
    assert av_d == d_should_be


@pytest.mark.parametrize("sig_len,desired", itertools.product([50, 100, 150], [50, 100, 150]))
def test_get_start_stop(sig_len, desired):
    sig = np.random.randn(sig_len)
    start, stop = utils.get_wav_random_start_stop(len(sig), desired_len=desired)
    # Start must be chosen so that len(sig[start:]) is at least `desired`
    # (or if `desired < sig_len`, then exactly `sig_len`).
    assert start < max(1, sig_len - desired)
    # Stop must be chosen so that `start + stop == desired`
    # (or if `desired < sig_len`, then exactly `sig_len`).
    assert stop == start + min(sig_len, desired)


def test_unet_decoder_args():
    a, b, c, d = np.random.randint(1, 100, size=4)
    encoders = (
        (a, b, "ks1", "st1", "pad1"),
        (b, c, "ks2", "st2", "pad2"),
        (c, d, "ks3", "st3", "pad3"),
    )
    assert utils.unet_decoder_args(encoders, skip_connections=False) == (
        (1 * d, c, "ks3", "st3", "pad3"),
        (1 * c, b, "ks2", "st2", "pad2"),
        (1 * b, a, "ks1", "st1", "pad1"),
    )
    assert utils.unet_decoder_args(encoders, skip_connections=True) == (
        (1 * d, c, "ks3", "st3", "pad3"),
        (2 * c, b, "ks2", "st2", "pad2"),
        (2 * b, a, "ks1", "st1", "pad1"),
    )
