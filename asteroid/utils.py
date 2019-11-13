"""
Package-level utils.
@author : Manuel Pariente, Inria-Nancy
"""
import torch
import inspect
import argparse


def has_arg(fn, name):
    """Checks if a callable accepts a given keyword argument.
    # Arguments
        fn: Callable to inspect.
        name: Check if `fn` can be called with `name` as a keyword argument.

    # Returns
        bool, whether `fn` accepts a `name` keyword argument.
    """
    signature = inspect.signature(fn)
    parameter = signature.parameters.get(name)
    if parameter is None:
        return False
    return (parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                               inspect.Parameter.KEYWORD_ONLY))


def to_cuda(tensors):
    """Transfer tensor, dict or list of tensors to GPU.
    Args:
        tensors: torch.Tensor, dict or list of torch.Tensor.
    Returns:
        Same as input but transferred to cuda. Goes through lists and dicts
        and transfers the torch.Tensor to cuda. Leaves the rest untouched.
    """
    if isinstance(tensors, torch.Tensor):
        return tensors.cuda()
    elif isinstance(tensors, list):
        return [to_cuda(tens) for tens in tensors]
    elif isinstance(tensors, dict):
        for key in tensors.keys():
            tensors[key] = to_cuda(tensors[key])
    else:
        return tensors


def prepare_parser_from_dict(dic, parser=None):
    """ Prepare an argparser from a dictionary.

    Args:
        dic: Dictionary. Two-level config dictionary with unique
            bottom-level keys.
        parser: ArgumentParser instance (optional). If a parser already
            exists add the keys from the dictionary on the top of it.
    Returns:
        argparse.ArgumentParser instance with groups corresponding to the
        first level keys and arguments corresponding to the second level keys
        with default values given by the values.
    """
    if parser is None:
        parser = argparse.ArgumentParser()
    for k in dic.keys():
        group = parser.add_argument_group(k)
        for kk in dic[k].keys():
            group.add_argument('--' + kk, default=dic[k][kk],
                               type=type(dic[k][kk]))
    return parser


def parse_args_as_dict(parser, return_plain_args=False):
    """ Post-process `parser.parse_args()` to get a dictionary of
    dictionary. Top-level keys corresponding to groups and bottom-level
    keys corresponding to arguments.
    Under `'main_args'`, the arguments which don't belong to a argparse group
    (i.e main arguments defined before parsing from a dict)can be found.
    Args:
        parser: ArgumentParser instance containing groups
            Output of `prepare_parser_from_dict`.
        return_plain_args: Boolean. Whether to return
            the output or `parser.parse_args()`.
    Returns:
        A dictionary of dictionary containing the arguments.
        Optionally the direct output `parser.parse_args()`.
    """
    args = parser.parse_args()
    args_dic = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None)
                      for a in group._group_actions}
        args_dic[group.title] = group_dict
    args_dic['main_args'] = args_dic['optional arguments']
    del args_dic['optional arguments']
    if return_plain_args:
        return args_dic, args
    return args_dic
