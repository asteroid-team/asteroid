import collections
import torch
import inspect
import numpy as np
import argparse


def has_arg(fn, name):
    """ Checks if a callable accepts a given keyword argument.

    Args:
        fn (callable): Callable to inspect.
        name (str): Check if `fn` can be called with `name` as a keyword
            argument.

    Returns:
        bool: whether `fn` accepts a `name` keyword argument.
    """
    signature = inspect.signature(fn)
    parameter = signature.parameters.get(name)
    if parameter is None:
        return False
    return (parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                               inspect.Parameter.KEYWORD_ONLY))


def to_cuda(tensors):  # pragma: no cover (No CUDA on travis)
    """ Transfer tensor, dict or list of tensors to GPU.

    Args:
        tensors (:class:`torch.Tensor`, list or dict): May be a single, a
            list or a dictionary of tensors.

    Returns:
        :class:`torch.Tensor`:
            Same as input but transferred to cuda. Goes through lists and dicts
            and transfers the torch.Tensor to cuda. Leaves the rest untouched.
    """
    if isinstance(tensors, torch.Tensor):
        return tensors.cuda()
    if isinstance(tensors, list):
        return [to_cuda(tens) for tens in tensors]
    if isinstance(tensors, dict):
        for key in tensors.keys():
            tensors[key] = to_cuda(tensors[key])
        return tensors
    raise TypeError('tensors must be a tensor or a list or dict of tensors. '
                    ' Got tensors of type {}'.format(type(tensors)))


def tensors_to_device(tensors, device):
    """ Transfer tensor, dict or list of tensors to device.

    Args:
        tensors (:class:`torch.Tensor`): May be a single, a list or a
            dictionary of tensors.
        device (:class: `torch.device`): the device where to place the tensors.

    Returns:
        Union [:class:`torch.Tensor`, list, tuple, dict]:
            Same as input but transferred to device.
            Goes through lists and dicts and transfers the torch.Tensor to
            device. Leaves the rest untouched.
    """
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device)
    elif isinstance(tensors, (list, tuple)):
        return [tensors_to_device(tens, device) for tens in tensors]
    elif isinstance(tensors, dict):
        for key in tensors.keys():
            tensors[key] = tensors_to_device(tensors[key], device)
        return tensors
    else:
        return tensors


def prepare_parser_from_dict(dic, parser=None):
    """ Prepare an argparser from a dictionary.

    Args:
        dic (dict): Two-level config dictionary with unique bottom-level keys.
        parser (argparse.ArgumentParser, optional): If a parser already
            exists, add the keys from the dictionary on the top of it.

    Returns:
        argparse.ArgumentParser:
            Parser instance with groups corresponding to the first level keys
            and arguments corresponding to the second level keys with default
            values given by the values.
    """
    def standardized_entry_type(value):
        """ If the default value is None, replace NoneType by str_int_float.
            If the default value is boolean, look for boolean strings."""
        if value is None:
            return str_int_float
        if isinstance(str2bool(value), bool):
            return str2bool_arg
        return type(value)

    if parser is None:
        parser = argparse.ArgumentParser()
    for k in dic.keys():
        group = parser.add_argument_group(k)
        for kk in dic[k].keys():
            entry_type = standardized_entry_type(dic[k][kk])
            group.add_argument('--' + kk, default=dic[k][kk],
                               type=entry_type)
    return parser


def str_int_float(value):
    """ Type to convert strings to int, float (in this order) if possible.

    Args:
        value (str): Value to convert.

    Returns:
        int, float, str: Converted value.
    """
    if isint(value):
        return int(value)
    if isfloat(value):
        return float(value)
    elif isinstance(value, str):
        return value


def str2bool(value):
    """ Type to convert strings to Boolean (returns input if not boolean) """
    if not isinstance(value, str):
        return value
    if value.lower() in ('yes', 'true', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'n', '0'):
        return False
    else:
        return value


def str2bool_arg(value):
    """ Argparse type to convert strings to Boolean """
    value = str2bool(value)
    if isinstance(value, bool):
        return value
    raise argparse.ArgumentTypeError('Boolean value expected.')


def isfloat(value):
    """ Computes whether `value` can be cast to a float.

    Args:
        value (str): Value to check.

    Returns:
        bool: Whether `value` can be cast to a float.

    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def isint(value):
    """ Computes whether `value` can be cast to an int

    Args:
        value (str): Value to check.

    Returns:
        bool: Whether `value` can be cast to an int.

    """
    try:
        int(value)
        return True
    except ValueError:
        return False


def parse_args_as_dict(parser, return_plain_args=False, args=None):
    """ Get a dict of dicts out of process `parser.parse_args()`

    Top-level keys corresponding to groups and bottom-level keys corresponding
    to arguments. Under `'main_args'`, the arguments which don't belong to a
    argparse group (i.e main arguments defined before parsing from a dict) can
    be found.

    Args:
        parser (argparse.ArgumentParser): ArgumentParser instance containing
            groups. Output of `prepare_parser_from_dict`.
        return_plain_args (bool): Whether to return the output or
            `parser.parse_args()`.
        args (list): List of arguments as read from the command line.
            Used for unit testing.

    Returns:
        dict:
            Dictionary of dictionaries containing the arguments. Optionally the
            direct output `parser.parse_args()`.
    """
    args = parser.parse_args(args=args)
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


def flatten_dict(d, parent_key='', sep='_'):
    """ Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from https://stackoverflow.com/questions/6027558/
    flatten-nested-dictionaries-compressing-keys?answertab=votes#tab-top

    Args:
        d (collections.MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.

    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def average_arrays_in_dic(dic):
    """ Take average of numpy arrays in a dictionary.

    Args:
        dic (dict): Input dictionary to take average from

    Returns:
        dict: New dictionary with array averaged.

    """
    # Copy dic first
    dic = dict(dic)
    for k, v in dic.items():
        if isinstance(v, np.ndarray):
            dic[k] = float(v.mean())
    return dic


def get_wav_random_start_stop(signal_len, desired_len=4*8000):
    """ Get indexes for a chunk of signal of a given length.

    Args:
        signal_len (int): length of the signal to trim.
        desired_len (int): the length of [start:stop]

    Returns:
        tuple: random start integer, stop integer.
    """
    if signal_len == desired_len or desired_len is None:
        rand_start = 0
    else:
        rand_start = np.random.randint(0, signal_len - desired_len)
    if desired_len is None:
        stop = None
    else:
        stop = rand_start + desired_len
    return rand_start, stop
