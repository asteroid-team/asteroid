import inspect
import collections
import numpy as np


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
    return parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def flatten_dict(d, parent_key="", sep="_"):
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


def get_wav_random_start_stop(signal_len, desired_len=4 * 8000):
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
