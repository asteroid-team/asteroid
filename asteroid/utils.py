"""
Package-level utils.
@author : Manuel Pariente, Inria-Nancy
"""
import inspect
import torch


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
