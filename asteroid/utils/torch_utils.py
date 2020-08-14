import torch
from torch import nn
from collections import OrderedDict


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
    raise TypeError(
        "tensors must be a tensor or a list or dict of tensors. "
        " Got tensors of type {}".format(type(tensors))
    )


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


def pad_x_to_y(x, y, axis=-1):
    """  Pad first argument to have same size as second argument

    Args:
        x (torch.Tensor): Tensor to be padded.
        y (torch.Tensor): Tensor to pad x to.
        axis (int): Axis to pad on.

    Returns:
        torch.Tensor, x padded to match y's shape.
    """
    if axis != -1:
        raise NotImplementedError
    inp_len = y.size(axis)
    output_len = x.size(axis)
    return nn.functional.pad(x, [0, inp_len - output_len])


def load_state_dict_in(state_dict, model):
    """ Strictly loads state_dict in model, or the next submodel.
        Useful to load standalone model after training it with System.

    Args:
        state_dict (OrderedDict): the state_dict to load.
        model (torch.nn.Module): the model to load it into

    Returns:
        torch.nn.Module: model with loaded weights.

    # .. note:: Keys in a state_dict look like object1.object2.layer_name.weight.etc
        We first try to load the model in the classic way.
        If this fail we removes the first left part of the key to obtain
        object2.layer_name.weight.etc.
        Blindly loading with strictly=False should be done with some logging
        of the missing keys in the state_dict and the model.

    """
    try:
        # This can fail if the model was included into a bigger nn.Module
        # object. For example, into System.
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        # keys look like object1.object2.layer_name.weight.etc
        # The following will remove the first left part of the key to obtain
        # object2.layer_name.weight.etc.
        # Blindly loading with strictly=False should be done with some
        # new_state_dict of the missing keys in the state_dict and the model.
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_k = k[k.find(".") + 1 :]
            new_state_dict[new_k] = v
        model.load_state_dict(new_state_dict, strict=True)
    return model


def are_models_equal(model1, model2):
    """ Check for weights equality between models.

    Args:
        model1 (nn.Module): model instance to be compared.
        model2 (nn.Module): second model instance to be compared.

    Returns:
        bool: Whether all model weights are equal.
    """
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True
