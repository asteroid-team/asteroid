import functools

import torch
from torch import nn
from collections import OrderedDict


def to_cuda(tensors):  # pragma: no cover
    """Transfer tensor, dict or list of tensors to GPU.

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
    """Transfer tensor, dict or list of tensors to device.

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


def get_device(tensor_or_module, default=None):
    """Get the device of a tensor or a module.

    Args:
        tensor_or_module (Union[torch.Tensor, torch.nn.Module]):
            The object to get the device from. Can be a ``torch.Tensor``,
            a ``torch.nn.Module``, or anything else that has a ``device`` attribute
            or a ``parameters() -> Iterator[torch.Tensor]`` method.
        default (Optional[Union[str, torch.device]]): If the device can not be
            determined, return this device instead. If ``None`` (the default),
            raise a ``TypeError`` instead.

    Returns:
        torch.device: The device that ``tensor_or_module`` is on.
    """
    if hasattr(tensor_or_module, "device"):
        return tensor_or_module.device
    elif hasattr(tensor_or_module, "parameters"):
        return next(tensor_or_module.parameters()).device
    elif default is None:
        raise TypeError(f"Don't know how to get device of {type(tensor_or_module)} object")
    else:
        return torch.device(default)


def is_tracing():
    # Taken for pytorch for compat in 1.6.0
    """
    Returns ``True`` in tracing (if a function is called during the tracing of
    code with ``torch.jit.trace``) and ``False`` otherwise.
    """
    return torch._C._is_tracing()


def script_if_tracing(fn):
    # Taken for pytorch for compat in 1.6.0
    """
    Compiles ``fn`` when it is first called during tracing. ``torch.jit.script``
    has a non-negligible start up time when it is first called due to
    lazy-initializations of many compiler builtins. Therefore you should not use
    it in library code. However, you may want to have parts of your library work
    in tracing even if they use control flow. In these cases, you should use
    ``@torch.jit.script_if_tracing`` to substitute for
    ``torch.jit.script``.

    Arguments:
        fn: A function to compile.

    Returns:
        If called during tracing, a :class:`ScriptFunction` created by `
        `torch.jit.script`` is returned. Otherwise, the original function ``fn`` is returned.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not is_tracing():
            # Not tracing, don't do anything
            return fn(*args, **kwargs)

        compiled_fn = torch.jit.script(wrapper.__original_fn)  # type: ignore
        return compiled_fn(*args, **kwargs)

    wrapper.__original_fn = fn  # type: ignore
    wrapper.__script_if_tracing_wrapper = True  # type: ignore

    return wrapper


@script_if_tracing
def pad_x_to_y(x: torch.Tensor, y: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Right-pad or right-trim first argument to have same size as second argument

    Args:
        x (torch.Tensor): Tensor to be padded.
        y (torch.Tensor): Tensor to pad `x` to.
        axis (int): Axis to pad on.

    Returns:
        torch.Tensor, `x` padded to match `y`'s shape.
    """
    if axis != -1:
        raise NotImplementedError
    inp_len = y.shape[axis]
    output_len = x.shape[axis]
    return nn.functional.pad(x, [0, inp_len - output_len])


def load_state_dict_in(state_dict, model):
    """Strictly loads state_dict in model, or the next submodel.
        Useful to load standalone model after training it with System.

    Args:
        state_dict (OrderedDict): the state_dict to load.
        model (torch.nn.Module): the model to load it into

    Returns:
        torch.nn.Module: model with loaded weights.

    .. note:: Keys in a state_dict look like ``object1.object2.layer_name.weight.etc``
        We first try to load the model in the classic way.
        If this fail we removes the first left part of the key to obtain
        ``object2.layer_name.weight.etc``.
        Blindly loading with ``strictly=False`` should be done with some logging
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
    """Check for weights equality between models.

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


@script_if_tracing
def jitable_shape(tensor):
    """Gets shape of ``tensor`` as ``torch.Tensor`` type for jit compiler

    .. note::
        Returning ``tensor.shape`` of ``tensor.size()`` directly is not torchscript
        compatible as return type would not be supported.

    Args:
        tensor (torch.Tensor): Tensor

    Returns:
        torch.Tensor: Shape of ``tensor``
    """
    return torch.tensor(tensor.shape)
