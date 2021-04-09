import warnings
import inspect
from functools import wraps


class VisibleDeprecationWarning(UserWarning):
    """Visible deprecation warning.

    By default, python will not show deprecation warnings, so this class
    can be used when a very visible warning is helpful, for example because
    the usage is most likely a user bug.

    """

    # Taken from numpy


def mark_deprecated(message, version=None):
    """Decorator to add deprecation message.

    Args:
        message: Migration steps to be given to users.
    """

    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            from_what = "a future release" if version is None else f"asteroid v{version}"
            warn_message = (
                f"{func.__module__}.{func.__name__} has been deprecated "
                f"and will be removed from {from_what}. "
                f"{message}"
            )
            warnings.warn(warn_message, VisibleDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapped

    return decorator


def is_overridden(method_name, obj, parent=None) -> bool:
    """Check if `method_name` from parent is overridden in `obj`.

    Args:
        method_name (str): Name of the method.
        obj: Instance or class that potentially overrode the method.
        parent: parent class with which to compare. If None, traverse the MRO
            for the first parent that has the method.

    Raises RuntimeError if `parent` is not a parent class and if `parent`
    doesn't have the method. Or, if `parent` was None, that none of the
    potential parents had the method.
    """

    def get_mro(cls):
        try:
            return inspect.getmro(cls)
        except AttributeError:
            return inspect.getmro(cls.__class__)

    def first_parent_with_method(fn, mro_list):
        for cls in mro_list[::-1]:
            if hasattr(cls, fn):
                return cls
        return None

    if not hasattr(obj, method_name):
        return False

    try:
        instance_attr = getattr(obj, method_name)
    except AttributeError:
        return False
        return False

    mro = get_mro(obj)[1:]  # All parent classes in order, self excluded
    parent = parent if parent is not None else first_parent_with_method(method_name, mro)

    if parent not in mro:
        raise RuntimeError(f"`{obj}` has no parent that defined method {method_name}`.")

    if not hasattr(parent, method_name):
        raise RuntimeError(f"Parent `{parent}` does have method `{method_name}`")

    super_attr = getattr(parent, method_name)
    return instance_attr.__code__ is not super_attr.__code__
