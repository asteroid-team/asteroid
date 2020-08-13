import warnings


class VisibleDeprecationWarning(UserWarning):
    """Visible deprecation warning.

    By default, python will not show deprecation warnings, so this class
    can be used when a very visible warning is helpful, for example because
    the usage is most likely a user bug.

    """

    # Taken from numpy


class DeprecationMixin:
    """ Deprecation mixin. Example to come """

    def warn_deprecated(self):
        warnings.warn(
            "{} is deprecated since v0.1.0, it will be removed in "
            "v0.2.0. Please use {} instead."
            "".format(self.__class__.__name__, self.__class__.__bases__[0].__name__),
            VisibleDeprecationWarning,
        )


def deprecate_func(func, old_name):
    """ Function to return DeprecationWarning when a deprecated function
    is called. Example to come."""

    def func_with_warning(*args, **kwargs):
        """ Deprecated function, please read your warnings. """
        warnings.warn(
            "{} is deprecated since v0.1.0, it will be removed in "
            "v0.2.0. Please use {} instead."
            "".format(old_name, func.__name__),
            VisibleDeprecationWarning,
        )
        return func(*args, **kwargs)

    return func_with_warning
