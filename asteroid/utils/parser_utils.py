import argparse


def prepare_parser_from_dict(dic, parser=None):
    """Prepare an argparser from a dictionary.

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
        """If the default value is None, replace NoneType by str_int_float.
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
            group.add_argument("--" + kk, default=dic[k][kk], type=entry_type)
    return parser


def str_int_float(value):
    """Type to convert strings to int, float (in this order) if possible.

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
    if value.lower() in ("yes", "true", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "n", "0"):
        return False
    else:
        return value


def str2bool_arg(value):
    """ Argparse type to convert strings to Boolean """
    value = str2bool(value)
    if isinstance(value, bool):
        return value
    raise argparse.ArgumentTypeError("Boolean value expected.")


def isfloat(value):
    """Computes whether `value` can be cast to a float.

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
    """Computes whether `value` can be cast to an int

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
    """Get a dict of dicts out of process `parser.parse_args()`

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
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        args_dic[group.title] = group_dict
    args_dic["main_args"] = args_dic["optional arguments"]
    del args_dic["optional arguments"]
    if return_plain_args:
        return args_dic, args
    return args_dic
