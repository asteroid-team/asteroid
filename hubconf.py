from asteroid import models

dependencies = ['torch']


def conv_tasnet(name_url_or_file=None, *args, **kwargs):
    """ Load (pretrained) ConvTasNet model

    Args:
        name_url_or_file (str): Model name (we'll find the URL),
            model URL to download model, path to model file.
            If None (default), ConvTasNet is instantiated but no pretrained
            weights are loaded.
        *args: Arguments to pass to ConvTasNet.
        **kwargs: Keyword arguments to pass to ConvTasNet.

    Returns:
        ConvTasNet instance (with ot without pretrained weights).

    Examples:
        >>> from torch import hub
        >>> # Instantiate without pretrained weights
        >>> model = hub.load('mpariente/asteroid', 'conv_tasnet', n_src=2)
        >>> # Use pretrained weights
        >>> URL = "TOCOME"
        >>> model = hub.load('mpariente/asteroid', 'conv_tasnet', URL)
    """
    # No pretrained weights
    if name_url_or_file is None:
        return models.ConvTasNet(*args, **kwargs)
    return models.ConvTasNet.from_pretrained(name_url_or_file, *args, **kwargs)


def dprnn_tasnet(name_url_or_file=None, *args, **kwargs):
    """ Load (pretrained) DPRNNTasNet model

    Args:
        name_url_or_file (str): Model name (we'll find the URL),
            model URL to download model, path to model file.
            If None (default), DPRNNTasNet is instantiated but no pretrained
            weights are loaded.
        *args: Arguments to pass to DPRNNTasNet.
        **kwargs: Keyword arguments to pass to DPRNNTasNet.

    Returns:
        DPRNNTasNet instance (with ot without pretrained weights).

    Examples:
        >>> from torch import hub
        >>> # Instantiate without pretrained weights
        >>> model = hub.load('mpariente/asteroid', 'dprnn_tasnet')
        >>> # Use pretrained weights
        >>> URL = "TOCOME"
        >>> model = hub.load('mpariente/asteroid', 'dprnn_tasnet', URL)
    """
    # No pretrained weights
    if name_url_or_file is None:
        return models.DPRNNTasNet(*args, **kwargs)
    return models.DPRNNTasNet.from_pretrained(name_url_or_file)
