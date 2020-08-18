from asteroid import models

dependencies = ["torch"]


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


def lstm_tasnet(name_url_or_file=None, *args, **kwargs):
    """ Load (pretrained) LSTM TasNet model

    Args:
        name_url_or_file (str): Model name (we'll find the URL),
            model URL to download model, path to model file.
            If None (default), LSTMTasNet is instantiated but no pretrained
            weights are loaded.
        *args: Arguments to pass to LSTMTasNet.
        **kwargs: Keyword arguments to pass to LSTMTasNet.

    Returns:
        LSTMTasNet instance (with ot without pretrained weights).

    Examples:
        >>> from torch import hub
        >>> # Instantiate without pretrained weights
        >>> model = hub.load('mpariente/asteroid', 'lstm_tasnet')
        >>> # Use pretrained weights
        >>> URL = "TOCOME"
        >>> model = hub.load('mpariente/asteroid', 'lstm_tasnet', URL)
    """
    # No pretrained weights
    if name_url_or_file is None:
        return models.LSTMTasNet(*args, **kwargs)
    return models.LSTMTasNet.from_pretrained(name_url_or_file)


def dpt_net(name_url_or_file=None, *args, **kwargs):
    """ Load (pretrained) DualPathTransformer (DPTNet) model

    Args:
        name_url_or_file (str): Model name (we'll find the URL),
            model URL to download model, path to model file.
            If None (default), DPTNet is instantiated but no pretrained
            weights are loaded.
        *args: Arguments to pass to DPTNet.
        **kwargs: Keyword arguments to pass to DPTNet.

    Returns:
        DPTNet instance (with ot without pretrained weights).

    Examples:
        >>> from torch import hub
        >>> # Instantiate without pretrained weights
        >>> model = hub.load('mpariente/asteroid', 'dpt_net')
        >>> # Use pretrained weights
        >>> URL = "TOCOME"
        >>> model = hub.load('mpariente/asteroid', 'dpt_net', URL)
    """
    # No pretrained weights
    if name_url_or_file is None:
        return models.DPTNet(*args, **kwargs)
    return models.DPTNet.from_pretrained(name_url_or_file)


def sudormrf_net(name_url_or_file=None, *args, **kwargs):
    """ Load (pretrained) SuDORMRF model.

    Args:
        name_url_or_file (str): Model name (we'll find the URL),
            model URL to download model, path to model file.
            If None (default), SuDORMRFNet is instantiated but no pretrained
            weights are loaded.
        *args: Arguments to pass to SuDORMRFNet.
        **kwargs: Keyword arguments to pass to SuDORMRFNet.

    Returns:
        SuDORMRF instance (with ot without pretrained weights).

    Examples:
        >>> from torch import hub
        >>> # Instantiate without pretrained weights
        >>> model = hub.load('mpariente/asteroid', 'sudormrf_net')
        >>> # Use pretrained weights
        >>> URL = "TOCOME"
        >>> model = hub.load('mpariente/asteroid', 'sudormrf_net', URL)
    """
    # No pretrained weights
    if name_url_or_file is None:
        return models.SuDORMRFNet(*args, **kwargs)
    return models.SuDORMRFNet.from_pretrained(name_url_or_file)


def sudormrf_improved_net(name_url_or_file=None, *args, **kwargs):
    """ Load (pretrained) SuDORMRFImprovedNet improved model

    Args:
        name_url_or_file (str): Model name (we'll find the URL),
            model URL to download model, path to model file.
            If None (default), SuDORMRFImprovedNet is instantiated but no pretrained
            weights are loaded.
        *args: Arguments to pass to SuDORMRFImprovedNet.
        **kwargs: Keyword arguments to pass to SuDORMRFImprovedNet.

    Returns:
        SuDORMRFImprovedNet instance (with ot without pretrained weights).

    Examples:
        >>> from torch import hub
        >>> # Instantiate without pretrained weights
        >>> model = hub.load('mpariente/asteroid', 'lstm_tasnet')
        >>> # Use pretrained weights
        >>> URL = "TOCOME"
        >>> model = hub.load('mpariente/asteroid', 'lstm_tasnet', URL)
    """
    # No pretrained weights
    if name_url_or_file is None:
        return models.SuDORMRFImprovedNet(*args, **kwargs)
    return models.SuDORMRFImprovedNet.from_pretrained(name_url_or_file)
