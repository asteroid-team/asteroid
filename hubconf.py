from asteroid import models

dependencies = ["torch", "numpy", "scipy", "asteroid_filterbanks", "requests", "filelock"]


def demask(name_url_or_file=None, **kwargs):
    """Load (pretrained) DeMask model

    Args:
        name_url_or_file (str): Model name (we'll find the URL),
            model URL to download model, path to model file.
            If None (default), DeMask is instantiated but no pretrained
            weights are loaded.
        **kwargs: Keyword arguments to pass to DeMask.

    Returns:
        DeMask instance (with ot without pretrained weights).

    Examples:
        >>> from torch import hub
        >>> # Instantiate without pretrained weights
        >>> model = hub.load('asteroid-team/asteroid', 'demask')
        >>> # Use pretrained weights
        >>> URL = "popcornell/DeMask_Surgical_mask_speech_enhancement_v1"
        >>> model = hub.load('asteroid-team/asteroid', 'demask', URL)
    """
    # No pretrained weights
    if name_url_or_file is None:
        return models.DeMask(**kwargs)
    return models.DeMask.from_pretrained(name_url_or_file, **kwargs)


def conv_tasnet(name_url_or_file=None, **kwargs):
    """Load (pretrained) ConvTasNet model

    Args:
        name_url_or_file (str): Model name (we'll find the URL),
            model URL to download model, path to model file.
            If None (default), ConvTasNet is instantiated but no pretrained
            weights are loaded.
        **kwargs: Keyword arguments to pass to ConvTasNet.

    Returns:
        ConvTasNet instance (with ot without pretrained weights).

    Examples:
        >>> from torch import hub
        >>> # Instantiate without pretrained weights
        >>> model = hub.load('asteroid-team/asteroid', 'conv_tasnet', n_src=2)
        >>> # Use pretrained weights
        >>> URL = "TOCOME"
        >>> model = hub.load('asteroid-team/asteroid', 'conv_tasnet', URL)
    """
    # No pretrained weights
    if name_url_or_file is None:
        return models.ConvTasNet(**kwargs)
    return models.ConvTasNet.from_pretrained(name_url_or_file, **kwargs)


def dprnn_tasnet(name_url_or_file=None, **kwargs):
    """Load (pretrained) DPRNNTasNet model

    Args:
        name_url_or_file (str): Model name (we'll find the URL),
            model URL to download model, path to model file.
            If None (default), DPRNNTasNet is instantiated but no pretrained
            weights are loaded.
        **kwargs: Keyword arguments to pass to DPRNNTasNet.

    Returns:
        DPRNNTasNet instance (with ot without pretrained weights).

    Examples:
        >>> from torch import hub
        >>> # Instantiate without pretrained weights
        >>> model = hub.load('asteroid-team/asteroid', 'dprnn_tasnet')
        >>> # Use pretrained weights
        >>> URL = "TOCOME"
        >>> model = hub.load('asteroid-team/asteroid', 'dprnn_tasnet', URL)
    """
    # No pretrained weights
    if name_url_or_file is None:
        return models.DPRNNTasNet(**kwargs)
    return models.DPRNNTasNet.from_pretrained(name_url_or_file, **kwargs)


def lstm_tasnet(name_url_or_file=None, **kwargs):
    """Load (pretrained) LSTM TasNet model

    Args:
        name_url_or_file (str): Model name (we'll find the URL),
            model URL to download model, path to model file.
            If None (default), LSTMTasNet is instantiated but no pretrained
            weights are loaded.
        **kwargs: Keyword arguments to pass to LSTMTasNet.

    Returns:
        LSTMTasNet instance (with ot without pretrained weights).

    Examples:
        >>> from torch import hub
        >>> # Instantiate without pretrained weights
        >>> model = hub.load('asteroid-team/asteroid', 'lstm_tasnet')
        >>> # Use pretrained weights
        >>> URL = "TOCOME"
        >>> model = hub.load('asteroid-team/asteroid', 'lstm_tasnet', URL)
    """
    # No pretrained weights
    if name_url_or_file is None:
        return models.LSTMTasNet(**kwargs)
    return models.LSTMTasNet.from_pretrained(name_url_or_file, **kwargs)


def dpt_net(name_url_or_file=None, **kwargs):
    """Load (pretrained) DualPathTransformer (DPTNet) model

    Args:
        name_url_or_file (str): Model name (we'll find the URL),
            model URL to download model, path to model file.
            If None (default), DPTNet is instantiated but no pretrained
            weights are loaded.
        **kwargs: Keyword arguments to pass to DPTNet.

    Returns:
        DPTNet instance (with ot without pretrained weights).

    Examples:
        >>> from torch import hub
        >>> # Instantiate without pretrained weights
        >>> model = hub.load('asteroid-team/asteroid', 'dpt_net')
        >>> # Use pretrained weights
        >>> URL = "TOCOME"
        >>> model = hub.load('asteroid-team/asteroid', 'dpt_net', URL)
    """
    # No pretrained weights
    if name_url_or_file is None:
        return models.DPTNet(**kwargs)
    return models.DPTNet.from_pretrained(name_url_or_file, **kwargs)


def sudormrf_net(name_url_or_file=None, **kwargs):
    """Load (pretrained) SuDORMRF model.

    Args:
        name_url_or_file (str): Model name (we'll find the URL),
            model URL to download model, path to model file.
            If None (default), SuDORMRFNet is instantiated but no pretrained
            weights are loaded.
        **kwargs: Keyword arguments to pass to SuDORMRFNet.

    Returns:
        SuDORMRF instance (with ot without pretrained weights).

    Examples:
        >>> from torch import hub
        >>> # Instantiate without pretrained weights
        >>> model = hub.load('asteroid-team/asteroid', 'sudormrf_net')
        >>> # Use pretrained weights
        >>> URL = "TOCOME"
        >>> model = hub.load('asteroid-team/asteroid', 'sudormrf_net', URL)
    """
    # No pretrained weights
    if name_url_or_file is None:
        return models.SuDORMRFNet(**kwargs)
    return models.SuDORMRFNet.from_pretrained(name_url_or_file, **kwargs)


def sudormrf_improved_net(name_url_or_file=None, **kwargs):
    """Load (pretrained) SuDORMRFImprovedNet improved model

    Args:
        name_url_or_file (str): Model name (we'll find the URL),
            model URL to download model, path to model file.
            If None (default), SuDORMRFImprovedNet is instantiated but no pretrained
            weights are loaded.
        **kwargs: Keyword arguments to pass to SuDORMRFImprovedNet.

    Returns:
        SuDORMRFImprovedNet instance (with ot without pretrained weights).

    Examples:
        >>> from torch import hub
        >>> # Instantiate without pretrained weights
        >>> model = hub.load('asteroid-team/asteroid', 'lstm_tasnet')
        >>> # Use pretrained weights
        >>> URL = "TOCOME"
        >>> model = hub.load('asteroid-team/asteroid', 'lstm_tasnet', URL)
    """
    # No pretrained weights
    if name_url_or_file is None:
        return models.SuDORMRFImprovedNet(**kwargs)
    return models.SuDORMRFImprovedNet.from_pretrained(name_url_or_file, **kwargs)
