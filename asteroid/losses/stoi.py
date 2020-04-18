from torch_stoi import NegSTOILoss

asteroid_examples = """
Examples:
    >>> import torch
    >>> from asteroid.losses import PITLossWrapper
    >>> targets = torch.randn(10, 2, 32000)
    >>> est_targets = torch.randn(10, 2, 32000)
    >>> loss_func = PITLossWrapper(NegSTOILoss(), pit_from='pw_pt')
    >>> loss = loss_func(est_targets, targets)
"""

NegSTOILoss.__doc__ += asteroid_examples
