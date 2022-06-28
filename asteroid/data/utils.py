import torch
from torch.utils.data._utils.collate import default_collate


def online_mixing_collate(batch):
    """Mix target sources to create new mixtures.
    Output of the default collate function is expected to return two objects:
    inputs and targets.
    """
    # Inputs (batch, time) / targets (batch, n_src, time)
    inputs, targets = default_collate(batch)
    batch, n_src, _ = targets.shape

    energies = torch.sum(targets**2, dim=-1, keepdim=True)
    new_src = []
    for i in range(targets.shape[1]):
        new_s = targets[torch.randperm(batch), i, :]
        new_s = new_s * torch.sqrt(energies[:, i] / (new_s**2).sum(-1, keepdims=True))
        new_src.append(new_s)

    targets = torch.stack(new_src, dim=1)
    inputs = targets.sum(1)
    return inputs, targets
