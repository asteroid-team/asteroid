import numpy as np


def normalize_estimates(est_np, mix_np):
    """ Normalizes estimates according to the mixture maximum amplitude

    Args:
        est_np (numpy array): Estimates (nb_estimates, time)
        mix_np (numpy array): One mixture


    """
    mix_max = np.max(np.abs(mix_np))
    return np.stack([est * mix_max / np.max(np.abs(est)) for est in est_np], dim=0)
