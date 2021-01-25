import numpy as np


def declipp(mix_np, est_np):
    """

    Args:
        mix_np (numpy array): One mixture
        est_np (numpy array): Estimates (nb_estimates, time)

    """
    for i in range(len(est_np)):
        est_np[i] *= np.max(np.abs(mix_np)) / np.max(np.abs(est_np[i]))
    return est_np
