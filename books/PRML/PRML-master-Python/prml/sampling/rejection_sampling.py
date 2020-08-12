import random
import numpy as np


def rejection_sampling(func, rv, k, n):
    """
    perform rejection sampling n times

    Parameters
    ----------
    func : callable
        (un)normalized distribution to be sampled from
    rv : RandomVariable
        distribution to generate sample
    k : float
        constant to be multiplied with the distribution
    n : int
        number of samples to draw

    Returns
    -------
    sample : (n, ndim) ndarray
        generated sample
    """
    assert hasattr(rv, "draw"), "the distribution has no method to draw random samples"
    sample = []
    while len(sample) < n:
        sample_candidate = rv.draw()
        accept_proba = func(sample_candidate) / (k * rv.pdf(sample_candidate))
        if random.random() < accept_proba:
            sample.append(sample_candidate[0])
    sample = np.asarray(sample)
    assert sample.shape == (n, rv.ndim), sample.shape
    return sample
