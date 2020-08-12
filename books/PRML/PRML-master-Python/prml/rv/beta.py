import numpy as np
from scipy.special import gamma
from prml.rv.rv import RandomVariable


np.seterr(all="ignore")


class Beta(RandomVariable):
    """
    Beta distribution
    p(mu|n_ones, n_zeros)
    = gamma(n_ones + n_zeros)
      * mu^(n_ones - 1) * (1 - mu)^(n_zeros - 1)
      / gamma(n_ones) / gamma(n_zeros)
    """

    def __init__(self, n_zeros, n_ones):
        """
        construct beta distribution

        Parameters
        ----------
        n_zeros : int, float, or np.ndarray
            pseudo count of zeros
        n_ones : int, float, or np.ndarray
            pseudo count of ones
        """
        super().__init__()
        if not isinstance(n_ones, (int, float, np.number, np.ndarray)):
            raise ValueError(
                "{} is not supported for n_ones"
                .format(type(n_ones))
            )
        if not isinstance(n_zeros, (int, float, np.number, np.ndarray)):
            raise ValueError(
                "{} is not supported for n_zeros"
                .format(type(n_zeros))
            )
        n_ones = np.asarray(n_ones)
        n_zeros = np.asarray(n_zeros)
        if n_ones.shape != n_zeros.shape:
            raise ValueError(
                "the sizes of the arrays don't match: {}, {}"
                .format(n_ones.shape, n_zeros.shape)
            )
        self.n_ones = n_ones
        self.n_zeros = n_zeros

    @property
    def ndim(self):
        return self.n_ones.ndim

    @property
    def size(self):
        return self.n_ones.size

    @property
    def shape(self):
        return self.n_ones.shape

    def _pdf(self, mu):
        return (
            gamma(self.n_ones + self.n_zeros)
            * np.power(mu, self.n_ones - 1)
            * np.power(1 - mu, self.n_zeros - 1)
            / gamma(self.n_ones)
            / gamma(self.n_zeros)
        )

    def _draw(self, sample_size=1):
        return np.random.beta(
            self.n_ones, self.n_zeros, size=(sample_size,) + self.shape
        )
