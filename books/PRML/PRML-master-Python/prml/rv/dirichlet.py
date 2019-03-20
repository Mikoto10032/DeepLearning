import numpy as np
from scipy.special import gamma
from prml.rv.rv import RandomVariable


class Dirichlet(RandomVariable):
    """
    Dirichlet distribution
    p(mu|alpha)
    = gamma(sum(alpha))
      * prod_k mu_k ^ (alpha_k - 1)
      / gamma(alpha_1) / ... / gamma(alpha_K)
    """

    def __init__(self, alpha):
        """
        construct dirichlet distribution

        Parameters
        ----------
        alpha : (size,) np.ndarray
            pseudo count of each outcome, aka concentration parameter
        """
        super().__init__()
        self.alpha = alpha

    @property
    def alpha(self):
        return self.parameter["alpha"]

    @alpha.setter
    def alpha(self, alpha):
        assert isinstance(alpha, np.ndarray)
        assert alpha.ndim == 1
        assert (alpha >= 0).all()
        self.parameter["alpha"] = alpha

    @property
    def ndim(self):
        return self.alpha.ndim

    @property
    def size(self):
        return self.alpha.size

    @property
    def shape(self):
        return self.alpha.shape

    def _pdf(self, mu):
        return (
            gamma(self.alpha.sum())
            * np.prod(mu ** (self.alpha - 1), axis=-1)
            / np.prod(gamma(self.alpha), axis=-1)
        )

    def _draw(self, sample_size=1):
        return np.random.dirichlet(self.alpha, sample_size)
