import numpy as np
from prml.nn.math.gamma import gamma
from prml.nn.math.log import log
from prml.nn.math.product import prod
from prml.nn.math.sum import sum
from prml.nn.random.random import RandomVariable
from prml.nn.tensor.tensor import Tensor


class Dirichlet(RandomVariable):
    """
    Dirichlet distribution
    Parameters
    ----------
    alpha : (..., K) tensor_like
        pseudo-count of each outcome
    axis : int
        axis along which represents each outcome
    data : tensor_like
        realization
    p : RandomVariable
        original distribution of a model
    Attributes
    ----------
    n_category : int
        number of categories
    """

    def __init__(self, alpha, axis=-1, data=None, p=None):
        super().__init__(data, p)
        assert axis == -1
        self.axis = axis
        self.alpha = self._convert2tensor(alpha)

    @property
    def alpha(self):
        return self.parameter["alpha"]

    @alpha.setter
    def alpha(self, alpha):
        self._atleast_ndim(alpha, 1)
        if (alpha.value <= 0).any():
            raise ValueError("alpha must all be positive")
        self.parameter["alpha"] = alpha

    def forward(self):
        if self.alpha.ndim == 1:
            return Tensor(np.random.dirichlet(self.alpha.value), function=self)
        else:
            raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def _pdf(self, x):
        return (
            gamma(self.alpha.sum(axis=self.axis))
            * prod(
                x ** (self.alpha - 1)
                / gamma(self.alpha),
                axis=self.axis
            )
        )

    def _log_pdf(self, x):
        return (
            log(gamma(self.alpha.sum(axis=self.axis)))
            + sum(
                (self.alpha - 1) * log(x) - log(gamma(self.alpha)),
                axis=self.axis
            )
        )
