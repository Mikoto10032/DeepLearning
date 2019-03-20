import numpy as np
from prml.nn.array.broadcast import broadcast_to
from prml.nn.linalg.cholesky import cholesky
from prml.nn.linalg.det import det
from prml.nn.linalg.logdet import logdet
from prml.nn.linalg.solve import solve
from prml.nn.linalg.trace import trace
from prml.nn.math.exp import exp
from prml.nn.math.log import log
from prml.nn.math.sqrt import sqrt
from prml.nn.random.random import RandomVariable
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor


class MultivariateGaussian(RandomVariable):
    """
    Multivariate Gaussian distribution
    p(x|mu, cov)
    = exp{-0.5 * (x - mu)^T cov^-1 (x - mu)} * (1 / 2pi) ** (d / 2) * |cov^-1| ** 0.5
    where d = dimensionality
    Parameters
    ----------
    mu : (d,) tensor_like
        mean parameter
    cov : (d, d) tensor_like
        variance-covariance matrix
    data : (..., d) tensor_like
        observed data
    p : RandomVariable
        original distribution of a model
    """

    def __init__(self, mu, cov, data=None, p=None):
        super().__init__(data, p)
        self.mu, self.cov = self._check_input(mu, cov)

    def _check_input(self, mu, cov):
        mu = self._convert2tensor(mu)
        cov = self._convert2tensor(cov)
        self._equal_ndim(mu, 1)
        self._equal_ndim(cov, 2)
        if cov.shape != (mu.size, mu.size):
            raise ValueError("Mismatching dimensionality of mu and cov")
        return mu, cov

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        self.parameter["mu"] = mu

    @property
    def cov(self):
        return self.parameter["cov"]

    @cov.setter
    def cov(self, cov):
        try:
            self.L = cholesky(cov)
        except np.linalg.LinAlgError:
            raise ValueError("cov must be positive-difinite matrix")
        self.parameter["cov"] = cov

    def forward(self):
        self.eps = np.random.normal(size=self.mu.size)
        output = self.mu.value + self.L.value @ self.eps
        if isinstance(self.mu, Constant) and isinstance(self.cov, Constant):
            return Constant(output)
        return Tensor(output, self)

    def backward(self, delta):
        dmu = delta
        dL = delta * self.eps[:, None]
        self.mu.backward(dmu)
        self.L.backward(dL)

    def _pdf(self, x):
        assert x.shape[-1] == self.mu.size
        if x.ndim == 1:
            squeeze = True
            x = broadcast_to(x, (1, self.mu.size))
        else:
            squeeze = False
        assert x.ndim == 2
        d = x - self.mu
        d = d.transpose()
        p = (
            exp(-0.5 * (solve(self.cov, d) * d).sum(axis=0))
            / (2 * np.pi) ** (self.mu.size * 0.5)
            / sqrt(det(self.cov))
        )
        if squeeze:
            p = p.sum()

        return p

    def _log_pdf(self, x):
        assert x.shape[-1] == self.mu.size
        if x.ndim == 1:
            squeeze = True
            x = broadcast_to(x, (1, self.mu.size))
        else:
            squeeze = False
        assert x.ndim == 2
        d = x - self.mu
        d = d.transpose()

        logp = (
            -0.5 * (solve(self.cov, d) * d).sum(axis=0)
            - (self.mu.size * 0.5) * log(2 * np.pi)
            - 0.5 * logdet(self.cov)
        )
        if squeeze:
            logp = logp.sum()

        return logp
