import numpy as np
from prml.rv.rv import RandomVariable
from prml.rv.beta import Beta


class Bernoulli(RandomVariable):
    """
    Bernoulli distribution
    p(x|mu) = mu^x (1 - mu)^(1 - x)
    """

    def __init__(self, mu=None):
        """
        construct Bernoulli distribution

        Parameters
        ----------
        mu : np.ndarray or Beta
            probability of value 1 for each element
        """
        super().__init__()
        self.mu = mu

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        if isinstance(mu, (int, float, np.number)):
            if mu > 1 or mu < 0:
                raise ValueError(f"mu must be in [0, 1], not {mu}")
            self.parameter["mu"] = np.asarray(mu)
        elif isinstance(mu, np.ndarray):
            if (mu > 1).any() or (mu < 0).any():
                raise ValueError("mu must be in [0, 1]")
            self.parameter["mu"] = mu
        elif isinstance(mu, Beta):
            self.parameter["mu"] = mu
        else:
            if mu is not None:
                raise TypeError(f"{type(mu)} is not supported for mu")
            self.parameter["mu"] = None

    @property
    def ndim(self):
        if hasattr(self.mu, "ndim"):
            return self.mu.ndim
        else:
            return None

    @property
    def size(self):
        if hasattr(self.mu, "size"):
            return self.mu.size
        else:
            return None

    @property
    def shape(self):
        if hasattr(self.mu, "shape"):
            return self.mu.shape
        else:
            return None

    def _fit(self, X):
        if isinstance(self.mu, Beta):
            self._bayes(X)
        elif isinstance(self.mu, RandomVariable):
            raise NotImplementedError
        else:
            self._ml(X)

    def _ml(self, X):
        n_zeros = np.count_nonzero((X == 0).astype(np.int))
        n_ones = np.count_nonzero((X == 1).astype(np.int))
        assert X.size == n_zeros + n_ones, (
            "{X.size} is not equal to {n_zeros} plus {n_ones}"
        )
        self.mu = np.mean(X, axis=0)

    def _map(self, X):
        assert isinstance(self.mu, Beta)
        assert X.shape[1:] == self.mu.shape
        n_ones = (X == 1).sum(axis=0)
        n_zeros = (X == 0).sum(axis=0)
        assert X.size == n_zeros.sum() + n_ones.sum(), (
            f"{X.size} is not equal to {n_zeros} plus {n_ones}"
        )
        n_ones = n_ones + self.mu.n_ones
        n_zeros = n_zeros + self.mu.n_zeros
        self.prob = (n_ones - 1) / (n_ones + n_zeros - 2)

    def _bayes(self, X):
        assert isinstance(self.mu, Beta)
        assert X.shape[1:] == self.mu.shape
        n_ones = (X == 1).sum(axis=0)
        n_zeros = (X == 0).sum(axis=0)
        assert X.size == n_zeros.sum() + n_ones.sum(), (
            "input X must only has 0 or 1"
        )
        self.mu.n_zeros += n_zeros
        self.mu.n_ones += n_ones

    def _pdf(self, X):
        assert isinstance(mu, np.ndarray)
        return np.prod(
            self.mu ** X * (1 - self.mu) ** (1 - X)
        )

    def _draw(self, sample_size=1):
        if isinstance(self.mu, np.ndarray):
            return (
                self.mu > np.random.uniform(size=(sample_size,) + self.shape)
            ).astype(np.int)
        elif isinstance(self.mu, Beta):
            return (
                self.mu.n_ones / (self.mu.n_ones + self.mu.n_zeros)
                > np.random.uniform(size=(sample_size,) + self.shape)
            ).astype(np.int)
        elif isinstance(self.mu, RandomVariable):
            return (
                self.mu.draw(sample_size)
                > np.random.uniform(size=(sample_size,) + self.shape)
            )
