import numpy as np
from scipy.special import gamma, digamma
from prml.rv.rv import RandomVariable


class StudentsT(RandomVariable):
    """
    Student's t-distribution
    p(x|mu, tau, dof)
    = (1 + tau * (x - mu)^2 / dof)^-(D + dof)/2 / const.
    """

    def __init__(self, mu=None, tau=None, dof=None):
        super().__init__()
        self.mu = mu
        self.tau = tau
        self.dof = dof

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        if isinstance(mu, (int, float, np.number)):
            self.parameter["mu"] = np.array(mu)
        elif isinstance(mu, np.ndarray):
            self.parameter["mu"] = mu
        else:
            assert mu is None
            self.parameter["mu"] = None

    @property
    def tau(self):
        return self.parameter["tau"]

    @tau.setter
    def tau(self, tau):
        if isinstance(tau, (int, float, np.number)):
            tau = np.array(tau)
            assert tau.shape == self.shape
            self.parameter["tau"] = tau
        elif isinstance(tau, np.ndarray):
            assert tau.shape == self.shape
            self.parameter["tau"] = tau
        else:
            assert tau is None
            self.parameter["tau"] = None

    @property
    def dof(self):
        return self.parameter["dof"]

    @dof.setter
    def dof(self, dof):
        if isinstance(dof, (int, float, np.number)):
            self.parameter["dof"] = dof
        else:
            assert dof is None
            self.parameter["dof"] = None

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

    def _fit(self, X, learning_rate=0.01):
        self.mu = np.mean(X, axis=0)
        self.tau = 1 / np.var(X, axis=0)
        self.dof = 1
        params = np.hstack(
            (self.mu.ravel(),
             self.tau.ravel(),
             self.dof)
        )
        while True:
            E_eta, E_lneta = self._expectation(X)
            self._maximization(X, E_eta, E_lneta, learning_rate)
            new_params = np.hstack(
                (self.mu.ravel(),
                 self.tau.ravel(),
                 self.dof)
            )
            if np.allclose(params, new_params):
                break
            else:
                params = new_params

    def _expectation(self, X):
        d = X - self.mu
        a = 0.5 * (self.dof + 1)
        b = 0.5 * (self.dof + self.tau * d ** 2)
        E_eta = a / b
        E_lneta = digamma(a) - np.log(b)
        return E_eta, E_lneta

    def _maximization(self, X, E_eta, E_lneta, learning_rate):
        self.mu = np.sum(E_eta * X, axis=0) / np.sum(E_eta, axis=0)
        d = X - self.mu
        self.tau = 1 / np.mean(E_eta * d ** 2, axis=0)
        N = len(X)
        self.dof += learning_rate * 0.5 * (
            N * np.log(0.5 * self.dof) + N
            - N * digamma(0.5 * self.dof)
            + np.sum(E_lneta - E_eta, axis=0)
        )

    def _pdf(self, X):
        d = X - self.mu
        D_sq = self.tau * d ** 2
        return (
            gamma(0.5 * (self.dof + 1))
            * self.tau ** 0.5
            * (1 + D_sq / self.dof) ** (-0.5 * (1 + self.dof))
            / gamma(self.dof * 0.5)
            / (np.pi * self.dof) ** 0.5
        )
