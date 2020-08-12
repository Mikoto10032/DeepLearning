import numpy as np
from prml.linear.logistic_regression import LogisticRegression


class VariationalLogisticRegression(LogisticRegression):

    def __init__(self, alpha:float=None, a0:float=1., b0:float=1.):
        """
        construct variational logistic regressor

        Parameters
        ----------
        alpha : float
            precision parameter of the prior
            if None, this is also the subject to estimate
        a0 : float
            a parameter of hyper prior Gamma dist.
            Gamma(alpha|a0,b0)
            if alpha is not None, this argument will be ignored
        b0 : float
            another parameter of hyper prior Gamma dist.
            Gamma(alpha|a0,b0)
            if alpha is not None, this argument will be ignored
        """
        if alpha is not None:
            self.__alpha = alpha
        else:
            self.a0 = a0
            self.b0 = b0

    def fit(self, X:np.ndarray, t:np.ndarray, iter_max:int=1000):
        """
        variational bayesian estimation of the parameter

        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        t : (N,) np.ndarray
            training dependent variable
        iter_max : int, optional
            maximum number of iteration (the default is 1000)
        """
        N, D = X.shape
        if hasattr(self, "a0"):
            self.a = self.a0 + 0.5 * D
        xi = np.random.uniform(-1, 1, size=N)
        I = np.eye(D)
        param = np.copy(xi)
        for _ in range(iter_max):
            lambda_ = np.tanh(xi) * 0.25 / xi
            self.w_var = np.linalg.inv(I / self.alpha + 2 * (lambda_ * X.T) @ X)
            self.w_mean = self.w_var @ np.sum(X.T * (t - 0.5), axis=1)
            xi = np.sqrt(np.sum(X @ (self.w_var + self.w_mean * self.w_mean[:, None]) * X, axis=-1))
            if np.allclose(xi, param):
                break
            else:
                param = np.copy(xi)

    @property
    def alpha(self):
        if hasattr(self, "__alpha"):
            return self.__alpha
        else:
            try:
                self.b = self.b0 + 0.5 * (np.sum(self.w_mean ** 2) + np.trace(self.w_var))
            except AttributeError:
                self.b = self.b0
            return self.a / self.b

    def proba(self, X:np.ndarray):
        """
        compute probability of input belonging class 1

        Parameters
        ----------
        X : (N, D) np.ndarray
            training data independent variable

        Returns
        -------
        (N,) np.ndarray
            probability of positive
        """
        mu_a = X @ self.w_mean
        var_a = np.sum(X @ self.w_var * X, axis=1)
        y = self._sigmoid(mu_a / np.sqrt(1 + np.pi * var_a / 8))
        return y
