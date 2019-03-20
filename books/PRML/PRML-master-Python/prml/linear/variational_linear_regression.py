import numpy as np
from prml.linear.regression import Regression


class VariationalLinearRegression(Regression):
    """
    variational bayesian estimation of linear regression model
    p(w,alpha|X,t)
    ~ q(w)q(alpha)
    = N(w|w_mean, w_var)Gamma(alpha|a,b)

    Attributes
    ----------
    a : float
        a parameter of variational posterior gamma distribution
    b : float
        another parameter of variational posterior gamma distribution
    w_mean : (n_features,) ndarray
        mean of variational posterior gaussian distribution
    w_var : (n_features, n_feautures) ndarray
        variance of variational posterior gaussian distribution
    n_iter : int
        number of iterations performed
    """

    def __init__(self, beta:float=1., a0:float=1., b0:float=1.):
        """
        construct variational linear regressor
        Parameters
        ----------
        beta : float
            precision of observation noise
        a0 : float
            a parameter of prior gamma distribution
            Gamma(alpha|a0,b0)
        b0 : float
            another parameter of prior gamma distribution
            Gamma(alpha|a0,b0)
        """
        self.beta = beta
        self.a0 = a0
        self.b0 = b0

    def fit(self, X:np.ndarray, t:np.ndarray, iter_max:int=100):
        """
        variational bayesian estimation of parameter

        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        t : (N,) np.ndarray
            training dependent variable
        iter_max : int, optional
            maximum number of iteration (the default is 100)
        """
        D = np.size(X, 1)
        self.a = self.a0 + 0.5 * D
        self.b = self.b0
        I = np.eye(D)
        for _ in range(iter_max):
            param = self.b
            self.w_var = np.linalg.inv(self.a * I / self.b + self.beta * X.T @ X)
            self.w_mean = self.beta * self.w_var @ X.T @ t
            self.b = self.b0 + 0.5 * (np.sum(self.w_mean ** 2) + np.trace(self.w_var))
            if np.allclose(self.b, param):
                break

    def predict(self, X:np.ndarray, return_std:bool=False):
        """
        make prediction of input

        Parameters
        ----------
        X : (N, D) np.ndarray
            independent variable
        return_std : bool, optional
            return standard deviation of predictive distribution if True
            (the default is False)

        Returns
        -------
        y : (N,) np.ndarray
            mean of predictive distribution
        y_std : (N,) np.ndarray
            standard deviation of predictive distribution
        """
        y = X @ self.w_mean
        if return_std:
            y_var = 1 / self.beta + np.sum(X @ self.w_var * X, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y
