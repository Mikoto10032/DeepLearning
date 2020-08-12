import numpy as np
from prml.linear.bayesian_regression import BayesianRegression


class EmpiricalBayesRegression(BayesianRegression):
    """
    Empirical Bayes Regression model
    a.k.a.
    type 2 maximum likelihood,
    generalized maximum likelihood,
    evidence approximation

    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1))
    evidence function p(t|X,alpha,beta) = S p(t|w;X,beta)p(w|0;alpha) dw
    """

    def __init__(self, alpha:float=1., beta:float=1.):
        super().__init__(alpha, beta)

    def fit(self, X:np.ndarray, t:np.ndarray, max_iter:int=100):
        """
        maximization of evidence function with respect to
        the hyperparameters alpha and beta given training dataset

        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        t : (N,) np.ndarray
            training dependent variable
        max_iter : int
            maximum number of iteration
        """
        M = X.T @ X
        eigenvalues = np.linalg.eigvalsh(M)
        eye = np.eye(np.size(X, 1))
        N = len(t)
        for _ in range(max_iter):
            params = [self.alpha, self.beta]

            w_precision = self.alpha * eye + self.beta * X.T @ X
            w_mean = self.beta * np.linalg.solve(w_precision, X.T @ t)

            gamma = np.sum(eigenvalues / (self.alpha + eigenvalues))
            self.alpha = float(gamma / np.sum(w_mean ** 2).clip(min=1e-10))
            self.beta = float(
                (N - gamma) / np.sum(np.square(t - X @ w_mean))
            )
            if np.allclose(params, [self.alpha, self.beta]):
                break
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(w_precision)

    def _log_prior(self, w):
        return -0.5 * self.alpha * np.sum(w ** 2)

    def _log_likelihood(self, X, t, w):
        return -0.5 * self.beta * np.square(t - X @ w).sum()

    def _log_posterior(self, X, t, w):
        return self._log_likelihood(X, t, w) + self._log_prior(w)

    def log_evidence(self, X:np.ndarray, t:np.ndarray):
        """
        logarithm or the evidence function

        Parameters
        ----------
        X : (N, D) np.ndarray
            indenpendent variable
        t : (N,) np.ndarray
            dependent variable
        Returns
        -------
        float
            log evidence
        """
        N = len(t)
        D = np.size(X, 1)
        return 0.5 * (
            D * np.log(self.alpha) + N * np.log(self.beta)
            - np.linalg.slogdet(self.w_precision)[1] - D * np.log(2 * np.pi)
        ) + self._log_posterior(X, t, self.w_mean)
