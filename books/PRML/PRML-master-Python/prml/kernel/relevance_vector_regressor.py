import numpy as np


class RelevanceVectorRegressor(object):

    def __init__(self, kernel, alpha=1., beta=1.):
        """
        construct relevance vector regressor

        Parameters
        ----------
        kernel : Kernel
            kernel function to compute components of feature vectors
        alpha : float
            initial precision of prior weight distribution
        beta : float
            precision of observation
        """
        self.kernel = kernel
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, t, iter_max=1000):
        """
        maximize evidence with respect to hyperparameter

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input
        t : (sample_size,) ndarray
            corresponding target
        iter_max : int
            maximum number of iterations

        Attributes
        -------
        X : (N, n_features) ndarray
            relevance vector
        t : (N,) ndarray
            corresponding target
        alpha : (N,) ndarray
            hyperparameter for each weight or training sample
        cov : (N, N) ndarray
            covariance matrix of weight
        mean : (N,) ndarray
            mean of each weight
        """
        if X.ndim == 1:
            X = X[:, None]
        assert X.ndim == 2
        assert t.ndim == 1
        N = len(t)
        Phi = self.kernel(X, X)
        self.alpha = np.zeros(N) + self.alpha
        for _ in range(iter_max):
            params = np.hstack([self.alpha, self.beta])
            precision = np.diag(self.alpha) + self.beta * Phi.T @ Phi
            covariance = np.linalg.inv(precision)
            mean = self.beta * covariance @ Phi.T @ t
            gamma = 1 - self.alpha * np.diag(covariance)
            self.alpha = gamma / np.square(mean)
            np.clip(self.alpha, 0, 1e10, out=self.alpha)
            self.beta = (N - np.sum(gamma)) / np.sum((t - Phi.dot(mean)) ** 2)
            if np.allclose(params, np.hstack([self.alpha, self.beta])):
                break
        mask = self.alpha < 1e9
        self.X = X[mask]
        self.t = t[mask]
        self.alpha = self.alpha[mask]
        Phi = self.kernel(self.X, self.X)
        precision = np.diag(self.alpha) + self.beta * Phi.T @ Phi
        self.covariance = np.linalg.inv(precision)
        self.mean = self.beta * self.covariance @ Phi.T @ self.t

    def predict(self, X, with_error=True):
        """
        predict output with this model

        Parameters
        ----------
        X : (sample_size, n_features)
            input
        with_error : bool
            if True, predict with standard deviation of the outputs

        Returns
        -------
        mean : (sample_size,) ndarray
            mean of predictive distribution
        std : (sample_size,) ndarray
            standard deviation of predictive distribution
        """
        if X.ndim == 1:
            X = X[:, None]
        assert X.ndim == 2
        phi = self.kernel(X, self.X)
        mean = phi @ self.mean
        if with_error:
            var = 1 / self.beta + np.sum(phi @ self.covariance * phi, axis=1)
            return mean, np.sqrt(var)
        return mean
