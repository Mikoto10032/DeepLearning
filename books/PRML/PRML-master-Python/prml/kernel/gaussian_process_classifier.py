import numpy as np


class GaussianProcessClassifier(object):

    def __init__(self, kernel, noise_level=1e-4):
        """
        construct gaussian process classifier

        Parameters
        ----------
        kernel
            kernel function to be used to compute Gram matrix
        noise_level : float
            parameter to ensure the matrix to be positive
        """
        self.kernel = kernel
        self.noise_level = noise_level

    def _sigmoid(self, a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def fit(self, X, t):
        if X.ndim == 1:
            X = X[:, None]
        self.X = X
        self.t = t
        Gram = self.kernel(X, X)
        self.covariance = Gram + np.eye(len(Gram)) * self.noise_level
        self.precision = np.linalg.inv(self.covariance)

    def predict(self, X):
        if X.ndim == 1:
            X = X[:, None]
        K = self.kernel(X, self.X)
        a_mean = K @ self.precision @ self.t
        return self._sigmoid(a_mean)
