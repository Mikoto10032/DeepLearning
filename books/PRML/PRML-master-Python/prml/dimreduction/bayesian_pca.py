import numpy as np
from prml.dimreduction.pca import PCA


class BayesianPCA(PCA):

    def fit(self, X, iter_max=100, initial="random"):
        """
        empirical bayes estimation of pca parameters

        Parameters
        ----------
        X : (sample_size, n_features) ndarray
            input data
        iter_max : int
            maximum number of em steps

        Returns
        -------
        mean : (n_features,) ndarray
            sample mean fo the input data
        W : (n_features, n_components) ndarray
            projection matrix
        var : float
            variance of observation noise
        """
        initial_list = ["random", "eigen"]
        self.mean = np.mean(X, axis=0)
        self.I = np.eye(self.n_components)
        if initial not in initial_list:
            print("availabel initializations are {}".format(initial_list))
        if initial == "random":
            self.W = np.eye(np.size(X, 1), self.n_components)
            self.var = 1.
        elif initial == "eigen":
            self.eigen(X)
        self.alpha = len(self.mean) / np.sum(self.W ** 2, axis=0).clip(min=1e-10)
        for i in range(iter_max):
            W = np.copy(self.W)
            stats = self._expectation(X - self.mean)
            self._maximization(X - self.mean, *stats)
            self.alpha = len(self.mean) / np.sum(self.W ** 2, axis=0).clip(min=1e-10)
            if np.allclose(W, self.W):
                break
        self.n_iter = i + 1

    def _maximization(self, X, Ez, Ezz):
        self.W = X.T @ Ez @ np.linalg.inv(np.sum(Ezz, axis=0) + self.var * np.diag(self.alpha))
        self.var = np.mean(
            np.mean(X ** 2, axis=-1)
            - 2 * np.mean(Ez @ self.W.T * X, axis=-1)
            + np.trace((Ezz @ self.W.T @ self.W).T) / len(self.mean))

    def maximize(self, D, Ez, Ezz):
        self.W = D.T.dot(Ez).dot(np.linalg.inv(np.sum(Ezz, axis=0) + self.var * np.diag(self.alpha)))
        self.var = np.mean(
            np.mean(D ** 2, axis=-1)
            - 2 * np.mean(Ez.dot(self.W.T) * D, axis=-1)
            + np.trace(Ezz.dot(self.W.T).dot(self.W).T) / self.ndim)
