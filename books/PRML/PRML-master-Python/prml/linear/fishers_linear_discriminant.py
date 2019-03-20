import numpy as np
from prml.linear.classifier import Classifier
from prml.rv.gaussian import Gaussian


class FishersLinearDiscriminant(Classifier):
    """
    Fisher's Linear discriminant model
    """

    def __init__(self, w:np.ndarray=None, threshold:float=None):
        self.w = w
        self.threshold = threshold

    def fit(self, X:np.ndarray, t:np.ndarray):
        """
        estimate parameter given training dataset

        Parameters
        ----------
        X : (N, D) np.ndarray
            training dataset independent variable
        t : (N,) np.ndarray
            training dataset dependent variable
            binary 0 or 1
        """
        X0 = X[t == 0]
        X1 = X[t == 1]
        m0 = np.mean(X0, axis=0)
        m1 = np.mean(X1, axis=0)
        cov_inclass = np.cov(X0, rowvar=False) + np.cov(X1, rowvar=False)
        self.w = np.linalg.solve(cov_inclass, m1 - m0)
        self.w /= np.linalg.norm(self.w).clip(min=1e-10)

        g0 = Gaussian()
        g0.fit((X0 @ self.w))
        g1 = Gaussian()
        g1.fit((X1 @ self.w))
        root = np.roots([
            g1.var - g0.var,
            2 * (g0.var * g1.mu - g1.var * g0.mu),
            g1.var * g0.mu ** 2 - g0.var * g1.mu ** 2
            - g1.var * g0.var * np.log(g1.var / g0.var)
        ])
        if g0.mu < root[0] < g1.mu or g1.mu < root[0] < g0.mu:
            self.threshold = root[0]
        else:
            self.threshold = root[1]

    def transform(self, X:np.ndarray):
        """
        project data

        Parameters
        ----------
        X : (N, D) np.ndarray
            independent variable

        Returns
        -------
        y : (N,) np.ndarray
            projected data
        """
        return X @ self.w

    def classify(self, X:np.ndarray):
        """
        classify input data

        Parameters
        ----------
        X : (N, D) np.ndarray
            independent variable to be classified

        Returns
        -------
        (N,) np.ndarray
            binary class for each input
        """
        return (X @ self.w > self.threshold).astype(np.int)
