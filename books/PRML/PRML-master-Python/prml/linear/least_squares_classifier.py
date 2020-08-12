import numpy as np
from prml.linear.classifier import Classifier
from prml.preprocess.label_transformer import LabelTransformer


class LeastSquaresClassifier(Classifier):
    """
    Least squares classifier model

    X : (N, D)
    W : (D, K)
    y = argmax_k X @ W
    """

    def __init__(self, W:np.ndarray=None):
        self.W = W

    def fit(self, X:np.ndarray, t:np.ndarray):
        """
        least squares fitting for classification

        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        t : (N,) or (N, K) np.ndarray
            training dependent variable
            in class index (N,) or one-of-k coding (N,K)
        """
        if t.ndim == 1:
            t = LabelTransformer().encode(t)
        self.W = np.linalg.pinv(X) @ t

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
            class index for each input
        """
        return np.argmax(X @ self.W, axis=-1)
