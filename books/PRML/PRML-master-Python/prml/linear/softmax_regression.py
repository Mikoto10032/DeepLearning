import numpy as np
from prml.linear.classifier import Classifier
from prml.preprocess.label_transformer import LabelTransformer


class SoftmaxRegression(Classifier):
    """
    Softmax regression model
    aka
    multinomial logistic regression,
    multiclass logistic regression,
    maximum entropy classifier.

    y = softmax(X @ W)
    t ~ Categorical(t|y)
    """

    @staticmethod
    def _softmax(a):
        a_max = np.max(a, axis=-1, keepdims=True)
        exp_a = np.exp(a - a_max)
        return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

    def fit(self, X:np.ndarray, t:np.ndarray, max_iter:int=100, learning_rate:float=0.1):
        """
        maximum likelihood estimation of the parameter

        Parameters
        ----------
        X : (N, D) np.ndarray
            training independent variable
        t : (N,) or (N, K) np.ndarray
            training dependent variable
            in class index or one-of-k encoding
        max_iter : int, optional
            maximum number of iteration (the default is 100)
        learning_rate : float, optional
            learning rate of gradient descent (the default is 0.1)
        """
        if t.ndim == 1:
            t = LabelTransformer().encode(t)
        self.n_classes = np.size(t, 1)
        W = np.zeros((np.size(X, 1), self.n_classes))
        for _ in range(max_iter):
            W_prev = np.copy(W)
            y = self._softmax(X @ W)
            grad = X.T @ (y - t)
            W -= learning_rate * grad
            if np.allclose(W, W_prev):
                break
        self.W = W

    def proba(self, X:np.ndarray):
        """
        compute probability of input belonging each class

        Parameters
        ----------
        X : (N, D) np.ndarray
            independent variable

        Returns
        -------
        (N, K) np.ndarray
            probability of each class
        """
        return self._softmax(X @ self.W)

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
        return np.argmax(self.proba(X), axis=-1)
