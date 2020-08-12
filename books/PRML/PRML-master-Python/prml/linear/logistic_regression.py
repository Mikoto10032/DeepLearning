import numpy as np
from prml.linear.classifier import Classifier


class LogisticRegression(Classifier):
    """
    Logistic regression model

    y = sigmoid(X @ w)
    t ~ Bernoulli(t|y)
    """

    @staticmethod
    def _sigmoid(a):
        return np.tanh(a * 0.5) * 0.5 + 0.5

    def fit(self, X:np.ndarray, t:np.ndarray, max_iter:int=100):
        """
        maximum likelihood estimation of logistic regression model

        Parameters
        ----------
        X : (N, D) np.ndarray
            training data independent variable
        t : (N,) np.ndarray
            training data dependent variable
            binary 0 or 1
        max_iter : int, optional
            maximum number of paramter update iteration (the default is 100)
        """
        w = np.zeros(np.size(X, 1))
        for _ in range(max_iter):
            w_prev = np.copy(w)
            y = self._sigmoid(X @ w)
            grad = X.T @ (y - t)
            hessian = (X.T * y * (1 - y)) @ X
            try:
                w -= np.linalg.solve(hessian, grad)
            except np.linalg.LinAlgError:
                break
            if np.allclose(w, w_prev):
                break
        self.w = w

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
        return self._sigmoid(X @ self.w)

    def classify(self, X:np.ndarray, threshold:float=0.5):
        """
        classify input data

        Parameters
        ----------
        X : (N, D) np.ndarray
            independent variable to be classified
        threshold : float, optional
            threshold of binary classification (default is 0.5)

        Returns
        -------
        (N,) np.ndarray
            binary class for each input
        """
        return (self.proba(X) > threshold).astype(np.int)
