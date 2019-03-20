import numpy as np


class GaussianProcessRegressor(object):

    def __init__(self, kernel, beta=1.):
        """
        construct gaussian process regressor

        Parameters
        ----------
        kernel
            kernel function
        beta : float
            precision parameter of observation noise
        """
        self.kernel = kernel
        self.beta = beta

    def fit(self, X, t, iter_max=0, learning_rate=0.1):
        """
        maximum likelihood estimation of parameters in kernel function

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input
        t : ndarray (sample_size,)
            corresponding target
        iter_max : int
            maximum number of iterations updating hyperparameters
        learning_rate : float
            updation coefficient

        Attributes
        ----------
        covariance : ndarray (sample_size, sample_size)
            variance covariance matrix of gaussian process
        precision : ndarray (sample_size, sample_size)
            precision matrix of gaussian process

        Returns
        -------
        log_likelihood_list : list
            list of log likelihood value at each iteration
        """
        if X.ndim == 1:
            X = X[:, None]
        log_likelihood_list = [-np.Inf]
        self.X = X
        self.t = t
        I = np.eye(len(X))
        Gram = self.kernel(X, X)
        self.covariance = Gram + I / self.beta
        self.precision = np.linalg.inv(self.covariance)
        for i in range(iter_max):
            gradients = self.kernel.derivatives(X, X)
            updates = np.array(
                [-np.trace(self.precision.dot(grad)) + t.dot(self.precision.dot(grad).dot(self.precision).dot(t)) for grad in gradients])
            for j in range(iter_max):
                self.kernel.update_parameters(learning_rate * updates)
                Gram = self.kernel(X, X)
                self.covariance = Gram + I / self.beta
                self.precision = np.linalg.inv(self.covariance)
                log_like = self.log_likelihood()
                if log_like > log_likelihood_list[-1]:
                    log_likelihood_list.append(log_like)
                    break
                else:
                    self.kernel.update_parameters(-learning_rate * updates)
                    learning_rate *= 0.9
        log_likelihood_list.pop(0)
        return log_likelihood_list

    def log_likelihood(self):
        return -0.5 * (
            np.linalg.slogdet(self.covariance)[1]
            + self.t @ self.precision @ self.t
            + len(self.t) * np.log(2 * np.pi))

    def predict(self, X, with_error=False):
        """
        mean of the gaussian process

        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input

        Returns
        -------
        mean : ndarray (sample_size,)
            predictions of corresponding inputs
        """
        if X.ndim == 1:
            X = X[:, None]
        K = self.kernel(X, self.X)
        mean = K @ self.precision @ self.t
        if with_error:
            var = (
                self.kernel(X, X, False)
                + 1 / self.beta
                - np.sum(K @ self.precision * K, axis=1))
            return mean.ravel(), np.sqrt(var.ravel())
        return mean
