import numpy as np


class RandomVariable(object):
    """
    base class for random variables
    """

    def __init__(self):
        self.parameter = {}

    def __repr__(self):
        string = f"{self.__class__.__name__}(\n"
        for key, value in self.parameter.items():
            string += (" " * 4)
            if isinstance(value, RandomVariable):
                string += f"{key}={value:8}"
            else:
                string += f"{key}={value}"
            string += "\n"
        string += ")"
        return string

    def __format__(self, indent="4"):
        indent = int(indent)
        string = f"{self.__class__.__name__}(\n"
        for key, value in self.parameter.items():
            string += (" " * indent)
            if isinstance(value, RandomVariable):
                string += f"{key}=" + value.__format__(str(indent + 4))
            else:
                string += f"{key}={value}"
            string += "\n"
        string += (" " * (indent - 4)) + ")"
        return string

    def fit(self, X, **kwargs):
        """
        estimate parameter(s) of the distribution

        Parameters
        ----------
        X : np.ndarray
            observed data
        """
        self._check_input(X)
        if hasattr(self, "_fit"):
            self._fit(X, **kwargs)
        else:
            raise NotImplementedError

    # def ml(self, X, **kwargs):
    #     """
    #     maximum likelihood estimation of the parameter(s)
    #     of the distribution given data

    #     Parameters
    #     ----------
    #     X : (sample_size, ndim) np.ndarray
    #         observed data
    #     """
    #     self._check_input(X)
    #     if hasattr(self, "_ml"):
    #         self._ml(X, **kwargs)
    #     else:
    #         raise NotImplementedError

    # def map(self, X, **kwargs):
    #     """
    #     maximum a posteriori estimation of the parameter(s)
    #     of the distribution given data

    #     Parameters
    #     ----------
    #     X : (sample_size, ndim) np.ndarray
    #         observed data
    #     """
    #     self._check_input(X)
    #     if hasattr(self, "_map"):
    #         self._map(X, **kwargs)
    #     else:
    #         raise NotImplementedError

    # def bayes(self, X, **kwargs):
    #     """
    #     bayesian estimation of the parameter(s)
    #     of the distribution given data

    #     Parameters
    #     ----------
    #     X : (sample_size, ndim) np.ndarray
    #         observed data
    #     """
    #     self._check_input(X)
    #     if hasattr(self, "_bayes"):
    #         self._bayes(X, **kwargs)
    #     else:
    #         raise NotImplementedError

    def pdf(self, X):
        """
        compute probability density function
        p(X|parameter)

        Parameters
        ----------
        X : (sample_size, ndim) np.ndarray
            input of the function

        Returns
        -------
        p : (sample_size,) np.ndarray
            value of probability density function for each input
        """
        self._check_input(X)
        if hasattr(self, "_pdf"):
            return self._pdf(X)
        else:
            raise NotImplementedError

    def draw(self, sample_size=1):
        """
        draw samples from the distribution

        Parameters
        ----------
        sample_size : int
            sample size

        Returns
        -------
        sample : (sample_size, ndim) np.ndarray
            generated samples from the distribution
        """
        assert isinstance(sample_size, int)
        if hasattr(self, "_draw"):
            return self._draw(sample_size)
        else:
            raise NotImplementedError

    def _check_input(self, X):
        assert isinstance(X, np.ndarray)
