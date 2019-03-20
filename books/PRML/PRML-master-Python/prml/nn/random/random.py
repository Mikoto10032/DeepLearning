from prml.nn.function import Function
from prml.nn.tensor.constant import Constant


class RandomVariable(Function):
    """
    base class for random variables
    """

    def __init__(self, data=None, p=None):
        """
        construct a random variable
        Parameters
        ----------
        data : tensor_like
            observed data
        p : RandomVariable
            original distribution of a model
        Returns
        -------
        parameter : dict
            dictionary of parameters
        observed : bool
            flag of observed or not
        """
        if data is not None and p is not None:
            raise ValueError("Cannot assign both data and p at a time")
        if data is not None:
            data = self._convert2tensor(data)
        self.data = data
        self.observed = isinstance(data, Constant)
        self.p = p
        self.parameter = dict()

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        if p is not None and not isinstance(p, RandomVariable):
            raise TypeError("p must be RandomVariable")
        self._p = p

    def __repr__(self):
        string = f"{self.__class__.__name__}(\n"
        for key, value in self.parameter.items():
            string += (" " * 4)
            string += f"{key}={value}"
            string += "\n"
        string += ")"
        return string

    def draw(self):
        """
        generate a sample
        Returns
        -------
        sample : tensor
            sample generated from this random variable
        """
        if self.observed:
            raise ValueError("draw method cannot be used for observed random variable")
        self.data = self.forward()
        return self.data

    def pdf(self, x=None):
        """
        compute probability density function
        Parameters
        ----------
        x : tensor_like
            observed data
        Returns
        -------
        p : Tensor
            value of probability density function for each input
        """
        if not hasattr(self, "_pdf"):
            raise NotImplementedError
        if x is None:
            if self.data is None:
                raise ValueError("There is no given or sampled data")
            return self._pdf(self.data)
        return self._pdf(x)

    def log_pdf(self, x=None):
        """
        logarithm of probability density function
        Parameters
        ----------
        x : tensor_like
            observed data
        Returns
        -------
        output : Tensor
            logarithm of probability density function
        """
        if not hasattr(self, "_log_pdf"):
            raise NotImplementedError
        if x is None:
            if self.data is None:
                raise ValueError("No given or sampled data")
            return self._log_pdf(self.data)
        return self._log_pdf(x)

    def KLqp(self):
        r"""
        compute Kullback Leibler Divergence
        KL(q(self)||p) = \int q(x) ln(q(x) / p(x)) dx
        Returns
        -------
        kl : Tensor
            KL divergence
        """
        if self.p is None:
            raise ValueError("There is no assigned distribution p")
        if self.data is None:
            raise ValueError("There is no sampled data")
        return self.log_pdf() - self.p.log_pdf(self.data)
