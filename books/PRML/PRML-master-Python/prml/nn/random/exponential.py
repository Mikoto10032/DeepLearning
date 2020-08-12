import numpy as np
from prml.nn.math.exp import exp
from prml.nn.math.log import log
from prml.nn.random.random import RandomVariable
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor


class Exponential(RandomVariable):
    """
    Exponential distribution aka negative exponential distribution
    p(x|rate) = rate * exp(-rate * x)
    rate > 0
    Parameters
    ----------
    rate : tensor_like
        rate parameter
    data : tensor_like
        realization of this distribution
    p : RandomVariable
        original distribution of a model
    """

    def __init__(self, rate, data=None, p=None):
        super().__init__(data, p)
        rate = self._convert2tensor(rate)
        self.rate = rate

    @property
    def rate(self):
        return self.parameter["rate"]

    @rate.setter
    def rate(self, rate):
        try:
            ispositive = (rate.value > 0).all()
        except AttributeError:
            ispositive = (rate.value > 0)

        if not ispositive:
            raise ValueError("value of rate must be positive")
        self.parameter["rate"] = rate

    def forward(self):
        eps = np.random.standard_exponential(size=self.rate.shape)
        self.output = eps / self.rate.value
        if isinstance(self.rate, Constant):
            return Constant(self.output)
        return Tensor(self.output, self)

    def backward(self, delta):
        drate = -delta * self.output / self.rate.value
        self.rate.backward(drate)

    def _pdf(self, x):
        return self.rate * exp(-self.rate * x)

    def _log_pdf(self, x):
        return -self.rate * x + log(self.rate)
