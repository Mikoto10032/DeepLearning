import numpy as np
import scipy.special as sp
from prml.nn.array.broadcast import broadcast_to
from prml.nn.math.exp import exp
from prml.nn.math.gamma import gamma
from prml.nn.math.log import log
from prml.nn.random.random import RandomVariable
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor


class Gamma(RandomVariable):
    """
    Gamma distribution
    p(x|a(shape), b(rate))
    = b^a * x^(a - 1) * e^(-bx) / Gamma(a)
    Parameters
    ----------
    shape : tensor_like
        shape parameter
    rate : tensor_like
        rate parameter
    data : tensor_like
        realization
    p : RandomVariable
        original distribution of a model
    """

    def __init__(self, shape, rate, data=None, p=None):
        super().__init__(data, p)
        shape, rate = self._check_input(shape, rate)
        self.shape = shape
        self.rate = rate

    def _check_input(self, shape, rate):
        shape = self._convert2tensor(shape)
        rate = self._convert2tensor(rate)
        if shape.shape != rate.shape:
            shape_ = np.broadcast(shape.value, rate.value).shape
            if shape.shape != shape_:
                shape = broadcast_to(shape, shape_)
            if rate.shape != shape_:
                rate = broadcast_to(rate, shape_)
        return shape, rate

    @property
    def shape(self):
        return self.parameter["shape"]

    @shape.setter
    def shape(self, shape):
        try:
            ispositive = (shape.value > 0).all()
        except AttributeError:
            ispositive = (shape.value > 0)

        if not ispositive:
            raise ValueError("value of shape must be positive")
        self.parameter["shape"] = shape

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
        self.output = np.random.gamma(self.shape.value, 1 / self.rate.value)
        if isinstance(self.shape, Constant) and isinstance(self.rate, Constant):
            return Constant(self.output)
        return Tensor(self.output, function=self)

    def backward(self, delta):
        a = self.shape.value
        psia = sp.digamma(a)
        psi1a = sp.polygamma(1, a)
        sqrtpsi1a = np.sqrt(psi1a)
        psi2a = sp.polygamma(2, a)
        b = self.rate.value
        eps = (np.log(self.output) - psia + np.log(b)) / sqrtpsi1a
        dshape = self.output * (0.5 * eps * psi2a / sqrtpsi1a + psi1a) * delta
        drate = -delta * self.output / b
        self.shape.backward(dshape)
        self.rate.backward(drate)

    def _pdf(self, x):
        return (
            self.rate ** self.shape
            * x ** (self.shape - 1)
            * exp(-self.rate * x)
            / gamma(self.shape)
        )

    def _log_pdf(self, x):
        return (
            self.shape * log(self.rate)
            + (self.shape - 1) * log(x)
            - self.rate * x
            - log(gamma(self.shape))
        )
