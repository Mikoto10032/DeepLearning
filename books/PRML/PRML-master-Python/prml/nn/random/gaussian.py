import numpy as np
from prml.nn.array.broadcast import broadcast_to
from prml.nn.function import Function
from prml.nn.math.exp import exp
from prml.nn.math.log import log
from prml.nn.math.sqrt import sqrt
from prml.nn.math.square import square
from prml.nn.random.random import RandomVariable
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor


class Gaussian(RandomVariable):
    """
    The Gaussian distribution
    p(x|mu(mean), sigma(std))
    = exp{-0.5 * (x - mu)^2 / sigma^2} / sqrt(2pi * sigma^2)
    Parameters
    ----------
    mu : tensor_like
        mean parameter
    std : tensor_like
        std parameter
    var : tensor_like
        variance parameter
    tau : tensor_like
        precision parameter
    data : tensor_like
        observed data
    p : RandomVariable
        original distribution of a model
    """

    def __init__(self, mu, std=None, var=None, tau=None, data=None, p=None):
        super().__init__(data, p)
        if std is not None and var is None and tau is None:
            self.mu, self.std = self._check_input(mu, std)
        elif std is None and var is not None and tau is None:
            self.mu, self.var = self._check_input(mu, var)
        elif std is None and var is None and tau is not None:
            self.mu, self.tau = self._check_input(mu, tau)
        elif std is None and var is None and tau is None:
            raise ValueError("Either std, var, or tau must be assigned")
        else:
            raise ValueError("Cannot assign more than two of these: std, var, tau")

    def _check_input(self, x, y):
        x = self._convert2tensor(x)
        y = self._convert2tensor(y)
        if x.shape != y.shape:
            shape = np.broadcast(x.value, y.value).shape
            if x.shape != shape:
                x = broadcast_to(x, shape)
            if y.shape != shape:
                y = broadcast_to(y, shape)
        return x, y

    @property
    def mu(self):
        return self.parameter["mu"]

    @mu.setter
    def mu(self, mu):
        self.parameter["mu"] = mu

    @property
    def std(self):
        return self.parameter["std"]

    @std.setter
    def std(self, std):
        try:
            ispositive = (std.value > 0).all()
        except AttributeError:
            ispositive = (std.value > 0)

        if not ispositive:
            raise ValueError("value of std must all be positive")
        self.parameter["std"] = std

    @property
    def var(self):
        try:
            return self._var
        except AttributeError:
            return square(self.std)

    @var.setter
    def var(self, var):
        try:
            ispositive = (var.value > 0).all()
        except AttributeError:
            ispositive = (var.value > 0)

        if not ispositive:
            raise ValueError("value of var must all be positive")
        self._var = var
        self.parameter["std"] = sqrt(var)

    @property
    def tau(self):
        try:
            return self._tau
        except AttributeError:
            return 1 / square(self.std)

    @tau.setter
    def tau(self, tau):
        try:
            ispositive = (tau.value > 0).all()
        except AttributeError:
            ispositive = (tau.value > 0)

        if not ispositive:
            raise ValueError("value of tau must be positive")
        self._tau = tau
        self.parameter["std"] = 1 / sqrt(tau)

    def forward(self):
        self.eps = np.random.normal(size=self.mu.shape)
        output = self.mu.value + self.std.value * self.eps
        if isinstance(self.mu, Constant) and isinstance(self.var, Constant):
            return Constant(output)
        return Tensor(output, self)

    def backward(self, delta):
        dmu = delta
        dstd = delta * self.eps
        self.mu.backward(dmu)
        self.std.backward(dstd)

    def _pdf(self, x):
        return (
            exp(-0.5 * square((x - self.mu) / self.std))
            / sqrt(2 * np.pi) / self.std
        )

    def _log_pdf(self, x):
        return GaussianLogPDF().forward(x, self.mu, self.tau)


class GaussianLogPDF(Function):

    def _check_input(self, x, mu, tau):
        x = self._convert2tensor(x)
        mu = self._convert2tensor(mu)
        tau = self._convert2tensor(tau)
        if not x.shape == mu.shape == tau.shape:
            shape = np.broadcast(x.value, mu.value, tau.value).shape
            if x.shape != shape:
                x = broadcast_to(x, shape)
            if mu.shape != shape:
                mu = broadcast_to(mu, shape)
            if tau.shape != shape:
                tau = broadcast_to(tau, shape)
        return x, mu, tau

    def forward(self, x, mu, tau):
        x, mu, tau = self._check_input(x, mu, tau)
        self.x = x
        self.mu = mu
        self.tau = tau
        output = (
            -0.5 * np.square(x.value - mu.value) * tau.value
            + 0.5 * np.log(tau.value)
            - 0.5 * np.log(2 * np.pi)
        )
        return Tensor(output, function=self)

    def backward(self, delta):
        dx = -0.5 * delta * (self.x.value - self.mu.value) * self.tau.value
        dmu = -0.5 * delta * (self.mu.value - self.x.value) * self.tau.value
        dtau = 0.5 * delta * (
            1 / self.tau.value
            - (self.x.value - self.mu.value) ** 2
        )
        self.x.backward(dx)
        self.mu.backward(dmu)
        self.tau.backward(dtau)
