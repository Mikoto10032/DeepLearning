import numpy as np
from prml.nn.array.broadcast import broadcast_to
from prml.nn.function import Function
from prml.nn.math.log import log
from prml.nn.nonlinear.sigmoid import sigmoid
from prml.nn.random.random import RandomVariable
from prml.nn.tensor.tensor import Tensor


class Bernoulli(RandomVariable):
    """
    Bernoulli distribution
    p(x|mu) = mu^x (1 - mu)^(1 - x)
    Parameters
    ----------
    mu : tensor_like
        probability of value 1
    logit : tensor_like
        log-odd of value 1
    data : tensor_like
        observed data
    p : RandomVariable
        original distribution of a model
    """

    def __init__(self, mu=None, logit=None, data=None, p=None):
        super().__init__(data, p)
        if mu is not None and logit is None:
            mu = self._convert2tensor(mu)
            self.mu = mu
        elif mu is None and logit is not None:
            logit = self._convert2tensor(logit)
            self.logit = logit
        elif mu is None and logit is None:
            raise ValueError("Either mu or logit must not be None")
        else:
            raise ValueError("Cannot assign both mu and logit")

    @property
    def mu(self):
        try:
            return self.parameter["mu"]
        except KeyError:
            return sigmoid(self.logit)

    @mu.setter
    def mu(self, mu):
        try:
            inrange = (0 <= mu.value <= 1)
        except ValueError:
            inrange = ((mu.value >= 0).all() and (mu.value <= 1).all())

        if not inrange:
            raise ValueError("value of mu must all be positive")
        self.parameter["mu"] = mu

    @property
    def logit(self):
        try:
            return self.parameter["logit"]
        except KeyError:
            raise AttributeError("no attribute named logit")

    @logit.setter
    def logit(self, logit):
        self.parameter["logit"] = logit

    def forward(self):
        return (np.random.uniform(size=self.mu.shape) < self.mu.value).astype(np.int)

    def _pdf(self, x):
        return self.mu ** x * (1 - self.mu) ** (1 - x)

    def _log_pdf(self, x):
        try:
            return -SigmoidCrossEntropy().forward(self.logit, x)
        except AttributeError:
            return x * log(self.mu) + (1 - x) * log(1 - self.mu)


class SigmoidCrossEntropy(Function):
    """
    sum of cross entropies for binary data
    logistic sigmoid
    y_i = 1 / (1 + exp(-x_i))
    cross_entropy_i = -t_i * log(y_i) - (1 - t_i) * log(1 - y_i)
    Parameters
    ----------
    x : ndarary
        input logit
    y : ndarray
        corresponding target binaries
    """

    def _check_input(self, x, t):
        x = self._convert2tensor(x)
        t = self._convert2tensor(t)
        if x.shape != t.shape:
            shape = np.broadcast(x.value, t.value).shape
            if x.shape != shape:
                x = broadcast_to(x, shape)
            if t.shape != shape:
                t = broadcast_to(t, shape)
        return x, t


    def forward(self, x, t):
        x, t = self._check_input(x, t)
        self.x = x
        self.t = t
        # y = self.forward(x)
        # np.clip(y, 1e-10, 1 - 1e-10, out=y)
        # return np.sum(-t * np.log(y) - (1 - t) * np.log(1 - y))
        loss = (
            np.maximum(x.value, 0)
            - t.value * x.value
            + np.log1p(np.exp(-np.abs(x.value)))
        )
        return Tensor(loss, function=self)

    def backward(self, delta):
        y = np.tanh(self.x.value * 0.5) * 0.5 + 0.5
        dx = delta * (y - self.t.value)
        dt = - delta * self.x.value
        self.x.backward(dx)
        self.t.backward(dt)
