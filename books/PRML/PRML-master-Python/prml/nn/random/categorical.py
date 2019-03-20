import numpy as np
from prml.nn.array.broadcast import broadcast_to
from prml.nn.function import Function
from prml.nn.math.log import log
from prml.nn.math.product import prod
from prml.nn.nonlinear.softmax import softmax
from prml.nn.random.random import RandomVariable
from prml.nn.tensor.tensor import Tensor


class Categorical(RandomVariable):
    """
    Categorical distribution
    Parameters
    ----------
    mu : (..., K) tensor_like
        probability of each index
    logit : (..., K) tensor_like
        log-odd of each index
    axis : int
        axis along which represents each outcome
    data : tensor_like
        realization
    p : RandomVariable
        original distribution of a model
    Attributes
    ----------
    n_category : int
        number of categories
    """

    def __init__(self, mu=None, logit=None, axis=-1, data=None, p=None):
        super().__init__(data, p)
        assert axis == -1
        self.axis = axis
        if mu is not None and logit is None:
            self.mu = self._convert2tensor(mu)
        elif mu is None and logit is not None:
            self.logit = self._convert2tensor(logit)
        elif mu is None and logit is None:
            raise ValueError("Either mu or logit must not be None")
        else:
            raise ValueError("Cannot assign both mu and logit")

    @property
    def mu(self):
        try:
            return self.parameter["mu"]
        except KeyError:
            return softmax(self.parameter["logit"])

    @mu.setter
    def mu(self, mu):
        self._atleast_ndim(mu, 1)
        if not ((mu.value >= 0).all() and (mu.value <= 1).all()):
            raise ValueError("values of mu must be in [0, 1]")
        if not np.allclose(mu.value.sum(axis=self.axis), 1):
            raise ValueError(f"mu must be normalized along axis {self.axis}")
        self.parameter["mu"] = mu
        self.n_category = mu.shape[self.axis]

    @property
    def logit(self):
        try:
            return self.parameter["logit"]
        except KeyError:
            raise AttributeError("no attribute named logit")

    @logit.setter
    def logit(self, logit):
        self._atleast_ndim(logit, 1)
        self.parameter["logit"] = logit
        self.n_category = logit.shape[self.axis]

    def forward(self):
        if self.mu.ndim == 1:
            index = np.random.choice(self.n_category, p=self.mu.value)
            return np.eye(self.n_category)[index]
        elif self.mu.ndim == 2:
            indices = np.array(
                [np.random.choice(self.n_category, p=p.value) for p in self.mu.value]
            )
            return np.eye(self.n_category)[indices]
        else:
            raise NotImplementedError

    def _pdf(self, x):
        return prod(self.mu ** x, axis=self.axis)

    def _log_pdf(self, x):
        try:
            return -SoftmaxCrossEntropy(axis=self.axis).forward(self.logit, x)
        except (KeyError, AttributeError):
            return (x * log(self.mu)).sum(axis=self.axis)


class SoftmaxCrossEntropy(Function):

    def __init__(self, axis=-1):
        self.axis = axis

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

    def _softmax(self, array):
        y = np.exp(array - np.max(array, self.axis, keepdims=True))
        y /= np.sum(y, self.axis, keepdims=True)
        return y

    def forward(self, x, t):
        x, t = self._check_input(x, t)
        self.x = x
        self.t = t
        self.y = self._softmax(x.value)
        np.clip(self.y, 1e-10, 1, out=self.y)
        loss = -t.value * np.log(self.y)
        return Tensor(loss, function=self)

    def backward(self, delta):
        dx = delta * (self.y - self.t.value)
        dt = - delta * np.log(self.y)
        self.x.backward(dx)
        self.t.backward(dt)
