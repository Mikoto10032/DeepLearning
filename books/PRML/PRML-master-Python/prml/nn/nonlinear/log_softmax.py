import numpy as np
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function


class LogSoftmax(Function):

    def __init__(self, axis=-1):
        self.axis = axis

    def _logsumexp(self, x):
        x_max = np.max(x, axis=self.axis, keepdims=True)
        y = x - x_max
        np.exp(y, out=y)
        np.log(y.sum(axis=self.axis, keepdims=True), out=y)
        y += x_max
        return y

    def _forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.output = x.value - self._logsumexp(x.value)
        return Tensor(self.output, function=self)

    def _backward(self, delta):
        dx = delta
        dx -= np.exp(self.output) * dx.sum(axis=self.axis, keepdims=True)
        self.x.backward(dx)


def log_softmax(x, axis=-1):
    return LogSoftmax(axis=axis).forward(x)
