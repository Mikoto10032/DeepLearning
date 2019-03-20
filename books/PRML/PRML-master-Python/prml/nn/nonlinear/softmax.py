import numpy as np
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function


class Softmax(Function):

    def __init__(self, axis=-1):
        if not isinstance(axis, int):
            raise TypeError("axis must be int")
        self.axis = axis

    def _softmax(self, array):
        y = array - np.max(array, self.axis, keepdims=True)
        np.exp(y, out=y)
        y /= y.sum(self.axis, keepdims=True)
        return y

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.output = self._softmax(x.value)
        if isinstance(x, Constant):
            return Constant(self.output)
        return Tensor(self.output, function=self)

    def backward(self, delta):
        dx = self.output * delta
        dx -= self.output * dx.sum(self.axis, keepdims=True)
        self.x.backward(dx)


def softmax(x, axis=-1):
    """
    softmax function along specified axis
    y_k = exp(x_k) / sum_i(exp(x_i))
    """
    return Softmax(axis=axis).forward(x)
