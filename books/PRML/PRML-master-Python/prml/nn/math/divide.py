import numpy as np
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function
from prml.nn.array.broadcast import broadcast_to


class Divide(Function):
    """
    divide arguments element-wise
    """

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

    def forward(self, x, y):
        x, y = self._check_input(x, y)
        self.x = x
        self.y = y
        if isinstance(self.x, Constant) and isinstance(self.y, Constant):
            return Constant(x.value / y.value)
        return Tensor(x.value / y.value, function=self)

    def backward(self, delta):
        dx = delta / self.y.value
        dy = - delta * self.x.value / self.y.value ** 2
        self.x.backward(dx)
        self.y.backward(dy)


def divide(x, y):
    return Divide().forward(x, y)


def rdivide(x, y):
    return Divide().forward(y, x)
