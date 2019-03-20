import numpy as np
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function
from prml.nn.array.broadcast import broadcast_to


class Power(Function):
    """
    First array elements raised to powers from second array
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
        self.output = np.power(x.value, y.value)
        if isinstance(self.x, Constant) and isinstance(self.y, Constant):
            return Constant(self.output)
        return Tensor(self.output, function=self)

    def backward(self, delta):
        dx = self.y.value * np.power(self.x.value, self.y.value - 1) * delta
        if self.x.size == 1:
            if self.x.value > 0:
                dy = self.output * np.log(self.x.value) * delta
            else:
                dy = None
        else:
            if (self.x.value > 0).all():
                dy = self.output * np.log(self.x.value) * delta
            else:
                dy = None
        self.x.backward(dx)
        self.y.backward(dy)


def power(x, y):
    """
    First array elements raised to powers from second array
    """
    return Power().forward(x, y)


def rpower(x, y):
    return Power().forward(y, x)
