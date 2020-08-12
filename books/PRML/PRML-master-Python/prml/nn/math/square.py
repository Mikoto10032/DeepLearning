import numpy as np
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function


class Square(Function):
    """
    element-wise square of the input
    y = x * x
    """

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        if isinstance(self.x, Constant):
            return Constant(np.square(x.value))
        return Tensor(np.square(x.value), function=self)

    def backward(self, delta):
        dx = 2 * self.x.value * delta
        self.x.backward(dx)


def square(x):
    """
    element-wise square of the input
    y = x * x
    """
    return Square().forward(x)
