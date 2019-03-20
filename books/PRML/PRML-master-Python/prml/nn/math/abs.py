import numpy as np
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function


class Abs(Function):

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.output = np.abs(x.value)
        if isinstance(x, Constant):
            return Constant(self.output)
        self.sign = np.sign(x.value)
        return Tensor(self.output, function=self)

    def backward(self, delta):
        dx = self.sign * delta
        self.x.backward(dx)


def abs(x):
    """
    element-wise absolute function
    """
    return Abs().forward(x)
