import numpy as np
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function


class Sqrt(Function):
    """
    element-wise square root of the input
    y = sqrt(x)
    """

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.output = np.sqrt(x.value)
        if isinstance(self.x, Constant):
            return Constant(self.output)
        return Tensor(self.output, function=self)

    def backward(self, delta):
        dx = 0.5 * delta / self.output
        self.x.backward(dx)


def sqrt(x):
    """
    element-wise square root of the input
    y = sqrt(x)
    """
    return Sqrt().forward(x)
