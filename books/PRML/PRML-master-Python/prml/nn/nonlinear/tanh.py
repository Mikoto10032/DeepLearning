import numpy as np
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function


class Tanh(Function):

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.output = np.tanh(x.value)
        if isinstance(self.x, Constant):
            return Constant(self.output)
        return Tensor(self.output, function=self)

    def backward(self, delta):
        dx = (1 - np.square(self.output)) * delta
        self.x.backward(dx)


def tanh(x):
    """
    hyperbolic tangent function
    y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    return Tanh().forward(x)
