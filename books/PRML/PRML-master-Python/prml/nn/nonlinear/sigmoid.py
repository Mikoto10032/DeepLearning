import numpy as np
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function


class Sigmoid(Function):
    """
    logistic sigmoid function
    y = 1 / (1 + exp(-x))
    """

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.output = np.tanh(x.value * 0.5) * 0.5 + 0.5
        if isinstance(self.x, Constant):
            return Constant(self.output)
        return Tensor(self.output, function=self)

    def backward(self, delta):
        dx = self.output * (1 - self.output) * delta
        self.x.backward(dx)


def sigmoid(x):
    """
    logistic sigmoid function
    y = 1 / (1 + exp(-x))
    """
    return Sigmoid().forward(x)
