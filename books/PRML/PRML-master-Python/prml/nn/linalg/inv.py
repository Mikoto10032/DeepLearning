import numpy as np
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function


class Inverse(Function):

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self._equal_ndim(x, 2)
        self.output = np.linalg.inv(x.value)
        if isinstance(self.x, Constant):
            return Constant(self.output)
        return Tensor(self.output, function=self)

    def backward(self, delta):
        dx = -self.output.T @ delta @ self.output.T
        self.x.backward(dx)


def inv(x):
    """
    inverse of a matrix
    Parameters
    ----------
    x : (d, d) tensor_like
        a matrix to be inverted
    Returns
    -------
    output : (d, d) tensor_like
        inverse of the input
    """
    return Inverse().forward(x)
