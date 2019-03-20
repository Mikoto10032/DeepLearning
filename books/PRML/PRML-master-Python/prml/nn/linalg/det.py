import numpy as np
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function


class Determinant(Function):

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self._equal_ndim(x, 2)
        self.output = np.linalg.det(x.value)
        if isinstance(self.x, Constant):
            return Constant(self.output)
        return Tensor(self.output, function=self)

    def backward(self, delta):
        dx = delta * self.output * np.linalg.inv(self.x.value.T)
        self.x.backward(dx)


def det(x):
    """
    determinant of a matrix
    Parameters
    ----------
    x : (d, d) tensor_like
        a matrix to compute its determinant
    Returns
    -------
    output : (d, d) tensor_like
        determinant of the input matrix
    """
    return Determinant().forward(x)
