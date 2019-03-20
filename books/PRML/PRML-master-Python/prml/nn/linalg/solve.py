import numpy as np
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function


class Solve(Function):

    def forward(self, a, b):
        a = self._convert2tensor(a)
        b = self._convert2tensor(b)
        self._equal_ndim(a, 2)
        self._equal_ndim(b, 2)
        self.a = a
        self.b = b
        self.output = np.linalg.solve(a.value, b.value)
        if isinstance(self.a, Constant) and isinstance(self.b, Constant):
            return Constant(self.output)
        return Tensor(self.output, function=self)

    def backward(self, delta):
        db = np.linalg.solve(self.a.value.T, delta)
        da = -db @ self.output.T
        self.a.backward(da)
        self.b.backward(db)


def solve(a, b):
    """
    solve a linear matrix equation
    ax = b
    Parameters
    ----------
    a : (d, d) tensor_like
        coefficient matrix
    b : (d, k) tensor_like
        dependent variable
    Returns
    -------
    output : (d, k) tensor_like
        solution of the equation
    """
    return Solve().forward(a, b)
