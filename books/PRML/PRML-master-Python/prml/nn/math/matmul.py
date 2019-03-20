from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function


class MatMul(Function):
    """
    Matrix multiplication function
    """

    def _check_input(self, x, y):
        x = self._convert2tensor(x)
        y = self._convert2tensor(y)
        self._equal_ndim(x, 2)
        self._equal_ndim(y, 2)
        if x.shape[1] != y.shape[0]:
            raise ValueError(
                "shapes {} and {} not aligned: {} (dim 1) != {} (dim 0)"
                .format(x.shape, y.shape, x.shape[1], y.shape[0])
            )
        return x, y

    def forward(self, x, y):
        x, y = self._check_input(x, y)
        self.x = x
        self.y = y
        if isinstance(self.x, Constant) and isinstance(self.y, Constant):
            return Constant(x.value @ y.value)
        return Tensor(x.value @ y.value, function=self)

    def backward(self, delta):
        dx = delta @ self.y.value.T
        dy = self.x.value.T @ delta
        self.x.backward(dx)
        self.y.backward(dy)


def matmul(x, y):
    return MatMul().forward(x, y)


def rmatmul(x, y):
    return MatMul().forward(y, x)
