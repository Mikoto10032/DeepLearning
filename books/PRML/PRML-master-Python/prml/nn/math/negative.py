from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function


class Negative(Function):
    """
    element-wise negative
    y = -x
    """

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        if isinstance(self.x, Constant):
            return Constant(-x.value)
        return Tensor(-x.value, function=self)

    def backward(self, delta):
        dx = -delta
        self.x.backward(dx)


def negative(x):
    """
    element-wise negative
    """
    return Negative().forward(x)
