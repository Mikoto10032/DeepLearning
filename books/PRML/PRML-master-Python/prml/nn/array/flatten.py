from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function


class Flatten(Function):
    """
    flatten array
    """

    def forward(self, x):
        x = self._convert2tensor(x)
        self._atleast_ndim(x, 2)
        self.x = x
        if isinstance(self.x, Constant):
            return Constant(x.value.flatten())
        return Tensor(x.value.flatten(), function=self)

    def backward(self, delta):
        dx = delta.reshape(*self.x.shape)
        self.x.backward(dx)


def flatten(x):
    """
    flatten N-dimensional array (N >= 2)
    """
    return Flatten().forward(x)
