from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function


class Reshape(Function):
    """
    reshape array
    """

    def forward(self, x, shape):
        x = self._convert2tensor(x)
        self._atleast_ndim(x, 1)
        self.x = x
        if isinstance(self.x, Constant):
            return Constant(x.value.reshape(*shape))
        return Tensor(x.value.reshape(*shape), function=self)

    def backward(self, delta):
        dx = delta.reshape(*self.x.shape)
        self.x.backward(dx)


def reshape(x, shape):
    """
    reshape N-dimensional array (N >= 1)
    """
    return Reshape().forward(x, shape)


def reshape_method(x, *args):
    return Reshape().forward(x, args)
