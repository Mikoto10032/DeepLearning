import numpy as np
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function


class BroadcastTo(Function):
    """
    Broadcast a tensor to an new shape
    """

    def forward(self, x, shape):
        x = self._convert2tensor(x)
        self.x = x
        output = np.broadcast_to(x.value, shape)
        if isinstance(self.x, Constant):
            return Constant(output)
        return Tensor(output, function=self)

    def backward(self, delta):
        dx = delta
        if delta.ndim != self.x.ndim:
            dx = dx.sum(axis=tuple(range(dx.ndim - self.x.ndim)))
            if isinstance(dx, np.number):
                dx = np.array(dx)
        axis = tuple(i for i, len_ in enumerate(self.x.shape) if len_ == 1)
        if axis:
            dx = dx.sum(axis=axis, keepdims=True)
        self.x.backward(dx)


def broadcast_to(x, shape):
    """
    Broadcast a tensor to an new shape
    """
    return BroadcastTo().forward(x, shape)
