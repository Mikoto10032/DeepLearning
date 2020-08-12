import numpy as np
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor


class Function(object):
    """
    Base class for differentiable functions
    """

    def _convert2tensor(self, x):
        if isinstance(x, (int, float, np.number, np.ndarray)):
            x = Constant(x)
        elif not isinstance(x, Tensor):
            raise TypeError(
                "Unsupported class for input: {}".format(type(x))
            )
        return x

    def _equal_ndim(self, x, ndim):
        if x.ndim != ndim:
            raise ValueError(
                "dimensionality of the input must be {}, not {}"
                .format(ndim, x.ndim)
            )

    def _atleast_ndim(self, x, ndim):
        if x.ndim < ndim:
            raise ValueError(
                "dimensionality of the input must be"
                " larger or equal to {}, not {}"
                .format(ndim, x.ndim)
            )
