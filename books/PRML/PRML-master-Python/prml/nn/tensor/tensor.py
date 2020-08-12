import numpy as np


class Tensor(object):
    """
    a base class for tensor object
    """
    __array_ufunc__ = None

    def __init__(self, value, function=None):
        """
        construct Tensor object
        Parameters
        ----------
        value : array_like
            value of this tensor
        function : Function
            function output this tensor
        """
        if not isinstance(value, (int, float, np.number, np.ndarray)):
            raise TypeError(
                "Unsupported class for Tensor: {}".format(type(value))
            )
        self.value = value
        self.function = function

    def __format__(self, *args, **kwargs):
        return self.__repr__()

    def __repr__(self):
        if isinstance(self.value, np.ndarray):
            return (
                "{0}(shape={1.shape}, dtype={1.dtype})"
                .format(self.__class__.__name__, self.value)
            )
        else:
            return (
                "{0}(value={1})".format(self.__class__.__name__, self.value)
            )

    @property
    def ndim(self):
        if hasattr(self.value, "ndim"):
            return self.value.ndim
        else:
            return 0

    @property
    def shape(self):
        if hasattr(self.value, "shape"):
            return self.value.shape
        else:
            return ()

    @property
    def size(self):
        if hasattr(self.value, "size"):
            return self.value.size
        else:
            return 1

    def backward(self, delta=1, **kwargs):
        """
        back-propagate error
        Parameters
        ----------
        delta : array_like
            derivative with respect to this array
        """
        if isinstance(delta, np.ndarray):
            if delta.shape != self.shape:
                raise ValueError(
                    "shapes {} and {} not aligned"
                    .format(delta.shape, self.shape)
                )
        elif isinstance(delta, (int, float, np.number)):
            if self.shape != ():
                raise ValueError(
                    "delta must be np.ndarray"
                )
        else:
            raise TypeError(
                "unsupported class for delta"
            )
        self._backward(delta, **kwargs)

    def _backward(self, delta, **kwargs):
        if hasattr(self.function, "backward"):
            self.function.backward(delta, **kwargs)
