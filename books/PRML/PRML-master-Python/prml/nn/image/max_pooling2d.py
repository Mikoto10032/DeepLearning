import numpy as np
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function
from prml.nn.image.util import img2patch, patch2img


class MaxPooling2d(Function):

    def __init__(self, pool_size, stride, pad):
        """
        construct 2 dimensional max-pooling function

        Parameters
        ----------
        pool_size : int or tuple of ints
            pooling size
        stride : int or tuple of ints
            stride of kernel application
        pad : int or tuple of ints
            padding image
        """
        self.pool_size = self._check_tuple(pool_size, "pool_size")
        self.stride = self._check_tuple(stride, "stride")
        self.pad = self._check_tuple(pad, "pad")
        self.pad = (0,) + self.pad + (0,)

    def _check_tuple(self, tup, name):
        if isinstance(tup, int):
            tup = (tup,) * 2
        if not isinstance(tup, tuple):
            raise TypeError(
                "Unsupported type for {}: {}".format(name, type(tup))
            )
        if len(tup) != 2:
            raise ValueError(
                "the length of {} must be 2, not {}".format(name, len(tup))
            )
        if not all([isinstance(n, int) for n in tup]):
            raise TypeError(
                "Unsuported type for {}".format(name)
            )
        if not all([n >= 0 for n in tup]):
            raise ValueError(
                "{} must be non-negative values".format(name)
            )
        return tup

    def forward(self, x):
        x = self._convert2tensor(x)
        self._equal_ndim(x, 4)
        self.x = x
        img = np.pad(x.value, [(p,) for p in self.pad], "constant")
        patch = img2patch(img, self.pool_size, self.stride)
        n_batch, xlen_out, ylen_out, _, _, in_channels = patch.shape
        patch = patch.reshape(n_batch, xlen_out, ylen_out, -1, in_channels)
        self.shape = img.shape
        self.index = patch.argmax(axis=3)
        return Tensor(patch.max(axis=3), function=self)

    def backward(self, delta):
        delta_patch = np.zeros(delta.shape + (np.prod(self.pool_size),))
        index = np.where(delta == delta) + (self.index.ravel(),)
        delta_patch[index] = delta.ravel()
        delta_patch = np.reshape(delta_patch, delta.shape + self.pool_size)
        delta_patch = delta_patch.transpose(0, 1, 2, 4, 5, 3)
        dx = patch2img(delta_patch, self.stride, self.shape)
        slices = [slice(p, len_ - p) for p, len_ in zip(self.pad, self.shape)]
        dx = dx[slices]
        self.x.backward(dx)


def max_pooling2d(x, pool_size, stride=1, pad=0):
    """
    spatial max pooling

    Parameters
    ----------
    x : (n_batch, xlen, ylen, in_channel) Tensor
        input tensor
    pool_size : int or tuple of ints (kx, ky)
        pooling size
    stride : int or tuple of ints (sx, sy)
        stride of pooling application
    pad : int or tuple of ints (px, py)
        padding input

    Returns
    -------
    output : (n_batch, xlen', ylen', out_channel) Tensor
        max pooled image
        len' = (len + p - k) // s + 1
    """
    return MaxPooling2d(pool_size, stride, pad).forward(x)
