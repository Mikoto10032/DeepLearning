import numpy as np
from prml.nn.array.broadcast import broadcast_to
from prml.nn.math.log import log
from prml.nn.math.square import square
from prml.nn.random.random import RandomVariable
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor


class Cauchy(RandomVariable):
    """
    Cauchy distribution aka Lorentz distribution
    p(x|x0(loc), scale)
    = 1 / [pi*scale * {1 + (x - x0)^2 / scale^2}]
    Parameters
    ----------
    loc : tensor_like
        location parameter
    scale : tensor_like
        scale parameter
    data : tensor_like
        realization
    p : RandomVariable
        original distribution of a model
    """

    def __init__(self, loc, scale, data=None, p=None):
        super().__init__(data, p)
        self.loc, self.scale = self._check_input(loc, scale)

    def _check_input(self, loc, scale):
        loc = self._convert2tensor(loc)
        scale = self._convert2tensor(scale)
        if loc.shape != scale.shape:
            shape = np.broadcast(loc.value, scale.value).shape
            if loc.shape != shape:
                loc = broadcast_to(loc, shape)
            if scale.shape != shape:
                scale = broadcast_to(scale, shape)
        return loc, scale

    @property
    def loc(self):
        return self.parameter["loc"]

    @loc.setter
    def loc(self, loc):
        self.parameter["loc"] = loc

    @property
    def scale(self):
        return self.parameter["scale"]

    @scale.setter
    def scale(self, scale):
        try:
            ispositive = (scale.value > 0).all()
        except AttributeError:
            ispositive = (scale.value > 0)

        if not ispositive:
            raise ValueError("value of scale must be positive")
        self.parameter["scale"] = scale

    def forward(self):
        self.eps = np.random.standard_cauchy(size=self.loc.shape)
        self.output = self.scale.value * self.eps + self.loc.value
        if isinstance(self.loc, Constant):
            return Constant(self.output)
        return Tensor(self.output, function=self)

    def backward(self, delta):
        dloc = delta
        dscale = delta * self.eps
        self.loc.backward(dloc)
        self.scale.backward(dscale)

    def _pdf(self, x):
        return (
            1 / (np.pi * self.scale * (1 + square((x - self.loc) / self.scale)))
        )

    def _log_pdf(self, x):
        return (
            -np.log(np.pi)
            - log(self.scale)
            - log(1 + square((x - self.loc) / self.scale))
        )
