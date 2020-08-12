import numpy as np
from prml.nn.array.broadcast import broadcast_to
from prml.nn.math.abs import abs
from prml.nn.math.exp import exp
from prml.nn.math.log import log
from prml.nn.random.random import RandomVariable
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor


class Laplace(RandomVariable):
    """
    Laplace distribution
    p(x|loc, scale)
    = exp(-|x - loc|/scale) / (2 * scale)
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
        eps = 0.5 - np.random.uniform(size=self.loc.shape)
        self.eps = np.sign(eps) * np.log(1 - 2 * np.abs(eps))
        self.output = self.loc.value - self.scale.value * self.eps
        if isinstance(self.loc, Constant) and isinstance(self.scale, Constant):
            return Constant(self.output)
        return Tensor(self.output, function=self)

    def backward(self, delta):
        dloc = delta
        dscale = -delta * self.eps
        self.loc.backward(dloc)
        self.scale.backward(dscale)

    def _pdf(self, x):
        return 0.5 * exp(-abs(x - self.loc) / self.scale) / self.scale

    def _log_pdf(self, x):
        return np.log(0.5) - abs(x - self.loc) / self.scale - log(self.scale)
