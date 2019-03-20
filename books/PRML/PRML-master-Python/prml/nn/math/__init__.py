from prml.nn.math.add import add
from prml.nn.math.divide import divide, rdivide
from prml.nn.math.exp import exp
from prml.nn.math.log import log
from prml.nn.math.matmul import matmul, rmatmul
from prml.nn.math.mean import mean
from prml.nn.math.multiply import multiply
from prml.nn.math.negative import negative
from prml.nn.math.power import power, rpower
from prml.nn.math.product import prod
from prml.nn.math.sqrt import sqrt
from prml.nn.math.square import square
from prml.nn.math.subtract import subtract, rsubtract
from prml.nn.math.sum import sum


from prml.nn.tensor.tensor import Tensor
Tensor.__add__ = add
Tensor.__radd__ = add
Tensor.__truediv__ = divide
Tensor.__rtruediv__ = rdivide
Tensor.mean = mean
Tensor.__matmul__ = matmul
Tensor.__rmatmul__ = rmatmul
Tensor.__mul__ = multiply
Tensor.__rmul__ = multiply
Tensor.__neg__ = negative
Tensor.__pow__ = power
Tensor.__rpow__ = rpower
Tensor.prod = prod
Tensor.__sub__ = subtract
Tensor.__rsub__ = rsubtract
Tensor.sum = sum


__all__ = [
    "add",
    "divide",
    "exp",
    "log",
    "matmul",
    "mean",
    "multiply",
    "power",
    "prod",
    "sqrt",
    "square",
    "subtract",
    "sum"
]
