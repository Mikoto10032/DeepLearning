from prml.nn.array.broadcast import broadcast_to
from prml.nn.array.flatten import flatten
from prml.nn.array.reshape import reshape, reshape_method
from prml.nn.array.split import split
from prml.nn.array.transpose import transpose, transpose_method
from prml.nn.tensor.tensor import Tensor


Tensor.flatten = flatten
Tensor.reshape = reshape_method
Tensor.transpose = transpose_method

__all__ = [
    "broadcast_to",
    "flatten",
    "reshape",
    "split",
    "transpose"
]
