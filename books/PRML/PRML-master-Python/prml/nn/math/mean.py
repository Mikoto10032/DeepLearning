from prml.nn.math.sum import sum


def mean(x, axis=None, keepdims=False):
    """
    returns arithmetic mean of the elements along given axis
    """
    if axis is None:
        return sum(x, axis=None, keepdims=keepdims) / x.size
    elif isinstance(axis, int):
        N = x.shape[axis]
        return sum(x, axis=axis, keepdims=keepdims) / N
    elif isinstance(axis, tuple):
        N = 1
        for ax in axis:
            N *= x.shape[ax]
        return sum(x, axis=axis, keepdims=keepdims) / N
    else:
        raise TypeError(
            "Unsupported type for axis: {}".format(type(axis))
        )
