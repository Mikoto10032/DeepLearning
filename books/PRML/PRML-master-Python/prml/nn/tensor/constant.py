from prml.nn.tensor.tensor import Tensor


class Constant(Tensor):
    """
    constant tensor class
    """

    def __init__(self, value):
        super().__init__(value)
