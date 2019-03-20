import numpy as np
from prml.nn.tensor.constant import Constant
from prml.nn.tensor.tensor import Tensor
from prml.nn.function import Function


class Cholesky(Function):

    def forward(self, x):
        x = self._convert2tensor(x)
        self.x = x
        self.output = np.linalg.cholesky(x.value)
        if isinstance(self.x, Constant):
            return Constant(self.output)
        return Tensor(self.output, function=self)

    def backward(self, delta):
        delta_lower = np.tril(delta)
        P = phi(self.output.T @ delta_lower)
        S = np.linalg.solve(
            self.output.T,
            P @ np.linalg.inv(self.output)
        )
        dx = S + S.T + np.diag(np.diag(S))
        self.x.backward(dx)


def phi(x):
    return 0.5 * (np.tril(x) + np.tril(x, -1))


def cholesky(x):
    """
    cholesky decomposition of positive-definite matrix
    x = LL^T
    Parameters
    ----------
    x : (d, d) tensor_like
        positive-definite matrix
    Returns
    -------
    L : (d, d)
        cholesky decomposition
    """
    return Cholesky().forward(x)
