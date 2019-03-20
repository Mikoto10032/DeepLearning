from prml.nn.tensor.tensor import Tensor


class Parameter(Tensor):
    """
    parameter to be optimized
    """

    def __init__(self, value, prior=None):
        super().__init__(value, function=None)
        self.grad = None
        self.prior = prior

    def _backward(self, delta, **kwargs):
        if self.grad is None:
            self.grad = delta
        else:
            self.grad += delta

    def cleargrad(self):
        self.grad = None
        if self.prior is not None:
            loss = -self.prior.log_pdf(self).sum()
            loss.backward()
