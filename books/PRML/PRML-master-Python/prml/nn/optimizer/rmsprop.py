import numpy as np
from prml.nn.optimizer.optimizer import Optimizer


class RMSProp(Optimizer):
    """
    RMSProp optimizer
    initial
    msg = 0
    update rule
    msg = rho * msg + (1 - rho) * gradient ** 2
    param -= learning_rate * gradient / (sqrt(msg) + eps)
    """

    def __init__(self, parameter, learning_rate=1e-3, rho=0.9, epsilon=1e-8):
        super().__init__(parameter, learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.mean_squared_grad = []
        for p in self.parameter:
            self.mean_squared_grad.append(np.zeros(p.shape))

    def update(self):
        """
        update parameters
        """
        self.increment_iteration()
        for p, msg in zip(self.parameter, self.mean_squared_grad):
            if p.grad is None:
                continue
            grad = p.grad
            msg *= self.rho
            msg += (1 - self.rho) * grad ** 2
            p.value -= (
                self.learning_rate * grad / (np.sqrt(msg) + self.epsilon)
            )
