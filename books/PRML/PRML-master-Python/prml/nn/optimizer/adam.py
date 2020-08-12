import numpy as np
from prml.nn.optimizer.optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer
    initialization
    m1 = 0 (Initial 1st moment of gradient)
    m2 = 0 (Initial 2nd moment of gradient)
    n_iter = 0
    update rule
    n_iter += 1
    learning_rate *= sqrt(1 - beta2^n) / (1 - beta1^n)
    m1 = beta1 * m1 + (1 - beta1) * gradient
    m2 = beta2 * m2 + (1 - beta2) * gradient^2
    param += learning_rate * m1 / (sqrt(m2) + epsilon)
    """

    def __init__(self, parameter, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        construct Adam optimizer
        Parameters
        ----------
        parameters : list
            list of parameters to be optimized
        learning_rate : float
        beta1 : float
            exponential decay rate for the 1st moment
        beta2 : float
            exponential decay rate for the 2nd moment
        epsilon : float
            small constant to be added to denominator for numerical stability
        Attributes
        ----------
        n_iter : int
            number of iterations performed
        moment1 : dict
            1st moment of each learnable parameter
        moment2 : dict
            2nd moment of each learnable parameter
        """
        super().__init__(parameter, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moment1 = []
        self.moment2 = []
        for p in self.parameter:
            self.moment1.append(np.zeros(p.shape))
            self.moment2.append(np.zeros(p.shape))

    def update(self):
        """
        update parameter of the neural network
        """
        self.increment_iteration()
        lr = (
            self.learning_rate
            * (1 - self.beta2 ** self.n_iter) ** 0.5
            / (1 - self.beta1 ** self.n_iter))
        for p, m1, m2 in zip(self.parameter, self.moment1, self.moment2):
            if p.grad is None:
                continue
            m1 += (1 - self.beta1) * (p.grad - m1)
            m2 += (1 - self.beta2) * (p.grad ** 2 - m2)
            p.value += lr * m1 / (np.sqrt(m2) + self.epsilon)
