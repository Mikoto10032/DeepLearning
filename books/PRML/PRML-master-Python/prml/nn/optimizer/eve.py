import numpy as np
from prml.nn.optimizer.optimizer import Optimizer


class Eve(Optimizer):
    """
    Eve optimizer

    initialization
    m1 = 0 (initial 1st moment of gradient)
    m2 = 0 (initial 2nd moment of gradient)
    n_iter = 0

    update rule
    n_iter += 1
    learning
    """

    def __init__(self,
                 network,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 beta3=0.999,
                 lower_threshold=0.1,
                 upper_threshold=10.,
                 epsilon=1e-8):
        """
        construct Eve optimizer

        Parameters
        ----------
        network : Network
            neural network to be optmized
        learning_rate : float
        beta1 : float
            exponential decay rate for the 1st moment
        beta2 : float
            exponential decay rate for the 2nd moment
        beta3 : float
            exponential decay rate for computing relative change
        lower_threshold : float
             lower threshold for relative change
        upper_threshold : float
             upper threshold for relative change
        epsilon : float
            small constant to be added to denominator for numerical stability

        Attributes
        ----------
        n_iter : int
            number of iterations performed
        moment1 : dict
            1st moment of each parameter
        moment2 : dict
            2nd moment of each parameter
        """
        super().__init__(network, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.epsilon = epsilon
        self.moment1 = {}
        self.moment2 = {}
        self.f = 1.
        self.d = 1.
        for key, param in self.params.items():
            self.moment1[key] = np.zeros(param.shape)
            self.moment2[key] = np.zeros(param.shape)

    def update(self, loss):
        loss = float(loss)
        self.increment_iteration()
        if self.n_iter > 1:
            if loss > self.f:
                delta = self.lower_threshold + 1
                Delta = self.upper_threshold + 1
            else:
                delta = 1 / (self.upper_threshold + 1)
                Delta = 1 / (self.lower_threshold + 1)
            c = min(max(delta, loss / self.f), Delta)
            f = c * self.f
            r = abs(f - self.f) / min(f, self.f)
            self.d = self.beta3 * self.d * (1 - self.beta3) * r
            self.f = f
        else:
            self.f = loss
            self.d = 1
        lr = (
            self.learning_rate
            * (1 - self.beta2 ** self.n_iter) ** 0.5
            / (1 - self.beta1 ** self.n_iter)
        )
        for key, param in self.params.items():
            m1 = self.moment1[key]
            m2 = self.moment2[key]
            m1 += (1 - self.beta1) * (param.grad - m1)
            m2 += (1 - self.beta2) * (param.grad ** 2 - m2)
            param.value -= lr * m1 / (self.d * np.sqrt(m2) + self.epsilon)
