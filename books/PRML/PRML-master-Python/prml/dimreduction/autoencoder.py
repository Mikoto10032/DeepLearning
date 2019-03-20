import numpy as np
from prml import nn


class Autoencoder(nn.Network):

    def __init__(self, *args):
        self.n_unit = len(args)
        super().__init__()
        for i in range(self.n_unit - 1):
            self.parameter[f"w_encode{i}"] = nn.Parameter(np.random.randn(args[i], args[i + 1]))
            self.parameter[f"b_encode{i}"] = nn.Parameter(np.zeros(args[i + 1]))
            self.parameter[f"w_decode{i}"] = nn.Parameter(np.random.randn(args[i + 1], args[i]))
            self.parameter[f"b_decode{i}"] = nn.Parameter(np.zeros(args[i]))

    def transform(self, x):
        h = x
        for i in range(self.n_unit - 1):
            h = nn.tanh(h @ self.parameter[f"w_encode{i}"] + self.parameter[f"b_encode{i}"])
        return h.value

    def forward(self, x):
        h = x
        for i in range(self.n_unit - 1):
            h = nn.tanh(h @ self.parameter[f"w_encode{i}"] + self.parameter[f"b_encode{i}"])
        for i in range(self.n_unit - 2, 0, -1):
            h = nn.tanh(h @ self.parameter[f"w_decode{i}"] + self.parameter[f"b_decode{i}"])
        x_ = h @ self.parameter["w_decode0"] + self.parameter["b_decode0"]
        self.px = nn.random.Gaussian(x_, 1., data=x)

    def fit(self, x, n_iter=100, learning_rate=1e-3):
        optimizer = nn.optimizer.Adam(self.parameter, learning_rate)
        for _ in range(n_iter):
            self.clear()
            self.forward(x)
            log_likelihood = self.log_pdf()
            log_likelihood.backward()
            optimizer.update()
