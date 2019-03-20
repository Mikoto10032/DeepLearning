import unittest
import numpy as np
from prml import nn


class TestBernoulli(unittest.TestCase):

    def test_bernoulli(self):
        np.random.seed(1234)
        obs = np.random.choice(2, 1000, p=[0.1, 0.9])
        a = nn.Parameter(0)
        for _ in range(100):
            a.cleargrad()
            x = nn.random.Bernoulli(logit=a, data=obs)
            x.log_pdf().sum().backward()
            a.value += a.grad * 0.01
        self.assertAlmostEqual(x.mu.value, np.mean(obs))


if __name__ == '__main__':
    unittest.main()
