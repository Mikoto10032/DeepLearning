import unittest
import numpy as np
from prml import nn


class TestCategorical(unittest.TestCase):

    def test_categorical(self):
        np.random.seed(1234)
        obs = np.random.choice(3, 100, p=[0.2, 0.3, 0.5])
        obs = np.eye(3)[obs]
        a = nn.Parameter(np.zeros(3))
        for _ in range(100):
            a.cleargrad()
            x = nn.random.Categorical(logit=a, data=obs)
            x.log_pdf().sum().backward()
            a.value += 0.01 * a.grad
        self.assertTrue(np.allclose(np.mean(obs, 0), x.mu.value))


if __name__ == '__main__':
    unittest.main()
