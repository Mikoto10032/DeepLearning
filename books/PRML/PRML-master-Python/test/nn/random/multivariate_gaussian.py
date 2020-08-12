import unittest
import numpy as np
from prml import nn


class TestMultivariateGaussian(unittest.TestCase):

    def test_multivariate_gaussian(self):
        self.assertRaises(ValueError, nn.random.MultivariateGaussian, np.zeros(2), np.eye(3))
        self.assertRaises(ValueError, nn.random.MultivariateGaussian, np.zeros(2), np.eye(2) * -1)

        x_train = np.array([
            [1., 1.],
            [1., -1],
            [-1., 1.],
            [-1., -2.]
        ])
        mu = nn.Parameter(np.ones(2))
        cov = nn.Parameter(np.eye(2) * 2)
        for _ in range(1000):
            mu.cleargrad()
            cov.cleargrad()
            x = nn.random.MultivariateGaussian(mu, cov + cov.transpose(), data=x_train)
            log_likelihood = x.log_pdf().sum()
            log_likelihood.backward()
            mu.value += 0.1 * mu.grad
            cov.value += 0.1 * cov.grad
        self.assertTrue(np.allclose(mu.value, x_train.mean(axis=0)))
        self.assertTrue(np.allclose(np.cov(x_train, rowvar=False, bias=True), x.cov.value))


if __name__ == '__main__':
    unittest.main()
