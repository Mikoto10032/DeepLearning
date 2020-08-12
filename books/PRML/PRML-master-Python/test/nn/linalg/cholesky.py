import unittest
import numpy as np
from prml import nn


class TestCholesky(unittest.TestCase):

    def test_cholesky(self):
        A = np.array([
            [2., -1],
            [-1., 5.]
        ])
        L = np.linalg.cholesky(A)
        Ap = nn.Parameter(A)
        L_test = nn.linalg.cholesky(Ap)
        self.assertTrue((L == L_test.value).all())

        T = np.array([
            [1., 0.],
            [-1., 2.]
        ])
        for _ in range(1000):
            Ap.cleargrad()
            L_ = nn.linalg.cholesky(Ap)
            loss = nn.square(T - L_).sum()
            loss.backward()
            Ap.value -= 0.1 * Ap.grad

        self.assertTrue(np.allclose(Ap.value, T @ T.T))


if __name__ == '__main__':
    unittest.main()
