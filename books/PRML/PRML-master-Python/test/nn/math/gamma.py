import unittest
from prml import nn


class TestGamma(unittest.TestCase):

    def test_gamma(self):
        self.assertEqual(24, nn.gamma(5).value)
        a = nn.Parameter(2.5)
        eps = 1e-5
        b = nn.gamma(a)
        b.backward()
        num_grad = ((nn.gamma(a + eps) - nn.gamma(a - eps)) / (2 * eps)).value
        self.assertAlmostEqual(a.grad, num_grad)


if __name__ == '__main__':
    unittest.main()
