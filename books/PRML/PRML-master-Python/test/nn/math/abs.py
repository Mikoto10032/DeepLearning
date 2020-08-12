import unittest
import numpy as np
from prml import nn


class TestAbs(unittest.TestCase):

    def test_abs(self):
        np.random.seed(1234)
        x = nn.Parameter(np.random.randn(5, 7))
        sign = np.sign(x.value)
        y = nn.abs(x)
        self.assertTrue((y.value == np.abs(x.value)).all())

        for _ in range(10000):
            x.cleargrad()
            y = nn.abs(x)
            nn.square(y - 0.01).sum().backward()
            x.value -= x.grad * 0.001
        self.assertTrue(np.allclose(x.value, 0.01 * sign))


if __name__ == '__main__':
    unittest.main()
