import unittest
import numpy as np
from prml import nn


class TestPower(unittest.TestCase):

    def test_power(self):
        x = nn.Parameter(2.)
        y = 2 ** x
        self.assertEqual(y.value, 4)
        y.backward()
        self.assertEqual(x.grad, 4 * np.log(2))

        x = np.random.rand(10, 2)
        xp = nn.Parameter(x)
        y = xp ** 3
        self.assertTrue((y.value == x ** 3).all())
        y.backward(np.ones((10, 2)))
        self.assertTrue(np.allclose(xp.grad, 3 * x ** 2))


if __name__ == '__main__':
    unittest.main()
