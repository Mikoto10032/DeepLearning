import unittest
import numpy as np
from prml import nn


class TestExp(unittest.TestCase):

    def test_exp(self):
        x = nn.Parameter(2.)
        y = nn.exp(x)
        self.assertEqual(y.value, np.exp(2))
        y.backward()
        self.assertEqual(x.grad, np.exp(2))

        x = np.random.rand(5, 3)
        p = nn.Parameter(x)
        y = nn.exp(p)
        self.assertTrue((y.value == np.exp(x)).all())
        y.backward(np.ones((5, 3)))
        self.assertTrue((p.grad == np.exp(x)).all())


if __name__ == '__main__':
    unittest.main()
