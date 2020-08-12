import unittest
import numpy as np
from prml import nn


class TestSqrt(unittest.TestCase):

    def test_sqrt(self):
        x = nn.Parameter(2.)
        y = nn.square(x)
        self.assertEqual(y.value, 4)
        y.backward()
        self.assertEqual(x.grad, 4)


if __name__ == '__main__':
    unittest.main()
