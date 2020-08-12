import unittest
import numpy as np
from prml import nn


class TestGaussian(unittest.TestCase):

    def test_gaussian(self):
        self.assertRaises(ValueError, nn.random.Gaussian, 0, -1)
        self.assertRaises(ValueError, nn.random.Gaussian, 0, np.array([1, -1]))


if __name__ == '__main__':
    unittest.main()
