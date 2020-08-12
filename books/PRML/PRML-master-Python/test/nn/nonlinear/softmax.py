import unittest
import numpy as np
from prml import nn


class TestSoftmax(unittest.TestCase):

    def test_softmax(self):
        self.assertTrue(np.allclose(nn.softmax(np.ones(4)).value, 0.25))


if __name__ == '__main__':
    unittest.main()
