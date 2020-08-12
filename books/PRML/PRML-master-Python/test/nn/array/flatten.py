import unittest
import numpy as np
from prml import nn


class TestFlatten(unittest.TestCase):

    def test_flatten(self):
        self.assertRaises(TypeError, nn.flatten, "abc")
        self.assertRaises(ValueError, nn.flatten, np.ones(1))

        x = np.random.rand(5, 4)
        p = nn.Parameter(x)
        y = p.flatten()
        self.assertTrue((y.value == x.flatten()).all())
        y.backward(np.ones(20))
        self.assertTrue((p.grad == np.ones((5, 4))).all())


if __name__ == '__main__':
    unittest.main()
