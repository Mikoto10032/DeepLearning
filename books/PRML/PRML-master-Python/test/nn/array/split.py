import unittest
import numpy as np
from prml import nn


class TestSplit(unittest.TestCase):

    def test_split(self):
        x = np.random.rand(10, 7)
        a = nn.Parameter(x)
        b, c = nn.split(a, (3,), axis=-1)
        self.assertTrue((b.value == x[:, :3]).all())
        self.assertTrue((c.value == x[:, 3:]).all())
        b.backward(np.ones((10, 3)))
        self.assertIs(a.grad, None)
        c.backward(np.ones((10, 4)))
        self.assertTrue((a.grad == np.ones((10, 7))).all())


if __name__ == '__main__':
    unittest.main()
