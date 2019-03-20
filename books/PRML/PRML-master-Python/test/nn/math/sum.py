import unittest
import numpy as np
from prml import nn


class TestSum(unittest.TestCase):

    def test_sum(self):
        x = np.random.rand(5, 1, 2)
        xp = nn.Parameter(x)
        z = xp.sum()
        self.assertEqual(z.value, x.sum())
        z.backward()
        self.assertTrue((xp.grad == np.ones((5, 1, 2))).all())
        xp.cleargrad()

        z = xp.sum(axis=0, keepdims=True)
        self.assertEqual(z.shape, (1, 1, 2))
        self.assertTrue((z.value == x.sum(axis=0, keepdims=True)).all())
        z.backward(np.ones((1, 1, 2)))
        self.assertTrue((xp.grad == np.ones((5, 1, 2))).all())


if __name__ == '__main__':
    unittest.main()
