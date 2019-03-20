import unittest
import numpy as np
from prml import nn


class TestMean(unittest.TestCase):

    def test_mean(self):
        x = np.random.rand(5, 1, 2)
        xp = nn.Parameter(x)
        z = xp.mean()
        self.assertEqual(z.value, x.mean())
        z.backward()
        self.assertTrue((xp.grad == np.ones((5, 1, 2)) / 10).all())
        xp.cleargrad()

        z = xp.mean(axis=0, keepdims=True)
        self.assertEqual(z.shape, (1, 1, 2))
        self.assertTrue((z.value == x.mean(axis=0, keepdims=True)).all())
        z.backward(np.ones((1, 1, 2)))
        self.assertTrue((xp.grad == np.ones((5, 1, 2)) / 5).all())


if __name__ == '__main__':
    unittest.main()
