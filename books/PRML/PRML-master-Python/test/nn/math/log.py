import unittest
import numpy as np
from prml import nn


class TestLog(unittest.TestCase):

    def test_log(self):
        x = nn.Parameter(2.)
        y = nn.log(x)
        self.assertEqual(y.value, np.log(2))
        y.backward()
        self.assertEqual(x.grad, 0.5)

        x = np.random.rand(4, 6)
        p = nn.Parameter(x)
        y = nn.log(p)
        self.assertTrue((y.value == np.log(x)).all())
        y.backward(np.ones((4, 6)))
        self.assertTrue((p.grad == 1 / x).all())


if __name__ == '__main__':
    unittest.main()
