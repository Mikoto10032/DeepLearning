import unittest
import numpy as np
from prml import nn
from prml.nn.array.broadcast import broadcast_to


class TestBroadcastTo(unittest.TestCase):

    def test_broadcast(self):
        x = nn.Parameter(np.ones((1, 1)))
        shape = (5, 2, 3)
        y = broadcast_to(x, shape)
        self.assertEqual(y.shape, shape)
        y.backward(np.ones(shape))
        self.assertTrue((x.grad == np.ones((1, 1)) * 30).all())


if __name__ == '__main__':
    unittest.main()
