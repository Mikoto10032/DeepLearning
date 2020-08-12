import unittest
import numpy as np
from prml import nn


class TestTrace(unittest.TestCase):

    def test_trace(self):
        arrays = [
            np.random.normal(size=(2, 2)),
            np.random.normal(size=(3, 4))
        ]

        for arr in arrays:
            arr = nn.Parameter(arr)
            tr_arr = nn.linalg.trace(arr)
            self.assertEqual(tr_arr.value, np.trace(arr.value))

        a = np.array([
            [1.5, 0],
            [-0.1, 1.1]
        ])
        a = nn.Parameter(a)
        for _ in range(100):
            a.cleargrad()
            loss = nn.square(nn.linalg.trace(a) - 2)
            loss.backward()
            a.value -= 0.1 * a.grad
        self.assertEqual(nn.linalg.trace(a).value, 2)


if __name__ == '__main__':
    unittest.main()
