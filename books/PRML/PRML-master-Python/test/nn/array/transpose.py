import unittest
import numpy as np
from prml import nn


class TestTranspose(unittest.TestCase):

    def test_transpose(self):
        arrays = [
            np.random.normal(size=(2, 3)),
            np.random.normal(size=(2, 3, 4))
        ]
        axes = [
            None,
            (2, 0, 1)
        ]

        for arr, ax in zip(arrays, axes):
            arr = nn.Parameter(arr)
            arr_t = nn.transpose(arr, ax)
            self.assertEqual(arr_t.shape, np.transpose(arr.value, ax).shape)
            da = np.random.normal(size=arr_t.shape)
            arr_t.backward(da)
            self.assertEqual(arr.grad.shape, arr.shape)


if __name__ == '__main__':
    unittest.main()
