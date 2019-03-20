import unittest
import numpy as np
from prml import nn


class TestSoftplus(unittest.TestCase):

    def test_softplus(self):
        self.assertEqual(nn.softplus(0).value, np.log(2))


if __name__ == '__main__':
    unittest.main()
