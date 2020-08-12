import unittest
from prml import nn


class TestSigmoid(unittest.TestCase):

    def test_sigmoid(self):
        self.assertEqual(nn.sigmoid(0).value, 0.5)


if __name__ == '__main__':
    unittest.main()
