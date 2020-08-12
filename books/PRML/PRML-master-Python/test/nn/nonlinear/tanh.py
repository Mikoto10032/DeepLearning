import unittest
from prml import nn


class TestTanh(unittest.TestCase):

    def test_tanh(self):
        self.assertEqual(nn.tanh(0).value, 0)


if __name__ == '__main__':
    unittest.main()
