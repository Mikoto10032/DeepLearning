import unittest
import numpy as np
from prml import bayesnet as bn


class TestDiscrete(unittest.TestCase):

    def test_discrete(self):
        a = bn.discrete([0.2, 0.8])
        b = bn.discrete([[0.1, 0.2], [0.9, 0.8]], a)
        self.assertTrue(np.allclose(b.proba, [0.18, 0.82]))
        a.observe(0)
        self.assertTrue(np.allclose(b.proba, [0.1, 0.9]))

        a = bn.discrete([0.1, 0.9])
        b = bn.discrete([[0.7, 0.2], [0.3, 0.8]], a)
        c = bn.discrete([[0.8, 0.4], [0.2, 0.6]], b)
        self.assertTrue(np.allclose(a.proba, [0.1, 0.9]))
        self.assertTrue(np.allclose(b.proba, [0.25, 0.75]))
        self.assertTrue(np.allclose(c.proba, [0.5, 0.5]))
        c.observe(0)
        self.assertTrue(np.allclose(a.proba, [0.136, 0.864]))
        self.assertTrue(np.allclose(b.proba, [0.4, 0.6]))
        self.assertTrue(np.allclose(c.proba, [1, 0]))

        a = bn.discrete([0.1, 0.9], name="p(a)")
        b = bn.discrete([0.1, 0.9], name="p(b)")
        c = bn.discrete(
            [[[0.9, 0.8],
              [0.8, 0.2]],
             [[0.1, 0.2],
              [0.2, 0.8]]],
            a, b, name="p(c|a,b)"
        )
        c.observe(0)
        self.assertTrue(np.allclose(a.proba, [0.25714286, 0.74285714]))
        b.observe(0)
        self.assertTrue(np.allclose(a.proba, [0.11111111, 0.88888888]))

        a = bn.discrete([0.1, 0.9], name="p(a)")
        b = bn.discrete([0.1, 0.9], name="p(b)")
        c = bn.discrete(
            [[[0.9, 0.8],
              [0.8, 0.2]],
             [[0.1, 0.2],
              [0.2, 0.8]]],
            a, b, name="p(c|a,b)"
        )
        c.observe(0, proprange=0)
        self.assertTrue(np.allclose(a.proba, [0.1, 0.9]))
        b.observe(0, proprange=1)
        self.assertTrue(np.allclose(a.proba, [0.1, 0.9]))
        b.send_message(proprange=2)
        self.assertTrue(np.allclose(a.proba, [0.11111111, 0.88888888]))
        a.send_message()
        c.send_message()
        self.assertTrue(np.allclose(a.proba, [0.11111111, 0.88888888]), a.message_from)

        a = bn.discrete([0.1, 0.9], name="p(a)")
        b = bn.discrete([0.1, 0.9], name="p(b)")
        c = bn.discrete(
            [[[0.9, 0.8],
              [0.8, 0.2]],
             [[0.1, 0.2],
              [0.2, 0.8]]],
            a, b, name="p(c|a,b)"
        )
        b.observe(0)
        self.assertTrue(np.allclose(a.proba, [0.1, 0.9]))
        c.send_message()
        self.assertTrue(np.allclose(a.proba, [0.1, 0.9]), a.message_from)

    def test_joint_discrete(self):
        a = bn.DiscreteVariable(2)
        b = bn.DiscreteVariable(2)
        bn.discrete([[0.1, 0.2], [0.3, 0.4]], out=[a, b])
        b.observe(1)
        self.assertTrue(np.allclose(a.proba, [1/3, 2/3]))


if __name__ == "__main__":
    unittest.main()
