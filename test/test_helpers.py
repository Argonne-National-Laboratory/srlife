import unittest

import numpy as np
import numpy.random as ra

from srlife import helpers

class TestRandom(unittest.TestCase):
  def test_ms2ts(self):
    C = ra.random((6,6))
    C1 = helpers.ms2ts(C)
    C2 = helpers.ms2ts_faster(C)

    self.assertTrue(np.allclose(C1,C2))

  def test_usym(self):
    M = ra.random((6,))
    M1 = helpers.usym(M)
    M2 = helpers.usym_faster(M)

    self.assertTrue(np.allclose(M1,M2))

  def test_sym(self):
    T = ra.random((3,3))
    T = 0.5*(T+T.T)

    M1 = helpers.sym(T)
    M2 = helpers.sym_faster(T)

    self.assertTrue(np.allclose(M1,M2))
