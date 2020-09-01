import unittest

import sys
sys.path.append("test/solvers/tube")
sys.path.append("../../..")
from srlife import structural

from run_structural_tube_verification import cases

class TestSimpleCases(unittest.TestCase):
  def setUp(self):
    self.atol = 1.0e-3
    self.rtol = 1.0e-3

    self.problems = cases
    self.solver = structural.PythonTubeSolver()

  def _check_case(self, d, case):
    tube = case.run_comparison(d, self.solver)
    a, r = case.evaluate_comparison(tube)

    self.assertTrue(a < self.atol or r < self.rtol)

  def test_all(self):
    for d in range(1,4):
      for c in self.problems:
        self._check_case(d, c)
