import unittest

import numpy as np

import sys
sys.path.append("test/solvers/tube")
sys.path.append("../../..")
from srlife import structural, receiver

from run_structural_tube_verification import cases, do_complete_comparison

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

class TestMOOSEComparison(unittest.TestCase):
  def setUp(self):
    self.solver = structural.PythonTubeSolver()
    self.atol = 1.0e-3

  def test_1D(self):
    a, r = do_complete_comparison(1, self.solver)
    self.assertTrue(a < self.atol)

  def test_2D(self):
    a, r = do_complete_comparison(2, self.solver)
    self.assertTrue(a < self.atol)
  
  # Too heavy right now
  #def test_3D(self):
  #  a, r = do_complete_comparison(3, self.solver)
  #  self.assertTrue(a < self.atol)

