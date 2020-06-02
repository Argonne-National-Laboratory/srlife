import unittest

import numpy as np

from srlife import materials, thermal

import sys
sys.path.append("test/solvers")
from thermalsol import ManufacturedSolution

class TestThermal(unittest.TestCase):
  def setUp(self):
    self.problems = [
      ManufacturedSolution(1, 
        lambda t, r: t, 
        lambda t, k, alpha, r: k/alpha * (r*0.0+1)),
      ManufacturedSolution(1, 
        lambda t, r: np.sin(t)*np.log(r), 
        lambda t, k, alpha, r: k/alpha * np.log(r) * np.cos(t)),
      ManufacturedSolution(1,
        lambda t, r: np.sin(r),
        lambda t, k, alpha, r: k * np.sin(r) - k/r*np.cos(r))
      ]

    self.solver = thermal.FiniteDifferenceImplicitThermalSolver()
    self.material = materials.ConstantThermalMaterial("Test", 10.0, 5.0)
    self.fluid = materials.ConstantFluidMaterial({"Test": 7.5})

    self.tol = 1e-3

  def _check_case(self, case):
    res = case.solve(self.solver, self.material, self.fluid)
    self.assertTrue(case.assess_comparison(res, self.tol))

  def test_all_problems(self):
    for case in self.problems:
      self._check_case(case)
