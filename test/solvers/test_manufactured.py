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
          lambda t, k, alpha, r: k * np.sin(r) - k/r*np.cos(r)),
        ManufacturedSolution(2,
            lambda t, r, th: t,
            lambda t, k, alpha, r, th: k/alpha * (r*0.0+1)),
        ManufacturedSolution(2,
          lambda t, r, th: np.sin(r),
          lambda t, k, alpha, r, th: k * np.sin(r) - k/r * np.cos(r)),
        ManufacturedSolution(2,
          lambda t, r, th: np.cos(th),
          lambda t, k, alpha, r, th: k * np.cos(th) / r**2.0),
        ManufacturedSolution(2,
          lambda t, r, th: np.cos(th) / r,
          lambda t, k, alpha, r, th: 0.0*r),
        ManufacturedSolution(2,
          lambda t, r, th: np.log(r) * np.sin(th) / (t+1),
          lambda t, k, alpha, r, th: k*np.log(r)*np.sin(th)/((t+1)*r**2.0) - k/alpha * np.log(r) * np.sin(th) / (t+1)**2.0),
        ManufacturedSolution(3,
          lambda t, r, th, z: t,
          lambda t, k, alpha, r, th, z: k/alpha * (r*0.0+1)),
        ]

    self.solver = thermal.FiniteDifferenceImplicitThermalSolver()
    self.material = materials.ConstantThermalMaterial("Test", 10.0, 5.0)
    self.fluid = materials.ConstantFluidMaterial({"Test": 7.5})

    self.tol = 1e-2
    self.atol = 1e-1

  def _check_case(self, case):
    res = case.solve(self.solver, self.material, self.fluid)
    self.assertTrue(case.assess_comparison(res, self.tol, self.atol))

  def test_all_problems(self):
    for i,case in enumerate(self.problems):
      print(i)
      self._check_case(case)
