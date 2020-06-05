#!/usr/bin/env python3

import sys
sys.path.append('../..')
from srlife import materials, thermal

from thermalsol import ManufacturedSolution

import numpy as np

import matplotlib.pyplot as plt

problems = [
    #ManufacturedSolution(1, 
      #lambda t, r: t, 
      #lambda t, k, alpha, r: k/alpha * (r*0.0+1)),
    #ManufacturedSolution(1, 
      #lambda t, r: np.sin(t)*np.log(r), 
      #lambda t, k, alpha, r: k/alpha * np.log(r) * np.cos(t)),
    #ManufacturedSolution(1,
      #lambda t, r: np.sin(r),
      #lambda t, k, alpha, r: k * np.sin(r) - k/r*np.cos(r)),
    #ManufacturedSolution(2,
      #lambda t, r, th: t,
      #lambda t, k, alpha, r, th: k/alpha * (r*0.0+1)),
    #ManufacturedSolution(2,
      #lambda t, r, th: np.sin(r),
      #lambda t, k, alpha, r, th: k * np.sin(r) - k/r * np.cos(r)),
    # ManufacturedSolution(2,
      #lambda t, r, th: np.cos(th),
      #lambda t, k, alpha, r, th: k * np.cos(th) / r**2.0),
    #ManufacturedSolution(2,
      #lambda t, r, th: np.cos(th) / r,
      #lambda t, k, alpha, r, th: 0.0*r),
    #ManufacturedSolution(2,
      #lambda t, r, th: np.log(r) * np.sin(th) / (t+1),
      #lambda t, k, alpha, r, th: k*np.log(r)*np.sin(th)/((t+1)*r**2.0) - k/alpha * np.log(r) * np.sin(th) / (t+1)**2.0),
    ManufacturedSolution(3,
      lambda t, r, th, z: t,
      lambda t, k, alpha, r, th, z: k/alpha * (r*0.0+1)),
    ]


def run_with(solver, material, fluid):
  """
    Run the standard problems with the provided solver and material
  """
  for problem in problems:
    res = problem.solve(solver, material, fluid)
    problem.plot_comparison(res)
    plt.show()

if __name__ == "__main__":
  run_with(thermal.FiniteDifferenceImplicitThermalSolver(), 
      materials.ConstantThermalMaterial("Test", 10.0, 5.0),
      materials.ConstantFluidMaterial({"Test": 7.5}))
