#!/usr/bin/env python3

import sys

sys.path.append("../../..")
from srlife import materials, thermal

from thermalsol import ManufacturedSolution

import numpy as np

problem = ManufacturedSolution(
    "test",
    3,
    lambda t, r, th, z: np.log(r) * np.sin(th) * np.cos(z) / (t + 1.0),
    lambda t, k, alpha, r, th, z: k
    * np.log(r)
    * np.sin(th)
    * np.cos(z)
    / (t + 1)
    * (1.0 + 1 / r**2.0 - 1.0 / (alpha * (t + 1))),
)


def run_with(solver, material, fluid):
    """
    Run the standard problems with the provided solver and material
    """
    for problem in problems:
        res = problem.solve(solver, material, fluid)
        problem.plot_comparison(res)
        plt.show()


if __name__ == "__main__":
    solver = thermal.FiniteDifferenceImplicitThermalSolver()
    tmat = materials.ConstantThermalMaterial("Test", 10.0, 5.0)
    fmat = materials.ConstantFluidMaterial({"Test": 7.5})

    if len(sys.argv) == 2:
        n = int(sys.argv[1])
    else:
        n = 1

    for i in range(n):
        res = problem.solve(solver, tmat, fmat)
