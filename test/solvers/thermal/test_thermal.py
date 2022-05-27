import unittest
from parameterized import parameterized

import numpy as np

from srlife import materials, thermal, receiver

import sys

sys.path.append("test/solvers/thermal")
from thermalsol import ManufacturedSolution


class TestThermalManufactured(unittest.TestCase):
    def setUp(self):
        self.solver = thermal.FiniteDifferenceImplicitThermalSolver()
        self.material = materials.ConstantThermalMaterial("Test", 10.0, 5.0)
        self.fluid = materials.ConstantFluidMaterial({"Test": 7.5})

        self.tol = 1e-2
        self.atol = 1e-3

    def _check_case(self, case):
        res = case.solve(self.solver, self.material, self.fluid)
        if not case.assess_comparison(res, self.tol, self.atol):
            print(case.name)
            self.assertTrue(False)

    @parameterized.expand(
        [
            (
                ManufacturedSolution(
                    "1D: no spatial",
                    1,
                    lambda t, r: t,
                    lambda t, k, alpha, r: k / alpha * (r * 0.0 + 1),
                ),
            ),
            (
                ManufacturedSolution(
                    "1D: spatial and time",
                    1,
                    lambda t, r: np.sin(t) * np.log(r),
                    lambda t, k, alpha, r: k / alpha * np.log(r) * np.cos(t),
                ),
            ),
            (
                ManufacturedSolution(
                    "1D: just spatial",
                    1,
                    lambda t, r: np.sin(r),
                    lambda t, k, alpha, r: k * np.sin(r) - k / r * np.cos(r),
                ),
            ),
            (
                ManufacturedSolution(
                    "1D: steady",
                    1,
                    lambda t, r: np.sin(r),
                    lambda t, k, alpha, r: k * np.sin(r) - k / r * np.cos(r),
                    steady=True,
                ),
            ),
            (
                ManufacturedSolution(
                    "2D: just time",
                    2,
                    lambda t, r, th: t,
                    lambda t, k, alpha, r, th: k / alpha * (r * 0.0 + 1),
                ),
            ),
            (
                ManufacturedSolution(
                    "2D: just r",
                    2,
                    lambda t, r, th: np.sin(r),
                    lambda t, k, alpha, r, th: k * np.sin(r) - k / r * np.cos(r),
                ),
            ),
            (
                ManufacturedSolution(
                    "2D: just theta",
                    2,
                    lambda t, r, th: np.cos(th),
                    lambda t, k, alpha, r, th: k * np.cos(th) / r**2.0,
                ),
            ),
            (
                ManufacturedSolution(
                    "2D: theta and r",
                    2,
                    lambda t, r, th: np.cos(th) / r,
                    lambda t, k, alpha, r, th: -k * np.cos(th) / r**3.0,
                ),
            ),
            (
                ManufacturedSolution(
                    "2D: steady",
                    2,
                    lambda t, r, th: np.cos(th) / r,
                    lambda t, k, alpha, r, th: -k * np.cos(th) / r**3.0,
                    steady=True,
                ),
            ),
            (
                ManufacturedSolution(
                    "2D: all three",
                    2,
                    lambda t, r, th: np.log(r) * np.sin(th) / (t + 1),
                    lambda t, k, alpha, r, th: k
                    * np.log(r)
                    * np.sin(th)
                    / ((t + 1) * r**2.0)
                    - k / alpha * np.log(r) * np.sin(th) / (t + 1) ** 2.0,
                ),
            ),
            (
                ManufacturedSolution(
                    "3D: just t",
                    3,
                    lambda t, r, th, z: t,
                    lambda t, k, alpha, r, th, z: k / alpha * (r * 0.0 + 1),
                ),
            ),
            (
                ManufacturedSolution(
                    "3D: just r",
                    3,
                    lambda t, r, th, z: np.sin(r),
                    lambda t, k, alpha, r, th, z: k * np.sin(r) - k / r * np.cos(r),
                ),
            ),
            (
                ManufacturedSolution(
                    "3D: just theta",
                    3,
                    lambda t, r, th, z: np.cos(th),
                    lambda t, k, alpha, r, th, z: k * np.cos(th) / r**2.0,
                ),
            ),
            (
                ManufacturedSolution(
                    "3D: just z",
                    3,
                    lambda t, r, th, z: np.sin(z),
                    lambda t, k, alpha, r, th, z: k * np.sin(z),
                ),
            ),
            (
                ManufacturedSolution(
                    "3D: all spatial",
                    3,
                    lambda t, r, th, z: z**2.0 * np.cos(th) / r,
                    lambda t, k, alpha, r, th, z: -k
                    * np.cos(th)
                    / r
                    * ((z / r) ** 2.0 + 2),
                ),
            ),
            (
                ManufacturedSolution(
                    "3D: steady",
                    3,
                    lambda t, r, th, z: z**2.0 * np.cos(th) / r,
                    lambda t, k, alpha, r, th, z: -k
                    * np.cos(th)
                    / r
                    * ((z / r) ** 2.0 + 2),
                    steady=True,
                ),
            ),
            (
                ManufacturedSolution(
                    "3D: everything",
                    3,
                    lambda t, r, th, z: np.log(r) * np.sin(th) * np.cos(z) / (t + 1.0),
                    lambda t, k, alpha, r, th, z: k
                    * np.log(r)
                    * np.sin(th)
                    * np.cos(z)
                    / (t + 1)
                    * (1.0 + 1 / r**2.0 - 1.0 / (alpha * (t + 1))),
                ),
            ),
        ]
    )
    def test_case(self, case):
        self._check_case(case)


class TestThermalBCs(unittest.TestCase):
    def setUp(self):
        self.k = 15.0
        self.a = 5.0
        self.hcoef = 1.0

        self.solver = thermal.FiniteDifferenceImplicitThermalSolver()
        self.material = materials.ConstantThermalMaterial("Test", self.k, self.a)
        self.fluid = materials.ConstantFluidMaterial({"Test": self.hcoef})

        self.t = 0.2
        self.r = 2.0
        self.h = 1.0
        self.nr = 10

        self.nt = 200
        self.nz = 2

        self.tmax = 1000.0
        self.ntime = 10

        self.tube = receiver.Tube(self.r, self.t, self.h, self.nr, self.nt, self.nz, 0)

        self.times = np.linspace(0, self.tmax, self.ntime + 1)
        self.tube.set_times(self.times)

        self.tube.make_1D(self.tube.h / 2, 0)

        self.ttimes, self.thetas, self.zs = np.meshgrid(
            self.times,
            np.linspace(0, 2 * np.pi, self.nt + 1)[:-1],
            np.linspace(0, self.h, self.nz),
            indexing="ij",
        )

        self.rs = np.linspace(self.r - self.t, self.r, self.nr)

    def test_dirichlet(self):
        T0 = lambda x: np.sin(2 * np.pi * (x - self.r - self.t) / self.t)

        Tleft = receiver.FixedTempBC(
            self.r - self.t,
            self.h,
            self.nt,
            self.nz,
            self.times,
            np.zeros(self.ttimes.shape),
        )
        self.tube.set_bc(Tleft, "inner")

        Tright = receiver.FixedTempBC(
            self.r, self.h, self.nt, self.nz, self.times, np.zeros(self.ttimes.shape)
        )
        self.tube.set_bc(Tright, "outer")

        self.solver.solve(self.tube, self.material, self.fluid, T0=T0)

        # Correct solution: temperatures head towards zero
        T = self.tube.results["temperature"][-1]
        self.assertTrue(np.allclose(T, 0))

    def test_neumann_left(self):
        T0 = lambda x: np.sin(2 * np.pi * (x - self.r - self.t) / self.t)
        q = 10.0

        Tleft = receiver.HeatFluxBC(
            self.r - self.t,
            self.h,
            self.nt,
            self.nz,
            self.times,
            np.ones(self.ttimes.shape) * q,
        )
        self.tube.set_bc(Tleft, "inner")

        Tright = receiver.FixedTempBC(
            self.r, self.h, self.nt, self.nz, self.times, np.zeros(self.ttimes.shape)
        )
        self.tube.set_bc(Tright, "outer")

        self.solver.solve(self.tube, self.material, self.fluid, T0=T0)

        # Correct solution:
        Tright = -q * (self.r - self.t) / self.k * np.log(self.rs) + q * (
            self.r - self.t
        ) / self.k * np.log(self.r)

        T = self.tube.results["temperature"][-1]

        self.assertTrue(np.allclose(T, Tright, rtol=1e-2))

    def test_neumann_right(self):
        T0 = lambda x: np.sin(2 * np.pi * (x - self.r - self.t) / self.t)
        q = 10.0

        Tright = receiver.HeatFluxBC(
            self.r, self.h, self.nt, self.nz, self.times, np.ones(self.ttimes.shape) * q
        )
        self.tube.set_bc(Tright, "outer")

        Tleft = receiver.FixedTempBC(
            self.r - self.t,
            self.h,
            self.nt,
            self.nz,
            self.times,
            np.zeros(self.ttimes.shape),
        )
        self.tube.set_bc(Tleft, "inner")

        self.solver.solve(self.tube, self.material, self.fluid, T0=T0)

        # Correct solution:
        Tright = q * self.r / self.k * np.log(self.rs) - q * self.r / self.k * np.log(
            self.r - self.t
        )

        T = self.tube.results["temperature"][-1]

        self.assertTrue(np.allclose(T, Tright, rtol=1e-2))

    def test_convection_left(self):
        T0 = lambda x: np.sin(2 * np.pi * (x - self.r - self.t) / self.t)

        T_inner = 25

        Tin = receiver.ConvectiveBC(
            self.r - self.t,
            self.h,
            self.nz,
            self.times,
            np.ones((self.ntime + 1, self.nz)) * T_inner,
        )
        self.tube.set_bc(Tin, "inner")

        Tright = receiver.FixedTempBC(
            self.r, self.h, self.nt, self.nz, self.times, np.zeros(self.ttimes.shape)
        )
        self.tube.set_bc(Tright, "outer")

        self.solver.solve(self.tube, self.material, self.fluid, T0=T0)

        T = self.tube.results["temperature"][-1]

        ri = self.r - self.t

        C1 = (
            T_inner
            * self.hcoef
            / (self.hcoef * np.log(ri) - self.hcoef * np.log(self.r) - self.k / ri)
        )
        C2 = -C1 * np.log(self.r)

        Texact = C1 * np.log(self.rs) + C2

        self.assertTrue(np.allclose(T, Texact, rtol=1.0e-2))

    def test_convection_right(self):
        T0 = lambda x: np.sin(2 * np.pi * (x - self.r - self.t) / self.t)

        T_outer = 25

        Tout = receiver.ConvectiveBC(
            self.r,
            self.h,
            self.nz,
            self.times,
            np.ones((self.ntime + 1, self.nz)) * T_outer,
        )
        self.tube.set_bc(Tout, "outer")

        Tleft = receiver.FixedTempBC(
            self.r - self.t,
            self.h,
            self.nt,
            self.nz,
            self.times,
            np.zeros(self.ttimes.shape),
        )
        self.tube.set_bc(Tleft, "inner")

        self.solver.solve(self.tube, self.material, self.fluid, T0=T0)

        T = self.tube.results["temperature"][-1]

        ri = self.r - self.t

        C1 = (
            self.hcoef
            * T_outer
            / (self.k / self.r + self.hcoef * np.log(self.r) - self.hcoef * np.log(ri))
        )
        C2 = -C1 * np.log(ri)

        Texact = C1 * np.log(self.rs) + C2

        self.assertTrue(np.allclose(T, Texact, rtol=1.0e-2))

    def test_substep(self):
        T0 = lambda x: np.sin(2 * np.pi * (x - self.r - self.t) / self.t)

        T_outer = 250

        Tout = receiver.ConvectiveBC(
            self.r,
            self.h,
            self.nz,
            self.times,
            np.ones((self.ntime + 1, self.nz)) * T_outer,
        )
        self.tube.set_bc(Tout, "outer")

        Tleft = receiver.FixedTempBC(
            self.r - self.t,
            self.h,
            self.nt,
            self.nz,
            self.times,
            np.zeros(self.ttimes.shape),
        )
        self.tube.set_bc(Tleft, "inner")

        self.solver.solve(self.tube, self.material, self.fluid, T0=T0, substep=10)

        T = self.tube.results["temperature"][-1]

        ri = self.r - self.t

        C1 = (
            self.hcoef
            * T_outer
            / (self.k / self.r + self.hcoef * np.log(self.r) - self.hcoef * np.log(ri))
        )
        C2 = -C1 * np.log(ri)

        Texact = C1 * np.log(self.rs) + C2

        self.assertTrue(np.allclose(T, Texact, rtol=1.0e-2))


class TestFunction(unittest.TestCase):
    def setUp(self):
        self.k = 15.0
        self.a = 5.0
        self.hcoef = 1.0

        self.solver = thermal.FiniteDifferenceImplicitThermalSolver()
        self.material = materials.ConstantThermalMaterial("Test", self.k, self.a)
        self.fluid = materials.ConstantFluidMaterial({"Test": self.hcoef})

        self.t = 0.2
        self.r = 2.0
        self.h = 1.0

        self.nr = 5

        self.nt = 10
        self.nz = 5

        self.tmax = 10.0
        self.ntime = 10

        self.tube = receiver.Tube(self.r, self.t, self.h, self.nr, self.nt, self.nz, 0)

        self.times = np.linspace(0, self.tmax, self.ntime + 1)
        self.tube.set_times(self.times)

        self.T_inner = 100

        Tin = receiver.ConvectiveBC(
            self.r - self.t,
            self.h,
            self.nz,
            self.times,
            np.ones((self.ntime + 1, self.nz)) * self.T_inner,
        )
        self.tube.set_bc(Tin, "inner")

        Tright = receiver.HeatFluxBC(
            self.r,
            self.h,
            self.nt,
            self.nz,
            self.times,
            np.zeros((self.ntime + 1, self.nt, self.nz)),
        )
        self.tube.set_bc(Tright, "outer")

    def test_1D(self):
        self.tube.make_1D(self.tube.h / 2, 0)
        self.solver.solve(self.tube, self.material, self.fluid)

    def test_2D(self):
        self.tube.make_2D(self.tube.h / 2)
        self.solver.solve(self.tube, self.material, self.fluid)

    def test_3D(self):
        self.solver.solve(self.tube, self.material, self.fluid)
