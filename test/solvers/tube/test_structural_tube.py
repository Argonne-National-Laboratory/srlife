import unittest

import os.path

# Shut up matplotlib
import logging

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

import numpy as np

import sys

sys.path.append("test/solvers/tube")
sys.path.append("../../..")
from srlife import structural, receiver

from neml import elasticity, models, parse

from run_structural_tube_verification import cases, do_complete_comparison


def differentiate(fn, x0, eps=1.5e-6):
    if np.isscalar(x0):
        sx = (1,)
    else:
        sx = x0.shape

    f0 = fn(x0)

    if np.isscalar(f0):
        sf = (1,)
    else:
        sf = f0.shape

    tshape = sf + sx
    rshape = sx + sf

    D = np.zeros(rshape)

    for index in np.ndindex(sx):
        if np.isscalar(x0):
            diff = np.abs(x0 * eps)
        else:
            diff = np.abs(x0[index] * eps)
        if diff < eps:
            diff = eps

        if np.isscalar(x0):
            xp = diff
        else:
            xp = np.zeros(sx)
            xp[index] = diff

        fp = fn(xp + x0)

        D[index] = (fp - f0) / diff

    Df = np.zeros(tshape)
    # Reverse
    for ind1 in np.ndindex(sx):
        for ind2 in np.ndindex(sf):
            Df[ind2 + ind1] = D[ind1 + ind2]

    if np.isscalar(x0):
        return Df[0][0]
    else:
        return Df


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
        for d in range(1, 4):
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
    # def test_3D(self):
    #  a, r = do_complete_comparison(3, self.solver)
    #  self.assertTrue(a < self.atol)


class TestAxialStiffnessExact(unittest.TestCase):
    def setUp(self):
        self.ro = 5
        self.ri = 4.5
        self.h = 2.5

        self.nr = 10
        self.nt = 20
        self.nz = 5

        self.E = 150000.0
        self.nu = 0.35

        self.tube = receiver.Tube(
            self.ro, self.ro - self.ri, self.h, self.nr, self.nt, self.nz
        )
        self.times = np.array([0, 1])
        self.tube.set_times(self.times)

        self.tube.set_pressure_bc(receiver.PressureBC(self.times, self.times * 0))

        self.emodel = elasticity.IsotropicLinearElasticModel(
            self.E, "youngs", self.nu, "poissons"
        )
        self.mat = models.SmallStrainElasticity(self.emodel)

        self.d = 0.25

        self.force_exact = (
            np.pi * (self.ro**2.0 - self.ri**2.0) * self.E * self.d / self.h
        )
        self.stiffness_exact = (
            np.pi * (self.ro**2.0 - self.ri**2.0) * self.E / self.h
        )

        self.solver = structural.PythonTubeSolver(verbose=False)

    def _solve(self, d):
        self.solver.setup_tube(self.tube)
        state_n = self.solver.init_state(self.tube, self.mat)

        return self.solver.solve(self.tube, 1, state_n, d)

    def test_1D(self):
        self.tube.make_1D(self.h / 2, 0)

        state = self._solve(self.d)

        self.assertAlmostEqual(self.force_exact, state.force, places=3)
        self.assertAlmostEqual(self.stiffness_exact, state.stiffness, places=3)

    def test_2D(self):
        self.tube.make_2D(self.h / 2)

        state = self._solve(self.d)

        cf = np.sum(state.basis.dx) / (np.pi * (self.ro**2.0 - self.ri**2.0))

        self.assertAlmostEqual(self.force_exact, state.force / cf, places=3)
        self.assertAlmostEqual(self.stiffness_exact, state.stiffness / cf, places=3)

    def test_3D(self):
        state = self._solve(self.d)

        cf = (
            np.sum(state.basis.dx)
            / self.h
            / (np.pi * (self.ro**2.0 - self.ri**2.0))
        )

        self.assertAlmostEqual(self.force_exact, state.force / cf, places=3)
        self.assertAlmostEqual(self.stiffness_exact, state.stiffness / cf, places=3)


class TestAxialStiffnessNumerical(unittest.TestCase):
    def setUp(self):
        self.ro = 5
        self.ri = 4.5
        self.h = 2.5

        self.nr = 10
        self.nt = 20
        self.nz = 5

        self.E = 150000.0
        self.nu = 0.35

        self.tube = receiver.Tube(
            self.ro, self.ro - self.ri, self.h, self.nr, self.nt, self.nz
        )
        self.times = np.array([0, 1])
        self.tube.set_times(self.times)

        self.tube.set_pressure_bc(receiver.PressureBC(self.times, self.times / 50.0))

        self.mat = parse.parse_xml(
            os.path.join(os.path.dirname(__file__), "moose-verification", "model.xml"),
            "creeping",
        )

        self.d = 0.25

        self.solver = structural.PythonTubeSolver(verbose=False)

    def _solve(self, d):
        self.solver.setup_tube(self.tube)
        state_n = self.solver.init_state(self.tube, self.mat)

        return self.solver.solve(self.tube, 1, state_n, d)

    def test_1D(self):
        self.tube.make_1D(self.h / 2, 0)

        exact = self._solve(self.d).stiffness
        numerical = differentiate(lambda d: self._solve(d).force, self.d)

        self.assertTrue(np.isclose(exact, numerical, rtol=1e-4))

    def test_2D(self):
        self.tube.make_2D(self.h / 2)

        exact = self._solve(self.d).stiffness
        numerical = differentiate(lambda d: self._solve(d).force, self.d)

        self.assertTrue(np.isclose(exact, numerical, rtol=1e-4))

    # Too heavy
    def test_3D(self):
        exact = self._solve(self.d).stiffness
        numerical = differentiate(lambda d: self._solve(d).force, self.d)

        self.assertTrue(np.isclose(exact, numerical, rtol=1e-4))
