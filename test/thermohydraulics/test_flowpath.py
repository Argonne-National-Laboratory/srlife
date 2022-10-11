import unittest
import tempfile

from absl import logging

logging.set_verbosity(logging.FATAL)

import numpy as np
import numpy.random as ra
from srlife.thermohydraulics import thermalfluid, flowpath


class TestBasicSetup(unittest.TestCase):
    def setUp(self):
        self.cp = np.array([0.1046, 944.622])
        self.rho = np.array([-0.522, 1903.7])
        self.mu = np.array([1.784e-8, -2.91e-5, 1.4965e-2])
        self.k = np.array([-0.0001, 0.5047])

        self.material = thermalfluid.PolynomialThermalFluidMaterial(
            self.cp, self.rho, self.mu, self.k
        )

        self.times = np.array([0.0, 1.0, 2.0])
        self.mass_flow = np.array([10.0, 100.0, 10.0])
        self.weights = np.array([2.0, 4.0, 2.0])
        self.ri = 25.4
        self.h = 1000.0
        self.ntime = len(self.times)
        self.ntube = len(self.weights)
        self.nt = 12
        self.nz = 14
        self.metal_temp = ra.random((self.ntime, self.ntube, self.nt, self.nz))

        self.panel1 = flowpath.SimplePanelLink(
            self.times,
            self.mass_flow,
            self.weights,
            self.ri,
            self.h,
            self.metal_temp,
            self.material,
        )

        self.t = 1.2

    def test_basic_balance(self):
        T_start = np.array([100.0])
        T_end = np.array([120.0, 140.0, 110.0])

        Q_mass = self.panel1.Q_mass(T_start, T_end, self.t)
        self.assertEqual(Q_mass.shape, (self.ntube,))

        Q_conv = self.panel1.Q_conv(T_start, T_end, self.t)
        self.assertEqual(Q_conv.shape, (self.ntube,))

        R = self.panel1.residual(T_start, T_end, self.t)
        self.assertTrue(np.allclose(R, Q_mass - Q_conv))


class TestIsothermal(unittest.TestCase):
    def setUp(self):
        self.cp = np.array([994.6])
        self.rho = np.array([1903.7])
        self.mu = np.array([1.5e-2])
        self.k = np.array([0.5])

        self.material = thermalfluid.PolynomialThermalFluidMaterial(
            self.cp, self.rho, self.mu, self.k
        )

        self.times = np.array([0.0, 1.0])
        self.mass_flow = np.array([0.0, 100.0])
        self.inlet_temperature = np.array([0.0, 50.0])

        self.ntime = len(self.times)
        self.nt = 12
        self.nz = 10

        self.weights_1 = np.array([1.0, 1.0, 1.0])
        self.ri_1 = 25.4
        self.h_1 = 1000.0
        self.metal_1 = np.zeros((self.ntime, len(self.weights_1), self.nt, self.nz))

        self.metal_1[1, 0] = 50.0
        self.metal_1[1, 1] = 50.0
        self.metal_1[1, 2] = 50.0

        self.model = flowpath.FlowPath(
            self.times, self.mass_flow, self.inlet_temperature
        )
        self.model.add_panel(
            self.weights_1, self.ri_1, self.h_1, self.metal_1, self.material
        )

    def test_no_heat_in(self):
        T1 = self.model.solve(1.0)
        self.assertTrue(np.isclose(T1[-1], self.inlet_temperature[-1]))


class TestAnalytic(unittest.TestCase):
    def setUp(self):
        self.cp = np.array([994.6])
        self.rho = np.array([1.9])
        self.mu = np.array([1.5e-6])
        self.k = np.array([0.5])

        self.material = thermalfluid.PolynomialThermalFluidMaterial(
            self.cp, self.rho, self.mu, self.k
        )

        self.times = np.array([0.0, 1.0])
        self.mass_flow = np.array([0.0, 10.0])
        self.inlet_temperature = np.array([0.0, 50.0])

        self.ntime = len(self.times)
        self.nt = 12
        self.nz = 10

        self.weights_1 = np.array([2.0, 10.0, 2.0])
        self.ri_1 = 100.0
        self.h_1 = 1000.0
        self.metal_1 = np.zeros((self.ntime, len(self.weights_1), self.nt, self.nz))

        self.metal_T = 100.0

        self.metal_1[1, 0] = self.metal_T
        self.metal_1[1, 1] = self.metal_T
        self.metal_1[1, 2] = self.metal_T

        self.model = flowpath.FlowPath(
            self.times, self.mass_flow, self.inlet_temperature
        )
        self.model.add_panel(
            self.weights_1, self.ri_1, self.h_1, self.metal_1, self.material
        )

    def test_no_heat_in(self):
        T = self.model.solve(1.0)
        Tnumer = T[-1]
        u = self.mass_flow[1] / (
            self.rho * np.sum(self.weights_1) * np.pi * self.ri_1**2.0
        )
        h_film = self.material.film_coefficient(
            np.array([100.0]), u, np.array([self.ri_1])
        )[0]

        mdot = self.mass_flow[-1]
        cp = self.cp[0]
        Ts = self.inlet_temperature[-1]
        r = self.ri_1
        h = self.h_1
        Tm = self.metal_T
        n = np.sum(self.weights_1)

        Tanal = (
            mdot * cp * Ts + n * 2.0 * np.pi * r * h * h_film * (Tm - 0.5 * Ts)
        ) / (mdot * cp + n * 2.0 * np.pi * r * h * h_film * 0.5)

        self.assertAlmostEqual(Tanal, Tnumer)
