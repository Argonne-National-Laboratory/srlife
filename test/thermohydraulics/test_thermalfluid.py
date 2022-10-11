import unittest
import tempfile

import numpy as np
from srlife.thermohydraulics import thermalfluid


class TestPolynomialThermalFluidMaterial(unittest.TestCase):
    def setUp(self):
        self.cp = np.array([0.0000290556, 0.254458])
        self.rho = np.array([-5.52e-10, 2.0544788e-6])
        self.mu = np.array([6.4224e-8, -0.000139846, 0.087281])
        self.k = np.array([-1e-7, 0.000532015])

        self.model = thermalfluid.PolynomialThermalFluidMaterial(
            self.cp, self.rho, self.mu, self.k
        )
        self.T = np.array([100.0, 200.0, 300.0]) + 273.15
        self.u = np.array([1.0, 2.0, 3.0])
        self.r = np.array([5.0, 10.0, 15.0])

    def test_definitions(self):
        self.assertTrue(np.allclose(self.model.cp(self.T), np.polyval(self.cp, self.T)))
        self.assertTrue(
            np.allclose(self.model.rho(self.T), np.polyval(self.rho, self.T))
        )
        self.assertTrue(np.allclose(self.model.mu(self.T), np.polyval(self.mu, self.T)))
        self.assertTrue(np.allclose(self.model.k(self.T), np.polyval(self.k, self.T)))

    def test_recover(self):
        tfile = tempfile.mktemp()
        self.model.save(tfile, "whatever")

        recover = thermalfluid.ThermalFluidMaterial.load(tfile, "whatever")

        self.assertTrue(np.allclose(recover.cp_poly, self.model.cp_poly))
        self.assertTrue(np.allclose(recover.rho_poly, self.model.rho_poly))
        self.assertTrue(np.allclose(recover.mu_poly, self.model.mu_poly))
        self.assertTrue(np.allclose(recover.k_poly, self.model.k_poly))

        self.assertTrue(np.isclose(recover.film_min, self.model.film_min))
        self.assertTrue(np.isclose(recover.T_max, self.model.T_max))
        self.assertTrue(np.isclose(recover.T_min, self.model.T_min))
        self.assertTrue(np.isclose(recover.laminar_cutoff, self.model.laminar_cutoff))
        self.assertTrue(np.isclose(recover.laminar_value, self.model.laminar_value))
