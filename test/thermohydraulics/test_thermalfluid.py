import unittest
import tempfile

import numpy as np
from srlife.thermohydraulics import thermalfluid

class TestPolynomialThermalFluidMaterial(unittest.TestCase):
    def setUp(self):
        self.cp = np.array([0.1046, 944.622])
        self.rho = np.array([-0.522, 1903.7])
        self.mu = np.array([1.784e-8, -2.91e-5, 1.4965e-2])
        self.k = np.array([-0.0001, 0.5047])

        self.model = thermalfluid.PolynomialThermalFluidMaterial(
            self.cp, self.rho, self.mu, self.k
        )
        self.T = np.array([100.0, 200.0, 300.0]) + 273.15
        self.u = np.array([1.0, 2.0, 3.0])
        self.r = np.array([5.0, 10.0, 15.0])

    def test_definitions(self):
        self.assertTrue(
            np.allclose(self.model.cp(self.T), np.polyval(self.cp, self.T - 273.15))
        )
        self.assertTrue(
            np.allclose(self.model.rho(self.T), np.polyval(self.rho, self.T - 273.15))
        )
        self.assertTrue(
            np.allclose(self.model.mu(self.T), np.polyval(self.mu, self.T - 273.15))
        )
        self.assertTrue(
            np.allclose(self.model.k(self.T), np.polyval(self.k, self.T - 273.15))
        )

    def test_recover(self):
        tfile = tempfile.mktemp()
        self.model.save(tfile, "whatever")

        recover = thermalfluid.ThermalFluidMaterial.load(tfile, "whatever")

        self.assertTrue(np.allclose(recover.cp_poly, self.model.cp_poly))
        self.assertTrue(np.allclose(recover.rho_poly, self.model.rho_poly))
        self.assertTrue(np.allclose(recover.mu_poly, self.model.mu_poly))
        self.assertTrue(np.allclose(recover.k_poly, self.model.k_poly))
