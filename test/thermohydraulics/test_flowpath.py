import unittest
import tempfile

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
        self.mass_flow = np.array([10.0,100.0,10.0])
        self.weights = np.array([2.0, 4.0, 2.0])
        self.ri = 25.4
        self.h = 1000.0
        self.ntime = len(self.times)
        self.ntube = len(self.weights)
        self.nt = 12
        self.nz = 14
        self.metal_temp = ra.random((self.ntime,self.ntube,self.nt,self.nz))

        self.panel1 = flowpath.SimplePanelLink(self.times, 
                self.mass_flow, self.weights, self.ri, self.h, 
                self.metal_temp, self.material)

        self.t = 1.2

    def test_basic_balance(self):
        T_start = np.array([100.0])
        T_end = np.array([120.0,140.0,110.0])
        
        Q_mass = self.panel1.Q_mass(T_start, T_end, self.t)
        self.assertEqual(Q_mass.shape, (self.ntube,))

        Q_conv = self.panel1.Q_conv(T_start, T_end, self.t)
        self.assertEqual(Q_conv.shape, (self.ntube,))

        R = self.panel1.residual(T_start, T_end, self.t)
        self.assertTrue(np.allclose(R, Q_mass + Q_conv))

class TestSimpleFlow(unittest.TestCase):
    def setUp(self):
        self.cp = np.array([994.6])
        self.rho = np.array([1903.7])
        self.mu = np.array([1.5e-2])
        self.k = np.array([0.5])

        self.material = thermalfluid.PolynomialThermalFluidMaterial(
                self.cp, self.rho, self.mu, self.k)
        
        self.times = np.array([0.0, 1.0])
        self.mass_flow = np.array([0.0, 100.0])
        self.inlet_temperature = np.array([0.0, 50.0])

        self.ntime = len(self.times)
        self.nt = 12
        self.nz = 10

        self.weights_1 = np.array([2.0, 4.0, 2.0])
        self.ri_1 = 25.4
        self.h_1 = 1000.0
        self.metal_1 = np.zeros((self.ntime,len(self.weights_1),self.nt,self.nz))
        self.metal_1[1,0] = 100.0
        self.metal_1[1,1] = 150.0
        self.metal_1[1,2] = 200.0

        self.weights_2 = np.array([1.0, 3.0, 5.0, 1.0])
        self.ri_2 = 50.0
        self.h_2 = 1500.0
        self.metal_2 = np.zeros((self.ntime, len(self.weights_2), self.nt, 
            self.nz))
        self.metal_2[1,0] = 200.0
        self.metal_2[1,1] = 75.0
        self.metal_2[1,2] = 100.0
        self.metal_2[1,3] = 150.0

        self.model = flowpath.FlowPath(self.times, self.mass_flow,
                self.inlet_temperature)

        self.model.add_panel(self.weights_1, self.ri_1, 
                self.h_1, self.metal_1, self.material)
        self.model.add_panel(self.weights_2, self.ri_2, 
                self.h_2, self.metal_2, self.material)

    def test_solutions(self):
        T1 = self.model.solve(1.0)


