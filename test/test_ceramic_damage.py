import unittest

import numpy as np

from srlife import materials, damage, solverparams

class TestPIAModel(unittest.TestCase):
  def setUp(self):

    self.stress = np.array([[
     [100.0,25.0,50.0,0,0,0],[-1.5,-25.0,-5.6,0,0,0]],
     [[15.0,6.0,20.0,0,0,0],[-15,6,-20,0,0,0]]])
    self.temperatures = np.array([[1.0,3.0],[100.0,10.0]])
    self.volumes = np.array([[0.1,0.1],[0.1,0.1]])

    self.s0 = 70
    self.m = 3.5
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.PIAModel({})

  def test_definition(self):
    k = self.s0**(-self.m)

    mod_stress = self.stress
    mod_stress[mod_stress<0] = 0

    should = -k  * np.sum(mod_stress**self.m, axis = -1) * self.volumes
    
    actual = self.model.calculate_element_log_reliability(self.stress,
        self.temperatures, self.volumes, self.material)

    R_PIA = np.exp(np.sum(actual))
    Pf_PIA = 1 - np.exp(np.sum(actual))
    print("Reliability PIA = ",R_PIA)
    print("Probability of failure PIA = ",Pf_PIA)

    self.assertTrue(np.allclose(should, actual))

class TestWNTSAModel(unittest.TestCase):
  def setUp(self):

    # Case 1: Testing for a stress tensor with only principal components, over two time steps
    self.stress = np.array([[
     [100.0,25.0,50.0,0,0,0],[-1.5,-25.0,-5.6,0,0,0]],
     [[15.0,6.0,20.0,0,0,0],[-15,6,-20,0,0,0]]])
    self.temperatures = np.array([[1.0,3.0],[100.0,10.0]])
    self.volumes = np.array([[0.1,0.1],[0.1,0.1]])

    # Case 2: Testing for an entirely random stress tensor over two time steps
    # self.stress = np.array([[
    #   [14.0,-17.0,4.3,105,2,15.5],[105.0,205.0,5.0,0,0,0],[-1.5,-25.0,-5.6,17.1,-6.6,-301]],
    #   [[54.0,-7.0,0.3,10,200,15.5],[-100.0,25.0,50.0,0,0,0],[-1.0,-205.0,-56.0,-11.7,-0.6,-3000]]])
    # self.temperatures = np.array([[1.0,3.0,15.0],[100.0,10.0,11.0]])
    # self.volumes = np.array([[1,1,1],[1,1,1]])

    self.s0 = 70
    self.m = 3.5
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.WNTSAModel(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    # Hand - calculated
    # Case 1
    self.avg_stress = np.array([[65.9504,0],[14.7457,2.18744]]) # calculated from mathematica

    # Case 2
    #self.avg_stress = np.array([[ 29.96868454,72.5,87.11816342], [ 69.25896754,  47.5,        885.60234269]]) # calculated from mathematica

    should = -kp * (self.avg_stress**self.m) * self.volumes

    actual = self.model.calculate_element_log_reliability(self.stress, self.temperatures, self.volumes, self.material)

    R_weibull = np.exp(np.sum(actual))
    Pf_weibull = 1 - np.exp(np.sum(actual))
    print("Reliability weibull = ",R_weibull)
    print("Probability of failure weibull = ",Pf_weibull)

    self.assertTrue(np.allclose(should, actual))
