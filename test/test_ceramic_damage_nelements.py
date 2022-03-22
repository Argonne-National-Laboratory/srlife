import unittest

import numpy as np

from srlife import materials, damage, solverparams

class TestPIAModel(unittest.TestCase):
  def setUp(self):
    #self.stress = np.array([
    #  [50.0,-60.0,30.0,0,0,0]])
    self.stress = np.array([1.793,0,0,0,0,0])
    self.temperatures = np.array([100.0])
    #self.volumes = np.array([0.25])
    self.volumes = np.array([1])

    #self.s0 = 70.0
    self.s0 = 1.697
    #self.m = 3.5
    self.m = 28.53
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.PIAModel({})

  def test_definition(self):
    k = self.s0**(-self.m)

    mod_stress = self.stress
    mod_stress[mod_stress<0] = 0
    print("stress_hand_calculated = ",mod_stress)

    should = -k  * np.sum(mod_stress**self.m, axis = -1) * self.volumes
    print("Pf = ",1-np.exp(should))

    actual = self.model.calculate_element_log_reliability(self.stress,
        self.temperatures, self.volumes, self.material)
    print("actual = ",actual)

    self.assertTrue(np.allclose(should, actual))

class TestWeibullNormalTensileAveragingModel(unittest.TestCase):
  def setUp(self):
    #self.stress = np.array([[
    #  [14.0,-17.0,4.3,105,2,15.5],[105.0,205.0,5.0,0,0,0],[-1.5,-25.0,-5.6,17.1,-6.6,-301]],
    #  [[54.0,-7.0,0.3,10,200,15.5],[100.0,25.0,50.0,0,0,0],[-1.0,-205.0,-56.0,-11.7,-0.6,-3000]]])
    #self.stress = np.array([[
    #  [100.0,25.0,50.0,0,0,0],[100.0,25.0,50.0,0,0,0],[100.0,25.0,50.0,0,0,0]],
    #  [[100.0,25.0,50.0,0,0,0],[100.0,25.0,50.0,0,0,0],[100.0,25.0,50.0,0,0,0]]])
    self.stress = np.array([1.241,0,0,0,0,0])
    #self.temperatures = np.array([[1.0,3.0,15.0],[100.0,10.0,11.0]])
    self.temperatures = np.array([1.0])
    #self.volumes = np.array([[0.1,0.1,0.1],[0.1,0.1,0.1]])
    self.volumes = np.array([0.1])
    #print(len(self.stress))
    #self.s0 = 70.0
    self.s0 = 169.7
    #self.m = 3.5
    self.m = 28.53
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    #self.nalpha = 10
    #self.nbeta = 20

    self.model = damage.WeibullNormalTensileAveragingModel(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    #self.avg_stress = np.array([25.1943,47.5,341.827]) # calculated from mathematica
    self.avg_stress = np.array([[29.52341388,79.58100559,85.40471012],[71.82904589,49.8603352,857.42973823]]
) # calculated from mathematica

    print("avg_normal_tensile_stress_hand_calculated = ",self.avg_stress)

    should = -kp*(self.avg_stress**self.m)*self.volumes
    print("should = ",should)
    actual = self.model.calculate_element_log_reliability(self.stress, self.temperatures, self.volumes, self.material)
    print("actual = ",actual)
    self.assertTrue(np.allclose(should, actual))
