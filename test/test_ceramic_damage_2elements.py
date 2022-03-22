import unittest

import numpy as np

from srlife import materials, damage

class TestPIAModel(unittest.TestCase):
  def setUp(self):
    self.stress = np.array([
      [50.0,-60.0,30.0,0,0,0]])
    self.temperatures = np.array([100.0])
    self.volumes = np.array([0.25])

    self.s0 = 70.0
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

    self.assertTrue(np.allclose(should, actual))

class TestWeibullNormalTensileAveragingModel(unittest.TestCase):
  def setUp(self):
    self.stress = np.array([
      [50.0,60.0,30.0,0,0,0],[50.0,60.0,30.0,0,0,0]])
    self.temperatures = np.array([100.0,0.0])
    self.volumes = np.array([0.25,0.0])

    self.nelements = self.stress.shape[0]

    self.s0 = 70.0
    self.m = 3.5
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.nalpha = 10
    self.nbeta = 10

    self.model = damage.WeibullNormalTensileAveragingModel({})

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    self.avg_stress = 32.4241026255112 #40.9677 from mathematica
    #mod_stress = self.stress
    #mod_stress[mod_stress<0] = 0

    should = -kp*(self.avg_stress**self.m)*self.volumes

    actual = self.model.calculate_element_log_reliability(self.stress,
        self.temperatures, self.nalpha, self.nbeta, self.volumes, self.material)

    self.assertTrue(np.allclose(should, actual))
