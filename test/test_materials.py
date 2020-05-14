import unittest
import tempfile

import numpy as np
import scipy.interpolate as inter

from srlife import materials
from helpers import differentiate

class TestLoadStore(unittest.TestCase):
  def setUp(self):
    self.example = {"one": {"two" : {"three": "data", "four": "yaydata"}, 
      "five": "moredata"}, "six": "evenmore"}

  def test_recover(self):
    tf = tempfile.mktemp()
    
    materials.save_dict_xml(self.example, tf)
    rn, comp = materials.load_dict_xml(tf)

    self.assertTrue(comp == self.example)

class CommonThermalMaterial:
  def test_derivative(self):
    Tvals = np.linspace(np.min(self.T), np.max(self.T))[1:-1]

    for T in Tvals:
      self.assertTrue(np.isclose(self.obj.dconductivity(T),
        differentiate(lambda x: self.obj.conductivity(x), T)))
      self.assertTrue(np.isclose(self.obj.ddiffusivity(T),
        differentiate(lambda x: self.obj.diffusivity(x), T)))


class TestPiecewiseLinearThermalMaterial(CommonThermalMaterial, unittest.TestCase):
  def setUp(self):
    self.name = "Computonium"
    self.T = np.array([10.0,100.0,150.0,250.0])
    self.cond = np.array([25.0,50.0,10.0,30.0])
    self.diff = np.array([100.0,20.0,50.0,60.0])

    self.obj = materials.PiecewiseLinearThermalMaterial(self.name, 
        self.T, self.cond, self.diff)

  def test_interpolate(self):
    T = 125.0
    self.assertTrue(np.isclose(inter.interp1d(self.T, self.cond)(T), 
      self.obj.conductivity(T)))
    self.assertTrue(np.isclose(inter.interp1d(self.T, self.diff)(T), 
      self.obj.diffusivity(T)))

  def test_store_receover(self):
    tf = tempfile.mktemp()

    self.obj.save(tf)
    
    rep = materials.ThermalMaterial.load(tf)

    self.assertEqual(rep.name, self.obj.name)
    self.assertTrue(np.allclose(rep.temps, self.obj.temps))
    self.assertTrue(np.allclose(rep.cond, self.obj.cond))
    self.assertTrue(np.allclose(rep.diff, self.obj.diff))
