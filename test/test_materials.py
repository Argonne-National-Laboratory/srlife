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


class CommonFluidMaterial:
  def test_derivtive(self):
    for T in self.Trange:
      self.assertAlmostEqual(self.obj.dcoefficient(self.mat, T),
          differentiate(lambda x: self.obj.coefficient(self.mat, x), T))

class TestPiecewiseLinearFluidMaterial(CommonFluidMaterial, unittest.TestCase):
  def setUp(self):
    self.data = {"Computonium": (np.array([10.0,100.0,200.0,300.0]), np.array([50,10.0,5.0,2.0])),
        "Other": (np.array([10,100,200.0,300.0]), np.array([10.0,15.0,20.0,25.0]))}
    self.obj = materials.PiecewiseLinearFluidMaterial(self.data)

    self.Trange = np.linspace(10,300)[1:-1]
    self.mat = "Other"

  def test_recover(self):
    tfile = tempfile.mktemp()
    self.obj.save(tfile)

    comp = materials.FluidMaterial.load(tfile)

    for k, (T, vals) in self.obj.data.items():
      self.assertTrue(k in comp.data)
      self.assertTrue(np.allclose(T, comp.data[k][0]))
      self.assertTrue(np.allclose(vals, comp.data[k][1]))

  def test_interpolate(self):
    T = 125.0
    mat = "Computonium"

    v1 = inter.interp1d(self.data[mat][0], self.data[mat][1])(T)

    self.assertAlmostEqual(v1, self.obj.coefficient(mat, T))

class TestRuptureTime(unittest.TestCase):
    def setUp(self):
        self.mat="Computonium"
        self.prop="Rupture"
        self.T = 850.15
        self.stress=100
        self.C=17.16
        self.a=np.array([-1475.23, 7289.41, -16642.64, 35684.60])
        self.n=np.array([3,2,1,0])
        self.data={self.mat: {self.prop: {"C": "17.16",
        "a": "-1475.23 7289.41 -16642.64 35684.60", "n": "3 2 1 0"}}}

    def test_rupturetime(self):
        tfrupture = tempfile.mktemp()
        materials.save_dict_xml(self.data, tfrupture)
        rtime=materials.rupturetime(tfrupture, self.mat, self.prop, self.T, self.stress)

        sum = 0
        for (b,m) in zip(self.a,self.n):
              sum+=b*np.log10(self.stress)**m
        sum=10**(sum/self.T-self.C)

        self.assertTrue(np.isclose(sum, rtime))

class TestCyclesToFail(unittest.TestCase):
    def setUp(self):
        self.mat="Computonium"
        self.prop="Fatigue"
        self.T = 850.15
        self.erange=5e-3
        self.a=np.array([-9.2e-01, -3.8e+00, -8.4e+00, -4.1e+00])
        self.n=np.array([3,2,1,0])
        self.cutoff=1.5e-3
        self.data={self.mat: {self.prop: {"curve1": {"T": "950",
        "a": "-9.2e-01 -3.8e+00 -8.4e+00 -4.1e+00", "n": "3 2 1 0", "cutoff": "1.5e-3"},
        "curve2": {"T": "800", "a": "-19.2e-01 -3.8e+00 -18.4e+00 -14.1e+00", "n": "3 2 1 0", "cutoff": "1.5e-3"},
        "curve3": {"T": "500", "a": "-0.2e-01 -0.8e+00 -0.4e+00 -0.1e+00", "n": "3 2 1 0", "cutoff": "1.5e-3"}}}}

    def test_cyclestofail(self):
        tffatigue = tempfile.mktemp()
        materials.save_dict_xml(self.data, tffatigue)
        fcycles=materials.cyclestofail(tffatigue, self.mat, self.prop, self.T, self.erange)

        sum = 0
        if self.erange<=self.cutoff:
            for (b,m) in zip(self.a,self.n):
                sum+=b*np.log10(self.cutoff)**m
        else:
            for (b,m) in zip(self.a,self.n):
                sum+=b*np.log10(self.erange)**m
        return 10**sum

        self.assertTrue(np.isclose(sum, fcycles))
