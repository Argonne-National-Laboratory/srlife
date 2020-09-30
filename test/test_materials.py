import unittest
import tempfile

import numpy as np
import scipy.interpolate as inter

from srlife import materials
from helpers import differentiate

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

  def test_store_recover(self):
    tf = tempfile.mktemp()

    self.obj.save(tf, "blah")

    rep = materials.ThermalMaterial.load(tf, "blah")

    self.assertEqual(rep.name, self.obj.name)
    self.assertTrue(np.allclose(rep.temps, self.obj.temps))
    self.assertTrue(np.allclose(rep.cond, self.obj.cond))
    self.assertTrue(np.allclose(rep.diff, self.obj.diff))


class CommonFluidMaterial:
  def test_derivtive(self):
    for T in self.Trange:
      self.assertAlmostEqual(self.obj.dcoefficient(self.mat, T),
          differentiate(lambda x: self.obj.coefficient(self.mat, x), T))

class TestConstantFluidMaterial(CommonFluidMaterial, unittest.TestCase):
  def setUp(self):
    self.value = 10.0
    self.data = {"Computonium": self.value}
    self.obj = materials.ConstantFluidMaterial(self.data)

    self.Trange = np.linspace(0, 100)
    self.mat = "Computonium"

  def test_recover(self):
    tfile = tempfile.mktemp()
    self.obj.save(tfile, "blah")

    comp = materials.FluidMaterial.load(tfile, "blah")

    self.assertAlmostEqual(self.obj.coefficient("Computonium", 0),
        comp.coefficient("Computonium", 0))

  def test_interpolate(self):
    self.assertAlmostEqual(self.obj.coefficient("Computonium", 0), self.value)


class TestPiecewiseLinearFluidMaterial(CommonFluidMaterial, unittest.TestCase):
  def setUp(self):
    self.data = {"Computonium": (np.array([10.0,100.0,200.0,300.0]), np.array([50,10.0,5.0,2.0])),
        "Other": (np.array([10,100,200.0,300.0]), np.array([10.0,15.0,20.0,25.0]))}
    self.obj = materials.PiecewiseLinearFluidMaterial(self.data)

    self.Trange = np.linspace(10,300)[1:-1]
    self.mat = "Other"

  def test_recover(self):
    tfile = tempfile.mktemp()
    self.obj.save(tfile, "blah")

    comp = materials.FluidMaterial.load(tfile, "blah")

    for k, (T, vals) in self.obj.data.items():
      self.assertTrue(k in comp.data)
      self.assertTrue(np.allclose(T, comp.data[k][0]))
      self.assertTrue(np.allclose(vals, comp.data[k][1]))

  def test_interpolate(self):
    T = 125.0
    mat = "Computonium"

    v1 = inter.interp1d(self.data[mat][0], self.data[mat][1])(T)

    self.assertAlmostEqual(v1, self.obj.coefficient(mat, T))

class TestStructuralMaterial(unittest.TestCase):
    def setUp(self):
        self.mat="Computonium"
        self.T = 850.15
        self.stress=100
        self.erange=5e-3
        self.true_inside_envelope=np.array([0.533, 0.2])
        self.false_inside_envelope=np.array([0.1, 0.77])
        self.error_inside_envelope=np.array([-0.01, 0.01])
        self.fatigue_pname="nFatigue"
        self.rupture_pname="avgRupture"
        self.cfinteraction_pname="cfinteraction"
        self.data={"nFatigue": {"curve1": {"T": "950","a": "-9.2e-01 -3.8e+00 -8.4e+00 -4.1e+00", "n": "3 2 1 0", "cutoff": "1.5e-3"},
        "curve2": {"T": "800", "a": "-19.2e-01 -3.8e+00 -18.4e+00 -14.1e+00", "n": "3 2 1 0", "cutoff": "1.5e-3"},
        "curve3": {"T": "500", "a": "-0.2e-01 -0.8e+00 -0.4e+00 -0.1e+00", "n": "3 2 1 0", "cutoff": "1.5e-3"}},
        "avgRupture": {"C": "17.16","a": "-1475.23 7289.41 -16642.64 35684.60", "n": "3 2 1 0"},
        "lbRupture": {"C": "10.0","a": "-1000.0 7000.0 -10000.0 30000.0", "n": "3 2 1 0"},
        "cfinteraction":"0.3 0.3"}

        self.structmat = materials.StructuralMaterial(self.data)

    def test_store_receover(self):
      tfile = tempfile.mktemp()
      self.structmat.save(tfile, "blah")
      test = materials.StructuralMaterial.load(tfile, "blah")

    def test_cycles_to_fail(self):
        fcycles = self.structmat.cycles_to_fail(self.fatigue_pname,self.T,self.erange)

        a=np.array([-9.2e-01, -3.8e+00, -8.4e+00, -4.1e+00])
        n=np.array([3,2,1,0])
        cutoff=1.5e-3
        sum = 0
        if self.erange<=cutoff:
            for (b,m) in zip(a,n):
                sum+=b*np.log10(cutoff)**m
        else:
            for (b,m) in zip(a,n):
                sum+=b*np.log10(self.erange)**m
        return 10**sum

        self.assertTrue(np.isclose(sum, fcycles))

    def test_rupturetime(self):
        rtime=self.structmat.time_to_rupture(self.rupture_pname, 
            np.array([self.T]), np.array([self.stress]))[0]

        C=17.16
        a=np.array([-1475.23, 7289.41, -16642.64, 35684.60])
        n=np.array([3,2,1,0])
        sum = 0
        for (b,m) in zip(a,n):
              sum+=b*np.log10(self.stress)**m
        sum=10**(sum/self.T-C)

        self.assertTrue(np.isclose(sum, rtime))

    def test_inside_envelope(self):
      """
        Test interaction_fatigue and interaction_creep
      """
      self.assertTrue(self.structmat.inside_envelope(self.cfinteraction_pname,self.true_inside_envelope[0],self.true_inside_envelope[1]))
      self.assertFalse(self.structmat.inside_envelope(self.cfinteraction_pname,self.false_inside_envelope[0],self.false_inside_envelope[1]))
      with self.assertRaises(ValueError):
          self.structmat.inside_envelope(self.cfinteraction_pname,self.error_inside_envelope[0],self.error_inside_envelope[1])
