import unittest
import tempfile
import os, os.path

import numpy as np

from srlife import receiver, thermal, materials

def make_tube(D, period):
  R = 1.0
  t = 0.1
  h = 10.0
  nr = 10
  nt = 20
  nz = 10
  T0 = 50

  tmax = period

  ntime = 10

  T0 = 300
  
  tube = receiver.Tube(R, t, h, nr, nt, nz, T0 = T0)
  if D == 1:
    tube.make_1D(h/2,1)
  elif D == 2:
    tube.make_2D(h/2)

  times = np.linspace(0,tmax, ntime)

  tube.set_times(times)

  Tf_0 = 300
  Tf_m = 600

  def fluid_T(t):
    if t < tmax / 2.0:
      return Tf_0 + (Tf_m - Tf_0) / (tmax/2.0) * t
    else:
      return Tf_m - (Tf_m - Tf_0) / (tmax/2.0) * (t - tmax/2.0)

  ftemps = np.array([fluid_T(t) * np.ones((nz,)) for t in times])

  inner_convection = receiver.ConvectiveBC(R - t, h, nz, 
      times, ftemps)
  tube.set_bc(inner_convection, "inner")

  hflux = receiver.HeatFluxBC(R, h, nt, nz, times, 
      np.zeros((ntime,nt,nz))+2.0)
  tube.set_bc(hflux, "outer")

  solver = thermal.FiniteDifferenceImplicitThermalSolver()

  tmat = materials.ConstantThermalMaterial("dummy", 20.0e-3, 4.8)
  fmat = materials.ConstantFluidMaterial({"dummy": 8.1e-3})

  solver.solve(tube, tmat, fmat)

  return tube

def make_panel(D, ntube, period, stiff):
  panel = receiver.Panel(stiff)

  # To save some time
  t = make_tube(D, period)
  
  for i in range(ntube):
    panel.add_tube(t)

  return panel

def make_receiver(D, npanel, ntube, period = 8 * 3600.0, days = 1, 
    panel_stiffness = 0.0, manifold_stiffness = 0.0):
  r = receiver.Receiver(period, days, panel_stiffness)
  for i in range(npanel):
    r.add_panel(make_panel(D, ntube, period, manifold_stiffness))

  return r


class TestVTKWriter(unittest.TestCase):
  def setUp(self):
    self.period = 8 * 3600.0
    self.npanel = 2
    self.ntube = 3
    self.days = 1
    self.panel_stiffness = 100.0
    self.manifold_stiffness = 100.0

  def test_1D(self):
    r = make_receiver(1, self.npanel, self.ntube, period = self.period,
        days = self.days, 
        panel_stiffness = self.panel_stiffness,
        manifold_stiffness = self.manifold_stiffness)
    tdir = tempfile.mkdtemp()

    r.write_vtk(os.path.join(tdir, "test"))

    nfile = len(os.listdir(tdir))

    self.assertEqual(nfile, self.npanel * self.ntube)

  def test_2D(self):
    r = make_receiver(2, self.npanel, self.ntube, period = self.period,
        days = self.days,
        panel_stiffness = self.panel_stiffness,
        manifold_stiffness = self.manifold_stiffness)
    tdir = tempfile.mkdtemp()

    r.write_vtk(os.path.join(tdir, "test"))

    nfile = len(os.listdir(tdir))

    self.assertEqual(nfile, self.npanel * self.ntube)

  def test_3D(self):
    r = make_receiver(2, self.npanel, self.ntube, period = self.period,
        days = self.days, 
        panel_stiffness = self.panel_stiffness,
        manifold_stiffness = self.manifold_stiffness)
    tdir = tempfile.mkdtemp()

    r.write_vtk(os.path.join(tdir, "test"))

    nfile = len(os.listdir(tdir))

    self.assertEqual(nfile, self.npanel * self.ntube)

