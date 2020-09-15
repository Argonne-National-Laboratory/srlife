#!/usr/bin/env python3

import sys
sys.path.append('..')

from srlife import receiver, thermal, materials, structural
from neml import parse, models

import numpy as np

if __name__ == "__main__":
  R = 1.0
  t = 0.1
  h = 10.0
  nr = 10
  nt = 20
  nz = 10
  T0 = 50

  tmax = 8.0 * 3600.0

  ntime = 10

  T0 = 300
  
  D = 3

  tube = receiver.Tube(R, t, h, nr, nt, nz, T0 = T0)
  if D == 1:
    tube.make_1D(h/2,1)
  elif D == 2:
    tube.make_2D(h/2)

  times = np.linspace(0,tmax, ntime)

  tube.set_times(times)

  Tf_0 = 300 + 273.15
  Tf_m = 600 + 273.15

  def fluid_T(t):
    if t < tmax / 2.0:
      return Tf_0 + (Tf_m - Tf_0) / (tmax/2.0) * t
    else:
      return Tf_m - (Tf_m - Tf_0) / (tmax/2.0) * (t - tmax/2.0)

  ftemps = np.array([fluid_T(t) * np.ones((nz,)) for t in times])

  inner_convection = receiver.ConvectiveBC(R - t, h, nz, 
      times, ftemps)
  tube.set_bc(inner_convection, "inner")
  
  
  T, TH, ZS = np.meshgrid(tube.times, 
      np.linspace(0, 2.0*np.pi, tube.nt+1)[:tube.nt],
      np.linspace(0, tube.h, tube.nz), indexing = 'ij')


  hflux = receiver.HeatFluxBC(R, h, nt, nz, times, 
      np.sin(T/tmax*2*np.pi) * np.cos(TH) * (ZS/h))
  tube.set_bc(hflux, "outer")

  pressure = receiver.PressureBC(times, times / 3600.0 * 2.0)
  tube.set_pressure_bc(pressure)

  solver = thermal.FiniteDifferenceImplicitThermalSolver()

  tmat = materials.ConstantThermalMaterial("dummy", 20.0e-3, 4.8)
  fmat = materials.ConstantFluidMaterial({"dummy": 8.1e-3})

  solver.solve(tube, tmat, fmat)
  
  ssolver = structural.PythonTubeSolver(verbose = True)

  smat = parse.parse_xml("A740H_structural.xml", "elastic_model")

  ssolver.setup_tube(tube)
  state_n = ssolver.init_state(tube, smat)

  for i in range(1,len(tube.times)):
    state_np1 = ssolver.solve(tube, i, state_n, 0)
    ssolver.dump_state(tube, i, state_np1)
    state_n = state_np1

  tube.write_vtk("structural")
