#!/usr/bin/env python3

import sys
sys.path.append('..')

from srlife import receiver, thermal, materials

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

  tube.write_vtk("test")

  for ts in [0,ntime//2,ntime-1]:
    if D == 1:
      print(tube.results['temperature'][ts,:])
    elif D == 2:
      print(tube.results['temperature'][ts,:,0])
      print(tube.results['temperature'][ts,:,-1])
    else:
      print(tube.results['temperature'][ts,:,0,nz//2])
      print(tube.results['temperature'][ts,:,-1,nz//2])

