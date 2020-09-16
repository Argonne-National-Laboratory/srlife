#!/usr/bin/env python3

import sys
sys.path.append('..')

import numpy as np

from srlife import receiver, thermal, spring

def make_tube(R, t, h, nr, nt, nz, ntime, tmax, T0, Tf_0, Tf_m, hmax, pmax,
    D = 1):
  """
    Make a tube object
  """
  tube = receiver.Tube(R, t, h, nr, nt, nz, T0 = T0)
  if D == 1:
    tube.make_1D(h/2,1)
  elif D == 2:
    tube.make_2D(h/2)

  times = np.linspace(0, tmax, ntime)

  tube.set_times(times)

  def fluid_T(t):
    if t < tmax / 2.0:
      return Tf_0 + (Tf_m - Tf_0) / (tmax/2.0) * t
    else:
      return Tf_m - (Tf_m - Tf_0) / (tmax/2.0) * (t - tmax/2.0)

  ftemps = np.array([fluid_T(t) * np.ones((nz,)) for t in times])

  T, TH, ZS = np.meshgrid(tube.times, 
      np.linspace(0, 2.0*np.pi, tube.nt+1)[:tube.nt],
      np.linspace(0, tube.h, tube.nz), indexing = 'ij')

  hflux = receiver.HeatFluxBC(R, h, nt, nz, times, 
      hmax * np.sin(T/tmax*2*np.pi) * np.cos(TH) * (ZS/h))
  tube.set_bc(hflux, "outer")

  pressure = receiver.PressureBC(times, times / 3600.0 * 2.0)
  tube.set_pressure_bc(pressure)

  return tube

if __name__ == "__main__":
  R = 1.0
  t = 0.1
  h = 1000.0

  nr = 10
  nt = 20
  nz = 10

  ntime = 10
  tmax = 8.0*3600.0

  T0 = 300

  ntubes = 9
  Tf_0 = 300
  Tf_m = np.linspace(400,700,ntubes)
  hmax = np.linspace(0.5,1,ntubes)
  pmax = 2.0

  tubes = [make_tube(R, t, h, nr, nt, nz, ntime, tmax, T0, Tf_0, Tf_mi,
    hmax_i, pmax) for Tf_mi, hmax_i in zip(Tf_m, hmax)]

  # Thermal material

  # Structural material


  
  # Solve the thermal history of each tube
  tsolver = thermal.FiniteDifferenceImplicitThermalSolver()
  for tube in tubes:
    pass
