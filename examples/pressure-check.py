#!/usr/bin/env python3

import sys
sys.path.append('..')

from srlife import receiver, thermal, materials, structural
from neml import parse, models

import numpy as np

if __name__ == "__main__":
  R = 100.0
  t = 1
  h = 1000.0
  nr = 10
  nt = 20
  nz = 5

  tmax = 1
  ntime = 2

  D = 3

  tube = receiver.Tube(R, t, h, nr, nt, nz)
  if D == 1:
    tube.make_1D(h/2,1)
  elif D == 2:
    tube.make_2D(h/2)

  times = np.linspace(0,tmax, ntime)

  tube.set_times(times)

  pressure = receiver.PressureBC(times, times * 2)
  tube.set_pressure_bc(pressure)

  ssolver = structural.PythonTubeSolver(verbose = True)

  smat = parse.parse_xml("A740H_structural.xml", "elastic_model")

  structural.setup_tube_structural_solve(tube)
  state_n = ssolver.init_state(tube, smat)

  for i in range(1,len(tube.times)):
    state_np1 = ssolver.solve(tube, i, state_n, 0.0)
    ssolver.dump_state(tube, i, state_np1)
    state_n = state_np1

  tube.write_vtk("pressure")
