#!/usr/bin/env python3

import os.path
import time

import numpy as np
import numpy.linalg as la
import scipy.io as sio

import matplotlib.pyplot as plt
from neml import models, elasticity, parse

import sys
sys.path.append('../../..')
from srlife import receiver, structural

# Moose comparison stuff
# MOOSE test stuff
ramp = lambda x: np.piecewise(x, 
    [x < 1, x>=1],
    [lambda xx: xx, lambda xx: 1.0+xx*0.0])
unit_temperature = lambda x, y, z: (np.sqrt(x**2.0+y**2.0)-9)/(10-9) * np.cos(np.arctan2(y,x))*(z/10.0 + 1)
temperature = lambda t, x, y, z: np.array([100*ramp(ti)*unit_temperature(x,y,z) for ti in t])
pressure = lambda t: 1.0 * ramp(t)

times = np.array([0, 0.5, 1, 101, 201, 301, 401, 501, 601, 701, 801, 901, 1001])
mat = parse.parse_xml(os.path.join(os.path.dirname(__file__),
  'moose-verification', 'model.xml'), 'creeping')

ri = 9.0
ro = 10.0
h = 10.0
nr = 11
nt = 20
nz = 6

moose_base = os.path.join(os.path.dirname(__file__), 'moose-verification')
moose_ver = [os.path.join(moose_base, f) for f in ['1d_out.e', '2d_out.e', '3d_out.e']]

def run_reference_simulation(d, solver):
  tube = receiver.Tube(ro, ro - ri, h, nr, nt, nz)
  tube.set_times(times)

  if d == 1:
    tube.make_1D(0, 0)
  elif d == 2:
    tube.make_2D(0)

  R, T, Z = tube.mesh
  X = R * np.cos(T)
  Y = R * np.sin(T)
  
  Ts = temperature(times, X, Y, Z).reshape((len(times),) + tube.dim[:tube.ndim])

  tube.add_results("temperature", Ts)
  tube.set_pressure_bc(receiver.PressureBC(times, pressure(times)))

  solver.setup_tube(tube)

  state_n = solver.init_state(tube, mat)

  for i in range(1,len(tube.times)):
    state_np1 = solver.solve(tube, i, state_n, 0.0)
    solver.dump_state(tube, i, state_np1)
    state_n = state_np1
  
  return tube


if __name__ == "__main__":
  if len(sys.argv) != 2:
    raise ValueError("Need to supply one argument, problem dimension!")

  d = int(sys.argv[1])

  if d not in [1,2,3]:
    raise ValueError("Invalid problem dimension %i!" % d)

  solver = structural.PythonTubeSolver()

  print("Running %iD problem..." % d)
  
  t1 = time.time()
  tube = run_reference_simulation(d, solver)
  t2 = time.time()

  print("Walltime: %f s" % (t2-t1))
  

