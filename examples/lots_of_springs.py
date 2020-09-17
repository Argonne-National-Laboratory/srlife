#!/usr/bin/env python3

import sys
sys.path.append('..')

import numpy as np

from srlife import receiver, thermal, spring, materials, structural
from neml import parse, models

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

  ntime = 50
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
  tmat = materials.ConstantThermalMaterial("dummy", 20.0e-3, 4.8)
  fmat = materials.ConstantFluidMaterial({"dummy": 8.1e-3})

  # Structural material
  smat = parse.parse_xml("A740H_structural.xml", "creep_plasticity")

  # Solve the thermal history of each tube
  tsolver = thermal.FiniteDifferenceImplicitThermalSolver()
  for tube in tubes:
    tsolver.solve(tube, tmat, fmat)

  # Structural solver
  ssolver = structural.PythonTubeSolver(atol = 1.0e-4, verbose = False)

  # Setup the spring network (manually in this case)
  network = spring.SpringNetwork()
  for i in range(22):
    network.add_node(i)

  k1 = 100.0
  k2 = 200.0

  network.add_edge(0,1, object = spring.LinearSpring(k1))

  network.add_edge(1,2, object = "rigid")
  network.add_edge(1,3, object = "rigid")
  network.add_edge(1,4, object = "rigid")

  network.add_edge(2,5, object = spring.TubeSpring(tubes[0], ssolver, smat))
  network.add_edge(3,6, object = spring.TubeSpring(tubes[1], ssolver, smat))
  network.add_edge(4,7, object = spring.TubeSpring(tubes[2], ssolver, smat))

  network.add_edge(0,8, object = spring.LinearSpring(k2))

  network.add_edge(8,9, object = spring.LinearSpring(k1))
  network.add_edge(8,10, object = spring.LinearSpring(k1))
  network.add_edge(8,11, object = spring.LinearSpring(k1))

  network.add_edge(9, 12, object = spring.TubeSpring(tubes[3], ssolver, smat))
  network.add_edge(10, 13, object = spring.TubeSpring(tubes[4], ssolver, smat))
  network.add_edge(10, 14, object = spring.TubeSpring(tubes[5], ssolver, smat))

  network.add_edge(0, 15, object = "disconnect")

  network.add_edge(15, 16, object = spring.LinearSpring(k2))
  network.add_edge(15, 17, object = "rigid")
  network.add_edge(15, 18, object = "rigid")

  network.add_edge(16, 19, object = spring.TubeSpring(tubes[6], ssolver, smat))
  network.add_edge(17, 20, object = spring.TubeSpring(tubes[7], ssolver, smat))
  network.add_edge(18, 21, object = spring.TubeSpring(tubes[8], ssolver, smat))
  
  for i in [5,6,7,12,13,14,19,20,21]:
    network.displacement_bc(i, lambda t: 0.0)

  network.validate_setup()

  subproblems = network.reduce_graph()
  
  for problem in subproblems:
    problem.solve_all()

  print(np.max(tubes[0].quadrature_results['stress_xx']))
  print(np.max(tubes[-1].quadrature_results['stress_yy']))
