#!/usr/bin/env python3

import sys
sys.path.append('..')

import numpy as np

from  srlife import receiver, structural

from neml import models, surfaces, hardening, ri_flow, elasticity

if __name__ == "__main__":
  E = 100000.0
  nu = 0.3
  sy = 100.0
  K = E / 50.0

  emodel = elasticity.IsotropicLinearElasticModel(E, "youngs", nu, "poissons")
  surface = surfaces.IsoJ2()
  iso = hardening.LinearIsotropicHardeningRule(sy, K)
  flow = ri_flow.RateIndependentAssociativeFlow(surface, iso)
  mat = models.SmallStrainRateIndependentPlasticity(emodel, flow)

  ri = 25
  t = 3.0
  ro = ri + t
  h = 1000.0

  nr = 11
  nt = 15
  nz = 10

  p = 15.0

  tube = receiver.Tube(ro, t, h, nr, nt, nz)
  times = np.array([0,1])
  tube.set_times(times)

  tube.set_pressure_bc(receiver.PressureBC(times, times * p))

  solver = structural.PythonTubeSolver(verbose = True)

  solver.setup_tube(tube)
  state_n = solver.init_state(tube, mat)
  state_np1 = solver.solve(tube, 1, state_n, 0)

