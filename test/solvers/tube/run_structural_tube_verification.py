#!/usr/bin/env python3

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from neml import models, elasticity

import sys
sys.path.append('../../..')
from srlife import receiver, structural

class TestCase:
  def __init__(self, name, T, analytic, ri = 0.9, ro = 1.0, h = 10.0,
      alpha = 1.0e-5, E = 100000.0, nu = 0.3, p = 1.0):
    self.name = name
    self.Tfn = T
    self.afn = analytic

    self.ri = ri
    self.ro = ro
    self.h = h

    self.alpha = alpha
    self.E = E
    self.nu = nu

    self.p = p

  def T(self, r):
    return self.Tfn(r, self.ri, self.ro)

  def exact(self, r):
    return self.afn(r, self.p, self.ri, self.ro, self.E, self.nu, self.alpha)

  def make_mat(self):
    emodel = elasticity.IsotropicLinearElasticModel(self.E, "youngs", 
        self.nu, "poissons")
    return models.SmallStrainElasticity(emodel, alpha = self.alpha)

  def make_tube(self, dim, nr = 15, nt = 30, nz = 5):
    tube = receiver.Tube(self.ro, self.ro - self.ri, self.h, nr, nt, nz)

    if dim == 1:
      tube.make_1D(self.h/2, 0)
    elif dim == 2:
      tube.make_2D(self.h/2)

    times = np.array([0,1])
    tube.set_times(times)

    R, _, _ = tube.mesh
    Ts = np.zeros((2,) + R.shape[:dim])
    Ts[1] = self.T(R)

    tube.add_results("temperature", Ts)

    if self.p != 0:
      tube.set_pressure_bc(receiver.PressureBC(times, times * self.p))

    return tube

  def run_comparison(self, dim, solver, axial_strain = 0, nr = 10, 
      nt = 20, nz = 10):
    mat = self.make_mat()
    tube = self.make_tube(dim, nr, nt, nz)

    structural.setup_tube_structural_solve(tube)
    state_n = solver.init_state(tube, mat)

    state_np1 = solver.solve(tube, 1, state_n, axial_strain)

    solver.dump_state(tube, 1, state_np1)

    tube.write_vtk("wtf")

    return tube

  def get_comparison(self, tube):
    if tube.ndim == 3:
      z = tube.nz // 2
      x_avg = np.mean(tube.results['disp_x'])
      u = tube.results['disp_x'][1,:,0,z] - 2*x_avg
      r = tube.mesh[0][:,0,z]
    elif tube.ndim == 2:
      # The displacements tend to drift, need to recenter
      x_avg = np.mean(tube.results['disp_x'])
      u = tube.results['disp_x'][1,:,0] - 2*x_avg
      r = tube.mesh[0][:,0,0]
    else:
      u = tube.results['disp_x'][1]
      r = tube.mesh[0][:,0,0]

    return u, r

  def plot_comparison(self, tube):
    u, r = self.get_comparison(tube)

    plt.figure()
    plt.plot(r, u, 'k-')
    plt.plot(r, self.exact(r), 'k--')
    plt.xlabel("Radial position")
    plt.ylabel("Radial displacement")
    plt.title(self.name + ": " + "%iD" % tube.ndim)
    plt.show()

  def evaluate_comparison(self, tube):
    u, r = self.get_comparison(tube)
    
    err = np.abs(u - self.exact(r))
    rel = err / np.abs(self.exact(r))

    return np.max(err), np.max(rel)


def exact1(r, p, ri, ro, E, v, a):
  A = p / (1.0/ro**2.0 - 1.0/ri**2.0)
  C = -A/(2*ro**2.0)

  res = (1.0+v)/E * (-A/r + 2.0*(1-2.0*v) * C* r)

  return res

cases = [
    TestCase("Inner pressure", lambda T, ri, ro: 0.0, exact1,
      p = 100, ri = 8, ro = 10.0)
    ]

if __name__ == "__main__":
  solver = structural.PythonTubeSolver()
  for d in range(1,4):
    for case in cases:
      tube = case.run_comparison(d, solver)

      a, r = case.evaluate_comparison(tube)
      print(case.name + ": " "%iD" % d)
      print("Max absolute error: %e" % a)
      print("Max relative error: %e" % r)
      print("")

      case.plot_comparison(tube)
      
