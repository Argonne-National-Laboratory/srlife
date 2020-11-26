#!/usr/bin/env python3

import os.path

import numpy as np
import numpy.linalg as la
import scipy.io as sio

import matplotlib.pyplot as plt
from neml import models, elasticity, parse

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
    
    solver.setup_tube(tube)
    state_n = solver.init_state(tube, mat)

    state_np1 = solver.solve(tube, 1, state_n, axial_strain)

    solver.dump_state(tube, 1, state_np1)

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

def load_displacements_exodus(efilename, d):
  nf = sio.netcdf_file(efilename)
  cnames = ['coordx', 'coordy', 'coordz']
  coords = np.array([
    np.copy(nf.variables[c][:]) for c in cnames[:d]])

  names = [b''.join(v).decode('utf-8') for v in nf.variables['name_nod_var']]
  disps = np.array([np.copy(nf.variables['vals_nod_var%i' % (names.index(nm)+1)][:]) for 
      nm in ['disp_x', 'disp_y', 'disp_z'][:d]])

  return coords, disps

def compare_nodal_field(c1, r1, c2, r2, d, dec = 6, gtol = 1.0e-8):
  c1 = np.round(c1, decimals=dec)
  c2 = np.round(c2, decimals=dec)
  for i in range(d-1,-1,-1):
    one_args = c1[i,:].argsort(kind='mergesort')
    c1 = c1[:,one_args]

    two_args = c2[i,:].argsort(kind='mergesort')
    c2 = c2[:,two_args]

    for j in range(d):
      for k in range(r1.shape[1]):
        r1[j,k] = r1[j,k,one_args]
        r2[j,k] = r2[j,k,two_args]

  # Sanity check
  if la.norm(c1-c2) > gtol:
    raise ValueError("Point clouds aren't equivalent...")

  error = np.abs(r1 - r2)
  aerror = np.max(error)

  keep = np.abs(r1) > gtol
  rerror = np.max(error[keep] / np.abs(r1[keep]))

  return aerror, rerror

def do_complete_comparison(d, solver):
  tube = run_reference_simulation(d, solver)

  R, T, Z = tube.mesh
  X = R * np.cos(T)
  Y = R * np.sin(T) 
  cs = [X,Y,Z]
  coords = np.array([c.flatten() for c in cs[:d]])
  dres = ['disp_x', 'disp_y', 'disp_z']
  disps = np.array([tube.results[d].reshape((len(tube.times),-1)) for d in dres[:d]])

  mc, md = load_displacements_exodus(moose_ver[d-1], d)

  return compare_nodal_field(mc, md, coords, disps, d)


if __name__ == "__main__":
  solver = structural.PythonTubeSolver(verbose = False)

  print("Analytical comparison")
  print("=====================")
  print("")
  for d in range(1,4):
    for case in cases:
      tube = case.run_comparison(d, solver)

      a, r = case.evaluate_comparison(tube)
      print(case.name + ": " "%iD" % d)
      print("Max absolute error: %e" % a)
      print("Max relative error: %e" % r)
      print("")

      case.plot_comparison(tube)
  
  print("MOOSE comparison")
  print("================")
  print("")
  for d in range(1,4):
    a, r = do_complete_comparison(d,solver)
    print("%iD" % d)
    print("Max absolute error: %e" % a)
    print("")
