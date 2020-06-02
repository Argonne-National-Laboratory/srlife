import sys
sys.path.append('../..')

import numpy as np
import matplotlib.pyplot as plt

from srlife import thermal, receiver, materials

class ManufacturedSolution:
  """
    A manufactured heat transport solution
  """
  def __init__(self, dim, solution, source):
    """
      Parameters:
        dim:        dimension of solution
        solution:   function of t, r, ...
        source      function of t, k, alpha, r, ...
        flux        function of t, k, alpha, r, ... (for BC)
    """
    self.dim = dim
    self.soln = solution
    self.source = source

  def solve(self, solver, thermal, fluid, r = 1.0, t = 0.2, h = 1, time = 1, 
      ntime = 11, nr = 10, nt = 20, nz = 10, T0 = 0.0):
    """
      Generate the appropriate tube and solve with the provided solver
      
      Parameters:
        solver:         the thermal solver to test
        thermal:        the thermal model to test

      Other Parameters:
        r               tube radius
        t               tube thickness
        h               tube height
        time            maximum time
        ntime           number of time steps
        nr              number of radial increments
        nt              number of circumferential increments
        nz              number of axial increments
        T0:             initial temperature
    """
    tube = receiver.Tube(r, t, h, nr, nt, nz, T0)

    times = np.linspace(0, time, ntime)
    tube.set_times(times)

    # Set the temperatures on the edges
    dr = t / nr
    smesh = self._generate_surface_mesh(t, h, times, nt, nz)
    bc1 = receiver.FixedTempBC(r - t, h, nt, nz, times, 
        self.soln(smesh[0], (r-t-dr/2)*np.ones(smesh[0].shape),
          *smesh[1:self.dim]))
    tube.set_bc(bc1, "inner")
    bc2 = receiver.FixedTempBC(r, h, nt, nz, times,
        self.soln(smesh[0], (r+dr/2)*np.ones(smesh[0].shape),
          *smesh[1:self.dim]))
    tube.set_bc(bc2, "outer")

    if self.dim == 2:
      tube.make_2D(tube.h/2)
    elif self.dim == 1:
      tube.make_1D(tube.h/2, 0)

    def T0(*args):
      return self.soln(0, *args)

    def sfn(t, *args):
      k = thermal.conductivity(self.soln(t, *args))
      a = thermal.diffusivity(self.soln(t, *args))

      return self.source(t, k, a, *args)

    solver.solve(tube, thermal, materials.ConstantFluidMaterial({}), 
        source = sfn, T0 = T0)

    return tube

  def plot_comparison(self, soln):
    """
      Plot a comparison to a manufactured solution

      Parameters:
        soln:           particular tube object with the solution
    """
    mesh = self._generate_mesh(soln.r, soln.t, soln.h, soln.times, 
        soln.nr, soln.nt, soln.nz)
    T = self.soln(*mesh)

    plt.figure()
    if self.dim == 1:
      plot = [0, soln.nr // 2, -1]
      for p in plot:
        l, = plt.plot(mesh[0][:,p], soln.results['temperature'][:,p])
        plt.plot(mesh[0][:,p], T[:,p], ls = '--', color = l.get_color())
      plt.xlabel("Time")
      plt.ylabel("Temperature")
      plt.title("1D: left, middle, right")

  def assess_comparison(self, soln, tol):
    """
      Return true if solution matches, false otherwise

      Parameters:
        soln:           particular tube object with the solution
        tol:            relative tolerance
    """
    mesh = self._generate_mesh(soln.r, soln.t, soln.h, soln.times, 
        soln.nr, soln.nt, soln.nz)
    T = self.soln(*mesh)

    return np.allclose(T, soln.results['temperature'], rtol = tol)

  def _generate_mesh(self, r, t, h, times, nr, nt, nz):
    """
      Generate the appropriate finite difference mesh for a particular problem
      
      Parameters:
        r           radius
        t           thickness
        h           height
        times       discrete time steps
        nr          number of radial increments
        nt          number of circumferential increments
        nz          number of axial increments
    """
    rs = np.linspace(r-t, r, nr + 1)
    rs = (rs[1:] + rs[:-1]) / 2.0
    ts = np.linspace(0, 2*np.pi, nt + 1)
    ts = (ts[1:] + ts[:-1]) / 2.0
    zs = np.linspace(0, h, nz + 1)
    zs = (zs[1:] + zs[:-1]) / 2.0

    geom = [rs, ts, zs]

    return np.meshgrid(times, *geom[:self.dim], indexing = 'ij')

  def _generate_surface_mesh(self, t, h, times, nt, nz):
    """
      Generate the appropriate finite difference mesh for a particular problem
      
      Parameters:
        t           thickness
        h           height
        times       discrete time steps
        nr          number of radial increments
        nt          number of circumferential increments
        nz          number of axial increments
    """
    ts = np.linspace(0, 2*np.pi, nt + 1)
    ts = (ts[1:] + ts[:-1]) / 2.0
    zs = np.linspace(0, h, nz + 1)
    zs = (zs[1:] + zs[:-1]) / 2.0

    geom = [ts, zs]

    return np.meshgrid(times, *geom, indexing = 'ij')

