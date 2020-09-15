import sys
sys.path.append('../../..')

import numpy as np
import matplotlib.pyplot as plt

from srlife import thermal, receiver, materials

class ManufacturedSolution:
  """
    A manufactured heat transport solution
  """
  def __init__(self, name, dim, solution, source):
    """
      Parameters:
        name:       descriptive name
        dim:        dimension of solution
        solution:   function of t, r, ...
        source      function of t, k, alpha, r, ...
        flux        function of t, k, alpha, r, ... (for BC)
    """
    self.name = name
    self.dim = dim
    self.soln = solution
    self.source = source

  def solve(self, solver, thermal, fluid, r = 1.0, t = 0.2, h = 1, time = 1, 
      ntime = 11, nr = 11, nt = 20, nz = 10, T0 = 0.0):
    """
      Generate the appropriate tube and solve with the provided solver
      
      Parameters:
        solver:         the thermal solver to test
        thermal:        the thermal model to test

      Other Parameters:
        r:              tube radius
        t:              tube thickness
        h:              tube height
        time:           maximum time
        ntime:          number of time steps
        nr:             number of radial increments
        nt:             number of circumferential increments
        nz:             number of axial increments
        T0:             initial temperature
    """
    tube = receiver.Tube(r, t, h, nr, nt, nz, T0)

    times = np.linspace(0, time, ntime)
    tube.set_times(times)

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
        source = sfn, T0 = T0, fix_edge = self.soln)

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
    
    print(self.name)
    print("Max absolute error: %e" % np.max(np.abs(T - soln.results['temperature'])))
    keep = np.abs(T) > 0
    print("Max relative error: %e" % np.max(np.abs(T[keep] - soln.results['temperature'][keep])/np.abs(T[keep])))
    print("")

    plt.figure()
    if self.dim == 1:
      plot = [1, soln.nr // 2, -2]
      for p in plot:
        l, = plt.plot(mesh[0][:,p], soln.results['temperature'][:,p])
        plt.plot(mesh[0][:,p], T[:,p], ls = '--', color = l.get_color())
      plt.xlabel("Time")
      plt.ylabel("Temperature")
      plt.title("1D: left, middle, right")
    elif self.dim == 2:
      plot_r = [0,1, soln.nr // 2, -2,-1]
      plot_t = [0,1, soln.nt // 2, -2,-1]

      for rp in plot_r:
        for tp in plot_t:
          l, = plt.plot(mesh[0][:,rp,tp], soln.results['temperature'][:,rp,tp])
          plt.plot(mesh[0][:,rp,tp], T[:,rp, tp], ls = '--', color = l.get_color())
      plt.xlabel("Time")
      plt.ylabel("Temperature")
      plt.title("2D, r/theta slices")
    elif self.dim == 3:
      plot_r = [0,1, soln.nr // 2, -2,-1]
      plot_t = [0,1, soln.nt // 2, -2,-1]
      plot_z = [0,1, soln.nz // 2, -2,-1]
      for rp in plot_r:
        for tp in plot_t:
          for zp in plot_z:
            l, = plt.plot(mesh[0][:,rp,tp,zp], soln.results['temperature'][:,rp,tp,zp])
            plt.plot(mesh[0][:,rp,tp,zp], T[:,rp, tp,zp], ls = '--', color = l.get_color())
      plt.xlabel("Time")
      plt.ylabel("Temperature")
      plt.title("3D, r/theta/z slices")

  def assess_comparison(self, soln, tol, atol):
    """
      Return true if solution matches, false otherwise

      Parameters:
        soln:           particular tube object with the solution
        tol:            relative tolerance
    """
    mesh = self._generate_mesh(soln.r, soln.t, soln.h, soln.times, 
        soln.nr, soln.nt, soln.nz)
    T = self.soln(*mesh)

    return np.allclose(T, soln.results['temperature'], rtol = tol, atol = atol)

  def _generate_mesh(self, r, t, h, times, nr, nt, nz):
    """
      Generate the appropriate finite difference mesh for a particular problem
      
      Parameters:
        r:           radius
        t:           thickness
        h            height
        times:       discrete time steps
        nr:          number of radial increments
        nt:          number of circumferential increments
        nz:          number of axial increments
    """
    rs = np.linspace(r-t, r, nr)
    ts = np.linspace(0, 2*np.pi, nt + 1)[:-1]
    zs = np.linspace(0, h, nz)

    geom = [rs, ts, zs]

    return np.meshgrid(times, *geom[:self.dim], indexing = 'ij')

  def _generate_surface_mesh(self, t, h, times, nt, nz):
    """
      Generate the appropriate finite difference mesh for a particular problem
      
      Parameters:
        t:           thickness
        h:           height
        times:       discrete time steps
        nr:          number of radial increments
        nt:          number of circumferential increments
        nz:          number of axial increments
    """
    ts = np.linspace(0, 2*np.pi, nt + 1)[:-1]
    zs = np.linspace(0, h, nz)

    geom = [ts, zs]

    return np.meshgrid(times, *geom, indexing = 'ij')

