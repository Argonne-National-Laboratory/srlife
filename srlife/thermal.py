"""
  This module defines 1D, 2D, and 3D thermal solvers.
"""

from abc import ABC, abstractmethod

import numpy as np
import scipy.optimize as opt

from srlife import receiver

class ThermalSolver(ABC):
  """
    When called this class takes as input:
      1) A Tube object
      2) A ThermalMaterial
      3) The FluidMaterial object 

    The solver must read the Tube abstraction parameter and 
    return a 3D, 2D, or 1D analysis, as appropriate
  """
  @abstractmethod
  def solve(self, tube, material, fluid):
    """
      Solve the thermal problem defined for a single tube

      Parameters:
        tube        the Tube object defining the geometry, loading,
                    and analysis abstraction
        material    the thermal material object describing the material
                    conductivity and diffusivity
        fluid       the fluid material object describing the convective
                    heat transfer coefficient
    """
    pass

class FiniteDifferenceImplicitThermalSolver(ThermalSolver):
  """
    Solves the heat transfer problem using the finite difference method.

    Solver handles the *cylindrical* 1D, 2D, or 3D cases.
  """
  def solve(self, tube, material, fluid, source = None, 
      T0 = None):
    """
      Solve the thermal problem defined for a single tube

      Parameters:
        tube        the Tube object defining the geometry, loading,
                    and analysis abstraction
        material    the thermal material object describing the material
                    conductivity and diffusivity
        fluid       the fluid material object describing the convective
                    heat transfer coefficient

      Other Parameters:
        source:     if present, the source term as a function of t and 
                    then the coordinates
        T0:         if present override the tube IC with a function of
                    the coordinates
    """
    temperatures = FiniteDifferenceImplicitThermalProblem(tube, 
        material, fluid, source, T0).solve()

    tube.add_results("temperature", temperatures)

class FiniteDifferenceImplicitThermalProblem:
  """
    The actual finite difference solver created to solve a single
    tube problem
  """
  def __init__(self, tube, material, fluid, source = None, T0 = None):
    """
      Parameters:
        tube        Tube object to solve
        material    ThermalMaterial object
        fluid       FluidMaterial object

      Other Parameters:
        source      source function (t, r, ...)
        T0          initial condition function
    """
    self.tube = tube
    self.material = material
    self.fluid = fluid

    self.source = source
    self.T0 = T0

    self.dr = self.tube.t / self.tube.nr
    self.dt = 2.0 * np.pi / self.tube.nt
    self.dz = self.tube.h / self.tube.nz

    self.rs = np.linspace(self.tube.r - self.dr - self.tube.t, self.tube.r + self.dr, self.tube.nr+3)
    self.rs = np.array([(r1+r2)/2 for r1, r2 in zip(self.rs[:-1], self.rs[1:])])

    self.dts = np.diff(self.tube.times)

    self.dim = (self.tube.nr, self.tube.nt, self.tube.nz)

    if self.tube.abstraction == "3D":
      self.dim = self.dim 
      self.ndim = 3
    elif self.tube.abstraction == "2D":
      self.dim = self.dim[:2]
      self.ndim = 2
    elif self.tube.abstraction == "1D":
      self.dim = self.dim[:1]
      self.ndim = 1
    else:
      raise ValueError("Thermal solver does not know how to handle abstraction %s" % self.tube.abstraction)

    # Ghost in r
    self.dim = list(self.dim)
    self.dim[0] += 2
    self.dim = tuple(self.dim)

    # Useful for later
    self.mesh = self._generate_mesh()
    self.r = self.mesh[0]
    
    if self.ndim > 1:
      self.theta = self.mesh[1]
    else:
      self.theta = self.tube.angle

    if self.ndim > 2:
      self.z = self.mesh[2]
    else:
      self.z = self.tube.plane
    

  def solve(self):
    """
      Actually solve the problem...
    """
    # Setup the initial time
    T = np.zeros((self.tube.ntime,) + self.dim)
    if self.T0 is not None:
      T[0] = self.T0(*self.mesh)
    else:
      T[0] = self.tube.T0
    
    for i,(time,dt) in enumerate(zip(self.tube.times[1:],self.dts)):
      T[i+1] = self.solve_step(T[i], time, dt)

    return T[:,1:-1] # Don't return the ghost values

  def solve_step(self, T_n, time, dt):
    """
      Solve a single step

      Parameters:
        T_n         previous temperatures
        time        current time
        dt          current dt
    """
    T = np.copy(T_n)

    sol = opt.root(lambda x: self.RJ(x, T_n, time, dt), T.flatten(), method = 'lm')
    T = sol.x.reshape(T_n.shape)

    return T

  def RJ(self, T, T_n, time, dt):
    """
      Actually form the residual and jacobian for a step

      Parameters:
        T       current temperatures
        T_n     previous temperatures

    """
    # Some setup
    T = T.reshape(T_n.shape)

    k = self.material.conductivity(T)
    a = self.material.diffusivity(T)

    dTdr = (T[2:] - T[:-2]) / (2.0 * self.dr)
    d2Tdr2 = (T[2:] - 2.0*T[1:-1] + T[:-2]) / self.dr**2.0

    dkdr = (k[2:] - k[:-2]) / (2.0 * self.dr)

    R = np.zeros(T.shape)

    # 1) The inertia term
    R[1:-1] += k[1:-1] / a[1:-1] * (T[1:-1] - T_n[1:-1])

    # 2) The r term
    R[1:-1] -= (dkdr * dTdr + k[1:-1] / self.r[1:-1]*dTdr + k[1:-1]*d2Tdr2) * dt
    
    # The left BC
    # Zero flux
    if self.tube.inner_bc is None:
      R[0] = T[2] - T[0]
    elif isinstance(self.tube.inner_bc, receiver.FixedTempBC):
      R[0] = T[0] - self.tube.inner_bc.temperature(time, self.theta, self.z)

    # The right BC
    # Zero flux
    if self.tube.outer_bc is None:
      R[-1] = T[-1] - T[-3]
    elif isinstance(self.tube.inner_bc, receiver.FixedTempBC):
      R[-1] = T[-1] - self.tube.outer_bc.temperature(time, self.theta, self.z)

    # 3) The theta term
    if self.ndim == 2:
      pass
    
    # 4) The z term
    if self.ndim == 3:
      pass
    
    # 5) Source term (only used for testing at this point)
    if self.source is not None:
      R[1:-1] -= self.source(time, *self.mesh)[1:-1] * dt
    
    return R.flatten()
  
  def _generate_mesh(self):
    """
      Produce the r, theta, z mesh
    """
    rs = np.linspace(self.tube.r - self.tube.t - self.dr, self.tube.r + self.dr, self.tube.nr + 3)
    rs = (rs[:-1] + rs[1:]) / 2.0
    ts = np.linspace(0, 2.0*np.pi, self.tube.nt + 1)
    ts = (ts[:-1] + ts[1:]) / 2.0
    zs = np.linspace(0, self.tube.h, self.tube.nz + 1)
    zs = (zs[:-1] + zs[1:]) / 2.0

    geom = [rs, ts, zs]

    return np.meshgrid(*geom[:self.ndim], indexing = 'ij', copy = True)
