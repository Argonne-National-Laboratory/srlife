"""
  This module defines 1D, 2D, and 3D thermal solvers.
"""

from abc import ABC, abstractmethod

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
    temperatures = FiniteDifferenceImplicitThermalProblem(tube, 
        material, fluid).solve()

    tube.add_results("temperature", temperatures)

class FiniteDifferenceImpliciatThermalProblem:
  """
    The actual finite difference solver created to solve a single
    tube problem
  """
  def __init__(self, tube, material, fluid):
    """
      Parameters:
        tube        Tube object to solve
        material    ThermalMaterial object
        fluid       FluidMaterial object
    """
    self.tube = tube
    self.material = material
    self.fluid = fluid

    self.dr = self.tube.t / self.tube.nr
    self.dt = 2.0 * np.pi / self.tube.nt
    self.dz = self.tube.h / self.tube.nz

    self.rs = np.linspace(self.tube.r - self.dr - self.tube.t, self.tube.r + self.dr, self.tube.nr+3)
    self.rs = np.array([(r1+r2)/2 for r1, r2 in zip(self.rs[:-1], self.rs[1:])])

    self.dts = np.diff(self.tube.times)

    self.dim = (self.tube.nr, self.tube.nt, self.tube.nz)

    if self.tube.abstraction == "3D":
      self.dim = self.dim 
    elif self.tube.abstraction == "2D":
      self.dim = self.dim[:2]
    elif self.tube.abstraction == "1D":
      self.dim = self.dim[:1]
    else:
      raise ValueError("Thermal solver does not know how to handle abstraction %s" % self.tube.abstraction)
    
    # Ghost in r
    self.dim[0] += 2

  def solve(self):
    """
      Actually solve the problem...
    """
    T = np.zeros((self.tube.ntime,) + self.dim)
    T[0] = self.tube.T0
    
    for i,(time,dt) in enumerate(zip(self.tube.times[1:],self.dts)):
      T[i+1] = solve_step(T[i], time, dt)

    return T[:,1:-1] # Don't return the ghost values

  def solve_step(self, T_n, time, dt):
    """
      Solve a single step

      Parameters:
        T_n         previous temperatures
        time        current time
        dt          current dt
    """
    def RJ(self, Ti):
      pass
