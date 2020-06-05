"""
  This module defines 1D, 2D, and 3D thermal solvers.
"""

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

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

    self.source_term = source
    self.T0 = T0

    self.dr = self.tube.t / (self.tube.nr-1)
    self.dt = 2.0 * np.pi / (self.tube.nt)
    self.dz = self.tube.h / (self.tube.nz-1)

    self.dts = np.diff(self.tube.times)
  
    # Ghost
    self.dim = (self.tube.nr + 2, self.tube.nt + 2, self.tube.nz + 2)

    if self.tube.abstraction == "3D":
      self.dim = self.dim
      self.nr, self.nt, self.nz = self.dim
      self.ndim = 3
    elif self.tube.abstraction == "2D":
      self.dim = self.dim[:2]
      self.nr, self.nt = self.dim
      self.ndim = 2
    elif self.tube.abstraction == "1D":
      self.dim = self.dim[:1]
      self.nr, = self.dim
      self.ndim = 1
    else:
      raise ValueError("Thermal solver does not know how to handle abstraction %s" % self.tube.abstraction)

    # Useful for later
    self.mesh = self._generate_mesh()
    self.r = self.mesh[0]

    if self.ndim > 1:
      self.theta = self.mesh[1]
    else:
      self.theta = np.array([[self.tube.angle]])

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
    
    # Don't return ghost values
    if self.ndim == 3:
      return T[:,1:-1,1:-1,1:-1]
    elif self.ndim == 2:
      return T[:,1:-1,1:-1]
    else:
      return T[:,1:-1]

  def solve_step(self, T_n, time, dt):
    """
      Solve a single step

      Parameters:
        T_n         previous temperatures
        time        current time
        dt          current dt
    """
    # Store reusable stuff
    self.setup_step(T_n, time, dt)

    if self.ndim == 1:
      A, b = self.form_1D_system(T_n, time, dt)

    elif self.ndim == 2:
      A, b = self.form_2D_system(T_n, time, dt)

    T_np1 =  sla.spsolve(A, b).reshape(self.dim)
    
    return T_np1

  def form_1D_system(self, T_n, time, dt):
    """
      Form the 1D system of equations to solve for the updated temperatures

      Parameters:
        T_n:        previous temperature
        time:       current time
        dt:         current increment
    """
    # Matrix entries
    main = np.zeros((self.ndof,))
    upper = np.zeros((self.ndof-1,))
    lower = np.zeros((self.ndof-1,))
    RHS = np.zeros((self.ndof,))
    
    # Form matrix and source
    # Radial contribution
    U, D, L = self.radial(T_n, time)
    upper[1:] = U
    main[1:-1] = D
    lower[:-1] = L

    # Source
    RHS[1:-1] = -self.source(T_n, time)
    
    # Impose r BCs
    D, U, R = self.impose_left_radial_bc(T_n[1], T_n[0], time)
    main[:1] = D
    upper[:1] = U
    RHS[:1] = R

    D, L, R = self.impose_right_radial_bc(T_n[-2], T_n[-1], time)
    main[-1:] = D
    lower[-1:] = L
    RHS[-1:] = R

    # Apply the dt and 1.0+
    upper[1:] *= dt
    main[1:-1] *= dt
    main[1:-1] += 1
    lower[:-1] *= dt
    
    # Setup matrix and RHS
    A = sp.diags((upper,main,lower), (1,0,-1), shape = (self.ndof, self.ndof)).tocsr()
    b = RHS
    b[1:-1] *= dt
    b[1:-1] += T_n[1:-1].flatten()

    return A, b

  def form_2D_system(self, T_n, time, dt):
    """
      Form the 2D implicit system of equations

      Parameters:
        T_n:        previous temperature
        time:       current time
        dt:         time increment
      
      Returns:
        A matrix and b vector
    """
    # Matrix entries
    RHS = np.zeros((self.ndof,))
    main = np.zeros((self.ndof,))
    u1 = self.nt
    upper1 = np.zeros((self.ndof - u1,))
    l1 = -self.nt
    lower1 = np.zeros((self.ndof + l1,))
    u2 = 1
    upper2 = np.zeros((self.ndof - u2,))
    l2 = -1
    lower2 = np.zeros((self.ndof + l2,))
    u3 = self.nt - 2
    upper3 = np.zeros((self.ndof - u3,))
    l3 = -(self.nt-2)
    lower3 = np.zeros((self.ndof + l3,))

    
    # Radial contribution
    U, D, L = self.radial(T_n, time)
    for i in range(self.nr-2):
      upper1[1+self.nt+i*self.nt:1+self.nt+(i+1)*self.nt-2] += U[i*(self.nt-2):(i+1)*(self.nt-2)] * dt
      main[1+self.nt+i*self.nt:1+self.nt+(i+1)*self.nt-2] += D[i*(self.nt-2):(i+1)*(self.nt-2)] * dt
      lower1[1+i*self.nt:1+(i+1)*self.nt-2] += L[i*(self.nt-2):(i+1)*(self.nt-2)] * dt

    # Circumferential contribution
    U, D, L = self.circumfrential(T_n, time)
    for i in range(self.nr-2):
      upper2[1+self.nt+i*self.nt:1+self.nt+(i+1)*self.nt-2] += U[i*(self.nt-2):(i+1)*(self.nt-2)] * dt
      main[1+self.nt+i*self.nt:1+self.nt+(i+1)*self.nt-2] += D[i*(self.nt-2):(i+1)*(self.nt-2)] * dt
      lower2[self.nt+i*self.nt:self.nt+(i+1)*self.nt-2] += L[i*(self.nt-2):(i+1)*(self.nt-2)] * dt
    
    # Source and T_n contribution
    R = self.source(T_n, time)
    Tf = T_n.flatten()
    for i in range(self.nr-2):
      RHS[1+self.nt+i*self.nt:1+self.nt+(i+1)*self.nt-2] += (-R[i*(self.nt-2):(i+1)*(self.nt-2)] * dt + Tf[i*(self.nt-2):(i+1)*(self.nt-2)])

    # Radial left BC
    D, U, R = self.impose_left_radial_bc(T_n[1], T_n[0], time)
    main[:self.nt] = D
    upper1[:self.nt] = U
    RHS[:self.nt] = R

    # Radial right BC
    D, L, R = self.impose_right_radial_bc(T_n[-2], T_n[-1], time)
    main[-self.nt:] = D
    lower1[-self.nt:] = L
    RHS[-self.nt:] = R

    # Theta left BC
    main[::self.nt] = 1.0
    upper3[::self.nt] = -1.0
    RHS[::self.nt] = 0

    # Theta right BC
    main[self.nt-1::self.nt] = 1.0
    lower3[1::self.nt] = -1.0
    RHS[self.nt-1::self.nt] = 0

    # Add 1 to the diagonal
    for i in range(self.nr-2):
      main[1+self.nt+i*self.nt:1+self.nt+(i+1)*self.nt-2] += 1.0

    # Setup matrix and RHS
    A = sp.diags((upper1,upper2,upper3,main,lower1,lower2,lower3), (u1,u2,u3,0,l1,l2,l3), shape = (self.ndof, self.ndof)).tocsr()

    return A, RHS

  def setup_step(self, T, time, dt):
    """
      Setup common reusable arrays

      Parameters:
        T           temperature of interest
        time        current time
        dt          current dt
    """
    self.k = self.material.conductivity(T)
    self.a = self.material.diffusivity(T)
    
    ae = np.pad(self.a, self._pad_values(0), mode = 'edge')
    self.a_r = 0.5*(ae[1:] + ae[:-1])
    re = np.pad(self.r, self._pad_values(0), mode = 'edge')
    self.r_r = 0.5*(re[1:] + re[:-1])

    self.ndof = T.size

    if self.ndim == 2:
      ae = np.pad(self.a, self._pad_values(1), mode = 'edge')
      self.a_t = 0.5*(ae[:,1:] + ae[:,:-1])
      re = np.pad(self.r, self._pad_values(1), mode = 'edge')
      self.r_t = 0.5*(re[:,1:] + re[:,:-1])

  def _pad_values(self, axis):
    """
      Pad with zeros along a given axis

      Parameters:
        axis:       which axis!
    """
    res = []
    for i in range(self.ndim):
      if i == axis:
        res.append((1,1))
      else:
        res.append((0,0))

    return tuple(res)

  def adofs(self, X):
    """
      Return a view into the actual non-dummy dofs

      Parameters:
        X       matrix to reduce
    """
    if self.ndim == 3:
      return X[1:-1,1:-1,1:-1]
    elif self.ndim == 2:
      return X[1:-1,1:-1]
    else:
      return X[1:-1]

  def radial(self, T, time):
    """
      The tridiagonal radial contributions to the problem

      Parameters:
        T:      full temperature field at which we want to calculate coefficients
        time:   current time

      Returns:
        upper, diagonal, and lower matrix entries
    """
    p1 = self.r_r[1:] * self.a_r[1:]
    p2 = self.r_r[:-1] * self.a_r[:-1]

    U = self.adofs((1.0 / self.r * 1.0 / self.dr**2.0 * p1)).flatten()
    D = self.adofs((-1.0 / self.r * 1.0 / self.dr**2.0 * (p1 + p2))).flatten()
    L = self.adofs((1.0 / self.r * 1.0 / self.dr**2.0 * p2)).flatten()

    return U,D,L

  def impose_left_radial_bc(self, Tb, Tg, time):
    """
      Impose the radial BC on the left side of the domain

      Parameters:
        Tb:     temperatures on actual boundary
        Tg:     ghost temperatures
        time:   time

      Returns:
        Upper, diagonal, and RHS entries
    """
    # Zero flux
    if self.tube.inner_bc is None:
      D = -1.0
      U = 1.0
      RHS = 0.0
    # Fixed temperature
    elif isinstance(self.tube.inner_bc, receiver.FixedTempBC):
      D = 0.0
      U = 1.0
      RHS = np.array([self.tube.inner_bc.temperature(time, t, self.z) for t in self.theta[0]]).flatten()
    else:
      raise ValueError("Unknown boundary condition!")

    return D, U, RHS

  def impose_right_radial_bc(self, Tb, Tg, time):
    """
      Impose the radial BC on the right side of the domain

      Parameters:
        Tb:     temperatures on actual boundary
        Tg:     ghost temperatures
        time:   time

      Returns:
        diagonal, lower, and RHS entries
    """
    # Zero flux
    if self.tube.outer_bc is None:
      D = -1.0
      L = 1.0
      RHS = 0.0
    # Fixed temperature
    elif isinstance(self.tube.outer_bc, receiver.FixedTempBC):
      D = 0.0
      L = 1.0
      RHS = np.array([self.tube.outer_bc.temperature(time, t, self.z) for t in self.theta[-1]]).flatten()
    else:
      raise ValueError("Unknown boundary condition!")

    return D, L, RHS

  def circumfrential(self, T, time):
    """
      The tridiagonal + periodic theta contributions to the problem

      Parameters:
        T:      full temperature field at which we want to calculate coefficients
        time:   current time

      Returns:
        upper, diagonal, and lower entries
    """
    p1 = self.a_t[:,1:]
    p2 = self.a_t[:,:-1]

    U = self.adofs(1.0 / self.r**2.0 * 1.0 / self.dt**2.0 * p1).flatten()
    D = self.adofs(-1.0 / self.r**2.0 * 1.0 / self.dt**2.0 * (p1 + p2)).flatten()
    L = self.adofs(1.0 / self.r**2.0 * 1.0 / self.dt**2.0 * p2).flatten()

    return U, D, L

  def axial(self, T, time, dt):
    pass

  def source(self, T, time):
    """
      RHS source term

      Parameters:
        T:      full temperature field at which we want to calculate coefficients
        time:   current time
        R:      RHS to add to

      Returns:
        source contribution
    """
    return self.adofs(self.a / self.k * self.source_term(time, *self.mesh)).flatten()

  def _generate_mesh(self):
    """
      Produce the r, theta, z mesh
    """
    rs = np.linspace(self.tube.r - self.tube.t - self.dr, self.tube.r + self.dr, self.tube.nr + 2)
    ts = np.linspace(0 - self.dt, 2.0*np.pi, self.tube.nt + 2)
    zs = np.linspace(0 - self.dz, self.tube.h + self.dz, self.tube.nz + 2)

    geom = [rs, ts, zs]

    return np.meshgrid(*geom[:self.ndim], indexing = 'ij', copy = True)
