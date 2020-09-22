"""
  This module defines 1D, 2D, and 3D thermal solvers.
"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from srlife import receiver, solverparams

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
    return

class FiniteDifferenceImplicitThermalSolver(ThermalSolver):
  """
    Solves the heat transfer problem using the finite difference method.

    Solver handles the *cylindrical* 1D, 2D, or 3D cases.
  """
  def __init__(self, pset = solverparams.ParameterSet(),
      rtol = 1.0e-6, atol = 1.0e-2, miter = 100, substep = 1, verbose = False):
    """
      Setup the solver

      Additional parameters:
        pset        object with solver parameters 
        rtol        iteration relative tolerance
        atol        iteration absolute tolerance
        miter       maximum iterations
        substep     divide user-provided time increments into smaller values 
        verbose     print a lot of debug info
    """
    self.rtol = pset.get_default("rtol", rtol)
    self.atol = pset.get_default("atol", atol)
    self.miter = pset.get_default("miter", miter)
    self.substep = pset.get_default("substep", substep)
    self.verbose = pset.get_default("verbose", verbose)

  def solve(self, tube, material, fluid, source = None, 
      T0 = None, fix_edge = None, rtol = 1e-6, atol = 1e-2, 
      miter = 100, substep = 1):
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
        fix_edge:   an exact solution to fix edge BCs for testing
    """
    temperatures = FiniteDifferenceImplicitThermalProblem(tube, 
        material, fluid, source, T0, fix_edge, self.rtol, self.atol, 
        self.miter, self.substep, self.verbose).solve()

    tube.add_results("temperature", temperatures)

    return temperatures

class FiniteDifferenceImplicitThermalProblem:
  """
    The actual finite difference solver created to solve a single
    tube problem
  """
  def __init__(self, tube, material, fluid, source = None, T0 = None,
      fix_edge = None, rtol = 1.0e-6, atol = 1e-8, miter= 50, substep = 1,
      verbose = False):
    """
      Parameters:
        tube        Tube object to solve
        material    ThermalMaterial object
        fluid       FluidMaterial object

      Other Parameters:
        source      source function (t, r, ...)
        T0          initial condition function
        fix_edge:   an exact solution to fix edge BCs for testing
        rtol        relative tolerance
        atol        absolute tolerance
        miter       maximum iterations
        substep     divide user provided time increments into smaller steps
        verbose     print a lot of debug info
    """
    self.tube = tube
    self.material = material
    self.fluid = fluid
    
    self.rtol = rtol
    self.atol = atol
    self.miter = miter

    self.substep = substep

    self.verbose = verbose

    self.source_term = source
    self.T0 = T0
    self.fix_edge = fix_edge

    self.dr = self.tube.t / (self.tube.nr-1)
    self.dt = 2.0 * np.pi / (self.tube.nt)
    self.dz = self.tube.h / (self.tube.nz-1)

    self.dts = np.diff(self.tube.times)

    # Ghost
    self.dim = (self.tube.nr + 2, self.tube.nt+2, self.tube.nz + 2)

    if self.tube.abstraction == "3D":
      self.dim = self.dim
      self.nr, self.nt, self.nz = self.dim
      self.ndim = 3
      self.fdim = self.dim
    elif self.tube.abstraction == "2D":
      self.dim = self.dim[:2]
      self.nr, self.nt = self.dim
      self.nz = 1
      self.ndim = 2
      self.fdim = (self.nr, self.nt, 1)
    elif self.tube.abstraction == "1D":
      self.dim = self.dim[:1]
      self.nr, = self.dim
      self.nt = 1
      self.nz = 1
      self.ndim = 1
      self.fdim = (self.nr, 1, 1)
    else:
      raise ValueError("Thermal solver does not know how to handle"
      " abstraction %s" % self.tube.abstraction)

    # Useful for later
    self.mesh = self._generate_mesh()
    self.r = self.mesh[0].reshape(self.fdim)

    if self.ndim > 1:
      self.theta = self.mesh[1].reshape(self.fdim)
    else:
      self.theta = np.ones(self.r.shape) * self.tube.angle

    if self.ndim > 2:
      self.z = self.mesh[2].reshape(self.fdim)
    else:
      self.z = np.ones(self.r.shape) * self.tube.plane
   
  def _generate_mesh(self):
    """
      Produce the r, theta, z mesh
    """
    rs = np.linspace(self.tube.r - self.tube.t - self.dr, self.tube.r + self.dr, 
        self.tube.nr + 2)
    ts = np.linspace(-self.dt , 2.0*np.pi, self.tube.nt+2)
    zs = np.linspace(0 - self.dz, self.tube.h + self.dz, self.tube.nz + 2)

    geom = [rs, ts, zs]

    return np.meshgrid(*geom[:self.ndim], indexing = 'ij', copy = True)

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
    
    # Iterate through steps
    for i,(time,dt) in enumerate(zip(self.tube.times[1:],self.dts)):
      T[i+1] = self.solve_step_substep(T[i], time, dt)
    
    # Don't return ghost values
    if self.ndim == 3:
      return T[:,1:-1,1:-1,1:-1]
    elif self.ndim == 2:
      return T[:,1:-1,1:-1]
    else:
      return T[:,1:-1]

  def solve_step_substep(self, T_n, time, dt):
    """
      Do substepping, if requested by the user

      Parameters:
        T_n         previous full step temperature
        time        target time
        dt          target dt
    """
    T = np.copy(T_n)
    t_n = time - dt
    dti = dt / self.substep

    for i in range(1, self.substep + 1):
      t = t_n + dti * i
      T = self.solve_step(T, t, dti)

    return T

  def setup_step(self, T):
    """
      Setup common reusable arrays

      Parameters:
        T           temperature of interest
        time        current time
        dt          current dt
    """
    self.k = self.material.conductivity(T).reshape(self.fdim)
    self.a = self.material.diffusivity(T).reshape(self.fdim)

    # Useful matrix that gives you the actual dofs
    self.act = np.pad(np.ones(tuple(d-2 for d in self.dim)), [(1,1)] * self.ndim, 
        constant_values = [(0,0)] * self.ndim)

    self.act = self.act.reshape(self.fdim)

    self.ndof = T.size

  def _generate_A(self):
    """
      Generate the base differential operator for the current step
    """
    A = self.radial()

    if self.ndim > 1:
      A += self.circumfrential()
    
    if self.ndim > 2:
      A += self.axial()
    
    return A

  def _generate_id(self):
    """
      Return the non-boundary affecting ID matrix
    """
    return sp.diags([self.act.flatten()], offsets = (0,),
        shape = (self.ndof, self.ndof), format = 'coo')

  def _generate_source(self, time):
    """
      Generate the source part of the RHS

      Parameters:
        time        current time
    """
    if self.source_term:
      return (self.act * self.a / self.k * 
          self.source_term(time,*[self.r, self.theta, self.z][:self.ndim])).flatten()
    else:
      return np.zeros((self.ndof,))

  def _generate_prev_temp(self, T_n):
    """
      The RHS vector with the temperature contributions in the correct locations

      Parmaeters:
        T_n         previous temperatures
    """
    return (T_n * self.act).flatten()

  def _generate_bc_matrix(self, T_n, time, dt):
    """
      Generate the BC matrix terms (fixed)
    """
    M = self._ID_BC() + self._OD_BC()

    if self.ndim > 1:
      M += self._left_BC() + self._right_BC()

    if self.ndim > 2:
      M += self._top_BC() + self._bot_BC()

    return M

  def _generate_fixed_bc_RHS(self, T_n, time, dt):
    """
      Generate the constant (i.e. axial) contributions to the RHS

      Parameters:

      T_n       previous temperature
      time      current time
      dt        time increment
    """
    if self.ndim > 2:
      return self._top_BC_R(T_n, time, T_n) + self._bot_BC_R(T_n, time, T_n)
    else:
      return np.zeros((self.ndof,))
  
  # pylint: disable=too-many-locals
  def solve_step(self, T_n, time, dt):
    """
      Actually setup and solve for a single load step

      Parameters:
        T_n         previous temperatures
        time        current time
        dt          time increment
    """
    # Generic setup
    self.setup_step(T_n)

    # Add dimensions, if necessary
    T_n = T_n.reshape(self.fdim)

    # FD contributions
    A = self._generate_A()
    # Identity
    ID = self._generate_id()
    # Source term
    S = self._generate_source(time) 
    # Previous temp
    Tn = self._generate_prev_temp(T_n)
    
    # System without BCs
    M = (ID-A*dt)
    R = S*dt + Tn
    
    # Matrix and fixed RHS boundary contributions
    B = self._generate_bc_matrix(T_n, time, dt)
    M += B    
    BRF = self._generate_fixed_bc_RHS(T_n, time, dt)
    R += BRF
    
    # Dummy dofs
    D = self._generate_dummy_dofs()
    M += D

    # Covert over our fixed matrix
    M = M.tocsr()
    
    # This would be the iteration step
    T = np.copy(T_n)

    for i in range(self.miter):
      Ri = R + self._ID_BC_R(T, time, T_n) + self._OD_BC_R(T, time, T_n)
      J = self._d_ID_BC_R(T, time, T_n) + self._d_OD_BC_R(T, time, T_n)

      res = M.dot(T.flatten()) - Ri
      nr = la.norm(res)

      if i == 0:
        nr0 = nr

      if (nr < self.atol or nr/nr0 < self.rtol) and i > 0:
        break

      T -= sla.spsolve(M - J, res).reshape(self.fdim)
    else:
      raise RuntimeError("Too many iterations in newton solver!")
    
    return T.reshape(self.dim)
  
  # pylint: disable=too-many-branches
  def _generate_dummy_dofs(self):
    """
      Provide on-diagonal 1.0s for the dummy dofs
    """
    I = []
    J = []

    if self.ndim == 2:
      for i in self.dummy_loop_r():
        for j in self.dummy_loop_t():
          for k in self.dummy_loop_z():
            I.append(self.dof(i,j,k))
            J.append(self.dof(i,j,k))
    elif self.ndim == 3:
      for i in self.dummy_loop_r():
        for j in self.dummy_loop_t():
          for k in self.full_loop_z():
            I.append(self.dof(i,j,k))
            J.append(self.dof(i,j,k))
      for i in self.full_loop_r():
        for j in self.dummy_loop_t():
          for k in self.dummy_loop_z():
            I.append(self.dof(i,j,k))
            J.append(self.dof(i,j,k))
      for i in self.dummy_loop_r():
        for j in self.full_loop_t():
          for k in self.dummy_loop_z():
            I.append(self.dof(i,j,k))
            J.append(self.dof(i,j,k))

    return sp.coo_matrix((np.ones((len(I),)),(I,J)), shape = (self.ndof, self.ndof))

  def _ID_BC_R(self, T, time, T_n):
    """
      The inner diameter BC RHS contribution

      Parameters:
        T       current temperatures
        time    current time
        T_n     previous temperatures
    """
    R = np.zeros((self.ndof,))
    i = 0
    for j in self.loop_t():
      for k in self.loop_z():
        if self.fix_edge:
          R[self.dof(i,j,k)] = self.fix_edge(time, *[self.r[i,j,k], 
            self.theta[i,j,k], self.z[i,j,k]][:self.ndim])
        # Zero flux
        elif self.tube.inner_bc is None:
          R[self.dof(i,j,k)] = 0.0
        # Fixed temperature
        elif isinstance(self.tube.inner_bc, receiver.FixedTempBC):
          R[self.dof(i,j,k)] = self.tube.inner_bc.temperature(time, 
              self.theta[1,j,k], self.z[1,j,k])
        # Fixed flux
        elif isinstance(self.tube.inner_bc, receiver.HeatFluxBC):
          R[self.dof(i,j,k)] = -self.dr * self.tube.inner_bc.flux(time, 
              self.theta[1,j,k], self.z[1,j,k]) / self.k[1,j,k]
        # Convection
        elif isinstance(self.tube.inner_bc, receiver.ConvectiveBC):
          R[self.dof(i,j,k)] = self.dr * self.fluid.coefficient(self.material.name,
              T_n[1,j,k]) * (T[1,j,k] - self.tube.inner_bc.fluid_temperature(
                time, self.z[1,j,k])) / self.k[1,j,k]
        else:
          raise ValueError("Unknown boundary condition!")
    return R

  def _d_ID_BC_R(self, T, time, T_n):
    """
      Derivative of the inner diameter BC RHS contribution

      Parameters:
        T       current temperatures
        time    current time
        T_n     previous temperatures
    """
    I = []
    J = []
    D = []
    if isinstance(self.tube.inner_bc, receiver.ConvectiveBC):
      i = 0
      for j in self.loop_t():
        for k in self.loop_z():
          I.append(self.dof(i,j,k))
          J.append(self.dof(1,j,k))
          D.append(self.dr * self.fluid.coefficient(self.material.name, 
            T_n[1,j,k]) / self.k[1,j,k])
    
    return sp.coo_matrix((D,(I,J)), shape = (self.ndof, self.ndof))

  def _OD_BC_R(self, T, time, T_n):
    """
      The outer diameter BC RHS contribution

      Parameters:
        T       current temperatures
        time    current time
        T_n     previous temperatures
    """
    R = np.zeros((self.ndof,))
    i = self.nr-1
    for j in self.loop_t():
      for k in self.loop_z():
        if self.fix_edge:
          R[self.dof(i,j,k)] = self.fix_edge(time, *[self.r[i,j,k], 
            self.theta[i,j,k], self.z[i,j,k]][:self.ndim])
        # Zero flux
        elif self.tube.outer_bc is None:
          R[self.dof(i,j,k)] = 0.0
        # Fixed temperature
        elif isinstance(self.tube.outer_bc, receiver.FixedTempBC):
          R[self.dof(i,j,k)] = self.tube.outer_bc.temperature(time, 
              self.theta[self.nr-2,j,k], self.z[self.nr-2,j,k])
        # Fixed flux
        elif isinstance(self.tube.outer_bc, receiver.HeatFluxBC):
          R[self.dof(i,j,k)] = -self.dr * self.tube.outer_bc.flux(time, 
              self.theta[self.nr-2,j,k], self.z[self.nr-2,j,k]) / self.k[self.nr-2,j,k]
        # Convection
        elif isinstance(self.tube.outer_bc, receiver.ConvectiveBC):
          R[self.dof(i,j,k)] = (self.dr *
              self.fluid.coefficient(self.material.name, T_n[self.nr-2,j,k]) * 
              (T[self.nr-2,j,k] - 
                self.tube.outer_bc.fluid_temperature(time, self.z[self.nr-2,j,k])) / 
              self.k[self.nr-2,j,k])
        else:
          raise ValueError("Unknown boundary condition!")
    return R

  def _d_OD_BC_R(self, T, time, T_n):
    """
      Derivative of the outer diameter BC RHS contribution

      Parameters:
        T       current temperatures
        time    current time
        T_n     previous temperatures
    """
    I = []
    J = []
    D = []
    if isinstance(self.tube.outer_bc, receiver.ConvectiveBC):
      i = self.nr-1
      for j in self.loop_t():
        for k in self.loop_z():
          I.append(self.dof(i,j,k))
          J.append(self.dof(self.nr-2,j,k))
          D.append(self.dr * self.fluid.coefficient(self.material.name, 
            T_n[self.nr-2,j,k]) / self.k[self.nr-2,j,k])
    
    return sp.coo_matrix((D,(I,J)), shape = (self.ndof, self.ndof))

  def _ID_BC(self):
    """
      Inner diameter boundary condition contribution matrix
    """
    I = []
    J = []
    D = []
    i = 0
    for j in self.loop_t():
      for k in self.loop_z():
        if self.fix_edge:
          I.append(self.dof(i,j,k)) 
          J.append(self.dof(i,j,k))
          D.append(1.0)
        # Zero flux
        elif self.tube.inner_bc is None:
          I.append(self.dof(i,j,k)) 
          J.append(self.dof(1,j,k))
          D.append(1.0)
          I.append(self.dof(i,j,k)) 
          J.append(self.dof(0,j,k))
          D.append(-1.0)
        # Fixed temperature
        elif isinstance(self.tube.inner_bc, receiver.FixedTempBC):
          I.append(self.dof(i,j,k)) 
          J.append(self.dof(1,j,k))
          D.append(1.0)
        # Fixed flux
        elif isinstance(self.tube.inner_bc, receiver.HeatFluxBC):
          I.append(self.dof(i,j,k)) 
          J.append(self.dof(1,j,k))
          D.append(1.0)
          I.append(self.dof(i,j,k)) 
          J.append(self.dof(0,j,k))
          D.append(-1.0)
        # Convection
        elif isinstance(self.tube.inner_bc, receiver.ConvectiveBC):
          I.append(self.dof(i,j,k)) 
          J.append(self.dof(1,j,k))
          D.append(1.0)
          I.append(self.dof(i,j,k)) 
          J.append(self.dof(0,j,k))
          D.append(-1.0)
        else:
          raise ValueError("Unknown boundary condition!")

    return sp.coo_matrix((D,(I,J)), shape = (self.ndof, self.ndof))

  def _OD_BC(self):
    """
      Outer diameter contribution to the BC matrix
    """
    I = []
    J = []
    D = []
    i = self.nr-1
    for j in self.loop_t():
      for k in self.loop_z():
        if self.fix_edge:
          I.append(self.dof(i,j,k)) 
          J.append(self.dof(i,j,k))
          D.append(1.0)
        # Zero flux
        elif self.tube.outer_bc is None:
          I.append(self.dof(i,j,k)) 
          J.append(self.dof(self.nr-2,j,k))
          D.append(1.0)
          
          I.append(self.dof(i,j,k)) 
          J.append(self.dof(self.nr-1,j,k))
          D.append(-1.0)
        # Fixed temperature
        elif isinstance(self.tube.outer_bc, receiver.FixedTempBC):
          I.append(self.dof(i,j,k)) 
          J.append(self.dof(self.nr-2,j,k))
          D.append(1.0)
        # Fixed flux
        elif isinstance(self.tube.outer_bc, receiver.HeatFluxBC):
          I.append(self.dof(i,j,k)) 
          J.append(self.dof(self.nr-2,j,k))
          D.append(1.0)
          I.append(self.dof(i,j,k))
          J.append(self.dof(self.nr-1,j,k))
          D.append(-1.0)
        # Convection
        elif isinstance(self.tube.outer_bc, receiver.ConvectiveBC):
          I.append(self.dof(i,j,k)) 
          J.append(self.dof(self.nr-2,j,k))
          D.append(1.0)
          I.append(self.dof(i,j,k))
          J.append(self.dof(self.nr-1,j,k))
          D.append(-1.0)
        else:
          raise ValueError("Unknown boundary condition!")

    return sp.coo_matrix((D,(I,J)), shape = (self.ndof, self.ndof))

  def _left_BC(self):
    """
      Periodic contribution to the BC matrix
    """
    I = []
    J = []
    D = []
    j = 0
    for i in self.loop_r():
      for k in self.loop_z():
        I.append(self.dof(i,j,k))
        J.append(self.dof(i,j,k))
        D.append(1.0)

        I.append(self.dof(i,j,k))
        J.append(self.dof(i,self.nt-2,k))
        D.append(-1.0)

    return sp.coo_matrix((D,(I,J)), shape = (self.ndof, self.ndof))

  def _right_BC(self):
    """
      Periodic contribution to the BC matrix
    """
    I = []
    J = []
    D = []
    j = self.nt-1
    for i in self.loop_r():
      for k in self.loop_z():
        I.append(self.dof(i,j,k))
        J.append(self.dof(i,j,k))
        D.append(1.0)

        I.append(self.dof(i,j,k))
        J.append(self.dof(i,1,k))
        D.append(-1.0)

    return sp.coo_matrix((D,(I,J)), shape = (self.ndof, self.ndof))

  def _top_BC(self):
    """
      Axial top contribution to the BC matrix
    """
    I = []
    J = []
    D = []
    k = 0
    for i in self.loop_r():
      for j in self.loop_t():
        if self.fix_edge:
          I.append(self.dof(i,j,k))
          J.append(self.dof(i,j,k))
          D.append(1.0)
        else:
          I.append(self.dof(i,j,k))
          J.append(self.dof(i,j,1))
          D.append(1.0)
          I.append(self.dof(i,j,k))
          J.append(self.dof(i,j,0))
          D.append(-1.0)

    return sp.coo_matrix((D,(I,J)), shape = (self.ndof, self.ndof))

  def _bot_BC(self):
    """
      Axial bottom contribution to the BC matrix
    """
    I = []
    J = []
    D = []
    k = self.nz - 1
    for i in self.loop_r():
      for j in self.loop_t():
        if self.fix_edge:
          I.append(self.dof(i,j,k))
          J.append(self.dof(i,j,k))
          D.append(1.0)
        else:
          I.append(self.dof(i,j,k))
          J.append(self.dof(i,j,self.nz-2))
          D.append(1.0)
          I.append(self.dof(i,j,k))
          J.append(self.dof(i,j,self.nz-1))
          D.append(-1.0)

    return sp.coo_matrix((D,(I,J)), shape = (self.ndof, self.ndof))

  def _top_BC_R(self, T, time, T_n):
    """
      RHS contribution of the top axial BC

      Parameters:
        T           current temperatures
        time        current time
        T_n         previous temperatures
    """
    R = np.zeros((self.ndof,))
    k = 0
    for i in self.loop_r():
      for j in self.loop_t():
        if self.fix_edge:
          R[self.dof(i,j,k)] = self.fix_edge(time, 
              *[self.r[i,j,k], self.theta[i,j,k], 
                self.z[i,j,k]][:self.ndim])

    return R

  def _bot_BC_R(self, T, time, T_n):
    """
      RHS contribution of the bottom axial BC

      Parameters:
        T           current temperatures
        time        current time
        T_n         previous temperatures
    """
    R = np.zeros((self.ndof,))
    k = self.nz - 1
    for i in self.loop_r():
      for j in self.loop_t():
        if self.fix_edge:
          R[self.dof(i,j,k)] = self.fix_edge(time, 
              *[self.r[i,j,k], self.theta[i,j,k],
                self.z[i,j,k]][:self.ndim])

    return R

  def dof(self, i, j, k):
    """
      Return the DOF corresponding to the given grid position

      Parameters:
        i       r index
        j       theta index
        k       z index
    """
    return i * self.nt * self.nz + j * self.nz + k

  def loop_r(self):
    """
      Loop over non-ghost dofs
    """
    return range(1,self.nr-1)

  def loop_t(self):
    """
      Loop over non-ghost dofs
    """
    if self.ndim > 1:
      return range(1,self.nt-1)
    else:
      return [0]

  def loop_z(self):
    """
      Loop over non-ghost dofs
    """
    if self.ndim > 2:
      return range(1,self.nz-1)
    else:
      return [0]

  def full_loop_r(self):
    """
      Loop over all dofs
    """
    return range(0,self.nr)

  def full_loop_t(self):
    """
      Loop over all dofs
    """
    if self.ndim > 1:
      return range(0,self.nt)
    else:
      return [0]

  def full_loop_z(self):
    """
      Loop over all dofs
    """
    if self.ndim > 2:
      return range(0,self.nz)
    else:
      return [0]

  def dummy_loop_r(self):
    """
      Loop over ghost dofs
    """
    return [0,self.nr-1]

  def dummy_loop_t(self):
    """
      Loop over ghost dofs
    """
    return [0,self.nt-1]

  def dummy_loop_z(self):
    """
      Loop over ghost dofs
    """
    if self.ndim > 2:
      return [0,self.nz-1]
    else:
      return [0]

  def radial(self):
    """
      Insert the radial FD contribution into a sparse matrix
    """
    rh = (self.r[:-1] + self.r[1:]) / 2.0
    ah = (self.a[:-1] + self.a[1:]) / 2.0
    rhah = np.pad(rh * ah, ((1,1),(0,0),(0,0)), mode = 'edge')
   
    D1 = (self.act * rhah[:-1] / (self.r * self.dr**2.0)).flatten()[self.nt*self.nz:]
    D2 = -(self.act * (rhah[:-1] + rhah[1:]) / (self.r * self.dr**2.0)).flatten()
    D3 = (self.act * rhah[1:] / (self.r * self.dr**2.0)).flatten()[:-self.nt*self.nz]

    return sp.diags((D1,D2,D3), offsets = (-self.nt*self.nz,0,self.nt*self.nz), 
        shape = (self.ndof, self.ndof), format = 'coo')

  def circumfrential(self):
    """
      Insert the circumferential FD contribution into a coo matrix
    """
    ah = np.pad((self.a[:,:-1] + self.a[:,1:]) / 2.0,
        ((0,0),(1,1),(0,0)), mode = 'edge')

    D1 = (self.act * ah[:,:-1] / (self.r**2.0 * self.dt**2.0)).flatten()[self.nz:]
    D2 = -(self.act * (ah[:,:-1] + ah[:,1:]) / (self.r**2.0 * self.dt**2.0)).flatten()
    D3 = (self.act * ah[:,1:] / (self.r**2.0 * self.dt**2.0)).flatten()[:-self.nz]

    return sp.diags((D1,D2,D3), offsets = (-self.nz, 0, self.nz),
        shape = (self.ndof, self.ndof), format = 'coo')

  def axial(self):
    """
      Insert the axial FD contribution into a coo matrix
    """
    ah = np.pad((self.a[:,:,:-1] + self.a[:,:,1:]) / 2.0, 
        ((0,0),(0,0),(1,1)), mode = 'edge')

    D1 = (self.act * ah[:,:,:-1] / self.dz**2.0).flatten()[1:]
    D2 = (-self.act * (ah[:,:,:-1] + ah[:,:,1:]) / self.dz**2.0).flatten()
    D3 = (self.act * ah[:,:,1:] / self.dz**2.0).flatten()[:-1]

    return sp.diags([D1,D2,D3], offsets = (-1,0,1),
        shape = (self.ndof, self.ndof), format = 'coo')
