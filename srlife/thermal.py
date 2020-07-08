"""
  This module defines 1D, 2D, and 3D thermal solvers.
"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.linalg as la
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
    return

class FiniteDifferenceImplicitThermalSolver(ThermalSolver):
  """
    Solves the heat transfer problem using the finite difference method.

    Solver handles the *cylindrical* 1D, 2D, or 3D cases.
  """
  def solve(self, tube, material, fluid, source = None, 
      T0 = None, fix_edge = None, atol = 1e-2, miter = 100,
      substep = 1):
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
        atol        iteration absolute tolerance
        miter       maximum iterations
        substep     divide user-provided time increments into smaller values
    """
    temperatures = FiniteDifferenceImplicitThermalProblem(tube, 
        material, fluid, source, T0, fix_edge, atol, miter, substep).solve()

    tube.add_results("temperature", temperatures)

class FiniteDifferenceImplicitThermalProblem:
  """
    The actual finite difference solver created to solve a single
    tube problem
  """
  def __init__(self, tube, material, fluid, source = None, T0 = None,
      fix_edge = None, atol = 1e-2, miter= 50, substep = 1):
    """
      Parameters:
        tube        Tube object to solve
        material    ThermalMaterial object
        fluid       FluidMaterial object

      Other Parameters:
        source      source function (t, r, ...)
        T0          initial condition function
        fix_edge:   an exact solution to fix edge BCs for testing
        atol        absolute tolerance
        miter       maximum iterations
        substep     divide user provided time increments into smaller steps
    """
    self.tube = tube
    self.material = material
    self.fluid = fluid

    self.atol = atol
    self.miter = miter

    self.substep = substep

    self.source_term = source
    self.T0 = T0
    self.fix_edge = fix_edge

    self.dr = self.tube.t / (self.tube.nr-1)
    self.dt = 2.0 * np.pi / (self.tube.nt)
    self.dz = self.tube.h / (self.tube.nz-1)

    self.dts = np.diff(self.tube.times)

    self.requires_iteration = (isinstance(self.tube.inner_bc, receiver.ConvectiveBC) or
        isinstance(self.tube.outer_bc, receiver.ConvectiveBC))
  
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
      raise ValueError("Thermal solver does not know how to handle"
      " abstraction %s" % self.tube.abstraction)

    # Useful for later
    self.mesh = self._generate_mesh()
    self.r = self.mesh[0]

    if self.ndim > 1:
      self.theta = self.mesh[1]
    else:
      self.theta = np.ones(self.r.shape) * self.tube.angle

    if self.ndim > 2:
      self.z = self.mesh[2]
    else:
      self.z = np.ones(self.r.shape) * self.tube.plane
    
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
    T = np.copy(T_n)
    t_n = time - dt
    dti = dt / self.substep

    for i in range(1, self.substep+1):
      t = t_n + dti * i

      if self.requires_iteration:
        T = self._solve_step_iter(T, t, dti)
      else:
        T = self._solve_step_noiter(T, t, dti)

    return T

  def _solve_step_iter(self, T_n, time, dt):
    """
      Solve a single step when iteration is required

      Parameters:
        T_n         previous temperatures
        time        current time
        dt          current dt
    """
    # Store reusable stuff
    self.setup_step(T_n)
    A = self._get_system(T_n, time, dt)

    def RJ(T):
      b, J = self._get_rhs(T, T_n, time, dt, deriv = True)
      return A.dot(T.flatten()) - b, A - J
    
    # Newton's method
    T = np.copy(T_n)    
    R, J = RJ(T)
    for i in range(self.miter):
      if la.norm(R) < self.atol:
        break
      T -= sla.spsolve(J, R).reshape(self.dim)
      R, J = RJ(T)
    else:
      raise Exception("Too many iterations (%i) in Newton loop!" % i)
    
    return T

  def _solve_step_noiter(self, T_n, time, dt):
    """
      Solve a single step when no iteration is required

      Parameters:
        T_n         previous temperatures
        time        current time
        dt          current dt
    """
    # Store reusable stuff
    self.setup_step(T_n)
    A = self._get_system(T_n, time, dt)
    b = self._get_rhs(T_n, T_n, time, dt)

    return sla.spsolve(A, b).reshape(self.dim)

  def _get_system(self, T_n, time, dt):
    """
      Helper method to get the linear system of equations

      Parameters:
        T_n     previous temperature
        time    current time
        dts     current dt
    """
    if self.ndim == 1:
      return self.form_1D_system(T_n, time, dt)
    elif self.ndim == 2:
      return self.form_2D_system(T_n, time, dt)
    elif self.ndim == 3:
      return self.form_3D_system(T_n, time, dt)
    else:
      raise ValueError("Unknown dimension %i!" % self.ndim)

  def _get_rhs(self, T, T_n, time, dt, deriv = False):
    """
      Helper method to get the RHS part of the system of equations

      Parameters:
        T       current temperature
        T_n     previous temperature
        time    current time
        dts     current dt

      Additional Parameters:
        deriv   also return the derivative of the RHS
    """
    if self.ndim == 1:
      return self.form_1D_RHS(T, T_n, time, dt, deriv = deriv)
    elif self.ndim == 2:
      return self.form_2D_RHS(T, T_n, time, dt, deriv = deriv)
    elif self.ndim == 3:
      return self.form_3D_RHS(T, T_n, time, dt, deriv = deriv)
    else:
      raise ValueError("Unknown dimension %i!" % self.ndim)

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
    
    # Form matrix and source
    # Radial contribution
    U, D, L = self.radial()
    upper[1:] = U
    main[1:-1] = D
    lower[:-1] = L

    # Impose r BCs
    D, U, _ = self.impose_left_radial_bc(T_n[1], T_n[1], time)
    main[:1] = D
    upper[:1] = U

    D, L, _ = self.impose_right_radial_bc(T_n[-2], T_n[-2], time)
    main[-1:] = D
    lower[-1:] = L

    # Apply the dt and 1.0+
    upper[1:] *= dt
    main[1:-1] *= dt
    main[1:-1] += 1
    lower[:-1] *= dt
    
    # Setup matrix
    A = sp.diags((upper,main,lower), (1,0,-1), shape = (self.ndof, self.ndof)).tocsr()

    return A

  def form_1D_RHS(self, T, T_n, time, dt, deriv = False):
    """
      Form the 1D RHS and, if requested, the derivative of the RHS

      Parameters:
        T       temperature
        T_n     previous temperature
        time    current time
        dt      current increment

      Optional parameters:
        deriv   also form the derivative of the RHS vector
    """
    RHS = np.zeros((self.ndof,))

    # Source
    RHS[1:-1] = -self.source(time)

    # Impose r BCs
    _, _, R = self.impose_left_radial_bc(T[1], T_n[1], time)
    RHS[:1] = R

    _, _, R = self.impose_right_radial_bc(T[-2], T_n[-2], time)
    RHS[-1:] = R

    b = RHS
    b[1:-1] *= dt
    b[1:-1] += T_n[1:-1].flatten()

    if deriv:
      upper = np.zeros((self.ndof-1,))
      lower = np.zeros((self.ndof-1,))
      
      L = self.left_radial_derivative(T[1], T_n[1], time)
      R = self.right_radial_derivative(T[-2], T_n[-2], time)

      upper[:1] = L
      lower[-1:] = R

      return b, sp.diags((upper,lower),(1,-1), shape = (self.ndof, self.ndof)).tocsr()

    return b

  # pylint: disable=too-many-locals
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
    U, D, L = self.radial()
    for i in range(self.nr-2):
      upper1[1+self.nt+i*self.nt:1+self.nt+(i+1)*self.nt-2] += U[
          i*(self.nt-2):(i+1)*(self.nt-2)] * dt
      main[1+self.nt+i*self.nt:1+self.nt+(i+1)*self.nt-2] += D[
          i*(self.nt-2):(i+1)*(self.nt-2)] * dt
      lower1[1+i*self.nt:1+(i+1)*self.nt-2] += L[
          i*(self.nt-2):(i+1)*(self.nt-2)] * dt

    # Circumferential contribution
    U, D, L = self.circumfrential()
    for i in range(self.nr-2):
      upper2[1+self.nt+i*self.nt:1+self.nt+(i+1)*self.nt-2] += U[
          i*(self.nt-2):(i+1)*(self.nt-2)] * dt
      main[1+self.nt+i*self.nt:1+self.nt+(i+1)*self.nt-2] += D[
          i*(self.nt-2):(i+1)*(self.nt-2)] * dt
      lower2[self.nt+i*self.nt:self.nt+(i+1)*self.nt-2] += L[
          i*(self.nt-2):(i+1)*(self.nt-2)] * dt
    
    # Radial left BC
    D, U, _ = self.impose_left_radial_bc(T_n[1], T_n[1], time)
    main[:self.nt] = D
    upper1[:self.nt] = U

    # Radial right BC
    D, L, _ = self.impose_right_radial_bc(T_n[-2], T_n[-2], time)
    main[-self.nt:] = D
    lower1[-self.nt:] = L

    # Theta left BC
    main[::self.nt] = 1.0
    upper3[::self.nt] = -1.0

    # Theta right BC
    main[self.nt-1::self.nt] = 1.0
    lower3[1::self.nt] = -1.0

    # Add 1 to the diagonal
    for i in range(self.nr-2):
      main[1+self.nt+i*self.nt:1+self.nt+(i+1)*self.nt-2] += 1.0

    # Setup matrix and RHS
    A = sp.diags((upper1,upper2,upper3,main,lower1,lower2,lower3), 
        (u1,u2,u3,0,l1,l2,l3), shape = (self.ndof, self.ndof)).tocsr()

    return A
 
  def form_2D_RHS(self, T, T_n, time, dt, deriv = False):
    """
      Form the 1D RHS

      Parameters:
        T       temperature
        T_n     previous temperature
        time    current time
        dt      current increment

      Additional parameters:
        deriv   also form the derivative of the RHS
    """
    RHS = np.zeros((self.ndof,))

    # Source and T_n contribution
    R = self.source(time)
    Tf = T_n.flatten()
    for i in range(self.nr-2):
      RHS[1+self.nt+i*self.nt:1+self.nt+(i+1)*self.nt-2] += (
          -R[i*(self.nt-2):(i+1)*(self.nt-2)] * dt + 
          Tf[i*(self.nt-2):(i+1)*(self.nt-2)])

    # Radial left BC
    _, _, R = self.impose_left_radial_bc(T[1], T_n[1], time)
    RHS[:self.nt] = R

    # Radial right BC
    _, _, R = self.impose_right_radial_bc(T[-2], T_n[-2], time)
    RHS[-self.nt:] = R

    # Theta left BC
    RHS[::self.nt] = 0

    # Theta right BC
    RHS[self.nt-1::self.nt] = 0

    if deriv:
      u1 = self.nt
      upper1 = np.zeros((self.ndof - u1,))
      l1 = -self.nt
      lower1 = np.zeros((self.ndof + l1,))
      
      L = self.left_radial_derivative(T[1], T_n[1], time)
      R = self.right_radial_derivative(T[-2], T_n[-2], time)

      upper1[:self.nt] = L
      lower1[-self.nt:] = R

      return RHS, sp.diags((upper1,lower1),(u1,l1), shape = (self.ndof, self.ndof)).tocsr()

    return RHS

  # pylint: disable=too-many-statements
  def form_3D_system(self, T_n, time, dt):
    """
      Form the 3D implicit system of equations

      Parameters:
        T_n:        previous temperature
        time:       current time
        dt:         time increment
      
      Returns:
        A matrix and b vector
    """
    # Matrix entries
    main = np.zeros((self.ndof,))
    u1 = self.nt * self.nz
    upper1 = np.zeros((self.ndof - u1,))
    l1 = -self.nt * self.nz
    lower1 = np.zeros((self.ndof + l1,))
    u2 = self.nz
    upper2 = np.zeros((self.ndof - u2,))
    l2 = -self.nz
    lower2 = np.zeros((self.ndof + l2,))
    u3 = 1
    upper3 = np.zeros((self.ndof - u3,))
    l3 = -1
    lower3 = np.zeros((self.ndof + l3,))
    u4 = self.nz * (self.nt - 2)
    upper4 = np.zeros((self.ndof - u4,))
    l4 = -self.nz * (self.nt - 2)
    lower4 = np.zeros((self.ndof + l4,))

    # Useful: number of entries to insert at a time
    n = self.nz - 2

    # Radial, axial, and circumferential contributions
    Ur, Dr, Lr = self.radial()
    Uc, Dc, Lc = self.circumfrential()
    Ua, Da, La = self.axial()

    # Stick into diagonals
    k = 0
    for i in range(1, self.nr-1):
      for j in range(1, self.nt-1):
        st = i*self.nt*self.nz + j*self.nz + 1
        main[st:st+n] = (1.0 + Dr[k:k+n] + Dc[k:k+n] + Da[k:k+n]) * dt
        upper1[st:st+n] = Ur[k:k+n] * dt
        lower1[st-self.nt*self.nz:st-self.nt*self.nz+n] = Lr[k:k+n] * dt
        upper2[st:st+n] = Uc[k:k+n] * dt
        lower2[st-self.nz:st-self.nz+n] = Lc[k:k+n] * dt
        upper3[st:st+n] = Ua[k:k+n] * dt
        lower3[st-1:st-1+n] = La[k:k+n] * dt

        k += n
   
    # BCs apply uniformly
    n = self.nz

    # Radial left BC
    D, U, _ = self.impose_left_radial_bc(T_n[1], T_n[1], time)
    main[:self.nt*self.nz] = D
    upper1[:self.nt*self.nz] = U

    # Radial right BC
    D, L, _ = self.impose_right_radial_bc(T_n[-2], T_n[-2], time)
    main[-self.nt*self.nz:] = D
    lower1[-self.nt*self.nz:] = L

    # Theta left BC
    k = 0
    for i in range(0,self.nr):
      st = i * self.nt * self.nz + 1
      main[st:st+n] = 1.0
      st = i *self.nt * self.nz
      upper4[st:st+n] = -1.0
      k += n

    # Theta right BC
    k = 0
    for i in range(0,self.nr):
      st = i*self.nt*self.nz + (self.nt-1)*self.nz + 1 
      main[st:st+n] = 1.0
      st = i * self.nt * self.nz + self.nz
      lower4[st:st+n] = -1.0
      k += n

    # Axial left BC
    D, U, _ = self.impose_left_axial_bc(time)
    main[::self.nz] = D
    upper3[::self.nz] = U

    # Axial right BC
    D, L, _ = self.impose_right_axial_bc(time)
    main[self.nz-1::self.nz] = D
    lower3[self.nz-2::self.nz] = L

    # Setup matrix and RHS
    A = sp.diags((upper1,upper2,upper3,upper4,main,lower1,lower2,lower3,lower4),
        (u1,u2,u3,u4,0,l1,l2,l3,l4), 
        shape = (self.ndof, self.ndof)).tocsr()
    
    return A

  def form_3D_RHS(self, T, T_n, time, dt, deriv = False):
    """
      Form the 3D RHS

      Parameters:
        T       temperature
        T_n     previous temperature
        time    current time
        dt      current increment

      Additional parameters:
        deriv   also form the derivative of the RHS
    """
    # Matrix entries
    RHS = np.zeros((self.ndof,))

    # Useful: number of entries to insert at a time
    n = self.nz - 2

    # Source and T_n contributions
    R = self.source(time)
    Tf = T_n.flatten()
    
    # Stick into diagonals
    k = 0
    for i in range(1, self.nr-1):
      for j in range(1, self.nt-1):
        st = i*self.nt*self.nz + j*self.nz + 1
        RHS[st:st+n] += (-R[k:k+n] * dt + Tf[k:k+n])
        k += n
   
    # BCs apply uniformly
    n = self.nz

    # Radial left BC
    _, _, R = self.impose_left_radial_bc(T[1], T_n[1], time)
    RHS[:self.nt*self.nz] = R

    # Radial right BC
    _, _, R = self.impose_right_radial_bc(T[-2], T_n[-2], time)
    RHS[-self.nt*self.nz:] = R

    # Theta left BC
    k = 0
    for i in range(0,self.nr):
      st = i * self.nt * self.nz + 1
      RHS[st:st+n] = 0
      st = i *self.nt * self.nz
      k += n

    # Theta right BC
    k = 0
    for i in range(0,self.nr):
      st = i*self.nt*self.nz + (self.nt-1)*self.nz + 1 
      RHS[st:st+n] = 0
      st = i * self.nt * self.nz + self.nz
      k += n

    # Axial left BC
    _, _, R = self.impose_left_axial_bc(time)
    RHS[::self.nz] = R

    # Axial right BC
    _, _, R = self.impose_right_axial_bc(time)
    RHS[self.nz-1::self.nz] = R

    if deriv:
      u1 = self.nt * self.nz
      upper1 = np.zeros((self.ndof - u1,))
      l1 = -self.nt * self.nz
      lower1 = np.zeros((self.ndof + l1,))
      
      L = self.left_radial_derivative(T[1], T_n[1], time)
      R = self.right_radial_derivative(T[-2], T_n[-2], time)

      upper1[:self.nt*self.nz] = L
      lower1[-self.nt*self.nz:] = R

      return RHS, sp.diags((upper1,lower1),(u1,l1), shape = (self.ndof, self.ndof)).tocsr()

    return RHS

  def setup_step(self, T):
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

    if self.ndim > 1:
      ae = np.pad(self.a, self._pad_values(1), mode = 'edge')
      self.a_t = 0.5*(ae[:,1:] + ae[:,:-1])
      re = np.pad(self.r, self._pad_values(1), mode = 'edge')
      self.r_t = 0.5*(re[:,1:] + re[:,:-1])

    if self.ndim > 2:
      ae = np.pad(self.a, self._pad_values(2), mode = 'edge')
      self.a_z = 0.5*(ae[:,:,1:] + ae[:,:,:-1])
      re = np.pad(self.r, self._pad_values(2), mode = 'edge')
      self.r_z = 0.5*(re[:,:,1:] + re[:,:,:-1])

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

  def radial(self):
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

  def impose_left_radial_bc(self, Tb, Tb_n, time):
    """
      Impose the radial BC on the left side of the domain

      Parameters:
        Tb:     temperatures on actual boundary
        Tb_n:   previous temperatures on the boundary
        time:   time

      Returns:
        Upper, diagonal, and RHS entries
    """
    # For testing we can impose a fixed solution on the edges
    if self.fix_edge:
      D = 1.0
      U = 0.0
      RHS = self.fix_edge(time, *self._edge_mesh(0)).flatten()
    # Zero flux
    elif self.tube.inner_bc is None:
      D = -1.0
      U = 1.0
      RHS = 0.0
    # Fixed temperature
    elif isinstance(self.tube.inner_bc, receiver.FixedTempBC):
      D = 0.0
      U = 1.0
      RHS = self.tube.inner_bc.temperature(time, self.theta[0], self.z[0]).flatten()
    # Fixed flux
    elif isinstance(self.tube.inner_bc, receiver.HeatFluxBC):
      D = 1.0
      U = -1.0
      RHS = (self.dr * self.tube.inner_bc.flux(time, self.theta[1], 
        self.z[1]) / self.k[1]).flatten()
    # Convection
    elif isinstance(self.tube.inner_bc, receiver.ConvectiveBC):
      D = 1.0
      U = -1.0
      RHS = -self.dr * (self.fluid.coefficient(self.material.name, Tb_n
        )*(Tb - self.tube.inner_bc.fluid_temperature(time, self.z[1])
          ) / self.k[1]).flatten()
    else:
      raise ValueError("Unknown boundary condition!")

    return D, U, RHS

  def left_radial_derivative(self, Tb, Tb_n, time):
    """
      The Jacobian contribution of the left radial derivative

      Parameters:
        Tb:     temperatures on the boundary
        Tb_n:   previous temperature on the boundary
        time:   time
    """
    if isinstance(self.tube.inner_bc, receiver.ConvectiveBC):
      return -self.dr * (self.fluid.coefficient(self.material.name, Tb_n
        ) / self.k[1]).flatten()
    else:
      return np.zeros(Tb.shape).flatten() 

  def impose_right_radial_bc(self, Tb, Tb_n, time):
    """
      Impose the radial BC on the right side of the domain

      Parameters:
        Tb:     temperatures on actual boundary
        Tb_n:   previous temperature on the boundary
        time:   time

      Returns:
        diagonal, lower, and RHS entries
    """
    if self.fix_edge:
      D = 1.0
      L = 0.0
      RHS = self.fix_edge(time, *self._edge_mesh(-1)).flatten()
    # Zero flux
    elif self.tube.outer_bc is None:
      D = -1.0
      L = 1.0
      RHS = 0.0
    # Fixed temperature
    elif isinstance(self.tube.outer_bc, receiver.FixedTempBC):
      D = 0.0
      L = 1.0
      RHS = self.tube.outer_bc.temperature(time, self.theta[-2], self.z[-2]).flatten()
    # Fixed flux
    elif isinstance(self.tube.outer_bc, receiver.HeatFluxBC):
      D = 1.0
      L = -1.0
      RHS = (self.dr * self.tube.outer_bc.flux(time, self.theta[-2], 
        self.z[-2]) / self.k[-2]).flatten()
    # Convection
    elif isinstance(self.tube.outer_bc, receiver.ConvectiveBC):
      D = 1.0
      L = -1.0
      RHS = -self.dr * (self.fluid.coefficient(self.material.name, Tb_n
        )*(Tb - self.tube.outer_bc.fluid_temperature(time, self.z[-2])
          ) / self.k[-2]).flatten()
    else:
      raise ValueError("Unknown boundary condition!")

    return D, L, RHS

  def right_radial_derivative(self, Tb, Tb_n, time):
    """
      The diagonal Jacobian contribution of the right radial derivative

      Parameters:
        Tb:     temperatures on the boundary
        Tb_n:   previous temperatures on the boundary
        time:   time
    """
    if isinstance(self.tube.outer_bc, receiver.ConvectiveBC):
      return -self.dr * (self.fluid.coefficient(self.material.name, Tb_n
        ) / self.k[-2]).flatten()
    else:
      return np.zeros(Tb.shape).flatten() 

  def _edge_mesh(self, ind):
    """
      Return the edge r mesh, just for evaluating BCs for testing
    """
    return tuple(self.mesh[i][ind] for i in range(self.ndim))

  def circumfrential(self):
    """
      The tridiagonal theta contributions to the problem

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

  def axial(self):
    """
      The tridiagonal z contributions to the problem

      Parameters:
        T:      full temperature field at which we want to calculate coefficients
        time:   current time

      Returns:
        upper, diagonal, and lower entries
    """
    p1 = self.a_z[:,:,1:]
    p2 = self.a_z[:,:,:-1]

    U = self.adofs(1.0 / self.dz**2.0 * p1).flatten()
    D = self.adofs(-1.0 / self.dz**2.0 * (p1 + p2)).flatten()
    L = self.adofs(1.0 / self.dz**2.0 * p2).flatten()

    return U, D, L

  def _edge_axial_mesh(self, ind):
    """
      Return the edge z mesh, just for evaluating BCs for testing
    """
    return tuple(self.mesh[i][:,:,ind] for i in range(self.ndim))

  def impose_left_axial_bc(self, time):
    """
      Impose the radial BC on the left side of the domain

      Parameters:
        time:   time

      Returns:
        Upper, diagonal, and RHS entries
    """
    # For testing we can impose a fixed solution on the edges
    if self.fix_edge:
      D = 1.0
      U = 0.0
      RHS = self.fix_edge(time, *self._edge_axial_mesh(0)).flatten()
    # Zero flux
    else:
      D = -1.0
      U = 1.0
      RHS = 0.0

    return D, U, RHS

  def impose_right_axial_bc(self, time):
    """
      Impose the radial BC on the right side of the domain

      Parameters:
        time:   time

      Returns:
        diagonal, lower, and RHS entries
    """
    if self.fix_edge:
      D = 1.0
      L = 0.0
      RHS = self.fix_edge(time, *self._edge_axial_mesh(-1)).flatten()
    # Zero flux
    else:
      D = -1.0
      L = 1.0
      RHS = 0.0

    return D, L, RHS

  def source(self, time):
    """
      RHS source term

      Parameters:
        time:   current time
        R:      RHS to add to

      Returns:
        source contribution
    """
    if self.source_term:
      return self.adofs(self.a / self.k * self.source_term(time, *self.mesh)).flatten()
    else:
      return self.adofs(np.zeros(self.a.shape)).flatten()

  def _generate_mesh(self):
    """
      Produce the r, theta, z mesh
    """
    rs = np.linspace(self.tube.r - self.tube.t - self.dr, self.tube.r + self.dr, self.tube.nr + 2)
    ts = np.linspace(0 - self.dt, 2.0*np.pi, self.tube.nt + 2)
    zs = np.linspace(0 - self.dz, self.tube.h + self.dz, self.tube.nz + 2)

    geom = [rs, ts, zs]

    return np.meshgrid(*geom[:self.ndim], indexing = 'ij', copy = True)
