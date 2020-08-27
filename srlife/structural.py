"""
  This module solves the receiver system or the single tube
  structural problem.
"""

from abc import ABC, abstractmethod
from skfem import *
from skfem import mesh, element, mapping
from skfem.helpers import dot as skdot

import scipy.sparse as sp
import scipy.sparse.linalg as spla

import copy

import numpy as np

class TubeSolver(ABC):
  """
    This class takes as input:
      1) A state object containing whatever the solver needs to execute the solve
         These objects must contain a method for dumping the required results fields into
         a Tube object
      2) A input net axial strain (bottom assumed fixed)

    It must return:
      1) A copied and updated state
      2) The force on the top face
      3) The derivative of the force with respect to the net strain
  """
  @abstractmethod
  def solve(self, tube, i, state_n, strain):
    """
      Solve the structural tube problem for a single time step

      Parameters:
        tube:       tube object with all bcs
        i:          time index to reference in tube results
        state:      state object
        strain:     top strain
    """
    pass

  @abstractmethod
  def init_state(self, tube, mat):
    """
      Initialize the solver state

      Parameters:
        tube:       tube object
        mat:        NEML material
    """
    pass

  @abstractmethod
  def dump_state(self, tube, i, state):
    """
      Update the required results fields in a tube object with the
      current state

      Parameters:
        tube:       tube to update
        i:          which time step this is
        state:      state object
    """
    pass

def mesh_tube(tube):
  """
    Make a simple, regular Cartesian mesh of a Tube using:
      1D:   linear Lagrange (line)
      2D:   bilinear Lagrange (Quad4)
      3D:   trilinear Lagrange (Hex8)
    elements
  """
  if tube.abstraction == "3D":
    return mesh3D(tube)
  elif tube.abstraction == "2D":
    return mesh2D(tube)
  elif tube.abstraction == "1D":
    return mesh1D(tube)
  else:
    raise ValueError("Unknown tube abstraction %s!" % tube.abstraction)

def mesh1D(tube):
  """
    Make a 1D linear Lagrange mesh of a tube
  """
  return mesh.MeshLine(tube.mesh[0].flatten())

def mesh2D(tube):
  """
    Make a 2D Cartesian Lagrange mesh of a tube
  """
  rs = np.linspace(tube.r-tube.t, tube.r, tube.nr)
  ts = np.linspace(0, 2*np.pi, tube.nt+1)[:tube.nt]

  coords = np.swapaxes(np.array([[ri * np.cos(ts), ri * np.sin(ts)] 
    for ri in rs]), 0, 1).reshape(2, len(rs)*len(ts))
  
  conn = []
  for i in range(tube.nr-1):
    for j in range(tube.nt):
      conn.append(
          [i*tube.nt+j, i*tube.nt+((j+1)%tube.nt), 
           (i+1)*tube.nt+((j+1)%tube.nt), (i+1)*tube.nt+j])

  conn = np.array(conn, dtype = int).T

  return mesh.MeshQuad(coords, conn)

def mesh3D(tube):
  """
    Make a 3D Cartesian Lagrange mesh of a tube
  """
  rs = np.linspace(tube.r-tube.t, tube.r, tube.nr)
  ts = np.linspace(0, 2*np.pi, tube.nt+1)[:tube.nt] 
  zs = np.linspace(0, tube.h, tube.nz)
  
  npr = tube.nt * tube.nz
  npt = tube.nz

  coords = np.ascontiguousarray(
      np.array([[r*np.cos(t), r*np.sin(t), z] for r in rs for t in ts for z in zs]).T)

  mapper = lambda r, c, h: r * npr + (c % tube.nt) * npt + h

  conn = []
  for i in range(tube.nr-1):
    for j in range(tube.nt):
      for k in range(tube.nz-1):
        conn.append(
            [
              mapper(i+1,j+1,k),
              mapper(i,j+1,k),
              mapper(i+1,j+1,k+1),
              mapper(i+1,j,k),
              mapper(i,j+1,k+1),
              mapper(i,j,k),
              mapper(i+1,j,k+1),
              mapper(i,j,k+1)
            ])

  conn = np.ascontiguousarray(np.array(conn, dtype = int).T)

  return mesh.MeshHex(coords, conn)

def setup_tube_structural_solve(tube):
  """
    Setup a tube for the structural solve by initializing all
    the output fields with zeros

    Parameters:
      tube      tube object
  """
  nt = tube.ntime
  
  suffixes = ['_xx', '_yy', '_zz', '_yz', '_xz', '_xy']
  fields = ['stress', 'strain', 'mechanical_strain', 'thermal_strain']

  for field in fields:
    for suffix in suffixes:
      tube.add_results(field+suffix, np.zeros((tube.ntime,) + tube.dim[:tube.ndim]))

class PythonTubeSolver(TubeSolver):
  """
    Tube solver class coded up with scikit.fem and the scipy sparse solvers
  """
  def __init__(self, rtol = 1.0e-6, atol = 1.0e-8, qorder = 1):
    """
      Setup the solver with common parameters

      Parameters:
        rtol        relative tolerance for NR iterations
        atol        absolute tolerance for NR iterations
        qorder      quadrature order
    """
    self.rtol = rtol
    self.atol = atol
    self.qorder = qorder

  def solve(self, tube, i, state_n, strain):
    """
      Solve the structural tube problem for a single time step

      Parameters:
        tube:       tube object with all bcs
        i:          time index to reference in tube results
        state:      state object
        strain:     top strain
    """
    state_np1 = state_n.copy()

    return state_np1

  def init_state(self, tube, mat):
    """
      Initialize the solver state

      Parameters:
        tube:       tube object
        mat:        NEML material
    """
    return PythonTubeSolver.State(tube, mat, self.qorder)

  def dump_state(self, tube, i, state):
    """
      Update the required results fields in a tube object with the
      current state

      Parameters:
        tube:       tube to update
        i:          which time step this is
        state:      state object
    """
    order = ['_xx', '_yy', '_zz', '_yz', '_xz', '_xy']
    fields = ['stress', 'strain', 'mechanical_strain', 'thermal_strain']
    data = [state.stress, state.strain, state.mechanical_strain, 
        state.thermal_strain]
    for k,(d,f,o) in enumerate(zip(data,fields,order)):
      tube.results[f+o][i] = self._fea2tube(tube, self._quad2res(state, 
        d[:,:,k]))

  def _tube2fea(self, tube, f):
    """
      Map a result field in the tube to the flat FEA vector
    """
    return f.flatten()

  def _fea2tube(self, tube, f):
    """
      Map a result field in the FEA to the right shape for the tube
    """
    return f.reshape(tube.dim[:tube.ndim])

  def _quad2res(self, state, f):
    """
      Quadrature results to a field result

      This does Laplacian smoothing
    """
    mass = BilinearForm(lambda u,v,w: u*v)
    force = LinearForm(lambda v,w: skdot(w['values'],v))
    Md = asm(mass, state.sbasis)
    fd = asm(force, state.sbasis, values = f)

    return spla.spsolve(Md,fd)

  def _res2quad(self, state, f):
    """
      Results field to the quadrature points
    """
    return state.sbasis.interpolate(f).value
  
  class State:
    """
      Subclass for maintaining state with the python solver
    """
    def __init__(self, tube, mat, qorder):
      """
        Initialize a full state object
      """
      self.material = mat
      self.mesh = mesh_tube(tube)

      betype = (element.ElementLineP1(), element.ElementQuad1(), 
          element.ElementHex1())[tube.ndim-1]
      if tube.ndim == 1:
        self.basis = InteriorBasis(mesh = self.mesh, 
            elem = betype, intorder = qorder)
        self.sbasis = self.basis
      else:
        mapping = MappingIsoparametric(self.mesh, betype)      
        etype = element.ElementVectorH1(betype)
        self.basis = InteriorBasis(mesh = self.mesh, elem = etype, 
            mapping = mapping, intorder = qorder)
        self.sbasis = InteriorBasis(mesh = self.mesh,
            elem = betype, intorder = qorder)
      
      # Now that is out of the way, setup the actual required storage
      self.stress = np.zeros((self.basis.nelems,self.nqi,6))
      self.strain = np.zeros((self.basis.nelems,self.nqi,6))
      self.mechanical_strain = np.zeros((self.basis.nelems,self.nqi,6))
      self.thermal_strain = np.zeros((self.basis.nelems,self.nqi,6))
      self.history = np.repeat(self.material.init_store()[:,np.newaxis], self.nq,
          axis = 1).reshape(self.basis.nelems, self.nqi, self.material.nstore)

    def copy(self):
      """
        Return a copy

        Soft copy the scikit-fem stuff
        Hard copy the results
      """
      new = copy.copy(self)
      new.stress = np.zeros(new.stress.shape)
      new.strain = np.zeros(new.strain.shape)
      new.mechanical_strain = np.zeros(new.mechanical_strain.shape)
      new.thermal_strain = np.zeros(new.thermal_strain.shape)
      new.history = np.zeros(new.history.shape)
      return new
    
    @property
    def nqi(self):
      return len(self.basis.quadrature[1])

    @property
    def nq(self):
      return self.nqi * self.basis.mesh.nelements
