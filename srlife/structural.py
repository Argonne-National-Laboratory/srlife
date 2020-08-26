"""
  This module solves the receiver system or the single tube
  structural problem.
"""

from abc import ABC, abstractmethod
from skfem import mesh

import numpy as np

class TubeSolver(ABC):
  """
    This class takes as input:
      1) A Tube object with a defined temperature field and an internal
         pressure BC.  Both are optional, the thermal or pressure stress
         just won't be included in the analysis if it's not there.
      2) A material (as a neml material)
      3) A input top displacement (bottom assumed fixed)

    It must return:
      1) A dictionary of results in the right shape to add to the end
         of the Tube time results.
      2) The force on the top face
      3) The derivative of the force with respect to the displacement
  """
  @abstractmethod
  def solve(self, i, tube, material, disp):
    """
      Solve the structural tube problem for a single time step

      Parameters:
        i           time step to apply loads from
        tube        tube object with geometry and loading
        material    NEML model
        disp        top displacement
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
  
  nb = len(rs) * len(ts)
  bcoords = np.swapaxes(np.array([[ri * np.cos(ts), ri * np.sin(ts)] 
    for ri in rs]), 0, 1).reshape(2, nb)  
  
  coords = np.ascontiguousarray(
      np.array([np.vstack((bcoords, z*np.ones((1,nb)))) for z in zs]).swapaxes(0,1).reshape(3,
      nb * len(zs)))

  conn = []
  for k in range(tube.nz-1):
    for i in range(tube.nr-1):
      for j in range(tube.nt):
        conn.append(
            [
              k*tube.nr*tube.nt+i*tube.nt+j, 
              k*tube.nr*tube.nt+(i+1)*tube.nt+j,
              (k+1)*tube.nr*tube.nt+i*tube.nt+j,
              k*tube.nr*tube.nt+i*tube.nt+((j+1)%tube.nt), 
              (k+1)*tube.nr*tube.nt+(i+1)*tube.nt+j,
              k*tube.nr*tube.nt + (i+1)*tube.nt+((j+1)%tube.nt), 
              (k+1)*tube.nr*tube.nt+i*tube.nt+((j+1)%tube.nt), 
              (k+1)*tube.nr*tube.nt + (i+1)*tube.nt+((j+1)%tube.nt)
            ])

  conn = np.ascontiguousarray(np.array(conn, dtype = int).T)

  return mesh.MeshHex(coords, conn)
