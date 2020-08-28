from __future__ import division

import numpy as np
import itertools

mandel = ((0,0),(1,1),(2,2),(1,2),(0,2),(0,1))
mandel_mults = (1,1,1,np.sqrt(2),np.sqrt(2),np.sqrt(2))

def ms2ts(C):
  """
    Convert a Mandel notation stiffness matrix to a full stiffness tensor.
  """
  Ct = np.zeros((3,3,3,3))
  for a in range(6):
    for b in range(6):
      ind_a = itertools.permutations(mandel[a], r=2)
      ind_b = itertools.permutations(mandel[b], r=2)
      ma = mandel_mults[a]
      mb = mandel_mults[b]
      indexes = tuple(ai+bi for ai, bi in itertools.product(ind_a, ind_b))
      for ind in indexes:
        Ct[ind] = C[a,b] / (ma*mb)

  return Ct

def ts2ms(C):
  """
    Convert a stiffness tensor into a Mandel notation stiffness matrix
  """
  Cv = np.zeros((6,6))
  for i in range(6):
    for j in range(6):
      ma = mandel_mults[i]
      mb = mandel_mults[j]
      Cv[i,j] = C[mandel[i]+mandel[j]] * ma * mb

  return Cv

def sym(A):
  """
    Take a symmetric matrix to the Mandel convention vector.
  """
  return np.array([A[0,0], A[1,1], A[2,2], np.sqrt(2)*A[1,2], 
    np.sqrt(2)*A[0,2], np.sqrt(2)*A[0,1]])

def usym(v):
  """
    Take a Mandel symmetric vector to the full matrix.
  """
  return np.array([
    [v[0], v[5]/np.sqrt(2), v[4]/np.sqrt(2)],
    [v[5]/np.sqrt(2), v[1], v[3]/np.sqrt(2)],
    [v[4]/np.sqrt(2), v[3]/np.sqrt(2), v[2]]
    ])
