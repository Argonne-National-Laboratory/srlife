#!/usr/bin/env python3

import numpy as np
import numpy.linalg as la
import numpy.random as ra

def calculate_principal_stress(stress):
  """
    Calculate the principal stresses given the Mandel vector
  """
  tensor = np.zeros(stress.shape[:2] + (3,3))
  inds = [[(0,0)],[(1,1)],[(2,2)],[(1,2),(2,1)],[(0,2),(2,0)],[(0,1),(1,0)]]
  mults = [1.0, 1.0, 1.0, np.sqrt(2), np.sqrt(2), np.sqrt(2)]

  for i,(grp, m) in enumerate(zip(inds, mults)):
    for a,b in grp:
      tensor[...,a,b] = stress[...,i] / m

  return la.eigvalsh(tensor)

if __name__ == "__main__":
  shape = (3,2)
  values = np.array([[-1.0,-205.0,-56.0,-11.7,-0.6,-3000]])

  pstress1 = calculate_principal_stress(values)
  pstress2 = calculate_principal_stress(values)

  print(np.allclose(pstress1, pstress2))

  # For a known stress state
  values = np.zeros(shape + (6,))
  values[...,0] = -2226.78935086
  values[...,1] = -55.99586774
  values[...,2] = 2020.7852186

  pstress = calculate_principal_stress(values)

  print(np.allclose(pstress[...,0].flatten(), -2226.78935086))
  print(np.allclose(pstress[...,1].flatten(), -55.99586774))
  print(np.allclose(pstress[...,2].flatten(), 2020.7852186))
