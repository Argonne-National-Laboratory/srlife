"""
  Nonlinear solvers and other accessories
"""

import numpy as np
import numpy.linalg as la

def newton(RJ, x0, rel_tol = 1.0-6, abs_tol = 1.0e-8, miters = 20,
    linear_solver = la.solve, verbose = True, return_extra = False):
  """
    Simple newton-raphson solver

    Parameters:
      RJ            function that gives the residual and jacobian values
      x0            initial guess

    Additional parameters:
      reL_tol       relative convergence tolerance
      abs_tol       absolute convergence tolerance
      miters        maximum number of iterations
      linear_solver function that solves the linear system A x = b
      verbose       if true, print debug info,
      return_extra  if true also return the final residual vector and
                    Jacobian matrix
  """
  x = np.copy(x0)

  R, J = RJ(x)

  nR = la.norm(R)
  nR0 = nR

  if verbose:
    print("Iter\tnR\t\tnR/nR0")
    print("%i\t%3.2e" % (0, nR0))

  for i in range(miters):
    if nR < abs_tol or nR/nR0 < rel_tol:
      break
    x -= linear_solver(J, R)
    R, J = RJ(x)    
    nR = la.norm(R)
    if verbose:
      print("%i\t%3.2e\t%3.2e" % (i+1, nR, nR/nR0))
  else:
    raise RuntimeError("Nonlinear solver did not converge!")

  if verbose:
    print("")

  if return_extra:
    return x, R, J
  else:
    return x
