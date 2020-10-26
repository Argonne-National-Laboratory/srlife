"""
  Nonlinear solvers and other accessories
"""

import numpy as np
import numpy.linalg as la

def newton(RJ, x0, rel_tol = 1.0-6, abs_tol = 1.0e-8, miters = 20,
    linear_solver = la.solve, verbose = True, return_extra = False,
    backtrack = True, max_backtrack = 25):
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
      backtrack     turn on backtracking linesearch
      max_backtrack heuristic to stop backtracking if things go wrong
  """
  x = np.copy(x0)

  R, J = RJ(x)

  nR = la.norm(R)
  nR0 = nR

  if verbose:
    if backtrack:
      print("Iter\tnR\t\tnR/nR0\t\talpha")      
    else:
      print("Iter\tnR\t\tnR/nR0")
    print("%i\t%3.2e" % (0, nR0))

  for i in range(miters):
    if nR < abs_tol or nR/nR0 < rel_tol:
      break
    dx = linear_solver(J, R)
    if backtrack:
      nbt = 0
      prev = nR
      alpha = 1.0
      while True:
        xp = x - dx * alpha
        R, J = RJ(xp)
        nR = la.norm(R)
        if nR < prev:
          break
        else:
          alpha /= 2
          nbt += 1
          if nbt >= max_backtrack:
            raise RuntimeError("Backtracking could not improve the residual!")
      x = xp
    else:
      x -= dx
      R, J = RJ(x)    
      nR = la.norm(R)

    if verbose:
      if backtrack:
        print("%i\t%3.2e\t%3.2e\t%3.2e" % (i+1, nR, nR/nR0,alpha))
      else:
        print("%i\t%3.2e\t%3.2e" % (i+1, nR, nR/nR0))
  else:
    raise RuntimeError("Nonlinear solver did not converge!")

  if verbose:
    print("")

  if return_extra:
    return x, R, J
  else:
    return x
