"""
  Nonlinear solvers and other accessories
"""

import numpy as np
import numpy.linalg as la

# pylint: disable=too-many-branches
def newton(
    RJ,
    x0,
    rel_tol=1.0 - 6,
    abs_tol=1.0e-8,
    miters=20,
    linear_solver=la.solve,
    verbose=True,
    return_extra=False,
    linesearch=True,
    max_search=10,
):
    """Simple newton-raphson solver

    Args:
      RJ:                           function that gives the residual and
                                    jacobian values
      x0:                           initial guess
      rel_tol (Optional[1.0e-6]):   relative convergence tolerance
      abs_tol (Optional[1.0e-8]):   absolute convergence tolerance
      miters (Optional[20]):        maximum number of iterations
      linear_solver (Optional[numpy.linalg.solve]): function that solves
                                                    the linear system A x = b
      verbose (Optional[True]):     if true, print debug info,
      return_extra (Optional[False]):     if true also return the final residual vector and Jacobian
      linesearch (Optional[True]):  if true do backtracking linesearch
      max_search (Optional[10]):    max number of backtracking steps
    """
    x = np.copy(x0)

    R, J = RJ(x)

    nR = la.norm(R)
    nR0 = nR

    if verbose:
        if linesearch:
            print("Iter\tnR\t\tnR/nR0\t\talpha")
            print("%i\t%3.2e\t%3.2e" % (0, nR0, 1))
        else:
            print("Iter\tnR\t\tnR/nR0")
            print("%i\t%3.2e" % (0, nR0))

    for i in range(miters):
        if nR < abs_tol or nR / nR0 < rel_tol:
            break
        dx = linear_solver(J, R)
        if linesearch:
            alpha = 1.0
            nR_last = nR
            x_last = np.copy(x)
            for _ in range(max_search):
                x = x_last - alpha * dx
                R, J = RJ(x)
                nR = la.norm(R)
                if nR < nR_last:
                    break
                alpha /= 2.0
        else:
            x -= dx
            R, J = RJ(x)
            nR = la.norm(R)

        if verbose:
            if linesearch:
                print("%i\t%3.2e\t%3.2e\t%3.2e" % (i + 1, nR, nR / nR0, alpha))
            else:
                print("%i\t%3.2e\t%3.2e" % (i + 1, nR, nR / nR0))
    else:
        raise RuntimeError("Nonlinear solver did not converge!")

    if verbose:
        print("")

    if return_extra:
        return x, R, J
    else:
        return x
