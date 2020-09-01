#!/usr/bin/env python3

import numpy as np

class TestCase:
  def __init__(self, inner, outer, T, analytic, ri = 0.9, ro = 1.0,
      alpha = 1.0e-5, E = 100000.0, nu = 0.3):
    self.p_inner = inner
    self.p_outer = outer
    self.Tfn = T
    self.afn = analytic

    self.ri = ri
    self.ro = ro

  def T(self, r):
    return self.Tfn(r, self.ri, self.ro)

  def exact(self, r):
    return self.afn(r, self.ri, self.ro, self.E, self.nu, self.alpha)

  def make_mat(self):
    pass

def exact1(r, ri, ro, E, nu, a):
  A = -ro**2.0/((ro/ri)**2.0+1.0)
  C = -1.0 / (2.0*(ro/ri)**2.0 + 1.0)

  return (1.0+nu)/E * (-A/r + 2.0*(1-2.0*v) * C* r)

cases = [
    TestCase(1,0,lambda T, ri, ro: 0.0, exact1)
    ]

if __name__ == "__main__":
  pass
