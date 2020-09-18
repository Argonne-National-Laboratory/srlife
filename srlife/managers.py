"""
  Solution managers actually walk a model through all the steps to solve
"""

import multiprocess
import itertools

from srlife import solverparams

class SolutionManager:
  """
    High level solution manager walking through the thermal, structural,
    and damage calculations
  """
  def __init__(self, receiver, thermal_solver, thermal_material,
      fluid_material, structural_solver, deformation_material,
      damage_material, system_solver, pset = solverparams.ParameterSet()):
    """
      Parameters:
        receiver                receiver object to solve
        thermal_solver          how to solve the heat transport
        thermal_material        solid thermal properties
        fluid_material          fluid thermal properties
        structural_solver       how to solve the mechanics problem
        deformation_material    how things deform with time
        damage_material         how to calculate creep damage
        system_solver           how to tie tubes into the structural system

      Additional Parameters:
        pset                    optional set of solver parameters
    """
    self.receiver = receiver
    self.thermal_solver = thermal_solver
    self.thermal_material = thermal_material
    self.fluid_material = fluid_material
    self.structural_solver = structural_solver
    self.deformation_material = deformation_material
    self.damage_material = damage_material
    self.system_solver = system_solver

    self.pset = pset
    self.nthreads = pset.get_default("nthreads", 1)
    self.progress = pset.get_default("progress_bars", False)

  @property
  def tubes(self):
    """
      Direct iterator over tubes
    """
    return itertools.chain(*(panel.tubes.values() 
      for panel in self.receiver.panels.values()))

  def solve_life(self):
    """
      The trigger for everything: solve the complete problem and report the
      best-estimate life
    """
    self.solve_heat_transfer()

    return 0

  def solve_heat_transfer(self):
    """
      Solve the heat transfer problem for each tube
    """
    with multiprocess.Pool(self.nthreads) as p:
      list(p.imap(lambda x: self.thermal_solver.solve(x, self.thermal_material, self.fluid_material),
        self.tubes))
