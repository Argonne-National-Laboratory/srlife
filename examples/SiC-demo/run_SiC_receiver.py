#!/usr/bin/env python3

import sys
sys.path.append('../..')

from srlife import receiver, solverparams, spring, structural, thermal, system, library, managers, damage
import numpy as np
import math
def sample_parameters():
  params = solverparams.ParameterSet()

  params["nthreads"] = 2
  params["progress_bars"] = True

  params["thermal"]["rtol"] = 1.0e-6
  params["thermal"]["atol"] = 1.0e-8
  params["thermal"]["miter"] = 20
  params["thermal"]["substep"] = 5

  params["structural"]["rtol"] = 1.0e-6
  params["structural"]["atol"] = 1.0e-8
  params["structural"]["miter"] = 50
  params["structural"]["verbose"] = False
  params["structural"]["qorder"] = 1
  params["structural"]["dof_tol"] = 1.0e-6

  params["system"]["rtol"] = 1.0e-6
  params["system"]["atol"] = 1.0e-8
  params["system"]["miter"] = 50
  params["system"]["verbose"] = False
  # If true store results on disk (slower, but less memory)
  params["page_results"] = True

  return params

if __name__ == "__main__":
  # Load the receiver datastructure containing the:
  #     Receiver topology
  #     Tube geometry
  #     Thermal boundary conditions
  #     Pressure boundary conditions
  #     Interconnect stiffnesses
  model = receiver.Receiver.load("chlorideSalt-SiC_receiver.hdf5")
  lengthlocs = 1000*np.array([16,5,16,5,16,5]) #max temp loc#np.array([16,5,16,5,16,5])
  # lengthlocs = 1000*np.array([5,16,16,16,16,16]) #max flux loc #np.array([5,16,16,16,16,16])
  # Cut down on run time for now
  for (panel,lengthloc) in zip(model.panels.values(),lengthlocs):   # uncomment for 1D analyses
    for tube in panel.tubes.values():                               # uncomment for 1D analyses
    #   tube.make_2D(lengthloc)                                       # uncomment for 2D analyses
       tube.make_1D(lengthloc, 0.0)                                  # uncomment for 1D analyses

  # Load some customized solution parameters
  # These are all optional, all the solvers have default values
  # for parameters not provided by the user
  params = sample_parameters()

  # Define the thermal solver to use in solving the heat transfer problem
  thermal_solver = thermal.FiniteDifferenceImplicitThermalSolver(params["thermal"])
  # Define the structural solver to use in solving the individual tube problems
  structural_solver = structural.PythonTubeSolver(params["structural"])
  # Define the system solver to use in solving the coupled structural system
  system_solver = system.SpringSystemSolver(params["system"])
  # Damage model to use in calculating life
  # damage_model = damage.PIAModel(params["damage"])
  # damage_model = damage.WNTSAModel(params["damage"])
  # damage_model = damage.MTSModelGriffithFlaw(params["damage"])
  # damage_model = damage.MTSModelPennyShapedFlaw(params["damage"])
  # damage_model = damage.CSEModelGriffithFlaw(params["damage"])
  # damage_model = damage.CSEModelPennyShapedFlaw(params["damage"])
  # damage_model = damage.SMMModelGriffithFlaw(params["damage"])
  damage_model = damage.SMMModelPennyShapedFlaw(params["damage"])

  # Load the materials
  fluid = library.load_fluid("salt_SiC", "base")
  thermal, deformation, damage = library.load_material("SiC", "base","elastic_model", "base")

  # The solution manager
  solver = managers.SolutionManager(model, thermal_solver, thermal, fluid,
      structural_solver, deformation, damage, system_solver, damage_model,
      pset = params)


  # Heuristics would go here

  # Calculate reliability
  reliability = solver.solve_life()

  print("Individual tube reliabilities:")
  print(reliability["tube_reliability"])

  print("Overall structure reliability:")
  print(reliability["overall_reliability"])

  for pi, panel in model.panels.items():
    for ti, tube in panel.tubes.items():
      tube.write_vtk("tube-%s-%s" % (pi, ti))
