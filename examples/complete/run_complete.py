#!/usr/bin/env python3

import sys
sys.path.append('../..')

from srlife import receiver, solverparams, spring, structural, thermal, system, library, managers

def sample_parameters():
  params = solverparams.ParameterSet()

  params["nthreads"] = 4
  params["progress_bars"] = True

  params["thermal"]["rtol"] = 1.0e-6
  params["thermal"]["atol"] = 1.0e-4
  params["thermal"]["miter"] = 20

  params["structural"]["rtol"] = 1.0e-6
  params["structural"]["rtol"] = 1.0e-4
  params["structural"]["miter"] = 20

  params["system"]["rtol"] = 1.0e-6
  params["system"]["atol"] = 1.0e-4
  params["system"]["miter"] = 10
  
  return params

if __name__ == "__main__":
  # Load the receiver datastructure containing the:
  #     Receiver topology
  #     Tube geometry
  #     Thermal boundary conditions
  #     Pressure boundary conditions
  #     Interconnect stiffnesses
  model = receiver.Receiver.load("example-receiver.hdf5")

  # Load some customized solution parameters
  # These are all optional, all the solvers have default values
  # for parameters not provided by the user
  params = sample_parameters()

  # Define the thermal solver to use in solving the heat transfer problem
  thermal_solver = thermal.FiniteDifferenceImplicitThermalSolver(
      params["thermal"])
  # Define the structural solver to use in solving the individual tube problems
  structural_solver = structural.PythonTubeSolver(params["structural"])
  # Define the system solver to use in solving the coupled structural system
  system_solver = system.SpringSystemSolver(params["system"])

  # Load the materials
  fluid = library.load_fluid("salt", "base")
  thermal, deformation, damage = library.load_material("740H", "base", 
      "elastic_model", "base")

  # The solution manager
  solver = managers.SolutionManager(model, thermal_solver, thermal, fluid,
      structural_solver, deformation, damage, system_solver, 
      pset = params)

  # Heuristics would go here

  # Report the best-estimate life of the receiver 
  life = solver.solve_life()

  print("Best estimate life: %f years" % life)
