#!/usr/bin/env python3

import numpy as np

import sys

sys.path.append("../..")

from srlife import (
    receiver,
    solverparams,
    spring,
    structural,
    thermal,
    system,
    library,
    managers,
    damage,
)


def sample_parameters():
    params = solverparams.ParameterSet()

    params["nthreads"] = 2
    params["progress_bars"] = True
    # If true store results on disk (slower, but less memory)
    params["page_results"] = False

    params["thermal"]["miter"] = 200
    params["thermal"]["verbose"] = False

    params["thermal"]["solid"]["rtol"] = 1.0e-6
    params["thermal"]["solid"]["atol"] = 1.0e-4
    params["thermal"]["solid"]["miter"] = 20

    params["thermal"]["fluid"]["rtol"] = 1.0e-6
    params["thermal"]["fluid"]["atol"] = 1.0e-2
    params["thermal"]["fluid"]["miter"] = 50

    params["structural"]["rtol"] = 1.0e-6
    params["structural"]["atol"] = 1.0e-8
    params["structural"]["miter"] = 50
    params["structural"]["verbose"] = False

    params["system"]["rtol"] = 1.0e-6
    params["system"]["atol"] = 1.0e-8
    params["system"]["miter"] = 10
    params["system"]["verbose"] = False

    # How to extrapolate damage forward in time based on the cycles provided
    # Options:
    #     "lump" = D_future = sum(D_simulated) / N * days
    #     "last" = D_future = sum(D_simulated[:-1]) + D_simulated[-1] * days
    #     "poly" = polynomial extrapolation with order given by the "order" param
    params["damage"]["extrapolate"] = "lump"
    params["damage"]["order"] = 2

    return params


if __name__ == "__main__":
    # Load the receiver datastructure containing the:
    #     Receiver topology
    #     Tube geometry
    #     Thermal boundary conditions
    #     Pressure boundary conditions
    #     Interconnect stiffnesses
    model = receiver.Receiver.load("example-receiver.hdf5")

    # Demonstration on how to setup a flowpath
    # Setup the flow path information
    flowpath = []
    for name_panel, panel in model.panels.items():
        flowpath.append(name_panel)
        for name_tube, tube in panel.tubes.items():
            times = tube.times
            tube.multiplier_val = 16
            tube.T0 = 300 # k

    mass_flow = 460 # kg/s
    inlet_temp = 550 # C

    mass_flow *= 3600.0 # kg/hr
    mass_flow *= np.copy(tube.outer_bc.data[:, 0, 0])
    mass_flow /=np.copy(tube.outer_bc.data[:, 0, 0]).max()
    mass_flow[0] = mass_flow[1]
    mass_flow[-1] = mass_flow[-2]
    inlet_temp += 273.15 # K
    inlet_temp = inlet_temp * np.ones_like(times)

    model.add_flowpath(flowpath, times, mass_flow, inlet_temp)
    model.save("example-with-flowpath-variable_flow.hdf5")
    model = receiver.Receiver.load("example-with-flowpath-variable_flow.hdf5")

    # Load some customized solution parameters
    # These are all optional, all the solvers have default values
    # for parameters not provided by the user
    params = sample_parameters()

    # Define the thermal solver to use in solving the heat transfer problem
    thermal_solver = thermal.ThermohydraulicsThermalSolver(params["thermal"])
    # Define the structural solver to use in solving the individual tube problems
    structural_solver = structural.PythonTubeSolver(params["structural"])
    # Define the system solver to use in solving the coupled structural system
    system_solver = system.SpringSystemSolver(params["system"])
    # Damage model to use in calculating life
    damage_model = damage.TimeFractionInteractionDamage(params["damage"])

    # Load the materials
    fluid = library.load_thermal_fluid("32MgCl2-68KCl", "base")
    thermal, deformation, damage = library.load_material(
        "740H", "base", "elastic_model", "base"
    )

    # The solution manager
    solver = managers.SolutionManager(
        model,
        thermal_solver,
        thermal,
        fluid,
        structural_solver,
        deformation,
        damage,
        system_solver,
        damage_model,
        pset=params,
    )

    # Heuristics
    solver.add_heuristic(managers.CycleResetHeuristic())

    # Report the best-estimate life of the receiver
    life = solver.solve_life()

    print("Best estimate life: %f daily cycles" % life)

    model.save("example-with-results-variable_flow.hdf5")

    for pi, panel in model.panels.items():
        for ti, tube in panel.tubes.items():
            tube.write_vtk("variable_flow_tube-%s-%s" % (pi, ti))
