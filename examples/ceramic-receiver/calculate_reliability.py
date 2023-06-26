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
    params["page_results"] = True  # False

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

    params["damage"]["shear_sensitive"] = True

    # How to extrapolate damage forward in time based on the cycles provided
    # Options:
    #     "lump" = D_future = sum(D_simulated) / N * days
    #     "last" = D_future = sum(D_simulated[:-1]) + D_simulated[-1] * days
    #     "poly" = polynomial extrapolation with order given by the "order" param
    params["damage"]["extrapolate"] = "last"
    # params["damage"]["order"] = 2

    return params


if __name__ == "__main__":
    model = receiver.Receiver.load("SiC_1pt00mm_spath_Sresults.hdf")

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
    damage_models = [
        damage.PIAModel(params["damage"]),
        damage.WNTSAModel(params["damage"]),
        damage.MTSModelGriffithFlaw(params["damage"]),
        damage.MTSModelPennyShapedFlaw(params["damage"]),
        damage.CSEModelGriffithFlaw(params["damage"]),
        damage.CSEModelPennyShapedFlaw(params["damage"]),
        damage.SMMModelGriffithFlaw(params["damage"]),
        damage.SMMModelPennyShapedFlaw(params["damage"]),
    ]

    # Load the materials
    fluid = library.load_thermal_fluid("32MgCl2-68KCl", "base")
    thermal, deformation, damage = library.load_material(
        "SiC", "base", "cares", "cares"
    )

    reliability_filename = "SiC_1pt00mm_Reliability.txt"
    file = open(reliability_filename, "w")

    for damage_model in damage_models:
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

        # Report the best-estimate life of the receiver
        reliability = solver.calculate_reliability(time=100000.0)
        model.save("SiC_1pt00mm_spath_Rresults.hdf")

        for pi, panel in model.panels.items():
            for ti, tube in panel.tubes.items():
                tube.write_vtk("SiC_1mm_tube-%s-%s" % (pi, ti))

        print(damage_model)
        print("Individual tube reliabilities:")
        print(reliability["tube_reliability"])
        print("Individual panel reliabilities:")
        print(reliability["panel_reliability"])
        print("Overall reliability:")
        print(reliability["overall_reliability"])
        print("Minimum tube reliabilities:")
        print(min(reliability["tube_reliability"]))

        file.write("model = %s \n" % (damage_model))
        file.write(
            "minimum tube reliability = %f \n" % (min(reliability["tube_reliability"]))
        )
    file.close()
