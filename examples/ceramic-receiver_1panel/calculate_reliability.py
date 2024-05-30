#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import time

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

    params["nthreads"] = 1
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
    
    # Record the start time
    start_time = time.time()

    # model = receiver.Receiver.load("SiC_1pt00mm_spath_Sresults.hdf5")
    model = receiver.Receiver.load("Tube_st.hdf5")

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
        # damage.PIAModel(params["damage"]),
        # damage.WNTSAModel(params["damage"]),
        # damage.MTSModelGriffithFlaw(params["damage"]),
        # damage.MTSModelPennyShapedFlaw(params["damage"]),
        damage.CSEModelGriffithFlaw(params["damage"]),
        # damage.CSEModelPennyShapedFlaw(params["damage"]),
        # damage.CSEModelGriffithNotch(params["damage"]),
        # damage.SMMModelGriffithFlaw(params["damage"]),
        # damage.SMMModelGriffithNotch(params["damage"]),
        # damage.SMMModelPennyShapedFlaw(params["damage"]),
        # damage.SMMModelSemiCircularCrack(params["damage"]),
    ]

    # Load the materials
    fluid = library.load_thermal_fluid("32MgCl2-68KCl", "base")
    thermal, deformation, damage = library.load_material(
        "SiC", "base", "cares", "cares"
    )

    reliability_filename = "example_Reliability.txt"
    file = open(reliability_filename, "w")
    # with open("example_Reliability.txt", "a+") as external_file:

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
        reliability_volume = solver.calculate_reliability_volume_flaw(time=100.0)
        # model.save("SiC_1pt00mm_spath_Rresults.hdf5")

        # for pi, panel in model.panels.items():
        #     for ti, tube in panel.tubes.items():
        #         tube.write_vtk("SiC_1mm_tube-%s-%s" % (pi, ti))

        for pi, panel in model.panels.items():
            for ti, tube in panel.tubes.items():
                tube.write_vtk("variable_flow_tube-%s-%s" % (pi, ti))

        print(damage_model)
   
        # if reliability_volume["tube_reliability_volume_flaw"].all() != 1.:
        print("Individual tube reliabilities (volume):")
        print(reliability_volume["tube_reliability_volume_flaw"])
        # external_file.write("Individual tube reliabilities (volume):  %f" % ({reliability["tube_reliability_volume_flaw"]}))
        print("Individual panel reliabilities (volume):")
        print(reliability_volume["panel_reliability_volume_flaw"])

        print("Overall reliability (volume):")
        print(reliability_volume["overall_reliability_volume_flaw"])

        print("Minimum tube reliabilities (volume):")
        print(min(reliability_volume["tube_reliability_volume_flaw"]))

        reliability_surface = solver.calculate_reliability_surface_flaw(time=100.0)
        
        # if reliability_surface["tube_reliability_surface_flaw"].all() != 1.:
        print("Individual tube reliabilities (surface):")
        print(reliability_surface["tube_reliability_surface_flaw"])

        print("Individual panel reliabilities (surface):")
        print(reliability_surface["panel_reliability_surface_flaw"])

        print("Overall reliability (surface):")
        print(reliability_surface["overall_reliability_surface_flaw"])

        print("Minimum tube reliabilities (surface):")
        print(min(reliability_surface["tube_reliability_surface_flaw"]))

        reliability_combined = solver.calculate_reliability_combined(time=100.0)
        
        # if reliability_combined["tube_reliability_combined"].all() != 1.:
        print("Individual tube reliabilities (combined):")
        print(reliability_combined["tube_reliability_combined"])

        print("Individual panel reliabilities (combined):")
        print(reliability_combined["panel_reliability_combined"])

        print("Overall reliability (combined):")
        print(reliability_combined["overall_reliability_combined"])

        print("Minimum tube reliabilities (combined):")
        print(min(reliability_combined["tube_reliability_combined"]))

        model.save("example_with_Rresults_1panel.hdf5")
        
        # file.write("model = %s \n" % (damage_model))
        # file.write(
        #     # "Individual tube reliabilities (volume): "
        #     "minimum tube reliability (volume)= %f \n" % (min(reliability["tube_reliability_volume_flaw"]))
        # )      
            
        # Create bar plot of the reliabilities and save as .pdf
        # damage_model = "PIAModel"
        # Calculate the width for side-by-side bars
        bar_width = 0.4  # Adjust the width as needed
        bar_spacing = 0.1

        # # Tube
        # # Create a list of indices to be used as x-axis labels
        # indices = [x + 1 for x in range(len(reliability_volume["tube_reliability_volume_flaw"]))]
        # x = np.arange(len(indices))
        # # Create the bar plot
        # plt.figure()
        # if reliability_volume["tube_reliability_volume_flaw"].all() != 1.: plt.bar(x - bar_width/3, reliability_volume["tube_reliability_volume_flaw"],width=bar_width,label="Volume")
        # if reliability_surface["tube_reliability_surface_flaw"].all() != 1.: plt.bar(x, reliability_surface["tube_reliability_surface"],width=bar_width,label="Surface")
        # if reliability_combined["tube_reliability_combined"].all() != 1.: plt.bar(x + bar_width/3, reliability_combined["tube_reliability_combined"],width=bar_width,label="Total")
        # plt.xticks(x,indices,rotation=45, ha='right')
        # # Add labels and title
        # plt.xlabel("Tube numbers")
        # plt.ylabel("Reliability")
        # plt.legend(loc='lower right')
        # # plt.ylim(0.9,1.01)
        # plt.title("Tube reliablities")
        # # Save the plot
        # plt.savefig(f"tube_reliability_{damage_model}.png")
        # # plt.savefig("tube_reliability_volume_flaw.eps", format="eps")

        # # Panel
        # # Create the bar plot
        # plt.figure()
        # indices = [x + 1 for x in range(len(reliability_volume["panel_reliability_volume_flaw"]))]
        # x = np.arange(len(indices))
        # if reliability_volume["panel_reliability_volume_flaw"].all() != 1.: plt.bar(x - bar_width/3, reliability_volume["panel_reliability_volume_flaw"],width=bar_width,label="Volume")
        # if reliability_surface["panel_reliability_surface_flaw"].all() != 1.: plt.bar(x, reliability_surface["panel_reliability_surface_flaw"],width=bar_width,label="Surface")
        # if reliability_combined["panel_reliability_combined"].all() != 1.: plt.bar(x + bar_width/3, reliability_combined["panel_reliability_combined"],width=bar_width,label="Total")
        # # Add labels and title
        # plt.xticks(x,indices,rotation=45, ha='right')
        # plt.xlabel("Panel numbers")
        # plt.ylabel("Reliability")
        # plt.legend(loc='lower right')
        # # plt.ylim(0.9,1.01)
        # plt.title("Panel reliablities")
        # # Save the plot
        # plt.savefig(f"panel_reliability_{damage_model}.png")
        # # plt.savefig("panel_reliability_volume.eps", format="eps")

        # # Overall
        # # Create the bar plot
        # plt.figure()
        # if reliability_volume["overall_reliability_volume_flaw"].all() != 1.: plt.bar(-bar_width/3,reliability_volume["overall_reliability_volume_flaw"],width=bar_width,label="Volume")
        # if reliability_surface["overall_reliability_surface_flaw"].all() != 1.: plt.bar(0,reliability_surface["overall_reliability_surface_flaw"],width=bar_width,label="Surface")
        # if reliability_combined["overall_reliability_combined"].all() != 1.: plt.bar(bar_width/3,reliability_combined["overall_reliability_combined"],width=bar_width,label="Total")
        # # Add labels and title
        # # plt.xlabel("Tube numbers")
        # plt.ylabel("Reliability")
        # plt.legend(loc='lower right')
        # # plt.ylim(0.9,1.01)
        # plt.title("Overall reliablity")
        # # Save the plot
        # plt.savefig(f"overall_reliability_{damage_model}.png")
        # # plt.savefig("overall_reliability_volume.eps", format="eps")

        # # Show the plot
        # # plt.show()
    # # Record the end time
    # end_time = time.time()

    # # Calculate the runtime
    # runtime = end_time - start_time

    # print("Runtime:", runtime, "seconds")
    # file.close()