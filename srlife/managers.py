"""
  Solution managers actually walk a model through all the steps to solve
"""
import multiprocess
import numpy as np

import tqdm

from srlife import solverparams, thermal


def merge_dicts(idict):
    """
    Merge an iterator over dictionaries into a single dict
    """
    return {k: v for d in idict for k, v in d.items()}


class SolutionManager:
    """Solution manager

    High level solution manager walking through the thermal, structural,
    and damage calculations.

    Args:
      receiver (receiver.Receiver): receiver object to solve
      thermal_solver (thermal.ThermalSolver): how to solve the heat transport
      thermal_material (materials.ThermalMaterial): solid thermal properties
      fluid_material (materials.FluidMaterial): fluid thermal properties
      structural_solver (structural.TubeSolver):  how to solve the mechanics problem
      deformation_material (materials.DeformationMaterial):  how things deform with time
      damage_material (materials.StructuralMaterial):  how to calculate creep damage
      system_solver (system.SystemSolver):  how to tie tubes into the structural system
      damage_model (damage.DamageCalculator): how to calculate damage from the results
      pset (Optional[solverparams.ParameterSet]): optional set of solver parameters
    """

    def __init__(
        self,
        receiver,
        thermal_solver,
        thermal_material,
        fluid_material,
        structural_solver,
        deformation_material,
        damage_material,
        system_solver,
        damage_model,
        pset=solverparams.ParameterSet(),
    ):
        self.receiver = receiver
        self.thermal_solver = thermal_solver
        self.thermal_material = thermal_material
        self.fluid_material = fluid_material
        self.structural_solver = structural_solver
        self.deformation_material = deformation_material
        self.damage_material = damage_material
        self.system_solver = system_solver
        self.damage_model = damage_model

        self.pset = pset
        self.nthreads = pset.get_default("nthreads", 1)
        self.progress = pset.get_default("progress_bars", False)

        self.page = pset.get_default("page_results", False)
        self.receiver.set_paging(self.page)

        self.heuristics = []

    def add_heuristic(self, heuristic):
        """Add a heuristic to use during the solve

        Args:
          heuristics (list): new heuristic to add
        """
        self.heuristics.append(heuristic)

    @property
    def tubes(self):
        """Direct iterator over tubes

        Returns:
          iterator over tubes
        """
        return self.receiver.tubes

    @property
    def ntubes(self):
        """Pass through to get the number of tubes that need to be analyzed

        Returns:
          int:  total number of tubes in entire receiver
        """
        return len(list(self.tubes))

    def progress_decorator(self, base, ntotal):
        """Either wrap with a progress bar decorator or return a dummy

        Args:
          base (function): base function to wrap
          ntotal (int): total number in iterator, needed for wrapping iterators

        Returns:
          either function wrapped with decorator or base function
        """
        if self.progress:
            return tqdm.tqdm(base, total=ntotal)
        else:
            return base

    def solve_life(self):
        """User interface: solve everything and return receiver life

        The trigger for everything: solve the complete problem and report the
        best-estimate life.

        Returns:
          float:  Number of allowed daily cycles
        """
        self.solve_heat_transfer()
        self.solve_structural()

        return self.calculate_damage()

    def calculate_damage(self):
        """Calculate damage from the results

        Returns:
          float:    Number of allowed daily cycles
        """
        if self.progress:
            print("Calculating damage:")
        return self.damage_model.determine_life(
            self.receiver,
            self.damage_material,
            nthreads=self.nthreads,
            decorator=self.progress_decorator,
        )

    def solve_reliability(self):
        """User interface: solve everything and return receiver reliability

        The trigger for everything: solve the complete problem and report the
        best-estimate reliability.

        Returns:
          float:  Reliability between 0 and 1
        """
        self.solve_heat_transfer()
        self.solve_structural()

        return self.calculate_reliability()

    def calculate_reliability(self):
        """Calculate reliability from the results

        Returns:
          float:    Reliability between 0 and 1
        """
        if self.progress:
            print("Calculating reliability :")
        return self.damage_model.determine_reliability(
            self.receiver,
            self.damage_material,
            nthreads=self.nthreads,
            decorator=self.progress_decorator,
        )

    def solve_heat_transfer(self):
        """Solve heat transfer for the receiver

        Adds thermal results in each tube
        """
        if isinstance(self.thermal_solver, thermal.ThermohydraulicsThermalSolver):
            if self.progress:
                print("Running thermohydraulic analysis")
            self.thermal_solver.solve_receiver(
                self.receiver,
                self.thermal_material,
                self.fluid_material,
                decorator=self.progress_decorator,
                nthreads=self.nthreads,
                **merge_dicts(
                    h.args_for_thermohydraulic_solver(self.receiver)
                    for h in self.heuristics
                )
            )
        else:
            self.solve_heat_transfer_tube()

    def solve_heat_transfer_tube(self):
        """Solve the heat transfer problem for each tube

        Adds the thermal results to each receiver.Tube object
        """
        # pylint: disable=no-member
        if self.progress:
            print("Running thermal analysis:")
        with multiprocess.Pool(self.nthreads) as p:
            temps = list(
                self.progress_decorator(
                    p.imap(
                        lambda x: self.thermal_solver.solve(
                            x,
                            self.thermal_material,
                            self.fluid_material,
                            **merge_dicts(
                                h.args_for_thermal_solver(self.receiver, x)
                                for h in self.heuristics
                            )
                        ),
                        self.tubes,
                    ),
                    self.ntubes,
                )
            )
        for tube, temps in zip(self.tubes, temps):
            tube.add_results("temperature", temps)

    def solve_structural(self):
        """Solve the structural problem for the complete system

        Adds the structural results to each tube.
        """
        if self.progress:
            print("Running structural analysis:")
        self.system_solver.solve(
            self.receiver,
            self.deformation_material,
            self.structural_solver,
            nthreads=self.nthreads,
            decorator=self.progress_decorator,
            verbose=self.progress,
        )


class Heuristic:
    """Solution heuristic superclass

    Class that defines a heuristic to modify the basic 3D solve process in some
    way.

    To implement a specific heuristic override the appropriate pure virtual
    methods.
    """

    def args_for_tube_thermal_solver(self, receiver, tube):
        """Add tube solver args

        Args:
          receiver (receiver.receiver):       receiver object affected
          tube (receiver.tube):               tube object affected

        """
        return {}

    def args_for_thermohydraulic_solver(self, receiver):
        """Add tube solver args

        Args:
          receiver (receiver.receiver):       receiver object affected

        """
        return {}


class CycleResetHeuristic(Heuristic):
    """
    Reset the tube temperatures each cycle to the initial values
    """

    def args_for_tube_thermal_solver(self, receiver, tube):
        """Modify the tube solver before the solve

        Args:
          receiver (receiver.receiver):       receiver object affected
          tube (receiver.tube):               tube object affected
        """
        resetter = thermal.TemperatureResetter(
            lambda t: np.isclose(t % receiver.period, 0), tube.T0
        )

        return {"resetters": [resetter]}

    def args_for_thermohydraulic_solver(self, receiver):
        """Add tube solver args

        Args:
          receiver (receiver.receiver):       receiver object affected

        """
        resetter = thermal.ReceiverResetter(
            lambda t: np.isclose(t % receiver.period, 0)
        )

        return {"resetters": [resetter]}
