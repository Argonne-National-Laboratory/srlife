"""
  This module defines 1D, 2D, and 3D thermal solvers.
"""

import multiprocess

import numpy as np
import numpy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from srlife import receiver, solverparams
from srlife.thermohydraulics import flowpath


class TemperatureResetter:
    """
    Reset the tube temperatures to a fixed value every so often.
    """

    def __init__(self, trigger, vals):
        """
        Setup the resetter

        Parameters:
          trigger     function which triggers the reset
          vals        values to reset to
        """
        self.trigger = trigger
        self.vals = vals

    def apply(self, time, step, temps):
        """
        Actually make the modification

        Parameters:
          time        current time
          step        current step
          temps       current temperatures
        """
        if self.trigger(time):
            temps[step] = self.vals


class ReceiverResetter:
    """
    Reset the tube temperatures to a fixed value every so often.
    This object works on the entire receiver at once
    """

    def __init__(self, trigger):
        """
        Setup the resetter

        Parameters:
          trigger     function which triggers the reset
        """
        self.trigger = trigger

    def apply(self, time, step, model):
        """
        Actually make the modification

        Parameters:
          time        current time
          step        current step
          temps       current temperatures
        """
        if self.trigger(time):
            for tube in model.tubes:
                tube.quadrature_results["ghost_temperature"][step] = tube.T0


class ThermohydraulicsThermalSolver:
    """
    Solve the heat transfer problem by iterating between a simple
    1D thermal hydraulics model and a FiniteDifferenceImplicitThermalSolver.
    """

    def __init__(self, pset=solverparams.ParameterSet()):
        self.rtol = pset.get_default("rtol", 1.0e-6)
        self.atol = pset.get_default("atol", 1.0e-3)
        self.miter = pset.get_default("miter", 1000)
        self.verbose = pset.get_default("verbose", False)
        self.eps = pset.get_default("epsilon", 1.0e-10)

        self.solid_params = pset.get_default("solid", solverparams.ParameterSet())
        self.thermo_params = pset.get_default("fluid", solverparams.ParameterSet())

    def solve_receiver(
        self,
        model,
        solid_material,
        fluid_material,
        decorator=None,
        nthreads=1,
        resetters=None,
    ):
        """Solve the entire receiver by splitting into independent flow paths

        Args:
            model (Receiver): fully-populated receiver object
            solid_material (ThermalMaterial): solid thermal properties
            fluid_material (ThermalFluidMaterial): fluid thermal properties
            decorator (Optional[decorator]): progress decorator
            nthreads(Optional[int]): number of allowable threads to use
            resetters(Optional[list]): resetting objects to apply
        """
        if resetters is None:
            resetters = []

        # Setup
        self.receiver = model
        self.solid_material = solid_material
        self.fluid_material = fluid_material
        self.decorator = decorator
        self.nthreads = nthreads

        # Stupid algorithm to decide on times
        solve_times = self._determine_solve_times()

        # Add the result fields we'll need, specifically nodal temperatures
        # and axial fluid temperatures
        for tube in self.receiver.tubes:
            tube.set_times(solve_times)
            tube.add_blank_axial_results("fluid_temperature")
            tube.add_blank_axial_results("fluid_velocity")
            tube.add_blank_quadrature_results(
                "ghost_temperature", (tube.ntime,) + tube_dim(tube)
            )

        # Check that we have enough information to solve the problem and
        # setup the initial conditions for metal and fluid temperature
        self._setup()

        # Loop over time and solve each time step
        def do_stuff(x):
            i = x[0] + 1
            time = x[1][0]
            dt = x[1][1]
            self.solve_step(i, time, dt)

            for resetter in resetters:
                resetter.apply(time, i, self.receiver)

        list(
            decorator(
                map(
                    do_stuff,
                    enumerate(zip(solve_times[1:], solve_times[1:] - solve_times[:-1])),
                ),
                len(solve_times) - 1,
            )
        )

        # Add the unghosted temperature field
        for tube in self.receiver.tubes:
            tube.add_results("temperature", self._unghost(tube))

    def _unghost(self, tube):
        """Remove the ghost nodes from tube "ghost_temperature" results"""
        T = tube.quadrature_results["ghost_temperature"]
        # Don't return ghost values
        if tube.abstraction == "3D":
            return T[:, 1:-1, 1:-1, 1:-1]
        elif tube.abstraction == "2D":
            return T[:, 1:-1, 1:-1]
        elif tube.abstraction == "1D":
            return T[:, 1:-1]
        else:
            raise ValueError("Unknown abstraction %s" % tube.abstraction)

    def solve_step(self, i, time, dt):
        """Solve timestep i at time time

        Args:
            i (int):        time step
            time (float):   actual time value
            dt (float):     time increment
        """
        # Setup initial guesses for the metal and tube temperatures (use last step!)
        for tube in self.receiver.tubes:
            tube.quadrature_results["ghost_temperature"][i] = tube.quadrature_results[
                "ghost_temperature"
            ][i - 1]
            tube.axial_results["fluid_temperature"][i] = tube.axial_results[
                "fluid_temperature"
            ][i - 1]
            tube.axial_results["fluid_velocity"][i] = tube.axial_results[
                "fluid_velocity"
            ][i - 1]

        # Iterate between solving the metal temperatures and solving the
        # fluid temperatures until both are fairly stationary
        if self.verbose:
            print("Solving timestep %i, discrete time %f" % (i, time))
        for j in range(self.miter):
            previous_temps = np.array(
                [
                    tube.quadrature_results["ghost_temperature"][i]
                    for panel in self.receiver.panels.values()
                    for tube in panel.tubes.values()
                ]
            )
            self.solve_metal(i, time, dt)
            next_temps = np.array(
                [
                    tube.quadrature_results["ghost_temperature"][i]
                    for panel in self.receiver.panels.values()
                    for tube in panel.tubes.values()
                ]
            )
            temp_diff = np.abs(next_temps - previous_temps)
            temp_max_diff = np.max(temp_diff)
            temp_max_rel_diff = np.max(temp_diff / (previous_temps + self.eps))
            if self.verbose:
                print(
                    "\tSolved metal temperatures for iteration %i, "
                    "absolute difference %e / relative difference %e"
                    % (j, temp_max_diff, temp_max_rel_diff)
                )

            previous_fluid_temps = np.array(
                [
                    tube.axial_results["fluid_temperature"][i]
                    for panel in self.receiver.panels.values()
                    for tube in panel.tubes.values()
                ]
            )
            self.solve_fluid(i, time, dt)
            next_fluid_temps = np.array(
                [
                    tube.axial_results["fluid_temperature"][i]
                    for panel in self.receiver.panels.values()
                    for tube in panel.tubes.values()
                ]
            )
            fluid_diff = np.abs(next_fluid_temps - previous_fluid_temps)
            fluid_max_diff = np.max(fluid_diff)
            fluid_max_rel_diff = np.max(fluid_diff / (previous_fluid_temps + self.eps))
            if self.verbose:
                print(
                    "\tSolved fluid temperatures for iteration %i, "
                    "absolute difference %e / relative difference %e"
                    % (j, fluid_max_diff, fluid_max_rel_diff)
                )

            if (fluid_max_diff < self.atol and temp_max_diff < self.atol) or (
                fluid_max_rel_diff < self.rtol and temp_max_rel_diff < self.rtol
            ):
                break
        else:
            raise RuntimeError(
                "Picard iteration in the thermohydraulics solver did not converge"
            )

    def solve_metal(self, i, time, dt):
        """Setup and solve for the metal temperatures in each tube"""
        # pylint: disable=no-member
        # Setup the appropriate convective BCs on the ID of each tube
        # This is the best use of threads as best I can tell
        for tube in self.receiver.tubes:
            tube.set_bc(
                receiver.FilmCoefficientConvectiveBC(
                    tube.r - tube.t,
                    tube.h,
                    tube.nz,
                    tube.axial_results["fluid_temperature"][i],
                    self.fluid_material.film_coefficient(
                        tube.axial_results["fluid_temperature"][i],
                        tube.axial_results["fluid_velocity"][i],
                        tube.r - tube.t,
                    ),
                ),
                "inner",
            )

        def work(tube):
            # Need to add params
            tube_solver = FiniteDifferenceImplicitThermalProblem(
                tube,
                self.solid_material,
                self.fluid_material,
                **deparametrize_finite_difference(self.solid_params)
            )
            # Solve...
            return tube_solver.solve_step_substep(
                tube.quadrature_results["ghost_temperature"][i - 1], time, dt
            )

        with multiprocess.Pool(self.nthreads) as p:
            data = list(p.map(work, self.receiver.tubes))

        for tube, res in zip(self.receiver.tubes, data):
            tube.quadrature_results["ghost_temperature"][i] = res

    def solve_fluid(self, i, time, dt):
        """Setup and solve for the fluid temperatures in each tube"""
        # Setup the thermohydraulic solver for each flow path
        for path in self.receiver.flowpaths.values():
            # Add parameters
            model = flowpath.FlowPath(
                path["times"],
                path["mass_flow"],
                path["inlet_temp"],
                **flowpath.deparameterize_flow_path(self.thermo_params)
            )
            for panel_name in path["panels"]:
                model.add_panel_from_object(
                    self.receiver.panels[panel_name], self.fluid_material
                )

            # Solve for the fluid temperatures
            nodal_temps = model.solve(time)

            # Update the stored fluid temperature and flow velocities
            flow_rates, tube_temperatures = model.recover_tube_results(
                nodal_temps, time
            )

            for panel_name, panel_rates, panel_temps in zip(
                path["panels"], flow_rates, tube_temperatures
            ):
                for k, tube in enumerate(
                    self.receiver.panels[panel_name].tubes.values()
                ):
                    tube.axial_results["fluid_temperature"][i] = panel_temps[k]
                    tube.axial_results["fluid_velocity"][i] = panel_rates[k]

    def _setup(self):
        """Setup for solve

        Sets initial conditions on the metal and fluid temperature and does some
        checking to make sure the problem is fully-defined
        """
        # Check that every panel is in exactly 1 flow path
        panels_in_path = []
        for path in self.receiver.flowpaths.values():
            panels_in_path.extend(path["panels"])
        pset = set(panels_in_path)
        if len(pset) != len(panels_in_path):
            raise ValueError("There are panels in more than one flow path!")
        if pset != set(self.receiver.panels.keys()):
            raise ValueError("At least one panel is not in a flow path!")

        # Setup each tube
        for path in self.receiver.flowpaths.values():
            for panel_name in path["panels"]:
                panel = self.receiver.panels[panel_name]
                for tube in panel.tubes.values():
                    tube.quadrature_results["ghost_temperature"][0] = tube.T0
                    tube.axial_results["fluid_temperature"][0] = path["inlet_temp"][0]
                    tube.axial_results["fluid_velocity"][
                        0
                    ] = 100000000.0  # Need to provide IC?

    def _determine_solve_times(self):
        """Determine which times to solve the temperatures for

        Dumb algorithm to determine times to solve at -- just look at a tube
        """
        return list(list(self.receiver.panels.values())[0].tubes.values())[0].times


def tube_dim(tube):
    """
    Figure out a tube's nodal field array size based on the discertization and
    abstraction
    """
    dim = (tube.nr + 2, tube.nt + 2, tube.nz + 2)
    if tube.abstraction == "1D":
        return dim[:1]
    elif tube.abstraction == "2D":
        return dim[:2]
    elif tube.abstraction == "3D":
        return dim
    else:
        raise ValueError("Unknown abstraction %s" % tube.abstraction)


def deparametrize_finite_difference(pset):
    """
    Convert a ParameterSet to kwargs with defaults
    """
    return {
        "rtol": pset.get_default("rtol", 1.0e-6),
        "atol": pset.get_default("atol", 1.0e-2),
        "miter": pset.get_default("miter", 100),
        "substep": pset.get_default("substep", 1),
        "verbose": pset.get_default("verbose", False),
        "steady": pset.get_default("steady", False),
    }


class FiniteDifferenceImplicitThermalSolver:
    """
    Solves the heat transfer problem using the finite difference method.

    Solver handles the *cylindrical* 1D, 2D, or 3D cases.
    """

    def __init__(
        self,
        pset=solverparams.ParameterSet(),
        rtol=1.0e-6,
        atol=1.0e-2,
        miter=100,
        substep=1,
        verbose=False,
        steady=False,
    ):
        """
        Setup the solver

        Additional parameters:
          pset        object with solver parameters
          rtol        iteration relative tolerance
          atol        iteration absolute tolerance
          miter       maximum iterations
          substep     divide user-provided time increments into smaller values
          verbose     print a lot of debug info
          steady      ignore thermal mass and use conduction only
        """
        self.rtol = pset.get_default("rtol", rtol)
        self.atol = pset.get_default("atol", atol)
        self.miter = pset.get_default("miter", miter)
        self.substep = pset.get_default("substep", substep)
        self.verbose = pset.get_default("verbose", verbose)
        self.steady = pset.get_default("steady", steady)

    def solve(
        self,
        tube,
        material,
        fluid,
        source=None,
        T0=None,
        fix_edge=None,
        rtol=1e-6,
        atol=1e-2,
        miter=100,
        substep=1,
        resetters=None,
    ):
        """
        Solve the thermal problem defined for a single tube

        Parameters:
          tube        the Tube object defining the geometry, loading,
                      and analysis abstraction
          material    the thermal material object describing the material
                      conductivity and diffusivity
          fluid       the fluid material object describing the convective
                      heat transfer coefficient

        Other Parameters:
          source      if present, the source term as a function of t and
                      then the coordinates
          T0          if present override the tube IC with a function of
                      the coordinates
          fix_edge    an exact solution to fix edge BCs for testing
          rtol        solver relative tolerance
          atol        solver absolute tolerance
          miter       maximum number of nonlinear iterations
          substep     subdivide thermal steps into smaller increments
          resetters   list of reset objects to apply
        """
        if resetters is None:
            resetters = []

        temperatures = FiniteDifferenceImplicitThermalProblem(
            tube,
            material,
            fluid,
            source,
            T0,
            fix_edge,
            self.rtol,
            self.atol,
            self.miter,
            self.substep,
            self.verbose,
            self.steady,
        ).solve(resetters)

        tube.add_results("temperature", temperatures)

        return temperatures


class FiniteDifferenceImplicitThermalProblem:
    """
    The actual finite difference solver created to solve a single
    tube problem
    """

    def __init__(
        self,
        tube,
        material,
        fluid,
        source=None,
        T0=None,
        fix_edge=None,
        rtol=1.0e-4,
        atol=1e-2,
        miter=50,
        substep=1,
        verbose=False,
        steady=False,
    ):
        """
        Parameters:
          tube        Tube object to solve
          material    ThermalMaterial object
          fluid       FluidMaterial object

        Other Parameters:
          source      source function (t, r, ...)
          T0          initial condition function
          fix_edge:   an exact solution to fix edge BCs for testing
          rtol        relative tolerance
          atol        absolute tolerance
          miter       maximum iterations
          substep     divide user provided time increments into smaller steps
          verbose     print a lot of debug info
          steady      use steady state (conduction) theory
        """
        self.tube = tube
        self.material = material
        self.fluid = fluid

        self.rtol = rtol
        self.atol = atol
        self.miter = miter

        self.substep = substep

        self.verbose = verbose

        self.steady = steady

        self.source_term = source
        self.T0 = T0
        self.fix_edge = fix_edge

        self.dr = self.tube.t / (self.tube.nr - 1)
        self.dt = 2.0 * np.pi / (self.tube.nt)
        self.dz = self.tube.h / (self.tube.nz - 1)

        self.dts = np.diff(self.tube.times)

        # Ghost
        self.dim = (self.tube.nr + 2, self.tube.nt + 2, self.tube.nz + 2)

        if self.tube.abstraction == "3D":
            self.dim = self.dim
            self.nr, self.nt, self.nz = self.dim
            self.ndim = 3
            self.fdim = self.dim
        elif self.tube.abstraction == "2D":
            self.dim = self.dim[:2]
            self.nr, self.nt = self.dim
            self.nz = 1
            self.ndim = 2
            self.fdim = (self.nr, self.nt, 1)
        elif self.tube.abstraction == "1D":
            self.dim = self.dim[:1]
            (self.nr,) = self.dim
            self.nt = 1
            self.nz = 1
            self.ndim = 1
            self.fdim = (self.nr, 1, 1)
        else:
            raise ValueError(
                "Thermal solver does not know how to handle"
                " abstraction %s" % self.tube.abstraction
            )

        # Useful for later
        self.mesh = self._generate_mesh()
        self.r = self.mesh[0].reshape(self.fdim)

        if self.ndim > 1:
            self.theta = self.mesh[1].reshape(self.fdim)
        else:
            self.theta = np.ones(self.r.shape) * self.tube.angle

        if self.ndim > 2:
            self.z = self.mesh[2].reshape(self.fdim)
        else:
            self.z = np.ones(self.r.shape) * self.tube.plane

    def _generate_mesh(self):
        """
        Produce the r, theta, z mesh
        """
        rs = np.linspace(
            self.tube.r - self.tube.t - self.dr, self.tube.r + self.dr, self.tube.nr + 2
        )
        ts = np.linspace(-self.dt, 2.0 * np.pi, self.tube.nt + 2)
        zs = np.linspace(0 - self.dz, self.tube.h + self.dz, self.tube.nz + 2)

        geom = [rs, ts, zs]

        return np.meshgrid(*geom[: self.ndim], indexing="ij", copy=True)

    def solve(self, resetters=None):
        """
        Actually solve the problem...
        """
        if resetters is None:
            resetters = []

        # Setup the initial time
        T = np.zeros((self.tube.ntime,) + self.dim)
        if self.T0 is not None:
            T[0] = self.T0(*self.mesh)
        else:
            T[0] = self.tube.T0

        # Iterate through steps
        for i, (time, dt) in enumerate(zip(self.tube.times[1:], self.dts)):
            T[i + 1] = self.solve_step_substep(T[i], time, dt)
            for r in resetters:
                r.apply(time, i + 1, T)

        return self._real_results(T)

    def _real_results(self, T):
        """Helper that only returns the actual results, without the ghosts"""
        # Don't return ghost values
        if self.ndim == 3:
            return T[..., 1:-1, 1:-1, 1:-1]
        elif self.ndim == 2:
            return T[..., 1:-1, 1:-1]
        else:
            return T[..., 1:-1]

    def solve_step_substep(self, T_n, time, dt):
        """
        Do substepping, if requested by the user

        Parameters:
          T_n         previous full step temperature
          time        target time
          dt          target dt
        """
        T = np.copy(T_n)
        t_n = time - dt
        dti = dt / self.substep

        for i in range(1, self.substep + 1):
            t = t_n + dti * i
            T = self.solve_step(T, t, dti)

        return T

    def setup_step(self, T):
        """
        Setup common reusable arrays

        Parameters:
          T           temperature of interest
          time        current time
          dt          current dt
        """
        self.k = self.material.conductivity(T).reshape(self.fdim)
        self.a = self.material.diffusivity(T).reshape(self.fdim)

        if self.steady:
            self.c = self.k
            self.qc = 1.0
        else:
            self.c = self.a
            self.qc = self.a / self.k

        # Useful matrix that gives you the actual dofs
        self.act = np.pad(
            np.ones(tuple(d - 2 for d in self.dim)),
            [(1, 1)] * self.ndim,
            constant_values=[(0, 0)] * self.ndim,
        )

        self.act = self.act.reshape(self.fdim)

        self.ndof = T.size

    def _generate_A(self):
        """
        Generate the base differential operator for the current step
        """
        A = self.radial()

        if self.ndim > 1:
            A += self.circumfrential()

        if self.ndim > 2:
            A += self.axial()

        return A

    def _generate_id(self):
        """
        Return the non-boundary affecting ID matrix
        """
        return sp.diags(
            [self.act.flatten()],
            offsets=(0,),
            shape=(self.ndof, self.ndof),
            format="coo",
        )

    def _generate_source(self, time):
        """
        Generate the source part of the RHS

        Parameters:
          time        current time
        """
        if self.source_term:
            return (
                self.act
                * self.qc
                * self.source_term(time, *[self.r, self.theta, self.z][: self.ndim])
            ).flatten()
        else:
            return np.zeros((self.ndof,))

    def _generate_prev_temp(self, T_n):
        """
        The RHS vector with the temperature contributions in the correct locations

        Parmaeters:
          T_n         previous temperatures
        """
        return (T_n * self.act).flatten()

    def _generate_bc_matrix(self, T_n, time):
        """
        Generate the BC matrix terms (fixed)
        """
        M = self._ID_BC() + self._OD_BC()

        if self.ndim > 1:
            M += self._left_BC() + self._right_BC()

        if self.ndim > 2:
            M += self._top_BC() + self._bot_BC()

        return M

    def _generate_fixed_bc_RHS(self, T_n, time):
        """
        Generate the constant (i.e. axial) contributions to the RHS

        Parameters:

        T_n       previous temperature
        time      current time
        dt        time increment
        """
        if self.ndim > 2:
            return self._top_BC_R(T_n, time, T_n) + self._bot_BC_R(T_n, time, T_n)
        else:
            return np.zeros((self.ndof,))

    # pylint: disable=too-many-locals
    def solve_step(self, T_n, time, dt):
        """
        Actually setup and solve for a single load step

        Parameters:
          T_n         previous temperatures
          time        current time
          dt          time increment
        """
        # Generic setup
        self.setup_step(T_n)

        # Add dimensions, if necessary
        T_n = T_n.reshape(self.fdim)

        # FD contributions
        A = self._generate_A()
        # Identity
        ID = self._generate_id()
        # Source term
        S = self._generate_source(time)
        # Previous temp
        Tn = self._generate_prev_temp(T_n)

        # System without BCs
        if self.steady:
            M = -A
            R = S
        else:
            M = ID - A * dt
            R = S * dt + Tn

        # Matrix and fixed RHS boundary contributions
        B = self._generate_bc_matrix(T_n, time)
        M += B
        BRF = self._generate_fixed_bc_RHS(T_n, time)
        R += BRF

        # Dummy dofs
        D = self._generate_dummy_dofs()
        M += D

        # Covert over our fixed matrix
        M = M.tocsr()

        # This would be the iteration step
        T = np.copy(T_n)

        for i in range(self.miter):
            Ri = R + self._ID_BC_R(T, time, T_n) + self._OD_BC_R(T, time, T_n)
            J = self._d_ID_BC_R(T, time, T_n) + self._d_OD_BC_R(T, time, T_n)

            res = M.dot(T.flatten()) - Ri
            nr = la.norm(res)

            if i == 0:
                nr0 = nr

            if (nr < self.atol or nr / nr0 < self.rtol) and i > 0:
                break

            T -= sla.spsolve(M - J, res).reshape(self.fdim)
        else:
            raise RuntimeError("Too many iterations in newton solver!")

        return T.reshape(self.dim)

    # pylint: disable=too-many-branches
    def _generate_dummy_dofs(self):
        """
        Provide on-diagonal 1.0s for the dummy dofs
        """
        I = []
        J = []

        if self.ndim == 2:
            for i in self.dummy_loop_r():
                for j in self.dummy_loop_t():
                    for k in self.dummy_loop_z():
                        I.append(self.dof(i, j, k))
                        J.append(self.dof(i, j, k))
        elif self.ndim == 3:
            for i in self.dummy_loop_r():
                for j in self.dummy_loop_t():
                    for k in self.full_loop_z():
                        I.append(self.dof(i, j, k))
                        J.append(self.dof(i, j, k))
            for i in self.full_loop_r():
                for j in self.dummy_loop_t():
                    for k in self.dummy_loop_z():
                        I.append(self.dof(i, j, k))
                        J.append(self.dof(i, j, k))
            for i in self.dummy_loop_r():
                for j in self.full_loop_t():
                    for k in self.dummy_loop_z():
                        I.append(self.dof(i, j, k))
                        J.append(self.dof(i, j, k))

        return sp.coo_matrix((np.ones((len(I),)), (I, J)), shape=(self.ndof, self.ndof))

    def _ID_BC_R(self, T, time, T_n):
        """
        The inner diameter BC RHS contribution

        Parameters:
          T       current temperatures
          time    current time
          T_n     previous temperatures
        """
        R = np.zeros((self.ndof,))
        i = 0
        for j in self.loop_t():
            for k in self.loop_z():
                if self.fix_edge:
                    R[self.dof(i, j, k)] = self.fix_edge(
                        time,
                        *[self.r[i, j, k], self.theta[i, j, k], self.z[i, j, k]][
                            : self.ndim
                        ]
                    )
                # Zero flux
                elif self.tube.inner_bc is None:
                    R[self.dof(i, j, k)] = 0.0
                # Fixed temperature
                elif isinstance(self.tube.inner_bc, receiver.FixedTempBC):
                    R[self.dof(i, j, k)] = self.tube.inner_bc.temperature(
                        time, self.theta[1, j, k], self.z[1, j, k]
                    )
                # Fixed flux
                elif isinstance(self.tube.inner_bc, receiver.HeatFluxBC):
                    R[self.dof(i, j, k)] = (
                        -self.dr
                        * self.tube.inner_bc.flux(
                            time, self.theta[1, j, k], self.z[1, j, k]
                        )
                        / self.k[1, j, k]
                    )
                # Convection
                elif isinstance(self.tube.inner_bc, receiver.ConvectiveBC):
                    fluid_T = self.tube.inner_bc.fluid_temperature(
                        time, self.z[1, j, k]
                    )[0]
                    R[self.dof(i, j, k)] = (
                        self.dr
                        * self.fluid.coefficient(self.material.name, fluid_T)
                        * (T[1, j, k] - fluid_T)
                        / self.k[1, j, k]
                    )
                # Convection giving the film coefficient and fluid temperature
                elif isinstance(
                    self.tube.inner_bc, receiver.FilmCoefficientConvectiveBC
                ):
                    fluid_T = self.tube.inner_bc.fluid_temperature(
                        time, self.z[1, j, k]
                    )
                    h = self.tube.inner_bc.film_coefficient(time, self.z[1, j, k])
                    R[self.dof(i, j, k)] = (
                        self.dr * h * (T[1, j, k] - fluid_T) / self.k[1, j, k]
                    )
                else:
                    raise ValueError("Unknown boundary condition!")
        return R

    def _d_ID_BC_R(self, T, time, T_n):
        """
        Derivative of the inner diameter BC RHS contribution

        Parameters:
          T       current temperatures
          time    current time
          T_n     previous temperatures
        """
        I = []
        J = []
        D = []
        if isinstance(self.tube.inner_bc, receiver.ConvectiveBC):
            i = 0
            for j in self.loop_t():
                for k in self.loop_z():
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(1, j, k))
                    fluid_T = self.tube.inner_bc.fluid_temperature(
                        time, self.z[1, j, k]
                    )[0]
                    D.append(
                        self.dr
                        * self.fluid.coefficient(self.material.name, fluid_T)
                        / self.k[1, j, k]
                    )
        elif isinstance(self.tube.inner_bc, receiver.FilmCoefficientConvectiveBC):
            i = 0
            for j in self.loop_t():
                for k in self.loop_z():
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(1, j, k))
                    fluid_T = self.tube.inner_bc.fluid_temperature(
                        time, self.z[1, j, k]
                    )
                    h = self.tube.inner_bc.film_coefficient(time, self.z[1, j, k])
                    D.append(self.dr * h / self.k[1, j, k])

        return sp.coo_matrix((D, (I, J)), shape=(self.ndof, self.ndof))

    def _OD_BC_R(self, T, time, T_n):
        """
        The outer diameter BC RHS contribution

        Parameters:
          T       current temperatures
          time    current time
          T_n     previous temperatures
        """
        R = np.zeros((self.ndof,))
        i = self.nr - 1
        for j in self.loop_t():
            for k in self.loop_z():
                if self.fix_edge:
                    R[self.dof(i, j, k)] = self.fix_edge(
                        time,
                        *[self.r[i, j, k], self.theta[i, j, k], self.z[i, j, k]][
                            : self.ndim
                        ]
                    )
                # Zero flux
                elif self.tube.outer_bc is None:
                    R[self.dof(i, j, k)] = 0.0
                # Fixed temperature
                elif isinstance(self.tube.outer_bc, receiver.FixedTempBC):
                    R[self.dof(i, j, k)] = self.tube.outer_bc.temperature(
                        time, self.theta[self.nr - 2, j, k], self.z[self.nr - 2, j, k]
                    )
                # Fixed flux
                elif isinstance(self.tube.outer_bc, receiver.HeatFluxBC):
                    R[self.dof(i, j, k)] = (
                        -self.dr
                        * self.tube.outer_bc.flux(
                            time,
                            self.theta[self.nr - 2, j, k],
                            self.z[self.nr - 2, j, k],
                        )
                        / self.k[self.nr - 2, j, k]
                    )
                # Convection
                elif isinstance(self.tube.outer_bc, receiver.ConvectiveBC):
                    fluid_T = self.tube.outer_bc.fluid_temperature(
                        time, self.z[self.nr - 2, j, k]
                    )[0]
                    R[self.dof(i, j, k)] = (
                        self.dr
                        * self.fluid.coefficient(self.material.name, fluid_T)
                        * (T[self.nr - 2, j, k] - fluid_T)
                        / self.k[self.nr - 2, j, k]
                    )
                else:
                    raise ValueError("Unknown boundary condition!")
        return R

    def _d_OD_BC_R(self, T, time, T_n):
        """
        Derivative of the outer diameter BC RHS contribution

        Parameters:
          T       current temperatures
          time    current time
          T_n     previous temperatures
        """
        I = []
        J = []
        D = []
        if isinstance(self.tube.outer_bc, receiver.ConvectiveBC):
            i = self.nr - 1
            for j in self.loop_t():
                for k in self.loop_z():
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(self.nr - 2, j, k))
                    fluid_T = self.tube.outer_bc.fluid_temperature(
                        time, self.z[self.nr - 2, j, k]
                    )[0]
                    D.append(
                        self.dr
                        * self.fluid.coefficient(self.material.name, fluid_T)
                        / self.k[self.nr - 2, j, k]
                    )

        return sp.coo_matrix((D, (I, J)), shape=(self.ndof, self.ndof))

    def _ID_BC(self):
        """
        Inner diameter boundary condition contribution matrix
        """
        I = []
        J = []
        D = []
        i = 0
        for j in self.loop_t():
            for k in self.loop_z():
                if self.fix_edge:
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(i, j, k))
                    D.append(1.0)
                # Zero flux
                elif self.tube.inner_bc is None:
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(1, j, k))
                    D.append(1.0)
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(0, j, k))
                    D.append(-1.0)
                # Fixed temperature
                elif isinstance(self.tube.inner_bc, receiver.FixedTempBC):
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(1, j, k))
                    D.append(1.0)
                # Fixed flux
                elif isinstance(self.tube.inner_bc, receiver.HeatFluxBC):
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(1, j, k))
                    D.append(1.0)
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(0, j, k))
                    D.append(-1.0)
                # Convection
                elif isinstance(
                    self.tube.inner_bc,
                    (receiver.ConvectiveBC, receiver.FilmCoefficientConvectiveBC),
                ):
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(1, j, k))
                    D.append(1.0)
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(0, j, k))
                    D.append(-1.0)
                else:
                    raise ValueError("Unknown boundary condition!")

        return sp.coo_matrix((D, (I, J)), shape=(self.ndof, self.ndof))

    def _OD_BC(self):
        """
        Outer diameter contribution to the BC matrix
        """
        I = []
        J = []
        D = []
        i = self.nr - 1
        for j in self.loop_t():
            for k in self.loop_z():
                if self.fix_edge:
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(i, j, k))
                    D.append(1.0)
                # Zero flux
                elif self.tube.outer_bc is None:
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(self.nr - 2, j, k))
                    D.append(1.0)

                    I.append(self.dof(i, j, k))
                    J.append(self.dof(self.nr - 1, j, k))
                    D.append(-1.0)
                # Fixed temperature
                elif isinstance(self.tube.outer_bc, receiver.FixedTempBC):
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(self.nr - 2, j, k))
                    D.append(1.0)
                # Fixed flux
                elif isinstance(self.tube.outer_bc, receiver.HeatFluxBC):
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(self.nr - 2, j, k))
                    D.append(1.0)
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(self.nr - 1, j, k))
                    D.append(-1.0)
                # Convection
                elif isinstance(self.tube.outer_bc, receiver.ConvectiveBC):
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(self.nr - 2, j, k))
                    D.append(1.0)
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(self.nr - 1, j, k))
                    D.append(-1.0)
                else:
                    raise ValueError("Unknown boundary condition!")

        return sp.coo_matrix((D, (I, J)), shape=(self.ndof, self.ndof))

    def _left_BC(self):
        """
        Periodic contribution to the BC matrix
        """
        I = []
        J = []
        D = []
        j = 0
        for i in self.loop_r():
            for k in self.loop_z():
                I.append(self.dof(i, j, k))
                J.append(self.dof(i, j, k))
                D.append(1.0)

                I.append(self.dof(i, j, k))
                J.append(self.dof(i, self.nt - 2, k))
                D.append(-1.0)

        return sp.coo_matrix((D, (I, J)), shape=(self.ndof, self.ndof))

    def _right_BC(self):
        """
        Periodic contribution to the BC matrix
        """
        I = []
        J = []
        D = []
        j = self.nt - 1
        for i in self.loop_r():
            for k in self.loop_z():
                I.append(self.dof(i, j, k))
                J.append(self.dof(i, j, k))
                D.append(1.0)

                I.append(self.dof(i, j, k))
                J.append(self.dof(i, 1, k))
                D.append(-1.0)

        return sp.coo_matrix((D, (I, J)), shape=(self.ndof, self.ndof))

    def _top_BC(self):
        """
        Axial top contribution to the BC matrix
        """
        I = []
        J = []
        D = []
        k = 0
        for i in self.loop_r():
            for j in self.loop_t():
                if self.fix_edge:
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(i, j, k))
                    D.append(1.0)
                else:
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(i, j, 1))
                    D.append(1.0)
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(i, j, 0))
                    D.append(-1.0)

        return sp.coo_matrix((D, (I, J)), shape=(self.ndof, self.ndof))

    def _bot_BC(self):
        """
        Axial bottom contribution to the BC matrix
        """
        I = []
        J = []
        D = []
        k = self.nz - 1
        for i in self.loop_r():
            for j in self.loop_t():
                if self.fix_edge:
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(i, j, k))
                    D.append(1.0)
                else:
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(i, j, self.nz - 2))
                    D.append(1.0)
                    I.append(self.dof(i, j, k))
                    J.append(self.dof(i, j, self.nz - 1))
                    D.append(-1.0)

        return sp.coo_matrix((D, (I, J)), shape=(self.ndof, self.ndof))

    def _top_BC_R(self, T, time, T_n):
        """
        RHS contribution of the top axial BC

        Parameters:
          T           current temperatures
          time        current time
          T_n         previous temperatures
        """
        R = np.zeros((self.ndof,))
        k = 0
        for i in self.loop_r():
            for j in self.loop_t():
                if self.fix_edge:
                    R[self.dof(i, j, k)] = self.fix_edge(
                        time,
                        *[self.r[i, j, k], self.theta[i, j, k], self.z[i, j, k]][
                            : self.ndim
                        ]
                    )

        return R

    def _bot_BC_R(self, T, time, T_n):
        """
        RHS contribution of the bottom axial BC

        Parameters:
          T           current temperatures
          time        current time
          T_n         previous temperatures
        """
        R = np.zeros((self.ndof,))
        k = self.nz - 1
        for i in self.loop_r():
            for j in self.loop_t():
                if self.fix_edge:
                    R[self.dof(i, j, k)] = self.fix_edge(
                        time,
                        *[self.r[i, j, k], self.theta[i, j, k], self.z[i, j, k]][
                            : self.ndim
                        ]
                    )

        return R

    def dof(self, i, j, k):
        """
        Return the DOF corresponding to the given grid position

        Parameters:
          i       r index
          j       theta index
          k       z index
        """
        return i * self.nt * self.nz + j * self.nz + k

    def loop_r(self):
        """
        Loop over non-ghost dofs
        """
        return range(1, self.nr - 1)

    def loop_t(self):
        """
        Loop over non-ghost dofs
        """
        if self.ndim > 1:
            return range(1, self.nt - 1)
        else:
            return [0]

    def loop_z(self):
        """
        Loop over non-ghost dofs
        """
        if self.ndim > 2:
            return range(1, self.nz - 1)
        else:
            return [0]

    def full_loop_r(self):
        """
        Loop over all dofs
        """
        return range(0, self.nr)

    def full_loop_t(self):
        """
        Loop over all dofs
        """
        if self.ndim > 1:
            return range(0, self.nt)
        else:
            return [0]

    def full_loop_z(self):
        """
        Loop over all dofs
        """
        if self.ndim > 2:
            return range(0, self.nz)
        else:
            return [0]

    def dummy_loop_r(self):
        """
        Loop over ghost dofs
        """
        return [0, self.nr - 1]

    def dummy_loop_t(self):
        """
        Loop over ghost dofs
        """
        return [0, self.nt - 1]

    def dummy_loop_z(self):
        """
        Loop over ghost dofs
        """
        if self.ndim > 2:
            return [0, self.nz - 1]
        else:
            return [0]

    def radial(self):
        """
        Insert the radial FD contribution into a sparse matrix
        """
        rh = (self.r[:-1] + self.r[1:]) / 2.0
        ah = (self.c[:-1] + self.c[1:]) / 2.0
        rhah = np.pad(rh * ah, ((1, 1), (0, 0), (0, 0)), mode="edge")

        D1 = (self.act * rhah[:-1] / (self.r * self.dr**2.0)).flatten()[
            self.nt * self.nz :
        ]
        D2 = -(self.act * (rhah[:-1] + rhah[1:]) / (self.r * self.dr**2.0)).flatten()
        D3 = (self.act * rhah[1:] / (self.r * self.dr**2.0)).flatten()[
            : -self.nt * self.nz
        ]

        return sp.diags(
            (D1, D2, D3),
            offsets=(-self.nt * self.nz, 0, self.nt * self.nz),
            shape=(self.ndof, self.ndof),
            format="coo",
        )

    def circumfrential(self):
        """
        Insert the circumferential FD contribution into a coo matrix
        """
        ah = np.pad(
            (self.c[:, :-1] + self.c[:, 1:]) / 2.0,
            ((0, 0), (1, 1), (0, 0)),
            mode="edge",
        )

        D1 = (self.act * ah[:, :-1] / (self.r**2.0 * self.dt**2.0)).flatten()[
            self.nz :
        ]
        D2 = -(
            self.act * (ah[:, :-1] + ah[:, 1:]) / (self.r**2.0 * self.dt**2.0)
        ).flatten()
        D3 = (self.act * ah[:, 1:] / (self.r**2.0 * self.dt**2.0)).flatten()[
            : -self.nz
        ]

        return sp.diags(
            (D1, D2, D3),
            offsets=(-self.nz, 0, self.nz),
            shape=(self.ndof, self.ndof),
            format="coo",
        )

    def axial(self):
        """
        Insert the axial FD contribution into a coo matrix
        """
        ah = np.pad(
            (self.c[:, :, :-1] + self.c[:, :, 1:]) / 2.0,
            ((0, 0), (0, 0), (1, 1)),
            mode="edge",
        )

        D1 = (self.act * ah[:, :, :-1] / self.dz**2.0).flatten()[1:]
        D2 = (-self.act * (ah[:, :, :-1] + ah[:, :, 1:]) / self.dz**2.0).flatten()
        D3 = (self.act * ah[:, :, 1:] / self.dz**2.0).flatten()[:-1]

        return sp.diags(
            [D1, D2, D3], offsets=(-1, 0, 1), shape=(self.ndof, self.ndof), format="coo"
        )
