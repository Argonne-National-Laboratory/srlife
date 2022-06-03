# pylint: disable=unused-wildcard-import, wildcard-import, too-many-lines

"""
  This module solves the receiver system or the single tube
  structural problem.
"""

from abc import ABC, abstractmethod

import copy

import scipy.sparse.linalg as spla
import numpy.linalg as la

import numpy as np

from skfem import *
from skfem import mesh, element
from skfem import helpers, utils
from skfem.element.discrete_field import DiscreteField

from neml import block

from srlife import solverparams


class TubeSolver(ABC):
    """
    This class takes as input:
      1) A state object containing whatever the solver needs to execute the solve
         These objects must contain a method for dumping the required results fields into
         a Tube object
      2) A input top displacement

    It must return:
      1) A copied and updated state
      2) The force on the top face
      3) The derivative of the force with respect to the displacement

      These three are grouped in a State object
    """

    @abstractmethod
    def solve(self, tube, i, state_n, dtop):
        """
        Solve the structural tube problem for a single time step

        Parameters:
          tube:       tube object with all bcs
          i:          time index to reference in tube results
          state:      state object
          dtop:       top displacement
        """
        return

    @abstractmethod
    def setup_tube(self, tube):
        """
        Setup all the quadrature results fields in the tube

        Parameters:
          tube:       tube object
        """
        return

    @abstractmethod
    def init_state(self, tube, mat):
        """
        Initialize the solver state

        Parameters:
          tube:       tube object
          mat:        NEML material
        """
        return

    @abstractmethod
    def dump_state(self, tube, i, state):
        """
        Update the required results fields in a tube object with the
        current state

        Parameters:
          tube:       tube to update
          i:          which time step this is
          state:      state object
        """
        return


def mesh_tube(tube):
    """
    Make a simple, regular Cartesian mesh of a Tube using:
      1D:   linear Lagrange (line)
      2D:   bilinear Lagrange (Quad4)
      3D:   trilinear Lagrange (Hex8)
    elements
    """
    if tube.abstraction == "3D":
        return mesh3D(tube)
    elif tube.abstraction == "2D":
        return mesh2D(tube)
    elif tube.abstraction == "1D":
        return mesh1D(tube)
    else:
        raise ValueError("Unknown tube abstraction %s!" % tube.abstraction)


def mesh1D(tube):
    """
    Make a 1D linear Lagrange mesh of a tube
    """
    return mesh.MeshLine(tube.mesh[0].flatten())


def mesh2D(tube):
    """
    Make a 2D Cartesian Lagrange mesh of a tube
    """
    rs = np.linspace(tube.r - tube.t, tube.r, tube.nr)
    ts = np.linspace(0, 2 * np.pi, tube.nt + 1)[: tube.nt]

    coords = np.swapaxes(
        np.array([[ri * np.cos(ts), ri * np.sin(ts)] for ri in rs]), 0, 1
    ).reshape(2, len(rs) * len(ts))

    conn = []
    for i in range(tube.nr - 1):
        for j in range(tube.nt):
            conn.append(
                [
                    i * tube.nt + j,
                    i * tube.nt + ((j + 1) % tube.nt),
                    (i + 1) * tube.nt + ((j + 1) % tube.nt),
                    (i + 1) * tube.nt + j,
                ]
            )

    conn = np.array(conn, dtype=int).T

    return mesh.MeshQuad(coords, conn)


def mesh3D(tube):
    """
    Make a 3D Cartesian Lagrange mesh of a tube
    """
    rs = np.linspace(tube.r - tube.t, tube.r, tube.nr)
    ts = np.linspace(0, 2 * np.pi, tube.nt + 1)[: tube.nt]
    zs = np.linspace(0, tube.h, tube.nz)

    npr = tube.nt * tube.nz
    npt = tube.nz

    coords = np.ascontiguousarray(
        np.array(
            [[r * np.cos(t), r * np.sin(t), z] for r in rs for t in ts for z in zs]
        ).T
    )

    mapper = lambda r, c, h: r * npr + (c % tube.nt) * npt + h

    conn = []
    for i in range(tube.nr - 1):
        for j in range(tube.nt):
            for k in range(tube.nz - 1):
                conn.append(
                    [
                        mapper(i + 1, j + 1, k),
                        mapper(i, j + 1, k),
                        mapper(i + 1, j + 1, k + 1),
                        mapper(i + 1, j, k),
                        mapper(i, j + 1, k + 1),
                        mapper(i, j, k),
                        mapper(i + 1, j, k + 1),
                        mapper(i, j, k + 1),
                    ]
                )

    conn = np.ascontiguousarray(np.array(conn, dtype=int).T)

    return mesh.MeshHex(coords, conn)


class PythonTubeSolver(TubeSolver):
    """
    Tube solver class coded up with scikit.fem and the scipy sparse solvers
    """

    def __init__(
        self,
        pset=solverparams.ParameterSet(),
        rtol=1.0e-6,
        atol=1.0e-8,
        qorder=1,
        dof_tol=1.0e-6,
        miter=10,
        verbose=False,
        max_divide=4,
        force_divide=False,
        max_linesearch=10,
    ):
        """
        Setup the solver with common parameters

        Additional Parameters:
          pset:            parameter set with solver parameters
          rtol:            relative tolerance for NR iterations
          atol:            absolute tolerance for NR iterations
          qorder:          quadrature order
          dof_to:l         geometric tolerance on finding boundary
                           degrees of freedom
          miter:           maximum newton-raphson iterations
          verbose:         verbose solve
          max_divide:      maximum adaptive integration subdivisions
          force_divide:    force adaptive substeps, for tests or
                           accuracy checks
          max_linesearch   maximum line search cutbacks
        """
        self.qorder = qorder
        self.rtol = pset.get_default("rtol", rtol)
        self.atol = pset.get_default("atol", atol)
        self.dof_tol = pset.get_default("dof_tol", dof_tol)
        self.miter = pset.get_default("miter", miter)
        self.verbose = pset.get_default("verbose", verbose)
        self.max_divide = pset.get_default("max_divide", max_divide)
        self.force_divide = pset.get_default("force_divide", force_divide)
        self.max_search = pset.get_default("max_linesearch", max_linesearch)
        self.solver_options = {
            "rtol": self.rtol,
            "atol": self.atol,
            "dof_tol": self.dof_tol,
            "miter": self.miter,
            "verbose": self.verbose,
            "max_linesearch": self.max_search,
        }

    def setup_tube(self, tube):
        """
        Add all the quadrature result fields we will need

        Parameters:
          tube        tube object
        """
        suffixes = ["_xx", "_yy", "_zz", "_yz", "_xz", "_xy"]
        fields = ["stress", "strain", "mechanical_strain", "thermal_strain"]

        for field in fields:
            for suffix in suffixes:
                tube.add_blank_quadrature_results(
                    field + suffix, (tube.ntime,) + self.qshape(tube)
                )

        tube.add_blank_quadrature_results(
            "temperature", (tube.ntime,) + self.qshape(tube)
        )

        suffixes = ["_x", "_y", "_z"]
        for i in range(tube.ndim):
            tube.add_blank_results(
                "disp" + suffixes[i], (tube.ntime,) + tube.dim[: tube.ndim]
            )

    def qshape(self, tube):
        """
        Number of elements x number of quadrature points
        """
        if tube.dim[2] == 1:
            nz = 1
        else:
            nz = tube.dim[2] - 1
        nelem = (tube.dim[0] - 1) * (tube.dim[1]) * nz
        return (nelem, (self.qorder + 1) ** tube.ndim)

    def _setup_state(self, sf, tube, i, state_n):
        """
        Setup all the information needed to solve the problem for
        some step fraction of a whole step

        Parameters:
          sf:         step fraction
          tube:       tube object with all the BC information
          i:          time index for state n+1
          state_n:    state at time n
        """
        state_next = state_n.copy()

        t = tube.times[i - 1] + (tube.times[i] - tube.times[i - 1]) * sf

        if "temperature" in tube.results:
            T_np1 = self._res2quad(
                state_next, self._tube2fea(tube, tube.results["temperature"][i])
            )
            T_n = self._res2quad(
                state_n, self._tube2fea(tube, tube.results["temperature"][i - 1])
            )
            T = T_n + (T_np1 - T_n) * sf
            state_next.temperature = T
        else:
            state_next.temperature = np.zeros(state_next.temperature.shape)

        if tube.pressure_bc:
            p = tube.pressure_bc.pressure(t)
        else:
            p = 0

        return state_next, p, t

    # pylint: disable=too-many-branches
    def solve(self, tube, i, state_n, dtop):
        """
        Solve the structural tube problem for a single time step
        using adaptive integration

        Parameters:
          tube:       tube object with all bcs
          i:          time index to reference in tube results
          state:      state object
          dtop:       top displacement
        """
        # Setup initial state_n
        if "temperature" in tube.results:
            state_n.temperature = self._res2quad(
                state_n, self._tube2fea(tube, tube.results["temperature"][i - 1])
            )
        else:
            state_n.temperature = np.zeros(state_n.temperature.shape)

        state_last = state_n.copy()

        t_last = tube.times[i - 1]
        if tube.pressure_bc:
            p_last = tube.pressure_bc.pressure(t_last)
        else:
            p_last = 0.0

        # Adaptively integrate
        tprog = 2**self.max_divide
        cprog = 0
        if self.force_divide:
            inc = 1
            mdiv = self.max_divide - 1
        else:
            inc = tprog
            mdiv = 0

        while cprog < tprog:
            sf = float(cprog + inc) / float(tprog)

            state_next, p_next, t_next = self._setup_state(sf, tube, i, state_n)

            try:
                if tube.ndim == 1:
                    solve_python_1d(
                        state_last,
                        t_last,
                        p_last,
                        state_next,
                        t_next,
                        p_next,
                        dtop * sf,
                        self.solver_options,
                    )
                elif tube.ndim == 2:
                    solve_python_2d(
                        state_last,
                        t_last,
                        p_last,
                        state_next,
                        t_next,
                        p_next,
                        dtop * sf,
                        self.solver_options,
                    )
                elif tube.ndim == 3:
                    solve_python_3d(
                        state_last,
                        t_last,
                        p_last,
                        state_next,
                        t_next,
                        p_next,
                        dtop * sf,
                        self.solver_options,
                    )
                else:
                    raise ValueError("Unknown dimension %i" % tube.ndim)
            except RuntimeError:
                inc /= 2
                mdiv += 1
                if mdiv >= self.max_divide:
                    break

            state_last = state_next
            t_last = t_next
            p_last = p_next
            cprog += inc

        if mdiv >= self.max_divide:
            raise RuntimeError("Adaptive integration failed")

        return state_next

    def init_state(self, tube, mat, i=None):
        """
        Initialize the solver state

        Parameters:
          tube:       tube object
          mat:        NEML material

        Additional Parameters:
          i:          if not none, also dump state into the tube
        """
        return PythonTubeSolver.State(tube, mat, self.qorder, i=i)

    def dump_state(self, tube, i, state):
        """
        Update the required results fields in a tube object with the
        current state

          Parameters:
          tube:       tube to update
          i:          which time step this is
          state:      state object
        """
        # Displacement
        suffixes = ["_x", "_y", "_z"]
        for k in range(state.ndim):
            tube.results["disp" + suffixes[k]][i] = self._fea2tube(
                tube, state.displacements[k :: state.ndim]
            )

        # The tensor fields
        order = ["_xx", "_yy", "_zz", "_yz", "_xz", "_xy"]
        inds = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        fields = ["stress", "strain", "mechanical_strain", "thermal_strain"]
        data = [
            state.stress,
            state.strain,
            state.mechanical_strain,
            state.thermal_strain,
        ]
        for f, d in zip(fields, data):
            for ind, o in zip(inds, order):
                tube.quadrature_results[f + o][i] = self._fea2tube_element(tube, d[ind])

        tube.quadrature_results["temperature"][i] = self._fea2tube_element(
            tube, state.temperature
        )

    def _fea2tube_element(self, tube, f):
        """
        Rearrange the elements (first index) of the field to match the vtk
        convention
        """
        return f

    def _tube2fea(self, tube, f):
        """
        Map a result field in the tube to the flat FEA vector
        """
        return f.flatten()

    def _fea2tube(self, tube, f):
        """
        Map a result field in the FEA to the right shape for the tube
        """
        return f.reshape(tube.dim[: tube.ndim])

    def _quad2res(self, state, f):
        """
        Quadrature results to a field result

        This does Laplacian smoothing
        """
        Md = asm(BilinearForm(lambda u, v, w: v * u), state.sbasis)
        fd = asm(
            LinearForm(lambda v, w: v * w["values"]),
            state.sbasis,
            values=DiscreteField(f),
        )

        return spla.spsolve(Md, fd)

    def _res2quad(self, state, f):
        """
        Results field to the quadrature points
        """
        return state.sbasis.interpolate(f).value

    class State:
        """
        Subclass for maintaining state with the python solver
        """

        def __init__(self, tube, mat, qorder, i=None):
            """
            Initialize a full state object
            """
            self.mat = mat
            self.mesh = mesh_tube(tube)
            self.qorder = qorder
            self.ndim = tube.ndim

            self.ri = tube.r - tube.t
            self.ro = tube.r
            self.h = tube.h

            # Define the pressure boundary
            self.define_boundary(tube)

            # Base element type
            self.betype = (
                element.ElementLineP1(),
                element.ElementQuad1(),
                element.ElementHex1(),
            )[tube.ndim - 1]

            # Define the scalar basis for interpolation
            self.define_scalar_basis()

            # Define the interior basis for momentum balance
            self.define_interior_basis()

            # Define the side basis for pressure
            self.define_exterior_basis()

            # Now that is out of the way, setup the actual required storage
            self.stress = np.zeros((3, 3, self.basis.nelems, self.nqi))
            self.strain = np.zeros((3, 3, self.basis.nelems, self.nqi))
            self.mechanical_strain = np.zeros((3, 3, self.basis.nelems, self.nqi))
            self.thermal_strain = np.zeros((3, 3, self.basis.nelems, self.nqi))
            self.history = np.repeat(
                self.material.init_store()[:, np.newaxis], self.nq, axis=1
            ).reshape(self.material.nstore, self.basis.nelems, self.nqi)
            self.tangent = np.zeros((3, 3, 3, 3, self.basis.nelems, self.nqi))
            self.temperature = np.zeros((self.basis.nelems, self.nqi))
            self.displacements = self.basis.zeros()
            self.time = 0.0

            self.energy = np.zeros((self.basis.nelems, self.nqi))
            self.dissipation = np.zeros((self.basis.nelems, self.nqi))

            self.force = 0.0
            self.stiffness = 0.0

            if i is not None:
                self.temperature = self.sbasis.interpolate(
                    tube.results["temperature"][i].flatten()
                ).value

        @property
        def material(self):
            """
            Getter for material object
            """
            return get_mat(self.mat)

        def define_boundary(self, tube, tol=0.5):
            """
            Define the pressure boundary

            Parameters:
              tube:     tube object for geometry

            Additional parameters:
              tol:      thickness tolerance for finding faces
            """
            atol = tol * tube.t
            if self.ndim == 1:
                self.mesh = self.mesh.with_boundaries(
                    {
                        "pressure": lambda x: np.logical_and(
                            x > tube.r - tube.t - atol, x < tube.r - tube.t + atol
                        )
                    }
                )
            else:
                self.mesh = self.mesh.with_boundaries(
                    {
                        "pressure": lambda x: np.logical_and(
                            np.sqrt(x[0] ** 2.0 + x[1] ** 2.0) > tube.r - tube.t - atol,
                            np.sqrt(x[0] ** 2.0 + x[1] ** 2.0) < tube.r - tube.t + atol,
                        )
                    }
                )

        def define_scalar_basis(self):
            """
            Define the scalar basis for mapping around result fields
            """
            self.sbasis = InteriorBasis(
                mesh=self.mesh, elem=self.betype, intorder=self.qorder
            )

        def define_interior_basis(self):
            """
            Define the interior basis for the balance equation
            """
            etype = element.ElementVectorH1(self.betype)
            self.basis = InteriorBasis(mesh=self.mesh, elem=etype, intorder=self.qorder)

        def define_exterior_basis(self):
            """
            Define the sideset basis for applying pressure
            """
            etype = element.ElementVectorH1(self.betype)
            self.pbasis = FacetBasis(
                mesh=self.mesh,
                elem=etype,
                intorder=self.qorder,
                facets=self.mesh.boundaries["pressure"],
            )

        def copy(self):
            """
            Return a copy

            Soft copy the scikit-fem stuff
            Hard copy the results
            """
            new = copy.copy(self)
            new.stress = np.copy(self.stress)
            new.strain = np.copy(self.strain)
            new.mechanical_strain = np.copy(self.mechanical_strain)
            new.thermal_strain = np.copy(self.thermal_strain)
            new.history = np.copy(self.history)
            new.tangent = np.copy(self.tangent)
            new.temperature = np.copy(self.temperature)
            new.displacements = np.copy(self.displacements)
            new.energy = np.copy(self.energy)
            new.dissipation = np.copy(self.dissipation)
            return new

        @property
        def nqi(self):
            """
            Integration points per element
            """
            return len(self.basis.quadrature[1])

        @property
        def nq(self):
            """
            Total number of integration points (size of array)
            """
            return self.nqi * self.basis.mesh.nelements

        @property
        def ne(self):
            """
            Number of elements
            """
            return self.basis.mesh.nelements


def solve_python_1d(
    state_n, t_n, p_n, state_np1, t_np1, p_np1, top_np1, solver_options
):
    """
    Solve an increment using the python 1d solver

    Parameters:
      state_n           previous state
      t_n               previous time
      p_n               previous pressure
      state_np1         next state
      t_np1             next time
      p_np1             next pressure
      top_np1           next axial displacement
      solver_options    various solver options
    """
    solver = PythonSolver(
        state_n, state_np1, solver_options, set_strain=top_np1 / state_np1.h
    )

    solver.solve(t_n, t_np1, p_np1)


def solve_python_2d(
    state_n, t_n, p_n, state_np1, t_np1, p_np1, top_np1, solver_options
):
    """
    Solve an increment using the python 2d solver

    Parameters:
      state_n           previous state
      t_n               previous time
      p_n               previous pressure
      state_np1         next state
      t_np1             next time
      p_np1             next pressure
      top_np1           next axial displacement
      solver_options    various solver options
    """
    solver = PythonSolver(
        state_n, state_np1, solver_options, set_strain=top_np1 / state_np1.h
    )

    # Fix the xy line
    node = state_np1.mesh.nodes_satisfying(
        lambda x: np.logical_and(
            np.logical_and(
                x[0] < state_np1.ro + solver_options["dof_tol"],
                x[0] > state_np1.ro - solver_options["dof_tol"],
            ),
            np.logical_and(
                x[1] < solver_options["dof_tol"], x[1] > -solver_options["dof_tol"]
            ),
        )
    )
    dofs_x = state_np1.basis.nodal_dofs[0, node].flatten()
    solver.add_dirichlet_bc(dofs_x, 0.0)
    dofs_y = state_np1.basis.nodal_dofs[1, node].flatten()
    solver.add_dirichlet_bc(dofs_y, 0.0)

    # Fix the y line
    node = state_np1.mesh.nodes_satisfying(
        lambda x: np.logical_and(
            np.logical_and(
                -x[0] < state_np1.ro + solver_options["dof_tol"],
                -x[0] > state_np1.ro - solver_options["dof_tol"],
            ),
            np.logical_and(
                x[1] < solver_options["dof_tol"], x[1] > -solver_options["dof_tol"]
            ),
        )
    )
    dofs_y = state_np1.basis.nodal_dofs[1, node].flatten()
    solver.add_dirichlet_bc(dofs_y, 0.0)

    solver.solve(t_n, t_np1, p_np1)


def solve_python_3d(
    state_n, t_n, p_n, state_np1, t_np1, p_np1, top_np1, solver_options
):
    """
    Solve an increment using the python 3d solver

    Parameters:
      state_n           previous state
      t_n               previous time
      p_n               previous pressure
      state_np1         next state
      t_np1             next time
      p_np1             next pressure
      top_np1           next axial displacement
      solver_options    various solver options
    """
    solver = PythonSolver(state_n, state_np1, solver_options)

    # Fix the xy line
    node = state_np1.mesh.nodes_satisfying(
        lambda x: np.logical_and(
            np.logical_and(
                np.logical_and(
                    x[0] < state_np1.ro + solver_options["dof_tol"],
                    x[0] > state_np1.ro - solver_options["dof_tol"],
                ),
                np.logical_and(
                    x[1] < solver_options["dof_tol"], x[1] > -solver_options["dof_tol"]
                ),
            ),
            x[2] < solver_options["dof_tol"],
        )
    )
    dofs_x = state_np1.basis.nodal_dofs[0, node].flatten()
    solver.add_dirichlet_bc(dofs_x, 0.0)
    dofs_y = state_np1.basis.nodal_dofs[1, node].flatten()
    solver.add_dirichlet_bc(dofs_y, 0.0)

    # Fix the y line
    node = state_np1.mesh.nodes_satisfying(
        lambda x: np.logical_and(
            np.logical_and(
                np.logical_and(
                    -x[0] < state_np1.ro + solver_options["dof_tol"],
                    -x[0] > state_np1.ro - solver_options["dof_tol"],
                ),
                np.logical_and(
                    x[1] < solver_options["dof_tol"], x[1] > -solver_options["dof_tol"]
                ),
            ),
            x[2] < solver_options["dof_tol"],
        )
    )
    dofs_y = state_np1.basis.nodal_dofs[1, node].flatten()
    solver.add_dirichlet_bc(dofs_y, 0.0)

    # Fix the bottom face
    node = state_np1.mesh.nodes_satisfying(lambda x: x[2] < solver_options["dof_tol"])
    dofs = state_np1.basis.nodal_dofs[2, node].flatten()
    solver.add_dirichlet_bc(dofs, 0.0)

    # Fix the top face
    node = state_np1.mesh.nodes_satisfying(
        lambda x: x[2] > state_np1.h - solver_options["dof_tol"]
    )
    dofs = state_np1.basis.nodal_dofs[2, node].flatten()
    solver.add_dirichlet_bc(dofs, top_np1)

    solver.solve(t_n, t_np1, p_np1)


class PythonSolver:
    """
    Actually manages the solve with scikit-fem
    """

    def __init__(self, state_n, state_np1, solver_options, set_strain=None):
        """
        Setup solver

        Parameters:
          state_n         previous state
          state_np1       current state
          solver_options  various 'how to solve' parameters

        Additional parameters:
          set_strain      set the zz strain to this value, if given
        """
        self.state_n = state_n
        self.state_np1 = state_np1
        self.options = solver_options
        self.set_strain = set_strain

        self.ebcs = []

        self.edofs = []
        self.evalues = []

        # Initialize the guess
        self.setup_guess()

        # The operators!
        if self.ndim == 1:
            # 1D axisymmetric is different than 2D/3D cartesian
            self.internal = LinearForm(
                lambda v, w: (
                    v.grad[0][0] * w["radial"]
                    - v.value[0] * w["radial"] / w.x[0]
                    + v.value[0] * w["hoop"] / w.x[0]
                )
            )
            self.jac = BilinearForm(
                lambda u, v, w: v.grad[0][0] * w["Crr"] * u.grad[0][0]
                + v.grad[0][0] * w["Crt"] * u.value[0] / w.x[0]
                - v.value[0]
                * (w["Crr"] * u.grad[0][0] + w["Crt"] * u.value[0] / w.x[0])
                / w.x[0]
                + v.value[0]
                * (w["Ctr"] * u.grad[0][0] + w["Ctt"] * u.value[0] / w.x[0])
                / w.x[0]
            )
        else:
            self.internal = LinearForm(
                lambda v, w: helpers.ddot(helpers.sym_grad(v), w["stress"])
            )
            self.jac = BilinearForm(
                lambda u, v, w: helpers.ddot(
                    helpers.sym_grad(v), helpers.ddot(w["C"], helpers.sym_grad(u))
                )
            )
        self.external = LinearForm(
            lambda v, w: helpers.dot((-1.0) * w["pressure"] * w.n, v)
        )

    @property
    def ndim(self):
        """
        Problem dimension
        """
        return self.state_np1.ndim

    def setup_guess(self):
        """
        Estimate the displacements
        """
        self.state_np1.displacements = self.state_n.displacements

    def solve(self, t_n, t_np1, p):
        """
        Do the actual solve using Newton-Raphson iteration

        Parameters:
          t_n     previous time
          t_np1   next time
          p       pressure
        """
        # Setup the time step
        self.state_n.time = t_n
        self.state_np1.time = t_np1

        # Assemble the dirichlet BCs into big vectors
        self.assemble_dirichlet()

        # Do the initial stress update
        self.update_state()

        # Calculate the initial residual and jacobian
        R = self.residual(p)
        nR0 = la.norm(R[self.kdofs])
        nR = nR0

        # Printing, if you want
        if self.options["verbose"]:
            print("Iter\tnR\t\tnR/nR0\t\talpha")
            print("%i\t%3.2e\t" % (0, nR0))

        # Newton-Raphson iteration
        for i in range(self.options["miter"]):
            # Can immediately skip if the initial residual is
            # less than the absolute tol
            if nR0 < self.options["atol"]:
                if self.options["verbose"]:
                    print("")
                break
            # Get the actual sparse jacobian
            J = self.jacobian()
            # Direction
            dx = self.linear_solve(J, R)
            # Linesearch
            alpha = 1.0
            x0 = np.copy(self.state_np1.displacements)
            nR_start = nR
            for _ in range(self.options["max_linesearch"]):
                # Trial
                self.state_np1.displacements = x0 - alpha * dx
                # Recalculate the state
                self.update_state()
                # Calculate the residual
                R = self.residual(p)
                # Norm of residual
                nR = la.norm(R[self.kdofs])
                # Check linesearch criteria
                if nR < nR_start:
                    break
                alpha /= 2.0

            # Check convergence
            if self.options["verbose"]:
                print("%i\t%3.2e\t%3.2e\t%3.2e" % (i + 1, nR, nR / nR0, alpha))
            if nR < self.options["atol"] or nR / nR0 < self.options["rtol"]:
                if self.options["verbose"]:
                    print("")
                break
        else:
            raise RuntimeError("Nonlinear iteration did not converge!")

        # Calculate the axial force and stiffness
        J = self.jacobian()  # Need the current tangent
        if self.state_np1.ndim == 1 or self.state_np1.ndim == 2:
            self.calculate_axial_from_stress(R, J)
        else:
            self.calculate_axial_from_fea(R, J)

    def calculate_axial_from_stress(self, R, J):
        """
        Calculate the axial force by integrating sigma_zz over the area

        Parameters:
          R       final residual
          J       final jacobian
        """
        # pylint: disable=no-value-for-parameter, invalid-unary-operand-type
        if self.state_np1.ndim == 1:
            dx = (
                self.state_np1.basis.interpolate(self.state_np1.mesh.p[0]).value[0]
                * self.state_np1.basis.dx
                * 2.0
                * np.pi
            )
        else:
            dx = self.state_np1.basis.dx

        fake_force = self._internal_force(self.state_np1.tangent[:, :, 2, 2])
        de = utils.solve(
            *utils.condense(J, fake_force, D=self.edofs),
            solver=utils.solver_direct_scipy()
        )
        fake_strain = self.calculate_strain(de)
        integrand1 = np.einsum("iijk", self.state_np1.tangent[2, 2] * fake_strain)
        integrand2 = self.state_np1.tangent[2, 2, 2, 2]

        self.state_np1.force = np.sum(self.state_np1.stress[2, 2] * dx)
        self.state_np1.stiffness = (
            np.sum((-integrand1 + integrand2) * dx) / self.state_np1.h
        )

    def calculate_axial_from_fea(self, R, J):
        """
        Calculate the axial force by summing the discrete forces on the top dofs

        Parameters:
          R       final residual
          J       final jacobian
        """
        node = self.state_np1.mesh.nodes_satisfying(
            lambda x: x[2] > self.state_np1.h - self.options["dof_tol"]
        )
        dofs = self.state_np1.basis.nodal_dofs[2, node].flatten()

        # There is a better way to do this
        sdofs = [list(self.edofs).index(d) for d in dofs]
        dotme = np.zeros((len(self.edofs),))
        dotme[sdofs] = 1.0

        K = self.state_np1.basis.complement_dofs(self.edofs)

        # Straightforward
        self.state_np1.force = np.sum(R[dofs])

        # Expensive but eh can optimize later
        Jl = J.tolil()
        J11 = Jl[K, :][:, K].tocsr()
        J12 = Jl[K, :][:, self.edofs]

        J21 = Jl[self.edofs, :][:, K]
        J22 = Jl[self.edofs, :][:, self.edofs]

        self.state_np1.stiffness = np.dot(
            dotme, J22.dot(dotme) - J21.dot(spla.spsolve(J11, J12.dot(dotme)))
        )

    def linear_solve(self, J, R):
        """
        Do the linear solve for a particular jacobian and residual

        Parameters:
          J       Jacobian sparse matrix
          R       residual vector
        """
        # pylint: disable=no-value-for-parameter
        return utils.solve(
            *utils.condense(J, R, D=self.edofs), solver=utils.solver_direct_scipy()
        )

    def residual(self, p):
        """
        Actually assemble the residual

        Parameters:
          p       Pressure, for the BC
        """
        F_int = self._internal_force(self.state_np1.stress)
        pv = self.state_np1.pbasis.interpolate(self.state_np1.pbasis.zeros() + p)
        F_ext = asm(self.external, self.state_np1.pbasis, pressure=pv)

        return F_int - F_ext

    def _internal_force(self, stress):
        """
        Assemble the internal force

        Parameters:
          stress      stresses to use
        """
        if self.ndim == 1:
            return self._internal_1d(stress[0, 0], stress[1, 1])
        else:
            return self._internal_nd(stress[: self.ndim, : self.ndim])

    def _internal_1d(self, radial, hoop):
        """
        Assemble the internal force for the axisymmetric case

        Parameters:
          radial      radial stress
          hoop        hoop stress
        """
        return asm(
            self.internal,
            self.state_np1.basis,
            radial=DiscreteField(radial),
            hoop=DiscreteField(hoop),
        )

    def _internal_nd(self, stress):
        """
        Internal force of the Cartesian cases

        Parameters:
          stress      stresses to use
        """
        return asm(self.internal, self.state_np1.basis, stress=DiscreteField(stress))

    def jacobian(self):
        """
        Actually assemble the jacobian
        """
        if self.ndim == 1:
            return asm(
                self.jac,
                self.state_np1.basis,
                Crr=DiscreteField(self.state_np1.tangent[0, 0, 0, 0]),
                Crt=DiscreteField(self.state_np1.tangent[0, 0, 1, 1]),
                Ctt=DiscreteField(self.state_np1.tangent[1, 1, 1, 1]),
                Ctr=DiscreteField(self.state_np1.tangent[1, 1, 0, 0]),
            )
        else:
            return asm(
                self.jac,
                self.state_np1.basis,
                C=DiscreteField(
                    self.state_np1.tangent[
                        : self.ndim, : self.ndim, : self.ndim, : self.ndim
                    ]
                ),
            )

    def add_dirichlet_bc(self, dofs, value):
        """
        Add a dirichlet BC

        Parameters:
          dofs        plain dof list
          value       value to set
        """
        self.ebcs.append((dofs, value))

    def assemble_dirichlet(self):
        """
        Assemble the dirichlet BCs into big vectors
        """
        self.edofs = []
        self.evalues = []

        for d, v in self.ebcs:
            self.edofs.extend(list(d))
            self.evalues.extend([v] * len(d))

        self.edofs = np.array(self.edofs, dtype=int)
        self.evalues = np.array(self.evalues)

        inds = np.argsort(self.edofs)
        self.edofs = self.edofs[inds]
        self.evalues = self.evalues[inds]
        self.kdofs = self.state_np1.basis.complement_dofs(self.edofs)

    def update_state(self):
        """
        Make the state consistent with the new displacements
        """
        # Enforce the Dirichlet BCs in the vector
        self.state_np1.displacements[self.edofs] = self.evalues

        # Calculate total strain
        self.state_np1.strain = self.calculate_strain(self.state_np1.displacements)

        # Add the optional extra strain
        if self.set_strain is not None:
            self.state_np1.strain[2, 2] = self.set_strain

        # Calculate thermal strain and mechanical strain
        self.calculate_mechanical_strain()

        # Calculate stress and tangent
        self.calculate_stress_update()

    def calculate_strain(self, D):
        """
        Calculate the strain based on the current displacements

        Parameters:
          D           displacements to use
        """
        U = self.state_np1.basis.interpolate(D)

        E = np.zeros((3, 3) + U.value[0].shape)

        E[: self.ndim, : self.ndim] = helpers.sym_grad(U)
        if self.ndim == 1:
            # Cast needed b/c they don't have a divide operator!
            E[1, 1] = np.array(U) / np.array(
                self.state_np1.basis.interpolate(self.state_np1.mesh.p.flatten())
            )

        return E

    def calculate_mechanical_strain(self):
        """
        Calculate the thermal strain increment and update mechanical strain
        """
        # Grab CTE values from NEML
        cte_old = np.zeros(self.state_n.temperature.shape)
        cte_new = np.zeros(self.state_np1.temperature.shape)
        for ind in np.ndindex(self.state_np1.temperature.shape):
            cte_old[ind] = self.state_n.material.alpha(self.state_n.temperature[ind])
            cte_new[ind] = self.state_np1.material.alpha(
                self.state_np1.temperature[ind]
            )

        # Thermal strain = thermal_strain_old + alpha * Tdot
        self.state_np1.thermal_strain = self.state_n.thermal_strain + np.eye(3)[
            :, :, np.newaxis, np.newaxis
        ] * (
            (cte_new + cte_old)
            / 2.0
            * (self.state_np1.temperature - self.state_n.temperature)
        )

        # Mechanical strain = total - thermal
        self.state_np1.mechanical_strain = (
            self.state_np1.strain - self.state_np1.thermal_strain
        )

    def calculate_stress_update(self):
        """
        The expensive function: update the stress and tangent
        """
        # Ya, you can't a view of these arrays...
        # We could reorder everything and then it would just work natively
        stress_np1 = np.zeros((self.state_np1.nq, 3, 3))
        hist_np1 = np.zeros((self.state_np1.nq, self.state_np1.material.nstore))
        A_np1 = np.zeros((self.state_np1.nq, 3, 3, 3, 3))

        # pylint: disable=c-extension-no-member
        block.block_evaluate(
            self.state_np1.material,
            self.state_np1.mechanical_strain.transpose(2, 3, 0, 1).reshape(-1, 3, 3),
            self.state_n.mechanical_strain.transpose(2, 3, 0, 1).reshape(-1, 3, 3),
            self.state_np1.temperature.flatten(),
            self.state_n.temperature.flatten(),
            self.state_np1.time,
            self.state_n.time,
            stress_np1,
            self.state_n.stress.transpose(2, 3, 0, 1).reshape(-1, 3, 3),
            hist_np1,
            self.state_n.history.transpose(1, 2, 0).reshape(
                -1, self.state_n.material.nstore
            ),
            A_np1,
            self.state_np1.energy.flatten(),
            self.state_n.energy.flatten(),
            self.state_np1.dissipation.flatten(),
            self.state_n.dissipation.flatten(),
        )

        self.state_np1.stress = stress_np1.reshape(
            (self.state_np1.ne, self.state_np1.nqi, 3, 3)
        ).transpose(2, 3, 0, 1)
        self.state_np1.history = hist_np1.reshape(
            (self.state_np1.ne, self.state_np1.nqi, self.state_np1.material.nstore)
        ).transpose(2, 0, 1)
        self.state_np1.tangent = A_np1.reshape(
            (self.state_np1.ne, self.state_np1.nqi, 3, 3, 3, 3)
        ).transpose(2, 3, 4, 5, 0, 1)


def get_mat(thing):
    """
    Small helper to wrap NEML for pickling issues
    """
    try:
        return thing.get_neml_model()
    except AttributeError:
        return thing
