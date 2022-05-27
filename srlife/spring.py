# pylint: disable=no-member

"""
  Meta-structural solver defining a network of nonlinear springs
"""

from abc import ABC, abstractmethod

import numpy as np
import networkx as nx
import multiprocess

from srlife.solvers import newton


class Spring(ABC):
    """
    Spring base class defining interface.

    Class will accommodate full nonlinear tubes (through FEA) as well
    as simple linear springs
    """

    @abstractmethod
    def force_and_stiffness(self, i, d):
        """
        Return the current forces and stiffness for some displacement

        Parameters:
          i           timestep
          d           displacement
        """
        return

    @abstractmethod
    def update_state(self, i):
        """
        Update to the next state after a successful solve

        Parameters:
          i           universal timestep number
        """
        return


class LinearSpring(Spring):
    """
    A linear elastic spring
    """

    def __init__(self, k):
        """
        Parameters:
          k       spring stiffness
        """
        self.k = k
        self.state_np1 = None
        self.state_n = None

    def force_and_stiffness(self, i, d):
        """
        Simple linear update

        Parameters:
          i       universal timestep index
          d       current displacement
        """
        return self.k * d, self.k

    def update_state(self, i):
        """
        We don't have state!

        Parameters:
          i       universal timestep number
        """
        return


class TubeSpring(Spring):
    """
    A spring object encapsulating the entire tube thermo-structural solver
    """

    def __init__(self, tube, solver, material):
        """
        Parameters:
          tube        tube object, with thermal history defined
          solver      structural solver to use
          material    tube material model (NEML)
        """
        self.tube = tube
        self.solver = solver
        self.material = material

        # Setup the tube to receive results
        self.solver.setup_tube(self.tube)

        # Get the first state
        self.state_n = self.solver.init_state(self.tube, self.material, i=0)

        # Dump this first state to the Tube
        self.solver.dump_state(self.tube, 0, self.state_n)

    def force_and_stiffness(self, i, d):
        """
        Parameters:
          d       current displacement
        """
        self.state_np1 = self.solver.solve(self.tube, i, self.state_n, d)

        return self.state_np1.force, self.state_np1.stiffness

    def update_state(self, i):
        """
        Parameters:
          i       timestep
        """
        self.solver.dump_state(self.tube, i, self.state_np1)
        self.state_n = self.state_np1


class SpringNetwork(nx.MultiGraph):
    """
    This mimics the NEML class with some additional features.

    Graph structure:
      Nodes are displacements
      Edges can be:
        Bars
        "disconnect"
        "rigid"

      Where "free" and "fixed" get factored out in the final network

    Boundary conditions:
      Live at nodes
      Can be imposed forces or displacements (though only displacements are
        going to bused in the end)
      Are simply tuples of (type, function) where
        type:
          force
          displacement
        function:
          function of time giving the values

    The object takes the time values from individual tubes by default, but
    this can be overrode with new step values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.times = None
        self.displacements = None
        self.atol = kwargs.pop("atol", 1.0e-8)
        self.rtol = kwargs.pop("rtol", 1.0e-6)
        self.miter = kwargs.pop("miter", 25)
        self.verbose = kwargs.pop("verbose", False)

    def __copy(self, other):
        """
        Copy ancillary information from one network to another

        Parameters:
          other       other network
        """
        other.times = self.times
        other.rtol = self.rtol
        other.atol = self.atol
        other.miter = self.miter
        other.verbose = self.verbose

    def set_times(self, times):
        """
        Set the discrete times to a list

        Parameters:
          times       new times
        """
        self.times = times

    def validate_setup(self):
        """
        Validate the data structure
        """
        if self.times is None:
            self.__set_default_times()

        # Validate the type of each edge and the internal times (if provided)
        for _, _, edge in self.edges(data=True):
            if isinstance(edge["object"], TubeSpring):
                if not np.allclose(edge["object"].tube.times, self.times):
                    raise ValueError(
                        "A tube object has discrete times that are"
                        " not consistent with the spring network!"
                    )

            if not isinstance(edge["object"], (Spring, str)):
                raise ValueError("An edge object is not either a spring or a string!")

            if isinstance(edge["object"], str) and edge["object"] not in (
                "disconnect",
                "rigid",
            ):
                raise ValueError("Special edge type %s not recognized!" % edge)

    def __set_default_times(self):
        """
        Grab times from the first Tube object or raise error
        """
        for _, _, edge in self.edges(data=True):
            if isinstance(edge["object"], TubeSpring):
                self.times = edge["object"].tube.times
                break
        else:
            raise ValueError("No times provided and no Tube objects to copy from!")

    def force_bc(self, node, function):
        """
        Add a force boundary condition

        Parameters:
          node        location
          function    force as a function of time
        """
        self.nodes[node]["bc"] = ("force", function)

    def displacement_bc(self, node, function):
        """
        Add a displacement boundary condition

        Parameters:
          node        location
          function    displacement as a function of time
        """
        self.nodes[node]["bc"] = ("displacement", function)

    def reduce_graph(self):
        """
        Reduce the graph structure to a list of graphs which can be
        solved by:

        1) Removing rigid links
        2) Splitting the graph at disconnect
        """
        self.remove_rigid()
        return self.split_disconnect()

    def remove_rigid(self):
        """
        Remove rigid links by merging the nodes in place
        """
        while True:
            for i, j, key, edge in self.edges(data=True, keys=True):
                # Can't handle duplicate springs
                if edge["object"] == "rigid":
                    if self.number_of_edges(i, j) != 1:
                        raise RuntimeError("Cannot have a rigid link across a spring!")
                    bci = "bc" in self.nodes[i]
                    bcj = "bc" in self.nodes[j]
                    if bci and bcj:
                        raise RuntimeError("Cannot merge two nodes with BCs!")
                    self.remove_edge(i, j, key=key)

                    if i < j:
                        if bcj:
                            raise RuntimeError("Internal error: deleting BC")
                        self.replace(nx.contracted_nodes(self, i, j))
                    else:
                        if bci:
                            raise RuntimeError("Internal error: deleting BC")
                        self.replace(nx.contracted_nodes(self, j, i))
                    break
            else:
                break

    def replace(self, other):
        """
        Replace the graph data structure with a new graph
        """
        self.clear()
        self.update(other)

    def split_disconnect(self):
        """
        Split the graph into disjoint parts by taking apart at the disconnects
        """
        while True:
            for i, j, key, edge in self.edges(data=True, keys=True):
                if edge["object"] == "disconnect":
                    self.remove_edge(i, j, key=key)
                    break
            else:
                break

        components = list(nx.connected_components(self))

        nobjs = [SpringNetwork(self.subgraph(nset)) for nset in components]
        for obj in nobjs:
            self.__copy(obj)

        fobjs = []
        for obj in nobjs:
            if obj.size() != 0:
                fobjs.append(obj)

        return fobjs

    def validate_solve(self):
        """
        Validate that the graph is ready to actually solve

        Conditions:
          1) All dummy edges removed
          2) At least one fixed condition
          3) All nodes sequentially numbers from zero
        """
        # Check that we have times
        if self.times is None:
            raise RuntimeError("Discrete times must be established at solve!")

        # Check that everything is a spring
        for _, _, edge in self.edges(data=True):
            if not isinstance(edge["object"], Spring):
                raise RuntimeError("All edges must be springs at solve!")

        # Check that there is at least one fixed boundary
        ndisp = 0
        for node in self.nodes():
            if "bc" in self.nodes[node]:
                if self.nodes[node]["bc"][0] == "displacement":
                    ndisp += 1

        if ndisp == 0:
            raise RuntimeError("Spring network requires at least one fixed BC!")

        # To to see if the graph is fully-connected
        if not nx.is_connected(self):
            raise RuntimeError("Spring network must be fully connected at solve!")

    def solve(self, i, nthreads=1):
        """
        Solve discrete step i

        Parameter:
          i           discrete time step

        Additional parameters:
          nthreads    number of threads to use
        """
        # Keep forces and displacements for debug
        if self.displacements is None:
            self.displacements = np.zeros((len(self.nodes),))

        # Make sure we can solve
        self.validate_solve()

        # Set the time step
        self.i = i

        # Figure out free/fixed displacements
        (
            self.dmap,
            self.free,
            self.forces,
            self.fixed,
            self.fixed_displacements,
        ) = self.dof_maps(i)

        # Actually solve
        d = newton(
            lambda x: self.RJ(x, nthreads=nthreads),
            self.displacements[self.dmap[self.free]],
            verbose=self.verbose,
            rel_tol=self.rtol,
            abs_tol=self.atol,
            miters=self.miter,
        )

        # Store the displacements, for fun
        self.displacements[self.dmap[self.free]] = d
        self.displacements[self.dmap[self.fixed]] = self.fixed_displacements

        # Advance the state
        for _, _, edge in self.edges(data=True):
            edge["object"].update_state(self.i)

    def RJ(self, d, nthreads=1):
        """
        Actually calculate the residual and Jacobian equation for the step

        Parameters:
          d       free displacements
        """
        dall = np.zeros((len(self.nodes),))
        dall[self.dmap[self.fixed]] = self.fixed_displacements
        dall[self.dmap[self.free]] = d

        if nthreads > 1:
            with multiprocess.Pool(nthreads) as p:
                res = list(p.map(lambda e: self.fj(dall, *e), self.edges(data=True)))
        else:
            res = list(map(lambda e: self.fj(dall, *e), self.edges(data=True)))

        Fint = sum(r[0] for r in res)
        J = sum(r[1] for r in res)

        for k, (_, _, edge) in enumerate(self.edges(data=True)):
            edge["object"].state_np1 = res[k][2]

        return (
            Fint[self.dmap[self.free]] - self.forces,
            J[self.dmap[self.free], :][:, self.dmap[self.free]],
        )

    def fj(self, dall, i, j, edge):
        """
        Calculate the force and jacobian contributions for a single edge
        """
        ii = self.dmap[i]
        jj = self.dmap[j]
        ss = np.sign(ii - jj)
        Fint = np.zeros((len(self.nodes),))
        J = np.zeros((len(self.nodes), len(self.nodes)))
        f, k = edge["object"].force_and_stiffness(self.i, (dall[jj] - dall[ii]) * ss)
        Fint[ii] += -f
        Fint[jj] += f

        J[ii, ii] += k
        J[ii, jj] += -k
        J[jj, ii] += -k
        J[jj, jj] += k

        return Fint * ss, J, edge["object"].state_np1

    def dof_maps(self, i):
        """
        Get the degree of freedom maps for the given time step

        Parameters:
          i       discrete time step
        """
        free = []
        fixed = []
        forces = []
        displacements = []
        time = self.times[i]
        nodes = []
        for node in sorted(self.nodes):
            nodes.append(node)
            if "bc" in self.nodes[node]:
                if self.nodes[node]["bc"][0] == "displacement":
                    fixed.append(node)
                    displacements.append(self.nodes[node]["bc"][1](time))
                elif self.nodes[node]["bc"][0] == "force":
                    free.append(node)
                    forces.append(self.nodes[node]["bc"][1](time))
                else:
                    raise ValueError("Unknown BC type %s!" % self.nodes[node]["bc"][0])
            else:
                free.append(node)
                forces.append(0)

        nodes = np.array(nodes, dtype=int)
        total = (np.array(range(0, max(nodes) + 1), dtype=int) * 0) - 1
        rnums = np.array(range(len(nodes)), dtype=int)
        total[nodes] = rnums

        return (
            total,
            np.array(free, dtype=int),
            np.array(forces),
            np.array(fixed, dtype=int),
            np.array(displacements),
        )

    def solve_all(self, nthreads=1, decorator=lambda x, nitems: x):
        """
        Solve all time steps

        Additional parameters:
          nthreads:       number of threads to use
          decorator:      progress bar decorator
        """
        self.validate_solve()
        list(
            decorator(
                map(lambda i: self.solve(i, nthreads), range(1, len(self.times))),
                len(self.times) - 1,
            )
        )
        return self
