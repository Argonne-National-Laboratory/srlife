import numpy as np
import numpy.linalg as la
import networkx as nx


class Network(nx.MultiGraph):
    """
    Simplified version of panel network where everything is an elastic spring

    Boundary conditions are always fixed on the bottom and free on the top
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(args) == 0:
            self.add_node("top", distance=0)
            self.cpanel = 0

    def add_panel(self, k):
        """
        Add a panel to the network with stiffness k

        Parameters:
          k:      either a float stiffness or "rigid" or "disconnect"

        Returns:
          id of the new panel
        """
        name = self._panel_name(self.cpanel)
        self.cpanel += 1
        self.add_node(name, distance=1)
        self.add_edge("top", name, stiffness=k)

        return self.cpanel - 1

    def add_tube(self, panel, kconnect, k):
        """
        Add a tube to a panel

        Parameters:
          panel:      panel id
          kconnect:   either a float stiffness or "rigid" or "disconnect"
          k:          float stiffness
        """
        if panel >= self.cpanel:
            raise ValueError("Panel %i does not exist!" % panel)

        name = self._tube_name(panel, self._next_tube(panel))
        self.add_node(name + "_top", distance=2)
        self.add_node(name + "_bottom", distance=3)

        self.add_edge(self._panel_name(panel), name + "_top", stiffness=kconnect)
        self.add_edge(name + "_top", name + "_bottom", stiffness=k)

    def simplify_network(self):
        """
        Simplify the tube network and return a list of subproblems
        """
        # Remove all rigid nodes
        self._remove_rigid()

        # Split at all disconnects
        return self._split_disconnect()

    def solve_network(self, tforce):
        """
        Impose the standard BCs and solve for the displacements at each node

        Parameters:
          tforce:     thermal force on the bars
        """
        n = len(self)
        K = np.zeros((n, n))
        d = np.zeros((n,))
        f = np.zeros((n,))

        # Make the dof maps
        self.backmap = {}
        for i, name in enumerate(self.nodes()):
            self.backmap[name] = i

        # Fix all the ones with "_bottom" in the name to zero
        fixed = []
        for i, name in enumerate(self.nodes()):
            if name[:4] == "tube" and name.split("_")[1] == "bottom":
                fixed.append(i)
        free = sorted(list(set(list(range(n))) - set(fixed)))

        # Impose the thermal force and the stiffness matrix
        for a, b, data in self.edges(data=True):
            dofs = [self.backmap[a], self.backmap[b]]
            s = np.sign(self.nodes[b]["distance"] - self.nodes[a]["distance"])
            K[np.ix_(dofs, dofs)] += data["stiffness"] * np.array([[1, -1], [-1, 1]])

            if a[:4] == "tube" and b[:4] == "tube":
                f[dofs] += tforce * np.array([1, -1]) * s

        # Solve
        fprime = f[free] - np.dot(K[np.ix_(free, fixed)], d[fixed])
        Kprime = K[np.ix_(free, free)]
        dprime = la.solve(Kprime, fprime)
        d[free] = dprime

        # Backsub the displacements
        for i, (name, data) in enumerate(self.nodes(data=True)):
            data["displacement"] = d[i]

    def _remove_rigid(self):
        """
        Remove all rigid connections by deleting the edge and joining the
        nodes
        """
        while True:
            for a, b, key, data in self.edges(data=True, keys=True):
                if data["stiffness"] == "rigid":
                    self.remove_edge(a, b, key=key)
                    # Need to keep the one marked "tube"
                    if a[:4] == "tube":
                        ng = nx.contracted_nodes(self, a, b)
                    else:
                        ng = nx.contracted_nodes(self, b, a)
                    self.clear()
                    self.update(ng)
                    break
            else:
                break

    def _split_disconnect(self):
        """
        Split the graph at all disconnects
        """
        # First just split
        while True:
            for i, j, key, data in self.edges(keys=True, data=True):
                if data["stiffness"] == "disconnect":
                    self.remove_edge(i, j, key=key)
                    break
            else:
                break

        # Setup the new structures
        base = [
            Network(self.subgraph(c))
            for c in nx.connected_components(self)
            if len(c) > 1
        ]
        nstructs = []
        for b in base:
            ns = [n[:4] for n in b.nodes()]
            if "tube" in ns:
                nstructs.append(b)

        return nstructs

    def _panel_name(self, k):
        """
        Turn an integer into a panel id

        Parameters:
          k:      panel integer id
        """
        return "panel-%i" % k

    def _tube_name(self, panel, tube):
        """
        Turn integer panel and tube ids into a base tube string id
        """
        return "tube-%i-%i" % (panel, tube)

    def _next_tube(self, panel):
        """
        Figure out what the next tube id is for a panel
        """
        start = 0
        for n in self.neighbors(self._panel_name(panel)):
            if n != "top":
                start += 1

        return start
