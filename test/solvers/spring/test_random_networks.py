#!/usr/bin/env python3

import sys

sys.path.append("../../..")

import random as ra

import matplotlib.pyplot as plt

from srlife import spring, structural, receiver

from neml import models, elasticity

import unittest

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


def gen_tube(k, tforce, tube_length, tube_ri, E, alpha, nr=5):
    emodel = elasticity.IsotropicLinearElasticModel(E, "youngs", 0.25, "poissons")
    mmodel = models.SmallStrainElasticity(emodel, alpha=alpha)

    solver = structural.PythonTubeSolver(atol=1.0e-4)

    ro = np.sqrt(tube_length * k / (np.pi * E) + tube_ri**2.0)
    A = np.pi * (ro**2.0 - tube_ri**2.0)

    tube = receiver.Tube(ro, ro - tube_ri, tube_length, nr, 1, 1)
    tube.set_times(np.array([0, 1.0]))
    tube.make_1D(tube_length / 2.0, 0)

    dT = tforce / (E * alpha * A)

    temp = np.zeros((2, nr))
    temp[1] = dT

    tube.add_results("temperature", temp)
    tube.set_pressure_bc(receiver.PressureBC(np.array([0, 1]), np.array([0, 0])))

    return spring.TubeSpring(tube, solver, mmodel)


def copy_network(base, tube_length, tube_ri, E, alpha, tforce):
    other = spring.SpringNetwork()

    for n in base.nodes():
        other.add_node(n)

    for a, b, data in base.edges(data=True):
        if a[:4] == "tube" and b[:4] == "tube":
            other.add_edge(
                a,
                b,
                object=gen_tube(
                    data["stiffness"], tforce, tube_length, tube_ri, E, alpha
                ),
            )
        else:
            if data["stiffness"] in ["rigid", "disconnect"]:
                other.add_edge(a, b, object=data["stiffness"])
            else:
                other.add_edge(a, b, object=spring.LinearSpring(data["stiffness"]))

    for n in other.nodes():
        if n[:4] == "tube" and n.split("_")[1] == "bottom":
            other.displacement_bc(n, lambda x: 0)

    other.set_times([0, 1])
    other.validate_setup()

    return other


class TestNetworksWithDisconnects(unittest.TestCase):
    def setUp(self):
        self.ntrials = 4
        self.tforce = 1.2

        self.kpc = 10.0
        self.ktc = 20.0
        self.kt = 5.0

        self.npanels = 5
        self.ntubes = 5

        self.tube_length = 1.0
        self.tube_ri = 0.9
        self.E = 120000.0
        self.alpha = 1.0e-5

    def test_random(self):
        ran = 0

        while ran < self.ntrials:
            ref, main = self.gen_network()

            subprobs1 = ref.simplify_network()
            subprobs1.sort(key=lambda n: len(n))
            subprobs2 = main.reduce_graph()
            subprobs2.sort(key=lambda n: len(n))

            sz1 = [len(n) for n in subprobs1]
            sz2 = [len(n) for n in subprobs2]

            b1 = [n for n in sz1 if n > 2]

            if len(b1) > 1:
                for i, j in zip(b1[1:], b1[:-1]):
                    if i == j:
                        break
                else:
                    continue

            for prob in subprobs1:
                prob.solve_network(self.tforce)

            for prob in subprobs2:
                prob.solve_all()

            for p1, p2 in zip(subprobs1, subprobs2):
                disp1 = np.sort(
                    np.array([data["displacement"] for n, data in p1.nodes(data=True)])
                )
                disp2 = np.sort(p2.displacements)

                self.assertEqual(len(disp1), len(disp2))
                print(disp1, disp2)
                self.assertTrue(np.allclose(disp1, disp2))

            ran += 1

    def gen_network(self):
        test = Network()

        for i in range(self.npanels):
            pid = test.add_panel(
                ra.choice(
                    [self.kpc, 0.8 * self.kpc, 1.2 * self.kpc] * 3
                    + ["rigid", "disconnect"]
                )
            )
            for j in range(self.ntubes):
                test.add_tube(
                    pid,
                    ra.choice(
                        [self.ktc, 0.8 * self.ktc, 1.2 * self.ktc] * 3
                        + ["rigid", "disconnect"]
                    ),
                    self.kt,
                )

        nn = copy_network(
            test, self.tube_length, self.tube_ri, self.E, self.alpha, self.tforce
        )
        old2new = {}
        new2old = {}
        for i, n in enumerate(nn.nodes()):
            old2new[n] = i
            new2old[i] = n
        nx.relabel_nodes(nn, old2new, copy=False)

        return test, nn


class TestNetworksNoDisconnects(unittest.TestCase):
    def setUp(self):
        self.ntrials = 4
        self.tforce = 1.2

        self.kpc = 10.0
        self.ktc = 20.0
        self.kt = 5.0

        self.npanels = 5
        self.ntubes = 5

        self.tube_length = 1.0
        self.tube_ri = 0.9
        self.E = 120000.0
        self.alpha = 1.0e-5

    def test_random(self):
        ran = 0

        while ran < self.ntrials:
            ref, main = self.gen_network()

            subprobs1 = ref.simplify_network()
            subprobs1.sort(key=lambda n: len(n))
            subprobs2 = main.reduce_graph()
            subprobs2.sort(key=lambda n: len(n))

            sz1 = [len(n) for n in subprobs1]
            sz2 = [len(n) for n in subprobs2]

            b1 = [n for n in sz1 if n > 2]

            if len(b1) > 1:
                for i, j in zip(b1[1:], b1[:-1]):
                    if i == j:
                        break
                else:
                    continue

            for prob in subprobs1:
                prob.solve_network(self.tforce)

            for prob in subprobs2:
                prob.solve_all()

            for p1, p2 in zip(subprobs1, subprobs2):
                disp1 = np.sort(
                    np.array([data["displacement"] for n, data in p1.nodes(data=True)])
                )
                disp2 = np.sort(p2.displacements)

                self.assertEqual(len(disp1), len(disp2))
                print(disp1, disp2)
                self.assertTrue(np.allclose(disp1, disp2, rtol=1e-4))

            ran += 1

    def gen_network(self):
        test = Network()

        for i in range(self.npanels):
            pid = test.add_panel(
                ra.choice([self.kpc, 0.8 * self.kpc, 1.2 * self.kpc] * 3 + ["rigid"])
            )
            for j in range(self.ntubes):
                test.add_tube(
                    pid,
                    ra.choice(
                        [self.ktc, 0.8 * self.ktc, 1.2 * self.ktc] * 3 + ["rigid"]
                    ),
                    ra.choice([self.kt, 0.8 * self.kt, 1.2 * self.kt]),
                )

        nn = copy_network(
            test, self.tube_length, self.tube_ri, self.E, self.alpha, self.tforce
        )
        old2new = {}
        new2old = {}
        for i, n in enumerate(nn.nodes()):
            old2new[n] = i
            new2old[i] = n
        nx.relabel_nodes(nn, old2new, copy=False)

        return test, nn
