import unittest

import numpy as np
import sys

# import csv
import matplotlib.pyplot as plt

import os.path

from srlife import (
    materials,
    damage,
    solverparams,
)


# class TestPIAModel(unittest.TestCase):
#     """
#     Args:
#       outer_radius (float): tube outer radius
#       thickness (float): tube thickness
#       height (float): tube height
#       nr (int): number of radial increments
#       nt (int): number of circumferential increments
#       nz (int): number of axial increments
#     """

#     def setUp(self):
#         data = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 # "Spinning_disk_60k_70k.csv",
#                 "Spinning_disk_60k_80k.csv",
#             ),
#             delimiter=",",
#             skiprows=1,
#             usecols=list(range(1, 49)),
#         )

#         # defining surfaces and normals
#         self.r = 41.28  # outer_radius
#         self.t = 34.93  # thickness
#         self.h = 3.8  # height

#         self.nr = 9
#         self.nt = 24
#         self.nz = 2

#         r = np.zeros((self.nr - 1,), dtype=bool)
#         r[:] = True  # all true as all elements are surface elements
#         theta = np.ones((self.nt,), dtype=bool)[1]
#         z = np.ones((self.nz - 1,), dtype=bool)
#         self.surface = np.outer(np.outer(r, theta), z).flatten()

#         # Taking only one element along theta
#         t = (np.linspace(0, 2 * np.pi, self.nt)[1])/2
#         ns1 = np.vstack([np.cos(t), np.sin(t), np.zeros_like(t)]).T
#         ns2 = np.vstack(
#             [np.zeros_like(t), np.zeros_like(t), np.ones_like(t)]).T
#         ns3 = np.vstack(
#             [np.zeros_like(t), np.zeros_like(t), -np.ones_like(t)]).T
#         ns4 = np.vstack(
#             [np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)]).T

#         # Normals for ID
#         normals1 = np.stack((-ns1, ns2, ns3), axis=1)
#         normals2 = np.stack((ns2, ns3, ns4), axis=1)
#         normals3 = np.stack((ns1, ns2, ns3), axis=1)

#         self.normals = np.stack((normals1, normals2, normals2, normals2,
#                                  normals2, normals2, normals2, normals3), axis=1).reshape(-1, 3, 3)
#         np.set_printoptions(threshold=np.inf)

#         # Surface areas of 8 elements along radial direction
#         self.surface_areas = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "surfaces_8.csv",
#             ),
#             delimiter=",",
#         )

#         self.stress = data.reshape(data.shape[0], 8, -1)

#         vol_factor = 360 / 15

#         self.volumes = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "volumes_8.csv",
#             ),
#             delimiter=",",
#         )

#         self.volumes = vol_factor * self.volumes
#         self.temperatures = np.ones((data.shape[0], 8))

#         # Number of cycles to failure
#         self.nf = 1
#         self.period = 0.01
#         print("service life =", self.nf*self.period)
#         self.time = np.linspace(0, self.period, self.stress.shape[0])

#         # Material properties
#         self.m = 7.65
#         # Volume scale parameter in mm  74.79 in m
#         # self.s0 = 74.79 * ((1000) ** (3 / self.m))
#         # Surface scale parameter in mm  232.0 in m
#         self.s0 = 232 * ((1000) ** (2 / self.m))
#         self.c_bar = 0.82
#         self.nu = 0.219
#         self.Nv = 30
#         self.Bv = 320

#         self.material = materials.StandardCeramicMaterial(
#             np.array([0, 1000.0]),
#             np.array([self.s0, self.s0]),
#             np.array([0, 1000.0]),
#             np.array([self.m, self.m]),
#             self.c_bar,
#             self.nu,
#             np.array([0, 1000.0]),
#             np.array([self.Nv, self.Nv]),
#             np.array([0, 1000.0]),
#             np.array([self.Bv, self.Bv]),
#         )

#         self.model_time_dep = damage.PIAModel(solverparams.ParameterSet())

#     def test_definition(self):
#         actual1 = self.model_time_dep.calculate_surface_element_log_reliability(
#             self.time,
#             self.stress,
#             self.surface,
#             self.normals,
#             self.temperatures,
#             self.surface_areas,
#             self.material,
#             self.nf * self.period,
#         )
#         actual2 = self.model_time_dep.calculate_volume_element_log_reliability(
#             self.time,
#             self.stress,
#             self.temperatures,
#             self.volumes,
#             self.material,
#             self.nf * self.period,
#         )

#         # Summing up log probabilities over nelem and taking the value of one
#         R_PIA_s = np.exp(np.sum(actual1))
#         print("Time dep surface Reliability PIA = ", R_PIA_s)

#         R_PIA_v = np.exp(np.sum(actual2))
#         print("Time dep volume Reliability PIA = ", R_PIA_v)

#         # Evaluating Probability of Failure
#         Pf_PIA_s = 1 - R_PIA_s
#         print("Time dep surface Probability of failure PIA = ", Pf_PIA_s)

#         Pf_PIA_v = 1 - R_PIA_v
#         print("Time dep volume Probability of failure PIA = ", Pf_PIA_v)

#         with open("PIA_60k_80k.txt", "a+") as external_file:
#             external_file.write("\n")
#             external_file.write(f"{Pf_PIA_s}")
#             external_file.write("\n")
#             external_file.write(f"{Pf_PIA_v}")


# class TestCSEModelGriffithFlaw(unittest.TestCase):
#     def setUp(self):
#         data = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 # "Spinning_disk_60k_70k.csv",
#                 "Spinning_disk_60k_80k.csv",
#             ),
#             delimiter=",",
#             skiprows=1,
#             usecols=list(range(1, 49)),
#         )

#         # defining surfaces and normals
#         self.r = 41.28  # outer_radius
#         self.t = 34.93  # thickness
#         self.h = 3.8  # height

#         self.nr = 9
#         self.nt = 24
#         self.nz = 2

#         r = np.zeros((self.nr - 1,), dtype=bool)
#         r[:] = True  # all true as all elements are surface elements
#         theta = np.ones((self.nt,), dtype=bool)[1]
#         z = np.ones((self.nz - 1,), dtype=bool)
#         self.surface = np.outer(np.outer(r, theta), z).flatten()

#         # Taking only one element along theta
#         t = (np.linspace(0, 2 * np.pi, self.nt)[1])/2
#         ns1 = np.vstack([np.cos(t), np.sin(t), np.zeros_like(t)]).T
#         ns2 = np.vstack(
#             [np.zeros_like(t), np.zeros_like(t), np.ones_like(t)]).T
#         ns3 = np.vstack(
#             [np.zeros_like(t), np.zeros_like(t), -np.ones_like(t)]).T
#         ns4 = np.vstack(
#             [np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)]).T

#         # Normals for ID
#         normals1 = np.stack((-ns1, ns2, ns3), axis=1)
#         normals2 = np.stack((ns2, ns3, ns4), axis=1)
#         normals3 = np.stack((ns1, ns2, ns3), axis=1)

#         self.normals = np.stack((normals1, normals2, normals2, normals2,
#                                  normals2, normals2, normals2, normals3), axis=1).reshape(-1, 3, 3)
#         np.set_printoptions(threshold=np.inf)

#         # Surface areas of 8 elements along radial direction
#         self.surface_areas = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "surfaces_8.csv",
#             ),
#             delimiter=",",
#         )
#         self.stress = data.reshape(data.shape[0], 8, -1)

#         vol_factor = 360 / 15

#         self.volumes = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "volumes_8.csv",
#             ),
#             delimiter=",",
#         )

#         self.volumes = vol_factor * self.volumes
#         self.temperatures = np.ones((data.shape[0], 8))

#         # Number of cycles to failure
#         self.nf = 1
#         self.period = 0.01
#         print("service life =", self.nf*self.period)
#         self.time = np.linspace(0, self.period, self.stress.shape[0])

#         # Material properties
#         self.m = 7.65
#         # Volume scale parameter in mm  74.79 in m
#         # self.s0 = 74.79 * ((1000) ** (3 / self.m))
#         # Surface scale parameter in mm  232.0 in m
#         self.s0 = 232 * ((1000) ** (2 / self.m))
#         self.c_bar = 0.82
#         self.nu = 0.219
#         self.Nv = 30
#         self.Bv = 320

#         self.material = materials.StandardCeramicMaterial(
#             np.array([0, 1000.0]),
#             np.array([self.s0, self.s0]),
#             np.array([0, 1000.0]),
#             np.array([self.m, self.m]),
#             self.c_bar,
#             self.nu,
#             np.array([0, 1000.0]),
#             np.array([self.Nv, self.Nv]),
#             np.array([0, 1000.0]),
#             np.array([self.Bv, self.Bv]),
#         )

#         self.model_time_dep = damage.CSEModelGriffithFlaw(
#             solverparams.ParameterSet())

#     def test_definition(self):
#         actual1 = self.model_time_dep.calculate_surface_element_log_reliability(
#             self.time,
#             self.stress,
#             self.surface,
#             self.normals,
#             self.temperatures,
#             self.surface_areas,
#             self.material,
#             self.nf * self.period,
#         )
#         actual2 = self.model_time_dep.calculate_volume_element_log_reliability(
#             self.time,
#             self.stress,
#             self.temperatures,
#             self.volumes,
#             self.material,
#             self.nf * self.period,
#         )

#         # Summing up log probabilities over nelem and taking the value of one
#         R_CSE_GF_s = np.exp(np.sum(actual1))
#         print("Time dep surface Reliability CSE GF = ", R_CSE_GF_s)

#         R_CSE_GF_v = np.exp(np.sum(actual2))
#         print("Time dep volume Reliability CSE GF = ", R_CSE_GF_v)

#         # Evaluating Probability of Failure
#         Pf_CSE_GF_s = 1 - R_CSE_GF_s
#         print("Time dep surface Probability of failure CSE GF = ", Pf_CSE_GF_s)

#         Pf_CSE_GF_v = 1 - R_CSE_GF_v
#         print("Time dep volume Probability of failure CSE GF = ", Pf_CSE_GF_v)

#         with open("CSE_GF_60k_80k.txt", "a+") as external_file:
#             external_file.write("\n")
#             external_file.write(f"{Pf_CSE_GF_s}")
#             external_file.write("\n")
#             external_file.write(f"{Pf_CSE_GF_v}")


# class TestSMMModelGriffithFlaw(unittest.TestCase):
#     def setUp(self):
#         data = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 # "Spinning_disk_60k_70k.csv",
#                 "Spinning_disk_60k_80k.csv",
#             ),
#             delimiter=",",
#             skiprows=1,
#             usecols=list(range(1, 49)),
#         )

#         # defining surfaces and normals
#         self.r = 41.28  # outer_radius
#         self.t = 34.93  # thickness
#         self.h = 3.8  # height

#         self.nr = 9
#         self.nt = 24
#         self.nz = 2

#         r = np.zeros((self.nr - 1,), dtype=bool)
#         r[:] = True  # all true as all elements are surface elements
#         theta = np.ones((self.nt,), dtype=bool)[1]
#         z = np.ones((self.nz - 1,), dtype=bool)
#         self.surface = np.outer(np.outer(r, theta), z).flatten()

#         # Taking only one element along theta
#         t = (np.linspace(0, 2 * np.pi, self.nt)[1])/2
#         ns1 = np.vstack([np.cos(t), np.sin(t), np.zeros_like(t)]).T
#         ns2 = np.vstack(
#             [np.zeros_like(t), np.zeros_like(t), np.ones_like(t)]).T
#         ns3 = np.vstack(
#             [np.zeros_like(t), np.zeros_like(t), -np.ones_like(t)]).T
#         ns4 = np.vstack(
#             [np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)]).T

#         # Normals for ID
#         normals1 = np.stack((-ns1, ns2, ns3), axis=1)
#         normals2 = np.stack((ns2, ns3, ns4), axis=1)
#         normals3 = np.stack((ns1, ns2, ns3), axis=1)

#         self.normals = np.stack((normals1, normals2, normals2, normals2,
#                                  normals2, normals2, normals2, normals3), axis=1).reshape(-1, 3, 3)
#         np.set_printoptions(threshold=np.inf)

#         # Surface areas of 8 elements along radial direction
#         self.surface_areas = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "surfaces_8.csv",
#             ),
#             delimiter=",",
#         )
#         self.stress = data.reshape(data.shape[0], 8, -1)

#         vol_factor = 360 / 15

#         self.volumes = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "volumes_8.csv",
#             ),
#             delimiter=",",
#         )

#         self.volumes = vol_factor * self.volumes
#         self.temperatures = np.ones((data.shape[0], 8))

#         # Number of cycles to failure
#         self.nf = 1
#         self.period = 0.01
#         print("service life =", self.nf*self.period)
#         self.time = np.linspace(0, self.period, self.stress.shape[0])

#         # Material properties
#         self.m = 7.65
#         # Volume scale parameter in mm  74.79 in m
#         # self.s0 = 74.79 * ((1000) ** (3 / self.m))
#         # Surface scale parameter in mm  232.0 in m
#         self.s0 = 232 * ((1000) ** (2 / self.m))
#         self.c_bar = 0.82
#         self.nu = 0.219
#         self.Nv = 30
#         self.Bv = 320

#         self.material = materials.StandardCeramicMaterial(
#             np.array([0, 1000.0]),
#             np.array([self.s0, self.s0]),
#             np.array([0, 1000.0]),
#             np.array([self.m, self.m]),
#             self.c_bar,
#             self.nu,
#             np.array([0, 1000.0]),
#             np.array([self.Nv, self.Nv]),
#             np.array([0, 1000.0]),
#             np.array([self.Bv, self.Bv]),
#         )

#         self.model_time_dep = damage.SMMModelGriffithFlaw(
#             solverparams.ParameterSet())

#     def test_definition(self):
#         actual1 = self.model_time_dep.calculate_surface_element_log_reliability(
#             self.time,
#             self.stress,
#             self.surface,
#             self.normals,
#             self.temperatures,
#             self.surface_areas,
#             self.material,
#             self.nf * self.period,
#         )
#         actual2 = self.model_time_dep.calculate_volume_element_log_reliability(
#             self.time,
#             self.stress,
#             self.temperatures,
#             self.volumes,
#             self.material,
#             self.nf * self.period,
#         )

#         # Summing up log probabilities over nelem and taking the value of one
#         R_SMM_GF_s = np.exp(np.sum(actual1))
#         print("Time dep surface Reliability SMM_GF = ", R_SMM_GF_s)

#         R_SMM_GF_v = np.exp(np.sum(actual2))
#         print("Time dep volume Reliability SMM_GF = ", R_SMM_GF_v)

#         # Evaluating Probability of Failure
#         Pf_SMM_GF_s = 1 - R_SMM_GF_s
#         print("Time dep surface Probability of failure SMM_GF = ", Pf_SMM_GF_s)

#         Pf_SMM_GF_v = 1 - R_SMM_GF_v
#         print("Time dep volume Probability of failure SMM_GF = ", Pf_SMM_GF_v)

#         with open("SMM_GF_60k_80k.txt", "a+") as external_file:
#             external_file.write("\n")
#             external_file.write(f"{Pf_SMM_GF_s}")
#             external_file.write("\n")
#             external_file.write(f"{Pf_SMM_GF_v}")


# class TestSMMModelSemiCircularCrack(unittest.TestCase):
#     def setUp(self):
#         data = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 # "Spinning_disk_60k_70k.csv",
#                 "Spinning_disk_60k_80k.csv",
#             ),
#             delimiter=",",
#             skiprows=1,
#             usecols=list(range(1, 49)),
#         )

#         # defining surfaces and normals
#         self.r = 41.28  # outer_radius
#         self.t = 34.93  # thickness
#         self.h = 3.8  # height

#         self.nr = 9
#         self.nt = 24
#         self.nz = 2

#         r = np.zeros((self.nr - 1,), dtype=bool)
#         r[:] = True  # all true as all elements are surface elements
#         theta = np.ones((self.nt,), dtype=bool)[1]
#         z = np.ones((self.nz - 1,), dtype=bool)
#         self.surface = np.outer(np.outer(r, theta), z).flatten()

#         # Taking only one element along theta
#         t = (np.linspace(0, 2 * np.pi, self.nt)[1])/2
#         ns1 = np.vstack([np.cos(t), np.sin(t), np.zeros_like(t)]).T
#         ns2 = np.vstack(
#             [np.zeros_like(t), np.zeros_like(t), np.ones_like(t)]).T
#         ns3 = np.vstack(
#             [np.zeros_like(t), np.zeros_like(t), -np.ones_like(t)]).T
#         ns4 = np.vstack(
#             [np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)]).T

#         # Normals for ID
#         normals1 = np.stack((-ns1, ns2, ns3), axis=1)
#         normals2 = np.stack((ns2, ns3, ns4), axis=1)
#         normals3 = np.stack((ns1, ns2, ns3), axis=1)

#         self.normals = np.stack((normals1, normals2, normals2, normals2,
#                                  normals2, normals2, normals2, normals3), axis=1).reshape(-1, 3, 3)
#         np.set_printoptions(threshold=np.inf)

#         # Surface areas of 8 elements along radial direction
#         self.surface_areas = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "surfaces_8.csv",
#             ),
#             delimiter=",",
#         )

#         self.stress = data.reshape(data.shape[0], 8, -1)

#         vol_factor = 360 / 15

#         self.volumes = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "volumes_8.csv",
#             ),
#             delimiter=",",
#         )

#         self.volumes = vol_factor * self.volumes
#         self.temperatures = np.ones((data.shape[0], 8))

#         # Number of cycles to failure
#         self.nf = 1
#         self.period = 0.01
#         print("service life =", self.nf*self.period)
#         self.time = np.linspace(0, self.period, self.stress.shape[0])

#         # Material properties
#         self.m = 7.65
#         # Volume scale parameter in mm  74.79 in m
#         # self.s0 = 74.79 * ((1000) ** (3 / self.m))
#         # Surface scale parameter in mm  232.0 in m
#         self.s0 = 232 * ((1000) ** (2 / self.m))
#         self.c_bar = 0.82
#         self.nu = 0.219
#         self.Nv = 30
#         self.Bv = 320

#         self.material = materials.StandardCeramicMaterial(
#             np.array([0, 1000.0]),
#             np.array([self.s0, self.s0]),
#             np.array([0, 1000.0]),
#             np.array([self.m, self.m]),
#             self.c_bar,
#             self.nu,
#             np.array([0, 1000.0]),
#             np.array([self.Nv, self.Nv]),
#             np.array([0, 1000.0]),
#             np.array([self.Bv, self.Bv]),
#         )
#         self.model_time_dep = damage.SMMModelSemiCircularCrack(
#             solverparams.ParameterSet()
#         )

#     def test_definition(self):
#         actual1 = self.model_time_dep.calculate_surface_element_log_reliability(
#             self.time,
#             self.stress,
#             self.surface,
#             self.normals,
#             self.temperatures,
#             self.surface_areas,
#             self.material,
#             self.nf * self.period,
#         )
#         actual2 = self.model_time_dep.calculate_volume_element_log_reliability(
#             self.time,
#             self.stress,
#             self.temperatures,
#             self.volumes,
#             self.material,
#             self.nf * self.period,
#         )

#         # Summing up log probabilities over nelem and taking the value of one
#         R_SMM_SCC_s = np.exp(np.sum(actual1))
#         print("Time dep surface ReliabilityM SMM_SCC = ", R_SMM_SCC_s)

#         R_SMM_SCC_v = np.exp(np.sum(actual2))
#         print("Time dep volume Reliability SMM_SCC = ", R_SMM_SCC_v)

#         # Evaluating Probability of Failure
#         Pf_SMM_SCC_s = 1 - R_SMM_SCC_s
#         print("Time dep surface Probability of failure SMM_SCC = ", Pf_SMM_SCC_s)

#         Pf_SMM_SCC_v = 1 - R_SMM_SCC_v
#         print("Time dep volume Probability of failure SMM_SCC = ", Pf_SMM_SCC_v)

#         with open("SMM_SCC_60k_80k.txt", "a+") as external_file:
#             external_file.write("\n")
#             external_file.write(f"{Pf_SMM_SCC_s}")
#             external_file.write("\n")
#             external_file.write(f"{Pf_SMM_SCC_v}")

