import unittest

import numpy as np

# import csv
import matplotlib.pyplot as plt

import os.path

from srlife import (
    materials,
    damage,
    solverparams,
)


# class TestPIAModel(unittest.TestCase):
#     def setUp(self):
#         data = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "Spinning_disk_90k.csv",
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
#         # t = (np.linspace(0, 2 * np.pi, self.nt)[1])/2
#         t = np.linspace(0, 2 * np.pi, self.nt)[1]
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
        
#         self.nr = 8
#         self.nt = 24
#         self.nz = 2

#         # Calculating and printing surface areas of surface elements
#         t = np.linspace(0, 2 * np.pi, self.nt+1)
#         r = np.linspace(self.r - self.t, self.r, self.nr)
#         z = np.linspace(0, self.h, self.nz)
#         theta = np.diff(t)
#         heights = np.diff(z)
    
#         surface_area2 = np.zeros((self.nr-1,self.nz-1,self.nt))
#         surface_area2[0,:,:] = heights[:,np.newaxis]*theta[np.newaxis,:]*r[0]
#         surface_area2[-1,:,:] = heights[:,np.newaxis]*theta[np.newaxis,:]*r[-1]
#         surface_area2 = np.concatenate((surface_area2[0,:,:],surface_area2[-1,:,:]))
#         print("surface_area2 =",surface_area2)
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
#         self.nf = 100
#         self.period = 0.01
#         print("service life =", self.nf*self.period)
#         self.time = np.linspace(0, self.period, self.stress.shape[0])

#         # Material properties
#         self.m_v = 7.65
#         self.m_s = 7.65
#         # Volume scale parameter in mm  74.79 in m
#         self.s0_v = 74.79 * ((1000) ** (3 / self.m_v))
#         # Surface scale parameter in mm  232.0 in m
#         self.s0_s = 232 * ((1000) ** (2 / self.m_s))
#         self.c_bar = 0.82
#         self.nu = 0.219
#         self.Nv = 30
#         self.Ns = 30
#         self.Bv = 320
#         self.Bs = 320

#         self.material = materials.StandardCeramicMaterial(
#             np.array([0, 1000.0]),
#             np.array([self.s0_v, self.s0_v]),
#             np.array([self.s0_s, self.s0_s]),
#             np.array([0, 1000.0]),
#             np.array([self.m_v, self.m_v]),
#             np.array([self.m_s, self.m_s]),
#             self.c_bar,
#             self.nu,
#             np.array([0, 1000.0]),
#             np.array([self.Nv, self.Nv]),
#             np.array([self.Ns, self.Ns]),
#             np.array([0, 1000.0]),
#             np.array([self.Bv, self.Bv]),
#             np.array([self.Bs, self.Bs]),
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

#         with open("PIA_100k.txt", "a+") as external_file:
#             external_file.write("\n")
#             external_file.write(f"{Pf_PIA_s}")
#             external_file.write("\n")
#             external_file.write(f"{Pf_PIA_v}")

# class TestWNTSAModel(unittest.TestCase):
#     def setUp(self):
#         data = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "Spinning_disk_90k.csv",
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

#         self.model_time_dep = damage.WNTSAModel(solverparams.ParameterSet())

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
#         R_WNTSA_s = np.exp(np.sum(actual1))
#         print("Time dep surface Reliability WNTSA = ", R_WNTSA_s)

#         R_WNTSA_v = np.exp(np.sum(actual2))
#         print("Time dep volume Reliability WNTSA = ", R_WNTSA_v)

#         # Evaluating Probability of Failure
#         Pf_PIA_s = 1 - R_WNTSA_s
#         print("Time dep surface Probability of failure WNTSA = ", Pf_PIA_s)

#         Pf_PIA_v = 1 - R_WNTSA_v
#         print("Time dep volume Probability of failure WNTSA = ", Pf_PIA_v)

#         with open("WNTSA_100k.txt", "a+") as external_file:
#             external_file.write("\n")
#             external_file.write(f"{Pf_WNTSA_s}")
#             external_file.write("\n")
#             external_file.write(f"{Pf_WNTSA_v}")

# class TestMTSModelGriffithFlaw(unittest.TestCase):
#     def setUp(self):
#         data = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "Spinning_disk_100k.csv",
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
#         self.nf = 0
#         self.period = 0.01
#         print("service life =", self.nf*self.period)
#         self.time = np.linspace(0, self.period, self.stress.shape[0])

#         # Material properties
#         self.m = 7.65
#         # Volume scale parameter in mm  74.79 in m
#         self.s0 = 74.79 * ((1000) ** (3 / self.m))
#         # Surface scale parameter in mm  232.0 in m
#         # self.s0 = 232 * ((1000) ** (2 / self.m))
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

#         self.model_time_dep = damage.MTSModelGriffithFlaw(solverparams.ParameterSet())

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
#         R_MTS_GF_s = np.exp(np.sum(actual1))
#         print("Time dep surface Reliability MTS_GF = ", R_MTS_GF_s)

#         R_MTS_GF_v = np.exp(np.sum(actual2))
#         print("Time dep volume Reliability MTS_GF = ", R_MTS_GF_v)

#         # Evaluating Probability of Failure
#         Pf_MTS_GF_s = 1 - R_MTS_GF_s
#         print("Time dep surface Probability of failure MTS_GF = ", Pf_MTS_GF_s)

#         Pf_MTS_GF_v = 1 - R_MTS_GF_v
#         print("Time dep volume Probability of failure MTS_GF = ", Pf_MTS_GF_v)

#         with open("MTS_GF_90k.txt", "a+") as external_file:
#             external_file.write("\n")
#             external_file.write(f"{Pf_MTS_GF_s}")
#             external_file.write("\n")
#             external_file.write(f"{Pf_MTS_GF_v}")

# class TestMTSModelPennyShapedFlaw(unittest.TestCase):
#     def setUp(self):
#         data = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "Spinning_disk_100k.csv",
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
#         self.nf = 0
#         self.period = 0.01
#         print("service life =", self.nf*self.period)
#         self.time = np.linspace(0, self.period, self.stress.shape[0])

#         # Material properties
#         self.m = 7.65
#         # Volume scale parameter in mm  74.79 in m
#         self.s0 = 74.79 * ((1000) ** (3 / self.m))
#         # Surface scale parameter in mm  232.0 in m
#         # self.s0 = 232 * ((1000) ** (2 / self.m))
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

#         self.model_time_dep = damage.MTSModelPennyShapedFlaw(solverparams.ParameterSet())

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
#         R_MTS_PSF_s = np.exp(np.sum(actual1))
#         print("Time dep surface Reliability MTS_PSF = ", R_MTS_PSF_s)

#         R_MTS_PSF_v = np.exp(np.sum(actual2))
#         print("Time dep volume Reliability MTS_PSF = ", R_MTS_PSF_v)

#         # Evaluating Probability of Failure
#         Pf_MTS_PSF_s = 1 - R_MTS_PSF_s
#         print("Time dep surface Probability of failure MTS_PSF = ", Pf_MTS_PSF_s)

#         Pf_MTS_PSF_v = 1 - R_MTS_PSF_v
#         print("Time dep volume Probability of failure MTS_PSF = ", Pf_MTS_PSF_v)

#         with open("MTS_PSF_90k.txt", "a+") as external_file:
#             external_file.write("\n")
#             external_file.write(f"{Pf_MTS_PSF_s}")
#             external_file.write("\n")
#             external_file.write(f"{Pf_MTS_PSF_v}")


class TestCSEModelGriffithFlaw(unittest.TestCase):
    def setUp(self):
        data = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_90k.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(1, 49)),
        )

        # defining surfaces and normals
        self.r = 41.28  # outer_radius
        self.t = 34.93  # thickness
        self.h = 3.8  # height

        self.nr = 9
        self.nt = 24
        self.nz = 2

        r = np.zeros((self.nr - 1,), dtype=bool)
        r[:] = True  # all true as all elements are surface elements
        theta = np.ones((self.nt,), dtype=bool)[1]
        z = np.ones((self.nz - 1,), dtype=bool)
        self.surface = np.outer(np.outer(r, theta), z).flatten()

        # Taking only one element along theta
        t = (np.linspace(0, 2 * np.pi, self.nt)[1])/2
        ns1 = np.vstack([np.cos(t), np.sin(t), np.zeros_like(t)]).T
        ns2 = np.vstack(
            [np.zeros_like(t), np.zeros_like(t), np.ones_like(t)]).T
        ns3 = np.vstack(
            [np.zeros_like(t), np.zeros_like(t), -np.ones_like(t)]).T
        ns4 = np.vstack(
            [np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)]).T

        # Normals for ID
        normals1 = np.stack((-ns1, ns2, ns3), axis=1)
        normals2 = np.stack((ns2, ns3, ns4), axis=1)
        normals3 = np.stack((ns1, ns2, ns3), axis=1)

        self.normals = np.stack((normals1, normals2, normals2, normals2,
                                 normals2, normals2, normals2, normals3), axis=1).reshape(-1, 3, 3)
        np.set_printoptions(threshold=np.inf)
        # Surface areas of 8 elements along radial direction
        self.surface_areas = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "surfaces_8.csv",
            ),
            delimiter=",",
        )

        self.stress = data.reshape(data.shape[0], 8, -1)

        vol_factor = 360 / 15

        self.volumes = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "volumes_8.csv",
            ),
            delimiter=",",
        )

        self.volumes = vol_factor * self.volumes
        self.temperatures = np.ones((data.shape[0], 8))

        # Number of cycles to failure
        self.nf = 100
        self.period = 0.01
        print("service life =", self.nf*self.period)
        self.time = np.linspace(0, self.period, self.stress.shape[0])

        # Material properties
        self.m_v = 7.65
        self.m_s = 7.65
        # Volume scale parameter in mm  74.79 in m
        self.s0_v = 74.79 * ((1000) ** (3 / self.m_v))
        # Surface scale parameter in mm  232.0 in m
        self.s0_s = 232 * ((1000) ** (2 / self.m_s))
        self.c_bar = 0.82
        self.nu = 0.219
        self.Nv = 30
        self.Bv = 320
        self.Ns = 30
        self.Bs = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0_v, self.s0_v]),
            np.array([self.s0_s, self.s0_s]),
            np.array([0, 1000.0]),
            np.array([self.m_v, self.m_v]),
            np.array([self.m_s, self.m_s]),
            self.c_bar,
            self.nu,
            np.array([0, 1000.0]),
            np.array([self.Nv, self.Nv]),
            np.array([self.Ns, self.Ns]),
            np.array([0, 1000.0]),
            np.array([self.Bv, self.Bv]),
            np.array([self.Bs, self.Bs]),
        )

        self.model_time_dep = damage.CSEModelGriffithFlaw(
            solverparams.ParameterSet())

    def test_definition(self):
        actual1 = self.model_time_dep.calculate_surface_element_log_reliability(
            self.time,
            self.stress,
            self.surface,
            self.normals,
            self.temperatures,
            self.surface_areas,
            self.material,
            self.nf * self.period,
        )
        actual2 = self.model_time_dep.calculate_volume_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )

        # Summing up log probabilities over nelem and taking the value of one
        R_CSE_GF_s = np.exp(np.sum(actual1))
        print("Time dep surface Reliability CSE GF = ", R_CSE_GF_s)

        R_CSE_GF_v = np.exp(np.sum(actual2))
        print("Time dep volume Reliability CSE GF = ", R_CSE_GF_v)

        # Evaluating Probability of Failure
        Pf_CSE_GF_s = 1 - R_CSE_GF_s
        print("Time dep surface Probability of failure CSE GF = ", Pf_CSE_GF_s)

        Pf_CSE_GF_v = 1 - R_CSE_GF_v
        print("Time dep volume Probability of failure CSE GF = ", Pf_CSE_GF_v)

        with open("CSE_GF_100k.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_CSE_GF_s}")
            external_file.write("\n")
            external_file.write(f"{Pf_CSE_GF_v}")

# class TestCSEModelPennyShapedFlaw(unittest.TestCase):
#     def setUp(self):
#         data = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "Spinning_disk_100k.csv",
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
#         self.nf = 0
#         self.period = 0.01
#         print("service life =", self.nf*self.period)
#         self.time = np.linspace(0, self.period, self.stress.shape[0])

#         # Material properties
#         self.m = 7.65
#         # Volume scale parameter in mm  74.79 in m
#         self.s0 = 74.79 * ((1000) ** (3 / self.m))
#         # Surface scale parameter in mm  232.0 in m
#         # self.s0 = 232 * ((1000) ** (2 / self.m))
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

#         self.model_time_dep = damage.CSEModelPennyShapedFlaw(
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
#         R_CSE_PSF_s = np.exp(np.sum(actual1))
#         print("Time dep surface Reliability CSE PSF = ", R_CSE_PSF_s)

#         R_CSE_PSF_v = np.exp(np.sum(actual2))
#         print("Time dep volume Reliability CSE PSF = ", R_CSE_PSF_v)

#         # Evaluating Probability of Failure
#         Pf_CSE_PSF_s = 1 - R_CSE_PSF_s
#         print("Time dep surface Probability of failure CSE PSF = ", Pf_CSE_PSF_s)

#         Pf_CSE_PSF_v = 1 - R_CSE_PSF_v
#         print("Time dep volume Probability of failure CSE PSF = ", Pf_CSE_PSF_v)

#         with open("CSE_PSF_100k.txt", "a+") as external_file:
#             external_file.write("\n")
#             external_file.write(f"{Pf_CSE_PSF_s}")
#             external_file.write("\n")
#             external_file.write(f"{Pf_CSE_PSF_v}")

# class TestCSEModelGriffithNotch(unittest.TestCase):
#     def setUp(self):
#         data = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "Spinning_disk_120k.csv",
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
#         self.nf = 0
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

#         self.model_time_dep = damage.CSEModelGriffithNotch(
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
#         R_CSE_GN_s = np.exp(np.sum(actual1))
#         print("Time dep surface Reliability CSE GN = ", R_CSE_GN_s)

#         R_CSE_GN_v = np.exp(np.sum(actual2))
#         print("Time dep volume Reliability CSE GN = ", R_CSE_GN_v)

#         # Evaluating Probability of Failure
#         Pf_CSE_GN_s = 1 - R_CSE_GN_s
#         print("Time dep surface Probability of failure CSE GN = ", Pf_CSE_GN_s)

#         Pf_CSE_GN_v = 1 - R_CSE_GN_v
#         print("Time dep volume Probability of failure CSE GN = ", Pf_CSE_GN_v)

#         with open("CSE_GN_100k.txt", "a+") as external_file:
#             external_file.write("\n")
#             external_file.write(f"{Pf_CSE_GN_s}")
#             external_file.write("\n")
#             external_file.write(f"{Pf_CSE_GN_v}")

# class TestSMMModelGriffithFlaw(unittest.TestCase):
#     def setUp(self):
#         data = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "Spinning_disk_90k.csv",
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
#         self.nf = 100
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

#         with open("SMM_GF_100k.txt", "a+") as external_file:
#             external_file.write("\n")
#             external_file.write(f"{Pf_SMM_GF_s}")
#             external_file.write("\n")
#             external_file.write(f"{Pf_SMM_GF_v}")

# class TestSMMModelGriffithNotch(unittest.TestCase):
#     def setUp(self):
#         data = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "Spinning_disk_110k.csv",
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
#         self.nf = 0
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

#         self.model_time_dep = damage.SMMModelGriffithNotch(
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
#         R_SMM_GN_s = np.exp(np.sum(actual1))
#         print("Time dep surface ReliabilityM SMM_GN = ", R_SMM_GN_s)

#         R_SMM_GN_v = np.exp(np.sum(actual2))
#         print("Time dep volume Reliability SMM_GN = ", R_SMM_GN_v)

#         # Evaluating Probability of Failure
#         Pf_SMM_GN_s = 1 - R_SMM_GN_s
#         print("Time dep surface Probability of failure SMM_GN = ", Pf_SMM_GN_s)

#         Pf_SMM_GN_v = 1 - R_SMM_GN_v
#         print("Time dep volume Probability of failure SMM_GN = ", Pf_SMM_GN_v)

#         with open("SMM_GN_100k.txt", "a+") as external_file:
#             external_file.write("\n")
#             external_file.write(f"{Pf_SMM_GN_s}")
#             external_file.write("\n")
#             external_file.write(f"{Pf_SMM_GN_v}")

# class TestSMMModelPennyShapedFlaw(unittest.TestCase):
#     def setUp(self):
#         data = np.loadtxt(
#             os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "Spinning_disk_100k.csv",
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
#         self.nf = 0
#         self.period = 0.01
#         print("service life =", self.nf*self.period)
#         self.time = np.linspace(0, self.period, self.stress.shape[0])

#         # Material properties
#         self.m = 7.65
#         # Volume scale parameter in mm  74.79 in m
#         self.s0 = 74.79 * ((1000) ** (3 / self.m))
#         # Surface scale parameter in mm  232.0 in m
#         # self.s0 = 232 * ((1000) ** (2 / self.m))
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

#         self.model_time_dep = damage.SMMModelPennyShapedFlaw(
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
#         R_SMM_PSF_s = np.exp(np.sum(actual1))
#         print("Time dep surface ReliabilityM SMM_PSF = ", R_SMM_PSF_s)

#         R_SMM_PSF_v = np.exp(np.sum(actual2))
#         print("Time dep volume Reliability SMM_PSF = ", R_SMM_PSF_v)

#         # Evaluating Probability of Failure
#         Pf_SMM_PSF_s = 1 - R_SMM_PSF_s
#         print("Time dep surface Probability of failure SMM_PSF = ", Pf_SMM_PSF_s)

#         Pf_SMM_PSF_v = 1 - R_SMM_PSF_v
#         print("Time dep volume Probability of failure SMM_PSF = ", Pf_SMM_PSF_v)

#         with open("SMM_PSF_100k.txt", "a+") as external_file:
#             external_file.write("\n")
#             external_file.write(f"{Pf_SMM_PSF_s}")
#             external_file.write("\n")
#             external_file.write(f"{Pf_SMM_PSF_v}")


# class TestSMMModelSemiCircularCrack(unittest.TestCase):
    # def setUp(self):
    #     data = np.loadtxt(
    #         os.path.join(
    #             os.path.dirname(os.path.abspath(__file__)),
    #             "Spinning_disk_90k.csv",
    #         ),
    #         delimiter=",",
    #         skiprows=1,
    #         usecols=list(range(1, 49)),
    #     )

    #     # defining surfaces and normals
    #     self.r = 41.28  # outer_radius
    #     self.t = 34.93  # thickness
    #     self.h = 3.8  # height

    #     self.nr = 9
    #     self.nt = 24
    #     self.nz = 2

    #     r = np.zeros((self.nr - 1,), dtype=bool)
    #     r[:] = True  # all true as all elements are surface elements
    #     theta = np.ones((self.nt,), dtype=bool)[1]
    #     z = np.ones((self.nz - 1,), dtype=bool)
    #     self.surface = np.outer(np.outer(r, theta), z).flatten()

    #     # Taking only one element along theta
    #     t = (np.linspace(0, 2 * np.pi, self.nt)[1])/2
    #     ns1 = np.vstack([np.cos(t), np.sin(t), np.zeros_like(t)]).T
    #     ns2 = np.vstack(
    #         [np.zeros_like(t), np.zeros_like(t), np.ones_like(t)]).T
    #     ns3 = np.vstack(
    #         [np.zeros_like(t), np.zeros_like(t), -np.ones_like(t)]).T
    #     ns4 = np.vstack(
    #         [np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)]).T

    #     # Normals for ID
    #     normals1 = np.stack((-ns1, ns2, ns3), axis=1)
    #     normals2 = np.stack((ns2, ns3, ns4), axis=1)
    #     normals3 = np.stack((ns1, ns2, ns3), axis=1)

    #     self.normals = np.stack((normals1, normals2, normals2, normals2,
    #                              normals2, normals2, normals2, normals3), axis=1).reshape(-1, 3, 3)
    #     np.set_printoptions(threshold=np.inf)
    #     # Surface areas of 8 elements along radial direction
    #     self.surface_areas = np.loadtxt(
    #         os.path.join(
    #             os.path.dirname(os.path.abspath(__file__)),
    #             "surfaces_8.csv",
    #         ),
    #         delimiter=",",
    #     )

    #     self.stress = data.reshape(data.shape[0], 8, -1)

    #     vol_factor = 360 / 15

    #     self.volumes = np.loadtxt(
    #         os.path.join(
    #             os.path.dirname(os.path.abspath(__file__)),
    #             "volumes_8.csv",
    #         ),
    #         delimiter=",",
    #     )

    #     self.volumes = vol_factor * self.volumes
    #     self.temperatures = np.ones((data.shape[0], 8))

    #     # Number of cycles to failure
    #     self.nf = 100
    #     self.period = 0.01
    #     print("service life =", self.nf*self.period)
    #     self.time = np.linspace(0, self.period, self.stress.shape[0])

    #     # Material properties
    #     self.m = 7.65
    #     # Volume scale parameter in mm  74.79 in m
    #     # self.s0 = 74.79 * ((1000) ** (3 / self.m))
    #     # Surface scale parameter in mm  232.0 in m
    #     self.s0 = 232 * ((1000) ** (2 / self.m))
    #     self.c_bar = 0.82
    #     self.nu = 0.219
    #     self.Nv = 30
    #     self.Bv = 320

    #     self.material = materials.StandardCeramicMaterial(
    #         np.array([0, 1000.0]),
    #         np.array([self.s0, self.s0]),
    #         np.array([0, 1000.0]),
    #         np.array([self.m, self.m]),
    #         self.c_bar,
    #         self.nu,
    #         np.array([0, 1000.0]),
    #         np.array([self.Nv, self.Nv]),
    #         np.array([0, 1000.0]),
    #         np.array([self.Bv, self.Bv]),
    #     )
    #     self.model_time_dep = damage.SMMModelSemiCircularCrack(
    #         solverparams.ParameterSet()
    #     )

    # def test_definition(self):
    #     actual1 = self.model_time_dep.calculate_surface_element_log_reliability(
    #         self.time,
    #         self.stress,
    #         self.surface,
    #         self.normals,
    #         self.temperatures,
    #         self.surface_areas,
    #         self.material,
    #         self.nf * self.period,
    #     )
    #     actual2 = self.model_time_dep.calculate_volume_element_log_reliability(
    #         self.time,
    #         self.stress,
    #         self.temperatures,
    #         self.volumes,
    #         self.material,
    #         self.nf * self.period,
    #     )

    #     # Summing up log probabilities over nelem and taking the value of one
    #     R_SMM_SCC_s = np.exp(np.sum(actual1))
    #     print("Time dep surface ReliabilityM SMM_SCC = ", R_SMM_SCC_s)

    #     R_SMM_SCC_v = np.exp(np.sum(actual2))
    #     print("Time dep volume Reliability SMM_SCC = ", R_SMM_SCC_v)

    #     # Evaluating Probability of Failure
    #     Pf_SMM_SCC_s = 1 - R_SMM_SCC_s
    #     print("Time dep surface Probability of failure SMM_SCC = ", Pf_SMM_SCC_s)

    #     Pf_SMM_SCC_v = 1 - R_SMM_SCC_v
    #     print("Time dep volume Probability of failure SMM_SCC = ", Pf_SMM_SCC_v)

    #     with open("SMM_SCC_100k.txt", "a+") as external_file:
    #         external_file.write("\n")
    #         external_file.write(f"{Pf_SMM_SCC_s}")
    #         external_file.write("\n")
    #         external_file.write(f"{Pf_SMM_SCC_v}")
