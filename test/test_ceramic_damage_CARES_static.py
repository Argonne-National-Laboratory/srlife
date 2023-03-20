import unittest

import numpy as np

# import csv
import matplotlib.pyplot as plt

import os.path

from srlife import (
    materials,
    damage_time_dep_cyclic,
    solverparams,
)


class TestPIAModel(unittest.TestCase):
    def setUp(self):
        # data = np.loadtxt("Spinning_disk_cyclic_60000_70000.txt")
        # data = np.loadtxt(
        #     os.path.join(
        #         os.path.dirname(os.path.abspath(__file__)),
        #         "Spinning_disk_cyclic_60000_70000.txt",
        #         # "Spinning_disk_cyclic_60000_80000.txt",
        #     )
        # )
        # data = np.loadtxt("Spinning_disk_static_60000.txt")
        data = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_60k.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(1, 49)),
        )

        self.stress = data.reshape(data.shape[0], 8, -1)

        vol_factor = 360 / 15
        # self.volumes = np.loadtxt(
        #     os.path.join(
        #         os.path.dirname(os.path.abspath(__file__)),
        #         "Spinning_disk_volumes.txt",
        #     )
        # )

        self.volumes = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "volumes_8.csv",
            ),
            delimiter=",",
            # skiprows=1,
            # usecols=list(range(1, 55)),
        )
        self.volumes = vol_factor * self.volumes
        self.temperatures = np.ones((data.shape[0], 8))

        # Number of cycles to failure
        self.nf = 0
        self.period = 0.01
        self.time = np.linspace(0, self.period, self.stress.shape[0])

        # Material properties
        self.m = 7.65
        self.s0 = 74.79 * ((1000) ** (3 / self.m))  # in mm  74.79 in m
        self.c_bar = 0.82
        self.nu = 0.219
        self.Nv = 30
        self.Bv = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            np.array([0, 1000.0]),
            np.array([self.m, self.m]),
            self.c_bar,
            self.nu,
            self.Nv,
            self.Bv,
        )

        self.model_time_dep = damage_time_dep_cyclic.PIAModel(
            solverparams.ParameterSet()
        )

    def test_definition(self):
        # k = self.s0 ** (-self.m)
        # kp = (2 * self.m + 1) * k

        actual = self.model_time_dep.calculate_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )

        # Summing up log probabilities over nelem and taking the value of one
        R_PIA = np.exp(np.sum(actual))
        print("Time dep Reliability PIA = ", R_PIA)

        # Evaluating Probability of Failure
        Pf_PIA = 1 - R_PIA
        print("Time dep Probability of failure PIA = ", Pf_PIA)

        with open("PIA_60K.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_PIA}")


class TestWNTSAModel(unittest.TestCase):
    def setUp(self):
        data = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_60k.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(1, 49)),
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
        self.nf = 0
        self.period = 0.01
        self.time = np.linspace(0, self.period, self.stress.shape[0])

        # Material properties
        self.m = 7.65
        self.s0 = 74.79 * ((1000) ** (3 / self.m))  # in mm  74.79 in m
        self.c_bar = 0.82
        self.nu = 0.219
        self.Nv = 30
        self.Bv = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            np.array([0, 1000.0]),
            np.array([self.m, self.m]),
            self.c_bar,
            self.nu,
            self.Nv,
            self.Bv,
        )

        self.model_time_dep = damage_time_dep_cyclic.WNTSAModel(
            solverparams.ParameterSet()
        )

    def test_definition(self):
        # k = self.s0 ** (-self.m)
        # kp = (2 * self.m + 1) * k

        actual = self.model_time_dep.calculate_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )

        # Summing up log probabilities over nelem and taking the value of one
        R_WNTSA = np.exp(np.sum(actual))
        print("Time dep Reliability WNTSA = ", R_WNTSA)

        # Evaluating Probability of Failure
        Pf_WNTSA = 1 - R_WNTSA
        print("Time dep Probability of failure WNTSA = ", Pf_WNTSA)

        with open("WNTSA_60K.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_WNTSA}")


class TestMTSModelGriffithFlaw(unittest.TestCase):
    def setUp(self):
        data = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_60k.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(1, 49)),
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
        self.nf = 0
        self.period = 0.01
        self.time = np.linspace(0, self.period, self.stress.shape[0])

        # Material properties
        self.m = 7.65
        self.s0 = 74.79 * ((1000) ** (3 / self.m))  # in mm  74.79 in m
        self.c_bar = 0.82
        self.nu = 0.219
        self.Nv = 30
        self.Bv = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            np.array([0, 1000.0]),
            np.array([self.m, self.m]),
            self.c_bar,
            self.nu,
            self.Nv,
            self.Bv,
        )

        self.model_time_dep = damage_time_dep_cyclic.MTSModelGriffithFlaw(
            solverparams.ParameterSet()
        )

    def test_definition(self):
        actual = self.model_time_dep.calculate_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )

        # Summing up log probabilities over nelem and taking the value of one
        R_MTS_GF = np.exp(np.sum(actual, axis=1))[0]
        # R_MTS_GF = np.exp(np.sum(actual,axis=tuple(range(actual.ndim))[1:]))[-1]
        print("Time dep Reliability MTS GF = ", R_MTS_GF)

        # Evaluating Probability of Failure
        Pf_MTS_GF = 1 - R_MTS_GF
        print("Time dep Probability of failure MTS GF = ", Pf_MTS_GF)

        with open("MTS_GF_60K.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_MTS_GF}")


class TestMTSModelPennyShapedFlaw(unittest.TestCase):
    def setUp(self):
        data = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_60k.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(1, 49)),
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
        self.nf = 100000
        self.period = 0.01
        self.time = np.linspace(0, self.period, self.stress.shape[0])

        # Material properties
        self.m = 7.65
        self.s0 = 74.79 * ((1000) ** (3 / self.m))  # in mm  74.79 in m
        self.c_bar = 0.82
        self.nu = 0.219
        self.Nv = 30
        self.Bv = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            np.array([0, 1000.0]),
            np.array([self.m, self.m]),
            self.c_bar,
            self.nu,
            self.Nv,
            self.Bv,
        )

        self.model_time_dep = damage_time_dep_cyclic.MTSModelPennyShapedFlaw(
            solverparams.ParameterSet()
        )

    def test_definition(self):
        actual = self.model_time_dep.calculate_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )

        # Summing up log probabilities over nelem and taking the value of one
        R_MTS_PSF = np.exp(np.sum(actual, axis=1))[0]
        # R_MTS_PSF = np.exp(np.sum(actual,axis=tuple(range(actual.ndim))[1:]))[-1]
        print("Time dep Reliability MTS_PSF = ", R_MTS_PSF)

        # Evaluating Probability of Failure
        Pf_MTS_PSF = 1 - R_MTS_PSF
        print("Time dep Probability of failure MTS_PSF = ", Pf_MTS_PSF)

        with open("MTS_PSF_60K.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_MTS_PSF}")


class TestCSEModelGriffithFlaw(unittest.TestCase):
    def setUp(self):
        data = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_60k.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(1, 49)),
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
        self.nf = 0
        self.period = 0.01
        self.time = np.linspace(0, self.period, self.stress.shape[0])

        # Material properties
        self.m = 7.65
        self.s0 = 74.79 * ((1000) ** (3 / self.m))  # in mm  74.79 in m
        self.c_bar = 0.82
        self.nu = 0.219
        self.Nv = 30
        self.Bv = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            np.array([0, 1000.0]),
            np.array([self.m, self.m]),
            self.c_bar,
            self.nu,
            self.Nv,
            self.Bv,
        )

        self.model_time_dep = damage_time_dep_cyclic.CSEModelGriffithFlaw(
            solverparams.ParameterSet()
        )

    def test_definition(self):
        actual = self.model_time_dep.calculate_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )

        # Summing up log probabilities over nelem and taking the value of one
        R_CSE_GF = np.exp(np.sum(actual, axis=1))[0]
        # R_CSE_GF = np.exp(np.sum(actual,axis=tuple(range(actual.ndim))[1:]))[-1]
        print("Time dep Reliability CSE GF = ", R_CSE_GF)

        # Evaluating Probability of Failure
        Pf_CSE_GF = 1 - R_CSE_GF
        print("Time dep Probability of failure CSE GF = ", Pf_CSE_GF)

        with open("CSE_GF_60K.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_CSE_GF}")


class TestCSEModelPennyShapedFlaw(unittest.TestCase):
    def setUp(self):
        data = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_60k.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(1, 49)),
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
        self.nf = 0
        self.period = 0.01
        self.time = np.linspace(0, self.period, self.stress.shape[0])

        # Material properties
        self.m = 7.65
        self.s0 = 74.79 * ((1000) ** (3 / self.m))  # in mm  74.79 in m
        self.c_bar = 0.82
        self.nu = 0.219
        self.Nv = 30
        self.Bv = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            np.array([0, 1000.0]),
            np.array([self.m, self.m]),
            self.c_bar,
            self.nu,
            self.Nv,
            self.Bv,
        )

        self.model_time_dep = damage_time_dep_cyclic.CSEModelPennyShapedFlaw(
            solverparams.ParameterSet()
        )

    def test_definition(self):
        actual = self.model_time_dep.calculate_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )

        # Summing up log probabilities over nelem and taking the value of one
        R_CSE_PSF = np.exp(np.sum(actual, axis=1))[0]
        # R_CSE_PSF = np.exp(np.sum(actual,axis=tuple(range(actual.ndim))[1:]))[-1]
        print("Time dep Reliability CSE_PSF = ", R_CSE_PSF)

        # Evaluating Probability of Failure
        Pf_CSE_PSF = 1 - R_CSE_PSF
        print("Time dep Probability of failure CSE_PSF = ", Pf_CSE_PSF)

        with open("CSE_PSF_60K.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_CSE_PSF}")


class TestSMMModelGriffithFlaw(unittest.TestCase):
    def setUp(self):
        data = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_60k.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(1, 49)),
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
        self.time = np.linspace(0, self.period, self.stress.shape[0])

        # Material properties
        self.m = 7.65
        self.s0 = 74.79 * ((1000) ** (3 / self.m))  # in mm  74.79 in m
        self.c_bar = 0.82
        self.nu = 0.219
        self.Nv = 30
        self.Bv = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            np.array([0, 1000.0]),
            np.array([self.m, self.m]),
            self.c_bar,
            self.nu,
            self.Nv,
            self.Bv,
        )

        self.model_time_dep = damage_time_dep_cyclic.SMMModelGriffithFlaw(
            solverparams.ParameterSet()
        )

    def test_definition(self):
        actual = self.model_time_dep.calculate_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )

        # Summing up log probabilities over nelem and taking the value of one
        R_SMM_GF = np.exp(np.sum(actual, axis=1))[0]
        print("Time dep Reliability SMM_GF = ", R_SMM_GF)

        # Evaluating Probability of Failure
        Pf_SMM_GF = 1 - R_SMM_GF
        print("Time dep Probability of failure SMM_GF = ", Pf_SMM_GF)

        with open("SMM_GF_60K_80K.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_SMM_GF}")


class TestSMMModelPennyShapedFlaw(unittest.TestCase):
    def setUp(self):
        data = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_60k.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(1, 49)),
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
        self.time = np.linspace(0, self.period, self.stress.shape[0])

        # Material properties
        self.m = 7.65
        self.s0 = 74.79 * ((1000) ** (3 / self.m))  # in mm  74.79 in m
        self.c_bar = 0.82
        self.nu = 0.219
        self.Nv = 30
        self.Bv = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            np.array([0, 1000.0]),
            np.array([self.m, self.m]),
            self.c_bar,
            self.nu,
            self.Nv,
            self.Bv,
        )
        self.model_time_dep = damage_time_dep_cyclic.SMMModelPennyShapedFlaw(
            solverparams.ParameterSet()
        )

    def test_definition(self):
        # k = self.s0 ** (-self.m)
        # kp = (2.99) * k

        actual = self.model_time_dep.calculate_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )
        # print("actual shape =", actual.shape)

        # Summing up log probabilities over nelem and taking the value of one
        R_SMM_PSF = np.exp(np.sum(actual, axis=1))[0]
        print("Time dep Reliability SMM_PSF = ", R_SMM_PSF)

        # Evaluating Probability of Failure
        Pf_SMM_PSF = 1 - R_SMM_PSF
        print("Time dep Probability of failure SMM_PSF = ", Pf_SMM_PSF)

        with open("SMM_PSF_60K.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_SMM_PSF}")
