import unittest

import numpy as np

# import csv
import matplotlib.pyplot as plt

import os.path

from srlife import (
    materials,
    damage,
    damage_time_dep_cyclic,
    solverparams,
)


class TestPIAModel(unittest.TestCase):
    def setUp(self):
        # 60,000 to 70,000 rpm
        # self.stress = np.array(
        #     [
        #         [
        #             [44.08801004196201, 160.56004134571623, 0, 0, 0, 0],
        #             [70.56728995109746, 125.49389020226235, 0, 0, 0, 0],
        #             [73.64178822023885, 110.33416575148763, 0, 0, 0, 0],
        #             [68.42220922221401, 99.97016362056426, 0, 0, 0, 0],
        #             [58.56207074702157, 90.74836601949357, 0, 0, 0, 0],
        #             [45.29913637906247, 81.43100936387464, 0, 0, 0, 0],
        #             [29.14665988998162, 71.50483988206251, 0, 0, 0, 0],
        #             [10.348573211586011, 60.72592564225028, 0, 0, 0, 0],
        #         ],
        #         [
        #             [51.13165661671334, 186.21164558438093, 0, 0, 0, 0],
        #             [81.84135402612488, 145.5432099387185, 0, 0, 0, 0],
        #             [85.40704432643088, 127.96151767628152, 0, 0, 0, 0],
        #             [79.35356809203516, 115.94172822266626, 0, 0, 0, 0],
        #             [67.9181412213978, 105.24662567941266, 0, 0, 0, 0],
        #             [52.53627651062867, 94.44069725041082, 0, 0, 0, 0],
        #             [33.80322685465324, 82.92869004073522, 0, 0, 0, 0],
        #             [12.001895558999173, 70.42770074485833, 0, 0, 0, 0],
        #         ],
        #         [
        #             [44.08801004196201, 160.56004134571623, 0, 0, 0, 0],
        #             [70.56728995109746, 125.49389020226235, 0, 0, 0, 0],
        #             [73.64178822023885, 110.33416575148763, 0, 0, 0, 0],
        #             [68.42220922221401, 99.97016362056426, 0, 0, 0, 0],
        #             [58.56207074702157, 90.74836601949357, 0, 0, 0, 0],
        #             [45.29913637906247, 81.43100936387464, 0, 0, 0, 0],
        #             [29.14665988998162, 71.50483988206251, 0, 0, 0, 0],
        #             [10.348573211586011, 60.72592564225028, 0, 0, 0, 0],
        #         ],
        #         [
        #             [37.5661150653404, 136.80855593954516, 0, 0, 0, 0],
        #             [60.12834173347949, 106.92970526109927, 0, 0, 0, 0],
        #             [62.74803256635736, 94.01254359890069, 0, 0, 0, 0],
        #             [58.30058063904624, 85.18167787787723, 0, 0, 0, 0],
        #             [49.89904253000652, 77.32405151956847, 0, 0, 0, 0],
        #             [38.59808070168636, 69.3850020615263, 0, 0, 0, 0],
        #             [24.835023811581966, 60.92720084625443, 0, 0, 0, 0],
        #             [8.81771918620344, 51.742800547242815, 0, 0, 0, 0],
        #         ],
        #         [
        #             [44.08801004196201, 160.56004134571623, 0, 0, 0, 0],
        #             [70.56728995109746, 125.49389020226235, 0, 0, 0, 0],
        #             [73.64178822023885, 110.33416575148763, 0, 0, 0, 0],
        #             [68.42220922221401, 99.97016362056426, 0, 0, 0, 0],
        #             [58.56207074702157, 90.74836601949357, 0, 0, 0, 0],
        #             [45.29913637906247, 81.43100936387464, 0, 0, 0, 0],
        #             [29.14665988998162, 71.50483988206251, 0, 0, 0, 0],
        #             [10.348573211586011, 60.72592564225028, 0, 0, 0, 0],
        #         ],
        #     ]
        # )

        # 60,000 to 80,000 rpm
        # self.stress = np.array(
        #     [
        #         [
        #             [51.131656616713364, 186.21164558438102, 0, 0, 0, 0],
        #             [81.8413540261249, 145.54320993871852, 0, 0, 0, 0],
        #             [85.40704432643089, 127.96151767628156, 0, 0, 0, 0],
        #             [79.3535680920352, 115.9417282226663, 0, 0, 0, 0],
        #             [67.91814122139785, 105.2466256794127, 0, 0, 0, 0],
        #             [52.53627651062872, 94.44069725041086, 0, 0, 0, 0],
        #             [33.803226854653296, 82.92869004073528, 0, 0, 0, 0],
        #             [12.001895558999221, 70.42770074485836, 0, 0, 0, 0],
        #         ],
        #         [
        #             [66.78420456060523, 243.21521055919152, 0, 0, 0, 0],
        #             [106.89482974840807, 190.09725379750992, 0, 0, 0, 0],
        #             [111.55205789574653, 167.1334108424902, 0, 0, 0, 0],
        #             [103.64547669163787, 151.43409400511518, 0, 0, 0, 0],
        #             [88.70940894223394, 137.46498047923293, 0, 0, 0, 0],
        #             [68.61881013633139, 123.35111477604686, 0, 0, 0, 0],
        #             [44.15115344281247, 108.31502372667464, 0, 0, 0, 0],
        #             [15.67594521991733, 91.98720097287624, 0, 0, 0, 0],
        #         ],
        #         [
        #             [51.131656616713364, 186.21164558438102, 0, 0, 0, 0],
        #             [81.8413540261249, 145.54320993871852, 0, 0, 0, 0],
        #             [85.40704432643089, 127.96151767628156, 0, 0, 0, 0],
        #             [79.3535680920352, 115.9417282226663, 0, 0, 0, 0],
        #             [67.91814122139785, 105.2466256794127, 0, 0, 0, 0],
        #             [52.53627651062872, 94.44069725041086, 0, 0, 0, 0],
        #             [33.803226854653296, 82.92869004073528, 0, 0, 0, 0],
        #             [12.001895558999221, 70.42770074485836, 0, 0, 0, 0],
        #         ],
        #         [
        #             [37.56611506534044, 136.80855593954522, 0, 0, 0, 0],
        #             [60.128341733479516, 106.92970526109931, 0, 0, 0, 0],
        #             [62.74803256635741, 94.01254359890075, 0, 0, 0, 0],
        #             [58.30058063904628, 85.18167787787728, 0, 0, 0, 0],
        #             [49.899042530006575, 77.32405151956851, 0, 0, 0, 0],
        #             [38.5980807016864, 69.38500206152634, 0, 0, 0, 0],
        #             [24.835023811582005, 60.92720084625447, 0, 0, 0, 0],
        #             [8.817719186203488, 51.74280054724287, 0, 0, 0, 0],
        #         ],
        #         [
        #             [51.13165661671336, 186.211645584381, 0, 0, 0, 0],
        #             [81.8413540261249, 145.54320993871852, 0, 0, 0, 0],
        #             [85.40704432643089, 127.96151767628156, 0, 0, 0, 0],
        #             [79.3535680920352, 115.94172822266628, 0, 0, 0, 0],
        #             [67.91814122139785, 105.24662567941269, 0, 0, 0, 0],
        #             [52.536276510628696, 94.44069725041085, 0, 0, 0, 0],
        #             [33.80322685465327, 82.92869004073526, 0, 0, 0, 0],
        #             [12.001895558999221, 70.42770074485836, 0, 0, 0, 0],
        #         ],
        #     ]
        # )

        # data = np.loadtxt("Spinning_disk_cyclic_60000_70000.txt")
        data = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_cyclic_60000_80000.txt",
            )
        )
        # data = np.loadtxt("Spinning_disk_static_60000.txt")

        self.stress = data.reshape(data.shape[0], 8, -1)

        vol_factor = 1
        self.volumes = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_volumes.txt",
            )
        )
        self.volumes = vol_factor * (np.tile(self.volumes, data.shape[0])).reshape(
            data.shape[0], 8
        )

        self.temperatures = np.ones((data.shape[0], 8))

        # Number of cycles to failure
        self.nf = 1

        # Material properties
        self.m = 7.65
        self.s0 = 74.79 * ((1000) ** (3 / self.m))  # in mm  74.79 in m
        self.c_bar = 0.82
        self.nu = 0.219

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            np.array([0, 1000.0]),
            np.array([self.m, self.m]),
            self.c_bar,
            self.nu,
        )

        self.model_time_dep = damage_time_dep_cyclic.PIAModel(
            solverparams.ParameterSet()
        )

    def test_definition(self):
        # k = self.s0 ** (-self.m)
        # kp = (2 * self.m + 1) * k

        actual = self.model_time_dep.calculate_element_log_reliability(
            self.stress, self.temperatures, self.volumes, self.nf, self.material
        )

        # Summing up log probabilities over nelem and taking the value of one
        R_PIA = np.exp(np.sum(actual, axis=1))[0]
        print("Time dep Reliability PIA = ", R_PIA)

        # Evaluating Probability of Failure
        Pf_PIA = 1 - R_PIA
        print("Time dep Probability of failure PIA = ", Pf_PIA)


class TestCSEModelGriffithFlaw(unittest.TestCase):
    def setUp(self):
        # data = np.loadtxt("Spinning_disk_cyclic_60000_70000.txt")
        data = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_cyclic_60000_80000.txt",
            )
        )
        # data = np.loadtxt("Spinning_disk_static_60000.txt")

        self.stress = data.reshape(data.shape[0], 8, -1)

        vol_factor = 1
        self.volumes = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_volumes.txt",
            )
        )
        self.volumes = vol_factor * (np.tile(self.volumes, data.shape[0])).reshape(
            data.shape[0], 8
        )

        self.temperatures = np.ones((data.shape[0], 8))

        # Number of cycles to failure
        self.nf = 1

        # Material properties
        self.m = 7.65
        self.s0 = 74.79 * ((1000) ** (3 / self.m))  # in mm  74.79 in m
        self.c_bar = 0.82
        self.nu = 0.219

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            np.array([0, 1000.0]),
            np.array([self.m, self.m]),
            self.c_bar,
            self.nu,
        )

        # self.model_time_indep = damage.CSEModelGriffithFlaw(solverparams.ParameterSet())
        self.model_time_dep = damage_time_dep_cyclic.CSEModelGriffithFlaw(
            solverparams.ParameterSet()
        )
        # self.model = damage.CSEModelGriffithFlaw(solverparams.ParameterSet())

    def test_definition(self):
        # k = self.s0 ** (-self.m)
        # kp = (self.m + 1) * k

        actual = self.model_time_dep.calculate_element_log_reliability(
            self.stress, self.temperatures, self.volumes, self.nf, self.material
        )

        # Summing up log probabilities over nelem and taking the value of one
        R_CSE_GF = np.exp(np.sum(actual, axis=1))[0]
        # R_CSE_GF = np.exp(np.sum(actual,axis=tuple(range(actual.ndim))[1:]))[-1]
        print("Time dep Reliability CSE GF = ", R_CSE_GF)

        # Evaluating Probability of Failure
        Pf_CSE_GF = 1 - R_CSE_GF
        print("Time dep Probability of failure CSE GF = ", Pf_CSE_GF)


class TestCSEModelPennyShapedFlaw(unittest.TestCase):
    def setUp(self):
        # data = np.loadtxt("Spinning_disk_cyclic_60000_70000.txt")
        data = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_cyclic_60000_80000.txt",
            )
        )
        # data = np.loadtxt("Spinning_disk_static_60000.txt")

        self.stress = data.reshape(data.shape[0], 8, -1)

        vol_factor = 1
        self.volumes = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_volumes.txt",
            )
        )
        self.volumes = vol_factor * (np.tile(self.volumes, data.shape[0])).reshape(
            data.shape[0], 8
        )

        self.temperatures = np.ones((data.shape[0], 8))

        # Number of cycles to failure
        self.nf = 1

        # Material properties
        self.m = 7.65
        self.s0 = 74.79 * ((1000) ** (3 / self.m))  # in mm  74.79 in m
        self.c_bar = 0.82
        self.nu = 0.219

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            np.array([0, 1000.0]),
            np.array([self.m, self.m]),
            self.c_bar,
            self.nu,
        )

        self.model_time_dep = damage_time_dep_cyclic.CSEModelPennyShapedFlaw(
            solverparams.ParameterSet()
        )

    def test_definition(self):
        # k = self.s0 ** (-self.m)
        # kp = (self.m + 1) * k

        actual = self.model_time_dep.calculate_element_log_reliability(
            self.stress, self.temperatures, self.volumes, self.nf, self.material
        )

        # Summing up log probabilities over nelem and taking the value of one
        R_CSE_PSF = np.exp(np.sum(actual, axis=1))[0]
        # R_CSE_PSF = np.exp(np.sum(actual,axis=tuple(range(actual.ndim))[1:]))[-1]
        print("Time dep Reliability CSE_PSF = ", R_CSE_PSF)

        # Evaluating Probability of Failure
        Pf_CSE_PSF = 1 - R_CSE_PSF
        print("Time dep Probability of failure CSE_PSF = ", Pf_CSE_PSF)


class TestSMMModelGriffithFlaw(unittest.TestCase):
    def setUp(self):
        # data = np.loadtxt("Spinning_disk_cyclic_60000_70000.txt")
        data = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_cyclic_60000_80000.txt",
            )
        )
        # data = np.loadtxt("Spinning_disk_static_60000.txt")

        self.stress = data.reshape(data.shape[0], 8, -1)

        vol_factor = 1
        self.volumes = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_volumes.txt",
            )
        )
        self.volumes = vol_factor * (np.tile(self.volumes, data.shape[0])).reshape(
            data.shape[0], 8
        )

        self.temperatures = np.ones((data.shape[0], 8))
        # Number of cycles to failure
        self.nf = 1

        # Material properties
        self.m = 7.65
        self.s0 = 74.79 * ((1000) ** (3 / self.m))  # in mm  74.79 in m
        self.c_bar = 0.82
        self.nu = 0.219

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            np.array([0, 1000.0]),
            np.array([self.m, self.m]),
            self.c_bar,
            self.nu,
        )

        self.model_time_dep = damage_time_dep_cyclic.SMMModelGriffithFlaw(
            solverparams.ParameterSet()
        )

    def test_definition(self):
        # k = self.s0 ** (-self.m)
        # kp = (2.99) * k

        actual = self.model_time_dep.calculate_element_log_reliability(
            self.stress, self.temperatures, self.volumes, self.nf, self.material
        )

        # Summing up log probabilities over nelem and taking the value of one
        R_SMM_GF = np.exp(np.sum(actual, axis=1))[0]

        # Evaluating Probability of Failure
        Pf_SMM_GF = 1 - R_SMM_GF
        print("Time dep Probability of failure SMM_GF = ", Pf_SMM_GF)


class TestSMMModelPennyShapedFlaw(unittest.TestCase):
    def setUp(self):
        # data = np.loadtxt("Spinning_disk_cyclic_60000_70000.txt")
        data = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_cyclic_60000_80000.txt",
            )
        )
        # data = np.loadtxt("Spinning_disk_static_60000.txt")

        self.stress = data.reshape(data.shape[0], 8, -1)

        vol_factor = 1
        self.volumes = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "Spinning_disk_volumes.txt",
            )
        )
        self.volumes = vol_factor * (np.tile(self.volumes, data.shape[0])).reshape(
            data.shape[0], 8
        )

        self.temperatures = np.ones((data.shape[0], 8))

        # Number of cycles to failure
        self.nf = 1

        # Material properties
        self.m = 7.65
        self.s0 = 74.79 * ((1000) ** (3 / self.m))  # in mm  74.79 in m
        self.c_bar = 0.82
        self.nu = 0.219

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            np.array([0, 1000.0]),
            np.array([self.m, self.m]),
            self.c_bar,
            self.nu,
        )

        self.model_time_dep = damage_time_dep_cyclic.SMMModelPennyShapedFlaw(
            solverparams.ParameterSet()
        )

    def test_definition(self):
        # k = self.s0 ** (-self.m)
        # kp = (2.99) * k

        actual = self.model_time_dep.calculate_element_log_reliability(
            self.stress, self.temperatures, self.volumes, self.nf, self.material
        )
        print("actual shape =", actual.shape)

        # Summing up log probabilities over nelem and taking the value of one
        R_SMM_PSF = np.exp(np.sum(actual, axis=1))[0]
        print("Time dep Reliability SMM_PSF = ", R_SMM_PSF)

        # Evaluating Probability of Failure
        Pf_SMM_PSF = 1 - R_SMM_PSF
        print("Time dep Probability of failure SMM_PSF = ", Pf_SMM_PSF)
