import unittest

import numpy as np

from srlife import materials, damage, solverparams


class TestPIAModel(unittest.TestCase):
    def setUp(self):
        # Case 1: Single stress tensor over 1 time step
        # self.stress = np.array([[100.0,25.0,50.0,0,0,0]])
        # self.temperatures = np.array([1.0])
        # self.volumes = np.array([0.1])

        # Case 2: MUltiple stress tensors and 2 time steps
        self.stress = np.array(
            [
                [[100.0, 25.0, 50.0, 0, 0, 0], [-1.5, -25.0, -5.6, 0, 0, 0]],
                [[15.0, 6.0, 20.0, 0, 0, 0], [-15, 6, -20, 0, 0, 0]],
            ]
        )
        self.temperatures = np.array([[1.0, 3.0], [100.0, 10.0]])
        self.volumes = np.array([[0.1, 0.1], [0.1, 0.1]])

        # Case 3: Testing for an entirely random stress tensor over 3 time steps
        # self.stress = np.array([[
        # [14.0,-17.0,4.3,105,2,15.5],[15.0,25.0,5.0,0,0,0]],[[-1.5,-25.0,-5.6,17.1,-6.6,-301],
        # [54.0,-7.0,0.3,10,20,15.5]],[[-10.0,25.0,50.0,0,0,0],
        # [-1.0,-205.0,-56.0,-11.7,-0.6,-30]]])
        # self.temperatures = np.array([[1.0,3.0],[15.0,100.0],[10.0,11.0]])
        # self.volumes = np.array([[0.1,0.1],[0.1,0.1],[0.1,0.1]])

        self.s0 = 70
        self.m = 3.5
        self.c_bar = 0.8
        self.nu = 0.25

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            self.m,
            self.c_bar,
            self.nu,
        )

        self.model = damage.PIAModel(solverparams.ParameterSet())

    def test_definition(self):
        k = self.s0 ** (-self.m)

        # mod_stress = self.stress
        # mod_stress[mod_stress<0] = 0

        # Hand - calculated
        # Case 1
        # self.p_stress = np.array([[100,25,50]]) # calculated from mathematica

        # Case 2
        self.p_stress = np.array(
            [[[25, 50, 100], [0, 0, 0]], [[6, 15, 20], [0, 0, 6]]]
        )  # calculated from mathematica

        # Case 3
        # self.p_stress = np.array([[[0,13.3417,69.8812],[5,15,25]],[[0,0,200.577],
        # [0,0,59.7854]],[[0,25,50],[0,0,1.1857]]]) # calculated from mathematica

        should = -k * np.sum(self.p_stress**self.m, axis=-1) * self.volumes

        actual = self.model.calculate_element_log_reliability(
            self.stress, self.temperatures, self.volumes, self.material
        )

        # Evaluating Reliability
        R_PIA = np.exp(np.sum(actual))
        print("Reliability PIA = ", R_PIA)

        # Evaluating Probability of Failure
        Pf_PIA = 1 - np.exp(np.sum(actual))
        print("Probability of failure PIA = ", Pf_PIA)

        self.assertTrue(np.allclose(should, actual))


class TestWNTSAModel(unittest.TestCase):
    def setUp(self):

        # Case 1: Single stress tensor over 1 time step
        # self.stress = np.array([[100.0,25.0,50.0,0,0,0]])
        # self.temperatures = np.array([1.0])
        # self.volumes = np.array([0.1])

        # Case 2: Testing for multiple (principal) stress tensors over 2 time steps
        self.stress = np.array(
            [
                [[100.0, 25.0, 50.0, 0, 0, 0], [-1.5, -25.0, -5.6, 0, 0, 0]],
                [[15.0, 6.0, 20.0, 0, 0, 0], [-15, 6, -20, 0, 0, 0]],
            ]
        )
        self.temperatures = np.array([[1.0, 3.0], [100.0, 10.0]])
        self.volumes = np.array([[0.1, 0.1], [0.1, 0.1]])

        # Case 3: Testing for an entirely random stress tensor over 3 time steps
        # self.stress = np.array([[[14.0,-17.0,4.3,105,2,15.5],[15.0,25.0,5.0,0,0,0]],
        # [[-1.5,-25.0,-5.6,17.1,-6.6,-301],[54.0,-7.0,0.3,10,20,15.5]],
        # [[-10.0,25.0,50.0,0,0,0],[-1.0,-205.0,-56.0,-11.7,-0.6,-30]]])
        # self.temperatures = np.array([[1.0,3.0],[15.0,100.0],[10.0,11.0]])
        # self.volumes = np.array([[0.1,0.1],[0.1,0.1],[0.1,0.1]])

        self.s0 = 70
        self.m = 3.5
        self.c_bar = 0.8
        self.nu = 0.25

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            self.m,
            self.c_bar,
            self.nu,
        )

        self.model = damage.WNTSAModel(solverparams.ParameterSet())

    def test_definition(self):
        k = self.s0 ** (-self.m)
        kp = (2 * self.m + 1) * k

        # Hand-calculated
        # Case 1
        # self.avg_stress = np.array([[65.9504]]) # calculated from mathematica

        # Case 2
        self.avg_stress = np.array(
            [[65.9504, 0], [14.7457, 2.18744]]
        )  # calculated from mathematica

        # Case 3
        # self.avg_stress = np.array([[35.4182, 16.9277], [97.8709, 31.9819],
        # [30.4218, 0.229781]]) # calculated from mathematica

        should = -kp * (self.avg_stress**self.m) * self.volumes

        actual = self.model.calculate_element_log_reliability(
            self.stress, self.temperatures, self.volumes, self.material
        )

        # Evaluating Reliability
        R_weibull = np.exp(np.sum(actual))
        print("Reliability weibull = ", R_weibull)

        # Evaluating Probability of Failure
        Pf_weibull = 1 - np.exp(np.sum(actual))
        print("Probability of failure weibull = ", Pf_weibull)

        self.assertTrue(np.allclose(should, actual))


class TestMTSModelGriffithFlaw(unittest.TestCase):
    def setUp(self):

        # Case 1: Single stress tensor over 1 time step
        # self.stress = np.array([[100.0,25.0,50.0,0,0,0]])
        # self.temperatures = np.array([1.0])
        # self.volumes = np.array([0.1])

        # Case 2: Testing for multiple (principal) stress tensors over 2 time steps
        self.stress = np.array(
            [
                [[100.0, 25.0, 50.0, 0, 0, 0], [-1.5, -25.0, -5.6, 0, 0, 0]],
                [[15.0, 6.0, 20.0, 0, 0, 0], [-15, 6, -20, 0, 0, 0]],
            ]
        )
        self.temperatures = np.array([[1.0, 3.0], [100.0, 10.0]])
        self.volumes = np.array([[0.1, 0.1], [0.1, 0.1]])

        # Case 3: Testing for an entirely random stress tensor over 3 time steps
        # self.stress = np.array([[[14.0,-17.0,4.3,105,2,15.5],[15.0,25.0,5.0,0,0,0]],
        # [[-1.5,-25.0,-5.6,17.1,-6.6,-301],[54.0,-7.0,0.3,10,20,15.5]],[[-10.0,25.0,50.0,0,0,0],
        # [-1.0,-205.0,-56.0,-11.7,-0.6,-30]]])
        # self.temperatures = np.array([[1.0,3.0],[15.0,100.0],[10.0,11.0]])
        # self.volumes = np.array([[0.1,0.1],[0.1,0.1],[0.1,0.1]])

        self.s0 = 70
        self.m = 3.5
        self.c_bar = 0.8
        self.nu = 0.25

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            self.m,
            self.c_bar,
            self.nu,
        )

        self.model = damage.MTSModelGriffithFlaw(solverparams.ParameterSet())

    def test_definition(self):
        k = self.s0 ** (-self.m)
        kp = (2 * self.m + 1) * k

        # Hand - calculated
        # Case 1
        # self.eq_stress = np.array([[77.7895]]) # calculated from mathematica

        # Case 2
        self.eq_stress = np.array(
            [[77.7895, 1.89075], [17.2038, 4.2574]]
        )  # calculated from mathematica

        # Case 3
        # self.eq_stress = np.array([[47.430277, 19.946982],[132.18159, 40.157049],
        # [37.012046, 18.300173]]) # calculated from mathematica

        should = -(2 * kp / np.pi) * (self.eq_stress**self.m) * self.volumes

        actual = self.model.calculate_element_log_reliability(
            self.stress, self.temperatures, self.volumes, self.material
        )

        # Evaluating Reliability
        R_MTS_GF = np.exp(np.sum(actual))
        print("Reliability MTS GF = ", R_MTS_GF)

        # Evaluating Probability of Failure
        Pf_MTS_GF = 1 - np.exp(np.sum(actual))
        print("Probability of failure MTS GF = ", Pf_MTS_GF)

        self.assertTrue(np.allclose(should, actual))


class TestMTSModelPennyShapedFlaw(unittest.TestCase):
    def setUp(self):

        # Case 1: Single stress tensor over 1 time step
        # self.stress = np.array([[100.0,25.0,50.0,0,0,0]])
        # self.temperatures = np.array([1.0])
        # self.volumes = np.array([0.1])

        # Case 2: Testing for multiple (principal) stress tensors over 2 time steps
        self.stress = np.array(
            [
                [[100.0, 25.0, 50.0, 0, 0, 0], [-1.5, -25.0, -5.6, 0, 0, 0]],
                [[15.0, 6.0, 20.0, 0, 0, 0], [-15, 6, -20, 0, 0, 0]],
            ]
        )
        self.temperatures = np.array([[1.0, 3.0], [100.0, 10.0]])
        self.volumes = np.array([[0.1, 0.1], [0.1, 0.1]])

        # Case 3: Testing for an entirely random stress tensor over 3 time steps
        # self.stress = np.array([[[14.0,-17.0,4.3,105,2,15.5],[15.0,25.0,5.0,0,0,0]],
        # [[-1.5,-25.0,-5.6,17.1,-6.6,-301],[54.0,-7.0,0.3,10,20,15.5]],[[-10.0,25.0,50.0,0,0,0],
        # [-1.0,-205.0,-56.0,-11.7,-0.6,-30]]])
        # self.temperatures = np.array([[1.0,3.0],[15.0,100.0],[10.0,11.0]])
        # self.volumes = np.array([[0.1,0.1],[0.1,0.1],[0.1,0.1]])

        self.s0 = 70
        self.m = 3.5
        self.c_bar = 0.8
        self.nu = 0.25

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            self.m,
            self.c_bar,
            self.nu,
        )

        self.model = damage.MTSModelPennyShapedFlaw(solverparams.ParameterSet())

    def test_definition(self):
        k = self.s0 ** (-self.m)
        kp = (2 * self.m + 1) * k

        # Hand - calculated
        # Case 1
        # self.eq_stress = np.array([[78.4648]]) # calculated from mathematica

        # Case 2
        self.eq_stress = np.array(
            [[78.464821, 2.3585169], [17.301933, 4.7971274]]
        )  # calculated from mathematica

        # Case 3
        # self.eq_stress = np.array([[49.595931, 20.115328],[138.39622, 41.172063],
        # [37.656223, 22.5172]]) # calculated from mathematica

        should = -(2 * kp / np.pi) * (self.eq_stress**self.m) * self.volumes

        actual = self.model.calculate_element_log_reliability(
            self.stress, self.temperatures, self.volumes, self.material
        )

        # Evaluating Reliability
        R_MTS_PSF = np.exp(np.sum(actual))
        print("Reliability MTS PSF = ", R_MTS_PSF)

        # Evaluating Probability of Failure
        Pf_MTS_PSF = 1 - np.exp(np.sum(actual))
        print("Probability of failure MTS PSF = ", Pf_MTS_PSF)

        self.assertTrue(np.allclose(should, actual))


class TestCSEModelGriffithFlaw(unittest.TestCase):
    def setUp(self):

        # Case 1: Single stress tensor over 1 time step
        # self.stress = np.array([[100.0, 25.0, 50.0, 0, 0, 0]])
        # self.temperatures = np.array([1.0])
        # self.volumes = np.array([0.1])

        # Case 2: Testing for multiple (principal) stress tensors over 2 time steps
        self.stress = np.array(
            [
                [[100.0, 25.0, 50.0, 0, 0, 0], [-1.5, -25.0, -5.6, 0, 0, 0]],
                [[15.0, 6.0, 20.0, 0, 0, 0], [-15, 6, -20, 0, 0, 0]],
            ]
        )
        self.temperatures = np.array([[1.0, 3.0], [100.0, 10.0]])
        self.volumes = np.array([[0.1, 0.1], [0.1, 0.1]])

        # Case 3: Testing for an entirely random stress tensor over 3 time steps
        # self.stress = np.array([[[14.0,-17.0,4.3,105,2,15.5],[15.0,25.0,5.0,0,0,0]],
        # [[-1.5,-25.0,-5.6,17.1,-6.6,-301],[54.0,-7.0,0.3,10,20,15.5]],[[-10.0,25.0,50.0,0,0,0],
        # [-1.0,-205.0,-56.0,-11.7,-0.6,-30]]])
        # self.temperatures = np.array([[1.0,3.0],[15.0,100.0],[10.0,11.0]])
        # self.volumes = np.array([[0.1,0.1],[0.1,0.1],[0.1,0.1]])

        self.s0 = 70
        self.m = 3.5
        self.c_bar = 0.8
        self.nu = 0.25

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            self.m,
            self.c_bar,
            self.nu,
        )

        self.model = damage.CSEModelGriffithFlaw(solverparams.ParameterSet())

    def test_definition(self):
        k = self.s0 ** (-self.m)
        kp = (2 * self.m + 1) * k

        # Hand - calculated
        # Case 1
        # self.eq_stress = np.array([[80.168972]]) # calculated from mathematica

        # Case 2
        self.eq_stress = np.array(
            [[80.168972, 18.764062], [17.54572, 17.504612]]
        )  # calculated from mathematica

        # Case 3
        # self.eq_stress = np.array([[74.122529, 20.54913],[207.22137, 44.87708],
        # [39.892764, 156.49114]]) # calculated from mathematica

        should = -(2 * kp / np.pi) * (self.eq_stress**self.m) * self.volumes

        actual = self.model.calculate_element_log_reliability(
            self.stress, self.temperatures, self.volumes, self.material
        )

        # Evaluating Reliability
        R_CSE_GF = np.exp(np.sum(actual))
        print("Reliability MTS GF = ", R_CSE_GF)

        # Evaluating Probability of Failure
        Pf_CSE_GF = 1 - np.exp(np.sum(actual))
        print("Probability of failure CSE GF = ", Pf_CSE_GF)

        self.assertTrue(np.allclose(should, actual))


class TestCSEModelPennyShapedFlaw(unittest.TestCase):
    def setUp(self):

        # Case 1: Single stress tensor over 1 time step
        # self.stress = np.array([[100.0,25.0,50.0,0,0,0]])
        # self.temperatures = np.array([1.0])
        # self.volumes = np.array([0.1])

        # Case 2: Testing for multiple (principal) stress tensors over 2 time steps
        self.stress = np.array(
            [
                [[100.0, 25.0, 50.0, 0, 0, 0], [-1.5, -25.0, -5.6, 0, 0, 0]],
                [[15.0, 6.0, 20.0, 0, 0, 0], [-15, 6, -20, 0, 0, 0]],
            ]
        )
        self.temperatures = np.array([[1.0, 3.0], [100.0, 10.0]])
        self.volumes = np.array([[0.1, 0.1], [0.1, 0.1]])

        # Case 3: Testing for an entirely random stress tensor over 3 time steps
        # self.stress = np.array([[[14.0,-17.0,4.3,105,2,15.5],[15.0,25.0,5.0,0,0,0]],
        # [[-1.5,-25.0,-5.6,17.1,-6.6,-301],[54.0,-7.0,0.3,10,20,15.5]],[[-10.0,25.0,50.0,0,0,0],
        # [-1.0,-205.0,-56.0,-11.7,-0.6,-30]]])
        # self.temperatures = np.array([[1.0,3.0],[15.0,100.0],[10.0,11.0]])
        # self.volumes = np.array([[0.1,0.1],[0.1,0.1],[0.1,0.1]])

        self.s0 = 70
        self.m = 3.5
        self.c_bar = 0.8
        self.nu = 0.25

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            self.m,
            self.c_bar,
            self.nu,
        )

        self.model = damage.CSEModelPennyShapedFlaw(solverparams.ParameterSet())

    def test_definition(self):
        k = self.s0 ** (-self.m)
        kp = (2 * self.m + 1) * k

        # Hand - calculated
        # Case 1
        # self.eq_stress = np.array([[81.584298]]) # calculated from mathematica

        # Case 2
        self.eq_stress = np.array(
            [[81.584298, 19.415373], [17.752073, 18.267442]]
        )  # calculated from mathematica

        # Case 3
        # self.eq_stress = np.array([[80.502237, 20.908868],[224.87259, 47.426973],
        # [41.532897, 162.1895]]) # calculated from mathematica

        should = -(2 * kp / np.pi) * (self.eq_stress**self.m) * self.volumes

        actual = self.model.calculate_element_log_reliability(
            self.stress, self.temperatures, self.volumes, self.material
        )

        # Evaluating Reliability
        R_CSE_PSF = np.exp(np.sum(actual))
        print("Reliability MTS GF = ", R_CSE_PSF)

        # Evaluating Probability of Failure
        Pf_CSE_PSF = 1 - np.exp(np.sum(actual))
        print("Probability of failure CSE PSF = ", Pf_CSE_PSF)

        self.assertTrue(np.allclose(should, actual))


class TestSMMModelGriffithFlaw(unittest.TestCase):
    def setUp(self):

        # Case 1: Single stress tensor over 1 time step
        # self.stress = np.array([[100.0,25.0,50.0,0,0,0]])
        # self.temperatures = np.array([1.0])
        # self.volumes = np.array([0.1])

        # Case 2: Testing for multiple (principal) stress tensors over 2 time steps
        self.stress = np.array(
            [
                [[100.0, 25.0, 50.0, 0, 0, 0], [-1.5, -25.0, -5.6, 0, 0, 0]],
                [[15.0, 6.0, 20.0, 0, 0, 0], [-15, 6, -20, 0, 0, 0]],
            ]
        )
        self.temperatures = np.array([[1.0, 3.0], [100.0, 10.0]])
        self.volumes = np.array([[0.1, 0.1], [0.1, 0.1]])

        # Case 3: Testing for an entirely random stress tensor over 3 time steps
        # self.stress = np.array([[[14.0,-17.0,4.3,105,2,15.5],[15.0,25.0,5.0,0,0,0]],
        # [[-1.5,-25.0,-5.6,17.1,-6.6,-301],[54.0,-7.0,0.3,10,20,15.5]],[[-10.0,25.0,50.0,0,0,0],
        # [-1.0,-205.0,-56.0,-11.7,-0.6,-30]]])
        # self.temperatures = np.array([[1.0,3.0],[15.0,100.0],[10.0,11.0]])
        # self.volumes = np.array([[0.1,0.1],[0.1,0.1],[0.1,0.1]])

        self.s0 = 70
        self.m = 3.5
        self.c_bar = 0.8
        self.nu = 0.25

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            self.m,
            self.c_bar,
            self.nu,
        )

        self.model = damage.SMMModelGriffithFlaw(solverparams.ParameterSet())

    def test_definition(self):
        k = self.s0 ** (-self.m)
        kp = (2 * self.m + 1) * k

        # Hand - calculated
        # Case 1
        # self.eq_stress = np.array([[88.366538]]) # calculated from mathematica

        # Case 2
        self.eq_stress = np.array(
            [[88.366538, 7.6976808], [18.79631, 11.020469]]
        )  # calculated from mathematica

        # Case 3
        # self.eq_stress = np.array([[79.857326, 22.614905],[223.55189, 55.103919],
        # [47.083578, 69.367347]]) # calculated from mathematica

        should = -(2 * kp / np.pi) * (self.eq_stress**self.m) * self.volumes

        actual = self.model.calculate_element_log_reliability(
            self.stress, self.temperatures, self.volumes, self.material
        )

        # Evaluating Reliability
        R_SMM_GF = np.exp(np.sum(actual))
        print("Reliability MTS GF = ", R_SMM_GF)

        # Evaluating Probability of Failure
        Pf_SMM_GF = 1 - np.exp(np.sum(actual))
        print("Probability of failure SMM GF = ", Pf_SMM_GF)

        self.assertTrue(np.allclose(should, actual))


class TestSMMModelPennyShapedFlaw(unittest.TestCase):
    def setUp(self):

        # Case 1: Single stress tensor over 1 time step
        # self.stress = np.array([[100.0,25.0,50.0,0,0,0]])
        # self.temperatures = np.array([1.0])
        # self.volumes = np.array([0.1])

        # Case 2: Testing for multiple (principal) stress tensors over 2 time steps
        self.stress = np.array(
            [
                [[100.0, 25.0, 50.0, 0, 0, 0], [-1.5, -25.0, -5.6, 0, 0, 0]],
                [[15.0, 6.0, 20.0, 0, 0, 0], [-15, 6, -20, 0, 0, 0]],
            ]
        )
        self.temperatures = np.array([[1.0, 3.0], [100.0, 10.0]])
        self.volumes = np.array([[0.1, 0.1], [0.1, 0.1]])

        # Case 3: Testing for an entirely random stress tensor over 3 time steps
        # self.stress = np.array([[[14.0,-17.0,4.3,105,2,15.5],[15.0,25.0,5.0,0,0,0]],
        # [[-1.5,-25.0,-5.6,17.1,-6.6,-301],[54.0,-7.0,0.3,10,20,15.5]],[[-10.0,25.0,50.0,0,0,0],
        # [-1.0,-205.0,-56.0,-11.7,-0.6,-30]]])
        # self.temperatures = np.array([[1.0,3.0],[15.0,100.0],[10.0,11.0]])
        # self.volumes = np.array([[0.1,0.1],[0.1,0.1],[0.1,0.1]])

        self.s0 = 70
        self.m = 3.5
        self.c_bar = 0.8
        self.nu = 0.25

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.s0, self.s0]),
            self.m,
            self.c_bar,
            self.nu,
        )

        self.model = damage.SMMModelPennyShapedFlaw(solverparams.ParameterSet())

    def test_definition(self):
        k = self.s0 ** (-self.m)
        kp = (2 * self.m + 1) * k

        # Hand - calculated
        # Case 1
        # self.eq_stress = np.array([[91.801598]]) # calculated from mathematica

        # Case 2
        self.eq_stress = np.array(
            [[91.801598, 9.2479246], [19.335503, 12.793856]]
        )  # calculated from mathematica

        # Case 3
        # self.eq_stress = np.array([[89.324462, 23.492053],[249.97531, 59.574085],
        # [50.276554, 82.8113]]) # calculated from mathematica

        should = -(2 * kp / np.pi) * (self.eq_stress**self.m) * self.volumes

        actual = self.model.calculate_element_log_reliability(
            self.stress, self.temperatures, self.volumes, self.material
        )

        # Evaluating Reliability
        R_SMM_PSF = np.exp(np.sum(actual))
        print("Reliability MTS GF = ", R_SMM_PSF)

        # Evaluating Probability of Failure
        Pf_SMM_PSF = 1 - np.exp(np.sum(actual))
        print("Probability of failure SMM PSF = ", Pf_SMM_PSF)

        self.assertTrue(np.allclose(should, actual))
