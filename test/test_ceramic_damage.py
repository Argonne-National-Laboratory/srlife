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
    self.stress = np.array([[
     [100.0,25.0,50.0,0,0,0],[-1.5,-25.0,-5.6,0,0,0]],
     [[15.0,6.0,20.0,0,0,0],[-15,6,-20,0,0,0]]])
    self.temperatures = np.array([[1.0,3.0],[100.0,10.0]])
    self.volumes = np.array([[0.1,0.1],[0.1,0.1]])

    # Case 3: Testing for an entirely random stress tensor over 3 time steps
    # self.stress = np.array([[
    #   [14.0,-17.0,4.3,105,2,15.5],[15.0,25.0,5.0,0,0,0]],[[-1.5,-25.0,-5.6,17.1,-6.6,-301],
    #   [54.0,-7.0,0.3,10,20,15.5]],[[-10.0,25.0,50.0,0,0,0],[-1.0,-205.0,-56.0,-11.7,-0.6,-30]]])
    # self.temperatures = np.array([[1.0,3.0],[15.0,100.0],[10.0,11.0]])
    # self.volumes = np.array([[0.1,0.1],[0.1,0.1],[0.1,0.1]])

    self.s0 = 70
    self.m = 3.5
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.PIAModel({})

  def test_definition(self):
    k = self.s0**(-self.m)

    # mod_stress = self.stress
    # mod_stress[mod_stress<0] = 0

    # Hand - calculated
    # Case 1
    # self.p_stress = np.array([[100,25,50]]) # calculated from mathematica

    # Case 2
    self.p_stress = np.array([[[25,50,100],[0,0,0]],[[6,15,20],[0,0,6]]]) # calculated from mathematica

    # Case 3
    # self.p_stress = np.array([[[0,13.3417,69.8812],[5,15,25]],[[0,0,200.577],[0,0,59.7854]],[[0,25,50],[0,0,1.1857]]]) # calculated from mathematica

    should = -k  * np.sum(self.p_stress**self.m, axis = -1) * self.volumes

    actual = self.model.calculate_element_log_reliability(self.stress,
        self.temperatures, self.volumes, self.material)

    # Evaluating Reliability
    R_PIA = np.exp(np.sum(actual))
    print("Reliability PIA = ",R_PIA)

    # Evaluating Probability of Failure
    Pf_PIA = 1 - np.exp(np.sum(actual))
    print("Probability of failure PIA = ",Pf_PIA)

    self.assertTrue(np.allclose(should, actual))

class TestWNTSAModel(unittest.TestCase):
  def setUp(self):

    # Case 1: Single stress tensor over 1 time step
    # self.stress = np.array([[100.0,25.0,50.0,0,0,0]])
    # self.temperatures = np.array([1.0])
    # self.volumes = np.array([0.1])

    # Case 2: Testing for multiple (principal) stress tensors over 2 time steps
    self.stress = np.array([[
     [100.0,25.0,50.0,0,0,0],[-1.5,-25.0,-5.6,0,0,0]],
     [[15.0,6.0,20.0,0,0,0],[-15,6,-20,0,0,0]]])
    self.temperatures = np.array([[1.0,3.0],[100.0,10.0]])
    self.volumes = np.array([[0.1,0.1],[0.1,0.1]])

    # Case 3: Testing for an entirely random stress tensor over 3 time steps
    # self.stress = np.array([[
    #   [14.0,-17.0,4.3,105,2,15.5],[15.0,25.0,5.0,0,0,0]],[[-1.5,-25.0,-5.6,17.1,-6.6,-301],
    #   [54.0,-7.0,0.3,10,20,15.5]],[[-10.0,25.0,50.0,0,0,0],[-1.0,-205.0,-56.0,-11.7,-0.6,-30]]])
    # self.temperatures = np.array([[1.0,3.0],[15.0,100.0],[10.0,11.0]])
    # self.volumes = np.array([[0.1,0.1],[0.1,0.1],[0.1,0.1]])

    self.s0 = 70
    self.m = 3.5
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.WNTSAModel(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    # Hand - calculated
    # Case 1
    # self.avg_stress = np.array([[65.9504]]) # calculated from mathematica

    # Case 2
    self.avg_stress = np.array([[65.9504,0],[14.7457,2.18744]]) # calculated from mathematica

    # Case 3
    # self.avg_stress = np.array([[35.4182, 16.9277], [97.8709, 31.9819], [30.4218, 0.229781]]) # calculated from mathematica

    should = -kp * (self.avg_stress**self.m) * self.volumes

    actual = self.model.calculate_element_log_reliability(self.stress, self.temperatures, self.volumes, self.material)

    # Evaluating Reliability
    R_weibull = np.exp(np.sum(actual))
    print("Reliability weibull = ",R_weibull)

    # Evaluating Probability of Failure
    Pf_weibull = 1 - np.exp(np.sum(actual))
    print("Probability of failure weibull = ",Pf_weibull)

    self.assertTrue(np.allclose(should, actual))

class TestMTSModel_GF(unittest.TestCase):
  def setUp(self):

    # Case 1: Single stress tensor over 1 time step
    # self.stress = np.array([[100.0,25.0,50.0,0,0,0]])
    # self.temperatures = np.array([1.0])
    # self.volumes = np.array([0.1])

    # Case 2: Testing for multiple (principal) stress tensors over 2 time steps
    self.stress = np.array([[
     [100.0,25.0,50.0,0,0,0],[-1.5,-25.0,-5.6,0,0,0]],
     [[15.0,6.0,20.0,0,0,0],[-15,6,-20,0,0,0]]])
    self.temperatures = np.array([[1.0,3.0],[100.0,10.0]])
    self.volumes = np.array([[0.1,0.1],[0.1,0.1]])

    # Case 3: Testing for an entirely random stress tensor over 3 time steps
    # self.stress = np.array([[
    #   [14.0,-17.0,4.3,105,2,15.5],[15.0,25.0,5.0,0,0,0]],[[-1.5,-25.0,-5.6,17.1,-6.6,-301],
    #   [54.0,-7.0,0.3,10,20,15.5]],[[-10.0,25.0,50.0,0,0,0],[-1.0,-205.0,-56.0,-11.7,-0.6,-30]]])
    # self.temperatures = np.array([[1.0,3.0],[15.0,100.0],[10.0,11.0]])
    # self.volumes = np.array([[0.1,0.1],[0.1,0.1],[0.1,0.1]])

    self.s0 = 70
    self.m = 3.5
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.MTSModel_GF(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    # Hand - calculated
    # Case 1
    # self.eq_stress = np.array([[80.1197]]) # calculated from mathematica

    # Case 2
    self.eq_stress = np.array([[80.1197,1.91569],[17.6925,4.40511]]) # calculated from mathematica

    # Case 3
    # self.eq_stress = np.array([[49.1661, 20.5435],[137.242, 41.6465],[38.2424, 18.5759]]) # calculated from mathematica

    should = -(2*kp/np.pi)*(self.eq_stress**self.m)*self.volumes

    actual = self.model.calculate_element_log_reliability(self.stress, self.temperatures, self.volumes, self.material)

    # Evaluating Reliability
    R_MTS_GF = np.exp(np.sum(actual))
    print("Reliability MTS GF = ",R_MTS_GF)

    # Evaluating Probability of Failure
    Pf_MTS_GF = 1 - np.exp(np.sum(actual))
    print("Probability of failure MTS GF = ",Pf_MTS_GF)

    self.assertTrue(np.allclose(should, actual))

class TestMTSModel_PSF(unittest.TestCase):
  def setUp(self):

    # Case 1: Single stress tensor over 1 time step
    # self.stress = np.array([[100.0,25.0,50.0,0,0,0]])
    # self.temperatures = np.array([1.0])
    # self.volumes = np.array([0.1])

    # Case 2: Testing for multiple (principal) stress tensors over 2 time steps
    self.stress = np.array([[
     [100.0,25.0,50.0,0,0,0],[-1.5,-25.0,-5.6,0,0,0]],
     [[15.0,6.0,20.0,0,0,0],[-15,6,-20,0,0,0]]])
    self.temperatures = np.array([[1.0,3.0],[100.0,10.0]])
    self.volumes = np.array([[0.1,0.1],[0.1,0.1]])

    # Case 3: Testing for an entirely random stress tensor over 3 time steps
    # self.stress = np.array([[
    #   [14.0,-17.0,4.3,105,2,15.5],[15.0,25.0,5.0,0,0,0]],[[-1.5,-25.0,-5.6,17.1,-6.6,-301],
    #   [54.0,-7.0,0.3,10,20,15.5]],[[-10.0,25.0,50.0,0,0,0],[-1.0,-205.0,-56.0,-11.7,-0.6,-30]]])
    # self.temperatures = np.array([[1.0,3.0],[15.0,100.0],[10.0,11.0]])
    # self.volumes = np.array([[0.1,0.1],[0.1,0.1],[0.1,0.1]])

    self.s0 = 70
    self.m = 3.5
    self.c_bar = 1.5
    #self.nu = 0.3

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.MTSModel_PSF(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    # Hand - calculated
    # Case 1
    # self.eq_stress = np.array([[80.7815]]) # calculated from mathematica

    # Case 2
    self.eq_stress = np.array([[80.7815,2.38915],[17.7883,4.94117]]) # calculated from mathematica

    # Case 3
    # self.eq_stress = np.array([[51.26391928,20.7078933],[143.27373523,42.63638728],[38.86744881,22.84607459]]) # calculated from mathematica

    should = -(2*kp/np.pi)*(self.eq_stress**self.m)*self.volumes

    actual = self.model.calculate_element_log_reliability(self.stress, self.temperatures, self.volumes, self.material)

    # Evaluating Reliability
    R_MTS_PSF = np.exp(np.sum(actual))
    print("Reliability MTS PSF = ",R_MTS_PSF)

    # Evaluating Probability of Failure
    Pf_MTS_PSF = 1 - np.exp(np.sum(actual))
    print("Probability of failure MTS PSF = ",Pf_MTS_PSF)

    self.assertTrue(np.allclose(should, actual))

class TestCSEModel_GF(unittest.TestCase):
  def setUp(self):

    # Case 1: Single stress tensor over 1 time step
    # self.stress = np.array([[100.0,25.0,50.0,0,0,0]])
    # self.temperatures = np.array([1.0])
    # self.volumes = np.array([0.1])

    # Case 2: Testing for multiple (principal) stress tensors over 2 time steps
    self.stress = np.array([[
     [100.0,25.0,50.0,0,0,0],[-1.5,-25.0,-5.6,0,0,0]],
     [[15.0,6.0,20.0,0,0,0],[-15,6,-20,0,0,0]]])
    self.temperatures = np.array([[1.0,3.0],[100.0,10.0]])
    self.volumes = np.array([[0.1,0.1],[0.1,0.1]])

    # Case 3: Testing for an entirely random stress tensor over 3 time steps
    # self.stress = np.array([[
    #   [14.0,-17.0,4.3,105,2,15.5],[15.0,25.0,5.0,0,0,0]],[[-1.5,-25.0,-5.6,17.1,-6.6,-301],
    #   [54.0,-7.0,0.3,10,20,15.5]],[[-10.0,25.0,50.0,0,0,0],[-1.0,-205.0,-56.0,-11.7,-0.6,-30]]])
    # self.temperatures = np.array([[1.0,3.0],[15.0,100.0],[10.0,11.0]])
    # self.volumes = np.array([[0.1,0.1],[0.1,0.1],[0.1,0.1]])

    self.s0 = 70
    self.m = 3.5
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.CSEModel_GF(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    # Hand - calculated
    # Case 1
    # self.eq_stress = np.array([[82.452]]) # calculated from mathematica

    # Case 2
    self.eq_stress = np.array([[82.452,18.9726],[18.0262,17.7775]]) # calculated from mathematica

    # Case 3
    # self.eq_stress = np.array([[75.3826,21.1317],[210.846,46.2446],[41.0402,158.244]]) # calculated from mathematica

    should = -(2*kp/np.pi)*(self.eq_stress**self.m)*self.volumes

    actual = self.model.calculate_element_log_reliability(self.stress, self.temperatures, self.volumes, self.material)

    # Evaluating Reliability
    R_CSE_GF = np.exp(np.sum(actual))
    print("Reliability MTS GF = ",R_CSE_GF)

    # Evaluating Probability of Failure
    Pf_CSE_GF = 1 - np.exp(np.sum(actual))
    print("Probability of failure CSE GF = ",Pf_CSE_GF)

    self.assertTrue(np.allclose(should, actual))

class TestCSEModel_PSF(unittest.TestCase):
  def setUp(self):

    # Case 1: Single stress tensor over 1 time step
    self.stress = np.array([[100.0,25.0,50.0,0,0,0]])
    self.temperatures = np.array([1.0])
    self.volumes = np.array([0.1])

    # Case 2: Testing for multiple (principal) stress tensors over 2 time steps
    # self.stress = np.array([[
    #  [100.0,25.0,50.0,0,0,0],[-1.5,-25.0,-5.6,0,0,0]],
    #  [[15.0,6.0,20.0,0,0,0],[-15,6,-20,0,0,0]]])
    # self.temperatures = np.array([[1.0,3.0],[100.0,10.0]])
    # self.volumes = np.array([[0.1,0.1],[0.1,0.1]])

    # Case 3: Testing for an entirely random stress tensor over 3 time steps
    # self.stress = np.array([[
    #   [14.0,-17.0,4.3,105,2,15.5],[15.0,25.0,5.0,0,0,0]],[[-1.5,-25.0,-5.6,17.1,-6.6,-301],
    #   [54.0,-7.0,0.3,10,20,15.5]],[[-10.0,25.0,50.0,0,0,0],[-1.0,-205.0,-56.0,-11.7,-0.6,-30]]])
    # self.temperatures = np.array([[1.0,3.0],[15.0,100.0],[10.0,11.0]])
    # self.volumes = np.array([[0.1,0.1],[0.1,0.1],[0.1,0.1]])

    self.s0 = 70
    self.m = 3.5
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.CSEModel_PSF(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    # Hand - calculated
    # Case 1
    self.eq_stress = np.array([[83.8427]]) # calculated from mathematica

    # Case 2
    # self.eq_stress = np.array([[83.8427,19.6321],[18.2279,18.5428]]) # calculated from mathematica

    # Case 3
    # self.eq_stress = np.array([[ 81.7748,21.484],[228.531,48.7533],[42.6446,164.011]]) # calculated from mathematica

    should = -(2*kp/np.pi)*(self.eq_stress**self.m)*self.volumes

    actual = self.model.calculate_element_log_reliability(self.stress, self.temperatures, self.volumes, self.material)

    # Evaluating Reliability
    R_CSE_PSF = np.exp(np.sum(actual))
    print("Reliability MTS GF = ",R_CSE_PSF)

    # Evaluating Probability of Failure
    Pf_CSE_PSF = 1 - np.exp(np.sum(actual))
    print("Probability of failure CSE PSF = ",Pf_CSE_PSF)

    self.assertTrue(np.allclose(should, actual))

class TestSMMModel_GF(unittest.TestCase):
  def setUp(self):

    # Case 1: Single stress tensor over 1 time step
    # self.stress = np.array([[100.0,25.0,50.0,0,0,0]])
    # self.temperatures = np.array([1.0])
    # self.volumes = np.array([0.1])

    # Case 2: Testing for multiple (principal) stress tensors over 2 time steps
    self.stress = np.array([[
     [100.0,25.0,50.0,0,0,0],[-1.5,-25.0,-5.6,0,0,0]],
     [[15.0,6.0,20.0,0,0,0],[-15,6,-20,0,0,0]]])
    self.temperatures = np.array([[1.0,3.0],[100.0,10.0]])
    self.volumes = np.array([[0.1,0.1],[0.1,0.1]])

    # Case 3: Testing for an entirely random stress tensor over 3 time steps
    # self.stress = np.array([[
    #   [14.0,-17.0,4.3,105,2,15.5],[15.0,25.0,5.0,0,0,0]],[[-1.5,-25.0,-5.6,17.1,-6.6,-301],
    #   [54.0,-7.0,0.3,10,20,15.5]],[[-10.0,25.0,50.0,0,0,0],[-1.0,-205.0,-56.0,-11.7,-0.6,-30]]])
    # self.temperatures = np.array([[1.0,3.0],[15.0,100.0],[10.0,11.0]])
    # self.volumes = np.array([[0.1,0.1],[0.1,0.1],[0.1,0.1]])

    self.s0 = 70
    self.m = 3.5
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.SMMModel_GF(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    # Hand - calculated
    # Case 1
    # self.eq_stress = np.array([[90.5452]]) # calculated from mathematica

    # Case 2
    self.eq_stress = np.array([[90.5452,7.79091],[19.2531,11.218]]) # calculated from mathematica

    # Case 3
    # self.eq_stress = np.array([[81.269,23.1647],[227.778,56.4233],[48.1351,70.2519]]) # calculated from mathematica

    should = -(2*kp/np.pi)*(self.eq_stress**self.m)*self.volumes

    actual = self.model.calculate_element_log_reliability(self.stress, self.temperatures, self.volumes, self.material)

    # Evaluating Reliability
    R_SMM_GF = np.exp(np.sum(actual))
    print("Reliability MTS GF = ",R_SMM_GF)

    # Evaluating Probability of Failure
    Pf_SMM_GF = 1 - np.exp(np.sum(actual))
    print("Probability of failure SMM GF = ",Pf_SMM_GF)

    self.assertTrue(np.allclose(should, actual))

class TestSMMModel_PSF(unittest.TestCase):
  def setUp(self):

    # Case 1: Single stress tensor over 1 time step
    # self.stress = np.array([[100.0,25.0,50.0,0,0,0]])
    # self.temperatures = np.array([1.0])
    # self.volumes = np.array([0.1])

    # Case 2: Testing for multiple (principal) stress tensors over 2 time steps
    self.stress = np.array([[
     [100.0,25.0,50.0,0,0,0],[-1.5,-25.0,-5.6,0,0,0]],
     [[15.0,6.0,20.0,0,0,0],[-15,6,-20,0,0,0]]])
    self.temperatures = np.array([[1.0,3.0],[100.0,10.0]])
    self.volumes = np.array([[0.1,0.1],[0.1,0.1]])

    # Case 3: Testing for an entirely random stress tensor over 3 time steps
    # self.stress = np.array([[
    #   [14.0,-17.0,4.3,105,2,15.5],[15.0,25.0,5.0,0,0,0]],[[-1.5,-25.0,-5.6,17.1,-6.6,-301],
    #   [54.0,-7.0,0.3,10,20,15.5]],[[-10.0,25.0,50.0,0,0,0],[-1.0,-205.0,-56.0,-11.7,-0.6,-30]]])
    # self.temperatures = np.array([[1.0,3.0],[15.0,100.0],[10.0,11.0]])
    # self.volumes = np.array([[0.1,0.1],[0.1,0.1],[0.1,0.1]])

    self.s0 = 70
    self.m = 3.5
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.SMMModel_PSF(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    # Hand - calculated
    # Case 1
    # self.eq_stress = np.array([[93.9531]]) # calculated from mathematica

    # Case 2
    self.eq_stress = np.array([[93.9531,9.35912],[19.7845,13.0138]]) # calculated from mathematica

    # Case 3
    # self.eq_stress = np.array([[90.7725,24.0325],[254.311,60.8945],[51.3082,83.8516]]) # calculated from mathematica

    should = -(2*kp/np.pi)*(self.eq_stress**self.m)*self.volumes

    actual = self.model.calculate_element_log_reliability(self.stress, self.temperatures, self.volumes, self.material)

    # Evaluating Reliability
    R_SMM_PSF = np.exp(np.sum(actual))
    print("Reliability MTS GF = ",R_SMM_PSF)

    # Evaluating Probability of Failure
    Pf_SMM_PSF = 1 - np.exp(np.sum(actual))
    print("Probability of failure SMM PSF = ",Pf_SMM_PSF)

    self.assertTrue(np.allclose(should, actual))
