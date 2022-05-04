import unittest

import numpy as np
import csv
import matplotlib.pyplot as plt
import os.path

from srlife import materials, damage, solverparams


class TestPIAModel(unittest.TestCase):
  def setUp(self):

    data1 = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),"data_ptsupt_finerHex20_p1pt724.csv"),
    delimiter = ',', skiprows = 1, usecols=list(range(0,10)))

    p1 = 1.724;
    p2 = 1.517;
    pressure_factor = p2/p1;

    sigma_xx = data1[:,0]*pressure_factor
    sigma_xy = data1[:,1]*pressure_factor
    sigma_xz = data1[:,2]*pressure_factor
    sigma_yy = data1[:,3]*pressure_factor
    sigma_yz = data1[:,4]*pressure_factor
    sigma_zz = data1[:,5]*pressure_factor
    #self.cell_n = data1[:,0]

    # data2 = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),"vol2.csv"),
    # delimiter = ',', skiprows = 1, usecols=list(range(1,5)))

    angle = 90.0;  #segment angle
    self.vol_factor = 360.0/angle;

    self.volumes = data1[:,9]*self.vol_factor

    row_count = len(sigma_xx)

    self.stress = np.array([[sigma_xx[0],sigma_yy[0],sigma_zz[0],
    np.sqrt(2)*sigma_yz[0],np.sqrt(2)*sigma_xz[0],np.sqrt(2)*sigma_xy[0]]])

    for i in range(1,row_count):
        self.stress = np.append(self.stress,np.array([[sigma_xx[i],sigma_yy[i],sigma_zz[i],
        np.sqrt(2)*sigma_yz[i],np.sqrt(2)*sigma_xz[i],np.sqrt(2)*sigma_xy[i]]]),axis=0)

    # self.stress = np.array([[sigma_xx[0],sigma_yy[0],sigma_zz[0],np.sqrt(2)*sigma_yz[0],
    #     np.sqrt(2)*sigma_xz[0],np.sqrt(2)*sigma_xy[0]]])
    # for i in range(1,row_count):
    #     self.stress = np.append(self.stress,np.array([[sigma_xx[i],sigma_yy[i],sigma_zz[i],
    #     np.sqrt(2)*sigma_yz[i],np.sqrt(2)*sigma_xz[i],np.sqrt(2)*sigma_xy[i]]]),axis=0)


    self.temperatures = np.ones(row_count)

    self.s0 = 350.864 # in mm  169.7 in m
    self.m = 28.53
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.PIAModel({})

  def test_definition(self):
    k = self.s0**(-self.m)

    mod_stress = self.stress
    mod_stress[mod_stress<0] = 0

    # should = -k  * np.sum(mod_stress**self.m, axis = -1) * self.volumes
    # Pf = 1-np.exp(should)

    actual = self.model.calculate_element_log_reliability(self.stress,
        self.temperatures, self.volumes, self.material)

    R_PIA = np.exp(np.sum(actual))
    Pf_PIA = 1 - R_PIA
    # Pf_PIA_max = np.max(Pf_PIA)

    # plt.plot(R_PIA,label = 'Reliabilities')
    # plt.plot(Pf_PIA, label = 'Probabilities')
    # plt.legend()
    # plt.show()

    print("Pf_PIA = ",Pf_PIA)

    # writer = pd.ExcelWriter('PIA_ptsupport_241.xlsx', engine = 'openpyxl')
    # wb = writer.book
    # df = pd.DataFrame({'Cell':self.cell_n,'Reliabilities':R_PIA,'Probabilities':Pf_PIA,'Pf_PIA_max ':Pf_PIA_max})
    # df.to_excel(writer,index=False)
    # wb.save("PIA_ptsupport_241.xlsx")

    #self.assertTrue(np.allclose(should, actual))

class TestWNTSAModel(unittest.TestCase):
  def setUp(self):

    data1 = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),"data_ptsupt_finerHex20_p1pt724.csv"),
    delimiter = ',', skiprows = 1, usecols=list(range(0,10)))

    p1 = 1.724;
    p2 = 1.517;
    pressure_factor = p2/p1;

    sigma_xx = data1[:,0]*pressure_factor
    sigma_xy = data1[:,1]*pressure_factor
    sigma_xz = data1[:,2]*pressure_factor
    sigma_yy = data1[:,3]*pressure_factor
    sigma_yz = data1[:,4]*pressure_factor
    sigma_zz = data1[:,5]*pressure_factor

    angle = 90.0;  #segment angle
    self.vol_factor = 360.0/angle;

    self.volumes = data1[:,9]*self.vol_factor

    row_count = len(sigma_xx)

    self.stress = np.array([[sigma_xx[0],sigma_yy[0],sigma_zz[0],
    np.sqrt(2)*sigma_yz[0],np.sqrt(2)*sigma_xz[0],np.sqrt(2)*sigma_xy[0]]])

    for i in range(1,row_count):
        self.stress = np.append(self.stress,np.array([[sigma_xx[i],sigma_yy[i],sigma_zz[i],
        np.sqrt(2)*sigma_yz[i],np.sqrt(2)*sigma_xz[i],np.sqrt(2)*sigma_xy[i]]]),axis=0)


    self.temperatures = np.ones(row_count)

    self.s0 = 350.864 # in mm  169.7 in m
    self.m = 28.53
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.WNTSAModel(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    actual = self.model.calculate_element_log_reliability(self.stress,
        self.temperatures, self.volumes, self.material)

    R_weibull = np.exp(np.sum(actual))
    Pf_weibull = 1 - R_weibull

    # plt.plot(R_weibull,label = 'Reliabilities')
    # plt.plot(Pf_weibull, label = 'Probabilities')
    # plt.legend()
    # plt.show()

    print("Pf_weibull = ",Pf_weibull)

class TestMTSModel_GF(unittest.TestCase):
  def setUp(self):

    data1 = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),"data_ptsupt_finerHex20_p1pt724.csv"),
    delimiter = ',', skiprows = 1, usecols=list(range(0,10)))

    p1 = 1.724;
    p2 = 1.517;
    pressure_factor = p2/p1;

    sigma_xx = data1[:,0]*pressure_factor
    sigma_xy = data1[:,1]*pressure_factor
    sigma_xz = data1[:,2]*pressure_factor
    sigma_yy = data1[:,3]*pressure_factor
    sigma_yz = data1[:,4]*pressure_factor
    sigma_zz = data1[:,5]*pressure_factor

    angle = 90.0;  #segment angle
    self.vol_factor = 360.0/angle;

    self.volumes = data1[:,9]*self.vol_factor

    row_count = len(sigma_xx)

    self.stress = np.array([[sigma_xx[0],sigma_yy[0],sigma_zz[0],
    np.sqrt(2)*sigma_yz[0],np.sqrt(2)*sigma_xz[0],np.sqrt(2)*sigma_xy[0]]])

    for i in range(1,row_count):
        self.stress = np.append(self.stress,np.array([[sigma_xx[i],sigma_yy[i],sigma_zz[i],
        np.sqrt(2)*sigma_yz[i],np.sqrt(2)*sigma_xz[i],np.sqrt(2)*sigma_xy[i]]]),axis=0)


    self.temperatures = np.ones(row_count)

    self.s0 = 350.864 # in mm  169.7 in m
    self.m = 28.53
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.MTSModel_GF(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    actual = self.model.calculate_element_log_reliability(self.stress,
        self.temperatures, self.volumes, self.material)

    R_MTS_GF = np.exp(np.sum(actual))
    Pf_MTS_GF = 1 - R_MTS_GF

    # plt.plot(R_weibull,label = 'Reliabilities')
    # plt.plot(Pf_weibull, label = 'Probabilities')
    # plt.legend()
    # plt.show()

    print("Pf_MTS_GF = ",Pf_MTS_GF)

class TestMTSModel_PSF(unittest.TestCase):
  def setUp(self):

    data1 = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),"data_ptsupt_finerHex20_p1pt724.csv"),
    delimiter = ',', skiprows = 1, usecols=list(range(0,10)))

    p1 = 1.724;
    p2 = 1.517;
    pressure_factor = p2/p1;

    sigma_xx = data1[:,0]*pressure_factor
    sigma_xy = data1[:,1]*pressure_factor
    sigma_xz = data1[:,2]*pressure_factor
    sigma_yy = data1[:,3]*pressure_factor
    sigma_yz = data1[:,4]*pressure_factor
    sigma_zz = data1[:,5]*pressure_factor

    angle = 90.0;  #segment angle
    self.vol_factor = 360.0/angle;

    self.volumes = data1[:,9]*self.vol_factor

    row_count = len(sigma_xx)

    self.stress = np.array([[sigma_xx[0],sigma_yy[0],sigma_zz[0],
    np.sqrt(2)*sigma_yz[0],np.sqrt(2)*sigma_xz[0],np.sqrt(2)*sigma_xy[0]]])

    for i in range(1,row_count):
        self.stress = np.append(self.stress,np.array([[sigma_xx[i],sigma_yy[i],sigma_zz[i],
        np.sqrt(2)*sigma_yz[i],np.sqrt(2)*sigma_xz[i],np.sqrt(2)*sigma_xy[i]]]),axis=0)


    self.temperatures = np.ones(row_count)

    self.s0 = 350.864 # in mm  169.7 in m
    self.m = 28.53
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.MTSModel_PSF(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    actual = self.model.calculate_element_log_reliability(self.stress,
        self.temperatures, self.volumes, self.material)

    R_MTS_PSF = np.exp(np.sum(actual))
    Pf_MTS_PSF = 1 - R_MTS_PSF

    # plt.plot(R_weibull,label = 'Reliabilities')
    # plt.plot(Pf_weibull, label = 'Probabilities')
    # plt.legend()
    # plt.show()

    print("Pf_MTS_PSF = ",Pf_MTS_PSF)

class TestCSEModel_GF(unittest.TestCase):
  def setUp(self):

    data1 = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),"data_ptsupt_finerHex20_p1pt724.csv"),
    delimiter = ',', skiprows = 1, usecols=list(range(0,10)))

    p1 = 1.724;
    p2 = 1.517;
    pressure_factor = p2/p1;

    sigma_xx = data1[:,0]*pressure_factor
    sigma_xy = data1[:,1]*pressure_factor
    sigma_xz = data1[:,2]*pressure_factor
    sigma_yy = data1[:,3]*pressure_factor
    sigma_yz = data1[:,4]*pressure_factor
    sigma_zz = data1[:,5]*pressure_factor

    angle = 90.0;  #segment angle
    self.vol_factor = 360.0/angle;

    self.volumes = data1[:,9]*self.vol_factor

    row_count = len(sigma_xx)

    self.stress = np.array([[sigma_xx[0],sigma_yy[0],sigma_zz[0],
    np.sqrt(2)*sigma_yz[0],np.sqrt(2)*sigma_xz[0],np.sqrt(2)*sigma_xy[0]]])

    for i in range(1,row_count):
        self.stress = np.append(self.stress,np.array([[sigma_xx[i],sigma_yy[i],sigma_zz[i],
        np.sqrt(2)*sigma_yz[i],np.sqrt(2)*sigma_xz[i],np.sqrt(2)*sigma_xy[i]]]),axis=0)


    self.temperatures = np.ones(row_count)

    self.s0 = 350.864 # in mm  169.7 in m
    self.m = 28.53
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.CSEModel_GF(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    actual = self.model.calculate_element_log_reliability(self.stress,
        self.temperatures, self.volumes, self.material)

    R_CSE_GF = np.exp(np.sum(actual))
    Pf_CSE_GF = 1 - R_CSE_GF

    # plt.plot(R_weibull,label = 'Reliabilities')
    # plt.plot(Pf_weibull, label = 'Probabilities')
    # plt.legend()
    # plt.show()

    print("Pf_CSE_GF = ",Pf_CSE_GF)

class TestCSEModel_PSF(unittest.TestCase):
  def setUp(self):

    data1 = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),"data_ptsupt_finerHex20_p1pt724.csv"),
    delimiter = ',', skiprows = 1, usecols=list(range(0,10)))

    p1 = 1.724;
    p2 = 1.517;
    pressure_factor = p2/p1;

    sigma_xx = data1[:,0]*pressure_factor
    sigma_xy = data1[:,1]*pressure_factor
    sigma_xz = data1[:,2]*pressure_factor
    sigma_yy = data1[:,3]*pressure_factor
    sigma_yz = data1[:,4]*pressure_factor
    sigma_zz = data1[:,5]*pressure_factor

    angle = 90.0;  #segment angle
    self.vol_factor = 360.0/angle;

    self.volumes = data1[:,9]*self.vol_factor

    row_count = len(sigma_xx)

    self.stress = np.array([[sigma_xx[0],sigma_yy[0],sigma_zz[0],
    np.sqrt(2)*sigma_yz[0],np.sqrt(2)*sigma_xz[0],np.sqrt(2)*sigma_xy[0]]])

    for i in range(1,row_count):
        self.stress = np.append(self.stress,np.array([[sigma_xx[i],sigma_yy[i],sigma_zz[i],
        np.sqrt(2)*sigma_yz[i],np.sqrt(2)*sigma_xz[i],np.sqrt(2)*sigma_xy[i]]]),axis=0)


    self.temperatures = np.ones(row_count)

    self.s0 = 350.864 # in mm  169.7 in m
    self.m = 28.53
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.CSEModel_PSF(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    actual = self.model.calculate_element_log_reliability(self.stress,
        self.temperatures, self.volumes, self.material)

    R_CSE_PSF = np.exp(np.sum(actual))
    Pf_CSE_PSF = 1 - R_CSE_PSF

    # plt.plot(R_weibull,label = 'Reliabilities')
    # plt.plot(Pf_weibull, label = 'Probabilities')
    # plt.legend()
    # plt.show()

    print("Pf_CSE_PSF = ",Pf_CSE_PSF)

class TestSMMModel_GF(unittest.TestCase):
  def setUp(self):

    data1 = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),"data_ptsupt_finerHex20_p1pt724.csv"),
    delimiter = ',', skiprows = 1, usecols=list(range(0,10)))

    p1 = 1.724;
    p2 = 1.517;
    pressure_factor = p2/p1;

    sigma_xx = data1[:,0]*pressure_factor
    sigma_xy = data1[:,1]*pressure_factor
    sigma_xz = data1[:,2]*pressure_factor
    sigma_yy = data1[:,3]*pressure_factor
    sigma_yz = data1[:,4]*pressure_factor
    sigma_zz = data1[:,5]*pressure_factor

    angle = 90.0;  #segment angle
    self.vol_factor = 360.0/angle;

    self.volumes = data1[:,9]*self.vol_factor

    row_count = len(sigma_xx)

    self.stress = np.array([[sigma_xx[0],sigma_yy[0],sigma_zz[0],
    np.sqrt(2)*sigma_yz[0],np.sqrt(2)*sigma_xz[0],np.sqrt(2)*sigma_xy[0]]])

    for i in range(1,row_count):
        self.stress = np.append(self.stress,np.array([[sigma_xx[i],sigma_yy[i],sigma_zz[i],
        np.sqrt(2)*sigma_yz[i],np.sqrt(2)*sigma_xz[i],np.sqrt(2)*sigma_xy[i]]]),axis=0)


    self.temperatures = np.ones(row_count)

    self.s0 = 350.864 # in mm  169.7 in m
    self.m = 28.53
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.SMMModel_GF(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    actual = self.model.calculate_element_log_reliability(self.stress,
        self.temperatures, self.volumes, self.material)

    R_SMM_GF = np.exp(np.sum(actual))
    Pf_SMM_GF = 1 - R_SMM_GF

    # plt.plot(R_weibull,label = 'Reliabilities')
    # plt.plot(Pf_weibull, label = 'Probabilities')
    # plt.legend()
    # plt.show()

    print("Pf_SMM_GF = ",Pf_SMM_GF)

class TestSMMModel_PSF(unittest.TestCase):
  def setUp(self):

    data1 = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),"data_ptsupt_finerHex20_p1pt724.csv"),
    delimiter = ',', skiprows = 1, usecols=list(range(0,10)))

    p1 = 1.724;
    p2 = 1.517;
    pressure_factor = p2/p1;

    sigma_xx = data1[:,0]*pressure_factor
    sigma_xy = data1[:,1]*pressure_factor
    sigma_xz = data1[:,2]*pressure_factor
    sigma_yy = data1[:,3]*pressure_factor
    sigma_yz = data1[:,4]*pressure_factor
    sigma_zz = data1[:,5]*pressure_factor

    angle = 90.0;  #segment angle
    self.vol_factor = 360.0/angle;

    self.volumes = data1[:,9]*self.vol_factor

    row_count = len(sigma_xx)

    self.stress = np.array([[sigma_xx[0],sigma_yy[0],sigma_zz[0],
    np.sqrt(2)*sigma_yz[0],np.sqrt(2)*sigma_xz[0],np.sqrt(2)*sigma_xy[0]]])

    for i in range(1,row_count):
        self.stress = np.append(self.stress,np.array([[sigma_xx[i],sigma_yy[i],sigma_zz[i],
        np.sqrt(2)*sigma_yz[i],np.sqrt(2)*sigma_xz[i],np.sqrt(2)*sigma_xy[i]]]),axis=0)


    self.temperatures = np.ones(row_count)

    self.s0 = 350.864 # in mm  169.7 in m
    self.m = 28.53
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage.SMMModel_PSF(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    actual = self.model.calculate_element_log_reliability(self.stress,
        self.temperatures, self.volumes, self.material)

    R_SMM_PSF = np.exp(np.sum(actual))
    Pf_SMM_PSF = 1 - R_SMM_PSF

    # plt.plot(R_weibull,label = 'Reliabilities')
    # plt.plot(Pf_weibull, label = 'Probabilities')
    # plt.legend()
    # plt.show()

    print("Pf_SMM_PSF = ",Pf_SMM_PSF)
