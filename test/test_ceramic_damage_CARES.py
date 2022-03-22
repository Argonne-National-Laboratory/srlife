import unittest

import numpy as np
import csv
from openpyxl import load_workbook
import pandas as pd

from srlife import materials, damage_CARES, solverparams

class TestPIAModel(unittest.TestCase):
  def setUp(self):

    with open('ptsupport_p_1pt241_wo_block2.csv',newline='') as csvfile:
        sigma_xx = []
        sigma_yy = []
        sigma_zz = []
        sigma_yz = []
        sigma_xz = []
        sigma_xy = []
        Volume = []
        row_count = []
        self.cell_n = []
        reader = csv.DictReader(csvfile)
        for row in reader:
           sigmaxx = row['stress_rr']
           sigmayy = row['stress_tt']
           sigmazz = row['stress_zz']
           sigmaxy = row['stress_rt']
           sigmaxz = row['stress_rz']
           sigmayz = row['stress_tz']
           cell = row['cell']
           vol = row['vol']

           sigma_xx.append(sigmaxx)
           sigma_yy.append(sigmayy)
           sigma_zz.append(sigmazz)
           sigma_xy.append(sigmaxy)
           sigma_xz.append(sigmaxz)
           sigma_yz.append(sigmayz)
           self.cell_n.append(cell)
           Volume.append(vol)

    csvfile.close()
    sigma_xx = np.array([float(x) for x in sigma_xx])
    sigma_yy = np.array([float(x) for x in sigma_yy])
    sigma_zz = np.array([float(x) for x in sigma_zz])
    sigma_yz = np.array([float(x) for x in sigma_yz])
    sigma_xz = np.array([float(x) for x in sigma_xz])
    sigma_xy = np.array([float(x) for x in sigma_xy])
    Volume = np.array([float(x) for x in Volume])
    self.cell_n = np.array([float(x) for x in self.cell_n])
    row_count = len(sigma_xx)

    self.stress = np.array([[sigma_xx[0],sigma_yy[0],sigma_zz[0],np.sqrt(2)*sigma_yz[0],
        np.sqrt(2)*sigma_xz[0],np.sqrt(2)*sigma_xy[0]]])
    for i in range(1,row_count):
        self.stress = np.append(self.stress,np.array([[sigma_xx[i],sigma_yy[i],sigma_zz[i],
        np.sqrt(2)*sigma_yz[i],np.sqrt(2)*sigma_xz[i],np.sqrt(2)*sigma_xy[i]]]),axis=0)

    self.temperatures = np.ones(row_count)
    self.volumes = Volume

    self.s0 = 350.864 # in mm  169.7 in m
    self.m = 28.53
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage_CARES.PIAModel({})

  def test_definition(self):
    k = self.s0**(-self.m)

    mod_stress = self.stress
    print(self.stress.shape)
    print(self.volumes.shape)
    mod_stress[mod_stress<0] = 0

    # should = -k  * np.sum(mod_stress**self.m, axis = -1) * self.volumes
    # Pf = 1-np.exp(should)

    actual = self.model.calculate_element_log_reliability(self.stress,
        self.temperatures, self.volumes, self.material)

    R_PIA = np.exp(actual)
    Pf_PIA = 1 - np.exp(actual)
    Pf_PIA_max = np.max(Pf_PIA)
    print("Pf_PIA_max = ",Pf_PIA_max)

    writer = pd.ExcelWriter('PIA_ptsupport_241.xlsx', engine = 'openpyxl')
    wb = writer.book
    df = pd.DataFrame({'Cell':self.cell_n,'Reliabilities':R_PIA,'Probabilities':Pf_PIA,'Pf_PIA_max ':Pf_PIA_max})
    df.to_excel(writer,index=False)
    wb.save("PIA_ptsupport_241.xlsx")

    #self.assertTrue(np.allclose(should, actual))

class TestWeibullNormalTensileAveragingModel(unittest.TestCase):
  def setUp(self):

    with open('ptsupport_p_1pt241_wo_block2.csv',newline='') as csvfile:
        sigma_xx = []
        sigma_yy = []
        sigma_zz = []
        sigma_yz = []
        sigma_xz = []
        sigma_xy = []
        Volume = []
        row_count = []
        self.cell_n = []
        reader = csv.DictReader(csvfile)
        for row in reader:
           sigmaxx = row['stress_rr']
           sigmayy = row['stress_tt']
           sigmazz = row['stress_zz']
           sigmaxy = row['stress_rt']
           sigmaxz = row['stress_rz']
           sigmayz = row['stress_tz']
           cell = row['cell']
           vol = row['vol']

           sigma_xx.append(sigmaxx)
           sigma_yy.append(sigmayy)
           sigma_zz.append(sigmazz)
           sigma_xy.append(sigmaxy)
           sigma_xz.append(sigmaxz)
           sigma_yz.append(sigmayz)
           self.cell_n.append(cell)
           Volume.append(vol)

    csvfile.close()
    sigma_xx = np.array([float(x) for x in sigma_xx])
    sigma_yy = np.array([float(x) for x in sigma_yy])
    sigma_zz = np.array([float(x) for x in sigma_zz])
    sigma_yz = np.array([float(x) for x in sigma_yz])
    sigma_xz = np.array([float(x) for x in sigma_xz])
    sigma_xy = np.array([float(x) for x in sigma_xy])
    Volume = np.array([float(x) for x in Volume])
    self.cell_n = np.array([float(x) for x in self.cell_n])
    row_count = len(sigma_xx)

    self.stress = np.array([[sigma_xx[0],sigma_yy[0],sigma_zz[0],np.sqrt(2)*sigma_yz[0],
        np.sqrt(2)*sigma_xz[0],np.sqrt(2)*sigma_xy[0]]])
    for i in range(1,row_count):
        self.stress = np.append(self.stress,np.array([[sigma_xx[i],sigma_yy[i],sigma_zz[i],
        np.sqrt(2)*sigma_yz[i],np.sqrt(2)*sigma_xz[i],np.sqrt(2)*sigma_xy[i]]]),axis=0)

    self.temperatures = np.ones(row_count)
    self.volumes = Volume

    self.s0 = 350.864 # in mm  169.7 in m
    self.m = 28.53
    self.c_bar = 1.5

    self.material = materials.StandardCeramicMaterial(
        np.array([0,1000.0]), np.array([self.s0,self.s0]),
        self.m, self.c_bar)

    self.model = damage_CARES.WeibullNormalTensileAveragingModel(solverparams.ParameterSet())

  def test_definition(self):
    k = self.s0**(-self.m)
    kp = (2*self.m + 1)*k

    # Hand - calculated
    #self.avg_stress = np.array([[65.9504,0],[14.7457,2.18744]]) # calculated from mathematica

    #should = -kp * (self.avg_stress**self.m) * self.volumes

    actual = self.model.calculate_element_log_reliability(self.stress,
        self.temperatures, self.volumes, self.material)

    R_weibull = np.exp(actual)
    Pf_weibull = 1 - np.exp(actual)
    Pf_weibull_max = np.max(Pf_weibull)
    print("Pf_weibull_max = ",Pf_weibull_max)

    writer = pd.ExcelWriter('Weibull_ptsupport_241.xlsx', engine = 'openpyxl')
    wb = writer.book
    df = pd.DataFrame({'Cell':self.cell_n,'Reliabilities':R_weibull,'Probabilities':Pf_weibull,'Pf_PIA_max ':Pf_weibull_max})
    df.to_excel(writer,index=False)
    wb.save("Weibull_ptsupport_241.xlsx")

    #self.assertTrue(np.allclose(should, actual))
