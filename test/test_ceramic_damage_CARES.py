import unittest

import numpy as np
import csv
import matplotlib.pyplot as plt
import os.path

from srlife import materials, damage, solverparams

p2 = 1.793
class TestPIAModel(unittest.TestCase):
    def setUp(self):

        data1 = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data_ptsupt_finerHex20_p1pt724.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(0, 10)),
        )

        p1 = 1.724
        # p2 = 1.793
        pressure_factor = p2 / p1

        sigma_xx = data1[:, 0] * pressure_factor
        sigma_xy = data1[:, 1] * pressure_factor
        sigma_xz = data1[:, 2] * pressure_factor
        sigma_yy = data1[:, 3] * pressure_factor
        sigma_yz = data1[:, 4] * pressure_factor
        sigma_zz = data1[:, 5] * pressure_factor
        # self.cell_n = data1[:,0]

        # data2 = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),"vol2.csv"),
        # delimiter = ',', skiprows = 1, usecols=list(range(1,5)))

        angle = 90.0
        # segment angle
        self.vol_factor = 360.0 / angle

        self.volumes = data1[:, 9] * self.vol_factor

        row_count = len(sigma_xx)

        self.stress = np.array(
            [
                [
                    sigma_xx[0],
                    sigma_yy[0],
                    sigma_zz[0],
                    np.sqrt(2) * sigma_yz[0],
                    np.sqrt(2) * sigma_xz[0],
                    np.sqrt(2) * sigma_xy[0],
                ]
            ]
        )

        for i in range(1, row_count):
            self.stress = np.append(
                self.stress,
                np.array(
                    [
                        [
                            sigma_xx[i],
                            sigma_yy[i],
                            sigma_zz[i],
                            np.sqrt(2) * sigma_yz[i],
                            np.sqrt(2) * sigma_xz[i],
                            np.sqrt(2) * sigma_xy[i],
                        ]
                    ]
                ),
                axis=0,
            )
        nelem = self.stress.shape[0]
        ntime_steps = 1
        self.stress = self.stress.reshape(ntime_steps,nelem,-1)
        # self.stress = np.array([[sigma_xx[0],sigma_yy[0],sigma_zz[0],np.sqrt(2)*sigma_yz[0],
        #     np.sqrt(2)*sigma_xz[0],np.sqrt(2)*sigma_xy[0]]])
        # for i in range(1,row_count):
        #     self.stress = np.append(self.stress,np.array([[sigma_xx[i],sigma_yy[i],sigma_zz[i],
        #     np.sqrt(2)*sigma_yz[i],np.sqrt(2)*sigma_xz[i],np.sqrt(2)*sigma_xy[i]]]),axis=0)

        # self.temperatures = np.ones(row_count)
        self.temperatures = np.ones((ntime_steps, nelem))
        
        # Number of cycles to failure
        self.nf = 100
        self.period = 1
        print("service life =", self.nf*self.period)
        self.time = np.linspace(0, self.period,ntime_steps+1)

        # Material properties (modified)
        self.su_v = 283.56497051
        self.su_s = 283.56497051
        # self.m_v = 28.53 # 2Parameter
        self.m_v = 3.4132857 # 3Parameter
        self.m_s = 28.53
        # self.s0_v = 350.864  # in mm  169.7 in m 2Parameter
        self.s0_v = 72.47538312  # in mm  169.7 in m
        self.s0_s = 350.864  # in mm  169.7 in m
        self.c_bar = 0.8
        self.nu = 0.25
        self.Nv = 30
        self.Ns = 30
        self.Bv = 320
        self.Bs = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.su_v,self.su_v]),
            np.array([self.su_s,self.su_s]),
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

        self.model = damage.PIAModel(solverparams.ParameterSet())

    def test_definition(self):
        k = self.s0_v ** (-self.m_v)
        kp = (2 * self.m_v + 1) * k

        mod_stress = self.stress
        mod_stress[mod_stress < 0] = 0

        actual = self.model.calculate_volume_flaw_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )

        R_PIA = np.exp(np.sum(actual))
        Pf_PIA = 1 - R_PIA
        # Pf_PIA_max = np.max(Pf_PIA)

        # plt.plot(R_PIA,label = 'Reliabilities')
        # plt.plot(Pf_PIA, label = 'Probabilities')
        # plt.legend()
        # plt.show()

        print("Pf_PIA = ", Pf_PIA)

        with open("PIA_vol.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_PIA}")
        # writer = pd.ExcelWriter('PIA_ptsupport_241.xlsx', engine = 'openpyxl')
        # wb = writer.book
        # df = pd.DataFrame({'Cell':self.cell_n,'Reliabilities':R_PIA,'Probabilities':Pf_PIA,'Pf_PIA_max ':Pf_PIA_max})
        # df.to_excel(writer,index=False)
        # wb.save("PIA_ptsupport_241.xlsx")

        # self.assertTrue(np.allclose(should, actual))


class TestWNTSAModel(unittest.TestCase):
    def setUp(self):

        data1 = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data_ptsupt_finerHex20_p1pt724.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(0, 10)),
        )

        p1 = 1.724
        # p2 = 1.793
        pressure_factor = p2 / p1

        sigma_xx = data1[:, 0] * pressure_factor
        sigma_xy = data1[:, 1] * pressure_factor
        sigma_xz = data1[:, 2] * pressure_factor
        sigma_yy = data1[:, 3] * pressure_factor
        sigma_yz = data1[:, 4] * pressure_factor
        sigma_zz = data1[:, 5] * pressure_factor

        angle = 90.0
        # segment angle
        self.vol_factor = 360.0 / angle

        self.volumes = data1[:, 9] * self.vol_factor

        row_count = len(sigma_xx)

        self.stress = np.array(
            [
                [
                    sigma_xx[0],
                    sigma_yy[0],
                    sigma_zz[0],
                    np.sqrt(2) * sigma_yz[0],
                    np.sqrt(2) * sigma_xz[0],
                    np.sqrt(2) * sigma_xy[0],
                ]
            ]
        )

        for i in range(1, row_count):
            self.stress = np.append(
                self.stress,
                np.array(
                    [
                        [
                            sigma_xx[i],
                            sigma_yy[i],
                            sigma_zz[i],
                            np.sqrt(2) * sigma_yz[i],
                            np.sqrt(2) * sigma_xz[i],
                            np.sqrt(2) * sigma_xy[i],
                        ]
                    ]
                ),
                axis=0,
            )

        nelem = self.stress.shape[0]
        ntime_steps = 1
        self.stress = self.stress.reshape(ntime_steps,nelem,-1)
        
        self.temperatures = np.ones((ntime_steps, nelem))

        # Number of cycles to failure
        self.nf = 100
        self.period = 1
        print("service life =", self.nf*self.period)
        self.time = np.linspace(0, self.period,ntime_steps+1)

        # Material properties (modified)
        self.su_v = 283.56497051
        self.su_s = 283.56497051
        # self.m_v = 28.53 # 2Parameter
        self.m_v = 3.4132857 # 3Parameter
        self.m_s = 28.53
        # self.s0_v = 350.864  # in mm  169.7 in m 2Parameter
        self.s0_v = 72.47538312  # in mm  169.7 in m
        self.s0_s = 350.864  # in mm  169.7 in m
        self.c_bar = 0.8
        self.nu = 0.25
        self.Nv = 30
        self.Ns = 30
        self.Bv = 320
        self.Bs = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.su_v,self.su_v]),
            np.array([self.su_s,self.su_s]),
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

        self.model = damage.WNTSAModel(solverparams.ParameterSet())

    def test_definition(self):
        k = self.s0_v ** (-self.m_v)
        kp = (2 * self.m_v + 1) * k

        actual = self.model.calculate_volume_flaw_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )

        R_weibull = np.exp(np.sum(actual))
        Pf_weibull = 1 - R_weibull

        # plt.plot(R_weibull,label = 'Reliabilities')
        # plt.plot(Pf_weibull, label = 'Probabilities')
        # plt.legend()
        # plt.show()

        print("Pf_weibull = ", Pf_weibull)
        
        with open("WNTSA_vol.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_weibull}")


class TestMTSModelGriffithFlaw(unittest.TestCase):
    def setUp(self):

        data1 = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data_ptsupt_finerHex20_p1pt724.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(0, 10)),
        )

        p1 = 1.724
        # p2 = 1.793
        pressure_factor = p2 / p1

        sigma_xx = data1[:, 0] * pressure_factor
        sigma_xy = data1[:, 1] * pressure_factor
        sigma_xz = data1[:, 2] * pressure_factor
        sigma_yy = data1[:, 3] * pressure_factor
        sigma_yz = data1[:, 4] * pressure_factor
        sigma_zz = data1[:, 5] * pressure_factor

        angle = 90.0
        # segment angle
        self.vol_factor = 360.0 / angle

        self.volumes = data1[:, 9] * self.vol_factor

        row_count = len(sigma_xx)

        self.stress = np.array(
            [
                [
                    sigma_xx[0],
                    sigma_yy[0],
                    sigma_zz[0],
                    np.sqrt(2) * sigma_yz[0],
                    np.sqrt(2) * sigma_xz[0],
                    np.sqrt(2) * sigma_xy[0],
                ]
            ]
        )

        for i in range(1, row_count):
            self.stress = np.append(
                self.stress,
                np.array(
                    [
                        [
                            sigma_xx[i],
                            sigma_yy[i],
                            sigma_zz[i],
                            np.sqrt(2) * sigma_yz[i],
                            np.sqrt(2) * sigma_xz[i],
                            np.sqrt(2) * sigma_xy[i],
                        ]
                    ]
                ),
                axis=0,
            )

        nelem = self.stress.shape[0]
        ntime_steps = 1
        self.stress = self.stress.reshape(ntime_steps,nelem,-1)

        self.temperatures = np.ones((ntime_steps, nelem))

        # Number of cycles to failure
        self.nf = 100
        self.period = 1
        print("service life =", self.nf*self.period)
        self.time = np.linspace(0, self.period,ntime_steps+1)

        # Material properties (modified)
        self.su_v = 283.56497051
        self.su_s = 283.56497051
        # self.m_v = 28.53 # 2Parameter
        self.m_v = 3.4132857 # 3Parameter
        self.m_s = 28.53
        # self.s0_v = 350.864  # in mm  169.7 in m 2Parameter
        self.s0_v = 72.47538312  # in mm  169.7 in m
        self.s0_s = 350.864  # in mm  169.7 in m
        self.c_bar = 0.8
        self.nu = 0.25
        self.Nv = 30
        self.Ns = 30
        self.Bv = 320
        self.Bs = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.su_v,self.su_v]),
            np.array([self.su_s,self.su_s]),
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

        self.model = damage.MTSModelGriffithFlaw(solverparams.ParameterSet())

    def test_definition(self):
        k = self.s0_v ** (-self.m_v)
        kp = (2 * self.m_v + 1) * k

        actual = self.model.calculate_volume_flaw_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )

        R_MTS_GF = np.exp(np.sum(actual))
        Pf_MTS_GF = 1 - R_MTS_GF

        # plt.plot(R_weibull,label = 'Reliabilities')
        # plt.plot(Pf_weibull, label = 'Probabilities')
        # plt.legend()
        # plt.show()

        print("Pf_MTS_GF = ", Pf_MTS_GF)

        with open("MTS_GF_vol.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_MTS_GF}")


class TestMTSModelPennyShapedFlaw(unittest.TestCase):
    def setUp(self):

        data1 = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data_ptsupt_finerHex20_p1pt724.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(0, 10)),
        )

        p1 = 1.724
        # p2 = 1.241
        pressure_factor = p2 / p1

        sigma_xx = data1[:, 0] * pressure_factor
        sigma_xy = data1[:, 1] * pressure_factor
        sigma_xz = data1[:, 2] * pressure_factor
        sigma_yy = data1[:, 3] * pressure_factor
        sigma_yz = data1[:, 4] * pressure_factor
        sigma_zz = data1[:, 5] * pressure_factor

        angle = 90.0
        # segment angle
        self.vol_factor = 360.0 / angle

        self.volumes = data1[:, 9] * self.vol_factor

        row_count = len(sigma_xx)

        self.stress = np.array(
            [
                [
                    sigma_xx[0],
                    sigma_yy[0],
                    sigma_zz[0],
                    np.sqrt(2) * sigma_yz[0],
                    np.sqrt(2) * sigma_xz[0],
                    np.sqrt(2) * sigma_xy[0],
                ]
            ]
        )

        for i in range(1, row_count):
            self.stress = np.append(
                self.stress,
                np.array(
                    [
                        [
                            sigma_xx[i],
                            sigma_yy[i],
                            sigma_zz[i],
                            np.sqrt(2) * sigma_yz[i],
                            np.sqrt(2) * sigma_xz[i],
                            np.sqrt(2) * sigma_xy[i],
                        ]
                    ]
                ),
                axis=0,
            )

        nelem = self.stress.shape[0]
        ntime_steps = 1
        self.stress = self.stress.reshape(ntime_steps,nelem,-1)

        self.temperatures = np.ones((ntime_steps, nelem))

        # Number of cycles to failure
        self.nf = 100
        self.period = 1
        print("service life =", self.nf*self.period)
        self.time = np.linspace(0, self.period,ntime_steps+1)

        # Material properties (modified)
        self.su_v = 283.56497051
        self.su_s = 283.56497051
        # self.m_v = 28.53 # 2Parameter
        self.m_v = 3.4132857 # 3Parameter
        self.m_s = 28.53
        # self.s0_v = 350.864  # in mm  169.7 in m 2Parameter
        self.s0_v = 72.47538312  # in mm  169.7 in m
        self.s0_s = 350.864  # in mm  169.7 in m
        self.c_bar = 0.8
        self.nu = 0.25
        self.Nv = 30
        self.Ns = 30
        self.Bv = 320
        self.Bs = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.su_v,self.su_v]),
            np.array([self.su_s,self.su_s]),
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

        self.model = damage.MTSModelPennyShapedFlaw(solverparams.ParameterSet())

    def test_definition(self):
        k = self.s0_v ** (-self.m_v)
        kp = (2 * self.m_v + 1) * k

        actual = self.model.calculate_volume_flaw_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )

        R_MTS_PSF = np.exp(np.sum(actual))
        Pf_MTS_PSF = 1 - R_MTS_PSF

        # plt.plot(R_weibull,label = 'Reliabilities')
        # plt.plot(Pf_weibull, label = 'Probabilities')
        # plt.legend()
        # plt.show()

        print("Pf_MTS_PSF = ", Pf_MTS_PSF)

        with open("MTS_PSF_vol.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_MTS_PSF}")


class TestCSEModelGriffithFlaw(unittest.TestCase):
    def setUp(self):

        data1 = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data_ptsupt_finerHex20_p1pt724.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(0, 10)),
        )

        p1 = 1.724
        # p2 = 1.517
        pressure_factor = p2 / p1

        sigma_xx = data1[:, 0] * pressure_factor
        sigma_xy = data1[:, 1] * pressure_factor
        sigma_xz = data1[:, 2] * pressure_factor
        sigma_yy = data1[:, 3] * pressure_factor
        sigma_yz = data1[:, 4] * pressure_factor
        sigma_zz = data1[:, 5] * pressure_factor

        angle = 90.0
        # segment angle
        self.vol_factor = 360.0 / angle

        self.volumes = data1[:, 9] * self.vol_factor

        row_count = len(sigma_xx)

        self.stress = np.array(
            [
                [
                    sigma_xx[0],
                    sigma_yy[0],
                    sigma_zz[0],
                    np.sqrt(2) * sigma_yz[0],
                    np.sqrt(2) * sigma_xz[0],
                    np.sqrt(2) * sigma_xy[0],
                ]
            ]
        )

        for i in range(1, row_count):
            self.stress = np.append(
                self.stress,
                np.array(
                    [
                        [
                            sigma_xx[i],
                            sigma_yy[i],
                            sigma_zz[i],
                            np.sqrt(2) * sigma_yz[i],
                            np.sqrt(2) * sigma_xz[i],
                            np.sqrt(2) * sigma_xy[i],
                        ]
                    ]
                ),
                axis=0,
            )

        nelem = self.stress.shape[0]
        ntime_steps = 1
        self.stress = self.stress.reshape(ntime_steps,nelem,-1)

        self.temperatures = np.ones((ntime_steps, nelem))

        # Number of cycles to failure
        self.nf = 100
        self.period = 1
        print("service life =", self.nf*self.period)
        self.time = np.linspace(0, self.period,ntime_steps+1)

        # Material properties (modified)
        self.su_v = 283.56497051
        self.su_s = 283.56497051
        # self.m_v = 28.53 # 2Parameter
        self.m_v = 3.4132857 # 3Parameter
        self.m_s = 28.53
        # self.s0_v = 350.864  # in mm  169.7 in m 2Parameter
        self.s0_v = 72.47538312  # in mm  169.7 in m
        self.s0_s = 350.864  # in mm  169.7 in m
        self.c_bar = 0.8
        self.nu = 0.25
        self.Nv = 30
        self.Ns = 30
        self.Bv = 320
        self.Bs = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.su_v,self.su_v]),
            np.array([self.su_s,self.su_s]),
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

        self.model = damage.CSEModelGriffithFlaw(solverparams.ParameterSet())

    def test_definition(self):
        k = self.s0_v ** (-self.m_v)
        kp = (2 * self.m_v + 1) * k

        actual = self.model.calculate_volume_flaw_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )

        R_CSE_GF = np.exp(np.sum(actual))
        Pf_CSE_GF = 1 - R_CSE_GF

        # plt.plot(R_weibull,label = 'Reliabilities')
        # plt.plot(Pf_weibull, label = 'Probabilities')
        # plt.legend()
        # plt.show()

        print("Pf_CSE_GF = ", Pf_CSE_GF)

        with open("CSE_GF_vol.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_CSE_GF}")


class TestCSEModelPennyShapedFlaw(unittest.TestCase):
    def setUp(self):

        data1 = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data_ptsupt_finerHex20_p1pt724.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(0, 10)),
        )

        p1 = 1.724
        # p2 = 1.517
        pressure_factor = p2 / p1

        sigma_xx = data1[:, 0] * pressure_factor
        sigma_xy = data1[:, 1] * pressure_factor
        sigma_xz = data1[:, 2] * pressure_factor
        sigma_yy = data1[:, 3] * pressure_factor
        sigma_yz = data1[:, 4] * pressure_factor
        sigma_zz = data1[:, 5] * pressure_factor

        angle = 90.0
        # segment angle
        self.vol_factor = 360.0 / angle

        self.volumes = data1[:, 9] * self.vol_factor

        row_count = len(sigma_xx)

        self.stress = np.array(
            [
                [
                    sigma_xx[0],
                    sigma_yy[0],
                    sigma_zz[0],
                    np.sqrt(2) * sigma_yz[0],
                    np.sqrt(2) * sigma_xz[0],
                    np.sqrt(2) * sigma_xy[0],
                ]
            ]
        )

        for i in range(1, row_count):
            self.stress = np.append(
                self.stress,
                np.array(
                    [
                        [
                            sigma_xx[i],
                            sigma_yy[i],
                            sigma_zz[i],
                            np.sqrt(2) * sigma_yz[i],
                            np.sqrt(2) * sigma_xz[i],
                            np.sqrt(2) * sigma_xy[i],
                        ]
                    ]
                ),
                axis=0,
            )

        nelem = self.stress.shape[0]
        ntime_steps = 1
        self.stress = self.stress.reshape(ntime_steps,nelem,-1)
        
        self.temperatures = np.ones((ntime_steps, nelem))

        # Number of cycles to failure
        self.nf = 100
        self.period = 1
        print("service life =", self.nf*self.period)
        self.time = np.linspace(0, self.period,ntime_steps+1)

        # Material properties (modified)
        self.su_v = 283.56497051
        self.su_s = 283.56497051
        # self.m_v = 28.53 # 2Parameter
        self.m_v = 3.4132857 # 3Parameter
        self.m_s = 28.53
        # self.s0_v = 350.864  # in mm  169.7 in m 2Parameter
        self.s0_v = 72.47538312  # in mm  169.7 in m
        self.s0_s = 350.864  # in mm  169.7 in m
        self.c_bar = 0.8
        self.nu = 0.25
        self.Nv = 30
        self.Ns = 30
        self.Bv = 320
        self.Bs = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.su_v,self.su_v]),
            np.array([self.su_s,self.su_s]),
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

        self.model = damage.CSEModelPennyShapedFlaw(solverparams.ParameterSet())

    def test_definition(self):
        k = self.s0_v ** (-self.m_v)
        kp = (2 * self.m_v + 1) * k

        actual = self.model.calculate_volume_flaw_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )

        R_CSE_PSF = np.exp(np.sum(actual))
        Pf_CSE_PSF = 1 - R_CSE_PSF

        # plt.plot(R_weibull,label = 'Reliabilities')
        # plt.plot(Pf_weibull, label = 'Probabilities')
        # plt.legend()
        # plt.show()

        print("Pf_CSE_PSF = ", Pf_CSE_PSF)

        with open("CSE_PSF_vol.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_CSE_PSF}")


class TestSMMModelGriffithFlaw(unittest.TestCase):
    def setUp(self):

        data1 = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data_ptsupt_finerHex20_p1pt724.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(0, 10)),
        )

        p1 = 1.724
        # p2 = 1.517
        pressure_factor = p2 / p1

        sigma_xx = data1[:, 0] * pressure_factor
        sigma_xy = data1[:, 1] * pressure_factor
        sigma_xz = data1[:, 2] * pressure_factor
        sigma_yy = data1[:, 3] * pressure_factor
        sigma_yz = data1[:, 4] * pressure_factor
        sigma_zz = data1[:, 5] * pressure_factor

        angle = 90.0
        # segment angle
        self.vol_factor = 360.0 / angle

        self.volumes = data1[:, 9] * self.vol_factor

        row_count = len(sigma_xx)

        self.stress = np.array(
            [
                [
                    sigma_xx[0],
                    sigma_yy[0],
                    sigma_zz[0],
                    np.sqrt(2) * sigma_yz[0],
                    np.sqrt(2) * sigma_xz[0],
                    np.sqrt(2) * sigma_xy[0],
                ]
            ]
        )

        for i in range(1, row_count):
            self.stress = np.append(
                self.stress,
                np.array(
                    [
                        [
                            sigma_xx[i],
                            sigma_yy[i],
                            sigma_zz[i],
                            np.sqrt(2) * sigma_yz[i],
                            np.sqrt(2) * sigma_xz[i],
                            np.sqrt(2) * sigma_xy[i],
                        ]
                    ]
                ),
                axis=0,
            )

        nelem = self.stress.shape[0]
        ntime_steps = 1
        self.stress = self.stress.reshape(ntime_steps,nelem,-1)
        
        self.temperatures = np.ones((ntime_steps, nelem))

        # Number of cycles to failure
        self.nf = 100
        self.period = 1
        print("service life =", self.nf*self.period)
        self.time = np.linspace(0, self.period,ntime_steps+1)

        # Material properties (modified)
        self.su_v = 283.56497051
        self.su_s = 283.56497051
        # self.m_v = 28.53 # 2Parameter
        self.m_v = 3.4132857 # 3Parameter
        self.m_s = 28.53
        # self.s0_v = 350.864  # in mm  169.7 in m 2Parameter
        self.s0_v = 72.47538312  # in mm  169.7 in m
        self.s0_s = 350.864  # in mm  169.7 in m
        self.c_bar = 0.8
        self.nu = 0.25
        self.Nv = 30
        self.Ns = 30
        self.Bv = 320
        self.Bs = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.su_v,self.su_v]),
            np.array([self.su_s,self.su_s]),
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

        self.model = damage.SMMModelGriffithFlaw(solverparams.ParameterSet())

    def test_definition(self):
        k = self.s0_v ** (-self.m_v)
        kp = (2 * self.m_v + 1) * k

        actual = self.model.calculate_volume_flaw_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )

        R_SMM_GF = np.exp(np.sum(actual))
        Pf_SMM_GF = 1 - R_SMM_GF

        # plt.plot(R_weibull,label = 'Reliabilities')
        # plt.plot(Pf_weibull, label = 'Probabilities')
        # plt.legend()
        # plt.show()

        print("Pf_SMM_GF = ", Pf_SMM_GF)

        with open("SMM_GF_vol.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_SMM_GF}")


class TestSMMModelPennyShapedFlaw(unittest.TestCase):
    def setUp(self):

        data1 = np.loadtxt(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data_ptsupt_finerHex20_p1pt724.csv",
            ),
            delimiter=",",
            skiprows=1,
            usecols=list(range(0, 10)),
        )

        p1 = 1.724
        # p2 = 1.517
        pressure_factor = p2 / p1

        sigma_xx = data1[:, 0] * pressure_factor
        sigma_xy = data1[:, 1] * pressure_factor
        sigma_xz = data1[:, 2] * pressure_factor
        sigma_yy = data1[:, 3] * pressure_factor
        sigma_yz = data1[:, 4] * pressure_factor
        sigma_zz = data1[:, 5] * pressure_factor

        angle = 90.0
        # segment angle
        self.vol_factor = 360.0 / angle

        self.volumes = data1[:, 9] * self.vol_factor

        row_count = len(sigma_xx)

        self.stress = np.array(
            [
                [
                    sigma_xx[0],
                    sigma_yy[0],
                    sigma_zz[0],
                    np.sqrt(2) * sigma_yz[0],
                    np.sqrt(2) * sigma_xz[0],
                    np.sqrt(2) * sigma_xy[0],
                ]
            ]
        )

        for i in range(1, row_count):
            self.stress = np.append(
                self.stress,
                np.array(
                    [
                        [
                            sigma_xx[i],
                            sigma_yy[i],
                            sigma_zz[i],
                            np.sqrt(2) * sigma_yz[i],
                            np.sqrt(2) * sigma_xz[i],
                            np.sqrt(2) * sigma_xy[i],
                        ]
                    ]
                ),
                axis=0,
            )

        nelem = self.stress.shape[0]
        ntime_steps = 1
        self.stress = self.stress.reshape(ntime_steps,nelem,-1)
        
        self.temperatures = np.ones((ntime_steps, nelem))

        # Number of cycles to failure
        self.nf = 100
        self.period = 1
        print("service life =", self.nf*self.period)
        self.time = np.linspace(0, self.period,ntime_steps+1)

        # Material properties (modified)
        self.su_v = 283.56497051
        self.su_s = 283.56497051
        # self.m_v = 28.53 # 2Parameter
        self.m_v = 3.4132857 # 3Parameter
        self.m_s = 28.53
        # self.s0_v = 350.864  # in mm  169.7 in m 2Parameter
        self.s0_v = 72.47538312  # in mm  169.7 in m
        self.s0_s = 350.864  # in mm  169.7 in m
        self.c_bar = 0.8
        self.nu = 0.25
        self.Nv = 30
        self.Ns = 30
        self.Bv = 320
        self.Bs = 320

        self.material = materials.StandardCeramicMaterial(
            np.array([0, 1000.0]),
            np.array([self.su_v,self.su_v]),
            np.array([self.su_s,self.su_s]),
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

        self.model = damage.SMMModelPennyShapedFlaw(solverparams.ParameterSet())

    def test_definition(self):
        k = self.s0_v ** (-self.m_v)
        kp = (2 * self.m_v + 1) * k

        actual = self.model.calculate_volume_flaw_element_log_reliability(
            self.time,
            self.stress,
            self.temperatures,
            self.volumes,
            self.material,
            self.nf * self.period,
        )

        R_SMM_PSF = np.exp(np.sum(actual))
        Pf_SMM_PSF = 1 - R_SMM_PSF

        # plt.plot(R_weibull,label = 'Reliabilities')
        # plt.plot(Pf_weibull, label = 'Probabilities')
        # plt.legend()
        # plt.show()

        print("Pf_SMM_PSF = ", Pf_SMM_PSF)

        with open("SMM_PSF_vol.txt", "a+") as external_file:
            external_file.write("\n")
            external_file.write(f"{Pf_SMM_PSF}")
