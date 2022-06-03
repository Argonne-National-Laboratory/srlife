import unittest

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from srlife import receiver, writers


class TestVolumeCalculation(unittest.TestCase):
    def setUp(self):
        self.ro = 9.0
        self.t = 1.2
        self.h = 15.0

        self.nr = 10
        self.nt = 36
        self.nz = 11

        self.tube = receiver.Tube(self.ro, self.t, self.h, self.nr, self.nt, self.nz)

    def test_3D(self):
        v1 = self.tube.element_volumes()

        writer = writers.VTKWriter(self.tube, "dummy.vtk")
        obj = writer.make_vtk_object()

        meshQuality = vtk.vtkMeshQuality()
        meshQuality.SetInputData(obj)
        meshQuality.SetHexQualityMeasureToVolume()
        meshQuality.Update()
        res = meshQuality.GetOutput()
        vols = vtk_to_numpy(res.GetCellData().GetArray("Quality"))

        err = np.abs(v1 - vols) / np.abs(vols)

        self.assertTrue(np.allclose(v1, vols, rtol=1e-4))

    def test_2D(self):
        self.tube.make_2D(self.h / 2)
        v1 = self.tube.element_volumes()

        writer = writers.VTKWriter(self.tube, "dummy.vtk")
        obj = writer.make_vtk_object()

        meshQuality = vtk.vtkMeshQuality()
        meshQuality.SetInputData(obj)
        meshQuality.SetQuadQualityMeasureToArea()
        meshQuality.Update()
        res = meshQuality.GetOutput()
        vols = vtk_to_numpy(res.GetCellData().GetArray("Quality")) * self.h

        self.assertTrue(np.allclose(v1, vols, rtol=1e-4))

    def test_1D(self):
        self.tube.make_1D(self.h / 2, 0)
        v1 = self.tube.element_volumes()
        vt = np.sum(v1)

        v_check = np.pi * (self.ro**2.0 - (self.ro - self.t) ** 2.0) * self.h

        self.assertAlmostEqual(vt, v_check)
