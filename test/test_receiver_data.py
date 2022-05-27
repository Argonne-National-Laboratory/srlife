import unittest
import tempfile

import numpy as np
import h5py

from srlife import receiver

outer_radius = 1.0
thickness = 0.1
height = 100.0

nr = 10
nt = 10
nz = 10

inner_radius = outer_radius - thickness

period = 10.0
ndays = 10

nsteps = 10

times = np.linspace(0, period * ndays, ndays * nsteps)

panel_k = 0
tube_k = np.inf


def get_temp_hdf():
    return h5py.File(tempfile.mktemp(), "w")


def valid_pressure_bc():
    return receiver.PressureBC(times, np.zeros((len(times),)))


def valid_convective_bc(loc):
    if loc == "outer":
        r = outer_radius
    else:
        r = inner_radius
    return receiver.ConvectiveBC(r, height, nz, times, np.zeros((len(times), nz)))


def valid_heatflux_bc(loc):
    if loc == "outer":
        r = outer_radius
    else:
        r = inner_radius
    return receiver.HeatFluxBC(r, height, nt, nz, times, np.zeros((len(times), nt, nz)))


def valid_fixedtemp_bc(loc):
    if loc == "outer":
        r = outer_radius
    else:
        r = inner_radius
    return receiver.FixedTempBC(
        r, height, nt, nz, times, np.zeros((len(times), nt, nz))
    )


def valid_tube(results=[]):
    tube = receiver.Tube(outer_radius, thickness, height, nr, nt, nz)

    tube.set_times(times)

    for res in results:
        tube.add_results(res, np.zeros((len(times), nr, nt, nz)))

    tube.set_bc(valid_heatflux_bc("inner"), "inner")
    tube.set_bc(valid_convective_bc("outer"), "outer")
    tube.set_pressure_bc(valid_pressure_bc())

    return tube


def valid_tube_1D(results=[]):
    tube = receiver.Tube(outer_radius, thickness, height, nr, nt, nz)

    tube.set_times(times)

    tube.make_1D(height / 2, np.pi / 3)

    for res in results:
        tube.add_results(res, np.zeros((len(times), nr)))

    return tube


def valid_panel(n=0, results=[]):
    panel = receiver.Panel(tube_k)

    for i in range(n):
        panel.add_tube(valid_tube(results))

    return panel


def valid_receiver(npanel=0, ntube=0, results=[]):
    rec = receiver.Receiver(period, ndays, panel_k)

    for i in range(npanel):
        rec.add_panel(valid_panel(ntube, results))

    return rec


class TestReceiver(unittest.TestCase):
    def test_construct(self):
        rec = valid_receiver(10, 10, ["stress", "strain"])

    def test_next_number(self):
        rec = valid_receiver(2)
        rec.add_panel(valid_panel(), name="A")
        rec.add_panel(valid_panel())

        self.assertEqual(rec.npanels, 4)

        self.assertFalse("3" in rec.panels.keys())
        self.assertTrue("2" in rec.panels.keys())

    def test_store(self):
        base = valid_receiver(10, 20, ["stress", "strain"])
        f = get_temp_hdf()
        base.save(f)
        new = receiver.Receiver.load(f)

        self.assertTrue(base.close(new))


class TestPanel(unittest.TestCase):
    def test_construct(self):
        panel = valid_panel(2)

    def test_next_number(self):
        panel = valid_panel(2)
        panel.add_tube(valid_tube(), name="A")
        panel.add_tube(valid_tube())

        self.assertEqual(panel.ntubes, 4)

        self.assertFalse("3" in panel.tubes.keys())
        self.assertTrue("2" in panel.tubes.keys())

    def test_store(self):
        base = valid_panel(5, ["stress"])

        f = get_temp_hdf()
        base.save(f)
        new = receiver.Panel.load(f)

        self.assertTrue(base.close(new))


class TestTube(unittest.TestCase):
    def test_construct(self):
        tube = valid_tube(["stress", "strain"])

    def test_invalid_size(self):
        tube = valid_tube(["stress"])
        with self.assertRaises(ValueError):
            tube.add_results("strain", np.zeros((1,)))

    def test_invalid_times(self):
        tube = valid_tube(["stress"])
        with self.assertRaises(ValueError):
            tube.set_times(np.linspace(0, 1, 10))

    def test_valid_bc(self):
        tube = valid_tube()
        tube.set_bc(valid_heatflux_bc("outer"), "outer")
        tube.set_bc(valid_heatflux_bc("inner"), "inner")

    def test_invalid_bc(self):
        tube = valid_tube()
        with self.assertRaises(ValueError):
            tube.set_bc(valid_heatflux_bc("outer"), "inner")

    def test_invalid_bc_location(self):
        tube = valid_tube()
        with self.assertRaises(ValueError):
            tube.set_bc(valid_heatflux_bc("outer"), "blah")

    def test_store(self):
        tube = valid_tube(["stress"])
        tube.set_bc(valid_heatflux_bc("outer"), "outer")

        f = get_temp_hdf()
        tube.save(f)
        new = receiver.Tube.load(f)

        self.assertTrue(tube.close(new))

    def test_store_1d(self):
        tube = valid_tube_1D(["stress"])
        tube.set_bc(valid_heatflux_bc("outer"), "outer")

        f = get_temp_hdf()
        tube.save(f)
        new = receiver.Tube.load(f)

        self.assertTrue(tube.close(new))


class TestPressureBC(unittest.TestCase):
    """
    Basic setup for PressureBC
    """

    def test_construct(self):
        bc = valid_pressure_bc()

    def test_store(self):
        f = get_temp_hdf()
        orig = valid_pressure_bc()
        orig.save(f)
        new = receiver.PressureBC.load(f)
        self.assertTrue(orig.close(new))


class TestHeatFluxBC(unittest.TestCase):
    """
    Test basic setup of a HeatFluxBC
    """

    def test_construct(self):
        bc = valid_heatflux_bc("outer")

    def test_wrong_size(self):
        with self.assertRaises(ValueError):
            bc = receiver.HeatFluxBC(
                inner_radius, height, nt, nz, times, np.zeros((1,))
            )

    def test_store(self):
        f = get_temp_hdf()
        orig = valid_heatflux_bc("outer")
        orig.save(f)
        new = receiver.ThermalBC.load(f)
        self.assertTrue(orig.close(new))


class TestFixedTempBC(unittest.TestCase):
    """
    Test basic setup of a FixedTempBC
    """

    def test_construct(self):
        bc = valid_fixedtemp_bc("outer")

    def test_wrong_size(self):
        with self.assertRaises(ValueError):
            bc = receiver.FixedTempBC(
                inner_radius, height, nt, nz, times, np.zeros((1,))
            )

    def test_store(self):
        f = get_temp_hdf()
        orig = valid_fixedtemp_bc("outer")
        orig.save(f)
        new = receiver.ThermalBC.load(f)
        self.assertTrue(orig.close(new))


class TestConvectiveBC(unittest.TestCase):
    """
    Test basic setup of a ConvectiveBC
    """

    def test_construct(self):
        bc = valid_convective_bc("outer")

    def test_wrong_size(self):
        with self.assertRaises(ValueError):
            bc = receiver.ConvectiveBC(inner_radius, height, nz, times, np.zeros((1,)))

    def test_store(self):
        f = get_temp_hdf()
        orig = valid_convective_bc("outer")
        orig.save(f)
        new = receiver.ThermalBC.load(f)
        self.assertTrue(orig.close(new))
