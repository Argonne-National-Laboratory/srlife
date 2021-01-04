"""
  This module define the data structures used as input and output to the analysis module.

  The require input data can be provided by constructing a Receiver object in python or by 
  loading an HDF5 datafile, which will populate the python class hierarchy.
"""

import itertools

import numpy as np
import scipy.interpolate as inter
import h5py

from srlife import writers

class Receiver:
  """ Basic definition of the tubular receiver geometry.

    A receiver is a collection of panels linked together by
    an elastic spring stiffness.  This stiffness can be a real number,
    "rigid" or "disconnect"

    Panels can be labeled by strings.  By default the names
    are sequential numbers.

    In addition this object stores some required metadata:
      1) The daily cycle period (which can be less than 24 hours
         if the analysis neglects some of the night period)
      2) The number of days (see #1) explicitly represented in the
         analysis results.

    Args:
      period (float): single daily cycle period
      days (int): number of daily cycles explicitly represented
      panel_stiffness (float or string): panel stiffness (float) or "rigid" or "disconnect"
  """
  def __init__(self, period, days, panel_stiffness):
    """ Initialize a Receiver object
    """
    self.period = period
    self.days = days
    self.panels = {}
    self.stiffness = panel_stiffness

  def write_vtk(self, basename):
    """  Write out the receiver as individual panels with names basename_panelname

    The VTK format is mostly used for additional postprocessing.  The VTK
    format cannot be used for input.

    Args:
      basename (str):  base file name
    """
    for n, panel in self.panels.items():
      panel.write_vtk(basename + "_" + n)

  def close(self, other):
    """ Check to see if two objects are nearly equal.

      Primarily used for testing

      Args:
        other (Receiver):      the object to compare against

      Returns:
        bool:   True if the receivers are similar.
    """
    base = (
        np.isclose(self.period, other.period)
        and np.isclose(self.days, other.days)
        and np.isclose(self.stiffness, other.stiffness)
        )
    for name, panel in self.panels.items():
      if name not in other.panels:
        return False
      base = (base and panel.close(other.panels[name]))

    return base
  
  @property
  def tubes(self):
    """ Shortcut iterator over all tubes

      Returns:
        iterator over panels
    """
    return itertools.chain(*(panel.tubes.values() 
      for panel in self.panels.values()))

  @property
  def ntubes(self):
    """ Shortcut for total number of tubes

    Returns:
      int: Number of tubes in all panels
    """
    return len(list(self.tubes))

  @property
  def npanels(self):
    """ Number of panels in the receiver

    Returns:
      int:  Number of panels
    """
    return len(self.panels)

  def add_panel(self, panel, name = None):
    """ Add a panel object to the receiver

      Args:
        panel (Panel):          panel object
        name (Optional[str]):   panel name, by default follows fixed scheme
      """
    if not name:
      name = next_name(self.panels.keys())

    self.panels[name] = panel

  def save(self, fobj):
    """ Save to an HDF5 file

      This saves a Receiver object to the HDF5 format.

      Args:
        fobj (str):  either a h5py file object or a filename
    """
    if isinstance(fobj, str):
      fobj = h5py.File(fobj, 'w')

    fobj.attrs['period'] = self.period
    fobj.attrs['days'] = self.days
    fobj.attrs['stiffness'] = self.stiffness

    grp = fobj.create_group("panels")

    for name, panel in self.panels.items():
      sgrp = grp.create_group(name)
      panel.save(sgrp)

  @classmethod
  def load(cls, fobj):
    """ Load a Receiver from an HDF5 file

      A full description of the HDF format is included in the module documentation

      Args:
        fobj (string):  either a h5py file object or a filename

      Returns:
        Receiver: The constructed receiver object.
    """
    if isinstance(fobj, str):
      fobj = h5py.File(fobj, 'r')

    res = cls(fobj.attrs['period'], fobj.attrs['days'], fobj.attrs['stiffness'])

    grp = fobj["panels"]

    for name in grp:
      res.add_panel(Panel.load(grp[name]), name)

    return res

class Panel:
  """ Basic definition of a panel in a tubular receiver.

    A panel is a collection of Tube object linked together by
    an elastic spring stiffness.  This stiffness can be a real number,
    a string "disconnect" or a string "rigid"

    Tubes in the panel can be labeled by strings.  By default the
    names are sequential numbers.

    Args:
      stiffness:       manifold spring stiffness
  """
  def __init__(self, stiffness):
    """ Initialize the panel
    """
    self.tubes = {}
    self.stiffness = stiffness

  def write_vtk(self, basename):
    """ Write out the panels as individual tubes with names basename_tubename

      Args:
        basename (string): base file name
    """
    for n, tube in self.tubes.items():
      tube.write_vtk(basename + "_" + n)

  def close(self, other):
    """ Check to see if two objects are nearly equal.

      Primarily used for testing

      Args:
        other (Panel): the object to compare against

      Returns:
        bool: true if the panels are sufficiently similar
    """
    base = np.isclose(self.stiffness, other.stiffness)
    for name, tube in self.tubes.items():
      if name not in other.tubes:
        return False
      base = (base and tube.close(other.tubes[name]))

    return base

  @property
  def ntubes(self):
    """ Number of tubes in the panel

    Returns:
      int:  number of tubes in the panel
    """
    return len(self.tubes)

  def add_tube(self, tube, name = None):
    """ Add a tube object to the panel

      Args:
        tube (Tube): tube object
        name (Optional[str]): Tube name, defaults to fixed scheme.
    """
    if not name:
      name = next_name(self.tubes.keys())

    self.tubes[name] = tube

  def save(self, fobj):
    """ Save to an HDF5 file

      Args:
        fobj (h5py.Group):  h5py group
    """
    fobj.attrs['stiffness'] = self.stiffness

    grp = fobj.create_group("tubes")

    for name, tube in self.tubes.items():
      sgrp = grp.create_group(name)
      tube.save(sgrp)

  @classmethod
  def load(cls, fobj):
    """ Load from an HDF5 file

      Args:
        fobj (h5py.Group):  h5py group containing the panel
    """
    res = cls(fobj.attrs['stiffness'])

    grp = fobj["tubes"]

    for name in grp:
      res.add_tube(Tube.load(grp[name]), name)

    return res

def next_name(names):
  """ Determine the next numeric string name based on a list

    Args:
      names (list): list of current names (string)
  """
  curr_ints = []
  for name in names:
    try:
      curr_ints.append(int(name))
    except ValueError:
      continue

  if len(curr_ints) == 0:
    return str(0)
  return str(max(curr_ints) + 1)

class Tube:
  """ Geometry, boundary conditions, and results for a single tube.

    The basic tube geometry is defined by an outer radius, thickness, and
    height.

    Results are given at fixed times and
    on a regular polar grid defined by a number of
    r, theta, and z increments.  The grid points are then deduced by
    linear subdivision in r between the outer radius and the
    outer radius - t, 0 to 2 pi, and 0 to the tube height.

    Result fields are general and provided by a list of names.
    The receiver package uses the metal temperatures, stresses, 
    mechanical strains, and inelastic strains.

    Analysis results can be provided over the full 3D grid (default),
    a single 2D plane (identified by a height), or a single 1D line
    (identified by a height and a theta position)

    Boundary conditions may be provided in two ways, either
    as fluid conditions or net heat fluxes.  These are defined
    in the HeatFluxBC or ConvectionBC objects below.

    Args:
      outer_radius (float): tube outer radius
      thickness (float): tube thickness
      height (float): tube height
      nr (int): number of radial increments
      nt (int): number of circumferential increments
      nz (int): number of axial increments
      T0 (Optional[float]): initial temperature
  """
  def __init__(self, outer_radius, thickness, height, nr, nt, nz, 
      T0 = 0.0):
    """ Initialize the tube
    """
    self.r = outer_radius
    self.t = thickness
    self.h = height

    self.nr = nr
    self.nt = nt
    self.nz = nz

    self.abstraction = "3D"

    self.times = []
    self.results = {}
    self.quadrature_results = {}

    self.outer_bc = None
    self.inner_bc = None
    self.pressure_bc = None

    self.T0 = T0

  def copy_results(self, other):
    """ Copy the results fields from one tube to another

      Parameters:
        other:      other tube object
    """
    self.results = other.results
    self.quadrature_results = other.quadrature_results

  @property
  def ndim(self):
    """ Number of problem dimensions

    Returns:
      int:  tube dimension
    """
    if self.abstraction == "3D":
      return 3
    elif self.abstraction == "2D":
      return 2
    elif self.abstraction == "1D":
      return 1
    else:
      raise ValueError("Tube abstraction unknown!")

  @property
  def dim(self):
    """ Actual problem discretization

    Returns:
      tuple(int): tuple giving the fixed grid discretization
    """
    if self.abstraction == "3D":
      return (self.nr, self.nt, self.nz)
    elif self.abstraction == "2D":
      return (self.nr, self.nt, 1)
    elif self.abstraction == "1D":
      return (self.nr, 1, 1)
    else:
      raise ValueError("Tube abstraction unknown!")

  @property
  def mesh(self):
    """ Calculate the problem mesh (should only be needed for I/O)

    Returns:
      Results of np.meshgrid over the problem discretization
    """
    r = np.linspace(self.r-self.t, self.r, self.nr)
    if self.ndim > 1:
      t = np.linspace(0, 2*np.pi, self.nt+1)[:self.nt]
    else:
      t = [self.angle]
    if self.ndim > 2:
      z = np.linspace(0, self.h, self.nz)
    else:
      z = [self.plane]

    return np.meshgrid(*[r,t,z], indexing = 'ij')

  def write_vtk(self, fname):
    """ Write to a VTK file

      The tube VTK files are only used for output and 
      postprocessign

      Args:
        fname (string): base filename
    """
    writer = writers.VTKWriter(self, fname)
    writer.write()

  def make_2D(self, height):
    """ Abstract the tube as 2D

      Reduce to a 2D abstraction by slicing the tube at the
      indicated height

      Args:
        height (float): the height at which to slice
    """
    if height < 0.0 or height > self.h:
      raise ValueError("2D slice height must be within the tube height")

    self.abstraction = "2D"
    self.plane = height

  def make_1D(self, height, angle):
    """ Abstract the tube as 1D

      Reduce to a 1D abstraction along a ray given by the provided
      height and angle.

      Args:
        height (float): the height of the ray
        angle (float): the angle, in radians
    """
    if height < 0.0 or height > self.h:
      raise ValueError("Ray height must be within the tube height")

    self.abstraction = "1D"
    self.plane = height
    self.angle = angle

  def close(self, other):
    """ Check to see if two objects are nearly equal.

      Primarily used for testing

      Args:
        other (Tube): the object to compare against

      Returns:
        bool: true if the tubes are similar
    """
    base = (
        np.isclose(self.r, other.r)
        and np.isclose(self.t, other.t)
        and np.isclose(self.h, other.h)
        and (self.nr == other.nr)
        and (self.nt == other.nt)
        and (self.nz == other.nz)
        and (np.allclose(self.times, other.times)))

    for name, data in self.results.items():
      if name not in other.results:
        return False
      base = (base and np.allclose(data, other.results[name]))
    
    if self.outer_bc:
      if not other.outer_bc:
        return False
      base = (base and self.outer_bc.close(other.outer_bc))

    if self.inner_bc:
      if not other.inner_bc:
        return False
      base = (base and self.inner_bc.close(other.inner_bc))

    if self.pressure_bc:
      if not other.pressure_bc:
        return False
      base = (base and self.pressure_bc.close(other.pressure_bc))

    base = (base and self.abstraction == other.abstraction)
    if self.abstraction == "2D" or self.abstraction == "1D":
      base = (base and np.isclose(self.plane, other.plane))
    if self.abstraction == "1D":
      base = (base and np.isclose(self.angle, other.angle))

    base = (base and np.isclose(self.T0, other.T0))

    return base

  @property
  def ntime(self):
    """ Number of time steps

    Returns:
      int:  number of time steps
    """
    return len(self.times)

  def set_times(self, times):
    """ Set the times at which data is provided

      All results arrays must provide data at these 
      discrete times.

      Args:
        times (np.array): time values
    """
    for _,res in self.results.items():
      if res.shape[0] != len(times):
        raise ValueError("Cannot change times to provided values, will be"
            " incompatible with existing results")
    self.times = times

  def add_results(self, name, data):
    """ Add a node point result field

      Args:
        name (str): parameter set name
        data (np.array): actual results data
    """
    self._check_rdim(data)
    self.results[name] = data

  def add_quadrature_results(self, name, data):
    """ Add a result at the quadrature points

      Args:
        name (str): parameter set name
        data (np.array): actual results data
    """
    if data.shape[0] != self.ntime:
      raise ValueError("Quadrature data must have time axis first!")
    self.quadrature_results[name] = data

  def _check_rdim(self, data):
    """ Verify the dimensions of a results array

      Make sure the results array aligns with the correct dimension for the
      abstraction

      Args:
        data (np.array): data array

      Raises:
        ValueError: If the data array shape is not correct for the problem dimensions
    """
    if self.abstraction == "3D":
      if data.shape != (self.ntime, self.nr, self.nt, self.nz):
        raise ValueError("Data array shape must equal ntime x nr x nt x nz!")
    elif self.abstraction == "2D":
      if data.shape != (self.ntime, self.nr, self.nt):
        raise ValueError("Data array shape must equal ntime x nr x nt!")
    elif self.abstraction == "1D":
      if data.shape != (self.ntime, self.nr):
        raise ValueError("Data array shape must equal ntime x nr!")
    else:
      raise ValueError("Internal error: unknown abstraction type %s" % 
          self.abstraction)

  def set_bc(self, bc, loc):
    """ Set the inner or outer heat flux BC

      Args:
        bc (ThermalBC):  boundary condition object
        loc (string): location -- either "inner" or "outer" wall
    """
    if loc == "inner":
      if not np.isclose(bc.r, self.r - self.t) or not np.isclose(bc.h, self.h):
        raise ValueError("Inner BC radius must match inner tube radius!")
      self.inner_bc = bc
    elif loc == "outer":
      if not np.isclose(bc.r, self.r) or not np.isclose(bc.h, self.h):
        raise ValueError("Outer BC radius must match outer tube radius!")
      self.outer_bc = bc
    else:
      raise ValueError("Wall location must be either inner or outer")

  def set_pressure_bc(self, bc):
    """ Set the pressure boundary condition

      Args:
        bc (PressureBC):  boundary condition object
    """
    self.pressure_bc = bc

  def save(self, fobj):
    """ Save to an HDF5 file

      Args:
        fobj (h5py.Group):  h5py group to save to
    """
    fobj.attrs["r"] = self.r
    fobj.attrs["t"] = self.t
    fobj.attrs["h"] = self.h

    fobj.attrs["nr"] = self.nr
    fobj.attrs["nt"] = self.nt
    fobj.attrs["nz"] = self.nz

    fobj.attrs["abstraction"] = self.abstraction
    if self.abstraction == "2D" or self.abstraction == "1D":
      fobj.attrs["plane"] = self.plane
    if self.abstraction == "1D":
      fobj.attrs["angle"] = self.angle

    fobj.create_dataset("times", data = self.times)

    grp = fobj.create_group("results")
    for name, result in self.results.items():
      grp.create_dataset(name, data = result)

    grp = fobj.create_group("quadrature_results")
    for name, result in self.quadrature_results.items():
      grp.create_dataset(name, data = result)

    if self.outer_bc:
      grp = fobj.create_group("outer_bc")
      self.outer_bc.save(grp)

    if self.inner_bc:
      grp = fobj.create_group("inner_bc")
      self.inner_bc.save(grp)

    if self.pressure_bc:
      grp = fobj.create_group("pressure_bc")
      self.pressure_bc.save(grp)

    fobj.attrs["T0"] = self.T0

  @classmethod
  def load(cls, fobj):
    """ Load from an HDF5 file

      Parameters:
        fobj (h5py.Group):  h5py to load from
    """
    res = cls(fobj.attrs["r"], fobj.attrs["t"], fobj.attrs["h"], fobj.attrs["nr"], fobj.attrs["nt"],
        fobj.attrs["nz"], T0 = fobj.attrs["T0"])

    res.abstraction = fobj.attrs["abstraction"]
    if res.abstraction == "2D" or res.abstraction == "1D":
      res.plane = fobj.attrs["plane"]
    if res.abstraction == "1D":
      res.angle = fobj.attrs["angle"]

    res.set_times(np.copy(fobj["times"]))

    grp = fobj["results"]
    for name in grp:
      res.add_results(name, np.copy(grp[name]))

    grp = fobj["quadrature_results"]
    for name in grp:
      res.add_quadrature_results(name, np.copy(grp[name]))

    if "outer_bc" in fobj:
      res.set_bc(ThermalBC.load(fobj["outer_bc"]), "outer")

    if "inner_bc" in fobj:
      res.set_bc(ThermalBC.load(fobj["inner_bc"]), "inner")

    if "pressure_bc" in fobj:
      res.set_pressure_bc(PressureBC.load(fobj["pressure_bc"]))

    return res

def _vector_interpolate(base, data):
  """ Interpolate as a vector

  Args:
    base (function): base interpolation method
    data (np.array): data to interpolate
  """
  res = np.zeros(data[0].shape)
  
  # pylint: disable=not-an-iterable
  for ind in np.ndindex(*res.shape):
    res[ind] = base([d[ind] for d in data])

  return res

def _make_ifn(base):
  """ Helper to deal with getting both a scalar and a vector input

  Args:
    base (function): base interpolation method

  Returns:
    function: a function that interpolates a vector using base at each component
  """
  def ifn(mdata):
    """
      Interpolation function that handles both the scalar and vector cases
    """
    allscalar = all(map(np.isscalar, mdata))
    anyscalar = any(map(np.isscalar, mdata))
    if allscalar:
      return base(mdata)
    elif anyscalar:
      shapes = [a.shape for a in mdata if not np.isscalar(a)]
      # Could check they are all the same, but eh
      shape = shapes[0]
      ndata = [np.ones(shape) * d for d in mdata]
      return _vector_interpolate(base, ndata)
    else:
      return _vector_interpolate(base, ndata)

  return ifn

class PressureBC:
  """ Stores information about the tube internal pressure

    Simple class to store tube pressure, assumed to be constant
    in space and just vary with time.

    Args:
      times (np.array): times throughout load cycle
      data (np.array):  pressure values
  """
  def __init__(self, times, data):
    """ Initialize the PressureBC
    """
    self.times = times
    if self.times.shape != data.shape:
      raise ValueError("Times and data should have the same shape!")
    self.data = data

    self.ifn = inter.interp1d(self.times, self.data)

  @classmethod
  def load(cls, fobj):
    """ Load from an HDF5 file

      Args:
        fobj (h5py.Group):  h5py group to load from
    """
    return cls(np.copy(fobj["times"]), np.copy(fobj["data"]))

  def save(self, fobj):
    """ Save to an HDF5 file

      Args:
        fobj (h5py.Group):  h5py group to save to
    """
    fobj.create_dataset("times", data = self.times)
    fobj.create_dataset("data", data = self.data)

  @property
  def ntime(self):
    """ Number of time steps

    Returns:
      int:      number of time steps in the pressure history definition
    """
    return len(self.times)

  def pressure(self, t):
    """ Return the pressure as a function of time

    Args:
      t (float): time

    Returns:
      float: internal pressure at that time
    """
    return self.ifn(t)

  def close(self, other):
    """ Test method for comparing BCs

      Args:
        other (PressureBC): other object

      Returns:
        bool: true if the objects are sufficiently similar
    """
    return (np.allclose(self.times, other.times)
        and np.allclose(self.data, other.data))

class ThermalBC:
  """ Superclass for thermal boundary conditions.

    Currently just here to handle dispatch for HDF files
  """
  @classmethod
  def load(cls, fobj):
    """ Load from an HDF5 file

      Args:
        fobj (h5py.Group): h5py group to load from
    """
    if fobj.attrs["type"] == "HeatFlux":
      return HeatFluxBC.load(fobj)
    elif fobj.attrs["type"] == "Convective":
      return ConvectiveBC.load(fobj)
    elif fobj.attrs["type"] == "FixedTemp":
      return FixedTempBC.load(fobj)
    else:
      raise ValueError("Unknown BC type %s" % fobj.attrs["type"])
  
  # pylint: disable=no-member
  def _generate_surface_mesh(self):
    """ Make the finite difference mesh for the BC

      Generate the appropriate finite difference surface
      mesh for a particular problem

      Returns:
        np.array:   discrete times
        np.array:   theta coordinates
        np.array:   z coordinates
    """
    ts = np.linspace(0, 2*np.pi, self.nt + 1)[:-1]
    zs = np.linspace(0, self.h, self.nz)
    
    return self.times, ts, zs

  def _generate_ifn(self, data):
    """ Generate an interpolation function for the given data array

      Args:
        data (np.array):  (ntime, ntheta, nz) shaped array

      Returns:
        function: appropriate interpolation function
    """
    base = inter.RegularGridInterpolator(self._generate_surface_mesh(),
        data, method = "linear", bounds_error = False, fill_value = None)

    return _make_ifn(base)

class HeatFluxBC(ThermalBC):
  """
    A net heat flux on the radius of a tube.  Positive is heat input,
    negative is heat output.

    These conditions are defined on the surface of a tube at fixed
    times given by
    a radius and a height.  The radius is not used in defining the 
    BC but is used to ensure the BC is consistent with the Tube object.

    The heat flux is given on a regular grid of theta, z points each defined
    but a number of increments.  This grid need not agree with the Tube
    solid grid.

    Args:
      radius (float): boundary condition application radius
      height (float): tube height
      nt (int): number of circumferential increments
      nz (int): number of axial increments
      times (np.array): heat flux times
      data (np.array): heat flux data
  """
  def __init__(self, radius, height, nt, nz, times, data):
    self.r = radius
    self.h = height

    self.nt = nt
    self.nz = nz

    self.times = times

    if data.shape != (len(self.times), nt, nz):
      raise ValueError("Heat flux shape must equal ntime x ntheta x nz!")

    self.data = data

    self.ifn = self._generate_ifn(self.data)

  @property
  def ntime(self):
    """ Number of time steps

    Returns:
      int:  number of time steps
    """
    return len(self.times)

  def flux(self, t, theta, z):
    """ Flux as a function of time, angle, and height

      Args:
        t (float): time
        theta (float): angle
        z (float): height

      Returns:
        float:  the flux value at this time and location
    """
    return self.ifn([t, theta, z])

  def save(self, fobj):
    """ Save to an HDF5 file

      Args:
        fobj (h5py.Group): h5py group to save to
    """
    fobj.attrs["type"] = "HeatFlux"
    fobj.attrs["r"] = self.r
    fobj.attrs["h"] = self.h

    fobj.attrs["nt"] = self.nt
    fobj.attrs["nz"] = self.nz

    fobj.create_dataset("times", data = self.times)
    fobj.create_dataset("data", data = self.data)

  @classmethod
  def load(cls, fobj):
    """ Load from an HDF5 file

      Args:
        fobj (h5py.Group): h5py group to load from
    """
    return cls(fobj.attrs["r"], fobj.attrs["h"], fobj.attrs["nt"], fobj.attrs["nz"],
        np.copy(fobj["times"]), np.copy(fobj["data"]))

  def close(self, other):
    """ Check to see if two objects are nearly equal.

      Primarily used for testing

      Args:
        other (HeatFluxBC): the object to compare against

      Returns:
        bool: returns true if the objects are similar
    """
    return (
        np.isclose(self.r, other.r)
        and np.isclose(self.h, other.h)
        and (self.nt == other.nt)
        and (self.nz == other.nz)
        and np.allclose(self.times, other.times)
        and np.allclose(self.data, other.data)
        )

class FixedTempBC(ThermalBC):
  """ Fixed temperature BC.

    These conditions are defined on the surface of a tube at fixed
    times given by
    a radius and a height.  The radius is not used in defining the 
    BC but is used to ensure the BC is consistent with the Tube object.

    The heat flux is given on a regular grid of theta, z points each defined
    but a number of increments.  This grid need not agree with the Tube
    solid grid.

    Args:
      radius (float): boundary condition application radius
      height (float): tube height
      nt (int): number of circumferential increments
      nz (int): number of axial increments
      times (np.array): fixed temperature times
      data (np.array): fixed temperature data
  """
  def __init__(self, radius, height, nt, nz, times, data):
    self.r = radius
    self.h = height

    self.nt = nt
    self.nz = nz

    self.times = times

    if data.shape != (len(self.times), nt, nz):
      raise ValueError("Discrete temperature shape must equal ntime x ntheta x nz!")

    self.data = data

    self.ifn = self._generate_ifn(self.data)

  @property
  def ntime(self):
    """ Number of time steps

    Returns:
      int:  number of time steps
    """
    return len(self.times)

  def temperature(self, t, theta, z):
    """ Return the temperature at a given time and position

      Args:
        t (float): time
        theta (float): angle
        z (float):  height

      Returns:
        float: Fixed temperature at given time/location
    """
    return self.ifn([t, theta, z])

  def save(self, fobj):
    """ Save to an HDF5 file

      Args:
        fobj (h5py.Group):  h5py group to save to
    """
    fobj.attrs["type"] = "FixedTemp"
    fobj.attrs["r"] = self.r
    fobj.attrs["h"] = self.h

    fobj.attrs["nt"] = self.nt
    fobj.attrs["nz"] = self.nz

    fobj.create_dataset("times", data = self.times)
    fobj.create_dataset("data", data = self.data)

  @classmethod
  def load(cls, fobj):
    """ Load from an HDF5 file

      Args:
        fobj (h5py.Group): h5py group to load from
    """
    return cls(fobj.attrs["r"], fobj.attrs["h"], fobj.attrs["nt"], fobj.attrs["nz"],
        np.copy(fobj["times"]), np.copy(fobj["data"]))

  def close(self, other):
    """ Check to see if two objects are nearly equal.

      Primarily used for testing

      Args:
        other (FixedTempBC): the object to compare against

      Returns:
        bool: true if the objects are similar
    """
    return (
        np.isclose(self.r, other.r)
        and np.isclose(self.h, other.h)
        and (self.nt == other.nt)
        and (self.nz == other.nz)
        and np.allclose(self.times, other.times)
        and np.allclose(self.data, other.data)
        )

class ConvectiveBC(ThermalBC):
  """ A convective BC on the surface of a tube defined by a radius and height.

    The radius is not used in defining the BC, but is used to check
    consistency with the Tube object.

    This condition is defined axially by a fluid temperature at 
    fixed times on a fixed grid of z points defined by a number of 
    increments.

    Args:
      radius (float): radius of application
      height (float): height of fluid temperature info
      nz (int): number of axial increments
      times (np.array): data times
      data (np.array): actual fluid temperature data
  """
  def __init__(self, radius, height, nz, times, data):
    self.r = radius
    self.h = height

    self.nz = nz

    self.times = times

    if data.shape != (len(self.times), nz):
      raise ValueError("Fluid temperature data shape must equal "
          "ntime x nz!")

    self.data = data

    zs = np.linspace(0, self.h, self.nz)
    base = inter.RegularGridInterpolator((self.times, zs), self.data, 
        bounds_error=False, fill_value = None, method = 'linear')

    self.ifn = _make_ifn(base)

  @property
  def ntime(self):
    """ Number of time steps

    Returns:
      int:  number of discrete time steps
    """
    return len(self.times)

  def fluid_temperature(self, t, z):
    """ Return the fluid temperature at a given time and position

      Args:
        t (float): time
        z (float): height

      Return:
        float: fluid temperature at this location and time
    """
    return self.ifn([t, z])

  def save(self, fobj):
    """ Save to an HDF5 file

      Args:
        fobj (h5py.Group): h5py group to save to
    """
    fobj.attrs["type"] = "Convective"
    fobj.attrs["r"] = self.r
    fobj.attrs["h"] = self.h

    fobj.attrs["nz"] = self.nz

    fobj.create_dataset("times", data = self.times)

    fobj.create_dataset("data", data = self.data)

  @classmethod
  def load(cls, fobj):
    """ Load from an HDF5 file

      Args:
        fobj (h5py.Group): h5py group to load from
    """
    return cls(fobj.attrs["r"], fobj.attrs["h"], fobj.attrs["nz"], 
        np.copy(fobj["times"]), np.copy(fobj["data"]))

  def close(self, other):
    """ Check to see if two objects are nearly equal.

      Primarily used for testing

      Args:
        other (ConvectiveBC): the object to compare against

      Returns:
        bool: true if sufficiently similar
    """
    return (
        np.isclose(self.r, other.r) 
        and np.isclose(self.h, other.h)
        and (self.nz == other.nz)
        and np.allclose(self.times, other.times)
        and np.allclose(self.data, other.data)
        )
