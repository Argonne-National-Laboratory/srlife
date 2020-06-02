import xml.etree.ElementTree as ET
from collections import ChainMap

import numpy as np
import scipy.interpolate as inter

class ThermalMaterial:
  """
    Material thermal properties.
    
    This object needs to provide:
      1) material name
      2) the conductivity, as a function of temperature and its derivative
      3) the diffusivity, as a function of temperature and its derivative
  """
  @classmethod
  def load(cls, fname):
    """
      Load from a dictionary

      Parameters:
        fname       filename
    """
    root, data = load_dict_xml(fname)

    if root == "PiecewiseLinearThermalMaterial":
      return PiecewiseLinearThermalMaterial.load(data)
    elif root == "ConstantThermalMaterial":
      return ConstantThermalMaterial.load(data)
    else:
      raise ValueError("Unknown ThermalMaterial type %s" % root)

class PiecewiseLinearThermalMaterial(ThermalMaterial):
  """
    Interpolate thermal properties linearly from a table
  """
  def __init__(self, name, temps, cond, diff):
    """
      Properties:
        name:           material name
        temps:          list of temperature points
        cond:           list of conductivity values
        diff:           list of diffusivity values
    """
    if len(temps) != len(cond) or len(temps) != len(diff):
      raise ValueError("The lists of temperatures, conductivity, and diffusivity values must have equal lengths!")
    
    self.name = name
    self.temps = np.array(temps)
    self.cond = np.array(cond)
    self.diff = np.array(diff)
    
    self.fcond, self.dfcond = make_piecewise(self.temps, self.cond)
    self.fdiff, self.dfdiff = make_piecewise(self.temps, self.diff)

  def conductivity(self, T):
    """
      Conductivity as a function of temperature

      Parameters:
        T       temperature
    """
    return self.fcond(T)

  def diffusivity(self, T):
    """
      Diffusivity as a function of temperature

      Parameters:
        T       temperature
    """
    return self.fdiff(T)

  def dconductivity(self, T):
    """
      Derivative of conductivity as a function of temperature

      Parameters:
        T       temperature
    """
    return self.dfcond(T)

  def ddiffusivity(self, T):
    """
      Derivative of diffusivity as a function of temperature

      Parameters:
        T       temperature
    """
    return self.dfdiff(T)

  def save(self, fname):
    """
      Save to an XML file

      Parameters:
        fname       filename
    """
    dictrep = {"name": self.name, "temps": string_array(self.temps),
        "cond": string_array(self.cond), "diff": string_array(self.diff)}
    save_dict_xml(dictrep, fname, "PiecewiseLinearThermalMaterial")

  @classmethod
  def load(cls, values):
    """
      Load from a dictionary

      Parameters:
        values  dictionary values
    """
    return cls(values["name"], destring_array(values["temps"]), destring_array(values["cond"]), 
        destring_array(values["diff"]))

class ConstantThermalMaterial(ThermalMaterial):
  """
    Constant thermal properties
  """
  def __init__(self, name, k, alpha):
    """
      Properties:
        name:           material name
        k:              conductivity
        alpha:          diffusivity
    """
    self.name = name
    self.cond = k
    self.diff = alpha

  def conductivity(self, T):
    """
      Conductivity as a function of temperature

      Parameters:
        T       temperature
    """
    return T * 0.0 + self.cond

  def diffusivity(self, T):
    """
      Diffusivity as a function of temperature

      Parameters:
        T       temperature
    """
    return T * 0.0 + self.diff

  def dconductivity(self, T):
    """
      Derivative of conductivity as a function of temperature

      Parameters:
        T       temperature
    """
    return T * 0.0

  def ddiffusivity(self, T):
    """
      Derivative of diffusivity as a function of temperature

      Parameters:
        T       temperature
    """
    return T * 0.0

  def save(self, fname):
    """
      Save to an XML file

      Parameters:
        fname       filename
    """
    dictrep = {"name": self.name, "k": str(self.cond), 
        "alpha": str(self.diff)}
    save_dict_xml(dictrep, fname, "PiecewiseConstantThermalMaterial")

  @classmethod
  def load(cls, values):
    """
      Load from a dictionary

      Parameters:
        values  dictionary values
    """
    return cls(values["name"], float(values["k"]), float(values["alpha"]))

class FluidMaterial:
  """
    Properties for convective heat transfer.

    This object needs to store:
      1) Fluid name
      2) A map between a ThermalMaterial name and the corresponding
         temperature-dependent film coefficient and its derivative
  """
  @classmethod
  def load(cls, fname):
    """
      Load a FluidMaterial object from a file

      Parameters:
        fname       file name to load from
    """
    root, data = load_dict_xml(fname)

    if root == "PiecewiseLinearFluidMaterial":
      return PiecewiseLinearFluidMaterial.load(data)
    elif root == "ConstantFluidMaterial":
      return ConstantFluidMaterial.load(data)
    else:
      raise ValueError("Unknown FluidMaterial type %s" % root)

class ConstantFluidMaterial:
  """
    Supply a mapping between the material type and a constant
    film coefficient
  """
  def __init__(self, data):
    """
      Dictionary of the form {name: value} mapping
      a material name to the definition of the piecewise linear map

      Parameters:
        data:       the dictionary
    """
    self.data = data

  def save(self, fname):
    """
      Save to an XML file

      Parameters:
        fname       file name to save to
    """
    dictrep = {k: str(v) for k, v in self.data.items()} 
    save_dict_xml(dictrep, fname, "ConstantFluidMaterial")

  @classmethod
  def load(cls, values):
    """
      Load from a dictionary

      Parameters:
        values      dictionary data
    """
    data = {k: float(val) for k, val in values.items()}
    return cls(data)

  def coefficient(self, material, T):
    """
      Return the film coefficient for the given material and temperature

      Parameters:
        material:       material name
        T:              temperatures
        
    """
    return T*0.0 + self.data[material]

  def dcoefficient(self, material, T):
    """
      Return the derivative of the film coefficient with respect to
      temperature for the give material and temperature.

      Parameters:
        material:       material name
        T:              temperatures
    """
    return T * 0.0

class PiecewiseLinearFluidMaterial:
  """
    Supply a mapping between the material type and a piecewise linear
    interpolate defining the film coefficient as a function of temperature.
  """
  def __init__(self, data):
    """
      Dictionary of the form {name: (temperatures, values)} mapping
      a material name to the definition of the piecewise linear map

      Parameters:
        data:       the dictionary
    """
    self.data = data

    self.fns = {name: make_piecewise(T, v) for name, (T,v) in data.items()}

  def save(self, fname):
    """
      Save to an XML file

      Parameters:
        fname       file name to save to
    """
    dictrep = {k: {'temp': string_array(T), 'values': string_array(v)} for k, (T, v) in self.data.items()} 
    save_dict_xml(dictrep, fname, "PiecewiseLinearFluidMaterial")

  @classmethod
  def load(cls, values):
    """
      Load from a dictionary

      Parameters:
        values      dictionary data
    """
    data = {k: (destring_array(pair['temp']), destring_array(pair['values'])) for k, pair in values.items()}
    return cls(data)

  def coefficient(self, material, T):
    """
      Return the film coefficient for the given material and temperature

      Parameters:
        material:       material name
        T:              temperatures
        
    """
    return self.fns[material][0](T)

  def dcoefficient(self, material, T):
    """
      Return the derivative of the film coefficient with respect to
      temperature for the give material and temperature.

      Parameters:
        material:       material name
        T:              temperatures
    """
    return self.fns[material][1](T)

def make_piecewise(x, y):
  """
    Make two piecewise interpolation functions: a piecewise linear
    interpolate between x and y and the corresponding derivative.
  """
  ydiff = np.zeros(y.shape)
  ydiff[:-1] = np.diff(y) / np.diff(x)
  ydiff[-1] = ydiff[-2]

  return inter.interp1d(x, y), inter.interp1d(x, ydiff, kind = "previous")

def save_dict_xml(data, fname, rootname = "data"):
  """
    Dump a python dictionary to file, recursion is allowed

    Parameters:
      data:     dictionary of data (can be recursive)
      fname:    filename to use

    Other Parameters:
      rootname: what to call the root node
  """
  root = ET.Element(rootname)
  for k,v in data.items():
    save_node(k, v, root)

  et = ET.ElementTree(element = root)
  et.write(fname)

def save_node(name, entry, node):
  """
    Save a dictionary to a particular node

    Parameters:
      name:     name of the new node
      entry:    entry of interest
      node:     ET parent node object
  """
  nnode = ET.Element(name)
  if isinstance(entry, dict):
    for k,v in entry.items():
      save_node(k, v, nnode)
  else:
    nnode.text = entry
  node.append(nnode)

def load_dict_xml(fname):
  """
    Load a python dictionary from a file, recursion is allowed

    Parameters:
      fname:        file with the data
  """
  tree = ET.parse(fname)
  rootname = tree.getroot().tag
  return rootname, load_node(tree.getroot())[rootname]

def load_node(node):
  """
    The actual function that does the loading by walking the XML file

    Parameters:
      node:      xml node object from ET
  """
  if len(node) > 0:
    return {node.tag: dict(ChainMap(*(load_node(child) for child in node)))}
  else:
    return {node.tag: node.text}

def string_array(array):
  """
    Make a numpy array a space separated string
  """
  return " ".join(map(str, array))

def destring_array(string):
  """
    Make an array from a space separated string
  """
  return np.array(list(map(float, string.split(" "))))
