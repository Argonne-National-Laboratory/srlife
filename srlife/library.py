"""
  Module that facilitates loading in all the material data
"""
import os.path

from srlife import materials
from neml import parse, models

LIBRARY_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data'))


def get_file(directory, name):
  """
    Return a file path or raise an error
  """
  attempt = os.path.join(directory, name+".xml")
  if not os.path.exists(attempt):
    raise RuntimeError("Material with name %s does not exists in database!" 
        % name)
  return attempt

def load_fluid(name, model):
  """
    Load fluid material properties:

    Parameters:
      name      name of the fluid (title of xml file)
      model     particular convection model to use
  """
  fdir = os.path.join(LIBRARY_DIR, "fluid")
  filename = get_file(fdir, name)
  return materials.FluidMaterial.load(filename, model)

def load_material(name, thermal_model, deformation_model, damage_model):
  """
    Load solid material properties

    Parameters:
      name              name of the material (title of xml file)
      thermal_model     which thermal model variant to use
      deformation_model which deformation model variant to use
      damage_model      which damage model variant to use
  """
  return load_thermal(name, thermal_model), load_deformation(
      name, deformation_model), load_damage(name, damage_model)

def load_thermal(name, model):
  """
    Load thermal material data

    Parameters:
      name          name of the material
      model         model variant
  """
  fdir = os.path.join(LIBRARY_DIR, "thermal")
  filename = get_file(fdir, name)
  return materials.ThermalMaterial.load(filename, model)

def load_deformation(name, model):
  """
    Load a deformation model from file

    Parameters:
      name          name of the material
      model         model variant
  """
  fdir = os.path.join(LIBRARY_DIR, "deformation")
  filename = get_file(fdir, name)
  return materials.DeformationMaterial(filename, model)

def load_damage(name, model):
  """
    Load a damage model from file

    Parameters:
      name          name of the material
      model         model variant
  """
  fdir = os.path.join(LIBRARY_DIR, "damage")
  filename = get_file(fdir, name)
  return materials.StructuralMaterial.load(filename, model)
