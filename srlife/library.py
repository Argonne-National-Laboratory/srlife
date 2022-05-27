"""
  Module that facilitates loading in all the material data
"""
import os.path

import xml.etree.ElementTree as ET

from srlife import materials

LIBRARY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))


def get_file(directory, name):
    """Return a file path or raise an error

    Args:
      directory (str): directory the file should be in
      name (str): actual file name

    Raises:
      RunetimeError: if the file does not exist
    """
    attempt = os.path.join(directory, name + ".xml")
    if not os.path.exists(attempt):
        raise RuntimeError("Material with name %s does not exists in database!" % name)
    return attempt


def load_fluid(name, model):
    """Load fluid material properties:

    Args:
      name (str): name of the fluid (title of xml file)
      model (str):  particular convection model to use

    Returns:
      material.FluidMaterial: fluid material model
    """
    fdir = os.path.join(LIBRARY_DIR, "fluid")
    filename = get_file(fdir, name)
    return materials.FluidMaterial.load(filename, model)


def load_material(name, thermal_model, deformation_model, damage_model):
    """Load solid material properties

    Args:
      name (str): name of the material (title of xml file)
      thermal_model (str): which thermal model variant to use
      deformation_model (str): which deformation model variant to use
      damage_model (str): which damage model variant to use

    Returns:
      materials.ThermalMaterial: thermal material model
      materials.DeformationMaterial: deformation material model
      materials.StructuralMaterial: damage material model
    """
    return (
        load_thermal(name, thermal_model),
        load_deformation(name, deformation_model),
        load_damage(name, damage_model),
    )


def load_thermal(name, model):
    """Load thermal material data

    Args:
      name: name of the material
      model: model variant

    Returns:
      materials.ThermalMaterial: thermal material model
    """
    fdir = os.path.join(LIBRARY_DIR, "thermal")
    filename = get_file(fdir, name)
    return materials.ThermalMaterial.load(filename, model)


def load_deformation(name, model):
    """Load a deformation model from file

    Args:
      name: name of the material
      model: model variant

    Returns:
      materials.DeformationMaterial: deformation material model
    """
    fdir = os.path.join(LIBRARY_DIR, "deformation")
    filename = get_file(fdir, name)
    return materials.DeformationMaterial(filename, model)


def load_damage(name, model):
    """Load a damage model from file

    Args:
      name: name of the material
      model: model variant

    Returns:
      materials.StructuralMaterial: damage material model
    """
    fdir = os.path.join(LIBRARY_DIR, "damage")
    filename = get_file(fdir, name)

    mat_type = get_type(filename)

    if mat_type == "metallic":
        return materials.StructuralMaterial.load(filename, model)
    elif mat_type == "ceramic":
        return materials.CeramicMaterial.load(filename, model)
    else:
        raise ValueError("Unknown material type %s in XML damage file." % mat_type)


def get_type(filename):
    """
    Report if this is a metallic or ceramic material
    """
    return ET.parse(filename).getroot().attrib["type"]
