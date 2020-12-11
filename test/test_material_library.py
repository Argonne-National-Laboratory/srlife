import unittest

from srlife import library

fluids = ['salt']
alloys = ['316H','800H','A617','740H','A282','A230']

class TestMaterials(unittest.TestCase):
  def test_load_materials(self):
    for alloy in alloys:
      thermal, deformation, damage = library.load_material(alloy, "base","base","base")
      deformation.get_neml_model()
      thermal, deformation, damage = library.load_material(alloy, "base","elastic_creep","base")
      deformation.get_neml_model()
      thermal, deformation, damage = library.load_material(alloy, "base","elastic_model","base")
      deformation.get_neml_model()

    def test_load_fluids(self):
      for fluid in fluids:
        library.load_fluid(fluid,"base")
