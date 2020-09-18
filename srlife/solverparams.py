"""
  A very small module defining a class to help organize solver parameters
"""

from collections import defaultdict

class ParameterSet(defaultdict):
  def __init__(self, **kwargs):
    super().__init__(lambda: self.__class__(**kwargs))
    self.__dict__.update(kwargs)

  def __reduce__(self):
    return (type(self), (), None, None, iter(self.items()))

  def get_default(self, key, value):
    """
      Get a key, if not present return the default value
    """
    if key in self.keys():
      return self[key]
    else:
      return value
