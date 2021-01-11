# This file lightly adapted from the example CMake build provided in the pybind11 distribution
# located at https://github.com/pybind/cmake_example/blob/master/setup.py
# That file is released under an open source license, contained in the pybind11 subdirectory
# of this package.

import os
import sys
import platform

from setuptools import setup, find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding = 'utf-8') as f:
  long_description = f.read()

setup (
    # Name of the project
    name = 'srlife',
    # Version
    version = '1.0.2',
    # One line-description
    description = "Evaluate the structural life of a solar receiver",
    # README
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    # Project webpage
    url='https://github.com/Argonne-National-Laboratory/srlife',
    # Author
    author='Argonne National Laboratory',
    # email
    author_email = 'messner@anl.gov',
    # Various things for pypi
    classifiers=[
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3',
      'Operating System :: OS Independent'
      ],
    # Which version of python is needed
    python_requires='>=3.6',
    # Keywords to help find
    keywords='materials structures modeling',
    # Definitely not zip safe
    zip_safe=True,
    # Get the python files
    packages=find_packages(),
    # Python dependencies
    install_requires=[
      'numpy==1.18',
      'scipy',
      'h5py',
      'vtk', 
      'nose', 
      'matplotlib',
      'pylint', 
      'neml', 
      'scikit-fem', 
      'meshio', 
      'networkx', 
      'tqdm', 
      'multiprocess', 
      'dill'
      ],
    include_package_data=True,
    package_data={'': ['data/damage/*.xml', 'data/deformation/*.xml', 
      'data/fluid/*.xml', 'data/thermal/*.xml']},
)
