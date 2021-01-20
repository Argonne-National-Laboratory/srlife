Installing srlife
=================

srlife is available in the `pypi <https://pypi.org/>`_ package repository
and can be installed with `pip`.  srlife uses python3 and requires several
additional python packages, all of which are available pypi.  The only
additional requirement are cmake, the python development headers, and
development version of BLAS and LAPACK.  All of these additional requirements
are needed to compile `neml <https://github.com/Argonne-National-Laboratory/neml>`_, which provides the nonlinear constitutive model response required for
the srlife analysis modules.  

Install using the pip package manager
-------------------------------------

The easiest way to install the package is to use the `pip` package manager, installing srlife from pypi automatically.  However, the neml source package requires compiling the module from source, which still has a few system-level dependencies.

Ubuntu Linux 18.04
""""""""""""""""""

.. code-block:: console

   sudo apt install build-essential cmake libblas-dev liblapack-dev python3-dev python3-setuptools python3-pip python3-wheel 
   pip3 install --user srlife

MacOS Sierra 10.14 Mojave
"""""""""""""""""""""""""

Go to `brew.sh <https://brew.sh/>`_ and follow the directions to install homebrew.

Open up a terminal and run:

.. code-block:: console

   brew install cmake openblas superlu python vtk
   pip3 install --user srlife

`srlife` will then be available as a package through the homebrew version of python (often available as `python3` instead of `python`).

Install from the github repository directly
-------------------------------------------

If you want to use the current development version of srlife or if you want
to also obtain the tutorial, example, and test files you can install the
package directly from `github <https://github.com/Argonne-National-Laboratory/srlife>`_.  In addition to the, cmake, BLAS, and LAPACK requirements you will
need git and, optionally, the nose package to automatically run the tests.

Ubuntu Linux 18.04
""""""""""""""""""

The following installs the prerequisites, downloads srlife, sets up the python package, and runs the automated test suite.

.. code-block:: console

   sudo apt install build-essential cmake libblas-dev liblapack-dev python3-dev python3-setuptools python3-pip python3-nose 
   git clone https://github.com/Argonne-National-Laboratory/srlife.git
   cd srlife
   pip3 install --user wheel
   pip3 install --user -r requirements.txt
   nosetests3

Note the package is installed wherever the user executed the `git clone` command.  Using the package outside this directory
requires adding it to the `PYTHONPATH` environment variable.

