Installing srlife
=================

srlife is available in the `pypi <https://pypi.org/>`_ package repository
and can be installed with `pip`.  srlife uses python3 and requires several
additional python packages, all of which are available pypi.  The only
additional requirement are cmake, the python development headers, and
development version of BLAS and LAPACK.  All of these additional requirements
are needed to compile `neml <https://github.com/Argonne-National-Laboratory/neml>`_, which provides the nonlinear constitutive model response required for
the srlife analysis modules.  

OS install instructions
-----------------------

Ubuntu Linux
""""""""""""

.. code-block:: console

   sudo apt install build-essential cmake libboost-dev libblas-dev liblapack-dev python3-dev
   pip install --user srlife
