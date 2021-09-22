Managers: running an analysis
=============================

The :py:class:`srlife.managers.SolutionManager` class manages an analysis,
taking the basic input information:

   1. The fully-populated :py:class:`srlife.receiver.Receiver` class.
   2. The thermal, fluid, deformation, and damage material models.
   3. The thermal, structural, and system solvers.
   4. Optionally, a :py:class:`srlife.solverparams.ParameterSet` class
      defining solution parameters, including the number of parallel
      threads to use in the analysi

and providing the estimated life of the receiver as a number of 
repetitions of the daily cycle.

Once the manager class is constructed the user only needs to call the

.. code-block::

   life = manager.solve_life()

method, which completes the full analysis and returns the estimated life in
terms of the number of expected single-day repetitions.  The calculation scales
the results appropriately given the number of explicitly-defined `days` 
provided to the :py:class:`srlife.receiver.Receiver`.

SolutionManager description
----------------------------

The :py:class:`srlife.managers.SolutionManager` is a wrapper around several 
internal srlife subclasses.  Specifically, the manager handles the process of:

   1. Solving for the tube temperatures given the thermal boundary conditions.
   2. Solving for the stress/strain deformation response of each tube.
      Depending on the interconnect stiffnesses provided by the user these
      tube problems may be coupled in a 1D sense through the tube top-surface
      displacements.
   3. Using the temperature and stress/strain information to solve for the
      damage in each tube.
   4. Finding the worst-case tube and calculating the estimated life.

.. autoclass:: srlife.managers.SolutionManager
   :members:

ParameterSet description
------------------------

A :py:class:`srlife.solverparams.ParameterSet` is a hierarchical dictionary.
The top level dictionary contains global parameters that apply to all
solvers, for example

.. code-block::
   
   params = solverparams.ParameterSet()
   params["nthreads"] = 4

specifies that all solvers can use up to 4 parallel threads.  
In addition, the top level object contains subdictionaries describing the
parameters for the thermal, structural, and receiver system solvers.
For example, the code

.. code-block::
   
   params["thermal"]["rtol"] = 1.0e-6

sets the relative tolerance of the thermal solver.  The tables below
provide the options currently available at each level.

Global options
^^^^^^^^^^^^^^

+----------+-----------+---------+----------------------------------------------+
| Option   | Data type | Default | Explanation                                  |
+==========+===========+=========+==============================================+
| nthreads | int       | 1       | Number of parallel threads to use in solves. |
+----------+-----------+---------+----------------------------------------------+
| progress | bool      | False   | Provide progress bar in the command line     |
+----------+-----------+---------+----------------------------------------------+

Thermal solver options
^^^^^^^^^^^^^^^^^^^^^^

+---------+-----------+---------+--------------------------------------------------------------+
| Option  | Data type | Default | Explanation                                                  |
+=========+===========+=========+==============================================================+
| rtol    | float     | 1.0e-6  | Nonlinear solver relative tolerance                          |
+---------+-----------+---------+--------------------------------------------------------------+
| atol    | float     | 1.0e-2  | Nonlinear solver absolute tolerance                          |
+---------+-----------+---------+--------------------------------------------------------------+
| miter   | int       | 100     | Maximum nonlinear solver iterations                          |
+---------+-----------+---------+--------------------------------------------------------------+
| substep | int       | 1       | Divide each higher-level timestep into substep smaller steps |
+---------+-----------+---------+--------------------------------------------------------------+
| verbose | bool      | False   | Print debug information to the terminal                      |
+---------+-----------+---------+--------------------------------------------------------------+
| steady  | bool      | False   | Use steady state heat transfer, i.e. conduction only         |
+---------+-----------+---------+--------------------------------------------------------------+

Tube solver options
^^^^^^^^^^^^^^^^^^^

+---------+-----------+---------+-------------------------------------------------+
| Option  | Data type | Default | Explanation                                     |
+=========+===========+=========+=================================================+
| rtol    | float     | 1.0e-6  | Nonlinear solver relative tolerance             |
+---------+-----------+---------+-------------------------------------------------+
| atol    | float     | 1.0e-8  | Nonlinear solver absolute tolerance             |
+---------+-----------+---------+-------------------------------------------------+
| miter   | int       | 10      | Maximum nonlinear solver iterations             |
+---------+-----------+---------+-------------------------------------------------+
| qorder  | int       | 1       | Quadrature order for the finite element method  |
+---------+-----------+---------+-------------------------------------------------+
| verbose | bool      | False   | Print debug information to the terminal         |
+---------+-----------+---------+-------------------------------------------------+
| dof_tol | float     | 1.0e-6  | Geometric tolerance for finding nodes on planes |
+---------+-----------+---------+-------------------------------------------------+

Structural solver options
^^^^^^^^^^^^^^^^^^^^^^^^^

+---------+-----------+---------+-----------------------------------------+
| Option  | Data type | Default | Explanation                             |
+=========+===========+=========+=========================================+
| rtol    | float     | 1.0e-6  | Nonlinear solver relative tolerance     |
+---------+-----------+---------+-----------------------------------------+
| atol    | float     | 1.0e-4  | Nonlinear solver absolute tolerance     |
+---------+-----------+---------+-----------------------------------------+
| miter   | int       | 25      | Maximum nonlinear solver iterations     |
+---------+-----------+---------+-----------------------------------------+
| verbose | bool      | False   | Print debug information to the terminal |
+---------+-----------+---------+-----------------------------------------+

Class description
^^^^^^^^^^^^^^^^^

.. autoclass:: srlife.solverparams.ParameterSet
   :members:
