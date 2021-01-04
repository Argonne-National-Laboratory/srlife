Receiver module: Defining the receiver geometry
===============================================

Receiver system
---------------

As described in the overview, the user must provide the geometric
definition of the tubular receiver and the mechanical and thermal
boundary conditions as part of the required package input.  
The :py:mod:`srlife.receiver` module defines data structures for providing
this information and some simple methods for manipulating the data.

The python class :py:class:`srlife.receiver.Receiver` provides the top level
interface  and uses a hierarchy of objects to describe the receiver
geometry and boundary conditions.  The figure below describes the structure.

.. image:: tube-data.png
   :width: 1600
   :alt: Receiver data structure

The overall data structure is hierarchical, descending from the single 
:py:class:`srlife.receiver.Receiver` object describing the entire receiver.
The receiver has one or more :py:class:`srlife.receiver.Panel` objects
each of which have one ore more :py:class:`srlife.receiver.Tube` objects.
Structurally, each tube in a panel is connected through its top-surface
displacement through a spring.  Similarly, each panel is connected to the
other panels through a spring.  The spring stiffness defines the type 
of connection, options are:

   1. A string `"disconnect"` meaning the spring has zero stiffness and
      the member floats freely, not connected to the result of the objects
      of its type.
   2. A string `"rigid"` which means the object is rigidly connected to the
      others of its type with an infinitely-rigid spring.
   3. A `float` giving the discrete spring stiffness.

The :py:class:`srlife.receiver.Receiver` structure itself stores additional metadata:

   1. The cycle period.  Typically this will be 24 hours, but could be a 
      reduced representation, for example cutting out a period of the 
      hold in the cold nighttime condition.
   2. The number of days (cycle repetitions) actually represented in the
      structural and thermal boundary conditions.  This could, and almost
      always will be, not the full receiver life in days as the prescribed
      loads could represent a larger number of actual repetitions in service.

The Tube object holds basic geometric information describing a cylindrical
tube, the tube discretization which is a fixed grid in cylindrical coordinates
given by a fixed number of radial, circumferential, and axial subdivisions,
the initial tube temperature, the discrete times at which the object stores boundary condition information and results, zero or more nodal results fields, 
zero or more quadrature result fields, an optional inner pressure boundary
condition, an optional inner diameter thermal boundary condition, and an
optional outer diameter thermal boundary condition.  In addition, the
tube carries a flag indicating whether the tube should be treated with a 
full 3D analysis or a reduced 2D or 1D analysis.  For the reduced analysis the
tube retains the height of the slice (2D and 1D) and the angle of the
slice (1D), with respect to the full 3D cylindrical coordinate system.

The tube result and boundary condition times are a 1D array of time values, 
starting at zero.  This array has length :math:`n_{time}` and must be
consistent for all the nodal results, quadrature results, pressure 
boundary conditions, and thermal boundary conditions.

Nodal results are stored on a fixed grid defined by an array of size

.. math::

   n_{time} \times n_{r} \times n_{\theta} \times n_{z}

where :math:`n_{r}` is the number of radial grid points, :math:`n_{\theta}`
is the number of circumferential grid points, and :math:`n_z` is the 
number of axial grid points.  A 2D representation leaves off the last
axis and a 1D representation leaves off the last two axes.

The inner pressure boundary condition :py:class:`srlife.receiver.PressureBC` holds a 1D array of pressures
defining the pressure at discrete times.  The size of this 1D array
must be :math:`n_{time}` and match the tube object.

The user has three options for thermal boundary conditions, which can
be specified on the inner diameter, outer diameter, or both:

   1. Fixed flux boundary condition :py:class:`srlife.receiver.HeatFluxBC`.
   2. Fixed temperature boundary condition :py:class:`srlife.receiver.FixedTempBC`.
   3. Convective heat transfer :py:class:`srlife.receiver.ConvectiveBC`.

The data for the flux and temperature boundary conditions is an array of
flux or temperature values of size

.. math::

   n_{time} \times n_{\theta} \times n_{z}

representing a fixed grid of points on either the tube inner or outer diameter.
The user must provide the full 3D data.

Similarly, the data for the convective heat transfer boundary condition
is an array of fluid temperature data of size

.. math::

   n_{time} \times n_{z}

Again, the user must always provide the full 3D information.

The user can provide the required input data in two ways:
   1. The :ref:`python-receiver`
   2. An :ref:`hdf-receiver`

Both methods provide options for linking into external programs.

.. _python-receiver:

Python objects
--------------

Receiver geometry
^^^^^^^^^^^^^^^^^

The Receiver object contains panels and the interconnect spring stiffness
between the panels.

.. autoclass:: srlife.receiver.Receiver
   :members:

The Panel object contains tubes and the interconnect spring stiffness
between each tube in the panel

.. autoclass:: srlife.receiver.Panel
   :members:

The Tube object contains the majority of the analysis information and
results of the analysis.  The structure holds the basic tube geometry,
the tube discretization, described as intervals in a cylindrical
coordinate system, and initial tube temperature, the thermal and
mechanical boundary conditions, and result fields stored at discrete
times and either at node points or quadrature points.

.. autoclass:: srlife.receiver.Tube
   :members:

Boundary conditions
^^^^^^^^^^^^^^^^^^^

The only mechanical boundary condition the user needs to specify
is the Tube internal pressure.

.. autoclass:: srlife.receiver.PressureBC
   :members:

Superclass handing HDF storage dispatch for thermal boundary conditions

.. autoclass:: srlife.receiver.ThermalBC

Thermal boundary conditions options are fixed flux (HeatFluxBC), fixed
temperature (FixedTempBC), and convection (ConvectiveBC).

.. autoclass:: srlife.receiver.HeatFluxBC

.. autoclass:: srlife.receiver.FixedTempBC

.. autoclass:: srlife.receiver.ConvectiveBC

.. _hdf-receiver:

HDF5 file
---------

`HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`_ provides a hierarchical
data structure aimed at storing and transferring scientific data.
srlife uses the format, through the `h5py <https://www.h5py.org/>`_ package, to serialize and store the definition of a complete receiver to file.  
A user could write an HDF5 file with the appropriate format to interface
with srlife from an external program.

The interface uses HDF5 attributes, datasets, and groups to store the data.
An attributes is metadata, a single string, bool, float, integer, etc.  
A dataset is a fixed-size array, essentially in the h5py package a numpy
array.  A group is a container that can hold attributes, datasets, or
additional groups.

The format adopted here is then an exact mirror of the type structure
defined :ref:`above <python-receiver>`.  The :py:class:`srlife.receiver.Receiver` object's metadata and data sits at the top of the HDF5 file (i.e. not
contained in a group), and stores the
:py:class:`srlife.receiver.Panel`, :py:class:`srlife.receiver.Tube`, :py:class:`srlife.receiver.PressureBC`, and :py:class:`srlife.receiver.ThermalBC` as
groups and subgroups.  The HDF5 file maintains the hierarchical structure of
the python data structures, so the receiver stores the panels in a group, each panel stores its tubes in a group, and the thermal and pressure BCs are subgroups
of the tubes.

The following summarizes the HDF5 format.  Example HDF5 files are contained in the `srlife/examples` directory.

Receiver
^^^^^^^^

This is the top-level of the HDF5 file.

+-----------+-----------+--------------+----------------------------------------+----------------------------------------------+
| Field     | Type      | Data type    | Explanation                            | Notes                                        |
+===========+===========+==============+========================================+==============================================+
| period    | attribute | float        | Daily cycle period                     |                                              |
+-----------+-----------+--------------+----------------------------------------+----------------------------------------------+
| days      | attribute | int          | Number of cycle repetitions            |                                              |
+-----------+-----------+--------------+----------------------------------------+----------------------------------------------+
| stiffness | attribute | float/string | Panel interconnect stiffness           | "disconnect", "rigid", or value of stiffness |
+-----------+-----------+--------------+----------------------------------------+----------------------------------------------+
| panels    | group     | n/a          | Each panel is a subgroup of this group | Default naming scheme: 0, 1, 2, ...          |
+-----------+-----------+--------------+----------------------------------------+----------------------------------------------+

Panel
^^^^^

+-----------+-----------+--------------+---------------------------------------+----------------------------------------------+
| Field     | Type      | Data type    | Explanation                           | Notes                                        |
+===========+===========+==============+=======================================+==============================================+
| stiffness | attribute | float/string | Panel interconnect stiffness          | "disconnect", "rigid", or value of stiffness |
+-----------+-----------+--------------+---------------------------------------+----------------------------------------------+
| tubes     | group     | n/a          | Each tube is a subgroup of this group | Default naming scheme: 0, 1, 2, ...          |
+-----------+-----------+--------------+---------------------------------------+----------------------------------------------+

Tube
^^^^

When providing an HDF5 file as input the `results` and `quadrature_results` groups must exist but the user does not
need to provide any pre-populated results fields (srlife will add these during the analysis).

+--------------------+-----------+---------------+---------------------------------+--------------------------------+
| Field              | Type      | Data type     | Explanation                     | Notes                          |
+====================+===========+===============+=================================+================================+
| r                  | attribute | float         | Tube inner radius               |                                |
+--------------------+-----------+---------------+---------------------------------+--------------------------------+
| t                  | attribute | float         | Tube thickness                  |                                |
+--------------------+-----------+---------------+---------------------------------+--------------------------------+
| h                  | attribute | float         | Tube height                     |                                |
+--------------------+-----------+---------------+---------------------------------+--------------------------------+
| nr                 | attribute | int           | Number of radial nodes          |                                |
+--------------------+-----------+---------------+---------------------------------+--------------------------------+
| nt                 | attribute | int           | Number of circumferential nodes |                                |
+--------------------+-----------+---------------+---------------------------------+--------------------------------+
| nz                 | attribute | int           | Number of axial nodes           |                                |
+--------------------+-----------+---------------+---------------------------------+--------------------------------+
| abstraction        | attribute | string        | Tube dimension                  | "1D", "2D", or "3D"            |
+--------------------+-----------+---------------+---------------------------------+--------------------------------+
| times              | dataset   | dim: (ntime,) | Discrete time points            |                                |
+--------------------+-----------+---------------+---------------------------------+--------------------------------+
| results            | group     | n/a           | Node point results              | Each result field is a dataset |
+--------------------+-----------+---------------+---------------------------------+--------------------------------+
| quadrature_results | group     | n/a           | Quadrature point results        | Each result field is a dataset |
+--------------------+-----------+---------------+---------------------------------+--------------------------------+
| outer_bc           | group     | n/a           | Outer diameter thermal BC       | See below                      |
+--------------------+-----------+---------------+---------------------------------+--------------------------------+
| inner_bc           | group     | n/a           | Inner diameter thermal BC       | See below                      |
+--------------------+-----------+---------------+---------------------------------+--------------------------------+
| pressure_bc        | group     | n/a           | Internal pressure               | See below                      |
+--------------------+-----------+---------------+---------------------------------+--------------------------------+

HeatFluxBC
^^^^^^^^^^

The user must always provide the full flux information (i.e. over the full 3D tube).

+-------+-----------+----------------------+---------------------------------+-------------------------------------------+
| Field | Type      | Data type            | Explanation                     | Notes                                     |
+=======+===========+======================+=================================+===========================================+
| type  | attribute | string               | Thermal BC type                 | Must be "HeatFlux"                        |
+-------+-----------+----------------------+---------------------------------+-------------------------------------------+
| r     | attribute | float                | Radius of application           | Must match tube inner or outer radius     |
+-------+-----------+----------------------+---------------------------------+-------------------------------------------+
| h     | attribute | float                | Tube height                     | Must match tube height                    |
+-------+-----------+----------------------+---------------------------------+-------------------------------------------+
| nt    | attribute | int                  | Number of circumferential nodes |                                           |
+-------+-----------+----------------------+---------------------------------+-------------------------------------------+
| nz    | attribute | int                  | Number of axial nodes           |                                           |
+-------+-----------+----------------------+---------------------------------+-------------------------------------------+
| times | dataset   | dim: (ntime,)        | Discrete times points           |                                           |
+-------+-----------+----------------------+---------------------------------+-------------------------------------------+
| data  | dataset   | dim: (ntime, nt, nz) | Discrete flux data              | Fixed array over tube inner/outer surface |
+-------+-----------+----------------------+---------------------------------+-------------------------------------------+

FixedTempBC
^^^^^^^^^^^

The user must always provide the full temperature information (i.e. over the full 3D tube).

+-------+-----------+----------------------+---------------------------------+-------------------------------------------+
| Field | Type      | Data type            | Explanation                     | Notes                                     |
+=======+===========+======================+=================================+===========================================+
| type  | attribute | string               | Thermal BC type                 | Must be "FixedTemp"                       |
+-------+-----------+----------------------+---------------------------------+-------------------------------------------+
| r     | attribute | float                | Radius of application           | Must match tube inner or outer radius     |
+-------+-----------+----------------------+---------------------------------+-------------------------------------------+
| h     | attribute | float                | Tube height                     | Must match tube height                    |
+-------+-----------+----------------------+---------------------------------+-------------------------------------------+
| nt    | attribute | int                  | Number of circumferential nodes |                                           |
+-------+-----------+----------------------+---------------------------------+-------------------------------------------+
| nz    | attribute | int                  | Number of axial nodes           |                                           |
+-------+-----------+----------------------+---------------------------------+-------------------------------------------+
| times | dataset   | dim: (ntime,)        | Discrete times points           |                                           |
+-------+-----------+----------------------+---------------------------------+-------------------------------------------+
| data  | dataset   | dim: (ntime, nt, nz) | Discrete temperature data       | Fixed array over tube inner/outer surface |
+-------+-----------+----------------------+---------------------------------+-------------------------------------------+

ConvectiveBC
^^^^^^^^^^^^

The user must always provide the full fluid temperature information (i.e. over the full 3D tube).

+-------+-----------+------------------+---------------------------------+---------------------------------------+
| Field | Type      | Data type        | Explanation                     | Notes                                 |
+=======+===========+==================+=================================+=======================================+
| type  | attribute | string           | Thermal BC type                 | Must be "Convective"                  |
+-------+-----------+------------------+---------------------------------+---------------------------------------+
| r     | attribute | float            | Radius of application           | Must match tube inner or outer radius |
+-------+-----------+------------------+---------------------------------+---------------------------------------+
| h     | attribute | float            | Tube height                     | Must match tube height                |
+-------+-----------+------------------+---------------------------------+---------------------------------------+
| nz    | attribute | int              | Number of axial nodes           |                                       |
+-------+-----------+------------------+---------------------------------+---------------------------------------+
| times | dataset   | dim: (ntime,)    | Discrete times points           |                                       |
+-------+-----------+------------------+---------------------------------+---------------------------------------+
| data  | dataset   | dim: (ntime, nz) | Discrete fluid temperature data | Fixed array over tube height          |
+-------+-----------+------------------+---------------------------------+---------------------------------------+

PressureBC
^^^^^^^^^^

+-------+---------+---------------+------------------------+-------+
| Field | Type    | Data type     | Explanation            | Notes |
+=======+=========+===============+========================+=======+
| times | dataset | dim: (ntime,) | Discrete times points  |       |
+-------+---------+---------------+------------------------+-------+
| data  | dataset | dim: (ntime,) | Discrete pressure data |       |
+-------+---------+---------------+------------------------+-------+
