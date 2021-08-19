An overview of srlife
=====================

srlife is a package for estimating the life of high temperature solar receivers.
The package currently focuses on metallic, tubular, panel receivers somewhat
skewed towards new designs using molten chloride salt as the working fluid,
though the package includes data for evaluating different types of working
fluids and receiver materials.

The package provides a complete assessment of a receiver design starting
from the thermal and mechanical boundary conditions applied to the 
individual tubes in the receiver.  This means that the package requires
input from additional simulations in order to generate the boundary conditions.
Specifically, srlife would usually sit on top of a simulation of the
heliostat to calculate the indicant solar flux, simulations or
measurements of the effective absorption of the receiver tubes, and system-level
and detailed thermohydraulic simulations to determine the local 
thermomechanical boundary conditions on each tube.  

Given this information, presented as time-dependent boundary 
conditions representing representative conditions on one or more
thermal days, the package estimates the structural life of the 
receiver, providing this estimate as a number of repetitions of the
user-provided daily cycle(s).

The package includes material information for a variety of 
receiver structural materials and working fluids.  The user can
add additional material models by manipulating a fixed XML file format --
they do not need to alter the package source code to add new materials or
provide variant material models for the materials already included in the 
base release.

srlife provides modules to:

   1. Define the receiver geometry and topology -- how panels are connected to each other and how tubes are connected within a panel.
   2. Provide thermomechanical boundary conditions, specifically options for:
         a. Inner or outer diameter incident heat flux
         b. Inner or outer diameter fixed temperature
         c. Inner or outer diameter convective heat transfer
         d. Inner pressure
   3. Finite difference, transient heat transfer solvers to convert the thermal boundary conditions into the receiver tube temperature fields.
   4. A full-scale finite element solver to take the tube temperatures and mechanical boundary conditions to the tube stress/strain/displacement fields.
   5. Connections to a extensive nonlinear material model library `neml <https://github.com/Argonne-National-Laboratory/neml>`_ to provide accurate inelastic constitutive models.
   6. A receiver system solver that can account for connections between tubes in a panel and panels in a receiver, to accurately model the effect of structural connections with an abstract, numerically inexpensive representation.
   7. Damage solvers to estimate the level of creep-fatigue damage in a tube given the structural and thermal results.
   8. An extensive material property library covering common high temperature metallic receiver materials.

Conventions
-----------

With user-defined material models the end-user can apply any unit system they want.  However, the built-in material library uses the following unit conventions:

+------------------+-------------------+
| Quantity         | Unit              |
+==================+===================+
| length           | mm                |
+------------------+-------------------+
| angle            | radians           |
+------------------+-------------------+
| stress           | MPa               |
+------------------+-------------------+
| time             | hr                |
+------------------+-------------------+
| strain           | mm/mm             |
+------------------+-------------------+
| temperature      | K                 |
+------------------+-------------------+
| flux             | W/mm\ :sup:`2`    |
+------------------+-------------------+
| conductivity     | W/mm-K            |
+------------------+-------------------+
| diffusivity      | mm\ :sup:`2`\ /hr |
+------------------+-------------------+
| film coefficient | W/mm\ :sup:`2`-K  |
+------------------+-------------------+

.. warning::

   The built-in material model library aims for *average* life estimation,
   approximating the average time to failure for a particular component.
   This type of estimation is not suitable for a full design calculation, 
   where lower-bound properties with adequate design margin must be used.
