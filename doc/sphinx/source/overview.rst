An overview of srlife
=====================

srlife is a package for estimating the life of high temperature solar receivers.
The package currently focuses on tubular, panel receivers
though the package includes data for evaluating arbitrary high temperature
components if the user provides the corresponding analysis results.

srlife provides life estimation approaches for both metallic and ceramic
materials.  These approaches differ, as described elsewhere in the
documentation.

The package provides a complete assessment of a receiver design starting
from the thermal and mechanical boundary conditions applied to the 
individual tubes in the receiver.  This means that the package requires
input from additional simulations in order to generate the boundary conditions.
Specifically, srlife would usually sit on top of a simulation of the
heliostat to calculate the indicant solar flux and simulations or
measurements of the effective absorption of the receiver tube.  
The package, however, provides a simple thermohydraulic system solver for
tubular receivers which can find the fluid and tube temperatures along 
each flow path if provided with the flow path inlet temperatures and
flow rates.

Given this information, presented as time-dependent boundary 
conditions representing representative conditions on one or more
thermal days, the package estimates the structural life of the 
receiver, providing this estimate as a number of repetitions of the
user-provided daily cycle(s).  For metallic materials the package
provides a best-estimate life prediction using average material properties.
For ceramic receivers the analysis is currently time independent and
reports simply the reliability of the system under the provided
thermomechanical conditions, i.e. the probability the design will
not fail.  Future versions of the software will extend the
ceramic failure models to account for time dependent subcritical crack growth.
When these improvements are completed the ceramic models will report the reliability as 
a function of time or, alternatively, the time until the receiver reliability falls
below some reliability metric.

The package includes material information for a variety of 
receiver structural materials and working fluids.  The user can
add additional material models by manipulating a fixed XML file format --
they do not need to alter the package source code to add new materials or
provide variant material models for the materials already included in the 
base release.

srlife provides modules to:

   1. Define the receiver geometry and topology -- how panels are connected to each other and how tubes are connected within a panel.
   2. Provide thermomechanical boundary conditions on the receiver, specifically options for:
         a. Outer diameter incident heat flux on each tube
         b. Inner pressure in each tube
         c. Collections of panels/tubes arranged in flow paths, giving:
            i. The inlet temperature of each flow path as a function of time
            ii. The flow path flow rate, again as a function of time
   3. A coupled finite difference, transient solid heat transfer code linked to a simple 1D thermohydraulic model for heat transfer through the receiver.  The output of these solvers is both the fluid temperature as a function of
      position along each flow path and time and the solid temperatures in each tube given as a function of position and time.
   4. A full-scale finite element solver to take the tube temperatures and mechanical boundary conditions to the tube stress/strain/displacement fields.
   5. Connections to a extensive nonlinear material model library `neml <https://github.com/Argonne-National-Laboratory/neml>`_ to provide accurate inelastic constitutive models.
   6. A receiver system solver that can account for connections between tubes in a panel and panels in a receiver, to accurately model the effect of structural connections with an abstract, numerically inexpensive representation.
   7. Damage solvers to estimate the level of creep-fatigue damage in a tube given the structural and thermal results for metallic materials and ceramic reliability calculations that link the applied stresses and temperatures
      to an expected reliability.
   8. An extensive material property library covering common high temperature metallic and ceramic receiver materials.

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
