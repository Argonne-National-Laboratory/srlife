Materials: thermal and structural material models
=================================================

srlife includes a library of material models to provide the data required
to solve the thermal, structural, and damage analysis calculations.
These models represent a best estimate of *average* material properties,
meaning the analysis results and the final life estimate represent average 
and not *lower bound design* estimates.

The material system references the tube metallic material.  The table
below lists the available metallic material types along with a comment
on the source of data and reliability of the current model.

+---------------------------+-----------+--------------------------------------------------------------------------+
| Material                  | String ID | Comments                                                                 |
+===========================+===========+==========================================================================+
| 316H stainless steel      | "316H"    | Large US DOE nuclear energy database                                     |
+---------------------------+-----------+--------------------------------------------------------------------------+
| 800H high alloy steel     | "800H"    | Large US DOE nuclear energy database                                     |
+---------------------------+-----------+--------------------------------------------------------------------------+
| Alloy 230 Ni-based alloy  | "A230"    | Extremely limited literature data, creep-fatigue properties questionable |
+---------------------------+-----------+--------------------------------------------------------------------------+
| Alloy 617 Ni-based alloy  | "A617"    | Large US DOE nuclear energy database                                     |
+---------------------------+-----------+--------------------------------------------------------------------------+
| Alloy 740H Ni-based alloy | "740H"    | Limited creep-fatigue database on single heat                            |
+---------------------------+-----------+--------------------------------------------------------------------------+
| Alloy 282 Ni-based alloy  | "A282"    | Limited literature data, creep-fatigue properties preliminary            |
+---------------------------+-----------+--------------------------------------------------------------------------+
| Silicon Carbide           | "SiC"     | Literature data for one specific batch of SiC material                   |
+---------------------------+-----------+--------------------------------------------------------------------------+

While the SiC failure data provided in the package is specific to one batch of material and would need to be
altered to reflect the actual properties of the material under consideration, the thermal and deformation
models should be applicable to most commercial monolithic SiC.

The material system accommodates variants within each model type.  
The srlife package provides a `"base"` variant for all the materials.
The user can add additional material variants to explore alternate model forms
or models calibrated against different datasets.  The comments in the table
above only apply to the `"base"` material variant

.. todo::
   
   The documentation currently only covers the user interface to the material
   system and not the underlying implementation.  Future version will
   include documentation on how to add additional materials and document
   the material model internals.

Material model descriptions
---------------------------

Thermal materials
^^^^^^^^^^^^^^^^^

The thermal material model provides the material's conductivity and diffusivity as a function of temperature.

Deformation materials
^^^^^^^^^^^^^^^^^^^^^

The deformation model system is a thin wrapper around the `neml <https://github.com/Argonne-National-Laboratory/neml>`_ -- a nonlinear
constitutive model system focused on high temperature materials developed by Argonne National Laboratory.
The `"base"` models for each material are decoupled creep-plasticity models representing both rate independent plasticity and
rate dependent plasticity (elasticity only for SiC).  These models are not as accurate as fully-coupled viscoplastic models, but are easily calibrated against
commonly-available experimental data.

Damage (structural) materials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The current version of srlife includes two damage models:

* An ASME-type approach that uses Miner's rule, time-fraction creep damage, a
  and a creep-fatigue interaction diagram to determine creep-fatigue failure.
  The damage material model provides the required data, including nominal strain-based fatigue curves, a creep rupture correlation, and
  the interaction diagram.  This model is suitable for metallic materials.
* A time-independent ceramic reliability models representing an extension to 3D of standard uniaxial Weibull failure statistics.
  This model is suitable for ceramic materials.

Thermal fluid materials
^^^^^^^^^^^^^^^^^^^^^^^

The fluid material system is different than the other three materials in that it relates to the receiver working fluid and not
the solid material.  This system provides the physical properties (density, dynamic viscosity, heat capacity, and conductivity)
and thermohydraulic correlations (a Nusselt correlation) required to calculate heat transfer along a flow path, including
convective heat transfer between the tube material and the working fluid.

+--------------------------------------------+--------------------+----------+
| Fluid                                      | String ID          | Comments |
+============================================+====================+==========+
| Magnesium-potassium eutectic chloride salt | "32MgCl2-68KCl"    |          |
+--------------------------------------------+--------------------+----------+
| Supercritical carbon dioxide               | "sCO2"             |          |
+--------------------------------------------+--------------------+----------+

As with the solid material models, the fluid material system also subdivides models with a variant specification.  Again, srlife provides a
`"base"` variant and users could expand the system to other models.  

Loading material models
-----------------------

The user only needs to interact with two functions to load in material models.  The first loads in the thermal, deformation, and structural
models for a particular tube material:

.. autofunction:: srlife.library.load_material

The second loads in data for a particular working fluid:

.. autofunction:: srlife.library.load_thermal_fluid
