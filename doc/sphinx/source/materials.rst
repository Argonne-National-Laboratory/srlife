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

The thermal material model provides the metal's conductivity and diffusivity as a function of temperature.

Deformation materials
^^^^^^^^^^^^^^^^^^^^^

The deformation model system is a thin wrapper around the `neml <https://github.com/Argonne-National-Laboratory/neml>`_ -- a nonlinear
constitutive model system focused on high temperature materials developed by Argonne National Laboratory.
The `"base"` models for each material are decoupled creep-plasticity models representing both rate independent plasticity and
rate dependent plasticity.  These models are not as accurate as fully-coupled viscoplastic models, but are easily calibrated against
commonly-available experimental data.

Damage (structural) materials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The current version of srlife includes only one damage model: an ASME-type approach that uses Miner's rule, time-fraction creep damage,
and a creep-fatigue interaction diagram to determine creep-fatigue failure.
The damage material model provides the required data, including nominal strain-based fatigue curves, a creep rupture correlation, and
the interaction diagram.

Fluid materials
^^^^^^^^^^^^^^^

The fluid material system is different than the other three materials.   The model describes the convective heat transfer coefficient between the
base metal and a given fluid as a function of temperature.  Note that this neglects the influence of flow rate, which may be incorporated into future
versions.  Instead of being indexed against the tube material the fluid material systems indexes the models first against the coolant type.  The table 
below describes the options currently embedded in NEML.

+----------------------+-----------+----------+
| Fluid                | String ID | Comments |
+======================+===========+==========+
| Molten chloride salt | "salt"    |          |
+----------------------+-----------+----------+

As with the metallic material models, the fluid material system also subdivides models with a variant specification.  Again, srlife provides a
`"base"` variant and users could expand the system to other models.  

Convective heat transfer properties could vary both with the working fluid type and with the tube material.  The fluid material system provides a
default option suitable for most metallic tube material and, where data is available available, also specializes the fluid model to specific materials.

Loading material models
-----------------------

The user only needs to interact with two functions to load in material models.  The first loads in the thermal, deformation, and structural
models for a particular tube material:

.. autofunction:: srlife.library.load_material

The second loads in data for a particular working fluid:

.. autofunction:: srlife.library.load_fluid
