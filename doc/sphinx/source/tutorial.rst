Tutorial: how to use the package
================================

This tutorial guides you through setting up a very simple receiver model
using the python interface, saving the model to an HDF5 file for archiving
and as an example of how to use the file interface to the package,
setting up an analysis, running the analysis, and examining detailed results.
The tutorial assumes you have :any:`installed <installation>` srlife, have a working text editor,
and can run python programs from the command line.  The example commands
here are for Ubuntu Linux, but the step-by-step directions are similar
for any operating system.

The files produced by this tutorial are available in the full srlife 
source package, which can be obtained via git:

.. code-block:: console

   git clone https://github.com/Argonne-National-Laboratory/srlife.git
   cd srlife

The files are in the `examples/tutorial` directory.  *Obtaining the files
is not necessary as this tutorial will walk you through creating the
files and output data available in that source directory.*

Receiver geometry, loading, and materials
-----------------------------------------

The following image describes the very simple receiver used in this example:


The receiver has two panels each with two tubes.  This is (of course) not a
realistic configuration, but setup to ensure the life estimation only takes
a short period of time, even for a full 3D analysis.

The thermal boundary conditions are an incident flux on the tube outer diameter,
described mathematically by the function:

.. math::

   h = h_{max} \cos \theta \left( \frac{\cos \phi + 1.0}{2} + \frac{1}{2} \right) 

where :math:`h_{max} = xx` and :math:`\phi` varies by tube:


The tube inner diameter experiences convective heat transfer 

Defining the receiver geometry and loading conditions
-----------------------------------------------------


Defining the analysis material models and analysis parameters
-------------------------------------------------------------


Running the life estimation analysis
------------------------------------


Visualizing tube results
------------------------


Complete example scripts
------------------------
