.. raw:: html

    <style> .orange {color:DarkOrange} </style>

.. role:: orange

.. raw:: html

    <style> .green {color:green} </style>

.. role:: green

.. raw:: html

    <style> .Magenta {color:Magenta} </style>

.. role:: Magenta

Telluric lines
==============

This tutorial is part of the `spectral reduction <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_reduc/procedures_reduc.html>`_ steps, and details how to 
generate the static files that contain the list of spectral lines required for the telluric correction.

The synthetic telluric models need as input a list of all spectroscopic lines that fall in a given wavelength range, for each species that you want to include in the model. 
The telluric linelists are thus created independently from the specific characteristics of the detector but they do depend on the wavelength range covered by the instrument. 
Static files have already been computed for several species and spectrographs. 
New files can be generated as needed using the python file `Create_static_files_ALL <https://gitlab.unige.ch/spice_dune/antaress/-/blob/main/src/antaress/ANTARESS_corrections/Telluric_processing/Create_static_files_ALL.py>`_., which you can download in 
a local directory and use directly:

1. Get the spectroscopic linelists for your species of interest, named :orange:`MolNAME`, and store it in :orange:`/src/antaress/ANTARESS_corrections/Telluric_processing/Input_files_statics/Molecules/molNAME/specNAME/`, where :orange:`specNAME` is the name of your spectrograph.
   By default we retrieve these linelists as :orange:`.txt` files from the `HITRAN database <https://hitran.org/>`_. If you want to use another source, you need to adapt the ``read_hitran2012_parfile()`` function.


2. Get a :orange:`QT.txt` file to use as input of the ``static_hitran_file()`` function, and store it in :orange:`/src/antaress/ANTARESS_corrections/Telluric_processing/Input_files_statics/Molecules/molNAME/specNAME/`.
   The ``static_hitran_file()`` function reads the linelist files and creates the static file ``Static_hitran_qt_molNAME.fits``, which will be fed into the ``lines_to_fit()`` function and will be later used by ``ANTARESS`` to compute the final synthetic telluric spectrum covering the full range of the spectrograph. 
   Store this static file in :orange:`/src/antaress/ANTARESS_corrections/Telluric_processing/Static_model/specNAME/`.


3. Set up the input parameters of the ``lines_to_fit()`` function according to your specific needs and the wavelength range covered by your spectrograph. 
   The ``lines_to_fit()`` function selects, among all spectroscopic lines of a given species, the first K (~tens) strongest lines.
   It saves this information in the static file ``Static_hitran_strongest_lines_molNAME.fits``, which should be stored in the same directory as the ``Static_hitran_qt_molNAME.fits`` file.
   ``ANTARESS`` will use this sample of K strong lines to generate the synthetic telluric spectrum that is fitted to the observed spectrum.
   At this point you can exclude some regions of the wavelength range from which you do not want to get any strong line or, equivalently, indicate the spectral region which you want all your strongest lines to be selected from. 
   You also need to specify how many strongest lines you want. See the last portion of the python file for reference, under the section ``files to compute``.

Look `here <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/contributions.html>`_ if you want your updates to be propagated in the stable version of ``ANTARESS``. 

.. Tip:: 
 Existing static files have been optimized for the complete spectral range of a given spectrograph. 
 If you use ``ANTARESS`` to process a specific spectral range, you may consider selecting a strong linelist within this range to improve the telluric fit.

