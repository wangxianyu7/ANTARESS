.. raw:: html

    <style> .orange {color:DarkOrange} </style>

.. role:: orange

.. raw:: html

    <style> .green {color:green} </style>

.. role:: green

.. raw:: html

    <style> .Magenta {color:Magenta} </style>

.. role:: Magenta

Resolution maps
===============

This tutorial details how to generate the resolution map of a spectrograph.

Each detector presents an arbitrary nominal resolving power (instrumental resolution) given by the constructor.
Yet, this value may vary slightly over the detector, depending on boundary conditions and the performances of each chip.
The observed spectrum is affected by changes in resolution over the detector.

A *resolution map* tracks down the true resolving power of the detector at any given position on its surface, thus it associates a specific resolving power to each wavelength in the spectral range covered by the instrument.  
It is then possible to convolve theoretical models of the stellar and telluric spectra using the resolution map instead of a fixed resolution, providing a more realistic comparison with the measured spectra.

Create a directory in your local environment, in which you will download the notebook `ANTARESS_nbook_resolution_maps <https://gitlab.unige.ch/spice_dune/antaress/-/blob/main/Notebooks/ANTARESS_nbook_resolution_maps.ipynb>`_ and the 
python file `Create_resolution_maps_ALL <https://gitlab.unige.ch/spice_dune/antaress/-/blob/main/src/antaress/ANTARESS_corrections/Telluric_processing/Create_resolution_maps_ALL.py>`_.  
We describe below the python file and how you need to update it for a new spectrograph, before running the notebook to generate resolution maps. 

The new maps must be stored in your installed ``ANTARESS`` distribution under :orange:`/src/antaress/ANTARESS_corrections/Telluric_processing/Static_resolution/specNAME/`, where :orange:`specNAME` is the name of your spectrograph.
Then you need to update the function ``open_resolution_map()`` within the file :orange:`/src/antaress/ANTARESS_corrections/ANTARESS_tellurics.py` so that it can open the new maps.
Look `here <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/contributions.html>`_ if you want your updates to be propagated in the stable version of ``ANTARESS``.  

 
Reproducing the 2D detector shape
---------------------------------
  
Imagine the detector as a MxN matrix, where *M* is the number of orders on the y axis and *N* the number of pixels on the x axis.
The ``shape_s2d_()`` function gives you a single row of N pixels that reproduces the width of the detector.
The outcoming resolution map will be computed on a matrix of M rows of dimension N, which basically represents the hard structure of the detector and thus the boundaries of the fitting.

**How to update :**

You need to define your own ``shape_s2d_()`` function. The more sophisticated the detector, the more articulated this function will be. 
As detectors age with time and undergo routine or exceptional interventions, you may want to: 

1. Exclude some parts of the detector *a priori* from the fitting (e.g., if they work bad).

2. Define a different detector shape for each epoch of the instrument. Pixel binning for read-out purposes also needs to be considered.

Look at the function ``shape_s2d_ESPRESSO()`` for a reference.

.. Note:: 
  At the end of the day all you need to do is to fix the width of your detector, namely the number of pixels along the x axis, for all epochs of the detector.




Creating the resolution map
---------------------------

The ``Instrumental_resolution_()`` function computes the resolution map for a given detector. To do so, it needs as input:  

1. A reference of well-characterized instrumental resolution values spread over the detector.
 
2. An interpolating function of any sort in 2D f(x,y), preferably a polynomial.

The interpolation runs over all *(2D-position on the detector, instrumental resolution)* reference points and gives you a 2D resolution map, i.e. a resolution value R = R(x,y) for each pixel on the detector 

**How to update :**

1. Get a reference of well-characterized instrumental resolution values spread over the detector.

   Standard spectrographs benefit from wavelength calibration information. The detector is illuminated with a thorium-argon lamp, so that for each thorium-argon line we know precisely where it falls on the detector (pixel on the x-axis and order on the y-axis) and its resolution. 
   For HARPS and HARPS-N, for example, this information is contained in the ``THAR_LINE_TABLE_THAR_FP_A.fits`` file as a product of their data reduction software.
   
   *Other echelle spectrographs are expected to provide a similar information in a similar format.*

   .. Note:: 
     If your instrument performances change over time, you need to compute a specific resolution map for each epoch (using different reference points for the interpolation every time). 

2. Get the best-fit 2D function for the interpolation

   We have tested several 2D polynomial functions p(x,y) up to the fourth degree. Our metric to evaluate the goodness of a fit is the BIC. 
   You can use the ``Instrumental_resolution_`` functions for HARPS and HARPS-N as a template, and adapt it to your needs and your specific detector.
   In principle, you only need to modify the first part of the function and provide specific information related to your instrument (e.g., the number of orders).
   These functions have two use cases : automatically test several interpolating functions, or directly generate a resolution map with a specified interpolating function. 

Several plotting functions are also provided. They are optimized for the above-mentioned instruments only, but can be easily adapted to other similar echelle spectrographs.


