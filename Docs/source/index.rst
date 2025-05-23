.. raw:: html

    <style> .orange {color:DarkOrange} </style>

.. role:: orange

.. raw:: html

    <style> .lskblue {color:LightSkyBlue} </style>

.. role:: lskblue

:lskblue:`A`\dvanced and :lskblue:`N`\eat :lskblue:`T`\echniques for the :lskblue:`A`\ccurate :lskblue:`R`\etrieval of :lskblue:`E`\xoplanetary and :lskblue:`S`\tellar :lskblue:`S`\pectra
*******************************************************************************************************************************************************************************************


About ANTARESS
==============

The ``ANTARESS`` workflow, implemented as a Python-3 library, is a set of methods to process high-resolution spectroscopy datasets in a robust way and extract accurate exoplanetary and stellar spectra. While fast preliminary analysis can be run on order-merged 1D spectra and CCF, the workflow was optimally designed for extracted 2D echelle spectra. Input data from multiple instruments and epochs can be corrected for environmental and instrumental effects, processed homogeneously, and analyzed independently or jointly. 

The two main applications of the workflow are the extraction and analysis of planet-occulted stellar spectra along the transit chord, and planetary atmospheric spectra in absorption and emission. Stellar spectra, cleaned from planetary contamination, provide direct comparison with theoretical stellar models and enable a spectral and spatial mapping of the photosphere, allowing in particular to derive the orbital architecture of planetary systems thanks to the Rossiter-McLaughlin Revolutions technique. Planetary spectra, cleaned from stellar contamination, in turn provide direct comparison with theoretical atmospheric models and enable a spectral and longitudinal mapping of the atmospheric layers.

Because the workflow is modular and its concepts are general, it can support new methods, be extended to additional spectrographs, and find a range of applications beyond the proposed scope. Look `here <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/contributions.html>`_ if you are interested in collaborations or contributions to ``ANTARESS``.

Please cite:

- `Bourrier et al. 2024, A&A, 691, A113 <https://www.aanda.org/articles/aa/full_html/2024/11/aa49203-24/aa49203-24.html>`_: if you use ``ANTARESS`` for any purpose.
- TBD: if you use ``ANTARESS`` to extract and analyze planetary spectra.
- TBD: if you use ``ANTARESS`` to extract and analyze spotted stellar spectra.

Documentation
=============

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Fixed_files/installation
   Fixed_files/contributions
   Fixed_files/getting_started
   Fixed_files/data_access
   Fixed_files/notebooks   
   Fixed_files/sequences
   Fixed_files/procedures
   Fixed_files/documentation
