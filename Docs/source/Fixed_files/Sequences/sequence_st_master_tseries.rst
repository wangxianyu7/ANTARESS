.. raw:: html

    <style> .orange {color:DarkOrange} </style>

.. role:: orange

.. raw:: html

    <style> .green {color:green} </style>

.. role:: green

Stellar master
==============

Description
-----------

This sequence generates a master spectrum from a time-series of 2D echelle spectra in ESO format. 

First, put all files to be binned within the same directory of your choosing, so that ``ANTARESS`` considers them as a single visit.
You need to provide both the :orange:`_S2D_A.fits` and :orange:`_S2D_BLAZE_A.fits` files, which can be retrieved as explained `here <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/data_access.html>`_.  

Then copy the `configuration file <https://gitlab.unige.ch/spice_dune/antaress/-/blob/main/src/antaress/ANTARESS_launch/ANTARESS_settings.py>`_ as :orange:`ANTARESS_settings_sequence.py` in your working directory, and modify the following fields::

 gen_dic['type'] = {instrument:'spec2D'}
 gen_dic['data_dir_list'] = {instrument:{'all':data_path}}

Where :green:`instrument` is the name of the spectrograph used to acquire your dataset, and :green:`data_path` is the absolute path toward your dataset directory.
By default the sequence generates a 1D master spectrum. If you prefer to generate the master in echelle format, with the spectrum set back to count levels, deactivate the 2D/1D conversion through::

 gen_dic['spec_1D_DI'] = False

The sequence can then be executed from your working directory as:: 

 antaress --sequence st_master_tseries --custom_settings ANTARESS_settings_sequence.py
 
 
Procedure
--------- 
 
This sequence carries out the following reduction steps:

- Data upload and formatting
- Calibration and noise estimates 
- Telluric correction
- Global flux balance correction
- Cosmics correction
 
Details about these steps can be found in the `reduction tutorial <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_proc/procedures_reduc.html>`_. 
The sequence then carries out the following processing steps:

- Alignment of disk-integrated profiles: this is done using the RV measurements provided within the .fits files, which correspond to the relative motion between the star and the Solar system barycenter. The master will thus be aligned in the stellar rest frame.
- Broadband flux scaling: the sequence assumes there are no exposures taken during planetary transits (it is up to the user to provide out-of-transit exposures as input), so that the scaling will simply set all spectra to a common flux level.

Details about these steps can be found in the `processing tutorial <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_proc/procedures_proc.html>`_. 
The sequence then carries out the following conversion steps:

- Conversion 2D/1D: unless you disabled this option to generate the master in echelle format
- Binning: this will perform a weighted average of all the processed exposures.

Details about these steps can be found in the `conversion tutorials <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_conv/procedures_conv.html>`_. 



Application
----------- 

We describe here how this sequence was combined with a wrap-up routine to automatically generate master stellar spectra over all datasets stored on the `DACE database <https://dace.unige.ch/>`_.

The wrap-up routine loops over all available datasets, defined as the exposures obtained for a given star with a given instrument configuration (we distinguish between different upgrades of the same instrument, that may have led to changes in their LSF).
For each dataset, the routine first checks whether master spectra already exist. In that case, it evaluates the relevance of updating the masters, based on the following criteria:

- New exposures were obtained since the previous masters were computed
- **TBD** : the new SNR*BERV is higher than the existing one ? And/or that the cumulated SNR of the new selection is higher than the one from the previous selection ?

If masters are to be computed for the dataset, the routine then applies the following selection criteria:

1. If the star is known to host transiting planets, get their mid-transit reference time :math:`T_0^{ref}`, orbital period `P`, and transit duration :math:`T_{14}` from the `NASA Exoplanet Archive <https://exoplanetarchive.ipac.caltech.edu/>`_, and exclude exposures that fall within
   :math:`T_0^{n}` ± ( 0.5 x :math:`T_{14}` + :math:`\sigma[T_0^{n}]`), where :math:`\sigma[T_0^{n}]` is the uncertainty on :math:`T_0^{n}` propagated `n` transits away from :math:`T_0^{ref}`.
2. Continue if there are at least :math:`N_{min}` (default 5) exposures in the dataset.
3. Exclude exposures with RVs and FWHM outliers, based on a MAD threshold (default 5).
4. Keep the :math:`F_{high}` percentile (default 10\%) of exposures with highest merit value, up to a maximum of :math:`N_{max}`/2 exposures (default :math:`N_{max}` = 100), in each half of the exposure BERV distribution separated at the median `<BERV>`. 
   The merit criterion for exposure `i` is defined as :math:`S/R_i` x (:math:`BERV_i  - <BERV>`).
   This approach ensures the selection of about the same number of exposures with high S/R at both extremes of the BERV distribution, so that the flux masked in spectral ranges contamined by deep telluric lines at minimum (resp. maximum) BERV can be estimated from exposures acquired at maximum (resp. minimum) BERV.

The selected exposures are then passed as input to ``ANTARESS`` through a dedicated function that calls the sequence a first time to reduce and process the 2D exposures, and compute a master 2D spectrum, and then calls the sequence a second time to retrieve the processed 2D exposures, and compute a 1D spectrum.
Masters are available on the DACE platform under **link or typical access TBD**

