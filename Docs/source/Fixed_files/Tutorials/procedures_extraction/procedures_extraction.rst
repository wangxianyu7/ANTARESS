.. raw:: html

    <style> .orange {color:DarkOrange} </style>

.. role:: orange

.. raw:: html

    <style> .green {color:green} </style>

.. role:: green

Profile extraction
==================

The present tutorial relates to the extraction of differential and intrinsic profiles from a time-series of disk-integrated profiles. 
The generic term profile designates extracted 2D echelle spectra, order-merged 1D spectra, or cross-correlation functions (CCFs).
We illustrate the tutorial with an extraction from 2D ESPRESSO spectra of TOI-421c acquired on 2023-11-06. We use another visit, acquired for TOI-421b on 2022-11-17, to illustrate multi-visits possibilities. 

We assume that you have already `set up your system and dataset <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_setup/procedures_setup.html>`_, `reduced <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_reduc/procedures_reduc.html>`_ and `processed <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_proc/procedures_proc.html>`_ your spectral transit time-series.


.. _Extra_sec_Diff:

Differential profiles
---------------------

Differential profiles are the difference between a master profile for the disk-integrated star uncontaminated by any planet, and the profiles from each exposure in your time-series.
At this stage, differential profiles may contain planetary signatures as well as the spectrum of the stellar region occulted by the planet during its transit. 
Differential profiles can thus be used outside of ``ANTARESS`` to compute transmission spectra and fit for atmospheric properties, or they can be further processed with the workflow to isolate specific planetary and stellar contributions, as described in this tutorial. 

Activate the ``Differential profiles extraction`` module (:green:`gen_dic['diff_data'] = True`) and set it to *calculation* mode (:green:`gen_dic['calc_diff_data'] = True`).
 
In the present version of ``ANTARESS`` a single master profile is defined for each visit. 
You can control which visits should be used to calculate the master associated with a given instrument (and thus with all its visits), through::

 data_dic['Diff']['vis_in_bin']={'ESPRESSO':{['20221117','20231106']} 

Here we would request that the master used for all processed ESPRESSO visits is calculated from visits :green:`20221117` and :green:`20231106`.
You can further control which exposures in each visit should be used for the calculation of the master through::

 data_dic['Diff']['idx_in_bin']={'ESPRESSO':{['20231106':range(15)]}   

Here we would request that the master is calculated using only the 15 pre-transit exposures in the :green:`20231106` visit.
It is possible to leave both :green:`vis_in_bin` and :green:`idx_in_bin` empty, in which case the master of each visit is calculated using all its own out-of-transit exposures. 
This is the default setting, as the stellar line is expected to vary in shape between visits and the master is more representative of in-transit exposures when calculated from the adjacent stellar baseline.
Combining multiple visits may however be relevant when few out-of-transit exposures could be obtained in a given visit.

.. Tip::
   Each exposure is automatically identified as in- or out-of-transit by ``ANTARESS``, unless you force its status with :green:`data_dic['DI']['idx_ecl']`.
   A quick way to assess the status of an exposure and get its global or in-transit index is to plot its visit light curve (:green:`plot_dic['input_LC']='pdf'`). 
    
You can plot the differential profiles from each exposure (:numref:`Fig_DiffProf_TOI421`) with :green:`plot_dic['Diff_prof']='pdf'` (saved in :orange:`/Working_dir/Star/Planet_Plots/Instrument_Visit_Indiv/Data/`) and a map of all profiles (:numref:`Fig_DiffMap_TOI421`) 
with :green:`plot_dic['map_Diff_prof']='pdf'` (saved in :orange:`/Working_dir/Star/Planet_Plots/Diff_data/Instrument_Visit_Map/Data/?`). These plots are useful to check the noise quality and the presence of unexpected features in out-of-transit differential profiles, but
we note that in-transit differential profiles are difficult to compare due to their varying flux level (resulting from the broadband planet occultation and stellar intensity variations). 
This comparison will be easier to perform on intrinsic profiles.

.. figure:: Diff_Prof.png
  :width: 800
  :name: Fig_DiffProf_TOI421
  
  Example of differential spectral profile during the transit of TOI-421c, in the region of the sodium doublet.

.. figure:: DiffMap_TOI421.png
  :width: 800
  :name: Fig_DiffMap_TOI421
  
  Flux map of differential spectral profiles over the 20231106 visit, as a function of wavelength in the star rest frame (in abscissa) and orbital phase (in ordinate).
  Transit contacts are shown as green dashed lines. 
  
You can now set the present module to *retrieval* mode (:green:`gen_dic['calc_Diff_data'] = False`) if you want to further process differential profiles. Otherwise they can be retrieved for external use
in the :orange:`/Working_dir/Star/Planet_Saved_data/Res_data/` directory.
  

.. _Extra_sec_Intr:

Intrinsic profiles
------------------

Assuming that the master profile used in :ref:`the previous section <Extra_sec_Diff>` is representative of the star over the entire visit, the differential profile in a given in-transit exposure can be expressed as the 
spectrum of the stellar region occulted by the planet during this exposure multiplied by the sum of the equivalent surfaces from:
 
- the optically thick layers of the planet atmospheric disk, whose apparent size varies slowly with wavelength.
- the optically thin layers of the planet atmospheric limb, whose apparent size varies sharply with wavelength in the transitions associated with species present in the atmosphere.
 
This module processes in-transit differential profiles to isolate the *intrinsic* narrow stellar lines of the planet-occulted regions, corrected for planetary contamination and broadband intensity variations.

..
  Spectra can be expressed as
  Fstar = sum(unocc,fi*si)   + sum(occ thick , fi_t*si ) +  sum(occ thin , fi*si )
  Fin =   sum(unocc,fi_t*si) +         0                 +  sum(occ thin , fi_t*si*Ti_t )
  Thus differential spectra as
  Fstar - Fin = sum(unocc, (fi - fi_t)*si)  
              + sum(occ thick , fi_t*si )
              + sum(occ thin , (fi - fi_t*Ti_t)*si)
  Assuming the star remains stable:
  Fstar - Fin = 0
              + sum(occ thick , fi*si )
              + sum(occ thin , (1-Ti)*fi*si)
  Assuming uniform stellar emission and atmospheric properties over the occulted regions:
  Fstar - Fin = focc*Sthick + (1-T)*focc*Sthin    
  With A = 1-T the absorption from the atmosphere (0 if transparent, 1 if fully opaque)

Before activating the module you need to consider whether the planetary limb contaminates the stellar spectrum in the spectral range you are interested in.
If that is the case, provide the path to a file containing the list of lines you want to mask (see the configuration file for possible file formats)::

  data_dic['Atm']['CCF_mask'] = file_path

Then define the radial velocity range that you want to mask around each line, for example 20 km/s on both sides::

  data_dic['Atm']['plrange'] = [-20.,20.] 

Finally request from the workflow that those lines are masked in intrinsic profiles with::

  data_dic['Atm']['no_plrange'] = ['Intr']
  
.. Tip::
   If your goal is to perform a `Rossiter-McLaughlin analysis <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_RM/procedures_RM.html>`_ you likely do not need to 
   exclude lines absorbed by the planetary atmosphere, as the RM analysis relies on CCFs built with a list of lines typically found in the stellar atmosphere but not in the planet atmosphere. 
   This may become relevant for ultra-hot Jupiters, whose atmosphere may absorb in the same iron lines as the host star.

You can now activate the ``Intrinsic profiles extraction`` module (:green:`gen_dic['intr_data'] = True`) and set it to *calculation* mode (:green:`gen_dic['calc_intr_data'] = True`).

Broadband contributions from the optically thick layers of the planetary atmosphere and from the stellar spatial intensity variations are automatically corrected for, so that the module does
not require more input than you already provided in previous steps of the workflow. 

However, the module allows you to control more finely the intrinsic profile continuum, which is the range assumed to represent the part of the stellar spectrum outside of absorption lines. 
Knowledge of this continuum is required in other modules to analyze intrinsic profiles, and in the present module if you want to adjust their flux level. 
Indeed, imprecisions from the `flux balance corrections <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_reduc/procedures_reduc.html>`_ of the disk-integrated spectra may result in slight deviations from 
the continuum level that should be common to all intrinsic profiles. The continuum range is defined through::

 data_dic['Intr']['cont_range'] = {instrument:{idx_order:[[x1,x2],[x3,x4]]}
 
Where :green:`idx_order` is the index of the spectral order (0 if you intend to convert intrinsic spectra into CCFs), and :green:`x` defines the boundaries of the ranges covering the continuum in the star rest frame (in :math:`\\A` for spectra and km/s for CCF). 
The continuum range is specific to a given instrument because the measured line widths depend on the spectrograph broadening. 
Adjustment of the continuum level is then activated through::
    
 data_dic['Intr']['cont_norm'] = False
 
And is automatically applied to the latest processed intrinsic profiles (eg, if intrinsic spectra are converted into CCFs in a subsequent module the adjustment will be applied to the intrinsic CCFs).


You can plot intrinsic profiles from each exposure (:numref:`Fig_IntrProf_TOI421`) with :green:`plot_dic['Intr_prof']='pdf'` (saved in :orange:`/Working_dir/Star/Planet_Plots/Intr_data/Instrument_Visit_Indiv/Data/Data_type/`) and a map of all profiles (:numref:`Fig_IntrMap_TOI421`) 
with :green:`plot_dic['map_Intr_prof']='pdf'` (saved in :orange:`/Working_dir/Star/Planet_Plots/Intr_data/Instrument_Visit_Map/Data/Data_type/`). If your dataset has sufficient S/R the map will reveal bright tracks that corresponds to the stellar absorption lines shifting with the 
radial velocity of the stellar regions occulted along the transit chord.

.. figure:: Intr_Prof.png
  :width: 800
  :name: Fig_IntrProf_TOI421
  
  Example of intrinsic spectral profile during the transit of TOI-421c, in the region of the sodium doublet.

.. figure:: IntrMap_TOI421.png
  :width: 800
  :name: Fig_IntrMap_TOI421
  
  Flux map of intrinsic spectral profiles during the transit of TOI-421c, as a function of wavelength in the star rest frame (in abscissa) and orbital phase (in ordinate).
  Transit contacts are shown as green dashed lines. Solid green lines highlight the track of the planet-occulted stellar lines. 

You can now set the present module to *retrieval* mode (:green:`gen_dic['calc_intr_data'] = False`) if you want to further process intrinsic profiles. Otherwise they can be retrieved for external use
in the :orange:`/Working_dir/Star/Planet_Saved_data/Intr_data/` directory.
