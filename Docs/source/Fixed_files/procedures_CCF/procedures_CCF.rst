.. raw:: html

    <style> .orange {color:DarkOrange} </style>

.. role:: orange

.. raw:: html

    <style> .green {color:green} </style>

.. role:: green

CCF computations
================

``ANTARESS`` tutorials detail how to use the configuration file (which you should have copied in your working directory from the `default configuration file <https://gitlab.unige.ch/spice_dune/antaress/-/blob/main/src/antaress/ANTARESS_launch/ANTARESS_settings.py>`_) to carry out a specific procedure. 
Additional options are typically available in this file, so do not hesitate to read it in details. 
``ANTARESS`` plots are activated via the configuration file and have default settings, which can be extensively customized through the plot configuration file (which you should also have copied in your working directory from the `default plot configuration file <https://gitlab.unige.ch/spice_dune/antaress/-/blob/main/src/antaress/ANTARESS_plots/ANTARESS_plot_settings.py>`_) 
and detailed in the `plots tutorial <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_plots/procedures_plots.html>`_.

The present tutorial relates to the cross-correlation of 2D echelle spectra from the star or from the planetary atmosphere with weighted masks. 
We illustrate the tutorial with 2D ESPRESSO spectra of TOI-421c acquired on 2023-11-06.  
After being converted with the relevant module, all subsequent processing of the datasets by the workflow will be performed under CCF form.

We assume that you have already `set up your system and dataset <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_setup/procedures_setup.html>`_ and `reduced <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_reduc/procedures_reduc.html>`_ your spectral time-series.

.. toctree::
   :hidden:

   procedures_CCF_mask/procedures_CCF_mask

.. _CCF_sec_DI:

Disk-integrated CCFs
--------------------
   
For this conversion we assume that you have `processed <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_proc/procedures_proc.html>`_ your spectral transit time-series.

First, you need to define the radial velocity (rv) grid upon which CCFs will be computed, by choosing its boundaries and step size (in km/s)::

 gen_dic['start_RV'] = -100.    
 gen_dic['end_RV']   = 100.
 gen_dic['dRV']      = None 

Boundaries are defined in the solar barycentric rest frame. By default the step size is left undefined, so that CCFs are automatically computed using the spectrograph pixel size and the introduction of correlated noise is limited.
One the major interests of ``ANTARESS``, however, is a careful propagation of covariances in the processed datataset and its accounting in fits. This allows you to oversample CCFs compared to the instrumental pixel size to potentially better resolve the stellar line profiles.

Then, you need to provide a weighted mask as::

 gen_dic['CCF_mask'] = {'ESPRESSO':mask_path}

The mask is specific to a given spectrograph because the spectral coverage and resolution may change the contrast and wavelength distribution of the measured stellar lines.
The configuration file provide more details about the format and content of the mask file. 
We note that `masks customized to your target star <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_CCF/procedures_CCF_mask/procedures_CCF_mask.html>`_ can be computed from the processed datasets with ``ANTARESS``.

By default the full spectrograph range is considered for CCF computation. You can however limit the computation to specific orders using::

 gen_dic['orders4ccf']={'ESPRESSO':order_idx}
 
Where :green:`order_idx` is a list of order indexes, counted from 0. 

.. Tip::
   Computing CCFs over a selection of orders can be useful to search for chromatic effects associated with the spectrograph or the Earth atmosphere.
   It is also a way to check whether an apparent `Rossiter-McLaughlin signature <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_RM/procedures_RM.html>`_ is truly arising 
   from a transiting planet, as we do not expect a strong chromatic dependence of the RM signal.
    

Activate the ``CCF conversion for disk-integrated spectra`` module (:green:`gen_dic['DI_CCF'] = True`) and set it to *calculation* mode (:green:`gen_dic['calc_DI_CCF'] = True`). 
    
The workflow converts 2D echelle spectra into CCFs after they have been reduced but before they have been `aligned in the star rest frame <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_proc/procedures_proc.html>`_.
This allows using disk-integrated CCFs to search for systematic trends in the stellar line shape and position over the processed visit, and to measure the systemic rv of the system, as described in the `detrending tutorial <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_analysis/procedures_DI_ana/procedures_detrending.html>`_.

.. Tip::
   You can recompute the CCFs after activating the detrending module, to check the quality of your correction. 

Once all datasets have been converted you can set the present module to *retrieval* mode (:green:`gen_dic['calc_DI_CCF'] = False`) if you want to further process the disk-integrated CCFs. 
Otherwise they can be retrieved for external use in the :orange:`/Working_dir/Star/Planet_Saved_data/DI_data/CCFfromSpec/` directory.






.. _CCF_sec_Intr:

Intrinsic CCFs
--------------

For this conversion we assume that you have `processed <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_proc/procedures_proc.html>`_ your spectral transit time-series and extracted the `intrinsic stellar profiles <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_extraction/procedures_extraction.html>`_.

The rv grid, mask, and order selection are set up in the same way as for the disk-integrated CCFs (:ref:`section <CCF_sec_DI>`). 
The rv grid is automatically shifted by the pipeline from the solar barycentric rest frame to the star rest frame, based on your input `system properties <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_setup/procedures_setup.html>`_ and `systemic rv <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_reduc/procedures_reduc.html>`_

Activate the ``CCF conversion for intrinsic spectra`` module (:green:`gen_dic['Intr_CCF'] = True`) and set it to *calculation* mode (:green:`gen_dic['calc_Intr_CCF'] = True`). 

The module will also convert `out-of-transit differential spectra <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_extraction/procedures_extraction.html>`_ into CCFs, so that you can 
`evaluate the noise quality  <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_analysis/procedures_DI_ana/procedures_PCA.html>`_ of the dataset and better interpert intrinsic CCFs.

Once all datasets have been converted you can set the present module to *retrieval* mode (:green:`gen_dic['calc_Intr_CCF'] = False`) if you want to further process the intrinsic CCFs. 
Otherwise they can be retrieved for external use in the :orange:`/Working_dir/Star/Planet_Saved_data/Intr_data/CCFfromSpec/` directory (:orange:`/Diff_data/CCFfromSpec/` for out-of-transit differential CCFs).




.. _CCF_sec_Atm:

Atmospheric CCFs
----------------

WIP