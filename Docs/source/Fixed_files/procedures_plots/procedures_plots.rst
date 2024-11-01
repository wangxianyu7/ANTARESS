.. raw:: html

    <style> .orange {color:DarkOrange} </style>

.. role:: orange

.. raw:: html

    <style> .green {color:green} </style>

.. role:: green

.. raw:: html

    <style> .Magenta {color:Magenta} </style>

.. role:: Magenta

Plot tutorials
==============

This tutorial explains how to customize ``ANTARESS`` plots.
All plots are activated via the `configuration file <https://gitlab.unige.ch/spice_dune/antaress/-/blob/main/src/antaress/ANTARESS_launch/ANTARESS_settings.py>`_
We detail here the options available through the plot configuration file `plot configuration file <https://gitlab.unige.ch/spice_dune/antaress/-/blob/main/src/antaress/ANTARESS_plots/ANTARESS_plot_settings.py>`_.

Disk-integrated profiles : Individual profiles
----------------------------------------------
----------------------------------------------

Original spectra
----------------

You can plot flux spectra at various stages of the processing with :green:`plot_dic['sp_raw']='pdf'`.

You can normalize spectra with :green:`'norm_prof'`, control the plotted range with :green:`'x_range'`, the plotted orders with :green:`'orders_to_plot'`, the plotted exposures with :green:`'iexp_plot'`, and overplot exposures with :green:`'multi_exp'`.



Transmission spectra
--------------------



   
Flux balance
------------

You can plot flux balance variations within each visit with :green:`plot_dic['Fbal_corr']='pdf'`, and between visits with :green:`plot_dic['Fbal_corr_vis']='pdf'`.

You can set a gap between plotted exposures (e.g., 0.1) through the keyword :green:`'gap_exp'`, which will make inspecting the variations and model fit much easier.















