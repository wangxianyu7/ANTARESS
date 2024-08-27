.. raw:: html

    <style> .orange {color:DarkOrange} </style>

.. role:: orange

.. raw:: html

    <style> .green {color:green} </style>

.. role:: green

.. raw:: html

    <style> .Magenta {color:Magenta} </style>

.. role:: Magenta

Rossiter McLaughlin analysis
============================

These procedures relate to the analysis of the Rossiter McLaughlin signal.

We assume here that you have already `set up your system <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_setup/procedures_setup.html>`_, processed a `spectral transit time-series <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_reduc/procedures_reduc.html>`_, extract `planet-occulted stellar profiles <TBD>`_, and 
convert them into `intrinsic CCFs <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_CCF/procedures_CCF_Intr/procedures_CCF_Intr.html>`_.

Analysis of individual intrinsic CCFs
-------------------------------------

The goal of this step is to identify which exposures should be included in the RM Revolutions fit, by assessing the quality of intrinsic CCFs. You will first fit these CCFs, and then analyze the derived properties.

To fit the CCFs, activate the ``Intrinsic profiles analysis`` module by setting :orange:`gen_dic['fit_Intr'] = True`.






Run this cell to perform a Gaussian fit to each intrinsic CCF. This will return the rv, FWHM, and contrast time-series of the average stellar line from regions occulted along the transit chord, which you can visualize below. 

Intrinsic CCFs are usually measured with low S/N. In that case, you can run the fits using a MCMC (`fit_mode = "MCMC"`) rather than the default least-square minimization (`fit_mode = "chi2"`). To overwrite the default priors, you can further uncomment the `priors` field and define the lower and upper boundary of the uniform prior ranges. Look at the time-series of properties plotted below to adjust the priors, or at the MCMC chains and corner plots in the directories `/Working_dir/planet_Saved_data/Introrig_prop/instrument_night_mcmc/iexp*/`. Since running the MCMC may take some time, you can set `calc_fit` to `False` once the MCMC is done and you only want to manipulate the plots.


