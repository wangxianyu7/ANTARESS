.. raw:: html

    <style> .orange {color:DarkOrange} </style>

.. role:: orange

.. raw:: html

    <style> .green {color:green} </style>

.. role:: green

.. raw:: html

    <style> .Magenta {color:Magenta} </style>

.. role:: Magenta

Wiggle correction
=================

This tutorial is part of the `spectral reduction <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_reduc/procedures_reduc.html>`_ steps.

This module offers two methods to correct for the ESPRESSO *wiggles*, using either an analytical model over a full visit dataset (preferred approach), or a filter in each exposure. 
The analytical model describes a beat pattern between two sine-like components, which are dominating the wiggle pattern in most ESPRESSO datasets. Because this model only describes the wiggle signal and is constrained by all exposures, it prevents overfitting and provides a robust and homogeneous correction between ESPRESSO datasets.
The filter approach is more efficient and easier to set-up, but introduces the risk of overcorrecting planetary and stellar features at medium spectral resolution. It should be limited to observations in which the wiggle pattern is too complex to be captured with the analytical model.
 
Activate the ``ESPRESSO "wiggles"`` module (:green:`gen_dic['corr_wig']= True`) and set it to *calculation* mode (:green:`gen_dic['calc_wig'] = True`).

Initialization
--------------

Wiggles are processed as a function of light frequency (:math:`\nu = c/\lambda`), to be distinguished from the frequency `F` of the wiggle patterns.

..
 Erik: missing all the init fields here




Screening
---------

This step is common to both wiggle correction methods. It serves two purposes:

+ Identifying the spectral ranges to be included in the analysis (i.e., where wiggles are significant compared to noise)
+ Determining which wiggle components affect your dataset.

Activate the screening submodule with :green:`gen_dic['wig_exp_init']= True`. You can leave the other submodule settings to their default value.

Plots of the spectral ratios between each exposure spectrum and a chosen master (ie, transmission spectra) are automatically saved in the :orange:`/Working_dir/Star/Planet_Plots/Spec_raw/Wiggles/Exp_fit/Instrument_Visit/Init/` directory.
As illustrated in :numref:`screening`, wiggles usually decrease in amplitude toward the blue of the spectrum, where the S/R also decreases (as the flux gets lower due to Earth diffusion and the black body of typical exoplanet host stars). You thus need to decide beyond which light frequency the transmission spectra do not bring any more constraints to the wiggle characterization. Here we chose :math:`\nu` = 6.73 nHz.   
Wiggles are also typically dominated by noise toward the center of the center of the spectrum (:math:`\nu \sim` 57-58 nHz) at the edges of the blue and red detectors.

.. Tip:: 
 Although this is not the case in this example, some datasets may display an additional wiggle pattern with lower frequency and localized S-shaped features (typically at :math:`\nu \sim` 47, 51, 55 :math:`10^13` Hz).
 The current version of the analytical model does not account for this pattern. You can either ignore them, if they fall in a range you do not plan on analyzing, or follow the approach described in the next section.


.. figure:: wig_screening.png
  :width: 800
  :name: screening

  Transmission spectrum in one of the 20221117 exposures, as a function of light frequency. 
  The wiggle pattern is clearly visible, but dominated by noise at the center and blue end of the spectrum. The spectrum is colour coded by spectral order.



From the transmission spectrum identify spectral ranges that are too noisy to be included in the fit::

 gen_dic['wig_range_fit'] = { 
            '20221117': [[20.,57.1],[57.8,67.3] ],   
            '20231106': [[20.,50.6],[51.1,54.2],[54.8,57.1],[57.8,67.3] ],         
        }

The final transmission spectrum with the excluded regions should show some clear periodic signals, as shown in :numref:`screening_final`.

.. figure:: screening.png
  :width: 800
  :name: screening_final

  Final transmission spectrum after removing noisy regions. The bottom panel shows the mean periodogram computed for all exposures from the observation.

After excluding spectral ranges with high noise levels, the wiggle pattern and associated peaks in the periodogram should become clearly visible, as shown in :numref:`screening_final`. 
If they remain indistinct, wiggles may be small enough that a correction is not required. 
Otherwise you can now deactivate this step (:green:`gen_dic['wig_exp_init']= False`) and move on to either the :ref:`filter <Wig_sec_filt>`) or :ref:`analytical <Wig_sec_ana>` correction.



.. _Wig_sec_filt:

Method 1: filter
----------------

Activate the filter approach by setting :green:`mode` to :Magenta:`mode` in:: 

 gen_dic['wig_exp_filt']={
         'mode':True,
         'win':0.3,
         'deg':4,
         'plot':True
         }
         
Choose values for the filter smoothing window (:green:`win`) and polynomial degree (:green:`deg`) that are fine enough to capture the wiggle pattern without fitting spurious features in the data. 

The :green:`plot` field allows you to check the efficiency of the correction in the transmission spectra (saved in the :orange:`/Working_dir/Star/Planet_Plots/Spec_raw/Wiggles/Exp_fit/Instrument_Visit/Filter/` directory), as shown in :numref:`plot_filter`.

.. figure:: plot_filter.png
  :width: 800
  :name: plot_filter

  Transmission spectrum before and after filtering.


A drawback of this approach is that it may smooth out spectral features and potentially remove signals of planetary or stellar origin. 
However, this method allows you to isolate and correct specific spectral ranges in which unexpected features may appear that cannot be modeled analytically.
After this correction, you can then re-inject into the wiggle module the corrected spectra and apply the analytical model.




.. _Wig_sec_ana:

Method 2: Analytical model
--------------------------

We have determined from previous analysis that wiggles are best described as the sum of multiple sinusoidal components:

:math:`W(\nu, t) = 1 + \sum _k A_k(\nu, t) \sin(2\pi \int (F_k(\nu,t)d\nu ) - \Phi_k(t)).`

This module follows an iterative approach to determine the best-fitting parameters to model the wiggle pattern. 
The first two key components to estimate are the frequencies and amplitudes, denoted as :math:`F_k(\nu)` and :math:`A_k(\nu)`, respectively. 
They are expressed as polynomial expansions:

:math:`A_k (\nu, t) = \sum_{i=0}^{d_{a,k}} a_{\text{chrom},k,i}(t)(\nu - \nu_{\text{ref}})^i`,

:math:`F_k (\nu, t) = \sum_{i=0}^{d_{f,k}} f_{\text{chrom},k,i}(t)(\nu - \nu_{\text{ref}})^i`.

Where:

+ :math:`A_k(\nu,t)` represents the amplitude variation as a function of light frequency and time.
+ :math:`F_k(\nu,t)` represents the frequency variation as a function of light frequency and time.
+ :math:`\nu_\text{ref}` is a light frequency reference used for normalization.
+ :math:`d_\text{a,k}` and :math:`d_\text{f,k}` define the polynomial order for amplitude and light frequency variations.
+ The coefficients :math:`a_\text{chrom,k,i}(t)` and :math:`f_\text{chrom,k,i}(t)` capture the chromatic dependence of the amplitude and light frequency, respectively.
+ :math:`\Phi_k(t)` represents the phase offset of the sinusoidal comopnent at time :math:`t`.


Step 1: Sampling Chromatic Variations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the screening step you identified spectral regions that can be used to constrain the wiggle pattern and assess the strength of its components. 
In this step, activated with :green:`gen_dic['wig_exp_samp']['mode']= True`, you will sample the chromatic variations of the wiggle component across a representative set of exposures.
This means sampling the frequency and amplitude of each component as a function of light frequency :math:`\nu`.
 
First, select a set of exposures to sample using:: 

 gen_dic['wig_exp_in_fit'] =  {
    '20221117':np.arange(0,28,5),
    '20231106':np.arange(0,54,5)}
    

.. Tip:: 
 Since the wiggle pattern evolves relatively slowly with time, we do not need to sample their variations in every exposure.
 For the TOI-421 datasets, we thus sample one every fifth exposure.
 
In narrow bands, the wiggles can be approximated by a sine with constant frequency and amplitude.     
The sampling is thus performed automatically by sliding a fixed window over a given transmission spectrum, and at each sampled position:

+ Apply a periodogram to estimate the local wiggle frequency :math:`F_k(\nu)`.
+ Fit a sine function at this frequency to estimate the local wiggle amplitude :math:`A_k(\nu)`.
        
The window size must be large enough to include several oscillation cycles of the wiggle pattern. 
Furthermore, we recommend overlapping successive windows to sample more finely the wiggle pattern. 

This is an iterative process. Once the first component is processed, the piecewise model built over the sliding windows is used to temporarily correct the transmission spectrum (:numref:`samp_1`). 
The second component can then be sampled and analysed in the same way (:numref:`samp_2`).
We describe below the settings controlling this process, using the 20221117 visit as example.

Process the highest-frequency component by setting::

 gen_dic['wig_exp_samp']['comp_ids'] = [1]
 
Once this first component is analysed, you will correct it and process the second component by setting :green:`[1,2]`. 
You need to provide an estimate of each component frequency :math:`F_k(\nu)`, described as a polynomial with coefficients :green:`ci`:: 

 gen_dic['wig_exp_samp']['freq_guess']:{
         1:{ 'c0':3.72, 'c1':0., 'c2':0.},
         2:{ 'c0':2.05, 'c1':0., 'c2':0.}}   

Approximating :math:`F_k(\nu)` to a constant is usually sufficient for this step, and the above values should be similar for other datasets.
This guess frequency is also used to calculate the width of the sliding window, which is set by the number of cycles you want to sample for each component::

 gen_dic['wig_exp_samp']['nsamp'] = { 1 : 8 , 2 : 8 } 

The oversampling of the sliding windows is controlled by shifts in :math:`\nu` (in :math:`10^13 s^{-1}`) set as::

 gen_dic['wig_exp_samp']['sampbands_shifts'] = {
     1:np.arange(16)*0.15,
     2:np.arange(16)*0.30 }

The pipeline loops over the shifts, positions the first window at the lowest :math:`\nu` of the spectrum plus the shift, and then slides the window over consecutive (non overlapping) positions.
When processing the second component, the first one is corrected for using the piecewise model built over the sliding windows positioned for a given shift, whose index within :green:`sampbands_shifts` is set as:: 

 gen_dic['wig_exp_samp'][2] = 0

Periodograms associated with each window are searched for the peak wiggle frequency over a broad default range (:green:`'mod':None`), or within a range (in :math:`10^13 s^{-1}`) centered on the guess :math:`F_k(\nu)` for this window::

 gen_dic['wig_exp_samp']['src_perio']={
         1:{'mod':'slide','range':[0.5,0.5] ,'up_bd':False },
         2:{'mod':'slide','range':[0.5,0.5] ,'up_bd':True  }}

Where :green:`'up_bd':True` restricts the range upper boundary for the second component to the first component frequency.
The sine fit is then only performed in the window if the FAP of its peak periodogram frequency is below a threshold (in \%)::

 gen_dic['wig_exp_samp']['fap_thresh'] = 5
 
To better converge on the sine fit it is repeated iteratively :green:`gen_dic['wig_exp_samp']['nit']` times. 

You can improve the quality of the sampling after having completed :ref:`Step 3 <Wig_sec_ana3>`, by fixing the frequency of the components (here the first) to its value from the best-fit model in each exposure::

 gen_dic['wig_exp_samp']['fix_freq2expmod'] = [1]

And after having completed :ref:`Step 5 <Wig_sec_ana3>`, by fixing the frequency of the components to their values from the best-fit model in each visit (stored at a given path)::

 gen_dic['wig_exp_samp']['fix_freq2vismod'] = { 
     comps:[1,2] , 
     '20221117' : 'path1/Outputs_final.npz', 
     '20231106' : 'path2/Outputs_final.npz' }

The :green:`gen_dic['wig_exp_samp']['plot']` field plots the sampled transmission spectra and sampling analyses, stored under :orange:`/Working_dir/Star/Planet_Plots/Spec_raw/Wiggles/Exp_fit/Instrument_Visit/Sampling/`.

.. figure:: wiggle_sampling_1.png
  :width: 800
  :name: samp_1

  Sampling of the first wiggle component in the 20221117 visit.

.. figure:: wiggle_sampling_2.png
  :width: 800
  :name: samp_2

  Sampling of the second wiggle component in the 20221117 visit. The piecewise model built from the sampling of the first component has been corrected for.
  
You can now deactivate this step (:green:`gen_dic['wig_exp_samp']['mode']= False`) and move on to the next one.


Step 2: Chromatic analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~
  

.. _Wig_sec_ana3:

Step 3: Exposure fit
~~~~~~~~~~~~~~~~~~~~


Step 4: Pointing Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~




.. _Wig_sec_ana5:

Step 5: Visit fit
~~~~~~~~~~~~~~~~~



Applying the correction
-----------------------

