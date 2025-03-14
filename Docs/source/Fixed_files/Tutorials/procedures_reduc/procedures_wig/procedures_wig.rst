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

TBD

Screening: identifying relevant spectral ranges
---------

This step is common to both wiggle correction methods. It serves two purposes:

+ Identifying spectral ranges where wiggles are significant compared to noise.
+ Determining which wiggle components affect your dataset.

Activate the screening submodule with :green:`gen_dic['wig_exp_init']= True`. You can leave the other submodule settings to their default value.

Plots of the spectral ratios between each exposure spectrum and a chosen master (ie, transmission spectra) are automatically saved in the :orange:`/Working_dir/Star/Planet_Plots/Spec_raw/Wiggles/Exp_fit/Instrument_Visit/Init/` directory.
As illustrated in :numref:`screening`, wiggles usually decrease in amplitude toward the blue of the spectrum, where the S/R also decreases (as the flux gets lower due to Earth diffusion and the black body of typical exoplanet host stars). You thus need to decide beyond which light frequency the transmission spectra do not bring any more constraints to the wiggle characterization. Here we chose :math:`\nu` = 6.73 nHz.   
Wiggles are also typically dominated by noise toward the center of the center of the spectrum (:math:`\nu \sim` 57-58 nHz) at the edges of the blue and red detectors.

.. Tip:: 
 Although this is not the case in this example, some datasets may display an additional wiggle pattern with lower frequency and localized S-shaped features (typically at :math:`\nu \sim` 47, 51, 55 :math:`10^13` Hz).
 The current version of the analytical model does not account for this pattern. You can exclude those S-shaped features to correct for the classical wiggles with the analytical model, and 
 ignore them (if they fall in a range you do not plan on analyzing) or correct them locally later on. Or you can use the filter approach, keeping in mind that you may overcorrect other signals of interest.   






.. figure:: wig_screening.png
  :width: 800
  :name: screening

  Transmission spectrum in one of the 20221117 exposures, as a function of light frequency. 
  The wiggle pattern is clearly visible, but dominated by noise at the center and blue end of the spectrum. The spectrum is colour coded by spectral order.

.. Erik
  can you redo this first figure with no excluded range at all ? (Do you mean y-range? Or have I already corrected for this?)

In general, you will see large noise levels at the 

From the transmission spectrum identify spectral ranges that are too noisy to be included in the fit::

 gen_dic['wig_range_fit'] = { 
            '20221117': [[20.,57.1],[57.8,67.3] ],   
            '20231106': [[20.,50.6],[51.1,54.2],[54.8,57.1],[57.8,67.3] ],         
        }

The final transmission spectrum with the excluded regions should show some clear periodic signals, as shown in :numref:`screening_final`.

.. figure:: screening.png
  :width: 800
  :name: screening_final

  Final transmission spectrum after removing the noisy regions. Bottom shows the mean periodogram computed for all exposures from the observation.

Plots are found in :orange:`/Working_dir/Star/Planet/Spec_raw/Wiggles/Exp_fit/Instrument_Visit/Init/`.

After removing spectral ranges with high noise levels, the wiggle pattern and periodic signals should become clearly visible, as shown in :numref:`screening_final`. If they remain indistinct, the wiggle correction will not be applied.

Method 1: filter
-------------------------------------

After removing the noisy ranges the wiggle pattern should be clearly visible from the screening. When the spectral ranges to be included have been defined you can charecatrise the wiggles using the filter approach. Choose values for 'win' and 'deg', that are fine enough to capture the wiggle pattern without fitting spurious features in the data.::

 gen_dic['wig_exp_filt']={
         'mode':True,
         'win':0.3,
         'deg':4,
         'plot':True
         }

A drawback of this approach is that it may smooth out spectral features and potentially remove signals of planetary or stellar origin. However, it is fast and easy to apply. Additionally, if unexpected features appear in the wiggle pattern that cannot be modeled analytically, this method allows you to isolate and correct those specific ranges. After addressing the abnormal features with the filter approach, you can then apply the analytical model to the remaining spectra.

Method 2: Analytical model
-------------------------------------

Previous analyses have shown that wiggles are best described as the sum of multiple sinusoidal components. The wiggle pattern can be expressed as:

:math:`W(\nu, t) = 1 + \sum _k A_k(\nu, t) \sin(2\pi \int (F_k(\nu,t)d\nu ) - \Phi_k(t)).`

This module follows an iterative approach to determine the best-fitting parameters for modeling the wiggle pattern. The first two key components to estimate are the frequencies and amplitudes, denoted as :math:`F_k(\nu)` and :math:`A_k(\nu)`, respectively. These are expressed as polynomial expansions:

:math:`A_k (\nu, t) = \sum_{i=0}^{d_{a,k}} a_{\text{chrom},k,i}(t)(\nu - \nu_{\text{ref}})^i`,

:math:`F_k (\nu, t) = \sum_{i=0}^{d_{f,k}} f_{\text{chrom},k,i}(t)(\nu - \nu_{\text{ref}})^i`.

Where:

+ :math:`A_k(\nu,t)` represents the amplitude variation as a function of frequency and time.
+ :math:`F_k(\nu,t)` represents the frequency variation as a function of frequency and time.
+ :math:`\nu_\text{ref}` is a reference frequency used for normalization.
+ :math:`d_\text{a,k}` and :math:`d_\text{f,k}` define the polynomial order for amplitude and frequency  ariations.
+ The coefficients :math:`a_\text{chrom,k,i}(t)` and :math:`f_\text{chrom,k,i}(t)` capture the chromatic  ependence of the amplitude and frequency, respectively.
+ :math:`\Phi_k(t)` represents the phase shift of the sinusoidal comopnent at time :math:`t`.


Step 1: Sampling Chromatic Variations
-------------------------------------
In an earlier step, screening, you should have identified spectral regions that can be used to constrain the wiggle pattern and assess the strength of its components. The next step is to sample the chromatic variations across a set of exposures.

Here, we sample the frequency and amplitude of the wiggle components as a function of frequency :math:`\nu`. Select a set of exposures for sampling under the field ``Exposures to be characterized``. For TOI-421, we sample every fifth exposure:
::
 gen_dic['wig_exp_in_fit'] =  {
    '20221117':np.arange(0,28,5),
    '20231106':np.arange(0,54,5)
        }

Chromatic Sampling Process
----
For chromatic sampling, we use a sliding window over each transmission spectrum to:

+ Identify the strongest peak in each window at every window position.
+ Fit a sine function to the window spectrum using the frequency of the strongest peak.

In narrow bands, the wiggles can be approximated by constant frequencies. In this step, we sample the frequencies :math:`F_k(\nu)` and amplitudes :math:`A_k(\nu)` for each window position.

The window size must be large enough to include several oscillation periods of the frequency. Additionally, successive window positions overlap to ensure enough measurements are sampled. For 20221117, we applied the following settings for the chromatic sampling of the first component:
::
 gen_dic['wig_exp_samp']={
     'mode':True,
     'comp_ids':[1],#[1,2] for sampling second component
     'freq_guess':{
         1:{ 'c0':3.72, 'c1':0., 'c2':0.},
         2:{ 'c0':2.05, 'c1':0., 'c2':0.},
            },
     'nsamp':{1:8,2:8}, 
     'sampbands_shifts':{1:np.arange(16)*0.15,2:np.arange(16)*0.3},
     'direct_samp' : {2:0,3:0},
     'nit':40,
     'src_perio' : {
         1:{'mod':'slide','range':[0.5,0.5] ,'up_bd':False  },
         2:{'mod':'slide','range':[0.5,0.5] ,'up_bd':True  },
            }
     'fap_thresh':5,
     'fix_freq2expmod':[],
     'fix_freq2vismod':{},
     'plot':True
     }

.. Note::
 Description of parameters and variables:

    + :green:`comp_ids` which component to analyse, start with the first component (the high frequency component), when the first component is analysed add the second component to the list. Once the first component is processed the piecewise model built from the windows is used to temporarily correct the transmission spectrum, and the second component will be sampled and analysed. See :numref:`samp_1` and :numref:`samp_2`, for the example of TOI-421 b.
    + :green:`freq_guess` is the polynomial coefficient describing the model frequency for each component. The models control the definition of the sampling bands.
    + :green:`nsamp` number of cycles to sample for each compojent in a given band, this is based on the guess frequency.
    + :green:`nsampbands_shifts` set the shifts for the window between samples.
    + :green:`direct_samp` (check this one with vincent)
    + :green:`nit` number of iterations in each band
    + :green:`src_perio` frequency ranges within which periodograms are searched for each component (in :math:`1e-10 s^{-1}`). Use :green:`{'mod':None}` for default search range. To define the search range use :green:`{'mod':'slide', 'range':[y,z]}`. Use :green:`'up_bd':True` to use the the higher component as the upper bound of the search window.
    + :green:`fap_thresh` wiggle in a band is fitted if the FAP is below this threshold (in %).
    + :green:`fix_freq2expmod` [compi_id] fixes the frequency of 'comp_id' using the fit results from 'wig_exp_point_ana'.
    + :green:`fix_freq2vismod` fixes the frequency of 'comps' using the fit results from :green:`'wig_vis_fit'` at the given path for each visit, format is :green:`{comps:[x,y] , vis1:path1, vis2:path2 }`.
    + :green:`plot` plot the sampled transmission spectra and band sample analyses.

.. figure:: wiggle_sampling_1.png
  :width: 800
  :name: samp_1

  Sampling of the first component of TOI-421 b.

.. figure:: wiggle_sampling_2.png
  :width: 800
  :name: samp_2

  Sampling of the second component of TOI-421 b, here the piecewise model built from the sampling of the first component has been corrected for.

Plots are found in :orange:`/Working_dir/Star/Planet/Spec_raw/Wiggles/Exp_fit/Instrument_Visit/Sampling/`.

Step 2: Chromatic analysis
-------------------------------------
In this step, we analyze the frequency and amplitude of the wiggles for the sampled exposures from the previous step. We model them as polynomials of :math:`\nu`.

In most cases, both the frequency and amplitude can be described as either linear or quadratic functions of :math:`\nu`. This step allows us to determine the polynomial degree and initial guess values that are suitable for the chromatic coefficients :math:`a_{\text{chrom},k,i}(t)` and :math:`f_{\text{chrom},k,i}(t)` in each sampled exposure.

For visit 20221117, we used the following settings to determine the chromatic coefficients for frequency and amplitude for the two wiggle components:
::
 gen_dic['wig_exp_nu_ana']={
     'mode':True,
     'comp_ids':[1,2],
     'thresh':3.,
     'plot':True
     }
 gen_dic['wig_deg_Freq'][1] = 1  # Linear polynomial for component 1
 gen_dic['wig_deg_Freq'][2] = 0  # Constant function for component 2
 gen_dic['wig_deg_Amp'][1] = 2   # Quadratic polynomial for component 1
 gen_dic['wig_deg_Amp'][2] = 2   # Quadratic polynomial for component 2

.. Note::
 Parameter descriptions:

    + :green:`comp_ids` components to be analysed.
    + :green:`thresh` threshold for automatic exclusion of outliers.

    + :green:`wig_deg_Freq` polynomial degree for Frequency component [n].
    + :green:`wig_deg_Amp` polynomial degree for Amplitude component [n].

.. figure:: chrom_ana.png
  :width: 800
  :name: chrom_ana

  Chromatic analysis of the first and second wiggle components for amplitude and frequency. The first component of the amplitude is best described as a second-degree polynomial of frequency (top left panel), while the second component is modeled as a linear function of frequency (bottom left panel). The right panel shows the wiggle frequency as a function of frequency for the first and second components, both modeled using linear relations.

The resulting plots are automatically saved in: :orange:`/Working_dir/Star/Planet/Spec_raw/Wiggles/Exp_fit/Instrument_Visit/Chrom/`.

By analyzing the chromatic analysis plots, you can identify poorly fitted spectral ranges, which appear as sudden jumps in the sampled values over smaller frequency ranges. These jumps occur due to the sliding window sampling method and indicate regions that may need further refinement.

At this stage, it is recommended to:

+ Review the initial screening and fitting steps to ensure an accurate selection of spectral ranges.
+ Proceed with the next step ``Exposure Fit`` to evaluate how well the model performs across different regions, especially in areas where the chromatic analysis showed significant variations.

Step 3: Exposure fit
---------------------

In this step, the spectral wiggle model :math:`W(\nu)` is initialized using the results from the previous step and derived independently for each exposure. This provides more accurate estimates of the chromatic coefficients :math:`a_{\text{chrom},k,i}(t)`, :math:`f_{\text{chrom},k,i}(t)`, and the phase shift :math:`\Phi(t)`.

Some options in this step rely on the model derived in the subsequent step. For the first run, these options should be left empty. It is best to use an iterative approach between this step (Step 3) and the next step (Step 4).
::
 gen_dic['wig_exp_fit']={
    'mode':False, 
    'comp_ids':[1,2], 
    'init_chrom':True,
    'freq_guess':{
        1:{ 'c0':3.8, 'c1':0., 'c2':0.},
        2:{ 'c0':2.0, 'c1':0., 'c2':0.}
        },
    'nit':20, 
    'fit_method':'leastsq', 
    'use':True,
    'fixed_pointpar':{},
    'prior_par':{
        'par_coeff':['low','high']
    },
    'model_par':{ 
        'par_coeff':['y_min','y_max']
        },
    'plot':True,
    }
 
.. Note::
 Parameter descriptions:

    + :green:`init_chrom` Initializes the fit guess values using the results of the `chromatic analysis` from the closest sampled exposure. Running wig_exp_samp on representative exposures that sample wiggle variations is sufficient. Make sure the chromatic analysis is run with the same components as used here.
    + :green:`freq_guess` Defines polynomial coefficients describing the model frequency for each component. This is used to initialize frequency values if `init_chrom` is set to False.
    + :green:`nit` Specifies the number of iterations for the fitting process.
    + :green:`fit_method`Defines the optimization method used for fitting, `'leastsq'` or `'nelder'`.
    + :green:`use` Determines whether to execute the fitting process. Setting it to False allows retrieving previously computed fits without running the fit again.
    + :green:`fixed_pointpar` Allows fixing selected properties to their model values obtained from the next step, the pointing analysis.
    + :green:`prior_par` Bounds the properties using a uniform prior over a specified range. The range is defined as :green:`{ 'low': val, 'high': val }` and applies to all exposures. Use results from the next step )pointing analysis) to determine an appropriate prior range. Note that if :green:`{ 'guess': val }` is specified, it overrides the default or chromatic initialization.
    + :green:`model_par` Initializes a property to its exposure value :math:`v(t)` from the model derived in the next step (pointing analysis) and applies a uniform prior in the range: :math:`[v(t) - y_\text{min} , v(t) + y_\text{max}]`.

By applying the model and analyzing the resulting corrections, you can assess the model’s performance before moving on to the next step. If you still notice prominent spikes in the periodogram after applying the correction, it's a good idea to examine the areas where the model fails. In some datasets, using a higher-degree polynomial in the chromatic analysis may be necessary to achieve a better fit in the higher-frequency range and improve the correction of the wiggle pattern.

.. figure:: Exp_fit.png
  :width: 800
  :name: Exp_fit

  Exposure fit example for the visit on 20221117: The periodogram at the top shows a strong peak, which disappears in the residuals after the correction.

The resulting plots are automatically saved in: :orange:`/Working_dir/Star/Planet/Spec_raw/Wiggles/Exp_fit/Instrument_Visit/Global/`.

Step 4: Pointing Analysis
-------------------------------------

In this step, we evaluate the parameters derived in the previous Exposure Fit step as time series and fit them to the pointing coordinates of the telescope. This allows us to assess the pointing parameters for the final model, including :math:`a_{\text{point},k,i}(t)`, :math:`f_{\text{point},k,i}(t)`, and :math:`\Phi_{\text{point},k,i}`.
::
 gen_dic['wig_exp_point_ana']={
     'mode':False,
     'source':'glob',
     'thresh':3.,
     'fit_range':{},
     'fit_undef':False,
     'stable_pointpar':[],
     'conv_amp_phase':False,
     'plot':True
     }
 
.. Note::
 Parameter descriptions:

    + :green:`source` Specifies the origin of the fitting coefficients. `'samp'`: Uses coefficients derived from sampled fits. `'glob'`: Uses coefficients from a global spectral fit.
    + :green:`thresh` Defines the threshold for automatic outlier exclusion. Set to None to disable automatic outlier exclusion.
    + :green:`fit_range` Allows defining custom fit ranges for each parameter in a given visit.
    + :green:`fit_undef` Determines how undefined (or missing) values should be treated during the fit. If True, attempts to fit even when some parameters are undefined. If False, ignores undefined values and excludes them from the fit.
    + :green:`stable_pointpar`  Specifies parameters that should be fitted with a constant value (i.e., not varying across exposures).
    + :green:`conv_amp_phase` Automatically adjusts the amplitude sign and phase value to correct for degeneracies.



.. Tip:: 
   The pointing analysis is based on the parameters derived in the previous step, Exposure Fit. Once you have a model for the pointing analysis, you can further refine the model by using it as the basis for your priors in the exposure fit. Model priors are calculated from the model :math:`v(t)` as :math:`[v(t) - \text{model_par}[0], v(t) + \text{model_par}[1]]`. By analyzing the model (see :numref:pointing_analysis) for each component, you can adjust the constraints for the priors until you achieve a good model and fit.

   Here is how priors and the model priors were set up for the analysis of night 20221117:   
   ::
     gen_dic['wig_exp_fit']['prior_par']['20221117']={
            'AmpGlob2_c0':{'low':-4e-3,'high':4e-3},            
            'AmpGlob2_c2':{'low':-1e-7,'high':1e-7},
               'Freq1_c0':{'low':3.84,'high':3.8575},      
               'Freq2_c0':{'low':1.7,'high':2.2},  
                   'Phi2':{'low':-30.,'high':30.},     
        }  
    gen_dic['wig_exp_fit']['model_par']['20221117']={ 
            'AmpGlob1_c0':[0.3e-4,0.3e-4],
            'AmpGlob1_c1':[0.3e-5,0.3e-5],
               'Freq1_c0':[0.0004,0.0004],
               'Freq1_c1':[0.00025,0.00025],
               'Freq2_c0':[0.02,0.02],
                   'Phi1':[0.1,0.1],
                   'Phi2':[0.5,0.5],
        }

.. figure:: pointing_analysis.png
  :width: 800
  :name: pointing_analysis

   Pointing analysis displaying the time evolution of the main component of the three functions :math:`a_{\text{point},k,i}(t)`, :math:`f_{\text{point},k,i}(t)`, and :math:`\Phi_{\text{point},k,i}`. The gray vertical dashed line indicates the pointing passing the meridian and the vertical black dashed line represents a change of guide star. 

The resulting plots are automatically saved in: :orange:`/Working_dir/Star/Planet_Plots/Spec_raw/Wiggles/Exp_fit/Instrument_Visit/Coord/`.

+ The model derived in this step can be used to create model priors with a uniform distribution around the model value for each exposure when recomputing the exposure fit.
+ The vertical lines in the pointing analysis represent critical time steps in the pointing and wiggle model.
    + When passing the meridian, the derivatives of the model parameters reset completely.
    + The change of guide star results in a complete reset of model parameters unless the option :green:`gen_dic['wig_no_guidchange']` is used to ignore this reset.


Step 5: Global / visit fit
-------------------------------------

The full spectro-temporal wiggle model :math:`W(\nu, t)` is initialized using the results from the previous step and fitted to all exposures simultaneously. Due to the large number of free parameters and the complexity of a model composed of cumulative sine functions, it is crucial to provide guess values close to the best-fit solution to ensure proper convergence.
::
 gen_dic['wig_vis_fit']={
     'mode':False ,
     'fit_method':'leastsq',   
     'wig_fit_ratio': False,
     'wig_conv_rel_thresh':1e-5,
     'nit':15,
     'comp_ids':[1,2],
     'fixed':False, 
     'reuse':{},
     'fixed_pointpar':[],      
     'fixed_par':[],
     'fixed_amp' : [],
     'fixed_freq' : [],
     'stable_pointpar':[],
     'n_save_it':1,
     'plot_mod':True    ,
     'plot_par_chrom':True  ,
     'plot_chrompar_point':True  ,
     'plot_pointpar_conv':True    ,
     'plot_hist':True,
     'plot_rms':True ,
     }

 # Fields below only relevant for 'fit_method' = 'ns'
 gen_dic['wig_vis_fit']['ns'] = {        
     'nthreads': int(0.8*cpu_count()),
     'run_mode': 'use',
     'nlive': 2500,
     'reboot':''
 }


.. Note::
  Parameter descriptions:

    + :green:`fit_method` specifies the optimization method to use for fitting:
        + 'leastsq': Fast and sufficient if the initialization is correct.
        + 'nelder': More robust but slower, useful when convergence is difficult.
        + 'ns': Uses nested sampling (currently under development).
    + :green:`nit` number of fit iterations to perform. A higher value allows for better convergence but increases computation time.
    + :green:`comp_ids` nefines which model components to include in the fit. This allows selecting specific terms of the model to optimize.
    + :green:`fixed` determines whether the model parameters are kept fixed during fitting:
        + True: The model remains fixed to the initialization or previous fit results.
        + False: The parameters are free to vary during fitting.
    + :green:`reuse` specifies whether to reuse a previous fit file:
        + {}: No reuse.
        + Path to a fit file: The file is retrieved for post-processing (fixed=True) or used as an initial guess (fixed=False).
    + :green:`fixed_pointpar` list of pointing parameters that should remain fixed during the global fit.
    + :green:`fixed_par` list of general model parameters that should remain fixed during the fit.
    + :green:`fixed_amp` specifies a list of components whose amplitudes should remain fixed during fitting.
    + :green:`fixed_freq` specifies a list of components whose frequencies should remain fixed during fitting.
    + :green:`stable_pointpar` list of pointing parameters that should be fitted with a constant value instead of varying across exposures.
    + :green:`n_save_it` frequency at which fit results are saved (every n_save_it iterations). Helps track progress and recover data in case of interruptions.
    + :green:`plot_hist` generates a cumulative periodogram over all exposures to visualize residual periodic structures before and after correction.
    + :green:`plot_rms` plots the RMS (root mean square) of pre/post-corrected data over the entire visit to assess the effectiveness of the correction.


Step 6: Applying the correction
-------------------------------------
In this step, the spectro-temporal wiggle model correction is applied to the data. This correction uses the model derived from the ``global fit`` and applies it across the relevant spectral range(s) and exposures.

This step must be applied regardless of whether you are using the filter method or the analytical method for the final correction.
::
 gen_dic['wig_corr'] = {
     'mode':False,
     'path':{},
     'exp_list':{},
     'comp_ids':[1,2],
     'range':{},
     }

.. Note::
  Parameter descriptions:

    + :green:`mode` enables or disables the correction step. Set to True to apply the correction.
    + :green:`path` specifies the path to the correction file for each visit. If left empty ({}), the most recent result from :green:`'wig_vis_fit'` is used. Result files from ``Global fit`` are stored in: :orange:`/working_dir/Star/Planet/Corr_data/Wiggles/Vis_fit/Instrument_Visit/`. Format used is:
        ::
         'path':{'visit':'file_path'}

    + :green:`exp_list` defines which exposures to correct for each visit. If left empty ({}), the correction is applied to all exposures.
    + :green:`comp_ids` list of components to include in the correction. These components must be present in the global fit model (wig_vis_fit).
    + :green:`range` specifies the spectral range(s) (in Å) over which the correction should be applied. If left empty ({}), the correction is applied to the full spectrum.
    + :green:`plot_dic['trans_sp']` this plotting dictionary is used to assess the correction visually, ensuring that the wiggle patterns have been properly removed. In the ``plot_settings`` file under ``'trans_sp'`` choose :green:`['plot_pre']='cosm'` and  :green:`['plot_post']='wig'`, to plot the transmission spectra before and after wiggle correction.














