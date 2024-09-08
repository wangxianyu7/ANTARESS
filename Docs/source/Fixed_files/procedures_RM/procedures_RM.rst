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

This tutorial relates to the analysis of a Rossiter McLaughlin signal from a transiting planet, using the *Revolutions* technique (`Bourrier et al. 2021 <https://www.aanda.org/articles/aa/full_html/2021/10/aa41527-21/aa41527-21.html>`_). We illustrate the tutorial with ESPRESSO observations of TOI-421b (2022-11-17) and TOI-421c (2023-11-06). 
Additional options beyond the present example can be found in the `configuration file <LINK TBD>`_.

We assume here that you have already `set up your system <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_setup/procedures_setup.html>`_, processed a `spectral transit time-series <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_reduc/procedures_reduc.html>`_, extracted `planet-occulted stellar profiles <TBD>`_, and 
converted them into `intrinsic CCFs <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_CCF/procedures_CCF_Intr/procedures_CCF_Intr.html>`_.

Analysis of individual intrinsic lines
--------------------------------------

The goal of this step is to fit a model line profile to each intrinsic CCF, deriving series of line properties that will then be used to identify which exposures should be included in the RM Revolutions fit, and what are the best models to be used.
Activate the ``Intrinsic profiles analysis`` module by setting :green:`gen_dic['fit_Intr'] = True`, and set it to calculation model by setting :green:`gen_dic['calc_fit_Intr'] = True`.
 
First, define the velocity range that you assume to represent the line continuum::

 data_dic['Intr']['cont_range'] = {'ESPRESSO':{0:[[-60,-20],[20,60]]}
 
Then, define the velocity range over which the line model is fitted. You can use specific ranges for each visit, in case the line is affected by spurious feature in some of them. Here this is not the case, and we set::

 data_dic['Intr']['fit_range']= {'ESPRESSO':{'20231106':[[-60,60]]}

Both continuum and fitted ranges are defined in the star rest frame, and specific to a given instrument because the measured line width depends on the spectrograph broadening. 
The ranges can be made of several independent intervals, so that you can exclude features that are not captured by your line model. 
If left  undefined, the ranges are set to default values based on the value you provided for the stellar rotational velocity. It is advised to define specific values for your datasets, by looking at the time series of intrinsic profiles (e.g., :numref:`Intrinsic_CCF`). 
They can be plotted by activating :green:`plot_dic['Intr_prof']='pdf'`, and can be found in :orange:`Working_dir/Star/Planet_Plots/Intr_data/Instrument_Visit_Indiv/Data/`.  

.. figure:: Intrinsic_CCF.png
  :width: 800
  :name: Intrinsic_CCF
  
  Intrinsic CCF occulted by TOI-421c. Blue shaded areas indicate the continuum ranges. Grey shaded areas are excluded from the fit.


.. Tip::
   For slow rotators the disk-integrated and intrinsic lines will have similar shapes. You can thus use fitting and continuum ranges based on the disk-integrated line profile, which is particularly useful when the intrinsic line is measured at low S/R and not visible by eye.
   On the other hand, for fast rotators you will want to use narrower ranges for the intrinsic line than for the disk-integrated line.


Next, define the best model for the line profile. Intrinsic stellar lines are typically well described by the default Gaussian model, which would otherwise be set up as:: 

 data_dic['Intr']['model']['ESPRESSO']='gauss' 

.. Tip::
   If the stellar line is not well visible in individual intrinsic profiles, you can determine its shape by analyzing a higher S/N master of all intrinsic profiles along the transit chord.

We advise applying instrumental convolution to the line model by activating :green:`data_dic['Intr']['conv_model']=True`. 
 
In that case the properties that you will derive from the fit will correspond to the model line profile before convolution. This is particularly useful to trace the *intrinsic* line properties, and compare results between different instruments and with theoretical predictions.
Model properties are set up through:: 

 data_dic['Intr']['mod_prop']={
     'rv':{'vary':True,'ESPRESSO':{'20231106':{'guess':0.,'bd':[-2.,2.]}}},
     'ctrst':{'vary':True,'ESPRESSO':{'20231106':{'guess':0.5,'bd':[0.2,0.9]}}},
     'FWHM':{'vary':True,'ESPRESSO':{'20231106':{'guess':8.,'bd':[0.,15.]}}}}  

Since we are using a Gaussian model, its profile is determined by the line centroid (:green:`rv`, tracing the average radial velocity of the photospheric regions occulted by the planet), contrast (:green:`ctrst`), and full width at half maximum (:green:`FWHM`). Different models implemented in ``ANTARESS`` would require additional or different properties. 

Since intrinsic CCFs are often measured with low S/N it is advised to fit them with a MCMC approach (:green:`data_dic['Intr']['fit_mode']='MCMC'`) rather than the default least-square minimization. Since running the MCMCs for all exposures may take some time, we remind that you can set :green:`gen_dic['calc_fit_Intr']=False` once the analysis is done, so that ANTARESS retrieves the results and you are free to manipulate the associated plots.
Generic information for :math:`\chi^2` and MCMC fits with ``ANTARESS`` can be found in the `fit tutorial <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_fits/procedures_fits.html>`_.
If the fit is performed via :math:`\chi^2` minimization, it is initialized at the value of :green:`guess`.
If the fit is performed using a MCMC approach, its walkers are randomly initialized over the range defined by :green:`bd`. 
The default values for the number of MCMC walkers, number of steps, and burn-in phase are usually good enough for these fits, but you should check the chains saved in :orange:`/Working_dir/Star/Planet_Saved_data/Introrig_prop/instrument_night_mcmc/iexp*/` and if need be adjust the settings as described in the `fit tutorial <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_fits/procedures_fits.html>`_.
If you set :green:`vary = False` for a parameter then it will be fixed to the value of :green:`guess`. 

Default priors on the fitted properties can be overwritten as:: 

 data_dic['Intr']['line_fit_priors']={
     'rv':{'mod': 'uf', 'low':-5.,'high':5.},  
     'FWHM':{'mod': 'uf', 'low':0.,'high':20.}, 
     'ctrst':{'mod': 'uf', 'low':0.,'high':1.}}   

Here :green:`mod = uf` indicates that we use uniform priors (see the `generic settings <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_fits/procedures_fits.html>`_ for other possibilities), with lower and upper boundaries defined by :green:`low` and :green:`high`.

Guess values, walker boundaries, and priors on model parameters can be defined in two steps.
First, there may be physical constraints (e.g. typical rotational velocity, line depth and width for a given stellar type) or specific knowledge on your host star from the literature. 
Here the surface RVs were conservatively bounded within :math:`\pm` 5 km/s, at about 3 :math:`\sigma` from the spectroscopic stellar rotational velocity from Carleo+2020 (1.8 :math:`\pm` 1 km/s).
A conservative upper boundary was set on the FWHM at three times the width of the disk-integrated CCF.
The contrast was bounded by its physical range between 0 and 1.

The PDFs for the fitted properties (e.g., :numref:`Intrinsic_PDFs`) can be plotted by activating :green:`plot_dic['prop_Intr_mcmc_PDFs'] = 'pdf'`, and can be found in :orange:`Working_dir/Star/Planet_Plots/Intr_prop/MCMC/`.  

.. figure:: Intrinsic_PDFs.png
  :width: 800
  :name: Intrinsic_PDFs
  
  PDFs for the model RVs of the intrinsic stellar lines occulted by TOI-421c.

The time-series of fitted properties (e.g., :numref:`Intrinsic_props`) can be plotted by activating :green:`plot_dic['prop_Intr'] = 'pdf'`, and can be found in :orange:`Working_dir/Star/Planet_Plots/Intr_prop/`.  

.. figure:: Intrinsic_props.png
  :width: 800
  :name: Intrinsic_props
  
  Time-series of properties for the intrinsic stellar lines occulted by TOI-421c.

Here the intrinsic CCFs are measured with high-enough S/R that the PDFs for the fitted properties are all well-defined, and do not need to be further constrained by more stringent priors. 
Otherwise, a second step would consist in setting up narrower priors based on the derived property series, their PDFs, their MCMC chains and corner plots (:orange:`/Working_dir/Star/Planet_Saved_data/Introrig_prop/instrument_night_mcmc/iexp*/`), or their fit (see next section). 
For example, if the fit converged poorly for one intrinsic CCF during the TOI-421c transit, due to a lower S/R, we could reasonably assume from the time-series in Fig XX that intrinsic line contrast can be bounded within 0.4 and 0.8.
As another example, the fit to the surface RV series of TOI-421c and later-on to the full intrinsic CCF dataset will yield a projected rotational velocity of 1.6 km/s, which could be set as an upper limit on the RV of the stellar lines occulted by TOI-421c.

In the end, however, the goal of the present analysis is not to derive accurate properties to be fitted, but to identify which exposures to include in the global Revolutions fit, and what models to use to describe the line profile.
For TOI-421c we excluded the first and last in-transit exposures from further analysis, because the PDFs of their derived properties are much broader than the rest of the series and will not constrain the global RMR fit (:numref:`Intrinsic_props`).
For TOI-421b we kept all exposures. The first exposure obtained during ingress was manually flagged as not belonging to the in-transit series at the start of the workflow (see :green:`data_dic['DI']['idx_ecl']`) because the planet did not occult the star during most of the exposure and it was clear that the corresponding intrinsic profile would not be useful to the analysis.

.. Tip::
   Intrinsic stellar lines are often measured with lower S/R at the stellar limbs than in the rest of the series, due to limb-darkening and the smaller occulting fraction from the partially transiting planet. Those exposures may not bring much constraints to the fits and should be considered for exclusion.
   
Since the fits converged well for all exposures and the time-series of derived properties show no outliers, we have no reason to exclude further exposures. We note that a spot crossing occured during the transit of TOI-421c, but the planet-occulted stellar lines are not significantly affected (see Bourrier+2025).

We are now goig to use the next module to define the best models to use for the global RMR fit. 
You can deactivate the present module by setting :green:`gen_dic['fit_Intr'] = False`.


Analysis of intrinsic line properties
-------------------------------------

The goal of this step is to determine the best models to describe variations of the intrinsic stellar line profile along the transit chord. To do so you are going to try fitting the times-series of each intrinsic properties with various models, over individual or combined visits. 
Activate the ``Intrinsic stellar properties fit`` module by setting :green:`gen_dic['fit_IntrProp'] = True`.

Since we now fit the property time-series as a whole, we must indicate which exposures should be included::

 glob_fit_dic['IntrProp']['idx_in_fit']={'ESPRESSO':{
     '20221117':'all',
     '20231106':np.delete(np.arange(29),[0,27])}}

In the previous module we decided to keep in the analysis all exposures of the 2022-11-17 visit, which can be done by setting its field to :green:`'all'`, and to remove the first and last in-transit exposures in the 2023-11-06 visit.
Indexes are relative to the in-transit series of exposures (i.e., 0 corresponds to the first exposure during which a planet starts occulting the star).

.. Tip::
   Each exposure is automatically defined as in- or out-of-transit by ``ANTARESS``, unless you force its definition with :green:`data_dic['DI']['idx_ecl']`.
   A quick way to check the status of an exposure and get its global and in-transit indexes is to plot a visit light curve by activating :green:`plot_dic['input_LC']='pdf'`. 

Intrinsic line centroids are described by a specific model for the stellar surface RV. At minimum, when the data is only sensitive to the solid-body rotation of the star, the model depends on the sky-projected angle `lambda` (in degrees) between the stellar spin and orbital normal, and on the sky-projected stellar rotational velocity `veq` (in km/s). 
Model properties are set up in the same way as in the previous section::

 glob_fit_dic['IntrProp']['mod_prop']['rv']={
     'veq':{'vary':True,'guess':2.,'bd':[0.,5.]},
     'lambda_rad__plTOI421b':{'vary':True,'guess':0.,'bd':[-np.pi,np.pi]},
     'lambda_rad__plTOI421c':{'vary':True,'guess':0.,'bd':[-np.pi,np.pi]},     
     }

Unless the stellar inclination is known, you should have set it to 90$^{\circ}$ in the `system property file <link>`_. Under the assumption of solid-body rotation it is degenerate and 'veq' stands for 'vsini'.
If your data is sensitive to stellar differential rotation you can break this degeneracy and fit for the stellar inclination (through its cosine :green:`cos_istar`) and the coefficient :green:`alpha_rot` (0 corresponds to solid rotation, 1 to the poles not rotating)::

 glob_fit_dic['IntrProp']['mod_prop']['rv'].update({
     'alpha_rot':{'vary':True,'guess':0.1,'bd':[0. , 0.5]},                          
     'cos_istar':{'vary':True,'guess':np.cos(90.*np.pi/180.),'bd':[-1.,1.]}})  

The surface RV model can further be modulated by convective blueshift, defined as a polynomial of the center-to-limb angle for which you can control the linear or quadratic coefficients :green:`c1_CB` and :green:`c2_CB`::

 glob_fit_dic['IntrProp']['mod_prop']['rv'].update({
     'c1_CB':{'vary':True,'guess':0.1,'bd':[-0.5,0.5]},  
     'c2_CB':{'vary':True,'guess':0.1,'bd':[-1.,1.]}})
                
The analysis of TOI-421c data did not support the detectability of differential rotation or convective blueshift, and so we will not include them hereafter. 
Here the exposure and surface RV model were set up to fit together the TOI-421b and TOI-421c data. However we advise to first fit different visits independently to assess the consistency of their results.

Morphological line properties (e.g., FWHM and contrast if a Gaussian model was used to derive the intrinsic line properties) are described by polynomial models as a function of a given stellar surface coordinate, the default being the sky-projected distance from star center::

 glob_fit_dic['IntrProp']['coord_fit']={'ctrst':'r_proj','FWHM':'r_proj'}

Other possibilities are available in the `configuration file <LINK TBD>`_. The polynomial models can further be defined in an absolute (:math:`m(x) = \sum_{i\geq0}c_i x^i)`) or modulated (:math:`m(x) = m_0 (1 + \sum_{i\geq1}c_i x^i)`) way, set through::

 glob_fit_dic['IntrProp']['pol_mode']='abs' or 'modul' 

The latter possibility allows for a common dependence of the property with stellar coordinate `x`, with a scaling :math:`m_0` specific to each epoch. A modulated linear contrast variation would be set up as:: 

 glob_fit_dic['IntrProp']['mod_prop']['ctrst'] = {
     'ctrst__ord0__IS__VS20221117':{'vary':True ,'guess':0.5,'bd':[0.3,1.]},   
     'ctrst__ord0__IS__VS20231106':{'vary':True ,'guess':0.5,'bd':[0.3,1.]},   
     'ctrst__ord1__IS__VS_':{'vary':True ,'guess':0.0,'bd':[-0.1,0.1]}} 

.. Tip::
   Convention for the name of a model parameter is :green:`prop__ordi__ISinst_VSvis`, with
   
    + :green:`prop` the name of the property
    + :green:`i` the degree of the polynomial coefficient
    + :green:`inst` the name of the instrument, which can be set to :green:`_` if the parameter is common to all instruments (and thus all their visits)
    + :green:`vis` the name of the visit, which can be set to :green:`_` if the parameter is common to all visits of instrument :green:`inst`          

Here we are working with a single instrument, thus there is no need to define it in the parameter names.
As an example above, we use the :green:`ord0` coefficient to describe a modulation specific to each visit, and we set a linear coefficient :green:`ord1` common to both visits.
However the actual data on TOI-421b does not have sufficient precision to detect variation of the line shape along the transit chord, and the analysis of the TOI-421c data alone favors a constant contrast and FWHM along the transit chord. 
It is thus not relevant to use modulated models, and the contrast and FWHM will be described hereafter with a constant coefficient :green:`ord0` alone. 
We will allow this coefficient to be different between the two visits, as we expect the stellar line to vary in shape over time.

You are now ready to set up the fit on the property time-series, choosing the fit mode with :green:`data_dic['Intr']['fit_mode']`. You can start with a simple :math:`\chi^2` fit to narrow down the parameter space, but we recommend using a MCMC approach to properly compare different best-fit models for the line properties.
As in the previous stepyou can adjust the number of MCMC walkers, steps, and burn-in phase to your dataset as described in the `fit tutorial <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_fits/procedures_fits.html>`_, looking at the MCMC chains
in the :orange:`/Working_dir/Star/Planet_Saved_data/Joined_fits/IntrProp/mcmc/prop/` directory.

Uniform priors on the fitted properties are set with:: 
   
 glob_fit_dic['IntrProp']['priors'].update({
     'veq':{'mod':'uf','low':0.,'high':5.},  
     'lambda_rad__plTOI421b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
     'lambda_rad__plTOI421c':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
     'ctrst__ord0__IS__VS20221117':{'mod':'uf','low':0.,'high':5.},  
     'ctrst__ord0__IS__VS20231106':{'mod':'uf','low':0.,'high':5.}})  

.. Tip::
   We set the prior range on lambda to avoid the walkers bumping into the prior boundaries, in case the best-fit is close to +-180 deg. Lambda values can be folded during post-processing, using the field :green:`'deriv_prop'` as described in the `fit tutorial <https://obswww.unige.ch/~bourriev/antaress/doc/html/Fixed_files/procedures_fits/procedures_fits.html>`_, 

You can now run the fit. It will be fast in :math:`\chi^2` mode but may take some time with a MCMC. To gain time, once the MCMC fit is done do not forget that you can set :green:`data_dic['Intr']['mcmc_run_mode']='reuse'`to retrieve and manipulate the fit results. 

.. Tip::
   If the star is too faint or the planet too small, intrinsic properties may all be derived with a precision that is too low to analyze them in this step. 
   In that case, you can apply directly the joint RM Revolutions fit (see next step) with the simplest models to describe these properties. 





IMAGES








METTRE EN NOTE POUR DIRE POURQUOI IMPORTANT D'UTILISER ANTARESS ET PAS UN SIMPLE FIT AU CENTRE DE l'expo
Finally, in this notebook we are using a simple Gaussian profile to fit the intrinsic lines. Although this model includes instrumental convolution, it is an approximation compared to using ANTARESS numerical stellar grid to simulate realistic line profiles, as is done when applying the RM Revolutions fit. Here we do not account in particular for the blurring induced by the planet motion, which is significant for long exposure times and fast-rotators. Use the `configuration file <LINK TBD>`_ if you want to fit more finely individual intrinsic profiles. 

Guess values for the global fit can be informed by the results of this step, printed in the log below.


