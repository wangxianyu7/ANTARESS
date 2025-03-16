.. raw:: html

    <style> .orange {color:DarkOrange} </style>

.. role:: orange

.. raw:: html

    <style> .green {color:green} </style>

.. role:: green

Fitting methods
===============

``ANTARESS`` is using the same fitting routines in several modules. We describe here the general settings defined in the function ANTARESS_analysis_settings() of the `configuration file <LINK TBD>`_, and associated with an arbitrary fit dictionary :green:`fit_dic` 

You can choose to run the fit via :math:`\chi^2` minimization (:green:`fit_dic['fit_mode']='chi2'`) or a MCMC approach (:green:`fit_dic['fit_mode']='mcmc'`). A third option (:green:`fit_dic['fit_mode']='fixed'`) allows you to calculate the model in forward mode with fixed parameter values.



#%%%%% Printing fits results
Explain use

data_dic['Intr']['verbose'] = False  
data_dic['Intr']['print_par'] = False  


#%%% Priors on variable properties
#    - structure is priors = { 'par_name' : {prior_mode: X, prior_val: Y} }
#      where par_name is specific to the model selected, and prior_mode is one of the possibilities defined below
#    - otherwise priors can be set to :
# > uniform ('uf') : 'low', 'high'
# > gaussian ('gauss') : 'val', 's_val'
# > asymetrical gaussian ('dgauss') : 'val', 's_val_low', 's_val_high'
#    - chi2 fit can only use uniform priors
#    - if left undefined, default uniform priors are used
local_dic[data_type]['priors']={}    

Describe various priors and way to set them up    

Uniform Prior
^^^^^^^^^^^^^

:green:``mod:'uf'``. Set with lower range (``low``) and higher range (``high``). The parameter is 
uniformly distributed between ``low`` and ``high``, meaning every values within this range has an equal 
probability. Set as::

 compos_dic['priors'] = { prop : {
    'mod': 'uf', 
    'low': val, 
    'high': val} 
    }

Gaussian Prior
^^^^^^^^^^^^^^

:green:``mod:'gauss'``. Set with mean of the Gaussian distribution (``val``), which is the most likely
values for the parameter; and standard devitation of Gaussian distribution (``s_val``), indicating
how spread out the values are around the mean. Smaller ``s_val`` lead to a narrower distribution,
implying higher confidence in the mean values. Set as::

 compo_dic['priors'] = { prop : {
    'mod': 'gauss' ,
    'val': val ,
    's_val': val} 
    } 

Double-Gaussian Prior
^^^^^^^^^^^^^^^^^^^^^

:green:``mod: 'dgauss'``. Set with ``val``, and standard deviations for values below the mean 
(``s_val_low``), controlling the spread of the lower side of the distribution, and the standard
deviation above the mean (``s_val_high``), controlling the spread of the upper side.
Double-Gaussian approach allows an asymmetric distribution, where uncertainties might differ
between the lower and upper bounds. Set as::

 compos_dic['priors'] = { prop: {
    'mod': 'dgauss', 
    'val': val, 
    's_val_low': val, 
    's_val_high': val}
    }



Describe 
    local_dic[data_type]['deriv_prop']={}




MCMC settings
-------------

Since a MCMC can take several hours to run, the 

#%%%%%% Calculating/retrieving
data_dic['Intr']['run_mode']='use'

#    - set to
# + 'use': runs MCMC  
# + 'reuse' (with gen_dic['calc_fit_X']=True): load MCMC results, allow changing nburn and error definitions without running the mcmc again

Run Mode
~~~~~~~~

Define the mode for running the MCMC simulations:

- ``'use'``: Runs (new) MCMC from scratch.
- ``'reuse'``: Loads previously saved MCMC results, which is saved as a :orange:`*.npz` file in :orange:`{path_to_directory}/working/output/GJ436` directory. Reusing MCMCM allows changes to the burn-in phase (``nburn``) and error definitions without re-running the MCMC from scratch.

Example configuration::

  compo_dic['mcmc_run_mode'] = 'use' or 'reuse'
  
  
  


#%%% Monitor MCMC
local_dic[data_type]['progress']= True



#%%%% Runs to re-use
#    - list of mcmc runs to reuse
#    - if 'reuse' is requested, leave empty to automatically retrieve the mcmc run available in the default directory
#  or set the list of mcmc runs to retrieve (they must have been run with the same settings, but the burnin can be specified for each run)
local_dic[data_type]['reuse']={}

Runs to Re-use
~~~~~~~~~~~~~~

When ``mcmc_run_mode`` is set to ``'reuse'``, either leave the entry empty to automatically retrieve the MCMC result available in the default directory or specify the list of MCMC results (:orange:`*.npz`) to retrieve.
The retrived MCMC result must have been run with the same Fit Setting and the same number of walkers (``'nwalkers'``), but the number of step (``'nsteps'``) and burn-in phrase (``'nburn'``) can be specified for each run as::

  compo_dic['mcmc_reuse'] = 
    {'paths':['{path_to_directory}/working/output/GJ436/raw_chains_walk40_steps500.npz',
              '{path_to_directory}/working/output/GJ436/raw_chains_walk40_steps100.npz'],
              'nburn':[200,0]}

Where the :orange:`raw_chains_walk40_steps500.npz` and :orange:`raw_chains_walk40_steps100.npz` are
the two MCMC results which to be re-use. `'nburn'` specifies the burn-in phase for each MCMC
results.




#%%%%%% Runs to re-start
#    - indicate path to a 'raw_chains' file
#      the mcmc will restart the same walkers from their last step, and run from the number of steps indicated in 'walkers_set'
local_dic[data_type]['reboot']=''

Run to Re-start
~~~~~~~~~~~~~~~

When ``mcmc_run_mode`` is set to `'use'`, either set an empty string to start a new MCMC run, or specify the path to an ``*.npz`` file with the same number of walkers (``'nwalkers'``) and extend the MCMC run for additional number of steps (``'nsteps'``) indicated in the ``'mcmc_set'`` from the last step.

Example configuration::

  compo_dic['mcmc_reboot'] = '{path_to_directory}/working/output/GJ436/raw_chains_walk40_steps200.npz'






ANTARESS allows you to reboot an existing MCMC run so that it is advised to run

#%%%%%% Walkers
    data_dic['Intr']['walkers_set']={'nwalkers':{'ESPRESSO':{'20221117':30,'20231106':30}},
                                  'nsteps':{'ESPRESSO':{'20221117':1000,'20231106':1000}},
                                  'nburn':{'ESPRESSO':{'20221117':200,'20231106':200}}} 
                                  

Tip: when defining ranges for walkers initialization, it is advised to define broad range for the first runs to ensure a good exploration of the parameter space
for final and refined runs, ranges can be set to narrower windows aroudn the expected best fit so that it converges faster
                  
MCMC Walkers
~~~~~~~~~~~~

Define the MCMC walkers settings:

- ``'nwalkers'``: Number of walkers.
- ``'nsteps'``: Number of samples.
- ``'nburn'``: Burn-in phase.

(Noted by Emi:) I am interested to run some more trail runs. Reasons: if possible I would like to share with the user the computation time and fititng accuracy versus 'nwalkers' and 'nsteps'. 
Which one 'nwalkers' or 'nsteps' is more a limiting factor toward the computation time? 
Perhaps a plotting and presenting a figures of 'nwalkers' and 'nsteps' v.s., computaional time. 
And also remind the 'nwalkers' cannot be set to be too low, otherwise the MCMC will not converge. 
I need to check the minimum number.
Vincent, if you have some MCMC result, e.g., the 'output' file for different run (if you did not
overwrite it), could you please share it with me? I can check the computation time ('Duration ') and fitting accuracy ('Best Chi-sq' and 'Best reduced Chi-sq') versus 'nwalkers' and 'nsteps'.

Example configuration::

    compo_dic['mcmc_set'] = {
        'nwalkers': 20,
        'nsteps': 1000,
        'nburn': 200
    }                  
                  
                  
                  
                  
                                  

#%%%%%% Complex priors
#    - to be defined manually within the code
#    - leave empty, or put in field for each priors and corresponding options
local_dic[data_type]['prior_func']={}      


#%%%% Manual walkers exclusion        
#    - excluding manually some of the walkers
#    - define conditions within routine
local_dic[data_type]['exclu_walk']=  False           


#%%%%%% Automatic walkers exclusion        
#    - set to None, or exclusion threshold
local_dic[data_type]['exclu_walk_autom']= None  


#%%%% Sample exclusion 
#    - keep samples within the requested ranges of the chosen parameter (on original fit parameters)
#    - format: 'par' : [[x1,x2],[x3,x4],...] 
local_dic[data_type]['exclu_samp']={}
    

#%%%% Derived errors
#    - 'quant' (quantiles) or 'HDI' (highest density intervals)
#    - if 'HDI' is selected:
# + by default a smoothed density profile is used to define HDI intervals
# + multiple HDI intervals can be avoided by defined the density profile as a histogram (by setting its resolution 'HDI_dbins') or by defining the bandwith factor of the smoothed profile ('HDI_bw')
local_dic[data_type]['out_err_mode']='HDI'
local_dic[data_type]['HDI']='1s'   


#%%%% Derived lower/upper limits
#    - format: {par:{'bound':val,'type':str,'level':[...]}}
# where 'bound' sets the limit, 'type' is 'upper' or 'lower', 'level' is a list of thresholds ('1s', '2s', '3s')
local_dic[data_type]['conf_limits']={}   


##################################################################################################         
#%%% Plot settings
################################################################################################## 

#%%%%% MCMC chains
local_dic[data_type]['save_MCMC_chains']='png'        



Samples explored with the MCMC can be used to compute 1D PDF and 2D correlation diagrams of the variable fit parameters in a common plot, activated by setting :green:`local_dic[data_type]['save_MCMC_corner']='pdf'`.
Many options are available for this plot through the :green:`local_dic[data_type]['corner_options']` dictionary, such as::

 bins_1D_par : the number of bins in the range covered by the 1D PDFs, common to all parameters if set to a single integer value, or defined as a dictionary with keys the parameter names and values their specific bin number.
 
        bins_2D_par=20 if 'bins_2D_par' not in corner_options else corner_options['bins_2D_par']
        range_par=None if 'range_par' not in corner_options else corner_options['range_par']
        major_int=None if 'major_int' not in corner_options else corner_options['major_int']
        minor_int=None if 'minor_int' not in corner_options else corner_options['minor_int']
        color_levels='black'  if 'color_levels' not in corner_options else corner_options['color_levels']
        smooth2D=None if 'smooth2D' not in corner_options else corner_options['smooth2D']
        plot_HDI=False if 'plot_HDI' not in corner_options else corner_options['plot_HDI']        
        plot1s_1D=True if 'plot1s_1D' not in corner_options else corner_options['plot1s_1D']  
        best_val = fit_dic['med_parfinal'] if (('plot_best' not in corner_options) or corner_options['plot_best']) else None
        if 'fontsize' in corner_options:
             label_kwargs={'fontsize':corner_options['fontsize']}
             tick_kwargs={'labelsize':corner_options['fontsize']}
        else:
             label_kwargs=None
             tick_kwargs=None   

Activating :green:`local_dic[data_type]['save_sim_points_corner']='pdf'` will generate a plot in the same format as the PDF plot but displaying the density of simulations as a function of variable parameters. 
This plot is useful when running a manual grid of simulations for a model with long computing time, with the PDF plot generated from samples drawn from importance sampling. 
For example it allows checking that a region of high probability was sufficiently sampled by the simulations.
Plot options are the same as for the PDF plot.


MCMC Corner Plot
~~~~~~~~~~~~~~~~

Define the options for generating the MCMC corner plot at the end of internal structure retrieval runs:

- ``'plot_HDI'``: Whether to plot the Highest Density Interval (HDI).
- ``'use_arviz'``: Whether to use ArviZ for plotting.
- ``'plot1s_1D'``: Whether to plot 1-sigma intervals in 1D plots.

Example configuration:

.. code-block:: python

    compo_dic['corner_options'] = {
        'plot_HDI': True,
        'use_arviz': True,
        'plot1s_1D': False
    }








#%%%%% MCMC 1D PDF
#    - on properties derived from the fits to individual profiles
if data_type in ['DI','Intr','Atm']:
    plot_dic['prop_'+data_type+'_PDFs']=''      


#%%%%% Chi2 values
#    - plot chi2 values for each datapoint
if 'Prop' in data_type:
    plot_dic['chi2_fit_'+data_type]=''                                    
    
    
data_dic['Intr']['HDI_dbins'] ?




Fit directory
----------------


:orange:`/Working_dir/Star/Planet_Saved_data/Joined_fits/IntrProp/fit_mode/prop/

describe contents: chains, corr diag, npz, merged, outputs, raw


Model comparison
----------------

All ``ANTARESS`` fit output files store the Bayesian Information Criterion (BIC) of the fits. You can ...





GO THROUGH ALL CUSTOM SETTINGS AND CHECK FOR TIPS


