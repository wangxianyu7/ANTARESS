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


Describe 
    local_dic[data_type]['deriv_prop']={}




MCMC settings
-------------

Since a MCMC can take several hours to run, the 

#%%%%%% Calculating/retrieving
data_dic['Intr']['mcmc_run_mode']='use'

#    - set to
# + 'use': runs MCMC  
# + 'reuse' (with gen_dic['calc_fit_X']=True): load MCMC results, allow changing nburn and error definitions without running the mcmc again



#%%% Monitor MCMC
local_dic[data_type]['progress']= True



#%%%% Runs to re-use
#    - list of mcmc runs to reuse
#    - if 'reuse' is requested, leave empty to automatically retrieve the mcmc run available in the default directory
#  or set the list of mcmc runs to retrieve (they must have been run with the same settings, but the burnin can be specified for each run)
local_dic[data_type]['mcmc_reuse']={}


#%%%%%% Runs to re-start
#    - indicate path to a 'raw_chains' file
#      the mcmc will restart the same walkers from their last step, and run from the number of steps indicated in 'mcmc_set'
local_dic[data_type]['mcmc_reboot']=''

#%%%%%% Walkers
    data_dic['Intr']['mcmc_set']={'nwalkers':{'ESPRESSO':{'20221117':30,'20231106':30}},
                                  'nsteps':{'ESPRESSO':{'20221117':1000,'20231106':1000}},
                                  'nburn':{'ESPRESSO':{'20221117':200,'20231106':200}}} 
                                  
                                  

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


#%%%%% MCMC corner plot
#    - see function for options
local_dic[data_type]['corner_options']={}


#%%%%% MCMC 1D PDF
#    - on properties derived from the fits to individual profiles
if data_type in ['DI','Intr','Atm']:
    plot_dic['prop_'+data_type+'_mcmc_PDFs']=''      


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


