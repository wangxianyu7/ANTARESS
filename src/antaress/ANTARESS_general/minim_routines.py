#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import emcee
exec('log10=np.log10')  #necessary to use log10 in expressions linking parameters
import logging
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None
from matplotlib.ticker import MultipleLocator
import os as os_system
from pathos.multiprocessing import Pool
import scipy.linalg
from scipy import stats
from copy import deepcopy
from lmfit import minimize, report_fit
from scipy import special
from ..ANTARESS_plots.utils_plots import custom_axis,autom_tick_prop
from ..ANTARESS_general.utils import np_where1D,stop,npint,init_parallel_func,get_time
    
##################################################################################################
#%%% Probability distributions
##################################################################################################   

def ln_prior_func(p_step,fixed_args): 
    r"""**Log(prior)**

    Calculates the sum of natural logarithms from selected prior probability distributions.

    Args:
        TBD
    
    Returns:
        TBD
    
    """
    ln_p = 0.

    #Priors on variable parameters
    for parname in fixed_args['var_par_list']:
        if parname in fixed_args['varpar_priors']:
    
            #Parameter value
            parval=p_step[parname]
    
            #Parameter prior properties
            parprior=fixed_args['varpar_priors'][parname]
    
            #Uniform prior
            #    pr(x) = 1./(b-a) if a <= x <= b else 0
            #    ln(pr(x) = -ln(b-a) or -inf
            if parprior['mod']=='uf':
                if (parprior['low'] <= parval <= parprior['high']):	
                    ln_p += - np.log(parprior['high']-parprior['low'])
                else: 
                    ln_p += - np.inf  
                    break
        
            #Gaussian prior
            #    pr(x) = exp(- chi2_x / 2.)
            #        with chi2_x = ( (x - x_constraint)/s_constraint  )^2
            #    pr(x) = exp(- ( (x - x_constraint)/sqrt(2)*s_constraint  )^2 )            
            #    which can be normalized as 
            #    pr_norm(x) = (1./(sqrt(2pi)*s_constraint )) * exp( - (( y_step - y_val )/(sqrt(2)*s_constraint)  )^2  )
            #    ln(pr_norm(x)) =  - 0.5* ( ln( 2pi*s_constraint^2)  + chi2_x)     
            elif parprior['mod']=='gauss':
                ln_p += - 0.5*(np.log(2.*np.pi*parprior['s_val']**2.) + ( (parval - parprior['val'])/parprior['s_val']  )**2.)        
    
            #Gaussian prior with different halves
            elif parprior['mod']=='dgauss':
                if (parval <= parprior['val']):ln_p += - 0.5*(np.log(2.*np.pi*parprior['low']**2.) + ( (parval - parprior['val'])/parprior['low']  )**2.)                       
                else:ln_p += - 0.5*(np.log(2.*np.pi*parprior['high']**2.) + ( (parval - parprior['val'])/parprior['high']  )**2.)   
    
            #Undefined prior
            else:
                stop('Undefined prior')
    
    #Additional priors using multiple parameters and complex functions
    if ('global_ln_prior_func' in fixed_args) and (~ np.isinf(ln_p)): 
        ln_p += fixed_args['global_ln_prior_func'](p_step,fixed_args)

    return ln_p



def ln_lkhood_func_mcmc(p_step,fixed_args):
    r"""**Log(likelihood)**

    Calculates the natural logarithm of the likelihood.
    Likelihood is defined as
    
    .. math::      
       L[\mathrm{step}](x) = \Pi_{x}{ \frac{1}{\sqrt{2 \pi} \sigma_\mathrm{val}^\mathrm{jitt}(x) } \exp^{ - (\frac{ y_\mathrm{mod}[\mathrm{step}](x) - y_\mathrm{val}(x) }{\sqrt{2} \sigma_\mathrm{val}^\mathrm{jitt}(x)  })^2  } }
       
    With 
    
        + :math:`y_\mathrm{mod}[\mathrm{step}]` : model for parameter values at current step  
        + :math:`y_\mathrm{val}` : fitted measurements  
        + :math:`\sigma_\mathrm{val}` : error on fitted measurements  
        + :math:`\sigma_\mathrm{jitt}`: additional error term (jitter) added quadratically to measurement errors, if requested.  
          It is then a variable parameter of the mcmc 
        + :math:`\sigma_\mathrm{val}^\mathrm{jitt} = \sqrt{ \sigma_\mathrm{val}^2 + \sigma_\mathrm{jitt}^2 }`

 
    Likelihood can also write as:

    .. math::          
       L[\mathrm{step}](x) &= \Pi_{x}{ \frac{1}{\sqrt{2 \pi} \sigma_\mathrm{val}^\mathrm{jitt}(x) }} \Pi_{x}{ \exp^{ - (\frac{ y_\mathrm{mod}[\mathrm{step}](x) - y_\mathrm{val}(x) }{\sqrt{2} \sigma_\mathrm{val}^\mathrm{jitt}(x)  })^2  }   }  \\
                           &= \Pi_{x}{ \frac{1}{\sqrt{2 \pi} \sigma_\mathrm{val}^\mathrm{jitt}(x) }} \exp^{ - \sum_{x}{  (\frac{ y_\mathrm{mod}[\mathrm{step}](x) - y_\mathrm{val}(x) }{\sqrt{2} \sigma_\mathrm{val}^\mathrm{jitt}(x)  })^2 } }  \\
                           &= \Pi_{x}{ \frac{1}{\sqrt{2 \pi} \sigma_\mathrm{val}^\mathrm{jitt}(x) }} \exp^{ - \frac{\chi^2[\mathrm{step}]}{2} }
 
    With 

    .. math:: 
       \chi^2[\mathrm{step}] = \sum_{x}{ (\frac{ y_\mathrm{mod}[\mathrm{step}](x) - y_\mathrm{val}(x) }{\sqrt{2} \sigma_\mathrm{val}^\mathrm{jitt}(x)  })^2 }

    The ln of likelihood is used to avoid to avoid issues with large :math:`\chi^2` and small exponential values.
    For similar reasons we use in our calculation the sum of ln rather than the ln of the product 
    
    .. math::     
       \ln{L[\mathrm{step}]}(x) &=     \ln(\Pi_{x}{ \frac{1}{\sqrt{2 \pi} \sigma_\mathrm{val}^\mathrm{jitt}(x) } })  + \ln(\exp{- \frac{\chi^2[\mathrm{step}]}{2}} )   \\
       \ln{L[\mathrm{step}]}(x) &=   -\sum_{x}{ \ln( \sqrt{2 \pi} \sigma_\mathrm{val}^\mathrm{jitt}(x) )}  - \frac{\chi^2[\mathrm{step}]}{2}  
                                    
    Which also writes as (yielding the form given on the `emcee` website)                                    

    .. math:: 
       \ln{L[\mathrm{step}]}(x) &=   - \sum_{x}{  \ln( ( 2 \pi \sigma_\mathrm{val}^\mathrm{jitt}(x)^2 )^{0.5}) }  - \frac{\chi^2[\mathrm{step}]}{2}  \\
       \ln{L[\mathrm{step}]}(x) &=   - \sum_{x}{ 0.5 \ln( 2 \pi \sigma_\mathrm{val}^\mathrm{jitt}(x)^2) }  - \frac{\chi^2[\mathrm{step}]}{2}        \\
       \ln{L[\mathrm{step}]}(x) &=   - 0.5 ( \sum_{x}(  \ln( 2 \pi \sigma_\mathrm{val}^\mathrm{jitt}(x)^2) )  + \chi^2[\mathrm{step}])   

    Args:
        TBD
    
    Returns:
        TBD
    
    """
    

    #Model for current set of parameter values  
    #    - fit_func returns either 
    # + the model, in which case y_val is set to the data and s_val ; cov_val to its variance ; covariance)
    # + a 'chi' array defined as the individual elements of 'chi2_step' below, in which case y_val is set to 0 and s_val to 1 (use_cov is set to False so that we enter the 'variance' condition below)
    #   so that 'res' below equals 'y_step' which is 'chi', and 'chi2_step' is the sum of the 'chi' values squared
    y_step = fixed_args['fit_func'](p_step, fixed_args['x_val'],args=fixed_args)
    
    #Fitted pixels
    idx_fit = fixed_args['idx_fit']
    
    #Residuals from data    
    #    - unfitted (and possibly undefined) residual values are replaced by 0, so that they do not contribute to the chi2
    res = np.zeros(len(y_step),dtype=float)
    res[idx_fit] = y_step[idx_fit] - fixed_args['y_val'][idx_fit]

    #Use of covariance matrix
    #    - must be calculated with the contiguous covariance matrix    
    #    - chi2 = r^t C^-1 r = r^t L^-1^t L^-1 r = chi^t chi avec chi = L^-1 r 
    if fixed_args['use_cov']:
        L_mat = scipy.linalg.cholesky_banded(fixed_args['cov_val'],lower=True)
        chi = scipy.linalg.blas.dtbsv(L_mat.shape[0]-1, L_mat, res, lower=True)

        #Remove chi of unfitted pixels
        chi2_step = np.sum(np.power(chi[idx_fit],2.)) 
        ln_lkhood = - np.sum(np.log(np.sqrt(2.*np.pi)*L_mat[0][idx_fit])) - (chi2_step/2.)   
        
    #Use of variance alone
    else:
    
        #Modification of error bars on fitted values in case of jitter used as free parameter
        if (fixed_args['jitter']):sjitt_val=np.sqrt(fixed_args['cov_val'][0,idx_fit] + p_step['jitter']**2.)
        else:sjitt_val=np.sqrt(fixed_args['cov_val'][0,idx_fit])
            
        #Chi2
        chi2_step=np.sum(  np.power( res[idx_fit]/sjitt_val,2.) )

        #Ln likelihood
        ln_lkhood = - np.sum(np.log(np.sqrt(2.*np.pi)*sjitt_val)) - (chi2_step/2.)   

    return ln_lkhood,chi2_step
    



def ln_prob_func_mcmc(p_step,fixed_args):
    r"""**Log-probability function: MCMC**

    Calculates the complete log-probability function combining prior and likelihood.

    Args:
        p_step (dict) : contains the values of the variable parameters at current MCMC step
        fixed_args (dict) : contains parameters useful to the calculation of the function, in particular the values of parameters that are fixed.
    
    Returns:
        ln_prob (float) : log-probability
    
    """
    
    #Combine fixed and chain parameters into single dictionary for model        
    p_step_all={}

    #Include variable parameters
    #    - the order of variable parameters in 'var_par_list' must match how they were defined in 'p_step'
    #    - we do not include yet the variable parameters as they might be modified through expression based on the local variable parameters
    for ipar,parname in enumerate(fixed_args['var_par_list']):
        p_step_all[parname]=p_step[ipar]

    #Update value of fixed parameters linked to variable parameters through an expression
    if len(fixed_args['linked_par_expr'])>0:
        
        #Attribute parameters values directly to their names so that they can be identified in the expressions
        #    - it would be better to only define parameters used in expression but they are not easy to identify from the expressions
        #    - exec allows us to attribute to the parameter name (and not the string of its name) its value
        #    - we also attribute their fixed value to parameters that are not variable and not linked via an expression, but might be used in the expression
        # of other non-variable parameters (it has to be done within each step, values are not exported from one function to the other)
        for par in fixed_args['var_par_list']:
            exec(str(par)+'='+str(p_step_all[par]))
        for par in fixed_args['fixed_par_val_noexp_list']:
            exec(str(par)+'='+str(fixed_args['fixed_par_val'][par]))

        #Update parameters with associated expression
        for par in fixed_args['linked_par_expr']:
            fixed_args['fixed_par_val'][par]=eval(fixed_args['linked_par_expr'][par])
         
    #Include fixed parameters into local input dictionary
    p_step_all.update(fixed_args['fixed_par_val'])
    
    #Prior function
    ln_prior = ln_prior_func(p_step_all,fixed_args)
    
    #Undefined log-prior
    #     - if the walker move to parameters outside the defined prior range, then the log-prior will be set to inf.
    #       as a result, there is no point in calculating the log-likelihood in this case.
    if not np.isinf(ln_prior):

        #Likelihood function
        ln_lkhood = ln_lkhood_func_mcmc(p_step_all,fixed_args)[0]

        #Set log-probability to -inf if likelihood is nan
        #    - happens when parameters go beyond their boundaries (hence ln_prior=-inf) but the model fails (hence ln_lkhood = nan)
        ln_prob=-np.inf if np.isnan(ln_lkhood) else ln_prior + ln_lkhood

    else: ln_prob=-np.inf

    return ln_prob


 
def ln_prob_func_lmfit(p_step, x_val, fixed_args=None):
    r"""**Log-probability function: lmfit**

    Calculates the complete log-probability function combining prior and likelihood.
    The function `minimizer()` requires the objective function to return an array of residuals 'chi' of which it minimizes the sum of squares, ie 

    .. math::    
       x &= \arg \min(\sum_{k} \chi_{k}^2) \\ 
         &= \arg \min(\chi^2)    \\
         &= \arg \min(-2 \ln{P}) 
         
    The `lmfit` minimizer expects `\chi` as an array, but defining :math:`\chi = [\sqrt{-2 \ln{P}/n_{k}}]` so that :math:`x = \arg \min(-2 \mathrm{\ln{P}})` seems to raise issues with error determination. 
      
    Args:
        p_step (dict) : contains the values of the variable parameters at current MCMC step
        x_val (array) : mock grid, required to call `ln_prob_func_lmfit()` with `minimizer()` 
        fixed_args (dict) : contains parameters useful to the calculation of the function, in particular the values of parameters that are fixed.
    
    Returns:
        chi (array, float) : :math:`\chi` values
    
    """
    #Model for current set of parameter values  
    y_step = fixed_args['fit_func'](p_step, fixed_args['x_val'],args=fixed_args)

    #Fitted pixels
    idx_fit = fixed_args['idx_fit']

    #Residuals from data    
    #    - unfitted (and possibly undefined) residual values are replaced by 0, so that they do not contribute to the chi2
    res = np.zeros(len(y_step),dtype=float)
    res[idx_fit] = y_step[idx_fit] - fixed_args['y_val'][idx_fit]

    #Likelihood
    #    - normalisation factor of Likelihood is ignored to retrieve the equivalent of chi2 
    #    - must be calculated with the contiguous covariance matrix
    if fixed_args['use_cov']:
        L_mat = scipy.linalg.cholesky_banded(fixed_args['cov_val'],lower=True)
        chi = scipy.linalg.blas.dtbsv(L_mat.shape[0]-1, L_mat, res, lower=True)   
     
        #Remove chi2 of unfitted pixels
        chi = chi[idx_fit]
    else:

        #Limit residuals and error table to fitted pixels
        chi = res[idx_fit]/np.sqrt(fixed_args['cov_val'][0,idx_fit])
        
    return chi


def gen_hrand_chain(par_med,epar_low,epar_high,n_throws):
    r"""**PDF generator**

    Generates normal or half-normal distributions of a parameter.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    if epar_high==epar_low:
        hrand_chain = np.random.normal(par_med, epar_high, n_throws)
    else:
        if n_throws>1:
            if n_throws<20:n_throws_half = 10*n_throws
            else:n_throws_half = 2*n_throws
            rand_draw_right = np.random.normal(loc=par_med, scale=epar_high, size=n_throws_half)
            rand_draw_right = rand_draw_right[rand_draw_right>par_med]
            rand_draw_right = rand_draw_right[0:int(n_throws/2)]
            rand_draw_left = np.random.normal(loc=par_med, scale=epar_low, size=n_throws_half)
            rand_draw_left = rand_draw_left[rand_draw_left<=par_med]
            rand_draw_left = rand_draw_left[0:n_throws-int(n_throws/2)]
            hrand_chain = np.append(rand_draw_left,rand_draw_right)
        else:
            if np.random.normal(loc=0., scale=1., size=1)>0:hrand_chain = np.random.normal(loc=par_med, scale=epar_high, size=1)
            else:hrand_chain = np.random.normal(loc=par_med, scale=epar_low, size=1)
    return hrand_chain   




##################################################################################################
#%%% Minimization routines
##################################################################################################   

def init_fit(fit_dic,fixed_args,p_start,fit_prop_dic,model_par_names,model_par_units):
    r"""**Fit initialization**

    Initializes lmfit and MCMC.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """
    
    #Start counter
    fit_dic['st0']=get_time()     

    #Fixed parameters values
    #    - retrieve parameters in the 'p_start' structure that will remain fixed to their value 
    #    - p_start cannot be given as input to the MCMC with both fixed and variable parameters (ie, they must be given separately)
    fixed_args['fixed_par_val']={par:p_start[par].value for par in p_start if not p_start[par].vary}
    
    #Parameters linked to variable parameters through an expression
    fixed_args['linked_par_expr']={par:p_start[par].expr for par in p_start if p_start[par].expr!=None}

    #Parameters names
    #    - order will be used as such afterward
    fixed_args['par_names']=[par for par in p_start]

    #List of variable parameters name and their indices in the table of all parameters
    var_par_list=[]
    ivar_par_list=[]
    ifix_par_list=[]
    fix_par_list=[]
    iexp_par_list=[]
    exp_par_list=[]
    var_par_names=[]
    var_par_units=[]
    for ipar,par in enumerate(fixed_args['par_names']):
        if p_start[par].vary:
            var_par_list+=[par]
            ivar_par_list+=[ipar]
            par_name_fit = par.split('__')[0]
            par_name_loc = model_par_names(par_name_fit)
            if ('__' in par):
                inst_par = '_'
                vis_par = '_'
                pl_name = None                
                sp_name = None
                
                #Parameter depends on epoch
                if ('__IS') and ('_VS') in par:
                    inst_vis_par = par.split('__IS')[1]
                    inst_par  = inst_vis_par.split('_VS')[0]
                    vis_par  = inst_vis_par.split('_VS')[1]   
                    if ('__sp' in par):sp_name = (par.split('__IS')[0]).split('__sp')[1]                    
                    if ('__pl' in par):pl_name = (par.split('__IS')[0]).split('__pl')[1]
                
                #Parameter does not depend on epoch
                else:
                    if ('__sp' in par):sp_name = par.split('__sp')[1]   
                    if ('__pl' in par):pl_name = par.split('__pl')[1]                                                                 
                if sp_name is not None:par_name_loc+='['+sp_name+']'
                if pl_name is not None:par_name_loc+='['+pl_name+']'                             
                if inst_par != '_':
                    par_name_loc+='['+inst_par+']'
                    if vis_par != '_':par_name_loc+='('+vis_par+')'
            var_par_names+=[par_name_loc]
            var_par_units+=[model_par_units(par_name_fit)]

        else:
            ifix_par_list+=[ipar]
            fix_par_list+=[par]
        if p_start[par].expr!=None:
            p_start[par].vary=False
            iexp_par_list+=[ipar]
            exp_par_list+=[par]
    fixed_args['var_par_list']=np.array(var_par_list,dtype='U50')  
    fixed_args['ivar_par_list']=ivar_par_list
    fixed_args['ifix_par_list']=ifix_par_list
    fixed_args['fix_par_list']=fix_par_list
    fixed_args['iexp_par_list']=iexp_par_list
    fixed_args['exp_par_list']=exp_par_list
    fixed_args['var_par_names']=np.array(var_par_names,dtype='U50')
    fixed_args['var_par_units']=np.array(var_par_units,dtype='U50')

    #Retrieve the number of spots that are present (whether their parameters are fixed or fitted)
    spot_names=[]
    for par in fixed_args['par_names']:
        if '__sp' in par:
            spot_names.append(par.split('__sp')[1])
    fixed_args['num_spots']=len(np.unique(spot_names))

    #Update value of fixed parameters linked to variable parameters through an expression
    if len(fixed_args['linked_par_expr'])>0:
        
        #Attribute their fixed value to parameters that are not variable and not linked via an expression, but might be used in the expression of other non-variable parameters
        fixed_args['fixed_par_val_noexp_list']=[par for par in fixed_args['fixed_par_val'] if par not in fixed_args['linked_par_expr']]

    #Number of free parameters    
    if fit_dic['fit_mode']=='':fit_dic['merit']['n_free'] = 0.
    else:fit_dic['merit']['n_free'] = len(var_par_list) 

    #Initialize save file
    fit_dic['save_outputs']=True if ('save_outputs' not in fit_prop_dic) else fit_prop_dic['save_outputs'] 
    if fit_dic['save_outputs']:
        if (not os_system.path.exists(fit_dic['save_dir'])):os_system.makedirs(fit_dic['save_dir'])
        fit_dic['file_save']=open(fit_dic['save_dir']+'Outputs','w+')

    #No jitter by default
    #    - can be used to reach a reduced chi2 of 1 by adding quadratically the adjusted jitter value to the measured errors
    fixed_args['jitter']=False if ('jitter' not in fixed_args) else fixed_args['jitter']        
    
    #Default settings
    if fit_dic['fit_mode']=='mcmc':

        #Monitor progress
        fit_dic['progress']=True if ('progress' not in fit_prop_dic) else fit_prop_dic['progress'] 

        #Do not use complex prior function by default
        fixed_args['prior_func'] = False if ('prior_func' not in fit_prop_dic) else fit_prop_dic['prior_func']

        #Excluding manually some of the walkers
        fit_dic['exclu_walk']=False if ('exclu_walk' not in fit_prop_dic) else fit_prop_dic['exclu_walk'] 

        #Excluding automatically walkers with median beyond +- threshold * (1 sigma) of global median 
        #    - set to None, or exclusion threshold
        fit_dic['exclu_walk_autom']=None if ('exclu_walk_autom' not in fit_prop_dic) else fit_prop_dic['exclu_walk_autom'] 

        #Excluding manually some of the samples
        fit_dic['exclu_samp']={} if ('exclu_samp' not in fit_prop_dic) else fit_prop_dic['exclu_samp'] 

        #Quantiles calculated by default
        fit_dic['calc_quant']=True if ('calc_quant' not in fit_prop_dic) else fit_prop_dic['calc_quant'] 

        #No thinning of the chains
        fit_dic['thin_MCMC']=False if ('thin_MCMC' not in fit_prop_dic) else fit_prop_dic['thin_MCMC'] 

        #Impose a specific maximum correlation length to thin the chains
        #    - otherwise set to 0 for automatic determination
        if fit_dic['thin_MCMC']:
            fit_dic['max_corr_length']=50. if ('max_corr_length' not in fit_prop_dic) else fit_prop_dic['max_corr_length']         

        #Calculation of 1sigma HDI intervals
        #    - set fit_dic['HDI'] to None in options to prevent calculation
        #    - applied within 'postMCMCwrapper_2' to the modified final chains
        fit_dic['HDI']='1s' if ('HDI' not in fit_prop_dic) else fit_prop_dic['HDI'] 
        
        #Number of bins in 1D histograms used for HDI definition
        #    - adjust HDI_nbins or HDI_dbins for each parameter: there must be enough bins for the HDI interval to contain a fraction of samples close
        # to the requested confidence interval, but not so much that bins within the histogram are empty and artificially create several HDI intervals
        #      alternatively set fit_dic['HDI_nbins']= {} or set it to None for a given value for automatic definition (preferred solution for unimodal PDFs)
        fit_dic['HDI_nbins']={} if ('HDI_nbins' not in fit_prop_dic) else fit_prop_dic['HDI_nbins'] 

        #No calculation of upper/lower limits
        #    - to be used for PDFs bounded by the parameter space
        #    - define the type of limits and the confidence level
        #      limits will then be calculated from the minimum or maximum of the distribution
        #    - this should return similar results as the HDI intervals, but more precise because it does not rely on the sampling of the PDF
        fit_dic['conf_limits']={} if ('conf_limits' not in fit_prop_dic) else fit_prop_dic['conf_limits']

        #Retrieve model sample
        #    - calculation of models sampling randomly the full distribution of the parameters
        #    - disabled by default
        fit_dic['calc_sampMCMC']=False if ('calc_sampMCMC' not in fit_prop_dic) else fit_prop_dic['calc_sampMCMC'] 

        #No calculation of envelopes
        #    - calculation of models using parameter values within their 1sigma range
        fit_dic['calc_envMCMC']=False if ('calc_envMCMC' not in fit_prop_dic) else fit_prop_dic['calc_envMCMC']
        if fit_dic['calc_envMCMC']:
            fit_dic['st_samp']=10 if ('st_samp' not in fit_prop_dic) else fit_prop_dic['st_samp']
            fit_dic['end_samp']=10 if ('end_samp' not in fit_prop_dic) else fit_prop_dic['end_samp']
            fit_dic['n_samp']=100 if ('n_samp' not in fit_prop_dic) else fit_prop_dic['n_samp']

        #On-screen printing of errors
        fit_dic['sig_list']=['1s'] if ('sig_list' not in fit_prop_dic) else fit_prop_dic['sig_list']  

        #Plot correlation diagram for final parameters      
        fit_dic['save_MCMC_corner']='pdf' if ('save_MCMC_corner' not in fit_prop_dic) else fit_prop_dic['save_MCMC_corner'] 
        if ('corner_options' in fit_prop_dic):fit_dic['corner_options'] = fit_prop_dic['corner_options'] 
        else:
            fit_dic['corner_options']={'plot_HDI':True,'color_levels':['deepskyblue','lime']}
        
        #Plot chains for MCMC parameters    
        fit_dic['save_MCMC_chains']='png' if ('save_MCMC_chains' not in fit_prop_dic) else fit_prop_dic['save_MCMC_chains'] 
        
        #Run name
        fit_dic['run_name']='' if ('run_name' not in fit_prop_dic) else fit_prop_dic['run_name']
        
        #MCMC reboot
        fit_dic['mcmc_reboot']='' if ('mcmc_reboot' not in fit_prop_dic) else fit_prop_dic['mcmc_reboot']  

        #MCMC monitoring
        fit_dic['monitor']=False if ('monitor' not in fit_prop_dic) else fit_prop_dic['monitor']  
        
    return None


def call_lmfit(p_use, xtofit, ytofit, covtofit, f_use,method='leastsq', maxfev=None, xtol=1e-7, ftol=1e-7,verbose=False,fixed_args=None,show_correl=False):
    r"""**Wrapper to lmfit**

    Runs `lmfit` minimizer and outputs results and merit values.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """

    #Call to minimization
    #    - WARNING: it is essential to use scale_covar = False to prevent the covariance matrix to be scaled using the reduced chi2 (in which case errors on the
    # fitted parameters do not depend on the scaling of the errors on the datapoints)
    #    - 'minimize' calls the 'ln_prob_func_lmfit' function with arguments:
    # param -> p_use
    # x -> xtofit
    # fixed_args ->  fixed_args
    #    - covtofit must have banded matrix structure, ie (nd+1,n) where nd is the number of sub-diagonals
    argstofit=deepcopy(fixed_args)
    argstofit['fit_func'] = f_use
    argstofit['x_val'] = deepcopy(xtofit)
    argstofit['y_val'] = deepcopy(ytofit)
    argstofit['cov_val'] = deepcopy(covtofit)
    if maxfev is not None:max_nfev = maxfev
    else:max_nfev = 2000*(len(xtofit)+1)
    st0=get_time()
    if method=='leastsq':
        result = minimize(ln_prob_func_lmfit, p_use, args=(xtofit, argstofit), method=method, max_nfev=max_nfev, xtol=xtol, ftol=ftol,scale_covar = False )
    else:
        if method=='lbfgsb':meth_args = {'tol':ftol}
        else:meth_args = {}
        result = minimize(ln_prob_func_lmfit, p_use, args=(xtofit, argstofit), method=method, max_nfev=max_nfev,scale_covar = False , **meth_args)
    if verbose:print('   duration : '+str((get_time()-st0)/60.)+' mn')
    
    #Best-fit parameters
    #    - attributes of the Minimizer object (here result):
    # nfev    number of function evaluations
    # success    boolean (True/False) for whether fit succeeded.
    # errorbars    boolean (True/False) for whether uncertainties were estimated.
    # message    message about fit success.
    # ier    integer error value from scipy.optimize.leastsq
    # lmdif_message    message from scipy.optimize.leastsq
    # nvarys    number of variables in fit  N_{\rm varys}
    # ndata    number of data points:  N
    # nfree    degrees of freedom in fit:  N - N_{\rm varys}
    # ln_prob_func_lmfit    ln_prob_func_lmfit array (return of func():  {\rm Resid}
    # chisqr    chi-square: \chi^2 = \sum_i^N [{\rm Resid}_i]^2
    # redchi    reduced chi-square: \chi^2_{\nu}= {\chi^2} / {(N - N_{\rm varys})}
    p_best=result.params 
    merit={}
    
    #Model function with best-fit parameters
    merit['fit'] = f_use(p_best, xtofit,args=argstofit)
    
    #Corresponding residuals
    merit['resid'] = merit['fit'] - ytofit
    
    #Dispersion of residuals
    merit['rms'] = merit['resid'].std()

    #Chi2 value
    merit['chi2'] = np.sum(ln_prob_func_lmfit(p_best, xtofit, fixed_args=argstofit)**2.)
    merit['chi2r'] = merit['chi2']/result.nfree
 
    #Bayesian Indicator Criterion   
    merit['BIC'] = merit['chi2'] + result.nvarys*np.log(result.ndata)
    
    #Cumulative distribution function
    #    - cdf = (0.05<cdf<0.95, if not: too bad/good fit or big/small error)
    merit['cdf'] = special.chdtrc(result.nfree,merit['chi2'])

    #Print information on screen
    if verbose:
        print("fit report:")
        print("[[Fit Statistics]]")
        print("    # fit success           = %r"%result.success,)
        if not result.success:print(": " + result.message[:-1])
        else:
            if len(result.message)>32:print(": " , result.message)
            else:print(": " , result.message[:-1])
        print("    # function evals        = %i"%result.nfev)
        print("    # data points           = %i"%result.ndata)
        print("    # degree of freedom     = %i"%result.nfree)
        print("    # free variables        = %i"%result.nvarys)
        print("    chi-square              = %f"%merit['chi2'])
        print("    Bayesian ind crit (BIC) = %f"%merit['BIC'])
        print("    reduced chi-square      = %f"%(merit['chi2']/result.nfree))
        print("    RMS                     = %f"%merit['rms'])
        print("    cumul dist funct (cdf)  = %f"%merit['cdf'])
        report_fit(p_best,show_correl=show_correl, min_correl=0.1)

    return result, merit ,p_best
    
    



def call_MCMC(nthreads,fixed_args,fit_dic,run_name='',verbose=True,save_raw=True):
    r"""**Wrapper to MCMC**

    Runs `emcee` and outputs results and merit values.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """
    #Automatic definition of undefined priors                
    for par in fixed_args['var_par_list']:
        if par not in fixed_args['varpar_priors']:fixed_args['varpar_priors'][par]={'mod':'uf','low':-1e10,'high':1e10}

    #Set initial parameter distribution
    fit_dic['initial_distribution'] = np.zeros((fit_dic['nwalkers'],fit_dic['merit']['n_free']))
    if (len(fit_dic['mcmc_reboot'])>0):
        print('         Rebooting previous run')
        
        #Reboot MCMC from end of previous run
        walker_chains_last=np.load(fit_dic['mcmc_reboot'])['walker_chains'][:,-1,:]  #(nwalkers, nsteps, n_free)
  
        #Overwrite starting values of new chains
        for ipar in range(len(fixed_args['var_par_list'])):
            fit_dic['initial_distribution'][:,ipar] = walker_chains_last[:,ipar] 
          
    else:
        
        #Random distribution within defined range
        for ipar,par in enumerate(fixed_args['var_par_list']):
            fit_dic['initial_distribution'][:,ipar]=np.random.uniform(low=fit_dic['uf_bd'][par][0], high=fit_dic['uf_bd'][par][1], size=fit_dic['nwalkers'])                     

    #By default use variance
    if 'use_cov' not in fixed_args:fixed_args['use_cov']=False

    #Save temporary walkers in case of crash
    if ('monitor' in fit_dic) and fit_dic['monitor']:
        backend = emcee.backends.HDFBackend(fit_dic['save_dir']+'monitor'+str(fit_dic['nwalkers'])+'_steps'+str(fit_dic['nsteps'])+run_name+'.h5')
        backend.reset(fit_dic['nwalkers'], fit_dic['merit']['n_free'])
    else:backend=None

    
    #Call to MCMC
    st0=get_time()
    n_free=np.shape(fit_dic['initial_distribution'])[1]

    #Multiprocessing
    if nthreads>1:
        pool_proc = Pool(processes=nthreads)  
        print('         Running with '+str(nthreads)+' threads')    
        sampler = emcee.EnsembleSampler(fit_dic['nwalkers'],            #Number of walkers
                                        n_free,                         #Number of free parameters in the model
                                        ln_prob_func_mcmc,              #Log-probability function 
                                        args=[fixed_args],              #Fixed arguments for the calculation of the likelihood and priors
                                        pool = pool_proc,
                                        backend=backend)                #Monitor chain progress 
    else:sampler = emcee.EnsembleSampler(fit_dic['nwalkers'],n_free,ln_prob_func_mcmc,args=[fixed_args],backend=backend)         
        
    #Run MCMC
    #    - possible options:
    # + iterations: number of iterations to run            
    sampler.run_mcmc(fit_dic['initial_distribution'], fit_dic['nsteps'],progress=fit_dic['progress'])
    if verbose:print('   duration : '+str((get_time()-st0)/60.)+' mn')
   
    #Walkers chain
    #    - sampler.chain is of shape (nwalkers, nsteps, n_free)
    #     - parameters have the same order as in 'initial_distribution' and 'var_par_list'
    walker_chains = sampler.chain    
 
    #Save raw MCMC results 
    if save_raw:
        if (not os_system.path.exists(fit_dic['save_dir'])):os_system.makedirs(fit_dic['save_dir'])
        np.savez(fit_dic['save_dir']+'raw_chains_walk'+str(fit_dic['nwalkers'])+'_steps'+str(fit_dic['nsteps'])+run_name,walker_chains=walker_chains, initial_distribution=fit_dic['initial_distribution'])

    #Delete temporary chains after final walkers are saved
    if backend is not None:os_system.remove(fit_dic['save_dir']+'monitor'+str(fit_dic['nwalkers'])+'_steps'+str(fit_dic['nsteps'])+run_name+'.h5')

    #Close workers
    if nthreads>1:    
        pool_proc.close()
        pool_proc.join() 	

    return walker_chains



##################################################################################################
#%%% Post-processing
##################################################################################################   
       
def fit_merit(p_final_in,fixed_args,fit_dic,verbose):
    r"""**Post-proc: fit merit values**

    Calculates various indicators of the best-fit merit.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """
    if verbose:print('     Calculating merit values') 

    #Convert parameters() structure into dictionary 
    if fit_dic['fit_mode'] !='mcmc': 
        p_final={}
        for par in p_final_in:p_final[par]=p_final_in[par].value   
        if fit_dic['fit_mode']=='chi2':
            fit_dic['sig_parfinal_err']={'1s':np.zeros([2,fit_dic['merit']['n_free']])}            
            for ipar,par in enumerate(fixed_args['var_par_list']): 
                fit_dic['sig_parfinal_err']['1s'][:,ipar]=p_final_in[par].stderr  
    else:p_final = deepcopy(p_final_in)
 
    #Calculation of best-fit model equivalent to the observations, corresponding residuals, and RMS
    #    - only in the case where the function does return the model
    if not fixed_args['inside_fit']:
        res_tab = fixed_args['y_val'] - fixed_args['fit_func'](p_final,fixed_args['x_val'],args=fixed_args) 
        fit_dic['merit']['rms']=res_tab.std()       
    else:fit_dic['merit']['rms']='Undefined'

    #Merit values 
    if fit_dic['fit_mode'] =='':fit_dic['merit']['mode']='forward'    
    else:fit_dic['merit']['mode']='fit'    
    fit_dic['merit']['dof']=fit_dic['nx_fit']-fit_dic['merit']['n_free']
    
    if fit_dic['fit_mode'] in ['','chi2']: fit_dic['merit']['chi2']=np.sum(ln_prob_func_lmfit(p_final,fixed_args['x_val'], fixed_args=fixed_args)**2.)
    elif fit_dic['fit_mode'] =='mcmc': fit_dic['merit']['chi2']=ln_lkhood_func_mcmc(p_final,fixed_args)[1] 
    fit_dic['merit']['red_chi2']=fit_dic['merit']['chi2']/fit_dic['merit']['dof']
    fit_dic['merit']['BIC']=fit_dic['merit']['chi2']+fit_dic['merit']['n_free']*np.log(fit_dic['nx_fit'])      
    fit_dic['merit']['AIC']=fit_dic['merit']['chi2']+2.*fit_dic['merit']['n_free']
    
    if verbose:
        print('     + Npts = ',fit_dic['nx_fit'])
        print('     + Nfree = ',fit_dic['merit']['n_free'])
        print('     + d.o.f =',fit_dic['merit']['dof'])
        print('     + Best chi2 = '+str(fit_dic['merit']['chi2']))
        print('     + Reduced Chi2 ='+str(fit_dic['merit']['red_chi2']))
        print('     + RMS of residuals = '+str(fit_dic['merit']['rms'])) 
        print('     + BIC ='+str(fit_dic['merit']['BIC']))       
        print('     + Parameters :')
        for par in fixed_args['fixed_par_val']:print('        ',par,'=',"{0:.10e}".format(p_final[par]))                   
        if fit_dic['fit_mode'] =='':
            for par in fixed_args['var_par_list']:print('        ',par,'=',"{0:.10e}".format(p_final[par]))                
        else:
            for ipar,par in enumerate(fixed_args['var_par_list']):print('        ',par,'=',"{0:.10e}".format(p_final[par]),'+-',"{0:.10e}".format(fit_dic['sig_parfinal_err']['1s'][0,ipar]))   

    #End counter
    fit_dic['fit_dur'] = get_time()-fit_dic['st0']
    if verbose:print('     Fit duration =',fit_dic['fit_dur'],' s')
            
    #Fit information
    if fit_dic['save_outputs']:
        save_fit_results('merit',fixed_args,fit_dic,fit_dic['fit_mode'],p_final)
        save_fit_results('nominal',fixed_args,fit_dic,fit_dic['fit_mode'],p_final)    

    return p_final
    

      
def save_fit_results(part,fixed_args,fit_dic,fit_mode,p_final):
    r"""**Post-proc: fit outputs**

    Saves merit indicators of the best-fit model, as well as best-fit values and confidence intervals for the original and derived parameters.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """
    file_path=fit_dic['file_save']

    if part=='merit':
        np.savetxt(file_path,[['----------------------------------']],fmt=['%s'])
        np.savetxt(file_path,[['Merit values']],fmt=['%s']) 
        np.savetxt(file_path,[['----------------------------------']],fmt=['%s']) 
        np.savetxt(file_path,[['Duration : '+str(fit_dic['fit_dur'])+' s']],delimiter='\t',fmt=['%s']) 
        np.savetxt(file_path,[['Npts : '+str(fit_dic['nx_fit'])]],delimiter='\t',fmt=['%s']) 
        np.savetxt(file_path,[['Nfree : '+str(fit_dic['merit']['n_free'])]],delimiter='\t',fmt=['%s']) 
        np.savetxt(file_path,[['d.o.f : '+str(fit_dic['merit']['dof'])]],delimiter='\t',fmt=['%s']) 
        np.savetxt(file_path,[['Best chi2 : '+str(fit_dic['merit']['chi2'])]],delimiter='\t',fmt=['%s']) 
        np.savetxt(file_path,[['Reduced chi2 : '+str(fit_dic['merit']['red_chi2'])]],delimiter='\t',fmt=['%s'])
        np.savetxt(file_path,[['RMS of residuals : '+str(fit_dic['merit']['rms'])]],delimiter='\t',fmt=['%s']) 
        np.savetxt(file_path,[['BIC : '+str(fit_dic['merit']['BIC'])]],delimiter='\t',fmt=['%s'])
        np.savetxt(file_path,[['AIC : '+str(fit_dic['merit']['AIC'])]],delimiter='\t',fmt=['%s'])
        np.savetxt(file_path,[['----------------------------------']],fmt=['%s']) 
        np.savetxt(file_path,[['']],fmt=['%s']) 
    
    if part in ['nominal','derived']:
        if part=='nominal':        
            np.savetxt(file_path,[['----------------------------------']],fmt=['%s'])
            np.savetxt(file_path,[['Nominal best-fit parameters']],fmt=['%s']) 
            np.savetxt(file_path,[['----------------------------------']],fmt=['%s'])   
            np.savetxt(file_path,[['Fixed']],delimiter='\t',fmt=['%s'])
            for parname in fixed_args['fixed_par_val']:  
                np.savetxt(file_path,[['']],delimiter='\t',fmt=['%s'])
                np.savetxt(file_path,[[parname+'\t'+"{0:.10e}".format(p_final[parname])]],delimiter='\t',fmt=['%s'])
            np.savetxt(file_path,[['-----------------']],fmt=['%s'])
        elif part=='derived': 
            np.savetxt(file_path,[['----------------------------------']],fmt=['%s'])
            np.savetxt(file_path,[['Derived parameters']],fmt=['%s']) 
            np.savetxt(file_path,[['----------------------------------']],fmt=['%s'])     

            #Calculation of null model hypothesis
            #    - to calculate chi2 (=BIC) with respect to a null level for comparison of best-fit model with null hypothesis
            if 'p_null' in fit_dic:
                if fit_dic['fit_mode'] in ['','chi2']: chi2_null=np.sum(ln_prob_func_lmfit(fit_dic['p_null'], fixed_args['x_val'], fixed_args=fixed_args)**2.)
                elif fit_dic['fit_mode'] =='mcmc':chi2_null=ln_lkhood_func_mcmc(fit_dic['p_null'],fixed_args)[1]        
                np.savetxt(file_path,[['']],delimiter='\t',fmt=['%s'])
                np.savetxt(file_path,[['Null chi2 : '+str(chi2_null)]],delimiter='\t',fmt=['%s']) 
                np.savetxt(file_path,[['----------------------------------']],fmt=['%s'])
                np.savetxt(file_path,[['']],delimiter='\t',fmt=['%s'])
                
        np.savetxt(file_path,[['Parameters','med','-1s','+1s','med-1s','med+1s']],delimiter='\t',fmt=['%s']*6)
        for ipar,parname in enumerate(fixed_args['var_par_list']):    
            nom_val = p_final[parname]
            if 'sig_parfinal_err' in fit_dic:
                lower_sig= fit_dic['sig_parfinal_err']['1s'][0,ipar]
                upper_sig= fit_dic['sig_parfinal_err']['1s'][1,ipar] 
            else:
                lower_sig = np.nan
                upper_sig = np.nan
            np.savetxt(file_path,[['']],delimiter='\t',fmt=['%s'])
            data_save =parname+'\t'+"{0:.10e}".format(nom_val)+'\t'+"{0:.10e}".format(lower_sig)+'\t'+"{0:.10e}".format(upper_sig)+'\t'+"{0:.10e}".format(nom_val-lower_sig)+'\t'+"{0:.10e}".format(nom_val+upper_sig)
            np.savetxt(file_path,[data_save],delimiter='\t',fmt=['%s']) 
            if (fit_mode=='mcmc') and (part=='derived'):
                if (fit_dic['HDI'] is not None):
                    np.savetxt(file_path,['     HDI '+fit_dic['HDI']+' int : '+fit_dic['HDI_interv_txt'][ipar]],delimiter='\t',fmt=['%s'])
                    np.savetxt(file_path,['     HDI '+fit_dic['HDI']+' err: '+fit_dic['HDI_sig_txt'][ipar]],delimiter='\t',fmt=['%s'])
                if parname in fit_dic['conf_limits']:
                    for lev in fit_dic['conf_limits'][parname]['level']: 
                        np.savetxt(file_path,['     '+fit_dic['conf_limits'][parname]['limits'][lev]],delimiter='\t',fmt=['%s'])            
        np.savetxt(file_path,[['']],fmt=['%s'])    

    return None
    
    

  

##################################################################################################
#%%%% MCMC analysis
##################################################################################################   

def MCMC_corr_length(fit_dic,max_corr_length,nthreads,var_par_list,merged_chain,istart,iend,verbose=False):
    r"""**MCMC post-proc: correlation length**

    Calculates correlation length of each fitted parameter using MCMC chains.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """    
    #--------------------------------------
    #Correlation length is calculated   
    if max_corr_length==0.:    
        n_free=len(merged_chain[0,:])
        
        #Calculation: at least 10 points to measure CL
        if (iend-istart)>10.:     
            corr_length=np.zeros(n_free)
    
            #Moyenne des valeurs parcourues
            mean_params  =np.mean(merged_chain[istart:iend+1,:],axis=0)
        
            #Ecart des valeurs a la moyenne (npts x n_free)
            d_param=merged_chain-mean_params
            
            #Limitation a la premiere moitie de la chaine
            di=int((iend-istart)/2.)
            idx_di=istart+np.arange(di+1)
            d_param_di=d_param.take(idx_di,axis=0)
        
            #Moyenne des ecart sur la premiere moitie de la chaine, au carre
            mean_d_param_2  = np.mean(d_param_di,axis=0)**2.
        
            #Moyenne des ecart au carre sur la premiere moitie de la chaine
            mean_d2_param = np.mean(d_param_di**2.,axis=0)

            #Pour chaque param variable:
            for ipar,(mean_d2_par,mean_d_par_2) in enumerate(zip(mean_d2_param,mean_d_param_2)): 
    
                #Moyenne du produit des ecarts a la moyenne sur la premiere moitie de la chaine, et sur la meme longueur decalee successivement de 1 pixel
                if (nthreads>1):                
                    common_args=(d_param_di[:,ipar],d_param[:,ipar],idx_di)
                    chunkable_args=[np.arange(di)]
                    corr_j=parallel_sub_MCMC_corr_length(sub_MCMC_corr_length,nthreads,di,chunkable_args,common_args)                           
                else:
                    corr_j=sub_MCMC_corr_length(np.arange(di),**common_args)
    
                #Correlation values  
                cl_j=(corr_j-mean_d_par_2)/(mean_d2_par-mean_d_par_2)
    
                #Correlation length  
                ind=np_where1D(cl_j <= 0.5)
                if len(ind) != 0 : corr_length[ipar]=ind[0]
    
            #Effective number of uncorrelated points along each dimension (in the post burn-in chain)
            eff_length=len(merged_chain[:,0])/corr_length

            #Lengths converted to int
            corr_length=npint(corr_length)
            eff_length=npint(eff_length) 

            #Print results
            if verbose==True:
                print(' > Correlation and effective lengths :')
                for ipar,parname in enumerate(var_par_list):
                    print('    ',parname+' : ',str(corr_length[ipar]),' - ',str(eff_length[ipar]))

            #Sauvegarde
            if fit_dic['save_outputs']:
                np.savetxt(fit_dic['file_save'],[['Correlation lengths : ']],fmt=['%s'])                 
                for ipar,parname in enumerate(var_par_list):
                    np.savetxt(fit_dic['file_save'],[[parname,str(corr_length[ipar])]],delimiter='\t',fmt=['%s','%s']) 
    
        #Not enough points to measure correlation length
        else:
            corr_length=np.repeat(0,n_free)
            eff_length=np.repeat(0,n_free)
            print('WARNING : Not enough steps to calculate correlation lengths')    

    #--------------------------------------
    #Correlation length is fixed
    else:
        corr_length=[int(max_corr_length)]

        #Sauvegarde
        if fit_dic['save_outputs']:
            np.savetxt(fit_dic['file_save'],[['Fixed correlation lengths : ']],fmt=['%s'])                 
            np.savetxt(fit_dic['file_save'],[[str(corr_length[0])]],delimiter='\t',fmt=['%s']) 

    return corr_length  
  

def sub_MCMC_corr_length(pix_shift,d_par_di,d_par,idx_di):
    r"""**Chain correlation length**

    Calculates correlation length over a given parameter chain.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """     
    #Average of product of deviation from the mean over the first half of the chain, and over the same length shifted successively by one pixel.
    #    - time-consuming routine
    corr_j_loc=np.array([np.mean(d_par_di*d_par[j+idx_di]) for j in pix_shift])
    
    return corr_j_loc
    
def parallel_sub_MCMC_corr_length(func_input,nthreads,n_elem,y_inputs,common_args):
    r"""**Multithreading routine for sub_MCMC_corr_length().**

    Args:
        func_input (function): multi-threaded function
        nthreads (int): number of threads
        n_elem (int): number of elements to thread
        y_inputs (list): threadable function inputs 
        common_args (tuple): common function inputs
    
    Returns:
        y_output (None or specific to func_input): function outputs 
    
    """      
    pool_proc = Pool(processes=nthreads)   #cannot be passed through lmfit   
    ind_chunk_list=init_parallel_func(nthreads,n_elem)
    chunked_args=[(y_inputs[0][ind_chunk[0]:ind_chunk[1]],)+common_args for ind_chunk in ind_chunk_list]	  #1 array of n dictionary elements
    all_results=tuple(tab for tab in pool_proc.starmap(func_input,chunked_args)) 
    y_output=np.concatenate(tuple(all_results[i] for i in range(nthreads)))   #array with dimensions n 		
    pool_proc.close()
    pool_proc.join() 	
    return y_output
    
  
    
  
    

def MCMC_thin_chains(corr_length,merged_chain):  
    r"""**MCMC post-proc: thinning**

    Thins chains based on their correlation length.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    corr_length_max=int(max(corr_length))
    if corr_length_max==0.:
        print('No thinning: correlation length is null')
    else:
        merged_chain=merged_chain[0::corr_length_max,:]
        print('Merged chain is thinned: '+str(len(merged_chain[:,0]))+' samples remaining')
    nsteps_final_merged=len(merged_chain[:,0])

    return nsteps_final_merged,merged_chain  

  
  

def MCMC_retrieve_sample(fixed_args,fix_par_list,exp_par_list,iexp_par_list,ifix_par_list,par_names,fixed_par_val,calc_envMCMC,merged_chain,n_free,nsteps_final_merged,p_best_in,var_par_list,
                         ivar_par_list,calc_sampMCMC,linked_par_expr,fit_dic,st_samp,end_samp,n_samp):  
    r"""**MCMC post-proc: sampling**

    Retrieve MCMC samples randomy or within 1 sigma from the best fit. 
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Number of model parameters
    n_par=len(p_best_in)    
    
    if (calc_envMCMC==True) or (calc_sampMCMC==True):        
      
        #Create array that will contain all input parameters (fixed and variable) at their position in the original ordered dictionary  
        var_chain=np.zeros([nsteps_final_merged,n_par])
       
        #Variable parameters
        #    - 'ivar_par_list' contains their position in the original ordered dictionary
        var_chain[:,ivar_par_list]=merged_chain        
        
        #Fixed parameters
        for ipar_fix,par_fix in zip(ifix_par_list,fix_par_list): 
            var_chain[:,ipar_fix]=fixed_par_val[par_fix]        
           
        #Define values step by step for fixed parameters defined via expressions 
        if len(linked_par_expr)>0: 
            
            #Non-variable parameters not defined via expressions will not be updated, and have constant value
            for ipar_fix,par_fix in zip(ifix_par_list,fixed_args['fixed_par_val_noexp_list']):   
                exec(str(par_fix)+'='+str(var_chain[0,ipar_fix]))            
            
            for istep in range(nsteps_final_merged): 

                #Attribute variable parameters values directly to their names so that they can be identified in the expressions
                #    - we use indexes in the original order
                #    - we also attribute the value of fixed parameters not defined via expression, which can be used in expressions 
                for ipar_var,par_var in zip(ivar_par_list,var_par_list):   
                    exec(str(par_var)+'='+str(var_chain[istep,ipar_var]))
                    
                #Define non-variable parameters via expressions
                #    - we use indexes in the original order
                for ipar_exp,par_exp in zip(iexp_par_list,exp_par_list): 
                    var_chain[istep,ipar_exp]=eval(linked_par_expr[par_exp])
 
    #---------------------------------------------------
    #Retrieve model envelope 
    #    - to calculate models within the +-1 sigma range of the parameters, or within the HDI intervals
    if calc_envMCMC:

        #Lower/upper boundaries of 1 sigma HDI range for each parameter (size 2 x n_par)
        #    - we need the HDI intervals to already be calculated
        if fit_dic['HDI'] is not None:
            if fit_dic['HDI']!='1s':stop('Set HDI to 1 sigma for envelope calculation')
            sig1_par_range=np.zeros([2,n_par])
            for ipar,ipar_var in enumerate(ivar_par_list):
                sig1_par_range[0,ipar_var]=fit_dic['HDI_interv'][ipar][0][0]
                sig1_par_range[1,ipar_var]=fit_dic['HDI_interv'][ipar][-1][1]
        
        #Lower/upper boundaries of 1 sigma range for each parameter (size 2 x n_par)
        #    - see MCMC_estimates for explanations
        #    - it is easier to consider all parameters, even fixed ones, as they can be defined via expressions
        else:
            sig1_par_range=np.percentile(var_chain,[15.865525393145703, 84.1344746068543], axis=0) 

        #Retrieve chain steps for which all parameters are within their 1 sigma range
        #    - fixed parameters do not need to be checked
        cond_sig1=np.repeat(True,nsteps_final_merged)
        for ipar in ivar_par_list:
            cond_sig1=cond_sig1 & ((var_chain[:,ipar]>=sig1_par_range[0,ipar]) & (var_chain[:,ipar]<=sig1_par_range[1,ipar]))
        n_samp_sig1=np.sum(cond_sig1)

        #Retrieve parameters set for the selected samples
        var_chain_samp=var_chain[cond_sig1,:]

        #Map array elements into dictionary
        par_sample_sig1=np.empty([n_samp_sig1],dtype=object)
        for isamp in range(n_samp_sig1):
            par_sample_sig1[isamp]=dict(zip(par_names,var_chain_samp[isamp,:]))

    else:
        par_sample_sig1=None

    #----------------------------------------------

    #Retrieve model sample 
    #    - distribution of models following the PDF of the parameters
    if calc_sampMCMC==True:
        
        #Draw nsample parameters from the chain to evaluate the model function there
        #    - we can draw the merged_chain with a fixed frequency along the chain, but it should be larger than the 
        # maximum correlation length
        d_samp=int((nsteps_final_merged-end_samp-st_samp)/n_samp)
        if d_samp==0:
            par_sample=None
            calc_sampMCMC=False
        else:
            sample_plot=np.arange(st_samp,nsteps_final_merged-end_samp,d_samp,dtype=int) 

            #Retrieve parameters set for the selected samples
            var_chain_samp=var_chain[sample_plot,:]
    
            #Map array elements into dictionary
            par_sample=np.empty([n_samp],dtype=object)
            for isamp in range(n_samp):                       
                par_sample[isamp]=dict(zip(par_names,var_chain_samp[isamp,:]))

    else:
        par_sample=None
          
    return par_sample_sig1,par_sample  
  
  
  
def MCMC_HDI(chain_par,nbins_par,dbins_par,bw_fact,frac_search,HDI_interv_par,HDI_interv_txt_par,HDI_sig_txt_par,med_par):
    r"""**MCMC post-proc: HDI intervals**

    Calculates Highest Density Intervals of fitted MCMC parameters.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """      
    #Perform preliminary calculation with default smoothed density profile  
    dbin_par,dens_par,bin_edges_par = PDF_smooth(chain_par,1.)
    jumpind,bins_in_HDI,sorted_binnumber = prep_MCMC_HDI(dbin_par,dens_par,frac_search)

    #Alternative approach to avoid multiple intervals due to poor resolution of the PDF
    if (len(jumpind)>0) and ((nbins_par is not None) or (dbins_par is not None) or (bw_fact is not None)):

        #Calculate PDF of the sample distribution with manual bin size
        if (nbins_par is not None) or (dbins_par is not None):
            dbin_par,dens_par,bin_edges_par = PDF_hist(chain_par,nbins_par,dbins_par)
        
        #Define smoothed density profile using Gaussian kernels      
        else:
            dbin_par,dens_par,bin_edges_par = PDF_smooth(chain_par,bw_fact)
            
        #Calculate intervals from density distribution
        jumpind,bins_in_HDI,sorted_binnumber = prep_MCMC_HDI(dbin_par,dens_par,frac_search) 
    
    #Single interval
    if len(jumpind)==0:
        HDI_sub=[np.min(bin_edges_par[bins_in_HDI]),np.max(bin_edges_par[bins_in_HDI+1])]
        HDI_interv_par+=[HDI_sub]
        HDI_interv_txt_par+='['+"{0:.3e}".format(HDI_sub[0])+' ; '+"{0:.3e}".format(HDI_sub[1])+']'
        HDI_sig_txt_par+='[-'+"{0:.3e}".format(med_par-HDI_sub[0])+' +'+"{0:.3e}".format(HDI_sub[1]-med_par)+']'
    
    #Multiple intervals
    else:        
        for i_int in range(len(jumpind)+1):
            if i_int==0:ji=0
            else:ji=jumpind[i_int-1]+1     
            if i_int==len(jumpind):jf=-1
            else:jf=jumpind[i_int]
            HDI_sub=[bin_edges_par[sorted_binnumber[ji]],bin_edges_par[sorted_binnumber[jf]+1]]
            HDI_interv_par+=[HDI_sub]
            HDI_interv_txt_par+='['+"{0:.3e}".format(HDI_sub[0])+' ; '+"{0:.3e}".format(HDI_sub[1])+']' 
            HDI_sig_txt_par+='[-'+"{0:.3e}".format(med_par-HDI_sub[0])+' +'+"{0:.3e}".format(HDI_sub[1]-med_par)+']' 
   
    #Convert into array
    HDI_interv_par=np.array(HDI_interv_par)            

    #Calculate HDI fraction for verification
    HDI_frac_par=np.sum(dens_par[bins_in_HDI],dtype=float)/np.sum(dens_par,dtype=float)
    HDI_interv_txt_par+=' ('+"{0:.2f}".format(100.*HDI_frac_par)+' %)'    

    return HDI_interv_txt_par,HDI_frac_par,HDI_sig_txt_par  
  

def prep_MCMC_HDI(dbin_par,dens_par,frac_search):
    r"""**MCMC post-proc: HDI intervals initialization**

    Preliminary analysis of sample to prepare HDI calculations.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """   
    #Indexes of bins sorted by decreasing order
    isort = np.argsort(dens_par)[::-1]
       
    #Start adding bins until the requested fraction of the mass (q) is reached
    frac_HDI = 0.
    for isub,(hist_par_loc,dbin_par_loc) in enumerate(zip(dens_par[isort],dbin_par[isort])): 
        frac_HDI += hist_par_loc*dbin_par_loc                  
        if frac_HDI >= frac_search:break    
        
    #Keep only bins in 100*q% HDI
    bins_in_HDI = isort[0:isub+1]
  
    #Sort bin indexes to find non-contiguous bins
    if len(bins_in_HDI)>1:
        sorted_binnumber = bins_in_HDI[np.argsort(bins_in_HDI)]
        jumpind = np_where1D(  (sorted_binnumber[1::]-sorted_binnumber[0:-1]) > 1.)
    else:
        jumpind=[]
        
    return jumpind,bins_in_HDI,sorted_binnumber

    
def PDF_hist(chain_par,nbins_par,dbins_par):
    r"""**Histogram-based PDF**

    Calculates sample distribution as an histogram with manual settings.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Calculate PDF of a sample distribution with manual bin size
    #    - normalized by total "mass" so that we get a density distribution
    if (nbins_par is not None):nbins_par_loc = nbins_par
    elif (dbins_par is not None):nbins_par_loc = int(np.ceil((np.max(chain_par)-np.min(chain_par))/dbins_par))        
    hist_par,bin_edges_par=np.histogram(chain_par,bins=nbins_par_loc)
    dbin_par=bin_edges_par[1::]-bin_edges_par[0:-1]
    dens_par=hist_par/np.sum(hist_par*dbin_par)  
        
    return dbin_par,dens_par,bin_edges_par

def PDF_smooth(chain_par,bw_fact):
    r"""**Smoothed PDF**

    Calculates sample distribution as a smoothed density profile using Gaussian kernels. 
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Define smoothed density profile using Gaussian kernels      
    #    - includes automatic bandwidth determination    
    #    - works best for a unimodal distribution; bimodal or multi-modal distributions tend to be oversmoothed.
    #    - default 'scott' approach used bw = npts**(-1./(dimension + 4))
    dens_func = stats.gaussian_kde(chain_par,bw_method=bw_fact*len(chain_par)**(-1./5.))
    
    #Calculate density profile over HR grid
    bin_edges_par = np.linspace(np.min(chain_par), np.max(chain_par), 2001)
    dbin_par=bin_edges_par[1::]-bin_edges_par[0:-1]
    bin_cen_par = 0.5*(bin_edges_par[1::]+bin_edges_par[0:-1])
    dens_par = dens_func(bin_cen_par) 

    return dbin_par,dens_par,bin_edges_par


def quantile(x, q, weights=None):
    r"""**MCMC post-proc: quantiles**
    
    Compute sample quantiles with support for weighted samples.

    Note:
        When ``weights`` is ``None``, this method simply calls numpy's percentile function with the values of ``q`` multiplied by 100.

    Args:
        x (array, float): the samples.
        
        q (array, float): the list of quantiles to compute. These should all be in the range ``[0, 1]``.

        weights (array, float): an optional weight corresponding to each sample. 

    Returns:
        quantiles (array, float): the sample quantiles computed at ``q``.

    Raises:
        ValueError: For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch between ``x`` and ``weights``.

    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, 100.0 * q)
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()


 
def MCMC_estimates(merged_chain,fixed_args,fit_dic,verbose=True,print_par=True,calc_quant=True,verb_shift=''):
    r"""**MCMC post-proc: best fit**

    Calculates best estimates and confidence intervals for fitted MCMC parameters.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """  
    n_par=len(merged_chain[0,:])
  
    #Median
    med_par=np.median(merged_chain, axis=0)   
    
    #Use median values of variable parameters as best estimate for the nominal model
    #    - parameters in 'merged_chain' and thus 'med_par' have the same order as in var_par_list
    #    - p_best is a regular dictionary and thus loses the order of the initial p_best
    #    - fixed parameters defined through expressions are defined after the best-fit variable values
    p_best={}
    p_best.update(fixed_args['fixed_par_val'])
    for ipar,parname in enumerate(fixed_args['var_par_list']):
        p_best[parname]=med_par[ipar]  
        
    #Attribute value of variable parameters /non-variable parameters non defined via expressions directly to their names so that they can be identified in the expressions                
    if len(fixed_args['linked_par_expr'])>0:
        for ipar,parname in enumerate(fixed_args['var_par_list']):        
            exec(str(parname)+'='+str(p_best[parname]))
        for ipar,parname in enumerate(fixed_args['fixed_par_val_noexp_list']):        
            exec(str(parname)+'='+str(p_best[parname]))

    #Update fixed parameters with associated expression
    for par in fixed_args['linked_par_expr']:
        p_best[par]=eval(fixed_args['linked_par_expr'][par])    

    #----------------------------------------------------        
       
    #Compute the q-th percentiles of the 1D distribution, ie the value X when distributions are ordered where q/100 points are contained between the minimum value and X
    #    - eg the 50-th percentile gives the median
    #    - we define error bars at x sigma by the interval on either side of the median that contain 0.5*frac(x) of the total distribution
    #           frac(1 - 2 - 3 sigma) = 68.26894921370859, 95.4499736103642, 99.7300204 %
    #      the lower/upper boundaries of the x-sigma interval are thus given by the (50 +- 0.5*frac(x))-th percentiles  
    #           percentiles for 1 sigma: (15.865525393145703, 84.1344746068543)       
    #           percentiles for 2 sigma: (2.2750131948179018, 97.7249868051821)
    #           percentiles for 3 sigma: (0.1349897999999996, 99.8650102)
    sig_par_val={}
    if calc_quant:
        sig_par_err={'1s':np.zeros([2,n_par]),'2s':np.zeros([2,n_par]),'3s':np.zeros([2,n_par])}
        sig_perc={'1s':[15.865525393145703, 84.1344746068543],'2s':[2.2750131948179018, 97.7249868051821],'3s':[0.1349897999999996, 99.8650102]}
        for sig in ['1s','2s','3s']:
            #Lower/upper boundaries at x sigma (2 times n params )
            sig_par_val[sig]=np.percentile(merged_chain,sig_perc[sig], axis=0) 
            #Corresponding errors
            sig_par_err[sig][0,:]=med_par-sig_par_val[sig][0,:]
            sig_par_err[sig][1,:]=sig_par_val[sig][1,:]-med_par
    else:sig_par_err={}

    #----------------------------------------------------        
   
    #Lower/upper limits for PDFs bounded by the parameter space
    if (len(fit_dic['conf_limits'])>0):     

        #Process model parameters
        for ipar,parname in enumerate(fixed_args['var_par_list']):
            if parname in fit_dic['conf_limits']:
                fit_dic['conf_limits'][parname]['limits']={}
                
                #Process limits
                for lev in fit_dic['conf_limits'][parname]['level']:
                    if lev=='1s':
                        frac_search=0.6826894921370859
                    elif lev=='2s':
                        frac_search=0.954499736103642
                    elif lev=='3s':
                        frac_search=0.997300204                  
                                        
                    #Upper limits
                    if fit_dic['conf_limits'][parname]['type']=='upper':
                        lim_loc=np.quantile(merged_chain[:,ipar], frac_search) 
    
                    #Lower limits
                    elif fit_dic['conf_limits'][parname]['type']=='lower':
                        lim_loc=np.quantile(merged_chain[:,ipar], 1.-frac_search) 
                        
                    #Save
                    fit_dic['conf_limits'][parname]['limits'][lev] =  lev+' '+fit_dic['conf_limits'][parname]['type']+' lim : '+"{0:.8e}".format(lim_loc)                
                

    #----------------------------------------------------  
          
    #Calculate HDI intervals for the chosen confidence level
    #    - the x% highest density interval (HDI) contains x% of the distribution mass such that no point outside the interval has a higher density 
    # than any point within it. The HDI are equivalent to the range of values that encompass x% of the distribution on each side of the 
    # median when the distribution is Gaussian. By construction, the mode of the distribution is inside the HDI.
    if (fit_dic['HDI'] is not None):         

        #Fractions associated with each confidence level
        frac_search = {'1s':0.6826894921370859,
                       '2s':0.954499736103642,
                       '3s':0.997300204}[fit_dic['HDI']]

        #Process variable parameters
        HDI_interv=[ [] for i in range(n_par) ]
        HDI_interv_txt=np.zeros(n_par,dtype='U100')   #np.zeros(n_par,dtype='S100')
        HDI_sig_txt_par=np.zeros(n_par,dtype='U100')   #np.zeros(n_par,dtype='S100')
        HDI_frac=np.zeros(n_par)
        for ipar,parname in enumerate(fixed_args['var_par_list']):
            
            #Number or resolution of bins in histogram
            nbins_par=None if (('HDI_nbins' not in fit_dic) or (parname not in fit_dic['HDI_nbins'])) else fit_dic['HDI_nbins'][parname]
            dbins_par=None if (('HDI_dbins' not in fit_dic) or (parname not in fit_dic['HDI_dbins'])) else fit_dic['HDI_dbins'][parname]
            bw_fact=None if (('HDI_bwf' not in fit_dic) or (parname not in fit_dic['HDI_bwf'])) else fit_dic['HDI_bwf'][parname]
            
            #HDI intervals
            HDI_interv_txt[ipar],HDI_frac[ipar],HDI_sig_txt_par[ipar]=MCMC_HDI(merged_chain[:,ipar],nbins_par,dbins_par,bw_fact,frac_search,HDI_interv[ipar],HDI_interv_txt[ipar],HDI_sig_txt_par[ipar],med_par[ipar])
            
        #Convert into array
        HDI_interv=np.array(HDI_interv,dtype=object)   
          
    else:
        HDI_interv=None
        HDI_interv_txt=None
        HDI_sig_txt_par = None
            
    #----------------------------------------------------        

    #Print results
    if verbose or print_par:
        print(verb_shift+'-------------------------------') 
        print(verb_shift+'> Results')       
    if verbose:    
        print(verb_shift+"  Chain covariance: "+str(np.cov(np.transpose(merged_chain))))
        print(verb_shift+"  Chain coefficient correlations: "+str(np.corrcoef(np.transpose(merged_chain))))
    if print_par:             
        for ipar,(parname,parunit) in enumerate(zip(fixed_args['var_par_list'],fixed_args['var_par_units'])):
            if ipar>0:print(verb_shift+'-------------------------------') 
            print(verb_shift+'  Parameter '+parname+' ['+parunit+']')
            print(verb_shift+'    med : '+"{0:.3e}".format(med_par[ipar]))            
            if fit_dic['HDI'] is not None:
                print(verb_shift+'    HDI    '+fit_dic['HDI']+' err : '+str(HDI_sig_txt_par[ipar]))
                print(verb_shift+'              int : '+str(HDI_interv_txt[ipar]))
            else:
                for sig in fit_dic['sig_list']:
                    print(verb_shift+'    quant. '+sig+' err : -'+"{0:.3e}".format(sig_par_err[sig][0,ipar])+' +'+"{0:.3e}".format(sig_par_err[sig][1,ipar]))  
                    print(verb_shift+'              int : ['+"{0:.3e}".format(sig_par_val[sig][0,ipar])+' ; '+"{0:.3e}".format(sig_par_val[sig][1,ipar])+']')  
            if parname in fit_dic['conf_limits']:
                for lev in fit_dic['conf_limits'][parname]['level']: 
                    print(verb_shift+'    '+fit_dic['conf_limits'][parname]['limits'][lev] )

    return p_best,med_par,sig_par_val,sig_par_err,HDI_interv,HDI_interv_txt,HDI_sig_txt_par  
    
    

   
def postMCMCwrapper_1(fit_dic,fixed_args,walker_chains,nthreads,par_names,verbose=True,verb_shift=''):    
    r"""**MCMC post-proc: raw chains**

    Process and analyze MCMC chains of original parameters.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """
    #Burn-in
    fit_dic['nburn']=int(fit_dic['nburn'])    

    #Remove burn-in steps from chains of variable parameters
    #    - walker_chains is of shape (nwalkers, nsteps, n_free)
    n_free = (walker_chains.shape)[2]
    unburnt_chains = walker_chains[:, :, :]	
    burnt_chains = walker_chains[:, fit_dic['nburn']:, :]     

    #Number of post burn-in points in each chain
    fit_dic['nsteps_pb_walk']=fit_dic['nsteps']-fit_dic['nburn'] 

    #Automatic exclusion of chains
    if fit_dic['exclu_walk_autom'] is not None:

        #Merged chain
        merged_chain = burnt_chains.reshape((-1, n_free))  #(nsteps-nburn x nfree)	  
        
        #Median values for all parameters
        med_par=np.median(merged_chain, axis=0)

        #Lower/upper boundaries at 1 sigma and at threshold * (1 sigma) for all parameters
        sig1_par_val=np.percentile(merged_chain,[15.865525393145703, 84.1344746068543], axis=0)      
        low_thresh_par_val = med_par - fit_dic['exclu_walk_autom']*(med_par-sig1_par_val[0,:])
        high_thresh_par_val = med_par + fit_dic['exclu_walk_autom']*(sig1_par_val[1,:]-med_par)

        #Median value of all chains for all parameters
        #    - dimension (nwalkers, n_free)
        med_par_chain = np.median(burnt_chains, axis=1)
        
        #Condition to remove chain
        #    - a chain is removed if its median is beyond the threshold for at least one parameter
        #    - condition table has dimension (nwalkers, n_free), and is kept only if fullfill conditions for all parameters             
        keep_chain = np.all((med_par_chain>=low_thresh_par_val[None,:]) & (med_par_chain<=high_thresh_par_val[None,:]),axis=1)  
 
    else:    
        keep_chain = np.repeat(True,fit_dic['nwalkers'])
        low_thresh_par_val,high_thresh_par_val=None,None

    #Plot burnt chains
    if (not os_system.path.exists(fit_dic['save_dir'])):os_system.makedirs(fit_dic['save_dir'])
    if (fit_dic['save_MCMC_chains']!=''):
        MCMC_plot_chains(fit_dic['save_MCMC_chains'],fit_dic['save_dir'],fixed_args['var_par_list'],fixed_args['var_par_names'],walker_chains,burnt_chains,fit_dic['nsteps'],fit_dic['nsteps_pb_walk'],
                    fit_dic['nwalkers'],fit_dic['nburn'],keep_chain,low_thresh_par_val,high_thresh_par_val,fit_dic['exclu_walk_autom'],verbose=verbose,verb_shift=verb_shift)
     
    #Remove chains if required
    if (False in keep_chain):
        unburnt_chains = unburnt_chains[keep_chain]
        burnt_chains = burnt_chains[keep_chain]
        fit_dic['nwalkers']=np.sum(keep_chain)  
        
    #Merge chains
    #    - we reshape into (nwalkers*(nsteps-nburn) , n_free)        
    merged_chain = burnt_chains.reshape((-1, n_free))   

    #Manual exclusion of samples
    if len(fit_dic['exclu_samp'])>0:
        cond_keep = False
        for par_loc in fit_dic['exclu_samp']:
            ipar_loc = np_where1D(fixed_args['var_par_list']==par_loc)
            if len(ipar_loc)>0:
                for bd_int in fit_dic['exclu_samp'][par_loc]:
                    cond_keep |= (merged_chain[:,ipar_loc[0]]>=bd_int[0]) & (merged_chain[:,ipar_loc[0]]<=bd_int[1]) 
        merged_chain = merged_chain[cond_keep]
 
    #Number of points remaining in the merged chain     
    fit_dic['nsteps_final_merged']=len(merged_chain[:,0])
   
    #Saving MCMC information
    if fit_dic['save_outputs']:
        np.savetxt(fit_dic['file_save'],[['-----------------']],fmt=['%s'])
        np.savetxt(fit_dic['file_save'],[['MCMC run']],fmt=['%s'])      
        np.savetxt(fit_dic['file_save'],[['-----------------']],fmt=['%s']) 
        np.savetxt(fit_dic['file_save'],[['nwalkers : ',str(fit_dic['nwalkers'])]],delimiter='\t',fmt=['%s','%s']) 
        np.savetxt(fit_dic['file_save'],[['nburn : ',str(fit_dic['nburn'])]],delimiter='\t',fmt=['%s','%s']) 
        np.savetxt(fit_dic['file_save'],[['nsteps (initial, per walker) : ',str(fit_dic['nsteps'])]],delimiter='\t',fmt=['%s','%s']) 
        np.savetxt(fit_dic['file_save'],[['nsteps (final, all walkers) : ',str(fit_dic['nsteps_final_merged'])]],delimiter='\t',fmt=['%s','%s']) 
        np.savetxt(fit_dic['file_save'],[['']],fmt=['%s']) 
 
    #Calculate correlation lengths for each parameter
    #    - we use the post burn-in chain and start the calculation from first pixel
    #    - this is a time-consuming operation
    if fit_dic['thin_MCMC']:
        print(' > Thinning chains') 

        #Longueurs de correlation
        corr_length=MCMC_corr_length(fit_dic,fit_dic['max_corr_length'],nthreads,fixed_args['var_par_list'],merged_chain,0,fit_dic['nsteps_final_merged'],verbose=True)        
 
        #Thin the chain by keeping only one point every maximum correlation length
        #    - in this way the merged_chain over all pararameters are uncorrelated
        fit_dic['nsteps_final_merged'],merged_chain=MCMC_thin_chains(corr_length,merged_chain)

    #Best-fit parameters for model calculations
    p_final,fit_dic['med_parfinal'],fit_dic['sig_parfinal_val'],fit_dic['sig_parfinal_err'],fit_dic['HDI_interv'],fit_dic['HDI_interv_txt'],fit_dic['HDI_sig_txt']=MCMC_estimates(merged_chain,fixed_args,fit_dic,verbose=verbose,print_par=fit_dic['print_par'],calc_quant=fit_dic['calc_quant'],verb_shift=verb_shift)

    #Plot merged chains for MCMC parameters
    if (fit_dic['save_MCMC_chains']!=''):
        MCMC_plot_merged_chains(fit_dic['save_MCMC_chains'],fit_dic['save_dir'],fixed_args['var_par_list'],fixed_args['var_par_names'],merged_chain,fit_dic['nsteps_final_merged'],verbose=verbose,verb_shift=verb_shift)
 
    #Save 1-sigma and envelope samples for plot in ANTARESS_main
    if fit_dic['calc_envMCMC'] or fit_dic['calc_sampMCMC']:
        par_sample_sig1,par_sample=MCMC_retrieve_sample(fixed_args,fixed_args['fix_par_list'],fixed_args['exp_par_list'],fixed_args['iexp_par_list'],fixed_args['ifix_par_list'],par_names,fixed_args['fixed_par_val'],fit_dic['calc_envMCMC'],merged_chain,fit_dic['merit']['n_free'],fit_dic['nsteps_final_merged'],
                                                        p_final,fixed_args['var_par_list'],fixed_args['ivar_par_list'],fit_dic['calc_sampMCMC'],fixed_args['linked_par_expr'],fit_dic,fit_dic['st_samp'],fit_dic['end_samp'],fit_dic['n_samp'])
        if fit_dic['calc_envMCMC']:print('Saved ',len(par_sample_sig1),'1-sigma samples')
        if fit_dic['calc_sampMCMC']:print('Saved ',len(par_sample),' random samples')
        np.savez(fit_dic['save_dir']+'Sample_dics',par_sample_sig1=[par_sample_sig1],par_sample=[par_sample])    
    else:
        par_sample_sig1=None
        par_sample=None
    
    return p_final,merged_chain,par_sample_sig1,par_sample
    
    
def postMCMCwrapper_2(fit_dic,fixed_args,merged_chain):
    r"""**MCMC post-proc: modified chains**

    Analyze MCMC chains of modified parameters.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """    
    #Parameter estimates and confidence interval
    #    - define confidence levels to be printed
    #    - use the modified chains that can contain different parameters than those used in the model
    #      the best-fit model will not be modified, but the PDFs, best-fit parameters and associated uncertainties will be derived from the modified chains
    p_final,fit_dic['med_parfinal'],fit_dic['sig_parfinal_val'],fit_dic['sig_parfinal_err'],fit_dic['HDI_interv'],fit_dic['HDI_interv_txt'],fit_dic['HDI_sig_txt']=MCMC_estimates(merged_chain,fixed_args,fit_dic,verbose=False,print_par=False,calc_quant=fit_dic['calc_quant'],verb_shift=fit_dic['verb_shift'])

    #Save merged chains for derived parameters and various estimates
    data_save = {'merged_chain':merged_chain,'HDI_interv':fit_dic['HDI_interv'],'sig_parfinal_val':fit_dic['sig_parfinal_val']['1s'],'var_par_list':fixed_args['var_par_list'],'var_par_names':fixed_args['var_par_names'],'med_parfinal':fit_dic['med_parfinal']}
    np.savez(fit_dic['save_dir']+'merged_deriv_chains_walk'+str(fit_dic['nwalkers'])+'_steps'+str(fit_dic['nsteps'])+fit_dic['run_name'],data=data_save,allow_pickle=True)

    #Plot correlation diagram for all param    
    if fit_dic['save_MCMC_corner']!='':
        corner_options=fit_dic['corner_options'] if 'corner_options' in fit_dic else {}      

        #Reduce to required parameters
        var_par_list = np.array(fixed_args['var_par_list'])
        var_par_names = np.array(fixed_args['var_par_names'])
        if 'plot_par' in fit_dic['corner_options']:
            ikept = []
            for par_loc in fit_dic['corner_options']['plot_par']:
                ipar = np_where1D(var_par_list==par_loc)
                if len(ipar)>0:ikept+=[ipar[0]]
                else:stop('Parameter '+par_loc+' was not fitted.')
            if len(ikept)==0:stop('No parameters kept in corner plot')
            var_par_list = var_par_list[ikept]
            var_par_names = var_par_names[ikept] 
            merged_chain = merged_chain[:,ikept] 
            fit_dic['med_parfinal'] = fit_dic['med_parfinal'][ikept] 
            fit_dic['HDI_interv'] = fit_dic['HDI_interv'][ikept] 
            
        #Remove constant parameters
        for par_loc in var_par_list:
            ipar = np_where1D(var_par_list==par_loc)[0]
            if np.min(merged_chain[:,ipar])==np.max(merged_chain[:,ipar]):
                var_par_list = np.delete(var_par_list,ipar)
                var_par_names = np.delete(var_par_names,ipar)
                merged_chain = np.delete(merged_chain,ipar,axis=1) 
                fit_dic['med_parfinal'] = np.delete(fit_dic['med_parfinal'],ipar)
                fit_dic['HDI_interv'] = np.delete( fit_dic['HDI_interv'],ipar,axis=0) 

        #Default options    
        bins_1D_par=20 if 'bins_1D_par' not in corner_options else corner_options['bins_1D_par']
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
       
        #Plot
        MCMC_corner_plot(fit_dic['save_MCMC_corner'],fit_dic['save_dir'],merged_chain,fit_dic['HDI_interv'],
                         labels_raw = var_par_list,
                         labels=var_par_names,
                         truths=best_val,
                         bins_1D_par=bins_1D_par,
                         bins_2D_par=bins_2D_par,
                         range_par=range_par,
                         major_int=major_int,
                         minor_int=minor_int,
                         levels=(0.39346934028,0.86466471),
                         color_levels=color_levels,
                         smooth2D=smooth2D, 
                         plot_HDI=plot_HDI ,
                         plot1s_1D=plot1s_1D   ,
                         label_kwargs=label_kwargs,tick_kwargs=tick_kwargs                                       
                         )    

    return p_final   
    
    
    

    

  
  
  
  
  

##################################################################################################
#%%%% MCMC plots
##################################################################################################   
 
def MCMC_plot_chains(save_mode,save_dir_MCMC,var_par_list,var_par_names,chain,burnt_chains,nsteps,nsteps_pb_walk,nwalkers,nburn,keep_chain,low_thresh_par_val,high_thresh_par_val,exclu_walk_autom,verbose=True,verb_shift=''):
    r"""**MCMC post-proc: walker chains plot**

    Plots the chains for each fitted parameter over all walkers. 
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """
    if verbose:
        print(verb_shift+'-----------------------------------')
        print(verb_shift+'> Plotting walker chains')

    #Font size
    font_size=14
    
    #Taille plot
    margins=[0.15,0.15,0.95,0.8] 
        
    #Linewidth
    lw_plot=0.1
          
    #----------------------------------------------------------------
    #Loop on parameters
    med_par=np.median(burnt_chains, axis=(0,1))
    for ipar,(parname,partxt) in enumerate(zip(var_par_list,var_par_names)): 
        plt.ioff()        
        plt.figure(figsize=(10, 6))
       
        #Median value       
        plt.plot([0,nsteps],[med_par[ipar],med_par[ipar]],color='black',linestyle='--',zorder=10)               

        #Chains with burn-in phase, and removed chains
        for iwalk,keep_chain_loc in enumerate(keep_chain):
            if keep_chain_loc:
                x_tab=range(nburn)
                plt.plot(x_tab,chain[iwalk,x_tab,ipar],color='red',linestyle='-',lw=lw_plot,zorder=0)                
                x_tab=nburn+np.arange(nsteps_pb_walk,dtype=int)
                plt.plot(x_tab,chain[iwalk,x_tab,ipar],color='dodgerblue',linestyle='-',lw=lw_plot,zorder=0)                           
            else:
                plt.plot(np.arange(nsteps,dtype=int),chain[iwalk,:,ipar],color='red',linestyle='-',lw=lw_plot,zorder=0) 

        #Automatic exclusion limits
        if exclu_walk_autom is not None:
            plt.plot([0,nsteps],[low_thresh_par_val[ipar],low_thresh_par_val[ipar]],color='black',linestyle=':',zorder=10)             
            plt.plot([0,nsteps],[high_thresh_par_val[ipar],high_thresh_par_val[ipar]],color='black',linestyle=':',zorder=10) 
        
        #Plot frame  
        plt.title('Chain for param '+partxt)
        y_min = np.min(chain[:,:,ipar])
        y_max = np.max(chain[:,:,ipar])
        dy_range = y_max-y_min
        y_range = [y_min-0.05*dy_range,y_max+0.05*dy_range]    
        ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)
        custom_axis(plt,position=margins,
                    y_range=y_range,dir_y='out', 
                    ymajor_int=ymajor_int,yminor_int=yminor_int,ymajor_form=ymajor_form,
                    x_title='Steps',y_title=partxt,
                    font_size=font_size,xfont_size=font_size,yfont_size=font_size)

        plt.savefig(save_dir_MCMC+'/Chain_'+parname+'.'+save_mode) 
        plt.close()

    return None  



def MCMC_plot_merged_chains(save_mode,save_dir_MCMC,var_par_list,var_par_names,merged_chain,nsteps_final_merged,verbose=True,verb_shift=''):
    r"""**MCMC post-proc: merged chains plot**

    Plots the cleaned, merged chains from all workers for each fitted parameter.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """
    if verbose:
        print(verb_shift+'-----------------------------------')
        print(verb_shift+'> Plotting merged chains')

    #Font size
    font_size=14
    
    #Taille plot
    margins=[0.15,0.15,0.95,0.8] 
        
    #Linewidth
    lw_plot=0.1
          
    #----------------------------------------------------------------
    #Loop on parameters
    med_par=np.median(merged_chain, axis=0)
    for ipar,parname in enumerate(var_par_list):  
        plt.ioff()
        plt.figure(figsize=(10, 6))
       
        #Median value
        x_tab=[0,nsteps_final_merged]
        plt.plot(x_tab,[med_par[ipar],med_par[ipar]],color='black',linestyle='--',zorder=10)               

        #Post-burn-in merged chain
        x_tab=range(int(nsteps_final_merged))
        plt.plot(x_tab,merged_chain[x_tab,ipar],color='dodgerblue',linestyle='-',lw=lw_plot,zorder=0)

        #Plot frame  
        parname_txt=parname.replace('_','-')
        plt.title('Merged chain for param '+parname_txt)
        y_min = np.min(merged_chain[x_tab,ipar])
        y_max = np.max(merged_chain[x_tab,ipar])
        dy_range = y_max-y_min
        y_range = [y_min-0.05*dy_range,y_max+0.05*dy_range]    
        ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)
        custom_axis(plt,position=margins,
                    # x_range=x_range,
                    y_range=y_range,dir_y='out', 
                    ymajor_int=ymajor_int,yminor_int=yminor_int,ymajor_form=ymajor_form,
                    #		    xmajor_int=1.,xminor_int=0.5,
                    #xmajor_form='%.1f',ymajor_form='%.3f',
                    x_title='Steps',y_title=parname_txt,
                    font_size=font_size,xfont_size=font_size,yfont_size=font_size)
        plt.savefig(save_dir_MCMC+'/Merged_chain_'+parname+'.'+save_mode) 
        plt.close()

    return None  

    

    
def MCMC_corner_plot(save_mode,save_dir_MCMC,xs,HDI_interv, 
                     labels_raw = None,labels=None,truths=None,truth_color="#4682b4",bins_1D_par=20,bins_2D_par=20,quantiles=[0.15865525393145703,0.841344746068543],
                     plot1s_1D=True,plot_HDI=False,color_1D_quant='darkorange',color_1D_HDI='green',levels=(0.39346934028,0.86466471,0.988891003),  
                     plot_contours = True,color_levels='black',use_math_text=True,range_par=None,smooth1d=None,smooth2D=None,weights=None, color="k",  
                     label_kwargs=None,tick_kwargs=None,show_titles=False,title_fmt=".2f",title_kwargs=None,scale_hist=False, 
                     verbose=False, fig=None,major_int=None,minor_int=None,max_n_ticks=5, top_ticks=False, hist_kwargs=None, **hist2d_kwargs):
    r"""**MCMC post-proc: corner plot**

    Plots correlation diagram, showing the projections of a data set in a multi-dimensional space. kwargs are passed to MCMC_corner_plot_hist2d() or used for `matplotlib` styling.
      
    Args:
        save_mode (str): extension of the figure (png, pdf, jpg) 
        save_dir_MCMC (str): path of the directory in which the figure is saved
        xs (array, float): Samples. This should be a 1- or 2-dimensional array with dimensions [nsamples, ndim]. 
                           For a 1-D array this results in a simple histogram. 
                           For a 2-D array, the zeroth axis is the list of samples and the next axis are the dimensions of the space.
        HDI_interv (array, object): HDI intervals for each parameter

        labels_raw (list, str): names of variable parameters, as used in the model
        labels (list, str): names of variable fitted parameters, as used in the plots. If ``xs`` is a ``pandas.DataFrame``, labels will default to column names.
        truths (array, float): a list of reference values to indicate on the plots.  Individual values can be omitted by using ``None``. Typically the best-fit values
        truth_color (str): a ``matplotlib`` style color for the ``truths`` makers
        bins_1D_par (int or dict): number of bins in the 1D histogram ranges (default 20)
                                   if int, defines bins for all parameters
                                   if dict, defines bins for parameters set as keys  
        bins_2D_par (int or dict): same as `bins_1D_par` for the 2D histograms
        quantiles (list, float): x-sigma confidence intervals around the median for the 1D histogram
                                 default 1 and 2-sigma
        plot1s_1D (bool): condition to plot 1D confidence intervals  
        plot_HDI (bool): condition to plot HDI (if `HDI_interv`) is defined
        color_1D_HDI (str): ``matplotlib`` style color for quantiles.
        color_1D_quant (str): ``matplotlib`` style color for HDIs.
        levels (tuple): sigma contours for the 2D histograms, defined as (1-np.exp(-0.5*(x sigma)**2.),)     
                        default 1, 2, and 3 sigma
        plot_contours (bool): draw contours for dense regions of the 2D histograms
        color_levels (str): ``matplotlib`` style color for `plot_contours`
        use_math_text (bool): if true, then axis tick labels for very large or small exponents will be displayed as powers of 10 rather than using `e`
        range_par (list): each element is either a length 2 tuple containing lower and upper bounds, or a float in (0., 1.) giving the fraction of samples to include in bounds, e.g.,
                          [(0.,10.), (1.,5), 0.999, etc.]. If a fraction, the bounds are chosen to be equal-tailed.
        smooth1d, smooth2D (float or list of float): standard deviation for Gaussian kernel passed to `scipy.ndimage.gaussian_filter` to smooth the 2-D and 1-D histograms respectively. 
                                                     If `None` (default), no smoothing is applied. If list, each element is associated with a parameter 
        weights (array, float): weight of each sample. If `None` (default), samples are given equal weight.
        color (str): A ``matplotlib`` style color for all histograms.
        label_kwargs (dict): any extra keyword arguments to send to the `set_xlabel` and `set_ylabel` methods.
        tick_kwargs (dict): any extra keyword arguments to send to the `tick_params` methods.    
        show_titles (bool): displays a title above each 1-D histogram showing the 0.5 quantile with the upper and lower errors supplied by the quantiles argument.
        title_fmt (str): the format string for the quantiles given in titles. If you explicitly set ``show_titles=True`` and ``title_fmt=None``, the labels will be
                         shown as the titles. (default: ``.2f``)
        title_kwargs (dict): any extra keyword arguments to send to the `set_title` command
        scale_hist (bool): scale the 1-D histograms in such a way that the zero line is visible
        verbose (bool): if true, print quantile and HDI values
        fig (matplotlib.Figure): overplot onto the provided figure object
        major_int, minor_int (list): list of major/minor tick intervals for each parameter.
                                     Some elements can be set to None, in which case `max_n_ticks` is used 
        max_n_ticks (int): maximum number of ticks to try to use
        top_ticks (bool): if true, label the top ticks of each axis    
        hist_kwargs (dict): any extra keyword arguments to send to the 1-D histogram plots.
        **hist2d_kwargs: any remaining keyword arguments are sent to `corner.hist2d` to generate the 2-D histogram plots
    
    Returns:
        None
    
    """



    if verbose:
        print(' -----------------------------------')
        print(' > Plot MCMC corr. diagram')

    plt.ioff() 

    if (quantiles is None) or (plot1s_1D==False):
        quantiles = []
    if HDI_interv is None:
        plot_HDI = False
    if title_kwargs is None:
        title_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()
    if tick_kwargs is None:
        tick_kwargs = dict()        
        
        

    # Try filling in labels from pandas.DataFrame columns.
    if labels is None:
        try:
            labels = xs.columns
        except AttributeError:
            pass

    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], "I don't believe that you want more " \
                                       "dimensions than samples!"
    npar=xs.shape[1]                                       

    # Parse the weight array.
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("Weights must be 1-D")
        if xs.shape[1] != weights.shape[0]:
            raise ValueError("Lengths of weights must match number of samples")

    # Parse the parameter ranges.
    if range_par is None:
        if "extents" in hist2d_kwargs:
            logging.warn("Deprecated keyword argument 'extents'. "
                         "Use 'range_par' instead.")
            range_par = hist2d_kwargs.pop("extents")
        else:
            range_par = [[x.min(), x.max()] for x in xs]
            # Check for parameters that never change.
            m = np.array([e[0] == e[1] for e in range_par], dtype=bool)
            if np.any(m):
                raise ValueError(("It looks like the parameter(s) in "
                                  "column(s) {0} have no dynamic range. "
                                  "Please provide a `range_par` argument.")
                                 .format(", ".join(map(
                                     "{0}".format, np.arange(len(m))[m]))))

    elif type(range_par)==list:

        # If any of the extents are percentiles, convert them to ranges.
        # Also make sure it's a normal list.
        range_par = list(range_par)
        for i, _ in enumerate(range_par):
            try:
                emin, emax = range_par[i]
            except TypeError:
                q = [0.5 - 0.5*range_par[i], 0.5 + 0.5*range_par[i]]
                range_par[i] = quantile(xs[i], q, weights=weights)

    elif type(range_par)==dict:
        if labels is None:raise ValueError("Parameters must be named to be set in `range_par`")
        range_par_in = deepcopy(range_par)
        range_par = [[] for i in range(xs.shape[0])]
        for par in range_par_in:
            idx_par = np_where1D(labels==par)
            if len(idx_par)>0:range_par[idx_par[0]] = range_par_in[par]
        for i,x in enumerate(xs):
            if len(range_par[i])==0:range_par[i] = [x.min(), x.max()]

    if len(range_par) != xs.shape[0]:
        raise ValueError("Dimension mismatch between samples and range_par")

    #Define bin sizes
    #    - if given as number, converted into array of relevant dimension
    if type(bins_1D_par) in [int,float]:bins_1D_par=np.repeat(bins_1D_par,npar).astype(int)
    elif type(bins_1D_par)==dict:
        if labels_raw is None:raise ValueError("Parameters must be named to be set in `range_par`")
        bins_1D_par_in = deepcopy(bins_1D_par)
        bins_1D_par = np.repeat(20,npar).astype(int)
        for par in bins_1D_par_in:
            idx_par = np_where1D(labels_raw==par)
            if len(idx_par)>0:bins_1D_par[idx_par[0]] = bins_1D_par_in[par]
            else:stop('Parameter'+par+' not in fitted list.')
    if type(bins_2D_par) in [int,float]:bins_2D_par=np.repeat(bins_2D_par,npar).astype(int)
    elif type(bins_2D_par)==dict:
        if labels_raw is None:raise ValueError("Parameters must be named to be set in `range_par`")
        bins_2D_par_in = deepcopy(bins_2D_par)
        bins_2D_par = np.repeat(20,npar).astype(int)
        for par in bins_2D_par_in:
            idx_par = np_where1D(labels_raw==par)
            if len(idx_par)>0:bins_2D_par[idx_par[0]] = bins_2D_par_in[par] 
            else:stop('Parameter'+par+' not in fitted list.')

    #Ticks interval undefined
    if major_int is None:major_int=np.repeat(None,npar)
    if minor_int is None:minor_int=np.repeat(None,npar)
        
    #Smoothing for 2D histograms
    if smooth2D is None:smooth2D=np.repeat(None,npar)

    # Some magic numbers for pretty axis layout.
    K = len(xs)
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim

    # Create a new figure if one wasn't provided.
    if fig is None:
        fig, axes = plt.subplots(K, K, figsize=(dim, dim))
    else:
        try:
            axes = np.array(fig.axes).reshape((K, K))
        except:
            raise ValueError("Provided figure has {0} axes, but data has "
                             "dimensions K={1}".format(len(fig.axes), K))

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    # Set up the default histogram keywords.
    if hist_kwargs is None:
        hist_kwargs = dict()
    hist_kwargs["color"] = hist_kwargs.get("color", color)
    if smooth1d is None:
        hist_kwargs["histtype"] = hist_kwargs.get("histtype", "step")

    for i, x in enumerate(xs):
        # Deal with masked arrays.
        if hasattr(x, "compressed"):
            x = x.compressed()

        if np.shape(xs)[0] == 1:
            ax = axes
        else:
            ax = axes[i, i]
        # Plot the histograms.
        if smooth1d is None:
            n, _, _ = ax.hist(x, bins=bins_1D_par[i], weights=weights,
                              range=np.sort(range_par[i]), **hist_kwargs)
        else:
            if gaussian_filter is None:
                raise ImportError("Please install scipy for smoothing")
            n, b = np.histogram(x, bins=bins_1D_par[i], weights=weights,
                                range_par=np.sort(range_par[i]))
            n = gaussian_filter(n, smooth1d)
            x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
            y0 = np.array(list(zip(n, n))).flatten()
            ax.plot(x0, y0, **hist_kwargs)

        if truths is not None and truths[i] is not None:
            ax.axvline(truths[i], color=truth_color)


        #Replace quantiles with HDI 
        if (plot_HDI==True):
            if verbose:
                print("HDI:") 
                print(['['+str(HDI_sub[0])+';'+str(HDI_sub[1])+']' for HDI_sub in HDI_interv[i]])  
                
            #Process each HDI interval
            for HDI_sub in HDI_interv[i]:
                ax.axvline(HDI_sub[0], ls="dashed", color=color_1D_HDI) 
                ax.axvline(HDI_sub[1], ls="dashed", color=color_1D_HDI)

        #Plot quantiles if wanted.
        if len(quantiles) > 0:
            qvalues = quantile(x, quantiles, weights=weights)
            for q in qvalues:
                ax.axvline(q, ls="dashed", color=color_1D_quant) 

            if verbose:
                print("Quantiles:")
                print([item for item in zip(quantiles, qvalues)])

        if show_titles:
            title = None
            if title_fmt is not None:
                # Compute the quantiles for the title. This might redo
                # unneeded computation but who cares.
                q_16, q_50, q_84 = quantile(x, [0.16, 0.5, 0.84],
                                            weights=weights)
                q_m, q_p = q_50-q_16, q_84-q_50

                # Format the quantile display.
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))

                # Add in the column name if it's given.
                if labels is not None:
                    title = "{0} = {1}".format(labels[i], title)

            elif labels is not None:
                title = "{0}".format(labels[i])

            if title is not None:
                ax.set_title(title, **title_kwargs)

        # Set up the axes.
        ax.set_xlim(range_par[i])
        if scale_hist:
            maxn = np.max(n)
            ax.set_ylim(-0.1 * maxn, 1.1 * maxn)
        else:
            ax.set_ylim(0, 1.1 * np.max(n))
        ax.set_yticklabels([])
        
	    #Interval between ticks		
        if major_int[i] is None:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
        else:
            ax.xaxis.set_major_locator(MultipleLocator(major_int[i]))
        if minor_int[i] is not None:
            ax.xaxis.set_minor_locator(MultipleLocator(minor_int[i]))

        if i < K - 1:
            if top_ticks:
                ax.xaxis.set_ticks_position("top")
                [l.set_rotation(45) for l in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
        else:
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            if labels is not None:
                ax.set_xlabel(labels[i], **label_kwargs)
                ax.xaxis.set_label_coords(0.5, -0.3)
                ax.tick_params('x', **tick_kwargs)

            # use MathText for axes ticks
            ax.xaxis.set_major_formatter(
                ScalarFormatter(useMathText=use_math_text))

        for j, y in enumerate(xs):
            if np.shape(xs)[0] == 1:
                ax = axes
            else:
                ax = axes[i, j]
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                continue

            # Deal with masked arrays.
            if hasattr(y, "compressed"):
                y = y.compressed()

            #Local 2D histogram
            MCMC_corner_plot_hist2d(y, x, ax=ax, range_par=[range_par[j], range_par[i]], weights=weights,
                   color=color,levels=levels,color_levels=color_levels,rasterized=True, 
                   smooth=[smooth2D[j],smooth2D[i]], bins_2D=[bins_2D_par[j], bins_2D_par[i]],
                   **hist2d_kwargs)

            if truths is not None:
                if truths[i] is not None and truths[j] is not None:
                    ax.plot(truths[j], truths[i], "s", color=truth_color)
                if truths[j] is not None:
                    ax.axvline(truths[j], color=truth_color)
                if truths[i] is not None:
                    ax.axhline(truths[i], color=truth_color)

        	#Interval between major ticks		
            if major_int[j] is None:
                ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            else:
                ax.xaxis.set_major_locator(MultipleLocator(major_int[j]))
            if major_int[i] is None:
                ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            else:
                ax.yaxis.set_major_locator(MultipleLocator(major_int[i]))

        	#Interval between minor ticks	
            if minor_int[j] is not None:
                ax.xaxis.set_minor_locator(MultipleLocator(minor_int[j]))
            if minor_int[i] is not None:
                ax.yaxis.set_minor_locator(MultipleLocator(minor_int[i]))


            if i < K - 1:
                ax.set_xticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                if labels is not None:
                    ax.set_xlabel(labels[j], **label_kwargs)
                    ax.xaxis.set_label_coords(0.5, -0.3)
                    ax.tick_params('x', **tick_kwargs)

                # use MathText for axes ticks
                ax.xaxis.set_major_formatter(
                    ScalarFormatter(useMathText=use_math_text))

            if j > 0:
                ax.set_yticklabels([])
            else:
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                if labels is not None:
                    ax.set_ylabel(labels[i], **label_kwargs)
                    ax.yaxis.set_label_coords(-0.3, 0.5)
                    ax.tick_params('y', **tick_kwargs)

                # use MathText for axes ticks
                ax.yaxis.set_major_formatter(
                    ScalarFormatter(useMathText=use_math_text))

    plt.savefig(save_dir_MCMC+'/Corr_diag.'+save_mode) 
    plt.close()     

    return None
    
    
    
    





def MCMC_corner_plot_hist2d(x, y, bins_2D=[20,20], range_par=None, weights=None, levels=None, smooth=None,
           ax=None, color=None, plot_datapoints=True, plot_density=True,
           plot_contours=True, no_fill_contours=False, fill_contours=False,
           contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,color_levels='black',
           **kwargs):
    r"""**MCMC post-proc: 2-D histograms**
    
    Plots 2-D histograms of samples for correlation diagram

    Args:
        x (array, float): The samples.    
        y (array, float): The samples.    
        levels (array, float): The contour levels to draw.    
        ax (matplotlib.Axes): A axes instance on which to add the 2-D histogram.    
        plot_datapoints (bool): Draw the individual data points.    
        plot_density (bool): Draw the density colormap.    
        plot_contours (bool): Draw the contours.    
        no_fill_contours (bool): Add no filling at all to the contours (unlike setting ``fill_contours=False``, which still adds a white fill at the densest points).   
        fill_contours (bool): Fill the contours.    
        contour_kwargs (dict): Any additional keyword arguments to pass to the `contour` method.    
        contourf_kwargs (dict): Any additional keyword arguments to pass to the `contourf` method.
        data_kwargs (dict): Any additional keyword arguments to pass to the `plot` method when adding the individual data points.

    Returns:
        None

    """

    if ax is None:
        ax = plt.gca()

    # Set the default range based on the data range if not provided.
    if range_par is None:
        if "extent" in kwargs:
            logging.warn("Deprecated keyword argument 'extent'. "
                         "Use 'range_par' instead.")
            range_par = kwargs["extent"]
        else:
            range_par = [[x.min(), x.max()], [y.min(), y.max()]]

    # Set up the default plotting arguments.
    if color is None:
        color = "k"

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # This color map is used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels)+1)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins_2D,
                                 range=list(map(np.sort, range_par)),
                                 weights=weights)
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range. You could try using the "
                         "'range' argument.")

    if (smooth is not None) and (None not in smooth):
        if gaussian_filter is None:
            raise ImportError("Please install scipy for smoothing")
        H = gaussian_filter(H, smooth)

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        logging.warning("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([
        X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
        X1,
        X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
    ])
    Y2 = np.concatenate([
        Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
        Y1,
        Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
    ])

    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = dict()
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = dict()
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased",
                                                             False)
        ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]),
                    **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:

#        # This is the color map for the density plot, over-plotted to indicate the
#        # density of the points near the center.
#        density_cmap = LinearSegmentedColormap.from_list(
#            "density_cmap", [color, (1, 1, 1, 0)])        

        density_cmap = plt.get_cmap('gist_heat')         
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = dict()
        contour_kwargs["colors"] = contour_kwargs.get("colors", color_levels)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    ax.set_xlim(range_par[0])
    ax.set_ylim(range_par[1])    
    
    return None
    
    
 
    
