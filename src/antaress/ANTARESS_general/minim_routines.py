#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import emcee
import dynesty
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
import lmfit
from lmfit import minimize
from scipy import special
import arviz
from ..ANTARESS_plots.utils_plots import custom_axis,autom_tick_prop
from ..ANTARESS_general.utils import np_where1D,stop,npint,init_parallel_func,get_time
    


##################################################################################################
#%%% Formatting routines
##################################################################################################   

def up_var_par(args,par):
    r"""**Parameter conversion: parameter update**

    Update input parameter in list of variables. 

    Args:
        TBD

    Returns:
        TBD
        
    """   
    args['var_par_list']=np.append(args['var_par_list'],par)
    args['var_par_names']=np.append(args['var_par_names'],args['model_par_names'](par))  
    args['var_par_units']=np.append(args['var_par_units'],args['model_par_units'](par)) 
    return None


def par_formatting(p_start,model_prop,priors_prop,fit_dic,fixed_args,inst,vis):
    r"""**Parameter formatting: generic**

    Defines parameters in format relevant for fits and models, using input parameters.

    Args:
        TBD
    
    Returns:
        TBD
    
    """
 
    #Parameters are used for fitting   
    fit_used=False
    if isinstance(p_start,lmfit.parameter.Parameters):
        par_struct = True
        if (fit_dic['fit_mode']!='fixed'):
            fit_used=True
            fit_dic['uf_bd']={}            
            fit_dic['gauss_bd']={}
    else:par_struct = False

    #Process default / additional parameters
    fixed_args['varpar_priors']={}
    var_par_list_temp=[]
    for par in np.unique( list(p_start.keys()) + list(model_prop.keys())  ):  
        
        #Activate jitter if requested as parameter
        if par=='jitter':fixed_args['jitter'] = True

        #Overwrite default properties 
        #    - parameters can be set up as:
        # + model_prop : {par : value}
        # + model_prop : {par : { inst : value }}
        # + model_prop : {par : { inst : { vis : value }}}
        #    - default parameter settings are only overwritten under the condition that the parameter settings are set up for current instrument
        #      for example if 'par'  is processed for inst1 and inst2 but model_prop[par] = { inst1 : value }
        #      this is only relevant when instrument and visits are processed successively
        #      when a joint fit is performed over several instruments and visits, model_prop[par] has never a sub-structure in inst > vis (ie, all parameters are at the same level and 
        # their potential inst / vis dependance is defined through the parameter name itself as 'parname__ISinst__VSvis'), thus the routine is called with inst='' and vis='' so that properties are overwritten in any case 
        if (par in model_prop) and ((inst in par) or ('__IS' in par) or ((inst in model_prop[par]) and (vis in model_prop[par][inst]))):

            #Properties depend on instrument and/or visit or is common to all
            if par_struct:
                if (inst in model_prop[par]):
                    if (vis in model_prop[par][inst]):model_prop_par = model_prop[par][inst][vis]
                    else:model_prop_par = model_prop[par][inst]
                else:model_prop_par = model_prop[par] 
            else:model_prop_par = model_prop[par] 
            
            #Property value
            if par_struct:
                par_val = model_prop_par['guess'] 
                
                #Value linked to other parameter
                if ('expr' in model_prop_par):par_expr = model_prop_par['expr']  
                else:par_expr=None
            else:
                par_val = model_prop_par
            
            #Fitted/fixed property
            if fit_used:par_vary = model_prop[par]['vary']  
            else:par_vary = False 
        
            #Overwrite existing property fields or add property
            if par_struct:
                if (par in p_start):
                    p_start[par].value  = par_val
                    p_start[par].vary  = par_vary     
                else:  
                    p_start.add(par, par_val  ,par_vary ,None , None, None)  
                p_start[par].expr  = par_expr   
            else:p_start[par] = par_val

        #Variable parameter
        if fit_used and (par in p_start) and p_start[par].vary:
            var_par_list_temp+=[par]
            
            #Chi2 fit
            if (fit_dic['fit_mode']=='chi2'):
                if (par in priors_prop) and (priors_prop[par]['mod']!='uf'):stop('Prior error: only uniform priors can be used with chi2 minimization')
                
                #Input priors
                #    - default priors have been set at the initialization of p_start
                if (par in priors_prop):
                    if 'prior_check' in fixed_args:fixed_args['prior_check'](par,priors_prop[par],p_start,fixed_args)
                    p_start[par].min = priors_prop[par]['low']
                    p_start[par].max = priors_prop[par]['high']

                #Change guess value if beyond prior range
                if (not np.isinf(p_start[par].min)) and (np.isinf(p_start[par].max)) and (p_start[par].value<p_start[par].min):p_start[par].value=p_start[par].min
                if (np.isinf(p_start[par].min)) and (not np.isinf(p_start[par].max)) and (p_start[par].value>p_start[par].max):p_start[par].value=p_start[par].max
                if (not np.isinf(p_start[par].min)) and (not np.isinf(p_start[par].max)) and ((p_start[par].value<p_start[par].min) or (p_start[par].value>p_start[par].max)):p_start[par].value=0.5*(p_start[par].min+p_start[par].max)

            #MCMC fit
            elif (fit_dic['fit_mode'] in ['mcmc','ns']):
                
                #Range for walkers initialization
                #    - walkers can be initialized with either a uniform or gaussian distribution. Only one can be provided.
                #    - if parameters are not included we default to a uniform distribution
                if (par in model_prop):
                    if ('bd' in model_prop_par) and ('gauss' in model_prop_par):stop('ERROR : Provide either "bd" or "gauss" but not both in initialization of '+par)
                    if 'bd' in model_prop_par:fit_dic['uf_bd'][par]=model_prop_par['bd']
                    if 'gauss' in model_prop_par:fit_dic['gauss_bd'][par]=model_prop_par['gauss']
                else:
                    uf_bd=[-1e6,1e6]
                    if (not np.isinf(p_start[par].min)):uf_bd[0]=p_start[par].min
                    if (not np.isinf(p_start[par].max)):uf_bd[1]=p_start[par].max
                    fit_dic['uf_bd'][par]=uf_bd
                    
                #Check walkers initialization
                if ('MCMC_walkers_check' in fixed_args):fixed_args['MCMC_walkers_check'](par,fit_dic,p_start,fixed_args)

                #Input priors
                if (par in priors_prop):
                    fixed_args['varpar_priors'][par] = priors_prop[par] 
      
                #Default priors
                else:
                    varpar_priors=[-1e6,1e6]
                    if (not np.isinf(p_start[par].min)):varpar_priors[0]=p_start[par].min
                    if (not np.isinf(p_start[par].max)):varpar_priors[1]=p_start[par].max                
                    fixed_args['varpar_priors'][par]={'mod':'uf','low':varpar_priors[0],'high':varpar_priors[1]}
      
                #Prior check on standard properties
                #    - for uniform priors only
                if (fixed_args['varpar_priors'][par]['mod']=='uf') and ('prior_check' in fixed_args):fixed_args['prior_check'](par,fixed_args['varpar_priors'][par],p_start,fixed_args)

                #Change guess value and walker range if beyond prior range
                if fixed_args['varpar_priors'][par]['mod']=='uf':
                    if ((p_start[par].value<fixed_args['varpar_priors'][par]['low']) or (p_start[par].value>fixed_args['varpar_priors'][par]['high'])):p_start[par].value=0.5*(fixed_args['varpar_priors'][par]['low']+fixed_args['varpar_priors'][par]['high'])
                    if par in fit_dic['uf_bd']:
                        if (fit_dic['uf_bd'][par][0]<fixed_args['varpar_priors'][par]['low']):fit_dic['uf_bd'][par][0]=fixed_args['varpar_priors'][par]['low']
                        if (fit_dic['uf_bd'][par][1]>fixed_args['varpar_priors'][par]['high']):fit_dic['uf_bd'][par][1]=fixed_args['varpar_priors'][par]['high']
                    elif par in fit_dic['gauss_bd']:
                        if (fit_dic['gauss_bd'][par][0]<fixed_args['varpar_priors'][par]['low']):fit_dic['gauss_bd'][par][0]=fixed_args['varpar_priors'][par]['low']
                        if (fit_dic['gauss_bd'][par][0]>fixed_args['varpar_priors'][par]['high']):fit_dic['gauss_bd'][par][0]=fixed_args['varpar_priors'][par]['high']

    return p_start


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
                if (parval <= parprior['val']):ln_p += - 0.5*(np.log(2.*np.pi*parprior['s_val_low']**2.) + ( (parval - parprior['val'])/parprior['s_val_low']  )**2.)                       
                else:ln_p += - 0.5*(np.log(2.*np.pi*parprior['s_val_high']**2.) + ( (parval - parprior['val'])/parprior['s_val_high']  )**2.)   
    
            #Undefined prior
            else:
                stop('Undefined prior')
    
    #Additional priors using multiple parameters and complex functions
    if (fixed_args['global_ln_prior']) and (~ np.isinf(ln_p)): 
        ln_p += global_ln_prior_func(p_step,fixed_args)

    return ln_p

def ln_prior_func_NS(cube,fixed_args): 
    r"""**Prior transform function for nested sampling.**

    The prior transform function is used to implicitly specify the Bayesian prior for dynesty. 
    It functions as a transformation from a space where variables are i.i.d. within the n-dimensional
     unit cube (i.e. uniformly distributed from 0 to 1) to the parameter space of interest.

    Args:
        TBD
    
    Returns:
        TBD
    
    """
    x = cube.copy()  # copy cube

    #Priors on variable parameters
    for idx, parname in enumerate(fixed_args['var_par_list']):
        if parname in fixed_args['varpar_priors']:
    
            #Cube prior considered
            cube_prior=cube[idx]
    
            #Parameter prior properties
            parprior=fixed_args['varpar_priors'][parname]
    
            #Uniform prior
            if parprior['mod']=='uf':
                x[idx] = cube_prior * (parprior['high'] - parprior['low']) + parprior['low']
        
            #Gaussian prior   
            elif parprior['mod']=='gauss':
                x[idx] = stats.norm.ppf(cube_prior, parprior['val'], parprior['s_val'])
    
            #Gaussian prior with different halves
            elif parprior['mod'] == 'dgauss':
                # Map to the left half
                if cube_prior < 0.5:x[idx] = stats.norm.ppf(cube_prior, loc=parprior['val'], scale=parprior['low'])
                # Map to the right half
                else:x[idx] = stats.norm.ppf(cube_prior, loc=parprior['val'], scale=parprior['high'])

            #Undefined prior
            else:
                stop('Undefined prior')

    return x


def global_ln_prior_func(p_step,fixed_args):
    r"""**Global prior function.**

    Calculates :math:`\log(p)` cumulated over the chosen priors. 
    
    See `ln_prior_func()` for details on prior definition.

    Args:
        TBD

    Returns:
        TBD
        
    """ 
    ln_p_loc = 0.
    for key in fixed_args['prior_func']:ln_p_loc+=fixed_args['prior_func'][key]['func'](p_step,fixed_args,fixed_args['prior_func'][key])
    return ln_p_loc


def sub_ln_lkhood_func(p_step,fixed_args):
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
    # + a 'chi' array defined as the individual elements of 'step_chi2' below, in which case y_val is set to 0 and s_val to 1 (use_cov is set to False so that we enter the 'variance' condition below)
    #   so that 'res' below equals 'y_step' which is 'chi', and 'step_chi2' is the sum of the 'chi' values squared
    y_step,step_outputs = fixed_args['fit_func'](p_step, fixed_args['x_val'],args=fixed_args)

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
        step_chi2 = np.sum(np.power(chi[idx_fit],2.)) 
        ln_lkhood = - np.sum(np.log(np.sqrt(2.*np.pi)*L_mat[0][idx_fit])) - (step_chi2/2.)   
        
    #Use of variance alone
    else:
    
        #Modification of error bars on fitted values in case of jitter used as free parameter
        if (fixed_args['jitter']):sjitt_val=np.sqrt(fixed_args['cov_val'][0,idx_fit] + p_step['jitter']**2.)
        else:sjitt_val=np.sqrt(fixed_args['cov_val'][0,idx_fit])
            
        #Chi2
        step_chi2=np.sum(  np.power( res[idx_fit]/sjitt_val,2.) )

        #Ln likelihood
        ln_lkhood = - np.sum(np.log(np.sqrt(2.*np.pi)*sjitt_val)) - (step_chi2/2.)

    #Outputs
    if not fixed_args['step_chi2']:step_chi2 = None
    if not fixed_args['step_output']:step_outputs = None

    return ln_lkhood,step_chi2,step_outputs


def ln_lkhood_func_mcmc(p_step,fixed_args):
    r"""**Log(likelihood) with MCMC**

    Calculates the natural logarithm of the likelihood with MCMC.
    
    Args:
        TBD
    
    Returns:
        TBD
    
    """
    p_step_all = p_step
    ln_lkhood,step_chi2,step_outputs = sub_ln_lkhood_func(p_step_all,fixed_args) 
    return ln_lkhood,step_chi2,step_outputs


def ln_lkhood_func_ns(p_step,fixed_args):
    r"""**Log(likelihood) with nested sampling**

    Calculates the natural logarithm of the likelihood with nested sampling.
    
    Args:
        TBD
    
    Returns:
        TBD
    
    """
    
    #If nested sampling is used, posterior function is not called and dictionary of p_step must be initialized
    if (type(p_step)!=dict):
        
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

    else:
        p_step_all = p_step
        
    #Log(likelihood)
    ln_lkhood,step_chi2,step_outputs = sub_ln_lkhood_func(p_step_all,fixed_args) 

    #Store blobs
    blob = {'step_chi2':step_chi2,
            'step_outputs':step_outputs}

    return ln_lkhood,blob

    



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
        ln_lkhood,step_chi2,step_outputs = ln_lkhood_func_mcmc(p_step_all,fixed_args)

        #Set log-probability to -inf if likelihood is nan
        #    - happens when parameters go beyond their boundaries (hence ln_prior=-inf) but the model fails (hence ln_lkhood = nan)
        ln_prob=-np.inf if np.isnan(ln_lkhood) else ln_prior + ln_lkhood

    else: 
        ln_prob=-np.inf
        step_chi2 = None
        step_outputs = None

    return ln_prob,step_chi2,step_outputs


 
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
    y_step = fixed_args['fit_func'](p_step, fixed_args['x_val'],args=fixed_args)[0]

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

def lmfit_fitting_method_check(method):
    r"""**Method function**

    Returns string for the chi2 minimization method used.

    Args:
        method (str): method string input by user.

    Returns:
        method_str (str): method string which can be output.        
    """
    method_dic = {
    'leastsq': 'Levenberg-Marquardt (default)',
    'least_squares': 'Least-Squares minimization, using Trust Region Reflective method',
    'differential_evolution': 'differential evolution',
    'brute': 'brute force method',
    'basinhopping': 'basinhopping',
    'ampgo': 'Adaptive Memory Programming for Global Optimization',
    'nelder': 'Nelder-Mead',
    'lbfgsb': 'L-BFGS-B',
    'powell': 'Powell',
    'cg': 'Conjugate-Gradient',
    'newton': 'Newton-CG',
    'cobyla': 'Cobyla',
    'bfgs': 'BFGS',
    'tnc': 'Truncated Newton',
    'trust-ncg': 'Newton-CG trust-region',
    'trust-exact': 'nearly exact trust-region',
    'trust-krylov': 'Newton GLTR trust-region',
    'trust-constr': 'trust-region for constrained optimization',
    'dogleg': 'Dog-leg trust-region',
    'slsqp': 'Sequential Linear Squares Programming',
    'emcee': 'Maximum likelihood via Monte-Carlo Markov Chain',
    'shgo': 'Simplicial Homology Global Optimization',
    'dual_annealing': 'Dual Annealing optimization'
        } 
    if method in method_dic:method_str = method_dic[method]
    else:stop('ERROR: Invalid chi2 minimization method. Check lmfit documentation for valid methods.')
    return method_str


def init_fit(fit_dic,fixed_args,p_start,model_par_names,model_par_units):
    r"""**Fit initialization**

    Initializes lmfit, MCMC, and nested sampling.

    Args:
        TBD
    
    Returns:
        TBD
    
    """
    
    #Start counter
    fit_dic['st0']=get_time()     

    #Parameters names
    #    - order will be used as such afterward
    fixed_args['par_names']=[par for par in p_start]

    #List of parameters name and their indices in the table of all parameters
    var_par_list=[]
    ivar_par_list=[]
    ifix_par_list=[]
    fix_par_list=[]
    iexp_par_list=[]
    exp_par_list=[]
    var_par_names=[]
    var_par_units=[]
    fix_par_names=[]
    fix_par_units=[]
    fixed_args['fixed_par_val'] = {}
    fixed_args['linked_par_expr'] = {}
    for ipar,par in enumerate(fixed_args['par_names']):
        
        #Variable parameter
        if p_start[par].vary:
            var_par_list+=[par]
            ivar_par_list+=[ipar]

        #Fixed parameter
        #    - retrieve parameters in the 'p_start' structure that will remain fixed to their value 
        #    - p_start cannot be given as input to the MCMC with both fixed and variable parameters (ie, they must be given separately)
        else:
            fixed_args['fixed_par_val'][par] = p_start[par].value
            ifix_par_list+=[ipar]
            fix_par_list+=[par]

        #Parameter name and unit    
        inst_par = '_'
        vis_par = '_'
        pl_name = None                
        ar_name = None
        
        #Parameter depends on epoch
        if ('__IS') and ('_VS') in par:
            inst_vis_par = par.split('__IS')[1]
            inst_par  = inst_vis_par.split('_VS')[0]
            vis_par  = inst_vis_par.split('_VS')[1]   
            if ('__ar' in par) or ('__pl' in par):
                if ('__ar' in par):
                    ar_name = (par.split('__IS')[0]).split('__ar')[1]  
                    par_name_root = par.split('__ar')[0]
                if ('__pl' in par):
                    pl_name = (par.split('__IS')[0]).split('__pl')[1]
                    par_name_root = par.split('__pl')[0]
            else:
                par_name_root = par.split('__IS')[0]
        
        #Parameter does not depend on epoch
        else:
            if ('__ar' in par) or ('__pl' in par):
                if ('__ar' in par):
                    ar_name = par.split('__ar')[1]  
                    par_name_root = par.split('__ar')[0]
                if ('__pl' in par):
                    pl_name = par.split('__pl')[1]    
                    par_name_root = par.split('__pl')[0]
            else: 
                par_name_root = par                                                     
        par_name_loc = model_par_names(par_name_root)
        
        if ar_name is not None:par_name_loc+='['+ar_name+']'
        if pl_name is not None:par_name_loc+='['+pl_name+']'                             
        if inst_par != '_':
            par_name_loc+='['+inst_par+']'
            if vis_par != '_':par_name_loc+='('+vis_par+')'
            
        if p_start[par].vary:            
            var_par_names+=[par_name_loc]
            var_par_units+=[model_par_units(par_name_root)]
        else:            
            fix_par_names+=[par_name_loc]
            fix_par_units+=[model_par_units(par_name_root)]
            
        #Parameters linked to variable parameters through an expression
        if p_start[par].expr!=None:
            fixed_args['linked_par_expr'][par] = p_start[par].expr
            p_start[par].vary=False
            iexp_par_list+=[ipar]
            exp_par_list+=[par]
    fixed_args['var_par_list']=np.array(var_par_list,dtype='U50')  
    fixed_args['var_par_list_nom'] = deepcopy(fixed_args['var_par_list'])
    fixed_args['ivar_par_list']=ivar_par_list
    fixed_args['ifix_par_list']=ifix_par_list
    fixed_args['fix_par_list']=fix_par_list
    fixed_args['iexp_par_list']=iexp_par_list
    fixed_args['exp_par_list']=exp_par_list
    fixed_args['var_par_names']=np.array(var_par_names,dtype='U50')
    fixed_args['var_par_units']=np.array(var_par_units,dtype='U50')
    fixed_args['fix_par_names']=np.array(fix_par_names,dtype='U50')
    fixed_args['fix_par_units']=np.array(fix_par_units,dtype='U50')

    #Retrieve the number of active regions that are present (whether their parameters are fixed or fitted)
    ar_names=[]
    for par in fixed_args['par_names']:
        if '__ar' in par:
            ar_names.append(par.split('__ar')[1])
    fixed_args['num_ar']=len(np.unique(ar_names))

    #Update value of fixed parameters linked to variable parameters through an expression
    if len(fixed_args['linked_par_expr'])>0:
        
        #Attribute their fixed value to parameters that are not variable and not linked via an expression, but might be used in the expression of other non-variable parameters
        fixed_args['fixed_par_val_noexp_list']=[par for par in fixed_args['fixed_par_val'] if par not in fixed_args['linked_par_expr']]

    #Number of free parameters    
    fit_dic['merit'] = {}
    if fit_dic['fit_mode']=='fixed':fit_dic['merit']['n_free'] = 0.
    else:fit_dic['merit']['n_free'] = len(var_par_list) 
    if 'calc_merit' not in fit_dic:fit_dic['calc_merit'] = True

    #Initialize save file
    if ('save_outputs' not in fit_dic):fit_dic['save_outputs']=True 
    if fit_dic['save_outputs']:
        if (not os_system.path.exists(fit_dic['save_dir'])):os_system.makedirs(fit_dic['save_dir'])
        fit_dic['file_save']=open(fit_dic['save_dir']+'Outputs','w+')

    #No jitter by default
    #    - can be used to reach a reduced chi2 of 1 by adding quadratically the adjusted jitter value to the measured errors
    fixed_args['jitter']=False if ('jitter' not in fixed_args) else fixed_args['jitter']        
    
    #Default settings
    if fit_dic['fit_mode']=='chi2':

        #Fitting method
        if ('chi2_fitting_method' not in fit_dic):fit_dic['chi2_fitting_method']='leastsq'
    
    elif fit_dic['fit_mode'] in ['mcmc','ns']:
     
        #Reboot
        if ('reboot' not in fit_dic):fit_dic['reboot']=''     

        #Option to initialize walker distribution using a pre-computed Hessian matrix
        if ('use_hess' not in fit_dic):fit_dic['use_hess']=''

        #Monitor progress
        if ('progress' not in fit_dic):fit_dic['progress']=True 
        
        #Store outputs from MCMC steps
        if ('step_output' not in fixed_args):fixed_args['step_output'] = False

        #Store chi2 outputs from MCMC steps
        if ('step_chi2' not in fixed_args):fixed_args['step_chi2'] = True
    
        #Excluding manually some of the walkers
        if ('exclu_walk' not in fit_dic):fit_dic['exclu_walk']=False 

        #Excluding automatically walkers with median beyond +- threshold * (1 sigma) of global median 
        #    - set to None, or exclusion threshold
        if ('exclu_walk_autom' not in fit_dic):fit_dic['exclu_walk_autom']=None 

        #Excluding manually some of the samples
        if ('exclu_samp' not in fit_dic):fit_dic['exclu_samp']={} 

        #Quantiles calculated by default
        if ('calc_quant' not in fit_dic):fit_dic['calc_quant']=True 

        #No thinning of the chains
        if ('thin_MCMC' not in fit_dic):fit_dic['thin_MCMC']=False  

        #Calculation of 1sigma HDI intervals
        #    - set fit_dic['HDI'] to None in options to prevent calculation
        #    - applied within 'postMCMCwrapper_2' to the modified final chains
        if ('HDI' not in fit_dic):fit_dic['HDI']='1s' 
        
        #Number of bins in 1D histograms used for HDI definition
        #    - adjust HDI_nbins or HDI_dbins for each parameter: there must be enough bins for the HDI interval to contain a fraction of samples close
        # to the requested confidence interval, but not so much that bins within the histogram are empty and artificially create several HDI intervals
        #      alternatively set fit_dic['HDI_nbins']= {} or set it to None for a given value for automatic definition (preferred solution for unimodal PDFs)
        if ('HDI_nbins' not in fit_dic):fit_dic['HDI_nbins']={} 
        
        #Use custom HDI calculation by default
        if ('use_arviz' not in fit_dic):fit_dic['use_arviz'] = False

        #No calculation of upper/lower limits
        #    - to be used for PDFs bounded by the parameter space
        #    - define the type of limits and the confidence level
        #      limits will then be calculated from the minimum or maximum of the distribution
        #    - this should return similar results as the HDI intervals, but more precise because it does not rely on the sampling of the PDF
        if ('conf_limits' not in fit_dic):fit_dic['conf_limits']={} 

        #Retrieve model sample
        #    - calculation of models sampling randomly the full distribution of the parameters
        #    - disabled by default
        if ('calc_sampMCMC' not in fit_dic):fit_dic['calc_sampMCMC']=False  
    
        #On-screen printing of errors
        if ('sig_list' not in fit_dic):fit_dic['sig_list']=['1s'] 

        #Saving final results by default
        if 'save_results' not in fit_dic:fit_dic['save_results']=True 

        #Plot correlation diagram for final parameters      
        if ('save_MCMC_corner' not in fit_dic):fit_dic['save_MCMC_corner']='pdf' 
        if ('corner_options' not in fit_dic):fit_dic['corner_options']={'plot_HDI':True,'color_levels':['deepskyblue','lime']}

        #Plot corner diagram for simulated points
        if ('save_sim_points_corner' not in fit_dic):fit_dic['save_sim_points_corner']='' 
        if ('sim_corner_options' not in fit_dic):fit_dic['sim_corner_options']={}
        
        #Plot chains for MCMC parameters    
        if ('save_MCMC_chains' not in fit_dic):fit_dic['save_MCMC_chains']='png'

        #Plot chi2 chains for MCMC run    
        if ('save_chi2_chains' not in fit_dic):fit_dic['save_chi2_chains']=''
        
        #Run name
        if ('run_name' not in fit_dic):fit_dic['run_name']='' 

        #Default thread number
        if 'nthreads' not in fit_dic:fit_dic['nthreads'] = 1

        #Sampler dictionary
        if 'sampler_set' not in fit_dic:fit_dic['sampler_set']={}  

        #MCMC walker / NS sampler monitoring
        if ('monitor' not in fit_dic):fit_dic['monitor']=False 

        if fit_dic['fit_mode']=='mcmc':
    
            #Do not use complex prior function by default
            fixed_args['global_ln_prior'] = False if (('global_ln_prior' not in fixed_args) or ('prior_func' not in fixed_args)) else fixed_args['global_ln_prior']
    
            #Impose a specific maximum correlation length to thin the chains
            #    - otherwise set to 0 for automatic determination
            if fit_dic['thin_MCMC'] and ('max_corr_length' not in fit_dic):fit_dic['max_corr_length']=50.         
    
            #No calculation of envelopes
            #    - calculation of models using parameter values within their 1sigma range
            if ('calc_envMCMC' not in fit_dic):fit_dic['calc_envMCMC']=False 
            if fit_dic['calc_envMCMC']:
                if ('st_samp' not in fit_dic):fit_dic['st_samp']=10 
                if ('end_samp' not in fit_dic):fit_dic['end_samp']=10 
                if ('n_samp' not in fit_dic):fit_dic['n_samp']=100 
            
            if 'nwalkers' not in fit_dic['sampler_set']:fit_dic['nwalkers'] = int(3*fit_dic['merit']['n_free'])
            else:fit_dic['nwalkers'] = fit_dic['sampler_set']['nwalkers']
            if 'nsteps' not in fit_dic['sampler_set']:fit_dic['nsteps'] = 5000
            else:fit_dic['nsteps'] = fit_dic['sampler_set']['nsteps']
            if 'nburn' not in fit_dic['sampler_set']:fit_dic['nburn'] = 1000
            else:fit_dic['nburn'] = fit_dic['sampler_set']['nburn']
    
            #Disable multi-threading
            if ('unthreaded_op' in fit_dic) and ('emcee' in fit_dic['unthreaded_op']):fit_dic['emcee_nthreads']=1
            else:fit_dic['emcee_nthreads'] = fit_dic['nthreads']
        
        elif fit_dic['fit_mode']=='ns': 

            #Restoring
            if ('restore' not in fit_dic):fit_dic['restore']=''
            
            #No calculation of envelopes
            #    - calculation of models using parameter values within their 1sigma range
            fit_dic['calc_envMCMC']=False 

            #Live points
            if 'nlive' not in fit_dic['sampler_set']:fit_dic['nlive'] = fit_dic['merit']['n_free'] * (fit_dic['merit']['n_free'] + 1) / 2
            else:fit_dic['nlive'] = fit_dic['sampler_set']['nlive']

            #Bounding method used 
            if ('bound_method' not in fit_dic['sampler_set']):fit_dic['bound_method']='auto'
            else:fit_dic['bound_method']=fit_dic['sampler_set']['bound_method']
            
            #Sampling method used 
            if ('sample_method' not in fit_dic['sampler_set']):fit_dic['sample_method']='unif'
            else:fit_dic['sample_method']=fit_dic['sampler_set']['sample_method']

            #Threshold on the log-likelihood difference between subsequent NS steps. Once the difference falls below this threshold, the run stops.
            if ('dlogz' not in fit_dic['sampler_set']):fit_dic['dlogz']=0.1
            else:fit_dic['dlogz']=fit_dic['sampler_set']['dlogz']

            #Checkpointing the sampler
            if ('monitor' in fit_dic['sampler_set']):fit_dic['monitor']|=fit_dic['sampler_set']['monitor']

            #Set burn-in steps to 0 to makes post-processing with the MCMC functions possible
            fit_dic['nburn'] = 0.

            #Disable multi-threading
            if ('unthreaded_op' in fit_dic) and ('dynesty' in fit_dic['unthreaded_op']):fit_dic['dynesty_nthreads']=1
            else:fit_dic['dynesty_nthreads'] = fit_dic['nthreads']

    return None


def call_lmfit(p_use, xtofit, ytofit, covtofit, f_use,method='leastsq', maxfev=None, xtol=1e-7, ftol=1e-7,verbose=False,fixed_args=None,show_correl=False,fit_dic=None):
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
    if method=='leastsq':
        result = minimize(ln_prob_func_lmfit, p_use, args=(xtofit, argstofit), method=method, max_nfev=max_nfev, xtol=xtol, ftol=ftol,scale_covar = False )
    else:
        if method=='lbfgsb':meth_args = {'tol':ftol}
        else:meth_args = {}
        result = minimize(ln_prob_func_lmfit, p_use, args=(xtofit, argstofit), method=method, max_nfev=max_nfev,scale_covar = False , **meth_args)
    
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
    merit['fit'] = f_use(p_best, xtofit,args=argstofit)[0]
    
    #Corresponding residuals
    merit['resid'] = merit['fit'] - ytofit
    
    #Dispersion of residuals
    merit['rms'] = merit['resid'].std()

    #Chi2 value
    merit['chi2'] = np.sum(ln_prob_func_lmfit(p_best, xtofit, fixed_args=argstofit)**2.)
    merit['chi2r'] = merit['chi2']/result.nfree

    #Hessian matrix
    merit['hess_matrix']=compute_Hessian(p_best, ln_prob_func_lmfit, [xtofit], {'fixed_args':argstofit})
    
    #Bayesian Indicator Criterion   
    merit['BIC'] = merit['chi2'] + result.nvarys*np.log(result.ndata)
    
    #Cumulative distribution function
    #    - cdf = (0.05<cdf<0.95, if not: too bad/good fit or big/small error)
    merit['cdf'] = special.chdtrc(result.nfree,merit['chi2'])
    
    #Varia
    merit['method'] = result.method
    merit['success'] = result.success
    merit['message'] = result.message
    merit['eval'] = result.nfev
    
    if fit_dic is not None:
        fit_dic['merit'].update(merit)
        fit_dic['merit']['method'] = method

    return result, merit ,p_best
    
    

def call_MCMC(run_mode,nthreads,fixed_args,fit_dic,run_name='',verbose=True,save_raw=True):
    r"""**Wrapper to MCMC**

    Runs `emcee` and outputs results and merit values.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """
    
    #Run MCMC
    if run_mode=='use':
        print('         Applying MCMC') 

        #Automatic definition of undefined priors                
        for par in fixed_args['var_par_list']:
            if par not in fixed_args['varpar_priors']:fixed_args['varpar_priors'][par]={'mod':'uf','low':-1e10,'high':1e10}
    
        #Set initial parameter distribution
        fit_dic['initial_distribution'] = np.zeros((fit_dic['nwalkers'],fit_dic['merit']['n_free']))
        if (len(fit_dic['reboot'])>0):
            print('         Rebooting previous run')
            
            #Reboot MCMC from end of previous run
            walker_chains_last=np.load(fit_dic['reboot'])['walker_chains'][:,-1,:]  #(nwalkers, nsteps, n_free)
      
            #Overwrite starting values of new chains
            for ipar in range(len(fixed_args['var_par_list'])):
                fit_dic['initial_distribution'][:,ipar] = walker_chains_last[:,ipar] 
              
        else:
            
            #Custom initialization
            if 'custom_init_walkers' in fit_dic:fit_dic['custom_init_walkers'](fit_dic,fixed_args)

            #Random distribution within defined range
            else:
                if fit_dic['use_hess'] != '':
                    print('         Initializing walkers with Hessian')
                    #Retrieve Hessian matrix
                    hess_matrix = np.load(fit_dic['use_hess'],allow_pickle=True)['data'].item()['hess_matrix']

                    #Build covariance matrix
                    cov_matrix = np.linalg.inv(hess_matrix)

                    #Checking that chi2 fit and current MCMC fit are run on same parameters
                    if len(fixed_args['var_par_list']) != cov_matrix.shape[0] : stop('Chi2 fit used to estimate the Hessian and current MCMC run do not share the same parameters.')

                    #Retrieving central location of parameters
                    central_loc = np.zeros(len(fixed_args['var_par_list']), dtype=float)
                    for ipar, param in enumerate(fixed_args['var_par_list']):central_loc[ipar] = fit_dic['mod_prop'][param]['guess']

                    fit_dic['initial_distribution'] = np.random.multivariate_normal(central_loc, cov_matrix, size=fit_dic['nwalkers'])
                
                else:
                    print('         Initializing walkers with uniform or gaussian distributions')
                    for ipar,par in enumerate(fixed_args['var_par_list']):
                        if par in fit_dic['uf_bd']:fit_dic['initial_distribution'][:,ipar]=np.random.uniform(low=fit_dic['uf_bd'][par][0], high=fit_dic['uf_bd'][par][1], size=fit_dic['nwalkers']) 
                        elif par in fit_dic['gauss_bd']:fit_dic['initial_distribution'][:, ipar]=np.random.normal(loc=fit_dic['gauss_bd'][par][0], scale=fit_dic['gauss_bd'][par][1], size=fit_dic['nwalkers']) 
                   
    
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

        #Defining blobs format
        blobs_dtype = [("step_chi2", dict) , ("step_outputs", float)]
    
        #Multiprocessing
        if nthreads>1:
            pool_proc = Pool(processes=nthreads)  
            print('         Running with '+str(nthreads)+' threads')    
            sampler = emcee.EnsembleSampler(fit_dic['nwalkers'],            #Number of walkers
                                            n_free,                         #Number of free parameters in the model
                                            ln_prob_func_mcmc,              #Log-probability function 
                                            args=[fixed_args],              #Fixed arguments for the calculation of the likelihood and priors
                                            pool = pool_proc,
                                            backend=backend,                #Monitor chain progress 
                                            blobs_dtype=blobs_dtype)                
        else:sampler = emcee.EnsembleSampler(fit_dic['nwalkers'],n_free,ln_prob_func_mcmc,args=[fixed_args],backend=backend,blobs_dtype=blobs_dtype)         
            
        #Run MCMC
        #    - possible options:
        # + iterations: number of iterations to run            
        sampler.run_mcmc(fit_dic['initial_distribution'], fit_dic['nsteps'],progress=fit_dic['progress'])
        if verbose:print('   duration : '+str((get_time()-st0)/60.)+' mn')
       
        #Walkers chain
        #    - sampler.chain is of shape (nwalkers, nsteps, n_free)
        #     - parameters have the same order as in 'initial_distribution' and 'var_par_list'
        walker_chains = sampler.chain    
        
        #Complementary outputs
        #    - shape (nsteps,nwalkers,2)
        blobs = sampler.get_blobs()

        #Step-by-step chi2 chain
        if fixed_args['step_chi2']:fixed_args['chi2_storage'] = blobs['step_chi2']
        else:fixed_args['chi2_storage'] = None
        
        #Step-by-step complementary function output
        if fixed_args['step_output']:step_outputs = blobs['step_outputs']
        else:step_outputs = None

        #Save raw MCMC results 
        if save_raw:
            if (not os_system.path.exists(fit_dic['save_dir'])):os_system.makedirs(fit_dic['save_dir'])
            np.savez(fit_dic['save_dir']+'raw_chains_walk'+str(fit_dic['nwalkers'])+'_steps'+str(fit_dic['nsteps'])+run_name,walker_chains=walker_chains, initial_distribution=fit_dic['initial_distribution'],step_outputs = step_outputs, step_chi2 = fixed_args['chi2_storage'])

        #Delete temporary chains after final walkers are saved
        if backend is not None:os_system.remove(fit_dic['save_dir']+'monitor'+str(fit_dic['nwalkers'])+'_steps'+str(fit_dic['nsteps'])+run_name+'.h5')
    
        #Close workers
        if nthreads>1:    
            pool_proc.close()
            pool_proc.join() 	

    #---------------------------------------------------------------  
   
    #Reuse MCMC
    elif run_mode=='reuse':
        print('         Retrieving MCMC') 
        
        #Retrieve mcmc run from standard mcmc directory
        if len(fit_dic['reuse'])==0:
            mcmc_load = np.load(fit_dic['save_dir']+'raw_chains_walk'+str(fit_dic['nwalkers'])+'_steps'+str(fit_dic['nsteps'])+fit_dic['run_name']+'.npz',allow_pickle = True)
            walker_chains=mcmc_load['walker_chains']            #(nwalkers, nsteps, n_free)
            step_outputs=mcmc_load['step_outputs']              #(nsteps, nwalkers)
            fixed_args['chi2_storage'] = mcmc_load['step_chi2'] #(nsteps, nwalkers)

        #Retrieve mcmc run(s) from list of input paths
        else:
            walker_chains = np.empty([fit_dic['nwalkers'],0,fit_dic['merit']['n_free'] ],dtype=float)
            if fixed_args['step_output']:step_outputs = np.empty([0,fit_dic['nwalkers']],dtype=object)
            else:step_outputs=None
            if fixed_args['step_chi2']:fixed_args['chi2_storage'] = np.empty([0,fit_dic['nwalkers']],dtype=object)
            else:fixed_args['chi2_storage']=None
            fit_dic['nsteps'] = 0
            fit_dic['nburn'] = 0
            for mcmc_path,nburn in zip(fit_dic['reuse']['paths'],fit_dic['reuse']['nburn']):
                mcmc_load=np.load(mcmc_path, allow_pickle=True)
                walker_chains_loc=mcmc_load['walker_chains'][:,nburn::,:] 
                if fixed_args['step_output']:step_output_loc=mcmc_load['step_outputs'][nburn::] 
                if fixed_args['step_chi2']:step_chi2_loc=mcmc_load['step_chi2'][nburn::] 
                fit_dic['nsteps']+=(walker_chains_loc.shape)[1]
                walker_chains = np.append(walker_chains,walker_chains_loc,axis=1)
                if fixed_args['step_output']:step_outputs = np.append(step_outputs,step_output_loc,axis=0)
                if fixed_args['step_chi2']:fixed_args['chi2_storage'] = np.append(fixed_args['chi2_storage'],step_chi2_loc,axis=0)

    return walker_chains,step_outputs



def call_NS(run_mode,nthreads,fixed_args,fit_dic,run_name='',verbose=True,save_raw=True):
    r"""**Wrapper to Nested Sampling**

    Runs `dynesty` and outputs results and merit values.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """
    
    #Run nested sampling
    if run_mode=='use':
        print('         Applying nested sampling') 

        #Automatic definition of undefined priors                
        for par in fixed_args['var_par_list']:
            if par not in fixed_args['varpar_priors']:fixed_args['varpar_priors'][par]={'mod':'uf','low':-1e10,'high':1e10}
    
        #Dictionary entry for storage
        fit_dic['initial_distribution'] = [np.zeros((fit_dic['nlive'],fit_dic['merit']['n_free']), dtype=float), #Prior values of starting points
                                           np.zeros((fit_dic['nlive'],fit_dic['merit']['n_free']), dtype=float), #Coordinates of starting live points
                                           np.zeros((fit_dic['nlive'],), dtype=float), #Corresponding log-likelihood values
                                           np.zeros((fit_dic['nlive']), dtype=dict)] 

        if (len(fit_dic['reboot'])>0):
            print('         Rebooting previous run')
            
            #Reboot nested sampling from end of previous run
            fit_dic['initial_distribution'][1]=np.load(fit_dic['reboot'])['walker_chains'][:,-1,:]  #(nwalkers, nsteps, n_free)

        else:
            
            #Custom initialization
            if 'custom_init_nlive' in fit_dic:fit_dic['custom_init_nlive'](fit_dic,fixed_args)

            #Random distribution within defined range
            else:
                if fit_dic['use_hess'] != '':
                    print('         Initializing live points with Hessian')
                    
                    #Retrieve Hessian matrix
                    hess_matrix = np.load(fit_dic['use_hess'],allow_pickle=True)['data'].item()['hess_matrix']

                    #Build covariance matrix
                    cov_matrix = np.linalg.inv(hess_matrix)

                    #Checking that chi2 fit and current MCMC fit are run on same parameters
                    if len(fixed_args['var_par_list']) != cov_matrix.shape[0] : stop('Chi2 fit used to estimate the Hessian and current MCMC run do not share the same parameters.')

                    #Retrieving central location of parameters
                    central_loc = np.zeros(len(fixed_args['var_par_list']), dtype=float)
                    for ipar, param in enumerate(fixed_args['var_par_list']):central_loc[ipar] = fit_dic['mod_prop'][param]['guess']

                    fit_dic['initial_distribution'][1] = np.random.multivariate_normal(central_loc, cov_matrix, size=fit_dic['nlive'])

                else:
                    print('         Initializing live points with uniform/gaussian distributions')
                    print('         WARNING : It is crucial that the initial set of live points have been sampled from the prior. Failure to provide a set of valid live points will result in incorrect results.')
                    for ipar,par in enumerate(fixed_args['var_par_list']):
                        if par in fit_dic['uf_bd']:fit_dic['initial_distribution'][1][:,ipar]=np.random.uniform(low=fit_dic['uf_bd'][par][0], high=fit_dic['uf_bd'][par][1], size=fit_dic['nlive']) 
                        elif par in fit_dic['gauss_bd']:fit_dic['initial_distribution'][1][:, ipar]=np.random.normal(loc=fit_dic['gauss_bd'][par][0], scale=fit_dic['gauss_bd'][par][1], size=fit_dic['nlive']) 

        #Retrieve prior values of starting points
        for idx, parname in enumerate(fixed_args['var_par_list']):  
            if parname in fit_dic['uf_bd']:fit_dic['initial_distribution'][0][:, idx] = stats.uniform.cdf(fit_dic['initial_distribution'][1][:, idx], loc=fit_dic['uf_bd'][par][0], scale=fit_dic['uf_bd'][par][1]-fit_dic['uf_bd'][par][0])
            elif parname in fit_dic['gauss_bd']:fit_dic['initial_distribution'][0][:, idx] = stats.norm.cdf(fit_dic['initial_distribution'][1][:, idx], loc=fit_dic['gauss_bd'][par][0], scale=fit_dic['gauss_bd'][par][1])
            else:stop('         Starting point distribution for {parname} should be gaussian or uniform.')

        #Retrieve likelihood values
        for idx, live_point in enumerate(fit_dic['initial_distribution'][1]):     
            ln_lkhood,blob = ln_lkhood_func_ns(live_point, fixed_args)
            fit_dic['initial_distribution'][2][idx] = ln_lkhood
            fit_dic['initial_distribution'][3][idx] = blob

        #By default use variance
        if 'use_cov' not in fixed_args:fixed_args['use_cov']=False

        #Save state of sampler in case of crash
        if ('monitor' in fit_dic) and fit_dic['monitor']:dynesty_checkpoint = fit_dic['save_dir']+'monitor_dynesty.save'
        else:dynesty_checkpoint=None

        #Call to NS
        st0=get_time()
        n_free=np.shape(fit_dic['initial_distribution'][0])[1]
        
        #Multiprocessing
        use_threads = (nthreads > 1)
        pool_proc = Pool(processes=nthreads) if use_threads else None  

        #Restoring a previous run
        if fit_dic['restore']:
            print(f"         Restoring previous run with {nthreads} threads" if use_threads else "         Restoring previous run")
            sampler = dynesty.DynamicNestedSampler.restore(fit_dic['restore'], pool=pool_proc)

        #Doing a new run
        else:
            print(f"         Running with {nthreads} threads" if use_threads else "         Running")
            sampler_kwargs = {
                "bound": fit_dic["bound_method"],            #Prior bounding method
                "sample": fit_dic["sample_method"],          #Likelihood sampling method
                "nlive": fit_dic['nlive'],                   #Number of live points
                "ndim": n_free,                              #Number of parameters accepted by ln_prior_func_NS
                "logl_args": [fixed_args],                   #Fixed arguments for the calculation of the likelihood
                "ptform_args": [fixed_args],                 #Fixed arguments for the calculation of the priors
                "blob": True,                                #Whether blobs are present or not
            }
            if use_threads:
                sampler_kwargs.update({"pool": pool_proc,                           #Multiprocessing pool considered
                                       "queue_size": nthreads                       #Multiprocessing queue size
                                       })
            sampler = dynesty.DynamicNestedSampler(ln_lkhood_func_ns,                          #Log-likelihood function
                                                   ln_prior_func_NS,                           #Log-prior function
                                                   **sampler_kwargs)

        
        # Run NS with restore or initial parameters
        sampler.run_nested(
            resume=bool(fit_dic['restore']),
            live_points=fit_dic['initial_distribution'],
            print_progress=fit_dic['progress'],
            dlogz_init=fit_dic['dlogz'],
            checkpoint_file=dynesty_checkpoint,
        )

        if verbose:print('   duration : '+str((get_time()-st0)/60.)+' mn')

        #Chains
        #    - result.samples is of shape (nsteps, n_free), and we will reshape it to (1, nsteps, n_free) to make the next steps easier
        #     - parameters have the same order as in 'initial_distribution' and 'var_par_list'
        result = sampler.results
        NS_chains = result.samples
        fit_dic['nsteps'] = NS_chains.shape[0]
        fit_dic['nwalkers'] = 1
        fit_dic['nburn'] = 1
        walker_chains = NS_chains.reshape((fit_dic['nwalkers'], fit_dic['nsteps'], n_free))
        
        #Complementary outputs
        #    - shape (nsteps,2)
        blobs = result['blob']
        #    - building storage for step-by-step function output and chi2 chains
        blob_steps = np.zeros(fit_dic['nsteps'])
        blob_chi2 = np.zeros(fit_dic['nsteps'])
        #    - Populating arrays
        for step in range(fit_dic['nsteps']):
            for blob_arr, blob_name in zip([blob_steps, blob_chi2], ["step_outputs", "step_chi2"]):blob_arr[step]=blobs[step][blob_name]
        #    - storing them in fixed_args for future steps
        if fixed_args['step_output']:step_outputs = blob_steps
        else:step_outputs = None
        if fixed_args['step_chi2']:fixed_args['chi2_storage'] = blob_chi2.reshape((fit_dic['nsteps'],fit_dic['nwalkers']))
        else:fixed_args['chi2_storage'] = None

        #Save raw NS results 
        if save_raw:
            if (not os_system.path.exists(fit_dic['save_dir'])):os_system.makedirs(fit_dic['save_dir'])
            np.savez(fit_dic['save_dir']+'raw_chains_live'+str(fit_dic['nlive'])+run_name,walker_chains=walker_chains, initial_distribution=fit_dic['initial_distribution'][0], step_outputs = step_outputs, step_chi2 = fixed_args['chi2_storage'])
        
        #Close workers
        if nthreads>1:    
            pool_proc.close()
            pool_proc.join()    

    #---------------------------------------------------------------  
   
    #Reuse NS
    elif run_mode=='reuse':
        print('         Retrieving NS') 

        #Fix/initialize number of walkers and steps  - needed for subsequent analyses
        fit_dic['nwalkers'] = 1
        
        #Retrieve NS run from standard NS directory
        if len(fit_dic['reuse'])==0:
            NS_load = np.load(fit_dic['save_dir']+'raw_chains_live'+str(fit_dic['nlive'])+fit_dic['run_name']+'.npz',allow_pickle = True)
            walker_chains=NS_load['walker_chains']  #(1, nsteps, n_free)
            step_outputs=NS_load['step_outputs']    #(nsteps, nwalkers)
            fixed_args['chi2_storage'] = NS_load['step_chi2'] #(nsteps, nwalkers)


        #Retrieve NS run(s) from list of input paths
        else:
            walker_chains = np.empty([fit_dic['nwalkers'],0,fit_dic['merit']['n_free'] ],dtype=float)
            if fixed_args['step_output']:step_outputs = np.empty([0,fit_dic['nwalkers']],dtype=object)
            else:step_outputs=None
            if fixed_args['step_chi2']:fixed_args['chi2_storage'] = np.empty([0,fit_dic['nwalkers']],dtype=object)
            else:fixed_args['chi2_storage']=None
            fit_dic['nsteps'] = 0
            fit_dic['nburn'] = 0
            for NS_path, nburn in zip(fit_dic['reuse']['paths'], fit_dic['reuse']['nburn']):
                NS_load=np.load(NS_path, allow_pickle=True)
                walker_chains_loc=NS_load['walker_chains'][:,nburn::,:]
                if fixed_args['step_output']:step_output_loc=NS_load['step_outputs'][nburn::]
                if fixed_args['step_chi2']:step_chi2_loc=NS_load['step_chi2'][nburn::]
                fit_dic['nsteps']+=(walker_chains_loc.shape)[1]
                walker_chains = np.append(walker_chains,walker_chains_loc,axis=1)
                if fixed_args['step_output']:step_outputs = np.append(step_outputs,step_output_loc,axis=0)
                if fixed_args['step_chi2']:fixed_args['chi2_storage'] = np.append(fixed_args['chi2_storage'],step_chi2_loc,axis=0)

    return walker_chains,step_outputs




##################################################################################################
#%%% Post-processing
##################################################################################################   
       
def fit_merit(mode,p_final_in,fixed_args,fit_dic,verbose,verb_shift = ''):
    r"""**Post-proc: model merit and parameters**

    Calculates, prints, and saves merit indicators of the best-fit model, as well as best-fit values and confidence intervals for the original and derived parameters. 
     
    Args:
        TBD
    
    Returns:
        TBD
    
    """
    if verbose or fit_dic['save_outputs']:
        if (mode=='derived') and (len([parname for parname in fixed_args['var_par_list'] if (parname not in fixed_args['var_par_list_nom'])])>0):print_der = True
        else:print_der = False        
        txt_print = []
    else:txt_print = None
        
    if mode=='nominal': 

        #End counter
        fit_dic['fit_dur'] = get_time()-fit_dic['st0']
    
        #Convert parameters() structure into dictionary 
        if (fit_dic['fit_mode'] !='mcmc') and (fit_dic['fit_mode'] !='ns'): 
            p_final={}
            for par in p_final_in:p_final[par]=p_final_in[par].value   
            if fit_dic['fit_mode']=='chi2':
                fit_dic['sig_parfinal_err']={'1s':np.zeros([2,fit_dic['merit']['n_free']])}            
                for ipar,par in enumerate(fixed_args['var_par_list']): 
                    fit_dic['sig_parfinal_err']['1s'][:,ipar]=p_final_in[par].stderr  
        else:p_final = deepcopy(p_final_in)
        
        #Merit values
        if fit_dic['calc_merit']:
         
            #Calculation of best-fit model equivalent to the observations, corresponding residuals, and RMS
            #    - only in the case where the function does return the model
            if not fixed_args['inside_fit']:
                res_tab = fixed_args['y_val'] - fixed_args['fit_func'](p_final,fixed_args['x_val'],args=fixed_args)[0] 
                fit_dic['merit']['rms']=res_tab.std()       
            else:fit_dic['merit']['rms']='Undefined'
   
            #Merit values 
            if fit_dic['fit_mode'] =='fixed':fit_dic['merit']['mode']='forward'    
            else:fit_dic['merit']['mode']='fit'    
            fit_dic['merit']['dof']=fit_dic['nx_fit']-fit_dic['merit']['n_free']
            if fit_dic['fit_mode'] in ['fixed','chi2']: fit_dic['merit']['chi2']=np.sum(ln_prob_func_lmfit(p_final,fixed_args['x_val'], fixed_args=fixed_args)**2.)
            elif fit_dic['fit_mode'] =='mcmc': fit_dic['merit']['chi2']=ln_lkhood_func_mcmc(p_final,fixed_args)[1]  
            elif fit_dic['fit_mode'] =='ns': fit_dic['merit']['chi2']=ln_lkhood_func_ns(p_final,fixed_args)[1]['step_chi2'] 
            if fit_dic['merit']['chi2'] is not None:
                fit_dic['merit']['red_chi2']=fit_dic['merit']['chi2']/fit_dic['merit']['dof']
                fit_dic['merit']['BIC']=fit_dic['merit']['chi2']+fit_dic['merit']['n_free']*np.log(fit_dic['nx_fit'])      
                fit_dic['merit']['AIC']=fit_dic['merit']['chi2']+2.*fit_dic['merit']['n_free']
            else:
                fit_dic['merit']['chi2'] = 'Undefined'
                fit_dic['merit']['red_chi2'] = 'Undefined'
                fit_dic['merit']['BIC'] = 'Undefined'
                fit_dic['merit']['AIC'] = 'Undefined'
            if fit_dic['fit_mode'] in ['mcmc','ns']:fit_dic['merit']['GR_stat']=fit_dic['GR_stat']
            else:fit_dic['merit']['GR_stat']='N/A, MCMC run needed.'

            
            #Print fit statistics and results on screen
            if txt_print is not None:
                txt_print+=[["==============================================================================="],
                            ["Fit statistics"],
                            ["==============================================================================="],
                            [" "],
                            ['Mode : '+{'chi2':'Chi square','mcmc':'MCMC','ns':'Nested sampling','fixed':'Forward'}[fit_dic['fit_mode']]]]  
                if fit_dic['fit_mode']=='chi2':
                    txt_print+=[["Fitting method                = %r"%fit_dic['merit']['method']]]
                    txt_print+=[["Fit success                 = %r"%fit_dic['merit']['success']]]
                    if not fit_dic['merit']['success']:txt_print+=[["  " + fit_dic['merit']['message'][:-1]]]  
                    else:
                        if len(fit_dic['merit']['message'])>32:txt_print+=[["  " + fit_dic['merit']['message']]]  
                        elif (fit_dic['merit']['message'][:-1]!='Fit succeeded'):txt_print+=[["  " + fit_dic['merit']['message'][:-1]]]  
                    txt_print+=[["Function evals              = %i"%fit_dic['merit']['eval']]]    
                elif fit_dic['fit_mode'] in ['mcmc','ns']:
                    txt_print+=[
                        ["Walkers                     = "+str(fit_dic['nwalkers'])],
                        ["Burn-in steps               = "+str(fit_dic['nburn'])],
                        ["Steps (initial, per walker) = "+str(fit_dic['nsteps'])],
                        ["Steps (final, all walkers)  = "+str(fit_dic['nsteps_final_merged'])],
                    ]        
                def fmt_loc(val):
                    if type(val)==str:return val
                    else:return '%f'%val
                txt_print+=[
                    ["Duration                    = "+"{0:.4f}".format(fit_dic['fit_dur'])+' s'],
                    ['Data points                 = %i'%fit_dic['nx_fit']],
                    ['Free variables              = %i'%fit_dic['merit']['n_free']],
                    ['Degree of freedom           = %i'%fit_dic['merit']['dof']],
                    ['Best Chi-square             = '+fmt_loc(fit_dic['merit']['chi2'])],
                    ['Reduced Chi-square          = '+fmt_loc(fit_dic['merit']['red_chi2'])],
                    ['RMS of residuals            = '+fmt_loc(fit_dic['merit']['rms'])],
                    ['Bayesian Info. crit. (BIC)  = '+fmt_loc(fit_dic['merit']['BIC'])],
                    ['Akaike Info. crit. (AIC)    = '+fmt_loc(fit_dic['merit']['AIC'])],
                    ['Gelman-Rubin statistic      = %s'%fit_dic['merit']['GR_stat']],
                    ]
                if fit_dic['fit_mode']=='chi2':txt_print+=[["Cumul. dist. funct. (cdf)   = %e"%fit_dic['merit']['cdf']]]
                txt_print+=[[" "]]

    elif mode=='derived':
        p_final = deepcopy(p_final_in)

    if txt_print is not None:
        if (mode=='nominal') or (print_der):
            if (mode=='nominal'):
                txt_print+=[                
                    [''],    
                    ['==============================================================================='],    
                    ['Nominal parameters'],     
                    ['==============================================================================='],       
                    ['']]
                if len(fixed_args['fix_par_list'])>0:
                    max_len_fix = 0
                    for parname in fixed_args['fixed_par_val']:max_len_fix = max([max_len_fix,len(parname)]) 
                    txt_print+=[                
                        ['==========================================================='],    
                        ['Fixed'],
                        ['==========================================================='],    
                        [''],     
                        ['Name'+" "*(max_len_fix-len('Name'))+'\t'+'Value'+" "*16+'\t'+'Unit'],
                        ['-----------------------------------------------------------'],      
                        ['']]                 
                    for ipar,(parname,parunit) in enumerate(zip(fixed_args['fix_par_list'],fixed_args['fix_par_units'])):           
                        txt_print+=[[parname+" "*(max_len_fix-len(parname))+'\t'+"{0:.10e}".format(p_final[parname])+'\t'+'['+parunit+']']]                  
            elif print_der: 
                txt_print+=[          
                    [''],
                    ['==============================================================================='],
                    ['Derived parameters'],
                    ['===============================================================================']]    
            max_len_med = 0
            for parname in fixed_args['var_par_list']:max_len_med = max([max_len_med,len(parname)]) 
            txt_print+=[
                [''],
                ['==========================================================='],    
                ['Variable'],
                ['==========================================================='],    
                [''],                   
                ['Name'+" "*(max_len_med-len('Name'))+'\t'+'Median'+" "*16+'\t'+'Unit']]
            if 'sig_parfinal_err' in fit_dic:
                txt_print+=[['     1-sigma uncertainties from quantiles']]
                txt_print+=[['     Interval around median from 1-sigma quantiles']]
            if (fit_dic['fit_mode'] in ['mcmc','ns']) and (fit_dic['HDI'] is not None): 
                sig_txt = {'1s':'1-sigma','2s':'2-sigma','3s':'3-sigma'}[fit_dic['HDI']]
                txt_print+=[['     '+sig_txt+' uncertainties from HDI']]
                txt_print+=[['     Interval around median from '+sig_txt+' HDI']]     
                txt_print+=[['-----------------------------------------------------------']]    
    
            #Print variable parameters
            for ipar,(parname,parunit) in enumerate(zip(fixed_args['var_par_list'],fixed_args['var_par_units'])):
        
                #Only derived parameters not already printed as nominal ones are printed
                if (mode=='nominal') or print_der:
                    
                    #Median
                    nom_val = p_final[parname]
        
                    #Print median value
                    txt_print+=[['']]
                    txt_print+=[[parname+" "*(max_len_med-len(parname))+'\t'+"{0:.10e}".format(nom_val)+'\t'+'['+parunit+']']]
        
                    #Quantile uncertainties
                    if 'sig_parfinal_err' in fit_dic:
                        lower_sig= fit_dic['sig_parfinal_err']['1s'][0,ipar]
                        upper_sig= fit_dic['sig_parfinal_err']['1s'][1,ipar]  
                        txt_print+=[['     Quant. 1s err. = -'+"{0:.10e}".format(lower_sig)+'\t'+"+"+"{0:.10e}".format(upper_sig)]] 
                        txt_print+=[['     Quant. 1s int. = ['+"{0:.10e}".format(nom_val-lower_sig)+'\t'+"{0:.10e}".format(nom_val+upper_sig)+"]"]]
                    
                    #HDI (MCMC only)
                    if (fit_dic['fit_mode'] in ['mcmc','ns']):
                        if (fit_dic['HDI'] is not None):
                            txt_print+=[['     HDI '+fit_dic['HDI']+' err.    = '+fit_dic['HDI_sig_txt'][ipar]]]
                            txt_print+=[['     HDI '+fit_dic['HDI']+' int.    = '+fit_dic['HDI_interv_txt'][ipar]]]
                        if parname in fit_dic['conf_limits']:
                            for lev in fit_dic['conf_limits'][parname]['level']:
                                txt_print+=[['     '+fit_dic['conf_limits'][parname]['limits'][lev]]]         
    
        #Calculation of null model hypothesis
        #    - to calculate chi2 (=BIC) with respect to a null level for comparison of best-fit model with null hypothesis
        if (mode=='derived') and ('p_null' in fit_dic) and fit_dic['calc_merit']:
            if fit_dic['fit_mode'] in ['fixed','chi2']: chi2_null=np.sum(ln_prob_func_lmfit(fit_dic['p_null'], fixed_args['x_val'], fixed_args=fixed_args)**2.)
            elif fit_dic['fit_mode'] =='mcmc':chi2_null=ln_lkhood_func_mcmc(fit_dic['p_null'],fixed_args)[1]
            elif fit_dic['fit_mode'] =='ns':chi2_null=ln_lkhood_func_ns(fit_dic['p_null'],fixed_args)[1]['step_chi2']   
            txt_print+=[            
                [''],
                ['==============================================================================='],    
                ['Null hypothesis'],
                ['==============================================================================='],    
                [''],
                ['Chi-square               = '+"{0:.4f}".format(chi2_null)],
                ['(Null - Best) Chi-square = '+"{0:.4f}".format(chi2_null-fit_dic['merit']['chi2'])],
                ['']]

    #Print fit information on screen
    if verbose: 
        print('')
        for txt_line in txt_print:print(verb_shift+txt_line[0])         

    #Save fit information on disk
    if fit_dic['save_outputs']:
        file_path=fit_dic['file_save']
        for txt_line in txt_print:np.savetxt(file_path,[txt_line],delimiter='\t',fmt=['%s'])        

    return p_final
    
    
##################################################################################################
#%%%% Chi2 analysis
##################################################################################################   

def compute_Hessian(params, func, args, kwargs, epsilon=1e-5):
    r"""**Compute the Hessian matrix of a function using finite differences**
    
    Args:
        func: The function whose Hessian is to be computed. It should accept a Parameters object.
        params: The lmfit.Parameters object.
        epsilon: The small perturbation for finite differences (default: 1e-5).
        args: additional arguments required by the function used to compute the Hessian.
        kwargs: additional keyword parameters required by the function used to compute the Hessian.
    
    Returns:
        hessian_matrix: The Hessian matrix.
    """

    def params_to_array(params):
        """Convert lmfit.Parameters object to a NumPy array."""
        return np.array([params[key].value for key in params])

    def array_to_params(arr, params_template):
        """Convert a NumPy array back to lmfit.Parameters object."""
        params = lmfit.parameter.Parameters()
        for idx, key in enumerate(params_template.keys()):
            params.add(key, value=arr[idx])
        return params


    # Convert the parameters object to an array and find the indices of varying parameters
    x0 = params_to_array(params)
    varying_indices = [i for i, key in enumerate(params.keys()) if params[key].vary]
    num_varying_params = len(varying_indices)

    # Create a Hessian matrix for the varying parameters
    hessian_matrix = np.zeros((num_varying_params, num_varying_params), dtype=float)
    
    # Loop over each element to compute the second derivatives
    if num_varying_params == 1:
        # Compute the Hessian as the second derivative
        x_plus_plus = x0.copy()
        x_minus_minus = x0.copy()
        x_plus_plus[varying_indices[0]] += epsilon
        x_minus_minus[varying_indices[0]] -= epsilon

        p_plus_plus = array_to_params(x_plus_plus, params)
        p_minus_minus = array_to_params(x_minus_minus, params)

        f_plus_plus = np.sum(func(p_plus_plus, *args, **kwargs)**2)
        f_minus_minus = np.sum(func(p_minus_minus, *args, **kwargs)**2)
        f_original = np.sum(func(params, *args, **kwargs)**2)

        # Calculate the second derivative (Hessian) using finite differences
        hessian_matrix[0, 0] = (f_plus_plus - 2 * f_original + f_minus_minus) / (epsilon ** 2)

    else:
        for i in range(num_varying_params):
            for j in range(num_varying_params):
                
                # Construct perturbed arrays only for the varying parameters
                x_perturbed_ij_plus = x0.copy()
                x_perturbed_ij_minus = x0.copy()
                x_perturbed_i_plus_j_minus = x0.copy()
                x_perturbed_i_minus_j_plus = x0.copy()

                # Add/subtract epsilon only to/from the varying parameters
                x_perturbed_ij_plus[varying_indices[i]] += epsilon
                x_perturbed_ij_plus[varying_indices[j]] += epsilon

                x_perturbed_ij_minus[varying_indices[i]] -= epsilon
                x_perturbed_ij_minus[varying_indices[j]] -= epsilon

                x_perturbed_i_plus_j_minus[varying_indices[i]] += epsilon
                x_perturbed_i_plus_j_minus[varying_indices[j]] -= epsilon

                x_perturbed_i_minus_j_plus[varying_indices[i]] -= epsilon
                x_perturbed_i_minus_j_plus[varying_indices[j]] += epsilon


                p_ij_plus = array_to_params(x_perturbed_ij_plus, params)
                p_ij_minus = array_to_params(x_perturbed_ij_minus, params)
                p_i_plus_j_minus = array_to_params(x_perturbed_i_minus_j_plus, params)
                p_i_minus_j_plus = array_to_params(x_perturbed_i_minus_j_plus, params)
                
                f_ij_plus = np.sum(func(p_ij_plus, *args, **kwargs)**2)
                f_ij_minus = np.sum(func(p_ij_minus, *args, **kwargs)**2)
                f_i_plus_j_minus = np.sum(func(p_i_plus_j_minus, *args, **kwargs)**2)
                f_i_minus_j_plus = np.sum(func(p_i_minus_j_plus, *args, **kwargs)**2)

                
                hessian_matrix[i, j] = (f_ij_plus - f_i_plus_j_minus - f_i_minus_j_plus + f_ij_minus) / (4 * epsilon ** 2)
    
    return hessian_matrix

  

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

  
def MCMC_GR_stat(chains):
    r"""**MCMC post-proc: Gelman-Rubin statistic**

    Calculates the Gelman-Rubin statistic for an ensemble of chains, following the 
    implementation of Vats & Knudson 2020. In particular, we follow their approach of using
    batch means estimators and distinguishing two cases: univariate and multivariate MCMC runs.
    In practice, GR < 1.1 is used to declare convergence of chains.
      
    Args:
        chains (array): three-dimensional array of the chains from an MCMC run.
    
    Returns:
        GR (float): Gelman-Rubin statistic.
    
    """
    num_chains, num_steps, num_params = chains.shape
    batch_size = int(np.floor(num_steps**(1/3)))

    #Use the multivariate approach
    if num_params > 1:

        #Define arrays to store the average parameter values across all chains (C_star)
        #and the variances of parameters over each chain (S)
        S = np.zeros((num_chains, num_params, num_params), dtype=float)
        C_star = np.zeros(num_params, dtype=float)
        for j in range(num_chains):
            chain_j = chains[j,:,:]
            C_j = np.sum(chain_j, axis=0)/num_steps
            S[j,:,:] = (chain_j[0,:] - C_j).reshape(num_params,1)@(chain_j[0,:] - C_j).reshape(num_params,1).T
            for i in range(1, num_steps):
                S[j,:,:] += (chain_j[i,:] - C_j).reshape(num_params,1)@(chain_j[i,:] - C_j).reshape(num_params,1).T
            S[j,:,:] /= (num_steps-1)
            C_star += C_j
        
        #Calculate the average parameter values and variances across all chains (C_star, S_tot)
        C_star /= num_chains
        S_tot = np.sum(S, axis=0)/num_chains
        
        #Define a function to calculate the batch means estimator for varying batch sizes.
        def calc_Tb(b):
            a = int(num_steps/b)
            T = np.zeros((num_chains, num_params, num_params), dtype=float)
            for j in range(num_chains):
                for k in range(a):
                    chain_j_k = chains[j,k*b : (k+1)*b,:]
                    Y = np.sum(chain_j_k, axis=0)/b
                    T[j, :, :] += (Y - C_star).reshape(num_params,1)@(Y - C_star).reshape(num_params,1).T
            T_tot = np.sum(T, axis=0) * (b/(a*num_chains - 1))
            return T_tot

        #Final step to calculate the GR statistic
        T_L = 2*calc_Tb(batch_size) - calc_Tb(int(np.floor(batch_size/3)))
        GR = np.sqrt( ((num_steps-1)/num_steps) + ((1/num_steps) * (np.linalg.det(T_L)/np.linalg.det(S_tot))**(1/num_params)))

    #Use the univariate approach
    else: 

        #Define arrays to store the average parameter value of each chain (x)
        #and the corresponding variance for each chain (s)
        s = np.zeros((num_chains,), dtype=float)
        x = np.zeros((num_chains,), dtype=float)
        for j in range(num_chains):
            x[j] = np.sum(chains[j,:,:], axis=0)/num_steps
            s[j] = np.sum((chains[j,:,:] - x[j])**2, axis=0)/(num_steps-1)
        
        #Calculate the average parameter value and variance across all chains (mu, s_tot)
        mu = np.sum(x)/num_chains
        s_tot = np.sum(s)/num_chains

        #Define a function to calculate the batch means estimator for varying batch sizes.
        def calc_tb(b):
            a = int(num_steps/b)
            t = np.zeros((num_chains,), dtype=float)
            for j in range(num_chains):
                for k in range(a):
                    y = np.sum(chains[j, k*b:(k+1)*b, :], axis=0)/b
                    t[j] += (y - mu)**2
            t_tot = np.sum(t) * (b/(a*num_chains - 1))
            return t_tot

        #Final step to calculate the GR statistic
        t_L = 2*calc_tb(batch_size) - calc_tb(int(np.floor(batch_size/3)))
        sigma_L = ((num_steps - 1)/num_steps) * s_tot + (t_L / num_steps)
        GR = np.sqrt(sigma_L / s_tot)
    return GR
  

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
    
    if (calc_envMCMC) or (calc_sampMCMC):        
      
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
    if calc_sampMCMC:
        
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
  
  
  
def MCMC_HDI(chain_par,nbins_par,dbins_par,bw_fact,frac_search,HDI_interv_par,HDI_interv_txt_par,HDI_sig_txt_par,med_par,use_arviz=False):
    r"""**MCMC post-proc: HDI intervals**

    Calculates Highest Density Intervals of fitted MCMC parameters.
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """  
    #Use arviz approach
    if use_arviz:jumpind=[]
    
    #Use custom approach    
    else:
        
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
        if use_arviz:HDI_sub = arviz.hdi(chain_par, hdi_prob=frac_search)
        else:HDI_sub=[np.min(bin_edges_par[bins_in_HDI]),np.max(bin_edges_par[bins_in_HDI+1])]
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


 
def MCMC_estimates(merged_chain,fixed_args,fit_dic,verbose=True,calc_quant=True,verb_shift=''):
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
            HDI_interv_txt[ipar],HDI_frac[ipar],HDI_sig_txt_par[ipar]=MCMC_HDI(merged_chain[:,ipar],nbins_par,dbins_par,bw_fact,frac_search,HDI_interv[ipar],HDI_interv_txt[ipar],HDI_sig_txt_par[ipar],med_par[ipar],use_arviz = fit_dic['use_arviz'])
            
        #Convert into array
        HDI_interv=np.array(HDI_interv,dtype=object)   
          
    else:
        HDI_interv=None
        HDI_interv_txt=None
        HDI_sig_txt_par = None
            
    #----------------------------------------------------        

    #Print chain information
    if verbose:
        print(verb_shift+'-------------------------------') 
        print(verb_shift+'> Chain properties')         
        print(verb_shift+"   Covariance: "+str(np.cov(np.transpose(merged_chain))))
        print(verb_shift+"   Coefficient correlations: "+str(np.corrcoef(np.transpose(merged_chain))))

    return p_best,med_par,sig_par_val,sig_par_err,HDI_interv,HDI_interv_txt,HDI_sig_txt_par  
    
    

   
def postMCMCwrapper_1(fit_dic,fixed_args,walker_chains,step_outputs,nthreads,par_names,verbose=True,verb_shift=''):    
    r"""**MCMC post-proc: raw chains**

    Processes and analyzes MCMC chains of original parameters.
    Returns :
        
        - best-fit parameters for model calculation
        - 1-sigma and envelope samples for plot
        - plot of model parameter chains
        - save file 
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """
    #Burn-in
    fit_dic['nburn']=int(fit_dic['nburn'])    

    #Remove burn-in steps from chains of variable parameters
    #    - walker_chains is of shape (nwalkers, nsteps, n_free)
    #    - step_outputs is of shape (nsteps,nwalkers)
    n_free = (walker_chains.shape)[2]
    burnt_chains = walker_chains[:, fit_dic['nburn']:, :]    
    if fixed_args['step_output']:burnt_outputs = step_outputs[fit_dic['nburn']:]    

    #Number of post burn-in points in each chain
    fit_dic['nsteps_pb_walk']=fit_dic['nsteps']-fit_dic['nburn'] 

    #Automatic exclusion of chains
    if fit_dic['exclu_walk_autom'] is not None:
        print(verb_shift+'> Automatic chain clipping')
        
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

    #Plot the chi2 chain of each walker
    if (fit_dic['save_chi2_chains']!=''):
        MCMC_plot_chains_chi2(fit_dic,walker_chains,keep_chain,fixed_args,verbose=verbose,verb_shift=verb_shift)
        
    #Remove chains if required
    if (False in keep_chain):
        print(verb_shift+'  '+str(np.sum(~keep_chain))+' chains removed')
        burnt_chains = burnt_chains[keep_chain]
        if fixed_args['step_output']:burnt_outputs = burnt_outputs[:,keep_chain]
        fit_dic['nwalkers']=np.sum(keep_chain)  
        
    #Calculate Gelman-Rubin statistic
    # We define the batch size as x^(1/3) with x the number of production steps. We then shorten the batch size to x^(1/3)/3.
    # Therefore, in order to have a batch size >0 we need x^(1/3)/3 > 1 i.e. x>27.
    if burnt_chains.shape[1] < 27.: 
        print(verb_shift+'WARNING: Not enough MCMC steps to compute Gelman-Rubin statistic. Need at least 27 production steps.')
        fit_dic['GR_stat'] = ' N/A'
    else:fit_dic['GR_stat'] = MCMC_GR_stat(burnt_chains)
        
    #Merge chains
    #    - we reshape into (nwalkers*(nsteps-nburn) , n_free)        
    merged_chain = burnt_chains.reshape((-1, n_free))  
    if fixed_args['step_output']:merged_outputs = burnt_outputs.reshape((-1))  
    else:merged_outputs=None

    #Manual exclusion of samples
    if len(fit_dic['exclu_samp'])>0:
        cond_keep = False
        for par_loc in fit_dic['exclu_samp']:
            ipar_loc = np_where1D(fixed_args['var_par_list']==par_loc)
            if len(ipar_loc)>0:
                for bd_int in fit_dic['exclu_samp'][par_loc]:
                    cond_keep |= (merged_chain[:,ipar_loc[0]]>=bd_int[0]) & (merged_chain[:,ipar_loc[0]]<=bd_int[1]) 
        merged_chain = merged_chain[cond_keep]
        if fixed_args['step_output']:merged_outputs = merged_outputs[cond_keep]
 
    #Number of points remaining in the merged chain     
    fit_dic['nsteps_final_merged']=len(merged_chain[:,0])

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
    p_final,fit_dic['med_parfinal'],fit_dic['sig_parfinal_val'],fit_dic['sig_parfinal_err'],fit_dic['HDI_interv'],fit_dic['HDI_interv_txt'],fit_dic['HDI_sig_txt']=MCMC_estimates(merged_chain,fixed_args,fit_dic,verbose=verbose,calc_quant=fit_dic['calc_quant'],verb_shift=verb_shift)

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
    
    return p_final,merged_chain,merged_outputs,par_sample_sig1,par_sample
    
    
def postMCMCwrapper_2(fit_dic,fixed_args,merged_chain,sim_points = None):
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
    p_final,fit_dic['med_parfinal'],fit_dic['sig_parfinal_val'],fit_dic['sig_parfinal_err'],fit_dic['HDI_interv'],fit_dic['HDI_interv_txt'],fit_dic['HDI_sig_txt']=MCMC_estimates(merged_chain,fixed_args,fit_dic,verbose=False,calc_quant=fit_dic['calc_quant'],verb_shift=fit_dic['verb_shift'])

    #Save merged chains for derived parameters and various estimates
    if fit_dic['save_results']:
        data_save = {'merged_chain':merged_chain,'HDI_interv':fit_dic['HDI_interv'],'sig_parfinal_val':fit_dic['sig_parfinal_val']['1s'],'var_par_list':fixed_args['var_par_list'],'var_par_names':fixed_args['var_par_names'],'med_parfinal':fit_dic['med_parfinal']}
        if fit_dic['fit_mode']=='mcmc':np.savez(fit_dic['save_dir']+'merged_deriv_chains_walk'+str(fit_dic['nwalkers'])+'_steps'+str(fit_dic['nsteps'])+fit_dic['run_name'],data=data_save,allow_pickle=True)
        elif fit_dic['fit_mode']=='ns':np.savez(fit_dic['save_dir']+'merged_deriv_chains_live'+str(fit_dic['nlive'])+fit_dic['run_name'],data=data_save,allow_pickle=True)

    #Corner plots
    #    - for samples and/or simulations
    corner_options_dic={}
    if (fit_dic['save_MCMC_corner']!=''):corner_options_dic['samples']=fit_dic['corner_options'] 
    if (fit_dic['save_sim_points_corner']!=''):corner_options_dic['sims']=fit_dic['sim_corner_options']  
    for key in corner_options_dic:
        
        #Default options
        corner_options = corner_options_dic[key]
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
        truth_color="#4682b4"  if 'best_color' not in corner_options else corner_options['best_color']
        color_1D_HDI="green"  if 'color_1D_HDI' not in corner_options else corner_options['color_1D_HDI']
        if 'fontsize' in corner_options:
             label_kwargs={'fontsize':corner_options['fontsize']}
             tick_kwargs={'labelsize':corner_options['fontsize']}
        else:
             label_kwargs=None
             tick_kwargs=None   

        #Reduce to required parameters
        var_par_list = np.array(fixed_args['var_par_list'])
        var_par_names = np.array(fixed_args['var_par_names'])
        if 'plot_par' in corner_options:
            ikept = []
            for par_loc in corner_options['plot_par']:
                ipar = np_where1D(var_par_list==par_loc)
                if len(ipar)>0:ikept+=[ipar[0]]
                else:stop('Parameter '+par_loc+' was not fitted.')
            if len(ikept)==0:stop('No parameters kept in corner plot')
            var_par_list = var_par_list[ikept]
            var_par_names = var_par_names[ikept] 
            if key=='samples':merged_chain = merged_chain[:,ikept] 
            if (sim_points is not None):sim_points = sim_points[:,ikept] 
            if best_val is not None:best_val = best_val[ikept] 
            fit_dic['HDI_interv'] = fit_dic['HDI_interv'][ikept]              

        #Remove constant parameters
        for par_loc in var_par_list:
            ipar = np_where1D(var_par_list==par_loc)[0]
            if np.min(merged_chain[:,ipar])==np.max(merged_chain[:,ipar]):
                var_par_list = np.delete(var_par_list,ipar)
                var_par_names = np.delete(var_par_names,ipar)
                if key=='samples':merged_chain = np.delete(merged_chain,ipar,axis=1) 
                if (sim_points is not None):sim_points = np.delete(sim_points,ipar,axis=1) 
                if best_val is not None:best_val = best_val[ikept] 
                fit_dic['HDI_interv'] = np.delete( fit_dic['HDI_interv'],ipar,axis=0)              

        #Simulation points
        if key=='samples':
            if (('plot_sim' not in corner_options) or (not corner_options['plot_sim']) or (sim_points is None)):
                 sim_points_sec = None
            else:sim_points_sec = sim_points
        elif key=='sims':sim_points_sec = None

        #Plot
        if key=='samples':
            save_fmt = fit_dic['save_MCMC_corner']
            save_name = 'Corr_diag'
            corner_data = merged_chain
            levels = (0.39346934028,0.86466471)
        elif key=='sims':
            save_fmt = fit_dic['save_sim_points_corner']
            save_name = 'Corner_sims'
            corner_data = sim_points
            levels = None
            smooth2D = None
        MCMC_corner_plot(key,save_fmt,save_name,fit_dic['save_dir'],corner_data,fit_dic['HDI_interv'],
                         sim_points = sim_points_sec,
                         labels_raw = var_par_list,
                         labels=var_par_names,
                         truths=best_val,
                         truth_color=truth_color,
                         color_1D_HDI=color_1D_HDI,
                         bins_1D_par=bins_1D_par,
                         bins_2D_par=bins_2D_par,
                         range_par=range_par,
                         major_int=major_int,
                         minor_int=minor_int,
                         levels=levels,
                         color_levels=color_levels,
                         smooth1d = None,
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


def MCMC_plot_chains_chi2(fit_dic,chain,keep_chain,fixed_args,verbose=True,verb_shift=''):
    r"""**MCMC post-proc: walker chains' chi2 plot**

    Plots the chi2 values for the chain of each walker. 
      
    Args:
        TBD
    
    Returns:
        TBD
    
    """
    if verbose:
        print(verb_shift+'-----------------------------------')
        print(verb_shift+'> Plotting chi2 chains')

    #Font size
    font_size=14
    
    #Plot size
    margins=[0.15,0.15,0.95,0.8] 
        
    #Linewidth
    lw_plot=0.1

    #----------------------------------------------------------------
    if verbose:
        print(verb_shift+'   + Retrieval')

    #Retrieving important values
    nwalkers,nsteps,_ = chain.shape

    chi2_chains = fixed_args['chi2_storage']
    if chi2_chains is None:stop('Chi2 chains were not stored. Make sure fixed_args[\'step_chi2\'] is turned on.')
    chi2_chains = chi2_chains.T

    #----------------------------------------------------------------
    if verbose:
        print(verb_shift+'   + Global Plotting')

    #Plot the chi2 chains of all walkers
    plt.ioff()        
    plt.figure(figsize=(10, 6))
   
    #Chi2 chains with burn-in phase, and removed chains
    for iwalk,keep_chain_loc in enumerate(keep_chain):
        if keep_chain_loc:
            x_tab=range(fit_dic['nburn'])
            plt.plot(x_tab,chi2_chains[iwalk,x_tab],color='red',linestyle='-',lw=lw_plot,zorder=0)                
            x_tab=fit_dic['nburn']+np.arange(nsteps-fit_dic['nburn'],dtype=int)
            plt.plot(x_tab,chi2_chains[iwalk,x_tab],color='dodgerblue',linestyle='-',lw=lw_plot,zorder=0)                           
        else:
            plt.plot(np.arange(nsteps,dtype=int),chi2_chains[iwalk, :],color='red',linestyle='-',lw=lw_plot,zorder=0) 
           
    #Plot frame  
    plt.title('Chain for Chi2')
    y_min = np.min(chi2_chains)
    y_max = np.max(chi2_chains)
    dy_range = y_max-y_min
    y_range = [y_min-0.05*dy_range,y_max+0.05*dy_range]    
    ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)
    custom_axis(plt,position=margins,
                y_range=y_range,dir_y='out', 
                ymajor_int=ymajor_int,yminor_int=yminor_int,ymajor_form=ymajor_form,
                x_title='Steps',y_title='Chi2',
                font_size=font_size,xfont_size=font_size,yfont_size=font_size)
    plt.yscale('log')

    plt.savefig(fit_dic['save_dir']+'/Chain_Chi2.'+fit_dic['save_MCMC_chains']) 
    plt.close()
    
    #----------------------------------------------------------------
    if nwalkers > 1.:
        if verbose:
            print(verb_shift+'   + Individual Plotting')
        
        #Plot the chi2 chains of individual walkers

        #Make directory to store the distributions
        if (not os_system.path.exists(fit_dic['save_dir']+'Indiv_Chi2_Chains')):os_system.makedirs(fit_dic['save_dir']+'Indiv_Chi2_Chains')   
        
        #Each plot will have a maximum of 10 chains (to not overload the visuals)
        n_group = 10

        #Create groups of chains
        for iwalk in range(0, nwalkers, n_group):
            
            #Define chain group
            end = min(iwalk + n_group, nwalkers)
            walker_group = chi2_chains[iwalk:end, :]

            #In each subplot highlight one chain with greyscale
            plt.ioff()        
            fig, axes = plt.subplots(n_group, 1, figsize=(8, 12), sharex=True)

            for i, walker in enumerate(walker_group):
                ax = axes[i]
                ax.plot(walker, color='black', lw = lw_plot+1, zorder=1)

                for otheri, otherwalker in enumerate(walker_group):
                    if otheri!=i: ax.plot(otherwalker, color='black', alpha=0.8, lw=lw_plot, zorder=0)
               
                ax.yaxis.set_visible(False)
                ax.set_yscale('log')

            # Turn off unused subplots if the walker_group is smaller than n_group
            if len(walker_group) < n_group:
                for j in range(len(walker_group), n_group):
                    axes[j].set_visible(False)

            #Plot frame  
            axes[0].set_title('Individual Chi2 chains '+str(iwalk)+'-'+str(end), fontsize=font_size)
            axes[len(walker_group)-1].set_xlabel('Steps', fontsize=font_size)
            axes[len(walker_group)-1].tick_params(axis='x', which='both', labelbottom=True)

            plt.savefig(fit_dic['save_dir']+'Indiv_Chi2_Chains/Chain_Chi2_'+str(iwalk)+'-'+str(end)+'.'+fit_dic['save_MCMC_chains']) 
            plt.close()



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

    

    
def MCMC_corner_plot(mode,save_mode,save_name,save_dir_MCMC,xs,HDI_interv, 
                     sim_points = None, labels_raw = None,labels=None,truths=None,truth_color="#4682b4",bins_1D_par=20,bins_2D_par=20,quantiles=[0.15865525393145703,0.841344746068543],
                     plot1s_1D=True,plot_HDI=False,color_1D_quant='darkorange',color_1D_HDI='green',levels=(0.39346934028,0.86466471,0.988891003),  
                     plot_contours = True,color_levels='black',use_math_text=True,range_par=None,smooth1d=None,smooth2D=None,weights=None, color="k",  
                     label_kwargs=None,tick_kwargs=None,show_titles=False,title_fmt=".2f",title_kwargs=None,scale_hist=False, 
                     verbose=False, fig=None,major_int=None,minor_int=None,max_n_ticks=5, top_ticks=False, hist_kwargs=None, **hist2d_kwargs):
    r"""**MCMC post-proc: corner plot**

    Plots correlation diagram, showing the projections of a data set in a multi-dimensional space. kwargs are passed to MCMC_corner_plot_hist2d() or used for `matplotlib` styling.
    If mode = 'sims' the plot is used to display density histograms of simulated points, rather than corner plot of MCMC samples
      
    Args:
        mode (str): use of plot
        save_mode (str): extension of the figure (png, pdf, jpg) 
        save_dir_MCMC (str): path of the directory in which the figure is saved
        xs (array, float): Samples. This should be a 1- or 2-dimensional array with dimensions [nsamples, ndim]. 
                           For a 1-D array this results in a simple histogram. 
                           For a 2-D array, the zeroth axis is the list of samples and the next axis are the dimensions of the space.
        HDI_interv (array, object): HDI intervals for each parameter

        sim_points (array, float): model points. This should be a 1- or 2-dimensional array with dimensions [npoints, ndim]. 
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
        range_par (list or dict): each element is either a length 2 tuple containing lower and upper bounds, or a float in (0., 1.) giving the fraction of samples to include in bounds, e.g.,
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
        if mode=='samples':print(' > Plot MCMC correlation diagram')
        elif mode=='sims':print(' > Plot simulations corner plot')
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
            else:print('WARNING : '+par+' set in "range_par" not in labels')
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
            else:stop('Parameter "'+par+'" not in fitted list.')
    if type(bins_2D_par) in [int,float]:bins_2D_par=np.repeat(bins_2D_par,npar).astype(int)
    elif type(bins_2D_par)==dict:
        if labels_raw is None:raise ValueError("Parameters must be named to be set in `range_par`")
        bins_2D_par_in = deepcopy(bins_2D_par)
        bins_2D_par = np.repeat(20,npar).astype(int)
        for par in bins_2D_par_in:
            idx_par = np_where1D(labels_raw==par)
            if len(idx_par)>0:bins_2D_par[idx_par[0]] = bins_2D_par_in[par] 
            else:stop('Parameter "'+par+'" not in fitted list.')

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
            
        # Plot 1D histograms.
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

        #Plot model points
        if sim_points is not None:
            for x_sim in sim_points[:,i]:ax.axvline(x_sim, color='grey')

        #Replace quantiles with HDI 
        if (plot_HDI):
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
            MCMC_corner_plot_hist2d(mode , y, x, j , i , ax=ax, range_par=[range_par[j], range_par[i]], weights=weights,
                   color=color,levels=levels,color_levels=color_levels,rasterized=True, 
                   smooth=[smooth2D[j],smooth2D[i]], bins_2D=[bins_2D_par[j], bins_2D_par[i]],sim_points=sim_points,
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

    plt.savefig(save_dir_MCMC+'/'+save_name+'.'+save_mode) 
    plt.close()     

    return None
    
    
    
    





def MCMC_corner_plot_hist2d(mode,x, y, i , j , bins_2D=[20,20], range_par=None, weights=None, levels=None, smooth=None,
           ax=None, color=None, plot_datapoints=True, plot_density=True,
           plot_contours=True, no_fill_contours=False, fill_contours=False,
           contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,color_levels='black',
           sim_points = None, 
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
        
    # Plot model points
    if sim_points is not None:
        x_sim_points = sim_points[:,i]
        y_sim_points = sim_points[:,j]
        plt.plot(x_sim_points,y_sim_points,marker='o',markersize=2,markerfacecolor='grey',markeredgecolor='white')

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
    
    
 
    
