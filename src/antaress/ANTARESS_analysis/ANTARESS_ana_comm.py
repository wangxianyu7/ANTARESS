#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from copy import deepcopy
from lmfit import Parameters
import lmfit
import numpy as np
import bindensity as bind
from ..ANTARESS_general.utils import stop,np_where1D,npint,dataload_npz,gen_specdopshift,closest,def_edge_tab,check_data
from ..ANTARESS_general.minim_routines import init_fit,call_MCMC,postMCMCwrapper_1,postMCMCwrapper_2,save_fit_results,fit_merit,call_lmfit,gen_hrand_chain
from ..ANTARESS_general.constant_data import Rsun,c_light
from ..ANTARESS_grids.ANTARESS_star_grid import calc_CB_RV,get_LD_coeff
from ..ANTARESS_grids.ANTARESS_occ_grid import sub_calc_plocc_spot_prop,up_plocc_prop
from ..ANTARESS_grids.ANTARESS_prof_grid import init_custom_DI_par,init_custom_DI_prof,custom_DI_prof,theo_intr2loc,def_Cfunc_prof
from ..ANTARESS_analysis.ANTARESS_model_prof import para_cust_mod_true_prop,proc_cust_mod_true_prop,cust_mod_true_prop,gauss_intr_prop,calc_biss,\
    dgauss,gauss_poly,voigt,gauss_herm_lin,gen_fit_prof
from ..ANTARESS_analysis.ANTARESS_inst_resp import convol_prof,def_st_prof_tab,cond_conv_st_prof_tab,resamp_st_prof_tab,get_FWHM_inst,calc_FWHM_inst
from ..ANTARESS_grids.ANTARESS_coord import excl_plrange


##################################################################################################    
#%%% Definition functions
################################################################################################## 

def par_formatting(p_start,model_prop,priors_prop,fit_dic,fixed_args,inst,vis,line_type):
    r"""**Parameter formatting**

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
        if (fit_dic['fit_mode']!=''):fit_used=True
    else:par_struct = False

    #Process default / additional parameters
    fixed_args['varpar_priors']={}
    for par in np.unique( list(p_start.keys()) + list(model_prop.keys())  ):  
        
        #Activate jitter if requested as parameter
        if par=='jitter':fixed_args['jitter'] = True

        #Overwrite default properties 
        if (par in model_prop):
            
            #Check property is used for current instrument
            #    - this is to avoid modifying a property for other instruments than the one(s) it is set up with in model_prop
            #    - except for planet/star properties independent of a dataset (defined as 'physical')
            if (inst in par) or ('__IS' in par) or ((inst in model_prop[par]) and (vis in model_prop[par][inst])) or (('physical' in model_prop[par]) and (model_prop[par]['physical'] is True)):

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
            
            #Chi2 fit
            #    - overwrite default priors
            if (fit_dic['fit_mode']=='chi2'):
                if (par in priors_prop) and (priors_prop[par]['mod']=='uf'):
                    if ('ang' in par) and (priors_prop[par]['high']>90):stop('Prior error: Spot angular size cannot exceed 90 deg. Re-define your priors.')
                    elif ('veq' in par) and (priors_prop[par]['low']<0):stop('Prior error: Cannot have negative stellar rotation velocity. Re-define your priors.')
                    elif ('Tcenter' in par) and ((priors_prop[par]['low'] <= p_start[par].value - fixed_args['Peq']) or (priors_prop[par]['high'] >= p_start[par].value + fixed_args['Peq'])):stop('Prior error: Spot crossing time priors should be less/more than the rotational period to avoid aliases.')
                    p_start[par].min = priors_prop[par]['low']
                    p_start[par].max = priors_prop[par]['high']

                #Change guess value if beyond prior range
                if (not np.isinf(p_start[par].min)) and (np.isinf(p_start[par].max)) and (p_start[par].value<p_start[par].min):p_start[par].value=p_start[par].min
                if (np.isinf(p_start[par].min)) and (not np.isinf(p_start[par].max)) and (p_start[par].value>p_start[par].max):p_start[par].value=p_start[par].max
                if (not np.isinf(p_start[par].min)) and (not np.isinf(p_start[par].max)) and ((p_start[par].value<p_start[par].min) or (p_start[par].value>p_start[par].max)):p_start[par].value=0.5*(p_start[par].min+p_start[par].max)

            #MCMC fit
            elif (fit_dic['fit_mode']=='mcmc'):
                
                #Range for walkers initialization
                if (par in model_prop):fit_dic['uf_bd'][par]=model_prop_par['bd']
                else:
                    uf_bd=[-1e6,1e6]
                    if (not np.isinf(p_start[par].min)):uf_bd[0]=p_start[par].min
                    if (not np.isinf(p_start[par].max)):uf_bd[1]=p_start[par].max
                    fit_dic['uf_bd'][par]=uf_bd
                if 'ang' in par and fit_dic['uf_bd'][par][1]>90:fit_dic['uf_bd'][par][1]=90
                elif 'veq' in par and fit_dic['uf_bd'][par][0]<0:fit_dic['uf_bd'][par][0]=0
                elif ('Tcenter' in par) and ((fit_dic['uf_bd'][par][0] <= p_start[par].value - fixed_args['Peq']) or (fit_dic['uf_bd'][par][1] >= p_start[par].value + fixed_args['Peq'])):fit_dic['uf_bd'][par] = [p_start[par].value - fixed_args['Peq'] + 0.001, p_start[par].value + fixed_args['Peq'] - 0.001]
                
                #Priors
                if (par in priors_prop):
                    fixed_args['varpar_priors'][par] = priors_prop[par]    
                    if 'ang' in par and priors_prop[par]['high']>90:stop('Prior error: Spot angular size cannot exceed 90deg. Re-define your priors.')
                    elif 'veq' in par and priors_prop[par]['low']<0:stop('Prior error: Cannot have negative stellar rotation velocity. Re-define your priors.')
                    elif ('Tcenter' in par) and ((priors_prop[par]['low'] <= p_start[par].value - fixed_args['Peq']) or (priors_prop[par]['high'] >= p_start[par].value + fixed_args['Peq'])):stop('Prior error: Spot crossing time priors should be less/more than the rotational period to avoid aliases.')
                else:
                    if par == 'jitter':varpar_priors=[0.,1e6]
                    elif par == 'veq':varpar_priors=[0.,100.]
                    elif 'ang' in par:varpar_priors=[0.,90.]
                    elif 'Tcenter' in par:varpar_priors=[p_start[par].value - fixed_args['Peq'] + 0.001, p_start[par].value + fixed_args['Peq'] - 0.001]
                    else:varpar_priors=[-1e6,1e6]
                    if (not np.isinf(p_start[par].min)):varpar_priors[0]=p_start[par].min
                    if (not np.isinf(p_start[par].max)):varpar_priors[1]=p_start[par].max                
                    fixed_args['varpar_priors'][par]={'mod':'uf','low':varpar_priors[0],'high':varpar_priors[1]}
             
                #Change if guess value is beyond prior range
                if fixed_args['varpar_priors'][par]['mod']=='uf':
                    if ((p_start[par].value<fixed_args['varpar_priors'][par]['low']) or (p_start[par].value>fixed_args['varpar_priors'][par]['high'])):p_start[par].value=0.5*(fixed_args['varpar_priors'][par]['low']+fixed_args['varpar_priors'][par]['high'])
                    if (fit_dic['uf_bd'][par][0]<fixed_args['varpar_priors'][par]['low']):fit_dic['uf_bd'][par][0]=fixed_args['varpar_priors'][par]['low']
                    if (fit_dic['uf_bd'][par][1]>fixed_args['varpar_priors'][par]['high']):fit_dic['uf_bd'][par][1]=fixed_args['varpar_priors'][par]['high']

    #---------------------------------------------------

    #Associate the correct input names for model functions with instrument and visit dependence
    #     - properties must be defined in p_start as 'propN__ISx_VSy'
    #  + N is the degree of the coefficient (if relevant)
    #  + x is the id of the instrument
    #  + y is the id of the visit associated with the instrument
    #     - x and y can be set to '_' so that the coefficient is common to all instruments and/or visits
    fixed_args['name_prop2input'] = {}
    fixed_args['coeff_ord2name'] = {}
    fixed_args['linevar_par'] = {}

    #Retrieve all root name parameters with instrument/visit dependence, possibly with several orders (polynomial degree, or list)
    par_list=[]
    root_par_list=[]
    fixed_args['genpar_instvis']  = {}
    if 'inst_list' not in fixed_args:fixed_args['inst_list']=[inst]
    if 'inst_vis_list' not in fixed_args:fixed_args['inst_vis_list']={inst:[vis]}
    for par in p_start:
 
        #Parameter depends on instrument and visit
        if ('__IS') and ('_VS') in par:
            root_par = par.split('__IS')[0]
            inst_vis_par = par.split('__IS')[1]
            inst_par  = inst_vis_par.split('_VS')[0]
            vis_par  = inst_vis_par.split('_VS')[1]              
            if root_par not in fixed_args['genpar_instvis'] :fixed_args['genpar_instvis'][root_par] = {}          
            if inst_par not in fixed_args['genpar_instvis'][root_par]:fixed_args['genpar_instvis'][root_par][inst_par]=[]
            if vis_par not in fixed_args['genpar_instvis'][root_par][inst_par]:fixed_args['genpar_instvis'][root_par][inst_par]+=[vis_par]                  
            root_par_list+=[root_par]            
            par_list+=[par]

            #Parameter vary as polynomial of spatial stellar coordinate
            if ('_ord' in par):
                gen_root_par = par.split('_ord')[0] 
                
                #Define parameter for current instrument and visit (if specified) or all instruments and visits (if undefined) 
                if inst_par in fixed_args['inst_list']:inst_list = [inst_par]
                elif inst_par=='_':inst_list = fixed_args['inst_list'] 
                for inst_loc in inst_list:
                    if inst_loc not in fixed_args['coeff_ord2name']:fixed_args['coeff_ord2name'][inst_loc] = {}
                    if vis_par in fixed_args['inst_vis_list'][inst_loc]:vis_list = [vis_par]
                    elif vis_par=='_':vis_list = fixed_args['inst_vis_list'][inst_loc]              
                    for vis_loc in vis_list:
                        if vis_loc not in fixed_args['coeff_ord2name'][inst_loc]:fixed_args['coeff_ord2name'][inst_loc][vis_loc] = {}                
                        if gen_root_par not in fixed_args['coeff_ord2name'][inst_loc][vis_loc]:fixed_args['coeff_ord2name'][inst_loc][vis_loc][gen_root_par]={}
    
                        #Identify stellar line properties with polynomial spatial dependence 
                        if gen_root_par in ['ctrst','FWHM','amp_l2c','rv_l2c','FWHM_l2c','a_damp','rv_line']:
                            if (line_type!='ana') and (gen_root_par!='rv_line'):stop('Cannot use parameter '+gen_root_par+'with line model of type '+line_type)
                            if inst_loc not in fixed_args['linevar_par']:fixed_args['linevar_par'][inst_loc]={}
                            if vis_loc not in fixed_args['linevar_par'][inst_loc]:fixed_args['linevar_par'][inst_loc][vis_loc]=[]
                            if gen_root_par not in fixed_args['linevar_par'][inst_loc][vis_loc]:fixed_args['linevar_par'][inst_loc][vis_loc]+=[gen_root_par]                     

    #Process parameters with dependence on instrument/visit
    for root_par in np.unique(root_par_list):

        #Parameter is associated with order coefficient
        if ('_ord' in root_par):
            deg_coeff=int(root_par[-1])
            gen_root_par = root_par.split('_ord')[0]  
        else:deg_coeff=None
         
        #Property common to all instruments is also common to all visits
        #    - we associate the property for current instrument and visit to the property common to all instruments and visits
        if any([root_par+'__IS__' in str_loc for str_loc in par_list]):
            for inst in fixed_args['inst_list']:
                for vis in fixed_args['inst_vis_list'][inst]:
                    par_input = root_par+'__IS'+inst+'_VS'+vis                    
                    fixed_args['name_prop2input'][par_input] = root_par+'__IS__VS_'
                    if deg_coeff is not None:
                        fixed_args['coeff_ord2name'][inst][vis][gen_root_par][deg_coeff]=root_par+'__IS__VS_' 
            
        #Property is specific to a given instrument
        else:

            #Process all fitted instruments associated with the property
            for inst in fixed_args['inst_list']:
                if (inst in fixed_args['genpar_instvis'][root_par]):

                    #Property is common to all visits of current instrument
                    #    - we associate the property for current instrument and visit to the value specific to this instrument, common to all visits
                    if any([root_par+'__IS'+inst+'_VS_' in str_loc for str_loc in par_list]): 
                        for vis in fixed_args['inst_vis_list'][inst]:
                            par_input = root_par+'__IS'+inst+'_VS'+vis 
                            fixed_args['name_prop2input'][par_input] = root_par+'__IS'+inst+'_VS_'  
                            if deg_coeff is not None:fixed_args['coeff_ord2name'][inst][vis][gen_root_par][deg_coeff]=root_par+'__IS'+inst+'_VS_'      
        
                    #Property is specific to the visit
                    #    - we associate the property for current instrument and visit to the value specific to this instrument, and this visit
                    else:
                        
                        #Process all fitted visits associated with the property
                        for vis in fixed_args['inst_vis_list'][inst]:
                            if (vis in fixed_args['genpar_instvis'][root_par][inst]):   
                                par_input = root_par+'__IS'+inst+'_VS'+vis 
                                fixed_args['name_prop2input'][par_input] = root_par+'__IS'+inst+'_VS'+vis   
                                if deg_coeff is not None:fixed_args['coeff_ord2name'][inst][vis][gen_root_par][deg_coeff]=root_par+'__IS'+inst+'_VS'+vis  
       
    return p_start


def model_par_names(par):
    r"""**Naming function**

    Returns name and unit of input variable for plot display.

    Args:
        None

    Returns:
        name_par (str): parameter name

    Example:
        >>> model_par_names(x)
        x_name
        
    """
    name_dic = {
        'veq':'v$_\mathrm{eq}$ (km s$^{-1}$)','vsini':'v$_\mathrm{eq}$sin i$_{*}$ (km/s)',
        'Peq':'P$_\mathrm{eq}$ (d)',
        'alpha_rot':r'$\alpha_\mathrm{rot}$','beta_rot':r'$\beta_\mathrm{rot}$',       
        'cos_istar':r'cos(i$_{*}$)','istar_deg':'i$_{*}(^{\circ}$)',
        'lambda_rad':'$\lambda$', 
        'c1_CB':'CB$_{1}$ (km s$^{-1}$)','c2_CB':'CB$_{2}$ (km s$^{-1}$)','c3_CB':'CB$_{3}$ (km s$^{-1}$)',  
        'inclination':'i$_\mathrm{p}$ ($^{\circ}$)','inclin_rad':'i$_\mathrm{p}$ (rad)',
        'aRs':'a/R$_{*}$',
        'Rstar':'R$_{*}$',
        'ctrst':'C','ctrst_ord0':'C$_{0}$','ctrst_ord1':'C$_{1}$','ctrst_ord2':'C$_{2}$','ctrst_ord3':'C$_{3}$','ctrst_ord4':'C$_{4}$', 
        'FWHM_ord0':'FWHM$_{0}$','FWHM_ord1':'FWHM$_{1}$','FWHM_ord2':'FWHM$_{2}$','FWHM_ord3':'FWHM$_{3}$','FWHM_ord4':'FWHM$_{4}$',
        'FWHM_LOR':'FWHM$_\mathrm{Lor}$',
        'a_damp':'a$_\mathrm{damp}$',
        'amp':'Amp',
        'rv':'RV (km/s)',
        'FWHM':'FWHM (km/s)',
        'rv_l2c':'RV$_{l}$-RV$_{c}$','amp_l2c':'A$_{l}$/A$_{c}$','FWHM_l2c':'FWHM$_{l}$/FWHM$_{c}$',
        'cont':'P$_\mathrm{cont}$',
        'c1_pol':'c$_1$','c2_pol':'c$_2$','c3_pol':'c$_3$','c4_pol':'c$_4$',
        'LD_u1':'LD$_1$','LD_u2':'LD$_2$','LD_u3':'LD$_3$','LD_u4':'LD$_4$',
        'f_GD':'f$_{\rm GD}$','beta_GD':'$\beta_{\rm GD}$','Tpole':'T$_{\rm pole}$',
        'eta_R':r'$\eta_{\rm R}$','eta_T':r'$\eta_{\rm T}$','ksi_R':r'\Ksi$_\mathrm{R}$','ksi_T':r'\Ksi$_\mathrm{T}$',
        'Tcenter' : 'T$_{sp}$', 'ang' : r'$\alpha_{sp}$', 'lat' : 'lat$_{sp}$', 'Fctrst' : 'F$_{sp}$',
        } 
    if par in name_dic:name_par = name_dic[par]
    else:name_par = par
    return name_par

def model_par_units(par):
    r"""**Unit function**

    Returns plain text unit of input variable.

    Args:
        None

    Returns:
        unit_par (str): parameter unit

    Example:
        >>> model_par_units(x)
        x_name
        
    """
    unit_dic = {
        'veq':'km/s','vsini':'km/s',
        'Peq':'d',
        'alpha_rot':'','beta_rot':'',       
        'cos_istar':'','istar_deg':'deg',
        'lambda_rad':'rad', 
        'c1_CB':'km/s','c2_CB':'km/s','c3_CB':'km/s',  
        'inclination':'deg','inclin_rad':'rad',
        'aRs':'',
        'Rstar':'Rsun',
        'ctrst':'','ctrst_ord0':'','ctrst_ord1':'','ctrst_ord2':'','ctrst_ord3':'','ctrst_ord4':'', 
        'FWHM_ord0':'km/s','FWHM_ord1':'km/s','FWHM_ord2':'km/s','FWHM_ord3':'km/s','FWHM_ord4':'km/s',
        'FWHM_LOR':'km/s',
        'a_damp':'',
        'amp':'',
        'rv':'km/s',
        'FWHM':'km/s',
        'rv_l2c':'km/s','amp_l2c':'','FWHM_l2c':'',
        'cont':'',
        'c1_pol':'','c2_pol':'','c3_pol':'','c4_pol':'',
        'LD_u1':'','LD_u2':'','LD_u3':'LD$_3$','LD_u4':'LD$_4$',
        'f_GD':'f$_{\rm GD}$','beta_GD':'$\beta_{\rm GD}$','Tpole':'T$_{\rm pole}$',
        'eta_R':r'$\eta_{\rm R}$','eta_T':r'$\eta_{\rm T}$','ksi_R':r'\Ksi$_\mathrm{R}$','ksi_T':r'\Ksi$_\mathrm{T}$',
        'Tcenter' : 'BJD',
        } 
    if par in unit_dic:unit_par = unit_dic[par]
    else:unit_par = ''
    return unit_par



##################################################################################################    
#%%% Initialization functions
################################################################################################## 

def init_joined_routines(data_mode,gen_dic,system_param,theo_dic,data_dic,fit_prop_dic):
    r"""**Joined fits: general initialization.**

    Initializes properties for the joined fits to stellar and planetary lines and properties.

    Args:
        TBD
    
    Returns:
        TBD
    
    """
    
    #Fit dictionary
    fit_dic={
        'merit':{},
        'verbose':fit_prop_dic['verbose'],
        'verb_shift':'   ',
        'print_par':fit_prop_dic['print_par'],
        'fit_mode':fit_prop_dic['fit_mode'],
        'uf_bd':{},
        'nx_fit':0,
        'run_name':'_'+gen_dic['main_pl_text'],
        'save_dir' : gen_dic['save_data_dir']+'/Joined_fits/'+data_mode+'/'+fit_prop_dic['fit_mode']+'/'}
    
    #--------------------------------------------------------------------------------

    #Arguments to be passed to the fit function
    fixed_args={
            
        #Global model properties        
        'system_param':deepcopy(system_param),
        'system_prop':deepcopy(data_dic['DI']['system_prop']),
        'system_spot_prop':deepcopy(data_dic['DI']['spots_prop']), 
        'DI_grid':False,
        'coord_line':fit_prop_dic['dim_fit'],
        'pol_mode':fit_prop_dic['pol_mode'],

        #Fit parameters
        'par_list':[],
        
        #Exposures to be fitted
        'nexp_fit_all':{},
        'idx_in_fit':{},        
        'master_out':{},

        #Intrinsic continuum flux
        #    - IntrProp: required for the intensity weighing but absolute value does not matter
        #    - IntrProf: required for parameter initialization, but set within the fit function to the visit-specific flux
        'flux_cont':1.,     
        
        
        'inst_list':[],
        'prior_func':fit_prop_dic['prior_func'], 
        'inst_vis_list':{},
        'transit_pl':{},
        'cond_transit_pl':False,
        'transit_sp':{},
        'cond_transit_sp':False,
        'bin_mode':{},
        'update_crosstime':False,
        'fit' : {'chi2':True,'':False,'mcmc':True}[fit_prop_dic['fit_mode']], 
        'unthreaded_op':fit_prop_dic['unthreaded_op'],     
        }
   
    #Checks
    if len(fit_prop_dic['idx_in_fit'])==0:stop('No exposures are included in the fit')
    
    #Initializes C-based profile calculation
    if 1==0:
        fixed_args['fun_to_use'],fixed_args['fun_to_free'] = def_Cfunc_prof()

    return fixed_args,fit_dic

def init_joined_routines_inst(rout_mode,inst,fit_prop_dic,fixed_args):
    r"""**Joined fits: instrument initialization.**

    Initializes properties for the joined fits to stellar and planetary lines.

    Args:
        inst (str) : Instrument considered.
        fit_prop_dic (dict) : Dictionary containing the parameters of the fitting process.
        fixed_args (dict) : Dictionary containing the arguments that will be passed to the fitting function.

    Returns:
        None
    
    """
    #Instrument is fitted
    fit_prop_dic[inst]={}
    fixed_args['inst_list']+=[inst]
    fixed_args['inst_vis_list'][inst]=[]  
    for key in ['ph_fit','nexp_fit_all','transit_pl','transit_sp','bin_mode','idx_in_fit']:fixed_args[key][inst]={}
    if ('Intr' in rout_mode) or ('Res' in rout_mode):fixed_args['coord_fit'][inst]={}

    return None

def init_joined_routines_vis(inst,vis,fit_prop_dic,fixed_args):
    r"""**Joined fits: visit initialization.**

    Initializes properties for the joined fits to stellar and planetary lines.

    Args:
        inst (str) : Instrument considered.
        vis (str) : Visit considered.
        fit_prop_dic (dict) : Dictionary containing the parameters of the fitting process.
        fixed_args (dict) : Dictionary containing the arguments that will be passed to the fitting function.
    
    Returns:
        None
    
    """
    #Identify whether visit is fitted over original or binned exposures
    #    - for simplicity we then use the original visit name in all fit dictionaries, as a visit will not be fitted at the same time in its original and binned format
    if (vis in fit_prop_dic['idx_in_fit'][inst]) and (len(fit_prop_dic['idx_in_fit'][inst][vis])>0):fixed_args['bin_mode'][inst][vis]=''
    elif (vis+'_bin' in fit_prop_dic['idx_in_fit'][inst]) and (len(fit_prop_dic['idx_in_fit'][inst][vis+'_bin'])>0):fixed_args['bin_mode'][inst][vis]='_bin'
    else:fixed_args['bin_mode'][inst][vis]=None

    return None

def init_joined_routines_vis_fit(rout_mode,inst,vis,fit_prop_dic,fixed_args,data_vis,gen_dic,data_dic,coord_dic):
    r"""**Joined fits: initialization.**

    Initializes properties for the joined fits to stellar and planetary lines.

    Args:
        TBD
    
    Returns:
        TBD
    
    """
    fit_prop_dic[inst][vis]={}

    #Binned data
    if fixed_args['bin_mode'][inst][vis]=='_bin':
        if 'Intr' in fixed_args['rout_mode']:data_vis_bin = dataload_npz(gen_dic['save_data_dir']+'/Intrbin_data/'+inst+'_'+vis+'_'+data_dic['Intr']['dim_bin']+'_add')
        elif 'Res' in fixed_args['rout_mode']:data_vis_bin = dataload_npz(gen_dic['save_data_dir']+'/Resbin_data/'+inst+'_'+vis+'_'+data_dic['Res']['dim_bin']+'_add')  
        data_mode = data_vis_bin['type']      
        n_in_tr = data_vis_bin['n_in_tr']
        n_in_visit = data_vis_bin['n_in_visit']

    #Original data
    else:
        data_vis_bin = None
        n_in_tr = data_vis['n_in_tr']    
        n_in_visit = data_vis['n_in_visit']
        data_mode = data_dic[inst][vis]['type']
        
    #Planets are transiting
    if len(data_vis['transit_pl'] )>0:
        fixed_args['cond_transit_pl'] = True
        
        #Check for multi-transits
        #    - if two planets are transiting the properties derived from the fits to intrinsic profiles cannot be fitted, as the model only contains a single line profile
        if rout_mode=='IntrProp':
            if len(data_vis['transit_pl'])>1:stop('Multi-planet transit must be modelled with full intrinsic profiles')
            fixed_args['transit_pl'][inst][vis]=[data_vis['transit_pl'][0]] 
        else:fixed_args['transit_pl'][inst][vis]=data_vis['transit_pl'] 
        fixed_args['transit_pl'][inst][vis]=data_vis['transit_pl'] 

    #Spots are visible
    if len(data_vis['transit_sp'] )>0:
        fixed_args['cond_transit_sp'] = True    
        fixed_args['transit_sp'][inst][vis]=data_vis['transit_sp']
        
    #Fitted exposures
    if rout_mode in ['DIprof', 'ResProf']:n_default_fit = n_in_visit
    else: n_default_fit = n_in_tr
    fixed_args['inst_vis_list'][inst]+=[vis]
    if fit_prop_dic['idx_in_fit'][inst][vis]=='all':fixed_args['idx_in_fit'][inst][vis]=range(n_default_fit)
    else:fixed_args['idx_in_fit'][inst][vis]=np.intersect1d(fit_prop_dic['idx_in_fit'][inst][vis],range(n_default_fit))

    #Keep defined profiles
    if (data_mode in ['Intr','Res']):fixed_args['idx_in_fit'][inst][vis]=np.intersect1d(fixed_args['idx_in_fit'][inst][vis],data_dic[data_mode][inst][vis+fixed_args['bin_mode'][inst][vis]]['idx_def'])
    fixed_args['nexp_fit_all'][inst][vis]=len(fixed_args['idx_in_fit'][inst][vis])     

    #Store coordinates of fitted exposures in global table
    if fixed_args['bin_mode'][inst][vis]=='_bin':
        sub_idx_in_fit = fixed_args['idx_in_fit'][inst][vis]
        coord_vis = data_vis_bin['coord']
    else:
        if rout_mode=='ResProf':sub_idx_in_fit = fixed_args['idx_in_fit'][inst][vis]        
        else:sub_idx_in_fit = gen_dic[inst][vis]['idx_in'][fixed_args['idx_in_fit'][inst][vis]]
        coord_vis = coord_dic[inst][vis]
    for par in ['coord_fit','ph_fit']:fixed_args[par][inst][vis]={}
    if fixed_args['cond_transit_pl']:
        for pl_loc in fixed_args['transit_pl'][inst][vis]:
            fixed_args['ph_fit'][inst][vis][pl_loc] = np.vstack((coord_vis[pl_loc]['st_ph'][sub_idx_in_fit],coord_vis[pl_loc]['cen_ph'][sub_idx_in_fit],coord_vis[pl_loc]['end_ph'][sub_idx_in_fit]) ) 
            fixed_args['coord_fit'][inst][vis][pl_loc] = {}
            for key in ['cen_pos','st_pos','end_pos']:fixed_args['coord_fit'][inst][vis][pl_loc][key] = coord_vis[pl_loc][key][:,sub_idx_in_fit]    
            fixed_args['coord_fit'][inst][vis][pl_loc]['ecl'] = coord_vis[pl_loc]['ecl'][sub_idx_in_fit]  
    if fixed_args['cond_transit_sp']:
        for spot in fixed_args['transit_sp'][inst][vis]:
            fixed_args['coord_fit'][inst][vis][spot] = {}
            for key in ['Tcenter', 'ang', 'ang_rad', 'lat', 'ctrst']:
                fixed_args['coord_fit'][inst][vis][spot][key] = coord_vis[spot][key][sub_idx_in_fit] 
            for key in ['lat_rad_exp','sin_lat_exp','cos_lat_exp','long_rad_exp','sin_long_exp','cos_long_exp','x_sky_exp','y_sky_exp','z_sky_exp']:
                fixed_args['coord_fit'][inst][vis][spot][key] = coord_vis[spot][key][:,sub_idx_in_fit] 
            fixed_args['coord_fit'][inst][vis][spot]['is_visible'] = coord_vis[spot]['is_visible'][:,sub_idx_in_fit] 
        fixed_args['coord_fit'][inst][vis]['bjd']=coord_vis['bjd'][sub_idx_in_fit]
        fixed_args['coord_fit'][inst][vis]['t_dur']=coord_vis['t_dur'][sub_idx_in_fit]
    
    return data_vis_bin
    
    



##################################################################################################    
#%%% Analysis functions
################################################################################################## 

def com_joint_fits(rout_mode,fit_dic,fixed_args,fit_prop_dic,gen_dic,data_dic,theo_dic,mod_prop):
    r"""**Wrap-up for time-series fits.**

    Performs joint fits to time-series.
    
    Args:
        TBD

    Returns:
        TBD
        
    """

    #------------------------------------------------------------------------------------------------------------------------------------------------
    #Set optimization level for line profile calculation
    if 'Prof' in rout_mode:   
    
        #Optimization levels
        fixed_args['C_OS_grid']=False
        fixed_args['OS_grid'] = False
        
        #Multithreading turned off for levels 1,2,3
        if fit_prop_dic['Opt_Lvl']>=1:fixed_args['unthreaded_op'] += ['prof_grid']
        
        #Over-simplified grid building turned on for levels 2,3
        if fit_prop_dic['Opt_Lvl']>=2:fixed_args['OS_grid'] = True
        
        #Over-simplified grid building turned on and coded in C for level 3
        if fit_prop_dic['Opt_Lvl']==3:fixed_args['C_OS_grid'] = True
    
        #Model fit and calculation
        print('Opt Level:', fit_prop_dic['Opt_Lvl'])        
    
    #------------------------------------------------------------------------------------------------------------------------------------------------
        
    #Fit parameters
    p_start = Parameters()  

    #Model parameters
    #    - we define here parameters common to the different fit routines, but they can be updated and specific parameters defined in the routines later
    #
    #------------------------------------------------------------
    #    - each parameter in p_start is defined by its name, guess/fixed value, fit flag, lower and upper boundary of explored range, expression 
    #    - some parameters can be left undefined and will be set to default values:
    # + alpha, beta : 0
    # + CB coefficients: 0
    # + cos_istar : 0.
    # + aRs, inclin_rad : values defined in ANTARESS_settings 
    #    - parameters are ordered
    #    - use 'expression' to set properties to the same value :
    # define par1, then define par2 and set it to expr='par1'
    #      do not include in the expression of a parameter another parameter linked with an expression
    #    - all parameter options are valid for both chi2 and mcmc, except for boundary conditions that must be defined differently for the mcmc
    #    - model parameters are :
    # + lambda_rad : sky-projected obliquity (rad) 
    # + veq : true equatorial rotational velocity (km/s)    
    # + cos(istar) : inclination of stellar spin axis (rad)
    #                for the fit we use cos(istar), natural variable in the model and linked with istar through bijection in the allowed range
    # + alpha_rot, beta_rot : parameters of DR law
    # + c1, c2 and c3 : coefficients of the mu-dependent CB velocity polynomial (km/s) 
    # + aRs, inclin_rad: in some cases these properties can be better constrained by the fit to the local RVs
    #                    since they control the orbital trajectory of the planet, fitting these parameters will make the code recalculate the coordinates of the planet-occulted regions    
    #      for more details see calc_plocc_spot_prop()  
    #    - parameters specific to a given planet should be defined as 'parname__plX', where X is the name of the planet used throughout the pipeline
    #    - the model uses the nominal RpRs and LD coefficients, which must be suited to the spectral band from which the local stellar properties were derived
    #------------------------------------------------------------
    #Priors on variable model parameters
    #    - see MCMC_routines > ln_prior_func() for the possible priors
    #    - priors must be defined for all variable parameters 
    #    - there is a general, uniform prior on cos(istar) between -1 and 1 that allows us to limit istar in 0:180 
    # since there is a bijection between istar and cos(istar) over this range
    #    - lambda is defined over the entire angular space, however it might need to be limited to a fixed range to prevent the 
    # mcmc to switch from one best-fit region defined in [-180;180] to the next in x+[-180;180]. By default we take -2*180:2*180. 
    #      this range might need to be shifted if the best-fit is at +-180
    #      the posterior distribution is folded over x+[-180;180] in the end
    #    - we use the same approach for the orbital inclination, which must be limited in the post-processing to the range [0-90]°
    #
    #------------------------------------------------------------
    #Starting points of walkers for variable parameters
    #    - they must be set to different values for each walker
    #    - for simplicity we define randomly for each walker 
    #    - parameters must be defined in the same order as in 'p_start'
    p_start = init_custom_DI_par(fixed_args,gen_dic,data_dic['DI']['system_prop'],fixed_args['system_param']['star'],p_start,[0.,None,None])

    #Condition to calculate CB
    if ('c1_CB' in mod_prop) or ('c2_CB' in mod_prop)  or ('c3_CB' in mod_prop):fixed_args['par_list']+=['CB_RV']
    
    #Fit spin-orbit angle by default when relevant
    #    - assuming common values to all datasets
    if ((rout_mode=='IntrProp') and (fixed_args['prop_fit']=='rv')) or (rout_mode in ['IntrProf','ResProf']):
        for pl_loc in gen_dic['studied_pl']:
            if 'lambda_rad__pl'+pl_loc not in mod_prop:p_start.add_many(('lambda_rad__pl'+pl_loc, 0.,   True, -2.*np.pi,2.*np.pi,None))
        
    #Initialize line properties
    #    - using Gaussian line as default for intrinsic profiles
    if ((rout_mode=='IntrProp') and (fixed_args['prop_fit']=='ctrst')) or (rout_mode in ['IntrProf','ResProf']):
        if not any(['ctrst_' in prop for prop in mod_prop]):p_start.add_many(('ctrst_ord0__IS__VS_', 0.5,   True, 0.,1.  ,None))
    if ((rout_mode=='IntrProp') and (fixed_args['prop_fit']=='FWHM')) or (rout_mode in ['IntrProf','ResProf']):
        if not any(['FWHM_' in prop for prop in mod_prop]):p_start.add_many(('FWHM_ord0__IS__VS_', 5.,   True, 0.,100.  ,None))
        
    #Re-defining the spot's Tcenter bounds, guess and priors with the cross-time supplement
    if fixed_args['update_crosstime']:
        for inst in list(fixed_args['spot_crosstime_supp'].keys()):
            for vis in list(fixed_args['spot_crosstime_supp'][inst].keys()):
                for par in mod_prop:
                    if ('Tcenter' in par) and (inst in par) and (vis in par):
                        mod_prop[par]['guess'] -= fixed_args['spot_crosstime_supp'][inst][vis]
                        mod_prop[par]['bd'] = list(np.array(mod_prop[par]['bd']))-fixed_args['spot_crosstime_supp'][inst][vis]
                        fit_prop_dic['priors'][par]['low'] -= fixed_args['spot_crosstime_supp'][inst][vis]
                        fit_prop_dic['priors'][par]['high'] -= fixed_args['spot_crosstime_supp'][inst][vis]
    
    #Retrieving the rotational period - to set the priors on the Tcenter and prevent aliases
    #    - in days
    if 'veq' in mod_prop:fixed_args['Peq'] = (2*np.pi*fixed_args['system_param']['star']['Rstar_km'])/(mod_prop['veq']['guess']*24*3600)
    else:fixed_args['Peq'] = deepcopy(fixed_args['system_param']['star']['Peq'])
        
    #Initializing stellar properties
    if rout_mode=='IntrProp':    
        fixed_args['grid_dic'] = deepcopy(theo_dic)
        fixed_args['grid_dic']['precision'] = 'low'      #to calculate intensity-weighted properties
        line_type='ana'                                  #to avoid raising warning, even though properties are not used to calculate a line profile

    #Initializing stellar profiles
    else: line_type = fixed_args['mode']

    #Parameter initialization
    #    - default system properties are overwritten in p_start if they are defined in 'mod_prop', whether the model is fitted or called in forward mode
    p_start = par_formatting(p_start,mod_prop,fit_prop_dic['priors'],fit_dic,fixed_args,'','',line_type)
   
    #Initializing stellar profile grid
    #    - must be done after 'par_formatting' to identify variable line parameters
    if rout_mode!='IntrProp':  
        if fixed_args['cond_transit_sp']:     
            fixed_args = init_custom_DI_prof(fixed_args,gen_dic,data_dic['DI']['system_prop'],data_dic['DI']['spots_prop'],theo_dic,fixed_args['system_param']['star'],p_start)
        else:
            fixed_args = init_custom_DI_prof(fixed_args,gen_dic,data_dic['DI']['system_prop'],{},theo_dic,fixed_args['system_param']['star'],p_start)

    #Stellar grid properties
    fixed_args['grid_dic'].update({'Ssub_Sstar_pl':theo_dic['Ssub_Sstar_pl'],'x_st_sky_grid_pl':theo_dic['x_st_sky_grid_pl'],'y_st_sky_grid_pl':theo_dic['y_st_sky_grid_pl'],'nsub_Dpl':theo_dic['nsub_Dpl'],'d_oversamp':theo_dic['d_oversamp'],'Istar_norm_achrom':theo_dic['Istar_norm_achrom']})
    if fixed_args['cond_transit_sp']:
        fixed_args['grid_dic'].update({'Ssub_Sstar_sp':theo_dic['Ssub_Sstar_sp'],'x_st_sky_grid_sp':theo_dic['x_st_sky_grid_sp'],'y_st_sky_grid_sp':theo_dic['y_st_sky_grid_sp'],'nsub_Dspot':theo_dic['nsub_Dspot'],'d_oversamp_spot':theo_dic['d_oversamp_spot']})

    #Determine if orbital and light curve properties are fitted or whether nominal values are used
    #    - this depends on whether parameters required to calculate coordinates of planet-occulted regions are fitted  
    #    - in case the model is calculate in forward mode, we activate the condition as well so that the nominal system properties are updated with those defined in 'mod_prop'  
    if fixed_args['cond_transit_pl']:
        par_orb=['inclin_rad','aRs','lambda_rad']
        par_LC=['RpRs']    
        for par in par_orb+par_LC:fixed_args[par+'_pl']=[]
        fixed_args['fit_orbit']=False
        fixed_args['fit_RpRs']=False
    if fixed_args['cond_transit_sp']:
        par_spot=['lat', 'Tcenter', 'ang', 'ctrst']    
        for par in par_spot:fixed_args[par+'_sp']=[]
        fixed_args['fit_spot']=False
        fixed_args['fit_spot_ang']=[]
    for par in p_start:
        
        #Check if rootname of orbital/LC properties is one of the parameters left free to vary for a given planet    
        #    - if so, store name of planet for this property
        if fixed_args['cond_transit_pl']:
            for par_check in par_orb:
                if (par_check in par):
                    if ('__IS' in par):pl_name = (par.split('__pl')[1]).split('__IS')[0]                  
                    else:pl_name = (par.split('__pl')[1]) 
                    if (p_start[par].vary) or fit_dic['fit_mode']=='':
                        fixed_args[par_check+'_pl']+= [pl_name]
                        fixed_args['fit_orbit']=True                     
            for par_check in par_LC:
                if (par_check in par):
                    if ('__IS' in par):pl_name = (par.split('__pl')[1]).split('__IS')[0]                  
                    else:pl_name = (par.split('__pl')[1])                 
                    if (p_start[par].vary) or fit_dic['fit_mode']=='':
                        fixed_args[par_check+'_pl'] += [pl_name]
                        fixed_args['fit_RpRs']=True 

        #Check if rootname of spot orbital properties is one of the parameters left free to vary for a given spot    
        #    - if so, store name of spot for this property 
        if fixed_args['cond_transit_sp']:        
            for par_check in par_spot:
                if (par_check in par) and ('_SP' in par):
                    spot_name = par.split('_SP')[1]
                    if (p_start[par].vary) or fit_dic['fit_mode']=='':
                        fixed_args[par_check+'_sp']+= [spot_name]
                        fixed_args['fit_spot']=True
                    if ('ang' in par_check) and p_start[par].vary:
                             fixed_args['fit_spot_ang']+=[spot_name]
            
    #Unique list of planets with variable properties  
    if fixed_args['cond_transit_pl']:                
        for par in par_orb:fixed_args[par+'_pl'] = list(np.unique(fixed_args[par+'_pl']))
        for par in par_LC:fixed_args[par+'_pl'] = list(np.unique(fixed_args[par+'_pl']))
        fixed_args['b_pl'] = list(np.unique(fixed_args['inclin_rad_pl']+fixed_args['aRs_pl']))
   
    #Unique list of spots with variable properties
    if fixed_args['cond_transit_sp']:  
        for par in par_spot:fixed_args[par+'_sp'] = list(np.unique(fixed_args[par+'_sp']))

    #Store the number of threads - needed when fitting joined residual profiles
    fixed_args['nthreads']=fit_prop_dic['nthreads']

    #Fit initialization
    init_fit(fit_dic,fixed_args,p_start,fit_prop_dic,model_par_names,model_par_units)     
    merged_chain = None
    
    ########################################################################################################  
    #Calculating a first model to check all in-transit model exposures are defined
    if fit_dic['fit_mode'] in ['chi2','mcmc'] and ('Intr' in rout_mode):
        mod_dic,coeff_line_dic,mod_prop_dic = fixed_args['mod_func'](p_start,fixed_args)
        if rout_mode=='IntrProp':
            if True in np.isnan(mod_dic):
                print('WARNING: the model planet does not occult the star over in-transit exposures at indexes:')
                for inst in mod_prop_dic:
                    for vis in mod_prop_dic[inst]:
                        idx_unocc=np_where1D(np.isnan(mod_prop_dic[inst][vis]))
                        if len(idx_unocc)>0:print('   ',idx_unocc,'in '+inst+' '+vis)
                
                print('Exclude them from the fit.')                
                stop()

        if rout_mode=='IntrProf':
            cond_unocc=False
            for inst in mod_prop_dic:
                for vis in mod_prop_dic[inst]: 
                    for pl_loc in mod_prop_dic[inst][vis]:
                        idx_unocc=np_where1D(np.isnan(mod_prop_dic[inst][vis]['rv']))
                        if len(idx_unocc)>0:
                            cond_unocc=True
                            print('   ',idx_unocc,'in '+inst+' '+vis+' for '+pl_loc)
            if cond_unocc:
                print('WARNING: the model planet does not occult the star over in-transit exposures at these indexes. Exclude them from the fit.')
                stop()
            
    ########################################################################################################   

    #Fit by chi2 minimization
    if fit_dic['fit_mode']=='chi2':
        fixed_args['fit'] = True
        print('       Chi2 fit')   
        p_final = call_lmfit(p_start,fixed_args['x_val'],fixed_args['y_val'],fixed_args['cov_val'],fixed_args['fit_func'],verbose=fit_prop_dic['verbose'],fixed_args=fixed_args)[2]

    ########################################################################################################    
    #Fit par emcmc 
    elif fit_dic['fit_mode']=='mcmc':  
        fixed_args['fit'] = True
        print('       MCMC fit')
        
        #Default options
        if 'nwalkers' not in fit_prop_dic['mcmc_set']:fit_dic['nwalkers'] = int(3*fit_dic['merit']['n_free'])
        else:fit_dic['nwalkers'] = fit_prop_dic['mcmc_set']['nwalkers']
        if 'nsteps' not in fit_prop_dic['mcmc_set']:fit_dic['nsteps'] = 5000
        else:fit_dic['nsteps'] = fit_prop_dic['mcmc_set']['nsteps']
        if 'nburn' not in fit_prop_dic['mcmc_set']:fit_dic['nburn'] = 1000
        else:fit_dic['nburn'] = fit_prop_dic['mcmc_set']['nburn']

        #Disable multi-threading
        if 'emcee' in fit_prop_dic['unthreaded_op']:mcmc_threads=1
        else:mcmc_threads = fit_prop_dic['nthreads']

        #Run MCMC
        if fit_prop_dic['mcmc_run_mode']=='use':
            print('         Applying MCMC') 

            #Complex prior function
            if (len(fixed_args['prior_func'])>0):fixed_args['global_ln_prior_func']=global_ln_prior_func

            #Call to MCMC
            walker_chains=call_MCMC(mcmc_threads,fixed_args,fit_dic,run_name=fit_dic['run_name'],verbose=fit_dic['verbose'])
               
        #---------------------------------------------------------------  
       
        #Reuse MCMC
        elif fit_prop_dic['mcmc_run_mode']=='reuse':
            print('         Retrieving MCMC') 
            if len(fit_prop_dic['mcmc_reuse'])==0:
                walker_chains=np.load(fit_dic['save_dir']+'raw_chains_walk'+str(fit_dic['nwalkers'])+'_steps'+str(fit_dic['nsteps'])+fit_dic['run_name']+'.npz')['walker_chains']  #(nwalkers, nsteps, n_free)
            else:
                walker_chains = np.empty([fit_dic['nwalkers'],0,fit_dic['merit']['n_free'] ],dtype=float)
                fit_dic['nsteps'] = 0
                fit_dic['nburn'] = 0
                for mcmc_path,nburn in zip(fit_prop_dic['mcmc_reuse']['paths'],fit_prop_dic['mcmc_reuse']['nburn']):
                     walker_chains_loc=np.load(mcmc_path)['walker_chains'][:,nburn::,:] 
                     fit_dic['nsteps']+=(walker_chains_loc.shape)[1]
                     walker_chains = np.append(walker_chains,walker_chains_loc,axis=1)
                    
        #Excluding parts of the chains
        if fit_dic['exclu_walk']:
            print('       Excluding walkers manually')
            
            #Joined fit
            if gen_dic['fit_IntrProf']:
            
                # ipar_loc=np_where1D(fixed_args['var_par_list']=='lambda_rad__plHD3167_c')
                # # wgood=np_where1D(np.min(walker_chains[:,:,ipar_loc],axis=1)>-1.)
                # # wgood=np_where1D(np.median(walker_chains[:,:,ipar_loc],axis=1)>-2.5)
                # wgood=np_where1D(np.median(walker_chains[:,:,ipar_loc],axis=1)<-1.)
                # wgood=np_where1D(np.median(walker_chains[:,:,ipar_loc],axis=1)<-1.5)
    
                # ipar_loc=np_where1D(fixed_args['var_par_list']=='lambda_rad__plTOI858b')
                # wgood=np_where1D(np.median(walker_chains[:,:,ipar_loc],axis=1)<5.)
    
                if gen_dic['star_name'] == 'GJ436':
                    # wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='amp_l2c__ISESPRESSO_VS20190228')],axis=1)<1.) &\
                    #                  (np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='amp_l2c__ISESPRESSO_VS20190429')],axis=1)<1.)&\
                    #                  (np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='FWHM_ord0__ISESPRESSO_VS20190429')],axis=1)>2.))
    
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='amp_l2c__IS__VS_')],axis=1)<1.)&\
                                      (np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='FWHM_ord0__ISESPRESSO_VS20190228')],axis=1)>2.)&\
                                      (np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='FWHM_ord0__ISESPRESSO_VS20190429')],axis=1)>2.))

                elif gen_dic['star_name'] == 'HAT_P3':
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='FWHM_ord0__ISHARPN_VS20200130')],axis=1)>0.))

                elif gen_dic['star_name'] == 'HAT_P33':
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='FWHM_ord0__ISHARPN_VS20191204')],axis=1)>0.))


                elif gen_dic['star_name'] == 'HAT_P49':
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='FWHM_ord0__ISHARPN_VS20200730')],axis=1)<20.))

                elif gen_dic['star_name'] == 'WASP107':
                    # wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='lambda_rad__plWASP107b')],axis=1)<0.))
                    # wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='lambda_rad__plWASP107b')],axis=1)>0.))
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='a_damp__ISCARMENES_VIS_VS20180224')],axis=1)<10.))

                elif gen_dic['star_name'] == 'HIP41378':
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='lambda_rad__plHIP41378d')],axis=1)<4.))

                elif gen_dic['star_name'] == 'WASP156':
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='a_damp__ISCARMENES_VIS_VS20190928')],axis=1)<2.))

                elif gen_dic['star_name'] == 'WASP166':
                    # wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='FWHM_ord0__IS__VS_')],axis=1)>5.))
                    # wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='ctrst_ord0__ISHARPS_VS_')],axis=1)<0.63) )
                    # wgood=np_where1D((np.min(walker_chains[:,250:750,np_where1D(fixed_args['var_par_list']=='veq')],axis=1)>5.) )
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='veq')],axis=1)>4.5) )
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='Rstar')],axis=1)<1.45) )

                elif gen_dic['star_name'] == 'Kepler25':
                    ilamb = np_where1D(fixed_args['var_par_list']=='lambda_rad__plKepler25c')
                    for iwalk in range(fit_dic['nwalkers']):
                        lambda_temp=np.squeeze(walker_chains[iwalk,:,ilamb])+np.pi
                        w_gt_360=(lambda_temp > 2.*np.pi)
                        if True in w_gt_360:
                            walker_chains[iwalk,w_gt_360,ilamb]=np.mod(lambda_temp[w_gt_360],2.*np.pi)-np.pi
                        w_lt_0=(lambda_temp < 0.)
                        if True in w_lt_0:
                            i_mod=npint(np.abs(lambda_temp[w_lt_0])/(2.*np.pi))+1.
                            walker_chains[iwalk,w_lt_0,ilamb] = lambda_temp[w_lt_0]+i_mod*2.*np.pi-np.pi
                        
                    wgood=np_where1D( (np.max(walker_chains[:,:,ilamb],axis=1)<80.*np.pi/180.) & (np.min(walker_chains[:,:,ilamb],axis=1)>-80.*np.pi/180.) )


                elif gen_dic['star_name'] == '55Cnc':
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='veq')],axis=1)>0.5) )

                elif gen_dic['star_name'] == 'WASP76':
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='lambda_rad__plWASP76b')],axis=1)<-1.1) )

                elif gen_dic['star_name'] == 'HD189733':
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='cos_istar')],axis=1)<-0.35) )

            #Surface RV fit
            if gen_dic['fit_IntrProp']:
    
                ipar_loc=np_where1D(fixed_args['var_par_list']=='lambda_rad__plGJ436_b')
                wgood=np_where1D(np.min(walker_chains[:,:,ipar_loc],axis=1)>-1.)    
    
                # ipar_loc=np_where1D(fixed_args['var_par_list']=='veq')
                # wgood=np_where1D(walker_chains[:,-1,ipar_loc]>2.)                    
                # ipar_loc=np_where1D(fixed_args['var_par_list']=='cos_istar')
                # wgood=np_where1D(walker_chains[:,-1,ipar_loc]<0.)   
    
                ipar_loc=np_where1D(fixed_args['var_par_list']=='inclin_rad__plGJ436_b')
                wgood=np_where1D(walker_chains[:,-1,ipar_loc]<=0.5*np.pi)  


                if gen_dic['star_name'] == 'HD189733':
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='cos_istar')],axis=1)<0.) )





            print('   ',len(wgood),' walkers kept / ',fit_dic['nwalkers'])
            walker_chains=np.take(walker_chains,wgood,axis=0)     
            fit_dic['nwalkers']=len(wgood)  
               
        #------------------------------------------------------------------------------------------------            
 
        #Processing:
        #    - best-fit parameters for model calculation
        #    - 1-sigma and envelope samples for plot
        #    - plot of model parameter chains
        #    - save file 
        p_final,merged_chain,par_sample_sig1,par_sample=postMCMCwrapper_1(fit_dic,fixed_args,walker_chains,mcmc_threads,fixed_args['par_names'],verbose=fit_dic['verbose'],verb_shift=fit_dic['verb_shift']+'    ')

    ########################################################################################################  
    #No fit is performed: guess parameters are kept
    else:
        fixed_args['fit'] = False
        print('       Fixed model')
        p_final = deepcopy(p_start)   
    
    ########################################################################################################      

    #Merit values     
    p_final=fit_merit(p_final,fixed_args,fit_dic,fit_prop_dic['verbose'])                

    return merged_chain,p_final













def MAIN_single_anaprof(vis_mode,data_type,data_dic,gen_dic,inst,vis,coord_dic,theo_dic,plot_dic,star_param):
    r"""**Main analysis routine for single profiles**

    Initializes and calls analysis of line profiles
    
     - profiles in their input rest frame, original exposures, for all formats
     - profiles in their input or star (if aligned) rest frame, original exposures, converted from 2D->1D 
     - profiles in their input or star (if aligned) rest frame, binned exposures, all formats
     - profiles in the star rest frame, binned exposures, all formats   

    Formats are CCF profiles, 1D spectra, or a specific 2D spectral order.
    Each profile is analyzed independently from the others.
    The routine can also be used to fit the unocculted disk-integrated stellar profile using best estimates for the intrinsic stellar profiles (measured, theoretical, imported) 

    Args:
        TBD

    Returns:
        TBD
        
    """  
    
    #Analyzing profiles
    if 'orig' in data_type:   #unbinned profiles
        bin_mode=''
        txt_print=' '
        vis_det=vis  
        data_mode = data_dic[inst][vis]['type']
    elif 'bin' in data_type:  #binned profiles
        bin_mode='bin'   
        txt_print=' binned '
        if vis_mode=='':            #from single visit
            vis_det=vis
            data_mode = data_dic[inst][vis]['type']    
        elif vis_mode=='multivis':  #from multiple visits
            vis_det='binned'
            data_mode = data_dic[inst]['type'] 
            txt_print+=' (multi-vis.) '
    
    if ('DI' in data_type) or ('Intr' in data_type):
        if ('DI' in data_type):data_type_gen='DI' 
        elif ('Intr' in data_type):data_type_gen='Intr'               
        print('   > Analyzing'+txt_print+gen_dic['type_name'][data_type_gen]+' stellar profiles')                 
    elif ('Atm' in data_type):   
        data_type_gen='Atm' 
        print('   > Analyzing'+txt_print+gen_dic['type_name'][data_type_gen]+data_dic['Atm']['pl_atm_sign']+' profiles') 

    save_path = gen_dic['save_data_dir']+data_type+'_prop/'+gen_dic['add_txt_path'][data_type_gen]+inst+'_'+vis_det    
    cond_calc = gen_dic['calc_fit_'+data_type_gen+bin_mode+vis_mode]
    prop_dic = deepcopy(data_dic[data_type_gen])   
    if prop_dic['cst_err'+bin_mode]:print('         Using constant error')
    if (inst in prop_dic['sc_err']) and (vis_det in prop_dic['sc_err'][inst]): print('         Scaling data error')
    
    #Analyzing
    if cond_calc:
        print('         Calculating data') 
        fit_dic={}
        data_inst = data_dic[inst]

        #Generic fit options
        fit_properties={'type':data_mode,'nthreads':gen_dic['fit_prof_nthreads'],'rv_osamp_line_mod':theo_dic['rv_osamp_line_mod'],'use_cov':gen_dic['use_cov'],'varpar_priors':{},'jitter':False,'inst_list':[inst],'inst_vis_list':{inst:[vis_det]},
                        'save_dir':save_path+'_'+prop_dic['fit_mode']+'/','conv_model' : prop_dic['conv_model'],'resamp_mode' : gen_dic['resamp_mode'],'line_trans':prop_dic['line_trans']}
        fit_properties.update({**star_param})
        
        #Chromatic intensity grid
        if ('spec' in data_mode) and ('chrom' in data_inst['system_prop']):
            fit_properties['chrom'] = data_inst['system_prop']['chrom']
        else:
            fit_properties['chrom'] = None        
               
        #Exposure-specific properties        
        if ('mod_def' in prop_dic) and (inst in prop_dic['mod_def']):fit_properties.update(prop_dic['mod_def'][inst])

        #Order selection
        #    - for 2D spectra we select a specific order to perform the comparison, as it is too heavy otherwise
        #    - we remove the order structure from CCF and 1D spectra to have one-dimensional tables
        if data_mode in ['CCF','spec1D']:iord_sel = 0
        elif data_mode=='spec2D':
            if inst not in prop_dic['fit_prof']['order']:stop('Define fitted order')
            else:iord_sel = prop_dic['fit_prof']['order'][inst] 

        #Continuum and fitted ranges
        #    - spectral tables are defined in:
        # > input frame for original and binned unaligned disk-integrated data
        #   star rest frame for binned and aligned disk-integrated data
        # > star frame for original / binned intrinsic data
        # > star frame for original / binned planetary data
        #    - continuum and fit ranges are defined in:
        # > input frame for disk-integrated data
        # > star frame for intrinsic data
        # > star frame for planetary data 
        fit_range = prop_dic['fit_range'][inst][vis_det]
        cont_range = prop_dic['cont_range'][inst][iord_sel]
        trim_range = prop_dic['fit_prof']['trim_range'][inst] if (inst in prop_dic['fit_prof']['trim_range']) else None 
 
        #MCMC fit default options
        if prop_dic['fit_mode']=='mcmc': 
            if ('mcmc_set' not in prop_dic):prop_dic['mcmc_set']={}
            for key in ['nwalkers','nsteps','nburn']:
                if key not in prop_dic['mcmc_set']:prop_dic['mcmc_set'][key] = {}
                if (inst not in prop_dic['mcmc_set'][key]):prop_dic['mcmc_set'][key][inst] = {}
            if (vis not in prop_dic['mcmc_set']['nwalkers'][inst]):prop_dic['mcmc_set']['nwalkers'][inst][vis] = 50
            if (vis not in prop_dic['mcmc_set']['nsteps'][inst]):prop_dic['mcmc_set']['nsteps'][inst][vis] = 1000
            if (vis not in prop_dic['mcmc_set']['nburn'][inst]):prop_dic['mcmc_set']['nburn'][inst][vis] = 200
       
        #Default model
        if ('model' not in prop_dic):prop_dic['model']={}
        if (inst not in prop_dic['model']):prop_dic['model'][inst]='gauss'
        
        #Binned data
        if bin_mode=='bin':   
            data_bin = np.load(gen_dic['save_data_dir']+data_type_gen+'bin_data/'+gen_dic['add_txt_path'][data_type_gen]+inst+'_'+vis_det+'_'+prop_dic['dim_bin']+'_add.npz',allow_pickle=True)['data'].item()
            fit_dic['n_exp'] = data_bin['n_exp']
            rest_frame = data_bin['rest_frame']
     
            #Defined fitted bins for each exposure     
            dim_exp = data_bin['dim_exp']
            nspec = dim_exp[1]       

        #Defined fitted bins for each exposure
        elif bin_mode=='':
            dim_exp = data_inst[vis]['dim_exp']
            nspec = data_inst[vis]['nspec']  
            rest_frame = data_dic[data_type_gen][inst][vis]['rest_frame']

        #Upload analytical surface RVs
        if (('DI' in data_type) and (inst in prop_dic['occ_range'])) or ('Intr' in data_type):
            if len(data_inst[vis]['transit_pl'])>1:stop('Adapt model to multiple planets')
            else:
                ref_pl = data_inst[vis]['transit_pl'][0]
                if bin_mode=='':surf_rv_mod = dataload_npz(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis)['achrom'][ref_pl]['rv'][0]
                else:surf_rv_mod = data_bin['achrom'][ref_pl]['rv'][0]
 
        #Disk-integrated profiles
        if 'DI' in data_type:
        
            #Original profiles
            if data_type=='DIorig':
                fit_dic['n_exp'] = data_inst[vis]['n_in_visit'] 
                idx_exp2in = gen_dic[inst][vis]['idx_exp2in']
                idx_in = gen_dic[inst][vis]['idx_in'] 

            #Binned profiles
            elif data_type=='DIbin':
                idx_exp2in = data_bin['idx_exp2in']
                idx_in = data_bin['idx_in'] 
           
                #Shift ranges from the solar barycentric rest frame (receiver, where they are defined) to the star barycentric rest frame (source)
                #      see gen_specdopshift() :
                # w_source = w_receiver / (1+ rv[s/r]/c))
                # w_starbar = w_solbar / (1+ rv[starbar/solbar]/c))  
                #           = w_solbar / (1+ sysvel/c))         
                #      where we neglect the Keplerian motion of the star in the definition of the ranges in the star rest frame
                if rest_frame=='star':
                    fit_range = np.array(fit_range)
                    cont_range = np.array(cont_range)
                    if ('spec' in data_mode):
                        conv_fact = 1./gen_specdopshift(data_bin['sysvel'])
                        fit_range*=conv_fact
                        cont_range*=conv_fact
                        if (trim_range is not None):trim_range = np.array(trim_range)*conv_fact
                    elif (data_mode=='CCF'):
                        fit_range-=data_bin['sysvel']
                        cont_range-=data_bin['sysvel']
                        if (trim_range is not None):trim_range = np.array(trim_range)-data_bin['sysvel']
                    if ('rv' in prop_dic['mod_prop']) and (inst in prop_dic['mod_prop']['rv']) and (vis in prop_dic['mod_prop']['rv'][inst]) and (vis_mode==''):prop_dic['mod_prop']['rv'][inst][vis]['guess']-=data_bin['sysvel']

            #Defined exposures
            iexp_def = range(fit_dic['n_exp'])

            #Definition of ranges covered by occulted stellar lines / for signal integration 
            if (inst in prop_dic['occ_range']) or ('EW' in prop_dic['meas_prop']) or ('biss' in prop_dic['meas_prop']):
                
                #RV shift from the star rest frame to the rest frame of the analyzed disk-integrated profiles
                if rest_frame=='star':
                    rv_star_starbar = np.repeat(0.,fit_dic['n_exp'])
                    rv_starbar_solbar = np.repeat(0.,fit_dic['n_exp'])
                else:
                    if 'orig' in data_type:
                        rv_star_starbar = coord_dic[inst][vis]['RV_star_stelCDM']
                        rv_starbar_solbar = data_dic['DI']['sysvel'][inst][vis]                       
                    elif 'bin' in data_type:
                        rv_star_starbar = data_bin['RV_star_stelCDM']
                        rv_starbar_solbar = data_bin['sysvel']
                    
                #Ranges for occulted stellar lines
                #    - 'occ_range' define the range covered by local stellar lines in the photosphere rest frame (source), and is shifted to the star or solar barycentric rest frame (receiver) if relevant 
                #       see gen_specdopshift() :
                # w_receiver = w_source * (1+ rv[s/r]/c))
                # w_star = w_photo * (1+ rv[photo/star]/c))
                # w_solbar = w_photo * (1+ rv[photo/star]/c)) * (1+ rv[star/starbar]/c)) * (1+ rv[starbar/solbar]/c))
                #      'line_range' define the maximum extension of the disk-integrated stellar line in the star rest frame (source), and is shifted to the solar barycentric rest frame (receiver) if relevant
                # w_solbar = w_star * (1+ rv[star/starbar]/c)) * (1+ rv[starbar/solbar]/c))              
                if (inst in prop_dic['occ_range']):
                    if ('spec' in data_mode):
                        specdopshift_star_solbar = gen_specdopshift(rv_star_starbar[idx_in])*gen_specdopshift(rv_starbar_solbar[idx_in])
                        occ_exclu_range = prop_dic['line_trans']*gen_specdopshift(np.array(prop_dic['occ_range'][inst])[:,None])*gen_specdopshift(surf_rv_mod)*specdopshift_star_solbar
                        line_range = prop_dic['line_trans']*gen_specdopshift(np.array(prop_dic['line_range'][inst])[:,None])*specdopshift_star_solbar
                    else:
                        specdopshift_star_solbar = rv_star_starbar[idx_in]+rv_starbar_solbar[idx_in]
                        occ_exclu_range = np.array(prop_dic['occ_range'][inst])[:,None] + (surf_rv_mod+specdopshift_star_solbar)   
                        line_range = np.array(prop_dic['line_range'][inst])[:,None] + specdopshift_star_solbar
                        
                #Ranges for signal integration
                #    - used in rv space
                #    - defined in the star rest frame 
                #    - if relevant, shifted from the star rest frame (source) to the solar barycentric rest frame (receiver)
                # rv(range/solbar) = rv(range/star) + rv(star/starbar) + rv(starbar/solbar)
                if ('EW' in prop_dic['meas_prop']):
                    prop_dic['EW_range_frame'] = np.array(prop_dic['meas_prop']['EW']['rv_range'])[:,None] + rv_star_starbar+rv_starbar_solbar
                if ('biss' in prop_dic['meas_prop']):
                    prop_dic['biss_range_frame'] = np.array(prop_dic['meas_prop']['biss']['rv_range'])[:,None] + rv_star_starbar+rv_starbar_solbar

        #Intrinsic profiles
        elif 'Intr' in data_type:

            #Original profiles
            if 'orig' in data_type:
                fit_dic['n_exp'] = data_inst[vis]['n_in_tr']
                iexp_def = prop_dic[inst][vis]['idx_def'] 
             
            #Binned profiles
            elif 'bin' in data_type:iexp_def = range(fit_dic['n_exp'])

            #Ranges for signal integration
            #    - used in rv space
            #    - defined in the photosphere rest frame  
            #    - if relevant, shifted from the photosphere rest frame (source) to the star rest frame (receiver)
            # rv(range/star) = rv(range/photo) + rv(photo/star)        
            if ('EW' in prop_dic['meas_prop']) or ('biss' in prop_dic['meas_prop']):
                if rest_frame=='surf':rv_photo_star = np.repeat(0.,fit_dic['n_exp'])
                else:rv_photo_star = surf_rv_mod
                if ('EW' in prop_dic['meas_prop']):prop_dic['EW_range_frame'] = np.array(prop_dic['meas_prop']['EW']['rv_range'])[:,None] + rv_photo_star
                if ('biss' in prop_dic['meas_prop']):prop_dic['biss_range_frame'] = np.array(prop_dic['meas_prop']['biss']['rv_range'])[:,None] + rv_photo_star

        #Atmospheric profiles
        elif 'Atm' in data_type:               

            #Original profiles / converted 2D->1D 
            if 'orig' in data_type:
                if len(data_inst[vis]['transit_pl'])>1:stop('Adapt model to multiple planets')
                
                #Defined exposures
                if prop_dic['pl_atm_sign']=='Absorption':fit_dic['n_exp'] = data_inst[vis]['n_in_tr']
                elif prop_dic['pl_atm_sign']=='Emission':fit_dic['n_exp'] = data_inst[vis]['n_in_visit']
                iexp_def = prop_dic[inst][vis]['idx_def']

            #Binned profiles
            elif 'bin' in data_type:iexp_def = range(fit_dic['n_exp'])

            #Ranges for signal integration
            #    - used in rv space
            #    - defined in the planet rest frame  
            #    - if relevant, shifted from the planet rest frame (source) to the star rest frame (receiver)
            # rv(range/pl) = rv(range/pl) + rv(pl/star)  
            if ('int_sign' in prop_dic['meas_prop']) or ('EW' in prop_dic['meas_prop']):
                if rest_frame=='pl':rv_pl_star = np.repeat(0.,fit_dic['n_exp'])
                else:
                    if 'orig' in data_type:
                        if prop_dic['pl_atm_sign']=='Absorption':rv_pl_star = coord_dic[inst][vis]['rv_pl'][idx_in]
                        elif prop_dic['pl_atm_sign']=='Emission':rv_pl_star = coord_dic[inst][vis]['rv_pl']
                    elif 'bin' in data_type:rv_pl_star = data_bin[ref_pl]['rv_pl']
                if ('int_sign' in prop_dic['meas_prop']):prop_dic['int_sign_range_frame'] = np.array(prop_dic['meas_prop']['int_sign']['rv_range'])[:,None] + rv_pl_star
                if ('EW' in prop_dic['meas_prop']):prop_dic['EW_range_frame'] = np.array(prop_dic['meas_prop']['EW']['rv_range'])[:,None] + rv_pl_star

        #Stellar line model
        if ('DI' in data_type) or ('Intr' in data_type):        

            #Custom model
            if (prop_dic['model'][inst]=='custom'):

                #Retrieve binned intrinsic profiles if required for custom model                
                if fit_properties['mode']=='Intrbin':
                    
                    #Check that profiles were aligned
                    if fit_properties['vis']=='':vis_Intrbin = vis
                    elif fit_properties['vis']=='binned':vis_Intrbin = 'binned'
                    data_Intrbin = dataload_npz(gen_dic['save_data_dir']+'Intrbin_data/'+gen_dic['add_txt_path'][data_type_gen]+inst+'_'+vis_Intrbin+'_'+fit_properties['dim_bin']+'_add')
                    if not data_Intrbin['FromAligned']:stop('Intrinsic profiles must be aligned before binning')
    
                    #Central coordinate of binned profiles along chosen dimension
                    fit_properties['cen_dim_Intrbin'] =data_Intrbin['cen_bindim']

            #Analytical model
            #    - intrinsic profile mode is set to analytical to activate conditions, even if intrinsic profiles are not used
            else:
                fit_properties['mode']='ana'

        #Trimming data
        #    - the trimming is applied to the common table, so that all processed profiles keep the same dimension after trimming
        if (trim_range is not None):
            
            #Common spectral table
            if vis_mode=='':data_com = dataload_npz(data_inst[vis]['proc_com_data_paths'])  
            elif vis_mode=='binned':data_com = dataload_npz(data_inst['proc_com_data_path'])
             
            #Trimmed range
            idx_range_kept = np_where1D((data_com['edge_bins'][iord_sel,0:-1]>=trim_range[0]) & (data_com['edge_bins'][iord_sel,1::]<=trim_range[1]))
            nspec = len(idx_range_kept)
            if nspec==0:stop('Empty trimmed range')
            
        else:
            idx_range_kept = np.arange(nspec,dtype=int)
    
        #Preparation
        cond_def_exp = np.zeros(fit_dic['n_exp'],dtype=bool)
        fit_dic['cond_def_cont_all']= np.zeros([fit_dic['n_exp'],nspec],dtype=bool)
        fit_dic['cond_def_fit_all']=np.zeros([fit_dic['n_exp'],nspec],dtype=bool)
        fit_dic['idx_excl_bd_ranges']={}
        data_fit = {}
        for isub,iexp in enumerate(iexp_def):
            cond_def_exp[isub] = True
            
            #Upload profile               
            if bin_mode=='':     data_fit_loc = dataload_npz(data_inst[vis]['proc_'+data_type_gen+'_data_paths']+str(iexp))
            elif bin_mode=='bin':data_fit_loc = np.load(gen_dic['save_data_dir']+data_type_gen+'bin_data/'+gen_dic['add_txt_path'][data_type_gen]+inst+'_'+vis_det+'_'+prop_dic['dim_bin']+str(iexp)+'.npz',allow_pickle=True)['data'].item()    
            data_fit[isub] = {}

            #Trimming profile         
            for key in ['cen_bins','flux','cond_def']:data_fit[isub][key] = data_fit_loc[key][iord_sel,idx_range_kept]
            data_fit[isub]['edge_bins'] = np.append(data_fit_loc['edge_bins'][iord_sel,idx_range_kept],data_fit_loc['edge_bins'][iord_sel,idx_range_kept[-1]+1]) 
            data_fit[isub]['dcen_bins'] = data_fit[isub]['edge_bins'][1::] - data_fit[isub]['edge_bins'][0:-1]          
            data_fit[isub]['cov'] = data_fit_loc['cov'][iord_sel][:,idx_range_kept]
            
            #Initializing ranges in the relevant rest frame
            if len(cont_range)==0:fit_dic['cond_def_cont_all'][isub] = True    
            else:
                for bd_int in cont_range:fit_dic['cond_def_cont_all'][isub] |= (data_fit[isub]['edge_bins'][0:-1]>=bd_int[0]) & (data_fit[isub]['edge_bins'][1:]<=bd_int[1])        
            if len(fit_range)==0:fit_dic['cond_def_fit_all'][isub] = True    
            else:
                for bd_int in fit_range:fit_dic['cond_def_fit_all'][isub] |= (data_fit[isub]['edge_bins'][0:-1]>=bd_int[0]) & (data_fit[isub]['edge_bins'][1:]<=bd_int[1])        

            #Accounting for undefined pixels
            fit_dic['cond_def_cont_all'][isub] &= data_fit[isub]['cond_def']            
            fit_dic['cond_def_fit_all'][isub] &= data_fit[isub]['cond_def']   

            #Exclusion of planetary ranges
            #    - not required for intrinsic profiles if already applied to their definition, and if not already applied contamination is either negligible or neglected
            #    - not required for binned disk-integrated profiles, as planetary ranges can be excluded from their construction
            #    - not required for atmospheric profiles for obvious reasons
            if (data_type=='DIorig') and ('DI_prof' in data_dic['Atm']['no_plrange']) and (iexp in data_dic['Atm'][inst][vis]['iexp_no_plrange']):   
                cond_kept_plrange,idx_excl_bd_ranges = excl_plrange(data_fit[isub]['cond_def'],data_dic['Atm'][inst][vis]['exclu_range_'+rest_frame],iexp,data_fit[isub]['edge_bins'],data_mode)
                fit_dic['idx_excl_bd_ranges'][isub] = idx_excl_bd_ranges
                fit_dic['cond_def_cont_all'][isub] &= cond_kept_plrange
                fit_dic['cond_def_fit_all'][isub]  &= cond_kept_plrange
            else:fit_dic['idx_excl_bd_ranges'][isub]=None
 
            #Exclusion of occulted stellar line range
            #    - relevant for disk-integrated profiles, in-transit
            #    - occulted lines never cover ranges outside of the disk-integrated stellar line
            if (data_type_gen=='DI') and (inst in data_dic['DI']['occ_range']) and (idx_exp2in[iexp]>-1): 
                i_in = idx_exp2in[iexp]
                cond_occ = (data_fit[isub]['edge_bins'][0:-1]>=np.max([occ_exclu_range[0,i_in],line_range[0,i_in]])) & (data_fit[isub]['edge_bins'][1:]<=np.min([occ_exclu_range[1,i_in],line_range[1,i_in]]))
                fit_dic['cond_def_cont_all'][isub,cond_occ] = False
                fit_dic['cond_def_fit_all'][isub,cond_occ] = False
       
        #Continuum common to all processed profiles
        #    - collapsed along temporal axis
        cond_cont_com  = np.all(fit_dic['cond_def_cont_all'][cond_def_exp],axis=0)
        if np.sum(cond_cont_com)==0.:stop('No pixels in common continuum')   

        #Common continuum flux in fitted intrinsic profiles
        #    - calculated over the defined bins common to all processed profiles
        #    - defined as a weighted mean because intrinsic profiles at the limbs can be very poorly defined due to the partial occultation and limb-darkening
        #    - we use the covariance diagonal to define a representative weight
        if (data_type_gen=='Intr'):
            cont_intr = np.zeros(fit_dic['n_exp'])*np.nan
            wcont_intr = np.zeros(fit_dic['n_exp'])*np.nan
            for isub,iexp in enumerate(iexp_def):
                dw_sum = np.sum(data_fit[isub]['dcen_bins'][cond_cont_com])
                cont_intr[isub] = np.sum(data_fit[isub]['flux'][cond_cont_com]*data_fit[isub]['dcen_bins'][cond_cont_com])/dw_sum
                wcont_intr[isub] = dw_sum**2./np.sum(data_fit[isub]['cov'][0,cond_cont_com]*data_fit[isub]['dcen_bins'][cond_cont_com]**2.)
            flux_cont=np.nansum(cont_intr*wcont_intr)/np.nansum(wcont_intr)
            
        #------------------------------------------------------------------------------------------

        #Retrieving binned intrinsic profiles for disk-integrated profile fitting
        #    - all binned profiles are defined over the same table
        if (data_type_gen=='DI') and (prop_dic['model'][inst]=='custom') and (fit_properties['mode']=='Intrbin'):
            fit_properties['edge_bins_Intrbin'] = np.append(data_Intrbin['edge_bins'][iord_sel,idx_range_kept],data_Intrbin['edge_bins'][iord_sel,idx_range_kept[-1]+1]) 
            dcen_bins_Intrbin = (fit_properties['edge_bins_Intrbin'][1::] - fit_properties['edge_bins_Intrbin'][0:-1])
            fit_properties['flux_Intrbin'] = np.zeros([data_Intrbin['n_exp'],nspec],dtype=float)
            cont_Intrbin = np.zeros(data_Intrbin['n_exp'])*np.nan
            wcont_Intrbin = np.zeros(data_Intrbin['n_exp'])*np.nan            
            for isub,iexp in enumerate(data_Intrbin['n_exp']):
                data_Intrbin_loc = np.load(gen_dic['save_data_dir']+'Intrbin_data/'+inst+'_'+vis_Intrbin+'_'+fit_properties['dim_bin']+str(iexp)+'.npz',allow_pickle=True)['data'].item()         
                if False in data_Intrbin_loc['cond_def'][iord_sel,idx_range_kept] :stop('Binned intrinsic profiles must be fully defined to be used in the reconstruction')    
                fit_properties['flux_Intrbin'][isub] = data_Intrbin_loc['flux'][iord_sel,idx_range_kept]
                cov_loc = data_Intrbin_loc['cov'][iord_sel][:,idx_range_kept]
                dw_sum = np.sum(dcen_bins_Intrbin[cond_cont_com])                
                cont_Intrbin[isub] = np.sum(fit_properties['flux_Intrbin'][isub][cond_cont_com]*dcen_bins_Intrbin[cond_cont_com])/dw_sum
                wcont_Intrbin[isub] = dw_sum**2./np.sum(cov_loc[0,cond_cont_com]*dcen_bins_Intrbin[cond_cont_com]**2.)

            #Scaling binned intrinsic profiles to a continuum unity
            flux_cont_Intrbin = np.sum(cont_Intrbin*wcont_Intrbin)/np.sum(wcont_Intrbin)
            fit_properties['flux_Intrbin']/=flux_cont_Intrbin

        #------------------------------------------------------------------------------------------

        #Process exposures    
        key_det = 'idx_force_det'+bin_mode+vis_mode
        fit_dic['cond_detected'] = np.repeat(True,fit_dic['n_exp'])
        for isub,iexp in enumerate(iexp_def):            

            #Disk-integrated profile
            if data_type_gen=='DI':                    
                if data_type=='DIorig':iexp_orig = iexp

                #Continuum flux
                flux_cont = np.sum(data_fit[isub]['flux'][cond_cont_com]*data_fit[isub]['dcen_bins'][cond_cont_com])/np.sum(data_fit[isub]['dcen_bins'][cond_cont_com])
              
                #Set constant error to the sqrt() of the continuum flux, ie covariance to the mean continuum flux
                #    - this is in case errors are not defined on disk-integrated profiles
                #    - the important is to use a constant error over the fitted range, its value can then be scaled using sc_err
                if prop_dic['cst_err'+bin_mode]:cov_exp = np.tile(flux_cont*gen_dic['g_err'][inst],[1,nspec])                         
                else:cov_exp = data_fit[isub]['cov']   
                
                #Estimate of CCF centroid
                if data_type=='DIorig':fit_properties['RV_cen'] = coord_dic[inst][vis]['RV_star_solCDM'][iexp_orig]
                elif data_type=='DIbin':
                    if gen_dic['align_DI']:fit_properties['RV_cen'] = 0. 
                    else:fit_properties['RV_cen']=data_bin['sysvel']
                
            #Intrinsic profile
            #    - see proc_intr_data(), intrinsic CCFs have the same continuum flux level, reset to the mean over all intrinsic profiles
            #      we here fix the continuum level of the model to this value, rather than fit it in each exposure 
            #    - errors are defined, so that if the fit is to be performed with a constant error we set it to the mean error over the continuum
            elif data_type_gen=='Intr':                                  
                if data_type=='Introrig':iexp_orig = gen_dic[inst][vis]['idx_in'][iexp]
                if prop_dic['cst_err'+bin_mode]:cov_exp = np.tile(np.mean(data_fit[isub]['cov'][0])*gen_dic['g_err'][inst],[1,nspec])
                else:cov_exp = data_fit[isub]['cov']

            #Atmospheric profile
            elif data_type_gen=='Atm':   
                flux_cont = 0.                  
                if data_type=='Atmorig': 
                    if prop_dic['pl_atm_sign']=='Absorption':iexp_orig = gen_dic[inst][vis]['idx_in'][iexp]
                    elif prop_dic['pl_atm_sign']=='Emission':iexp_orig = iexp
                    fit_properties['RV_cen']=coord_dic[inst][vis]['rv_pl'][iexp_orig]     #guess
                if prop_dic['cst_err'+bin_mode]:cov_exp = np.tile(np.mean(data_fit[isub]['cov'][0]),[1,nspec])
                else:cov_exp = data_fit[isub]['cov']

            #Scaling data errors
            if (inst in prop_dic['sc_err']) and (vis_det in prop_dic['sc_err'][inst]):cov_exp*=prop_dic['sc_err'][inst][vis]**2. 


            #Forced detection
            if (inst in prop_dic[key_det]) and (vis_det in prop_dic[key_det][inst]) and (iexp in prop_dic[key_det][inst][vis_det]):idx_force_det=prop_dic[key_det][inst][vis_det][iexp]
            else:idx_force_det=None 
          
            #-------------------------------------------------

            #Perform analysis of individual profile
            #    - profiles are fitted on their original table, converted in RV space for single spectral line fitted with analytical models
            #    - analytical models can be calculated on an oversampled table and resampled before the fit
            fit_properties.update({'iexp':iexp,'flux_cont':flux_cont})
            fit_dic[iexp]=single_anaprof(isub,iexp,inst,data_dic,vis_det,prop_dic,gen_dic,prop_dic['verbose'],fit_dic['cond_def_fit_all'][isub],fit_dic['cond_def_cont_all'][isub] ,data_type_gen,data_fit[isub]['edge_bins'],data_fit[isub]['cen_bins'],data_fit[isub]['flux'],cov_exp,
                                        idx_force_det,theo_dic,star_param,fit_properties,prop_dic['line_fit_priors'],prop_dic['model'][inst],prop_dic['mod_prop'],data_mode)
            
            #-------------------------------------------------

            #Detection flag
            fit_dic['cond_detected'][iexp] = fit_dic[iexp]['detected']

            #Calculating residuals from Keplerian (km/s)
            #    - we report the errors on the raw velocities 
            if data_type=='DIorig':
                fit_dic[iexp]['RVmod']=coord_dic[inst][vis]['RV_star_solCDM'][iexp]
                if ('rv_pip' in prop_dic[inst][vis]):
                    fit_dic[iexp]['rv_pip_res']=prop_dic[inst][vis]['rv_pip'][iexp]-fit_dic[iexp]['RVmod']
                    fit_dic[iexp]['err_rv_pip_res']=np.repeat(prop_dic[inst][vis]['erv_pip'][iexp],2)  
                if prop_dic['model'][inst]=='dgauss':
                    fit_dic[iexp]['RV_lobe_res']=fit_dic[iexp]['RV_lobe']-fit_dic[iexp]['RVmod']
                    fit_dic[iexp]['err_RV_lobe_res']=fit_dic[iexp]['err_RV_lobe']  

            elif data_type=='DIbin':    
                fit_dic[iexp]['RVmod']=data_bin['RV_star_solCDM'][iexp]

            #Calculating residuals from RRM model (km/s)
            #    - we report the errors on the local velocities
            elif data_type=='Introrig':
                fit_dic[iexp]['RVmod']=surf_rv_mod[iexp] 
               
            #Calculating residuals from orbital RVs (km/s)
            elif data_type=='Atmorig': 
                fit_dic[iexp]['RVmod']=coord_dic[inst][vis]['rv_pl'][iexp_orig]
            
            #Errors are reported from measured velocities
            fit_dic[iexp]['rv_res']=fit_dic[iexp]['rv']-fit_dic[iexp]['RVmod']
            if prop_dic['fit_mode']!='':fit_dic[iexp]['err_rv_res']=fit_dic[iexp]['err_rv']
            else:fit_dic[iexp]['err_rv_res'] = np.nan

        #Systemic rv
        if prop_dic['fit_mode']!='':

            #From master (rv(CDM/sun) in km/s)
            #    - for CCFs, the disk-integrated master is centered in the CDM rest frame, hence its fit returns the systemic velocity         
            #      we assume the master is defined well enough that the uncertainty on the velocity v(CDM/sun) is negligible
            if (data_type=='DIbin') and ((vis_mode=='multivis') or (fit_dic['n_exp']==1)):
                rv_sys,erv_sys=0.,0.
                for iexp in iexp_def:
                    rv_sys+=fit_dic[iexp]['rv']
                    erv_sys+=np.mean(fit_dic[iexp]['err_rv'])**2.
                rv_sys/=fit_dic['n_exp']
                erv_sys = np.sqrt(erv_sys)/fit_dic['n_exp']
                print('         Systemic rv from master profile =',"{0:.6f}".format(rv_sys),'+-',"{0:.6e}".format(erv_sys),'km/s')
            
            #From time-series
            elif data_type=='DIorig':
                rv_res_out = []
                erv_res_out = []
                for iexp in gen_dic[inst][vis]['idx_out']:
                    rv_res_out+=[fit_dic[iexp]['rv_res']]
                    erv_res_out+=[np.mean(fit_dic[iexp]['err_rv_res'])]
                weights = 1./np.array(erv_res_out)**2.
                rv_sys = np.sum(np.array(rv_res_out)*weights)/np.sum(weights)
                erv_sys = 1./np.sqrt(np.sum(weights))
                print('         Systemic rv from time-series =',"{0:.6f}".format(rv_sys),'+-',"{0:.6e}".format(erv_sys),'km/s')


        #Saving data
        fit_dic['cont_range'] = cont_range
        fit_dic['fit_range'] = fit_range        
        np.savez_compressed(save_path,data=fit_dic,allow_pickle=True)
        
    #Checking data has been calculated
    else:
        data_paths={'path':save_path}      
        check_data(data_paths)    
                                  
    return None






def single_anaprof(isub_exp,iexp,inst,data_dic,vis,fit_prop_dic,gen_dic,verbose,cond_def_fit,cond_def_cont,prof_type,  
                  edge_bins,cen_bins,flux_loc,cov_loc,idx_force_det,theo_dic,star_params,fit_properties,line_fit_priors,model_choice,model_prop,data_type): 
    r"""**Single profile analysis routine**

    Performs fits and measurements of line profiles.
    
    Args:
        TBD

    Returns:
        TBD
        
    """  
    
    #Arguments to be passed to the fit function
    fixed_args = deepcopy(fit_properties)
    output_prop_dic={}

    #Fitted data tables 
    #    - to avoid issues with gaps, resampling/convolution, and chi2 calculation the model is calculated on the full continuous velocity table, and then limited to fitted pixels
    #    - fitted profiles are trimmed and calculated over the smallest continuous range encompassing all fitted pixels to avoid computing models over a too-wide table
    #    - final model is calculated on the same bins as the input data, limited to the minimum global definition range for the fit
    idx_def_fit = np_where1D(cond_def_fit)                  #trimmed and fit indexes in original table
    if len(idx_def_fit)==0.:stop('No bin in fitted range')
    cen_bins_fit = cen_bins[cond_def_fit]                   #trimmed and fitted values
    flux_loc_fit=flux_loc[cond_def_fit]                     #trimmed and fitted values
    idx_mod = range(idx_def_fit[0],idx_def_fit[-1]+1)       
    fixed_args['idx_fit'] = np_where1D(cond_def_fit[idx_mod])         #trimmed and fitted indexes, reduced to model indexes    
    output_prop_dic['idx_mod'] = idx_mod      
    fixed_args['ncen_bins'] = len(idx_mod)
    fixed_args['cen_bins']= cen_bins[idx_mod]
    fixed_args['dim_exp']= [1,idx_mod]
    fixed_args['edge_bins'] = def_edge_tab(fixed_args['cen_bins'][None,:][None,:])[0,0]
    fixed_args['dcen_bins'] = fixed_args['edge_bins'][1::]-fixed_args['edge_bins'][0:-1] 
    fixed_args['x_val']=cen_bins[idx_mod]
    fixed_args['y_val']=flux_loc[idx_mod]
    fixed_args['cov_val'] = cov_loc[:,idx_mod]
    
    #Activation of spectral conversion and resampling 
    cond_conv_st_prof_tab(theo_dic['rv_osamp_line_mod'],fixed_args,data_type)

    #Resampled spectral table for model line profile
    if fixed_args['resamp']:resamp_st_prof_tab(None,None,None,fixed_args,gen_dic,1,theo_dic['rv_osamp_line_mod'])
    
    #Effective instrumental convolution
    if fixed_args['conv_model']:fixed_args['FWHM_inst'] = get_FWHM_inst(inst,fixed_args,fixed_args['cen_bins'])    
    else:fixed_args['FWHM_inst'] = None

    #Effective table for model calculation
    fixed_args['args_exp'] = def_st_prof_tab(None,None,None,fixed_args)

    #------------------------------------------------------------------------------- 
    #Guess and prior values
    #------------------------------------------------------------------------------- 

    #-----------------------------------
    #Identification of CCF peak
    #-----------------------------------
    if (prof_type=='Intr'):
        
        #For intrinsic CCFs we assume the local stellar RV is bounded by +=0.5*FWHM, with FWHM an estimate given as prior, and take the RV at minimum in this range   
        if ('FWHM' in model_prop):
            cond_peak=(cen_bins_fit>-0.5*model_prop['FWHM'][inst][vis]['guess']) & (cen_bins_fit<0.5*model_prop['FWHM'][inst][vis]['guess'])
            if np.sum(cond_peak)==0:CCF_peak=0.            
            else:CCF_peak=min(flux_loc_fit[cond_peak])
        else:CCF_peak=min(flux_loc_fit)
        
    elif prof_type=='DI':
        
        #For disk-integrated CCFs we take the RV at minimum as guess 
        CCF_peak=min(flux_loc_fit)
        
    elif prof_type=='Atm':
        
        #For atmospheric CCFs we take the maximum as guess
        CCF_peak=max(flux_loc_fit)

    #-----------------------------------
    #Centroid velocity (km/s) 
    #-----------------------------------
    RV_guess_tab = [None,None,None]
    
    #Estimate from system properties
    if ('RV_cen' in fixed_args):RV_guess_tab[0] = fixed_args['RV_cen']
    
    #Estimate from measured CCF peak
    else:
        if CCF_peak==0.:RV_guess_tab[0]=np.mean(cen_bins_fit)
        else:RV_guess_tab[0]=cen_bins_fit[flux_loc_fit==CCF_peak][0]
        
    #Guess value from input
    #    - overwrites default values
    if ('rv' in model_prop) and (inst in model_prop['rv']) and (vis in model_prop['rv'][inst]):RV_guess_tab[0] = model_prop['rv'][inst][vis]['guess']
    
    #Prior range  
    if prof_type=='DI':
        RV_guess_tab[1] = RV_guess_tab[0]-200.
        RV_guess_tab[2] = RV_guess_tab[0]+200.       
    elif prof_type=='Intr':
        RV_guess_tab[1] = -3.*star_params['vsini']
        RV_guess_tab[2] = +3.*star_params['vsini']
    elif prof_type=='Atm':
        RV_guess_tab[1] = RV_guess_tab[0]-50.
        RV_guess_tab[2] = RV_guess_tab[0]+50.

    #Guesses for analytical models
    if fixed_args['mode']=='ana':
            
        #-----------------------------------    
        #FWHM (km/s)
        #    - sigma = FWHM / (2*sqrt(2*ln(2)))
        #    - find the approximate position of CCF, and then FWHM (for master out only, otherwise overestimate the FWHM)
        #      for intrinsic CCFs we start from the FWHM of the master out if known
        #    - if there is no signature of the intrinsic CCF, the minimum of the CCF can be found at the edges of the table
        #    - set to 10km/s for atmospheric CCFs
        #-----------------------------------
        if (prof_type in ['DI','Intr']): 
            idx_low=np.where(cen_bins_fit < RV_guess_tab[0])[0]
            if len(idx_low)==0:vlow_FWHM=cen_bins_fit[0]
            else:
                ilow_FWHM=closest(flux_loc_fit[idx_low],0.5*(CCF_peak+fixed_args['flux_cont']))
                vlow_FWHM=cen_bins_fit[idx_low][ilow_FWHM]
            idx_high=np.where(cen_bins_fit >= RV_guess_tab[0])[0]
            if len(idx_high)==0:vhigh_FWHM=cen_bins_fit[-1]
            else:
                ihigh_FWHM=closest(flux_loc_fit[idx_high],0.5*(CCF_peak+fixed_args['flux_cont']))
                vhigh_FWHM=cen_bins_fit[idx_high][ihigh_FWHM]     
            FWHM_guess=(vhigh_FWHM - vlow_FWHM)
        elif (prof_type=='Atm'):
            FWHM_guess=10.
            
        #Upper boundary
        if prof_type=='DI':
            FWHM_max = 200.
        elif prof_type=='Intr':
            FWHM_max = 100.
        elif prof_type=='Atm':
            FWHM_max = 50.
            
        #-----------------------------------
        #Guess of CCF contrast  
        #    - contrast is considered as always positive, thus we define a sign for the amplitude depending on the case
        #-----------------------------------

        ctrst_guess = -(CCF_peak-fixed_args['flux_cont'])/fixed_args['flux_cont']    
        if prof_type in ['DI','Intr']:fixed_args['amp_sign']=-1.
        elif prof_type=='Atm':fixed_args['amp_sign']=1.

    #-------------------------------------------------------------------------------    
    #Parameters and functions
    #    - lower / upper values in 'p_start' will be used as default uniform prior ranges if none are provided as inputs
    #                                                     as default walkers starting ranges (if mcmc fit)
    #-------------------------------------------------------------------------------    

    # Initialise model parameters
    p_start = Parameters()

    #Generic fit function
    fixed_args['fit_func']=gen_fit_prof
    fixed_args['inside_fit'] = False 

    #-------------------------------------------------------------------------------
    #Custom model for disk-integrated profiles
    #    - obtained via numerical integration over the disk
    #    - intrinsic CCFs do not necessarily have here the same flux level as the fitted disk-integrated profile, so that the continuum level should be left as a free parameter
    if (model_choice=='custom') and (prof_type=='DI'):
        
        #Custom model initialization
        fixed_args.update({
            'mac_mode':theo_dic['mac_mode'], 
            'inst':inst,
            'vis':vis})
        
        #Assuming no spots are fitted
        fixed_args['unquiet_star'] = None

        #Grid initialization
        fixed_args['DI_grid'] = True
        fixed_args['conv2intr'] = False
        p_start = init_custom_DI_par(fixed_args,gen_dic,data_dic[inst]['system_prop'],star_params,p_start,RV_guess_tab)

        #Model function
        fixed_args['func_nam']=custom_DI_prof

    #Analytical models
    else:
        
        #Continuum
        p_start.add_many(('cont',    fixed_args['flux_cont'], False,   None,None,None))      
        
        #Centroid
        p_start.add_many(('rv',      RV_guess_tab[0],      True,   RV_guess_tab[1], RV_guess_tab[2],     None)) 

        #Contrast and FWHM
        p_start.add_many(('ctrst',   ctrst_guess,   True,   0.,     1. ,        None),              
                         ('FWHM',    FWHM_guess,    True,   0.,     FWHM_max,   None))        
   
        #Simple inverted gaussian or Voigt profile
        if (model_choice in ['gauss','voigt']):
            
            #Fit parameters 
            p_start.add_many(('skewA',   0.0,           False,   -1e3,1e3,None),
                             ('kurtA',   0.0,           False,   -100.,100.,None))        

            #Continuum
            for ideg in range(1,5):p_start.add_many(('c'+str(ideg)+'_pol',          0.,              False,    None,None,None)) 
        
            #Simple inverted gaussian
            #    - the main component of the model is an inverted gaussian with constant baseline
            #    - for fast rotators or sloped continuum use skewness and slope properties for the gaussian
            if (model_choice=='gauss'):
                fixed_args['func_nam']=gauss_herm_lin 
                
            #Inverted Voigt profile
            elif (model_choice=='voigt'):
                fixed_args['func_nam']=voigt  
                p_start.add_many((  'a_damp',  1., True,    0.,       1e15,  None))
    
            # #Complex prior function     -> left as an example, but now that contrast is a direct parameter a function is not required anymore
            # if (fit_dic['fit_mode']=='mcmc') and (len(fit_prop_dic['prior_func'])>0):
                
            #     def global_ln_prior_func(p_step_loc,args_loc):
            #         ln_p_loc = 0.
            
            #         #Uniform prior on contrast between 0 and 1
            #         #    - contrast is defined as C = amplitude / mean 
            #         ctrst_loc =args_loc['amp_sign']*p_step_loc['amp']/p_step_loc['cont']
            #         if (ctrst_loc<0) or (ctrst_loc>1):ln_p_loc += -np.inf
    
            #         return ln_p_loc 
                
            #     fixed_args['global_ln_prior_func']=global_ln_prior_func          
    
        #-------------------------------------------------------------------------------
        #Flat continuum with 6th-order polynomial at center, then added to an inverted gaussian
        #    - polynomial is made continuous in value and derivative, at a velocity value symmetric with respect to the centroid RV
        elif (model_choice=='gauss_poly'):
            
            #Guess of distance from centroid where lobes end and continuum starts
            #    - typically about three times the FWHM
            dRV_joint=3.*FWHM_guess
            
            #Polynomial coefficients
            #    - center of polynomial continuum is at cont + a4*dx^4, and set to the maximum of the CCF:
            # a4 = (max - cont)/dx^4
            #    - we assume here that the 6th order coefficient is null to estimate the guess
            c4_pol=(max(flux_loc_fit)- fixed_args['flux_cont'])/ dRV_joint**4.        
            c6_pol=0.
            
            #Parameters
            p_start.add_many( ( 'dRV_joint',    dRV_joint,      True,   0.,None,  None),
                              ( 'c4_pol',       c4_pol,         True,   None,None,  None),
                              ( 'c6_pol',       c6_pol,         True,   None,None,  None))
    
            #Model function
            fixed_args['func_nam']=gauss_poly
            
        #-------------------------------------------------------------------------------
        #-------------------------------------------------------------------------------
        #Gaussian whose wings are the continuum added to an inverted gaussian
        #    - positions, widths and contrast are linked 
        elif (model_choice=='dgauss'):
    
            #Amplitude ratio 
            if 'amp_l2c' in model_prop:amp_l2c = model_prop['amp_l2c'][inst][vis]['guess']
            else:amp_l2c=0.5
            
            #FWHM ratio
            if 'FWHM_l2c' in model_prop:FWHM_l2c = model_prop['FWHM_l2c'][inst][vis]['guess']
            else:FWHM_l2c=2.
                
            #Shift between the RV centroids of the two components
            if 'rv_l2c' in model_prop:rv_l2c = model_prop['rv_l2c'][inst][vis]['guess']
            else:rv_l2c=0.
    
            #Contrast guess
            amp_core=2.*(CCF_peak-fixed_args['flux_cont'])  
            ctrst_guess = -amp_core*( 1. - amp_l2c) / fixed_args['flux_cont']
    
            #-----------------------
            
            #Parameters
            p_start.add_many(( 'rv_l2c',        rv_l2c,             True ,  None ,  None,  None), 
                             ( 'amp_l2c',       amp_l2c,            True ,  0.,None,  None),
                             ( 'FWHM_l2c',      FWHM_l2c,           True ,  0.,None,  None))


            #Continuum
            for ideg in range(1,5):p_start.add_many(('c'+str(ideg)+'_pol',          0.,              False,    None,None,None)) 
        
            #Model function
            fixed_args['func_nam']=dgauss

    ######################################################################################################## 

    #Fit dictionary
    fit_dic={
        'merit':{},
        'fit_mode':fit_prop_dic['fit_mode'],
        'print_par':fit_prop_dic['print_par'],
        'verb_shift':'',
        'uf_bd':{},
        'nx_fit':len(fixed_args['y_val'])
        }
    fixed_args['fit'] = {'chi2':True,'':False,'mcmc':True}[fit_prop_dic['fit_mode']]

    #Parameter initialization
    p_start = par_formatting(p_start,model_prop,line_fit_priors,fit_dic,fixed_args,inst,vis,fixed_args['mode'])

    #Model initialization
    #    - must be done after the final parameter initialization
    if (model_choice=='custom') and (prof_type=='DI'):
        fixed_args = init_custom_DI_prof(fixed_args,gen_dic,data_dic[inst]['system_prop'],data_dic['DI']['spots_prop'],theo_dic,star_params,p_start)
        
        #Flux grid will be updated within custom_DI_prof() if args['fit'] and args['var_star_grid']
        if (not fixed_args['fit']) or (not fixed_args['var_star_grid']):
            fixed_args['Fsurf_grid_spec'] = theo_intr2loc(fixed_args['grid_dic'],fixed_args['system_prop'],fixed_args['args_exp'],fixed_args['args_exp']['ncen_bins'],fixed_args['grid_dic']['nsub_star'])         
        
    #Fit initialization
    fit_dic['save_dir'] = fixed_args['save_dir']+'iexp'+str(fixed_args['iexp'])+'/'
    init_fit(fit_dic,fixed_args,p_start,fit_prop_dic,model_par_names,model_par_units)     

    #--------------------------------------------------------------
    #Fit by chi2 minimization
    if fit_prop_dic['fit_mode']=='chi2':
        p_final = call_lmfit(p_start,fixed_args['x_val'],fixed_args['y_val'],fixed_args['cov_val'],fixed_args['fit_func'],verbose=verbose,fixed_args=fixed_args)[2]
     
    #--------------------------------------------------------------   
    #Fit by emcmc 
    elif fit_prop_dic['fit_mode']=='mcmc': 
        
        #Calculate HDI for error definition
        #    - automatic definition of PDF resolution is used unless histogram resolution is set
        fit_dic['HDI_dbins']= {}
        fit_dic['HDI_bwf']= {}
        for param_loc in fixed_args['var_par_list']:
            if ('HDI_dbins' in fit_prop_dic) and (param_loc in fit_prop_dic['HDI_dbins']) and (inst in fit_prop_dic['HDI_dbins'][param_loc]) and (vis in fit_prop_dic['HDI_dbins'][param_loc][inst]):
                fit_dic['HDI_dbins'][param_loc]=fit_prop_dic['HDI_dbins'][param_loc][inst][vis]
            elif ('HDI_bwf' in fit_prop_dic) and (param_loc in fit_prop_dic['HDI_bwf']) and (inst in fit_prop_dic['HDI_bwf'][param_loc]) and (vis in fit_prop_dic['HDI_bwf'][param_loc][inst]):
                fit_dic['HDI_bwf'][param_loc]=fit_prop_dic['HDI_bwf'][param_loc][inst][vis]
    
        #Store options
        for key in ['nwalkers','nsteps','nburn']:fit_dic[key] = fit_prop_dic['mcmc_set'][key][inst][vis]

        #Run MCMC
        if fit_prop_dic['mcmc_run_mode']=='use':
            walker_chains=call_MCMC(gen_dic['fit_prof_nthreads'],fixed_args,fit_dic,verbose=verbose)
                           
        #Reuse MCMC
        elif fit_prop_dic['mcmc_run_mode']=='reuse':
            walker_chains=np.load(fit_dic['save_dir']+'/raw_chains_walk'+str(fit_prop_dic['mcmc_set']['nwalkers'][inst][vis])+'_steps'+str(fit_prop_dic['mcmc_set']['nsteps'][inst][vis])+fit_dic['run_name']+'.npz')['walker_chains']  
 
        #Excluding parts of the chains
        if fit_dic['exclu_walk']:
            if gen_dic['star_name'] == 'HD106315':
                wgood=np_where1D((np.min(walker_chains[:,400::,np_where1D(fixed_args['var_par_list']=='veq')],axis=1)>8.))

            walker_chains=np.take(walker_chains,wgood,axis=0)     
            fit_dic['nwalkers']=len(wgood) 
 
    
        #Processing:
        p_final,merged_chain,par_sample_sig1,par_sample=postMCMCwrapper_1(fit_dic,fixed_args,walker_chains,gen_dic['fit_prof_nthreads'],fixed_args['par_names'],verbose=verbose)

    #--------------------------------------------------------------   
    #Fixed model
    elif fit_prop_dic['fit_mode']=='': 
        p_final = p_start
  
    #Merit values     
    p_final=fit_merit(p_final,fixed_args,fit_dic,verbose)    

    ########################################################################################################    
    #Post-processing
    ########################################################################################################   
    fixed_args['fit'] = False 

    #Store outputs
    output_prop_dic['BIC']=fit_dic['merit']['BIC']    
    output_prop_dic['red_chi2']=fit_dic['merit']['red_chi2'] 
 
    #Best-fit model to the full profile (over the original fit range) at the observed resolution
    output_prop_dic['edge_bins'] = fixed_args['edge_bins']
    output_prop_dic['cen_bins'] = fixed_args['cen_bins']
    fixed_args['cond_def_fit']=np.repeat(True,fixed_args['ncen_bins'])    
    output_prop_dic['cond_def'] = np.ones(fixed_args['ncen_bins'],dtype=bool)
    output_prop_dic['flux']=fixed_args['fit_func'](p_final,None,args=fixed_args)
  
    #Double gaussian model: output the two components
    if (model_choice=='dgauss'):
        output_prop_dic['gauss_core'],output_prop_dic['gauss_lobe']=fixed_args['func_nam'](p_final,cen_bins)[1:3] 
        if fixed_args['FWHM_inst'] is not None:
            output_prop_dic['gauss_core'] = convol_prof(output_prop_dic['gauss_core'],cen_bins,fixed_args['FWHM_inst'])  
            output_prop_dic['gauss_lobe'] = convol_prof(output_prop_dic['gauss_lobe'],cen_bins,fixed_args['FWHM_inst']) 

    #Gaussian + polynomial model: output the two components
    elif (model_choice=='gauss_poly'):
        output_prop_dic['gauss_core'],output_prop_dic['poly_lobe']=fixed_args['func_nam'](p_final,cen_bins)[1:3] 
        if fixed_args['FWHM_inst'] is not None:
            output_prop_dic['gauss_core'] = convol_prof(output_prop_dic['gauss_core'],cen_bins,fixed_args['FWHM_inst'])  
            output_prop_dic['poly_lobe'] = convol_prof(output_prop_dic['poly_lobe'],cen_bins,fixed_args['FWHM_inst']) 
            
    #Custom model: output the line without continuum
    elif (model_choice=='custom'):
        output_prop_dic['core'],output_prop_dic['core_norm']=fixed_args['func_nam'](p_final,cen_bins,args=fixed_args)[1:3]
        if fixed_args['FWHM_inst'] is not None:
            output_prop_dic['core'] = convol_prof(output_prop_dic['core'],cen_bins,fixed_args['FWHM_inst'])  
            output_prop_dic['core_norm'] = convol_prof(output_prop_dic['core_norm'],cen_bins,fixed_args['FWHM_inst']) 

    #---------------------------------------------------------------------------------------------------

    #Best-fit model to the full line profile, including instrumental convolution if requested
    #    - observed tables are not used anymore and are overwritten in fixed_args
    if (inst not in fit_prop_dic['best_mod_tab']):fit_prop_dic['best_mod_tab'][inst]={}
    if 'dx' not in fit_prop_dic['best_mod_tab'][inst]:dx_mod = np.median(fixed_args['dcen_bins'])
    else:dx_mod=fit_prop_dic['best_mod_tab'][inst]['dx']
    if 'min_x' not in fit_prop_dic['best_mod_tab'][inst]:min_x = fixed_args['edge_bins'][0]
    else:min_x=fit_prop_dic['best_mod_tab'][inst]['min_x']
    if 'max_x' not in fit_prop_dic['best_mod_tab'][inst]:max_x = fixed_args['edge_bins'][-1]
    else:max_x=fit_prop_dic['best_mod_tab'][inst]['max_x']
    fixed_args['ncen_bins'] =  int((max_x-min_x)/dx_mod) 
    dx_mod = (max_x-min_x)/fixed_args['ncen_bins'] 
    fixed_args['edge_bins'] = min_x+np.arange(fixed_args['ncen_bins']+1)*dx_mod    
    fixed_args['cen_bins'] = 0.5*(fixed_args['edge_bins'][0:-1]+fixed_args['edge_bins'][1::])
    fixed_args['dim_exp']= [1,fixed_args['ncen_bins']]
    fixed_args['dcen_bins'] = np.repeat(dx_mod,fixed_args['ncen_bins'])
    fixed_args['cond_def_fit']=np.repeat(True,fixed_args['ncen_bins']) 
    
    #Deactivate resampling 
    #    - model resolution can be directly adjusted
    fixed_args['resamp'] = False
    fixed_args['args_exp'] = def_st_prof_tab(None,None,None,fixed_args)
    
    #Custom model
    if (model_choice=='custom') and (prof_type=='DI'):
        fixed_args = init_custom_DI_prof(fixed_args,gen_dic,data_dic[inst]['system_prop'],theo_dic,star_params,p_final)
        fixed_args['Fsurf_grid_spec'] = theo_intr2loc(fixed_args['grid_dic'],fixed_args['system_prop'],fixed_args['args_exp'],fixed_args['args_exp']['ncen_bins'],fixed_args['grid_dic']['nsub_star']) 
            
    #Store final model
    output_prop_dic['cen_bins_HR']=fixed_args['cen_bins']       
    output_prop_dic['flux_HR']=fixed_args['fit_func'](p_final,None,args=fixed_args)

    ########################################################################################################             
    #Derived parameters
    #    - with chi2 fit: best-fit value and error of the derived parameter are defined here 
    #      with mcmc fit: the chain of the derived parameter is defined here, and its best-fit value and error are then derived in postMCMCwrapper_2()
    ########################################################################################################  
    if (fit_prop_dic['thresh_area'] is not None) or (fit_prop_dic['thresh_amp'] is not None):fit_prop_dic['deriv_prop'] += ['amp'] 
    if len(fit_prop_dic['deriv_prop'])>0:
        
        if (model_choice in ['gauss','voigt']):
            
            #Amplitude
            if (('amp' in fit_prop_dic['deriv_prop']) or ('area' in fit_prop_dic['deriv_prop'])):
                
                #Contrast is defined as C = 1 - ( CCF minimum / mean continuum flux)
                #                       C = (mean continuum flux -  CCF minimum) / mean continuum flux)        
                #                       C = -amp / cont 
                #Amplitude is defined as amp = -C*cont  
                if fit_dic['fit_mode'] in ['chi2','']:
                    p_final['amp']=fixed_args['amp_sign']*p_final['ctrst']*p_final['cont']                
                    if fit_dic['fit_mode']=='chi2':                   
                        #d[A]  = sqrt( (err(C)/cont)^2 + (err(cont)/C)^2 )   
                        if 'ctrst' in fixed_args['var_par_list']:err_ctrst= fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='ctrst')[0]]  
                        else:err_ctrst=0.
                        if 'cont' in fixed_args['var_par_list']:err_cont= fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='cont')[0]]  
                        else:err_cont=0.
                        sig_loc=np.sqrt( (err_ctrst/p_final['cont'])**2. + (err_cont/p_final['ctrst'])**2. )  
                        fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))  
                elif fit_dic['fit_mode']=='mcmc':    
                    if 'ctrst' in fixed_args['var_par_list']:ctrst_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='ctrst')[0]]
                    else:cont_chain=p_final['ctrst']
                    if 'cont' in fixed_args['var_par_list']:cont_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='cont')[0]]
                    else:cont_chain=p_final['cont']
                    chain_loc=fixed_args['amp_sign']*ctrst_chain*cont_chain
                    merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)            
                fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'amp')
                fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],'Amp') 
                fixed_args['var_par_units']=np.append(fixed_args['var_par_units'],'')      

            #Integral under the gaussian model
            #    - formula valid if there is no asymetry to the gaussian shape:
            # A = 0.5*amp*FWHM*sqrt(pi/ln(2))
            if ('area' in fit_prop_dic['deriv_prop']) and (model_choice=='gauss') and (p_final['skewA']==0.) and (p_final['kurtA']==0.):    
                if fit_dic['fit_mode'] in ['chi2','']:
                    p_final['area']=np.abs(p_final['amp'])*p_final['FWHM']*np.sqrt(np.pi)/(2.*np.sqrt(np.log(2))) 
                    if fit_dic['fit_mode']=='chi2':                 
                        if 'amp' in fixed_args['var_par_list']:err_amp= fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='amp')[0]]  
                        else:err_amp=0.
                        if 'FWHM' in fixed_args['var_par_list']:err_FWHM= fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='FWHM')[0]]  
                        else:err_FWHM=0.
                        #d[A]  = 0.5*sqrt( (err(amp)*FWHM)^2 + (err(FWHM)*amp)^2 )*sqrt(pi/ln(2))
                        sig_loc=np.sqrt( (err_amp*p_final['FWHM'])**2. + (err_FWHM*p_final['amp'])**2. )*np.sqrt(np.pi)/(2.*np.sqrt(np.log(2)))   
                        fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))  
                elif fit_dic['fit_mode']=='mcmc': 
                    if 'amp' in fixed_args['var_par_list']:amp_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='amp')[0]]
                    else:amp_chain=p_final['amp']
                    if 'FWHM' in fixed_args['var_par_list']:FWHM_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='FWHM')[0]]
                    else:FWHM_chain=p_final['FWHM']
                    chain_loc=np.abs(amp_chain)*FWHM_chain*np.sqrt(np.pi)/(2.*np.sqrt(np.log(2))) 
                    merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)   
                fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'area')
                fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],'Area') 
                fixed_args['var_par_units']=np.append(fixed_args['var_par_units'],'')        
    
    
        if (prof_type in ['DI','Intr']):
    
            #FWHM of the Lorentzian and Voigt profiles
            #    - see function definition
            #    - fL = a*fG/sqrt(ln(2))
            #    - fV = 0.5436*fL+ sqrt(0.2166*fL^2+fG^2)
            # for the determination of uncertainty in chi2 mode we must take derivatives with respect to a and fG 
            #      fV = (0.5436*a*fG/sln2 )+ sqrt( 0.2166*(a*fG)^2/sln2^2  + fG^2 ) 
            # with sln2 = sqrt(ln(2))
            #      fV = fG*( (0.5436*a/sln2 )+ sqrt( 0.2166*a^2/sln2^2  + 1 ) )   
            #         = fG*( (0.5436*a)+ sqrt( 0.2166*a^2 + sln2^2 ) )/sln2  
            # we define
            #      fa = ( (0.5436*a)+ sqrt( 0.2166*a^2 + sln2^2 ) )/sln2  
            #      sfa = (0.5436 + (d[ 0.2166*a^2 + sln2^2  ]/(2*sqrt( 0.2166*a^2  + sln2^2 ))))*sa/sln2     
            #          = (0.5436+ ( 0.2166*a)/sqrt( 0.2166*a^2 + sln2^2  ))*sa/sln2         
            # with fV = fG*fa independent
            #      sfV = sqrt( (fa*sfG)^2 + (fG*sfa)^2 )    
            if (model_choice=='voigt') and (('FWHM_LOR' in fit_prop_dic['deriv_prop']) or ('FWHM_voigt' in fit_prop_dic['deriv_prop'])):
                if fit_dic['fit_mode'] in ['chi2','']:
                    sln2 = np.sqrt(np.log(2.))
                    p_final['FWHM_LOR'] = p_final['a_damp']*p_final['FWHM']/sln2
                    p_final['FWHM_voigt'] = 0.5436*p_final['FWHM_LOR']+ np.sqrt(0.2166*p_final['FWHM_LOR']**2.+p_final['FWHM']**2.)                 
                    if fit_dic['fit_mode']=='chi2': 
                        if 'FWHM' in fixed_args['var_par_list']:sfwhm_gauss= fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='FWHM')[0]] 
                        else:sfwhm_gauss=0.
                        if 'a_damp' in fixed_args['var_par_list']:sa_damp= fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='a_damp')[0]] 
                        else:sa_damp=0.
                        sfwhm_lor = p_final['FWHM_LOR']*np.sqrt( ( (sa_damp/p_final['a_damp'])**2. + (sfwhm_gauss/p_final['FWHM'])**2. ) )
                        fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sfwhm_lor],[sfwhm_lor]])) 
                        fa = (0.5436*p_final['a_damp']/sln2 )+ np.sqrt( 0.2166*p_final['a_damp']**2./sln2**2.  + 1. )
                        sfa = ( 0.5436 + (0.2166*p_final['a_damp'])/np.sqrt( 0.2166*p_final['a_damp']**2.  + sln2**2. ) )*sa_damp/sln2 
                        sig_loc = np.sqrt( (fa*sfwhm_gauss)**2. + (p_final['FWHM']*sfa)**2. ) 
                        fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]])) 
                elif fit_dic['fit_mode']=='mcmc':   
                    if 'FWHM' in fixed_args['var_par_list']:fwhm_gauss_chain = merged_chain[:,np_where1D(fixed_args['var_par_list']=='FWHM')[0]]
                    else:fwhm_gauss_chain = p_final['FWHM'] 
                    if 'a_damp' in fixed_args['var_par_list']:a_damp_chain = merged_chain[:,np_where1D(fixed_args['var_par_list']=='a_damp')[0]]
                    else:a_damp_chain = p_final['a_damp']                
                    fwhm_lor_chain = a_damp_chain*fwhm_gauss_chain/np.sqrt(np.log(2.))                
                    merged_chain=np.concatenate((merged_chain,fwhm_lor_chain[:,None]),axis=1) 
                    chain_loc=0.5436*fwhm_lor_chain+ np.sqrt(0.2166*fwhm_lor_chain**2.+fwhm_gauss_chain**2.) 
                    merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)          
                fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],['FWHM_LOR','FWHM_voigt'])
                fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],['a$_\mathrm{damp}$','FWHM$_\mathrm{Voigt}$']) 
                fixed_args['var_par_units']=np.append(fixed_args['var_par_units'],['','km/s'])               
                    
    
            #True properties of the global line
            #    - mesured on the high-resolution model, after instrumental convolution if requested
            if (model_choice in ['dgauss','custom']):
                if any([par_name in fit_prop_dic['deriv_prop'] for par_name in ['true_ctrst','true_FWHM','true_amp']]):
                    if fit_dic['fit_mode'] in ['chi2','']:
                        p_final['true_ctrst'],p_final['true_FWHM'],p_final['true_amp'] = cust_mod_true_prop(p_final,output_prop_dic['cen_bins_HR'],fixed_args)  
                        if fit_dic['fit_mode']=='chi2': 
                            sig_loc=np.nan
                            fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]])) 
                            fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]])) 
                            fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]])) 
                    elif fit_dic['fit_mode']=='mcmc':           
                        p_final_loc=deepcopy(p_final)
                        fixed_args['cen_bins_HR'] = output_prop_dic['cen_bins_HR']
                        if fixed_args['nthreads']>1:chain_loc=para_cust_mod_true_prop(proc_cust_mod_true_prop,fixed_args['nthreads'],fit_dic['nsteps_final_merged'],[merged_chain],(fixed_args,p_final_loc,))                           
                        else:chain_loc=proc_cust_mod_true_prop(merged_chain,fixed_args,p_final_loc)       
                        merged_chain=np.concatenate((merged_chain,chain_loc.T),axis=1)  
                    fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],['true_ctrst','true_FWHM','true_amp',])
                    fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],['C$_\mathrm{true}$','FWHM$_\mathrm{true}$','A$_\mathrm{true}$'])
                    fixed_args['var_par_units']=np.append(fixed_args['var_par_units'],['','km/s',''])
        
                #Stellar rotational velocity
                if (model_choice=='custom') and ('vsini' in fit_prop_dic['deriv_prop']):
                    if ('veq' in fixed_args['var_par_list']):iveq=np_where1D(fixed_args['var_par_list']=='veq')[0] 
                    if ('cos_istar' in fixed_args['var_par_list']):iistar = np_where1D(fixed_args['var_par_list']=='cos_istar')[0] 
                    if fit_dic['fit_mode']=='chi2':             
                        #    - vsini = veq*sin(i)
                        #      dvsini = vsini*sqrt( (dveq/veq)^2 + (dsini/sini)^2 )
                        #      d[sini] = cos(i)*di
                        cosistar = (p_final['cos_istar']-(1.)) % 2 - 1.
                        sin_istar = np.sqrt(1.-cosistar**2.)
                        p_final['vsini'] = p_final['veq']*sin_istar
                        if ('veq' in fixed_args['var_par_list']):sig_veq = fit_dic['sig_parfinal_err']['1s'][0,iveq]
                        else:sig_veq=0.
                        if ('cos_istar' in fixed_args['var_par_list']):sig_cos_istar = fit_dic['sig_parfinal_err']['1s'][0,iistar]
                        else:sig_cos_istar=0.
                        if (sig_veq>0.) or (sig_cos_istar>0.):
                            dsini = p_final['cos_istar']*sig_cos_istar/ sin_istar
                            sig_loc = p_final['vsini']*np.sqrt( (sig_veq/p_final['veq'])**2. + (dsini/sin_istar)**2. )  
                        else:sig_loc=np.nan
                        fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))     
                        merged_chain = None
                    elif fit_dic['fit_mode']=='mcmc':                  
                        if ('veq' in fixed_args['var_par_list']):veq_chain=merged_chain[:,iveq]  
                        else:veq_chain=deepcopy(merged_chain[:,iveq])  
                        if ('cos_istar' in fixed_args['var_par_list']):cosistar_chain=merged_chain[:,iistar]  
                        else:cosistar_chain=p_final['cos_istar']
                        cosistar_chain = (cosistar_chain-(1.)) % 2 - 1.
                        chain_loc = veq_chain*np.sqrt(1.-cosistar_chain*cosistar_chain)
                        merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)                 
                    fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'vsini')            
                    fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],'v$_\mathrm{eq}$sin i$_{*}$ (km/s)')
                    fixed_args['var_par_units']=np.append(fixed_args['var_par_units'],'km/s')     
        
                    #Convert cos(istar[rad]) into istar[deg]
                    conv_cosistar('conv',fixed_args,fit_dic,p_final,merged_chain)
                
                
                if (model_choice=='dgauss'):
            
                    #Amplitude and contrast from continuum
                    #   C = 1 - ( CCF minimum / mean continuum flux)  
                    # see in function for the CCF minimum 
                    #   C = 1 - ( (cont + invert_amp + amp_lobe) / cont)    
                    #   C = 1 - ( (cont + invert_amp -invert_amp*amp_l2c) / cont) 
                    #   C = invert_amp*( amp_l2c-1) / cont 
                    #   C = cont_amp / cont
                    if any([par_name in fit_prop_dic['deriv_prop'] for par_name in ['cont_amp','amp']]):
                        if fit_dic['fit_mode']=='chi2':  
                            
                            #Core component amplitude    
                            #    - Ac = C*cont/( amp_l2c - 1 )
                            #      dAc = Ac*sqrt( (d(C)/C)^2 + (d(cont)/cont)^2 + (d(amp_l2c)/(amp_l2c - 1))^2 )    
                            p_final['amp'] = p_final['ctrst']*p_final['cont']/( p_final['amp_l2c'] - 1. )
                            if 'ctrst' in fixed_args['var_par_list']:err_ctrst=fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='ctrst')[0]]  
                            else:err_ctrst=0.
                            if 'cont' in fixed_args['var_par_list']:err_cont=fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='cont')[0]]  
                            else:err_cont=0.
                            if 'amp_l2c' in fixed_args['var_par_list']:err_amp_l2c=fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='amp_l2c')[0]]  
                            else:err_amp_l2c=0.            
                            sig_Ac = np.abs(p_final['amp'])*np.sqrt( (err_ctrst/p_final['ctrst'])**2. + (err_cont/p_final['cont'])**2. + (err_amp_l2c/(p_final['amp_l2c'] - 1.))**2. )
                            fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_Ac],[sig_Ac]])) 
                
                            #Amplitude from continuum 
                            #   - Acont = Ac*( amp_l2c-1) 
                            #     d[Acont] = sqrt( (d(Ac)*(amp_l2c-1))^2 + (d(amp_l2c)*invert_amp)^2 ) 
                            p_final['cont_amp'] = -fixed_args['amp_sign']*p_final['amp']*(p_final['amp_l2c']-1.) 
                            sig_cont_amp=np.sqrt( (sig_Ac*(p_final['amp_l2c']-1.))**2. + (err_amp_l2c*p_final['amp'])**2. )
                            fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_cont_amp],[sig_cont_amp]]))  
                
                        elif fit_dic['fit_mode']=='mcmc': 
                            if 'ctrst' in fixed_args['var_par_list']:ctrst_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='ctrst')[0]] 
                            else:ctrst_chain=p_final['ctrst']            
                            if 'cont' in fixed_args['var_par_list']:cont_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='cont')[0]]
                            else:cont_chain=p_final['cont']            
                            if 'amp_l2c' in fixed_args['var_par_list']:amp_l2c_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='amp_l2c')[0]]
                            else:amp_l2c_chain=p_final['amp_l2c']         
                            amp_chain = ctrst_chain*cont_chain/( amp_l2c_chain - 1. )
                            cont_amp_chain=-fixed_args['amp_sign']*amp_chain*(amp_l2c_chain-1.)   
                            merged_chain=np.concatenate((merged_chain,cont_amp_chain[:,None],amp_chain[:,None]),axis=1)   
                    
                        fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],['cont_amp','amp'])
                        fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],['Acont','Amp'])
                        fixed_args['var_par_units']=np.append(fixed_args['var_par_units'],['',''])
        
        
                    #Lobe properties
                    if ('RV_lobe' in fit_prop_dic['deriv_prop']):                      
                        irv = np_where1D(fixed_args['var_par_list']=='rv')[0] 
                        if fit_dic['fit_mode']=='chi2': 
                            if 'rv_l2c' in fixed_args['var_par_list']:err_rv_l2c=fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='rv_l2c')[0]]  
                            else:err_rv_l2c=0.
                            p_final['RV_lobe']=p_final['rv'] + p_final['rv_l2c']
                            sig_loc=np.sqrt( fit_dic['sig_parfinal_err']['1s'][0,irv]**2.  + err_rv_l2c**2.  )     
                            fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))         
                        elif fit_dic['fit_mode']=='mcmc': 
                            if 'rv_l2c' in fixed_args['var_par_list']:rv_l2c_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='rv_l2c')[0]]
                            else:rv_l2c_chain=p_final['rv_l2c']
                            chain_loc = merged_chain[:,irv] + rv_l2c_chain
                            merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)
                        fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],['RV_lobe'])
                        fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],['rv$_\mathrm{lobe}$ (km s$^{-1}$)'])  
                        fixed_args['var_par_units']=np.append(fixed_args['var_par_units'],['km/s'])                   
                    
                    if ('amp_lobe' in fit_prop_dic['deriv_prop']):            
                        if fit_dic['fit_mode']=='chi2': 
                            if 'amp_l2c' in fixed_args['var_par_list']:err_amp_l2c=fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='amp_l2c')[0]]  
                            else:err_amp_l2c=0.
                            p_final['amp_lobe']=p_final['amp']*p_final['amp_l2c']
                            sig_loc=np.sqrt((sig_Ac*p_final['amp_l2c'])**2.+(err_amp_l2c*p_final['amp'])**2.)
                            fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]])) 
                        elif fit_dic['fit_mode']=='mcmc': 
                            if 'amp_l2c' in fixed_args['var_par_list']:amp_l2c_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='amp_l2c')[0]]
                            else:amp_l2c_chain=p_final['amp_l2c']
                            chain_loc=merged_chain[:,np_where1D(fixed_args['var_par_list']=='amp')[0]]*amp_l2c_chain
                            merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)
                        fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],['amp_lobe'])
                        fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],['A$_\mathrm{lobe}$'])   
                        fixed_args['var_par_units']=np.append(fixed_args['var_par_units'],[''])                  
                    
                    if ('FWHM_lobe' in fit_prop_dic['deriv_prop']):                     
                        if fit_dic['fit_mode']=='chi2': 
                            if 'FWHM_l2c' in fixed_args['var_par_list']:err_FWHM_l2c=fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='FWHM_l2c')[0]]  
                            else:err_FWHM_l2c=0.
                            if 'FWHM' in fixed_args['var_par_list']:err_FWHM=fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='FWHM')[0]]  
                            else:err_FWHM=0.     
                            p_final['FWHM_lobe']=p_final['FWHM']*p_final['FWHM_l2c']
                            sig_loc=np.sqrt((err_FWHM*p_final['FWHM_l2c'])**2.+(err_FWHM_l2c*p_final['FWHM'])**2.) 
                            fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))   
                        elif fit_dic['fit_mode']=='mcmc': 
                            if 'FWHM_l2c' in fixed_args['var_par_list']:FWHM_l2c_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='FWHM_l2c')[0]]
                            else:FWHM_l2c_chain=p_final['FWHM_l2c']
                            if 'FWHM' in fixed_args['var_par_list']:FWHM_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='FWHM')[0]]
                            else:FWHM_chain=p_final['FWHM']
                            chain_loc=FWHM_chain*FWHM_l2c_chain
                            merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)
                        fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],['FWHM_lobe'])
                        fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],['FWHM$_\mathrm{lobe}$ (km s$^{-1}$)'])
                        fixed_args['var_par_units']=np.append(fixed_args['var_par_units'],['km/s'])
            

         
            


    ########################################################################################################   
    #Processing and storing best-fit values and confidence interval for the final parameters  
    #    - derived properties (not used in the model definitions) must have been added above before postMCMCwrapper_2() is run 
    ########################################################################################################       
    if fit_dic['fit_mode']=='mcmc':
        
        #Add new properties to relevant dictionaries
        for param_loc in fixed_args['var_par_list']:
            if ('HDI_dbins' in fit_prop_dic) and (param_loc in fit_prop_dic['HDI_dbins']) and (inst in fit_prop_dic['HDI_dbins'][param_loc]) and (vis in fit_prop_dic['HDI_dbins'][param_loc][inst]):
                fit_dic['HDI_dbins'][param_loc]=fit_prop_dic['HDI_dbins'][param_loc][inst][vis]
            elif ('HDI_bwf' in fit_prop_dic) and (param_loc in fit_prop_dic['HDI_bwf']) and (inst in fit_prop_dic['HDI_bwf'][param_loc]) and (vis in fit_prop_dic['HDI_bwf'][param_loc][inst]):
                fit_dic['HDI_bwf'][param_loc]=fit_prop_dic['HDI_bwf'][param_loc][inst][vis]        
       
        #Process
        p_final=postMCMCwrapper_2(fit_dic,fixed_args,merged_chain)
       
    #Storing best-fit values and selected uncertainties in output dictionary
    for key in ['cont','rv','FWHM','amp','ctrst','area','FWHM_LOR','a_damp','FWHM_voigt','rv_l2c','amp_l2c','FWHM_l2c','cont_amp','RV_lobe','amp_lobe','FWHM_lobe','true_amp','true_ctrst','true_FWHM','veq','cos_istar','vsini','ctrst_ord0__IS__VS_','FWHM_ord0__IS__VS_']:
        
        if key in p_final:
            output_prop_dic[key]  = p_final[key]  
            
            #Variable parameter
            if (key in fixed_args['var_par_list']):
                ipar = np_where1D(fixed_args['var_par_list']==key)[0]
                
                #Errors set to percentiles-based uncertainties
                if (fit_dic['fit_mode']=='chi2') or ((fit_dic['fit_mode']=='mcmc') & (fit_prop_dic['out_err_mode']=='quant')):
                    output_prop_dic['err_'+key]= fit_dic['sig_parfinal_err']['1s'][:,ipar]  
                
                #Errors set to HDI-based uncertainties  
                elif (fit_dic['fit_mode']=='mcmc') & (fit_prop_dic['out_err_mode']=='HDI'):
                    output_prop_dic['err_'+key]= np.array([p_final[key]-fit_dic['HDI_interv'][ipar][0][0],fit_dic['HDI_interv'][ipar][-1][1]-p_final[key]])           
                
            else:output_prop_dic['err_'+key]=[0.,0.]        
            
    #Save derived parameters
    save_fit_results('derived',fixed_args,fit_dic,fit_dic['fit_mode'],p_final)
  
    #Close save file
    fit_dic['file_save'].close()

    ######################################################################################################## 
    #Criterion for line detection
    ######################################################################################################## 
    if (fit_prop_dic['thresh_area'] is not None) or (fit_prop_dic['thresh_amp'] is not None): 
        if np.sum(cond_def_cont)==0.:stop('No bin in CCF continuum')
      
        #Continuum dispersion
        disp_cont = flux_loc[cond_def_cont].std()
        
        #Area and amplitude criterion
        #    - we ensure that 
        # abs(cont -  peak) > disp            
        #      and that 
        # abs(cont -  peak)*sqrt(FWHM) > disp*sqrt(pix)
        #      which can be seen as requesting that the area of a box equivalent to the CCF is larger than the area of a box equivalent to a bin with amplitude the dispersion                
        #    - see also Allart+2017
        if (model_choice in ['gauss','voigt']) and ('amp' in output_prop_dic):
            output_prop_dic['crit_area']=np.abs(output_prop_dic['amp'])*np.sqrt(output_prop_dic['FWHM']) / (disp_cont*np.sqrt(gen_dic['pix_size_v'][inst]))            
            output_prop_dic['crit_amp']=np.abs(output_prop_dic['amp'])/disp_cont 
        if (model_choice in ['dgauss','custom']) and ('true_amp' in output_prop_dic):
            output_prop_dic['crit_area']=np.abs(output_prop_dic['true_amp'])*np.sqrt(output_prop_dic['true_FWHM']) / (disp_cont*np.sqrt(gen_dic['pix_size_v'][inst]))
            output_prop_dic['crit_amp']=np.abs(output_prop_dic['true_amp'])/disp_cont

        #---------------------------------------------------------
    
        #Force detection
        if (idx_force_det is not None):
            output_prop_dic['forced_det']=True 
            output_prop_dic['detected']=idx_force_det
        
        #Assess detection
        #    - Amplitude >= threshold x continuum dispersion
        #      this criterion includes the check that amplitude is larger than 0
        #    - Area criterion 
        else:
            output_prop_dic['forced_det']=False
            if ('crit_amp' in output_prop_dic) and ('crit_area' in output_prop_dic):  
                output_prop_dic['detected']= (output_prop_dic['crit_amp']>fit_prop_dic['thresh_amp']) & (output_prop_dic['crit_area']>fit_prop_dic['thresh_area'])            
            else:output_prop_dic['detected'] = False

    else:
        output_prop_dic['forced_det']=False
        output_prop_dic['detected'] = ''

    ######################################################################################################## 
    #Direct measurements
    ######################################################################################################## 

    #Calculation of bissector
    if ('biss' in fit_prop_dic['meas_prop']):
        biss_prop = fit_prop_dic['meas_prop']['biss']

        #Spectrum selection   
        if biss_prop['source']=='obs':
            rv_biss = deepcopy(cen_bins)
            flux_biss = deepcopy(flux_loc)
        elif biss_prop['source']=='mod':
            rv_biss = deepcopy(output_prop_dic['cen_bins_HR'])
            flux_biss = deepcopy(output_prop_dic['flux_HR'])            

        #Normalize flux with best-fit continuum        
        flux_biss /= p_final['cont']
        
        #Calculating bissector
        output_prop_dic['RV_biss'],output_prop_dic['F_biss'],output_prop_dic['RV_biss_span'],output_prop_dic['F_biss_span'],output_prop_dic['biss_span']=calc_biss(flux_biss,rv_biss,output_prop_dic['rv'],fit_prop_dic['biss_range_frame'][:,iexp],biss_prop['dF'],gen_dic['resamp_mode'],biss_prop['Cspan'])
        output_prop_dic['F_biss']*=p_final['cont']

    #Calculation of equivalent width
    if ('EW' in fit_prop_dic['meas_prop']):

        #Normalize flux with best-fit continuum      
        flux_norm,cov_norm = bind.mul_array(flux_loc,cov_loc,1./np.repeat(p_final['cont'],len(flux_loc)))

        #Equivalent width 
        #    - defined as int( x, dx*(1 - Fnorm(x))  ), and calculated here in RV space as Delta_RV - int(x , Fnorm(x)*dw)
        bd_int = fit_prop_dic['EW_range_frame'][:,iexp]
        idx_int_sign = np_where1D((edge_bins[0:-1]>=bd_int[0]) & (edge_bins[1::]<=bd_int[1]))
        if len(idx_int_sign)==0:stop('No pixels in "EW_range"')
        width_range = bd_int[1]-bd_int[0]
        int_flux,cov_int_flux = bind.resampling(bd_int,edge_bins, flux_norm, cov =  cov_norm, kind=gen_dic['resamp_mode']) 
        output_prop_dic['EW'] = width_range - int_flux[0]
        output_prop_dic['err_EW'] = np.repeat(np.sqrt(cov_int_flux[0,0]),2)

    #Calculation of mean integrated signal
    if prof_type=='Atm': 
        if ('int_sign' in fit_prop_dic['meas_prop']):
            
            #Signal in each requested range
            int_sign = []
            var_int_sign = []
            for bd_int in fit_prop_dic['int_sign_range_frame']:
                idx_int_sign = np_where1D((edge_bins[0:-1]>=bd_int[0,iexp]) & (edge_bins[1::]<=bd_int[1,iexp]))
                if len(idx_int_sign)==0:stop('No pixels in "int_sign_range"')
                width_range = bd_int[1,iexp]-bd_int[0,iexp]
                int_sign_loc,cov_int_sign_loc = bind.resampling(bd_int,edge_bins, flux_loc, cov =  cov_loc, kind=gen_dic['resamp_mode']) 
                int_sign+=[int_sign_loc[0]/width_range]
                var_int_sign+=[cov_int_sign_loc[0,0]/width_range**2.]
                
            #Global signal
            #    - calculated as the mean of the signal over the requested ranges
            output_prop_dic['int_sign'] = np.mean(int_sign)
            output_prop_dic['err_int_sign'] = np.sqrt(np.sum(var_int_sign))
            output_prop_dic['R_sign'] = fit_dic[iexp]['int_sign']/fit_dic[iexp]['err_int_sign']
            output_prop_dic['err_int_sign'] = np.repeat(output_prop_dic['err_int_sign'],2.)

    return output_prop_dic
    





##################################################################################################    
#%%% Prior functions
################################################################################################## 

def prior_sini_geom(p_step_loc,fixed_args,prior_func_prop):
    r"""**Prior: sin(istar)**

    Calculates :math:`\log(p)` for stellar inclination :math:`i_\star`.
    
    Assumes isotropic distribution of stellar orientations (equivalent to draw a uniform distribution on :math:`\cos(i_\star)`) that favor
    stellar inclination closer to 90:math:`^\circ`
    
    .. math::  
       &p(x) = \sin(i_\star(x))/2  \\
       &\ln(p(x)) = \ln(0.5 \sin(i_\star(x)))
      
    Where `p` is normalized by the integral of :math:`\sin(i_\star)` over the definition space of :math:`i_\star` (0:180:math:`^\circ`), which is 2. 

    Args:
        TBD

    Returns:
        TBD
        
    """    
    sin_istar=np.sqrt(1.-p_step_loc['cos_istar']**2.)  
    return -np.log(0.5*sin_istar) 


def prior_cosi(p_step_loc,fixed_args,prior_func_prop): 
    r"""**Prior: cos(ip)**

    Calculates :math:`\log(p)` for orbital inclination :math:`i_p`.
    
    Prior imposes that :math:`0<i_p<90`^\circ`

    Args:
        TBD

    Returns:
        TBD
        
    """ 
    ln_p = 0.
    for pl_loc in fixed_args['inclin_rad_pl']:
        if np.cos(p_step_loc['inclin_rad__pl'+pl_loc]) < 0:
            ln_p = -np.inf
            break
    return ln_p

def prior_sini(p_step_loc,fixed_args,prior_func_prop): 
    r"""**Prior: sin(ip)**

    Calculates :math:`\log(p)` for orbital inclination :math:`i_p`.
    
    Imposes Gaussian prior on :math:`\sin(i_p)`.

    Args:
        TBD

    Returns:
        TBD
        
    """ 
    return -0.5*( (     np.sin(p_step_loc['inclin_rad__pl'+prior_func_prop['pl']]) - prior_func_prop['val'])/prior_func_prop['sig'])**2.                


def prior_b(p_step_loc,fixed_args,prior_func_prop): 
    r"""**Prior: b**

    Calculates :math:`\log(p)` for impact parameter.
    
    Prior imposes that :math:`b<1`.  
    
    Used when  orbital inclination :math:`i_p` and/or semi-major axis :math:`a/R_\star` are varied.
    If only :math:`i_p` or :math:`a/R_\star` is free to vary, the other property should still be defined as a constant parameter for this prior to be used.

    Args:
        TBD

    Returns:
        TBD
        
    """     
    ln_p = 0.
    for pl_loc in fixed_args['b_pl']:
        if np.abs(p_step_loc['aRs__pl'+pl_loc]*np.cos(p_step_loc['inclin_rad__pl'+pl_loc])) > 1:
            ln_p = -np.inf
            break
    return ln_p    


def prior_DR(p_step_loc,fixed_args,prior_func_prop):
    r"""**Prior: DR**

    Calculates :math:`\log(p)` for differential rotation coefficients :math:`\alpha_\mathrm{rot}` and :math:`\beta_\mathrm{rot}`.
    
    If both :math:`\alpha_\mathrm{rot}` and :math:`\beta_\mathrm{rot}` are used, their sum cannot be larger than 1 if the star rotates in the same direction at all latitudes.

    Args:
        TBD

    Returns:
        TBD
        
    """    
    if (p_step_loc['alpha_rot']+p_step_loc['beta_rot']>1.):return -np.inf	
    else: return 0.


def prior_vsini(p_step_loc,fixed_args,prior_func_prop):
    r"""**Prior: veq sin(istar)**

    Calculates :math:`\log(p)` for sky-projected stellar rotational velocity :math:`v_\mathrm{eq} \sin(i_\star)`.
    
    Imposes Gaussian prior on :math:`v_\mathrm{eq} \sin(i_\star)`.
    Relevant when :math:`v_\mathrm{eq}` and :math:`i_\star` are fitted independently, otherwise set a prior on :math:`v_\mathrm{eq}` alone.

    Args:
        TBD

    Returns:
        TBD
        
    """     
    vsini = p_step_loc['veq']*np.sqrt(1.-p_step_loc['cos_istar']**2.)
    return -0.5*( (vsini - prior_func_prop['val'])/prior_func_prop['sig'])**2.                


def prior_vsini_deriv(p_step_loc,fixed_args,prior_func_prop):
    r"""**Prior: veq sin(istar)**

    Calculates :math:`\log(p)` for sky-projected stellar rotational velocity :math:`v_\mathrm{eq} \sin(i_\star)`.
    
    Imposes Gaussian prior on :math:`v_\mathrm{eq} \sin(i_\star)`.
    Relevant when :math:`P_\mathrm{eq}` and :math:`i_\star` are fitted independently, otherwise set a prior on :math:`v_\mathrm{eq}` alone.

    Args:
        TBD

    Returns:
        TBD
        
    """
    vsini = 2.*np.pi*p_step_loc['Rstar']*Rsun*np.sqrt(1.-p_step_loc['cos_istar']**2.)/(p_step_loc['Peq']*24.*3600.)
    return -0.5*( (vsini - prior_func_prop['val'])/prior_func_prop['sig'])**2.    



def prior_contrast(p_step_loc,args_in,prior_func_prop):
    r"""**Prior: intrinsic contrast**

    Calculates :math:`\log(p)` for the contrast :math:`C_\mathrm{Intr}` of intrinsic stellar lines.
    
    Prior imposes that :math:`0<C_\mathrm{Intr}<1`. 
    The contrast of intrinsic (and a fortiori measured) stellar lines should be physical.
    See details on the calculation of line contrast in `joined_intr_prof()` (for now assumes a single planet per visit transits).

    Args:
        TBD

    Returns:
        TBD
        
    """
    ln_p_loc = 0.
    args = deepcopy(args_in)
    args['grid_dic']['precision']='low'
    for inst in args['inst_list']:
        args['inst']=inst
        for vis in args['inst_vis_list'][inst]:   
            args['vis']=vis
            pl_vis = args['transit_pl'][inst][vis][0]
            system_param_loc,coord_pl,param_val = up_plocc_prop(inst,vis,args,p_step_loc,[pl_vis],args['ph_fit'][inst][vis],args['coord_fit'][inst][vis])
            surf_prop_dic,spot_prop_dic,surf_prop_dic_common = sub_calc_plocc_spot_prop([args['chrom_mode']],args,[args['coord_line']],[pl_vis],system_param_loc,args['grid_dic'],args['system_prop'],param_val,args['coord_fit'][inst][vis],range(args['nexp_fit_all'][inst][vis]),False)
            ctrst_vis = surf_prop_dic[pl_vis]['ctrst'][0]       
            break_cond = (ctrst_vis<0.) | (ctrst_vis>1.)
            if True in break_cond:
                ln_p_loc+= -np.inf	
                break 
    return ln_p_loc


def prior_FWHM_vsini(p_step_loc,args,prior_func_prop):
    r"""**Prior: intrinsic FWHM**

    Calculates :math:`\log(p)` for the \mathrm{FWHM} :math:`\mathrm{FWHM}_\mathrm{Intr}` of intrinsic stellar lines.
    
    Prior imposes that :math:`\mathrm{FWHM}_\mathrm{Intr}^2 + \mathrm{FWHM}_\mathrm{Inst}^2 + v_\mathrm{eq} \sin(i_\star)^2<\mathrm{FWHM}_\mathrm{DI}`. 
    We assume that the intrinsic line cannot be larger than the disk-integrated one, after rotational broadening.
    This sets an upper limit on :math:`\mathrm{FWHM}_\mathrm{Intr}`, considering there can be other broadening contributions 
    See details on the calculation of line contrast in `joined_intr_prof()` (for now assumes intrinsic line must have constant width).

    Args:
        TBD

    Returns:
        TBD
        
    """    
    ln_p_loc = 0.
    vsini = p_step_loc['veq']*np.sqrt(1.-p_step_loc['cos_istar']**2.)
    for inst in args['inst_list']:
        for vis in args['inst_vis_list'][inst]:  

            #Width of disk-integrated profile, rotationally broadened, after instrumental convolution
            FWHM_intr = p_step_loc[args['name_prop2input']['FWHM_ord0__IS'+inst+'_VS'+vis]]
            FWHM_DI_mod2 = FWHM_intr**2. + args['FWHM_inst'][inst]**2. + vsini**2.
            
            #Width must be smaller than width of measured disk-integrated profile
            if FWHM_DI_mod2>prior_func_prop['FWHM_DI']**2.:
                ln_p_loc+= -np.inf	
                break 

    return ln_p_loc


prior_functions={
    'sinistar_geom':prior_sini_geom,
    'cosi':prior_cosi,
    'sini':prior_sini,
    'b':prior_b,
    'DR':prior_DR,
    'vsini':prior_vsini,
    'vsini_deriv':prior_vsini_deriv,
    'contrast':prior_contrast,
    'FWHM_vsini':prior_FWHM_vsini
}
def global_ln_prior_func(p_step_loc,fixed_args):
    r"""**Global prior function.**

    Calculates :math:`\log(p)` cumulated over the chosen priors. 
    
    See `ln_prior_func()` for details on prior definition.

    Args:
        TBD

    Returns:
        TBD
        
    """ 
    ln_p_loc = 0.
    for key in ['sinistar_geom','cosi','sini','b','DR','vsini','vsini_deriv','contrast','FWHM_vsini']:
        if key in fixed_args['prior_func']:ln_p_loc+=prior_functions[key](p_step_loc,fixed_args,fixed_args['prior_func'][key])
    return ln_p_loc










##################################################################################################    
#%%% Post-processing 
################################################################################################## 

def com_joint_postproc(p_final,fixed_args,fit_dic,merged_chain,fit_prop_dic,gen_dic):
    r"""**Wrap-up for post-processing of time-series fits.**

    Performs joint fits to time-series.
    
    Args:
        TBD

    Returns:
        TBD
        
    """
    
    #Call to specific post-processing
    #    - Fit from chi2
    # + combination/modification of best-fit results to add new parameters
    #    - Fit from mcmc
    # + combination/modification of MCMC chains to add new parameters
    # + new parameters are not used for model
    # + we calculate median and errors after chain are added   
    if (fit_dic['fit_mode'] in ['chi2','mcmc']) and ('deriv_prop' in fit_prop_dic) and len(fit_prop_dic['deriv_prop'])>0:
        print('       Post-processing')
        deriv_prop = fit_prop_dic['deriv_prop']
    
        #Fold cos(istar) within -1 : 1
        if 'cosistar_fold' in deriv_prop:
            print('        + Folding cos(istar)')
            if fit_dic['fit_mode']=='chi2': 
                cosistar = (p_final['cos_istar']-(1.)) % 2 - 1.
                p_final['cos_istar'] = cosistar
            elif fit_dic['fit_mode']=='mcmc':   
                iistar = np_where1D(fixed_args['var_par_list']=='cos_istar')
                if ('cos_istar' in fixed_args['var_par_list']):cosistar_chain=merged_chain[:,iistar]  
                else:cosistar_chain=p_final['cos_istar']     
                merged_chain[:,iistar]= (cosistar_chain-(1.)) % 2 - 1.         
    
        #Convert Rstar and Peq into veq
        #    - veq = (2*pi*Rstar/Peq)
        if 'veq_from_Peq_Rstar' in deriv_prop:
            print('        + Deriving veq from Peq and Rstar ')
            
            if fit_dic['fit_mode']=='chi2': 
                p_final['veq']=(2.*np.pi*p_final['Peq']*Rsun)/(p_final['Peq']*3600.*24.)
                sig_loc=np.nan
                fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))
            elif fit_dic['fit_mode']=='mcmc':   
                n_chain = len(merged_chain[:,0])
                if 'Rstar' in fixed_args['var_par_list']:
                    print('           Using fitted Rstar')
                    Rstar_chain = np.squeeze(merged_chain[:,np_where1D(fixed_args['var_par_list']=='Rstar')])
                else:
                    print('           Using external Rstar')
                    if gen_dic['star_name']=='HD189733':
                        Rstar_med = 0.784  
                        Rstar_err=0.007   
                    Rstar_chain = np.random.normal(Rstar_med, Rstar_err, n_chain)
                iPeq = np_where1D(fixed_args['var_par_list']=='Peq')
                chain_loc=(2.*np.pi*Rstar_chain*Rsun)/(np.squeeze(merged_chain[:,iPeq])*3600.*24.)
                merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)  
            fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'veq')
            fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],'v$_{eq}$ (km/s)')
            fixed_args['var_par_units']=np.append(fixed_args['var_par_units'],'km/s')
    
    
        #Convert veq into veq*sin(istar) to be comparable with solid-body values
        #    - must be done before modification of istar chain
        if ('vsini' in deriv_prop) and ((fixed_args['rout_mode'] in ['IntrProf','ResProf']) or ((fixed_args['rout_mode']=='IntrProp') and (fixed_args['prop_fit']=='rv'))):
            print('        + Converting veq to vsini')
            if 'veq' in fixed_args['var_par_list']:iveq=np_where1D(fixed_args['var_par_list']=='veq')
            else:stop('Activate veq_from_Peq_Rstar')
            iistar = np_where1D(fixed_args['var_par_list']=='cos_istar')
            if fit_dic['fit_mode']=='chi2':             
                #    - vsini = veq*sin(i)
                #      dvsini = vsini*sqrt( (dveq/veq)^2 + (dsini/sini)^2 )
                #      d[sini] = cos(i)*di
                cosistar = (p_final['cos_istar']-(1.)) % 2 - 1.
                sin_istar = np.sqrt(1.-cosistar**2.)
                p_final['vsini'] = p_final['veq']*sin_istar
                if ('cos_istar' in fixed_args['var_par_list']):
                    dsini = p_final['cos_istar']*fit_dic['sig_parfinal_err']['1s'][0,iistar]/ sin_istar
                    sig_temp = p_final['vsini']*np.sqrt( (fit_dic['sig_parfinal_err']['1s'][0,iveq]/p_final['veq'])**2. + (dsini/sin_istar)**2. )  
                else:
                    sig_temp = fit_dic['sig_parfinal_err']['1s'][:,iveq]*sin_istar            
                fit_dic['sig_parfinal_err']['1s'][:,iveq] = sig_temp
            elif fit_dic['fit_mode']=='mcmc':                  
                if ('cos_istar' in fixed_args['var_par_list']):
                    cosistar_chain=merged_chain[:,iistar]  
                else:
                    #Stellar inclination is fixed
                    cosistar_chain=p_final['cos_istar']
                    
                #Folding cos(istar) within -1 : 1
                cosistar_chain = (cosistar_chain-(1.)) % 2 - 1.
                    
                veq_chain=deepcopy(merged_chain[:,iveq])            
                merged_chain[:,iveq]=veq_chain*np.sqrt(1.-cosistar_chain*cosistar_chain)
            fixed_args['var_par_list'][iveq]='vsini'            
            fixed_args['var_par_names'][iveq]=model_par_names('vsini') 
            fixed_args['var_par_units'][iveq]=model_par_units('vsini')            
    
            
        #-------------------------------------------------            

        #Replace cos(istar[rad]) by istar[deg]
        if ('istar_deg_conv' in deriv_prop) or ('istar_deg_add' in deriv_prop):
            print('        + Converting cos(istar) to istar')
            conv_cosistar(deriv_prop,fixed_args,fit_dic,p_final,merged_chain)

        #-------------------------------------------------            

        #Folding istar[deg] around 90
        #    - only use if all other parameters are degenerate with istar
        #    - by default we fold over 0-90
        if ('fold_istar' in deriv_prop):        
            print('        + Folding istar')
            iistar = np_where1D(fixed_args['var_par_list']=='istar_deg')
            istar_temp=np.squeeze(merged_chain[:,iistar])
            w_gt_90=(istar_temp > 90.)
            if True in w_gt_90:merged_chain[w_gt_90,iistar]=180.-istar_temp[w_gt_90]

        #-------------------------------------------------
        #Add istar using the value derived from vsini and independent measurement of Peq and Rstar
        #    - prefer the use of Peq and Rstar as fit parameters
        #    - vsini = veq*sin(istar) = (2*pi*Rstar/Peq)*sin(istar)
        #      istar = np.arcsin( vsini*peq/(2*pi*Rstar) )
        if ('istar_Peq' in deriv_prop) or ('istar_Peq_vsini' in deriv_prop):
            if ('cos_istar' in fixed_args['var_par_list']):stop('    istar has been fitted')
            print('        + Deriving istar from vsini and Peq')
            
            #Nominal values and 1s errors for Rstar (Rsun) and Peq (d)
            # Rstar_med = 0.850       #HD3167
            # Rstar_err = 0.020
            # Peq_med = 23.52         #HD3167
            # Peq_err = 2.87
            # Rstar_med = 0.438       #GJ436
            # Rstar_err = 0.013
            # Peq_med = 44.09       
            # Peq_err = 0.08
            if gen_dic['star_name']=='HAT_P3':
                Rstar_med = 0.85       
                Rstar_err = 0.021
                Peq_med = 19.9 
                Peq_elow = 1.5
                Peq_ehigh = 1.5
            elif gen_dic['star_name']=='HAT_P11':
                Rstar_med = 0.74       
                Rstar_err = 0.01
                Peq_med = 30.5       
                Peq_elow = 3.2
                Peq_ehigh = 4.1
            # Rstar_med = 0.901       #Kepler-63
            # Rstar_err = 0.0245
            # Peq_med = 5.401
            # Peq_err =   0.014  
            elif gen_dic['star_name']=='WASP107':            
                Rstar_med = 0.67       
                Rstar_err = 0.02
                Peq_med = 17.1
                Peq_elow = 1.
                Peq_ehigh = 1.  
            # Rstar_med = 1.273       #HIP 41378
            # Rstar_err = 0.015
            # Peq_med = 6.4
            # Peq_err = 0.8  
            elif gen_dic['star_name']=='Kepler25':
                Rstar_med = 1.316       
                Rstar_err = 0.016
                Peq_med = 23.147         
                Peq_elow = 0.039
                Peq_ehigh = 0.039
            elif gen_dic['star_name']=='WASP47':
                Rstar_med = 1.137       
                Rstar_err = 0.013
                Peq_med = 39.4         
                Peq_elow = 4.5
                Peq_ehigh = 2.2
            elif gen_dic['star_name']=='WASP166':
                Rstar_med = 1.22       
                Rstar_err = 0.06
                Peq_med = 12.3      
                Peq_elow = 1.9
                Peq_ehigh = 1.9                
            elif gen_dic['star_name']=='WASP69':
                Rstar_med = 0.813  
                Rstar_err=0.028 
                Peq_med = 23.07 
                Peq_elow =  0.16 
                Peq_ehigh = 0.16 
                 
            #Conversions
            Rstar_med*=Rsun
            Rstar_err*=Rsun
            Peq_med*=24.*3600.
            Peq_elow*=24.*3600.
            Peq_ehigh*=24.*3600.
            
            if fit_dic['fit_mode']=='chi2':             
                p_final['istar_deg']=np.arcsin(p_final['vsini']*Peq_med/(2.*np.pi*Rstar_med))*180./np.pi
                sig_loc=np.nan
                fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))
            elif fit_dic['fit_mode']=='mcmc':   
                n_chain = len(merged_chain[:,0])
                if 'istar_Peq_vsini' in deriv_prop:
                    print('           Using external vsini')
                    if gen_dic['star_name']=='WASP47':
                        vsini_med = 1.80         
                        vsini_elow = 0.16
                        vsini_ehigh = 0.24   
                    vsini_chain = gen_hrand_chain(vsini_med,vsini_elow,vsini_ehigh,n_chain)
                elif 'istar_Peq' in deriv_prop:
                    print('           Using derived vsini')
                    if np.abs(p_final['cos_istar'])>1e-14:stop('             istar should have been fixed to 90 degrees')
                    iveq = np_where1D(fixed_args['var_par_list']=='veq')  
                    if len(iveq)==0:stop('             veq not fitted properties')
                    vsini_chain = np.squeeze(merged_chain[:,iveq])

                #Generate gaussian distribution for Rstar and Peq
                Rstar_chain = np.random.normal(Rstar_med, Rstar_err, n_chain)
                Peq_chain = gen_hrand_chain(Peq_med,Peq_elow,Peq_ehigh,n_chain)

                #Calculate sin(istar) chain
                sinistar_chain = vsini_chain*Peq_chain/(2.*np.pi*Rstar_chain)
                               
                #Replace non-physical values
                cond_good = np.abs(sinistar_chain)<=1.
                n_good = np.sum(cond_good)
                while n_good<n_chain:
                    Rstar_add = np.random.normal(Rstar_med, Rstar_err, n_chain-n_good)
                    Peq_add = gen_hrand_chain(Peq_med,Peq_elow,Peq_ehigh,n_chain-n_good)
                    if 'istar_Peq_vsini' in deriv_prop:vsini_add = gen_hrand_chain(vsini_med,vsini_elow,vsini_ehigh,n_chain-n_good)
                    elif 'istar_Peq' in deriv_prop:vsini_add = np.random.choice(vsini_chain,n_chain-n_good)
                    sinistar_chain = np.append(sinistar_chain[cond_good],vsini_add*Peq_add/(2.*np.pi*Rstar_add))
                    cond_good = np.abs(sinistar_chain)<=1.
                    n_good = np.sum(cond_good)
                
                #istar chain
                chain_loc=np.arcsin( sinistar_chain )*180./np.pi 
                
                # #pi-istar chain
                # chain_loc=180.-chain_loc   
                
                merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)  
            fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'istar_deg')
            fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],model_par_names('istar_deg'))
            fixed_args['var_par_units']=np.append(fixed_args['var_par_units'],model_par_units('istar_deg'))
    
        #-------------------------------------------------
        #Add Peq using the value derived from veq and independent measurement of Rstar
        #    - Peq = (2*pi*Rstar/veq)
        if 'Peq_veq' in deriv_prop:
            if ('cos_istar' in fixed_args['var_par_list']):stop('    istar has been fitted')
            print('        + Deriving Peq from veq ')
            
            #Nominal values and 1s errors for Rstar (Rsun) and Peq (d)
            Rstar_med = 0.850       #HD3167
            Rstar_err = 0.020
    
            #Conversions
            Rstar_med*=Rsun
            Rstar_err*=Rsun
            
            if fit_dic['fit_mode']=='chi2': 
                p_final['Peq_d']=(2.*np.pi*Rstar_med)/(p_final['veq']*3600.*24.)
                sig_loc=np.nan
                fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))
            elif fit_dic['fit_mode']=='mcmc':   
            
                #Generate gaussian distribution for Rstar
                Rstar_chain = np.random.normal(Rstar_med, Rstar_err, len(merged_chain[:,0]))
                
                #Calculate Peq chain
                iveq = np_where1D(fixed_args['var_par_list']=='veq')
                chain_loc=(2.*np.pi*Rstar_chain)/(np.squeeze(merged_chain[:,iveq])*3600.*24.)
                merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)  
            fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'Peq')
            fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],model_par_names('Peq'))
            fixed_args['var_par_units']=np.append(fixed_args['var_par_units'],model_par_units('Peq'))
    
        #-------------------------------------------------
        #Add Peq using the value derived from vsini and independent measurement of Rstar and istar
        #    - Peq = (2*pi*Rstar*sin(istar)/vsini)
        if 'Peq_vsini' in deriv_prop:
            print('        + Deriving Peq from vsini')
            
            #Nominal values and 1s errors for Rstar (Rsun) 
            if gen_dic['star_name']=='Kepler25':
                Rstar_med = 1.316       
                Rstar_err = 0.016

                
            #Generate distribution for istar
            if gen_dic['star_name']=='Kepler25':
                istar_mean = 66.7*np.pi/180.    
                istar_high = 12.1*np.pi/180.
                istar_low  = 7.4*np.pi/180.
    
            #Conversions
            Rstar_med*=Rsun
            Rstar_err*=Rsun

            if fit_dic['fit_mode']=='chi2': 
                stop('TBD')
            elif fit_dic['fit_mode']=='mcmc':  
                n_chain = len(merged_chain[:,0])
            
                #Generate gaussian distribution for Rstar
                Rstar_chain = np.random.normal(Rstar_med, Rstar_err,n_chain )
                
                #Generate distribution for sin(istar) 
                sinistar_chain = np.sin(gen_hrand_chain(istar_mean,istar_low,istar_high,n_chain))
                
                #Calculate Peq chain
                if 'vsini' not in fixed_args['var_par_list']:stop('Add vsini chain')
                ivsini = np_where1D(fixed_args['var_par_list']=='vsini')
                chain_loc=(2.*np.pi*Rstar_chain*sinistar_chain)/(np.squeeze(merged_chain[:,ivsini])*3600.*24.)
                merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)  
            fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'Peq')
            fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],model_par_names('Peq'))  
            fixed_args['var_par_units']=np.append(fixed_args['var_par_units'],model_par_units('Peq')) 
    
        #-------------------------------------------------
                
        #Add true obliquity
        #    - psi = acos(sin(istar)*cos(lamba)*sin(ip) + cos(istar)*cos(ip))
        #    - must be done before modification of istar and lambda chains
        if ('psi' in deriv_prop) or ('psi_lambda' in deriv_prop):
            print('        + Adding true obliquity')
            for pl_loc in fixed_args['lambda_rad_pl']:              
                lambda_rad_pl = 'lambda_rad__pl'+pl_loc
                if fit_dic['fit_mode']=='chi2':              
                    istar = p_final['istar_deg']*np.pi/180.
                    if lambda_rad_pl in fixed_args['genpar_instvis']:
                        for inst in fixed_args['genpar_instvis'][lambda_rad_pl]:
                            for vis in fixed_args['genpar_instvis'][lambda_rad_pl][inst]:
                                p_final['Psi__pl'+pl_loc+'__IS'+inst+'_VS'+vis]=np.arccos(np.sin(istar)*np.cos(p_final[lambda_rad_pl+'__IS'+inst+'_VS'+vis])*np.sin(p_final['inclin_rad__pl'+pl_loc]) + np.cos(istar)*np.cos(p_final['inclin_rad__pl'+pl_loc]))*180./np.pi
                                fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[np.nan],[np.nan]]))                    
                    else:        
                        p_final['Psi__pl'+pl_loc]=np.arccos(np.sin(istar)*np.cos(p_final[lambda_rad_pl])*np.sin(p_final['inclin_rad__pl'+pl_loc]) + np.cos(istar)*np.cos(p_final['inclin_rad__pl'+pl_loc]))*180./np.pi
                        fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[np.nan],[np.nan]]))
                elif fit_dic['fit_mode']=='mcmc':    
                    n_chain = len(merged_chain[:,0])
                    
                    #Obliquity 
                    if 'psi_lambda' in deriv_prop:
                        print('         Using external lambda')
                        if gen_dic['star_name']=='WASP47':
                            lambda_med = 0.         
                            lambda_elow = 24.
                            lambda_ehigh = 24.  
                            
                        lambda_med*=np.pi/180.
                        lambda_elow*=np.pi/180.
                        lambda_ehigh*=np.pi/180.
                        lamb_chain = {'':{'':gen_hrand_chain(lambda_med,lambda_elow,lambda_ehigh,n_chain)}}
                    elif 'psi' in deriv_prop:
                        print('           Using derived lambda')
                        if lambda_rad_pl in fixed_args['genpar_instvis']:
                            lamb_chain={}
                            for inst in fixed_args['genpar_instvis'][lambda_rad_pl]:
                                lamb_chain[inst] = {}
                                for vis in fixed_args['genpar_instvis'][lambda_rad_pl][inst]:      
                                    lamb_chain[inst][vis] = np.squeeze(merged_chain[:,np_where1D(fixed_args['var_par_list']==lambda_rad_pl+'__IS'+inst+'_VS'+vis)]  ) 
                        else:
                            lamb_chain = {'':{'':np.squeeze(merged_chain[:,np_where1D(fixed_args['var_par_list']==lambda_rad_pl)]  )}} 
                    
                    #Stellar inclination
                    if ('istar_deg' in fixed_args['var_par_list']):
                        print('           Using derived istar')
                        istarN_chain=np.squeeze(merged_chain[:,np_where1D(fixed_args['var_par_list']=='istar_deg')]    )*np.pi/180. 
                        istarS_chain=np.pi-istarN_chain
                    else:
                        print('           Using external istar')
                        
                        #Complex PDF on istar
                        if pl_loc=='HD89345b':                        
                            istar_mean = 37.*np.pi/180. 
                            frac_chain = 0.75
                            rand_draw_right = np.random.uniform(low=istar_mean, high=90.*np.pi/180., size=4*n_chain)
                            rand_draw_right = rand_draw_right[rand_draw_right>istar_mean]
                            rand_draw_right = rand_draw_right[0:int(frac_chain*n_chain)]    #cf Fig 5 Van Eylen+2018, 68% contenu dans la partie uf   
                            rand_draw_left = np.random.normal(loc=istar_mean, scale=15.*np.pi/180., size=4*n_chain)
                            rand_draw_left = rand_draw_left[(rand_draw_left<=istar_mean) & (rand_draw_left>=0.)]
                            rand_draw_left = rand_draw_left[0:n_chain-len(rand_draw_right)]   
                            istarN_chain = np.append(rand_draw_left,rand_draw_right)
                    
                            # #Check distribution
                            # hist_val, bin_edg_val = np.histogram(istarN_chain, bins=50,density=True)
                            # grid_val = 0.5*(bin_edg_val[0:-1]+bin_edg_val[1:])
                            # cdf_val = np.cumsum(hist_val)
                            # cdf_val = (cdf_val-np.min(cdf_val))/(np.max(cdf_val)-np.min(cdf_val))                            
                            # rand_draw = np.random.uniform(low=0.0, high=1.0, size=len(istarN_chain))
                            # irand_pts = np_interp(rand_draw,cdf_val,grid_val)  
                            # hist_itest, bin_edg_itest = np.histogram(irand_pts, bins=50,density=True)
                            # grid_itest = 0.5*(bin_edg_itest[0:-1]+bin_edg_itest[1:])
                            # plt.plot(grid_itest*180./np.pi,hist_itest,drawstyle='steps-mid',color='orange')
                            # plt.show()
                            # stop()                            

                        #Gaussian or half-gaussian PDFs on istar
                        else:                    
    
                            #Generate distribution for istar
                            if pl_loc=='Kepler25c':
                                istar_mean = 66.7*np.pi/180.    
                                istar_high = 12.1*np.pi/180.
                                istar_low  = 7.4*np.pi/180.
                            elif pl_loc=='Kepler63b':
                                istar_mean = 138.*np.pi/180.    
                                istar_high = 7.*np.pi/180.
                                istar_low  = 7.*np.pi/180.                                
                            elif pl_loc=='HAT_P11b':
                                # istar_mean = 80.*np.pi/180.    
                                # istar_high = 5.*np.pi/180.
                                # istar_low  = 3.*np.pi/180.

                                istar_mean = 160.*np.pi/180.    
                                istar_high = 9.*np.pi/180.
                                istar_low  = 19.*np.pi/180.
                            istarN_chain = gen_hrand_chain(istar_mean,istar_low,istar_high,n_chain)

                        #Symmetrical PDF around 90°
                        istarS_chain=np.pi-istarN_chain
                    
                    
                    
                    
                    
                    #Orbital inclination
                    if ('inclin_rad__pl'+pl_loc in fixed_args['var_par_list']):
                        inclin_rad_chain=np.squeeze(merged_chain[:,np_where1D(fixed_args['var_par_list']=='inclin_rad__pl'+pl_loc)])
                    else:
                        
                        #Generate distribution for ip
                        if pl_loc=='HAT_P3b':
                            ip_mean = 86.31*np.pi/180.    
                            ip_high = 0.19*np.pi/180.
                            ip_low  = 0.19*np.pi/180.
                        elif pl_loc=='HAT_P11b':
                            ip_mean = 89.05*np.pi/180.    
                            ip_high = 0.15*np.pi/180.
                            ip_low  = 0.09*np.pi/180.
                        elif pl_loc=='Kepler25c':
                            ip_mean = 87.236*np.pi/180.    
                            ip_high = 0.039*np.pi/180.
                            ip_low  = 0.042*np.pi/180.
                        elif pl_loc=='HD89345b':
                            ip_mean = 87.68*np.pi/180.    
                            ip_high = 0.1*np.pi/180.
                            ip_low  = 0.1*np.pi/180.
                        elif pl_loc=='Kepler63b':
                            ip_mean = 87.806*np.pi/180.    
                            ip_high = 0.018*np.pi/180.
                            ip_low  = 0.019*np.pi/180.                            
                        elif pl_loc=='WASP107b':
                            ip_mean = 89.56*np.pi/180.    
                            ip_high = 0.078*np.pi/180.
                            ip_low  = 0.078*np.pi/180.  
                        elif pl_loc=='HIP41378d':
                            ip_mean = 89.80*np.pi/180.     
                            ip_high = 0.02*np.pi/180.
                            ip_low  = 0.02*np.pi/180.  
                        elif pl_loc=='WASP47d':
                            ip_mean = 89.55*np.pi/180.     
                            ip_high = 0.30*np.pi/180.
                            ip_low  = 0.27*np.pi/180.  
                        elif pl_loc=='WASP166b':
                            ip_mean = 87.95*np.pi/180.     
                            ip_high = 0.59*np.pi/180.
                            ip_low  = 0.62*np.pi/180.  
                        elif pl_loc=='WASP69b':
                            ip_mean = 86.71*np.pi/180.     
                            ip_high = 0.2*np.pi/180.
                            ip_low  = 0.2*np.pi/180.  

                        n_chain = len(merged_chain[:,0])
                        inclin_rad_chain = gen_hrand_chain(ip_mean,ip_low,ip_high,n_chain)
                    
                    for inst in lamb_chain:
                        for vis in lamb_chain[inst]:
    
                            PsiN_chain=np.arccos(np.sin(istarN_chain)*np.cos(lamb_chain[inst][vis])*np.sin(inclin_rad_chain) + np.cos(istarN_chain)*np.cos(inclin_rad_chain))*180./np.pi
                            PsiS_chain=np.arccos(np.sin(istarS_chain)*np.cos(lamb_chain[inst][vis])*np.sin(inclin_rad_chain) + np.cos(istarS_chain)*np.cos(inclin_rad_chain))*180./np.pi
                              
                            #Combined Psi for istar and pi-istar, assumed equiprobable
                            Psi_chain = 0.5*( PsiN_chain + PsiS_chain   ) 
                        
                            merged_chain=np.concatenate((merged_chain,PsiN_chain[:,None]),axis=1)   
                            merged_chain=np.concatenate((merged_chain,PsiS_chain[:,None]),axis=1)   
                            merged_chain=np.concatenate((merged_chain,Psi_chain[:,None]),axis=1)   
                
                if lambda_rad_pl in fixed_args['genpar_instvis']:  
                    for inst in fixed_args['genpar_instvis'][lambda_rad_pl]:
                        for vis in fixed_args['genpar_instvis'][lambda_rad_pl][inst]:
                            fixed_args['var_par_list']=np.concatenate((fixed_args['var_par_list'],['PsiN__pl'+pl_loc+'__IS'+inst+'_VS'+vis,'PsiS__pl'+pl_loc+'__IS'+inst+'_VS'+vis]))
                            fixed_args['var_par_names']=np.concatenate((fixed_args['var_par_names'],[pl_loc+'_$\psi_{N}$['+inst+']('+vis+')',pl_loc+'_$\psi_{S}$',pl_loc+'_$\psi$['+inst+']('+vis+')']))   
                            fixed_args['var_par_units']=np.concatenate((fixed_args['var_par_units'],['deg','deg','deg']))        
                else:
                    fixed_args['var_par_list']=np.concatenate((fixed_args['var_par_list'],['PsiN__pl'+pl_loc,'PsiS__pl'+pl_loc,'Psi__pl'+pl_loc]))
                    fixed_args['var_par_names']=np.concatenate((fixed_args['var_par_names'],[pl_loc+'_$\psi_{N}$',pl_loc+'_$\psi_{S}$',pl_loc+'_$\psi$']))   
                    fixed_args['var_par_units']=np.concatenate((fixed_args['var_par_units'],['deg','deg','deg']))   
             
        #-------------------------------------------------
            
        #Add argument of ascending node
        #    - Om = np.arctan( -sin(lambda)*tan(ip) )
        #    - must be done before modification of ip and lambda chains
        if 'om' in deriv_prop:
            print('        + Add argument of ascending node')
            for pl_loc in fixed_args['lambda_rad_pl']:
                lambda_rad_pl = 'lambda_rad__pl'+pl_loc
                if ('inclin_rad__'+pl_loc not in fixed_args['var_par_list']):ip_loc=fixed_args['planets_params'][pl_loc]['inclin_rad'] 
                if fit_dic['fit_mode']=='chi2':  
                    if ('inclin_rad__pl'+pl_loc in fixed_args['var_par_list']):ip_loc=p_final['inclin_rad__pl'+pl_loc]
                    if lambda_rad_pl in fixed_args['genpar_instvis']:
                        for inst in fixed_args['genpar_instvis'][lambda_rad_pl]:
                            for vis in fixed_args['genpar_instvis'][lambda_rad_pl][inst]:
                                p_final['Omega__pl'+pl_loc+'__IS'+inst+'_VS'+vis]=np.arctan( -np.sin(p_final[lambda_rad_pl+'__IS'+inst+'_VS'+vis])*np.tan(ip_loc) )*180./np.pi
                                fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[np.nan],[np.nan]]))
                    else: 
                        p_final['Omega__pl'+pl_loc]=np.arctan( -np.sin(p_final[lambda_rad_pl])*np.tan(ip_loc) )*180./np.pi
                        fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[np.nan],[np.nan]]))
                elif fit_dic['fit_mode']=='mcmc': 
                    if ('inclin_rad__pl'+pl_loc in fixed_args['var_par_list']):ip_loc=merged_chain[:,np_where1D(fixed_args['var_par_list']=='inclin_rad__pl'+pl_loc)]
                    if lambda_rad_pl in fixed_args['genpar_instvis']:
                        lamb_chain={}
                        for inst in fixed_args['genpar_instvis'][lambda_rad_pl]:
                            lamb_chain[inst] = {}
                            for vis in fixed_args['genpar_instvis'][lambda_rad_pl][inst]:      
                                lamb_chain[inst][vis] = np.squeeze(merged_chain[:,np_where1D(fixed_args['var_par_list']==lambda_rad_pl+'__IS'+inst+'_VS'+vis)]  ) 
                    else:
                        lamb_chain = {'':{'':np.squeeze(merged_chain[:,np_where1D(fixed_args['var_par_list']==lambda_rad_pl)]  )}} 
                    for inst in lamb_chain:
                        for vis in lamb_chain[inst]:   
                            chain_loc=np.arctan( -np.sin(lamb_chain[inst][vis])*np.tan(ip_loc) )*180./np.pi
                            merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)   

                if lambda_rad_pl in fixed_args['genpar_instvis']:  
                    for inst in fixed_args['genpar_instvis'][lambda_rad_pl]:
                        for vis in fixed_args['genpar_instvis'][lambda_rad_pl][inst]:
                            fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'Omega__pl'+pl_loc+'__IS'+inst+'_VS'+vis)
                            fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],pl_loc+'_$\Omega$['+inst+']('+vis+')') 
                            fixed_args['var_par_units']=np.append(fixed_args['var_par_units'],'deg') 
                else:
                    fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'Omega__pl'+pl_loc)
                    fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],pl_loc+'_$\Omega$')  
                    fixed_args['var_par_units']=np.append(fixed_args['var_par_units'],'deg')               

        #-------------------------------------------------
            
        #Add impact parameter
        #    - b = aRs*cos(ip)
        if ('b' in deriv_prop) and (fixed_args['fit_orbit']):
            print('        + Adding impact parameter')
            for pl_loc in fixed_args['lambda_rad_pl']:
                if ('inclin_rad__pl'+pl_loc not in fixed_args['var_par_list']):ip_loc=fixed_args['planets_params'][pl_loc]['inclin_rad'] 
                else:iip=np_where1D(fixed_args['var_par_list']=='inclin_rad__pl'+pl_loc)[0]
                if ('aRs__pl'+pl_loc not in fixed_args['var_par_list']):aRs_loc=fixed_args['planets_params'][pl_loc]['aRs']   
                else:iaRs = np_where1D(fixed_args['var_par_list']=='aRs__pl'+pl_loc)[0]
                if fit_dic['fit_mode']=='chi2':  
                    #    - db = b*sqrt( (daRs/aRs)^2 + (dcos(ip)/cos(ip))^2 )
                    #      dcos(ip) = sin(ip)*dip 
                    #    - db = b*sqrt( (daRs/aRs)^2 + (tan(ip)*dip)^2 )                
                    if ('inclin_rad__pl'+pl_loc in fixed_args['var_par_list']):
                        ip_loc=p_final['inclin_rad__pl'+pl_loc]
                        dip_loc = fit_dic['sig_parfinal_err']['1s'][:,iip] 
                    else:dip_loc = 0.
                    if ('aRs__pl'+pl_loc in fixed_args['aRs__pl'+pl_loc]):
                        aRs_loc=p_final['aRs__pl'+pl_loc] 
                        daRs = fit_dic['sig_parfinal_err']['1s'][:,iaRs]   
                    else:daRs=0.
                    p_final['b']=aRs_loc*np.abs(np.cos(ip_loc))
                    sig_loc=p_final['b']*np.sqrt( (daRs/aRs_loc)**2. + (np.tan(ip_loc)*dip_loc)**2. )  
                    fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]])    )            
                    
                elif fit_dic['fit_mode']=='mcmc':             
                    if ('inclin_rad__pl'+pl_loc in fixed_args['var_par_list']):ip_loc=merged_chain[:,iip]
                    if ('aRs__pl'+pl_loc in fixed_args['var_par_list']):aRs_loc=merged_chain[:,iaRs]          
                    chain_loc=aRs_loc*np.abs(np.cos(ip_loc))
                    merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)   
                fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],pl_loc+'_b')
                fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],pl_loc+'_b')
                fixed_args['var_par_units']=np.append(fixed_args['var_par_units'],'')
    
        #-------------------------------------------------            
            
        #Orbital inclination
        #    - convert ip[rad] to ip[deg]
        if ('ip' in deriv_prop) and (fixed_args['fit_orbit']):
            print('        + Converting ip in degrees')
            for pl_loc in fixed_args['lambda_rad_pl']:
                iip=np_where1D(fixed_args['var_par_list']=='inclin_rad__pl'+pl_loc)
                if fit_dic['fit_mode']=='chi2':   
                    p_final['ip_deg']=p_final['inclin_rad__pl'+pl_loc]*180./np.pi                     
                elif fit_dic['fit_mode']=='mcmc':                      
                    merged_chain[:,iip]*=180./np.pi   
                    
                    #Fold ip over 0-90
                    print('       + Folding ip')
                    ip_temp=np.squeeze(merged_chain[:,iip])
                    w_gt_90=(ip_temp > 90.)
                    if True in w_gt_90:
                        merged_chain[w_gt_90,iip]=np.mod(ip_temp[w_gt_90],90.)
                    w_lt_0=(ip_temp < 0.)
                    if True in w_lt_0:
                        i_mod=npint(np.abs(ip_temp[w_lt_0])/90.)+1.
                        merged_chain[w_lt_0,iip] = ip_temp[w_lt_0]+i_mod*90.                    
                    
                fixed_args['var_par_list'][iip]='ip_deg__pl'+pl_loc            
                fixed_args['var_par_names'][iip]='i$_\mathrm{p}$['+pl_loc+']($^{\circ}$)'  
                fixed_args['var_par_units'][iip]='deg' 
    
        #-------------------------------------------------            
            
        #Convert lambda[rad] to lambda[deg]
        if ('lambda_deg' in deriv_prop) and ((fixed_args['rout_mode'] in ['IntrProf','ResProf']) or ((fixed_args['rout_mode']=='IntrProp') and (fixed_args['prop_fit']=='rv'))):  
            print('        + Converting lambda in degrees')       
            for pl_loc in fixed_args['lambda_rad_pl']:
                lambda_rad_pl = 'lambda_rad__pl'+pl_loc
                lambda_deg_pl = 'lambda_deg__pl'+pl_loc
                if fit_dic['fit_mode']=='chi2': 
                    def sub_func(lamb_name,new_lamb_name,new_lamb_name_txt):
                        mid_shift = -180.
                        ilamb=np_where1D(fixed_args['var_par_list']==lamb_name)                    
                        lambda_temp = (p_final[lamb_name]*180./np.pi) - mid_shift
                        if lambda_temp>360.:lambda_temp = np.mod(lambda_temp,360.) + mid_shift
                        elif lambda_temp<0.:
                            i_mod=npint(np.abs(lambda_temp)/360.)+1.
                            lambda_temp += i_mod*360.+mid_shift  
                        else:lambda_temp += mid_shift
                        p_final[new_lamb_name] = lambda_temp
                        sig_temp  = fit_dic['sig_parfinal_err']['1s'][0,ilamb]*180./np.pi 
                        fit_dic['sig_parfinal_err']['1s'][:,ilamb] = sig_temp  
                        fixed_args['var_par_list'][ilamb]=new_lamb_name       
                        fixed_args['var_par_names'][ilamb]= new_lamb_name_txt   
                        fixed_args['var_par_units'][ilamb]= 'deg'                         
                        return None
                    
                    if lambda_rad_pl in fixed_args['genpar_instvis']:
                        for inst in fixed_args['genpar_instvis'][lambda_rad_pl]:
                            for vis in fixed_args['genpar_instvis'][lambda_rad_pl][inst]:
                                sub_func(lambda_rad_pl+'__IS'+inst+'_VS'+vis,lambda_deg_pl+'__IS'+inst+'_VS'+vis,'$\lambda$['+pl_loc+']['+inst+']('+vis+')($^{\circ}$)')
                    else:sub_func(lambda_rad_pl,lambda_deg_pl,'$\lambda$['+pl_loc+']($^{\circ}$)')   
                
                    
                elif fit_dic['fit_mode']=='mcmc':   
                    def sub_func(lamb_name,new_lamb_name,new_lamb_name_txt):  
                        ilamb=np_where1D(fixed_args['var_par_list']==lamb_name)                     
                        merged_chain[:,ilamb]*=180./np.pi  
            
                        #Fold lambda over x+[-180;180]
                        #    - choose final range so that the best-fit is not close to a boundary
                        #    - we want lambda in x+[-180;180] ie lambda-x+180 in 0;360
                        #      we fold over 0;360 and then get back to the final range
                        if pl_loc=='GJ436_b':x_mid=70.
                        elif pl_loc=='HD3167_b':x_mid=80.
                        elif pl_loc=='HD3167_c':x_mid=-110.                
                        elif pl_loc=='HAT_P3b':x_mid=0.               
                        elif pl_loc=='K2_105b':x_mid=-80.         
                        elif pl_loc=='HD89345b':x_mid=70. 
                        elif pl_loc=='WASP107b':x_mid=-160. 
                        elif pl_loc=='Kepler25c':x_mid=80. 
                        elif pl_loc=='WASP156b':x_mid=80. 
                        elif pl_loc=='55Cnc_e' and ('20140226' in lamb_name):x_mid = -100.
                        elif pl_loc=='WASP76b':x_mid=-75. 
                        else:x_mid=np.median(merged_chain[:,ilamb])
        
                        print('        + Folding '+lamb_name+' around ',x_mid)
                        mid_shift = x_mid-180.
                        lambda_temp=np.squeeze(merged_chain[:,ilamb])-mid_shift
                        w_gt_360=(lambda_temp > 360.)
                        if True in w_gt_360:
                            merged_chain[w_gt_360,ilamb]=np.mod(lambda_temp[w_gt_360],360.)+mid_shift
                        w_lt_0=(lambda_temp < 0.)
                        if True in w_lt_0:
                            i_mod=npint(np.abs(lambda_temp[w_lt_0])/360.)+1.
                            merged_chain[w_lt_0,ilamb] = lambda_temp[w_lt_0]+i_mod*360.+mid_shift
                
                        fixed_args['var_par_list'][ilamb]=new_lamb_name
                        fixed_args['var_par_names'][ilamb]=new_lamb_name_txt  
                        fixed_args['var_par_units'][ilamb]='deg'      
                        
                        return None
    
                    if lambda_rad_pl in fixed_args['genpar_instvis']:
                        for inst in fixed_args['genpar_instvis'][lambda_rad_pl]:
                            for vis in fixed_args['genpar_instvis'][lambda_rad_pl][inst]:
                                sub_func(lambda_rad_pl+'__IS'+inst+'_VS'+vis,lambda_deg_pl+'__IS'+inst+'_VS'+vis,'$\lambda$['+pl_loc+']['+inst+']('+vis+')($^{\circ}$)')
                    else:sub_func(lambda_rad_pl,lambda_deg_pl,'$\lambda$['+pl_loc+']($^{\circ}$)')       
    
    
        #-------------------------------------------------            
         
        #Retrieve zeroth order CB coefficient
        #    - independent of visits or planets
        if 'c0' in deriv_prop: 
            print('        + Adding c0(CB)')
            if fit_dic['fit_mode']=='chi2': 
                p_final['c0_CB'] = calc_CB_RV(get_LD_coeff(fixed_args['system_prop']['achrom'],0),fixed_args['system_prop']['achrom']['LD'][0],p_final['c1_CB'],p_final['c2_CB'],p_final['c3_CB'],fixed_args['system_param']['star'])[0]            
                sig_loc=np.nan  
                fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]) )     
                          
            elif fit_dic['fit_mode']=='mcmc':   
                chain_loc=np.empty(0,dtype=float)            
                p_final_loc={}
                p_final_loc.update(fixed_args['fixed_par_val'])
                for istep in range(fit_dic['nsteps_final_merged']): 
                    for ipar,par in enumerate(fixed_args['var_par_list']):
                        p_final_loc[par]=merged_chain[istep,ipar]     
                        if len(fixed_args['linked_par_expr'])>0:exec(str(par)+'='+str(p_final_loc[par]))
                    for par in fixed_args['linked_par_expr']:
                        p_final_loc[par]=eval(fixed_args['linked_par_expr'][par])
                    c0_CB = calc_CB_RV(get_LD_coeff(fixed_args['system_prop']['achrom'],0),fixed_args['system_prop']['achrom']['LD'][0],p_final['c1_CB'],p_final['c2_CB'],p_final['c3_CB'],fixed_args['system_param']['star'])[0]            
                    chain_loc=np.append(chain_loc,c0_CB)
                merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)   
            fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'c0_CB')
            fixed_args['var_par_names'] = np.append(fixed_args['var_par_names'],'CB$_{0}$')  
            fixed_args['var_par_units'] = np.append(fixed_args['var_par_units'],'km/s')           
    
    
        #-------------------------------------------------            
         
        #Convert CB coefficients from km/s to m/s
        if 'CB_ms' in deriv_prop: 
            print('        + Converting CB coefficients to m/s') 
            ipar_mult_loc=[]
            for ipar_name in ['c0_CB','c1_CB','c2_CB','c3_CB']:
                ipar_loc=np_where1D(fixed_args['var_par_list']==ipar_name)
                if len(ipar_loc)>0:ipar_mult_loc+=[ipar_loc]
                if ipar_name=='c0_CB':fixed_args['var_par_names'][ipar_loc]='CB$_{0}$ (m s$^{-1}$)'
                if ipar_name=='c1_CB':fixed_args['var_par_names'][ipar_loc]='CB$_{1}$ (m s$^{-1}$)'
                if ipar_name=='c2_CB':fixed_args['var_par_names'][ipar_loc]='CB$_{2}$ (m s$^{-1}$)'
                if ipar_name=='c3_CB':fixed_args['var_par_names'][ipar_loc]='CB$_{3}$ (m s$^{-1}$)'  
                fixed_args['var_par_units'][ipar_loc]='m/s'            
                if fit_dic['fit_mode']=='chi2': 
                    p_final[ipar_name]*=1e3 
                    fit_dic['sig_parfinal_err']['1s'][:,ipar_loc]*=1e3  
                elif fit_dic['fit_mode']=='mcmc':  
                    merged_chain[:,ipar_loc]*=1e3 
    
        #-------------------------------------------------            
            
        #Convert FWHM and contrast of true intrinsic stellar profiles into values for observed profiles
        #    - only for constant FWHM and contrast with no variation in mu
        #    - for double-gaussian profiles the values can be converted into the 'true' contrast and FWHM, either of the intrinsic or of the observed profile
        if ('CF0_meas_add' in deriv_prop) or ('CF0_meas_conv' in deriv_prop):  
            print('        + Converting intrinsic ctrst and FWHM to measured values') 
            merged_chain = conv_CF_intr_meas(deriv_prop,fixed_args['inst_list'],fixed_args['inst_vis_list'],fixed_args,merged_chain,gen_dic,p_final,fit_dic,fit_prop_dic)
        if ('CF0_DG_add' in deriv_prop) or ('CF0_DG_conv' in deriv_prop):  
            print('        + Converting DG ctrst and FWHM to true intrinsic values') 
            merged_chain = conv_CF_intr_meas(deriv_prop,fixed_args['inst_list'],fixed_args['inst_vis_list'],fixed_args,merged_chain,gen_dic,p_final,fit_dic,fit_prop_dic)
            
    #---------------------------------------------------------------            
    #Process:
    #    - best-fit values and confidence interval for the final parameters, potentially modified
    #    - correlation diagram plot
    if fit_dic['fit_mode']=='mcmc':   
        p_final=postMCMCwrapper_2(fit_dic,fixed_args,merged_chain)

    #----------------------------------------------------------
    #Save derived parameters
    #----------------------------------------------------------
    save_fit_results('derived',fixed_args,fit_dic,fit_dic['fit_mode'],p_final)

    #Close save file
    fit_dic['file_save'].close() 

    return None



def conv_cosistar(deriv_prop,fixed_args_in,fit_dic_in,p_final_in,merged_chain_in):
    r"""**Parameter conversion: cos(i_star)**

    Converts results from :math:`\chi^2` or mcmc fitting: from :math:`\cos(i_\star)` to :math:`i_\star`. 

    Args:
        TBD

    Returns:
        TBD
        
    """    
    if 'cos_istar' in fixed_args_in['var_par_list']:                    
        iistar=np_where1D(fixed_args_in['var_par_list']=='cos_istar')                    
        if fit_dic_in['fit_mode']=='chi2':
            #Folding cos(istar) within -1 : 1
            #    new = old - n*2 thus error or new is the same as error on old
            cosistar = (p_final_in['cos_istar']-(1.)) % 2 - 1.                     
            #    - dcosi = sin(i)*di
            #      di = dcosi/sin(i)                      
            p_final_in['istar_deg']= np.arccos(cosistar)*180./np.pi   
            sig_loc= (180./np.pi)*fit_dic_in['sig_parfinal_err']['1s'][0,iistar][0] / np.sqrt(1.-cosistar**2.)   
            if ('istar_deg_conv') in deriv_prop:fit_dic_in['sig_parfinal_err']['1s'][:,iistar] = [[sig_loc],[sig_loc]]
            elif ('istar_deg_add') in deriv_prop:fit_dic_in['sig_parfinal_err']['1s'] = np.hstack((fit_dic_in['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]])  )               
        elif fit_dic_in['fit_mode']=='mcmc':
            #Folding cos(istar) within -1 : 1
            cosistar_chain = (merged_chain_in[:,iistar]-(1.)) % 2 - 1.
            chain_loc = np.arccos(cosistar_chain)*180./np.pi     
            if ('istar_deg_conv') in deriv_prop:merged_chain_in[:,iistar]= chain_loc
            elif ('istar_deg_add') in deriv_prop:merged_chain_in=np.concatenate((merged_chain_in,chain_loc[:,None]),axis=1)  
        if ('istar_deg_conv') in deriv_prop:    
            fixed_args_in['var_par_list'][iistar]='istar_deg'            
            fixed_args_in['var_par_names'][iistar]='i$_{*}(^{\circ}$)'   
            fixed_args_in['var_par_units'][iistar]='deg'   
        elif ('istar_deg_add') in deriv_prop: 
            fixed_args_in['var_par_list']=np.append(fixed_args_in['var_par_list'],'istar_deg')
            fixed_args_in['var_par_names']=np.append(fixed_args_in['var_par_names'],'i$_{*}(^{\circ}$)') 
            fixed_args_in['var_par_units']=np.append(fixed_args_in['var_par_units'],'deg')
    else:
        if fit_dic_in['fit_mode']=='chi2':p_final_in['istar_deg']=np.arccos(p_final_in['cos_istar'])*180./np.pi  
        elif fit_dic_in['fit_mode']=='mcmc':fixed_args_in['fixed_par_val']['istar_deg']=np.arccos(p_final_in['cos_istar'])*180./np.pi                          
    return merged_chain_in



def conv_CF_intr_meas(deriv_prop,inst_list,inst_vis_list,fixed_args,merged_chain,gen_dic,p_final,fit_dic,fit_prop_dic):
    r"""**Parameter conversion: line contrast and FWHM**

    Converts results from :math:`\chi^2` or mcmc fitting: from intrinsic width and contrast to measured-like values. 

    Args:
        TBD

    Returns:
        TBD
        
    """ 
    #HR RV table to calculate FWHM and contrast at high enough precision                       
    if fixed_args['func_prof_name']=='dgauss':
        fixed_args['velccf_HR'] = gen_dic['RVstart_HR_mod'] + gen_dic['dRV_HR_mod']*np.arange(  int((gen_dic['RVend_HR_mod']-gen_dic['RVstart_HR_mod'])/gen_dic['dRV_HR_mod'])  )
    for inst in inst_list:               
        fixed_args_loc = deepcopy(fixed_args)
        if ('CF0_DG_add' in deriv_prop) or ('CF0_DG_conv' in deriv_prop):fixed_args_loc['FWHM_inst']=None
        else:fixed_args_loc['FWHM_inst'] = calc_FWHM_inst(inst,c_light)   
        for vis in inst_vis_list[inst]:
            if fixed_args['func_prof_name']=='gauss':
                if any('ctrst_ord1' in par_loc for par_loc in fixed_args['var_par_list']):stop('    Contrast must be constant')
                if any('FWHM_ord1' in par_loc for par_loc in fixed_args['var_par_list']):stop('    FWHM must be constant')                            
            ctrst0_name = fixed_args['name_prop2input']['ctrst_ord0__IS'+inst+'_VS'+vis]
            varC=any('ctrst_ord0' in par_loc for par_loc in fixed_args['var_par_list'])
            FWHM0_name = fixed_args['name_prop2input']['FWHM_ord0__IS'+inst+'_VS'+vis]   
            varF=any('FWHM_ord0' in par_loc for par_loc in fixed_args['var_par_list'])                      
            if fixed_args['func_prof_name']=='dgauss': 
                p_final_loc=deepcopy(p_final)
                p_final_loc['rv'] = 0.    #the profile position does not matter
                p_final_loc['ctrst'] = p_final_loc[ctrst0_name]  
                p_final_loc['FWHM'] = p_final_loc[FWHM0_name]                           
                for par_sub in ['amp_l2c','FWHM_l2c','rv_l2c']:p_final_loc[par_sub]=p_final_loc[fixed_args['name_prop2input'][par_sub+'__IS'+inst+'_VS'+vis]]   
                p_final_loc['cont'] = 1.  #the profile continuum does not matter                  
            if fit_dic['fit_mode']=='chi2': 
                if fixed_args['func_prof_name']=='gauss':
                    p_final['ctrst0__IS'+inst+'_VS'+vis],p_final['FWHM0__IS'+inst+'_VS'+vis] = gauss_intr_prop(p_final[ctrst0_name],p_final[FWHM0_name],fixed_args_loc['FWHM_inst']) 
                elif fixed_args['func_prof_name']=='dgauss':
                    p_final['ctrst0__IS'+inst+'_VS'+vis],p_final['FWHM0__IS'+inst+'_VS'+vis]=cust_mod_true_prop(p_final_loc,fixed_args['velccf_HR'],fixed_args_loc)[0:3]   
                sig_loc=np.nan 
                if varC:fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]])) 
                if varF:fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))                                                      
            elif fit_dic['fit_mode']=='mcmc':  
                if varC:ictrst_loc = np_where1D(fixed_args['var_par_list']==ctrst0_name)[0]
                if varF:iFWHM_loc = np_where1D(fixed_args['var_par_list']==FWHM0_name)[0] 
                if fixed_args['func_prof_name']=='gauss':
                    if varC:chain_ctrst0 = np.squeeze(merged_chain[:,ictrst_loc])
                    else:chain_ctrst0=np.repeat(p_final[ctrst0_name],fit_dic['nsteps_final_merged'])
                    if varF:chain_FWHM0 = np.squeeze(merged_chain[:,iFWHM_loc])
                    else:chain_FWHM0=np.repeat(p_final[FWHM0_name],fit_dic['nsteps_final_merged'])                             
                    chain_ctrst_temp,chain_FWHM_temp = gauss_intr_prop(chain_ctrst0,chain_FWHM0,fixed_args_loc['FWHM_inst'])                        
                if fixed_args['func_prof_name']=='dgauss': 
                    if varC:fixed_args_loc['var_par_list'][ictrst_loc]='ctrst'
                    if varF:fixed_args_loc['var_par_list'][iFWHM_loc]='FWHM'
                    for par_sub in ['amp_l2c','FWHM_l2c','rv_l2c']:
                        if any(par_sub in par_loc for par_loc in fixed_args['var_par_list']):fixed_args_loc['var_par_list'][fixed_args['var_par_list']==fixed_args['name_prop2input'][par_sub+'__IS'+inst+'_VS'+vis]]=par_sub       
                    if fit_prop_dic['nthreads']>1:chain_loc=para_cust_mod_true_prop(proc_cust_mod_true_prop,fit_prop_dic['nthreads'],fit_dic['nsteps_final_merged'],[merged_chain],(fixed_args_loc,p_final_loc,))                           
                    else:  chain_loc=proc_cust_mod_true_prop(merged_chain,fixed_args_loc,p_final_loc)   
                    chain_ctrst_temp = chain_loc[0]
                    chain_FWHM_temp = chain_loc[1]     
                if ('CF0_meas_add' in deriv_prop) or ('CF0_DG_add' in deriv_prop):   #add parameters
                    if varC:merged_chain=np.concatenate((merged_chain,chain_ctrst_temp[:,None]),axis=1) 
                    if varF:merged_chain=np.concatenate((merged_chain,chain_FWHM_temp[:,None]),axis=1) 
                elif ('CF0_meas_conv' in deriv_prop) or ('CF0_DG_conv' in deriv_prop):   #replace parameters
                    if varC:merged_chain[:,ictrst_loc] = chain_ctrst_temp
                    if varF:merged_chain[:,iFWHM_loc] = chain_FWHM_temp              
            if ('CF0_meas_add' in deriv_prop) or ('CF0_DG_add' in deriv_prop):
                if varC:
                    fixed_args['var_par_list']=np.concatenate((fixed_args['var_par_list'],[fixed_args['var_par_list'][ictrst_loc]+'_'+inst]))
                    fixed_args['var_par_names']=np.concatenate((fixed_args['var_par_names'],['Contrast$_\mathrm{'+inst+'}$']))
                    fixed_args['var_par_units']=np.concatenate((fixed_args['var_par_units'],['']))
                if varF:
                    fixed_args['var_par_list']=np.concatenate((fixed_args['var_par_list'][fixed_args['var_par_list'][iFWHM_loc]+'_'+inst]))
                    fixed_args['var_par_names']=np.concatenate((fixed_args['var_par_names']['FWHM$_\mathrm{'+inst+'}$(km/s)']))
                    fixed_args['var_par_units']=np.concatenate((fixed_args['var_par_units'],['km/s']))
            elif ('CF0_meas_conv' in deriv_prop) or ('CF0_DG_conv' in deriv_prop):
                if varC:
                    fixed_args['var_par_list'][ictrst_loc] = fixed_args['var_par_list'][ictrst_loc]+'_'+inst
                    fixed_args['var_par_names'][ictrst_loc] = 'Contrast$_\mathrm{'+inst+'}$'
                    fixed_args['var_par_units'][ictrst_loc] = ''
                if varF:                                
                    fixed_args['var_par_list'][iFWHM_loc] = fixed_args['var_par_list'][iFWHM_loc]+'_'+inst
                    fixed_args['var_par_names'][iFWHM_loc] = 'FWHM$_\mathrm{'+inst+'}$(km/s)'
                    fixed_args['var_par_units'][iFWHM_loc] = ''

    return merged_chain           

