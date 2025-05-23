#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np
import scipy.linalg
from scipy.interpolate import interp1d
import bindensity as bind
import os
from ..ANTARESS_general.constant_data import c_light
from ..ANTARESS_general.utils import stop,np_where1D,dataload_npz,gen_specdopshift
from ..ANTARESS_analysis.ANTARESS_ana_comm import init_joined_routines,init_joined_routines_inst,init_joined_routines_vis,init_joined_routines_vis_fit,com_joint_fits,com_joint_postproc
from ..ANTARESS_grids.ANTARESS_occ_grid import sub_calc_plocc_ar_prop,up_plocc_arocc_prop,calc_ar_tiles, calc_plocced_tiles
from ..ANTARESS_grids.ANTARESS_prof_grid import gen_theo_intr_prof,theo_intr2loc,custom_DI_prof,init_st_intr_prof
from ..ANTARESS_analysis.ANTARESS_inst_resp import get_FWHM_inst,resamp_st_prof_tab,def_st_prof_tab,conv_st_prof_tab,cond_conv_st_prof_tab,convol_prof
from ..ANTARESS_grids.ANTARESS_star_grid import up_model_star
from ..ANTARESS_conversions.ANTARESS_binning import weights_bin_prof,weights_bin_prof_calc
from ..ANTARESS_analysis.ANTARESS_model_prof import polycoeff_def
from ..ANTARESS_corrections.ANTARESS_detrend import detrend_prof_gen_mul,detrend_prof_gen_add


def joined_Star_ana(glob_fit_dic,system_param,theo_dic,data_dic,gen_dic,plot_dic,coord_dic,data_prop):
    r"""**Joined stellar fits**

    Wrap-up function to call joint fits of stellar properties and profiles

    Args:
        TBD
    
    Returns:
        TBD
    
    """        
    #Fitting disk-integrated stellar properties with a linked model
    if gen_dic['fit_DIProp']:
        main_joined_DIProp('DIProp',glob_fit_dic['DIProp'],gen_dic,system_param,theo_dic,plot_dic,coord_dic,data_dic,data_prop)   

    #Fitting stellar surface properties with a linked model
    if gen_dic['fit_IntrProp']:
        main_joined_IntrProp('IntrProp',glob_fit_dic['IntrProp'],gen_dic,system_param,theo_dic,plot_dic,coord_dic,data_dic)    
    
    #Fitting intrinsic stellar lines with joined model
    if gen_dic['fit_IntrProf']:
        main_joined_IntrProf('IntrProf',data_dic,gen_dic,system_param,glob_fit_dic['IntrProf'],theo_dic,plot_dic,coord_dic)    

    #Fitting local stellar lines with joined model, including active regions in the fitted parameters
    if gen_dic['fit_DiffProf']:
        main_joined_DiffProf('DiffProf',data_dic,gen_dic,system_param,glob_fit_dic['DiffProf'],theo_dic,plot_dic,coord_dic)    

    return None


def main_joined_DIProp(rout_mode,fit_prop_dic,gen_dic,system_param,theo_dic,plot_dic,coord_dic,data_dic,data_prop):
    r"""**Joined disk-integrated stellar property fits**

    Main routine to fit time-series of properties derived from disk-integrated profiles as a function of various parameters, to search for systematic trends
    
    Individual visits can still be fitted, but the use of a joint model over instruments and visits allows better characterizing stellar variations.
    
    Results of the analysis are saved, to be used in the detrending module. 

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    print('   > Fitting single disk-integrated stellar properties')
    
    #Initializations
    for prop_loc in fit_prop_dic['mod_prop']:  
        fixed_args,fit_dic = init_joined_routines(rout_mode,gen_dic,system_param,theo_dic,data_dic,fit_prop_dic)
        print('     - '+{'rv':'RV residuals','FWHM':'Line FWHM','ctrst':'Line contrast'}[prop_loc])        
        fit_dic['save_dir']+=prop_loc+'/'       
    
        #Arguments to be passed to the fit function
        fixed_args.update({
            'coord_list':{},
            'coord_obs':{},
            'coord_fit':{},
            'SNRorders':{},
            'coord_ref':fit_prop_dic['coord_ref']
            })    
        if prop_loc=='rv':fixed_args['prop_fit'] = 'rv_res'
        else:fixed_args['prop_fit'] = prop_loc

        #Initialization
        for inst in np.intersect1d(data_dic['instrum_list'],list(fit_dic['idx_in_fit'].keys())):    
            init_joined_routines_inst(inst,fit_dic,fixed_args)
            fixed_args['coord_list'][inst]={}
            for vis in data_dic[inst]['visit_list']:
                init_joined_routines_vis(inst,vis,fit_dic,fixed_args)
                fixed_args['coord_list'][inst][vis]={}
      
        #Identify models and coordinates
        #    - must be done here to retrieve coordinate grids within init_joined_routines_vis_fit()
        for par in fit_dic['mod_prop'][prop_loc]:
            coord = par.split('__')[0]
            if coord!='c':
                model_type = []
                if 'pol' in par:model_type+=['pol']
                if 'sin' in par:model_type+=['sin']
                if 'puls' in par:model_type+=['puls']
                if 'ramp' in par:model_type+=['ramp']
                inst_vis_coord = par.split('__IS')[1]
                inst_coord  = inst_vis_coord.split('_VS')[0]
                vis_coord  = inst_vis_coord.split('_VS')[1] 
                if inst_coord=='_':inst_list = fixed_args['inst_list']             
                elif inst_coord in fixed_args['inst_list']:inst_list = [inst_coord]
                else: stop('ERROR: Instrument '+inst_coord+' in '+par+' is not included in processed instruments')
                for inst_loc in inst_list:
                    if vis_coord=='_':vis_list = fixed_args['inst_vis_list'][inst_loc]
                    elif vis_coord in fixed_args['inst_vis_list'][inst_loc]:vis_list = [vis_coord]              
                    else:stop('ERROR: Visit '+vis_coord+' in '+par+' is not included in processed visits')  
                    for vis_loc in vis_list:
                        fixed_args['coord_list'][inst_loc][vis_loc][coord]=model_type
     
        #Construction of the fit tables
        for par in ['s_val','y_val']:fixed_args[par]=np.zeros(0,dtype=float)
        idx_fit2vis={}
        for inst in fixed_args['inst_list']:    
            idx_fit2vis[inst] = {}
            for vis in fixed_args['inst_vis_list'][inst]:
                data_vis=data_dic[inst][vis]
                init_joined_routines_vis_fit('DIProp',inst,vis,fit_dic,fixed_args,data_vis,gen_dic,data_dic,coord_dic,theo_dic,data_prop,plot_dic)
       
                #Binned/original data
                if fixed_args['bin_mode'][inst][vis]=='_bin':
                    data_load = dataload_npz(gen_dic['save_data_dir']+'/DIbin_prop/'+inst+'_'+vis)
                else:
                    data_load = dataload_npz(gen_dic['save_data_dir']+'/DIorig_prop/'+inst+'_'+vis)
         
                #Fit tables
                idx_fit2vis[inst][vis] = range(fit_dic['nx_fit'],fit_dic['nx_fit']+fixed_args['nexp_fit_all'][inst][vis])
                fit_dic['nx_fit']+=fixed_args['nexp_fit_all'][inst][vis]
                for iexp in fixed_args['idx_in_fit'][inst][vis]:    
                    fixed_args['y_val'] = np.append(fixed_args['y_val'],data_load[iexp][fixed_args['prop_fit']])
                    fixed_args['s_val'] = np.append(fixed_args['s_val'],np.mean(data_load[iexp]['err_'+fixed_args['prop_fit']]))
                
                #Scaling variance
                fixed_args['s_val']*=np.sqrt(fixed_args['sc_var'])

        fixed_args['idx_fit'] = np.ones(fit_dic['nx_fit'],dtype=bool)
        fixed_args['nexp_fit'] = fit_dic['nx_fit']
        fixed_args['x_val']=range(fixed_args['nexp_fit'])
        fixed_args['fit_func'] = FIT_joined_DIProp
        fixed_args['mod_func'] = joined_DIProp
        fixed_args['inside_fit'] = False    

        #Uncertainties on the property are given a covariance matrix structure for consistency with the fit routine 
        fixed_args['cov_val'] = np.array([fixed_args['s_val']**2.])
        fixed_args['use_cov'] = False       

        #Model fit and calculation
        merged_chain,p_final = com_joint_fits('DIProp',fit_dic,fixed_args,gen_dic,data_dic,theo_dic,fit_dic['mod_prop'][prop_loc])   

        #Best-fit model and properties
        fit_save={}
        fixed_args['fit'] = False
        mod_tab,fit_save['prop_mod'] = fixed_args['mod_func'](p_final,fixed_args)

        #Save best-fit properties
        fit_save.update({'p_final':p_final,'name_prop2input':fixed_args['name_prop2input'],'coeff_ord2name':fixed_args['coeff_ord2name'],'merit':fit_dic['merit'],'SNRorders':fixed_args['SNRorders'],'coord_ref':fixed_args['coord_ref']})
        if (plot_dic['prop_DI']!='') or (plot_dic['chi2_fit_DIProp']!=''):
            for key in ['coord_obs','coord_list','coord_fit']:fit_save[key] = fixed_args[key]
            fit_save['coord_mod'] = fixed_args['coord_fit']
            key_list = ['prop_fit','err_prop_fit']
            for key in key_list:fit_save[key] = {}
            for inst in fixed_args['inst_list']:
                for key in key_list:fit_save[key][inst] = {}
                for vis in fixed_args['inst_vis_list'][inst]:
                    fit_save['prop_fit'][inst][vis] = fixed_args['y_val'][idx_fit2vis[inst][vis]]
                    fit_save['err_prop_fit'][inst][vis] = fixed_args['s_val'][idx_fit2vis[inst][vis]]
        np.savez(fit_dic['save_dir']+'Fit_results',data=fit_save,allow_pickle=True)
    
        #Post-processing
        fit_dic['p_null'] = deepcopy(p_final)
        for par in fit_dic['p_null']:
            fit_dic['p_null'][par]=1e-10   #to avoid division by 0
        fit_dic['p_null']['cont']=1.
        com_joint_postproc(p_final,fixed_args,fit_dic,merged_chain,gen_dic)

    print('     ----------------------------------')    
    
    return None


def FIT_joined_DIProp(param,x_tab,args=None):
    r"""**Fit function: joined global stellar property**

    Calls corresponding model function for optimization

    Args:
        TBD
    
    Returns:
        TBD
    """
    return joined_DIProp(param,args)[0],None


def mod_DIProp(param,args,inst,vis,n_coord):
    r"""**Model function: global stellar property**

    Defines the model for global stellar property over a visit.

    Args:
        TBD
    
    Returns:
        TBD
    
    """   

    #Model property for the visit 
    #    - see description in ANTARESS_settings.py (field 'glob_fit_dic['DIProp']['mod_prop']')      
    mod_prop=param[args['name_prop2input']['c__ord0__IS'+inst+'_VS'+vis]]*np.ones(n_coord)     
    
    #Processing coordinates for current visit
    for coord in args['coord_list'][inst][vis]:  
        coord_grid = args['coord_fit'][inst][vis][coord]
        
        #Processing model associated with coordinate
        for mod in args['coord_list'][inst][vis][coord]:                
            corr_prop = {}
         
            #Sinusoidal variation 
            if mod=='sin':corr_prop['sin'] = [param[args['name_prop2input'][coord+'__sin__'+corr_val+'__IS'+inst+'_VS'+vis]] for corr_val in ['amp', 'off', 'per'] ]
            
            #Pulsation pattern
            elif mod=='puls':corr_prop['puls'] = [param[args['name_prop2input'][coord+'__puls__'+corr_val+'__IS'+inst+'_VS'+vis]] for corr_val in ['ampHF', 'phiHF', 'freqHF','ampLF', 'phiLF', 'freqLF','f'] ]
            
            #Polynomial variation
            elif mod=='pol':corr_prop['pol']= polycoeff_def(param,args['coeff_ord2name'][inst][vis][coord+'__pol'])[1::]   

            #Ramp
            elif mod=='ramp':corr_prop['ramp'] = [param[args['name_prop2input'][coord+'__ramp__'+corr_val+'__IS'+inst+'_VS'+vis]] for corr_val in ['lnk',  'alpha' , 'tau' , 'xr' ] ]
            
            #Contribution to model
            if args['prop_fit']=='rv_res':mod_prop = detrend_prof_gen_add( coord, corr_prop, coord_grid, mod_prop, args)  
            else:mod_prop = detrend_prof_gen_mul( coord, corr_prop, coord_grid, mod_prop , args )             

    return mod_prop


def joined_DIProp(param,args):
    r"""**Model function: joined global stellar property**

    Defines the joined model for global stellar properties

    Args:
        TBD
    
    Returns:
        TBD
    
    """   
    mod_tab=np.zeros(0,dtype=float)
    mod_prop_dic = {}

    #Process each visit
    for inst in args['inst_list']:
        mod_prop_dic[inst]={}
        for vis in args['inst_vis_list'][inst]: 
            
            #Model property for the visit 
            mod_prop_vis = mod_DIProp(param,args,inst,vis,args['nexp_fit_all'][inst][vis])
            
            #Appending over all visits
            mod_prop_dic[inst][vis] = mod_prop_vis
            mod_tab=np.append(mod_tab,mod_prop_vis)

    return mod_tab,mod_prop_dic
    


    








def main_joined_IntrProp(rout_mode,fit_prop_dic,gen_dic,system_param,theo_dic,plot_dic,coord_dic,data_dic):
    r"""**Joined intrinsic stellar property fits**

    Main routine to fit a given stellar surface property from planet-occulted regions with a joined model over instruments and visits.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    print('   > Fitting single stellar surface property')
    
    #Initializations
    for prop_loc in fit_prop_dic['mod_prop']:  
        fixed_args,fit_dic = init_joined_routines(rout_mode,gen_dic,system_param,theo_dic,data_dic,fit_prop_dic)
        print('     - '+{'rv':'Surface RVs','FWHM':'Intrinsic line FWHM','ctrst':'Intrinsic line contrast','a_damp':'Intrinsic line damping coefficient'}[prop_loc])        
        fit_dic['save_dir']+=prop_loc+'/'        
    
        #Arguments to be passed to the fit function
        #    - fit is performed on achromatic average properties
        #    - the full stellar line profile are not calculated, since we only fit the average properties of the occulted regions
        fixed_args.update({
            'chrom_mode':'achrom',
            'mode':'ana',  #to activate the calculation of line profile properties
            'prop_fit':prop_loc})

        #Coordinate as a function of which the model describing the line shape property is defined
        if prop_loc != 'rv':
            if fit_prop_dic['coord_fit'][prop_loc] in ['abs_y_st','y_st2']:fixed_args['coord_line']='y_st'    
            else:fixed_args['coord_line']=fit_prop_dic['coord_fit'][prop_loc]
        else:fixed_args['coord_line']=None
        
        #Property required to calculate the model
        fixed_args['par_list']+=[fixed_args['prop_fit']]
        if fixed_args['coord_line'] is not None:fixed_args['par_list']+=[fixed_args['coord_line']]
    
        #Construction of the fit tables
        for par in ['s_val','y_val']:fixed_args[par]=np.zeros(0,dtype=float)
        for par in ['coord_obs','coord_fit','ph_fit']:fixed_args[par]={}
        idx_fit2vis={}
        for inst in np.intersect1d(data_dic['instrum_list'],list(fit_dic['idx_in_fit'].keys())):    
            init_joined_routines_inst(inst,fit_dic,fixed_args)
            idx_fit2vis[inst] = {}
            for vis in data_dic[inst]['visit_list']:
                init_joined_routines_vis(inst,vis,fit_dic,fixed_args)
    
                #Visit is fitted
                if vis in fixed_args['inst_vis_list'][inst]:
                    data_vis=data_dic[inst][vis]
                    init_joined_routines_vis_fit('IntrProp',inst,vis,fit_dic,fixed_args,data_vis,gen_dic,data_dic,coord_dic,theo_dic,None,None)
    
                    #Binned/original data
                    if fixed_args['bin_mode'][inst][vis]=='_bin':data_load = dataload_npz(gen_dic['save_data_dir']+'/Intrbin_prop/'+inst+'_'+vis)
                    else:data_load = dataload_npz(gen_dic['save_data_dir']+'/Introrig_prop/'+inst+'_'+vis)
                  
                    #Fit tables
                    idx_fit2vis[inst][vis] = range(fit_dic['nx_fit'],fit_dic['nx_fit']+fixed_args['nexp_fit_all'][inst][vis])
                    fit_dic['nx_fit']+=fixed_args['nexp_fit_all'][inst][vis]
                    for i_in in fixed_args['idx_in_fit'][inst][vis]:    
                        fixed_args['y_val'] = np.append(fixed_args['y_val'],data_load[i_in][fixed_args['prop_fit']])
                        fixed_args['s_val'] = np.append(fixed_args['s_val'],np.mean(data_load[i_in]['err_'+fixed_args['prop_fit']]))

                    #Scaling variance
                    fixed_args['s_val']*=np.sqrt(fixed_args['sc_var'])

        #Final processing
        for idx_inst,inst in enumerate(fixed_args['inst_list']):
            
            #Common data type        
            if idx_inst==0:fixed_args['type'] = data_dic[inst]['type']
            elif fixed_args['type'] != data_dic[inst]['type']:stop('Incompatible data types')

        fixed_args['idx_fit'] = np.ones(fit_dic['nx_fit'],dtype=bool)
        fixed_args['nexp_fit'] = fit_dic['nx_fit']
        fixed_args['x_val']=range(fixed_args['nexp_fit'])
        fixed_args['fit_func'] = FIT_joined_IntrProp
        fixed_args['mod_func'] = joined_IntrProp
        fixed_args['inside_fit'] = False

        #Uncertainties on the property are given a covariance matrix structure for consistency with the fit routine 
        fixed_args['cov_val'] = np.array([fixed_args['s_val']**2.])
        fixed_args['use_cov'] = False   

        #Model fit and calculation
        if prop_loc not in fit_dic['mod_prop']:fit_dic['mod_prop'][prop_loc] = {}
        merged_chain,p_final = com_joint_fits('IntrProp',fit_dic,fixed_args,gen_dic,data_dic,theo_dic,fit_dic['mod_prop'][prop_loc])   
        
        #Best-fit model and properties
        fit_save={}
        fixed_args['fit'] = False
        mod_tab,coeff_line_dic,fit_save['prop_mod'],fit_save['coord_mod']= fixed_args['mod_func'](p_final,fixed_args)
      
        #Save best-fit properties
        fit_save.update({'p_final':p_final,'coeff_line_dic':coeff_line_dic,'name_prop2input':fixed_args['name_prop2input'],'coord_line':fixed_args['coord_line'],'pol_mode':fit_prop_dic['pol_mode'],'genpar_instvis':fixed_args['genpar_instvis'],'linevar_par':fixed_args['linevar_par'],
                         'merit':fit_dic['merit']})
        if (plot_dic['prop_Intr']!='') or (plot_dic['chi2_fit_IntrProp']!=''):
            key_list = ['prop_fit','err_prop_fit']
            for key in key_list:fit_save[key] = {}
            for inst in fixed_args['inst_list']:
                for key in key_list:fit_save[key][inst] = {}
                for vis in fixed_args['inst_vis_list'][inst]:
                    fit_save['prop_fit'][inst][vis] = fixed_args['y_val'][idx_fit2vis[inst][vis]]
                    fit_save['err_prop_fit'][inst][vis] = fixed_args['s_val'][idx_fit2vis[inst][vis]]
        np.savez(fit_dic['save_dir']+'Fit_results',data=fit_save,allow_pickle=True)
    
        #Post-processing
        fit_dic['p_null'] = deepcopy(p_final)
        if fixed_args['prop_fit']=='rv':fit_dic['p_null']['veq'] = 1e-9 #not zero to prevent division by 0
        else:
            for par in fit_dic['p_null']:fit_dic['p_null'][par]=0.
            fit_dic['p_null']['cont']=1.
        com_joint_postproc(p_final,fixed_args,fit_dic,merged_chain,gen_dic)

    print('     ----------------------------------')    
  
    return None






def FIT_joined_IntrProp(param,x_tab,args=None):
    r"""**Fit function: joined local stellar property**

    Calls corresponding model function for optimization

    Args:
        TBD
    
    Returns:
        TBD
    """
    return joined_IntrProp(param,args)[0],None




def joined_IntrProp(param,args):
    r"""**Model function: joined stellar property**

    Defines the joined model for stellar properties

    Args:
        TBD
    
    Returns:
        TBD
    
    """   
    mod_tab=np.zeros(0,dtype=float)
    mod_prop_dic = {}
    mod_coord_dic = {}
    coeff_line_dic = {}
    
    #Update stellar grid
    if args['var_star_grid']:up_model_star(args, param)
    
    #Process each visit
    for inst in args['inst_list']:
        args['inst']=inst
        mod_prop_dic[inst]={}
        mod_coord_dic[inst]={}
        coeff_line_dic[inst]={}
        for vis in args['inst_vis_list'][inst]: 
            args['vis']=vis 
            
            #Calculate coordinates and properties of occulted regions 
            system_param_loc,coord_pl,param_val = up_plocc_arocc_prop(inst,vis,args,param,args['studied_pl'][inst][vis],args['ph_fit'][inst][vis],args['coord_fit'][inst][vis])
            surf_prop_dic,_,_ = sub_calc_plocc_ar_prop([args['chrom_mode']],args,args['par_list'],args['studied_pl'][inst][vis],[],system_param_loc,args['grid_dic'],args['system_prop'],param_val,coord_pl,range(args['nexp_fit_all'][inst][vis]))

            #Properties associated with the transiting planet in the visit 
            pl_vis = args['studied_pl'][inst][vis][0]
            theo_vis = surf_prop_dic['achrom'][pl_vis]      
            
            #Fit coordinate
            #    - only used for plots
            if (not args['fit']):
                if args['prop_fit']=='rv':
                    mod_coord_dic[inst][vis] = args['ph_fit'][inst][vis]
                    coeff_line_dic[inst][vis] = None
                elif ('coeff_line' in args):
                    mod_coord_dic[inst][vis] = theo_vis[args['coord_line']]
                    coeff_line_dic[inst][vis] = args['coeff_line']
            else:
                mod_coord_dic[inst][vis] = None
                coeff_line_dic[inst][vis] = None

            #Model property for the visit 
            mod_prop_dic[inst][vis] = theo_vis[args['prop_fit']][0] 
         
            #Appending over all visits
            mod_tab=np.append(mod_tab,mod_prop_dic[inst][vis])

    return mod_tab,coeff_line_dic,mod_prop_dic,mod_coord_dic
    


















def main_joined_IntrProf(rout_mode,data_dic,gen_dic,system_param,fit_prop_dic,theo_dic,plot_dic,coord_dic):
    r"""**Joined stellar profile fits**

    Main routine to fit intrinsic stellar profiles from planet-occulted regions with a joined model over instruments and visits.

    Profile description
    
     - We use analytical models, measured profiles, or theoretical models to describe the intrinsic profiles.
     - Positions of the profiles along the transit chord are linked by the stellar surface RV model.
     - Shapes of the profiles, when analytical, are linked across the transit chord by polynomial laws as a function of a chosen dimension.
       Polynomial coefficients can depend on the visit and their associated instrument, to account for possible variations in the line shape between visits
     - Stellar line profiles are defined before instrumental convolution, so that data from all instruments and visits can be fitted together
    
    Beware that the intrinsic and disk-integrated profiles have the same continuum, but that it is not necessarily unity
    Thus the continuum of analytical and theoretical model profiles must be let free to vary

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    print('   > Fitting joined intrinsic profiles')

    #Initializations
    fixed_args,fit_dic = init_joined_routines(rout_mode,gen_dic,system_param,theo_dic,data_dic,fit_prop_dic)

    ######################################################################################################## 

    #Arguments to be passed to the fit function
    fixed_args.update({ 
        'model':fit_prop_dic['model'],
        'mode':fit_prop_dic['mode'],
        'cen_bins':{},
        'edge_bins':{},
        'dcen_bins':{},
        'dim_exp':{},
        'ncen_bins':{},
        'cond_fit':{},
        'cond_def_cont_all':{},
        'flux_cont_all':{},
        'cond_def':{},
        'flux':{},
        'cov':{},
        'nexp_fit':0,
        'FWHM_inst':{},
        'n_pc':{},
        'chrom_mode':data_dic['DI']['system_prop']['chrom_mode'],
        'conv2intr':True,
        'mac_mode':theo_dic['mac_mode'],
        })
    if len(fit_prop_dic['PC_model'])>0:
        fixed_args.update({
            'eig_res_matr':{},
            'nx_fit_PCout':0.,
            'n_free_PCout':0.,
            'chi2_PCout':0.,
            })
    fit_save={'idx_trim_kept':{}}
        
    #Stellar surface coordinate required to calculate spectral line profiles
    #    - other required properties are automatically added in the sub_calc_plocc_ar_prop() function
    fixed_args['par_list']+=['line_prof']
    if fixed_args['mode']=='ana':
        if fit_prop_dic['coord_fit'] in ['abs_y_st','y_st2']:fixed_args['coord_line']='y_st'    
        elif fit_prop_dic['coord_fit']!='phase':fixed_args['coord_line']=fit_prop_dic['coord_fit']
        else:stop('ERROR : coordinate '+fit_prop_dic['coord_fit']+' not usable for line profiles')
        fixed_args['par_list']+=[fixed_args['coord_line']]
    else:
        fixed_args['par_list']+=['mu']

    #Activation of spectral conversion and resampling 
    cond_conv_st_prof_tab(theo_dic['rv_osamp_line_mod'],fixed_args,data_dic[data_dic['instrum_list'][0]]['type'])                           

    #Construction of the fit tables
    for par in ['coord_obs','coord_fit','ph_fit']:fixed_args[par]={}
    for inst in np.intersect1d(data_dic['instrum_list'],list(fit_dic['idx_in_fit'].keys())):  
        init_joined_routines_inst(inst,fit_dic,fixed_args)
        for key in ['cen_bins','edge_bins','dcen_bins','cond_fit','cond_def_cont_all','flux_cont_all','flux','cov','cond_def','n_pc','dim_exp','ncen_bins']:fixed_args[key][inst]={}
        if len(fit_prop_dic['PC_model'])>0:fixed_args['eig_res_matr'][inst]={}
        fit_save['idx_trim_kept'][inst] = {}
        if (fixed_args['mode']=='ana') and (inst not in fixed_args['model']):fixed_args['model'][inst] = 'gauss'
        if (inst in fit_prop_dic['fit_order']):iord_sel =  fit_prop_dic['fit_order'][inst]
        else:iord_sel = 0
        
        #Setting continuum range to default if undefined
        if inst not in fit_prop_dic['cont_range']:fit_prop_dic['cont_range'] = data_dic['Intr']['cont_range']
        cont_range = fit_prop_dic['cont_range'][inst][iord_sel]

        #Setting fitted range to default if undefined
        if inst not in fit_prop_dic['fit_range']:fit_prop_dic['fit_range'][inst] = data_dic['Intr']['fit_range'][inst]
        
        #Setting trimmed range to default if undefined
        if (inst in fit_prop_dic['trim_range']):trim_range = fit_prop_dic['trim_range'][inst]
        else:trim_range = None 
        
        #Processing visit
        for vis in data_dic[inst]['visit_list']:
            init_joined_routines_vis(inst,vis,fit_dic,fixed_args)
            
            #Visit is fitted
            if vis in fixed_args['inst_vis_list'][inst]:   
                data_vis=data_dic[inst][vis]
                init_joined_routines_vis_fit('IntrProf',inst,vis,fit_dic,fixed_args,data_vis,gen_dic,data_dic,coord_dic,theo_dic,None,None)

                #Enable PC noise model
                if (inst in fit_prop_dic['PC_model']) and (vis in fit_prop_dic['PC_model'][inst]):
                    if fixed_args['bin_mode'][inst][vis]=='_bin':stop('PC correction not available for binned data')
                    data_pca = dataload_npz(fit_prop_dic['PC_model'][inst][vis]['PC_path'])

                    #Accounting for PCA fit in merit values
                    #    - PC have been fitted in the PCA module to the out-of-transit data
                    if len(fit_prop_dic['PC_model'][inst][vis]['idx_out'])>0:
                        if fit_prop_dic['PC_model'][inst][vis]['idx_out']=='all':idx_pc_out =  np.intersect1d(gen_dic[inst][vis]['idx_out'],data_pca['idx_corr'])
                        else:idx_pc_out =  np.intersect1d(fit_prop_dic['PC_model'][inst][vis]['idx_out'],data_pca['idx_corr'])
                    
                        #Increment :
                        #    + number of fitted datapoints = number of pixels fitted per exposure in the PCA module
                        #    + number of free parameters = number of PC fitted in the PCA module
                        #    + chi2: chi2 for current exposure from the correction applied in the PCA module
                        for idx_out in idx_pc_out:
                            fixed_args['nx_fit_PCout']+=data_pca['nfit_tab'][idx_out]
                            if fit_prop_dic['PC_model'][inst][vis]['noPC']:
                                fixed_args['n_free_PCout']+=0. 
                                fixed_args['chi2_PCout']+=data_pca['chi2null_tab'][idx_out]                                   
                            else:
                                fixed_args['n_free_PCout']+=data_pca['n_pc']
                                fixed_args['chi2_PCout']+=data_pca['chi2_tab'][idx_out] 
                                
                    #Number of PC used for the fit
                    #    - set by the correction applied in the PCA module, for consistency
                    if fit_prop_dic['PC_model'][inst][vis]['noPC']:
                        fixed_args['n_pc'][inst][vis] = None
                    else:
                        fixed_args['n_pc'][inst][vis]=data_pca['n_pc']
                        fixed_args['eig_res_matr'][inst][vis] = {} 

                else:fixed_args['n_pc'][inst][vis] = None

                #Fit tables
                #    - models must be calculated over the full, continuous spectral tables to allow for convolution
                #      the fit is then performed on defined pixels only
                for key in ['dcen_bins','cen_bins','edge_bins','flux','cov','cond_def']:fixed_args[key][inst][vis]=np.zeros(fixed_args['nexp_fit_all'][inst][vis],dtype=object)
                for isub,i_in in enumerate(fixed_args['idx_in_fit'][inst][vis]):
              
                    #Upload latest processed intrinsic data
                    if fixed_args['bin_mode'][inst][vis]=='_bin':data_exp = dataload_npz(gen_dic['save_data_dir']+'Intrbin_data/'+inst+'_'+vis+'_phase'+str(i_in))               
                    else:data_exp = dataload_npz(data_dic[inst][vis]['proc_Intr_data_paths']+str(i_in))
                    
                    #Initialization
                    if isub==0:

                        #Instrumental convolution
                        if (inst not in fixed_args['FWHM_inst']):                
                            fixed_args['FWHM_inst'][inst] = get_FWHM_inst(inst,fixed_args,data_exp['cen_bins'][iord_sel])
                                                
                        #Trimming data
                        #    - the trimming is defined using the first table and not each individual table, so that all processed profiles keep the same dimension after trimming
                        #    - data is trimmed to the minimum range encompassing the continuum to limit useless computations
                        if (trim_range is not None):
                            min_trim_range = trim_range[0]
                            max_trim_range = trim_range[1]
                        else:
                            if len(cont_range)==0:
                                min_trim_range=-1e100
                                max_trim_range=1e100
                            else:
                                min_trim_range = cont_range[0][0] + 3.*fixed_args['FWHM_inst'][inst]
                                max_trim_range = cont_range[-1][1] - 3.*fixed_args['FWHM_inst'][inst]
                        idx_range_kept = np_where1D((data_exp['edge_bins'][iord_sel,0:-1]>=min_trim_range) & (data_exp['edge_bins'][iord_sel,1::]<=max_trim_range))
                        ncen_bins = len(idx_range_kept)
                        if ncen_bins==0:stop('Empty trimmed range')                  
                        
                        fit_save['idx_trim_kept'][inst][vis] = idx_range_kept
                        fixed_args['ncen_bins'][inst][vis] = ncen_bins  
                        fixed_args['dim_exp'][inst][vis] = [1,ncen_bins] 
                        fixed_args['cond_fit'][inst][vis]=np.zeros([fixed_args['nexp_fit_all'][inst][vis],ncen_bins],dtype=bool)
                        fixed_args['cond_def_cont_all'][inst][vis] = np.zeros([fixed_args['nexp_fit_all'][inst][vis],ncen_bins],dtype=bool)                      
                    
                    #Trimming profile         
                    for key in ['cen_bins','flux','cond_def']:fixed_args[key][inst][vis][isub] = data_exp[key][iord_sel,idx_range_kept]
                    fixed_args['edge_bins'][inst][vis][isub] = data_exp['edge_bins'][iord_sel,idx_range_kept[0]:idx_range_kept[-1]+2]   
                    fixed_args['dcen_bins'][inst][vis][isub] = fixed_args['edge_bins'][inst][vis][isub][1::]-fixed_args['edge_bins'][inst][vis][isub][0:-1]  
                    fixed_args['cov'][inst][vis][isub] = data_exp['cov'][iord_sel][:,idx_range_kept]  

                    #Oversampled line profile model table
                    if fixed_args['resamp']:resamp_st_prof_tab(inst,vis,isub,fixed_args,gen_dic,fixed_args['nexp_fit_all'][inst][vis],theo_dic['rv_osamp_line_mod'])
               
                    #Initializing ranges in the relevant rest frame
                    if len(cont_range)==0:fixed_args['cond_def_cont_all'][inst][vis][isub] = True    
                    else:
                        for bd_int in cont_range:fixed_args['cond_def_cont_all'][inst][vis][isub] |= (fixed_args['edge_bins'][inst][vis][isub][0:-1]>=bd_int[0]) & (fixed_args['edge_bins'][inst][vis][isub][1:]<=bd_int[1])         
                    if len(fit_prop_dic['fit_range'][inst][vis])==0:fixed_args['cond_fit'][inst][vis][isub] = True    
                    else:
                        for bd_int in fit_prop_dic['fit_range'][inst][vis]:
                            fixed_args['cond_fit'][inst][vis][isub] |= (fixed_args['edge_bins'][inst][vis][isub][0:-1]>=bd_int[0]) & (fixed_args['edge_bins'][inst][vis][isub][1:]<=bd_int[1])        

                    #Accounting for undefined pixels
                    fixed_args['cond_def_cont_all'][inst][vis][isub] &= fixed_args['cond_def'][inst][vis][isub]           
                    fixed_args['cond_fit'][inst][vis][isub] &= fixed_args['cond_def'][inst][vis][isub]          
                    fit_dic['nx_fit']+=np.sum(fixed_args['cond_fit'][inst][vis][isub])
                    
                    #Setting constant error
                    if fixed_args['cst_err']:
                        var_loc = fixed_args['cov'][inst][vis][isub][0,fixed_args['cond_def_cont_all'][inst][vis][isub]]
                        fixed_args['cov'][inst][vis][isub] = np.tile(np.mean(var_loc),[1,ncen_bins])
             
                    #Scaling covariance matrix
                    fixed_args['cov'][inst][vis][isub]*=fixed_args['sc_var']

                    #Initialize PCs 
                    if fixed_args['n_pc'][inst][vis] is not None:
                    
                        #PC matrix interpolated on current exposure table
                        fixed_args['eig_res_matr'][inst][vis][i_in] = np.zeros([fixed_args['n_pc'][inst][vis],fixed_args['ncen_bins'][inst][vis]],dtype=float)
                    
                        #Process each PC
                        for i_pc in range(fixed_args['n_pc'][inst][vis]):
                            
                            #PC profile
                            fixed_args['eig_res_matr'][inst][vis][i_in][i_pc] = interp1d(data_pca['cen_bins'],data_pca['eig_res_matr'][i_pc],fill_value='extrapolate')(fixed_args['cen_bins'][inst][vis][isub])       
                            
                            #PC free amplitude
                            pc_name = 'aPC_idxin'+str(i_in)+'_ord'+str(i_pc)+'__IS'+inst+'_VS'+vis
                            fit_dic['mod_prop'][pc_name]={'vary':True,'guess':0.} 
                            if i_pc==0:fit_dic['mod_prop'][pc_name]['bd'] = [-4.,4.]
                            elif i_pc==1:fit_dic['mod_prop'][pc_name]['bd'] = [-3.,2.]
                            else:fit_dic['mod_prop'][pc_name]['bd'] = [-1.,1.]
                            fit_dic['priors'][pc_name]={'low':-100. ,'high':100.,'mod':'uf'}

                #Number of fitted exposures
                fixed_args['nexp_fit']+=fixed_args['nexp_fit_all'][inst][vis]

    #Active region crossing time supplement
    #    - to avoid fitting timing values on the order of 2400000 we use as fitted property T_ar - <Tvisit> and add back <Tvisit> within the model 
    if fixed_args['cond_studied_ar']:
        fixed_args['bjd_time_shift']={}

    #Final processing
    for idx_inst,inst in enumerate(fixed_args['inst_list']):
        
        #Common data type        
        if idx_inst==0:fixed_args['type'] = data_dic[inst]['type']
        elif fixed_args['type'] != data_dic[inst]['type']:stop('Incompatible data types')
        
        #Active region
        if fixed_args['cond_studied_ar']:
            fixed_args['bjd_time_shift'][inst]={}

        #Visits
        for vis in fixed_args['inst_vis_list'][inst]:
            
            #Defining active region crossing time supplement
            if fixed_args['cond_studied_ar']:
                fixed_args['bjd_time_shift'][inst][vis]=np.floor(fixed_args['coord_fit'][inst][vis]['bjd'][0])+2400000.

            #Continuum common to all processed profiles within visit
            #    - collapsed along temporal axis
            cond_cont_com  = np.all(fixed_args['cond_def_cont_all'][inst][vis],axis=0)
            if np.sum(cond_cont_com)==0.:stop('No pixels in common continuum')  

            #Continuum flux
            #    - calculated over the defined bins common to all processed profiles
            #    - defined as a weighted mean because intrinsic profiles at the limbs can be very poorly defined due to the partial occultation and limb-darkening
            #    - we use the covariance diagonal to define a representative weight
            cont_intr = np.zeros(fixed_args['nexp_fit_all'][inst][vis])*np.nan
            wcont_intr = np.zeros(fixed_args['nexp_fit_all'][inst][vis])*np.nan
            for isub in range(fixed_args['nexp_fit_all'][inst][vis]):
                dw_sum = np.sum(fixed_args['dcen_bins'][inst][vis][isub][cond_cont_com])
                cont_intr[isub] = np.sum(fixed_args['flux'][inst][vis][isub][cond_cont_com]*fixed_args['dcen_bins'][inst][vis][isub][cond_cont_com])/dw_sum
                wcont_intr[isub] = dw_sum**2./np.sum(fixed_args['cov'][inst][vis][isub][0,cond_cont_com]*fixed_args['dcen_bins'][inst][vis][isub][cond_cont_com]**2.)
            fixed_args['flux_cont_all'][inst][vis]=np.nansum(cont_intr*wcont_intr)/np.nansum(wcont_intr)

    #Artificial observation table
    #    - covariance condition is set to False so that chi2 values calculated here are not further modified within the residual() function
    #    - unfitted pixels are removed from the chi2 table passed to residual() , so that they are then summed over the full tables
    if fit_dic['nx_fit']==0:stop('No points in fitted ranges')
    fixed_args['idx_fit'] = np.ones(fit_dic['nx_fit'],dtype=bool)   
    fixed_args['x_val']=range(fit_dic['nx_fit'])
    fixed_args['y_val'] = np.zeros(fit_dic['nx_fit'],dtype=float)  
    fixed_args['s_val'] = np.ones(fit_dic['nx_fit'],dtype=float)          
    fixed_args['cov_val'] = np.array([fixed_args['s_val']**2.])  
    fixed_args['use_cov'] = False
    fixed_args['use_cov_eff'] = gen_dic['use_cov']
    fixed_args['fit_func'] = FIT_joined_IntrProf
    fixed_args['mod_func'] = joined_IntrProf
    fixed_args['inside_fit'] = True    
    
    #Model fit and calculation
    merged_chain,p_final = com_joint_fits('IntrProf',fit_dic,fixed_args,gen_dic,data_dic,theo_dic,fit_dic['mod_prop'])            

    #PC correction
    if len(fit_prop_dic['PC_model'])>0:
        fit_save.update({'eig_res_matr':fixed_args['eig_res_matr'],'n_pc':fixed_args['n_pc'] })
       
        #Accouting for out-of-transit PCA fit in merit values
        #    - PC have been fitted in the PCA module to the out-of-transit data
        #      we include their merit values to those from RMR + PC fit 
        if fixed_args['nx_fit_PCout']>0:
            fit_dic_PCout = {}
            fit_dic_PCout['nx_fit'] = fit_dic['nx_fit'] + fixed_args['nx_fit_PCout']
            fit_dic_PCout['n_free']  = fit_dic['merit']['n_free'] + fixed_args['n_free_PCout']
            fit_dic_PCout['dof'] = fit_dic_PCout['nx_fit'] - fit_dic_PCout['n_free']
            fit_dic_PCout['chi2'] = fit_dic['merit']['chi2'] + fixed_args['chi2_PCout']          
            fit_dic_PCout['red_chi2'] = fit_dic_PCout['chi2']/fit_dic_PCout['dof']
            fit_dic_PCout['BIC']=fit_dic_PCout['chi2']+fit_dic_PCout['n_free']*np.log(fit_dic_PCout['nx_fit'])      
            fit_dic_PCout['AIC']=fit_dic_PCout['chi2']+2.*fit_dic_PCout['n_free']  
            
            np.savetxt(fit_dic['file_save'],[['----------------------------------']],fmt=['%s'])
            np.savetxt(fit_dic['file_save'],[['Merit values with out-of-transit PC fit']],fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['----------------------------------']],fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['Npts : '+str(fit_dic_PCout['nx_fit'])]],delimiter='\t',fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['Nfree : '+str(fit_dic_PCout['n_free'])]],delimiter='\t',fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['d.o.f : '+str(fit_dic_PCout['dof'])]],delimiter='\t',fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['Best chi2 : '+str(fit_dic_PCout['chi2'])]],delimiter='\t',fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['Reduced chi2 : '+str(fit_dic_PCout['red_chi2'])]],delimiter='\t',fmt=['%s'])
            np.savetxt(fit_dic['file_save'],[['BIC : '+str(fit_dic_PCout['BIC'])]],delimiter='\t',fmt=['%s'])
            np.savetxt(fit_dic['file_save'],[['AIC : '+str(fit_dic_PCout['AIC'])]],delimiter='\t',fmt=['%s'])
            np.savetxt(fit_dic['file_save'],[['----------------------------------']],fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['']],fmt=['%s']) 


    #Best-fit model and derived properties
    fixed_args['fit'] = False
    mod_dic,coeff_line_dic,mod_prop_dic,_ = fixed_args['mod_func'](p_final,fixed_args)

    #Save best-fit properties
    #    - with same structure as fit to individual profiles 
    fit_save.update({'p_final':p_final,'coeff_line_dic':coeff_line_dic,'model':fixed_args['model'],'name_prop2input':fixed_args['name_prop2input'],'coord_line':fixed_args['coord_line'],'merit':fit_dic['merit'],
                     'pol_mode':fit_prop_dic['pol_mode'],'coeff_ord2name':fixed_args['coeff_ord2name'],'idx_in_fit':fixed_args['idx_in_fit'],'genpar_instvis':fixed_args['genpar_instvis'],'linevar_par':fixed_args['linevar_par'],
                     'ph_fit':fixed_args['ph_fit'], 'system_prop':fixed_args['system_prop'],'grid_dic':fixed_args['grid_dic'],'var_par_list':fixed_args['var_par_list'], 'fit_orbit':fixed_args['fit_orbit'], 'fit_RpRs':fixed_args['fit_RpRs'],
                     'system_ar_prop':fixed_args['system_ar_prop']})
    if fixed_args['mode']=='ana':fit_save['func_prof'] = fixed_args['func_prof']
    np.savez(fit_dic['save_dir']+'Fit_results',data=fit_save,allow_pickle=True)
    if (plot_dic['Intr_prof']!='') or (plot_dic['Intr_prof_res']!='') or (plot_dic['prop_Intr']!='') or (plot_dic['sp_Intr_1D']!=''):
        for inst in fixed_args['inst_list']:
            for vis in fixed_args['inst_vis_list'][inst]:
                prof_fit_dic={'fit_range':fit_prop_dic['fit_range'][inst][vis]}
                if fixed_args['bin_mode'][inst][vis]=='_bin':prof_fit_dic['loc_prof_est_path'] = gen_dic['save_data_dir']+'Intrbin_data/'+inst+'_'+vis+'_phase'          
                else:prof_fit_dic['loc_prof_est_path'] = data_dic[inst][vis]['proc_Intr_data_paths']
                for isub,i_in in enumerate(fixed_args['idx_in_fit'][inst][vis]):
                    prof_fit_dic[i_in]={
                        'cen_bins':fixed_args['cen_bins'][inst][vis][isub],
                        'flux':mod_dic[inst][vis][isub],
                        'cond_def_fit':fixed_args['cond_fit'][inst][vis][isub],
                        'cond_def_cont':fixed_args['cond_def_cont_all'][inst][vis][isub]
                        }
                    for pl_loc in fixed_args['studied_pl'][inst][vis]:
                        prof_fit_dic[i_in][pl_loc] = {}
                        for prop_loc in mod_prop_dic[inst][vis][pl_loc]:prof_fit_dic[i_in][pl_loc][prop_loc] = mod_prop_dic[inst][vis][pl_loc][prop_loc][isub]
                np.savez_compressed(fit_dic['save_dir']+'IntrProf_fit_'+inst+'_'+vis+fixed_args['bin_mode'][inst][vis],data={'prof_fit_dic':prof_fit_dic},allow_pickle=True)

    #Post-processing    
    fit_dic['p_null'] = deepcopy(p_final)
    for par in [ploc for ploc in fit_dic['p_null'] if 'ctrst' in ploc]:fit_dic['p_null'][par] = 0.    
    com_joint_postproc(p_final,fixed_args,fit_dic,merged_chain,gen_dic)
    print('     ----------------------------------')  
    return None




def FIT_joined_IntrProf(param,x_tab,args=None):
    r"""**Fit function: joined intrinsic stellar profiles**

    Calls corresponding model function for optimization

    Args:
        TBD
    
    Returns:
        TBD
    """
    
    #Models over fitted spectral ranges
    mod_dic=joined_IntrProf(param,args)[0]
   
    #Merit table
    chi = calc_chi_Prof(mod_dic,args)
    
    return chi,None




def joined_IntrProf(param,fixed_args):
    r"""**Model function: joined intrinsic stellar profiles**

    Defines the joined model for intrinsic stellar profiles.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    args = deepcopy(fixed_args)
    mod_dic = {}
    mod_prop_dic = {}
    coeff_line_dic = {}

    #Update stellar grid
    if args['var_star_grid']:up_model_star(args, param)
    
    #Updating theoretical line profile series
    #    - we assume abundance is common to all instruments and visits
    if (args['mode']=='theo') and (args['var_line']):  
        for sp in args['abund_sp']:args['grid_dic']['sme_grid']['abund'][sp]=param['abund_'+sp]
        gen_theo_intr_prof(args['grid_dic']['sme_grid'])

    #Processing instruments
    for inst in args['inst_list']:
        args['inst']=inst
        mod_dic[inst]={}
        coeff_line_dic[inst]={}
        mod_prop_dic[inst]={}

        #Processing visits
        for vis in args['inst_vis_list'][inst]:   
            args['vis']=vis

            #Retrieve updated coordinates of occulted regions or use imported values
            system_param_loc,coord_pl_ar,param_val = up_plocc_arocc_prop(inst,vis,args,param,args['studied_pl'][inst][vis],args['ph_fit'][inst][vis],args['coord_fit'][inst][vis],studied_ar=args['studied_ar'][inst][vis])

            #-----------------------------------------------------------
            #Outputs
            if not args['fit']:outputs_Prof(inst,vis,coeff_line_dic,mod_prop_dic,args,param_val)

            #-----------------------------------------------------------
            #Variable line model for each exposure 
            #    - the intrinsic stellar line profile is convolved with the LSF kernel specific to each instrument
            #    - we assume a flat continuum, set after the PC component so that intrinsic profiles models later on can be defined with a continuum unity
            #-----------------------------------------------------------
            mod_dic[inst][vis]=np.zeros(args['nexp_fit_all'][inst][vis],dtype=object)
            for isub,i_in in enumerate(args['idx_in_fit'][inst][vis]):

                #Table for model calculation
                args_exp = def_st_prof_tab(inst,vis,isub,args)

                #Intrinsic profile for current exposure
                #    - occulted stellar cells (from planet and active regions) are automatically identified within sub_calc_plocc_ar_prop() 
                #    - see joined_DiffProf() for details about active region contribution
                #    - the planet-occulted profile is calculated over both quiet and active region cells
                surf_prop_dic,_,_ = sub_calc_plocc_ar_prop([args['chrom_mode']],args_exp,args['par_list'],args['studied_pl'][inst][vis],args['studied_ar'][inst][vis],system_param_loc,args['grid_dic'],args['system_prop'],param_val,coord_pl_ar,[isub],system_ar_prop_in=args['system_ar_prop'])
                sp_line_model = surf_prop_dic[args['chrom_mode']]['line_prof'][:,0] 

                #Conversion and resampling 
                mod_dic[inst][vis][isub] = conv_st_prof_tab(inst,vis,isub,args,args_exp,sp_line_model,args['FWHM_inst'][inst])

                #Add PC noise model
                #    - added to the convolved profiles since PC are derived from observed data
                if args['n_pc'][inst][vis] is not None:
                    for i_pc in range(args['n_pc'][inst][vis]):mod_dic[inst][vis][isub]+=param_val[args['name_prop2input']['aPC_idxin'+str(i_in)+'_ord'+str(i_pc)+'__IS'+inst+'_VS'+vis]]*args['eig_res_matr'][inst][vis][i_in][i_pc]
               
                #Set to continuum level
                #    - profiles are internally calculated with a continuum unity
                #    - intrinsic profiles have the same continuum level as out-of-transit disk-integrated profiles, but since the observed profile 
                # continuum may not be well defined or has been scaled on a different range than the fitted one, it is difficult to measure its value
                mod_dic[inst][vis][isub]*=param_val['cont']
             
                #Properties of all planet-occulted regions used to calculate spectral line profiles
                if not args['fit']:
                    for pl_loc in args['studied_pl'][inst][vis]:                    
                        for prop_loc in mod_prop_dic[inst][vis][pl_loc]:mod_prop_dic[inst][vis][pl_loc][prop_loc][isub] = surf_prop_dic[args['chrom_mode']][pl_loc][prop_loc][0] 

    return mod_dic,coeff_line_dic,mod_prop_dic,None
























































def main_joined_DiffProf(rout_mode,data_dic,gen_dic,system_param,fit_prop_dic,theo_dic,plot_dic,coord_dic):    
    r"""**Joined differential profiles fits**

    Main routine to fit a given stellar surface property from planet-occulted regions with a joined model over instruments and visits.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    print('   > Fitting joined differential stellar CCFs, including active regions')

    #Initializations
    fixed_args,fit_dic = init_joined_routines(rout_mode,gen_dic,system_param,theo_dic,data_dic,fit_prop_dic)

    #Arguments to be passed to the fit function
    fixed_args.update({
        'ref_pl':fit_prop_dic['ref_pl'],
        'model':fit_prop_dic['model'],
        'mode':fit_prop_dic['mode'],
        'cen_bins' :{},
        'edge_bins':{},
        'dcen_bins' :{},
        'dim_exp':{},
        'ncen_bins':{},
        'cond_fit' :{},
        'cond_def_cont_all':{},
        'flux_cont_all':{},
        'cond_def' :{},
        'flux':{},
        'cov' :{},
        'nexp_fit':0,
        'FWHM_inst':{},
        'n_pc':{},
        'chrom_mode':data_dic['DI']['system_prop']['chrom_mode'],
        'conv2intr':False,
        'mac_mode':theo_dic['mac_mode'],
        })

    if len(fit_prop_dic['PC_model'])>0:
        fixed_args.update({
            'eig_res_matr':{},
            'nx_fit_PCout':0.,
            'n_free_PCout':0.,
            'chi2_PCout':0.,
            })
    fit_save={'idx_trim_kept':{}}

    #Define master-out dictionary
    for key in ['multivisit_list','idx_in_master_out','master_out_tab','scaled_data_paths','sing_gcal','multivisit_weights_total','weights','flux','multivisit_flux','EFsc2','calc_cond']:fixed_args['master_out'][key]={}
    fixed_args['raw_DI_profs']={}

    #Profile generation
    fixed_args['idx_out']={}
    fixed_args['idx_in']={}
    fixed_args['DI_scaling_val']=data_dic['DI']['scaling_val']

    #Stellar surface coordinate required to calculate spectral line profiles
    #    - other required properties are automatically added in the sub_calc_plocc_ar_prop() function
    fixed_args['par_list']+=['line_prof']
    if fixed_args['mode']=='ana':
        if fit_prop_dic['coord_fit'] in ['abs_y_st','y_st2']:fixed_args['coord_line']='y_st'    
        else:fixed_args['coord_line']=fit_prop_dic['coord_fit']
        fixed_args['par_list']+=[fixed_args['coord_line']]
    else:
        fixed_args['par_list']+=['mu']

    #Activation of spectral conversion and resampling 
    cond_conv_st_prof_tab(theo_dic['rv_osamp_line_mod'],fixed_args,data_dic[data_dic['instrum_list'][0]]['type'])
    
    #Construction of the fit tables
    #Initializing entries that will store the coordinates of the planets, of the active regions, and the respective phase of the fitted exposures.
    for par in ['coord_obs','coord_fit','ph_fit']:fixed_args[par]={}
    
    #Initialize variables to store the max and min limits of the fit tables
    low_bound=1e100
    high_bound=-1e100
    num_pts=0

    # Master-out general properties
    # - Common table
    fixed_args['master_out']['master_out_tab']['resamp_mode']=gen_dic['resamp_mode']

    # - Weights
    fixed_args['master_out']['nord']=1
    fixed_args['master_out']['corr_Fbal']=gen_dic['corr_Fbal']
    fixed_args['master_out']['corr_FbalOrd']=gen_dic['corr_FbalOrd']
    fixed_args['master_out']['flux_sc']=gen_dic['flux_sc']
    fixed_args['master_out']['save_data_dir']=gen_dic['save_data_dir']
    for inst in np.intersect1d(data_dic['instrum_list'],list(fit_dic['idx_in_fit'].keys())):    
        init_joined_routines_inst(inst,fit_dic,fixed_args)

        fixed_args['master_out']['idx_in_master_out'][inst]={}
        fixed_args['raw_DI_profs'][inst]={}
        fixed_args['master_out']['calc_cond'][inst]={}
        fixed_args['master_out']['weights'][inst]={}
        fixed_args['master_out']['flux'][inst]={}
        fixed_args['master_out']['scaled_data_paths'][inst]={}
        fixed_args['master_out']['sing_gcal'][inst]={}
        fixed_args['master_out']['EFsc2'][inst]={}
        fixed_args['idx_out'][inst]={}
        fixed_args['idx_in'][inst]={}
        
        if (inst not in fixed_args['ref_pl']) and (fixed_args['ref_pl']!={}):fixed_args['ref_pl'][inst]={}
        for key in ['cen_bins','edge_bins','dcen_bins','cond_fit','cond_def_cont_all','flux_cont_all','flux','cov','cond_def','n_pc','dim_exp','ncen_bins']:fixed_args[key][inst]={}
        if len(fit_prop_dic['PC_model'])>0:fixed_args['eig_res_matr'][inst]={}
        fit_save['idx_trim_kept'][inst] = {}
        if (fixed_args['mode']=='ana') and (inst not in fixed_args['model']):fixed_args['model'][inst] = 'gauss'
        if (inst in fit_prop_dic['fit_order']):iord_sel =  fit_prop_dic['fit_order'][inst]
        else:iord_sel = 0

        #Setting continuum range to default if undefined
        if inst not in fit_prop_dic['cont_range']:fit_prop_dic['cont_range'] = data_dic['Diff']['cont_range']
        cont_range = fit_prop_dic['cont_range'][inst][iord_sel]

        #Setting fitted range to default if undefined
        if inst not in fit_prop_dic['fit_range']:fit_prop_dic['fit_range'][inst] = data_dic['Diff']['fit_range'][inst]
        
        #Setting trimmed range to default if undefined
        if (inst in fit_prop_dic['trim_range']):trim_range = fit_prop_dic['trim_range'][inst]
        else:trim_range = None 

        #Defining visits used in the master-out calculation 
        fixed_args['master_out']['multivisit_list'][inst]=[]
        if (inst in data_dic['Diff']['vis_in_bin']) and (len(data_dic['Diff']['vis_in_bin'][inst])>1):fixed_args['master_out']['multivisit_list'][inst]=data_dic['Diff']['vis_in_bin'][inst]
        for multivisit in fixed_args['master_out']['multivisit_list'][inst]:
            if multivisit not in data_dic[inst]['visit_list']:stop('Problem: '+multivisit+' was selected for master-out calculation but is not used in the fit.')

        #Initializing weight calculation conditions
        calc_EFsc2,calc_var_ref2,calc_flux_sc_all,var_key_def = weights_bin_prof_calc('DI','DI',gen_dic,data_dic,inst)   
        fixed_args['master_out']['calc_cond'][inst] = (calc_EFsc2,calc_var_ref2,calc_flux_sc_all)
    
        #Processing visit
        for vis_index, vis in enumerate(data_dic[inst]['visit_list']):
            init_joined_routines_vis(inst,vis,fit_dic,fixed_args)

            #Visit is fitted
            if vis in fixed_args['inst_vis_list'][inst]:
                data_vis=data_dic[inst][vis]
                init_joined_routines_vis_fit('DiffProf',inst,vis,fit_dic,fixed_args,data_vis,gen_dic,data_dic,coord_dic,theo_dic,None,None)

                #Master-out properties
                fixed_args['raw_DI_profs'][inst][vis]={} 
                fixed_args['master_out']['weights'][inst][vis]={}
                fixed_args['master_out']['idx_in_master_out'][inst][vis]=[]

                #Indexes
                if (inst in data_dic['Diff']['idx_in_bin']) and (vis in data_dic['Diff']['idx_in_bin'][inst]):
                    if data_dic['Diff']['idx_in_bin'][inst][vis]!={}:fixed_args['master_out']['idx_in_master_out'][inst][vis]=list(data_dic['Diff']['idx_in_bin'][inst][vis])
                if len(fixed_args['master_out']['idx_in_master_out'][inst][vis])==0:stop('No exposures defined in visit '+vis+' for the master-out calculation.')                
                fixed_args['master_out']['idx_in_master_out'][inst][vis] = list(np.intersect1d(fixed_args['master_out']['idx_in_master_out'][inst][vis], fixed_args['idx_in_fit'][inst][vis]))

                #Needed for weight calculation
                fixed_args['master_out']['scaled_data_paths'][inst][vis]={}
                if gen_dic['flux_sc'] and calc_flux_sc_all:fixed_args['master_out']['scaled_data_paths'][inst][vis] = data_dic[inst][vis]['scaled_DI_data_paths']
                else:fixed_args['master_out']['scaled_data_paths'][inst][vis] = None
                fixed_args['master_out']['sing_gcal'][inst][vis]={}
                fixed_args['master_out']['EFsc2'][inst][vis]={}

                #Define in and out of transit exposures - needed for profile generation
                fixed_args['idx_out'][inst][vis]=gen_dic[inst][vis]['idx_out']
                fixed_args['idx_in'][inst][vis]=gen_dic[inst][vis]['idx_in']

                #Defining reference planet if left undefined
                if fixed_args['ref_pl']=={}:fixed_args['ref_pl']={inst:{vis:data_dic[inst][vis]['studied_pl'][0]}}
                elif (vis_index==0) and (vis not in fixed_args['ref_pl'][inst]):fixed_args['ref_pl'][inst][vis]=data_dic[inst][vis]['studied_pl'][0]
                elif vis not in fixed_args['ref_pl'][inst]:fixed_args['ref_pl'][inst][vis]=fixed_args['ref_pl'][inst][data_dic[inst]['visit_list'][vis_index-1]]

                #Enable PC noise model
                if (inst in fit_prop_dic['PC_model']) and (vis in fit_prop_dic['PC_model'][inst]):
                    if fixed_args['bin_mode'][inst][vis]=='_bin':stop('PC correction not available for binned data')
                    data_pca = np.load(fit_prop_dic['PC_model'][inst][vis]['PC_path'],allow_pickle=True)['data'].item()  

                    #Accounting for PCA fit in merit values
                    #    - PC have been fitted in the PCA module to the out-of-transit data
                    if len(fit_prop_dic['PC_model'][inst][vis]['idx_out'])>0:
                        if fit_prop_dic['PC_model'][inst][vis]['idx_out']=='all':idx_pc_out =  np.intersect1d(gen_dic[inst][vis]['idx_out'],data_pca['idx_corr'])
                        else:idx_pc_out =  np.intersect1d(fit_prop_dic['PC_model'][inst][vis]['idx_out'],data_pca['idx_corr'])

                        #Increment :
                        #    + number of fitted datapoints = number of pixels fitted per exposure in the PCA module
                        #    + number of free parameters = number of PC fitted in the PCA module
                        #    + chi2: chi2 for current exposure from the correction applied in the PCA module
                        for idx_out in idx_pc_out:
                            fixed_args['nx_fit_PCout']+=data_pca['nfit_tab'][idx_out]
                            if fit_prop_dic['PC_model'][inst][vis]['noPC']:
                                fixed_args['n_free_PCout']+=0. 
                                fixed_args['chi2_PCout']+=data_pca['chi2null_tab'][idx_out]                                   
                            else:
                                fixed_args['n_free_PCout']+=data_pca['n_pc']
                                fixed_args['chi2_PCout']+=data_pca['chi2_tab'][idx_out] 
                                
                    #Number of PC used for the fit
                    #    - set by the correction applied in the PCA module, for consistency
                    if fit_prop_dic['PC_model'][inst][vis]['noPC']:
                        fixed_args['n_pc'][inst][vis] = None
                    else:
                        fixed_args['n_pc'][inst][vis]=data_pca['n_pc']
                        fixed_args['eig_res_matr'][inst][vis] = {} 

                else:fixed_args['n_pc'][inst][vis] = None

                #Fit tables
                #    - models must be calculated over the full, continuous spectral tables to allow for convolution
                #      the fit is then performed on defined pixels only
                for key in ['dcen_bins','cen_bins','edge_bins','flux','cov','cond_def']:fixed_args[key][inst][vis]=np.zeros(fixed_args['nexp_fit_all'][inst][vis],dtype=object)
                if (data_dic[inst][vis]['type']=='spec2D') and calc_EFsc2 and ('sing_gcal_DI_data_paths' not in data_dic[inst][vis]):stop('ERROR : weighing calibration profiles undefined; make sure you activate gen_dic["calc_proc_data"] and gen_dic["calc_gcal"] when running this module.')  
                for isub,iexp in enumerate(fixed_args['idx_in_fit'][inst][vis]):

                    #Upload latest processed differential data
                    if fixed_args['bin_mode'][inst][vis]=='_bin':data_exp = dataload_npz(gen_dic['save_data_dir']+'Diffbin_data/'+inst+'_'+vis+'_phase'+str(iexp))               
                    else:data_exp = dataload_npz(data_dic[inst][vis]['proc_Diff_data_paths']+str(iexp))

                    #Initialization
                    if isub==0:
     
                        #Instrumental convolution
                        if (inst not in fixed_args['FWHM_inst']):                
                            fixed_args['FWHM_inst'][inst] = get_FWHM_inst(inst,fixed_args,data_exp['cen_bins'][iord_sel])

                        #Trimming data
                        #    - the trimming is applied to the common table, so that all processed profiles keep the same dimension after trimming
                        #    - data is trimmed to the minimum range encompassing the continuum to limit useless computations
                        if (trim_range is not None):
                            min_trim_range = trim_range[0]
                            max_trim_range = trim_range[1]
                        else:
                            if len(cont_range)==0:
                                min_trim_range=-1e100
                                max_trim_range=1e100
                            else:
                                min_trim_range = cont_range[0][0] + 3.*fixed_args['FWHM_inst'][inst]
                                max_trim_range = cont_range[-1][1] - 3.*fixed_args['FWHM_inst'][inst]
                        idx_range_kept = np_where1D((data_exp['edge_bins'][iord_sel,0:-1]>=min_trim_range) & (data_exp['edge_bins'][iord_sel,1::]<=max_trim_range))
                        ncen_bins = len(idx_range_kept)
                        if ncen_bins==0:stop('Empty trimmed range') 
        
                        fit_save['idx_trim_kept'][inst][vis] = idx_range_kept
                        fixed_args['ncen_bins'][inst][vis] = ncen_bins
                        fixed_args['dim_exp'][inst][vis] = [1,ncen_bins]
                        fixed_args['cond_fit'][inst][vis]=np.zeros([fixed_args['nexp_fit_all'][inst][vis],ncen_bins],dtype=bool)
                        fixed_args['cond_def_cont_all'][inst][vis] = np.zeros([fixed_args['nexp_fit_all'][inst][vis],ncen_bins],dtype=bool)  

                    #Trimming profile         
                    for key in ['cen_bins','flux','cond_def']:fixed_args[key][inst][vis][isub] = data_exp[key][iord_sel,idx_range_kept]
                    fixed_args['edge_bins'][inst][vis][isub] = data_exp['edge_bins'][iord_sel,idx_range_kept[0]:idx_range_kept[-1]+2]   
                    fixed_args['dcen_bins'][inst][vis][isub] = fixed_args['edge_bins'][inst][vis][isub][1::]-fixed_args['edge_bins'][inst][vis][isub][0:-1]  
                    fixed_args['cov'][inst][vis][isub] = data_exp['cov'][iord_sel][:,idx_range_kept]

                    #Calibration profile for weighing    
                    if (data_dic[inst][vis]['type']=='spec2D') and calc_EFsc2:fixed_args['master_out']['sing_gcal'][inst][vis][isub] = dataload_npz(data_dic[inst][vis]['sing_gcal_DI_data_paths'][iexp])['sing_gcal'][iord_sel,idx_range_kept]
                    else:fixed_args['master_out']['sing_gcal'][inst][vis][isub] = None
                    
                    #Estimate of true variance for DI profiles
                    #    - relevant (and defined) if 2D profiles were converted into 1D
                    if var_key_def=='EFsc2':fixed_args['master_out']['EFsc2'][inst][vis][isub] = dataload_npz(data_dic[inst][vis]['EFsc2_DI_data_paths'][iexp])['var'][iord_sel,idx_range_kept]  
                    else:fixed_args['master_out']['EFsc2'][inst][vis][isub] = None                    

                    #Oversampled line profile model table
                    if fixed_args['resamp']:resamp_st_prof_tab(inst,vis,isub,fixed_args,gen_dic,fixed_args['nexp_fit_all'][inst][vis],theo_dic['rv_osamp_line_mod'])

                    #Initializing ranges in the relevant rest frame
                    if len(cont_range)==0:fixed_args['cond_def_cont_all'][inst][vis][isub] = True    
                    else:
                        for bd_int in cont_range:fixed_args['cond_def_cont_all'][inst][vis][isub] |= (fixed_args['edge_bins'][inst][vis][isub][0:-1]>=bd_int[0]) & (fixed_args['edge_bins'][inst][vis][isub][1:]<=bd_int[1])         
                    if len(fit_prop_dic['fit_range'][inst][vis])==0:fixed_args['cond_fit'][inst][vis][isub] = True  
                    else:
                        for bd_int in fit_prop_dic['fit_range'][inst][vis]:
                            fixed_args['cond_fit'][inst][vis][isub] |= (fixed_args['edge_bins'][inst][vis][isub][0:-1]>=bd_int[0]) & (fixed_args['edge_bins'][inst][vis][isub][1:]<=bd_int[1])

                    #Accounting for undefined pixels
                    fixed_args['cond_def_cont_all'][inst][vis][isub] &= fixed_args['cond_def'][inst][vis][isub]           
                    fixed_args['cond_fit'][inst][vis][isub]&= fixed_args['cond_def'][inst][vis][isub]          
                    fit_dic['nx_fit']+=np.sum(fixed_args['cond_fit'][inst][vis][isub])

                    #Setting constant error
                    if fixed_args['cst_err']:
                        var_loc = fixed_args['cov'][inst][vis][isub][0,fixed_args['cond_def_cont_all'][inst][vis][isub]]
                        fixed_args['cov'][inst][vis][isub] = np.tile(np.mean(var_loc),[1,ncen_bins])

                    #Scaling variance
                    fixed_args['cov'][inst][vis][isub] *= fixed_args['sc_var']
                    
                    #Initialize PCs 
                    if fixed_args['n_pc'][inst][vis] is not None:
                    
                        #PC matrix interpolated on current exposure table
                        fixed_args['eig_res_matr'][inst][vis][iexp] = np.zeros([fixed_args['n_pc'][inst][vis],fixed_args['ncen_bins'][inst][vis]],dtype=float)
                    
                        #Process each PC
                        for i_pc in range(fixed_args['n_pc'][inst][vis]):
                            
                            #PC profile
                            fixed_args['eig_res_matr'][inst][vis][iexp][i_pc] = interp1d(data_pca['cen_bins'],data_pca['eig_res_matr'][i_pc],fill_value='extrapolate')(fixed_args['cen_bins'][inst][vis][isub])       
                            
                            #PC free amplitude
                            pc_name = 'aPC_idxin'+str(iexp)+'_ord'+str(i_pc)+'__IS'+inst+'_VS'+vis
                            fit_dic['mod_prop'][pc_name]={'vary':True,'guess':0.} 
                            if i_pc==0:fit_dic['mod_prop'][pc_name]['bd'] = [-4.,4.]
                            elif i_pc==1:fit_dic['mod_prop'][pc_name]['bd'] = [-3.,2.]
                            else:fit_dic['mod_prop'][pc_name]['bd'] = [-1.,1.]
                            fit_dic['priors'][pc_name]={'low':-100. ,'high':100.,'mod':'uf'}

                    #Updating the limits of the fit tables
                    low_bound=min(low_bound,np.min(fixed_args['edge_bins'][inst][vis][isub]))
                    high_bound=max(high_bound,np.max(fixed_args['edge_bins'][inst][vis][isub]))
                    num_pts=max(num_pts,len(fixed_args['edge_bins'][inst][vis][isub]))

                #Number of fitted exposures
                fixed_args['nexp_fit']+=fixed_args['nexp_fit_all'][inst][vis]

    if fit_prop_dic['master_out_tab']!=[]:
        if len(fit_prop_dic['master_out_tab'])!=3:stop('Incorrect master-out table format. The format should be [low_bd, up_bd, num_pts].')
        else:fixed_args['master_out']['master_out_tab']['edge_bins']=np.linspace(fit_prop_dic['master_out_tab'][0],fit_prop_dic['master_out_tab'][1],num=fit_prop_dic['master_out_tab'][2])
    else:fixed_args['master_out']['master_out_tab']['edge_bins']=np.linspace(low_bound,high_bound,num=num_pts+1)
    fixed_args['master_out']['master_out_tab']['dcen_bins']=fixed_args['master_out']['master_out_tab']['edge_bins'][1::]-fixed_args['master_out']['master_out_tab']['edge_bins'][0:-1]
    fixed_args['master_out']['master_out_tab']['cen_bins']=fixed_args['master_out']['master_out_tab']['edge_bins'][:-1]+(fixed_args['master_out']['master_out_tab']['dcen_bins']/2)

    #Active regions crossing time supplement
    if fixed_args['cond_studied_ar']:
        fixed_args['bjd_time_shift']={}

    #Final processing
    for idx_inst,inst in enumerate(fixed_args['inst_list']):

        #Common data type   
        if idx_inst==0:fixed_args['type'] = data_dic[inst]['type']
        elif fixed_args['type'] != data_dic[inst]['type']:stop('Incompatible data types')

        #Active region
        if fixed_args['cond_studied_ar']:
            fixed_args['bjd_time_shift'][inst]={}
        
        #Defining multi-visit master-out and weights
        if len(fixed_args['master_out']['multivisit_list'][inst])>0:
            fixed_args['master_out']['multivisit_flux'][inst]=np.zeros(len(fixed_args['master_out']['master_out_tab']['cen_bins']), dtype=float)
            fixed_args['master_out']['multivisit_weights_total'][inst]=np.zeros(len(fixed_args['master_out']['master_out_tab']['cen_bins']), dtype=float)
        
        #Visits
        for vis in fixed_args['inst_vis_list'][inst]:
            
            #Defining active region crossing time supplement
            if fixed_args['cond_studied_ar']:
                fixed_args['bjd_time_shift'][inst][vis]=np.floor(fixed_args['coord_fit'][inst][vis]['bjd'][0])+2400000.

            #Defining flux table
            fixed_args['master_out']['flux'][inst][vis]=np.zeros([len(fixed_args['master_out']['master_out_tab']['cen_bins'])], dtype=float)

            #Continuum common to all processed profiles within visit
            #    - collapsed along temporal axis
            cond_cont_com  = np.all(fixed_args['cond_def_cont_all'][inst][vis],axis=0)
            if np.sum(cond_cont_com)==0.:stop('No pixels in common continuum')  

            #Continuum flux
            #    - calculated over the defined bins common to all processed profiles
            #    - defined as a weighted mean because intrinsic profiles at the limbs can be very poorly defined due to the partial occultation and limb-darkening
            #    - we use the covariance diagonal to define a representative weight
            cont_intr = np.zeros(fixed_args['nexp_fit_all'][inst][vis])*np.nan
            wcont_intr = np.zeros(fixed_args['nexp_fit_all'][inst][vis])*np.nan
            for isub in range(fixed_args['nexp_fit_all'][inst][vis]):
                dw_sum = np.sum(fixed_args['dcen_bins'][inst][vis][isub][cond_cont_com])
                cont_intr[isub] = np.sum(fixed_args['flux'][inst][vis][isub][cond_cont_com]*fixed_args['dcen_bins'][inst][vis][isub][cond_cont_com])/dw_sum
                wcont_intr[isub] = dw_sum**2./np.sum(fixed_args['cov'][inst][vis][isub][0,cond_cont_com]*fixed_args['dcen_bins'][inst][vis][isub][cond_cont_com]**2.)
            fixed_args['flux_cont_all'][inst][vis]=np.nansum(cont_intr*wcont_intr)/np.nansum(wcont_intr)

    #Artificial observation table
    #    - covariance condition is set to False so that chi2 values calculated here are not further modified within the residual() function
    #    - unfitted pixels are removed from the chi2 table passed to residual() , so that they are then summed over the full tables
    if fit_dic['nx_fit']==0:stop('No points in fitted ranges')
    fixed_args['idx_fit'] = np.ones(fit_dic['nx_fit'],dtype=bool)  
    fixed_args['x_val']=range(fit_dic['nx_fit'])
    fixed_args['y_val'] = np.zeros(fit_dic['nx_fit'],dtype=float)  
    fixed_args['s_val'] = np.ones(fit_dic['nx_fit'],dtype=float)          
    fixed_args['cov_val'] = np.array([fixed_args['s_val']**2.])  
    fixed_args['use_cov'] = False
    fixed_args['use_cov_eff'] = gen_dic['use_cov']
    fixed_args['fit_func'] = FIT_joined_DiffProf
    fixed_args['mod_func'] = joined_DiffProf
    fixed_args['inside_fit'] = True 

    #Model fit and calculation
    merged_chain,p_final = com_joint_fits('DiffProf',fit_dic,fixed_args,gen_dic,data_dic,theo_dic,fit_dic['mod_prop'])

    #PC correction
    if len(fit_prop_dic['PC_model'])>0:
        fit_save.update({'eig_res_matr':fixed_args['eig_res_matr'],'n_pc':fixed_args['n_pc'] })
       
        #Accouting for out-of-transit PCA fit in merit values
        #    - PC have been fitted in the PCA module to the out-of-transit data
        #      we include their merit values to those from RMR + PC fit 
        if fixed_args['nx_fit_PCout']>0:
            fit_dic_PCout = {}
            fit_dic_PCout['nx_fit'] = fit_dic['nx_fit'] + fixed_args['nx_fit_PCout']
            fit_dic_PCout['n_free']  = fit_dic['merit']['n_free'] + fixed_args['n_free_PCout']
            fit_dic_PCout['dof'] = fit_dic_PCout['nx_fit'] - fit_dic_PCout['n_free']
            fit_dic_PCout['chi2'] = fit_dic['merit']['chi2'] + fixed_args['chi2_PCout']          
            fit_dic_PCout['red_chi2'] = fit_dic_PCout['chi2']/fit_dic_PCout['dof']
            fit_dic_PCout['BIC']=fit_dic_PCout['chi2']+fit_dic_PCout['n_free']*np.log(fit_dic_PCout['nx_fit'])      
            fit_dic_PCout['AIC']=fit_dic_PCout['chi2']+2.*fit_dic_PCout['n_free']  
            
            np.savetxt(fit_dic['file_save'],[['----------------------------------']],fmt=['%s'])
            np.savetxt(fit_dic['file_save'],[['Merit values with out-of-transit PC fit']],fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['----------------------------------']],fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['Npts : '+str(fit_dic_PCout['nx_fit'])]],delimiter='\t',fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['Nfree : '+str(fit_dic_PCout['n_free'])]],delimiter='\t',fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['d.o.f : '+str(fit_dic_PCout['dof'])]],delimiter='\t',fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['Best chi2 : '+str(fit_dic_PCout['chi2'])]],delimiter='\t',fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['Reduced chi2 : '+str(fit_dic_PCout['red_chi2'])]],delimiter='\t',fmt=['%s'])
            np.savetxt(fit_dic['file_save'],[['BIC : '+str(fit_dic_PCout['BIC'])]],delimiter='\t',fmt=['%s'])
            np.savetxt(fit_dic['file_save'],[['AIC : '+str(fit_dic_PCout['AIC'])]],delimiter='\t',fmt=['%s'])
            np.savetxt(fit_dic['file_save'],[['----------------------------------']],fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['']],fmt=['%s']) 

    #Best-fit model and derived properties
    fixed_args['fit'] = False

    #Evaluate best-fit model
    _,coeff_line_dic,_ = fixed_args['mod_func'](p_final,fixed_args)

    #Save best-fit properties
    #    - with same structure as fit to individual profiles 
    fit_save.update({'p_final':p_final,'coeff_line_dic':coeff_line_dic,'model':fixed_args['model'],'name_prop2input':fixed_args['name_prop2input'],'coord_line':fixed_args['coord_line'],'merit':fit_dic['merit'],
                     'pol_mode':fit_prop_dic['pol_mode'],'coeff_ord2name':fixed_args['coeff_ord2name'],'idx_in_fit':fixed_args['idx_in_fit'],'genpar_instvis':fixed_args['genpar_instvis'],'linevar_par':fixed_args['linevar_par'],
                     'ph_fit':fixed_args['ph_fit'], 'system_prop':fixed_args['system_prop'], 'system_ar_prop':fixed_args['system_ar_prop'], 'grid_dic':fixed_args['grid_dic'],
                     'var_par_list':fixed_args['var_par_list'],'fit_orbit':fixed_args['fit_orbit'], 'fit_RpRs':fixed_args['fit_RpRs'], 'fit_star_ar':fixed_args['fit_star_ar'], 'fit_star_pl':fixed_args['fit_star_pl'],
                     'master_out':fixed_args['master_out'], 'unthreaded_op':fixed_args['unthreaded_op'], 'ref_pl':fixed_args['ref_pl'], 'fit_order':fit_prop_dic['fit_order'], 'fit_mode':fit_prop_dic['fit_mode'],
                     'fit_ar':fixed_args['fit_ar'], 'fit_ar_ang':fixed_args['fit_ar_ang'], 'chi2_storage':fixed_args['chi2_storage']})
    if fixed_args['mode']=='ana':fit_save['func_prof'] = fixed_args['func_prof']
    if fit_prop_dic['fit_mode']=='chi2':fit_save['hess_matrix'] = fixed_args['hess_matrix']
    np.savez(fit_dic['save_dir']+'Fit_results',data=fit_save,allow_pickle=True)

    #Post-processing    
    fit_dic['p_null'] = deepcopy(p_final)
    for par in [ploc for ploc in fit_dic['p_null'] if 'ctrst' in ploc]:fit_dic['p_null'][par] = 0.
    com_joint_postproc(p_final,fixed_args,fit_dic,merged_chain,gen_dic)
    
    print('     ----------------------------------')
    
    return None

    


def FIT_joined_DiffProf(param,x_tab,args=None):
    r"""**Fit function: joined differential stellar profiles**

    Calls corresponding model function for optimization

    Args:
        TBD
    
    Returns:
        TBD
    """
    
    #Models over fitted spectral ranges
    mod_dic=joined_DiffProf(param,args)[0]

    #Merit table
    chi = calc_chi_Prof(mod_dic,args)
    
    return chi,None





   
def joined_DiffProf(param,fixed_args):
    r"""**Model function: joined differential profiles**

    Defines the joined model for differential profiles. This is done in three steps
    
     1. We calculate all DI profiles of the star (fitted exposures + exposures that contributed to the master-out), and we scale 
        them at the same value as after the `Broadband flux Scaling module`.
        A given out-of-transit profile corresponds to 

        .. math:: 
           F_\mathrm{DI}(t_\mathrm{out}) &= \sum_{k} dF_\mathrm{quiet}(k) + \sum_{i(t)} dF_\mathrm{quiet}(i(t)) + \sum_{j(t)} dF_\mathrm{active region}(j(t))   \\ 

        Where stellar cells are either quiet or within active regions, and `k` indicates cells that are never within active regions. 
        Other cells nature depend on time because the active region and planet are moving. 
        A given in-transit profile corresponds to 

        .. math::         
           F_\mathrm{DI}(t_\mathrm{in})  &= \sum_{k} dF_\mathrm{quiet}(k) + \sum_{i(t),nopl} dF_\mathrm{quiet}(i(t)) + \sum_{i(t),pl} dF_\mathrm{quiet}(i(t)) + \sum_{j(t),nopl} dF_\mathrm{active region}(j(t))  + \sum_{j(t),pl} dF_\mathrm{active region}(j(t))   \\       
                                         &= \sum_{k} dF_\mathrm{quiet}(k) + \sum_{i(t),nopl} dF_\mathrm{quiet}(i(t)) + \sum_{j(t),nopl} dF_\mathrm{active region}(j(t))  \\       

        Where the `pl` cells are occulted by the planet at time `t` and thus null. 
        
     2. We compute the master out, with same weights as those used in the corresponding module.
        The master out writes (neglecting weights) as 

     .. math:: 
        F_\mathrm{DI}(out) = \sum_{k} dF_\mathrm{quiet}(k) + <_{t_\mathrm{out}} \sum_{i(t)} dF_\mathrm{quiet}(i(t)) + \sum_{j(t)} dF_\mathrm{active region}(j(t)) >  \\ 

     3. We extract differential profiles as :math:`F_\mathrm{diff} = F_\mathrm{out} - F_\mathrm{sc}`, corresponding to 

        .. math:: 
           F_\mathrm{diff}(t_\mathrm{out}) &= <_{t_\mathrm{out}} \sum_{i(t)} dF_\mathrm{quiet}(i(t)) + \sum_{j(t)} dF_\mathrm{active region}(j(t)) > - \sum_{i(t)} dF_\mathrm{quiet}(i(t)) + \sum_{j(t)} dF_\mathrm{active region}(j(t))   \\ 
           F_\mathrm{diff}(t_\mathrm{in})  &=  <_{t_\mathrm{out}} \sum_{i(t)} dF_\mathrm{quiet}(i(t)) + \sum_{j(t)} dF_\mathrm{active region}(j(t)) - \sum_{i(t),nopl} dF_\mathrm{quiet}(i(t)) - \sum_{j(t),nopl} dF_\mathrm{active region}(j(t))> 

        If the active region is fixed, then

        .. math:: 
           F_\mathrm{diff}(t_\mathrm{out}) &= 0  \\ 
           F_\mathrm{diff}(t_\mathrm{in})  &=  \sum_{i(t)} dF_\mathrm{quiet}(i(t)) + \sum_{j} dF_\mathrm{active region}(j) - \sum_{i(t),nopl} dF_\mathrm{quiet}(i(t)) - \sum_{j,nopl} dF_\mathrm{active region}(j) 
                                           &=  \sum_{i(t),pl} dF_\mathrm{quiet}(i(t)) + \sum_{j,pl} dF_\mathrm{active region}(j) 

        In that case in-transit differential profiles can be fitted directly with the `joined_IntrProf` routine, calculating the quiet or active regions profiles of planet-occulted cells.

    Args:
        TBD
    
    Returns:
        TBD
    
    """
    args = deepcopy(fixed_args)
    mod_dic = {}
    mod_prop_dic = {}
    coeff_line_dic = {}

    #Updating spectral grid for master out
    if 'rv_shift' in param:
        if 'spec' in args['type']:spec_dopshift = 1./gen_specdopshift(param['rv_shift'])
        for key in ['cen_bins', 'edge_bins']:
            if 'spec' in args['type']:args['master_out']['master_out_tab'][key]*=spec_dopshift
            else:args['master_out']['master_out_tab'][key]-=param['rv_shift']

    #Updating stellar grid
    #    - necessary to determine the global 2D quiet star grid on the most up to date version of the stellar grid
    if args['var_star_grid']:up_model_star(args, param)

    #Processing instruments
    for inst in args['inst_list']:
        args['inst']=inst
        mod_dic[inst]={}
        coeff_line_dic[inst]={}
        mod_prop_dic[inst]={}

        #Processing visits
        for vis in args['inst_vis_list'][inst]:
            args['vis']=vis
            cond_def_all=np.zeros([len(args['master_out']['idx_in_master_out'][inst][vis]),len(args['master_out']['master_out_tab']['cen_bins'])], dtype=bool)
            cond_undef_weights=np.zeros(len(args['master_out']['master_out_tab']['cen_bins']), dtype=bool)
            args['master_out']['weights'][inst][vis]=np.zeros([len(args['master_out']['idx_in_master_out'][inst][vis]),len(args['master_out']['master_out_tab']['cen_bins'])], dtype=float)
            contrib_profs=np.zeros([len(args['master_out']['idx_in_master_out'][inst][vis]),len(args['master_out']['master_out_tab']['cen_bins'])], dtype=float)

            #Outputs
            if not args['fit']:outputs_Prof(inst,vis,coeff_line_dic,mod_prop_dic,args,param) 

            #Retrieve updated coordinates of planet- and active region-oculted regions or use imported values
            system_param_loc,coord_pl_ar,param_val = up_plocc_arocc_prop(inst,vis,args,param,args['studied_pl'][inst][vis],args['ph_fit'][inst][vis],args['coord_fit'][inst][vis],studied_ar=args['studied_ar'][inst][vis])

            #-----------------------------------------------------------
            #Defining the base stellar profile
            #-----------------------------------------------------------
            
            #Table for model calculation - wavelength table of the exposure considered
            args_DI = def_st_prof_tab(inst,vis,0,args)

            #Initializing broadband scaling of intrinsic profiles into local profiles
            args_DI['Fsurf_grid_spec'] = theo_intr2loc(args_DI['grid_dic'],args_DI['system_prop'],args_DI,args_DI['ncen_bins'],args_DI['grid_dic']['nsub_star']) 
            
            #Initializing construction of stellar flux grid
            if not args['fit']:init_st_intr_prof(args_DI,args_DI['grid_dic'],param)
            
            #Construction of stellar flux grid 
            base_DI_prof = custom_DI_prof(param_val,None,args=args_DI)[0]

            #Making profile for each exposure
            for isub,iexp in enumerate(args['idx_in_fit'][inst][vis]):

                #Table for model calculation - wavelength table of the exposure considered
                args_exp = deepcopy(def_st_prof_tab(inst,vis,isub,args))

                #Model DI profile for current exposure accounting for deviations from the nominal profile - on the wavelength table of the exposure considered
                #    - we subtract from the quiet disk-integrated stellar profile:
                # + the total local profile from regions occulted by all transiting planets, accounting for their overlaps
                #   with occulted cells that may be part of active regions 
                # + the total deviation profile from active region-occulted regions, which is the difference between the quiet stellar emission and the active region emission  
                #   cells occulted by planets do not contribute to this profile 
                #    - occulted stellar cells (from planet and active regions) are automatically identified within sub_calc_plocc_ar_prop() 
                surf_prop_dic,surf_prop_dic_ar,_ = sub_calc_plocc_ar_prop([args['chrom_mode']],args_exp,args['par_list'],args['studied_pl'][inst][vis],args['studied_ar'][inst][vis],system_param_loc,args['grid_dic'],args['system_prop'],param_val,coord_pl_ar,[isub],system_ar_prop_in=args['system_ar_prop'])
                sp_line_model = base_DI_prof - surf_prop_dic[args['chrom_mode']]['line_prof'][:,0] - surf_prop_dic_ar[args['chrom_mode']]['line_prof'][:,0]

                #Properties of all planet- and active region-occulted regions used to calculate spectral line profiles
                # - Since we are analyzing differential profiles, we have to check if the planets/active regions are in the exposure considered.
                # - If this is not the case, an entry for them in the surf_prop_dic/surf_prop_dic_ar won't be initialized
                if not args['fit']:
                    for pl_loc in args['studied_pl'][inst][vis]:  
                        if np.abs(coord_pl_ar[pl_loc]['ecl'][isub])!=1:                
                            for prop_loc in mod_prop_dic[inst][vis][pl_loc]:
                                mod_prop_dic[inst][vis][pl_loc][prop_loc][isub] = surf_prop_dic[args['chrom_mode']][pl_loc][prop_loc][0]

                    for ar in args['studied_ar'][inst][vis]:
                        if np.sum(coord_pl_ar[ar]['is_visible'][:, isub]):
                            for prop_loc in mod_prop_dic[inst][vis][ar]:
                                mod_prop_dic[inst][vis][ar][prop_loc][isub] = surf_prop_dic_ar[args['chrom_mode']][ar][prop_loc][0]

                #Convolve model profiles to instrument resolution
                conv_base_DI_prof = convol_prof(base_DI_prof,args_exp['cen_bins'],args['FWHM_inst'][inst])
                conv_line_model = convol_prof(sp_line_model,args_exp['cen_bins'],args['FWHM_inst'][inst])

                #Set negative flux values to null
                conv_line_model[conv_line_model<base_DI_prof[0]-1] = 0.
                                
                #Store the model DI profiles for calculation of the differential profiles later
                args['raw_DI_profs'][inst][vis][isub] = conv_line_model

                #Loop over exposures contributing to the master-out
                if iexp in args['master_out']['idx_in_master_out'][inst][vis]:
                    
                    #Storing the index of the exposure considered in the array of master-out indices
                    master_isub = args['master_out']['idx_in_master_out'][inst][vis].index(iexp)

                    #Re-sample model DI profile on a common grid
                    resamp_conv_base_DI_prof = bind.resampling(args['master_out']['master_out_tab']['edge_bins'],args['edge_bins'][inst][vis][isub],conv_base_DI_prof,kind=args['master_out']['master_out_tab']['resamp_mode'])
                    resamp_line_model = bind.resampling(args['master_out']['master_out_tab']['edge_bins'],args['edge_bins'][inst][vis][isub],conv_line_model,kind=args['master_out']['master_out_tab']['resamp_mode'])
                    

                    #Making weights for the master-out
                    #    - assuming no detector noise and a constant calibration
                    #    - if DI profiles were converted from 2D to 1D, we use directly their variance profiles
                    raw_weights=weights_bin_prof(range(args['master_out']['nord']), args['master_out']['scaled_data_paths'][inst][vis],inst,vis,args['master_out']['corr_Fbal'],args['master_out']['corr_FbalOrd'],\
                                                        args['master_out']['save_data_dir'],args['master_out']['nord'],isub,'DI',args['type'],args['dim_exp'][inst][vis],args['master_out']['sing_gcal'][inst][vis][isub],\
                                                        None,np.array([args['cen_bins'][inst][vis][isub]]),args['coord_fit'][inst][vis]['t_dur'][isub],np.array([resamp_conv_base_DI_prof]),\
                                                        None,args['master_out']['calc_cond'][inst],EFsc2_all_in = args['master_out']['EFsc2'][inst][vis][isub])[0]

                    # - Re-sample the weights
                    resamp_weights = bind.resampling(args['master_out']['master_out_tab']['edge_bins'],args['edge_bins'][inst][vis][isub],raw_weights,kind=args['master_out']['master_out_tab']['resamp_mode'])

                    # - Set nan values and corresponding weights to 0 
                    cond_def_all[master_isub] = ~np.isnan(resamp_line_model)
                    resamp_weights[~cond_def_all[master_isub]] = 0.
                    resamp_line_model[~cond_def_all[master_isub]] = 0.

                    # - Find pixels where there is undefined or negative weights 
                    cond_undef_weights |= ( (np.isnan(resamp_weights) | (resamp_weights<0) ) & cond_def_all[master_isub] ) 

                    # - Store the weights 
                    args['master_out']['weights'][inst][vis][master_isub] = resamp_weights

                    #Store the contributing profiles to the master-out
                    contrib_profs[master_isub] = resamp_line_model

            #Defined bins in binned spectrum
            cond_def_binned = np.sum(cond_def_all,axis=0)>0

            #Disable weighing in all binned profiles for pixels validating at least one of these conditions:
            # + 'cond_null_weights' : pixel has null weights at all defined flux values (weight_exp_all is null at undefined flux values, so if its sum is null in a pixel 
            # fulfilling cond_def_binned it implies it is null at all defined flux values for this pixel)
            # + 'cond_undef_weights' : if at least one profile has an undefined weight for a defined flux value, it messes up with the weighted average     
            #    - in both cases we thus set all weights to a common value (arbitrarily set to unity for the pixel), ie no weighing is applied
            #    - pixels with undefined flux values do not matter as their flux has been set to 0, so they can be attributed an arbitrary weight
            cond_null_weights = (np.sum(args['master_out']['weights'][inst][vis],axis=0)==0.) & cond_def_binned
            args['master_out']['weights'][inst][vis][:, cond_undef_weights | cond_null_weights] = 1.

            #Global weight table
            #    - pixels that do not contribute to the binning (eg due to planetary range masking) have null flux and weight values, and thus do not contribute to the total weight
            #    - weight tables only depend on each original exposure but their weight is specific to the new exposures and the original exposures it contains
            x_low = args['ph_fit'][inst][vis][args['ref_pl'][inst][vis]][0,args['master_out']['idx_in_master_out'][inst][vis]]
            x_high = args['ph_fit'][inst][vis][args['ref_pl'][inst][vis]][2,args['master_out']['idx_in_master_out'][inst][vis]]
            dx_ov_in = x_high - x_low
            dx_ov_all = np.ones([len(args['master_out']['idx_in_master_out'][inst][vis]),len(args['master_out']['master_out_tab']['cen_bins'])],dtype=float) if (np.sum(dx_ov_in)==0) else dx_ov_in[:,None] 
            args['master_out']['weights'][inst][vis] *= dx_ov_all

            #Storing normalization information
            glob_weights_tot = np.sum(args['master_out']['weights'][inst][vis][:, cond_def_binned],axis=0)
            if vis in args['master_out']['multivisit_list'][inst]:args['master_out']['multivisit_weights_total'][inst][cond_def_binned]+=glob_weights_tot

            #Perform the weighted average to retrieve the master-out
            # - We can disregard the division by the sum of the weights since the weights are normalized
            if vis in args['master_out']['multivisit_list'][inst]:args['master_out']['multivisit_flux'][inst][cond_def_binned] += np.sum(contrib_profs[:, cond_def_binned]*args['master_out']['weights'][inst][vis][:, cond_def_binned], axis=0)
            else:args['master_out']['flux'][inst][vis][cond_def_binned] = np.sum(contrib_profs[:, cond_def_binned]*args['master_out']['weights'][inst][vis][:, cond_def_binned], axis=0)/glob_weights_tot

        #Need to step out of the loops to finish the master-out calculation if multiple visits are combined
        if len(args['master_out']['multivisit_list'][inst])>0:args['master_out']['multivisit_flux'][inst] /= args['master_out']['multivisit_weights_total'][inst]

        #-----------------------------------------------------------
        #Building differential profiles
        #-----------------------------------------------------------
        for vis in args['inst_vis_list'][inst]:
            
            #-----------------------------------------------------------
            #Variable line model for each exposure 
            #    - the intrinsic stellar line profile is convolved with the LSF kernel specific to each instrument
            #    - we assume a flat continuum, set after the PC component so that intrinsic profiles models later on can be defined with a continuum unity
            #-----------------------------------------------------------
            mod_dic[inst][vis]=np.zeros(args['nexp_fit_all'][inst][vis],dtype=object)
            for isub,iexp in enumerate(args['idx_in_fit'][inst][vis]):
                
                #Retrieving the master-out flux
                if vis in args['master_out']['multivisit_list'][inst]:master_out_flux=args['master_out']['multivisit_flux'][inst]
                else:master_out_flux=args['master_out']['flux'][inst][vis]

                #Re-sample master on table of the exposure considered
                resamp_master = bind.resampling(args['edge_bins'][inst][vis][isub],args['master_out']['master_out_tab']['edge_bins'],master_out_flux, kind=args['master_out']['master_out_tab']['resamp_mode'])

                #Calculate the differential profile on the wavelength table of the exposure considered (Isn't this gonna be an issue when making the residual map?)
                mod_dic[inst][vis][isub] = resamp_master - args['raw_DI_profs'][inst][vis][isub]

                #Add PC noise model
                #    - added to the convolved profiles since PC are derived from observed data
                if args['n_pc'][inst][vis] is not None:
                    for i_pc in range(args['n_pc'][inst][vis]):mod_dic[inst][vis][isub]+=param_val[args['name_prop2input']['aPC_idxin'+str(iexp)+'_ord'+str(i_pc)+'__IS'+inst+'_VS'+vis]]*args['eig_res_matr'][inst][vis][iexp][i_pc]

    return mod_dic,coeff_line_dic,mod_prop_dic

    
def calc_chi_Prof(mod_dic,args):
    r"""**Fit function: merit grid**

    Calculates merit table for optimization.
    
    Because exposures are specific to each visit, defined on different bins, and stored as objects we define the output table as : 
        
        chi = concatenate( exp, (obs(exp)-mod(exp))/err(exp)) ) 

    Or the equivalent with the covariance matrix, so that the merit function will compare chi to a table of same size filled with 0 and with errors of 1 
    in the residual() function (where the condition to use covariance has been set to False for this purpose)
    
    Observed intrinsic profiles may have gaps, but due to the convolution the model must be calculated over the continuous table and then limited to fitted bins
    
    Args:
        TBD
    
    Returns:
        TBD
    """

    chi = np.zeros(0,dtype=float)
    if args['use_cov_eff']:
        for inst in args['inst_list']:
            for vis in args['inst_vis_list'][inst]:    
                for iexp in range(args['nexp_fit_all'][inst][vis]):
                    L_mat = scipy.linalg.cholesky_banded(args['cov'][inst][vis][iexp], lower=True)
                    res = args['flux'][inst][vis][iexp]-mod_dic[inst][vis][iexp]
                    cond_fit = args['cond_fit'][inst][vis][iexp]
                    res[~cond_fit] = 0.  
                    chiexp  = scipy.linalg.blas.dtbsv(L_mat.shape[0]-1, L_mat, res, lower=True)
                    chi = np.append( chi, chiexp[cond_fit] )                     

    else:
        for inst in args['inst_list']:
            for vis in args['inst_vis_list'][inst]:    
                for isub in range(args['nexp_fit_all'][inst][vis]):
                    cond_fit = args['cond_fit'][inst][vis][isub]
                    res = args['flux'][inst][vis][isub][cond_fit]-mod_dic[inst][vis][isub][cond_fit]
                    chi = np.append( chi, res/np.sqrt( args['cov'][inst][vis][isub][0][cond_fit]) )
         
    return chi    
    
def outputs_Prof(inst,vis,coeff_line_dic,mod_prop_dic,args,param):
    r"""**Fit function: outputs**

    Defines outputs for profile fit functions
    
    Args:
        TBD
    
    Returns:
        TBD
    """

    #Coefficients describing the polynomial variation of spectral line properties as a function of the chosen coordinate
    if ('coeff_line' in args):
        coeff_line_dic[inst][vis]={}
        for par_loc in args['linevar_par'][inst][vis]:    
            coeff_line_dic[inst][vis][par_loc] = polycoeff_def(param,args['coeff_ord2name'][inst][vis][par_loc]) 
    else:coeff_line_dic[inst][vis] = None              
    
    #Properties of all planet-occulted regions used to calculate spectral line profiles
    mod_prop_dic[inst][vis]={} 
    linevar_par_list = ['rv']
    if (len(args['linevar_par'])>0):linevar_par_list+=args['linevar_par'][inst][vis]
    for pl_loc in args['studied_pl'][inst][vis]:
        mod_prop_dic[inst][vis][pl_loc]={}   
        for prop_loc in linevar_par_list:mod_prop_dic[inst][vis][pl_loc][prop_loc] = np.zeros(len(args['idx_in_fit'][inst][vis]))*np.nan  
    for ar in args['studied_ar'][inst][vis]:
        mod_prop_dic[inst][vis][ar]={}   
        for prop_loc in linevar_par_list:mod_prop_dic[inst][vis][ar][prop_loc] = np.zeros(len(args['idx_in_fit'][inst][vis]))*np.nan
         
    return None