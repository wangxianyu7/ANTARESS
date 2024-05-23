#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import bindensity as bind
import lmfit
import os
from lmfit import Parameters
from ..ANTARESS_general.utils import stop,np_where1D,dataload_npz
from ..ANTARESS_analysis.ANTARESS_ana_comm import init_joined_routines,init_joined_routines_inst,init_joined_routines_vis,init_joined_routines_vis_fit,com_joint_fits,com_joint_postproc
from ..ANTARESS_grids.ANTARESS_occ_grid import sub_calc_plocc_spot_prop,up_plocc_prop,calc_spotted_tiles, calc_plocced_tiles
from ..ANTARESS_grids.ANTARESS_prof_grid import gen_theo_intr_prof,theo_intr2loc,init_custom_DI_prof,custom_DI_prof
from ..ANTARESS_analysis.ANTARESS_inst_resp import get_FWHM_inst,resamp_st_prof_tab,def_st_prof_tab,conv_st_prof_tab,cond_conv_st_prof_tab,convol_prof
from ..ANTARESS_grids.ANTARESS_star_grid import up_model_star
from ..ANTARESS_conversions.ANTARESS_binning import weights_bin_prof




def joined_Star_ana(glob_fit_dic,system_param,theo_dic,data_dic,gen_dic,plot_dic,coord_dic):
    r"""**Joined stellar fits**

    Wrap-up function to call joint fits of stellar properties and profiles

    Args:
        TBD
    
    Returns:
        TBD
    
    """        
    #Fitting disk-integrated stellar properties with a linked model
    if gen_dic['fit_DIProp']:
        main_joined_DIProp('DIProp',glob_fit_dic['DIProp'],gen_dic,system_param,theo_dic,plot_dic,coord_dic,data_dic)   

    #Fitting stellar surface properties with a linked model
    if gen_dic['fit_IntrProp']:
        main_joined_IntrProp('IntrProp',glob_fit_dic['IntrProp'],gen_dic,system_param,theo_dic,plot_dic,coord_dic,data_dic)    
    
    #Fitting intrinsic stellar lines with joined model
    if gen_dic['fit_IntrProf']:
        main_joined_IntrProf('IntrProf',data_dic,gen_dic,system_param,glob_fit_dic['IntrProf'],theo_dic,plot_dic,coord_dic)    

    #Fitting local stellar lines with joined model, including spots in the fitted parameters
    if gen_dic['fit_ResProf']:
        main_joined_ResProf('ResProf',data_dic,gen_dic,system_param,glob_fit_dic['ResProf'],theo_dic,plot_dic,coord_dic)    

    return None


def main_joined_DIProp(data_mode,fit_prop_dic,gen_dic,system_param,theo_dic,plot_dic,coord_dic,data_dic):
    r"""**Joined disk-integrated stellar property fits**

    Main routine to fit a given disk-integrated stellar property with a joined model over instruments and visits.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    print('   > Fitting single disk-integrated stellar property')
    
    #Initializations
    for prop_loc in fit_prop_dic['mod_prop']:  
        fixed_args,fit_dic = init_joined_routines(data_mode,gen_dic,system_param,theo_dic,data_dic,fit_prop_dic)
        print('     - '+{'rv_res':'RV residuals','FWHM':'Line FWHM','ctrst':'Line contrast'}[prop_loc])        
        fit_dic['save_dir']+=prop_loc+'/'       
    
        #Arguments to be passed to the fit function
        fixed_args.update({
            'rout_mode':'DIProp',
            'prop_fit':prop_loc})    
    
        #Construction of the fit tables
        for par in ['s_val','y_val']:fixed_args[par]=np.zeros(0,dtype=float)
        idx_fit2vis={}
        for inst in np.intersect1d(data_dic['instrum_list'],list(fit_prop_dic['idx_in_fit'].keys())):    
            init_joined_routines_inst('DIProp',inst,fit_prop_dic,fixed_args)
            idx_fit2vis[inst] = {}
            for vis in data_dic[inst]['visit_list']:
                init_joined_routines_vis(inst,vis,fit_prop_dic,fixed_args)
    
                #Visit is fitted
                if fixed_args['bin_mode'][inst][vis] is not None: 
                    data_vis=data_dic[inst][vis]
                    init_joined_routines_vis_fit('DIProp',inst,vis,fit_prop_dic,fixed_args,data_vis,gen_dic,data_dic,coord_dic)
    
                    #Binned/original data
                    if fixed_args['bin_mode'][inst][vis]=='_bin':data_load = dataload_npz(gen_dic['save_data_dir']+'/DIbin_prop/'+inst+'_'+vis)
                    else:data_load = dataload_npz(gen_dic['save_data_dir']+'/DIorig_prop/'+inst+'_'+vis)
                  
                    #Fit tables
                    idx_fit2vis[inst][vis] = range(fit_dic['nx_fit'],fit_dic['nx_fit']+fixed_args['nexp_fit_all'][inst][vis])
                    fit_dic['nx_fit']+=fixed_args['nexp_fit_all'][inst][vis]
                    for i_in in fixed_args['idx_in_fit'][inst][vis]:    
                        fixed_args['y_val'] = np.append(fixed_args['y_val'],data_load[i_in][fixed_args['prop_fit']])
                        fixed_args['s_val'] = np.append(fixed_args['s_val'],np.mean(data_load[i_in]['err_'+fixed_args['prop_fit']]))    

        fixed_args['idx_fit'] = np.ones(fit_dic['nx_fit'],dtype=bool)
        fixed_args['nexp_fit'] = fit_dic['nx_fit']
        fixed_args['x_val']=range(fixed_args['nexp_fit'])
        fixed_args['fit_func'] = FIT_joined_DIProp
        fixed_args['inside_fit'] = False    

        #Uncertainties on the property are given a covariance matrix structure for consistency with the fit routine 
        fixed_args['cov_val'] = np.array([fixed_args['s_val']**2.])
        fixed_args['use_cov'] = False       

        #Model fit and calculation
        if prop_loc not in fit_prop_dic['mod_prop']:fit_prop_dic['mod_prop'][prop_loc] = {}
        merged_chain,p_final = com_joint_fits('DIProp',fit_dic,fixed_args,fit_prop_dic,gen_dic,data_dic,theo_dic,fit_prop_dic['mod_prop'][prop_loc])   
        
        #Best-fit model and properties
        fit_save={}
        fixed_args['fit'] = False
        mod_tab,coeff_line_dic,fit_save['prop_mod']= joined_DIProp(p_final,fixed_args)
      
        #Save best-fit properties
        fit_save.update({'p_final':p_final,'name_prop2input':fixed_args['name_prop2input'],'merit':fit_dic['merit']})
        if (plot_dic['prop_DI']!='') or (plot_dic['chi2_fit_DIProp']!=''):
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
        if fixed_args['prop_fit']=='rv':fit_dic['p_null']['veq'] = 0.
        else:
            for par in fit_dic['p_null']:fit_dic['p_null'][par]=0.
            fit_dic['p_null']['cont']=1.
        com_joint_postproc(p_final,fixed_args,fit_dic,merged_chain,fit_prop_dic,gen_dic)

    print('     ----------------------------------')    
    
    return None












def main_joined_IntrProp(data_mode,fit_prop_dic,gen_dic,system_param,theo_dic,plot_dic,coord_dic,data_dic):
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
        fixed_args,fit_dic = init_joined_routines(data_mode,gen_dic,system_param,theo_dic,data_dic,fit_prop_dic)
        print('     - '+{'rv':'Surface RVs','FWHM':'Intrinsic line FWHM','ctrst':'Intrinsic line contrast','a_damp':'Intrinsic line damping coefficient'}[prop_loc])        
        fit_dic['save_dir']+=prop_loc+'/'        
    
        #Arguments to be passed to the fit function
        #    - fit is performed on achromatic average properties
        #    - the full stellar line profile are not calculated, since we only fit the average properties of the occulted regions
        fixed_args.update({
            'rout_mode':'IntrProp',
            'chrom_mode':'achrom',
            'mode':'ana',  #to activate the calculation of line profile properties
            'prop_fit':prop_loc})
        
        #Coordinate and property to calculate for the fit
        fixed_args['par_list']+=[fixed_args['prop_fit']]
        if fit_prop_dic['dim_fit'] in ['abs_y_st','y_st2']:fixed_args['coord_line']='y_st'    
        else:fixed_args['coord_line']=fit_prop_dic['dim_fit']
        fixed_args['par_list']+=[fixed_args['coord_line']]
    
        #Construction of the fit tables
        for par in ['s_val','y_val']:fixed_args[par]=np.zeros(0,dtype=float)
        for par in ['coord_fit','ph_fit']:fixed_args[par]={}
        idx_fit2vis={}
        for inst in np.intersect1d(data_dic['instrum_list'],list(fit_prop_dic['idx_in_fit'].keys())):    
            init_joined_routines_inst('IntrProp',inst,fit_prop_dic,fixed_args)
            idx_fit2vis[inst] = {}
            for vis in data_dic[inst]['visit_list']:
                init_joined_routines_vis(inst,vis,fit_prop_dic,fixed_args)
    
                #Visit is fitted
                if fixed_args['bin_mode'][inst][vis] is not None: 
                    data_vis=data_dic[inst][vis]
                    init_joined_routines_vis_fit('IntrProp',inst,vis,fit_prop_dic,fixed_args,data_vis,gen_dic,data_dic,coord_dic)
    
                    #Binned/original data
                    if fixed_args['bin_mode'][inst][vis]=='_bin':data_load = dataload_npz(gen_dic['save_data_dir']+'/Intrbin_prop/'+inst+'_'+vis)
                    else:data_load = dataload_npz(gen_dic['save_data_dir']+'/Introrig_prop/'+inst+'_'+vis)
                  
                    #Fit tables
                    idx_fit2vis[inst][vis] = range(fit_dic['nx_fit'],fit_dic['nx_fit']+fixed_args['nexp_fit_all'][inst][vis])
                    fit_dic['nx_fit']+=fixed_args['nexp_fit_all'][inst][vis]
                    for i_in in fixed_args['idx_in_fit'][inst][vis]:    
                        fixed_args['y_val'] = np.append(fixed_args['y_val'],data_load[i_in][fixed_args['prop_fit']])
                        fixed_args['s_val'] = np.append(fixed_args['s_val'],np.mean(data_load[i_in]['err_'+fixed_args['prop_fit']]))

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
        if prop_loc not in fit_prop_dic['mod_prop']:fit_prop_dic['mod_prop'][prop_loc] = {}
        merged_chain,p_final = com_joint_fits('IntrProp',fit_dic,fixed_args,fit_prop_dic,gen_dic,data_dic,theo_dic,fit_prop_dic['mod_prop'][prop_loc])   
        
        #Best-fit model and properties
        fit_save={}
        fixed_args['fit'] = False
        mod_tab,coeff_line_dic,fit_save['prop_mod']= fixed_args['mod_func'](p_final,fixed_args)
      
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
        if fixed_args['prop_fit']=='rv':fit_dic['p_null']['veq'] = 0.
        else:
            for par in fit_dic['p_null']:fit_dic['p_null'][par]=0.
            fit_dic['p_null']['cont']=1.
        com_joint_postproc(p_final,fixed_args,fit_dic,merged_chain,fit_prop_dic,gen_dic)

    print('     ----------------------------------')    
  
    return None






def FIT_joined_IntrProp(param,x_tab,args=None):
    r"""**Fit function: joined stellar property**

    Calls corresponding model function for optimization

    Args:
        TBD
    
    Returns:
        TBD
    """
    return joined_IntrProp(param,args)[0]




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
    coeff_line_dic = {}
    mod_coord_dic={}
    for inst in args['inst_list']:
        args['inst']=inst
        mod_prop_dic[inst]={}
        coeff_line_dic[inst]={}
        mod_coord_dic[inst]={}
        for vis in args['inst_vis_list'][inst]: 
            args['vis']=vis 
            
            #Calculate coordinates and properties of occulted regions 
            system_param_loc,coord_pl,param_val = up_plocc_prop(inst,vis,args,param,args['transit_pl'][inst][vis],args['ph_fit'][inst][vis],args['coord_fit'][inst][vis])
            surf_prop_dic,spotocc_prop,surf_prop_dic_common = sub_calc_plocc_spot_prop([args['chrom_mode']],args,args['par_list'],args['transit_pl'][inst][vis],system_param_loc,args['grid_dic'],args['system_prop'],param_val,coord_pl,range(args['nexp_fit_all'][inst][vis]))

            
            #Properties associated with the transiting planet in the visit 
            pl_vis = args['transit_pl'][inst][vis][0]
            theo_vis = surf_prop_dic['achrom'][pl_vis]      
            
            #Fit coordinate
            #    - only used for plots
            if (not args['fit']) and ('coeff_line' in args):coeff_line_dic[inst][vis] = args['coeff_line']
            else:coeff_line_dic[inst][vis] = None

            #Model property for the visit 
            mod_prop_dic[inst][vis] = theo_vis[args['prop_fit']][0] 
         
            #Appending over all visits
            mod_tab=np.append(mod_tab,mod_prop_dic[inst][vis])

    return mod_tab,coeff_line_dic,mod_prop_dic
    


















def main_joined_IntrProf(data_mode,data_dic,gen_dic,system_param,fit_prop_dic,theo_dic,plot_dic,coord_dic):
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
    fixed_args,fit_dic = init_joined_routines(data_mode,gen_dic,system_param,theo_dic,data_dic,fit_prop_dic)

    ######################################################################################################## 

    #Arguments to be passed to the fit function
    fixed_args.update({ 
        'rout_mode':'IntrProf',
        'func_prof_name':fit_prop_dic['func_prof_name'],
        'mode':fit_prop_dic['mode'],
        'cen_bins':{},
        'edge_bins':{},
        'dcen_bins':{},
        'dim_exp':{},
        'ncen_bins':{},
        'cond_fit':{},
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
    #    - other required properties are automatically added in the sub_calc_plocc_spot_prop() function
    fixed_args['par_list']+=['line_prof']
    if fixed_args['mode']=='ana':
        if fit_prop_dic['dim_fit'] in ['abs_y_st','y_st2']:fixed_args['coord_line']='y_st'    
        else:fixed_args['coord_line']=fit_prop_dic['dim_fit']
        fixed_args['par_list']+=[fixed_args['coord_line']]
    else:
        fixed_args['par_list']+=['mu']

    #Activation of spectral conversion and resampling 
    cond_conv_st_prof_tab(theo_dic['rv_osamp_line_mod'],fixed_args,data_dic[data_dic['instrum_list'][0]]['type'])                           

    #Construction of the fit tables
    for par in ['coord_fit','ph_fit']:fixed_args[par]={}
    for inst in np.intersect1d(data_dic['instrum_list'],list(fit_prop_dic['idx_in_fit'].keys())):  
        init_joined_routines_inst('IntrProf',inst,fit_prop_dic,fixed_args)
        for key in ['cen_bins','edge_bins','dcen_bins','cond_fit','flux','cov','cond_def','n_pc','dim_exp','ncen_bins']:fixed_args[key][inst]={}
        if len(fit_prop_dic['PC_model'])>0:fixed_args['eig_res_matr'][inst]={}
        fit_save['idx_trim_kept'][inst] = {}
        if (fixed_args['mode']=='ana') and (inst not in fixed_args['func_prof_name']):fixed_args['func_prof_name'][inst] = 'gauss'
        if (inst in fit_prop_dic['order']):iord_sel =  fit_prop_dic['order'][inst]
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
            init_joined_routines_vis(inst,vis,fit_prop_dic,fixed_args)
            
            #Visit is fitted
            if fixed_args['bin_mode'][inst][vis] is not None:   
                data_vis=data_dic[inst][vis]
                init_joined_routines_vis_fit('IntrProf',inst,vis,fit_prop_dic,fixed_args,data_vis,gen_dic,data_dic,coord_dic)   
                data_com = dataload_npz(data_dic[inst][vis]['proc_com_data_paths'])             
                
                #Instrumental convolution
                if (inst not in fixed_args['FWHM_inst']):                
                    fixed_args['FWHM_inst'][inst] = get_FWHM_inst(inst,fixed_args,data_com['cen_bins'][iord_sel])
                
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
                idx_range_kept = np_where1D((data_com['edge_bins'][iord_sel,0:-1]>=min_trim_range) & (data_com['edge_bins'][iord_sel,1::]<=max_trim_range))
                ncen_bins = len(idx_range_kept)
                if ncen_bins==0:stop('Empty trimmed range')                  
                
                fit_save['idx_trim_kept'][inst][vis] = idx_range_kept
                fixed_args['ncen_bins'][inst][vis] = ncen_bins  
                fixed_args['dim_exp'][inst][vis] = [1,ncen_bins] 

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
                for key in ['dcen_bins','cen_bins','edge_bins','cond_fit','flux','cov','cond_def']:fixed_args[key][inst][vis]=np.zeros(fixed_args['nexp_fit_all'][inst][vis],dtype=object)
                fit_prop_dic[inst][vis]['cond_def_fit_all']=np.zeros([fixed_args['nexp_fit_all'][inst][vis],ncen_bins],dtype=bool)
                fit_prop_dic[inst][vis]['cond_def_cont_all'] = np.zeros([fixed_args['nexp_fit_all'][inst][vis],ncen_bins],dtype=bool)  
                for isub,i_in in enumerate(fixed_args['idx_in_fit'][inst][vis]):
              
                    #Upload latest processed intrinsic data
                    if fixed_args['bin_mode'][inst][vis]=='_bin':data_exp = dataload_npz(gen_dic['save_data_dir']+'Intrbin_data/'+inst+'_'+vis+'_phase'+str(i_in))               
                    else:data_exp = dataload_npz(data_dic[inst][vis]['proc_Intr_data_paths']+str(i_in))
                    
                    #Trimming profile         
                    for key in ['cen_bins','flux','cond_def']:fixed_args[key][inst][vis][isub] = data_exp[key][iord_sel,idx_range_kept]
                    fixed_args['edge_bins'][inst][vis][isub] = data_exp['edge_bins'][iord_sel,idx_range_kept[0]:idx_range_kept[-1]+2]   
                    fixed_args['dcen_bins'][inst][vis][isub] = fixed_args['edge_bins'][inst][vis][isub][1::]-fixed_args['edge_bins'][inst][vis][isub][0:-1]  
                    fixed_args['cov'][inst][vis][isub] = data_exp['cov'][iord_sel][:,idx_range_kept]  # *0.6703558343325438
                    
                    #Oversampled line profile model table
                    if fixed_args['resamp']:resamp_st_prof_tab(inst,vis,isub,fixed_args,gen_dic,fixed_args['nexp_fit_all'][inst][vis],theo_dic['rv_osamp_line_mod'])
               
                    #Initializing ranges in the relevant rest frame
                    if len(cont_range)==0:fit_prop_dic[inst][vis]['cond_def_cont_all'][isub] = True    
                    else:
                        for bd_int in cont_range:fit_prop_dic[inst][vis]['cond_def_cont_all'][isub] |= (fixed_args['edge_bins'][inst][vis][isub][0:-1]>=bd_int[0]) & (fixed_args['edge_bins'][inst][vis][isub][1:]<=bd_int[1])         
                    if len(fit_prop_dic['fit_range'][inst][vis])==0:fit_prop_dic[inst][vis]['cond_def_fit_all'][isub] = True    
                    else:
                        for bd_int in fit_prop_dic['fit_range'][inst][vis]:fit_prop_dic[inst][vis]['cond_def_fit_all'][isub] |= (fixed_args['edge_bins'][inst][vis][isub][0:-1]>=bd_int[0]) & (fixed_args['edge_bins'][inst][vis][isub][1:]<=bd_int[1])        

                    #Accounting for undefined pixels
                    fit_prop_dic[inst][vis]['cond_def_cont_all'][isub] &= fixed_args['cond_def'][inst][vis][isub]           
                    fit_prop_dic[inst][vis]['cond_def_fit_all'][isub] &= fixed_args['cond_def'][inst][vis][isub]          
                    fit_dic['nx_fit']+=np.sum(fit_prop_dic[inst][vis]['cond_def_fit_all'][isub])
                    fixed_args['cond_fit'][inst][vis][isub] = fit_prop_dic[inst][vis]['cond_def_fit_all'][isub]

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
                            fit_prop_dic['mod_prop'][pc_name]={'vary':True,'guess':0.} 
                            if i_pc==0:fit_prop_dic['mod_prop'][pc_name]['bd'] = [-4.,4.]
                            elif i_pc==1:fit_prop_dic['mod_prop'][pc_name]['bd'] = [-3.,2.]
                            else:fit_prop_dic['mod_prop'][pc_name]['bd'] = [-1.,1.]
                            fit_prop_dic['priors'][pc_name]={'low':-100. ,'high':100.,'mod':'uf'}

                #Number of fitted exposures
                fixed_args['nexp_fit']+=fixed_args['nexp_fit_all'][inst][vis]

    #Common data type
    for idx_inst,inst in enumerate(data_dic['instrum_list']):
        if idx_inst==0:fixed_args['type'] = data_dic[inst]['type']
        elif fixed_args['type'] != data_dic[inst]['type']:stop('Incompatible data types')

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
    merged_chain,p_final = com_joint_fits('IntrProf',fit_dic,fixed_args,fit_prop_dic,gen_dic,data_dic,theo_dic,fit_prop_dic['mod_prop'])            

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
    mod_dic,coeff_line_dic,mod_prop_dic = fixed_args['mod_func'](p_final,fixed_args)

    #Save best-fit properties
    #    - with same structure as fit to individual profiles 
    fit_save.update({'p_final':p_final,'coeff_line_dic':coeff_line_dic,'func_prof_name':fixed_args['func_prof_name'],'name_prop2input':fixed_args['name_prop2input'],'coord_line':fixed_args['coord_line'],'merit':fit_dic['merit'],
                     'pol_mode':fit_prop_dic['pol_mode'],'coeff_ord2name':fixed_args['coeff_ord2name'],'idx_in_fit':fixed_args['idx_in_fit'],'genpar_instvis':fixed_args['genpar_instvis'],'linevar_par':fixed_args['linevar_par'],
                     'ph_fit':fixed_args['ph_fit'], 'system_prop':fixed_args['system_prop'],'grid_dic':fixed_args['grid_dic'],'var_par_list':fixed_args['var_par_list'], 'fit_orbit':fixed_args['fit_orbit'], 'fit_RpRs':fixed_args['fit_RpRs']})
    if fixed_args['mode']=='ana':fit_save['func_prof'] = fixed_args['func_prof']
    np.savez(fit_dic['save_dir']+'Fit_results',data=fit_save,allow_pickle=True)
    if (plot_dic['Intr_prof']!='') or (plot_dic['Intr_prof_res']!='') or (plot_dic['prop_Intr']!='') or (plot_dic['sp_Intr_1D']!=''):
        for inst in fixed_args['inst_list']:
            for vis in fixed_args['inst_vis_list'][inst]:
                prof_fit_dic={'fit_range':fit_prop_dic['fit_range'][inst][vis]}
                if fixed_args['bin_mode'][inst][vis]=='_bin':prof_fit_dic['loc_data_corr_path'] = gen_dic['save_data_dir']+'Intrbin_data/'+inst+'_'+vis+'_phase'          
                else:prof_fit_dic['loc_data_corr_path'] = data_dic[inst][vis]['proc_Intr_data_paths']
                for isub,i_in in enumerate(fixed_args['idx_in_fit'][inst][vis]):
                    prof_fit_dic[i_in]={
                        'cen_bins':fixed_args['cen_bins'][inst][vis][isub],
                        'flux':mod_dic[inst][vis][isub],
                        'cond_def_fit':fit_prop_dic[inst][vis]['cond_def_fit_all'][isub],
                        'cond_def_cont':fit_prop_dic[inst][vis]['cond_def_cont_all'][isub]
                        }
                    for pl_loc in fixed_args['transit_pl'][inst][vis]:
                        prof_fit_dic[i_in][pl_loc] = {}
                        for prop_loc in mod_prop_dic[inst][vis][pl_loc]:prof_fit_dic[i_in][pl_loc][prop_loc] = mod_prop_dic[inst][vis][pl_loc][prop_loc][isub]
                np.savez_compressed(fit_dic['save_dir']+'IntrProf_fit_'+inst+'_'+vis+fixed_args['bin_mode'][inst][vis],data={'prof_fit_dic':prof_fit_dic},allow_pickle=True)

    #Post-processing    
    fit_dic['p_null'] = deepcopy(p_final)
    for par in [ploc for ploc in fit_dic['p_null'] if 'ctrst' in ploc]:fit_dic['p_null'][par] = 0.    
    com_joint_postproc(p_final,fixed_args,fit_dic,merged_chain,fit_prop_dic,gen_dic)
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
    #    - because exposures are specific to each visit, defined on different bins, and stored as objects we define the output table as :
    # chi = concatenate( exp, (obs(exp)-mod(exp))/err(exp)) ) or the equivalent with the covariance matrix
    #      so that the merit function will compare chi to a table of same size filled with 0 and with errors of 1 in the residual() function (where the condition to use covariance has been set to False for this purpose)
    #    - observed intrinsic profiles may have gaps, but due to the convolution the model must be calculated over the continuous table and then limited to fitted bins
    chi = np.zeros(0,dtype=float)
    if args['use_cov_eff']:
        for inst in args['inst_list']:
            for vis in args['inst_vis_list'][inst]:    
                for iexp in range(args['nexp_fit_all'][inst][vis]):
                    L_mat = scipy.linalg.cholesky_banded(args['cov'][inst][vis][iexp], lower=True)
                    res = args['flux'][inst][vis][iexp]-mod_dic[inst][vis][iexp]
                    cond_fit = args['cond_fit'][inst][vis][iexp]
                    res[~cond_fit] = 0.  
                    chi_exp  = scipy.linalg.blas.dtbsv(L_mat.shape[0]-1, L_mat, res, lower=True)
                    chi = np.append( chi, chi_exp[cond_fit] )                     

    else:
        for inst in args['inst_list']:
            for vis in args['inst_vis_list'][inst]:    
                for isub in range(args['nexp_fit_all'][inst][vis]):
                    cond_fit = args['cond_fit'][inst][vis][isub]
                    res = args['flux'][inst][vis][isub][cond_fit]-mod_dic[inst][vis][isub][cond_fit]
                    chi = np.append( chi, res/np.sqrt( args['cov'][inst][vis][isub][0][cond_fit]) )
         
    return chi




def joined_IntrProf(param,args):
    r"""**Model function: joined intrinsic stellar profiles**

    Defines the joined model for intrinsic stellar profiles.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    mod_dic = {}
    mod_prop_dic = {}
    coeff_line_dic = {}
    
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

            #Outputs
            if not args['fit']:

                #Coefficients describing the polynomial variation of spectral line properties as a function of the chosen coordinate
                if ('coeff_line' in args):coeff_line_dic[inst][vis] = args['coeff_line']  
                else:coeff_line_dic[inst][vis] = None              

                #Properties of all planet-occulted regions used to calculate spectral line profiles
                mod_prop_dic[inst][vis]={} 
                linevar_par_list = ['rv']
                if (len(args['linevar_par'])>0):linevar_par_list+=args['linevar_par'][inst][vis]
                for pl_loc in args['transit_pl'][inst][vis]:
                    mod_prop_dic[inst][vis][pl_loc]={}   
                    for prop_loc in linevar_par_list:mod_prop_dic[inst][vis][pl_loc][prop_loc] = np.zeros(len(args['idx_in_fit'][inst][vis]))*np.nan  
                
            #-----------------------------------------------------------
            #Calculate coordinates of occulted regions or use imported values
            system_param_loc,coord_pl,param_val = up_plocc_prop(inst,vis,args,param,args['transit_pl'][inst][vis],args['ph_fit'][inst][vis],args['coord_fit'][inst][vis])

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
                surf_prop_dic,spotocc_prop,surf_prop_dic_common = sub_calc_plocc_spot_prop([args['chrom_mode']],args_exp,args['par_list'],args['transit_pl'][inst][vis],system_param_loc,args['grid_dic'],args['system_prop'],param_val,coord_pl,[isub])
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
                    for pl_loc in args['transit_pl'][inst][vis]:                    
                        for prop_loc in mod_prop_dic[inst][vis][pl_loc]:mod_prop_dic[inst][vis][pl_loc][prop_loc][isub] = surf_prop_dic[args['chrom_mode']][pl_loc][prop_loc][0] 

    return mod_dic,coeff_line_dic,mod_prop_dic
























































def main_joined_ResProf(data_mode,data_dic,gen_dic,system_param,fit_prop_dic,theo_dic,plot_dic,coord_dic):    
    r"""**Joined residual profiles fits**

    Main routine to fit a given stellar surface property from planet-occulted regions with a joined model over instruments and visits.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    print('   > Fitting joined residual stellar CCFs, including spots')

    #Initializations
    fixed_args,fit_dic = init_joined_routines(data_mode,gen_dic,system_param,theo_dic,data_dic,fit_prop_dic)

    #Arguments to be passed to the fit function
    fixed_args.update({
        'ref_pl':fit_prop_dic['ref_pl'],
        'rout_mode':'ResProf',
        'func_prof_name':fit_prop_dic['func_prof_name'],
        'mode':fit_prop_dic['mode'], 
        'cen_bins' :{},
        'edge_bins':{},
        'dcen_bins' :{},
        'dim_exp':{},
        'ncen_bins':{},
        'cond_fit' :{},
        'cond_def' :{},
        'flux':{},
        'cov' :{},
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

    #Define master-out dictionary
    fixed_args['master_out']['multivisit_list']={}
    fixed_args['master_out']['idx_in_master_out']={}
    fixed_args['master_out']['master_out_tab']={}
    fixed_args['master_out']['scaled_data_paths']={}
    fixed_args['master_out']['multivisit_weights_total']={}
    fixed_args['master_out']['weights']={}
    fixed_args['master_out']['flux']={}
    fixed_args['master_out']['multivisit_flux']={}
    fixed_args['raw_DI_profs']={}

    #Profile generation
    fixed_args['idx_out']={}
    fixed_args['idx_in']={}
    fixed_args['DI_scaling_val']=data_dic['DI']['scaling_val']

    #Spot-crossing time supplement
    fixed_args['spot_crosstime_supp']={}
    fixed_args['update_crosstime']=True

    #Stellar surface coordinate required to calculate spectral line profiles
    #    - other required properties are automatically added in the sub_calc_plocc_spot_prop() function
    fixed_args['par_list']+=['line_prof']
    if fixed_args['mode']=='ana':
        if fit_prop_dic['dim_fit'] in ['abs_y_st','y_st2']:fixed_args['coord_line']='y_st'    
        else:fixed_args['coord_line']=fit_prop_dic['dim_fit']
        fixed_args['par_list']+=[fixed_args['coord_line']]
    else:
        fixed_args['par_list']+=['mu']

    #Activation of spectral conversion and resampling 
    cond_conv_st_prof_tab(theo_dic['rv_osamp_line_mod'],fixed_args,data_dic[data_dic['instrum_list'][0]]['type']) 
    
    #Construction of the fit tables
    #Initializing entries that will store the coordinates of the planets, of the spots, and the respective phase of the fitted exposures.
    for par in ['coord_fit','ph_fit']:fixed_args[par]={}
    
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
    for inst in np.intersect1d(data_dic['instrum_list'],list(fit_prop_dic['idx_in_fit'].keys())):    
        init_joined_routines_inst('ResProf',inst,fit_prop_dic,fixed_args)

        fixed_args['master_out']['idx_in_master_out'][inst]={}
        fixed_args['raw_DI_profs'][inst]={}
        fixed_args['master_out']['weights'][inst]={}
        fixed_args['master_out']['flux'][inst]={}
        fixed_args['master_out']['scaled_data_paths'][inst]={}
        fixed_args['idx_out'][inst]={}
        fixed_args['idx_in'][inst]={}
        fixed_args['spot_crosstime_supp'][inst]={}
        
        if (inst not in fixed_args['ref_pl']) and (fixed_args['ref_pl']!={}):fixed_args['ref_pl'][inst]={}
          
        for key in ['cen_bins','edge_bins','dcen_bins','cond_fit','flux','cov','cond_def','n_pc','dim_exp','ncen_bins']:fixed_args[key][inst]={}
        if len(fit_prop_dic['PC_model'])>0:fixed_args['eig_res_matr'][inst]={}
        fit_save['idx_trim_kept'][inst] = {}
        if (fixed_args['mode']=='ana') and (inst not in fixed_args['func_prof_name']):fixed_args['func_prof_name'][inst] = 'gauss'
        if (inst in fit_prop_dic['order']):iord_sel =  fit_prop_dic['order'][inst]
        else:iord_sel = 0

        #Setting continuum range to default if undefined
        if inst not in fit_prop_dic['cont_range']:fit_prop_dic['cont_range'] = data_dic['Res']['cont_range']
        cont_range = fit_prop_dic['cont_range'][inst][iord_sel]

        #Setting fitted range to default if undefined
        if inst not in fit_prop_dic['fit_range']:fit_prop_dic['fit_range'][inst] = data_dic['Res']['fit_range'][inst]
        
        #Setting trimmed range to default if undefined
        if (inst in fit_prop_dic['trim_range']):trim_range = fit_prop_dic['trim_range'][inst]
        else:trim_range = None 

        #Defining visits used in the master-out calculation 
        fixed_args['master_out']['multivisit_list'][inst]=[]
        if (inst in data_dic['Res']['vis_in_bin']) and (len(data_dic['Res']['vis_in_bin'][inst])>1):fixed_args['master_out']['multivisit_list'][inst]=data_dic['Res']['vis_in_bin'][inst]
        for multivisit in fixed_args['master_out']['multivisit_list'][inst]:
            if multivisit not in data_dic[inst]['visit_list']:stop('Problem: '+multivisit+' was selected for master-out calculation but is not used in the fit.')

        #Processing visit
        for vis_index, vis in enumerate(data_dic[inst]['visit_list']):
            init_joined_routines_vis(inst,vis,fit_prop_dic,fixed_args)

            #Visit is fitted
            if fixed_args['bin_mode'][inst][vis] is not None: 
                data_vis=data_dic[inst][vis]

                init_joined_routines_vis_fit('ResProf',inst,vis,fit_prop_dic,fixed_args,data_vis,gen_dic,data_dic,coord_dic)
                data_com = dataload_npz(data_dic[inst][vis]['proc_com_data_paths'])  

                #Master-out properties
                fixed_args['raw_DI_profs'][inst][vis]={} 
                fixed_args['master_out']['weights'][inst][vis]={}
                fixed_args['master_out']['idx_in_master_out'][inst][vis]=[]

                # - Indexes
                if (inst in data_dic['Res']['idx_in_bin']) and (vis in data_dic['Res']['idx_in_bin'][inst]):
                        if data_dic['Res']['idx_in_bin'][inst][vis]!={}:fixed_args['master_out']['idx_in_master_out'][inst][vis]=list(data_dic['Res']['idx_in_bin'][inst][vis])
        
                if len(fixed_args['master_out']['idx_in_master_out'][inst][vis])==0:stop('No exposures defined in visit '+vis+' for the master-out calculation.')

                # - Needed for weight calculation
                fixed_args['master_out']['scaled_data_paths'][inst][vis]={}
                if gen_dic['flux_sc']:fixed_args['master_out']['scaled_data_paths'][inst][vis] = data_dic[inst][vis]['scaled_DI_data_paths']
                else:fixed_args['master_out']['scaled_data_paths'][inst][vis] = None

                #Define in and out of transit exposures - needed for profile generation
                fixed_args['idx_out'][inst][vis]=gen_dic[inst][vis]['idx_out']
                fixed_args['idx_in'][inst][vis]=gen_dic[inst][vis]['idx_in']

                #Defining spot crossing time supplement
                fixed_args['spot_crosstime_supp'][inst][vis]=np.floor(fixed_args['coord_fit'][inst][vis]['bjd'][0])+2400000.

                #Defining reference planet if left undefined
                if fixed_args['ref_pl']=={}:fixed_args['ref_pl']={inst:{vis:data_dic[inst][vis]['transit_pl'][0]}}
                elif (vis_index==0) and (vis not in fixed_args['ref_pl'][inst]):fixed_args['ref_pl'][inst][vis]=data_dic[inst][vis]['transit_pl'][0]
                elif vis not in fixed_args['ref_pl'][inst]:fixed_args['ref_pl'][inst][vis]=fixed_args['ref_pl'][inst][data_dic[inst]['visit_list'][vis_index-1]]

                #Instrumental convolution
                if (inst not in fixed_args['FWHM_inst']):                
                    fixed_args['FWHM_inst'][inst] = get_FWHM_inst(inst,fixed_args,data_com['cen_bins'][iord_sel])

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
                idx_range_kept = np_where1D((data_com['edge_bins'][iord_sel,0:-1]>=min_trim_range) & (data_com['edge_bins'][iord_sel,1::]<=max_trim_range))
                ncen_bins = len(idx_range_kept)
                if ncen_bins==0:stop('Empty trimmed range') 

                fit_save['idx_trim_kept'][inst][vis] = idx_range_kept
                fixed_args['ncen_bins'][inst][vis] = ncen_bins  
                fixed_args['dim_exp'][inst][vis] = [1,ncen_bins] 

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
                for key in ['dcen_bins','cen_bins','edge_bins','cond_fit','flux','cov','cond_def']:fixed_args[key][inst][vis]=np.zeros(fixed_args['nexp_fit_all'][inst][vis],dtype=object)
                fit_prop_dic[inst][vis]['cond_def_fit_all']=np.zeros([fixed_args['nexp_fit_all'][inst][vis],ncen_bins],dtype=bool)
                fit_prop_dic[inst][vis]['cond_def_cont_all'] = np.zeros([fixed_args['nexp_fit_all'][inst][vis],ncen_bins],dtype=bool)  
                for isub,i_in in enumerate(fixed_args['idx_in_fit'][inst][vis]):

                    #Upload latest processed residual data
                    if fixed_args['bin_mode'][inst][vis]=='_bin':data_exp = dataload_npz(gen_dic['save_data_dir']+'Resbin_data/'+inst+'_'+vis+'_phase'+str(i_in))               
                    else:data_exp = dataload_npz(data_dic[inst][vis]['proc_Res_data_paths']+str(i_in))

                    #Trimming profile         
                    for key in ['cen_bins','flux','cond_def']:fixed_args[key][inst][vis][isub] = data_exp[key][iord_sel,idx_range_kept]
                    fixed_args['edge_bins'][inst][vis][isub] = data_exp['edge_bins'][iord_sel,idx_range_kept[0]:idx_range_kept[-1]+2]   
                    fixed_args['dcen_bins'][inst][vis][isub] = fixed_args['edge_bins'][inst][vis][isub][1::]-fixed_args['edge_bins'][inst][vis][isub][0:-1]  
                    fixed_args['cov'][inst][vis][isub] = data_exp['cov'][iord_sel][:,idx_range_kept]

                    #Oversampled line profile model table
                    if fixed_args['resamp']:resamp_st_prof_tab(inst,vis,isub,fixed_args,gen_dic,fixed_args['nexp_fit_all'][inst][vis],theo_dic['rv_osamp_line_mod'])

                    #Initializing ranges in the relevant rest frame
                    if len(cont_range)==0:fit_prop_dic[inst][vis]['cond_def_cont_all'][isub] = True    
                    else:
                        for bd_int in cont_range:fit_prop_dic[inst][vis]['cond_def_cont_all'][isub] |= (fixed_args['edge_bins'][inst][vis][isub][0:-1]>=bd_int[0]) & (fixed_args['edge_bins'][inst][vis][isub][1:]<=bd_int[1])         
                    if len(fit_prop_dic['fit_range'][inst][vis])==0:fit_prop_dic[inst][vis]['cond_def_fit_all'][isub] = True    
                    else:
                        for bd_int in fit_prop_dic['fit_range'][inst][vis]:fit_prop_dic[inst][vis]['cond_def_fit_all'][isub] |= (fixed_args['edge_bins'][inst][vis][isub][0:-1]>=bd_int[0]) & (fixed_args['edge_bins'][inst][vis][isub][1:]<=bd_int[1])        

                    #Accounting for undefined pixels
                    fit_prop_dic[inst][vis]['cond_def_cont_all'][isub] &= fixed_args['cond_def'][inst][vis][isub]           
                    fit_prop_dic[inst][vis]['cond_def_fit_all'][isub] &= fixed_args['cond_def'][inst][vis][isub]          
                    fit_dic['nx_fit']+=np.sum(fit_prop_dic[inst][vis]['cond_def_fit_all'][isub])
                    fixed_args['cond_fit'][inst][vis][isub] = fit_prop_dic[inst][vis]['cond_def_fit_all'][isub]

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
                            fit_prop_dic['mod_prop'][pc_name]={'vary':True,'guess':0.} 
                            if i_pc==0:fit_prop_dic['mod_prop'][pc_name]['bd'] = [-4.,4.]
                            elif i_pc==1:fit_prop_dic['mod_prop'][pc_name]['bd'] = [-3.,2.]
                            else:fit_prop_dic['mod_prop'][pc_name]['bd'] = [-1.,1.]
                            fit_prop_dic['priors'][pc_name]={'low':-100. ,'high':100.,'mod':'uf'}

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

    #Defining necessary master-out tables
    for inst in data_dic['instrum_list']:
        #Defining multi-visit master-out and weights
        if len(fixed_args['master_out']['multivisit_list'][inst])>0:
            fixed_args['master_out']['multivisit_flux'][inst]=np.zeros(len(fixed_args['master_out']['master_out_tab']['cen_bins']), dtype=float)
            fixed_args['master_out']['multivisit_weights_total'][inst]=np.zeros(len(fixed_args['master_out']['master_out_tab']['cen_bins']), dtype=float)
        
        for vis in data_dic[inst]['visit_list']:
                # Defining flux table
                fixed_args['master_out']['flux'][inst][vis]=np.zeros([len(fixed_args['master_out']['master_out_tab']['cen_bins'])], dtype=float)

    #Common data type
    for idx_inst,inst in enumerate(data_dic['instrum_list']):
        if idx_inst==0:fixed_args['type'] = data_dic[inst]['type']
        elif fixed_args['type'] != data_dic[inst]['type']:stop('Incompatible data types')
         
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
    fixed_args['fit_func'] = FIT_joined_ResProf
    fixed_args['mod_func'] = joined_ResProf
    fixed_args['inside_fit'] = True 

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
    merged_chain,p_final = com_joint_fits('ResProf',fit_dic,fixed_args,fit_prop_dic,gen_dic,data_dic,theo_dic,fit_prop_dic['mod_prop'])

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

    #Initializing best-fit stellar profile generation
    fixed_args = init_custom_DI_prof(fixed_args,gen_dic,data_dic['DI']['system_prop'],data_dic['DI']['spots_prop'],theo_dic,fixed_args['system_param']['star'],p_final)

    #Do 2 model calls, to have the model with and without planets
    #   - With planets
    mod_dic,coeff_line_dic,mod_prop_dic = fixed_args['mod_func'](p_final,fixed_args)
    #   - Without planets
    new_fixed_args = deepcopy(fixed_args)
    for inst in fixed_args['inst_list']:
        for vis in fixed_args['inst_vis_list'][inst]:
            new_fixed_args['transit_pl'][inst][vis]=[]
    mod_dic_nopl = new_fixed_args['mod_func'](p_final,new_fixed_args)[0]

    #Save best-fit properties
    #    - with same structure as fit to individual profiles 
    fit_save.update({'p_final':p_final,'coeff_line_dic':coeff_line_dic,'func_prof_name':fixed_args['func_prof_name'],'name_prop2input':fixed_args['name_prop2input'],'coord_line':fixed_args['coord_line'],'merit':fit_dic['merit'],
                     'pol_mode':fit_prop_dic['pol_mode'],'coeff_ord2name':fixed_args['coeff_ord2name'],'idx_in_fit':fixed_args['idx_in_fit'],'genpar_instvis':fixed_args['genpar_instvis'],'linevar_par':fixed_args['linevar_par']})
    if fixed_args['mode']=='ana':fit_save['func_prof'] = fixed_args['func_prof']
    
    #Information necessary for plotting the local profiles
    fit_save.update({'ph_fit':fixed_args['ph_fit'], 'system_prop':fixed_args['system_prop'], 'system_spot_prop':fixed_args['system_spot_prop'], 'grid_dic':fixed_args['grid_dic'],
                  'var_par_list':fixed_args['var_par_list'], 'fit_orbit':fixed_args['fit_orbit'], 'fit_RpRs':fixed_args['fit_RpRs'], 'fit_spot':fixed_args['fit_spot'], 'fit_spot_ang':fixed_args['fit_spot_ang']})
    np.savez(fit_dic['save_dir']+'Fit_results',data=fit_save,allow_pickle=True)
    if (plot_dic['map_BF_Res_prof']!='' or plot_dic['map_BF_Res_prof_re']!=''):
        for inst in fixed_args['inst_list']:
            for vis in fixed_args['inst_vis_list'][inst]:

                #Make directory to store best-fit results
                save_exp_dir = gen_dic['save_data_dir']+'Joined_fits/ResProf/'+fit_prop_dic['fit_mode']+'/'+inst+'/'+vis+'/'
                if (not os.path.exists(save_exp_dir)):os.makedirs(save_exp_dir) 

                for isub,i_in in enumerate(fixed_args['idx_in_fit'][inst][vis]):
                    prof_fit_dic={'fit_range':fit_prop_dic['fit_range'][inst][vis]}
                    if fixed_args['bin_mode'][inst][vis]=='_bin':prof_fit_dic['loc_data_corr_path'] = gen_dic['save_data_dir']+'Resbin_data/'+inst+'_'+vis+'_phase'          
                    else:prof_fit_dic['loc_data_corr_path'] = data_dic[inst][vis]['proc_Res_data_paths']
                    prof_fit_dic={
                        'edge_bins':np.array([fixed_args['edge_bins'][inst][vis][isub]]),
                        'flux':np.array([mod_dic[inst][vis][isub]]),
                        'flux_nopl':np.array([mod_dic_nopl[inst][vis][isub]]),
                        'cond_def_fit':np.array([fit_prop_dic[inst][vis]['cond_def_fit_all'][isub]]),
                        'cond_def_cont':fit_prop_dic[inst][vis]['cond_def_cont_all'][isub]
                        }
                    for pl_loc in fixed_args['transit_pl'][inst][vis]:
                        prof_fit_dic[pl_loc] = {}
                        if np.abs(fixed_args['coord_fit'][inst][vis][pl_loc]['ecl'][isub])!=1:  
                            for prop_loc in mod_prop_dic[inst][vis][pl_loc]:prof_fit_dic[pl_loc][prop_loc] = mod_prop_dic[inst][vis][pl_loc][prop_loc][isub]
                    for spot in fixed_args['transit_sp'][inst][vis]:
                        prof_fit_dic[spot] = {}
                        if fixed_args['coord_fit'][inst][vis][spot]['is_visible'][1,i_in]:
                            for prop_loc in mod_prop_dic[inst][vis][spot]:prof_fit_dic[spot][prop_loc] = mod_prop_dic[inst][vis][spot][prop_loc][isub]

                    np.savez_compressed(save_exp_dir+'BestFit'+fixed_args['bin_mode'][inst][vis]+'_'+str(isub),data=prof_fit_dic,allow_pickle=True)

    #Post-processing    
    fit_dic['p_null'] = deepcopy(p_final)
    for par in [ploc for ploc in fit_dic['p_null'] if 'ctrst' in ploc]:fit_dic['p_null'][par] = 0.
    com_joint_postproc(p_final,fixed_args,fit_dic,merged_chain,fit_prop_dic,gen_dic)
    
    print('     ----------------------------------')
    
    return None

    


def FIT_joined_ResProf(param,x_tab,args=None):
    r"""**Fit function: joined residual stellar profiles**

    Calls corresponding model function for optimization

    Args:
        TBD
    
    Returns:
        TBD
    """
    
    #Models over fitted spectral ranges
    mod_dic=joined_ResProf(param,args)[0]

    #Merit table
    #    - because exposures are specific to each visit, defined on different bins, and stored as objects we define the output table as :
    # chi = concatenate( exp, (obs(exp)-mod(exp))/err(exp)) ) or the equivalent with the covariance matrix
    #      so that the merit function will compare chi to a table of same size filled with 0 and with errors of 1 in the residual() function (where the condition to use covariance has been set to False for this purpose)
    #    - observed intrinsic profiles may have gaps, but due to the convolution the model must be calculated over the continuous table and then limited to fitted bins
    chi = np.zeros(0,dtype=float)
    if args['use_cov_eff']:
        for inst in args['inst_list']:
            for vis in args['inst_vis_list'][inst]:    
                for iexp in range(args['nexp_fit_all'][inst][vis]):
                    L_mat = scipy.linalg.cholesky_banded(args['cov'][inst][vis][iexp], lower=True)
                    res = args['flux'][inst][vis][iexp]-mod_dic[inst][vis][iexp]
                    cond_fit = args['cond_fit'][inst][vis][iexp]
                    res[~cond_fit] = 0.  
                    chi_exp  = scipy.linalg.blas.dtbsv(L_mat.shape[0]-1, L_mat, res, lower=True)
                    chi = np.append( chi, chi_exp[cond_fit] ) 
    else:
        for inst in args['inst_list']:
            for vis in args['inst_vis_list'][inst]:    
                for isub in range(args['nexp_fit_all'][inst][vis]):
                    cond_fit = args['cond_fit'][inst][vis][isub]
                    res = args['flux'][inst][vis][isub][cond_fit]-mod_dic[inst][vis][isub][cond_fit]
                    chi = np.append( chi, res/np.sqrt( args['cov'][inst][vis][isub][0][cond_fit]) )

    return chi





   
def joined_ResProf(param,args):
    r"""**Model function: joined residual profiles**

    Defines the joined model for residual profiles. This is done in three steps
    
     1. We calculate all DI profiles of the star (fitted exposures + exposures that contributed to the master-out), and we scale 
        them at the same value as after the `Broadband flux Scaling module`.

     2. We compute the master out, with same weights as those used in the corresponding module.
    
     3. We extract residual profiles as :math:`F_\mathrm{res} = F_\mathrm{out} - F_\mathrm{sc}`   

    Args:
        TBD
    
    Returns:
        TBD
    
    """
    mod_dic = {}
    mod_prop_dic = {}
    coeff_line_dic = {}

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
            if not args['fit']:

                #Coefficients describing the polynomial variation of spectral line properties as a function of the chosen coordinate
                if ('coeff_line' in args):coeff_line_dic[inst][vis] = args['coeff_line']  
                else:coeff_line_dic[inst][vis] = None              

                #Properties of all planet-occulted and spotted regions used to calculate spectral line profiles
                mod_prop_dic[inst][vis]={} 
                linevar_par_list = ['rv']
                if (len(args['linevar_par'])>0):linevar_par_list+=args['linevar_par'][inst][vis]
                for pl_loc in args['transit_pl'][inst][vis]:
                    mod_prop_dic[inst][vis][pl_loc]={}   
                    for prop_loc in linevar_par_list:mod_prop_dic[inst][vis][pl_loc][prop_loc] = np.zeros(len(args['idx_in_fit'][inst][vis]))*np.nan  
                for spot in args['transit_sp'][inst][vis]:
                    mod_prop_dic[inst][vis][spot]={}   
                    for prop_loc in linevar_par_list:mod_prop_dic[inst][vis][spot][prop_loc] = np.zeros(len(args['idx_in_fit'][inst][vis]))*np.nan  

            #-----------------------------------------------------------
            #Updating the stellar grid -- necessary since we need to determine the 2D grid of booleans
            #on the most up to date version of the stellar grid
            if args['fit'] and args['var_star_grid']:
                param_up={}
                if isinstance(param,lmfit.parameter.Parameters):
                    for par in param:param_up[par]=param[par].value
                else:param_up=param
                up_model_star(args, param_up)
                args['Fsurf_grid_spec'] = theo_intr2loc(args['grid_dic'],args['system_prop'],args,args['ncen_bins'][inst][vis],args['grid_dic']['nsub_star'])     

            #Figuring out which cells of the stellar grid are never spotted or planet-occulted

            #Initialize a 2D grid (which is going to be a 1D array) that will contain booleans telling us which stellar grid cells 
            #are never planet-occulted or spotted over all the exposures (True = occulted or spotted, False = quiet)
            unquiet_star_grid = np.zeros(args['grid_dic']['nsub_star'], dtype=bool)

            #Retrieve updated coordinates of occulted and spotted regions or use imported values
            system_param_loc,coord_pl_sp,param_val = up_plocc_prop(inst,vis,args,param,args['transit_pl'][inst][vis],args['ph_fit'][inst][vis],args['coord_fit'][inst][vis],transit_spots=args['transit_sp'][inst][vis])

            for isub,i_in in enumerate(args['idx_in_fit'][inst][vis]): 

                #Figure out which cells are spotted
                spotted_star_grid=np.zeros(args['grid_dic']['nsub_star'], dtype=bool)
                for spot in args['transit_sp'][inst][vis]:
                    if np.sum(coord_pl_sp[spot]['is_visible'][:, isub]):
                        mini_spot_dic = {}
                        mini_spot_dic['x_sky_exp_start'],mini_spot_dic['x_sky_exp_center'],mini_spot_dic['x_sky_exp_end']=coord_pl_sp[spot]['x_sky_exp'][:, isub]
                        mini_spot_dic['y_sky_exp_start'],mini_spot_dic['y_sky_exp_center'],mini_spot_dic['y_sky_exp_end']=coord_pl_sp[spot]['y_sky_exp'][:, isub]
                        mini_spot_dic['ang_rad']=coord_pl_sp[spot]['ang_rad'][isub]
                        mini_spot_dic['cos_long_exp_start'],mini_spot_dic['cos_long_exp_center'],mini_spot_dic['cos_long_exp_end']=coord_pl_sp[spot]['cos_long_exp'][:, isub]
                        mini_spot_dic['sin_long_exp_start'],mini_spot_dic['sin_long_exp_center'],mini_spot_dic['sin_long_exp_end']=coord_pl_sp[spot]['sin_long_exp'][:, isub]
                        mini_spot_dic['cos_lat_exp_start'],mini_spot_dic['cos_lat_exp_center'],mini_spot_dic['cos_lat_exp_end']=coord_pl_sp[spot]['cos_lat_exp'][:, isub]
                        mini_spot_dic['sin_lat_exp_start'],mini_spot_dic['sin_lat_exp_center'],mini_spot_dic['sin_lat_exp_end']=coord_pl_sp[spot]['sin_lat_exp'][:, isub]
                        _, spot_spotted_star_grid = calc_spotted_tiles(mini_spot_dic, args['grid_dic']['x_st_sky'], args['grid_dic']['y_st_sky'], args['grid_dic']['z_st_sky'], args['grid_dic'], system_param_loc['star'])
                        spotted_star_grid |= spot_spotted_star_grid
        
                #Figure out which cells are planet-occulted
                plocced_star_grid=np.zeros(args['grid_dic']['nsub_star'], dtype=bool)
                for pl_loc in args['transit_pl'][inst][vis]:
                    if np.abs(coord_pl_sp[pl_loc]['ecl'][isub])!=1:
                        mini_pl_dic = {}
                        mini_pl_dic['x_orb_exp']=[coord_pl_sp[pl_loc]['st_pos'][0, isub], coord_pl_sp[pl_loc]['cen_pos'][0, isub], coord_pl_sp[pl_loc]['end_pos'][0, isub]]
                        mini_pl_dic['y_orb_exp']=[coord_pl_sp[pl_loc]['st_pos'][1, isub], coord_pl_sp[pl_loc]['cen_pos'][1, isub], coord_pl_sp[pl_loc]['end_pos'][1, isub]]
                        mini_pl_dic['RpRs']=args['system_prop']['achrom'][pl_loc][0]
                        if ('lambda_rad__pl'+pl_loc in args['genpar_instvis']):lamb_name = 'lambda_rad__pl'+pl_loc+'__IS'+inst+'_VS'+vis 
                        else:lamb_name = 'lambda_rad__pl'+pl_loc 
                        mini_pl_dic['lambda']=param_val[lamb_name]
                        pl_plocced_star_grid = calc_plocced_tiles(mini_pl_dic, args['grid_dic']['x_st_sky'], args['grid_dic']['y_st_sky'])
                        plocced_star_grid |= pl_plocced_star_grid

                #Update the global 2D quiet star grid
                unquiet_star_grid |= (spotted_star_grid | plocced_star_grid)

            #-----------------------------------------------------------
            #Variable line model for each exposure 
            #    - the intrinsic stellar line profile is convolved with the LSF kernel specific to each instrument
            #    - we assume a flat continuum, set after the PC component so that intrinsic profiles models later on can be defined with a continuum unity
            #-----------------------------------------------------------
            mod_dic[inst][vis]=np.zeros(args['nexp_fit_all'][inst][vis],dtype=object)

            #Defining the base stellar profile
            # # Table for model calculation - wavelength table of the exposure considered
            args_DI = def_st_prof_tab(inst,vis,0,args)

            # # Initializing broadband scaling of intrinsic profiles into local profiles
            if not args['fit']:
                args_DI['Fsurf_grid_spec'] = theo_intr2loc(args_DI['grid_dic'],args_DI['system_prop'],args_DI,args_DI['ncen_bins'],args_DI['grid_dic']['nsub_star']) 

            base_DI_prof = custom_DI_prof(param_val,None,args=args_DI,unquiet_star=unquiet_star_grid)[0]

            #Making profiles for each exposure
            for isub,i_in in enumerate(args['idx_in_fit'][inst][vis]):

                #Table for model calculation - wavelength table of the exposure considered
                args_exp = def_st_prof_tab(inst,vis,isub,args)

                #Model DI profile for current exposure accounting for deviations from the nominal profile - on the wavelength table of the exposure considered
                new_args_exp=deepcopy(args_exp)
                new_args_exp['conv2intr']=False
                surf_prop_dic,surf_prop_dic_sp,_ = sub_calc_plocc_spot_prop([args['chrom_mode']],new_args_exp,args['par_list'],args['transit_pl'][inst][vis],system_param_loc,args['grid_dic'],args['system_prop'],param_val,coord_pl_sp,[isub],system_spot_prop_in=args['system_spot_prop'])
                sp_line_model = base_DI_prof - surf_prop_dic[args['chrom_mode']]['line_prof'][:,0] - surf_prop_dic_sp[args['chrom_mode']]['line_prof'][:,0]

                #Properties of all planet-occulted and spotted regions used to calculate spectral line profiles
                # - Since we are analyzing residual profiles, we have to check if the planets/spots are in the exposure considered.
                # - If this is not the case, an entry for them in the surf_prop_dic/surf_prop_dic_sp won't be initialized
                if not args['fit']:
                    for pl_loc in args['transit_pl'][inst][vis]:  
                        if np.abs(coord_pl_sp[pl_loc]['ecl'][isub])!=1:                
                            for prop_loc in mod_prop_dic[inst][vis][pl_loc]:
                                mod_prop_dic[inst][vis][pl_loc][prop_loc][isub] = surf_prop_dic[args['chrom_mode']][pl_loc][prop_loc][0]

                    for spot in args['transit_sp'][inst][vis]:
                        if np.sum(coord_pl_sp[spot]['is_visible'][:, isub]):
                            for prop_loc in mod_prop_dic[inst][vis][spot]:
                                mod_prop_dic[inst][vis][spot][prop_loc][isub] = surf_prop_dic_sp[args['chrom_mode']][spot][prop_loc][0]

                #Convolve model profiles to instrument resolution
                conv_line_model = convol_prof(sp_line_model,args_exp['cen_bins'],args['FWHM_inst'][inst])

                #Set negative flux values to null
                conv_line_model[conv_line_model<base_DI_prof[0]-1] = 0.
                                
                #Store the model DI profiles for calculation of the residual profiles later
                args['raw_DI_profs'][inst][vis][isub] = conv_line_model

                #Loop over exposures contributing to the master-out
                if i_in in args['master_out']['idx_in_master_out'][inst][vis]:
                    
                    #Storing the index of the exposure considered in the array of master-out indices
                    master_isub = np.where(args['master_out']['idx_in_master_out'][inst][vis]==i_in)[0][0]

                    #Re-sample model DI profile on a common grid
                    resamp_line_model = bind.resampling(args['master_out']['master_out_tab']['edge_bins'],args['edge_bins'][inst][vis][isub],conv_line_model,kind=args['master_out']['master_out_tab']['resamp_mode'])

                    #Making weights for the master-out
                    raw_weights=weights_bin_prof(range(args['master_out']['nord']), args['master_out']['scaled_data_paths'][inst][vis],inst,vis,args['master_out']['corr_Fbal'],args['master_out']['corr_FbalOrd'],\
                                                        args['master_out']['save_data_dir'],args['type'],args['master_out']['nord'],isub,'DI',args['type'],args['dim_exp'][inst][vis],None,\
                                                        None,np.array([args['cen_bins'][inst][vis][isub]]),args['coord_fit'][inst][vis]['t_dur'][isub],np.array([conv_line_model]),\
                                                        np.array([args['cov'][inst][vis][isub]]),ref_val=base_DI_prof[0]-1, bdband_flux_sc=args['master_out']['flux_sc'])[0]

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

            # Defined bins in binned spectrum
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

    #Building residual profiles
        for vis in args['inst_vis_list'][inst]:
            for isub,i_in in enumerate(args['idx_in_fit'][inst][vis]):
                
                #Retrieving the master-out flux in the case where we combine multiple visits
                if vis in args['master_out']['multivisit_list'][inst]:master_out_flux=args['master_out']['multivisit_flux'][inst]
                else:master_out_flux=args['master_out']['flux'][inst][vis]

                #Re-sample master on table of the exposure considered
                resamp_master = bind.resampling(args['edge_bins'][inst][vis][isub],args['master_out']['master_out_tab']['edge_bins'],master_out_flux, kind=args['master_out']['master_out_tab']['resamp_mode'])

                #Calculate the residual profile on the wavelength table of the exposure considered (Isn't this gonna be an issue when making the residual map?)
                mod_dic[inst][vis][isub] = resamp_master - args['raw_DI_profs'][inst][vis][isub]

                #Add PC noise model
                #    - added to the convolved profiles since PC are derived from observed data
                if args['n_pc'][inst][vis] is not None:
                    for i_pc in range(args['n_pc'][inst][vis]):mod_dic[inst][vis][isub]+=param_val[args['name_prop2input']['aPC_idxin'+str(i_in)+'_ord'+str(i_pc)+'__IS'+inst+'_VS'+vis]]*args['eig_res_matr'][inst][vis][i_in][i_pc]

    return mod_dic,coeff_line_dic,mod_prop_dic

    
    
    
    