#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from pathos.multiprocessing import cpu_count
from copy import deepcopy
import bindensity as bind
from os import makedirs
from os.path import exists as path_exist
import glob
from astropy.io import fits
from dace_query.spectroscopy import Spectroscopy
from scipy.interpolate import CubicSpline
from ..ANTARESS_analysis.ANTARESS_model_prof import calc_macro_ker_anigauss,calc_macro_ker_rt 
from ..ANTARESS_grids.ANTARESS_star_grid import model_star
from ..ANTARESS_grids.ANTARESS_occ_grid import occ_region_grid,sub_calc_plocc_ar_prop,calc_plocc_ar_prop,retrieve_ar_prop_from_param,generate_ar_prop
from ..ANTARESS_grids.ANTARESS_prof_grid import init_custom_DI_prof,custom_DI_prof,theo_intr2loc,gen_theo_atm,var_stellar_prop
from ..ANTARESS_grids.ANTARESS_coord import calc_mean_anom_TR,calc_Kstar,calc_tr_contacts,calc_rv_star,get_timeorbit,coord_expos,coord_expos_ar
from ..ANTARESS_analysis.ANTARESS_inst_resp import return_pix_size,def_st_prof_tab,cond_conv_st_prof_tab,conv_st_prof_tab,get_FWHM_inst,resamp_st_prof_tab,return_spec_nord,flag_err_inst,return_cen_wav_ord
from ..ANTARESS_analysis.ANTARESS_ana_comm import par_formatting_inst_vis
from ..ANTARESS_general.minim_routines import par_formatting
from ..ANTARESS_corrections.ANTARESS_sp_reduc import red_sp_data_instru
from ..ANTARESS_analysis.ANTARESS_joined_star import joined_Star_ana
from ..ANTARESS_analysis.ANTARESS_joined_atm import joined_Atm_ana
from ..ANTARESS_plots.ANTARESS_plots_all import ANTARESS_plot_functions
from ..ANTARESS_corrections.ANTARESS_calib import calc_gcal
from ..ANTARESS_process.ANTARESS_plocc_spec import def_in_plocc_profiles,def_diff_profiles,eval_diff_profiles
from ..ANTARESS_conversions.ANTARESS_masks_gen import def_masks
from ..ANTARESS_conversions.ANTARESS_conv import DI_CCF_from_spec,DiffIntr_CCF_from_spec,Atm_CCF_from_spec,conv_2D_to_1D_spec
from ..ANTARESS_conversions.ANTARESS_binning import process_bin_prof
from ..ANTARESS_corrections.ANTARESS_detrend import detrend_prof,pc_analysis
from ..ANTARESS_process.ANTARESS_data_process import align_profiles,rescale_profiles,extract_diff_profiles,extract_intr_profiles,extract_pl_profiles,EvE_outputs 
from ..ANTARESS_analysis.ANTARESS_ana_comm import MAIN_single_anaprof
from ..ANTARESS_conversions.ANTARESS_sp_cont import process_spectral_cont
from ..ANTARESS_general.utils import air_index,dataload_npz,gen_specdopshift,stop,np_where1D,closest,datasave_npz,def_edge_tab,check_data,npint,path_singslash
from ..ANTARESS_general.constant_data import Rsun,Rjup,c_light,G_usi,Msun,AU_1


def ANTARESS_settings_overwrite(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic,input_dic):
    r"""**ANTARESS settings overwrite.**
    
    Overwrites ANTARESS default settings with custom inputs.  
    
    Args:
        TBD
    
    Returns:
        None
    
    """     
  
    #Overwriting full dictionaries
    if 'gen_dic' in input_dic['settings']:gen_dic.update(input_dic['settings']['gen_dic'])
    if 'mock_dic' in input_dic['settings']:mock_dic.update(input_dic['settings']['mock_dic'])
    if 'data_dic' in input_dic['settings']:
        for key in ['DI','Intr']:
            if key in input_dic['settings']['data_dic']:data_dic[key].update(input_dic['settings']['data_dic'][key])
    if 'glob_fit_dic' in input_dic['settings']:
        for key in ['DIProp','IntrProf','IntrProp','DiffProf']:
            if key in input_dic['settings']['glob_fit_dic']:glob_fit_dic[key].update(input_dic['settings']['glob_fit_dic'][key])
    if 'plot_dic' in input_dic['settings']:plot_dic.update(input_dic['settings']['plot_dic'])
    if 'detrend_prof_dic' in input_dic['settings']:detrend_prof_dic.update(input_dic['settings']['detrend_prof_dic'])
    
    #Overwriting specific fields
    if ('orders4ccf' in input_dic) and (len(input_dic['orders4ccf'])>0):
        for inst in input_dic['orders4ccf']:gen_dic['orders4ccf'][inst] = input_dic['orders4ccf'][inst]
    
    return None
    


def ANTARESS_main(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic,system_param,input_dic,custom_plot_settings):
    r"""**Main ANTARESS function.**

    Runs ANTARESS workflow. The pipeline is defined as modules than can be run independently. Each module takes as input the datasets produced by earlier modules, transforms or 
    analyzes them, and saves the outputs to disk. This approach allows the user to re-run the pipeline from any module, which is useful when several flow 
    options are available. It is even critical with large datasets such as the ones produced by ESPRESSO, which can take several hours to process with a 
    given module. Finally, this approach also allows a user to retrieve data at any stage of the process flow for external use.

    Args:
        TBD

    Returns:
        None
    """

    ##############################################################################
    #Initializations
    ##############################################################################
    coord_dic,data_prop = init_gen(data_dic,mock_dic,gen_dic,system_param,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic)
    
    ####################################################################################################################
    #Processing datasets for each visit of each instrument
    #    - binned visits are processed at the end
    ####################################################################################################################
    if gen_dic['sequence'] not in ['system_view']:
        for inst in data_dic['instrum_list']:
            print('')
            print('-----------------------')
            print('Processing instrument :',inst)        
            print('-----------------------')
    
            #Initialize instrument tables and dictionaries
            init_inst(mock_dic,inst,gen_dic,data_dic,theo_dic,data_prop,coord_dic,system_param,plot_dic)

            #Spectral data
            if ('spec' in data_dic[inst]['type']):

                #Estimating instrumental calibration
                if gen_dic['gcal']:
                    calc_gcal(gen_dic,data_dic,inst,plot_dic,coord_dic,data_prop)

                #Global corrections of spectral data
                #    - performed before the loop on individual visits because some corrections exploit information from all visits and require the full range of the data        
                red_sp_data_instru(inst,data_dic,plot_dic,gen_dic,data_prop,coord_dic,system_param)
    
            #-------------------------------------------------        
            #Processing all visits for current instrument
            for vis in data_dic[inst]['visit_list']:
                print('  -----------------')
                print('  Processing visit: '+vis) 
                print('  -----------------')
    
                #Initialization of visit properties
                init_vis(data_prop,data_dic,vis,coord_dic,inst,system_param,gen_dic)             
                
                #-------------------------------------------------  
                #Processing disk-integrated stellar profiles
                data_type_gen = 'DI'
                #------------------------------------------------- 
              
                #Spectral detrending   
                if gen_dic['detrend_prof'] and (detrend_prof_dic['full_spec']):
                    detrend_prof(detrend_prof_dic,data_dic,coord_dic,inst,vis,data_dic,data_prop,gen_dic,plot_dic)

                #Converting stellar spectra into CCFs
                if gen_dic[data_type_gen+'_CCF']:
                    DI_CCF_from_spec(inst,vis,data_dic,gen_dic)
                  
                #Single line detrending    
                if gen_dic['detrend_prof'] and (not detrend_prof_dic['full_spec']):
                    detrend_prof(detrend_prof_dic,data_dic,coord_dic,inst,vis,data_dic,data_prop,gen_dic,plot_dic)
    
                #Calculating theoretical properties of the planet-occulted and/or active regions 
                if gen_dic['theoPlOcc'] and ( (len(data_dic[inst][vis]['studied_pl'])>0) or (len(data_dic[inst][vis]['studied_ar'])>0) ):
                    calc_plocc_ar_prop(system_param,gen_dic,theo_dic,coord_dic,inst,vis,data_dic,calc_pl_atm=gen_dic['calc_pl_atm'],ar_dic=theo_dic)
                    
                #Analyzing original disk-integrated profiles
                if gen_dic['fit_'+data_type_gen]:
                    MAIN_single_anaprof('',data_type_gen+'orig',data_dic,gen_dic,inst,vis,coord_dic,theo_dic,plot_dic,system_param['star'])
    
                #Aligning disk-integrated profiles to star rest frame
                if (gen_dic['align_'+data_type_gen]):
                    align_profiles(data_type_gen,data_dic,inst,vis,gen_dic,coord_dic)

                #Rescaling profiles to their correct flux level  
                #    - the module is used to calculate light curves even if no scaling is needed                
                if gen_dic['flux_sc']:                   
                    rescale_profiles(data_dic[inst],inst,vis,data_dic,coord_dic,coord_dic[inst][vis]['t_dur_d'],gen_dic,plot_dic,system_param,theo_dic,ar_dic=theo_dic)   
             
                #Calculating master spectrum of the disk-integrated star used in weighted averages and continuum-normalization
                if gen_dic['DImast_weight']:              
                    process_bin_prof('',data_type_gen,gen_dic,inst,vis,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,masterDIweigh=True,ar_dic=theo_dic)
    
                #Converting 2D disk-integrated profiles into 1D, and analyzing them
                if gen_dic['spec_1D']:                
                    conv_2D_to_1D_gen_functions(data_type_gen,data_dic,inst,vis,gen_dic,coord_dic,theo_dic,plot_dic,system_param)
     
                #Processing binned disk-integrated profiles
                if gen_dic['bin']:
                    bin_gen_functions(data_type_gen,'',inst,gen_dic,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,vis=vis)
    
                #--------------------------------------------------------------------------------------------------
                #Processing differential profiles
                data_type_gen = 'Diff'
                #--------------------------------------------------------------------------------------------------
    
                #Extracting differential profiles
                if (gen_dic['diff_data']):
                    extract_diff_profiles(gen_dic,data_dic,inst,vis,data_prop,coord_dic)
    
                #Converting differential spectra into CCFs
                if gen_dic[data_type_gen+'_CCF']:
                    DiffIntr_CCF_from_spec(data_type_gen,inst,vis,data_dic,gen_dic)    
    
                #Converting and analyzing 2D differential spectra into 1D
                if gen_dic['spec_1D']:                
                    conv_2D_to_1D_gen_functions(data_type_gen,data_dic,inst,vis,gen_dic,coord_dic,theo_dic,plot_dic,system_param)
    
                #--------------------------------------------------------------------------------------------------
                #Processing intrinsic stellar profiles
                data_type_gen = 'Intr'
                #--------------------------------------------------------------------------------------------------
    
                #Extracting intrinsic stellar profiles
                if gen_dic['intr_data']:
                    extract_intr_profiles(data_dic,gen_dic,inst,vis,system_param['star'],coord_dic,theo_dic,plot_dic)
            
                #Converting out-of-transit differential and intrinsic spectra into CCFs
                if gen_dic[data_type_gen+'_CCF']:
                    DiffIntr_CCF_from_spec(data_type_gen,inst,vis,data_dic,gen_dic,data_type_gen)
                      
                #Applying PCA to out-of transit differential profiles
                if (gen_dic['pca_ana']):
                    pc_analysis(gen_dic,data_dic,inst,vis,data_prop,coord_dic)
    
                #Analyzing intrinsic stellar profiles in the star rest frame
                if gen_dic['fit_'+data_type_gen]:
                    MAIN_single_anaprof('',data_type_gen+'orig',data_dic,gen_dic,inst,vis,coord_dic,theo_dic,plot_dic,system_param['star'])
                
                #Aligning intrinsic stellar profiles to their local rest frame
                if gen_dic['align_'+data_type_gen]: 
                    align_profiles(data_type_gen,data_dic,inst,vis,gen_dic,coord_dic)
    
                #Converting and analyzing 2D out-of-transit differential and intrinsic spectra into 1D, and analyzing them
                if gen_dic['spec_1D']:                
                    conv_2D_to_1D_gen_functions(data_type_gen,data_dic,inst,vis,gen_dic,coord_dic,theo_dic,plot_dic,system_param)
       
                #Processing binned intrinsic profiles
                if gen_dic['bin']:
                    bin_gen_functions(data_type_gen,'',inst,gen_dic,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,vis=vis)
    
                #Saving EvE outputs
                if gen_dic['EvE_outputs']:
                    EvE_outputs()    
    
                #Building estimates for planet-occulted stellar profiles in in-transit exposures
                if gen_dic['loc_prof_est']:
                    def_in_plocc_profiles(inst,vis,gen_dic,data_dic,data_prop,coord_dic,system_param,theo_dic,glob_fit_dic,plot_dic)
    
                #Building estimates for differential profiles in all exposures
                #    - in-transit profiles include planet-occulted and active region stellar profiles
                if gen_dic['diff_prof_est']:
                    def_diff_profiles(inst,vis,gen_dic,data_dic,data_prop,coord_dic,system_param,theo_dic,glob_fit_dic,plot_dic)

                #Correcting exposures from (occulted/non-occulted) active region contamination using the differential profile estimates
                if gen_dic['corr_diff']:
                    eval_diff_profiles(inst, vis, gen_dic, data_dic,data_prop,coord_dic,system_param,theo_dic,glob_fit_dic,plot_dic,'corr')

                #Building best-fit differential profile time series
                if gen_dic['eval_bestfit']:
                    eval_diff_profiles(inst, vis, gen_dic, data_dic,data_prop,coord_dic,system_param,theo_dic,glob_fit_dic,plot_dic,'bestfit')
            
                #--------------------------------------------------------------------------------------------------
                #Processing atmospheric profiles
                data_type_gen = 'Atm'
                #--------------------------------------------------------------------------------------------------
    
                #Extracting atmospheric profiles
                if gen_dic['pl_atm']:
                    extract_pl_profiles(data_dic,inst,vis,gen_dic)
    
                #Converting atmospheric spectra into CCFs
                if gen_dic[data_type_gen+'_CCF'] and ('spec' in data_dic[inst][vis]['type']):
                    Atm_CCF_from_spec(inst,vis,data_dic,gen_dic)

                #Fitting atmospheric profiles in the star rest frame
                if gen_dic['fit_'+data_type_gen]:
                    MAIN_single_anaprof('',data_type_gen+'orig',data_dic,gen_dic,inst,vis,coord_dic,theo_dic,plot_dic,system_param['star'])
          
                #Aligning atmospheric profiles to the planet rest frame
                if gen_dic['align_'+data_type_gen]:   
                    align_profiles(data_type_gen,data_dic,inst,vis,gen_dic,coord_dic)      
    
                #Converted 2D atmospheric profiles into 1D, and analyzing them
                if gen_dic['spec_1D']:                
                    conv_2D_to_1D_gen_functions(data_type_gen,data_dic,inst,vis,gen_dic,coord_dic,theo_dic,plot_dic,system_param)
    
                #Processing binned atmospheric profiles
                if gen_dic['bin']:
                    bin_gen_functions(data_type_gen,'',inst,gen_dic,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,vis=vis)
    
            ### end of visits 
    
            #Update instrument properties   
            update_inst(data_dic,inst,gen_dic)        
    
            #Processing binned profiles over multiple visits
            #    - beware that data from different visits should be comparable to be binned 
            #      this is not the case, e.g, with blazed 2D spectra or if the stellar line shape changed 
            if (data_dic[inst]['n_visits_inst']>1) and (gen_dic['binmultivis']):
                print('  --------------------------')
                print('  Processing combined visits')  
                print('  --------------------------')
                for data_type_gen in ['DI','Intr','Atm']:
                    bin_gen_functions(data_type_gen,'multivis',inst,gen_dic,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic)
    
        ### end of instruments  
    
        #Update generic properties   
        if gen_dic['spec_1D'] or gen_dic['CCF_from_sp']:
            update_gen(data_dic,gen_dic) 
    
        ####################################################################################################################
        #Call to analysis function over combined instruments and/or visits
        ####################################################################################################################
        if gen_dic['joined_ana']:
            print('-------------------------------')
            print('Analyzing combined datasets')        
            print('-------------------------------')
            
            #Wrap-up function to fit stellar profiles and their properties
            if gen_dic['fit_DIProp'] or gen_dic['fit_IntrProf'] or gen_dic['fit_IntrProp'] or gen_dic['fit_DiffProf'] :
                joined_Star_ana(glob_fit_dic,system_param,theo_dic,data_dic,gen_dic,plot_dic,coord_dic,data_prop)
        
            #Wrap-up function to fit atmospheric profiles and their properties
            if gen_dic['fit_AtmProf'] or gen_dic['fit_AtmProp']:
                joined_Atm_ana(gen_dic)

    ##############################################################################
    #Call to plot functions
    ##############################################################################
    if gen_dic['plots_on']:
        ANTARESS_plot_functions(system_param,plot_dic,data_dic,gen_dic,coord_dic,theo_dic,data_prop,glob_fit_dic,mock_dic,input_dic,custom_plot_settings)

    return None
    
    


def conv_2D_to_1D_gen_functions(data_type_gen,data_dic,inst,vis,gen_dic,coord_dic,theo_dic,plot_dic,system_param):
    """**Wrap-up function for 2D->1D datasets.**

    Args:
        TBD:
    
    Returns:
        None
    
    """ 
    #Processing 2D spectra into new 1D spectra
    if gen_dic['spec_1D_'+data_type_gen] and (data_dic[inst][vis]['type']=='spec2D'): 
        conv_2D_to_1D_spec(data_type_gen,inst,vis,gen_dic,data_dic,data_dic[data_type_gen],coord_dic)  

    #Analyzing converted profiles
    if (data_type_gen!='Diff') and gen_dic['fit_'+data_type_gen+'_1D']: 
        MAIN_single_anaprof('',data_type_gen+'_1D',data_dic,gen_dic,inst,vis,coord_dic,theo_dic,plot_dic,system_param['star'])   

    return None

def bin_gen_functions(data_type_gen,mode,inst,gen_dic,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,mock_dic={},vis=None):
    """**Wrap-up function for binned datasets.**

    Args:
        TBD:
    
    Returns:
        None
    
    """ 
    #Binning profiles for analysis purpose 
    if gen_dic[data_type_gen+'bin'+mode]: 
        process_bin_prof(mode,data_type_gen,gen_dic,inst,vis,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,ar_dic=mock_dic)

    #Analyzing binned profiles
    if gen_dic['fit_'+data_type_gen+'bin'+mode]: 
        MAIN_single_anaprof(mode,data_type_gen+'bin',data_dic,gen_dic,inst,vis,coord_dic,theo_dic,plot_dic,system_param['star'])                        

    #Calculating generic stellar continuum from binned master spectrum
    if (data_type_gen in ['DI','Intr']) and gen_dic[data_type_gen+'_stcont']:
        process_spectral_cont(mode,data_type_gen,inst,data_dic,gen_dic,vis)

    #Defining CCF mask
    #    - over all visits if possible, or over the single processed visit
    if (data_type_gen in ['DI','Intr']) and gen_dic['def_'+data_type_gen+'masks']:
        if (mode=='' and (data_dic[inst]['n_visits_inst']==1)) or (mode=='multivis'):
            def_masks(mode,gen_dic,data_type_gen,inst,vis,data_dic,plot_dic,system_param,data_prop)

    return None






def init_gen(data_dic,mock_dic,gen_dic,system_param,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic):
    r"""**Initialization: generic**

    Initializes generic fields for the workflow.
    
    Args:
        TBD
    
    Returns:
        None
    
    """ 
      
    #Positional dictionary
    coord_dic={}
    
    #Additional properties
    data_prop={}
    
    #Data types
    gen_dic['type_name']={'DI':'disk-integrated','Diff':'differential','Intr':'intrinsic','Atm':'atmospheric','Absorption':'absorption','Emission':'emission'}  
    gen_dic['type_list'] = np.array(['DI','Diff','Intr','Atm'])
    
    #Planets with known orbits and transit properties
    #    - this does not imply that the planet is actually transiting
    gen_dic['def_pl']=[]
    for obj in system_param:
        if (obj!='star') and ('inclination' in system_param[obj]) and (obj in data_dic['DI']['system_prop']['achrom']):gen_dic['def_pl']+=[obj]
    
    #Deactivate all plot modules if requested
    if not gen_dic['plots_on']:
        for key in plot_dic:
            if plot_dic[key] in ['png','pdf','jpg']:plot_dic[key]=''

    #Check for active plots to activate call to global plot function
    if gen_dic['plots_on']:
        gen_dic['plots_on'] = False
        for key in plot_dic:
            if plot_dic[key] in ['png','pdf','jpg']:
                gen_dic['plots_on'] = True
                break

    #------------------------------------------------------------------------------    
    #Data is processed by the workflow
    if gen_dic['sequence'] not in ['system_view']:
        
        #Multi-threading
        print('Multi-threading: '+str(cpu_count())+' threads available')

        #Instruments to which are associated the different datasets
        if gen_dic['mock_data']:  
            data_dic['instrum_list']=list(mock_dic['visit_def'].keys())
            if len(data_dic['instrum_list'])==0:stop('ERROR : no instrument was requested for mock datasets.')   
            print('Running with artificial data') 
    
        else:  
            data_dic['instrum_list'] = list(gen_dic['data_dir_list'].keys())
            if len(data_dic['instrum_list'])==0:stop('ERROR : no instrument was requested for observed datasets.') 
            print('Running with observational data') 

        #Used visits
        for inst in data_dic['instrum_list']:
            if inst not in gen_dic['unused_visits']:gen_dic['unused_visits'][inst]=[]
            
        #Total number of processed instruments
        gen_dic['n_instru'] = len(data_dic['instrum_list'])
    
        #All processed types
        #    - if not set, assumed to be CCF by default
        #    - gen_dic['type'] traces the original types of datasets throughout the pipeline 
        gen_dic['all_types'] = []
        for inst in data_dic['instrum_list']:    
            if (inst not in gen_dic['type']):gen_dic['type'][inst]='CCF'
            gen_dic['all_types']+=[gen_dic['type'][inst]]
        gen_dic['all_types'] = list(np.unique(gen_dic['all_types']))
        gen_dic['specINtype'] = any('spec' in s for s in gen_dic['all_types'])
        gen_dic['ccfINtype'] = ('CCF' in gen_dic['all_types'])   
    
        #Use of covariance
        gen_dic['flag_err_inst'] = {}
        if gen_dic['mock_data']:gen_dic['use_cov'] = False    
        else:
            #Used by default with spectral datasets, unless CCF datasets are processed or user requests otherwise
            if gen_dic['ccfINtype']:gen_dic['use_cov'] = False  
        if not gen_dic['use_cov']:print('Covariances discounted')        
    
        #Automatic activation/deactivation
        if gen_dic['pca_ana']:gen_dic['intr_data'] = True
        if gen_dic['intr_data']:
            if not gen_dic['diff_data']:
                print('Automatic activation of differential profile extraction')
                gen_dic['diff_data'] = True    
        else:
            for key in ['map_Intr_prof','all_intr_data','Intr_prof']:plot_dic[key]=''

        #Deactivate general conditions
        if (not gen_dic['specINtype']):
            if gen_dic['DI_CCF']:
                print('WARNING: all datasets in CCF mode, automatic disabling of DI profiles conversion into CCFs.')
                gen_dic['DI_CCF']=False
            if detrend_prof_dic['full_spec']:
                print('WARNING: all datasets in CCF mode, automatic disabling of spectral detrending.')
                detrend_prof_dic['full_spec'] = False
        if gen_dic['Diff_CCF']:
            if (not gen_dic['specINtype']) or (gen_dic['DI_CCF']) or (not gen_dic['diff_data']) or (gen_dic['intr_data']):
                print('WARNING: automatic disabling of Diff profiles conversion into CCFs.')
                gen_dic['Diff_CCF']=False      
        if gen_dic['Intr_CCF']:                
            if (not gen_dic['specINtype']) or (gen_dic['DI_CCF']) or (not gen_dic['intr_data']):        
                print('WARNING: automatic disabling of Intr profiles conversion into CCFs.')
                gen_dic['Intr_CCF']=False     
   
        #Deactivate all calculation options
        if not gen_dic['calc_all']:
            for key in gen_dic:
                if ('calc_' in key) and (gen_dic[key]):gen_dic[key] = False 
    
        #Deactivate spectral corrections in CCF mode
        sp_corr_list = ['corr_tell','glob_mast','corr_FbalOrd','corr_Fbal','corr_Ftemp','corr_cosm','mask_permpeak','corr_wig','corr_fring','trim_spec']
        if (not gen_dic['specINtype']):
            for key in sp_corr_list:gen_dic[key]=False
            gen_dic['gcal']=False
            gen_dic['cond_plot_gcal'] = False

            #Deactivate EvE outputs
            if gen_dic['EvE_outputs']:
                print('WARNING (EvE outputs): data must be in spectral mode')
                gen_dic['EvE_outputs'] = False
            
        else:

            #1D spectra
            if ('spec2D' not in gen_dic['all_types']): 
                
                #Deactive if spectra are not 2D:
                #    - spectral calibration, spectral balance correction over orders, wiggles                
                gen_dic['gcal'] = False
                gen_dic['cond_plot_gcal'] = False
                gen_dic['corr_FbalOrd']=False
                plot_dic['wig_order_fit']=''
                
            #2D spectra
            else:
                
                #Automatic calculation of calibration profiles
                gen_dic['gcal'] = True
                gen_dic['cond_plot_gcal'] = (plot_dic['gcal_all']!='') or (plot_dic['gcal_ord']!='') or (plot_dic['noises_ord']!='')

            #Activate global master calculation if required for flux balance corrections
            if (gen_dic['corr_Fbal']) or (gen_dic['corr_FbalOrd']):
                gen_dic['glob_mast']=True

                #Default bin size for global flux balance (in 1e13 s-1)
                if (inst not in gen_dic['Fbal_bin_nu']):
                    Fbal_bin_nu_inst = {
                        'SOPHIE_HE':1.,
                        'SOPHIE_HR':1.,
                        'CORALIE':1.,
                        'HARPN':1.,
                        'HARPS':1.,
                        'ESPRESSO':1.,
                        'ESPRESSO_MR':1.,
                        'CARMENES_VIS':1.,
                        'EXPRES':1.,
                        'IGRINS2_Red':1.,
                        'IGRINS2_Blue':1.,
                        'MAROONX_Red':1.,
                        'MAROONX_Blue':1.,
                        'MIKE_Red':1.,
                        'MIKE_Blue':1.,
                        'NIGHT':1.,
                        'NIRPS_HA':1.,
                        'NIRPS_HE':1.}
                    if inst not in Fbal_bin_nu_inst:stop('ERROR : default value for "gen_dic["Fbal_bin_nu"]" undefined for '+inst+' (source : ANTARESS_main.py > init_gen()')
                    gen_dic['Fbal_bin_nu'][inst] = Fbal_bin_nu_inst[inst] 
 
                #Default bin size for order flux balance (in A)
                if (inst not in gen_dic['FbalOrd_binw']):
                    FbalOrd_binw_inst = {
                        'SOPHIE_HE':2.,
                        'SOPHIE_HR':2.,
                        'CORALIE':2.,
                        'HARPN':2.,
                        'HARPS':2.,
                        'ESPRESSO':2.,
                        'ESPRESSO_MR':2.,
                        'CARMENES_VIS':2.,
                        'EXPRES':2.,
                        'NIGHT':2.,
                        'NIRPS_HA':2.,
                        'NIRPS_HE':2.}
                    if inst not in FbalOrd_binw_inst:stop('ERROR : default value for "gen_dic["FbalOrd_binw"]" undefined for '+inst+' (source : ANTARESS_main.py > init_gen()')
                    gen_dic['FbalOrd_binw'][inst] = FbalOrd_binw_inst[inst]    
 
            #Deactivate arbitrary scaling if scaling not required
            if (not gen_dic['corr_Fbal']) and (not gen_dic['corr_FbalOrd']):gen_dic['Fbal_vis']=None
            
            #Check for arbitrary scaling        
            elif (gen_dic['Fbal_vis']!='ext') and (gen_dic['n_instru']>1):stop('Flux balance must be set to a theoretical reference for multiple instruments')

            #Deactivate transit scaling if temporal flux correction is applied
            if gen_dic['corr_Ftemp']:data_dic['DI']['rescale_DI'] = True  
                    
            #Activate correction flag
            gen_dic['corr_data'] = False
            for key in sp_corr_list: gen_dic['corr_data'] |=gen_dic[key]

        #Deactivate modules and fields irrelevant to mock datasets
        if gen_dic['mock_data']:
            for key in ['gcal','corr_tell','tell_weight','glob_mast','corr_Fbal','corr_FbalOrd','corr_Ftemp','corr_cosm','mask_permpeak','corr_wig','EvE_outputs']:gen_dic[key] = False
    
        #Deactivate plot if module is not called
        if (plot_dic['glob_mast']!='') and gen_dic['glob_mast']:
            plot_dic['glob_mast']==''
            print('Disabling "glob_mast" plot')
        if (plot_dic['Fbal_corr_vis']!='') and ((not gen_dic['corr_Fbal']) or (gen_dic['Fbal_vis'] is None)):        
            plot_dic['Fbal_corr_vis']==''
            print('Disabling "Fbal_corr_vis" plot')        
    
        #Set general condition for conversions
        gen_dic['spec_1D'] = False
        gen_dic['calc_spec_1D'] = False
        gen_dic['CCF_from_sp'] = False
        gen_dic['calc_CCF_from_sp'] = False
        gen_dic['type2var'] = {'DI':'EFsc2','Diff':'EFdiff2','Intr':'EFintr2','Emission':'EFem2','Absorption':'EAbs2'} 
        gen_dic['earliertypes4var'] = {'DI':['DI'],'Diff':['DI','Diff'],'Intr':['DI','Diff','Intr'],'Atm':['DI','Diff','Atm']}
        for key in gen_dic['type_list']:
            gen_dic['spec_1D'] |= gen_dic['spec_1D_'+key] 
            gen_dic['calc_spec_1D'] |=  gen_dic['calc_spec_1D_'+key]      
            gen_dic['CCF_from_sp'] |= gen_dic[key+'_CCF'] 
            gen_dic['calc_CCF_from_sp'] |=  gen_dic['calc_'+key+'_CCF']
            
            #Initialize flags
            data_dic[key]['type'] = {inst:deepcopy(gen_dic['type'][inst]) for inst in gen_dic['type']}
            data_dic[key]['spec_to_CCF'] = {inst:False for inst in gen_dic['type']}
            data_dic[key]['spec2D_to_spec1D'] = {inst:False for inst in gen_dic['type']}
        
            #Deactive 1D plots if no 1D conversion is applied
            if not gen_dic['spec_1D_'+key] and (plot_dic['sp_'+key+'_1D']!=''):
                print('WARNING: no 1D conversion of '+key+' profiles is applied, disabling plot_dic["sp_'+key+'_1D"]') 
                plot_dic['sp_'+key+'_1D'] = ''
            
        if gen_dic['CCF_from_sp']:gen_dic['ccfINtype'] = True

        
        #Set specific fields per instrument
        gen_dic['corr_FbalOrd_inst'] = {}
        for inst in gen_dic['type']:
            
            #Set specific conversion flags and final data mode for each type of profiles and instruments
            #    - the general instrument and visit dictionaries contain a field type that represents the mode of the data at the current stage of the pipeline
            #    - here we set the final mode for each type of profile, so that they can be retrieved in their original mode for plotting
            #      eg, if profiles are converted into CCF at the differential stage, we keep the information that disk-integrated profiles were in spectral mode
            #    - we also set a flag for each instrument related to the effective conversion of its datasets            
            for ikey,key in enumerate(gen_dic['type_list']):
                
                #Conversion of current type into CCF is requested
                if gen_dic[key+'_CCF']:
                    
                    #Checking global data format
                    if gen_dic['type'][inst]=='CCF':print('WARNING: input profiles for '+inst+' are in CCF format, no CCF conversion of '+key+' profiles will be applied.') 
                    else:
                        
                        #Checking format of previous data types
                        conv_ok = True
                        for iprev in range(0,ikey):
                            
                            #Stop if one of previous data types was already converted
                            if data_dic[gen_dic['type_list'][iprev]]['spec_to_CCF'][inst]:
                                conv_ok = False
                                print('WARNING: '+gen_dic['type_name'][gen_dic['type_list'][iprev]]+' profiles for '+inst+' were converted in CCF format, no CCF conversion of '+gen_dic['type_name'][key]+' profiles will be applied.') 
                            break
                        
                        #Flag conversion as ok and affect data format to current and later types
                        if conv_ok:
                            
                            #Flag is set to True only for the specific data type that was converted
                            data_dic[key]['spec_to_CCF'][inst]=True 
                            for isub in range(ikey,len(gen_dic['type_list'])):data_dic[gen_dic['type_list'][isub]]['type'][inst]='CCF'
                            
                            #Conversion of Intr profiles is also applied to OT Diff profiles
                            if key=='Intr':
                                data_dic['Diff']['spec_to_CCF'][inst]=True 
                                data_dic['Diff']['type'][inst]='CCF'

                #Conversion of current type into 1D spectra is requested
                if gen_dic['spec_1D_'+key]:
                    if gen_dic['type'][inst]=='CCF':print('WARNING: input profiles for '+inst+' are in CCF format, no 1D conversion of '+key+' profiles will be applied.') 
                    elif gen_dic['type'][inst]=='spec1D':print('WARNING: input profiles for '+inst+' are in 1D format, no 1D conversion of '+key+' profiles will be applied.') 
                    else:
                        conv_ok = True
                        for iprev in range(0,ikey):
                            if data_dic[gen_dic['type_list'][iprev]]['spec2D_to_spec1D'][inst]:
                                conv_ok = False
                                print('WARNING: '+gen_dic['type_name'][gen_dic['type_list'][iprev]]+' profiles for '+inst+' were converted in 1D format, no 1D conversion of '+gen_dic['type_name'][key]+' profiles will be applied.') 
                            break
                        if conv_ok:
                            data_dic[key]['spec2D_to_spec1D'][inst]=True 
                            for isub in range(ikey,len(gen_dic['type_list'])):data_dic[gen_dic['type_list'][isub]]['type'][inst]='spec1D'
                            if key=='Intr':
                                data_dic['Diff']['spec2D_to_spec1D'][inst]=True 
                                data_dic['Diff']['type'][inst]='spec1D'

        #Set general condition to activate binning modules
        for mode in ['','multivis']:
            gen_dic['bin'+mode]=False
            gen_dic['calc_bin'+mode]=False
            for data_type_gen in ['DI','Intr','Atm']:
                gen_dic['bin'+mode] |= ( gen_dic[data_type_gen+'bin'+mode] | gen_dic['fit_'+data_type_gen+'bin'+mode])
                gen_dic['calc_bin'+mode] |= ( gen_dic['calc_'+data_type_gen+'bin'+mode] | gen_dic['calc_fit_'+data_type_gen+'bin'+mode])
    
        #Set general condition for single profiles fits
        for key in ['DI','Intr','Atm']:
            gen_dic['fit_'+key+'_gen'] = gen_dic['fit_'+key] | gen_dic['fit_'+key+'bin'] | gen_dic['fit_'+key+'binmultivis'] 
        gen_dic['fit_Diff_gen'] = False
        
        #Automatic continuum and fit range
        for key in ['DI','Intr','Atm']:
            if gen_dic['fit_'+key+'_gen']:
                if ('cont_range' not in data_dic[key]):data_dic[key]['cont_range']={}
                if ('fit_range' not in data_dic[key]):data_dic[key]['fit_range']={}
        if (gen_dic['diff_data']) and ('cont_range' not in data_dic['Diff']):data_dic['Diff']['cont_range']={}
    
        #Deactivate conditions
        if gen_dic['DIbin']==False:
            data_dic['DI']['fit_MCCFout']=False
        if (not gen_dic['specINtype']) or (gen_dic['DI_CCF']):plot_dic['spectral_LC']=''
        if (not gen_dic['diff_data']):
            for key in ['map_Diff_prof','Diff_prof']:plot_dic[key]=''    
        if not gen_dic['pl_atm']:
            for key in ['map_Atm_prof','sp_atm','CCFatm']:plot_dic[key]=''  
        else:
            if gen_dic['Intr_CCF']:stop('Error: atmospheric extraction cannot be performed after Diff./Intr. CCF conversion')

        #------------------------------------------------------------------------------------------------------------------------
        #Weighing conditions
        #    - automatic activation to calculate or use contributors to variance profiles of disk-integrated spectra, based on the use of modules that average profiles requesting this variance
        # + extraction of Diff profiles, requiring DI master (if intrinsic or atmospheric extraction is required, the extraction of Diff profiles is as well) 
        # + estimate of Intr profiles based on DI master
        # + conversion of profiles from 2D into 1D
        # + binning profiles together
        #    - we disable calculation of DI weighing master if it is requested but no other module requiring this weight are recalculated
        weights_on = ''
        weights_off = ''
        cond_def = (gen_dic['diff_data'] | (gen_dic['loc_prof_est'] &  (data_dic['Intr']['opt_loc_prof_est']['corr_mode'] in ['DIbin','Intrbin'])) | gen_dic['spec_1D'] | gen_dic['bin'] | gen_dic['binmultivis'])
        gen_dic['DImast_weight'] = cond_def
        if gen_dic['DImast_weight']:weights_on+='(DI master) '
        else:weights_off+='(DI master) '
        if gen_dic['DImast_weight'] and gen_dic['calc_DImast']:gen_dic['calc_DImast'] =  gen_dic['calc_diff_data'] | (gen_dic['calc_loc_prof_est'] &  (data_dic['Intr']['opt_loc_prof_est']['corr_mode'] in ['DIbin','Intrbin'])) | gen_dic['calc_spec_1D'] | gen_dic['calc_bin'] | gen_dic['calc_binmultivis']
        if ('spec2D' in gen_dic['all_types']):gen_dic['cal_weight'] = cond_def | gen_dic['corr_wig']   
        else:gen_dic['cal_weight'] = False   
        if gen_dic['cal_weight']:weights_on+='(Calibration) '
        else:weights_off+='(Calibration) '
        if gen_dic['specINtype']:gen_dic['tell_weight'] = cond_def & gen_dic['corr_tell'] 
        else:gen_dic['tell_weight'] = False
        if gen_dic['tell_weight']:weights_on+='(Tellurics) '
        else:weights_off+='(Tellurics) '
        print('Weight contributors :')
        if weights_on != '':print('  On :'+weights_on)
        else:print('  On : None')
        if weights_off != '':print('  Off :'+weights_off)
        else:print('  Off : None')
        

        #------------------------------------------------------------------------------------------------------------------------

        #Automatic activation of light curve computation
        #    - required to calculate binned disk-integrated profiles or convert them from 2D to 1D, to convert Diff profiles into CCFs or from 2D to 1D, and to compute (and process) intrinsic and atmospheric profiles
        if (not gen_dic['flux_sc']) and (gen_dic['DImast_weight'] or gen_dic['spec_1D_DI'] or gen_dic['DIbin'] or gen_dic['Diff_CCF'] or gen_dic['spec_1D_Diff'] or gen_dic['intr_data'] or (gen_dic['pl_atm'] and data_dic['Atm']['pl_atm_sign']=='Absorption')):
            print('Automatic activation of flux scaling calculation')
            gen_dic['flux_sc'] = True

        #Set general conditions to activate joined datasets modules 
        gen_dic['joined_ana']=False
        gen_dic['fit_DIProf'] = False     #module undefined for now
        gen_dic['fit_DiffProp'] = False    #module undefined for now
        for key in gen_dic['type_list']:gen_dic['joined_ana'] |= (gen_dic['fit_'+key+'Prof'] or gen_dic['fit_'+key+'Prop'])

        #Standard-deviation curves with bin size for the out-of-transit differential CCFs
        #    - defining the maximum size of the binning window, and the binning size for the sliding window (we will average the bins in windows of width bin_size from 1 to 40 (arbitrary))
        if gen_dic['scr_search']:
            gen_dic['scr_srch_max_binwin']=40.     
            gen_dic['scr_srch_nperbins'] = 1.+np.arange(gen_dic['scr_srch_max_binwin'])
    
        #Default options for instrumental calibration
        if gen_dic['gcal']: 
            for inst in data_dic['instrum_list'] :
                
                #Spectral bin width (A)
                if (inst not in gen_dic['gcal_binw']):
                    def_gcal_binw = {'CORALIE':3.5,'ESPRESSO':0.5,'NIGHT':1.,'NIRPS_HE':1.}
                    if inst in def_gcal_binw:gen_dic['gcal_binw'][inst] = def_gcal_binw[inst]
                    else:stop('ERROR: no default value of "gen_dic["gcal_binw"]" for '+inst+'. Run the module with a custom value for this field.')

                #Threshold
                if (inst not in gen_dic['gcal_thresh']):gen_dic['gcal_thresh'][inst] = {'outliers':5.,'global':1e10}            
                
        #Stellar continuum
        for key in ['DI','Intr']:
            if (gen_dic[key+'_stcont'] and (not (gen_dic[key+'bin'] or gen_dic[key+'binmultivis']))):
                print("WARNING: binned "+key+" spectrum must be calculated to use gen_dic['"+key+"_stcont']")
    
        #Default options for stellar continuum calculation
        if gen_dic['mask_permpeak'] or gen_dic['DI_stcont'] or gen_dic['Intr_stcont']:
            for inst in data_dic['instrum_list'] :
                
                #Size of rolling window for peak exclusion
                #    - in A
                if (inst not in gen_dic['contin_roll_win']):gen_dic['contin_roll_win'][inst] = 2.
        
                #Size of smoothing window
                #    - in A
                if (inst not in gen_dic['contin_smooth_win']):gen_dic['contin_smooth_win'][inst] = 0.5
        
                #Size of local maxima window
                #    - in A
                if (inst not in gen_dic['contin_locmax_win']):gen_dic['contin_locmax_win'][inst] = 0.5
        
                #Flux/wavelength stretching
                if (inst not in gen_dic['contin_stretch']):gen_dic['contin_stretch'][inst] = 10. 
        
                #Rolling pin radius
                #    - value corresponds to the bluest wavelength of the processed spectra
                if (inst not in gen_dic['contin_pinR']):gen_dic['contin_pinR'][inst] = 5.

        #Raise warning
        if gen_dic['EvE_outputs']:
            if not gen_dic['DIbin']:print('WARNING (EvE outputs): a master-out spectrum must be calculated (gen_dic["DIbin"] = True)')
    
        #------------------------------------------------------------------------------------------------------------------------
    
        #Stellar mask generation is requested
        gen_dic['def_st_masks'] = gen_dic['def_DImasks'] | gen_dic['def_Intrmasks'] 
    
        #Mask used to compute CCF on stellar lines
        if len(gen_dic['CCF_mask'])>0:
            gen_dic['CCF_mask_wav']={}
            gen_dic['CCF_mask_wgt']={}
    
        #Condition to exclude ranges contaminated by planetary absorption
        #    - independent of atmospheric signal extraction
        if len(data_dic['Atm']['no_plrange'])>0:
            data_dic['Atm']['exc_plrange']=True
            data_dic['Atm']['plrange']=np.array(data_dic['Atm']['plrange'])
        else:data_dic['Atm']['exc_plrange']=False
    
        #Properties associated with planetary signal extraction
        if gen_dic['calc_pl_atm']:
      
            #Deactivate signal reduction for input CCFs
            if (not gen_dic['specINtype']):gen_dic['atm_CCF']=False
    
        #Mask used to compute atmospheric CCFs and/or exclude atmospheric planetary ranges
        if data_dic['Atm']['CCF_mask'] is not None:
            
            #Upload CCF mask
            ext = data_dic['Atm']['CCF_mask'].split('.')[-1]
            if (ext=='fits'):
                hdulist = fits.open(data_dic['Atm']['CCF_mask'])
                data_loc = hdulist[1].data
                data_dic['Atm']['CCF_mask_wav'] = data_loc['lambda']
                if data_dic['Atm']['use_maskW']:data_dic['Atm']['CCF_mask_wgt'] = data_loc['contrast']       
                hdulist.close()
            elif (ext=='csv'):
                data_loc = np.genfromtxt(data_dic['Atm']['CCF_mask'], delimiter=',', names=True)
                data_dic['Atm']['CCF_mask_wav'] = data_loc['wave']
                if data_dic['Atm']['use_maskW']:data_dic['Atm']['CCF_mask_wgt'] = data_loc['contrast']
            elif (ext in ['txt','dat']):
                data_loc = np.loadtxt(data_dic['Atm']['CCF_mask']).T            
                data_dic['Atm']['CCF_mask_wav'] = data_loc[0]
                if data_dic['Atm']['use_maskW']:data_dic['Atm']['CCF_mask_wgt'] = data_loc[1]        
            else:
                stop('CCF mask extension TBD') 
                
            #No weighing of lines
            if not data_dic['Atm']['use_maskW']:data_dic['Atm']['CCF_mask_wgt'] = np.repeat(1.,len(data_dic['Atm']['CCF_mask_wav']))        
    


    #------------------------------------------------------------------------------
    #Star
    #------------------------------------------------------------------------------
    if gen_dic['sequence'] not in ['st_master_tseries']:
        
        #Conversions and calculations
        system_param['star']['Rstar_km'] = system_param['star']['Rstar']*Rsun
        
        #Spherical star
        if ('f_GD' not in system_param['star']):
            system_param['star']['f_GD']=0.
            system_param['star']['RpoleReq'] = 1.
            
        #Oblate star
        elif system_param['star']['f_GD']>0.:
            print('Star is oblate')
            system_param['star']['RpoleReq']=1.-system_param['star']['f_GD']
    
        #Reference time for stellar phase (bjd)
        if 'Tcenter' not in system_param['star']:system_param['star']['Tcenter'] = 2400000.
    
        #Stellar equatorial rotation rate (rad/s)
        #    - om = 2*pi/P = v/R
        system_param['star']['om_eq'] = system_param['star']['veq']/system_param['star']['Rstar_km']    
        for key in ['_spots','_faculae']:
            if 'veq'+key in system_param['star']:system_param['star']['om_eq'+key]=system_param['star']['veq'+key]/system_param['star']['Rstar_km']
            else:
                system_param['star']['veq'+key]=system_param['star']['veq']
                system_param['star']['om_eq'+key]=system_param['star']['om_eq']
    
        #Stellar equatorial rotation period (d)
        #    - P = 2*pi*R/v
        for key in ['','_spots','_faculae']:
            system_param['star']['Peq'+key] = (2.*np.pi*system_param['star']['Rstar_km'])/(system_param['star']['veq'+key]*24*3600)
    
        #No GD
        if ('beta_GD' not in system_param['star']):system_param['star']['beta_GD']=0.
        if ('Tpole' not in system_param['star']):system_param['star']['Tpole']=0.
    
        #Conversions
        system_param['star']['istar_rad']=system_param['star']['istar']*np.pi/180.
        system_param['star']['cos_istar']=np.cos(system_param['star']['istar_rad'])
        system_param['star']['vsini']=system_param['star']['veq']*np.sin(system_param['star']['istar_rad'])    #km/s
        
        #Default parameters
        for key in ['alpha_rot','beta_rot','alpha_rot_spots','beta_rot_spots','alpha_rot_faculae','beta_rot_faculae','c1_CB','c2_CB','c3_CB','c1_pol','c2_pol','c3_pol','c4_pol']:
            if key not in system_param['star']:system_param['star'][key] = 0.
    
        #Conversion factor from the LOS velocity output (/Rstar/h) to RV in km/s
        system_param['star']['RV_conv_fact']=-system_param['star']['Rstar_km']/3600.
    
        #Macroturbulence
        if theo_dic['mac_mode'] is not None:
            if 'rt' in theo_dic['mac_mode']:
                theo_dic['mac_mode_func'] = calc_macro_ker_rt
                if theo_dic['mac_mode']=='rt_iso':
                    system_param['star']['A_T']=system_param['star']['A_R']
                    system_param['star']['ksi_T']=system_param['star']['ksi_R']   
            elif 'anigauss' in theo_dic['mac_mode']:
                theo_dic['mac_mode_func'] = calc_macro_ker_anigauss
                if theo_dic['mac_mode']=='anigauss_iso':
                    system_param['star']['eta_T']=system_param['star']['eta_R']

    #------------------------------------------------------------------------------
    #Planets
    #------------------------------------------------------------------------------

    #Planets defined in the system
    gen_dic['all_pl'] = [pl_loc for pl_loc in system_param.keys() if pl_loc!='star']

    #Planets considered for analysis
    gen_dic['studied_pl_list'] = list(gen_dic['studied_pl'].keys()) 
    if len(gen_dic['studied_pl_list'])>0:
        txt_print = 'Study of: '+gen_dic['studied_pl_list'][0]
        if len(gen_dic['studied_pl_list'])>1:
            for pl_loc in gen_dic['studied_pl_list'][1::]:txt_print+=', '+pl_loc
        print(txt_print)

    #Keplerian motion
    if ('kepl_pl' not in gen_dic):gen_dic['kepl_pl']=['all']
    if (gen_dic['kepl_pl']==['all']):
        print('Accounting for Keplerian motion from all planets')
        gen_dic['kepl_pl']=deepcopy(gen_dic['all_pl'])
    
    #Planet properties    
    gen_dic['tr_pl']=[]
    for pl_loc in gen_dic['all_pl']:
        print('Initializing planet '+pl_loc)
        
        #Checking if there is a "-" in a target name
        if '-' in pl_loc:stop('ERROR : Invalid target name: {}. Target names should not contain a hyphen.'.format(pl_loc))        
        PlParam_loc=system_param[pl_loc]
        
        #Converting argument of periastron from degree to radians
        PlParam_loc['omega_rad']=PlParam_loc['omega_deg']*np.pi/180.
        
        #Converting orbital period from days to seconds
        PlParam_loc['period_s'] = PlParam_loc['period']*24.*3600.

        #Planet with known orbit and transit properties
        if pl_loc in gen_dic['def_pl']:

            #Adding absolute semi-major axis
            if ('a' not in PlParam_loc):PlParam_loc['a']=PlParam_loc['aRs']*system_param['star']['Rstar_km']/AU_1  # in au            

            #Converting orbital inclination from degree to radians
            PlParam_loc['inclin_rad']=PlParam_loc['inclination']*np.pi/180.
            
            #Deriving planet mass (in Mj)
            if ('Msini' in PlParam_loc) and ('Mp' not in PlParam_loc):PlParam_loc['Mp']=PlParam_loc['Msini']/np.sin(PlParam_loc['inclin_rad'])

            #Semi-amplitude of planet orbital motion around the star (approximated with Mp << Mstar) in km/s
            PlParam_loc['Kp_orb'] = (2.*np.pi/PlParam_loc['period_s'])*np.sin(PlParam_loc['inclin_rad'])*PlParam_loc['a']*AU_1/np.sqrt(1.-PlParam_loc['ecc']**2.)

            #Stellar mass derived from semi-orbital motion around the star
            #    - assuming Mp << Ms : 
            # P^2/a^3 = 4*pi^2/G*Ms 
            # Ms = 4*pi^2*a^3/(G*P^2)
            PlParam_loc['Mstar_orb'] = 4.*np.pi**2.*(PlParam_loc['a']*AU_1*1e3)**3./(Msun*G_usi*PlParam_loc['period_s']**2.)

            #Converting sky-projected angle from degree to radians
            PlParam_loc['lambda_rad']=PlParam_loc['lambda_proj']*np.pi/180.

            #Impact parameter
            if 'b' not in PlParam_loc:PlParam_loc['b']=PlParam_loc['aRs']*np.cos(PlParam_loc['inclin_rad'])   
            if (PlParam_loc['ecc']==0.) & (PlParam_loc['b']>1.):print('WARNING: impact parameter for ',pl_loc,' > 1')

        #Calculation of transit center (or time of cunjunction for a non-transiting planet)
        #    - when T0 is not given but Tperiastron is known
        #    - we could calculate directly the true anomaly from Tperi, but all phases in the routine
        # are defined relatively with the transit center
        #    - mean anomaly is counted from pericenter as M(t) = 2pi*(t - Tperi)/P
        # M(Tcenter) = 2pi*(Tcenter - Tperi)/P 
        # Tcenter = Tperi + M(Tcenter)*P/2pi         
        if ('TCenter' not in PlParam_loc):    
            PlParam_loc['TCenter']=PlParam_loc['Tperi']
            if (PlParam_loc['ecc']>0):
                PlParam_loc['Mean_anom_TR'] = calc_mean_anom_TR(PlParam_loc['ecc'],PlParam_loc['omega_rad']) 
                PlParam_loc['TCenter']+=(PlParam_loc['Mean_anom_TR']*PlParam_loc["period"]/(2.*np.pi))
  
        #Keplerian semi-amplitude of the star from the planet (km/s) 
        PlParam_loc['Kstar_kms']=PlParam_loc['Kstar']/1000. if 'Kstar' in PlParam_loc else calc_Kstar(PlParam_loc,system_param['star'])/1000.

        #Orbital frequency, in year-1
        PlParam_loc['omega_p']=2.*np.pi*365.2425/PlParam_loc['period']

        #Transit check
        #    - this call is also used to check for transit status
        if (pl_loc in gen_dic['studied_pl_list']):
            if (pl_loc not in data_dic['DI']['system_prop']['achrom']):
                print('  No RpRs defined for '+pl_loc+' : assuming object is not transiting')
            elif ('inclination' not in PlParam_loc):
                print('  No inclination defined for '+pl_loc+' : assuming object is not transiting')
            else:
                
                #Contact phases
                contact_phases=calc_tr_contacts(data_dic['DI']['system_prop']['achrom'][pl_loc][0],PlParam_loc,plot_dic['stend_ph'],system_param['star'])
                
                #Planet is transiting
                if contact_phases is not None:
                    print('  Transiting object')
                    gen_dic['tr_pl']+=[pl_loc]
                
                    #Transit duration
                    if ('TLength' not in PlParam_loc):                
                        PlParam_loc['TLength'] = (contact_phases[3]-contact_phases[0])*PlParam_loc['period']           
                        print('  Automatic definition of T14['+str(pl_loc)+']='+"{0:.2f}".format(PlParam_loc['TLength']*24.)+' h')

                else:
                    print('  Non-transiting object')

    #Calculating theoretical properties of the planet-occulted regions
    if (len(gen_dic['tr_pl'])>0) and (gen_dic['sequence'] not in ['st_master_tseries']) and (not gen_dic['theoPlOcc']) and ((('CCF' in data_dic['Diff']['type'].values()) and (gen_dic['fit_Intr'])) or (gen_dic['align_Intr']) or (gen_dic['calc_pl_atm'])):
        print('Automatic activation of "gen_dic["theoPlOcc"]"')
        gen_dic['theoPlOcc']=True

    #Oversampling factor for values from planet-occulted regions
    #    - uses the nominal planet-to-star radius ratios, which must correspond to the band from which local properties are derived
    theo_dic['d_oversamp_pl']={}
    for pl_loc in theo_dic['n_oversamp']:
        if (pl_loc in gen_dic['tr_pl']) and (theo_dic['n_oversamp'][pl_loc]>0.):
            theo_dic['d_oversamp_pl'][pl_loc] = data_dic['DI']['system_prop']['achrom'][pl_loc][0]/theo_dic['n_oversamp'][pl_loc]
 
    #Set flag for errors on estimates for local stellar profiles (depending on whether they are derived from data or models)
    if data_dic['Intr']['opt_loc_prof_est']['corr_mode'] in ['DIbin','Intrbin']:data_dic['Intr']['cov_loc_star']=True
    else:data_dic['Intr']['cov_loc_star']=False    

    #Transit and stellar surfce chromatic properties
    #    - must be defined for data processing, transit scaling, calculation of planet properties
    #    - we remove the 'chrom' dictionary if no spectral data is used as input or if a single band is defined
    if ('LD' not in data_dic['DI']['system_prop']['achrom']):
        data_dic['DI']['system_prop']['achrom']['LD']=['uniform']
        print('WARNING : limb-darkening law set to "uniform". Define it with "data_dic["DI"]["system_prop"]["achrom"]"')
    for ideg in range(2,5):
        if 'LD_u'+str(ideg) not in data_dic['DI']['system_prop']['achrom']:data_dic['DI']['system_prop']['achrom']['LD_u'+str(ideg)] = [0.]
    if ('GD_dw' in data_dic['DI']['system_prop']['achrom']):
        system_param['star']['GD']=True
        print('Star is gravity-darkened')
    else:system_param['star']['GD']=False
    data_dic['DI']['system_prop']['achrom']['w']=[None]
    data_dic['DI']['system_prop']['achrom']['nw']=1
    data_dic['DI']['system_prop']['achrom']['cond_in_RpRs']={}
    data_dic['DI']['system_prop']['chrom_mode'] = 'achrom'
    if ('chrom' in data_dic['DI']['system_prop']):
        if (not gen_dic['specINtype']) or (len(data_dic['DI']['system_prop']['chrom']['w'])==1):data_dic['DI']['system_prop'].pop('chrom')
        else:
            data_dic['DI']['system_prop']['chrom_mode'] = 'chrom'
            data_dic['DI']['system_prop']['chrom']['w'] = np.array(data_dic['DI']['system_prop']['chrom']['w'])
            data_dic['DI']['system_prop']['chrom']['nw']=len(data_dic['DI']['system_prop']['chrom']['w'])
            data_dic['DI']['system_prop']['chrom']['cond_in_RpRs']={}
            
            #Typical scale of chromatic variations
            w_edge = def_edge_tab(data_dic['DI']['system_prop']['chrom']['w'][None,:][None,:])[0,0]    
            data_dic['DI']['system_prop']['chrom']['dw'] = w_edge[1::]-w_edge[0:-1]
            data_dic['DI']['system_prop']['chrom']['med_dw'] = np.median(data_dic['DI']['system_prop']['chrom']['dw'])

    #Store properties at the stage of broadband scaling 
    data_dic['DI']['system_prop_sc'] = deepcopy(data_dic['DI']['system_prop']) 
    if gen_dic['DI_CCF'] and ('chrom' in data_dic['DI']['system_prop']):
        data_dic['DI']['system_prop_sc']['chrom_mode'] = 'achrom'
        data_dic['DI']['system_prop_sc'].pop('chrom') 

    #Default transit model
    if 'transit_prop' not in data_dic['DI']:data_dic['DI']['transit_prop']={}    
        
    #Definition of grids discretizing planets disk to calculate planet-occulted properties
    theo_dic['x_st_sky_grid_pl']={}
    theo_dic['y_st_sky_grid_pl']={}
    theo_dic['Ssub_Sstar_pl']={}
    data_dic['DI']['system_prop']['RpRs_max']={}
    if ('nsub_Dpl' not in theo_dic):theo_dic['nsub_Dpl']={}
    for pl_loc in gen_dic['tr_pl']:

        #Largest possible planet size
        if ('chrom' in data_dic['DI']['system_prop']):data_dic['DI']['system_prop']['RpRs_max'][pl_loc] = np.max(data_dic['DI']['system_prop']['achrom'][pl_loc]+data_dic['DI']['system_prop']['chrom'][pl_loc])
        else:data_dic['DI']['system_prop']['RpRs_max'][pl_loc] = data_dic['DI']['system_prop']['achrom'][pl_loc][0]

        #Default grid scaled from a 51x51 grid for a hot Jupiter transiting a solar-size star
        #    - dsub_ref = (2*Rjup/Rsun)*(1/51)
        #      nsub_Dpl = int(2*RpRs/dsub_ref) = int( 51*RpRs*Rsun/Rjup )        
        if (pl_loc not in theo_dic['nsub_Dpl']):
            theo_dic['nsub_Dpl'][pl_loc] =int( 51.*data_dic['DI']['system_prop']['RpRs_max'][pl_loc]*Rsun/Rjup ) 
            print('Default nsub_Dpl['+str(pl_loc)+']='+str(theo_dic['nsub_Dpl'][pl_loc]))

        #Corresponding planet grid
        _,theo_dic['Ssub_Sstar_pl'][pl_loc],theo_dic['x_st_sky_grid_pl'][pl_loc],theo_dic['y_st_sky_grid_pl'][pl_loc],r_sub_pl2=occ_region_grid(data_dic['DI']['system_prop']['RpRs_max'][pl_loc],theo_dic['nsub_Dpl'][pl_loc])  
        
        #Identification of cells within the nominal and chromatic planet radii
        data_dic['DI']['system_prop']['achrom']['cond_in_RpRs'][pl_loc] = [(r_sub_pl2<data_dic['DI']['system_prop']['achrom'][pl_loc][0]**2.)]
        if ('chrom' in data_dic['DI']['system_prop']):
            data_dic['DI']['system_prop']['chrom']['cond_in_RpRs'][pl_loc]={}
            for iband in range(data_dic['DI']['system_prop']['chrom']['nw']):
                data_dic['DI']['system_prop']['chrom']['cond_in_RpRs'][pl_loc][iband] = (r_sub_pl2<data_dic['DI']['system_prop']['chrom'][pl_loc][iband]**2.)

    #------------------------------------------------------------------------------
    #Active regions
    #------------------------------------------------------------------------------

    #Generating multiple active regions
    if (mock_dic['auto_gen_ar'] != {}):generate_ar_prop(mock_dic, data_dic, gen_dic)

    #Active regions considered for analysis
    gen_dic['studied_ar_list'] = list(gen_dic['studied_ar'].keys()) 
    if len(gen_dic['studied_ar_list'])!=len(np.unique(gen_dic['studied_ar_list'])):stop('Active regions must have unique names in each visit')
    if len(gen_dic['studied_ar_list'])>0:
        txt_print = 'Study of: '+gen_dic['studied_ar_list'][0]
        if len(gen_dic['studied_ar_list'])>1:
            for pl_loc in gen_dic['studied_ar_list'][1::]:txt_print+=', '+pl_loc
        print(txt_print)

    #Initialize active region use
    theo_dic['x_st_sky_grid_ar']={}
    theo_dic['y_st_sky_grid_ar']={}
    theo_dic['Ssub_Sstar_ar'] = {}
    theo_dic['d_oversamp_ar']={}

    #Active region activation triggered
    gen_dic['ar_coord_par'] = ['lat_rad_exp','sin_lat_exp','cos_lat_exp','long_rad_exp','sin_long_exp','cos_long_exp','x_sky_exp','y_sky_exp','z_sky_exp']
    if len(gen_dic['studied_ar_list'])>0:
        print('Active regions are simulated')

        #Oversampling factor for active region-occulted regions
        #    - input corresponds to the half-angular opening of the active region
        #      taking the largest projected radius of the active region if placed along the LOS:
        # sin(ang) = Rproj/Rstar
        for ar in theo_dic['n_oversamp_ar']:
            if (theo_dic['n_oversamp_ar'][ar]>0.):
                theo_dic['d_oversamp_ar'][ar] = np.sin(data_dic['DI']['ar_prop']['achrom'][ar][0])/theo_dic['n_oversamp_ar'][ar]
    
        #Active region surface chromatic properties
        for ideg in range(2,5):
            if 'LD_u'+str(ideg) not in data_dic['DI']['ar_prop']['achrom']:data_dic['DI']['ar_prop']['achrom']['LD_u'+str(ideg)] = [0.]
    
        #Need to define chromatic band properties
        data_dic['DI']['ar_prop']['achrom']['w']=[None]
        data_dic['DI']['ar_prop']['achrom']['nw']=1
        data_dic['DI']['ar_prop']['chrom_mode'] = 'achrom'
        if ('chrom' in data_dic['DI']['ar_prop']):
            if (not gen_dic['specINtype']) or (len(data_dic['DI']['ar_prop']['chrom']['w'])==1):data_dic['DI']['ar_prop'].pop('chrom')
            else:
                data_dic['DI']['ar_prop']['chrom_mode'] = 'chrom'
                data_dic['DI']['ar_prop']['chrom']['w'] = np.array(data_dic['DI']['ar_prop']['chrom']['w'])
                data_dic['DI']['ar_prop']['chrom']['nw']=len(data_dic['DI']['ar_prop']['chrom']['w'])
                
                #Typical scale of chromatic variations
                w_edge = def_edge_tab(data_dic['DI']['ar_prop']['chrom']['w'][None,:][None,:])[0,0]    
                data_dic['DI']['ar_prop']['chrom']['dw'] = w_edge[1::]-w_edge[0:-1]
                data_dic['DI']['ar_prop']['chrom']['med_dw'] = np.median(data_dic['DI']['ar_prop']['chrom']['dw'])
    
        #Definition of grids discretizing active region disk to calculate active region properties
        for ar in gen_dic['studied_ar_list']:
            
            #Maximum projected active region radius
            RarRs_max = np.sin(data_dic['DI']['ar_prop']['achrom'][ar][0])

            #Default grid scaled from a 51x51 grid for an active region with projected radius equal to a hot Jupiter transiting a solar-size star
            #    - dsub_ref = (2*Rjup/Rsun)*(1/51)
            #      nsub_Dar = int(2*RpRs/dsub_ref) = int( 51*RpRs*Rsun/Rjup ) = int( 51*sin(ang)*Rsun/Rjup )  
            if (ar not in theo_dic['nsub_Dar']):
                theo_dic['nsub_Dar'][pl_loc] =int( 51.*RarRs_max*Rsun/Rjup ) 
                print('Default nsub_Dar['+str(ar)+']='+str(theo_dic['nsub_Dar'][ar]))

            #Default oversampling
            if (ar not in theo_dic['n_oversamp_ar']):
                if (theo_dic['n_oversamp_ar'] != {}):theo_dic['n_oversamp_ar'][ar] = theo_dic['n_oversamp_ar'][list(theo_dic['n_oversamp_ar'].keys())[0]]
                elif (theo_dic['n_oversamp'] != {}):theo_dic['n_oversamp_ar'][ar] = theo_dic['n_oversamp'][list(theo_dic['n_oversamp'].keys())[0]]
                else:theo_dic['n_oversamp_ar'][ar] = 1.
    
            #Corresponding active region grid
            _,theo_dic['Ssub_Sstar_ar'][ar],theo_dic['x_st_sky_grid_ar'][ar], theo_dic['y_st_sky_grid_ar'][ar],_ = occ_region_grid(RarRs_max, theo_dic['nsub_Dar'][ar],planet=False)


    #------------------------------------------------------------------------------
    #Generic path names
    gen_dic['main_pl_text'] = ''
    gen_dic['main_pl_ar_text'] = ''
    for pl_loc in gen_dic['studied_pl_list']:gen_dic['main_pl_text']+=pl_loc
    save_system = gen_dic['save_dir']+gen_dic['star_name']+'/'
    if gen_dic['main_pl_text'] != '':save_system+=gen_dic['main_pl_text']+'_'        
    gen_dic['save_data_dir'] = save_system+'Saved_data/'
    gen_dic['save_plot_dir'] = save_system+'Plots/'
    gen_dic['add_txt_path']={'DI':'','Intr':'','Diff':'','Atm':data_dic['Atm']['pl_atm_sign']+'/'}
    gen_dic['data_type_gen']={'DI':'DI','Diff':'Diff','Intr':'Intr','Absorption':'Atm','Emission':'Atm'}
    gen_dic['typegen2type']={'DI':'DI','Diff':'Diff','Intr':'Intr','Atm':data_dic['Atm']['pl_atm_sign']}  

    #Data is processed
    if gen_dic['sequence'] not in ['system_view']:    

        #------------------------------------------------------------------------------------------------------------------------
        #Model star
        #------------------------------------------------------------------------------------------------------------------------
        grid_type=[]
        if any('spec' in s for s in data_dic['DI']['type'].values()):grid_type+=['spec']
        if ('CCF' in data_dic['DI']['type'].values()):grid_type+=['ccf']  

        #Calculation of total stellar flux for use in simulated light curves
        if gen_dic['calc_flux_sc'] and (data_dic['DI']['transit_prop']['nsub_Dstar'] is not None): 
            model_star('Ftot',theo_dic,grid_type,data_dic['DI']['system_prop'],data_dic['DI']['transit_prop']['nsub_Dstar'],system_param['star'],True,True) 
                        
        #Definition of model stellar grid to calculate local or disk-integrated properties
        #    - used throughout the pipeline, unless stellar properties are fitted
        if gen_dic['theoPlOcc'] or (data_dic['DI']['ar_prop'] != {}) or (gen_dic['fit_DI_gen'] and (('custom' in data_dic['DI']['model'].values()) or ('RT_ani_macro' in data_dic['DI']['model'].values()))) or gen_dic['mock_data'] \
            or gen_dic['fit_DiffProf'] or gen_dic['fit_IntrProf'] or gen_dic['loc_prof_est']:   #or gen_dic['correct_ar'] 
    
            #Stellar grid
            model_star('grid',theo_dic,grid_type,data_dic['DI']['system_prop'],theo_dic['nsub_Dstar'],system_param['star'],True,True) 
           
            #Theoretical atmosphere
            cond_st_atm = False
            if gen_dic['mock_data']:
                for inst in mock_dic['intr_prof']:
                    if (mock_dic['intr_prof'][inst]['mode']=='theo' ):cond_st_atm = True
            if (gen_dic['fit_DI_gen'] and ('custom' in data_dic['DI']['model'].values())):
                for inst in data_dic['DI']['mod_def']:
                    if (data_dic['DI']['mod_def'][inst]['mode'] == 'theo'):cond_st_atm = True
            if gen_dic['fit_IntrProf'] and (glob_fit_dic['IntrProf']['mode'] == 'theo'):cond_st_atm = True 
            if gen_dic['loc_prof_est'] and (data_dic['Intr']['opt_loc_prof_est']['corr_mode'] in ['glob_mod','indiv_mod']) and (data_dic['Intr']['opt_loc_prof_est']['mode']=='theo'):cond_st_atm = True  
            if cond_st_atm:
                if theo_dic['st_atm']['calc']:
                    theo_dic['sme_grid'] = gen_theo_atm(theo_dic['st_atm'],system_param['star'])
                    datasave_npz(gen_dic['save_data_dir']+'Introrig_prop/IntrProf_grid',{'sme_grid':theo_dic['sme_grid']})
                else:theo_dic['sme_grid'] = dataload_npz(gen_dic['save_data_dir']+'Introrig_prop/IntrProf_grid')['sme_grid']
    
        #------------------------------------------------------------------------------------------------------------------------
    
        #Define conservative maximum shift of spectral table (km/s)
        #    - this is used to extend the common spectral table (oiginally defined in the input rest frame) so that it covers all possible wavelength shifts of the profiles in :
        # + the input rest frame, where profiles are shifted by up to the systemic rv + Keplerian semi-amplitude into the star rest frame
        #                                                       w_star ~ w_solbar * (1 - (rv_kep + rv_sys)/c) 
        # + the star rest frame, where profiles are shifted by up to the sky-projected rotational velocity into the surface rest frame
        #                                                       w_surf ~ w_star * (1 -  rv[surf/star]/c)) 
        # + the star rest frame, where profiles are shifted by up to the orbital semi-amplitude into the planetary rest frame
        #                                                      w_pl ~ w_star * (1 - rv[pl/star]/c)) 
        #    - where we approximate the Doppler shifts to define their maximum values
        #    - a negative rv is associated with a redshift of the table, so that we take the opposite of vsini and the semi-amplitude to evaluate the maximum positive extension
        gen_dic['min_rv_shift'] = 0.
        gen_dic['max_rv_shift'] = 0.
        for pl_loc in gen_dic['kepl_pl']:
            gen_dic['min_rv_shift']+=system_param[pl_loc]['Kstar_kms']
            gen_dic['max_rv_shift']-=system_param[pl_loc]['Kstar_kms']
        if gen_dic['intr_data']:
            gen_dic['min_rv_shift'] = np.max([gen_dic['min_rv_shift'], system_param['star']['vsini']])
            gen_dic['max_rv_shift'] = np.min([gen_dic['max_rv_shift'],-system_param['star']['vsini']])
        if gen_dic['pl_atm']:
            max_rv_shift_orb = 0.
            for pl_loc in gen_dic['studied_pl_list']:
                max_rv_shift_orb=system_param[pl_loc]['Kp_orb']
            gen_dic['min_rv_shift'] = np.max([gen_dic['min_rv_shift'], max_rv_shift_orb])
            gen_dic['max_rv_shift'] = np.min([gen_dic['max_rv_shift'],-max_rv_shift_orb])
        gen_dic['min_rv_shift'] -= system_param['star']['sysvel'] 
        gen_dic['min_rv_shift'] = np.max([gen_dic['min_rv_shift'],0.])   
        gen_dic['max_rv_shift'] += system_param['star']['sysvel']    
        gen_dic['max_rv_shift'] = np.min([gen_dic['max_rv_shift'],0.])          

        #------------------------------------------------------------------------------------------------------------------------
    
        #Create directories where data is saved/restored
        if (not path_exist(gen_dic['save_data_dir']+'Processed_data/Global/')):makedirs(gen_dic['save_data_dir']+'Processed_data/Global/')  
        if gen_dic['specINtype']:
            dir_name_dic={'gcal':'Processed_data/Calibration','CCF_from_sp':'Processed_data/CCFfromSpec'}
            for key in dir_name_dic.keys():
                if (gen_dic[key]) and (not path_exist(gen_dic['save_data_dir']+dir_name_dic[key]+'/')):makedirs(gen_dic['save_data_dir']+dir_name_dic[key]+'/')                   
            if (gen_dic['corr_data']):
                corr_path = gen_dic['save_data_dir']+'Corr_data/'
                if (not path_exist(corr_path)):makedirs(corr_path)
                dir_name_dic={'corr_tell':'Tell','glob_mast':'Global_Master','corr_Fbal':'Fbal','corr_FbalOrd':'Fbal/Orders','corr_Ftemp':'Ftemp','corr_cosm':'Cosm','mask_permpeak':'Permpeak','corr_wig':'Wiggles'}
                for key in dir_name_dic.keys():
                    if (gen_dic[key]) and (not path_exist(corr_path+dir_name_dic[key]+'/')):makedirs(corr_path+dir_name_dic[key]+'/')                         
                if (gen_dic['corr_wig']): 
                    
                    #Condition for exposure analysis
                    gen_dic['wig_exp_ana'] = gen_dic['wig_exp_init']['mode'] | gen_dic['wig_exp_filt']['mode'] | gen_dic['wig_exp_samp']['mode'] | gen_dic['wig_exp_nu_ana']['mode'] | gen_dic['wig_exp_fit']['mode'] | gen_dic['wig_exp_point_ana']['mode'] 
    
                    if gen_dic['wig_exp_ana']  and (not path_exist(corr_path+'Wiggles/Exp_fit/')):makedirs(corr_path+'Wiggles/Exp_fit/') 
                    if gen_dic['wig_vis_fit']['mode'] and (not path_exist(corr_path+'Wiggles/Vis_fit/')):makedirs(corr_path+'Wiggles/Vis_fit/')
                    if gen_dic['wig_corr']['mode'] and (not path_exist(corr_path+'Wiggles/Data/')):makedirs(corr_path+'Wiggles/Data/')
                if (gen_dic['corr_fring']) and (not path_exist(corr_path+'Fring/')):makedirs(corr_path+'Fring/')        
                if (gen_dic['trim_spec']) and (not path_exist(corr_path+'Trim/')):makedirs(corr_path+'Trim/')         
        if (gen_dic['detrend_prof']) and (not path_exist(gen_dic['save_data_dir']+'Detrend_prof/')):makedirs(gen_dic['save_data_dir']+'Detrend_prof/') 
        if (gen_dic['flux_sc']) and (not path_exist(gen_dic['save_data_dir']+'Scaled_data/')):makedirs(gen_dic['save_data_dir']+'Scaled_data/')
        if gen_dic['DImast_weight'] and (not path_exist(gen_dic['save_data_dir']+'DI_data/Master/')):makedirs(gen_dic['save_data_dir']+'DI_data/Master/')
        if (gen_dic['diff_data']) and (not path_exist(gen_dic['save_data_dir']+'Diff_data/')):makedirs(gen_dic['save_data_dir']+'Diff_data/')
        if gen_dic['pca_ana'] and (not path_exist(gen_dic['save_data_dir']+'PCA_results/')):makedirs(gen_dic['save_data_dir']+'PCA_results/')   
        if (gen_dic['intr_data']) and (not path_exist(gen_dic['save_data_dir']+'Intr_data/')):makedirs(gen_dic['save_data_dir']+'Intr_data/')
        if gen_dic['loc_prof_est']:
            if (not path_exist(gen_dic['save_data_dir']+'Loc_estimates/')):makedirs(gen_dic['save_data_dir']+'Loc_estimates/')        
            if (not path_exist(gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['opt_diff_prof_est']['corr_mode']+'/')):makedirs(gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['opt_loc_prof_est']['corr_mode']+'/')
        if gen_dic['diff_prof_est']:
            if (not path_exist(gen_dic['save_data_dir']+'Diff_estimates/')):makedirs(gen_dic['save_data_dir']+'Diff_estimates/')        
            if (not path_exist(gen_dic['save_data_dir']+'Diff_estimates/'+data_dic['Diff']['opt_diff_prof_est']['corr_mode']+'/')):makedirs(gen_dic['save_data_dir']+'Diff_estimates/'+data_dic['Diff']['opt_loc_prof_est']['corr_mode']+'/')          
        if (gen_dic['pl_atm']):
            if (not path_exist(gen_dic['save_data_dir']+'Atm_data/')):makedirs(gen_dic['save_data_dir']+'Atm_data/')        
            if (not path_exist(gen_dic['save_data_dir']+'Atm_data/'+data_dic['Atm']['pl_atm_sign']+'/')):makedirs(gen_dic['save_data_dir']+'Atm_data/'+data_dic['Atm']['pl_atm_sign']+'/')
        
        for data_type in ['DI','Intr','Atm']:
            if gen_dic['align_'+data_type] and (not path_exist(gen_dic['save_data_dir']+'Aligned_'+data_type+'_data/'+gen_dic['add_txt_path'][data_type])):makedirs(gen_dic['save_data_dir']+'Aligned_'+data_type+'_data/'+gen_dic['add_txt_path'][data_type])    
            if gen_dic['spec_1D_'+data_type]:
                if (not path_exist(gen_dic['save_data_dir']+data_type+'_data/1Dfrom2D/'+gen_dic['add_txt_path'][data_type])):makedirs(gen_dic['save_data_dir']+data_type+'_data/1Dfrom2D/'+gen_dic['add_txt_path'][data_type])   
                if (data_type=='Intr') and (not path_exist(gen_dic['save_data_dir']+'Diff_data/1Dfrom2D/'+gen_dic['add_txt_path'][data_type])):makedirs(gen_dic['save_data_dir']+'Diff_data/1Dfrom2D/'+gen_dic['add_txt_path'][data_type])  
            if (gen_dic[data_type+'bin'] or gen_dic[data_type+'binmultivis']) and (not path_exist(gen_dic['save_data_dir']+data_type+'bin_data/'+gen_dic['add_txt_path'][data_type])):makedirs(gen_dic['save_data_dir']+data_type+'bin_data/'+gen_dic['add_txt_path'][data_type])
            if (gen_dic['fit_'+data_type+'bin'] or gen_dic['fit_'+data_type+'binmultivis']) and (not path_exist(gen_dic['save_data_dir']+data_type+'bin_prop/'+gen_dic['add_txt_path'][data_type])):makedirs(gen_dic['save_data_dir']+data_type+'bin_prop/'+gen_dic['add_txt_path'][data_type])    
            if gen_dic['def_'+data_type+'masks'] and (not path_exist(gen_dic['save_data_dir']+'CCF_masks_'+data_type+'/'+gen_dic['add_txt_path'][data_type])):makedirs(gen_dic['save_data_dir']+'CCF_masks_'+data_type+'/'+gen_dic['add_txt_path'][data_type])    
            if gen_dic[data_type+'_CCF']:
                if (not path_exist(gen_dic['save_data_dir']+data_type+'_data/CCFfromSpec/'+gen_dic['add_txt_path'][data_type])):makedirs(gen_dic['save_data_dir']+data_type+'_data/CCFfromSpec/'+gen_dic['add_txt_path'][data_type])    
                if (data_type=='Intr') and (not path_exist(gen_dic['save_data_dir']+'Diff_data/CCFfromSpec/'+gen_dic['add_txt_path'][data_type])):makedirs(gen_dic['save_data_dir']+'Diff_data/CCFfromSpec/'+gen_dic['add_txt_path'][data_type]) 
    
        if (gen_dic['fit_DI'] or gen_dic['sav_keywords']) and (not path_exist(gen_dic['save_data_dir']+'DIorig_prop/')):makedirs(gen_dic['save_data_dir']+'DIorig_prop/')        
        if ((gen_dic['fit_Intr']) or (gen_dic['theoPlOcc'])) and (not path_exist(gen_dic['save_data_dir']+'Introrig_prop/')):makedirs(gen_dic['save_data_dir']+'Introrig_prop/')
        if (gen_dic['fit_Atm']) and (not path_exist(gen_dic['save_data_dir']+'Atmorig_prop/'+data_dic['Atm']['pl_atm_sign']+'/')):makedirs(gen_dic['save_data_dir']+'Atmorig_prop/'+data_dic['Atm']['pl_atm_sign']+'/')
    
        for key in ['IntrProp','IntrProf','AtmProf','AtmProp']:
            if (gen_dic['fit_'+key]) and (not path_exist(gen_dic['save_data_dir']+'Joined_fits/'+key+'/')):makedirs(gen_dic['save_data_dir']+'Joined_fits/'+key+'/') 

        #Create EvE output directory
        if gen_dic['EvE_outputs'] and (not path_exist(gen_dic['save_data_dir']+'EvE_outputs/')):makedirs(gen_dic['save_data_dir']+'EvE_outputs/') 

    return coord_dic,data_prop









def init_inst(mock_dic,inst,gen_dic,data_dic,theo_dic,data_prop,coord_dic,system_param,plot_dic):
    r"""**Initialization: instrument**

    Initializes instrument-specific fields for the workflow.
    
    Args:
        TBD
    
    Returns:
        None
    
    """ 
    if (gen_dic['type'][inst]=='CCF'):inst_data='CCFs'
    elif (gen_dic['type'][inst]=='spec1D'):inst_data='1D spectra'    
    elif (gen_dic['type'][inst]=='spec2D'):inst_data='2D echelle spectra'
    print('  Reading and initializing '+inst_data)    

    #Initialize dictionaries for current instrument
    for key_dic in [gen_dic,coord_dic,data_dic['DI'],data_dic,data_dic['Diff'],data_dic['Intr'],data_dic['Atm'],data_prop,theo_dic]:key_dic[inst]={} 
    DI_data_inst = data_dic['DI'][inst]
    if (inst not in gen_dic['scr_lgth']):gen_dic['scr_lgth'][inst]={}

    #Facility and reduction id
    facil_inst_all = {
        'HARPS':'ESO',
        'HARPN':'TNG',   
        'CARMENES_VIS':'CAHA','CARMENES_VIS_CCF':'CAHA',
        'CORALIE':'ESO',
        'ESPRESSO':'ESO','ESPRESSO_MR':'ESO',
        'EXPRES':'DCT',
        'MIKE_Red':'LCO','MIKE_Blue':'LCO',
        'IGRINS2_Blue':'Gemini-N','IGRINS2_Red':'Gemini-N',
        'MAROONX_Blue':'Gemini-N','MAROONX_Red':'Gemini-N',
        'NIGHT':'ESO',    #UPDATE TO 'OHP' WHEN FINAL FORMAT AVAILABLE
        'NIRPS_HE':'ESO','NIRPS_HA':'ESO',
        'SOPHIE':'OHP',
        }
    if inst not in facil_inst_all:stop('ERROR : define facility ID for '+inst+' in "facil_inst_all"')
    facil_inst = facil_inst_all[inst]

    #Error definition
    if not gen_dic['mock_data']:
        gen_dic['flag_err_inst'][inst] = flag_err_inst(inst)
        if (not gen_dic['flag_err_inst'][inst]) and gen_dic['gcal']:stop('ERROR : error table must be available to estimate calibration')
        if (inst in gen_dic['force_flag_err']):gen_dic['flag_err_inst'][inst]=False
        if gen_dic['flag_err_inst'][inst]:print('   > Errors propagated from raw data')
        else:print('   > Beware: custom definition of errors')
    else:
        if (inst in mock_dic['set_err']) and (mock_dic['set_err'][inst]):gen_dic['flag_err_inst'][inst] = True
        else:gen_dic['flag_err_inst'][inst] = False

    #Mask used to compute CCF on stellar lines
    if (gen_dic['CCF_from_sp']) and (inst in gen_dic['CCF_mask']):
        
        #Upload CCF mask
        ext = gen_dic['CCF_mask'][inst].split('.')[-1]
        if (ext=='fits'):
            hdulist = fits.open(gen_dic['CCF_mask'][inst])
            data_loc = hdulist[1].data
            gen_dic['CCF_mask_wav'][inst] = data_loc['lambda']
            if gen_dic['use_maskW']:gen_dic['CCF_mask_wgt'][inst] = data_loc['contrast']          
            hdulist.close()

        elif (ext=='csv'):
            data_loc = np.genfromtxt(gen_dic['CCF_mask'][inst], delimiter=',', names=True)
            gen_dic['CCF_mask_wav'][inst] = data_loc['wave']
            if gen_dic['use_maskW']:gen_dic['CCF_mask_wgt'][inst] = data_loc['contrast']  

        elif (ext in ['txt','dat']):
            data_loc = np.loadtxt(gen_dic['CCF_mask'][inst]).T            
            gen_dic['CCF_mask_wav'][inst] = data_loc[0]
            if gen_dic['use_maskW']:gen_dic['CCF_mask_wgt'][inst] = data_loc[1]       
        else:
            stop('ERROR : CCF mask extension TBD') 

        #No weighing of lines
        if not gen_dic['use_maskW']:gen_dic['CCF_mask_wgt'][inst] = np.repeat(1.,len(gen_dic['CCF_mask_wav'][inst]))  
            
        #Same convention as the ESPRESSO DRS, which takes the 'contrast' column from mask files and use its squares as weights
        #    - masks weights should have been normalized so that mean( weights ) = 1, where weight = CCF_mask_wgt^2
        gen_dic['CCF_mask_wgt'][inst]=gen_dic['CCF_mask_wgt'][inst]**2.

    #Resampling exposures of a given visit on a common table or not
    #    - CCFs will be resampled on a common RV table, as we consider they should only be used for preliminary analysis
    if gen_dic['type'][inst]=='CCF':gen_dic['comm_sp_tab'][inst]=True
    elif (inst not in gen_dic['comm_sp_tab']):gen_dic['comm_sp_tab'][inst]=False
    if (not gen_dic['comm_sp_tab'][inst]):print('   > Data processed on individual spectral tables for each exposure')      
    else:print('   > Data resampled on a common spectral table')

    #Flux balance correction
    gen_dic['corr_FbalOrd_inst'][inst] = deepcopy(gen_dic['corr_FbalOrd'])
    if gen_dic['corr_FbalOrd_inst'][inst]:
        if (gen_dic['type'][inst]!='spec2D'):
            print('WARNING: input profiles for '+inst+' are not in 2D format, order flux balance correction will not be applied.') 
            gen_dic['corr_FbalOrd_inst'][inst] = False
        elif (inst=='ESPRESSO'):
            print('WARNING: it is advised to disable order flux balance correction with ESPRESSO 2D spectra, as it may partly absorb wiggle signal.') 

    #Deactivate conditions
    if gen_dic['pl_atm'] and (data_dic['Intr']['opt_loc_prof_est']['corr_mode'] in ['Intrbin','rec_prof']) and (data_dic['Atm']['type'][inst]!=data_dic['Intr']['type'][inst]):
        stop('Error: intrinsic profiles for '+inst+' (in '+data_dic['Intr']['type'][inst]+') must have the same format as planetary profiles (in '+data_dic['Atm']['type'][inst]+') if also requested for their extraction)')
    

    ##############################################################################################################################
    #Retrieval and pre-processing of data
    ##############################################################################################################################    

    #Data is calculated and not retrieved
    if (gen_dic['calc_proc_data']):
        print('         Calculating data')

        #Initialize dictionaries for current instrument
        data_inst={'visit_list':[]}
        data_dic[inst]=data_inst  

        #Initialize current data type for the instrument
        #    - data need to be processed in their input mode at the start of each new instrument or visit
        #    - if mode is switched to CCFs/s1d at some point in the reduction of a visit, the visit type will be switched but the instrument type will remain the same, so that the next visits of the instrument are processed in their original mode
        #      only after all visits have been processed is the instrument type switched
        data_inst['type']=deepcopy(gen_dic['type'][inst])
        if (inst=='NIGHT') and (data_inst['type']!='spec2D'):stop('ERROR : wrong data type for NIGHT (only S2D are available)')
        
        #Total number of orders for current instrument
        #    - we define an artificial order that contains the CCF or 1D spectrum, so that the pipeline can process in the same way as with 2D spectra
        #    - if orders are fully removed from an instrument its default structure is updated
        #      if orders are trimmed after the spectral reduction the default structure is kept
        #    - mock data created with a single order
        if gen_dic['mock_data']: 
            data_inst['nord'] = 1
            gen_dic[inst]['norders_instru'] = 1
        else:
            norders_instru_ref = return_spec_nord(inst)
            data_inst['idx_ord_ref']=deepcopy(np.arange(norders_instru_ref))      #to keep track of the original orders indexes
            idx_ord_kept = list(np.arange(norders_instru_ref))
            gen_dic[inst]['wav_ord_inst'] = return_cen_wav_ord(inst)
            if (data_inst['type'] in ['spec1D','CCF']):
                data_inst['nord'] = 1
                gen_dic[inst]['norders_instru']=norders_instru_ref
            elif (data_inst['type']=='spec2D'):
                if (inst in gen_dic['del_orders']) and (len(gen_dic['del_orders'][inst])>0):
                    print('           Removing '+str(len(gen_dic['del_orders'][inst]))+' orders from the analysis')
                    idx_ord_kept = list(np.delete(np.arange(norders_instru_ref),gen_dic['del_orders'][inst]))
                    gen_dic[inst]['wav_ord_inst'] = gen_dic[inst]['wav_ord_inst'][idx_ord_kept]
                    data_inst['idx_ord_ref'] = data_inst['idx_ord_ref'][idx_ord_kept]
                gen_dic[inst]['norders_instru'] = len(idx_ord_kept)
                data_inst['nord']=deepcopy(len(idx_ord_kept))   
        data_inst['nord_spec']=deepcopy(data_inst['nord'])                   #to keep track of the number of orders after spectra are trimmed
        data_inst['nord_ref']=deepcopy(gen_dic[inst]['norders_instru'])      #to keep track of the original number of orders (after deletion)
   
        #Orders contributing to calculation of spectral CCFs
        #    - set automatically in case of S1D to use the generic pipeline structure, even though S1D have no orders
        #    - indexes are made relative to order list after initial selection
        if data_inst['type']=='spec1D':gen_dic[inst]['orders4ccf']=[0]
        elif data_inst['type']=='spec2D':
            if (inst in gen_dic['orders4ccf']): gen_dic[inst]['orders4ccf'] = np.intersect1d(gen_dic['orders4ccf'][inst],idx_ord_kept,return_indices=True)[2]     
            else:gen_dic[inst]['orders4ccf'] = np.arange(gen_dic[inst]['norders_instru'],dtype=int)

        #Weighing calibration condition
        #    - S2D calibration is calculated for scaling even if not needed for weights
        #    - condition below is deactivated if conversion into CCFs or 2D->1D
        data_inst['gcal_blaze_vis']=[]
        if (data_inst['type']=='spec2D') and gen_dic['cal_weight']:data_inst['cal_weight'] = True
        else:data_inst['cal_weight'] = False

        #Initialize flag that exposures in all visits of the instrument share a common spectral table
        data_inst['comm_sp_tab'] = True 

        #Initialize list of instrument visits contained within a single night
        data_inst['single_night'] = []
        
        #Calibration settings
        if (inst not in gen_dic['gcal_nooutedge']):gen_dic['gcal_nooutedge'][inst] = [0.,0.]

        #Processing each visit
        if gen_dic['mock_data']:vis_list=list(mock_dic['visit_def'][inst].keys())
        else:vis_list=list(gen_dic['data_dir_list'][inst].keys())
        data_inst['dates']={}
        data_inst['midpoints']={}
        ivis=-1
        for vis in vis_list:
            
            #Artificial data
            if gen_dic['mock_data']:
                
                #Generate time table
                if 'bin_low' in mock_dic['visit_def'][inst][vis]:
                    bjd_exp_low = np.array(mock_dic['visit_def'][inst][vis]['bin_low'])
                    bjd_exp_high = np.array(mock_dic['visit_def'][inst][vis]['bin_high'])
                    n_in_visit = len(bjd_exp_low)
                elif 'exp_range' in mock_dic['visit_def'][inst][vis]:
                    dbjd =  (mock_dic['visit_def'][inst][vis]['exp_range'][1]-mock_dic['visit_def'][inst][vis]['exp_range'][0])/mock_dic['visit_def'][inst][vis]['nexp']
                    n_in_visit = int((mock_dic['visit_def'][inst][vis]['exp_range'][1]-mock_dic['visit_def'][inst][vis]['exp_range'][0])/dbjd)
                    bjd_exp_low = mock_dic['visit_def'][inst][vis]['exp_range'][0] + dbjd*np.arange(n_in_visit)
                    bjd_exp_high = bjd_exp_low+dbjd      
                bjd_exp_all = 0.5*(bjd_exp_low+bjd_exp_high)        
                
            #Observational data        
            else:
                
                #Adding "/" in case user forgets
                vis_path_root = path_singslash(gen_dic['data_dir_list'][inst][vis]+'/')
        
                #List of all exposures for current instrument
                if inst in ['SOPHIE']:
                    vis_path= vis_path_root+ {
                        'CCF':'*ccf*',
                        'spec2D':'*e2ds_',
                        }[data_inst['type']]
                elif inst in ['CORALIE']:
                    vis_path= vis_path_root+ {
                        'CCF':'*ccf*',
                        'spec2D':'*e2ds_',
                        }[data_inst['type']]
                elif inst=='CARMENES_VIS':
                    vis_path= vis_path_root+ {
                        'spec2D':'*sci-allr-vis_',
                        }[data_inst['type']]                    
                elif inst=='CARMENES_VIS_CCF':     #current reduction is not standard, better to consider as independent instrument
                    vis_path= vis_path_root+ {
                        'CCF':'*CCF*',
                        }[data_inst['type']]  
                elif inst in ['ESPRESSO','ESPRESSO_MR','HARPN','HARPS']:
                    vis_path= vis_path_root+ {
                        'CCF':'*CCF_',
                        'spec1D':'*S1D_',
                        'spec2D':'*S2D_',
                        }[data_inst['type']] 
                elif inst=='NIGHT':
                    vis_path= vis_path_root+'*S2D_'                   
                elif inst in ['NIRPS_HA','NIRPS_HE']:
                    vis_path= vis_path_root+ {
                        'CCF':'*CCF_TELL_CORR_',       #CCF from telluric-corrected spectra
                        'spec1D':'*S1D_',
                        'spec2D':'*S2D_',
                        }[data_inst['type']] 
                elif inst in ['EXPRES']:
                    vis_path= vis_path_root+ {
                        'spec2D':'*',
                        }[data_inst['type']]                      
                else:stop('ERROR : root names undefined for '+inst)

                #Nominal data files
                if (inst not in ['EXPRES','CARMENES_VIS_CCF']):nom_ext = 'A.fits'
                else:nom_ext = '.fits'
                vis_path_exp = np.array(glob.glob(vis_path+nom_ext))
                n_in_visit=len(vis_path_exp) 
                if n_in_visit==0:stop('ERROR : No data found at '+vis_path+nom_ext+'. Check path.')
          
                #Retrieved file root names
                #    - so that we can retrieve sky-corrected and blaze files in the same order
                exp_rootnames = [path.split(vis_path_root)[1].split(nom_ext)[0] for path in vis_path_exp]

                #Retrieving sky-corrected data
                if (inst in gen_dic['fibB_corr']) and (vis in gen_dic['fibB_corr'][inst]) and (len(gen_dic['fibB_corr'][inst][vis])>0):
                    if inst in ['ESPRESSO','ESPRESSO_MR','HARPS','HARPN','NIRPS_HA','NIRPS_HE']:skcorr_ext = 'SKYSUB_A.fits'
                    else:skcorr_ext = 'C.fits'                      
                    vis_path_skysub_exp = np.array([ vis_path_root+ exp_rootname+skcorr_ext for exp_rootname in exp_rootnames   ])
                    exp_filenames = np.array([ exp_rootname+skcorr_ext for exp_rootname in exp_rootnames   ])
                    if len(vis_path_skysub_exp)==0:stop('ERROR : No sky-sub data found. Check path.') 
                    
                    #Orders to be replaced
                    if gen_dic['fibB_corr'][inst][vis]=='all':idx_ord_skysub = 'all'
                    else:idx_ord_skysub = gen_dic['fibB_corr'][inst][vis]
                    
                else:
                    vis_path_skysub_exp = None
                    exp_filenames = np.array([ exp_rootname+nom_ext for exp_rootname in exp_rootnames   ])

                #Retrieving blazed data
                if (data_inst['type']=='spec2D') and gen_dic['gcal_blaze']:   
                    if (inst in ['ESPRESSO','ESPRESSO_MR','HARPN','HARPS','NIRPS_HA','NIRPS_HE']):
                        vis_path_blaze_exp = np.array([ vis_path_root+ exp_rootname+'BLAZE_A.fits' for exp_rootname in exp_rootnames   ])                        
                        if len(vis_path_blaze_exp)==0:print('No blaze data found. Switching to estimate mode.')
                        else:data_inst['gcal_blaze_vis']+=[vis]
                    else:print('Blaze names undefined for '+inst+'. Switching to estimate mode.')

            #Remove/keep visits
            if (vis in gen_dic['unused_visits'][inst]):   
                print('       > Visit '+vis+' is removed')
            else:
                print('         Initializing visit '+vis)
                data_dic[inst]['visit_list']+=[vis]     
                ivis+=1

                #Print visit date if available 
                #    - taking the day at the start of the night, rather than the day at the start of the exposure series
                if (not gen_dic['mock_data']) and (inst in ['CORALIE','ESPRESSO','ESPRESSO_MR','HARPS','HARPN','CARMENES_VIS','CARMENES_VIS_CCF','EXPRES','NIGHT','NIRPS_HA','NIRPS_HE']):    
                    vis_day_exp_all=[]
                    vis_hour_exp_all=[]
                    bjd_exp_all = []
                    for file_path in vis_path_exp:
                        hdulist =fits.open(file_path)
                        hdr =hdulist[0].header 
                        if inst=='CORALIE':
                            vis_day_exp_all+= [int(hdr['HIERARCH ESO CORA SHUTTER START DATE'][6:8])]
                            vis_hour_exp_all+=[float(hdr['HIERARCH ESO CORA SHUTTER START HOUR'])]      
                            bjd_exp_all  += [ hdr['HIERARCH ESO DRS BJD'] - 2400000. ]
                        elif inst=='EXPRES':
                            vis_day_exp_all+= [int(hdr['DATE-OBS'].split(' ')[0].split('-')[2]) ]  
                            vis_hour_exp_all+=[int(hdr['DATE-OBS'].split(' ')[1].split(':')[0])]                            
                        elif inst=='CARMENES_VIS_CCF':bjd_exp_all  += [  hdr['BJD']  ] 
                        elif inst in ['ESPRESSO','ESPRESSO_MR','HARPS','HARPN','CARMENES_VIS','NIGHT','NIRPS_HA','NIRPS_HE']:
                            vis_day_exp_all+= [int(hdr['DATE-OBS'].split('T')[0].split('-')[2]) ]  
                            vis_hour_exp_all+=[int(hdr['DATE-OBS'].split('T')[1].split(':')[0])]
                            if inst in ['ESPRESSO','ESPRESSO_MR','HARPS','HARPN','NIGHT','NIRPS_HA','NIRPS_HE']:bjd_exp_all +=[  hdr['HIERARCH '+facil_inst+' QC BJD'] - 2400000. ]
                            elif inst=='CARMENES_VIS':bjd_exp_all  += [  hdr['CARACAL BJD']  ]                        
                    if inst=='CORALIE':
                        vis_yr = hdr['HIERARCH ESO CORA SHUTTER START DATE'][0:4]
                        vis_mt = hdr['HIERARCH ESO CORA SHUTTER START DATE'][4:6]
                    elif inst=='EXPRES':
                        vis_yr = hdr['DATE-OBS'].split(' ')[0].split('-')[0]
                        vis_mt = hdr['DATE-OBS'].split(' ')[0].split('-')[1]                            
                    elif (inst not in ['CARMENES_VIS_CCF']):
                        vis_yr = hdr['DATE-OBS'].split('T')[0].split('-')[0]
                        vis_mt = hdr['DATE-OBS'].split('T')[0].split('-')[1]  
                          
                    #BJD midpoint
                    bjd_exp_all = np.array(bjd_exp_all)
                    bjd_vis = np.mean(bjd_exp_all)
                    data_inst['midpoints'][vis] = '%.5f'%(bjd_vis+2400000.)+' BJD'
                        
                    #Take the day before as reference if exposure is past midnight (ie, not between 12 and 23) 
                    if len(vis_day_exp_all)>0:
                        vis_day_exp_all = np.array(vis_day_exp_all)
                        vis_hour_exp_all = np.array(vis_hour_exp_all)
                        vis_day_exp_all[vis_hour_exp_all<=12]-=1
                        if np.sum(vis_day_exp_all==0)>0:stop('ERROR: adapt date retrieval')                           
                        
                        #Check wether visit is contained within a single night
                        #    - either:
                        # + all exposures on same day before midnight or after midnight
                        # + all exposures within two consecutive days between noon and midnight 
                        min_day = np.min(vis_day_exp_all)
                        max_day = np.max(vis_day_exp_all) 
                        min_hr = np.min(vis_hour_exp_all)
                        max_hr = np.max(vis_hour_exp_all)
                        if ((max_day==min_day) and ((min_hr>=12) or (max_hr<=12))) or ((max_day==min_day+1) and ((min_hr>=12) and (max_hr<=12))): 
                            data_inst['single_night']+=[vis]
                            vis_day = np.min(vis_day_exp_all)  
                            vis_day_txt = '0'+str(vis_day) if vis_day<10 else str(vis_day)        
                            data_inst['dates'][vis] = str(vis_yr)+'/'+str(vis_mt)+'/'+str(vis_day_txt)
                        else:                    
                            vis_day_txt_all = np.array(['0'+str(vis_day) if vis_day<10 else str(vis_day) for vis_day in vis_day_exp_all])
                    else:
                        print('WARNING: data could not be retrieved')
                        if (np.max(bjd_exp_all) - np.min(bjd_exp_all))<1.:data_inst['single_night']+=[vis]
                            
                else:
                    bjd_vis = np.mean(bjd_exp_all) - 2400000.    
                    data_inst['single_night']+=[vis]

                #Initializing dictionaries for visit
                theo_dic[inst][vis]={}
                data_dic['Atm'][inst][vis]={}   
                data_inst[vis] = {'n_in_visit':n_in_visit,'studied_pl':[],'studied_ar':[],'comm_sp_tab':True} 
                coord_dic[inst][vis] = {}
                for pl_loc in gen_dic['studied_pl_list']:
                    if (inst in gen_dic['studied_pl'][pl_loc]) and (vis in gen_dic['studied_pl'][pl_loc][inst]):data_inst[vis]['studied_pl']+=[pl_loc]
                for ar in gen_dic['studied_ar']:
                    if (inst in gen_dic['studied_ar'][ar]) and (vis in gen_dic['studied_ar'][ar][inst]):data_inst[vis]['studied_ar']+=[ar]                    
                data_prop[inst][vis] = {}
                data_dic_temp={}
                gen_dic[inst][vis] = {}
                DI_data_inst[vis] = {}  
                         
                #Associating telluric spectrum with each exposure
                if not gen_dic['mock_data']:data_inst[vis]['tell_DI_data_paths']={}

                #Exposure-specific calibration profile for weighing
                #    - the path is created even if weights are not needed to store temporarily blaze-derived calibration profiles
                if data_inst['cal_weight'] or (vis in data_inst['gcal_blaze_vis']):data_inst[vis]['sing_gcal_DI_data_paths']={}             
                
                #Initialize current data type for the visit
                data_inst[vis]['type']=deepcopy(data_inst['type'])                
                
                #Initialize exposure tables
                #    - spatial positions in x/y in front of the stellar disk
                # + x position along the node line
                #  y position along the projection of the normal to the orbital plane
                # +calculated also for the out-of-transit positions, as in case of exposures binning some of the 
                #      ingress/egress exposures may contribute to the binned exposures average positions            
                #    - phase of in/out exposures, per instrument and per visit   
                #    - radial velocity of planet in star rest frame  
                for key in ['bjd','t_dur','RV_star_solCDM','RV_star_stelCDM','cen_ph_st','st_ph_st','end_ph_st']:coord_dic[inst][vis][key] = np.zeros(n_in_visit,dtype=float)*np.nan
                for pl_loc in list(set(gen_dic['studied_pl_list']+gen_dic['kepl_pl'])):
                    coord_dic[inst][vis][pl_loc]={} 
                    if pl_loc in data_inst[vis]['studied_pl']:
                        for key in ['ecl','cen_ph','st_ph','end_ph','ph_dur','rv_pl','v_pl']:coord_dic[inst][vis][pl_loc][key] = np.zeros(n_in_visit,dtype=float)*np.nan
                        for key in ['cen_pos','st_pos','end_pos']:coord_dic[inst][vis][pl_loc][key] = np.zeros([3,n_in_visit],dtype=float)*np.nan
             
                    #Definition of mid-transit times for each planet associated with the visit 
                    if (pl_loc in gen_dic['Tcenter_visits']) and (inst in gen_dic['Tcenter_visits'][pl_loc]) and (vis in gen_dic['Tcenter_visits'][pl_loc][inst]):
                        if vis not in data_inst['single_night']:stop('ERROR : gen_dic["Tcenter_visits"] not available for multi-epochs visits')
                        coord_dic[inst][vis][pl_loc]['Tcenter'] = gen_dic['Tcenter_visits'][pl_loc][inst][vis]
                    else:
                        norb = round((bjd_vis+2400000.-system_param[pl_loc]['TCenter'])/system_param[pl_loc]["period"])
                        coord_dic[inst][vis][pl_loc]['Tcenter']  = system_param[pl_loc]['TCenter']+norb*system_param[pl_loc]["period"]

                #Observation properties
                for key in ['AM','IWV_AM','TEMP','PRESS','seeing','colcorrmin','colcorrmax','mean_SNR','alt','az','BERV']:data_prop[inst][vis][key] = np.zeros(n_in_visit,dtype=float)*np.nan
                data_prop[inst][vis]['exp_filenames']=np.empty([n_in_visit],dtype='U50')
                if inst=='ESPRESSO_MR':
                    for key in ['AM_UT','seeing_UT']:data_prop[inst][vis][key] = np.zeros(n_in_visit,dtype=object)*np.nan 
                data_prop[inst][vis]['PSF_prop'] = np.zeros([n_in_visit,4],dtype=float)
                data_prop[inst][vis]['SNRs']=np.zeros([n_in_visit,data_inst['nord_ref']],dtype=float)*np.nan 
                if inst in ['ESPRESSO','HARPN','HARPS','NIRPS_HA','NIRPS_HE']:
                    data_prop[inst][vis]['colcorr_ord']=np.zeros([n_in_visit,data_inst['nord_ref']],dtype=float)*np.nan
                    for key in ['BLAZE_A','BLAZE_B','WAVE_MATRIX_THAR_FP_A','WAVE_MATRIX_THAR_FP_B']:data_prop[inst][vis][key]=np.empty([n_in_visit],dtype='U35')
                    data_prop[inst][vis]['satur_check']=np.zeros([n_in_visit],dtype=int)*np.nan
                    data_prop[inst][vis]['adc_prop']=np.zeros([n_in_visit,6],dtype=float)*np.nan
                    data_prop[inst][vis]['piezo_prop']=np.zeros([n_in_visit,4],dtype=float)*np.nan
                if inst=='ESPRESSO':
                    data_prop[inst][vis]['guid_coord']=np.zeros([n_in_visit,2],dtype=float)*np.nan
                    
                #Set error flag per visit
                #    - same for a given instrument, but this allows dealing with binned visits
                gen_dic[inst][vis]['flag_err']=deepcopy(gen_dic['flag_err_inst'][inst])     
        
                #Raw CCF properties
                if inst in ['HARPN','HARPS','CORALIE','SOPHIE','ESPRESSO','ESPRESSO_MR','NIRPS_HA','NIRPS_HE']:
                    for key in ['RVdrift','rv_pip','FWHM_pip','ctrst_pip']:DI_data_inst[vis][key] =  np.zeros(n_in_visit,dtype=float)*np.nan  
                    for key in ['erv_pip','eFWHM_pip','ectrst_pip']:DI_data_inst[vis][key] =  np.zeros(n_in_visit,dtype=float)  

                #Activity indexes
                data_inst[vis]['act_idx'] = []
                if inst in ['HARPN','HARPS','CORALIE','SOPHIE','ESPRESSO','ESPRESSO_MR','NIRPS_HA','NIRPS_HE'] and gen_dic['DACE_sp']:
                    DACE_sp = True

                    #Corresponding DACE star name, instrument and visit
                    dace_name =  hdr['HIERARCH '+facil_inst+' OBS TARG NAME']
                    DACE_sp = Spectroscopy.get_timeseries(dace_name, sorted_by_instrument=False)

                    #Condition to retrieve current visit
                    dace_red_inst = hdr['INSTRUME']
                    if vis in data_inst['single_night']:
                        if dace_red_inst=='ESPRESSO': 
                            if bjd_vis<=58649.5:dace_red_inst+='18'
                            else:dace_red_inst+='19'  
                        elif dace_red_inst=='HARPS': 
                            if bjd_vis<=57167.5:dace_red_inst+='03'
                            else:dace_red_inst+='15'  
                        idx_dace_vis = np_where1D((np.asarray(DACE_sp['ins_name']) == dace_red_inst) & (np.asarray(DACE_sp['date_night']) == vis_yr+'-'+vis_mt+'-'+vis_day_txt))
                    else:
                        if dace_red_inst=='ESPRESSO':inst_off_all = ['18' if bjd_loc<58649.5 else '19' for bjd_loc in bjd_exp_all]
                        elif dace_red_inst=='HARPS':inst_off_all = ['03' if bjd_loc<57167.5 else '15' for bjd_loc in bjd_exp_all]   
                        dace_red_inst_all = np.array([dace_red_inst+inst_off for inst_off in inst_off_all])
                        idx_dace_vis = np.zeros(0,dtype=int)
                        vis_day_txt_un,idx_vis_day_txt_un = np.unique(vis_day_txt_all,return_index=True)
                        for vis_day_txt,dace_red_inst in zip(vis_day_txt_un,dace_red_inst_all[idx_vis_day_txt_un]):
                            idx_dace_vis = np.append(idx_dace_vis, np_where1D((np.asarray(DACE_sp['ins_name']) == dace_red_inst) & (np.asarray(DACE_sp['date_night']) == vis_yr+'-'+vis_mt+'-'+vis_day_txt)))
                    if len(idx_dace_vis)==0:
                        print('         Visit not found in DACE') 
                        DACE_sp = False
                    else:

                        #Retrieve activity indexes and bjd
                        DACE_idx = {}
                        DACE_idx['bjd'] =  np.array(DACE_sp['rjd'],dtype=float)[idx_dace_vis]
                        data_inst[vis]['act_idx'] = ['ha','na','ca','s','rhk']
                        for key in data_inst[vis]['act_idx']:
                            data_prop[inst][vis][key] = np.zeros([n_in_visit,2],dtype=float)*np.nan
                            if key=='rhk':dace_key = 'rhk'
                            else:dace_key = key+'index'
                            DACE_idx[key] = np.array(DACE_sp[dace_key],dtype=float)[idx_dace_vis]
                            DACE_idx[key+'_err'] = np.array(DACE_sp[dace_key+'_err'],dtype=float)[idx_dace_vis]                    
                else:
                    DACE_sp = False
                    if inst=='EXPRES': 
                        data_inst[vis]['act_idx'] = ['ha','s']
                        for key in data_inst[vis]['act_idx']:data_prop[inst][vis][key] = np.zeros([n_in_visit,2],dtype=float)*np.nan
                    
                #------------------------------------------------------------------------------------------------------------     

                #Simulated active regions 
                if (len(data_inst[vis]['studied_ar'])>0):
                    if gen_dic['mock_data']:
                        if (inst in mock_dic['ar_prop']) and (vis in mock_dic['ar_prop'][inst]):ar_prop_nom = retrieve_ar_prop_from_param(mock_dic['ar_prop'][inst][vis], inst, vis)
                        else:stop('ERROR: active regions are required in visit '+vis+' but their mock properties are not defined')
                    else:
                        if (inst in theo_dic['ar_prop']) and (vis in theo_dic['ar_prop'][inst]):ar_prop_nom = retrieve_ar_prop_from_param(theo_dic['ar_prop'][inst][vis], inst, vis)
                        else:stop('ERROR: active regions are required in visit '+vis+' but their theoretical properties are not defined')
                    ar_prop_nom['cos_istar']=system_param['star']['cos_istar']
                    for ar in data_inst[vis]['studied_ar']: 
                        coord_dic[inst][vis][ar]={}
                        for key in gen_dic['ar_coord_par']:coord_dic[inst][vis][ar][key] = np.zeros([3,n_in_visit],dtype=float)*np.nan
                        coord_dic[inst][vis][ar]['is_visible'] = np.zeros([3,n_in_visit],dtype=bool) 
                        for key in ['Tc_ar', 'ang_rad','lat_rad', 'fctrst']:coord_dic[inst][vis][ar][key] = ar_prop_nom[ar][key]  

                #Pre-processing all exposures in visit
                for isub_exp,iexp in enumerate(range(n_in_visit)):

                    #Artificial data            
                    if gen_dic['mock_data']:                
                        coord_dic[inst][vis]['bjd'][iexp] = bjd_exp_all[iexp]  - 2400000.  
                        coord_dic[inst][vis]['t_dur'][iexp] = (bjd_exp_high[iexp]-bjd_exp_low[iexp])*24.*3600.

                    #Observational data
                    else:
                        
                        #File name
                        data_prop[inst][vis]['exp_filenames'][iexp] = exp_filenames[iexp]

                        #Data of the fits file 
                        hdulist =fits.open(vis_path_exp[iexp])
               
                        #Header 
                        hdr =hdulist[0].header 

                        #Retrieve exposure bjd in instrument/visit table
                        if inst in ['HARPS','HARPN','ESPRESSO','ESPRESSO_MR','NIGHT','NIRPS_HA','NIRPS_HE']:
                            bjd_exp =  hdr['HIERARCH '+facil_inst+' QC BJD'] - 2400000.      
                        elif inst in ['CORALIE']:bjd_exp =  hdr['HIERARCH ESO DRS BJD'] - 2400000.  
                        elif inst in ['SOPHIE']:bjd_exp =  hdr['HIERARCH OHP DRS BJD'] - 2400000.  
                        elif inst in ['CARMENES_VIS']:bjd_exp =  hdr['CARACAL BJD']   
                        elif inst=='EXPRES':bjd_exp = hdulist[1].header['BARYMJD'] +0.5
                        else:stop('ERROR : bjd keyword for '+inst+' is undefined')
                        coord_dic[inst][vis]['bjd'][iexp] = bjd_exp
                      
                        #Exposure time (s)    
                        if inst=='SOPHIE':             
                            
                            #start time
                            st_hmns=(hdr['HIERARCH OHP OBS DATE START'].split('T')[1]).split(':')
                            h_start=float(st_hmns[0])            
                            mn_start=float(st_hmns[1]) 
                            s_start=float(st_hmns[2]) 
                            
                            #end time
                            end_hmns=(hdr['HIERARCH OHP OBS DATE END'].split('T')[1]).split(':')  
                            h_end=float(end_hmns[0])
                            if h_end<h_start:h_end+=24.
                            mn_end=float(end_hmns[1]) 
                            s_end=float(end_hmns[2])
                    
                            #exposure time (s)
                            coord_dic[inst][vis]['t_dur'][iexp]=( (h_end-h_start)*3600. + (mn_end-mn_start)*60. + (s_end-s_start) )

                        elif inst=='EXPRES':coord_dic[inst][vis]['t_dur'][iexp] = hdr['AEXPTIME']  
                        else:coord_dic[inst][vis]['t_dur'][iexp] = hdr['EXPTIME']  
                
                        #RV drift  
                        #    - drift is measured from the calibration lamps or FP 
                        #    - drift is corrected automatically in new reductions for ESPRESSO and kin spectrographs, direcly included in the wavelength solution
                        if inst in ['CORALIE']:
                            DI_data_inst[vis]['RVdrift'][iexp]=hdr['HIERARCH ESO DRS DRIFT SPE RV']  #in m/s  
                        elif inst=='SOPHIE':
                            DI_data_inst[vis]['RVdrift'][iexp]=hdr['HIERARCH OHP DRS DRIFT RV']      #in m/s                 
                        elif inst in ['HARPS','HARPN','ESPRESSO','ESPRESSO_MR','NIRPS_HA','NIRPS_HE']:
                            if 'HIERARCH '+facil_inst+' QC DRIFT DET0 MEAN' in hdr:
                                DI_data_inst[vis]['RVdrift'][iexp]=hdr['HIERARCH '+facil_inst+' QC DRIFT DET0 MEAN']*return_pix_size(inst)*1e3    #in pix -> m/s
                    
                    #Stellar phase
                    if gen_dic['sequence'] not in ['st_master_tseries']:
                        coord_dic[inst][vis]['st_ph_st'][iexp],coord_dic[inst][vis]['cen_ph_st'][iexp],coord_dic[inst][vis]['end_ph_st'][iexp] = get_timeorbit(system_param['star']['Tcenter'],coord_dic[inst][vis]['bjd'][iexp], {'period':system_param['star']['Peq']}, coord_dic[inst][vis]['t_dur'][iexp])[0:3] 

                    #Orbital coordinates for each studied planet
                    for pl_loc in data_inst[vis]['studied_pl']:
                        coord_dic[inst][vis][pl_loc]['cen_pos'][:,iexp],coord_dic[inst][vis][pl_loc]['st_pos'][:,iexp],coord_dic[inst][vis][pl_loc]['end_pos'][:,iexp],coord_dic[inst][vis][pl_loc]['ecl'][iexp],coord_dic[inst][vis][pl_loc]['rv_pl'][iexp],coord_dic[inst][vis][pl_loc]['v_pl'][iexp],\
                        coord_dic[inst][vis][pl_loc]['st_ph'][iexp],coord_dic[inst][vis][pl_loc]['cen_ph'][iexp],coord_dic[inst][vis][pl_loc]['end_ph'][iexp],coord_dic[inst][vis][pl_loc]['ph_dur'][iexp]=coord_expos(pl_loc,coord_dic,inst,vis,system_param['star'],
                                            system_param[pl_loc],coord_dic[inst][vis]['bjd'][iexp],coord_dic[inst][vis]['t_dur'][iexp],data_dic,data_dic['DI']['system_prop']['achrom'][pl_loc][0])                    
                        
                    #Surface coordinates for each studied active region  
                    for ar in data_inst[vis]['studied_ar']:
                        ar_prop_exp = coord_expos_ar(ar,coord_dic[inst][vis]['bjd'][iexp],ar_prop_nom,system_param['star'],coord_dic[inst][vis]['t_dur'][iexp],gen_dic['ar_coord_par'])                           
                        for key in ar_prop_exp:coord_dic[inst][vis][ar][key][:, iexp] = [ar_prop_exp[key][0],ar_prop_exp[key][1],ar_prop_exp[key][2]]  

                #--------------------------------------------------------------------------------------------------
                #Processing all exposures in visit
                for isub_exp,iexp in enumerate(range(n_in_visit)):

                    #Artificial data 
                    if gen_dic['mock_data']: 
                        
                        #Initialize data at first exposure
                        if isub_exp==0:
                            print('           Building exposures ... ')
                            data_inst[vis]['mock'] = True
                            fixed_args = {}
                            if inst not in mock_dic['intr_prof']:
                                print('           Automatic definition of mock line profile')
                                mock_dic['intr_prof'][inst] = {'mode':'ana','coord_line':'mu','model': 'gauss', 'line_trans':None,'mod_prop':{'ctrst_ord0__IS__VS_' : 0.5,'FWHM_ord0__IS__VS_'  : 5. },'pol_mode' : 'modul'}                                             
                            fixed_args.update(mock_dic['intr_prof'][inst])

                            #Activation of spectral conversion and resampling 
                            cond_conv_st_prof_tab(theo_dic['rv_osamp_line_mod'],fixed_args,data_inst[vis]['type'])
                                                         
                            #Mock data spectral table
                            #    - fixed for all exposures, defined in the star rest frame
                            #    - the model table is defined in the star rest frame, so that models calculated / defined on this table remain centered independently of the systemic and keplerian RV
                            #      models for each exposure are then resampled on the original spectral table if required, and the table is only then shifted / scaled by the systemic and keplerian RV
                            #    - theoretical models are defined in wavelength space
                            #      analytical models are defined in RV space, and their table converted into spectral space afterwards if relevant
                            data_inst[vis]['nspec'] = int(np.ceil((mock_dic['DI_table']['x_end']-mock_dic['DI_table']['x_start'])/mock_dic['DI_table']['dx']))
                            delta_cen_bins = mock_dic['DI_table']['x_end'] - mock_dic['DI_table']['x_start']
                            dcen_bins = delta_cen_bins/data_inst[vis]['nspec'] 
                            fixed_args['cen_bins']=mock_dic['DI_table']['x_start']+dcen_bins*(0.5+np.arange(data_inst[vis]['nspec']))
                            fixed_args['edge_bins']=def_edge_tab(fixed_args['cen_bins'][None,:][None,:])[0,0]
                            fixed_args['ncen_bins']=data_inst[vis]['nspec']
                            fixed_args['dcen_bins']=fixed_args['edge_bins'][1::] - fixed_args['edge_bins'][0:-1]   
                            fixed_args['dim_exp']=[data_inst['nord'],fixed_args['ncen_bins']]                           
                            
                            #Resampled spectral table for model line profile
                            if fixed_args['resamp']:resamp_st_prof_tab(None,None,None,fixed_args,gen_dic,1,theo_dic['rv_osamp_line_mod'])

                            #Effective instrumental convolution
                            fixed_args['FWHM_inst'] = get_FWHM_inst(inst,fixed_args,fixed_args['cen_bins'])
                 
                            #Initialize intrinsic profile properties   
                            #    - removing Peq, as it is always defined in the nominal stellar properties, but used as condition to switch from veq to Peq within var_stellar_prop() when Peq is defined as model parameter instead of veq
                            params_mock = deepcopy(system_param['star']) 
                            params_mock.pop('Peq')
                            if inst not in mock_dic['flux_cont']:mock_dic['flux_cont'][inst]={}
                            if vis not in mock_dic['flux_cont'][inst]:mock_dic['flux_cont'][inst][vis] = 1.
                            params_mock.update({'rv':0.,'cont':mock_dic['flux_cont'][inst][vis]})  
                            params_mock = par_formatting(params_mock,fixed_args['mod_prop'],None,None,fixed_args,inst,vis)
                            params_mock = par_formatting_inst_vis(params_mock,fixed_args,inst,vis,mock_dic['intr_prof'][inst]['mode']) 

                            #Generic properties required for model calculation
                            if inst not in mock_dic['sysvel']:mock_dic['sysvel'][inst]={}
                            if vis not in mock_dic['sysvel'][inst]:mock_dic['sysvel'][inst][vis] = 0.
                            fixed_args.update({ 
                                'mac_mode':theo_dic['mac_mode'],
                                'type':data_inst[vis]['type'],  
                                'nord':data_inst['nord'],
                                'nthreads':mock_dic['nthreads'], 
                                'unthreaded_op':mock_dic['unthreaded_op'], 
                                'resamp_mode' : gen_dic['resamp_mode'], 
                                'conv2intr':False,
                                'inst':inst,
                                'vis':vis, 
                                'fit':False,                                
                                'system_param':system_param,
                                'ar_coord_par':gen_dic['ar_coord_par'],
                                'rout_mode':'Intr_prop',
                                })

                            #Active region properties
                            if (inst in mock_dic['ar_prop']) and (data_dic['DI']['ar_prop'] != {}):
                                params_mock['use_ar']=True
                                ar_prop = data_dic['DI']['ar_prop']
                            else:
                                params_mock['use_ar']=False 
                                ar_prop = {}

                            #Initializing stellar properties
                            fixed_args = var_stellar_prop(fixed_args,theo_dic,data_dic['DI']['system_prop'],ar_prop,system_param['star'],params_mock)
                           
                    #Observational data            
                    else: 

                        #Observational data for current exposure
                        hdulist =fits.open(vis_path_exp[iexp])
                        if vis_path_skysub_exp is not None:hdulist_skysub =fits.open(vis_path_skysub_exp[iexp])
                        hdr =hdulist[0].header                     

                        #VLT UT used by ESPRESSO
                        if (inst=='ESPRESSO') and ((isub_exp==0) or (vis not in data_inst['single_night'])):
                            tel_inst = {
                                'ESO-VLT-U1':'1',
                                'ESO-VLT-U2':'2',
                                'ESO-VLT-U3':'3',
                                'ESO-VLT-U4':'4'
                                }[hdr['TELESCOP']]                                
                       
                        #Initialize data at first exposure
                        if isub_exp==0:                        
                            data_inst[vis]['mock'] = False                      

                            #Instrumental mode
                            if inst in ['ESPRESSO']: 
                                data_prop[inst][vis]['ins_mod'] = hdr['HIERARCH ESO INS MODE']    
                                data_prop[inst][vis]['det_binx'] = hdr['HIERARCH ESO DET BINX']                                
                            elif inst in ['NIRPS_HA','NIRPS_HE']:
                                data_prop[inst][vis]['ins_mod'] = hdr['HIERARCH ESO INS MODE']    
                                data_prop[inst][vis]['det_binx'] = hdr['HIERARCH ESO DET WIN1 BINX']
                                
                            #Radial velocity tables for input CCFs
                            #    - assumed to be common for all CCFs of the visit dataset, and thus calculated only once
                            if (data_inst['type']=='CCF'):
                                if inst in ['HARPN','HARPS','ESPRESSO','ESPRESSO_MR','NIRPS_HA','NIRPS_HE']:
                                    start_rv =  hdr['HIERARCH '+facil_inst+' RV START']    # first velocity
                                    delta_rv = hdr['HIERARCH '+facil_inst+' RV STEP']     # delta vel step
                                    n_vel_vis = (hdulist[1].header)['NAXIS1']                                              
                                else:                        
                                    start_rv =  hdr['CRVAL1']    # first wavelength/velocity
                                    delta_rv = hdr['CDELT1']     # delta lambda/vel step
                                    n_vel_vis = hdr['NAXIS1']
                                    
                                #Velocity table and resolution
                                velccf = start_rv+delta_rv*np.arange(n_vel_vis)

                                #Screening
                                #    - check for oversampling, screening the profiles so that they are defined over uncorrelated points with a resolution similar to the instrumental pixel width.
                                #    - CCFs are screened (ie, keeping one point every specified length) with a length defined as input, or the instrumental bin length 
                                #      the goal is to keep only uncorrelated points, on which we can measure the dispersion as a white noise error 
                                #    - we remove scr_lgth-1 points between two kept points, ie we keep one point in scr_lgth+1
                                #      if the original table is x0,x1,x2,x3,x4.. and scr_lgth=3 we keep x0,x3.. 
                                #    - we define in 'scr_lgth' the number of CCF bins to be removed depending on the chosen length, and the CCF resolution 
                                #    - we assume the same value can be used for a given visit:
                                # + if the bin size is chosen as typical length, then it will not depend on the visits but the CCFs may have been computed with a 
                                # different resolution for each visit. In that case the number of screened bins is calculated once for the visit as:
                                #      scr_lgth = bin size/CCF_step
                                #   with CCF typically oversampled, CCF_step is lower than the bin size and we keep about one CCF point every bin size   
                                # + if the correlation length is chosen, then it may depend on the visit conditions but should be constant for a given visit. In that case 
                                # the number of screened bins has already been set in 'scr_lgth' for the visit, using the empirical estimation on the differential CCFs            
                                if vis not in gen_dic['scr_lgth'][inst]:
                                    gen_dic[inst][vis]['scr_lgth']=int(round(return_pix_size(inst)/delta_rv))
                                    if gen_dic[inst][vis]['scr_lgth']<=1:gen_dic[inst][vis]['scr_lgth']=1  
                                    if gen_dic[inst][vis]['scr_lgth']==1:print('           No CCF screening required')
                                    else:print('           Screening by '+str(gen_dic[inst][vis]['scr_lgth'])+' pixels')
                                if gen_dic[inst][vis]['scr_lgth']>1:
                                    idx_scr_bins=np.arange(n_vel_vis,dtype=int)[gen_dic['ist_scr']::gen_dic[inst][vis]['scr_lgth']]
                                    velccf = velccf[idx_scr_bins]
            
                                #Table dimensions                             
                                data_inst[vis]['nspec'] = len(velccf)
    
                            #----------------------------------------
                            
                            #Number of wavelength bins for spectra 
                            elif ('spec' in data_inst['type']):
                
                                #1D spectra
                                if (data_inst['type']=='spec1D'):
                                    if inst in ['ESPRESSO','ESPRESSO_MR','HARPN']:data_inst[vis]['nspec']  = (hdulist[1].header)['NAXIS2']
                                    else:stop('TBD') 
            
                                #2D spectra
                                elif (data_inst['type']=='spec2D'):
                                    if inst in ['ESPRESSO','ESPRESSO_MR','HARPN','HARPS','CARMENES_VIS','NIGHT','NIRPS_HA','NIRPS_HE']:data_inst[vis]['nspec']  = (hdulist[1].header)['NAXIS1']
                                    elif inst=='EXPRES':data_inst[vis]['nspec'] = 7920
                                    else:stop('ERROR : number of pixels for '+inst+' undefined (set header key)')

                    #Initialize data at first exposure
                    if isub_exp==0: 

                        #Table dimensions
                        data_inst[vis]['dim_all'] = [n_in_visit,data_inst['nord'],data_inst[vis]['nspec']]
                        data_inst[vis]['dim_exp'] = [data_inst['nord'],data_inst[vis]['nspec']]
                        data_inst[vis]['dim_sp'] = [n_in_visit,data_inst['nord']]
                        data_inst[vis]['dim_ord'] = [n_in_visit,data_inst[vis]['nspec']]

                        #Initialize spectral table, flux table, banded covariance matrix     
                        data_dic_temp['cen_bins'] = np.zeros(data_inst[vis]['dim_all'], dtype=float)
                        data_dic_temp['flux'] = np.zeros(data_inst[vis]['dim_all'], dtype=float)*np.nan
                        data_dic_temp['cov'] = np.zeros(data_inst[vis]['dim_sp'], dtype=object)
                        
                        #Initialize blaze-derived and detector noise profiles
                        #    - detector noise is only required for weighing, but is retrieved by default to avoid recomputing the calibration module if it was ran without weighing activated
                        if (vis in data_inst['gcal_blaze_vis']):
                            data_dic_temp['gcal'] = np.zeros(data_inst[vis]['dim_all'], dtype=float)*np.nan
                            data_dic_temp['sdet2'] = np.zeros(data_inst[vis]['dim_all'], dtype=float)*np.nan
                            
                        #Telluric spectrum
                        if ('spec' in data_inst[vis]['type']) and gen_dic['tell_weight'] and (gen_dic['calc_tell_mode']=='input'):
                            data_dic_temp['tell'] = np.ones(data_inst[vis]['dim_all'], dtype=float)

                    ### End of first exposure            
        
                    #------------------------------------------------------------------------------------
                    #Retrieve data for current exposure
                    #------------------------------------------------------------------------------------

                    #-----------------------------------                      
                    #Artificial data  
                    #-----------------------------------
                    if gen_dic['mock_data']: 
                        #print('            building exposure '+str(iexp)+'/'+str(n_in_visit - 1))
                        param_exp = deepcopy(params_mock) 

                        #Table for model calculation
                        args_exp = def_st_prof_tab(None,None,None,deepcopy(fixed_args))

                        #Initializing stellar profiles
                        args_exp = init_custom_DI_prof(args_exp,gen_dic,param_exp)

                        #Initializing broadband scaling of intrinsic profiles into local profiles
                        #    - defined in forward mode at initialization, or defined in fit mode only if the stellar grid is not updated through the fit
                        #    - there are no default pipeline tables for this scaling because they depend on the local spectral tables of the line profiles
                        args_exp['Fsurf_grid_spec'] = theo_intr2loc(args_exp['grid_dic'],args_exp['system_prop'],args_exp,args_exp['ncen_bins'],theo_dic['nsub_star'])                        
                        
                        #Add jitter to the intrinsic profile properties (simulating stellar activity)
                        if (fixed_args['mode']=='ana') and (inst in mock_dic['drift_intr']) and (vis in mock_dic['drift_intr'][vis]) and (len(mock_dic['drift_intr'][inst][vis]>0)):
                            for par_drift in mock_dic['drift_intr'][inst][vis] : 
                                if par_drift in param_exp:
                                    if (par_drift=='rv'):param_exp[par_drift] += mock_dic['drift_intr'][inst][vis][par_drift][iexp]
                                    else:param_exp[par_drift] *= mock_dic['drift_intr'][inst][vis][par_drift][iexp]
                            
                        #Disk-integrated stellar line     
                        base_DI_prof = custom_DI_prof(param_exp,None,args=args_exp)[0]

                        #Deviation from nominal stellar profile 
                        surf_prop_dic, surf_prop_dic_ar,_ = sub_calc_plocc_ar_prop([data_dic['DI']['system_prop']['chrom_mode']],args_exp,['line_prof'],data_dic[inst][vis]['studied_pl'],data_dic[inst][vis]['studied_ar'],deepcopy(system_param),theo_dic,args_exp['system_prop'],param_exp,coord_dic[inst][vis],[iexp], system_ar_prop_in=args_exp['system_ar_prop'])

                        #Adding to base profile the deviations from planet and active region contributions
                        DI_prof_exp = base_DI_prof - surf_prop_dic[data_dic['DI']['system_prop']['chrom_mode']]['line_prof'][:,0] - surf_prop_dic_ar[data_dic['DI']['system_prop']['chrom_mode']]['line_prof'][:,0]

                        #Instrumental response 
                        #    - in RV space for analytical model, in wavelength space for theoretical profiles
                        #    - resolution can be modified to model systematic variations from the instrument or atmosphere
                        #    - disabled if measured profiles as used as proxy for the intrinsic profiles
                        if (fixed_args['mode']!='Intrbin') and (inst in mock_dic['drift_post']) and (vis in mock_dic['drift_post'][vis]) and ('resol' in mock_dic['drift_post'][inst][vis]):
                            fixed_args['FWHM_inst'] = fixed_args['ref_conv']/mock_dic['drift_post'][inst][vis]['resol'][iexp]

                        #Convolution, conversion and resampling 
                        DI_prof_exp = conv_st_prof_tab(None,None,None,fixed_args,args_exp,DI_prof_exp,fixed_args['FWHM_inst'])

                        #Set negative flux values to null
                        DI_prof_exp[DI_prof_exp<0.] = 0.

                        #Define number of photoelectrons extracted during the exposure
                        #   - the model is a density of photoelectrons per unit of time, with continuum set to the input mean flux density
                        if (inst in mock_dic['gcal']):mock_gcal = mock_dic['gcal'][inst]
                        else:mock_gcal = 1.
                        DI_prof_exp_Ftrue = mock_gcal*DI_prof_exp*coord_dic[inst][vis]['t_dur'][iexp] 
                        
                        #Keplerian motion and systemic shift of the disk-integrated profile 
                        #    - we shift profiles from the star rest frame (source) to the solar barycentric rest frame (receiver)
                        #      see gen_specdopshift() :
                        # w_receiver = w_source * (1+ rv[s/r]/c))
                        # w_solbar = w_star * (1+ rv[star/starbar]/c))* (1+ rv[starbar/solbar]/c))
                        #      we include variations in the systemic rv if requested
                        RV_star_stelCDM_mock = calc_rv_star(coord_dic,inst,vis,system_param,gen_dic, coord_dic[inst][vis]['bjd'][iexp],coord_dic[inst][vis]['t_dur'][iexp],mock_dic['sysvel'][inst][vis])[0]
                        sysvel_mock = mock_dic['sysvel'][inst][vis]
                        if (inst in mock_dic['drift_post']) and (vis in mock_dic['drift_post'][vis]) and ('rv' in mock_dic['drift_post'][inst][vis]):sysvel_mock+=mock_dic['drift_post'][inst][vis]['rv'][iexp]
                        if ('spec' in data_inst[vis]['type']):
                            data_dic_temp['cen_bins'][iexp,0]=fixed_args['cen_bins']*gen_specdopshift(RV_star_stelCDM_mock)*gen_specdopshift(sysvel_mock)
                        else:data_dic_temp['cen_bins'][iexp,0] = fixed_args['cen_bins'] + (RV_star_stelCDM_mock+sysvel_mock)        

                        #Defining flux and error table     
                        #    - see weights_bin_prof(), the measured (total, not density) flux can be defined as:
                        # F_meas(t,w) = gcal(band) Nmeas(t,w)
                        #      where Nmeas(t,w), drawn from a Poisson distribution with number of events Ntrue(t,w), is the number of photo-electrons measured durint texp
                        #      the estimate of the error is
                        # EF_meas(t,w)^2 = gcal(band) F_meas(t,w) + Ern^2 ~ gcal(band) F_meas(t,w) 
                        #      which is a biased estimate of the true error but corresponds to what is returned by DRS
                        #      we neglect read noise, also because only S2D tables are available in ANTARESS
                        #    - the S/N of the mock profiles is F_meas(t,w)/EF_meas(t,w) = sqrt(F_meas(t,w)/gcal(band)) = sqrt(Nmeas(t,w)) proportional to sqrt(texp*cont)
                        #      errors in the continuum of the normalized profiles are proportional to 1/sqrt(texp*cont) 
                        #      beware that this error does not necessarily match the flux dispersion
                        if (inst in mock_dic['set_err']) and mock_dic['set_err'][inst]:
                            DI_prof_exp_Fmeas = np.array(list(map(np.random.poisson, DI_prof_exp_Ftrue,  data_inst[vis]['nspec']*[1]))).flatten()
                        else:
                            DI_prof_exp_Fmeas = DI_prof_exp_Ftrue
                        DI_err_exp_Emeas2 = mock_gcal*DI_prof_exp_Fmeas
                        data_dic_temp['flux'][iexp,0] = DI_prof_exp_Fmeas                      
                        data_dic_temp['cov'][iexp,0] = DI_err_exp_Emeas2[None,:]                        
                        if mock_dic['verbose_flux_cont']:print('         Exposure %s: Average SNR = %.0f, Average photon count = %.0f' % (str(isub_exp), np.mean(DI_prof_exp_Fmeas/np.sqrt(DI_err_exp_Emeas2)), np.mean(DI_prof_exp_Fmeas)))       

                    #-----------------------------------
                    #Observational data
                    #-----------------------------------
                    else: 
                        cond_bad_exp = np.zeros(data_inst[vis]['dim_exp'],dtype=bool)
                    
                        #Retrieve BERV
                        #    - in km/s
                        if inst=='EXPRES':
                            data_prop[inst][vis]['BERV'][iexp] = hdulist[2].header['HIERARCH wtd_mdpt_bc']*c_light
                        else:
                            if inst=='CARMENES_VIS': reduc_txt = 'CARACAL' 
                            elif inst in ['HARPN','HARPS','ESPRESSO','ESPRESSO_MR','NIGHT','NIRPS_HA','NIRPS_HE']:reduc_txt = facil_inst+' QC'
                            else:stop('ERROR : define BERV retrieval')
                            data_prop[inst][vis]['BERV'][iexp] = hdr['HIERARCH '+reduc_txt+' BERV']

                        #Retrieve CCFs
                        #    - HARPS 'hdu' has 73 x n_velocity size
                        # + 72 CCFs for each of the spectrograph order
                        # + last element is the total CCF over all orders (n_velocity size) 
                        #    - HARPN 'hdu' has 70 x n_velocity size  
                        # + 69 CCFs for each of the spectrograph order
                        # + last element is the total CCF over all orders (n_velocity size)   
                        #    - SOPHIE 'hdu' has 40 x n_velocity size
                        # + 39 CCFs for each of the spectrograph order
                        # + last element is the total CCF over all orders (n_velocity size)                           
                        if data_inst['type']=='CCF': 
                      
                            #Bin centers
                            #    - we associate the velocity table to all exposures to keep the same structure as spectra
                            data_dic_temp['cen_bins'][iexp,0] = velccf
      
                            #CCF tables per order 
                            if (vis_path_skysub_exp is not None) and (gen_dic['fibB_corr'][inst][vis]=='all'):hdulist_dat= hdulist_skysub
                            else:hdulist_dat = hdulist
                            if inst in ['HARPS','ESPRESSO','ESPRESSO_MR','HARPN','NIRPS_HA','NIRPS_HE']:
                                all_ccf  = hdulist_dat[1].data  
                                all_eccf = hdulist_dat[2].data 
                                if (vis_path_skysub_exp is not None) and (gen_dic['fibB_corr'][inst][vis]!='all'):
                                    all_ccf[idx_ord_skysub] = (hdulist_skysub[1].data)[idx_ord_skysub]
                                    all_eccf[idx_ord_skysub] = (hdulist_skysub[2].data)[idx_ord_skysub]
                            else:                      
                                all_ccf =hdulist_dat[0].data
                                if (vis_path_skysub_exp is not None) and (gen_dic['fibB_corr'][inst][vis]!='all'):
                                    all_ccf[idx_ord_skysub] = (hdulist_skysub[0].data)[idx_ord_skysub]           
                                
                            #Construction of the total CCF
                            #    - we re-calculate the CCF rather than taking the total CCF stored in the last column of the matrix to be able to account for sky-correction in specific orders
                            #    - we use the input "gen_dic['orders4ccf'][inst]" rather than "gen_dic[inst]['orders4ccf']", which is used for CCF computed from spectral data 
                            if inst not in gen_dic['orders4ccf']:flux_raw = np.nansum(all_ccf[ range(return_spec_nord(inst))],axis=0)     
                            else:flux_raw = np.nansum(all_ccf[gen_dic['orders4ccf'][inst]],axis=0)             

                            #Screening CCF
                            if gen_dic[inst][vis]['scr_lgth']>1:flux_raw = flux_raw[idx_scr_bins]
                            data_dic_temp['flux'][iexp,0] = flux_raw
                            
                            #Error table
                            if gen_dic[inst][vis]['flag_err']:
                                if inst not in gen_dic['orders4ccf']:err_raw = np.sqrt(np.nansum(all_eccf[range(return_spec_nord(inst))]**2.,axis=0))      
                                else:err_raw = np.sqrt(np.nansum(all_eccf[gen_dic['orders4ccf'][inst]]**2.,axis=0))                                                          
                                if gen_dic[inst][vis]['scr_lgth']>1:err_raw = err_raw[idx_scr_bins]
                                err_raw = np.tile(err_raw,[data_inst['nord'],1])

                        #-------------------------------------------------------------------------------------------------------------------------------------------
                                
                        #Retrieve spectral data
                        elif ('spec' in data_inst['type']):
                                              
                            #1D spectra
                            if data_inst['type']=='spec1D':
                                if inst in ['HARPS','ESPRESSO','ESPRESSO_MR','HARPN','NIRPS_HA','NIRPS_HE']:
                                    
                                    #Replacing single order with sky-corrected data
                                    if (vis_path_skysub_exp is not None):data_loc = hdulist_skysub[1].data         
                                    else:data_loc = hdulist[1].data                                    

                                    #Bin centers
                                    if gen_dic['sp_frame']=='air':data_dic_temp['cen_bins'][iexp] = data_loc['wavelength_air'] 
                                    elif gen_dic['sp_frame']=='vacuum':data_dic_temp['cen_bins'][iexp] = data_loc['wavelength']                             
                                    
                                    #Spectra
                                    #    - flux and errors are per unit of wavelength but not per unit of time, ie that they correspond to the number of photoelectrons measured during the exposure
                                    data_dic_temp['flux'][iexp] = data_loc['flux']
                                    err_raw = data_loc['error']

                                    #Pixels quality
                                    qualdata_exp = data_loc['quality']    

                                else:
                                    stop('ERROR : spectra upload undefined for '+inst) 
        
                            #------------------------------
            
                            #2D spectra
                            elif data_inst['type']=='spec2D':
                                    
                                #Sky-corrected data
                                if (vis_path_skysub_exp is not None):
                                    if (gen_dic['fibB_corr'][inst][vis]=='all'):hdulist_dat= hdulist_skysub
                                    else:
                                        cond_skysub = np.repeat(False,return_spec_nord(inst))     #Conditions on original order indexes 
                                        cond_skysub[idx_ord_skysub] = True  
                                        idxsub_ord_skysub = np_where1D(cond_skysub[idx_ord_kept])          #Indexes in reduced tables 
                                else:hdulist_dat = hdulist
                                    
                                if inst in ['HARPS','ESPRESSO','ESPRESSO_MR','HARPN','NIGHT','NIRPS_HA','NIRPS_HE']:
                           
                                    #Bin centers
                                    #    - dimension norder x nbins
                                    if gen_dic['sp_frame']=='air':ikey_wav = 5
                                    elif gen_dic['sp_frame']=='vacuum':ikey_wav = 4
                                    data_dic_temp['cen_bins'][iexp] = (hdulist_dat[ikey_wav].data)[idx_ord_kept]
                                    dll = (hdulist_dat[ikey_wav+2].data)[idx_ord_kept]
                                    if (vis_path_skysub_exp is not None) and (gen_dic['fibB_corr'][inst][vis]!='all'):
                                        data_dic_temp['cen_bins'][iexp][idxsub_ord_skysub] = (hdulist_skysub[ikey_wav].data)[idx_ord_kept][idxsub_ord_skysub]                                 
                                        dll[idxsub_ord_skysub] = (hdulist_skysub[ikey_wav+2].data)[idx_ord_kept][idxsub_ord_skysub]   

                                    #Spectra   
                                    #    - the overall flux level is changed in the sky-corrected data, messing up with the flux balance corrections if only some orders were corrected
                                    #      we thus roughly rescale the flux in the corrected orders to their original level (assuming constant resolution over the order to calculate the mean flux)   
                                    #    - the flux is per unit of pixel, and must be divided by the pixel spectral size stored in dll to be passed in units of wavelength
                                    #    - the flux is not per unit of time, but corresponds to the number of photoelectrons measured during the exposure corrected for the blaze function
                                    data_dic_temp['flux'][iexp] = (hdulist_dat[1].data)[idx_ord_kept]/dll
                                    err_raw = (hdulist_dat[2].data)[idx_ord_kept]/dll  

                                    #Replacing with sky-corrected data
                                    if (vis_path_skysub_exp is not None) and (gen_dic['fibB_corr'][inst][vis]!='all'):
                                        dll_ord_skysub = dll[idxsub_ord_skysub]
                                        mean_Fraw_ord_skysub = np.nansum(data_dic_temp['flux'][iexp][idxsub_ord_skysub]*dll_ord_skysub,axis=1)/np.nansum(dll_ord_skysub,axis=1)
                                        data_dic_temp['flux'][iexp][idxsub_ord_skysub] = (hdulist_skysub[1].data)[idx_ord_kept][idxsub_ord_skysub]/dll_ord_skysub
                                        mean_Fcorr_ord_skysub = np.nansum(data_dic_temp['flux'][iexp][idxsub_ord_skysub]*dll_ord_skysub,axis=1)/np.nansum(dll_ord_skysub,axis=1)
                                        skysub_corrfact = (mean_Fraw_ord_skysub/mean_Fcorr_ord_skysub)[:,None]
                                        data_dic_temp['flux'][iexp][idxsub_ord_skysub]*=skysub_corrfact
                                        err_raw[idxsub_ord_skysub] = ((hdulist_skysub[2].data)[idx_ord_kept][idxsub_ord_skysub]/dll_ord_skysub)*skysub_corrfact

                                    #Pixels quality
                                    #    - unextracted pixels (on each side of the shorter blue orders and in between the blue/red chips in ESPRESSO) are considered as undefined
                                    qualdata_exp = (hdulist_dat[3].data)[idx_ord_kept]
                                    if (vis_path_skysub_exp is not None) and (gen_dic['fibB_corr'][inst][vis]!='all'):
                                        qualdata_exp[idxsub_ord_skysub] = (hdulist_skysub[3].data)[idx_ord_kept][idxsub_ord_skysub]
                                    cond_bad_exp |= (qualdata_exp == 16384)

                                    #Removing pixels at the edge of the red detector
                                    #    - usually too poorly defined even for bright stars
                                    if inst=='ESPRESSO':
                                        for iord_bad in [90,91]: 
                                            isub_ord = np_where1D(np.array(idx_ord_kept)==iord_bad)
                                            if len(isub_ord)>0:
                                                data_dic_temp['flux'][iexp][isub_ord,0:30] = np.nan
                                                qualdata_exp[isub_ord,0:30] = 1                                   

                                    #Calibration and detector noise profiles
                                    #    - see weights_bin_prof() for details       
                                    #    - non sky-corrected data must be used
                                    if (vis in data_inst['gcal_blaze_vis']):
                                        
                                        #Retrieving counts per pixels
                                        #    - blazed and unblazed pixels are undefined and null in the same pixels
                                        hdulist_dat_blaze =fits.open(vis_path_blaze_exp[iexp])
                                        count_exp = (hdulist[1].data)[idx_ord_kept]
                                        count_blaze_exp = (hdulist_dat_blaze[1].data)[idx_ord_kept]
                                        err_count_blaze_exp = (hdulist_dat_blaze[2].data)[idx_ord_kept]
                                        cond_def = (~np.isnan(count_exp)) & (~cond_bad_exp)
                                        cond_gcal = cond_def & (count_exp!=0.)
                                        cond_pos_ct = count_blaze_exp>0.
                                        cond_def_pos = cond_def & cond_pos_ct
                                        cond_def_neg = cond_def & ~cond_pos_ct

                                        #Processing each order
                                        for iord in range(data_inst['nord']):
    
                                            #Defining calibration profile
                                            #    - gcal(w,t,v) = 1/(dw(w)*bl(w,t,v)))
                                            # with bl(w,t,v) = N_meas[bl](w,t,v)/N_meas(w,t,v) 
                                            #    - negative values are set to undefined, so that they are replaced by the calibration model in the calc_gcal() routine
                                            idx_gcal_ord = np_where1D(cond_gcal[iord])
                                            gcal_temp = 1./(dll[iord,idx_gcal_ord]*count_blaze_exp[iord,idx_gcal_ord]/count_exp[iord,idx_gcal_ord])
                                            cond_gcal_pos = gcal_temp > 0.
                                            data_dic_temp['gcal'][iexp,iord,idx_gcal_ord[cond_gcal_pos]] = gcal_temp[cond_gcal_pos] 
                                            
                                            #Defining detector noise
                                            #    - Edet_meas(w,t,v)^2 = EN_meas[bl](w,t,v)^2 - N_meas[bl](w,t,v)
                                            #    - set to 0 where negative or undefined within definition range, after duplicating the latest defined value at the edges
                                            #    - at pixels where counts are negative the DRS consider them null to define error tables, so that the detector noise is then retrieved as
                                            #      Edet_meas(w,t,v)^2 = EN_meas[bl](w,t,v)^2
                                            #    - some pixels have large errors despite not having large fluxes; this may come from hot pixels subtracted from the extracted counts
                                            idx_def_pos = np_where1D(cond_def_pos[iord])
                                            data_dic_temp['sdet2'][iexp,iord,idx_def_pos] = err_count_blaze_exp[iord,idx_def_pos]**2. - count_blaze_exp[iord,idx_def_pos]
                                            if idx_def_pos[0]>0:data_dic_temp['sdet2'][iexp,iord,0:idx_def_pos[0]] = data_dic_temp['sdet2'][iexp,iord,idx_def_pos[0]]              
                                            if idx_def_pos[-1]<data_inst[vis]['nspec']-1:data_dic_temp['sdet2'][iexp,iord,idx_def_pos[-1]+1::] = data_dic_temp['sdet2'][iexp,iord,idx_def_pos[-1]]                      
                                            data_dic_temp['sdet2'][iexp,iord][np.isnan(data_dic_temp['sdet2'][iexp,iord])] = 0.
                                            data_dic_temp['sdet2'][iexp,iord][data_dic_temp['sdet2'][iexp,iord]<0.] = 0.
                                            data_dic_temp['sdet2'][iexp,iord,cond_def_neg[iord]] = err_count_blaze_exp[iord,cond_def_neg[iord]]**2. 

                                elif inst=='CARMENES_VIS':             
                                    if (vis_path_skysub_exp is not None):stop('No sky-corrected data available')

                                    #Bin centers
                                    #    - dimension norder x nbins
                                    #    - in vacuum
                                    data_dic_temp['cen_bins'][iexp] = (hdulist_dat['WAVE'].data)[idx_ord_kept]
                                    if gen_dic['sp_frame']=='air':data_dic_temp['cen_bins'][iexp]/=air_index(data_dic_temp['cen_bins'][iexp], t=15., p=760.)

                                    #Spectra   
                                    data_dic_temp['flux'][iexp] = (hdulist_dat['SPEC'].data)[idx_ord_kept]
                                    err_raw = (hdulist_dat['SIG'].data)[idx_ord_kept]                            

                                    #BERV correction to shift exposures from the Earth laboratory to the solar System barycenter
                                    #    - see gen_specdopshift():
                                    # w_receiver = w_source * (1+ rv[source/receiver]/c))     
                                    #      we correct for the BERV, the radial velocity of the Earth with respect to the solar System barycenter (source is the Earth, where signals are measured, and the receiver the barycenter)
                                    # w_solbar = w_Earth * (1+ rv[Earth/sun bary]/c))                                      
                                    # w_solbar = w_Earth * (1+ (BERV/c))                                       
                                    #      we include a general+special relativity correction for the eccentricity and change of potential of Earth orbit around the Sun, averaged over a revolution (from C. Lovis)                                  
                                    data_dic_temp['cen_bins'][iexp]*=gen_specdopshift(data_prop[inst][vis]['BERV'][iexp])*(1.+1.55e-8) 

                                elif inst=='EXPRES':              
                                    if (vis_path_skysub_exp is not None):stop('No sky-corrected data available')
                                    
                                    #Bin centers
                                    #    - dimension norder x nbins
                                    #    - we use the chromatic-barycentric-corrected Excalibur wavelengths 
                                    #      Excalibur is a non-parametric, hierarchical method for wavelength calibration. These wavelengths suffer less from systematic or instrumental errors, but cover a smaller spectral range
                                    #      at undefined Excalibur wavelengths we nonetheless set the regular barycentric-corrected wavelength, as ANTARESS requires fully defined spectral tables
                                    #      since flux values are set to nan at undefined Excalibur wavelengths this has no impact on the results
                                    data_dic_temp['cen_bins'][iexp] = ((hdulist_dat[1].data)['bary_wavelength'])[idx_ord_kept]     
                                    bary_excalibur = ((hdulist_dat[1].data)['bary_excalibur'])[idx_ord_kept]
                                    cond_def_wav = (hdulist_dat[1].data)['excalibur_mask'][idx_ord_kept]
                                    data_dic_temp['cen_bins'][iexp][cond_def_wav] = bary_excalibur[cond_def_wav]                                                                        
                                    if gen_dic['sp_frame']=='air':data_dic_temp['cen_bins'][iexp]/=air_index(data_dic_temp['cen_bins'][iexp], t=15., p=760.)

                                    #Spectra 
                                    #    - the flux is not per unit of time, but corresponds to the number of photoelectrons measured during the exposure corrected for the blaze function   
                                    data_dic_temp['flux'][iexp] = (hdulist_dat[1].data)['spectrum'][idx_ord_kept]
                                    err_raw = (hdulist_dat[1].data)['uncertainty'][idx_ord_kept]

                                    #Set flux to nan at undefined Excalibur wavelengths
                                    #    - for consistency with the way pixels are usually defined
                                    #    - excalibur_mask: gives a mask that excludes pixels without Excalibur wavelengths, which have been set to NaNs. These are pixels that do not fall between calibration lines in the order, and so are only on the edges of each order, or outside the range of LFC lines, as described above.
                                    data_dic_temp['flux'][iexp][~cond_def_wav]=np.nan

                                    #Pixels quality
                                    #    - bad pixels are set to False originally
                                    #    - pixel_mask: pixels with too low signal to return a proper extracted value. This encompasses largely pixels on the edges of orders
                                    qualdata_exp = np.zeros(data_dic_temp['flux'][iexp].shape)
                                    qualdata_exp[~(hdulist_dat[1].data)['pixel_mask'][idx_ord_kept]] = 1.
                                    
                                    #Telluric spectrum
                                    if ('tell' in data_dic_temp):
                                        data_dic_temp['tell'][iexp] = (hdulist_dat[1].data)['tellurics'][idx_ord_kept]  

                                else:
                                    stop('ERROR : spectra upload undefined for '+inst) 
                        
                        #Set bad quality pixels to nan
                        #    - bad pixels have flag > 0
                        #    - unless requested by the user those pixels are not set to undefined, because we cannot propagate detailed quality information when a profile is resampled / modified and leaving them undefined 
                        # will 'bleed' to adjacent pixels when resampling and will limit some calculations
                        if gen_dic['bad2nan']:cond_bad_exp |= (qualdata_exp > 0.) 
                        if True in cond_bad_exp:data_dic_temp['flux'][iexp][cond_bad_exp] = np.nan

                        #Set undefined errors to square-root of flux as first-order approximation
                        #    - errors will be redefined for local CCF using their continuum dispersion
                        #    - pixels with negative values are set to undefined, as there is no simple way to attribute errors to them
                        if not gen_dic[inst][vis]['flag_err']:
                            cond_pos_exp = data_dic_temp['flux'][iexp]>=0.
                            data_dic_temp['flux'][iexp][~cond_pos_exp] = np.nan
                            err_raw = np.zeros(data_inst[vis]['dim_exp'],dtype=float)
                            err_raw[cond_pos_exp]=np.sqrt(data_dic_temp['flux'][iexp][cond_pos_exp]) 


                        #------------------------------

                        #Processing orders
                        for iord in range(data_inst['nord']):
                            
                            #Set pixels with null flux on order sides to undefined    
                            #    - those pixels contain no information and can be removed to speed calculations
                            #    - we do not set to undefined all null pixels in case some are within the spectra
                            if ('spec' in data_inst['type']):
                                cumsum = np.nancumsum(data_dic_temp['flux'][iexp][iord])
                                if (True in (cumsum>0.)):
                                    idx_null_left = np_where1D(cumsum>0.)[0]-1
                                    if (idx_null_left>=0) and (np.nansum(data_dic_temp['flux'][iexp][iord][0:idx_null_left+1])==0.):data_dic_temp['flux'][iexp][iord][0:idx_null_left+1] = np.nan
                                if (True in (cumsum<cumsum[-1])):                                
                                    idx_null_right = np_where1D(cumsum<cumsum[-1])[-1]+2                      
                                    if (idx_null_right<data_inst[vis]['nspec']) and (np.nansum(data_dic_temp['flux'][iexp][iord][idx_null_right::])==0.):data_dic_temp['flux'][iexp][iord][idx_null_right::] = np.nan

                            #Banded covariance matrix
                            #    - dimensions (nd+1,nsp) for a given (exposure,order)
                            #    - here we have to assume there are no correlations in the uploaded profile, and store its error table in the matrix diagonal
                            data_dic_temp['cov'][iexp,iord] = (err_raw[iord]**2.)[None,:]

                    #-----------------------------------------------------
                    #Observational properties associated with each exposure
                    #-----------------------------------------------------
                    if not gen_dic['mock_data']: 
                        
                        #Airmass (mean between start and end of exposure)
                        #    - AM = sec(z) = 1/cos(z) with z the zenith angle
                        if inst in ['HARPN','CARMENES_VIS']:data_prop[inst][vis]['AM'][iexp]=hdr['AIRMASS']
                        elif inst=='CORALIE':data_prop[inst][vis]['AM'][iexp]=hdr['HIERARCH ESO OBS TARG AIRMASS']
                        elif inst=='EXPRES':data_prop[inst][vis]['AM'][iexp]=hdr['AIRMASS']
                        else:
                            if inst in ['HARPS','SOPHIE','ESPRESSO','NIRPS_HA','NIRPS_HE']:
                                if inst in ['SOPHIE','HARPS','NIRPS_HA','NIRPS_HE']:pre_txt=''                             
                                elif inst=='ESPRESSO':pre_txt = tel_inst
                                data_prop[inst][vis]['AM'][iexp]=0.5*(hdr['HIERARCH '+facil_inst+' TEL'+pre_txt+' AIRM START']+hdr['HIERARCH '+facil_inst+' TEL'+pre_txt+' AIRM START'] )                                                                
                            elif inst=='ESPRESSO_MR':
                                data_prop[inst][vis]['AM_UT'][iexp]=[]
                                for tel_inst_loc in ['TEL1','TEL2','TEL3','TEL4']:
                                    pre_txt='HIERARCH ESO '+tel_inst_loc+' AIRM'
                                    data_prop[inst][vis]['AM_UT'][iexp]+=[0.5*(hdr[pre_txt+' START']+hdr[pre_txt+' END'] )]    
                                data_prop[inst][vis]['AM'][iexp]=np.mean(data_prop[inst][vis]['AM_UT'][iexp])
                                
                        #Integrated water vapor at zenith (mm)
                        if inst=='ESPRESSO':
                            data_prop[inst][vis]['IWV_AM'][iexp]=0.5*(hdr['HIERARCH '+facil_inst+' TEL'+tel_inst+' AMBI IWV START']+hdr['HIERARCH '+facil_inst+' TEL'+tel_inst+' AMBI IWV START'] )                                                             
                        #No radiometer available at La Silla, TNG and CAHA
                        elif inst in ['CARMENES_VIS','HARPN','HARPS','SOPHIE','NIRPS_HA','NIRPS_HE']:
                            data_prop[inst][vis]['IWV_AM'][iexp] = 1.
                            
                        #Temperature (K)
                        if inst in ['HARPS','SOPHIE','ESPRESSO','NIRPS_HA','NIRPS_HE']:
                            if inst in ['SOPHIE','HARPS','NIRPS_HA','NIRPS_HE']:pre_txt=''                             
                            elif inst=='ESPRESSO':pre_txt = tel_inst
                            data_prop[inst][vis]['TEMP'][iexp] = hdr['HIERARCH '+facil_inst+' TEL'+pre_txt+' AMBI TEMP'] + 273.15
                        elif inst=='HARPN':   #Validated by Rosario Cosentino (HARPSN instrument scientist)
                            pre_txt=''
                            data_prop[inst][vis]['TEMP'][iexp] = hdr['HIERARCH '+facil_inst+' METEO TEMP10M'] + 273.15
                        elif inst=='CARMENES_VIS':
                            pre_txt=''
                            data_prop[inst][vis]['TEMP'][iexp] = hdr['HIERARCH '+facil_inst+' GEN AMBI TEMPERATURE'] + 273.15
                            
                        #Pressure (atm)
                        #    - pressure is in hPa and must be converted into atm unit (1 atm = 101325 Pa = 1013.25 hPa)
                        if inst in ['HARPS','SOPHIE','ESPRESSO','NIRPS_HA','NIRPS_HE']:
                            if inst in ['SOPHIE','HARPS','NIRPS_HA','NIRPS_HE']:pre_txt=''                             
                            elif inst=='ESPRESSO':pre_txt = tel_inst
                            data_prop[inst][vis]['PRESS'][iexp]=0.5*(hdr['HIERARCH '+facil_inst+' TEL'+pre_txt+' AMBI PRES START']+hdr['HIERARCH '+facil_inst+' TEL'+pre_txt+' AMBI PRES START'] )/1013.2501                                                             
                        elif inst=='HARPN':    #Validated by Rosario Cosentino (HARPSN instrument scientist)
                            pre_txt=''
                            data_prop[inst][vis]['PRESS'][iexp] = hdr['HIERARCH '+facil_inst+' METEO PRESSURE'] / 1013.2501
                        elif inst=='CARMENES_VIS':
                            pre_txt=''
                            data_prop[inst][vis]['PRESS'][iexp] = hdr['HIERARCH '+facil_inst+' GEN AMBI PRESSURE'] / 1013.2501
                            
                        #Seeing (mean between start and end of exposure)
                        #    - not saved for HARPSN data
                        if inst=='CARMENES_VIS':
                            data_prop[inst][vis]['seeing'][iexp]=hdr['HIERARCH CAHA GEN AMBI SEEING']  
                        elif inst=='HARPS':
                            pre_txt='HIERARCH ESO TEL AMBI FWHM' 
                            if (pre_txt+' START' in hdr):data_prop[inst][vis]['seeing'][iexp]=0.5*(hdr[pre_txt+' START']+hdr[pre_txt+' END'] )  
                        elif inst=='SOPHIE':
                            data_prop[inst][vis]['seeing'][iexp]=hdr['HIERARCH OHP GUID SEEING']  
                        elif inst=='ESPRESSO_MR':
                            data_prop[inst][vis]['seeing_UT'][iexp]=[]
                            for tel_inst_loc in ['TEL1','TEL2','TEL3','TEL4']:
                                data_prop[inst][vis]['seeing_UT'][iexp]+=[hdr['HIERARCH ESO '+tel_inst_loc+' IA FWHM']]  
                            data_prop[inst][vis]['seeing'][iexp]=np.mean(data_prop[inst][vis]['seeing_UT'][iexp])                            
                        elif inst=='ESPRESSO':
                            data_prop[inst][vis]['seeing'][iexp]=hdr['HIERARCH ESO TEL'+tel_inst+' IA FWHM']      #seeing corrected for airmass                 
                        elif inst in ['NIRPS_HA','NIRPS_HE']:
                            data_prop[inst][vis]['seeing'][iexp]=hdr['HIERARCH ESO INS2 AOS ATM SEEING']        
            
                        #Shape of PSF
                        #      first and second values: size in X and Y
                        #      third value : average (quadratic) size
                        #      fourth value : angle      
                        if inst in ['CORALIE','SOPHIE','HARPS','ESPRESSO','ESPRESSO_MR','HARPN']:
                            if inst=='HARPN':
                                pre_txt='HIERARCH '+facil_inst+' AG ACQ'                    
                                data_prop[inst][vis]['PSF_prop'][iexp,0]=hdr[pre_txt+' FWHMX']
                                data_prop[inst][vis]['PSF_prop'][iexp,1]=hdr[pre_txt+' FWHMY']
                            elif inst in ['ESPRESSO','ESPRESSO_MR']:
                                pre_txt='HIERARCH '+facil_inst+' OCS ACQ3'                    
                                if pre_txt+' FWHMX ARCSEC' in hdr:
                                    data_prop[inst][vis]['PSF_prop'][iexp,0]=hdr[pre_txt+' FWHMX ARCSEC']
                                    data_prop[inst][vis]['PSF_prop'][iexp,1]=hdr[pre_txt+' FWHMY ARCSEC']                
                                    data_prop[inst][vis]['PSF_prop'][iexp,2]=np.sqrt(data_prop[inst][vis]['PSF_prop'][iexp,0]**2.+data_prop[inst][vis]['PSF_prop'][iexp,1]**2.)
                            if data_prop[inst][vis]['PSF_prop'][iexp,0]==0.:
                                data_prop[inst][vis]['PSF_prop'][iexp,3]=np.nan
                            else:
                                data_prop[inst][vis]['PSF_prop'][iexp,3]=np.arctan2(data_prop[inst][vis]['PSF_prop'][iexp,1],data_prop[inst][vis]['PSF_prop'][iexp,0])*180./np.pi
            
                        #Coefficient of color correction for Earth atmosphere
                        #    - a polynomial is fitted to the ratio between the spectrum averaged over each order, and a stellar template
                        #    - the coefficient below give the minimum and maximum values of this ratio (they may thus correspond to any order)
                        #      a low value corresponds to a strong flux decrease
                        if inst in ['CORALIE','SOPHIE','HARPS','ESPRESSO','ESPRESSO_MR','HARPN']:
                            if inst in ['CORALIE']:reduc_txt='ESO DRS '      
                            elif inst=='SOPHIE':reduc_txt='OHP DRS '     
                            elif inst in ['HARPS','ESPRESSO','ESPRESSO_MR']:reduc_txt='ESO QC '
                            elif inst=='HARPN':reduc_txt='ESO QC ' if ('HIERARCH ESO QC FLUX CORR') in hdr else 'TNG QC '
                            data_prop[inst][vis]['colcorrmin'][iexp]=hdr['HIERARCH '+reduc_txt+'FLUX CORR MIN'] 
                            data_prop[inst][vis]['colcorrmax'][iexp]=hdr['HIERARCH '+reduc_txt+'FLUX CORR MAX'] 
        
                            #Polynomial for color correction   
                            #    - retrieved for each order
                            if inst in ['HARPS','ESPRESSO','HARPN']:
                                for iord in range(data_inst['nord']):
                                    data_prop[inst][vis]['colcorr_ord'][iexp,iord]=hdr['HIERARCH '+reduc_txt+'ORDER'+str(iord+1)+' FLUX CORR'] 
                                
                        #Saturation check
                        if inst in ['HARPS','ESPRESSO','HARPN']:
                            data_prop[inst][vis]['satur_check'][iexp]=hdr['HIERARCH '+reduc_txt+'SATURATION CHECK']

                        #ADC and PIEZO properties
                        #    - properties are sometimes not correctly saved in headers, and are not critical for this sequence
                        if (inst in ['ESPRESSO']) and (gen_dic['sequence'] not in ['st_master_tseries']):
                            ref_adc = {'ESO-VLT-U1':'','ESO-VLT-U2':'2','ESO-VLT-U3':'3','ESO-VLT-U4':'4'}[hdr['TELESCOP']]
                            data_prop[inst][vis]['adc_prop'][iexp,0]=hdr['HIERARCH ESO INS'+ref_adc+' ADC1 POSANG']
                            data_prop[inst][vis]['adc_prop'][iexp,3]=hdr['HIERARCH ESO INS'+ref_adc+' ADC2 POSANG']
                            
                            #Conversion from hhmmss.ssss to deg
                            data_prop[inst][vis]['adc_prop'][iexp,1]=hdr['HIERARCH ESO INS'+ref_adc+' ADC1 RA']
                            data_prop[inst][vis]['adc_prop'][iexp,2]=hdr['HIERARCH ESO INS'+ref_adc+' ADC1 DEC']
                            data_prop[inst][vis]['adc_prop'][iexp,4]=hdr['HIERARCH ESO INS'+ref_adc+' ADC2 RA']
                            data_prop[inst][vis]['adc_prop'][iexp,5]=hdr['HIERARCH ESO INS'+ref_adc+' ADC2 DEC']
                            for ikey in [1,2,4,5]:
                                str_split = str(data_prop[inst][vis]['adc_prop'][iexp,ikey]).split('.')
                                h_key = float(str_split[0][-6:-4])
                                min_key = float(str_split[0][-4:-2])
                                sec_key = float(str_split[0][-2:]+'.'+str_split[1])
                                data_prop[inst][vis]['adc_prop'][iexp,ikey] = 15.*h_key+15.*min_key/60.+15.*sec_key/3600.

                            #PIEZO properties
                            data_prop[inst][vis]['piezo_prop'][iexp,0]=hdr['HIERARCH ESO INS'+ref_adc+' TILT1 VAL1']
                            data_prop[inst][vis]['piezo_prop'][iexp,1]=hdr['HIERARCH ESO INS'+ref_adc+' TILT1 VAL2']
                            data_prop[inst][vis]['piezo_prop'][iexp,2]=hdr['HIERARCH ESO INS'+ref_adc+' TILT2 VAL1']
                            data_prop[inst][vis]['piezo_prop'][iexp,3]=hdr['HIERARCH ESO INS'+ref_adc+' TILT2 VAL2']   

                        #Guide star coordinates
                        if (inst in ['ESPRESSO']) and (data_inst['type']=='spec2D') :
                            data_prop[inst][vis]['guid_coord'][iexp,0]=hdr['HIERARCH ESO ADA'+tel_inst+' GUID RA']
                            data_prop[inst][vis]['guid_coord'][iexp,1]=hdr['HIERARCH ESO ADA'+tel_inst+' GUID DEC']

                        #Properties derived by the pipeline
                        if ('rv_pip' in DI_data_inst[vis]):
                            if inst in ['CORALIE']:
                                pre_txt='HIERARCH ESO DRS CCF'      
                                DI_data_inst[vis]['rv_pip'][iexp]=hdr[pre_txt+' RVC'] #Baryc RV (drift corrected) (km/s)
                                DI_data_inst[vis]['erv_pip'][iexp]=hdr['HIERARCH ESO DRS DVRMS']/1e3                           
                            elif inst=='SOPHIE':
                                pre_txt='HIERARCH OHP DRS CCF'               
                                DI_data_inst[vis]['rv_pip'][iexp]=hdr[pre_txt+' RV']  #Baryc RV (drift corrected ?? ) (km/s) 
                                DI_data_inst[vis]['erv_pip'][iexp]=hdr[pre_txt+' ERR']
                            elif inst in ['HARPS','ESPRESSO','ESPRESSO_MR','HARPN','NIRPS_HA','NIRPS_HE']:
                                reduc_txt=facil_inst+' QC '
                                pre_txt='HIERARCH '+reduc_txt+'CCF'        
                                DI_data_inst[vis]['rv_pip'][iexp]=hdr[pre_txt+' RV'] #Baryc RV (drift corrected) (km/s)   
                                DI_data_inst[vis]['erv_pip'][iexp]=hdr[pre_txt+' RV ERROR']
                            DI_data_inst[vis]['FWHM_pip'][iexp]=hdr[pre_txt+' FWHM'] 
                            DI_data_inst[vis]['ctrst_pip'][iexp]=hdr[pre_txt+' CONTRAST'] 
                            if inst in ['HARPS','SOPHIE','ESPRESSO','ESPRESSO_MR','HARPN','NIRPS_HA','NIRPS_HE']:
                                DI_data_inst[vis]['eFWHM_pip'][iexp]=hdr[pre_txt+' FWHM ERROR'] 
                                DI_data_inst[vis]['ectrst_pip'][iexp]=hdr[pre_txt+' CONTRAST ERROR'] 

                        #SNRs in all orders
                        #    - SNRs tables are kept associated with original orders
                        if inst in ['SOPHIE','CORALIE']: 
                            if inst in ['CORALIE']:pre_txt='SPE'
                            elif inst=='SOPHIE':pre_txt='CAL'  
                            for iorder in range(data_inst['nord_ref']):
                                data_prop[inst][vis]['SNRs'][iexp,iorder]=hdr['HIERARCH '+facil_inst+' DRS '+pre_txt+' EXT SN'+str(iorder)]        
                        elif inst=='CARMENES_VIS':
                            for iorder in range(data_inst['nord_ref']):
                                data_prop[inst][vis]['SNRs'][iexp,iorder]=hdr['HIERARCH CARACAL FOX SNR '+str(iorder)]  
                        elif inst in ['HARPS','ESPRESSO','ESPRESSO_MR','HARPN','NIRPS_HA','NIRPS_HE']:
                            for iorder in range(data_inst['nord_ref']):
                                data_prop[inst][vis]['SNRs'][iexp,iorder]=hdr['HIERARCH '+facil_inst+' QC ORDER'+str(iorder+1)+' SNR']  
                        elif inst=='EXPRES':
                            #Correspondance channel : original orders
                            #   1 : 28 ; 2 : 37 ; 3 : 45 ; 4 : 51 ; 5 : 58 ; 6 : 63 ; 7 : 68 ; 8 : 72
                            SNR_exp = np.zeros(86,dtype=float)*np.nan 
                            for ichannel,iord in zip(range(1,9),[28,37,45,51,58,63,68,72]): 
                                SNR_exp[iord] = (hdulist[2].header)['HIERARCH channel'+str(ichannel)+'_SNR']
                            data_prop[inst][vis]['SNRs'][iexp] = SNR_exp    

                        #Blaze and wavelength calibration files
                        if inst=='ESPRESSO':
                            for ii in range(1,34):
                                if hdr['HIERARCH ESO PRO REC1 CAL%i CATG'%(ii)] == 'BLAZE_A':
                                    data_prop[inst][vis]['BLAZE_A'][iexp]=hdr['HIERARCH ESO PRO REC1 CAL%i DATAMD5'%(ii)]
                                elif hdr['HIERARCH ESO PRO REC1 CAL%i CATG'%(ii)] == 'BLAZE_B':
                                    data_prop[inst][vis]['BLAZE_B'][iexp]=hdr['HIERARCH ESO PRO REC1 CAL%i DATAMD5'%(ii)]
                                elif hdr['HIERARCH ESO PRO REC1 CAL%i CATG'%(ii)] == 'WAVE_MATRIX_THAR_FP_A':
                                    data_prop[inst][vis]['WAVE_MATRIX_THAR_FP_A'][iexp]=hdr['HIERARCH ESO PRO REC1 CAL%i DATAMD5'%(ii)]                            
                                elif hdr['HIERARCH ESO PRO REC1 CAL%i CATG'%(ii)] == 'WAVE_MATRIX_FP_THAR_B':
                                    data_prop[inst][vis]['WAVE_MATRIX_THAR_FP_B'][iexp]=hdr['HIERARCH ESO PRO REC1 CAL%i DATAMD5'%(ii)]
                                                                    
                        #Telescope altitude angle (degres)
                        if inst=='ESPRESSO':
                            data_prop[inst][vis]['alt'][iexp]=hdr['HIERARCH ESO TEL'+tel_inst+' ALT'] 

                        #Telescope azimuth angle (degrees)
                        if inst=='ESPRESSO':
                            data_prop[inst][vis]['az'][iexp]=hdr['HIERARCH ESO TEL'+tel_inst+' AZ']                                         
                        
                        #Activity indexes
                        if DACE_sp:
                            dace_idexp = closest( DACE_idx['bjd'] ,coord_dic[inst][vis]['bjd'][iexp] )
                            for key in data_inst[vis]['act_idx']:
                                data_prop[inst][vis][key][iexp,:] = [DACE_idx[key][dace_idexp],DACE_idx[key+'_err'][dace_idexp]]
                        elif inst=='EXPRES':
                            data_prop[inst][vis]['ha'][iexp,:] = [(hdulist[1].header)['HALPHA'],1.879E-2]
                            data_prop[inst][vis]['s'][iexp,:] = [(hdulist[1].header)['S-VALUE'], 6.611E-3]
                            

                    ### End of current exposure    
                    
                ### End of exposures in visit
                
                #------------------------------------------------------------------------------------

                #Scaling of errors
                if not gen_dic[inst][vis]['flag_err']:print('           Uncertainties are set to sqrt(F)')
                if (inst not in gen_dic['g_err']):gen_dic['g_err'][inst]=1.
                elif gen_dic['g_err'][inst]!=1.:
                    print('           Uncertainties are scaled by sqrt('+str(gen_dic['g_err'][inst])+')')
                    data_dic_temp['cov']*=gen_dic['g_err'][inst]

                #Sort exposures by increasing time
                #    - remove exposures if requested, using their index after time-sorting 
                if (inst in gen_dic['used_exp']) and (vis in gen_dic['used_exp'][inst]) and (len(gen_dic['used_exp'][inst][vis])>0):
                    remove_exp=True
                    print('           Removing '+str(data_inst[vis]['n_in_visit']-len(gen_dic['used_exp'][inst][vis]))+' exposures from the analysis')
                else:remove_exp=False
                w_sorted=coord_dic[inst][vis]['bjd'].argsort()    
                for pl_loc in data_inst[vis]['studied_pl']:
                    for key in ['ecl','cen_ph','st_ph','end_ph','ph_dur','rv_pl','v_pl']:
                        coord_dic[inst][vis][pl_loc][key]=coord_dic[inst][vis][pl_loc][key][w_sorted]                        
                        if remove_exp:coord_dic[inst][vis][pl_loc][key] = coord_dic[inst][vis][pl_loc][key][gen_dic['used_exp'][inst][vis]]
                    for key in ['cen_pos','st_pos','end_pos']:
                        coord_dic[inst][vis][pl_loc][key]=coord_dic[inst][vis][pl_loc][key][:,w_sorted] 
                        if remove_exp:coord_dic[inst][vis][pl_loc][key] = coord_dic[inst][vis][pl_loc][key][:,gen_dic['used_exp'][inst][vis]] 
                for ar in data_inst[vis]['studied_ar']:
                    for key in list(gen_dic['ar_coord_par'])+['is_visible']:
                        coord_dic[inst][vis][ar][key]=coord_dic[inst][vis][ar][key][:,w_sorted] 
                        if remove_exp:coord_dic[inst][vis][ar][key] = coord_dic[inst][vis][ar][key][:,gen_dic['used_exp'][inst][vis]]
                for key in ['bjd','t_dur','RV_star_solCDM','RV_star_stelCDM','cen_ph_st','st_ph_st','end_ph_st']:
                    coord_dic[inst][vis][key]=coord_dic[inst][vis][key][w_sorted]
                    if remove_exp:coord_dic[inst][vis][key] = coord_dic[inst][vis][key][gen_dic['used_exp'][inst][vis]]
                            
                for key in data_prop[inst][vis]:
                    if type(data_prop[inst][vis][key]) not in [str,int]:
                        if len(data_prop[inst][vis][key].shape)==1:
                            data_prop[inst][vis][key]=data_prop[inst][vis][key][w_sorted]
                            if remove_exp:data_prop[inst][vis][key] = data_prop[inst][vis][key][gen_dic['used_exp'][inst][vis]]
                        else:
                            data_prop[inst][vis][key]=data_prop[inst][vis][key][w_sorted,:] 
                            if remove_exp:data_prop[inst][vis][key] = data_prop[inst][vis][key][gen_dic['used_exp'][inst][vis],:]
                for key in data_dic_temp:
                    data_dic_temp[key]= data_dic_temp[key][w_sorted]
                    if remove_exp:data_dic_temp[key] = data_dic_temp[key][gen_dic['used_exp'][inst][vis]]
                for key in DI_data_inst[vis]:
                    DI_data_inst[vis][key]=DI_data_inst[vis][key][w_sorted]   
                    if remove_exp:DI_data_inst[vis][key] = DI_data_inst[vis][key][gen_dic['used_exp'][inst][vis]] 
                if remove_exp:
                    n_in_visit = len(gen_dic['used_exp'][inst][vis])
                    data_inst[vis]['n_in_visit'] = n_in_visit
                    data_inst[vis]['dim_all'][0] = n_in_visit
                    data_inst[vis]['dim_sp'][0] = n_in_visit
                    data_inst[vis]['dim_ord'][0] = n_in_visit
             
                #Check if exposures have same duration
                coord_dic[inst][vis]['t_dur_d'] = coord_dic[inst][vis]['t_dur']/(3600.*24.) 
                if (coord_dic[inst][vis]['t_dur']==coord_dic[inst][vis]['t_dur'][0]).all():coord_dic[inst][vis]['cst_tdur']=True
                else:coord_dic[inst][vis]['cst_tdur']=False   

                #Mean SNR
                #    - used to weigh CCFs (over contributing orders)
                #    - used to weigh spectra (over all orders)
                if (not gen_dic['mock_data']):
                    if ('CCF' in data_inst['type']):  
                        if inst not in gen_dic['orders4ccf']:data_prop[inst][vis]['mean_SNR'] = np.nanmean(data_prop[inst][vis]['SNRs'],axis=1)  
                        else:data_prop[inst][vis]['mean_SNR'] = np.nanmean(data_prop[inst][vis]['SNRs'][:,gen_dic['orders4ccf'][inst]],axis=1)
                    elif ('spec' in data_inst['type']):
                        data_prop[inst][vis]['mean_SNR'] = np.nanmean(data_prop[inst][vis]['SNRs'],axis=1)

                #Identify exposures west/east of meridian
                #    - celestial meridian is at azimuth = 180, with azimuth counted clockwise from the celestial north
                #      exposures at az > 180 are to the west of the meridian, exposures at az < 180 to the east
                if (inst=='ESPRESSO') and (not gen_dic['mock_data']):
                    cond_eastmer = data_prop[inst][vis]['az'] < 180.
                    data_inst[vis]['idx_eastmer'] = np_where1D(cond_eastmer)
                    data_inst[vis]['idx_westmer'] = np_where1D(~cond_eastmer) 
                    data_inst[vis]['idx_mer'] = closest(data_prop[inst][vis]['az'] , 180.)               
                    
                    #Estimate telescope altitude (deg) at meridian
                    min_bjd = coord_dic[inst][vis]['bjd'][0]
                    max_bjd = coord_dic[inst][vis]['bjd'][-1]
                    dbjd_HR = 1./(3600.*24.)
                    nbjd_HR = round((max_bjd-min_bjd)/dbjd_HR)
                    bjd_HR=min_bjd+dbjd_HR*np.arange(nbjd_HR)
                    az_HR = CubicSpline(coord_dic[inst][vis]['bjd'],data_prop[inst][vis]['az'])(bjd_HR)
                    idx_mer_HR = closest(az_HR,180.)
                    alt_HR = CubicSpline(coord_dic[inst][vis]['bjd'],data_prop[inst][vis]['alt'])(bjd_HR)
                    data_inst[vis]['alt_mer'] = alt_HR[idx_mer_HR]  
                    data_inst[vis]['z_mer']  = np.sin(alt_HR[idx_mer_HR]*np.pi/180.) 
                     
                
                #------------------------------------------------------------------------------------
               
                #Undefined bins
                data_dic_temp['cond_def'] = ~np.isnan(data_dic_temp['flux'])

                #Definition of edge spectral table
                #    - spectral bins must be continuous for the resampling routine of the pipeline to work
                #    - bin width are included in ESPRESSO files, but to have exactly the same lower/upper boundary between successive bins we redefine them manually  
                data_dic_temp['edge_bins'] = def_edge_tab(data_dic_temp['cen_bins'])

                #Lower/upper spectral edges 
                #    - over all exposures in each order
                low_bin_ord_def = np.min(data_dic_temp['edge_bins'][:,:,0],axis=0)   
                high_bin_ord_def = np.max(data_dic_temp['edge_bins'][:,:,-1],axis=0)

                #Nominal common spectral grid
                nom_nspec = data_inst[vis]['nspec']
                cen_bins_com = data_dic_temp['cen_bins'][0]
                edge_bins_com = data_dic_temp['edge_bins'][0]
                low_edge_bins_com = edge_bins_com[:,0:-1]
                high_edge_bins_com = edge_bins_com[:,1::]

                #Bin size of spectral edges in each order of the common grid
                dbin_low_ord = edge_bins_com[:,1]-edge_bins_com[:,0]
                dbin_high_ord = edge_bins_com[:,-1]-edge_bins_com[:,-2]    

                #Common spectral tables
                #    - the common table is set to the table of the first exposure, defined in the input rest frame
                #      it is then extended on both side of each order, from the min/max over all exposures, to the maximum rv shift in all frames of the workflow 
                #    - we give the table the same structure as for spectra, with dimensions nord x nspec
                #      to keep a constant number of bins per order we use the maximum number of extended bins over all orders
                if ('spec' in data_inst['type']):

                    #Upper and lower extension
                    #    - from the lower / upper spectral edges in each order
                    #    - we use the approximation w_shifted = w*(1 - (rv_shift/c) ) to define the boundaries 
                    low_bin_ord_exten = low_bin_ord_def*(1. - (gen_dic['min_rv_shift']/c_light) ) 
                    high_bin_ord_exten = high_bin_ord_def*(1. - (gen_dic['max_rv_shift']/c_light) ) 
                    
                    #Number of extended bins
                    #    - we extend with a regular bin size in ln(w) space
                    #      see details in "Defining 1D table for conversion from 2D spectra"
                    dlnbin_low_ord = dbin_low_ord/low_edge_bins_com[:,0]
                    dlnbin_high_ord = dbin_high_ord/high_edge_bins_com[:,-1]
                    nlow_bin_exten = np.max(npint( np.floor( np.log(low_edge_bins_com[:,0]/low_bin_ord_exten)/np.log( dlnbin_low_ord + 1. ) ) ) )
                    nhigh_bin_exten = np.max(npint( np.floor( np.log(high_bin_ord_exten/high_edge_bins_com[:,-1])/np.log( dlnbin_high_ord + 1. ) ) ) )
   
                    #Extended edges
                    low_edge_bins_exten = ((low_edge_bins_com[:,0]*( dlnbin_low_ord + 1. )**(-np.arange(nlow_bin_exten+1))[:,None]).T)[:,::-1]
                    high_edge_bins_exten = (high_edge_bins_com[:,-1]*( dlnbin_high_ord + 1. )**np.arange(nhigh_bin_exten+1)[:,None]).T

                else:
                    
                    #Upper and lower extension
                    low_bin_ord_exten = low_bin_ord_def   - gen_dic['min_rv_shift']
                    high_bin_ord_exten = high_bin_ord_def - gen_dic['max_rv_shift']
                    
                    #Maximum number of extended bins
                    #    - taking the maximum over all order, between the edges of the common table and the min/max extensions
                    nlow_bin_exten  = np.max(npint((low_edge_bins_com[:,0]-low_bin_ord_exten)/dbin_low_ord)+1)
                    nhigh_bin_exten = np.max(npint((high_bin_ord_exten-high_edge_bins_com[:,-1])/dbin_high_ord)+1)
                    
                    #Extended edges
                    low_edge_bins_exten = ((low_edge_bins_com[:,0] - dbin_low_ord*np.arange(nlow_bin_exten+1)[:,None]).T)[:,::-1]
                    high_edge_bins_exten = (high_edge_bins_com[:,-1] + dbin_high_ord*np.arange(nhigh_bin_exten+1)[:,None]).T
                    
                #Extended centers
                low_cen_bins_exten = 0.5*(low_edge_bins_exten[:,0:-1]+low_edge_bins_exten[:,1::])
                high_cen_bins_exten = 0.5*(high_edge_bins_exten[:,0:-1]+high_edge_bins_exten[:,1::])

                #Number of bins in extended grid
                exten_nspec = nlow_bin_exten + nom_nspec + nhigh_bin_exten   

                #Final grid
                cen_bins_com_exten = np.zeros([data_inst['nord'],exten_nspec],float)*np.nan
                edge_bins_com_exten = np.zeros([data_inst['nord'],exten_nspec+1],float)*np.nan
                for iord in range(data_inst['nord']):
                    cen_bins_com_exten[iord] = np.concatenate((low_cen_bins_exten[iord], cen_bins_com[iord], high_cen_bins_exten[iord]))
                    edge_bins_com_exten[iord] = np.concatenate((low_edge_bins_exten[iord,0:-1], edge_bins_com[iord], high_edge_bins_exten[iord,1::]))
                    
                #Storing
                data_inst[vis]['proc_com_data_paths'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_'+vis+'_com'
                data_com = {'cen_bins':cen_bins_com_exten,'edge_bins':edge_bins_com_exten}
                data_com['nspec'] = exten_nspec
                data_com['dim_ord'] = [n_in_visit,exten_nspec]
                data_com['dim_all'] = [n_in_visit,data_inst['nord'],exten_nspec]
                data_com['dim_exp'] = [data_inst['nord'],exten_nspec]
                data_com['dim_sp'] = [n_in_visit,data_inst['nord']]

                #Initializes the common table in the star rest frame
                #    - it will be updated if disk-integrated profiles are aligned
                data_inst[vis]['proc_com_star_data_paths'] = deepcopy(data_inst[vis]['proc_com_data_paths'])

                #Exposures in current visit are defined on different tables
                if ('spec' in data_inst['type']) and (np.nanmax(np.abs(data_dic_temp['edge_bins']-data_dic_temp['edge_bins'][0]))>1e-6):
                    data_inst[vis]['comm_sp_tab'] = False

                    #Resampling data on wavelength table common to the visit if requested
                    if gen_dic['comm_sp_tab'][inst]:
                        print('           Resampling on common table')

                        #Processing each exposure    
                        flux_resamp = np.zeros(data_com['dim_all'], dtype=float)*np.nan
                        cov_resamp = np.zeros(data_com['dim_sp'], dtype=object)   
                        if ('tell' in data_dic_temp):tell_resamp =  np.ones(data_com['dim_all'], dtype=float)
                        if (vis in data_inst['gcal_blaze_vis']):
                            gcal_blaze_resamp = np.zeros(data_com['dim_all'], dtype=float)*np.nan
                            sdet2_resamp = np.zeros(data_com['dim_all'], dtype=float)
                        for iexp in range(n_in_visit):
                            for iord in range(data_inst['nord']): 
                                flux_resamp[iexp,iord],cov_resamp[iexp][iord] = bind.resampling(data_com['edge_bins'][iord], data_dic_temp['edge_bins'][iexp,iord], data_dic_temp['flux'][iexp,iord] , cov = data_dic_temp['cov'][iexp][iord], kind=gen_dic['resamp_mode']) 
                                
                                #Resampling input telluric spectrum
                                if ('tell' in data_dic_temp):
                                    tell_resamp[iexp,iord] = bind.resampling(data_com['edge_bins'][iord], data_dic_temp['edge_bins'][iexp,iord], data_dic_temp['tell'][iexp,iord], kind=gen_dic['resamp_mode']) 
                                    
                                #Resampling blazed-derived calibration and detector noise profiles
                                if (vis in data_inst['gcal_blaze_vis']):
                                    gcal_blaze_resamp[iexp,iord] = bind.resampling(data_com['edge_bins'][iord], data_dic_temp['edge_bins'][iexp,iord], data_dic_temp['gcal'][iexp,iord], kind=gen_dic['resamp_mode'])
                                    sdet2_resamp[iexp,iord] = bind.resampling(data_com['edge_bins'][iord], data_dic_temp['edge_bins'][iexp,iord], data_dic_temp['sdet2'][iexp,iord], kind=gen_dic['resamp_mode'])

                        #Overwrite exposure tables
                        data_dic_temp['cen_bins'][:] = deepcopy(data_com['cen_bins'])
                        data_dic_temp['edge_bins'][:] = deepcopy(data_com['edge_bins'])  
                        data_dic_temp['flux'] = flux_resamp 
                        data_dic_temp['cov'] = cov_resamp 
                        data_dic_temp['cond_def'] = ~np.isnan(data_dic_temp['flux'])
                        if ('tell' in data_dic_temp):data_dic_temp['tell'] = tell_resamp
                        if (vis in data_inst['gcal_blaze_vis']):
                            data_dic_temp['gcal'] = gcal_blaze_resamp
                            data_dic_temp['sdet2'] = sdet2_resamp
                        
                        #Set flag that all exposures in the visit are now defined on a common table
                        data_inst[vis]['comm_sp_tab'] = True

                #Save instrument table common to all visits
                #    - set to the common table of the first instrument visit
                if ivis==0:
                    data_dic[inst]['proc_com_data_path'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_com'
                    data_dic[inst]['proc_com_star_data_path'] = deepcopy(data_dic[inst]['proc_com_data_path'])
                    data_dic[inst]['dim_exp'] = deepcopy(data_inst[vis]['dim_exp'])
                    data_dic[inst]['nspec'] = deepcopy(data_inst[vis]['nspec'])
                    data_dic[inst]['com_vis'] = vis   
                    data_com_inst = deepcopy(data_com)
                    datasave_npz(data_dic[inst]['proc_com_data_path'],data_com_inst)

                #Check wether all visits of current instrument share a common table
                #    - flag remains True if the common table of current visit is the same as the common table for the instrument
                #    - flag is set to False if :
                # + exposures of current visit are still defined on individual table (= no common table even for the visit) 
                # + exposures of current visit are defined on a common table, but it is different from the instrument table (only possible if current visit is not the first, used as reference)   
                if (not data_inst[vis]['comm_sp_tab']) or ((ivis>0) and ((data_com_inst['nspec']!=data_com['nspec']) or (np.nanmax(np.abs(data_com_inst['edge_bins']-data_com['edge_bins']))>1e-6))):
                      data_dic[inst]['comm_sp_tab'] = False

                #------------------------------------------------------------------------------------

                #Masking condition
                if ('spec' in data_inst['type']) and (inst in gen_dic['masked_pix']) and (vis in gen_dic['masked_pix'][inst]):
                    cond_mask_gen = True
                    print('           Masking spectral ranges from the analysis')
                else:cond_mask_gen = False
            
                #Final processing of exposures
                data_inst[vis]['proc_DI_data_paths']=gen_dic['save_data_dir']+'Processed_data/'+inst+'_'+vis+'_'
                for iexp in range(n_in_visit):

                    #Set masked pixels to nan
                    #    - masked orders and positions are assumed to be the same for all exposures, but positions have to be specific to a given order as orders can overlap
                    if cond_mask_gen:
                        mask_exp = gen_dic['masked_pix'][inst][vis]['exp_list']
                        if (len(mask_exp)==0) or (iexp in mask_exp):
                            for iord in gen_dic['masked_pix'][inst][vis]['ord_list']:
                                mask_bd = gen_dic['masked_pix'][inst][vis]['ord_list'][iord]
                                cond_mask  = np.zeros(data_inst[vis]['nspec'],dtype=bool)
                                for bd_int in mask_bd:
                                    cond_mask |= (data_dic_temp['edge_bins'][iexp,iord,0:-1]>=bd_int[0]) & (data_dic_temp['edge_bins'][iexp,iord,1:]<=bd_int[1])
                                data_dic_temp['cond_def'][iexp ,iord , cond_mask ] = False    
                                data_dic_temp['flux'][iexp ,iord , cond_mask ]  = np.nan                            

                    #Telluric spectrum
                    #    - if defined with input data
                    #    - path is made specific to the exposure to be able to point from in-transit to global profiles without copying them to disk
                    if ('tell' in data_dic_temp):
                        data_inst[vis]['tell_DI_data_paths'][iexp] = data_inst[vis]['proc_DI_data_paths']+'tell_'+str(iexp)
                        datasave_npz(data_inst[vis]['tell_DI_data_paths'][iexp],{'tell':data_dic_temp['tell'][iexp]}) 

                    #Blazed-derived calibration and detector noise profiles
                    #    - if defined with input data
                    #    - path is made specific to the exposure to be able to point from in-transit to global profiles without copying them to disk
                    if ('sing_gcal_DI_data_paths' in data_inst[vis]):data_inst[vis]['sing_gcal_DI_data_paths'][iexp] = data_inst[vis]['proc_DI_data_paths']+'sing_gcal_'+str(iexp)
                    if (vis in data_inst['gcal_blaze_vis']):
                        data_gcal_save = {'gcal':data_dic_temp['gcal'][iexp]}
                        data_gcal_save['sdet2'] = data_dic_temp['sdet2'][iexp]
                        datasave_npz(data_inst[vis]['sing_gcal_DI_data_paths'][iexp],data_gcal_save)
                        
                    #Saving path to initialized raw data
                    #    - saving data per exposure to prevent size issue with npz files 
                    data_exp = {key:data_dic_temp[key][iexp] for key in ['cen_bins','edge_bins','flux','cov','cond_def']}
                    datasave_npz(data_inst[vis]['proc_DI_data_paths']+str(iexp),data_exp)
                               
                #Check for empty orders
                idx_ord_empty = np_where1D(np.sum(data_dic_temp['cond_def'],axis=(0,2)) == 0.)
                if len(idx_ord_empty)>0:
                    txt_str = '['
                    for iord in idx_ord_empty[0:-1]:txt_str+=str(iord)+','
                    txt_str+=str(idx_ord_empty[-1])+']'
                    print('           Empty orders : '+txt_str)

                #Defined edges of common spectral table
                low_bins_def = deepcopy(data_dic_temp['edge_bins'][:,:,0:-1])
                high_bins_def = deepcopy(data_dic_temp['edge_bins'][:,:,1::])
                low_bins_def[~data_dic_temp['cond_def']] = np.nan
                high_bins_def[~data_dic_temp['cond_def']] = np.nan
                data_com['min_edge_ord'] = np.nanmin(low_bins_def,axis=(0,2))             
                data_com['max_edge_ord'] = np.nanmax(high_bins_def,axis=(0,2))
                datasave_npz(data_inst[vis]['proc_com_data_paths'],data_com)

                #Saving useful keyword file        
                if gen_dic['sav_keywords']:
                    tup_save=(np.append('index',[str(iexp) for iexp in range(n_in_visit)]),)
                    form_save=('%-6s',)
                    for key,form in zip(['BLAZE_A','BLAZE_B','WAVE_MATRIX_THAR_FP_A','WAVE_MATRIX_THAR_FP_B'],
                                        ['%-35s'  ,'%-35s'  ,'%-35s'                ,'%-35s'   ]):
                        if key in data_prop[inst][vis]:
                            form_save+=(form,)
                            col_loc = np.append(key,[str(data_prop[inst][vis][key][iexp]) for iexp in range(n_in_visit)])
                            tup_save+=(col_loc,)
                    np.savetxt(gen_dic['save_data_dir']+'DIorig_prop/'+inst+'_'+vis+'_keywords.dat', np.column_stack(tup_save),fmt=form_save) 

                #Saving dictionary elements defined within the routine
                np.savez_compressed(gen_dic['save_data_dir']+'Processed_data/Global/'+inst+'_'+vis,data_add=data_inst[vis],coord_add=coord_dic[inst][vis],data_prop_add=data_prop[inst][vis],gen_add=gen_dic[inst][vis],DI_data_add=DI_data_inst[vis],allow_pickle=True)

            ### End of visits

        #Saving general information
        np.savez_compressed(gen_dic['save_data_dir']+'Processed_data/Global/'+inst,gen_inst_add=gen_dic[inst],data_inst_add=data_dic[inst],allow_pickle=True) 

    #Retrieving data
    else:
        check_data({'path':gen_dic['save_data_dir']+'Processed_data/Global/'+inst})    
        data_load = np.load(gen_dic['save_data_dir']+'Processed_data/Global/'+inst+'.npz',allow_pickle=True)
        data_dic[inst]=data_load['data_inst_add'].item()   
        gen_dic[inst]= data_load['gen_inst_add'].item() 
       
        #Remove visits
        #    - this allows removing visits after processing all of them once via gen_dic['calc_proc_data']
        for vis in gen_dic['unused_visits'][inst]:
            if vis in data_dic[inst]['visit_list']:data_dic[inst]['visit_list'].remove(vis)

        #Retrieve/initialize visit dictionaries
        for vis in data_dic[inst]['visit_list']:     
            data_load = np.load(gen_dic['save_data_dir']+'Processed_data/Global/'+inst+'_'+vis+'.npz',allow_pickle=True)
            data_dic[inst][vis]=data_load['data_add'].item()
            data_dic[inst][vis]['proc_DI_data_paths']  = gen_dic['save_data_dir']+'Processed_data/'+inst+'_'+vis+'_'           #temporary redef; retrieved in  data_load['data_add'].item() no ?
            data_dic[inst][vis]['proc_com_data_paths'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_'+vis+'_com'        #temporary redef; retrieved in  data_load['data_add'].item() no ?
            coord_dic[inst][vis]=data_load['coord_add'].item()
            data_prop[inst][vis]=data_load['data_prop_add'].item() 
            gen_dic[inst][vis]=data_load['gen_add'].item()
            data_dic['DI'][inst][vis]=data_load['DI_data_add'].item()
            theo_dic[inst][vis]={}
            data_dic['Atm'][inst][vis]={} 
            
    #Default transit model
    if (inst not in data_dic['DI']['transit_prop']):data_dic['DI']['transit_prop'][inst] = {}

    #Duplicate chromatic properties so that they are not overwritten by conversions
    data_dic[inst]['system_prop'] = deepcopy(data_dic['DI']['system_prop'])
    data_dic[inst]['ar_prop'] = deepcopy(data_dic['DI']['ar_prop'])

    #Creating relevant directories
    if gen_dic['EvE_outputs']:
        if (not path_exist(gen_dic['save_data_dir']+'EvE_outputs/'+inst)):makedirs(gen_dic['save_data_dir']+'EvE_outputs/'+inst)

    #Final processing
    if len(data_dic[inst]['visit_list'])>1:
        if (not data_dic[inst]['comm_sp_tab']):print('         Visits do not share a common spectral table')      
        else:print('         All visits share a common spectral table')    
    ar_check = False
    for vis in data_dic[inst]['visit_list']:
        if (vis in data_dic[inst]['single_night']):    
            print('         Processing visit '+vis)
            if not gen_dic['mock_data']:                                  
                if vis in data_dic[inst]['dates']:print('           Date (night start) : '+data_dic[inst]['dates'][vis])
                if vis in data_dic[inst]['midpoints']:print('           Visit midpoint: '+data_dic[inst]['midpoints'][vis])
        else:
            print('         Processing multi-epoch visit '+vis)              
        data_vis = data_dic[inst][vis]
        coord_vis = coord_dic[inst][vis]
        gen_vis = gen_dic[inst][vis] 
        if (not data_vis['comm_sp_tab']):print('           Exposures do not share a common spectral table')      
        else:print('           All exposures share a common spectral table')   

        #Keeping track of original dimensions
        data_vis['nspec_ref'] = deepcopy(data_vis['nspec'])

        #Initialize all exposures as being defined
        data_dic['DI'][inst][vis]['idx_def'] = np.arange(data_vis['n_in_visit'],dtype=int)
    
        #Creating relevant directories
        if gen_dic['EvE_outputs']:
            if (not path_exist(gen_dic['save_data_dir']+'EvE_outputs/'+inst+'/'+vis)):makedirs(gen_dic['save_data_dir']+'EvE_outputs/'+inst+'/'+vis)

        #------------------------------------------------------------------------------------

        #Default systemic velocity
        if (inst not in data_dic['DI']['sysvel']):data_dic['DI']['sysvel'][inst] = {}
        if (vis not in data_dic['DI']['sysvel'][inst]):
            data_dic['DI']['sysvel'][inst][vis] = system_param['star']['sysvel']
            print('           WARNING : sysvel set to default '+str(system_param['star']['sysvel'])+' km/s')

        #Min/max BERV
        if ('spec' in data_dic[inst]['type']):
            data_vis['min_BERV'] = np.nanmin(data_prop[inst][vis]['BERV'])
            data_vis['max_BERV'] = np.nanmax(data_prop[inst][vis]['BERV'])
            
        #Define CCF resolution, range, and sysvel here so that spectral data needs not be uploaded again after processing
        if gen_dic['CCF_from_sp'] or gen_dic['Intr_CCF']:
          
            #Velocity table in input frame
            if gen_dic['dRV'] is None:dvelccf= return_pix_size(inst)
            else:dvelccf= gen_dic['dRV']  
            n_vel = int((gen_dic['end_RV']-gen_dic['start_RV'])/dvelccf)+1
            data_vis['nvel'] = n_vel
            data_vis['velccf'] = gen_dic['start_RV']+dvelccf*np.arange(n_vel)
            data_vis['edge_velccf'] = np.append(data_vis['velccf']-0.5*dvelccf,data_vis['velccf'][-1]+0.5*dvelccf)

            #Common velocity table in original frame
            nlow_bin_exten  = int( gen_dic['min_rv_shift']/dvelccf)+1
            nhigh_bin_exten  = int(-gen_dic['max_rv_shift']/dvelccf)+1
            low_edge_bins_exten = (data_vis['edge_velccf'][0] - dvelccf*np.arange(nlow_bin_exten+1))[::-1]
            high_edge_bins_exten = (data_vis['edge_velccf'][-1] + dvelccf*np.arange(nhigh_bin_exten+1))           
            low_cen_bins_exten = 0.5*(low_edge_bins_exten[0:-1]+low_edge_bins_exten[1::])
            high_cen_bins_exten = 0.5*(high_edge_bins_exten[0:-1]+high_edge_bins_exten[1::])            
            data_vis['velccf_com'] = np.concatenate((low_cen_bins_exten, data_vis['velccf'], high_cen_bins_exten))
            data_vis['edge_velccf_com'] = np.concatenate((low_edge_bins_exten[0:-1], data_vis['edge_velccf'], high_edge_bins_exten[1::]))

            #Shifting the CCFs velocity table by RV(CDM_star/CDM_sun)
            for key in ['velccf','edge_velccf','velccf_com','edge_velccf_com']:data_vis[key+'_star']=data_vis[key]-data_dic['DI']['sysvel'][inst][vis]  
    
        #Automatic continuum and fit range
        #    - done here rather than in the 'calc_proc_data' condition so that ranges can be defined for already-processed observed or mock datasets, even if the analysis modules were not activated 
        # at the time of processing
        #    - called if fit of single or joined profiles are requested, and if the continuum of differential profiles is not defined in CCF mode
        for key in gen_dic['type_list']:
            if ((key=='Diff') and gen_dic['diff_data'] and (data_dic[key]['type'][inst]=='CCF')) or gen_dic['fit_'+key+'_gen'] or gen_dic['fit_'+key+'Prof']:
                autom_cont = True if (inst not in data_dic[key]['cont_range']) else False
                autom_fit = True if (key!='Diff') and ((inst not in data_dic[key]['fit_range']) or (vis not in data_dic[key]['fit_range'][inst])) else False
                if autom_cont or autom_fit:

                    #Path to data used for automatic definition
                    #    - if spectra are converted into CCFs, we cannot use the original spectral data
                    if (data_dic[key]['type'][inst]=='CCF') and (data_dic[inst]['type']=='spec2D'):
                        data_path = gen_dic['save_data_dir']+key+'_data/CCFfromSpec/'+gen_dic['add_txt_path'][key]+'/'+inst+'_'+vis+'_'
                        nspec = data_vis['nvel']
                    else:
                        data_path = data_vis['proc_DI_data_paths']
                        nspec = data_vis['nspec']

                    #Processing CCF order or selected spectral order 
                    if (data_dic[key]['type'][inst]=='CCF'):iord_sel = 0
                    else:iord_sel = data_dic[key]['fit_order'][inst]
                    cen_bins_all = np.zeros([0,nspec],dtype=float)
                    edge_bins_all = np.zeros([0,nspec+1],dtype=float)
                    flux_all = np.zeros([0,nspec],dtype=float)   
                    for iexp in range(data_vis['n_in_visit']):
                        data_exp = dataload_npz(data_path+str(iexp))
                        cen_bins_all=np.append(cen_bins_all,[data_exp['cen_bins'][iord_sel]],axis=0)     #dimension nexp x nspec
                        edge_bins_all=np.append(edge_bins_all,[data_exp['edge_bins'][iord_sel]],axis=0)  #dimension nexp x nspec+1
                        flux_all=np.append(flux_all,[data_exp['flux'][iord_sel]],axis=0)                 #dimension nexp x nspec
  
                    #Range space
                    #    - continuum is defined in RV space if data is in CCF mode, and in wavelength space if data is in spectral mode
                    #    - fit range is defined in RV space if data is in CCF mode or in spectral mode and fitted with an analytical model on a single line, and in wavelength space otherwise
                    if autom_fit:
                        if ('spec' in data_dic[key]['type'][inst]) and (data_dic[key]['fit_prof']['mode']=='ana'):
                            if data_dic[key]['fit_prof']['line_trans'] is None:stop('Define "line_trans" to fit spectral data with "mode = ana"')
                            spec2rv = True
                        else:spec2rv = False     
                    else:spec2rv = False
                    
                    #Condition to define line-specific boundaries
                    if autom_fit or (autom_cont and ((data_dic[key]['type'][inst]=='CCF') or spec2rv)):
                        if (data_dic[key]['type'][inst]=='CCF'):
                            RVcen_bins = cen_bins_all
                            RVedge_bins = edge_bins_all
                        else:
                            RVcen_bins = c_light*((cen_bins_all/data_dic[key]['line_trans']) - 1.)
                            RVedge_bins = c_light*((edge_bins_all/data_dic[key]['line_trans']) - 1.)

                        #Estimate of systemic velocity for disk-integrated profiles
                        #    - as median of the RV corresponding to the CCF minima over the visit
                        #    - differential, intrinsic and atmospheric profiles are analyzed in the star rest frame
                        cond_check = (RVcen_bins[0]>-500.) & (RVcen_bins[0]<500.)
                        idx_min_all = np.argmin(flux_all[:,cond_check],axis=1)    #dimension nexp (index of minimum over nspec_check for each exposure)
                        sys_RV = np.nanmedian(np.take_along_axis(RVcen_bins[:,cond_check],idx_min_all[:,None],axis=1))
                        if key=='DI':cen_RV = sys_RV
                        else:cen_RV = 0.
    
                        #Minimum/maximum velocity of the CCF range in the input rest frame
                        min_bin = np.max(np.nanmin(RVedge_bins))
                        max_bin = np.min(np.nanmax(RVedge_bins)) 
                        if key in ['Diff','Intr','Atm']:
                            min_bin-=sys_RV
                            max_bin-=sys_RV
                           
                        #Excluded range for disk-integrated, differential, and intrinsic profiles 
                        #    - we assume +-3 vsini accounts for both rotational and thermal broadening of DI profiles, and for rotational shift + thermal broadening
                        if key in ['DI','Diff','Intr']:
                            min_exc = np.max([cen_RV - 3.*system_param['star']['vsini'] -5.,min_bin+5.])
                            max_exc = np.min([cen_RV + 3.*system_param['star']['vsini'] +5.,max_bin-5.])
    
                        #Excluded planet range
                        #    - we account for a width of 20km/s over the range of studied orbital velocities, for all planets considered for atmospheric signals
                        if (key=='Atm') or (len(data_dic['Atm']['no_plrange'])>0):
                            min_pl_RV = 1e100
                            max_pl_RV = -1e100
                            if data_dic['Atm']['pl_atm_sign']=='Absorption':idx_atm = gen_dic[inst][vis]['idx_in']                                 
                            elif data_dic['Atm']['pl_atm_sign']=='Emission': idx_atm = data_dic[inst][vis]['n_in_tr']
                            for pl_atm in data_dic[inst][vis]['pl_with_atm']:
                                min_pl_RV = np.min([min_pl_RV,np.min(coord_dic[inst][vis][pl_atm]['rv_pl'][idx_atm])])
                                max_pl_RV = np.max([max_pl_RV,np.max(coord_dic[inst][vis][pl_atm]['rv_pl'][idx_atm])])                          
                            
                            #Exclusion in atmospheric profiles
                            if key=='Atm':
                                min_exc = np.min(min_pl_RV -20.,min_bin+5.)
                                max_exc = np.max(max_pl_RV +20.,max_bin-5.)   
                                
                            else:
                            
                                #Exclusion in other profiles
                                if (key=='DI') and ('DI' in key_loc for key_loc in data_dic['Atm']['no_plrange']):
                                    min_exc = np.min((min_exc,min_pl_RV -20.+sys_RV,min_bin+5.))
                                    max_exc = np.max((max_exc,max_pl_RV +20.+sys_RV,max_bin-5.))   
                                    
                                elif ((key=='Diff') and ('Diff' in key_loc for key_loc in data_dic['Atm']['no_plrange'])) or ((key=='Intr') and ('Intr' in key_loc for key_loc in data_dic['Atm']['no_plrange'])):
                                    min_exc = np.min((min_exc,min_pl_RV -20.,min_bin+5.))
                                    max_exc = np.max((max_exc,max_pl_RV +20.,max_bin-5.))                                   
                            
                        #Outer boundaries of continuum and fitted ranges in RV space
                        min_contfit = np.max([min_bin, min_exc - 50.])
                        max_contfit = np.min([max_bin, max_exc + 50.])                            

                    #Continuum range
                    #    - in spectral mode, the continuum is set to the full line range if a fit is requested in spectral mode with an analytical model
                    if autom_cont:
                        if (data_dic[key]['type'][inst]=='CCF'):
                            data_dic[key]['cont_range'][inst] = {iord_sel:[[min_contfit,min_exc],[max_exc,max_contfit]]}
                        else:
                            if spec2rv:
                                min_contrange = data_dic[key]['line_trans']*gen_specdopshift(min_contfit)  
                                min_exrange = data_dic[key]['line_trans']*gen_specdopshift(min_exc)  
                                max_contrange = data_dic[key]['line_trans']*gen_specdopshift(max_contfit)  
                                max_exrange = data_dic[key]['line_trans']*gen_specdopshift(max_exc)  
                                data_dic[key]['cont_range'][inst] = {iord_sel:[[min_contrange,min_exrange],[max_exrange,max_contrange]]}
                            else:
                                data_dic[key]['cont_range'][inst] = {iord_sel:[[edge_bins_all[0],edge_bins_all[-1]]]}
             
                    #Fitting range
                    if autom_fit:      
                        if inst not in data_dic[key]['fit_range']:data_dic[key]['fit_range'][inst]={}
                        if spec2rv:                        
                            min_fitrange = data_dic[key]['line_trans']*gen_specdopshift(min_contfit)  
                            max_fitrange = data_dic[key]['line_trans']*gen_specdopshift(max_contfit)  
                        else:
                            min_fitrange = min_contfit
                            max_fitrange = max_contfit
                        data_dic[key]['fit_range'][inst][vis] = [[min_fitrange,max_fitrange]]

        #------------------------------------------------------------------------------------

        #Keplerian motion relative to the stellar CDM and the Sun (km/s)
        for iexp in range(data_dic[inst][vis]['n_in_visit']):
            coord_vis['RV_star_stelCDM'][iexp],coord_vis['RV_star_solCDM'][iexp] = calc_rv_star(coord_dic,inst,vis,system_param,gen_dic,coord_dic[inst][vis]['bjd'][iexp],coord_dic[inst][vis]['t_dur'][iexp],data_dic['DI']['sysvel'][inst][vis])

        #Using sky-corrected data for current visit
        if (inst in gen_dic['fibB_corr']) and (vis in gen_dic['fibB_corr'][inst]):print('          ',vis,'is sky-corrected') 

        #Indices of in/out exposures in global tables
        print('           '+str(data_vis['n_in_visit'])+' exposures')  
        if ('in' in data_dic['DI']['idx_ecl']) and (inst in data_dic['DI']['idx_ecl']['in']) and (vis in data_dic['DI']['idx_ecl']['in'][inst]):
            for iexp in data_dic['DI']['idx_ecl']['in'][inst][vis]:
                for pl_loc in data_vis['studied_pl']:coord_vis[pl_loc]['ecl'][iexp] = 3*np.sign(coord_vis[pl_loc]['ecl'][iexp]) 
        if ('out' in data_dic['DI']['idx_ecl']) and (inst in data_dic['DI']['idx_ecl']['out']) and (vis in data_dic['DI']['idx_ecl']['out'][inst]):
            for iexp in data_dic['DI']['idx_ecl']['out'][inst][vis]:
                for pl_loc in data_vis['studied_pl']:coord_vis[pl_loc]['ecl'][iexp] = 1*np.sign(coord_vis[pl_loc]['ecl'][iexp]) 
        cond_in = np.zeros(data_vis['n_in_visit'],dtype=bool)
        cond_pre = np.ones(data_vis['n_in_visit'],dtype=bool) 
        cond_post = np.ones(data_vis['n_in_visit'],dtype=bool) 
        for pl_loc in data_vis['studied_pl']:
            cond_in|=(np.abs(coord_vis[pl_loc]['ecl'])>1) 
            cond_pre&=(coord_vis[pl_loc]['ecl']==-1.)
            cond_post&=(coord_vis[pl_loc]['ecl']==1.)
        gen_vis['idx_in']=np_where1D(cond_in)
        gen_vis['idx_out']=np_where1D(~cond_in)
        gen_vis['idx_pretr']=np_where1D(cond_pre)  
        gen_vis['idx_posttr']=np_where1D(cond_post) 
        data_vis['n_in_tr'] = len(gen_vis['idx_in'])
        data_vis['n_out_tr'] = len(gen_vis['idx_out'])
        gen_vis['idx_exp2in'] = np.zeros(data_vis['n_in_visit'],dtype=int)-1
        gen_vis['idx_exp2in'][gen_vis['idx_in']]=np.arange(data_vis['n_in_tr'])
        gen_vis['idx_in2exp'] = np.arange(data_vis['n_in_visit'],dtype=int)[gen_vis['idx_in']]        
        
        #Duplicate chromatic properties so that they are not overwritten by conversions
        data_vis['system_prop'] = deepcopy(data_dic['DI']['system_prop'])
        data_vis['ar_prop'] = deepcopy(data_dic['DI']['ar_prop'])
        if len(data_vis['studied_ar'])>0:ar_check = True
    if (len(gen_dic['studied_ar'])>0) and (not ar_check):stop('ERROR: no active regions are associated with any visit but are requested in simulation - reset workflow with "gen_dic["calc_proc_data"]"')
    if (len(gen_dic['studied_ar'])==0) and (ar_check):stop('ERROR: active regions are associated with visits but not requested in simulation - reset workflow with "gen_dic["calc_proc_data"]"')
        
    #Total number of visits for current instrument
    gen_dic[inst]['n_visits'] = len(data_dic[inst]['visit_list'])
    data_dic[inst]['n_visits_inst']=len(data_dic[inst]['visit_list'])
 
    #Defining 1D table for conversion from 2D spectra
    #    - uniformly spaced in ln(w)
    #    - we impose d[ln(w)] = dw/w = ( w(i+1)-w(i) ) / w(i) 
    #      thus the table can be built with:
    # d[ln(w)]*w(i)  = ( w(i+1) - w(i) )  
    # w(i+1) = ( d[ln(w)] + 1 )*w(i) 
    # w(i+1) = ( d[ln(w)] + 1 )^2*w(i-1)
    # w(i) = ( d[ln(w)] + 1 )^i*w(0)
    #       the table ends at the largest n so that :
    # w(n) <= wmax < w(n+1)
    # ( d[ln(w)] + 1 )^n <= wmax/w(0) < ( d[ln(w)] + 1 )^(n+1)
    # n*ln( d[ln(w)] + 1 ) <= ln(wmax/w(0)) < (n+1)*ln( d[ln(w)] + 1 )
    # n <= ln(wmax/w(0))/ln( d[ln(w)] + 1 ) < (n+1)
    # n = E( ln(wmax/w(0))/ln( d[ln(w)] + 1 ) )
    #    - we do not set an automatic definition because tables for intrinsic and atmospheric profiles will be shifted compared to the original tables
    if gen_dic['spec_1D']:
        def def_1D_bins(prop_dic):
            spec_1D_prop = prop_dic['spec_1D_prop'][inst]  
            spec_1D_prop['nspec'] = int( np.floor(   np.log(spec_1D_prop['w_end']/spec_1D_prop['w_st'])/np.log( spec_1D_prop['dlnw'] + 1. ) )  ) 
            spec_1D_prop['edge_bins'] = spec_1D_prop['w_st']*( spec_1D_prop['dlnw'] + 1. )**np.arange(spec_1D_prop['nspec']+1)    
            spec_1D_prop['cen_bins'] = 0.5*(spec_1D_prop['edge_bins'][0:-1]+spec_1D_prop['edge_bins'][1::])      
            return None
        for key in ['DI','Intr','Atm']:
            if gen_dic['spec_1D_'+key]:
            
                #Default 1D table
                #    - tables are uniformely spaced in ln(w), with d[ln(w)] = dw/w = drv/c
                #      we use the instrument pixel size defined in rv space to set a default value
                if (inst not in data_dic[key]['spec_1D_prop']):
                    dlnw_inst = return_pix_size(inst)/c_light
                    w_st_inst_all =  {'ESPRESSO':3000.,'HARPN':3000.,'HARPS':3000.,'CARMENES_VIS':4500., 'NIRPS_HA':9500.}
                    if (inst not in w_st_inst_all):print('ERROR : define w_st for "'+inst+'"')
                    w_end_inst_all = {'ESPRESSO':8500.,'HARPN':7500.,'HARPS':7500.,'CARMENES_VIS':11500.,'NIRPS_HA':21000.}
                    if (inst not in w_end_inst_all):print('ERROR : define w_end for "'+inst+'"')
                    data_dic[key]['spec_1D_prop'][inst] = {'dlnw':dlnw_inst,'w_st':w_st_inst_all[inst],'w_end':w_end_inst_all[inst]}
                    print('          Automatic definition of 1D '+gen_dic['type_name'][key]+' table : dlnw = '+"{0:.3e}".format(dlnw_inst)+ ' ; w_st = '+"{0:.3f}".format(w_st_inst_all[inst])+' A ; w_end = '+"{0:.3f}".format(w_end_inst_all[inst])+' A.')

                #Defining table                                
                def_1D_bins(data_dic[key])
        
    return None  






def init_vis(data_prop,data_dic,vis,coord_dic,inst,system_param,gen_dic):
    r"""**Initialization: visit**

    Initializes visit-specific fields for the workflow.
    
    Args:
        TBD
    
    Returns:
        None
    
    """     
    #Reset data mode to input
    data_dic[inst]['nord'] = deepcopy(data_dic[inst]['nord_spec'])    
    
    #Dictionaries initialization
    data_vis=data_dic[inst][vis]
    coord_vis = coord_dic[inst][vis] 
    data_dic['Diff'][inst][vis]={}
    gen_vis = gen_dic[inst][vis] 
    data_dic['Intr'][inst][vis]={}    

    #Current rest frame
    data_dic['DI'][inst][vis]['rest_frame'] = 'input'   
    
    #Current data
    data_vis['raw_exp_data_paths'] = deepcopy(data_vis['proc_DI_data_paths'])
    
    #Indices of in/out exposures in global tables
    print('   > '+str(data_vis['n_in_visit'])+' exposures')  
    print('        ',data_vis['n_in_tr'],'in-transit')
    txt_loc = '         '+str(data_vis['n_out_tr'])+' out-of-transit'
    if data_vis['n_in_tr']>0:txt_loc+=' ('+str(len(gen_vis['idx_pretr']))+' pre / '+str(len(gen_vis['idx_posttr']))+' post)'
    print(txt_loc)
    data_vis['dim_in'] = [data_vis['n_in_tr']]+data_vis['dim_exp']

    #Definition of CCF continuum, and ranges contributing to master out
    #    - the continuum of a given CCFs account for the ranges selected as input, the planetary exclusion ranges, and the bins defined in the CCFs
    #    - the final continuum is common to all CCFs
    #      if at least one bin is undefined in one exposure it is undefined in the continuum
    plAtm_vis = data_dic['Atm'][inst][vis]    

    #Generic definition of planetary range exclusion
    plAtm_vis['exclu_range_input']={'CCF':{},'spec':{}}
    plAtm_vis['exclu_range_star']={'CCF':{},'spec':{}}
    if (data_dic['Atm']['exc_plrange']):

        #Exposures to be processed
        #    - ranges are excluded from in-transit data only if left undefined as input, since most of the time only in-transit absorption signals are analyzed
        #      beware if an emission or absorption signal is present outside of the transit, in this case out-of-transit should be included manually
        if (inst in data_dic['Atm']['iexp_no_plrange']) and (vis in data_dic['Atm']['iexp_no_plrange'][inst]):plAtm_vis['iexp_no_plrange'] = data_dic['Atm']['iexp_no_plrange'][inst][vis]
        else:plAtm_vis['iexp_no_plrange'] = gen_vis['idx_in']
        iexp_no_plrange = plAtm_vis['iexp_no_plrange']
        
        #Planetary range in star and input rest frame
        #    - we define here the lower/upper wavelength boundaries of excluded ranges for each planetary mask line in each requested exposure, based on a common rv range excluded in the planet rest frame
        #    - the excluded range 'plrange' is defined in the planet rest frame, ie as RV(plrange/pl) 
        #    - masked profiles are either the input disk-integrated profiles, defined in their original sun barycentric rest frame (over RV(M/star) + RV(star/CDM_sun) ) or
        # processed profiles defined in the star rest frame (over RV(M/star))
        #    - tables have dimension (nline,2,nexp) in spectral mode, (2,nexp) in rv mode 
        #    - if input data is in spectral mode, ranges are defined in both spectral and rv space so that they are available in case of CCF conversion
        #      if input data is in CCF mode, ranges are only defined in rv  
        plrange_star={}
        for pl_loc in data_vis['studied_pl']:
            
            #Planetary range shifted from planet (source) to star (receiver) rest frame
            #    - we define the range excluded in the planet rest frame as rv(atom/pl), where we consider an absorbing or emitting atom in the planet as the source
            #      see gen_specdopshift():
            # w_receiver = w_source * (1+ rv[s/r]/c))
            # w_star = w_pl * (1+ (rv[atom/pl]/c)) * (1+ (rv[pl/star]/c))    
            plrange_star[pl_loc] = np.vstack((np.repeat(1e10,data_vis['n_in_visit']),np.repeat(-1e10,data_vis['n_in_visit'])))
            plrange_star[pl_loc][:,iexp_no_plrange] = data_dic['Atm']['plrange'][:,None] + coord_vis[pl_loc]['rv_pl'][iexp_no_plrange] 
            if (True in np.isnan(coord_vis[pl_loc]['rv_pl'])):stop('  Run gen_dic["calc_proc_data"] again to calculate "rv_pl"')
            plAtm_vis['exclu_range_star']['CCF'][pl_loc] = plrange_star[pl_loc]
            
            #Planetary range shifted from planet (source) to solar (receiver) barycentric rest frame
            # w_solbar = w_star * (1+ rv[star/starbar]/c)) * (1+ rv[starbar/solbar]/c))    
            #          = w_pl * (1+ (rv[atom/pl]/c)) * (1+ (rv[pl/star]/c)) * (1+ (rv[star/starbar]/c)) * (1+ (rv[starbar/solbar]/c))
            plAtm_vis['exclu_range_input']['CCF'][pl_loc] =  plrange_star[pl_loc] + coord_vis['RV_star_solCDM'][None,:]
            
            #Shifted spectral ranges
            if ('spec' in data_dic[inst]['type']):
                plAtm_vis['exclu_range_star']['spec'][pl_loc] = np.zeros([len(data_dic['Atm']['CCF_mask_wav']),2,data_vis['n_in_visit']])*np.nan
                plAtm_vis['exclu_range_star']['spec'][pl_loc][:,:,iexp_no_plrange] = data_dic['Atm']['CCF_mask_wav'][:,None,None]*gen_specdopshift(data_dic['Atm']['plrange'])[:,None]*gen_specdopshift(coord_vis[pl_loc]['rv_pl'][iexp_no_plrange],v_s = coord_vis[pl_loc]['v_pl'][iexp_no_plrange])  

                plAtm_vis['exclu_range_input']['spec'][pl_loc] = np.zeros([len(data_dic['Atm']['CCF_mask_wav']),2,data_vis['n_in_visit']])*np.nan
                plAtm_vis['exclu_range_input']['spec'][pl_loc][:,:,iexp_no_plrange] = plAtm_vis['exclu_range_star']['spec'][pl_loc][:,:,iexp_no_plrange]*gen_specdopshift(coord_vis['RV_star_stelCDM'][iexp_no_plrange])*gen_specdopshift(data_dic['DI']['sysvel'][inst][vis])    
                
    return None


def update_inst(data_dic,inst,gen_dic):
    r"""**Update: instrument**

    Updates instrument-specific fields for the workflow, once all visits have been processed.
    
    Args:
        TBD
    
    Returns:
        None
    
    """ 
    data_inst = data_dic[inst] 
    
    #Common table for the reference visit
    #    - after this stage, all modules call the common table in the star rest frame, so we update this one while keeping the original one for plotting purposes
    #      if profiles were not aligned the common visit table below points toward the input one
    data_com_ref = dataload_npz(data_inst[data_inst['com_vis']]['proc_com_star_data_paths']) 

    #Check alignment status
    if gen_dic['align_DI']:com_star='_star'
    else:com_star=''

    #Defining instrument table
    #    - we define a specific path to prevent overwriting the original common grid
    if data_inst['comm_sp_tab'] and (gen_dic['align_DI']) and (data_dic['DI']['type'][inst]=='CCF'):
        data_inst['proc_com_star_data_path'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_com_up'+com_star  
    elif gen_dic['spec_1D'] or gen_dic['CCF_from_sp']:
        if gen_dic['spec_1D']:
            data_inst['type']='spec1D'     
            data_inst['proc_com_star_data_path'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_com_up'+com_star 
        elif gen_dic['CCF_from_sp']:
            data_inst['type']='CCF'                    
            data_inst['proc_com_star_data_path'] = gen_dic['save_data_dir']+'Processed_data/CCFfromSpec/'+inst+'_com_up'+com_star 
            if ('chrom' in data_inst['system_prop']):
                data_inst['system_prop']['chrom_mode'] = 'achrom'
                data_inst['system_prop'].pop('chrom')
    else:
        data_inst['proc_com_star_data_path'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_com_up'+com_star 
        
    #Common instrument table is set to the table of the visit taken as reference
    datasave_npz(data_inst['proc_com_star_data_path'],data_com_ref)  

    #Updating dimensions
    if gen_dic['spec_1D'] or gen_dic['CCF_from_sp']:    
        data_inst['comm_sp_tab']=True
        data_inst['dim_exp'] = deepcopy(data_com_ref['dim_exp'])
        data_inst['nspec'] = deepcopy(data_com_ref['nspec'])
        data_inst['nord'] = 1
        
    return None


def update_gen(data_dic,gen_dic):
    r"""**Update: generic**

    Updates generic fields for the workflow, once all instruments have been processed.
    
    Args:
        TBD
    
    Returns:
        None
    
    """     
    if gen_dic['CCF_from_sp']:
        if ('chrom' in data_dic['DI']['system_prop']):
            data_dic['DI']['system_prop']['chrom_mode'] = 'achrom'
            data_dic['DI']['system_prop'].pop('chrom')
 
    return None
