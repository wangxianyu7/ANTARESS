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
from ..ANTARESS_grids.ANTARESS_occ_grid import occ_region_grid,sub_calc_plocc_spot_prop,calc_plocc_spot_prop
from ..ANTARESS_grids.ANTARESS_prof_grid import init_custom_DI_prof,custom_DI_prof,theo_intr2loc,gen_theo_atm
from ..ANTARESS_grids.ANTARESS_coord import calc_mean_anom_TR,calc_Kstar,calc_tr_contacts,calc_rv_star,coord_expos
from ..ANTARESS_analysis.ANTARESS_inst_resp import return_pix_size,def_st_prof_tab,cond_conv_st_prof_tab,conv_st_prof_tab,get_FWHM_inst,resamp_st_prof_tab
from ..ANTARESS_analysis.ANTARESS_ana_comm import par_formatting
from ..ANTARESS_corrections.ANTARESS_sp_reduc import red_sp_data_instru
from ..ANTARESS_analysis.ANTARESS_joined_star import joined_Star_ana
from ..ANTARESS_analysis.ANTARESS_joined_atm import joined_Atm_ana
from ..ANTARESS_plots.ANTARESS_plots_all import ANTARESS_plot_functions
from ..ANTARESS_corrections.ANTARESS_calib import calc_gcal
from ..ANTARESS_process.ANTARESS_plocc_spec import def_in_plocc_profiles,def_diff_profiles
from ..ANTARESS_conversions.ANTARESS_masks_gen import def_masks
from ..ANTARESS_conversions.ANTARESS_conv import CCF_from_spec,ResIntr_CCF_from_spec,conv_2D_to_1D_spec
from ..ANTARESS_conversions.ANTARESS_binning import process_bin_prof
from ..ANTARESS_corrections.ANTARESS_detrend import detrend_prof,pc_analysis
from ..ANTARESS_process.ANTARESS_data_process import align_profiles,rescale_profiles,extract_res_profiles,extract_intr_profiles,extract_pl_profiles 
from ..ANTARESS_analysis.ANTARESS_ana_comm import MAIN_single_anaprof
from ..ANTARESS_conversions.ANTARESS_sp_cont import process_spectral_cont
from ..ANTARESS_general.utils import air_index,dataload_npz,gen_specdopshift,stop,np_where1D,closest,datasave_npz,def_edge_tab,check_data
from ..ANTARESS_general.constant_data import Rsun,Rjup,c_light,G_usi,Msun,AU_1


def ANTARESS_settings_overwrite(gen_dic,plot_dic,corr_spot_dic,data_dic,mock_dic,theo_dic,glob_fit_dic,detrend_prof_dic,input_dic):
    r"""**ANTARESS settings overwrite.**
    
    Overwrites ANTARESS settings with inputs.  
    
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
        for key in ['IntrProf','IntrProp']:
            if key in input_dic['settings']['glob_fit_dic']:glob_fit_dic[key].update(input_dic['settings']['glob_fit_dic'][key])
    if 'plot_dic' in input_dic['settings']:plot_dic.update(input_dic['settings']['plot_dic'])
    
    #Overwriting specific fields
    if ('orders4ccf' in input_dic) and (len(input_dic['orders4ccf'])>0):
        for inst in input_dic['orders4ccf']:gen_dic['orders4ccf'][inst] = input_dic['orders4ccf'][inst]
    
    return None
    





def ANTARESS_main(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic, corr_spot_dic,system_param,input_dic,custom_plot_settings):
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
    for inst in data_dic['instrum_list']:
        print('')
        print('-----------------------')
        print('Processing instrument :',inst)        
        print('-----------------------')

        #Initialize instrument tables and dictionaries
        init_inst(mock_dic,inst,gen_dic,data_dic,theo_dic,data_prop,coord_dic,system_param,plot_dic)

        #Estimating instrumental calibration
        if gen_dic['gcal']:
            calc_gcal(gen_dic,data_dic,inst,plot_dic,coord_dic)

        #Global corrections of spectral data
        #    - performed before the loop on individual visits because some corrections exploit information from all visits and require the full range of the data
        if ('spec' in data_dic[inst]['type']):
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

            #Converting DI stellar spectra into CCFs
            if gen_dic[data_type_gen+'_CCF']:
                CCF_from_spec(data_type_gen,inst,vis,data_dic,gen_dic,data_dic[data_type_gen])
        
            #Single line detrending    
            if gen_dic['detrend_prof'] and (not detrend_prof_dic['full_spec']):
                detrend_prof(detrend_prof_dic,data_dic,coord_dic,inst,vis,data_dic,data_prop,gen_dic,plot_dic)

            #Calculating theoretical properties of the planet-occulted and/or spotted regions 
            if gen_dic['theoPlOcc'] or (data_dic['DI']['spots_prop'] != {}):
                calc_plocc_spot_prop(system_param,gen_dic,theo_dic,coord_dic,inst,vis,data_dic,calc_pl_atm=gen_dic['calc_pl_atm'],spot_dic=data_dic['DI']['spots_prop'])
                
            #Analyzing original disk-integrated profiles
            if gen_dic['fit_'+data_type_gen]:
                MAIN_single_anaprof('',data_type_gen+'orig',data_dic,gen_dic,inst,vis,coord_dic,theo_dic,plot_dic,system_param['star'])

            #Aligning disk-integrated profiles to star rest frame
            if (gen_dic['align_'+data_type_gen]):
                align_profiles(data_type_gen,data_dic,inst,vis,gen_dic,coord_dic)

            # #Correcting for spot contamination 
            # if gen_dic['correct_spots'] : 
            #     corr_spot(corr_spot_dic, coord_dic,inst,vis,data_dic,data_prop,gen_dic, theo_dic, system_param)
              
            #Rescaling profiles to their correct flux level                  
            if gen_dic['flux_sc']:                   
                rescale_profiles(data_dic[inst],inst,vis,data_dic,coord_dic,coord_dic[inst][vis]['t_dur_d'],gen_dic,plot_dic,system_param,theo_dic)   
         
            #Calculating master spectrum of the disk-integrated star used in weighted averages and continuum-normalization
            if gen_dic['DImast_weight']:              
                process_bin_prof('',data_type_gen,gen_dic,inst,vis,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,masterDI=True,spot_dic=data_dic['DI']['spots_prop'])

            #Processing converted 2D disk-integrated profiles
            if gen_dic['spec_1D']:                
                conv_2D_to_1D_gen_functions(data_type_gen,data_dic,inst,vis,gen_dic,coord_dic,theo_dic,plot_dic,system_param)
 
            #Processing binned disk-integrated profiles
            if gen_dic['bin']:
                bin_gen_functions(data_type_gen,'',inst,gen_dic,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,vis=vis)

            #--------------------------------------------------------------------------------------------------
            #Processing residual and intrinsic stellar profiles
            data_type_gen = 'Intr'
            #--------------------------------------------------------------------------------------------------

            #Extracting residual profiles
            if (gen_dic['res_data']):
                extract_res_profiles(gen_dic,data_dic,inst,vis,data_prop,coord_dic)

            #Extracting intrinsic stellar profiles
            if gen_dic['intr_data']:
                extract_intr_profiles(data_dic,gen_dic,inst,vis,system_param['star'],coord_dic,theo_dic,plot_dic)
        
            #Converting out-of-transit residual and intrinsic spectra into CCFs
            if gen_dic[data_type_gen+'_CCF']:
                ResIntr_CCF_from_spec(inst,vis,data_dic,gen_dic)
                  
            #Applying PCA to out-of transit residual profiles
            if (gen_dic['pca_ana']):
                pc_analysis(gen_dic,data_dic,inst,vis,data_prop,coord_dic)

            #Fitting intrinsic stellar profiles in the star rest frame
            if gen_dic['fit_'+data_type_gen]:
                MAIN_single_anaprof('',data_type_gen+'orig',data_dic,gen_dic,inst,vis,coord_dic,theo_dic,plot_dic,system_param['star'])
            
            #Aligning intrinsic stellar profiles to their local rest frame
            if gen_dic['align_'+data_type_gen]: 
                align_profiles(data_type_gen,data_dic,inst,vis,gen_dic,coord_dic)

            #Processing converted 2D intrinsic and residual profiles
            if gen_dic['spec_1D']:                
                conv_2D_to_1D_gen_functions(data_type_gen,data_dic,inst,vis,gen_dic,coord_dic,theo_dic,plot_dic,system_param)

            #Processing binned intrinsic profiles
            if gen_dic['bin']:
                bin_gen_functions(data_type_gen,'',inst,gen_dic,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,vis=vis)

            #Building estimates for planet-occulted stellar profiles in in-transit exposures
            if gen_dic['loc_data_corr']:
                def_in_plocc_profiles(inst,vis,gen_dic,data_dic,data_prop,coord_dic,system_param,theo_dic,glob_fit_dic,plot_dic)

            #Building estimates for differential profiles in all exposures
            #    - in-transit profiles include planet-occulted and spotted stellar profiles
            if gen_dic['diff_data_corr']:
                def_diff_profiles(inst,vis,gen_dic,data_dic,data_prop,coord_dic,system_param,theo_dic,glob_fit_dic,plot_dic)
        
            #--------------------------------------------------------------------------------------------------
            #Processing atmospheric profiles
            data_type_gen = 'Atm'
            #--------------------------------------------------------------------------------------------------

            #Extracting atmospheric profiles
            if gen_dic['pl_atm']:
                extract_pl_profiles(data_dic,inst,vis,gen_dic)

            #Converting atmospheric spectra into CCFs
            if gen_dic[data_type_gen+'_CCF'] and ('spec' in data_dic[inst][vis]['type']):
                CCF_from_spec(data_type_gen,inst,vis,data_dic,gen_dic,data_dic[data_type_gen])

            #Fitting atmospheric profiles in the star rest frame
            if gen_dic['fit_'+data_type_gen]:
                MAIN_single_anaprof('',data_type_gen+'orig',data_dic,gen_dic,inst,vis,coord_dic,theo_dic,plot_dic,system_param['star'])
      
            #Aligning atmospheric profiles to the planet rest frame
            if gen_dic['align_'+data_type_gen]:   
                align_profiles(data_type_gen,data_dic,inst,vis,gen_dic,coord_dic)      

            #Processing converted 2D intrinsic profiles
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
    #Call to analysis function over combined visits and instruments
    ####################################################################################################################
    if gen_dic['multi_inst']:
        print('-------------------------------')
        print('Processing combined instruments')        
        print('-------------------------------')
        
        #Wrap-up function to fit stellar profiles and their properties
        if gen_dic['fit_DIProp'] or gen_dic['fit_IntrProf'] or gen_dic['fit_IntrProp'] or gen_dic['fit_ResProf'] :
            joined_Star_ana(glob_fit_dic,system_param,theo_dic,data_dic,gen_dic,plot_dic,coord_dic)
    
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
        conv_2D_to_1D_spec(data_type_gen,inst,vis,gen_dic,data_dic,data_dic[data_type_gen])  

    #Analyzing converted profiles
    if gen_dic['fit_'+data_type_gen+'_1D']: 
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
        process_bin_prof(mode,data_type_gen,gen_dic,inst,vis,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,mock_dic=mock_dic)

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
    
    #Multi-threading
    print('Multi-threading: '+str(cpu_count())+' threads available')
  
    #Positional dictionary
    coord_dic={}
    
    #Instrument root names
    gen_dic['inst_root']={
        'SOPHIE_HE':'SOPHIE',
        'SOPHIE_HR':'SOPHIE',
        'CORALIE':'CORALIE',        
        'HARPN':'HARPN',
        'HARPS':'HARPS',
        'ESPRESSO':'ESPRESSO',
        'ESPRESSO_MR':'ESPRESSO',
        'CARMENES_VIS':'CARMENES',
        'EXPRES':'EXPRES',
        'NIRPS_HA':'NIRPS',
        'NIRPS_HE':'NIRPS',
    }     

    #Number of physical orders per instrument
    gen_dic['norders_instru']={
        'SOPHIE_HE':39,
        'SOPHIE_HR':39,
        'CORALIE':69,           
        'HARPN':69,
        'HARPS':71,
        'ESPRESSO':170,
        'ESPRESSO_MR':85,
        'CARMENES_VIS':61,
        'EXPRES':86,
        'NIRPS_HA':71,
        'NIRPS_HE':71,
    } 

    #Return flag that errors on input spectra are defined or not for each instrument   
    gen_dic['flag_err_inst']={          
        'SOPHIE_HE':False,
        'SOPHIE_HR':False,
        'CORALIE':False,         
        'HARPN':True   ,
        'HARPS':True,
        'ESPRESSO':True,
        'ESPRESSO_MR':True,
        'CARMENES_VIS':True,
        'EXPRES':True,
        'NIRPS_HA':True,'NIRPS_HE':True} 
    
    #Central wavelengths of orders for known instruments
    gen_dic['wav_ord_inst']={     
        'ESPRESSO':10.*np.repeat(np.flip([784.45 ,774.52, 764.84, 755.40, 746.19, 737.19, 728.42 ,719.85, 711.48, 703.30, 695.31, 687.50, 679.86, 672.39, 665.08,
                                      657.93 ,650.93 ,644.08, 637.37, 630.80, 624.36, 618.05, 611.87, 605.81, 599.87, 594.05, 588.34, 582.74, 577.24, 571.84,
                                      566.55, 561.35, 556.25, 551.24, 546.31, 541.48, 536.73, 532.06, 527.48, 522.97, 522.97, 518.54, 514.18, 509.89, 505.68,
                                      501.53, 497.46, 493.44, 489.50 ,485.61, 481.79 ,478.02, 474.32, 470.67, 467.08, 463.54 ,460.05, 456.62, 453.24, 449.91,
                                      446.62, 443.39, 440.20, 437.05, 433.95 ,430.90, 427.88 ,424.91, 421.98, 419.09, 416.24, 413.43, 410.65, 407.91, 405.21,
                                      402.55, 399.92, 397.32, 394.76, 392.23, 389.73, 387.26, 384.83, 382.42, 380.04]),2),
        'HARPN':np.array([3896.9385, 3921.912 , 3947.2075, 3972.8315, 3998.7905, 4025.0913,4051.7397, 4078.7437, 4106.11  , 4133.8457, 4161.9595, 4190.458 ,
                          4219.349 , 4248.641 , 4278.3433, 4308.464 , 4339.011 , 4369.9946,4401.424 , 4433.3086, 4465.6587, 4498.484 , 4531.796 , 4565.6045,
                          4599.922 , 4634.759 , 4670.127 , 4706.0396, 4742.509 , 4779.548 ,4817.169 , 4855.388 , 4894.2188, 4933.675 , 4973.773 , 5014.5273,
                          5055.955 , 5098.074 , 5140.9004, 5184.452 , 5228.747 , 5273.8066,5319.6494, 5366.297 , 5413.7686, 5462.088 , 5511.2773, 5561.3613,
                          5612.3633, 5664.311 , 5717.2285, 5771.1436, 5826.086 , 5882.084 ,5939.1685, 5997.3726, 6056.7285, 6117.271 , 6179.0366, 6242.062 ,
                          6306.3867, 6372.0503, 6439.0957, 6507.568 , 6577.511 , 6648.9746,6722.0083, 6796.6646, 6872.9976]),
        'HARPS': np.array([3824.484, 3848.533, 3872.886, 3897.549, 3922.529, 3947.831, 3973.461,
                            3999.426, 4025.733, 4052.388, 4079.398, 4106.771, 4134.515, 4162.635, 4191.141,
                            4220.039, 4249.338, 4279.048, 4309.175, 4339.73 , 4370.722, 4402.16 , 4434.053,
                            4466.411, 4499.245, 4532.564, 4566.384, 4600.709, 4635.555, 4670.933, 4706.854,
                            4743.333, 4780.382, 4818.015, 4856.243, 4895.084, 4934.552, 4974.66 , 5015.426,
                            5056.866, 5098.997, 5141.835, 5185.398, 5229.708, 5274.78 , 5367.266, 5414.749,
                            5463.079, 5512.279, 5562.374, 5613.388, 5665.346, 5718.276, 5772.203, 5827.157,
                            5883.167, 5940.266, 5998.481, 6057.852, 6118.408, 6180.188, 6243.229, 6307.567,
                            6373.246, 6440.308, 6508.795, 6578.756, 6650.235, 6723.287, 6797.961, 6874.313]),                
        'CARMENES_VIS': np.array([5185.683800259293,5230.002514675239,5275.085301397565,5320.952091123464,5367.623513832012,
                                  5415.120929724138,5463.466461819996,5512.683030318242,5562.794388829332,5613.825162603174,5665.800888880403,5718.748059506192,5772.694165955971,
                                  5827.667746933892,5883.6984387171215,5940.817028432618,5999.055510467573,6058.447146230714,6119.026527498995,6180.829643603155,6243.893952726336,
                                  6308.2584576125455,6373.9637860064695,6441.052276173239,6509.5680678764,6579.557199224864,6651.067709835355,6724.149750796099,6798.855701960672,
                                  6875.24029714847,6953.36075788067,7033.276936338339,7115.051468293246,7198.749936832545,7284.4410477767,7372.196817776685,7462.09277617269,
                                  7554.208181803426,7648.626256074004,7745.4344337228085,7844.724632875495,7946.5935461392,8051.142954674568,8158.480067389824,8268.717887632938,
                                  8381.975610018162,8498.379050316224,8618.061111667314,8741.162290748554,8867.83122794844,8998.225306077602,9132.51130268578,9270.866101669406,
                                  9413.477470553611,9560.54491063029,9712.280588045605,9868.910354974394,10030.674871216968,10197.83083793165,10370.652356804127,10549.432429789025]),    
        'NIRPS_HA': np.array([   9793.31830725 , 9859.95301593 , 9927.49931744 , 9995.97622395,10065.40380975, 10135.80171767, 10207.19048503, 10279.59121586,
                                10353.02577856, 10427.51660496, 10503.0864558 , 10579.75913144,10657.5589196 , 10736.51096413, 10816.64116324, 10897.97569896,
                                10980.54199941, 11064.36827222, 11149.48372245, 11235.91830631,11323.70300177, 11412.86989695, 11503.45207046, 11595.48314928,
                                11688.99793186, 11784.03276228, 11880.62559736, 11978.81474198,12078.63947148, 12180.1413448 , 12283.36401067, 12388.35148928,
                                12495.14844764, 12603.80175443, 12714.36064267, 12826.87609104,12941.40020952, 13057.98722241, 13176.69424781, 13297.58039703,
                                13420.70477935, 13546.12854527, 13673.91830748, 14072.1783805,14210.13779528, 14350.82710154, 14494.33106043, 14640.73394106,
                                14790.12380228, 14942.59417032, 15098.24089619, 15257.16474828,15419.46907017, 15585.26310369, 15754.66203085, 15927.78376522,
                                16104.75265749, 16285.69620041, 16470.75172016, 16660.06255469,16853.77645383, 17052.04937274, 17255.0418695 , 17462.92163421,
                                17675.87345983, 17894.08476988, 18117.75049316, 18347.07795686,18582.28588236, 18823.6030827 , 19071.27048963]),
        'EXPRES': np.array([3827.18212341, 3851.28596202, 3875.69435186, 3900.41312958 ,3925.44828033,3950.80594266 ,3976.49241363 ,4002.51415404 ,4028.87779393 ,4055.59013823,
                              4082.65817269 ,4110.08906999 ,4137.89019612 ,4166.069117   ,4194.63360538,4223.591648   ,4252.95145307 ,4282.72145806 ,4312.91033776 ,4343.52701277,
                              4374.58065826 ,4406.08071317 ,4438.03688981 ,4470.45918379 ,4503.3578845,4536.74358599 ,4570.62719834 ,4605.01995956 ,4639.93344803 ,4675.37959548,
                              4711.3707006  ,4747.91944324 ,4785.03889938 ,4822.74255663 ,4861.04433066,4899.95858228 ,4939.50013539 ,4979.68429586 ,5020.52687121 ,5062.04419143,
                              5104.2531307  ,5147.17113026 ,5190.81622248 ,5235.20705609 ,5280.3629228,5326.3037852  ,5373.05030624 ,5420.62388022 ,5469.04666543 ,5518.34161856,
                              5568.53253104 ,5619.6440673  ,5671.70180522 ,5724.73227878 ,5778.76302323,5833.82262268 ,5889.94076065 ,5947.14827335 ,6005.47720621 ,6064.96087378,
                              6125.63392315 ,6187.53240127 ,6250.69382638 ,6315.15726382 ,6380.96340664,6448.15466126 ,6516.77523859 ,6586.87125105 ,6658.49081593 ,6731.68416551,
                              6806.50376456 ,6883.00443578 ,6961.24349375 ,7041.2808881  ,7123.17935679,7207.00459002 ,7292.82540601 ,7380.71393943 ,7470.74584357 ,7563.00050759,
                              7657.56128994 ,7754.51576965 ,7853.9560168  ,7955.97888421 ,8060.68632206,8168.18571772])


}
    gen_dic['wav_ord_inst']['NIRPS_HE'] = gen_dic['wav_ord_inst']['NIRPS_HA'] 

    #Data type
    if gen_dic['mock_data']: 
        print('Running with artificial data')  
        data_dic['instrum_list']=list(mock_dic['visit_def'].keys())
        gen_dic['tell_weight']=False    

    else: 
        print('Running with observational data')  
        
        #Instruments to which are associated the different datasets
        data_dic['instrum_list'] = list(gen_dic['data_dir_list'].keys())
        
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
    if gen_dic['mock_data']:gen_dic['use_cov'] = False    
    else:
        #Used by default with spectral datasets, unless CCF datasets are processed or user requests otherwise
        if gen_dic['ccfINtype']:gen_dic['use_cov'] = False  
    if not gen_dic['use_cov']:print('Covariances discounted')        

    #Automatic activation/deactivation
    if gen_dic['pca_ana']:gen_dic['intr_data'] = True
    if gen_dic['intr_data']:
        if not gen_dic['res_data']:
            print('Automatic activation of residual profile extraction')
            gen_dic['res_data'] = True
        if not gen_dic['flux_sc']:
            print('Automatic activation of flux scaling calculation')
            gen_dic['flux_sc'] = True

    else:
        for key in ['map_Intr_prof','all_intr_data','Intr_prof']:plot_dic[key]=''

    #Deactivate conditions
    if (not gen_dic['specINtype']):
        gen_dic['DI_CCF']=False
        detrend_prof_dic['full_spec'] = False
    if (not gen_dic['specINtype']) or (gen_dic['DI_CCF']) or (not gen_dic['intr_data']):        
        gen_dic['Intr_CCF']=False     
        
    #Deactivate all calculation options
    if not gen_dic['calc_all']:
        for key in gen_dic:
            if ('calc_' in key) and (gen_dic[key]):gen_dic[key] = False 

    #Deactivate spectral corrections in CCF mode
    sp_corr_list = ['corr_tell','corr_FbalOrd','corr_Fbal','corr_Ftemp','corr_cosm','mask_permpeak','corr_wig','corr_fring','trim_spec','glob_mast','cal_weight','gcal']
    if (not gen_dic['specINtype']):
        for key in sp_corr_list:gen_dic[key]=False
    else:
        #Activate spectral calibration calculation
        gen_dic['gcal'] = True
        
        #Deactivate spectral balance correction over orders if not 2D spectra
        if ('spec2D' not in gen_dic['all_types']): 
            gen_dic['corr_FbalOrd']=False
            plot_dic['wig_order_fit']=''

        #Activate global master calculation if required for flux balance corrections
        if (gen_dic['corr_Fbal']) or (gen_dic['corr_FbalOrd']):gen_dic['glob_mast']=True
        
        #Deactivate arbitrary scaling if global scaling not required
        if (not gen_dic['corr_Fbal']):gen_dic['Fbal_vis']=None
        
        #Check for arbitrary scaling        
        elif (gen_dic['Fbal_vis']!='theo') and (gen_dic['n_instru']>1):stop('Flux balance must be set to a theoretical reference for multiple instruments')

        #Deactivate transit scaling if temporal flux correction is applied
        if gen_dic['corr_Ftemp']: gen_dic['flux_sc']=False
            
        #Activate correction flag
        gen_dic['corr_data'] = False
        for key in sp_corr_list: gen_dic['corr_data'] |=gen_dic[key]

    #Deactivate modules irrelevant to mock datasets
    if gen_dic['mock_data']:
        for key in ['gcal','corr_tell','glob_mast','corr_Fbal','corr_FbalOrd','corr_Ftemp','corr_cosm','mask_permpeak','corr_wig']:gen_dic[key] = False

    #Deactivate plot if module is not called
    if (plot_dic['glob_mast']!='') and gen_dic['glob_mast']:
        plot_dic['glob_mast']==''
        print('Disabling "glob_mast" plot')
    if (plot_dic['Fbal_corr_vis']!='') and (~gen_dic['corr_Fbal'] or gen_dic['Fbal_vis'] is None):        
        plot_dic['Fbal_corr_vis']==''
        print('Disabling "Fbal_corr_vis" plot')        

    #Set general condition for CCF conversions
    gen_dic['CCF_from_sp'] = gen_dic['DI_CCF'] | gen_dic['Intr_CCF'] | gen_dic['Atm_CCF']
    if gen_dic['CCF_from_sp']:gen_dic['ccfINtype'] = True
        
    #Set general condition for 2D/1D conversion
    gen_dic['spec_1D'] = gen_dic['spec_1D_DI'] | gen_dic['spec_1D_Intr'] | gen_dic['spec_1D_Atm']
    gen_dic['calc_spec_1D'] = gen_dic['calc_spec_1D_DI'] | gen_dic['calc_spec_1D_Intr'] | gen_dic['calc_spec_1D_Atm']

    #Set final data mode for each type of profiles
    #    - the general instrument and visit dictionaries contain a field type that represents the mode of the data at the current stage of the pipeline
    #    - here we set the final mode for each type of profile, so that they can be retrieved in their original mode for plotting
    #      eg, if profiles are converted into CCF at the Residual stage, we keep the information that disk-integrated profiles were in spectral mode
    for key in ['DI','Res','Intr','Atm']:data_dic[key]['type'] = {inst:deepcopy(gen_dic['type'][inst]) for inst in gen_dic['type']}
    for inst in gen_dic['type']:
        if 'spec' in gen_dic['type'][inst]: 
        
            #Set general condition that spectral data is converted into CCFs
            if gen_dic['DI_CCF']:
                data_dic['DI']['type'][inst]='CCF'
            if gen_dic['DI_CCF'] or gen_dic['Intr_CCF']:
                data_dic['Res']['type'][inst]='CCF'          
                data_dic['Intr']['type'][inst]='CCF'
            if gen_dic['DI_CCF'] or gen_dic['Atm_CCF']:
                data_dic['Atm']['type'][inst]='CCF'

            #Set general condition that 2D spectral data is converted into 1D spectral data
            if gen_dic['spec_1D_DI']:
                data_dic['DI']['type'][inst]='spec1D'
            if gen_dic['spec_1D_DI'] or gen_dic['spec_1D_Intr']:
                data_dic['Res']['type'][inst]='spec1D'
                data_dic['Intr']['type'][inst]='spec1D'
            if gen_dic['spec_1D_DI'] or gen_dic['spec_1D_Atm']:
                data_dic['Atm']['type'][inst]='spec1D'

    #Set general condition to activate binning modules
    for mode in ['','multivis']:
        gen_dic['bin'+mode]=False
        gen_dic['calc_bin'+mode]=False
        for data_type_gen in ['DI','Intr','Atm']:
            gen_dic['bin'+mode] |= ( gen_dic[data_type_gen+'bin'+mode] | gen_dic['fit_'+data_type_gen+'bin'+mode])
            gen_dic['calc_bin'+mode] |= ( gen_dic['calc_'+data_type_gen+'bin'+mode] | gen_dic['calc_fit_'+data_type_gen+'bin'+mode])

    #Set general condition for profiles fits
    for key in ['DI','Intr','Atm']:
        gen_dic['fit_'+key+'_gen'] = gen_dic['fit_'+key] | gen_dic['fit_'+key+'bin'] | gen_dic['fit_'+key+'binmultivis'] 

    #Automatic continuum and fit range
    for key in ['DI','Intr','Atm']:
        if gen_dic['fit_'+key+'_gen']:
            if ('cont_range' not in data_dic[key]):data_dic[key]['cont_range']={}
            if ('fit_range' not in data_dic[key]):data_dic[key]['fit_range']={}
    if (gen_dic['res_data']) and ('cont_range' not in data_dic['Res']):data_dic['Res']['cont_range']={}

    #Deactivate conditions
    if gen_dic['DIbin']==False:
        data_dic['DI']['fit_MCCFout']=False
    if (not gen_dic['specINtype']) or (gen_dic['DI_CCF']):plot_dic['spectral_LC']=''
    if (not gen_dic['res_data']):
        for key in ['map_Res_prof','Res_prof']:plot_dic[key]=''    
    if not gen_dic['pl_atm']:
        for key in ['map_Atm_prof','sp_atm','CCFatm']:plot_dic[key]=''  
    else:
        if gen_dic['Intr_CCF']:stop('Atmospheric extraction cannot be performed after Res./Intr. CCF conversion')
    if gen_dic['Intr_CCF'] and (gen_dic['pl_atm']) and (any('spec' in s for s in data_dic['Atm']['type'].values())) and (data_dic['Intr']['opt_loc_data_corr']['corr_mode'] in ['Intrbin','rec_prof']):stop('Intrinsic profiles cannot be converted into CCFs if also requested for planetary spectra extraction)')
    
    #Check for active plots
    if gen_dic['plots_on']:
        gen_dic['plots_on'] = False
        for key in plot_dic:
            if plot_dic[key] in ['png','pdf','jpg']:
                gen_dic['plots_on'] = True
                break
    
    #Telluric condition
    if (not gen_dic['specINtype']):
        gen_dic['tell_weight']=False
        gen_dic['corr_tell']=False
    else:
        gen_dic['tell_weight'] &= gen_dic['corr_tell']
    
    #Set general condition to calculate master spectrum of the disk-integrated star and use it in weighted averages
    #    - the master needs to be calculated if weighing is needed for one of the modules below
    if gen_dic['DImast_weight']:gen_dic['DImast_weight'] |= (gen_dic['res_data'] | (gen_dic['loc_data_corr'] &  (data_dic['Intr']['opt_loc_data_corr']['corr_mode'] in ['DIbin','Intrbin'])) | gen_dic['spec_1D'] | gen_dic['bin'] | gen_dic['binmultivis'])
    if gen_dic['DImast_weight'] and gen_dic['calc_DImast']:gen_dic['calc_DImast'] =  gen_dic['calc_res_data'] | (gen_dic['calc_loc_data_corr'] &  (data_dic['Intr']['opt_loc_data_corr']['corr_mode'] in ['DIbin','Intrbin'])) | gen_dic['calc_spec_1D'] | gen_dic['calc_bin'] | gen_dic['calc_binmultivis']
  
    #Set general conditions to activate multi-instrument modules     # Stage Théo 
    if gen_dic['fit_IntrProf'] or gen_dic['fit_IntrProp'] or gen_dic['fit_AtmProf'] or gen_dic['fit_AtmProp'] or gen_dic['fit_ResProf'] : gen_dic['multi_inst']=True
    else:gen_dic['multi_inst']=False

    #Import bin size dictionary
    gen_dic['pix_size_v']=return_pix_size()    

    #Additional properties
    data_prop={}

    #Standard-deviation curves with bin size for the out-of-transit residual CCFs
    #    - defining the maximum size of the binning window, and the binning size for the sliding window (we will average the bins in windows of width bin_size from 1 to 40 (arbitrary))
    if gen_dic['scr_search']:
        gen_dic['scr_srch_max_binwin']=40.     
        gen_dic['scr_srch_nperbins'] = 1.+np.arange(gen_dic['scr_srch_max_binwin'])

    #Default options for instrumental calibration
    if gen_dic['gcal']: 
        for inst in data_dic['instrum_list'] :
            
            #Threshold
            gen_dic['gcal_thresh'][inst] = {'outliers':5.,'global':1e10}            
            
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


    #------------------------------------------------------------------------------
    #Star
    #------------------------------------------------------------------------------
    
    #Conversions and calculations
    star_params=system_param['star']
    star_params['Rstar_km'] = star_params['Rstar']*Rsun
    
    #Spherical star
    if ('f_GD' not in star_params):
        star_params['f_GD']=0.
        star_params['RpoleReq'] = 1.
        
    #Oblate star
    elif star_params['f_GD']>0.:
        print('Star is oblate')
        star_params['RpoleReq']=1.-star_params['f_GD']

    #Stellar equatorial rotation period (d)
    #    - P = 2*pi*R/v
    star_params['Peq'] = (2.*np.pi*star_params['Rstar_km'])/(star_params['veq']*24*3600)

    #Stellar equatorial rotation rate (rad/s)
    #    - om = 2*pi/P = v/R
    star_params['om_eq'] = star_params['veq']/star_params['Rstar_km']

    #Spot Equatorial rotation rate (rad/s)
    if 'veq_spots' in star_params:star_params['om_eq_spots']=star_params['veq_spots']/star_params['Rstar_km']
    else:star_params['om_eq_spots']=star_params['om_eq']

    #No GD
    if ('beta_GD' not in star_params):star_params['beta_GD']=0.
    if ('Tpole' not in star_params):star_params['Tpole']=0.

    #Conversions
    star_params['istar_rad']=star_params['istar']*np.pi/180.
    star_params['cos_istar']=np.cos(star_params['istar_rad'])
    star_params['vsini']=star_params['veq']*np.sin(star_params['istar_rad'])    #km/s
    
    #Default parameters
    for key in ['alpha_rot','beta_rot','alpha_rot_spots','beta_rot_spots','c1_CB','c2_CB','c3_CB','c1_pol','c2_pol','c3_pol','c4_pol']:
        if key not in star_params:star_params[key] = 0.

    #Conversion factor from the LOS velocity output (/Rstar/h) to RV in km/s
    star_params['RV_conv_fact']=-star_params['Rstar_km']/3600.

    #Macroturbulence
    if theo_dic['mac_mode'] is not None:
        if 'rt' in theo_dic['mac_mode']:
            theo_dic['mac_mode_func'] = calc_macro_ker_rt
            if theo_dic['mac_mode']=='rt_iso':
                star_params['A_T']=star_params['A_R']
                star_params['ksi_T']=star_params['ksi_R']   
        elif 'anigauss' in theo_dic['mac_mode']:
            theo_dic['mac_mode_func'] = calc_macro_ker_anigauss
            if theo_dic['mac_mode']=='anigauss_iso':
                star_params['eta_T']=star_params['eta_R']

    #------------------------------------------------------------------------------
    #Planets
    #------------------------------------------------------------------------------
    
    #Planets in the system
    gen_dic['all_pl'] = [pl_loc for pl_loc in system_param.keys() if pl_loc!='star']

    #Planets considered for transit
    gen_dic['studied_pl'] = list(gen_dic['transit_pl'].keys()) 
    txt_print = 'Study of: '+gen_dic['studied_pl'][0]
    if len(gen_dic['studied_pl'])>1:
        for pl_loc in gen_dic['studied_pl'][1::]:txt_print+=', '+pl_loc

    #Keplerian motion
    if ('kepl_pl' not in gen_dic):gen_dic['kepl_pl']=['all']
    if (gen_dic['kepl_pl']==['all']):
        print('Accounting for Keplerian motion from all planets')
        gen_dic['kepl_pl']=deepcopy(gen_dic['all_pl'])
    
    #Planet properties    
    for pl_loc in list(set(gen_dic['studied_pl']+gen_dic['kepl_pl'])):
        
        #Checking if there is a "-" in a target name
        if '-' in pl_loc:stop('Invalid target name: {}. Target names should not contain a hyphen.'.format(pl_loc))        
        
        #Automatic definitions and conversions
        PlParam_loc=system_param[pl_loc]
        PlParam_loc['omega_rad']=PlParam_loc['omega_deg']*np.pi/180.
        if ('a' not in PlParam_loc) and ('aRs' in PlParam_loc) and ('Rstar' in star_params):
            PlParam_loc['a']=PlParam_loc['aRs']*star_params['Rstar_km']/AU_1  # in au
        PlParam_loc['period_s'] = PlParam_loc['period']*24.*3600.
        if 'inclination' in PlParam_loc:
            PlParam_loc['inclin_rad']=PlParam_loc['inclination']*np.pi/180.

            #Semi-amplitude of planet orbital motion around the star (approximated with Mp << Mstar) in km/s
            if ('a' in PlParam_loc):
                PlParam_loc['Kp_orb'] = (2.*np.pi/PlParam_loc['period_s'])*np.sin(PlParam_loc['inclin_rad'])*PlParam_loc['a']*AU_1/np.sqrt(1.-PlParam_loc['ecc']**2.)

                #Stellar mass derived from semi-orbital motion around the star
                #    - assuming Mp << Ms : 
                # P^2/a^3 = 4*pi^2/G*Ms 
                # Ms = 4*pi^2*a^3/(G*P^2)
                PlParam_loc['Mstar_orb'] = 4.*np.pi**2.*(PlParam_loc['a']*AU_1*1e3)**3./(Msun*G_usi*PlParam_loc['period_s']**2.)

        if pl_loc in gen_dic['studied_pl']:
            PlParam_loc['lambda_rad']=PlParam_loc['lambda_proj']*np.pi/180.

            PlParam_loc['b']=PlParam_loc['aRs']*np.cos(PlParam_loc['inclin_rad'])   
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
  
        #Keplerian semi-amplitude from the studied planet (km/s) 
        PlParam_loc['Kstar_kms']=PlParam_loc['Kstar']/1000. if 'Kstar' in PlParam_loc else calc_Kstar(PlParam_loc,star_params)/1000.

        #Orbital frequency, in year-1
        PlParam_loc['omega_p']=2.*np.pi*365.2425/PlParam_loc['period']

        #Transit duration
        if (pl_loc in gen_dic['studied_pl']) and ('TLength' not in PlParam_loc):
            contact_phases=calc_tr_contacts(data_dic['DI']['system_prop']['achrom'][pl_loc][0],PlParam_loc,plot_dic['stend_ph'],star_params)
            PlParam_loc['TLength'] = (contact_phases[3]-contact_phases[0])*PlParam_loc['period']           
            print('Automatic definition of T14['+str(pl_loc)+']='+"{0:.2f}".format(PlParam_loc['TLength']*24.)+' h')

    #Calculating theoretical properties of the planet-occulted regions
    if (not gen_dic['theoPlOcc']) and ((('CCF' in data_dic['Res']['type'].values()) and (gen_dic['fit_Intr'])) or (gen_dic['align_Intr']) or (gen_dic['calc_pl_atm'])):
        gen_dic['theoPlOcc']=True

    #Oversampling factor for values from planet-occulted regions
    #    - uses the nominal planet-to-star radius ratios, which must correspond to the band from which local properties are derived
    theo_dic['d_oversamp']={}
    for pl_loc in theo_dic['n_oversamp']:
        if (theo_dic['n_oversamp'][pl_loc]>0.):
            theo_dic['d_oversamp'][pl_loc] = data_dic['DI']['system_prop']['achrom'][pl_loc][0]/theo_dic['n_oversamp'][pl_loc]
         
    #Set flag for errors on estimates for local stellar profiles (depending on whether they are derived from data or models)
    if data_dic['Intr']['opt_loc_data_corr']['corr_mode'] in ['DIbin','Intrbin']:data_dic['Intr']['cov_loc_star']=True
    else:data_dic['Intr']['cov_loc_star']=False    

    #Transit and stellar surfce chromatic properties
    #    - must be defined for data processing, transit scaling, calculation of planet properties
    #    - we remove the 'chrom' dictionary if no spectral data is used as input or if a single band is defined
    for ideg in range(2,5):
        if 'LD_u'+str(ideg) not in data_dic['DI']['system_prop']['achrom']:data_dic['DI']['system_prop']['achrom']['LD_u'+str(ideg)] = [0.]
    if ('GD_dw' in data_dic['DI']['system_prop']['achrom']):
        star_params['GD']=True
        print('Star is gravity-darkened')
    else:star_params['GD']=False
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
    for pl_loc in gen_dic['studied_pl']:

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
    #Spots
    #------------------------------------------------------------------------------

    #Initialize spot use
    gen_dic['studied_sp'] = list(gen_dic['transit_sp'].keys()) 
    theo_dic['x_st_sky_grid_sp']={}
    theo_dic['y_st_sky_grid_sp']={}
    theo_dic['Ssub_Sstar_sp'] = {}
    theo_dic['d_oversamp_spot']={}

    #If spot activation has been triggered
    if (data_dic['DI']['spots_prop'] != {}):

        #Oversampling factor for spot-occulted regions
        #    - use the spot radius provided as input
        for spot in theo_dic['n_oversamp_spot']:
            if (theo_dic['n_oversamp_spot'][spot]>0.):
                theo_dic['d_oversamp_spot'][spot] = np.sin(data_dic['DI']['spots_prop']['achrom'][spot][0])/theo_dic['n_oversamp_spot'][spot]
    
        #Spot surface chromatic properties
        for ideg in range(2,5):
            if 'LD_u'+str(ideg) not in data_dic['DI']['spots_prop']['achrom']:data_dic['DI']['spots_prop']['achrom']['LD_u'+str(ideg)] = [0.]
    
        #Need to define chromatic band properties
        data_dic['DI']['spots_prop']['achrom']['w']=[None]
        data_dic['DI']['spots_prop']['achrom']['nw']=1
        data_dic['DI']['spots_prop']['chrom_mode'] = 'achrom'
        if ('chrom' in data_dic['DI']['spots_prop']):
            if (not gen_dic['specINtype']) or (len(data_dic['DI']['spots_prop']['chrom']['w'])==1):data_dic['DI']['spots_prop'].pop('chrom')
            else:
                data_dic['DI']['spots_prop']['chrom_mode'] = 'chrom'
                data_dic['DI']['spots_prop']['chrom']['w'] = np.array(data_dic['DI']['spots_prop']['chrom']['w'])
                data_dic['DI']['spots_prop']['chrom']['nw']=len(data_dic['DI']['spots_prop']['chrom']['w'])
                
                #Typical scale of chromatic variations
                w_edge = def_edge_tab(data_dic['DI']['spots_prop']['chrom']['w'][None,:][None,:])[0,0]    
                data_dic['DI']['spots_prop']['chrom']['dw'] = w_edge[1::]-w_edge[0:-1]
                data_dic['DI']['spots_prop']['chrom']['med_dw'] = np.median(data_dic['DI']['spots_prop']['chrom']['dw'])
    
        #Definition of grids discretizing planets disk to calculate planet-occulted properties
        for spot in theo_dic['nsub_Dspot']:
            
            #Retrieve spot size
            spot_size = data_dic['DI']['spots_prop']['achrom'][spot][0]
    
            #Define a default grid size if the spot grid hasn't been defined (should be done outside of for loop)
            _,theo_dic['Ssub_Sstar_sp'][spot],theo_dic['x_st_sky_grid_sp'][spot], theo_dic['y_st_sky_grid_sp'][spot],_ = occ_region_grid(spot_size, theo_dic['nsub_Dspot'][spot],spot=True)

    #------------------------------------------------------------------------------------------------------------------------
    #Model star
    #------------------------------------------------------------------------------------------------------------------------
    grid_type=[]
    if any('spec' in s for s in data_dic['DI']['type'].values()):grid_type+=['spec']
    if ('CCF' in data_dic['DI']['type'].values()):grid_type+=['ccf']    

    #Calculation of total stellar flux for use in simulated light curves
    if gen_dic['calc_flux_sc'] and (data_dic['DI']['transit_prop']['nsub_Dstar'] is not None): 
        model_star('Ftot',theo_dic,grid_type,data_dic['DI']['system_prop'],data_dic['DI']['transit_prop']['nsub_Dstar'],star_params) 
                    
    #Definition of model stellar grid to calculate local or disk-integrated properties
    #    - used througout the pipeline, unless stellar properties are fitted
    if gen_dic['theoPlOcc'] or (gen_dic['theo_spots']) or (gen_dic['fit_DI_gen'] and (('custom' in data_dic['DI']['model'].values()) or ('RT_ani_macro' in data_dic['DI']['model'].values()))) or gen_dic['mock_data'] \
        or gen_dic['fit_ResProf'] or gen_dic['correct_spots'] or gen_dic['fit_IntrProf'] or gen_dic['loc_data_corr']:

        #Stellar grid
        model_star('grid',theo_dic,grid_type,data_dic['DI']['system_prop'],theo_dic['nsub_Dstar'],star_params) 
       
        #Theoretical atmosphere
        cond_st_atm = False
        if gen_dic['mock_data']:
            for inst in mock_dic['intr_prof']:
                if (mock_dic['intr_prof'][inst]['mode']=='theo' ):cond_st_atm = True
        if (gen_dic['fit_DI_gen'] and ('custom' in data_dic['DI']['model'].values())):
            for inst in data_dic['DI']['mod_def']:
                if (data_dic['DI']['mod_def'][inst]['mode'] == 'theo'):cond_st_atm = True
        if gen_dic['fit_IntrProf'] and (glob_fit_dic['IntrProf']['mode'] == 'theo'):cond_st_atm = True 
        if gen_dic['loc_data_corr'] and (data_dic['Intr']['opt_loc_data_corr']['corr_mode'] in ['glob_mod','indiv_mod']) and (data_dic['Intr']['opt_loc_data_corr']['mode']=='theo'):cond_st_atm = True  
        if cond_st_atm:
            if theo_dic['st_atm']['calc']:
                theo_dic['sme_grid'] = gen_theo_atm(theo_dic['st_atm'],star_params)
                datasave_npz(gen_dic['save_data_dir']+'Introrig_prop/IntrProf_grid',{'sme_grid':theo_dic['sme_grid']})
            else:theo_dic['sme_grid'] = dataload_npz(gen_dic['save_data_dir']+'Introrig_prop/IntrProf_grid')['sme_grid']

    #------------------------------------------------------------------------------
    #Generic path names
    gen_dic['main_pl_text'] = ''
    for pl_loc in gen_dic['studied_pl']:gen_dic['main_pl_text']+=pl_loc
    gen_dic['save_data_dir'] = gen_dic['save_dir']+gen_dic['main_pl_text']+'_Saved_data/'
    gen_dic['save_plot_dir'] = gen_dic['save_dir']+gen_dic['main_pl_text']+'_Plots/'
    gen_dic['add_txt_path']={'DI':'','Intr':'','Res':'','Atm':data_dic['Atm']['pl_atm_sign']+'/'}
    gen_dic['data_type_gen']={'DI':'DI','Res':'Res','Intr':'Intr','Absorption':'Atm','Emission':'Atm'}
    gen_dic['type_name']={'DI':'disk-integrated','Res':'residual','Intr':'intrinsic','Atm':'atmospheric','Absorption':'absorption','Emission':'emission'}    

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
    #    - weights set to 1 unless requested
    if data_dic['Atm']['CCF_mask'] is not None:
        
        #Upload CCF mask
        ext = data_dic['Atm']['CCF_mask'].split('.')[-1]
        if (ext=='fits'):
            hdulist = fits.open(data_dic['Atm']['CCF_mask'])
            data_loc = hdulist[1].data
            data_dic['Atm']['CCF_mask_wav'] = data_loc['lambda']
            data_dic['Atm']['CCF_mask_wgt'] = data_loc['contrast'] if data_dic['Atm']['use_maskW'] == True else np.repeat(1.,len(data_dic['Atm']['CCF_mask_wav']))        
            hdulist.close()
        elif (ext=='csv'):
            data_loc = np.genfromtxt(data_dic['Atm']['CCF_mask'], delimiter=',', names=True)
            data_dic['Atm']['CCF_mask_wav'] = data_loc['wave']
            data_dic['Atm']['CCF_mask_wgt'] = data_loc['contrast'] if data_dic['Atm']['use_maskW'] == True else np.repeat(1.,len(data_dic['Atm']['CCF_mask_wav'])) 
        elif (ext in ['txt','dat']):
            data_loc = np.loadtxt(data_dic['Atm']['CCF_mask']).T            
            data_dic['Atm']['CCF_mask_wav'] = data_loc[0]
            data_dic['Atm']['CCF_mask_wgt'] = data_loc[1]        
        else:
            stop('CCF mask extension TBD') 

    #------------------------------------------------------------------------------------------------------------------------

    #Create directories where data is saved/restored
    if (not path_exist(gen_dic['save_data_dir']+'Processed_data/Global/')):makedirs(gen_dic['save_data_dir']+'Processed_data/Global/')  
    if gen_dic['specINtype']:
        if gen_dic['gcal'] and (not path_exist(gen_dic['save_data_dir']+'Processed_data/Calibration/')):makedirs(gen_dic['save_data_dir']+'Processed_data/Calibration/')  
        if gen_dic['CCF_from_sp'] and (not path_exist(gen_dic['save_data_dir']+'Processed_data/CCFfromSpec/')):makedirs(gen_dic['save_data_dir']+'Processed_data/CCFfromSpec/')  
        if (gen_dic['corr_data']):
            corr_path = gen_dic['save_data_dir']+'Corr_data/'
            if (not path_exist(corr_path)):makedirs(corr_path)
            if (gen_dic['corr_tell']) and (not path_exist(corr_path+'Tell/')):makedirs(corr_path+'Tell/')         
            if (gen_dic['glob_mast']) and (not path_exist(corr_path+'Global_Master/')):makedirs(corr_path+'Global_Master/')    
            if (gen_dic['corr_Fbal']) and (not path_exist(corr_path+'Fbal/')):makedirs(corr_path+'Fbal/')
            if (gen_dic['corr_FbalOrd']) and (not path_exist(corr_path+'Fbal/Orders/')):makedirs(corr_path+'Fbal/Orders/')            
            if (gen_dic['corr_Ftemp']) and (not path_exist(corr_path+'Ftemp/')):makedirs(corr_path+'Ftemp/')
            if (gen_dic['corr_cosm']) and (not path_exist(corr_path+'Cosm/')):makedirs(corr_path+'Cosm/')
            if (gen_dic['mask_permpeak']) and (not path_exist(corr_path+'Permpeak/')):makedirs(corr_path+'Permpeak/')
            if (gen_dic['corr_wig']): 
                if not path_exist(corr_path+'Wiggles/'):makedirs(corr_path+'Wiggles/')
                
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
    if (gen_dic['res_data']) and (not path_exist(gen_dic['save_data_dir']+'Res_data/')):makedirs(gen_dic['save_data_dir']+'Res_data/')
    if gen_dic['pca_ana'] and (not path_exist(gen_dic['save_data_dir']+'PCA_results/')):makedirs(gen_dic['save_data_dir']+'PCA_results/')   
    if (gen_dic['intr_data']) and (not path_exist(gen_dic['save_data_dir']+'Intr_data/')):makedirs(gen_dic['save_data_dir']+'Intr_data/')
    if gen_dic['loc_data_corr']:
        if (not path_exist(gen_dic['save_data_dir']+'Loc_estimates/')):makedirs(gen_dic['save_data_dir']+'Loc_estimates/')        
        if (not path_exist(gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['opt_loc_data_corr']['corr_mode']+'/')):makedirs(gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['opt_loc_data_corr']['corr_mode']+'/')  
    if gen_dic['diff_data_corr']:
        if (not path_exist(gen_dic['save_data_dir']+'Diff_estimates/')):makedirs(gen_dic['save_data_dir']+'Diff_estimates/')        
        if (not path_exist(gen_dic['save_data_dir']+'Diff_estimates/'+data_dic['Res']['opt_loc_data_corr']['corr_mode']+'/')):makedirs(gen_dic['save_data_dir']+'Diff_estimates/'+data_dic['Res']['opt_loc_data_corr']['corr_mode']+'/')          
    if (gen_dic['pl_atm']):
        if (not path_exist(gen_dic['save_data_dir']+'Atm_data/')):makedirs(gen_dic['save_data_dir']+'Atm_data/')        
        if (not path_exist(gen_dic['save_data_dir']+'Atm_data/'+data_dic['Atm']['pl_atm_sign']+'/')):makedirs(gen_dic['save_data_dir']+'Atm_data/'+data_dic['Atm']['pl_atm_sign']+'/')
    
    for data_type in ['DI','Intr','Atm']:
        if gen_dic['align_'+data_type] and (not path_exist(gen_dic['save_data_dir']+'Aligned_'+data_type+'_data/'+gen_dic['add_txt_path'][data_type])):makedirs(gen_dic['save_data_dir']+'Aligned_'+data_type+'_data/'+gen_dic['add_txt_path'][data_type])    
        if gen_dic['spec_1D_'+data_type]:
            if (not path_exist(gen_dic['save_data_dir']+data_type+'_data/1Dfrom2D/'+gen_dic['add_txt_path'][data_type])):makedirs(gen_dic['save_data_dir']+data_type+'_data/1Dfrom2D/'+gen_dic['add_txt_path'][data_type])   
            if (data_type=='Intr') and (not path_exist(gen_dic['save_data_dir']+'Res_data/1Dfrom2D/'+gen_dic['add_txt_path'][data_type])):makedirs(gen_dic['save_data_dir']+'Res_data/1Dfrom2D/'+gen_dic['add_txt_path'][data_type])  
        if (gen_dic[data_type+'bin'] or gen_dic[data_type+'binmultivis']) and (not path_exist(gen_dic['save_data_dir']+data_type+'bin_data/'+gen_dic['add_txt_path'][data_type])):makedirs(gen_dic['save_data_dir']+data_type+'bin_data/'+gen_dic['add_txt_path'][data_type])
        if (gen_dic['fit_'+data_type+'bin'] or gen_dic['fit_'+data_type+'binmultivis']) and (not path_exist(gen_dic['save_data_dir']+data_type+'bin_prop/'+gen_dic['add_txt_path'][data_type])):makedirs(gen_dic['save_data_dir']+data_type+'bin_prop/'+gen_dic['add_txt_path'][data_type])    
        if gen_dic['def_'+data_type+'masks'] and (not path_exist(gen_dic['save_data_dir']+'CCF_masks_'+data_type+'/'+gen_dic['add_txt_path'][data_type])):makedirs(gen_dic['save_data_dir']+'CCF_masks_'+data_type+'/'+gen_dic['add_txt_path'][data_type])    
        if gen_dic[data_type+'_CCF']:
            if (not path_exist(gen_dic['save_data_dir']+data_type+'_data/CCFfromSpec/'+gen_dic['add_txt_path'][data_type])):makedirs(gen_dic['save_data_dir']+data_type+'_data/CCFfromSpec/'+gen_dic['add_txt_path'][data_type])    
            if (data_type=='Intr') and (not path_exist(gen_dic['save_data_dir']+'Res_data/CCFfromSpec/'+gen_dic['add_txt_path'][data_type])):makedirs(gen_dic['save_data_dir']+'Res_data/CCFfromSpec/'+gen_dic['add_txt_path'][data_type]) 

    if (gen_dic['fit_DI'] or gen_dic['sav_keywords']) and (not path_exist(gen_dic['save_data_dir']+'DIorig_prop/')):makedirs(gen_dic['save_data_dir']+'DIorig_prop/')        
    if ((gen_dic['fit_Intr']) or (gen_dic['theoPlOcc'])) and (not path_exist(gen_dic['save_data_dir']+'Introrig_prop/')):makedirs(gen_dic['save_data_dir']+'Introrig_prop/')
    if (gen_dic['fit_Atm']) and (not path_exist(gen_dic['save_data_dir']+'Atmorig_prop/'+data_dic['Atm']['pl_atm_sign']+'/')):makedirs(gen_dic['save_data_dir']+'Atmorig_prop/'+data_dic['Atm']['pl_atm_sign']+'/')

    for key in ['IntrProp','IntrProf','AtmProf','AtmProp']:
        if (gen_dic['fit_'+key]) and (not path_exist(gen_dic['save_data_dir']+'Joined_fits/'+key+'/')):makedirs(gen_dic['save_data_dir']+'Joined_fits/'+key+'/') 
    
    # # Stage Théo
    # if gen_dic['correct_spots'] and (not path_exist(gen_dic['save_data_dir']+'Spot_corr_DI_data/'+gen_dic['add_txt_path']['DI'])) :
    #     makedirs(gen_dic['save_data_dir']+'Spot_corr_DI_data/'+gen_dic['add_txt_path']['DI'])   

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
    for key_dic in [gen_dic,coord_dic,data_dic['DI'],data_dic,data_dic['Res'],data_dic['Intr'],data_dic['Atm'],data_prop,theo_dic]:key_dic[inst]={} 
    DI_data_inst = data_dic['DI'][inst]
    if (inst not in gen_dic['scr_lgth']):gen_dic['scr_lgth'][inst]={}

    #Facility and reduction id
    facil_inst = {
        'SOPHIE':'OHP',
        'HARPS':'ESO',
        'HARPN':'TNG',   
        'CARMENES_VIS':'CAHA',
        'CORALIE':'ESO','ESPRESSO':'ESO','ESPRESSO_MR':'ESO',
        'EXPRES':'DCT','NIRPS_HE':'ESO','NIRPS_HA':'ESO'
        }[inst]

    #Error definition
    if not gen_dic['mock_data']:
        if (not gen_dic['flag_err_inst'][inst]) and gen_dic['gcal']:stop('Error table must be available to estimate calibration')
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
            gen_dic['CCF_mask_wgt'][inst] = data_loc['contrast']          
            hdulist.close()

        elif (ext=='csv'):
            data_loc = np.genfromtxt(gen_dic['CCF_mask'][inst], delimiter=',', names=True)
            gen_dic['CCF_mask_wav'][inst] = data_loc['wave']
            gen_dic['CCF_mask_wgt'][inst] = data_loc['contrast']  

        elif (ext in ['txt','dat']):
            data_loc = np.loadtxt(gen_dic['CCF_mask'][inst]).T            
            gen_dic['CCF_mask_wav'][inst] = data_loc[0]
            gen_dic['CCF_mask_wgt'][inst] = data_loc[1]       
        else:
            stop('CCF mask extension TBD') 
            
        #Same convention as the ESPRESSO DRS, which takes the 'contrast' column from mask files and use its squares as weights
        #    - masks weights should have been normalized so that mean( weights ) = 1, where weight = CCF_mask_wgt^2
        gen_dic['CCF_mask_wgt'][inst]=gen_dic['CCF_mask_wgt'][inst]**2.

    #Resampling exposures of a given visit on a common table or not
    #    - CCFs will be resampled on a common RV table, as we consider they should only be used for preliminary analysis
    if gen_dic['type'][inst]=='CCF':gen_dic['comm_sp_tab'][inst]=True
    elif (inst not in gen_dic['comm_sp_tab']):gen_dic['comm_sp_tab'][inst]=False
    if (not gen_dic['comm_sp_tab'][inst]):print('   > Data processed on individual spectral tables for each exposure')      
    else:print('   > Data resampled on a common spectral table')



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
        
        #Total number of order for current instrument
        #    - we define an artificial order that contains the CCF or 1D spectrum, so that the pipeline can process in the same way as with 2D spectra
        #    - if orders are fully removed from an instrument its default structure is updated
        #      if orders are trimmed after the spectral reduction the default structure is kept
        #    - mock data created with a single order
        if gen_dic['mock_data']: 
            data_inst['nord'] = 1
            gen_dic[inst]['norders_instru'] = 1
        else:
            data_inst['idx_ord_ref']=deepcopy(np.arange(gen_dic['norders_instru'][inst]))      #to keep track of the original orders
            idx_ord_kept = list(np.arange(gen_dic['norders_instru'][inst]))
            if (data_inst['type'] in ['spec1D','CCF']):
                data_inst['nord'] = 1
                gen_dic[inst]['wav_ord_inst'] = gen_dic['wav_ord_inst'][inst]
                gen_dic[inst]['norders_instru']=gen_dic['norders_instru'][inst]
            elif (data_inst['type']=='spec2D'):
                if inst in gen_dic['del_orders']:
                    idx_ord_kept = list(np.delete(np.arange(gen_dic['norders_instru'][inst]),gen_dic['del_orders'][inst]))
                    gen_dic[inst]['wav_ord_inst'] = gen_dic['wav_ord_inst'][inst][idx_ord_kept]
                    data_inst['idx_ord_ref'] = data_inst['idx_ord_ref'][idx_ord_kept]
                gen_dic[inst]['norders_instru'] = len(idx_ord_kept)
                data_inst['nord']=deepcopy(len(idx_ord_kept))   
        data_inst['nord_spec']=deepcopy(data_inst['nord'])                   #to keep track of the number of orders after spectra are trimmed
        data_inst['nord_ref']=deepcopy(gen_dic[inst]['norders_instru'])      #to keep track of the original number of orders
   
        #Orders contributing to calculation of spectral CCFs
        #    - set automatically in case of S1D to use the generic pipeline structure, even though S1D have no orders
        #    - indexes are made relative to order list after initial selection
        if data_inst['type']=='spec1D':gen_dic[inst]['orders4ccf']=[0]
        elif data_inst['type']=='spec2D':
            if (inst in gen_dic['orders4ccf']): gen_dic[inst]['orders4ccf'] = np.intersect1d(gen_dic['orders4ccf'][inst],idx_ord_kept,return_indices=True)[2]     
            else:gen_dic[inst]['orders4ccf'] = np.arange(gen_dic[inst]['norders_instru'],dtype=int)

        #Telluric spectrum condition
        if ('spec' in data_inst['type']) and gen_dic['tell_weight'] or gen_dic['corr_tell']:data_inst['tell_sp'] = True
        else:data_inst['tell_sp'] = False

        #Calibration condition
        #    - must be calculated even if not needed for weights, but can be deactivated if conversion into CCFs or 2D->1D
        if ('spec' in data_inst['type']):data_inst['mean_gdet'] = True
        else:data_inst['mean_gdet'] = False

        #Initialize flag that exposures in all visits of the instrument share a common spectral table
        data_inst['comm_sp_tab'] = True 
        
        #Calibration settings
        if (inst not in gen_dic['gcal_nooutedge']):gen_dic['gcal_nooutedge'][inst] = [0.,0.]
        
        #Processing each visit
        if gen_dic['mock_data']:vis_list=list(mock_dic['visit_def'][inst].keys())
        else:vis_list=list(gen_dic['data_dir_list'][inst].keys())
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
                
                #Adding / in case user forgets
                vis_path = gen_dic['data_dir_list'][inst][vis]+'/'
        
                #List of all exposures for current instrument
                if inst in ['SOPHIE']:
                    vis_path+= {
                        'CCF':'*ccf*',
                        'spec2D':'*e2ds_',
                        }[data_inst['type']]
                elif inst in ['CORALIE']:
                    vis_path+= {
                        'CCF':'*ccf*',
                        'spec2D':'*S2D_',
                        }[data_inst['type']]
                elif inst in ['CARMENES_VIS']:
                    vis_path+= {
                        'spec2D':'*sci-allr-vis_',
                        }[data_inst['type']]                    
                elif inst in ['ESPRESSO','ESPRESSO_MR','HARPN','HARPS']:
                    vis_path+= {
                        'CCF':'*CCF_',
                        'spec1D':'*S1D_',
                        'spec2D':'*S2D_',
                        }[data_inst['type']] 
                elif inst in ['NIRPS_HA','NIRPS_HE']:
                    vis_path+= {
                        'CCF':'*CCF_TELL_CORR_',       #CCF from telluric-corrected spectra
                        'spec1D':'*S1D_',
                        'spec2D':'*S2D_',
                        }[data_inst['type']] 
                elif inst in ['EXPRES']:
                    vis_path+= {
                        'spec2D':'*',
                        }[data_inst['type']]                      
                else:stop('Instrument undefined')
                
                #Use blazed data
                if (inst in gen_dic['blazed']) and (data_inst['type']=='spec2D'):
                    if (inst in ['ESPRESSO','ESPRESSO_MR','HARPN','HARPS','NIRPS_HA','NIRPS_HE']):vis_path+='BLAZE_'
                    else:stop('Define blaze file names')

                #Use sky-corrected data
                if (inst in gen_dic['fibB_corr']) and (vis in gen_dic['fibB_corr'][inst]) and (len(gen_dic['fibB_corr'][inst][vis])>0):
                    if inst in ['ESPRESSO','ESPRESSO_MR','HARPS','HARPN','NIRPS_HA','NIRPS_HE']:vis_path_skysub=vis_path+'SKYSUB_A'
                    else:vis_path_skysub=vis_path+'C'   
                    vis_path_skysub_exp = np.array(glob.glob(vis_path_skysub+'.fits'))
                    if len(vis_path_skysub_exp)==0:stop('No sky-sub data found. Check path.') 
                    
                    #Orders to be replaced
                    if gen_dic['fibB_corr'][inst][vis]=='all':idx_ord_skysub = 'all'
                    else:idx_ord_skysub = gen_dic['fibB_corr'][inst][vis]
                    
                else:vis_path_skysub_exp = None
              
                #Path of visits exposures
                if inst not in ['EXPRES']:vis_path+='A'
                vis_path_exp = np.array(glob.glob(vis_path+'.fits'))
                n_in_visit=len(vis_path_exp) 
                if n_in_visit==0:stop('No data found. Check path.')

            #Remove/keep visits
            if (vis in gen_dic['unused_visits'][inst]):   
                print('       > Visit '+vis+' is removed')
            else:
                print('         Initializing visit '+vis)
                data_dic[inst]['visit_list']+=[vis]     
                ivis+=1

                #Print visit date if available 
                #    - taking the day at the start of the night, rather than the day at the star of the exposure series
                if (not gen_dic['mock_data']):
                    if (inst in ['CORALIE','ESPRESSO','ESPRESSO_MR','HARPS','HARPN','CARMENES_VIS','EXPRES','NIRPS_HA','NIRPS_HE']):    
                        vis_day_exp=[]
                        vis_hour_exp=[]
                        bjd_exp = []
                        for file_path in vis_path_exp:
                            hdulist =fits.open(file_path)
                            hdr =hdulist[0].header 
                            if inst=='CORALIE':
                                vis_day_exp+= [int(hdr['HIERARCH ESO CORA SHUTTER START DATE'][6:8])]
                                bjd_exp  += [ hdr['HIERARCH ESO DRS BJD'] - 2400000. ]
                                stop('Define time for CORALIE')
                            elif inst in ['ESPRESSO','ESPRESSO_MR','HARPS','HARPN','CARMENES_VIS','NIRPS_HA','NIRPS_HE']:
                                vis_day_exp+= [int(hdr['DATE-OBS'].split('T')[0].split('-')[2]) ]  
                                vis_hour_exp+=[int(hdr['DATE-OBS'].split('T')[1].split(':')[0])]
                                if inst in ['ESPRESSO','ESPRESSO_MR','HARPS','HARPN','NIRPS_HA','NIRPS_HE']:bjd_exp +=[  hdr['HIERARCH '+facil_inst+' QC BJD'] - 2400000. ]
                                elif inst=='CARMENES_VIS':bjd_exp  += [  hdr['CARACAL BJD']  ]                        
                            elif inst=='EXPRES':
                                vis_day_exp+= [int(hdr['DATE-OBS'].split(' ')[0].split('-')[2]) ]  
                                vis_hour_exp+=[int(hdr['DATE-OBS'].split(' ')[1].split(':')[0])]
                        if inst=='CORALIE':
                            vis_yr = hdr['HIERARCH ESO CORA SHUTTER START DATE'][0:4]
                            vis_mt = hdr['HIERARCH ESO CORA SHUTTER START DATE'][4:6]
                        elif inst=='EXPRES':
                            vis_yr = hdr['DATE-OBS'].split(' ')[0].split('-')[0]
                            vis_mt = hdr['DATE-OBS'].split(' ')[0].split('-')[1]                            
                        else:
                            vis_yr = hdr['DATE-OBS'].split('T')[0].split('-')[0]
                            vis_mt = hdr['DATE-OBS'].split('T')[0].split('-')[1]
    
                        #Take the day before if all exposures are past midnight (ie, no exposures between 12 and 23) 
                        vis_day = np.min(vis_day_exp)
                        vis_hour_exp = np.array(vis_hour_exp)
                        bjd_vis = np.mean(bjd_exp)
                        if np.sum((vis_hour_exp>=12) & (vis_hour_exp<=23)  )==0:
                            if vis_day==1:stop('Adapt date retrieval')
                            vis_day-=1
                        vis_day_txt = '0'+str(vis_day) if vis_day<10 else str(vis_day)                    
                        print('           Date :',vis_yr,'/',vis_mt,'/',vis_day_txt)
                
                else:
                    bjd_vis = np.mean(bjd_exp_all) - 2400000.                   
                        
                #Initializing dictionaries for visit
                theo_dic[inst][vis]={}
                data_dic['Atm'][inst][vis]={}   
                data_inst[vis] = {'n_in_visit':n_in_visit,'transit_pl':[],'transit_sp':[],'comm_sp_tab':True} 
                coord_dic[inst][vis] = {}
                for pl_loc in gen_dic['studied_pl']:
                    if (inst in gen_dic['transit_pl'][pl_loc]) and (vis in gen_dic['transit_pl'][pl_loc][inst]):data_inst[vis]['transit_pl']+=[pl_loc]
                for spot in gen_dic['studied_sp']:
                    if (inst in gen_dic['transit_sp'][spot]) and (vis in gen_dic['transit_sp'][spot][inst]):data_inst[vis]['transit_sp']+=[spot]                    
                data_prop[inst][vis] = {}
                data_dic_temp={}
                gen_dic[inst][vis] = {}
                DI_data_inst[vis] = {}  
                         
                #Associating telluric spectrum with each exposure
                if not gen_dic['mock_data']:data_inst[vis]['tell_DI_data_paths']={}
                
                #Initialize current data type and conditions for the visit
                for key in ['type','tell_sp','mean_gdet']:data_inst[vis][key]=deepcopy(data_inst[key])                
                
                #Initialize exposure tables
                #    - spatial positions in x/y in front of the stellar disk
                # + x position along the node line
                #  y position along the projection of the normal to the orbital plane
                # +calculated also for the out-of-transit positions, as in case of exposures binning some of the 
                #      ingress/egress exposures may contribute to the binned exposures average positions            
                #    - phase of in/out exposures, per instrument and per visit   
                #    - radial velocity of planet in star rest frame  
                for key in ['bjd','t_dur','RV_star_solCDM','RV_star_stelCDM']:coord_dic[inst][vis][key] = np.zeros(n_in_visit,dtype=float)*np.nan
                for pl_loc in list(set(gen_dic['studied_pl']+gen_dic['kepl_pl'])):
                    coord_dic[inst][vis][pl_loc]={} 
                    if pl_loc in data_inst[vis]['transit_pl']:
                        for key in ['ecl','cen_ph','st_ph','end_ph','ph_dur','rv_pl','v_pl']:coord_dic[inst][vis][pl_loc][key] = np.zeros(n_in_visit,dtype=float)*np.nan
                        for key in ['cen_pos','st_pos','end_pos']:coord_dic[inst][vis][pl_loc][key] = np.zeros([3,n_in_visit],dtype=float)*np.nan
    
                    #Definition of mid-transit times for each planet associated with the visit 
                    if (pl_loc in gen_dic['Tcenter_visits']) and (inst in gen_dic['Tcenter_visits'][pl_loc]) and (vis in gen_dic['Tcenter_visits'][pl_loc][inst]):
                        coord_dic[inst][vis][pl_loc]['Tcenter'] = gen_dic['Tcenter_visits'][pl_loc][inst][vis]
                    else:
                        norb = round((bjd_vis+2400000.-system_param[pl_loc]['TCenter'])/system_param[pl_loc]["period"])
                        coord_dic[inst][vis][pl_loc]['Tcenter']  = system_param[pl_loc]['TCenter']+norb*system_param[pl_loc]["period"]

                #Observation properties
                for key in ['AM','IWV_AM','TEMP','PRESS','seeing','colcorrmin','colcorrmax','mean_SNR','alt','az','BERV']:data_prop[inst][vis][key] = np.zeros(n_in_visit,dtype=float)*np.nan        
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
                if gen_dic['mock_data']:gen_dic[inst][vis]['flag_err']=False
                else:gen_dic[inst][vis]['flag_err']=deepcopy(gen_dic['flag_err_inst'][inst])     
        
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
                    if dace_red_inst=='ESPRESSO': 
                        if bjd_vis<=58649.5:dace_red_inst+='18'
                        else:dace_red_inst+='19'  
                    elif dace_red_inst=='HARPS': 
                        if bjd_vis<=57167.5:dace_red_inst+='03'
                        else:dace_red_inst+='15'  
                    cond_dace_vis = (np.asarray(DACE_sp['ins_name']) == dace_red_inst) & (np.asarray(DACE_sp['date_night']) == vis_yr+'-'+vis_mt+'-'+vis_day_txt)
                    if True not in cond_dace_vis:
                        print('         Visit not found in DACE')                    
                    else:

                        #Retrieve activity indexes and bjd
                        DACE_idx = {}
                        DACE_idx['bjd'] =  np.array(DACE_sp['rjd'],dtype=float)[cond_dace_vis]
                        data_inst[vis]['act_idx'] = ['ha','na','ca','s','rhk']
                        for key in data_inst[vis]['act_idx']:
                            data_prop[inst][vis][key] = np.zeros([n_in_visit,2],dtype=float)*np.nan
                            if key=='rhk':dace_key = 'rhk'
                            else:dace_key = key+'index'
                            DACE_idx[key] = np.array(DACE_sp[dace_key],dtype=float)[cond_dace_vis]
                            DACE_idx[key+'_err'] = np.array(DACE_sp[dace_key+'_err'],dtype=float)[cond_dace_vis]                    
                else:
                    DACE_sp = False
                    if inst=='EXPRES': 
                        data_inst[vis]['act_idx'] = ['ha','s']
                        for key in data_inst[vis]['act_idx']:data_prop[inst][vis][key] = np.zeros([n_in_visit,2],dtype=float)*np.nan
                    
                #------------------------------------------------------------------------------------------------------------     
   
                #Process all exposures in visit
                for isub_exp,iexp in enumerate(range(n_in_visit)):

                    #Artificial data            
                    if gen_dic['mock_data']:                
                        coord_dic[inst][vis]['bjd'][iexp] = bjd_exp_all[iexp]  - 2400000.  
                        coord_dic[inst][vis]['t_dur'][iexp] = (bjd_exp_high[iexp]-bjd_exp_low[iexp])*24.*3600.
                
                    #Observational data
                    else:
                    
                        #Data of the fits file 
                        hdulist =fits.open(vis_path_exp[iexp])
                        if vis_path_skysub_exp is not None:hdulist_skysub =fits.open(vis_path_skysub_exp[iexp])
                
                        #Header 
                        hdr =hdulist[0].header 

                        #Retrieve exposure bjd in instrument/visit table
                        if inst in ['HARPS','HARPN','ESPRESSO','ESPRESSO_MR','NIRPS_HA','NIRPS_HE']:
                            bjd_exp =  hdr['HIERARCH '+facil_inst+' QC BJD'] - 2400000.      
                        elif inst in ['CORALIE']:bjd_exp =  hdr['HIERARCH ESO DRS BJD'] - 2400000.  
                        elif inst in ['SOPHIE']:bjd_exp =  hdr['HIERARCH OHP DRS BJD'] - 2400000.  
                        elif inst in ['CARMENES_VIS']:bjd_exp =  hdr['CARACAL BJD']   
                        elif inst=='EXPRES':bjd_exp = hdulist[1].header['BARYMJD'] +0.5
                        else:stop('Undefined')
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
                                DI_data_inst[vis]['RVdrift'][iexp]=hdr['HIERARCH '+facil_inst+' QC DRIFT DET0 MEAN']*gen_dic['pix_size_v'][inst]*1e3    #in pix -> km/s
                    
                    #Orbital coordinates for each studied planet
                    for pl_loc in data_inst[vis]['transit_pl']:
                        coord_dic[inst][vis][pl_loc]['cen_pos'][:,iexp],coord_dic[inst][vis][pl_loc]['st_pos'][:,iexp],coord_dic[inst][vis][pl_loc]['end_pos'][:,iexp],coord_dic[inst][vis][pl_loc]['ecl'][iexp],coord_dic[inst][vis][pl_loc]['rv_pl'][iexp],coord_dic[inst][vis][pl_loc]['v_pl'][iexp],\
                        coord_dic[inst][vis][pl_loc]['st_ph'][iexp],coord_dic[inst][vis][pl_loc]['cen_ph'][iexp],coord_dic[inst][vis][pl_loc]['end_ph'][iexp],coord_dic[inst][vis][pl_loc]['ph_dur'][iexp]=coord_expos(pl_loc,coord_dic,inst,vis,system_param['star'],
                                            system_param[pl_loc],coord_dic[inst][vis]['bjd'][iexp],coord_dic[inst][vis]['t_dur'][iexp],data_dic,data_dic['DI']['system_prop']['achrom'][pl_loc][0])                    
                       
                    #--------------------------------------------------------------------------------------------------
        
                    #Initialize data at first exposure
                    if isub_exp==0:
 
                        #Artificial data 
                        if gen_dic['mock_data']: 
                            print('            Building exposures ... ')
                            data_inst[vis]['mock'] = True
                            fixed_args = {}
                            if inst not in mock_dic['intr_prof']:
                                print('           Automatic definition of mock line profile')
                                mock_dic['intr_prof'][inst] = {'mode':'ana','coord_line':'mu','func_prof_name': 'gauss', 'line_trans':None,'mod_prop':{'ctrst_ord0__IS__VS_' : 0.5,'FWHM_ord0__IS__VS_'  : 5. },'pol_mode' : 'modul'}                                             
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
                            params_mock = deepcopy(system_param['star']) 
                            if inst not in mock_dic['flux_cont']:mock_dic['flux_cont'][inst]={}
                            if vis not in mock_dic['flux_cont'][inst]:mock_dic['flux_cont'][inst][vis] = 1.
                            params_mock.update({'rv':0.,'cont':mock_dic['flux_cont'][inst][vis]})  
                            params_mock = par_formatting(params_mock,fixed_args['mod_prop'],None,None,fixed_args,inst,vis,mock_dic['intr_prof'][inst]['mode']) 
             
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
                                'unquiet_star':None,
                                })

                            #Spots properties
                            if (inst in mock_dic['spots_prop']) and (vis in mock_dic['spots_prop'][inst]):
                                params_mock['use_spots']=True
                                par_formatting(params_mock,mock_dic['spots_prop'][inst][vis],None,None,fixed_args,inst,vis) 
                                
                                #Figuring out the number of spots
                                num_spots = 0
                                for par in params_mock:
                                    if 'lat__IS'+inst+'_VS'+vis+'_SP' in par:num_spots+=1
                                params_mock['num_spots']=num_spots
                                params_mock['inst']=inst
                                params_mock['vis']=vis
                            else:params_mock['use_spots']=False 
                                
                        #Observational data            
                        else:   
                            data_inst[vis]['mock'] = False                      
    
                            #VLT UT used by ESPRESSO
                            if inst=='ESPRESSO':
                                tel_inst = {
                                    'ESO-VLT-U1':'1',
                                    'ESO-VLT-U2':'2',
                                    'ESO-VLT-U3':'3',
                                    'ESO-VLT-U4':'4'
                                    }[hdr['TELESCOP']]                                

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
                                # the number of screened bins has already been set in 'scr_lgth' for the visit, using the empirical estimation on the residuals CCFs            
                                if vis not in gen_dic['scr_lgth'][inst]:
                                    gen_dic[inst][vis]['scr_lgth']=int(round(gen_dic['pix_size_v'][inst]/delta_rv))
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
                                    if inst in ['ESPRESSO','ESPRESSO_MR','HARPN','HARPS','CARMENES_VIS','NIRPS_HA','NIRPS_HE']:data_inst[vis]['nspec']  = (hdulist[1].header)['NAXIS1']
                                    elif inst=='EXPRES':data_inst[vis]['nspec'] = 7920
                                    else:stop('TBD')

                        #Table dimensions
                        data_inst[vis]['dim_all'] = [n_in_visit,data_inst['nord'],data_inst[vis]['nspec']]
                        data_inst[vis]['dim_exp'] = [data_inst['nord'],data_inst[vis]['nspec']]
                        data_inst[vis]['dim_sp'] = [n_in_visit,data_inst['nord']]
                        data_inst[vis]['dim_ord'] = [n_in_visit,data_inst[vis]['nspec']]

                        #Initialize spectral table, flux table, banded covariance matrix                   
                        data_dic_temp['cen_bins'] = np.zeros(data_inst[vis]['dim_all'], dtype=float)
                        data_dic_temp['flux'] = np.zeros(data_inst[vis]['dim_all'], dtype=float)*np.nan
                        data_dic_temp['cov'] = np.zeros(data_inst[vis]['dim_sp'], dtype=object)

                        #Telluric spectrum
                        if data_inst[vis]['tell_sp'] and (gen_dic['calc_tell_mode']=='input'):data_dic_temp['tell'] = np.ones(data_inst[vis]['dim_all'], dtype=float)

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
                        args_exp = def_st_prof_tab(None,None,None,fixed_args)

                        #Initializing stellar profiles
                        args_exp = init_custom_DI_prof(args_exp,gen_dic,data_dic['DI']['system_prop'],data_dic['DI']['spots_prop'],theo_dic,system_param['star'],param_exp)
                        
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
                        surf_prop_dic, surf_prop_dic_sp,_ = sub_calc_plocc_spot_prop([data_dic['DI']['system_prop']['chrom_mode']],args_exp,['line_prof'],data_dic[inst][vis]['transit_pl'],deepcopy(system_param),theo_dic,args_exp['system_prop'],param_exp,coord_dic[inst][vis],[iexp], system_spot_prop_in=args_exp['system_spot_prop'])

                        #Correcting the disk-integrated profile for planet and spot contributions
                        if param_exp['use_spots']:
                            DI_prof_exp = base_DI_prof - surf_prop_dic[data_dic['DI']['system_prop']['chrom_mode']]['line_prof'][:,0] - surf_prop_dic_sp[data_dic['DI']['system_prop']['chrom_mode']]['line_prof'][:,0]
                        
                        #Correcting the disk-integrated profile for planet contribution alone
                        else:
                            DI_prof_exp = base_DI_prof - surf_prop_dic[data_dic['DI']['system_prop']['chrom_mode']]['line_prof'][:,0]

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
                        # EF_meas(t,w) = sqrt(gcal(band) F_meas(t,w))
                        #      which is a biased estimate of the true error but corresponds to what is returned by DRS
                        #    - the S/N of the mock profiles is F_meas(t,w)/EF_meas(t,w) = sqrt(F_meas(t,w)/gcal(band)) = sqrt(Nmeas(t,w)) proportional to sqrt(texp*cont)
                        #      errors in the continuum of the normalized profiles are proportional to 1/sqrt(texp*cont) 
                        #      beware that this error does not necessarily match the flux dispersion
                        if (inst in mock_dic['set_err']) and mock_dic['set_err'][inst]:
                            DI_prof_exp_Fmeas = np.array(list(map(np.random.poisson, DI_prof_exp_Ftrue,  data_inst[vis]['nspec']*[1]))).flatten()
                            DI_err_exp_Emeas = np.sqrt(mock_gcal*DI_prof_exp_Fmeas)
                        else:
                            DI_prof_exp_Fmeas = DI_prof_exp_Ftrue
                            DI_err_exp_Emeas = np.zeros(data_inst[vis]['nspec'],dtype=float)
                        data_dic_temp['flux'][iexp,0] = DI_prof_exp_Fmeas                      
                        data_dic_temp['cov'][iexp,0] = (DI_err_exp_Emeas**2.)[None,:]                        




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
                            elif inst in ['HARPN','HARPS','ESPRESSO','ESPRESSO_MR','NIRPS_HA','NIRPS_HE']:reduc_txt = facil_inst+' QC'
                            else:stop('Define BERV retrieval')
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
                            if inst not in gen_dic['orders4ccf']:flux_raw = np.nansum(all_ccf[ range(gen_dic['norders_instru'][inst])],axis=0)     
                            else:flux_raw = np.nansum(all_ccf[gen_dic['orders4ccf'][inst]],axis=0)             

                            #Screening CCF
                            if gen_dic[inst][vis]['scr_lgth']>1:flux_raw = flux_raw[idx_scr_bins]
                            data_dic_temp['flux'][iexp,0] = flux_raw
                            
                            #Error table
                            if gen_dic[inst][vis]['flag_err']:
                                if inst not in gen_dic['orders4ccf']:err_raw = np.sqrt(np.nansum(all_eccf[range(gen_dic['norders_instru'][inst])]**2.,axis=0))      
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
                                    stop('Spectra upload TBD for this instrument') 
        
                            #------------------------------
            
                            #2D spectra
                            elif data_inst['type']=='spec2D':
                                    
                                #Sky-corrected data
                                if (vis_path_skysub_exp is not None):
                                    if (gen_dic['fibB_corr'][inst][vis]=='all'):hdulist_dat= hdulist_skysub
                                    else:
                                        cond_skysub = np.repeat(False,gen_dic['norders_instru'][inst])     #Conditions on original order indexes 
                                        cond_skysub[idx_ord_skysub] = True  
                                        idxsub_ord_skysub = np_where1D(cond_skysub[idx_ord_kept])          #Indexes in reduced tables 
                                else:hdulist_dat = hdulist
                                    
                                if inst in ['HARPS','ESPRESSO','ESPRESSO_MR','HARPN','NIRPS_HA','NIRPS_HE']:

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
                                    
                                    #Remove pixels at the edge of the red detector
                                    #    - usually too poorly defined even for bright stars
                                    if inst=='ESPRESSO':
                                        for iord_bad in [90,91]: 
                                            isub_ord = np_where1D(np.array(idx_ord_kept)==iord_bad)
                                            if len(isub_ord)>0:
                                                data_dic_temp['flux'][iexp][isub_ord,0:30] = np.nan
                                                qualdata_exp[isub_ord,0:30] = 1                                   

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
                                    if data_inst[vis]['tell_sp'] and (gen_dic['calc_tell_mode']=='input'):
                                        data_dic_temp['tell'][iexp] = (hdulist_dat[1].data)['tellurics'][idx_ord_kept]  

                                else:
                                    stop('Spectra upload TBD for this instrument') 
                        
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
                        elif inst=='HARPN':   #Validated by Rosario Cosentino (harpsn instrument scientist)
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
                        elif inst=='HARPN':    #Validated by Rosario Cosentino (harpsn instrument scientist)
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
                        if inst in ['ESPRESSO']:
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
                        if inst in ['ESPRESSO']:
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
                            dace_idexp = closest( DACE_idx['bjd'] ,bjd_exp )
                            for key in data_inst[vis]['act_idx']:
                                data_prop[inst][vis][key][iexp,:] = [DACE_idx[key][dace_idexp],DACE_idx[key+'_err'][dace_idexp]]
                        elif inst=='EXPRES':
                            data_prop[inst][vis]['ha'][iexp,:] = [(hdulist[1].header)['HALPHA'],1.879E-2]
                            data_prop[inst][vis]['s'][iexp,:] = [(hdulist[1].header)['S-VALUE'], 6.611E-3]
                            

                    ### End of current exposure    
                    
                ### End of exposures in visit
           
                #------------------------------------------------------------------------------------

                #Scaling of errors
                if (inst not in gen_dic['g_err']):gen_dic['g_err'][inst]=1.
                elif gen_dic['g_err'][inst]!=1.:
                    print('           Uncertainties are scaled by sqrt('+str(gen_dic['g_err'][inst])+')')
                    data_dic_temp['cov']*=gen_dic['g_err'][inst]

                #Sort exposures by increasing time
                #    - remove exposures if requested, using their index after time-sorting 
                if (inst in gen_dic['used_exp']) and (vis in gen_dic['used_exp'][inst]) and (len(gen_dic['used_exp'][inst][vis])>0):remove_exp=True
                else:remove_exp=False
                w_sorted=coord_dic[inst][vis]['bjd'].argsort()    
                for pl_loc in data_inst[vis]['transit_pl']:
                    for key in ['ecl','cen_ph','st_ph','end_ph','ph_dur','rv_pl','v_pl']:
                        coord_dic[inst][vis][pl_loc][key]=coord_dic[inst][vis][pl_loc][key][w_sorted]                        
                        if remove_exp:coord_dic[inst][vis][pl_loc][key] = coord_dic[inst][vis][pl_loc][key][gen_dic['used_exp'][inst][vis]]
                    for key in ['cen_pos','st_pos','end_pos']:
                        coord_dic[inst][vis][pl_loc][key]=coord_dic[inst][vis][pl_loc][key][:,w_sorted] 
                        if remove_exp:coord_dic[inst][vis][pl_loc][key] = coord_dic[inst][vis][pl_loc][key][:,gen_dic['used_exp'][inst][vis]] 
                for key in ['bjd','t_dur','RV_star_solCDM','RV_star_stelCDM']:
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
    
                #Common spectral tables
                #    - the common table is set to the table of the first exposure, defined in the input rest frame
                #    - for spectra it will be used indifferently to resample spectra defined in the input or the stellar rest frame
                #    - for CCF it will be shifted to the star rest frame in the alignmen tmodule, if called 
                #      we give the table the same structure as for spectra, with dimensions nord x nspec
                data_inst[vis]['proc_com_data_paths'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_'+vis+'_com'
                data_com = {'cen_bins':data_dic_temp['cen_bins'][0],'edge_bins':data_dic_temp['edge_bins'][0],'dim_exp':data_inst[vis]['dim_exp'],'nspec':data_inst[vis]['nspec']}

                #Exposures in current visit are defined on different tables
                if ('spec' in data_inst['type']) and (np.nanmax(np.abs(data_dic_temp['edge_bins']-data_com['edge_bins'][None,:,:]))>1e-6):
                    data_inst[vis]['comm_sp_tab'] = False

                    #Resampling data on wavelength table common to the visit if requested
                    if gen_dic['comm_sp_tab'][inst]:
                        print('           Resampling on common table')
                        for iexp in range(n_in_visit):
                            for iord in range(data_inst['nord']): 
                                data_dic_temp['flux'][iexp,iord],data_dic_temp['cov'][iexp][iord] = bind.resampling(data_com['edge_bins'][iord], data_dic_temp['edge_bins'][iexp,iord], data_dic_temp['flux'][iexp,iord] , cov = data_dic_temp['cov'][iexp][iord], kind=gen_dic['resamp_mode']) 
                                
                                #Resampling input telluric spectrum
                                if data_inst[vis]['tell_sp'] and (gen_dic['calc_tell_mode']=='input'):
                                    data_dic_temp['tell'][iexp,iord] = bind.resampling(data_com['edge_bins'][iord], data_dic_temp['edge_bins'][iexp,iord], data_dic_temp['tell'][iexp,iord], kind=gen_dic['resamp_mode'])         

                        #Overwrite exposure tables
                        data_dic_temp['cen_bins'][:] = deepcopy(data_com['cen_bins'])
                        data_dic_temp['edge_bins'][:] = deepcopy(data_com['edge_bins'])                                    
                        data_dic_temp['cond_def'] = ~np.isnan(data_dic_temp['flux'])
                        
                        #Set flag that all exposures in the visit are now defined on a common table
                        data_inst[vis]['comm_sp_tab'] = True

                #Save table common to all visits
                #    - set to the common table of the first instrument visit
                if ivis==0:
                    data_dic[inst]['proc_com_data_path'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_com'
                    data_dic[inst]['dim_exp'] = deepcopy(data_inst[vis]['dim_exp'])
                    data_dic[inst]['nspec'] = deepcopy(data_inst[vis]['nspec'])
                    data_dic[inst]['com_vis'] = vis   
                    data_com_inst = deepcopy(data_com)
                    np.savez_compressed(data_dic[inst]['proc_com_data_path'],data = data_com_inst,allow_pickle=True)

                #Check wether the common table of current visit is the same as the common table for the instrument
                #    - flag set to False if :
                # + exposures of current visit are still defined on individual table 
                # + exposures of current visit are defined on a common table, but it is different from the instrument table (only possible if current visit is not the first, used as reference)   
                if (not data_inst[vis]['comm_sp_tab']) or ((ivis>0) and ((data_com_inst['nspec']!=data_com['nspec']) or (np.nanmax(np.abs(data_com_inst['edge_bins']-data_com['edge_bins']))>1e-6))):
                      data_dic[inst]['comm_sp_tab'] = False

                #------------------------------------------------------------------------------------

                #Final processing of exposures
                data_inst[vis]['proc_DI_data_paths']=gen_dic['save_data_dir']+'Processed_data/'+inst+'_'+vis+'_'
                for iexp in range(n_in_visit):

                    #Set masked pixels to nan
                    #    - masked orders and positions are assumed to be the same for all exposures, but positions have to be specific to a given order as orders can overlap
                    if (inst in gen_dic['masked_pix']) and (vis in gen_dic['masked_pix'][inst]):
                        mask_exp = gen_dic['masked_pix'][inst][vis]['exp_list']
                        if (len(mask_exp)==0) or (iexp in gen_dic['masked_pix'][inst][vis]):
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
                    if data_inst[vis]['tell_sp'] and (gen_dic['calc_tell_mode']=='input'):
                        data_inst[vis]['tell_DI_data_paths'][iexp] = data_inst[vis]['proc_DI_data_paths']+'tell_'+str(iexp)
                        np.savez_compressed(data_inst[vis]['tell_DI_data_paths'][iexp], data = {'tell':data_dic_temp['tell'][iexp]},allow_pickle=True) 
                        data_dic_temp.pop('tell')    
                        
                    #Saving path to initialized raw data
                    #    - saving data per exposure to prevent size issue with npz files 
                    data_exp = {key:data_dic_temp[key][iexp] for key in ['cen_bins','edge_bins','flux','cov','cond_def']}                    
                    np.savez_compressed(data_inst[vis]['proc_DI_data_paths']+str(iexp),data=data_exp,allow_pickle=True)   
                               
                #Check for empty orders
                idx_ord_empty = np_where1D(np.sum(data_dic_temp['cond_def'],axis=(0,2)) == 0.)
                if len(idx_ord_empty)>0:
                    txt_str = '['
                    for iord in idx_ord_empty[0:-1]:txt_str+=str(iord)+','
                    txt_str+=str(idx_ord_empty[-1])+']'
                    print('           Empty orders : '+txt_str)

                #Saving defined edges of common spectral table
                low_bins_def = deepcopy(data_dic_temp['edge_bins'][:,:,0:-1])
                high_bins_def = deepcopy(data_dic_temp['edge_bins'][:,:,1::])
                low_bins_def[~data_dic_temp['cond_def']] = np.nan
                high_bins_def[~data_dic_temp['cond_def']] = np.nan
                data_com['min_edge_ord'] = np.nanmin(low_bins_def,axis=(0,2))             
                data_com['max_edge_ord'] = np.nanmax(high_bins_def,axis=(0,2))
                np.savez_compressed(data_inst[vis]['proc_com_data_paths'],data = data_com,allow_pickle=True)

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
            data_dic[inst][vis]['proc_DI_data_paths']  = gen_dic['save_data_dir']+'Processed_data/'+inst+'_'+vis+'_'
            data_dic[inst][vis]['proc_com_data_paths'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_'+vis+'_com'
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

    #Final processing
    if len(data_dic[inst]['visit_list'])>1:
        if (not data_dic[inst]['comm_sp_tab']):print('         Visits do not share a common spectral table')      
        else:print('         All visits share a common spectral table')    
    for vis in data_dic[inst]['visit_list']:   
        data_vis = data_dic[inst][vis]
        coord_vis = coord_dic[inst][vis]
        gen_vis = gen_dic[inst][vis] 
        if (not data_vis['comm_sp_tab']):print('           Exposures in '+vis+' do not share a common spectral table')      
        else:print('           All exposures in '+vis+' share a common spectral table')   

        #------------------------------------------------------------------------------------
              
        #Automatic continuum and fit range
        #    - done here rather than in the 'calc_proc_data' condition so that ranges can be defined for already-processed observed or mock datasets, even if the analysis modules were not activated 
        # at the time of processing
        for key in ['DI','Intr','Atm']:
            if gen_dic['fit_'+key+'_gen'] or ((key=='Intr') & gen_dic['fit_IntrProf']):
                autom_cont = True if (inst not in data_dic[key]['cont_range']) else False
                autom_fit = True if ((inst not in data_dic[key]['fit_range']) or (vis not in data_dic[key]['fit_range'][inst])) else False
                if autom_cont or autom_fit:
                    if (data_dic[key]['type'][inst]=='CCF'):iord_fit = 0
                    else:iord_fit = data_dic[key]['fit_prof']['order'][inst]
                    cen_bins_all = np.zeros([0,data_vis['nspec']],dtype=float)
                    edge_bins_all = np.zeros([0,data_vis['nspec']+1],dtype=float)
                    flux_all = np.zeros([0,data_vis['nspec']],dtype=float)
                    for iexp in range(data_vis['n_in_visit']):
                        data_exp = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp))
                        cen_bins_all=np.append(cen_bins_all,[data_exp['cen_bins'][iord_fit]],axis=0)
                        edge_bins_all=np.append(edge_bins_all,[data_exp['edge_bins'][iord_fit]],axis=0)
                        flux_all=np.append(flux_all,[data_exp['flux'][iord_fit]],axis=0)
                    if (data_dic[key]['type'][inst]=='CCF'):
                        RVcen_bins = cen_bins_all
                        RVedge_bins = edge_bins_all
                    else:
                        RVcen_bins = c_light*((cen_bins_all/data_dic[key]['line_trans']) - 1.)
                        RVedge_bins = c_light*((edge_bins_all/data_dic[key]['line_trans']) - 1.)

                    #Estimate of systemic velocity for disk-integrated profiles
                    #    - as median of the RV corresponding to the CCF minima over the visit
                    #    - intrinsic and atmospheric profiles are fitted in the star rest frame
                    idx_check = (RVcen_bins[0]>-500.) & (RVcen_bins[0]<500.) 
                    idx_min_all = np.argmin(flux_all,axis=1)
                    sys_RV = np.nanmedian(RVcen_bins[:,idx_check][:,idx_min_all])
                    if key=='DI':cen_RV = sys_RV
                    else:cen_RV = 0.

                    #Minimum/maximum velocity of the CCF range in the input rest frame
                    min_bin = np.max(np.nanmin(RVedge_bins))
                    max_bin = np.min(np.nanmax(RVedge_bins)) 
                    if key in ['Intr','Atm']:
                        min_bin-=sys_RV
                        max_bin-=sys_RV
                       
                    #Excluded range for disk-integrated and intrinsic profiles 
                    #    - we assume +-3 vsini accounts for both rotational and thermal broadening of DI profiles, and for rotational shift + thermal broadening
                    if key in ['DI','Intr']:
                        min_exc = np.max([cen_RV - 3.*system_param['star']['vsini'] -5.,min_bin+5.])
                        max_exc = np.min([cen_RV + 3.*system_param['star']['vsini'] +5.,max_bin-5.])

                    #Excluded range for atmospheric profiles
                    #    - we account for a width of 20km/s over the range of studied orbital velocities, for all planets considered for atmospheric signals
                    elif key=='Atm':
                        min_pl_RV = 1e100
                        max_pl_RV = -1e100
                        if data_dic['Atm']['pl_atm_sign']=='Absorption':idx_atm = gen_dic[inst][vis]['idx_in']                                 
                        elif data_dic['Atm']['pl_atm_sign']=='Emission': idx_atm = data_dic[inst][vis]['n_in_tr']
                        for pl_atm in data_dic[inst][vis]['pl_with_atm']:
                            min_pl_RV = np.min([min_pl_RV,np.min(coord_dic[inst][vis][pl_atm]['rv_pl'][idx_atm])])
                            max_pl_RV = np.max([max_pl_RV,np.max(coord_dic[inst][vis][pl_atm]['rv_pl'][idx_atm])])  
                        min_exc = np.min(min_pl_RV -20.,min_bin+5.)
                        max_exc = np.max(max_pl_RV +20.,max_bin-5.)    

                    #Outer boundaries of continuum and fitted ranges
                    min_contfit = np.max([min_bin, min_exc - 50.])
                    max_contfit = np.min([max_bin, max_exc + 50.])

                    #RV/wave conversion
                    if (data_dic[key]['type'][inst]!='CCF'):
                        min_contfit = data_dic[key]['line_trans']*(1.+(min_contfit/c_light))
                        max_contfit = data_dic[key]['line_trans']*(1.+(max_contfit/c_light))
                        min_exc = data_dic[key]['line_trans']*(1.+(min_exc/c_light))
                        max_exc = data_dic[key]['line_trans']*(1.+(max_exc/c_light))

                    #Continuum range
                    if autom_cont:
                        data_dic[key]['cont_range'][inst] = {0:[[min_contfit,min_exc],[max_exc,max_contfit]]}
             
                    #Fitting range
                    if autom_fit:      
                        if inst not in data_dic[key]['fit_range']:data_dic[key]['fit_range'][inst]={}
                        data_dic[key]['fit_range'][inst][vis] = [[min_contfit,max_contfit]]

        #------------------------------------------------------------------------------------
    
        #Default systemic velocity
        if (inst not in data_dic['DI']['sysvel']):data_dic['DI']['sysvel'][inst] = {}
        if (vis not in data_dic['DI']['sysvel'][inst]):
            data_dic['DI']['sysvel'][inst][vis] = 0.
            print('         WARNING : sysvel set to 0 km/s for ',inst,' : ',vis)

        #Define CCF resolution, range, and sysvel here so that spectral data needs not be uploaded again after processing
        if gen_dic['CCF_from_sp'] or gen_dic['Intr_CCF']:
          
            #Velocity table in original frame
            if gen_dic['dRV'] is None:dvelccf= gen_dic['pix_size_v'][inst]
            else:dvelccf= gen_dic['dRV']  
            n_vel = int((gen_dic['end_RV']-gen_dic['start_RV'])/dvelccf)+1
            data_vis['nvel'] = n_vel
            data_vis['velccf'] = gen_dic['start_RV']+dvelccf*np.arange(n_vel)

            #Bin edges in velocity space
            data_vis['edge_velccf'] = np.append(data_vis['velccf']-0.5*dvelccf,data_vis['velccf'][-1]+0.5*dvelccf)

            #Shifting the CCFs velocity table by RV(CDM_star/CDM_sun)
            for key in ['velccf','edge_velccf']:data_vis[key+'_star']=data_vis[key]-data_dic['DI']['sysvel'][inst][vis]  
    
        #Keplerian motion relative to the stellar CDM and the Sun (km/s)
        for iexp in range(data_dic[inst][vis]['n_in_visit']):
            coord_vis['RV_star_stelCDM'][iexp],coord_vis['RV_star_solCDM'][iexp] = calc_rv_star(coord_dic,inst,vis,system_param,gen_dic,coord_dic[inst][vis]['bjd'][iexp],coord_dic[inst][vis]['t_dur'][iexp],data_dic['DI']['sysvel'][inst][vis])

        #Using sky-corrected data for current visit
        if (inst in gen_dic['fibB_corr']) and (vis in gen_dic['fibB_corr'][inst]):print('          ',vis,'is sky-corrected') 

        #Indices of in/out exposures in global tables
        print('   > '+str(data_vis['n_in_visit'])+' exposures')  
        if ('in' in data_dic['DI']['idx_ecl']) and (inst in data_dic['DI']['idx_ecl']['in']) and (vis in data_dic['DI']['idx_ecl']['in'][inst]):
            for iexp in data_dic['DI']['idx_ecl']['in'][inst][vis]:
                for pl_loc in data_vis['transit_pl']:coord_vis[pl_loc]['ecl'][iexp] = 3*np.sign(coord_vis[pl_loc]['ecl'][iexp]) 
        if ('out' in data_dic['DI']['idx_ecl']) and (inst in data_dic['DI']['idx_ecl']['out']) and (vis in data_dic['DI']['idx_ecl']['out'][inst]):
            for iexp in data_dic['DI']['idx_ecl']['out'][inst][vis]:
                for pl_loc in data_vis['transit_pl']:coord_vis[pl_loc]['ecl'][iexp] = 1*np.sign(coord_vis[pl_loc]['ecl'][iexp]) 
        cond_in = np.zeros(data_vis['n_in_visit'],dtype=bool)
        cond_pre = np.ones(data_vis['n_in_visit'],dtype=bool) 
        cond_post = np.ones(data_vis['n_in_visit'],dtype=bool) 
        for pl_loc in data_vis['transit_pl']:
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
            if gen_dic['spec_1D_'+key]:def_1D_bins(data_dic[key])
        
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
    data_dic['Res'][inst][vis]={}
    gen_vis = gen_dic[inst][vis] 
    data_dic['Intr'][inst][vis]={}    

    #Current rest frame
    data_dic['DI'][inst][vis]['rest_frame'] = 'input'   
    
    #Current data
    data_vis['raw_exp_data_paths'] = deepcopy(data_vis['proc_DI_data_paths'])
    
    #Indices of in/out exposures in global tables
    print('   > '+str(data_vis['n_in_visit'])+' exposures')  
    print('        ',data_vis['n_in_tr'],'in-transit')
    print('        ',data_vis['n_out_tr'],'out-of-transit ('+str(len(gen_vis['idx_pretr']))+' pre / '+str(len(gen_vis['idx_posttr']))+' post)')
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
        for pl_loc in data_vis['transit_pl']:
            
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
                plAtm_vis['exclu_range_input']['spec'][pl_loc][:,:,iexp_no_plrange] = plAtm_vis['exclu_range_star']['spec'][pl_loc][:,:,iexp_no_plrange]*gen_specdopshift(coord_vis[pl_loc]['RV_star_stelCDM'][iexp_no_plrange])*gen_specdopshift(data_dic['DI']['sysvel'][inst][vis])    
                
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
    data_com_ref = dataload_npz(data_inst[data_inst['com_vis']]['proc_com_data_paths']) 
    if data_inst['comm_sp_tab'] and (gen_dic['align_DI']) and (data_dic['DI']['type'][inst]=='CCF'):
        data_inst['proc_com_data_path']=gen_dic['save_data_dir']+'Processed_data/'+inst+'_com_star'    
    elif gen_dic['spec_1D'] or gen_dic['CCF_from_sp']:
        if gen_dic['spec_1D']:
            data_inst['type']='spec1D' 
            data_inst['mean_gdet'] = False     
            data_inst['proc_com_data_path'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_com' 
        elif gen_dic['CCF_from_sp']:
            data_inst['type']='CCF'                 
            data_inst['tell_sp'] = False 
            data_inst['mean_gdet'] = False     
            data_inst['proc_com_data_path'] = gen_dic['save_data_dir']+'Processed_data/CCFfromSpec/'+inst+'_com' 
            if ('chrom' in data_inst['system_prop']):
                data_inst['system_prop']['chrom_mode'] = 'achrom'
                data_inst['system_prop'].pop('chrom')
        else:
            data_inst['proc_com_data_path'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_com' 
        
    #Common instrument table is set to the table of the visit taken as reference
    np.savez_compressed(data_inst['proc_com_data_path'],data = data_com_ref,allow_pickle=True)  
    
    if gen_dic['spec_1D'] or gen_dic['CCF_from_sp']:    
        data_inst['dim_exp'] = deepcopy(data_com_ref['dim_exp'])
        data_inst['nspec'] = deepcopy(data_com_ref['nspec'])
        data_inst['nord'] = 1
        data_inst['comm_sp_tab']=True
        
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
