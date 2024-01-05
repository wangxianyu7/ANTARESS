from ANTARESS_corrections.ANTARESS_sp_reduc import red_sp_data_instru
from ANTARESS_analysis.ANTARESS_joined_star import joined_Star_ana
from ANTARESS_analysis.ANTARESS_joined_atm import joined_Atm_ana
from ANTARESS_plots.ANTARESS_plots_all import ANTARESS_plot_functions
from ANTARESS_routines.ANTARESS_calib import calc_gcal
from ANTARESS_routines.ANTARESS_plocc_spec import def_plocc_profiles
from ANTARESS_conversions.ANTARESS_masks_gen import def_masks
from ANTARESS_conversions.ANTARESS_conv import CCF_from_spec,ResIntr_CCF_from_spec,conv_2D_to_1D_spec
from ANTARESS_grids.ANTARESS_plocc_grid import calc_plocc_prop
from ANTARESS_grids.ANTARESS_spots import calc_spots_prop, corr_spot
from ANTARESS_routines.ANTARESS_binning import process_bin_prof
from ANTARESS_corrections.ANTARESS_detrend import detrend_prof,pc_analysis
from ANTARESS_routines.ANTARESS_data_process import init_prop,init_data_instru,update_data_inst,init_visit,align_profiles,rescale_data,extract_res_profiles,extract_intr_profiles,extract_pl_profiles 
from ANTARESS_analysis.ANTARESS_ana_comm import MAIN_single_anaprof
from ANTARESS_routines.ANTARESS_sp_cont import process_spectral_cont


def ANTARESS_main(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic, corr_spot_dic,system_param,input_dic,user):
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
    coord_dic,data_prop = init_prop(data_dic,mock_dic,gen_dic,system_param,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic)
   
    ####################################################################################################################
    #Processing datasets for each visit of each instrument
    #    - binned visits are processed at the end
    ####################################################################################################################
    for inst in data_dic['instrum_list']:
        print('')
        print('--------------------------------')
        print('Processing instrument :',inst)        
        print('--------------------------------')

        #Initialize instrument tables and dictionaries
        init_data_instru(mock_dic,inst,gen_dic,data_dic,theo_dic,data_prop,coord_dic,system_param,plot_dic)

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
            
            #Initialization of visit properties
            init_visit(data_prop,data_dic,vis,coord_dic,inst,system_param,gen_dic)             
            
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

            #Calculating theoretical properties of the planet-occulted regions 
            if (gen_dic['theoPlOcc']): 
                calc_plocc_prop(system_param,gen_dic,theo_dic,coord_dic,inst,vis,data_dic,calc_pl_atm=gen_dic['calc_pl_atm'],mock_dic=mock_dic)
                
            #Calculating theoretical properties of the spot-occulted regions 

            if (gen_dic['theo_spots']): 
                calc_spots_prop(gen_dic,system_param['star'],theo_dic,inst,data_dic)

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
                rescale_data(data_dic[inst],inst,vis,data_dic,coord_dic,coord_dic[inst][vis]['t_dur_d'],gen_dic,plot_dic,system_param,theo_dic)   
         
            #Calculating master spectrum of the disk-integrated star used in weighted averages and continuum-normalization
            if gen_dic['DImast_weight']:              
                process_bin_prof('',data_type_gen,gen_dic,inst,vis,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,masterDI=True,mock_dic=mock_dic)


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

            #Building estimates for complete local stellar profiles
            if gen_dic['loc_data_corr']:
                def_plocc_profiles(inst,vis,gen_dic,data_dic,data_prop,coord_dic,system_param,theo_dic,glob_fit_dic,plot_dic)
            
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

        #Update instrument visits   
        if gen_dic['spec_1D'] or gen_dic['CCF_from_sp']:
            update_data_inst(data_dic,inst,gen_dic)        

        #Processing binned profiles over multiple visits
        #    - beware that data from different visits should be comparable to be binned 
        #      this is not the case, e.g, with blazed 2D spectra or if the stellar line shape changed 
        if (data_dic[inst]['n_visits_inst']>1) and (gen_dic['binmultivis']):
            print('--------------------------------')
            print('  Processing combined visits')         
            for data_type_gen in ['DI','Intr','Atm']:
                bin_gen_functions(data_type_gen,'multivis',inst,gen_dic,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic)

    ### end of instruments  



    ####################################################################################################################
    #Call to analysis function over combined visits and instruments
    ####################################################################################################################
    if gen_dic['multi_inst']:
        print('--------------------------------')
        print('Processing combined instruments')        
    
        #Wrap-up function to fit intrinsic stellar profiles and surface RVs   
        if gen_dic['fit_IntrProf'] or gen_dic['fit_IntrProp'] or gen_dic['fit_ResProf'] :
            joined_Star_ana(glob_fit_dic,system_param,theo_dic,data_dic,gen_dic,plot_dic,coord_dic)
    
        #Wrap-up function to fit atmospheric profiles and their properties
        if gen_dic['fit_AtmProf'] or gen_dic['fit_AtmProp']:
            joined_Atm_ana(gen_dic)

    ##############################################################################
    #Call to plot functions
    ##############################################################################
    if gen_dic['plots_on']:
        ANTARESS_plot_functions(system_param,plot_dic,data_dic,gen_dic,coord_dic,theo_dic,data_prop,glob_fit_dic,mock_dic,input_dic,user)

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




