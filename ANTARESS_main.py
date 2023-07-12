"""
ANTARESS : Advocating a Neat Technique for the Accurate Retrieval of Exoplanetary and Stellar Spectra
"""

from ANTARESS_routines import extract_intr_profiles,ana_prof,calc_plocc_prop,calc_spots_prop,def_plocc_profiles,calc_gcal,\
                                rescale_data,extract_pl_profiles,CCF_from_spec,detrend_prof,ResIntr_CCF_from_spec,\
                                init_prop,init_visit,update_data_inst,align_profiles,init_data_instru,extract_res_profiles,\
                                process_bin_prof,conv_2D_to_1D_spec, corr_spot,pc_analysis,def_masks,process_spectral_cont
from ANTARESS_sp_reduc import red_sp_data_instru
from ANTARESS_joined_routines import fit_intr_funcs,fit_atm_funcs
from ANTARESS_plots import ANTARESS_plot_functions
from copy import deepcopy
from utils import stop

"""
Main ANTARESS routines
"""
def ANTARESS_main(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic,PropAtm_fit_dic,AtmProf_fit_dic, corr_spot_dic,system_param):

    print('****************************************')
    print('Launching ANTARESS')
    print('     Study of :') 
    for pl_loc in gen_dic['transit_pl'].keys():print('      ',pl_loc)
    print('****************************************')
    print('')
        
    ##############################################################################
    #Initializations
    ##############################################################################
    coord_dic,data_prop = init_prop(data_dic,mock_dic,gen_dic,system_param,theo_dic,plot_dic,glob_fit_dic,PropAtm_fit_dic,detrend_prof_dic)
   
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
            curr_type = 'DI'
            #------------------------------------------------- 

            #Spectral detrending   
            if gen_dic['detrend_prof'] and (detrend_prof_dic['full_spec']):
                detrend_prof(detrend_prof_dic,data_dic,coord_dic,inst,vis,data_dic,data_prop,gen_dic,plot_dic)

            #Converting DI stellar spectra into CCFs
            if gen_dic[curr_type+'_CCF']:
                CCF_from_spec(curr_type,inst,vis,data_dic,gen_dic)
     
            #Single line detrending    
            if gen_dic['detrend_prof'] and (not detrend_prof_dic['full_spec']):
                detrend_prof(detrend_prof_dic,data_dic,coord_dic,inst,vis,data_dic,data_prop,gen_dic,plot_dic)

            #Calculating theoretical properties of the planet occulted-regions 
            if (gen_dic['theoPlOcc']): 
                calc_plocc_prop(system_param,gen_dic,theo_dic,coord_dic,inst,vis,data_dic,calc_pl_atm=gen_dic['calc_pl_atm'])
                
            #Calculating theoretical properties of the planet occulted-regions 
            if (gen_dic['theo_spots']): 
                calc_spots_prop(gen_dic,system_param['star'],theo_dic,inst,data_dic)

            #Analyzing original disk-integrated profiles
            if gen_dic['fit_'+curr_type]:
                ana_prof('',curr_type+'orig',data_dic,gen_dic,inst,vis,coord_dic,theo_dic,plot_dic,system_param['star'])

            #Aligning disk-integrated profiles to star rest frame
            if (gen_dic['align_'+curr_type]):
                align_profiles(curr_type,data_dic,inst,vis,gen_dic,coord_dic)
               
            # #Correcting for spot contamination 
            # if gen_dic['correct_spots'] : 
            #     corr_spot(corr_spot_dic, coord_dic,inst,vis,data_dic,data_prop,gen_dic, theo_dic, system_param)
              
            #Rescaling profiles to their correct flux level                  
            if gen_dic['flux_sc']:                   
                rescale_data(data_dic[inst],inst,vis,data_dic,coord_dic,coord_dic[inst][vis]['t_dur_d'],gen_dic,plot_dic,system_param,theo_dic)   

            #Calculating master spectrum of the disk-integrated star used in weighted averages and continuum-normalization
            if gen_dic['DImast_weight']:              
                process_bin_prof('',curr_type,gen_dic,inst,vis,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,masterDI=True)

            #Processing converted 2D disk-integrated profiles
            if gen_dic['spec_1D']:                
                conv_2D_to_1D_gen_functions(curr_type,data_dic,inst,vis,gen_dic,coord_dic,theo_dic,plot_dic,system_param)
 
            #Processing binned disk-integrated profiles
            if gen_dic['bin']:
                bin_gen_functions(curr_type,'',inst,gen_dic,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,vis=vis)

            #--------------------------------------------------------------------------------------------------
            #Processing residual and intrinsic stellar profiles
            curr_type = 'Intr'
            #--------------------------------------------------------------------------------------------------

            #Extracting residual profiles
            if (gen_dic['res_data']):
                extract_res_profiles(gen_dic,data_dic,inst,vis,data_prop,coord_dic)
           
            #Extracting intrinsic stellar profiles
            if gen_dic['intr_data']:
                extract_intr_profiles(data_dic,gen_dic,inst,vis,system_param['star'],coord_dic,theo_dic,plot_dic)

            #Converting out-of-transit residual and intrinsic spectra into CCFs
            if gen_dic[curr_type+'_CCF'] and ('spec' in data_dic[inst][vis]['type']):
                ResIntr_CCF_from_spec(inst,vis,data_dic,gen_dic)

            #Applying PCA to out-of transit residual profiles
            if (gen_dic['pca_ana']):
                pc_analysis(gen_dic,data_dic,inst,vis,data_prop,coord_dic)

            #Fitting intrinsic stellar profiles in the star rest frame
            if gen_dic['fit_'+curr_type]:
                ana_prof('',curr_type+'orig',data_dic,gen_dic,inst,vis,coord_dic,theo_dic,plot_dic,system_param['star'])
            
            #Aligning intrinsic stellar profiles to their local rest frame
            if gen_dic['align_'+curr_type]: 
                align_profiles(curr_type,data_dic,inst,vis,gen_dic,coord_dic)

            #Processing converted 2D intrinsic profiles
            if gen_dic['spec_1D']:                
                conv_2D_to_1D_gen_functions(curr_type,data_dic,inst,vis,gen_dic,coord_dic,theo_dic,plot_dic,system_param)

            #Processing binned intrinsic profiles
            if gen_dic['bin']:
                bin_gen_functions(curr_type,'',inst,gen_dic,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,vis=vis)

            #Building estimates for complete local stellar profiles
            if gen_dic['loc_data_corr']:
                def_plocc_profiles(inst,vis,gen_dic,data_dic,data_prop,coord_dic,system_param,theo_dic,glob_fit_dic)
        
            #--------------------------------------------------------------------------------------------------
            #Processing atmospheric profiles
            curr_type = 'Atm'
            #--------------------------------------------------------------------------------------------------

            #Extracting atmospheric profiles
            if gen_dic['pl_atm']:
                extract_pl_profiles(data_dic,inst,vis,gen_dic)

            #Converting atmospheric spectra into CCFs
            if gen_dic[curr_type+'_CCF'] and ('spec' in data_dic[inst][vis]['type']):
                CCF_from_spec(data_dic['Atm']['pl_atm_sign'],inst,vis,data_dic,gen_dic)

            #Fitting atmospheric profiles in the star rest frame
            if gen_dic['fit_'+curr_type]:
                ana_prof('',curr_type+'orig',data_dic,gen_dic,inst,vis,coord_dic,theo_dic,plot_dic,system_param['star'])
      
            #Aligning atmospheric profiles to the planet rest frame
            if gen_dic['align_'+curr_type]:   
                align_profiles(curr_type,data_dic,inst,vis,gen_dic,coord_dic)      

            #Processing converted 2D intrinsic profiles
            if gen_dic['spec_1D']:                
                conv_2D_to_1D_gen_functions(curr_type,data_dic,inst,vis,gen_dic,coord_dic,theo_dic,plot_dic,system_param)

            #Processing binned atmospheric profiles
            if gen_dic['bin']:
                bin_gen_functions(curr_type,'',inst,gen_dic,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,vis=vis)

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
            for data_type in ['DI','Intr','Atm']:
                bin_gen_functions(data_type,'multivis',inst,gen_dic,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic)

    ### end of instruments  



    ####################################################################################################################
    #Call to analysis function over combined visits and instruments
    ####################################################################################################################
    if gen_dic['multi_inst']:
        print('--------------------------------')
        print('Processing combined instruments')        
    
        #Wrap-up function to fit intrinsic stellar profiles and surface RVs   
        if gen_dic['fit_IntrProf'] or gen_dic['fit_IntrProp'] or gen_dic['fit_ResProf'] :
            fit_intr_funcs(glob_fit_dic,system_param,theo_dic,data_dic,gen_dic,plot_dic,coord_dic)
    
        #Wrap-up function to fit atmospheric profiles and their properties
        if gen_dic['fit_AtmProf'] or gen_dic['fit_AtmProp']:
            fit_atm_funcs(PropAtm_fit_dic,gen_dic)

 
    ##############################################################################
    #Call to plot functions
    ##############################################################################
    if gen_dic['plots_on']:
        ANTARESS_plot_functions(system_param,plot_dic,data_dic,gen_dic,coord_dic,theo_dic,data_prop,glob_fit_dic,PropAtm_fit_dic)

    return None
    
    

'''
Wrap-up function for 2D->1D visits
'''
def conv_2D_to_1D_gen_functions(data_type,data_dic,inst,vis,gen_dic,coord_dic,theo_dic,plot_dic,system_param):

    #Processing 2D disk-integrated spectra into new 1D spectra
    if gen_dic['spec_1D_'+data_type] and (data_dic[inst][vis]['type']=='spec2D'): 
        conv_2D_to_1D_spec(data_type,inst,vis,gen_dic,data_dic,data_dic[data_type])  

    #Analyzing converted profiles
    if gen_dic['fit_'+data_type+'_1D']: 
        ana_prof('',data_type+'_1D',data_dic,gen_dic,inst,vis,coord_dic,theo_dic,plot_dic,system_param['star'])   

    return None

'''
Wrap-up function for binned visits
'''
def bin_gen_functions(data_type,mode,inst,gen_dic,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,vis=None):

    #Binning profiles for analysis purpose 
    if gen_dic[data_type+'bin'+mode]: 
        process_bin_prof(mode,data_type,gen_dic,inst,vis,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic)

    #Analyzing binned profiles
    if gen_dic['fit_'+data_type+'bin'+mode]: 
        ana_prof(mode,data_type+'bin',data_dic,gen_dic,inst,vis,coord_dic,theo_dic,plot_dic,system_param['star'])                        

    #Calculating generic stellar continuum from binned master spectrum
    if (data_type in ['DI','Intr']) and gen_dic[data_type+'_stcont']:
        process_spectral_cont(mode,data_type,inst,data_dic,gen_dic,vis)

    #Defining CCF mask
    #    - over all visits if possible, or over the single processed visit
    if (data_type in ['DI','Intr']) and gen_dic['def_'+data_type+'masks']:
        if (mode=='' and (data_dic[inst]['n_visits_inst']==1)) or (mode=='multivis'):
            def_masks(mode,gen_dic,data_type,inst,vis,data_dic,plot_dic,system_param,data_prop)

    return None




