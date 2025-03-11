#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from copy import deepcopy
from pathos.multiprocessing import cpu_count
import numpy as np

##################################################################################################    
#%% Global settings
##################################################################################################  

def ANTARESS_settings(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic):
    r"""**ANTARESS default settings: global**
    
    Initializes ANTARESS configuration settings with default values.  
    
    ANTARESS can process data from the following instruments, with the associated designation in the workflow:
        
     - CARMENES (visible detector): 'CARMENES_VIS'
     - CORALIE : 'CORALIE'   
     - ESPRESSO (1 UT) : 'ESPRESSO'
     - ESPRESSO (4 UT) : 'ESPRESSO_MR'
     - EXPRES : 'EXPRES'
     - HARPS-N : 'HARPN'
     - HARPS : 'HARPS'
     - MIKE (blue arm) : 'MIKE_Blue'
     - MIKE (red arm) : 'MIKE_Red'   
     - NIRPS (high-accuracy mode) : 'NIRPS_HA'
     - NIRPS (high-efficiency mode) : 'NIRPS_HE'
     - SOPHIE (high-efficiency mode) : 'SOPHIE_HE'
     - SOPHIE (high-resolution mode) : 'SOPHIE_HR'
     - MIKE (red arm) : 'MIKE_Red'
     - MIKE (blue arm) : 'MIKE_Blue'

    Args:
        TBD
    
    Returns:
        None
    
    """  

    ##################################################################################################    
    #%%% Settings: generic
    ##################################################################################################    

    #%%%% Planetary system
    
    #%%%%% Star name
    gen_dic['star_name']='Arda' 
    
    
    #%%%%% Transiting planets
    #    - indicate names (as defined in ANTARESS_systems) of the transiting planets to be processed
    #    - required to retrieve parameters for the stellar system and light curve
    #    - for each planet, indicate the instrument and visits in which its transit should be taken into account (visit names are those given through 'data_dir_list')
    #      if the pipeline is runned with no data, indicate the names of the mock dataset created artifially with the pipeline
    #    - if you process multiple visits, consider associating a transiting planet to all of them even if it does not transit so that all datasets can be studied as a function of this planet orbital phase
    #    - format: 'planet':{'inst':['vis']}
    gen_dic['studied_pl']={}  
    

    #%%%%% Visible active regions
    #    - indicate names (as defined in ANTARESS_systems) of the visible active regions to be processed
    #    - for each active region, indicate the instrument and visits in which it should be taken into account (visit names are those given through 'data_dir_list')
    #    - format: 'ar':{'inst':['vis']}
    gen_dic['studied_ar']={}  
    
    
    #%%%%% TTVs
    #    - if a visit is defined in this dictionary, the mid-transit time for this visit will be set to the specific value defined here
    #    - for single-night visits only
    #    - format: {'planet':{'inst':{'vis': value}}}
    gen_dic['Tcenter_visits'] = {}
    
    
    #%%%%% Keplerian planets    
    #    - list all planets to consider in the system for the star keplerian motion
    #    - set to 'all' for all defined planets to be accounted for
    gen_dic['kepl_pl'] = ['all']


    #%%%% Datasets

    #%%%%% Processing    
    
    #%%%%%% Calculating/retrieving
    gen_dic['calc_proc_data']= True
    
    
    #%%%%%% Disable calculation for all activated modules
    #    - if set to False: data will be retrieved, if present
    #    - if set to True: selection is based upon each module option
    gen_dic['calc_all'] = True 


    #%%%%%% Grid run
    #    - if set to True, ANTARESS is ran over a grid of values for the settings defined in ANTARESS_gridrun (using the nominal settings properties for other fields)
    gen_dic['grid_run'] = False
     
    
    #%%%%%% Workflow sequence
    #    - set to None to activate/deactivate manually each module of the workflow
    #      otherwise set to one of the following to enable a specific sequence:
    # + 'system_view' : only plot a view of the system, based on input properties and plot settings
    gen_dic['sequence'] = None 

    
    #%%%%% Input data type
    #    - for each instrument select among: 
    # + 'CCF': CCFs calculated by standard pipelines on stellar spectra
    # + 'spec1D': 1D stellar spectra
    # + 'spec2D': echelle stellar spectra
    gen_dic['type']={'ESPRESSO':'CCF'}
    
      
    #%%%%% Spectral frame
    #    - input spectra will be put into the requested frame ('air' or 'vacuum') if relevant
    #    - input frames:
    # + air: ESPRESSO, HARPS, HARPN, NIRPS_HE, NIRPS_HA
    # + vacuum: CARMENES_VIS, EXPRES 
    gen_dic['sp_frame']='air'


    #%%%%% Uncertainties
    
    #%%%%%% Using covariance matrix
    #    - set to True to propagate full covariance matrix and use it in fits (default for spectra)
    #      otherwise variance alone is used (imposed for CCFs)
    gen_dic['use_cov']=True


    #%%%%%% Manual variance table 
    #    - set instrument in list for its error tables to be considered undefined 
    #    - for spectral profiles errors are set to sqrt(F) for disk-integrated profiles and propagated afterwards
    #      error can be scaled with 'g_err'
    #    - for CCFs the same is done for disk-integrated profiles, but errors on local profiles are set to their continuum dispersion (and propagated afterwards)
    gen_dic['force_flag_err']=[]
    
    
    #%%%%%% Error scaling 
    #    - if no errors are provided with input tables, ANTARESS will automatically attribute a variance to flux values as sigma = sqrt(g_err*F)
    # where F is the number of photoelectrons received during an exposure. 
    #    - all error bars will be multiplied by sqrt(g_err) upon retrieval/definition
    #    - format: 'g_err' = {inst : value}
    #    - leave empty to prevent scaling
    gen_dic['g_err']={}
    

    #%%%%% CCF calculation
    
    #%%%%%% Radial velocity table
    #    - boundaries are defined in the solar barycentric rest frame
    #      the table is use for all CCFs throughout the pipeline:
    # + directly for CCFs on raw disk-integrated spectra
    # + after being shifted automatically into the star rest frame for local and atmospheric spectra
    #    - set dRV to None to use instrumental resolution
    #      these CCFs will not be screened, so be careful about the selected resolution (lower than instrumental will introduce correlations)
    gen_dic['start_RV']=-100.    
    gen_dic['end_RV']=100.
    gen_dic['dRV']=None      
    
    
    #%%%%%% Mask for stellar spectra
    #    - indicate path to mask
    #    - file format can be fits, csv, txt, dat with two columns: line wavelengths (A) and weights
    #    - beware that weights are set to the square of the mask line contrasts (for consistency with the ESPRESSO, HARPS and HARPS-N DRS)
    #    - can be used in one of these steps :
    # + CCF on input disk-integrated stellar spectra
    # + CCF on extracted local stellar spectra
    #    - CCF on atmospheric signals will be calculated with a specific mask
    #    - can be defined for the purpose of the plots (set to None to prevent upload)
    #    - CCF masks should be in the frame requested via gen_dic['sp_frame']
    gen_dic['CCF_mask'] = {}
    
    
    #%%%%%% Weights
    #    - use mask weights or not in the calculation of the CCFs    
    gen_dic['use_maskW'] = True
    
    
    #%%%%%% Orders
    #    - define orders over which the order-dependent CCFs should be coadded into a single CCF
    #    - data in CCF format are co-added from the CCFs of selected orders
    #      data in spectral format are cross-correlated over the selected orders only
    #    - beware that orders removed because of spectral range exclusion will not be used
    #    - leave empty to calculate the CCF over all of the specrograph orders
    #      otherwise select for each instrument the indices of the orders as an array (with indexes relative to the original instrument orders)
    #    - HARPS: i=0 to 71
    #      HARPN: i=0 to 68
    #      SOPHIE: i=0 to 38  
    #      ESPRESSO: i=0 to 169  
    gen_dic['orders4ccf']={}
    
    
    #%%%%%% Screening
    
    #%%%%%%% First pixel for screening
    #    - we keep only bins at indexes ist_bin + i*n_per_bin
    #      where n_per_bin is the correlation length of the CCFs
    #      ie we remove n_scsr_bins-1 points between two kept points, ie we keep one point in scr_lgth+1 
    #    - ist_bin is thus in [ 0 ; n_per_bin-1 ] 
    gen_dic['ist_scr']=0
    
    
    #%%%%%%% Screening length determination
    #    - select to calculate and plot the dispersion vs bin size on the Master out CCF continuum
    #    - the bin size where the noise becomes white can be used as screening length (set manually through scr_lgth)
    gen_dic['scr_search']=False
    
    
    #%%%%%%% Screening lengths
    #    - set manually for each visit of each instrument
    #    - if a visit is not defined for a given instrument, standard pixel size will be used (ie, no screening)
    gen_dic['scr_lgth']={}
    
    
    #%%%%%%% Plots: screening length analysis
    plot_dic['scr_search']=''    
    
    
    #%%%%% Resampling    
    
    #%%%%%% Resampling mode
    #    - linear interpolation ('linear') is faster than cubic interpolation ('cubic') but can introduce spurious features at the location of lines, blurred differently when resampled over different spectral tables
    gen_dic['resamp_mode']='cubic'  
    
    
    #%%%%%% Common spectral table
    #    - if set to True, data will always be resampled on the same table, specific to a given visit
    #      otherwise resampling operations will be avoided whenever possible, to prevent blurring and loss of resolution
    #    - this option will not resample different visits of a same instrument onto a common table
    #      this is only relevant for specific operations combining different visits, in which case it has to be done in any case 
    #    - imposed for CCFs
    #    - set to False if left empty
    gen_dic['comm_sp_tab'] = {}

    
    #%%%% Plot settings    
        
    #%%%%% Deactivating all plot routines
    #    - set to False to deactivate
    gen_dic['plots_on'] = True
    
    
    #Planetary system
    
    #Star name

    # gen_dic['star_name']='AUMic'
    # gen_dic['star_name']='AU_Mic'
    # gen_dic['star_name']='fakeAU_Mic'
    # gen_dic['star_name']='V1298tau'
    # gen_dic['star_name']='TRAPPIST1'
    # gen_dic['star_name']='TOI3884'
    gen_dic['star_name']='HD189733'

    # Zodiacs
    # gen_dic['star_name']='Capricorn'
    # gen_dic['star_name']='Cancer'
    # gen_dic['star_name']='Gemini'
    # gen_dic['star_name']='Sagittarius'
    # gen_dic['star_name']='Leo'
    # gen_dic['star_name']='Aquarius'
    # gen_dic['star_name']='Aries'
    # gen_dic['star_name']='Libra'
    # gen_dic['star_name']='Taurus'
    # gen_dic['star_name']='Scorpio'
    # gen_dic['star_name']='Virgo'
    # gen_dic['star_name']='Pisces'


    #Transiting planets
    if gen_dic['star_name']=='HD189733':
        gen_dic['studied_pl'] = {
            'HD189733b':{'ESPRESSO' : ['visit1']},
            }
        gen_dic['kepl_pl'] = ['HD189733b']

    if gen_dic['star_name']=='TOI3884':
        gen_dic['studied_pl'] = {
            'TOI3884_b':{'MIKE_Red' : ['mockvis']},
            }
        gen_dic['kepl_pl'] = ['TOI3884_b']

    if gen_dic['star_name']=='TRAPPIST1':
        gen_dic['studied_pl'] = {
            # 'TRAPPIST1_b':{'NIRPS_HE' : ['mockvis']},
            'TRAPPIST1_c':{'NIRPS_HE' : ['mockvis']},
            # 'TRAPPIST1_d':{'NIRPS_HE' : ['mockvis']},
            # 'TRAPPIST1_e':{'NIRPS_HE' : ['mockvis']},
            # 'TRAPPIST1_f':{'NIRPS_HE' : ['mockvis']},
            # 'TRAPPIST1_g':{'NIRPS_HE' : ['mockvis']},
            # 'TRAPPIST1_h':{'NIRPS_HE' : ['mockvis']}
            }
        gen_dic['kepl_pl'] = ['TRAPPIST1_b','TRAPPIST1_c','TRAPPIST1_d','TRAPPIST1_e','TRAPPIST1_f','TRAPPIST1_g','TRAPPIST1_h']
        # gen_dic['Tcenter_visits']={'TRAPPIST1_b':{'ESPRESSO':{'mockvis':2457322.514193}}}

    if gen_dic['star_name']=='AU_Mic':
        gen_dic['studied_pl'] = {
            'AU_Mic_b':{'ESPRESSO' : ['visit1']},
            }
        gen_dic['kepl_pl'] = ['AU_Mic_b']
        # gen_dic['Tcenter_visits']={'AU_Mic_b':{'ESPRESSO':{'visit1':2458330.39080}}}

    if gen_dic['star_name']=='fakeAU_Mic':
        gen_dic['studied_pl'] = {
            'fakeAU_Mic_b':{'ESPRESSO' : ['mockvisit1']},
            }
        gen_dic['kepl_pl'] = ['fakeAU_Mic_b']
        gen_dic['Tcenter_visits']={'fakeAU_Mic_b':{'ESPRESSO':{'mockvisit1':2458330.39080}}}

    if gen_dic['star_name']=='AUMic':
        gen_dic['studied_pl'] = {
            'AUMicb':{'ESPRESSO' : ['mock_vis', #--base
                                    'mock_vis1', 
                                    'mock_vis2', 
                                    'mock_vis3', 
                                    'mock_vis4', 
                                    'mock_vis5', 
                                    'mock_vis6', 
                                    'mock_vis7', 
                                    'mock_vis8', 
                                    'mock_vis9']},
            # 'AUMicc':{'ESPRESSO' : ['mock_vis']}, 
            }
        gen_dic['kepl_pl'] = ['AUMicb']
    
    if gen_dic['star_name']=='V1298tau':
        gen_dic['studied_pl'] = {
            'V1298tau_b':{'ESPRESSO' : ['mock_vis']}, 
            }
        gen_dic['kepl_pl'] = ['V1298tau_b']

    

    # Zodiacs
    for zodiac in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        if gen_dic['star_name']==zodiac:
            zodiac_pl =zodiac+'_b'
            gen_dic['studied_pl'] = {
                zodiac_pl:{'ESPRESSO' : ['mock_vis']}, 
                }
            gen_dic['kepl_pl'] = [zodiac_pl]




   #Transiting active regions
    # if gen_dic['star_name']=='HD189733':
    #     gen_dic['studied_ar'] = {
    #         'spot1':{'ESPRESSO' : ['visit1']} 
    #         }

    if gen_dic['star_name']=='TOI3884':
        gen_dic['studied_ar'] = {
            'spot1':{'MIKE_Red' : ['mockvis']} 
    #         'spot2':{'MIKE_Red' : ['mockvis']},
    #         'spot3':{'MIKE_Red' : ['mockvis']}, 
    #         'facula1':{'MIKE_Red' : ['mockvis']},
    #         'facula2':{'MIKE_Red' : ['mockvis']}, 
            }

    if gen_dic['star_name']=='TRAPPIST1':
        # gen_dic['studied_ar'] = {}
        gen_dic['studied_ar'] = {
            'spot1':{'NIRPS_HE' : ['mockvis']}, 
            'spot2':{'NIRPS_HE' : ['mockvis']}, 
            'facula1':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot3':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot4':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot5':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot6':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot7':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot8':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot9':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot10':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot11':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot12':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot13':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot14':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot15':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot16':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot17':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot18':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot19':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot20':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot21':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot22':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot23':{'NIRPS_HE' : ['mockvis']}, 
            # 'spot24':{'NIRPS_HE' : ['mockvis']}, 
            }

    if gen_dic['star_name']=='AUMic':
        # gen_dic['studied_ar'] = {}
        gen_dic['studied_ar'] = {
            'spot1':{'ESPRESSO' : ['mock_vis']},
            # 'facula1':{'ESPRESSO' : ['mock_vis']}, 
            }

    if gen_dic['star_name']=='AU_Mic':
        gen_dic['studied_ar'] = {
            'spot1':{'ESPRESSO' : ['visit1']},
            'spot2':{'ESPRESSO' : ['visit1']}, 
            }

    if gen_dic['star_name']=='fakeAU_Mic':
        gen_dic['studied_ar'] = {
            'spot1':{'ESPRESSO' : ['mockvisit1']},
            'spot2':{'ESPRESSO' : ['mockvisit1']}, 
            }

    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        gen_dic['studied_ar'] = {
            'spot1':{'ESPRESSO' : ['mock_vis']}, 
            }

    #Plot settings    
    
    #Input data type
    if gen_dic['star_name']=='HD189733':
        gen_dic['type']={'ESPRESSO':'CCF'}
        gen_dic['type']={'ESPRESSO':'spec2D'}

    if gen_dic['star_name']=='TOI3884':
        gen_dic['type']={'MIKE_Red':'CCF'}

    if gen_dic['star_name']=='TRAPPIST1':
        gen_dic['type']={'NIRPS_HE':'CCF'}  

    if gen_dic['star_name']=='AUMic':
        gen_dic['type']={'ESPRESSO':'CCF'}      
    
    if gen_dic['star_name']=='AU_Mic':
        gen_dic['type']={'ESPRESSO':'CCF'}

    if gen_dic['star_name']=='fakeAU_Mic':
        gen_dic['type']={'ESPRESSO':'CCF'}

    if gen_dic['star_name']=='V1298tau':
        gen_dic['type']={'ESPRESSO':'CCF'}
  
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        gen_dic['type']={'ESPRESSO':'CCF'}   

    #Spectral frame
    gen_dic['sp_frame']='air'

    #Data uncertainties

    #Using covariance matrix
    gen_dic['use_cov']=False

    # print('ATTENTION NOCOV')
    # gen_dic['use_cov']=False

    #Manual variance table 
    # gen_dic['force_flag_err']=['ESPRESSO']


    #Error scaling 
    # gen_dic['g_err']={'ESPRESSO':0.4}    
    # if 'HD3167_b' in gen_dic['transit_pl']:gen_dic['g_err']={'ESPRESSO':0.8173865854983544}    
    # gen_dic['g_err']={'CORALIE':1.5}   #TOI858, CCFs
    # gen_dic['g_err']={'CORALIE':2.4}   #TOI858, master out
    # gen_dic['g_err']={'ESPRESSO':10.}   #GJ436, prop. errors, CCF DIs indiv
    # if gen_dic['star_name']=='MASCARA1':gen_dic['g_err']={'ESPRESSO':10.}
    #RM survey, fit DI/Intr
    # if gen_dic['star_name']=='HAT_P3':gen_dic['g_err']={'HARPN':1225.821753159515}
    # if gen_dic['star_name']=='Kepler25':gen_dic['g_err']={'HARPN':1077.0782388569025}
    # if gen_dic['star_name']=='HAT_P33':gen_dic['g_err']={'HARPN':101.2502291529179}
    # if gen_dic['star_name']=='HD89345':gen_dic['g_err']={'HARPN':21325.166477498904}
    # if gen_dic['star_name']=='HAT_P49':gen_dic['g_err']={'HARPN':531.0020126395225}


    #Resampling    

    #Resampling mode
    # gen_dic['resamp_mode']='linear'
    gen_dic['resamp_mode']='cubic'  

    #Common spectral table
    gen_dic['comm_sp_tab'] = {}

       
    # if gen_dic['star_name']=='HD209458':
    #     print('ATTENTION RESAMPLING EVERYWHERE') 
    #     gen_dic['comm_sp_tab'] = {'ESPRESSO':True}    


    #Mask for stellar spectra
    #gen_dic['CCF_mask'] = '/Travaux/Radial_velocity/RV_masks/ESPRESSO_F9.fits'        #in the air 
    if gen_dic['star_name']=='HD189733':
        gen_dic['CCF_mask']['ESPRESSO'] = '/Users/samsonmercier/Desktop/Work/Master/2023-2024/antaress/src/antaress/ANTARESS_conversions/DRS_CCF_masks/ESPRESSO/ESPRESSO_new_K2.fits'   #K2V, taken as final

           
    #Orders
    #gen_dic['orders4ccf']={'HARPS':np.arange(36),'HARPN':np.arange(36)} 

    #Screening

    #First pixel for screening
    gen_dic['ist_scr']=0
    
    #Screening length determination
    gen_dic['scr_search']=False
   
    #Screening lengths
    gen_dic['scr_lgth']={}
     
    #Plots: screening length analysis
    plot_dic['scr_search']=''    


    #Grid run
    # gen_dic['grid_run'] = {'ESPRESSO':np.delete(np.arange(85),[59,68,74,75,76,77,78,82])}     #empty CCF orders (tellurics)


    #Data processing

    #Calculating/retrieving
    if gen_dic['star_name']=='HD189733':gen_dic['calc_proc_data']=True  #& False
    if gen_dic['star_name']=='TOI3884':gen_dic['calc_proc_data']=True  #& False
    if gen_dic['star_name']=='TRAPPIST1':gen_dic['calc_proc_data']=True  #& False
    if gen_dic['star_name']=='AUMic':gen_dic['calc_proc_data']=True  #& False
    if gen_dic['star_name']=='AU_Mic':gen_dic['calc_proc_data']=True  #& False
    if gen_dic['star_name']=='V1298tau':gen_dic['calc_proc_data']=True  #& False
    if gen_dic['star_name']=='fakeAU_Mic':gen_dic['calc_proc_data']=True  #& False

    #Zodiacs
    if gen_dic['star_name']in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:gen_dic['calc_proc_data']=True  #& False

        
    
    
    

    
    

    ##################################################################################################       
    #%%%Module: mock dataset 
    #    - if activated, the pipeline will generate/retrieve a mock dataset instead of observational data
    #    - use to define mock datasets that can then be processed in the same way as observational datasets
    #    - for now, limited to mocking a single band
    #    - the module uses options for the planet and star grid defined throughout the pipeline:
    #         + limb-darkening in data_dic['DI']['system_prop'] 
    #         + gravity-darkening in data_dic['DI']['system_prop'], if the star is oblate 
    #         + 'nsub_Dstar' in 'theo_dic' defines the resolution of the stellar grid (must be an odd number)
    #         + 'nsub_Dpl' in 'theo_dic'  defines the resolution of the planet grid
    #         + 'n_oversamp' in 'theo_dic' defines the oversampling of the planet position over each exposure    
    #         + 'precision' in 'theo_dic'  defines the precision at which planet-occulted profiles are defined
    ################################################################################################## 
    
    #%%%% Activating module
    gen_dic['mock_data'] =  False
    
    
    #%%%% Multi-threading
    mock_dic['nthreads'] = int(0.8*cpu_count())       
    
    
    #%%%%% Unthreaded operations
    #    - all operations are multi-threaded by default, but overheads of sharing data between threads may counterbalance the benefits of threading the model
    #    - select here which operations not to thread:
    # + 'prof_grid'
    mock_dic['unthreaded_op'] = []    
    
    
    #%%%% Defining artificial visits
    #    - exposures are defined for each instrument/visit and can be defined
    # + manually: indicate lower/upper exposure boundaries ('bin_low' and 'bin_high', ordered)
    #             format is {instrument : {visit : { 'exp_range':[t_st,t_end],'nexp': int}}}
    # + automatically : indicate total range ('exp_range' in BJD) and number of exposures ( 'nexp')
    #                   format is {instrument : {visit : {'bin_low':t_st , 'bin_high':t_end }}}
    #    - indicate times in BJD
    mock_dic['visit_def']={}
    
    
    #%%%% Spectral profile settings
    
    #%%%%% Spectral table for disk-integrated profiles 
    #    - in star rest frame, in A or km/s depending on chosen data type and model (see 'intr_prof')
    mock_dic['DI_table']={'x_start':-150.,'x_end':150.,'dx':0.01}
    
    
    #%%%%% Heliocentric stellar RV
    #    - in km/s
    #    - keplerian motion is added automatically to each exposure using the gen_dic['kepl_pl'] planets
    #    - format: 'sysvel' = {inst : {vis : value}}  
    mock_dic['sysvel']= {}  
        
    
    #%%%%% Intrinsic stellar spectra
    #    - we detail here the options and settings used throughout the pipeline (in gen_dic['mock_data'], gen_dic['fit_DI'], gen_dic['fit_IntrProf'], and gen_dic['loc_prof_est']) to define intrinsic profiles
    #    - line settings are defined per instrument
    #    - 'mode' = 'ana': intrinsic profiles are calculated analytically from input properties 
    # + set line_trans = None for the analytical model to be generated in RV space (CCF mode), or set it to the rest wavelength of the considered transition in the star rest frame (spectral mode)
    # + models are calculated directly on the 'DI_table', which is defined in RV space even in spectral mode to facilitate line profile calculation
    #   the model table table can be oversampled using the theo_dic['rv_osamp_line_mod'] field 
    # + line properties can vary as a polynomial along the chosen dimension 'coord_line'
    #   the role of the coefficients depends on the polynomial mode (absolute or polynomial) 
    #   'pol_mode' = 'abs' : coeff_pol[n]*x^n + coeff_pol[n-1]*x^(n-1) .. + coeff_pol[0]*x^0
    #   'pol_mode' = 'modul': (coeff_pol[n]*x^n + coeff_pol[n-1]*x^(n-1) .. + 1)*coeff_pol[0]
    # + lines properties can be derived from the fit to disk-integrated or intrinsic line profiles 
    # + line properties are defined through 'mod_prop', with the suffix "__IS__VS_" to keep the structure of the fit routine while defining properties visit per visit here 
    #   set ISx to x = instrument of definition (if  more than one)
    #   set VSy to y = visit (if  more than one for instrument x)    
    # + mactroturbulence is included if theo_dic['mac_mode'] is enabled               
    #    - 'mode' = 'theo': generate theoretical model grid as a function of mu
    # + the nominal atmosphere model based on the bulk stellar properties and settings defined via 'theo_dic' is used, so that no option needs to be defined here  
    # + models will be resampled on the 'DI_table', which must be spectral
    #    - 'mode' = 'Intrbin': using binned intrinsic profiles produced via gen_dic['Intrbin'] for the chosen dimension 'coord_line' (avoid using phase coordinates to bin, as it is not a physical coordinate)
    # + set 'vis' to '' to use intrinsic profiles binned from each processed visit, or to 'binned' to use intrinsic profiles binned from multiple visits in gen_dic['Intrbin']    
    # + intrinsic profiles must have been aligned before being binned
    # + instrumental convolution is automatically disabled, as the binned profiles are already convolved (it is stil an approximation, but better than to convolve twice)      
    mock_dic['intr_prof']={}        
        
    
    #%%%%% Continuum level
    #    - mean flux density of the unocculted star over the 'DI_range' band (specific to each visit), ie number of photoelectrons received for an exposure time of 1s
    #    - format: {inst:{vis:value}}
    mock_dic['flux_cont']={}

    
    #%%%%% SNR and photon count
    #   - toggle to print the average SNR and photon count of each simulated exposure.
    #   - can help to get a better sense of how to adjust the continuum level to achieve a certain SNR.
    mock_dic['verbose_flux_cont']=False
    

    #%%%%% Instrumental gain
    #    - the final count level is proportional to 'flux_cont' x 'gcal' but we separate the two fields to control separately the stellar emission and instrumental gain
    #    - set to 1 if undefined
    #    - format: {inst:{value}}
    mock_dic['gcal']={}


    #%%%% Active regions
    #    - can be (dark) spots and/or (bright) faculae
       
    #%%%%% Properties
    #    - active region inclusion is conditioned by this dictionary being filled in
    #    - active regions are defined by 4 parameters : 
    # + 'lat' : constant latitude of the active region, in star rest frame (in deg)
    # + 'Tc_ar' : Time (bjd) at which the active region is at longitude 0
    # + 'ang' : half-angular size (in deg) of the active region (we assume each region is circular)
    # + 'fctrst' : the flux level of the active region surface, relative to the quiet surface of the star
    #              0 = no emission, 1 = maximum emission (no contrast with the stellar surface), and >1 = emission greater than the quiet surface 
    #    - format: {inst : {vis : {prop : val}}}
    #      where prop is defined as par_ISinst_VSvis_ARar, to match with the structure used in gen_dic['fit_diff_prof']    
    mock_dic['ar_prop'] = {}

    #%%%%% Automatic generation of active regions
    #    - Instead of defining individual active regions, define multiple active regions
    #    - by providing distribution and relevant parameters from which 
    #    - to draw values for the latitude, crossing time, size, and contrast.
    #    - format: {inst : {vis : {prop : distrib}}}
    #      where prop is defined as par_ISinst_VSvis_ARactreg_name, to match with the structure used in gen_dic['fit_diff_prof']
    #      and distrib is a dictionary with the following possible formats:
    #    - {distrib : 'gauss', val, s_val}    # Drawing from a Gaussian distribution with median val and standard deviation s_val
    #    - {distrib : 'uf', low, high}        # Drawing from a Uniform distribution with boundaries low and high
    #    - Additionally, you must provide the number of active regions to generate with the following format:
    #    - {inst : {vis : {num}}}
    mock_dic['auto_gen_ar'] = {}

    #%%%% Noise settings
    
    #%%%%% Measured-like profile
    #    - format: {inst:bool}
    #    - set to True to draw randomly flux values in each pixel based on the model number of measured counts 
    #      leave empty or set to False to maintain flux values to the exact model profile
    #    - noise value are always defined as mock_dic['gcal'][inst] times the flux values
    mock_dic['set_err'] = {}    
     
     
    #%%%%% Jitter on intrinsic profile properties
    #    - for analytical models only
    #    - used to simulate local stellar activity
    #    - defined individually for all exposures
    #    - format: {inst:{vis:{prop1:value,prop2:value,...}}
    mock_dic['drift_intr'] = {}
           
    
    #%%%%% Systematic variations on disk-integrated profiles
    #    - for all types of models
    #    - possibilities: RV shift, change in instrumental resolution (replacing nominal instrumental convolution)
    #    - format: {inst:{vis:{rv:value,resol:value}}
    mock_dic['drift_post'] = {}
    
         
    #Activating module
    gen_dic['mock_data'] =  True & False

    #Setting number of threads 
    mock_dic['nthreads'] = 2 

    #Defining artificial visits
    if gen_dic['star_name'] == 'TOI3884':
        mock_dic['visit_def']={
            # 'MIKE_Red':{'mockvis' :{'exp_range':2459556.51669+np.array([-0.05, 0.05]),'nexp':30.}}} #Libby-Robert+2023
            # 'MIKE_Red':{'mockvis' :{'exp_range':2459556.51669+np.array([-0.0686, 0.0686]),'nexp':30.}}} #Libby-Robert+2023 # updated
            'MIKE_Red':{'mockvis' :{'exp_range':2459642.86314+np.array([-0.0686, 0.0686]),'nexp':30.}}} #Almenara+2022 #updated
            # 'MIKE_Red':{'mockvis' :{'exp_range':2459642.86314+np.array([-0.05, 0.05]),'nexp':30.}}} #Almenara+2022


    if gen_dic['star_name'] == 'TRAPPIST1':
        mock_dic['visit_def']={
            # 'NIRPS_HE':{'mockvis' :{'exp_range':2457322.514193+np.array([-0.2,0.2]),'nexp':200.}}} #--base
            # 'NIRPS_HE':{'mockvis' :{'exp_range':2460472.2+np.array([-0.2,0.2]),'nexp':2.}}} #--base
            # 'NIRPS_HE':{'mockvis' :{'exp_range':[2460469.7, 2460472.4],'nexp':200.}}} #--base
            # 'NIRPS_HE':{'mockvis' :{'exp_range':[2460470, 2460473],'nexp':10.}}} #--contains all below-mentioned transits
            # 'NIRPS_HE':{'mockvis' :{'exp_range':2460472.586+np.array([-0.08, 0.16]),'nexp':70.}}} #-- TRAPPIST1_b transit
            # 'NIRPS_HE':{'mockvis' :{'exp_range':2460472.502+np.array([-0.243, 0.243]),'nexp':2.}}} #-- TRAPPIST1_c transit
            'NIRPS_HE':{'mockvis' :{'exp_range':2460472.502+np.array([-0.029, 0.029]),'nexp':30.}}} #-- TRAPPIST1_c transit high SNR
            # 'NIRPS_HE':{'mockvis' :{'exp_range':2460472.206+np.array([-0.12, 0.12]),'nexp':70.}}} #-- TRAPPIST1_d transit
            # 'NIRPS_HE':{'mockvis' :{'exp_range':2460472.934+np.array([-0.12, 0.12]),'nexp':70.}}} #-- TRAPPIST1_e transit
            # 'NIRPS_HE':{'mockvis' :{'exp_range':2460470.466+np.array([-0.12, 0.12]),'nexp':70.}}} #-- TRAPPIST1_f transit



    if gen_dic['star_name'] == 'AUMic':
        mock_dic['visit_def']={
            'ESPRESSO':{'mock_vis' :{'exp_range':2458330.39051+np.array([-0.15,0.15]),'nexp':30.}} #--base
            # 'ESPRESSO':{'mock_vis' :{'exp_range':2458330.39051+np.array([-0.15,0.15]),'nexp':180.}} #- ESPRESSO exposure time
            
            # 'ESPRESSO':{'mock_vis1' :{'exp_range':2458330.39051+np.array([-0.15,0.15]),'nexp':30},
            #             'mock_vis2' :{'exp_range':2458330.39051+np.array([-0.15,0.15]),'nexp':30},
            #             'mock_vis3' :{'exp_range':2458330.39051+np.array([-0.15,0.15]),'nexp':30},
            #             'mock_vis4' :{'exp_range':2458330.39051+np.array([-0.15,0.15]),'nexp':30},
            #             'mock_vis5' :{'exp_range':2458330.39051+np.array([-0.15,0.15]),'nexp':30},
            #             'mock_vis6' :{'exp_range':2458330.39051+np.array([-0.15,0.15]),'nexp':30},
            #             'mock_vis7' :{'exp_range':2458330.39051+np.array([-0.15,0.15]),'nexp':30},
            #             'mock_vis8' :{'exp_range':2458330.39051+np.array([-0.15,0.15]),'nexp':30},
            #             'mock_vis9' :{'exp_range':2458330.39051+np.array([-0.15,0.15]),'nexp':30}
            #             }
            }
            # 'ESPRESSO':{'mock_vis' :{'exp_range':2458330.39051+np.array([-0.15,0.15]),'nexp':250}}} #-- plotting purposes
            # 'ESPRESSO':{'mock_vis' :{'exp_range':2458330.39051+np.array([-2,2])/24.,'nexp':50}}}
                       
    if gen_dic['star_name'] == 'V1298tau':
        mock_dic['visit_def']={
            'ESPRESSO':{'mock_vis' :{'exp_range':2457067.0488+np.array([-0.25, 0.25]),'nexp':50}}}

    
    if gen_dic['star_name'] == 'fakeAU_Mic':
        mock_dic['visit_def']={
            'ESPRESSO':{'mockvisit1' :{'exp_range':2458702.76484+np.array([-0.12122463487321511,0.1253348653553985]),'nexp':84}}} #--base


    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        mock_dic['visit_def']={
            # 'ESPRESSO':{'mock_vis' :{'exp_range':2458330.39051+np.array([-0.15,0.15]),'nexp':30.}}}
            'ESPRESSO':{'mock_vis' :{'exp_range':2458330.39051+np.array([-0.15,0.15]),'nexp':180.}}}

    #Spectral profile settings
    
    #Spectral table for disk-integrated profiles
    if gen_dic['star_name'] == 'TOI3884' :
        mock_dic['DI_table']={'x_start':-30.,'x_end':30.,'dx':0.2} 
    if gen_dic['star_name'] == 'TRAPPIST1' :
        mock_dic['DI_table']={'x_start':-100.,'x_end':100.,'dx':0.82}
    if gen_dic['star_name'] == 'AUMic' :
        mock_dic['DI_table']={'x_start':-200.,'x_end':200.,'dx':0.82}
    if gen_dic['star_name'] == 'fakeAU_Mic' :
        mock_dic['DI_table']={'x_start':-20.2,'x_end':20.3,'dx':0.42}
    if gen_dic['star_name'] == 'V1298tau':
        mock_dic['DI_table']={'x_start':-150.,'x_end':150.,'dx':0.8}
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        mock_dic['DI_table']={'x_start':-100.,'x_end':100.,'dx':0.82}

    #Heliocentric stellar RV
    if gen_dic['star_name']=='TOI3884':
        mock_dic['sysvel']= {
            'MIKE_Red' : {'mockvis' : 0., #--base 
            }
        }
    if gen_dic['star_name']=='TRAPPIST1':
        mock_dic['sysvel']= {
            'NIRPS_HE' : {'mockvis' : 0., #--base 
            }
        }

    if gen_dic['star_name']=='AUMic':
        mock_dic['sysvel']= {
            'ESPRESSO' : {'mock_vis' : 0., #--base
                        'mock_vis1' : 0.,
                          'mock_vis2' : 0.,
                          'mock_vis3' : 0.,
                          'mock_vis4' : 0.,
                          'mock_vis5' : 0.,
                          'mock_vis6' : 0.,
                          'mock_vis7' : 0.,
                          'mock_vis8' : 0.,
                          'mock_vis9' : 0.,  
            }
        } 
    if gen_dic['star_name'] == 'V1298tau' : 
        mock_dic['sysvel']= {'ESPRESSO' : {'mock_vis' : 0.}}  

    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        mock_dic['sysvel']= {'ESPRESSO' : {'mock_vis' : 0.}} 

    if gen_dic['star_name'] == 'fakeAU_Mic' : 
        mock_dic['sysvel']= {'ESPRESSO' : {'mockvisit1' : 0.}} 
    


    #Defining active region properties
    if gen_dic['star_name'] == 'TOI3884': 
        mock_dic['ar_prop']={
             'MIKE_Red':{
                 'mockvis':{

    #                 # For the spot 'spot1' : -- Libby-Roberts+2023
    #                  'lat__ISMIKE_Red_VSmockvis_ARspot1'     : -66,
    #                  'Tc_ar__ISMIKE_Red_VSmockvis_ARspot1' : 2459556.51669 + 2.4 + 0.8,
    #                  'ang__ISMIKE_Red_VSmockvis_ARspot1'     : 9,
    #                  'fctrst__ISMIKE_Red_VSmockvis_ARspot1'    : 0.45,

    #                 # For the spot 'spot2' : -- Libby-Roberts+2023
    #                  'lat__ISMIKE_Red_VSmockvis_ARspot2'     : -82,
    #                  'Tc_ar__ISMIKE_Red_VSmockvis_ARspot2' : 2459556.51669 - 0.5 + 0.8,
    #                  'ang__ISMIKE_Red_VSmockvis_ARspot2'     : 17,
    #                  'fctrst__ISMIKE_Red_VSmockvis_ARspot2'    : 0.45,

    #                 # For the spot 'spot3' : -- Libby-Roberts+2023
    #                  'lat__ISMIKE_Red_VSmockvis_ARspot3'     : -55,
    #                  'Tc_ar__ISMIKE_Red_VSmockvis_ARspot3' : 2459556.51669 + 0.2 + 0.8,
    #                  'ang__ISMIKE_Red_VSmockvis_ARspot3'     : 5,
    #                  'fctrst__ISMIKE_Red_VSmockvis_ARspot3'    : 0.45,

                    # For the spot 'spot1' : -- Almenara+2022
                     'lat__ISMIKE_Red_VSmockvis_ARspot1'     : -90,
                     'Tc_ar__ISMIKE_Red_VSmockvis_ARspot1' : 2459642.86314 + 2.4,
                     'ang__ISMIKE_Red_VSmockvis_ARspot1'     : 48.6,
                     'fctrst__ISMIKE_Red_VSmockvis_ARspot1'    : 0.59,

    #                 # For the facula 'facula1' : -- equatorial band hypothesis
    #                 'lat__ISMIKE_Red_VSmockvis_ARfacula1'     : 89.,
    #                 'Tc_ar__ISMIKE_Red_VSmockvis_ARfacula1' : 2459556.51669,
    #                 'ang__ISMIKE_Red_VSmockvis_ARfacula1'     : 75.,
    #                 'fctrst__ISMIKE_Red_VSmockvis_ARfacula1'    : 1.5,

    #                 # For the facula 'facula2' : -- equatorial band hypothesis 
    #                 'lat__ISMIKE_Red_VSmockvis_ARfacula2'     : -89,
    #                 'Tc_ar__ISMIKE_Red_VSmockvis_ARfacula2' : 2459556.51669,
    #                 'ang__ISMIKE_Red_VSmockvis_ARfacula2'     : 75.,
    #                 'fctrst__ISMIKE_Red_VSmockvis_ARfacula2'    : 1.5,
                    },
                }
            }


    if gen_dic['star_name']=='TRAPPIST1':
        mock_dic['ar_prop']={
             'NIRPS_HE':{
                 'mockvis':{
        #             # # For the spot 'spot1' : 
        #             'lat__ISNIRPS_HE_VSmockvis_ARspot1'     : 30,
        #             'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot1' : 2460472.0,
        #             'ang__ISNIRPS_HE_VSmockvis_ARspot1'     : 35, # 10, #15
        #             'fctrst__ISNIRPS_HE_VSmockvis_ARspot1'    : 0.7,

        #             # For the spot 'spot2' :
        #             'lat__ISNIRPS_HE_VSmockvis_ARspot2'     : -20,
        #             'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot2' : 2460472.8,
        #             'ang__ISNIRPS_HE_VSmockvis_ARspot2'     : 25, # 10, #15
        #             'fctrst__ISNIRPS_HE_VSmockvis_ARspot2'    : 0.7,

                    # # For the spot 'spot3' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot3'     : 4,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot3' : 2460471.1+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot3'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot3'    : 0.7,


                    # # For the spot 'spot4' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot4'     : 6,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot4' : 2460471.16+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot4'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot4'    : 0.7,


                    # # For the spot 'spot5' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot5'     : 12,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot5' : 2460471.14+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot5'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot5'    : 0.7,


                    # # For the spot 'spot6' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot6'     : -20,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot6' : 2460471.1+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot6'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot6'    : 0.7,


                    # # For the spot 'spot7' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot7'     : -20,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot7' : 2460471.0+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot7'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot7'    : 0.7,


                    # # For the spot 'spot8' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot8'     : -35,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot8' : 2460471.05+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot8'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot8'    : 0.7,


                    # # For the spot 'spot9' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot9'     : -46,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot9' : 2460471.05+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot9'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot9'    : 0.7,


                    # # For the spot 'spot10' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot10'     : -38,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot10' : 2460470.88+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot10'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot10'    : 0.7,


                    # # For the spot 'spot11' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot11'     : -26,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot11' : 2460470.65+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot11'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot11'    : 0.7,


                    # # For the spot 'spot12' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot12'     : -46,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot12' : 2460470.70+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot12'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot12'    : 0.7,


                    # # For the spot 'spot13' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot13'     : -15,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot13' : 2460470.5+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot13'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot13'    : 0.7,


                    # # For the spot 'spot14' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot14'     : -60,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot14' : 2460471.5+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot14'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot14'    : 0.7,


                    # # For the spot 'spot15' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot15'     : 8,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot15' : 2460471.6+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot15'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot15'    : 0.7,


                    # # For the spot 'spot16' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot16'     : 22,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot16' : 2460471.65+.51,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot16'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot16'    : 0.7,


                    # # For the spot 'spot17' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot17'     : 25,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot17' : 2460471.6+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot17'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot17'    : 0.7,


                    # # For the spot 'spot18' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot18'     : 20,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot18' : 2460471.45+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot18'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot18'    : 0.7,


                    # # For the spot 'spot19' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot19'     : 40,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot19' : 2460471.18+.51,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot19'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot19'    : 0.7,


                    # # For the spot 'spot20' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot20'     : 44,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot20' : 2460471.24+.51,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot20'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot20'    : 0.7,


                    # # For the spot 'spot21' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot21'     : 48,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot21' : 2460470.9+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot21'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot21'    : 0.7,


                    # # For the spot 'spot22' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot22'     : 60,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot22' : 2460470.4+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot22'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot22'    : 0.7,


                    # # For the spot 'spot23' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot23'     : 70,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot23' : 2460471.3+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot23'     : 3, # 10, #15
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot23'    : 0.7,


                    # # For the spot 'spot24' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot24'     : 85,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot24' : 2460470.3+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot24' : 3,
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot24'    : 0.7,


                    # # For the spot 'spot25' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot25'     : -20,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot25' : 2460471.7+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot25' : 3,
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot25'    : 0.7,


                    # # For the spot 'spot26' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot26'     : -10,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot26' : 2460471.4+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot26' : 3,
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot26'    : 0.7,

                    
                    # # For the spot 'spot27' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot27'     : -35,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot27' : 2460471.55+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot27' : 3,
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot27'    : 0.7,

                    
                    # # For the spot 'spot28' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot28'     : -42,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot28' : 2460472.75,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot28' : 3,
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot28'    : 0.7,


                    # # For the spot 'spot29' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot29'     : 85,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot29' : 2460470.3+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot29' : 3,
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot29'    : 0.7,

                    
                    # # For the spot 'spot30' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot30'     : 85,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot30' : 2460470.3+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot30' : 3,
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot30'    : 0.7,

                    
                    # # For the spot 'spot31' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot31'     : 85,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot31' : 2460470.3+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot31' : 3,
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot31'    : 0.7,

                    
                    # # For the spot 'spot32' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot32'     : 85,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot32' : 2460470.3+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot32' : 3,
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot32'    : 0.7,

                    
                    # # For the spot 'spot33' :
                    # 'lat__ISNIRPS_HE_VSmockvis_ARspot33'     : 85,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot33' : 2460470.3+1.5,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARspot33' : 3,
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARspot33'    : 0.7,

                    
                    # For the spot 'spot34' :
                    'lat__ISNIRPS_HE_VSmockvis_ARspot34'     : 85,
                    'Tc_ar__ISNIRPS_HE_VSmockvis_ARspot34' : 2460470.3+1.5,
                    'ang__ISNIRPS_HE_VSmockvis_ARspot34' : 3,
                    'fctrst__ISNIRPS_HE_VSmockvis_ARspot34'    : 0.7,

                    # For the facula 'facula1' : -- base grid run
                    # 'lat__ISNIRPS_HE_VSmockvis_ARfacula1'     : 30,
                    # 'Tc_ar__ISNIRPS_HE_VSmockvis_ARfacula1' : 2460473.15,
                    # 'ang__ISNIRPS_HE_VSmockvis_ARfacula1'     : 20.,
                    # 'fctrst__ISNIRPS_HE_VSmockvis_ARfacula1'    : 1.5,
                    },
                }
            }


    if gen_dic['star_name'] == 'AUMic': 
        mock_dic['ar_prop']={
             'ESPRESSO':{
                 'mock_vis':{

                     # # For the spot 'spot1' : - testing
                     # 'lat__ISESPRESSO_VSmock_vis_ARspot1'     : 30,
                     # 'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : 2458330.39051,
                     # 'ang__ISESPRESSO_VSmock_vis_ARspot1'     : 25,
                     # 'fctrst__ISESPRESSO_VSmock_vis_ARspot1'    : 0.6,

                     # # For the spot 'spot1' : 
                     # 'lat__ISESPRESSO_VSmock_vis_ARspot1'     : 30,
                     # 'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : 2458330.39051,
                     # 'ang__ISESPRESSO_VSmock_vis_ARspot1'     : 15,
                     # 'fctrst__ISESPRESSO_VSmock_vis_ARspot1'    : 0.2,
                    
                     # # #For the spot 'spot2' : 
                     # 'lat__ISESPRESSO_VSmock_vis_ARspot2'     : -20,
                     # 'Tc_ar__ISESPRESSO_VSmock_vis_ARspot2' : 2458330.39051 - 0.2,
                     # 'ang__ISESPRESSO_VSmock_vis_ARspot2'     : 25,
                     # 'fctrst__ISESPRESSO_VSmock_vis_ARspot2'    : 0.9,

                     # For the spot 'spot1' : -- plotting purposes
                     # 'lat__ISESPRESSO_VSmock_vis_ARspot1'     : 0,
                     # 'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : 2458330.39051,
                     # 'ang__ISESPRESSO_VSmock_vis_ARspot1'     : 25,
                     # 'fctrst__ISESPRESSO_VSmock_vis_ARspot1'    : 0.05,
                     

                     # #For the spot 'spot2' :  -- plotting purposes
                     # 'lat__ISESPRESSO_VSmock_vis_ARspot2'     : 15,
                     # 'Tc_ar__ISESPRESSO_VSmock_vis_ARspot2' : 2458330.39051 - 0.4,
                     # 'ang__ISESPRESSO_VSmock_vis_ARspot2'     : 25,
                     # 'fctrst__ISESPRESSO_VSmock_vis_ARspot2'    : 0.1,

                    # For the spot 'spot1' : -- base grid run
                     'lat__ISESPRESSO_VSmock_vis_ARspot1'     : -30,
                     'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : 2458330.39051 - 0.3,
                     'ang__ISESPRESSO_VSmock_vis_ARspot1'     : 15,
                    'fctrst__ISESPRESSO_VSmock_vis_ARspot1'    : 0.1,

                    # For the facula 'facula1' : -- base grid run
                    # 'lat__ISESPRESSO_VSmock_vis_ARfacula1'     : 0,
                    # 'Tc_ar__ISESPRESSO_VSmock_vis_ARfacula1' : 2458330.39051 - 0.3,
                    # 'ang__ISESPRESSO_VSmock_vis_ARfacula1'     : 25,
                    # 'fctrst__ISESPRESSO_VSmock_vis_ARfacula1'    : 1.5,
                    },
                }
            }

        # mock_dic['auto_gen_ar'] = {
        #     'ESPRESSO':{
        #              'mock_vis':{

        #                  'lat'     : {'distrib':'uf', 'low':-20, 'high':20},
        #                  'Tc_ar' : {'distrib':'uf', 'low':2458330.39051 - 0.8, 'high':2458330.39051 + 0.8},
        #                  'ang'     : {'distrib':'uf', 'low':10, 'high':20},
        #                 'fctrst'    : {'distrib':'uf', 'low':0.3, 'high':0.4},
        #                  'num': 2,
        #                 },
        #          }
        # }

    
    if gen_dic['star_name']=='fakeAU_Mic':
        mock_dic['ar_prop']={
             'ESPRESSO':{
                 'mockvisit1':{

                     # For the spot 'spot1' : 
                     'lat__ISESPRESSO_VSmockvisit1_ARspot1'     : -45,
                     'Tc_ar__ISESPRESSO_VSmockvisit1_ARspot1' : 2458702.26484,
                     'ang__ISESPRESSO_VSmockvisit1_ARspot1'     : 20, # 10, #15
                     'fctrst__ISESPRESSO_VSmockvisit1_ARspot1'    : 0.45,

                     # For the spot 'spot2' : 
                     'lat__ISESPRESSO_VSmockvisit1_ARspot2'     : 55,
                     'Tc_ar__ISESPRESSO_VSmockvisit1_ARspot2' : 2458702.76484, 
                     'ang__ISESPRESSO_VSmockvisit1_ARspot2'     : 12, #12
                     'fctrst__ISESPRESSO_VSmockvisit1_ARspot2'    : 0.45,
                         }
                    }
                }

    if gen_dic['star_name']=='V1298tau':
        mock_dic['ar_prop']={
             'ESPRESSO':{
                 'mock_vis':{

                     # For the spot 'spot1' : 
                     'lat__ISESPRESSO_VSmock_vis_ARspot1'     : -30,
                     'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : 2457067.0488 - 12/24,
                     'ang__ISESPRESSO_VSmock_vis_ARspot1'     : 20,
                     'fctrst__ISESPRESSO_VSmock_vis_ARspot1'    : 0.6,
                    
                     # For the spot 'spot2' : 
                     'lat__ISESPRESSO_VSmock_vis_ARspot2'     : 40,
                     'Tc_ar__ISESPRESSO_VSmock_vis_ARspot2' : 2457067.0488 + 5/24,
                     'ang__ISESPRESSO_VSmock_vis_ARspot2'     : 25,
                     'fctrst__ISESPRESSO_VSmock_vis_ARspot2'    : 0.4
                         }
                    }
                }

    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Taurus','Scorpio','Virgo','Pisces']:
        mock_dic['ar_prop']={
             'ESPRESSO':{
                 'mock_vis':{

                    # For the spot 'spot1' : -- base grid run
                     'lat__ISESPRESSO_VSmock_vis_ARspot1'     : -30,
                     'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : 2458330.39051 - 0.1,
                     'ang__ISESPRESSO_VSmock_vis_ARspot1'     : 15,
                    'fctrst__ISESPRESSO_VSmock_vis_ARspot1'    : 0.4,
                    },
                }
            }
    if gen_dic['star_name']=='Leo':
        mock_dic['ar_prop']={
             'ESPRESSO':{
                 'mock_vis':{

                    # For the spot 'spot1' : -- base grid run
                     'lat__ISESPRESSO_VSmock_vis_ARspot1'     : -30,
                     'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : 2458330.39051 - 0.1,
                     'ang__ISESPRESSO_VSmock_vis_ARspot1'     : 30,
                    'fctrst__ISESPRESSO_VSmock_vis_ARspot1'    : 0.4,
                    },
                }
            }
    if gen_dic['star_name']=='Aquarius':
        mock_dic['ar_prop']={
             'ESPRESSO':{
                 'mock_vis':{

                    # For the spot 'spot1' : -- base grid run
                     'lat__ISESPRESSO_VSmock_vis_ARspot1'     : -30,
                     'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : 2458330.39051 - 0.1,
                     'ang__ISESPRESSO_VSmock_vis_ARspot1'     : 5,
                    'fctrst__ISESPRESSO_VSmock_vis_ARspot1'    : 0.4,
                    },
                }
            }
    if gen_dic['star_name']=='Aries':
        mock_dic['ar_prop']={
             'ESPRESSO':{
                 'mock_vis':{

                    # For the spot 'spot1' : -- base grid run
                     'lat__ISESPRESSO_VSmock_vis_ARspot1'     : -30,
                     'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : 2458330.39051 - 0.1,
                     'ang__ISESPRESSO_VSmock_vis_ARspot1'     : 15,
                    'fctrst__ISESPRESSO_VSmock_vis_ARspot1'    : 0.05,
                    },
                }
            }
    if gen_dic['star_name']=='Libra':
        mock_dic['ar_prop']={
             'ESPRESSO':{
                 'mock_vis':{

                    # For the spot 'spot1' : -- base grid run
                     'lat__ISESPRESSO_VSmock_vis_ARspot1'     : -30,
                     'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : 2458330.39051 - 0.1,
                     'ang__ISESPRESSO_VSmock_vis_ARspot1'     : 15,
                    'fctrst__ISESPRESSO_VSmock_vis_ARspot1'    : 0.8,
                    },
                }
        }


    #Intrinsic stellar spectra
    if gen_dic['star_name'] == 'TOI3884' :
        mock_dic['intr_prof']={'MIKE_Red':{
            'mode':'ana',        
            'coord_line':'mu',
            'model': 'gauss',
            'line_trans':None, 
            'mod_prop':{'ctrst__ord0__IS__VS_' : 0.7,
                        'FWHM__ord0__IS__VS_'  : 8 },
            'pol_mode' : 'modul'}
            }

    if gen_dic['star_name'] == 'TRAPPIST1' :
        mock_dic['intr_prof']={'NIRPS_HE':{
            'mode':'ana',        
            'coord_line':'mu',
            'model': 'gauss',
            'line_trans':None, 
            'mod_prop':{'ctrst__ord0__IS__VS_' : 0.7,
                        'FWHM__ord0__IS__VS_'  : 8 },
            'pol_mode' : 'modul'}
            }

    if gen_dic['star_name'] == 'AUMic' :
        mock_dic['intr_prof']={'ESPRESSO':{
            'mode':'ana',        
            'coord_line':'mu',
            'model': 'gauss',
            'line_trans':None, 
            'mod_prop':{'ctrst__ord0__IS__VS_' : 0.7,
                        'FWHM__ord0__IS__VS_'  : 8 },
            'pol_mode' : 'modul'}
            }

    if gen_dic['star_name'] == 'fakeAU_Mic' :
        mock_dic['intr_prof']={'ESPRESSO':{
            'mode':'ana',        
            'coord_line':'mu',
            'model': 'gauss',
            'line_trans':None, 
            'mod_prop':{'ctrst__ord0__IS__VS_' : 0.66439240394,
                        'FWHM__ord0__IS__VS_'  : 1.9571368614},
            'pol_mode' : 'modul'}
            }

    if gen_dic['star_name'] =='V1298tau' :
        mock_dic['intr_prof']={'ESPRESSO':{
            'mode':'ana',        
            'coord_line':'r_proj',
            'model': 'gauss',             
            'mod_prop':{'ctrst__ord0__IS__VS_' : 0.7,
                        'FWHM__ord0__IS__VS_'  : 4,
                        
                        'amp_l2c__ISESPRESSO_VSmock_vis' : 0.1,
                        'RV_l2c__ISESPRESSO_VSmock_vis' : 0,
                        'FWHM_l2c__ISESPRESSO_VSmock_vis' : 4,
                        
                        'a_damp__ISESPRESSO_VSmock_vis' : 0.5,
                        'slope__ISESPRESSO_VSmock_vis' : 0,
                        },
            'pol_mode' : 'modul'}}  

    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        mock_dic['intr_prof']={'ESPRESSO':{
            'mode':'ana',        
            'coord_line':'mu',
            'model': 'gauss',
            'line_trans':None, 
            'mod_prop':{'ctrst__ord0__IS__VS_' : 0.7,
                        'FWHM__ord0__IS__VS_'  : 8 },
            'pol_mode' : 'modul'}
            }

    #Count continuum level
    if gen_dic['star_name'] == 'TOI3884' :
        mock_dic['flux_cont']={'MIKE_Red':{
        # 'mockvis':40., #--Libby-Roberts+2023
        'mockvis':40., #--Equatorial band hypothesis
        # 'mockvis':40., #--Almenara+2022
            }
        }
        mock_dic['verbose_flux_cont']= True

    if gen_dic['star_name'] == 'TRAPPIST1' :
        mock_dic['flux_cont']={'NIRPS_HE':{
        'mockvis':260., #--base
            }
        }   

    if gen_dic['star_name'] == 'AUMic' :
        mock_dic['flux_cont']={'ESPRESSO':{
        'mock_vis':1e8,
        'mock_vis1':1e8,
        'mock_vis2':1e8,
        'mock_vis3':1e8,
        'mock_vis4':1e8,
        'mock_vis5':1e8,
        'mock_vis6':1e8,
        'mock_vis7':1e8,
        'mock_vis8':1e8,
        'mock_vis9':1e8,
            }
        }
        mock_dic['verbose_flux_cont']= True & False  
            
    if gen_dic['star_name'] == 'V1298tau' :
        mock_dic['flux_cont']={'ESPRESSO':{'mock_vis':1e5}}   
    
    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        mock_dic['flux_cont']={'ESPRESSO':{'mock_vis':80.}}
        mock_dic['verbose_flux_cont']= True 

    if gen_dic['star_name'] == 'fakeAU_Mic' :
        mock_dic['flux_cont']={'ESPRESSO':{'mockvisit1':1e8}}

    #Noise settings
    
    #Instrumental calibration  
    if gen_dic['star_name'] == 'TOI3884' : 
        mock_dic['gcal'] = {'MIKE_Red' : 1.}

    if gen_dic['star_name'] == 'TRAPPIST1' : 
        mock_dic['gcal'] = {'NIRPS_HE' : 1.}

    if gen_dic['star_name'] == 'AUMic' : 
        mock_dic['gcal'] = {'ESPRESSO' : 1.}   

    if gen_dic['star_name'] == 'fakeAU_Mic' : 
        mock_dic['gcal'] = {'ESPRESSO' : 1.} 

    if gen_dic['star_name'] == 'V1298tau' : 
        mock_dic['gcal'] = {'ESPRESSO' : 1.}  

    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        mock_dic['gcal'] = {'ESPRESSO' : 1.}   

    #Flux errors
    if gen_dic['star_name'] == 'TOI3884': mock_dic['set_err']={'MIKE_Red':True}
    if gen_dic['star_name'] == 'TRAPPIST1': mock_dic['set_err']={'NIRPS_HE':True}
    if gen_dic['star_name'] == 'AUMic': mock_dic['set_err']={'ESPRESSO':True}
    if gen_dic['star_name'] == 'fakeAU_Mic': mock_dic['set_err']={'ESPRESSO':True}
    if gen_dic['star_name'] == 'V1298tau': mock_dic['set_err']={'ESPRESSO':True}
    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:mock_dic['set_err']={'ESPRESSO':True}


    #Jitter on intrinsic profile properties
       
    
    #Systematic variations on disk-integrated profiles
                 
                
                
                
            
    
    
    
    
    ##################################################################################################
    #%%% Settings: observational datasets
    ##################################################################################################
    
    #%%%%% Paths to data directories
    #    - data must be stored in a unique directory for each instrument, and unique sub-directories for each instrument visit
    #    - the fields defined here will determine which instruments/visits are processed, and which names are used for each visit 
    #    - format: {inst:{vis:path}}
    gen_dic['data_dir_list']={'ESPRESSO':{'20151021':'default_path_TBD'}}

    
    #%%%% Saving log of useful keywords
    gen_dic['sav_keywords']=  False  


    #%%%% Activity indexes
    #    - retrieving activity indexes from DACE if target and data are available
    gen_dic['DACE_sp'] = False
    
    
    #%%%% Using fiber-B corrected data
    #    - if available
    #    - for each visit set the field to 'all' for all orders to be replaced by their sky-corrected version, or by a list of orders otherwise
    #      leave empty to use fiber-A data
    #    - format: {inst:{vis:[iord] or 'all'}} where iord are original order indexes
    gen_dic['fibB_corr']={}
        
    
    #%%%% Data exclusion
    
    #%%%%% Visits to remove from analysis
    #    - leave empty to use everything
    gen_dic['unused_visits']={}
         
    
    #%%%%% Exposures to keep in analysis
    #    - leave instrument / visit undefined to use everything
    #    - otherwise indicate for each visit the indexes of exposures (in order of increasing time) to be kept 
    gen_dic['used_exp']={}
    
    
    #%%%%% Orders to remove from analysis
    #    - applied to 2D spectral datasets only
    #    - orders will not be uploaded from the dataframes and are thus excluded from all operations
    #    - apply this option only if the order is too contaminated to be exploited by the reduction steps, or undefined
    #    - order is removed in all visits of a given instrument to keep a common structure
    gen_dic['del_orders'] = {}
    
    
    #%%%%% Spectral ranges to remove from analysis
    #    - permanently set masked pixels to nan in the rest of the processing
    #    - masking is done prior to any other operation
    #    - set ranges of masked pixels in the spectral format of the input data, in the following format
    # inst > vis > { exp_list : [ iexp0, iexp1, ... ] ; ord_list : {iord0 : [ [x1,x2] , [x3,x4] .. ], ... }}
    #      exposures index relate to time-ordered tables after 'used_exp' has been applied, leave empty to mask all exposures 
    #      order indexes are relative to the effective order list, after orders are possibly excluded
    #      define position xk in A in the input rest frame
    #    - only relevant if part of the order is masked, otherwise remove the full order 
    #    - if several visits of a given instrument are processed together, it is advised to exclude the minimum ranges common to all of them so that CCF are comparable
    #    - plot flux and transmission spectra after telluric and flux balance corrections to identify the ranges to exclude 
    gen_dic['masked_pix'] = {}
    
         
    #%%%%% Bad quality pixels
    #    - setting bad quality pixels to undefined
    #    - beware that undefined pixels will 'bleed' on adjacent pixels in successive resamplings
    gen_dic['bad2nan'] = False
    
    #%%%%% Paths to data directories
    if gen_dic['star_name']=='AU_Mic':
        gen_dic['data_dir_list']={'ESPRESSO':{'visit1':'/Users/samsonmercier/Desktop/Work/Master/2023-2024/AUMic_data/2019-08-06'}}
        # gen_dic['fibB_corr']={'ESPRESSO':{'visit1':'all'}}

    if gen_dic['star_name']=='HD189733':
        if gen_dic['type']['ESPRESSO']=='CCF':
            gen_dic['data_dir_list']={'ESPRESSO':{'visit1':'/Users/samsonmercier/Desktop/Work/Master/2023-2024/HD189733_data/HD189733_20210830_ESPRESSO_KitCat_CCF'}}
        elif gen_dic['type']['ESPRESSO']=='spec2D':            
            gen_dic['data_dir_list']={'ESPRESSO':{'visit1':'/Users/samsonmercier/Desktop/Work/Master/2023-2024/HD189733_data/ESPRESSO_data/DRS3.2.5/20210830'}}
        # gen_dic['fibB_corr']={'ESPRESSO':{'visit1':'all'}}

    #%%%%% Spectral ranges to remove from analysis
    if gen_dic['star_name']=='HD189733':
        #Order check based on flux balance correction + telluric correction
        # 128,129,130,131: 6275,6316   : DO NOT EXCLUDE AT THIS STAGE, strong telluric lines used for telluric fit range between 6278.329987095841 and 6316.724823709858
        # 146, 147, 148, 149: 6865, 6965
        # 152, 153,154, 155, 156, 157: 7165, 7320   : DO NOT EXCLUDE AT THIS STAGE, strong telluric lines used for telluric fit range between 7178.50262359 and 7312.39401687
        # 162, 163: 7592 , 7668
        # 164, 165: exclure 7500 , 7668 ; 7670, 7672 ; 7675.5 , 7678 ; 7682 , 7685 ; 7688.5 , 7691.5 ; 7695 , 7698 ; 7702 - 7705.
        # 166, 167: exclure 7600 - 7705
        gen_dic['masked_pix'] = {'ESPRESSO':{'visit1':{'exp_list':[],'ord_list':{
            #128: [[6275,6316]],130:[[6275,6316]],
            146: [[6865,6965]],148:[[6865, 6965]],
            #152: [[7165,7320]],154:[[7165, 7320]],156:[[7165, 7320]],
            162: [[7592,7668]],
            164: [[7500,7668],[7670, 7672 ],[ 7675.5 , 7678 ],[ 7682 , 7685 ],[ 7688.5 , 7691.5 ],[ 7695 , 7698 ],[ 7702 , 7705.]],
            166: [[7600,7705]]}}}}
        for vis in gen_dic['masked_pix']['ESPRESSO']:
            for iord in [146,148,162,164,166]:gen_dic['masked_pix']['ESPRESSO'][vis]['ord_list'][iord+1]=gen_dic['masked_pix']['ESPRESSO'][vis]['ord_list'][iord]

    #---------------------------------------------------------------------------------------------
    #%%%% Weighing settings 
    #    - controls the weight profiles used for temporal/spatial resampling:
    # + mean calibration profile (choice to use calculated profile)
    # + telluric profile (choice to use input/calculated profiles)
    # + master stellar spectrum (calculation/retrieval and choice to use)
    #---------------------------------------------------------------------------------------------
    
    #%%%%% Using instrumental calibration
    gen_dic['cal_weight'] = True    
    
    
    #%%%%% Using telluric spectra
    #    - if available from input files, or from telluric correction module
    gen_dic['tell_weight'] = True   
    
    
    #%%%%% Master stellar spectrum
    #    - not calculated if weighing not required
    
    #%%%%%% Calculating/retrieving
    #    - master stellar spectrum for weighing, specific to each visit
    #    - calculated after alignment and broadband flux scaling
    gen_dic['calc_DImast'] = True   &   False
    
    
    #%%%%%% Exposures to be binned
    #    - indexes of exposures that contribute to the bin series, for each instrument/visit
    #    - indexes are relative to the global table in each visit
    #    - leave empty to use all out-exposures 
    gen_dic['DImast_idx_in_bin']={}
    
    
    #%%%%%% Using stellar spectrum  
    gen_dic['DImast_weight'] = True & False
    if gen_dic['star_name']=='HD189733':gen_dic['DImast_weight'] = True
    
    
    #%%%%%% Plots: weighing master 
    #    - the master is plotted after first calculation (ie before undergoing the same processing as the dataset) 
    plot_dic['DImast']=''       

    
    
    ##################################################################################################
    #%%% Module: stellar continuum
    #    - continuum of the disk-integrated or intrinsic stellar spectrum in the star or surface rest frame
    #    - the CCF mask generation (gen_dic['def_DImasks']) and spectral detrending (gen_dic['detrend_prof']) require the stellar continuum to have been estimated with this module
    #    - the stellar continuum is calculated internally to the persistent peak masking module (gen_dic['mask_permpeak']), if activated, using the settings defined here
    ##################################################################################################
    
    #%%%%% Activating
    gen_dic['DI_stcont'] = False
    gen_dic['Intr_stcont'] = False  
    
    
    #%%%%% Calculating/retrieving 
    #    - calculated on disk-integrated or intrinsic stellar spectra, if a single 1D binned master has been calculated
    gen_dic['calc_DI_stcont'] = True
    gen_dic['calc_Intr_stcont'] = True    
    
    
    #%%%%% Rolling window for peak exclusion
    #    - set to 2 A by default
    gen_dic['contin_roll_win']={}
        
    
    #%%%%% Smoothing window
    #    - set to 0.5 A by default
    gen_dic['contin_smooth_win']={} 
    
    
    #%%%%% Local maxima window
    #    - set to 0.5 A by default
    gen_dic['contin_locmax_win']={}
    
    
    #%%%%% Flux/wavelength stretching
    #    - set to 10 by default (>=1)
    #    - adjust so that maxima are well selected at the top of the spectra
    gen_dic['contin_stretch']={}
    
    
    #%%%%% Rolling pin radius
    #    - value corresponds to the bluest wavelength of the processed spectra
    gen_dic['contin_pinR']={}  
    
    
    #Activating
    if gen_dic['star_name']=='HD189733':
        gen_dic['DI_stcont'] = True #& False
        gen_dic['calc_DI_stcont'] = True #& False

    #%%%%% Rolling window for peak exclusion
    if gen_dic['star_name']=='HD189733':
        gen_dic['contin_roll_win'] = {'ESPRESSO':2} 

    #%%%%% Smoothing window
    if gen_dic['star_name']=='HD189733':
        gen_dic['contin_smooth_win'] = {'ESPRESSO':0.3}    

    #%%%%% Flux/wavelength stretching
    #Test DI CCF masks ESPRESSO : 5 captures less well the continuum structure than 15; 25 works better for some targets; 30 captures too finely
    if gen_dic['star_name']=='HD189733':
        gen_dic['contin_stretch'] = {'ESPRESSO':25} 
    
    #%%%%% Rolling pin radius
    if gen_dic['star_name']=='HD189733':
        gen_dic['contin_pinR'] = {'ESPRESSO':10}     #DI CCF masks

    ##################################################################################################
    #%%% Module: stellar, active region, and planet-occulted grids
    ##################################################################################################
    
    #%%%% Activating module
    #    - calculated by default if transiting planets are attributed to a visit, and user has requested analysis and alignement of intrinsic local stellar profiles, or extraction and analysis of the atmospheric profiles
    #                            if active regions are attributed to a visit
    #      can be set to True to calculate nonetheless
    gen_dic['theoPlOcc'] = False 
        
    
    #%%%% Calculating/retrieving
    gen_dic['calc_theoPlOcc']=True  
    
    
    #%%%% Star
    
    #%%%%% Discretization
    #    - number of subcells along the star diameter for model fits
    #    - must be an odd number
    #    - used (if model relevant) in gen_dic['fit_DI']
    #    - format: : value
    theo_dic['nsub_Dstar']=101       
    
            
    #%%%%% Macroturbulence
    #    - for the broadening of analytical intrinsic line profile models
    #    - set to None, or to 'rt' (Radial–tangential macroturbulence) or 'anigauss' (anisotropic Gaussian macroturbulence)
    #      add '_iso' in the name of the chosen mode to force isotropy
    #      the chosen mode will then be used througout the entire pipeline 
    #      default macroturbulence properties must be defined in the star properties to calculate default values, but can be fitted in the stellar line models
    theo_dic['mac_mode'] = None
    
    
    #%%%%% Theoretical atmosphere
    #    - this generates a nominal atmospheric structure that can be used as is, or adjusted for fitting, to then generate a series of local intrinsic spectra
    #    - the series is calculated over the spectral grid and mu series defined here, common for the whole processing
    #      the mu grid must be fine enough to be interpolated over all mu in the stellar grid
    #    - settings:
    # + atmosphere model: define the model use to generate the spectra
    #    > a set of models are available at https://pysme-astro.readthedocs.io/en/latest/usage/lfs.html#lfs 
    #      spherical (s) models are only available for low log g; use the plane-parallel (p) model ‘marcs2012p_tvmic.0.sav’ (vmic = 0, 1, 2 km/s), which generally suits F-K type stars 
    #      for M dwarfs use 'marcs2012tvmiccooldwarfs.sav' (vmic = 00, 01, 02 km/s) 
    #    > to add specific models, create an account at https://marcs.astro.uu.se/
    #      retrieve the desired model and put it in ~/.sme/atmospheres/
    # + NLTE: set pre-computed grids of NLTE departure coefficients for specific species 
    #    > to add specific models, copy the new grid manually using cp /Users/Download_grid_directory/nlte_X_pysme.grd ./.sme/nlte_grids/nlte_X_pysme.grd   
    #      a variety of grids can be found at https://zenodo.org/record/7088951#.ZFObQS9yr0o
    #    > the NLTE species must have transitions in the processed spectral band  
    # + spectral grid: define 'wav_min', 'wav_max' and 'dwav' in A, in the star rest frame
    # + mu grid: define array 'mu_grid' of mu coordinates
    #            the resolution in mu has little impact on computing time
    # + linelist: indicate the path 'linelist' of the linelist generated from the VALD database
    #             connect into VALD at http://vald.astro.uu.se/ with your registered email address (you will need to ask for an account using their Contact form)
    #             choose 'Extract Stellar', and define:
    #    > start and end wavelength (A, vacuum): should be wide enough to contain all transitions in the simulated band (it can be larger and is automatically cropped to the simulated spectral range)
    #    > line detection threshold: typically set to 0.1 
    #    > microturbulence (km/s): must be consistent with stellar value
    #    > Teff (K): must be consistent with stellar value
    #    > log g (g in cgs): must be consistent with stellar value
    #    > chemical composition: define individual abundances ('species') or overall metallicity ('M/H' consistent with stellar value) 
    #    > generate in 'long' format if NLTE is required
    #             beware to provide the linelist in the same frame as gen_dic['sp_frame'] 
    # + abundances: set by default to solar (Asplund+2009)
    #    > to change a specific abundance define 'abund':{'X':A(X)}, where A(X) = log10( N(X)/N(H) ) + 12
    #    > to change the overall metallicity define 'MovH', where A(X) = Anominal(X) + [M/H] for X != H and He   
    # + calc: calculate/retrieve grid 
    theo_dic['st_atm']={
        'atm_model':'marcs2012p_t1.0',        
        'nlte':{},        
        'wav_min':5600.,'wav_max':6200.,'dwav':0.01,
        'mu_grid':np.logspace(-2.,0.,15),
        'linelist': '',
        'abund':{},
        'calc':True,
        }  


    #%%%% Planets
    
    #%%%%% Discretization        
    #    - number of subcells along a planet diameter to define the grid of subcells discretizing the stellar regions it occults 
    #    - used for calculations of theoretical properties from planet-occulted regions, and for simulated light curves
    #    - beware to use a fine enough grid, depending on the system and dataset
    #    - must be an odd number
    #    - set to default value if undefined
    #    - format: {'planet':value}
    theo_dic['nsub_Dpl']={} 
    
    
    #%%%%% Exposure oversampling
    #    - oversampling factor of the observed exposures to calculate theoretical properties of planet-occulted regions in the entire pipeline
    #    - distance from start to end of exposure will be sampled by RpRs/n_oversamp
    #    - set to 0 or leave undefined to prevent oversampling, but beware that it must be defined to bin profiles over other dimensions than phase
    #    - oversampling of the flux in the flux scaling module is controlled independently
    #    - format: {'planet':value}
    theo_dic['n_oversamp']={}  
    
    
    #%%%% Occulted profiles
    
    #%%%%%% Multi-threading for single-line analysis
    #    - set to 1 to prevent
    #    - used for profile fits
    gen_dic['fit_prof_nthreads'] = int(0.8*cpu_count()) 

    
    #%%%%% Precision
    #    - precision at which planet-occulted profiles are computed for each exposure:
    # + 'low' : line properties are calculated as the flux-weighted average of planet-occulted stellar cells, cumulated over the oversampled planet positions
    #           a single planet-occulted profile is then calculated with the average line properties
    # + 'medium' : line properties are calculated as the flux-weighted average of planet-occulted stellar cells, for each oversampled planet position
    #              line profiles are then calculated for each oversampled planet position, and averaged into a single profile
    # + 'high' : line profiles are calculated for each planet-occulted stellar cells, summed for each oversampled planet position, and averaged into a single line profile
    theo_dic['precision'] = 'high'
    
    
    #%%%%% Oversampling 
    #    - for all analytical line models in the pipeline
    #    - in km/s (profiles in wavelength space are modelled in RV space and then converted) 
    #    - can be relevant to fit profiles measured at low resolution and for fast rotators
    #    - set to None for no oversampling
    theo_dic['rv_osamp_line_mod']=None


    #%%%% Active regions

    #%%%%% Nominal properties
    #    - same as mock_dic['ar_prop']
    #    - required for the calculation of nominal active region coordinates used throughout the pipeline
    theo_dic['ar_prop']={}


    #%%%%% Discretization     
    #    - format : {ar : val}} 
    # where each simulated active region must be associated with a unique name
    theo_dic['nsub_Dar']={} 


    #%%%%% Exposure oversampling     
    #    - format : {ar : val}} 
    # where each simulated active region must be associated with a unique name
    theo_dic['n_oversamp_ar']={}  
    
    
    #%%%% Plot settings
    
    #%%%%% Planetary orbit discretization
    #    - number of points discretizing the planetary orbits in plots
    plot_dic['npts_orbit'] = 10000
    
    
    #%%%%% Contact determination
    #    - start/end phase in fraction of ingress/egress phase
    plot_dic['stend_ph'] = 1.3
    
    
    #%%%%% Transit chord discretization        
    #    - number of points discretizing models of local stellar properties along the transit chord
    plot_dic['nph_HR'] = 2000
    
    
    #%%%%% Range of planet-occulted properties
    #    - calculated numerically
    #    - enable oversampling to have accurate ranges
    plot_dic['plocc_ranges']=''    
    
    
    #%%%%% Planet-occulted stellar regions
    plot_dic['occulted_regions']=''
    
    
    #%%%%% Planetary system architecture
    plot_dic['system_view']=''  
        
    #Activating module
    gen_dic['theoPlOcc'] = True #  &  False

    #Calculating/retrieving
    gen_dic['calc_theoPlOcc']=True  # &  False  

    # Precision
    theo_dic['precision'] = 'high'
    # theo_dic['precision'] = 'medium'
    # theo_dic['precision'] = 'low'



    #Star discretization  
    if gen_dic['star_name']=='HD189733':
        theo_dic['nsub_Dstar']=111.

    if gen_dic['star_name']=='TOI3884':
        # theo_dic['nsub_Dstar']=81.
        theo_dic['nsub_Dstar']=201. #211. #-- for fitting purposes
        # theo_dic['nsub_Dstar']=301. #-- for plotting purposes

    if gen_dic['star_name']=='TRAPPIST1':
        # theo_dic['nsub_Dstar']=81.
        theo_dic['nsub_Dstar']=101. #211. #-- for fitting purposes
        # theo_dic['nsub_Dstar']=301. #-- for plotting purposes

    if gen_dic['star_name']=='AUMic':
        # theo_dic['nsub_Dstar']=81.
        theo_dic['nsub_Dstar']=101. #211. #-- for fitting purposes
        # theo_dic['nsub_Dstar']=301. #-- for plotting purposes
    
    if gen_dic['star_name']=='V1298tau':
        theo_dic['nsub_Dstar']=201.
            
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        # theo_dic['nsub_Dstar']=5.
        # theo_dic['nsub_Dstar']=11.
        # theo_dic['nsub_Dstar']=21.
        theo_dic['nsub_Dstar']=37. #<- Sagittarius optimal
        # theo_dic['nsub_Dstar']=41.
        # theo_dic['nsub_Dstar']=61.
        # theo_dic['nsub_Dstar']=81.
        # theo_dic['nsub_Dstar']=101.
        # theo_dic['nsub_Dstar']=121.
        # theo_dic['nsub_Dstar']=141.
        # theo_dic['nsub_Dstar']=181.
        # theo_dic['nsub_Dstar']=201.
        # theo_dic['nsub_Dstar']=251.

    if gen_dic['star_name']=='AU_Mic':
        theo_dic['nsub_Dstar']=111.

    if gen_dic['star_name']=='fakeAU_Mic':
        theo_dic['nsub_Dstar']=111.

    #Stellar macroturbulence
    theo_dic['mac_mode'] = None



    #Theoretical stellar atmosphere
    if gen_dic['star_name']=='HD189733':
        theo_dic['st_atm']['calc']=False

    if gen_dic['star_name']=='TOI3884':
        theo_dic['st_atm']['calc']=False

    if gen_dic['star_name']=='TRAPPIST1':
        theo_dic['st_atm']['calc']=False

    if gen_dic['star_name']=='AUMic':
        theo_dic['st_atm']['calc']=False
    
    if gen_dic['star_name']=='V1298tau':
        theo_dic['st_atm']['calc']=False
    
    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        theo_dic['st_atm']['calc']=False

    if gen_dic['star_name']=='AU_Mic':
        theo_dic['st_atm']['calc']=False

    if gen_dic['star_name']=='fakeAU_Mic':
        theo_dic['st_atm']['calc']=False    

    #Planet discretization
    if gen_dic['star_name']=='TOI3884':
        theo_dic['nsub_Dpl']= {'TOI3884_b':71.}

    if gen_dic['star_name']=='TRAPPIST1':
        theo_dic['nsub_Dpl']= {'TRAPPIST1_b':31.,
                                'TRAPPIST1_c':31.,
                                'TRAPPIST1_d':31.,
                                'TRAPPIST1_e':31.,
                                'TRAPPIST1_f':31.,
                                'TRAPPIST1_g':31.,
                                'TRAPPIST1_h':31.}


    if gen_dic['star_name']=='AUMic':
        theo_dic['nsub_Dpl']= {'AUMicb':31.} #33.}#, 'AUMicc':101.} #-- for fitting purposes 
        # theo_dic['nsub_Dpl']= {'AUMicb':23.}           

    if gen_dic['star_name']=='HD189733':
        theo_dic['nsub_Dpl']= {'HD189733b':51.}

    if gen_dic['star_name']=='AU_Mic':
        theo_dic['nsub_Dpl']= {'AU_Mic_b':33.}

    if gen_dic['star_name']=='fakeAU_Mic':
        theo_dic['nsub_Dpl']= {'fakeAU_Mic_b':33.}

    if gen_dic['star_name']=='V1298tau':
        theo_dic['nsub_Dpl']={'V1298tau_b':101.}

    #Zodiacs
    for zodiac in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        if gen_dic['star_name']==zodiac:
            # theo_dic['nsub_Dpl']= {zodiac_pl:5.}
            # theo_dic['nsub_Dpl']= {zodiac_pl:11.} #-- for fitting purposes
            # theo_dic['nsub_Dpl']= {zodiac_pl:21.}
            # theo_dic['nsub_Dpl']= {zodiac_pl:35.}
            theo_dic['nsub_Dpl']= {zodiac_pl:33.} #<- Sagittarius optimal
            # theo_dic['nsub_Dpl']= {zodiac_pl:51.}
            # theo_dic['nsub_Dpl']= {zodiac_pl:61.}
            # theo_dic['nsub_Dpl']= {zodiac_pl:81.}
            # theo_dic['nsub_Dpl']= {zodiac_pl:101.}
            # theo_dic['nsub_Dpl']= {zodiac_pl:125.}
            # theo_dic['nsub_Dpl']= {zodiac_pl:151.}
            # theo_dic['nsub_Dpl']= {zodiac_pl:201.}


    #Active region discretization
    if gen_dic['star_name']=='TOI3884':
        theo_dic['nsub_Dar']={'spot1':101., 'spot2':31., 'spot3':31.,
                              # 'facula1':31., 'facula2':31.,
                             }

    if gen_dic['star_name']=='HD189733':
        theo_dic['nsub_Dar']={'spot1':101.}

    if gen_dic['star_name']=='TRAPPIST1':
        theo_dic['nsub_Dar']={'spot1':31.,
                              'spot2':31.,
                              # 'spot3':31.,
                              # 'spot4':31.,
                              # 'spot5':31.,
                              # 'spot6':31.,
                              # 'spot7':31.,
                              # 'spot8':31.,
                              # 'spot9':31.,
                              # 'spot10':31.,
                              # 'spot11':31.,
                              # 'spot12':31.,
                              # 'spot13':31.,
                              # 'spot14':31.,
                              # 'spot15':31.,
                              # 'spot16':31.,
                              # 'spot17':31.,
                              # 'spot18':31.,
                              # 'spot19':31.,
                              # 'spot20':31.,
                              # 'spot21':31.,
                              # 'spot22':31.,
                              # 'spot23':31.,
                              # 'spot24':31.,
                              'facula1':31.
                              }

    if gen_dic['star_name']=='AUMic':
        theo_dic['nsub_Dar']={ #'spot1':50., 'spot2':50.}
                              'spot1':31., #33.} #-- for fitting purposes
                              # 'spot1':51.,
                              'facula1':31.} #-- for fitting purposes

    if gen_dic['star_name']=='AU_Mic':
        theo_dic['nsub_Dar']={'spot1':33., 'spot2':33.}

    if gen_dic['star_name']=='fakeAU_Mic':
        theo_dic['nsub_Dar']={'spot1':33., 'spot2':33.}

    if gen_dic['star_name']=='V1298tau':
        theo_dic['nsub_Dar']={'spot1':50., 'spot2':50.}
    
    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        # theo_dic['nsub_Dar']={'spot1':5.}
        # theo_dic['nsub_Dar']={'spot1':11.} #-- for fitting purposes
        # theo_dic['nsub_Dar']={'spot1':21.}
        # theo_dic['nsub_Dar']={'spot1':35.}
        theo_dic['nsub_Dar']= {'spot1':33.} #<- Sagittarius optimal
        # theo_dic['nsub_Dar']={'spot1':51.}
        # theo_dic['nsub_Dar']={'spot1':61.}
        # theo_dic['nsub_Dar']={'spot1':81.}
        # theo_dic['nsub_Dar']={'spot1':101.}
        # theo_dic['nsub_Dar']={'spot1':125.}
        # theo_dic['nsub_Dar']={'spot1':151.}
        # theo_dic['nsub_Dar']={'spot1':201.}

    #Exposure discretization
    if gen_dic['star_name']=='TOI3884':
        theo_dic['n_oversamp']={'TOI3884_b':5.}#, 'AUMicc': 5.}
        theo_dic['n_oversamp_ar']={
                                        'spot1':5.
                                        #'facula1':5.,'facula2':5.,
                                      }

    if gen_dic['star_name']=='TRAPPIST1':
        theo_dic['n_oversamp']={'TRAPPIST1_b':5.,
                                'TRAPPIST1_c':5.,
                                'TRAPPIST1_d':5.,
                                'TRAPPIST1_e':5.,
                                'TRAPPIST1_f':5.,
                                'TRAPPIST1_g':5.,
                                'TRAPPIST1_h':5.
                                }
        theo_dic['n_oversamp_ar']={'spot1':5.,
                                    'spot2':5.,
                                    # 'spot3':5.,
                                    # 'spot4':5.,
                                    # 'spot5':5.,
                                    # 'spot6':5.,
                                    # 'spot7':5.,
                                    # 'spot8':5.,
                                    # 'spot9':5.,
                                    # 'spot10':5.,
                                    # 'spot11':5.,
                                    # 'spot12':5.,
                                    # 'spot13':5.,
                                    # 'spot14':5.,
                                    # 'spot15':5.,
                                    # 'spot16':5.,
                                    # 'spot17':5.,
                                    # 'spot18':5.,
                                    # 'spot19':5.,
                                    # 'spot20':5.,
                                    # 'spot21':5.,
                                    # 'spot22':5.,
                                    # 'spot23':5.,
                                    # 'spot24':5.,
                                    'facula1':5.,
                                    }


    if gen_dic['star_name']=='AUMic':
        theo_dic['n_oversamp']={'AUMicb':5.}#, 'AUMicc': 5.}
        theo_dic['n_oversamp_ar']={ #'spot1':5., 'spot2':5.}
                                       'spot1':5.,
                                       # 'facula1':5.,
                                       }
    
    if gen_dic['star_name']=='AU_Mic':
        theo_dic['n_oversamp']={'AU_Mic_b':5.}
        theo_dic['n_oversamp_ar']={'spot1':5., 'spot2':5.}

    if gen_dic['star_name']=='HD189733':
        theo_dic['n_oversamp']={'HD189733b':5.}
        theo_dic['n_oversamp_ar']={'spot1':5.}

    if gen_dic['star_name']=='fakeAU_Mic':
        theo_dic['n_oversamp']={'fakeAU_Mic_b':5.}
        theo_dic['n_oversamp_ar']={'spot1':5., 'spot2':5.}

    if gen_dic['star_name']=='V1298tau':
        theo_dic['n_oversamp'] = {'V1298tau_b':5.}
        theo_dic['n_oversamp_ar']={'spot1':5., 'spot2':5.}

    #Zodiacs
    for zodiac in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        if gen_dic['star_name']==zodiac:
            theo_dic['n_oversamp']={zodiac_pl:5.}
            theo_dic['n_oversamp_ar']={'spot1':5.}
    
    #RV table        

    # #Oversampling 
    # theo_dic['rv_osamp_line_mod']=0.5 #None  #1.
    # print('ATTENTION rv_osamp_line_mod')

    if gen_dic['star_name']=='AU_Mic':
        theo_dic['ar_prop']={
        'ESPRESSO':{
                 'visit1':{

                    # For the spot 'spot1' : -- base grid run
                    'lat__ISESPRESSO_VSvisit1_ARspot1'     : 0,
                    'Tc_ar__ISESPRESSO_VSvisit1_ARspot1' : 2458702.76484-0.8,
                    'ang__ISESPRESSO_VSvisit1_ARspot1'     : 10,
                    'fctrst__ISESPRESSO_VSvisit1_ARspot1'    : 0.65,

                    # For the spot 'spot2' : -- base grid run
                    'lat__ISESPRESSO_VSvisit1_ARspot2'     : -10,
                    'Tc_ar__ISESPRESSO_VSvisit1_ARspot2' : 2458702.76484,
                    'ang__ISESPRESSO_VSvisit1_ARspot2'     : 14,
                    'fctrst__ISESPRESSO_VSvisit1_ARspot2'    : 0.65,
                    },
                }
        }

    # if gen_dic['star_name']=='HD189733':
    #     theo_dic['ar_prop']={
    #     'ESPRESSO':{
    #              'visit1':{

    #                 # For the spot 'spot1' : -- base grid run
    #                 'lat__ISESPRESSO_VSvisit1_ARspot1'     : 0,
    #                 'Tc_ar__ISESPRESSO_VSvisit1_ARspot1' : 2459457.589323-0.2,
    #                 'ang__ISESPRESSO_VSvisit1_ARspot1'     : 10,
    #                 'fctrst__ISESPRESSO_VSvisit1_ARspot1'    : 0.6,
    #                 },
    #             }
    #     }

    if gen_dic['star_name'] in ['TRAPPIST1','AUMic','fakeAU_Mic','TOI3884']:
        theo_dic['ar_prop'] = mock_dic['ar_prop']

    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        theo_dic['ar_prop'] = mock_dic['ar_prop']


    #Plot settings

    #Planetary orbit discretization
    plot_dic['npts_orbit'] = 10000

    #Contact determination
    # if (gen_dic['star_name']=='HD209458') and gen_dic['mock_data']:    plot_dic['stend_ph'] = 2.      #Plot multi-pl

    #Transit chord discretization        
    # if gen_dic['star_name']=='MASCARA1':plot_dic['nph_HR'] = 100

    #Range of planet-occulted properties
    plot_dic['plocc_ranges']=''    
    
    #Planet-occulted stellar regions
    plot_dic['occulted_regions']=''
    
    #Planetary system architecture
    plot_dic['system_view']=''   #png

    if gen_dic['star_name'] in ['fakeAU_Mic','AUMic','TRAPPIST1','TOI3884','HD189733']:
        #Range of planet-occulted properties
        plot_dic['plocc_ranges']=''    
        
        #Planet-occulted stellar regions
        plot_dic['occulted_regions']='png'
        
        #Planetary system architecture
        plot_dic['system_view']='png'   #png

        #Transit chord discretization        
        plot_dic['nph_HR'] = 100
    
    if gen_dic['star_name']=='AU_Mic':
        #Range of planet-occulted properties
        plot_dic['plocc_ranges']=''    
        
        #Planet-occulted stellar regions
        plot_dic['occulted_regions']='png'
        
        #Planetary system architecture
        plot_dic['system_view']='png'   #png

        #Transit chord discretization        
        plot_dic['nph_HR'] = 100

    if gen_dic['star_name']=='V1298tau':
        #Range of planet-occulted properties
        plot_dic['plocc_ranges']=''    
        
        #Planet-occulted stellar regions
        plot_dic['occulted_regions']='png'
        
        #Planetary system architecture
        plot_dic['system_view']='png'   #png

        #Transit chord discretization        
        plot_dic['nph_HR'] = 100
    
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        #Range of planet-occulted properties
        plot_dic['plocc_ranges']=''    
        
        #Planet-occulted stellar regions
        plot_dic['occulted_regions']='png'
        
        #Planetary system architecture
        plot_dic['system_view']='png'   #png

        #Transit chord discretization        
        plot_dic['nph_HR'] = 100    
    
    
    
    
    
    ##################################################################################################
    #%% Global spectral corrections
    ##################################################################################################
    
    #%%% Plot settings
    
    #%%%% Individual disk-integrated flux spectra
    #    - before/after the various spectral corrections for disk-integrated data
    plot_dic['flux_sp']=''    
    
    
    #%%%% Individual disk-integrated transmission spectra
    #    - before/after the various spectral corrections for disk-integrated data
    plot_dic['trans_sp']=''    
    

    
    
    
    ##################################################################################################
    #%%% Module: instrumental calibration
    #    - always activated in spectral mode, to allow rescaling spectral flux profiles to blazed counts and,if requested, for weighing
    #    - disabled in CCF mode
    ##################################################################################################
    
    #%%%% Calculating/retrieving
    gen_dic['calc_gcal']=True  


    #%%%% Multi-threading
    gen_dic['gcal_nthreads'] =  int(0.8*cpu_count())         
    
    
    #%%%% Origin
    #    - set to True to measure calibration profile and detector noise from blazed data
    #      S2D_BLAZE fits file must be provided in the input data directory    
    #    - if set to False, or if blazed data are not provided for a given visit, calibration profiles are derived from input flux and error tables or, if error tables are not associated
    # with input dara, they are set to a constant value
    gen_dic['gcal_blaze']=True
    

    #%%%% Bin size
    
    #%%%%% Spectral bin size (in A)
    #    - calibration profiles are binned before being fitted with a model to extrapolate or complete them
    #      the binning is justified by the low-frequency variations of the calibration profile, and the otherwise too heavy size of the full pixel grid
    #    - applied over each order independently
    #    - if set to a larger value than an order width, calibration will not be fitted but set to the measured value over each order
    #      binw should be large enough to smoot out sharp variations in the model calibration profile
    #    - format: inst : value
    #      leave empty to use default values 
    gen_dic['gcal_binw'] = {}
    
    
    #%%%%% Temporal bin size
    #    - not relevant for blaze-derived profiles
    #    - with low-SNR data it might be necessary to group exposures to perform the calibration estimates
    #    - format: value
    gen_dic['gcal_binN'] = 1    
    
    
    #%%%% Edge polynomials
    #    - model is made of a blue, a central, and a red polynomial
    #    - set the fraction (in 0-1) of the order width that define the blue and red ranges
    #    - set the order of the polynomials (between 2 and 4)
    #    - beware that this calibration model will propagate into the weighing and the photoelectron rescaling, and thus sharp variations should be avoided 
    #    - if input data are CCFs or 'gcal_binw' is larger than the spectral order width, calibration is set to a constant value  
    #    - format: {prop : value}    
    gen_dic['gcal_edges']={'blue':0.3,'red':0.3}    
    gen_dic['gcal_deg']={'blue':4,'mid':2,'red':4}
    
        
    #%%%% Outliers    
    #    - not relevant for blaze-derived profiles 
        
    #%%%%% Threshold
    #    - calibration values above the global threshold, or outliers in the residuals from a preliminary fit, are sigma-clipped and not fitted
    #    - format: {inst : {'outliers' : val,'global' : val} }   
    gen_dic['gcal_thresh']={}
    
    
    #%%%%% Non-exclusion range
    #    - in A
    #    - outliers are automatically excluded before fitting the final model
    #      we prevent this exclusion over the edges of the orders, where sharp variations are not well captured and can be attributed to outliers
    #    - format : {inst : [x1,x2] }  
    gen_dic['gcal_nooutedge']={}
    
    
    #%%%% Plots: instrumental calibration
    #    - options:
    # + 'gcal_all': mean calibration over each order, for all orders and all exposures 
    # + 'gcal_ord': spectral calibration profile over each order, for each exposure 
    # + 'noises_ord': noise contributions (if available from blaze measurements) 
    plot_dic['gcal_all']=''
    plot_dic['gcal_ord']=''
    plot_dic['noises_ord']=''    
    
    #%%%%% Spectral bin size (in A)
    if gen_dic['star_name']=='HD189733':gen_dic['gcal_binw'] = {'ESPRESSO': 1.}

    #%%%%% Threshold
    if gen_dic['star_name']=='HD189733':gen_dic['gcal_thresh'] = {'ESPRESSO':{'outliers':5.,'global':3e6}}

    ##################################################################################################
    #%%% Module: telluric correction
    #    - use plot_dic['flux_sp'] to compare spectra before/after correction and identify orders in which tellurics are too deep and numerous to be well corrected, and that should be excluded from the entire analysis
    ##################################################################################################
    
    #%%%% Activating
    gen_dic['corr_tell']=True   
    
    
    #%%%% Calculating/retrieving
    gen_dic['calc_corr_tell']=True 
    
    
    #%%%% Multi-threading
    gen_dic['tell_nthreads'] =   int(0.8*cpu_count())          
    
    
    #%%%% Correction mode
    #    - 'input' : if telluric spectra are contained in input files
    #    - 'autom': automatic telluric correction
    gen_dic['calc_tell_mode']='autom'      
    
    
    #%%%% Telluric species
    gen_dic['tell_species']=['H2O','O2']
    
    
    #%%%% Orders to be fitted
    #    - if left empty, all orders and the full spectrum is used
    #    - format: {inst:{vis: [iord] }}   
    gen_dic['tell_ord_fit'] = {}
    
    
    #%%%% Telluric CCF
    
    #%%%%% Definition range
    #    - in Earth rest frame, common to all molecule
    #    - +-40 km/s if set to None
    gen_dic['tell_def_range']= None
    
    
    #%%%%% Continuum polynomial degree
    #    - format: {inst:{vis:{mol: ideg }}} 
    # with ideg from 1 to 4 
    #    - default: 0 for flat, constant continuum
    gen_dic['tell_CCFcont_deg'] = {}   
    
    
    #%%%%% Continuum range
    #    - in Earth rest frame, for each molecule
    #    - format: {inst:{vis:{mol: [x0,x1] }}} 
    #    - continuum range excludes +-15 km/s if undefined
    gen_dic['tell_cont_range']={}
    
    
    #%%%%% Fit range
    #    - in Earth rest frame, for each molecule
    #    - format is : {inst:{vis:{mol: [x0,x1] }}} 
    #    - adjust the fitted range to optimize the results
    #    - fit range set to the definition range if undefined
    gen_dic['tell_fit_range']={}
    
    
    #%%%% Fixed/variable properties
    #    - format is : mod_prop = { inst : { vis : { molec : { par : { 'vary' : bool , 'value': float , 'min': float, 'max': float } } } }}        
    #      leave empty the various fields to use default values
    #    - see details of fit settings in data_dic['DI']['mod_prop'] 
    #    - 'par' can be one of:
    # + 'Temperature'  (in K) : temperature of the Earth model layer
    # + 'Pressure_LOS' (in atm) : average pressure over the layers occupied by the species
    # + 'ISV_LOS' (in cm-2) : integrated species vapour along the LOS
    gen_dic['tell_mod_prop']={}
    
    
    #%%%% Correction settings
    
    #%%%%% Threshold 
    #    - flux values where telluric contrast is deeper than this threshold (between 0 for no telluric absorption and 1 for full telluric absorption) are set to nan
    gen_dic['tell_thresh_corr'] = 0.9      
    
    
    #%%%%% Exposures to be corrected
    #    - format is : {inst:{vis: [idxi] }} 
    #    - leave empty for all exposures to be corrected
    gen_dic['tell_exp_corr'] = {}
    
    
    #%%%%% Orders to be corrected
    #    - format is : {inst:{vis: [idxi] }}
    #    - if left empty, all orders and the full spectrum is used
    gen_dic['tell_ord_corr'] = {}
    
    
    #%%%%% Spectral range(s) to be corrected
    #    - format is : {inst:{vis: [[x0,x1],[x2,x3],..] }}
    #    - if left empty, applied to the the full spectrum
    gen_dic['tell_range_corr'] = {}
    
    
    #%%%% Plot settings
    
    #%%%%% Telluric CCFs (automatic correction)
    plot_dic['tell_CCF']=''      
    
    
    #%%%%% Fit results (automatic correction)
    plot_dic['tell_prop']=''      
    
 
    #%%%% Activating
    if gen_dic['star_name']=='HD189733':
        gen_dic['corr_tell']=True #& False
        gen_dic['calc_corr_tell']=True & False

    #%%%% Correction mode    
    if gen_dic['star_name']=='HD189733':gen_dic['calc_tell_mode']='autom'   
    
    #%%%% Telluric species
    if gen_dic['star_name']=='HD189733':gen_dic['tell_species']=['H2O','O2']  
 
    #%%%% Fixed/variable properties
    if gen_dic['star_name']=='HD189733':
        gen_dic['tell_mod_prop']={'ESPRESSO' : {
            '20210830' : {
#            'O2':{'Pressure_LOS':{ 'vary' : True , 'value':300. , 'min':0.39, 'max':0.45 } } } } }
            'O2':{'Pressure_LOS':{ 'vary' : True , 'value':0.407 , 'min':0.39, 'max':0.45 } } } }
}
 
    #%%%% Plot settings
    if gen_dic['star_name']=='HD189733':
        #%%%%% Telluric CCFs (automatic correction)
        plot_dic['tell_CCF']='pdf'  #      

        #%%%%% Fit results (automatic correction)
        plot_dic['tell_prop']='pdf'  # 


    ##################################################################################################
    #%%% Modules: flux balance corrections
    ##################################################################################################
    
    #%%%% Multi-threading
    gen_dic['Fbal_nthreads'] = int(0.8*cpu_count())          
    
    
    ##################################################################################################
    #%%%% Module: stellar masters
    ##################################################################################################
    
    #%%%%% Activating  
    gen_dic['glob_mast']=True & False   
    
    
    #%%%%% Calculating/retrieving
    gen_dic['calc_glob_mast']=True & False
    
    
    #%%%%% Measured masters
    #    - calculated by default for each visit, and over all visits of the same instrument if gen_dic['Fbal_vis']=='meas'  
    #    - these masters are not used for other modules since they are calculated from raw uncorrected data
    #      they are calculated automatically for flux balance corrections, or on request for preliminary visualization in plots 
    
    #%%%%%% Mean ('mean') or median ('med') 
    #    - mode of master calculation
    gen_dic['glob_mast_mode']='med'    
    
    
    #%%%%%% Exposures used in master calculations
    #    - format: {inst:{vis:[idxi]}}
    #    - set 'all' (default if left empty) or a list of exposures
    gen_dic['glob_mast_exp'] = {}       
           
    
    #%%%%% External masters
    #    - format: {inst:{vis:path}}
    #    - set path to spectrum file (two columns: wavelength in star rest frame in A, flux density in arbitrary units)
    #      spectrum must be defined over a larger range than the processed spectra
    #    - only required if gen_dic['Fbal_vis']=='ext', to reset all spectra from different instruments to a common balance, or to reset spectra from a given visit 
    # to a specific stellar balance in a given epoch
    gen_dic['Fbal_refFstar']= {}  
    
    
    #%%%%% Plots: masters
    plot_dic['glob_mast']=''     
    
    
    #%%%%%% Mean ('mean') or median ('med')  
    if gen_dic['star_name']=='HD189733':
        gen_dic['glob_mast']=True #& False   

        gen_dic['calc_glob_mast']=True & False

        gen_dic['glob_mast_exp'] = {'ESPRESSO':{'visit1':'all'}}
        gen_dic['glob_mast_mode']='med'        #ANTARESS I
   
    
    ##################################################################################################
    #%%%% Module: global flux balance
    #    - a first correction based on the ratio between each exposure and its visit master is performed, to avoid inducing local-scale variations due to changes in stellar line shape
    #      a second correction based on the low-resolution ratio between the visit master and a reference is then performed, to avoid biases when comparing or combining visits together
    #      the latter correction is performed after the intra-order one, if requested 
    #    - only the spectral balance is corrected for, not the global flux differences. This can be done via gen_dic['flux_sc']
    #    - do not disable, unless input spectra have already the right balance    
    ##################################################################################################
    
    #%%%%% Activating
    gen_dic['corr_Fbal']=True    
    
    
    #%%%%% Calculating/retrieving
    gen_dic['calc_corr_Fbal']=True  
    
    
    #%%%%% Reference master
    #    - after spectra in a given visit are scaled (globally and per order) to their measured visit master, a second global scaling:
    # + 'None': is not applied (valid only if a single instrument and visit are processed)
    # + 'meas': is applied toward the mean of the measured visit masters (valid only if a single instrument with multiple visits is processed)
    # + 'ext': is applied toward the external input spectrum provided via gen_dic['Fbal_refFstar'] 
    #    - the latter option allows accounting for variations on the global stellar balance between visits, and is otherwise necessary to set spectra from different instruments (ie, with different coverages) to the same balance 
    gen_dic['Fbal_vis']='meas'  
    
    
    #%%%%% Fit settings 
    
            
    #%%%%%% Spectral bin size
    #    - format: {inst:val}
    #      bin size of the fitted data (in 1e13 s-1)
    # dnu[1e13 s1] = c[km s-1]*dw[A]/w[A]^2
    #      for ESPRESSO dnu < 0.9 (0.5) yields more than 1 (2) bins in most orders
    #    - for the correction relative to measured visit masters: binning is applied over each order (set a value larger than an order width to bin over the entire order)
    #      for the correction relative to reference masters, binning is applied over full orders by default
    #    - bin size should be small enough to capture low-frequency flux balance variations but large enough to smooth high-frequency variations and reduce computing time.
    gen_dic['Fbal_bin_nu'] = {} 
    
    
    #%%%%%% Spectral range(s) to be fitted
    #    - even if a localized region is studied afterward, the flux balance should be corrected over as much as possible of the spectrum
    #      however the module can also be used to correct locally (ie in the region of a single absorption line) for the spectral flux balance
    #    - format: {inst:{vis:[[x0,x1],[x2,x3],...]}} with x in A
    #    - if left empty, all orders and the full spectrum is used
    gen_dic['Fbal_range_fit'] = {}
    
    
    #%%%%%% Orders to be fitted
    #    - format: {inst:{vis:[idx]}}
    gen_dic['Fbal_ord_fit'] = {}
               
    
    #%%%%%% Phantom bins
    #    - format: float (in 1e13 s-1)
    #    - range in 'nu' on the blue side of the fitted spectrum that is fitted with a linear model and mirrored in the fitted spectrum
    #      this limits the divergence of the model on the blue side
    #    - set to None for automatic determination and to 0 to prevent
    gen_dic['Fbal_phantom_range'] = None  
      
    
    #%%%%%% Uncertainty scaling
    #    - variance of fitted bins is put to the chosen power (0 = equal weights; 1 = original errors with no scaling; increase to give more weight to data with low errors)
    #    - applied to the scaling with the measured visit master only
    gen_dic['Fbal_expvar'] = 1./4.        
    
    
    #%%%%%% Automatic sigma-clipping
    #    - applied to the scaling with the measured visit master only
    gen_dic['Fbal_clip'] = True  
        
    
    #%%%%% Flux balance model
        
    #%%%%%% Model
    #    - types:
    # + 'pol' : polynomial function 
    # + 'spline' : 1-D smoothing spline
    #    - degree and smoothing factor must be defined for the scaling to the visit-specific master, and if relevant for the scaling to the reference masters
    gen_dic['Fbal_mod']='pol'
    
    
    #%%%%%% Polynomial degree 
    #    - 'Fbal_deg' applies to the ratio between indivisual exposure and the visit-specific references
    #      'Fbal_deg_vis' applies to the ratio between visit-specific references and the global reference
    #    - default = 4 (decrease to smooth)
    gen_dic['Fbal_deg'] ={}
    gen_dic['Fbal_deg_vis'] ={}
    
    
    #%%%%%% Spline smoothing factor
    #    - 'Fbal_smooth' applies to the ratio between indivisual exposure and the visit-specific references
    #      'Fbal_smooth_vis' applies to the ratio between visit-specific references and the global reference
    #    - default = 1e-4 (increase to smooth)
    gen_dic['Fbal_smooth']={}
    gen_dic['Fbal_smooth_vis']={}   
          
    
    #%%%%% Spectral range(s) to be corrected
    #    - set to [] to apply to the the full spectrum
    gen_dic['Fbal_range_corr'] = [ ]           
        
    
    #%%%%% Plot settings
    
    #%%%%%% Exposures/visit balance 
    #    - between exposure and their visit master
    plot_dic['Fbal_corr']=''  
    
    #%%%%%% Exposures/visit balance (DRS)
    #    - if available
    plot_dic['Fbal_corr_DRS']=''  
    
    #%%%%%% Measured/reference visit balance
    plot_dic['Fbal_corr_vis']=''  
    


    #%%%%% Activating
    gen_dic['corr_Fbal']=True     &  False
    if gen_dic['star_name']=='HD189733':gen_dic['corr_Fbal']=True  #&  False
    
    #%%%%% Calculating/retrieving
    gen_dic['calc_corr_Fbal']=True   & False


    #%%%%% Reference master 
    if gen_dic['star_name']=='HD189733':
        gen_dic['Fbal_vis'] = 'meas'    
    
    #%%%%%% Spectral bin size
    if gen_dic['star_name']=='HD189733':
        # gen_dic['Fbal_bin_nu'] = {'ESPRESSO':1}  #smooth
        gen_dic['Fbal_bin_nu'] = {'ESPRESSO':0.7}  #final        

    #%%%%% Flux balance model  
    #%%%%%% Model
    if gen_dic['star_name']=='HD189733':
        gen_dic['Fbal_mod']='spline'        #ANTARESS I   

    #%%%%%% Spline smoothing factor
    if gen_dic['star_name']=='HD189733':
        gen_dic['Fbal_smooth_vis'] = {'ESPRESSO':{'visit1':2e-5  }} 
        #ANTARESS II, low-freq correction (dnu1) to analyze spurious features and test filter wiggle correction 
        # gen_dic['Fbal_smooth'] = {'ESPRESSO':{'visit1':3e-4  }}        
        gen_dic['Fbal_smooth'] = {'ESPRESSO':{'visit1':1.5e-4  }}     #ANTARESS II, fine correction (dnu0.7, 1.5e-4)

    #%%%%% Plot settings
    if gen_dic['star_name']=='HD189733':
        #%%%%%% Exposures/visit balance 
        #    - between exposure and their visit master
        plot_dic['Fbal_corr']='pdf'  
        
        #%%%%%% Exposures/visit balance (DRS)
        #    - if available
        plot_dic['Fbal_corr_DRS']='pdf'  

    ##################################################################################################
    #%%%% Module: order flux balance
    #    - same as the global correction, over each independent order
    ##################################################################################################
    
    #%%%%% Activating
    #    - for 2D spectra only
    gen_dic['corr_FbalOrd']=True   
    
    
    #%%%%% Calculating/retrieving
    gen_dic['calc_corr_FbalOrd']=True   
    
    
    #%%%%% Model polynomial degree 
    gen_dic['Fbal_deg_ord'] = {}
    
    
    #%%%%% Spectral range(s) to be fitted
    #    - set to [] to use the full spectrum
    gen_dic['FbalOrd_range_fit'] = []
    
    
    #%%%%% Orders to be fitted
    gen_dic['FbalOrd_ord_fit'] = {}
    
    
    #%%%%% Spectral bin size
    #    - in A
    gen_dic['Fbal_binw_ord'] = 2.
    
    
    #%%%%% Automatic sigma-clipping
    gen_dic['Fbal_ord_clip'] = True
    
    
    #%%%%% Plots: flux balance correction 
    #    - can be heavy to plot, use png
    plot_dic['Fbal_corr_ord']='' 
    
    
    if gen_dic['star_name']=='HD189733':
        #%%%%% Activating
        #    - for 2D spectra only
        gen_dic['corr_FbalOrd']=True 
        
        
        #%%%%% Calculating/retrieving
        gen_dic['calc_corr_FbalOrd']=True  &   False
        
    
    ##################################################################################################
    #%%%% Module: temporal flux correction
    #    - if applied, it implies that relative flux variations (in particular in-transit) can be retrieved by extrapolating the corrections fitted on other exposures
    #      thus the transit scaling module should not be applied and is automatically deactivated
    ##################################################################################################
    
    #%%%%% Activating
    gen_dic['corr_Ftemp']= False
    
    
    #%%%%% Calculating/retrieving
    gen_dic['calc_corr_Ftemp']=True   
    
    
    #%%%%% Spectral range(s) to be fitted
    #    - if left empty, all orders and the full spectrum is used 
    gen_dic['Ftemp_range_fit'] = {}
    
    
    #%%%%% Exposures excluded from the fit
    gen_dic['idx_nin_Ftemp_fit']={}
    
    
    #%%%%% Model polynomial degree 
    gen_dic['Ftemp_deg'] = {}
    
    
    #%%%%% Plots: temporal flux correction
    #    - can be heavy to plot, use png
    plot_dic['Ftemp_corr']=''  
    
    
    
    ##################################################################################################
    #%%% Module: cosmics correction
    ##################################################################################################
    
    #%%%% Activating
    gen_dic['corr_cosm']=True   
    
    
    #%%%% Calculating/retrieving
    gen_dic['calc_cosm']=True    
        
    
    #%%%% Multi-threading
    gen_dic['cosm_nthreads'] = int(0.8*cpu_count())           

    #%%%% Comparison settings    
    
    #%%%%% Alignment mode
    #    - choose option to align spectra prior to cosmic identification and correction
    # + 'kep': Keplerian curve 
    # + 'pip': pipeline RVs (if available)
    # + 'autom': for automatic alignment using the specified options 'range' and 'RVrange_cc'
    #            'range' : define the spectral range(s) used to cross-correlate spectra
    #                      use a large range for increased precision, at the cost of computing time
    #                      set to [] to use the full spectrum
    #            'RVrange_cc' : define the RV range and step used to cross-correlate spectra
    #                           should cover the maximum velocity shift between two exposures in the visit
    #    - the Keplerian option should be preferred, as the others will be biased by the RM effect 
    gen_dic['al_cosm']={'mode':'kep'}
    
    
    #%%%%% Adjacent spectra
    #    - define the total number of spectra around each exposure used to identify and replace cosmics
    gen_dic['cosm_ncomp'] = 6    
    
    
    #%%%%% Outlier threshold 
    #    - format is {instrument : {visit: value }}
    #    - a pixel is flagged as cosmic hit if its flux deviates from the mean over adjacent exposures by more than a 'cosm_thresh' times
    # the standard-deviation over adjacent exposures and the error on the pixel flux
    gen_dic['cosm_thresh'] = {} 

    #%%%% Correction settings     
        
    #%%%%% Exposures
    #    - leave empty for all exposures to be corrected
    gen_dic['cosm_exp_corr']={}
            
            
    #%%%%% Orders
    #    - leave empty for all orders to be corrected
    gen_dic['cosm_ord_corr']={}


    #%%%%% Pixels
    #    - format is {inst : { vis : n }}
    #      where n is the number of pixels on each side of a cosmic-flagged pixel that will be corrected, to account for local charge bleeding
    #    - leave empty to correct cosmic-flagged pixels only (default, n=0)
    gen_dic['cosm_n_wings']={}
        
    
    #%%%% Plots:cosmics
    plot_dic['cosm_corr']=''    
    
    
    #%%%% Activating
    gen_dic['corr_cosm']=True     &  False
    if gen_dic['star_name']=='HD189733':gen_dic['corr_cosm']=True  #& False

    #%%%% Calculating/retrieving
    gen_dic['calc_cosm']=True   &  False  

    #%%%% Comparison spectra
    if gen_dic['star_name']=='HD189733':gen_dic['cosm_ncomp'] = 10 

    #%%%% Outlier threshold  
    if gen_dic['star_name']=='HD189733':    
        gen_dic['cosm_thresh'] = {'ESPRESSO':{'visit1':15}}    #ANTARESS II

    #%%%% Plots:cosmics
    if gen_dic['star_name']=='HD189733':
        plot_dic['cosm_corr']='pdf'        

    ##################################################################################################
    #%%% Module: persistent peak masking
    #    - a stellar continuum is estimated internally to this module, using the settings from the continuum module (gen_dic['DI_stcont'])
    ##################################################################################################
    
    #%%%% Activating
    gen_dic['mask_permpeak']= False
    
    
    #%%%% Calculating/retrieving 
    gen_dic['calc_permpeak']=True  
    
    
    #%%%% Multi-threading
    gen_dic['permpeak_nthreads'] = int(0.8*cpu_count())      


    #%%%% Peaks exclusion settings
    
    #%%%%% Spurious peaks threshold 
    #    - set on the flux difference from the stellar continuum, compared to their error
    gen_dic['permpeak_outthresh']=4
    
    
    #%%%%% Spurious peaks window
    #    - range around flagged pixels marked for exclusion
    #    - format: { inst : val }}
    #      in A
    gen_dic['permpeak_peakwin']={}  
    
    
    #%%%%% Bad consecutive exposures 
    #    - a peak is masked if it is flagged in at least max(permpeak_nbad,3) consecutive exposures
    gen_dic['permpeak_nbad']=3 

    
    #%%%% Correction settings
    
    #%%%%% Exposures to be corrected
    #    - format: { inst : { vis : [ idx_exp ] }}
    #    - leave empty for all exposures to be corrected
    gen_dic['permpeak_exp_corr']={}
    
    
    #%%%%% Orders to be corrected
    #    - format: { inst : { vis : [ idx_ord ] }}
    #    - leave empty for all orders to be corrected
    gen_dic['permpeak_ord_corr']={}
    
    
    #%%%%% Spectral range(s) to be corrected
    #    - format: { inst : { ord : [[l1,l2],[l3,l4], ..] }}
    #      with l in A
    #    - leave undefined or empty to take the full range 
    gen_dic['permpeak_range_corr'] = {}
    
        
    #%%%%% Non-masking of edges
    #    - format: { inst : [[dl_blue,dl_red] }
    #      with dl in A
    #    - we prevent masking over the edges of the orders, where the continuum is often not well-defined
    gen_dic['permpeak_edges']={}
    
    
    #%%%% Plots: master and continuum
    #    - to check the flagged pixels use plot_dic['flux_sp'] before/after correction
    plot_dic['permpeak_corr']=''  
    
    
    
    
    ##################################################################################################
    #%%% Module: ESPRESSO "wiggles"
    #    - this module is used to characterize and correct wiggles, using either an analytical model over the full visit, or a filter
    #      the analytical model should be preferred whenever possible to keep as much as possible of the planetary and stellar feature at medium-resolution
    #      if the wiggle pattern is however too complex to be captured by the model, apply the filter (low-resolution variations should have been previously corrected with the flux balance module)
    #    - wiggles are processed in wave_number space nu[1e13 s-1] = c[km s-1]/w[A]
    #      for example, w = 4000 [A] correspond to nu = 75 [1e13 s-1]
    #      wiggle frequencies Fnu corresponds to wiggle periods Pw[A] = w[A]^2/(Fnu[1e-13 s]*c[km s-1]) = w[A]^2*Pnu[1e13 s-1]/c[km s-1]
    ##################################################################################################
    
    
    #%%%% Activating 
    gen_dic['corr_wig']= False
    
    
    #%%%% Calculating/retrieving
    gen_dic['calc_wig']=True   
    
    
    #%%%% Guide shift reset
    #    - disable automatic reset of wiggle properties following guide star shift, for the chosen list of visits
    #    - relevant with analytical model only
    gen_dic['wig_no_guidchange'] = []   
    
    
    #%%%% Forced order normalization
    #    - transmission spectrum is normalized to unity over each order 
    #    - the analytical wiggle model can only capture medium-frequency modulations around unity
    #      low-frequency variations must be captured with the flux balance correction or with the wiggle filter, or later on in planetary spectra (it is in this later case 
    # that normalization can be forced here temporarily)
    #    - it is advised to disable this field with filter mode
    gen_dic['wig_norm_ord'] = True  
    
    
    #%%%% Visits to be processed 
    #    - leave empty to process all visits
    gen_dic['wig_vis'] = []
    
    
    #%%%% Stellar master
    
    #%%%%% Resampling 
    #    - calculate master on the table of each exposure or resample a common master
    #    - using individual masters has a negligible impact 
    gen_dic['wig_indiv_mast'] = False
    
    
    #%%%%% Meridian split
    #    - use different masters before/after meridian 
    #    - a priori using a global master over the night is better to smooth out the wiggles
    gen_dic['wig_merid_diff'] = False
    
    
    #%%%%% Exposure selection
    #    - set to 'all' to use all exposures
    gen_dic['wig_exp_mast'] = {}
        
        
    #%%%% Analysis settings        
        
    #%%%%% Exposures to be characterized
    #    - instrument > visit
    #    - set to 'all' to use all exposures
    gen_dic['wig_exp_in_fit'] = {}
    
    
    #%%%%% Groups of exposures to be characterized together
    #    - leave empty to perform the fit on individual exposures
    #    - this is useful to boost SNR, especially in the bluest orders, without losing the spectral resolution over orders
    #      beware however that wiggles amplitude and offsets change over time, and will thus be blurred by this average
    gen_dic['wig_exp_groups']={}
        
    
    #%%%%% Spectral range(s) to be characterized
    #    - format is {vis:[[nu1,nu2],[nu3,nu4],..]}
    #    - if left empty, the full spectrum is used
    #    - units are c/w (1e13 s-1)
    gen_dic['wig_range_fit'] = {}
        
        
    #%%%%% Spectral bin size
    #    - all spectra are binned prior to the analysis
    #    - in nu space (1e13 s-1), with dnu[1e13 s-1] = c[km s-1]*dw[A]/w[A]^2   
    #    - bin size should be small enough to sample the period of the wiggles, but large enough to limit computing time and remove possible correlations between bins 
    # + for the two dominant wiggle component set dw = 2 A, ie dnu = 0.0166 (mind that it creates strong peaks in the periodograms at F = 60, 120, 180)
    # + for the mini-wiggle component set dw = 0.05 A, ie dnu = 0.0004 (mind that it creates a signal in the periodograms at F = 2400)
    #    - even if the spectra are analyzed at their native resolution it is necessary to resample them to merge overlapping bands at order edges
    gen_dic['wig_bin'] = 0.0166   
        
    
    #%%%%% Orders to be fitted
    #    - if left empty, all orders are used 
    gen_dic['wig_ord_fit'] = {}
    
    
    #%%%% Analysis
    
    #%%%%% Screening 
    #    - use to identify which ranges to include in the analysis, and which wiggle components are present
    #    - use 'plot_spec' to identify which ranges / orders are of poor quality and need to be excluded from the fit and/or the correction
    #      use as much of the ESPRESSO range as possible, but exclude the bluest orders where the noise is much larger than the wiggles
    #    - use 'plot_hist' to plot the periodogram from all exposures together, to identify the number and approximate frequency of wiggle components 
    gen_dic['wig_exp_init']={
        'mode':True  ,
        'plot_spec':True,
        'plot_hist':True,
        'y_range':None
        }
    
    #%%%%% Filter
    #    - characterize wiggles using a Savitzky–Golay filter of the binned transmission spectrum in each exposure 
    #    - 'win': size of the smoothing window, in nu 
    #    - 'deg': order of the polynomial used to fit the smoothed spectrum
    gen_dic['wig_exp_filt']={
        'mode':False,
        'win':0.2,
        'deg':3, 
        'plot':True        
        }

    #%%%%% Analytical model   
    #    - run the following steps sequentially to determine the model
    
    #%%%%%% Step 1: Chromatic sampling
    #    - to sample the frequency and amplitude of each wiggle component with nu, in a representative selection of exposure
    #    - sampling the frequency and amplitude of each wiggle component with nu
    #    - wiggle properties are sampled using a sliding periodogram
    #    - only a representative subset of exposures needs to be sampled, using 'wig_exp_in_fit'
    #    - set 'comp_ids' between 1 and 5
    #      only the component with highest 'comp_ids' is sampled using all shifts in 'sampbands_shifts'
    #      lower components are fitted with a single shift from 'sampbands_shifts', chosen through 'direct_samp'
    #      thus, start by smapling the highest component, and proceed by including lower ones iteratively
    #    - 'freq_guess': define the polynomial coefficients describing the model frequency for each component 
    #                    these models control the definition of the sampling bands 
    #    - 'nsamp' : number of cycles to sample for each component, in a given band (defines the size of the sampling band, based on the guess frequency)
    #                must not be too high to ensure that the component frequency remains constant within the sampled bands 
    #    - 'sampbands_shifts': oversampling of sampling bands (nu in 1e13 s-1)
    #                          adjust to the scale of the frequency or amplitude variations of each component
    #                          set to [None] to prevent sampling (the full spectrum is fitted with one model)
    #                          to estimate size of shifts: nu = c/w -> dnu = cdw/w^2 
    #    - 'direct_samp': set direct_samp[comp_id] to the index of sampbands_shifts[comp_id-1] for which the sampling of comp_id should be applied
    #    - 'src_perio': frequency ranges within which periodograms are searched for each component (in 1e13 s-1). 
    #                       + {'mod':None} : default search range  
    #                       + {'mod':'slide' , 'range':[y,z] } : the range is centered on the frequency calculated with 'freq_guess'        
    #                   the field 'up_bd':bool can be added to further use the frequency of the higher component as upper bound
    #                   if the frequency is fitted rather than fixed, the search range is used as prior
    #    - 'nit': number of fit iterations in each band 
    #    - 'fap_thresh': wiggle in a band is fitted only if its FAP is below this threshold (in %). 
    #                    set to >100 to always fit.
    #    - 'fix_freq2expmod' = [comp_id] fixes the frequency of 'comp_id' using the fit results from 'wig_exp_point_ana'
    #      'fix_freq2vismod' = {comps:[x,y] , vis1:path1, vis2:path2 } fixes the frequency of 'comps' using the fit results from 'wig_vis_fit' at the given path for each visit 
    #    - 'plot': plot sampled transmission spectra and band sampling analyses
    gen_dic['wig_exp_samp']={
        'mode':False,   
        'comp_ids':[1,2],          
        'freq_guess':{1:{ 'c0':3.72, 'c1':0., 'c2':0.},
                      2:{ 'c0':2.05, 'c1':0., 'c2':0.}},
        'nsamp':{1:8,2:8}, 
        'sampbands_shifts':{1:np.arange(16)*0.15,2:np.arange(16)*0.3},
        'direct_samp' : {2:0},         
        'nit':40, 
        'src_perio':{1:{'mod':None}, 2:{'mod':None,'up_bd':True}},  
        'fap_thresh':5,
        'fix_freq2expmod':[],
        'fix_freq2vismod':{},
        'plot':True
        }      
    
    
    #%%%%%% Step 2: Chromatic analysis
    #    - to fit the sampled frequency and amplitude with polynomials of nu, to evaluate their degree and chromatic coefficients
    #    - fitting the sampled frequency and amplitude with polynomials of nu
    #    - use this step to determine 'wig_deg_Freq' and 'wig_deg_Amp' for each component
    #    - 'comp_ids': component properties to analyze, among those sampled in 'wig_exp_samp'
    #    - 'thresh': threshold for automatic exclusion of outliers
    #    - 'plot': plot properties and their fit in each exposure
    gen_dic['wig_exp_nu_ana']={
        'mode':False  ,     
        'comp_ids':[1,2], 
        'thresh':3.,  
        'plot':True
        } 
    
    
    #%%%%%%% Frequency degree
    #    - maximum degree of polynomial frequency variations with nu
    #    - for some datasets the second order component may not be constrained by the blue bands and remain consistent with 0
    gen_dic['wig_deg_Freq'] = {comp_id:1 for comp_id in range(1,6)}
    
    
    #%%%%%%% Amplitude degree
    #    - maximum degree of polynomial amplitude variations with nu
    #    - defined for each component
    gen_dic['wig_deg_Amp'] = {comp_id:2 for comp_id in range(1,6)}
    
    
    #%%%%%% Step 3: Exposure fit 
    #    - to fit the spectral wiggle model to each exposure individually, initialized by the results of 'wig_exp_nu_ana'   
    #    - fitting the spectral wiggle model to each exposure individually
    #    - 'comp_ids': components to include in the model
    #    - 'init_chrom': initialize the fit guess values using the results of 'wig_exp_nu_ana' on the closest exposure sampled in 'wig_exp_samp'
    #                    running 'wig_exp_samp' on a selection of representative exposures sampling the wiggle variations is thus sufficient
    #                    beware to run 'wig_exp_nu_ana' with the same components used in 'wig_exp_fit'
    #    - 'freq_guess': define for each component the polynomial coefficients describing the model frequency 
    #                    used to initialize frequency values if 'init_chrom' is False
    #    - 'nit': number of fit iterations 
    #    - 'fit_method': optimization method. 
    #                    if the initialization is correct, setting to 'leastsq' is sufficient and fast
    #                    if convergence is more difficult to reach, set to 'nelder'
    #    - 'use': set to False to retrieve fits 
    #             useful to analyze their results using 'wig_exp_point_ana', and periodograms automatically produced for each exposure
    #    - 'fixed_pointpar': fix values of chosen properties ( > vis > [prop1,prop2,..]) to their model from 'wig_exp_point_ana'
    #    - 'prior_par: bound properties with a uniform prior on the chosen range (common to all exposures, defined as par > {'low' : val, 'high' : val}). Use results from 'wig_exp_point_ana' to decide on the prior range.
    #                  if par > {'guess' : val} is defined it will overwrite the default or chromatic initialization 
    #    - 'model_par': initialize property to its exposure value v(t) from the 'wig_exp_point_ana' model, and set a uniform prior in [ v(t)-model_par[par][0] ; v(t)+model_par[par][1] ]
    #    - 'plot': plot transmission spectra with their models, residuals, associated periodograms, and overall rms
    gen_dic['wig_exp_fit']={
        'mode':False, 
        'comp_ids':[1,2],  
        'init_chrom':True,
        'freq_guess':{
            1:{ 'c0':3.72077571, 'c1':0., 'c2':0.},
            2:{ 'c0':2.0, 'c1':0., 'c2':0.}},
        'nit':20, 
        'fit_method':'leastsq', 
        'use':True,
        'fixed_pointpar':{},
        'prior_par':{},
        'model_par':{},
        'plot':True,
        }     
    
    
    #%%%%%% Step 4 Pointing analysis
    #    - to fit the phase, and the chromatic coefficients of frequency and amplitude derived from 'wig_exp_fit', as a function of the telescope pointing coordinates 
    #    - fitting the phase, and the chromatic coefficients of frequency and amplitude, as a function of the telescope pointing coordinates  
    #    - 'source': fitting coefficients derived from the sampling ('samp') or spectral ('glob') fits 
    #    - 'thresh': threshold for automatic outlier exclusion (set to None to prevent automatic exclusion)
    #    - 'fit_range': custom fit ranges for each vis > parameter
    #    - 'stable_pointpar': parameters fitted with a constant value
    #    - 'conv_amp_phase': automatically adjust amplitude sign and phase value, which can be degenerate 
    #    - 'plot': plot properties and their fit 
    gen_dic['wig_exp_point_ana']={
        'mode':False ,    
        'source':'glob',
        'thresh':3.,   
        'fit_range':{},
        'fit_undef':False,
        'stable_pointpar':[],
        'conv_amp_phase':False ,
        'plot':True
        } 
    
    #%%%%%% Step 5: Global fit 
    #    - to fit the spectro-temporal wiggle model to all exposures together, initialized by the results of 'wig_exp_point_ana'
    #    - fitting spectro-temporal model to all exposures together
    #    - by default the model is initialized using the results from 'wig_exp_point_ana'
    #    - options:
    #    - 'fit_method': optimization method. 
    #                    if the initialization is correct, setting to 'leastsq' is sufficient and fast
    #                    if convergence is more difficult to reach, set to 'nelder'
    # + 'nit': number of fit iterations 
    # + 'comp_ids': components to include in the model
    # + 'fixed': model is fixed to the initialization or previous fit results
    # + 'reuse': set to {}, or set to given path to retrieve fit file and post-process it (fixed=True) or use it as guess (fixed=False)  
    # + 'fixed_pointpar': list of pointing parameters to be kept fixed in the fit
    # + 'fixed_par': list of parameters to be kept fixed in the fit    
    # + 'fixed_amp' and 'fixed_freq': keep amplitude or frequency models fixed for the chosen list of components
    # + 'stable_pointpar': pointing parameters fitted with a constant value
    # + 'n_save_it': save fit results every 'n_save_it' iterations
    # + 'plot_hist': cumulated periodogram over all exposures
    # + 'plot_rms': rms of pre/post-corrected data over the visit
    gen_dic['wig_vis_fit']={
        'mode':False ,
        'fit_method':'leastsq',   
        'wig_fit_ratio': False,
        'wig_conv_rel_thresh':1e-5,
        'nit':15,
        'comp_ids':[1,2],
        'fixed':False, 
        'reuse':{},
        'fixed_pointpar':[],      
        'fixed_par':[],
        'fixed_amp' : [] ,
        'fixed_freq' : [] ,
        'stable_pointpar':[],
        'n_save_it':1,
        'plot_mod':True    ,
        'plot_par_chrom':True  ,
        'plot_chrompar_point':True  ,
        'plot_pointpar_conv':True    ,
        'plot_hist':True,
        'plot_rms':True ,
        } 
    
    #%%%% Correction
    #    - 'mode': apply correction
    #    - 'path': path to correction for each visit; leave empty to use last result from 'wig_vis_fit'
    #    - 'exp_list: exposures to be corrected for each visit; leave empty to correct all exposures 
    #    - 'comp_ids': components to include in the model; must be present in the 'wig_vis_fit' model
    #    - 'range': define the spectral range(s) over which correction should be applied (in A); leave empty to apply to the full spectrum
    #    - use plot_dic['trans_sp'] to assess the correction
    gen_dic['wig_corr'] = {
        'mode':False   ,
        'path':{},
        'exp_list':{},
        'comp_ids':[1,2],
        'range':{},
    }
    
    
    #%%%% Activating 
    gen_dic['corr_wig']=True    &  False    
    if gen_dic['star_name']=='HD189733':gen_dic['corr_wig']=True   #& False


    #%%%% Calculating/retrieving
    gen_dic['calc_wig']=True    &  False  
    
    #%%%% Guide shift reset
    gen_dic['wig_no_guidchange'] = []   
    
    #%%%% Forced order normalization
    gen_dic['wig_norm_ord'] = True 
    if gen_dic['star_name']=='HD189733':gen_dic['wig_norm_ord'] = False 
   
    
    #%%%%% Exposure selection
    if gen_dic['star_name']=='HD189733':
       gen_dic['wig_exp_mast']={'visit1':np.arange(16,24)}    

    #%%%%% Exposures to be fitted
    if gen_dic['star_name']=='HD189733':
       gen_dic['wig_exp_in_fit'] =  {'ESPRESSO':{'visit1':np.arange(0,43,5)}}
    
    #%%%%% Spectral range(s) to be fitted
    if gen_dic['star_name']=='HD189733':
        # gen_dic['wig_range_fit'] = {
        #     'visit1': [[20.,57.2],[57.8,74.2] ]   }

        # gen_dic['wig_range_fit'] = {    #isolation des nlles features
        #     'visit1': [[46.3,47.3],[50.7,51.3],[54.8,55.4],[64.5,65.4] ]   }
            
        # gen_dic['wig_range_fit'] = {    #TESTS
        #     'visit1': [[20.,40.] ]   }

        gen_dic['wig_range_fit'] = {    #Final correction filter
            'visit1': [[20.,57.2],[57.8,70.] ]   }

    #%%%%% Orders to be fitted
    if gen_dic['star_name']=='HD189733':
        
        gen_dic['wig_ord_fit'] = {
            'visit1':list(np.concatenate((  range(21,87),range(91,170)    ))),
        }

    #%%%% Fitting steps
    #%%%%% Step 1: Screening 
    gen_dic['wig_exp_init']={
        'mode':False  ,
        'plot_spec':True,
        'plot_hist':True,
        'y_range':[0.993,1.007],   #None
        }

    
    #%%%%% Filter
    if gen_dic['star_name']=='HD189733':
        #Les raies planetaires de Na font environ 2A de large, soit 0.016 en nu. Il faut faire attention de ne pas filtrer a si petite echelle.
        gen_dic['wig_exp_filt']={
            'mode':True,
            'win':0.2,    #0.1 ne fait pas de difference, je reste conservatif avec 0.2
            'deg':4,      #deg4 plutot que 3 diminue RMS des residus des wiggles et ne semble pas trop changer la zone du sodium planetaire
            'plot':True        
            }

    #%%%%% Step 2: Chromatic sampling
    gen_dic['wig_exp_samp']={
        'mode':False,   
        'comp_ids':[1],
        # 'comp_ids':[1,2],        
        # 'comp_ids':[1,2,3],
        # 'comp_ids':[1,2,4],         
        'freq_guess':{
            1:{ 'c0':3.72, 'c1':0., 'c2':0.},
            2:{ 'c0':2.05, 'c1':0., 'c2':0.},
            # 2:{ 'c0':1.96, 'c1':0., 'c2':0.},
            3:{ 'c0':3.55, 'c1':0., 'c2':0.},
            4:{ 'c0':156., 'c1':0., 'c2':0.},    #mini-wiggles
            },
        'nsamp':{1:8,2:8,3:5,4:200}, 
        # 'sampbands_shifts':{1:np.arange(31)*0.075,2:np.arange(31)*0.15,3:np.arange(31)*0.2},
        'sampbands_shifts':{1:np.arange(16)*0.15,2:np.arange(16)*0.3,3:np.arange(16)*0.15,4:np.arange(16)*10.},
        'direct_samp' : {1:0,2:0,3:0,4:0},         
        'nit':40, 
        'src_perio' : {1:{'mod':None}, 2:{'mod':None,'up_bd':True},3:{'mod':None,'up_bd':True},4:{'mod':None,'up_bd':True}},  
        'fap_thresh':5,
        # 'fix_freq2expmod':[1]
        'fix_freq2expmod':[],
        'fix_freq2vismod':{},
        # 'fix_freq2vismod':{'comps':[1,2],'20190720':'/Users/bourrier/Travaux/ANTARESS/Ongoing/HD209458b_Saved_data/Corr_data/Wiggles/Vis_fit/Indep_contin_valdval/ESPRESSO_20190720/V1/Outputs_final.npz',
        #                                  '20190911':'/Users/bourrier/Travaux/ANTARESS/Ongoing/HD209458b_Saved_data/Corr_data/Wiggles/Vis_fit/Indep_contin_valdval/ESPRESSO_20190911/V1/Outputs_loop0_it3.npz'},
        'plot':True
        } 

    if gen_dic['star_name']=='HD189733':
        gen_dic['wig_exp_samp']['src_perio'] = {
                1:{'mod':'slide','range':[0.02,0.02] ,'up_bd':False  },
                2:{'mod':'slide','range':[0.02,0.02] ,'up_bd':True  },
                }

    #%%%%% Step 3: Chromatic analysis
    gen_dic['wig_exp_nu_ana']={
        'mode':False  ,     
        'comp_ids':[1,2], 
        # 'comp_ids':[1,2,3], 
        # 'comp_ids':[1,2,4], 
        'thresh':3.,   #None 
        'plot':True
        }
    
    # gen_dic['wig_deg_Amp'][3]=2
    if gen_dic['star_name']=='HD189733':
        gen_dic['wig_deg_Amp'][1]=0
        gen_dic['wig_deg_Amp'][2]=0


    #%%%%% Step 4: Exposure fit 
    gen_dic['wig_exp_fit']={
        'mode':False, 
        'comp_ids':[1,2],  
        # 'comp_ids':[1,2,3], 
        # 'comp_ids':[1,2,4], 
        'init_chrom':True,# & False,
        'freq_guess':{
            1:{ 'c0':3.72077571, 'c1':0., 'c2':0.},
            2:{ 'c0':2.0, 'c1':0., 'c2':0.},
            3:{ 'c0':3.55, 'c1':0., 'c2':0.},
            4:{ 'c0':156., 'c1':0., 'c2':0.},
            },
        'nit':20, 
        'fit_method':'leastsq', 
        'use':True,
        'fixed_pointpar':{},
        'prior_par':{},
        'model_par':{},
        # 'model_par':{'AmpGlob1_c0':[0.001,0.001],'Phi1':[1.5,1.5]},
        'plot':True,
        } 

    #%%%%% Step 5: Pointing analysis
    gen_dic['wig_exp_point_ana']={
        'mode':False ,    
        # 'source':'samp',
        'source':'glob',
        'thresh':3.,   #None
        # 'thresh':None,
        'fit_range':{},
        'fit_undef':False,
        # 'fit_undef':True,
        'stable_pointpar':[],
        'conv_amp_phase':True & False ,
        'plot':True
        } 


    #%%%%% Step 6: Global fit 
    gen_dic['wig_vis_fit']={
        'mode':False ,
        'fit_method':'leastsq',  
        # 'fit_method':'nelder',  
        'wig_fit_ratio': False,
        'wig_conv_rel_thresh':1e-5,
        'nit':25,
        # 'nit':15,
        'comp_ids':[1,2],
        'fixed':False, 
        'reuse':{},
        'fixed_pointpar':[],
        # 'fixed_pointpar':['Phi1_apre','Phi1_ashift','Phi1_bpre','Phi1_bshift','Phi1_cpre','Phi1_cshift'],     #,'Phi1_off','Phi1_doff'
        # 'fixed_par':['Phi1'],        
        'fixed_par':[],
        'fixed_amp' : [] ,
        'fixed_freq' : [] ,
        'stable_pointpar':[],
        'n_save_it':1,
        'plot_mod':True    ,
        'plot_par_chrom':True  ,
        'plot_chrompar_point':True  ,
        'plot_pointpar_conv':True    ,
        'plot_hist':True,
        'plot_rms':True ,
        }

    #%%%% Correction
    gen_dic['wig_corr'] = {
        'mode':True, #& False,
        'path':{},
        'exp_list':{},
        'comp_ids':[1,2],
        'range':{},
    }
    if gen_dic['star_name']=='HD189733':
        gen_dic['wig_corr']['path'] = {'visit1':'/Users/samsonmercier/Desktop/Work/Master/2023-2024/antaress/Ongoing/HD189733/HD189733b_Saved_data/Corr_data/Wiggles/Vis_fit/ESPRESSO_visit1/Outputs_final'}

    ##################################################################################################
    #%%% Module: fringing
    ##################################################################################################
    
    #%%%% Activating
    gen_dic['corr_fring']=False
    
    
    #%%%% Calculating/retrieving
    gen_dic['calc_fring']=True  
    
    
    #%%%% Spectral range(s) to be corrected
    gen_dic['fring_range']=[]
    
     
    #%%%% Plots: correction
    plot_dic['fring_corr']=''   
    
    
        
    ##################################################################################################
    #%%% Module: trimming 
    ##################################################################################################
    
    #%%%% Activating
    gen_dic['trim_spec']=False
    
    
    #%%%% Calculating/retrieving 
    gen_dic['calc_trim_spec']=True  
    
    
    #%%%% Spectral ranges to be kept
    #    - define the spectral range(s) and orders over which spectra should be used in the pipeline
    #      format: [[w1,w2],[w3,w4],..]
    #      in A 
    #    - relevant if spectra are used as input
    #    - adapt the range to the steps that are activated in the pipeline. For example:
    # + spectra will usually need to be kept whole if used to compute CCFs on stellar line profiles
    # + spectra can be limited to specific regions that contain the mask lines used to compute CCFs on stellar / planetary spectra
    # + spectra can be limited to a narrow band if only a few stellar or planetary lines are studied
    #      this module should thus be used to limit spectra to the analysis range while benefitting from corrections derived from the full spectrum in previous modules
    #    - removing specific orders can be useful in complement of selecting specific spectral ranges (that can encompass several orders)
    #    - if left empty, no selection is applied  
    gen_dic['trim_range'] = []
    
    
    #%%%% Orders to be kept   
    #      format: {inst : [ idx_ord ]}
    #    - indexes are relative to order list after exclusion with gen_dic['del_orders'] 
    gen_dic['trim_orders'] = {}
    
        
    
    
    
    ##################################################################################################
    #%% Disk-integrated profiles
    ##################################################################################################  
    
    
    ##################################################################################################
    #%%% Module: CCF conversion for disk-integrated spectra
    #    - before spectra are aligned, but after they have been corrected for systematics, to get data comparable to standard DRS outputs  
    #    - every operation afterwards will be performed on those profiles 
    #    - applied to input data in spectral mode
    ##################################################################################################        
    
    ANTARESS_CCF_settings('DI',gen_dic)

    
    #%%%% Radial velocity table
    # gen_dic['dRV']=0.5    #res. ESPRESSO, EXPRES
    # gen_dic['dRV']=0.82   #res. HARPN 
    if gen_dic['star_name']=='HD189733':      
        gen_dic['start_RV']=-150.    
        gen_dic['end_RV']=150.        
        gen_dic['dRV']=None 
    
    
    ##################################################################################################
    #%%% Module: disk-integrated profiles detrending
    #    - use the 'disk-integrated stellar properties fit' module to derive coefficients for the detrending models. 
    ##################################################################################################
    
    #%%%% Activating
    gen_dic['detrend_prof'] = False
    
    
    #%%%% Calculating/retrieving 
    gen_dic['calc_detrend_prof']=True  
    
    
    #%%%% Spectral correction
    #    - only relevant in spectral mode
    #    - will be applied before spectra are converted into CCFs
    
    #%%%%% Full spectrum
    detrend_prof_dic['full_spec']=True
    
    #%%%%% Transition wavelength
    #    - for single line profile
    #    - in the star rest frame
    detrend_prof_dic['line_trans']=None
    
    
    #%%%%% Order to be corrected
    #    - only relevant for single line profile
    detrend_prof_dic['iord_line']=0
    
    
    #%%%% Trend correction
    
    #%%%%% Activating 
    detrend_prof_dic['corr_trend'] = False
    
    
    #%%%%% Property, coordinate, model     
    #    - structure is : inst > vis > prop_coord > mod > [ coefficients ] or 'path' 
    #                                  prop_c0 > val  
    #    - 'prop_coord' defines which property 'prop' should be corrected as a function of coordinate 'coord', as defined in `glob_fit_dic['DIProp']['mod_prop']`
    #      in that case set 'model' to:
    # + 'pol': polynomial, with coefficients set as [c1,c2,..] 
    # + 'sin': sinusoidal, with coefficients set as [amp,per,off]
    # + 'puls': stellar pulsation model, with coefficients set as [ampHF,phiHF,freqHF,ampLF,phiLF,freqLF,f]
    # + 'ramp': ramp, with coefficients set at [lnk,alpha,tau,xr]
    #      see the models and coefficients definition in `glob_fit_dic['DIProp']['mod_prop']`
    #      instead of coefficients you can provide the path to a 'Fit_results' file from the `glob_fit_dic['DIProp']['mod_prop']`, which must then contain the relevant 'mod' for the requested 'prop' and 'coord'  
    #    - if 'prop_c0' is left undefined the contrast and FWHM correction are normalized around their mean so that the mean level of the time-series is conserved
    #      otherwise set 'prop_c0' to c0_new / c0_old, where 'c0_old' is the level derived with `glob_fit_dic['DIProp']['mod_prop']` and 'c0_new' is the desired level of the corrected time-series
    #      note that setting 'prop_c0' to 1 will naturally set the time-series to the level associated with the correction from `glob_fit_dic['DIProp']['mod_prop']`
    #      this correction can be useful for the contrast and FWHM when a visit is used to define the out-of-transit master profile of another visit, and their line profile need to be made comparable
    #      this field is not relevant for RVs: the mean level of the RV time-series is conserved, and can be corrected for in the  alignment module with the 'sysvel' field
    #    - RV correction must be done in the input rest frame, as CCFs are corrected before being aligned
    #      if a FWHM correction is requested you must perform first the RV correction alone (if relevant), then determine and fix the systemic velocity, then perform the FWHM correction  
    detrend_prof_dic['prop']={}    


    #%%%%% Reference variables for pulsation model
    #    - see 'glob_fit_dic['DIProp']['coord_ref']'
    detrend_prof_dic['coord_ref']={}   
    
            
    #%%%%% SNR orders             
    #    - indexes of orders to be used to define the SNR, for corrections of correlations with snr
    #    - order indexes are relative to original instrumental orders
    detrend_prof_dic['SNRorders']={}   
    
    
    #%%%% PC correction 
    
    #%%%%% Activating
    #    - PCA module must have been ran first to generate the correction
    detrend_prof_dic['corr_PC'] = False
    
    
    #%%%%% PC coefficients from RMR
    #    - for each instrument and visit
    #    - indicate path to 'pca_ana' file to correct all profiles, using the PC profiles and coefficients derived in the module
    #    - the PC model fitted to intrinsic profiles in the PCA module can however absorb the RM signal
    #      if specified, the path to the 'fit_IntrProf' file will overwrite the PC coefficients from the 'pca_ana' fit and use those derived from the joint RMR + PC fit
    #      beware that the routine will still use the PC profiles from the 'pc_ana' file, which should thus be the same used for the 'fit_IntrProf' fit (defined with 'PC_model')
    detrend_prof_dic['PC_model']={}
    
            
    #%%%%% PC profiles
    #    - indexes of PC profiles to apply
    #    - by default should be left undefined, so that all PC fitted to the in- and out-transit differential profiles are used
    #    - this option is however useful to visualize each PC contribution through plot_dic['map_pca_prof']
    detrend_prof_dic['idx_PC']={}
    
    
    #%%%%% Plots: 2D PC noise model 
    #    - using the model generated with the above options
    plot_dic['map_pca_prof']=''   
    
        
    
    if (gen_dic['star_name']=='HD189733'):gen_dic['detrend_prof']=True  &   False    

    #%%%% Trend correction
    
    #%%%%% Activating 
    detrend_prof_dic['corr_trend'] = True    & False

    detrend_prof_dic['full_spec']=False

    #%%%%% Settings   
    if gen_dic['star_name']=='HD189733':
        if (gen_dic['type']=='CCF'):
            detrend_prof_dic['prop']={'ESPRESSO':
                    {'visit1':{'RV_phase':{'pol':1e-3*np.array([ 1.895454e+01])},'FWHM_phase':{'pol':np.array([-5.935886e-03])},'ctrst_phase':{'pol':np.array([ 3.161500e-03])},'ctrst_snrQ':{'pol':np.array([-4.384908e-06])}}}}
        else:
            #RAPPEL: pas de FWHM correction possible pour les spectres
            if ('new_K2' in gen_dic['CCF_mask']['ESPRESSO']):  
                detrend_prof_dic['prop']={'ESPRESSO':{'visit1':{'RV_phaseHD189733b':{'pol':1e-3*np.array([1.527529e+01])},'ctrst_snrQ':{'pol':np.array([-5.941947e-06])}}}} 
                # detrend_prof_dic['prop']={'ESPRESSO':{'visit1':{'RV_phase':{'pol':1e-3*np.array([1.527529e+01])},'ctrst_snrQ':{'pol':np.array([-9.076915e-06])},'ctrst_phase':{'pol':np.array([5.817248e-03])}}}} 
            else:stop('Define for mask')

    
    ##################################################################################################
    #%%% Module: disk-integrated profiles analysis
    #    - can be applied to:
    # + 'fit_DI': profiles in their input rest frame, original exposures, for all formats
    # + 'fit_DI_1D': profiles in their input or star (if aligned) rest frame, original exposures, converted from 2D->1D 
    # + 'fit_DIbin' : profiles in their input or star (if aligned) rest frame, binned exposures, all formats
    # + 'fit_DIbinmultivis' : profiles in the star rest frame, binned exposures, all formats
    #    - the disk-integrated stellar profile (in individual or binned exposures) can also be fitted using model or measured intrinsic profiles to tile the star
    ##################################################################################################
    
    #%%%% Activating
    gen_dic['fit_DI'] = False
    gen_dic['fit_DI_1D'] = False
    gen_dic['fit_DIbin']= False
    gen_dic['fit_DIbinmultivis']= False
    
    
    #%%%% Calculating/Retrieving
    gen_dic['calc_fit_DI']=True    
    gen_dic['calc_fit_DI_1D']=True  
    gen_dic['calc_fit_DIbin']=True  
    gen_dic['calc_fit_DIbinmultivis']=True 
    
    
    #%%%% Fitted data
    ANTARESS_fit_prof_settings('DI',data_dic)

    
    #%%%%% Occulted line exclusion
    #    - exclude range of occulted stellar lines
    #    - all models are assumed to correspond to the disk-integrated star, without planet contamination, with a profile defined as CCFmod(rv,t) = continuum(rv,t)*CCFmod_norm(rv)  
    # + in-transit CCFs are affected by planetary emission, absorption by the planet continuum, and absorption by the planet atmosphere
    #   the narrow, shifting ranges contaminated by the planetary emission and atmospheric absorption can be excluded from the fit and definition of the continuum via data_dic['Atm']['no_plrange'], in which case the CCF approximates as:
    #   CCF(rv,t) = Cref(t)*( CCFstar(rv) - Iocc(rv,t)*alpha(t)*Scont(t) )      where alpha represents low-frequency intensity variations 
    #   because the local stellar line occulted by the planet has a high-frequency profile, in-transit CCFs will not be captured by the CCF model even with the planet masking
    # + if the stellar rotation is truly negligible, we can assume Iocc(rv,t) = I0(rv) and:
    #   CCFstar(rv) = I0(rv)*sum(i,alpha(rv,i)*S(i))          
    #   thus
    #   CCF(rv,t) = Cref(t)*( CCFstar(rv) -CCFstar(rv)*alpha(t)*Scont(t)/sum(i,alpha(rv,i)*S(i)) )    
    #   CCF(rv,t) = Cref(t)*CCFstar(rv)*(1 - Cbias(t) )   
    #   in this case CCFmod can still capture the in-transit CCF profiles, as the bias will be absorbed by continuum(rv,t) that must be left free to vary
    # + however as soon as disk-integrated and intrinsic profiles are not equivalent the above approximation cannot be made
    #   the range that corresponds to the local stellar line absorbed by the planet must be masked, so that outside of the masked regions:
    #   CCF(rv,t) = Cref(t)*( CCFstar(rv) - Icont*alpha(t)*Scont(t) )            
    #   CCF(rv,t) = Cref(t)*( CCFstar(rv) - bias(t) )
    #   in this case CCFmod cannot capture the bias, and a time-dependent offset ('offset' in 'mod_prop') must be included in the model
    #   the continuum flux measured by the pipeline now corresponds to Cref(t)*CCFstar_cont so that it needs to be fitted in addition to the offset
    # + the bias cannot be directly linked with the scaling light curve, because the latter integrates the full intrinsic line profile while Icont is only its continuum
    # + in all cases the continuum range must be defined outside of the range covered by the disk-integrated stellar line 
    # + this method only works if a sufficient portion of the stellar line remains defined after exclusion, ie if the DI line is much larger than the intrinsic one
    #    - the exclusion of occulted stellar lines is only relevant for disk-integrated profiles, and is not proposed for their binning because in-transit profiles will never be equivalent to out-of-transit ones
    #    - the selected 'occ_range' range is centered on the brightness-averaged RV of each occulted stellar region, calculated analytically with the properties set for the star and planet
    #      excluded pixels must then fall within the shifted occ_range and line_range, which defines the maximum extension of the disk-integrated stellar line
    #      format for both ranges is range = {inst : [min_rv,max_rv]}
    #    - only applied to original, unbinned visits
    #    - define [rv1,rv2]] in the star rest frame
    data_dic['DI']['occ_range']={} 
    data_dic['DI']['line_range']={} 


    #%%%%% Continuum range
    #    - used to set the continuum level of models in fits, and for the contrast correction of CCFs
    #      unless requested as a variable parameter in 'mod_prop', the continuum level of the model is fixed to the value measured over 'cont_range'
    #      see details in 'mod_prop' regarding the fitting of the continuum for in-transit profiles
    #    - format: {inst : { order : { [ [x1,x2] , [x3,x4] , [x5,x6] , ... ] }} in the input data frame
    #      ranges will be automatically shifted to the star rest frame when relevant
    #      define x in RV space if data is in CCF mode, and in wavelength space if data is in spectral mode
    data_dic['DI']['cont_range'] = {}
    
    
    #%%%% Direct measurements
    #    - format: {prop_name:{options}}
    #    - possibilities:
    # + equivalent width: 'EW' : {'rv_range':[rv1,rv2] single range over which the integral is performed, defined in the star rest frame}                         
    # + bissector: 'biss' : {'source':'obs' or 'mod',
    #                        'rv_range':[rv1,rv2] maximum range over which bissector is calculated, defined in the star rest frame
    #                        'dF': flux resolution for line profile resampling,
    #                        'Cspan': percentage of line contrast at which to measure bissector span (1 corresponds to line minimum); set to None to take maximum RV deviation from minimum}
    data_dic['DI']['meas_prop']={}
    
    
    #%%%% Line profile model   
    
    #%%%%% Transition wavelength
    #    - in the star rest frame
    #    - used to center the line model, and the stellar / planetary exclusion ranges
    #    - only relevant in spectral mode if the fit is performed with an analytical model on a single line
    data_dic['DI']['line_trans']=None   
    
    
    #%%%%% Instrumental convolution
    #    - apply instrumental convolution or not (default) to model
    #    - beware that most derived properties will correspond to the model before convolution
    #    - should be set to True when using unconvolved profiles of the intrinsic line to fit the master DI
    data_dic['DI']['conv_model']=False
     
    
    #%%%%% Model type
    #    - specific to each instrument
    #    - options:
    # + 'gauss': inverted gaussian, possibly skewed, which is absorbing a polynomial continuum
    # + 'gauss_poly': inverted gaussian , which is absorbing flat continuum with 6th-order polynomial at line center
    # + 'dgauss': gaussian continuum added to inverted gaussian (very well suited to M dwarf CCFs), which is absorbing polynomial continuum
    # + 'voigt': voigt profile, which is absorbing polynomial continuum
    # + 'custom': a model star (grid set through 'theo_dic') is tiled using intrinsic profiles set through 'mod_def'
    #    - it is possible to fix the value, for each instrument and visit, of given parameters of the fit model
    #      if a field is given in 'mod_prop', the corresponding field in the model will be fixed to the given value     
    data_dic['DI']['model']={}
    
    
    #%%%%% Intrinsic line properties
    #    - used if 'model' = 'custom' 
    #    - see gen_dic['mock_data'] for options and settings (comments given here are specific to this module)
    #    - 'mode' = 'ana' 
    # + analytical model properties can be fixed and/or fitted 
    # + mactroturbulence, if enabled, can be fitted for the corresponding properties 
    # + set 'conv_model' to True as the properties describe the unconvolved intrinsic line 
    #    - 'mode' = 'theo'
    # + set 'conv_model' to True as the properties describe the unconvolved intrinsic line 
    #    - 'mode' = 'Intrbin'         
    # + set 'conv_model' to False as the binned profiles are already convolved (it is stil an approximation, but better than to convolve twice)
    data_dic['DI']['mod_def']={}  
    
    
    #%%%%% Fixed/variable properties
    #    - format is:
    # mod_prop = { prop_name : { 'vary' : bool , 'inst' : { 'visit' : { 'guess': x, 
    #                                                                   'bd': [x_low,x_high] OR 'gauss': [val,s_val] } } } }
    #      where 
    # > 'prop_name' defines a property of the selected model
    # > 'vary' indicates whether the parameter is fixed or variable
    #   if 'vary' = True:
    #       'guess' is the guess value of the parameter for a chi2 fit, also used in any fit to define default constraints
    #       walkers' starting positions are randomly drawn from a uniform (defined by 'bd') or gaussian (defined by 'gauss') distribution for a mcmc/ns fit
    #   if 'vary' = False:
    #       'guess' is the constant value of the parameter
    #   'guess' and 'bd' can be specific to a given instrument and visit
    #    - default values will be used if left undefined, specific to each model
    data_dic['DI']['mod_prop']={}
    
    
    #%%%%% Best model table
    #    - resolution (dx) and range (min_x,max_x) of final model used for post-processsing of fit results and plots
    #    - in rv space and km/s for analytical profiles (profiles in wavelength space are modelled in RV space and then converted), in space of origin for measured profiles, in wavelength space for theoretical profiles 
    #    - specific to the instrument
    data_dic['DI']['best_mod_tab']={}
    
    
    #%%%% Fit settings     
    ANTARESS_fit_def_settings('DI',data_dic,plot_dic)
    
    
    #%%%% Plot settings

    #%%%%% Individual disk-integrated profiles
    plot_dic['DI_prof']=''     
    
    
    #%%%%% Residuals from disk-integrated profiles
    plot_dic['DI_prof_res']=''   
    
    
    #%%%%% Housekeeping and derived properties 
    plot_dic['prop_DI']=''  
    
        
    #Activating
    gen_dic['fit_DI'] = True    &  False
    gen_dic['fit_DIbin']=True   &  False
    gen_dic['fit_DIbinmultivis']=True    &  False

    if gen_dic['star_name']=='HD189733':
        gen_dic['fit_DI'] = True    &  False
        gen_dic['calc_fit_DI']=True    &  False   


    #Calculating/Retrieving
    gen_dic['calc_fit_DI']=True    &  False   
    gen_dic['calc_fit_DIbin']=True   &  False  
    gen_dic['calc_fit_DIbinmultivis']=True    &  False  

    #Fitted data

    #Constant data errors
    data_dic['DI']['cst_err']=True   &  False
    data_dic['DI']['cst_errbin']=True   &  False

    #Scaled data errors

    #Occulted line exclusion 

    #Trimming 
    data_dic['DI']['fit_prof']['trim_range']={}


    #Order to be fitted
    if gen_dic['star_name']=='TOI3884':
        data_dic['DI']['fit_prof']['order']={'MIKE_Red':0} 

    if gen_dic['star_name']=='AUMic':
        data_dic['DI']['fit_prof']['order']={'ESPRESSO':0}     
    
    #if gen_dic['star_name']=='AU_Mic':
    #    data_dic['DI']['fit_prof']['order']={'ESPRESSO':0}     

    if gen_dic['star_name']=='V1298tau':
        data_dic['DI']['fit_prof']['order']={'ESPRESSO':0}     

    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        data_dic['DI']['fit_prof']['order']={'ESPRESSO':0} 

    #Continuum range
    if gen_dic['star_name']=='TOI3884':data_dic['DI']['cont_range']['MIKE_Red']={0:[[-30.,-20.],[20.,30.]]} 

    if gen_dic['star_name']=='AUMic':data_dic['DI']['cont_range']['ESPRESSO']={0:[[-100.,-80.],[80.,100.]]} 
        
    if gen_dic['star_name']=='AU_Mic':data_dic['DI']['cont_range']['ESPRESSO']={0:[[-25,-15.],[10.,20.]]} 

    if gen_dic['star_name']=='V1298tau':data_dic['DI']['cont_range']['ESPRESSO']={0:[[14.-90.,14.-40.],[14.+40.,14.+90.]]} 

    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:data_dic['DI']['cont_range']['ESPRESSO']={0:[[-25., -20.],[10.,15.]]} 

    if gen_dic['star_name']=='HD189733':
        sysguess = -2.
        data_dic['DI']['cont_range']['ESPRESSO']={0:[[sysguess-90.,sysguess-30.],[sysguess+30.,sysguess+90.]]}        


    #Spectral range(s) to be fitted
    if gen_dic['star_name']=='TOI3884':data_dic['DI']['fit_range']['MIKE_Red']={'mockvis':[[-20., 20.]]}  

    if gen_dic['star_name']=='AUMic':
        data_dic['DI']['fit_range']['ESPRESSO']={
        'mock_vis':[[-100.,100.]], #--base
        'mock_vis1':[[-100.,100.]],
        'mock_vis2':[[-100.,100.]],
        'mock_vis3':[[-100.,100.]],
        'mock_vis4':[[-100.,100.]],
        'mock_vis5':[[-100.,100.]],
        'mock_vis6':[[-100.,100.]],
        'mock_vis7':[[-100.,100.]],
        'mock_vis8':[[-100.,100.]],
        'mock_vis9':[[-100.,100.]],
        }  

    if gen_dic['star_name']=='V1298tau':data_dic['DI']['fit_range']['ESPRESSO']={'mock_vis':[[14.-90.,14.+90.]]}  
      
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:data_dic['DI']['fit_range']['ESPRESSO']={'mock_vis':[[-100.,100.]]}  

    if gen_dic['star_name']=='AU_Mic':data_dic['DI']['fit_range']['ESPRESSO']={'visit1':[[-25., 20]]}  

    if gen_dic['star_name']=='HD189733':
        data_dic['DI']['fit_range']['ESPRESSO']={'visit1':[[sysguess-90.,sysguess+90.]]}       

    #Direct measurements
    # data_dic['DI']['meas_prop']={
    #     # 'EW':{'rv_range':[-5.,5.]},
    #     'biss':{'source':'obs','rv_range':[-50.,50.],'dF':0.001,'Cspan':None}
    #     }


    #Line profile model   
    
    #Transition wavelength
    data_dic['DI']['line_trans']=None


   
    #Instrumental convolution
    data_dic['DI']['conv_model']=False    
            

    #Model type    
    if gen_dic['star_name']=='TOI3884':    
        data_dic['DI']['model']['MIKE_Red']='gauss'

    if gen_dic['star_name']=='AUMic':    
        data_dic['DI']['model']['ESPRESSO']='gauss'

    if gen_dic['star_name']=='AU_Mic':    
        data_dic['DI']['model']['ESPRESSO']='gauss'

    if gen_dic['star_name']=='V1298tau':    
        data_dic['DI']['model']['ESPRESSO']='gauss'

    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:    
        data_dic['DI']['model']['ESPRESSO']='gauss'
    
    if gen_dic['star_name']=='HD189733':
        data_dic['DI']['model']['ESPRESSO']='gauss' 
    #Intrinsic line properties  
    

    #Fixed/variable properties
        #    - structure is mod_prop = { 'par_name' : { 'vary' : bool , 'inst' : { 'visit' : {'guess':X , 'bd':[Y,Z] } } } }
    if gen_dic['star_name']=='AU_Mic':    
        data_dic['DI']['mod_prop']={'FWHM':{'vary':True, 'ESPRESSO':{'visit1':{'guess':5}}},#, 'bd':[1., 50.] } } },
                                'rv':{'vary':True, 'ESPRESSO':{'visit1':{'guess':-5}}},#, 'bd':[-4., -7.] } } },
                                'ctrst':{'vary':True, 'ESPRESSO':{'visit1':{'guess':0.6}}}}#:0.1, 'bd':[0., 1.] } } },
    
    if gen_dic['star_name']=='TOI3884':    
        data_dic['DI']['mod_prop']={'FWHM':{'vary':True, 'MIKE_Red':{'mockvis':{'guess':5}}},#, 'bd':[1., 50.] } } },
                                'rv':{'vary':True, 'MIKE_Red':{'mockvis':{'guess':0.}}},#, 'bd':[-4., -7.] } } },
                                'ctrst':{'vary':True, 'MIKE_Red':{'mockvis':{'guess':0.6}}}}#:0.1, 'bd':[0., 1.] } } },

    if gen_dic['star_name']=='HD189733':
        data_dic['DI']['mod_prop']={'rv':{'vary':True ,'ESPRESSO':{'visit1':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
                                    'ctrst':{'vary':True  ,'ESPRESSO':{'visit1':{'guess':0.5,'bd':[0.1,0.9]}}},
                                    'FWHM':{'vary':True  ,'ESPRESSO':{'visit1':{'guess':7.,'bd':[2.,30.]}}}}         

    #Best model table
    
    #Fitting mode 
    data_dic['DI']['fit_mode']='chi2'   
    # data_dic['DI']['fit_mode']='mcmc' 
    # data_dic['DI']['fit_mode']=''  
    
    #Printing fits results
    data_dic['DI']['verbose']=True  & False

   
    #Priors on variable properties
    data_dic['DI']['line_fit_priors']={}
    if gen_dic['star_name'] in ['AU_Mic','TOI3884']:
        data_dic['DI']['line_fit_priors']={'rv':{'mod': 'uf', 'low':0.,'high':100.},
                                    'ctrst':{'mod': 'uf', 'low':0.,'high':1.},
                                    'FWHM':{'mod': 'uf', 'low':0.1,'high':40.},                                    
                                    }

    if gen_dic['star_name']=='HD189733':
        data_dic['DI']['priors']={'veq':{'mod': 'uf', 'low':0.,'high':20.}}         

    #Derived properties
    deriv_prop=[]
    deriv_prop=['amp','area','true_ctrst','true_FWHM','true_amp']   #generic
    deriv_prop+=['FWHM_LOR','FWHM_voigt']   #voigt
    deriv_prop+=['cont_amp','RV_lobe','amp_lobe','FWHM_lobe']    #double-gaussian
    deriv_prop+=['vsini']    #custom
    deriv_prop=[]
    data_dic['DI']['deriv_prop'] = {}
    for par_loc in deriv_prop:data_dic['DI']['deriv_prop'][par_loc]={}

    #Calculating/retrieving
    data_dic['DI']['run_mode']='use'
    
    
    #Walkers
        
    #Walkers exclusion        
    data_dic['DI']['exclu_walk']=True     & False           
     
    
    #Derived errors
    data_dic['DI']['out_err_mode']='HDI'
    data_dic['DI']['HDI']='1s'   #None   #'3s'  
    
    #Derived lower/upper limits
    # data_dic['DI']['conf_limits']={'veq':{'bound':0.,'type':'upper','level':['1s','3s']}}      




    #Plot settings
    
    #1D PDF from mcmc
    plot_dic['prop_DI_PDFs']=''                 
        
    #Individual disk-integrated profiles
    plot_dic['DI_prof']=''   

    #Residuals from disk-integrated profiles
    plot_dic['DI_prof_res']=''   #pdf

    #Housekeeping and derived properties 
    plot_dic['prop_DI']=''   #''          
    # if gen_dic['star_name']=='HD189733':plot_dic['prop_DI']='pdf'

    
  #Turn this on and look at the properties


    # #Stage Théo : new fitting settings for V1298tau artificial visit


    # if gen_dic['mock_data'] : 
    
    
    #     if gen_dic['star_name']=='V1298tau':
    #         data_dic['DI']['model']['HARPN']='gauss'
    #         data_dic['DI']['cont_range']['HARPN']=[[-150.,-70.],[70.,150.]]    
    #         data_dic['DI']['fit_range']['HARPN']= { 'mock_vis' : [[-50, 50]]  }
    

    #         data_dic['DI']['mod_def']['HARPN']={'mode':'ana','coord_line':'mu','model':'gauss'} 
    #         data_dic['DI']['mod_prop']={}
    #         data_dic['DI']['mod_prop'].update({
    #                                         'rv':{'vary':True     ,'HARPN':{'mock_vis':{'guess':0,'bd':[-10.,10.]}}},
    #                                         'veq':{'vary':False   ,'HARPN':{'mock_vis':{'guess':23.5,'bd':[20.,30.]}}},                             
    #                                         'ctrst_ord0__IS__VS_':{'vary':True  ,'HARPN':{'mock_vis':{'guess':0.7,'bd':[0.2,1]}}},
    #                                         'FWHM_ord0__IS__VS_' :{'vary':True  ,'HARPN':{'mock_vis':{'guess':4,'bd':[0.,10.]}}},
    #                                         })  
                                            
      


    ##################################################################################################       
    #%%% Module: disk-integrated stellar properties fit
    #    - fitting single stellar disk-integrated property with a common model for all instruments/visits, or independently for each visit 
    #    - with properties derived from individual disk-integrated profiles
    #    - this module is used to derive the detrending models to be applied to disk-integrated profiles
    ##################################################################################################       
    
    #%%%% Activating 
    gen_dic['fit_DIProp'] = False
    
    
    #%%%% Multi-threading
    glob_fit_dic['DIProp']['nthreads'] = int(0.8*cpu_count())
    

    #%%%%% Unthreaded operations
    #    - all operations are multi-threaded by default, but overheads of sharing data between threads may counterbalance the benefits of threading the model
    #    - select here which operations not to thread:
    # + 'emcee'
    glob_fit_dic['DIProp']['unthreaded_op'] = []  
    
    
    #%%%% Fitted data
    
    #%%%%% Exposures to be fitted
    #    - indexes are relative to global tables
    #    - define instruments and visits to be fitted (they will not be fitted if not used as keys, or if set to [], which is the default value), set their value to 'all' for all out-transit exposures to be fitted
    #    - add '_bin' at the end of a visit name for its binned exposures to be fitted instead of the original ones (must have been calculated with the binning module)
    #      all other mentions of the visit (eg in parameter names) can still refer to the original visit name
    glob_fit_dic['DIProp']['idx_in_fit'] = {}


    #%%%%% Scaled data errors
    #    - local scaling of data errors
    #    - you can scale by sqrt(reduced chi2 of original fit) to ensure a reduced chi2 unity
    glob_fit_dic['DIProp']['sc_err']={}  


    #%%%% Fitted properties
    
    #%%%%% Property, coordinate, model
    #    - format is:
    # mod_prop = { 
    #  prop : { c__ord0__ISinst_VSvis':{'vary':True ,'guess':x,'bd':[x1,x2]} , ...}, 
    #           coord__pol__ordN__ISinst_VSvis':{'vary':True ,'guess':x,'bd':[x1,x2]} , ...},    
    #           coord__sin__Y__ISinst_VSvis':{'vary':True ,'guess':x,'bd':[x1,x2]} , ...}  
    #           coord__ramp__Y__ISinst_VSvis':{'vary':True ,'guess':x,'bd':[x1,x2]} , ...} 
    #           coord__puls__Y__ISinst_VSvis':{'vary':True ,'guess':x,'bd':[x1,x2]} , ...}
    #           }
    #    - 'prop' defines the measured property to be fitted:
    # + rv : residuals between disk-integrated RVs and the Keplerian model (km/s)
    # + ctrst : disk-integrated line contrast
    # + FWHM : disk-integrated line FWHM (km/s)   
    #    - 'coord' defines the coordinate as a function of which the property is modelled:
    # + 'time': absolute time in bjd
    # + 'starphase': stellar phase
    # + 'phasePlName' : orbital phase for planet 'PlName' 
    #                   this option allows for stellar line variations phased (and possibly induced) by a planet        
    # + 'AM', 'snr', 'snrQ' (for the SNR of orders provided as input to be summed quadratically - useful to combine ESPRESSO slices - rather than being averaged)
    # + 'ha', 'na', 'ca', 's', 'rhk' 
    #    - property can be modelled as a:
    # + 'c__ord0__ISinst_VSvis': constant level c0
    # + polynomial : pol(x) = c1*x + c2*x^2 + ...
    #                defined by coefficients cN = 'coord__pol__ordN__ISinst_VSvis', with N>0
    # + sinusoidal : sine(x) = amp*sin((x-off)/per)
    #                defined by its amplitude (Y='amp'), period (Y='per'), and offset (Y='off') in 'coord__sin__X__ISinst_VSvis'     
    # + ramp : ramp(x) = 1 - k*exp( - ((x - xr)/tau)^alpha )  
    #          defined by its log(amplitude) (Y='lnk'), exponent (Y='alpha'), decay constant (Y='tau'), and offset (Y='xr') in 'coord__ramp__X__ISinst_VSvis'     
    #          for contrast and FWHM only
    # + pulsation : puls(x) =  ampHF (1 + ampLF sin( 2pi x freqLF - phiLF) ) sin( 2pi x freqHF(x) - phiHF ), where freqHF(x) = freqHF*(1 + f*sin( 2pi x freqLF - phiLF))  
    #               defined by its HF amplitude (Y='ampHF'), HF offset (Y='phiHF'), nominal HF frequency (Y='freqHF'), LF amplitude (Y='ampLF'), LF offset (Y='phiLF'), LF frequency (Y='freqLF'), and frequency modulation (Y='f') in 'coord__puls__X__ISinst_VSvis'     
    #    - the same model of different coordinates, or different models of the same coordinates, can be defined and combined:  
    # + through multiplication, for contrast and FWHM models :
    #      F(x) = c0*(1 + pol(x) )*(1+sine(x))*ramp(x)*puls(x)
    #   with c0 in km/s for the FWHM  
    # + through addition, for rv models:
    #      F(x) = c0 + pol(x) + sine(x) + puls(x)
    #   with ci and Amp in km/s    
    glob_fit_dic['DIProp']['mod_prop']={
        'rv':{'c__ord0__IS__VS_':{'vary':True ,'guess':0,'bd':[-100.,100.]},
              'time__pol__ord1__IS__VS_':{'vary':True ,'guess':0,'bd':[-100.,100.]}},
        'ctrst':{'c__ord0__IS__VS_':{'vary':True ,'guess':0,'bd':[-100.,100.]},
                 'snr__pol__ord1__IS__VS_':{'vary':True ,'guess':0,'bd':[-100.,100.]}},
        'FWHM':{'c__ord0__IS__VS_':{'vary':True ,'guess':0,'bd':[-100.,100.]},
                'time__pol__ord1__IS__VS_':{'vary':True ,'guess':0,'bd':[-100.,100.]}},
        }
    
    #%%%%% Reference variables for pulsation model
    #    - because the model is defined as a sine, fit convergence is poor when the coordinates have large values
    #      for each coordinate used in 'coord__puls__X__ISinst_VSvis', define a reference value so that coord - coord_ref is small
    #    - we request the user to provide these values so that they remain fixed for various visits (as the phase offset is correlated with this reference)
    #    - format is : coord_ref : { coord : val }
    glob_fit_dic['DIProp']['coord_ref']={}    


    #%%%%% SNR orders             
    #    - indexes of orders to be used to define the SNR, for corrections of correlations with snr
    #    - order indexes are relative to original instrumental orders
    glob_fit_dic['DIProp']['SNRorders']={}   
    
    
    #%%%% Fit settings
    ANTARESS_fit_def_settings('DIProp',glob_fit_dic,plot_dic)


    if gen_dic['star_name']=='HD189733':

        gen_dic['fit_DIProp'] = True & False

        glob_fit_dic['DIProp']['idx_in_fit'] = {'ESPRESSO':{'visit1':'all'}}

        glob_fit_dic['DIProp']['mod_prop']={
        'rv':{'c__ord0__IS__VS_':{'vary':True ,'guess':0,'bd':[-100.,100.]},
              'time__pol__ord1__IS__VS_':{'vary':True ,'guess':0,'bd':[-100.,100.]}},
        'ctrst':{'c__ord0__IS__VS_':{'vary':True ,'guess':0,'bd':[-100.,100.]},
                 'snr__pol__ord1__IS__VS_':{'vary':True ,'guess':0,'bd':[-100.,100.]}},
        }

        # RV_phaseHD189733b
        # ctrst_snrQ




    
    
    
                                

    ##################################################################################################       
    #%%% Module: joined disk-integrated profiles fit    
    #    - not implemented for now
    ##################################################################################################     
            
    #%%%% Activating 
    gen_dic['fit_DIProf'] = False  
    
    
    

    ##################################################################################################
    #%%% Module: disk-integrated profiles alignment         
    ##################################################################################################
    
    #%%%% Activating
    gen_dic['align_DI'] = False  
    
    
    #%%%% Calculating/retrieving 
    gen_dic['calc_align_DI'] = True  
    
    
    #%%%% Systemic velocity 
    #    - for each instrument and visit (km/s)
    #    - this is the velocity of the center of mass of the system, relative to the Sun (or the input reference frame)
    #    - if the value is unknown, or a precise measurement is required for each visit, one can use input CCFs or CCFs from input spectra to determine it
    #      first set 'sysvel' to 0 km/s, then run a preliminary analysis to derive its value from the CCF, and update 'sysvel'
    #      it can be determined either from the centroid of the master out-of-transit (calculated with gen_dic['DIbin']) or from the mean value of the out-of-transit RV residuals from the keplerian model (via plot_dic['prop_DI'])
    #    - beware of using published values, because they can be derived from fits to many datasets, while there are still small instrumental offsets in the RV series in a given visit 
    #      (also, we are using the RV in the fits files which is not corrected for the secular acceleration)
    #    - when using spectra the value can be modified without running again the initialization module gen_dic['calc_proc_data'] and spectral correction modules, but any processing modules must still be re-run if the systemic velocity is changed
    #      if CCFs are given from input the pipeline must be fully re-run
    data_dic['DI']['sysvel']={}
    
    
    #%%%% Plots: aligned disk-integrated profiles
    #    - plotting all aligned DI profiles together in star rest frame
    plot_dic['all_DI_data']=''      
    
    
    
        
    #Activating
    gen_dic['align_DI'] = True    #&  False          
    
    #Calculating/retrieving 
    gen_dic['calc_align_DI']=True    #&  False  
        
    #Systemic velocity        
    if gen_dic['star_name']=='TOI3884':
        data_dic['DI']['sysvel']={'MIKE_Red' : 
                                    {'mockvis' : 0, #--base
                        }}

    if gen_dic['star_name']=='TRAPPIST1':
        data_dic['DI']['sysvel']={'NIRPS_HE' : 
                                    {'mockvis' : 0, #--base
                        }}

    if gen_dic['star_name']=='AUMic':
        data_dic['DI']['sysvel']={'ESPRESSO' : 
                                    {'mock_vis' : 0, #--base
                                    'mock_vis1' : 0,
                                    'mock_vis2' : 0,
                                    'mock_vis3' : 0,
                                    'mock_vis4' : 0,
                                    'mock_vis5' : 0,
                                    'mock_vis6' : 0,
                                    'mock_vis7' : 0,
                                    'mock_vis8' : 0,
                                    'mock_vis9' : 0,
                                    } 
                        }
     
    if gen_dic['star_name']=='fakeAU_Mic':
        data_dic['DI']['sysvel']={'ESPRESSO' : 
                                    {'mockvisit1' : 0}} #--base

    if gen_dic['star_name']=='V1298tau':
        data_dic['DI']['sysvel']={'ESPRESSO' : {'mock_vis' : 0}} 

    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        data_dic['DI']['sysvel']={'ESPRESSO' : {'mock_vis' : 0}} 

    if gen_dic['star_name']=='AU_Mic':
        data_dic['DI']['sysvel']={'ESPRESSO' : {'visit1' : -6.31}} #from Gaia DR2 and Palle+2020 seem to use the same

    if gen_dic['star_name']=='HD189733':
        data_dic['DI']['sysvel']['ESPRESSO']={'visit1':-0.} 
        if (gen_dic['type']=='CCF'):
        
            #CCF default DRS mask
            data_dic['DI']['sysvel']['ESPRESSO']={'visit1':-0.0512}  #From Mout
            data_dic['DI']['sysvel']['ESPRESSO']={'visit1':-0.0510852865}  #From RVres
            data_dic['DI']['sysvel']['ESPRESSO']={'visit1':-0.0510852865-1e-3*3.572992e-01}  #From RVres, trend-corr
        
        else:
            if ('new_K2' in gen_dic['CCF_mask']['ESPRESSO']):data_dic['DI']['sysvel']['ESPRESSO']={'visit1':-2.22896482}    #Mask K2, reduction pour Dany (+ ref. pour generation du masque)
            else:data_dic['DI']['sysvel']['ESPRESSO']={'visit1':-2.22819514}     

    #Plots: aligned disk-integrated profiles
    plot_dic['all_DI_data']=''     #pdf    
                        

    
        
    ##################################################################################################
    #%%% Module: broadband flux scaling
    #    - define here transit properties     
    ##################################################################################################
    
    #%%%% Activating
    gen_dic['flux_sc'] = False
    
    
    #%%%% Calculating/retrieving
    gen_dic['calc_flux_sc']=True  
    
    
    #%%%% Scaling disk-integrated profiles
    #    - this option should be disabled if absolute photometry is used or if ANTARESS is applied to mock profiles 
    #      in that case, the module calculates broadband flux variations to convert local profiles into intrinsic profiles, but they are not used to rescale disk-integrated profiles
    data_dic['DI']['rescale_DI'] = True    
    
    
    #%%%% Scaling spectral range
    #    - controls the range over which the flux is scaled, which should correspond to the one over which the light curve is defined
    #      it should thus also include ranges affected by planetary absorption, and the full line profile in the case of CCFs
    #    - define [ [x1,x2] , [x3,x4] , [x5,x6] , ... ] in the star rest frame, in km/s or A
    #    - leave empty for the full range of definition of the profiles to be used
    data_dic['DI']['scaling_range']=[]    
    
    
    #%%%% Out scaling flux
    #    - value of the out-of-transit flux in the scaling range
    #    - scaling value is set to the median out-of-transit flux in each visit if field is None
    data_dic['DI']['scaling_val']=1. 
    
    
    #%%%% Stellar and planet intensity settings
    #    - broadband stellar intensity properties (limb-darkening LD and/or gravity-darkening GD) and planet-to-star radius ratios need to be defined in 'system_prop' for each planet whose transit is studied, even if no scaling is applied    
    #    - format is 'system_prop' : {
    # 'achrom':{'LD':[LD_law],'LD_u1':[u1],'LD_u2':[u2],..,PlName:[RpRs]},
    # 'chrom': {'w':    [wA,     wB,     ..],
    #           'LD':   [LD_lawA,LD_lawB,..],
    #           'LD_u1':[u1_A,   u1_B,..],    
    #           'LD_u2':[u2_A,   u2_B,..],
    #           ...
    #           PlName: [RpRsA,  RpRsB, ..]}}
    #      with
    # + 'LD' : limb-darkening law
    #   'LD_ui' with i>=1 : limb-darkening coefficients
    #    possible limb-darkening laws (name, number of coefficients) :
    #       uniform (uf,0), linear(lin,1), quadratic(quad,2), squareroot(sr,2), logarithmic(log,2), exponential(exp,2), power2 (pw2, 2), nonlinear(nl,4)
    #    LD coefficients can be derived at first order with http://astroutils.astronomy.ohio-state.edu/exofast/limbdark.shtml   
    #    consider using the Limb Darkening Toolkit (LDTk, Parviainen & Aigrain 2015) tool do determine chromatic LD coefficients     
    # + PlName : planet-to-star radius ratio    
    #   can be defined from the transit depth = (Rpl/Rstar)^2     
    #    - the achromatic set ('achrom') must always be defined and is used:
    # + to scale input data, unless 'chrom' is used
    # + to define the transit contacts
    # + to calculate theoretical properties of the 'average' planet-occulted regions throughout the pipeline. The spectral band of the properties should thus match that over which measured planet-occulted properties were derived      
    #    - the chromatic set ('chrom') is used:  
    # + to calculate model or simulated light curves (or define the bands of input light curves) used to scale chromatically disk-integrated spectra
    # + to calculate chromatic RVs of planet-occulted regions used to align intrinsic spectral profiles (as those RVs are flux-weighted, and thus depend on the chromatic RpRs and LD)    
    #      in that case the fields are list of values associated with chromatic bands centered at wavelengths 'w'
    #      the chromatic bands are common to the star and all studied planets, so they should be sampled enough to resolve the shortest-frequency variations of stellar intensity and planet transit depths.
    #      if CCFs are used, if 'chrom' is not provided, or is provided with a single band, 'achrom' will be used automatically
    #    - if the star is oblate (defined through ANTARESS_system_properties), then GD can be accounted for in the same way as LD in the 'achrom' set
    #      note that this is not necessary, oblateness can be considered without inclusion of GD
    #      if requested GD is estimated based on a stellar blackbody flux, integrated between 'GD_min':[val] and 'GD_max':[val], at the resolution 'GD_dw':[val] 
    data_dic['DI']['system_prop']={}
      
    
    #%%%% Active region intensity settings
    #    - same format as 'system_prop'
    data_dic['DI']['ar_prop']={}
    
    
    #%%%% Transit light curve model
    #    - there are several possibilities to define the light curves, specific to each instrument and visit, via 'transit_prop':inst:vis
    # > Imported : {'mode':'imp','path':path}     
    #   + indicate the path of a file that must be defined with 1+nband columns equal to absolute time (BJD, at the time of the considered visit) and the normalized stellar flux (in/out, one column per each spectral band in 'system_prop')
    # > Modeled: {'mode':'model','dt':dt}
    #   + the batman package is used to calculate the light curve
    #   + a single planet can be taken into account. 
    #     the properties of the planet transiting in the visit are taken from 'planets_params', except for RpRs and the limb-darkening properties taken from 'transit_prop':'chrom'
    #   + dt is the time resolution of the light curve (in min)
    # > Simulated: {'mode':'simu','n_oversamp':n}
    #   + a numerical grid is used to calculate the light curves 
    #     'nsub_Dstar' controls the grid used to calculate the total stellar flux (number of subcells along the star diameter, odd number)
    #     set to None if simulated light curves are not needed 
    #   + multiple planets can be taken into account. Properties of the planet(s) transiting in the visit are taken from 'planets_params' and 'system_prop':'chrom'
    #   + n_oversamp is the oversampling factor of RpRs (set to 0 to prevent oversampling)
    #    - specific light curves can thus be defined for each visit via 'imp' (to account for changes in the planet luminosity and surface properties) or via 'sim' (to account for multiple transiting planets)
    #    - light curves can be chromatic and will be interpolated over input spectra, or achromatic for CCFs 
    #      use CCFs only for cases when the stellar emission does not vary between visits, or varies in the same way for all local stellar spectra 
    #      to account for changes in stellar brightness during some visits (eg due to spots/plages unocculted by the planet), use spectra rather than CCF, input the relevant stellar spectra 
    #      to set data to the right color balance, and the light curves normalized by the corresponding spectra. The absolute flux level of the stellar spectra does not matter, only their relative flux.
    #    - light curves imported or calculated are averaged within the duration of observed exposures, and must thus be defined at sufficient high resolution 
    data_dic['DI']['transit_prop']={'nsub_Dstar':None}
    
      
    #%%%% Forcing in/out transit flag
    #    - the user can force whether an exposure is considered in- or out-of-transit
    #      this can be useful if the planet only transits the star for a negligible fraction of an exposure
    #    - indexes are relative to the global table in each visit
    #    - will not modify the automatic in/out attribution of other exposures
    data_dic['DI']['idx_ecl']={}
    
    
    
    #%%%% Plot settings
    
    #%%%%% Model time resolution 
    #    - in sec
    plot_dic['dt_LC']= 30.   
    
    #%%%%% Input light curves 
    plot_dic['input_LC']='' 
    
    #%%%%% Scaling light curves
    #    - over a selection of wavelengths
    #    - for input spectra only
    plot_dic['spectral_LC']=''  
    
    #%%%%% 2D maps of disk-integrated profiles
    #    - in star rest frame 
    #    - data at different steps can be plotted from within the function: corrected for systematics, then aligned, then scaled
    #    - allows checking for spurious variations, visualizing the exposures used to build the master, the ranges excluded because of planetary contamination
    #      scaling light curve can be temporarily set to unity above to compare in-transit profiles with the rest of the series
    #    - too heavy to plot in another format than png
    plot_dic['map_DI_prof']=''  
    
        
        
    #Activating
    gen_dic['flux_sc']=True   #& False
    
    
    #Calculating/retrieving
    gen_dic['calc_flux_sc']=True  &  False    
    
        

    #Scaling disk-integrated profiles
    data_dic['DI']['rescale_DI'] = True    

    #Scaling spectral range
    # if gen_dic['star_name']=='WASP76':data_dic['DI']['scaling_range']=[]
    
    #Out scaling flux
    data_dic['DI']['scaling_val']=1. 

    #Stellar and planet intensity settings 
    if gen_dic['star_name']=='TOI3884':
        data_dic['DI']['system_prop']={
                'achrom':{
                    # 'TOI3884_b' : [0.197], #Libby-Roberts+2023
                    'TOI3884_b' : [0.1899], #Almenara+2022
                    'LD' : ['quadratic'],
                    'LD_u1' : [0.1155],    #Almenara+2022
                    'LD_u2' : [0.3578],    #Almenara+2022
                }
                } 

    if gen_dic['star_name']=='TRAPPIST1':
        data_dic['DI']['system_prop']={
                'achrom':{
                    'TRAPPIST1_b' : [0.08590], #Gilbert et al. 2022
                    'TRAPPIST1_c' : [0.08440], #Gilbert et al. 2022
                    'TRAPPIST1_d' : [0.06063], #Gilbert et al. 2022
                    'TRAPPIST1_e' : [0.07079], #Gilbert et al. 2022
                    'TRAPPIST1_f' : [0.08040], #Gilbert et al. 2022
                    'TRAPPIST1_g' : [0.08692], #Gilbert et al. 2022
                    'TRAPPIST1_h' : [0.05809], #Gilbert et al. 2022
                    'LD' : ['quadratic'],
                    'LD_u1' : [0.168],
                    'LD_u2' : [0.245],
                }
                }   

    if gen_dic['star_name']=='AUMic':
        data_dic['DI']['system_prop']={
                'achrom':{
                    'AUMicb' : [0.0512], #Gilbert et al. 2022
                    # 'AUMicc' : [0.1], #Gilbert et al. 2022
                    'LD' : ['quadratic'],
                    'LD_u1' : [0.35],
                    'LD_u2' : [0.16],
                }
                }   

    if gen_dic['star_name']=='fakeAU_Mic':
        data_dic['DI']['system_prop']={
                'achrom':{
                    'fakeAU_Mic_b' : [0.0488], #Gilbert et al. 2022
                    # 'AUMicc' : [0.1], #Gilbert et al. 2022
                    'LD' : ['quadratic'],
                    'LD_u1' : [0.63],
                    'LD_u2' : [0.15],
                }
                }   

    if gen_dic['star_name']=='V1298tau':
        data_dic['DI']['system_prop']={
                'achrom':{
                    'V1298tau_b' : [0.0700],
                    'LD':['linear'],
                    'LD_u1' : [0.41]
                }
                }

    #Zodiacs
    for zodiac in ['Capricorn','Cancer','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        if gen_dic['star_name']==zodiac:
            data_dic['DI']['system_prop']={
                    'achrom':{
                        zodiac_pl : [0.0512],
                        'LD' : ['quadratic'],
                        'LD_u1' : [0.35],
                        'LD_u2' : [0.16],
                    }
                    }  
    if gen_dic['star_name']=='Gemini':
        data_dic['DI']['system_prop']={
                'achrom':{
                    'Gemini_b' : [0.1339],
                    'LD' : ['quadratic'],
                    'LD_u1' : [0.35],
                    'LD_u2' : [0.16],
                }
                }
    if gen_dic['star_name']=='Sagittarius':
        data_dic['DI']['system_prop']={
                'achrom':{
                    'Sagittarius_b' : [0.0122],
                    'LD' : ['quadratic'],
                    'LD_u1' : [0.35],
                    'LD_u2' : [0.16],
                }
                }



    if gen_dic['star_name']=='AU_Mic':
        data_dic['DI']['system_prop']={
                'achrom':{
                    'AU_Mic_b' : [0.0488], #Wittrock+2023
                    'LD' : ['quadratic'],
                    'LD_u1' : [0.63], #Wittrock+2023 -> In V band
                    'LD_u2' : [0.15], #Wittrock+2023 -> In V band
                }
                }  

    if gen_dic['star_name']=='HD189733':  
        #I choose to use results from Pont+2013 (not RM-based results like Cristo+2023 because they depend on their RM model and are lower-precision)
        #I take their spot-corrected values (like Cristo+2023, I checked) from Table 6, excluding the high-res Na bin
        #LD laws from Hayek+2012 (like Pont+2013); they used non-linear law from Claret+2000; 
        #See calculations of final values in /Users/bourrier/Travaux/ANTARESS/Ongoing/HD189733b_Saved_data/Broadband_scaling_data/HD189_broadband_scaling.py
        data_dic['DI']['system_prop']={                            
            'chrom':{'w':np.array([3450., 3950., 4450., 4950., 5400., 5700., 5860., 5980., 6095., 6205., 6320., 6750., 7250., 7750., 8250., 8750.]),
                      'LD':np.repeat('nonlinear',16),
                      'LD_u1':np.array([0.6528,0.5641,0.5261,0.5476,0.6044,0.6432,0.6617,0.6746,0.6862,0.6968,0.7072,0.7405,0.7658,0.7743,0.7731,0.7807]),
                      'LD_u2':np.array([-1.0712,-0.7602,-0.4616,-0.3749,-0.3676,-0.3809,-0.3940,-0.4038,-0.4132,-0.4222,-0.4354,-0.4848,-0.5423,-0.5802,-0.5953,-0.6104]),
                      'LD_u3':np.array([1.8320,1.6103,1.3906,1.2243,1.1136,1.0668,1.0507,1.0386,1.0270,1.0160,1.0148,1.0104,1.0053,0.9936,0.9741,0.9546]),
                      'LD_u4':np.array([-0.4340,-0.4830,-0.5265,-0.4975,-0.4689,-0.4571,-0.4533,-0.4504,-0.4476,-0.4450,-0.4436,-0.4385,-0.4325,-0.4249,-0.4153,-0.4057]),
                      'HD189733b':np.array([0.15811 ,0.15751 ,0.15718 ,0.15695 ,0.15666 ,0.15644 ,0.15638 ,0.15631 ,0.15617 ,0.15600 ,0.15610 ,0.15585 ,0.15572 ,0.15586 ,0.15552 ,0.15553])},
            'achrom':{'w':[0.],
                    'LD':['nonlinear'],   #From Hayek+2012, Table 5 for STIS 3D
                    'LD_u1':[0.5598], 
                    'LD_u2':[-0.4055], 
                    'LD_u3':[1.2498], 
                    'LD_u4':[-0.4945], 
                    'HD189733b':[0.15667]}  #Sing+2011
            }

    #Intensity settings for the active regions 
    if gen_dic['star_name']=='TOI3884':
        # data_dic['DI']['ar_prop'] = {}
        data_dic['DI']['ar_prop']={
                'achrom':{
                    'spot1' : [mock_dic['ar_prop']['MIKE_Red']['mockvis']['ang__ISMIKE_Red_VSmockvis_ARspot1'] * np.pi/180],#--base
                    # 'spot2' : [mock_dic['ar_prop']['MIKE_Red']['mockvis']['ang__ISMIKE_Red_VSmockvis_ARspot2'] * np.pi/180],#--base
                    # 'spot3' : [mock_dic['ar_prop']['MIKE_Red']['mockvis']['ang__ISMIKE_Red_VSmockvis_ARspot3'] * np.pi/180],#--base
        #             'facula1' : [mock_dic['ar_prop']['MIKE_Red']['mockvis']['ang__ISMIKE_Red_VSmockvis_ARfacula1'] * np.pi/180],#--base
        #             'facula2' : [mock_dic['ar_prop']['MIKE_Red']['mockvis']['ang__ISMIKE_Red_VSmockvis_ARfacula2'] * np.pi/180],#--base
                    'LD' : ['quadratic'],
                    'LD_u1' : [0.1155],
                    'LD_u2' : [0.3578],
                },
                }

    # if gen_dic['star_name']=='TRAPPIST1':
    #     # data_dic['DI']['ar_prop'] = {}
    #     data_dic['DI']['ar_prop']={
    #             'achrom':{
    #                 'spot1' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot1'] * np.pi/180],#--base
    #                 'spot2' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot2'] * np.pi/180],#--base
    #                 # 'spot3' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot3'] * np.pi/180],#--base
    #                 # 'spot4' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot4'] * np.pi/180],#--base
    #                 # 'spot5' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot5'] * np.pi/180],#--base
    #                 # 'spot6' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot6'] * np.pi/180],#--base
    #                 # 'spot7' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot7'] * np.pi/180],#--base
    #                 # 'spot8' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot8'] * np.pi/180],#--base
    #                 # 'spot9' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot9'] * np.pi/180],#--base
    #                 # 'spot10' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot10'] * np.pi/180],#--base
    #                 # 'spot11' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot11'] * np.pi/180],#--base
    #                 # 'spot12' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot12'] * np.pi/180],#--base
    #                 # 'spot13' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot13'] * np.pi/180],#--base
    #                 # 'spot14' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot14'] * np.pi/180],#--base
    #                 # 'spot15' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot15'] * np.pi/180],#--base
    #                 # 'spot16' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot16'] * np.pi/180],#--base
    #                 # 'spot17' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot17'] * np.pi/180],#--base
    #                 # 'spot18' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot18'] * np.pi/180],#--base
    #                 # 'spot19' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot19'] * np.pi/180],#--base
    #                 # 'spot20' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot20'] * np.pi/180],#--base
    #                 # 'spot21' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot21'] * np.pi/180],#--base
    #                 # 'spot22' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot22'] * np.pi/180],#--base
    #                 # 'spot23' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot23'] * np.pi/180],#--base                    
    #                 # 'spot24' : [theo_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARspot23'] * np.pi/180],#--base 
    #                 # 'facula1' : [mock_dic['ar_prop']['NIRPS_HE']['mockvis']['ang__ISNIRPS_HE_VSmockvis_ARfacula1'] * np.pi/180],#--base                   
    #                 'LD' : ['quadratic'],
    #                 'LD_u1' : [0.168],
    #                 'LD_u2' : [0.245],
    #             },
    #             }

    if gen_dic['star_name']=='AUMic':
        # data_dic['DI']['ar_prop'] = {}
        data_dic['DI']['ar_prop']={
                'achrom':{
                    'spot1' : [mock_dic['ar_prop']['ESPRESSO']['mock_vis']['ang__ISESPRESSO_VSmock_vis_ARspot1'] * np.pi/180],#--base
                    # 'spot1' : [mock_dic['ar_prop']['ESPRESSO']['mock_vis1']['ang__ISESPRESSO_VSmock_vis1_ARspot1'] * np.pi/180],#--grid run
                    # 'spot2' : [mock_dic['ar_prop']['ESPRESSO']['mock_vis']['ang__ISESPRESSO_VSmock_vis_ARspot2'] * np.pi/180],
                    # 'facula1' : [mock_dic['ar_prop']['ESPRESSO']['mock_vis']['ang__ISESPRESSO_VSmock_vis_ARfacula1'] * np.pi/180],#--base
                    'LD' : ['quadratic'],
                    'LD_u1' : [0.35],
                    'LD_u2' : [0.16],
                },
                }

    # if gen_dic['star_name']=='HD189733':
    #     data_dic['DI']['ar_prop']={
    #             'achrom':{
    #                 'spot1' : [theo_dic['ar_prop']['ESPRESSO']['visit1']['ang__ISESPRESSO_VSvisit1_ARspot1'] * np.pi/180],#--base
    #                 'LD':['nonlinear'],   #From Hayek+2012, Table 5 for STIS 3D
    #                 'LD_u1':[0.5598], 
    #                 'LD_u2':[-0.4055], 
    #                 'LD_u3':[1.2498], 
    #                 'LD_u4':[-0.4945],
    #             },
    #             }

    if gen_dic['star_name']=='AU_Mic':
        data_dic['DI']['ar_prop'] = {}
        data_dic['DI']['ar_prop']={
                'achrom':{
                    'spot1' : [theo_dic['ar_prop']['ESPRESSO']['visit1']['ang__ISESPRESSO_VSvisit1_ARspot1'] * np.pi/180],#--base
                    'spot2' : [theo_dic['ar_prop']['ESPRESSO']['visit1']['ang__ISESPRESSO_VSvisit1_ARspot2'] * np.pi/180],#--base
                    'LD' : ['quadratic'],
                    'LD_u1' : [0.63],
                    'LD_u2' : [0.15],
                },
                }

    #Zodiacs
    for zodiac in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        if gen_dic['star_name']==zodiac:
            data_dic['DI']['ar_prop']={
                    'achrom':{
                        'spot1' : [theo_dic['ar_prop']['ESPRESSO']['mock_vis']['ang__ISESPRESSO_VSmock_vis_ARspot1'] * np.pi/180],#--base
                        'LD' : ['quadratic'],
                        'LD_u1' : [0.63],
                        'LD_u2' : [0.15],
                    },
                    }



    if gen_dic['star_name']=='fakeAU_Mic':
        # data_dic['DI']['ar_prop'] = {}
        data_dic['DI']['ar_prop']={
                'achrom':{
                    'spot1' : [20 * np.pi/180],#--base
                    'spot2' : [12 * np.pi/180],#--base
                    'LD' : ['quadratic'],
                    'LD_u1' : [0.63],
                    'LD_u2' : [0.15],
                },
                }

    if gen_dic['star_name']=='V1298tau':
        data_dic['DI']['ar_prop']={
                'achrom':{
                    'spot1' : [mock_dic['ar_prop']['ESPRESSO']['mock_vis']['ang__ISESPRESSO_VSmock_vis_ARspot1'] * np.pi/180],
                    'spot2' : [mock_dic['ar_prop']['ESPRESSO']['mock_vis']['ang__ISESPRESSO_VSmock_vis_ARspot2'] * np.pi/180],
                    'LD' : ['linear'],
                    'LD_u1' : [0.41],

                },
                }

    #Transit light curve model    
    if gen_dic['star_name']=='TOI3884':
        data_dic['DI']['transit_prop'].update({    
                'nsub_Dstar':101,
                'MIKE_Red':{
                'mockvis':{'mode':'simu','n_oversamp':5}, #--base
                }
                })

    if gen_dic['star_name']=='TRAPPIST1':
        data_dic['DI']['transit_prop'].update({    
                'nsub_Dstar':101,
                #'ESPRESSO':{'mock_vis':{'mode':'model', 'dt':0.05}}
                'NIRPS_HE':{
                'mockvis':{'mode':'simu','n_oversamp':5}, #--base
                }
                })

    if gen_dic['star_name']=='AUMic':
        data_dic['DI']['transit_prop'].update({    
                'nsub_Dstar':101,
                #'ESPRESSO':{'mock_vis':{'mode':'model', 'dt':0.05}}
                'ESPRESSO':{
                'mock_vis':{'mode':'simu','n_oversamp':5}, #--base
                'mock_vis1':{'mode':'simu','n_oversamp':5},
                'mock_vis2':{'mode':'simu','n_oversamp':5},
                'mock_vis3':{'mode':'simu','n_oversamp':5},
                'mock_vis4':{'mode':'simu','n_oversamp':5},
                'mock_vis5':{'mode':'simu','n_oversamp':5},
                'mock_vis6':{'mode':'simu','n_oversamp':5},
                'mock_vis7':{'mode':'simu','n_oversamp':5},
                'mock_vis8':{'mode':'simu','n_oversamp':5},
                'mock_vis9':{'mode':'simu','n_oversamp':5},
                }
                })
    
    if gen_dic['star_name']=='fakeAU_Mic':
        data_dic['DI']['transit_prop'].update({    
                'nsub_Dstar':101,
                # 'ESPRESSO':{'mockvisit1':{'mode':'simu','n_oversamp':5}}
                'ESPRESSO':{'mockvisit1':{'mode':'model', 'dt':0.05}}
                }
                )

    if gen_dic['star_name']=='V1298tau':
        data_dic['DI']['transit_prop'].update({
                'ESPRESSO':{'mock_vis':{'mode':'model', 'dt':0.05}}
                })  

    if gen_dic['star_name']=='AU_Mic':
        data_dic['DI']['transit_prop'].update({
                'ESPRESSO':{'mock_vis':{'mode':'model', 'dt':0.05}}
                }) 

    if gen_dic['star_name']=='HD189733':
        if (gen_dic['type']['ESPRESSO']=='CCF'):
            data_dic['DI']['transit_prop'].update({
                    # 'ESPRESSO':{'mock_vis':{'mode':'model', 'dt':0.05}}
                    'ESPRESSO':{'visit1':{'mode':'imp', 'path':'/Users/samsonmercier/Desktop/Work/Master/2023-2024/HD189733_data/Broadband_scaling_data/lc_wavelength-5500.0 Angstrom.dat'}}
                    }) 
        else:
            data_dic['DI']['transit_prop'].update({'ESPRESSO':{# '20210830':{'mode':'model','dt':0.05},    #used temporarily while finding Hritam's bug (spotted exposures are excluded from RM analysis anyway)
                                                                '20210830':{'mode':'imp','path':'/Users/samsonmercier/Desktop/Work/Master/2023-2024/HD189733_data/Broadband_scaling_data/lc_wavelength_chromatic.dat'}   #will use Hritam spotted chromatic fit to simultaneous EulerCamn photom
                                                    }})

    #Zodiacs 
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        data_dic['DI']['transit_prop'].update({
            'nsub_Dstar':101,
            'ESPRESSO':{'mock_vis':{'mode':'simu', 'n_oversamp':5}}
            })       


        # data_dic['DI']['transit_prop'].update({
        #         'nsub_Dstar':101,
        #         'ESPRESSO':{'visit1':{'mode':'simu', 'n_oversamp':5}}
        #         })
    #Forcing in/out transit flag


    #Plot settings


    #Model time resolution   

    #Input light curves 
    plot_dic['input_LC']='pdf'  #pdf

    #Scaling light curves
    plot_dic['spectral_LC']=''   #pdf

    #2D maps of disk-integrated profiles
    plot_dic['map_DI_prof']=''   #png   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        

    
    ##################################################################################################
    #%%% Module: 2D->1D conversion for disk-integrated spectra 
    ##################################################################################################
    
    ANTARESS_2D_1D_settings('DI',data_dic,gen_dic,plot_dic)

    if gen_dic['star_name']=='HD189733':
        gen_dic['spec_1D_DI']=True    #& False      
        gen_dic['calc_spec_1D_DI']= True    #& False    

    ##################################################################################################
    #%%% Module: disk-integrated profiles binning
    #    - for analysis purpose (original profiles are not replaced)
    #    - profiles should be aligned in the star rest frame before binning, via gen_dic['align_DI']
    #    - profiles should also be comparable when binned, which means that broadband scaling needs to be applied
    ##################################################################################################
    
    
    #%%%% Activating 
    gen_dic['DIbin'] = False
    gen_dic['DIbinmultivis'] = False
    
    
    #%%%% Calculating/retrieving
    gen_dic['calc_DIbin']=True
    gen_dic['calc_DIbinmultivis'] = True  
    
    
    #%%%% Visits to be binned    
    #    - visits to be included in the multi-visit binning, for each instrument
    #    - leave empty to use all visits
    data_dic['DI']['vis_in_bin']={}   
    
    #%%%% Exposures to be binned
    #    - indexes of exposures that contribute to the bin series, for each instrument/visit
    #    - indexes are relative to the global table in each visit
    #    - leave empty to use all out-exposures 
    data_dic['DI']['idx_in_bin']={}
    
    
    #%%%% Binning dimension
    #    - possibilities
    # + 'phase': profiles are binned over phase    
    #            this is not possible when binning multiple visits, if different planets are transiting (use the 'masterDI' option). In that case the 
    #            phase constraints are ignored and all out-of-transit exposures (default) or the selected ones are used
    #    - beware to use the alignement module to calculate binned profiles in the star rest frame
    data_dic['DI']['dim_bin']='phase' 
    
    
    #%%%% Bin definition
    #    - bins are defined for each instrument/visit
    #      format is inst : vis : dic_bin
    #    - bins can be defined
    # + manually: indicate lower/upper bin boundaries (ordered)
    #             format is dic_bin = {'bin_low' : [x0,x1,..],'bin_high' : [y0,y1,..]}
    # + automatically : indicate total range and number of bins 
    #                   format is dic_bin = {'bin_range':[x0,y0],'nbins': n}
    #    - for each visit, indicate a reference planet if more than one planet is transiting, and if the bin dimensions is specific to a given planet (not compatible with multi-visit binning)
    #      format is 'ref_pl' : { inst : {vis : pl_name}}
    data_dic['DI']['prop_bin']={}
    
                
    #%%%% Plot settings
    
    #%%%%% 2D maps of binned profiles
    plot_dic['map_DIbin']=''      
    
    
    #%%%%% Individual binned profiles
    plot_dic['DIbin']=''    
    
    
    #%%%%% Residuals from binned profiles
    plot_dic['DIbin_res']=''  
    
    #%%%% Activating 
    gen_dic['DIbin'] = True  &  False
    gen_dic['DIbinmultivis'] = True      &  False
    if gen_dic['star_name']=='HD189733':
        gen_dic['DIbin']=True #& False
        gen_dic['DIbinmultivis']=True   & False  
    
    
    #%%%% Calculating/retrieving
    gen_dic['calc_DIbin']=True   &  False  
    gen_dic['calc_DIbinmultivis'] = True      &  False
    if gen_dic['star_name']=='HD189733':
        gen_dic['calc_DIbin']=True   #&  False  
        gen_dic['calc_DIbinmultivis']= True  & False       

    #%%%% Plot settings

    #%%%%% Individual binned profiles
    plot_dic['DIbin']=''  
    if gen_dic['star_name']=='HD189733':plot_dic['DIbin']=''

    ##################################################################################################
    #%%% Module: disk-integrated CCF masks
    #    - spectra must have been aligned in the star rest frame (using a approximate 'sysvel'), converted into a 1D profile, and binned. The mask can then be used in the input rest frame (setting 'sysvel' to 0 km/s)
    #    - the mask is determined by default from a master spectrum built over all processed visits of an instrument, for consistency of the CCFs between visits
    #    - the mask is saved as a .txt file in air or vacuum (depending on the pipeline process) and as a .fits file in air to be read by ESPRESSO-like DRS
    ##################################################################################################
    
    
    #%%%% Activating
    gen_dic['def_DImasks'] = False
    
    
    #%%%% Multi-threading
    data_dic['DI']['mask']['nthreads'] = int(0.8*cpu_count())       
    
    
    #%%%% Print status
    data_dic['DI']['mask']['verbose'] = False
    
    
    #%%%% Estimate of line width 
    #    - in km/s
    #    - use the analysis of disk-integrated CCF properties to estimate this value
    data_dic['DI']['mask']['fwhm_ccf'] = 5. 
    
    
    #%%%%% Vicinity window
    #    - in fraction of 'fwhm_ccf'
    #    - window for extrema localization in pixels of the regular grid = int(min(fwhm_ccf*w_reg/(vicinity_fwhm*c_light*dw_reg)))
    #      beware that it should not be too small, otherwise small variations within the lines themselves will be mistaken for extrema
    data_dic['DI']['mask']['vicinity_fwhm'] = {} 
    
    
    #%%%%% Requested line weights
    #    - possibilities:
    # + 'weight_rv_sym' (default): based on line profile gradient over symetrical window
    # + 'weight_rv': based on line profile gradient over assymetrical window
    data_dic['DI']['mask']['mask_weights'] = 'weight_rv_sym'  
    
    
    #%%%% Master spectrum
    
    #%%%%% Oversampling
    #    - binwidth of regular grid over which the master spectrum is resampled, in A
    #    - choose a fine enough sampling, as mask lines are set to the centers of resampled bins 
    #    - format: {inst:value}    
    data_dic['DI']['mask']['dw_reg'] = {}
    
    
    #%%%%% Smoothing window   
    #    - in pixels, used for spectrum and derivative 
    #    - format: {inst:value}
    data_dic['DI']['mask']['kernel_smooth']={}
    data_dic['DI']['mask']['kernel_smooth_deriv2']={}
    
    
    #%%%% Line selection: rejection ranges
    #    - lines within these ranges are excluded (check using plot_dic['DImask_spectra'] with step='cont')
    #    - defined in the star rest frame
    #    - format: {inst:[[w1,w2],[w3,w4],..]}
    #    - typical strong telluric contamination in the optical occurs over [6865,6930] and [7590,7705] in the Earth rest frame
    data_dic['DI']['mask']['line_rej_range'] = {}
    
     
    #%%%% Line selection: depth and width   
    #    - check using plot_dic['DImask_spectra'] with step='sel1'
    
    #%%%%% Depth range
    #    - minimum/maximum line depths to be considered in the stellar mask (counted from the continuum with 'linedepth_cont_X' and from the local maxima with 'linedepth_X')
    #      between 0 and 1
    #      format is inst : val
    #    - use 'linedepth_contdepth' to define a linear threshold (line depth from maxima) = a*(line depth from continuum) + b
    #      format is inst : [a,b]
    #    - use plot_dic['DImask_ld'] to adjust
    data_dic['DI']['mask']['linedepth_cont_min'] = {}   
    data_dic['DI']['mask']['linedepth_min'] = {}  
    data_dic['DI']['mask']['linedepth_cont_max'] = {} 
    data_dic['DI']['mask']['linedepth_max'] = {}      
    data_dic['DI']['mask']['linedepth_contdepth'] = {} 

    
    #%%%%% Minimum depth and width
    #    - selection criteria on minimum line depth and half-width (between minima and closest maxima) to be kept (value > 10^(crit)) 
    #    - use plot_dic['DImask_ld_lw'] to adjust, excluding lines that contribute the least to the cumulated weight of the linelist
    data_dic['DI']['mask']['line_width_logmin'] = None
    data_dic['DI']['mask']['line_depth_logmin'] = None
    
    
    #%%%% Line selection: position
    #    - define RV window for line position fit (km/s)
    #    - set maximum RV deviation of fitted minimum from line minimum in resampled grid (m/s)
    #    - check using plot_dic['DImask_spectra'] with step='sel2'
    #      use plot_dic['DImask_RVdev_fit'] to adjust
    data_dic['DI']['mask']['win_core_fit'] = 1
    data_dic['DI']['mask']['abs_RVdev_fit_max'] = None
    
    
    #%%%% Line selection: tellurics
    
    #%%%%% Threshold on telluric line depth 
    #    - minimum depth above which to consider tellurics for exclusion
    data_dic['DI']['mask']['tell_depth_min'] = 0.001
    
    
    #%%%%% Thresholds on telluric/stellar lines depth ratio
    #    - telluric lines with ratio larger than minimum threshold are considered for exclusion
    #    - stellar lines with ratio larger than maximum threshold are excluded (the final threshold is applied after the VALD and morphological analysis)
    #    - check using plot_dic['DImask_spectra'] with step='sel3'
    #      use plot_dic['DImask_tellcont'] to adjust
    data_dic['DI']['mask']['tell_star_depthR_min'] = None
    data_dic['DI']['mask']['tell_star_depthR_max'] = None
    data_dic['DI']['mask']['tell_star_depthR_max_final'] = None
    
    
    #%%%% VALD cross-validation 
    #    - lines from the input VALD linelist are cross-matched with the lines identified by the module and stored for later use.
    #    - this step has no impact on the line selection    
    
    #%%%%% Path to VALD linelist
    #    - set to None to prevent
    #    - see details in theo_dic['st_atm']
    data_dic['DI']['mask']['VALD_linelist'] = None
    
    
    #%%%%% Adjusting VALD line depth
    data_dic['DI']['mask']['VALD_depth_corr'] = True
    
    
    #%%%% Line selection: morphological 
    
    #%%%%% Symmetry
    #    - selection criteria on maximum ratio between normalized continuum difference and relative line depth, and normalized asymetry parameter, to be kept (value < crit) 
    #    - check using plot_dic['DImask_spectra'] with step='sel4'
    #      use plot_dic['DImask_morphasym'] to adjust    
    data_dic['DI']['mask']['diff_cont_rel_max'] = None
    data_dic['DI']['mask']['asym_ddflux_max'] = None    
    
    
    #%%%%% Width and depth
    #    - selection criteria on minimum line depth (value > crit) and maximum line width (value < crit) to be kept
    #    - check using plot_dic['DImask_spectra'] with step='sel5'
    #      use plot_dic['DImask_morphshape'] to adjust   
    data_dic['DI']['mask']['width_max'] = None 
    data_dic['DI']['mask']['diff_depth_min'] = None
    
        
    #%%%% Line selection: RV dispersion 
    #    - set to True to activate 
    #    - check using plot_dic['DImask_spectra'] with step='sel6'
    data_dic['DI']['mask']['RV_disp_sel'] = True
    
    
    #%%%%% Exposures selection
    #    - indexes of exposures from which RV are derived
    #    - indexes are relative to the global table in each visit, but only exposures used to build the master spectrum will be considered (used by default if left empty)
    data_dic['DI']['mask']['idx_RV_disp_sel']={}
    
    
    #%%%%% RV deviation
    #    - lines with absolute RVs beyond this value are excluded (in m/s)
    #    - use plot_dic['DImask_RVdisp'] to adjust 
    data_dic['DI']['mask']['absRV_max'] = {}
    
    
    #%%%%% Dispersion deviation
    #    - lines with RV dispersion and RV dispersion/mean error over the exposure series beyond these values are excluded
    #    - beware not to use this criterion when there are too few exposures
    #    - use plot_dic['DImask_RVdisp'] to adjust 
    data_dic['DI']['mask']['RVdisp2err_max'] = {}
    data_dic['DI']['mask']['RVdisp_max'] = {}
    
    
    #%%%% Plot settings 
    
    #%%%%% Mask at successive steps
    #    - use the 'cont' step to adjust stellar continuum determination and excluded spectral ranges
    plot_dic['DImask_spectra'] = ''
    
    
    #%%%%% Depth range selection
    plot_dic['DImask_ld'] = ''
    
    
    #%%%%% Minimum depth and width selection
    plot_dic['DImask_ld_lw'] = ''
    
    
    #%%%%% Position selection
    plot_dic['DImask_RVdev_fit'] = ''
    
    
    #%%%%% Telluric selection
    plot_dic['DImask_tellcont'] = ''
    
    
    #%%%%% VALD selection
    plot_dic['DImask_vald_depthcorr'] = ''
    
    
    #%%%%% Morphological (asymmetry) selection
    plot_dic['DImask_morphasym'] = ''
    
    
    #%%%%% Morphological (shape) selection
    plot_dic['DImask_morphshape'] = ''
    
    
    #%%%%% RV dispersion selection
    plot_dic['DImask_RVdisp'] = ''
    
 
    #%%%% Acxtivating 
    gen_dic['def_DImasks'] = True & False

    #%%%% Estimate of line width 

    if gen_dic['star_name']=='HD189733':
        data_dic['DI']['mask']['fwhm_ccf'] = 7.7

    #%%%%% Vicinity window
    if gen_dic['star_name']=='HD189733':
        data_dic['DI']['mask']['vicinity_fwhm'] = {'ESPRESSO' : 10.}
        mask_level = 'strict'
        mask_level = 'relax'
        mask_level = 'rv_tell'
        mask_level = 'loose'
    
    #%%%% Line selection: rejection ranges
    if gen_dic['star_name']=='HD189733':
        if mask_level=='strict':data_dic['DI']['mask']['line_rej_range'] = {'ESPRESSO':[[3860.,3870.],[6863.,6967.],[7165.,7310.],[7705.,7707.]]}        
        if mask_level in ['relax','rv_tell','loose']:data_dic['DI']['mask']['line_rej_range'] = {'ESPRESSO':[[3860.,3870.],[6863.,6967.],[7705.,7707.]]}


    #%%%% Line selection: depth and width     

    #%%%%% Depth range   
    if gen_dic['star_name']=='HD189733':
        if mask_level=='strict':
            data_dic['DI']['mask']['linedepth_cont_min'] = {'ESPRESSO':0.1}   
            data_dic['DI']['mask']['linedepth_cont_max'] = {'ESPRESSO':0.95}   
            data_dic['DI']['mask']['linedepth_min'] = {'ESPRESSO':0.05}  
            data_dic['DI']['mask']['linedepth_contdepth'] = {'ESPRESSO':[1.0,0.02]}    

    #%%%%% Minimum depth and width
    if gen_dic['star_name']=='HD189733':
        if mask_level=='strict':
            data_dic['DI']['mask']['line_width_logmin'] = -1.3   #env 10% du poids cum 
            data_dic['DI']['mask']['line_depth_logmin'] = -1.2   #env 10% du poids cum 

    
    #%%%% Line selection: position

    if gen_dic['star_name']=='HD189733':
        if mask_level=='strict':data_dic['DI']['mask']['abs_RVdev_fit_max'] = 180.     #~2% of cum weights
        if mask_level=='relax':data_dic['DI']['mask']['abs_RVdev_fit_max'] = 225.  
        if mask_level in ['rv_tell','loose']:data_dic['DI']['mask']['abs_RVdev_fit_max'] = 300.  

    #%%%% Line selection: tellurics

    #%%%%% Threshold on telluric line depth 
    data_dic['DI']['mask']['tell_depth_min'] = 0.001

    #%%%%% Thresholds on telluric/stellar lines depth ratio
    if gen_dic['star_name']=='HD189733':
        if mask_level=='strict':
            data_dic['DI']['mask']['tell_star_depthR_max'] = 0.2            
            data_dic['DI']['mask']['tell_star_depthR_max_final'] = 0.015            

    #%%%% VALD cross-validation     

    #%%%%% Path to VALD linelist
    # if gen_dic['star_name']=='HD189733':data_dic['DI']['mask']['VALD_linelist'] = '/Users/bourrier/Travaux/Exoplanet_systems/HD/HD189733b/Star/VALD/VALD_HD189733'


    #%%%% Line selection: morphological 

    #%%%%% Symmetry  
    if gen_dic['star_name']=='HD189733':
        if mask_level=='strict':
            data_dic['DI']['mask']['diff_cont_rel_max'] = 1.6   #selected to keep 80% of total weights
            data_dic['DI']['mask']['asym_ddflux_max'] = 0.4     #selected to keep 80% of total weights  
        if mask_level=='relax':
            data_dic['DI']['mask']['diff_cont_rel_max'] = 5.   #selected to keep 90% of total weights
            data_dic['DI']['mask']['asym_ddflux_max'] = 0.7           
        if mask_level=='rv_tell':
            data_dic['DI']['mask']['diff_cont_rel_max'] = 20.   
            data_dic['DI']['mask']['asym_ddflux_max'] = 0.85
        if mask_level=='loose':
            data_dic['DI']['mask']['diff_cont_rel_max'] = 30.   
            data_dic['DI']['mask']['asym_ddflux_max'] = 0.9

    #%%%%% Width and depth
    if gen_dic['star_name']=='HD189733':
        if mask_level=='strict':
            data_dic['DI']['mask']['width_max'] = 10.     #selected to keep 90% of total weights
            data_dic['DI']['mask']['diff_depth_min'] = 0.2   #selected to keep 90% of total weights
        if mask_level=='relax':
            data_dic['DI']['mask']['width_max'] = 13.0 
            data_dic['DI']['mask']['diff_depth_min'] = 0.05
        if mask_level in ['rv_tell','loose']:
            data_dic['DI']['mask']['width_max'] = 15.0 
            data_dic['DI']['mask']['diff_depth_min'] = 0.02

    #%%%% Line selection: RV dispersion 

    data_dic['DI']['mask']['RV_disp_sel'] = True   #& False

    #%%%%% Exposures selection
    data_dic['DI']['mask']['idx_RV_disp_sel']={}

    #%%%%% RV deviation
    if gen_dic['star_name']=='HD189733':
        if mask_level=='strict':data_dic['DI']['mask']['absRV_max'] = {'ESPRESSO':2.9}  #5% of cum weights
        if mask_level=='relax':data_dic['DI']['mask']['absRV_max'] = {'ESPRESSO':10.}
        if mask_level in ['rv_tell','loose']:data_dic['DI']['mask']['absRV_max'] = {'ESPRESSO':50.}

    #%%%%% Dispersion deviation
    if gen_dic['star_name']=='HD189733':
        if mask_level=='strict':
            data_dic['DI']['mask']['RVdisp2err_max'] = {'ESPRESSO':1.4}  #5% of cum weights
            data_dic['DI']['mask']['RVdisp_max'] = {'ESPRESSO':50.}    #5% of cum weights
        if mask_level=='relax':
            data_dic['DI']['mask']['RVdisp2err_max'] = {'ESPRESSO':1.7}
            data_dic['DI']['mask']['RVdisp_max'] = {'ESPRESSO':150.}    
        if mask_level=='rv_tell':
            data_dic['DI']['mask']['RVdisp2err_max'] = {'ESPRESSO':2.8}
            data_dic['DI']['mask']['RVdisp_max'] = {'ESPRESSO':100.}               
        if mask_level=='loose':
            data_dic['DI']['mask']['RVdisp2err_max'] = {'ESPRESSO':2.8}
            data_dic['DI']['mask']['RVdisp_max'] = {'ESPRESSO':150.}             

    ##################################################################################################
    #%% Differential and intrinsic profiles
    ##################################################################################################  
    
    
    
    ##################################################################################################
    #%%% Module: differential profiles extraction
    #    - potentially affected by the planetary atmosphere
    #    - the master for the unocculted star is computed over phase without using specific windows, using all selected exposures in full
    #      out-of-transit exposures are used by default, potentially over several visits
    ##################################################################################################   
    
    #%%%% Activating
    gen_dic['diff_data'] = False
    
    
    #%%%% Calculating/retrieving 
    gen_dic['calc_diff_data'] = True 
    
    
    #%%%% Multi-threading
    gen_dic['nthreads_diff_data']= int(0.8*cpu_count())      
    
    
    #%%%% In-transit restriction
    #    - limit the extraction of differential profiles to in-transit exposures
    #    - this is only relevant when a specific master is calculated for each exposure (ie, when requesting the extraction of local profiles from 
    # spectral data that are not defined on common bins), otherwise it does not cost time to extract local profiles from all exposures using a common master
    #    - this will prevent plotting and analyzing differential profiles outside of the transit
    data_dic['Diff']['extract_in'] = False
    
    
    #%%%% Master visits
    #    - use this field to define which visits should be used to calculate the master stellar spectrum for a given instrument (ie, the master that will be used for all visits of this instrument)
    # format is : { inst : [vis0, vis1, ...] }  
    #      leave empty for the master of a given visit to be calculated with the exposures of this visit only
    #    - this option can be used to boost the SNR of the master and/or smooth out variations in its shape
    #    - if activated, this option requires that aligned, scaled disk-integrated profiles have been first calculated for all chosen visits
    #    - which exposures contribute to the master in each visit is set through data_dic['Diff']['idx_in_bin']
    #    - if multiple planets are transiting in binned visits, the reference planet for orbital phase (ie, the dimension along which exposures are binned) can be forced through data_dic['DI']['pl_in_bin']={inst:{vis:XX}} 
    #      this has however no impact since the weights associated with exposure duration do not depend on the planet phase, and out-of-transit exposures are defined accouting for all transiting planets
    data_dic['Diff']['vis_in_bin']={}  
    
    
    #%%%% Master exposures
    #    - indexes of exposures that contribute to the master, in each visit required for master calculation
    # format is : {inst : { vis : [idx_i, idx_j, ...]}}  
    #      indexes are relative to global grids
    #      visits are the ones requested in 'vis_in_bin', or the processed ones if left empty
    #    - set to out-of-transit exposures if left undefined
    data_dic['Diff']['idx_in_bin']={}
    
    
    #%%%% Continuum range
    #    - format: { inst : { ord : [ [x1,x2] , [x3,x4] , ... ] }}
    #      with x defined in the star rest frame
    #    - only used to set errors on differential profiles from dispersion in their continuum (see data_dic['Intr']['disp_err']), and as default for joined differential profile fits if undefined
    #    - the ranges are common to all differential profiles, ie that they must be large enough to cover the full range of RVs (with the width of the stellar line) from the regions along the transit chord    
    #      the range does not need to be as large as defined for the raw CCFs, which can be broadened by rotation 
    data_dic['Diff']['cont_range']={}
    

    #%%%% Plot settings
    
    #%%%%% 2D maps of differential profiles
    #    - in stellar rest frame
    #    - can be used to check for spurious variations in all exposures: to do so, apply the differential profile extraction after having aligned (and potentially corrected)
    # all profiles in the star rest frame, but without transit scaling (if flux balance correction has been applied) or after having applied a light curve unity
    plot_dic['map_Diff_prof']=''   
    
    
    #%%%%% Individual differential profiles
    plot_dic['Diff_prof']=''       
    
    
    
    
    #Activating
    gen_dic['diff_data'] = True  &  False

    #Calculating/retrieving 
    gen_dic['calc_diff_data'] = True   &  False


    #Multi-threading
    gen_dic['nthreads_diff_data']= 2

    #In-transit restriction
    data_dic['Diff']['extract_in']=True  &  False
    
   # %%%% Master visits
    if gen_dic['star_name']=='AUMic':
        data_dic['Diff']['vis_in_bin']={'ESPRESSO':['mock_vis']}  

    if gen_dic['star_name']=='fakeAU_Mic':
        data_dic['Diff']['vis_in_bin']={'ESPRESSO':['mockvisit1']}  

    if gen_dic['star_name']=='AU_Mic':
        data_dic['Diff']['vis_in_bin']={'ESPRESSO':['visit1']}  

    if gen_dic['star_name']=='HD189733':
        data_dic['Diff']['vis_in_bin']={'ESPRESSO':['visit1']} 

    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        data_dic['Diff']['vis_in_bin']={'ESPRESSO':['mock_vis']}  


   #  %%%% Master exposures
    if gen_dic['star_name']=='TOI3884':    
        # data_dic['Diff']['idx_in_bin']={'MIKE_Red':{'mockvis':list(np.arange(0, 4,dtype=int))+list(np.arange(26, 30,dtype=int))}}
        data_dic['Diff']['idx_in_bin']={'MIKE_Red':{'mockvis':list(np.arange(0, 7,dtype=int))+list(np.arange(23, 30,dtype=int))}} #Almenara+2022 updated

    if gen_dic['star_name']=='TRAPPIST1':    
        # data_dic['Diff']['idx_in_bin']={'ESPRESSO':{'mock_vis':list(np.arange(0, 45,dtype=int))+list(np.arange(135, 180,dtype=int))}}
        # data_dic['Diff']['idx_in_bin']={'NIRPS_HE':{'mockvis':list(np.arange(0, 19,dtype=int))+list(np.arange(28, 70,dtype=int))}} #b-transit
        # data_dic['Diff']['idx_in_bin']={'NIRPS_HE':{'mockvis':list(np.arange(0, 47,dtype=int))+list(np.arange(54, 100,dtype=int))}} #c-transit
        data_dic['Diff']['idx_in_bin']={'NIRPS_HE':{'mockvis':list(np.arange(0, 7,dtype=int))+list(np.arange(23, 30,dtype=int))}} #c-transit high SNR
        # data_dic['Diff']['idx_in_bin']={'NIRPS_HE':{'mockvis':list(np.arange(0, 29,dtype=int))+list(np.arange(40, 70,dtype=int))}} #d-transit
        # data_dic['Diff']['idx_in_bin']={'NIRPS_HE':{'mockvis':list(np.arange(0, 29,dtype=int))+list(np.arange(41, 70,dtype=int))}} #e-transit
        # data_dic['Diff']['idx_in_bin']={'NIRPS_HE':{'mockvis':list(np.arange(0, 97,dtype=int))+list(np.arange(99, 100,dtype=int))}} #f-transit

    if gen_dic['star_name']=='AUMic':    
        # data_dic['Diff']['idx_in_bin']={'ESPRESSO':{'mock_vis':list(np.arange(0, 45,dtype=int))+list(np.arange(135, 180,dtype=int))}}
        data_dic['Diff']['idx_in_bin']={'ESPRESSO':{'mock_vis':list(np.arange(0, 7,dtype=int))+list(np.arange(23, 30,dtype=int))}} #- base
        # data_dic['Diff']['idx_in_bin']={'ESPRESSO':{'mock_vis':list(np.arange(0, 45,dtype=int))+list(np.arange(135, 180,dtype=int))}} #- ESPRESSO exposure time
    
    if gen_dic['star_name']=='AU_Mic':    
        data_dic['Diff']['idx_in_bin']={'ESPRESSO':{'visit1':list(np.arange(0, 15,dtype=int))+list(np.arange(67, 82,dtype=int))}}

    if gen_dic['star_name']=='HD189733':    
        data_dic['Diff']['idx_in_bin']={'ESPRESSO':{'visit1':list(np.arange(0, 7,dtype=int))+list(np.arange(26, 43,dtype=int))}}

    if gen_dic['star_name']=='fakeAU_Mic':    
        data_dic['Diff']['idx_in_bin']={'ESPRESSO':{'mockvisit1':list(np.arange(0, 16,dtype=int))+list(np.arange(68, 84,dtype=int))}} 

    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        if gen_dic['star_name']=='Gemini':data_dic['Diff']['idx_in_bin']={'ESPRESSO':{'mock_vis':list(np.arange(0, 41,dtype=int))+list(np.arange(139, 180,dtype=int))}} 
        elif gen_dic['star_name']=='Sagittarius':data_dic['Diff']['idx_in_bin']={'ESPRESSO':{'mock_vis':list(np.arange(0, 47,dtype=int))+list(np.arange(133, 180,dtype=int))}} 
        else:data_dic['Diff']['idx_in_bin']={'ESPRESSO':{'mock_vis':list(np.arange(0, 45,dtype=int))+list(np.arange(135, 180,dtype=int))}} 


    #Continuum range
    if gen_dic['star_name']=='AUMic':data_dic['Diff']['cont_range']['ESPRESSO']={0 : [[-150.,-70.],[70.,150.]]}

    if gen_dic['star_name']=='fakeAU_Mic':data_dic['Diff']['cont_range']['ESPRESSO']={0 : [[-30.,-20.],[20.,30.]]}

    if gen_dic['star_name']=='V1298tau':data_dic['Diff']['cont_range']['ESPRESSO']={0 : [[-150.,-70.],[70.,150.]]  }

    if gen_dic['star_name']=='AU_Mic':data_dic['Diff']['cont_range']['ESPRESSO']={0 : [[-20.,-15.],[15.,20.]]}

    if gen_dic['star_name']=='HD189733':data_dic['Diff']['cont_range']['ESPRESSO']={0:[[-90.,-30.],[30.,90.]]}

    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:data_dic['Diff']['cont_range']['ESPRESSO']={0 : [[-150.,-70.],[70.,150.]]}


    #Error definition
    data_dic['Diff']['disp_err']=False


    #2D maps of residual profiles
    plot_dic['map_Diff_prof']='pdf'   #png 

    #Individual residual profiles
    plot_dic['Diff_prof']=''   #pdf    
    
    
    
    
    
    
    
    
    
    
    

    
    ##################################################################################################
    #%%% Module: intrinsic profiles extraction
    #    - derived from the local profiles (in-transit), reset to the same broadband flux level, with planetary contamination excluded 
    #    - if the scaling light curve correctly accounts for active regions, then their contribution is propagated in the broadband flux scaling and intrinsic profiles 
    # from active regions occulted by a planet will still be scaled to the common continuum of the series
    ##################################################################################################    
    
    #%%%% Calculating
    gen_dic['intr_data'] = False
    
    
    #%%%% Calculating/retrieving
    gen_dic['calc_intr_data'] = True  
     
    
    #%%%%% Continuum range
    #    - format: { inst : { ord : [ [x1,x2] , [x3,x4] , ... ] }}
    #      with x defined in the star rest frame
    #    - used to set errors on intrinsic profiles from dispersion in their continuum (see data_dic['Intr']['disp_err']), to define the intrinsic continuum level, to perform continuum correction of intrinsic profiles, and as default for joined intrinsic profile fits if undefined
    data_dic['Intr']['cont_range'] = deepcopy(data_dic['Diff']['cont_range'])
    
    
    #%%%% Calculating/retrieving continuum in each order
    #    - the continuum is calculated by default, this option controls whether it is calculated again or retrieved from a previous calculation
    #    - required for continuum correction, and used for plots
    data_dic['Intr']['calc_cont'] = True
    
    
    #%%%% Continuum correction
    #    - the continuum might show differences between exposures because of imprecisions on the flux balance correction
    #      this option corrects for these deviations and update the flux scaling values accordingly
    #    - the correction is applied order per order using pixels defined in all exposures, and may thus not be accurate in orders with low S/R and few defined pixels, especially for exposures with low flux at the stellar limbs
    #    - the correction is applied to the final processed intrinsic spectra, ie that if you want to generate intrinsic CCF or 1D spectra be sure to activate the corresponding modules when calling the correction for the first time, otherwise it will be
    # applied to the raw intrinsic spectra with less accuracy
    #    - beware that intrinsic profiles are overwritten by this correction: re-run the extraction without continuum correction in case you disable it 
    data_dic['Intr']['cont_norm'] = False
    
    
    #%%%% Plot settings
    
    #%%%%% 2D map: intrinsic stellar profiles
    #    - aligned or not
    plot_dic['map_Intr_prof']=''   
    
    
    #%%%%% Individual intrinsic stellar profiles
    #    - aligned or not
    plot_dic['Intr_prof']=''    
    
    
    #%%%%% Residuals from intrinsic stellar profiles
    #    - choose within the routine whether to plot fit to individual or to global profiles
    plot_dic['Intr_prof_res']=''  
    
    
    
    #Calculating
    gen_dic['intr_data'] = True    &  False
    if (gen_dic['star_name']=='HD189733'):
        if (gen_dic['type']['ESPRESSO']=='spec2D'):gen_dic['intr_data'] = True  & False
        else:gen_dic['intr_data'] = True  #& False 

    #Calculating/retrieving
    gen_dic['calc_intr_data'] = True   &  False      

    #Continuum range
    if gen_dic['star_name']=='AUMic':
        data_dic['Intr']['cont_range'] = deepcopy(data_dic['Diff']['cont_range'])

    if gen_dic['star_name']=='V1298tau':
        data_dic['Intr']['cont_range'] = deepcopy(data_dic['Diff']['cont_range'])

    if gen_dic['star_name']=='AU_Mic':
        data_dic['Intr']['cont_range'] = deepcopy(data_dic['Diff']['cont_range'])

    #Calculating/retrieving continuum 
    data_dic['Intr']['calc_cont'] = True# & False


    #Continuum correction
    data_dic['Intr']['cont_norm'] = True   &   False
    if gen_dic['star_name']=='HD189733':
        if (gen_dic['type']['ESPRESSO']=='spec2D'):data_dic['Intr']['cont_norm'] = True   & False  #activate if CCF intr are generated
        else: data_dic['Intr']['cont_norm'] = True   #&   False



    #2D maps of intrinsic stellar profiles
    plot_dic['map_Intr_prof']=''   #'png 

    if gen_dic['star_name']=='AUMic':
        plot_dic['map_Intr_prof']='png'

    if gen_dic['star_name']=='V1298tau':
        plot_dic['map_Intr_prof']='png'

    if gen_dic['star_name']=='AU_Mic':
        plot_dic['map_Intr_prof']='png'

    #Individual intrinsic stellar profiles
    plot_dic['sp_intr']=''  
    plot_dic['Intr_prof']=''   #pdf  

    #Residuals from intrinsic stellar profiles
    plot_dic['Intr_prof_res']=''  #pdf
    
    
    
    

    
    ##################################################################################################
    #%%% Module: CCF conversion for differential & intrinsic spectra 
    #    - calculating CCFs from OT differential and intrinsic stellar spectra
    #    - for analysis purpose, ie do not apply if atmospheric extraction is later requested
    #    - every analysis afterwards will be performed on those CCFs
    #    - ANTARESS will stop if intrinsic profiles are simultaneously required to extract atmospheric spectra 
    ##################################################################################################   

    ANTARESS_CCF_settings('Intr',gen_dic)
    
    #%%%% Error definition
    #    - if not None, forces errors on out-of-transit differential and intrinsic CCFs to their continuum dispersion times sqrt(disp_err)
    #    - if input data have no errors, disk-integrated error tables have already been set to sqrt(g_err*F) and propagated
    #      if activated, the present option will override these tables (whether the input data had error table or not originally) 
    data_dic['Intr']['disp_err']=None    
    
    
    
    
    
    
    
    
    
    
    ##################################################################################################
    #%%% Module: out-of-transit differential profiles PCA
    #    - can be applied to data in CCF format or to spectral data in a given order
    #    - use this module to derive PC and match their combination to differential and intrinsic profiles in the fit module
    #      correction is then applied through the CCF correction module
    #    - pca is applied to the pre-transit, post-transit, and out-of-transit data independently
    #      the noise model is fitted using the out-PC
    ##################################################################################################
    
    
    #%%%% Activating
    gen_dic['pca_ana'] = False
    
    
    #%%%% Calculating/retrieving
    gen_dic['calc_pca_ana'] = True     
    
    
    #%%%% Visits to be processed
    #    - instruments are processed if set
    #    - set visit field to 'all' to process all visits
    data_dic['PCA']['vis_list']={}  
    
    
    #%%%% Exposures contributing to the PCA
    #    - global indexes
    #    - all out-of-transit exposures are used if undefined
    data_dic['PCA']['idx_pca']={}


    #%%%% Order to be processed
    #    - for data in spectral mode only
    #    - format: {inst:{vis: iord }   
    data_dic['PCA']['ord_proc'] = 0
    
    
    #%%%% Spectral range to determine PCs
    data_dic['PCA']['ana_range']={}
      
    
    #%%%% Noise model
    
    #%%%%% Number of PC
    #    - use the fraction of variance explained and the BIC of the fits (to individual exposures, and in the RMR) to determine the number of PC to use in the noise model
    data_dic['PCA']['n_pc']={}
    
        
    #%%%%% Fitted exposures
    #    - global indexes
    #    - all exposures are fitted if left undefined
    data_dic['PCA']['idx_corr']={}
    
    
    #%%%%% Fitted spectral range
    #    - here the band is common as we want to fit the PC to the noise within the doppler track, in the in-transit profiles
    #    - leave undefined for fit_range to be set to 'ana_range'
    data_dic['PCA']['fit_range'] = {}
    
    
    #%%%% Corrected data
    
    #%%%%% Bootstrap analysis
    #    - number of bootstrap iterations for FFT analysis of corrected data
    data_dic['PCA']['nboot'] = 500
    
    
    #%%%%% Residuals histograms
    #    - number of bins to compute residual histograms of corrected data
    data_dic['PCA']['nbins'] = 50
    
    
    #%%%% Plots: PCA results
    plot_dic['pca_ana'] = ''
    
    
    
    
    
    
    ##################################################################################################
    #%%% Module: intrinsic profiles alignment
    #    - aligned in common frame
    #    - every analysis afterwards will be performed on those profiles
    ##################################################################################################  
    
    
    #%%%% Activating
    #    - required for some of the operations afterwards
    gen_dic['align_Intr'] = False
     
    
    #%%%% Calculating/retrieving
    gen_dic['calc_align_Intr'] = True 
    
    
    #%%%% Alignment mode
    #    - align profiles by their measured ('meas') or theoretical ('theo') RVs 
    #    - measured RV must have been derived from CCFs before, and are only applied if the stellar line was considered detected
    #    - if several planets are transiting in a given visit, use data_dic['Intr']['align_ref_pl']={inst:{vis}:{ref_pl}} to indicate which planet should be used
    data_dic['Intr']['align_mode']='theo'
    
    
    #%%%% Plots: all profiles 
    #    - plotting intrinsic stellar profiles together
    plot_dic['all_intr_data']=''   
    

    
    
    
        
    ##################################################################################################
    #%%% Module: 2D->1D conversion for differential & intrinsic spectra
    ##################################################################################################
    
    ANTARESS_2D_1D_settings('Intr',data_dic,gen_dic,plot_dic)
    
     
    
    ##################################################################################################
    #%%% Module: intrinsic profiles binning
    #    - for analysis purpose (original profiles are not replaced)
    ##################################################################################################
    
    #%%%% Activating
    gen_dic['Intrbin'] = False
    gen_dic['Intrbinmultivis'] = False
    
    
    #%%%% Calculating/retrieving
    gen_dic['calc_Intrbin']=True 
    gen_dic['calc_Intrbinmultivis']=True  
    
    
    #%%%% Visits to be binned
    #    - for the 'Intrbinmultivis' option
    #    - leave empty to use all visits
    data_dic['Intr']['vis_in_bin']={}  
    
    
    #%%%% Exposures to be binned
    #    - indexes are relative to the in-transit table in each visit
    #    - leave empty to use all in-exposures 
    #    - the selection will be used for the masters calculated for local profiles extraction as well
    data_dic['Intr']['idx_in_bin']={}
    
    
    #%%%% Binning dimension
    #    - possibilities:
    # + 'phase': profiles are binned over phase 
    # + 'xp_abs': profiles are binned over |xp| (absolute distance from projected orbital normal in the sky plane)
    # + 'r_proj': profiles are binned over r (distance from star center projected in the sky plane)
    #    - beware to use the alignement module if binned profiles should be calculated in the common rest frame
    data_dic['Intr']['dim_bin']='phase'  
    
    
    #%%%% Bin definition
    #    - see data_dic['DI']['prop_bin']
    data_dic['Intr']['prop_bin']={}
        
        
    #%%%% Plot settings        
        
    #%%%%% 2D maps of binned profiles
    plot_dic['map_Intrbin']=''    
    
    
    #%%%%% Individual binned profiles  
    plot_dic['Intrbin']=''   
    
    
    #%%%%% Residuals from binned profiles
    plot_dic['Intrbin_res']=''  
    
    
    #%%%%% Binned disk-integrated and intrinsic profiles comparison
    plot_dic['binned_DI_Intr']=''    
    
    

    
    ##################################################################################################
    #%%% Module: intrinsic CCF masks
    #    - spectra must have been aligned in the star rest frame, converted into a 1D profile, and binned. 
    #    - the mask is built by default over all processed visits of an instrument, for consistency of the CCFs between visits
    ##################################################################################################
    
    #%%%% Activating
    gen_dic['def_Intrmasks'] = False
    
    
    #%%%% Copying disk-integrated settings
    data_dic['Intr']['mask'] = deepcopy(data_dic['DI']['mask'])
    
    
    #%%%% Plot settings 
    
    #%%%%% Mask at successive steps
    plot_dic['Intrmask_spectra'] = ''
    
    
    #%%%%% Depth range selection
    plot_dic['Intrmask_ld']  = ''
    
    
    #%%%%% Minimum depth and width selection
    plot_dic['Intrmask_ld_lw'] = ''
    
    
    #%%%%% Position selection
    plot_dic['Intrmask_RVdev_fit'] = ''
    
    
    #%%%%% Telluric selection
    plot_dic['Intrmask_tellcont'] = ''
    
    
    #%%%%% VALD selection
    plot_dic['Intrmask_vald_depthcorr'] = ''
    
    
    #%%%%% Morphological (asymmetry) selection
    plot_dic['Intrmask_morphasym'] = ''
    
    
    #%%%%% Morphological (shape) selection
    plot_dic['Intrmask_morphshape'] = ''
    
    
    #%%%%% RV dispersion selection
    plot_dic['Intrmask_RVdisp'] = ''    
    

    
    ##################################################################################################
    #%%% Module: intrinsic profiles analysis
    #    - can be applied to:
    # + 'fit_Intr': profiles in the star rest frame, original exposures, for all formats
    # + 'fit_Intr_1D': profiles in the star or surface (if aligned) rest frame, original exposures, converted from 2D->1D 
    # + 'fit_Intrbin' : profiles in the star or surface (if aligned) rest frame, binned exposures, all formats
    #                   bin dimension of the fitted profile is set by data_dic['Intr']['dim_bin']
    # + 'fit_Intrbinmultivis' : profiles in the surface rest frame, binned exposures, all formats
    ##################################################################################################
    
    
    #%%%% Activating
    gen_dic['fit_Intr'] = False
    gen_dic['fit_Intr_1D'] = False
    gen_dic['fit_Intrbin']= False
    gen_dic['fit_Intrbinmultivis']= False
    
    
    #%%%% Calculating/Retrieving
    gen_dic['calc_fit_Intr']=True   
    gen_dic['calc_fit_Intr_1D']=True  
    gen_dic['calc_fit_Intrbin']=True  
    gen_dic['calc_fit_Intrbinmultivis']=True  
    
    
    #%%%% Fitted data
    ANTARESS_fit_prof_settings('Intr',data_dic)

    
    #%%%% Direct measurements
    #    - same as data_dic['DI']['meas_prop']={}
    #    - options that differ:
    # + equivalent width: 'EW' : {'rv_range':[rv1,rv2] single range over which the integral is performed, defined in the photosphere rest frame}                         
    # + bissector: 'biss' : {'rv_range':[rv1,rv2] maximum range over which bissector is calculated, defined in the photosphere rest frame}
    data_dic['Intr']['meas_prop']={}
    
    
    #%%%% Line profile model
    
    #%%%%% Transition wavelength
    #    - in the star rest frame
    #    - used to center the line analytical model
    #    - only relevant in spectral mode
    #    - do not use if the spectral fit is performed on more than a single line
    data_dic['Intr']['line_trans']=None
       
    
    #%%%%% Instrumental convolution
    #    - apply instrumental convolution (default) or not to model
    #    - beware that most derived properties will correspond to the model before convolution
    #      this is particularly useful to match the intrinsic line properties from the joint intrinsic fit with values derived here from the individual fits
    data_dic['Intr']['conv_model']=True  
    
    
    #%%%%% Model type
    #    - specific to each instrument
    #    - options: same as data_dic['DI']['model']
    #               for 'custom' the line profiles are calculated over the planet-occulted regions instead of the full star 
    #    - it is possible to fix the value, for each instrument and visit, of given parameters of the fit model
    #      if a field is given in 'mod_prop', the corresponding field in the model will be fixed to the given value    
    data_dic['Intr']['model']={}
    
    
    #%%%%% Intrinsic line properties
    #    - used if 'model' = 'custom' 
    data_dic['Intr']['mod_def']={}  
    
    
    #%%%%% Fixed/variable properties
    #    - same as data_dic['DI']['mod_prop']
    data_dic['Intr']['mod_prop']={}
    
    
    #%%%%% Best model table
    #    - resolution (dx) and range (min_x,max_x) of final model used for post-processsing of fit results and plots
    #    - in rv space and km/s for analytical profiles (profiles in wavelength space are modelled in RV space and then converted), in space of origin for measured profiles, in wavelength space for theoretical profiles 
    #    - specific to the instrument
    data_dic['Intr']['best_mod_tab']={}
    
    
    #%%%% Fit settings 
    #    - the width of the master disk-integrated profile can be used as upper limit for the prior on the intrinsic line FWHM
    ANTARESS_fit_def_settings('Intr',data_dic,plot_dic)
    
    
    #%%%% Plot settings

    #%%%%% Derived properties
    #    - from original or binned data
    plot_dic['prop_Intr']=''  
        
    
    
    #Activating
    gen_dic['fit_Intr'] = True   &  False
    gen_dic['fit_Intr_1D'] = True   &  False
    gen_dic['fit_Intrbin']=True     &  False
    gen_dic['fit_Intrbinmultivis']=True     &  False

    #Calculating/Retrieving
    gen_dic['calc_fit_Intr']=True #  &  False   
    gen_dic['calc_fit_Intr_1D']=True   &  False   
    gen_dic['calc_fit_Intrbin']=True    &  False    
    gen_dic['calc_fit_Intrbinmultivis']=True   &  False  

    #Constant data errors
    data_dic['Intr']['cst_err']=True   &  False
    data_dic['Intr']['cst_errbin']=True   &  False

    #Spectral range(s) to be fitted
    if gen_dic['star_name']=='AUMic':
        data_dic['Intr']['fit_range']['ESPRESSO']={'mock_vis' : [[-130.,130.]], #-- base 
                                                    'mock_vis1' : [[-130.,130.]],
                                                    'mock_vis2' : [[-130.,130.]],
                                                    'mock_vis3' : [[-130.,130.]],
                                                    'mock_vis4' : [[-130.,130.]],
                                                    'mock_vis5' : [[-130.,130.]],
                                                    'mock_vis6' : [[-130.,130.]],
                                                    'mock_vis7' : [[-130.,130.]],
                                                    'mock_vis8' : [[-130.,130.]],
                                                    'mock_vis9' : [[-130.,130.]],
        }
    
    if gen_dic['star_name']=='V1298tau':
        data_dic['Intr']['fit_range']['ESPRESSO']={'mock_vis' : [[-130.,130.]] }

    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:data_dic['Intr']['fit_range']['ESPRESSO']={'mock_vis' : [[-130.,130.]]}

    #Model type  
            
    #Intrinsic line properties
      
    #Fitting mode 
    data_dic['Intr']['fit_mode']=''
    data_dic['Intr']['fit_mode']='chi2'
    data_dic['Intr']['fit_mode']='mcmc'


    #Printing fits results
    data_dic['Intr']['verbose'] =  True      &   False  

    
    #Priors on variable properties                

  
    #Detection thresholds    
 
    
    #Force detection flag
      
    #Calculating/retrieving
    data_dic['Intr']['run_mode']='use'
    
    #Walkers
    
    #Walkers exclusion
    data_dic['Intr']['exclu_walk_autom']=None  #  5.
    
    
    #Derived errors


    #1D PDF from mcmc
    plot_dic['prop_Intr_PDFs']=''     

    #Derived properties
    plot_dic['prop_Intr']=''  
    
    
    
    
    
    
    
    
    

        
        
    ##################################################################################################       
    #%%% Module: intrinsic stellar properties fit
    #    - fitting single stellar surface property from planet-occulted regions with a common model for all instruments/visits 
    #    - with properties derived from individual local profiles
    #    - this module can be used to estimate the surface RV model and analytical laws describing the intrinsic line properties
    #      the final fit should be performed over the joined intrinsic line profiles with gen_dic['fit_IntrProf']
    ##################################################################################################       
    
    #%%%% Activating 
    gen_dic['fit_IntrProp'] = False
    
    
    #%%%% Multi-threading
    glob_fit_dic['IntrProp']['nthreads'] = int(0.8*cpu_count())

    
    #%%%%% Unthreaded operations
    #    - all operations are multi-threaded by default, but overheads of sharing data between threads may counterbalance the benefits of threading the model
    #    - select here which operations not to thread:
    # + 'emcee'
    glob_fit_dic['IntrProp']['unthreaded_op'] = []  
    
    
    #%%%% Fitted data
    
    #%%%%% Exposures to be fitted
    #    - indexes are relative to in-transit tables
    #    - define instruments and visits to be fitted (they will not be fitted if not used as keys, or if set to [], which is the default value), set their value to 'all' for all in-transit exposures to be fitted
    #    - add '_bin' at the end of a visit name for its binned exposures to be fitted instead of the original ones (must have been calculated with the binning module); it can be justified when surface RVs cannot be derived from unbinned profiles
    #      all other mentions of the visit (eg in parameter names) can still refer to the original visit name
    glob_fit_dic['IntrProp']['idx_in_fit'] = {}


    #%%%%% Scaled data errors
    #    - local scaling of data errors
    #    - you can scale by sqrt(reduced chi2 of original fit) to ensure a reduced chi2 unity
    glob_fit_dic['IntrProp']['sc_err']={}  


    #%%%% Fitted properties

    #%%%%% Properties and model
    #    - format is:
    # mod_prop = { prop_main : { prop_name : { 'vary' : bool , 'guess': x, 
    #                                                          'bd': [x_low,x_high] OR 'gauss': [val,s_val] } } }
    #      where 
    # > 'prop_main' defines the measured variable to be fitted
    # > 'prop_name' defines a property of the model describing 'prop_main'
    # > the other fields are described in data_dic['DI']['mod_prop'] 
    #    - typical variables:
    # + 'rv': fitted using surface RV model
    # + 'ctrst', 'FWHM': fitted using polynomial models
    #    - structure is different from data_dic['DI']['mod_prop'], where properties are fitted independently for each instrument and visit
    #      the names of properties varying as a function of 'coord_fit' and/or between visits must be defined as 'prop_name = prop__ordi__ISinst_VSvis'  
    # + 'i' is the polynomial degree
    # + 'inst' is the name of the instrument, which should be set to '_' for the property to be common to all instruments and their visits
    # + 'vis' is the name of the visit, which should be set to '_' for the property to be common to all visits of this instrument 
    #    - the names of properties specific to a given planet 'PL' must be defined as 'prop_name = prop__ordi__plPL'  
    #    - the names of properties specific to a given active region 'AR' must be defined as 'prop_name = prop__ordi__arAR' 
    #%%%% WARNING 
    # + for active region properties common to quiet star ones (veq, alpha_rot, beta_rot), no linking is done internally.
    # + therefore, the user must specify values for these properties if they wish to fix them to values which differ from the base ones provided in the systems file.
    glob_fit_dic['IntrProp']['mod_prop']={'rv':{}}


    #%%%%% Coordinate
    #    - define the coordinate as a function of which line shape properties are defined:
    # +'mu' angle       
    # +'xp_abs': absolute distance from projected orbital normal in the sky plane
    # +'r_proj': distance from star center projected in the sky plane      
    # +'abs_y_st' : sky-projected distance parallel to spin axis, absolute value   
    # +'y_st2' : sky-projected distance parallel to spin axis, squared
    #    - format is { prop1 : coord1, prop2 : coord2 }
    glob_fit_dic['IntrProp']['coord_fit']={'ctrst':'r_proj','FWHM':'r_proj'}
      
    
    #%%%%% Variation
    #    - fit line shape property as absolute ('abs') or modulated ('modul') polynomial
    glob_fit_dic['IntrProp']['pol_mode']='abs'     


    #%%%% Fit settings
    ANTARESS_fit_def_settings('IntrProp',glob_fit_dic,plot_dic)
      
    
        
        
        
        
        
        
        
    ##################################################################################################       
    #%%% Module: joined differential profiles fit    
    #    - fitting joined differential profiles from combined (unbinned) instruments and visits 
    #    - structure is similar to the joined intrinsic profiles fit
    #    - fits are performed on all in-transit and out-transit exposures
    #    - allows including active region properties (latitude, size, Tc_ar, flux level) in the fitted properties
    ##################################################################################################     
            
    #%%%% Activating 
    gen_dic['fit_DiffProf'] = False        
 
    
    #%%%% Multi-threading
    
    #%%%%% Allocated threads
    glob_fit_dic['DiffProf']['nthreads'] = int(0.8*cpu_count())
    
    
    #%%%%% Unthreaded operations
    #    - all operations are multi-threaded by default, but overheads of sharing data between threads may counterbalance the benefits of threading the model
    #    - select here which operations not to thread:
    # + 'emcee'
    # + 'prof_grid'
    glob_fit_dic['DiffProf']['unthreaded_op'] = []


    #%%%% Master out data

    #%%%%% Exposures to be used in the calculation of the master-out
    glob_fit_dic['DiffProf']['idx_in_master_out']={}


    #%%%%% Common table on which we want to define the master-out
    #     - Define the borders and the number of points of the table (e.g. [low_end, high_end, num_pts].
    glob_fit_dic['DiffProf']['master_out_tab']=[]


    #%%%%% Reference planet
    #     - choosing which planet to use as the reference
    glob_fit_dic['DiffProf']['ref_pl']={}

    
    #%%%% Fitted data
    ANTARESS_fit_prof_settings('DiffProf',glob_fit_dic)
    
    
    #%%%%% Continuum range
    glob_fit_dic['DiffProf']['cont_range'] = {}


    #%%%% Line profile model         
        
    #%%%%% Transition wavelength
    glob_fit_dic['DiffProf']['line_trans']=None        
    
    
    #%%%%% Model type
    glob_fit_dic['DiffProf']['mode'] = 'ana' 
    
     
    #%%%%% Analytical profile
    #    - default: 'gauss' 
    glob_fit_dic['DiffProf']['model'] = {}

    
    #%%%%% Fixed/variable properties
    #    - structure is the same as glob_fit_dic['IntrProp']['mod_prop']
    #    - intrinsic properties define the lines before instrumental convolution, which can then be applied specifically to each instrument  
    glob_fit_dic['DiffProf']['mod_prop']={}
                 
        
    #%%%%% Analytical profile coordinate
    #    - fit coordinate for the line properties of analytical profiles
    #    - see possibilities in gen_dic['fit_IntrProp']
    glob_fit_dic['DiffProf']['coord_fit']='r_proj'
    
    
    #%%%%% Analytical profile variation
    #    - fit line property as absolute ('abs') or modulated ('modul') polynomial        
    glob_fit_dic['DiffProf']['pol_mode']='abs'  


    #%%%%% PC noise model
    #    - TBD 
    glob_fit_dic['DiffProf']['PC_model']={}  


    #%%%%% Optimization levels 
    #    - relevant when local line profiles are calculated in each cell of the stellar grid (theo_dic['precision'] = 'high')
    #    - set 'Opt_Lvl' to:
    # + 0: multithreading is controlled by the number of threads provided by the user AND over-simplified grid building is not used.
    # + 1: multithreading turned off AND over-simplified grid building is not used.
    # + 2: multithreading turned off AND over-simplified grid building is used
    # + 3: multithreading turned off AND over-simplified grid building is used AND grid building function coded in C 
    #    - over-simplified grid building: instead of assigning complex profiles to individual cells and summing them for the entire disk, we now use Gaussian profiles for each cell. 
    #      additionally, we optimize performance by representing the grid of profiles as an array rather than a list.
    glob_fit_dic['DiffProf']['Opt_Lvl']=0
    
    
    #%%%% Fit settings  
    ANTARESS_fit_def_settings('DiffProf',glob_fit_dic,plot_dic)


    #Activating 
    gen_dic['fit_DiffProf'] = True  #&  False

    #%%%%% Optimization levels
    if gen_dic['star_name'] in ['HD189733','TOI3884','AU_Mic','AUMic','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        glob_fit_dic['DiffProf']['Opt_Lvl']=3
        # glob_fit_dic['DiffProf']['nthreads']=1



    # Indexes of exposures to be fitted, in each visit
    #    - define instruments and visits to be fitted (they will not be fitted if not used as keys, or if set to []), set their value to 'all' for all in-transit exposures to be fitted
    if gen_dic['star_name'] == 'AUMic':
        glob_fit_dic['DiffProf']['idx_in_fit'] = {'ESPRESSO':{'mock_vis':'all'}}

    if gen_dic['star_name'] == 'AU_Mic':
        glob_fit_dic['DiffProf']['idx_in_fit'] = {'ESPRESSO':{'visit1':'all'}}

    if gen_dic['star_name'] == 'HD189733':
        glob_fit_dic['DiffProf']['idx_in_fit'] = {'ESPRESSO':{'visit1':'all'}}

    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        glob_fit_dic['DiffProf']['idx_in_fit'] = {'ESPRESSO':{'mock_vis':'all'}}

    if gen_dic['star_name'] == 'TOI3884':
        glob_fit_dic['DiffProf']['idx_in_fit'] = {'MIKE_Red':{'mockvis':'all'}}




    # Master-out RV table
    if gen_dic['star_name']=='AUMic':
        # glob_fit_dic['DiffProf']['master_out_tab']=[-90, 90, 200]
        glob_fit_dic['DiffProf']['master_out_tab']=[]

    if gen_dic['star_name']=='HD189733':
        glob_fit_dic['DiffProf']['master_out_tab']=[-89, 89, 200]
        # glob_fit_dic['DiffProf']['master_out_tab']=[]

    #Zodiacs
    if gen_dic['star_name'] in ['TOI3884','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        glob_fit_dic['DiffProf']['master_out_tab']=[]




    # Reference planet
    if gen_dic['star_name']=='AUMic':
        glob_fit_dic['DiffProf']['ref_pl']={'ESPRESSO':{'mock_vis':'AUMicb'}}

    if gen_dic['star_name']=='AU_Mic':
        glob_fit_dic['DiffProf']['ref_pl']={'ESPRESSO':{'visit1':'AU_Mic_b'}}

    if gen_dic['star_name']=='HD189733':
        glob_fit_dic['DiffProf']['ref_pl']={'ESPRESSO':{'visit1':'HD189733b'}}

    if gen_dic['star_name']=='TOI3884':
        glob_fit_dic['DiffProf']['ref_pl']={'MIKE_Red':{'mockvis':'TOI3884_b'}}

    #Zodiacs
    for zodiac in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        if gen_dic['star_name'] == zodiac:
            glob_fit_dic['DiffProf']['ref_pl']={'ESPRESSO':{'mock_vis':zodiac_pl}}



    #Trimming
    glob_fit_dic['DiffProf']['trim_range'] = deepcopy(data_dic['DI']['fit_prof']['trim_range'])   



    #Continuum range
    if gen_dic['star_name'] == 'AUMic':
        glob_fit_dic['DiffProf']['cont_range']={'ESPRESSO':{0:[[-150.0,-70.0],[70.0,150.0]]}}

    if gen_dic['star_name'] == 'TOI3884':
        glob_fit_dic['DiffProf']['cont_range']={'MIKE_Red':{0:[[-30.0,-20.0],[20.0,30.0]]}}

    if gen_dic['star_name'] == 'AU_Mic':
        glob_fit_dic['DiffProf']['cont_range']={'ESPRESSO':{0:[[-20.0, -15.0],[15.0, 20.0]]}}

    if gen_dic['star_name'] == 'HD189733':
        glob_fit_dic['DiffProf']['cont_range']={'ESPRESSO':{0:[[-90.0, -30.0],[30.0, 90.0]]}}

    #Zodiacs
    if gen_dic['star_name'] in ['Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        glob_fit_dic['DiffProf']['cont_range']={'ESPRESSO':{0:[[-150.0,-70.0],[70.0,150.0]]}}




    #Spectral range(s) to be fitted            
    glob_fit_dic['DiffProf']['fit_range'] = deepcopy(data_dic['Intr']['fit_range'])
    if gen_dic['star_name'] == 'AU_Mic':
        glob_fit_dic['DiffProf']['fit_range']['ESPRESSO']={'visit1' : [[-20.,13.]]}
    if gen_dic['star_name']=='HD189733':
        glob_fit_dic['DiffProf']['fit_range']['ESPRESSO']={'visit1' : [[-90.,90.]]}
    if gen_dic['star_name'] == 'TOI3884':
        glob_fit_dic['DiffProf']['fit_range']['MIKE_Red']={'mockvis' : [[-15.,15.]]}

    #Model type

    # Analytical profile
    if gen_dic['star_name'] in ['TOI3884','AU_Mic','AUMic','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        glob_fit_dic['DiffProf']['model']={'ESPRESSO':'gauss'}
    if gen_dic['star_name']=='HD189733':
        glob_fit_dic['DiffProf']['model']={'ESPRESSO':'gauss'}
    
    #Analytical profile coordinate
    if gen_dic['star_name'] in ['HD189733','TOI3884','AU_Mic','AUMic','V1298tau','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        glob_fit_dic['DiffProf']['dim_fit']='r_proj'


    #Analytical profile variation
    if gen_dic['star_name'] in ['HD189733','TOI3884','AU_Mic','AUMic','V1298tau','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        glob_fit_dic['DiffProf']['pol_mode']='modul'  

     
    #Fixed/variable properties 
    if gen_dic['star_name']=='HD189733':
        #Round 1 - wide guessing
        # glob_fit_dic['DiffProf']['mod_prop']={
        # 'cont__IS__VS_':{'vary':False, 'guess':1.0, 'bd':[0.9, 1.1]},
        # 'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.65, 'bd':[0.1, 0.9]},
        # 'FWHM__ord0__IS__VS_':{'vary':True, 'guess':8, 'bd':[5, 15]},
        # 'lat__ISESPRESSO_VSvisit1_ARspot1'     : {'vary':True, 'guess':0, 'bd':[-90, 90]},
        # 'Tc_ar__ISESPRESSO_VSvisit1_ARspot1' : {'vary':True, 'guess':2459457.589323-0.2, 'bd':[2459457.589323 - 1., 2459457.589323 + 1.]},
        # 'ang__ISESPRESSO_VSvisit1_ARspot1'     : {'vary':True, 'guess':15, 'bd':[2, 60]},
        # 'fctrst__ISESPRESSO_VSvisit1_ARspot1'   : {'vary':True, 'guess':0.3, 'bd':[0.1, 0.9]},
        # 'lambda_rad__plHD189733b'                   : {'vary':False, 'guess':-0.014, 'bd':[-2*np.pi, 2*np.pi]}
        #                                     }
        #Round 2 - more specific guessing
        glob_fit_dic['DiffProf']['mod_prop']={
        'cont__IS__VS_':{'vary':False, 'guess':1.0, 'bd':[0.9, 1.1]},
        'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.53, 'bd':[0.45, 0.61]},
        'FWHM__ord0__IS__VS_':{'vary':True, 'guess':6.9, 'bd':[6.7, 7.1]},
        'lat__ISESPRESSO_VSvisit1_ARspot1'     : {'vary':True, 'guess':-37, 'bd':[-75, 5]},
        'Tc_ar__ISESPRESSO_VSvisit1_ARspot1' : {'vary':True, 'guess':2459457.6703, 'bd':[2459457.17, 2459458.2]},
        'ang__ISESPRESSO_VSvisit1_ARspot1'     : {'vary':True, 'guess':25, 'bd':[16, 34]},
        'fctrst__ISESPRESSO_VSvisit1_ARspot1'   : {'vary':True, 'guess':0.6, 'bd':[0.5, 0.8]},
        'lambda_rad__plHD189733b'                   : {'vary':False, 'guess':-0.014, 'bd':[-2*np.pi, 2*np.pi]}
                                            }

    if gen_dic['star_name']=='AUMic':
        glob_fit_dic['DiffProf']['mod_prop']={
        'cont__IS__VS_':{'vary':False, 'guess':1.0, 'bd':[0.9, 1.1]},
        # 'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.65, 'bd':[0.6, 0.8]},
        'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.65, 'bd':[0.6, 0.8]},
        # 'ctrst__ord0__IS__VS_':{'vary':False, 'guess':0.7, 'bd':[0.15, 1]},
        # 'FWHM__ord0__IS__VS_':{'vary':True, 'guess':12, 'bd':[5, 15]},
        'FWHM__ord0__IS__VS_':{'vary':False, 'guess':8, 'bd':[5, 15]},
        # 'veq':{'vary':True,'guess':5, 'bd':[1, 10]},
        'veq':{'vary':False,'guess':7.9, 'bd':[1, 10]},
        # 'vsini':{'vary':True,'guess':7, 'bd':[0., 10.]},
        # 'veq_spots':{'vary':True,'guess':9, 'bd':[1, 10]},
        # 'veq_spots':{'vary':False,'guess':9.2, 'bd':[1, 10]},
        # 'alpha_rot':{'vary':True,'guess':0., 'bd':[0, 1]},
        # 'alpha_rot':{'vary':False,'guess':0., 'bd':[0, 1]},
        # 'alpha_rot_spots':{'vary':True,'guess':0.2, 'bd':[0, 1]},
        # 'alpha_rot_spots':{'vary':False,'guess':0., 'bd':[0, 1]},
        # 'beta_rot':{'vary':True,'guess':0.1, 'bd':[0, 1]},
        # 'beta_rot':{'vary':False,'guess':0., 'bd':[0, 1]},
        # 'beta_rot_spots':{'vary':True,'guess':0., 'bd':[0, 1]},
        # 'beta_rot_spots':{'vary':False,'guess':0., 'bd':[0, 1]},
        # 'cos_istar':{'vary':True,'guess':0.1, 'bd':[-1., 1.]},
        'cos_istar':{'vary':False,'guess':0.01745240644, 'bd':[-1., 1.]},

        # 'lat__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':0, 'bd':[-50, 10]},
        'lat__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':False, 'guess':-30, 'bd':[-50, 10]},
        # 'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : {'vary':True, 'guess':2458330.39051, 'bd':[2458330.39051 - 10., 2458330.39051 + 10.]},
        'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : {'vary':False, 'guess':2458330.39051-0.3, 'bd':[2458330.39051 - 0.4, 2458330.39051 + 0.4]},
        # 'ang__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':15, 'bd':[2, 80]},
        'ang__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':False, 'guess':15, 'bd':[10, 50]},
        # 'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   : {'vary':True, 'guess':0.3, 'bd':[0.05, 0.7]},
        'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   : {'vary':False, 'guess':0.1, 'bd':[0.05, 0.7]},
        
        # # 'lat__ISESPRESSO_VSmock_vis_ARfacula1'     : {'vary':True, 'guess':0, 'bd':[-50, 10]},
        # 'lat__ISESPRESSO_VSmock_vis_ARfacula1'     : {'vary':False, 'guess':0, 'bd':[-50, 10]},
        # # 'Tc_ar__ISESPRESSO_VSmock_vis_ARfacula1' : {'vary':True, 'guess':2458330.39051, 'bd':[2458330.39051 - 10., 2458330.39051 + 10.]},
        # 'Tc_ar__ISESPRESSO_VSmock_vis_ARfacula1' : {'vary':False, 'guess':2458330.39051-0.3, 'bd':[2458330.39051 - 0.4, 2458330.39051 + 0.4]},
        # # 'ang__ISESPRESSO_VSmock_vis_ARfacula1'     : {'vary':True, 'guess':15, 'bd':[2, 80]},
        # 'ang__ISESPRESSO_VSmock_vis_ARfacula1'     : {'vary':False, 'guess':25, 'bd':[10, 50]},
        # # 'fctrst__ISESPRESSO_VSmock_vis_ARfacula1'   : {'vary':True, 'guess':1.5, 'bd':[1.3, 1.9]},
        # 'fctrst__ISESPRESSO_VSmock_vis_ARfacula1'   : {'vary':False, 'guess':1.5, 'bd':[1.3, 1.9]},

        # 'lambda_rad__plAUMicb'                   : {'vary':True, 'guess':0.01, 'bd':[-2*np.pi, 2*np.pi]}
        'lambda_rad__plAUMicb'                   : {'vary':False, 'guess':-0.08203047484, 'bd':[-2*np.pi, 2*np.pi]}
                                            }


    if gen_dic['star_name']=='AU_Mic':
        glob_fit_dic['DiffProf']['mod_prop']={
        'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.7, 'bd':[0.1, 1]},
        'FWHM__ord0__IS__VS_':{'vary':True, 'guess':8, 'bd':[1, 20]},
        'veq':{'vary':True,'guess':8.595, 'bd':[8.4, 8.75]},
        'alpha_rot':{'vary':True, 'guess':0.034, 'bd':[0.033,0.035]},
        'cos_istar':{'vary':True,'guess':0.1736, 'bd':[-1., 1.]},

        'lat__ISESPRESSO_VSvisit1_ARspot1'     : {'vary':True, 'guess':0, 'bd':[-90, 90]},
        'Tc_ar__ISESPRESSO_VSvisit1_ARspot1' : {'vary':True, 'guess':2458702.77-0.2, 'bd':[2458702.77 - 1, 2458702.77 + 1]},
        'ang__ISESPRESSO_VSvisit1_ARspot1'     : {'vary':True, 'guess':25, 'bd':[1, 60]},
        'fctrst__ISESPRESSO_VSvisit1_ARspot1'   : {'vary':True, 'guess':0.1, 'bd':[0.01, 0.99]},

        'lat__ISESPRESSO_VSvisit1_ARspot2'     : {'vary':True, 'guess':-30, 'bd':[-90, 90]},
        'Tc_ar__ISESPRESSO_VSvisit1_ARspot2' : {'vary':True, 'guess':2458702.77+0.2, 'bd':[2458702.77 - 1, 2458702.77 + 1]},
        'ang__ISESPRESSO_VSvisit1_ARspot2'     : {'vary':True, 'guess':25, 'bd':[1, 60]},
        'fctrst__ISESPRESSO_VSvisit1_ARspot2'   : {'vary':True, 'guess':0.1, 'bd':[0.01, 0.99]},

        'rv_shift'                        : {'vary':True, 'guess': 1, 'bd':[-5, 5]},  

        'lambda_rad__plAU_Mic_b'                   : {'vary':True, 'guess':0, 'bd':[-2*np.pi, 2*np.pi]}
                                            }

    if gen_dic['star_name']=='TOI3884':
        glob_fit_dic['DiffProf']['mod_prop']={
        'ctrst__ord0__IS__VS_':{'vary':False, 'guess':0.7, 'bd':[0.1, 1]},
        'FWHM__ord0__IS__VS_':{'vary':False, 'guess':8, 'bd':[1, 20]},
        'veq':{'vary':False,'guess':8.495, 'bd':[2, 30]},
        'cos_istar':{'vary':False,'guess':-0.906307787, 'bd':[-1., 1.]},

        'lat__ISMIKE_Red_VSmockvis_ARspot1'     : {'vary':False, 'guess':-90, 'bd':[-90, 90]},
        'Tc_ar__ISMIKE_Red_VSmockvis_ARspot1' : {'vary':False, 'guess':2459642.86314 + 2.4, 'bd':[2459642.86314 - 3, 2459642.86314 + 3]},
        'ang__ISMIKE_Red_VSmockvis_ARspot1'     : {'vary':False, 'guess':48.6, 'bd':[1, 60]},
        'fctrst__ISMIKE_Red_VSmockvis_ARspot1'   : {'vary':False, 'guess':0.41, 'bd':[0.001, 0.999]},

        'lambda_rad__plTOI3884_b'                   : {'vary':True, 'guess':0.7, 'bd':[-2*np.pi, 2*np.pi]}
                                            }
    
    #Zodiacs
    if gen_dic['star_name']=='Capricorn':
        glob_fit_dic['DiffProf']['mod_prop']={
        'cont__IS__VS_':{'vary':False, 'guess':1.0, 'bd':[0.9, 1.1]},
        'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.4, 'bd':[0.1, 1]},
        # 'ctrst__ord0__IS__VS_':{'vary':False, 'guess':0.7, 'bd':[0.1, 1]},
        'FWHM__ord0__IS__VS_':{'vary':True, 'guess':12, 'bd':[1, 20]},
        # 'FWHM__ord0__IS__VS_':{'vary':False, 'guess':8, 'bd':[1, 20]},
        'veq':{'vary':True,'guess':15, 'bd':[12, 24]},
        # 'veq':{'vary':False,'guess':12, 'bd':[2, 30]},
        'cos_istar':{'vary':True,'guess':0.01, 'bd':[-1., 1.]},

        'lat__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':0, 'bd':[-90, 90]},
        'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : {'vary':True, 'guess':2458330.59051, 'bd':[2458330.29051 - 1, 2458330.29051 + 1]},
        'ang__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':25, 'bd':[1, 60]},
        'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   : {'vary':True, 'guess':0.3, 'bd':[0.001, 0.999]},

        'lambda_rad__pl'+zodiac_pl                   : {'vary':True, 'guess':0., 'bd':[-2*np.pi, 2*np.pi]}
                                            }
    
    if gen_dic['star_name']=='Cancer':
        glob_fit_dic['DiffProf']['mod_prop']={
        'cont__IS__VS_':{'vary':False, 'guess':1.0, 'bd':[0.9, 1.1]},
        'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.4, 'bd':[0.1, 1]},
        'FWHM__ord0__IS__VS_':{'vary':True, 'guess':12, 'bd':[1, 20]},
        'veq':{'vary':True,'guess':0.15, 'bd':[0.1, 0.3]},
        'cos_istar':{'vary':True,'guess':0.01, 'bd':[-1., 1.]},

        'lat__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':0., 'bd':[-90, 90]},
        'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : {'vary':True, 'guess':2458330.59051, 'bd':[2458330.29051 - 1, 2458330.29051 + 1]},
        'ang__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':25, 'bd':[1, 60]},
        'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   : {'vary':True, 'guess':0.3, 'bd':[0.001, 0.999]},

        'lambda_rad__pl'+zodiac_pl                   : {'vary':True, 'guess':0., 'bd':[-2*np.pi, 2*np.pi]}
                                            }

    if gen_dic['star_name']=='Gemini':
        glob_fit_dic['DiffProf']['mod_prop']={
        'cont__IS__VS_':{'vary':False, 'guess':1.0, 'bd':[0.9, 1.1]},
        'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.4, 'bd':[0.1, 1]},
        # 'ctrst__ord0__IS__VS_':{'vary':False, 'guess':0.7, 'bd':[0.1, 1]},
        'FWHM__ord0__IS__VS_':{'vary':True, 'guess':12, 'bd':[1, 20]},
        # 'FWHM__ord0__IS__VS_':{'vary':False, 'guess':8, 'bd':[1, 20]},
        'veq':{'vary':True,'guess':8.0, 'bd':[7., 9.]},
        # 'veq':{'vary':False,'guess':7.8, 'bd':[2, 30]},
        # 'veq_spots':{'vary':True,'guess':10, 'bd':[2, 30]},
        'cos_istar':{'vary':True,'guess':0.01, 'bd':[-1., 1.]},
        # 'cos_istar':{'vary':False,'guess':0.0348994967, 'bd':[-1., 1.]},

        'lat__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':0, 'bd':[-90, 90]},
        # 'lat__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':False, 'guess':-30, 'bd':[-90, 90]},
        'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : {'vary':True, 'guess':2458330.59051, 'bd':[2458330.29051 - 1, 2458330.29051 + 1]},
        # 'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : {'vary':False, 'guess':2458330.29051, 'bd':[2458330.29051 - 1, 2458330.29051 + 1]},
        'ang__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':25, 'bd':[1, 60]},
        # 'ang__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':False, 'guess':15, 'bd':[1, 60]},
        'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   : {'vary':True, 'guess':0.3, 'bd':[0.001, 0.999]},
        # 'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   : {'vary':False, 'guess':0.4, 'bd':[0.001, 0.999]},

        'lambda_rad__pl'+zodiac_pl                   : {'vary':True, 'guess':0., 'bd':[-2*np.pi, 2*np.pi]}
        # 'lambda_rad__pl'+zodiac_pl                   : {'vary':False, 'guess':-0.08203047484, 'bd':[-2*np.pi, 2*np.pi]}
                                            }

    if gen_dic['star_name']=='Sagittarius':
        glob_fit_dic['DiffProf']['mod_prop']={
        'cont__IS__VS_':{'vary':False, 'guess':1.0, 'bd':[0.9, 1.1]},
        'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.4, 'bd':[0.1, 1]},
        # 'ctrst__ord0__IS__VS_':{'vary':False, 'guess':0.7, 'bd':[0.1, 1]},
        'FWHM__ord0__IS__VS_':{'vary':True, 'guess':12, 'bd':[1, 20]},
        # 'FWHM__ord0__IS__VS_':{'vary':False, 'guess':8, 'bd':[1, 20]},
        'veq':{'vary':True,'guess':8., 'bd':[7., 9.]},
        # 'veq':{'vary':False,'guess':7.8, 'bd':[2, 30]},
        'cos_istar':{'vary':True,'guess':0.01, 'bd':[-1., 1.]},
        # 'cos_istar':{'vary':False,'guess':0.0348994967, 'bd':[-1., 1.]},

        'lat__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':0, 'bd':[-90, 90]},
        # 'lat__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':False, 'guess':-30, 'bd':[-90, 90]},
        'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : {'vary':True, 'guess':2458330.59051, 'bd':[2458330.29051 - 1, 2458330.29051 + 1]},
        # 'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : {'vary':False, 'guess':2458330.29051, 'bd':[2458330.29051 - 1, 2458330.29051 + 1]},
        'ang__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':25, 'bd':[1, 60]},
        # 'ang__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':False, 'guess':15, 'bd':[1, 60]},
        'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   : {'vary':True, 'guess':0.3, 'bd':[0.001, 0.999]},
        # 'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   : {'vary':False, 'guess':0.4, 'bd':[0.001, 0.999]},

        'lambda_rad__pl'+zodiac_pl                   : {'vary':True, 'guess':0., 'bd':[-2*np.pi, 2*np.pi]},
        # 'lambda_rad__pl'+zodiac_pl                   : {'vary':False, 'guess':-0.08203047484, 'bd':[-2*np.pi, 2*np.pi]}
                                            }
                                            
    if gen_dic['star_name']=='Leo':
        glob_fit_dic['DiffProf']['mod_prop']={
        'cont__IS__VS_':{'vary':False, 'guess':1.0, 'bd':[0.9, 1.1]},
        'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.4, 'bd':[0.1, 1]},
        # 'ctrst__ord0__IS__VS_':{'vary':False, 'guess':0.7, 'bd':[0.1, 1]},
        'FWHM__ord0__IS__VS_':{'vary':True, 'guess':12, 'bd':[1, 20]},
        # 'FWHM__ord0__IS__VS_':{'vary':False, 'guess':8, 'bd':[1, 20]},
        'veq':{'vary':True,'guess':8., 'bd':[7., 9.]},
        # 'veq':{'vary':False,'guess':7.8, 'bd':[2, 30]},
        'cos_istar':{'vary':True,'guess':0.01, 'bd':[-1., 1.]},

        'lat__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':0, 'bd':[-90, 90]},
        'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : {'vary':True, 'guess':2458330.59051, 'bd':[2458330.29051 - 1, 2458330.29051 + 1]},
        'ang__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':20, 'bd':[1, 60]},
        'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   : {'vary':True, 'guess':0.3, 'bd':[0.001, 0.999]},

        'lambda_rad__pl'+zodiac_pl                   : {'vary':True, 'guess':0., 'bd':[-2*np.pi, 2*np.pi]}
                                            }
                                            
    if gen_dic['star_name']=='Aquarius':
        glob_fit_dic['DiffProf']['mod_prop']={
        'cont__IS__VS_':{'vary':False, 'guess':1.0, 'bd':[0.9, 1.1]},
        'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.4, 'bd':[0.1, 1]},
        # 'ctrst__ord0__IS__VS_':{'vary':False, 'guess':0.7, 'bd':[0.1, 1]},
        'FWHM__ord0__IS__VS_':{'vary':True, 'guess':12, 'bd':[1, 20]},
        # 'FWHM__ord0__IS__VS_':{'vary':False, 'guess':8, 'bd':[1, 20]},
        'veq':{'vary':True,'guess':8., 'bd':[7., 9.]},
        # 'veq':{'vary':False,'guess':7.8, 'bd':[2, 30]},
        'cos_istar':{'vary':True,'guess':0.0348994967, 'bd':[-1., 1.]},

        'lat__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':0, 'bd':[-90, 90]},
        'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : {'vary':True, 'guess':2458330.59051, 'bd':[2458330.29051 - 1, 2458330.29051 + 1]},
        'ang__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':20, 'bd':[1, 60]},
        'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   : {'vary':True, 'guess':0.3, 'bd':[0.001, 0.999]},

        'lambda_rad__pl'+zodiac_pl                   : {'vary':True, 'guess':0., 'bd':[-2*np.pi, 2*np.pi]}
                                            }
                                            
    if gen_dic['star_name']=='Aries':
        glob_fit_dic['DiffProf']['mod_prop']={
        'cont__IS__VS_':{'vary':False, 'guess':1.0, 'bd':[0.9, 1.1]},
        'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.4, 'bd':[0.1, 1]},
        # 'ctrst__ord0__IS__VS_':{'vary':False, 'guess':0.7, 'bd':[0.1, 1]},
        'FWHM__ord0__IS__VS_':{'vary':True, 'guess':12, 'bd':[1, 20]},
        # 'FWHM__ord0__IS__VS_':{'vary':False, 'guess':8, 'bd':[1, 20]},
        'veq':{'vary':True,'guess':8., 'bd':[7., 9.]},
        # 'veq':{'vary':False,'guess':7.8, 'bd':[2, 30]},
        'cos_istar':{'vary':True,'guess':0.01, 'bd':[-1., 1.]},

        'lat__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':0, 'bd':[-90, 90]},
        'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : {'vary':True, 'guess':2458330.59051, 'bd':[2458330.29051 - 1, 2458330.29051 + 1]},
        'ang__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':25, 'bd':[1, 60]},
        'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   : {'vary':True, 'guess':0.3, 'bd':[0.001, 0.999]},

        'lambda_rad__pl'+zodiac_pl                   : {'vary':True, 'guess':0., 'bd':[-2*np.pi, 2*np.pi]}
                                            }
                                            
    if gen_dic['star_name']=='Libra':
        glob_fit_dic['DiffProf']['mod_prop']={
        'cont__IS__VS_':{'vary':False, 'guess':1.0, 'bd':[0.9, 1.1]},
        'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.4, 'bd':[0.1, 1]},
        # 'ctrst__ord0__IS__VS_':{'vary':False, 'guess':0.7, 'bd':[0.1, 1]},
        'FWHM__ord0__IS__VS_':{'vary':True, 'guess':12, 'bd':[1, 20]},
        # 'FWHM__ord0__IS__VS_':{'vary':False, 'guess':8, 'bd':[1, 20]},
        'veq':{'vary':True,'guess':8., 'bd':[7., 9.]},
        # 'veq':{'vary':False,'guess':7.8, 'bd':[2, 30]},
        'cos_istar':{'vary':True,'guess':0.01, 'bd':[-1., 1.]},

        'lat__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':0, 'bd':[-90, 90]},
        'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : {'vary':True, 'guess':2458330.59051, 'bd':[2458330.29051 - 1, 2458330.29051 + 1]},
        'ang__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':25, 'bd':[1, 60]},
        'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   : {'vary':True, 'guess':0.6, 'bd':[0.001, 0.999]},

        'lambda_rad__pl'+zodiac_pl                   : {'vary':True, 'guess':0., 'bd':[-2*np.pi, 2*np.pi]}
                                            }
                                            
    if gen_dic['star_name']=='Taurus':
        glob_fit_dic['DiffProf']['mod_prop']={
        'cont__IS__VS_':{'vary':False, 'guess':1.0, 'bd':[0.9, 1.1]},
        'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.4, 'bd':[0.1, 1]},
        # 'ctrst__ord0__IS__VS_':{'vary':False, 'guess':0.7, 'bd':[0.1, 1]},
        'FWHM__ord0__IS__VS_':{'vary':True, 'guess':12, 'bd':[1, 20]},
        # 'FWHM__ord0__IS__VS_':{'vary':False, 'guess':8, 'bd':[1, 20]},
        'veq':{'vary':True,'guess':8., 'bd':[7., 9.]},
        # 'veq':{'vary':False,'guess':7.8, 'bd':[2, 30]},
        'cos_istar':{'vary':True,'guess':0.01, 'bd':[-1., 1.]},

        'lat__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':0, 'bd':[-90, 90]},
        'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : {'vary':True, 'guess':2458330.59051, 'bd':[2458330.29051 - 1, 2458330.29051 + 1]},
        'ang__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':25, 'bd':[1, 60]},
        'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   : {'vary':True, 'guess':0.7, 'bd':[0.001, 0.999]},

        'lambda_rad__pl'+zodiac_pl                   : {'vary':True, 'guess':0., 'bd':[-2*np.pi, 2*np.pi]}
                                            }
                                            
    if gen_dic['star_name']=='Scorpio':
        glob_fit_dic['DiffProf']['mod_prop']={
        'cont__IS__VS_':{'vary':False, 'guess':1.0, 'bd':[0.9, 1.1]},
        'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.4, 'bd':[0.1, 1]},
        # 'ctrst__ord0__IS__VS_':{'vary':False, 'guess':0.7, 'bd':[0.1, 1]},
        'FWHM__ord0__IS__VS_':{'vary':True, 'guess':12, 'bd':[1, 20]},
        # 'FWHM__ord0__IS__VS_':{'vary':False, 'guess':8, 'bd':[1, 20]},
        'veq':{'vary':True,'guess':8., 'bd':[7., 9.]},
        # 'veq':{'vary':False,'guess':7.8, 'bd':[2, 30]},
        'cos_istar':{'vary':True,'guess':0.01, 'bd':[-1., 1.]},

        'lat__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':0, 'bd':[-90, 90]},
        'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : {'vary':True, 'guess':2458330.59051, 'bd':[2458330.29051 - 1, 2458330.29051 + 1]},
        'ang__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':25, 'bd':[1, 60]},
        'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   : {'vary':True, 'guess':0.7, 'bd':[0.001, 0.999]},

        'lambda_rad__pl'+zodiac_pl                   : {'vary':True, 'guess':0., 'bd':[-2*np.pi, 2*np.pi]}
                                            }
                                            
    if gen_dic['star_name']=='Virgo':
        glob_fit_dic['DiffProf']['mod_prop']={
        'cont__IS__VS_':{'vary':False, 'guess':1.0, 'bd':[0.9, 1.1]},
        'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.4, 'bd':[0.1, 1]},
        # 'ctrst__ord0__IS__VS_':{'vary':False, 'guess':0.7, 'bd':[0.1, 1]},
        'FWHM__ord0__IS__VS_':{'vary':True, 'guess':12, 'bd':[1, 20]},
        # 'FWHM__ord0__IS__VS_':{'vary':False, 'guess':8, 'bd':[1, 20]},
        'veq':{'vary':True,'guess':8., 'bd':[7., 9.]},
        # 'veq':{'vary':False,'guess':7.8, 'bd':[2, 30]},
        'cos_istar':{'vary':True,'guess':0.01, 'bd':[-1., 1.]},

        'lat__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':0, 'bd':[-90, 90]},
        'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : {'vary':True, 'guess':2458330.59051, 'bd':[2458330.29051 - 1, 2458330.29051 + 1]},
        'ang__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':25, 'bd':[1, 60]},
        'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   : {'vary':True, 'guess':0.7, 'bd':[0.001, 0.999]},

        'lambda_rad__pl'+zodiac_pl                   : {'vary':True, 'guess':0., 'bd':[-2*np.pi, 2*np.pi]}
                                            }
                                            
    if gen_dic['star_name']=='Pisces':
        glob_fit_dic['DiffProf']['mod_prop']={
        'cont__IS__VS_':{'vary':False, 'guess':1.0, 'bd':[0.9, 1.1]},
        'ctrst__ord0__IS__VS_':{'vary':True, 'guess':0.4, 'bd':[0.1, 1]},
        # 'ctrst__ord0__IS__VS_':{'vary':False, 'guess':0.7, 'bd':[0.1, 1]},
        'FWHM__ord0__IS__VS_':{'vary':True, 'guess':12, 'bd':[1, 20]},
        # 'FWHM__ord0__IS__VS_':{'vary':False, 'guess':8, 'bd':[1, 20]},
        'veq':{'vary':True,'guess':8.0, 'bd':[7., 9.]},
        # 'veq':{'vary':False,'guess':7.8, 'bd':[2, 30]},
        'cos_istar':{'vary':True,'guess':0.01, 'bd':[-1., 1.]},

        'lat__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':0, 'bd':[-90, 90]},
        'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1' : {'vary':True, 'guess':2458330.59051, 'bd':[2458330.29051 - 1, 2458330.29051 + 1]},
        'ang__ISESPRESSO_VSmock_vis_ARspot1'     : {'vary':True, 'guess':25, 'bd':[1, 60]},
        'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   : {'vary':True, 'guess':0.3, 'bd':[0.001, 0.999]},

        'lambda_rad__pl'+zodiac_pl                   : {'vary':True, 'guess':0., 'bd':[-2*np.pi, 2*np.pi]}
        }

                                            
                                            


    #Fitting mode
    if gen_dic['star_name'] in ['HD189733','TOI3884','AU_Mic','AUMic','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        # glob_fit_dic['DiffProf']['fit_mode']='chi2' 
        glob_fit_dic['DiffProf']['fit_mode']='mcmc'
        # glob_fit_dic['DiffProf']['fit_mode']='ns' 

    #Fitting method - only if chi2 is used
    if gen_dic['star_name'] in ['HD189733','TOI3884','AU_Mic','AUMic','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        glob_fit_dic['DiffProf']['chi2_fitting_method']='bfgs' 

    #Printing fits results
    if gen_dic['star_name'] in ['HD189733','TOI3884','AU_Mic','AUMic','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        glob_fit_dic['DiffProf']['verbose']=True   #& False
    
    #Priors on variable properties
    if gen_dic['star_name'] == 'HD189733':
        glob_fit_dic['DiffProf']['priors']={
                    #Round 1 - wide guessing
                    # 'ctrst__ord0__IS__VS_'                  :{'mod':'uf','low':0.,'high':1.},  
                    # 'FWHM__ord0__IS__VS_'                   :{'mod':'uf','low':0,'high':100},
                    # 'lat__ISESPRESSO_VSvisit1_ARspot1'      :{'mod':'uf', 'low':-90., 'high':90.},
                    # 'Tc_ar__ISESPRESSO_VSvisit1_ARspot1'    :{'mod':'uf', 'low':2459457.589323 - 2., 'high':2459457.589323 +2.},
                    # 'ang__ISESPRESSO_VSvisit1_ARspot1'      :{'mod':'uf', 'low':0., 'high':80.},
                    # 'fctrst__ISESPRESSO_VSvisit1_ARspot1'   :{'mod':'uf', 'low':0, 'high':1},
                    # 'lambda_rad__plHD189733b'               :{'mod':'uf', 'low':-2*np.pi, 'high':2*np.pi},
                    
                    #Round 2 - more specific guessing based on previous round
                    'ctrst__ord0__IS__VS_'                  :{'mod':'uf','low':0.,'high':1.},  
                    'FWHM__ord0__IS__VS_'                   :{'mod':'uf','low':0,'high':100},
                    'lat__ISESPRESSO_VSvisit1_ARspot1'      :{'mod':'gauss', 'val':-37., 's_val':40.},
                    'Tc_ar__ISESPRESSO_VSvisit1_ARspot1'    :{'mod':'gauss', 'val':2459457.67, 's_val':0.5},
                    'ang__ISESPRESSO_VSvisit1_ARspot1'      :{'mod':'gauss', 'val':25., 's_val':9.},
                    'fctrst__ISESPRESSO_VSvisit1_ARspot1'   :{'mod':'gauss', 'val':0.6, 's_val':0.2},
                    'lambda_rad__plHD189733b'               :{'mod':'uf', 'low':-2*np.pi, 'high':2*np.pi},
                    }

    if gen_dic['star_name'] == 'AUMic':
        glob_fit_dic['DiffProf']['priors']={
                    'cont__IS__VS_'                           :{'mod':'uf','low':0.,'high':1e10},
                    'ctrst__ord0__IS__VS_'                    :{'mod':'uf','low':0.,'high':1.},  
                    'FWHM__ord0__IS__VS_'                     :{'mod':'uf','low':0,'high':30},
                    'veq'                                     :{'mod':'uf', 'low':1., 'high':100.},
                    # 'vsini'                                   :{'mod':'uf', 'low':0., 'high':100.},
                    'veq_spots'                               :{'mod':'uf', 'low':1., 'high':100.},
                    # 'alpha_rot'                               :{'mod':'uf', 'low':0., 'high':1.},
                    # 'alpha_rot_spots'                         :{'mod':'uf', 'low':0., 'high':1.},
                    # 'beta_rot'                                :{'mod':'uf', 'low':0., 'high':1.},
                    # 'beta_rot_spots'                          :{'mod':'uf', 'low':0., 'high':1.},
                    'cos_istar'                               :{'mod':'uf', 'low':-1., 'high':1.},
                    'lat__ISESPRESSO_VSmock_vis_ARspot1'      :{'mod':'uf', 'low':-90., 'high':90.},
                    'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1'    :{'mod':'uf', 'low':2458330.39051 - 2., 'high':2458330.39051 +2.},
                    'ang__ISESPRESSO_VSmock_vis_ARspot1'      :{'mod':'uf', 'low':0., 'high':45.},
                    'fctrst__ISESPRESSO_VSmock_vis_ARspot1'   :{'mod':'uf', 'low':0, 'high':1},
                    # 'lat__ISESPRESSO_VSmock_vis_ARfacula1'    :{'mod':'uf', 'low':-90., 'high':90.},
                    # 'Tc_ar__ISESPRESSO_VSmock_vis_ARfacula1'  :{'mod':'uf', 'low':2458330.39051 - 20., 'high':2458330.39051 +20.},
                    # 'ang__ISESPRESSO_VSmock_vis_ARfacula1'    :{'mod':'uf', 'low':0., 'high':45.},
                    # 'fctrst__ISESPRESSO_VSmock_vis_ARfacula1' :{'mod':'uf', 'low':1, 'high':10},
                    'lambda_rad__plAUMicb'                    :{'mod':'uf', 'low':-2*np.pi, 'high':2*np.pi},
                    }

    if gen_dic['star_name'] == 'AU_Mic':
        glob_fit_dic['DiffProf']['priors']={
                    'ctrst__ord0__IS__VS_'                    :{'mod':'uf','low':0.1,'high':1},  
                    'FWHM__ord0__IS__VS_'                     :{'mod':'uf','low':0,'high':25},
                    'veq'                                    :{'mod':'gauss', 'val':8.595, 's_val':0.20966},
                    'alpha_rot'                              :{'mod':'dgauss', 'val':0.034214996, 'low':0.00107471, 'high':0.001066154},

                    'cos_istar'                              :{'mod':'uf', 'low':-1., 'high':1.},

                    'lat__ISESPRESSO_VSvisit1_ARspot1'     :{'mod':'uf', 'low':-90., 'high':90.},
                    'Tc_ar__ISESPRESSO_VSvisit1_ARspot1' :{'mod':'uf', 'low':2458702.77 - 1., 'high':2458702.77 + 1.},
                    'ang__ISESPRESSO_VSvisit1_ARspot1'     :{'mod':'uf', 'low':0, 'high':89.},
                    'fctrst__ISESPRESSO_VSvisit1_ARspot1'   :{'mod':'uf', 'low':0, 'high':1},

                    'lat__ISESPRESSO_VSvisit1_ARspot2'     :{'mod':'uf', 'low':-90., 'high':90.},
                    'Tc_ar__ISESPRESSO_VSvisit1_ARspot2' :{'mod':'uf', 'low':2458702.77 - 1., 'high':2458702.77 + 1.},
                    'ang__ISESPRESSO_VSvisit1_ARspot2'     :{'mod':'uf', 'low':0, 'high':89.},
                    'fctrst__ISESPRESSO_VSvisit1_ARspot2'   :{'mod':'uf', 'low':0, 'high':1},

                    'rv_shift'   :{'mod':'uf', 'low':-10, 'high':10},

                    'lambda_rad__plAU_Mic_b'                   :{'mod':'uf', 'low':-2*np.pi, 'high':2*np.pi},
                    }

    #Zodiacs
    if gen_dic['star_name'] in ['temp']:
        glob_fit_dic['DiffProf']['priors']={
                    'ctrst__ord0__IS__VS_'                   :{'mod':'uf','low':0,'high':1},  
                    'FWHM__ord0__IS__VS_'                    :{'mod':'uf','low':0,'high':50},
                    'veq'                                    :{'mod':'uf', 'low':1., 'high':100.},
                    'veq_spots'                              :{'mod':'uf', 'low':1., 'high':100.},
                    'cos_istar'                              :{'mod':'uf', 'low':-1., 'high':1.},
                    'lat__ISESPRESSO_VSmock_vis_ARspot1'     :{'mod':'uf', 'low':-90., 'high':90.},
                    'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1'   :{'mod':'uf', 'low':2458330.39051 - 1.1, 'high':2458330.39051 +1.1},
                    'ang__ISESPRESSO_VSmock_vis_ARspot1'     :{'mod':'uf', 'low':0., 'high':70.},
                    'fctrst__ISESPRESSO_VSmock_vis_ARspot1'  :{'mod':'uf', 'low':0., 'high':1},
                    'lambda_rad__pl'+zodiac_pl               :{'mod':'uf', 'low':-2*np.pi, 'high':2*np.pi},
                    }       
    

    if gen_dic['star_name'] in ['Virgo','Taurus','Scorpio','Sagittarius','Gemini', 'Aries', 'Libra', 'Aquarius', 'Leo', 'Pisces']:
        glob_fit_dic['DiffProf']['priors']={
                    'ctrst__ord0__IS__VS_'                   :{'mod':'uf','low':0.1,'high':1},
                    'FWHM__ord0__IS__VS_'                    :{'mod':'uf','low':0,'high':25},
                    'veq'                                    :{'mod':'gauss', 'val':8.0, 's_val':0.2},
                    # 'veq_spots'                              :{'mod':'uf', 'low':1., 'high':100.},
                    'cos_istar'                              :{'mod':'uf', 'low':-1., 'high':1.},
                    'lat__ISESPRESSO_VSmock_vis_ARspot1'     :{'mod':'uf', 'low':-90., 'high':90.},
                    'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1'   :{'mod':'uf', 'low':2458330.39051 - 1.1, 'high':2458330.39051 +1.1},
                    'ang__ISESPRESSO_VSmock_vis_ARspot1'     :{'mod':'uf', 'low':0., 'high':45.},
                    'fctrst__ISESPRESSO_VSmock_vis_ARspot1'  :{'mod':'uf', 'low':0., 'high':1},
                    'lambda_rad__pl'+zodiac_pl               :{'mod':'uf', 'low':-2*np.pi, 'high':2*np.pi},
                    }   

    if gen_dic['star_name'] in ['Capricorn']:
        glob_fit_dic['DiffProf']['priors']={
                    'ctrst__ord0__IS__VS_'                   :{'mod':'uf','low':0.1,'high':1},
                    'FWHM__ord0__IS__VS_'                    :{'mod':'uf','low':0,'high':25},
                    'veq'                                    :{'mod':'gauss', 'val':20.0, 's_val':1.0},
                    # 'veq_spots'                              :{'mod':'uf', 'low':1., 'high':100.},
                    'cos_istar'                              :{'mod':'uf', 'low':-1., 'high':1.},
                    'lat__ISESPRESSO_VSmock_vis_ARspot1'     :{'mod':'uf', 'low':-90., 'high':90.},
                    'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1'   :{'mod':'uf', 'low':2458330.39051 - 1.1, 'high':2458330.39051 +1.1},
                    'ang__ISESPRESSO_VSmock_vis_ARspot1'     :{'mod':'uf', 'low':0., 'high':45.},
                    'fctrst__ISESPRESSO_VSmock_vis_ARspot1'  :{'mod':'uf', 'low':0., 'high':1},
                    'lambda_rad__pl'+zodiac_pl               :{'mod':'uf', 'low':-2*np.pi, 'high':2*np.pi},
                    }

    if gen_dic['star_name'] in ['Cancer']:
        glob_fit_dic['DiffProf']['priors']={
                    'ctrst__ord0__IS__VS_'                   :{'mod':'uf','low':0.1,'high':1},
                    'FWHM__ord0__IS__VS_'                    :{'mod':'uf','low':0,'high':25},
                    'veq'                                    :{'mod':'gauss', 'val':0.2, 's_val':0.1},
                    # 'veq_spots'                              :{'mod':'uf', 'low':1., 'high':100.},
                    'cos_istar'                              :{'mod':'uf', 'low':-1., 'high':1.},
                    'lat__ISESPRESSO_VSmock_vis_ARspot1'     :{'mod':'uf', 'low':-90., 'high':90.},
                    'Tc_ar__ISESPRESSO_VSmock_vis_ARspot1'   :{'mod':'uf', 'low':2458330.39051 - 1.1, 'high':2458330.39051 +1.1},
                    'ang__ISESPRESSO_VSmock_vis_ARspot1'     :{'mod':'uf', 'low':0., 'high':45.},
                    'fctrst__ISESPRESSO_VSmock_vis_ARspot1'  :{'mod':'uf', 'low':0., 'high':1},
                    'lambda_rad__pl'+zodiac_pl               :{'mod':'uf', 'low':-2*np.pi, 'high':2*np.pi},
                    } 

    if gen_dic['star_name'] =='TOI3884':
        glob_fit_dic['DiffProf']['priors']={
                    'ctrst__ord0__IS__VS_'                   :{'mod':'uf','low':0,'high':1},  
                    'FWHM__ord0__IS__VS_'                    :{'mod':'uf','low':0,'high':50},
                    'veq'                                    :{'mod':'uf', 'low':1., 'high':100.},
                    'veq_spots'                              :{'mod':'uf', 'low':1., 'high':100.},
                    'cos_istar'                              :{'mod':'uf', 'low':-1., 'high':1.},
                    'lat__ISMIKE_Red_VSmockvis_ARspot1'     :{'mod':'uf', 'low':-90., 'high':90.},
                    'Tc_ar__ISMIKE_Red_VSmockvis_ARspot1'   :{'mod':'uf', 'low':2459642.86314 - 20., 'high':2459642.86314 +20.},
                    'ang__ISMIKE_Red_VSmockvis_ARspot1'     :{'mod':'uf', 'low':0., 'high':90.},
                    'fctrst__ISMIKE_Red_VSmockvis_ARspot1'  :{'mod':'uf', 'low':0., 'high':1.},
                    'lambda_rad__plTOI3884_b'               :{'mod':'uf', 'low':-2*np.pi, 'high':2*np.pi},
                    }  



    #Derived properties
    if gen_dic['star_name'] in ['TOI3884','AU_Mic','AUMic','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        glob_fit_dic['DiffProf']['deriv_prop'] = {'lambda_deg':[]}#, 'Peq_veq_spots':{'Rstar':{'val':0.75, 's_val':0.1}}}
    
    #%%% MCMC / NS

    #Calculating/retrieving
    # glob_fit_dic['DiffProf']['run_mode']='use'    
    glob_fit_dic['DiffProf']['run_mode']='reuse'    

    #Re-using
    if gen_dic['star_name'] in ['TOI3884','AU_Mic','AUMic','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        # glob_fit_dic['DiffProf']['reuse']={}
        glob_fit_dic['DiffProf']['reuse']={
                    'paths':['/Users/samsonmercier/Desktop/Work/Master/2023-2024/ANTARESS_Backup/Zodiacs2.0/Raw/Cancer - slow star - done/ALL/Cancer_b_Saved_data/Joined_fits/DiffProf/mcmc/raw_chains_walk40_steps10000_Cancer_b.npz'],
                    'nburn':[3000]
                    }  
    #Re-starting
    if gen_dic['star_name'] in ['TOI3884','AU_Mic','AUMic','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        glob_fit_dic['DiffProf']['reboot']=''


    # MCMC specific
    # - Walkers
    if gen_dic['star_name'] in ['HD189733','TOI3884','AU_Mic','AUMic','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        glob_fit_dic['DiffProf']['sampler_set']={'nwalkers':40,'nsteps':10000,'nburn':3000}

    # - Complex priors        
         
    # - Walkers exclusion  
    glob_fit_dic['DiffProf']['exclu_walk']=True & False  
    if gen_dic['star_name'] in ['TOI3884','AUMic','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:   
        glob_fit_dic['DiffProf']['exclu_walk']=True #& False   
    

    # - Automatic exclusion of outlying chains
    glob_fit_dic['DiffProf']['exclu_walk_autom']=None  #  5.
    if gen_dic['star_name'] in ['TOI3884','AUMic','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:   
            glob_fit_dic['DiffProf']['exclu_walk_autom']= 5



    # NS specific
    
    # - Live points
    if gen_dic['star_name'] in ['TOI3884','AU_Mic','AUMic','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        glob_fit_dic['DiffProf']['sampler_set']={'nlive':400, 'bound_method':'multi', 'sample_method':'slice','dlogz':10.}


    #Derived errors         
    if gen_dic['star_name'] in ['HD189733','TOI3884','AU_Mic','AUMic','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']:
        glob_fit_dic['DiffProf']['out_err_mode']= 'HDI'
        glob_fit_dic['DiffProf']['HDI']='1s'


    #Derived lower/upper limits    

    
    #MCMC chains
    glob_fit_dic['DiffProf']['save_MCMC_chains']='png'   #png  
    if gen_dic['star_name'] in ['HD189733','TOI3884','AUMic','Capricorn','Cancer','Gemini','Sagittarius','Leo','Aquarius','Aries','Libra','Taurus','Scorpio','Virgo','Pisces']: 
        glob_fit_dic['DiffProf']['save_MCMC_chains']='png'
        glob_fit_dic['DiffProf']['save_chi2_chains']='png'


    # Corner plot
    glob_fit_dic['DiffProf']['corner_options']={
#            'bins_1D_par':[50,50,50,50],       #vsini, ip, lambda, b
        # 'bins_2D_par':[30,30,30,30,30,30,30,30,30], 
#            'range_par':[(0.,320.),(88.,90.),(86.,91.),(0.,0.13)],
        'plot_HDI':True , # & False,             

#            'bins_1D_par':[50,50,50,45,50,50,50],       #veq, ip, alpha, istar, lambda, psi, b 
##            'range_par':[(40.,300.),(87.8,89.2),(-0.6,0.5),(1.,14.),(84.8,89.),(86.8,89.1),(0.055,0.145)],     #low istar
##            'range_par':[(40.,300.),(87.8,89.2),(-0.6,0.5),(166.,179.),(84.8,89.),(90.3,92.),(0.055,0.145)],      #high istar
#            'plot_HDI':True,  

#            'bins_1D_par':[50,50,50,45,50,50],       #veq, ip, alpha, istar, lambda, psi    FIG PAPER
##            'range_par':[(40.,300.),(87.8,89.2),(-0.6,0.5),(1.,14.),(84.8,89.),(86.8,89.1)],     #low istar
#            'range_par':[(40.,300.),(87.8,89.2),(-0.6,0.5),(166.,179.),(84.8,89.),(90.3,92.)],      #high istar
#            'plot_HDI':True,
        'plot1s_1D':False,
        'plot_best':False,
            
            
#            'major_int':[0.2,50.],
#            'minor_int':[0.1,10.],
        'color_levels':['deepskyblue','lime'],
        # 'fontsize':15,
        'fontsize':10,
        # 'smooth2D':[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]*3, 
        # 'smooth1D':[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]*3 ,
#            'plot1s_1D':False
        }   
        
        
        
        
        
        
        
        
    
        
        
    ##################################################################################################       
    #%%% Module: joined intrinsic profiles fit  
    #    - fitting joined intrinsic stellar profiles from combined (unbinned) instruments and visits
    #    - use this module to fit the average stellar line profiles from planet-occulted regions
    #    - the module can also be used when active regions are present, as long as they remain fixed during a given visit (this requires theo_dic['precision'] = 'high'), otherwise use the 'fitting joined differential profiles' module.
    #      however beware that in this case constraints on the active regions come from the planet-occulted, active cells with a line profile different from the quiet star. 
    #      since we assume the same line shape for quiet and active cells, the active region is constrained by the brightness-weighted rv of active cells within the occulted region itself
    #      if the planet is small or the star a slow rotator, it is thus not possible to constrain the active region
    #      if the planet is large and the star is a fast rotator, the active region stills needs to have sufficient contrast or overlaps over a sufficient fraction of the planet-occulted region for the overall line position to differ from the un-active case 
    #    - use 'idx_in_fit' to choose which visits to fit (can be a single one)
    #    - the contrast and FWHM of the intrinsic stellar lines (before instrumental convolution) are fitted as polynomials of the chosen coordinate
    #      surface RVs are fitted using the reloaded RM model  
    #      beware that the disk-integrated and intrinsic stellar profile have the same continuum, but it is not necessarily unity as set in the analytical and theoretical models, whose continuum must thus let free to vary 
    #    - several options need to be controlled from within the function
    #    - use plot_dic['prop_Intr']='' to plot the properties of the derived profiles
    #      use plot_dic['Intrbin']='' to plot the derived profiles
    #      use gen_dic['loc_prof_est'] to visualize the derived profiles
    #    - to derive the stellar inclination from Rstar and Peq, use them as model parameters alongside cosistar, instead of veq  
    #      set priors on Rstar and Peq from the literature and a uniform prior on cosistar (=isotropic distribution), or more complex priors if relevant
    #      then istar can be directly derived from cosistar in post-processing (alongside veq and vsini), and will have been constrained by the independent priors on Peq, Rstar, and the data through the corresponding vsini   
    ##################################################################################################     
            
    #%%%% Activating 
    gen_dic['fit_IntrProf'] = False    
    
        
    #%%%% Multi-threading
    
    #%%%%% Allocated threads
    glob_fit_dic['IntrProf']['nthreads'] = int(0.8*cpu_count())
    
    
    #%%%%% Unthreaded operations
    #    - all operations are multi-threaded by default, but overheads of sharing data between threads may counterbalance the benefits of threading the model
    #    - select here which operations not to thread:
    # + 'emcee'
    # + 'prof_grid'
    glob_fit_dic['IntrProf']['unthreaded_op'] = []
    
    
    #%%%% Fitted data
    ANTARESS_fit_prof_settings('IntrProf',glob_fit_dic)


    #%%%%% Continuum range
    glob_fit_dic['IntrProf']['cont_range'] = {}


    #%%%% Line profile model         
        
    #%%%%% Transition wavelength
    glob_fit_dic['IntrProf']['line_trans']=None        
    
    
    #%%%%% Model type
    #    - local stellar lines are always calculated numerically using ANTARESS stellar grid
    #    - this field controls the type of elementary stellar lines (analytical, measured, or theoretical) used to tile the local stellar regions
    glob_fit_dic['IntrProf']['mode'] = 'ana' 
    
     
    #%%%%% Analytical profile
    #    - default: 'gauss' 
    glob_fit_dic['IntrProf']['model'] = {}

    
    #%%%%% Fixed/variable properties
    #    - structure is the same as glob_fit_dic['IntrProp']['mod_prop']
    #    - intrinsic properties define the lines before instrumental convolution, which can then be applied specifically to each instrument  
    glob_fit_dic['IntrProf']['mod_prop']={}
    
        
    #%%%%% Analytical profile coordinate
    #    - fit coordinate for the line properties of analytical profiles
    #    - see possibilities in gen_dic['fit_IntrProp']
    glob_fit_dic['IntrProf']['coord_fit']='r_proj'
    
    
    #%%%%% Analytical profile variation
    #    - fit line property as absolute ('abs') or modulated ('modul') polynomial        
    glob_fit_dic['IntrProf']['pol_mode']='abs'  


    #%%%%% PC noise model
    #    - indicate for each visit:
    # + the path to the PC matrix, already reduced to the PC requested to correct the visit in the PCA module
    #   beware that the correction will be applied only over the range of definition of the PC set in the PCA
    #   beware that one PC adds the number of fitted intrinsic profiles to the free parameters of the joint fit
    # + whether to account or not (idx_out = []) for the PCA fit in the calculation of the fit merit values, using all out exposures (idx_out = 'all') or a selection
    # + set noPC = True to account for the chi2 of the null hypothesis (no noise) on the out-of-transit data, without including PC to the RMR fit
    glob_fit_dic['IntrProf']['PC_model']={}  
    
    
    #%%%%% Optimization levels
    #     - see 'glob_fit_dic['DiffProf']' for details
    glob_fit_dic['IntrProf']['Opt_Lvl']=0    
                        
    
    #%%%% Fit settings 
    ANTARESS_fit_def_settings('IntrProf',glob_fit_dic,plot_dic)

    
        
    
    #Activating 
    gen_dic['fit_IntrProf'] = True   &  False

    #Optimization level 
    glob_fit_dic['IntrProf']['Opt_Lvl']=3 

    #Exposures to be fitted
    if gen_dic['star_name'] == 'AUMic':
     glob_fit_dic['IntrProf']['idx_in_fit'] = {'ESPRESSO':{'mock_vis':'all'}}

    if gen_dic['star_name'] == 'TOI3884':
        glob_fit_dic['IntrProf']['idx_in_fit'] = {'MIKE_Red':{'mockvis':np.arange(7, 23)}}
  
    if gen_dic['star_name'] == 'V1298tau':
     glob_fit_dic['IntrProf']['idx_in_fit'] = deepcopy(glob_fit_dic['IntrProp']['idx_in_fit'])

    #Trimming
    glob_fit_dic['IntrProf']['trim_range'] = deepcopy(data_dic['DI']['fit_prof']['trim_range'])   

    #Continuum range
    if gen_dic['star_name'] == 'AUMic':
        glob_fit_dic['IntrProf']['cont_range']={'ESPRESSO':{0:[[-150.0,-70.0],[70.0,150.0]]}}

    if gen_dic['star_name'] == 'TOI3884':
        glob_fit_dic['IntrProf']['cont_range']={'MIKE_Red':{0:[[-30.0,-20.0],[20.0,30.0]]}}

    #Spectral range(s) to be fitted            
    glob_fit_dic['IntrProf']['fit_range'] = deepcopy(data_dic['Intr']['fit_range'])
    
    #Model type

    #Analytical profile
    if gen_dic['star_name'] in ['AUMic','TOI3884'] :
        glob_fit_dic['IntrProf']['model']={'ESPRESSO':'gauss'}
    
    #Analytical profile coordinate
    if gen_dic['star_name'] in ['AUMic','V1298tau','TOI3884']:glob_fit_dic['IntrProf']['dim_fit']='r_proj'


    #Analytical profile variation
    if gen_dic['star_name'] in ['AUMic','V1298tau','TOI3884']:glob_fit_dic['IntrProf']['pol_mode']='modul'  

    
    #Fixed/variable properties   
    if gen_dic['star_name']=='AUMic':
        glob_fit_dic['IntrProf']['mod_prop']={
        'ctrst_ord0__IS__VS_':{'vary':True, 'guess':0.5, 'bd':[0.55, 0.8]},
        'FWHM_ord0__IS__VS_':{'vary':True, 'guess':10, 'bd':[7, 10]},
        'veq':{'vary':True,'guess':7, 'bd':[7, 8]},
                                            }

    if gen_dic['star_name']=='TOI3884':
        glob_fit_dic['IntrProf']['mod_prop']={
        'ctrst__ord0__IS__VS_':{'vary':False, 'guess':0.7, 'bd':[0.1, 1]},
        'FWHM__ord0__IS__VS_':{'vary':False, 'guess':8, 'bd':[1, 20]},
        'veq':{'vary':False,'guess':8.495, 'bd':[2, 30]},
        'cos_istar':{'vary':False,'guess':-0.906307787, 'bd':[-1., 1.]},

        'lat__ISMIKE_Red_VSmockvis_ARspot1'     : {'vary':False, 'guess':-90, 'bd':[-90, 90]},
        'Tc_ar__ISMIKE_Red_VSmockvis_ARspot1' : {'vary':False, 'guess':2459642.86314 + 2.4, 'bd':[2459642.86314 - 1, 2459642.86314 + 1]},
        'ang__ISMIKE_Red_VSmockvis_ARspot1'     : {'vary':False, 'guess':48.6, 'bd':[1, 60]},
        'fctrst__ISMIKE_Red_VSmockvis_ARspot1'   : {'vary':False, 'guess':0.41, 'bd':[0.001, 0.999]},

        'lambda_rad__plTOI3884_b'                   : {'vary':True, 'guess':0.7, 'bd':[-2*np.pi, 2*np.pi]}
                                            }

    #PC noise model
    
    #Fitting mode
    if gen_dic['star_name'] in ['AUMic','TOI3884']:
        # glob_fit_dic['IntrProf']['fit_mode']='chi2' 
        glob_fit_dic['IntrProf']['fit_mode']='mcmc' 


    #Printing fits results
    if gen_dic['star_name'] in ['AUMic','TOI3884']:
        glob_fit_dic['IntrProf']['verbose']=True   #& False
    
    #Priors on variable properties
    if gen_dic['star_name'] == 'AUMic':
        glob_fit_dic['IntrProf']['priors']={
                    'ctrst_ord0__IS__VS_':{'mod':'uf','low':0,'high':1},  
                    'FWHM_ord0__IS__VS_':{'mod':'uf','low':0,'high':100},
                    'veq':{'mod':'uf', 'low':0, 'high':100.},
                    }

    if gen_dic['star_name'] =='TOI3884':
        glob_fit_dic['IntrProf']['priors']={
                    'ctrst__ord0__IS__VS_'                   :{'mod':'uf','low':0,'high':1},  
                    'FWHM__ord0__IS__VS_'                    :{'mod':'uf','low':0,'high':50},
                    'veq'                                    :{'mod':'uf', 'low':1., 'high':100.},
                    'veq_spots'                              :{'mod':'uf', 'low':1., 'high':100.},
                    'cos_istar'                              :{'mod':'uf', 'low':-1., 'high':1.},
                    'lat__ISMIKE_Red_VSmockvis_ARspot1'     :{'mod':'uf', 'low':-90., 'high':90.},
                    'Tc_ar__ISMIKE_Red_VSmockvis_ARspot1'   :{'mod':'uf', 'low':2458330.39051 - 20., 'high':2458330.39051 +20.},
                    'ang__ISMIKE_Red_VSmockvis_ARspot1'     :{'mod':'uf', 'low':0., 'high':90.},
                    'fctrst__ISMIKE_Red_VSmockvis_ARspot1'  :{'mod':'uf', 'low':0., 'high':1.},
                    'lambda_rad__plTOI3884_b'               :{'mod':'uf', 'low':-2*np.pi, 'high':2*np.pi},
                    }  
    #Derived properties
    # glob_fit_dic['IntrProf']['deriv_prop'] = ['veq_from_Peq_Rstar','vsini','psi','om','b','ip','istar_deg_conv','fold_istar','lambda_deg','c0','CB_ms']
    # glob_fit_dic['IntrProf']['deriv_prop'] = ['vsini','lambda_deg']
    # glob_fit_dic['IntrProf']['deriv_prop'] = ['vsini','lambda_deg','ip']
    # glob_fit_dic['IntrProf']['deriv_prop'] = ['lambda_deg','istar_deg_conv','Peq_veq']
    #glob_fit_dic['IntrProf']['deriv_prop'] = ['veq_from_Peq_Rstar','vsini','lambda_deg','istar_deg_conv','fold_istar','psi']
    # glob_fit_dic['IntrProf']['deriv_prop'] = []
    if gen_dic['star_name'] in ['AUMic','TOI3884']:
        # glob_fit_dic['IntrProf']['deriv_prop'] = ['lambda_deg', 'Peq_veq']
        glob_fit_dic['IntrProf']['deriv_prop'] = ['']
    # glob_fit_dic['IntrProf']['deriv_prop'] = ['vsini','lambda_deg','Peq_vsini'] 
    # glob_fit_dic['IntrProf']['deriv_prop'] = ['istar_Peq_vsini'] 
    # glob_fit_dic['IntrProf']['deriv_prop'] = ['istar_Peq_vsini','psi_lambda'] 
    # glob_fit_dic['IntrProf']['deriv_prop'] = ['vsini','lambda_deg','istar_Peq','psi'] 
    # glob_fit_dic['IntrProf']['deriv_prop'] = ['vsini','lambda_deg','ip'] 
    # glob_fit_dic['IntrProf']['deriv_prop'] = ['vsini','lambda_deg','CF0_meas_conv'] 
    # glob_fit_dic['IntrProf']['deriv_prop'] = ['vsini','lambda_deg','CF0_DG_conv'] 
    # glob_fit_dic['IntrProf']['deriv_prop'] = ['vsini','lambda_deg','CF0_meas_add','b','ip'] 


    
    #Calculating/retrieving
    # glob_fit_dic['IntrProf']['run_mode']='use'    
    glob_fit_dic['IntrProf']['run_mode']='use'    

    #Re-using
    if gen_dic['star_name'] in ['AUMic','TOI3884']:
        glob_fit_dic['IntrProf']['reuse']={}
        # glob_fit_dic['IntrProf']['reuse']={
        #             'paths':['/Users/samsonmercier/Desktop/UNIGE/Fall_Semester_2023-2024/antaress/Ongoing/AUMicb_Saved_data/Joined_fits/IntrProf/mcmc/raw_chains_walk24_steps2000.npz'],
        #             'nburn':[500]
        #             }  
    #Re-starting
    if gen_dic['star_name'] in ['AUMic','TOI3884']:
        glob_fit_dic['IntrProf']['reboot']=''

    #Walkers
    if gen_dic['star_name'] in ['AUMic','TOI3884']:
        glob_fit_dic['IntrProf']['sampler_set']={'nwalkers':24,'nsteps':10000,'nburn':5000}

    #Complex priors        
         
    #Walkers exclusion  
    glob_fit_dic['IntrProf']['exclu_walk']=True & False  
    if gen_dic['star_name'] in ['AUMic','TOI3884']:   
        glob_fit_dic['IntrProf']['exclu_walk']=True & False   
    

    #Automatic exclusion of outlying chains
    glob_fit_dic['IntrProf']['exclu_walk_autom']=None  #  5.
    if gen_dic['star_name'] in ['AUMic','TOI3884']:   
            glob_fit_dic['IntrProf']['exclu_walk_autom']= 5


    #Derived errors         
    if gen_dic['star_name'] in ['AUMic','TOI3884']:  
        glob_fit_dic['IntrProf']['out_err_mode']= 'HDI'
        glob_fit_dic['IntrProf']['HDI']='1s'


    #Derived lower/upper limits    

    
    #MCMC chains
    glob_fit_dic['IntrProf']['save_MCMC_chains']='png'   #png  
    if gen_dic['star_name'] in ['AUMic','TOI3884']: 
        glob_fit_dic['IntrProf']['save_MCMC_chains']='png'


    #MCMC corner plot
    glob_fit_dic['IntrProf']['corner_options']={
#            'bins_1D_par':[50,50,50,50],       #vsini, ip, lambda, b
#            'bins_2D_par':[30,30,30,30], 
#            'range_par':[(0.,320.),(88.,90.),(86.,91.),(0.,0.13)],
        'plot_HDI':True , # & False,             

#            'bins_1D_par':[50,50,50,45,50,50,50],       #veq, ip, alpha, istar, lambda, psi, b 
##            'range_par':[(40.,300.),(87.8,89.2),(-0.6,0.5),(1.,14.),(84.8,89.),(86.8,89.1),(0.055,0.145)],     #low istar
##            'range_par':[(40.,300.),(87.8,89.2),(-0.6,0.5),(166.,179.),(84.8,89.),(90.3,92.),(0.055,0.145)],      #high istar
#            'plot_HDI':True,  

#            'bins_1D_par':[50,50,50,45,50,50],       #veq, ip, alpha, istar, lambda, psi    FIG PAPER
##            'range_par':[(40.,300.),(87.8,89.2),(-0.6,0.5),(1.,14.),(84.8,89.),(86.8,89.1)],     #low istar
#            'range_par':[(40.,300.),(87.8,89.2),(-0.6,0.5),(166.,179.),(84.8,89.),(90.3,92.)],      #high istar
#            'plot_HDI':True,
        'plot1s_1D':False,
        'plot_best':False,
            
            
#            'major_int':[0.2,50.],
#            'minor_int':[0.1,10.],
        'color_levels':['deepskyblue','lime'],
        # 'fontsize':15,
        'fontsize':10,
#            'smooth2D':[0.05,5.] 
#            'plot1s_1D':False
        }    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  
    
        
    ##################################################################################################       
    #%%% Module: planet-occulted profile estimates 
    #    - use the module to generate:
    # + local profiles that are then used to correct differential profiles from stellar contamination
    # + intrinsic profiles that are corrected from measured ones to assess the quality of the estimates 
    #    - the choice to use measured ('meas') or theoretical ('theo') stellar surface RVs to shift local profiles is set by data_dic['Intr']['align_mode']
    ##################################################################################################     
    
    #%%%% Activating
    #    - for original and binned exposures in each visit
    gen_dic['loc_prof_est'] = False
    gen_dic['loc_prof_est_bin'] =  False
    
    
    #%%%% Calculating/retrieving
    gen_dic['calc_loc_prof_est']=True  
    gen_dic['calc_loc_prof_est_bin']=True  
    
    
    #%%%% Profile type
    #    - reconstructing local ('Diff') or intrinsic ('Intr') profiles
    #    - local profiles cannot be reconstructed for spectral data converted into CCFs, as in-transit differential CCFs are not calculated
    data_dic['Intr']['plocc_prof_type']='Intr'   
    
    
    #%%%% Model definition
    
    #%%%%% Model type and options
    #    - used to define the estimates for the local stellar flux profiles
    #    - these options partly differ from those defining intrinsic profiles (see gen_dic['mock_data']) because local profiles are associated with observed exposures
    # + 'def_iord': reconstructed order
    # + 'def_range': define the range over which profiles are reconstructed
    #    - set 'corr_mode' to:
    # > 'DIbin': using the master-out
    # + option to select visits contributing to the binned profiles (leave empty to use considered visit)
    # + option to select exposures contributing to the binned profiles (leave empty to use all out-transit exposures)
    # + option to select the phase range of contributing exposures
    # > 'Intrbin': using binned intrinsic profiles series
    # + option to select visits contributing to the binned profiles (leave empty to use considered visit)
    # + the nearest binned profile along the binned dimension is used for a given exposure
    # + option to select exposures contributing to the binned profiles
    # + see possible bin dimensions in data_dic['Intr']['dim_bin']  
    # + see possible bin table definition in data_dic['Intr']['prop_bin']
    # > 'glob_mod': models derived from global fit to intrinsic profiles (default)
    # + 'mode' : 'ana' or 'theo'
    # + can be specific to the visit or common to all, depending on the fit
    # + line coordinate choice is retrieved automatically 
    # + indicate path to saved properties determining the line property variations in the processed dataset
    #   bulk system properties will be retrieved and used if fitted
    # + default options are used if left undefined
    # > 'indiv_mod': models fitted to each individual intrinsic profile in each visit
    # + 'mode' : 'ana' or 'theo'
    # + works only in exposures where the stellar line could be fitted after planet exclusion
    # > 'rec_prof':
    # + define each undefined pixel via a polynomial fit to defined pixels in complementary exposures
    #   or via a 2D interpolation ('linear' or 'cubic') over complementary exposures and a narrow spectral band (defined in band_pix_hw pixels on each side of undefined pixels)
    # + chose a dimension over which the fit/interpolation is performed         
    # + option to select exposures contributing to the fit/interpolation
    # > 'theo': use imported theoretical local intrinsic stellar profiles    
    data_dic['Intr']['opt_loc_prof_est']={'nthreads':int(0.8*cpu_count()),'corr_mode':'glob_mod','mode':'ana','def_range':[],'def_iord':0}
    
    
    #%%%% Plot settings
    
    #%%%%% 2D maps : theoretical intrinsic stellar profiles
    #    - for original and binned exposures
    #    - data to which the reconstruction was applied to is automatically used for this plot
    plot_dic['map_Intr_prof_est']=''   
    
    
    #%%%%% 2D maps : residuals from theoretical intrinsic stellar profiles
    #    - the map allows plotting the combined residuals from in-transit (intrinsic) and out-of-transit (differential) profiles
    #    - same format as 'map_Intr_prof_est'
    plot_dic['map_Intr_prof_res']=''   
    
   
        
        
        
        
    ##################################################################################################       
    #%%% Module: planet-occulted and active region profile estimates
    #    - use the module to generate:
    # + local profiles that are then used to correct differential profiles from stellar contamination
    # + intrinsic profiles that are corrected from measured ones to assess the quality of the estimates 
    #    - the choice to use measured ('meas') or theoretical ('theo') stellar surface RVs to shift local profiles is set by data_dic['Intr']['align_mode']
    ##################################################################################################     
    
    #%%%% Activating
    #    - for original and binned exposures in each visit
    gen_dic['diff_prof_est'] = False        
    gen_dic['diff_prof_est_bin']=False        

    
    #%%%% Calculating/retrieving
    gen_dic['calc_diff_prof_est'] = False        
    gen_dic['calc_diff_prof_est_bin']=False  
    
    
    #%%%% Model definition
    
    #%%%%% Model type and options
    #    - used to define the estimates for the differential profiles
    # + 'def_iord': reconstructed order
    # + 'def_range': define the range over which profiles are reconstructed
    #    - set 'corr_mode' to:
    # > 'DIbin': using the master-out
    # + option to select visits contributing to the binned profiles (leave empty to use considered visit)
    # + option to select exposures contributing to the binned profiles (leave empty to use all out-transit exposures)
    # + option to select the phase range of contributing exposures
    # > 'Diffbin': using binned differential profiles series
    # + option to select visits contributing to the binned profiles (leave empty to use considered visit)
    # + the nearest binned profile along the binned dimension is used for a given exposure
    # + option to select exposures contributing to the binned profiles
    # + see possible bin dimensions in data_dic['Intr']['dim_bin']  
    # + see possible bin table definition in data_dic['Intr']['prop_bin']
    # > 'glob_mod': models derived from global fit to differential profiles (default)
    # + 'mode' : 'ana' or 'theo'
    # + can be specific to the visit or common to all, depending on the fit
    # + line coordinate choice is retrieved automatically 
    # + indicate path to saved properties determining the line property variations in the processed dataset
    # + default options are used if left undefined
    # > 'indiv_mod': models fitted to each individual differential profile in each visit
    # + 'mode' : 'ana' or 'theo'
    # + works only in exposures where the stellar line could be fitted after planet exclusion
    # > 'rec_prof':
    # + define each undefined pixel via a polynomial fit to defined pixels in complementary exposures
    #   or via a 2D interpolation ('linear' or 'cubic') over complementary exposures and a narrow spectral band (defined in band_pix_hw pixels on each side of undefined pixels)
    # + chose a dimension over which the fit/interpolation is performed         
    # + option to select exposures contributing to the fit/interpolation
    # > 'theo': use imported theoretical local differential stellar profiles 
    gen_dic['diff_prof_est'] = True & False        
    gen_dic['calc_diff_prof_est'] = True & False        

    data_dic['Diff']['opt_diff_prof_est']={'nthreads':int(0.8*cpu_count()),'corr_mode':'glob_mod','mode':'ana','def_range':[],'def_iord':0}
    
    if gen_dic['star_name']=='AUMic':
        data_dic['Diff']['opt_diff_prof_est'].update({'DiffProf_prop_path':{
                                                                'ESPRESSO':{
                                                                    'mock_vis':'/Users/samsonmercier/Desktop/Work/Master/2023-2024/antaress/Ongoing/AUMic/AUMicb_Saved_data/Joined_fits/DiffProf/mcmc/Fit_results'
                                                                            }
                                                                        }
                                                    })

    #%%%% Plot settings
    
    #%%%%% 2D maps : "clean", theoretical planet-occulted and active region profiles
    #    - for original and binned exposures
    #    - planet-occulted profiles retrieved in the case where active regions were not included in the model
    plot_dic['map_Diff_prof_clean_ar_est']='png'
    plot_dic['map_Diff_prof_clean_pl_est']='png'   


    #%%%%% 2D maps : "un-clean", theoretical planet-occulted and active region profiles
    #    - for original and binned exposures
    #    - planet-occulted profiles retrieved in the case where active regions were not included in the model
    #    - computing both "clean" and "unclean" versions of these maps can help identify if planets occulted active regions during the transit or not
    plot_dic['map_Diff_prof_unclean_ar_est']='png'
    plot_dic['map_Diff_prof_unclean_pl_est']='png'   

    
    #%%%%% 2D maps : residuals theoretical planet-occulted and active region profiles (for "clean" and/or "unclean" profiles)
    #    - same format as 'map_Diff_prof_pl_est'
    plot_dic['map_Diff_prof_clean_ar_res']='png'
    plot_dic['map_Diff_prof_clean_pl_res']='png'
    plot_dic['map_Diff_prof_unclean_ar_res']='png'
    plot_dic['map_Diff_prof_unclean_pl_res']='png'    






    ##################################################################################################       
    #%%% Module: correcting differential profile series from stellar contamination
    #    - use the module to:
    # + call previously computed differential profile estimates and the necessary corrections and use them
    # + to perform the correction.
    ##################################################################################################     
    
    #%%%% Activating
    #    - for original exposures in each visit
    gen_dic['corr_diff'] = False        

    
    #%%%% Calculating/retrieving
    gen_dic['calc_corr_diff'] = False        
    
    
    #%%%% Model definition
    
    #%%%%% Model type and options
    #    - used to define the estimates for the differential profiles
    # + 'def_iord': reconstructed order
    # + 'def_range': define the range over which profiles are reconstructed
    gen_dic['corr_diff'] = True & False        
    gen_dic['calc_corr_diff'] = True & False        

    data_dic['Diff']['corr_diff_dict']={'mode':'ana','def_range':[],'def_iord':0}
    if gen_dic['star_name']=='AUMic':
        data_dic['Diff']['corr_diff_dict'].update({'DiffProf_prop_path':{
                                                                'ESPRESSO':{
                                                                    'mock_vis':'/Users/samsonmercier/Desktop/Work/Master/2023-2024/antaress/Ongoing/AUMic/AUMicb_Saved_data/Joined_fits/DiffProf/mcmc/Fit_results'
                                                                            }
                                                                        }
                                                    })
    
    #%%%% Plot settings  
        
    #%%%%% 2D maps : differential profiles corrected for the impact of active regions
    plot_dic['map_Diff_corr_ar']='png'    






    ##################################################################################################       
    #%%% Module: Building best-fit differential profile series from fit results
    #    - use the module to:
    # + call previously computed differential profile estimates to build the time series of best fit
    # + differential profiles.
    ##################################################################################################     
    
    #%%%% Activating
    #    - for original exposures in each visit
    gen_dic['eval_bestfit'] = False        

    
    #%%%% Calculating/retrieving
    gen_dic['calc_eval_bestfit'] = False        
    
    
    #%%%% Model definition
    
    #%%%%% Model type and options
    #    - used to define the estimates for the differential profiles
    # + 'def_iord': reconstructed order
    # + 'def_range': define the range over which profiles are reconstructed
    gen_dic['eval_bestfit'] = True & False        
    gen_dic['calc_eval_bestfit'] = True & False        

    data_dic['Diff']['eval_bestfit_dict']={'mode':'ana','def_range':[],'def_iord':0}
    if gen_dic['star_name']=='AUMic':
        data_dic['Diff']['eval_bestfit_dict'].update({'DiffProf_prop_path':{
                                                                'ESPRESSO':{
                                                                    'mock_vis':'/Users/samsonmercier/Desktop/Work/Master/2023-2024/antaress/Ongoing/AUMic/AUMicb_Saved_data/Joined_fits/DiffProf/mcmc/Fit_results'
                                                                            }
                                                                        }
                                                    })
    
    #%%%% Plot settings  
    
    #%%%%% Plot best-fit 2D residual map
    plot_dic['map_BF_Diff_prof']='png'   
    
    
    #%%%%% 2D maps : Plot residuals from best-fit 2D residual map
    plot_dic['map_BF_Diff_prof_re']='png'   








        
        
        
        
        
    
    ##################################################################################################
    #%% Atmospheric profiles
    ##################################################################################################  

    ##################################################################################################
    #%%% General settings
    ##################################################################################################  

    #%%%% Orbital oversampling     
    #    - oversampling value for the planet radial orbital velocity, in orbital phase 
    data_dic['Atm']['dph_osamp_RVpl']=0.001


    #%%%% Mask for atmospheric spectra
    #    - relevant for input spectra only
    #    - same mask format as gen_dic['CCF_mask']
    #    - the mask will be used in two ways:
    # + to exclude spectral ranges contaminated by the planet, in all steps defined via data_dic['Atm']['no_plrange']
    #   this can be useful for stellar and RM study, to remove planetary contamination
    # + to compute atmospheric CCFs, if requested
    #   beware in that case of the definition of the mask weights
    #    - the mask can be reduced to a single line
    #    - can be defined for the purpose of the plots (set to None to prevent upload)
    data_dic['Atm']['CCF_mask'] = None
    
    
    #%%%% Exclusion of atmospheric signals
        
    #%%%%% Excluded range
    #    - range of the planetary signal, in the planet rest frame, in km/s
    data_dic['Atm']['plrange']=[-20.,20.]


    #%%%%% Excluded steps
    #    - exclude range of planetary signal
    #    - user can select the modules, and the exposures, to which planet exclusion is applied to
    #    - define below operations from which planetary signal should be excluded
    #      leave empty for no exclusion to be applied
    #    - operations :
    # + DI_Mast: for the calculation of disk-integrated masters        
    # + DI_prof : for the definition of model continuum and fitted ranges in DI profile fits, and corrections of DI profiles 
    # + Diff_prof: for the definition of errors on differential CCFs
    # + PCA_corr: for the PCA correction of differential data
    # + Intr: for the definition of model continuum and fitted ranges in Intr profile fits, and the continuum of Intr profiles
    #    - planetary ranges can be excluded even if calc_pl_atm = False and no atmospheric signal is extracted
    data_dic['Atm']['no_plrange']=[]    


    #%%%%% Excluded exposures
    #    - indexes of exposures from which planetary signal should be excluded, for each instrument/visit
    #    - indexes are relative to the global table in each visit
    #    - allows excluding signal from out-of-transit exposures in case of planetary emission signal
    #    - if undefined, set automatically to in-transit exposures
    data_dic['Atm']['iexp_no_plrange']={}
    
   
    

    ##################################################################################################
    #%%% Module: atmospheric signals extraction
    #    - atmospheric profiles are corrected for quiet stellar profiles built from the 'planet-occulted profile estimates' modules, or when relevant for quiet+active stellar profiles built from the 'planet-occulted and active region profile estimates' module.
    ##################################################################################################  

    #%%%% Activating
    gen_dic['pl_atm'] = False


    #%%%% Calculating/retrieving
    gen_dic['calc_pl_atm'] = True 


    #%%%% Extracted signal 
    #    - 'Emission': emission signal 
    #      'Absorption': in-transit absorption signal
    #    - signals are defined in the star rest frame
    data_dic['Atm']['pl_atm_sign']='Absorption'        


    #%%%%% Continuum range
    #    - common to all profiles, ie that they it be large enough to cover the full range of orbital RVs 
    data_dic['Atm']['cont_range']={}


    #%%%% Presence of active regions
    data_dic['Atm']['ar_model']=''


    #%%%% Plots

    #%%%%% 2D maps of atmospheric profiles
    #    - in the star or planet rest frame
    plot_dic['map_Atm_prof']=''  


    #%%%%% Individual atmospheric profiles
    #    - in the star or planet rest frame
    plot_dic['Atm_prof']=''  
    
    
    #%%%%% Residuals from atmospheric profiles
    #    - choose within the routine whether to plot fit to individual or to global profiles
    plot_dic['Atm_prof_res']=''  
    
    



    ##################################################################################################
    #%%% Module: CCF conversion for atmospheric spectra
    #    - every operation afterwards will be performed on those CCFs
    ##################################################################################################  

    ANTARESS_CCF_settings('Atm',gen_dic)
    

    #%%%% Weights
    #    - use mask weights or not in the calculation of the CCFs
    data_dic['Atm']['use_maskW'] = True    






    ##################################################################################################
    #%%% Module: atmospheric profiles alignment     
    ##################################################################################################

    #%%%% Activating
    gen_dic['align_Atm'] = False
 
    #%%%% Calculating/retrieving 
    gen_dic['calc_align_Atm'] = True 

    #%%%% Reference planet
    #    - profiles will be aligned in the rest frame of this planet
    data_dic['Atm']['ref_pl_align'] = ''
    

    #%%%% Plots: aligned atmospheric profiles
    #    - plotting all aligned profiles together in planet rest frame        
    plot_dic['all_atm_data']=''   
  




    ##################################################################################################
    #%%% Module: 2D->1D conversion for atmospheric spectra
    ##################################################################################################

    ANTARESS_2D_1D_settings('Atm',data_dic,gen_dic,plot_dic)
    

    ##################################################################################################
    #%%% Module: atmospheric profiles binning
    #    - for analysis purpose (original profiles are not replaced)
    #    - this module can be used to boost the SNR by combining exposures, or to calculate a global master, in a given visit or in several visits
    ##################################################################################################

    #%%%% Activating
    gen_dic['Atmbin'] = False
    gen_dic['Atmbinmultivis'] = False


    #%%%% Calculating/retrieving
    gen_dic['calc_Atmbin']=True 
    gen_dic['calc_Atmbinmultivis'] = True  


    #%%%% Visits to be binned
    #    - for the 'Atmbinmultivis' option
    #    - leave empty to use all visits
    data_dic['Atm']['vis_in_bin']={}      
    

    #%%%% Exposures to be binned
    #    - indexes are relative to the in-transit table (for absorption signals) and to the global table (for emission signals) in each visit
    #    - leave empty to use all exposures 
    data_dic['Atm']['idx_in_bin']={}   


    #%%%% Binning dimension
    #    - possibilities :
    # + 'phase': profiles are binned over phase
    #    - beware to use the alignement module if binned profiles should be calculated in the planet rest frame
    data_dic['Atm']['dim_bin']='phase'        
    
    
    #%%%% Bin definition
    #    - see data_dic['DI']['prop_bin']
    data_dic['Atm']['prop_bin']={}
        
        
    #%%%% Plot settings        
        
    #%%%%% 2D maps of binned profiles
    plot_dic['map_Atmbin']=''   
    
    #%%%%% Individual binned profiles
    plot_dic['Atmbin']=''    

    #%%%%% Residuals from binned profiles
    plot_dic['Atmbin_res']=''  






    
    ##################################################################################################
    #%%% Module: atmospheric CCF masks
    ##################################################################################################
    
    #%%%% Activating
    gen_dic['def_Atmmasks'] = False


    #%%%% Multi-threading
    data_dic['Atm']['mask']['nthreads'] = int(0.8*cpu_count())

    #%%%% Plot settings 

    
    






    ##################################################################################################
    #%%% Module: atmospheric profiles analysis
    #    - can be applied to:
    # + 'fit_Atm': profiles in the star rest frame, original exposures, for all formats
    # + 'fit_Atm_1D': profiles in the star or surface (if aligned) rest frame, original exposures, converted from 2D->1D 
    # + 'fit_Atmbin' : profiles in the star or surface (if aligned) rest frame, binned exposures, all formats
    # + 'fit_Atmbinmultivis' : profiles in the surface rest frame, binned exposures, all formats
    ##################################################################################################
    
    #%%%% Activating
    gen_dic['fit_Atm'] = False
    gen_dic['fit_Atm_1D'] = False
    gen_dic['fit_Atmbin']= False
    gen_dic['fit_Atmbinmultivis']= False
    
    
    #%%%% Calculating/Retrieving
    gen_dic['calc_fit_Atm']=True  
    gen_dic['calc_fit_Atm_1D']=True 
    gen_dic['calc_fit_Atmbin']=True 
    gen_dic['calc_fit_Atmbinmultivis']=True 
    
    
    #%%%% Fitted data
    ANTARESS_fit_prof_settings('Atm',data_dic)


    #%%%% Direct measurements
    #    - format: {prop_name:{options}}
    #    - possibilities:
    # + integrated signal: 'int_sign' : {'rv_range':[[rv1,rv2],[rv3,rv4]] exact ranges over which the integral is performed, in the planet rest frame, in km/s}
    data_dic['Atm']['meas_prop']={}
    
    
    #%%%% Line profile model
    
    #%%%%% Transition wavelength
    #    - in the star rest frame
    #    - used to center the line analytical model
    #    - only relevant in spectral mode
    #    - do not use if the spectral fit is performed on more than a single line
    data_dic['Atm']['line_trans']=None
    
    
    #%%%%% Model type
    #    - specific to each instrument
    #    - options: ??
    #    - it is possible to fix the value, for each instrument and visit, of given parameters of the fit model
    #      if a field is given in 'mod_prop', the corresponding field in the model will be fixed to the given value    
    data_dic['Atm']['model']={}
    
    
    #%%%%% Fixed/variable properties
    #    - same as data_dic['DI']['mod_prop']
    data_dic['Atm']['mod_prop']={}
    
    
    #%%%%% Best model table
    #    - resolution (dx) and range (min_x,max_x) of final model used for post-processsing of fit results and plots
    #    - in rv space and km/s for analytical profiles (profiles in wavelength space are modelled in RV space and then converted), in space of origin for measured profiles, in wavelength space for theoretical profiles 
    #    - specific to the instrument
    data_dic['Atm']['best_mod_tab']={}
    
    
    #%%%% Fit settings 
    ANTARESS_fit_def_settings('Atm',data_dic,plot_dic)
    
    
    #%%%% Plot settings

    #%%%%% Derived properties
    #    - from original or binned data
    plot_dic['prop_Atm']=''  




    ##################################################################################################       
    #%%% Module: atmospheric signal properties fit
    #    - fitting single atmospheric property with a common model for all instruments/visits 
    #    - with properties derived from individual atmospheric profiles
    #    - this module can be used to estimate the analytical laws describing the atmospheric line properties
    #      the final fit should be performed over the joined atmospheric line profiles with gen_dic['fit_AtmProf']
    ##################################################################################################       
    
    #%%%% Activating 
    gen_dic['fit_AtmProp'] = False
    
    
    #%%%% Multi-threading
    glob_fit_dic['AtmProp']['nthreads'] = int(0.8*cpu_count())

    
    #%%%%% Unthreaded operations
    #    - all operations are multi-threaded by default, but overheads of sharing data between threads may counterbalance the benefits of threading the model
    #    - select here which operations not to thread:
    # + 'emcee'
    glob_fit_dic['AtmProp']['unthreaded_op'] = []  
    
    
    #%%%% Fitted data
    
    #%%%%% Exposures to be fitted
    #    - same as in glob_fit_dic['IntrProp']
    #    - indexes are relative to in-transit tables for absorption signal, or global tables for emission signals
    glob_fit_dic['AtmProp']['idx_in_fit'] = {}
 

    #%%%% Fitted properties   
    
    #%%%%% Properties and model
    #    - same as in glob_fit_dic['IntrProp']
    glob_fit_dic['AtmProp']['mod_prop']={}

    
    #%%%%% Coordinate
    #    - define the coordinate as a function of which line shape properties are defined:
    # +'phase' : orbital phase       
    glob_fit_dic['AtmProp']['coord_fit']='phase'
      
    
    #%%%%% Variation
    #    - same as in glob_fit_dic['IntrProp']
    glob_fit_dic['AtmProp']['pol_mode']='abs'     
    

    #%%%% Fit settings
    ANTARESS_fit_def_settings('AtmProp',glob_fit_dic,plot_dic)
    






    ##################################################################################################       
    #%%% Module: joined atmospheric profiles fit
    # - fitting atmospheric intrinsic stellar profiles from combined (unbinned) instruments and visits 
    # - use 'idx_in_fit' to choose which visits to fit (can be a single one)
    # - the contrast, FWHM, and RVs of the atmospheric lines are fitted as polynomials of the chosen coordinate 
    # - use plot_dic['prop_Atm']='' to plot the properties of the derived profiles
    # - to derive the stellar inclination from Rstar and Peq, use them as model parameters alongside cosistar, instead of veq  
    ##################################################################################################     
            
    #%%%% Activating 
    gen_dic['fit_AtmProf'] = False    
    
        
    #%%%% Multi-threading
    glob_fit_dic['AtmProf']['nthreads'] = int(0.8*cpu_count())
    
    
    #%%%% Fitted data
    ANTARESS_fit_prof_settings('AtmProf',glob_fit_dic)


    #%%%%% Continuum range
    glob_fit_dic['AtmProf']['cont_range'] = {}

    
    #%%%% Line profile model         
        
    #%%%%% Transition wavelength
    glob_fit_dic['AtmProf']['line_trans']=None        
    
    
    #%%%%% Model type
    glob_fit_dic['AtmProf']['mode'] = 'ana' 
    
     
    #%%%%% Analytical profile
    #    - default: 'gauss' 
    glob_fit_dic['AtmProf']['model'] = {}

    
    #%%%%% Fixed/variable properties
    #    - same as in glob_fit_dic['IntrProf'] 
    glob_fit_dic['AtmProf']['mod_prop']={}
    
                    
    #%%%%% Analytical profile coordinate
    #    - fit coordinate for the line properties of analytical profiles
    #    - see possibilities in gen_dic['fit_AtmProp']
    glob_fit_dic['AtmProf']['coord_fit']='phase'
    
    
    #%%%%% Analytical profile variation
    #    - same as in glob_fit_dic['IntrProf']      
    glob_fit_dic['AtmProf']['pol_mode']='abs'  

    #%%%% Fit settings 
    ANTARESS_fit_def_settings('AtmProf',glob_fit_dic,plot_dic)
        
    return None



##################################################################################################    
#%% Conversion settings
##################################################################################################  

#%%% 2D -> 1D conversion
def ANTARESS_2D_1D_settings(data_type,local_dic,gen_dic,plot_dic):
    r"""**ANTARESS default settings: 2D -> 1D conversion modules**
    
    Initializes ANTARESS configuration settings with default values for conversion of 2D spectra into 1D spectra. 
    Converted 1D spectra replace 2D spectra in the workflow process, ie that every operation afterwards will be performed on those profiles.
    Prior to conversion, spectra are normalized in all orders to a flat, common continuum.
    
    Args:
        TBD
    
    Returns:
        None
    
    """  
    
    #%%%% Activating
    gen_dic['spec_1D_'+data_type] = False
    
    
    #%%%% Calculating/retrieving 
    gen_dic['calc_spec_1D_'+data_type]=True  
    
    
    #%%%% Multi-threading
    gen_dic['nthreads_spec_1D_'+data_type]= int(0.8*cpu_count())
    
    
    #%%%% 1D spectral table
    #    - specific to each instrument
    #    - tables are uniformely spaced in ln(w) (with d[ln(w)] = dw/w)
    #      start and end values given in A  
    local_dic[data_type]['spec_1D_prop']={}   
    
    
    #%%%% Plot settings
    
    #%%%%% 2D maps
    plot_dic['map_'+data_type+'_1D']=''   
    
    
    #%%%%% Individual spectra
    plot_dic['sp_'+data_type+'_1D']=''
    
    
    #%%%%% Residuals from model     
    plot_dic['sp_'+data_type+'_1D_res']=''     
    
    return None

#%%% CCF conversion
def ANTARESS_CCF_settings(data_type,gen_dic):
    r"""**ANTARESS default settings: CCF conversion modules**
    
    Initializes ANTARESS configuration settings with default values for conversion of spectra into CCFs. 
    Converted CCF replace spectra in the workflow process, ie that every operation afterwards will be performed on those profiles.
    
    Args:
        TBD
    
    Returns:
        None
    
    """  
    
    #%%%% Activating
    gen_dic[data_type+'_CCF'] = False
    
    
    #%%%% Calculating/retrieving 
    gen_dic['calc_'+data_type+'_CCF']=True      
        
    return None


##################################################################################################    
#%% Analysis settings
##################################################################################################  

#%%% Fitted profiles
def ANTARESS_fit_prof_settings(data_type,local_dic):
    r"""**ANTARESS default settings: fitted profiles**
    
    Initializes ANTARESS configuration settings with default values for fitted profiles in analysis modules. 
    
    Args:
        TBD
    
    Returns:
        None
    
    """  

    #%%% Constant data errors
    #    - ESPRESSO/HARPS(N) pipeline fits are performed with constant errors, to mitigate the impact of large residuals from a gaussian fit in the line core or wing (as a gaussian can not be the correct physical model for the lines)
    #      by default, ANTARESS fits are performed using the propagated error table 
    #      this option allows setting errors on all pixels of a given profile to the mean error over the profile continuum
    #    - if errors on disk-integrated profiles were not provided with input table, they were set to sqrt(g_err*F) upon initialization and propagated afterwards
    #    - constant errors can further be scaled locally here through local_dic[data_type]['sc_err']
    local_dic[data_type]['cst_err']=False    


    #%%% Scaled data errors
    #    - local scaling of data errors
    #    - you can scale by sqrt(reduced chi2 of original fit) to ensure a reduced chi2 unity
    local_dic[data_type]['sc_err']={}    


    #%%% Trimming 
    #    - format is inst > [x1,x2]
    #      with x in RV space (km/s) if data is in CCF mode, and in wavelength space (A) if data is in spectral mode
    #           x defined in the solar barycentric rest frame for disk-integrated profiles (automatically shifted to the star rest frame if relevant), and in the star rest frame otherwise
    #    - profiles are trimmed within [x1,x2] before being used for the fit
    #    - this is mostly relevant for data in spectral mode, to avoid manipulating large arrays
    #    - leave empty to use the full profile range 
    local_dic[data_type]['trim_range']={}


    #%%% Order to be fitted
    #    - relevant for 2D spectra only
    local_dic[data_type]['fit_order']={}   
    
    
    #%%% Spectral range(s) to be fitted
    #    - format: {inst : { vis : { [ [x1,x2] , [x3,x4] , [x5,x6] , ... ] } } } 
    #      with x in RV space (km/s) if data is in CCF mode or in spectral mode and fitted with an analytical model on a single line, and in wavelength space (A) otherwise
    #           x defined in the solar barycentric rest frame for disk-integrated profiles (automatically shifted to the star rest frame if relevant), and in the star rest frame otherwise
    #    - the option to define multiple, non-consecutive ranges allows excluding from the fit features that are not captured by the model (eg sidelobe patterns of M dwarf CCF, not reproduced by a gaussian model)
    #    - leave empty to fit over the entire range of definition
    local_dic[data_type]['fit_range']={}
        
    #Joint analysis settings
    if 'Prof' in data_type:
        
        #%%% Exposures to be fitted
        #    - indexes are relative to
        # + in-transit tables for Differential, Intrinsic, and Absorption profiles
        # + global tables for Disk-integrated and Emission profiles
        #    - define instruments and visits to be fitted (they will not be fitted if not used as keys, or if set to []), set their value to 'all' for all in-transit exposures to be fitted
        #    - add '_bin' at the end of a visit name for its binned exposures to be fitted instead of the original ones (must have been calculated with the binning module)
        #      all other mentions of the visit (eg in parameter names) can still refer to the original visit name
        local_dic[data_type]['idx_in_fit']={}    

    return None    


#%%% Fit definition
def ANTARESS_fit_def_settings(data_type,local_dic,plot_dic):
    r"""**ANTARESS default settings: fit properties**
    
    Initializes ANTARESS configuration settings with default values for fit properties in analysis modules. 
    
    Args:
        TBD
    
    Returns:
        None
    
    """  

    ################################################################################################## 
    #%%% General settings     
    ################################################################################################## 
    
    #%%%% Fitting mode 
    #    - options :
    # + 'chi2' : least-square minimization
    # + 'mcmc' : Markov chain Monte Carlo exploration
    # + 'ns' : nested sampling exploration
    # + 'fixed' : forward model
    local_dic[data_type]['fit_mode']='chi2'  
    
    
    #%%%% Printing fits results
    local_dic[data_type]['verbose']= False

    
    #%%%% Priors on variable properties
    #    - format : { p : {prior_mode: X, prior_val: Y} }
    #      where p is specific to the model selected, and 'prior_mode' defines the prior as
    #    - otherwise priors can be set to :
    # + uniform ('uf') : define lower ('low') and upper ('high') boundaries
    # + gaussian ('gauss') : define median ('val') and std ('s_val')
    # + asymetrical gaussian ('dgauss') : define median ('val'), and lower ('s_val_low') / upper ('s_val_high') std
    #    - chi2 fit can only use uniform priors
    #    - if left undefined, default uniform priors are used
    local_dic[data_type]['priors']={}
    
    
    #%%%% Derived properties
    #    - options for data_type =='IntrProp' :
    # + 'cosistar_fold' : folds cos(istar) within -1 : 1 (not required if constrained with prior)
    # + 'veq_from_Peq_Rstar' : converts 'Rstar' and 'Peq' into 'veq'
    #                          'Peq' must be a fit parameter; 'Rstar' can be a fit parameter or a user-provided measurement
    # + 'vsini' : converts 'veq' into veq*sin(istar) using fitted or fixed 'istar'
    # + 'istar_deg_conv' : replaces cos(istar) by istar[deg]
    # + 'fold_istar' : folds istar[deg] around 90 and returns the Northern (istar < 90, 'config' = 'North') or Southern (istar > 90, 'config' = 'South')) configurations.
    #                  this is relevant when
    #                       only sin(istar) is constrained and the stellar inclination remains degenerate between istar and 180-istar 
    #                       cos(istar) converges toward a mode well-defined and distinct from 0 (ie, istar = 90), because the MCMC converged toward this mode but we know the symmetrical mode around 0 is equally valid.
    # + 'istar_Peq' : derive the stellar inclination from the fitted 'vsini' and user-provided measurements of 'Rstar' and 'Peq'
    #                 warning: it is better to fit directly for 'Peq', 'cosistar', and 'Rstar'
    # + 'fold_Tc_ar' : folds the active region crossing time around a central Peq value that can be calculated in the following ways:
    #                       - If active region values for veq/Peq are fitted/specified, they take priority
    #                       - If veq/Peq/veq_spots/Peq_spots/veq_faculae/Peq_faculae is fit with an MCMC/NS, we use the corresponding chain
    #                       - If veq/Peq/veq_spots/Peq_spots/veq_faculae/Peq_faculae is fit with chi2 or fixed, we use the corresponding value
    #                       - If veq_spots/Peq_spots/veq_faculae/Peq_faculae is fixed but its values is different from the default, use the active region value
    #                       - If none of veq, Peq, veq_spots, Peq_spots, veq_faculae, Peq_faculae are fixed or fit, default to the value of Peq calculated from the systems configuration file.
    #                  warning: if faculae and spot values are both fitted (whether that is Peq or veq) the user must specify which one to use for the folding with deriv_prop['fold_Tc_ar']['reg_to_use']='spots'/'faculae'
    #                  warning: if veq/veq_spots/veq_faculae is used to perform the folding, the corresponding derived property option 'Peq_veq'/'Peq_veq_spots'/'Peq_veq_faculae' must be activated. 
    # + 'istar_Peq_vsini' : derive the stellar inclination from user-provided measurements of 'Rstar','Peq', and 'vsini'
    # + 'Peq_veq' : adds 'Peq' using the fitted 'veq' and a user-provided measurement of 'Rstar'
    # + 'Peq_veq_spots' : adds 'Peq_spots' using the fitted 'veq_spots' and a user-provided measurement of 'Rstar'
    # + 'Peq_veq_faculae' : adds 'Peq_faculae' using the fitted 'veq_faculae' and a user-provided measurement of 'Rstar'
    # + 'Peq_vsini' : adds 'Peq' using the fitted 'vsini' and user-provided measurements for 'Rstar' and 'istar' 
    # + 'psi' : adds 3D spin-orbit angle for all planets using the fitted 'lambda', and fitted or user-provided measurements for 'istar' and 'ip_plNAME'
    #           put 'North' and/or 'South' in  'config' to return the corresponding Psi configurations associated with istar (Northern configuration) and 180-istar (Southern configuration). This is only relevant if istar needed to be folded around 90 to manually produce the Northern or Southern configuration. 
    #           put 'combined' in 'config' to add the combined distribution from the Northern and Southern Psi PDFs, assumed to be equiprobable (make sure that the two distributions overlap sufficiently before combining them, otherwise they should be kept separate)
    #               in this case, 'fold_istar' must have been requested (whether to North or South does not matter, it is just for 'combined' to use separately the Northern and Southern configurations rather than the original full one)  
    # + 'psi_lambda' : adds 3D spin-orbit angle using user-provided measurements of 'lambda', and fitted or user-provided measurements for 'istar' and 'ip'
    #                  same settings as for 'psi' 
    # + 'lambda_deg' : converts lambda[rad] to lambda[deg]
    #                  lambda[deg] is folded over x+[-180;180], with x set by the subfield 'pl_name' if defined, or to the median of the chains by default
    #                  define x so that the peak of the PDF is well centered in the folded range
    # + 'i_mut' : adds mutual inclination between the orbital planes of two transiting planets, if relevant, using their fitted 'lambda'     
    # + 'b' : adds impact parameter, using fixed or fitted 'aRs' and 'ip'
    # + 'ip' : converts fitted orbital inclination from radian to degrees.    
    #    - user-provided measurements of a parameter 'par' are defined as subfields with format
    # par : {'val' : val, 's_val' : val} or par : {'val' : val, 's_val_low' : val, 's_val_high' : val} 
    #      units for par :
    # + stellar radius 'Rstar' : Rsun
    # + stellar inclination 'istar' : deg
    # + equatorial rotation period 'Peq' : days
    # + sky-projected rotational velocity 'vsini' : km/s
    # + sky-projected spin-orbit angle 'lambda_plNAME' : deg
    # + orbital inclination 'ip_plNAME' : deg
    local_dic[data_type]['deriv_prop']={}
    
    
    #%%%% Profile analysis settings
    if 'prop' not in data_type:
            
        #%%%%% Detection thresholds
        #    - define area and amplitude thresholds for detection of stellar line (in sigma)
        #    - for the amplitude, it might be more relevant to consider the actual SNR of the derived value (shown in plots)
        #    - requires 'true_amp' or 'amp' in 'deriv_prop'
        #    - if set to None, lines are considered as detected in all exposures
        local_dic[data_type]['thresh_area']=5.
        local_dic[data_type]['thresh_amp']=4.   
        
        
        #%%%%% Force detection flag 
        #    - set flag to True at relevant index for the CCFs to be considered detected, or false to force a non-detection
        #    - indices for each dataset are relative to:
        # + global indexes (from binned exposures if relevant) for disk-integrated stellar profiles, differential profiles, and atmospheric emission profiles
        # + in-transit indexes (from binned exposures if relevant) for intrinsic stellar profiles, and atmospheric absorption profiles
        #    - leave empty for automatic detection
        local_dic[data_type]['idx_force_det']={}
        local_dic[data_type]['idx_force_detbin']={} 
        local_dic[data_type]['idx_force_detbinmultivis']={} 

    ##################################################################################################         
    #%%% Chi2 settings
    ################################################################################################## 
    
    #%%%% Fitting method
    #    - fitting method used to perform the chi2 miniminzation with lmfit
    #    - options include leastsq, bfgs, newton, ... (string must be in the format supported by lmfit)
    local_dic[data_type]['chi2_fitting_method']='leastsq' 


    ##################################################################################################         
    #%%% MCMC and Nested-sampling settings
    ################################################################################################## 
    
    #%%%% Hessian matrix
    #    - string containing the location of a Fit_results.npz file containing a Hessian matrix.
    #      this Hessian matrix must have been computed from the same parameters are the ones used in the fit.
    #    - to use this option, we recommend users first run a fit with fit_mode set to chi2. The chi2 fit will automatically create and 
    #      store the Hessian matrix. An MCMC / NS  fit can subsequently be run with the path to the Hessian being set as the location of the chi2 fit results.
    #    - Evaluating the Hessian matrix prior to performing an MCMC or NS fit can be useful as it stores information on the local curvature of the target
    #      posterior distribution. This matrix can be used to have a more efficient sampling by initializing the starting location of the MCMC/NS with it. 
    local_dic[data_type]['use_hess'] = ''    


    #%%%% Monitor progress
    local_dic[data_type]['progress']= True

    
    #%%%% Run mode
    #    - set to
    # + 'use': runs the MCMC / NS fit 
    # + 'reuse' (with gen_dic['calc_fit_X']=True): load MCMC / NS results, allow changing nburn and/or error definitions without running the fit again
    local_dic[data_type]['run_mode']='use'    


    #%%%%% Runs to re-use
    #    - list of runs to reuse, when 'run_mode' = 'reuse'
    #    - leave empty to automatically retrieve the run available in the default directory or set the list of runs to retrieve
    # + for MCMC the runs must have been run with the same settings, but the burnin can be specified for each run :
    #   { 'paths' : ['path1/raw_chains_walkN_stepsM1_name.npz','path2/raw_chains_walkN_stepsM2_name.npz',..],
    #     'nburn' : [ n1, n2, ..]}
    # + for NS the runs must have been run with the same settings :
    #   { 'paths' : ['path1/raw_chains_liveL_name.npz','path2/raw_chains_liveL_name.npz',..] }
    local_dic[data_type]['reuse']={}


    #%%%%% Runs to re-start
    #    - indicate path to a 'raw_chains' file:
    # + for a MCMC fit : 'path1/raw_chains_walkN_stepsM1_name.npz'
    # + for a NS fit: 'path1/raw_chains_liveL_name.npz'
    #    - the fit will restart the same walkers from their last step, and run from the number of steps indicated in 'sampler_set'
    local_dic[data_type]['reboot']=''


    #%%%% Sampler settings
    #    - for a MCMC fit define:
    # + 'nwalkers' : number of walkers
    # + 'nsteps' : total number of steps
    # + 'nburn' : number of burn-in steps
    #
    #    - for a NS fit define:
    # + 'nlive' : number of live points. 
    #    Starting with 400 to 1000 is generally advisable.
    # + 'bound_method' : Prior bounding method. 
    #    If not specified it will default to 'auto' (see doc. in dynesty API to see what this specifically does).
    #    While the default bounding method is 'auto', 'multi' is better at dealing with posterior distributions with complex shapes and is therefore recommended when dealing with complex problems.
    # + 'sample_method' : method used to uniformly sample within the likelihood constraint, conditioned on the provided bounds. 
    #    If not specified the default is 'auto', i.e. dynesty will pick a method based on the dimensionaly of the problem. If dealing with posterior distributions with complex shapes, 'slice' is recommended.
    # + 'dlogz' : log-likelihood difference threshold below which the NS run with stop (dlogz). 
    #    Defaut is 0.1 and can be placed higher/lower to stop the run earlier/later.
    # + 'monitor' : store dynesty checkpoint files. Highly recommended for long NS runs that could potentially crash.
    local_dic[data_type]['sampler_set']={}
    
    
    #%%%% Complex priors
    #    - to be defined manually within the code
    #    - leave empty, or put in field for each priors and corresponding options
    local_dic[data_type]['prior_func']={}    


    #%%%% Manual walkers exclusion        
    #    - excluding manually some of the walkers
    #    - define conditions within routine
    local_dic[data_type]['exclu_walk']=  False           


    #%%%% Automatic walkers exclusion        
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
    #%%% NS settings
    ################################################################################################## 
    
    #%%%%% Hessian matrix
    #    - string containing the location of a Fit_results.npz file containing a Hessian matrix.
    #    - This Hessian matrix must have been computed from the same parameters are the ones used in the NS fit.
    #    - To use this option, we recommend users first run a fit with fit_mode set to chi2. The chi2 fit will automatically
    #    - create and store the Hessian matrix. A NS can subsequently be run with the path to the Hessian being set as
    #    - the location of the chi2 fit results.
    local_dic[data_type]['use_hess'] = ''


    #%%%% Run mode
    #    - set to
    # + 'use': runs NS  
    # + 'reuse' (with gen_dic['calc_fit_X']=True): load NS results, allow changing error definitions without running the ns again
    local_dic[data_type]['run_mode']='use'
    
    
    #%%%% Monitor NS
    local_dic[data_type]['progress']= True
    
    
    #%%%% Runs to re-use
    #    - list of ns runs to reuse
    #    - if 'reuse' is requested, leave empty to automatically retrieve the ns run available in the default directory
    #  or set the list of ns runs to retrieve (they must have been run with the same settings, but the burnin can be specified for each run)
    local_dic[data_type]['reuse']={}


    #%%%%%% Runs to re-start
    #    - indicate path to a 'raw_chains' file
    #      the ns will restart at the last step of the previous chains, and run with the parameters indicated in 'sampler_set'
    local_dic[data_type]['reboot']=''

    #%%%%%% Runs to restore
    #    - indicate path to a dynesty checkpoint file
    #      the ns will resume the sampler, and run with the parameters indicated in 'sampler_set'
    local_dic[data_type]['restore']=''
    
    #%%%%%% Complex priors
    #    - to be defined manually within the code
    #    - leave empty, or put in field for each priors and corresponding options
    local_dic[data_type]['prior_func']={}       


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
    

    #%%%%% Chi2 chains
    local_dic[data_type]['save_chi2_chains']=''
            
    
    #%%%%% Chi2 chains
    local_dic[data_type]['save_chi2_chains']=''
         
    
    #%%%%% Corner plot for MCMC / NS fits
    #    - see function for options
    local_dic[data_type]['corner_options']={}


    #%%%%% 1D PDF for MSMS / NS fits
    #    - on properties derived from the fits to individual profiles
    if data_type in ['DI','Intr','Atm']:
        plot_dic['prop_'+data_type+'_PDFs']=''      
    
    
    #%%%%% Chi2 values
    #    - plot chi2 values for each datapoint
    if 'Prop' in data_type:
        plot_dic['chi2_fit_'+data_type]='' 
        
    return None
