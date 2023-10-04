import numpy as np
from utils import stop
from constant_data import Rjup,Rearth,AU_1,Rsun,c_light
from copy import deepcopy
from ANTARESS_main import ANTARESS_main
from ANTARESS_systems import all_system_params

##################################################################################################    
#%%% General information
#    - each module can be activated independently 
#      in most modules the user can choose to calculate data (in which case it will then be automatically saved on disk) or to retrieve it (in which case the pipeline will check these data already exists)
#      keeping all data in memory is not possible when processing e2ds, which is why the pipeline works in each module by retrieving the relevant data from the disk
#   Required packages:
#    - scipy, lmfit, batman-package, astropy, emcee, pathos, pandas, dace_query, statsmodels, PyAstronomy        
#    - resampling routine: 
# + https://obswww.unige.ch/~delisle/staging/bindensity/doc/
#   pip install --extra-index-url https://vincent:cestpasfaux@obswww.unige.ch/~delisle/staging bindensity --upgrade
# + do not use routines with non-continuous tables, as it will mess up with banded covariance matrixes
#    - pySME
# + install gcc9 with 'brew install gcc@9'
# + run 'pip install pysme-astro'
#    - KitCat
# + install gsl with 'brew install gsl'
# + run 'python setup_lbl_fit.py build' after setting up the path to your local python installation in this file
#   copy the compiled file 'calculate_RV_line_by_line3.cpython-XX-darwin.so' into your ANTARESS/KitCat directory  
################################################################################################## 

#Initializing generic dictionaries
gen_dic={}
plot_dic={}
data_dic={
    'DI':{'fit_prof':{},'mask':{}},
    'Res':{},
    'PCA':{},
    'Intr':{'fit_prof':{},'mask':{}},
    'Atm':{'fit_prof':{}}}
mock_dic={}
theo_dic={}
detrend_prof_dic={}
glob_fit_dic={
    'IntrProp':{},
    'ResProf':{},
    'IntrProf':{},
    } 



##################################################################################################    
#%%% Settings: generic
##################################################################################################    

#%%%% Planetary system

#%%%%% Star name
gen_dic['star_name']='Earendil' 


#%%%%% Transiting planets
#    - indicate names of the transiting planets to be processed
#    - required to retrieve parameters for the stellar system and light curve
#    - name as given in all_systems_params
#    - for each planet, indicate the instrument and visits in which its transit should be taken into account (visit names are those given through 'data_dir_list')
#      if the pipeline is runned with no data, indicate the names of the mock dataset created artifially with the pipeline
gen_dic['transit_pl']={'Earendil_b':{'ESPRESSO':['20151021']}}     


#%%%%% TTVs
#    - if a visit is defined in this dictionary, the mid-transit time for this visit will be set to the specific value defined here
#    -  format is 'Tcenter_visits' = {'planet':{'inst':{'vis':T0}}}
gen_dic['Tcenter_visits'] = {}


#%%%%% Keplerian planets    
#    - list all planets to consider in the system for the star keplerian motion
#    - set to 'all' for all defined planets to be accounted for
gen_dic['kepl_pl'] = ['all']


#%%%% Saves directory 
gen_dic['save_dir']= ''  


#%%%% Plot settings    
    
#%%%%% Deactivating plot routines
#    - set to False to deactivate
gen_dic['plots_on'] = True


#%%%%% Using non-interactive backend 
gen_dic['non_int_back'] = False


#%%%% Grid run
#    - running ANTARESS with nominal settings properties or looping over a grid for some of these properties
gen_dic['grid_run'] = False
 

#%%%% Input data type
#    - for each instrument select among: 
# + 'CCF': CCFs calculated by standard pipelines on stellar spectra
# + 'spec1D': 1D stellar spectra
# + 'spec2D': echelle stellar spectra
gen_dic['type']={'ESPRESSO':'CCF'}

  
#%%%% Spectral frame
#    - input spectra will be put into the requested frame ('air' or 'vacuum') if relevant
#    - input frames:
# + air: ESPRESSO, HARPS, HARPN, NIRPS_HE, NIRPS_HA
# + vacuum: CARMENES_VIS, EXPRES 
gen_dic['sp_frame']='air'


#%%%% Multi-threading
#    - set to 1 to prevent

#Multi-threading for profile fits
gen_dic['fit_prof_nthreads'] = 14      



#%%%% Data uncertainties

#%%%%% Using covariance matrix
#    - set to True to propagate full covariance matrix and use it in fits (otherwise variance alone is used)
gen_dic['use_cov']=True


#%%%%% Manual variance table 
#    - set instrument in list for its error tables to be considered undefined 
#    - for spectral profiles errors are set to sqrt(F) for disk-integrated profiles and propagated afterwards
#      error can be scaled with 'g_err'
#    - for CCFs the same is done for disk-integrated profiles, but errors on local profiles are set to their continuum dispersion (and propagated afterwards)
gen_dic['force_flag_err']=[]


#%%%%% Error scaling 
#    - all error bars will be multiplied by sqrt(g_err) upon retrieval/definition
#    - format is 'g_err' = {inst : value}
#    - leave empty to prevent scaling
gen_dic['g_err']={}





#%%%% CCF calculation

#%%%%% Mask for stellar spectra
#    - indicate path to mask
#    - should contain at least those two columns: line wavelengths (A) and weights
#    - beware that weights are set to the square of the mask line contrasts (for consistency with the ESPRESSO, HARPS and HARPS-N DRS)
#    - format can be fits, csv, txt, dat
#    - can be used in one of these steps :
# + CCF on input disk-integrated stellar spectra
# + CCF on extracted local stellar spectra
#    - CCF on atmospheric signals will be calculated with a specific mask
#    - can be defined for the purpose of the plots (set to None to prevent upload)
#    - CCF masks should be in the frame requested via gen_dic['sp_frame']
gen_dic['CCF_mask'] = {}


#%%%%% Orders
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


#%%%%% Screening

#%%%%%% First pixel for screening
#    - we keep only bins at indexes ist_bin + i*n_per_bin
#      where n_per_bin is the correlation length of the CCFs
#      ie we remove n_scsr_bins-1 points between two kept points, ie we keep one point in scr_lgth+1 
#    - ist_bin is thus in [ 0 ; n_per_bin-1 ] 
gen_dic['ist_scr']=0


#%%%%%% Screening length determination
#    - select to calculate and plot the dispersion vs bin size on the Master out CCF continuum
#    - the bin size where the noise becomes white can be used as screening length (set manually through scr_lgth)
gen_dic['scr_search']=False


#%%%%%% Screening lengths
#    - set manually for each visit of each instrument
#    - if a visit is not defined for a given instrument, standard pixel size will be used (ie, no screening)
gen_dic['scr_lgth']={}


#%%%%%% Plots: screening length analysis
plot_dic['scr_search']=''    


#%%%% Resampling    

#%%%%% Resampling mode
#    - linear interpolation ('linear') is faster than cubic interpolation ('cubic') but can introduce spurious features at the location of lines, blurred differently when resampled over different spectral tables
# gen_dic['resamp_mode']='linear'
gen_dic['resamp_mode']='cubic'  


#%%%%% Common spectral table
#    - if set to True, data will always be resampled on the same table, specific to a given visit
#      otherwise resampling operations will be avoided whenever possible, to prevent blurring and loss of resolution
#    - this option will not resample different visits of a same instrument onto a common table
#      this is only relevant for specific operations combining different visits, in which case it has to be done in any case 
#    - imposed for CCFs
#    - set to False if left empty
gen_dic['comm_sp_tab'] = {}


#%%%% Data processing

#%%%%% Calculating/retrieving
gen_dic['calc_proc_data']= True


#%%%%% Disable calculation for all activated modules
#    - if set to False: data will be retrieved, if present
#    - if set to True: selection is based upon each module option
gen_dic['calc_all'] = True  




if __name__ == '__main__':
    user = 'mercier'

    #Planetary system
    
    #Star name
    # gen_dic['star_name']='55Cnc'
    # gen_dic['star_name']='GJ436'
    # gen_dic['star_name']='HD3167'
    # gen_dic['star_name']='WASP121'
    #gen_dic['star_name']='KELT9'
    # gen_dic['star_name']='WASP127' 
    gen_dic['star_name']='HD209458' 
    # gen_dic['star_name']='WASP76'        
    # gen_dic['star_name']='Corot7' 
    # gen_dic['star_name']='Nu2Lupi' 
    # gen_dic['star_name']='GJ9827' 
    # gen_dic['star_name']='TOI178' 
    # gen_dic['star_name']='TOI858' 
    # gen_dic['star_name']='Sun'      
    # gen_dic['star_name']='TIC61024636'   
    # gen_dic['star_name']='HIP41378'
    # gen_dic['star_name']='HD15337'
    # gen_dic['star_name']='Altair'    
    # gen_dic['star_name']='TOI-3362'       
    # gen_dic['star_name']='K2-139' 
    # gen_dic['star_name']='TIC257527578' 
    # gen_dic['star_name']='MASCARA1'  
    # gen_dic['star_name']='V1298tau' 
    # gen_dic['star_name']='GJ3090'    
    # gen_dic['star_name']='HD29291'   
    #RM survey
    # gen_dic['star_name']='HAT_P3'
    # gen_dic['star_name']='Kepler25'
    # gen_dic['star_name']='Kepler68'
    # gen_dic['star_name']='HAT_P33'
    # gen_dic['star_name']='K2_105'
    # gen_dic['star_name']='HD89345'
    # gen_dic['star_name']='Kepler63'    
    # gen_dic['star_name']='HAT_P49'   
    # gen_dic['star_name']='WASP47'   
    # gen_dic['star_name']='WASP107'     
    # gen_dic['star_name']='WASP166'        
    # gen_dic['star_name']='HAT_P11'
    # gen_dic['star_name']='WASP156'  
    # gen_dic['star_name']='HD106315'
    # gen_dic['star_name']='HD189733' 
    #NIRPS
    # gen_dic['star_name']='WASP43'
    # gen_dic['star_name']='L98_59'
    # gen_dic['star_name']='GJ1214' 
    # gen_dic['star_name']='WASP189' # vaulato
    #gen_dic['star_name']='AUMic' # mercier  
    # user = 'vaulato'



    #Transiting planets
    #if user=='mercier' and gen_dic['star_name']=='AUMic':
    #    gen_dic['transit_pl'] = {
    #        'AUMicb':{'ESPRESSO' : ['mock_vis']}, 
    #        'AUMicc':{'ESPRESSO' : ['mock_vis']}
    #        }
    #    gen_dic['kepl_pl'] = ['AUMicb', 'AUMicc']

    #gen_dic['transit_pl']='HD189733_b'
    #gen_dic['transit_pl']='Corot_9b'
    #gen_dic['transit_pl']='WASP_8b'
    if user=='vaulato' and gen_dic['star_name']=='WASP189':gen_dic['transit_pl']={'WASP189b':{'NIRPS_HE':['20230604']}} # vaulato
    if gen_dic['star_name']=='GJ436':
        gen_dic['transit_pl']={'GJ436_b':{'ESPRESSO':['20190228','20190429'],
                                          'HARPN':['20160318','20160411'],
                                          'HARPS':['20070509']}}       
    if gen_dic['star_name']=='55Cnc':
        gen_dic['transit_pl']={'55Cnc_e':{'ESPRESSO':['20200205','20210121','20210124'],
                                          'HARPS':['20120127','20120213','20120227','20120315'],
                                          'HARPN':['20121225','20131114','20131128','20140101','20140126','20140226','20140329'],
                                          'SOPHIE':['20120202','20120203','20120205','20120217','20120219','20120222','20120225','20120302','20120324','20120327','20130303'],
                                          'EXPRES':['20220131','20220406']
                                          }} 

    if  gen_dic['star_name']=='HD3167':
        # gen_dic['transit_pl']={'HD3167_b':{'ESPRESSO':['2019-10-09']}}
        # gen_dic['transit_pl']={'HD3167_c':{'HARPN':['2016-10-01']}}
        gen_dic['transit_pl']={'HD3167_b':{'ESPRESSO':['2019-10-09']},'HD3167_c':{'HARPN':['2016-10-01']}} 
    if gen_dic['star_name']=='WASP121':gen_dic['transit_pl']='WASP121b'
    #gen_dic['transit_pl']='Kelt9b'
    if gen_dic['star_name']=='WASP76':
        gen_dic['transit_pl']={'WASP76b':{'ESPRESSO':['20180902','20181030']}} 
    if gen_dic['star_name']=='HD209458': 
        #gen_dic['transit_pl']={'HD209458b':{'ESPRESSO':['20190720','20190911']}}       #ANTARESS paper I
        gen_dic['transit_pl']={'HD209458b':{'ESPRESSO':['mock_vis']}}       #ANTARESS paper I, mock, precisions
        #gen_dic['transit_pl']={'HD209458b':{'ESPRESSO':['mock_vis']},'HD209458c':{'ESPRESSO':['mock_vis']}}       #ANTARESS paper I, mock, multi-pl

    if gen_dic['star_name']=='Corot7':gen_dic['transit_pl']='Corot7b'
    if gen_dic['star_name']=='Nu2Lupi':gen_dic['transit_pl']={'Nu2Lupi_d':{'ESPRESSO':['mock_vis']}}      
    if gen_dic['star_name']=='GJ9827':
        gen_dic['transit_pl']='GJ9827d'
        # gen_dic['transit_pl']='GJ9827b'
    if gen_dic['star_name']=='TOI178':
        gen_dic['transit_pl']='TOI178d'
    if gen_dic['star_name']=='TOI858':gen_dic['transit_pl']={'TOI858b':{'CORALIE':['20191205','20210118']}} 
    if gen_dic['star_name']=='Sun':  
        gen_dic['transit_pl']={'Moon':{'HARPS':['2019-07-02','2020-12-14']}} 
        gen_dic['transit_pl']={'Mercury':{'HARPS':['2019-11-10']}}   
    if gen_dic['star_name']=='TIC61024636':gen_dic['transit_pl']={'TIC61024636b':{'ESPRESSO':['mock_vis']}}         
    if gen_dic['star_name']=='HIP41378':
        gen_dic['transit_pl']={'HIP41378d':{'HARPN':['20191218','20220401']}}  
    if gen_dic['star_name']=='HD15337':gen_dic['transit_pl']={'HD15337c':{'ESPRESSO_MR':['20191122']}}   
    if gen_dic['star_name']=='Altair':gen_dic['transit_pl']={'Altair_b':{'ESPRESSO':['mock_vis']}}      
    if gen_dic['star_name']=='TOI-3362':gen_dic['transit_pl']={'TOI-3362b':{'HARPS':['mock_vis'],'ESPRESSO':['mock_vis']}}  
    if gen_dic['star_name']=='K2-139':gen_dic['transit_pl']={'K2-139b':{'HARPS':['mock_vis'],'ESPRESSO':['mock_vis']}}  
    if gen_dic['star_name']=='TIC257527578':gen_dic['transit_pl']={'TIC257527578b':{'HARPS':['mock_vis'],'ESPRESSO':['mock_vis']}}      
    if gen_dic['star_name']=='MASCARA1':gen_dic['transit_pl']={'MASCARA1b':{'ESPRESSO':['20190714','20190811']}}  
    if gen_dic['star_name']=='V1298tau':gen_dic['transit_pl']={'V1298tau_b':{'HARPN':['20200128','20201207']}}  
    #RM survey
    if gen_dic['star_name']=='HAT_P3':gen_dic['transit_pl']={'HAT_P3b':{'HARPN':['20190415','20200130']}}  
    if gen_dic['star_name']=='Kepler25':gen_dic['transit_pl']={'Kepler25c':{'HARPN':['20190614']}}  
    if gen_dic['star_name']=='Kepler68':gen_dic['transit_pl']={'Kepler68b':{'HARPN':['20190803']}}  
    if gen_dic['star_name']=='HAT_P33':gen_dic['transit_pl']={'HAT_P33b':{'HARPN':['20191204']}}  
    if gen_dic['star_name']=='K2_105':gen_dic['transit_pl']={'K2_105b':{'HARPN':['20200118']}}  
    if gen_dic['star_name']=='HD89345':gen_dic['transit_pl']={'HD89345b':{'HARPN':['20200202']}}  
    if gen_dic['star_name']=='Kepler63':gen_dic['transit_pl']={'Kepler63b':{'HARPN':['20200513']}}
    if gen_dic['star_name']=='HAT_P49':gen_dic['transit_pl']={'HAT_P49b':{'HARPN':['20200730']}}
    if gen_dic['star_name']=='WASP47':
        gen_dic['transit_pl']={'WASP47d':{'HARPN':['20210730']},'WASP47e':{'HARPN':['20210730']}}
        gen_dic['transit_pl']={'WASP47d':{'HARPN':['20210730']}}
        
    if gen_dic['star_name']=='WASP107':gen_dic['transit_pl']={'WASP107b':{'HARPS':['20140406','20180201','20180313','mock_vis'],'CARMENES_VIS':['20180224']}}    
    if gen_dic['star_name']=='WASP166':gen_dic['transit_pl']={'WASP166b':{'HARPS':['20170114','20170304','20170315']}}       
    if gen_dic['star_name']=='HAT_P11':gen_dic['transit_pl']={'HAT_P11b':{'HARPN':['20150913','20151101'],'CARMENES_VIS':['20170807','20170812']}}
    if gen_dic['star_name']=='WASP156'  :gen_dic['transit_pl']={'WASP156b':{'CARMENES_VIS':['20190928','20191025','20191210']}}
    if gen_dic['star_name']=='HD106315':gen_dic['transit_pl']={'HD106315c':{'HARPS':['20170309','20170330','20180323']}}    
    elif gen_dic['star_name']=='GJ3090':
        gen_dic['transit_pl']={'GJ3090b':{'NIRPS_HE':['20221201'],'NIRPS_HA':['20221202']}} 
    elif gen_dic['star_name']=='HD29291':gen_dic['transit_pl']={'HD29291b':{'ESPRESSO':['20201130']}}     
    elif gen_dic['star_name']=='V1298tau':gen_dic['transit_pl']={'V1298tau_b':{'HARPN':['mock_vis']}} 
    elif gen_dic['star_name']=='HD189733':gen_dic['transit_pl']={'HD189733b':{'ESPRESSO':['20210810','20210830']}}     
    #NIRPS
    if gen_dic['star_name']=='WASP43':gen_dic['transit_pl']={'WASP43b':{'NIRPS_HE':['20230119']}} 
    if gen_dic['star_name']=='L98_59':
        gen_dic['transit_pl']={'L98_59c':{'NIRPS_HE':['20230411']},'L98_59d':{'NIRPS_HE':['20230411']}} 
        # gen_dic['transit_pl']={'L98_59c':{'NIRPS_HE':['20230411']}}
        # gen_dic['transit_pl']={'L98_59d':{'NIRPS_HE':['20230411']}}
    if gen_dic['star_name']=='GJ1214':gen_dic['transit_pl']={'GJ1214b':{'NIRPS_HE':['20230407']}}     
    if user=='vaulato' and gen_dic['star_name']=='WASP189':gen_dic['kepl_pl']=['WASP189b'] # vaulato 


    #TTVs
    if gen_dic['star_name']=='V1298tau':
        gen_dic['Tcenter_visits']={'V1298tau_b':{'HARPN':{'20200128':2458877.6306299972,   # -0.0078 +0.0098		28-01-2020         [calculated by Alessandro]
                                                            'mock_vis':2458877.6306299972,
                                                            '20201207':2459191.4343563486}}}    #-0.0046 +0.0045     	07-12-2020         [calculated by Alessandro] 
    elif gen_dic['star_name']=='55Cnc':      
        #Using a more recent T0 (from Meier-Valdes 2022) than the older visits                         
        gen_dic['Tcenter_visits']={'55Cnc_e':{'ESPRESSO':{'20200205':2458884.68696113,'20210121':2459236.75607819,'20210124':2459239.70226327},
                                              'EXPRES':{'20220131':2459610.921583350006,'20220406':2459675.001108840006}}} 
    
    
    
    #Keplerian planets    
    if gen_dic['star_name']=='55Cnc':
        gen_dic['kepl_pl']=['55Cnc_b','55Cnc_c','55Cnc_d','55Cnc_e','55Cnc_f','55Cnc_magc']
    #    gen_dic['kepl_pl']=['55Cnc_b','55Cnc_c','55Cnc_d',''55Cnc_e','55Cnc_f','55Cnc_magc','55Cnc_g']

    if gen_dic['star_name']=='HD3167':gen_dic['kepl_pl']=['HD3167_b','HD3167_c','HD3167_d']

    elif gen_dic['transit_pl']=='Corot7b':
        gen_dic['kepl_pl']=['Corot7c']
    elif gen_dic['transit_pl']=='Nu2Lupi_c':
        gen_dic['kepl_pl']=['Nu2Lupi_b','Nu2Lupi_d']        
    elif gen_dic['transit_pl']=='GJ9827d':gen_dic['kepl_pl']=['GJ9827b','GJ9827c'] 
    elif gen_dic['transit_pl']=='GJ9827b':gen_dic['kepl_pl']=['GJ9827c','GJ9827d']        
    elif gen_dic['transit_pl']=='TOI178d':
        gen_dic['kepl_pl']=['TOI178b','TOI178c','TOI178e','TOI178f','TOI178g']           
    elif gen_dic['star_name']=='TOI858':gen_dic['kepl_pl']=['TOI858b']
    elif gen_dic['star_name']=='HD209458':gen_dic['kepl_pl']=['HD209458b']
    elif gen_dic['star_name']=='WASP76':gen_dic['kepl_pl']=['WASP76b']
    elif gen_dic['star_name']=='GJ436':gen_dic['kepl_pl']=['GJ436_b']
    elif gen_dic['star_name']=='HIP41378':
        gen_dic['kepl_pl']=['HIP41378b','HIP41378c','HIP41378d','HIP41378e','HIP41378f','HIP41378g']
    elif gen_dic['star_name']=='HD15337':gen_dic['kepl_pl']=['HD15337b','HD15337c']
    elif gen_dic['star_name']=='MASCARA1':gen_dic['kepl_pl']=['MASCARA1b']
    elif gen_dic['star_name']=='V1298tau':gen_dic['kepl_pl']=['V1298tau_b']
    
    #RM survey
    elif gen_dic['star_name']=='HAT_P3':gen_dic['kepl_pl']=['HAT_P3b']
    elif gen_dic['star_name']=='Kepler25':gen_dic['kepl_pl']=['Kepler25b','Kepler25c','Kepler25d']
    elif gen_dic['star_name']=='Kepler68':gen_dic['kepl_pl']=['Kepler68b','Kepler68c','Kepler68d']
    elif gen_dic['star_name']=='HAT_P33':gen_dic['kepl_pl']=['HAT_P33b']
    elif gen_dic['star_name']=='K2_105':gen_dic['kepl_pl']=['K2_105b','K2_105c']
    elif gen_dic['star_name']=='HD89345':gen_dic['kepl_pl']=['HD89345b']
    elif gen_dic['star_name']=='Kepler63':gen_dic['kepl_pl']=['Kepler63b']
    elif gen_dic['star_name']=='HAT_P49':gen_dic['kepl_pl']=['HAT_P49b']
    elif gen_dic['star_name']=='WASP47':gen_dic['kepl_pl']=['WASP47b','WASP47c','WASP47d','WASP47e']
    elif gen_dic['star_name']=='WASP107':gen_dic['kepl_pl']=['WASP107b','WASP107c']
    elif gen_dic['star_name']=='WASP166':gen_dic['kepl_pl']=['WASP166b']
    elif gen_dic['star_name']=='HAT_P11':gen_dic['kepl_pl']=['HAT_P11b','HAT_P11c']
    elif gen_dic['star_name']=='WASP156':gen_dic['kepl_pl']=['WASP156b']
    elif gen_dic['star_name']=='HD106315':gen_dic['kepl_pl']=['HD106315b','HD106315c']
    
    elif gen_dic['star_name']=='HD189733':gen_dic['kepl_pl']=['HD189733b']
    elif gen_dic['star_name']=='WASP43':gen_dic['kepl_pl']=['WASP43b']
    elif gen_dic['star_name']=='L98_59':gen_dic['kepl_pl']=['L98_59b','L98_59c','L98_59d','L98_59e']
    elif gen_dic['star_name']=='GJ1214':gen_dic['kepl_pl']=['GJ1214b']    
    
    
    

    #Saves directory 
    gen_dic['save_dir']= '/Users/bourrier/Travaux/ANTARESS/En_cours/'  
    if user=='vaulato':gen_dic['save_dir']= '/Users/valentinavaulato/Documents/PhD/Works/ANTARESS/results/'  # vaulato
    if user=='mercier':gen_dic['save_dir']='/Users/samsonmercier/Desktop/UNIGE/Fall_Semester_2023-2024/antaress_plots' # mercier
    
    #Plot settings    
    
    #Input data type
    if gen_dic['star_name'] in ['HD209458','WASP76','HD29291']:
        #gen_dic['type']={'ESPRESSO':'spec2D'}
        gen_dic['type']={'ESPRESSO':'CCF'}      #ANTARESS I, mock dataset, precisions
        #gen_dic['type']={'ESPRESSO':'spec2D'}      #ANTARESS I, mock dataset, multi-tr
    if gen_dic['star_name']=='GJ436':gen_dic['type']={'ESPRESSO':'spec2D'} 
    if gen_dic['star_name']=='V1298tau':gen_dic['type']={'HARPN':'CCF'} 
    if gen_dic['star_name']=='HIP41378':gen_dic['type']={'HARPN':'CCF'}  
    if gen_dic['star_name']=='55Cnc':
        # gen_dic['type']={'ESPRESSO':'spec2D'}
        gen_dic['type']={'ESPRESSO':'CCF','HARPS':'CCF','HARPN':'CCF','SOPHIE':'CCF','EXPRES':'spec2D'}      
    #RM survey
    if gen_dic['star_name']=='HAT_P3':gen_dic['type']={'HARPN':'CCF'}  
    if gen_dic['star_name']=='Kepler25':gen_dic['type']={'HARPN':'CCF'}  
    if gen_dic['star_name']=='Kepler68':gen_dic['type']={'HARPN':'CCF'}  
    if gen_dic['star_name']=='HAT_P33':gen_dic['type']={'HARPN':'CCF'}  
    if gen_dic['star_name']=='K2_105':gen_dic['type']={'HARPN':'CCF'}  
    if gen_dic['star_name']=='HD89345':gen_dic['type']={'HARPN':'CCF'}  
    if gen_dic['star_name']=='Kepler63':gen_dic['type']={'HARPN':'CCF'}
    if gen_dic['star_name']=='HAT_P49':gen_dic['type']={'HARPN':'CCF'}
    if gen_dic['star_name']=='WASP47':gen_dic['type']={'HARPN':'CCF'}
    if gen_dic['star_name']=='WASP107':gen_dic['type']={'HARPS':'CCF','CARMENES_VIS':'spec2D'}    
    if gen_dic['star_name']=='WASP166':gen_dic['type']={'HARPS':'CCF'}       
    if gen_dic['star_name']=='HAT_P11':gen_dic['type']={'HARPN':'CCF','CARMENES_VIS':'spec2D'}
    if gen_dic['star_name']=='WASP156'  :gen_dic['type']={'CARMENES_VIS':'spec2D'}
    if gen_dic['star_name']=='HD106315':gen_dic['type']={'HARPS':'CCF'}      
    elif gen_dic['star_name']=='GJ3090':gen_dic['type']={'NIRPS_HA':'spec2D','NIRPS_HE':'spec2D'}
    elif gen_dic['star_name']=='HD189733':gen_dic['type']={'ESPRESSO':'CCF'}    
    #NIRPS    
    elif gen_dic['star_name']=='WASP43':gen_dic['type']={'NIRPS_HE':'CCF'}
    elif gen_dic['star_name']=='L98_59':gen_dic['type']={'NIRPS_HE':'CCF'}
    elif gen_dic['star_name']=='GJ1214':gen_dic['type']={'NIRPS_HE':'CCF'}
    elif user=='vaulato' and gen_dic['star_name']=='WASP189':gen_dic['type']={'NIRPS_HE':'spec2D'} # vaulato
  
    
  
    #Spectral frame
    gen_dic['sp_frame']='air'
    
    #Multi-threading for profile fits
    gen_dic['fit_prof_nthreads'] = 10

    #Data uncertainties

    #Using covariance matrix
    gen_dic['use_cov']=False
    if gen_dic['star_name'] in ['HD209458','WASP76','HD189733','GJ436']:gen_dic['use_cov']=True

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







    #Mask for stellar spectra
    #gen_dic['CCF_mask'] = '/Travaux/Radial_velocity/RV_masks/ESPRESSO_F9.fits'        #in the air 
    # gen_dic['CCF_mask'] = '/Travaux/ANTARESS/Method/Masks/Na_doublet_air.txt'        
    
    if gen_dic['star_name']=='HD209458':
        gen_dic['CCF_mask']['ESPRESSO'] = '/Users/bourrier/Travaux/Radial_velocity/RV_masks/ESPRESSO/New_meanC2unity/ESPRESSO_new_F9.fits'
        #gen_dic['CCF_mask']['ESPRESSO'] = '/Users/bourrier/Travaux/ANTARESS/En_cours/HD209458b_Saved_data/CCF_masks_DI/ESPRESSO_binned/Relaxed_selection/CCF_mask_DI_HD209458_ESPRESSO_t10.0_air.txt'       
        # gen_dic['CCF_mask']['ESPRESSO'] = '/Users/bourrier/Travaux/ANTARESS/En_cours/HD209458b_Saved_data/CCF_masks_DI/ESPRESSO_binned/Strict_selection/CCF_mask_DI_HD209458_ESPRESSO_t2.5_air.txt'
        
        # gen_dic['CCF_mask']['ESPRESSO'] = '/Users/bourrier/Travaux/ANTARESS/En_cours/HD209458b_Saved_data/CCF_masks_DI/CCF_mask_Na_air.txt'
        #print('ATTENTION MASQUE')

              

    if gen_dic['star_name']=='WASP76':
        # gen_dic['CCF_mask']['ESPRESSO'] = '/Users/bourrier/Travaux/Exoplanet_systems/WASP/WASP76b/ESPRESSO/Analyse_David/ESPRESSO_F9.fits'
        gen_dic['CCF_mask']['ESPRESSO'] = '/Users/bourrier/Travaux/Radial_velocity/RV_masks/ESPRESSO/New_meanC2unity/ESPRESSO_new_F9.fits'
        # gen_dic['CCF_mask']['ESPRESSO'] = '/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/CCF_masks_DI/ESPRESSO_binned/Relaxed_selection/CCF_mask_DI_WASP76_ESPRESSO_t10.0_air.txt'        
        # gen_dic['CCF_mask']['ESPRESSO'] = '/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/CCF_masks_DI/ESPRESSO_binned/Strict_selection/CCF_mask_DI_WASP76_ESPRESSO_t2.5_air.txt'        
               

    if 'HD3167_b' in gen_dic['transit_pl']:
        gen_dic['CCF_mask']['ESPRESSO']='/Travaux/Radial_velocity/RV_masks/Old_HARPN/K5_sqrt.txt'                              #in the air, old mask, unorm, w=contraste dans l'ancienne DRS
        gen_dic['CCF_mask']['ESPRESSO'] = '/Travaux/Radial_velocity/RV_masks/Old_meanC2unity/ESPRESSO_G9.fits'            #in the air, old mask
        gen_dic['CCF_mask']['ESPRESSO'] = '/Travaux/Radial_velocity/RV_masks/New_meanC2unity/ESPRESSO_new_G9.fits'        #in the air, new mask
        gen_dic['CCF_mask']['ESPRESSO'] = '/Travaux/Radial_velocity/RV_masks/New_meanC2unity/ESPRESSO_new_K2.fits'        #in the air, new mask
        gen_dic['CCF_mask']['ESPRESSO'] = '/Volumes/DiskSecLab/Travaux/Exoplanet_systems/HD3167/HD3167b/RM/Data/Masks/Mask_KitCat_HD3167_ESPRESSO19_t2_rv80_weight_rv_xav_SRF.txt'    #in the air, custom mask
        gen_dic['CCF_mask']['ESPRESSO'] = '/Volumes/DiskSecLab/Travaux/Exoplanet_systems/HD3167/HD3167b/RM/Data/Masks/Mask_KitCat_HD3167_ESPRESSO19_t2_rv80_weight_rv_SRF.txt'    #in the air, custom mask
        # gen_dic['CCF_mask']['ESPRESSO'] = '/Volumes/DiskSecLab/Travaux/Exoplanet_systems/HD3167/HD3167b/RM/Data/Masks/Mask_KitCat_HD3167_ESPRESSO19_t2_rv50_weight_rv_SRF.txt'    #in the air, custom mask
        gen_dic['CCF_mask']['ESPRESSO'] = '/Volumes/DiskSecLab/Travaux/Exoplanet_systems/HD3167/HD3167b/RM/Data/Masks/Mask_KitCat_HD3167_ESPRESSO19_t2_rv50_weight_rv_xav_SRF.txt'    #in the air, custom mask
        
    if 'HD3167_c' in gen_dic['transit_pl']:
        # gen_dic['CCF_mask']['HARPN'] = '/Travaux/Radial_velocity/RV_masks/Old_HARPN/K5_sqrt.txt'                              #in the air, old mask, unorm, w=contraste dans l'ancienne DRS
        # gen_dic['CCF_mask']['HARPN'] = '/Travaux/Radial_velocity/RV_masks/Old_meanC2unity/ESPRESSO_G9.fits'            #in the air, old mask
        # gen_dic['CCF_mask']['HARPN'] = '/Travaux/Radial_velocity/RV_masks/Old_meanC2unity/ESPRESSO_K6.fits'            #in the air, old mask
        # gen_dic['CCF_mask']['HARPN'] = '/Travaux/Radial_velocity/RV_masks/New_meanC2unity/ESPRESSO_new_G8.fits'        #in the air, new mask
        # gen_dic['CCF_mask']['HARPN'] = '/Travaux/Radial_velocity/RV_masks/New_meanC2unity/ESPRESSO_new_G9.fits'        #in the air, new mask
        # gen_dic['CCF_mask']['HARPN'] = '/Travaux/Radial_velocity/RV_masks/New_meanC2unity/ESPRESSO_new_K2.fits'        #in the air, new mask
        # gen_dic['CCF_mask']['HARPN'] = '/Travaux/Radial_velocity/RV_masks/New_meanC2unity/ESPRESSO_new_K6.fits'        #in the air, new mask
        # gen_dic['CCF_mask']['HARPN'] = '/Volumes/DiskSecLab/Travaux/Exoplanet_systems/HD3167/HD3167c/RM/Data/Masks/Mask_KitCat_HD3167_HARPN_t3_rv50_SRF.txt'     #in the air, custom mask
        gen_dic['CCF_mask']['HARPN'] = '/Volumes/DiskSecLab/Travaux/Exoplanet_systems/HD3167/HD3167c/RM/Data/Masks/Mask_KitCat_HD3167_HARPN_t3_rv50_weight_rv_xav_SRF.txt'    #in the air, custom mask
    
    if 'Moon' in gen_dic['transit_pl']:
        gen_dic['CCF_mask']['ESPRESSO'] = '/Travaux/Radial_velocity/RV_masks/New_meanC2unity/ESPRESSO_new_G2.fits'        #in the air, new mask        
    
    #RM survey
    if gen_dic['star_name']=='WASP107':
        gen_dic['CCF_mask']['CARMENES_VIS'] = '/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/Masks/Normalized/Mask_KitCat_WASP107_CARMENES_t3_rv500_SRF_norm.txt'
    if gen_dic['star_name']=='HAT_P11':
        gen_dic['CCF_mask']['CARMENES_VIS'] = '/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/Masks/Normalized/Mask_KitCat_HATP11_CARMENES_t3_rv200_SRF_norm.txt'
    if gen_dic['star_name']=='WASP156':
        gen_dic['CCF_mask']['CARMENES_VIS'] = '/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/Masks/Normalized/Mask_KitCat_WASP156_CARMENES_t5_rv200_SRF_norm.txt'     
    if gen_dic['star_name']=='GJ3090':
        gen_dic['CCF_mask']={'NIRPS_HE' : '/Users/bourrier/Travaux/Radial_velocity/RV_masks/NIRPS/NIRPS_K4_mask_norm.dat',
                             'NIRPS_HA' : '/Users/bourrier/Travaux/Radial_velocity/RV_masks/NIRPS/NIRPS_K4_mask_norm.dat'}  
    if gen_dic['star_name']=='55Cnc':
        gen_dic['CCF_mask']={'EXPRES' :'/Users/bourrier/Travaux/Exoplanet_systems/Divers/55_cancri/RM/RMR/Masks/Normalized/Mask_KitCat_HD75732_ESPRESSO19_t3_rv50_weight_rv_sym_SRF_ESPRESSO19_VAC_norm.txt'}          
    
        
        
    #Orders
    #gen_dic['orders4ccf']={'HARPS':np.arange(36),'HARPN':np.arange(36)}
    #gen_dic['orders4ccf']={'HARPS':[36,71],'HARPN':[36,68]}
    #gen_dic['orders4ccf']={'HARPS':[45,71],'HARPN':[45,68]}
    #gen_dic['orders4ccf']={'HARPN':range(35)}
    #gen_dic['orders4ccf']={'HARPN':range(35,69)}
    #gen_dic['orders4ccf']={'HARPS':range(45)}   #< 530 nm environ
    #gen_dic['orders4ccf']={'HARPS':range(45,69)}
    #gen_dic['orders4ccf']={'HARPN':range(20,40)}
#    gen_dic['orders4ccf']={'ESPRESSO':[50]}    
    # if gen_dic['transit_pl']=='Nu2Lupi_c':
    #     gen_dic['orders4ccf']={'ESPRESSO':range(90)}     #Blue detector   
    #     gen_dic['orders4ccf']={'ESPRESSO':range(90,170)} #Red detector      
    # if gen_dic['transit_pl']=='55Cnc_e':
    #     gen_dic['orders4ccf']={'ESPRESSO':range(43)}       
    #     gen_dic['orders4ccf']={'ESPRESSO':range(43,85)}      
    #     gen_dic['orders4ccf']={'ESPRESSO':range(85,128)}      
    #     gen_dic['orders4ccf']={'ESPRESSO':range(129,170)}     
    #     gen_dic['orders4ccf']={'ESPRESSO':range(85)}   
    #     gen_dic['orders4ccf']={'ESPRESSO':range(91,170)} 
    #     gen_dic['orders4ccf']={'ESPRESSO':range(0,1)} 
    #     # gen_dic['orders4ccf']={'ESPRESSO':range(90,170)} 
    #     gen_dic['orders4ccf']={'ESPRESSO':list(range(108,170))} 
    #     # gen_dic['orders4ccf']={'ESPRESSO':range(108,170)} 
    #     # gen_dic['orders4ccf']={'ESPRESSO':[168,169]} 

    # if gen_dic['transit_pl']=='HD3167_b':

    #     #Dispersed orders removed
    #     gen_dic['orders4ccf']={'ESPRESSO':list(range(170))}  
    #     for islice in [86,87,90,91,92,93,94,95,96,97,98,99,100,101,104,105,106,107]:gen_dic['orders4ccf']['ESPRESSO'].remove(islice)
        
    #     # #Dispersed orders removed + blue orders removed (final solution)
    #     # gen_dic['orders4ccf']={'ESPRESSO':list(range(20,170))}  
    #     # for islice in [86,87,90,91,92,93,94,95,96,97,98,99,100,101,104,105,106,107]:gen_dic['orders4ccf']['ESPRESSO'].remove(islice)

    #     # #blue orders removed 
    #     # gen_dic['orders4ccf']={'ESPRESSO':list(range(20,170))}  

        
    # if gen_dic['transit_pl']=='HD3167_c':
    #     gen_dic['orders4ccf']={'HARPN':list(range(4,69))}  
    #     for islice in [43,44,47,48]:gen_dic['orders4ccf']['HARPN'].remove(islice)      


    #Screening

    #First pixel for screening
    gen_dic['ist_scr']=0
    
    #Screening length determination
    gen_dic['scr_search']=False

    #Screening lengths
    gen_dic['scr_lgth']={}
     
    #Plots: screening length analysis
    plot_dic['scr_search']=''    






    #Data processing

    #Calculating/retrieving
    gen_dic['calc_proc_data']=True   &   False
    if gen_dic['star_name'] in ['HD189733','WASP43','L98_59','GJ1214','WASP107']:gen_dic['calc_proc_data']=True  & False
    if user=='vaulato' and gen_dic['star_name'] in ['WASP189']:gen_dic['calc_proc_data']=True #& False# vaulato












##################################################################################################       
#%%%Module: mock dataset 
#    - if activated, the pipeline will generate/retrieve a mock dataset instead of observational data
#    - use to define mock datasets that can then be processed in the same way as observational datasets
#    - for now, limited to mocking a single band
#    - the module uses options for the planet and star grid defined throughout the pipeline:
#         + 'nsub_Dstar' defines the resolution of the stellar grid (must be an odd number)
#         + 'nsub_Dpl' defines the resolution of the planet grid
#         + 'n_oversamp' defines the oversampling of the planet position over each exposure    
#         + limb-darkening (common)
#         + gravity-darkening, if the star is oblate (common)        
################################################################################################## 

#%%%% Activating module
gen_dic['mock_data'] =  False


#%%%% Multi-threading
mock_dic['nthreads'] = 14  


#%%%% Defining artificial visits
#    - exposures are defined for each instrument/visit
#    - exposures can be defined
# + manually: indicate lower/upper exposure boundaries ('bin_low' and 'bin_high', ordered)
# + automatically : indicate total range ('exp_range' in BJD) and number of exposures ( 'nexp')
#    - indicate times in BJD
mock_dic['visit_def']={}


#%%%% Spectral profile settings

#%%%%% Spectral table for disk-integrated profiles 
#    - in star rest frame, in A or km/s depending on chosen data type and model (see 'intr_prof')
mock_dic['DI_table']={'x_start':-150.,'x_end':150.,'dx':0.01}


#%%%%% Heliocentric stellar RV
#    - in km/s
#    - keplerian motion is added automatically to each exposure using the gen_dic['kepl_pl'] planets
#    - format is 'sysvel' = {inst : {vis : value}}  
mock_dic['sysvel']= {}  
    

#%%%%% Intrinsic stellar spectra
#    - we detail here the options and settings used throughout the pipeline (in gen_dic['mock_data'], gen_dic['fit_DI'], gen_dic['fit_IntrProf'], and gen_dic['loc_data_corr']) to define intrinsic profiles
#    - 'mode' = 'ana': intrinsic profiles are calculated analytically from input properties 
# + set line_trans = None for the analytical model to be generated in RV space (CCF mode), or set it to the rest wavelength of the considered transition in the star rest frame (spectral mode)
# + models are calculated directly on the 'DI_table', which is defined in RV space even in spectral mode to facilitate line profile calculation
#   the model table table can be oversampled using the theo_dic['rv_osamp_line_mod'] field 
# + line properties can be specific to each instrument/visit
# + line properties can vary as a polynomial along the chosen dimension 'coord_line'
#   the role of the coefficients depends on the polynomial mode (absolute or polynomial) 
#   'pol_mode' = 'abs' : coeff_pol[n]*x^n + coeff_pol[n-1]*x^(n-1) .. + coeff_pol[0]*x^0
#   'pol_mode' = 'modul': (coeff_pol[n]*x^n + coeff_pol[n-1]*x^(n-1) .. + 1)*coeff_pol[0]
# + lines properties can be derived from the fit to disk-integrated or intrinsic line profiles 
# + line properties are defined through 'mod_prop', with the suffix "__IS__VS_" to keep the structure of the fit routine while defining properties visit per visit here   
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
#    - format is {inst:{vis:value}}
mock_dic['flux_cont']={}

#%%%%% Instrumental gain
#    - the final count level is proportional to 'flux_cont' x 'gcal' but we separate the two fields to control separately the stellar emission and instrumental gain
#    - set to 1 if undefined
#    - format is {inst:{value}}
mock_dic['gcal']={}
   

#%%%% Noise settings

#%%%%% Flux errors
#    - controls error calculation
#    - noise value is drawn for each pixel based on number of measured counts
#    - leave undefined to prevent noise being defined
#    - format is {inst:{vis:bool}}
mock_dic['set_err'] = {}    
 
 
#%%%%% Jitter on intrinsic profile properties
#    - for analytical models only
#    - used to simulate local stellar activity
#    - defined individually for all exposures
#    - format is {inst:{vis:{prop1:value,prop2:value,...}}
mock_dic['drift_intr'] = {}
       

#%%%%% Systematic variations on disk-integrated profiles
#    - for all types of models
#    - possibilities: RV shift, change in instrumental resolution (replacing nominal instrumental convolution)
#    - format is {inst:{vis:{rv:value,resol:value}}
mock_dic['drift_post'] = {}
       



# # Add spots in the mock dataset 
# #    + Spots are defined by 4 parameters : 
# #        - 'lat' : constant lattitutde of the spot, in star rest frame
# #        - 'Tcenter' : Time (bjd) at wich the spot is at longitude 0
# #        - 'ang' : the angular size (in deg) of the spot
# #        - 'flux' : the flux level of the spot surface, relative to the 'normal' surface of the star.
# #    + Structure is par_ISinst_VSvis_SPspot_name, to match with the structure used in gen_dic['fit_res_prof']
 
# mock_dic['use_spots'] = True  & False
# mock_dic['spots_prop'] = {}

# if gen_dic['star_name'] == 'V1298tau' : 
#     mock_dic['spots_prop']={
#         'HARPN':{
#             'mock_vis':{
                
                
#                 # Pour le spot 'spot1' : 
#                 'lat__ISHARPN_VSmock_vis_SPspot1'     : 30,
#                 'Tcenter__ISHARPN_VSmock_vis_SPspot1' : 2458877.6306 - 12/24,     # 2458877.213933
#                 'ang__ISHARPN_VSmock_vis_SPspot1'     : 20,
#                 'flux__ISHARPN_VSmock_vis_SPspot1'    : 0.4,
                
#                 # Pour le spot 'spot2' : 
#                 'lat__ISHARPN_VSmock_vis_SPspot2'     : 40,
#                 'Tcenter__ISHARPN_VSmock_vis_SPspot2' : 2458877.6306 + 5/24,
#                 'ang__ISHARPN_VSmock_vis_SPspot2'     : 25,
#                 'flux__ISHARPN_VSmock_vis_SPspot2'    : 0.4
#                     },
                    
                    
#                 'mock_vis2' : {}
#                     }}
 

if __name__ == '__main__':


    #Activating module
    gen_dic['mock_data'] =  True     #& False

    #Setting number of threads 
    if user=='mercier':
        mock_dic['nthreads'] = 2 

    #Defining artificial visits
    if gen_dic['star_name'] == 'V1298tau' : 
        mock_dic['visit_def']={
            'HARPN':{'mock_vis' :{'exp_range':2458877.6306+np.array([-7.5,-1.5])/24.,'nexp':50},
                       #'mock_vis2':{'exp_range':2458877.6306+np.array([-1, 5.25])/24.,'nexp':50},
                       }}
    elif gen_dic['star_name'] == 'HD209458' : 
        mock_dic['visit_def']={
            'ESPRESSO':{'mock_vis' :{'exp_range':2454560.806755574+np.array([-3.5,3.5])/24.,'nexp':int(7.*60./20.)},   #ANTARESS I, mock, precisions
            # 'ESPRESSO':{'mock_vis' :{'bin_low':2454560.806755574+np.array([-6.,0.])/24.,'bin_high':2454560.806755574+np.array([-6.+(5./60.),0.+(5./60.)])/24.},    #ANTARESS I, mock, multi-tr,tests
            # 'ESPRESSO':{'mock_vis' :{'exp_range':2454560.806755574+np.array([-6.,6.])/24.,'nexp':int(12.*60./5.)},    #ANTARESS I, mock, multi-tr
                       }}
    if gen_dic['star_name'] == 'WASP107' : 
        mock_dic['visit_def']={
            'HARPS':{'mock_vis' :{'exp_range':2458574.147242+np.array([-3.5,3.5])/24.,'nexp':50},
                       }}
    if user=='mercier' and gen_dic['star_name'] == 'AUMic' :
        mock_dic['visit_def']={
            'ESPRESSO':{'mock_vis':{'exp_range': 2458330.39051+np.array([-4,4])/24., 'nexp':50}
                        }}
    
    #Spectral profile settings
    
    #Spectral table for disk-integrated profiles 
    if gen_dic['star_name'] == 'V1298tau' : 
        mock_dic['DI_table']={'x_start':-150.,'x_end':150.,'dx':0.8}
    if gen_dic['star_name'] == 'HD209458' : 
        mock_dic['DI_table']={'x_start':-25.,'x_end':15.,'dx':0.1}   #ANTARESS I, mock, precisions
        # mock_dic['DI_table']={'x_start':5889.95094-0.75,'x_end':5889.95094+0.75,'dx':0.01}       #w(Na_air) = 5889.95094  ;ANTARESS I, mock, multi-tr
    if gen_dic['star_name'] == 'WASP107' : 
        mock_dic['DI_table']={'x_start':-100.,'x_end':100.,'dx':0.8}
    if user=='mercier' and gen_dic['star_name'] == 'AUMic' :
        mock_dic['DI_table']={'x_start':-150.,'x_end':150.,'dx':0.01}

    
    #Heliocentric stellar RV
    if gen_dic['star_name'] == 'V1298tau' : 
        mock_dic['sysvel']= {'HARPN' : {'mock_vis' : 0.}}  
    if gen_dic['star_name'] == 'HD209458' :
        mock_dic['sysvel']={'ESPRESSO' : {'mock_vis' : 0.}}   #ANTARESS I, mock, precisions and multi-tr          
        # mock_dic['sysvel']={'ESPRESSO' : {'mock_vis' : 10.}} 
    if user=='mercier' and gen_dic['star_name']=='AUMic':
        mock_dic['sysvel']= {'ESPRESSO' : {'mock_vis' : 0.}}  

    
    
    
    #Intrinsic stellar spectra     
    if gen_dic['star_name'] in ['V1298tau'] :
        mock_dic['intr_prof']={'HARPN':{
            'mode':'ana',        
            'coord_line':'r_proj',
            'func_prof_name': 'gauss',             
            'mod_prop':{'ctrst_ord0__IS__VS_' : 0.7,
                        'FWHM_ord0__IS__VS_'  : 4,
                        
                        'amp_l2c__ISHARPN_VSmock_vis' : 0.1,
                        'RV_l2c__ISHARPN_VSmock_vis' : 0,
                        'FWHM_l2c__ISHARPN_VSmock_vis' : 4,
                        
                        'a_damp__ISHARPN_VSmock_vis' : 0.5,
                        'slope__ISHARPN_VSmock_vis' : 0,
                        },
            'pol_mode' : 'modul'}}   
    elif gen_dic['star_name'] == 'HD209458' : 
        mock_dic['intr_prof']={'ESPRESSO' :
            {'mode':'ana',        
            'coord_line':'mu',
            'func_prof_name': 'gauss', 
            'line_trans':5889.95094,     #w(Na_air) = 5889.95094
            'mod_prop':{'ctrst_ord0__IS__VS_' : 0.7,
                        'FWHM_ord0__IS__VS_'  : 4 },
            'pol_mode' : 'modul'}
            
            # {'mode':'theo',                     
            }        
    elif gen_dic['star_name'] == 'WASP107' : 
         mock_dic['intr_prof']={'HARPS' :{'mode':'ana','coord_line':'mu','func_prof_name': 'gauss', 'line_trans':None,'mod_prop':{'ctrst_ord0__IS__VS_' : 0.7,'FWHM_ord0__IS__VS_'  : 4 },'pol_mode' : 'modul'}}         

    if user=='mercier' and gen_dic['star_name'] == 'AUMic' : 
        mock_dic['intr_prof']={'ESPRESSO':{'mode':'ana',        
            'coord_line':'mu',
            'func_prof_name': 'gauss', 
            'line_trans':5889.95094,     #w(Na_air) = 5889.95094
            'mod_prop':{'ctrst_ord0__IS__VS_' : 0.7,
                        'FWHM_ord0__IS__VS_'  : 4 },
            'pol_mode' : 'modul'}
            }

    #Count continuum level
    if gen_dic['star_name'] == 'HD209458' : 
        mock_dic['flux_cont']={'ESPRESSO':{'mock_vis':100.}}
        #mock_dic['flux_cont']={'ESPRESSO':{'mock_vis':1e3}}   #ANTARESS I, mock, precisions  
    if user=='mercier' and gen_dic['star_name'] == 'AUMic' :
        mock_dic['flux_cont']={'ESPRESSO':{'mock_vis':100}}      
            
     
    #Noise settings
    
    #Instrumental calibration
    if gen_dic['star_name'] == 'V1298tau' : 
        mock_dic['gcal'] = {'HARPN' : {'mock_vis'  : 20}}
    if gen_dic['star_name'] == 'HD209458' : 
        mock_dic['gcal'] = {'ESPRESSO' : {'mock_vis'  : 1.}}   #ANTARESS I, mock, precisions             
    if user=='mercier' and gen_dic['star_name'] == 'AUMic' : 
        mock_dic['gcal'] = {'ESPRESSO' : {'mock_vis'  : 1.}}       
    
    #Jitter on intrinsic profile properties
    if gen_dic['star_name'] == 'V1298tau' : 
        mock_dic['drift_intr']   = {'HARPN' :  {'mock_vis' : {'rv'    : 0.005 + np.zeros(40),   # + 5 m/s on all CCF 
                                                                'FWHM_ord0__IS__VS_'  : 1.01*np.ones(40),        # FWHM icreased by 1%
                                                                'ctrst_ord0__IS__VS_' : 1.01*np.ones(40) ,       # ctrst increased by 1%
                                                                }}}        
    
    
    #Systematic variations on disk-integrated profiles
    if gen_dic['star_name'] == 'V1298tau' : 
        mock_dic['drift_post']   = {'HARPN' : {'mock_vis' :  {'rv'    : 0.005 + np.zeros(40),   # + 5 m/s on all CCF 
                                                                'resol'  : 1.01*np.ones(40)}}}        
            
            
    print(gen_dic['kepl_pl'], gen_dic['star_name'], gen_dic['transit_pl'])
            
            











##################################################################################################
#%%% Settings: observational datasets
##################################################################################################

#%%%% Saving log of useful keywords
gen_dic['sav_keywords']=  False  


#%%%% Paths to data directories
#    - data must be stored in a unique directory for each instrument, and unique sub-directories for each instrument visit
#    - the fields defined here will determine which instruments/visits are processed, and which names are used for each visit 
#    - format is {inst:{vis:path}}
gen_dic['data_dir_list']={'ESPRESSO':{'20151021':'default_path_TBD'}}


#%%%% Activity indexes
#    - retrieving activity indexes from DACE if target and data are available
gen_dic['DACE_sp'] = False


#%%%% Using fiber-B corrected data
#    - if available
#    - for each visit set the field to 'all' for all orders to be replaced by their sky-corrected version, or by a list of orders otherwise
#      leave empty to use fiber-A data
#    - format is {inst:{vis:[iord] or 'all'}} where iord are original order indexes
gen_dic['fibB_corr']={}
    

#%%%% Using blazed data
#    - define list of instruments
#    - if available 
#    - blaze-corrected data account for other correction, so that it is better to work with blaze-corrected data and use the flux-to-count module
gen_dic['blazed']=[]


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
# inst > vis > exp and ord > position 
#      exposures index relate to time-ordered tables after 'used_exp' has been applied, leave empty to mask all exposures 
#      define position as [ [x1,x2] , [x3,x4] .. ] in the input rest frame
#    - only relevant if part of the order is masked, otherwise remove the full order 
#    - order indexes are relative to the effective order list, after orders are possibly excluded
gen_dic['masked_pix'] = {}

     
#%%%%% Bad quality pixels
#    - setting bad quality pixels to undefined
#    - beware that undefined pixels will 'bleed' on adjacent pixels in successive resamplings
gen_dic['bad2nan'] = False


#---------------------------------------------------------------------------------------------
#%%%% Weighing settings 
#    - controls the weight profiles used for temporal/spatial resampling:
# + mean calibration profile (choice to use calculated profile)
# + telluric profile (choice to use input/calculated profiles)
# + master stellar spectrum (calculation/retrieval and choice to use)
#---------------------------------------------------------------------------------------------

#%%%%% Using instrumental calibration models
gen_dic['cal_weight'] = True    


#%%%%% Using telluric spectra
#    - if available from input files, or from telluric correction module
gen_dic['tell_weight'] = True   


#%%%%% Master stellar spectrum
#    - not calculated if weighing not required

#%%%%%% Calculating/retrieving
#    - master stellar spectrum for weighing, specific to each visit
#    - calculated after alignment and broadband flux scaling
gen_dic['calc_DImast'] = True   


#%%%%%% Exposures to be binned
#    - indexes of exposures that contribute to the bin series, for each instrument/visit
#    - indexes are relative to the global table in each visit
#    - leave empty to use all out-exposures 
gen_dic['DImast_idx_in_bin']={}


#%%%%%% Using stellar spectrum  
gen_dic['DImast_weight'] = True  


#%%%%%% Plots: weighing master 
#    - the master is plotted after first calculation (ie before undergoing the same processing as the dataset) 
plot_dic['DImast']=''       










if __name__ == '__main__':



    #Paths to data directories
    if gen_dic['transit_pl']=='Corot_9b':    
        gen_dic['data_dir_list']={'SOPHIE':'/Travaux/ANTARESS/Corot9b/Observations_SOPHIE/'}
    elif gen_dic['transit_pl']=='WASP_8b':
        gen_dic['data_dir_list']={'HARPS':'/Travaux/ANTARESS/WASP8b/Observations_HARPS/'}
    
    elif 'GJ436_b' in gen_dic['transit_pl']:
        #gen_dic['data_dir_list']={'HARPN':'/Travaux/ANTARESS/GJ436b/Observations/HARPSN_uncorrected/', #ccf non corrigees de la couleur
        #              'HARPS':'/Travaux/ANTARESS/GJ436b/Observations/HARPS_uncorrected/'}
        # gen_dic['data_dir_list']={'HARPN':'/Travaux/ANTARESS/GJ436b/Observations/HARPSN_dpix0.25/', #ccf corrigees de la couleur
        #               'HARPS':'/Travaux/ANTARESS/GJ436b/Observations/HARPS_dpix0.25/'}
        #gen_dic['data_dir_list']={'HARPN':'/Travaux/ANTARESS/GJ436b/Observations/HARPSN_dpix0.82/', #ccf corrigees de la couleur
        #              'HARPS':'/Travaux/ANTARESS/GJ436b/Observations/HARPS_dpix0.82/'}

        if gen_dic['type']=='CCF':
            # gen_dic['data_dir_list']={'ESPRESSO':['/Users/bourrier/Travaux/Exoplanet_systems/Glieses/GJ436b/RM/Data/ESPRESSO/Recent_red/Old_masks/2019-02-27/','/Users/bourrier/Travaux/Exoplanet_systems/Glieses/GJ436b/RM/Data/ESPRESSO/Recent_red/Old_masks/2019-04-29/']}    #new red, old mask            
            gen_dic['data_dir_list']={'ESPRESSO':{'20190228':'/Users/bourrier/Travaux/Exoplanet_systems/Glieses/GJ436b/RM/Data/ESPRESSO/Recent_red/New_masks/2019-02-27/','20190429':'/Users/bourrier/Travaux/Exoplanet_systems/Glieses/GJ436b/RM/Data/ESPRESSO/Recent_red/New_masks/2019-04-29/'}}    #new red, new mask              
            
            
            # gen_dic['data_dir_list']={
            # #                             'ESPRESSO':['/Users/bourrier/Travaux/Exoplanet_systems/Glieses/GJ436b/RM/Data/ESPRESSO/Recent_red/New_masks/2019-02-27/','/Users/bourrier/Travaux/Exoplanet_systems/Glieses/GJ436b/RM/Data/ESPRESSO/Recent_red/New_masks/2019-04-29/'],    #new red, new mask, final fit            
            #                             'HARPN':{'20160318':'/Users/bourrier/Travaux/Exoplanet_systems/Glieses/GJ436b/RM/Data/HARPN/Recent_red/V0/2016-03-18/','20160411':'/Users/bourrier/Travaux/Exoplanet_systems/Glieses/GJ436b/RM/Data/HARPN/Recent_red/V0/2016-04-12/'},               #new red, new mask     , final fit
            #                             'HARPS':{'20070509':'/Users/bourrier/Travaux/Exoplanet_systems/Glieses/GJ436b/RM/Data/HARPS/Recent_red/V0/2007-05-09/'}                                                                                                          #new red, new mask             , final fit
            # #                             # 'HARPN':['/Users/bourrier/Travaux/Exoplanet_systems/GJ436b/RM/Data/HARPN/Recent_red/BCcorr/2016-03-18/','/Users/bourrier/Travaux/Exoplanet_systems/GJ436b/RM/Data/HARPN/Recent_red/BCcorr/2016-04-12/'],             #new red, new mask, BC corrected                   
            # #                             # 'HARPS':['/Users/bourrier/Travaux/Exoplanet_systems/GJ436b/RM/Data/HARPS/Recent_red/BCcorr/2007-05-09/']                                                                                                                 #new red, new mask, BC corrected                                  
            #                             }
        else:
            gen_dic['data_dir_list']={'ESPRESSO':{'20190228':'/Volumes/T7/SAVE_TEMP_VINCENT/Travaux/Exoplanet_systems/Glieses/GJ436b/RM/Data/ESPRESSO/1D/Data/2019-02-27/','20190429':'/Volumes/T7/SAVE_TEMP_VINCENT/Travaux/Exoplanet_systems/Glieses/GJ436b/RM/Data/ESPRESSO/1D/Data/2019-04-29/'}}             
            
        
    elif gen_dic['star_name']=='55Cnc':
    # ##    #ccf non corrigees de la couleur
    # ##    gen_dic['data_dir_list']={'HARPS':'/Travaux/ANTARESS/55Cnc/Data/HARPS_HARPSN/NoColorCorrected/HARPS/'}
    # #    gen_dic['data_dir_list']={'HARPN':'/Travaux/ANTARESS/55Cnc/Data/HARPS_HARPSN/NoColorCorrected/HARPSN/'}
    # #    gen_dic['data_dir_list']={'HARPS':'/Travaux/ANTARESS/55Cnc/Data/HARPS_HARPSN/NoColorCorrected/HARPS/',
    # #                  'HARPN':'/Travaux/ANTARESS/55Cnc/Data/HARPS_HARPSN/NoColorCorrected/HARPSN/'}
    # #    gen_dic['data_dir_list']={'SOPHIE':'/Travaux/ANTARESS/55Cnc/Data/SOPHIE/NoColorCorrected/'}
    
        
    # #    #ccf corrigees de la couleur  
    # #    gen_dic['data_dir_list']={'HARPS':'/Travaux/ANTARESS/55Cnc/Data/HARPS_HARPSN/ColorCorrected/HARPS/'}
    # #    gen_dic['data_dir_list']={'HARPN':'/Travaux/ANTARESS/55Cnc/Data/HARPS_HARPSN/ColorCorrected/HARPSN/'} 
    # #    gen_dic['data_dir_list']={'HARPN':'/Travaux/ANTARESS/55Cnc/Data/HARPS_HARPSN/ColorCorrected/HARPSN_same_wave_calibration/'}
    # #    gen_dic['data_dir_list']={'HARPN':'/Travaux/ANTARESS/55Cnc/Data/HARPS_HARPSN/ColorCorrected/HARPSN/',
    # #                  'HARPS':'/Travaux/ANTARESS/55Cnc/Data/HARPS_HARPSN/ColorCorrected/HARPS/'}
    #     gen_dic['data_dir_list']={'SOPHIE':'/Travaux/ANTARESS/55Cnc/Data/SOPHIE/ColorCorrected/'}    
    
    #     if gen_dic['type']=='CCF':
    #         gen_dic['data_dir_list']={'ESPRESSO':['/Volumes/DiskSecLab/Travaux/ANTARESS/55Cnc/Data/ESPRESSO_CCF/2020-02-04/Default_DRS/']}  
    #         gen_dic['data_dir_list']={'ESPRESSO':['/Volumes/DiskSecLab/Travaux/ANTARESS/55Cnc/Data/ESPRESSO_CCF/2020-02-04/Custom_Fbal/']}    
    #     if gen_dic['type']=='spec2D':
    #         gen_dic['data_dir_list']={'ESPRESSO':'/Volumes/DiskSecLab/Travaux/ANTARESS/55Cnc/Data/ESPRESSO_e2ds/'}  

        gen_dic['data_dir_list']={}
        
        if gen_dic['type']['ESPRESSO']=='spec2D':
            gen_dic['data_dir_list']={'ESPRESSO':{
                #'20200205':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/55_cancri/RM/RMR/Data/ESPRESSO/S2D/20200205/',
                '20210121':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/55_cancri/RM/RMR/Data/ESPRESSO/S2D/20210121/',
                '20210124':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/55_cancri/RM/RMR/Data/ESPRESSO/S2D/20210124/'
                }}
        else:                    
            # gen_dic['data_dir_list']['ESPRESSO']={}     
            # for vis in ['20200205','20210121','20210124']:
            # # for vis in ['20210124']:
            #     gen_dic['data_dir_list']['ESPRESSO'][vis] = '/Users/bourrier/Travaux/Exoplanet_systems/Divers/55_cancri/RM/RMR/Data/ESPRESSO/CCF/'+vis+'/'

            # gen_dic['data_dir_list']['HARPS']={}     
            # for vis in ['20120127','20120213','20120227','20120315']:
            #     gen_dic['data_dir_list']['HARPS'][vis] = '/Users/bourrier/Travaux/Exoplanet_systems/Divers/55_cancri/RM/RMR/Data/HARPS_HARPSN/HARPS/CCF/'+vis+'/'
                
            # gen_dic['data_dir_list']['HARPN']={}     
            # # for vis in ['20121225','20131114','20131128','20140101','20140126','20140226','20140329']:
            # for vis in ['20131114','20131128','20140101','20140126','20140226','20140329']:
            #     gen_dic['data_dir_list']['HARPN'][vis] = '/Users/bourrier/Travaux/Exoplanet_systems/Divers/55_cancri/RM/RMR/Data/HARPS_HARPSN/HARPN/CCF/'+vis+'/'
                            
            # gen_dic['data_dir_list']['SOPHIE']={}     
            # for vis in ['20120202','20120203','20120205','20120217','20120219','20120222','20120225','20120302','20120324','20120327','20130303']:
            #     gen_dic['data_dir_list']['SOPHIE'][vis] = '/Users/bourrier/Travaux/Exoplanet_systems/Divers/55_cancri/RM/RMR/Data/SOPHIE/CCF/ColorCorrected/'+vis+'/'
            
            gen_dic['data_dir_list']['EXPRES']={}     
            for vis in ['20220131','20220406']:
            # for vis in ['20220131']:
            # for vis in ['20220406']:
                gen_dic['data_dir_list']['EXPRES'][vis] = '/Users/bourrier/Travaux/Exoplanet_systems/Divers/55_cancri/RM/RMR/Data/EXPRES/S2D/'+vis+'/'





    elif gen_dic['transit_pl']=='WASP121b':
    #    gen_dic['data_dir_list']={'HARPS':'/Travaux/ANTARESS/WASP121b/Data/HARPS_G2/'}      
    #    gen_dic['data_dir_list']={'HARPS':'/Travaux/ANTARESS/WASP121b/Data/HARPS_F0/reprocess_step_0_25_width_300_WASP121_depth_global_cont/'}    
    #    gen_dic['data_dir_list']={'HARPS':'/Travaux/ANTARESS/WASP121b/Data/HARPS_F0/reprocess_step_0_25_width_300_WASP121_depth_max_local_cont/'}  
    #    gen_dic['data_dir_list']={'HARPS':'/Travaux/ANTARESS/WASP121b/Data/HARPS_F0/reprocess_step_0_25_width_300_WASP121_depth_mean_max_local_cont/'}     
        gen_dic['data_dir_list']={'HARPS':'/Travaux/ANTARESS/WASP121b/Data/HARPS_F0/Final_Mask/'}  

        gen_dic['data_dir_list']={'ESPRESSO_MR':['/Travaux/ANTARESS/WASP121b/Data/ESPRESSO_MR/2018-12-01/'],
                                  'ESPRESSO':['/Travaux/ANTARESS/WASP121b/Data/ESPRESSO/2019-01-07/']}  
    elif gen_dic['transit_pl']=='Kelt9b':
        gen_dic['data_dir_list']={'HARPN':'/Travaux/ANTARESS/Kelt9b/Data/reprocess_step_0_25_width_300_A0/'}      
    elif gen_dic['star_name']=='WASP76':
        if gen_dic['type']['ESPRESSO']=='CCF':
            gen_dic['data_dir_list']={'ESPRESSO':{'20180902':'/Users/bourrier/Travaux/Exoplanet_systems/WASP76b/ESPRESSO/Data/CCF/2018-09-03/','20181030':'/Users/bourrier/Travaux/Exoplanet_systems/WASP76b/ESPRESSO/Data/CCF/2018-10-31/'}}
        elif gen_dic['type']['ESPRESSO']=='spec1D':
            gen_dic['data_dir_list']={'ESPRESSO':{'20180902':'/Users/bourrier/Travaux/Exoplanet_systems/WASP76b/ESPRESSO/Data/S1D/Red2.2.8/2018-09-03/','20181030':'/Users/bourrier/Travaux/Exoplanet_systems/WASP76b/ESPRESSO/Data/S1D/Red2.2.8/2018-10-31/'}}         
        elif gen_dic['type']['ESPRESSO']=='spec2D':
            gen_dic['data_dir_list']={'ESPRESSO':{'20180902':'/Users/bourrier/Travaux/Exoplanet_systems/WASP/WASP76b/ESPRESSO/Data/S2D/Red3.3.0/2018-09-02/','20181030':'/Users/bourrier/Travaux/Exoplanet_systems/WASP/WASP76b/ESPRESSO/Data/S2D/Red3.3.0/2018-10-30/'}}
            gen_dic['data_dir_list']={'ESPRESSO':{'20180902':'/Volumes/T7/SAVE_TEMP_VINCENT/Travaux/Exoplanet_systems/WASP/WASP76b/ESPRESSO/Data/S2D/Red3.3.0/2018-09-02/','20181030':'/Volumes/T7/SAVE_TEMP_VINCENT/Travaux/Exoplanet_systems/WASP/WASP76b/ESPRESSO/Data/S2D/Red3.3.0/2018-10-30/'}}
        
    elif gen_dic['star_name']=='HD209458':
        if gen_dic['type']['ESPRESSO']=='CCF':
            # gen_dic['data_dir_list']={'ESPRESSO':{'20190720':'/Travaux/ANTARESS/HD209458b/Data/ESPRESSO/2019-07-20/','20190911':'/Travaux/ANTARESS/HD209458b/Data/ESPRESSO/2019-09-11/'}} 
            gen_dic['data_dir_list']={'ESPRESSO':{'20190720':'/Users/bourrier/Travaux/Exoplanet_systems/HD209458b/ESPRESSO/Data/CCF/Red2.2.8/2019-07-20/','20190911':'/Users/bourrier/Travaux/Exoplanet_systems/HD209458b/ESPRESSO/Data/CCF/Red2.2.8/2019-09-11/'}}
        elif gen_dic['type']['ESPRESSO']=='spec1D':
            gen_dic['data_dir_list']={'ESPRESSO':{'20190720':'/Users/bourrier/Travaux/Exoplanet_systems/HD209458b/ESPRESSO/Data/S1D/Red2.2.8/2019-07-20/','20190911':'/Users/bourrier/Travaux/Exoplanet_systems/HD209458b/ESPRESSO/Data/S1D/Red2.2.8/2019-09-11/'}}         
        elif gen_dic['type']['ESPRESSO']=='spec2D':
            gen_dic['data_dir_list']={'ESPRESSO':{'20190720':'/Users/bourrier/Travaux/Exoplanet_systems/HD/HD209458b/ESPRESSO/Data/S2D/Red3.3.0/2019-07-19/','20190911':'/Users/bourrier/Travaux/Exoplanet_systems/HD/HD209458b/ESPRESSO/Data/S2D/Red3.3.0/2019-09-10/'}}
            # gen_dic['data_dir_list']={'ESPRESSO':{'20190720':'/Users/bourrier/Travaux/Exoplanet_systems/HD/HD209458b/ESPRESSO/Data/S2D/Red3.3.0/2019-07-19/'}}
            # gen_dic['data_dir_list']={}
            gen_dic['data_dir_list']={'ESPRESSO':{'20190720':'/Volumes/T7/SAVE_TEMP_VINCENT/Travaux/Exoplanet_systems/HD/HD209458b/ESPRESSO/Data/S2D/Red3.3.0/2019-07-19/','20190911':'/Volumes/T7/SAVE_TEMP_VINCENT/Travaux/Exoplanet_systems/HD/HD209458b/ESPRESSO/Data/S2D/Red3.3.0/2019-09-10/'}}
        

            
    elif gen_dic['transit_pl']=='WASP127b':
        gen_dic['data_dir_list']={'HARPS':'/Travaux/ANTARESS/WASP127b/Data/HARPS/'}  


    if 'HD3167_b' in gen_dic['transit_pl']:
        if gen_dic['type']=='CCF':
            # gen_dic['data_dir_list'].update({'ESPRESSO':['/Volumes/DiskSecLab/Travaux/ANTARESS/HD3167b/Data/ESPRESSO/2019-10-08_newred/']})    #DRS plage etendue, G9
            # gen_dic['data_dir_list'].update({'ESPRESSO':['/Volumes/DiskSecLab/Travaux/ANTARESS/HD3167b/Data/ESPRESSO/Reduc_DRS_Allart/CCF_DRS/']})    #Reduction Romain, masque DRS correction tell + micro-tell        
            # gen_dic['data_dir_list'].update({'ESPRESSO':['/Volumes/DiskSecLab/Travaux/ANTARESS/HD3167b/Data/ESPRESSO/Reduc_DRS_Allart/CCF_G8/']})    #Reduction Romain, masque G8 sans correction tell + micro-tell        
            # gen_dic['data_dir_list'].update({'ESPRESSO':['/Volumes/DiskSecLab/Travaux/ANTARESS/HD3167b/Data/ESPRESSO/Reduc_DRS_Allart/CCF_G8_tellcorr/']})    #Reduction Romain, masque G8 apres correction tell + micro-tell        
            # gen_dic['data_dir_list'].update({'ESPRESSO':['/Volumes/DiskSecLab/Travaux/ANTARESS/HD3167b/Data/ESPRESSO/Reduc_DRS_Allart/CCF_G9_square_contrast/']})    #Reduction Romain, DRS plage etendue, G9, poids = contrast       
            gen_dic['data_dir_list'].update({'ESPRESSO':['/Users/bourrier/Travaux/Exoplanet_systems/HD3167/HD3167b/RM/Data/ESPRESSO/2019-10-08_newmask/']})  #Reduction Christophe, nouveau masque custom
        else:
            gen_dic['data_dir_list'].update({'ESPRESSO':['/Volumes/DiskSecLab/Travaux/Exoplanet_systems/HD3167/HD3167b/RM/Data/ESPRESSO/S2D/']})
    
    if 'HD3167_c' in gen_dic['transit_pl']:
        if gen_dic['type']=='CCF':
            # gen_dic['data_dir_list'].update({'HARPN':'/Travaux/ANTARESS/HD3167c/Data/HARPSN_K5/'}    
            # gen_dic['data_dir_list'].update({'HARPN':['/Volumes/DiskSecLab/Travaux/Exoplanet_systems/HD3167/HD3167c/RM/Data/HARPN_new_red_K6/2016-10-01/']})         
            # gen_dic['data_dir_list'].update({'HARPN':['/Volumes/DiskSecLab/Travaux/Exoplanet_systems/HD3167/HD3167c/RM/Data/HARPN_newmasks_G9_meanC2unity/2016-10-01_Oldmask/']})  
            # gen_dic['data_dir_list'].update({'HARPN':['/Volumes/DiskSecLab/Travaux/Exoplanet_systems/HD3167/HD3167c/RM/Data/HARPN_newmasks_G9_meanC2unity/2016-10-01_Newmask/']})  
            gen_dic['data_dir_list'].update({'HARPN':['/Users/bourrier/Travaux/Exoplanet_systems/HD3167/HD3167c/RM/Data/reduced_HD3167_Mask_KitCat_HD3167_HARPN_t3_rv50_weight_rv_xav_SRF/']}) 
        else:        
            gen_dic['data_dir_list'].update({'HARPN':['/Volumes/DiskSecLab/Travaux/Exoplanet_systems/HD3167/HD3167c/RM/Data/S2D/Red2.2.3/2016-10-01/']}) 

            
            
    if gen_dic['transit_pl']=='Corot7b':
            gen_dic['data_dir_list']={'ESPRESSO':['/Volumes/DiskSecLab/Travaux/ANTARESS/Corot7b/Data/2019-02-19/']} 
    elif gen_dic['transit_pl']=='Nu2Lupi_c':
            gen_dic['data_dir_list']={'ESPRESSO':['/Volumes/DiskSecLab/Travaux/ANTARESS/Nu2Lupi_c/Data/2020-03-17/']} 
    elif gen_dic['transit_pl']=='GJ9827d':
        gen_dic['data_dir_list']={'ESPRESSO':['/Volumes/DiskSecLab/Travaux/ANTARESS/GJ9827/GJ9827d/Data/ESPRESSO/2019-08-24/']} 
        # gen_dic['data_dir_list']={'HARPS':['/Volumes/DiskSecLab/Travaux/ANTARESS/GJ9827/GJ9827d/Data/HARPS/2018-08-17/','/Volumes/DiskSecLab/Travaux/ANTARESS/GJ9827/GJ9827d/Data/HARPS/2018-09-17/']}
    elif gen_dic['transit_pl']=='GJ9827b':
        gen_dic['data_dir_list']={'HARPS':['/Volumes/DiskSecLab/Travaux/ANTARESS/GJ9827/GJ9827b/Data/HARPS/2018-08-03/','/Volumes/DiskSecLab/Travaux/ANTARESS/GJ9827/GJ9827b/Data/HARPS/2018-08-14/',
                                           '/Volumes/DiskSecLab/Travaux/ANTARESS/GJ9827/GJ9827b/Data/HARPS/2018-09-17/','/Volumes/DiskSecLab/Travaux/ANTARESS/GJ9827/GJ9827b/Data/HARPS/2018-09-18/']}
    if 'TOI858b' in gen_dic['transit_pl']:
        gen_dic['data_dir_list']={'CORALIE':['/Users/bourrier/Travaux/Exoplanet_systems/TOI-858/RM/Data/COR/2019-12-05/V2_fluxcorr/','/Users/bourrier/Travaux/Exoplanet_systems/TOI-858/RM/Data/COR/2021-01-18/V2_fluxcorr/']}   

    if 'Moon' in gen_dic['transit_pl']:
        if gen_dic['type']=='CCF':          
            #gen_dic['data_dir_list']={'HARPS':['/Users/Maxime/Desktop/Master 2eme/data/HARPSDRS/DRS-3.5/reduced/2019-07-02/']}  #First moon eclipse       
            gen_dic['data_dir_list']={'HARPS':['/Users/Maxime/Desktop/M2 2eme semestre/data 2/data/HARPSDRS/DRS-3.5/reduced/2020-12-14/']}   #Second moon eclipse
        else:
            #gen_dic['data_dir_list']={'HARPS':['/Volumes/NO NAME/dataeclipse/data/HARPSDRS/DRS-2.2.8/reduced/2020-12-14/']}
            gen_dic['data_dir_list']={'HARPS':['/Users/Maxime/Desktop/M2 2eme semestre/data 2/data/HARPSDRS/DRS-3.5/reduced/2020-12-14/']}        

    if 'Mercury' in gen_dic['transit_pl']:
        gen_dic['data_dir_list']={'HARPS':['/Volumes/MaxDisk/Mercury/data_filtered/']} 
        gen_dic['data_dir_list']={'HARPS':['/Users/bourrier/Travaux/Exoplanet_systems/Mercury/Data/CCF/DRS-3.5/2019-11-10_11/']}

    if gen_dic['star_name']=='HIP41378':
        gen_dic['data_dir_list']={'HARPN':{'20191218':'/Users/bourrier/Travaux/Exoplanet_systems/HIP/HIP_41378d/Data/F9_mask/20191218/'}}
        # gen_dic['data_dir_list']={'HARPN':{'20220401':'/Users/bourrier/Travaux/Exoplanet_systems/HIP/HIP_41378d/Data/20220401/'}}
        # gen_dic['data_dir_list']={'HARPN':{'20191218':'/Users/bourrier/Travaux/Exoplanet_systems/HIP/HIP_41378d/Data/20191218/','20220401':'/Users/bourrier/Travaux/Exoplanet_systems/HIP/HIP_41378d/Data/20220401_raw/'}}
        # gen_dic['data_dir_list']={'HARPN':{'20191218':'/Users/bourrier/Travaux/Exoplanet_systems/HIP/HIP_41378d/Data/20191218/','20220401':'/Users/bourrier/Travaux/Exoplanet_systems/HIP/HIP_41378d/Data/20220401/'}}
        gen_dic['data_dir_list']={'HARPN':{'20191218':'/Users/bourrier/Travaux/Exoplanet_systems/HIP/HIP_41378d/Data/G2_mask/20191218/'}}
    
    elif gen_dic['star_name']=='HD15337':
        gen_dic['data_dir_list']={'ESPRESSO_MR':['/Users/bourrier/Travaux/Exoplanet_systems/HD15337/Data/ESPRESSO/20191122/']}

    elif gen_dic['star_name']=='MASCARA1':
        # gen_dic['data_dir_list']={'ESPRESSO':{'20190714':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/MASCARA1b/Data/A-mask_nocolcorr/2019-07-14/','20190811':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/MASCARA1b/Data/A-mask_nocolcorr/2019-08-11/'}}
        # gen_dic['data_dir_list']={'ESPRESSO':{'20190714':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/MASCARA1b/Data/A-mask/2019-07-14/','20190811':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/MASCARA1b/Data/A-mask/2019-08-11/'}}
        # gen_dic['data_dir_list']={'ESPRESSO':{'20190714':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/MASCARA1b/Data/KitCat/2019-07-14/','20190811':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/MASCARA1b/Data/KitCat/2019-08-11/'}}
        # gen_dic['data_dir_list']={'ESPRESSO':{'20190714':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/MASCARA1b/Data/KitCat_noweights/2019-07-14/','20190811':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/MASCARA1b/Data/KitCat_noweights/2019-08-11/'}}
        # gen_dic['data_dir_list']={'ESPRESSO':{'20190714':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/MASCARA1b/Data/Manual_mask_Omar/2019-07-14/','20190811':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/MASCARA1b/Data/Manual_mask_Omar/2019-08-11/'}}
        gen_dic['data_dir_list']={'ESPRESSO':{'20190714':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/MASCARA1b/Data/Manual_mask_Vincent/2019-07-14/','20190811':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/MASCARA1b/Data/Manual_mask_Vincent/2019-08-11/'}}
             
    elif gen_dic['star_name']=='V1298tau':
        gen_dic['data_dir_list']={'HARPN':{'20200128':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/V1298tau/RM/HARPN/20200128/CCF_DRS/','20201207':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/V1298tau/RM/HARPN/20201207/CCF_DRS/'}}
        gen_dic['data_dir_list']={'HARPN':{'20200128':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/V1298tau/RM/HARPN/20200128/CCF_newK2/','20201207':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/V1298tau/RM/HARPN/20201207/CCF_newK2/'}}

    #RM survey
    if gen_dic['star_name']=='HAT_P3':
        gen_dic['data_dir_list']={'HARPN':{'20190415':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HAT-P-3_HARPN_newK2_CCF/20190415/','20200130':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HAT-P-3_HARPN_newK2_CCF/20200130/'}}
        gen_dic['data_dir_list']={'HARPN':{'20190415':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HAT-P-3_HARPN_KitCat_CCF/20190415/','20200130':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HAT-P-3_HARPN_KitCat_CCF/20200130/'}}             #FINAL     
    if gen_dic['star_name']=='Kepler25':
        gen_dic['data_dir_list']={'HARPN':{'20190614':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/Kepler-25_HARPN_newF9_CCF/20190614/'}}                 #FINAL     
        # gen_dic['data_dir_list']={'HARPN':{'20190614':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/Kepler-25_HARPN_KitCat_CCF/20190614/'}}  
    if gen_dic['star_name']=='Kepler68':
        # gen_dic['data_dir_list']={'HARPN':{'20190803':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/Kepler-68_HARPN_G1NormSqrt_CCF/20190803/'}}  
        # gen_dic['data_dir_list']={'HARPN':{'20190803':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/Kepler-68_HARPN_newG2_CCF/20190803/'}}  
        gen_dic['data_dir_list']={'HARPN':{'20190803':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/Kepler-68_HARPN_KitCat_CCF/20190803/'}}             #FINAL     
    if gen_dic['star_name']=='HAT_P33':
        # gen_dic['data_dir_list']={'HARPN':{'20191204':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HAT-P-33_HARPN_newF9_CCF/20191204/'}} 
        gen_dic['data_dir_list']={'HARPN':{'20191204':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HAT-P-33_HARPN_KitCat_CCF/20191204/'}}            #FINAL         
    if gen_dic['star_name']=='K2_105':
        # gen_dic['data_dir_list']={'HARPN':{'20200118':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/K2-105_HARPN_newK2_CCF/20200118/'}}
        gen_dic['data_dir_list']={'HARPN':{'20200118':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/K2-105_HARPN_KitCat_CCF/20200118/'}}           #FINAL     
    if gen_dic['star_name']=='HD89345':
        # gen_dic['data_dir_list']={'HARPN':{'20200202':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HD89345_HARPN_newG8_CCF/20200202/'}} 
        gen_dic['data_dir_list']={'HARPN':{'20200202':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HD89345_HARPN_KitCat_CCF/20200202/'}}            #FINAL     
    if gen_dic['star_name']=='Kepler63':
        # gen_dic['data_dir_list']={'HARPN':{'20200513':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/Kepler-63_HARPN_newG8_CCF/20200513/'}}
        gen_dic['data_dir_list']={'HARPN':{'20200513':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/Kepler-63_HARPN_KitCat_CCF/20200513/'}}            #FINAL             
    if gen_dic['star_name']=='HAT_P49':
        # gen_dic['data_dir_list']={'HARPN':{'20200730':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HAT-P-49_HARPN_newF9_CCF/20200730/'}}
        gen_dic['data_dir_list']={'HARPN':{'20200730':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HAT-P-49_HARPN_KitCat_CCF/20200730/'}}               #FINAL     
    elif gen_dic['star_name']=='WASP47':
        # gen_dic['data_dir_list']={'HARPN':{'20210730':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/WASP-47_HARPN_newG9_CCF/20210730/'}}
        gen_dic['data_dir_list']={'HARPN':{'20210730':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/WASP-47_HARPN_KitCat_CCF/20210730/'}}        #FINAL
    elif gen_dic['star_name']=='WASP107':
        gen_dic['data_dir_list']={}
        gen_dic['data_dir_list'].update({'CARMENES_VIS':{'20180224':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/CARMENES_Data/WASP-107_CARMENES_vis/20180224/'}})              #FINAL     
        
        # gen_dic['data_dir_list'].update({'HARPS':{'20180201':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/WASP-107_HARPS_K5NormSqrt_CCF/20180201/','20180313':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/WASP-107_HARPS_K5NormSqrt_CCF/20180313/'}})
        # gen_dic['data_dir_list'].update({'HARPS':{'20140406':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/WASP-107_HARPNewK6_CCF/20140406/','20180201':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/WASP-107_HARPNewK6_CCF/20180201/','20180313':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/WASP-107_HARPNewK6_CCF/20180313/'}} )                                 
        gen_dic['data_dir_list'].update({'HARPS':{'20140406':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/WASP-107_HARPS_KitCat_CCF/20140406/','20180201':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/WASP-107_HARPS_KitCat_CCF/20180201/','20180313':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/WASP-107_HARPS_KitCat_CCF/20180313/'}})                  #FINAL                                   

    elif gen_dic['star_name']=='WASP166':
        # gen_dic['data_dir_list']={'HARPS':{'20170114':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/WASP-166_HARPNewF9_CCF/20170114/','20170304':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/WASP-166_HARPNewF9_CCF/20170304/','20170315':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/WASP-166_HARPNewF9_CCF/20170315/'}}
        gen_dic['data_dir_list']={'HARPS':{'20170114':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/WASP-166_HARPS_KitCat_CCF/20170114/','20170304':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/WASP-166_HARPS_KitCat_CCF/20170304/','20170315':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/WASP-166_HARPS_KitCat_CCF/20170315/'}}                #FINAL         
    elif gen_dic['star_name']=='HAT_P11':
        gen_dic['data_dir_list']={}
        gen_dic['data_dir_list'].update({'CARMENES_VIS':{'20170807':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/CARMENES_Data/HAT-P-11b_CARMENES_vis/20170807/','20170812':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/CARMENES_Data/HAT-P-11b_CARMENES_vis/20170812/'}})
        # gen_dic['data_dir_list'].update({'HARPN':{'20150913':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HAT-P-11_HARPN_newK2_CCF/20150913/','20151101':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HAT-P-11_HARPN_newK2_CCF/20151101/'}})
        gen_dic['data_dir_list'].update({'HARPN':{'20150913':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HAT-P-11_HARPN_KitCat_CCF/20150913/','20151101':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HAT-P-11_HARPN_KitCat_CCF/20151101/'}})                         #FINAL          

    elif gen_dic['star_name']=='WASP156':
        # gen_dic['data_dir_list']={'CARMENES_VIS':{'20190928':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/CARMENES_Data/WASP-156_CARMENES_vis/20190928/','20191025':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/CARMENES_Data/WASP-156_CARMENES_vis/20191025/','20191210':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/CARMENES_Data/WASP-156_CARMENES_vis/20191210/'}}             
            
        #Last visit abandonned for the analysis
        gen_dic['data_dir_list']={'CARMENES_VIS':{'20190928':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/CARMENES_Data/WASP-156_CARMENES_vis/20190928/','20191025':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/CARMENES_Data/WASP-156_CARMENES_vis/20191025/'}}                 #FINAL          

    
    elif gen_dic['star_name']=='HD106315':
        # gen_dic['data_dir_list']={'HARPS':{'20170309':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HD106315_HARPNewF9_CCF/20170309/','20170330':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HD106315_HARPNewF9_CCF/20170330/','20180323':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HD106315_HARPNewF9_CCF/20180323/'}}
        gen_dic['data_dir_list']={'HARPS':{'20170309':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HD106315_HARPS_KitCat_CCF/20170309/','20170330':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HD106315_HARPS_KitCat_CCF/20170330/','20180323':'/Users/bourrier/Travaux/Exoplanet_systems/Projects/Desert_survey/HARPS_HARPN_CCFs/HD106315_HARPS_KitCat_CCF/20180323/'}}          #FINAL   

    elif gen_dic['star_name']=='GJ3090':
        gen_dic['data_dir_list']={
            'NIRPS_HE':{'20221201':'/Volumes/T7/DRS_NIRPS_BOURRIER/2022-12-01/'},
            'NIRPS_HA':{'20221202':'/Volumes/T7/DRS_NIRPS_BOURRIER/2022-12-02/'}
            }   
    elif gen_dic['star_name']=='HD29291':
        gen_dic['data_dir_list']={'ESPRESSO':{'20201130':'/Users/bourrier/Travaux/Exoplanet_systems/HD/HD29291/Data/20201129/S2D/'}}


    if gen_dic['star_name']=='HD189733':
        gen_dic['data_dir_list']={'ESPRESSO':{'20210810': '/Users/bourrier/Travaux/Exoplanet_systems/HD/HD189733b/RM/RMR/Data/HD189733_20210810_ESPRESSO_KitCat_CCF/',
                                              '20210830': '/Users/bourrier/Travaux/Exoplanet_systems/HD/HD189733b/RM/RMR/Data/HD189733_20210830_ESPRESSO_KitCat_CCF/'}}
    #NIRPS
    if gen_dic['star_name']=='WASP43':gen_dic['data_dir_list']={'NIRPS_HE':{'20230119':'/Users/bourrier/Travaux/Exoplanet_systems/WASP/WASP43b/NIRPS/CCF/2023-01-19/'}}
    if gen_dic['star_name']=='L98_59':gen_dic['data_dir_list']={'NIRPS_HE':{'20230411':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/L98_59/NIRPS/CCF/2023-04-11/'}}
    if gen_dic['star_name']=='GJ1214':gen_dic['data_dir_list']={'NIRPS_HE':{'20230407':'/Users/bourrier/Travaux/Exoplanet_systems/Glieses/GJ1214b/NIRPS/CCF/2023-04-07/'}}  
    if user=='vaulato' and gen_dic['star_name']=='WASP189':gen_dic['data_dir_list']={'NIRPS_HE':{'20230604':'/Users/valentinavaulato/Documents/PhD/Works/ANTARESS/WASP-189b/NIRPS/20230604/'}} # vaulato


    #Activity indexes
    if gen_dic['star_name'] in ['HAT_P49','55Cnc','HD189733']:gen_dic['DACE_sp'] = True   





    #Using fiber-B corrected data
    if gen_dic['star_name']=='GJ436':gen_dic['fibB_corr'].update({'ESPRESSO':{'20190228':'all','20190429':'all'}})        
    elif gen_dic['transit_pl']=='WASP121b':gen_dic['fibB_corr']=['ESPRESSO']
    elif gen_dic['transit_pl'] in ['Corot7b','GJ9827d']:gen_dic['fibB_corr']=['ESPRESSO']
    # elif 'HD3167_b' in gen_dic['transit_pl']:gen_dic['fibB_corr']=['ESPRESSO']
    # elif gen_dic['star_name']=='HD209458':
    #     gen_dic['fibB_corr']={'ESPRESSO':{'20190720':[116,117],'20190911':[116,117]}}
    elif gen_dic['star_name']=='MASCARA1':gen_dic['fibB_corr']=['ESPRESSO']
    # elif gen_dic['star_name']=='V1298tau':gen_dic['fibB_corr']=['HARPN']
    # elif gen_dic['star_name']=='HIP41378':gen_dic['fibB_corr']={'HARPN':['20191218']}   
    #RM survey
    # elif gen_dic['star_name']=='HAT_P3':gen_dic['fibB_corr']={'HARPN':['20200130']}     #NON
    # elif gen_dic['star_name']=='HAT_P33':gen_dic['fibB_corr']={'HARPN':['20191204']}    #NON
    # elif gen_dic['star_name']=='Kepler25':gen_dic['fibB_corr']=['HARPN']    #NON
    # elif gen_dic['star_name']=='Kepler68':gen_dic['fibB_corr']={'HARPN':['20190803']}         #NON 
    # elif gen_dic['star_name']=='K2_105':gen_dic['fibB_corr']=['HARPN']        #NON   
    # elif gen_dic['star_name']=='HD89345':gen_dic['fibB_corr']=['HARPN']      #NON   
    # elif gen_dic['star_name']=='HAT_P49':gen_dic['fibB_corr']=['HARPN']      #NON    
    elif gen_dic['star_name']=='WASP107':gen_dic['fibB_corr']={'HARPS':['20180201','20180313']}      #OUI 
    elif gen_dic['star_name']=='WASP166':gen_dic['fibB_corr']={'HARPS':['20170114']}   #OUI   
    # elif gen_dic['star_name']=='HAT_P11':gen_dic['fibB_corr']={'HARPN':['20150913','20151101']}   #NON 
    elif gen_dic['star_name']=='HD106315':gen_dic['fibB_corr']={'HARPS':['20170309']}   #OUI    # elif gen_dic['star_name']=='WASP47':gen_dic['fibB_corr']={'HARPN':['20210730']}    #NON

    elif gen_dic['star_name']=='55Cnc':    #only available for E1, HN1
        gen_dic['fibB_corr'].update({'ESPRESSO':['20200205']})    #FINAL
        # gen_dic['fibB_corr'].update({'HARPN':['20121225']})     #NON, deux expos seulement
    # if gen_dic['star_name'] in ['HD189733']:
    #     gen_dic['fibB_corr'].update({'ESPRESSO':{'20210810':'all','20210830':'all'}})     #NON


    #Using blazed data
    # if (gen_dic['transit_pl']=='WASP76b') and (gen_dic['type']=='spec2D'):gen_dic['blazed']+=['ESPRESSO']
    if ('HD3167_b' in gen_dic['transit_pl']) and (gen_dic['type']=='spec2D'):gen_dic['blazed']+=['ESPRESSO']
    if ('HD3167_c' in gen_dic['transit_pl']) and (gen_dic['type']=='spec2D'):gen_dic['blazed']+=['HARPN']



    #Data exclusion
    
    #Visits to remove from analysis
    #    - leave empty to use everything
    if gen_dic['transit_pl']=='HD189733_b':
        gen_dic['unused_visits']=[]
        #gen_dic['unused_visits']=['2006-07-29','2006-08-03','2006-09-07','2007-07-19','2007-08-28']
    
    elif gen_dic['transit_pl']=='GJ436_b':
        gen_dic['unused_visits']=[]
        #gen_dic['unused_visits']=['2007-05-09']                #nuits HARPS-N 
        #gen_dic['unused_visits']=['2016-03-18','2016-04-11']  #nuit HARPS seule
        #gen_dic['unused_visits']=['2016-04-11','2007-05-09']  #nuit HARPSN '2016-03-18' seule
        #gen_dic['unused_visits']=['2016-03-18','2007-05-09']  #nuit HARPSN '2016-04-11' seuleman
    
    elif gen_dic['star_name']=='55Cnc':
        gen_dic['unused_visits']={}
    
        #Remove HARPS visits
    #    gen_dic['unused_visits']=['2012-01-27','2012-02-27','2012-02-13','2012-03-15']
    #    gen_dic['unused_visits']=['2012-02-27','2012-02-13','2012-03-15']  # keep only '2012-01-27'
    #    gen_dic['unused_visits']=['2012-01-27','2012-02-13','2012-03-15']  # keep only '2012-02-27'
    #    gen_dic['unused_visits']=['2012-01-27','2012-02-27','2012-03-15']  # keep only '2012-02-13'
    #    gen_dic['unused_visits']=['2012-01-27','2012-02-27','2012-02-13']  # keep only '2012-03-15'
        
        #Nuits HARPS-N visits
    #    gen_dic['unused_visits']=['2012-12-25','2013-11-14','2013-11-28','2014-01-01','2014-01-26','2014-02-26','2014-03-29'] #all    
    #    gen_dic['unused_visits']+=['2013-11-14','2013-11-28','2014-01-01','2014-01-26','2014-02-26','2014-03-29']   #  keep only '2012-12-25' 
    #    gen_dic['unused_visits']+=['2012-12-25','2013-11-28','2014-01-01','2014-01-26','2014-02-26','2014-03-29']  #   keep only '2013-11-14'   
    #    gen_dic['unused_visits']+=['2012-12-25','2013-11-14','2014-01-01','2014-01-26','2014-02-26','2014-03-29']    #  keep only  '2013-11-28'
    #    gen_dic['unused_visits']+=['2012-12-25','2013-11-14','2013-11-28','2014-01-26','2014-02-26','2014-03-29'] #   keep only '2014-01-01'
    #    gen_dic['unused_visits']+=['2012-12-25','2013-11-14','2013-11-28','2014-01-01','2014-02-26','2014-03-29']  #  keep only  '2014-01-26'  
    #    gen_dic['unused_visits']+=['2012-12-25','2013-11-14','2013-11-28','2014-01-01','2014-01-26','2014-03-29']     # keep only '2014-02-26'
    #    gen_dic['unused_visits']+=['2012-12-25','2013-11-14','2013-11-28','2014-01-01','2014-01-26','2014-02-26']   # keep only  '2014-03-29'   
    #    gen_dic['unused_visits']+=['2012-12-25','2013-11-28','2013-11-14']   #analyse calibration commune
    #    gen_dic['unused_visits']=['2012-12-25','2014-01-01']                        #bin des 5 bonnes nuits    
    #    gen_dic['unused_visits']+=['2012-12-25','2014-01-01','2014-02-26']            #bin des 4 meilleures nuits  
    
    #     #Nuits SOPHIE
    #     gen_dic['unused_visits']+=['2013-02-28']   #toujours exclue car pas de points en transit
    # #    gen_dic['unused_visits']+=['2012-02-03','2012-02-05','2012-02-17','2012-02-19','2012-02-22','2012-02-25','2012-03-02','2012-03-24','2012-03-27','2013-02-28','2013-03-03']     #keep only 2012-02-02
    # #    gen_dic['unused_visits']+=['2012-02-02','2012-02-05','2012-02-17','2012-02-19','2012-02-22','2012-02-25','2012-03-02','2012-03-24','2012-03-27','2013-02-28','2013-03-03']     #keep only 2012-02-03

        # #Nuits ESPRESSO
        # gen_dic['unused_visits']=['2019-01-29']
    
    
    elif gen_dic['transit_pl']=='WASP121b':
    #    gen_dic['unused_visits']=['31-12-17']
    #    gen_dic['unused_visits']=['31-12-17','14-01-18']
        gen_dic['unused_visits']=[]
    #
    #elif gen_dic['transit_pl']=='Kelt9b':
    #    gen_dic['unused_visits']=['20-07-2018']
        
        
    # elif gen_dic['transit_pl']=='WASP76b':        
    #     gen_dic['unused_visits']={'ESPRESSO':['2018-10-31']}
 

    elif gen_dic['star_name']=='HAT_P3':
         gen_dic['unused_visits']={'HARPN':['20190415']}      
    
    
    
    #Exposures to keep in analysis
    if 'GJ436_b' in gen_dic['transit_pl']:
        gen_dic['used_exp'].update({'HARPN':{'20160318':np.arange(76),'20160411':np.arange(70)}})
    if 'HD3167_c' in gen_dic['transit_pl']:
        gen_dic['used_exp'].update({'HARPN':{'2016-10-01':np.arange(35)}})
    if gen_dic['transit_pl']=='GJ9827d': 
        gen_dic['used_exp']={'ESPRESSO':{'2019-08-25':range(44)}}
    if 'Moon' in gen_dic['transit_pl']:
         gen_dic['used_exp']={'HARPS':{'2019-07-02':np.arange(513)}}
         #with RMS/error < 1.2:
         gen_dic['used_exp']={'HARPS':{'2019-07-02':[3,5,8,9,13,17,18,19,20,23,24,28,30,31,33,34,36,38,39,40,42,43,44,45,46,47,48,49,51,52,54,55,56,57,58,59,60,
                                                     62,63,64,65,66,67,68,70,71,72,76,79,80,81,82,85,86,89,90,91,92,93,95,96,98,100,103,104,105,109,112,113,115,
                                                     118,120,121,122,125,126,127,129,130,131,133,134,135,137,138,139,141,150,151,152,155,156,158,160,163,166,168,
                                                     169,170,173,175,177,178,179,180,181,182,183,184,185,186,187,188,189,190,194,195,198,200,201,205,206,207,210,
                                                     211,212,214,216,217,218,219,221,222,224,230,233,234,235,236,237,238,239,243,244,245,246,247,248,251,252,253,
                                                     254,257,259,263,264,265,266,267,268,269,271,272,273,274,275,276,277,278,279,280,281,282,283,285,286,287,288,
                                                     289,290,291,292,293,294,295,297,298,300,302,303,305,309,310,312,313,316,319,322,323,326,327,329,330,331,332,
                                                     333,334,336,338,340,341,345,350,351,352,353,354,355,356,357,361,362,364,365,366,368,370,371,374,381,382,383,
                                                     384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,
                                                     411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,435,436,437,
                                                     438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,
                                                     465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,485,486,487,488,489,490,491,
                                                     492,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,509,510,511,512]}}

    if gen_dic['star_name']=='MASCARA1':
         gen_dic['used_exp']={'ESPRESSO':{'20190811':np.arange(4,125)}}


    elif gen_dic['star_name']=='55Cnc':
        gen_dic['used_exp']={'ESPRESSO':{'20200205':list(np.delete(np.arange(97),[69]))},    #last exp excluded
                             'HARPS':{'20120315':list(np.delete(np.arange(41),[38]))},    #low SNR
                             'HARPN':{'20140126':list(np.delete(np.arange(30),[8])),    #outlier RVs  
                                        '20140329':list(np.delete(np.arange(28),[27]))}}    #low SNR

    #RM survey      
    elif gen_dic['star_name']=='WASP107':
         gen_dic['used_exp']={
             'CARMENES_VIS':{'20180224':np.arange(19)},
             'HARPS':{'20140406':np.arange(25),'20180313':np.arange(34)},
             }
    elif gen_dic['star_name']=='WASP156':
         gen_dic['used_exp']={'CARMENES_VIS':{'20190928':list(np.delete(np.arange(20),[0,17,18]))}}
    elif gen_dic['star_name']=='HAT_P33':
        gen_dic['used_exp']={'HARPN':{'20191204':np.arange(56)}}
    elif gen_dic['star_name']=='K2_105':
         gen_dic['used_exp']={'HARPN':{'20200118':list(np.delete(np.arange(35),[3]))}}
    elif gen_dic['star_name']=='WASP166':
        gen_dic['used_exp']={'HARPS':{'20170114':list(np.delete(np.arange(75),[0,1,2,5])),'20170315':list(np.delete(np.arange(66),[32]))}}
    elif gen_dic['star_name']=='HAT_P11':
        gen_dic['used_exp']={'CARMENES_VIS':{'20170812':list(np.delete(np.arange(57),[1,2]))}}
    elif gen_dic['star_name']=='HD106315':
        gen_dic['used_exp']={'HARPS':{'20170330':np.arange(47),'20170309':list(np.delete(np.arange(75),[73]))}}
    # elif gen_dic['star_name']=='HAT_P49':
    #     gen_dic['used_exp']={'HARPN':{'20200730':np.arange(7,126)}}
    #     # gen_dic['used_exp']={'HARPN':{'20200730':np.arange(0,98)}}
    #     # gen_dic['used_exp']={'HARPN':{'20200730':np.arange(7,98)}}
    #     gen_dic['used_exp']={'HARPN':{'20200730':np.arange(7,112)}}
    elif gen_dic['star_name']=='WASP47':
        gen_dic['used_exp']={'HARPN':{'20210730':np.arange(19)}}
    elif gen_dic['star_name']=='HD189733':
        gen_dic['used_exp']={'ESPRESSO':{'20210810':np.arange(40)}}
    elif gen_dic['star_name']=='GJ3090':
        gen_dic['used_exp']={'NIRPS_HE':{'20221201':list(np.delete(np.arange(101),[6,7,8,9,   54,55,56,57,58,59,60,61,62,63,64,65,66,67  ]))},   #54->67 contamination en emission
                             'NIRPS_HA':{'20221202':list(np.delete(np.arange(62),[0,1,2,12,51,52,53]))}}   #51,52,53 ont des gains eleves/anormaux
    elif gen_dic['star_name']=='L98_59':
        #We remove exposures in-transit of planet d (resp. c) when analyzing the transit of planet c (resp. d) since they are independent 
        #We also remove the first two exposures obtained with much lower SNRs
        if list(gen_dic['transit_pl'].keys())==['L98_59c']:
            gen_dic['used_exp']={'NIRPS_HE':{'20230411':list(np.delete(np.arange(123),np.append([0,1],np.arange(73,102))))}}  
        elif list(gen_dic['transit_pl'].keys())==['L98_59d']:
            gen_dic['used_exp']={'NIRPS_HE':{'20230411':list(np.delete(np.arange(123),np.append([0,1],np.arange(15,54))))}} 
        else:
            gen_dic['used_exp']={'NIRPS_HE':{'20230411':list(np.delete(np.arange(123),[0,1]))}} 
                                 
                                 
                 

    #Orders to remove from analysis
    if gen_dic['star_name']=='GJ3090':
        gen_dic['del_orders']={'NIRPS_HE':[42,43,44, 67,68,69,70],'NIRPS_HA':[42,43,44, 67,68,69,70]}   
    if gen_dic['star_name']=='55Cnc':
        gen_dic['del_orders']={'EXPRES':[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,77,78,79,80,81,82,83,84,85]}          




    #Spectral ranges to remove from analysis
    if gen_dic['star_name'] == 'WASP107':
        gen_dic['masked_pix'] = {'CARMENES_VIS':{'20180224':{'exp_list':[],
                                                              'ord_list':{41:[[7916.2,7916.7]],45:[[8417.85,8418.25]],52:[[9308.8,9309.4],[9310.,9311.]]}}}}
    elif gen_dic['star_name'] == 'HAT_P11':
        gen_dic['masked_pix'] = {'CARMENES_VIS':{'20170807':{'exp_list':[],
                                                              'ord_list':{
                                                                  8:[[5578.7,5579.05]],25:[[6570.,6572.]],
                                                                  }}}}
        gen_dic['masked_pix']['CARMENES_VIS']['20170812'] = gen_dic['masked_pix']['CARMENES_VIS']['20170807']
    if gen_dic['star_name'] == 'WASP156':
        gen_dic['masked_pix'] = {'CARMENES_VIS':{'20191210':{'exp_list':[],
                                                              'ord_list':{53:[[9377.5,9378.1]]}}}}
    if gen_dic['star_name'] == 'GJ3090':
        gen_dic['masked_pix'] = {'NIRPS_HA':{'20221202':{'exp_list':[],'ord_list':{40:[[13470.,13600.]],41:[[13530.,13700.]],65-3:[[17980.,20000.]],66-3:[[18190.,20000.]]}}},
                                 'NIRPS_HE':{'20221201':{'exp_list':[],'ord_list':{40:[[13470.,13600.]],41:[[13530.,13700.]],65-3:[[17980.,20000.]],66-3:[[17000.,18025.],[18190.,20000.]]}}}}
    if gen_dic['star_name'] == 'HD209458':
        #Order check based on flux balance correction + telluric correction
        # 110, 111: ok
        # 112, 113: bande tellurique en plein milieu mais reste faible
        # 146, 147: exclure 6865 - 7000
        # 148, 149: exclure 6900 - 6955
        # 162, 163: exclure 7591 - 7650
        # 164, 165: 
        # 	V1: exclure 7500 - 7673 ; 7676 - 7679.5 ; 7682.5 - 7685.5 ; 7689 - 7691.5 ; 7696 - 7698 ; 7702.5 - 7705  
        # 	V2: exclure 7500 - 7673 ; 7675 - 7679. ; 7681.5 - 7685 ; 7688.5 - 7691. ; 7695 - 7697.5 ; 7702. - 7704.5 
        # 166, 167: exclure 7600 - 7715
        gen_dic['masked_pix'] = {'ESPRESSO':{'20190720':{'exp_list':[],'ord_list':{146:[[6865.,7000.]],148:[[6900.,6955.]],162:[[7591.,7650.]],
                                                                                   164:[[7500 , 7673],[7676 , 7679.5],[7682.5 , 7685.5],[7689 , 7691.5],[7696 , 7698],[7702.5 , 7705.]],166:[[7600.,7715.]]}},
                                             '20190911':{'exp_list':[],'ord_list':{146:[[6865.,7000.]],148:[[6900.,6955.]],162:[[7591.,7650.]],
                                                                                   164:[[7500 , 7673],[7675 , 7679.],[7681.5 , 7685.],[7688.5 , 7691.],[7695. , 7697.5],[7702. , 7704.5]],166:[[7600.,7715.]]}}}}
        for vis in gen_dic['masked_pix']['ESPRESSO']:
            for iord in [146,148,162,164,166]:gen_dic['masked_pix']['ESPRESSO'][vis]['ord_list'][iord+1]=gen_dic['masked_pix']['ESPRESSO'][vis]['ord_list'][iord]
            
    if gen_dic['star_name'] == 'WASP76':
        #Order check based on flux balance correction + telluric correction
        # 110, 111: ok
        # 112, 113: bande tellurique en plein milieu mais reste faible
        # 146, 147: exclure 6865 - 7000
        # 148, 149: exclure 6900 - 6950
        # 162, 163: exclure 7592 - 7650
        # 164, 165: 
        # 	V1: exclure 7500 - 7673 ; 7676 - 7679.5 ; 7682.5 - 7685.5 ; 7689 - 7691.5 ; 7696 - 7698 ; 7702.5 - 7705   
        # 	V2: exclure 7500 - 7673 ; 7675.5 - 7679. ; 7681.5 - 7685. ; 7688 - 7691. ; 7695 - 7697.5 ; 7702. - 7704.5  
        # 166, 167: exclure 7600 - 7715
        gen_dic['masked_pix'] = {'ESPRESSO':{'20180902':{'exp_list':[],'ord_list':{146:[[6865.,7000.]],148:[[6900.,6950.]],162:[[7592.,7650.]],
                                                                                   164:[[7500 , 7673],[7676 , 7679.5],[7682.5 , 7685.5],[7689 , 7691.5],[7696 , 7698],[7702.5 , 7705.]],166:[[7600.,7715.]]}},
                                             '20181030':{'exp_list':[],'ord_list':{146:[[6865.,7000.]],148:[[6900.,6950.]],162:[[7592.,7650.]],
                                                                                   164:[[7500 , 7673],[7675.5 , 7679.],[7681.5 , 7685.],[7688. , 7691.],[7695. , 7697.5],[7702. , 7704.5]],166:[[7600.,7715.]]}}}}
        for vis in gen_dic['masked_pix']['ESPRESSO']:
            for iord in [146,148,162,164,166]:gen_dic['masked_pix']['ESPRESSO'][vis]['ord_list'][iord+1]=gen_dic['masked_pix']['ESPRESSO'][vis]['ord_list'][iord]        

    if gen_dic['star_name'] == 'GJ436':
        #Order check based on flux balance correction + telluric correction
        gen_dic['masked_pix'] = {'ESPRESSO':{'20190228':{'exp_list':[],'ord_list':{146:[[6865.,7000.]],148:[[6900.,6950.]],162:[[7592.,7650.]],164:[[7500.,7685.]],166:[[7600.,7715.]]}},
                                             '20190429':{'exp_list':[],'ord_list':{146:[[6865.,7000.]],148:[[6900.,6950.]],162:[[7592.,7650.]],164:[[7500.,7685.]],166:[[7600.,7715.]]}}}}
        for vis in gen_dic['masked_pix']['ESPRESSO']:
            for iord in [146,148,162,164,166]:gen_dic['masked_pix']['ESPRESSO'][vis]['ord_list'][iord+1]=gen_dic['masked_pix']['ESPRESSO'][vis]['ord_list'][iord]        
    


    #---------------------------------------------------------------------------------------------
    #Weighing settings 
    #---------------------------------------------------------------------------------------------

    #Using instrumental calibration models
    gen_dic['cal_weight'] = True #   & False   


    #Using telluric spectra
    gen_dic['tell_weight'] = True  #  & False   



    #Master stellar spectrum

    #Calculating/retrieving
    gen_dic['calc_DImast'] = True  &   False
    if gen_dic['star_name'] in ['HD189733','WASP43','L98_59','GJ1214']:gen_dic['calc_DImast']=True

    #Using stellar spectrum  
    gen_dic['DImast_weight'] = True   #   & False
    # if gen_dic['star_name'] in ['WASP76','HD209458','GJ436','WASP107']:
    #     gen_dic['DImast_weight']=False
    #     print('ATTENTION WEIGHT')

    #Plots: weighing master 
    plot_dic['DImast']=''   #pdf     









##################################################################################################
#%%% Module: stellar continuum
#    - continuum of the disk-integrated or intrinsic stellar spectrum in the star or surface rest frame
#    - used in CCF mask generation and spectrum detrending 
#      for persistent peak masking the continuum is always calculated if the module is activated
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


if __name__ == '__main__':

    #Activating
    gen_dic['DI_stcont'] = True   & False
    gen_dic['Intr_stcont'] = False  
    
    #Calculating/retrieving 
    gen_dic['calc_DI_stcont'] = True #& False
    gen_dic['calc_Intr_stcont'] = True    

    #Rolling window for peak exclusion
    gen_dic['contin_roll_win'] = {'CARMENES_VIS':5}  
    if gen_dic['star_name'] in ['WASP76','HD209458','GJ436']:
        gen_dic['contin_roll_win'] = {'ESPRESSO':2}  
        
    #Smoothing window
    gen_dic['contin_smooth_win']={'CARMENES_VIS':0.5}
    if gen_dic['star_name'] in ['WASP76','HD209458','GJ436']:
        gen_dic['contin_smooth_win'] = {'ESPRESSO':0.3}  
        # gen_dic['contin_smooth_win'] = {'ESPRESSO':15.}   #Test Intr CCF masks : lower than 0.3 is worse, 15 smoohts a lot but works best on the lower S/R data
        
    #Local maxima window
    gen_dic['contin_locmax_win']={'CARMENES_VIS':0.3}
    if gen_dic['star_name'] in ['WASP76','HD209458','GJ436']:
        gen_dic['contin_locmax_win'] = {'ESPRESSO':0.1}  
        # gen_dic['contin_locmax_win'] = {'ESPRESSO':0.05}   #Test Intr CCF masks  : 0.5 worse than 0.1, 0.05 no clear change
        
    #Flux/wavelength stretching
    gen_dic['contin_stretch']={'CARMENES_VIS':10}    
    if gen_dic['star_name'] in ['WASP76','HD209458','GJ436']:
        gen_dic['contin_stretch'] = {'ESPRESSO':15}  
        # gen_dic['contin_stretch'] = {'ESPRESSO':5}   #Test Intr CCF masks 
        
    #Rolling pin radius
    gen_dic['contin_pinR']={'CARMENES_VIS':5}  
    if gen_dic['star_name'] in ['WASP76','HD209458','GJ436']:
        gen_dic['contin_pinR'] = {'ESPRESSO':15}     #DI CCF masks
        # gen_dic['contin_pinR'] = {'ESPRESSO':5}      #Intr CCF masks
 


































    
    

        


        











    
    
    
    
##################################################################################################
#%%% Module: stellar and planet-occulted grids
##################################################################################################

#%%%% Activating module
#    - calculated by default for the analysis and alignement of the intrinsic local stellar profiles, or the extraction and analysis of the atmospheric profiles
#      can be set to True to calculate nonetheless
gen_dic['theoPlOcc'] = True 
    

#%%%% Calculating/retrieving
gen_dic['calc_theoPlOcc']=True  


#%%%% Star

#%%%%% Discretization
#    - number of subcells along the star diameter for model fits
#    - must be an odd number
#    - used (if model relevant) in gen_dic['fit_DI']
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
#             connect into VALD with email address: http://vald.astro.uu.se/ and define:
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
    'linelist': '/Users/bourrier/Travaux/ANTARESS/Method/Secondary/Implementations/SME/Long_Na_band.lin',
    'abund':{},
    'calc':True,
    }  


#%%%% Planet

#%%%%% Discretization        
#    - number of subcells along a planet diameter to define the grid of subcells discretizing the stellar regions it occults 
#    - used for calculations of theoretical properties from planet-occulted regions, and for simulated light curves
#    - beware to use a fine enough grid, depending on the system and dataset
#    - must be an odd number
#    - set to default value if undefined
theo_dic['nsub_Dpl']={} 


#%%%%% Exposure oversampling
#    - oversampling factor of the observed exposures to calculate theoretical properties of planet-occulted regions in the entire pipeline
#    - distance from start to end of exposure will be sampled by RpRs/n_oversamp
#    - set to 0 or leave undefined to prevent oversampling, but beware that it must be defined to bin profiles over other dimensions than phase
#    - oversampling of the flux in the flux scaling module is controlled independently
theo_dic['n_oversamp']={}  


#%%%% Occulted profiles

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
#    - can be relevant to fit profiles measured at low resolution
#    - set to None for no oversampling
theo_dic['rv_osamp_line_mod']=None


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
  



if __name__ == '__main__':

    #Activating module
    gen_dic['theoPlOcc'] = True #  &  False

    #Calculating/retrieving
    gen_dic['calc_theoPlOcc']=True   &  False  

    #Precision
    theo_dic['precision'] = 'high'
    theo_dic['precision'] = 'medium'


    #Star discretization      
    if gen_dic['star_name']=='TOI-3362':theo_dic['nsub_Dstar']=201  
    elif gen_dic['star_name']=='Nu2Lupi':theo_dic['nsub_Dstar']=201 
    elif gen_dic['star_name']=='HD106315':
        theo_dic['nsub_Dstar']=51    
    if gen_dic['star_name']=='V1298tau': 
        if gen_dic['mock_data']:theo_dic['nsub_Dstar']=201   #201 
    if gen_dic['star_name']=='HD209458': 
        theo_dic['nsub_Dstar']=101    #Fit stellar grid, mock data
        
        
    # if gen_dic['star_name']=='WASP76':theo_dic['nsub_Dstar']=201
    if gen_dic['star_name']=='WASP107':theo_dic['nsub_Dstar']=1001    
        
            
    #Stellar macroturbulence
    theo_dic['mac_mode'] = None


    #Theoretical stellar atmosphere
    # theo_dic['st_atm']={
    #     'atm_model':'marcs2012p_t1.0',   #for 'vmic':0.85km/s    
    #     # 'nlte':{'Na':'marcs2012_Na'},
    #     # 'nlte':{'Na':'marcs2012_Na2011'},        
    #     # 'nlte':{'Na':'marcs2012p_t1.0_Na'},        
    #     'nlte':{'Fe':'nlte_Fe_ama51_Feb2022_pysme'},        
    #     'wav_min':5600.,'wav_max':6200.,'dwav':0.01,
    #     'mu_grid':np.logspace(-2.,0.,15),
    #     'linelist': '/Users/bourrier/Travaux/ANTARESS/Method/Secondary/Implementations/SME/Long_Na_band.lin'
    #     # 'MovH':0.01,
    #     # 'abund':{'Na':0.1},
    #     }      
    if gen_dic['star_name']=='HD209458': 
        theo_dic['st_atm']={
            'atm_model':'marcs2012p_t1.0',   #for 'vmic':0.85km/s           
            'nlte':{'Na':'nlte_Na_scatt_pysme'},        
            'wav_min':5880.,'wav_max':5906.,'dwav':0.003,    #range wide enough to fit locally the sodium doublet, step fine enough to not have to oversample the model
            # 'mu_grid':[1e-3,1.],    #for testing
            'mu_grid':np.logspace(-3.,0.,30),
            # 'linelist':'/Users/bourrier/Travaux/Exoplanet_systems/HD/HD209458b/Star/VALD/VALD_HD209458',
            'linelist':'/Volumes/T7/SAVE_TEMP_VINCENT/Travaux/Exoplanet_systems/HD/HD209458b/Star/VALD/VALD_HD209458',
            'MovH':0.02,
            'calc':False,
            }  


    #Planet discretization        
    if gen_dic['star_name']=='HD3167':    
        theo_dic['nsub_Dpl']={'HD3167_b':31.,'HD3167_c':31.} 
        # theo_dic['nsub_Dpl']={'HD3167_b':51.,'HD3167_c':51.}      #final fit + plots   
    elif gen_dic['star_name']=='TOI858':theo_dic['nsub_Dpl']={'TOI858b':51.} 
    elif gen_dic['star_name']=='HD209458':
        theo_dic['nsub_Dpl']={'HD209458b':51.} 
        # theo_dic['nsub_Dpl']={'HD209458b':101.}    #ANTARESS I, mock, precision
        # theo_dic['nsub_Dpl']={'HD209458b':101.,'HD209458c':101.}     #ANTARESS I, mock, multi-pl
    elif gen_dic['star_name']=='WASP76':
        theo_dic['nsub_Dpl']={'WASP76b':151.}    #pour gen_dic['intr_rv_corr'] assez precis (due au RpRs_max eleve, utilise pour definir grille pl generique) 
    elif gen_dic['star_name']=='TIC61024636':theo_dic['nsub_Dpl']={'TIC61024636b':51.} 
    elif gen_dic['star_name']=='GJ436':theo_dic['nsub_Dpl']={'GJ436_b':51.} 
    elif gen_dic['star_name']=='HIP41378':theo_dic['nsub_Dpl']={'HIP41378d':51.}         
    elif gen_dic['star_name']=='HD15337':theo_dic['nsub_Dpl']={'HD15337c':51.}         
    elif gen_dic['star_name']=='Altair':theo_dic['nsub_Dpl']={'Altair_b':101.}          
    elif gen_dic['star_name']=='TOI-3362':theo_dic['nsub_Dpl']={'TOI-3362b':201.}      
    elif gen_dic['star_name']=='Nu2Lupi':theo_dic['nsub_Dpl']={'Nu2Lupi_d':101.} 
    elif gen_dic['star_name']=='K2-139':theo_dic['nsub_Dpl']={'K2-139b':201.}   
    elif gen_dic['star_name']=='TIC257527578':theo_dic['nsub_Dpl']={'TIC257527578b':201.}  
    elif gen_dic['star_name']=='MASCARA1':theo_dic['nsub_Dpl']={'MASCARA1b':101.} 
    elif gen_dic['star_name']=='V1298tau':theo_dic['nsub_Dpl']={'V1298tau_b':21.} 
    elif gen_dic['star_name']=='55Cnc':theo_dic['nsub_Dpl']={'55Cnc_e':51.} 
    #RM survey
    elif gen_dic['star_name']=='HAT_P3':theo_dic['nsub_Dpl']={'HAT_P3b':51.}  
    elif gen_dic['star_name']=='Kepler25':theo_dic['nsub_Dpl']={'Kepler25c':51.}  
    elif gen_dic['star_name']=='Kepler68':theo_dic['nsub_Dpl']={'Kepler68b':31.} 
    elif gen_dic['star_name']=='HAT_P33':theo_dic['nsub_Dpl']={'HAT_P33b':51.} 
    elif gen_dic['star_name']=='K2_105':theo_dic['nsub_Dpl']={'K2_105b':31.}   
    elif gen_dic['star_name']=='HD89345':theo_dic['nsub_Dpl']={'HD89345b':31.}  
    elif gen_dic['star_name']=='HD106315':theo_dic['nsub_Dpl']={'HD106315c':31.}
    elif gen_dic['star_name']=='Kepler63':theo_dic['nsub_Dpl']={'Kepler63b':31.}  
    elif gen_dic['star_name']=='HAT_P49':theo_dic['nsub_Dpl']={'HAT_P49b':51.}   
    elif gen_dic['star_name']=='WASP107':theo_dic['nsub_Dpl']={'WASP107b':201.} 
    elif gen_dic['star_name']=='WASP166':theo_dic['nsub_Dpl']={'WASP166b':31.} 
    elif gen_dic['star_name']=='HAT_P11':theo_dic['nsub_Dpl']={'HAT_P11b':31.} 
    elif gen_dic['star_name']=='WASP47':theo_dic['nsub_Dpl']={'WASP47d':31.,'WASP47e':31.} 
    elif gen_dic['star_name']=='WASP156':theo_dic['nsub_Dpl']={'WASP156b':51.} 



    #Exposure discretization
    if gen_dic['star_name']=='HD3167': 
        theo_dic['n_oversamp']={'HD3167_b':0,'HD3167_c':0.}     #Mettre 0 pour faire fits rapides, 5 pour fit finaux, 50 pour les plots
        # theo_dic['n_oversamp']={'HD3167_b':5,'HD3167_c':5.}     #final fit
        # theo_dic['n_oversamp']={'HD3167_b':50.,'HD3167_c':50.}     #plots
    if gen_dic['star_name']=='TOI858':
        theo_dic['n_oversamp']={'TOI858b':0.}  
        # theo_dic['n_oversamp']={'TOI858b':5.}  
    if gen_dic['star_name']=='GJ436':
        theo_dic['n_oversamp']={'GJ436_b':0.}  
        theo_dic['n_oversamp']={'GJ436_b':5.} 
        theo_dic['n_oversamp']={'GJ436_b':2.} 
        # theo_dic['n_oversamp']={'GJ436_b':50.}       #plots    
    if gen_dic['star_name']=='HIP41378':theo_dic['n_oversamp']={'HIP41378d':3.}          
    if gen_dic['star_name']=='HD15337':theo_dic['n_oversamp']={'HD15337c':0.}   
    if gen_dic['star_name']=='Altair':theo_dic['n_oversamp']={'Altair_b':10.}   
    if gen_dic['star_name']=='MASCARA1':
        theo_dic['n_oversamp']={'MASCARA1b':0.}  
        # theo_dic['n_oversamp']={'MASCARA1b':10.}    
    elif gen_dic['star_name']=='V1298tau':
        theo_dic['n_oversamp']={'V1298tau_b':5.}  
    elif gen_dic['star_name']=='HD209458':
        theo_dic['n_oversamp']={'HD209458b':5.}    #ANTARESS I, mock, precisions  
        # theo_dic['n_oversamp']={'HD209458b':5.,'HD209458c':50.}   #ANTARESS I, mock, multi-tr  
    elif gen_dic['star_name']=='WASP76':
        theo_dic['n_oversamp']={'WASP76b':5.}          
        
    elif gen_dic['star_name']=='55Cnc':
        theo_dic['n_oversamp']={'55Cnc_e':0.}  
        print('METTRE OVERSAMP POUR FITS')
    #RM survey
    elif gen_dic['star_name']=='HAT_P3':theo_dic['n_oversamp']={'HAT_P3b':5.}  
    elif gen_dic['star_name']=='Kepler25':theo_dic['n_oversamp']={'Kepler25c':5.}  
    elif gen_dic['star_name']=='Kepler68':theo_dic['n_oversamp']={'Kepler68b':2.} 
    elif gen_dic['star_name']=='HAT_P33':theo_dic['n_oversamp']={'HAT_P33b':5.} 
    elif gen_dic['star_name']=='K2_105':theo_dic['n_oversamp']={'K2_105b':2.}  
    elif gen_dic['star_name']=='HD89345':theo_dic['n_oversamp']={'HD89345b':2.} 
    elif gen_dic['star_name']=='HD106315':theo_dic['n_oversamp']={'HD106315c':2.} 
    elif gen_dic['star_name']=='Kepler63':theo_dic['n_oversamp']={'Kepler63b':2.}  
    elif gen_dic['star_name']=='HAT_P49':theo_dic['n_oversamp']={'HAT_P49b':5.}  
    elif gen_dic['star_name']=='WASP107':theo_dic['n_oversamp']={'WASP107b':3.} 
    elif gen_dic['star_name']=='WASP166':theo_dic['n_oversamp']={'WASP166b':2.} 
    elif gen_dic['star_name']=='HAT_P11':theo_dic['n_oversamp']={'HAT_P11b':3.} 
    elif gen_dic['star_name']=='WASP47':theo_dic['n_oversamp']={'WASP47d':3.,'WASP47e':3.} 
    elif gen_dic['star_name']=='WASP156':theo_dic['n_oversamp']={'WASP156b':5.} 
    
    elif gen_dic['star_name']=='HD189733':theo_dic['n_oversamp']={'HD189733b':3.}
    #NIRPS
    elif gen_dic['star_name']=='WASP43':theo_dic['n_oversamp']={'WASP43b':3.}
    elif gen_dic['star_name']=='L98_59':theo_dic['n_oversamp']={'L98_59c':3.,'L98_59d':3.}
    elif gen_dic['star_name']=='GJ1214':theo_dic['n_oversamp']={'GJ1214b':3.}


    #RV table        

    # #Oversampling 
    # theo_dic['rv_osamp_line_mod']=0.5 #None  #1.
    # print('ATTENTION rv_osamp_line_mod')




    #Plot settings

    #Planetary orbit discretization
    plot_dic['npts_orbit'] = 10000

    #Contact determination
    if (gen_dic['star_name']=='HD209458') and gen_dic['mock_data']:    plot_dic['stend_ph'] = 2.      #Plot multi-pl

    #Transit chord discretization        
    if gen_dic['star_name']=='MASCARA1':plot_dic['nph_HR'] = 100

    #Range of planet-occulted properties
    plot_dic['plocc_ranges']=''    
    
    #Planet-occulted stellar regions
    plot_dic['occulted_regions']=''
    
    #Planetary system architecture
    plot_dic['system_view']=''   #png
  




    # '''   
    # Calculating properties of spots
    # ''' 

    #Activate
    gen_dic['theo_spots'] = True   &  False

    # #Calculating/retrieving
    # gen_dic['calc_theo_spots']=True   &  False  


    # #Number and properties of spots in each visit
    # theo_dic['spots_prop']={
    #     'HARPN':{
    #         '20200128':{
    #             'spot1':{'RspRs':0.1},
    #             'spot2':{'RspRs':0.1}},
    #         '20201207':{
    #             'spot1':{'RspRs':0.1},
    #             'spot2':{'RspRs':0.1}}
    #         }}            






 
			   






##################################################################################################
#%% Global spectral corrections
##################################################################################################

#%%% Plot settings

#%%%% Individual disk-integrated flux spectra
#    - before/after the various spectral corrections for disk-integrated data
plot_dic['sp_raw']=''    


#%%%% Individual disk-integrated transmission spectra
#    - before/after the various spectral corrections for disk-integrated data
plot_dic['trans_sp']=''    



if __name__ == '__main__':

    #Plot settings
    
    #Individual disk-integrated flux spectra
    plot_dic['sp_raw']=''   #pdf 

    #Individual disk-integrated transmission spectra
    plot_dic['trans_sp']=''   #pdf 








##################################################################################################
#%%% Module: instrumental calibration
#    - always activated in spectral mode, to be used in some modules for photoelectron rescaling (and if requested, for temporal weighing)
#      rescaling spectra to their original photoelectron levels in CCF calculation avoids amplifying artificially errors in regions of lower flux
#    - instrumental calibration is measured directly from the input data using the flux and error tables
#      if error tables are not provided with input data, instrumental calibration is still measured for consistency with the assumed error set to E = sqrt(g_err*F)
#    - disabled in CCF mode
##################################################################################################

#%%%% Calculating/retrieving
gen_dic['calc_gcal']=True  


#%%%% Multi-threading
gen_dic['gcal_nthreads'] =  14   


#%%%% Bin size

#%%%%% Spectral bin size (in A)
#    - applied over each order independently
#    - if set to a larger value than an order width, calibration will not be fitted but set to the measured value over each order
#      binw should be large enough to smoot out sharp variations in the model calibration profile
#    - format is : value
gen_dic['gcal_binw'] = 0.5


#%%%%% Temporal bin size
#    - with low-SNR data it might be necessary to group exposures to perform the calibration estimates
#    - format is : value
gen_dic['gcal_binN'] = 1    


#%%%% Edge polynomials
#    - model is made of a blue, a central, and a red polynomial
#    - set the fraction (in 0-1) of the order width that define the blue and red ranges
#    - set the order of the polynomials (between 2 and 4)
#    - beware that this calibration model will propagate into the weighing and the photoelectron rescaling, and thus sharp variations should be avoided 
#    - if input data are CCFs or 'gcal_binw' is larger than the spectral order width, calibration is set to a constant value  
#    - format is : {prop : value}    
gen_dic['gcal_edges']={'blue':0.3,'red':0.3}    
gen_dic['gcal_deg']={'blue':4,'mid':2,'red':4}

    
#%%%% Outliers     
    
#%%%%% Threshold
#    - calibration values above the global threshold, or outliers in the residuals from a preliminary fit, are sigma-clipped and not fitted
#    - format is : {inst : {prop : value} }   
gen_dic['gcal_thresh']={}


#%%%%% Non-exclusion range
#    - in A
#    - outliers are automatically excluded before fitting the final model
#      we prevent this exclusion over the edges of the orders, where sharp variations are not well captured and can be attributed to outliers
#    - format is : {inst : [x1,x2] }  
gen_dic['gcal_nooutedge']={}


#%%%% Plots: instrumental calibration
#    - over each order and over time
plot_dic['gcal']=''
plot_dic['gcal_ord']=''



if __name__ == '__main__':

    #Calculating/retrieving
    gen_dic['calc_gcal']=True  &  False 

    #Bin size

    #Spectral bin size (in A)
    if gen_dic['star_name'] in ['WASP76','HD209458','HD29291','GJ436']:gen_dic['gcal_binw'] = 1.
    if gen_dic['star_name'] in ['WASP107','HAT_P11','WASP156']:gen_dic['gcal_binw'] = 2.
    if gen_dic['star_name']=='GJ3090':gen_dic['gcal_binw'] = 2.
    if gen_dic['star_name']=='55Cnc':gen_dic['gcal_binw'] = 0.2
    
    #Temporal bin size   
    if gen_dic['star_name']=='WASP156':gen_dic['gcal_binN'] = 2 


    #Edge polynomials 
    if gen_dic['star_name'] in ['GJ3090']:      
        gen_dic['gcal_edges']={'blue':0.3,'red':0.3}    
        gen_dic['gcal_deg']={'blue':4,'mid':2,'red':4}
    if gen_dic['star_name'] in ['55Cnc']:      
        gen_dic['gcal_edges']={'blue':0.08,'red':0.03}    
        gen_dic['gcal_deg']={'blue':4,'mid':4,'red':4}
        
    #Outliers     
        
    #Threshold
    if gen_dic['star_name'] in ['HD209458']:gen_dic['gcal_thresh'] = {'ESPRESSO':{'outliers':5.,'global':3e6}}
    elif gen_dic['star_name'] in ['WASP76','GJ436']:gen_dic['gcal_thresh'] = {'ESPRESSO':{'outliers':5.,'global':1.5e7}}
    elif gen_dic['star_name'] in ['HD29291']:gen_dic['gcal_thresh'] = {'ESPRESSO':{'outliers':5.,'global':1000}}
    if gen_dic['star_name'] in ['WASP107','HAT_P11','WASP156']:gen_dic['gcal_thresh'] = {'ESPRESSO':{'outliers':3.,'global':1000}}
    if gen_dic['star_name'] in ['GJ3090']:gen_dic['gcal_thresh'] = {'NIRPS_HA':{'outliers':3.,'global':1000},'NIRPS_HE':{'outliers':3.,'global':3000}}
    if gen_dic['star_name'] in ['55Cnc']:gen_dic['gcal_thresh'] = {'EXPRES':{'outliers':5.,'global':1.}}
    if user=='vaulato' and gen_dic['star_name'] in ['WASP189']:gen_dic['gcal_thresh'] = {'NIRPS_HE':{'outliers':3.,'global':3e6}} # vaulato
    
    #Non-exclusion range
    if gen_dic['star_name'] in ['WASP76','HD209458','HD29291','GJ436']:gen_dic['gcal_nooutedge']={'ESPRESSO':[2.,0.]}
    if gen_dic['star_name'] in ['WASP107','HAT_P11','WASP156']:gen_dic['gcal_nooutedge']={'CARMENES_VIS':[2.,2.]}
    if gen_dic['star_name'] in ['55Cnc']:gen_dic['gcal_nooutedge']={'EXPRES':[2.,2.]}
    
    #Plots: instrumental calibration
    plot_dic['gcal']=''
    plot_dic['gcal_ord']=''









##################################################################################################
#%%% Module: telluric correction
#    - use plot_dic['sp_raw'] to compare spectra before/after correction and identify orders in which tellurics are too deep and numerous to be well corrected, and that should be excluded from the entire analysis
##################################################################################################

#%%%% Activating
gen_dic['corr_tell']=True   


#%%%% Calculating/retrieving
gen_dic['calc_corr_tell']=True 


#%%%% Multi-threading
gen_dic['tell_nthreads'] =   14    


#%%%% Correction mode
#    - 'input' : if telluric spectra are contained in input files
#    - 'autom': automatic telluric correction
gen_dic['calc_tell_mode']='autom'      


#%%%% Telluric species
gen_dic['tell_species']=['H2O','O2']


#%%%% Orders to be fitted
#    - if left empty, all orders and the full spectrum is used
#    - format is {inst:{vis: [iord] }   
gen_dic['tell_ord_fit'] = {}


#%%%% Telluric CCF

#%%%%% Definition range
#    - in Earth rest frame, common to all molecule
#    - +-40 km/s if set to None
gen_dic['tell_def_range']= None


#%%%%% Continuum polynomial degree
#    - for each instrument > molecule, from 1 to 4 
#    - default: 0 for flat, constant continuum
gen_dic['tell_CCFcont_deg'] = {}   


#%%%%% Continuum range
#    - in Earth rest frame, for each molecule
#    - continuum range excludes +-15 km/s if undefined
gen_dic['tell_cont_range']={}


#%%%%% Fit range
#    - in Earth rest frame, for each molecule
#    - fit range set to the definition range if undefined
gen_dic['tell_fit_range']={}


#%%%% Fixed/variable properties
#    - structure is mod_prop = { inst : { vis : molec : { par_name : { 'vary' : bool , 'value':X , min:Y, max:Z } } } }        
#      leave empty the various fields to use default values
#    - see details in data_dic['DI']['mod_prop'] 
gen_dic['tell_mod_prop']={}


#%%%% Correction settings

#%%%%% Threshold 
#    - flux values where telluric are below this threshold (between 0 and 1) are set to nan
gen_dic['tell_thresh_corr'] = 0.1      


#%%%%% Exposures to be corrected
#    - leave empty for all exposures to be corrected
gen_dic['tell_exp_corr'] = {}


#%%%%% Orders to be corrected
#    - if left empty, all orders and the full spectrum is used
gen_dic['tell_ord_corr'] = {}


#%%%%% Spectral range(s) to be corrected
#    - if left empty, applied to the the full spectrum
gen_dic['tell_range_corr'] = {}


#%%%% Plot settings

#%%%%% Telluric CCFs (automatic correction)
plot_dic['tell_CCF']=''      


#%%%%% Fit results (automatic correction)
plot_dic['tell_prop']=''      


if __name__ == '__main__':

    #Activating
    gen_dic['corr_tell']=True     &  False
    if gen_dic['star_name'] in ['WASP76','HD209458','GJ436']:gen_dic['corr_tell']=True  
    if gen_dic['star_name'] in ['55Cnc']:gen_dic['corr_tell']=True 
    
    #Calculating/retrieving
    gen_dic['calc_corr_tell']=True  & False
    
    #Multi-threading
    gen_dic['tell_nthreads'] =  14    
    
    #Correction mode     
    if gen_dic['star_name'] in ['55Cnc','WASP76','HD209458','GJ436']:gen_dic['calc_tell_mode']='autom'   
    
    #Telluric species
    if gen_dic['star_name'] in ['WASP76','HD209458','GJ436']:gen_dic['tell_species']=['H2O','O2']  
    if user=='vaulato' and gen_dic['star_name'] in ['WASP189']:gen_dic['tell_species']=['H2O', 'O2', 'CH4', 'CO2'] #vaulato
    
    #Orders to be fitted
    if gen_dic['star_name']=='xxx': 
        gen_dic['tell_ord_fit'] = {'ESPRESSO':{'xxx':list(np.concatenate(([0,1],range(4,164),range(166,170))))}}    

    #Telluric CCF

    #Definition range
    gen_dic['tell_def_range']= None
    
    #Continuum polynomial degree
    gen_dic['tell_CCFcont_deg'] = {}   

    #Continuum range
    gen_dic['tell_cont_range']={}

    #Fit range
    tell_fit_range_H2O = [-30.,30.]
    tell_fit_range_O2 = [-25.,25.]
    if gen_dic['star_name']=='HD209458':
        gen_dic['tell_fit_range'] = {'ESPRESSO':{'20190720':{'H2O':tell_fit_range_H2O,'O2':tell_fit_range_O2},'20190911':{'H2O':tell_fit_range_H2O,'O2':tell_fit_range_O2}}}
    elif gen_dic['star_name']=='WASP76':
        gen_dic['tell_fit_range'] = {'ESPRESSO':{'20180902':{'H2O':tell_fit_range_H2O,'O2':tell_fit_range_O2},'20181030':{'H2O':tell_fit_range_H2O,'O2':tell_fit_range_O2}}}
    elif user=='vaulato' and gen_dic['star_name']=='WASP189':
        tell_fit_range_CH4 = [-30., 30.] #vaulato
        tell_fit_range_CO2 = [-30., 30.] #vaulato
        gen_dic['tell_fit_range'] = {'NIRPS_HE':{'20230604':{'H2O':tell_fit_range_H2O,'O2':tell_fit_range_O2, 'CH4':tell_fit_range_CH4, 'CO2':tell_fit_range_CO2}}} #vaulato 


    #Fixed/variable properties
    if gen_dic['star_name']=='XXX':          
        gen_dic['tell_mod_prop']={'inst' : { 'vis' : {
            'H2O':{'Temperature':{ 'vary' : False , 'value':300. , 'min':0., 'max':1000. } } } } }



    #Correction settings
    
    #Threshold 
    gen_dic['tell_thresh_corr'] = 0.1      
    
    #Exposures to be corrected
    gen_dic['tell_exp_corr'] = {}
    # print('ENLEVER')
    # gen_dic['tell_exp_corr'] = {'ESPRESSO':{'20180902':[35]}}    

    
    #Orders to be corrected
    gen_dic['tell_ord_corr'] = {}
    
    #Spectral range(s) to be corrected
    gen_dic['tell_range_corr'] = {}


    #Plot settings

    #Telluric CCFs (automatic correction)
    plot_dic['tell_CCF']=''  #      

    #Fit results (automatic correction)
    plot_dic['tell_prop']=''  #    





 
        






##################################################################################################
#%%% Modules: flux balance corrections
##################################################################################################

#%%%% Multi-threading
gen_dic['Fbal_nthreads'] = 14    


##################################################################################################
#%%%% Module: stellar masters
##################################################################################################

#%%%%% Activating  
gen_dic['glob_mast']=True    


#%%%%% Calculating/retrieving
gen_dic['calc_glob_mast']=True     


#%%%%% Measured masters
#    - calculated by default for each visit, and over all visits of the same instrument if gen_dic['Fbal_vis']=='meas'  
#    - these masters are not used for other modules since they are calculated from raw uncorrected data
#      they are calculated automatically for flux balance corrections, or on request for preliminary visualization in plots 

#%%%%%% Mean ('mean') or median ('med') 
#    - mode of master calculation
gen_dic['glob_mast_mode']='med'    


#%%%%%% Exposures used in master calculations
#    - set 'all' (default if left empty) or a list of exposures
gen_dic['glob_mast_exp'] = {}       
       

#%%%%% Theoretical masters
#    - format is {inst:{vis:path}}
#    - set path to spectrum file (two columns: wavelength in star rest frame in A, flux density in arbitrary units)
#      spectrum must be defined over a larger range than the processed spectra
#    - only required if gen_dic['Fbal_vis']=='theo', to reset all spectra from different instruments to a common balance, or to reset spectra from a given visit 
# to a specific stellar balance in a given epoch
gen_dic['Fbal_refFstar']= {}  


#%%%%% Plots: masters
plot_dic['glob_mast']=''     



if __name__ == '__main__':

    #Activating  
    gen_dic['glob_mast']=True  #   &  False     
    
    #Calculating/retrieving
    gen_dic['calc_glob_mast']=True  &  False    

    
    #Multi-threading
    gen_dic['Fbal_nthreads'] = 4   
    
    #Measured masters
    
    #Mean ('mean') or median ('med')   
    if gen_dic['star_name'] in ['WASP76','HD209458','55Cnc','HD29291']:
        gen_dic['glob_mast_mode']='med'        #ANTARESS I
    
    #Exposures used in master calculations
    # if gen_dic['star_name']=='WASP76':
    #     # gen_dic['glob_mast_exp']={'ESPRESSO':{'20180902':np.concatenate((range(0,5),range(26,35))),'20181030':np.concatenate((range(0,13),range(52,69)))}}
    #     gen_dic['glob_mast_exp']={'ESPRESSO':{'20180902':[27],'20181030':[53]}}        
           
    
    #Theoretical masters
    if gen_dic['star_name']=='WASP76':
        gen_dic['Fbal_refFstar']= {'ESPRESSO':{vis:'/Travaux/ANTARESS/En_cours/WASP121b_Phoenix_spec.dat' for vis in ['2018-09-03','2018-10-31']}}   
        gen_dic['Fbal_refFstar']= {}      #ANTARESS I
    if gen_dic['star_name']=='HD209458':        
        gen_dic['Fbal_refFstar']= {}      #ANTARESS I    
    
    
    #Plots: masters
    plot_dic['glob_mast']=''   #pdf  









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
# + 'theo': is applied toward the theoretical input spectrum provided via gen_dic['Fbal_refFstar'] 
#    - the latter option allows accounting for variations on the global stellar balance between visits, and is otherwise necessary to set spectra from different instruments (ie, with different coverages) to the same balance 
gen_dic['Fbal_vis']='meas'  


#%%%%% Fit settings 

#%%%%%% Spectral range(s) to be fitted
#    - even if a localized region is studied afterward, the flux balance should be corrected over as much as possible of the spectrum
#      however the module can also be used to correct locally (ie in the region of a single absorption line) for the spectral flux balance
#    - if left empty, all orders and the full spectrum is used
gen_dic['Fbal_range_fit'] = {}


#%%%%%% Orders to be fitted
gen_dic['Fbal_ord_fit'] = {}
      
        
#%%%%%% Spectral bin size
#    - bin size of the fitted data (in 1e-10 s-1)
# dnu[1e10 s1] = c[m s-1]*dw[A]/w[A]^2
#      for ESPRESSO dnu < 0.9 (0.5) yields more than 1 (2) bins in most orders
#    - for the correction relative to measured visit masters: binning is applied over each order (set a value larger than an order width to bin over the entire order)
#      for the correction relative to reference masters, binning is applied over full orders by default
gen_dic['Fbal_bin_nu'] = 1.         
    

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
gen_dic['Fbal_deg'] ={}
gen_dic['Fbal_deg_vis'] ={}


#%%%%%% Spline smoothing factor
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



if __name__ == '__main__':

    #Activating
    gen_dic['corr_Fbal']=True     &  False
    if gen_dic['star_name'] in ['WASP76','HD209458','WASP107','HAT_P11','WASP156','55Cnc','GJ3090','HD29291','GJ436']:gen_dic['corr_Fbal']=True # &  False
    
    #Calculating/retrieving
    gen_dic['calc_corr_Fbal']=True   & False


    #Reference master  
    if gen_dic['star_name']=='WASP76':    
        gen_dic['Fbal_vis'] = 'meas'   
    elif gen_dic['star_name']=='HD209458':    
        gen_dic['Fbal_vis'] = 'meas'
    elif user=='vaulato' and gen_dic['star_name']=='WASP189':    
        gen_dic['Fbal_vis'] = None # vaulato: None for WASP-189 because only one instrument (NIRPS) and only one dataset (2023-06-04)

    #Fit settings 
    
    #Spectral range(s) to be fitted
    # gen_dic['Fbal_range_fit'] = [ [4000.,4500.],[4800.,5000.],[5200.,5300.] ] 
    # gen_dic['Fbal_range_fit'] = [ [3750.,5160.],[5250.,6800.],[6950.,7150.],[7300.,7550.],[7700.,8050.] ]
    # # gen_dic['Fbal_range_fit'] = [       # Removal of noisy blue end of spectrum
    # #             [3800.,3932.2],     # Stellar Ca II K line
    # #             [3935.2,3967.0],    # Stellar Ca II H line
    # #             [3970.0,4859.8],    # Stellar H-beta
    # #             [4862.8,5871.],     # H2O band, stellar/telluric Na I D lines
    # #             [6005.,6271.],      # O2 gamma band, telluric O I emission line
    # #             [6341.,6449.],      # H2O band, stellar H-alpha
    # #             [6605.,6861.],      # O2 B band and H2O band
    # #             [7416.,7588.],      # O2 A band
    # #             [7751.,7860.]]      # H2O band  
    # gen_dic['Fbal_range_fit'] = [ [3750.,5228.],[5230.,7950.] ]   
    # gen_dic['Fbal_range_fit'] = [ [3750.,7950.] ]   
    if gen_dic['star_name']=='HD3167':     
        gen_dic['Fbal_range_fit'] = {'ESPRESSO':{'2019-10-09':[ [3750.,7950.] ] }, 
                                   'HARPN':{'2016-10-01':[ [3900.,7000.] ]  }}      
    if 'Moon' in gen_dic['transit_pl']:gen_dic['Fbal_range_fit'] = {'HARPS':{'2019-07-02':[ [3800.,7000.] ]}}       

    # if gen_dic['star_name']=='WASP76':
    #     #ANTARESS I
    #     gen_dic['Fbal_range_fit'] = {'ESPRESSO':
    #             {'20180902':[ [3750.,7950.] ] ,
    #               '20181030':[ [3750.,7950.] ] }}              
    if gen_dic['star_name']=='HD209458':    
        gen_dic['Fbal_range_fit'] = {'ESPRESSO':
                {'20190720':[ [3750.,7950.] ] ,
                 '20190911':[ [3750.,7950.] ] }}
    elif gen_dic['star_name']=='55Cnc':    
        gen_dic['Fbal_range_fit'] = {'ESPRESSO':{'20200205':[ [3900.,7950.] ],'20210121':[ [3900.,7950.] ],'20210124':[ [3900.,7950.] ]},
                                     'EXPRES':{'20220406':[ [3500.,9000.] ]}
                }
    elif gen_dic['star_name']=='WASP107':    
        gen_dic['Fbal_range_fit'] = {'CARMENES_VIS':{'20180224':[ [1000.,6865.], [6940.,7590.],[7695.,12000.] ]}}
    elif gen_dic['star_name']=='HAT_P11':    
        gen_dic['Fbal_range_fit'] = {'CARMENES_VIS':{'20170807':[ [1000.,7590.], [7695.,12000.] ]}}
        gen_dic['Fbal_range_fit']['CARMENES_VIS']['20170812'] = gen_dic['Fbal_range_fit']['CARMENES_VIS']['20170807'] 
    elif gen_dic['star_name']=='WASP156':    
        gen_dic['Fbal_range_fit'] = {'CARMENES_VIS':{'20190928':[ [1000.,7590.], [7695.,12000.] ]}}
        gen_dic['Fbal_range_fit']['CARMENES_VIS']['20191025'] = gen_dic['Fbal_range_fit']['CARMENES_VIS']['20190928']
        gen_dic['Fbal_range_fit']['CARMENES_VIS']['20191210'] = gen_dic['Fbal_range_fit']['CARMENES_VIS']['20190928']
    elif user=='vaulato' and gen_dic['star_name']=='WASP189':   #vaulato 
        gen_dic['Fbal_range_fit'] = {'NIRPS_HE':{'20230604':[ [7000., 23000.] ]}}

    #Orders to be fitted
    if gen_dic['star_name']=='HD3167': 
        gen_dic['Fbal_ord_fit'] = {'ESPRESSO':{'2019-10-09':list(np.concatenate(([0,1],range(4,164),range(166,170))))}, 
                                   'HARPN':{'2016-10-01':list(np.concatenate(([0,1],range(3,6),range(7,14),range(15,70))))}}  
    if 'Moon' in gen_dic['transit_pl']:gen_dic['Fbal_ord_fit'] = {'HARPS':{'2020-12-14':list(np.concatenate(([0,1,2,3,4],range(6,62),range(63,70))))}} 
    if gen_dic['star_name']=='WASP76':
    #     # #dbin 60 (half order)
    #     # gen_dic['Fbal_ord_fit'] = {'ESPRESSO':
    #     #         {'20180902':list(np.concatenate((range(162),range(166,170)))), 
    #     #           '20181030':list(np.concatenate((range(162),range(166,170))))}} 
        #dbin 150 (full order),   #ANTARESS I   
        # gen_dic['Fbal_ord_fit'] = {'ESPRESSO':
        #         {'20180902':list(np.concatenate((range(146),range(148,162),range(166,170)))), 
        #           '20181030':list(np.concatenate((range(146),range(148,162),range(166,170))))}}
        gen_dic['Fbal_ord_fit'] = {
            'ESPRESSO':{
                # {'20180902':list(np.delete(np.arange(170),[90,91, 146,147,162,163,164,165])),
                # '20181030':list(np.delete(np.arange(170),[90,91, 146,147,162,163,164,165]))}}
                # {'20180902':list(np.delete(np.arange(170),[110,111,112,113,146,147,162,163,164,165])),
                # '20181030':list(np.delete(np.arange(170),[146,147,162,163,164,165]))}
                # '20180902':list(np.delete(np.arange(170),[88,89])),
                }}   

    if gen_dic['star_name']=='HD209458':         #ANTARESS I, final version: no exclusion needed   
    # #     gen_dic['Fbal_ord_fit'] = {'ESPRESSO':
    # #             {'20190720':list(np.delete(np.arange(170),[78,79,80,81,82,83,84,85,86,87,88,89,   146,147,  162,163,  164,165]))}}
    # #     gen_dic['Fbal_ord_fit']['ESPRESSO']['20190911']=gen_dic['Fbal_ord_fit']['ESPRESSO']['20190720']   

        #dbin 150 (full order), 
        gen_dic['Fbal_ord_fit'] = {
            'ESPRESSO':{
                # {'20190720':list(np.concatenate((range(14),range(16,146),range(148,162),range(166,170)))), 
                #   '20190911':list(np.concatenate((range(14),range(16,146),range(148,162),range(166,170))))}}
                # {'20190720':list(np.delete(np.arange(170),[146,147,162,163,164,165])),
                # '20190911':list(np.delete(np.arange(170),[146,147,162,163,164,165]))}
                # '20190720':list(np.delete(np.arange(170),[0,1]))}
                }}   

    
    if gen_dic['star_name']=='55Cnc':
        gen_dic['Fbal_ord_fit'] = {'ESPRESSO':
                {'20200205':list(np.concatenate((range(24),range(32,88),range(92,164),range(166,170)))),
                 '20210121':list(np.concatenate((range(88),range(92,164),range(166,170)))),
                 '20210124':list(np.concatenate((range(88),range(92,164),range(166,170))))}
                }            
            
    if gen_dic['star_name'] in ['WASP107']:
        gen_dic['Fbal_ord_fit'] = {'CARMENES_VIS':{'20180224':list(range(1,57))}}        
    if gen_dic['star_name'] in ['HAT_P11']:
        gen_dic['Fbal_ord_fit'] = {'CARMENES_VIS':{'20170807':list(range(57)),'20170812':list(range(57))}}               
    if gen_dic['star_name'] in ['WASP156']:
        gen_dic['Fbal_ord_fit'] = {'CARMENES_VIS':{'20190928':list(range(57)),'20191025':list(range(57)),'20191210':list(range(57))}} 
    # if gen_dic['star_name'] in ['GJ3090']:
    #     gen_dic['Fbal_ord_fit'] = {'NIRPS_HE':{'20221201':list(np.concatenate((range(41),range(44,68))))},
    #                                'NIRPS_HA':{'20221202':list(np.concatenate((range(41),range(44,67))))}}
    elif gen_dic['star_name']=='HD29291':
          gen_dic['Fbal_ord_fit'] = {'ESPRESSO':{'20201130':list(np.concatenate((range(88),range(92,164),range(166,170))))}}                          
            

    #Spectral bin size
    if gen_dic['star_name'] in ['HD209458','55Cnc','HD29291']:
        gen_dic['Fbal_bin_nu'] = 0.7         #ANTARESS I final version (nu space)
    if gen_dic['star_name'] =='WASP76':
         gen_dic['Fbal_bin_nu'] = 1.   #ANTARESS I final version (nu space)                
    if gen_dic['star_name'] in ['WASP107']:
        gen_dic['Fbal_bin_nu'] = 50.  
        gen_dic['Fbal_bin_nu'] = 100. 
        gen_dic['Fbal_bin_nu'] = 200.         
    if gen_dic['star_name'] in ['HAT_P11']:   
        gen_dic['Fbal_bin_nu'] = 50.  
        gen_dic['Fbal_bin_nu'] = 100. 
        gen_dic['Fbal_bin_nu'] = 200. 
    if gen_dic['star_name'] in ['WASP156']:   
        gen_dic['Fbal_bin_nu'] = 50.  
        gen_dic['Fbal_bin_nu'] = 200. 
    if gen_dic['star_name'] in ['GJ3090']:   
        gen_dic['Fbal_bin_nu'] = 500. 
    if user=='vaulato' and gen_dic['star_name'] in ['WASP189']:   #vaulato
        gen_dic['Fbal_bin_nu'] = 5000. 

        
    #Uncertainty scaling
    if gen_dic['star_name'] in ['WASP76','HD209458']:gen_dic['Fbal_expvar'] = 1./4.       
    if gen_dic['star_name'] in ['GJ436']:gen_dic['Fbal_expvar'] = 1./4.   
    
    #Automatic sigma-clipping
    gen_dic['Fbal_clip'] = True    & False
        

    #Flux balance model
        
    #Model
    if gen_dic['star_name'] in ['WASP76','HD209458','55Cnc','GJ436']:
        gen_dic['Fbal_mod']='spline'        #ANTARESS I   
    if gen_dic['star_name'] in ['WASP107','HAT_P11','WASP156','WASP189']:gen_dic['Fbal_mod']='pol'             

    #Polynomial degree 
    # if gen_dic['star_name']=='WASP76':    
    #     gen_dic['Fbal_deg'] = {'ESPRESSO':{'20180902':6 ,'20181030':6 }}
    if gen_dic['star_name']=='HD3167':     
        gen_dic['Fbal_deg'] = {'ESPRESSO':{'2019-10-09':6}, 
                                   'HARPN':{'2016-10-01':6  }}        
    elif 'Moon' in gen_dic['transit_pl']:gen_dic['Fbal_deg'] = {'HARPS':{'2019-07-02':10,'2020-12-14':10}}
    elif gen_dic['star_name']=='HD209458':    
        gen_dic['Fbal_deg'] = {'ESPRESSO':{'20190720':4 ,'20190911':4 }}
        gen_dic['Fbal_deg_vis'] = {'ESPRESSO':{'20190720':6 ,'20190911':6 }}
    elif gen_dic['star_name']=='WASP76':    
        gen_dic['Fbal_deg'] = {'ESPRESSO':{'20180902':4 ,'20181030':4 }}
        gen_dic['Fbal_deg_vis'] = {'ESPRESSO':{'20180902':6 ,'20181030':6 }}
    elif gen_dic['star_name']=='55Cnc':    
        gen_dic['Fbal_deg'] = {'ESPRESSO':{'20200205':4,'20210121':6,'20210124':6}}
    elif gen_dic['star_name'] in ['WASP107']:gen_dic['Fbal_deg']={'CARMENES_VIS':{'20180224':6}}  
    elif gen_dic['star_name'] in ['HAT_P11']:gen_dic['Fbal_deg']={'CARMENES_VIS':{'20170807':4,'20170812':4}}  
    elif gen_dic['star_name'] in ['WASP156']:
        gen_dic['Fbal_deg'] = {'CARMENES_VIS':{'20190928':4,'20191025':6,'20191210':6}} 
    elif gen_dic['star_name'] in ['GJ3090']:
        gen_dic['Fbal_deg'] = {'NIRPS_HE':{'20221201':3},'NIRPS_HA':{'20221202':3}}     
    elif gen_dic['star_name']=='HD29291':    
        gen_dic['Fbal_deg'] = {'ESPRESSO':{'20201130':4}}    
    elif user=='vaulato' and gen_dic['star_name']=='WASP189':    #vaulato
        gen_dic['Fbal_deg'] = {'NIRPS_HE':{'20230604':6}}     
    
    #Spline smoothing factor
    if gen_dic['star_name']=='WASP76':    
        gen_dic['Fbal_smooth_vis']={'ESPRESSO':{'20180902':8e-5  ,'20181030':2e-4  }}  #ANTARESS I, final 
        # gen_dic['Fbal_smooth'] = {'ESPRESSO':{'20180902':6e-5  ,'20181030':7e-5  }}    
        gen_dic['Fbal_smooth'] = {'ESPRESSO':{'20180902':6e-5  ,'20181030':7e-5  }}    #ANTARESS I, fit vs nu, weight 0.25   
    elif gen_dic['star_name']=='HD209458':    
        gen_dic['Fbal_smooth_vis'] = {'ESPRESSO':{'20190720':1e-5  ,'20190911':2e-5  }}   #ANTARESS I, final    
        gen_dic['Fbal_smooth'] = {'ESPRESSO':{'20190720':2e-4  ,'20190911':3e-4  }}  
        gen_dic['Fbal_smooth'] = {'ESPRESSO':{'20190720':7e-5  ,'20190911':7e-5  }}    #ANTARESS I, fit vs nu, weight 0.25     
  
    elif gen_dic['star_name']=='55Cnc':          
        gen_dic['Fbal_smooth'] = {'ESPRESSO':{'20210121':8e-4,'20210124':8e-4}}        
    if gen_dic['star_name']=='GJ436':    
        gen_dic['Fbal_smooth']={'ESPRESSO':{'20190228':2e-3  ,'20190429':6e-4  }}  
        gen_dic['Fbal_smooth_vis']={'ESPRESSO':{'20190228':1.7e-4  ,'20190429':1.2e-4  }}  

    #Spectral range(s) to be corrected
    gen_dic['Fbal_range_corr'] = [ [3000.,9000.] ]           
    if user=='vaulato':gen_dic['Fbal_range_corr'] = [] #vaulato  
    
    
    #Plot settings
    
    #Exposures/visit balance 
    plot_dic['Fbal_corr']=''  #'pdf

    #Exposures/visit balance (DRS)
    plot_dic['Fbal_corr_DRS']=''  #pdf

    #Measured/reference visit balance
    plot_dic['Fbal_corr_vis']=''  #pdf







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



if __name__ == '__main__':

    #Activating
    gen_dic['corr_FbalOrd']=True     &  False
    if gen_dic['star_name'] in ['WASP76','HD209458','55Cnc']:gen_dic['corr_FbalOrd']=True    &   False   #ANTARESS I

    #Calculating/retrieving
    gen_dic['calc_corr_FbalOrd']=True    &  False 

    #Model polynomial degree 
    if gen_dic['star_name']=='WASP76':    
        gen_dic['Fbal_deg_ord'] = {'ESPRESSO':{'20180902':4 ,'20181030':4 }}


    #Spectral range(s) to be fitted
    gen_dic['FbalOrd_range_fit'] = []
    
    #Orders to be fitted
    gen_dic['FbalOrd_ord_fit'] = {}

    #Spectral bin size
    gen_dic['Fbal_binw_ord'] = 2.

    #Automatic sigma-clipping
    gen_dic['Fbal_ord_clip'] = True

    #Plots: flux balance correction 
    plot_dic['Fbal_corr_ord']=''  #'png
    














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


if __name__ == '__main__':

    #Activating
    gen_dic['corr_Ftemp']=True    &  False

    #Calculating/retrieving
    gen_dic['calc_corr_Ftemp']=True    &  False  

    #Spectral range(s) to be fitted
    gen_dic['Ftemp_range_fit'] = {}

    #Exposures excluded from the fit
    gen_dic['idx_nin_Ftemp_fit']={}

    #Model polynomial degree 
    if gen_dic['star_name']=='HD209458':    
        gen_dic['Ftemp_deg'] = {'ESPRESSO':{'20190720':6 ,'20190911':6 }}

    #Plots: temporal flux correction
    plot_dic['Ftemp_corr']=''  #'png
















##################################################################################################
#%%% Module: cosmics correction
##################################################################################################

#%%%% Activating
gen_dic['corr_cosm']=True   


#%%%% Calculating/retrieving
gen_dic['calc_cosm']=True    
    

#%%%% Multi-threading
gen_dic['cosm_nthreads'] = 14     


#%%%% Alignment mode
#    - choose option to align spectra prior to cosmic identification and correction
# + 'kep': Keplerian curve 
# + 'pip': pipeline RVs (if available)
# + 'autom': for automatic alignment using the specified options 'range' and 'RVrange_cc'
#            'range' : define the spectral range(s) over which spectra are aligned
#                      use a large range for increased precision, at the cost of computing time
#                      set to [] to use the full spectrum
#            'RVrange_cc' : define the RV range and step used to cross-correlate spectra
#                           should cover the maximum velocity shift between two exposures in the visit
#    - the Keplerian option should be preferred, as the others will be biased by the RM effect 
gen_dic['al_cosm']={'mode':'kep'}


#%%%% Comparison spectra
#    - define the number of spectra around each exposure used to identify and replace cosmics
gen_dic['cosm_ncomp'] = 6    


#%%%% Outlier threshold  
gen_dic['cosm_thresh'] = {} 

    
#%%%% Exposures to be corrected
#    - leave empty for all exposures to be corrected
gen_dic['cosm_exp_corr']={}
        
        
#%%%% Orders to be corrected
#    - leave empty for all orders to be corrected
gen_dic['cosm_ord_corr']={}
    

#%%%% Plots:cosmics
plot_dic['cosm_corr']=''    


if __name__ == '__main__':

    #Activating
    gen_dic['corr_cosm']=True     &  False
    if gen_dic['star_name'] in ['WASP76','HD209458','WASP107','HAT_P11','WASP156','GJ3090','HD29291','55Cnc','GJ436']:gen_dic['corr_cosm']=True # & False

    #Calculating/retrieving
    gen_dic['calc_cosm']=True   &  False  

    #Alignment mode
    gen_dic['al_cosm']={
        'mode':'kep',

        # 'mode':'pip',

        # 'mode':'autom',
        # 'range' : [ [6000.,6500.]] ,
        #'RVrange_cc' : [-4.,5.,0.5],  

    }


    #Comparison spectra   
    if gen_dic['star_name']in ['WASP76','HD209458','WASP107','HAT_P11','WASP156','GJ3090','HD29291','55Cnc']:gen_dic['cosm_ncomp'] = 10 
    
    #Outlier threshold 
    if gen_dic['star_name']=='HD3167':     
        gen_dic['cosm_thresh'] = {'ESPRESSO':{'2019-10-09':5}, 
                                   'HARPN':{'2016-10-01':5  }} 
    elif 'Moon' in gen_dic['transit_pl']:gen_dic['cosm_thresh'] = {'HARPS':{'2019-07-02':5,'2020-12-14':5}}
    elif gen_dic['star_name']=='WASP76':    
        gen_dic['cosm_thresh'] = {'ESPRESSO':{'20180902':5 ,'20181030':5}}
        gen_dic['cosm_thresh'] = {'ESPRESSO':{'20180902':10 ,'20181030':10}}   #ANTARESS I
    elif gen_dic['star_name']=='HD209458':    
        gen_dic['cosm_thresh'] = {'ESPRESSO':{'20190720':10,'20190911':10}}    #ANTARESS I
    elif gen_dic['star_name']=='WASP107':    
        gen_dic['cosm_thresh'] = {'CARMENES_VIS':{'20180224':7}}
    elif gen_dic['star_name']=='HAT_P11':    
        gen_dic['cosm_thresh'] = {'CARMENES_VIS':{'20170807':7,'20170812':7}}
    elif gen_dic['star_name']=='WASP156':    
        gen_dic['cosm_thresh'] = {'CARMENES_VIS':{'20190928':7,'20191025':14,'20191210':7}}
    elif gen_dic['star_name']=='GJ3090':    
        gen_dic['cosm_thresh'] = {'NIRPS_HE':{'20221201':10},'NIRPS_HA':{'20221202':10}}        
    elif gen_dic['star_name']=='HD29291':    
        gen_dic['cosm_thresh'] = {'ESPRESSO':{'20201130':10}}
    elif gen_dic['star_name']=='55Cnc':    
        gen_dic['cosm_thresh'] = {'EXPRES':{'20220131':10,'20220406':10}} 
    elif user=='vaulato' and gen_dic['star_name']=='WASP189':    #vaulato
        gen_dic['cosm_thresh'] = {'NIRPS_HE':{'20230604':10}} 
        
    #Exposures to be corrected
    # if gen_dic['star_name']=='WASP76':
    #     gen_dic['cosm_exp_corr'] = {'ESPRESSO':
    #             {'20180902':[] ,
    #              '20181030':[] }} 
    if gen_dic['star_name']=='WASP107':
        gen_dic['cosm_exp_corr'] = {'CARMENES_VIS':
                {'20180224':list(range(19)) }} 
            
            
    #Orders to be corrected
    # if gen_dic['star_name']=='WASP76':
    #     gen_dic['cosm_ord_corr'] = {'ESPRESSO':
    #             #Trop de telluriques profondes, mal corriges, cree des features spurieuses  -> ok avec corr tell.
    #             {'20180902':list(np.concatenate((range(162),range(166,170)))),
    #              '20181030':list(np.concatenate((range(162),range(166,170))))}} 
        
    #Plots:cosmics
    plot_dic['cosm_corr']='' #png      












##################################################################################################
#%%% Module: persistent peak masking
##################################################################################################

#%%%% Activating
gen_dic['mask_permpeak']= False


#%%%% Calculating/retrieving 
gen_dic['calc_permpeak']=True  


#%%%% Multi-threading
gen_dic['permpeak_nthreads'] = 14


#%%%% Correction settings

#%%%%% Exposures to be corrected
#    - format inst > vis > exp_list
#    - leave empty for all exposures to be corrected
gen_dic['permpeak_exp_corr']={}

#%%%%% Orders to be corrected
#    - format inst > vis > ord_list
#    - leave empty for all orders to be corrected
gen_dic['permpeak_ord_corr']={}


#%%%%% Spectral range(s) to be corrected
#    - inst > order > [[x1,x2],[x3,x4], ..]
#    - leave undefined or empty to take the full range 
gen_dic['permpeak_range_corr'] = {}

    
#%%%%% Non-masking of edges
#    - in A
#    - we prevent masking over the edges of the orders, where the continuum is often not well-defined
gen_dic['permpeak_edges']={}


#%%%% Peaks exclusion settings

#%%%%% Spurious peaks threshold 
#    - set on the residuals from continuum, compared to their error
gen_dic['permpeak_outthresh']=4


#%%%%% Spurious peaks window
#    - in A
gen_dic['permpeak_peakwin']={}  


#%%%%% Bad consecutive exposures 
#    - a peak is masked if it is flagged in at least max(permpeak_nbad,3) consecutive exposures
gen_dic['permpeak_nbad']=3 


#%%%% Plots: master and continuum
#    - to check the flagged pixels use plot_dic['sp_raw'] before/after correction
plot_dic['permpeak_corr']=''  


if __name__ == '__main__':

    #Activating
    gen_dic['mask_permpeak']=True    &  False
    if gen_dic['star_name'] in ['WASP107','WASP156']:gen_dic['mask_permpeak']=True   
    if gen_dic['star_name'] in ['HAT_P11']:gen_dic['mask_permpeak']=False     
    if gen_dic['star_name']in ['WASP76','HD209458']:gen_dic['mask_permpeak']=True   & False  
    
    #Calculating/retrieving 
    gen_dic['calc_permpeak']=True   & False

    #Correction settings

    #Exposures to be corrected
    gen_dic['permpeak_exp_corr']={}

    #Orders to be corrected
    if gen_dic['star_name']=='WASP107':    
        gen_dic['permpeak_ord_corr']={'CARMENES_VIS':{'20180224':list( np.delete( np.arange(61) , [26,27,33,38] )  )}}
    if gen_dic['star_name']=='WASP156':    
        gen_dic['permpeak_ord_corr']={'CARMENES_VIS':{'20190928':list( np.delete( np.arange(61) , [23,37,38,40,46,51,54] )  ),
                                                      '20191025':list( np.delete( np.arange(61) , [25,30,33,34,37,38,43,51,55] )  ),
                                                      '20191210':list( np.delete( np.arange(61) , [25,37,38,40,44,46,53,58,59] )  ),
                                                      }}

    #Spectral range(s) to be corrected
    if gen_dic['star_name']=='WASP107':
        gen_dic['permpeak_range_corr']={'CARMENES_VIS':{18:[[6065,6160]]}}
    if gen_dic['star_name']=='WASP156':
        gen_dic['permpeak_range_corr']={'CARMENES_VIS':{6:[[5420.,5470.]]}}

        
    #Non-masking of edges
    if gen_dic['star_name'] in ['HAT_P11','WASP156']:gen_dic['permpeak_edges']={'CARMENES_VIS':[2.,2.]}
    gen_dic['permpeak_edges']={'ESPRESSO':[2.,2.]}



    #Peaks exclusion settings
    
    #Spurious peaks threshold 
    if gen_dic['star_name'] in ['WASP107']:gen_dic['permpeak_outthresh']=4
    if gen_dic['star_name'] in ['HAT_P11']:gen_dic['permpeak_outthresh']=5
    if gen_dic['star_name'] in ['WASP156']:gen_dic['permpeak_outthresh']=4
    if gen_dic['star_name'] in ['WASP76','HD209458']:gen_dic['permpeak_outthresh']=10
    
    #Spurious peaks window
    gen_dic['permpeak_peakwin']={'CARMENES_VIS':0.2,'ESPRESSO':0.2}  


    #Bad consecutive exposures 
    if gen_dic['star_name'] in ['WASP107']:gen_dic['permpeak_nbad']=2   
    if gen_dic['star_name'] in ['HAT_P11','WASP156']:gen_dic['permpeak_nbad']=3  
    

    
    #Plots: master and continuum
    #    - to check the flagged pixels use plot_dic['sp_raw'] before/after correction
    plot_dic['permpeak_corr']='' #  









##################################################################################################
#%%% Module: ESPRESSO "wiggles"
#    - run sequentially those steps :
# + 'wig_exp_init' to identify which ranges to include in the analysis, and which wiggle components are present
# + 'wig_exp_samp' to sample the frequency and amplitude of each wiggle component with nu, in a representative selection of exposure
# + 'wig_exp_nu_ana' to fit the sampled frequency and amplitude with polynomials of nu, to evaluate their degree and chromatic coefficients
# + 'wig_exp_fit' to fit the spectral wiggle model to each exposure individually, initialized by the results of 'wig_exp_nu_ana'   
# + 'wig_exp_point_ana' to fit the phase, and the chromatic coefficients of frequency and amplitude derived from 'wig_exp_fit', as a function of the telescope pointing coordinates 
# + 'wig_vis_fit' to fit the spectro-temporal wiggle model to all exposures together, initialized by the results of 'wig_exp_point_ana'
#    - wiggles are processed in wave_number space nu[1e-10 s-1] = c[m s-1]/w[A]
#      wiggle frequencies Fnu corresponds to wiggle periods Pw[A] = w[A]^2/(Fnu[1e10 s]*c[m s-1])
##################################################################################################


#%%%% Activating 
gen_dic['corr_wig']= False


#%%%% Calculating/retrieving
gen_dic['calc_wig']=True   


#%%%% Guide shift reset
#    - disable automatic reset of wiggle properties following guide star shift, for the chosen list of visits
gen_dic['wig_no_guidchange'] = []   


#%%%% Forced order normalization
#    - transmission spectrum is normalized to unity over each order 
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
    
    
#%%%% Fit settings        
    
#%%%%% Exposures to be fitted
#    - instrument > visit
#    - set to 'all' to use all exposures
gen_dic['wig_exp_in_fit'] = {}


#%%%%% Groups of exposures to be fitted together
#    - leave empty to perform the fit on individual exposures
#    - this is useful to boost SNR, especially in the bluest orders, without losing the spectral resolution over orders
#      beware however that wiggles amplitude and offsets change over time, and will thus be blurred by this average
gen_dic['wig_exp_groups']={}
    

#%%%%% Spectral range(s) to be fitted
#    - if left empty, the full spectrum is used
#    - units are c/w (10-10 s-1)
gen_dic['wig_range_fit'] = []
    
    
#%%%%% Spectral bin size
#    - all spectra are binned prior to the analysis
#    - in nu space (1e-10 s-1), with dnu[1e10 s1] = c[m s-1]*dw[A]/w[A]^2   
#    - bin size should be small enough to sample the period of the wiggles, but large enough to limit computing time and remove possible correlations between bins 
# + for the two dominant wiggle component set dw = 2 A, ie dnu = 0.0166 (mind that it creates strong peaks in the periodograms at F = 60, 120, 180)
# + for the mini-wiggle component set dw = 0.05 A, ie dnu = 0.0004 (mind that it creates a signal in the periodograms at F = 2400)
#    - even if the spectra are analyzed at their native resolution it is necessary to resample them to merge overlapping bands at order edges
gen_dic['wig_bin'] = 0.0166   
    

#%%%%% Orders to be fitted
#    - if left empty, all orders are used 
gen_dic['wig_ord_fit'] = {}


#%%%% Fitting steps

#%%%%% Step 1: Screening 
#    - use 'plot_spec' to identify which ranges / orders are of poor quality and need to be excluded from the fit and/or the correction
#      use as much of the ESPRESSO range as possible, but exclude the bluest orders where the noise is much larger than the wiggles
#    - use 'plot_hist' to plot the periodogram from all exposures together, to identify the number and approximate frequency of wiggle components 
gen_dic['wig_exp_init']={
    'mode':True  ,
    'plot_spec':True,
    'plot_hist':True,
    'y_range':None
    }


#%%%%% Step 2: Chromatic sampling
#    - sampling the frequency and amplitude of each wiggle component with nu
#    - wiggle properties are sampled using a sliding periodogram
#    - only a representative subset of exposures needs to be sampled, using 'wig_exp_in_fit'
#    - set 'comp_ids' between 1 and 5
#      only the component with highest 'comp_ids' is sampled using all shifts in 'sampbands_shifts'
#      lower components are fitted with a single shift from 'sampbands_shifts', chosen through 'direct_samp'
#      thus, start by smapling the highest component, and proceed by including lower ones iteratively
#    - 'freq_guess': define the polynomial coefficients describing the model frequency for each component 
#                    these models control the definition of the sampling bands 
#    - 'nsamp' : number of cycles to sample for each component
#                must not be too high to ensure that the component frequency remains constant within the sampled bands 
#    - 'sampbands_shifts': oversampling of sampling bands (nu in 1e-10 s-1)
#                          adjust to the scale of the frequency or amplitude variations of each component
#                          set to [None] to prevent sampling (the full spectrum is fitted with one model)
#                          to estimate size of shifts: nu = c/w -> dnu = cdw/w^2 
#    - 'direct_samp': set direct_samp[comp_id] to the index of sampbands_shifts[comp_id-1] for which the sampling of comp_id should be applied
#    - 'src_perio': frequency ranges within which periodograms are searched for each component (in 1e-10 s-1). 
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


#%%%%% Step 3: Chromatic analysis
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


#%%%%%% Frequency degree
#    - maximum degree of polynomial frequency variations with nu
#    - for some datasets the second order component may not be constrained by the blue bands and remain consistent with 0
gen_dic['wig_deg_Freq'] = {comp_id:1 for comp_id in range(1,6)}


#%%%%%% Amplitude degree
#    - maximum degree of polynomial amplitude variations with nu
#    - defined for each component
gen_dic['wig_deg_Amp'] = {comp_id:2 for comp_id in range(1,6)}


#%%%%% Step 4: Exposure fit 
#    - fitting the spectral wiggle model to each exposure individually
#    - 'comp_ids': components to include in the model
#    - 'init_chrom': initialize the fit using the results of 'wig_exp_nu_ana' on the closest exposure sampled in 'wig_exp_samp'
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
#    - 'prior_par: bound properties with a uniform prior on the chosen range (common to all exposures) informed by 'wig_exp_point_ana'
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


#%%%%% Step 5: Pointing analysis
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

#%%%%% Step 6: Global fit 
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
    'mode':True   ,
    'path':{},
    'exp_list':{},
    'comp_ids':[1,2],
    'range':{},
}




    
if __name__ == '__main__':
    
    #Activating 
    gen_dic['corr_wig']=True    &  False    
    if gen_dic['star_name'] in ['HD209458','WASP76','55Cnc','HD29291']:gen_dic['corr_wig']=True  # & False

    #Calculating/retrieving
    gen_dic['calc_wig']=True  &  False  
    
    #Guide shift reset
    gen_dic['wig_no_guidchange'] = []   
    
    #Forced order normalization
    gen_dic['wig_norm_ord'] = True      
    
    #Visits to be processed 
    gen_dic['wig_vis'] = []
    if gen_dic['star_name']=='HD209458':
        gen_dic['wig_vis'] = ['20190720','20190911'] 
        # gen_dic['wig_vis'] =  ['20190720']    
        # gen_dic['wig_vis'] =  ['20190911']       
    elif gen_dic['star_name']=='WASP76':
        gen_dic['wig_vis'] =  ['20180902','20181030'] 
        # gen_dic['wig_vis'] =  ['20180902']     
    elif gen_dic['star_name']=='55Cnc':  
        gen_dic['wig_vis'] =  ['20200205','20210121','20210124']     
    elif gen_dic['star_name']=='HD29291':
        gen_dic['wig_vis'] = ['20201130'] 

    #Stellar master

    #Resampling 
    gen_dic['wig_indiv_mast'] = True  & False

    #Meridian split
    gen_dic['wig_merid_diff'] = True   & False

    #Exposure selection
    if gen_dic['star_name']=='WASP76':
        
        # gen_dic['wig_exp_mast'] = '2018-09-03':list(range(35)) } 
        # for iexp in [23,24,26]:gen_dic['wig_exp_mast']['2018-09-03'].remove(iexp)
        
        gen_dic['wig_exp_mast'] = { 
            '20180902':'all',
            # '20181030':'all'}       
            # '20181030':list(np.delete(range(69),[0,33]))}   #First expo bad in slice 106 ; expo 33 bad due to guide star change         
            '20181030':list(np.delete(range(69),[33]))}   #Expo 33 bad due to guide star change 

        # #FOR TESTING
        # gen_dic['wig_exp_mast'] =  {    
        #     '20180902':np.array([10,30]),
        #     '20181030':np.array([10,60])}   
    
    # }    

    if gen_dic['star_name']=='HD209458':
        gen_dic['wig_exp_mast'] =  {    
            '20190720':'all',
            '20190911':'all'
    # #TEST
    #         '20190720':range(10),
    #         '20190911':range(10)
        }            
    
    elif gen_dic['star_name']=='55Cnc':    
        # gen_dic['wig_exp_mast'] = { '20200205':list(np.delete(np.arange(97),[0,34,35,37,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,61,84])),   
        #                                         '20210121':list(np.delete(np.arange(99),[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16,17,18,19,20, 21,22  , 56, 57])),
        #                                         '20210124':list(np.delete(np.arange(102),[0, 1, 2])),
        #                                         } 
        gen_dic['wig_exp_mast'] =  { '20200205':'all',   
                                                '20210121':'all',
                                                '20210124':'all',
                                                } 

    elif gen_dic['star_name']=='HD29291':
        gen_dic['wig_exp_mast']={'20201130':'all'}
        

    #Fit settings        
        

    #Exposures to be fitted
    if gen_dic['star_name']=='HD209458':
        gen_dic['wig_exp_in_fit'] ={    

            # '20190720':[88],  #tests
            
            #Init, full analysis
            '20190720':'all',
            '20190911':'all',            

            # #Sampling
            # '20190720':np.arange(0,88,5),
            # # '20190720':np.arange(0,88,10),
            # '20190911':np.arange(0,85,5),
            # # '20190911':np.arange(0,85,10),

            # '20190911':[14],  #Plots ANTARESS I
            # '20190911':[79], 
            } 
    if gen_dic['star_name']=='WASP76':
        gen_dic['wig_exp_in_fit'] ={    

            #Init, full analysis
            '20180902':'all',
            #'20181030':'all',  
            # '20181030':list(np.delete(range(69),[0])),   #First expo bad in slice 106 ; expo 33 bad due to guide star change  
            # '20181030':list(np.delete(range(69),[0,33])),   #First expo bad in slice 106 ; expo 33 bad due to guide star change    
            '20181030':list(np.delete(range(69),[33])),   #Expo 33 bad due to guide star change 

            # #Sampling
            # # '20181030':[0,5,10], 
            # '20180902':np.arange(0,35,5),
            # '20181030':np.arange(0,69,8),

            
            }
    elif gen_dic['star_name']=='55Cnc': 
        gen_dic['wig_exp_in_fit'] =  { '20200205':'all',   
                                                '20210121':'all',
                                                '20210124':'all',
                                                }   
        # gen_dic['wig_exp_in_fit'] = { '20210121':np.arange(0,99,10),
        #                                         '20210124':np.arange(0,102,10),
        #                                         }          
        
    elif gen_dic['star_name']=='HD29291':
        gen_dic['wig_exp_in_fit']={'20201130':'all'}
        # gen_dic['wig_exp_in_fit']={'20201130':np.arange(0,38)}
        # gen_dic['wig_exp_in_fit']={'20201130':np.arange(0,38,4)}        
        # gen_dic['wig_exp_in_fit']={'20201130':np.arange(45,69)} 
    
    


    #Groups of exposures to be fitted together
    # if gen_dic['transit_pl']=='WASP76b':  
    #     gen_dic['wig_exp_groups']={
    #         '2018-09-03':[list(range(i,i+4))  for i in range(0,35,4) ]}
    # if gen_dic['star_name']=='HD209458':  
    #     gen_dic['wig_exp_groups']= {
    #         #First identification of similar groups among original exposures
    #         # '20190720':[list(range(0,7)),list(range(7,14)),list(range(14,22)),list(range(22,30)),list(range(30,35)),list(range(35,46)),list(range(46,56)),list(range(56,73)),list(range(73,79)),list(range(79,85)),list(range(85,89))],
    #         # '20190911':[list(range(0,8)),list(range(8,12)),list(range(12,22)),list(range(22,28)),list(range(28,37)),list(range(37,39)),list(range(39,45)),list(range(45,51)),list(range(51,66)),list(range(66,74)),list(range(74,77)),list(range(77,85))],               

    #         #Second identification of similar groups among original exposures
    #         '20190720':[list(range(0,7)),list(range(7,22)),list(range(22,35)),list(range(35,46)),list(range(46,73)),list(range(73,89))],
    #         '20190911':[list(range(0,12)),list(range(12,22)),list(range(22,37)),list(range(37,39)),list(range(39,51)),list(range(51,66)),list(range(66,74)),list(range(74,85))],               

    #         }    
            
    
    
    
    #Spectral range(s) to be fitted
    if gen_dic['star_name']=='WASP76':
        gen_dic['wig_range_fit'] =  {
    #     gen_dic['wig_range_fit'] = [ [4450.,5235.], [5250.,6270.], [6300.,6900.], [6930.,7585.], [7615.,9000.] ] 
    
        # #Analysis 2022
        # '20180902':[ [33.31027311,43.41672093],[43.66969527,57.04899296], [57.32169369,149.896229] ],    
        # '20181030':[ [33.31027311,43.41672093],[43.66969527,57.04899296], [57.32169369,149.896229] ],    
        #                                         } 
        # #Analysis 2023
        # '20180902':[ [33.31027311,43.41672093],[43.66969527,47.6],[47.8,57.04899296], [57.32169369,149.896229] ],    
        # '20181030':[ [33.31027311,43.41672093],[43.66969527,57.04899296], [57.32169369,149.896229] ],    
        #                                         } 
            #Final analysis
            '20180902': [[20.,57.2],[57.8,67.] ],   
            '20181030': [[20.,57.1],[57.8,67.] ]   }  

        
    elif gen_dic['star_name']=='HD209458':        
        gen_dic['wig_range_fit'] = {        
            #Analysis 2022
            # '20190720':[ [2000.,5230.],[5250.,6265.],[6300.,6860],[6930.,7580.], [7620.,9000.] ],   #spike et triangles dans spectre transmission, slices 92, 128-131, slices 146,147, slices 162,163
            # '20190720': c_light/np.array( [ [2000.,5190.],[5250.,5480.],[5540.,6265.],[6300.,6860],[6930.,7580.],[7705.,9000.] ])[::-1],   #update apres dvp nouveau modele
            # '20190911': c_light/np.array( [ [2000.,5230.],[5250.,6265.],[6300.,6860],[6930.,7580.], [7620.,9000.] ])[::-1],
            # #Analysis 2023
            # '20190720': [[33.31027311111111, 38.90881998702141],[39.550456200527705, 43.26009494949495],[43.701524489795915, 47.586104444444445],[47.85194860335195, 54.11416209386282],[54.70665291970803, 57.10332533333333],[57.76347938342967, 149.896229] ],   #update apres dvp nouveau modele
            # '20190911': [[33.31027311111111, 39.342842257217846],[39.550456200527705, 43.26009494949495],[43.701524489795915, 47.586104444444445],[47.85194860335195, 57.10332533333333],[57.321693690248566, 149.896229]]   }        
            #Final analysis
            '20190720': [[20.,38.9],[39.2,57.2],[57.8,67.] ],   
            '20190911': [[20.,57.2],[57.8,67.] ]   }        


    elif gen_dic['star_name']=='55Cnc':
        # gen_dic['wig_range_fit'] =  {'20200205':[ [5100.,5210.],[5260.,5445.],[5475.,5870.],[5930.,6270.],[6285.,6860.],[7020.,7160.],[7355.,9000.] ],
        #                                          '20210121':[ [4000.,5205.],[5250.,6270.],[6315.,6860.],[7060.,7165.],[7200.,9000.] ], 
        #                                          '20210124':[ [4000.,5230.],[5255.,6270.],[6315.,6860.],[7060.,7160.],[7330.,9000.] ] 
        #                                          } 
        gen_dic['wig_range_fit'] =  {'20210121':[ [30.,41.6],[42.5,43.4],[43.8,57.1],[57.4,71.3] ], 
                                                 '20210124':[ [30.,41.6],[41.9,43.3],[43.9,57.15],[57.5,71.3] ], 
                                                 } 
    elif gen_dic['star_name']=='HD29291':    
        gen_dic['wig_range_fit'] = {'20201130':[[3000.,5190.],[5220.,5425.],[5465.,5850.],[5910.,6860.],[6880.,7180.],[7300.,7590.],[7650.,7900.]]}
        gen_dic['wig_range_fit'] = {'20201130':[[33.31027311111111, 39.],[39.550456200527705, 43.26009494949495],[43.701524489795915,45.],[45.6,50.7],[51.25,54.9],[55.25,56.7],[57.75,64.8],[65.5,67.2],[68.,150.]]   }    
    
        # gen_dic['wig_range_fit'] = {'20201130':[[45.6,50.7]]  }       


    #Spectral bin size
    # gen_dic['wig_bin'] = 0.0004     #to further correct for the 'mini-wiggles' 
    if gen_dic['star_name'] in ['HD209458','WASP76']:gen_dic['wig_bin'] = 0.0166
    

        
    #Orders to be fitted
    if gen_dic['star_name']=='WASP76':

        gen_dic['wig_ord_fit'] =  {

            # #Analysis 2023
            # #    - no need to exclude 107, just 106
            # '20180902':list(np.concatenate((  range(64,88),range(92,106),range(107,162),range(166,170)    ))),   
            # '20181030':list(np.concatenate((  range(64,88),range(92,106),range(107,162),range(166,170)    ))), 
            #Final analysis 
            '20180902':list(np.concatenate((  range(49,87),range(91,170)    ))),
            '20181030':list(np.concatenate((  range(49,87),range(91,170)    ))),    

    
    }

    elif gen_dic['star_name']=='HD209458':
        
        gen_dic['wig_ord_fit'] = {   
            # #Analysis 2023
            # '20190720':list(np.concatenate((  range(60,88),range(92,164),range(166,170)    ))),
            # '20190911':list(np.concatenate((  range(60,88),range(92,164),range(166,170)    ))), 
            #Final analysis 
            '20190720':list(np.concatenate((  range(49,87),range(91,170)    ))),
            '20190911':list(np.concatenate((  range(49,87),range(91,170)    ))), 
        }
    

    
    elif gen_dic['star_name']=='55Cnc':

        gen_dic['wig_ord_fit'] =  {
            '20200205':list(np.concatenate((  range(76,88),range(92,150),range(152,162),range(166,170)    ))),
            '20210121':list(np.concatenate((  range(64,88),range(92,154),range(158,162),range(166,170)     ))),
            '20210124':list(np.concatenate((  range(64,88),range(92,162),range(166,170)     )))
            }

        gen_dic['wig_ord_fit'] = {
            '20210121':list(np.concatenate((range(32,88),range(92,162),range(166,170)))),
            '20210124':list(np.concatenate((range(32,88),range(92,162),range(166,170)))),
            } 
    

    elif gen_dic['star_name']=='HD29291':
        gen_dic['wig_ord_fit']={'20201130':list(range(16,170))}



    #Fitting steps

    #Step 1: Screening 
    gen_dic['wig_exp_init']={
        'mode':False  ,
        'plot_spec':True,
        'plot_hist':True,
        'y_range':[0.993,1.007],   #None
        }


    #Step 2: Chromatic sampling
    gen_dic['wig_exp_samp']={
        'mode':False,   
        # 'comp_ids':[1],
        'comp_ids':[1,2],        
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
        'direct_samp' : {2:0,3:0,4:0},         
        'nit':40, 
        'src_perio' : {1:{'mod':None}, 2:{'mod':None,'up_bd':True},3:{'mod':None,'up_bd':True},4:{'mod':None,'up_bd':True}},  
        'fap_thresh':5,
        # 'fix_freq2expmod':[1]
        'fix_freq2expmod':[],
        'fix_freq2vismod':{},
        # 'fix_freq2vismod':{'comps':[1,2],'20190720':'/Users/bourrier/Travaux/ANTARESS/En_cours/HD209458b_Saved_data/Corr_data/Wiggles/Vis_fit/Indep_contin_valdval/ESPRESSO_20190720/V1/Outputs_final.npz',
        #                                  '20190911':'/Users/bourrier/Travaux/ANTARESS/En_cours/HD209458b_Saved_data/Corr_data/Wiggles/Vis_fit/Indep_contin_valdval/ESPRESSO_20190911/V1/Outputs_loop0_it3.npz'},
        'plot':True
        }   
    
    if gen_dic['star_name']=='HD209458':
        gen_dic['wig_exp_samp']['src_perio'] = {
                1:{'mod':'slide', 'range':[1.,1.] ,'up_bd':False  }, 
                # 1:{'mod':'slide', 'range':[0.4,0.4] ,'up_bd':False  },  
                # # 2:{'mod':'slide', 'range':[0.,3.] ,'up_bd':True  }, 
                2:{'mod':'slide','range':[0.3,0.5] ,'up_bd':True  }, 
                # 2:{'mod':'slide', 'range':[0.2,0.2] ,'up_bd':True  },    
                # 3:{'mod':'slide', 'range':[0.15,0.15] ,'up_bd':True  },     
                4:{'mod':'slide', 'range':[4.,4.] ,'up_bd':False  }, 
                } 
    if gen_dic['star_name']=='WASP76':
        gen_dic['wig_exp_samp']['src_perio'] = {
                1:{'mod':'slide','range':[0.5,0.5] ,'up_bd':False  },
                2:{'mod':'slide','range':[0.3,0.3] ,'up_bd':True  }, 
                3:{'mod':'slide','range':[0.2,0.2] ,'up_bd':False  },                  
                } 
        # gen_dic['wig_exp_samp']['fix_freq2vismod'] = { 'comps':[1,2],
        #         '20180902':'/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Corr_data/Wiggles/Vis_fit/ESPRESSO_20180902/Cont_valdval_degF1_A2/Outputs_final.npz',
        #         }


    #Step 3: Chromatic analysis
    gen_dic['wig_exp_nu_ana']={
        'mode':False  ,     
        'comp_ids':[1,2], 
        # 'comp_ids':[1,2,3], 
        # 'comp_ids':[1,2,4], 
        'thresh':3.,   #None 
        'plot':True
        } 

    #Frequency degree
    if gen_dic['star_name']=='55Cnc':
        gen_dic['wig_deg_Freq'][1] = 1
        gen_dic['wig_deg_Freq'][2] = 1
    elif gen_dic['star_name']=='HD209458':
        gen_dic['wig_deg_Freq'][1] = 1  #final
        gen_dic['wig_deg_Freq'][2] = 1  #final
        gen_dic['wig_deg_Freq'][3] = 1
        gen_dic['wig_deg_Freq'][4] = 1
    elif gen_dic['star_name']=='WASP76':
        gen_dic['wig_deg_Freq'][1] = 1  #final
        gen_dic['wig_deg_Freq'][2] = 1  #final

    #Amplitude degree
    if gen_dic['star_name']=='HD209458':
        # gen_dic['wig_deg_Amp'][1]=3
        gen_dic['wig_deg_Amp'][1]=2   #final
        gen_dic['wig_deg_Amp'][2]=2   #final
        gen_dic['wig_deg_Amp'][3]=2
        gen_dic['wig_deg_Amp'][4]=1
    if gen_dic['star_name']=='WASP76':
        gen_dic['wig_deg_Amp'][1]=2  #final
        gen_dic['wig_deg_Amp'][2]=2  #final
        # gen_dic['wig_deg_Amp'][3]=2
    


    #Step 4: Exposure fit 
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


    if gen_dic['star_name']=='HD209458':
        gen_dic['wig_exp_fit']['prior_par']['20190720']={
        # #     'AmpGlob1_c0':{'low':4e-4},'AmpGlob1_c1':{'high':5e-5},
        #     'AmpGlob2_c0':{'low':1e-4},
            # 'AmpGlob2_c1':{'low':-1.5e-4,'high':0.},
        # 'AmpGlob2_c2':{'low':-6e-6},'AmpGlob2_c3':{'high':6e-7},
            # 'AmpGlob2_c0':{'low':-1e-3,'high':1e-3},
            # 'AmpGlob2_c1':{'low':-1.3e-4,'high':1.3e-4},
            'AmpGlob2_c2':{'low':-1.2e-5,'high':1.2e-5},
            # 'AmpGlob3_c0':{'low':0.,'high':8e-4},'AmpGlob3_c1':{'low':-5e-5,'high':1e-5},'AmpGlob3_c2':{'low':-5e-6,'high':4e-6},
            # 'AmpGlob4_c0':{'low':1e-4,'high':5e-4},'AmpGlob4_c1':{'low':-6e-5,'high':0.},
            # 'Freq1_c0':{'low':3.707,'high':3.74},'Freq1_c1':{'low':0.0005,'high':0.02},
            # 'Freq2_c0':{'low':2.0,'high':2.15 },'Freq2_c1':{'low':-0.005,'high':0.01},
            # 'Freq2_c0':{'low':2.035,'high':2.12},
            # 'Freq2_c1':{'low':-0.009,'high':0.01},
            # 'Freq3_c0':{'low':3.53,'high':3.57},'Freq3_c1':{'low':-0.002,'high':0.006},
            # 'Freq4_c0':{'low':156.25,'high':156.38},'Freq4_c1':{'low':0.092,'high':0.108},
            # 'Phi2':{'low':-3.,'high':3.},
            # 'Phi3':{'low':-6.,'high':7.},
            # 'Phi4':{'low':-2.,'high':8.},
        }
        gen_dic['wig_exp_fit']['model_par']['20190720']={
            # 'AmpGlob1_c0':[4e-4,4e-4],'AmpGlob1_c1':[2.5e-5,2.5e-5],'AmpGlob1_c2':[3e-6,3e-6], 
            # 'AmpGlob2_c0':[1e-3,4e-4],
            # 'AmpGlob2_c1':[1e-4,1e-4],
            # 'AmpGlob2_c2':[1e-5,8e-6],
            # # 'AmpGlob3_c0':[5e-6,4e-6],
            # # 'AmpGlob4_c0':[2e-4,2e-4],'AmpGlob4_c1':[2e-5,2e-5],
            # 'Freq1_c0':[5e-3,5e-3],'Freq1_c1':[1e-3,1e-3],
            # 'Freq2_c0':[5e-2,5e-2],
            # 'Freq2_c1':[8e-3,8e-3],
            # # 'Freq4_c0':[5e-2,5e-2],'Freq4_c1':[6e-3,6e-3],
            # 'Phi1':[0.6,0.6],
            # 'Phi2':[1.5,1.],
            # # 'Phi3':[3.,3.],'Phi4':[1.,1.], 
            } 
        gen_dic['wig_exp_fit']['fixed_pointpar']['20190720']=['AmpGlob1_c0','AmpGlob1_c1','AmpGlob1_c2','Freq1_c0','Freq1_c1','Phi1','AmpGlob2_c0','AmpGlob2_c1','Freq2_c0','Freq2_c1','Phi2'] 
     
        
        gen_dic['wig_exp_fit']['prior_par']['20190911']={
            # 'AmpGlob2_c0':{'low':-0.5e-3,'high':0.8e-3},
            # 'AmpGlob2_c1':{'low':-2e-4,'high':1e-4},
            # 'AmpGlob2_c2':{'low':-1.5e-5,'high':2e-5}, 
            # 'AmpGlob4_c0':{'low':0.,'high':1e-3},'AmpGlob4_c1':{'low':-6e-5,'high':6e-5},
            # 'Freq1_c0':{'low':3.6,'high':3.85},'Freq1_c1':{'low':0.0017,'high':0.0034},
            # 'Freq2_c0':{'low':1.9,'high':2.16},'Freq2_c1':{'low':-0.01,'high':0.014},
            # 'Freq4_c0':{'low':156.,'high':156.8},'Freq4_c1':{'low':0.07,'high':0.15},
            # 'Phi2':{'low':-10.,'high':10.},
            # 'Phi4':{'low':-10.,'high':10.},
            }
        gen_dic['wig_exp_fit']['model_par']['20190911']={
            # 'AmpGlob1_c0':[4e-4,4e-4],'AmpGlob1_c1':[2e-5,2e-5],'AmpGlob1_c2':[2e-6,2e-6],
            # # 'AmpGlob2_c0':[3e-4,3e-4],'AmpGlob2_c1':[6e-5,4e-5],'AmpGlob2_c2':[6e-6,5e-6],
            # 'Freq1_c0':[ 7e-3,7e-3],'Freq1_c1':[1.5e-3,1.5e-3],
            'Freq2_c0':[4e-2,4e-2],
            'Freq2_c1':[5e-3,1e-3],
            # 'Phi1':[0.6,0.6],
            # 'Phi2':[1.5,1.5  ],           
            }  
        gen_dic['wig_exp_fit']['fixed_pointpar']['20190911']=['AmpGlob1_c0','AmpGlob1_c1','AmpGlob1_c2','Freq1_c0','Freq1_c1','Phi1','AmpGlob2_c0','AmpGlob2_c1','AmpGlob2_c2','Phi2'] 


    if gen_dic['star_name']=='WASP76':
        gen_dic['wig_exp_fit']['prior_par']['20180902']={
            # 'AmpGlob1_c0':{'low':-7e-3,'high':3.3e-3},
            # 'AmpGlob1_c1':{'low':-1.5e-4,'high':0.},
            # 'AmpGlob1_c2':{'low':-7e-6,'high':2e-6},
            # 'AmpGlob2_c0':{'low':-1.5e-3,'high':1.5e-3},
            # 'AmpGlob2_c1':{'low':-3e-4,'high':3e-4},
            # 'AmpGlob2_c2':{'low':-1.5e-5,'high':1.5e-5},
            # 'Freq1_c0':{'low':3.715,'high':3.735},
            # 'Freq1_c1':{'low':0.0024,'high':0.004},
            # 'Freq2_c0':{'low':1.9,'high':2.12},
            # 'Freq2_c1':{'low':-0.02,'high':0.012},
    # # #         'Freq3_c0':{'low':2.3,'high':3.2},
    # # #         'Freq3_c1':{'low':-0.08,'high':0.08},
            # 'Phi2':{'low':-10.,'high':10.},
        }  
        gen_dic['wig_exp_fit']['model_par']['20180902']={ 
            # 'AmpGlob1_c0':[2e-4,2e-4], 
            # 'AmpGlob1_c1':[2e-5,2e-5], 
            # 'AmpGlob1_c2':[2e-6,2e-6],
            # 'AmpGlob2_c0':[1e-3,1e-3],             
            # 'AmpGlob2_c1':[3e-4,3e-4],         
            # 'AmpGlob2_c2':[3e-5,3e-5],
               #  'Freq1_c0':[2.5e-3,2.5e-3],
               # 'Freq1_c1':[8e-4,8e-4],
               # 'Freq2_c0':[5e-2,5e-2],
               # 'Freq2_c1':[1e-3,1e-3],
              # 'Phi1':[0.5,0.5],
                # 'Phi2':[1.,1.],
            } 
        gen_dic['wig_exp_fit']['fixed_pointpar']['20180902']=['AmpGlob1_c0','AmpGlob1_c1','Freq1_c0','Freq1_c1','Freq2_c0','Freq2_c1','Phi1','Phi2','AmpGlob2_c0','AmpGlob2_c1','AmpGlob2_c2']   
        
        
        
        gen_dic['wig_exp_fit']['prior_par']['20181030']={
            # 'AmpGlob2_c0':{'low':-6e-4,'high':1e-3},
            # 'AmpGlob2_c1':{'low':-4e-4,'high':4e-4},
            # 'AmpGlob2_c2':{'low':-2.5e-5,'high':4e-5},
    # #         # 'Freq1_c0':{'low':3.71,'high':3.732},
            # 'Freq1_c1':{'low':0.002,'high':0.0036},
            # 'Freq2_c0':{'low':1.97,'high':2.35},
            # 'Freq2_c1':{'low':-0.01,'high':0.03},      
            # 'Phi2':{'low':-20.,'high':20.}, 
        }
        gen_dic['wig_exp_fit']['model_par']['20181030']={
            # 'AmpGlob1_c0':[4e-4,4e-4], 'AmpGlob1_c1':[4e-5,4e-5], 'AmpGlob1_c2':[5e-6,5e-6],
            # 'AmpGlob2_c0':[3e-4,2e-4],
            # 'AmpGlob2_c1':[3e-5,3e-5],
            # 'AmpGlob2_c2':[3e-6,3e-6],
               # 'Freq1_c0':[5e-3,5e-3],
                # 'Freq1_c1':[1e-3,1e-3],
                'Freq2_c0':[7e-2,3e-2],
                'Freq2_c1':[4e-2,5e-3],
                # 'Phi1':[0.5,0.5],
                # 'Phi2':[3,3.],
            }         
        
        gen_dic['wig_exp_fit']['fixed_pointpar']['20181030']=['AmpGlob1_c0','AmpGlob1_c1','AmpGlob1_c2','AmpGlob2_c0','AmpGlob2_c1','AmpGlob2_c2','Freq1_c0','Freq1_c1','Phi1','Phi2']   







    #Step 5: Pointing analysis
    gen_dic['wig_exp_point_ana']={
        'mode':False ,    
        # 'source':'samp',
        'source':'glob',
        'thresh':3.,   #None
        'fit_range':{},
        'fit_undef':False,
        'stable_pointpar':[],
        'conv_amp_phase':True & False ,
        'plot':True
        } 

    # if gen_dic['star_name']=='HD209458':   
    #     gen_dic['wig_exp_point_ana']['stable_pointpar'] = ['Freq3_c0','Freq3_c1']


    # if gen_dic['star_name']=='WASP76':   
    #     gen_dic['wig_exp_point_ana']['stable_pointpar'] = ['Freq1_c1','Freq2_c1','AmpGlob1_c2','AmpGlob2_c0','AmpGlob2_c1','AmpGlob2_c2']

    # if gen_dic['star_name']=='WASP76':  
    #     gen_dic['wig_exp_point_ana']['fit_range']['20181030']={}        
    #     # for key in ['AmpGlob2_c0','AmpGlob2_c1','AmpGlob2_c2','Freq2_c0','Freq2_c1','Phi2']:
    #     #     # gen_dic['wig_exp_point_ana']['fit_range']['20181030'][key] = [[-0.2,0.002],[0.008,0.064],[0.077,0.2]]
    #     #     # gen_dic['wig_exp_point_ana']['fit_range']['20181030'][key] = [[-0.2,-0.032],[-0.02,-0.01],[-0.008,0.002],[0.008,0.044],[0.048,0.064],[0.077,0.2]]
    #     #     # gen_dic['wig_exp_point_ana']['fit_range']['20181030'][key] = [[-0.2,-0.032],[-0.02,-0.01],[-0.008,0.002],[0.008,0.044],[0.077,0.2]]
    #     #     # gen_dic['wig_exp_point_ana']['fit_range']['20181030'][key] = [[-0.2,-0.01],[-0.008,0.002],[0.008,0.044],[0.067,0.2]]
    #     #     # gen_dic['wig_exp_point_ana']['fit_range']v[key] = [[-0.2,-0.032],[-0.02,0.044],[0.048,0.064],[0.077,0.2]]
    #     #     gen_dic['wig_exp_point_ana']['fit_range']['20181030'][key] = [[-0.2,-0.032],[-0.02,0.002],[0.008,0.064],[0.077,0.2]]

    #     for key in ['AmpGlob1_c0','AmpGlob1_c1','AmpGlob1_c2','Freq1_c1','Phi1']:
    #         gen_dic['wig_exp_point_ana']['fit_range']['20181030'][key] = [[-0.2,0.002],[0.008,0.2]]






    #Step 6: Global fit 
    gen_dic['wig_vis_fit']={
        'mode':False ,
        'fit_method':'leastsq',  
        # 'fit_method':'nelder',  
        'wig_fit_ratio': False,
        'wig_conv_rel_thresh':1e-5,
        # 'nit':25,
        'nit':15,
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
    # if gen_dic['star_name']=='HD209458':    
    #     gen_dic['wig_vis_fit']['reuse']={'20190720':'/Users/bourrier/Travaux/ANTARESS/En_cours/HD209458b_Saved_data/Corr_data/Wiggles/Vis_fit/Vfinal_3_lsq/ESPRESSO_20190720/Outputs_final.npz',
    #                                       '20190911':'/Users/bourrier/Travaux/ANTARESS/En_cours/HD209458b_Saved_data/Corr_data/Wiggles/Vis_fit/Vfinal_3_lsq/ESPRESSO_20190911/Outputs_final.npz'}


    # if gen_dic['star_name']=='WASP76':   
    #     gen_dic['wig_vis_fit']['reuse']={
    #         '20180902':'/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Corr_data/Wiggles/Vis_fit/ESPRESSO_20180902/Outputs_final.npz',
    #         # '20180902':'/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Corr_data/Wiggles/Vis_fit/ESPRESSO_20180902/Cont_valdval_degF1_A2/V16_ratio_eps/Outputs_loop6.npz',
    #         '20181030':'/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Corr_data/Wiggles/Vis_fit/ESPRESSO_20181030/Cont_valdval_degF1_A2/Outputs_loop0_it6.npz'}  
    #     # gen_dic['wig_vis_fit']['stable_pointpar'] = ['Freq1_c1','Freq2_c1','AmpGlob1_c2','AmpGlob2_c0','AmpGlob2_c1','AmpGlob2_c2']







    #Correction
    gen_dic['wig_corr'] = {
        'mode':True   ,
        'path':{},
        'exp_list':{},
        'comp_ids':[1,2],
        'range':{},
    }

    if gen_dic['star_name']=='HD209458':
        gen_dic['wig_corr']['path'] = {'20190720':'/Users/bourrier/Travaux/ANTARESS/En_cours/HD209458b_Saved_data/Corr_data/Wiggles/Vis_fit/Vfinal_15_lsq/ESPRESSO_20190720/Outputs_final.npz',
                                        '20190911':'/Users/bourrier/Travaux/ANTARESS/En_cours/HD209458b_Saved_data/Corr_data/Wiggles/Vis_fit/Vfinal_15_lsq/ESPRESSO_20190911/Outputs_final.npz'}
    if gen_dic['star_name']=='WASP76':
        gen_dic['wig_corr']['path'] = {'20180902':'/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Corr_data/Wiggles/Vis_fit/Vfinal_4_lsq/ESPRESSO_20180902/Outputs_final.npz',
                                        '20181030':'/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Corr_data/Wiggles/Vis_fit/Vfinal_4_lsq/ESPRESSO_20181030/Outputs_final.npz'}
    




    















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


if __name__ == '__main__':

    #Activating
    gen_dic['corr_fring']=True    &  False    

    #Calculating/retrieving
    gen_dic['calc_fring']=True    &  False  
    
    #Spectral range(s) to be corrected
    gen_dic['fring_range']=[6000.,8000.]
    
 
    #Plots: correction
    plot_dic['fring_corr']=''   #pdf      
    
    
    
    
    
    
    
    
    
##################################################################################################
#%%% Module: trimming 
##################################################################################################

#%%%% Activating
gen_dic['trim_spec']=False


#%%%% Calculating/retrieving 
gen_dic['calc_trim_spec']=True  


#%%%% Spectral ranges to be kept
#    - define the spectral range(s) and orders over which spectra should be used in the pipeline
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
#    - indexes are relative to order list after exclusion with gen_dic['del_orders'] 
gen_dic['trim_orders'] = {}

    

if __name__ == '__main__':    

    #Activating
    gen_dic['trim_spec']=True    &  False
    if gen_dic['star_name'] in ['WASP107','HAT_P11','WASP156','HD209458']:gen_dic['trim_spec']=True  & False

    #Calculating/retrieving 
    gen_dic['calc_trim_spec']=True   &  False 


    #Spectral ranges to be kept
    # gen_dic['trim_range'] = [ [3200.,5160.],[5250.,6800.],[6950.,7150.],[7300.,7550.],[7700.,8500.] ]
    # gen_dic['trim_range'] = [ [4500.,4600.] ] 
    # gen_dic['trim_range'] = [ [3700.,3850.] ]
    # gen_dic['trim_range'] = [ [5700.,6100.] ]  #Na doublet
    # gen_dic['trim_range'] = [ [5875.,5897.] ]


    #Orders to be kept    
    # if gen_dic['star_name']=='HD3167': 
    #     gen_dic['trim_orders'] = {'ESPRESSO':list(range(20,170))}       #remove bluest orders
    if gen_dic['star_name'] in ['WASP107']:
        gen_dic['trim_orders'] = {'CARMENES_VIS':list(range(1,57))}       #remove first, and reddest orders
    if gen_dic['star_name'] in ['HAT_P11','WASP156']:
        gen_dic['trim_orders'] = {'CARMENES_VIS':list(range(57))}       #remove reddest orders        
    if gen_dic['star_name']=='HD209458': 
        gen_dic['trim_orders'] = {'ESPRESSO':[114,115,116,117,118,119]}    #sodium doublet and surrounding orders
    if gen_dic['star_name']=='WASP76': 
        gen_dic['trim_orders'] = {'ESPRESSO':[114,115,116,117,118,119]}    #sodium doublet and surrounding orders











##################################################################################################
#%% Disk-integrated profiles
##################################################################################################  



##################################################################################################
#%%% Module: CCF conversion for disk-integrated spectra
#    - before spectra are aligned, but after they have been corrected for systematics, to get data comparable to standard DRS outputs  
#    - every operation afterwards will be performed on those profiles 
#    - applied to input data in spectral mode
##################################################################################################        

#%%%% Activating
gen_dic['DI_CCF'] = False

    
#%%%% Calculating/retrieving
gen_dic['calc_DI_CCF']= False    


#%%%% Radial velocity table
#    - define for raw CCFs in the original rest frame
#      the table will be shifted automatically into the star rest frame, and used for local and atmospheric CCFs
#    - set dRV to None to use instrumental resolution
#      these CCFs will not be screened, so be careful about the selected resolution (lower than instrumental will introduce correlations)
gen_dic['start_RV']=-100.    
gen_dic['end_RV']=100.
gen_dic['dRV']=None  



if __name__ == '__main__':   

    #Activating
    gen_dic['DI_CCF'] = True  &  False
    if gen_dic['star_name'] in ['WASP107','HAT_P11','WASP156','GJ3090','55Cnc']:gen_dic['DI_CCF'] = True   & False
        
    #Calculating/retrieving
    gen_dic['calc_DI_CCF']=True  &  False      

    #Radial velocity table
    # gen_dic['dRV']=0.5    #res. ESPRESSO, EXPRES
    # gen_dic['dRV']=0.82   #res. HARPN 
    if gen_dic['star_name']=='HD3167':   
        gen_dic['start_RV']=-60.    
        gen_dic['end_RV']=100.    
    elif gen_dic['star_name']=='WASP107':   
        gen_dic['start_RV']=-150.    
        gen_dic['end_RV']=150.        
        gen_dic['dRV']=1.1    
    elif gen_dic['star_name']=='HAT_P11':   
        gen_dic['start_RV']=-150.    
        gen_dic['end_RV']=150.        
        gen_dic['dRV']=1.1    
    elif gen_dic['star_name']=='WASP156':   
        gen_dic['start_RV']=-150.    
        gen_dic['end_RV']=150.        
        gen_dic['dRV']=1.1    
    elif gen_dic['star_name']=='GJ3090':   
        gen_dic['start_RV']=-300.    
        gen_dic['end_RV']=300.        
        gen_dic['dRV']=None
    elif gen_dic['star_name']=='55Cnc':   
        gen_dic['start_RV']=-200.    
        gen_dic['end_RV']=200.        
        gen_dic['dRV']=None
    elif gen_dic['star_name']=='HD209458':   
        gen_dic['start_RV']=-150.    
        gen_dic['end_RV']=150.        
        gen_dic['dRV']=None
    elif gen_dic['star_name']=='WASP76':   
        gen_dic['start_RV']=-150.    
        gen_dic['end_RV']=150.        
        gen_dic['dRV']=None
















##################################################################################################
#%%% Module: detrending disk-integrated profiles
##################################################################################################

#%%%% Activating
gen_dic['detrend_prof'] = False


#%%%% Calculating/retrieving 
gen_dic['calc_detrend_prof']=True  


#%%%% Spectral correction
#    - only relevant in spectral mode
#    - will be applied before spectra are converted into CCFs

#%%%%% Full spectrum
detrend_prof_dic['full_spec']=False

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


#%%%%% Settings     
#    - structure is : inst > vis > correction > coefficients
#    - define correction as 'prop_var': 
# with 'prop' among: 'RV', 'ctrst', 'FWHM' 
#      'var' among: 'phase', 'snr', 'AM', 'ha', 'na', 'ca', 's', 'rhk'
#      use '_snrQ' for the SNR of orders provided as input to be summed quadratically (useful to combine ESPRESSO slices) rather than being averaged
#    - contrast and FWHM corrections are defined as
# F(x) = a0*(1 + c1*x + c2*x^2 + ... )*(1+A*sin((x-xref)/P))
#    - RV correction is defined as
# F(x) = a0 + a1*x + a2*x^2 + ... + A*sin((x-xref)/P)) 
#      with ai and A in m/s
#    - the polynomial coefficients are used if defined via 'pol'
#      the sinusoidal coefficients are used if defined via 'sin'
#    - the constant level a0 is left undedefined :  for contrast and FWHM models are normalized to their mean, and for RVs the level is controlled by the alignment module and sysvel
#    - RV correction must be done in the input rest frame, as CCFs are corrected before being aligned
detrend_prof_dic['prop']={}    

        
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
#    - by default should be left undefined, so that all PC fitted to the residual and intrinsic profiles are used
#    - this option is however useful to visualize each PC contribution through plot_dic['map_pca_prof']
detrend_prof_dic['idx_PC']={}


#%%%%% Plots: 2D PC noise model 
#    - using the model generated with the above options
plot_dic['map_pca_prof']=''   

    


if __name__ == '__main__':  

    #Activating
    gen_dic['detrend_prof']=True   &   False
    if gen_dic['star_name'] in ['Kepler68','HAT_P33','HD89345','HAT_P49','WASP107','WASP166','HAT_P11','WASP156','Kepler25','55Cnc']:gen_dic['detrend_prof']=True   
    # if gen_dic['star_name'] in ['HD189733']:gen_dic['detrend_prof']=True 
    # if gen_dic['star_name'] in ['WASP43']:gen_dic['detrend_prof']=True 
    if (gen_dic['star_name'] in ['HD209458','WASP76']):gen_dic['detrend_prof']=True # &   False
    

    # gen_dic['detrend_prof']= False  
    # print('ATTENTION')

    #Calculating/retrieving 
    gen_dic['calc_detrend_prof']=True  &  False   

    
    #Full spectrum
    detrend_prof_dic['full_spec']=True
    
    #Transition wavelength
    detrend_prof_dic['line_trans']=None
    
    
    #Order to be corrected
    detrend_prof_dic['iord_line']=0
    



    #Trend correction
    
    #Activating 
    detrend_prof_dic['corr_trend'] = True #   & False

    #Settings     
    # if gen_dic['transit_pl']=='WASP_8b':
    #     detrend_prof_dic['prop']={'HARPS':{'2008-10-04':{}}}  
    # elif gen_dic['transit_pl']=='WASP121b':
    #     detrend_prof_dic['prop']={'HARPS':{'31-12-17':{}}}  
    # elif gen_dic['transit_pl']=='KELT10b':
    #     detrend_prof_dic['prop']={'HARPS':{'2016-07-21':{}}}
    
    if gen_dic['star_name']=='55Cnc':
        detrend_prof_dic['prop']={
            
              # #FINAL : PCA correction preferred
              # 'ESPRESSO':{
              #      # '20200205':{'RV_phase':{'pol':1e-3*np.array([-4.817888e-02,-1.965599e+01])},'FWHM_phase':{'pol':np.array([-5.685398e-04])},'FWHM_snrQ':{'pol':np.array([4.923796e-06])},'ctrst_phase':{'pol':np.array([-1.503451e-03])},'ctrst_snrQ':{'pol':np.array([3.206209e-05,-7.303334e-08])}},    #default
              #       '20200205':{'RV_phase':{'pol':1e-3*np.array([-4.449146e-02,-3.183332e+01])},'FWHM_phase':{'pol':np.array([3.288106e-05,-8.594587e-03])},'FWHM_snrQ':{'pol':np.array([-2.010399e-05,5.400259e-08])},'ctrst_phase':{'pol':np.array([-4.348351e-04,-6.086417e-03])},'ctrst_snrQ':{'pol':np.array([2.753372e-05,-6.718497e-08])}},    #moon corr                          
              #      # '20200205':{'RV_phase':{'pol':1e-3*np.array([9.386842e-02,-1.853804e+00])},'FWHM_snrQ':{'pol':np.array([-1.424411e-05,4.299159e-08])},'ctrst_snrQ':{'pol':np.array([3.379979e-05,-7.980484e-08])}},    #moon corr, conservative correction  
              #      '20210121':{'RV_snrQ':{'pol':1e-3*np.array([3.307761e-03])},'FWHM_phase':{'pol':np.array([-2.518389e-04,-8.177916e-03])},'FWHM_snrQ':{'pol':np.array([3.503354e-06])},'ctrst_phase':{'pol':np.array([1.686988e-04,-1.048868e-02])},'ctrst_snrQ':{'pol':np.array([-3.645895e-05,4.468659e-08])}},
              #      '20210124':{'RV_phase':{'pol':1e-3*np.array([-9.673485e-01])},'RV_snrQ':{'pol':1e-3*np.array([-2.630128e-02,5.469735e-05])},'FWHM_phase':{'pol':np.array([8.586828e-04])},'FWHM_snrQ':{'pol':np.array([5.198959e-06])},'ctrst_snrQ':{'pol':np.array([-9.114192e-05,2.524038e-07,-2.646706e-10])}}}

              #       # '20200205':{'RV_phase':{'pol':1e-3*np.array([-4.199375e-02,8.645111e-01]),'sin':[2.074109e-04,2.685015e-03,4.886826e-02]}},
              #       # '20210121':{'RV_phase':{'pol':1e-3*np.array([8.346631e-03,6.493626e-01]),'sin':[2.230221e-04,-1.936969e-02,6.091807e-02]}},
              #       # '20210124':{'RV_phase':{'pol':1e-3*np.array([-3.284966e-02,2.372787e-01]),'sin':[2.541079e-04,-1.039229e-02,5.599010e-02]}}},   

               'HARPS':{'20120127':{'RV_phase':{'pol':1e-3*np.array([6.057300e+00])}},   #FINAL
                        '20120315':{'RV_phase':{'pol':1e-3*np.array([1.063915e+01])}},   #FINAL
                },


                # 'HARPN':{
                #            '20131114':{'RV_phase':{'pol':1e-3*np.array([9.914448e+00])},'ctrst_snr':{'pol':np.array([-5.590134e-06])}},
                #            '20131128':{'RV_phase':{'pol':1e-3*np.array([7.805636e+00])},'ctrst_snr':{'pol':np.array([-4.098729e-05,9.664600e-08])}},
                #            '20140101':{'RV_phase':{'pol':1e-3*np.array([5.631273e+00])},'ctrst_snr':{'pol':np.array([-6.456629e-06])}},
                #            '20140126':{'RV_phase':{'pol':1e-3*np.array([-9.336609e+00])},'FWHM_phase':{'pol':np.array([-2.578572e-03])},'ctrst_snr':{'pol':np.array([-9.223894e-05,2.313920e-07])}},
                #            '20140226':{'RV_phase':{'pol':1e-3*np.array([7.852731e+00])},'ctrst_snr':{'pol':np.array([6.264133e-05,-3.335921e-07])}},
                #            '20140329':{'FWHM_snr':{'pol':np.array([1.906481e-06])},'ctrst_snr':{'pol':np.array([-7.439530e-06])}},
                #  }

                'EXPRES':{
                    '20220131':{'RV_phase':{'pol':1e-3*np.array([3.726873e+00])},'FWHM_phase':{'pol':np.array([1.979122e-03])}},   
                    '20220406':{'FWHM_phase':{'pol':np.array([8.758985e-04])},'ctrst_phase':{'pol':np.array([-1.058184e-03])}}}   
               
                                                                                             
              }            
    
    
    elif gen_dic['star_name']=='GJ436':
        detrend_prof_dic['prop']={'HARPN':{'20160318':{'ctrst_snr':np.array([3.210585e-01,1.304690e-04])},'20160411':{'ctrst_snr':np.array([3.208316e-01,1.543300e-04])}},
                              'HARPS':{'20070509':{'ctrst_snr':np.array([3.098972e-01,8.871225e-04])}}}  
    elif gen_dic['star_name']=='HD15337':        
        detrend_prof_dic['prop']={'ESPRESSO_MR':{'20191122':{'ctrst_snr':np.array([5.597405e-01,2.065194e-04,-5.500241e-07]),
                                                         'RV_snr':np.array([7.615809e+01-7.61583e+01,1.883024e-05]),    #after ctrst correction, around sysvel so as not to change it 
                                                         'FWHM_snr':np.array([7.551256e+00,-6.496007e-06])}}}      #after ctrst and RV correction

    #RM survey
    elif gen_dic['star_name']=='HAT_P3':    
        # detrend_prof_dic['prop']={'HARPN':{'20200130':{'ctrst_snr':{'pol':np.array([None,1.268132e-04])},'FWHM_snr':{'pol':np.array([None,-1.731632e-04])}}}}    #new mask, no skycorr
        # detrend_prof_dic['prop']={'HARPN':{'20200130':{'ctrst_snr':{'pol':np.array([None,1.372159e-04])}}}}    #kitcat mask, skycorr
        detrend_prof_dic['prop']={'HARPN':{'20200130':{'RV_snr':{'pol':1e-3*np.array([7.768602e+00,-2.131915e-01])}}}}    #kitcat mask, no skycorr        

        
    elif gen_dic['star_name']=='Kepler68':    
        # detrend_prof_dic['prop']={'HARPN':{'20190803':{'ctrst_snr':{'pol':np.array([None,2.403199e-03,-3.496778e-05])}}}}    #new mask
        # detrend_prof_dic['prop']={'HARPN':{'20190803':{'ctrst_snr':{'pol':np.array([5.129138e-01,1.433217e-04])}}}}  
        detrend_prof_dic['prop']={'HARPN':{'20190803':{'ctrst_snr':{'pol':np.array([None,2.693647e-03,-4.065767e-05])}}}}    #kitcat final
        

        
    elif gen_dic['star_name']=='HAT_P33':    
        # detrend_prof_dic['prop']={'HARPN':{'20191204':{'ctrst_phase':{'pol':np.array([None,-1.010879e-01])}}}}    
        detrend_prof_dic['prop']={'HARPN':{'20191204':{'ctrst_snr':{'pol':np.array([None,5.189697e-04])}}}}   #final
        
    elif gen_dic['star_name']=='Kepler25':
        detrend_prof_dic['prop']={
            'HARPN':{'20190614':{'ctrst_phase':{'pol':np.array([None,1.246878e-01])}}}}    #new mask
            # 'HARPN':{'20190614':{'ctrst_phase':{'pol':np.array([None,1.060643e-01])}}}}    #kitcat mask        
        
    elif gen_dic['star_name']=='HD89345':
        # detrend_prof_dic['prop']={'HARPN':{'20200202':{'ctrst_phase':{'pol':np.array([None,3.544506e-02])},'FWHM_phase':{'pol':np.array([None,-7.902347e-02])}}}}      #kitcat        
        detrend_prof_dic['prop']={'HARPN':{'20200202':{'ctrst_snr':{'pol':np.array([None,-4.354953e-05])},'FWHM_snr':{'pol':np.array([None,1.030280e-04])}}}}      #kitcat, final

    elif gen_dic['star_name']=='HAT_P49':
        # detrend_prof_dic['prop']={'HARPN':{'20200730':{'ctrst_phase':{'pol':np.array([None,-1.146138e-01])},'FWHM_phase':{'pol':np.array([None,1.019980e-01])},'RV_phase':{'pol':1e-3*np.array([-2.046618e+01,-9.318635e+02])}}}}    #new        
        detrend_prof_dic['prop']={'HARPN':{'20200730':{
            'ctrst_phase':{'pol':np.array([None,-1.060665e-01])},
            'FWHM_phase':{'pol':np.array([None,8.297072e-02])},
            'RV_phase':{'pol':1e-3*np.array([-1.715367e+01,-8.310732e+02])}
            }}}    #Kitcat, final

        # #Fit_noTexp2_post111
        # detrend_prof_dic['prop']={'HARPN':{'20200730':{'ctrst_phase':{'pol':np.array([None,-9.428715e-02])},'FWHM_phase':{'pol':np.array([None,8.607855e-02])},
        #                                                'RV_phase':{'pol':1e-3*np.array([-2.922000e+01,-8.512816e+02])}
        #                                              }}}    #Kitcat, final



    
    elif gen_dic['star_name']=='WASP107':
        # detrend_prof_dic['prop']={'HARPS':{'20140406':{'ctrst_snr':{'pol':np.array([None,-1.240177e-03])}},'20180201':{'ctrst_snr':{'pol':np.array([None,-3.768760e-04])}}}}     #new
        detrend_prof_dic['prop']={'HARPS':{'20140406':{'ctrst_snr':{'pol':np.array([None,-1.931094e-03])}}}}     #kitcat, final

        detrend_prof_dic['prop'].update({'CARMENES_VIS':{'20180224':{'FWHM_phase':{'pol':np.array([None,-1.341420e-01])}}}})      #final
        # detrend_prof_dic['prop'].update({'CARMENES_VIS':{'20180224':{'FWHM_snr':{'pol':np.array([None,4.511839e-04])}}}})          
        
        
    elif gen_dic['star_name']=='WASP166':
        # detrend_prof_dic['prop']={'HARPS':{'20170304':{'FWHM_snr':{'pol':np.array([None,3.380117e-03,-7.161555e-05,4.802533e-07])}}}}     #new
        # detrend_prof_dic['prop']={'HARPS':{'20170304':{'ctrst_snr':{'pol':np.array([None,2.722663e-05])}}}}     #kitcat
        detrend_prof_dic['prop']={'HARPS':{'20170304':{'ctrst_phase':{'pol':np.array([None,-3.306329e-02])}}}}     #kitcat, final

    elif gen_dic['star_name']=='HAT_P11':
        detrend_prof_dic['prop']={'CARMENES_VIS':
                              
                #Fit Voigt / gauss  
                # {'20170807':{'ctrst_phase':{'pol':np.array([4.917992e-01,-6.555059e-04,-3.690622e+00,-7.582130e+01])},'RV_phase':{'pol':1e-3*np.array([-4.114537e+02,4.431178e+01])},'FWHM_phase':{'pol':np.array([8.063138e+00,-1.088590e-01,-5.838092e+00,-6.338006e+01])}},
                #  '20170812':{'ctrst_phase':{'pol':np.array([4.921464e-01,1.341383e-02,-2.356178e+00,-8.244950e+01,2.181136e+03])},'RV_phase':{'pol':1e-3*np.array([-4.062659e+02,7.087727e+01,-3.866739e+03,1.613418e+05])},'FWHM_phase':{'pol':np.array([8.089434e+00,-9.535964e-02,-1.777242e+00,-3.786594e+00,-2.113113e+03])}}},

                # #Fit gaussien
                # {'20170807':{ 'ctrst_phase':{'pol':np.array([None,-2.662604e-02,-3.996406e+00,-7.146266e+01])},'RV_phase':{'pol':1e-3*np.array([-1.995151e-01])},'FWHM_phase':{'pol':np.array([None,-1.119276e-01,-5.883027e+00,-6.312566e+01])}},
                #   '20170812':{'ctrst_phase':{'pol':np.array([None,7.488234e-03,-1.883012e+00,-7.162502e+01,1.697838e+03])},'RV_phase':{'pol':1e-3*np.array([-9.063176e-01,5.794139e+01,-4.939871e+03,1.856915e+05])},'FWHM_phase':{'pol':np.array([None,-9.618411e-02,-2.041933e+00,-4.688981e-01,-2.074621e+03])}}},

                #Fit Voigt, adamp fixed, final
                {'20170807':{ 'ctrst_phase':{'pol':np.array([None,-2.952423e-03,-3.605686e+00,-7.256693e+01])},'RV_phase':{'pol':1e-3*np.array([-4.114478e+02,4.536909e+01])},'FWHM_phase':{'pol':np.array([None,-1.448769e-01,-6.526284e+00,-6.507219e+01])}},
                  '20170812':{'ctrst_phase':{'pol':np.array([None,1.421886e-02,-2.287824e+00,-8.382571e+01,2.169159e+03])},'RV_phase':{'pol':1e-3*np.array([-4.063358e+02,7.269076e+01,-3.830395e+03,1.602711e+05])},'FWHM_phase':{'pol':np.array([None,-1.124903e-01,-1.035176e+00,2.013563e+01,-2.855368e+03])}}},
     

                              'HARPN':   

                # #New mask, fit gaussien                  
                # {'20150913':{'FWHM_snr':{'pol':np.array([None,-7.011227e-04,5.870110e-06])},'ctrst_snr':{'pol':np.array([None,2.389120e-03,-4.292582e-05,2.616092e-07])}},
                #   '20151101':{'FWHM_snr':{'pol':np.array([None,-2.268101e-04])},'ctrst_snr':{'pol':np.array([None,2.617238e-04])}}}
                                  
                #Kitcat mask, fit gaussien, final                  
                {'20150913':{'ctrst_snr':{'pol':np.array([5.171265e-01,3.472035e-04,-3.146707e-06])}},
                  '20151101':{'ctrst_snr':{'pol':np.array([5.177017e-01,1.327720e-04])}}},
                }

            
            
    elif gen_dic['star_name']=='WASP156':
        detrend_prof_dic['prop']={'CARMENES_VIS':
                {'20190928':{'FWHM_phase':{'pol':np.array([None,1.158613e-01,-9.515480e+00])}},
                 '20191025':{'FWHM_phase':{'pol':np.array([None,5.495522e-01,-1.267518e+01,7.688958e+01])}},
                    '20191210':{
                        'ctrst_phase':{'pol':np.array([None,3.091100e-01,1.040965e+01])},'ctrst_snr':{'pol':np.array([None,-1.852118e-02,4.441020e-04,-3.314824e-06])},  
                        # 'ctrst_phase':{'pol':np.array([None,8.939266e-02,2.234939e+01,4.940425e+02])},'ctrst_snr':{'pol':np.array([None,6.607536e-03,-5.302900e-05])},                       
                        'RV_phase':{'pol':1e-3*np.array([3.985214e+00,2.223849e+02])},
                        'FWHM_phase':{'pol':np.array([None,-1.875410e-01,-3.367617e+00])}}
                }}
    elif gen_dic['star_name']=='HD189733':
        detrend_prof_dic['prop']={'ESPRESSO':
                {'20210810':{'RV_phase':{'pol':1e-3*np.array([-2.685771e+01])},'FWHM_phase':{'pol':np.array([ 6.503003e-03])},'ctrst_phase':{'pol':np.array([-2.855554e-03])},'ctrst_snrQ':{'pol':np.array([-7.774156e-06])}},
                 '20210830':{'RV_phase':{'pol':1e-3*np.array([ 1.895454e+01])},'FWHM_phase':{'pol':np.array([-5.935886e-03])},'ctrst_phase':{'pol':np.array([ 3.161500e-03])},'ctrst_snrQ':{'pol':np.array([-4.384908e-06])}}}}
            
    elif gen_dic['star_name']=='WASP43':
        detrend_prof_dic['prop']={'NIRPS_HE':
                {'20230119':{'RV_phase':{'pol':1e-3*np.array([1.422554e+03])},'FWHM_phase':{'pol':np.array([-8.521608e-02])},'ctrst_phase':{'pol':np.array([1.957026e-01])}}}}

    elif gen_dic['star_name']=='HD209458':
        if ('new_F9' in gen_dic['CCF_mask']['ESPRESSO']):            
            if gen_dic['corr_wig']:
                detrend_prof_dic['prop']={'ESPRESSO':{'20190720':{'RV_phase':{'pol':1e-3*np.array([2.037544e+01])},'ctrst_snrQ':{'pol':np.array([-6.238248e-06])}},
                                                      '20190911':{                                                 'ctrst_snrQ':{'pol':np.array([-7.661844e-06])}}}}

                # #Perf Fbal                
                # detrend_prof_dic['prop']={'ESPRESSO':{'20190720':{'RV_phase':{'pol':1e-3*np.array([2.050853e+01])},'ctrst_snrQ':{'pol':np.array([-6.450480e-06])}},
                #                                       '20190911':{                                                 'ctrst_snrQ':{'pol':np.array([-7.697481e-06])}}}}                
                
            else:
                detrend_prof_dic['prop']={'ESPRESSO':{'20190720':{'RV_phase':{'pol':1e-3*np.array([2.006465e+01])},'ctrst_snrQ':{'pol':np.array([-6.412763e-06])}},
                                                      '20190911':{                                                 'ctrst_snrQ':{'pol':np.array([-7.634255e-06])}}}}                

        elif ('Relaxed' in gen_dic['CCF_mask']['ESPRESSO']):            
            detrend_prof_dic['prop']={'ESPRESSO':{'20190720':{'RV_phase':{'pol':1e-3*np.array([2.179469e+01])},                                               'ctrst_snrQ':{'pol':np.array([-6.736634e-06])}},
                                                    '20190911':{                                                 'FWHM_phase':{'pol':np.array([-3.817432e-03])},'ctrst_snrQ':{'pol':np.array([-7.659002e-06])}}}}            
        elif ('Strict' in gen_dic['CCF_mask']['ESPRESSO']):            
            detrend_prof_dic['prop']={'ESPRESSO':{'20190720':{'RV_phase':{'pol':1e-3*np.array([2.034672e+01])},                                               'ctrst_snrQ':{'pol':np.array([-7.066771e-06])}},
                                                    '20190911':{                                                 'FWHM_phase':{'pol':np.array([-4.517629e-03])},'ctrst_snrQ':{'pol':np.array([-9.481128e-06])}}}}            
    elif gen_dic['star_name']=='WASP76':
        if ('new_F9' in gen_dic['CCF_mask']['ESPRESSO']):  
            if gen_dic['corr_wig']:
                detrend_prof_dic['prop']={'ESPRESSO':{'20180902':{'RV_phase':{'pol':1e-3*np.array([3.480551e+01])},'FWHM_snrQ': {'pol':np.array([-1.321403e-05])},'ctrst_snrQ':{'pol':np.array([-7.496479e-06])}}}} 
                
                # #Perf. Fbal
                # detrend_prof_dic['prop']={'ESPRESSO':{'20180902':{'RV_phase':{'pol':1e-3*np.array([3.485136e+01])},'FWHM_snrQ': {'pol':np.array([-1.333959e-05])},'ctrst_snrQ':{'pol':np.array([-7.676005e-06])}}}}                  
            else:
                detrend_prof_dic['prop']={'ESPRESSO':{'20180902':{'RV_phase':{'pol':1e-3*np.array(['recalc. with rvres'])},'FWHM_snrQ': {'pol':np.array([-1.326882e-05])},'ctrst_snrQ':{'pol':np.array([-7.633751e-06])}}}} 
        elif ('Relaxed' in gen_dic['CCF_mask']['ESPRESSO']):            
            detrend_prof_dic['prop']={'ESPRESSO':{'20180902':{'RV_phase':{'pol':1e-3*np.array([3.151763e+01])},'FWHM_snrQ': {'pol':np.array([-9.633247e-06])},'ctrst_snrQ':{'pol':np.array([-9.708301e-06])}}}}  
        elif ('Strict' in gen_dic['CCF_mask']['ESPRESSO']):            
            detrend_prof_dic['prop']={'ESPRESSO':{'20180902':{'RV_phase':{'pol':1e-3*np.array([3.541942e+01])},'ctrst_snrQ':{'pol':np.array([-1.215972e-05])}}}}                                    
        
            
            
            
    #SNR orders             
    detrend_prof_dic['SNRorders']={}


    #PC correction 
    
    #Activating
    detrend_prof_dic['corr_PC'] = True  & False

    #PC coefficients from RMR
    if gen_dic['star_name']=='55Cnc':
        nPC=1
        detrend_prof_dic['PC_model']={
            # #Pour calcul vsys
            # 'ESPRESSO':{
            #     # '20200205':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/'+str(nPC)+'PC/ESPRESSO_20200205_mooncorr.npz'},                 
            #     '20200205':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/'+str(nPC)+'PC/ESPRESSO_20200205.npz'},
            #     '20210121':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/'+str(nPC)+'PC/ESPRESSO_20210121.npz'},            
            #     '20210124':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/'+str(nPC)+'PC/ESPRESSO_20210124.npz'}

            # #Pour correction res + fit joint CCF_Intr vis indiv
            #   'ESPRESSO':{
            #       '20200205':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/1PC/ESPRESSO_20200205_mooncorr_aligned.npz',
            #                   'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/mcmc/20200205_mooncorr_1PC/Fit_results.npz'},
            #       '20210121':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/2PC/ESPRESSO_20210121_aligned.npz',
            #                   'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/mcmc/20210121_2PC/run2/Fit_results.npz'},
            #       '20210124':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/2PC/ESPRESSO_20210124_aligned.npz',
            #                   'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/mcmc/20210124_2PC/run3/Fit_results.npz'}

            #Pour correction res + fit joint CCF_Intr all visits
            # 'ESPRESSO':{
            #       '20200205':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/1PC/ESPRESSO_20200205_mooncorr_aligned.npz',
            #                     'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/ESPRESSO/All_vis/Intr_only/1PC2PC3PC/Fit_results.npz'},
            #       '20210121':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/2PC/ESPRESSO_20210121_aligned.npz',
            #                     'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/ESPRESSO/All_vis/Intr_only/1PC2PC3PC/Fit_results.npz'},
            #       '20210124':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/3PC/ESPRESSO_20210124_aligned.npz',
            #                     'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/ESPRESSO/All_vis/Intr_only/1PC2PC3PC/Fit_results.npz'},

            #  },

            #Pour correction res + fit joint CCF_Intr all visits (UPDATE)
            'ESPRESSO':{
                  '20200205':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/5PC/ESPRESSO_20200205_mooncorr_aligned.npz',
                                'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/HARPS_ESPRESSO_HARPN/HARPScorrtrend_ESPRESSOwithOutPC_HARPNwithOutPCnoV4/Fit_results.npz'},
                  '20210121':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/6PC/ESPRESSO_20210121_aligned.npz',
                                'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/HARPS_ESPRESSO_HARPN/HARPScorrtrend_ESPRESSOwithOutPC_HARPNwithOutPCnoV4/Fit_results.npz'},
                  '20210124':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/3PC/ESPRESSO_20210124_aligned.npz',
                                'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/HARPS_ESPRESSO_HARPN/HARPScorrtrend_ESPRESSOwithOutPC_HARPNwithOutPCnoV4/Fit_results.npz'},

              },

              
            # #Pour calcul vsys
            # 'HARPS':{
            #     '20120127':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPS/Range200/'+str(nPC)+'PC/HARPS_20120127.npz'},
            #     '20120213':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPS/Range200/'+str(nPC)+'PC/HARPS_20120213.npz'},            
            #     '20120227':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPS/Range200/'+str(nPC)+'PC/HARPS_20120227.npz'},
            #     '20120315':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPS/Range200/'+str(nPC)+'PC/HARPS_20120315.npz'},
            #  },

            # #Pour verification props out
            # 'HARPS':{
            #     '20120127':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPS/Range200/'+str(nPC)+'PC/Aligned/HARPS_20120127.npz'},
            #     '20120213':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPS/Range200/'+str(nPC)+'PC/Aligned/HARPS_20120213.npz'},            
            #     '20120227':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPS/Range200/'+str(nPC)+'PC/Aligned/HARPS_20120227.npz'},
            #     '20120315':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPS/Range200/'+str(nPC)+'PC/Aligned/HARPS_20120315.npz'},
            #  },
            
            # #Pour calcul vsys
            # 'HARPN':{
            #     '20131114':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/'+str(nPC)+'PC/HARPN_20131114.npz'},
            #     '20131128':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/'+str(nPC)+'PC/HARPN_20131128.npz'},            
            #     '20140101':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/'+str(nPC)+'PC/HARPN_20140101.npz'},
            #     '20140126':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/'+str(nPC)+'PC/HARPN_20140126.npz'},
            #     '20140226':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/'+str(nPC)+'PC/HARPN_20140226.npz'},
            #     '20140329':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/'+str(nPC)+'PC/HARPN_20140329.npz'},
            #     },

            #Pour correction res + fit joint CCF_Intr vis indiv
            'HARPN':{
                '20131114':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/1PC/Aligned/HARPN_20131114.npz',
                            'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/HARPS_ESPRESSO_HARPN/HARPScorrtrend_ESPRESSOwithOutPC_HARPNwithOutPCnoV4/Fit_results.npz'},
                '20131128':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/1PC/Aligned/HARPN_20131128.npz',
                            'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/HARPS_ESPRESSO_HARPN/HARPScorrtrend_ESPRESSOwithOutPC_HARPNwithOutPCnoV4/Fit_results.npz'},         
                '20140101':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/4PC/Aligned/HARPN_20140101.npz',
                            'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/HARPN/Indiv_vis/With_outPC/PCA_200/20140101/Prior_global_fit/4PC/Fit_results.npz'},
                '20140126':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/4PC/Aligned/HARPN_20140126.npz',
                            'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/HARPS_ESPRESSO_HARPN/HARPScorrtrend_ESPRESSOwithOutPC_HARPNwithOutPCnoV4/Fit_results.npz'},
                '20140226':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/5PC/Aligned/HARPN_20140226.npz',
                            'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/HARPS_ESPRESSO_HARPN/HARPScorrtrend_ESPRESSOwithOutPC_HARPNwithOutPCnoV4/Fit_results.npz'},
                '20140329':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/3PC/Aligned/HARPN_20140329.npz',
                            'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/HARPS_ESPRESSO_HARPN/HARPScorrtrend_ESPRESSOwithOutPC_HARPNwithOutPCnoV4/Fit_results.npz'},
                },

            # #Pour calcul vsys
            # 'EXPRES':{
            #     '20220131':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/EXPRES/Range200_clean/'+str(nPC)+'PC/EXPRES_20220131.npz'},
            #     '20220406':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/EXPRES/Range200_clean/'+str(nPC)+'PC/EXPRES_20220406.npz'},            
            #   }, 
            #Pour correction res + fit joint CCF_Intr all visits            
            'EXPRES':{
                '20220131':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/EXPRES/Range200_clean/10PC/Aligned/EXPRES_20220131.npz',
                #             # 'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/EXPRES/Indiv_vis/With_outPC/PCA_200_clean/20220131/'+str(nPC)+'PC/Fit_results.npz'},
                            # 'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/EXPRES/All_vis/With_outPC/Commline/1010/Fit_results.npz'
                            },   
                '20220406':{'all':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/EXPRES/Range200_clean/10PC/Aligned/EXPRES_20220406.npz', 
                            # 'in':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/EXPRES/All_vis/With_outPC/Commline/1010/Fit_results.npz'
                            },              
              },             
        
        }
            
    #PC profiles
    detrend_prof_dic['idx_PC']={}
    # if gen_dic['star_name']=='55Cnc':
    #     detrend_prof_dic['idx_PC']={
    #         'ESPRESSO':{'20210121':[2]}}

    #Plots: 2D PC noise model 
    plot_dic['map_pca_prof']=''   #'png 


    
















##################################################################################################
#%%% Module: analyzing disk-integrated profiles
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

#%%%%% Constant data errors
#    - ESPRESSO/HARPS(N) pipeline fits are performed with constant errors, to mitigate the impact of large residuals from a gaussian fit in the line core or wing (as a gaussian can not be the correct physical model for the lines)
#    - by default, ANTARESS fits are performed using the propagated error table (if available, or the sqrt(flux) otherwise).
#      this option will set all errors in a fitted profile to the sqrt() of its average continuum flux
#    - errors, original or constant, can be scaled from input using gen_dic['g_err'] and re-running 'calc_proc_data', or locally for the module using data_dic['DI']['sc_err']
data_dic['DI']['cst_err']=False
data_dic['DI']['cst_errbin']= False


#%%%%% Scaled data errors
#    - local scaling of data errors
#    - scale by sqrt(reduced chi2 of original fit) to ensure a reduced chi2 unity
data_dic['DI']['sc_err']={}


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
#      excluded pixels must then fall within the shifted occ_range and line_range, wich defines the maximum extension of the disk-integrated stellar line
#    - only applied to original, unbinned visits
#    - define [rv1,rv2]] in the star rest frame
data_dic['DI']['occ_range']={} 
data_dic['DI']['line_range']={} 


#%%%%% Trimming 
#    - profiles will be trimmed over this range (defined in A or km/s) before being used for the fit
#    - this is mostly relevant for data in spectral mode
#    - leave empty to use full range 
#    - define [x1,x2] in the input data frame
data_dic['DI']['fit_prof']['trim_range']={}


#%%%%% Order to be fitted
#    - relevant for 2D spectra only
data_dic['DI']['fit_prof']['order']={}   


#%%%%% Continuum range
#    - used to set the continuum level of models in fits, and for the contrast correction of CCFs
#      unless requested as a variable parameter in 'mod_prop', the continuum level of the model is fixed to the value measured over 'cont_range'
#      see details in 'mod_prop' regarding the fitting of the continuum for in-transit profiles
#    - define [ [x1,x2] , [x3,x4] , [x5,x6] , ... ] in the input data frame
#      ranges will be automatically shifted to the star rest frame when relevant
data_dic['DI']['cont_range'] = {}


#%%%%% Spectral range(s) to be fitted
#    - define [ [x1,x2] , [x3,x4] , [x5,x6] , ... ] in the input data frame
#      ranges will be automatically shifted to the star rest frame when relevant
data_dic['DI']['fit_range']={}


#%%%% Direct measurements
#    - format is {prop_name:{options}}
#    - possibilities:
# + equivalent width: 'EW' : {'rv_range':[rv1,rv2] single range over which the integral is performed}                         
# + bissector: 'biss' : {'source':'obs' or 'mod',
#                        'rv_range':[rv1,rv2] maximum range over which bissector is calculated,
#                        'dF': flux resolution for line profile resampling,
#                        'Cspan': percentage of line contrast at which to measure bissector span (1 corresponds to line minimum); set to None to take maximum RV deviation from minimum}
data_dic['DI']['meas_prop']={}


#%%%% Line profile model   

#%%%%% Transition wavelength
#    - in the star rest frame
#    - used to center the line model, and the stellar / planetary exclusion ranges
#    - only relevant in spectral mode
#    - do not use if the spectral fit is performed on more than a single line
data_dic['DI']['line_trans']=None   


#%%%%% Instrumental convolution
#    - apply instrumental convolution or not (default) to model
#    - beware that most derived properties will correspond to the model before convolution
#    - should be set to True when using unconvolved profiles of the intrinsic line to fit the master DI
data_dic['DI']['conv_model']=False
 

#%%%%% Model type
#    - specific to each instrument
#    - options:
# + 'gauss': inverted gaussian, possibly skewed, absorbing polynomial continuum
# + 'gauss_poly': inverted gaussian , absorbing flat continuum with 6th-order polynomial at line center
# + 'dgauss': gaussian continuum added to inverted gaussian (very well suited to M dwarf CCFs),  absorbing polynomial continuum
# + 'voigt': voigt profile, absorbing polynomial continuum
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
#    - structure is mod_prop = { 'par_name' : { 'vary' : bool , 'inst' : { 'visit' : {'guess':X , 'bd':[Y,Z] } } } }
# > par_name is specific to the model selected
# > 'vary' indicates whether the parameter is fixed or variable
#   if 'vary' = True:
#       'guess' is the guess value of the parameter for a chi2 fit, also used in any fit to define default constraints
#       'bd' is the range from which walkers starting points are randomly drawn for a mcmc fit
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

#%%%%% Fitting mode 
#    - 'chi2', 'mcmc', ''
data_dic['DI']['fit_mod']='chi2'  


#%%%%% Printing fits results
data_dic['DI']['verbose']= False


#%%%%% Priors on variable properties
#    - structure is priors_CCF = { 'par_name' : {prior_mode: X, prior_val: Y} }
#      where par_name is specific to the model selected, and prior_mode is one of the possibilities defined below
#    - otherwise priors can be set to :
# > uniform ('uf') : 'low', 'high'
# > gaussian ('gauss') : 'val', 's_val'
# > asymetrical gaussian ('dgauss') : 'val', 's_val_low', 's_val_high'
#    - chi2 fit can only use uniform priors
#    - if left undefined, default uniform priors are used
data_dic['DI']['line_fit_priors']={}


#%%%%% Derived properties
data_dic['DI']['deriv_prop']=['']


#%%%%% Detection thresholds
#    - define area and amplitude thresholds for detection of stellar line (in sigma)
#    - require 'true_amp' or 'amp' in 'deriv_prop'
data_dic['DI']['thresh_area']=5.
data_dic['DI']['thresh_amp']=4.   


#%%%%% Force detection flag 
#    - set flag to True at relevant index for the CCFs to be considered detected, or false to force a non-detection
#    - indices for each dataset are relative to in-transit indices / binning properties
#    - leave empty for automatic detection
data_dic['DI']['idx_force_det']={}
data_dic['DI']['idx_force_detbin']={} 
data_dic['DI']['idx_force_detbinmultivis']={} 

    
#%%%%% MCMC settings

#%%%%%% Calculating/retrieving
#    - set to 'reuse' if gen_dic['calc_fit_intr']=True, allow changing nburn and error definitions without running the mcmc again
data_dic['DI']['mcmc_run_mode']='use'


#%%%%%% Walkers
#    - settings per instrument & visit
data_dic['DI']['mcmc_set']={}


#%%%%%% Walkers exclusion        
#    - excluding manually some of the walkers
#    - define conditions within routine
data_dic['DI']['exclu_walk']=  False           
    

#%%%%%% Derived errors
#    - 'quant' or 'HDI'
#    - if 'HDI' is selected:
# + by default a smoothed density profile is used to define HDI intervals
# + multiple HDI intervals can be avoided by defined the density profile as a histogram (by setting its resolution 'HDI_dbins') or by defining the bandwith factor of the smoothed profile ('HDI_bw')
data_dic['DI']['out_err_mode']='HDI'
data_dic['DI']['HDI']='1s'   


#%%%%%% Derived lower/upper limits
#    - format is {par:{'bound':val,'type':str,'level':[...]}}
# where 'bound' sets the limit, 'type' is 'upper' or 'lower', 'level' is a list of thresholds ('1s', '2s', '3s')
data_dic['DI']['conf_limits']={}   


#%%%% Plot settings

#%%%%% 1D PDF from mcmc
plot_dic['prop_DI_mcmc_PDFs']=''                 
    

#%%%%% Individual disk-integrated profiles
plot_dic['DI_prof']=''     


#%%%%% Residuals from disk-integrated profiles
plot_dic['DI_prof_res']=''   


#%%%%% Housekeeping and derived properties 
plot_dic['prop_raw']=''  




if __name__ == '__main__':  

    #Activating
    gen_dic['fit_DI'] = True    &  False
    gen_dic['fit_DIbin']=True   &  False
    gen_dic['fit_DIbinmultivis']=True    &  False
    if ((gen_dic['star_name'] in ['WASP76','HD209458','55Cnc']) and (gen_dic['type']['ESPRESSO']=='spec2D')): 
        gen_dic['fit_DI'] = False   #temporaire
        # gen_dic['fit_DIbin']=  True
    if gen_dic['star_name'] in ['HD189733','WASP43','L98_59','GJ1214']:
        gen_dic['fit_DI'] = True
        gen_dic['fit_DIbin']=True


    #Calculating/Retrieving
    gen_dic['calc_fit_DI']=True   #  &  False   
    gen_dic['calc_fit_DIbin']=True  # &  False  
    gen_dic['calc_fit_DIbinmultivis']=True    &  False  

    #Fitted data

    #Constant data errors
    data_dic['DI']['cst_err']=True   &  False
    data_dic['DI']['cst_errbin']=True   &  False

    #Scaled data errors
    if (gen_dic['star_name']=='WASP76') and gen_dic['fit_DIbin'] and not gen_dic['fit_DI']:data_dic['DI']['sc_err']={'ESPRESSO':{'20180902':np.sqrt(2240.8611793620207),'20181030':np.sqrt(3915.852685385398)}}

    #Occulted line exclusion
    if gen_dic['star_name']=='MASCARA1':data_dic['DI']['occ_range']['ESPRESSO']=[-25.,25.] 
    # if gen_dic['star_name']=='HD209458':
    #     data_dic['DI']['occ_range']['ESPRESSO']=[-10.,10.] 
    #     data_dic['DI']['line_range']['ESPRESSO']=[-13.,13.] 

    #Trimming 
    data_dic['DI']['fit_prof']['trim_range']={}


    #Order to be fitted
    # data_dic['DI']['fit_prof']['order']={'ESPRESSO':85} 
    data_dic['DI']['fit_prof']['order']={} 
    # data_dic['DI']['fit_prof']['order']={'ESPRESSO':0}   #mock dataset     
    if (gen_dic['star_name']=='WASP76') and gen_dic['trim_spec']:data_dic['DI']['fit_prof']['order']={'ESPRESSO':2}



    #Continuum range
    if gen_dic['transit_pl']=='WASP_8b':data_dic['DI']['cont_range']=[[-50,-1.5-10.],[-1.5+10.,50.]]  
    elif 'GJ436_b' in gen_dic['transit_pl']:
        data_dic['DI']['cont_range']=[[9.7-40.,9.7-15.],[9.7+15.,9.7+40.]]     
    elif gen_dic['star_name']=='55Cnc':
        sysguess = 0
        data_dic['DI']['cont_range']['ESPRESSO']=[[sysguess-80.,sysguess-30.],[sysguess+30.,sysguess+80.]] 
        data_dic['DI']['cont_range']['HARPS']=[[sysguess-80.,sysguess-30.],[sysguess+30.,sysguess+80.]] 
        data_dic['DI']['cont_range']['HARPN']=[[sysguess-80.,sysguess-30.],[sysguess+30.,sysguess+80.]] 
        data_dic['DI']['cont_range']['EXPRES']=[[sysguess-80.,sysguess-65.],[sysguess-42.,sysguess-37.],[sysguess+30.,sysguess+80.]] 
        sysguess = 27.
        data_dic['DI']['cont_range']['SOPHIE']=[[sysguess-80.,sysguess-30.],[sysguess+30.,sysguess+80.]] 
        
    elif gen_dic['transit_pl']=='WASP121b':
        data_dic['DI']['cont_range']=[[-10.,38.4-25.],[38.4+25.,90.]]         #mask G
        data_dic['DI']['cont_range']=[[38.5-98.2,38.5-53.3],[38.5+53.3,38.5+98.2]]     #mask F
        # data_dic['DI']['cont_range']=[[38.5-300.,38.5-53.3],[38.5+53.3,38.5+300.]]     #mask F avec atmo
        data_dic['DI']['cont_range']=[[38.5-300,38.5-40.],[38.5+40.,38.5+300.]]       #ESPRESSO
    elif gen_dic['transit_pl']=='Kelt9b':data_dic['DI']['cont_range']=[[-300.,-150.],[120.,300.]]     
    elif gen_dic['transit_pl']=='WASP127b':data_dic['DI']['cont_range']=[[-150.,-19.],[1.,150.]]       
    
    elif gen_dic['star_name']=='HD209458': 
        data_dic['DI']['cont_range']['ESPRESSO']=[[5800.,5889.55],[5890.35,5900.]]    #ANTARESS I, mock, multi-tr
        data_dic['DI']['cont_range']['ESPRESSO']={0:-14.8+np.array([[-80.,-20.],[20.,80.]])}    #ANTARESS I, CCF fit
        if gen_dic['trim_spec']:data_dic['DI']['cont_range']['ESPRESSO'] = np.array([[ 5883. , 5885.],[5901., 5903. ]])/1.000049    #ANTARESS fit sodium doublet        
    elif gen_dic['star_name']=='WASP76':
        data_dic['DI']['cont_range']['ESPRESSO']={0:[[-120.,-80.],[80.,120.]]}     #to have a symmetrical range accounting for the widest planetary exclusion
        if gen_dic['fit_DIbin'] and not gen_dic['fit_DI']:data_dic['DI']['cont_range']['ESPRESSO']=[[-60.,-20.],[20.,60.]]     #master is fully defined
        if gen_dic['trim_spec']:
            data_dic['DI']['cont_range']['ESPRESSO'] = np.array([[ 5885.6 , 5888.5],[5891.5, 5894.5 ]])    #ANTARESS fit NaID2     
            data_dic['DI']['cont_range']['ESPRESSO'] = np.array([[5847.3 , 5847.82],[5848.38 , 5848.75]])    
            data_dic['DI']['cont_range']['ESPRESSO'] = np.array([[5851.45 , 5851.86],[5852.55 ,5852.94]])  
            data_dic['DI']['cont_range']['ESPRESSO'] = np.array([[5852.5 ,5852.95],[5853.94,5854.11],[ 5854.55,5854.8]])              
            data_dic['DI']['cont_range']['ESPRESSO'] = np.array( [[5854.54 , 5854.82],[5855.33 , 5855.70]])             
            data_dic['DI']['cont_range']['ESPRESSO'] = np.array( [[5855.36 , 5855.72],[5856.30 , 5856.77]])            
            data_dic['DI']['cont_range']['ESPRESSO'] = np.array( [[5861.33 , 5861.83],[5863.04 , 5863.57]])               
            data_dic['DI']['cont_range']['ESPRESSO'] = np.array([[5865.22 , 5865.96],[5866.68 ,5866.83]])               
            data_dic['DI']['cont_range']['ESPRESSO'] = np.array([[5929.04 ,5929.45],[5930.46 ,5931.]])   



            
    elif gen_dic['star_name']=='HD3167': 
        data_dic['DI']['cont_range']=[[19.403622-30.-40.,19.403622-30.],[19.403622+30.,19.403622+30.+40.]] 
        # data_dic['DI']['cont_range']=[[19.403622-30.-40.-160.,19.403622-30.-160],[19.403622+30.-160,19.403622+30.+40.-160]]    #continu bleu
        # data_dic['DI']['cont_range']=[[19.403622-30.-40.-100.,19.403622-30.-100],[19.403622+30.-100,19.403622+30.+40.-100]]    #continu bleu close
        # data_dic['DI']['cont_range']=[[19.403622-30.-40.+100,19.403622-30.+100],[19.403622+30.+100,19.403622+30.+40.+100]]     #continu red close
        # data_dic['DI']['cont_range']=[[19.526350-30.-40.,19.526350-30.],[19.526350+30.,19.526350+30.+40.]] 
        # data_dic['DI']['cont_range']=[[19.526350-20.,19.526350-10.],[19.526350+10.,19.526350+20.]] 
        data_dic['DI']['cont_range']=[[-30.-40.,-30.],[30.,30.+40.]]    #avec custom mask dans ref star
    elif gen_dic['transit_pl']=='Corot7b':data_dic['DI']['cont_range']=[[-20.,31.059477-10.],[31.059477+10.,50.]] 
    elif gen_dic['transit_pl']=='Nu2Lupi_c':data_dic['DI']['cont_range']=[[-300.,-68.794511-10.],[-68.794511+10.,300.]] 
    elif gen_dic['star_name']=='GJ9827': 
        data_dic['DI']['cont_range']=[[-300.,31.953947-10.],[31.953947+10.,300.]]       #ESPRESSO, HARPS
    elif gen_dic['star_name']=='TOI858':data_dic['DI']['cont_range']=[[64.4-100.,64.4-15.],[64.4+15.,64.4+100.]]
    elif 'Moon' in gen_dic['transit_pl']:data_dic['DI']['cont_range']=[[-300.,-10.],[10.,300.]]
    elif 'TIC61024636b' in gen_dic['transit_pl']:data_dic['DI']['cont_range']=[[-300.,-10.],[10.,300.]]
    elif gen_dic['star_name']=='HIP41378': data_dic['DI']['cont_range']['HARPN']=[[50.-100.,50.-30.],[50.+30.,50.+100.]]    
    elif gen_dic['star_name']=='HD15337': data_dic['DI']['cont_range']=[[-300.,76.-30.],[76.+30.,300.]]    
    elif gen_dic['star_name']=='Altair': data_dic['DI']['cont_range']=[[-300.,-10.],[10.,300.]]           
    elif gen_dic['star_name']=='TOI-3362':data_dic['DI']['cont_range']=[[-300.,-10.],[10.,300.]]    
    elif 'Nu2Lupi_d' in gen_dic['transit_pl']:data_dic['DI']['cont_range']=[[-300.,-10.],[10.,300.]]
    elif gen_dic['star_name']=='K2-139':data_dic['DI']['cont_range']=[[-300.,-10.],[10.,300.]] 
    elif gen_dic['star_name']=='TIC257527578':data_dic['DI']['cont_range']=[[-300.,-10.],[10.,300.]]    
    elif gen_dic['star_name']=='MASCARA1':data_dic['DI']['cont_range']=[[-350.,-174.],[129.,175.]] 
    elif gen_dic['star_name']=='V1298tau':data_dic['DI']['cont_range']['HARPN']=[[14.-90.,14.-40.],[14.+40.,14.+90.]]     
    #RM survey
    elif gen_dic['star_name']=='HAT_P3':
        if ('KitCat' in gen_dic['data_dir_list']['HARPN']['20200130']):sysguess = 0.
        else:sysguess = -23. 
        data_dic['DI']['cont_range']['HARPN']=[[sysguess-75.,sysguess-20.],[sysguess+20.,sysguess+75.]] 
    elif gen_dic['star_name']=='Kepler25':
        if ('KitCat' in gen_dic['data_dir_list']['HARPN']['20190614']):sysguess = 0.
        else:sysguess = -8.6
        data_dic['DI']['cont_range']['HARPN']=[[sysguess-120.,sysguess-30.],[sysguess+30.,sysguess+120.]] 
    elif gen_dic['star_name']=='Kepler68':
        if ('KitCat' in gen_dic['data_dir_list']['HARPN']['20190803']):sysguess = 0.
        else:sysguess = -20.8      
        data_dic['DI']['cont_range']['HARPN']=[[sysguess-80.,sysguess-25.],[sysguess+25.,sysguess+80.]] 
    elif gen_dic['star_name']=='HAT_P33':
        if ('KitCat' in gen_dic['data_dir_list']['HARPN']['20191204']):sysguess = 0.
        else:sysguess = 23.           
        data_dic['DI']['cont_range']['HARPN']=[[sysguess-120.,sysguess-40.],[sysguess+40.,sysguess+120.]] 
    elif gen_dic['star_name']=='K2_105':
        if ('KitCat' in gen_dic['data_dir_list']['HARPN']['20200118']):sysguess = 0.
        else:sysguess = -32.       
        data_dic['DI']['cont_range']['HARPN']=[[sysguess-70.,sysguess-20.],[sysguess+20.,sysguess+70.]] 
    elif gen_dic['star_name']=='HD89345':
        if ('KitCat' in gen_dic['data_dir_list']['HARPN']['20200202']):sysguess = 0.
        else:sysguess = 2.2      
        data_dic['DI']['cont_range']['HARPN']=[[sysguess-80.,sysguess-20.],[sysguess+20.,sysguess+80.]]         
    elif gen_dic['star_name']=='Kepler63':
        if ('KitCat' in gen_dic['data_dir_list']['HARPN']['20200513']):sysguess = 0.
        else:sysguess = -23.   
        data_dic['DI']['cont_range']['HARPN']=[[sysguess-80.,sysguess-25.],[sysguess+25.,sysguess+80.]]   
    elif gen_dic['star_name']=='HAT_P49':
        if ('KitCat' in gen_dic['data_dir_list']['HARPN']['20200730']):sysguess = 0.
        else:sysguess = 14.
        data_dic['DI']['cont_range']['HARPN']=[[sysguess-180.,sysguess-55.],[sysguess+55.,sysguess+180.]]  
    elif gen_dic['star_name']=='WASP47':
        if ('KitCat' in gen_dic['data_dir_list']['HARPN']['20210730']):sysguess = 0.
        else:sysguess = -28.
        data_dic['DI']['cont_range']['HARPN']=[[sysguess-70.,sysguess-30.],[sysguess+30.,sysguess+70.]]          
    elif gen_dic['star_name']=='WASP107':
        if (('HARPS' in gen_dic['data_dir_list']) and ('KitCat' in gen_dic['data_dir_list']['HARPS']['20180313'])):
            data_dic['DI']['cont_range']['HARPS'] = [[-80.,-25.],[25.,80.]]
        else:
            sysguess = 14.     
            data_dic['DI']['cont_range']['HARPS'] = [[sysguess-80.,sysguess-25.],[sysguess+25.,sysguess+80.]]  
        data_dic['DI']['cont_range']['CARMENES_VIS'] = [[-100.,-50.],[50.,100.]]  

    elif gen_dic['star_name']=='WASP166':
        if ('KitCat' in gen_dic['data_dir_list']['HARPS']['20170114']):sysguess = 0.
        else:sysguess = 23.
        data_dic['DI']['cont_range']['HARPS']=[[sysguess-75.,sysguess-25.],[sysguess+25.,sysguess+75.]]         
    elif gen_dic['star_name']=='HAT_P11':
        if (('HARPN' in gen_dic['data_dir_list']) and ('KitCat' in gen_dic['data_dir_list']['HARPN']['20150913'])):sysguess=0.
        else:sysguess = -63.
        data_dic['DI']['cont_range']['HARPN'] = [[sysguess-90.,sysguess-30.],[sysguess+30.,sysguess+100.]]  
        data_dic['DI']['cont_range']['CARMENES_VIS'] = [[-110.,-40.],[40.,110.]]          
       
    elif gen_dic['star_name']=='WASP156':
        data_dic['DI']['cont_range']['CARMENES_VIS']=[[-150.,-50.],[50.,150.]]     
    elif gen_dic['star_name']=='HD106315':
        if ('KitCat' in gen_dic['data_dir_list']['HARPS']['20170309']):sysguess = 0.
        else:sysguess = -3.6
        data_dic['DI']['cont_range']['HARPS']=[[sysguess-150.,sysguess-60.],[sysguess+60.,sysguess+150.]]         
    elif gen_dic['star_name']=='GJ3090':
        sysguess = 0.
        data_dic['DI']['cont_range']['NIRPS_HA']=[[sysguess-150.,sysguess-60.],[sysguess+60.,sysguess+150.]]      
        data_dic['DI']['cont_range']['NIRPS_HE']=[[sysguess-150.,sysguess-60.],[sysguess+60.,sysguess+150.]]           
    elif gen_dic['star_name']=='HD189733':
        sysguess = 0.
        data_dic['DI']['cont_range']['ESPRESSO']=[[sysguess-80.,sysguess-25.],[sysguess+25.,sysguess+80.]]        
    #NIRPS
    elif gen_dic['star_name']=='WASP43':
        sysguess = 30.
        data_dic['DI']['cont_range']['NIRPS_HE']=[[sysguess-100.,sysguess-7.],[sysguess+7.,sysguess+100.]]      
    elif gen_dic['star_name']=='L98_59':
        sysguess = -15.834
        data_dic['DI']['cont_range']['NIRPS_HE']=[[sysguess-100.,sysguess-5.],[sysguess+5.,sysguess+100.]]   
    elif gen_dic['star_name']=='GJ1214':
        sysguess = 8.5
        data_dic['DI']['cont_range']['NIRPS_HE']=[[sysguess-100.,sysguess-5.],[sysguess+5.,sysguess+100.]]   
    

        
    #Spectral range(s) to be fitted
    if gen_dic['transit_pl']=='WASP_8b':
        data_dic['DI']['fit_range']=[[-50.,50.]] 
    elif 'GJ436_b' in gen_dic['transit_pl']:
        data_dic['DI']['fit_range']=[[-16.3,-3.6],[8.,12.],[23.,35.7]] #GJ436, single gaussian model
        data_dic['DI']['fit_range']=[[9.7-40.,9.7+40]]                 #GJ436, double gaussian model or single gaussian model without exclusion
    elif gen_dic['star_name']=='55Cnc':
        sysguess = 0.
        
        data_dic['DI']['fit_range']['ESPRESSO']={}     
        for vis in ['20200205','20210121','20210124','binned']:
            data_dic['DI']['fit_range']['ESPRESSO'][vis] = [[sysguess-70.,sysguess+70.]]
            
        data_dic['DI']['fit_range']['HARPS']={}     
        for vis in ['20120127','20120213','20120227','20120315','binned']:
            data_dic['DI']['fit_range']['HARPS'][vis] = [[sysguess-70.,sysguess+70.]]
            
        data_dic['DI']['fit_range']['HARPN']={}     
        for vis in ['20121225','20131114','20131128','20140101','20140126','20140226','20140329','binned']:
            data_dic['DI']['fit_range']['HARPN'][vis] = [[sysguess-70.,sysguess+70.]]

        data_dic['DI']['fit_range']['EXPRES']={}   
        data_dic['DI']['fit_range']['EXPRES']['20220131'] = [[sysguess-70.,sysguess-37.],[sysguess-22.,sysguess+70.]]
        data_dic['DI']['fit_range']['EXPRES']['20220406'] = [[sysguess-70.,sysguess-65.],[sysguess-42.,sysguess+70.]]
        data_dic['DI']['fit_range']['EXPRES']['binned'] = [[sysguess-70.,sysguess-65.],[sysguess-42.,sysguess-37.],[sysguess-22.,sysguess+70.]]

                        
        sysguess = 27.
        data_dic['DI']['fit_range']['SOPHIE']={}     
        for vis in ['20120202','20120203','20120205','20120217','20120219','20120222','20120225','20120302','20120324','20120327','20130303','binned']:
            data_dic['DI']['fit_range']['SOPHIE'][vis] = [[sysguess-70.,sysguess+70.]]       
        
    elif gen_dic['transit_pl']=='WASP121b':
        data_dic['DI']['fit_range']=[[-15.,90.]]         #mask G
        data_dic['DI']['fit_range']=[[38.5-98.29,38.5+98.29]]   #mask F  
        data_dic['DI']['fit_range']=[[38.5-300.,38.5+300.]]   #mask F avec atmo
    elif gen_dic['transit_pl']=='Kelt9b':
        data_dic['DI']['fit_range']=[[-300.,300.]]     
    elif gen_dic['transit_pl']=='WASP127b':
        data_dic['DI']['fit_range']=[[-150.,150.]]   
    elif gen_dic['star_name']=='HD209458':
        # data_dic['DI']['fit_range']=[[-150.,150.]] 
 
        data_dic['DI']['fit_range']['ESPRESSO']={'mock_vis':[[5000.,6000.]] }      #ANTARESS I, mock, multi-tr 
        data_dic['DI']['fit_range']['ESPRESSO']={'20190720':-14.8+np.array([[-80.,80.]]),'20190911':-14.8+np.array([[-80.,80.]])}       #ANTARESS I, CCF fit
        if gen_dic['trim_spec']:data_dic['DI']['fit_range']['ESPRESSO']['binned'] = np.array([[ 5885.6 ,5890.95 ],[5891.45,5892.55],[5893.1 ,5897. ],[5897.45, 5899.2 ],[5900.,5901.]])/1.000049   #Na doublet + continuum
    elif gen_dic['star_name']=='WASP76':
        data_dic['DI']['fit_range']['ESPRESSO']={'20180902':np.array([[-120.,120.]]),'20181030':np.array([[-120.,120.]]),'binned':np.array([[-120.,120.]])}       #ANTARESS I, CCF fit
        if gen_dic['fit_DIbin'] and not gen_dic['fit_DI']:data_dic['DI']['fit_range']['ESPRESSO']={'20180902':np.array([[-60.,60.]]),'20181030':np.array([[-60.,60.]])}    #master is fully defined
        if gen_dic['trim_spec']:
            for vis in data_dic['DI']['fit_range']['ESPRESSO']:
                data_dic['DI']['fit_range']['ESPRESSO'][vis] = np.array([[ 5885.6 ,5890.95 ],[5891.45,5892.55],[5893.1 ,5894.5 ]])   #NaID2 line 
                data_dic['DI']['fit_range']['ESPRESSO'][vis] = np.array([[5847.3 , 5848.75]])   
                data_dic['DI']['fit_range']['ESPRESSO'][vis] = np.array([[5851.45 , 5852.94]])  
                data_dic['DI']['fit_range']['ESPRESSO'][vis] = np.array([[5852.5 ,5852.95],[5853.5,5854.11],[ 5854.55,5854.8]])  
                data_dic['DI']['fit_range']['ESPRESSO'][vis] = np.array([[5854.54 , 5855.70]]) 
                data_dic['DI']['fit_range']['ESPRESSO'][vis] = np.array([[5855.36 , 5856.77]]) 
                data_dic['DI']['fit_range']['ESPRESSO'][vis] = np.array([[5861.33 , 5862.46],[5863.04 , 5863.57]]) 
                data_dic['DI']['fit_range']['ESPRESSO'][vis] = np.array([[5865.22 , 5865.96],[5866.34 , 5866.83]]) 
                data_dic['DI']['fit_range']['ESPRESSO'][vis] = np.array([[5929.04 ,5929.45],[5929.91 ,5931.]]) 



    elif gen_dic['transit_pl']=='HAT_P3b':data_dic['DI']['fit_range']=[[-30.,17]] 
    elif gen_dic['star_name']=='HD3167': 
        data_dic['DI']['fit_range']=[[19.403622-30.-40.,19.403622+30.+40.]] 
        data_dic['DI']['fit_range']=[[19.526350-30.-40.,19.526350+30.+40.]] 
        # data_dic['DI']['fit_range']=[[19.526350-500.,19.526350+500.]] 
        data_dic['DI']['fit_range']=[[-30.-40.,30.+40.]]    #avec custom mask dans ref star
    elif gen_dic['transit_pl']=='Corot7b':data_dic['DI']['fit_range']=[[-150.,150.]]
    elif gen_dic['transit_pl']=='Nu2Lupi_c':data_dic['DI']['fit_range']=[[-150.,150.]]
    elif gen_dic['star_name']=='GJ9827':data_dic['DI']['fit_range']=[[-150.,150.]]  
    elif gen_dic['star_name']=='TOI858':data_dic['DI']['fit_range']=[[-150.,150.]]  
    elif 'Moon' in gen_dic['transit_pl']:data_dic['DI']['fit_range']=[[-100.,100.]]
    elif gen_dic['star_name']=='HIP41378':data_dic['DI']['fit_range']['HARPN']={'20191218':[[50.-100.,50.+100.]], '20220401':[[50.-100.,50.+100.]]}  
    elif gen_dic['star_name']=='HD15337': data_dic['DI']['fit_range']=[[76.-70,76.+70.]]   
    elif gen_dic['star_name']=='MASCARA1':
        # data_dic['DI']['fit_range']=[[-350.,-174.],[6.3 + (6.3+130),190.]]    #fit continu, mauvaises bandes exclues        
        data_dic['DI']['fit_range']=[[-350.,-174.],[-100.,190.]]    #fit global, mauvaises bandes exclues
    elif gen_dic['star_name']=='V1298tau':data_dic['DI']['fit_range']['HARPN']=[[14.-90.,14.+90.]]  
    #RM survey
    elif gen_dic['star_name']=='HAT_P3':
        fit_range = [[sysguess-75.,sysguess+75.]]
        data_dic['DI']['fit_range']['HARPN']={'20190415':fit_range,'20200130':fit_range}
    elif gen_dic['star_name']=='Kepler25':data_dic['DI']['fit_range']['HARPN']={'20190614':[[sysguess-120.,sysguess+120.]]}
    elif gen_dic['star_name']=='Kepler68':data_dic['DI']['fit_range']['HARPN']={'20190803':[[sysguess-80.,sysguess+80.]]}
    elif gen_dic['star_name']=='HAT_P33':
        data_dic['DI']['fit_range']['HARPN']={'20191204':[[sysguess-120.,sysguess+120.]]}
    elif gen_dic['star_name']=='K2_105':data_dic['DI']['fit_range']['HARPN']={'20200118':[[sysguess-70.,sysguess+70.]]}
    elif gen_dic['star_name']=='HD89345':data_dic['DI']['fit_range']['HARPN']={'20200202':[[sysguess-80.,sysguess+80.]]}
    elif gen_dic['star_name']=='Kepler63':data_dic['DI']['fit_range']['HARPN']={'20200513':[[sysguess-80.,sysguess+80.]]}
    elif gen_dic['star_name']=='HAT_P49':data_dic['DI']['fit_range']['HARPN']={'20200730':[[sysguess-180.,sysguess+180.]]}
    elif gen_dic['star_name']=='WASP47':data_dic['DI']['fit_range']['HARPN']={'20210730':[[sysguess-70.,sysguess+70.]]}
    elif gen_dic['star_name']=='WASP107':
        if (('HARPS' in gen_dic['data_dir_list']) and ('KitCat' in gen_dic['data_dir_list']['HARPS']['20180313'])):sysguess=0.
        else:sysguess = 14.  
        fit_range = [[sysguess-100.,sysguess+100.]] 
        data_dic['DI']['fit_range']['HARPS']={'20140406':fit_range,'20180201':fit_range,'20180313':fit_range,'binned':fit_range}
        data_dic['DI']['fit_range']['CARMENES_VIS'] ={'20180224': [[-100.,100.]],'binned':[[-100.,100.]] }  
    elif gen_dic['star_name']=='WASP166':
        fit_range = [[sysguess-75.,sysguess+75.]]
        data_dic['DI']['fit_range']['HARPS']={'20170114':fit_range,'20170304':fit_range,'20170315':fit_range,'binned':fit_range}
    elif gen_dic['star_name']=='HAT_P11':
        if (('HARPN' in gen_dic['data_dir_list']) and ('KitCat' in gen_dic['data_dir_list']['HARPN']['20150913'])) or (('CARMENES_VIS' in gen_dic['data_dir_list'])):sysguess=0.
        else:sysguess = -63.
        fit_range = [[sysguess-90.,sysguess+90.]]
        data_dic['DI']['fit_range']['HARPN']={ '20150913':fit_range,'20151101':fit_range,'binned':fit_range}
        fit_range =  [[-90.,90.]]  
        data_dic['DI']['fit_range']['CARMENES_VIS'] ={ '20170807':fit_range,'20170812':fit_range,'binned':fit_range}
    elif gen_dic['star_name']=='WASP156':
        sysguess=0.
        fit_range = [[sysguess-100.,sysguess+100.]]
        data_dic['DI']['fit_range']['CARMENES_VIS']={'20190928':fit_range,'20191025':fit_range,'20191210':fit_range}
        # data_dic['DI']['fit_range']['CARMENES_VIS']['20190928'] = [[-100.,13.],[30.,100.]]#test exclusion feature     
        # data_dic['DI']['fit_range']['CARMENES_VIS']['20191025'] = [[-110.,-30.],[-10.,10.],[30.,110.]]               #test exclusion feature     
        data_dic['DI']['fit_range']['CARMENES_VIS']['20191210'] =[[-100.,-55.],[-20.,100.]]   #test exclusion feature     
        
    elif gen_dic['star_name']=='HD106315':
        fit_range = [[sysguess-150.,sysguess+150.]]
        data_dic['DI']['fit_range']['HARPS']={'20170309':fit_range,'20170330':fit_range,'20180323':fit_range,'binned':fit_range}

    elif gen_dic['star_name']=='HD189733':
        data_dic['DI']['fit_range']['ESPRESSO']={'20210810':[[sysguess-100.,sysguess+100.]],'20210830':[[sysguess-100.,sysguess+100.]]}       
    #NIRPS
    elif gen_dic['star_name']=='GJ3090':
        fit_range = [[sysguess-150.,sysguess+150.]]
        data_dic['DI']['fit_range']['NIRPS_HA']={'20221202':[[sysguess-150.,sysguess+150.]]}
        data_dic['DI']['fit_range']['NIRPS_HE']={'20221201':[[sysguess-150.,sysguess+150.]]}
    elif gen_dic['star_name']=='WASP43':
        data_dic['DI']['fit_range']['NIRPS_HE']={'20230119':[[sysguess-100.,sysguess+100.]]}
    elif gen_dic['star_name']=='L98_59':
        data_dic['DI']['fit_range']['NIRPS_HE']={'20230411':[[sysguess-100.,sysguess+100.]]} 
    elif gen_dic['star_name']=='GJ1214':
        data_dic['DI']['fit_range']['NIRPS_HE']={'20230407':[[sysguess-100.,sysguess+100.]]}



    
    #Direct measurements
    # data_dic['DI']['meas_prop']={
    #     # 'EW':{'rv_range':[-5.,5.]},
    #     'biss':{'source':'obs','rv_range':[-50.,50.],'dF':0.001,'Cspan':None}
    #     }


    #Line profile model   
    
    #Transition wavelength
    data_dic['DI']['line_trans']=None
    # data_dic['DI']['line_trans']=5889.95094   #mock dataset    
    if (gen_dic['star_name']=='WASP76') and gen_dic['trim_spec']:
        # data_dic['DI']['line_trans']=5889.95094   #NaID2
        #Lines from relaxed mask
        # data_dic['DI']['line_trans']=5848.1155771291    
        # data_dic['DI']['line_trans']=5852.2195772402   
        # data_dic['DI']['line_trans']=5853.6825772798   
        # data_dic['DI']['line_trans']=5855.0785773176   
        # data_dic['DI']['line_trans']=5856.0915773451   
        # data_dic['DI']['line_trans']=5862.3625775149   
        # data_dic['DI']['line_trans']=5866.4565776258      
        data_dic['DI']['line_trans']=5930.1875793517   


   
    #Instrumental convolution
    data_dic['DI']['conv_model']=False
    if 'GJ436_b' in gen_dic['transit_pl']:data_dic['DI']['conv_model'] = True  &  False
    if gen_dic['star_name']=='MASCARA1':data_dic['DI']['conv_model'] = True   &  False
    if (gen_dic['star_name']=='55cnc') and gen_dic['fit_DIbinmultivis']:data_dic['DI']['conv_model'] = True   
    if (gen_dic['star_name']=='HD209458') and gen_dic['trim_spec'] and gen_dic['fit_DIbinmultivis']:data_dic['DI']['conv_model'] = True       

            

    #Model type    
    if 'GJ436_b' in gen_dic['transit_pl']:data_dic['DI']['model']='dgauss'
    elif gen_dic['star_name']=='55Cnc':
        for inst in ['ESPRESSO','HARPS','HARPN','SOPHIE']:data_dic['DI']['model'][inst]='gauss'
                
        if gen_dic['fit_DIbinmultivis']:
            data_dic['DI']['model']['ESPRESSO']='custom'       
            
    elif gen_dic['transit_pl']=='WASP121b':
        data_dic['DI']['model']='gauss'          #ESPRESSO
        # data_dic['DI']['model']='dgauss'   #mask F
    elif gen_dic['transit_pl']=='Kelt9b':
        data_dic['DI']['model']='gauss'
       
    elif gen_dic['star_name']=='MASCARA1':
        data_dic['DI']['model']='custom'
    elif gen_dic['star_name']=='V1298tau':
        data_dic['DI']['model']['HARPN']='custom'
    elif gen_dic['star_name'] in ['WASP_8','55Cnc','WASP127','Corot7','Nu2Lupi','GJ9827','HIP41378']:
        for inst in ['ESPRESSO','HARPS','HARPN']:data_dic['DI']['model'][inst]='gauss'  
    #RM survey
    elif gen_dic['star_name'] in ['HAT_P3','Kepler25','Kepler68','HAT_P33','K2_105','HD89345','Kepler63','HAT_P49','WASP47','WASP107','WASP166','HAT_P11','WASP156','HD106315']:
        for inst in ['HARPS','HARPN']:data_dic['DI']['model'][inst]='gauss'
        if (gen_dic['star_name'] in ['WASP107','HAT_P11','WASP156']):data_dic['DI']['model']['CARMENES_VIS']='voigt'
        if (gen_dic['star_name'] in ['HAT_P11']) and ('HARPN' in gen_dic['data_dir_list']) and ('new' in gen_dic['data_dir_list']['HARPN']['20150913']):data_dic['DI']['model']['HARPN']='voigt'
        if (gen_dic['star_name'] in ['WASP107']) and ('HARPS' in gen_dic['data_dir_list']) and ('K5NormSqrt' in gen_dic['data_dir_list']['HARPS']['20180313']):data_dic['DI']['model']['HARPS']='voigt'


        # #Fit DI with Intr
        # for inst in ['HARPS','HARPN','CARMENES_VIS']:data_dic['DI']['model'][inst]='custom'
    elif gen_dic['star_name']=='HD189733':
        data_dic['DI']['model']['ESPRESSO']='gauss'    
    #NIRPS
    elif gen_dic['star_name'] in ['GJ3090','WASP43','L98_59','GJ1214']:
        data_dic['DI']['model']['NIRPS_HA']='gauss'
        data_dic['DI']['model']['NIRPS_HE']='gauss'
    elif (gen_dic['star_name']=='HD209458'):
        if gen_dic['trim_spec'] and gen_dic['fit_DIbinmultivis']:data_dic['DI']['model']['ESPRESSO']='custom'
        else:
            data_dic['DI']['model']['ESPRESSO']='gauss'
            
            # #Fit DI with Intr
            # data_dic['DI']['model']['ESPRESSO']='custom'  

    elif (gen_dic['star_name']=='WASP76'):
        data_dic['DI']['model']['ESPRESSO']='gauss'
        
        # #Fit DI with Intr ana
        # data_dic['DI']['model']['ESPRESSO']='custom'  



    #Intrinsic line properties  
    if gen_dic['star_name']=='MASCARA1': 
        
        #Define intrinsic stellar profile
        data_dic['DI']['mod_def']['ESPRESSO']={
        
            # 'mode':'imp',
            # 'path':{inst:{vis:'XXX.dat'}},
    
            'mode':'ana','coord_line':'mu','func_prof_name':'gauss',    

            # 'mode':'Intrbin',
            #'coord_line':'r_proj'
            #'vis':''


        }        


    elif gen_dic['star_name']=='V1298tau': 
        data_dic['DI']['mod_def']['HARPN']={
            'mode':'ana','coord_line':'mu','func_prof_name':'gauss'} 
        
    elif gen_dic['star_name']=='HD209458': 
        if gen_dic['trim_spec'] and gen_dic['fit_DIbinmultivis']:
            data_dic['DI']['mod_def']['ESPRESSO']={
                'mode':'theo',
            } 
        else:
            data_dic['DI']['mod_def']['ESPRESSO']={
                #Fit DI with Intr model 
                'mode':'ana','coord_line':'r_proj','func_prof_name':'gauss','pol_mode':'modul'
                } 
            data_dic['DI']['conv_model'] = True 

    elif (gen_dic['star_name']=='WASP76') and (data_dic['DI']['model']['ESPRESSO']=='custom'): 
        data_dic['DI']['mod_def']['ESPRESSO']={'mode':'ana','coord_line':'r_proj','func_prof_name':'gauss','pol_mode':'modul'} 
        data_dic['DI']['conv_model'] = True 


        
    elif gen_dic['star_name']=='55Cnc':
        for inst in ['ESPRESSO','HARPS','HARPN','SOPHIE']:
            if (inst in data_dic['DI']['model']) and (data_dic['DI']['model'][inst])=='custom':

                #Fit DI with Intr model 
                data_dic['DI']['mod_def'][inst]={'mode':'ana','coord_line':'mu','pol_mode':'abs','func_prof_name':'gauss'} 
                data_dic['DI']['conv_model'] = True 
            
                # #Fit DI with Intr profile
                # data_dic['DI']['mod_def'][inst]={'mode':'Intrbin','coord_line':'mu'}
                # data_dic['DI']['mod_def'][inst]['vis']=''
                # # data_dic['DI']['mod_def'][inst]['vis']='binned'
                # data_dic['DI']['conv_model'] = False

    #RM survey
    elif gen_dic['star_name'] in ['HAT_P3','Kepler25','Kepler68','HAT_P33','K2_105','HD89345','Kepler63','HAT_P49','WASP47','WASP107','WASP166','HAT_P11','WASP156','HD106315']:
        for inst in ['HARPS','HARPN','CARMENES_VIS']:
            if (inst in data_dic['DI']['model']) and (data_dic['DI']['model'][inst])=='custom':
          
                #Fit DI with Intr model 
                data_dic['DI']['mod_def'][inst]={'mode':'ana','coord_line':'r_proj','pol_mode':'abs'}
                if inst in ['HARPS','HARPN']:data_dic['DI']['mod_def'][inst]['func_prof_name'] = 'gauss' 
                elif inst=='CARMENES_VIS':data_dic['DI']['mod_def'][inst]['func_prof_name'] = 'voigt'   
                data_dic['DI']['conv_model'] = True 
            
                # #Fit DI with Intr profile
                # data_dic['DI']['mod_def'][inst]={'mode':'Intrbin','coord_line':'r_proj'}
                # data_dic['DI']['mod_def'][inst]['vis']=''
                # # data_dic['DI']['mod_def'][inst]['vis']='binned'
                # data_dic['DI']['conv_model'] = False
                         
      
                

    #Fixed/variable properties
    # if gen_dic['transit_pl']=='GJ436_b':
    #     data_dic['DI']['mod_prop']={}        
    #     #data_dic['DI']['mod_prop']={'HARPN':{'2016-03-18':{'RV_l2c':0.,'amp_l2c':0.5187,'FWHM_l2c':1.828},          #GJ436b, sans correction de couleur
    #     #                            '2016-04-11':{'RV_l2c':0.,'amp_l2c':0.5185,'FWHM_l2c':1.833}},
    #     #                 'HARPS':{'2007-05-09':{'RV_l2c':0.,'amp_l2c':0.5359,'FWHM_l2c':1.809}}}
    #     #data_dic['DI']['mod_prop']={'HARPN':{'2016-03-18':{'RV_l2c':0.,'amp_l2c':0.5186,'FWHM_l2c':1.830},          #GJ436b, avec correction de couleur
    #     #                            '2016-04-11':{'RV_l2c':0.,'amp_l2c':0.5180,'FWHM_l2c':1.831}},
    #     #                 'HARPS':{'2007-05-09':{'RV_l2c':0.,'amp_l2c':0.5357,'FWHM_l2c':1.808}},
    #     #                 'binned':{'HARPSN-binned':{'RV_l2c':0.,'amp_l2c':0.5183,'FWHM_l2c':1.8308}}}
    #     # data_dic['DI']['mod_prop']={'HARPN':{'2016-03-18':{'RV_l2c': 1.062e-2,'amp_l2c':0.5183,'FWHM_l2c':1.831},          #GJ436b, avec correction de couleur, et redshift
    #     #                             '2016-04-11':{'RV_l2c':1.094e-2,'amp_l2c':0.5177,'FWHM_l2c':1.832}},
    #     #                  'HARPS':{'2007-05-09':{'RV_l2c':1.692e-2,'amp_l2c':0.5354,'FWHM_l2c':1.809}},
    #     #                  'binned':{'HARPSN-binned':{'RV_l2c':1.0839e-2,'amp_l2c':0.5180,'FWHM_l2c':1.8318}}}
    #     data_dic['DI']['mod_prop']={'ESPRESSO':{'2019-02-28':{'RV_l2c':-6.95668e-04,'amp_l2c':9.95784e-01,'FWHM_l2c':1.00268},
    #                                         '2019-04-29':{'RV_l2c':-6.90909e-04,'amp_l2c':9.95757e-01,'FWHM_l2c':1.00269}}}
        
    if 'GJ436_b' in gen_dic['transit_pl']:          
        data_dic['DI']['mod_prop']={}
        
    
        #General guess, and bounds for MCMC
        data_dic['DI']['mod_prop'].update({'rv':{'vary':True,'ESPRESSO':{'20190228':{'guess':10.,'bd':[5.,15.]},'20190429':{'guess':10.,'bd':[5.,15.]}}},
                                        'ctrst':{'vary':True,'ESPRESSO':{'20190228':{'guess':0.3,'bd':[0.2,0.4]},'20190429':{'guess':0.3,'bd':[0.2,0.4]}}},
                                        'FWHM':{'vary':True,'ESPRESSO':{'20190228':{'guess':5.,'bd':[2.,8.]},'20190429':{'guess':5.,'bd':[2.,8.]}}}})      
        data_dic['DI']['mod_prop']['rv'].update({'HARPN':{'20160318':{'guess':10.},'20160411':{'guess':10.}},'HARPS':{'20070509':{'guess':10.}}})
        data_dic['DI']['mod_prop']['ctrst'].update({'HARPN':{'20160318':{'guess':0.3},'20160411':{'guess':0.3}},'HARPS':{'20070509':{'guess':0.3}}})
        data_dic['DI']['mod_prop']['FWHM'].update({'HARPN':{'20160318':{'guess':5.},'20160411':{'guess':5.}},'HARPS':{'20070509':{'guess':5.}}})      
         
        #----------------
        #Fixed template, erreur propagee, valeurs issues de fit en chi2
        #----------------
        
        #No sky-correction
        if len(gen_dic['fibB_corr'])==0:
        
            # #New mask, averaged out-of-tr properties
            # data_dic['DI']['mod_prop']={'RV_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':-1.12259e-03},'20190429':{'guess':-9.18669e-04}}},
            #                           'amp_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':9.74085e-01},'20190429':{'guess':9.79098e-01}}},
            #                           'FWHM_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':1.01791e+00},'20190429':{'guess':1.01448e+00}}}}   
        
            # #New mask, averaged out-of-tr properties, erreur propagee x10
            # data_dic['DI']['mod_prop']={'RV_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':-1.14321e-03},'20190429':{'guess':-9.06093e-04}}},
            #                           'amp_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':9.73590e-01},'20190429':{'guess':9.79375e-01}}},
            #                           'FWHM_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':1.01823e+00 },'20190429':{'guess':1.01429e+00}}}}       

            #New mask, derived from DI master after keplerian correction
            data_dic['DI']['mod_prop']={'RV_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':-1.6880695747e-04},'20190429':{'guess':-2.1454740809e-04}}},
                                      'amp_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':9.9608337556e-01},'20190429':{'guess':9.9508301997e-01}}},
                                      'FWHM_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':1.0026832897e+00},'20190429':{'guess':1.0033628022e+00}}}}   
        
        
        #Sky-correction
        else:

            #New mask, averaged out-of-tr properties            
            # data_dic['DI']['mod_prop']={'RV_l2c':{'vary':True,'ESPRESSO':{'20190228':{'guess':-1.07515e-03},'20190429':{'guess':-8.94843e-04},'binned':{'guess':0.5*(-1.07515e-03-8.94843e-04)}}},
            #                           'amp_l2c':{'vary':True,'ESPRESSO':{'20190228':{'guess':9.76438e-01},'20190429':{'guess':9.80422e-01},'binned':{'guess':0.5*(9.76438e-01+9.80422e-01)}}},
            #                           'FWHM_l2c':{'vary':True,'ESPRESSO':{'20190228':{'guess':1.01636e+00},'20190429':{'guess':1.01356e+00},'binned':{'guess':0.5*(1.01636e+00+1.01356e+00)}}}}      
    
        
            #Old mask, averaged out-of-tr properties
            # data_dic['DI']['mod_prop']={'RV_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':-7.26956e-04},'20190429':{'guess':-7.05481e-04}}},
            #                           'amp_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':9.95652e-01},'20190429':{'guess':9.95715e-01}}},
            #                           'FWHM_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':1.00288e+00},'20190429':{'guess':1.00283e+00}}}}      

            # #Old mask, derived from DI master after keplerian correction
            # data_dic['DI']['mod_prop']={'RV_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':-2.0770398737e-04},'20190429':{'guess':-1.3411191145e-04}}},
            #                           'amp_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':9.9875721488e-01},'20190429':{'guess':9.9918365186e-01}}},
            #                           'FWHM_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':1.0008200178e+00},'20190429':{'guess':1.0005368285e+00}}}}   
    
            #New mask, derived from DI master after keplerian correction
            data_dic['DI']['mod_prop'].update({'RV_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':-2.3686128554e-04},'20190429':{'guess':-1.8285586781e-04},'binned':{'guess':None}}},
                                            'amp_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':9.9477541529e-01},'20190429':{'guess':9.9597117963e-01},'binned':{'guess':None}}},
                                            'FWHM_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':1.0035865365e+00},'20190429':{'guess':1.0027560641e+00},'binned':{'guess':None}}}})              

        #----------------

        #HARPS/HARPS-N uncorrected for snr correlations        
        if not gen_dic['detrend_prof']:
            data_dic['DI']['mod_prop']['RV_l2c'].update({'HARPN':{'20160318':{'guess':-3.3510189653e-02},'20160411':{'guess':-3.1743326463e-02}},
                                                      'HARPS':{'20070509':{'guess':-5.3893910472e-03}}})
            data_dic['DI']['mod_prop']['amp_l2c'].update({'HARPN':{'20160318':{'guess':7.3522272825e-01},'20160411':{'guess':7.4445209977e-01}},
                                                        'HARPS':{'20070509':{'guess':9.4729356648e-01}}})
            data_dic['DI']['mod_prop']['FWHM_l2c'].update({'HARPN':{'20160318':{'guess':1.2254739251e+00},'20160411':{'guess':1.2148813565e+00}},
                                                        'HARPS':{'20070509':{'guess':1.0338761874e+00}}})              

        #HARPS/HARPS-N corrected for snr correlations
        if gen_dic['detrend_prof']:
            data_dic['DI']['mod_prop']['RV_l2c'].update({'HARPN':{'20160318':{'guess':-3.3511175497e-02},'20160411':{'guess':-3.1744061734e-02}},
                                                      'HARPS':{'20070509':{'guess':-4.7232827857e-03}}})
            data_dic['DI']['mod_prop']['amp_l2c'].update({'HARPN':{'20160318':{'guess':7.3521731614e-01},'20160411':{'guess':7.4444730731e-01}},
                                                        'HARPS':{'20070509':{'guess':9.5392977894e-01}}})
            data_dic['DI']['mod_prop']['FWHM_l2c'].update({'HARPN':{'20160318':{'guess':1.2254799712e+00},'20160411':{'guess':1.2148866936e+00}},
                                                        'HARPS':{'20070509':{'guess':1.0294427603e+00}}}) 
    
    


        # #Compa DI/intr
        # data_dic['DI']['mod_prop']={'veq':{'vary':True,'ESPRESSO':{'binned':{'guess':0.3,'bd':[0.1,0.4]}}}}
                      
        # #Erreur propagee, chi2, skycorr
        # data_dic['DI']['mod_prop'].update({'RV_l2c__IS__VS_':{'vary':False,'guess':0.5*(-1.07515e-03-8.94843e-04)},
        #                                                 'amp_l2c__IS__VS_':{'vary':False,'guess':0.5*(9.76438e-01+9.80422e-01)},
        #                                                 'FWHM_l2c__IS__VS_':{'vary':False,'guess':0.5*(1.01636e+00+1.01356e+00)}})   
        # #Best-fit intrinsic model
        # data_dic['DI']['mod_prop'].update({'ctrst_ord0__IS__VS_':{'vary':False,'guess':0.5*(3.4345735952e-01+3.8264295042e-01),'bd':[0.1,0.6]},
        #                                                 'FWHM_ord0__IS__VS_':{'vary':False,'guess':0.5*(4.6482430376e+00+4.3530235308e+00),'bd':[1.,3.]}})
     



  
    
    # elif gen_dic['transit_pl']=='WASP121b':    
    #     data_dic['DI']['mod_prop']={}   
    # #     if data_dic['DI']['model']=='dgauss':   #Proprietes fixees aux valeurs derivees des Mout (apres avoir fixe le modele des nuits pour deriver ceux des nuits binnees)
    
    # #         data_dic['DI']['mod_prop']={
    # #                 'HARPS':{'09-01-18':{'amp_l2c':0.532,'FWHM_l2c':1.32,'RV_l2c':0.305},    #green
    # #                          '14-01-18':{'amp_l2c':0.556,'FWHM_l2c':1.30,'RV_l2c':0.305},    #blue
    # #                          '31-12-17':{'amp_l2c':0.494,'FWHM_l2c':1.37,'RV_l2c':0.386}},   #red  
    # # #                         '31-12-17':{'amp_l2c':0.495,'FWHM_l2c':1.3655,'RV_l2c':0.3704}},   #red   preTR
    # # #                         '31-12-17':{'amp_l2c':0.491,'FWHM_l2c':1.3700,'RV_l2c':0.4237}},   #red  postTR
    # #                          'binned':{'HARPS-binned':{'RV_l2c':0.0987,'amp_l2c':0.504,'FWHM_l2c':1.36}, 
    # #                           'HARPS-binned-2018':{'RV_l2c':0.0883,'amp_l2c':0.521,'FWHM_l2c':1.34}},
    # #                 } 
    
    # #        data_dic['DI']['mod_prop']={     #pre-TR
    # #                'HARPS':{'09-01-18':{'amp_l2c':0.557,'FWHM_l2c':1.2956,'RV_l2c':0.2945},    #green
    # #                         '14-01-18':{'amp_l2c':0.563,'FWHM_l2c':1.2911,'RV_l2c':0.2968},    #blue
    # #                         '31-12-17':{'amp_l2c':0.495,'FWHM_l2c':1.3655,'RV_l2c':0.3704}},   #red 
    # #                } 
                             
    # #        data_dic['DI']['mod_prop']={    #post-TR
    # #                'HARPS':{'09-01-18':{'amp_l2c':0.526,'FWHM_l2c':1.3303,'RV_l2c':0.3065},    #green
    # #                         '14-01-18':{'amp_l2c':0.538,'FWHM_l2c':1.3190,'RV_l2c':0.3285},    #blue
    # #                         '31-12-17':{'amp_l2c':0.491,'FWHM_l2c':1.3700,'RV_l2c':0.4237}},   #red 
    # #                }  


    elif gen_dic['star_name']=='TOI858':   
        #Compa DI/intr        
        data_dic['DI']['mod_prop']={'veq':{'vary':True,'CORALIE':{'20191205':{'guess':4.37799708e+00,'bd':[0.,10.]},'20210118':{'guess':4.40730157e+00,'bd':[0.,10.]}}},     #best fit
                                                    'ctrst0':{'vary':False,'CORALIE':{'20191205':{'guess':7.1364575193e-01,'bd':[0.,1.]},'20210118':{'guess':7.4280712811e-01,'bd':[0.,1.]}}},      #best-fit Visits12_indiv_osamp5
                                                    'FWHM0':{'vary':True,'CORALIE':{'20191205':{'guess':4.88663590e+00,'bd':[0.,20.]},'20210118':{'guess':4.93785225e+00,'bd':[0.,20.]}}}}     #best fit    
    
    elif gen_dic['star_name']=='MASCARA1': 
        data_dic['DI']['mod_prop']={}
        
        #Fit to individual masters
        data_dic['DI']['mod_prop'].update({
                                        # 'rv':{'vary':True & False ,'ESPRESSO':{'20190714':{'guess':6.2483596269e+00,'bd':[-10.,10.]},'20190811':{'guess':6.3286491586e+00,'bd':[-10.,10.]}}},
                                        'rv':{'vary':True ,'ESPRESSO':{'20190714':{'guess':6.2483596269e+00,'bd':[-10.,10.]},'20190811':{'guess':6.3286491586e+00,'bd':[-10.,10.]}}},
                                        'veq':{'vary':True  & False,'ESPRESSO':{'20190714':{'guess':1.2639595219e+02,'bd':[100.,150.]},'20190811':{'guess':1.1861806195e+02,'bd':[100.,150.]}}},                             
                                        'ctrst_ord0__IS__VS_':{'vary':True  & False ,'ESPRESSO':{'20190714':{'guess':3.6098785180e-01,'bd':[0.2,0.4]},'20190811':{'guess':3.8765894603e-01,'bd':[0.2,0.4]}}},
                                        'FWHM_ord0__IS__VS_':{'vary':True  & False ,'ESPRESSO':{'20190714':{'guess':2.6176503901e+01,'bd':[20.,30.]},'20190811':{'guess':2.4378173483e+01,'bd':[20.,30.]}}},
                                        'cont':{'vary':True  & False ,'ESPRESSO':{'20190714':{'guess':1.0199059877e+00,'bd':[0.9,1.1]},'20190811':{'guess':1.0199789349e+00,'bd':[0.9,1.1]}}},
                                        'c1_pol':{'vary':True & False  ,'ESPRESSO':{'20190714':{'guess':1.1647989195e-05,'bd':[-1.,1.]},'20190811':{'guess':1.1867256833e-05,'bd':[-1.,1.]}}},
                                        'c2_pol':{'vary':True & False ,'ESPRESSO':{'20190714':{'guess':-7.5636722736e-08,'bd':[-1.,1.]},'20190811':{'guess':-7.7017991429e-08,'bd':[-1.,1.]}}},
                                        'c3_pol':{'vary':True & False  ,'ESPRESSO':{'20190714':{'guess':1.4523037807e-10,'bd':[-1.,1.]},'20190811':{'guess':1.4305736596e-10,'bd':[-1.,1.]}}},
                                        # 'c4_pol':{'vary':True,'ESPRESSO':{'20190714':{'guess':0.,'bd':[-1.,1.]},'20190811':{'guess':0.,'bd':[-1.,1.]}}},
                                        
                                        'cos_istar':{'vary':True & False,'ESPRESSO':{'20190714':{'guess':-5.5915741743e-01,'bd':[-1.,1.]},'20190811':{'guess':-4.7321096800e-01,'bd':[-1.,1.]}}},
                                        'LD_u1':{'vary':True  & False,'ESPRESSO':{'20190714':{'guess':1.9906387034e-02,'bd':[0.,1.]},'20190811':{'guess':1.5448362713e-02,'bd':[0.,1.]}}},                                        
                                        'LD_u2':{'vary':True & False ,'ESPRESSO':{'20190714':{'guess':5.9597745773e-03,'bd':[0.,1.]},'20190811':{'guess':2.1749107654e-04,'bd':[0.,1.]}}}, 
                                        'f_GD':{'vary':True  & False ,'ESPRESSO':{'20190714':{'guess':1.8393143350e-01,'bd':[0.,1.]},'20190811':{'guess':4.8465761632e-01,'bd':[0.,1.]}}}, 
                                        'beta_GD':{'vary':True & False,'ESPRESSO':{'20190714':{'guess':3.0370905333e-01,'bd':[0.,1.]},'20190811':{'guess':1.0271813342e-01,'bd':[0.,1.]}}}, 
                                        'Tpole':{'vary':True & False ,'ESPRESSO':{'20190714':{'guess':7490.,'bd':[7000.,8000.]},'20190811':{'guess':7490.,'bd':[7000.,8000.]}}}, 
                                        
                                        })      


        # #Fit individual exposures with template set to master
        # data_dic['DI']['mod_prop'].update({'rv':{'vary':True ,'ESPRESSO':{'20190714':{'guess':6.2483596269e+00,'bd':[-10.,10.]},'20190811':{'guess':6.3286491586e+00,'bd':[-10.,10.]}}},
        #                                 'veq':{'vary':True  & False,'ESPRESSO':{'20190714':{'guess':1.2639595219e+02,'bd':[100.,150.]},'20190811':{'guess':1.1861806195e+02,'bd':[100.,150.]}}},                             
        #                                 'ctrst_ord0__IS__VS_':{'vary':True  & False ,'ESPRESSO':{'20190714':{'guess':3.6098785180e-01,'bd':[0.2,0.4]},'20190811':{'guess':3.8765894603e-01,'bd':[0.2,0.4]}}},
        #                                 'FWHM_ord0__IS__VS_':{'vary':True  & False ,'ESPRESSO':{'20190714':{'guess':2.6176503901e+01,'bd':[20.,30.]},'20190811':{'guess':2.4378173483e+01,'bd':[20.,30.]}}},
        #                                 'cont':{'vary':True & False  ,'ESPRESSO':{'20190714':{'guess':1.0199059877e+00,'bd':[0.9,1.1]},'20190811':{'guess':1.0199789349e+00,'bd':[0.9,1.1]}}},
        #                                 'c1_pol':{'vary':True & False  ,'ESPRESSO':{'20190714':{'guess':1.1647989195e-05,'bd':[-1.,1.]},'20190811':{'guess':1.1867256833e-05,'bd':[-1.,1.]}}},
        #                                 'c2_pol':{'vary':True & False ,'ESPRESSO':{'20190714':{'guess':-7.5636722736e-08,'bd':[-1.,1.]},'20190811':{'guess':-7.7017991429e-08,'bd':[-1.,1.]}}},
        #                                 'c3_pol':{'vary':True & False  ,'ESPRESSO':{'20190714':{'guess':1.4523037807e-10,'bd':[-1.,1.]},'20190811':{'guess':1.4305736596e-10,'bd':[-1.,1.]}}},
        #                                 # 'c4_pol':{'vary':True,'ESPRESSO':{'20190714':{'guess':0.,'bd':[-1.,1.]},'20190811':{'guess':0.,'bd':[-1.,1.]}}},
                                        
        #                                 'cos_istar':{'vary':True & False,'ESPRESSO':{'20190714':{'guess':-5.5915741743e-01,'bd':[-1.,1.]},'20190811':{'guess':-4.7321096800e-01,'bd':[-1.,1.]}}},
        #                                 'LD_u1':{'vary':True  & False,'ESPRESSO':{'20190714':{'guess':1.9906387034e-02,'bd':[0.,1.]},'20190811':{'guess':1.5448362713e-02,'bd':[0.,1.]}}},                                        
        #                                 'LD_u2':{'vary':True & False ,'ESPRESSO':{'20190714':{'guess':5.9597745773e-03,'bd':[0.,1.]},'20190811':{'guess':2.1749107654e-04,'bd':[0.,1.]}}}, 
        #                                 'f_GD':{'vary':True  & False ,'ESPRESSO':{'20190714':{'guess':1.8393143350e-01,'bd':[0.,1.]},'20190811':{'guess':4.8465761632e-01,'bd':[0.,1.]}}}, 
        #                                 'beta_GD':{'vary':True & False,'ESPRESSO':{'20190714':{'guess':3.0370905333e-01,'bd':[0.,1.]},'20190811':{'guess':1.0271813342e-01,'bd':[0.,1.]}}}, 
        #                                 'Tpole':{'vary':True & False ,'ESPRESSO':{'20190714':{'guess':7490.,'bd':[7000.,8000.]},'20190811':{'guess':7490.,'bd':[7000.,8000.]}}}, 
                                        
        #                                 'offset':{'vary':True  ,'ESPRESSO':{'20190714':{'guess':0.,'bd':[-1.,1.]},'20190811':{'guess':0.,'bd':[-1.,1.]}}},
                                       
        #                                 })      
  
    
    elif gen_dic['star_name']=='V1298tau':
        
        #Fit Gaussien
        # data_dic['DI']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20200128':{'guess':15.,'bd':[0.,30.]},'20201207':{'guess':15.,'bd':[0.,30.]}}},
        #                             'ctrst':{'vary':True  ,'HARPN':{'20200128':{'guess':0.15,'bd':[0.1,0.2]},'20201207':{'guess':0.15,'bd':[0.1,0.2]}}},
        #                             'FWHM':{'vary':True  ,'HARPN':{'20200128':{'guess':30.,'bd':[20.,40.]},'20201207':{'guess':30.,'bd':[20.,40.]}}}}


        data_dic['DI']['mod_prop']={}
        
        # #Fit custom to individual masters
        # data_dic['DI']['mod_prop'].update({
        #                                 'rv':{'vary':True  ,'HARPN':{'20200128':{'guess':14.6,'bd':[-10.,10.]},'20201207':{'guess':14.6,'bd':[-10.,10.]}}},
        #                                 'veq':{'vary':True   ,'HARPN':{'20200128':{'guess':24.,'bd':[100.,150.]},'20201207':{'guess':24.,'bd':[100.,150.]}}},                             
        #                                 'ctrst_ord0__IS__VS_':{'vary':True   ,'HARPN':{'20200128':{'guess':0.5,'bd':[0.2,0.4]},'20201207':{'guess':0.5,'bd':[0.2,0.4]}}},
        #                                 'FWHM_ord0__IS__VS_':{'vary':True  ,'HARPN':{'20200128':{'guess':9.,'bd':[20.,30.]},'20201207':{'guess':9.,'bd':[20.,30.]}}},
        #                                 'cont':{'vary':True   ,'HARPN':{'20200128':{'guess':1.,'bd':[0.9,1.1]},'20201207':{'guess':1.,'bd':[0.9,1.1]}}},
        #                                 })  

        #Fit custom to individual exposures
        #    - the fit is applied to raw CCFs before they are normalized: do not set a guess to the continuum, as the raw continuum is much larger than 1
        data_dic['DI']['mod_prop'].update({
                                        'rv':{'vary':True  ,'HARPN':{'20200128':{'guess':14.6,'bd':[-10.,10.]},'20201207':{'guess':14.6,'bd':[-10.,10.]}}},
                                        'veq':{'vary':True & False    ,'HARPN':{'20200128':{'guess':2.4039504025e+01,'bd':[100.,150.]},'20201207':{'guess':2.4482610718e+01,'bd':[100.,150.]}}},                             
                                        'ctrst_ord0__IS__VS_':{'vary':True & False   ,'HARPN':{'20200128':{'guess':4.9253266352e-01,'bd':[0.2,0.4]},'20201207':{'guess':5.9048604092e-01,'bd':[0.2,0.4]}}},
                                        'FWHM_ord0__IS__VS_':{'vary':True & False  ,'HARPN':{'20200128':{'guess':1.0336828577e+01,'bd':[20.,30.]},'20201207':{'guess':8.5805519323e+00,'bd':[20.,30.]}}},
                                        })  


    elif gen_dic['star_name']=='55Cnc':
        vis_list_dic = { 
            'ESPRESSO': ['20200205','20210121','20210124','binned'],
            'HARPS':  ['20120127','20120213','20120227','20120315','binned']        ,   
            'HARPN':   ['20121225','20131114','20131128','20140101','20140126','20140226','20140329','binned']  ,        
            'SOPHIE': ['20120202','20120203','20120205','20120217','20120219','20120222','20120225','20120302','20120324','20120327','20130303','binned']  }         

        #Fit raw DI
        data_dic['DI']['mod_prop']={'rv':{'vary':True},'ctrst':{'vary':True},'FWHM':{'vary':True}}
        for inst in ['ESPRESSO','HARPS','HARPN','SOPHIE']:
            if inst=='SOPHIE':sysguess = 27.
            else:sysguess = 0.
            for key in data_dic['DI']['mod_prop']:data_dic['DI']['mod_prop'][key][inst]={}
            for vis in vis_list_dic[inst]:
                data_dic['DI']['mod_prop']['rv'][inst][vis] =  {'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}
                data_dic['DI']['mod_prop']['ctrst'][inst][vis] = {'guess':0.5,'bd':[0.1,0.9]}        
                data_dic['DI']['mod_prop']['FWHM'][inst][vis] = {'guess':7.,'bd':[2.,30.]}          
        
        
        # #Fit DI / Intr
        # data_dic['DI']['mod_prop']={'rv':{'vary':False},'ctrst_ord0__IS__VS_':{'vary':True},'FWHM_ord0__IS__VS_':{'vary':True},'cont':{'vary':True},'veq':{'vary':True}}
        # sysguess = 0.
        # rv_subdic = {'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}
        # cont_subdic = {'guess':1.,'bd':[0.9,1.1]}
        # veq_subdic = {'guess':1.,'bd':[0.,2.]}
        # ctrst_loc = { 
        #     'ESPRESSO': {'20200205':None,'20210121':None,'20210124':None,'binned':0.67579855038},
        #     'HARPS':  {'20120127':None,'20120213':None,'20120227':None,'20120315':None,'binned':None},
        #     'HARPN': {'20121225':None,'20131114':None,'20131128':None,'20140101':None,'20140126':None,'20140226':None,'20140329':None,'binned':None},        
        #     'SOPHIE': {'20120202':None,'20120203':None,'20120205':None,'20120217':None,'20120219':None,'20120222':None,'20120225':None,'20120302':None,'20120324':None,'20120327':None,'20130303':None,'binned':None}}    
        # FWHM_loc = { 
        #     'ESPRESSO': {'20200205':None,'20210121':None,'20210124':None,'binned':4.3555040238},
        #     'HARPS':  {'20120127':None,'20120213':None,'20120227':None,'20120315':None,'binned':None},
        #     'HARPN': {'20121225':None,'20131114':None,'20131128':None,'20140101':None,'20140126':None,'20140226':None,'20140329':None,'binned':None},        
        #     'SOPHIE': {'20120202':None,'20120203':None,'20120205':None,'20120217':None,'20120219':None,'20120222':None,'20120225':None,'20120302':None,'20120324':None,'20120327':None,'20130303':None,'binned':None}}    
        # for inst in ['ESPRESSO','HARPS','HARPN','SOPHIE']:
        #     for key in data_dic['DI']['mod_prop']:data_dic['DI']['mod_prop'][key][inst]={}
        #     for vis in vis_list_dic[inst]+['binned']:
        #         data_dic['DI']['mod_prop']['rv'][inst][vis] = rv_subdic
        #         data_dic['DI']['mod_prop']['cont'][inst][vis] = cont_subdic        
        #         data_dic['DI']['mod_prop']['veq'][inst][vis] = veq_subdic                
        #         data_dic['DI']['mod_prop']['ctrst_ord0__IS__VS_'][inst][vis] = {'guess':ctrst_loc[inst][vis],'bd':[0.2,0.4]}        
        #         data_dic['DI']['mod_prop']['FWHM_ord0__IS__VS_'][inst][vis] = {'guess':FWHM_loc[inst][vis],'bd':[0.2,0.4]}                          



    #RM survey
    elif gen_dic['star_name']=='HAT_P3':
        data_dic['DI']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20190415':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]},'20200130':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
                                    'ctrst':{'vary':True  ,'HARPN':{'20190415':{'guess':0.5,'bd':[0.1,0.9]},'20200130':{'guess':0.5,'bd':[0.1,0.9]}}},
                                    'FWHM':{'vary':True  ,'HARPN':{'20190415':{'guess':7.,'bd':[2.,30.]},'20200130':{'guess':7.,'bd':[2.,30.]}}}}
       

        #Fit DI / Intr
        if data_dic['DI']['model']['HARPN']=='custom':
            data_dic['DI']['mod_prop']={'rv':{'vary':True & False ,'HARPN':{'20200130':{'guess':-0.,'bd':[-10.,10.]}}},
                                        'ctrst_ord0__IS__VS_':{'vary':True  & False ,'HARPN':{'20200130':{'guess':5.8433852613e-01,'bd':[0.2,0.4]}}},
                                        'FWHM_ord0__IS__VS_':{'vary':True  & False ,'HARPN':{'20200130':{'guess':6.3866079528e+00,'bd':[0.,15.]}}},
                                        'cont':{'vary':True   ,'HARPN':{'20200130':{'guess':1.022108270,'bd':[0.9,1.1]}}},
                                        'veq':{'vary':True    ,'HARPN':{'20200130':{'guess':0.,'bd':[0.,5.]}}},            
                                            }    

        
    elif gen_dic['star_name']=='Kepler25':
        # data_dic['DI']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20190614':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
        #                             'ctrst':{'vary':True  ,'HARPN':{'20190614':{'guess':0.5,'bd':[0.1,0.9]}}},
        #                             'FWHM':{'vary':True  ,'HARPN':{'20190614':{'guess':7.,'bd':[2.,30.]}}}}
        
        #Fit DI / Intr
        if data_dic['DI']['model']['HARPN']=='custom':
            data_dic['DI']['mod_prop']={'rv':{'vary':True & False ,'HARPN':{'20190614':{'guess':-0.,'bd':[-10.,10.]}}},
                                    'ctrst_ord0__IS__VS_':{'vary':True  & False ,'HARPN':{'20190614':{'guess':5.0922792293e-01,'bd':[0.2,0.4]}}},
                                    'FWHM_ord0__IS__VS_':{'vary':True  & False ,'HARPN':{'20190614':{'guess':9.0524725179e+00,'bd':[0.,15.]}}},
                                    'cont':{'vary':True   ,'HARPN':{'20190614':{'guess':1.0182026547e+00,'bd':[0.9,1.1]}}},
                                    'veq':{'vary':True    ,'HARPN':{'20190614':{'guess':9.8605300904e+00,'bd':[0.,5.]}}},            
                                        }         
        
        
    elif gen_dic['star_name']=='Kepler68':
        data_dic['DI']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20190803':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
                                    'ctrst':{'vary':True  ,'HARPN':{'20190803':{'guess':0.5,'bd':[0.1,0.9]}}},
                                    'FWHM':{'vary':True  ,'HARPN':{'20190803':{'guess':7.,'bd':[2.,30.]}}}}
        
    elif gen_dic['star_name']=='HAT_P33':
        # data_dic['DI']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20191204':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
        #                             'ctrst':{'vary':True  ,'HARPN':{'20191204':{'guess':0.5,'bd':[0.1,0.9]}}},
        #                             'FWHM':{'vary':True  ,'HARPN':{'20191204':{'guess':7.,'bd':[2.,30.]}}}}
        
        #Fit DI / Intr
        if data_dic['DI']['model']['HARPN']=='custom':
            data_dic['DI']['mod_prop']={'rv':{'vary':True & False ,'HARPN':{'20191204':{'guess':0.,'bd':[-10.,10.]}}},
                                    'ctrst_ord0__IS__VS_':{'vary':True  & False ,'HARPN':{'20191204':{'guess':4.5140061939e-01,'bd':[0.2,0.4]}}},
                                    'FWHM_ord0__IS__VS_':{'vary':True  & False ,'HARPN':{'20191204':{'guess':	1.0898145316e+01,'bd':[0.,15.]}}},
                                    'cont':{'vary':True   ,'HARPN':{'20191204':{'guess':1.0182026547e+00,'bd':[0.9,1.1]}}},
                                    'veq':{'vary':True    ,'HARPN':{'20191204':{'guess':15.,'bd':[0.,5.]}}},            
                                        }         
                
        
    elif gen_dic['star_name']=='K2_105':
        # data_dic['DI']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20200118':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
        #                             'ctrst':{'vary':True  ,'HARPN':{'20200118':{'guess':0.5,'bd':[0.1,0.9]}}},
        #                             'FWHM':{'vary':True  ,'HARPN':{'20200118':{'guess':7.,'bd':[2.,30.]}}}}
        
        
        #Fit DI / Intr
        if data_dic['DI']['model']['HARPN']=='custom':
            data_dic['DI']['mod_prop']={'rv':{'vary':True & False ,'HARPN':{'20200118':{'guess':-0.,'bd':[-10.,10.]}}},
                                    'ctrst_ord0__IS__VS_':{'vary':True  & False ,'HARPN':{'20200118':{'guess':3.9243590607e-01,'bd':[0.2,0.4]}}},
                                    'FWHM_ord0__IS__VS_':{'vary':True  & False ,'HARPN':{'20200118':{'guess':7.8943588728e+00,'bd':[0.,15.]}}},
                                    'cont':{'vary':True   ,'HARPN':{'20200118':{'guess':1.0182026547e+00,'bd':[0.9,1.1]}}},
                                    'veq':{'vary':True    ,'HARPN':{'20200118':{'guess':2.1264145833e+00,'bd':[0.,5.]}}},            
                                        }            
        
        
    elif gen_dic['star_name']=='HD89345':
        # data_dic['DI']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20200202':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
        #                             'ctrst':{'vary':True  ,'HARPN':{'20200202':{'guess':0.5,'bd':[0.1,0.9]}}},
        #                             'FWHM':{'vary':True  ,'HARPN':{'20200202':{'guess':7.,'bd':[2.,30.]}}}}
               
        #Fit DI / Intr
        if data_dic['DI']['model']['HARPN']=='custom':
            data_dic['DI']['mod_prop']={'rv':{'vary':True & False ,'HARPN':{'20200202':{'guess':0.,'bd':[-10.,10.]}}},
                                    'ctrst_ord0__IS__VS_':{'vary':True  & False ,'HARPN':{'20200202':{'guess':6.7327730515e-01,'bd':[0.2,0.4]}}},
                                    'FWHM_ord0__IS__VS_':{'vary':True  & False ,'HARPN':{'20200202':{'guess':4.1685966245e+00,'bd':[0.,15.]}}},
                                    'cont':{'vary':True   ,'HARPN':{'20200202':{'guess':1.0182026547e+00,'bd':[0.9,1.1]}}},
                                    'veq':{'vary':True    ,'HARPN':{'20200202':{'guess':5.8461681830e-01,'bd':[0.,5.]}}},            
                                        }            
                
        
    elif gen_dic['star_name']=='Kepler63':
        # data_dic['DI']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20200513':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
        #                             'ctrst':{'vary':True  ,'HARPN':{'20200513':{'guess':0.5,'bd':[0.1,0.9]}}},
        #                             'FWHM':{'vary':True  ,'HARPN':{'20200513':{'guess':7.,'bd':[2.,30.]}}}}
        
        #Fit DI / Intr
        if data_dic['DI']['model']['HARPN']=='custom':
            data_dic['DI']['mod_prop']={'rv':{'vary':True & False ,'HARPN':{'20200513':{'guess':0.,'bd':[-10.,10.]}}},
                                    'ctrst_ord0__IS__VS_':{'vary':True  & False ,'HARPN':{'20200513':{'guess':3.6208137586e-01,'bd':[0.2,0.4]}}},
                                    'FWHM_ord0__IS__VS_':{'vary':True  & False ,'HARPN':{'20200513':{'guess':1.0646477855e+01,'bd':[0.,15.]}}},
                                    'cont':{'vary':True   ,'HARPN':{'20200513':{'guess':1.0182026547e+00,'bd':[0.9,1.1]}}},
                                    'veq':{'vary':True    ,'HARPN':{'20200513':{'guess':7.4734129210e+00,'bd':[0.,5.]}}},            
                                        }           
        
        
        
    elif gen_dic['star_name']=='HAT_P49':
        # data_dic['DI']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20200730':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
        #                             'ctrst':{'vary':True  ,'HARPN':{'20200730':{'guess':0.5,'bd':[0.1,0.9]}}},
        #                             'FWHM':{'vary':True  ,'HARPN':{'20200730':{'guess':7.,'bd':[2.,30.]}}}}
        
        
        #Fit DI / Intr
        if data_dic['DI']['model']['HARPN']=='custom':
            data_dic['DI']['mod_prop']={'rv':{'vary':True & False ,'HARPN':{'20200730':{'guess':-0.,'bd':[-10.,10.]}}},
                                    'ctrst_ord0__IS__VS_':{'vary':True  & False ,'HARPN':{'20200730':{'guess':4.1835024960e-01,'bd':[0.2,0.4]}}},
                                    'FWHM_ord0__IS__VS_':{'vary':True  & False ,'HARPN':{'20200730':{'guess':1.0074056044e+01,'bd':[0.,15.]}}},
                                    'cont':{'vary':True   ,'HARPN':{'20200730':{'guess':1.0182026547e+00,'bd':[0.9,1.1]}}},
                                    'veq':{'vary':True    ,'HARPN':{'20200730':{'guess':1.0682424499e+01,'bd':[0.,5.]}}},            
                                        }           
        
        
    elif gen_dic['star_name']=='WASP47':
        data_dic['DI']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20210730':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
                                    'ctrst':{'vary':True  ,'HARPN':{'20210730':{'guess':0.5,'bd':[0.1,0.9]}}},
                                    'FWHM':{'vary':True  ,'HARPN':{'20210730':{'guess':7.,'bd':[2.,30.]}}}}
        
        
        
    elif gen_dic['star_name']=='WASP107':
        data_dic['DI']['mod_prop']={'rv':{'vary':True ,'HARPS':{'20140406':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]},'20180201':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]},'20180313':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}},
                                                       'CARMENES_VIS':{'20180224':{'guess':sysguess,'bd':[sysguess-3.,sysguess+3.]}}},
                                    'ctrst':{'vary':True  ,'HARPS':{'20140406':{'guess':0.5,'bd':[0.1,0.9]},'20180201':{'guess':0.5,'bd':[0.1,0.9]},'20180313':{'guess':0.5,'bd':[0.1,0.9]}},
                                                                         'CARMENES_VIS':{'20180224':{'guess':0.46,'bd':[0.4,0.55]}}},   
                                    'FWHM':{'vary':True  ,'HARPS':{'20140406':{'guess':7.,'bd':[2.,30.]},'20180201':{'guess':7.,'bd':[2.,30.]},'20180313':{'guess':7.,'bd':[2.,30.]}},
                                                                        'CARMENES_VIS':{'20180224':{'guess':3.4,'bd':[2.,4.]}}}
                                    }
        if 'voigt' in data_dic['DI']['model'].values(): 
            # data_dic['DI']['mod_prop'].update({'FWHM_LOR':{'vary':False ,'CARMENES_VIS':{'20180224':{'guess':6.1,'bd':[6.,7.]}}}})
            # if data_dic['DI']['model']['HARPS']=='voigt':
            #     data_dic['DI']['mod_prop']['FWHM_LOR']['HARPS'] = {'20140406':{'guess':4.7740,'bd':[3.,5.]},'20180201':{'guess':4.7740,'bd':[3.,5.]},'20180313':{'guess':4.6629,'bd':[3.,5.]}}
                              
            #New Voigt model  
            data_dic['DI']['mod_prop'].update({'a_damp':{'vary':True & False ,'CARMENES_VIS':{'20180224':{'guess':1.7748247760e+00,'bd':[1.8,2.]}}}})   #from master-out, MCMC



        #Fit DI / Intr
        if data_dic['DI']['model']['HARPS']=='custom':
            data_dic['DI']['mod_prop']={'rv':{'vary':True & False ,'HARPS':{'20140406':{'guess':0.},'20180201':{'guess':0.},'20180313':{'guess':0.},'binned':{'guess':0.}},
                                                                   'CARMENES_VIS':{'20180224':{'guess':-0.},'binned':{'guess':0.}}},
                                        'ctrst_ord0__IS__VS_':{'vary':True  & False ,'HARPS':{'20140406':{'guess':4.3616144214e-01},'20180201':{'guess':4.3616144214e-01},'20180313':{'guess':4.3616144214e-01},'binned':{'guess':4.3616144214e-01}},
                                                                                     'CARMENES_VIS':{'20180224':{'guess':4.8951151325e-01},'binned':{'guess':4.8951151325e-01}}},
                                        'ctrst_ord1__IS__VS_':{'vary':True  & False ,'HARPS':{'20140406':{'guess':2.8240504842e-01},'20180201':{'guess':2.8240504842e-01},'20180313':{'guess':2.8240504842e-01},'binned':{'guess':2.8240504842e-01}},
                                                                                     'CARMENES_VIS':{'20180224':{'guess':2.8240504842e-01},'binned':{'guess':2.8240504842e-01}}},  
                                        'FWHM_ord0__IS__VS_':{'vary':True  & False ,'HARPS':{'20140406':{'guess':5.2825992455e+00},'20180201':{'guess':5.2825992455e+00},'20180313':{'guess':5.2825992455e+00},'binned':{'guess':5.2825992455e+00}},
                                                                                     'CARMENES_VIS':{'20180224':{'guess':9.6019355537e-01},'binned':{'guess':9.6019355537e-01}}},                                       
                                        'a_damp__IS__VS_':{'vary':True  & False ,'HARPS':{'20140406':{'guess':0.},'20180201':{'guess':0.},'20180313':{'guess':0.},'binned':{'guess':0.}},
                                                                                     'CARMENES_VIS':{'20180224':{'guess':4.},'binned':{'guess':4.}}},  
                                        'cont':{'vary':True  ,'HARPS':{'20140406':{'guess':1.,'bd':[0.9,1.1]},'20180201':{'guess':1.,'bd':[0.9,1.1]},'20180313':{'guess':1.,'bd':[0.9,1.1]},'binned':{'guess':1.,'bd':[0.9,1.1]}},
                                                                                     'CARMENES_VIS':{'20180224':{'guess':1.,'bd':[0.9,1.1]},'binned':{'guess':1.,'bd':[0.9,1.1]}}},        
                                        'veq':{'vary':True   ,'HARPS':{'20140406':{'guess':1.,'bd':[1.,3.]},'20180201':{'guess':1.,'bd':[1.,3.]},'20180313':{'guess':1.,'bd':[1.,3.]},'binned':{'guess':1.,'bd':[1.,3.]}},
                                                                                     'CARMENES_VIS':{'20180224':{'guess':1.,'bd':[1.,3.]},'binned':{'guess':1.,'bd':[1.,3.]}}},            
                                        }                                    
                
                
    elif gen_dic['star_name']=='WASP166':
        data_dic['DI']['mod_prop']={'rv':{'vary':True ,'HARPS':{'20170114':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]},'20170304':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]},'20170315':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
                                    'ctrst':{'vary':True  ,'HARPS':{'20170114':{'guess':0.4,'bd':[0.1,0.9]},'20170304':{'guess':0.5,'bd':[0.1,0.9]},'20170315':{'guess':0.5,'bd':[0.1,0.9]}}},
                                    'FWHM':{'vary':True  ,'HARPS':{'20170114':{'guess':9.,'bd':[2.,30.]},'20170304':{'guess':7.,'bd':[2.,30.]},'20170315':{'guess':7.,'bd':[2.,30.]}}}}
        
        #Fit DI / Intr
        if data_dic['DI']['model']['HARPS']=='custom':
            data_dic['DI']['mod_prop']={'rv':{'vary':True & False ,'HARPS':{'20170114':{'guess':0.,'bd':[-10.,10.]},'20170304':{'guess':0.,'bd':[-10.,10.]},'20170315':{'guess':0.,'bd':[-10.,10.]},'binned':{'guess':0.,'bd':[-10.,10.]}}},
                                        'ctrst_ord0__IS__VS_':{'vary':True  & False ,'HARPS':{}},
                                        'FWHM_ord0__IS__VS_':{'vary':True  & False ,'HARPS':{}},
                                        'cont':{'vary':True   ,'HARPS':{}},
                                        'veq':{'vary':True   ,'HARPS':{}},
                                        }               
            for vis in ['20170114','20170304','20170315','binned']:
                data_dic['DI']['mod_prop']['ctrst_ord0__IS__VS_']['HARPS'][vis] = {'guess':5.6818397517e-01,'bd':[0.2,0.4]}
                data_dic['DI']['mod_prop']['FWHM_ord0__IS__VS_']['HARPS'][vis] = {'guess':6.3816059188e+00,'bd':[0.2,0.4]}
                data_dic['DI']['mod_prop']['cont']['HARPS'][vis] = {'guess':1.,'bd':[0.2,0.4]}
                data_dic['DI']['mod_prop']['veq']['HARPS'][vis] = {'guess':5.5,'bd':[0.2,0.4]}
            
        
        
    elif gen_dic['star_name']=='HAT_P11':
        data_dic['DI']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20150913':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]},'20151101':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}},
                                                       'CARMENES_VIS':{'20170807':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]},'20170812':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
                                    'ctrst':{'vary':True  ,'HARPN':{'20150913':{'guess':0.7,'bd':[0.7,0.73]},'20151101':{'guess':0.7,'bd':[0.7,0.73]}},
                                                                         'CARMENES_VIS':{'20170807':{'guess':0.5,'bd':[0.45,0.55]},'20170812':{'guess':0.5,'bd':[0.45,0.55]}}},   
                                    'FWHM':{'vary':True  ,'HARPN':{'20150913':{'guess':7.,'bd':[4.,5.]},'20151101':{'guess':7.,'bd':[4.,5.]}},
                                                          'CARMENES_VIS':{'20170807':{'guess':7.,'bd':[3.,5.]},'20170812':{'guess':7.,'bd':[3.,5.]}}}
                                    }
        if 'voigt' in data_dic['DI']['model'].values():     #From master-out, MCMC
            data_dic['DI']['mod_prop'].update({'a_damp':{'vary':True & False   ,'CARMENES_VIS':{'20170807':{'guess':1.0725235156e+00,'bd':[1.,1.1]},'20170812':{'guess':1.0585824731e+00,'bd':[1.,1.1]}}}})
            if data_dic['DI']['model']['HARPN']=='voigt':
                data_dic['DI']['mod_prop']['a_damp']['HARPN'] = {'20150913':{'guess':7.3004951501e-01,'bd':[0.6,0.8]},'20151101':{'guess':7.2517700924e-01,'bd':[0.6,0.8]}}
                              
                              


        #Fit DI / Intr
        if data_dic['DI']['model']['HARPN']=='custom':
            data_dic['DI']['mod_prop']={'rv':{'vary':True & False ,'HARPN':{'20150913':{'guess':0.},'20151101':{'guess':0.},'binned':{'guess':0.}},
                                                                   'CARMENES_VIS':{'20170807':{'guess':0.},'20170812':{'guess':0.},'binned':{'guess':0.}}},
                                        'ctrst_ord0__IS__VS_':{'vary':True  & False ,'HARPN':{'20150913':{'guess':6.3159051094e-01},'20151101':{'guess':6.3159051094e-01},'binned':{'guess':6.3159051094e-01}},
                                                                                     'CARMENES_VIS':{'20170807':{'guess':5.8700381496e-01},'20170812':{'guess':5.8700381496e-01},'binned':{'guess':5.8700381496e-01}}},
                                        'FWHM_ord0__IS__VS_':{'vary':True  & False ,'HARPN':{'20150913':{'guess':4.4592294931e+00},'20151101':{'guess':4.4592294931e+00},'binned':{'guess':4.4592294931e+00}},
                                                                                     'CARMENES_VIS':{'20170807':{'guess':3.9553749758e+00},'20170812':{'guess':3.9553749758e+00},'binned':{'guess':3.9553749758e+00}}},                                       
                                        'a_damp__IS__VS_':{'vary':True  & False ,'HARPN':{'20150913':{'guess':0.},'20151101':{'guess':0.},'binned':{'guess':0.}},
                                                                                     'CARMENES_VIS':{'20170807':{'guess':5.6982154625e-01},'20170812':{'guess':5.6982154625e-01},'binned':{'guess':5.6982154625e-01}}},  
                                        'cont':{'vary':True  ,'HARPN':{'20150913':{'guess':1.,'bd':[0.9,1.1]},'20151101':{'guess':1.,'bd':[0.9,1.1]},'binned':{'guess':1.,'bd':[0.9,1.1]}},
                                                                                     'CARMENES_VIS':{'20170807':{'guess':1.,'bd':[0.9,1.1]},'20170812':{'guess':1.,'bd':[0.9,1.1]},'binned':{'guess':1.,'bd':[0.9,1.1]}}},        
                                        'veq':{'vary':True   ,'HARPN':{'20150913':{'guess':1.,'bd':[1.,3.]},'20151101':{'guess':1.,'bd':[1.,3.]},'binned':{'guess':1.,'bd':[1.,3.]}},
                                                                                     'CARMENES_VIS':{'20170807':{'guess':1.,'bd':[1.,3.]},'20170812':{'guess':1.,'bd':[1.,3.]},'binned':{'guess':1.,'bd':[1.,3.]}}},            
                                        }              
        
        
    elif gen_dic['star_name']=='WASP156':
        data_dic['DI']['mod_prop']={'rv':{'vary':True ,'CARMENES_VIS':{'20190928':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]},'20191025':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]},'20191210':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]},'binned':{'guess':0.,'bd':[-10.,10.]}}},
                                    'ctrst':{'vary':True  ,'CARMENES_VIS':{'20190928':{'guess':0.5,'bd':[0.4,0.6]},'20191025':{'guess':0.5,'bd':[0.4,0.6]},'20191210':{'guess':0.5,'bd':[0.4,0.6]},'binned':{'guess':0.5,'bd':[-10.,10.]}}},
                                    'FWHM':{'vary':True  ,'CARMENES_VIS':{'20190928':{'guess':4.,'bd':[3.,5.]},'20191025':{'guess':4.,'bd':[3.,5.]},'20191210':{'guess':4.,'bd':[3.,5.]},'binned':{'guess':0.,'bd':[-10.,10.]}}}}
        
        if 'voigt' in data_dic['DI']['model'].values():  
            # data_dic['DI']['mod_prop'].update({'FWHM_LOR':{'vary':False ,'CARMENES_VIS':{'20190928':{'guess':4.8,'bd':[0.,10.]},'20191025':{'guess':4.8,'bd':[0.,10.]},'20191210':{'guess':5.,'bd':[0.,10.]}}}})

            #New Voigt model  
            data_dic['DI']['mod_prop'].update({'a_damp':{'vary':True & False ,'CARMENES_VIS':{'20190928':{'guess':9.6108080927e-01,'bd':[0.,2.]},'20191025':{'guess':1.0154129374e+00,'bd':[0.,2.]},'20191210':{'guess':1.1884282555e+00,'bd':[0.,2.]}}}})   #from master-out, MCMC

                                       
        
        
    elif gen_dic['star_name']=='HD106315':
        data_dic['DI']['mod_prop']={'rv':{'vary':True ,'HARPS':{'20170309':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]},'20170330':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]},'20180323':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
                                    'ctrst':{'vary':True  ,'HARPS':{'20170309':{'guess':0.5,'bd':[0.1,0.9]},'20170330':{'guess':0.5,'bd':[0.1,0.9]},'20180323':{'guess':0.5,'bd':[0.1,0.9]}}},
                                    'FWHM':{'vary':True  ,'HARPS':{'20170309':{'guess':7.,'bd':[2.,30.]},'20170330':{'guess':7.,'bd':[2.,30.]},'20180323':{'guess':7.,'bd':[2.,30.]}}}}
   
        #Fit DI / Intr
        if data_dic['DI']['model']['HARPS']=='custom':
            data_dic['DI']['mod_prop']={'rv':{'vary':True & False ,'HARPS':{'20170309':{'guess':0.,'bd':[-10.,10.]},'20170330':{'guess':0.,'bd':[-10.,10.]},'20180323':{'guess':0.,'bd':[-10.,10.]},'binned':{'guess':0.,'bd':[-10.,10.]}}},
                                        'ctrst_ord0__IS__VS_':{'vary':True  & False ,'HARPS':{}},
                                        'FWHM_ord0__IS__VS_':{'vary':True  & False ,'HARPS':{}},
                                        'cont':{'vary':True   ,'HARPS':{}},
                                        'veq':{'vary':True   ,'HARPS':{}},
                                        }               
            for vis in ['20170309','20170330','20180323','binned']:
                data_dic['DI']['mod_prop']['ctrst_ord0__IS__VS_']['HARPS'][vis] = {'guess':4.5882739304e-01,'bd':[0.2,0.4]}
                data_dic['DI']['mod_prop']['FWHM_ord0__IS__VS_']['HARPS'][vis] = {'guess':1.2801166639e+01,'bd':[0.2,0.4]}
                data_dic['DI']['mod_prop']['cont']['HARPS'][vis] = {'guess':1.,'bd':[0.9,1.1]}
                data_dic['DI']['mod_prop']['veq']['HARPS'][vis] = {'guess':9.,'bd':[8.,13.]}    
    
            if 'mac_mode' in data_dic['DI']['mod_def']['HARPS']:
                # data_dic['DI']['mod_prop']['eta_R']={'vary':True,'HARPS':{'binned' : {'guess':6.,'bd':[1.,20.]}}}  
                data_dic['DI']['mod_prop']['ksi_R']={'vary':True,'HARPS':{'binned' : {'guess':1.,'bd':[0.,20.]}}}      

    elif gen_dic['star_name']=='HD189733':
        data_dic['DI']['mod_prop']={'rv':{'vary':True ,'ESPRESSO':{'20210810':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]},'20210830':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
                                    'ctrst':{'vary':True  ,'ESPRESSO':{'20210810':{'guess':0.5,'bd':[0.1,0.9]},'20210830':{'guess':0.5,'bd':[0.1,0.9]}}},
                                    'FWHM':{'vary':True  ,'ESPRESSO':{'20210810':{'guess':7.,'bd':[2.,30.]},'20210830':{'guess':7.,'bd':[2.,30.]}}}}         
    #NIRPS
    elif gen_dic['star_name']=='WASP43':
        data_dic['DI']['mod_prop']={'rv':{'vary':True ,'NIRPS_HE':{'20230119':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
                                    'ctrst':{'vary':True  ,'NIRPS_HE':{'20230119':{'guess':0.5,'bd':[0.1,0.9]}}},
                                    'FWHM':{'vary':True  ,'NIRPS_HE':{'20230119':{'guess':7.,'bd':[2.,30.]}}}}           
    elif gen_dic['star_name']=='L98_59':
        data_dic['DI']['mod_prop']={'rv':{'vary':True ,'NIRPS_HE':{'20230411':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
                                    'ctrst':{'vary':True  ,'NIRPS_HE':{'20230411':{'guess':0.5,'bd':[0.1,0.9]}}},
                                    'FWHM':{'vary':True  ,'NIRPS_HE':{'20230411':{'guess':7.,'bd':[2.,30.]}}}}          
        
    elif gen_dic['star_name']=='GJ1214':
        data_dic['DI']['mod_prop']={'rv':{'vary':True ,'NIRPS_HE':{'20230407':{'guess':sysguess,'bd':[sysguess-10.,sysguess+10.]}}},
                                    'ctrst':{'vary':True  ,'NIRPS_HE':{'20230407':{'guess':0.5,'bd':[0.1,0.9]}}},
                                    'FWHM':{'vary':True  ,'NIRPS_HE':{'20230407':{'guess':7.,'bd':[2.,30.]}}}}          

    if (gen_dic['star_name']=='HD209458'):
        if gen_dic['trim_spec'] and gen_dic['fit_DIbinmultivis']:
            data_dic['DI']['mod_prop']={
                'cont':{'vary':True ,'ESPRESSO':{'binned':{'guess':1.0143390541}}},
                'abund_Na':{'vary':True ,'ESPRESSO':{'binned':{'guess':6.0401333437}}},
                'rv':{'vary':True ,'ESPRESSO':{'binned':{'guess':-7.4594687324e-01}}},
                'c1_pol':{'vary':True ,'ESPRESSO':{'binned':{'guess':-1.402581e-03}}},
                'c2_pol':{'vary':True ,'ESPRESSO':{'binned':{'guess':1.5675445018e-04}}},
                'c3_pol':{'vary':True ,'ESPRESSO':{'binned':{'guess':2.0344107970e-05}}},
                'c4_pol':{'vary':True ,'ESPRESSO':{'binned':{'guess':-1.1498681177e-06}}},
                }
        else:
            data_dic['DI']['mod_prop']={
                # 'cont':{'vary':False },
                'cont':{'vary':True ,'ESPRESSO':{'20190720':{'guess':1e7,'bd':[1e6,1e8]},'20190911':{'guess':1e7,'bd':[1e6,1e8]}}},
                # 'offset':{'vary':True ,'ESPRESSO':{'20190720':{'guess':0.,'bd':[-1.,1.]},'20190911':{'guess':0.,'bd':[-1.,1.]}}},
                'rv':{'vary':True ,'ESPRESSO':{'20190720':{'guess':-14.,'bd':[-14.8,-14.7]},'20190911':{'guess':-14.,'bd':[-14.8,-14.7]}}},
                'FWHM':{'vary':True ,'ESPRESSO':{'20190720':{'guess':8.8,'bd':[8.86,8.94]},'20190911':{'guess':8.8,'bd':[8.86,8.94]}}},
                'ctrst':{'vary':True ,'ESPRESSO':{'20190720':{'guess':0.57,'bd':[0.484,0.488]},'20190911':{'guess':0.57,'bd':[0.484,0.488]}}}}

            # #Fit to individual masters            
            # 'rv':{'vary':True  & False,'ESPRESSO':{'20190720':{'guess':0.,'bd':[-10.,10.]},'20190911':{'guess':0.,'bd':[-10.,10.]}}},
            # 'veq':{'vary':True ,'ESPRESSO':{'20190720':{'guess':4.,'bd':[1.,6.]},'20190911':{'guess':4.,'bd':[1.,6.]}}},                             
            # 'ctrst_ord0__IS__VS_':{'vary':True  & False ,'ESPRESSO':{'20190720':{'guess':6.7575217865e-01,'bd':[0.2,0.4]},'20190911':{'guess':6.8420475237e-01,'bd':[0.2,0.4]}}},
            # 'ctrst_ord1__IS__VS_':{'vary':True  & False ,'ESPRESSO':{'20190720':{'guess':-1.6464323607e-01,'bd':[0.2,0.4]},'20190911':{'guess':-1.6464323607e-01,'bd':[0.2,0.4]}}},
            # 'FWHM_ord0__IS__VS_':{'vary':True  & False ,'ESPRESSO':{'20190720':{'guess':5.6752543392e+00,'bd':[20.,30.]},'20190911':{'guess':5.6428910034e+00,'bd':[20.,30.]}}},
            # 'FWHM_ord1__IS__VS_':{'vary':True  & False ,'ESPRESSO':{'20190720':{'guess':3.8883520955e-01,'bd':[20.,30.]},'20190911':{'guess':3.8883520955e-01,'bd':[20.,30.]}}},
            # 'cont':{'vary':False},
            #     } 


             
    if (gen_dic['star_name']=='WASP76'):
        if (data_dic['DI']['model']['ESPRESSO']=='custom'):
            data_dic['DI']['mod_prop']={
            'rv':{'vary':False,'ESPRESSO':{'20180902':{'guess':-1.2006343405,'bd':[-10.,10.]},'20181030':{'guess':-1.2060707814,'bd':[-10.,10.]}}},   #rv defined in input rest frame (F9 mask, from RVres , corr trend, fit chi2   )
            'veq':{'vary':True ,'ESPRESSO':{'20180902':{'guess':1.,'bd':[0.,1.]},'20181030':{'guess':1.,'bd':[0.,1.]}}},                             
            'ctrst_ord0__IS__VS_':{'vary':True  & False ,'ESPRESSO':{'20180902':{'guess':6.5445646911e-01,'bd':[0.2,0.4]},'20181030':{'guess':6.6192786469e-01,'bd':[0.2,0.4]}}},
            'ctrst_ord1__IS__VS_':{'vary':True  & False ,'ESPRESSO':{'20180902':{'guess':-9.0882410087e-02,'bd':[0.2,0.4]},'20181030':{'guess':-9.0882410087e-02,'bd':[0.2,0.4]}}},
            'ctrst_ord2__IS__VS_':{'vary':True  & False ,'ESPRESSO':{'20180902':{'guess':-9.7508515532e-02,'bd':[0.2,0.4]},'20181030':{'guess':-9.7508515532e-02,'bd':[0.2,0.4]}}},
            'FWHM_ord0__IS__VS_':{'vary':True  & False ,'ESPRESSO':{'20180902':{'guess':7.5922402630e+00,'bd':[20.,30.]},'20181030':{'guess':7.6510927098e+00,'bd':[20.,30.]}}},
            'FWHM_ord1__IS__VS_':{'vary':True  & False ,'ESPRESSO':{'20180902':{'guess':-1.0736694900e-01,'bd':[20.,30.]},'20181030':{'guess':-1.0736694900e-01,'bd':[20.,30.]}}},
            'FWHM_ord2__IS__VS_':{'vary':True  & False ,'ESPRESSO':{'20180902':{'guess':4.2046450519e-01,'bd':[20.,30.]},'20181030':{'guess':4.2046450519e-01,'bd':[20.,30.]}}},
            'cont':{'vary':False}}            
        else:
            data_dic['DI']['mod_prop']={
                'cont':{'vary':False },
                'rv':{'vary':True ,'ESPRESSO':{'20180902':{'guess':-1.15,'bd':[-1.27,-1.14]},'20181030':{'guess':-1.15,'bd':[-1.27,-1.14]}}},
                'FWHM':{'vary':True ,'ESPRESSO':{'20180902':{'guess':8.8,'bd':[8.78,8.83]},'20181030':{'guess':8.8,'bd':[8.78,8.83]}}},
                'ctrst':{'vary':True ,'ESPRESSO':{'20180902':{'guess':0.57,'bd':[0.5713,0.5733]},'20181030':{'guess':0.57,'bd':[0.572,0.574]}}},
                }    

    
    
    #Best model table
    if (gen_dic['star_name']=='HD209458'):
        if gen_dic['trim_spec'] and gen_dic['fit_DIbinmultivis']:
            data_dic['DI']['best_mod_tab']={'ESPRESSO':{'dx':0.005,'min_x':5881.,'max_x':5905.}}

    
    #Fitting mode 
    data_dic['DI']['fit_mod']='chi2'   
    # data_dic['DI']['fit_mod']='mcmc' 
    # data_dic['DI']['fit_mod']=''  
    
    #Printing fits results
    data_dic['DI']['verbose']=True  & False

   
    #Priors on variable properties
    data_dic['DI']['line_fit_priors']={}
    # if gen_dic['star_name']=='GJ436':     
    #     data_dic['DI']['line_fit_priors']={'rv':{'mod': 'uf', 'low':0.,'high':20.},
    #                                 'FWHM':{'mod': 'uf', 'low':1.,'high':20.},
    #                                 'amp':{'mod': 'uf', 'low':-1e10,'high':-1e8}}

    ##Compa DI/intr
    #     data_dic['DI']['priors']={'veq':{'mod': 'uf', 'low':0.,'high':1.}}   

    # if gen_dic['star_name']=='TOI858': 
    ##Compa DI/intr
    #     data_dic['DI']['priors']={'veq':{'mod': 'uf', 'low':0.,'high':20.},'FWHM0':{'mod': 'uf', 'low':0.,'high':20.}}


    if gen_dic['star_name']=='MASCARA1': 
        data_dic['DI']['line_fit_priors']={'rv':{'mod': 'uf', 'low':0.,'high':12.},
                                   'veq':{'mod': 'uf', 'low':100.,'high':140.},
                                    'ctrst_ord0__IS__VS_':{'mod': 'uf', 'low':0.,'high':1.},
                                    'FWHM_ord0__IS__VS_':{'mod': 'uf', 'low':10.,'high':40.},                                    
                                    'cos_istar':{'mod': 'uf', 'low':-1.,'high':1.},                                  
                                    'LD_u1':{'mod': 'uf', 'low':0.,'high':1.},
                                    'LD_u2':{'mod': 'uf', 'low':0.,'high':1.}, 
                                    'f_GD':{'mod': 'uf', 'low':0.,'high':1.}, 
                                    'beta_GD':{'mod': 'uf', 'low':0.,'high':1.}, 
                                    'Tpole':{'mod': 'uf', 'low':7000.,'high':8000.}, 
                                    
                                    }
    elif gen_dic['star_name']=='WASP107':
        if  gen_dic['fit_DIbin'] and data_dic['DI']['fit_mod']=='mcmc': 
            data_dic['DI']['line_fit_priors']={'ctrst':{'mod': 'uf', 'low':0.4,'high':0.55},
                                                'FWHM':{'mod': 'uf', 'low':2.,'high':4.}}       
    elif gen_dic['star_name']=='HAT_P11':
        data_dic['DI']['line_fit_priors']={'rv':{'mod': 'uf', 'low':sysguess-20.,'high':sysguess+20.},  
                                           'ctrst':{'mod': 'uf', 'low':0.4,'high':0.8},
                                           'FWHM':{'mod': 'uf', 'low':0.,'high':30.},
                                           'a_damp':{'mod': 'uf', 'low':0.,'high':20.}}     
        
    elif gen_dic['star_name']=='WASP156':
        if  gen_dic['fit_DIbin'] and data_dic['DI']['fit_mod']=='mcmc': 
            data_dic['DI']['line_fit_priors']={'ctrst':{'mod': 'uf', 'low':0.4,'high':0.6},
                                                'FWHM':{'mod': 'uf', 'low':2.,'high':10.}}    

    #RM survey, fit DI / Intr
    elif gen_dic['star_name']=='HAT_P3':
        data_dic['DI']['line_fit_priors']={'veq':{'mod': 'uf', 'low':0.,'high':20.}}   
    elif gen_dic['star_name']=='Kepler25':
        data_dic['DI']['line_fit_priors']={'veq':{'mod': 'uf', 'low':0.,'high':30.}}  
    elif gen_dic['star_name']=='HAT_P33':
        data_dic['DI']['line_fit_priors']={'veq':{'mod': 'uf', 'low':0.,'high':40.}} 
    elif gen_dic['star_name']=='K2_105':
        data_dic['DI']['line_fit_priors']={'veq':{'mod': 'uf', 'low':0.,'high':10.}}     
    elif gen_dic['star_name']=='HD89345':
        data_dic['DI']['line_fit_priors']={'veq':{'mod': 'uf', 'low':0.,'high':20.}}     
    elif gen_dic['star_name']=='WASP107':
        data_dic['DI']['line_fit_priors']={'veq':{'mod': 'uf', 'low':0.,'high':20.}}   
    elif gen_dic['star_name']=='WASP166':
        data_dic['DI']['line_fit_priors']={'veq':{'mod': 'uf', 'low':0.,'high':20.}} 
    elif gen_dic['star_name']=='HD106315':
        # data_dic['DI']['line_fit_priors']={'veq':{'mod': 'uf', 'low':1.,'high':30.},'eta_R':{'mod': 'uf', 'low':0.,'high':30.}}         
        # data_dic['DI']['line_fit_priors']={'veq':{'mod': 'gauss', 'val':9.66,'s_val':0.65},'eta_R':{'mod': 'uf', 'low':0.,'high':30.}}          
        data_dic['DI']['line_fit_priors']={'veq':{'mod': 'uf', 'low':1.,'high':30.},'ksi_R':{'mod': 'uf', 'low':0.,'high':1e5}}           
        
    elif gen_dic['star_name']=='HD189733':
        data_dic['DI']['line_fit_priors']={'veq':{'mod': 'uf', 'low':0.,'high':20.}}         
    elif gen_dic['star_name']=='WASP43':
        data_dic['DI']['line_fit_priors']={'veq':{'mod': 'uf', 'low':0.,'high':20.}}          
    elif gen_dic['star_name']=='L98_59':
        data_dic['DI']['line_fit_priors']={'veq':{'mod': 'uf', 'low':0.,'high':20.}}          
    elif gen_dic['star_name']=='GJ1214':
        data_dic['DI']['line_fit_priors']={'veq':{'mod': 'uf', 'low':0.,'high':20.}}          
    elif gen_dic['star_name']=='WASP76':
        # data_dic['DI']['line_fit_priors']={        
        #     'rv':{'mod': 'uf', 'low':-1.27,'high':-1.14},       
        #     'FWHM':{'mod': 'uf', 'low':8.77,'high':8.83},
        #     'ctrst':{'mod': 'uf', 'low':0.571,'high':0.574}}

        if (data_dic['DI']['model']['ESPRESSO']=='custom'):
            data_dic['DI']['line_fit_priors']={'veq':{'mod': 'uf', 'low':0.,'high':6.}}


    # elif gen_dic['star_name']=='HD209458':
    #     data_dic['DI']['line_fit_priors']={   
    #         'offset':{'mod': 'uf', 'low':-1e10,'high':1e10}, 
    #         'rv':{'mod': 'uf', 'low':-14.9,'high':-14.6},       
    #         'FWHM':{'mod': 'uf', 'low':8.85,'high':8.95},
    #         'ctrst':{'mod': 'uf', 'low':0.483,'high':0.489}}    

    #Derived properties
    data_dic['DI']['deriv_prop']=['amp','area','true_ctrst','true_FWHM','true_amp']   #generic
    data_dic['DI']['deriv_prop']+=['FWHM_LOR','FWHM_voigt']   #voigt
    data_dic['DI']['deriv_prop']+=['cont_amp','RV_lobe','amp_lobe','FWHM_lobe']    #double-gaussian
    data_dic['DI']['deriv_prop']+=['vsini']    #custom
    data_dic['DI']['deriv_prop']=['']


    #Calculating/retrieving
    data_dic['DI']['mcmc_run_mode']='use'
    
    
    #Walkers
    if gen_dic['star_name']=='GJ436':     
        data_dic['DI']['mcmc_set']={         
            'nwalkers':{'ESPRESSO':{'20190228':100,'binned':10},'HARPN':{'20190429':100}},
            'nsteps':{'ESPRESSO':{'20190228':2000,'binned':500},'HARPN':{'20190429':2000}},            
            'nburn':{'ESPRESSO':{'20190228':500,'binned':100},'HARPN':{'20190429':500}},                                
            }
    if gen_dic['star_name']=='TOI858':     
        data_dic['DI']['mcmc_set']={        
            #Compa DI/intr
            'nwalkers':{'CORALIE':{'20191205':20,'20210118':20}},
            'nsteps':{'CORALIE':{'20191205':500,'20210118':500}},            
            'nburn':{'CORALIE':{'20191205':100,'20210118':100}},                                
            }
    elif gen_dic['star_name']=='HAT_P11':
        data_dic['DI']['mcmc_set']={'nwalkers':{'HARPN':{'20150913':20,'20151101':20},'CARMENES_VIS':{'20170807':20,'20170812':20}},
                                    'nsteps':{'HARPN':{'20150913':1000,'20151101':1000},'CARMENES_VIS':{'20170807':1000,'20170812':1000}},
                                    'nburn':{'HARPN':{'20150913':400,'20151101':400},'CARMENES_VIS':{'20170807':400,'20170812':400}}}
    elif gen_dic['star_name']=='WASP156':
        data_dic['DI']['mcmc_set']={'nwalkers':{'CARMENES_VIS':{'20190928':20,'20191025':20,'20191210':20}},
                                    'nsteps':{'CARMENES_VIS':{'20190928':1000,'20191025':1000,'20191210':1000}},
                                    'nburn':{'CARMENES_VIS':{'20190928':400,'20191025':400,'20191210':400}}}
    elif gen_dic['star_name']=='HAT_P3':
        data_dic['DI']['mcmc_set']={'nwalkers':{'HARPN':{'20200130':20}},'nsteps':{'HARPN':{'20200130':500}},'nburn':{'HARPN':{'20200130':150}}}
    elif gen_dic['star_name']=='Kepler25':
        data_dic['DI']['mcmc_set']={'nwalkers':{'HARPN':{'20190614':20}},'nsteps':{'HARPN':{'20190614':500}},'nburn':{'HARPN':{'20190614':150}}}
    elif gen_dic['star_name']=='HAT_P33':
        data_dic['DI']['mcmc_set']={'nwalkers':{'HARPN':{'20191204':20}},'nsteps':{'HARPN':{'20191204':500}},'nburn':{'HARPN':{'20191204':150}}}
    elif gen_dic['star_name']=='K2_105':
        data_dic['DI']['mcmc_set']={'nwalkers':{'HARPN':{'20200118':20}},'nsteps':{'HARPN':{'20200118':500}},'nburn':{'HARPN':{'20200118':150}}}
    elif gen_dic['star_name']=='HD89345':
        data_dic['DI']['mcmc_set']={'nwalkers':{'HARPN':{'20200202':20}},'nsteps':{'HARPN':{'20200202':500}},'nburn':{'HARPN':{'20200202':150}}}
    elif gen_dic['star_name']=='HAT_P49':
        data_dic['DI']['mcmc_set']={'nwalkers':{'HARPN':{'20200730':20}},'nsteps':{'HARPN':{'20200730':500}},'nburn':{'HARPN':{'20200730':150}}}
    elif gen_dic['star_name']=='WASP107':
        data_dic['DI']['mcmc_set']={'nwalkers':{'CARMENES_VIS':{'20180224':20}},
                                    'nsteps':{'CARMENES_VIS':{'20180224':1000}},
                                    'nburn':{'CARMENES_VIS':{'20180224':400}}}
        data_dic['DI']['mcmc_set']={'nwalkers':{'HARPS':{'20140406':20,'20180201':20,'20180313':20},'CARMENES_VIS':{'20180224':20}},
                                    'nsteps':{'HARPS':{'20140406':500,'20180201':500,'20180313':500},'CARMENES_VIS':{'20180224':500}},
                                    'nburn':{'HARPS':{'20140406':150,'20180201':150,'20180313':150},'CARMENES_VIS':{'20180224':150}}}
    elif gen_dic['star_name']=='WASP166':
        data_dic['DI']['mcmc_set']={'nwalkers':{'HARPS':{'binned':20}},
                                    'nsteps':{'HARPS':{'binned':1000}},
                                    'nburn':{'HARPS':{'binned':400}}}        
    elif gen_dic['star_name']=='HD106315':
        data_dic['DI']['mcmc_set']={'nwalkers':{'HARPS':{'binned':20}},
                                    'nsteps':{'HARPS':{'binned':1000}},
                                    'nburn':{'HARPS':{'binned':400}}}   
    elif gen_dic['star_name']=='WASP76':
        # data_dic['DI']['mcmc_set']={'nwalkers':{'ESPRESSO':{'20180902':20,'20181030':20}},'nsteps':{'ESPRESSO':{'20180902':500,'20181030':500}},'nburn':{'ESPRESSO':{'20180902':150,'20181030':150}}}
        data_dic['DI']['mcmc_set']={'nwalkers':{'ESPRESSO':{'20180902':30,'20181030':30}},'nsteps':{'ESPRESSO':{'20180902':1500,'20181030':1500}},'nburn':{'ESPRESSO':{'20180902':400,'20181030':400}}}
        if (data_dic['DI']['model']['ESPRESSO']=='custom'):
            data_dic['DI']['mcmc_set']={'nwalkers':{'ESPRESSO':{'20180902':10,'20181030':10}},'nsteps':{'ESPRESSO':{'20180902':200,'20181030':200}},'nburn':{'ESPRESSO':{'20180902':50,'20181030':50}}}
    elif gen_dic['star_name']=='HD209458':
            data_dic['DI']['mcmc_set']={'nwalkers':{'ESPRESSO':{'20190720':30,'20190911':30}},'nsteps':{'ESPRESSO':{'20190720':1500,'20190911':1500}},'nburn':{'ESPRESSO':{'20190720':400,'20190911':400}}}

    #Walkers exclusion        
    data_dic['DI']['exclu_walk']=True     & False           
     
    
    #Derived errors
    data_dic['DI']['out_err_mode']='HDI'
    # data_dic['DI']['HDI']='2s'   #None   #'3s'  
    
    #Derived lower/upper limits
    # data_dic['DI']['conf_limits']={'veq':{'bound':0.,'type':'upper','level':['1s','3s']}}      




    #Plot settings
    
    #1D PDF from mcmc
    plot_dic['prop_DI_mcmc_PDFs']=''                 
        
    #Individual disk-integrated profiles
    plot_dic['DI_prof']=''  #pdf   

    #Residuals from disk-integrated profiles
    plot_dic['DI_prof_res']=''   #pdf

    #Housekeeping and derived properties 
    plot_dic['prop_raw']=''   #''  
    if gen_dic['star_name'] in ['HD189733','WASP43','L98_59','GJ1214']:plot_dic['prop_raw']=''
    
    
    
  


    # #Stage Théo : new fitting settings for V1298tau artificial visit


    # if gen_dic['mock_data'] : 
    
    
    #     if gen_dic['star_name']=='V1298tau':
    #         data_dic['DI']['model']['HARPN']='gauss'
    #         data_dic['DI']['cont_range']['HARPN']=[[-150.,-70.],[70.,150.]]    
    #         data_dic['DI']['fit_range']['HARPN']= { 'mock_vis' : [[-50, 50]]  }
    

    #         data_dic['DI']['mod_def']['HARPN']={'mode':'ana','coord_line':'mu','func_prof_name':'gauss'} 
    #         data_dic['DI']['mod_prop']={}
    #         data_dic['DI']['mod_prop'].update({
    #                                         'rv':{'vary':True     ,'HARPN':{'mock_vis':{'guess':0,'bd':[-10.,10.]}}},
    #                                         'veq':{'vary':False   ,'HARPN':{'mock_vis':{'guess':23.5,'bd':[20.,30.]}}},                             
    #                                         'ctrst_ord0__IS__VS_':{'vary':True  ,'HARPN':{'mock_vis':{'guess':0.7,'bd':[0.2,1]}}},
    #                                         'FWHM_ord0__IS__VS_' :{'vary':True  ,'HARPN':{'mock_vis':{'guess':4,'bd':[0.,10.]}}},
    #                                         })  
                                        
                                        
                                        

















##################################################################################################
#%%% Module: aligning disk-integrated profiles         
##################################################################################################

#%%%% Activating
gen_dic['align_DI'] = False  


#%%%% Calculating/retrieving 
gen_dic['calc_align_DI'] = False  


#%%%% Systemic velocity 
#    - for each instrument and visit (km/s)
#    - this is the velocity of the center of mass of the system, relative to the Sun (or the input reference frame)
#    - if the value is unknown, or a precise measurement is required for each visit, one can use input CCFs or CCFs from input spectra to determine it
#      first set 'sysvel' to 0 km/s, then run a preliminary analysis to derive its value from the CCF, and update 'sysvel'
#      it can be determined either from the centroid of the master out-of-transit (calculated with gen_dic['DIbin']) or from the mean value of the out-of-transit RV residuals from the keplerian model (via plot_dic['prop_raw'])
#    - beware of using published values, because they can be derived from fits to many datasets, while there
# are still small instrumental offsets in the RV series in a given visit (also, we are using the RV in the fits files which is not corrected for the secular acceleration)
#    - when using spectra the value can be modified without running again the initialization module gen_dic['calc_proc_data'] and spectral correction modules, but any processing modules must still be re-run if the systemic velocity is changed
#      if CCFs are given from input the pipeline must be fully re-run
data_dic['DI']['sysvel']={}

#%%%% Plots: aligned disk-integrated profiles
#    - plotting all aligned DI profiles together in star rest frame
plot_dic['all_DI_data']=''      




if __name__ == '__main__': 

    #Activating
    gen_dic['align_DI'] = True    &  False      
    if ((gen_dic['star_name'] in ['55Cnc']) and (gen_dic['type']['ESPRESSO']=='spec2D')) or \
       ((gen_dic['star_name'] in ['GJ3090']) and ((gen_dic['type']['NIRPS_HA']=='spec2D') or (gen_dic['type']['NIRPS_HE']=='spec2D'))):
        gen_dic['align_DI'] = False   

    
    #Calculating/retrieving 
    gen_dic['calc_align_DI']=True    &  False  
        
    if gen_dic['star_name'] in ['HD189733','WASP43','L98_59','GJ1214']:
        gen_dic['align_DI']=True    
        gen_dic['calc_align_DI']=True 
    if gen_dic['star_name'] in ['HD209458','WASP76']:  
        gen_dic['align_DI']=True  #  & False  
        gen_dic['calc_align_DI']=True  & False                   
    
    #Systemic velocity 
    if gen_dic['star_name']=='55Cnc':
        # data_dic['DI']['sysvel']={'HARPS':27.46902,'HARPN':27.45191} 
        # data_dic['DI']['sysvel']={'ESPRESSO':{'20200205':27.408029-0.009705}}    #measured from input CCFs (default DRS)         
        # data_dic['DI']['sysvel']={'ESPRESSO':{'20200205':27.393682-6.88305e-4}}        #measured from input CCFs (custom Fbal)   
        # data_dic['DI']['sysvel']={'ESPRESSO':{'2020-02-05':27.395458-0.002422149}}        #measured from input CCFs (custom Fbal), with RV correction 
        # data_dic['DI']['sysvel']={'ESPRESSO':{'2020-02-05':27.410727+7.14839e-04}}        #measured from input CCFs, ordres 0:42
        # data_dic['DI']['sysvel']={'ESPRESSO':{'2020-02-05':27.389380-3.66282e-04}}        #measured from input CCFs, ordres 43:84
        # data_dic['DI']['sysvel']={'ESPRESSO':{'2020-02-05':27.390693-3.25311e-3}}        #measured from input CCFs, ordres 85:127
        # data_dic['DI']['sysvel']={'ESPRESSO':{'2020-02-05':27.408590-1.84756e-3}}        #measured from input CCFs, ordres 128:169
        # data_dic['DI']['sysvel']={'ESPRESSO':{'2020-02-05':27.395100-1.15367e-04}}        #measured from input CCFs, ordres 0:90
        # data_dic['DI']['sysvel']={'ESPRESSO':{'2020-02-05':27.390434-2.96477e-3}}        #measured from input CCFs, ordres 91:169
        # data_dic['DI']['sysvel']={'ESPRESSO':{'2020-02-05':27.399778-1.17450e-3}}                     #measured from input CCFs, ordres 108:169
        # data_dic['DI']['sysvel']={'ESPRESSO':{'20200205':27.393682-6.88305e-4,'20210121':27.393682-6.88305e-4,'20210124':27.393682-6.88305e-4}} 

        #New analysis
        # data_dic['DI']['sysvel']['ESPRESSO']={'20200205':6.29317e-2,'20210121':7.57045e-2,'20210124':7.59231e-2}    #default reduction, from RVres                   
        # data_dic['DI']['sysvel']['ESPRESSO']={'20200205':6.30586e-2,'20210121':7.47696e-2,'20210124':7.88608e-2},    #trends corrected, from RVres  
        # data_dic['DI']['sysvel']['ESPRESSO']={'20200205':6.28207e-2,'20210121':7.47696e-2,'20210124':7.88608e-2}    #moon+trends corrected, from RVres      
        # data_dic['DI']['sysvel']['ESPRESSO']={'20200205':6.28871e-2,'20210121':7.57040e-2,'20210124':7.59224e-2},    #From RVres on PC-corrected
        data_dic['DI']['sysvel']['ESPRESSO']={'20200205':6.25447e-2,'20210121':7.57040e-2,'20210124':7.59223592e-2}    #From RVres on PC-corrected, 20200205 mooncorr, FINAL   
    
        # data_dic['DI']['sysvel']['HARPS']={'20120127':-8.06601e-2,'20120213':-8.15590e-2,'20120227':-7.96747e-2,'20120315':-7.86426e-2}    #default reduction, from RVres      
        data_dic['DI']['sysvel']['HARPS']={'20120127':-8.08735e-2,'20120213':-8.15590e-2,'20120227':-7.96747e-2,'20120315':-7.88591e-2}    #trends corrected, from RVres  
        # data_dic['DI']['sysvel']['HARPS']={'20120127':-8.08735e-2 -2.83899e-04,'20120213':-8.15590e-2 -2.83899e-04,'20120227':-7.96747e-2 -2.83899e-04,'20120315':-7.88591e-2 -2.83899e-04},    #trends corrected, from RVres + correction of multivis master 
        # data_dic['DI']['sysvel']['HARPS']={'20120127':-8.065e-2,'20120213':-8.15593e-2,'20120227':-7.96746e-2,'20120315':-7.86417e-2},    #From RVres on PC-corrected  
    

        # data_dic['DI']['sysvel']['HARPN']={'20131114':3.60510e-2,'20131128':3.49689e-2,'20140101':3.54840e-2,'20140126':3.29646e-2,'20140226':3.48829e-2,'20140329':3.27603e-2}     #default reduction, from RVres               
        # data_dic['DI']['sysvel']['HARPN']={'20131114':3.64383e-2,'20131128':3.53444e-2,'20140101':3.56835e-2,'20140126':3.33772e-2,'20140226':3.49617e-2,'20140329':3.27603e-2}          #trends corrected, from RVres
        data_dic['DI']['sysvel']['HARPN']={'20131114':3.60485e-2,'20131128':3.49718e-2,'20140101':3.54820e-2,'20140126':3.29627e-2,'20140226':3.48834e-2,'20140329':3.27612e-2}          #From RVres on PC-corrected  
           
        
        data_dic['DI']['sysvel']['SOPHIE']={'20120202':0.,'20120203':0.,'20120205':0.,'20120217':0.,'20120219':0.,'20120222':0.,'20120225':0.,'20120302':0.,'20120324':0.,'20120327':0.,'20130303':0.}     #default reduction, from RVres  
   
        data_dic['DI']['sysvel']['EXPRES']={'20220131':0.0703424,'20220406':0.0779854 + 1e-5}     #default reduction, from RVres, wo EXPRES RV correction 
        data_dic['DI']['sysvel']['EXPRES']={'20220131':0.0702927,'20220406':0.0781423}          #default reduction, from RVres, with EXPRES RV correction  
        # data_dic['DI']['sysvel']['EXPRES']={'20220131':0.0702969+ 1e-5,'20220406':0.0781993+ 1e-5}     #From RVres on PC-corrected, with EXPRES team RV correction      
        data_dic['DI']['sysvel']['EXPRES']['20220131'] = 0.0702149     #From RVres on trend-corrected, with EXPRES team RV correction      
        
        
     
        

    elif gen_dic['star_name']=='GJ436':
        
        if (gen_dic['type']=='CCF'):
            data_dic['DI']['sysvel']={  
                'ESPRESSO':{'20190228':0.,'20190429':0.},
                'HARPN':{'20160318':0.,'20160411':0.},                    
                'HARPS':{'20070509':0.}}
                
            #ESPRESSO No sky-correction, prop err., chi2, DI master after direct alignment, lobe/core fixed          
            if len(gen_dic['fibB_corr'])==0:            

                #New mask  
                data_dic['DI']['sysvel']={'ESPRESSO':{'20190228':9.7717424537e+00,'20190429':9.7707921816e+00}}           
         
            #ESPRESSO sky-correction, prop err., chi2, DI master after direct alignment, lobe/core fixed          
            else:

                # #Old mask  
                # data_dic['DI']['sysvel']={'ESPRESSO':{'20190228':9.7115655217e+00,'20190429':9.7115768481e+00}}      
                
                #New mask     
                data_dic['DI']['sysvel']={'ESPRESSO':{'20190228':9.7704022220,'20190429':9.7695838786}}            
            
            #HARPS/HARPS-N uncorrected for snr correlations, prop err., chi2, DI master after direct alignment, lobe/core fixed               
            if not gen_dic['detrend_prof']:
                data_dic['DI']['sysvel'].update({'HARPN':{'20160318':9.753977,'20160411':9.752293},'HARPS':{'20070509':9.753737}})

            #HARPS/HARPS-N corrected for snr correlations, prop err., chi2, DI master after direct alignment, lobe/core fixed               
            if gen_dic['detrend_prof']:
                data_dic['DI']['sysvel'].update({'HARPN':{'20160318':9.753977,'20160411':9.752293},'HARPS':{'20070509':9.753063}})

        else:
            data_dic['DI']['sysvel']={'ESPRESSO':{'20190228':9.7704022220,'20190429':9.7695838786}} 

                
        
        
    elif gen_dic['transit_pl']=='WASP121b':
    #    data_dic['DI']['sysvel']={'HARPS':38.3348}    #RVsys de la keplerienne HARPS
        # data_dic['DI']['sysvel']={'HARPS':{'14-01-18':38.36,'09-01-18':38.36,'31-12-17':38.36}}    #RV dont a ete corrige le masque des CCFs, initialement aligne dans le ref barycentrique stellaire
        data_dic['DI']['sysvel']={'ESPRESSO_MR':{'2018-12-01':38.296757},'ESPRESSO':{'2019-01-07':38.376631}}    

    elif gen_dic['transit_pl']=='Kelt9b':
    #    data_dic['DI']['sysvel']={'HARPN':-17.70}    #Hoeijmakers+2019
        data_dic['DI']['sysvel']={'HARPN':-20.565}    #Gaudi+2017      

    elif gen_dic['transit_pl']=='WASP127b':
        data_dic['DI']['sysvel']={'HARPS':{'2017-02-28':-9.193537,'2017-03-20':-9.195641,'2018-02-13':-9.196316,'2018-03-31':-9.201588}}   
    
    
    elif gen_dic['star_name']=='HD209458':
        data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':-14.762032,'20190911':-14.760520}}     #From fit to master out CCF  ; no sky-corr
        data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':-14.762240,'20190911':-14.760934}}     #From fit to master out CCF  ;sky-corr, param. Nuria
        data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':-14.761894,'20190911':-14.760615}}     #From fit to master out CCF  ;sky-corr, param. M. Lendl
        # data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':0.,'20190911':0.,'mock_vis':0.}} 
        
        if gen_dic['DI_CCF']:
            if ('new_F9' in gen_dic['CCF_mask']['ESPRESSO']): 
                if gen_dic['corr_wig']:
                    # data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':-14.761894 - 3.50617e-2 ,'20190911':-14.760615 - 3.42603e-2}}      #From RVres
                    data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':-14.761894 - 3.50617e-2 - 3.413778e-4 ,'20190911':-14.760615 - 3.42603e-2}}      #From RVres, corr_trend
                    
                    # data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':-14.79768500 ,'20190911':-14.79524766}}    #Perf Fbal
                    
                    
                else:
                    data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':-14.7976761805,'20190911':-14.7952565153}}  
            elif ('Relaxed' in gen_dic['CCF_mask']['ESPRESSO']): 
                # data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':-14.761894 - 5.54193e-5,'20190911':-14.760615 + 9.172439e-4}}         #From RVres
                data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':-14.761894 - 5.54193e-5 - 3.64871e-4, '20190911':-14.760615 + 9.172439e-4}}         #From RVres, corr trend
            elif ('Strict' in gen_dic['CCF_mask']['ESPRESSO']): 
                # data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':-14.761894 + 2.3964126e-3,'20190911':-14.760615 +  9.172439e-4}}      #From RVres
                data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':-14.761894 + 2.3964126e-3- 3.40889e-4,'20190911':-14.760615 +  9.172439e-4 +2.592704e-3}}      #From RVres, corr_trend
            else:                
                data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':-14.761894 -3.84184514e-2 ,'20190911':-14.760615 -3.76576703e-2 }}      #For pipeline values, from RVres
        else:
            if gen_dic['corr_wig']:
                data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':-14.761894 -3.54030778e-2,'20190911':-14.760615 - 3.42603e-2}}  #F9 mask taken as final reference

                # data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':-14.79768500 ,'20190911':-14.79524766}}    #Perf Fbal

            else:
                data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':-14.7976761805,'20190911':-14.7952565153}}   #F9 mask taken as final reference
          
                
            # print('ATTENTION RVSYS')
            # data_dic['DI']['sysvel']={'ESPRESSO':{'20190720':-14.761894 - 5.54193e-5 - 3.64871e-4, '20190911':-14.760615 + 9.172439e-4}}   #Mask relaxed fro IntrCCFs
            
    
    elif gen_dic['star_name']=='WASP76':
        if (gen_dic['type']['ESPRESSO']=='CCF'):
            data_dic['DI']['sysvel']={ 
                'ESPRESSO':{'20181030':-1.180597,'20180902':-1.172899}}    #measured from input CCFs  
        elif (gen_dic['type']['ESPRESSO']=='spec2D'):
            # data_dic['DI']['sysvel']={
                # 'ESPRESSO':{'2018-10-31':-1.17911,'2018-09-03':-1.17911}}  
                # 'ESPRESSO':{'2018-10-31':-1.14,'2018-09-03':-1.14}}    #measured from CCFs on input spectra
                # 'ESPRESSO':{'20181030':-1.142014+0.000015,'20180902':-1.135829-0.000006}}    #measured from CCFs on input spectra   
                # 'ESPRESSO':{'20180902':0.,'20181030':0.}} 
            if gen_dic['DI_CCF']:
                if ('new_F9' in gen_dic['CCF_mask']['ESPRESSO']): 
                    if gen_dic['corr_wig']:
                        data_dic['DI']['sysvel']={'ESPRESSO':{'20180902':-1.141999 -0.0574565799 ,'20181030':-1.135835 -0.070238130}}      #From RVres
                        data_dic['DI']['sysvel']={'ESPRESSO':{'20180902':-1.141999 -0.0586287599 ,'20181030':-1.135835 -0.070238130}}      #From RVres , corr trend    
        
                        data_dic['DI']['sysvel']={'ESPRESSO':{'20180902':-1.2006343405 ,'20181030':-1.2060707814}}      #From RVres , corr trend, fit chi2   
                        # data_dic['DI']['sysvel']={'ESPRESSO':{'20180902':-1.2003378388 ,'20181030':-1.2058374293}}      #From master out , corr trend, fit chi2   

                        # data_dic['DI']['sysvel']={'ESPRESSO':{'20180902':-1.20071886 ,'20181030':-1.20593882 }}    #Perf Fbal

                    else:
                        data_dic['DI']['sysvel']={'ESPRESSO':{'20180902':-1.1995008764-1.16900e-3,'20181030':-1.2060368871-0.0215491e-3}}       #From RVres , corr trend, fit chi2  
    
                    
                elif ('Relaxed' in gen_dic['CCF_mask']['ESPRESSO']): 
                    data_dic['DI']['sysvel']={'ESPRESSO':{'20180902':-1.142014 -0.01677219,'20181030':-1.135829 -0.03050356}}      #From RVres                
                    data_dic['DI']['sysvel']={'ESPRESSO':{'20180902':-1.142014 -0.01782769,'20181030':-1.135829 -0.03050356}}      #From RVres, corr trend     
                elif ('Strict' in gen_dic['CCF_mask']['ESPRESSO']): 
                    data_dic['DI']['sysvel']={'ESPRESSO':{'20180902':-1.142014 + 0.00857479,'20181030':-1.135829 -0.0049743}}      #From RVres      
                    data_dic['DI']['sysvel']={'ESPRESSO':{'20180902':-1.142014 + 0.00738557,'20181030':-1.135829 -0.0049743}}        #From RVres, corr trend   
                    
            else:
                if gen_dic['trim_spec']:   #Na D2
                    data_dic['DI']['sysvel']={'ESPRESSO':{'20180902':-3.4729213437e-01,'20181030':-3.4601561907e-01}}      #From master out NaID2 , corr trend, fit chi2  
                    data_dic['DI']['sysvel']={'ESPRESSO':{'20180902':0.,'20181030':0.}}                    
                else:
                    if gen_dic['corr_wig']:
                        data_dic['DI']['sysvel']={'ESPRESSO':{'20180902':-1.2006343405,'20181030':-1.2060707814}}  #F9 mask taken as final reference From RVres , corr trend, fit chi2                  

                        # data_dic['DI']['sysvel']={'ESPRESSO':{'20180902':-1.20071886 ,'20181030':-1.20593882 }}    #Perf Fbal
                    
                    else:
                        data_dic['DI']['sysvel']={'ESPRESSO':{'20180902':-1.1995008764-1.16900e-3,'20181030':-1.2060368871-0.0215491e-3}}    

    if gen_dic['star_name']=='HD3167':
        data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':0.}
        
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.405508-4.37756e-04}}     #dispersed orders + bluest orders removed, Gandolfi
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.402031-1.81033e-04}}     #all orders, Christiansen
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4019}}     #all orders, CHEOPS
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4044}}       #dispersed orders removed, CHEOPS
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4029}}     #dispersed + bluest orders removed, CHEOPS
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.405034-1.87949e-04}}     #dispersed orders removed, Christiansen
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.403048-2.14016e-04}}     #dispersed orders + bluest orders removed, Christiansen
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.401712+9.81311e-04}}     #dispersed orders + bluest orders removed, Christiansen + CHEOPS (final solution)
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.401712+9.81311e-04-160.}}     #continu bleu
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.401712+9.81311e-04-100.}}     #continu bleu close
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.401712+9.81311e-04+100.}}     #continu rouge close
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4186}}       #DRS G9 mask, wgt = ctrst, all orders, CHEOPS
        
        
        
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.400969+7.73834e-04}}     #DRS mask, corrected from micro-tell, Christiansen + CHEOPS 
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.420177+1.06377e-03}}     #G8, not corrected from micro-tell, Christiansen + CHEOPS 
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.423236+1.19247e-03}}     #G8, Corrected from micro-tell, Christiansen + CHEOPS 
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4166}}     #DRS G9 mask, global flux balance, CCFs from E2DS, Christiansen + CHEOPS
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.3994}}     #DRS G9 mask, global flux balance, CCFs from E2DS, wgt = ctrst^2 Christiansen + CHEOPS
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4068}}     #DRS G9 mask, global flux balance, CCFs from E2DS, wgt = ctrst^2 & ctrst < 0.8, Christiansen + CHEOPS
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4157}}     #DRS G9 mask, global flux balance, CCFs from E2DS, wgt = ctrst^2 & ctrst < 0.7, Christiansen + CHEOPS
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4228}}     #DRS G9 mask, global flux balance, CCFs from E2DS, wgt = ctrst^2 & ctrst < 0.6, Christiansen + CHEOPS
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4315}}     #DRS G9 mask, global flux balance, CCFs from E2DS, wgt = ctrst^2 & ctrst < 0.5, Christiansen + CHEOPS
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4322}}     #DRS G9 mask, global flux balance, CCFs from E2DS, wgt = ctrst^2 & ctrst < 0.4, Christiansen + CHEOPS
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4404}}     #DRS G9 mask, global flux balance, CCFs from E2DS, wgt = ctrst^2 & ctrst < 0.3, Christiansen + CHEOPS
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.3905}}     #DRS G9 mask, global flux balance, CCFs from E2DS, wgt = ctrst^2 & ctrst > 0.3, Christiansen + CHEOPS
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.3847}}     #DRS G9 mask, global flux balance, CCFs from E2DS, wgt = ctrst^2 & ctrst > 0.4, Christiansen + CHEOPS
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.3677}}     #DRS G9 mask, global flux balance, CCFs from E2DS, wgt = ctrst^2 & ctrst > 0.5, Christiansen + CHEOPS
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4078}}     
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4007}}     #DRS G9 mask, global flux balance, CCFs from E2DS, wgt = ctrst^2 Christiansen + CHEOPS, orders > 40
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4160}}     #DRS G9 mask, global flux balance, CCFs from E2DS, wgt = ctrst Christiansen + CHEOPS, orders > 40
        
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4166}}     #DRS G9 mask, global + orders flux balance, CCFs from E2DS, Christiansen + CHEOPS         
        

        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.3991}}     #DRS G9 mask, global flux balance, CCFs from E2DS, wgt = ctrst^2 CHEOPS, orders > 20 
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4014}}     #DRS G9 mask, global flux balance, CCFs from E2DS, wgt = ctrst^2 CHEOPS, no trim, dispersed orders + bluest orders removed from CCF
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4165}}       #DRS G9 mask, global flux balance, CCFs from E2DS, wgt = ctrst^2 CHEOPS, no trim, bluest orders removed from CCF

        # #CCF from spectra      
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.5389}}     #all orders, old mask K5 HARPN, cst. error, C (as in old DRS) 
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.5395}}     #all orders, old mask K5 HARPN, prop. error, C (as in old DRS) 
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.3875}}     #all orders, old mask G9, cst. error, C^2  
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.4010}}     #all orders, old mask G9, prop. error, C^2  
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.3754}}     #all orders, new mask G9, cst. error, C^2   
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.3857}}     #all orders, new mask G9, prop. error, C^2
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.5165}}     #all orders, new mask K2, cst. error, C^2
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':19.5250}}     #all orders, new mask K2, prop. error, C^2
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':-0.0512172}}     #all orders, custom kitcat asym 50m/s, cst. error, C^2 
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':-0.0428055}}     #all orders, custom kitcat asym 50m/s, prop. error, C^2 
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':-0.0474594}}     #all orders, custom kitcat sym 50m/s, cst. error, C^2 
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':-0.0406353}}     #all orders, custom kitcat sym 50m/s, prop. error, C^2 
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':-0.0431638}}     #all orders, custom kitcat sym 80m/s, cst. error, C^2 
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':-0.0372752}}     #all orders, custom kitcat sym 80m/s, prop. error, C^2 
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':-0.0441640}}     #all orders, custom kitcat asym 80m/s, prop. error, C^2
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':-0.0528301}}     #all orders, custom kitcat asym 80m/s, cst. error, C^2 
        data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':-0.0369510}      #all orders, custom kitcat sym 80m/s, CCF DRS, fit prop. error 
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':-0.0369510-100.}}     #all orders, custom kitcat sym 80m/s, CCF DRS, fit prop. error - continu bleu close         
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':-0.0369510-160.}}     #all orders, custom kitcat sym 80m/s, CCF DRS, fit prop. error - continu bleu 
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':-0.0369510+100.}}     #all orders, custom kitcat sym 80m/s, CCF DRS, fit prop. error - continu red close
        # data_dic['DI']['sysvel']['ESPRESSO']={'2019-10-09':-0.0369510+160.}}     #all orders, custom kitcat sym 80m/s, CCF DRS, fit prop. error - continu red          
        

        data_dic['DI']['sysvel']['HARPN']={'2016-10-01':0.}   
        
        #data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.526460}}     #all orders, Christiansen + Guilluy
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.526350+0.000894192}}     #bluest orders removed, Christiansen
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.526350+0.004366803}}     #microtell orders + bluest orders removed, Christiansen
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.528560+5.35817e-04}}     #all orders, Gandolfi
        
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.3894}}     #all orders, old mask, prop. error
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.3714}}     #all orders, old mask, constant error
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.3779}}     #all orders, new mask, prop error
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.3672}}     #all orders, new mask, constant error
        
        #CCF from spectra      
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.3659}}     #all orders, new mask G9, cst. error, C^2  
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.3706}}     #no last expo, all orders, old mask G9, cst. error, C^2
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.3889}}     #no last expo, all orders, old mask G9, prop. error, C^2
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.5200}}     #no last expo, all orders, old mask K6, cst. error, C^2
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.5223}}     #no last expo, all orders, old mask K6, prop. error, C^2
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.3710}}     #no last expo, all orders, new mask G8, cst. error, C^2 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.3862}}     #no last expo, all orders, new mask G8, prop. error, C^2 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.3657}}     #no last expo, all orders, new mask G9, cst. error, C^2 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.3770}}     #no last expo, all orders, new mask G9, prop. error, C^2 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.5050}}     #no last expo, all orders, new mask K2, cst. error, C^2 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.5145}}     #no last expo, all orders, new mask K2, prop. error, C^2 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.4842}}     #no last expo, all orders, new mask K6, cst. error, C^2 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.4862}}     #no last expo, all orders, new mask K6, prop. error, C^2 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.5313}}     #no last expo, all orders, old mask K5 HARPN, cst. error, C (as in old DRS) 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.5321}}     #no last expo, all orders, old mask K5 HARPN, prop. error, C (as in old DRS) 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':-0.0494544}}  #no last expo, all orders, cst. error, mask custom kitkat, C^2 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':-0.0399564}}     #no last expo, all orders, prop. error, mask custom kitkat, C^2 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':-0.0441593}}     #no last expo, all orders, cst. error, mask custom kitkat sym. weights, C^2 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':-0.0369400}}     #no last expo, all orders, prop. error, mask custom kitkat sym. weights, C^2 
        
        # CCF from DRS
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.3712}}     #no last expo, all orders, old mask G9, cst. error 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.3896}}     #no last expo, all orders, old mask G9, prop. error 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.3670}}     #no last expo, all orders, new mask G9, cst. error 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':19.3778}}     #no last expo, all orders, new mask G9, prop. error 
        # data_dic['DI']['sysvel']['HARPN']={'2016-10-01':-0.0434370}}     #no last expo, all orders, custom kitcat sym 50m/s, cst. error
        data_dic['DI']['sysvel']['HARPN']={'2016-10-01':-0.0363489}     #no last expo, all orders, custom kitcat sym 50m/s, CCF DRS, fit prop. error
        
    if gen_dic['transit_pl']=='Corot7b':
        data_dic['DI']['sysvel']={'ESPRESSO':{'2019-02-20':31.051948}}  
        # data_dic['DI']['sysvel']={'ESPRESSO':{'2019-02-20':31.051948-14.}}     #fit dans continu bleu
        # data_dic['DI']['sysvel']={'ESPRESSO':{'2019-02-20':31.051948+14.}}     #fit dans continu rouge
    elif gen_dic['transit_pl']=='Nu2Lupi_c':
        data_dic['DI']['sysvel']={'ESPRESSO':{'2020-03-18':-68.805874}}     #master post-tr  
        data_dic['DI']['sysvel']={'ESPRESSO':{'2020-03-18':-68.802495}}     #master post-tr, blue detector   
        data_dic['DI']['sysvel']={'ESPRESSO':{'2020-03-18':-68.816950}}     #master post-tr, red detector            
    elif gen_dic['transit_pl']=='GJ9827d':
        data_dic['DI']['sysvel']={
            'ESPRESSO':{'2019-08-25':31.9606},  
            # 'ESPRESSO':{'2019-08-25':31.9606-14.},    #fit dans continu bleu  
            'HARPS':{'2018-08-18':31.9467,'2018-09-18':31.9460}, 
            }    
    elif gen_dic['transit_pl']=='GJ9827b':
        data_dic['DI']['sysvel']={
            'HARPS':{'2018-08-04':31.9463,'2018-08-15':31.9408,'2018-09-18':31.9470,'2018-09-19':31.9442},}            

    elif 'TOI858b' in gen_dic['transit_pl']:
        data_dic['DI']['sysvel']['CORALIE']={'20191205':0.,'20210118':0.}
        
        
        # data_dic['DI']['sysvel']={'CORALIE':{'20191205':6.43403e+01,'20210118':6.43373e+01}}     #measured on residuals from RV of DRS
        data_dic['DI']['sysvel']={'CORALIE':{'20191205':64.3529,'20210118':64.3579}}     #measured on masters out
    
    elif 'Moon' in gen_dic['transit_pl']:
        data_dic['DI']['sysvel']={'HARPS':{'2019-07-02':0.114279,'2020-12-14':0.112549}}  
    
    elif 'TIC61024636b' in gen_dic['transit_pl']:
        data_dic['DI']['sysvel']={'ESPRESSO':{'mock_vis':0.}}  

    elif gen_dic['star_name']=='HIP41378':
        # data_dic['DI']['sysvel']['HARPN']={'20191218':50.568022,'20220401':50.568022}    #from fit to DI master, F9
        # data_dic['DI']['sysvel']['HARPN']={'20191218':53.197977}    #from fit to DI master, G2, skycorr
        data_dic['DI']['sysvel']['HARPN']={'20191218':53.197728}    #from fit to DI master, G2

    elif gen_dic['star_name']=='HD15337':
        data_dic['DI']['sysvel']['ESPRESSO_MR']={'20191122':7.61583e+01}   
        

    elif gen_dic['star_name']=='Altair':
        data_dic['DI']['sysvel']['ESPRESSO']={'mock_vis':0.}           
        
    elif gen_dic['star_name']=='TOI-3362':
        data_dic['DI']['sysvel']={'HARPS':{'mock_vis':0.},'ESPRESSO':{'mock_vis':0.}}     
    elif 'Nu2Lupi_d' in gen_dic['transit_pl']:data_dic['DI']['sysvel']={'ESPRESSO':{'mock_vis':0.}}  
    elif gen_dic['star_name']=='K2-139':
        data_dic['DI']['sysvel']={'HARPS':{'mock_vis':0.},'ESPRESSO':{'mock_vis':0.}}         
    elif gen_dic['star_name']=='TIC257527578':
        data_dic['DI']['sysvel']={'HARPS':{'mock_vis':0.},'ESPRESSO':{'mock_vis':0.}}         
    elif gen_dic['star_name']=='MASCARA1':
        data_dic['DI']['sysvel']={'ESPRESSO':{'20190714':0.,'20190811':0.}}  
        # data_dic['DI']['sysvel']={'ESPRESSO':{'20190714':6.2483596269e+00,'20190811':6.3286491586e+00}}    #A-mask, colcorr
    elif gen_dic['star_name']=='V1298tau':
        data_dic['DI']['sysvel']={'HARPN':{'20200128':0.,'20201207':0.}}  
        data_dic['DI']['sysvel']={'HARPN':{'20200128':15.004766,'20201207':14.683693, 'mock_vis' : 0}}    #custom fit   # Stage Théo     
        
    #RM survey
    elif gen_dic['star_name']=='HAT_P3':
        data_dic['DI']['sysvel']={'HARPN':{'20190415':0.,'20200130':0.}}  
        if 'new' in gen_dic['data_dir_list']['HARPN']['20200130']:
            data_dic['DI']['sysvel']={'HARPN':{'20190415':-23.379680,'20200130':-23.378509}} 
            data_dic['DI']['sysvel']={'HARPN':{'20200130':-23.382481}}      #skycorr, no corr
            data_dic['DI']['sysvel']={'HARPN':{'20200130':-23.378509}}      #no corr
            data_dic['DI']['sysvel']={'HARPN':{'20200130':-23.378486}}      #corr

            data_dic['DI']['sysvel']={'HARPN':{'20200130':-23.403936}}      #no corr            
            
        if 'KitCat' in gen_dic['data_dir_list']['HARPN']['20200130']:
            data_dic['DI']['sysvel']={'HARPN':{'20190415':-0.037088,'20200130':-0.036960}} 
            data_dic['DI']['sysvel']={'HARPN':{'20200130':-0.037952}}   #skycorr, no corr 
            data_dic['DI']['sysvel']={'HARPN':{'20200130':-0.037914}}   #skycorr, corr 
            # data_dic['DI']['sysvel']={'HARPN':{'20200130':-0.036960}}      #no corr
            data_dic['DI']['sysvel']={'HARPN':{'20200130':-0.037934}}      #corr, from RV res
            data_dic['DI']['sysvel']={'HARPN':{'20200130':-0.062246}}      #no corr, from master
            data_dic['DI']['sysvel']={'HARPN':{'20200130':-0.0618036}}      #no corr, from RV
            
            
    elif gen_dic['star_name']=='Kepler25':
        data_dic['DI']['sysvel']={'HARPN':{'20190614':0.}}
        if 'new' in gen_dic['data_dir_list']['HARPN']['20190614']:
            data_dic['DI']['sysvel']={'HARPN':{'20190614':-8.633258}}             
            # data_dic['DI']['sysvel']={'HARPN':{'20190614':-8.632786}}     #skycorr
        if 'KitCat' in gen_dic['data_dir_list']['HARPN']['20190614']:
            data_dic['DI']['sysvel']={'HARPN':{'20190614':0.005400}}
    elif gen_dic['star_name']=='Kepler68':
        data_dic['DI']['sysvel']={'HARPN':{'20190803':0.}} 
        if 'G1NormSqrt' in gen_dic['data_dir_list']['HARPN']['20190803']:data_dic['DI']['sysvel']={'HARPN':{'20190803':-20.760106}} 
        if 'new' in gen_dic['data_dir_list']['HARPN']['20190803']:
            data_dic['DI']['sysvel']={'HARPN':{'20190803':-20.762823}}     #master, no skycorr, no correction / correction
            # data_dic['DI']['sysvel']={'HARPN':{'20190803':-20.765813}}     #master, skycorr, no correction          
        if 'KitCat' in gen_dic['data_dir_list']['HARPN']['20190803']:
            data_dic['DI']['sysvel']={'HARPN':{'20190803':0.010323}}    #master, no skycorr, no correction / correction
            # data_dic['DI']['sysvel']={'HARPN':{'20190803':0.009315}}     #skycorr
    elif gen_dic['star_name']=='HAT_P33':
        data_dic['DI']['sysvel']={'HARPN':{'20191204':0.}}
        if 'new' in gen_dic['data_dir_list']['HARPN']['20191204']:
            data_dic['DI']['sysvel']={'HARPN':{'20191204':23.084418}}     #master, no correction
            # data_dic['DI']['sysvel']={'HARPN':{'20191204':23.091859}}          #master, skycorr, no correction

        if 'KitCat' in gen_dic['data_dir_list']['HARPN']['20191204']:
            data_dic['DI']['sysvel']={'HARPN':{'20191204':0.156714}}        #master, no correction / correction
        
            
    elif gen_dic['star_name']=='K2_105':
        data_dic['DI']['sysvel']={'HARPN':{'20200118':0.}} 
        if 'new' in gen_dic['data_dir_list']['HARPN']['20200118']:
            data_dic['DI']['sysvel']={'HARPN':{'20200118':-32.390637}} 
            data_dic['DI']['sysvel']={'HARPN':{'20200118':-32.390743}}  #expo 3 removed
        if 'KitCat' in gen_dic['data_dir_list']['HARPN']['20200118']:
            data_dic['DI']['sysvel']={'HARPN':{'20200118':-0.007013}}
            # data_dic['DI']['sysvel']={'HARPN':{'20200118':-0.008606}}  #skycorr
            data_dic['DI']['sysvel']={'HARPN':{'20200118':-0.007167}}   #expo 3 removed
    elif gen_dic['star_name']=='HD89345':
        data_dic['DI']['sysvel']={'HARPN':{'20200202':0.}}  
        if 'new' in gen_dic['data_dir_list']['HARPN']['20200202']:data_dic['DI']['sysvel']={'HARPN':{'20200202':2.223394}} 
        if 'KitCat' in gen_dic['data_dir_list']['HARPN']['20200202']:
            data_dic['DI']['sysvel']={'HARPN':{'20200202':0.004324}}
            # data_dic['DI']['sysvel']={'HARPN':{'20200202':0.004563}}    #skycorr
    elif gen_dic['star_name']=='Kepler63':
        data_dic['DI']['sysvel']={'HARPN':{'20200513':0.}}
        if 'new' in gen_dic['data_dir_list']['HARPN']['20200513']:data_dic['DI']['sysvel']={'HARPN':{'20200513':-22.382079}} 
        if 'KitCat' in gen_dic['data_dir_list']['HARPN']['20200513']:data_dic['DI']['sysvel']={'HARPN':{'20200513':0.003823}}            
    elif gen_dic['star_name']=='HAT_P49':
        data_dic['DI']['sysvel']={'HARPN':{'20200730':0.}}
        if 'new' in gen_dic['data_dir_list']['HARPN']['20200730']:
            # data_dic['DI']['sysvel']={'HARPN':{'20200730':14.208478}}    #PCK solution
            data_dic['DI']['sysvel']={'HARPN':{'20200730':14.217373}}     #PCK circular, w/wo corrections
            # data_dic['DI']['sysvel']={'HARPN':{'20200730':14.213345}}    #Literature 
            # data_dic['DI']['sysvel']={'HARPN':{'20200730':14.205599}}    #K = 260  
        if 'KitCat' in gen_dic['data_dir_list']['HARPN']['20200730']:
            # data_dic['DI']['sysvel']={'HARPN':{'20200730':-0.142928}}     #PCK solution 
            # data_dic['DI']['sysvel']={'HARPN':{'20200730':-0.134203}}     #PCK circular, w/wo corrections
            data_dic['DI']['sysvel']={'HARPN':{'20200730':-0.131916}}     #PCK circular, w/wo corrections, master preTR
            # data_dic['DI']['sysvel']={'HARPN':{'20200730':-0.137236}}     #K = 180
            # data_dic['DI']['sysvel']={'HARPN':{'20200730':-0.138304}}     #K = 190  
            # data_dic['DI']['sysvel']={'HARPN':{'20200730':-0.139372}}     #K = 200   
            # data_dic['DI']['sysvel']={'HARPN':{'20200730':-0.141509}}     #K = 220    
            # data_dic['DI']['sysvel']={'HARPN':{'20200730':-0.143646}}     #K = 240 
            # data_dic['DI']['sysvel']={'HARPN':{'20200730':-0.144714}}     #K = 250  
            # data_dic['DI']['sysvel']={'HARPN':{'20200730':-0.145783}}     #K = 260  
            # data_dic['DI']['sysvel']={'HARPN':{'20200730':-0.146852}}     #K = 270 
            # data_dic['DI']['sysvel']={'HARPN':{'20200730':-0.150059}}     #K = 300    
            # data_dic['DI']['sysvel']={'HARPN':{'20200730':-0.155406}}     #K = 350         

            # data_dic['DI']['sysvel']={'HARPN':{'20200730':-0.137938}}     #K = 260, skycorr  

    elif gen_dic['star_name']=='WASP47':
        data_dic['DI']['sysvel']={'HARPN':{'20210730':0.}}
        if 'new' in gen_dic['data_dir_list']['HARPN']['20210730']:
            data_dic['DI']['sysvel']={'HARPN':{'20210730':-2.7185981603e+01}} #nocorr, noskycorr
            # data_dic['DI']['sysvel']={'HARPN':{'20210730':-2.7185887758e+01}} #nocorr, skycorr
            data_dic['DI']['sysvel']={'HARPN':{'20210730':-27.184120}} #nocorr, noskycorr, cut
        elif 'KitCat' in gen_dic['data_dir_list']['HARPN']['20210730']:
            data_dic['DI']['sysvel']={'HARPN':{'20210730':-0.054936}} #nocorr, noskycorr
            data_dic['DI']['sysvel']={'HARPN':{'20210730':-0.053328}} #nocorr, noskycorr, cut
            data_dic['DI']['sysvel']={'HARPN':{'20210730':-0.0544925}} #nocorr, noskycorr, cut, RVres            
            
        
    elif gen_dic['star_name']=='WASP107':
        data_dic['DI']['sysvel']={'HARPS':{'20140406':0.,'20180201':0.,'20180313':0.},'CARMENES_VIS':{'20180224':0.}}    
        # data_dic['DI']['sysvel']['CARMENES_VIS'] = {'20180224':-0.372230}    #from master
        data_dic['DI']['sysvel']['CARMENES_VIS'] = {'20180224':-0.377247}      #from res RVs    (voigt model)   
        
        data_dic['DI']['sysvel']['CARMENES_VIS'] = {'20180224':-0.373514}    #Voigt model, master, pre-corr    
        data_dic['DI']['sysvel']['CARMENES_VIS'] = {'20180224':-0.373301}    #Voigt model, master, post-corr FWHM vs phase (FINAL)
        data_dic['DI']['sysvel']['CARMENES_VIS'] = {'20180224':-0.374135}    #Voigt model, master, post-corr FWHM vs snr       
        data_dic['DI']['sysvel']['CARMENES_VIS'] = {'20180224':-0.376944,'binned':0.}    #Voigt model, master, post-corr FWHM vs snr 
        
        
        if ('HARPS' in gen_dic['data_dir_list']):              
            if 'K5NormSqrt' in gen_dic['data_dir_list']['HARPS']['20180313']:data_dic['DI']['sysvel']['HARPS']={'20180201':14.193642,'20180313':14.185311} 
            if 'new' in gen_dic['data_dir_list']['HARPS']['20180313']:
                data_dic['DI']['sysvel']['HARPS']={'20140406':14.162570,'20180201':14.168769,'20180313':14.164811} 
                data_dic['DI']['sysvel']['HARPS']={'20140406':14.162570,'20180201':14.170498,'20180313':14.164164}  #skycorr
                data_dic['DI']['sysvel']['HARPS']={'20140406':14.162558 ,'20180201':14.170497,'20180313':14.164172}  #skycorr updated
            if 'KitCat' in gen_dic['data_dir_list']['HARPS']['20180313']:
                data_dic['DI']['sysvel']['HARPS']={'20140406':0.009951,'20180201':0.018249,'20180313':0.015406} 
                data_dic['DI']['sysvel']['HARPS']={'20140406':0.009951,'20180201':0.019663,'20180313':0.016137}    #skycorr
                data_dic['DI']['sysvel']['HARPS']={'20140406':0.010037,'20180201':0.019741,'20180313':0.016254,'binned':0.}    #skycorr,updated
        data_dic['DI']['sysvel']['HARPS']['mock_vis'] = 0.
    elif gen_dic['star_name']=='WASP166':
        data_dic['DI']['sysvel']={'HARPS':{'20170114':0.,'20170304':0.,'20170315':0.}} 
        if 'new' in gen_dic['data_dir_list']['HARPS']['20170114']:
            # data_dic['DI']['sysvel']={'HARPS':{'20170114':23.504693,'20170304':23.506093,'20170315':23.512491}}   #no skycorr
            data_dic['DI']['sysvel']={'HARPS':{'20170114':23.506766,'20170304':23.506093,'20170315':23.512491}}   #skycorr V1         
        if 'KitCat' in gen_dic['data_dir_list']['HARPS']['20170114']:
            # data_dic['DI']['sysvel']={'HARPS':{'20170114':0.023395,'20170304':0.023985,'20170315':0.030346}}   
            # data_dic['DI']['sysvel']={'HARPS':{'20170114':0.025865,'20170304':0.024191,'20170315':0.030504}}   #skycorr all
            data_dic['DI']['sysvel']={'HARPS':{'20170114':0.025865,'20170304':0.023985,'20170315':0.030346,'binned':0.}}   #skycorr V1         
    elif gen_dic['star_name']=='HAT_P11':
        data_dic['DI']['sysvel']={'HARPN':{'20150913':0.,'20151101':0.},'CARMENES_VIS':{'20170807':0.,'20170812':0.}}
        data_dic['DI']['sysvel']['CARMENES_VIS'] = {'20170807':-0.411746,'20170812':-0.411190}        
        data_dic['DI']['sysvel']['CARMENES_VIS'] = {'20170807':-3.15301e-3,'20170812':-3.24049e-3}  #post-corr des RVs, mesure sur RVres (voigt model)
        data_dic['DI']['sysvel']['CARMENES_VIS'] = {'20170807':-3.26007e-3,'20170812':-3.15571e-3}  #post-corr des RVs+FWHM, mesure sur RVres (voigt model)
        data_dic['DI']['sysvel']['CARMENES_VIS'] = {'20170807':-0.413149,'20170812':-0.404817}  #gaussian fit, masters, corrections  
        data_dic['DI']['sysvel']['CARMENES_VIS'] = {'20170807':-0.002891,'20170812':-0.003087}  #Voigt model, master, post-corr

        
        if ('HARPN' in gen_dic['data_dir_list']):           
            if 'new' in gen_dic['data_dir_list']['HARPN']['20150913']:
                data_dic['DI']['sysvel']['HARPN']={'20150913':-63.419840,'20151101':-63.413375}               #from master
                data_dic['DI']['sysvel']['HARPN']={'20150913':-63.4193  ,'20151101':-63.4129}         #from RVres (voigt model)
                
                data_dic['DI']['sysvel']['HARPN']={'20150913': -63.420517,'20151101':-63.414324}   #gaussian fit, masters, no skycorr, no correction / correction C and FWHM
                # data_dic['DI']['sysvel']['HARPN']={'20150913': -63.422370,'20151101':-63.416884}   #gaussian fit, masters, skycorr, no correction
                # data_dic['DI']['sysvel']['HARPN']={'20150913': -63.419859,'20151101':-63.413409}   #voigt fit, masters, no skycorr, no correction                
                # data_dic['DI']['sysvel']['HARPN']={'20150913': 0.,'20151101':0.}   #gaussian fit, masters, no skycorr, correction                    
                
                
            if 'KitCat' in gen_dic['data_dir_list']['HARPN']['20150913']:
                data_dic['DI']['sysvel']['HARPN']={'20150913':0.017541,'20151101':0.024350}     #from master  
                data_dic['DI']['sysvel']['HARPN']={'20150913':1.80416e-2 ,'20151101':2.47988e-2} #from RVres (voigt model)  
                # data_dic['DI']['sysvel']['HARPN':{'20150913':1.83804e-2 ,'20151101':2.51501e-2}} #from RVres (voigt model), skycorr 
                data_dic['DI']['sysflux tovel']['HARPN']={'20150913':0.017552+0.0004981255,'20151101':0.024337+0.0004483264}     #gaussian fit, masters, no skycorr, correction               

                
    elif gen_dic['star_name']=='WASP156'  :
        data_dic['DI']['sysvel']={'CARMENES_VIS':{'20190928':0.,'20191025':0.,'20191210':0.}}
        data_dic['DI']['sysvel']={'CARMENES_VIS':{'20190928':-3.9028354334e-01,'20191025':-4.0217555728e-01,'20191210':-3.7579495512e-01}}      #from master 
        data_dic['DI']['sysvel']={'CARMENES_VIS':{'20190928':-3.85961e-1,'20191025':-4.05507e-1,'20191210':-3.74590e-1}}      #from RVres (voigt model)       
    
        data_dic['DI']['sysvel']={'CARMENES_VIS':{'20190928':-0.3860498816,'20191025':-0.4054742638,'20191210':-0.3744610255}}     #from rv, pre-corr     
        data_dic['DI']['sysvel']={'CARMENES_VIS':{'20190928':-0.3869163731,'20191025':-0.4041571152}}     #from rv, post-corr      
        data_dic['DI']['sysvel']['CARMENES_VIS']['20190928'] =-0.382992   #Without Mout over iexp=12-15

        data_dic['DI']['sysvel']={'CARMENES_VIS':{'20190928':-0.382992,'20191025':-0.4041571152,'20191210':-3.74590e-1}}      #final combination for Na analysis     
    
    elif gen_dic['star_name']=='HD106315':
        data_dic['DI']['sysvel']={'HARPS':{'20170309':0.,'20170330':0.,'20180323':0.}}  
        if 'new' in gen_dic['data_dir_list']['HARPS']['20170309']:
            data_dic['DI']['sysvel']={'HARPS':{'20170309':-3.6443361182e+00,'20170330':-3.6489859282e+00,'20180323':-3.6441380891e+00}} 
            data_dic['DI']['sysvel']={'HARPS':{'20170309':-3.64156,'20170330':-3.64348,'20180323':-3.63943}}    #skycorr
            data_dic['DI']['sysvel']={'HARPS':{'20170309':-3.64156,'20170330':-3.6489859282e+00,'20180323':-3.6441380891e+00}}    #skycorr V1
            data_dic['DI']['sysvel']={'HARPS':{'20170309':-3.641729,'20170330':-3.649771,'20180323':-3.643799}}    #skycorr V1, updated
        if 'KitCat' in gen_dic['data_dir_list']['HARPS']['20170309']:
            data_dic['DI']['sysvel']={'HARPS':{'20170309':-4.5287850620e-02,'20170330':-5.0763544306e-02,'20180323':-4.5212226166e-02}} 
            data_dic['DI']['sysvel']={'HARPS':{'20170309':-4.73371e-2,'20170330':-5.05132e-2,'20180323':-4.52431e-2}}    #skycorr
            data_dic['DI']['sysvel']={'HARPS':{'20170309':-4.73371e-2,'20170330':-5.0763544306e-02,'20180323':-4.5212226166e-02}}    #skycorr V1
            data_dic['DI']['sysvel']={'HARPS':{'20170309':-0.047076,'20170330':-0.051138,'20180323':-0.044951}}    #skycorr V1, updated
        
    elif gen_dic['star_name']=='GJ3090':
        data_dic['DI']['sysvel']={'NIRPS_HE':{'20221201':0.},'NIRPS_HA':{'20221202':0.}}         
    elif gen_dic['star_name']=='HD29291':
        data_dic['DI']['sysvel']={'ESPRESSO':{'20201130':0.}} 
    elif gen_dic['star_name']=='HD189733':
        data_dic['DI']['sysvel']['ESPRESSO']={'20210810':-0.0426,'20210830':-0.0512}  #From Mout
        data_dic['DI']['sysvel']['ESPRESSO']={'20210810':-0.0426300205,'20210830':-0.0510852865}  #From RVres
        data_dic['DI']['sysvel']['ESPRESSO']={'20210810':-0.0426300205-1e-3*8.242082e-02,'20210830':-0.0510852865-1e-3*3.572992e-01}  #From RVres, trend-corr
        
        data_dic['DI']['sysvel']['ESPRESSO']={'20210810':-0.,'20210830':-0.}  #From RVres, skycorr

        
    elif gen_dic['star_name']=='GJ3090':
        data_dic['DI']['sysvel']['NIRPS_HA']={'20221202':0.}
        data_dic['DI']['sysvel']['NIRPS_HE']={'20221201':0.}
    elif gen_dic['star_name']=='WASP43':
        data_dic['DI']['sysvel']['NIRPS_HE']={'20230119':-13.590977*0.}  #From RVres
        data_dic['DI']['sysvel']['NIRPS_HE']={'20230119':(-13.6597+2.13964e-2)*0.}  #From RVres, trend-corr
    elif gen_dic['star_name']=='L98_59':
        data_dic['DI']['sysvel']['NIRPS_HE']={'20230411':-15.843245}  #From RVres
    elif gen_dic['star_name']=='GJ1214':
        data_dic['DI']['sysvel']['NIRPS_HE']={'20230407':8.464783}   #From RVres

    elif user=='vaulato' and gen_dic['star_name']=='WASP189': # vaulato
        data_dic['DI']['sysvel']['NIRPS_HE']={'20230604':-24.452}   # km/h # Anderson et al. 2018 # vaulato


         
        
    #Plots: aligned disk-integrated profiles
    plot_dic['all_DI_data']=''     #pdf    
    

    
    
    
    
    
    
    
    
    
    

    
    
    
    
    # """
    # Routine to correct DI CFFs from spot occultation
    #     - CCF continuum are set to the expected value given planetary and spot contamination
    #     - spot occulted profiles are then added to the DI CCF
    #     - It's assumed that DI CCF are aligned in the star rest frame
    #     - 2 options : 
    #         - Ignore overlapping between spots and planet (in the case lambda_pl is unknown)
    #         - Or, if lambda is known, one can choose to take overlapping between spots and planet into account
    
    
    #     - Correction will be performed within the 'detrend_prof' module, after all other DI corrections. 
    # """
    
    
    
    # gen_dic['correct_spots'] = True &  False
    # gen_dic['calc_correct_spots'] = True  #&  False
    
    corr_spot_dic = {}
    
    # # List all spots that should be included in the correction 
    # if gen_dic['star_name'] == 'V1298tau' : 
    #     corr_spot_dic['spots_prop']={
    #         'HARPN':{
    #             'mock_vis':{
                    
    #                 # Pour le spot 'spot1' : 
    #                 'lat__ISHARPN_VSmock_vis_SPspot1'     : 30,
    #                 'Tcenter__ISHARPN_VSmock_vis_SPspot1' : 2458877.6306 - 12/24,     # 2458877.213933
    #                 'ang__ISHARPN_VSmock_vis_SPspot1'     : 20,
    #                 'flux__ISHARPN_VSmock_vis_SPspot1'    : 0.4,
                    
    #                 # Pour le spot 'spot2' : 
    #                 'lat__ISHARPN_VSmock_vis_SPspot2'     : 40,
    #                 'Tcenter__ISHARPN_VSmock_vis_SPspot2' : 2458877.6306 + 5/24,
    #                 'ang__ISHARPN_VSmock_vis_SPspot2'     : 25,
    #                 'flux__ISHARPN_VSmock_vis_SPspot2'    : 0.4
                    
    #                     },
                        
    #                 'mock_vis2' : {}
                    
    #                     }}
                        


    # # Properties of stellar ray
    # if gen_dic['star_name'] in ['V1298tau'] :
        
    #     corr_spot_dic['intr_prof']={
    #         'mode':'ana',        
    #         'coord_line':'mu',
    #         'func_prof_name': {'HARPN' : 'gauss'},             
    #         'mod_prop':{'ctrst_ord0__ISHARPN_VSmock_vis' : 0.7,
    #                     'FWHM_ord0__ISHARPN_VSmock_vis'  : 4,
    #                     }   ,   
    #         'pol_mode' : 'abs'
    #     }
        

    # # Precision used for calculating planet-occulted profiles (almost useless here, only relevant to the case where planet + spot overlap)
    # corr_spot_dic['precision']='high'

    # # Corresponding dimension (for now, stick to 'mu' and 'r_proj').
    # corr_spot_dic['coord_line']='mu'

    # # Choose between taking overlapping spot/planet into account or not. 
    # corr_spot_dic['overlap'] = True   #& False


    
    
    

    
    










    
##################################################################################################
#%%% Module: broadband flux scaling
#    - define here transit properties     
##################################################################################################


#%%%% Activating
gen_dic['flux_sc'] = False


#%%%% Calculating/retrieving
gen_dic['calc_flux_sc']=True  


#%%%% Scaling disk-integrated profiles
#    - this option should be disabled if absolute photometry is used
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
#    - limb-darkening properties and planet-to-star radius ratios need to be defined in 'system_prop' for each planet whose transit is studied, even if no scaling is applied
#    - the chromatic set of values ('chrom') is used:  
# + to calculate model or simulated light curves (or define the bands of input light curves) used to scale chromatically disk-integrated spectra
# + to calculate chromatic RVs of planet-occulted regions used to align intrinsic spectral profiles (as those RVs are flux-weighted, and thus depend on the chromatic RpRs and LD)    
#      if CCFs are used, if 'chrom' is not provided, or is provided with a single band, 'achrom' will be used automatically
#    - the achromatic set of values ('achrom') must always be defined and is used:
# + to scale input data, unless 'chrom' is used
# + to define the transit contacts
# + to calculate theoretical properties of the 'average' planet-occulted regions throughout the pipeline. The spectral band of the properties should thus match that over which measured planet-occulted properties were derived  
#    - planet-to-star radius ratios can be defined from the transit depth = (Rpl/Rstar)^2 
#    - possible limb-darkening laws (name, number of coefficients) include :
# uniform (uf,0), linear(lin,1), quadratic(quad,2), squareroot(sr,2), logarithmic(log,2), exponential(exp,2), power2 (pw2, 2), nonlinear(nl,4)
#      LD coefficients can be derived at first order with http://astroutils.astronomy.ohio-state.edu/exofast/limbdark.shtml   
#      consider using the Limb Darkening Toolkit (LDTk, Parviainen & Aigrain 2015) tool do determine chromatic LD coefficients 
#    - if the star is oblate (defined through ANTARESS_system_properties), then GD can be accounted for in the same way as LD
#      this is however not necessary: to account for oblateness but not GD, simply comment the 'GD_' fields
#      otherwise GD will be estimated based on a blackbody flux integrated between GD_min and GD_max, at the resolution GD_dw
data_dic['DI']['system_prop']={}
  

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





if __name__ == '__main__':     

    #Activating
    gen_dic['flux_sc']=True   & False
    if ((gen_dic['star_name'] in ['55Cnc','HD29291']) and (gen_dic['type']['ESPRESSO']=='spec2D')) or \
       ((gen_dic['star_name'] in ['GJ3090']) and ((gen_dic['type']['NIRPS_HA']=='spec2D') or (gen_dic['type']['NIRPS_HE']=='spec2D'))):
        gen_dic['flux_sc'] = False   
    
    
    #Calculating/retrieving
    gen_dic['calc_flux_sc']=True  &  False    
    

    if gen_dic['star_name'] in ['HD189733','WASP43','L98_59','GJ1214','WASP107']:
        gen_dic['flux_sc']=True        
        gen_dic['calc_flux_sc']=True

    if gen_dic['star_name'] in ['HD209458','WASP76']:
        gen_dic['flux_sc']=True #  & False     
        gen_dic['calc_flux_sc']=True   & False    
        

    #Scaling disk-integrated profiles
    data_dic['DI']['rescale_DI'] = True    

    #Scaling spectral range
    # if gen_dic['star_name']=='WASP76':data_dic['DI']['scaling_range']=[]
    if gen_dic['star_name']=='MASCARA1':data_dic['DI']['scaling_range']=[ [-130.,130.] ]
    if gen_dic['star_name']=='V1298tau':data_dic['DI']['scaling_range']=[ [-90.,90.] ]
    
    #Out scaling flux
    data_dic['DI']['scaling_val']=1. 

    #Stellar and planet intensity settings
    if gen_dic['star_name']=='HD3167': 
        #Christiansen 2017
        data_dic['DI']['system_prop']={     
            'achrom':{    
              'w':[0.],
              'LD':['linear'],'LD_u1':[0.27],
               'HD3167_b':[0.01744],'HD3167_c':[0.0313]}, 
              # 'HD3167_b':[0.01744+0.00170],'HD3167_c':[0.0313+0.0045]},    #+1s
              # 'HD3167_b':[0.01744-0.00089],'HD3167_c':[0.0313-0.0018]},  #-1s
              # 'LD':['quadratic'],'LD_u1':[0.54],'LD_u2':[0.04],     #Gandolfi+2017
               # 'HD3167_b':[0.01728],'HD3167_c':[0.03006]}, 
              # 'LD':['quadratic'],'LD_u1':[0.489],'LD_u2':[0.226],  #CHEOPS                
               # 'HD3167_b':[0.0153901],   
               #+ 0.0009277 / - 0.0009576  (± 0.0009427)      from 4 transits
               # 'HD3167_b':[0.01791],       # +0.00083/-0.00054            
              }

    elif user=='vaulato' and gen_dic['star_name']=='WASP189':  # vaulato
        data_dic['DI']['system_prop']={
                                   
                        'achrom':{
                            
                            'WASP189b' : [0.0049632],  # (Rp/Rs)^2 = (0.07045)^2
                            'LD':['quadratic'], 
                            'LD_u1' : [0.089533997], # EXOFAST output, input: J band, Teff, log(g) and metallicity [Fe/H] 
                            'LD_u2' : [0.25300000],  # EXOFAST output, input: J band, Teff, log(g) and metallicity [Fe/H]  
                            }}   

    elif gen_dic['star_name']=='55Cnc':
        data_dic['DI']['system_prop']={
                           
                'achrom':{
                    
                    '55Cnc_e' : [0.0182],  #+-2e−4
                    'LD':['quadratic'], 
                    'LD_u1' : [0.544],   #+-0.008
                    'LD_u2' : [0.186],   #+-0.004  

                    }}  
    elif gen_dic['star_name']=='GJ436': 
        data_dic['DI']['system_prop']={          
                'achrom':{
                    # 'GJ436_b' : [np.sqrt(0.006819)],'LD':['nonlinear'],'LD_u1' : [1.47],'LD_u2':[-1.1],'LD_u3':[1.09],'LD_u4':[-0.42]          #Bourrier+2018
                    # 'GJ436_b' : [0.08302],'LD':['power2'],'LD_u1' : [0.9],'LD_u2':[0.5077195017886956]       #ESPRESSO; params from Maxted+2021
                    'GJ436_b' : [0.08315],'LD':['power2'],'LD_u1' : [0.9],'LD_u2':[0.5077195017886956]       #ESPRESSO; moyenne ponderee depuis params from Maxted+2021
                    # 'GJ436_b' : [0.08315-3*0.00011],'LD':['power2'],'LD_u1' : [0.9],'LD_u2':[0.5077195017886956]       #ESPRESSO; moyenne ponderee depuis params from Maxted+2021

                    }}    

    # elif gen_dic['star_name']=='Corot7':
    #     data_dic['DI']['system_prop']={
    #         'Corot7b':{
    #                 'RpRs' : [0.01784],
    #                 'LD':['quadratic'], 
    #                 'LD_u1' : [0.515],
    #                 'LD_u2' : [0.188]
    #             }}  
        
    # elif gen_dic['star_name']=='Nu2Lupi': 
    #     data_dic['DI']['system_prop']={
    #         'Nu2Lupi_c':{
    #                 'RpRs' : [0.02522],
    #                 'LD':['quadratic'], 
    #                 'LD_u1' : [0.275],
    #                 'LD_u2' : [0.285]
    #             }}      
        
    # elif gen_dic['star_name']=='GJ9827': 
    #     data_dic['DI']['system_prop']={       #RICE+2019
    #         'GJ9827d':{
    #                 'RpRs' : [0.03073],   
    #                 'LD':['quadratic'], 
    #                 'LD_u1' : [0.3999],
    #                 'LD_u2' : [0.4372]
    #                 },
    #         'GJ9827b':{
    #                 'RpRs' : [0.02396],
    #                 'LD':['quadratic'], 
    #                 'LD_u1' : [0.3999],
    #                 'LD_u2' : [0.4372]
    #                 }    }  

    # elif gen_dic['star_name']=='TOI178': 
    #     data_dic['DI']['system_prop']={
    #         'TOI178d':{'RpRs' :[0.03719],'LD':['quadratic'],'LD_u1' : [0.],'LD_u2' : [0.]}}  

    # elif gen_dic['star_name']=='TOI858': 
    #     data_dic['DI']['system_prop']={
    #         'TOI858b':{'RpRs' :[0.09906],'LD':['quadratic'],'LD_u1' : [0.367],'LD_u2' : [0.288]}}  

    elif gen_dic['star_name']=='HD209458':  
        data_dic['DI']['system_prop']={
                           
                'achrom':{
                    
                    'HD209458b' : [0.12086],   #Torres+2008  
                    'LD':['quadratic'], 
                    'HD209458c' : [0.12300157*1.5],      #ANTARESS I, mock, multi pl
                    'LD_u1' : [0.38000000],   #Euler fit M. Lendl (fixed), computed with the routine by Nestor Espinoza (on his github site)
                    'LD_u2' : [0.23400000],
                    # 'LD_u1' : [0.40433600],  #Exofast
                    # 'LD_u2' : [0.29138968]
                    
                    # 'GD_wmin':[3800.],'GD_wmax':[7880.],   #ESPRESSO range
                    # 'GD_dw':[10.],

                    },
                
                # 'chrom':{
                #     'w' : [4000.,5000.],
                #     'HD209458b' : [0.12086*5,0.12086*2],  
                #     'LD':['linear','linear'], 
                #     'LD_u1' : [0.3,0.5]
                # },
                
                
                }

    elif gen_dic['star_name']=='Altair':  
        data_dic['DI']['system_prop']={          
                'achrom':{
                    
                    'Altair_b' : [1.*Rjup/(2.029*Rsun)],  
                    'LD':['linear'],'LD_u1' : [0.64],  
                    # 'LD':['uf'],

                    'GD_wmin':[5000.],   
                    'GD_wmax':[5200.],
                    'GD_dw':[10.],
                    
                    }}    

    elif gen_dic['star_name']=='WASP76': 
        wcen_WASP76,c1_WASP76,c2_WASP76,c3_WASP76,c4_WASP76,RpRs_WASP76=np.loadtxt('/Users/bourrier/Travaux/Exoplanet_systems/WASP/WASP76b/ESPRESSO/LowRes_planet_spectrum/Transit_spectrum_Fu2021_ANTARESS.txt').T
        data_dic['DI']['system_prop']={  
                           
            'chrom':{'w':wcen_WASP76,
                      'LD':np.repeat('nonlinear',len(wcen_WASP76)),
                      'LD_u1':c1_WASP76,
                      'LD_u2':c2_WASP76,
                      'LD_u3':c3_WASP76,
                      'LD_u4':c4_WASP76,
                        'WASP76b':RpRs_WASP76},
            'achrom':{'w':[0.],
                   'LD':['quadratic'], 
                   'LD_u1':[0.393], #Ehrenreich+2020
                   'LD_u2':[0.219], #Ehrenreich+2020
                   'WASP76b':[0.10852]} #Ehrenreich+2020
            }

    if gen_dic['star_name']=='TOI858': 
        data_dic['DI']['system_prop']={  
                           
            'achrom':{'w':[0.],'LD':['quadratic'],'LD_u1':[0.44818791],'LD_u2':[0.26759276],
                       'TOI858b':[0.09855]}}
                      # 'TOI858b':[0.09855+3*0.00066]}}   #+3*1sigma
                      # 'TOI858b':[0.09855-3*0.00067]}}   #-3*1sigma 

    if gen_dic['star_name']=='Sun': 
        data_dic['DI']['system_prop']={               
            'achrom':{'Moon':[None],
                   'Mercury':[0.00513],   #2019-11-11
                   'LD':['Sun'],
                      
                    #Reiners paper
                    #'LD_u1' : [0.28392], 'LD_u2' : [1.36896], 'LD_u3' : [1.75998], 'LD_u4' : [2.22154], 'LD_u5' : [1.56076], 'LD_u6' : [0.44630], 

                    #From H. Neckel and D. Labs (1994)
                    'LD_u1' :[0.26073], 'LD_u2' :[1.27428], 'LD_u3' :[1.30352], 'LD_u4' :[1.47085], 'LD_u5' :[0.96618],'LD_u6' :[0.26384]}}         

    elif gen_dic['star_name']=='TIC61024636':        
        data_dic['DI']['system_prop']={  
                           
            'achrom':{    
              'w':[0.],'LD':['quadratic'],'LD_u1':[0.49],'LD_u2':[0.27],'TIC61024636b':[0.1068]}}  

    elif gen_dic['star_name']=='HIP41378': 
        data_dic['DI']['system_prop']={
                'achrom':{'HIP41378d' : [0.0253],'LD':['quadratic'],'LD_u1' : [0.315],'LD_u2':[0.304]}}    

    elif gen_dic['star_name']=='HD15337': 
        data_dic['DI']['system_prop']={
                'achrom':{'HD15337c' : [0.02793],'LD':['linear'],'LD_u1' : [0.5]}}    

    elif gen_dic['star_name']=='TOI-3362':        
        data_dic['DI']['system_prop']={               
            'achrom':{'w':[0.],'LD':['quadratic'],'LD_u1':[0.18],'LD_u2':[0.30],'TOI-3362b':[1.142*Rjup/(1.830*Rsun)]}}  

    elif 'Nu2Lupi_d' in gen_dic['transit_pl']:        
        data_dic['DI']['system_prop']={              
            'achrom':{'w':[0.],'LD':['quadratic'],'LD_u1':[0.275],'LD_u2':[0.285],'Nu2Lupi_d':[0.02219]}}  

    elif gen_dic['star_name']=='K2-139':        
        data_dic['DI']['system_prop']={               
            'achrom':{'w':[0.],'LD':['quadratic'],'LD_u1':[0.37],'LD_u2':[0.48],'K2-139b':[0.0961]}}  

    elif gen_dic['star_name']=='TIC257527578':        
        data_dic['DI']['system_prop']={               
            'achrom':{'w':[0.],'LD':['quadratic'],'LD_u1':[0.31],'LD_u2':[0.26],'TIC257527578b':[0.063]}}  

    elif gen_dic['star_name']=='MASCARA1':  
        data_dic['DI']['system_prop']={                
                'achrom':{
                    
                    'MASCARA1b' : [0.07884],   #Hooton+2021
                    'LD':['quadratic'],'LD_u1' : [0.234],'LD_u2' : [0.405],    #CHEOPS range
                    'GD_wmin':[3800.],'GD_wmax':[7880.],   #ESPRESSO range
                    'GD_dw':[10.],
                    }}    

    elif gen_dic['star_name']=='V1298tau':
        data_dic['DI']['system_prop']={'achrom':{'V1298tau_b' : [0.0700],'LD':['linear'],'LD_u1' : [0.41]}}

    #RM survey
    #    - je prends une cadence temporelle du modele de 0.05 min = 3s, ce qui fait un oversample de 60 meme pour des expos de 180s 
    elif gen_dic['star_name']=='HAT_P3':
        data_dic['DI']['system_prop']={'achrom':{'HAT_P3b' : [0.11091],'LD':['quadratic'],'LD_u1' : [0.63252243],'LD_u2':[0.14115701]}}       
    elif gen_dic['star_name']=='Kepler25':
        data_dic['DI']['system_prop']={'achrom':{'Kepler25c' : [0.03637],'LD':['quadratic'],'LD_u1' : [0.38120373],'LD_u2':[0.30218844]}} 
    elif gen_dic['star_name']=='Kepler68':
        data_dic['DI']['system_prop']={'achrom':{'Kepler68b' : [0.01700],'LD':['quadratic'],'LD_u1' : [0.44908169],'LD_u2':[0.26676522]}}
    elif gen_dic['star_name']=='HAT_P33':
        data_dic['DI']['system_prop']={'achrom':{'HAT_P33b' : [0.10097],'LD':['quadratic'],'LD_u1' : [0.35476341],'LD_u2':[0.31251578]}}
    elif gen_dic['star_name']=='K2_105':
        data_dic['DI']['system_prop']={'achrom':{'K2_105b' : [0.03332],'LD':['quadratic'],'LD_u1' : [0.33841201],'LD_u2':[0.18580958]}}
    elif gen_dic['star_name']=='HD89345':
        data_dic['DI']['system_prop']={'achrom':{'HD89345b' : [0.03696],'LD':['quadratic'],'LD_u1' : [0.53464114],'LD_u2':[0.21451223]}}
    elif gen_dic['star_name']=='Kepler63':
        data_dic['DI']['system_prop']={'achrom':{'Kepler63b' : [0.0622],'LD':['quadratic'],'LD_u1' : [0.52658901],'LD_u2':[0.21657192]}}
    elif gen_dic['star_name']=='HAT_P49':
        data_dic['DI']['system_prop']={'achrom':{'HAT_P49b' : [0.0792],'LD':['quadratic'],'LD_u1' : [0.31234280],'LD_u2':[0.33622956]}}
    elif gen_dic['star_name']=='WASP47':
        # data_dic['DI']['system_prop']={'achrom':{'WASP47d' : [0.02876],'WASP47e' : [0.01458],'LD':['quadratic'],'LD_u1' : [0.53971972],'LD_u2':[0.20886806]}}
        data_dic['DI']['system_prop']={'achrom':{'WASP47d' : [0.02876],'WASP47e' : [0.01458],'LD':['quadratic'],'LD_u1' : [0.53971972],'LD_u2':[0.20886806]}}       
    elif gen_dic['star_name']=='WASP107':
        # data_dic['DI']['system_prop']={'achrom':{'WASP107b' : [0.142988],'LD':['quadratic'],'LD_u1' : [0.77070456],'LD_u2':[0.023162342]}}
        data_dic['DI']['system_prop']={'achrom':{'WASP107b' : [0.14427],'LD':['quadratic'],'LD_u1' : [0.435],'LD_u2':[0.361]}}  #Allart+        
    elif gen_dic['star_name']=='WASP166':
        data_dic['DI']['system_prop']={'achrom':{'WASP166b' : [0.0515],'LD':['quadratic'],'LD_u1' : [0.41621773],'LD_u2':[0.28676671]}}  
    elif gen_dic['star_name']=='HAT_P11':
        data_dic['DI']['system_prop']={'achrom':{'HAT_P11b' : [0.05850],'LD':['quadratic'],'LD_u1' : [0.73904397],'LD_u2':[0.053612443]}}
    elif gen_dic['star_name']=='WASP156'  :
        data_dic['DI']['system_prop']={'achrom':{'WASP156b' : [0.067654],'LD':['quadratic'],'LD_u1' : [0.69280875],'LD_u2':[0.092211683]}}
    elif gen_dic['star_name']=='HD106315':
        data_dic['DI']['system_prop']={'achrom':{'HD106315c' : [0.03481],'LD':['quadratic'],'LD_u1' : [0.33988535],'LD_u2':[0.31582049]}} 
    elif gen_dic['star_name']=='GJ3090':
        data_dic['DI']['system_prop']={'achrom':{'GJ3090b' : [0.0379],'LD':['quadratic'],'LD_u1' : [0.766],'LD_u2':[0.0384]}}
    elif gen_dic['star_name']=='HD29291':
        data_dic['DI']['system_prop']={'achrom':{'HD29291b' : [0.01],'LD':['linear'],'LD_u1' : [0.5],'LD_u2':[0.5]}}


    elif gen_dic['star_name']=='HD189733':    #Analysis Mounzer+2023
        data_dic['DI']['system_prop']={'achrom':{'HD189733b' : [0.15565],'LD':['quadratic'],'LD_u1' : [0.358],'LD_u2' : [0.239]}}
    elif gen_dic['star_name']=='WASP43':    
        data_dic['DI']['system_prop']={'achrom':{'WASP43b' : [0.1615],    #+0.0017-0.0025     Patel & Espinoza 2022
                                                 'LD':['linear'],'LD_u1' : [0.5156]}}    #Weighted mean of the 4 R bands in Table 3 of Esposito+2017 (0.66*(1./0.13)**2.+ 0.511*(1./0.075)**2. + 0.484*(1./0.057)**2. + 0.54*(1./0.11)**2.) /((1./0.13)**2.+ (1./0.075)**2. + (1./0.057)**2. + (1./0.11)**2.)    
    elif gen_dic['star_name']=='L98_59':    
        data_dic['DI']['system_prop']={'achrom':{'L98_59c' : [0.04088],   #+0.00068-0.00056   Demangeon 2021
                                                 'L98_59d' : [0.04480],   #+0.00106-0.00100   Demangeon 2021
                                                 'LD':['quadratic'],'LD_u1' : [-0.0051],'LD_u2' : [0.3056]}}   #Only TESS light curves; from Exofast in H band, Teff = 3415 K, Fe/H = -0.46, logg = 4.86 yields u1 = -0.0051 and u2 = 0.3056
    elif gen_dic['star_name']=='GJ1214':    
        data_dic['DI']['system_prop']={'achrom':{'GJ1214b' : [0.1160],   #+-0.0005 , Berta+2012
                                                 'LD':['squareroot'],'LD_u1' : [-0.3635],'LD_u2' : [ 1.0808]}}   #WFC3 band, Berta+2012, derived from Table 2 (c = -0.3635 ; d = 1.0808)        


    #Transit light curve model
    if gen_dic['star_name']=='HD3167': 
        data_dic['DI']['transit_prop']={
            'HARPN':{'2016-10-01':{'mode':'model','dt':0.1}},
            'ESPRESSO':{'2019-10-09':{'mode':'model','dt':0.1}},
            }
        # data_dic['DI']['transit_prop']={
        #      'nsub_Dstar':2001, 
        #     'HARPN':{'2016-10-01':{'mode':'simu','n_oversamp':5.}},
        #     'ESPRESSO':{'2019-10-09':{'mode':'simu','n_oversamp':5}},
        #     }
        
    elif gen_dic['star_name']=='55Cnc': 
        vis_list_dic = { 
            'ESPRESSO': ['20200205','20210121','20210124'],
            'HARPS':  ['20120127','20120213','20120227','20120315']        ,   
            'HARPN':   ['20121225','20131114','20131128','20140101','20140126','20140226','20140329']  ,        
            'SOPHIE': ['20120202','20120203','20120205','20120217','20120219','20120222','20120225','20120302','20120324','20120327','20130303'],
            'EXPRES': ['20220131','20220406']}        
        data_dic['DI']['transit_prop']={}
        for inst in ['ESPRESSO','HARPS','HARPN','SOPHIE','EXPRES']:        
            data_dic['DI']['transit_prop'][inst]={}
            for vis in vis_list_dic[inst]:data_dic['DI']['transit_prop'][inst][vis]={'mode':'model','dt':0.1}
  
    elif gen_dic['star_name']=='GJ436':   
        data_dic['DI']['transit_prop']={   
        'nsub_Dstar':None, #501, 
            'ESPRESSO':{'20190228':{'mode':'model','dt':0.1},'20190429':{'mode':'model','dt':0.1}},        
            'HARPN':{'20160318':{'mode':'model','dt':0.1},'20160411':{'mode':'model','dt':0.1}},  
            'HARPS':{'20070509':{'mode':'model','dt':0.1}}}          
                
                # 'GJ436_c':{    
                #     'RpRs' : [0.1],  
                #       'LD':['nonlinear'],'LD_u1':[1.47],'LD_u2':[-1.1],'LD_u3':[1.09],'LD_u4':[-0.42]},   
                # }

    # elif gen_dic['star_name']=='WASP121':
    #     data_dic['DI']['transit_prop']={
    #         'WASP121b':{
    #             'RpRs' : [0.12534],
    #             'LD':['quadratic'],
    #             'LD_u1' : [0.37584753],
    #             'LD_u2' : [0.17604912],
                
    #             #On importe le modele ajuste aux donnees Euler pour ne pas dependre des parametres qu'on modifie eventuellement
    #             'path':{'HARPS':{
    #                         '14-01-18':['/Travaux/ANTARESS/WASP121b/Data/System_properties/Photometry/Euler/V3/Euler_transit_mod.txt'],
    #                         '09-01-18':['/Travaux/ANTARESS/WASP121b/Data/System_properties/Photometry/Euler/V3/Euler_transit_mod.txt'],
    #                         '31-12-17':['/Travaux/ANTARESS/WASP121b/Data/System_properties/Photometry/Euler/V3/Euler_transit_mod.txt']}}
    #                               }}


    # elif gen_dic['star_name']=='WASP127':
    #     if gen_dic['type']=='CCF':
    #         data_dic['DI']['transit_prop']={
    #             'WASP127b':{
    #                 'RpRs' : [0.10183319694],
    #                 'LD_u1' : [0.44012],
    #                 'LD_u2' : [0.271056]
    #                 }} 

    elif gen_dic['star_name']=='HD209458':  
        
        data_dic['DI']['transit_prop'].update({
            'ESPRESSO':{'20190720':{'mode':'model','dt':0.1},'20190911':{'mode':'model','dt':0.1}}})
            # 'nsub_Dstar':501,'ESPRESSO':{'20190720':{'mode':'simu','n_oversamp':2.},'20190911':{'mode':'simu','n_oversamp':2.}}})     #ANTARESS I oblate


        if gen_dic['mock_data']:    #ANTARESS I multi pl
            data_dic['DI']['transit_prop']={    
                'nsub_Dstar':501,
                'ESPRESSO':{'mock_vis':{'mode':'simu','n_oversamp':5}},
                }

    elif gen_dic['star_name']=='Altair':     
        data_dic['DI']['transit_prop']={      'nsub_Dstar':501,  'ESPRESSO':{'mock_vis':{'mode':'simu','n_oversamp':10.}}}                 
            
    elif gen_dic['star_name']=='WASP76':  
        data_dic['DI']['transit_prop'].update({
            'ESPRESSO':{'20180902':{'mode':'model','dt':0.1},'20181030':{'mode':'model','dt':0.1}}})
 

    if gen_dic['star_name']=='TOI858': 
        data_dic['DI']['transit_prop']={
            'CORALIE':{'20191205':{'mode':'model','dt':0.1},'20210118':{'mode':'model','dt':0.1}}}


    if gen_dic['star_name']=='Sun': 
        if 'Moon' in gen_dic['transit_pl']:
            data_dic['DI']['transit_prop']={ 
            'nsub_Dstar':2001, 'HARPS':{'2019-07-02':{'mode':'simu','n_oversamp':5.},'2020-12-14':{'mode':'simu','n_oversamp':5.}}}
        if 'Mercury' in gen_dic['transit_pl']:
            data_dic['DI']['transit_prop']={ 
            'nsub_Dstar':2001, 'HARPS':{'2019-11-11':{'mode':'simu','n_oversamp':5.}}}



    elif gen_dic['star_name']=='TIC61024636':        
        data_dic['DI']['transit_prop']={
            'ESPRESSO':{'mock_vis':{'mode':'model','dt':0.01}}}       
        
 
    elif gen_dic['star_name']=='HIP41378':    
        data_dic['DI']['transit_prop']={'HARPN':{'20191218':{'mode':'model','dt':0.1},'20220401':{'mode':'model','dt':0.1}}}        
        

    elif gen_dic['star_name']=='HD15337':    
        data_dic['DI']['transit_prop']={'ESPRESSO_MR':{'20191122':{'mode':'model','dt':0.1}}}         
        

    elif gen_dic['star_name']=='TOI-3362':         
        data_dic['DI']['transit_prop']={'HARPS':{'mock_vis':{'mode':'model','dt':0.005}}}  

    elif 'Nu2Lupi_d' in gen_dic['transit_pl']:        
        data_dic['DI']['transit_prop']={'ESPRESSO':{'mock_vis':{'mode':'model','dt':0.0001}}}  

    elif gen_dic['star_name']=='K2-139':        
        data_dic['DI']['transit_prop']={'HARPS':{'mock_vis':{'mode':'model','dt':0.0001}}} 
      
    elif gen_dic['star_name']=='TIC257527578':        
        data_dic['DI']['transit_prop']={'HARPS':{'mock_vis':{'mode':'model','dt':0.00005}},'ESPRESSO':{'mock_vis':{'mode':'model','dt':0.00005}}}          

    elif gen_dic['star_name']=='MASCARA1':    
        # data_dic['DI']['transit_prop']={'ESPRESSO':{'20190714':{'mode':'simu','n_oversamp':10.},'20190811':{'mode':'simu','n_oversamp':10.}}}  
        data_dic['DI']['transit_prop']={'ESPRESSO':{'20190714':{'mode':'imp','path':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/MASCARA1b/RM/CHEOPS_LC/MASCARA-1b_Hooton21_CHEOPS_transit_ANTARESSformat_V1.dat'},
                                                    '20190811':{'mode':'imp','path':'/Users/bourrier/Travaux/Exoplanet_systems/Divers/MASCARA1b/RM/CHEOPS_LC/MASCARA-1b_Hooton21_CHEOPS_transit_ANTARESSformat_V2.dat'}}}  

    elif gen_dic['star_name']=='V1298tau':
        data_dic['DI']['transit_prop']={'HARPN':{'20200128':{'mode':'model','dt':0.05},'20201207':{'mode':'model','dt':0.05}, 'mock_vis':{'mode':'model','dt':0.05}}}  

          # main_pl_params={
        #     #Bourrier+2018
        #     'RpRs':0.0182,
        #     'LD_mod':'quadratic',
        #     'LD_c_u1':0.544,
        #     'LD_c_u2':0.186,
        #     'inclination':83.59}     
    
    
    #RM survey
    #    - je prends une cadence temporelle du modele de 0.05 min = 3s, ce qui fait un oversample de 60 meme pour des expos de 180s 
    elif gen_dic['star_name']=='HAT_P3':
        data_dic['DI']['transit_prop']={'HARPN':{'20190415':{'mode':'model','dt':0.05},'20200130':{'mode':'model','dt':0.05}}}           
    elif gen_dic['star_name']=='Kepler25':
        data_dic['DI']['transit_prop']={'HARPN':{'20190614':{'mode':'model','dt':0.05}}} 
    elif gen_dic['star_name']=='Kepler68':
        data_dic['DI']['transit_prop']={'HARPN':{'20190803':{'mode':'model','dt':0.05}}}  
    elif gen_dic['star_name']=='HAT_P33':
        data_dic['DI']['transit_prop']={'HARPN':{'20191204':{'mode':'model','dt':0.05}}}  
    elif gen_dic['star_name']=='K2_105':
        data_dic['DI']['transit_prop']={'HARPN':{'20200118':{'mode':'model','dt':0.05}}} 
    elif gen_dic['star_name']=='HD89345':
        data_dic['DI']['transit_prop']={'HARPN':{'20200202':{'mode':'model','dt':0.05}}}  
    elif gen_dic['star_name']=='Kepler63':
        data_dic['DI']['transit_prop']={'HARPN':{'20200513':{'mode':'model','dt':0.05}}}
    elif gen_dic['star_name']=='HAT_P49':
        data_dic['DI']['transit_prop']={'HARPN':{'20200730':{'mode':'model','dt':0.05}}}
    elif gen_dic['star_name']=='WASP47':
        # data_dic['DI']['transit_prop']={'HARPN':{'20210730':{'mode':'simu','n_oversamp':5.}}}  
        data_dic['DI']['transit_prop']={'HARPN':{'20210730':{'mode':'model','dt':0.05}}}        
    elif gen_dic['star_name']=='WASP107':
        data_dic['DI']['transit_prop']={'HARPS':{'20140406':{'mode':'model','dt':0.05},'20180201':{'mode':'model','dt':0.05},'20180313':{'mode':'model','dt':0.05}},'CARMENES_VIS':{'20180224':{'mode':'model','dt':0.05}}} 
        
        # data_dic['DI']['transit_prop']={'nsub_Dstar':101,'HARPS':{'mock_vis':{'mode':'simu','n_oversamp':20.}}} 
        data_dic['DI']['transit_prop']={'nsub_Dstar':None,'HARPS':{'mock_vis':{'mode':'model','dt':30.}}} 
    
    elif gen_dic['star_name']=='WASP166':
        data_dic['DI']['transit_prop']={'HARPS':{'20170114':{'mode':'model','dt':0.05},'20170304':{'mode':'model','dt':0.05},'20170315':{'mode':'model','dt':0.05}}}    
    elif gen_dic['star_name']=='HAT_P11':
        data_dic['DI']['transit_prop']={'HARPN':{'20150913':{'mode':'model','dt':0.05},'20151101':{'mode':'model','dt':0.05}},'CARMENES_VIS':{'20170807':{'mode':'model','dt':0.05},'20170812':{'mode':'model','dt':0.05}}}
    elif gen_dic['star_name']=='WASP156'  :
        data_dic['DI']['transit_prop']={'CARMENES_VIS':{'20190928':{'mode':'model','dt':0.05},'20191025':{'mode':'model','dt':0.05},'20191210':{'mode':'model','dt':0.05}}}
    elif gen_dic['star_name']=='HD106315':
        data_dic['DI']['transit_prop']={'HARPS':{'20170309':{'mode':'model','dt':0.05},'20170330':{'mode':'model','dt':0.05},'20180323':{'mode':'model','dt':0.05}}}  
    elif gen_dic['star_name']=='GJ3090':
        data_dic['DI']['transit_prop']={'NIRPS_HE':{'20121201':{'mode':'model','dt':0.05}},'NIRPS_HA':{'20121202':{'mode':'model','dt':0.05}}}  
    elif gen_dic['star_name']=='HD29291':
        data_dic['DI']['transit_prop']={'ESPRESSO':{'20201130':{'mode':'model','dt':0.05}}}  

    if gen_dic['star_name']=='HD189733':data_dic['DI']['transit_prop'].update({'ESPRESSO':{'20210810':{'mode':'model','dt':0.05},'20210830':{'mode':'model','dt':0.05}}})  
    if gen_dic['star_name']=='WASP43':data_dic['DI']['transit_prop'].update({'NIRPS_HE':{'20230119':{'mode':'model','dt':0.05}}})  
    if gen_dic['star_name']=='L98_59':
        if len(list(gen_dic['transit_pl'].keys()))==1:
            data_dic['DI']['transit_prop'].update({'NIRPS_HE':{'20230411':{'mode':'model','dt':0.05}}})  
        else:
            data_dic['DI']['transit_prop']={'nsub_Dstar':2001,'NIRPS_HE':{'20230411':{'mode':'simu','n_oversamp':10.}}}
        
    if gen_dic['star_name']=='GJ1214':data_dic['DI']['transit_prop'].update({'NIRPS_HE':{'20230407':{'mode':'model','dt':0.05}}})  



        
      
    #Forcing in/out transit flag
    if 'HD3167_c' in gen_dic['transit_pl']:
        data_dic['DI']['idx_ecl'].update({'in':{},
                                'out':{'HARPN':{'2016-10-01':[9]}}})    
    
    elif ('TOI858b' in gen_dic['transit_pl']):
        data_dic['DI']['idx_ecl']={'in':{},
                                'out':{'CORALIE':{'20191205':[0],'20210118':[0,7]}}}

    elif 'Moon' in gen_dic['transit_pl']:
        data_dic['DI']['idx_ecl']={'in':{}, #'HARPS':{'2020-12-14':[range(149,156),range(288,293)]},
                                'out':{}}#}    
    elif gen_dic['star_name']=='55Cnc':
        data_dic['DI']['idx_ecl']={'out':{'ESPRESSO':{'20210124':[25]},'EXPRES':{'20220406':[10]}}}


    #RM survey
    elif gen_dic['star_name']=='Kepler63':
        data_dic['DI']['idx_ecl']={'out':{'HARPN':{'20200513':[9]}}}
    elif gen_dic['star_name']=='WASP166':
        data_dic['DI']['idx_ecl']={'out':{'HARPS':{'20170304':[2,42]}}}
    elif gen_dic['star_name']=='WASP107':
        data_dic['DI']['idx_ecl']={'out':{'HARPS':{'20140406':[7]}}}


    #Plot settings

    #Model time resolution   
    if 'Nu2Lupi_d' in gen_dic['transit_pl']:plot_dic['dt_LC']= 2.
    elif 'K2-139b' in gen_dic['transit_pl']:plot_dic['dt_LC']= 3.
    elif 'HIP41378d' in gen_dic['transit_pl']:plot_dic['dt_LC']= 2.
    elif gen_dic['star_name']=='WASP107':plot_dic['dt_LC']= 10.

    #Input light curves 
    plot_dic['input_LC']=''  #pdf

    #Scaling light curves
    plot_dic['spectral_LC']=''   #pdf

    #2D maps of disk-integrated profiles
    plot_dic['map_DI_prof']=''   #png   















    



















##################################################################################################
#%%% Module: 2D->1D conversion for disk-integrated spectra 
#    - converting 2D disk-integrated spectra into 1D spectra
#    - every operation afterwards will be performed on those profiles 
#    - prior to conversion, spectra are normalized in all orders to a flat, common continuum
##################################################################################################

#%%%% Activating
gen_dic['spec_1D_DI'] = False


#%%%% Calculating/retrieving
gen_dic['calc_spec_1D_DI']=True  


#%%%% Multi-threading
gen_dic['nthreads_spec_1D_DI']= 4 


#%%%% 1D spectral table
#    - specific to each instrument
#    - tables are uniformely spaced in ln(w) (with d[ln(w)] = dw/w)
#      start and end values given in A    
data_dic['DI']['spec_1D_prop']={}


#%%%% Plot settings

#%%%%% 2D maps
plot_dic['map_DI_1D']='' 


#%%%%% Individual spectra
plot_dic['sp_DI_1D']=''        



if __name__ == '__main__':   


    #Activating
    gen_dic['spec_1D_DI']=True  & False
    
    #Calculating/retrieving
    gen_dic['calc_spec_1D_DI']=True   &  False  
    
    if gen_dic['star_name'] in ['HD209458','WASP76']:
        gen_dic['spec_1D_DI']=True    & False      
        gen_dic['calc_spec_1D_DI']= True  #   & False        
        

    #Multi-threading
    gen_dic['nthreads_spec_1D_DI']= 2 #14

    #1D spectral table 
    data_dic['DI']['spec_1D_prop']={
        # 'ESPRESSO':{'dlnw':1./5000.,'w_st':3000.,'w_end':8000.}}    
        'ESPRESSO':{'dlnw':0.01/6000.,'w_st':3000.,'w_end':9000.}}  
    if (gen_dic['star_name']=='HD209458') and gen_dic['trim_spec']:data_dic['DI']['spec_1D_prop']={'ESPRESSO':{'dlnw':0.01/6000.,'w_st':5780.,'w_end':6005.}}

    #2D maps
    plot_dic['map_DI_1D']=''   #'png 

    #Individual spectra
    plot_dic['sp_DI_1D']=''     #pdf     


















##################################################################################################
#%%% Module: binning disk-integrated profiles
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
#    - beware to use the alignement module to calculate binned profiles in the star rest frame
data_dic['DI']['dim_bin']='phase' 


#%%%% Bin definition
#    - bins are defined for each instrument/visit
#    - bins can be defined
# + manually: indicate lower/upper bin boundaries (ordered)
# + automatically : indicate total range and number of bins 
#    - for each visit, indicate a reference planet if more than one planet is transiting, and if the bin dimensions is specific to a given planet
data_dic['DI']['prop_bin']={}

            
#%%%% Plot settings

#%%%%% 2D maps of binned profiles
plot_dic['map_DIbin']=''      


#%%%%% Individual binned profiles
plot_dic['DIbin']=''    


#%%%%% Residuals from binned profiles
plot_dic['DIbin_res']=''  



if __name__ == '__main__':


    #Activating 
    gen_dic['DIbin'] = True  &  False
    gen_dic['DIbinmultivis'] = True      &  False
    if ((gen_dic['star_name'] in ['55Cnc']) and (gen_dic['type']['ESPRESSO']=='spec2D')) or \
       ((gen_dic['star_name'] in ['GJ3090']) and ((gen_dic['type']['NIRPS_HA']=='spec2D') or (gen_dic['type']['NIRPS_HE']=='spec2D'))):
        gen_dic['DIbin'] = False 
        gen_dic['DIbinmultivis'] =   False
    if gen_dic['star_name'] in ['HD189733','WASP43','L98_59','GJ1214']:gen_dic['DIbin']=True 
    
    
    #Calculating/retrieving
    gen_dic['calc_DIbin']=True   &  False  
    gen_dic['calc_DIbinmultivis'] = True      &  False
    if gen_dic['star_name'] in ['HD209458','WASP76']:
        gen_dic['DIbinmultivis']=True & False  
        gen_dic['calc_DIbinmultivis']= True  & False       
   
    
    #Visits to be binned    
    data_dic['DI']['vis_in_bin']={}   

    #Exposures to be binned
    if gen_dic['transit_pl']=='GJ436_b':
        data_dic['DI']['idx_in_bin']={}
    #     data_dic['DI']['idx_in_bin']={'HARPS':{'2007-05-09':[0,1,2,3,4,5]+list(np.arange(17,35,dtype=int))}}  
    #     data_dic['DI']['idx_in_bin']={'HARPS':{'2007-05-09':list(np.arange(17,34,dtype=int))}}  #post-transit


    # elif gen_dic['transit_pl']=='HD3167_b': 
    #     data_dic['DI']['idx_in_bin']={'ESPRESSO':{'2019-10-09':[0,1]}}        #pre-tr
    #     # data_dic['DI']['idx_in_bin']={'ESPRESSO':{'2019-10-09':range(19,30)}}        #1st half post-tr spectra
    #     # data_dic['DI']['idx_in_bin']={'ESPRESSO':{'2019-10-09':range(30,39)}}        #2nd half post-tr spectra


    # elif gen_dic['transit_pl']=='HD3167_c':    
#        data_dic['DI']['idx_in_bin']['HARPN']={'2016-10-01':list(np.arange(9,dtype=int))+list(np.arange(30,36,dtype=int))}}        #mask G2/K5, Dalal+2019    decommenter pour parametres nominaux

    # if gen_dic['transit_pl']=='55Cnc_e':
        
    #     data_dic['DI']['idx_in_bin']={'HARPS':{
    #                     '2012-03-15':[0,1,2,3]+list(np.arange(30,41,dtype=int))},   
    #                  'HARPN':{
    #                     '2012-12-25':list(np.arange(7,27,dtype=int)),
    #                     '2013-11-14':list(np.arange(25,dtype=int))+list(np.arange(46,53,dtype=int))},  
    #                  'binned':{'all_HARPSS':list(np.arange(6,dtype=int))+list(np.arange(37,59,dtype=int))}}          
        
    #     data_dic['DI']['idx_in_bin']={'ESPRESSO':{'2020-02-05':list(np.arange(97,dtype=int))}}   #derniere expo montre forte deviation de RV        
        
    if gen_dic['transit_pl']=='WASP121b':
        data_dic['DI']['idx_in_bin']={}
        
        data_dic['DI']['idx_in_bin']={}
    #    data_dic['DI']['idx_in_bin']={'HARPS':{
    #            '31-12-17':list(np.arange(10,dtype=int))+list(np.arange(26,35,dtype=int)), 
    #            '09-01-18':list(np.arange(8,dtype=int))+list(np.arange(29,55,dtype=int)),            
    #            '14-01-18':list(np.arange(19,dtype=int))+list(np.arange(39,50,dtype=int))
    #
    #            '31-12-17':list(np.arange(10,dtype=int)),    #pre-transit seul  
    #            '09-01-18':list(np.arange(8,dtype=int)),            
    #            '14-01-18':list(np.arange(19,dtype=int))
    
    #          
    #            '31-12-17':list(np.arange(26,35,dtype=int)),    #post-transit seul  
    #            '09-01-18':list(np.arange(29,55,dtype=int)),            
    #            '14-01-18':list(np.arange(39,50,dtype=int))
    #            }} 
        
    #    data_dic['DI']['idx_in_bin']={'HARPS':{
    ##            '31-12-17':list(np.arange(10,dtype=int)),    #pre-transit seul  
    ##            '09-01-18':list(np.arange(8,dtype=int)),            
    ##            '14-01-18':list(np.arange(19,dtype=int))            
    ##            
    #            '31-12-17':list(np.arange(26,35,dtype=int)),    #post-transit seul  
    ##            '09-01-18':list(np.arange(29,55,dtype=int)),            
    ##            '14-01-18':list(np.arange(39,50,dtype=int)) 
    #            }} 
        
    #if gen_dic['transit_pl']=='Kelt9b':
    #    data_dic['DI']['idx_in_bin']={'HARPN':{'20-07-2018':[0,1]}}
    

    if gen_dic['transit_pl']=='HAT_P3b':
        data_dic['DI']['idx_in_bin']={'HARPN':{'2019-04-15':[0,1,2,3,4,5,6,7,8,9,10,11]}}     
    if gen_dic['transit_pl']=='Nu2Lupi_c':
        data_dic['DI']['idx_in_bin']={'ESPRESSO':{'2020-03-18':list(range(0,24))+list(range(59,68))}}     #exclusion des 2 dernières expos 
        data_dic['DI']['idx_in_bin']={'ESPRESSO':{'2020-03-18':range(59,68)}}     #exclusion des 2 dernières expos + master post-tr
    # if gen_dic['transit_pl']=='GJ9827d':      with post-tr exposures excluded, all out-tr can be used
        
    elif 'Moon' in gen_dic['transit_pl']:    
        #'HARPS':{'2019-07-02':np.arange(370)},
        data_dic['DI']['idx_in_bin']=={'HARPS':{'2020-12-14':list(np.arange(88,139,dtype=int))+list(np.arange(303,354,dtype=int))}}      #list(range(0,370))}}    

    elif gen_dic['star_name']=='HAT_P11':
        data_dic['DI']['idx_in_bin'] ={'CARMENES_VIS':{
                '20170807':list(np.arange(31,dtype=int)),
                '20170812':list(np.arange(11,dtype=int))+list(np.arange(30,36,dtype=int))}}  
    
    elif gen_dic['star_name']=='HAT_P49':
        data_dic['DI']['idx_in_bin']={'HARPN':{'20200730':list(np.arange(7,22))}}         

    elif gen_dic['star_name']=='WASP156':
        data_dic['DI']['idx_in_bin']={'CARMENES_VIS':{'20190928':list(np.arange(12,16))}}  

    # elif gen_dic['star_name']=='55Cnc':
    #     if gen_dic['DIbinmultivis']:data_dic['DI']['idx_in_bin']={'HARPS':{'20120127':list(np.arange(47)),'20120213':list(np.arange(55)),'20120227':list(np.arange(36)),'20120315':list(np.arange(40))}}      
    

    
    #Binning dimension
    data_dic['DI']['dim_bin']='phase' 


    #Bin definition
    if gen_dic['star_name']=='HD3167':   
        data_dic['DI']['prop_bin']={
                                 'ESPRESSO':{'2019-10-09':{'bin_range':[-0.5,0.5],'nbins':1}},
                                 'HARPN':{'2016-10-01':{'bin_range':[-0.5,0.5],'nbins':1}}}   
    elif gen_dic['star_name']=='WASP76':   
        data_dic['DI']['prop_bin']={
            # 'ESPRESSO':{'2018-09-03':{'bin_low':np.arange(-0.03,0.03,0.01),'bin_high':np.arange(-0.03,0.03,0.01)+0.01},
            #             '2018-10-31':{'bin_low':np.arange(0.,0.03,0.01),'bin_high':np.arange(0.,0.03,0.01)+0.01}},
                        # 'binned':{'bin_low':np.arange(0.,0.03,0.01),'bin_high':np.arange(0.,0.03,0.01)+0.01}}
            
            'ESPRESSO':{'20180902':{'bin_range':[-0.5,0.5],'nbins':1},
                        '20181030':{'bin_range':[-0.5,0.5],'nbins':1},
                        'binned':{'bin_range':[-0.5,0.5],'nbins':1}}
            }        
        
    elif 'Moon' in gen_dic['transit_pl']:
        data_dic['DI']['prop_bin']={ 
            
            'HARPS':{'2020-12-14':{'bin_range':[-0.5,0.5],'nbins':1}}
            
            }
                #'2019-07-02':{'bin_range':[-0.5,0.5],'nbins':1},
                     
            #}        
        
    elif 'TOI858b' in gen_dic['transit_pl']:
        data_dic['DI']['prop_bin']={'CORALIE':{'20191205':{'bin_range':[-0.5,0.5],'nbins':1},'20210118':{'bin_range':[-0.5,0.5],'nbins':1}}}        

    # elif gen_dic['star_name']=='GJ436':   
    #     data_dic['DI']['prop_bin']={'ESPRESSO':{'20190228':{'bin_range':[-0.5,0.5],'nbins':1},'20190429':{'bin_range':[-0.5,0.5],'nbins':1},'binned':{'bin_range':[-0.5,0.5],'nbins':1}},
    #                              'HARPN':{'20160318':{'bin_range':[-0.5,0.5],'nbins':1},'20160411':{'bin_range':[-0.5,0.5],'nbins':1}},   
    #                              'HARPS':{'20070509':{'bin_range':[-0.5,0.5],'nbins':1}}}   
    
    
    elif gen_dic['star_name']=='HIP41378':   
        data_dic['DI']['prop_bin']={'HARPN':{'20191218':{'bin_range':[-0.5,0.5],'nbins':1},'20220401':{'bin_range':[-0.5,0.5],'nbins':1}}}                
    elif gen_dic['star_name']=='MASCARA1':   
        data_dic['DI']['prop_bin']={'ESPRESSO':{'20190714':{'bin_range':[-0.5,0.5],'nbins':1},'20190811':{'bin_range':[-0.5,0.5],'nbins':1}}}                
    elif gen_dic['star_name']=='V1298tau':   
        data_dic['DI']['prop_bin']={'HARPN':{'20200128':{'bin_range':[-0.5,0.5],'nbins':1},'20201207':{'bin_range':[-0.5,0.5],'nbins':1}}}                
                 
    elif gen_dic['star_name']=='55Cnc':           
        vis_list_dic = { 
            'ESPRESSO': ['20200205','20210121','20210124','binned'],
            'HARPS':  ['20120127','20120213','20120227','20120315','binned']        ,   
            'HARPN':   ['20121225','20131114','20131128','20140101','20140126','20140226','20140329','binned']  ,        
            'SOPHIE': ['20120202','20120203','20120205','20120217','20120219','20120222','20120225','20120302','20120324','20120327','20130303','binned']  }         
        for inst in ['ESPRESSO','HARPS','HARPN','SOPHIE']:     
            data_dic['DI']['prop_bin'][inst]={}
            for vis in vis_list_dic[inst]:
                if vis=='binned':
                    if inst=='ESPRESSO':data_dic['DI']['prop_bin'][inst][vis] = {'bin_range':[-0.16  ,0.16   ],'nbins':53}  
                    elif inst=='HARPS':data_dic['DI']['prop_bin'][inst][vis] = {'bin_range':[-0.06  ,0.105  ],'nbins':28}  
                else:data_dic['DI']['prop_bin'][inst][vis] = {'bin_range':[-0.5,0.5],'nbins':1}  
    elif gen_dic['star_name']=='HD209458':
        data_dic['DI']['prop_bin']={'ESPRESSO':{'mock_vis':{'bin_range':[-0.5,0.5],'nbins':1}}} 
        
    #RM survey
    elif gen_dic['star_name']=='HAT_P3':
        data_dic['DI']['prop_bin']={'HARPN':{'20190415':{'bin_range':[-0.5,0.5],'nbins':1},'20200130':{'bin_range':[-0.5,0.5],'nbins':1}}}           
    elif gen_dic['star_name']=='Kepler25':
        data_dic['DI']['prop_bin']={'HARPN':{'20190614':{'bin_range':[-0.5,0.5],'nbins':1}}} 
    elif gen_dic['star_name']=='Kepler68':
        data_dic['DI']['prop_bin']={'HARPN':{'20190803':{'bin_range':[-0.5,0.5],'nbins':1}}}  
    elif gen_dic['star_name']=='HAT_P33':
        data_dic['DI']['prop_bin']={'HARPN':{'20191204':{'bin_range':[-0.5,0.5],'nbins':1}}}  
    elif gen_dic['star_name']=='K2_105':
        data_dic['DI']['prop_bin']={'HARPN':{'20200118':{'bin_range':[-0.5,0.5],'nbins':1}}} 
    elif gen_dic['star_name']=='HD89345':
        data_dic['DI']['prop_bin']={'HARPN':{'20200202':{'bin_range':[-0.5,0.5],'nbins':1}}}  
    elif gen_dic['star_name']=='Kepler63':
        data_dic['DI']['prop_bin']={'HARPN':{'20200513':{'bin_range':[-0.5,0.5],'nbins':1}}}
    elif gen_dic['star_name']=='HAT_P49':
        data_dic['DI']['prop_bin']={'HARPN':{'20200730':{'bin_range':[-0.5,0.5],'nbins':1}}}
    elif gen_dic['star_name']=='WASP47':
        data_dic['DI']['prop_bin']={'HARPN':{'20210730':{'bin_range':[-0.5,0.5],'nbins':1}}}
    elif gen_dic['star_name']=='WASP107':
        data_dic['DI']['prop_bin']={'HARPS':{'20140406':{'bin_range':[-0.5,0.5],'nbins':1},'20180201':{'bin_range':[-0.5,0.5],'nbins':1},'20180313':{'bin_range':[-0.5,0.5],'nbins':1},'binned':{'bin_range':[-0.5,0.5],'nbins':1}},
                                    'CARMENES_VIS':{'20180224':{'bin_range':[-0.5,0.5],'nbins':1}},'binned':{'bin_range':[-0.5,0.5],'nbins':1}}    
    elif gen_dic['star_name']=='WASP166':
        data_dic['DI']['prop_bin']={'HARPS':{'20170114':{'bin_range':[-0.5,0.5],'nbins':1},'20170304':{'bin_range':[-0.5,0.5],'nbins':1},'20170315':{'bin_range':[-0.5,0.5],'nbins':1},'binned':{'bin_range':[-0.5,0.5],'nbins':1}}}    
    elif gen_dic['star_name']=='HAT_P11':
        data_dic['DI']['prop_bin']={'HARPN':{'20150913':{'bin_range':[-0.5,0.5],'nbins':1},'20151101':{'bin_range':[-0.5,0.5],'nbins':1},'binned':{'bin_range':[-0.5,0.5],'nbins':1}},
                                    'CARMENES_VIS':{'20170807':{'bin_range':[-0.5,0.5],'nbins':1},'20170812':{'bin_range':[-0.5,0.5],'nbins':1},'binned':{'bin_range':[-0.5,0.5],'nbins':1}}}    
    elif gen_dic['star_name']=='WASP156'  :
        data_dic['DI']['prop_bin']={'CARMENES_VIS':{'20190928':{'bin_range':[-0.5,0.5],'nbins':1},'20191025':{'bin_range':[-0.5,0.5],'nbins':1},'20191210':{'bin_range':[-0.5,0.5],'nbins':1}}}
    elif gen_dic['star_name']=='HD106315':
        data_dic['DI']['prop_bin']={'HARPS':{'20170309':{'bin_range':[-0.5,0.5],'nbins':1},'20170330':{'bin_range':[-0.5,0.5],'nbins':1},'20180323':{'bin_range':[-0.5,0.5],'nbins':1},'binned':{'bin_range':[-0.5,0.5],'nbins':1}}}    
        
        
     
    #2D maps of binned profiles
    plot_dic['map_DIbin']=''   #png       
    
    #Individual binned profiles
    plot_dic['DIbin']=''  
    if gen_dic['star_name'] in ['HD189733','WASP43','L98_59','GJ1214']:plot_dic['DIbin']='png'

    #Residuals from binned profiles
    plot_dic['DIbin_res']=''  #pdf





    














##################################################################################################
#%%% Module: disk-integrated CCF masks
#    - spectra must have been aligned in the star rest frame (using a approximate 'sysvel'), converted into a 1D profile, and binned
#    - the mask is determined by default from a master spectrum built over all processed visits of an instrument, for consistency of the CCFs between visits
#    - the mask is saved as a .txt file in air or vacuum (depending on the pipeline process) and as a .fits file in air to be read by ESPRESSO-like DRS
##################################################################################################


#%%%% Activating
gen_dic['def_DImasks'] = False


#%%%% Multi-threading
data_dic['DI']['mask']['nthreads'] = 14 


#%%%% Print status
data_dic['DI']['mask']['verbose'] = False


#%%%% Estimate of line width 
#    - in km/s
data_dic['DI']['mask']['fwhm_ccf'] = 5. 


#%%%%% Vicinity window
#    - in fraction of 'fwhm_ccf'
#    - window for extrema localization in pixels of the regular grid = int(min(fwhm_ccf*w_reg/(vicinity_fwhm*c_light*dw_reg)))
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
#    - format is {inst:[[w1,w2],[w3,w4],..]}
data_dic['DI']['mask']['line_rej_range'] = {}

 
#%%%% Line selection: depth and width   
#    - check using plot_dic['DImask_spectra'] with step='sel1'

#%%%%% Depth range
#    - minimum/maximum line depths to be considered in the stellar mask (counted from the continuum, and from the local maxima)
#    - use plot_dic['DImask_ld'] to adjust
#    - between 0 and 1, per instrument
data_dic['DI']['mask']['linedepth_cont_min'] = {}   
data_dic['DI']['mask']['linedepth_min'] = {}  
data_dic['DI']['mask']['linedepth_cont_max'] = {} 


#%%%%% Minimum depth and width
#    - selection criteria on minimum line depth and half-width (between minima and closest maxima) to be kept (value > 10^(crit)) 
#    - use plot_dic['DImask_ld_lw'] to adjust
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
data_dic['DI']['mask']['tell_star_depthR_min'] = None


#%%%% VALD cross-validation     

#%%%%% Path to VALD linelist
#    - set to None to prevent
#    - see details in theo_dic['st_atm']
data_dic['DI']['mask']['VALD_linelist'] = None


#%%%%% Adjusting VALD line depth
data_dic['DI']['mask']['VALD_depth_corr'] = True


#%%%% Line selection: morphological 

#%%%%% Symmetry
#    - selection criteria on maximum ratio between normalized continuum difference and relative line depth, and normalized asymetry parameter, to be kept (value < crit) 
data_dic['DI']['mask']['diff_cont_rel_max'] = None
data_dic['DI']['mask']['asym_ddflux_max'] = None    


#%%%%% Width and depth
#    - selection criteria on minimum line depth (value > crit) and maximum line width (value < crit) to be kept
data_dic['DI']['mask']['width_max'] = None 
data_dic['DI']['mask']['diff_depth_min'] = None

    
#%%%% Line selection: RV dispersion 
#    - set to True to activate 
data_dic['DI']['mask']['RV_disp_sel'] = True


#%%%%% Exposures selection
#    - indexes of exposures from which RV are derived
#    - indexes are relative to the global table in each visit, but only exposures used to build the master spectrum will be considered (used by default if left empty)
data_dic['DI']['mask']['idx_RV_disp_sel']={}


#%%%%% RV deviation
#    - lines with absolute RVs beyond this value are excluded (in m/s)
data_dic['DI']['mask']['absRV_max'] = {}


#%%%%% Dispersion deviation
#    - lines with RV dispersion/mean error over the exposure series beyond this value are excluded
#    - beware not to use this criterion when there are too few exposures
data_dic['DI']['mask']['RVdisp2err_max'] = {}


#%%%% Plot settings 

#%%%%% Mask at successive steps
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





if __name__ == '__main__':

    #Activating
    gen_dic['def_DImasks'] = True  &  False

    #Print status
    data_dic['DI']['mask']['verbose'] = True

    #Estimate of line width 
    if gen_dic['star_name']=='HD209458':
        data_dic['DI']['mask']['fwhm_ccf'] = 9.45 
    if gen_dic['star_name']=='WASP76':
        data_dic['DI']['mask']['fwhm_ccf'] = 9.71 

    #Line selection: rejection ranges
    if gen_dic['star_name']=='HD209458':
        data_dic['DI']['mask']['line_rej_range'] = {'ESPRESSO':[[6276.,6316.],[6930.,6957.],[7705.,7717.]]}        
    if gen_dic['star_name']=='WASP76':
        data_dic['DI']['mask']['line_rej_range'] = {'ESPRESSO':[[6930.,6950.],[7175.,7350.],[7705.,7717.]]}        #strict
        # data_dic['DI']['mask']['line_rej_range'] = {'ESPRESSO':[[6930.,6950.],[7705.,7717.]]}        #relaxed        

    
    #Depth range    
    if gen_dic['star_name']=='HD209458':
        #Strict criteria (default 0.1, strict: linedepth_cont_min = 0.1; linedepth_cont_max = 0.95; linedepth_min = 0.01) 
        data_dic['DI']['mask']['linedepth_cont_min'] = {'ESPRESSO':0.06}         
        #Relaxed criteria
        # data_dic['DI']['mask']['linedepth_cont_min'] = {'ESPRESSO':0.03}    #most lines are below this threshold and contribute to about 20% of the total weights
        # data_dic['DI']['mask']['linedepth_cont_max'] = {'ESPRESSO':0.98}        
    # if gen_dic['star_name']=='WASP76':
    #     #Relaxed criteria (default, strict: linedepth_cont_min = 0.1; linedepth_cont_max = 0.95; linedepth_min = 0.01)       
    #     data_dic['DI']['mask']['linedepth_cont_min'] = {'ESPRESSO':0.05}    
    #     data_dic['DI']['mask']['linedepth_cont_max'] = {'ESPRESSO':0.98}        


    #Minimum depth and width
    # if gen_dic['star_name']=='HD209458':
    #     #Relaxed criteria (default, strict: line_width_logmin = -1.3; line_depth_logmin = -1.4)
    #     data_dic['DI']['mask']['line_width_logmin'] = -1.7   
    #     data_dic['DI']['mask']['line_depth_logmin'] = -2.6  
    # if gen_dic['star_name']=='WASP76':
    #     #Relaxed criteria (default, strict: line_width_logmin = -1.3; line_depth_logmin = -1.4)
    #     data_dic['DI']['mask']['line_width_logmin'] = -1.6   
    #     data_dic['DI']['mask']['line_depth_logmin'] = -2.5  
        

    #Line selection: position
    data_dic['DI']['mask']['win_core_fit'] = 0.7    #yieds fewer outliers than the default 1
    if gen_dic['star_name']=='HD209458':
        #Strict criteria (default 1500)
        data_dic['DI']['mask']['abs_RVdev_fit_max'] = 225.        
        #Relaxed criteria
        # data_dic['DI']['mask']['abs_RVdev_fit_max'] = 600    
    if gen_dic['star_name']=='WASP76':
        #Strict criteria (default 1500)
        data_dic['DI']['mask']['abs_RVdev_fit_max'] = 250.        
        # #Relaxed criteria
        # data_dic['DI']['mask']['abs_RVdev_fit_max'] = 600   


    #Threshold on telluric line depth 
    data_dic['DI']['mask']['tell_depth_min'] = 0.001

    #Thresholds on telluric/stellar lines depth ratio
    if gen_dic['star_name']=='HD209458':
        #Strict criteria 
        data_dic['DI']['mask']['tell_star_depthR_max'] = 0.15
        data_dic['DI']['mask']['tell_star_depthR_max_final'] = 0.025
        # #Relaxed criteria (default 0.2, strict: tell_star_depthR_max = 0.15)
        # data_dic['DI']['mask']['tell_star_depthR_max'] = 0.35        
        # data_dic['DI']['mask']['tell_star_depthR_max_final'] = 0.1    
    if gen_dic['star_name']=='WASP76':
        #Strict criteria (default 0.2, strict: tell_star_depthR_max = 0.1)
        data_dic['DI']['mask']['tell_star_depthR_max'] = 0.1             
        data_dic['DI']['mask']['tell_star_depthR_max_final'] = 0.025
        # #Relaxed criteria (=default)
        # data_dic['DI']['mask']['tell_star_depthR_max'] = 0.3   
        # data_dic['DI']['mask']['tell_star_depthR_max_final'] = 0.1         
        

    #Path to VALD linelist
    if gen_dic['star_name']=='HD209458':data_dic['DI']['mask']['VALD_linelist'] = '/Users/bourrier/Travaux/Exoplanet_systems/HD/HD209458b/Star/VALD/VALD_HD209458'
    if gen_dic['star_name']=='WASP76':data_dic['DI']['mask']['VALD_linelist'] = '/Users/bourrier/Travaux/Exoplanet_systems/WASP/WASP76b/Star/VALD/VALD_WASP76'

    #Adjusting VALD line depth
    data_dic['DI']['mask']['VALD_depth_corr'] = True


    #Symmetry   
    if gen_dic['star_name']=='HD209458':
        #Strict criteria 
        data_dic['DI']['mask']['diff_cont_rel_max'] = 1.3   #selected to keep 80% of total weights
        data_dic['DI']['mask']['asym_ddflux_max'] = 0.3           
        # #Relaxed criteria 
        # data_dic['DI']['mask']['diff_cont_rel_max'] = 5.   #selected to keep 90% of total weights
        # data_dic['DI']['mask']['asym_ddflux_max'] = 0.6          
    if gen_dic['star_name']=='WASP76':
        #Strict criteria 
        data_dic['DI']['mask']['diff_cont_rel_max'] = 1.3   #selected to keep 80% of total weights
        data_dic['DI']['mask']['asym_ddflux_max'] = 0.3           
        # #Relaxed criteria 
        # data_dic['DI']['mask']['diff_cont_rel_max'] = 5.   #selected to keep 90% of total weights
        # data_dic['DI']['mask']['asym_ddflux_max'] = 0.6  

    
    
    #Width and depth
    if gen_dic['star_name']=='HD209458':
        #Strict
        data_dic['DI']['mask']['width_max'] = 12.0 
        data_dic['DI']['mask']['diff_depth_min'] = 0.1
        # #Relaxed criteria 
        # data_dic['DI']['mask']['width_max'] = 15.0 
        # data_dic['DI']['mask']['diff_depth_min'] = 0.05
    if gen_dic['star_name']=='WASP76':
        #Strict
        data_dic['DI']['mask']['width_max'] = 10.0 
        data_dic['DI']['mask']['diff_depth_min'] = 0.1
        # #Relaxed criteria 
        # data_dic['DI']['mask']['width_max'] = 15.0 
        # data_dic['DI']['mask']['diff_depth_min'] = 0.05

    
    #Line selection: RV dispersion 
    data_dic['DI']['mask']['RV_disp_sel'] = True

    #Exposures selection
    data_dic['DI']['mask']['idx_RV_disp_sel']={}

    #RV deviation
    if gen_dic['star_name']=='HD209458':
        data_dic['DI']['mask']['absRV_max'] = {'ESPRESSO':3.}
        # #Relaxed criteria 
        # data_dic['DI']['mask']['absRV_max'] = {'ESPRESSO':10.}
    if gen_dic['star_name']=='WASP76':
        #Strict criteria 
        data_dic['DI']['mask']['absRV_max'] = {'ESPRESSO':10.}
        # #Relaxed criteria 
        # data_dic['DI']['mask']['absRV_max'] = {'ESPRESSO':20.}


    #Dispersion deviation
    if gen_dic['star_name']=='HD209458':
        data_dic['DI']['mask']['RVdisp2err_max'] = {'ESPRESSO':1.5}
        # #Relaxed criteria 
        # data_dic['DI']['mask']['RVdisp2err_max'] = {'ESPRESSO':2.}
    if gen_dic['star_name']=='WASP76':
        #Strict criteria
        data_dic['DI']['mask']['RVdisp2err_max'] = {'ESPRESSO':1.5}
        # #Relaxed criteria 
        # data_dic['DI']['mask']['RVdisp2err_max'] = {'ESPRESSO':2.}


    # Mask at successive steps
    plot_dic['DImask_spectra'] = ''

    # Depth range selection
    plot_dic['DImask_ld'] = ''

    # Minimum depth and width selection
    plot_dic['DImask_ld_lw'] = ''

    # Position selection
    plot_dic['DImask_RVdev_fit'] = ''

    # Telluric selection
    plot_dic['DImask_tellcont'] = ''

    # VALD selection
    plot_dic['DImask_vald_depthcorr'] = ''

    # Morphological (asymmetry) selection
    plot_dic['DImask_morphasym'] = ''

    # Morphological (shape) selection
    plot_dic['DImask_morphshape'] = ''

    # RV dispersion selection
    plot_dic['DImask_RVdisp'] = ''












##################################################################################################
#%% Residual and intrinsic profiles
##################################################################################################  



##################################################################################################
#%%% Module: extracting residual profiles
#    - potentially affected by the planetary atmosphere
##################################################################################################   

#%%%% Activating
gen_dic['res_data'] = False


#%%%% Calculating/retrieving 
gen_dic['calc_res_data'] = True 


#%%%% Multi-threading
gen_dic['nthreads_res_data']= 14


#%%%% In-transit restriction
#    - limit the extraction of residual profiles to in-transit exposures
#    - this is only relevant when a specific master is calculated for each exposure (ie, when requesting the extraction of local profiles from 
# spectral data that are not defined on common bins), otherwise it does not cost time to extract local profiles from all exposures using a common master
#    - this will prevent plotting and analyzing local residuals outside of the transit
data_dic['Res']['extract_in'] = False


#%%%% Master visits
#    - visits to be included in the calculation of the master, for each instrument
#    - leave empty for the master to be calculated with the exposures of the considered visit only
#      otherwise this option can be used to boost the SNR of the master and/or smooth out variations in its shape
#      in that case aligned, scaled disk-integrated profiles must have been first calculated for all chosen visits
#    - which exposures contribute to the master in each visit is set through data_dic['Res']['idx_in_bin']
#    - if multiple planets are transiting in binned visits, set the planet to be used as reference for the phase (ie, the dimension along which exposures are binned) via data_dic['DI']['pl_in_bin']={inst:XX} 
data_dic['Res']['vis_in_bin']={}  


#%%%% Master exposures
#    - indexes of exposures that contribute to the master
#    - set to out-of-transit exposures if left undefined
data_dic['Res']['idx_in_bin']={}
        

#%%%% Continuum range
#    - format inst > ord > [ [x1,x2] , [x3,x4] , ... ] 
#    - used to set errors on local profiles from dispersion in their continuum, to set the continuum level or perform continuum correction of intrinsic profiles
#    - x are defined in the star rest frame
#      the ranges are common to all local profiles, ie that they must be large enough to cover the full range of RVs (with the width of the stellar
# line) from the regions along the transit chord    
#      the range does not need to be as large as defined for the raw CCFs, which can be broadened by rotation   
data_dic['Res']['cont_range']={}


#%%%% Error definition
#    - force errors on CCFs to their continuum dispersion
#    - if input data have no errors, error tables have already been set to C*sqrt(F) and propagated
#      if activated, the present option will override these tables (whether the input data had error table or not originally)
data_dic['Res']['disp_err']=False


#%%%% Plot settings

#%%%%% 2D maps of residual profiles
#    - in stellar rest frame
#    - can be used to check for spurious variations in all exposures: to do so, apply the local profile extraction after having aligned (and potentially corrected)
# all profiles, but without transit scaling (if flux balance correction has been applied) or after having applied a light curve unity
plot_dic['map_Res_prof']=''   


#%%%%% Individual residual profiles
plot_dic['sp_loc']=''    
plot_dic['CCF_Res']=''      




if __name__ == '__main__':

    #Activating
    gen_dic['res_data'] = True   #&  False
    if gen_dic['star_name'] in ['WASP43','L98_59','GJ1214']:gen_dic['res_data']=True
    if gen_dic['star_name']=='GJ436':gen_dic['res_data']=False

    #Calculating/retrieving 
    gen_dic['calc_res_data'] = True   &  False
    if gen_dic['star_name'] in ['HD209458','WASP76']:gen_dic['calc_res_data']=True   &False


    #Multi-threading
    gen_dic['nthreads_res_data']= 2

    #In-transit restriction
    data_dic['Res']['extract_in']=True  &  False
    
    #Master visits
    # if gen_dic['transit_pl']=='WASP76b':
    #     data_dic['Res']['vis_in_bin']={'ESPRESSO':['2018-09-03','2018-10-31']}  

    #Master exposures
    if gen_dic['star_name']=='HAT_P11':
        data_dic['Res']['idx_in_bin'] ={'CARMENES_VIS':{
            '20170807':list(np.arange(31,dtype=int)),
            '20170812':list(np.arange(11,dtype=int))+list(np.arange(30,36,dtype=int))}}     
    elif gen_dic['star_name']=='WASP156':
        data_dic['Res']['idx_in_bin'] ={'CARMENES_VIS':{
            '20190928':list(np.arange(12,16,dtype=int)),   #Test post-tr
            '20191025':list(np.arange(9,17,dtype=int)),
            # '20191210':list(np.arange(0,5,dtype=int)),   #no good
            }}  
    elif gen_dic['star_name']=='HAT_P49':    
        data_dic['Res']['idx_in_bin']={'HARPN':{'20200730':list(np.arange(7,22))}} 
    

    #Continuum range
    if 'GJ436_b' in gen_dic['transit_pl']:data_dic['Res']['cont_range']=[[-60.,-13.],[13.,60.]] 
    if gen_dic['star_name']=='55Cnc':
        data_dic['Res']['cont_range'] = {
         # 'ESPRESSO':[[-100.,-20.],[20.,100.]],  
         'ESPRESSO':[[-200.,-20.],[20.,200.]],      #to match the pca analysis
         'HARPS':[[-200.,-20.],[20.,200.]],         #to match the pca analysis
         'HARPN':[[-200.,-20.],[20.,200.]],       #to match the pca analysis
         'SOPHIE':[[-200.,-20.],[20.,200.]],        #to match the pca analysis
         'EXPRES':[[-200.,-65.],[-42.,-37.],[20.,200.]]}        #to match the pca analysis
        

    elif gen_dic['transit_pl']=='WASP121b':
        data_dic['Res']['cont_range']=[[-100.,-60.],[60.,100.]]     #Mask F
#        data_dic['Res']['cont_range']=[[-300.,-60.],[60.,300.]]     #Mask F + atmo
    elif gen_dic['transit_pl']=='Kelt9b':
        data_dic['Res']['cont_range']=[[-300.,-130.],[130.,300.]]       
    elif gen_dic['transit_pl']=='WASP127b':
        data_dic['Res']['cont_range']=[[-150.,-10.],[10.,150.]]      
    elif gen_dic['transit_pl']=='Corot7b':
        data_dic['Res']['cont_range']=[[-150.,-10.],[10.,150.]]  
        data_dic['Res']['cont_range']=[[10.+14.,21.+14.]]     #analyse continu bleu
        data_dic['Res']['cont_range']=[[-21.-14.,-10.-14.]]     #analyse continu rouge
    elif gen_dic['transit_pl']=='Nu2Lupi_c':
        data_dic['Res']['cont_range']=[[-150.,-9.],[9.,150.]]   
    elif gen_dic['transit_pl']=='GJ9827d':
        data_dic['Res']['cont_range']=[[-150.,-10.],[10.,150.]]    #Continu ESPRESSO ou HARPS  
        # data_dic['Res']['cont_range']=[[10.+14.,21.+14.]]     #analyse continu bleu
    elif gen_dic['transit_pl']=='GJ9827b':
        data_dic['Res']['cont_range']=[[-150.,-10.],[10.,150.]]    #Continu HARPS  
        # data_dic['Res']['cont_range']=[[10.+14.,21.+14.]]     #analyse continu bleu
    if gen_dic['star_name']=='HD3167': 
        # data_dic['Res']['cont_range']=[[-70.-160.,-20.-160.],[20.-160.,70.-160.]]      #continu bleu
        # data_dic['Res']['cont_range']=[[-70.-100.,-20.-100.],[20.-100.,70.-100.]]      #continu bleu close
        # data_dic['Res']['cont_range']=[[-70.+100.,-20.+100.],[20.+100.,70.+100.]]      #continu rouge close
        data_dic['Res']['cont_range']=[[-70.,-20.],[20.,70.]] 
        
    if 'Moon' in gen_dic['transit_pl']:
        data_dic['Res']['cont_range']=[[-100.,-10.],[10.,100.]]
        
    if 'Mercury' in gen_dic['transit_pl']:
        data_dic['Res']['cont_range']=[[-100.,-10.],[10.,100.]]        
        
    if 'TOI858b' in gen_dic['transit_pl']:data_dic['Res']['cont_range']=[[-100.,-15.],[18.,100.]] 
    if gen_dic['star_name']=='HIP41378':data_dic['Res']['cont_range']={'HARPN':[[-100.,-15.],[15.,100.]]}  
    elif gen_dic['star_name']=='MASCARA1':data_dic['Res']['cont_range']=[[-150.,-70.],[70.,150.]]  
    elif gen_dic['star_name']=='V1298tau':data_dic['Res']['cont_range']={'HARPN' : [[-150.,-70.],[70.,150.]]  }


    #RM survey
    elif gen_dic['star_name']=='HAT_P3':data_dic['Res']['cont_range']={'HARPN':[[-70.,-20.],[20.,70.]]}          
    elif gen_dic['star_name']=='Kepler25':data_dic['Res']['cont_range']={'HARPN':[[-90.,-30.],[30.,90.]]}     
    elif gen_dic['star_name']=='Kepler68':data_dic['Res']['cont_range']={'HARPN':[[-80.,-20.],[20.,80.]]}     
    elif gen_dic['star_name']=='HAT_P33':data_dic['Res']['cont_range']={'HARPN':[[-120.,-40.],[40.,120.]]} 
    elif gen_dic['star_name']=='K2_105':data_dic['Res']['cont_range']={'HARPN':[[-65.,-15.],[15.,65.]]} 
    elif gen_dic['star_name']=='HD89345':data_dic['Res']['cont_range']={'HARPN':[[-100.,-25.],[25.,100.]]} 
    elif gen_dic['star_name']=='Kepler63':data_dic['Res']['cont_range']={'HARPN':[[-80.,-25.],[25.,80.]]}  
    elif gen_dic['star_name']=='HAT_P49':data_dic['Res']['cont_range']={'HARPN':[[-150.,-30.],[30.,150.]]} 
    elif gen_dic['star_name']=='WASP47':data_dic['Res']['cont_range']={'HARPN':[[-70.,-25.],[25.,70.]]} 
    elif gen_dic['star_name']=='WASP107':data_dic['Res']['cont_range']={'HARPS':[[-80.,-25.],[25.,80.]],'CARMENES_VIS':[[-110.,-50.],[50.,110.]]} 
    elif gen_dic['star_name']=='WASP166':data_dic['Res']['cont_range']={'HARPS':[[-75.,-25.],[25.,75.]]} 
    elif gen_dic['star_name']=='HAT_P11':data_dic['Res']['cont_range']={'HARPN':[[-100.,-20.],[20.,40.]],'CARMENES_VIS':[[-110.,-50.],[50.,110.]]} 
    elif gen_dic['star_name']=='WASP156'  :data_dic['Res']['cont_range']={'CARMENES_VIS':[[-110.,-50.],[50.,110.]]} 
    elif gen_dic['star_name']=='HD106315':
        data_dic['Res']['cont_range']={'HARPS':[[-140.,-60.],[60.,140.]]} 
        data_dic['Res']['cont_range']={'HARPS':[[-120.,-60.],[60.,120.]]} 
    elif gen_dic['star_name']=='HD189733':data_dic['Res']['cont_range']['ESPRESSO']=[[-80.,-25.],[+25.,+80.]]        
    elif gen_dic['star_name']=='WASP43':data_dic['Res']['cont_range']['NIRPS_HE']=[[-100.,-7.],[+7.,+100.]]      
    elif gen_dic['star_name']=='L98_59':data_dic['Res']['cont_range']['NIRPS_HE']=[[-100.,-5.],[+5.,+100.]]   
    elif gen_dic['star_name']=='GJ1214':data_dic['Res']['cont_range']['NIRPS_HE']=[[-100.,-10.],[+10.,+100.]]  
    elif gen_dic['star_name']=='WASP76':
        data_dic['Res']['cont_range']['ESPRESSO']={0:[[-150.,-80.],[80.,150.]]}    #to avoid planet-excluded ranges  
    #     data_dic['Res']['cont_range_MCCF']=[[-200.,-60.],[60.,200.]]    
    elif gen_dic['star_name']=='HD209458':
        data_dic['Res']['cont_range']['ESPRESSO']={0:[[-80.,-20.],[20.,80.]]}   
    

    #Error definition
    data_dic['Res']['disp_err']=False


    #2D maps of residual profiles
    plot_dic['map_Res_prof']=''   #png 

    #Individual residual profiles
    plot_dic['sp_loc']=''    #png
    plot_dic['CCF_Res']=''   #pdf    
    













    
    


##################################################################################################
#%%% Module: extracting intrinsic profiles
#    - derived from the local profiles (in-transit), reset to the same broadband flux level, with planetary contamination excluded 
##################################################################################################    

#%%%% Calculating
gen_dic['intr_data'] = False


#%%%% Calculating/retrieving
gen_dic['calc_intr_data'] = True  
 

#%%%%% Continuum range
data_dic['Intr']['cont_range'] = {}


#%%%% Continuum correction
#    - the continuum might show differences between exposures because of imprecisions on the flux balance correction
#      this option correct for these deviations and update the flux scaling values accordingly
data_dic['Intr']['cont_norm'] = False


#%%%% Plot settings

#%%%%% 2D map: intrinsic stellar profiles
#    - aligned or not
plot_dic['map_Intr_prof']=''   


#%%%%% Individual intrinsic stellar profiles
#    - aligned or not
plot_dic['sp_intr']=''  
plot_dic['CCF_Intr']=''    


#%%%%% Residuals from intrinsic stellar profiles
#    - choose within the routine whether to plot fit to individual or to global profiles
plot_dic['CCF_Intr_res']=''  


if __name__ == '__main__':

    #Calculating
    gen_dic['intr_data'] = True    &  False
    if (gen_dic['star_name'] in ['WASP76','HD209458']) and (gen_dic['type']['ESPRESSO']=='spec2D'):gen_dic['intr_data'] = True#  & False
    if gen_dic['star_name'] in ['WASP43','L98_59','GJ1214']:gen_dic['intr_data']=True
    
    #Calculating/retrieving
    gen_dic['calc_intr_data'] = True   &  False  

    #Continuum range
    data_dic['Intr']['cont_range'] = deepcopy(data_dic['Res']['cont_range'])
    if gen_dic['star_name']=='HD209458':
        data_dic['Intr']['cont_range'] = deepcopy(data_dic['Res']['cont_range'])  #white CCFs
        # data_dic['Intr']['cont_range']['ESPRESSO']={0:[[-120.,-50.],[50.,120.]]}    #Na CCF
        if gen_dic['trim_spec']:
            data_dic['Intr']['cont_range']['ESPRESSO']={}
            for iord in range(4):data_dic['Intr']['cont_range']['ESPRESSO'][iord] = np.array([[ 5883. , 5885.],[5901., 5903. ]])    #ANTARESS fit sodium doublet


    #Continuum correction
    data_dic['Intr']['cont_norm'] = True   &   False
    if gen_dic['star_name'] in ['HD209458','WASP76']:
        data_dic['Intr']['cont_norm'] = True   #activate if CCF intr are generated





    #2D maps of intrinsic stellar profiles
    plot_dic['map_Intr_prof']=''   #'png 
    if gen_dic['star_name'] in ['HD189733','WASP43','L98_59','GJ1214']:plot_dic['map_Intr_prof']='png'

    #Individual intrinsic stellar profiles
    plot_dic['sp_intr']=''  
    plot_dic['CCF_Intr']=''   #pdf  

    #Residuals from intrinsic stellar profiles
    plot_dic['CCF_Intr_res']=''  #pdf
    












##################################################################################################
#%%% Module: CCF conversion for residual & intrinsic spectra 
#    - calculating CCFs from OT residual and intrinsic stellar spectra
#    - for analysis purpose, ie do not apply if atmospheric extraction is later requested
#    - every analysis afterwards will be performed on those CCFs
#    - ANTARESS will stop if intrinsic profiles are simultaneously required to extract atmospheric spectra 
##################################################################################################   
 

#%%%% Activating
gen_dic['Intr_CCF'] = False

 
#%%%% Calculating/retrieving 
gen_dic['calc_Intr_CCF'] = True 


#%%%% Error definition
#    - force errors on out-of-transit residual and intrinsic CCFs to their continuum dispersion
data_dic['Intr']['disp_err']=False


if __name__ == '__main__':    
 
    #Activating
    gen_dic['Intr_CCF'] = True  # &  False
    if gen_dic['star_name']=='GJ436':gen_dic['Intr_CCF'] =  False
 
    #Calculating/retrieving 
    gen_dic['calc_Intr_CCF'] = True  &  False

    #Error definition
    data_dic['Intr']['disp_err']=False

















##################################################################################################
#%%% Module: PCA of out-of-transit residual profiles
#    - for now only coded for CCF data type
#    - use this module to derive PC and match their combination to residual and intrinsic profiles in the fit module
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




if __name__ == '__main__':


    
    #Activating
    gen_dic['pca_ana'] = True  &  False
    
    #Calculating/retrieving
    gen_dic['calc_pca_ana'] = True # &  False      
    
    #Visits to be processed
    if gen_dic['star_name']=='55Cnc':
        data_dic['PCA']['vis_list']={
            'ESPRESSO':'all',
            'HARPS':'all',
            'HARPN':'all',
            'EXPRES':'all',
            }  

    #Exposures contributing to the PCA
    data_dic['PCA']['idx_pca']={}
    
    #Define spectral range over which PC are determined 
    if gen_dic['star_name']=='55Cnc':
        data_dic['PCA']['ana_range'] = {'ESPRESSO':{'20200205':[[-200.,200.]],'20210121':[[-200.,200.]],'20210124':[[-200.,200.]]},
            # 'HARPS':{'20120127':[[-200.,200.]],'20120213':[[-200.,200.]],'20120227':[[-200.,200.]],'20120315':[[-200.,200.]]}}
            'HARPN':{'20131114':[[-200.,200.]],'20131128':[[-200.,200.]],'20140101':[[-200.,200.]],'20140126':[[-200.,200.]],'20140226':[[-200.,200.]],'20140329':[[-200.,200.]]} ,
            # 'EXPRES':{'20220131':[[-200.,200.]],'20220406':[[-200.,200.]]},            
            'EXPRES':{'20220131':[[-200.,-37.],[-22.,200.]],'20220406':[[-200.,-65.],[-42.,200.]]},  
        }    
            

    #Number of PC
    n_PC = 10
    data_dic['PCA']['n_pc'] = {
        'ESPRESSO':{'20200205':6,'20210121':6,'20210124':6},
        'HARPS':{'20120127':n_PC,'20120213':n_PC,'20120227':n_PC,'20120315':n_PC},     
        'HARPN':{'20131114':n_PC,'20131128':n_PC,'20140101':n_PC,'20140126':n_PC,'20140226':n_PC,'20140329':n_PC},
        'EXPRES':{'20220131':n_PC,'20220406':n_PC}, 
        }
        
    #Fitted exposures
    data_dic['PCA']['idx_corr']={}

    #Fitted spectral range
    data_dic['PCA']['fit_range'] = {
          # 'EXPRES':{'20220131':[[-200.,-37.],[-22.,-8.],[8.,200.]]}, 
          # 'EXPRES':{'20220131':[[-200.,-8.],[8.,200.]]}, 
        }

    #Bootstrap analysis
    data_dic['PCA']['nboot'] = 500

    #Residuals histograms
    data_dic['PCA']['nbins'] = 50


    #Plots: PCA results
    plot_dic['pca_ana'] = ''















##################################################################################################
#%%% Module: aligning intrinsic profiles
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




if __name__ == '__main__':

    #Activating
    gen_dic['align_Intr'] = True  &  False
 
    #Calculating/retrieving
    gen_dic['calc_align_Intr'] = True  #&  False  

    #Alignment mode
    data_dic['Intr']['align_mode']='theo'
    
    #Plots: all profiles 
    plot_dic['all_intr_data']=''   #pdf
















    
##################################################################################################
#%%% Module: 2D->1D conversion for residual & intrinsic spectra
#    - every analysis afterwards will be performed on those profiles
##################################################################################################

#%%%% Activating
gen_dic['spec_1D_Intr'] = False


#%%%% Calculating/retrieving 
gen_dic['calc_spec_1D_Intr']=True  


#%%%% Multi-threading
gen_dic['nthreads_spec_1D_Intr']= 4


#%%%% 1D spectral table
#    - see DI module for details
data_dic['Intr']['spec_1D_prop']={}   


#%%%% Plot settings

#%%%%% 2D maps
plot_dic['map_Intr_1D']=''   

#%%%%% Individual spectra
plot_dic['sp_Intr_1D']=''

#%%%%% Residuals from model     
plot_dic['sp_Intr_1D_res']='' 

 

if __name__ == '__main__':

    #Activating
    gen_dic['spec_1D_Intr']=True  & False

    #Calculating/retrieving 
    gen_dic['calc_spec_1D_Intr']=True  &  False   


    #Multi-threading
    gen_dic['nthreads_spec_1D_Intr']=  4


    #1D spectral table
    data_dic['Intr']['spec_1D_prop']={
        'ESPRESSO':{'dlnw':0.01/6000.,'w_st':3000.,'w_end':9000.}}   
    if (gen_dic['star_name']=='HD209458') and gen_dic['trim_spec']:data_dic['Intr']['spec_1D_prop']={'ESPRESSO':{'dlnw':0.01/6000.,'w_st':5780.,'w_end':6005.}}

    #Redefining continuum range
    if gen_dic['star_name']=='HD209458':
        if gen_dic['trim_spec']:
            data_dic['Intr']['cont_range']['ESPRESSO']={}
            data_dic['Intr']['cont_range']['ESPRESSO'][0] = np.array([[ 5883. , 5885.],[5901., 5903. ]])    #ANTARESS fit sodium doublet

    #2D maps
    plot_dic['map_Intr_1D']=''   #'png

    #Individual spectra
    plot_dic['sp_Intr_1D']=''     #pdf           
    plot_dic['sp_Intr_1D_res']=''     
    
    












##################################################################################################
#%%% Module: binning intrinsic profiles
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
#    - bins are defined for each instrument/visit
#    - bins can be defined
# + manually: indicate lower/upper bin boundaries (ordered)
# + automatically : indicate total range and number of bins  
#    - leave empty for a single master to be calculated over all selected exposures
data_dic['Intr']['prop_bin']={}
    
    
#%%%% Plot settings        
    
#%%%%% 2D maps of binned profiles
plot_dic['map_Intrbin']=''    


#%%%%% Individual binned profiles
plot_dic['sp_Intrbin']=''  
plot_dic['CCF_Intrbin']=''   


#%%%%% Residuals from binned profiles
plot_dic['CCF_Intrbin_res']=''  


#%%%%% Binned disk-integrated and intrinsic profiles comparison
plot_dic['binned_DI_Intr']=''    



if __name__ == '__main__':


    #Activating
    gen_dic['Intrbin'] = True  &  False
    gen_dic['Intrbinmultivis'] = True   &  False


    #Calculating/retrieving
    gen_dic['calc_Intrbin']=True    &  False 
    gen_dic['calc_Intrbinmultivis']=True  #  &  False 

    if gen_dic['star_name'] in ['HD209458','WASP76']:
        gen_dic['Intrbinmultivis']=True & False    
        gen_dic['calc_Intrbinmultivis']= True #& False    


    #Visits to be binned
    if gen_dic['transit_pl']=='GJ9827d':data_dic['Intr']['vis_in_bin']={'HARPS':['2018-08-18','2018-09-18']} 
    elif gen_dic['transit_pl']=='GJ9827b':data_dic['Intr']['vis_in_bin']={'HARPS':[' 2018-08-04','2018-08-15','2018-09-18','2018-09-19']} 
    
    
    #Exposures to be binned
    if gen_dic['transit_pl']=='WASP_8b':
        data_dic['Intr']['idx_in_bin']={}
        #data_dic['Intr']['idx_in_bin']={'HARPS':{'2008-10-04':range(2,44)}}  #indexes 2 to 43  - all values ok in transit
        #data_dic['Intr']['idx_in_bin']={'HARPS':{'2008-10-04':range(2,24)}}  #indexes 2 to 23  - first half of transit
        #data_dic['Intr']['idx_in_bin']={'HARPS':{'2008-10-04':range(24,44)}}  #indexes 24 to 43  - second half of transit
        #data_dic['Intr']['idx_in_bin']={'HARPS':{'2008-10-04':range(7,40)}}  #indexes 7 to 39  - full in-transit
    elif gen_dic['star_name']=='GJ436': 
        data_dic['Intr']['idx_in_bin']={}
        #data_dic['Intr']['idx_in_bin']={'HARPS':{'2007-05-09':range(1,8)},   #GJ436b
        #             'HARPN':{'2016-03-18':range(1,8),                        
        #                        '2016-04-11':range(1,8)},
        #             'binned':{'HARPSN-binned':range(2,9)}} #les nouvelles expos avant/apres transit sont mises dans le transit  
        #data_dic['Intr']['idx_in_bin']={'HARPS':{'2007-05-09':[1, 2, 3, 7]},   #GJ436b, en excluant expos avec ctrst qui devie de > 2 sigma de la valeur DI
        #             'HARPN':{'2016-03-18':[1, 2, 3, 5, 6, 7],                        
        #                        '2016-04-11':[1, 2, 3, 5, 6, 7]},
        #             'binned':{'HARPSN-binned':[2,3,4,6,7,8]}} 
        data_dic['Intr']['idx_in_bin']={'HARPS':{'2007-05-09':[3,4,5,6]},   #GJ436b,  expos fort contraste a mu > 0.47
                      'HARPN':{'2016-03-18':[3,4,5],                        
                                '2016-04-11':[3,4,5]},
                      'binned':{'HARPSN-binned':[4,5,6]}} 
        #data_dic['Intr']['idx_in_bin']={'HARPS':{'2007-05-09':[1, 2, 7]},   #GJ436b,  expos faible contraste a mu < 0.47
        #             'HARPN':{'2016-03-18':[1, 2, 6, 7],                        
        #                        '2016-04-11':[1, 2, 6, 7]},
        #             'binned':{'HARPSN-binned':[2,3,7,8]}} 
        
        data_dic['Intr']['idx_in_bin']={'ESPRESSO':{'20190228':range(1,9),'20190429':range(1,9)}  }          

            # 'Intrbin':{   #GJ436b
            #     'idx_in_bin':{'ESPRESSO':{'20190228':range(1,9),'20190429':range(1,9)}},'dim_bin': 'r_proj',
            #     'ESPRESSO':{'binned':{'bin_range':[0.,1.],'nbins':1}},
            #     },  

    elif gen_dic['star_name']=='55Cnc':
        data_dic['Intr']['idx_in_bin']={
            'ESPRESSO':{
                '20200205':range(0,33),
                '20210121':range(1,33), #hors ingress
                '20210124':range(1,32), #hors egress
            },
            'HARPS':{
                '20120127':range(0,25),
                '20120213':range(0,28), 
                '20120227':range(1,26), 
                '20120315':range(1,25),
            },            
            }

    if gen_dic['star_name']=='HD209458':
        data_dic['Intr']['idx_in_bin']={'ESPRESSO':{'20190720':range(6,56),'20190911':range(6,56)}}    #for mask generation
        data_dic['Intr']['idx_in_bin']={'ESPRESSO':{'20190720':range(3,44),'20190911':range(3,44)}}    #for resampling the Na series 

    if gen_dic['star_name']=='WASP76':
        data_dic['Intr']['idx_in_bin']={'ESPRESSO':{'20180902':range(1,20),'20181030':range(3,35)}}    #for mask generation
        data_dic['Intr']['idx_in_bin']={'ESPRESSO':{'20180902':range(1,21),'20181030':range(1,38)}}    #for resampling
        data_dic['Intr']['idx_in_bin']={'ESPRESSO':{'20180902':list(np.delete(np.arange(21),[0,10,11,12,13,20])),'20181030':list(np.delete(np.arange(39),[0,1,18,19,20,21,22,23,24,25,37,38]))}}    #master local
        
        
    if gen_dic['star_name']=='HD3167': 
        data_dic['Intr']['idx_in_bin']['ESPRESSO']={'2019-10-09':range(1,16)}

        # data_dic['Intr']['idx_in_bin']={'HARPN':{'2016-10-01':range(19)}}     #Christiansen+Guilluy        
        # data_dic['Intr']['idx_in_bin']={'HARPN':{'2016-10-01':list(range(1,14))+[15,16,17]}}    #detected stellar line
        data_dic['Intr']['idx_in_bin']['HARPN']={'2016-10-01':range(17)}   
 
             # 'Intrbin':{   #HD3167c
            #     'idx_in_bin':{'HARPN':{'2016-10-01':range(19)}},'dim_bin': 'r_proj','HARPN':{'2016-10-01':{'bin_range':[0.,1.],'nbins':1}},
            #     },         
 
    elif 'Moon' in gen_dic['transit_pl']:
        data_dic['Intr']['idx_in_bin']={}    #{'HARPS':{'2019-07-02':range(19)}}        
        

    elif gen_dic['star_name']=='HIP41378':     
        data_dic['Intr']['idx_in_bin']={'HARPN':{'20191218':range(16)}  }    
    #RM survey
    elif gen_dic['star_name']=='HAT_P3':data_dic['Intr']['idx_in_bin']={'HARPN':{'20200130':range(1,8)}} 
    elif gen_dic['star_name']=='Kepler25':data_dic['Intr']['idx_in_bin']={'HARPN':{'20190614':range(1,19)}} 
    elif gen_dic['star_name']=='Kepler68':data_dic['Intr']['idx_in_bin']={'HARPN':{'20190803':[]}}     
    elif gen_dic['star_name']=='HAT_P33':data_dic['Intr']['idx_in_bin']={'HARPN':{'20191204':range(1,33)}}         
    elif gen_dic['star_name']=='K2_105':data_dic['Intr']['idx_in_bin']={'HARPN':{'20200118':[]}}  
    elif gen_dic['star_name']=='HD89345':data_dic['Intr']['idx_in_bin']={'HARPN':{'20200202':range(3,92)}}    
    elif gen_dic['star_name']=='Kepler63':data_dic['Intr']['idx_in_bin']={'HARPN':{'20200513':range(9)}}          
    elif gen_dic['star_name']=='HAT_P49':data_dic['Intr']['idx_in_bin']={'HARPN':{'20200730':range(3,71)}}         
    elif (gen_dic['star_name']=='WASP107') and (data_dic['Intr']['dim_bin']=='r_proj'):data_dic['Intr']['idx_in_bin']={'CARMENES_VIS':{'20180224':range(1,9)},'HARPS':{'20140406':range(1,11),'20180201':range(1,12),'20180313':range(1,12)}}        
    elif (gen_dic['star_name']=='WASP166') and (data_dic['Intr']['dim_bin']=='r_proj'):data_dic['Intr']['idx_in_bin']={'HARPS':{'20170114':range(1,39),'20170304':range(1,37),'20170315':range(1,33)}}        
    elif (gen_dic['star_name']=='HAT_P11') and (data_dic['Intr']['dim_bin']=='r_proj'):data_dic['Intr']['idx_in_bin']={'HARPN':{'20150913':range(2,26),'20151101':range(2,25)},'CARMENES_VIS':{'20170807':range(1,17),'20170812':range(2,18)}}  
    elif (gen_dic['star_name']=='HD106315') and (data_dic['Intr']['dim_bin']=='r_proj'):data_dic['Intr']['idx_in_bin']={'HARPS':{'20170309':range(3,44),'20170330':range(1,26),'20180323':range(1,27)}} 
    elif (gen_dic['star_name']=='WASP47') and (data_dic['Intr']['dim_bin']=='r_proj'):data_dic['Intr']['idx_in_bin']={'HARPN':{'20210730':range(13)}} 
    elif (gen_dic['star_name']=='WASP156') and (data_dic['Intr']['dim_bin']=='r_proj'):data_dic['Intr']['idx_in_bin']={'CARMENES_VIS':{'20190928':range(1,7),'20191025':range(1,6)}} 


        
    #Binning dimension
    data_dic['Intr']['dim_bin']='phase' 
    if gen_dic['star_name'] in ['HAT_P3','Kepler25','Kepler68','HAT_P33','K2_105','HD89345','Kepler63','HAT_P49','WASP47','WASP107','WASP166','HAT_P11','WASP156','HD106315']:
        data_dic['Intr']['dim_bin']='r_proj'
    if gen_dic['star_name'] in ['HD209458','WASP76']:
        data_dic['Intr']['dim_bin']='r_proj'    

    #Bin definition
    if gen_dic['star_name']=='WASP76':     #leave undefined for default value and CCF mask generation 
        data_dic['Intr']['prop_bin']={
   
    #         'ESPRESSO':{'2018-09-03':{'bin_low':np.arange(-0.03,0.03,0.01),'bin_high':np.arange(-0.03,0.03,0.01)+0.01},
    #                     '2018-10-31':{'bin_low':np.arange(0.,0.03,0.01),'bin_high':np.arange(0.,0.03,0.01)+0.01},
    #                     'binned':{'bin_low':np.arange(0.,0.03,0.01),'bin_high':np.arange(0.,0.03,0.01)+0.01}}
            

    #         # 'ESPRESSO':{'2018-09-03':{'bin_range':[-0.035,0.036],'nbins':3},
    #         #             '2018-10-31':{'bin_range':[0.,0.036],'nbins':3}} 
            
    #         #For master DI / intr comparison, phase
    #         #     # 'ESPRESSO':{'2018-09-03':{'bin_low':np.arange(-0.05,0.05,0.0125),'bin_high':np.arange(-0.05,0.05,0.0125)+0.0125},
    #         #     #             '2018-10-31':{'bin_low':np.arange(-0.05,0.05,0.0125),'bin_high':np.arange(-0.05,0.05,0.0125)+0.0125}}       
    #         #     # 'ESPRESSO':{'2018-09-03':{'bin_low':[-0.05,0.],'bin_high':[0.,0.05]},
    #         #     #             '2018-10-31':{'bin_low':[-0.05,0.],'bin_high':[0.,0.05]}}    
    #         #     # 'ESPRESSO':{'2018-09-03':{'bin_low':[-0.05],'bin_high':[0.05]},
    #         #     #             '2018-10-31':{'bin_low':[-0.05],'bin_high':[0.05]}}     
            
    #         #For master DI / intr comparison, xp_abs           
    #         #     # 'ESPRESSO':{'2018-09-03':{'bin_low':[0.],'bin_high':[1.]},'2018-10-31':{'bin_low':[0.],'bin_high':[1.]}} 
    #         #     'ESPRESSO':{'2018-09-03':{'bin_low':[0.,0.5],'bin_high':[0.5,1.]},'2018-10-31':{'bin_low':[0.,0.5],'bin_high':[0.5,1.]}}             
    #         }
    
            # #Bin of visit 2 for ANTARESS I, resampling section, r_proj
            # 'ESPRESSO':{'20180902':{'bin_low': [0.335,0.5715,0.7490,0.9425],'bin_high':[0.393,0.6300,0.8045,0.9700]},
            #             '20181030':{'bin_low': [0.335,0.5715,0.7490,0.9425],'bin_high':[0.393,0.6300,0.8045,0.9700]}}
            # }
            # 'ESPRESSO':{'20180902':{'bin_low': [0.335],'bin_high':[0.393]},
            #             '20181030':{'bin_low': [0.335],'bin_high':[0.393]}}
            # }
            'ESPRESSO':{'20180902':{'bin_range':[0.,1.],'nbins':1},'20181030':{'bin_range':[0.,1.],'nbins':1}}     #master local, r_proj
            }

    elif gen_dic['star_name']=='HD209458':     
        data_dic['Intr']['prop_bin']={
            #Bin of both visits, to illustrate change in line shape
            # 'ESPRESSO':{'binned':{'bin_low': [0.50,0.60,0.90],'bin_high':[0.60,0.90,1.00]}}}
            # 'ESPRESSO':{'binned':{'bin_low': [0.50,0.75],'bin_high':[0.75,1.00]}}}
            'ESPRESSO':{'binned':{'bin_low': [0.50,0.65],'bin_high':[0.65,1.00]}}}
            # 'ESPRESSO':{'binned':{'bin_low': [0.50,0.55,0.60,0.70,0.82],
            #                       'bin_high':[0.55,0.60,0.70,0.82,0.97]}}}
            
            

    if gen_dic['star_name']=='HD3167':  
        data_dic['Intr']['prop_bin']={
  
            # 'ESPRESSO':{'2019-10-09':{'bin_low':np.arange(-0.037,0.037,0.0125),'bin_high':np.arange(-0.037,0.037,0.0125)+0.0125}},
            
    
            # 'ESPRESSO':{'2019-10-09':{'bin_range':[-0.037,0.037],'nbins':1}}   #phase
            'ESPRESSO':{'2019-10-09':{'bin_range':[-0.05,0.05],'nbins':1}},   #phase
            # 'ESPRESSO':{'2019-10-09':{'bin_range':[-0.1,0.25],'nbins':1}}   #xp_abs
            # 'ESPRESSO':{'2019-10-09':{'bin_range':[0.3,1.],'nbins':1}}   #xp_abs
            # 'ESPRESSO':{'2019-10-09':{'bin_range':[0.,1.],'nbins':1}}     #r_proj, xp_abs 

            'HARPN':{'2016-10-01':{'bin_range':[-0.5,0.5],'nbins':1}}   #phase
            
            #     'ESPRESSO':{'2019-10-09':{'bin_range':[0,1],'nbins':1}},         'r_proj',        
            
            }

    elif gen_dic['transit_pl']=='GJ9827d':   
        data_dic['Intr']['prop_bin']={'HARPS':{'binned':{'bin_low':np.array([-0.0045,-0.0034,-0.0026,-0.0018,-0.0008,0.0001,0.001,0.002,0.0029,0.0038]),'bin_high':np.array([-0.0034,-0.0026,-0.0018,-0.0008,0.0001,0.001,0.002,0.0029,0.0038,0.0045])}}}
    elif gen_dic['transit_pl']=='GJ9827b':   
        data_dic['Intr']['prop_bin']={'HARPS':{'binned':{'bin_low':-0.03+0.003333*np.arange(18),'bin_high':-0.03+0.003333*np.arange(18) + 0.003333}}}
    
    
    if 'Moon' in gen_dic['transit_pl']:
        data_dic['Intr']['prop_bin']={

            'HARPS':{#'2019-07-02':{'bin_range':[-0.5,0.5],'nbins':1},
                     '2020-12-14':{'bin_range':[-0.5,0.5],'nbins':1}}   #phase
            }    
    
    if gen_dic['star_name']=='GJ436': 
        data_dic['Intr']['prop_bin']={'ESPRESSO':{'20190228':{'bin_range':[-0.5,0.5],'nbins':1},'20190429':{'bin_range':[-0.5,0.5],'nbins':1},
                                   'binned':{'bin_range':[-0.5,0.5],'nbins':1}}}    
        
    if gen_dic['star_name']=='HIP41378': 
        data_dic['Intr']['prop_bin']={'HARPN':{'20191218':{'bin_range':[-0.5,0.5],'nbins':1}}}       
    #RM survey
    elif gen_dic['star_name'] in ['HAT_P3','Kepler25','Kepler68','HAT_P33','K2_105','HD89345','Kepler63','HAT_P49','WASP47','WASP166','HAT_P11','WASP156','HD106315','WASP107']:
        gen_bin = {'bin_range':[0,1],'nbins':1}   #'r_proj'
        data_dic['Intr']['prop_bin'] = {}
        for inst in data_dic['Intr']['idx_in_bin']:
            data_dic['Intr']['prop_bin'][inst] = {}
            for vis in data_dic['Intr']['idx_in_bin'][inst]:data_dic['Intr']['prop_bin'][inst][vis] = gen_bin
            data_dic['Intr']['prop_bin'][inst]['binned'] = gen_bin

        if (gen_dic['star_name']=='HAT_P11') and (data_dic['Intr']['dim_bin']=='phase'):
            data_dic['Intr']['prop_bin']={'HARPN':{'binned':{'bin_low':np.array([-0.0105,-0.0097,-0.0089,-0.0082,-0.0074,-0.0066,-0.0059,-0.0051,-0.0043,-0.0035,-0.0028,-0.0020,-0.0012,-0.0005,0.0003,0.0011,0.0018,0.0026,0.0034,0.0042,0.0049,0.0057,0.0065,0.0072,0.0080,0.0088,0.0095,0.0103]),
                                                                'bin_high':np.array([-0.0097,-0.0089,-0.0082,-0.0074,-0.0067,-0.0059,-0.0051,-0.0044,-0.0036,-0.0028,-0.0020,-0.0012,-0.0005,0.0003,0.0010,0.0018,0.0026,0.0033,0.0041,0.0049,0.0057,0.0064,0.0072,0.0080,0.0087,0.0095,0.0103,0.0110])}},
                                          'CARMENES_VIS':{'binned':{'bin_low':np.array([-0.01095, -0.01 , -0.0085, -0.0073, -0.0062, -0.0052, -0.004  , -0.003  , -0.0019, -0.00085,  0.00025,  0.00135, 0.00245,  0.0035 ,  0.0046 ,  0.0056 ,  0.00675,  0.00775, 0.00895 , 0.01015]),
                                                                    'bin_high':np.array([-0.01   , -0.0085  , -0.0073, -0.0062, -0.0052 , -0.004,-0.003, -0.0019  , -0.00095,  0.0001 ,  0.00115,  0.0023 , 0.00335,  0.0045 ,  0.00555,  0.0066 ,  0.00775,  0.00875, 0.0099,0.01105])}}}

        if (gen_dic['star_name']=='HAT_P49') and (data_dic['Intr']['dim_bin']=='phase'):
            data_dic['Intr']['prop_bin']={'HARPN':{'20200730':{'bin_range':[-0.035,0.035],'nbins':23}}}  
            data_dic['Intr']['prop_bin']={'HARPN':{'20200730':{'bin_range':[-0.029,0.029],'nbins':20}}}              
                                      
        if (gen_dic['star_name']=='HD106315') and (data_dic['Intr']['dim_bin']=='phase'):
            data_dic['Intr']['prop_bin']={'HARPS':{'binned':{'bin_low':np.array([-0.0047,-0.0043,-0.0040,-0.0036,-0.0033,-0.0029,-0.0026,-0.0022,-0.0019,-0.0016,-0.0012,-0.0009,-0.0005,-0.0002,0.0002,0.0005,0.0009,0.0012,0.0016,0.0019,0.0023,0.0026,0.0030,0.0033,0.0036,0.0040,0.0043]),
                                                                'bin_high':np.array([-0.0044,-0.0040,-0.0037,-0.0033,-0.0030,-0.0026,-0.0023,-0.0019,-0.0016,-0.0012,-0.0009,-0.0005,-0.0002,0.0002,0.0005,0.0009,0.0012,0.0015,0.0019,0.0022,0.0026,0.0029,0.0033,0.0036,0.0040,0.0043,0.0047])}}}

        # if gen_dic['star_name']=='WASP107':      #Plot des variations de contraste
        #     if (data_dic['Intr']['dim_bin']=='r_proj'):
        #         gen_bin = {'bin_low':np.array([0.1,0.5]),'bin_high':np.array([0.5,1.]) }  #'r_proj'          
        #         data_dic['Intr']['prop_bin'] = {'CARMENES_VIS':{'20180224':gen_bin},'HARPS':{'binned':gen_bin}}
        #         for vis in data_dic['Intr']['idx_in_bin']['HARPS']:data_dic['Intr']['prop_bin']['HARPS'][vis] = gen_bin    

        #     if (data_dic['Intr']['dim_bin']=='phase'):
        #         data_dic['Intr']['prop_bin']={'HARPS':{'binned':{'bin_low':np.array([-0.0107,-0.0090,-0.0074,-0.0057,-0.0040,-0.0023,-0.0007,0.0010,0.0027,0.0044,0.0061,0.0076,0.0092]),
        #                                                          'bin_high':np.array([-0.0091,-0.0074,-0.0058,-0.0041,-0.0024,-0.0007,0.0010,0.0026,0.0044,0.0060,0.0076,0.0092,0.0107])}}}

        if (gen_dic['star_name']=='WASP166') and (data_dic['Intr']['dim_bin']=='phase'):
            # data_dic['Intr']['prop_bin']={'HARPS':{'binned':{'bin_low':np.array([-0.0141,-0.0133,-0.0126,-0.0119,-0.0113,-0.0105,-0.0098,-0.0091,-0.0084,-0.0077,-0.0069,-0.0063,-0.0056,-0.0049,-0.0041,-0.0034,-0.0028,-0.0020,-0.0013,-0.0006,0.0001,0.0008,0.0016,0.0022,0.0029,0.0036,0.0043,0.0050,0.0058,0.0065,0.0072,0.0079,0.0086,0.0093,0.0100,0.0107,0.0114,0.0121,0.0128,0.0135])-0.5e-4,
            #                                                  'bin_high':np.array([-0.0134,-0.0127,-0.0120,-0.0113,-0.0106,-0.0099,-0.0091,-0.0084,-0.0077,-0.0070,-0.0063,-0.0056,-0.0049,-0.0042,-0.0035,-0.0028,-0.0021,-0.0014,-0.0007,0.0000,0.0007,0.0015,0.0022,0.0029,0.0036,0.0043,0.0050,0.0057,0.0064,0.0071,0.0078,0.0085,0.0092,0.0099,0.0106,0.0113,0.0120,0.0127,0.0134,0.0142])-0.5e-4}}}

            #Les bornes en phase ne sont pas importantes puisque les bins sont limites aux expos in-tr
            data_dic['Intr']['prop_bin']={'HARPS':{'20170114':{'bin_range':[-0.02809  ,  0.02262],'nbins':20},
                                                   '20170304':{'bin_range':[-0.01739  ,  0.02076],'nbins':20},
                                                   '20170315':{'bin_range':[-0.01885  ,  0.03472],'nbins':20}}}

    if gen_dic['star_name']=='55Cnc': 
        data_dic['Intr']['prop_bin']={
            'ESPRESSO':{
                '20200205':{'bin_range':[-0.5,0.5],'nbins':1},
                '20210121':{'bin_range':[-0.5,0.5],'nbins':1},
                '20210124':{'bin_range':[-0.5,0.5],'nbins':1},     
                # 'binned':{'bin_range':[-0.5,0.5],'nbins':1},   
                'binned':{'bin_range':[-0.043,0.043],'nbins':32},
            },
            'HARPS':{
                'binned':{'bin_range':[-0.043,0.043],'nbins':26}}           
            }      
        

    #2D maps of binned profiles
    plot_dic['map_Intrbin']=''   #'png 
    
    
    #Individual binned profiles
    plot_dic['sp_Intrbin']=''  
    plot_dic['CCF_Intrbin']=''   #pdf


    #Residuals from binned profiles
    plot_dic['CCF_Intrbin_res']=''  # pdf


    #Binned disk-integrated and intrinsic profiles comparison
    plot_dic['binned_DI_Intr']=''   #pdf 








        


##################################################################################################
#%%% Module: intrinsic CCF masks
#    - spectra must have been aligned in the star rest frame, converted into a 1D profile, and binned
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


if __name__ == '__main__':
    
    #Activating
    gen_dic['def_Intrmasks'] = True    &  False

    #Estimate of line width 
    data_dic['Intr']['mask']['fwhm_ccf'] = 5. 
    if gen_dic['star_name']=='HD209458':
        data_dic['Intr']['mask']['fwhm_ccf'] = 8.5 

    # #Oversampling 
    # if gen_dic['star_name']=='HD209458':
    #     data_dic['Intr']['mask']['dw_reg'] = {'ESPRESSO':0.008}

    #Smoothing window   
    if gen_dic['star_name']=='HD209458':
        data_dic['Intr']['mask']['kernel_smooth']={'ESPRESSO':30}
        data_dic['Intr']['mask']['kernel_smooth_deriv2']={'ESPRESSO':30}

    #Line selection: rejection ranges
    if gen_dic['star_name']=='HD209458':
        data_dic['Intr']['mask']['line_rej_range'] = {'ESPRESSO':[[3000.,3800.],[6276.,6316.],[6930.,6957.],[7705.,7717.]]} 

    #Line selection: RV dispersion 
    data_dic['Intr']['mask']['RV_disp_sel'] = False


    
    
    
    
    
    
    
    




    

##################################################################################################
#%%% Module: analyzing intrinsic profiles
#    - can be applied to:
# + 'fit_Intr': profiles in the star rest frame, original exposures, for all formats
# + 'fit_Intr_1D': profiles in the star or surface (if aligned) rest frame, original exposures, converted from 2D->1D 
# + 'fit_Intrbin' : profiles in the star or surface (if aligned) rest frame, binned exposures, all formats
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

#%%%%% Constant data errors
#    - set to the mean error over the continuum
data_dic['Intr']['cst_err']= False
data_dic['Intr']['cst_errbin']= False


#%%%%% Scaled data errors
data_dic['Intr']['sc_err']={}


#%%%%% Trimming
data_dic['Intr']['fit_prof']['trim_range']={}


#%%%%% Order to be fitted
#    - relevant for 2D spectra only
data_dic['Intr']['fit_prof']['order']={}     

#%%%%% Spectral range(s) to be fitted
#   - leave empty to fit over the entire range of definition of the CCF
#   - otherwise, define [ [rv1,rv2] , [rv3,rv4] , [rv5,rv6] , ... ] with rv defined in the star velocity rest frame
#     this can be used to avoid sidelobe patterns of M dwarf CCF, not reproduced by a gaussian model
data_dic['Intr']['fit_range']={}


#%%%% Direct measurements
#    - same as data_dic['DI']['meas_prop']={}
data_dic['Intr']['meas_prop']={}


#%%%% Line profile model

#%%%%% Transition wavelength
#    - in the star rest frame
#    - used to center the line analytical model
#    - only relevant in spectral mode
#    - do not use if the spectral fit is performed on more than a single line
data_dic['Intr']['line_trans']=None
   

#%%%%% Instrumental convolution
#    - apply instrumental convolution or not (default) to model
#    - beware that most derived properties will correspond to the model before convolution
#      this is particularly useful to match the intrinsic line properties from the joint intrinsic fit with values derived here from the individual fits
data_dic['Intr']['conv_model']=False  


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

#%%%%% Fitting mode 
#    - chi2 or MCMC
data_dic['Intr']['fit_mod']=''


#%%%%% Printing fits results
data_dic['Intr']['verbose'] = False  


#%%%%% Priors on variable properties
#    - the width of the master disk-integrated profile can be used as upper limit
data_dic['Intr']['line_fit_priors']={}
             
    
#%%%%% Derived properties
data_dic['Intr']['deriv_prop'] = []

  
#%%%%% Detection thresholds    
#    - define area and amplitude thresholds for detection of stellar line (in sigma)
#    - for the amplitude, it might be more relevant to consider the actual SNR of the derived value (shown in plots)
#    - if set to None, lines are considered as detected in all exposures
data_dic['Intr']['thresh_area']=5.
data_dic['Intr']['thresh_amp']=4.   


#%%%%% Force detection flag
#    - set flag to True at relevant index for the local CCFs to be considered detected, or false to force a non-detection
#    - indices for each dataset are relative to in-transit indices (binned if relevant)
#    - leave empty for automatic detection
data_dic['Intr']['idx_force_det']={}
data_dic['Intr']['idx_force_detbin']={} 
data_dic['Intr']['idx_force_detbinmultivis']={} 


#%%%%% MCMC settings

#%%%%%% Calculating/retrieving
#    - set to 'reuse' if gen_dic['calc_fit_intr']=True, allow changing nburn and error definitions without running the mcmc again
data_dic['Intr']['mcmc_run_mode']='use'


#%%%%%% Walkers
#    - settings per instrument & visit
data_dic['Intr']['mcmc_set']={}


#%%%%%% Complex priors
#    - to be defined manually within the code
#    - leave empty, or put in field for each priors and corresponding options
data_dic['Intr']['prior_func']={}


#%%%%%% Walkers exclusion
#    - automatic exclusion of outlying chains
#    - set to None, or exclusion threshold
data_dic['Intr']['exclu_walk_autom']=None  


#%%%%%% Derived errors
#    - 'quant' or 'HDI'
#    - if 'HDI' is selected:
# + by default a smoothed density profile is used to define HDI intervals
# + multiple HDI intervals can be avoided by defining the density profile as a histogram (by setting its resolution 'HDI_dbins') or by 
# defining the bandwith factor of the smoothed profile ('HDI_bw')
data_dic['Intr']['out_err_mode']='HDI'


#%%%% Plot settings

#%%%%% 1D PDF from mcmc
plot_dic['prop_Intr_mcmc_PDFs']=''     


#%%%%% Derived properties
#    - from original or binned data
plot_dic['prop_Intr']=''  



if __name__ == '__main__':

    #Activating
    gen_dic['fit_Intr'] = True   &  False
    gen_dic['fit_Intr_1D'] = True   &  False
    gen_dic['fit_Intrbin']=True     &  False
    gen_dic['fit_Intrbinmultivis']=True     &  False

    #Calculating/Retrieving
    gen_dic['calc_fit_Intr']=True #  &  False   
    gen_dic['calc_fit_Intr_1D']=True #  &  False   
    gen_dic['calc_fit_Intrbin']=True  #  &  False    
    gen_dic['calc_fit_Intrbinmultivis']=True  #  &  False  

    #Constant data errors
    data_dic['Intr']['cst_err']=True   &  False
    data_dic['Intr']['cst_errbin']=True   &  False

    #Spectral range(s) to be fitted
    if gen_dic['transit_pl']=='WASP_8b':
        data_dic['Intr']['fit_range']=[[-50.,50.]]
    elif gen_dic['star_name']=='GJ436':
        data_dic['Intr']['fit_range']=[]
        data_dic['Intr']['fit_range']=[[-30.,-13.5],[-2.,2.],[13.5,30.]]
        data_dic['Intr']['fit_range']=[[-50.,-3.5],[8.,12.],[23.,50.]]
        data_dic['Intr']['fit_range']=[[-26.,26.]]
        data_dic['Intr']['fit_range']=[[-50.,50.]]        

    elif gen_dic['transit_pl']=='WASP121b':
        data_dic['Intr']['fit_range']=[[-40.,40.]]     #mask G
        data_dic['Intr']['fit_range']=[[-80.,80.]]   #mask F 
#        data_dic['Intr']['fit_range']=[[-300.,300.]]   #mask F + atmo 
        data_dic['Intr']['fit_range']=[[-50.,50.]]    #mask G 
        data_dic['Intr']['fit_range']=[[-90.,90.]]   #mask F        
    elif gen_dic['transit_pl']=='Kelt9b':
        data_dic['Intr']['fit_range']=[[-300.,300.]]
    if gen_dic['transit_pl']=='WASP127b':
        data_dic['Intr']['fit_range']=[[-150.,150.]]   
    if gen_dic['star_name']=='HD3167': 
        data_dic['Intr']['fit_range']=[[-26.,26.]]
        data_dic['Intr']['fit_range']=[[-50.,50.]]
    elif gen_dic['transit_pl']=='Nu2Lupi_c':data_dic['Intr']['fit_range']=[[-150.,150.]]
    elif gen_dic['transit_pl']=='GJ9827d':data_dic['Intr']['fit_range']=[[-150.,150.]] 
    elif 'Moon' in gen_dic['transit_pl']:
        data_dic['Intr']['fit_range']=[[-26.,26.]]
        data_dic['Intr']['fit_range']=[[-50.,50.]]    
    elif 'TOI858b' in gen_dic['transit_pl']:
        data_dic['Intr']['fit_range']=[[-50.,50.]]   
    elif 'HIP41378d' in gen_dic['transit_pl']:
        data_dic['Intr']['fit_range']['HARPN']={'20191218':[[-50.,50.]],'20220401': [[-50.,50.]]}
    elif gen_dic['star_name']=='MASCARA1':data_dic['Intr']['fit_range']=[[-130.,130.]] 
    elif gen_dic['star_name']=='V1298tau':data_dic['Intr']['fit_range']['HARPN']={'mock_vis' : [[-130.,130.]] }

    #RM survey             
    elif gen_dic['star_name']=='HAT_P3':
        fit_range = [[-70.,70.]]
        data_dic['Intr']['fit_range']['HARPN']={'20190415':fit_range,'20200130':fit_range}
    elif gen_dic['star_name']=='Kepler25':data_dic['Intr']['fit_range']['HARPN']={'20190614':[[-90.,90.]]}
    elif gen_dic['star_name']=='Kepler68':data_dic['Intr']['fit_range']['HARPN']={'20190803':[[-80.,80.]]}
    elif gen_dic['star_name']=='HAT_P33':data_dic['Intr']['fit_range']['HARPN']={'20191204':[[-120.,120.]]}     
    elif gen_dic['star_name']=='K2_105':data_dic['Intr']['fit_range']['HARPN']={'20200118':[[-65.,65.]]}     
    elif gen_dic['star_name']=='HD89345':data_dic['Intr']['fit_range']['HARPN']={'20200202':[[-100.,100.]]}     
    elif gen_dic['star_name']=='Kepler63':data_dic['Intr']['fit_range']['HARPN']={'20200513':[[-80.,80.]]}     
    elif gen_dic['star_name']=='HAT_P49':data_dic['Intr']['fit_range']['HARPN']={'20200730':[[-100.,100.]]}   
    elif gen_dic['star_name']=='WASP47':data_dic['Intr']['fit_range']['HARPN']={'20210730':[[-80.,80.]]}     
    elif gen_dic['star_name']=='WASP107':
        fit_range =[[-80.,80.]]
        data_dic['Intr']['fit_range']['HARPS']={'20140406':fit_range,'20180201':fit_range,'20180313':fit_range,'binned':fit_range}
        data_dic['Intr']['fit_range']['CARMENES_VIS'] ={'20180224': [[-110.,110.]]}  
    elif gen_dic['star_name']=='WASP166':
        fit_range = [[-75.,75.]]
        data_dic['Intr']['fit_range']['HARPS']={'20170114':fit_range,'20170304':fit_range,'20170315':fit_range,'binned':fit_range}
    elif gen_dic['star_name']=='HAT_P11':
        fit_range = [[-100.,40.]]
        data_dic['Intr']['fit_range']['HARPN']={ '20150913':fit_range,'20151101':fit_range,'binned':fit_range}
        fit_range =  [[-110.,110.]]
        data_dic['Intr']['fit_range']['CARMENES_VIS'] ={ '20170807':fit_range,'20170812':fit_range,'binned':fit_range}
    elif gen_dic['star_name']=='WASP156':
        fit_range = [[-110.,110.]]
        data_dic['Intr']['fit_range']['CARMENES_VIS']={'20190928':fit_range,'20191025':fit_range}
        #Excluding contaminated bands
        data_dic['Intr']['fit_range']['CARMENES_VIS']={'20190928':[[-110.,13.],[30.,110.]],'20191025':[[-110.,-30.],[-10.,10.],[30.,110.]]}
        data_dic['Intr']['fit_range']['CARMENES_VIS']={'20190928':fit_range,'20191025':[[-110.,-30.],[-10.,10.],[30.,110.]]}    #No need for V1 with Mout post-tr
    elif gen_dic['star_name']=='HD106315':
        fit_range = [[-140.,140.]] 
        fit_range = [[-120.,120.]] 
        data_dic['Intr']['fit_range']['HARPS']={'20170309':fit_range,'20170330':fit_range,'20180323':fit_range,'binned':fit_range}
    elif gen_dic['star_name']=='55Cnc':
        fit_range = [[-80.,80.]] 
        data_dic['Intr']['fit_range']['ESPRESSO']={'20200205':fit_range,'20210121':fit_range,'20210124':fit_range,'binned':fit_range}
    if gen_dic['star_name']=='WASP76':
        fit_range=[[-150.,150.]]   
        # fit_range=[[-100.,100.]]      #[[-80.,80.]]    
        data_dic['Intr']['fit_range']['ESPRESSO']={'20180902':fit_range,'20181030':fit_range}        
    if gen_dic['star_name']=='HD209458':
        fit_range = [[-80.,80.]]
        # fit_range = [[-100.,100.]]   #CCF Na
        data_dic['Intr']['fit_range']['ESPRESSO']={'20190720':fit_range,'20190911':fit_range}

    #Model type  
    if gen_dic['transit_pl']=='WASP_8b':data_dic['Intr']['model']='gauss'
    elif gen_dic['star_name']=='GJ436':data_dic['Intr']['model']='dgauss'
    elif gen_dic['transit_pl']=='WASP121b':
        data_dic['Intr']['model']='gauss'        #mask G
        data_dic['Intr']['model']='dgauss'    #mask F
    elif gen_dic['transit_pl']=='Kelt9b':data_dic['Intr']['model']='gauss'
    elif gen_dic['transit_pl']=='WASP76b':data_dic['Intr']['model']='gauss'      
    elif gen_dic['transit_pl'] in ['Corot7b','Nu2Lupi_c','GJ9827d']:
        data_dic['Intr']['model']='gauss'
    elif gen_dic['star_name']=='HIP41378':data_dic['Intr']['model']['HARPN']='gauss'
    #RM survey
    elif gen_dic['star_name'] in ['HAT_P3','Kepler25','Kepler68','HAT_P33','K2_105','HD89345','HAT_P49','Kepler63','WASP47']:
        data_dic['Intr']['model']['HARPN']='gauss'
    elif gen_dic['star_name'] in ['WASP107']:
        data_dic['Intr']['model']['HARPS']='gauss'
        data_dic['Intr']['model']['CARMENES_VIS']='gauss'
        data_dic['Intr']['model']['CARMENES_VIS']='voigt'     #too imprecise for individual fit, used here for the master Intr        
    elif gen_dic['star_name'] in ['WASP166','HD106315']:    
        data_dic['Intr']['model']['HARPS']='gauss'
    elif gen_dic['star_name'] in ['HAT_P11']:
        data_dic['Intr']['model']['HARPN']='gauss'
        data_dic['Intr']['model']['CARMENES_VIS']='gauss'  
        data_dic['Intr']['model']['CARMENES_VIS']='voigt'     #too imprecise for individual fit, used here for the master Intr         
    elif gen_dic['star_name'] in ['WASP156']:
        data_dic['Intr']['model']['CARMENES_VIS']='gauss'          
    elif gen_dic['star_name']=='55Cnc':
        data_dic['Intr']['model']['ESPRESSO']='gauss'      
        
    #Intrinsic line properties
    if gen_dic['star_name']=='HD209458':
        data_dic['Intr']['mod_def']={}       ### utiliser pour le fit theo Na    


    #Fixed/variable properties
    if (gen_dic['star_name']=='GJ436') and gen_dic['fit_Intr']:
        data_dic['Intr']['mod_prop']={}
        #data_dic['Intr']['mod_prop']={'HARPN':{'2016-03-18':{'RV_l2c':0.,'amp_l2c':0.5186,'FWHM_l2c':1.830},          #GJ436b, avec correction de couleur
        #                            '2016-04-11':{'RV_l2c':0.,'amp_l2c':0.5180,'FWHM_l2c':1.831}},
        #                 'HARPS':{'2007-05-09':{'RV_l2c':0.,'amp_l2c':0.5357,'FWHM_l2c':1.808}},
        #                 'binned':{'2016-03-18-binned':{'RV_l2c':0.,'amp_l2c':0.5186,'FWHM_l2c':1.830},          #memes nuits avec expos binnees
        #                           '2016-04-11-binned':{'RV_l2c':0.,'amp_l2c':0.5180,'FWHM_l2c':1.831},
        #                           '2007-05-09-binned':{'RV_l2c':0.,'amp_l2c':0.5357,'FWHM_l2c':1.808},
        #                           'HARPSN-binned':{'RV_l2c':0.,'amp_l2c':0.5183,'FWHM_l2c':1.8308}}}
        # data_dic['Intr']['mod_prop']={'HARPN':{'2016-03-18':{'RV_l2c': 1.062e-2,'amp_l2c':0.5183,'FWHM_l2c':1.831},          #GJ436b, avec correction de couleur, et redshift
        #                             '2016-04-11':{'RV_l2c':1.094e-2,'amp_l2c':0.5177,'FWHM_l2c':1.832}},
        #                  'HARPS':{'2007-05-09':{'RV_l2c':1.692e-2,'amp_l2c':0.5354,'FWHM_l2c':1.809}},
        #                  'binned':{'HARPSN-binned':{'RV_l2c':1.0839e-2,'amp_l2c':0.5180,'FWHM_l2c':1.8318}}}

        # data_dic['Intr']['mod_prop']={'ESPRESSO':{'2019-02-28':{'RV_l2c':-2.44899e-03,'amp_l2c':9.83824e-01,'FWHM_l2c':1.01036},    #modeles fixes a la moyenne des fits individuels de chaque visite
        #                                       '2019-04-29':{'RV_l2c':-3.61183e-02,'amp_l2c':9.31364e-01,'FWHM_l2c':1.07825}}}

        # data_dic['Intr']['mod_prop']={'ESPRESSO':{'2019-02-28':{'RV_l2c':-2.44899e-03,'amp_l2c':9.83824e-01,'FWHM_l2c':1.01036},    #modeles fixes a la moyenne des fits individuels de la 1ere visite
        #                                       '2019-04-29':{'RV_l2c':-2.44899e-03,'amp_l2c':9.83824e-01,'FWHM_l2c':1.01036}}}

        # data_dic['Intr']['mod_prop']={'ESPRESSO':{'2019-02-28':{'RV_l2c':-6.95668e-04,'amp_l2c':9.95784e-01,'FWHM_l2c':1.00268},    #modeles des CCFs out, final
        #                                     '2019-04-29':{'RV_l2c':-6.90909e-04,'amp_l2c':9.95757e-01,'FWHM_l2c':1.00269}}}

        # #Prop. moyennes sur fit CCFs DI out, Erreur propagee, chi2, skycorr
        # data_dic['Intr']['mod_prop']={'RV_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':-1.07515e-03},'20190429':{'guess':-8.94843e-04},'binned':{'guess':0.5*(-1.07515e-03-8.94843e-04)}}},
        #                           'amp_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':9.76438e-01},'20190429':{'guess':9.80422e-01},'binned':{'guess':0.5*(9.76438e-01+9.80422e-01)}}},
        #                           'FWHM_l2c':{'vary':False,'ESPRESSO':{'20190228':{'guess':1.01636e+00},'20190429':{'guess':1.01356e+00},'binned':{'guess':0.5*(1.01636e+00+1.01356e+00)}}}}      


        #Core/lobe properties fixed to those of DI master
        for key in ['RV_l2c','amp_l2c','FWHM_l2c']:data_dic['Intr']['mod_prop'][key] = deepcopy(data_dic['DI']['mod_prop'][key])     

        data_dic['Intr']['mod_prop'].update({'rv':{'vary':True,'ESPRESSO':{'20190228':{'guess':0.2,'bd':[0.,0.5]},'20190429':{'guess':0.2,'bd':[0.,0.5]},'binned':{'guess':0.2,'bd':[0.,0.5]}}},
                                          # 'amp':{'vary':True,'ESPRESSO':{'20190228':{'guess':-10.,'bd':[-20.,0.]},'20190429':{'guess':-10.,'bd':[-20.,0.]}}},
                                          'ctrst':{'vary':True,'ESPRESSO':{'20190228':{'guess':0.3,'bd':[0.2,0.4]},'20190429':{'guess':0.3,'bd':[0.2,0.4]},'binned':{'guess':0.3,'bd':[0.2,0.4]}}},
                                          'FWHM':{'vary':True,'ESPRESSO':{'20190228':{'guess':5.,'bd':[4.,6.]},'20190429':{'guess':5.,'bd':[4.,6.]},'binned':{'guess':5.,'bd':[4.,6.]}}}})
        data_dic['Intr']['mod_prop']['rv'].update({'HARPN':{'20160318':{'guess':0.2,'bd':[0.,0.5]},'20160411':{'guess':0.2,'bd':[0.,0.5]}},'HARPS':{'20070509':{'guess':0.2,'bd':[0.,0.5]}}})
        data_dic['Intr']['mod_prop']['ctrst'].update({'HARPN':{'20160318':{'guess':0.3,'bd':[0.2,0.4]},'20160411':{'guess':0.3,'bd':[0.2,0.4]}},'HARPS':{'20070509':{'guess':0.3,'bd':[0.2,0.4]}}})
        data_dic['Intr']['mod_prop']['FWHM'].update({'HARPN':{'20160318':{'guess':5.,'bd':[4.,6.]},'20160411':{'guess':5.,'bd':[4.,6.]}},'HARPS':{'20070509':{'guess':5.,'bd':[4.,6.]}}})   

    elif gen_dic['transit_pl']=='WASP121b':
        if data_dic['Intr']['model']=='dgauss':
    #        data_dic['Intr']['mod_prop']={}
    #        data_dic['Intr']['mod_prop']=deepcopy(data_dic['DI']['mod_prop'])     #Fit preliminaire pour recentrer les RV locales et construire le master local
    
            data_dic['Intr']['mod_prop']={                                   #Fit libre du master local full
                    'HARPS':{'09-01-18':{'RV_l2c':4.6343,'amp_l2c':0.1214,'FWHM_l2c':3.8269},    #green
                             '14-01-18':{'RV_l2c':4.4362,'amp_l2c':0.1009,'FWHM_l2c':3.7155},    #blue
                             '31-12-17':{'RV_l2c':2.0810,'amp_l2c':0.1263,'FWHM_l2c':3.7511}},   #red  
    #                         '31-12-17':{'RV_l2c':-0.2545,'amp_l2c':0.1284,'FWHM_l2c':3.6653}},   #red preTR 
    #                         '31-12-17':{'RV_l2c':9.6878,'amp_l2c':0.1186,'FWHM_l2c':4.1132}},   #red  postTR    
                    'binned':{'HARPS-binned':{'RV_l2c':3.4053,'amp_l2c':0.1102,'FWHM_l2c':3.8881}}} 
    
    #        data_dic['Intr']['mod_prop']={                                   #Fit libre du master local pre-TR
    #                'HARPS':{'09-01-18':{'RV_l2c':4.9195,'amp_l2c':0.1266,'FWHM_l2c':3.3949},    #green
    #                         '14-01-18':{'RV_l2c':4.3079,'amp_l2c':0.0973,'FWHM_l2c':3.6294},    #blue
    #                         '31-12-17':{'RV_l2c':-0.2545,'amp_l2c':0.1284,'FWHM_l2c':3.6653}},   #red    
    #                'binned':{'HARPS-binned':{'RV_l2c':2.1808,'amp_l2c':0.1133,'FWHM_l2c':3.6196}}} 
    
    #        data_dic['Intr']['mod_prop']={                                   #Fit libre du master local post-TR
    #                'HARPS':{'09-01-18':{'RV_l2c':4.4026,'amp_l2c':0.1198,'FWHM_l2c':3.9650},    #green
    #                         '14-01-18':{'RV_l2c':5.2218,'amp_l2c':0.1081,'FWHM_l2c':3.9938},    #blue
    #                         '31-12-17':{'RV_l2c':9.6878,'amp_l2c':0.1186,'FWHM_l2c':4.1132}},   #red    
    #                'binned':{'HARPS-binned':{'RV_l2c':6.4041,'amp_l2c':0.1099,'FWHM_l2c':4.1094}}} 


#            #We fix the width and contrast of the model when the planetary range is excluded
#            #    - contrast is not a fit parameter, but the amplitude varies even when the contrast does not because of the continuum
#            #    - the routine has been modified to take into account this constraint
#            data_dic['Intr']['mod_prop']['binned']['HARPS-binned'].update({'FWHM':5.553632264814631,'ctrst':0.4199})
    
    elif gen_dic['star_name']=='WASP76': 
        data_dic['Intr']['conv_model']=True  
        data_dic['Intr']['mod_prop']={
            'rv':{'vary':True ,'ESPRESSO':{'20180902':{'guess':0.,'bd':[-2.,2.]},'20181030':{'guess':0.,'bd':[-2.,2.]}}},
            'FWHM':{'vary':True ,'ESPRESSO':{'20180902':{'guess':7.,'bd':[7.,10.]},'20181030':{'guess':7.,'bd':[7.,10.]}}},
            'ctrst':{'vary':True ,'ESPRESSO':{'20180902':{'guess':0.6,'bd':[0.55,0.65]},'20181030':{'guess':0.6,'bd':[0.55,0.65]}}},                
                }
    elif gen_dic['star_name']=='HD209458': 
        data_dic['Intr']['conv_model']=True  
        data_dic['Intr']['mod_prop']={
            # 'cont':{'vary':True,'ESPRESSO':{'20190720':{'guess':13.1,'bd':[13.05,13.15]},'20190911':{'guess':13.1,'bd':[13.05,13.15]}}},
            'rv':{'vary':True ,'ESPRESSO':{'20190720':{'guess':0.,'bd':[-5.,5.]},'20190911':{'guess':0.,'bd':[-5.,5.]}}},
            'FWHM':{'vary':True ,'ESPRESSO':{'20190720':{'guess':7.,'bd':[6.5,8.5]},'20190911':{'guess':7.,'bd':[6.5,8.5]}}},
            'ctrst':{'vary':True ,'ESPRESSO':{'20190720':{'guess':0.6,'bd':[0.55,0.65]},'20190911':{'guess':0.6,'bd':[0.55,0.65]}}},                
                }
            
    elif gen_dic['transit_pl']=='WASP121b':
        if data_dic['Intr']['model']=='dgauss':
    #        data_dic['Intr']['mod_prop']={}       #pour ajuster librement le Master local et definir ses proprietes (tout en fixant le modele des CCFs locales pour les recentrer)
            data_dic['Intr']['mod_prop']=deepcopy(data_dic['Intr']['mod_prop'])      #pour deriver les proprietes locales
#            data_dic['Intr']['mod_prop']={'binned':{'HARPS-binned':{'RV_l2c':3.4053,'amp_l2c':0.1102,'FWHM_l2c':3.8881}}}   #avec atmo, pour laisser libre amp et sig     


    if gen_dic['star_name']=='HD3167': 
        data_dic['Intr']['mod_prop']={'rv':{'vary':True,
                                         'ESPRESSO':{'2019-10-09':{'guess':0.,'bd':[-5.,5.]}},
                                         'HARPN':{'2016-10-01':{'guess':0.,'bd':[-5.,5.]}}},
                                    'amp':{'vary':True,
                                           'ESPRESSO':{'2019-10-09':{'guess':-0.8,'bd':[-1.,0.]}},
                                           'HARPN':{'2016-10-01':{'guess':-0.8,'bd':[-1.,0.]}}},
                                   'FWHM':{'vary':True,
                                           'ESPRESSO':{'2019-10-09':{'guess':7.,'bd':[0.,15.]}},
                                           'HARPN':{'2016-10-01':{'guess':7.,'bd':[0.,15.]}}}}

    if gen_dic['star_name']=='TOI858': 
        data_dic['Intr']['mod_prop']={'rv':{'vary':True,'CORALIE':{'20191205':{'guess':3.,'bd':[0.,5.]},'20210118':{'guess':3.,'bd':[0.,5.]}}},
                                    'amp':{'vary':True,'CORALIE':{'20191205':{'guess':-0.6,'bd':[-1.,0.]},'20210118':{'guess':-0.6,'bd':[-1.,0.]}}},
                                   'FWHM':{'vary':True,'CORALIE':{'20191205':{'guess':6.,'bd':[0.,15.]},'20210118':{'guess':6.,'bd':[0.,15.]}}}}

    if gen_dic['star_name']=='HIP41378': 
        data_dic['Intr']['mod_prop']={'rv':{'vary':True,'HARPN':{'20191218':{'guess':0.,'bd':[-10.,10.]},'20220401':{'guess':0.,'bd':[-10.,10.]}}},
                                    'ctrst':{'vary':True,'HARPN':{'20191218':{'guess':0.6,'bd':[0.,1.]},'20220401':{'guess':0.6,'bd':[0.,1.]}}},
                                   'FWHM':{'vary':True,'HARPN':{'20191218':{'guess':7.,'bd':[0.,15.]},'20220401':{'guess':7.,'bd':[0.,5.]}}}}
        data_dic['Intr']['mod_prop'].update({'slope':{'vary':True & False ,'HARPN':{'20191218':{'guess':0.,'bd':[-0.1,0.1]},'20220401':{'guess':-2e-8,'bd':[-0.1,0.1]}}}})
        # data_dic['Intr']['mod_prop'].update({'cont':{'vary':True & False ,'HARPN':{'20191218':{'guess':0.,'bd':[0.9,1.1]},'20220401':{'guess':1.,'bd':[0.9,1.1]}}}})

        #Master-in, from props derived through joined fit
        data_dic['Intr']['conv_model']=True    
        data_dic['Intr']['mod_prop']={'rv':{'vary':False,'HARPN':{'20191218':{'guess':0.,'bd':[-10.,10.]}}},
                                     'ctrst':{'vary':False,'HARPN':{'20191218':{'guess':5.8668406446e-01,'bd':[0.,1.]}}},
                                     'FWHM':{'vary':False,'HARPN':{'20191218':{'guess':3.7484355842e+00,'bd':[0.,15.]}}}}



    elif gen_dic['star_name']=='V1298tau':
        data_dic['Intr']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20200128':{'guess':0.,'bd':[-10.,10.]},'20201207':{'guess':0.,'bd':[-10.,10.]}}},
                                    'ctrst':{'vary':True  ,'HARPN':{'20200128':{'guess':0.5,'bd':[0.1,0.9]},'20201207':{'guess':0.5,'bd':[0.1,0.9]}}},
                                    'FWHM':{'vary':True  ,'HARPN':{'20200128':{'guess':7.,'bd':[2.,30.]},'20201207':{'guess':7.,'bd':[2.,30.]}}}}

    #RM survey
    elif gen_dic['star_name']=='HAT_P3':
        data_dic['Intr']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20200130':{'guess':0.,'bd':[-1.,1.]}}},
                                    'ctrst':{'vary':True  ,'HARPN':{'20200130':{'guess':0.5,'bd':[0.4,0.6]}}},
                                    'FWHM':{'vary':True  ,'HARPN':{'20200130':{'guess':7.,'bd':[7.,8.]}}}}  
        
        #Master-in, from props derived through joined fit
        data_dic['Intr']['conv_model']=True    
        data_dic['Intr']['mod_prop']={'rv':{'vary':False,'HARPN':{'20200130':{'guess':0.}}},
                                      'ctrst':{'vary':False,'HARPN':{'20200130':{'guess':5.8433852613e-01}}},
                                      'FWHM':{'vary':False,'HARPN':{'20200130':{'guess':6.3866079528e+00}}}}        
        
        
    elif gen_dic['star_name']=='Kepler25':
        data_dic['Intr']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20190614':{'guess':0.,'bd':[-10.,10.]}}},
                                    'ctrst':{'vary':True  ,'HARPN':{'20190614':{'guess':0.5,'bd':[0.1,0.9]}}},
                                    'FWHM':{'vary':True  ,'HARPN':{'20190614':{'guess':7.,'bd':[2.,20.]}}}}
        
        #Master-in, from props derived through joined fit
        data_dic['Intr']['conv_model']=True    
        data_dic['Intr']['mod_prop']={'rv':{'vary':False,'HARPN':{'20190614':{'guess':0.}}},
                                      'ctrst':{'vary':False,'HARPN':{'20190614':{'guess':5.0922792293e-01}}},
                                      'FWHM':{'vary':False,'HARPN':{'20190614':{'guess':9.0524725179e+00}}}}           
        
        
    elif gen_dic['star_name']=='Kepler68':
        data_dic['Intr']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20190803':{'guess':0.,'bd':[-4.,4.]}}},
                                    'ctrst':{'vary':True  ,'HARPN':{'20190803':{'guess':0.5,'bd':[0.2,0.8]}}},
                                    'FWHM':{'vary':True  ,'HARPN':{'20190803':{'guess':7.,'bd':[2.,10.]}}}}
        
        #Mock master-in, for null detection
        data_dic['Intr']['conv_model']=True    
        data_dic['Intr']['mod_prop']={'rv':{'vary':False,'HARPN':{'20190803':{'guess':0.}}},
                                      'ctrst':{'vary':False,'HARPN':{'20190803':{'guess':0.}}},
                                      'FWHM':{'vary':False,'HARPN':{'20190803':{'guess':1.0646477855e+01}}}}              
        
    elif gen_dic['star_name']=='HAT_P33':
        data_dic['Intr']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20191204':{'guess':0.,'bd':[-20.,20.]}}},
                                    'ctrst':{'vary':True  ,'HARPN':{'20191204':{'guess':0.5,'bd':[0.3,0.8]}}},
                                    'FWHM':{'vary':True  ,'HARPN':{'20191204':{'guess':7.,'bd':[5.,25.]}}}}
        
        #Master-in, from props derived through joined fit
        data_dic['Intr']['conv_model']=True    
        data_dic['Intr']['mod_prop']={'rv':{'vary':False,'HARPN':{'20191204':{'guess':0.}}},
                                      'ctrst':{'vary':False,'HARPN':{'20191204':{'guess':4.5140061939e-01}}},
                                      'FWHM':{'vary':False,'HARPN':{'20191204':{'guess':1.0898145316e+01}}}}         
        
    elif gen_dic['star_name']=='K2_105':
        data_dic['Intr']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20200118':{'guess':0.,'bd':[-5.,5.]}}},
                                    'ctrst':{'vary':True  ,'HARPN':{'20200118':{'guess':0.5,'bd':[0.1,0.9]}}},
                                    'FWHM':{'vary':True  ,'HARPN':{'20200118':{'guess':7.,'bd':[0.,15.]}}}}
        
        
        #Master-in, from props derived through joined fit
        data_dic['Intr']['conv_model']=True    
        data_dic['Intr']['mod_prop']={'rv':{'vary':False,'HARPN':{'20200118':{'guess':0.}}},
                                      'ctrst':{'vary':False,'HARPN':{'20200118':{'guess':3.9243590607e-01}}},
                                      'FWHM':{'vary':False,'HARPN':{'20200118':{'guess':7.8943588728e+00}}}}           
        
        
    elif gen_dic['star_name']=='HD89345':
        data_dic['Intr']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20200202':{'guess':0.,'bd':[-4.,4.]}}},
                                    'ctrst':{'vary':True  ,'HARPN':{'20200202':{'guess':0.5,'bd':[0.2,0.8]}}},
                                    'FWHM':{'vary':True  ,'HARPN':{'20200202':{'guess':7.,'bd':[0.,15.]}}}}
        
        
        #Master-in, from props derived through joined fit
        data_dic['Intr']['conv_model']=True    
        data_dic['Intr']['mod_prop']={'rv':{'vary':False,'HARPN':{'20200202':{'guess':0.}}},
                                      'ctrst':{'vary':False,'HARPN':{'20200202':{'guess':6.7327730515e-01}}},
                                      'FWHM':{'vary':False,'HARPN':{'20200202':{'guess':4.1685966245e+00}}}}           
        
        
    elif gen_dic['star_name']=='Kepler63':
        data_dic['Intr']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20200513':{'guess':0.,'bd':[-4.,4.]}}},
                                    'ctrst':{'vary':True  ,'HARPN':{'20200513':{'guess':0.5,'bd':[0.2,0.8]}}},
                                    'FWHM':{'vary':True  ,'HARPN':{'20200513':{'guess':7.,'bd':[0.,15.]}}}}   
        
        #Master-in, from props derived through joined fit
        data_dic['Intr']['conv_model']=True    
        data_dic['Intr']['mod_prop']={'rv':{'vary':False,'HARPN':{'20200513':{'guess':0.}}},
                                      'ctrst':{'vary':False,'HARPN':{'20200513':{'guess':3.6208137586e-01}}},
                                      'FWHM':{'vary':False,'HARPN':{'20200513':{'guess':1.0646477855e+01}}}}          
        
    elif gen_dic['star_name']=='HAT_P49':
        data_dic['Intr']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20200730':{'guess':0.,'bd':[-10.,0.]},'binned':{'guess':0.,'bd':[-10.,0.]}}},
                                    'ctrst':{'vary':True  ,'HARPN':{'20200730':{'guess':0.5,'bd':[0.1,0.9]},'binned':{'guess':0.5,'bd':[0.1,0.9]}}},
                                    'FWHM':{'vary':True  ,'HARPN':{'20200730':{'guess':7.,'bd':[2.,30.]},'binned':{'guess':7.,'bd':[2.,30.]}}}}

        
        # #Master-in, from props derived through joined fit
        # data_dic['Intr']['conv_model']=True    
        # data_dic['Intr']['mod_prop']={'rv':{'vary':False,'HARPN':{'20200730':{'guess':0.}}},
        #                               'ctrst':{'vary':False,'HARPN':{'20200730':{'guess':4.1835024960e-01}}},
        #                               'FWHM':{'vary':False,'HARPN':{'20200730':{'guess':1.0074056044e+01}}}} 


    elif gen_dic['star_name']=='WASP47':
        data_dic['Intr']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20210730':{'guess':0.,'bd':[-6.,6.]}}},
                                    'ctrst':{'vary':True  ,'HARPN':{'20210730':{'guess':0.5,'bd':[0.1,0.9]}}},
                                    'FWHM':{'vary':True  ,'HARPN':{'20210730':{'guess':7.,'bd':[2.,14.]}}}}

    elif gen_dic['star_name']=='WASP107':
        data_dic['Intr']['mod_prop']={'rv':{'vary':True ,'HARPS':{'20140406':{'guess':0.,'bd':[-5.,5.]},'20180201':{'guess':0.,'bd':[-5.,5.]},'20180313':{'guess':0.,'bd':[-5.,5.]}},'CARMENES_VIS':{'20180224':{'guess':0.,'bd':[-5.,5.]}}},
                                    'ctrst':{'vary':True  ,'HARPS':{'20140406':{'guess':0.45,'bd':[0.4,0.5]},'20180201':{'guess':0.45,'bd':[0.4,0.5]},'20180313':{'guess':0.45,'bd':[0.4,0.5]}},'CARMENES_VIS':{'20180224':{'guess':0.45,'bd':[0.4,0.5]}}},
                                    'FWHM':{'vary':True  ,'HARPS':{'20140406':{'guess':6.,'bd':[2.,20.]},'20180201':{'guess':6.,'bd':[2.,20.]},'20180313':{'guess':6.,'bd':[2.,20.]}},'CARMENES_VIS':{'20180224':{'guess':8.,'bd':[2.,20.]}}}}
        if 'voigt' in data_dic['Intr']['model'].values(): 
            data_dic['Intr']['mod_prop'].update({'FWHM_LOR':{'vary':False ,'CARMENES_VIS':{'20180224':{'guess':5.6,'bd':[6.,7.]}}}})
       
        
        #Master-in, from props derived through joined fit
        data_dic['Intr']['conv_model']=True 
        r_proj_plot = 0.                        #pour la figure RMR     
        r_proj_plot = 0.7181                        #pour la section WASP-107
        mu_plot = np.sqrt(1.-r_proj_plot**2.)
        ctrst_plot = 4.3616144214e-01*np.poly1d([2.8240504842e-01 , 1.])(mu_plot)
        data_dic['Intr']['mod_prop']={'rv':{'vary':True & False ,'HARPS':{'20140406':{'guess':0.},'20180201':{'guess':0.},'20180313':{'guess':0.},'binned':{'guess':0.}},
                                                                   'CARMENES_VIS':{'20180224':{'guess':0.}}},
                                       'ctrst':{'vary':True  & False ,'HARPS':{'20140406':{'guess':ctrst_plot},'20180201':{'guess':ctrst_plot},'20180313':{'guess':ctrst_plot},'binned':{'guess':ctrst_plot}},
                                                                       'CARMENES_VIS':{'20180224':{'guess':4.8951151325e-01*np.poly1d([2.8240504842e-01 , 1.])(mu_plot)}}}, 
                                        'FWHM':{'vary':True  & False ,'HARPS':{'20140406':{'guess':5.2825992455e+00},'20180201':{'guess':5.2825992455e+00},'20180313':{'guess':5.2825992455e+00},'binned':{'guess':5.2825992455e+00}},
                                                                                     'CARMENES_VIS':{'20180224':{'guess':9.6019355537e-01}}},                                       
                                        'a_damp':{'vary':True  & False ,'HARPS':{'20140406':{'guess':0.},'20180201':{'guess':0.},'20180313':{'guess':0.},'binned':{'guess':0.}},
                                                                                     'CARMENES_VIS':{'20180224':{'guess':4.}}},                  
                                        } 
        
       
        
       
    elif gen_dic['star_name']=='WASP166':
        data_dic['Intr']['mod_prop']={'rv':{'vary':True ,'HARPS':{'20170114':{'guess':0.,'bd':[-10.,10.]},'20170304':{'guess':0.,'bd':[-10.,10.]},'20170315':{'guess':0.,'bd':[-10.,10.]}}},
                                    'ctrst':{'vary':True  ,'HARPS':{'20170114':{'guess':0.,'bd':[0.3,0.6]},'20170304':{'guess':0.5,'bd':[0.3,0.6]},'20170315':{'guess':0.5,'bd':[0.3,0.6]}}},
                                    'FWHM':{'vary':True  ,'HARPS':{'20170114':{'guess':0.,'bd':[2.,20.]},'20170304':{'guess':7.,'bd':[2.,20.]},'20170315':{'guess':7.,'bd':[2.,20.]}}}}    
    
        #Master-in, from props derived through joined fit
        data_dic['Intr']['conv_model']=True    
        data_dic['Intr']['mod_prop']={'rv':{'vary':False,'HARPS':{}},
                                      'ctrst':{'vary':False,'HARPS':{}},
                                      'FWHM':{'vary':False,'HARPS':{}}}     
        for vis in ['20170114','20170304','20170315','binned']:
            data_dic['Intr']['mod_prop']['rv']['HARPS'][vis] = {'guess':0.,'bd':[0.2,0.4]}
            data_dic['Intr']['mod_prop']['ctrst']['HARPS'][vis] = {'guess':5.6818397517e-01,'bd':[0.2,0.4]}
            data_dic['Intr']['mod_prop']['FWHM']['HARPS'][vis] = {'guess':6.3816059188e+00,'bd':[0.2,0.4]}
    
    
    elif gen_dic['star_name']=='HAT_P11':
        data_dic['Intr']['mod_prop']={'rv':{'vary':True ,'HARPN':{'20150913':{'guess':0.,'bd':[-5.,5.]},'20151101':{'guess':0.,'bd':[-5.,5.]}},'CARMENES_VIS':{'20170807':{'guess':0.,'bd':[-5.,5.]},'20170812':{'guess':0.,'bd':[-5.,5.]}}},
                                      'ctrst':{'vary':True  ,'HARPN':{'20150913':{'guess':0.5,'bd':[0.3,0.7]},'20151101':{'guess':0.5,'bd':[0.3,0.7]}},'CARMENES_VIS':{'20170807':{'guess':0.,'bd':[0.3,0.7]},'20170812':{'guess':0.5,'bd':[0.3,0.7]}}},
                                      'FWHM':{'vary':True  ,'HARPN':{'20150913':{'guess':7.,'bd':[2.,15.]},'20151101':{'guess':7.,'bd':[2.,15.]}},'CARMENES_VIS':{'20170807':{'guess':0.,'bd':[2.,15.]},'20170812':{'guess':7.,'bd':[2.,15.]}}}}

        
        #Master-in, from props derived through joined fit
        data_dic['Intr']['conv_model']=True 
        data_dic['Intr']['mod_prop']={'rv':{'vary':True  & False ,'HARPN':{'20150913':{'guess':0.},'20151101':{'guess':0.},'binned':{'guess':0.}},
                                                                   'CARMENES_VIS':{'20170807':{'guess':0.},'20170812':{'guess':0.},'binned':{'guess':0.}}},
                                        'ctrst':{'vary':True  & False ,'HARPN':{'20150913':{'guess':6.3159051094e-01},'20151101':{'guess':6.3159051094e-01},'binned':{'guess':6.3159051094e-01}},
                                                                                     'CARMENES_VIS':{'20170807':{'guess':5.8700381496e-01},'20170812':{'guess':5.8700381496e-01},'binned':{'guess':5.8700381496e-01}}},
                                        'FWHM':{'vary':True  & False ,'HARPN':{'20150913':{'guess':4.4592294931e+00},'20151101':{'guess':4.4592294931e+00},'binned':{'guess':4.4592294931e+00}},
                                                                                     'CARMENES_VIS':{'20170807':{'guess':3.9553749758e+00},'20170812':{'guess':3.9553749758e+00},'binned':{'guess':3.9553749758e+00}}},                                       
                                        'a_damp':{'vary':True  & False ,'HARPN':{'20150913':{'guess':0.},'20151101':{'guess':0.},'binned':{'guess':0.}},
                                                                                     'CARMENES_VIS':{'20170807':{'guess':5.6982154625e-01},'20170812':{'guess':5.6982154625e-01},'binned':{'guess':5.6982154625e-01}}},           
                                        }  



    elif gen_dic['star_name']=='WASP156'  :
        data_dic['Intr']['mod_prop']={'rv':{'vary':True ,'CARMENES_VIS':{'20190928':{'guess':0.,'bd':[-5.,5.]},'20191025':{'guess':0.,'bd':[-5.,5.]}}},
                                    'ctrst':{'vary':True  ,'CARMENES_VIS':{'20190928':{'guess':0.5,'bd':[0.3,0.8]},'20191025':{'guess':0.5,'bd':[0.3,0.8]}}},
                                    'FWHM':{'vary':True  ,'CARMENES_VIS':{'20190928':{'guess':7.,'bd':[2.,10.]},'20191025':{'guess':7.,'bd':[2.,10.]}}}} 
          
        #Master-in, from props derived through joined fit
        data_dic['Intr']['conv_model']=True 
        data_dic['Intr']['mod_prop']={'rv':{'vary':True& False  ,'CARMENES_VIS':{'20190928':{'guess':0.,'bd':[-5.,5.]},'20191025':{'guess':0.,'bd':[-5.,5.]}}},
                                    'ctrst':{'vary':True & False  ,'CARMENES_VIS':{'20190928':{'guess':4.8246003364e-01,'bd':[0.3,0.8]},'20191025':{'guess':0.5,'bd':[0.3,0.8]}}},
                                    'FWHM':{'vary':True & False  ,'CARMENES_VIS':{'20190928':{'guess':8.0697590311e+00 ,'bd':[2.,10.]},'20191025':{'guess':7.,'bd':[2.,10.]}}}}         
        
    elif gen_dic['star_name']=='HD106315':
        data_dic['Intr']['mod_prop']={'rv':{'vary':True ,'HARPS':{'20170309':{'guess':0.,'bd':[-20.,20.]},'20170330':{'guess':0.,'bd':[-20.,20.]},'20180323':{'guess':0.,'bd':[-20.,20.]}}},
                                    'ctrst':{'vary':True  ,'HARPS':{'20170309':{'guess':0.5,'bd':[0.1,0.9]},'20170330':{'guess':0.5,'bd':[0.1,0.9]},'20180323':{'guess':0.5,'bd':[0.1,0.9]}}},
                                    'FWHM':{'vary':True  ,'HARPS':{'20170309':{'guess':7.,'bd':[2.,20.]},'20170330':{'guess':7.,'bd':[2.,20.]},'20180323':{'guess':7.,'bd':[2.,20.]}}}} 

        #Master-in, from props derived through joined fit
        data_dic['Intr']['conv_model']=True    
        data_dic['Intr']['mod_prop']={'rv':{'vary':False,'HARPS':{}},
                                      'ctrst':{'vary':False,'HARPS':{}},
                                      'FWHM':{'vary':False,'HARPS':{}}}     
        for vis in ['20170309','20170330','20180323','binned']:
            data_dic['Intr']['mod_prop']['rv']['HARPS'][vis] = {'guess':0.,'bd':[0.2,0.4]}
            data_dic['Intr']['mod_prop']['ctrst']['HARPS'][vis] = {'guess':4.5882739304e-01,'bd':[0.2,0.4]}
            data_dic['Intr']['mod_prop']['FWHM']['HARPS'][vis] = {'guess':1.2801166639e+01,'bd':[0.2,0.4]}


    elif gen_dic['star_name']=='55Cnc'  :
        
        #Master-in
        data_dic['Intr']['conv_model']=True 
        data_dic['Intr']['mod_prop']={'rv':{'vary':True& False  ,'ESPRESSO':{'20200205':{'guess':0.,'bd':[-5.,5.]},'20210121':{'guess':0.,'bd':[-5.,5.]},'20210124':{'guess':0.,'bd':[-5.,5.]},'binned':{'guess':0.,'bd':[-5.,5.]}}},
                                    'ctrst':{'vary':True   ,'ESPRESSO':{'20200205':{'guess':4.8246003364e-01,'bd':[0.3,0.8]},'20210121':{'guess':0.5,'bd':[0.3,0.8]},'20210124':{'guess':0.5,'bd':[0.3,0.8]},'binned':{'guess':0.5,'bd':[0.3,0.8]}}},
                                    'FWHM':{'vary':True   ,'ESPRESSO':{'20200205':{'guess':8.0697590311e+00 ,'bd':[2.,10.]},'20210121':{'guess':7.,'bd':[2.,10.]},'20210124':{'guess':7.,'bd':[2.,10.]},'binned':{'guess':7.,'bd':[2.,10.]}}}}         
        


    #Fitting mode 
    data_dic['Intr']['fit_mod']=''
    data_dic['Intr']['fit_mod']='chi2'
    # data_dic['Intr']['fit_mod']='mcmc'


    #Printing fits results
    data_dic['Intr']['verbose'] =  True      &   False  

    
    #Priors on variable properties
    if gen_dic['transit_pl']=='55Cnc_e':
        data_dic['Intr']['line_fit_priors']={}
    #    data_dic['Intr']['line_fit_priors']={'H_FWHM':12.}
    elif gen_dic['star_name']=='WASP76':    
#        data_dic['Intr']['line_fit_priors']={
#                'c1_ctrst_fit':False,
#                'c2_ctrst_fit':False,
#                'c1_sig_fit':False,
#                'c2_sig_fit':False}  
        # data_dic['Intr']['line_fit_priors']={'RV_L':-1.,'RV_H':1.,'alpha_rot_L':0.,'alpha_rot_H':1.,'cos_istar_L':-1.,'cos_istar_H':1.,'veq_L':0.}        
        data_dic['Intr']['line_fit_priors']={       #priors defined by looking at the 1D PDFs and time-series properties, after exclusion of noisiest exposures     
            'rv':{'mod': 'uf', 'low':-1.5,'high':1.5},          
            'FWHM':{'mod': 'uf', 'low':6.,'high':13.},
            'ctrst':{'mod': 'uf', 'low':0.4,'high':0.75}}  
    elif gen_dic['star_name']=='HD209458':
        data_dic['Intr']['line_fit_priors']={       #priors defined by looking at the 1D PDFs and time-series properties, after exclusion of noisiest exposures        
            'rv':{'mod': 'uf', 'low':-7.,'high':7.},           
            'FWHM':{'mod': 'uf', 'low':6.,'high':9.},
            'ctrst':{'mod': 'uf', 'low':0.5,'high':0.7}}   

        # data_dic['Intr']['line_fit_priors']={       #priors for Na CCF fit   
        #     'rv':{'mod': 'uf', 'low':-10.,'high':10.},           
        #     'FWHM':{'mod': 'uf', 'low':0.,'high':30.},
        #     'ctrst':{'mod': 'uf', 'low':0.,'high':1.}} 
        
        
    
    elif gen_dic['star_name']=='GJ436':  
    #     # data_dic['Intr']['line_fit_priors']={'amp_l2c':1.,'FWHM_l2c':1.,'RV_l2c':0.}    
    #     # data_dic['Intr']['line_fit_priors']={'amp':-80.,'rv':0.,'FWHM':2.7} 
        
                
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-1.,'high':1.},
                                    # 'amp':{'mod': 'uf', 'low':-1e10,'high':0.},
                                    'ctrst':{'mod': 'uf', 'low':0.,'high':1.},
                                    'FWHM':{'mod': 'uf', 'low':0.,'high':10.}}

        
    elif gen_dic['star_name']=='HD3167':     
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-10.,'high':10.},'FWHM':{'mod': 'uf', 'low':0.,'high':20.},'amp':{'mod': 'uf', 'low':-2.,'high':2.}}
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-5.,'high':5.},'FWHM':{'mod': 'uf', 'low':0.,'high':20.},'amp':{'mod': 'uf', 'low':-2.,'high':2.}}
    elif gen_dic['star_name']=='TOI858':     
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-4.,'high':10.},'FWHM':{'mod': 'uf', 'low':0.,'high':20.},'amp':{'mod': 'uf', 'low':-2.,'high':2.}}
    elif gen_dic['star_name']=='TOI858':     
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-4.,'high':10.},'FWHM':{'mod': 'uf', 'low':0.,'high':20.},'amp':{'mod': 'uf', 'low':-2.,'high':2.}}
    if gen_dic['star_name']=='HIP41378':      
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-10.,'high':10.},'FWHM':{'mod': 'uf', 'low':0.,'high':15.},'ctrst':{'mod': 'uf', 'low':0.,'high':1.}}
        data_dic['Intr']['line_fit_priors'].update( {'cont':{'mod': 'uf', 'low':0.,'high':2.}} )        
        data_dic['Intr']['line_fit_priors'].update( {'slope':{'mod': 'uf', 'low':-100.,'high':100.}} ) 
                                                     

    
    #RM survey
    elif gen_dic['star_name']=='HAT_P3':     
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-5.,'high':5.},'FWHM':{'mod': 'uf', 'low':0.,'high':20.},'ctrst':{'mod': 'uf', 'low':-1.,'high':2.}}
    elif gen_dic['star_name']=='Kepler25':     
        # data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-15.,'high':15.},'FWHM':{'mod': 'uf', 'low':0.,'high':20.},'ctrst':{'mod': 'uf', 'low':-1.,'high':2.}}   #larges
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-15.,'high':15.},'FWHM':{'mod': 'uf', 'low':0.,'high':20.},'ctrst':{'mod': 'uf', 'low':0.,'high':1.}}    #forts
    elif gen_dic['star_name']=='Kepler68':     
        # data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-10.,'high':10.},'FWHM':{'mod': 'uf', 'low':0.,'high':20.},'ctrst':{'mod': 'uf', 'low':-1.,'high':2.}}   #larges
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-4.,'high':4.},'FWHM':{'mod': 'uf', 'low':0.,'high':10.},'ctrst':{'mod': 'uf', 'low':0.,'high':1.}}   #forts
    elif gen_dic['star_name']=='HAT_P33':     
        # data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-30.,'high':30.},'FWHM':{'mod': 'uf', 'low':0.,'high':30.},'ctrst':{'mod': 'uf', 'low':-1.,'high':2.}}     #larges
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-30.,'high':30.},'FWHM':{'mod': 'uf', 'low':5.,'high':25.},'ctrst':{'mod': 'uf', 'low':0.,'high':1.}}    #forts
    elif gen_dic['star_name']=='K2_105':     
        # data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-6.,'high':6.},'FWHM':{'mod': 'uf', 'low':0.,'high':30.},'ctrst':{'mod': 'uf', 'low':-1.,'high':2.}}     #larges
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-6.,'high':6.},'FWHM':{'mod': 'uf', 'low':0.,'high':15.},'ctrst':{'mod': 'uf', 'low':0.,'high':1.}}     #forts
    elif gen_dic['star_name']=='HD89345':     
        # data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-10.,'high':10.},'FWHM':{'mod': 'uf', 'low':0.,'high':30.},'ctrst':{'mod': 'uf', 'low':-1.,'high':2.}}     #larges
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-5.,'high':5.},'FWHM':{'mod': 'uf', 'low':0.,'high':20.},'ctrst':{'mod': 'uf', 'low':0.,'high':1.}}     #forts
    elif gen_dic['star_name']=='Kepler63':     
        # data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-20.,'high':20.},'FWHM':{'mod': 'uf', 'low':0.,'high':30.},'ctrst':{'mod': 'uf', 'low':-1.,'high':2.}}     #larges
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-10.,'high':10.},'FWHM':{'mod': 'uf', 'low':0.,'high':20.},'ctrst':{'mod': 'uf', 'low':0.,'high':1.}}     #forts
    elif gen_dic['star_name']=='HAT_P49':     
        # data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-20.,'high':10.},'FWHM':{'mod': 'uf', 'low':0.,'high':30.},'ctrst':{'mod': 'uf', 'low':-1.,'high':2.}}     #larges
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-20.,'high':10.},'FWHM':{'mod': 'uf', 'low':0.,'high':30.},'ctrst':{'mod': 'uf', 'low':0.,'high':1.}}     #forts
    elif gen_dic['star_name']=='WASP47':  
        # data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-15.,'high':15.},'FWHM':{'mod': 'uf', 'low':0.,'high':30.},'ctrst':{'mod': 'uf', 'low':-1.,'high':2.}}     #larges
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-6.,'high':6.},'FWHM':{'mod': 'uf', 'low':0.,'high':15.},'ctrst':{'mod': 'uf', 'low':0.,'high':1.}}     #forts
    
                                                          
    elif gen_dic['star_name']=='WASP107':     
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-5.,'high':5.},'FWHM':{'mod': 'uf', 'low':0.,'high':20.},'ctrst':{'mod': 'uf', 'low':0.,'high':1.}}     #forts
    elif gen_dic['star_name']=='WASP166':     
        # data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-15.,'high':15.},'FWHM':{'mod': 'uf', 'low':0.,'high':30.},'ctrst':{'mod': 'uf', 'low':-1.,'high':2.}}     #larges
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-10.,'high':10.},'FWHM':{'mod': 'uf', 'low':0.,'high':20.},'ctrst':{'mod': 'uf', 'low':0.,'high':1.}}     #forts
    elif gen_dic['star_name']=='HAT_P11':     
        # data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-15.,'high':15.},'FWHM':{'mod': 'uf', 'low':0.,'high':30.},'ctrst':{'mod': 'uf', 'low':-1.,'high':2.}}     #larges
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-5.,'high':5.},'FWHM':{'mod': 'uf', 'low':0.,'high':20.},'ctrst':{'mod': 'uf', 'low':0.,'high':1.}}     #forts
    elif gen_dic['star_name']=='WASP156':     
        # data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-20.,'high':20.},'FWHM':{'mod': 'uf', 'low':0.,'high':30.},'ctrst':{'mod': 'uf', 'low':-1.,'high':2.}}     #larges
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-5.,'high':5.},'FWHM':{'mod': 'uf', 'low':0.,'high':20.},'ctrst':{'mod': 'uf', 'low':0.,'high':1.}}     #forts
    elif gen_dic['star_name']=='HD106315':     
        # data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-30.,'high':30.},'FWHM':{'mod': 'uf', 'low':0.,'high':30.},'ctrst':{'mod': 'uf', 'low':-1.,'high':2.}}     #larges
        data_dic['Intr']['line_fit_priors']={'rv':{'mod': 'uf', 'low':-20.,'high':20.},'FWHM':{'mod': 'uf', 'low':0.,'high':20.},'ctrst':{'mod': 'uf', 'low':0.,'high':1.}}     #forts
                    

  
    #Detection thresholds    
    if gen_dic['star_name']=='HD3167':    
        data_dic['Intr']['thresh_area']=3.9    
        data_dic['Intr']['thresh_area']=3.  #olk K5 mask from S2D
        data_dic['Intr']['thresh_amp']=0.    
        # data_dic['Intr']['thresh_area']=3.       #'HD3167_b'
        # data_dic['Intr']['thresh_amp']=2.
    elif gen_dic['transit_pl']=='GJ436_b':    
        data_dic['Intr']['thresh_area']=5.    
        data_dic['Intr']['thresh_amp']=3. 
    elif gen_dic['transit_pl'] in ['Corot7b','GJ9827d']:
        data_dic['Intr']['thresh_area']=3.    
        data_dic['Intr']['thresh_amp']=2.
    elif gen_dic['transit_pl']=='Nu2Lupi_c':
        data_dic['Intr']['thresh_area']=4.    
        data_dic['Intr']['thresh_amp']=4.1
        data_dic['Intr']['thresh_amp']=3.      #for red detector


    
    #Force detection flag
    if gen_dic['transit_pl']=='WASP_8b':
        data_dic['Intr']['idx_force_det']={'HARPS':{'2008-10-04':{1:False,44:False}}}
    elif gen_dic['transit_pl']=='55Cnc_e':
        data_dic['Intr']['idx_force_det']={
            'binned':{
                'all_HARPSS':{2:False,26:False},
                'all_HARPS_adj':{25:False},
                'all_HARPS_adj2':{13:False},
                '2012-02-27_binned':{1:False,4:False,6:False,8:False,9:False,10:False,11:True,13:False,15:False},
                '2012-01-27_binned':{1:False,3:False,8:True,12:True},
                '2012-03-15_binned':{7:True,8:True},
                '2012-02-13_binned':{1:True,2:True},
                'good_HARPSN':{14:True},
    #            'good_HARPSN_adj':{12:True}   #sans 2013-11-14
    #            'good_HARPSN_adj':{10:True}   #sans 2013-11-28, 2014-01-26
                'HARPS_HARPSN_binHARPS':{0:False,1:False,13:False},   
                'HARPS_HARPSN_binHARPSN':{14:False},            
                },
            'HARPN':{'2012-12-25':{1:True},
                       '2013-11-28':{3:True},
                       '2014-01-26':{5:False,13:False},
                       '2014-02-26':{7:False},
                       },            
            }
    
#     elif gen_dic['transit_pl']=='HD3167_c':  
#         data_dic['Intr']['idx_force_det']={'HARPN':{'2016-10-01':{18:True,0:False,15:False}}}       #exclusion de la CCF a l'ingress, et de celle tres fine
#         data_dic['Intr']['idx_force_det']={'HARPN':{'2016-10-01':{0:False,15:False,17:True}}}       #mask K5
#         data_dic['Intr']['idx_force_det']={'HARPN':{'2016-10-01':{0:True,14:True,15:True,17:True,18:True,19:True}}}       #mask K5   si je conserve toutes 
# #        data_dic['Intr']['idx_force_det']={'HARPN':{'2016-10-01':{}}}
# #        for i in range(1,21):data_dic['Intr']['idx_force_det']['HARPN']['2016-10-01'][i]=True       #mask K5   dans le cas ou je varie le T0+1sig
#     #    data_dic['Intr']['idx_force_det']={'HARPN':{'2016-10-01':{0:True,14:True,15:True,17:True,18:False,19:False}}}       #mask K5   toutes celles detectees a plus de 4 sigma    
#     #    data_dic['Intr']['idx_force_det']={'HARPN':{'2016-10-01':{0:False,14:True,15:False,17:True,18:True,19:True}}}       #mask K5   toutes celles detectees sauf deux spurieuses

#         data_dic['Intr']['idx_force_det']={'HARPN':{'2016-10-01':{0:False}}}       #newred 


        
    if gen_dic['transit_pl']=='WASP121b': 
    #    data_dic['Intr']['idx_force_det']={'HARPS':{'14-01-18':{0:False,19:False},'31-12-17':{0:False,15:False}}}      #Mask G, critere contraste uniquement
        data_dic['Intr']['idx_force_det']={'HARPS':{'14-01-18':{1:False}}}      #Mask F
#        data_dic['Intr']['idx_force_det']={'binned':{'HARPS-binned':{7:False,8:False,9:False,10:False}}}      #Mask F + atmo
        
        
    if gen_dic['transit_pl']=='Kelt9b': 
        data_dic['Intr']['idx_force_det']={'HARPN':{'31-07-2017':{1:True,5:False,6:False,7:False,8:False,9:False,10:False,11:False,12:False,21:True}}}   
    if gen_dic['transit_pl']=='WASP127b': 
        data_dic['Intr']['idx_force_det']={'HARPS':{'2017-03-20':{0:False,29:False}}} 

    
    elif gen_dic['transit_pl']=='WASP76b':
        data_dic['Intr']['idx_force_det']={
            'ESPRESSO':{'2018-10-31':{},'2018-09-03':{}},
            'binned':{'ESP_binned':{}}     
        }   
        # for i in range(25-12,40-12):data_dic['Intr']['idx_force_det']['binned']['ESP_binned'][i]=False
        for i in range(15,28):data_dic['Intr']['idx_force_det']['ESPRESSO']['2018-10-31'][i]=False    
        for i in range(9,16):data_dic['Intr']['idx_force_det']['ESPRESSO']['2018-09-03'][i]=False      

    elif gen_dic['transit_pl']=='GJ9827d':
        data_dic['Intr']['idx_force_det']={
            'ESPRESSO':{'2019-08-25':{}}}
        data_dic['Intr']['idx_force_det']['ESPRESSO']['2019-08-25'][12]=False        
    

    
    #Calculating/retrieving
    data_dic['Intr']['mcmc_run_mode']='use'
    
    #Walkers
    data_dic['Intr']['mcmc_set']={}
    if gen_dic['star_name']=='HD3167':     
        data_dic['Intr']['mcmc_set']={         
            'nwalkers':{'ESPRESSO':{'2019-10-09':100},'HARPN':{'2016-10-01':100}},
            'nsteps':{'ESPRESSO':{'2019-10-09':2000},'HARPN':{'2016-10-01':2000}},            
            'nburn':{'ESPRESSO':{'2019-10-09':500},'HARPN':{'2016-10-01':500}},                                
            }
    elif gen_dic['star_name']=='TOI858':     
        data_dic['Intr']['mcmc_set']={         
            'nwalkers':{'CORALIE':{'20191205':100,'20210118':100}},
            'nsteps':{'CORALIE':{'20191205':2000,'20210118':2000}},            
            'nburn':{'CORALIE':{'20191205':500,'20210118':500}},                                
            }
    elif gen_dic['star_name']=='GJ436':     
        data_dic['Intr']['mcmc_set']={         
            'nwalkers':{'ESPRESSO':{'20190228':100,'20190429':100},'HARPS':{'20070509':100},'HARPN':{'20160318':100,'20160411':100}},
            'nsteps':{'ESPRESSO':{'20190228':2000,'20190429':2000},'HARPS':{'20070509':2000},'HARPN':{'20160318':2000,'20160411':2000}},          
            'nburn':{'ESPRESSO':{'20190228':500,'20190429':500},'HARPS':{'20070509':500},'HARPN':{'20160318':500,'20160411':500}},                                
            }  
    elif gen_dic['star_name']=='HIP41378':     
        data_dic['Intr']['mcmc_set']={'nwalkers':{'HARPN':{'20191218':50,'20220401':50}},'nsteps':{'HARPN':{'20191218':1000,'20220401':1000}},'nburn':{'HARPN':{'20191218':300,'20220401':300}}}  
    #RM survey    
    elif gen_dic['star_name']=='HAT_P3':     
        data_dic['Intr']['mcmc_set']={'nwalkers':{'HARPN':{'20200130':50}},'nsteps':{'HARPN':{'20200130':1000}},'nburn':{'HARPN':{'20200130':200}}}     
    elif gen_dic['star_name']=='Kepler25':     
        data_dic['Intr']['mcmc_set']={'nwalkers':{'HARPN':{'20190614':50}},'nsteps':{'HARPN':{'20190614':1000}},'nburn':{'HARPN':{'20190614':200}}}     
    elif gen_dic['star_name']=='Kepler68':     
        data_dic['Intr']['mcmc_set']={'nwalkers':{'HARPN':{'20190803':50}},'nsteps':{'HARPN':{'20190803':1000}},'nburn':{'HARPN':{'20190803':200}}}                    
    elif gen_dic['star_name']=='HAT_P33':     
        data_dic['Intr']['mcmc_set']={'nwalkers':{'HARPN':{'20191204':50}},'nsteps':{'HARPN':{'20191204':1000}},'nburn':{'HARPN':{'20191204':200}}}                     
    elif gen_dic['star_name']=='K2_105':     
        data_dic['Intr']['mcmc_set']={'nwalkers':{'HARPN':{'20200118':50}},'nsteps':{'HARPN':{'20200118':1000}},'nburn':{'HARPN':{'20200118':200}}}                     
    elif gen_dic['star_name']=='HD89345':     
        data_dic['Intr']['mcmc_set']={'nwalkers':{'HARPN':{'20200202':50}},'nsteps':{'HARPN':{'20200202':1000}},'nburn':{'HARPN':{'20200202':200}}}           
    elif gen_dic['star_name']=='Kepler63':     
        # data_dic['Intr']['mcmc_set']={'nwalkers':{'HARPN':{'20200513':100}},'nsteps':{'HARPN':{'20200513':1500}},'nburn':{'HARPN':{'20200513':400}}} 
        data_dic['Intr']['mcmc_set']={'nwalkers':{'HARPN':{'20200513':50}},'nsteps':{'HARPN':{'20200513':1000}},'nburn':{'HARPN':{'20200513':200}}}         
    elif gen_dic['star_name']=='HAT_P49':     
        data_dic['Intr']['mcmc_set']={'nwalkers':{'HARPN':{'20200730':100}},'nsteps':{'HARPN':{'20200730':1500}},'nburn':{'HARPN':{'20200730':400}}}           
    elif gen_dic['star_name']=='WASP47':   
        data_dic['Intr']['mcmc_set']={'nwalkers':{'HARPN':{'20210730':50}},'nsteps':{'HARPN':{'20210730':1000}},'nburn':{'HARPN':{'20210730':200}}}  
        
    elif gen_dic['star_name']=='WASP107':     
        data_dic['Intr']['mcmc_set']={'nwalkers':{'HARPS':{'20140406':50,'20180201':50,'20180313':50},'CARMENES_VIS':{'20180224':50}},
                                      'nsteps':{'HARPS':{'20140406':1000,'20180201':1000,'20180313':1000},'CARMENES_VIS':{'20180224':1000}},
                                      'nburn':{'HARPS':{'20140406':200,'20180201':200,'20180313':200},'CARMENES_VIS':{'20180224':200}}}         
    elif gen_dic['star_name']=='WASP166':     
        data_dic['Intr']['mcmc_set']={'nwalkers':{'HARPS':{'20170114':50,'20170304':50,'20170315':50}},
                                      'nsteps':{'HARPS':{'20170114':1000,'20170304':1000,'20170315':1000}},
                                      'nburn':{'HARPS':{'20170114':200,'20170304':200,'20170315':200}}}   
    elif gen_dic['star_name']=='HAT_P11':     
        data_dic['Intr']['mcmc_set']={'nwalkers':{'HARPN':{'20150913':50,'20151101':50},'CARMENES_VIS':{'20170807':50,'20170812':50}},
                                      'nsteps':{'HARPN':{'20150913':1000,'20151101':1000},'CARMENES_VIS':{'20170807':1000,'20170812':1000}},
                                      'nburn':{'HARPN':{'20150913':200,'20151101':200},'CARMENES_VIS':{'20170807':200,'20170812':200}}}         
    elif gen_dic['star_name']=='WASP156':     
        data_dic['Intr']['mcmc_set']={'nwalkers':{'CARMENES_VIS':{'20190928':50,'20191025':50}},
                                      'nsteps':{'CARMENES_VIS':{'20190928':1000,'20191025':1000}},
                                      'nburn':{'CARMENES_VIS':{'20190928':200,'20191025':200}}}   
    elif gen_dic['star_name']=='HD106315':     
        data_dic['Intr']['mcmc_set']={'nwalkers':{'HARPS':{'20170309':50,'20170330':50,'20180323':50}},
                                      'nsteps':{'HARPS':{'20170309':1000,'20170330':1000,'20180323':1000}},
                                      'nburn':{'HARPS':{'20170309':200,'20170330':200,'20180323':200}}}  
    elif gen_dic['star_name']=='HD209458':     
        data_dic['Intr']['mcmc_set']={'nwalkers':{'ESPRESSO':{'20190720':50,'20190911':50}},
                                      'nsteps':{'ESPRESSO':{'20190720':1000,'20190911':1000}},
                                      'nburn':{'ESPRESSO':{'20190720':200,'20190911':200}}}        
    elif gen_dic['star_name']=='WASP76':     
        data_dic['Intr']['mcmc_set']={'nwalkers':{'ESPRESSO':{'20180902':50,'20181030':50}},
                                      'nsteps':{'ESPRESSO':{'20180902':1000,'20181030':1000}},
                                      'nburn':{'ESPRESSO':{'20180902':200,'20181030':200}}}   
    
    #Walkers exclusion
    data_dic['Intr']['exclu_walk_autom']=None  #  5.
    
    
    #Derived errors
    if gen_dic['star_name']=='HD3167':      
    
        # data_dic['Intr']['HDI_bwf']={'rv':{'ESPRESSO':{'2019-10-09':4.}},'FWHM':{'ESPRESSO':{'2019-10-09':4.}},'ctrst':{'ESPRESSO':{'2019-10-09':4.}}}       
        data_dic['Intr']['HDI_bwf']={'ctrst':{'ESPRESSO':{'2019-10-09':4.},'HARPN':{'2016-10-01':2.5}}}
                                  
        data_dic['Intr']['HDI_dbins']={'rv':{'ESPRESSO':{'2019-10-09':2.},'HARPN':{'2016-10-01':1.7}},
                                    'FWHM':{'ESPRESSO':{'2019-10-09':2.5}}}  
    #     data_dic['Intr']['HDI_dbins']={'FWHM':{'ESPRESSO':{'2019-10-09':2.}}}

    elif gen_dic['star_name']=='K2_105': 
        data_dic['Intr']['HDI_dbins']={'rv':{'HARPN':{'20200118':1}},'ctrst':{'HARPN':{'20200118':0.2}},'FWHM':{'HARPN':{'20200118':2}}}    #a regler


    #1D PDF from mcmc
    plot_dic['prop_Intr_mcmc_PDFs']=''     

    #Derived properties
    plot_dic['prop_Intr']='pdf'  










        
        

    
    
    
    
##################################################################################################       
#%%% Module: fitting planet-occulted stellar properties
#    - fitting single stellar surface property from planet-occulted regions with a common model for all instruments/visits 
#    - with properties derived from individual local profiles
#    - this module can be used to estimate the surface RV model and analytical laws describing the intrinsic line properties
#      the final fit should be performed over the joined intrinsic line profiles with gen_dic['fit_IntrProf']
##################################################################################################       

#%%%% Activating 
gen_dic['fit_IntrProp'] = False


#%%%% Multi-threading
glob_fit_dic['IntrProp']['nthreads'] = 14 


#%%%% Fitted data

#%%%%% Exposures to be fitted
#    - indexes are relative to in-transit tables
#    - define instruments and visits to be fitted (they will not be fitted if not used as keys, or if set to [], which is the default value), set their value to 'all' for all in-transit exposures to be fitted
#    - add '_bin' at the end of a visit name for its binned exposures to be fitted instead of the original ones (must have been calculated with the binning module); it can be justified when surface RVs cannot be derived from unbinned profiles
#      all other mentions of the visit (eg in parameter names) can still refer to the original visit name
glob_fit_dic['IntrProp']['idx_in_fit'] = {}


#%%%% Fitted property
#    - adapt glob_fit_dic['IntrProp']['mod_prop'] to the chosen property
# + 'rv': fitted using surface RV model
# + 'ctrst', 'FWHM': fitted using polynomial models
glob_fit_dic['IntrProp']['prop'] = 'rv'


#%%%% Line property fit

#%%%%% Coordinate
#    - the line properties will be fitted as a function of this coordinate
# +'mu' angle       
# +'xp_abs': absolute distance from projected orbital normal in the sky plane
# +'r_proj': distance from star center projected in the sky plane      
# +'abs_y_st' : sky-projected distance parallel to spin axis, absolute value   
# +'y_st2' : sky-projected distance parallel to spin axis, squared
glob_fit_dic['IntrProp']['dim_fit']='mu'
  

#%%%%% Variation
#    - fit line property as absolute ('abs') or modulated ('modul') polynomial
glob_fit_dic['IntrProp']['pol_mode']='abs'     


#%%%%% Fixed/variable properties
#    - structure is different from data_dic['DI']['mod_prop'], where properties are fitted independently for each instrument and visit
#    - here the names of properties must be defined as 'prop__ISinst_VSvis'  
# + 'inst' is the name of the instrument, which should be set to '_' for the property to be common to all instruments and their visits
# + 'vis' is the name of the visit, which should be set to '_' for the property to be common to all visits of this instrument   
glob_fit_dic['IntrProp']['mod_prop']={}


#%%%% Fit settings

#%%%%% Fitting mode 
#    - 'chi2', 'mcmc', ''
glob_fit_dic['IntrProp']['fit_mod']=''  


#%%%%% Printing fits results
glob_fit_dic['IntrProp']['verbose'] = False


#%%%%% Priors on variable properties
#    - see gen_dic['fit_DI'] for details
glob_fit_dic['IntrProp']['priors']={} 
    

#%%%%% Derived properties
#    - each field calls a specific function (see routine for more details)
glob_fit_dic['IntrProp']['modif_list'] = []        


#%%%%% MCMC settings

#%%%%%% Calculating/retrieving
glob_fit_dic['IntrProp']['mcmc_run_mode']='use'


#%%%%%% Walkers 
glob_fit_dic['IntrProp']['mcmc_set']={}


#%%%%%% Complex priors
glob_fit_dic['IntrProp']['prior_func']={}  
 

#%%%%%% Walkers exclusion  
#    - define conditions within routine
glob_fit_dic['IntrProp']['exclu_walk']= False       

#%%%%%% Automatic exclusion of outlying chains
#    - set to None, or exclusion threshold
glob_fit_dic['IntrProp']['exclu_walk_autom']= None  

#%%%%%% Derived errors
#    - 'quant' or 'HDI'
glob_fit_dic['IntrProp']['out_err_mode']='HDI'  


#%%%%%% Derived lower/upper limits
glob_fit_dic['IntrProp']['conf_limits']={}  


#%%%% Plot settings

#%%%%% MCMC chains
glob_fit_dic['IntrProp']['save_MCMC_chains']=''        


#%%%%% MCMC corner plot
#    - see function for options
glob_fit_dic['IntrProp']['corner_options']={}


#%%%%% Chi2 values
#    - plot chi2 values for each datapoint
plot_dic['chi2_fit_IntrProp']=''        
  


if __name__ == '__main__':

    #Activating 
    gen_dic['fit_IntrProp'] = True   &  False

    #Exposures to be fitted
    # if gen_dic['star_name']=='HD3167':

    #     if list(gen_dic['transit_pl'].keys())==['HD3167_c']:
    #     # #    - exclusion of planet-contaminated range
    #     # for vis in []:
    #     #     cond_fit_dic['HARPS'][vis] &= (data_load[inst][vis][i_in]['rv']<=8.)
        
    #     #Force fit over all exposures but the last three ones
    #     cond_fit_dic[inst][vis] = np.repeat(True,20)
    #     cond_fit_dic[inst][vis][-3::]=False

    # if gen_dic['star_name']=='WASP121':
    #     # #    - exclusion of planet-contaminated range
    #     # for vis in []:
    #     #     cond_fit_dic['HARPS'][vis] &= ((data_load[inst][vis]['cen_ph']<= -0.008)   |  (data_load[inst][vis]['cen_ph']>= 0.02 ) )


    if gen_dic['star_name']=='GJ436':
        glob_fit_dic['IntrProp']['idx_in_fit']={
            # 'ESPRESSO':{'20190228':range(1,9),'20190429':range(1,9)},
            'HARPS':{'20070509':range(3,9)},
            'HARPN':{'20160318':range(1,8),'20160411':range(1,8)}
            }
    if gen_dic['star_name']=='MASCARA1':
        glob_fit_dic['IntrProp']['idx_in_fit']={'ESPRESSO':{'20190714':range(3,69),'20190811':range(2,68)}}
    #RM survey
    if gen_dic['star_name']=='HAT_P3':glob_fit_dic['IntrProp']['idx_in_fit']={'HARPN':{'20200130':range(1,8)}}
    elif gen_dic['star_name']=='HAT_P33':glob_fit_dic['IntrProp']['idx_in_fit']={'HARPN':{'20191204':range(1,33)}}
    elif gen_dic['star_name']=='WASP107':glob_fit_dic['IntrProp']['idx_in_fit']={'CARMENES_VIS':{'20180224':range(1,9)},'HARPS':{'20140406':range(1,11),'20180201':range(1,12),'20180313':range(1,12)}}
    elif gen_dic['star_name']=='WASP166':glob_fit_dic['IntrProp']['idx_in_fit']={'HARPS':{'20170114':range(1,39),'20170304':range(1,37),'20170315':range(1,33)}}
    elif gen_dic['star_name']=='HAT_P11':glob_fit_dic['IntrProp']['idx_in_fit']={'HARPN':{'20150913':range(2,26),'20151101':range(2,25)},'CARMENES_VIS':{'20170807':range(1,17),'20170812':range(2,18)}}
    elif gen_dic['star_name']=='WASP156':glob_fit_dic['IntrProp']['idx_in_fit']={'CARMENES_VIS':{'20190928':range(1,7),'20191025':range(1,6)}}

    if gen_dic['star_name']=='HD209458':
        # for vis in ['2019-09-11','2019-07-20']:
        #     #in-transit seul
        #     cond_fit_dic['ESPRESSO'][vis] &= ((data_load[inst][vis]['cen_ph']> -0.012)   &  (data_load[inst][vis]['cen_ph']< 0.012 ) )
            
        # #     #Pour faire le fit avec toujours la même série de points
        # #     cond_fit_dic['ESPRESSO'][vis] &= ((data_load[inst][vis]['cen_ph']> -0.017)   &  (data_load[inst][vis]['cen_ph']< 0.017 ) )
        glob_fit_dic['IntrProp']['idx_in_fit']={'ESPRESSO':{'20190720':range(2,46),'20190911':range(2,46)}}   #White CCFs
        # glob_fit_dic['IntrProp']['idx_in_fit']={'ESPRESSO':{'20190720':range(3,44),'20190911':range(3,44)}}   #Na CCF        
        
        
    if gen_dic['star_name']=='WASP76':
        glob_fit_dic['IntrProp']['idx_in_fit']={'ESPRESSO':{'20180902':list(np.delete(np.arange(21),[0,10,11,12,13,20])),'20181030':list(np.delete(np.arange(39),[0,1,18,19,20,21,22,23,24,25,37,38]))}}



    #Fitted property
    glob_fit_dic['IntrProp']['prop'] = 'ctrst'


    #Coordinate
    if gen_dic['star_name']=='MASCARA1':glob_fit_dic['IntrProp']['dim_fit']='phase'  
    if gen_dic['star_name'] in ['HD209458','WASP76']:glob_fit_dic['IntrProp']['dim_fit']='r_proj'  


    #Variation
    if gen_dic['star_name'] in ['WASP107','HAT_P11','HD209458','WASP76']:glob_fit_dic['IntrProp']['pol_mode']='modul'    
    


    #Fixed/variable properties
#     if gen_dic['main_pl']=='WASP_8b':        

#         p_start.add_many(('veq',1.9, True , 0. , None, None), 
#                         ('alpha_rot',0., False , -1. , 1., None),  
#                         ('beta_rot',0., False , None , None, None),  
#                         ('cos_istar',np.cos(90.*np.pi/180.), False , -1. , 1., None),  
#                         ('lambda_rad__'+pl_loc,-140*np.pi/180., True , -np.pi , np.pi, None),  
#                         ('c1_CB',0., False , None , None, None),  
#                         ('c2_CB',0., False , None , None, None),  
#                         ('c3_CB',0., False , None , None, None))


#     elif gen_dic['main_pl']=='55Cnc_e':        

#         p_start.add_many(('veq',1.152, True , 0. , None, None), 
#                         ('alpha_rot',0., False , -1. , 1., None),  
#                         ('beta_rot',0., False , None , None, None),  
#                         ('cos_istar',np.cos(90.*np.pi/180.), False , -1. , 1., None),  
#                         ('lambda_rad__'+pl_loc,72.3021645*np.pi/180., True , -np.pi , np.pi, None),  
#                         ('c1_CB',0., False , None , None, None),  
#                         ('c2_CB',0., False , None , None, None),  
#                         ('c3_CB',0., False , None , None, None))    

#         if fit_dic['fit_mod']=='mcmc': 
#             fixed_args['varpar_priors'].update({  
#                 'veq':{'mod':'uf','low':0.,'high':1e4}, 
#                 'alpha_rot':{'mod':'uf','low':-1.,'high':1.}, 
#                 'beta_rot':{'mod':'uf','low':-10.,'high':1.},  
#                 'lambda_rad__'+pl_loc:{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
#                 'c1_CB':{'mod':'uf','low':-1e5,'high':1e5}, 
#                 'c2_CB':{'mod':'uf','low':-1e5,'high':1e5}, 
#                 'c3_CB':{'mod':'uf','low':-1e5,'high':1e5}
#                  })     

#             fit_dic['uf_bd'].update({
#                 'veq':[0.,2.],
#                 'alpha_rot':[0.,1.],
#                 'beta_rot':[0.,1.],
#                 'lambda_rad__'+pl_loc:[-0.5*np.pi,0.5*np.pi],
#                 'c1_CB':[-1.,1.], 
#                 'c2_CB':[-1.,1.], 
#                 'c3_CB':[-1.,1.]
#                  })

#     elif gen_dic['main_pl']=='Kelt9b':    

#         p_start.add_many(('veq',112.9  , True , 0. , None, None), 
#                         ('alpha_rot',0., True , -1. , 1., None),  
#                         ('beta_rot',0., False , None , None, None),  
#                         ('cos_istar',np.cos(90.*np.pi/180.), True , -1. , 1., None),  
#                         ('lambda_rad__'+pl_loc,-86.2*np.pi/180., True , -np.pi , np.pi, None),  
#                         ('c1_CB',0., False , None , None, None),  
#                         ('c2_CB',0., False , None , None, None),  
#                         ('c3_CB',0., False , None , None, None))    

#         if fit_dic['fit_mod']=='mcmc': 
#             fixed_args['varpar_priors'].update({
#                 'veq':{'mod':'uf','low':0.,'high':1e4}, 
#                 'cos_istar':{'mod':'uf','low':-1.,'high':1.},
#                 'alpha_rot':{'mod':'uf','low':-1.,'high':1.},  
#                 'lambda_rad__'+pl_loc:{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},

# #                    'veq':{'mod':'uf','low':0.,'high':1e4}, 
# ##                    'cos_istar':{'mod':'uf','low':-1.,'high':-0.5},
# #                    'cos_istar':{'mod':'uf','low':-0.5,'high':1.},
# #                    'alpha_rot':{'mod':'uf','low':-1.,'high':1.},  
# #                    'lambda_rad__'+pl_loc:{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
#                  }) 

#             fit_dic['uf_bd'].update({
#                 'veq':[100.,240.],
#                 'cos_istar':[-0.9,0.9],
#                 'alpha_rot':[-0.3,0.3],
#                 'lambda_rad__'+pl_loc:[-1.55,-1.4],    #[-100.*np.pi/180.,-60.*np.pi/180.],

# #                    'veq':[130.,220.],
# ##                    'cos_istar':[-1.,-0.6],
# #                    'cos_istar':[-0.4,1.],
# #                    'alpha_rot':[-0.1,0.4],
# #                    'lambda_rad__'+pl_loc:[-1.5,-1.45],    

#                  })
    
#     elif gen_dic['main_pl']=='WASP121b':    
    
#         p_start.add_many(('veq',9.7705730e+01, True , 0. , None, None),
#                         ('inclin_rad__'+pl_loc,88.489694*np.pi/180.  , True , 0. , np.pi/2., None),
# #                        ('aRs__'+pl_loc,3.8219354549605975 , False , 0. , None, None),
# #                        ('aRs__'+pl_loc,3.821572414e+00 , False , 0. , None, None),
#                         ('aRs__'+pl_loc,3.8131, False , 0. , None, None),
#                         ('alpha_rot',7.9316453e-02, True , -1. , 1., None),  
#                         ('beta_rot',0., False , None , None, None),  
#                         ('cos_istar',np.cos(8.1384764*np.pi/180.), True , -1. , 1., None),  
#                         ('lambda_rad__'+pl_loc,8.7201724e+01*np.pi/180., True , -np.pi , np.pi, None),  
#                         ('c1_CB',0., False , None , None, None),  
#                         ('c2_CB',0., False , None , None, None),  
#                         ('c3_CB',0., False , None , None, None))    

#         if fit_dic['fit_mod']=='mcmc':  
#             #La distribution d'inclinaisons de TESS pique a 90 deg. On a pris comme valeur a 1sigma celle qui donne 68.27% des samples dans x-90
#             #Ici pour ne pas imposer de boundary dure au MCMC, on le laisse aller au dela de 90, mais du coup on definit le prior comme
#             #une distribution normale centree sur 90 avec comme std la moitie de (x-90)
                            
#             fixed_args['varpar_priors'].update({  
# #                    'veq':{'mod':'uf','low':0.,'high':1e4}, 
#                 'veq':{'mod':'uf','low':65.2780555815988,'high':1e4}, 
# #                    'veq':{'mod':'gauss','val':65.2780555815988,'s_val':1.3431698679341317},  
#                 'inclin_rad__'+pl_loc:{'mod':'gauss','val':0.5*np.pi,'s_val':0.5*0.9727354641270551*np.pi/180.},               
# #                    'aRs__'+pl_loc:{'mod':'gauss','val':3.8219354549605975,'s_val':0.007595549979677596}, 
#                 'cos_istar':{'mod':'uf','low':-1.,'high':1.},
# #                    'cos_istar':{'mod':'uf','low':0.,'high':1.},
# #                    'cos_istar':{'mod':'uf','low':np.cos(11.98916865744082*np.pi/180.),'high':1.},
# #                    'cos_istar':{'mod':'uf','low':-1.,'high':np.cos((180.-11.98916865744082)*np.pi/180.)},
#                 'alpha_rot':{'mod':'uf','low':-1.,'high':1.},  
#                 'lambda_rad__'+pl_loc:{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
#                  }) 

#             fit_dic['uf_bd'].update({
# #                    'veq':[1.,100.],
# #                    'veq':[66.,100.],
#                 'veq':[10.,150.],
# #                    'aRs__'+pl_loc:[3.81,3.83],
#                 'inclin_rad__'+pl_loc:[1.55,1.57],
#                 'cos_istar':[-1.,1.],
# #                    'cos_istar':[np.cos(11.98916865744082*np.pi/180.),1.],
# #                    'cos_istar':[-1.,np.cos((180.-11.98916865744082)*np.pi/180.)],
#                 'alpha_rot':[-1.,1.],
#                 'lambda_rad__'+pl_loc:[1.52,1.58],
#                  })
            
#             #Prior constraints
#             fixed_args['prior_list']+=['vsini']
#             fixed_args['vsini_pr']=13.56         #WASP-121b, Delrez+2016
#             fixed_args['sig_vsini_pr']=0.7
        


#     elif gen_dic['main_pl']=='Corot7b':    
    
#         p_start.add_many(('veq',1., True , 0. , 10., None),  
#                         ('lambda_rad__'+pl_loc,0.*np.pi/180., True , -np.pi , np.pi, None))  

#         if fit_dic['fit_mod']=='mcmc':
#             fixed_args['varpar_priors'].update({'veq':{'mod':'uf','low':0.,'high':10.},'lambda_rad__'+pl_loc:{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}}) 

#             fit_dic['uf_bd'].update({
#                 'veq':[0.1,3.],
#                 'lambda_rad__'+pl_loc:[-2.*np.pi,2.*np.pi]})

#     elif gen_dic['main_pl']=='GJ9827d':    
    
#         p_start.add_many(('veq',1., True , 0. , 10., None),  
#                         ('lambda_rad__'+pl_loc,0.*np.pi/180., True , -np.pi , np.pi, None))  

#         if fit_dic['fit_mod']=='mcmc':
#             fixed_args['varpar_priors'].update({'veq':{'mod':'uf','low':0.,'high':10.},'lambda_rad__'+pl_loc:{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}}) 

#             fit_dic['uf_bd'].update({
#                 'veq':[0.1,3.],
#                 'lambda_rad__'+pl_loc:[-2.*np.pi,2.*np.pi]})

#     elif gen_dic['main_pl']=='GJ9827b':    
    
#         p_start.add_many(('veq',1., True , 0. , 10., None),  
#                         ('lambda_rad__'+pl_loc,0.*np.pi/180., True , -np.pi , np.pi, None))  

#         if fit_dic['fit_mod']=='mcmc':
#             fixed_args['varpar_priors'].update({'veq':{'mod':'uf','low':0.,'high':10.},'lambda_rad__'+pl_loc:{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}}) 

#             fit_dic['uf_bd'].update({
#                 'veq':[0.1,3.],
#                 'lambda_rad__'+pl_loc:[-2.*np.pi,2.*np.pi]})

#     if gen_dic['main_pl']==['HD3167_b']:

#         p_start.add_many(('veq',2., True , 0. , 10., None),  
#                         ('lambda_rad__HD3167_b',0.*np.pi/180., True , -np.pi , np.pi, None),
#                         ('inclin_rad__HD3167_b',  planets_params['HD3167_b']['inclin_rad'] , True & False   , 0. , np.pi/2., None),
#                         ('aRs__HD3167_b',planets_params['HD3167_c']['aRs'], True & False  , 0. , None, None))
                       
#         if fit_dic['fit_mod']=='mcmc':
#             fixed_args['varpar_priors'].update({
#                 # 'veq':{'mod':'uf','low':0.,'high':5.}, 
#                 'veq':{'mod':'gauss','val':2.3687171817e+00,'s_val':4.1619648752e-01},        #value from HRRM fit to HD3167c
#                 # 'veq':{'mod':'gauss','val':1.7,'s_val':0.5},                  #value from spectroscopic fits
#                 'lambda_rad__HD3167_b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
#                 # 'inclin_rad__HD3167_b':{'mod':'dgauss','val':83.4*np.pi/180.,'s_val_low':7.7*np.pi/180.,'s_val_high':4.6*np.pi/180.},               
#                 # 'aRs__HD3167_b':{'mod':'dgauss','val':4.082,'s_val_low':0.986,'s_val_high':0.464},                    
#                 }) 

#             fit_dic['uf_bd'].update({
#                 'veq':[0.1,3.],
#                 'lambda_rad__HD3167_b':[-2.*np.pi,2.*np.pi],
#                 # 'inclin_rad__HD3167_b':[75*np.pi/180.,89.9*np.pi/180.],
#                 # 'aRs__HD3167_b':[3.,5.],
#                  })

#         #Prior constraints
#         if p_start['inclin_rad__HD3167_b'].vary:fixed_args['prior_list']+=['cosi','b'] 
#         if p_start['aRs__HD3167_b'].vary:fixed_args['prior_list']+=['b']  

#     if gen_dic['main_pl']==['HD3167_c']: 

#         p_start.add_many(('veq',2., True , 0. , None, None), 
#                         ('lambda_rad__HD3167_c',-100.*np.pi/180., True , -np.pi , np.pi, None),
#                         ('inclin_rad__HD3167_c', planets_params['HD3167_c']['inclin_rad']  , True & False  , 0. , np.pi/2., None),
#                         ('aRs__HD3167_c',planets_params['HD3167_c']['aRs'], True  & False , 0. , None, None))
        
#         if fit_dic['fit_mod']=='mcmc': 
#             fixed_args['varpar_priors'].update({  
#                 'veq':{'mod':'uf','low':0.,'high':1e4}, 
#                 # 'veq':{'mod':'gauss','val':1.7,'s_val':0.5},                  #value from spectroscopic fits
#                 'lambda_rad__HD3167_c':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
#                 # 'inclin_rad__HD3167_c':{'mod':'dgauss','val':89.3*np.pi/180.,'s_val_low':0.96*np.pi/180.,'s_val_high':0.5*np.pi/180.},               
#                 # 'aRs__HD3167_c':{'mod':'dgauss','val':40.323,'s_val_low':12.622,'s_val_high':5.549},                 
#                 })  

#             fit_dic['uf_bd'].update({
#                 'veq':[1.,4.],
#                 'lambda_rad__HD3167_c':[-130.*np.pi/180.,-90.*np.pi/180.],
#                 # 'inclin_rad__HD3167_c':[88*np.pi/180.,90.*np.pi/180.],
#                 # 'aRs__HD3167_c':[28.,46.],
#                  })

#         #Prior constraints
#         if p_start['inclin_rad__HD3167_c'].vary:fixed_args['prior_list']+=['cosi','b'] 
#         if p_start['aRs__HD3167_c'].vary:fixed_args['prior_list']+=['b']    
               
#                 # fixed_args['prior_list']=['contrast']


#     if ('HD3167_b' in gen_dic['main_pl']) and ('HD3167_c' in gen_dic['main_pl']):

#         p_start.add_many(('veq',2., True , 0. , None, None), 
#                         ('cos_istar',0., True , -1. , 1., None), 
#                         ('lambda_rad__HD3167_b',-100.*np.pi/180., True , -np.pi , np.pi, None),
#                         ('lambda_rad__HD3167_c',-100.*np.pi/180., True , -np.pi , np.pi, None))
        
#         if fit_dic['fit_mod']=='mcmc': 
#             fixed_args['varpar_priors'].update({  
#                 'veq':{'mod':'uf','low':0.,'high':1e4}, 
#                 'cos_istar':{'mod':'uf','low':-1.,'high':1.},
#                 'lambda_rad__HD3167_b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
#                 'lambda_rad__HD3167_c':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},           
#                 })  

#             fit_dic['uf_bd'].update({
#                 'veq':[1.,4.],
#                 'cos_istar':[-1.,1.],
#                 'lambda_rad__HD3167_b':[-50.*np.pi/180.,50.*np.pi/180.],
#                 'lambda_rad__HD3167_c':[-130.*np.pi/180.,-90.*np.pi/180.],
#                  })        

               
#                 # fixed_args['prior_list']=['contrast']


#     if (gen_dic['main_pl']==['TOI858b']):

#         p_start.add_many(('veq',5., True , 0. , None, None), 
#                         ('cos_istar',0., False , -1. , 1., None), 
#                         ('lambda_rad__TOI858b',90.*np.pi/180., True , -np.pi , np.pi, None))
        
#         if fit_dic['fit_mod']=='mcmc': 
#             fixed_args['varpar_priors'].update({  
#                 'veq':{'mod':'uf','low':0.,'high':1e4}, 
#                 'lambda_rad__TOI858b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},         
#                 })  

#             fit_dic['uf_bd'].update({
#                 'veq':[1.,4.],
#                 'lambda_rad__TOI858b':[0.*np.pi/180.,180.*np.pi/180.],
#                  })         
    
    if gen_dic['star_name']=='GJ436':
        glob_fit_dic['IntrProp']['mod_prop']={'veq':{'vary':True,'guess':100.,'bd':[0.1,0.4]},
                                     'lambda_rad__GJ436_b':{'vary':True,'guess':100.*np.pi/180.,'bd':[70.*np.pi/180.,140.*np.pi/180.]},   
                                     'alpha_rot':{'vary':True & False,'guess':0.,'bd':[0.,1.]},           
                                     'cos_istar':{'vary':True & False,'guess':0.,'bd':[-1.,1.]},   
        
                                     # 'inclin_rad__GJ436_b':{'vary':True,'guess':planets_params['GJ436_b']['inclin_rad'],'bd':[0. , np.pi/2.]},
                                     # 'aRs__GJ436_b':{'vary':True,'guess':planets_params['GJ436_b']['aRs'],'bd':[0.,1.]}}
                                     }

    if gen_dic['star_name']=='MASCARA1':
        if glob_fit_dic['IntrProp']['prop'] == 'rv':        
            glob_fit_dic['IntrProp']['mod_prop']={'veq':{'vary':True,'guess':0.2,'bd':[0.1,0.4]},
                                         'lambda_rad__MASCARA1b':{'vary':True,'guess':100.*np.pi/180.,'bd':[70.*np.pi/180.,140.*np.pi/180.]}}           

        if glob_fit_dic['IntrProp']['prop'] == 'ctrst':
            glob_fit_dic['IntrProp']['mod_prop']={
                        'ctrst_ord0__ISESPRESSO_VS20190714':{'vary':True ,'guess':0.47629,'bd':[0.1,0.6]},
                        'ctrst_ord1__ISESPRESSO_VS20190714':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'ctrst_ord2__ISESPRESSO_VS20190714':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'ctrst_ord3__ISESPRESSO_VS20190714':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'ctrst_ord4__ISESPRESSO_VS20190714':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'ctrst_ord0__ISESPRESSO_VS20190811':{'vary':True ,'guess':0.39768,'bd':[0.1,0.6]},
                        'ctrst_ord1__ISESPRESSO_VS20190811':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'ctrst_ord2__ISESPRESSO_VS20190811':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'ctrst_ord3__ISESPRESSO_VS20190811':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'ctrst_ord4__ISESPRESSO_VS20190811':{'vary':True ,'guess':0.,'bd':[-1.,1.]}
                            }
        if glob_fit_dic['IntrProp']['prop'] == 'FWHM':
            glob_fit_dic['IntrProp']['mod_prop']={
                        'FWHM_ord0__ISESPRESSO_VS20190714':{'vary':True ,'guess':20.,'bd':[0.1,0.6]},
                        'FWHM_ord1__ISESPRESSO_VS20190714':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'FWHM_ord2__ISESPRESSO_VS20190714':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'FWHM_ord3__ISESPRESSO_VS20190714':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'FWHM_ord4__ISESPRESSO_VS20190714':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'FWHM_ord0__ISESPRESSO_VS20190811':{'vary':True ,'guess':20.,'bd':[0.1,0.6]},
                        'FWHM_ord1__ISESPRESSO_VS20190811':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'FWHM_ord2__ISESPRESSO_VS20190811':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'FWHM_ord3__ISESPRESSO_VS20190811':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'FWHM_ord4__ISESPRESSO_VS20190811':{'vary':True ,'guess':0.,'bd':[-1.,1.]}
                            }

    elif gen_dic['star_name']=='V1298tau':
        if glob_fit_dic['IntrProp']['prop'] == 'rv':glob_fit_dic['IntrProp']['mod_prop']={
                        'veq':{'vary':True,'guess':0.2,'bd':[0.1,0.4]},
                        'lambda_rad__V1298tau_b':{'vary':True,'guess':100.*np.pi/180.,'bd':[70.*np.pi/180.,140.*np.pi/180.]}}           
        if glob_fit_dic['IntrProp']['prop'] == 'ctrst':glob_fit_dic['IntrProp']['mod_prop']={
                        'ctrst_ord0__ISHARPN_VS20200128':{'vary':True ,'guess':0.47629,'bd':[0.1,0.6]},
                        'ctrst_ord0__ISHARPN_VS20201207':{'vary':True ,'guess':0.39768,'bd':[0.1,0.6]}}
        if glob_fit_dic['IntrProp']['prop'] == 'FWHM':glob_fit_dic['IntrProp']['mod_prop']={
                        'FWHM_ord0__ISHARPN_VS20200128':{'vary':True ,'guess':0.47629,'bd':[0.1,0.6]},
                        'FWHM_ord0__ISHARPN_VS20201207':{'vary':True ,'guess':0.39768,'bd':[0.1,0.6]}}

    #RM survey
    elif gen_dic['star_name']=='HAT_P3':
        if glob_fit_dic['IntrProp']['prop'] == 'rv':glob_fit_dic['IntrProp']['mod_prop']={
                        'veq':{'vary':True,'guess':0.2,'bd':[0.1,0.4],'physical':True},
                        'lambda_rad__plHAT_P3b':{'vary':True,'guess':90.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True}}           
        if glob_fit_dic['IntrProp']['prop'] == 'ctrst':glob_fit_dic['IntrProp']['mod_prop']={
                        'ctrst_ord0__ISHARPN_VS20200130':{'vary':True ,'guess':0.39768,'bd':[0.1,0.6]},
                        'ctrst_ord1__ISHARPN_VS20200130':{'vary':True ,'guess':0.,'bd':[-1.,1.]},    
                        'ctrst_ord2__ISHARPN_VS20200130':{'vary':True ,'guess':0.,'bd':[-1.,1.]},      #best
                        # 'ctrst_ord3__ISHARPN_VS20200130':{'vary':True ,'guess':0.,'bd':[-1.,1.]},                      
                        }
        if glob_fit_dic['IntrProp']['prop'] == 'FWHM':glob_fit_dic['IntrProp']['mod_prop']={
                        'FWHM_ord0__ISHARPN_VS20200130':{'vary':True ,'guess':7.,'bd':[0.,10.]},
                        'FWHM_ord1__ISHARPN_VS20200130':{'vary':True ,'guess':0.,'bd':[-1.,1.]}, 
                        'FWHM_ord2__ISHARPN_VS20200130':{'vary':True ,'guess':0.,'bd':[-1.,1.]}, 
                        'FWHM_ord3__ISHARPN_VS20200130':{'vary':True ,'guess':0.,'bd':[-1.,1.]}, 
                        }
        
    elif gen_dic['star_name']=='HAT_P33':
        if glob_fit_dic['IntrProp']['prop'] == 'rv':glob_fit_dic['IntrProp']['mod_prop']={
                        'veq':{'vary':True,'guess':16.,'bd':[0.1,0.4],'physical':True},
                        'lambda_rad__plHAT_P33b':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True}}          
        
        if glob_fit_dic['IntrProp']['prop'] == 'ctrst':glob_fit_dic['IntrProp']['mod_prop']={
                        'ctrst_ord0__ISHARPN_VS20191204':{'vary':True ,'guess':0.39768,'bd':[0.1,0.6]}, 
                        'ctrst_ord1__ISHARPN_VS20191204':{'vary':True ,'guess':0.,'bd':[-1.,1.]},   
                        'ctrst_ord2__ISHARPN_VS20191204':{'vary':True ,'guess':0.,'bd':[-1.,1.]},   
                        # 'ctrst_ord3__ISHARPN_VS20191204':{'vary':True ,'guess':0.,'bd':[-1.,1.]},                 
                        }
    
        if glob_fit_dic['IntrProp']['prop'] == 'FWHM':glob_fit_dic['IntrProp']['mod_prop']={
                        'FWHM_ord0__ISHARPN_VS20191204':{'vary':True ,'guess':7.,'bd':[0.,10.]}, 
                        'FWHM_ord1__ISHARPN_VS20191204':{'vary':True ,'guess':0.,'bd':[-1.,1.]},  
                        # 'FWHM_ord2__ISHARPN_VS20191204':{'vary':True ,'guess':0.,'bd':[-1.,1.]},                 
                        }    

    elif gen_dic['star_name']=='WASP107':
        if glob_fit_dic['IntrProp']['prop'] == 'rv':glob_fit_dic['IntrProp']['mod_prop']={       
                        'veq':{'vary':True ,'guess':4.9381378895e-01,'bd':[0.1,10.],'physical':True},
                        'lambda_rad__plWASP107b':{'vary':True ,'guess':-1.5864e+02*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True},
                        # 'c1_CB':{'vary':True,'guess':0.1,'bd':[-1.,1.],'physical':True},
                        }   

        if glob_fit_dic['IntrProp']['prop'] == 'ctrst':glob_fit_dic['IntrProp']['mod_prop']={
                        # #Independent C for each dataset
                        # 'ctrst_ord0__ISCARMENES_VIS_VS20180224':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},
                        # # 'ctrst_ord1__ISCARMENES_VIS_VS20180224':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # 'ctrst_ord0__ISHARPS_VS20140406':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},                 
                        # # 'ctrst_ord1__ISHARPS_VS20140406':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # 'ctrst_ord0__ISHARPS_VS20180201':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},                  
                        # 'ctrst_ord1__ISHARPS_VS20180201':{'vary':True ,'guess':0.,'bd':[-1.,1.]},                 
                        # # 'ctrst_ord2__ISHARPS_VS20180201':{'vary':True ,'guess':0.,'bd':[-1.,1.]}, 
                        # 'ctrst_ord0__ISHARPS_VS20180313':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},
                        # # 'ctrst_ord1__ISHARPS_VS20180313':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        
                        # #Common C for HARPS datasets
                        # 'ctrst_ord0__ISCARMENES_VIS_VS20180224':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},
                        # 'ctrst_ord1__ISCARMENES_VIS_VS20180224':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # # 'ctrst_ord2__ISCARMENES_VIS_VS20180224':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # 'ctrst_ord0__ISHARPS_VS_':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},
                        # 'ctrst_ord1__ISHARPS_VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # # 'ctrst_ord2__ISHARPS_VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        
                        # #Common C for all datasets
                        # 'ctrst_ord0__IS__VS_':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},                       
                        # 'ctrst_ord1__IS__VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},  
                        #}

                        #Common modulation for all datasets, common C0 for each instrument
                        'ctrst_ord0__ISCARMENES_VIS_VS_':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},
                        'ctrst_ord0__ISHARPS_VS_':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},                         
                        
                        'ctrst_ord1__IS__VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # 'ctrst_ord2__IS__VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        }  


        if glob_fit_dic['IntrProp']['prop'] == 'FWHM':glob_fit_dic['IntrProp']['mod_prop']={
                        #Common C for HARPS datasets
                        'FWHM_ord0__ISCARMENES_VIS_VS20180224':{'vary':True ,'guess':7.,'bd':[0.1,10.]},
                        # 'FWHM_ord1__ISCARMENES_VIS_VS20180224':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'FWHM_ord0__ISHARPS_VS_':{'vary':True ,'guess':7.,'bd':[0.1,10.]},
                        # 'FWHM_ord1__ISHARPS_VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        }

        
    elif gen_dic['star_name']=='WASP166':
        if glob_fit_dic['IntrProp']['prop'] == 'rv':glob_fit_dic['IntrProp']['mod_prop']={       
                        'veq':{'vary':True ,'guess':5.,'bd':[0.1,10.],'physical':True},
                        'lambda_rad__plWASP166b':{'vary':True ,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True},
                        }          
        
        if glob_fit_dic['IntrProp']['prop'] == 'ctrst':glob_fit_dic['IntrProp']['mod_prop']={
                        # #Independent C for each dataset
                        # 'ctrst_ord0__ISHARPS_VS20170114':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},                 
                        # # 'ctrst_ord1__ISHARPS_VS20170114':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # 'ctrst_ord0__ISHARPS_VS20170304':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},                  
                        # # 'ctrst_ord1__ISHARPS_VS20170304':{'vary':True ,'guess':0.,'bd':[-1.,1.]}, 
                        # 'ctrst_ord0__ISHARPS_VS20170315':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},
                        # # 'ctrst_ord1__ISHARPS_VS20170315':{'vary':True ,'guess':0.,'bd':[-1.,1.]},  
                        # }

                        #Common C for HARPS datasets
                        'ctrst_ord0__ISHARPS_VS_':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},
                        'ctrst_ord1__ISHARPS_VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'ctrst_ord2__ISHARPS_VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        }

        if glob_fit_dic['IntrProp']['prop'] == 'FWHM':glob_fit_dic['IntrProp']['mod_prop']={
                        #Common for HARPS datasets
                        'FWHM_ord0__ISHARPS_VS_':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},
                        # 'FWHM_ord1__ISHARPS_VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # 'FWHM_ord2__ISHARPS_VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        }        
        
    elif gen_dic['star_name']=='HAT_P11':
        if glob_fit_dic['IntrProp']['prop'] == 'rv':glob_fit_dic['IntrProp']['mod_prop']={       
                        'veq':{'vary':True ,'guess':5.,'bd':[0.1,10.],'physical':True},
                        'lambda_rad__plHAT_P11b':{'vary':True ,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True},
                        }    

        if glob_fit_dic['IntrProp']['prop'] == 'ctrst':glob_fit_dic['IntrProp']['mod_prop']={
                
                        # #Independent C for each dataset
                        # 'ctrst_ord0__ISCARMENES_VIS_VS20170807':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},
                        # # 'ctrst_ord1__ISCARMENES_VIS_VS20170807':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # # 'ctrst_ord2__ISCARMENES_VIS_VS20170807':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # # 'ctrst_ord3__ISCARMENES_VIS_VS20170807':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # 'ctrst_ord0__ISCARMENES_VIS_VS20170812':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},                        
                        # # 'ctrst_ord1__ISCARMENES_VIS_VS20170812':{'vary':True ,'guess':0.,'bd':[-1.,1.]},                       
                        # # 'ctrst_ord2__ISCARMENES_VIS_VS20170812':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # 'ctrst_ord0__ISHARPN_VS20150913':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},                 
                        # # 'ctrst_ord1__ISHARPN_VS20150913':{'vary':True ,'guess':0.,'bd':[-1.,1.]},                 
                        # # 'ctrst_ord2__ISHARPN_VS20150913':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # 'ctrst_ord0__ISHARPN_VS20151101':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},                  
                        # # 'ctrst_ord1__ISHARPN_VS20151101':{'vary':True ,'guess':0.,'bd':[-1.,1.]},                   
                        # # 'ctrst_ord2__ISHARPN_VS20151101':{'vary':True ,'guess':0.,'bd':[-1.,1.]},      
                        # }
                        
                        # #Common C for each dataset
                        # 'ctrst_ord0__ISCARMENES_VIS_VS_':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},
                        # 'ctrst_ord1__ISCARMENES_VIS_VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # 'ctrst_ord2__ISCARMENES_VIS_VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        
                        # 'ctrst_ord0__ISHARPN_VS_':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},                 
                        # 'ctrst_ord1__ISHARPN_VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},                 
                        # 'ctrst_ord2__ISHARPN_VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},  
                        # }        


                        #Common modulation for all datasets, common C0 for each instrument
                        'ctrst_ord0__ISCARMENES_VIS_VS_':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},
                        'ctrst_ord0__ISHARPN_VS_':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},                         
                        
                        'ctrst_ord1__IS__VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        'ctrst_ord2__IS__VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},

                        }           
        
                        # #Common C for all datasets
                        # 'ctrst_ord0__IS__VS_':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},                       
                        # 'ctrst_ord1__IS__VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},                       
                        # 'ctrst_ord2__IS__VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},  
                        # }


        if glob_fit_dic['IntrProp']['prop'] == 'FWHM':glob_fit_dic['IntrProp']['mod_prop']={
                
                        # #Independent for each dataset
                        # 'FWHM_ord0__ISCARMENES_VIS_VS20170807':{'vary':True ,'guess':7.,'bd':[0.1,0.6]},
                        # 'FWHM_ord0__ISCARMENES_VIS_VS20170812':{'vary':True ,'guess':7.,'bd':[0.1,0.6]},  
                        # 'FWHM_ord0__ISHARPN_VS20150913':{'vary':True ,'guess':7.,'bd':[0.1,0.6]},  
                        # 'FWHM_ord0__ISHARPN_VS20151101':{'vary':True ,'guess':7.,'bd':[0.1,0.6]},       
                        # }                
                
                        #Common modulation per instru
                        'FWHM_ord0__ISCARMENES_VIS_VS_':{'vary':True ,'guess':7.,'bd':[0.1,0.6]},
                        # 'FWHM_ord1__ISCARMENES_VIS_VS_':{'vary':True ,'guess':0.,'bd':[0.1,0.6]},
                        # 'FWHM_ord2__ISCARMENES_VIS_VS_':{'vary':True ,'guess':0.,'bd':[0.1,0.6]},
                        'FWHM_ord0__ISHARPN_VS_':{'vary':True ,'guess':7.,'bd':[0.1,0.6]},        
                        # 'FWHM_ord1__ISHARPN_VS_':{'vary':True ,'guess':0.,'bd':[0.1,0.6]},          
                        # 'FWHM_ord2__ISHARPN_VS_':{'vary':True ,'guess':0.,'bd':[0.1,0.6]},   
                        }    

                

               


    elif gen_dic['star_name']=='WASP156'  :
    
        if glob_fit_dic['IntrProp']['prop'] == 'rv':glob_fit_dic['IntrProp']['mod_prop']={       
                        'veq':{'vary':True ,'guess':2.,'bd':[0.1,10.],'physical':True},
                        'lambda_rad__plWASP156b':{'vary':True ,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True},
                        }    
        
        if glob_fit_dic['IntrProp']['prop'] == 'ctrst':glob_fit_dic['IntrProp']['mod_prop']={
                
                        # #Independent for each dataset
                        # 'ctrst_ord0__ISCARMENES_VIS_VS20190928':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},
                        # # 'ctrst_ord1__ISCARMENES_VIS_VS20190928':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # # 'ctrst_ord2__ISCARMENES_VIS_VS20190928':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # 'ctrst_ord0__ISCARMENES_VIS_VS20191025':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},                        
                        # # 'ctrst_ord1__ISCARMENES_VIS_VS20191025':{'vary':True ,'guess':0.,'bd':[-1.,1.]},                         
                        # # 'ctrst_ord2__ISCARMENES_VIS_VS20191025':{'vary':True ,'guess':0.,'bd':[-1.,1.]},    
                        # }        

                        #Common for all datasets
                        'ctrst_ord0__IS__VS_':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},  
                        'ctrst_ord1__IS__VS_':{'vary':True ,'guess':0.,'bd':[0.1,0.6]}, 
                        }

        if glob_fit_dic['IntrProp']['prop'] == 'FWHM':glob_fit_dic['IntrProp']['mod_prop']={
                
                        # #Independent for each dataset
                        # 'FWHM_ord0__ISCARMENES_VIS_VS20190928':{'vary':True ,'guess':7.,'bd':[0.1,0.6]},
                        # # 'FWHM_ord1__ISCARMENES_VIS_VS20190928':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                        # 'FWHM_ord0__ISCARMENES_VIS_VS20191025':{'vary':True ,'guess':7.,'bd':[0.1,0.6]},                        
                        # # 'FWHM_ord1__ISCARMENES_VIS_VS20191025':{'vary':True ,'guess':0.,'bd':[-1.,1.]},     
                        # }        

                        #Common for all datasets
                        'FWHM_ord0__IS__VS_':{'vary':True ,'guess':7.,'bd':[0.1,0.6]},  
                        }

    if gen_dic['star_name']=='HD209458':    
    
        #                 ('inclin_rad__'+pl_loc,  main_pl_params['inclin_rad__'+pl_loc] , True & False , 0. , np.pi/2., None),
        #                 ('aRs__'+pl_loc,main_pl_params['aRs__'+pl_loc], True & False , 0. , None, None),
        #                 ('alpha_rot',0., True & False , -1. , 1., None),  
        #                 # ('alpha_rot',0., True  , -1. , 1., None),  
        #                 ('beta_rot',0., False , None , None, None),   

        if glob_fit_dic['IntrProp']['prop'] == 'rv':
            glob_fit_dic['IntrProp']['mod_prop']={
                        'veq':{'vary':False,'guess':4.2721415478,'bd':[3.,5.]},
                        'lambda_rad__plHD209458b':{'vary':False,'guess':1.0699092308*np.pi/180.,'bd':[-5.*np.pi/180.,5.*np.pi/180.]},
                        'c1_CB':{'vary':True,'guess':2.,'bd':[-1.,1.]}, 
                        # 'c2_CB':{'vary':True,'guess':0.,'bd':[-1.,1.]}, 
                        # 'alpha_rot':{'vary':True,'guess':0.,'bd':[0. , 1.]},  
                        # 'cos_istar':{'vary':True,'guess':np.cos(90.*np.pi/180.),'bd':[ -1. , 1.]},
                        'rv':{'vary':True,'guess':0.,'bd':[3.,5.]},
                        # 'rv_line_ord0__ISESPRESSO_VS20190720':{'vary':True,'guess':0.,'bd':[3.,5.]},
                        # 'rv_line_ord0__ISESPRESSO_VS20190911':{'vary':True,'guess':0.,'bd':[3.,5.]},
                        }           
        if glob_fit_dic['IntrProp']['prop'] == 'ctrst':
            glob_fit_dic['IntrProp']['mod_prop']={
                        # 'ctrst_ord0__IS__VS_':{'vary':True ,'guess':0.48,'bd':[0.1,0.6]},
                        'ctrst_ord0__ISESPRESSO_VS20190720':{'vary':True ,'guess':0.48,'bd':[0.1,0.6]},
                        'ctrst_ord0__ISESPRESSO_VS20190911':{'vary':True ,'guess':0.4,'bd':[0.1,0.6]},
                        # 'ctrst_ord1__ISESPRESSO_VS20190720':{'vary':True ,'guess':0.,'bd':[0.1,0.6]},
                        # 'ctrst_ord1__ISESPRESSO_VS20190911':{'vary':True ,'guess':0.,'bd':[0.1,0.6]}
                        'ctrst_ord1__IS__VS_':{'vary':True ,'guess':0.,'bd':[0.1,0.6]}
                        }
        if glob_fit_dic['IntrProp']['prop'] == 'FWHM':
            glob_fit_dic['IntrProp']['mod_prop']={
                        # 'FWHM_ord0__IS__VS_':{'vary':True ,'guess':15,'bd':[10.,20.]},   
                        'FWHM_ord0__ISESPRESSO_VS20190720':{'vary':True ,'guess':5.6752543392e+00,'bd':[7.,9.]},    #Best-fit RMR
                        'FWHM_ord0__ISESPRESSO_VS20190911':{'vary':True ,'guess':5.6428910034e+00,'bd':[7.,9.]},    #Best-fit RMR
                        # 'FWHM_ord1__ISESPRESSO_VS20190720':{'vary':True ,'guess':0.,'bd':[0.1,0.2]},
                        # 'FWHM_ord1__ISESPRESSO_VS20190911':{'vary':True ,'guess':0.,'bd':[0.1,0.2]}
                        'FWHM_ord1__IS__VS_':{'vary':False ,'guess':3.8883520955e-01,'bd':[0.1,0.2]},    #Best-fit RMR
                        # 'FWHM_ord2__IS__VS_':{'vary':True ,'guess':0.,'bd':[0.1,0.2]}
                        }            
            

    elif gen_dic['star_name']=='WASP76':    
    
        #                 #('inclin_rad__'+pl_loc,88.489694*np.pi/180.  , True & False , 0. , np.pi/2., None),
        #                 # ('aRs__'+pl_loc,3.8219354549605975 , False , 0. , None, None),
        #                 ('lambda_rad__'+pl_loc,8.7201724e+01*np.pi/180., True , -np.pi , np.pi, None))  

        # if fit_dic['fit_mod']=='mcmc':
        #     fixed_args['varpar_priors'].update({
        #         'veq':{'mod':'uf','low':0.,'high':1e4}, 
        #         'lambda_rad__'+pl_loc:{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
        #           }) 

        #     fit_dic['uf_bd'].update({
        #         'veq':[0.1,3.],
        #         'lambda_rad__'+pl_loc:[-5e-2,2e-2]
        #           })

        #     #Prior constraints
        #     fixed_args['prior_list']+=['cosi','b']

        if glob_fit_dic['IntrProp']['prop'] == 'rv':
            glob_fit_dic['IntrProp']['mod_prop']={
                        'veq':{'vary':True,'guess':1.,'bd':[3.,5.]},
                        'lambda_rad__plWASP76b':{'vary':True,'guess':0.*np.pi/180.,'bd':[-10.*np.pi/180.,10.*np.pi/180.]},
                        'c1_CB':{'vary':True,'guess':1.,'bd':[-1.,1.]}, 
                        # 'c2_CB':{'vary':True,'guess':0.,'bd':[-1.,1.]}, 
                        # 'alpha_rot':{'vary':True,'guess':0.,'bd':[0.1 , 0.9]},  
                        # 'cos_istar':{'vary':True,'guess':np.cos(90.*np.pi/180.),'bd':[ -0.9 , 0.9]},
                        }   

        if glob_fit_dic['IntrProp']['prop'] == 'ctrst':
            glob_fit_dic['IntrProp']['mod_prop']={
                        'ctrst_ord0__ISESPRESSO_VS20180902':{'vary':True ,'guess':0.48,'bd':[0.1,0.6]},
                        'ctrst_ord0__ISESPRESSO_VS20181030':{'vary':True ,'guess':0.4,'bd':[0.1,0.6]},
                        # 'ctrst_ord1__ISESPRESSO_VS20180902':{'vary':True ,'guess':0.,'bd':[0.1,0.6]},
                        # 'ctrst_ord1__ISESPRESSO_VS20181030':{'vary':True ,'guess':0.,'bd':[0.1,0.6]},
                        # 'ctrst_ord2__ISESPRESSO_VS20180902':{'vary':True ,'guess':0.,'bd':[0.1,0.6]},
                        # 'ctrst_ord2__ISESPRESSO_VS20181030':{'vary':True ,'guess':0.,'bd':[0.1,0.6]}
                        'ctrst_ord1__ISESPRESSO_VS_':{'vary':True ,'guess':0.,'bd':[0.1,0.6]},
                        'ctrst_ord2__ISESPRESSO_VS_':{'vary':True ,'guess':0.,'bd':[0.1,0.6]}
                        }
        if glob_fit_dic['IntrProp']['prop'] == 'FWHM':
            glob_fit_dic['IntrProp']['mod_prop']={
                        'FWHM_ord0__ISESPRESSO_VS20180902':{'vary':True ,'guess':8.,'bd':[7.,9.]},
                        'FWHM_ord0__ISESPRESSO_VS20181030':{'vary':True ,'guess':8.,'bd':[7.,9.]},
                        # 'FWHM_ord1__ISESPRESSO_VS20180902':{'vary':True ,'guess':0.,'bd':[0.1,0.2]},
                        # 'FWHM_ord1__ISESPRESSO_VS20181030':{'vary':True ,'guess':0.,'bd':[0.1,0.2]},
                        # 'FWHM_ord2__ISESPRESSO_VS20180902':{'vary':True ,'guess':0.,'bd':[0.1,0.2]},
                        # 'FWHM_ord2__ISESPRESSO_VS20181030':{'vary':True ,'guess':0.,'bd':[0.1,0.2]}
                        'FWHM_ord1__ISESPRESSO_VS_':{'vary':True ,'guess':0.,'bd':[0.1,0.2]},
                        'FWHM_ord2__ISESPRESSO_VS_':{'vary':True ,'guess':0.,'bd':[0.1,0.2]}
                        } 







    #Fitting mode 
    glob_fit_dic['IntrProp']['fit_mod']='mcmc'  # 'mcmc' 
    glob_fit_dic['IntrProp']['fit_mod']='chi2'  
    # glob_fit_dic['IntrProp']['fit_mod']='' 


    #Printing fits results
    glob_fit_dic['IntrProp']['verbose']=True & False
    


    #Priors on variable properties 
    if gen_dic['star_name']=='GJ436':    
        glob_fit_dic['IntrProp']['priors'].update({
            'veq':{'mod':'uf','low':0.,'high':10.},  
            'lambda_rad__GJ36_b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
    
            # 'alpha_rot':{'mod':'uf','low':0.,'high':1.},        
            # 'cos_istar':{'mod':'uf','low':-1.,'high':1.},        
            
            })    
    # elif gen_dic['star_name']=='HD209458':    
    #     glob_fit_dic['IntrProp']['priors'].update({
    #         'veq':{'mod':'uf','low':0.,'high':10.},  
    #         'lambda_rad__plHD209458b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
    #         'alpha_rot':{'mod':'uf','low':0.,'high':1.},  
    #         'cos_istar':{'mod':'uf','low':-1.,'high':1.},    
    #         })     
    elif gen_dic['star_name']=='WASP76':    
        glob_fit_dic['IntrProp']['priors'].update({
            'veq':{'mod':'uf','low':0.,'high':10.},  
            'lambda_rad__plWASP76b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
            'alpha_rot':{'mod':'uf','low':0.,'high':1.},  
            'cos_istar':{'mod':'uf','low':-1.,'high':1.},    
            })          
        
        
        
    

    #Derived properties
    if glob_fit_dic['IntrProp']['prop'] == 'rv':
        glob_fit_dic['IntrProp']['modif_list'] = ['vsini','psi','om','b','ip','istar_deg_conv','lambda_deg','c0','CB_ms']
        glob_fit_dic['IntrProp']['modif_list'] = ['vsini','lambda_deg','b','ip']  
        glob_fit_dic['IntrProp']['modif_list'] = ['vsini','lambda_deg','istar_deg_conv']   
        glob_fit_dic['IntrProp']['modif_list'] = ['vsini','lambda_deg']        
        glob_fit_dic['IntrProp']['modif_list'] = ['c0']    
        glob_fit_dic['IntrProp']['modif_list'] = [] 


    #Calculating/retrieving
    glob_fit_dic['IntrProp']['mcmc_run_mode']='use'

    #Walkers 
    #     if gen_dic['main_pl']==['HD3167_c']:  
    #         fit_dic['nwalkers'] = 12      #walkers              #HD3167c     vsini & lambda: 1.7 min pour 600 steps
    #         fit_dic['nsteps'] = 2000    #number of steps       
    #         fit_dic['nburn'] = 500      #burn-in  

    #     elif gen_dic['main_pl']=='WASP121b':  
    #         fit_dic['nwalkers'] = 100   #walkers              #WASP121b, SB, ip et aRs libres (35s for 1000 samples)
    #         fit_dic['nsteps'] = 4000  #number of steps       
    #         fit_dic['nburn'] = 1000   #burn-in  
    
    #         fit_dic['nwalkers'] = 300   #walkers              #WASP121b, DR
    #         fit_dic['nsteps'] = 5000  #number of steps       
    #         fit_dic['nburn'] = 1000   #burn-in  

    #     elif gen_dic['main_pl']=='Kelt9b': 
    #         fit_dic['nwalkers'] = 10   #walkers              #Kelt9b, SB
    #         fit_dic['nsteps'] = 5000  #number of steps       
    #         fit_dic['nburn'] = 300   #burn-in 
    
    #         fit_dic['nwalkers'] = 400   #walkers              #Kelt9b, DR full domain
    #         fit_dic['nsteps'] = 4000  #number of steps       
    #         fit_dic['nburn'] = 1000   #burn-in 
    # #        fit_dic['nwalkers'] = 200   #walkers              #Kelt9b, DR one domain
    # #        fit_dic['nsteps'] = 4000  #number of steps       
    # #        fit_dic['nburn'] = 1000   #burn-in 

    if gen_dic['star_name']=='HD209458':   
        glob_fit_dic['IntrProp']['mcmc_set']={'nwalkers':10,'nsteps':500,'nburn':100} 
    if gen_dic['star_name']=='WASP76':   
        glob_fit_dic['IntrProp']['mcmc_set']={'nwalkers':20,'nsteps':500,'nburn':100} 
    #     elif gen_dic['main_pl']=='GJ436_b':      #1000 = 10s      
    #         fit_dic['nwalkers'] = 30
    #         fit_dic['nsteps'] = 5000         
    #         fit_dic['nburn'] = 1000   

    #     elif gen_dic['main_pl']=='GJ9827d':      
    #         fit_dic['nwalkers'] = 10
    #         fit_dic['nsteps'] = 1000         
    #         fit_dic['nburn'] = 100   

    elif gen_dic['star_name']=='GJ436':     
        glob_fit_dic['IntrProp']['mcmc_set']={'nwalkers':20,'nsteps':2000,'nburn':500}  
        glob_fit_dic['IntrProp']['mcmc_set']={'nwalkers':10,'nsteps':1000,'nburn':500}  
        # glob_fit_dic['IntrProp']['mcmc_set']={'nwalkers':20,'nsteps':100,'nburn':0}  


    #omplex priors 
    if 'GJ436_b' in gen_dic['transit_pl']:
        
        #Prior sur le parametre donne par Maxted+
        if ('inclin_rad__GJ436_b' in glob_fit_dic['IntrProp']['mod_prop']) and (glob_fit_dic['IntrProp']['mod_prop']['inclin_rad__GJ436_b']['vary']):
            glob_fit_dic['IntrProp']['prior_func']={'sini_pr':{'val':0.99843,'sig':0.00003,'pl':'GJ436_b'}}

        
      
 
    #Walkers exclusion  
    glob_fit_dic['IntrProp']['exclu_walk']=True     & False       

    #Automatic exclusion of outlying chains
    glob_fit_dic['IntrProp']['exclu_walk_autom']=None  #  5.

    #Derived errors
    # glob_fit_dic['IntrProp']['HDI']='1s'   #None   #'3s'   
#        fit_dic['HDI_nbins']={'veq':310,'vsini':310,'inclin_rad__'+pl_loc:300,'lambda_rad__'+pl_loc:230,'lambda_deg':230,'ip_deg':250,'b':220}   #fit WASP121b SB        
#        fit_dic['HDI_nbins']={'veq':180,'inclin_rad__'+pl_loc:200,'lambda_rad__'+pl_loc:200,'lambda_deg':200,'ip_deg':200,'b':200,'cos_istar':50,'alpha_rot':300,'istar':50,'psi':100}   #fit WASP121b DR  low istar      
#        fit_dic['HDI_nbins']={'veq':100,'inclin_rad__'+pl_loc:100,'lambda_rad__'+pl_loc:200,'lambda_deg':200,'ip_deg':100,'b':100,'cos_istar':35,'alpha_rot':150,'istar':50,'psi':100}   #fit WASP121b DR  high istar
#        fit_dic['HDI_nbins']={'veq':50}   #pour l'intervalle a 3s
    # glob_fit_dic['IntrProp']['HDI_nbins']= {}   

    
    
    #Derived lower/upper limits
    # glob_fit_dic['IntrProp']['conf_limits']={'ecc_pl1':{'bound':0.,'type':'upper','level':['1s','3s']}}  


    #MCMC chains
    glob_fit_dic['IntrProp']['save_MCMC_chains']='png'   #png      
    
    #MCMC corner plot
    glob_fit_dic['IntrProp']['corner_options']={    
#            'bins_1D_par':[50,50,50,50],       #vsini, ip, lambda, b
#            'bins_2D_par':[30,30,30,30], 
#            'range_par':[(0.,320.),(88.,90.),(86.,91.),(0.,0.13)], 
##            'plot_HDI':True,             

#            'bins_1D_par':[50,50,50,45,50,50,50],       #veq, ip, alpha, istar, lambda, psi, b 
##            'range_par':[(40.,300.),(87.8,89.2),(-0.6,0.5),(1.,14.),(84.8,89.),(86.8,89.1),(0.055,0.145)],     #low istar
##            'range_par':[(40.,300.),(87.8,89.2),(-0.6,0.5),(166.,179.),(84.8,89.),(90.3,92.),(0.055,0.145)],      #high istar
#            'plot_HDI':True,  

#            'bins_1D_par':[50,50,50,45,50,50],       #veq, ip, alpha, istar, lambda, psi    FIG PAPER
##            'range_par':[(40.,300.),(87.8,89.2),(-0.6,0.5),(1.,14.),(84.8,89.),(86.8,89.1)],     #low istar
#            'range_par':[(40.,300.),(87.8,89.2),(-0.6,0.5),(166.,179.),(84.8,89.),(90.3,92.)],      #high istar
#            'plot_HDI':True,
#            'plot1s_1D':False,
                
                
#            'major_int':[0.2,50.],
#            'minor_int':[0.1,10.],
            'color_levels':['deepskyblue','lime'],
#            'smooth2D':[0.05,5.] 
#            'plot1s_1D':False
        }

    #Chi2 values
    plot_dic['chi2_fit_IntrProp']=''   #pdf      
  
    
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
##################################################################################################       
#%%% Module: fitting joined residual profiles    
##################################################################################################     

##### RECODER

    # '''
    # Fitting joined residual stellar profiles from combined (unbinned) instruments and visits 
    #     - structure is similar to the joined intrinsic profiles fit
    #     - fits are performed on all in-transit and out-transit exposures
    #     - allows including spot properties (latitude, size, Tcenter, flux level) in the fitted properties
        
    # '''  
    # 

    #Activating 
    gen_dic['fit_ResProf'] = True  &  False

    # #Indexes of exposures to be fitted, in each visit
    # #    - define instruments and visits to be fitted (they will not be fitted if not used as keys, or if set to []), set their value to 'all' for all in-transit exposures to be fitted
    # if gen_dic['star_name'] == 'V1298tau' :        
    #     glob_fit_dic['ResProf']['idx_in_fit']={'HARPN':{'mock_vis': range(40,50)}} 
    

    # #Model line
    # #    - specific to the instrument
    # if gen_dic['star_name'] == 'V1298tau' :
    #     glob_fit_dic['ResProf']['func_prof_name']={'HARPN':'gauss'}


    # #Fit coordinate for the line properties : mu or r_proj
    # glob_fit_dic['ResProf']['dim_fit']='mu'


    # #Define fitted range
    # glob_fit_dic['ResProf']['fit_range'] = {}
    # if gen_dic['star_name']=='V1298tau':
    #     fit_range = [[-30,30]] 
    #     glob_fit_dic['ResProf']['fit_range']['HARPN']={'mock_vis':fit_range} 


    # #Fit mode
    # #    - 'chi2', 'mcmc', or ''
    # glob_fit_dic['ResProf']['fit_mod']='mcmc' 




    # #Fit line property as absolute ('abs') or modulated ('modul') polynomial        
    # glob_fit_dic['ResProf']['pol_mode']='abs'  
        
    # #Model properties
    # #    - we define here coefficients specific to the line model:
    # # + polynomial coefficients to describe the line contrast and FWHM (km/s) variations with a chosen dimension
    # # + core/lobe relations for double-gaussian models
    # #    - properties names must be defined as 'prop__ISinst_VSvis'  
    # # + 'inst' is the name of the instrument, which should be set to '_' for the property to be common to all instruments and their visits
    # # + 'vis' is the name of the visit, which should be set to '_' for the property to be common to all visits of this instrument
    # # + the properties set here define the lines before instrumental convolution, which is then applied automatically for each instrument 
    # glob_fit_dic['ResProf']['mod_prop']={}
    
    
    

                                      
    
    # if gen_dic['star_name']=='V1298tau' :
    #     bjd_obs = 2458877.6306
    #     bjd_sp1 = 2458877.6306 - 12/24
    #     bjd_sp2 = 2458877.6306 +  5/24


    #     glob_fit_dic['ResProf']['mod_prop']={'veq':{'vary':True,'guess':23.,'bd':[23,24],'physical':True},
    #                                 'lambda_rad__plV1298tau_b':{'vary':True,'guess':0,'bd':[-np.pi , np.pi], 'physical':True},           
    #                                 'ctrst_ord0__ISHARPN_VSmock_vis':{'vary':True ,'guess':0.7,'bd':[0,1]},   
    #                                 'FWHM_ord0__ISHARPN_VSmock_vis': {'vary':True ,'guess':4, 'bd':[0.,10.]}, 
                
    #                                 # # Pour le spot 'spot1' : 
    #                                 # 'lat__ISHARPN_VSmock_vis_SPspot1'     : {'vary':True ,'guess':30,       'bd' : [-80,80]},
    #                                 # 'Tcenter__ISHARPN_VSmock_vis_SPspot1' : {'vary':True ,'guess':bjd_sp1, 'bd' : [bjd_sp1-1/3, bjd_sp1+1/3]},
    #                                 # 'ang__ISHARPN_VSmock_vis_SPspot1'     : {'vary':True ,'guess':20,      'bd' : [10,30]},
    #                                 # 'flux__ISHARPN_VSmock_vis_SPspot1'    : {'vary':True ,'guess':0.4,     'bd' : [0,1]},
    #                                 # 
    #                                 # # Pour le spot 'spot2' : 
    #                                 # 'lat__ISHARPN_VSmock_vis_SPspot2'     : {'vary':True ,'guess':40,       'bd' : [-80,80]},
    #                                 # 'Tcenter__ISHARPN_VSmock_vis_SPspot2' : {'vary':True ,'guess':bjd_sp2, 'bd' : [bjd_sp2-1/3, bjd_sp2+1/3]},
    #                                 # 'ang__ISHARPN_VSmock_vis_SPspot2'     : {'vary':True ,'guess':25,      'bd' : [10,30]},
    #                                 # 'flux__ISHARPN_VSmock_vis_SPspot2'    : {'vary':True ,'guess':0.4,     'bd' : [0,1]},
    #                                    }
                                       
                

    # #------------------------------------------
    # #MCMC options
    # #------------------------------------------
    
    # #Run mcmc or retrieve results
    # glob_fit_dic['ResProf']['mcmc_run_mode']='use'   # reuse




    # #Walkers settings 
    # glob_fit_dic['ResProf']['mcmc_set']={}
    
    # if gen_dic['star_name'] == 'V1298tau':
    #     glob_fit_dic['ResProf']['mcmc_set']={'nwalkers':20,'nsteps':1,'nburn':0}          


        
    # #Priors on variable model parameters
    # #    - see gen_dic['fit_DI'] for details
    # glob_fit_dic['ResProf']['priors']={}
    # if gen_dic['star_name'] == 'V1298tau' :
    #     bjd_obs = 2458877.6306
    #     bjd_sp1 = 2458877.6306 - 12/24
    #     bjd_sp2 = 2458877.6306 +  5/24

    #     glob_fit_dic['ResProf']['priors'].update({
    #         'veq':{'mod':'uf','low':23.,'high':24.},    
    #         'lambda_rad__plV1298tau_b':{'mod':'uf','low':-np.pi,'high':np.pi},       
    #         'FWHM_ord0__ISHARPN_VSmock_vis':{'mod':'uf','low':0.,'high':10.},
    #         'ctrst_ord0__ISHARPN_VSmock_vis':{'mod':'uf','low':0.,'high':1.},
                
    #         # # Pour le spot 'spot1' : 
    #         # 'lat__ISHARPN_VSmock_vis_SPspot1'     : {'mod':'uf' , 'low':-80,              'high':80},
    #         # 'Tcenter__ISHARPN_VSmock_vis_SPspot1' : {'mod':'uf' , 'low':bjd_sp1-1/3,      'high':bjd_sp1+1/3},
    #         # 'ang__ISHARPN_VSmock_vis_SPspot1'     : {'mod':'uf' , 'low':10,               'high':30},
    #         # 'flux__ISHARPN_VSmock_vis_SPspot1'    : {'mod':'uf' , 'low':0,                'high':1},
    #         # 
    #         # # Pour le spot 'spot2' : 
    #         # 'lat__ISHARPN_VSmock_vis_SPspot2'     : {'mod':'uf' , 'low':-80,           'high':80},
    #         # 'Tcenter__ISHARPN_VSmock_vis_SPspot2' : {'mod':'uf' , 'low': bjd_sp2-1/3,  'high': bjd_sp2 + 1/3},
    #         # 'ang__ISHARPN_VSmock_vis_SPspot2'     : {'mod':'uf' , 'low':10,            'high':30},
    #         # 'flux__ISHARPN_VSmock_vis_SPspot2'    : {'mod':'uf' , 'low':0,             'high':1},
            
                                    
    #                                 })








    
    # #Use complex prior function
    # glob_fit_dic['ResProf']['prior_func']={}        

    # #Excluding manually some of the walkers
    # #    - define conditions within routine
    # glob_fit_dic['ResProf']['exclu_walk']=True   & False       

    # #Automatic exclusion of outlying chains
    # #    - set to None, or exclusion threshold
    # glob_fit_dic['ResProf']['exclu_walk_autom']=None  #  5.

    # #Define lower/upper limits to be calculated
    # # glob_fit_dic['ResProf']['conf_limits']={'ecc_pl1':{'bound':0.,'type':'upper','level':['1s','3s']}} 

    # #Plot chains for MCMC parameters
    # glob_fit_dic['ResProf']['save_MCMC_chains']='png'   #png  

    # #Choose list of modifications to be made to the final chains 
    # #    - each field calls a specific function (see routine for more details)
    # glob_fit_dic['ResProf']['modif_list'] = ['vsini','lambda_deg']

    # #Define HDI interval to be calculated 
    # glob_fit_dic['ResProf']['HDI']='1s'   #None   #'3s'   



    
    # #Options for corner plot
    # #    - see function for options
    # glob_fit_dic['ResProf']['corner_options']={
    #     'plot_HDI':True , 
    #     'plot1s_1D':False,
    #     'color_levels':['deepskyblue','lime'],
    #     'fontsize':8,
    #     }    


    # #Print on-screen fit information
    # glob_fit_dic['ResProf']['verbose']=True  &  False


    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
##################################################################################################       
#%%% Module: fitting joined intrinsic profiles  
# - fitting joined intrinsic stellar profiles from combined (unbinned) instruments and visits 
# - use 'idx_in_fit' to choose which visits to fit (can be a single one)
# - the contrast and FWHM of the intrinsic stellar lines (before instrumental convolution) are fitted as polynomials of the chosen coordinate
#   surface RVs are fitted using the reloaded RM model  
#   beware that the disk-integrated and intrinsic stellar profile have the same continuum, but it is not necessarily unity as set in the analytical and theoretical models, whose continuum must thus let free to vary 
# - several options need to be controlled from within the function
# - use plot_dic['prop_Intr']='' to plot the properties of the derived profiles
#   use plot_dic['CCF_Intrbin']='' to plot the derived profiles
#   use gen_dic['loc_data_corr'] to visualize the derived profiles
# - to derive the stellar inclination from Rstar and Peq, use them as model parameters alongside cosistar, instead of veq  
#   set priors on Rstar and Peq from the literature and a uniform prior on cosistar (=isotropic distribution), or more complex priors if relevant
#   then istar can be directly derived from cosistar in post-processing (alongside veq and vsini), and will have been constrained by the independent priors on Peq, Rstar, and the data through the corresponding vsini
##################################################################################################     
        
#%%%% Activating 
gen_dic['fit_IntrProf'] = False    

    
#%%%% Multi-threading
glob_fit_dic['IntrProf']['nthreads'] = 6


#%%%% Fitted data

#%%%%% Exposures to be fitted
#    - indexes are relative to in-transit tables
#    - define instruments and visits to be fitted (they will not be fitted if not used as keys, or if set to []), set their value to 'all' for all in-transit exposures to be fitted
#    - add '_bin' at the end of a visit name for its binned exposures to be fitted instead of the original ones (must have been calculated with the binning module)
#      all other mentions of the visit (eg in parameter names) can still refer to the original visit name
glob_fit_dic['IntrProf']['idx_in_fit']={}


#%%%%% Trimming
glob_fit_dic['IntrProf']['trim_range'] = {}


#%%%%% Order to be fitted
glob_fit_dic['IntrProf']['order']={}  


#%%%%% Continuum range
glob_fit_dic['IntrProf']['cont_range'] = {}

                  
#%%%%% Spectral range(s) to be fitted
glob_fit_dic['IntrProf']['fit_range'] = {}


#%%%% Line profile model         
    
#%%%%% Transition wavelength
glob_fit_dic['IntrProf']['line_trans']=None        


#%%%%% Model type
glob_fit_dic['IntrProf']['mode'] = 'ana' 

 
#%%%%% Analytical profile
#    - default: 'gauss' 
glob_fit_dic['IntrProf']['func_prof_name'] = {}

    
#Analytical profile coordinate
#    - fit coordinate for the line properties of analytical profiles
#    - see possibilities in gen_dic['fit_IntrProp']
glob_fit_dic['IntrProf']['dim_fit']='mu'


#%%%%% Analytical profile variation
#    - fit line property as absolute ('abs') or modulated ('modul') polynomial        
glob_fit_dic['IntrProf']['pol_mode']='abs'  


#%%%%% Fixed/variable properties
#    - structure is the same as glob_fit_dic['IntrProp']['mod_prop']
#    - intrinsic properties define the lines before instrumental convolution, which can then be applied specifically to each instrument  
glob_fit_dic['IntrProf']['mod_prop']={}

   
#%%%%% PC noise model
#    - indicate for each visit:
# + the path to the PC matrix, already reduced to the PC requested to correct the visit in the PCA module
#   beware that the correction will be applied only over the range of definition of the PC set in the PCA
#   beware that one PC adds the number of fitted intrinsic profiles to the free parameters of the joint fit
# + whether to account or not (idx_out = []) for the PCA fit in the calculation of the fit merit values, using all out exposures (idx_out = 'all') or a selection
# + set noPC = True to account for the chi2 of the null hypothesis (no noise) on the out-of-transit data, without including PC to the RMR fit
glob_fit_dic['IntrProf']['PC_model']={}  

                    
#%%%% Fit settings 
    
#%%%%% Fitting mode
#    - 'chi2', 'mcmc', or ''
glob_fit_dic['IntrProf']['fit_mod']='' 


#%%%%% Printing fits results
glob_fit_dic['IntrProf']['verbose']= False


#%%%%% Priors on variable properties
#    - see gen_dic['fit_DI'] for details
glob_fit_dic['IntrProf']['priors']={}


#%%%%% Derived properties
#    - each field calls a specific function (see routine for more details)
glob_fit_dic['IntrProf']['modif_list'] = []


#%%%%% MCMC settings

#%%%%%% Calculating/retrieving
glob_fit_dic['IntrProf']['mcmc_run_mode']='use'


#%%%%%% Runs to re-use
#    - list of mcmc runs to reuse
#    - if 'reuse' is requested, leave empty to automatically retrieve the mcmc run available in the default directory
#  or set the list of mcmc runs to retrieve (they must have been run with the same settings, but the burnin can be specified for each run)
glob_fit_dic['IntrProf']['mcmc_reuse']={} 


#%%%%%% Runs to re-start
#    - indicate path to a 'raw_chains' file
#      the mcmc will restart the same walkers from their last step, and run from the number of steps indicated in 'mcmc_set'
glob_fit_dic['IntrProf']['mcmc_reboot']=''


#%%%%%% Walkers
glob_fit_dic['IntrProf']['mcmc_set']={}


#%%%%%% Complex priors
glob_fit_dic['IntrProf']['prior_func']={}     

    
#%%%%%% Walkers exclusion  
#    - define conditions within routine
glob_fit_dic['IntrProf']['exclu_walk']=False       


#%%%%%% Automatic exclusion of outlying chains
#    - set to None, or exclusion threshold
glob_fit_dic['IntrProf']['exclu_walk_autom']=None  


#%%%%%% Derived errors
#    - 'quant' or 'HDI'
glob_fit_dic['IntrProf']['out_err_mode']='HDI'
glob_fit_dic['IntrProf']['HDI']='1s'


#%%%%%% Derived lower/upper limits
glob_fit_dic['IntrProf']['conf_limits']={}


#%%%% Plot settings

#%%%%% MCMC chains
glob_fit_dic['IntrProf']['save_MCMC_chains']=''  


#%%%%% MCMC corner plot
#    - see function for options
glob_fit_dic['IntrProf']['corner_options']={}



if __name__ == '__main__':


    #Activating 
    gen_dic['fit_IntrProf'] = True #  &  False


    #Exposures to be fitted
    if gen_dic['transit_pl']=='Corot7b':glob_fit_dic['IntrProf']['idx_in_fit']={'ESPRESSO':{'2019-02-20':range(0,10)}}
    elif gen_dic['transit_pl']=='GJ9827d':
        glob_fit_dic['IntrProf']['idx_in_fit']={'ESPRESSO':{'2019-08-25':range(1,9)}}
        # glob_fit_dic['IntrProf']['idx_in_fit']={'HARPS':{'2018-08-18':range(1,8),'2018-09-18':range(1,6)}}
    elif gen_dic['transit_pl']=='GJ9827b':
        glob_fit_dic['IntrProf']['idx_in_fit']={'HARPS':{'2018-08-04':range(1,9), '2018-08-15':range(1,13), '2018-09-18':range(1,7), '2018-09-19':range(1,7)}}
    if gen_dic['star_name']=='HD3167':  
        if list(gen_dic['transit_pl'].keys())==['HD3167_c']:         
            glob_fit_dic['IntrProf']['idx_in_fit']={
                #'HARPN':{'2016-10-01':range(1,19)}}   #christiansen
                #'HARPN':{'2016-10-01':range(19)}}   #christiansen + Guilluy
                'HARPN':{'2016-10-01':range(17)}   #analyse finale
                }        

        elif list(gen_dic['transit_pl'].keys())==['HD3167_b']:   
            glob_fit_dic['IntrProf']['idx_in_fit']={
                #'ESPRESSO':{'2019-10-09':range(1,16)}}   #christiansen, christiansen + CHEOPS
                #'ESPRESSO':{'2019-10-09':range(1,15)}}   #test post-tr fit profils locaux idx_glob 25 a 38        
                #'ESPRESSO':{'2019-10-09':range(15)}}   #gandolfi
                'ESPRESSO':{'2019-10-09':range(1,16)},   #analyse finale               
                }            
        else:
            glob_fit_dic['IntrProf']['idx_in_fit']={
                #'ESPRESSO':{'2019-10-09':range(1,16)}}   #christiansen, christiansen + CHEOPS
                #'ESPRESSO':{'2019-10-09':range(1,15)}}   #test post-tr fit profils locaux idx_glob 25 a 38        
                #'ESPRESSO':{'2019-10-09':range(15)}}   #gandolfi
                'ESPRESSO':{'2019-10-09':range(1,16)},   #analyse finale
                
                #'HARPN':{'2016-10-01':range(1,19)}}   #christiansen
                #'HARPN':{'2016-10-01':range(19)}}   #christiansen + Guilluy
                'HARPN':{'2016-10-01':range(17)}   #analyse finale
                }

    if gen_dic['star_name']=='TOI858':          
        glob_fit_dic['IntrProf']['idx_in_fit']={'CORALIE':{'20191205':range(7),'20210118':range(1,6)}  }     
        # glob_fit_dic['IntrProf']['idx_in_fit']={'CORALIE':{'20191205':range(7),'20210118':[]}  }     
        # glob_fit_dic['IntrProf']['idx_in_fit']={'CORALIE':{'20191205':[],'20210118':range(1,6)}  } 

    if gen_dic['star_name']=='GJ436':          
        glob_fit_dic['IntrProf']['idx_in_fit']={
            # 'ESPRESSO':{'20190228':range(1,9),'20190429':range(1,9)},
            'HARPS':{'20070509':range(3,9)},
            'HARPN':{'20160318':range(1,8),'20160411':range(1,8)}}   
        
    if gen_dic['star_name']=='HIP41378':          
        glob_fit_dic['IntrProf']['idx_in_fit']={'HARPN':{'20191218':range(18)}  }  
    elif gen_dic['star_name'] == 'V1298tau' :        
        glob_fit_dic['IntrProf']['idx_in_fit']={'HARPN':{'mock_vis':range(1,19)}} 
    elif gen_dic['star_name']=='55Cnc':          
        glob_fit_dic['IntrProf']['idx_in_fit']={
            'ESPRESSO':{
                '20200205':range(1,32),
                '20210121':range(1,32),
                '20210124':range(1,32)
            },
            'HARPS':{
                '20120127':range(0,24),
                '20120213':range(0,28),
                '20120227':range(1,25),
                '20120315':range(1,25)
            },  
            'HARPN':{
                # '20131114':range(0,20),
                # '20131128':range(0,18),
                '20140101':range(0,13),
                # '20140126':range(0,12),
                # '20140226':range(0,13),
                # '20140329':range(1,14),
            }, 
            'EXPRES':{
                '20220131':range(15),
                '20220406':range(9)
                }
                         
            }


    #RM survey
    elif gen_dic['star_name']=='HAT_P3':glob_fit_dic['IntrProf']['idx_in_fit']={'HARPN':{'20200130':range(1,8)}} 
    elif gen_dic['star_name']=='Kepler25':glob_fit_dic['IntrProf']['idx_in_fit']={'HARPN':{'20190614':range(1,19)}} 
    elif gen_dic['star_name']=='Kepler68':glob_fit_dic['IntrProf']['idx_in_fit']={'HARPN':{'20190803':[]}}     
    elif gen_dic['star_name']=='HAT_P33':glob_fit_dic['IntrProf']['idx_in_fit']={'HARPN':{'20191204':range(1,33)}}         
    elif gen_dic['star_name']=='K2_105':glob_fit_dic['IntrProf']['idx_in_fit']={'HARPN':{'20200118':[]}}  
    elif gen_dic['star_name']=='HD89345':
        glob_fit_dic['IntrProf']['idx_in_fit']={'HARPN':{'20200202':range(2,93)}}  
        glob_fit_dic['IntrProf']['idx_in_fit']={'HARPN':{'20200202':range(3,92)}}    
    elif gen_dic['star_name']=='Kepler63':glob_fit_dic['IntrProf']['idx_in_fit']={'HARPN':{'20200513':range(9)}}          
    elif gen_dic['star_name']=='HAT_P49':glob_fit_dic['IntrProf']['idx_in_fit']={'HARPN':{'20200730':range(3,71)}}         
    elif gen_dic['star_name']=='WASP107':
        # glob_fit_dic['IntrProf']['idx_in_fit']={'CARMENES_VIS':{'20180224':range(1,9)}} 
        glob_fit_dic['IntrProf']['idx_in_fit']={'CARMENES_VIS':{'20180224':range(1,9)},'HARPS':{'20140406':range(1,11),'20180201':range(1,12),'20180313':range(1,12)}}        
    elif gen_dic['star_name']=='HAT_P11':
        glob_fit_dic['IntrProf']['idx_in_fit']={'HARPN':{'20150913':range(2,26),'20151101':range(2,25)},'CARMENES_VIS':{'20170807':range(1,17),'20170812':range(2,18)}}                 
        glob_fit_dic['IntrProf']['idx_in_fit']={'HARPN':{'20150913':range(2,26),'20151101':range(2,25)}}  
    elif gen_dic['star_name']=='WASP156':
        glob_fit_dic['IntrProf']['idx_in_fit']={'CARMENES_VIS':{'20190928':range(1,7),'20191025':range(1,6)}}                 
        glob_fit_dic['IntrProf']['idx_in_fit']={'CARMENES_VIS':{'20190928':range(1,7)}}  
        glob_fit_dic['IntrProf']['idx_in_fit']={'CARMENES_VIS':{'20191025':range(3,6)}}  
    elif gen_dic['star_name']=='HD106315':glob_fit_dic['IntrProf']['idx_in_fit']={'HARPS':{'20170309':range(3,44),'20170330':range(1,26),'20180323':range(1,27)}} 
    elif gen_dic['star_name']=='WASP166':
        glob_fit_dic['IntrProf']['idx_in_fit']={'HARPS':{'20170114':range(1,39),'20170304':range(1,37),'20170315':range(1,33)}} 
        # glob_fit_dic['IntrProf']['idx_in_fit']={'HARPS':{'20170114_bin':range(1,19),'20170304_bin':range(1,19),'20170315_bin':range(1,19)}} 
    elif gen_dic['star_name']=='WASP47':glob_fit_dic['IntrProf']['idx_in_fit']={'HARPN':{'20210730':[]}}         

    elif gen_dic['star_name'] in ['HD209458','WASP76']:
        glob_fit_dic['IntrProf']['idx_in_fit'] = deepcopy(glob_fit_dic['IntrProp']['idx_in_fit'])

        # print('TEST')
        # glob_fit_dic['IntrProf']['idx_in_fit']={'ESPRESSO':{'20190720':[15,30],'20190911':[15,30]}}   #Na CCF   

    #Trimming
    glob_fit_dic['IntrProf']['trim_range'] = deepcopy(data_dic['Intr']['fit_prof']['trim_range'])
    if gen_dic['star_name']=='HD209458': 
        if gen_dic['trim_spec']:glob_fit_dic['IntrProf']['trim_range'] = {'ESPRESSO':[5882.9,5903.1]}


    #Continuum range
    glob_fit_dic['IntrProf']['cont_range'] = deepcopy(data_dic['Intr']['cont_range'])
    if gen_dic['star_name']=='HAT_P33':glob_fit_dic['IntrProf']['cont_range']={'HARPN':[[-100.0, -40.0], [40.0, 100.0]]}
    elif gen_dic['star_name']=='Kepler68':glob_fit_dic['IntrProf']['cont_range']={'HARPN':[[-50.0, -20.0], [20.0, 50.0]]}
    elif gen_dic['star_name']=='HD89345':glob_fit_dic['IntrProf']['cont_range']={'HARPN':[[-60.0, -25.0], [25.0, 60.0]]}
    elif gen_dic['star_name']=='HAT_P49':glob_fit_dic['IntrProf']['cont_range']={'HARPN':[[-100.,-30.],[30.,100.]]}     
    elif gen_dic['star_name']=='HD106315':glob_fit_dic['IntrProf']['cont_range']={'HARPS':[[-110.,-50.],[50.,110.]]}   
    elif gen_dic['star_name']=='55Cnc':
        glob_fit_dic['IntrProf']['cont_range']={
            'ESPRESSO':[[-100.,-20.],[20.,100.]],
            'HARPS':[[-100.,-20.],[20.,100.]],
            'HARPN':[[-100.,-20.],[20.,100.]],
            # 'EXPRES':[[-100.,-20.],[20.,100.]],
            'EXPRES':[[-100.,-65.],[-42.,-37.],[20.,100.]]
            } 
    # elif gen_dic['star_name']=='HD209458': 
    #     if gen_dic['trim_spec']:glob_fit_dic['IntrProf']['cont_range']['ESPRESSO'] = np.array([[ 5883. , 5885.],[5901., 5903. ]])    #ANTARESS fit sodium doublet  


            
                  
    #Spectral range(s) to be fitted
    glob_fit_dic['IntrProf']['fit_range'] = deepcopy(data_dic['Intr']['fit_range'])    
    if gen_dic['star_name']=='HAT_P33':glob_fit_dic['IntrProf']['fit_range']={'HARPN':{'20191204':[[-100.0, 100.0]]}}
    elif gen_dic['star_name']=='Kepler68':glob_fit_dic['IntrProf']['fit_range']={'HARPN':{'20190803':[[-50.0,50.0]]}}
    elif gen_dic['star_name']=='HD89345':glob_fit_dic['IntrProf']['fit_range']={'HARPN':{'20200202':[[-60.0,60.0]]}}  
    elif gen_dic['star_name']=='HAT_P49':glob_fit_dic['IntrProf']['fit_range']={'HARPN':{'20200730':[[-100.0,100.0]]}}  
    elif gen_dic['star_name']=='HD106315':glob_fit_dic['IntrProf']['fit_range']={'HARPS':{'20170309':[[-110.0,110.0]],'20170330':[[-110.0,110.0]],'20180323':[[-110.0,110.0]]}}  
    elif gen_dic['star_name']=='55Cnc':
        glob_fit_dic['IntrProf']['fit_range'] = {inst:{vis:[[-100.,100.]] for vis in glob_fit_dic['IntrProf']['idx_in_fit'][inst]} for inst in ['ESPRESSO','HARPS','HARPN']}
        glob_fit_dic['IntrProf']['fit_range']['EXPRES'] = {'20220131':[[-100.,-37.],[-22.,100.]],'20220406':[[-100.,-65.],[-42.,100.]]}
    elif gen_dic['star_name']=='HD209458':             
        if gen_dic['trim_spec']:
            fit_range = np.array([[ 5885.6 ,5890.95 ],[5891.45,5892.55],[5893.1 ,5897. ],[5897.45, 5899.2 ],[5900.,5901.]])
            glob_fit_dic['IntrProf']['fit_range'] = {'ESPRESSO':{'20190720':fit_range,'20190911':fit_range}}  #Na doublet + continuum
        
        
    #Model type
    if gen_dic['star_name']=='HD209458': 
        if gen_dic['trim_spec']:glob_fit_dic['IntrProf']['mode'] = 'theo'         
        
 
    #Analytical profile
    if gen_dic['star_name']=='GJ436':glob_fit_dic['IntrProf']['func_prof_name']={'ESPRESSO':'dgauss'}    
    elif gen_dic['star_name'] in ['HAT_P3','Kepler25','Kepler68','HAT_P33','K2_105','HD89345','Kepler63','HAT_P49','HIP41378','WASP47']:glob_fit_dic['IntrProf']['func_prof_name']={'HARPN':'gauss'}        
    elif gen_dic['star_name']=='WASP107':glob_fit_dic['IntrProf']['func_prof_name']={'HARPS':'gauss','CARMENES_VIS':'voigt'}
    elif gen_dic['star_name'] in ['HD106315','WASP166']:glob_fit_dic['IntrProf']['func_prof_name']={'HARPS':'gauss'}
    elif gen_dic['star_name']=='WASP156':
        glob_fit_dic['IntrProf']['func_prof_name']={'CARMENES_VIS':'voigt'}
        glob_fit_dic['IntrProf']['func_prof_name']={'CARMENES_VIS':'gauss'}  
    elif gen_dic['star_name']=='HAT_P11':
        glob_fit_dic['IntrProf']['func_prof_name']={'HARPN':'gauss','CARMENES_VIS':'gauss'}
        glob_fit_dic['IntrProf']['func_prof_name']={'HARPN':'gauss','CARMENES_VIS':'voigt'}
    elif gen_dic['star_name'] == 'V1298tau' :
        glob_fit_dic['IntrProf']['func_prof_name']={'HARPN':'gauss'}
    elif gen_dic['star_name'] == '55Cnc' :
        glob_fit_dic['IntrProf']['func_prof_name']={'ESPRESSO':'gauss','HARPS':'gauss','HARPN':'gauss','EXPRES':'gauss'}

        
    #Analytical profile coordinate
    if gen_dic['star_name'] in ['HD209458','WASP76']:glob_fit_dic['IntrProf']['dim_fit']='r_proj'


    #Analytical profile variation
    if gen_dic['star_name'] in ['HD106315','WASP107','HAT_P11','HD209458','WASP76']:glob_fit_dic['IntrProf']['pol_mode']='modul'  


    
    #Fixed/variable properties
   #     #Parametres associes
    #     p_start.add_many(#Contrast polynomial, common to all fitted visits 
    #                     ( 'ctrst0__IS__VS_', 0.6,      True,None  ,  None ,  None),      
    #                     ( 'ctrst1__IS__VS_', 0.,       True,None ,  None,  None),                
    #                     ( 'ctrst2__IS__VS_', 0.,       True,None ,  None,  None),   
    #                     #FWHM polynomial, common to all fitted visits 
    #                     ( 'FWHM0__IS__VS_',8.,      True,None  ,  None ,  None),     
    #                     ( 'FWHM1__IS__VS_',0.,       True,None ,  None,  None),                   
    #                     ( 'FWHM2__IS__VS_',0.,       True,None ,  None,  None))         

    # elif gen_dic['main_pl']==['HD3167_b']: 
    #     fixed_args['func_prof_name'] = 'gauss'
        
    #     # p_start.add_many(( 'ctrst0__IS__VS_', 0.6,      True,None  ,  None ,  None),
    #     #                   ( 'FWHM0__IS__VS_',5.,      True,None  ,  None ,  None))     

    #     p_start.add_many(( 'ctrst0__IS__VS_', 0.6,      True,None  ,  None ,  None),     
    #                     ( 'ctrst1__IS__VS_', 0.,       True,None ,  None,  None),    
    #                     # ( 'ctrst2__IS__VS_', 0.,       True,None ,  None,  None), 
    #                       ( 'FWHM0__IS__VS_',5.,      True,None  ,  None ,  None))
    #                       # ( 'FWHM1__IS__VS_',0.,      True,None  ,  None ,  None))  

    # elif gen_dic['main_pl']==['HD3167_c']:    
    #     fixed_args['func_prof_name'] = 'gauss'
    #     p_start.add_many(( 'ctrst0__IS__VS_', 0.6,      True,None  ,  None ,  None),     
    #                     ( 'ctrst1__IS__VS_', 0.,       True,None ,  None,  None),      
    #                     # ( 'ctrst2__IS__VS_', 0.,       True,None ,  None,  None), 
    #                      ( 'FWHM0__IS__VS_',2.,      True,None  ,  None ,  None))    
    #                     # ( 'FWHM1__IS__VS_',0.,       True,None ,  None,  None),      
    #                     # ( 'FWHM2__IS__VS_',0.,       True,None ,  None,  None)) 

    # elif ('HD3167_b' in gen_dic['main_pl']) and ('HD3167_c' in gen_dic['main_pl']):  
    #     fixed_args['func_prof_name'] = 'gauss'
    #     p_start.add_many(( 'ctrst0__IS__VS_', 0.6,      True,None  ,  None ,  None),     
    #                     ( 'ctrst1__IS__VS_', 0.,       True,None ,  None,  None),    
    #                     ( 'ctrst2__IS__VS_', 0.,       True,None ,  None,  None), 
    #                      ( 'FWHM0__IS__VS_',0.,      True,None  ,  None ,  None))
    #                       # ( 'FWHM1__IS__VS_',1.,      True,None  ,  None ,  None))  

    # elif gen_dic['main_pl']=='Corot7b':  
    #     fixed_args['func_prof_name'] = 'gauss' 
    #     p_start.add_many(( 'ctrst0__IS__VS_', 0.6,      True,None  ,  None ,  None),
    #                      ( 'FWHM0__IS__VS_',2.,         True,None  ,  None ,  None))     

    # elif gen_dic['main_pl']=='GJ9827d':   
    #     fixed_args['func_prof_name'] = 'gauss'
    #     p_start.add_many(( 'ctrst0__IS__VS_', 0.6,      True,None  ,  None ,  None),
    #                      ( 'FWHM0__IS__VS_',3.,         True,None  ,  None ,  None))     

    # elif gen_dic['main_pl']=='GJ9827b':  
    #     fixed_args['func_prof_name'] = 'gauss'
    #     p_start.add_many(( 'ctrst0__IS__VS_', 0.6,      True,None  ,  None ,  None),
    #                      ( 'FWHM0__IS__VS_',3.,         True,None  ,  None ,  None))  

    # elif gen_dic['main_pl']==['TOI858b']:  
    #     fixed_args['func_prof_name'] = 'gauss'
    #     # p_start.add_many(( 'ctrst0__IS__VS_', 0.6,      True,None  ,  None ,  None), 
    #     #                   ( 'FWHM0__IS__VS_',2.,      True,None  ,  None ,  None))    

    #     p_start.add_many(( 'ctrst0__ISCORALIE_VS20191205', 0.6,      True,None  ,  None ,  None), 
    #                       ( 'FWHM0__ISCORALIE_VS20191205',2.,      True,None  ,  None ,  None),
    #                       ( 'ctrst0__ISCORALIE_VS20210118', 0.6,      True,None  ,  None ,  None), 
    #                       ( 'FWHM0__ISCORALIE_VS20210118',2.,      True,None  ,  None ,  None)) 


            # elif gen_dic['main_pl']==['HD3167_b']:
            #     # fit_dic['uf_bd'].update({
            #     # #     'ctrst0':[0.,1.],
            #     #     'ctrst0':[-2.,2.],
            #     #     'FWHM0':[0.,15.]})
                
            #     fit_dic['uf_bd'].update({
            #         'ctrst0':[-0.1,0.1],
            #         'ctrst1':[0.5,1.], 
            #         # 'ctrst2':[-0.7,-0.2],
            #         'FWHM0':[0.,1.]}) 
            #         # 'FWHM1':[5.,10.],
            #         # 'FWHM2':[-7.,-1.]}) 
                
            # elif gen_dic['main_pl']==['HD3167_c']:
            #     fit_dic['uf_bd'].update({
            #     #     'ctrst0':[0.,1.],
            #         'FWHM0':[0.,5.]
            #         })
            #     fit_dic['uf_bd'].update({
            #         'ctrst0':[-0.1,0.1],
            #         'ctrst1':[0.5,1.],
            #         # 'ctrst2':[-0.7,-0.2],
            #         # 'FWHM0':[0.,1.],
            #         # 'FWHM1':[5.,10.],
            #         # 'FWHM2':[-7.,-1.]
            #         })   
                
            # elif ('HD3167_b' in gen_dic['main_pl']) and ('HD3167_c' in gen_dic['main_pl']):
            #     fit_dic['uf_bd'].update({
            #         # 'ctrst0':[0.2,0.9],
            #         'ctrst0':[-0.1,0.1],
            #         'ctrst1':[-0.1,0.1],
            #         'ctrst2':[-0.1,0.1],
            #         'FWHM0':[2.,8.]})                
            #         # 'FWHM1':[0.,10.]})  

                  
            # elif gen_dic['main_pl']=='Corot7b':  
            #     fit_dic['uf_bd'].update({
            #         'ctrst0':[0.2,0.8],
            #         'FWHM0':[0.5,3.]})                
            # elif gen_dic['main_pl']=='GJ9827d':  
            #     fit_dic['uf_bd'].update({
            #         'ctrst0':[0.2,0.8],
            #         'FWHM0':[0.5,3.]})     
            # elif gen_dic['main_pl']=='GJ9827b':  
            #     fit_dic['uf_bd'].update({
            #         'ctrst0':[0.2,0.8],
            #         'FWHM0':[0.5,3.]})  
            # elif gen_dic['main_pl']==['TOI858b']:  
            #     # fit_dic['uf_bd'].update({
            #     #     'ctrst0':[0.2,0.8],
            #     #     'FWHM0':[2.,15.]})  

            #     fit_dic['uf_bd'].update({
            #         'ctrst0__ISCORALIE_VS20191205':[0.2,0.8],
            #         'FWHM0__ISCORALIE_VS20191205':[2.,15.],  
            #         'ctrst0__ISCORALIE_VS20210118':[0.2,0.8],
            #         'FWHM0__ISCORALIE_VS20210118':[2.,15.]})
    
    
    
    if gen_dic['star_name']=='GJ436':
        glob_fit_dic['IntrProf']['mod_prop']={'veq':{'vary':True,'guess':0.2,'bd':[0.1,0.4]},
                                      'lambda_rad__GJ436_b':{'vary':True,'guess':100.*np.pi/180.,'bd':[70.*np.pi/180.,140.*np.pi/180.]}}

        # #Ratio core/lobe variable entre visites
        # glob_fit_dic['IntrProf']['mod_prop'].update({'RV_l2c__ISESPRESSO_VS20190228':{'vary':True,'guess':0.2,'bd':[0.,0.5]},
        #                                      'amp_l2c__ISESPRESSO_VS20190228':{'vary':True,'guess':0.2,'bd':[0.1,1.]},
        #                                      'FWHM_l2c__ISESPRESSO_VS20190228':{'vary':True,'guess':0.2,'bd':[0.,3.]},
        #                                      'RV_l2c__ISESPRESSO_VS20190429':{'vary':True,'guess':0.2,'bd':[0.,0.5]},
        #                                      'amp_l2c__ISESPRESSO_VS20190429':{'vary':True,'guess':0.2,'bd':[0.1,1.]},
        #                                      'FWHM_l2c__ISESPRESSO_VS20190429':{'vary':True,'guess':0.2,'bd':[0.,3.]}})   


        # #BESTFIT: fixed to DI results directly aligned, Erreur propagee, chi2
        # #ESPRESSO skycorr
        # glob_fit_dic['IntrProf']['mod_prop'].update({ 'RV_l2c__ISESPRESSO_VS20190228':{'vary':False,'guess':-2.3686128554e-04},
        #                                       'amp_l2c__ISESPRESSO_VS20190228':{'vary':False,'guess':9.9477541529e-01},
        #                                       'FWHM_l2c__ISESPRESSO_VS20190228':{'vary':False,'guess':1.0035865365e+00},
        #                                       'RV_l2c__ISESPRESSO_VS20190429':{'vary':False,'guess':-1.8285586781e-04},
        #                                       'amp_l2c__ISESPRESSO_VS20190429':{'vary':False,'guess':9.9597117963e-01},
        #                                       'FWHM_l2c__ISESPRESSO_VS20190429':{'vary':False,'guess':1.0027560641e+00}})
        #HARPS/HARPS-N uncorrected for snr correlations        
        if not gen_dic['detrend_prof']:
            glob_fit_dic['IntrProf']['mod_prop'].update({ 'RV_l2c__ISHARPN_VS20160318':{'vary':False,'guess':-3.3510189653e-02},
                                                  'amp_l2c__ISHARPN_VS20160318':{'vary':False,'guess':7.3522272825e-01},
                                                  'FWHM_l2c__ISHARPN_VS20160318':{'vary':False,'guess':1.2254739251e+00},
                                                  'RV_l2c__ISHARPN_VS20160411':{'vary':False,'guess':-3.1743326463e-02},
                                                  'amp_l2c__ISHARPN_VS20160411':{'vary':False,'guess':7.4445209977e-01},
                                                  'FWHM_l2c__ISHARPN_VS20160411':{'vary':False,'guess':1.2148813565e+00},
                                                  'RV_l2c__ISHARPS_VS20070509':{'vary':False,'guess':-5.3893910472e-03},
                                                  'amp_l2c__ISHARPS_VS20070509':{'vary':False,'guess':9.4729356648e-01},
                                                  'FWHM_l2c__ISHARPS_VS20070509':{'vary':False,'guess':1.0338761874e+00}})            
            
        #HARPS/HARPS-N corrected for snr correlations
        if gen_dic['detrend_prof']:
            glob_fit_dic['IntrProf']['mod_prop'].update({ 'RV_l2c__ISHARPN_VS20160318':{'vary':False,'guess':-3.3511175497e-02},
                                                  'amp_l2c__ISHARPN_VS20160318':{'vary':False,'guess':7.3521731614e-01},
                                                  'FWHM_l2c__ISHARPN_VS20160318':{'vary':False,'guess':1.2254799712e+00},
                                                  'RV_l2c__ISHARPN_VS20160411':{'vary':False,'guess':-3.1744061734e-02},
                                                  'amp_l2c__ISHARPN_VS20160411':{'vary':False,'guess':7.4444730731e-01},
                                                  'FWHM_l2c__ISHARPN_VS20160411':{'vary':False,'guess':1.2148866936e+00},
                                                  'RV_l2c__ISHARPS_VS20070509':{'vary':False,'guess':-4.7232827857e-03},
                                                  'amp_l2c__ISHARPS_VS20070509':{'vary':False,'guess':9.5392977894e-01},
                                                  'FWHM_l2c__ISHARPS_VS20070509':{'vary':False,'guess':1.0294427603e+00}})               
            

        # #Ratio core/lobe common to visits
        # glob_fit_dic['IntrProf']['mod_prop'].update({'RV_l2c__IS__VS_':{'vary':True,'guess':0.2,'bd':[0.,0.5]},
        #                                      'amp_l2c__IS__VS_':{'vary':True,'guess':0.2,'bd':[0.1,1.]},
        #                                      'FWHM_l2c__IS__VS_':{'vary':True,'guess':0.2,'bd':[0.,3.]}})         



        #BESTFIT
        #Contrast and core FWHM variable between visits, constant with mu
        # glob_fit_dic['IntrProf']['mod_prop'].update({'ctrst_ord0__ISESPRESSO_VS20190228':{'vary':True,'guess':0.2,'bd':[0.1,0.6]},
        #                                       'FWHM_ord0__ISESPRESSO_VS20190228':{'vary':True,'guess':0.2,'bd':[1.,3.]},
        #                                       'ctrst_ord0__ISESPRESSO_VS20190429':{'vary':True,'guess':0.2,'bd':[0.1,0.6]},
        #                                       'FWHM_ord0__ISESPRESSO_VS20190429':{'vary':True,'guess':0.2,'bd':[1.,3.]}})
        glob_fit_dic['IntrProf']['mod_prop'].update({'ctrst_ord0__ISHARPN_VS20160318':{'vary':True,'guess':0.2,'bd':[0.1,0.6]},
                                              'FWHM_ord0__ISHARPN_VS20160318':{'vary':True,'guess':0.2,'bd':[1.,3.]},
                                              'ctrst_ord0__ISHARPN_VS20160411':{'vary':True,'guess':0.2,'bd':[0.1,0.6]},
                                              'FWHM_ord0__ISHARPN_VS20160411':{'vary':True,'guess':0.2,'bd':[1.,3.]}})
        glob_fit_dic['IntrProf']['mod_prop'].update({'ctrst_ord0__ISHARPS_VS20070509':{'vary':True,'guess':0.2,'bd':[0.1,0.6]},
                                              'FWHM_ord0__ISHARPS_VS20070509':{'vary':True,'guess':0.2,'bd':[1.,3.]}})                   


        # #Contrast and core FWHM common between ESPRESSO visits, constant with mu
        # glob_fit_dic['IntrProf']['mod_prop'].update({'ctrst_ord0__ISESPRESSO_VS_':{'vary':True,'guess':0.2,'bd':[0.1,0.6]},
        #                                       'FWHM_ord0__ISESPRESSO_VS_':{'vary':True,'guess':0.2,'bd':[1.,3.]}})
        # #Contrast and core FWHM common between HARPN visits, constant with mu
        # glob_fit_dic['IntrProf']['mod_prop'].update({'ctrst_ord0__ISHARPN_VS_':{'vary':True,'guess':0.2,'bd':[0.1,0.6]},
        #                                       'FWHM_ord0__ISHARPN_VS_':{'vary':True,'guess':0.2,'bd':[1.,3.]}})
        # #Contrast and core FWHM common between HARPN visits, constant with mu, set to ESPRESSO values
        # glob_fit_dic['IntrProf']['mod_prop'].update({'ctrst_ord0__ISHARPN_VS_':{'vary':True,'guess':0.2,'bd':[0.1,0.6],'expr':'ctrst_ord0__ISESPRESSO_VS_'},
        #                                       'FWHM_ord0__ISHARPN_VS_':{'vary':True,'guess':0.2,'bd':[1.,3.],'expr':'FWHM_ord0__ISESPRESSO_VS_'}})



        # #Contrast and core FWHM variable between visits, variable with mu
        # glob_fit_dic['IntrProf']['mod_prop'].update({'ctrst_ord0__ISESPRESSO_VS20190228':{'vary':True,'guess':0.2,'bd':[0.1,0.5]},
        #                                       'ctrst_ord1__ISESPRESSO_VS20190228':{'vary':True,'guess':0.2,'bd':[-0.2,0.4]},
        #                                       'FWHM_ord0__ISESPRESSO_VS20190228':{'vary':True,'guess':0.2,'bd':[4.,5.]},
        #                                       'ctrst_ord0__ISESPRESSO_VS20190429':{'vary':True,'guess':0.2,'bd':[0.1,0.5]},
        #                                       'ctrst_ord1__ISESPRESSO_VS20190429':{'vary':True,'guess':0.2,'bd':[-0.2,0.4]},
        #                                       'FWHM_ord0__ISESPRESSO_VS20190429':{'vary':True,'guess':0.2,'bd':[4.,5.]}})                   

        # #Contrast and core FWHM common between visits, variable with mu
        # glob_fit_dic['IntrProf']['mod_prop'].update({'ctrst_ord0__IS__VS_':{'vary':True,'guess':0.2,'bd':[0.1,0.3]},
        #                                      # 'ctrst_ord1__IS__VS_':{'vary':True,'guess':0.2,'bd':[0.1,0.3]},
        #                                       'FWHM_ord0__IS__VS_':{'vary':True,'guess':0.2,'bd':[2.,5.]},
        #                                       'FWHM_ord1__IS__VS_':{'vary':True,'guess':0.2,'bd':[-1.,1.]}})         
 
 
        # #Contrast and core FWHM common between all three instruments, constant with mu
        # glob_fit_dic['IntrProf']['mod_prop'].update({'ctrst_ord0__IS__VS_':{'vary':True,'guess':0.2,'bd':[0.1,0.3]},
        #                                      'FWHM_ord0__IS__VS_':{'vary':True,'guess':0.2,'bd':[2.,5.]}})
 

        # glob_fit_dic['IntrProf']['mod_prop'].update({'inclin_rad__GJ436_b':{'vary':True,'guess':0.2,'bd':[  np.arcsin( 0.99843-3.*0.00003)    ,np.arcsin( 0.99843+3.*0.00003) ]},
        #                                      'aRs__GJ436_b':{'vary':True,'guess':0.2,'bd':[14.46-0.08*3,14.46+0.08*3]}})         
        
        
    
    if gen_dic['star_name']=='HIP41378':
        glob_fit_dic['IntrProf']['mod_prop']={'veq':{'vary':True,'guess':5.,'bd':[0.,10.],'physical':True},
                                      'lambda_rad__plHIP41378d':{'vary':True,'guess':100.*np.pi/180.,'bd':[0.*np.pi/180.,180.*np.pi/180.],'physical':True},
                                      'ctrst_ord0__IS__VS_':{'vary':True,'guess':0.2,'bd':[0.1,0.6]},
                                      'FWHM_ord0__IS__VS_':{'vary':True,'guess':0.2,'bd':[1.,3.]}}

    #RM survey
    if gen_dic['star_name']=='HAT_P3':
        glob_fit_dic['IntrProf']['mod_prop']={
                                     #'veq':{'vary':True,'guess':0.8,'bd':[0.1,1.6],'physical':True},
                                      'lambda_rad__plHAT_P3b':{'vary':True,'guess':-30.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True},           
                                      'ctrst_ord0__ISHARPN_VS20200130':{'vary':True ,'guess':0.5,
                                                                          # 'bd':[0.,2.5]},
                                                                          'bd':[0.,1.]},
                                      # 'ctrst_ord1__ISHARPN_VS20200130':{'vary':True ,'guess':0.,'bd':[-6.,3.]},    
                                      # 'ctrst_ord2__ISHARPN_VS20200130':{'vary':True ,'guess':0.,'bd':[-2.,6.]},                        
                                      'FWHM_ord0__ISHARPN_VS20200130':{'vary':True ,'guess':7.,'bd':[4.,20.]},
                                      # 'FWHM_ord1__ISHARPN_VS20200130':{'vary':True ,'guess':0.,'bd':[-20.,5.]}, 
                                       }

        # glob_fit_dic['IntrProf']['mod_prop'].update({'inclin_rad__plHAT_P3b':{'vary':True,'guess':0.,'bd':[  (86.31 -3.*0.19 )*np.pi/180.    ,(86.31 +3.*0.19 )*np.pi/180. ],'physical':True},
        #                                       'aRs__plHAT_P3b':{'vary':True,'guess':0.2,'bd':[9.8105-0.2667*3,9.8105+0.2667*3],'physical':True}})         
        
        
        
        glob_fit_dic['IntrProf']['mod_prop'].update({
                        'cos_istar':{'vary':True,'guess':0.,'bd':[-1.,1.],'physical':True}, 
                        'Rstar':{'vary':True,'guess':0.,'bd':[0.8,0.9],'physical':True},        
                        'Peq':{'vary':True,'guess':0.,'bd':[17.,21.],'physical':True}})           


    elif gen_dic['star_name']=='Kepler25':
        glob_fit_dic['IntrProf']['mod_prop']={'veq':{'vary':True,'guess':8.,'bd':[6.,10.],'physical':True},
                                      'lambda_rad__plKepler25c':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True},           
                                      'ctrst_ord0__ISHARPN_VS20190614':{'vary':True ,'guess':0.5,'bd':[0.,1.]},                       
                                      'FWHM_ord0__ISHARPN_VS20190614':{'vary':True ,'guess':7.,'bd':[4.,20.]}, 
                                       }

    elif gen_dic['star_name']=='Kepler68':
        glob_fit_dic['IntrProf']['mod_prop']={'veq':{'vary':True,'guess':0.,'bd':[0.,2.],'physical':True},
                                      'lambda_rad__plKepler68b':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True},           
                                      'ctrst_ord0__ISHARPN_VS20190803':{'vary':True ,'guess':0.5,'bd':[0.,1.]},                       
                                      'FWHM_ord0__ISHARPN_VS20190803':{'vary':True ,'guess':5.,'bd':[1.,10.]}, 
                                       }

    elif gen_dic['star_name']=='HAT_P33':
        glob_fit_dic['IntrProf']['mod_prop']={'veq':{'vary':True,'guess':0.8,'bd':[14.,18.],'physical':True},
                                      'lambda_rad__plHAT_P33b':{'vary':True,'guess':0.*np.pi/180.,'bd':[-15.*np.pi/180.,10.*np.pi/180.],'physical':True},           
                                      'ctrst_ord0__ISHARPN_VS20191204':{'vary':True ,'guess':0.,'bd':[0.2,0.6]},
                                       # 'ctrst_ord1__ISHARPN_VS20191204':{'vary':True ,'guess':0.,'bd':[-0.1,0.1]},   
                                       # 'ctrst_ord2__ISHARPN_VS20191204':{'vary':True ,'guess':0.,'bd':[-0.1,0.1]},                    
                                      'FWHM_ord0__ISHARPN_VS20191204':{'vary':True ,'guess':0.,'bd':[15.,25.]},
                                       # 'FWHM_ord1__ISHARPN_VS20191204':{'vary':True ,'guess':0.,'bd':[-15.,-5.]}, 
                                       }
  
        # glob_fit_dic['IntrProf']['mod_prop'].update({'c1_CB':{'vary':True,'guess':0.1,'bd':[-1.,1.],'physical':True}})

        # glob_fit_dic['IntrProf']['mod_prop'].update({'alpha_rot':{'vary':True,'guess':0.,'bd':[0.,1.],'physical':True},
        #                                       'cos_istar':{'vary':True,'guess':0.,'bd':[0.,1.],'physical':True}, 
        #                                       })


        # glob_fit_dic['IntrProf']['mod_prop'].update({'inclin_rad__plHAT_P33b':{'vary':True,'guess':0.,'bd':[  (88.2 -3.*1.3 )*np.pi/180.    ,(88.2 +3.*1.2 )*np.pi/180. ],'physical':True},
        #                                       'aRs__plHAT_P33b':{'vary':True,'guess':0.2,'bd':[5.69-0.59*3,5.69+0.58*3],'physical':True}})         
        

    elif gen_dic['star_name']=='K2_105':
        glob_fit_dic['IntrProf']['mod_prop']={'veq':{'vary':True,'guess':0.,'bd':[0.5,3.],'physical':True},
                                      'lambda_rad__plK2_105b':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True},           
                                      'ctrst_ord0__ISHARPN_VS20200118':{'vary':True ,'guess':0.5,'bd':[0.,1.]},                       
                                      'FWHM_ord0__ISHARPN_VS20200118':{'vary':True ,'guess':5.,'bd':[1.,10.]}, 
                                       }

    elif gen_dic['star_name']=='HD89345':
        glob_fit_dic['IntrProf']['mod_prop']={'veq':{'vary':True,'guess':0.,'bd':[0.,4.],'physical':True},
                                      'lambda_rad__plHD89345b':{'vary':True,'guess':0.*np.pi/180.,'bd':[-80.*np.pi/180.,180.*np.pi/180.],'physical':True},           
                                      'ctrst_ord0__ISHARPN_VS20200202':{'vary':True ,'guess':0.5,'bd':[0.4,1.]},                       
                                      # 'ctrst_ord1__ISHARPN_VS20200202':{'vary':True ,'guess':0.5,'bd':[-1.,1.]},                         
                                      # 'ctrst_ord2__ISHARPN_VS20200202':{'vary':True ,'guess':0.5,'bd':[-1.,1.]},  
                                      'FWHM_ord0__ISHARPN_VS20200202':{'vary':True ,'guess':5.,'bd':[2.5,6.]}, 
                                      # 'FWHM_ord1__ISHARPN_VS20200202':{'vary':True ,'guess':0.,'bd':[-1.,1.]}, 
                                       }

        # glob_fit_dic['IntrProf']['mod_prop'].update({'c1_CB':{'vary':True,'guess':0.1,'bd':[-1.,1.],'physical':True}})

        # glob_fit_dic['IntrProf']['mod_prop'].update({'alpha_rot':{'vary':True,'guess':0.,'bd':[0.,1.],'physical':True},
        #                                       'cos_istar':{'vary':True,'guess':0.,'bd':[0.,1.],'physical':True}, 
        #                                       })

        # glob_fit_dic['IntrProf']['mod_prop'].update({'inclin_rad__plHD89345b':{'vary':True,'guess':0.,'bd':[  (87.68 -3.*0.1 )*np.pi/180.    ,(87.68 +3.*0.1 )*np.pi/180. ],'physical':True},
        #                                       'aRs__plHD89345b':{'vary':True,'guess':0.2,'bd':[13.625-0.027*3,13.625+0.027*3],'physical':True}})         
        
    elif gen_dic['star_name']=='HAT_P49':
        glob_fit_dic['IntrProf']['mod_prop']={'veq':{'vary':True,'guess':0.,'bd':[8.,12.],'physical':True},
                                      'lambda_rad__plHAT_P49b':{'vary':True,'guess':0.*np.pi/180.,'bd':[-120.*np.pi/180.,-90.*np.pi/180.],'physical':True},           
                                      'ctrst_ord0__ISHARPN_VS20200730':{'vary':True ,'guess':0.5,'bd':[0.4,1.]},        
                                      # 'ctrst_ord1__ISHARPN_VS20200730':{'vary':True ,'guess':0.5,'bd':[-1.,1.]}, 
                                      'FWHM_ord0__ISHARPN_VS20200730':{'vary':True ,'guess':18.,'bd':[15.,20.]},        
                                      # 'FWHM_ord1__ISHARPN_VS20200730':{'vary':True ,'guess':0.5,'bd':[-1.,1.]}, 
                                       }

        # glob_fit_dic['IntrProf']['mod_prop'].update({'c1_CB':{'vary':True,'guess':0.1,'bd':[-1.,1.],'physical':True}})

        # glob_fit_dic['IntrProf']['mod_prop'].update({'alpha_rot':{'vary':True,'guess':0.,'bd':[0.,1.],'physical':True},
        #                                       'cos_istar':{'vary':True,'guess':0.,'bd':[0.,1.],'physical':True}, 
        #                                       })

        # glob_fit_dic['IntrProf']['mod_prop'].update({'inclin_rad__plHAT_P49b':{'vary':True,'guess':0.,'bd':[  (86.2 -3.*1.7 )*np.pi/180.    ,(86.2 +3.*1.7 )*np.pi/180. ],'physical':True},
        #                                       'aRs__plHAT_P49b':{'vary':True,'guess':0.2,'bd':[5.13-0.30*3,5.13+0.19*3],'physical':True}})         
        

    elif gen_dic['star_name']=='Kepler63':
        glob_fit_dic['IntrProf']['mod_prop']={
            #'veq':{'vary':True,'guess':0.,'bd':[3.,8.],'physical':True},
                                      'lambda_rad__plKepler63b':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True},           
                                      'ctrst_ord0__ISHARPN_VS20200513':{'vary':True ,'guess':0.5,'bd':[0.4,1.]},   
                                      'FWHM_ord0__ISHARPN_VS20200513':{'vary':True ,'guess':10.,'bd':[7.,13.]},   
                                       }



        glob_fit_dic['IntrProf']['mod_prop'].update({
                        'cos_istar':{'vary':True,'guess':0.,'bd':[-1.,1.],'physical':True}, 
                        'Rstar':{'vary':True,'guess':0.,'bd':[0.85,0.95],'physical':True},        
                        'Peq':{'vary':True,'guess':0.,'bd':[5.3,5.5],'physical':True}})          
    
        

    elif gen_dic['star_name']=='WASP107':
        glob_fit_dic['IntrProf']['mod_prop']={
            # 'veq':{'vary':True,'guess':0.,'bd':[0.3,0.8],'physical':True},
                                      'lambda_rad__plWASP107b':{'vary':True,'guess':0.*np.pi/180.,'bd':[-200.*np.pi/180.,-100.*np.pi/180.],'physical':True}} 

        glob_fit_dic['IntrProf']['mod_prop'].update({
                        # 'a_damp__ISCARMENES_VIS_VS20180224':{'vary':True ,'guess':1.,'bd':[0.1,15.]},
                        'a_damp__ISCARMENES_VIS_VS20180224':{'vary':False ,'guess':4.,'bd':[0.1,15.]},
                        
                        #Common modulation for all datasets, common C0 for each instrument
                        'ctrst_ord0__ISCARMENES_VIS_VS_':{'vary':True ,'guess':0.5,'bd':[0.4,0.6]},
                        'ctrst_ord0__ISHARPS_VS_':{'vary':True ,'guess':0.5,'bd':[0.35,0.55]}, 
                        'ctrst_ord1__IS__VS_':{'vary':True ,'guess':0.,'bd':[0.1,0.5]},

                        #Common FWHM for HARPS datasets
                        'FWHM_ord0__ISCARMENES_VIS_VS20180224':{'vary':True ,'guess':7.,'bd':[0.7,1.2]},
                        'FWHM_ord0__ISHARPS_VS_':{'vary':True ,'guess':7.,'bd':[5.,5.6]},
                        })

        # glob_fit_dic['IntrProf']['mod_prop'].update({'c1_CB':{'vary':True,'guess':0.1,'bd':[-1.,1.],'physical':True}})

        # glob_fit_dic['IntrProf']['mod_prop'].update({'alpha_rot':{'vary':True,'guess':0.,'bd':[0.,1.],'physical':True},
        #                                       'cos_istar':{'vary':True,'guess':0.,'bd':[0.,1.],'physical':True}, 
        #                                       })


        # glob_fit_dic['IntrProf']['mod_prop'].update({'inclin_rad__plWASP107b':{'vary':True,'guess':0.,'bd':[  (89.56 -3.*0.078 )*np.pi/180.    ,(89.56 +3.*0.078 )*np.pi/180. ],'physical':True},
        #                                       'aRs__plWASP107b':{'vary':True,'guess':0.2,'bd':[18.02-0.27*3,18.02+0.27*3],'physical':True}})         
        

        glob_fit_dic['IntrProf']['mod_prop'].update({
                        'cos_istar':{'vary':True,'guess':0.,'bd':[-1.,1.],'physical':True}, 
                        'Rstar':{'vary':True,'guess':0.,'bd':[0.65,0.70],'physical':True},        
                        'Peq':{'vary':True,'guess':0.,'bd':[16.,18.],'physical':True}})          


    elif gen_dic['star_name']=='WASP166':
        glob_fit_dic['IntrProf']['mod_prop']={
                                      #'veq':{'vary':True,'guess':0.,'bd':[3.,8.],'physical':True},
                                      'lambda_rad__plWASP166b':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True},           
                                       'ctrst_ord0__ISHARPS_VS_':{'vary':True ,'guess':0.5,'bd':[0.4,1.]},   
                                      # 'ctrst_ord0__ISHARPS_VS20170114':{'vary':True ,'guess':10.,'bd':[0.4,1.]}, 
                                      # 'ctrst_ord0__ISHARPS_VS20170304':{'vary':True ,'guess':10.,'bd':[0.4,1.]}, 
                                      # 'ctrst_ord0__ISHARPS_VS20170315':{'vary':True ,'guess':10.,'bd':[0.4,1.]}, 
                                       'FWHM_ord0__ISHARPS_VS_':{'vary':True ,'guess':10.,'bd':[7.,13.]}, 
                                       # 'FWHM_ord1__ISHARPS_VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]}, 
                                      # 'FWHM_ord0__ISHARPS_VS20170114':{'vary':True ,'guess':10.,'bd':[7.,13.]},   
                                      # 'FWHM_ord0__ISHARPS_VS20170304':{'vary':True ,'guess':10.,'bd':[7.,13.]},    
                                      # 'FWHM_ord0__ISHARPS_VS20170315':{'vary':True ,'guess':10.,'bd':[7.,13.]},    


                                       }

        # glob_fit_dic['IntrProf']['mod_prop'].update({'c1_CB':{'vary':True,'guess':0.1,'bd':[-1.,1.],'physical':True}})
        # glob_fit_dic['IntrProf']['mod_prop'].update({'c2_CB':{'vary':True,'guess':0.1,'bd':[-1.,1.],'physical':True}})
        # glob_fit_dic['IntrProf']['mod_prop'].update({'c3_CB':{'vary':True,'guess':0.1,'bd':[-1.,1.],'physical':True}})


        # glob_fit_dic['IntrProf']['mod_prop'].update({'alpha_rot':{'vary':True,'guess':0.,'bd':[0.,1.],'physical':True},
        # 'cos_istar':{'vary':True,'guess':0.,'bd':[0.,1.],'physical':True},
        #                                       })

        # glob_fit_dic['IntrProf']['mod_prop'].update({'inclin_rad__plWASP166b':{'vary':True,'guess':0.,'bd':[ (87.95 -3.*0.59 )*np.pi/180.    ,(87.95 +3.*0.62 )*np.pi/180. ],'physical':True},
        # 'aRs__plWASP166b':{'vary':True,'guess':0.2,'bd':[11.14-0.50*3,11.14+0.42*3],'physical':True}}) 


        glob_fit_dic['IntrProf']['mod_prop'].update({
                        'cos_istar':{'vary':True,'guess':0.,'bd':[-1.,1.],'physical':True}, 
                        'Rstar':{'vary':True,'guess':0.,'bd':[1.1,1.4],'physical':True},        
                        'Peq':{'vary':True,'guess':0.,'bd':[9.,15.],'physical':True}})          


    elif gen_dic['star_name']=='HAT_P11':
        glob_fit_dic['IntrProf']['mod_prop']={'veq':{'vary':True,'guess':0.,'bd':[0.,5.],'physical':True},
                                      'lambda_rad__plHAT_P11b':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True},           
                                      'ctrst_ord0__ISCARMENES_VIS_VS_':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},   
                                      'ctrst_ord0__ISHARPN_VS_':{'vary':True ,'guess':10.,'bd':[0.1,0.6]},   
                                      # 'ctrst_ord1__IS__VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                                      # 'ctrst_ord2__IS__VS_':{'vary':True ,'guess':0.,'bd':[-1.,1.]},
                                      'FWHM_ord0__ISCARMENES_VIS_VS_':{'vary':True ,'guess':7.,'bd':[0.,15.]}, 
                                      'FWHM_ord0__ISHARPN_VS_':{'vary':True ,'guess':7.,'bd':[0.,15.]}, 
                                       }

        # glob_fit_dic['IntrProf']['pol_mode']='abs'
        glob_fit_dic['IntrProf']['mod_prop'].update({'a_damp__ISCARMENES_VIS_VS_':{'vary':True,'guess':4.,'bd':[0.,1.5]}})

        #Test corr C
        glob_fit_dic['IntrProf']['mod_prop']={'veq':{'vary':True,'guess':0.,'bd':[0.,5.],'physical':True},
                                      'lambda_rad__plHAT_P11b':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True},  
                                      'ctrst_ord0__ISHARPN_VS_':{'vary':True ,'guess':10.,'bd':[0.1,0.6]},   
                                      'FWHM_ord0__ISHARPN_VS_':{'vary':True ,'guess':7.,'bd':[0.,15.]}, 
                                       }

    elif gen_dic['star_name']=='HD106315':
        glob_fit_dic['IntrProf']['mod_prop']={'veq':{'vary':True,'guess':0.,'bd':[11.,16.],'physical':True},
                                      'lambda_rad__plHD106315c':{'vary':True,'guess':0.*np.pi/180.,'bd':[-50.*np.pi/180.,50.*np.pi/180.],'physical':True},           
                                      # 'ctrst_ord0__ISHARPS_VS20170309':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]}, 
                                      # 'ctrst_ord0__ISHARPS_VS20170330':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},                                       
                                      # 'ctrst_ord0__ISHARPS_VS20180323':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},                                     
                                       'ctrst_ord0__ISHARPS_VS_':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]}, 
                                      # 'FWHM_ord0__ISHARPS_VS20170309':{'vary':True ,'guess':0.5,'bd':[5.,20.]}, 
                                      # 'FWHM_ord0__ISHARPS_VS20170330':{'vary':True ,'guess':0.5,'bd':[5.,20.]},                                       
                                      # 'FWHM_ord0__ISHARPS_VS20180323':{'vary':True ,'guess':0.5,'bd':[5.,20.]},                                      
                                       'FWHM_ord0__ISHARPS_VS_':{'vary':True ,'guess':0.5,'bd':[5.,20.]}, 
   
                                        # #Common modulation
                                        # 'ctrst_ord1__IS__VS_':{'vary':True ,'guess':0.,'bd':[0.1,0.5]},
                                       
                                        
                                       # 'FWHM_ord1__ISHARPS_VS_':{'vary':True ,'guess':0.5,'bd':[-1.,1.]},
                                       }

        glob_fit_dic['IntrProf']['mod_prop'].update({'alpha_rot':{'vary':True,'guess':0.,'bd':[0.,1.],'physical':True},
        'cos_istar':{'vary':True,'guess':0.,'bd':[0.,1.],'physical':True}})


        glob_fit_dic['IntrProf']['mod_prop'].update({'c1_CB':{'vary':True,'guess':0.1,'bd':[-1.,1.],'physical':True}})
        glob_fit_dic['IntrProf']['mod_prop'].update({'c2_CB':{'vary':True,'guess':0.1,'bd':[-1.,1.],'physical':True}})

        
        
        
    elif gen_dic['star_name']=='WASP156':
        glob_fit_dic['IntrProf']['mod_prop']={'veq':{'vary':True,'guess':3.2660477746e+00,'bd':[0.,5.],'physical':True},
                                      'lambda_rad__plWASP156b':{'vary':True,'guess':1.6966466401e+00,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True},           
                                      # 'ctrst_ord0__ISCARMENES_VIS_VS20190928':{'vary':True ,'guess':4.8246003364e-01,'bd':[0.1,0.6]},   
                                       'ctrst_ord0__ISCARMENES_VIS_VS20191025':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},   
                                      # 'FWHM_ord0__ISCARMENES_VIS_VS20190928':{'vary':True ,'guess':8.0697590311e+00,'bd':[0.,15.]},   
                                       'FWHM_ord0__ISCARMENES_VIS_VS20191025':{'vary':True ,'guess':7.,'bd':[0.,15.]}, 
                                       # 'a_damp__ISCARMENES_VIS_VS20190928':{'vary':True ,'guess':0.5,'bd':[0.,1.5]}, 
                                      # 'a_damp__ISCARMENES_VIS_VS20191025':{'vary':True ,'guess':0.5,'bd':[0.,1.5]}, 
                                       }

        
    elif gen_dic['star_name']=='WASP47':
        glob_fit_dic['IntrProf']['mod_prop']={'veq':{'vary':True,'guess':0.,'bd':[1.,2.],'physical':True},
                                      'lambda_rad__plWASP47d':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True},           
                                      'ctrst_ord0__ISHARPN_VS20210730':{'vary':True ,'guess':0.5,'bd':[0.1,0.6]},     
                                      'FWHM_ord0__ISHARPN_VS20210730':{'vary':True ,'guess':7.,'bd':[0.,15.]},   
                                       }
        

    elif gen_dic['star_name']=='V1298tau':
        glob_fit_dic['IntrProf']['mod_prop']={'veq':{'vary':True,'guess':20.,'bd':[10.,30.],'physical':True},
                                      'lambda_rad__plV1298tau_b':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True},           
                                      'ctrst_ord0__ISHARPN_VSmock_vis':{'vary':True ,'guess':0.5,'bd':[0,1]},   
                                      'FWHM_ord0__ISHARPN_VSmock_vis':{'vary':True ,'guess':4.,'bd':[0.,10.]}, 
                                      
                                      # 'ctrst_ord0__ISHARPN_VSmock_vis2':{'vary':True ,'guess':0.5,'bd':[0,1]},   
                                      # 'FWHM_ord0__ISHARPN_VSmock_vis2':{'vary':True ,'guess':4.,'bd':[0.,10.]}, 
                                       }

    elif gen_dic['star_name']=='55Cnc':
        glob_fit_dic['IntrProf']['mod_prop']={'veq':{'vary':True,'guess':1.29,'bd':[1.,2.5],'physical':True},
                                        # 'lambda_rad__pl55Cnc_e':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.],'physical':True},           
                                        
                                        #  'lambda_rad__pl55Cnc_e__ISESPRESSO_VS20200205':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.]}, 
                                        # 'lambda_rad__pl55Cnc_e__ISESPRESSO_VS20210121':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.]}, 
                                        # 'lambda_rad__pl55Cnc_e__ISESPRESSO_VS20210124':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.]}, 
                                       
                                        # 'lambda_rad__pl55Cnc_e__ISHARPS_VS20120127':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.]}, 
                                        # 'lambda_rad__pl55Cnc_e__ISHARPS_VS20120213':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.]}, 
                                        # 'lambda_rad__pl55Cnc_e__ISHARPS_VS20120227':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.]}, 
                                        # 'lambda_rad__pl55Cnc_e__ISHARPS_VS20120315':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.]}, 
                                        
                                        #   'lambda_rad__pl55Cnc_e__ISHARPN_VS20131114':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.]}, 
                                        #   'lambda_rad__pl55Cnc_e__ISHARPN_VS20131128':{'vary':True,'guess':120.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.]}, 
                                        #   'lambda_rad__pl55Cnc_e__ISHARPN_VS20140101':{'vary':True,'guess':65.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.]}, 
                                        #   'lambda_rad__pl55Cnc_e__ISHARPN_VS20140126':{'vary':True,'guess':-70.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.]}, 
                                        #   'lambda_rad__pl55Cnc_e__ISHARPN_VS20140226':{'vary':True,'guess':-130.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.]}, 
                                        #   'lambda_rad__pl55Cnc_e__ISHARPN_VS20140329':{'vary':True,'guess':-80.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.]}, 

                                        'lambda_rad__pl55Cnc_e__ISEXPRES_VS20220131':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.]}, 
                                        'lambda_rad__pl55Cnc_e__ISEXPRES_VS20220406':{'vary':True,'guess':0.*np.pi/180.,'bd':[-180.*np.pi/180.,180.*np.pi/180.]}, 


                                        # 'ctrst_ord0__ISESPRESSO_VS20200205':{'vary':True ,'guess':0.5,'bd':[0.55,0.7]},     
                                        # 'FWHM_ord0__ISESPRESSO_VS20200205':{'vary':True ,'guess':7.,'bd':[4.,5.5]},
                                        # 'ctrst_ord0__ISESPRESSO_VS20210121':{'vary':True ,'guess':0.5,'bd':[0.5,0.7]},     
                                        # 'FWHM_ord0__ISESPRESSO_VS20210121':{'vary':True ,'guess':7.,'bd':[4.,5.]},
                                        # 'ctrst_ord0__ISESPRESSO_VS20210124':{'vary':True ,'guess':0.5,'bd':[0.5,0.7]},    
                                        # 'FWHM_ord0__ISESPRESSO_VS20210124':{'vary':True ,'guess':7.,'bd':[4.,5.]},
                                        # 'ctrst_ord0__ISESPRESSO_VS_':{'vary':True  ,'guess':6.2323597771e-01,'bd':[0.5,0.7]},    
                                        # 'FWHM_ord0__ISESPRESSO_VS_':{'vary':True  ,'guess':4.7913655970,'bd':[4.,5.]},
                                        
                                        # 'ctrst_ord0__ISHARPS_VS20120127':{'vary':True ,'guess':0.5,'bd':[0.5,0.7]},    
                                        # 'FWHM_ord0__ISHARPS_VS20120127':{'vary':True ,'guess':7.,'bd':[4.,5.]},
                                        # 'ctrst_ord0__ISHARPS_VS20120213':{'vary':True  ,'guess':6.2323597771e-01,'bd':[0.5,0.7]},    
                                        # 'FWHM_ord0__ISHARPS_VS20120213':{'vary':True  ,'guess':4.7913655970,'bd':[4.,5.]},                                         
                                        # 'ctrst_ord0__ISHARPS_VS20120227':{'vary':True ,'guess':0.5,'bd':[0.5,0.7]},    
                                        # 'FWHM_ord0__ISHARPS_VS20120227':{'vary':True ,'guess':7.,'bd':[4.,5.]},
                                        # 'ctrst_ord0__ISHARPS_VS20120315':{'vary':True  ,'guess':6.2323597771e-01,'bd':[0.5,0.7]},    
                                        # 'FWHM_ord0__ISHARPS_VS20120315':{'vary':True  ,'guess':4.7913655970,'bd':[4.,5.]}, 
                                        
                                        # 'ctrst_ord0__ISHARPN_VS20131114':{'vary':True ,'guess':0.5,'bd':[0.5,0.7]},    
                                        # 'FWHM_ord0__ISHARPN_VS20131114':{'vary':True ,'guess':7.,'bd':[4.,5.]},
                                        # 'ctrst_ord0__ISHARPN_VS20131128':{'vary':True  ,'guess':6.2323597771e-01,'bd':[0.5,0.7]},    
                                        # 'FWHM_ord0__ISHARPN_VS20131128':{'vary':True  ,'guess':4.7913655970,'bd':[4.,5.]}, 
                                        # 'ctrst_ord0__ISHARPN_VS20140101':{'vary':True ,'guess':0.5,'bd':[0.5,0.7]},    
                                        # 'FWHM_ord0__ISHARPN_VS20140101':{'vary':True ,'guess':7.,'bd':[4.,5.]},
                                        # 'ctrst_ord0__ISHARPN_VS20140126':{'vary':True  ,'guess':6.2323597771e-01,'bd':[0.5,0.7]},    
                                        # 'FWHM_ord0__ISHARPN_VS20140126':{'vary':True  ,'guess':4.7913655970,'bd':[4.,5.]},                                         
                                        # 'ctrst_ord0__ISHARPN_VS20140226':{'vary':True ,'guess':0.5,'bd':[0.5,0.7]},    
                                        # 'FWHM_ord0__ISHARPN_VS20140226':{'vary':True ,'guess':7.,'bd':[4.,5.]},
                                        # 'ctrst_ord0__ISHARPN_VS20140329':{'vary':True  ,'guess':6.2323597771e-01,'bd':[0.5,0.7]},    
                                        # 'FWHM_ord0__ISHARPN_VS20140329':{'vary':True  ,'guess':4.7913655970,'bd':[4.,5.]}, 

                                        # 'ctrst_ord0__ISEXPRES_VS20220131':{'vary':True ,'guess':0.5,'bd':[0.5,0.7]},    
                                        # 'FWHM_ord0__ISEXPRES_VS20220131':{'vary':True ,'guess':7.,'bd':[4.,5.]},
                                        # 'ctrst_ord0__ISEXPRES_VS20220406':{'vary':True  ,'guess':6.2323597771e-01,'bd':[0.5,0.7]},    
                                        # 'FWHM_ord0__ISEXPRES_VS20220406':{'vary':True  ,'guess':4.7913655970,'bd':[4.,5.]},
                                        
                                        'ctrst_ord0__IS__VS_':{'vary':True   ,'guess':0.76,'bd':[0.5,0.7]},    
                                        'FWHM_ord0__IS__VS_':{'vary':True   ,'guess':4.59,'bd':[4.,5.]},
                                        }

    if gen_dic['star_name']=='HD209458':  
        glob_fit_dic['IntrProf']['mod_prop']={
                    # 'cont':{'vary':True ,'guess':1.0,'bd':[0.9,1.4]},      #CCFs from DI CCFs             
                    'cont':{'vary':True ,'guess':13.,'bd':[12.9,13.1]},      #CCFs from Intr spectra     
                    'veq':{'vary':True,'guess':4.2721415478,'bd':[4.2,4.3]},
                    'lambda_rad__plHD209458b':{'vary':True,'guess':1.0699092308*np.pi/180.,'bd':[0.*np.pi/180.,2.*np.pi/180.]},
                    'ctrst_ord0__ISESPRESSO_VS20190720':{'vary':True ,'guess':0.65,'bd':[0.6,0.7]},
                    'ctrst_ord0__ISESPRESSO_VS20190911':{'vary':True ,'guess':0.65,'bd':[0.6,0.7]},
                    'ctrst_ord1__ISESPRESSO_VS_':{'vary':True ,'guess':-0.2,'bd':[-0.2,-0.1]},
                    'FWHM_ord0__ISESPRESSO_VS20190720':{'vary':True ,'guess':5.5,'bd':[5.3,6.]},
                    'FWHM_ord0__ISESPRESSO_VS20190911':{'vary':True ,'guess':5.5,'bd':[5.4,6.]},                        
                    'FWHM_ord1__ISESPRESSO_VS_':{'vary':True ,'guess':0.5,'bd':[0.2,0.6]}

                    #Fit Na doublet
                    # 'rv':{'vary':True,'guess':7.4590634981e-01,'bd':[0.,2.]},    
                    # 'abund_Na':{'vary':True ,'guess':6.19,'bd':[6.,6.2]},      #Init SME grid
                    # 'abund_Na':{'vary':False ,'guess':6.0450212411e+00,'bd':[4.,10.]},  #Best-fit DI  
                    }   
        
    if gen_dic['star_name']=='WASP76':  
        glob_fit_dic['IntrProf']['mod_prop']={
                    # 'cont':{'vary':True ,'guess':1.0,'bd':[0.9,1.4]},      #CCFs from DI CCFs         
                    'cont':{'vary':True ,'guess':13.225,'bd':[13.21,13.235]},      #CCFs from Intr spectra
                    'veq':{'vary':True,'guess':3.,'bd':[0.6,2.5]},
                    'lambda_rad__plWASP76b':{'vary':True,'guess':-80.*np.pi/180.,'bd':[-80.*np.pi/180.,80.*np.pi/180.]},
                    'ctrst_ord0__ISESPRESSO_VS20180902':{'vary':True ,'guess':0.65,'bd':[0.62,0.68]},
                    'ctrst_ord0__ISESPRESSO_VS20181030':{'vary':True ,'guess':0.65,'bd':[0.62,0.68]},
                    'ctrst_ord1__ISESPRESSO_VS_':{'vary':True ,'guess':-0.1,'bd':[-0.25,0.15]},
                    'ctrst_ord2__ISESPRESSO_VS_':{'vary':True ,'guess':-0.1,'bd':[-0.3,0.05]},
                    'FWHM_ord0__ISESPRESSO_VS20180902':{'vary':True ,'guess':7.5,'bd':[7.1,8.2]},
                    'FWHM_ord0__ISESPRESSO_VS20181030':{'vary':True ,'guess':7.5,'bd':[7.2,8.3]},                        
                    'FWHM_ord1__ISESPRESSO_VS_':{'vary':True ,'guess':-0.04,'bd':[-0.4,0.15]},                 
                    'FWHM_ord2__ISESPRESSO_VS_':{'vary':True ,'guess':0.3,'bd':[0.2,0.65]}
                    }                       
        glob_fit_dic['IntrProf']['mod_prop'].update({'inclin_rad__plWASP76b':{'vary':True,'guess':0.,'bd':[ (89.623 -3.*0.034   )*np.pi/180.    ,(89.623 +3.*0.005 )*np.pi/180. ],'physical':True},
                                                     'aRs__plWASP76b':{'vary':True,'guess':0.2,'bd':[4.08-0.06*3,4.08+0.02*3],'physical':True}}) 

        
        
    #PC noise model
    # if gen_dic['star_name']=='55Cnc':
    #     glob_fit_dic['IntrProf']['PC_model']={
    #         'ESPRESSO':{
    #             # '20200205':{'noPC':False,'idx_out':'all','PC_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/5PC/ESPRESSO_20200205_aligned.npz'}}},
    #             '20200205':{'noPC':False,'idx_out':'all','PC_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/5PC/ESPRESSO_20200205_mooncorr_aligned.npz'},           
    #             '20210121':{'noPC':False,'idx_out':'all','PC_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/6PC/ESPRESSO_20210121_aligned.npz'},    
    #             '20210124':{'noPC':False,'idx_out':'all','PC_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/ESPRESSO/Range200/3PC/ESPRESSO_20210124_aligned.npz'},
    #             },
    #         # 'HARPS':{
    #         #     # '20120127':{'noPC':True,'idx_out':'all','PC_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPS/Range200/1PC/Aligned/HARPS_20120127.npz'},           
    #         #     '20120213':{'noPC':True,'idx_out':'all','PC_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPS/Range200/1PC/Aligned/HARPS_20120213.npz'},    
    #         # #     # '20120227':{'noPC':True,'idx_out':'all','PC_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPS/Range200/1PC/Aligned/HARPS_20120227.npz'},
    #         # #     '20120315':{'noPC':True,'idx_out':'all','PC_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPS/Range200/1PC/Aligned/HARPS_20120315.npz'},
    #         #     } ,    
            
    #         'HARPN':{
    #             '20131114':{'noPC':False,'idx_out':'all','PC_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/1PC/Aligned/HARPN_20131114.npz'},   #FINAL
    #             '20131128':{'noPC':False,'idx_out':'all','PC_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/1PC/Aligned/HARPN_20131128.npz'},   #FINAL
    #             '20140101':{'noPC':False,'idx_out':'all','PC_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/8PC/Aligned/HARPN_20140101.npz'},   #FINAL
    #             '20140126':{'noPC':False,'idx_out':'all','PC_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/4PC/Aligned/HARPN_20140126.npz'},   #FINAL
    #             '20140226':{'noPC':False,'idx_out':'all','PC_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/5PC/Aligned/HARPN_20140226.npz'},   #FINAL
    #             '20140329':{'noPC':False,'idx_out':'all','PC_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/HARPN/Range200/3PC/Aligned/HARPN_20140329.npz'},   #FINAL
    #             }, 

    #         'EXPRES':{
    #             '20220131':{'noPC':False,'idx_out':'all','PC_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/EXPRES/Range200_clean/10PC/Aligned/EXPRES_20220131.npz'},   
    #             '20220406':{'noPC':False,'idx_out':'all','PC_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/PCA_results/EXPRES/Range200_clean/10PC/Aligned/EXPRES_20220406.npz'},   
    #             } 

    #         }    

    
    #Fitting mode
    glob_fit_dic['IntrProf']['fit_mod']='chi2' 
    glob_fit_dic['IntrProf']['fit_mod']='mcmc' 


    #Printing fits results
    glob_fit_dic['IntrProf']['verbose']=True   #& False
    
    #Priors on variable properties
    if list(gen_dic['transit_pl'].keys())==['HD3167_b']:  
        glob_fit_dic['IntrProf']['priors'].update({
            # 'ctrst0':{'mod':'uf','low':-2.,'high':2.},  
            'FWHM0':{'mod':'uf','low':0.,'high':20.},

        # fixed_args['varpar_priors'].update({
            'ctrst0':{'mod':'uf','low':-1e5,'high':1e5},  
            'ctrst1':{'mod':'uf','low':-1e5,'high':1e5}})  
            # 'ctrst2':{'mod':'uf','low':-1e5,'high':1e5}})  
            # 'FWHM0':{'mod':'uf','low':-1e5,'high':1e5}})  
            # 'FWHM1':{'mod':'uf','low':-1e5,'high':1e5}})  
            # 'FWHM2':{'mod':'uf','low':-1e5,'high':1e5}})                 
        
    elif list(gen_dic['transit_pl'].keys())==['HD3167_c']: 
        glob_fit_dic['IntrProf']['priors'].update({
        #     'ctrst0':{'mod':'uf','low':-10.,'high':10.},  
            'FWHM0':{'mod':'uf','low':0.,'high':20.}}) 

        glob_fit_dic['IntrProf']['priors'].update({
            'ctrst0':{'mod':'uf','low':-1e5,'high':1e5},  
            'ctrst1':{'mod':'uf','low':-1e5,'high':1e5},  
            # 'ctrst2':{'mod':'uf','low':-1e5,'high':1e5},  
            # 'FWHM0':{'mod':'uf','low':-1e5,'high':1e5},  
            # 'FWHM1':{'mod':'uf','low':-1e5,'high':1e5},  
            # 'FWHM2':{'mod':'uf','low':-1e5,'high':1e5}
            }) 

    elif ('HD3167_b' in gen_dic['transit_pl']) and ('HD3167_c' in gen_dic['transit_pl']):
        glob_fit_dic['IntrProf']['priors'].update({
            'ctrst0':{'mod':'uf','low':-1e5,'high':1e5},  
            'ctrst1':{'mod':'uf','low':-1e5,'high':1e5},  
            'ctrst2':{'mod':'uf','low':-1e5,'high':1e5},  
            'FWHM0':{'mod':'uf','low':-1e5,'high':1e5}})
            # 'FWHM1':{'mod':'uf','low':-1e5,'high':1e5}})

        # glob_fit_dic['IntrProf']['priors'].update({
        #     'ctrst0':{'mod':'uf','low':-2.,'high':2.},  
        #     'FWHM0':{'mod':'uf','low':0.,'high':20.}})

    elif 'Corot7b' in gen_dic['transit_pl']:  
        glob_fit_dic['IntrProf']['priors'].update({
            'ctrst0':{'mod':'uf','low':0.,'high':1.},  
            'FWHM0':{'mod':'uf','low':0.,'high':10.}}) 
    elif 'GJ9827d' in gen_dic['transit_pl']:  
        glob_fit_dic['IntrProf']['priors'].update({
            'ctrst0':{'mod':'uf','low':0.,'high':1.},  
            'FWHM0':{'mod':'uf','low':0.,'high':15.}}) 
    elif 'GJ9827b' in gen_dic['transit_pl']:  
        glob_fit_dic['IntrProf']['priors'].update({
            'ctrst0':{'mod':'uf','low':0.,'high':1.},  
            'FWHM0':{'mod':'uf','low':0.,'high':15.}}) 

    elif 'TOI858b' in gen_dic['transit_pl']:
        # glob_fit_dic['IntrProf']['priors'].update({
        #     'ctrst0':{'mod':'uf','low':-2.,'high':2.},  
        #     'FWHM0':{'mod':'uf','low':0.,'high':30.}})

        glob_fit_dic['IntrProf']['priors'].update({
            'ctrst0__ISCORALIE_VS20191205':{'mod':'uf','low':-2.,'high':2.},  
            'FWHM0__ISCORALIE_VS20191205':{'mod':'uf','low':0.,'high':30.},
            'ctrst0__ISCORALIE_VS20210118':{'mod':'uf','low':-2.,'high':2.},  
            'FWHM0__ISCORALIE_VS20210118':{'mod':'uf','low':0.,'high':30.}})        

        
    elif 'GJ436_b' in gen_dic['transit_pl']:
        glob_fit_dic['IntrProf']['priors'].update({
            'veq':{'mod':'uf','low':0.,'high':10.},  
            'lambda_rad__GJ36_b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}})
            
        
        # glob_fit_dic['IntrProf']['priors'].update({
        #     'RV_l2c__ISESPRESSO_VS20190228':{'mod':'uf','low':-10.,'high':10.},  
        #     'amp_l2c__ISESPRESSO_VS20190228':{'mod':'uf','low':0.,'high':2.},
        #     'FWHM_l2c__ISESPRESSO_VS20190228':{'mod':'uf','low':0.,'high':10.},  
        #     'RV_l2c__ISESPRESSO_VS20190429':{'mod':'uf','low':-10.,'high':10.},
        #     'amp_l2c__ISESPRESSO_VS20190429':{'mod':'uf','low':0.,'high':2.},
        #     'FWHM_l2c__ISESPRESSO_VS20190429':{'mod':'uf','low':0.,'high':10.}})    

        # glob_fit_dic['IntrProf']['priors'].update({
        #     'RV_l2c__IS__VS_':{'mod':'uf','low':-10.,'high':10.},  
        #     'amp_l2c__IS__VS_':{'mod':'uf','low':0.,'high':2.},
        #     'FWHM_l2c__IS__VS_':{'mod':'uf','low':0.,'high':10.}})  

        glob_fit_dic['IntrProf']['priors'].update({
            'ctrst_ord0__IS__VS_':{'mod':'uf','low':0.,'high':1.},  
            'FWHM_ord0__IS__VS_':{'mod':'uf','low':0.,'high':10.},            
            'ctrst_ord0__ISESPRESSO_VS20190228':{'mod':'uf','low':0.,'high':1.},  
            'FWHM_ord0__ISESPRESSO_VS20190228':{'mod':'uf','low':0.,'high':10.},
            'ctrst_ord0__ISESPRESSO_VS20190429':{'mod':'uf','low':0.,'high':1.},  
            'FWHM_ord0__ISESPRESSO_VS20190429':{'mod':'uf','low':0.,'high':6.},
            'ctrst_ord0__ISESPRESSO_VS_':{'mod':'uf','low':0.,'high':1.},  
            'FWHM_ord0__ISESPRESSO_VS_':{'mod':'uf','low':0.,'high':10.},
            'ctrst_ord0__ISHARPN_VS20160318':{'mod':'uf','low':0.,'high':1.},  
            'FWHM_ord0__ISHARPN_VS20160318':{'mod':'uf','low':0.,'high':10.},  
            'ctrst_ord0__ISHARPN_VS20160411':{'mod':'uf','low':0.,'high':1.},  
            'FWHM_ord0__ISHARPN_VS20160411':{'mod':'uf','low':0.,'high':10.},  
            'ctrst_ord0__ISHARPN_VS_':{'mod':'uf','low':0.,'high':1.},  
            'FWHM_ord0__ISHARPN_VS_':{'mod':'uf','low':0.,'high':10.},
            'ctrst_ord0__ISHARPS_VS20070509':{'mod':'uf','low':0.,'high':1.},  
            'FWHM_ord0__ISHARPS_VS20070509':{'mod':'uf','low':0.,'high':10.}})      


        # glob_fit_dic['IntrProf']['priors'].update({ 
        #     'aRs__GJ436_b':{'mod':'gauss','val':14.46,'s_val':0.08}})         #j'ai mis un prior sur sini (le parametre donne dans Maxted+) par l'intermediaire de la fonction de prior   
        
            
        
    elif 'HIP41378d' in gen_dic['transit_pl']:
        glob_fit_dic['IntrProf']['priors'].update({
            'veq':{'mod':'uf','low':0.,'high':20.},  
            'lambda_rad__plHIP41378d':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
            'ctrst_ord0__IS__VS_':{'mod':'uf','low':0.,'high':1.},  
            'FWHM_ord0__IS__VS_':{'mod':'uf','low':0.,'high':30.}})

    #RM survey
    elif gen_dic['star_name']=='HAT_P3': 
        glob_fit_dic['IntrProf']['priors'].update({
            # 'veq':{'mod':'uf','low':0.,'high':20.},  
            'lambda_rad__plHAT_P3b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}})

        glob_fit_dic['IntrProf']['priors'].update({         
            'ctrst_ord0__IS__VS_':{'mod':'uf','low':0.,'high':1.}, 
            'FWHM_ord0__IS__VS_':{'mod':'uf','low':0.,'high':30.},       
            })

        # glob_fit_dic['IntrProf']['priors'].update({ 
        #     'inclin_rad__plHAT_P3b':{'mod':'gauss','val':86.31*np.pi/180.,'s_val':0.19*np.pi/180.},     
        #     'aRs__plHAT_P3b':{'mod':'gauss','val':9.8105,'s_val':0.2667}})     

        glob_fit_dic['IntrProf']['priors'].update({         
            'Rstar':{'mod':'gauss','val':0.85,'s_val':0.021}, 
            'Peq':{'mod':'gauss','val':19.9,'s_val':1.5},       
            'cos_istar':{'mod':'uf','low':-1.,'high':1.},     
            })

    elif gen_dic['star_name']=='Kepler25': 
        glob_fit_dic['IntrProf']['priors'].update({
            # 'veq':{'mod':'uf','low':0.,'high':30.},  
            # 'veq':{'mod':'dgauss','val':9.34,'s_val_low':0.39,'s_val_high':0.37},       #de Benomar+2014 sur vsini (ici istar = 90 donc equivalent de mettre le prior sur veq)
            'veq':{'mod':'dgauss','val':9.13,'s_val_low':0.69,'s_val_high':0.60},       #de Benomar+2014 sur vsini astero pure (ici istar = 90 donc equivalent de mettre le prior sur veq)
            'lambda_rad__plKepler25c':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
            'ctrst_ord0__ISHARPN_VS20190614':{'mod':'uf','low':0.,'high':1.},  
            'FWHM_ord0__ISHARPN_VS20190614':{'mod':'uf','low':0.,'high':30.},  
            
            })

    elif gen_dic['star_name']=='Kepler68': 
        glob_fit_dic['IntrProf']['priors'].update({
            'veq':{'mod':'uf','low':0.,'high':2.},  
            # 'veq':{'mod':'gauss','val':0.5,'s_val':0.5},                  
            'lambda_rad__plKepler68b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
            'ctrst_ord0__ISHARPN_VS20190803':{'mod':'uf','low':0.,'high':1.},  
            'FWHM_ord0__ISHARPN_VS20190803':{'mod':'uf','low':0.,'high':30.},  
            
            })

    elif gen_dic['star_name']=='HAT_P33': 
        glob_fit_dic['IntrProf']['priors'].update({
            'veq':{'mod':'uf','low':0.,'high':40.},  
            'lambda_rad__plHAT_P33b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}})

        glob_fit_dic['IntrProf']['priors'].update({         
            'FWHM_ord0__ISHARPN_VS20191204':{'mod':'uf','low':0.,'high':30.},       
            })
        
        # glob_fit_dic['IntrProf']['priors'].update({         
        #     'alpha_rot':{'mod':'uf','low':0.,'high':1.},        
        #     'cos_istar':{'mod':'uf','low':-1.,'high':1.},
        #     })
        
        # glob_fit_dic['IntrProf']['priors'].update({ 
        #     'inclin_rad__plHAT_P33b':{'mod':'gauss','val':88.2*np.pi/180.,'s_val':1.3*np.pi/180.},     
        #     'aRs__plHAT_P33b':{'mod':'gauss','val':5.69,'s_val':0.59}})             




    elif gen_dic['star_name']=='K2_105':
        glob_fit_dic['IntrProf']['priors'].update({
            'veq':{'mod':'uf','low':0.,'high':5.},  
            # 'veq':{'mod':'gauss','val':1.76,'s_val':0.86}, 
            'lambda_rad__plK2_105b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
            'ctrst_ord0__ISHARPN_VS20200118':{'mod':'uf','low':0.,'high':1.},  
            'FWHM_ord0__ISHARPN_VS20200118':{'mod':'uf','low':0.,'high':30.},  
            })

    elif gen_dic['star_name']=='HD89345': 
        glob_fit_dic['IntrProf']['priors'].update({
            'veq':{'mod':'uf','low':0.,'high':40.},  
            # 'veq':{'mod':'gauss','val':2.6,'s_val':0.5},   
            'lambda_rad__plHD89345b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}})

        # glob_fit_dic['IntrProf']['priors'].update({         
        #     'alpha_rot':{'mod':'uf','low':0.,'high':1.},        
        #     'cos_istar':{'mod':'uf','low':-1.,'high':1.},
        #     })

        # glob_fit_dic['IntrProf']['priors'].update({ 
        #     'inclin_rad__plHD89345b':{'mod':'gauss','val':87.68*np.pi/180.,'s_val':0.1*np.pi/180.},     
        #     'aRs__plHD89345b':{'mod':'gauss','val':13.625,'s_val':0.027}}) 

    elif gen_dic['star_name']=='Kepler63': 
        glob_fit_dic['IntrProf']['priors'].update({
            # 'veq':{'mod':'uf','low':0.,'high':30.},    
            'lambda_rad__plKepler63b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
            'ctrst_ord0__ISHARPN_VS20200513':{'mod':'uf','low':0.,'high':1.},  
            # 'FWHM_ord0__ISHARPN_VS20200513':{'mod':'uf','low':0.,'high':20.}})
            # 'FWHM_ord0__ISHARPN_VS20200513':{'mod':'gauss','val':7.0,'s_val':0.64}})            
            'FWHM_ord0__ISHARPN_VS20200513':{'mod':'gauss','val':9.5,'s_val':4.}})           

        glob_fit_dic['IntrProf']['priors'].update({         
            'Rstar':{'mod':'dgauss','val':0.901,'s_val_high':0.027,'s_val_low':0.022}, 
            'Peq':{'mod':'gauss','val':5.401,'s_val':0.014},       
            'cos_istar':{'mod':'uf','low':-1.,'high':0.},   #to keep southern solution     
            })
        

    elif gen_dic['star_name']=='HAT_P49': 
        glob_fit_dic['IntrProf']['priors'].update({
            'veq':{'mod':'uf','low':0.,'high':30.},    
            'lambda_rad__plHAT_P49b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}})

        glob_fit_dic['IntrProf']['priors'].update({
            'ctrst_ord0__ISHARPN_VS20200730':{'mod':'uf','low':0.,'high':1.},  
            'FWHM_ord0__ISHARPN_VS20200730':{'mod':'uf','low':0.,'high':40.}})

        # glob_fit_dic['IntrProf']['priors'].update({         
        #     'alpha_rot':{'mod':'uf','low':0.,'high':1.},        
        #     'cos_istar':{'mod':'uf','low':-1.,'high':1.},
        #     })

        # glob_fit_dic['IntrProf']['priors'].update({ 
        #     'inclin_rad__plHAT_P49b':{'mod':'gauss','val':86.2*np.pi/180.,'s_val':1.7*np.pi/180.},     
        #     'aRs__plHAT_P49b':{'mod':'dgauss','val':5.13,'s_val_low':0.30,'s_val_high':0.19}}) 
        

    elif gen_dic['star_name']=='WASP107': 
        glob_fit_dic['IntrProf']['priors'].update({
            'veq':{'mod':'uf','low':0.,'high':30.},    
            'lambda_rad__plWASP107b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
            # 'a_damp__ISCARMENES_VIS_VS20180224':{'mod':'uf','low':0.,'high':50.},                
            'FWHM_ord0__ISCARMENES_VIS_VS20180224':{'mod':'uf','low':0.,'high':20.},
            'FWHM_ord0__ISHARPS_VS_':{'mod':'uf','low':0.,'high':20.},            
            })

        # glob_fit_dic['IntrProf']['priors'].update({         
        #     'alpha_rot':{'mod':'uf','low':0.,'high':1.},        
        #     'cos_istar':{'mod':'uf','low':-1.,'high':1.},
        #     })


        # glob_fit_dic['IntrProf']['priors'].update({ 
        #     'inclin_rad__plWASP107b':{'mod':'gauss','val':89.56*np.pi/180.,'s_val':0.078*np.pi/180.},     
        #     'aRs__plWASP107b':{'mod':'dgauss','val':18.02,'s_val_low':0.27,'s_val_high':0.27}}) 

        glob_fit_dic['IntrProf']['priors'].update({         
            'Rstar':{'mod':'gauss','val':0.67,'s_val':0.02}, 
            'Peq':{'mod':'gauss','val':17.1,'s_val':1.},       
            'cos_istar':{'mod':'uf','low':-1.,'high':1.},  
            })

    elif gen_dic['star_name']=='WASP166': 

        glob_fit_dic['IntrProf']['priors'].update({
            #'veq':{'mod':'uf','low':0.,'high':30.},    
            'lambda_rad__plWASP166b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}})        

        # glob_fit_dic['IntrProf']['priors'].update({
        #     'FWHM_ord0__ISHARPS_VS20170114':{'mod':'uf','low':0.,'high':50.}, 
        #     'FWHM_ord0__ISHARPS_VS20170304':{'mod':'uf','low':0.,'high':50.},
        #     'FWHM_ord0__ISHARPS_VS20170315':{'mod':'uf','low':0.,'high':50.}})

        glob_fit_dic['IntrProf']['priors'].update({
            'ctrst_ord0__ISHARPS_VS_':{'mod':'uf','low':0.,'high':1.}, 
            'FWHM_ord0__ISHARPS_VS_':{'mod':'uf','low':0.,'high':20.}
            }) 
            
        # glob_fit_dic['IntrProf']['priors'].update({
        #     'ctrst_ord0__ISHARPS_VS20170114':{'mod':'uf','low':0.,'high':1.}, 
        #     'ctrst_ord0__ISHARPS_VS20170304':{'mod':'uf','low':0.,'high':1.},
        #     'ctrst_ord0__ISHARPS_VS20170315':{'mod':'uf','low':0.,'high':1.}})

        # glob_fit_dic['IntrProf']['priors'].update({
        #     'alpha_rot':{'mod':'uf','low':-1.,'high':1.},
        #     'cos_istar':{'mod':'uf','low':-1.,'high':1.},
        #     })

        # glob_fit_dic['IntrProf']['priors'].update({
        # 'inclin_rad__plWASP166b':{'mod':'dgauss','val':87.95*np.pi/180.,'s_val_low':0.62*np.pi/180.,'s_val_high':0.59*np.pi/180.},
        # 'aRs__plWASP166b':{'mod':'dgauss','val':11.14,'s_val_low':0.50,'s_val_high':0.42}}) 

        glob_fit_dic['IntrProf']['priors'].update({         
            'Rstar':{'mod':'gauss','val':1.22,'s_val':0.06}, 
            'Peq':{'mod':'gauss','val':12.3,'s_val':1.9},       
            'cos_istar':{'mod':'uf','low':-1.,'high':1.},  
            })

    elif gen_dic['star_name']=='HAT_P11': 
        glob_fit_dic['IntrProf']['priors'].update({
            'veq':{'mod':'uf','low':0.,'high':4.},    
            'lambda_rad__plHAT_P11b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}})

        glob_fit_dic['IntrProf']['priors'].update({
            'FWHM_ord0__ISCARMENES_VIS_VS_':{'mod':'uf','low':0.,'high':15.},  
            'FWHM_ord0__ISHARPN_VS_':{'mod':'uf','low':0.,'high':15.}})


        glob_fit_dic['IntrProf']['priors'].update({
            'a_damp__ISCARMENES_VIS_VS_':{'mod':'uf','low':0.,'high':10.}}) 


    elif gen_dic['star_name']=='HD106315': 
        glob_fit_dic['IntrProf']['priors'].update({
            'veq':{'mod':'uf','low':0.,'high':100.},    
            'lambda_rad__plHD106315c':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}})

        # glob_fit_dic['IntrProf']['priors'].update({
        #     'ctrst_ord0__ISHARPS_VS20170309':{'mod':'uf','low':0.,'high':1.},
        #     'ctrst_ord0__ISHARPS_VS20170330':{'mod':'uf','low':0.,'high':1.},
        #     'ctrst_ord0__ISHARPS_VS20180323':{'mod':'uf','low':0.,'high':1.}})

        glob_fit_dic['IntrProf']['priors'].update({
            'ctrst_ord0__ISHARPS_VS_':{'mod':'uf','low':0.,'high':1.},
            'FWHM_ord0__ISHARPS_VS_':{'mod':'uf','low':0.,'high':40.}})
                                              

        glob_fit_dic['IntrProf']['priors'].update({
        #     # 'FWHM_ord0__ISHARPS_VS20170309':{'mod':'uf','low':0.,'high':40.},
        #     # 'FWHM_ord0__ISHARPS_VS20170330':{'mod':'uf','low':0.,'high':40.},
        #     # 'FWHM_ord0__ISHARPS_VS20180323':{'mod':'uf','low':0.,'high':40.}})   
            'FWHM_ord0__ISHARPS_VS_':{'mod':'uf','low':0.,'high':40.}})          

        glob_fit_dic['IntrProf']['priors'].update({
            'alpha_rot':{'mod':'uf','low':0.,'high':1.},
            'cos_istar':{'mod':'uf','low':-1.,'high':1.},
            })



    elif gen_dic['star_name']=='WASP156':
        glob_fit_dic['IntrProf']['priors'].update({
            # 'veq':{'mod':'uf','low':0.,'high':100.}, 
            'veq':{'mod':'gauss','val':3.8,'s_val':0.91},   
            'lambda_rad__plWASP156b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}})        
        
        glob_fit_dic['IntrProf']['priors'].update({
            'ctrst_ord0__ISCARMENES_VIS_VS20190928':{'mod':'uf','low':0.,'high':1.},
            'ctrst_ord0__ISCARMENES_VIS_VS20191025':{'mod':'uf','low':0.,'high':1.}
            })

        glob_fit_dic['IntrProf']['priors'].update({
            'FWHM_ord0__ISCARMENES_VIS_VS20190928':{'mod':'uf','low':0.,'high':40.},
            'FWHM_ord0__ISCARMENES_VIS_VS20191025':{'mod':'uf','low':0.,'high':40.}
            })          
        
        glob_fit_dic['IntrProf']['priors'].update({
            'a_damp__ISCARMENES_VIS_VS20190928':{'mod':'uf','low':0.,'high':10.},
            # 'a_damp__ISCARMENES_VIS_VS20191025':{'mod':'uf','low':0.,'high':10.}
            }) 

    elif gen_dic['star_name']=='WASP47': 
        glob_fit_dic['IntrProf']['priors'].update({
            # 'veq':{'mod':'uf','low':0.,'high':6.},  
            'veq':{'mod':'dgauss','val':1.8,'s_val_low':0.16,'s_val_high':0.24},
            'lambda_rad__plWASP47d':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},
            'ctrst_ord0__ISHARPN_VS20210730':{'mod':'uf','low':0.,'high':1.},  
            'FWHM_ord0__ISHARPN_VS20210730':{'mod':'uf','low':0.,'high':30.},  
            
            })

    elif gen_dic['star_name']=='55Cnc': 
        glob_fit_dic['IntrProf']['priors'].update({
            #  'lambda_rad__pl55Cnc_e__ISESPRESSO_VS20200205':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
            # 'lambda_rad__pl55Cnc_e__ISESPRESSO_VS20210121':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
            # 'lambda_rad__pl55Cnc_e__ISESPRESSO_VS20210124':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 

            # 'lambda_rad__pl55Cnc_e__ISHARPS_VS20120127':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
            # 'lambda_rad__pl55Cnc_e__ISHARPS_VS20120213':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
            # 'lambda_rad__pl55Cnc_e__ISHARPS_VS20120227':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
            # 'lambda_rad__pl55Cnc_e__ISHARPS_VS20120315':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
            
            # 'lambda_rad__pl55Cnc_e__ISHARPN_VS20131114':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
            # 'lambda_rad__pl55Cnc_e__ISHARPN_VS20131128':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
            # 'lambda_rad__pl55Cnc_e__ISHARPN_VS20140101':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
            # 'lambda_rad__pl55Cnc_e__ISHARPN_VS20140126':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
            # 'lambda_rad__pl55Cnc_e__ISHARPN_VS20140226':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
            # 'lambda_rad__pl55Cnc_e__ISHARPN_VS20140329':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},  

            'lambda_rad__pl55Cnc_e__ISEXPRES_VS20220131':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
            'lambda_rad__pl55Cnc_e__ISEXPRES_VS20220406':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
                
            # 'ctrst_ord0__ISESPRESSO_VS20200205':{'mod':'uf','low':0.,'high':1.},  
            # 'FWHM_ord0__ISESPRESSO_VS20200205':{'mod':'uf','low':0.,'high':30.},              
            # 'ctrst_ord0__ISESPRESSO_VS20210121':{'mod':'uf','low':0.,'high':1.},  
            # 'FWHM_ord0__ISESPRESSO_VS20210121':{'mod':'uf','low':0.,'high':30.},              
            # 'ctrst_ord0__ISESPRESSO_VS20210124':{'mod':'uf','low':0.,'high':1.},  
            # 'FWHM_ord0__ISESPRESSO_VS20210124':{'mod':'uf','low':0.,'high':30.}, 
              # 'ctrst_ord0__ISHARPS_VS20120127':{'mod':'uf','low':0.,'high':1.},  
              # 'FWHM_ord0__ISHARPS_VS20120127':{'mod':'uf','low':0.,'high':15.}, 
              # 'ctrst_ord0__ISHARPS_VS20120213':{'mod':'uf','low':0.,'high':1.},  
              # 'FWHM_ord0__ISHARPS_VS20120213':{'mod':'uf','low':0.,'high':15.},                                  
              # 'ctrst_ord0__ISHARPS_VS20120227':{'mod':'uf','low':0.,'high':1.},  
              # 'FWHM_ord0__ISHARPS_VS20120227':{'mod':'uf','low':0.,'high':15.}, 
              # 'ctrst_ord0__ISHARPS_VS20120315':{'mod':'uf','low':0.,'high':1.},  
              # 'FWHM_ord0__ISHARPS_VS20120315':{'mod':'uf','low':0.,'high':15.}, 
              #   'ctrst_ord0__ISHARPN_VS20131114':{'mod':'uf','low':0.,'high':1.},  
              #   'FWHM_ord0__ISHARPN_VS20131114':{'mod':'uf','low':0.,'high':15.}, 
              #   'ctrst_ord0__ISHARPN_VS20131128':{'mod':'uf','low':0.,'high':1.},  
              #   'FWHM_ord0__ISHARPN_VS20131128':{'mod':'uf','low':0.,'high':15.}, 
                 # 'ctrst_ord0__ISHARPN_VS20140101':{'mod':'uf','low':0.,'high':1.},  
                 # 'FWHM_ord0__ISHARPN_VS20140101':{'mod':'uf','low':0.,'high':15.}, 
              #   'ctrst_ord0__ISHARPN_VS20140126':{'mod':'uf','low':0.,'high':1.},  
              #   'FWHM_ord0__ISHARPN_VS20140126':{'mod':'uf','low':0.,'high':15.},                                     
              #   'ctrst_ord0__ISHARPN_VS20140226':{'mod':'uf','low':0.,'high':1.},    
              #   'FWHM_ord0__ISHARPN_VS20140226':{'mod':'uf','low':0.,'high':15.}, 
              #   'ctrst_ord0__ISHARPN_VS20140329':{'mod':'uf','low':0.,'high':1.},      
              #   'FWHM_ord0__ISHARPN_VS20140329':{'mod':'uf','low':0.,'high':15.},        

            # 'ctrst_ord0__ISEXPRES_VS20220131':{'mod':'uf','low':0.,'high':1.},  
            # 'FWHM_ord0__ISEXPRES_VS20220131':{'mod':'uf','low':0.,'high':30.},              
            # 'ctrst_ord0__ISEXPRES_VS20220406':{'mod':'uf','low':0.,'high':1.},  
            # 'FWHM_ord0__ISEXPRES_VS20220406':{'mod':'uf','low':0.,'high':30.},  

            'ctrst_ord0__IS__VS_':{'mod':'uf','low':0.,'high':1.},  
            'FWHM_ord0__IS__VS_':{'mod':'uf','low':0.,'high':15.}, 

            'veq':{'mod':'uf','low':0.,'high':10.},  
            'lambda_rad__pl55Cnc_e':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
        
            # 'veq':{'mod':'uf','low':2.6099460026e+00-2.*3.0271648816e-01,'high':2.6099460026e+00+2.*3.0271648816e-01},      #from HARPS > All_vis > Corr_trend > Comm_line            
            # 'ctrst_ord0__IS__VS_':{'mod':'uf','low':6.7739034862e-01-2.*3.4580162765e-02,'high':6.7739034862e-01+2.*3.4580162765e-02},    #from HARPS > All_vis > Corr_trend > Comm_line
            # 'FWHM_ord0__IS__VS_':{'mod':'uf','low':4.6510963685e+00-2.*3.3033396495e-01,'high':4.6510963685e+00+2.*3.3033396495e-01},   #from HARPS > All_vis > Corr_trend > Comm_line

            # 'veq':{'mod':'uf','low':1.21095752e+00,'high':1.38584791e+00},      #from HARPScorrtrend_ESPRESSOcorrPC_HARPNcorrPCnoV4
            # 'ctrst_ord0__IS__VS_':{'mod':'uf','low':6.97764243e-01 ,'high': 7.17962302e-01},    #from HARPScorrtrend_ESPRESSOcorrPC_HARPNcorrPCnoV4
            # 'FWHM_ord0__IS__VS_':{'mod':'uf','low':4.51249435e+00 ,'high': 4.68518061e+00},   #from HARPScorrtrend_ESPRESSOcorrPC_HARPNcorrPCnoV4
            # 'veq':{'mod':'dgauss','val':1.295232959,'s_val_low':0.084275439,'s_val_high':0.090614950600000},    #from HARPScorrtrend_ESPRESSOcorrPC_HARPNcorrPCnoV4
            # 'ctrst_ord0__IS__VS_':{'mod':'dgauss','val':7.0640591115e-01,'s_val_low':0.0086416681500000,'s_val_high':0.01155639085},   #from HARPScorrtrend_ESPRESSOcorrPC_HARPNcorrPCnoV4
            # 'FWHM_ord0__IS__VS_':{'mod':'dgauss','val':4.5943537966,'s_val_low':-0.0818594,'s_val_high':0.0908268},   #from HARPScorrtrend_ESPRESSOcorrPC_HARPNcorrPCnoV4
            })

    if gen_dic['star_name']=='HD209458':  
        glob_fit_dic['IntrProf']['priors'].update({
                    'cont':{'mod':'uf','low':0.,'high':100.},  
                    'veq':{'mod':'uf','low':0.,'high':10.},  
                    'lambda_rad__plHD209458b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
                    'ctrst_ord0__ISESPRESSO_VS20190720':{'mod':'uf','low':0.,'high':1.},  
                    'ctrst_ord0__ISESPRESSO_VS20190911':{'mod':'uf','low':0.,'high':1.},  
                    'ctrst_ord1__ISESPRESSO_VS_':{'mod':'uf','low':-1.,'high':1.},  
                    'FWHM_ord0__ISESPRESSO_VS20190720':{'mod':'uf','low':0.,'high':15.}, 
                    'FWHM_ord0__ISESPRESSO_VS20190911':{'mod':'uf','low':0.,'high':15.},                       
                    'FWHM_ord1__ISESPRESSO_VS_':{'mod':'uf','low':-1.,'high':1.}, 
                    'rv':{'mod':'uf','low':-5.,'high':5.},                     
                    'abund_Na':{'mod':'uf','low':4.,'high':10.},                     
                    })  
   
        
    if gen_dic['star_name']=='WASP76':  
        glob_fit_dic['IntrProf']['priors'].update({
                    'cont':{'mod':'uf','low':0.,'high':100.}, 
                    'veq':{'mod':'uf','low':0.,'high':10.},  
                    'lambda_rad__plWASP76b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi}, 
                    'ctrst_ord0__ISESPRESSO_VS20180902':{'mod':'uf','low':0.,'high':1.},  
                    'ctrst_ord0__ISESPRESSO_VS20181030':{'mod':'uf','low':0.,'high':1.},  
                    'ctrst_ord1__ISESPRESSO_VS_':{'mod':'uf','low':-10.,'high':10.},  
                    'ctrst_ord2__ISESPRESSO_VS_':{'mod':'uf','low':-10.,'high':10.},
                    'FWHM_ord0__ISESPRESSO_VS20180902':{'mod':'uf','low':0.,'high':15.}, 
                    'FWHM_ord0__ISESPRESSO_VS20181030':{'mod':'uf','low':0.,'high':15.},                       
                    'FWHM_ord1__ISESPRESSO_VS_':{'mod':'uf','low':-10.,'high':10.},                      
                    'FWHM_ord2__ISESPRESSO_VS_':{'mod':'uf','low':-10.,'high':10.}, 
                    }) 
        glob_fit_dic['IntrProf']['priors'].update({
            'inclin_rad__plWASP76b':{'mod':'dgauss','val':89.623*np.pi/180.,'low':0.034*np.pi/180.,'high':0.005*np.pi/180.},
            'aRs__plWASP76b':{'mod':'dgauss','val':4.08,'low':0.06,'high':0.02}}) 

 

    # Stage Théo

    if gen_dic['star_name'] == 'V1298tau' : 
        glob_fit_dic['IntrProf']['priors'].update({
            'veq':{'mod':'uf','low':0.,'high':30.},    
            'lambda_rad__plV1298tau_b':{'mod':'uf','low':-2.*np.pi,'high':2.*np.pi},       
            'FWHM_ord0__ISHARPN_VSmock_vis':{'mod':'uf','low':0.,'high':10.},
            'ctrst_ord0__ISHARPN_VSmock_vis':{'mod':'uf','low':0.,'high':1.},
            })


    #Derived properties
    glob_fit_dic['IntrProf']['modif_list'] = ['veq_from_Peq_Rstar','vsini','psi','om','b','ip','istar_deg_conv','fold_istar','lambda_deg','c0','CB_ms']
    glob_fit_dic['IntrProf']['modif_list'] = ['vsini','lambda_deg']
    glob_fit_dic['IntrProf']['modif_list'] = ['vsini','lambda_deg','ip']
    # glob_fit_dic['IntrProf']['modif_list'] = ['lambda_deg','istar_deg_conv','Peq_veq']
    #glob_fit_dic['IntrProf']['modif_list'] = ['veq_from_Peq_Rstar','vsini','lambda_deg','istar_deg_conv','fold_istar','psi']
    # glob_fit_dic['IntrProf']['modif_list'] = []
    # glob_fit_dic['IntrProf']['modif_list'] = ['vsini','lambda_deg','psi'] 
    # glob_fit_dic['IntrProf']['modif_list'] = ['vsini','lambda_deg','Peq_vsini'] 
    # glob_fit_dic['IntrProf']['modif_list'] = ['istar_Peq_vsini'] 
    # glob_fit_dic['IntrProf']['modif_list'] = ['istar_Peq_vsini','psi_lambda'] 
    # glob_fit_dic['IntrProf']['modif_list'] = ['vsini','lambda_deg','istar_Peq','psi'] 
    # glob_fit_dic['IntrProf']['modif_list'] = ['vsini','lambda_deg','ip'] 
    # glob_fit_dic['IntrProf']['modif_list'] = ['vsini','lambda_deg','CF0_meas_conv'] 
    # glob_fit_dic['IntrProf']['modif_list'] = ['vsini','lambda_deg','CF0_DG_conv'] 
    # glob_fit_dic['IntrProf']['modif_list'] = ['vsini','lambda_deg','CF0_meas_add','b','ip'] 


    
    #Calculating/retrieving
    glob_fit_dic['IntrProf']['mcmc_run_mode']='use'


    #Runs to re-use  
    # if gen_dic['star_name']=='55Cnc': 
    #     glob_fit_dic['IntrProf']['mcmc_reuse']={
    #         'paths':['/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/mcmc/All_visits/run2/raw_chains_walk400_steps1000.npz',\
    #                  '/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/mcmc/All_visits/run3/raw_chains_walk400_steps1000.npz'],
    #         'nburn':[0,0]}    
    # if gen_dic['star_name']=='WASP76': 
    #     glob_fit_dic['IntrProf']['mcmc_reuse']={
    #         # 'paths':['/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Joined_fits/IntrProf/mcmc_a/raw_chains_walk150_steps1600.npz',\
    #         #           '/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Joined_fits/IntrProf/mcmc_b/raw_chains_walk150_steps350.npz'],
    #         # 'nburn':[1200,0]}   
    #         # 'paths':['/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Joined_fits/IntrProf/CCF_from_DISpec/mcmc_contcorr/mcmc_e/raw_chains_walk150_steps240.npz',\
    #         #           '/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Joined_fits/IntrProf/CCF_from_DISpec/mcmc_contcorr/mcmc_f/raw_chains_walk150_steps315.npz'],
    #         # 'nburn':[0,0]}  
    #         'paths':['/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Joined_fits/IntrProf/CCF_from_IntrSpec/CONTCORR_CONTFIT/mcmc/mcmc_f/raw_chains_walk150_steps459.npz',\
    #                  '/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Joined_fits/IntrProf/CCF_from_IntrSpec/CONTCORR_CONTFIT/mcmc/mcmc_g/raw_chains_walk150_steps229.npz'],
    #         'nburn':[0,0]}              
            

    #Runs to re-start
    # if gen_dic['star_name']=='55Cnc':     
    #     # glob_fit_dic['IntrProf']['mcmc_reboot']='/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/mcmc/20210121_temp2/raw_chains_walk250_steps5000.npz'
    #     # glob_fit_dic['IntrProf']['mcmc_reboot']='/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/mcmc/20210124/run3/raw_chains_walk250_steps1000.npz'
    #     glob_fit_dic['IntrProf']['mcmc_reboot']='/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/mcmc/All_visits/run3/raw_chains_walk400_steps1000.npz'
    # if gen_dic['star_name']=='WASP76':     
        # glob_fit_dic['IntrProf']['mcmc_reboot']='/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Joined_fits/IntrProf/mcmc_V3a/raw_chains_walk150_steps300.npz'
        # glob_fit_dic['IntrProf']['mcmc_reboot']='/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Joined_fits/IntrProf/CCF_from_DISpec/mcmc_contcorr/mcmc_e/raw_chains_walk150_steps240.npz'





    #Walkers
    if gen_dic['star_name']=='Corot7':     #38 s for 1000 pts 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':20,'nsteps':5000,'nburn':1000}
    elif list(gen_dic['transit_pl'].keys())==['GJ9827d']:    #1 min for 1000 pts
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':20,'nsteps':5000,'nburn':1000} 
    elif list(gen_dic['transit_pl'].keys())==['GJ9827b']:      #20 min pour 1000 points
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':14,'nsteps':2000,'nburn':500}
    elif gen_dic['star_name']=='HD3167': 
        if list(gen_dic['transit_pl'].keys())==['HD3167_b']:     #Sans oversampling: 0.14 min for 1000 pts
            glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':20,'nsteps':10,'nburn':0}            
            glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':40,'nsteps':7000,'nburn':1000}
        if list(gen_dic['transit_pl'].keys())==['HD3167_c']:    
            glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':20,'nsteps':4000,'nburn':500}    #ancienne version : 4 min pour 20*4000 sans oversamp   
            glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':20,'nsteps':1,'nburn':0}  #ancienne version : 4 min pour 20*4000 sans oversamp   
        elif ('HD3167_b' in gen_dic['transit_pl']) and ('HD3167_c' in gen_dic['transit_pl']):     
            glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':40,'nsteps':6000,'nburn':500}     #sans oversamp, ndpl=31 : 0.41 min / 2000 pts   
    elif gen_dic['star_name']=='TOI858': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':40,'nsteps':2000,'nburn':500}    #sans oversamp, ndpl=51 : 0.24 min / 2000 pts   
    elif gen_dic['star_name']=='GJ436': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':60,'nsteps':10,'nburn':0}    #tests        
        # glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':60,'nsteps':6000,'nburn':1000}    #sans oversamp, ndpl=51 : 0.54 min / 2000 pts     
        # glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':70,'nsteps':6000,'nburn':1000}    #sans oversamp, ndpl=51 : 0.54 min / 2000 pts           
        # glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':60,'nsteps':3000,'nburn':800}    #oversamp 5, ndpl=51 : 1.08 min / 2000 pts     
        # glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':80,'nsteps':6000,'nburn':1000}    #oversamp 5, ndpl=51 : 1.08 min / 2000 pts      #joint fit all
        # glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':80,'nsteps':4000,'nburn':1000}    #oversamp 5, ndpl=51 : 1.08 min / 2000 pts      #joint fit all corr  
        # glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':70,'nsteps':4000,'nburn':1000}    #oversamp 5, ndpl=51 : 1.08 min / 2000 pts      #joint fit all corr, ESPRESSO common       
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':60,'nsteps':4000,'nburn':1000}    #oversamp 5, ndpl=51 : 2.1 min / 2000 pts      #joint fit HARPS/HARPS-N       
    elif gen_dic['star_name']=='HIP41378': 
        # glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':10,'nsteps':10,'nburn':0}             
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':40,'nsteps':2000,'nburn':500}  
    #RM survey
    elif gen_dic['star_name']=='HAT_P3': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':70,'nsteps':2000,'nburn':500}         #oversamp 5, ndpl=51 :  0.113051935 min / 200 pts
        # glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':80,'nsteps':2000,'nburn':500}         #oversamp 5, ndpl=51 :  0.113051935 min / 200 pts
    elif gen_dic['star_name']=='Kepler25': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':60,'nsteps':2000,'nburn':500}         #oversamp 5, ndpl=51 :  0.20482250054 min / 200 pts
    elif gen_dic['star_name']=='Kepler68': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':20,'nsteps':10,'nburn':0.}         #oversamp 2, ndpl=31 :  0.236920400 min / 200 pts
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':60,'nsteps':2000,'nburn':500.}        
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':60,'nsteps':1250,'nburn':250.} 
    elif gen_dic['star_name']=='HAT_P33': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':60,'nsteps':2000,'nburn':500.}         #oversamp 5, ndpl=51 : 0.232673  min / 200 pts
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':70,'nsteps':2000,'nburn':500.}         #oversamp 5, ndpl=51 
    elif gen_dic['star_name']=='K2_105': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':60,'nsteps':2000,'nburn':500}         #oversamp 5, ndpl=51 : 1 min / 600 pts
    elif gen_dic['star_name']=='HD89345': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':60,'nsteps':2000,'nburn':500}         #oversamp 5, ndpl=51 : 1.27 min / 600 pts            
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':65,'nsteps':1800,'nburn':400}         #oversamp 5, ndpl=51 
    elif gen_dic['star_name']=='Kepler63': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':60,'nsteps':2000,'nburn':500}         #oversamp 2, ndpl=31 : 0.05414 min / 200 pts  
    elif gen_dic['star_name']=='HAT_P49': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':60,'nsteps':2000,'nburn':500}         #oversamp 5, ndpl=51 : 0.3046 min / 200 pts  
    elif gen_dic['star_name']=='WASP107': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':100,'nsteps':2000,'nburn':500}         #oversamp 5, ndpl=51 : 0.592582 min / 200 pts  
        # glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':50,'nsteps':2000*3,'nburn':500*3}         ##test voigt
    elif gen_dic['star_name']=='WASP166': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':60,'nsteps':2000,'nburn':500}         #oversamp 2, ndpl=31 
        # glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':80,'nsteps':2000,'nburn':500}         #oversamp 2, ndpl=31 
        # glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':20,'nsteps':10,'nburn':0}         #oversamp 2, ndpl=31 
    elif gen_dic['star_name']=='HAT_P11': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':80,'nsteps':4000,'nburn':500}         #oversamp 3, ndpl=31 
        # glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':80,'nsteps':4000,'nburn':2500}         #oversamp 3, ndpl=31 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':70,'nsteps':2000,'nburn':500}         #oversamp 3, ndpl=31 
    elif gen_dic['star_name']=='HD106315': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':80,'nsteps':2000,'nburn':500}         #oversamp 2, ndpl=31 
    elif gen_dic['star_name']=='WASP156': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':80,'nsteps':2000,'nburn':500}          
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':40,'nsteps':2000,'nburn':500}   
    elif gen_dic['star_name']=='WASP47': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':60,'nsteps':2000,'nburn':500}         
    elif gen_dic['star_name']=='55Cnc':  
        # glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':20,'nsteps':100,'nburn':0}     #Test   
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':50,'nsteps':1000,'nburn':300}     #Fit RM alone, PC or trend-corrected

    elif gen_dic['star_name']=='HD209458': 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':80,'nsteps':400,'nburn':275}     #2000 steps = 25 min  (15s/it)
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':80,'nsteps':800,'nburn':400}     #2000 steps = 25 min  (15s/it)
        # glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':30,'nsteps':10,'nburn':0}         #
        
        
    elif gen_dic['star_name']=='WASP76':     
        # glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':30,'nsteps':10,'nburn':0}         #20 x 10 steps = 2.6 min en medium precision
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':150,'nsteps':1600,'nburn':800}                   
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':150,'nsteps':350,'nburn':0}    #restart 

        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':150,'nsteps':int(5.*3600/57.),'nburn':0}    #old laptop 
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':150,'nsteps':int(5.*3600/45.),'nburn':0}    #new laptop 
        
    # Stage Théo  
    
    if gen_dic['star_name'] == 'V1298tau':
        glob_fit_dic['IntrProf']['mcmc_set']={'nwalkers':20,'nsteps':2000,'nburn':500}          



    #Complex priors        
    if gen_dic['transit_pl']==['HD3167_b']:  
        glob_fit_dic['IntrProf']['prior_func']={'contrast':{}}
    elif gen_dic['transit_pl']==['HD3167_c']:
        glob_fit_dic['IntrProf']['prior_func']={'contrast':{}}

    elif gen_dic['star_name']=='HAT_P33': 
        glob_fit_dic['IntrProf']['prior_func']={'sinistar_geom':{}}        
        
    elif gen_dic['star_name']=='HIP41378': 
        glob_fit_dic['IntrProf']['prior_func']={'FWHM_vsini':{'FWHM_DI':9.8}}          

    # elif gen_dic['star_name']=='HAT_P3': 
    #     glob_fit_dic['IntrProf']['prior_func']={'FWHM_vsini':{'FWHM_DI':6.329}}            
        
        
    #Walkers exclusion  
    glob_fit_dic['IntrProf']['exclu_walk']=True     & False       
    

    #Automatic exclusion of outlying chains
    glob_fit_dic['IntrProf']['exclu_walk_autom']=None  #  5.



    #Derived errors 
    # glob_fit_dic['IntrProf']['HDI']='3s'   #None   #'3s' 
        # fit_dic['HDI_nbins']={'veq':50,'lambda_rad__'+pl_loc:50}   #pour l'intervalle a 1s
    if gen_dic['star_name']=='GJ436': glob_fit_dic['IntrProf']['HDI_nbins']= {'lambda_deg__GJ436_b':40}  
    
    
    #Derived lower/upper limits
    # glob_fit_dic['IntrProf']['conf_limits']={'ecc_pl1':{'bound':0.,'type':'upper','level':['1s','3s']}} 


    
    #MCMC chains
    glob_fit_dic['IntrProf']['save_MCMC_chains']='png'   #png  



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
        # 'plot_best':False,
            
            
#            'major_int':[0.2,50.],
#            'minor_int':[0.1,10.],
        'color_levels':['deepskyblue','lime'],
        # 'fontsize':15,
        'fontsize':10,
#            'smooth2D':[0.05,5.] 
#            'plot1s_1D':False
        }    
    if gen_dic['star_name']=='WASP76': 
        glob_fit_dic['IntrProf']['corner_options']['range_par'] = {'v$_\mathrm{eq}$sin i$_{*}$ (km/s)':  [0.5,2.5], '$\lambda$[WASP76b]($^{\circ}$)':[-90,90] }
        glob_fit_dic['IntrProf']['corner_options']['bins_1D_par'] = 40. #{'v$_\mathrm{eq}$sin i$_{*}$ (km/s)': 50, '$\lambda$[WASP76b]($^{\circ}$)':50 }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    


























    
##################################################################################################       
#%%% Module: estimates for planet-occulted profiles 
#    - use the module to generate:
# + local profiles that are then used to correct residual profiles from stellar contamination
# + intrinsic profiles that are corrected from measured ones to assess the quality of the estimates 
#    - the choice to use measured ('meas') or theoretical ('theo') stellar surface RVs to shift local profiles is set by data_dic['Intr']['align_mode']
##################################################################################################     

#%%%% Activating
#    - for original and binned exposures in each visit
gen_dic['loc_data_corr'] = False
gen_dic['loc_data_corr_bin'] =  False


#%%%% Calculating/retrieving
gen_dic['calc_loc_data_corr']=True  
gen_dic['calc_loc_data_corr_bin']=True  


#%%%% Profile type
#    - reconstructing local ('Res') or intrinsic ('Intr') profiles
#    - local profiles cannot be reconstructed for spectral data converted into CCFs, as in-transit residual CCFs are not calculated
data_dic['Intr']['plocc_prof_type']='Intr'   


#%%%% Model definition

#%%%%% Model type
#    - used to define the estimates for the local stellar flux profiles
#    - these options partly differ from those defining intrinsic profiles (see gen_dic['mock_data']) because local profiles are associated with observed exposures
#    - 'DIbin': using the master-out
# + option to select visits contributing to the binned profiles (leave empty to use considered visit)
# + option to select exposures contributing to the binned profiles (leave empty to use all out-transit exposures)
# + option to select the phase range of contributing exposures
#    - 'Intrbin': using binned intrinsic profiles series
# + option to select visits contributing to the binned profiles (leave empty to use considered visit)
# + the nearest binned profile along the binned dimension is used for a given exposure
# + option to select exposures contributing to the binned profiles
# + see possible bin dimensions in data_dic['Intr']['dim_bin']  
# + see possible bin table definition in data_dic['Intr']['prop_bin']
#    - 'glob_mod': analytical models derived from global fit to intrinsic profiles
# + can be specific to the visit or common to all, depending on the fit
# + line coordinate choice is retrieved automatically 
# + indicate path to saved properties determining the line property variations in the processed dataset
# + default options are used if left undefined
#    - 'indiv_mod': using models fitted to each individual intrinsic profile in each visit
# + works only in exposures where the stellar line could be fitted after planet exclusion
#    - 'rec_prof':
# + define each undefined pixel via a polynomial fit to defined pixels in complementary exposures
#   or via a 2D interpolation ('linear' or 'cubic') over complementary exposures and a narrow spectral band (defined in band_pix_hw pixels on each side of undefined pixels)
# + chose a dimension over which the fit/interpolation is performed         
# + option to select exposures contributing to the fit/interpolation
#    - 'theo': use imported theoretical local intrinsic stellar profiles    
data_dic['Intr']['mode_loc_data_corr']='glob_mod'   


#%%%%% Options
#    - 'def_iord': reconstructed order
#    - 'def_range': define the range over which profiles are reconstructed
data_dic['Intr']['opt_loc_data_corr']={'nthreads':14,'def_range':[],'def_iord':0}


#%%%% Plot settings

#%%%%% 2D maps : theoretical intrinsic stellar profiles
#    - for original and binned exposures
#    - data to which the reconstruction was applied to is automatically used for this plot
plot_dic['map_Intr_prof_est']=''   


#%%%%% 2D maps : residuals from theoretical intrinsic stellar profiles
#    - same format as 'map_Intr_prof_est'
plot_dic['map_Intr_prof_res']=''   




if __name__ == '__main__':
    
    #Estimating stellar profiles 
    gen_dic['loc_data_corr']=True   &  False
    gen_dic['loc_data_corr_bin']=True    &  False
    
    #Calculating/retrieving
    gen_dic['calc_loc_data_corr']=True   &  False   
    gen_dic['calc_loc_data_corr_bin']=True  #  &  False  
    
    #Profile type
    data_dic['Intr']['plocc_prof_type']='Intr'         
    
    #Method and options to define the local stellar flux profiles   
    data_dic['Intr']['mode_loc_data_corr']='glob_mod'    
    
    #Options specific to each method
    data_dic['Intr']['opt_loc_data_corr']['DIbin']={
            'vis_in_bin':{},
            # 'vis_in_bin':{'ESPRESSO':['2018-09-03','2018-10-31']},            
            'idx_in_bin':deepcopy(data_dic['DI']['idx_in_bin']),  
            # 'ESPRESSO':{'2018-09-03':{'bin_low':[-0.05],'bin_high':[0.05]},
            #             '2018-10-31':{'bin_low':[-0.05],'bin_high':[0.05]}}             
        }
        
        
    data_dic['Intr']['opt_loc_data_corr']['Intrbin']={        
            'vis_in_bin':{},
            # 'vis_in_bin':{'ESPRESSO':['2018-09-03','2018-10-31']},             
            'idx_in_bin':{'ESPRESSO':{'2018-09-03':range(1,20),'2018-10-31':range(1,38)}},
    
            # 'dim_bin': 'phase',
            # 'ESPRESSO':{'2018-09-03':{'bin_low':np.arange(-0.05,0.05,0.0125),'bin_high':np.arange(-0.05,0.05,0.0125)+0.0125},
            #             '2018-10-31':{'bin_low':np.arange(-0.05,0.05,0.0125),'bin_high':np.arange(-0.05,0.05,0.0125)+0.0125}}       
            # 'ESPRESSO':{'2018-09-03':{'bin_low':[-0.05,0.],'bin_high':[0.,0.05]},
            #             '2018-10-31':{'bin_low':[-0.05,0.],'bin_high':[0.,0.05]}}    
            # 'ESPRESSO':{'2018-09-03':{'bin_low':[-0.05],'bin_high':[0.05]},
            #             '2018-10-31':{'bin_low':[-0.05],'bin_high':[0.05]}} 
            'dim_bin': 'xp_abs',
            # 'ESPRESSO':{'2018-09-03':{'bin_low':[0.],'bin_high':[1.]},'2018-10-31':{'bin_low':[0.],'bin_high':[1.]}} 
            'ESPRESSO':{'2018-09-03':{'bin_low':[0.,0.5],'bin_high':[0.5,1.]},'2018-10-31':{'bin_low':[0.,0.5],'bin_high':[0.5,1.]}} 

        }


    data_dic['Intr']['opt_loc_data_corr']['rec_prof']={
            'vis_in_rec':{},
            # 'vis_in_rec':{'ESPRESSO':['2018-09-03','2018-10-31']},  
            'idx_in_rec':{'ESPRESSO':{'2018-09-03':range(1,20),'2018-10-31':range(1,38)}},
            'dim_bin': 'xp_abs',
            
            # 'rec_mode':'pix_polfit',
            # 'pol_deg':2,
            
            'rec_mode':'band_interp',
            'band_pix_hw':1,
            'interp_mode':'linear'
            
            }

    # if gen_dic['transit_pl']=='WASP76b':    
    #     glob_fit_dic['IntrProf']['IntrProf_prop_path']='/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Intr_data_prop/Fit/IntrProf_prop.npz'
    # if gen_dic['star_name']=='HD3167': 
    #     glob_fit_dic['IntrProf']['IntrProf_prop_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/HD3167_bHD3167_c_Saved_data/Intr_data_prop/Fit_osamp5_n51/Fit_C2_ystar2/mcmc/IntrProf_prop.npz'
    #     glob_fit_dic['IntrProf']['IntrProf_prop_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/HD3167_bHD3167_c_Saved_data/Intr_data_prop/No_osamp_n31/Fit_C2_ystar2_RpRs-1s/Cleaned/IntrProf_prop.npz'
        
    #     if list(gen_dic['transit_pl'].keys())==['HD3167_b']:    
    #         glob_fit_dic['IntrProf']['IntrProf_prop_path']='/Travaux/ANTARESS/En_cours/HD3167_b_Saved_data/Intr_data_prop/Fit/IntrProf_prop.npz'
    #         # glob_fit_dic['IntrProf']['IntrProf_prop_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/HD3167_b_Saved_data/Intr_data_prop/HRRM_loose_priors_Cdeg1_FWHMdeg0_vsini2.6/mcmc/IntrProf_prop.npz'      #BEST-FIT   
    #         glob_fit_dic['IntrProf']['IntrProf_prop_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/HD3167_b_Saved_data/Intr_data_prop/HRRM_loose_priors_Cdeg1_FWHMdeg0_vsini2.4_osamp5_n51/mcmc/IntrProf_prop.npz'      #BEST-FIT   revised  
    #         # glob_fit_dic['IntrProf']['IntrProf_prop_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/HD3167_b_Saved_data/Intr_data_prop/Fit/mcmc/IntrProf_prop.npz'
    #     if list(gen_dic['transit_pl'].keys())==['HD3167_c']:
    #         # glob_fit_dic['IntrProf']['IntrProf_prop_path']='/Travaux/ANTARESS/HD3167/Analyse_HD3167c/All_orders/Christiansen_nominal/HD3167_c_Saved_data/Intr_data_prop/Fit/IntrProf_prop.npz'
    #         # glob_fit_dic['IntrProf']['IntrProf_prop_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/HD3167_c_Saved_data/Intr_data_prop/Fit_loose_priors/mcmc/IntrProf_prop.npz'
    #         # glob_fit_dic['IntrProf']['IntrProf_prop_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/HD3167_c_Saved_data/Intr_data_prop/HRRM_loose_priors_Cdeg1_FWHMdeg0/mcmc/IntrProf_prop.npz'   #BEST-FIT
    #         glob_fit_dic['IntrProf']['IntrProf_prop_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/HD3167_c_Saved_data/Intr_data_prop/HRRM_loose_priors_Cdeg1_FWHMdeg0_dt0.1_nsub51_osamp5/mcmc/IntrProf_prop.npz'   #BEST-FIT revised
    #         # glob_fit_dic['IntrProf']['IntrProf_prop_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/HD3167_c_Saved_data/Intr_data_prop/Fit/mcmc/IntrProf_prop.npz'
   

    if gen_dic['star_name']=='HD3167':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'y_st2',
            # 'IntrProf_prop_path':'/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Intr_data_prop/Fit/IntrProf_prop.npz'},
            'IntrProf_prop_path':glob_fit_dic['IntrProf']['IntrProf_prop_path']
            }
    if gen_dic['star_name']=='TOI858':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'mu',
            'IntrProf_prop_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/TOI858b_Saved_data/Intr_data_prop/Fit/mcmc/Visits12_indiv_osamp5/IntrProf_prop.npz'}

    if gen_dic['star_name']=='GJ436':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'mu',
            'IntrProf_prop_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/GJ436_b_Saved_data/Joined_fits/Intr_prof/mcmc/ESPRESSO/DG_CLfromDI_proppervis_osamp5/IntrProf_prop.npz'}
    elif gen_dic['star_name']=='HIP41378':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'mu',
            'IntrProf_prop_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/HIP41378d_Saved_data/Joined_fits/Intr_prof/mcmc/G2_mask_prior_scaled/Fit_results.npz'}       




    #RM survey
    elif gen_dic['star_name']=='HAT_P3':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'mu',
            'IntrProf_prop_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/HAT_P3b_Saved_data/Joined_fits/Intr_prof/mcmc/Oversamp5_n51_C0_F0_scaled_FINAL/Fit_results.npz'}
    elif gen_dic['star_name']=='HAT_P11':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'mu',
            'IntrProf_prop_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/HAT_P11b_Saved_data/Joined_fits/Intr_prof/mcmc/Voigt_C0_scaled_FINAL/Fit_results.npz'}
    elif gen_dic['star_name']=='HAT_P33':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'mu',
            'IntrProf_prop_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/HAT_P33b_Saved_data/Joined_fits/Intr_prof/mcmc/Oversamp5_n51_C0_F0_scaled_FINAL/Fit_results.npz'}
    elif gen_dic['star_name']=='HAT_P49':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'mu',
            'IntrProf_prop_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/HAT_P49b_Saved_data/Joined_fits/Intr_prof/mcmc/Oversamp5_n51_scaled_FINAL/Fit_results.npz'}       
    elif gen_dic['star_name']=='HD89345':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'mu',
            'IntrProf_prop_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/HD89345b_Saved_data/Joined_fits/Intr_prof/mcmc/Oversamp2_n31_scaled_FINAL/Fit_results.npz'}           
    elif gen_dic['star_name']=='HD106315':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'mu',
            'IntrProf_prop_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/HD106315c_Saved_data/Joined_fits/Intr_prof/mcmc/noversamp2_npl31_scaled/C0comm_F0comm_FINAL/Fit_results.npz'}       
    elif gen_dic['star_name']=='K2_105':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'mu',
            # 'IntrProf_prop_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/K2_105b_Saved_data/Joined_fits/Intr_prof/mcmc/Oversamp2_n31/Fit_results.npz'}
            'IntrProf_prop_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/K2_105b_Saved_data/Joined_fits/Intr_prof/mcmc/Oversamp2_n31_vsiniprior_scaled_FINAL/Fit_results.npz'}        
    elif gen_dic['star_name']=='Kepler25':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'mu',
            'IntrProf_prop_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/Kepler25c_Saved_data/Joined_fits/Intr_prof/mcmc/Oversamp5_n51_priorvsini_scaled_FINAL/Fit_results.npz'} 
    elif gen_dic['star_name']=='Kepler63':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'mu',
            'IntrProf_prop_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/Kepler63b_Saved_data/Joined_fits/Intr_prof/mcmc/Oversamp2_n31_priorFWHM_9.5_4_FINAL/Fit_results.npz'}       
    elif gen_dic['star_name']=='WASP107':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'mu',
            'IntrProf_prop_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/WASP107b_Saved_data/Joined_fits/Intr_prof/mcmc/Oversamp3_n31_voigt_scaled_FINAL/Fit_results.npz'}       
    elif gen_dic['star_name']=='WASP166':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'mu',
            'IntrProf_prop_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/WASP166b_Saved_data/Joined_fits/Intr_prof/mcmc/Oversamp2_n31_scaled_FINAL/Fit_results.npz'}       
    elif gen_dic['star_name']=='WASP156':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'mu',
            'IntrProf_prop_path':'/Users/bourrier/Travaux/ANTARESS/En_cours/WASP156b_Saved_data/Joined_fits/Intr_prof/mcmc/noversamp5_npl51_V1_Moutpost_priorvsini/Fit_results.npz'}       
    
    elif gen_dic['star_name']=='55Cnc':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'dim_fit':'mu',
            'IntrProf_prop_path':
                {'ESPRESSO':{'20200205':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/ESPRESSO/All_vis/With_outPC/5PC6PC3PC/Fit_results.npz',
                             '20210121':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/ESPRESSO/All_vis/With_outPC/5PC6PC3PC/Fit_results.npz',
                             '20210124':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/chi2/ESPRESSO/All_vis/With_outPC/5PC6PC3PC/Fit_results.npz'
                             },
                 'EXPRES':{'20220131':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/mcmc/All_visits/EXPRES/Corr_trends/Commline/Fit_results.npz',
                           '20220406':'/Users/bourrier/Travaux/ANTARESS/En_cours/55Cnc_e_Saved_data/Joined_fits/Intr_prof/mcmc/All_visits/EXPRES/Corr_trends/Commline/Fit_results.npz',
                             }},
                }       
    elif gen_dic['star_name']=='HD209458':
        
        ##White CCFs
        # data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
        #     'mode':'ana',
        #     'IntrProf_prop_path':
        #         {'ESPRESSO':{'20190720':'/Users/bourrier/Travaux/ANTARESS/En_cours/HD209458b_Saved_data/Joined_fits/IntrProf/mcmc_rproj_deg1_comm/Fit_results',
        #                      '20190911':'/Users/bourrier/Travaux/ANTARESS/En_cours/HD209458b_Saved_data/Joined_fits/IntrProf/mcmc_rproj_deg1_comm/Fit_results'}}}     

        #Na doublet 1D
        data_dic['Intr']['opt_loc_data_corr']['def_range']=[5883.,5903.]
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'mode':'theo',
            'IntrProf_prop_path':
                {'ESPRESSO':{'20190720':'/Users/bourrier/Travaux/ANTARESS/En_cours/HD209458b_Saved_data/Joined_fits/IntrProf/Na_doublet/chi2/Fit_results',
                             '20190911':'/Users/bourrier/Travaux/ANTARESS/En_cours/HD209458b_Saved_data/Joined_fits/IntrProf/Na_doublet/chi2/Fit_results'}}}   

    elif gen_dic['star_name']=='WASP76':
        data_dic['Intr']['opt_loc_data_corr']['glob_mod']={        
            'mode':'ana',
            'IntrProf_prop_path':
                {'ESPRESSO':{'20180902':'/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Joined_fits/IntrProf/mcmc_V3b_final/Fit_results',
                             '20181030':'/Users/bourrier/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Joined_fits/IntrProf/mcmc_V3b_final/Fit_results'}}}  


    #Plot 2D maps of theoretical intrinsic stellar profiles
    plot_dic['map_Intr_prof_est']=''   #'png
    
    #Plotting 2D maps of residuals from theoretical intrinsic stellar profiles
    plot_dic['map_Intr_prof_res']=''   #'png    
    if gen_dic['star_name'] in ['HD189733','WASP43','L98_59','GJ1214']:plot_dic['map_Intr_prof_res']='png'








    
    
    
    










    
    
    
    
    
    
    
    
    

##################################################################################################
#%% Atmospheric profiles
##################################################################################################  
    
    
    ##############################################################################################################################
    #Atmospheric signals
    ##############################################################################################################################

    '''
    Exclusion of planetary signals
    '''
    
    #Range of the planetary signal, in the planet rest frame, in km/s
    data_dic['Atm']['plrange']=[-20.,20.]
    if gen_dic['star_name']=='WASP121':data_dic['Atm']['plrange']=[-20.,10.]
    elif gen_dic['star_name']=='WASP76':data_dic['Atm']['plrange']=[-25.,10.]
    elif gen_dic['star_name']=='HAT_P49':
        data_dic['Atm']['plrange']=[-25.,25.]
        data_dic['Atm']['plrange']=[-15.,15.]
    elif gen_dic['star_name']=='HAT_P33':data_dic['Atm']['plrange']=[-25.,25.]


    #Exclude range of planetary signal
    #    - define below operations from which planetary signal should be excluded
    #      leave empty for no exclusion to be applied
    #    - operations :
    # + DI_Mast: for the calculation of disk-integrated masters        
    # + DI_prof : for the definition of model continuum and fitted ranges in DI profile fits, and corrections of DI profiles 
    # + Res_prof: for the definition of errors on residual CCFs
    # + PCA_corr: for the PCA correction of residual data
    # + Intr: for the definition of model continuum and fitted ranges in Intr profile fits, and the continuum of Intr profiles
    #    - planetary ranges can be excluded even if calc_pl_atm = False and no atmospheric signal is extracted
    data_dic['Atm']['no_plrange']=[]    
    if gen_dic['star_name']=='WASP121':data_dic['Atm']['no_plrange']=['DI_prof','DI_Mast','Res_prof','Intr']
    elif gen_dic['star_name']=='WASP76':data_dic['Atm']['no_plrange']=['DI_prof','DI_Mast','Res_prof','Intr']
    elif gen_dic['star_name']=='HAT_P49':
        data_dic['Atm']['no_plrange']=['DI_prof','DI_Mast','Intr']
        data_dic['Atm']['no_plrange']=['Intr']
        data_dic['Atm']['no_plrange']=[]
    # elif gen_dic['star_name']=='HAT_P33':data_dic['Atm']['no_plrange']=['DI_prof','DI_Mast','Intr']


    #Indexes of exposures from which planetary signal should be excluded, for each instrument/visit
    #    - indexes are relative to the global table in each visit
    #    - allows excluding signal from out-of-transit exposures in case of planetary emission signal
    #    - if undefined, set automatically to in-transit exposures
    data_dic['Atm']['iexp_no_plrange']={}
    
    if gen_dic['transit_pl']=='WASP121b':
        data_dic['Atm']['iexp_no_plrange']={}
    #    data_dic['Atm']['iexp_no_plrange']={'HARPS':{
    ##            '31-12-17':list(np.arange(10,dtype=int)),    #pre-transit seul  
    ##            '09-01-18':list(np.arange(8,dtype=int)),            
    ##            '14-01-18':list(np.arange(19,dtype=int))            
    ##            
    #            '31-12-17':list(np.arange(26,35,dtype=int)),    #post-transit seul  
    ##            '09-01-18':list(np.arange(29,55,dtype=int)),            
    ##            '14-01-18':list(np.arange(39,50,dtype=int)) 
    #            }} 
        
    # if gen_dic['transit_pl']=='WASP76b':
    #     data_dic['Atm']['iexp_no_plrange']={'ESPRESSO':{'2018-09-03':np.arange(35)}}

    # elif gen_dic['star_name']=='HAT_P49':
    #     data_dic['Atm']['iexp_no_plrange']={'HARPN':{'20200730':np.arange(126)}}

    #Oversampling value for the planet radial orbital velocity, in orbital phase 
    data_dic['Atm']['dph_osamp_RVpl']=0.001









    

    '''
    Retrieval of planetary signals
    '''

    #Activate
    gen_dic['pl_atm'] = True   &  False

    #Calculating/retrieving
    gen_dic['calc_pl_atm'] = True #  &  False

    #Atmospheric signal to be extracted
    #    - 'Emission': emission signal 
    #      'Absorption': in-transit absorption signal
    #    - signals are defined in the star rest frame
    data_dic['Atm']['pl_atm_sign']='Absorption'        

    #-----------------------------

    #Calculate CCFs from atmospheric emission or absorption spectra
    #    - every operation afterwards will be performed on those CCFs
    gen_dic['Atm_CCF'] = True      & False
 
    #Calculating/retrieving
    gen_dic['calc_Atm_CCF'] = True       &  False   

    #Path to the mask for atmospheric lines
    #    - relevant for input spectra only
    #    - the mask will be used in two ways:
    # + to exclude spectral ranges contaminated by the planet, in all steps defined via data_dic['Atm']['no_plrange']
    #   this can be useful for stellar and RM study, to remove planetary contamination
    # + to compute atmospheric CCFs, if requested
    #   beware in that case of the definition of the mask weights
    #    - the mask can be reduced to a single line
    #    - can be defined for the purpose of the plots (set to None to prevent upload)
    data_dic['Atm']['CCF_mask'] = None
    # data_dic['Atm']['CCF_mask'] = '/Travaux/ANTARESS/WASP76b/Data/mask_different_alement_WASP76/mask_WASP76_Fe_2.csv'     #in the air 
    if gen_dic['star_name']=='WASP76':data_dic['Atm']['CCF_mask'] = '/Users/bourrier/Travaux/Radial_velocity/RV_masks/ESPRESSO/New_meanC2unity/ESPRESSO_new_F9.fits'        #in the air, new mask   ; ANTARESS I exlusion
    # data_dic['Atm']['CCF_mask']  = '/Travaux/ANTARESS/Method/Masks/Na_doublet_air.txt'

    #Use mask weights on atmospheric CCF
    data_dic['Atm']['use_maskW'] = True    &  False

    #Define range for the CCF continuum 
    #    - used to calculate errors from dispersion in the continuum
    #    - define [ [rv1,rv2] , [rv3,rv4] , [rv5,rv6] , ... ] 
    #      rv are defined in the star rest frame
    #      the ranges are common to all local CCFs, ie that they must be large enough to cover the full range of orbital RVs 
    data_dic['Atm']['cont_range']=[]
    if gen_dic['transit_pl']=='WASP121b':
        data_dic['Atm']['cont_range']=[[-200.,-50.],[50.,200.]] 
    elif gen_dic['transit_pl']=='Kelt9b':
        data_dic['Atm']['cont_range']=[[-100.,-50.],[50.,100.]]  
    elif gen_dic['transit_pl']=='WASP76b':
        data_dic['Atm']['cont_range']=[[-100.,-30.],[20.,100.]]      
    


    #-----------------------------

    #Aligning atmospheric profiles
    gen_dic['align_Atm'] = True   &  False
 
    #Calculating/retrieving
    gen_dic['calc_align_Atm'] = True  # &  False  
    
    #Reference planet for alignment
    if gen_dic['star_name']=='WASP121':data_dic['Atm']['ref_pl_align'] = 'WASP121b'
    elif gen_dic['star_name']=='Kelt9': data_dic['Atm']['ref_pl_align'] = 'Kelt9b'       
    elif gen_dic['star_name']=='WASP76': data_dic['Atm']['ref_pl_align'] = 'WASP76b'   
    elif gen_dic['star_name']=='HAT_P33': data_dic['Atm']['ref_pl_align'] = 'HAT_P33b'  
    elif gen_dic['star_name']=='HAT_P49': data_dic['Atm']['ref_pl_align'] = 'HAT_P49b'    

    #-----------------------------

    #Plot all atmospheric profiles together
    #    - if aligned
    plot_dic['all_atm_data']=''   #pdf

    #Plot 2D maps of atmospheric profiles
    #    - in the star or planet rest frame
    plot_dic['map_Atm_prof']=''   #png

    #Plot individual atmospheric spectra or CCFs
    #    - in the star or planet rest frame
    plot_dic['sp_atm']=''     #pdf  
    plot_dic['CCFatm']=''    












    '''
    Binning atmospheric profiles 
        - for analysis purpose (original profiles are not replaced)
        - this module can be used to boost the SNR by combining exposures, or to calculate a global master, in a given visit or in several visits
    '''

    #Activate 
    gen_dic['Atmbin'] = True  &  False
    gen_dic['Atmbinmultivis'] = True  &  False

    #Calculating/retrieving data
    gen_dic['calc_Atmbin']=True  #  &  False  
    gen_dic['calc_Atmbinmultivis'] = True  &  False

    #Visits to be included in the binning, for each instrument
    #    - leave empty to use all visits
    data_dic['Atm']['vis_in_bin']={}  
    if gen_dic['transit_pl']=='WASP76b':
        data_dic['Atm']['vis_in_bin']={'ESPRESSO':['2018-09-03','2018-10-31']}    
    

    #Indexes of exposures that contribute to the master series, for each instrument/visit
    #    - indexes are relative to the in-transit table for absorption signals, and to the global tables for emission signals
    #    - leave empty to use all in-exposures 
    data_dic['Atm']['idx_in_bin']={}   
    if gen_dic['transit_pl']=='WASP121b':
        data_dic['Atm']['idx_in_bin']={'binned':{'HARPS-binned':range(17,29)}}  #indexes in 2 to 13, full in-transit 
        data_dic['Atm']['idx_in_bin']={'binned':{'HARPS-binned':range(17,23)}}  #first half
        data_dic['Atm']['idx_in_bin']={'binned':{'HARPS-binned':range(23,29)}}  #second half
    elif gen_dic['transit_pl']=='Kelt9b':
        data_dic['Atm']['idx_in_bin']={'HARPN':{'31-07-2017':range(3,21)}} #a redefinir    #full in-transit
    elif gen_dic['transit_pl']=='WASP76b':
        data_dic['Atm']['idx_in_bin']={'ESPRESSO':{'2018-09-03':range(1,21),'2018-10-31':range(1,38)}}


    #Choose bin dimension
    #    - possibilities :
    # + 'phase': profiles are binned over phase
    #    - beware to use the alignement module to calculate binned profiles in the planet rest frame
    data_dic['Atm']['dim_bin']='phase'        
    
    #New bin definition
    #    - bins are defined for each instrument/visit
    #    - bins can be defined
    # + manually: indicate lower/upper bin boundaries (ordered)
    # + automatically : indicate total range and number of bins   
    if gen_dic['transit_pl']=='WASP76b':   
        data_dic['Atm']['prop_bin']={
   
            'ESPRESSO':{'2018-09-03':{'bin_low':np.arange(-0.03,0.03,0.01),'bin_high':np.arange(-0.03,0.03,0.01)+0.01},
                        '2018-10-31':{'bin_low':np.arange(0.,0.03,0.01),'bin_high':np.arange(0.,0.03,0.01)+0.01},
                        'binned':{'bin_low':np.arange(0.,0.03,0.01),'bin_high':np.arange(0.,0.03,0.01)+0.01}}
            
            # 'ESPRESSO':{'2018-09-03':{'bin_range':[-0.035,0.036],'nbins':3},
            #             '2018-10-31':{'bin_range':[0.,0.036],'nbins':3}} 
            }
    elif gen_dic['star_name']=='HAT_P33':   
        data_dic['Atm']['prop_bin']={'HARPN':{'20191204':{'bin_range':[-0.1,0.1],'nbins':1}}}
    elif gen_dic['star_name']=='HAT_P49':   
        data_dic['Atm']['prop_bin']={'HARPN':{'20200730':{'bin_range':[-0.1,0.1],'nbins':1}}}
        
    #Plot 2D maps of binned profiles
    plot_dic['map_Atmbin']=''   #png
    
    #Plot binned spectra or CCFs
    plot_dic['sp_Atmbin']=''  
    plot_dic['CCF_Atmbin']=''   #pdf  

    #Plot residuals between binned CCFs and their fit
    plot_dic['CCF_Atmbin_res']=''  # pdf












    ##################################################################################################
    #%%% Module: analyzing atmospheric profiles
    #    - can be applied to:
    # + 'fit_Atm': profiles in the star rest frame, original exposures, for all formats
    # + 'fit_Atm_1D': profiles in the star or surface (if aligned) rest frame, original exposures, converted from 2D->1D 
    # + 'fit_Atmbin' : profiles in the star or surface (if aligned) rest frame, binned exposures, all formats
    # + 'fit_Atmbinmultivis' : profiles in the surface rest frame, binned exposures, all formats
    ##################################################################################################
    
    #Activate for original or binned profiles
    gen_dic['fit_Atm'] = True    &  False
    gen_dic['fit_Atm_1D'] = True   &  False
    gen_dic['fit_Atmbin']=True     &  False
    gen_dic['fit_Atmbinmultivis']=True     &  False
    
    #Calculating/retrieving data
    gen_dic['calc_fit_Atm']=True   # &  False 
    gen_dic['calc_fit_Atm_1D']=True #  &  False  
    gen_dic['calc_fit_Atmbin']=True  #  &  False 
    gen_dic['calc_fit_Atmbinmultivis']=True  #  &  False  


    #Print fits evaluation
    data_dic['Atm']['verbose'] =True      &   False  

    #Fit mode 
    #    - chi2 or MCMC
    data_dic['Atm']['fit_mod']='chi2'
    
    #Define model
    data_dic['Atm']['model']='gauss'
    
    
    #Define area and amplitude thresholds for detection of atmospheric line (in sigma)
    data_dic['Atm']['thresh_area']=5.
    data_dic['Atm']['thresh_amp']=4.  

    #Fixed model properties
    data_dic['Atm']['mod_prop']={}
    if gen_dic['transit_pl']=='WASP121b':
        data_dic['Atm']['mod_prop']={'binned':{'HARPS-binned':{'rv':0.}}}
    if gen_dic['transit_pl']=='Kelt9b':
        data_dic['Atm']['mod_prop']={'HARPN':{'20-07-2018':{'rv':0.}, 
                                    '31-07-2017':{'rv':0.}}}
    
        
       
    #Priors on the fits to the CCFs
    data_dic['Atm']['line_fit_priors']={}
    if gen_dic['transit_pl']=='Kelt9b':
        data_dic['Atm']['line_fit_priors']={'H_FWHM':12.}
    
    
    #Force detection flag for original profiles
    #    - set flag to True at relevant index for the local CCFs to be considered detected, or false to force a non-detection
    #    - indices for each dataset are relative to in-transit indices for absorption signals, global indexes for emission signals
    #    - leave empty for automatic detection
    data_dic['Atm']['idx_force_det']={}
    if gen_dic['transit_pl']=='Kelt9b': 
        data_dic['Atm']['idx_force_det']={'HARPN':{'20-07-2018':{0:False,19:False},'31-07-2017':{0:False,15:False}}}     
    
    
    #Force detection flag for binned profiles
    #    - indices are specific to the binning properties
    data_dic['Atm']['idx_force_detbin']={} 
    data_dic['Atm']['idx_force_detbinmultivis']={}     

    
    #%%%% Direct measurements
    #    - format is {prop_name:{options}}
    #    - possibilities:
    # + integrated signal: 'int_sign' : {'rv_range':[[rv1,rv2],[rv3,rv4]] exact ranges over which the integral is performed, in the planet rest frame, in km/s}
    data_dic['Atm']['meas_prop']={}

    #Define range over which CCFs are fitted
    #   - leave empty to fit over the entire range of definition of the CCF
    #   - otherwise, define [ [rv1,rv2] , [rv3,rv4] , [rv5,rv6] , ... ] with rv defined in the star velocity rest frame
    if gen_dic['transit_pl']=='WASP121b':
        data_dic['Atm']['fit_range']=[[-100.,100.]]
    elif gen_dic['transit_pl']=='Kelt9b':
        data_dic['Atm']['fit_range']=[[-100.,100.]]
    elif gen_dic['transit_pl']=='WASP76b':
        data_dic['Atm']['fit_range']=[[-100.,100.]]   

    #Select order over which fit is performed
    #    - relevant for 2D spectra only
    data_dic['Atm']['fit_prof']['order']={} 

    #Trimming range
    data_dic['Atm']['fit_prof']['trim_range']={}


    #Plot properties of CCFs
    plot_dic['propCCFatm']=''

    #Plot residuals between CCFs and their fit
    plot_dic['CCFatm_res']=''  # pdf


















    '''
    Converting 2D atmospheric spectra into 1D spectra
        - every operation afterwards will be performed on those profiles
    '''

    #Activate
    gen_dic['spec_1D_Atm']=True  & False

    #Calculating/retrieving data
    gen_dic['calc_spec_1D_Atm']=True   # &  False   


    #%%%% Multi-threading
    gen_dic['nthreads_spec_1D_Atm']= 14


    #Properties of 1D spectral table, specific to each instrument
    #    - tables are uniformely spaced in ln(w) (with d[ln(w)] = dw/w)
    #      start and end values given in A    
    data_dic['Atm']['spec_1D_prop']={
        'ESPRESSO':{'dlnw':1./5000.,'w_st':3000.,'w_end':8000.}}
    
    #Plot 2D maps
    plot_dic['map_Atm_1D']=''   #'png

    #Plot individual spectra
    plot_dic['sp_1D_atm']=''     #pdf         
    
    
    
    
    
    
    
    ##################################################################################################
    #%%% Module: atmospheric CCF masks
    ##################################################################################################
    data_dic['Atm']['mask'] = {}
    
    #%%%% Activating
    gen_dic['def_Atmmasks'] = True  #  &  False

    #%%%% Multi-threading
    data_dic['Atm']['mask']['nthreads'] = 14 


    #%%%% Plot settings 

    
    
    
    
    
    
















    '''
    Analysis of atmospheric properties from each exposure with a common model
        - with properties derived from individual profiles
        - several options need to be controlled from within the function
    '''    
    PropAtm_fit_dic={}
    
    #Activate 
    gen_dic['fit_AtmProp'] = True    &  False
    
    #Instrument and visits to be fitted
    if gen_dic['transit_pl']=='WASP76b':
        PropAtm_fit_dic['fit_sel']={'ESPRESSO':['2018-09-03','2018-10-31']}
        
    #Fit mode
    #    - 'chi2' or 'mcmc'
    PropAtm_fit_dic['fit_mod']='mcmc' 

    #Print on-screen fit information
    PropAtm_fit_dic['verbose']=True
    
    #Plot chi2 values for each datapoint
    plot_dic['chi2_fit_AtmProp']=''   #pdf      
  
    
     
    
    
    
    
    
    
    

    '''
    Fit of joined atmospheric profiles from combined (unbinned) instruments and visits 
        - currently defined for CCF only
        - the contrast, FWHM, position of the atmospheric lines (before instrumental convolution) are fitted as polynomials of orbital phase
        - several options need to be controlled from within the function
    '''  
    AtmProf_fit_dic={}

    #Activate 
    gen_dic['fit_AtmProf'] = True   &  False

    #Indexes of in-transit exposures to be fitted, in each visit
    #    - indexes are relative to in-transit or global tables depending on the signal type
    #    - define instruments and visits to be fitted (they will not be fitted if not used as keys, or if set to []), set their value to 'all' for all in-transit exposures to be fitted
    #    - by default, all exposures with defined profiles will be fitted
    if gen_dic['transit_pl']=='WASP76b':
        AtmProf_fit_dic['idx_in_fit']={
#            'ESPRESSO':{'2018-10-31':range(2,13)+range(27,36),'2018-09-03':range(1,8)+range(14,20)},
            'ESPRESSO':{'2018-10-31':list(range(2,18))+list(range(25,37)),'2018-09-03':list(range(1,10))+list(range(14,20))}}

    #Define range over which CCFs are fitted
    #   - leave empty to fit over the entire range of definition of the CCF
    #   - otherwise, define [ [rv1,rv2] , [rv3,rv4] , [rv5,rv6] , ... ] with rv defined in the star velocity rest frame
    AtmProf_fit_dic['fit_range_all']=[]
    if gen_dic['transit_pl']=='WASP76b':AtmProf_fit_dic['fit_range_all']=[[-150.,150.]] 
        
    #Fit mode
    #    - 'chi2' or 'mcmc'
    AtmProf_fit_dic['fit_mod']='mcmc' 

    #Print on-screen fit information
    AtmProf_fit_dic['verbose']=True  &  False

    #Path to saved properties determining the profile variations in the processed dataset
    #    - properties used in the module are saved by default when module is run, and must be put manually in another directory
    #    - those properties will be used in the pipeline to define theoretical profiles
    if gen_dic['transit_pl']=='WASP76b':    
        glob_fit_dic['IntrProf']['AtmProf_prop_path']='/Travaux/ANTARESS/En_cours/WASP76b_Saved_data/Atm_data_prop/Fit/AtmProf_prop.npz'
 

    
    
    
    
    

    
    
    
    
    
    
    
    
    '''
    Call to main function 
    '''
    
    #Run over nominal settings properties
    if not gen_dic['grid_run']:
        ANTARESS_main(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic,PropAtm_fit_dic,AtmProf_fit_dic, corr_spot_dic,all_system_params[gen_dic['star_name']])
    
    #Run over a grid of properties
    else:
    
        # #Run the pipeline over each individual order in ESPRESSO data
        # for iord in range(85): 
        #     if iord not in [59,68,74,75,76,77,78,82]:    #empty CCF orders (tellurics)
        #     # if iord not in [59,76,77,78,82]:    
        #         print('--------------------------------------------')
        #         print('Order :',iord,'(slices = ',2*iord,2*iord+1,')')
        #         gen_dic['orders4ccf']['ESPRESSO'] = [2*iord,2*iord+1] 
                
        #         ANTARESS_main(data_dic,mock_dic,gen_dic,data_dic['Res'],data_dic['DI'],data_dic['Atm'],theo_dic,plot_dic,data_dic['Intr'],glob_fit_dic['IntrProp'],glob_fit_dic['IntrProf'],detrend_prof_dic,PropAtm_fit_dic,AtmProf_fit_dic, corr_spot_dic, glob_fit_dic['ResProf'],stars_params,planets_params)
                              
            
            
        #Run the pipeline over each individual order in HARPS-N data
        for iord in range(69): 
            print('Order ',str(iord))
            # if iord not in [54,63,68]:  #Old masks
            if iord not in [53,54,60,63,64]:   #New masks
                gen_dic['orders4ccf']['HARPN']=[iord] 
            
                ANTARESS_main(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic,PropAtm_fit_dic,AtmProf_fit_dic, corr_spot_dic,all_system_params[gen_dic['star_name']])
                          
        
    
    
    
    stop('End')    

