#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np
import sys
import os as os_system
import glob as glob
from os.path import exists as path_exist
from antaress.ANTARESS_general.utils import dataload_npz, datasave_npz,stop
from antaress.ANTARESS_analysis.ANTARESS_inst_resp import return_spec_nord

'''
Initialization functions
'''

def save_system(input_nbook):

    #Deactivate all notebook plots
    for key_plot in ['system_view','prop_DI','prop_Intr','DI_prof','Intr_prof','map_Intr_prof','map_Intr_prof_est','map_Intr_prof_res','map_Diff_prof','flux_ar','trans_ar','gcal_ord','noises_ord','tell_CCF','tell_prop','Fbal_corr','cosm_corr']:input_nbook['settings']['plot_dic'][key_plot] = ''
    
    input_nbook['saved_data_path'] = input_nbook['working_path']+'/'+input_nbook['par']['star_name'] +'/'+input_nbook['par']['main_pl'] + '_Saved_data'
    print('System stored in : ', input_nbook['saved_data_path'])
    if (not path_exist(input_nbook['saved_data_path'])): os_system.makedirs(input_nbook['saved_data_path'])

    #Saving previously processed notebooks
    if 'all_nbooks' in input_nbook:
        all_input_nbook = input_nbook['all_nbooks']
        input_nbook.pop('all_nbooks')
    else:all_input_nbook={}
    
    #Saving contents of current notebook under its name, so that we can track the origin of the settings in other notebooks
    all_input_nbook[input_nbook['type']] = input_nbook 
    
    #Saving contents in notebook-specific field, so that we can track the origin of the settings in other notebooks
    datasave_npz(input_nbook['saved_data_path']+'/'+'init_sys',all_input_nbook)
    
    return None


def load_nbook(input_nbook, nbook_type):
    all_input_nbook = dataload_npz(input_nbook['working_path']+'/'+input_nbook['star_name']+'/'+input_nbook['pl_name']+'_Saved_data/init_sys')
    curr_working_path = deepcopy(input_nbook['working_path'])

    #Retrieving relevant notebook settings
    if nbook_type in ['mock','Reduc']:
        input_nbook = all_input_nbook['setup']
    elif nbook_type=='Processing':
        if ('mock' in all_input_nbook) and ('Reduc' in all_input_nbook):
            stop('ERROR: do not generate a mock dataset of a system while processing a real dataset of the same system')
        elif ('mock' in all_input_nbook):input_nbook = all_input_nbook['mock']
        elif ('Reduc' in all_input_nbook):input_nbook = all_input_nbook['Reduc']
    elif nbook_type=='RMR': 
        input_nbook = all_input_nbook['Processing']  
    elif nbook_type=='Trends':     
        input_nbook = all_input_nbook['Reduc']
    else:stop('ERROR : notebook type '+nbook_type+' not recognized')

    #Updating notebook type to current notebook
    input_nbook['type'] = nbook_type 

    #Updating working directory to the one from current notebook
    input_nbook['working_path'] = curr_working_path
    input_nbook['plot_path']=input_nbook['working_path']+'/'+input_nbook['par']['star_name']+'/'+input_nbook['par']['main_pl']+'_Plots/'

    #Storing settings of all processed notebooks
    #    - keys of all_input_nbook are nbook names
    input_nbook['all_nbooks'] = deepcopy(all_input_nbook)
    
    #Retrieving dataset in ANTARESS format
    if nbook_type in ['Processing','RMR','Trends']:
        input_nbook['settings']['gen_dic']['calc_proc_data']=False
    
        #Deactivating all 'Reduc' plots and calculation modes
        if nbook_type=='Trends':
            for plot_key in input_nbook['settings']['plot_dic']:
                input_nbook['settings']['plot_dic'][plot_key]=''

            for calc_key in ['gcal','corr_tell','glob_mast','corr_Fbal','cosm','wig','detrend','DI_CCF']:input_nbook['settings']['gen_dic']['calc_'+calc_key] = False
    
    #Deactivating all default modules so that the workflow can be run with the notebook-selected modules
    if nbook_type=='Reduc':
        input_nbook['settings']['gen_dic']['corr_tell'] = False
        input_nbook['settings']['gen_dic']['glob_mast'] = False
        input_nbook['settings']['gen_dic']['corr_Fbal'] = False
        input_nbook['settings']['gen_dic']['corr_FbalOrd'] = False    
        input_nbook['settings']['gen_dic']['corr_cosm'] = False    
        input_nbook['settings']['gen_dic']['calc_FbalOrd'] = False    

    # Detrended data
    if ('detrend_prof' in input_nbook['settings']['gen_dic']) & (nbook_type!='Trends'):
        input_nbook['settings']['gen_dic']['detrend_prof'] = True
        input_nbook['settings']['gen_dic']['calc_detrend_prof']= False
        input_nbook['settings']['detrend_prof_dic']['corr_trend'] = True

    return input_nbook

def init():
    input_nbook = {
        #current notebook type
        'type':'setup',
        #notebook inputs that will overwrite system properties file
        'system' : {},  
        #notebook inputs that will overwrite configuration settings file
        'settings' : {'gen_dic':{'data_dir_list':{},'type':{}},
                      'mock_dic':{'visit_def':{},'sysvel':{},'intr_prof':{},'flux_cont':{},'set_err':{}},
                      'data_dic':{'DI':{'sysvel':{}},
                                  'Intr':{},'Diff':{}},
                      'glob_fit_dic':{'DIProp':{},'IntrProp':{},'IntrProf':{},'DiffProf':{}},
                      'plot_dic':{},
                      'detrend_prof_dic':{}
                     },
        #notebook inputs that will overwrite plot configuration settings file
        'plots' : {},
        #---------------------------------------
        #Notebook inputs for internal use
        #Processing and analysis
        'par' : {'loc_prof_est':False},      
        #Spectral reduction
        'sp_reduc':{},
        #Detrending of CCFs
        'DI_trend':{},
        #Tracks which fits were performed
        'fits':[],            
    }         

    return input_nbook

def init_star(input_nbook):
    input_nbook['settings']['gen_dic']['star_name'] = input_nbook['par']['star_name']
    if 'vsini' not in input_nbook['par']:vsini=1.
    else:vsini = input_nbook['par']['vsini']
    if 'istar' not in input_nbook['par']:istar=90.
    else:istar = input_nbook['par']['istar']
    input_nbook['system'][input_nbook['par']['star_name']]={  
            'star':{
                'Rstar':input_nbook['par']['Rs'],
                'veq':vsini,
                'istar':istar, 
                'sysvel':input_nbook['par']['sysvel'],
                }}
    input_nbook['settings']['data_dic']['DI']['system_prop']={'achrom':{'LD':['quadratic'],'LD_u1' : [input_nbook['par']['ld_u1']],'LD_u2' : [input_nbook['par']['ld_u2']]}}
    return None   

def init_spot(input_nbook,sp_type):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']
    if sp_type == 'main':
        input_nbook['settings']['mock_dic']['ar_prop']={inst:{
                                                                vis:{}
                                                                }
                                                           }
        input_nbook['settings']['gen_dic']['studied_ar'] = {}
        input_nbook['settings']['data_dic']['DI']['ar_prop'] = {'achrom':{'LD':['quadratic'],'LD_u1' : [input_nbook['par']['ld_spot_u1']],'LD_u2' : [input_nbook['par']['ld_spot_u2']]}}
        input_nbook['settings']['data_dic']['DI']['transit_prop'] = {'nsub_Dstar':201., 
                                                                     inst:{
                                                                          vis:{'mode':'simu', 'n_oversamp':5.}
                                                                          }
                                                                     }
    for key in ['lat', 'Tc', 'ang', 'fctrst']:
        if key=='Tc': temp=key+'_ar'
        else:temp=key
        input_nbook['settings']['mock_dic']['ar_prop'][inst][vis][temp+'__IS'+inst+'_VS'+vis+'_AR'+input_nbook['par']['spot_name']]=input_nbook['par'][key]
    input_nbook['settings']['gen_dic']['studied_ar'][input_nbook['par']['spot_name']]={inst:[vis]}
    input_nbook['settings']['data_dic']['DI']['ar_prop']['achrom'][input_nbook['par']['spot_name']]=[input_nbook['par']['ang']*np.pi/180.]
    input_nbook['settings']['theo_dic']=input_nbook['settings']['mock_dic']
    return None

def init_pl(input_nbook,pl_type):
    input_nbook['system'][input_nbook['par']['star_name']][input_nbook['par']['planet_name']]={  
                'period':input_nbook['par']['period'],
                'TCenter':input_nbook['par']['T0'],  
                'ecc':input_nbook['par']['ecc'],
                'omega_deg':input_nbook['par']['long_per'],   
                'Kstar':input_nbook['par']['Kstar'],
                }        
    if pl_type=='main':
        input_nbook['par']['main_pl'] = deepcopy(input_nbook['par']['planet_name'])
        input_nbook['settings']['gen_dic']['studied_pl']={input_nbook['par']['main_pl']:{}}
        input_nbook['system'][input_nbook['par']['star_name']][input_nbook['par']['planet_name']]['inclination']=input_nbook['par']['incl']
        if 'lambda' not in input_nbook['par']:lambda_pl=0.
        else:lambda_pl = input_nbook['par']['lambda']
        input_nbook['system'][input_nbook['par']['star_name']][input_nbook['par']['planet_name']]['lambda_proj']=lambda_pl
        input_nbook['system'][input_nbook['par']['star_name']][input_nbook['par']['planet_name']]['aRs']=input_nbook['par']['aRs']
        input_nbook['settings']['data_dic']['DI']['system_prop']['achrom'][input_nbook['par']['planet_name']]=[input_nbook['par']['RpRs']]
    
        #Paths
        input_nbook['plot_path'] = input_nbook['working_path']+'/'+input_nbook['par']['star_name']+'/'+input_nbook['par']['main_pl']+'_Plots/'

    return None     
    
def add_vis(input_nbook,mock=False):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']
    if inst not in input_nbook['settings']['gen_dic']['studied_pl'][input_nbook['par']['main_pl']]:
        input_nbook['settings']['gen_dic']['studied_pl'][input_nbook['par']['main_pl']][inst]=[]
    input_nbook['settings']['gen_dic']['studied_pl'][input_nbook['par']['main_pl']][inst]+=[vis]
    
    #Initializing mock dataset
    if mock:   
        input_nbook['settings']['gen_dic']['mock_data']=True
        input_nbook['settings']['gen_dic']['type'][inst] = 'CCF'
        input_nbook['par']['type']='CCF'
        if inst not in input_nbook['settings']['mock_dic']['visit_def']:
            input_nbook['settings']['mock_dic']['visit_def'][inst]={}
        input_nbook['settings']['mock_dic']['visit_def'][inst][vis]={'exp_range':np.array(input_nbook['par']['range']),'nexp':int(input_nbook['par']['nexp'])}
        
        dbjd =  (input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][1]-input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][0])/input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['nexp']
        n_in_visit = int((input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][1]-input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][0])/dbjd)
        bjd_exp_low = input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][0] + dbjd*np.arange(n_in_visit)
        bjd_exp_high = bjd_exp_low+dbjd      
        bjd_exp_all = 0.5*(bjd_exp_low+bjd_exp_high)
        input_nbook['par']['t_BJD'] = {'inst':inst,'vis':vis,'t':bjd_exp_all}
    
    #Initializing real dataset
    else:
        input_nbook['settings']['gen_dic']['mock_data']=False
        input_nbook['settings']['gen_dic']['type'][inst] = deepcopy(input_nbook['par']['type'])
        if inst not in input_nbook['settings']['gen_dic']['data_dir_list']:
            input_nbook['settings']['gen_dic']['data_dir_list'][inst]={}
        input_nbook['settings']['gen_dic']['data_dir_list'][inst][vis] = input_nbook['par']['data_dir']

    return None

def set_sysvel(input_nbook):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night'] 
    
    #For mock dataset generation
    if input_nbook['type'] == 'mock':
        if inst not in input_nbook['settings']['mock_dic']:input_nbook['settings']['mock_dic']['sysvel'][inst]={}
        input_nbook['settings']['mock_dic']['sysvel'][inst][vis] = input_nbook['par']['gamma']

    #For processing and trend characterization
    if inst not in input_nbook['settings']['data_dic']['DI']:input_nbook['settings']['data_dic']['DI']['sysvel'][inst]={}
    input_nbook['settings']['data_dic']['DI']['sysvel'][inst][vis] = input_nbook['par']['gamma']
    
    return None


'''
Processing functions
'''

def align_prof(input_nbook):
    input_nbook['settings']['gen_dic']['align_DI']=True
    return None

def flux_sc(input_nbook):
    input_nbook['settings']['gen_dic']['flux_sc']=True

    #Processing mock dataset: scaled to the correct level by construction
    if input_nbook['settings']['gen_dic']['mock_data']:
        input_nbook['settings']['data_dic']['DI']['rescale_DI'] = False 
    return None

def DImast_weight(input_nbook):
    input_nbook['settings']['gen_dic']['DImast_weight']=True
    return None

def extract_diff(input_nbook):
    input_nbook['settings']['gen_dic']['diff_data']=True
    input_nbook['settings']['gen_dic']['nthreads_diff_data'] = 8
    input_nbook['settings']['data_dic']['Diff']['extract_in'] = False
    return None

def extract_intr(input_nbook):
    input_nbook['settings']['gen_dic']['intr_data']=True
    return None

def conv_CCF(input_nbook,prof_type):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']
    if (input_nbook['settings']['gen_dic']['type'][inst]=='CCF'):
        print('Dataset already in CCF format : skipping conversion')
    else:
        input_nbook['par']['type'] = 'CCFfromspec'
        
        input_nbook['settings']['gen_dic'][prof_type+'_CCF'] = True
        input_nbook['settings']['gen_dic']['calc_'+prof_type+'_CCF'] = input_nbook['par']['calc_CCF']
    
        #ANTARESS RV grid settings are defined in the solar barycentric rest frame
        #   - settings are provided relative to the input systemic rv and must be shifted by the input 'sysvel' (at this
        # stage of the notebooks the visit-specific values are not available)
        rv_shift = input_nbook['system'][input_nbook['par']['star_name']]['star']['sysvel']
        input_nbook['settings']['gen_dic'].update({
            'start_RV' : input_nbook['par']['start_RV'] + rv_shift,
            'end_RV'   : input_nbook['par']['end_RV']   + rv_shift,
            'dRV'      : input_nbook['par']['dRV'],
            'CCF_mask' : {inst : input_nbook['working_path'] + '/' +input_nbook['par']['mask_path']}
            })
    return None


def loc_prof_est(input_nbook):
    input_nbook['settings']['gen_dic']['loc_prof_est']=True
    input_nbook['par']['loc_prof_est'] = True
    return None

def diff_prof_corr(input_nbook):
    input_nbook['settings']['gen_dic']['diff_prof_est']=True
    input_nbook['settings']['gen_dic']['calc_diff_prof_est']=True
    input_nbook['par']['diff_prof_corr'] = True
    return None

'''
Analysis functions
'''
def ana_prof(input_nbook,data_type):
    if ('CCF' not in input_nbook['par']['type']):
        print('Data in spectral mode: no fit performed')
    else:
        inst = input_nbook['par']['instrument']
        vis = input_nbook['par']['night'] 
        input_nbook['settings']['gen_dic']['fit_'+data_type]=True
    
        #Retrieval mode
        if 'calc_fit' in input_nbook['par']:
            input_nbook['settings']['gen_dic']['calc_fit_'+data_type] = deepcopy(input_nbook['par']['calc_fit'])
            input_nbook['par'].pop('calc_fit')

        #Fit and continuum ranges
        #    - notebook ranges are provided in the star rest frame
        #    - ANTARESS ranges are relative to the solar system barycenter for DI profiles (and must thus be shifted by the input 'sysvel', since at this stage of the notebooks the visit-specific values are not available), and relative to the star otherwise.
        if data_type=='DI':rv_shift = input_nbook['system'][input_nbook['par']['star_name']]['star']['sysvel']
        else:rv_shift=0.
        if 'cont_range' in input_nbook['par']:
            cont_range = deepcopy(input_nbook['par']['cont_range'])
            cont_range_shifted = []
            for bd in cont_range:cont_range_shifted+=[bd[0]+rv_shift,bd[1]+rv_shift]
            input_nbook['settings']['data_dic'][data_type]['cont_range']= {inst: {0:cont_range_shifted}}
            input_nbook['par'].pop('cont_range')
        if 'fit_range' in input_nbook['par']:
            input_nbook['settings']['data_dic'][data_type]['fit_range'] = {inst: {vis: input_nbook['par']['fit_range']+rv_shift}}
            input_nbook['par'].pop('fit_range')

        #Guess values
        if ('guess' in input_nbook['par']):
            input_nbook['settings']['data_dic']['mod_prop'] = {}
            for prop in input_nbook['par']['guess']:
                input_nbook['settings']['data_dic']['mod_prop'][prop] = {'vary':True, inst:{vis:{'guess':input_nbook['par']['guess'][prop]}}}

        #Fit settings
        if ('fit_mode' in input_nbook['par']):
            input_nbook['settings']['data_dic'][data_type]['fit_mode']=deepcopy(input_nbook['par']['fit_mode'])
            input_nbook['par'].pop('fit_mode')
            input_nbook['settings']['data_dic'][data_type]['progress']=False
        else:input_nbook['settings']['data_dic'][data_type]['fit_mode'] = 'chi2'
        if 'run_mode' in input_nbook['par']:
            input_nbook['settings']['data_dic'][data_type]['run_mode']=deepcopy(input_nbook['par']['run_mode'])   
            if input_nbook['par']['run_mode']=='reuse':
                input_nbook['settings']['data_dic'][data_type]['save_MCMC_chains']=''
                input_nbook['settings']['data_dic'][data_type]['save_MCMC_corner']=''
            input_nbook['par'].pop('run_mode')        
            
        #Manual priors
        if ('priors' in input_nbook['par']):
            input_nbook['settings']['data_dic'][data_type]['priors']=deepcopy(input_nbook['par']['priors'])
            for key in input_nbook['settings']['data_dic'][data_type]['priors']:
                input_nbook['settings']['data_dic'][data_type]['priors'][key]['mod'] = 'uf'
            input_nbook['par'].pop('priors') 

        #Deactivate detection thresholds to avoid automatic computation of amplitude
        input_nbook['settings']['data_dic'][data_type]['thresh_area']=None
        input_nbook['settings']['data_dic'][data_type]['thresh_amp']=None

    return None



def ana_jointprop(input_nbook,data_type):
    ana_jointcomm(input_nbook,data_type,'Prop')    
    return None

def ana_jointprof(input_nbook,data_type):
    ana_jointcomm(input_nbook,data_type,'Prof')  
    return None

def ana_jointcomm(input_nbook,data_type,ana_type):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']
    input_nbook['fits']+=[data_type+ana_type]
    input_nbook['settings']['gen_dic']['fit_'+data_type+ana_type] = True 
    
    #Fit mode
    if ('fit_mode' in input_nbook['par']):
        input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['fit_mode']=deepcopy(input_nbook['par']['fit_mode'])
        input_nbook['par'].pop('fit_mode')
        input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['progress']=False
    else:input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['fit_mode']='chi2'
    
    #Running the module, but retrieving the results if MCMC was used already
    if ('calc_fit' in input_nbook['par']) and (not input_nbook['par']['calc_fit']) and (input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['fit_mode']=='mcmc'):
        input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['run_mode'] = 'reuse'
        input_nbook['par'].pop('calc_fit')

    #Fitted exposures
    if input_nbook['type']=='Trends':
    
        #Fitting all out-of-transit exposures        
        input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['idx_in_fit']={inst:{vis:'all'}}

    else:
        input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['idx_in_fit'] = {inst:{vis:deepcopy(input_nbook['par']['idx_in_fit'])}}  

    #Fitted properties
    if (ana_type=='Prop'):
        if data_type=='Intr':
            input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['mod_prop'] = {
                'rv':{},
                'ctrst':{},
                'FWHM':{}}
        elif data_type=='DI':
            input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['verbose'] = True

            input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['mod_prop']={}
            for prop_in in ['FWHM','contrast','rv_res']:
                prop = {'FWHM':'FWHM','contrast':'ctrst','rv_res':'rv'}[prop_in]
                coord_ref = deepcopy(input_nbook['DI_trend'][prop_in]['coord'])
                if (inst=='ESPRESSO') and (coord_ref=='snr'):coord = 'snrQ'
                else:coord = coord_ref
                guess_val = {
                    'FWHM':5.,
                    'ctrst':0.5,
                    'rv':input_nbook['system'][input_nbook['par']['star_name']]['star']['sysvel'],
                }[prop]
                input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['mod_prop'][prop]={
                    'c__ord0__IS__VS_':{'vary':True ,'guess':guess_val,'bd':[-100.,100.]}}
                deg = input_nbook['DI_trend'][prop_in]['deg']
                if deg>0:
                    for ideg in range(1,int(deg)+1):
                        input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['mod_prop'][prop][coord+'__pol__ord'+str(ideg)+'__IS__VS_']={
                            {'vary':True ,'guess':0,'bd':[-100.,100.]}}
    
    elif (ana_type=='Prof'):
        input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['mod_prop'] = {}        

        #For joint intrinsic profiles the continuum is left free to vary but initialized within the workflow itself
        if data_type == 'Intr':
            #     - 'Opt_Lvl' set to 2 to avoid system-related issues with C grid file
            input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['Opt_Lvl']=2
            input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['verbose']=True

            #Continuum range
            #    - defined in star rest frame in both notebook and pipeline
            if 'cont_range' in input_nbook['par']:
                cont_range = deepcopy(input_nbook['par']['cont_range'])
                cont_range_shifted = []
                for bd in cont_range:cont_range_shifted+=[bd[0],bd[1]]
                input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['cont_range'] = {inst: {0:cont_range_shifted}} 
                input_nbook['par'].pop('cont_range')

    if ('priors' in input_nbook['par']):input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['priors']={}

    #Guess and prior ranges for intrinsic properties and profiles
    if data_type=='Intr':
        for prop in input_nbook['par']['mod_prop']:
            bd_prop = np.array(input_nbook['par']['mod_prop'][prop])
            if prop in input_nbook['par']['priors']:bd_prior = np.array([input_nbook['par']['priors'][prop]['low'],input_nbook['par']['priors'][prop]['high']])
            else:bd_prior=1.
            sc_fact = 1. 
    
            #RV model
            if prop=='veq':
                prop_main = 'rv'
                prop_name = 'veq'
            elif prop=='lambda':            
                prop_main = 'rv'
                prop_name = 'lambda_rad__pl'+input_nbook['par']['main_pl']   
                sc_fact = np.pi/180.
                bd_prop *= sc_fact
                bd_prior*= sc_fact
            elif prop in ['c1_CB','c2_CB']:              
                prop_main = 'rv'
                prop_name = prop           
            elif prop=='alpha':
                prop_main = 'rv'
                prop_name = 'alpha_rot'
            elif prop=='istar':
                prop_main = 'rv'
                prop_name = 'cos_istar'
                sc_fact = np.pi/180.
                bd_prop = np.cos(bd_prop*sc_fact)  
                bd_prior = np.cos(bd_prior*sc_fact)               
    
            #Line shape            
            elif 'contrast' in prop:
                ideg = int(prop.split('contrast_')[1])
                prop_main = 'ctrst'
                prop_name='ctrst__ord'+str(ideg)+'__IS__VS_'
            elif 'FWHM' in prop:
                ideg = int(prop.split('FWHM_')[1]) 
                prop_main='FWHM'
                prop_name='FWHM__ord'+str(ideg)+'__IS__VS_'
    
            #Spot properties
            elif (('lat' in prop) or ('Tc' in prop) or ('ang' in prop)):
                temp_prop_name,spot_name = prop.split('_')
                if 'Tc' in prop:temp_prop_name+='_ar'
                prop_name = temp_prop_name+'__IS'+input_nbook['par']['instrument']+'_VS'+input_nbook['par']['night']+'_AR'+spot_name
            elif  'fctrst' in prop:
                prop_name = 'fctrst__IS'+input_nbook['par']['instrument']+'_VS'+input_nbook['par']['night']+'_AR'
            mean_prop = np.mean(bd_prop)
            fit_prop_dic = {'vary':True,'guess':mean_prop,'bd':bd_prop}
            if (ana_type=='Prop'):input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['mod_prop'][prop_main][prop_name]=fit_prop_dic
            elif (ana_type=='Prof'):input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['mod_prop'][prop_name]=fit_prop_dic
            if prop in input_nbook['par']['priors']:
                input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['priors'][prop_name] = {'mod':'uf','low':bd_prior[0],'high':bd_prior[1]}
            
    if data_type == 'Diff':
        #Defining continuum range
        low_low = input_nbook['settings']['mock_dic']['DI_table']['x_start']
        low_high = input_nbook['settings']['mock_dic']['DI_table']['x_start'] + 5*input_nbook['settings']['mock_dic']['DI_table']['dx']
        high_low = input_nbook['settings']['mock_dic']['DI_table']['x_end'] - 5*input_nbook['settings']['mock_dic']['DI_table']['dx']
        high_high = input_nbook['settings']['mock_dic']['DI_table']['x_end']
        input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['cont_range'] = {input_nbook['par']['instrument']:{0:[[low_low,low_high],[high_low,high_high]]}}
        
        #Defining fitting range
        input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['fit_range'] = {input_nbook['par']['instrument']:{input_nbook['par']['night']:[[low_high,high_low]]}}
        
        #Defining optimization level
        #     - set to 2 to avoid system-related issues with C grid file 
        input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['Opt_Lvl'] = 2
            
    if ('priors' in input_nbook['par']):input_nbook['par'].pop('priors')
    
    #Walkers
    if ('sampler_set' in input_nbook['par']):
        input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['sampler_set']=deepcopy(input_nbook['par']['sampler_set'])
        input_nbook['par'].pop('sampler_set')  
        
    #Save chains by default
    input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['save_MCMC_chains']='png' 
   
    return None




'''
Mock dataset functions
'''
def set_mock_rv(input_nbook):
    input_nbook['settings']['mock_dic']['DI_table'] = {key:input_nbook['par'][key] for key in ['x_start','x_end','dx']}
    return None

def set_mock_prof(input_nbook):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']   
    if inst not in input_nbook['settings']['mock_dic']['intr_prof']:
        input_nbook['settings']['mock_dic']['intr_prof'][inst] = {'mode':'ana','coord_line':'mu','model': 'gauss','line_trans':None,'mod_prop':{},'pol_mode' : 'modul'} 
    input_nbook['settings']['mock_dic']['intr_prof'][inst]['mod_prop']['ctrst__ord0__IS'+inst+'_VS'+vis] = input_nbook['par']['contrast'] 
    input_nbook['settings']['mock_dic']['intr_prof'][inst]['mod_prop']['FWHM__ord0__IS'+inst+'_VS'+vis]  = input_nbook['par']['FWHM']   
    if inst not in input_nbook['settings']['mock_dic']['flux_cont']:input_nbook['settings']['mock_dic']['flux_cont'][inst] = {}
    input_nbook['settings']['mock_dic']['flux_cont'][inst][vis]  = input_nbook['par']['flux']    
    input_nbook['settings']['mock_dic']['set_err'][inst]  = input_nbook['par']['noise']    
    return None



def processing_mode(input_nbook):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']
    input_nbook['settings']['gen_dic']['calc_proc_data'] = input_nbook['sp_reduc']['calc_proc_data']

    #Spectral data
    if 'spec' in input_nbook['par']['type']:
        input_nbook['par']['nord'] = return_spec_nord(inst)

        #Masking
        if len(input_nbook['sp_reduc']['iexp2keep'])>0:input_nbook['settings']['gen_dic']['used_exp'] = {inst:{vis:input_nbook['sp_reduc']['iexp2keep']}}
        if len(input_nbook['sp_reduc']['iord2del'])>0:
            input_nbook['par']['nord']-=len(input_nbook['sp_reduc']['iord2del'])
            input_nbook['settings']['gen_dic']['del_orders'] = {inst:input_nbook['sp_reduc']['iord2del']}
        if len(input_nbook['sp_reduc']['wav2mask'])>0:
            input_nbook['settings']['gen_dic']['masked_pix'] = {inst:{vis:{'exp_list':[],'ord_list':{}}}}
            for iord in input_nbook['sp_reduc']['wav2mask']:
                input_nbook['settings']['gen_dic']['masked_pix'][inst][vis]['ord_list'][iord] = input_nbook['sp_reduc']['wav2mask'][iord]
    
    return None

'''
Spectral reduction functions
'''

def inst_cal(input_nbook):
    if 'spec' in input_nbook['par']['type']:
        input_nbook['settings']['gen_dic']['calc_gcal']=input_nbook['sp_reduc']['calc_gcal']
    return None

def tell_corr(input_nbook):
    if 'spec' in input_nbook['par']['type']:
        input_nbook['settings']['gen_dic']['corr_tell']=True 
        input_nbook['settings']['gen_dic']['calc_corr_tell']=input_nbook['sp_reduc']['calc_tell']
    
        input_nbook['settings']['gen_dic']['tell_species']    =input_nbook['sp_reduc']['tell_species']
        input_nbook['settings']['gen_dic']['tell_thresh_corr']=input_nbook['sp_reduc']['tell_thresh']
    return None

def fbal_corr(input_nbook):
    if 'spec' in input_nbook['par']['type']:
        inst = input_nbook['par']['instrument']
        vis = input_nbook['par']['night'] 
    
        input_nbook['settings']['gen_dic']['glob_mast'] = True
        input_nbook['settings']['gen_dic']['calc_glob_mast']=input_nbook['sp_reduc']['calc_Fbal']
        
        input_nbook['settings']['gen_dic']['corr_Fbal']=True
        input_nbook['settings']['gen_dic']['calc_corr_Fbal']=input_nbook['sp_reduc']['calc_Fbal']
    
        input_nbook['settings']['gen_dic']['Fbal_vis']=None    #for single visit
    
        #Orders excluded from the fit
        input_nbook['settings']['gen_dic']['Fbal_clip']=False  #forcing the user to select fitted orders
        if len(input_nbook['sp_reduc']['iord2excl']) > 0:
            input_nbook['settings']['gen_dic']['Fbal_ord_fit']  ={inst:{vis:[iord for iord in range(input_nbook['par']['nord']) if iord not in input_nbook['sp_reduc']['iord2excl']]}}
    
        #Using spline by default
        input_nbook['settings']['gen_dic']['Fbal_mod']          = 'spline'
        input_nbook['settings']['gen_dic']['Fbal_smooth']       ={inst: {vis: input_nbook['sp_reduc']['smooth_fac']}}    

    return None

def cosm_corr(input_nbook):
    if 'spec' in input_nbook['par']['type']:
        inst = input_nbook['par']['instrument']
        vis = input_nbook['par']['night']
    
        input_nbook['settings']['gen_dic']['corr_cosm'] = True
        input_nbook['settings']['gen_dic']['calc_cosm'] = input_nbook['sp_reduc']['calc_cosm']
    
        input_nbook['settings']['gen_dic']['al_cosm'] = {'mode':input_nbook['sp_reduc']['align']}
        input_nbook['settings']['gen_dic']['cosm_ncomp'] = input_nbook['sp_reduc']['ncomp']
        input_nbook['settings']['gen_dic']['cosm_thresh'] = {inst:{vis: input_nbook['sp_reduc']['thresh']}}

    return None

def wiggle_corr(input_nbook):
    if 'spec' in input_nbook['par']['type']:
        vis = input_nbook['par']['night']
        input_nbook['settings']['gen_dic']['corr_wig'] = True
        input_nbook['settings']['gen_dic']['calc_wig']= input_nbook['sp_reduc']['calc_wig']
    
        #Screening
        input_nbook['settings']['gen_dic']['wig_exp_init'] = {'mode'     :input_nbook['sp_reduc']['screening'],
                                                              'plot_spec':True,
                                                              'plot_hist':True,
                                                              'y_range'  :input_nbook['sp_reduc']['y_range_scr']}
        
        #Fitted ranges
        if input_nbook['sp_reduc']['fit_range']==[[]]:input_nbook['settings']['gen_dic']['wig_range_fit'] = {}
        else: input_nbook['settings']['gen_dic']['wig_range_fit'] = {vis : input_nbook['sp_reduc']['fit_range']}
    
        #Filtering
        if input_nbook['sp_reduc']['filter']:input_nbook['settings']['gen_dic']['wig_norm_ord'] = False
        input_nbook['settings']['gen_dic']['wig_exp_filt'] = {'mode':input_nbook['sp_reduc']['filter'],
                                                              'win' :input_nbook['sp_reduc']['window'],
                                                              'deg' :input_nbook['sp_reduc']['deg'],
                                                              'plot':True}
    
        #Conditionning correction to filter activation
        input_nbook['settings']['gen_dic']['wig_corr'] = {'mode':input_nbook['sp_reduc']['filter'],
                                                          'path':{},
                                                          'exp_list':{},
                                                          'range':{}}
    
    return None

    
def detrend(input_nbook):
    if ('CCF' not in input_nbook['par']['type']):
        print('Data in spectral mode: no fit performed')
    else:
        inst = input_nbook['par']['instrument']
        vis = input_nbook['par']['night']
        if len(input_nbook['sp_reduc']['detrend'])==0:stop('ERROR: no detrending defined')
    
        input_nbook['settings']['gen_dic']['detrend_prof'] = True
        input_nbook['settings']['gen_dic']['calc_detrend_prof']= input_nbook['sp_reduc']['calc_detrend']
        input_nbook['settings']['detrend_prof_dic']['corr_trend'] = True
    
        rv_shift = input_nbook['system'][input_nbook['par']['star_name']]['star']['sysvel']
        if ('cont_range' in input_nbook['par']):
            cont_range = deepcopy(input_nbook['par']['cont_range'])
            cont_range_shifted = []
            for bd in cont_range:cont_range_shifted+=[bd[0]+rv_shift,bd[1]+rv_shift]
            input_nbook['settings']['data_dic']['DI']['cont_range']= {inst: {0:cont_range_shifted}}
            input_nbook['par'].pop('cont_range')

        input_nbook['settings']['detrend_prof_dic']['prop'] = {inst:{vis:{}}}
        for prop_coord in input_nbook['sp_reduc']['detrend']:
            if (inst=='ESPRESSO') and ('snr' in prop_coord):
                prop_str = prop_coord.split('_snr')[0]
                prop_coord_ref = prop_str+'_snrQ'
            else:prop_coord_ref = prop_coord
            input_nbook['settings']['detrend_prof_dic']['prop'][inst][vis][prop_coord_ref]={'pol':np.array(input_nbook['sp_reduc']['detrend'][prop_coord])}

    return None


def calc_DImast(input_nbook):
    input_nbook['settings']['gen_dic']['calc_DImast'] = True
    return None

def convert_to_1D(input_nbook, plot=False):
    inst = input_nbook['par']['instrument']
    input_nbook['gen_dic']['spec_1D_DI'] = True

    if input_nbook['sp_reduc']['ncores'] != None:
        input_nbook['settings']['gen_dic']['nthreads_spec_1D_DI'] = input_nbook['sp_reduc']['ncores']

    input_nbook['settings']['data_dic']['DI']['spec_1D_prop'] = {inst:{
        'dlnw' : input_nbook['sp_reduc']['wav_step'],
        'w_st' : input_nbook['sp_reduc']['wav_start'],
        'w_end': input_nbook['sp_reduc']['wav_end']
        }}

    if plot:
        input_nbook['plot_dic']['sp_DI_1D'] = 'png'
    return None

def DI_CCF(input_nbook):
    inst = input_nbook['par']['instrument']
    input_nbook['settings']['gen_dic']['DI_CCF'] = True
    input_nbook['settings']['gen_dic']['calc_DI_CCF'] = input_nbook['sp_reduc']['calc_CCF']

    input_nbook['settings']['gen_dic'].update({
        'start_RV' : input_nbook['sp_reduc']['start_RV'] + input_nbook['par']['gamma'],
        'end_RV'   : input_nbook['sp_reduc']['end_RV'] + input_nbook['par']['gamma'],
        'dRV'      : input_nbook['sp_reduc']['dRV'],
        'CCF_mask' : {inst : input_nbook['working_path'] + '/' +input_nbook['sp_reduc']['mask_path']}
        })
    return None

def build_1D_master(input_nbook, plot=False):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']

    input_nbook['settings']['gen_dic']['DIbin'] = True
    input_nbook['settings']['data_dic']['DI'] = {
        'prop_bin':{inst:{vis:{'bin_range':[-0.5,0.5],'nbins':1}}}
    }
    if plot:input_nbook['settings']['plot_dic']['DIbin']='png'
    return None

    return None


'''
Plot functions
'''
def plot_system(input_nbook):
    input_nbook['settings']['plot_dic']['system_view'] = 'png' 
    input_nbook['plots']['system_view']={'t_BJD':input_nbook['par']['t_BJD'],'GIF_generation':True}
    return None


def plot_spec(input_nbook):
    if 'spec' in input_nbook['par']['type']:
        spec_keys=[]
        if input_nbook['sp_reduc']['flux_sp']:
            input_nbook['settings']['plot_dic']['flux_sp'] = 'png'
            spec_keys+=['DI_prof_corr']
            input_nbook['plots']['DI_prof_corr'] = {}
            input_nbook['plots']['DI_prof_corr']['y_range']   = input_nbook['sp_reduc']['y_range_flux']
     
        if input_nbook['sp_reduc']['trans_sp']:
            input_nbook['settings']['plot_dic']['trans_sp'] = 'png'
            spec_keys+=['trans_sp']
            input_nbook['plots']['trans_sp'] = {}     
            input_nbook['plots']['trans_sp']['gap_exp'] = input_nbook['sp_reduc']['gap_exp']
            if input_nbook['sp_reduc']['bin_width']>0.:
                input_nbook['plots']['trans_sp']['bin_width'] = input_nbook['sp_reduc']['bin_width']
                input_nbook['plots']['trans_sp']['plot_bin'] = True
            else:input_nbook['plots']['trans_sp']['plot_bin'] = False
            input_nbook['plots']['trans_sp']['y_range']   = input_nbook['sp_reduc']['y_range_trans']
    
        #Common options
        for plot_key in spec_keys:
            input_nbook['plots'][plot_key]['sp_var']   = 'wav'   
            input_nbook['plots'][plot_key]['rasterized'] = False
            input_nbook['plots'][plot_key]['plot_err'] = input_nbook['sp_reduc']['plot_err']
            input_nbook['plots'][plot_key]['x_range']  = input_nbook['sp_reduc']['x_range']
            input_nbook['plots'][plot_key]['plot_pre']  = input_nbook['sp_reduc']['pre']
            input_nbook['plots'][plot_key]['plot_post']  = input_nbook['sp_reduc']['post']
            input_nbook['plots'][plot_key]['iord2plot']  = input_nbook['sp_reduc']['iord2plot']
            if len(input_nbook['sp_reduc']['iord2plot'])>1:input_nbook['plots'][plot_key]['multi_ord'] = True
            
        for key in ['x_range','pre','post','iord2plot']:input_nbook['sp_reduc'].pop(key)
            
    return None


def inst_cal_plot(input_nbook):
    if 'spec' in input_nbook['par']['type']:
        input_nbook['settings']['plot_dic']['gcal_ord'] = 'png'
        input_nbook['settings']['plot_dic']['noises_ord'] = 'png'
        input_nbook['settings']['plot_dic']['legend'] = True
        input_nbook['plots']['gcal']={'iord2plot':[deepcopy(input_nbook['sp_reduc']['iord2plot_gcal'])],
                                      'iexp2plot':{input_nbook['par']['instrument']:{input_nbook['par']['night']:[deepcopy(input_nbook['sp_reduc']['iexp2plot_gcal'])]}}}
    return None

def tell_corr_plot(input_nbook):
    if 'spec' in input_nbook['par']['type']:
        input_nbook['settings']['plot_dic']['tell_CCF'] = 'png'
        input_nbook['settings']['plot_dic']['tell_prop'] = 'png'
    return None


def fbal_corr_plot(input_nbook):
    if 'spec' in input_nbook['par']['type']:
        input_nbook['settings']['plot_dic']['Fbal_corr'] = 'png'
        input_nbook['plots']['Fbal_corr'] = {'gap_exp': input_nbook['sp_reduc']['gap_exp'] }
    return None

    
def cosm_corr_plot(input_nbook):
    if 'spec' in input_nbook['par']['type']:
        input_nbook['settings']['plot_dic']['cosm_corr']='png'
    return None

def cosmic_search(iexp, iord, input_nbook):
    '''
    Function used to search for cosmics plot
    '''
    path = input_nbook['plot_path'] + 'Spec_raw/Cosmics/'+input_nbook['par']['instrument']+'_'+input_nbook['par']['night']

    if path_exist(path + '/idx'+ str(iexp) + '_iord' + str(iord) + '.png'):
        print('Exposure and order have detected cosmics')
        return True

    else:
        files = sorted(list(file for file in os_system.listdir(path) if file.startswith("idx")))
        l = []
        for file in files:
            val = re.split('[x _ d .]', str(file))
            to_add = 'Exposure ' + val[2] + ', order ' + val[-2]
            l.append(to_add)
        print('No cosmic detected in the defined exposure and order. \nExposures and Orders with cosmics are:\n', '\n'.join(l))
        return False


def plot_prop(input_nbook,data_type):
    inst = input_nbook['par']['instrument']
    input_nbook['settings']['plot_dic']['prop_'+data_type] = 'png' 

    #Plotted properties
    if input_nbook['type'] in ['Trends','Processing']:
        nbook_prop_names = ['rv','rv_res','contrast','FWHM']
        input_nbook['plots']['prop_'+data_type+'_ordin'] = ['rv','rv_res','ctrst','FWHM']
        input_nbook['par']['print_disp'] = ['plot']
    elif input_nbook['type']=='RMR':
        nbook_prop_names = ['rv','contrast','FWHM']
        input_nbook['plots']['prop_'+data_type+'_ordin'] = ['rv','ctrst','FWHM']
        input_nbook['par']['print_disp'] = ['plot']
    elif input_nbook['type']=='mock':
        nbook_prop_names = ['rv']
        input_nbook['plots']['prop_'+data_type+'_ordin'] = ['rv']
    input_nbook['plots']['prop_'+data_type+'_ordin'] = np.array(input_nbook['plots']['prop_'+data_type+'_ordin'])

    #Plot ranges
    for name_prop,plot_prop in zip(nbook_prop_names,input_nbook['plots']['prop_'+data_type+'_ordin']):
        input_nbook['plots']['prop_'+data_type+'_'+plot_prop]={}
        if 'x_range' in input_nbook['par']:
            input_nbook['plots']['prop_'+data_type+'_'+plot_prop]['x_range'] = deepcopy(input_nbook['par']['x_range'])
        if ('y_range' in input_nbook['par']) and (name_prop in input_nbook['par']['y_range']):
            input_nbook['plots']['prop_'+data_type+'_'+plot_prop]['y_range'] = deepcopy(input_nbook['par']['y_range'][name_prop])
    if 'x_range' in input_nbook['par']:input_nbook['par'].pop('x_range')            
    if 'y_range' in input_nbook['par']:input_nbook['par'].pop('y_range')              
    
    #Dispersion of properties
    if ('print_disp' in input_nbook['par']):  
        for plot_prop in input_nbook['plots']['prop_'+data_type+'_ordin']:
            input_nbook['plots']['prop_'+data_type+'_'+plot_prop]['print_disp']=input_nbook['par']['print_disp']
        input_nbook['par'].pop('print_disp') 
    
    #Set error bars depending on the type of fit
    if input_nbook['settings']['data_dic'][data_type]['fit_mode']=='mcmc':
        for plot_prop in input_nbook['plots']['prop_'+data_type+'_ordin']:
            input_nbook['plots']['prop_'+data_type+'_'+plot_prop]['plot_HDI']=True
            input_nbook['plots']['prop_'+data_type+'_'+plot_prop]['plot_err'] = False 

    if (data_type in ['DI','Intr']):      

        #Models
        prop_path = input_nbook['saved_data_path']+'/Joined_fits/'
        
        #Plot fit to joint properties if carried out
        if data_type+'Prop' in input_nbook['fits']:
            for plot_prop in input_nbook['plots']['prop_'+data_type+'_ordin']:
                input_nbook['plots']['prop_'+data_type+'_'+plot_prop].update({
                    data_type+'Prop_path' : prop_path+'/'+data_type+'Prop/'+input_nbook['settings']['glob_fit_dic'][data_type+'Prop']['fit_mode']+'/'   ,
                    'theo_HR_prop' : True})    
                
        if (data_type=='DI'):
            for plot_prop in input_nbook['plots']['prop_'+data_type+'_ordin']:
                if input_nbook['type']=='mock':coord = 'phase'
                elif input_nbook['type']=='Trends':
                    prop_fit = {'FWHM':'FWHM','ctrst':'contrast','rv_res':'rv_res','rv':'rv_res'}[plot_prop]
                    coord_ref = deepcopy(input_nbook['DI_trend'][prop_fit]['coord'])
                    if (inst=='ESPRESSO') and (coord_ref=='snr'):coord = 'snrQ'
                    elif 'phase' in coord_ref:coord = 'phase'
                    else:coord = coord_ref                
                input_nbook['plots']['prop_DI_'+plot_prop]['prop_DI_absc'] = coord
     
        elif (data_type=='Intr'):
            for plot_prop in input_nbook['plots']['prop_'+data_type+'_ordin']:
                input_nbook['plots']['prop_'+data_type+'_'+plot_prop]['plot_disp'] = False         

            #Plot fit to joint profiles if carried out
            if 'IntrProf' in input_nbook['fits']:        
                for plot_prop in input_nbook['plots']['prop_'+data_type+'_ordin']:
                    input_nbook['plots']['prop_Intr_'+plot_prop].update({
                        'IntrProf_path' : prop_path+'/IntrProf/'+input_nbook['settings']['glob_fit_dic'][data_type+'Prop']['fit_mode']+'/'   ,
                        'theo_HR_prof' : True}) 

    return None

def plot_prof(input_nbook,data_type):
    input_nbook['settings']['plot_dic'][data_type] = 'png'
    input_nbook['plots'][data_type]={
        'GIF_generation':True,
        'shade_cont':True,
        'plot_line_model':True,
        'plot_prop':False} 
    if input_nbook['type'] in ['Trends','RMR','Reduc']:
        if input_nbook['type'] in ['Trends','RMR']:input_nbook['par']['fit_type'] = 'indiv'   #overplot fits to individual exposures 
        input_nbook['plots'][data_type]['step'] = 'latest'
    if 'x_range' in input_nbook['par']:
        input_nbook['plots'][data_type]['x_range'] = deepcopy(input_nbook['par']['x_range'])
        input_nbook['par'].pop('x_range')
    if 'y_range' in input_nbook['par']:
        input_nbook['plots'][data_type]['y_range'] = deepcopy(input_nbook['par']['y_range'])
        input_nbook['par'].pop('y_range')
    if data_type in ['DI_prof','Intr_prof']:
        input_nbook['plots'][data_type]['norm_prof'] = True
    if 'fit_type' in input_nbook['par']:
        input_nbook['plots'][data_type]['fit_type'] = deepcopy(input_nbook['par']['fit_type'])
        input_nbook['par'].pop('fit_type')  
    return None

def plot_spot(input_nbook):
    input_nbook['plots']['system_view']['plot_spots'] = True
    input_nbook['plots']['system_view']['mock_ar_prop'] = True
    input_nbook['plots']['system_view']['n_spcell'] = 101
    return None

def plot_map(input_nbook,data_type):
    inst = input_nbook['par']['instrument']
    
    #Plot specific order only if spectral data
    if (input_nbook['settings']['gen_dic']['type'][inst]=='CCF'):input_nbook['par']['iord2plot']=0
    
    #Activate plot related to intrinsic CCF model only if model was calculated
    def_map = True
    if data_type in ['Intr_prof_est','Intr_prof_res'] and (not input_nbook['par']['loc_prof_est']):def_map=False
    if data_type in ['Diff_prof_est','Diff_prof_res'] and (not input_nbook['par']['diff_prof_corr']):def_map=False
    if def_map:
        input_nbook['settings']['plot_dic']['map_'+data_type] = 'png'
        input_nbook['plots']['map_'+data_type] = {}
        input_nbook['plots']['map_'+data_type]['verbose'] = False
        input_nbook['plots']['map_'+data_type]['iord2plot']=[deepcopy(input_nbook['par']['iord2plot'])]
        if 'x_range' in input_nbook['par']:
            input_nbook['plots']['map_'+data_type]['x_range'] = deepcopy(input_nbook['par']['x_range'])
        if 'v_range' in input_nbook['par']:
            input_nbook['plots']['map_'+data_type].update({'v_range_all':{inst:{input_nbook['par']['night']:deepcopy(input_nbook['par']['v_range'])}}}) 
            input_nbook['par'].pop('v_range')
        if data_type=='Intr_prof':
            input_nbook['plots']['map_'+data_type]['norm_prof'] = True
            input_nbook['plots']['map_'+data_type]['theoRV_HR'] = True
        elif data_type=='Intr_prof_est':
            input_nbook['plots']['map_'+data_type]['norm_prof'] = True
            input_nbook['plots']['map_'+data_type]['line_model']='rec'
        elif data_type=='Intr_prof_res':
            input_nbook['plots']['map_'+data_type]['cont_only']=False
            input_nbook['plots']['map_'+data_type]['line_model']='rec'
    return None


def find_exp(iexp, path):
    try:
        return glob.glob(path+'idx'+str(iexp)+'_*')[0]
    except: 
        print('Exposure plot does not exist')
        return None

def find_plot(path):
    try:
        return glob.glob(path)[0]
    except: 
        print('Plot does not exist')
        return None

def find_group(iexp, path):
    try:
        return glob.glob(path+'ExpGroup'+str(iexp)+'Band*')[0]

    except: 
        print('Exposure plot does not exist')
        return None



