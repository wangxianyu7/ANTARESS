#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np
import sys
import os as os_system
from os.path import exists as path_exist
from antaress.ANTARESS_general.utils import dataload_npz, datasave_npz,stop


'''
Initialization functions
'''

def save_system(input_nbook):
    path = input_nbook['working_path']+input_nbook['par']['star_name'] +'/'+input_nbook['par']['main_pl'] + '_Saved_data'
    print('Initialized system stored in : ', path)
    if (not path_exist(path)): os_system.makedirs(path)
    datasave_npz(path+'/'+'init_sys',input_nbook)
    return None


def load_nbook(input_nbook, nbook_type):
    input_nbook = dataload_npz(input_nbook['working_path']+'/'+input_nbook['star_name']+'/'+input_nbook['pl_name']+'_Saved_data/init_sys')
    input_nbook['type'] = nbook_type 

    #Retrieving dataset in ANTARESS format
    if nbook_type in ['Processing']:
        input_nbook['settings']['gen_dic']['calc_proc_data']=False

    return input_nbook


def init():
    input_nbook = {
        'settings' : {'gen_dic':{'data_dir_list':{},'type':{}},
                      'mock_dic':{'visit_def':{},'sysvel':{},'intr_prof':{},'flux_cont':{},'set_err':{}},
                      'data_dic':{'DI':{'sysvel':{}},
                                  'Intr':{},'Diff':{}},
                      'glob_fit_dic':{'IntrProp':{},'IntrProf':{},'DiffProf':{}},
                      'plot_dic':{},
                      'detrend_prof_dic':{}
                     },
        #notebook inputs related to system properties
        'system' : {},        
        #notebook inputs related to processing and analysis
        'par' : {'loc_prof_corr':False},      
        #notebook inputs related to the spectral reduction
        'sp_reduc':{},
        #notebook inputs related to the detrending of CCFs
        'DI_trend':{},
        #tracks which fits were performed
        'fits':[],            
        #notebook inputs related to plots
        'plots' : {}}         

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
        input_nbook['settings']['mock_dic']['spots_prop']={inst:{
                                                                vis:{}
                                                                }
                                                           }
        input_nbook['settings']['gen_dic']['transit_sp'] = {}
        input_nbook['settings']['data_dic']['DI']['spots_prop'] = {'achrom':{'LD':['quadratic'],'LD_u1' : [input_nbook['par']['ld_spot_u1']],'LD_u2' : [input_nbook['par']['ld_spot_u2']]}}
        input_nbook['settings']['data_dic']['DI']['transit_prop'] = {'nsub_Dstar':201., 
                                                                     inst:{
                                                                          vis:{'mode':'simu', 'n_oversamp':5.}
                                                                          }
                                                                     }
    for key in ['lat', 'Tc', 'ang', 'fctrst']:
        if key=='Tc': temp=key+'_sp'
        else:temp=key
        input_nbook['settings']['mock_dic']['spots_prop'][inst][vis][temp+'__IS'+inst+'_VS'+vis+'_SP'+input_nbook['par']['spot_name']]=input_nbook['par'][key]
    input_nbook['settings']['gen_dic']['transit_sp'][input_nbook['par']['spot_name']]={inst:[vis]}
    input_nbook['settings']['data_dic']['DI']['spots_prop']['achrom'][input_nbook['par']['spot_name']]=[input_nbook['par']['ang']*np.pi/180.]
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
    
    # #Mock dataset to be generated
    # if mock:
    #     input_nbook['settings']['gen_dic']['mock_data']=True
    #     input_nbook['settings']['gen_dic']['type'][inst] = 'CCF'
    #     if inst not in input_nbook['settings']['mock_dic']['visit_def']:
    #         input_nbook['settings']['mock_dic']['visit_def'][inst]={}
    #     input_nbook['settings']['mock_dic']['visit_def'][inst][vis]={'exp_range':np.array(input_nbook['par']['range']),'nexp':int(input_nbook['par']['nexp'])}
        
    #     dbjd =  (input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][1]-input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][0])/input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['nexp']
    #     n_in_visit = int((input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][1]-input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][0])/dbjd)
    #     bjd_exp_low = input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][0] + dbjd*np.arange(n_in_visit)
    #     bjd_exp_high = bjd_exp_low+dbjd      
    #     bjd_exp_all = 0.5*(bjd_exp_low+bjd_exp_high)
    #     input_nbook['par']['t_BJD'] = {'inst':inst,'vis':vis,'t':bjd_exp_all}
    
    # #Dataset to be processed
    # else:
        
    #     #Mock dataset
    #     if input_nbook['settings']['gen_dic']['mock_data']:
    #         input_nbook['settings']['gen_dic']['calc_proc_data']=False
    #         input_nbook['settings']['gen_dic']['mock_data']=True
    #         if inst not in input_nbook['settings']['mock_dic']['visit_def']:
    #             input_nbook['settings']['mock_dic']['visit_def'][inst]={}
    #         input_nbook['settings']['mock_dic']['visit_def'][inst][vis]=None

    #     #Observed dataset
    #     else:
    #         if inst not in input_nbook['settings']['gen_dic']['data_dir_list']:
    #             input_nbook['settings']['gen_dic']['data_dir_list'][inst]={}
    #         input_nbook['settings']['gen_dic']['data_dir_list'][inst][vis] = input_nbook['par']['data_dir']
            
    #         # For sp_reduc notebook only S2D type is processed
    #         if input_nbook['type'] == 'SP_reduc':input_nbook['settings']['gen_dic']['type'][inst] = 'spec2D'

    #Initializing mock dataset
    if mock:   
        input_nbook['settings']['gen_dic']['mock_data']=True
        input_nbook['settings']['gen_dic']['type'][inst] = 'CCF'
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

        if inst not in input_nbook['settings']['gen_dic']['data_dir_list']:
            input_nbook['settings']['gen_dic']['data_dir_list'][inst]={}
        input_nbook['settings']['gen_dic']['data_dir_list'][inst][vis] = input_nbook['par']['data_dir']

    #Define local profile type for internal use
    if input_nbook['settings']['gen_dic']['type'][inst]=='CCF':input_nbook['par']['prof_type']='CCF'
    else:input_nbook['par']['prof_type']='Spec'
    
    return None

def set_sysvel(input_nbook):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night'] 
    
    #For mock dataset generation
    if input_nbook['type'] == 'mock':
        if inst not in input_nbook['settings']['mock_dic']:input_nbook['settings']['mock_dic']['sysvel'][inst]={}
        input_nbook['settings']['mock_dic']['sysvel'][inst][vis] = input_nbook['par']['gamma']

    #For processing
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
    input_nbook['settings']['data_dic']['Diff']['extract_in'] = False
    return None

def extract_intr(input_nbook):
    input_nbook['settings']['gen_dic']['intr_data']=True
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']
    
    #Ranges for CCFs - to define the intrinsic continuum
    if (input_nbook['settings']['gen_dic']['type'][inst]=='CCF'):     #must not be used with mock generation; find better condition
        input_nbook['settings']['data_dic']['Intr']['cont_range'] = {
        inst:{vis:{0: input_nbook['par']['intr_cont_range']}}
    }

    return None

def conv_CCF(input_nbook,prof_type):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']
    if ('spec' not in input_nbook['settings']['gen_dic']['type'][inst]):stop('ERROR : dataset is already in CCF format')
    input_nbook['par']['prof_type'] = 'CCFfromSpec'
    
    input_nbook['settings']['gen_dic'][prof_type+'_CCF'] = True
    input_nbook['settings']['gen_dic']['calc_'+prof_type+'_CCF'] = input_nbook['par']['calc_CCF']

    #ANTARESS RV grid settings are defined in the solar barycentric rest frame
    #   - for notebook intrinsic spectra, settings are provided relative to the input systemic rv and must be shifted back
    if prof_type=='DI':rv_shift=0
    elif prof_type=='Intr':rv_shift = input_nbook['settings']['data_dic']['DI']['sysvel'][inst][vis]
    input_nbook['settings']['gen_dic'].update({
        'start_RV' : input_nbook['par']['start_RV'] - rv_shift,
        'end_RV'   : input_nbook['par']['end_RV']   - rv_shift,
        'dRV'      : input_nbook['par']['dRV'],
        'CCF_mask' : {inst : input_nbook['working_path'] + '/' +input_nbook['sp_reduc']['mask_path']}
        })
    return None


def loc_prof_corr(input_nbook):
    input_nbook['settings']['gen_dic']['loc_data_corr']=True
    input_nbook['par']['loc_prof_corr'] = True
    return None

def diff_prof_corr(input_nbook):
    input_nbook['settings']['gen_dic']['diff_data_corr']=True
    input_nbook['settings']['gen_dic']['calc_diff_data_corr']=True
    input_nbook['par']['diff_prof_corr'] = True
    return None

'''
Analysis functions
'''


def ana_prof(input_nbook,data_type):
    inst = input_nbook['par']['instrument']
    if ('spec' in input_nbook['settings']['gen_dic']['type'][inst]):
        print('Data in spectral mode: not fit performed')
    else:
        input_nbook['settings']['gen_dic']['fit_'+data_type]=True
    
        #Retrieval mode
        if 'calc_fit' in input_nbook['par']:
            input_nbook['settings']['gen_dic']['calc_fit_'+data_type] = deepcopy(input_nbook['par']['calc_fit'])
            input_nbook['par'].pop('calc_fit')
        
        if ('fit_mode' in input_nbook['par']):
            input_nbook['settings']['data_dic'][data_type]['fit_mode']=deepcopy(input_nbook['par']['fit_mode'])
            input_nbook['par'].pop('fit_mode')
            input_nbook['settings']['data_dic'][data_type]['progress']=False
        else:input_nbook['settings']['data_dic'][data_type]['fit_mode'] = 'chi2'
        if 'run_mode' in input_nbook['par']:
            input_nbook['settings']['data_dic'][data_type]['mcmc_run_mode']=deepcopy(input_nbook['par']['run_mode'])   
            if input_nbook['par']['run_mode']=='reuse':
                input_nbook['settings']['data_dic'][data_type]['save_MCMC_chains']=''
                input_nbook['settings']['data_dic'][data_type]['save_MCMC_corner']=''
            input_nbook['par'].pop('run_mode')        
            
        #Manual priors
        if ('priors' in input_nbook['par']):
            input_nbook['settings']['data_dic'][data_type]['line_fit_priors']=deepcopy(input_nbook['par']['priors'])
            for key in input_nbook['settings']['data_dic'][data_type]['line_fit_priors']:
                input_nbook['settings']['data_dic'][data_type]['line_fit_priors'][key]['mod'] = 'uf'
            input_nbook['par'].pop('priors')   
            
    return None


def ana_jointprop(input_nbook,data_type):
    ana_jointcomm(input_nbook,data_type,'Prop')    
    return None

def ana_jointprof(input_nbook,data_type):
    ana_jointcomm(input_nbook,data_type,'Prof')  
    return None

def ana_jointcomm(input_nbook,data_type,ana_type):
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
        input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['mcmc_run_mode'] = 'reuse'
        input_nbook['par'].pop('calc_fit')
    input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['idx_in_fit'] = {input_nbook['par']['instrument']:{input_nbook['par']['night']:deepcopy(input_nbook['par']['idx_in_fit'])}}  

    #Fitted properties
    if (ana_type=='Prop'):
        input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['mod_prop'] = {
            'rv':{},
            'ctrst':{},
            'FWHM':{}}
    elif (ana_type=='Prof'):
        input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['mod_prop'] = {}        
        
    if ('priors' in input_nbook['par']):input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['priors']={}

    #Guess and prior ranges
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
            prop_name='ctrst_ord'+str(ideg)+'__IS__VS_'
        elif 'FWHM' in prop:
            ideg = int(prop.split('FWHM_')[1]) 
            prop_main='FWHM'
            prop_name='FWHM_ord'+str(ideg)+'__IS__VS_'

        #Spot properties
        elif (('lat' in prop) or ('Tc' in prop) or ('ang' in prop)):
            temp_prop_name,spot_name = prop.split('_')
            if 'Tc' in prop:temp_prop_name+='_sp'
            prop_name = temp_prop_name+'__IS'+input_nbook['par']['instrument']+'_VS'+input_nbook['par']['night']+'_SP'+spot_name
        elif  'fctrst' in prop:
            prop_name = 'fctrst__IS'+input_nbook['par']['instrument']+'_VS'+input_nbook['par']['night']+'_SP'

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
            input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['Opt_Lvl'] = 3
            
    if ('priors' in input_nbook['par']):input_nbook['par'].pop('priors')
    
    #Walkers
    if ('mcmc_set' in input_nbook['par']):
        input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['mcmc_set']=deepcopy(input_nbook['par']['mcmc_set'])
        input_nbook['par'].pop('mcmc_set')  
        
    #Save chains by default
    input_nbook['settings']['glob_fit_dic'][data_type+ana_type]['save_MCMC_chains']='png' 
   
    return None


'''
Plot functions
'''
def plot_system(input_nbook):
    input_nbook['settings']['plot_dic']['system_view'] = 'png' 
    input_nbook['plots']['system_view']={'t_BJD':input_nbook['par']['t_BJD'],'GIF_generation':True}
    return None

def plot_prop(input_nbook,data_type):
    input_nbook['settings']['plot_dic']['prop_'+data_type] = 'png' 

    if input_nbook['type']=='RMR':
        input_nbook['par']['prop'] = ['rv','contrast','FWHM']
        input_nbook['par']['print_disp'] = ['plot']
    elif input_nbook['type']=='mock':
        input_nbook['par']['prop'] = ['rv']
    input_nbook['par']['prop'] = np.array(input_nbook['par']['prop'])

    if 'contrast' in input_nbook['par']['prop']:input_nbook['par']['prop'][input_nbook['par']['prop']=='contrast']= 'ctrst'
    input_nbook['plots']['prop_'+data_type+'_ordin'] = deepcopy(input_nbook['par']['prop'])
    input_nbook['par'].pop('prop')   
    for plot_prop in input_nbook['plots']['prop_'+data_type+'_ordin']:input_nbook['plots']['prop_'+data_type+'_'+plot_prop]={}
    
    if ('print_disp' in input_nbook['par']):  
        for plot_prop in input_nbook['plots']['prop_'+data_type+'_ordin']:
            input_nbook['plots']['prop_'+data_type+'_'+plot_prop]['print_disp']=input_nbook['par']['print_disp']
        input_nbook['par'].pop('print_disp') 
    
    #Set error bars depending on the type of fit
    if input_nbook['settings']['data_dic'][data_type]['fit_mode']=='mcmc':
        for plot_prop in input_nbook['plots']['prop_'+data_type+'_ordin']:
            input_nbook['plots']['prop_'+data_type+'_'+plot_prop]['plot_HDI']=True
            input_nbook['plots']['prop_'+data_type+'_'+plot_prop]['plot_err'] = False 
            
    if (data_type=='Intr'):
        for plot_prop in input_nbook['plots']['prop_'+data_type+'_ordin']:
            input_nbook['plots']['prop_'+data_type+'_'+plot_prop]['plot_disp'] = False         
        
        #Models
        prop_path = input_nbook['working_path']+input_nbook['par']['main_pl']+'_Saved_data/Joined_fits/'
        
        #Plot fit to joint properties if carried out
        if 'IntrProp' in input_nbook['fits']:
            for plot_prop in input_nbook['plots']['prop_'+data_type+'_ordin']:
                input_nbook['plots']['prop_Intr_'+plot_prop].update({
                    'IntrProp_path' : prop_path+'/IntrProp/'+input_nbook['settings']['glob_fit_dic'][data_type+'Prop']['fit_mode']+'/'   ,
                    'theo_HR_prop' : True}) 
            
        #Plot fit to joint profiles if carried out
        if 'IntrProf' in input_nbook['fits']:        
            for plot_prop in input_nbook['plots']['prop_'+data_type+'_ordin']:
                input_nbook['plots']['prop_Intr_'+plot_prop].update({
                    'IntrProf_path' : prop_path+'/IntrProf/'+input_nbook['settings']['glob_fit_dic'][data_type+'Prop']['fit_mode']+'/'   ,
                    'theo_HR_prof' : True}) 

    return None

def plot_prof(input_nbook,data_type):
    input_nbook['settings']['plot_dic'][data_type] = 'png'
    if input_nbook['type']=='RMR':
        input_nbook['par']['fit_type'] = 'indiv'   #overplot fits to individual exposures 
    
    input_nbook['plots'][data_type]={'GIF_generation':True,'shade_cont':True,'plot_line_model':True,'plot_prop':False} 
    if 'x_range' in input_nbook['par']:input_nbook['plots'][data_type]['x_range'] = deepcopy(input_nbook['par']['x_range'])
    if 'y_range' in input_nbook['par']:input_nbook['plots'][data_type]['y_range'] = deepcopy(input_nbook['par']['y_range'])
    if data_type=='Intr_prof':
        input_nbook['plots'][data_type]['norm_prof'] = True
    if 'fit_type' in input_nbook['par']:
        input_nbook['plots'][data_type]['fit_type'] = deepcopy(input_nbook['par']['fit_type'])
        input_nbook['par'].pop('fit_type')  
    return None

def plot_spot(input_nbook):
    input_nbook['plots']['system_view']['plot_spots'] = True
    input_nbook['plots']['system_view']['mock_spot_prop'] = True
    input_nbook['plots']['system_view']['n_spcell'] = 101
    return None

def plot_map(input_nbook,data_type):
    inst = input_nbook['par']['instrument']
    
    #Plot specific order only if spectral data
    if ('spec' in input_nbook['settings']['gen_dic']['type'][inst]):
        input_nbook['plots']['map_'+data_type]['orders_to_plot'] = deepcopy(input_nbook['par']['plot_ord'])
    else:input_nbook['par']['plot_ord']=0

    #Activate plot related to intrinsic CCF model only if model was calculated
    def_map = True
    if data_type in ['Intr_prof_est','Intr_prof_res'] and (not input_nbook['par']['loc_prof_corr']):def_map=False
    if data_type in ['Diff_prof_est','Diff_prof_res'] and (not input_nbook['par']['diff_prof_corr']):def_map=False
    if def_map:
        input_nbook['settings']['plot_dic']['map_'+data_type] = 'png'
        input_nbook['plots']['map_'+data_type] = {}
        input_nbook['plots']['map_'+data_type]['verbose'] = False
        if 'x_range' in input_nbook['par']:
            input_nbook['plots']['map_'+data_type]['x_range'] = deepcopy(input_nbook['par']['x_range'])
        if 'v_range' in input_nbook['par']:
            input_nbook['plots']['map_'+data_type].update({'v_range_all':{inst:{input_nbook['par']['night']:deepcopy(input_nbook['par']['v_range'])}}}) 
            input_nbook['par'].pop('v_range')
        if data_type=='Intr_prof':
            input_nbook['plots']['map_'+data_type]['norm_prof'] = True
            input_nbook['plots']['map_'+data_type]['theoRV_HR'] = True
        elif data_type=='Intr_prof_est':
            input_nbook['plots']['map_'+data_type]['line_model']='rec'
        elif data_type=='Intr_prof_res':
            input_nbook['plots']['map_'+data_type]['cont_only']=False
            input_nbook['plots']['map_'+data_type]['line_model']='rec'
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


'''
Spectral reduction functions
'''
def inst_cal(input_nbook, plot=False):
    '''
    input_nbook, parameters for the calibration will be stored in input_nbook
    plot, set to true to generate plots
    '''
    input_nbook['settings']['gen_dic']['calc_gcal']=input_nbook['sp_reduc']['calc_gcal']
    input_nbook['settings']['gen_dic']['gcal_blaze'] = input_nbook['sp_reduc']['blaze']

    #setting spectral reduction modules after calibration to False, will be acitvated in separate cells when defined
    input_nbook['settings']['gen_dic']['corr_tell'] = False
    input_nbook['settings']['gen_dic']['glob_mast'] = False
    input_nbook['settings']['gen_dic']['corr_Fbal'] = False
    input_nbook['settings']['gen_dic']['corr_FbalOrd'] = False    
    input_nbook['settings']['gen_dic']['corr_cosm'] = False    
    input_nbook['settings']['gen_dic']['calc_FbalOrd'] = False    


    if plot:
        input_nbook['settings']['plot_dic']['gcal'] = 'png'
        input_nbook['settings']['plot_dic']['gcal_ord'] = 'png'
#        input_nbook['settings']['plot_dic']['sdet_ord'] ='png'
    return None

def tell_corr(input_nbook, plot=False):
    '''
    input_nbook: containing parameters used for telluric correction
    plot: set to true to generate plots
    '''
    input_nbook['settings']['gen_dic']['corr_tell']=True 
    input_nbook['settings']['gen_dic']['calc_corr_tell']=input_nbook['sp_reduc']['calc_tell']

    input_nbook['settings']['gen_dic']['tell_species']    =input_nbook['sp_reduc']['tell_species']
    input_nbook['settings']['gen_dic']['tell_thresh_corr']=input_nbook['sp_reduc']['tell_thresh']
    if plot:
        input_nbook['settings']['plot_dic']['tell_CCF'] = 'png'
        input_nbook['settings']['plot_dic']['tell_prop'] = 'png'
    return None

def fbal_corr(input_nbook):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']   

    input_nbook['settings']['gen_dic']['corr_Fbal']=True
    input_nbook['settings']['gen_dic']['calc_corr_Fbal']=input_nbook['sp_reduc']['calc_Fbal']

    input_nbook['settings']['gen_dic']['Fbal_vis']=None
    input_nbook['settings']['gen_dic']['Fbal_range_corr']   ={}
    
    input_nbook['settings']['gen_dic']['Fbal_clip']         =input_nbook['sp_reduc']['sigma_clip']
    input_nbook['settings']['gen_dic']['Fbal_phantom_range']=input_nbook['sp_reduc']['phantom_range']
    input_nbook['settings']['gen_dic']['Fbal_expvar']       =input_nbook['sp_reduc']['unc_scaling']
    input_nbook['settings']['gen_dic']['Fbal_mod']          =input_nbook['sp_reduc']['fit_mode']

    input_nbook['settings']['gen_dic']['Fbal_deg']          ={inst: {vis: input_nbook['sp_reduc']['pol_deg']}}
    input_nbook['settings']['gen_dic']['Fbal_smooth']       ={inst: {vis: input_nbook['sp_reduc']['smooth_fac']}}    

    if ((input_nbook['sp_reduc']['nord'] != None) & (len(input_nbook['sp_reduc']['ord_excl_fit']) > 0)):
        ord_fit = range(input_nbook['sp_reduc']['nord'])
        ord_fit = [order for order in ord_fit if order not in input_nbook['sp_reduc']['ord_excl_fit']]
        input_nbook['settings']['gen_dic']['Fbal_ord_fit']  ={inst:{vis:ord_fit}}
    return None

def plot_fbal_corr(input_nbook):
    input_nbook['settings']['plot_dic']['Fbal_corr'] = 'png'
    if input_nbook['plots']['gap_exp']:
        input_nbook['plots']['Fbal_corr'] = {'gap_exp': 0.1}
    else:
        input_nbook['plots']['Fbal_corr'] = {'gap_exp': 0.0}
    return None

def cosm_corr(input_nbook, plot=False):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']

    input_nbook['settings']['gen_dic']['corr_cosm'] = True
    input_nbook['settings']['gen_dic']['calc_cosm'] = input_nbook['sp_reduc']['calc_cosm']

    input_nbook['settings']['gen_dic']['al_cosm'] = {'mode':input_nbook['sp_reduc']['align_method']}
    input_nbook['settings']['gen_dic']['cosm_ncomp'] = input_nbook['sp_reduc']['ncomp']
    input_nbook['settings']['gen_dic']['cosm_thresh'] = {inst:{vis: input_nbook['sp_reduc']['thresh']}}
    if plot:
        input_nbook['setting']['plot_dic']['cosm_corr']='png'
    return None

def wiggle_corr(input_nbook):
    vis = input_nbook['par']['night']
    input_nbook['settings']['gen_dic']['corr_wig']= input_nbook['sp_reduc']['corr_wig']
    input_nbook['settings']['gen_dic']['calc_wig']= input_nbook['sp_reduc']['calc_wig']

    if input_nbook['sp_reduc']['fit_range']==[[]]:
        input_nbook['settings']['gen_dic']['wig_range_fit'] = []
    else: input_nbook['settings']['gen_dic']['wig_range_fit'] = {vis : input_nbook['sp_reduc']['range_to_fit']}

    input_nbook['settings']['gen_dic']['wig_exp_init'] = {'mode'     :input_nbook['sp_reduc']['screening'],
                                                          'plot_spec':True,
                                                          'plot_hist':True,
                                                          'y_range'  :input_nbook['sp_reduc']['y_range']}

    input_nbook['settings']['gen_dic']['wig_exp_filt'] = {'mode':input_nbook['sp_reduc']['filter'],
                                                          'win' :input_nbook['sp_reduc']['window'],
                                                          'deg' :input_nbook['sp_reduc']['deg'],
                                                          'plot':True}
    return None

def processing_mode(input_nbook):
    input_nbook['settings']['gen_dic']['calc_proc_data'] = input_nbook['sp_reduc']['proc_data']
    return None

def mask_pix(input_nbook):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']
    input_nbook['settings']['gen_dic']['masked_pix'] = {inst:{vis:{'exp_list':[],'ord_list':{}}}}
    if input_nbook['sp_reduc']['order'] != []:
        for i,k in zip(input_nbook['sp_reduc']['order'],input_nbook['sp_reduc']['range']):
            input_nbook['settings']['gen_dic']['wig_exp_filt'][inst][vis]['ord_list'].update({
                i:[k]
                })
    return None

def plot_spec(input_nbook):
    input_nbook['plots']['sp_var'] = 'wav'

    if input_nbook['sp_reduc']['sp_raw']:
        input_nbook['settings']['plot_dic']['sp_raw'] = 'png'
    if input_nbook['sp_reduc']['trans_sp']:
        input_nbook['settings']['plot_dic']['sp_raw'] = 'png'
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

def detrend(input_nbook):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']

    input_nbook['settings']['gen_dic']['detrend_prof'] = input_nbook['par']['use']
    input_nbook['settings']['detrend_prof_dic']['corr_trend'] = input_nbook['par']['use']
    input_nbook['settings']['detrend_prof_dic']['prop'] = {inst:{vis:{}}}

    for (prop,i) in zip(input_nbook['par']['prop'], range(len(input_nbook['par']['prop']))):
        input_nbook['settings']['detrend_prof_dic']['prop'][inst][vis].update({
            prop: {'pol': np.array(input_nbook['par']['coeff'][i])}
            })

    return None

'''
Functions used for trend characterisation and detrending
'''
def fit_range(input_nbook):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']    

    input_nbook['settings']['gen_dic']['fit_DI'] = True
    input_nbook['settings']['gen_dic']['calc_fit_DI'] = True

    input_nbook['settings']['data_dic']['DI'] = {
        'cont_range': {inst: {0: input_nbook['DI_trend']['cont_range']}},
        'fit_range' : {inst: {vis: input_nbook['DI_trend']['fit_range']}}
        }
    return None

def fit_DI(input_nbook, plot=False):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']    

    input_nbook['settings']['data_dic']['mod_prop']={
        'rv'    : {'vary':True, inst:{vis:{'guess':input_nbook['DI_trend']['rv']}}},
        'FWHM'  : {'vary':True, inst:{vis:{'guess':input_nbook['DI_trend']['FWHM']}}},
        'ctrst' : {'vary':True, inst:{vis:{'guess':input_nbook['DI_trend']['ctrst']}}},
    }
    if plot:
        input_nbook['settings']['plot_dic']['porp_DI'] = 'png'
        if input_nbook['DI_trend']['plot_CCF']:
            input_nbook['settings']['plot_dic']['DI_prof'] = 'png'
            input_nbook['settings']['plot_dic']['DI_prof_res'] = 'png'
    return None

def fit_prop(input_nbook):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']    

    if ((inst=='ESPRESSO') & (input_nbook['DI_trend']['x_var']=='snr')):
        input_nbook['DI_trend']['x_var'] = 'snrQ'

    for prop in ['rv', 'rv_res', 'FWHM', 'ctrst']:
        input_nbook['plots']['prop_DI_'+ prop] = {
            'prop_DI_absc': input_nbook['DI_trend']['x_var']
        }
        input_nbook['plots']['prop_DI_'+ prop] = {
            {'deg_prop_fit': {inst: {vis:{ 
            input_nbook['DI_trend']['x_var']:input_nbook['DI_trend']['pol_deg']}
            }}}
        }

    input_nbook['settings']['plot_dic']['porp_DI'] = 'png'
    return None
