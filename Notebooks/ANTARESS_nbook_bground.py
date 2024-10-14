#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np
import os as os_system

'''
Generic functions
'''
def init(nbook_type):
    input_nbook = {
        'type':nbook_type,
        'settings' : {'gen_dic':{'data_dir_list':{}},
                      'mock_dic':{'visit_def':{},'sysvel':{},'intr_prof':{},'flux_cont':{},'set_err':{}},
                      'data_dic':{'DI':{'sysvel':{}},
                                  'Intr':{},'Res':{}},
                      'glob_fit_dic':{'IntrProp':{},'IntrProf':{},'ResProf':{}},
                      'plot_dic':{}
                     },
        #notebook inputs related to system properties
        'system' : {},        
        #notebook inputs related to processing and analysis
        'par' : {'loc_prof_corr':False},          
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
        input_nbook['settings']['gen_dic']['transit_pl']={input_nbook['par']['main_pl']:{}}
        input_nbook['system'][input_nbook['par']['star_name']][input_nbook['par']['planet_name']]['inclination']=input_nbook['par']['incl'] 
        if 'lambda' not in input_nbook['par']:lambda_pl=0.
        else:lambda_pl = input_nbook['par']['lambda']        
        input_nbook['system'][input_nbook['par']['star_name']][input_nbook['par']['planet_name']]['lambda_proj']=lambda_pl
        input_nbook['system'][input_nbook['par']['star_name']][input_nbook['par']['planet_name']]['aRs']=input_nbook['par']['aRs'] 
        input_nbook['settings']['data_dic']['DI']['system_prop']['achrom'][input_nbook['par']['planet_name']]=[input_nbook['par']['RpRs']]
    
        #Paths
        input_nbook['plot_path'] = input_nbook['working_path']+input_nbook['par']['star_name']+'/'+input_nbook['par']['main_pl']+'_Plots/'

    return None     
    
def add_vis(input_nbook,mock=False):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']
    if inst not in input_nbook['settings']['gen_dic']['transit_pl'][input_nbook['par']['main_pl']]:
        input_nbook['settings']['gen_dic']['transit_pl'][input_nbook['par']['main_pl']][inst]=[]
    input_nbook['settings']['gen_dic']['transit_pl'][input_nbook['par']['main_pl']][inst]+=[vis]
    
    #Generating mock dataset
    if mock:
        input_nbook['settings']['gen_dic']['mock_data']=True
        if inst not in input_nbook['settings']['mock_dic']['visit_def']:
            input_nbook['settings']['mock_dic']['visit_def'][inst]={}
        input_nbook['settings']['mock_dic']['visit_def'][inst][vis]={'exp_range':np.array(input_nbook['par']['range']),'nexp':int(input_nbook['par']['nexp'])}
        
        dbjd =  (input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][1]-input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][0])/input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['nexp']
        n_in_visit = int((input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][1]-input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][0])/dbjd)
        bjd_exp_low = input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][0] + dbjd*np.arange(n_in_visit)
        bjd_exp_high = bjd_exp_low+dbjd      
        bjd_exp_all = 0.5*(bjd_exp_low+bjd_exp_high)
        input_nbook['par']['t_BJD'] = {'inst':inst,'vis':vis,'t':bjd_exp_all}
    
    #Processing observed dataset
    else:
        
        #Mock dataset
        if input_nbook['par']['data_dir']=='mock':
            input_nbook['settings']['gen_dic']['calc_proc_data']=False
            input_nbook['settings']['gen_dic']['mock_data']=True
            if inst not in input_nbook['settings']['mock_dic']['visit_def']:
                input_nbook['settings']['mock_dic']['visit_def'][inst]={}
            input_nbook['settings']['mock_dic']['visit_def'][inst][vis]=None

        #Observed dataset
        else:
            if inst not in input_nbook['settings']['gen_dic']['data_dir_list']:
                input_nbook['settings']['gen_dic']['data_dir_list'][inst]={}
            input_nbook['settings']['gen_dic']['data_dir_list'][inst][vis] = input_nbook['par']['data_dir']

    return None

def set_sysvel(input_nbook,mock=False):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']    
    if mock:
        if inst not in input_nbook['settings']['mock_dic']:input_nbook['settings']['mock_dic']['sysvel'][inst]={}
        input_nbook['settings']['mock_dic']['sysvel'][inst][vis] = input_nbook['par']['gamma']
    else:
        if inst not in input_nbook['settings']['data_dic']['DI']:input_nbook['settings']['data_dic']['DI']['sysvel'][inst]={}
        input_nbook['settings']['data_dic']['DI']['sysvel'][inst][vis] = input_nbook['par']['gamma']
    return None

def ana_prof(input_nbook,data_type):
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

def align_prof(input_nbook):
    input_nbook['settings']['gen_dic']['align_DI']=True
    return None

def flux_sc(input_nbook,mock=False):
    input_nbook['settings']['gen_dic']['flux_sc']=True
    if mock:
        input_nbook['settings']['data_dic']['DI']['rescale_DI'] = False 
    return None

def DImast_weight(input_nbook):
    input_nbook['settings']['gen_dic']['DImast_weight']=True
    return None

def extract_res(input_nbook):
    input_nbook['settings']['gen_dic']['res_data']=True
    input_nbook['settings']['data_dic']['Res']['extract_in'] = False
    return None

def extract_intr(input_nbook):
    input_nbook['settings']['gen_dic']['intr_data']=True
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
            
        if data_type == 'Res':
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

    #Activate plot related to intrinsic CCF model only if model was calculated
    def_map = True
    if data_type in ['Intr_prof_est','Intr_prof_res'] and (not input_nbook['par']['loc_prof_corr']):def_map=False
    if data_type in ['Res_prof_est','Res_prof_res'] and (not input_nbook['par']['diff_prof_corr']):def_map=False
    if def_map:
        input_nbook['settings']['plot_dic']['map_'+data_type] = 'png'
        input_nbook['plots']['map_'+data_type] = {}
        input_nbook['plots']['map_'+data_type]['verbose'] = False
        if 'v_range' in input_nbook['par']:
            input_nbook['plots']['map_'+data_type].update({'v_range_all':{input_nbook['par']['instrument']:{input_nbook['par']['night']:deepcopy(input_nbook['par']['v_range'])}}}) 
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
        input_nbook['settings']['mock_dic']['intr_prof'][inst] = {'mode':'ana','coord_line':'mu','func_prof_name': 'gauss','line_trans':None,'mod_prop':{},'pol_mode' : 'modul'} 
    input_nbook['settings']['mock_dic']['intr_prof'][inst]['mod_prop']['ctrst_ord0__IS'+inst+'_VS'+vis] = input_nbook['par']['contrast'] 
    input_nbook['settings']['mock_dic']['intr_prof'][inst]['mod_prop']['FWHM_ord0__IS'+inst+'_VS'+vis]  = input_nbook['par']['FWHM']   
    if inst not in input_nbook['settings']['mock_dic']['flux_cont']:input_nbook['settings']['mock_dic']['flux_cont'][inst] = {}
    input_nbook['settings']['mock_dic']['flux_cont'][inst][vis]  = input_nbook['par']['flux']    
    input_nbook['settings']['mock_dic']['set_err'][inst]  = input_nbook['par']['noise']    
    return None


