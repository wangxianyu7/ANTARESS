#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 22:39:27 2023

@author: V. Bourrier
"""
from copy import deepcopy
import numpy as np

'''
Generic functions
'''
def init(nbook_type):
    input_nbook = {
        'settings' : {'gen_dic':{'data_dir_list':{}},
                      'data_dic':{'DI':{'sysvel':{}}},
                      'mock_dic':{'visit_def':{},'sysvel':{},'intr_prof':{},'flux_cont':{},'set_err':{}},
                      'plot_dic':{}
                     },
        'system' : {},    
        'par' : {},
        'plots' : {}}
    
    #Plot path
    if nbook_type=='mock':
        input_nbook['plot_path'] = '/Users/bourrier/Travaux/ANTARESS/Ongoing/Valinor_Plots/'

    return input_nbook

def init_star(input_nbook):
    input_nbook['settings']['gen_dic']['star_name'] = input_nbook['par']['star_name']
    input_nbook['system'][input_nbook['par']['star_name']]={  
            'star':{
                'Rstar':input_nbook['par']['Rs'],
                'veq':input_nbook['par']['vsini'],
                'istar':90, 
                }}
    input_nbook['settings']['data_dic']['DI']['system_prop']={'achrom':{'LD':['quadratic'],'LD_u1' : [input_nbook['par']['ld_u1']],'LD_u2' : [input_nbook['par']['ld_u2']]}}
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
        input_nbook['system'][input_nbook['par']['star_name']][input_nbook['par']['planet_name']]['lambda_proj']=input_nbook['par']['lambda'] 
        input_nbook['system'][input_nbook['par']['star_name']][input_nbook['par']['planet_name']]['aRs']=input_nbook['par']['aRs'] 
        input_nbook['settings']['data_dic']['DI']['system_prop']['achrom'][input_nbook['par']['planet_name']]=[input_nbook['par']['RpRs']]
    
    return None     
    
def add_vis(input_nbook,mock=False):
    inst = input_nbook['par']['instrument']
    vis = input_nbook['par']['night']
    if inst not in input_nbook['settings']['gen_dic']['transit_pl'][input_nbook['par']['main_pl']]:
        input_nbook['settings']['gen_dic']['transit_pl'][input_nbook['par']['main_pl']][inst]=[]
    input_nbook['settings']['gen_dic']['transit_pl'][input_nbook['par']['main_pl']][inst]+=[vis]
    if mock:
        input_nbook['settings']['gen_dic']['mock_data']=True
        if inst not in input_nbook['settings']['mock_dic']['visit_def']:
            input_nbook['settings']['mock_dic']['visit_def'][inst]={}
        input_nbook['settings']['mock_dic']['visit_def'][inst] = { vis:{'exp_range':np.array(input_nbook['par']['range']),'nexp':int(input_nbook['par']['nexp'])}}
        
        dbjd =  (input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][1]-input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][0])/input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['nexp']
        n_in_visit = int((input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][1]-input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][0])/dbjd)
        bjd_exp_low = input_nbook['settings']['mock_dic']['visit_def'][inst][vis]['exp_range'][0] + dbjd*np.arange(n_in_visit)
        bjd_exp_high = bjd_exp_low+dbjd      
        bjd_exp_all = 0.5*(bjd_exp_low+bjd_exp_high)
        input_nbook['par']['t_BJD'] = {'inst':inst,'vis':vis,'t':bjd_exp_all}
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

def ana_prof(input_nbook):
    input_nbook['settings']['gen_dic']['fit_DI']=True
    return None

def align_prof(input_nbook):
    input_nbook['settings']['gen_dic']['align_DI']=True
    return None

def flux_sc(input_nbook,mock=False):
    input_nbook['settings']['gen_dic']['flux_sc']=True
    if mock:
        input_nbook['settings']['data_dic']['DI']['rescale_DI'] = False 
    return None

def extract_intr(input_nbook):
    input_nbook['settings']['gen_dic']['intr_data']=True
    return None




'''
Plot functions
'''
def plot_system(input_nbook):
    input_nbook['settings']['plot_dic']['system_view'] = 'png' 
    input_nbook['plots']['system_view']={'t_BJD':input_nbook['par']['t_BJD'],'GIF_generation':True}
    return None

def plot_prop(input_nbook):
    input_nbook['settings']['plot_dic']['prop_raw'] = 'png' 
    return None

def plot_prof(input_nbook,data_type):
    input_nbook['settings']['plot_dic'][data_type] = 'png'
    input_nbook['plots'][data_type]={'GIF_generation':True,'shade_cont':True,'plot_line_model':True,'plot_prop':False} 
    if 'x_range' in input_nbook['par']:input_nbook['plots'][data_type]['x_range'] = deepcopy(input_nbook['par']['x_range'])
    if 'y_range' in input_nbook['par']:input_nbook['plots'][data_type]['y_range'] = deepcopy(input_nbook['par']['y_range'])
    if data_type=='Intr_prof':input_nbook['plots'][data_type]['norm_prof'] = True
    return None

def plot_map(input_nbook,data_type):
    input_nbook['settings']['plot_dic']['map_'+data_type] = 'png'
    input_nbook['plots']['map_'+data_type]={'v_range_all':{input_nbook['par']['instrument']:{input_nbook['par']['night']:deepcopy(input_nbook['par']['v_range'])}}}  
    if data_type=='Intr_prof':
        input_nbook['plots']['map_'+data_type]['norm_prof'] = True
        input_nbook['plots']['map_'+data_type]['theoRV_HR'] = True
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


