#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ANTARESS_process.ANTARESS_main import ANTARESS_main,ANTARESS_settings_overwrite
from ANTARESS_launch.ANTARESS_gridrun import ANTARESS_gridrun
import importlib
import os as os_system

def ANTARESS_launcher(nbook_dic = {} , user = ''):
    r"""**ANTARESS launch routine.**
    
    Runs ANTARESS with default or manual settings.  
    
    Args:
        TBD
    
    Returns:
        None
    
    """ 

    #Initializes main dictionaries
    gen_dic={}
    plot_dic={}
    corr_spot_dic={}
    data_dic={
        'DI':{'fit_prof':{},'mask':{}},
        'Res':{},
        'PCA':{},
        'Intr':{'fit_prof':{},'mask':{}},
        'Atm':{'fit_prof':{},'mask':{}}}
    mock_dic={}
    theo_dic={}
    detrend_prof_dic={}
    glob_fit_dic={
        'DIProp':{},
        'IntrProp':{},
        'ResProf':{},
        'IntrProf':{},
        'AtmProp':{},
        'AtmProf':{},
        }  
    
    #Retrieve default settings
    from ANTARESS_launch.ANTARESS_settings import ANTARESS_settings
    ANTARESS_settings(gen_dic,plot_dic,corr_spot_dic,data_dic,mock_dic,theo_dic,glob_fit_dic,detrend_prof_dic)
    
    #Overwrite with user settings
    if user!='':
        main_file = importlib.import_module('ANTARESS_launch.ANTARESS_settings_'+user)
        main_file.ANTARESS_settings(gen_dic,plot_dic,corr_spot_dic,data_dic,mock_dic,theo_dic,glob_fit_dic,detrend_prof_dic)

    #Overwrite with notebook settings
    if ('settings' in nbook_dic) and (len(nbook_dic['settings'])>0):
        ANTARESS_settings_overwrite(gen_dic,plot_dic,corr_spot_dic,data_dic,mock_dic,theo_dic,glob_fit_dic,detrend_prof_dic,nbook_dic)

    #----------------------------------------------------------------------------------------------------    
    
    #Retrieve default or user-defined system properties
    if user!='':systems_file = importlib.import_module('ANTARESS_launch.ANTARESS_systems_'+user)
    else:systems_file = importlib.import_module('ANTARESS_launch.ANTARESS_systems')
    all_system_params = systems_file.get_system_params()

    #Overwrite with notebook settings
    if ('system' in nbook_dic) and (len(nbook_dic['system'])>0):
        all_system_params.update(nbook_dic['system'])
        
    #Retrieve chosen system
    system_params = all_system_params[gen_dic['star_name']]

    print('****************************************')
    print('Launching ANTARESS')
    print('****************************************')
    print('')
    
    #Moving to code directory
    code_dir = os_system.path.dirname(__file__).split('Method')[0]
    gen_dic['save_dir']= code_dir+'Ongoing/'  
    os_system.chdir(code_dir+'Method/')
  
    #Run over nominal settings properties
    #    - notebook settings have already been used to overwrite congiguration settings, and are only passed on to overwrite the plot settings if relevant
    if not gen_dic['grid_run']:
        ANTARESS_main(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic, corr_spot_dic,system_params,nbook_dic,user)
    
    #Run over a grid of properties
    #    - will overwrite default and notebook configuration settings
    else:
        ANTARESS_gridrun(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic, corr_spot_dic,system_params,nbook_dic,user)

    print('End of workflow')
    return None







