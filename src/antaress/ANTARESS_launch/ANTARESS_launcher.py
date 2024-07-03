#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os as os_system
import argparse
import json
import logging
#The following relative imports are necessary to create an executable command
from ..ANTARESS_process.ANTARESS_main import ANTARESS_main,ANTARESS_settings_overwrite
from ..ANTARESS_launch.ANTARESS_gridrun import ANTARESS_gridrun
from ..ANTARESS_general.utils import import_module
 

def ANTARESS_launcher(custom_systems = '',custom_settings = '',custom_plot_settings = '',working_path='',nbook_dic = {} , exec_comm = True):
    r"""**ANTARESS launch routine.**
    
    Runs ANTARESS with default or manual settings.  
    
    Args:
        custom_systems (str): name of custom systems file (default "": ANTARESS_systems.py file is used)
        custom_settings (str): name of custom settings file (default "": ANTARESS_settings.py file is used)
        custom_plot_settings (str): name of custom plot settings file (default "": ANTARESS_plot_settings.py file is used)
        working_path (str): path to the working directory, in which the workflow outputs will be stored (default "": current directory is used)
                            if custom files are used, they should be placed in the working directory 
    
    Returns:
        None
    
    """ 
    
    #Suppress log messages from the fontTools package
    fontTools_logger = logging.getLogger('fontTools')
    fontTools_logger.addHandler(logging.NullHandler())
    
    #Read executable arguments
    #    - will be used when ANTARESS is called as an executable through terminal
    if exec_comm:
        parser=argparse.ArgumentParser(prog = "antaress",description='Launch ANTARESS workflow')
        parser.add_argument("--custom_systems",      type=str, default='',help = 'Name of custom systems file (default "": default file ANTARESS_systems.py is used)')
        parser.add_argument("--custom_settings",     type=str, default='',help = 'Name of custom settings file (default "": default file ANTARESS_settings.py is used)')
        parser.add_argument("--custom_plot_settings",type=str, default='',help = 'Name of custom plot settings file (default "": default file ANTARESS_plot_settings.py is used)')
        parser.add_argument("--working_path", type=str, default='' ,help = 'Path to user settings files (default "./": user files are retrieved from current directory)')
        parser.add_argument("-d", "--nbook_dic", type=json.loads, default={})
        input_args=parser.parse_args()
        custom_systems = input_args.custom_systems
        custom_settings = input_args.custom_settings
        custom_plot_settings = input_args.custom_plot_settings
        working_path = input_args.working_path
        nbook_dic = input_args.nbook_dic 

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
    from ..ANTARESS_launch.ANTARESS_settings import ANTARESS_settings
    ANTARESS_settings(gen_dic,plot_dic,corr_spot_dic,data_dic,mock_dic,theo_dic,glob_fit_dic,detrend_prof_dic)
    
    #Overwrite with user settings
    if custom_settings!='':import_module(working_path+custom_settings).ANTARESS_settings(gen_dic,plot_dic,corr_spot_dic,data_dic,mock_dic,theo_dic,glob_fit_dic,detrend_prof_dic)

    #Overwrite with notebook settings
    if ('settings' in nbook_dic) and (len(nbook_dic['settings'])>0):
        ANTARESS_settings_overwrite(gen_dic,plot_dic,corr_spot_dic,data_dic,mock_dic,theo_dic,glob_fit_dic,detrend_prof_dic,nbook_dic)

    #----------------------------------------------------------------------------------------------------    

    #Working directory
    if working_path=='':gen_dic['save_dir'] = os_system.getcwd()+'/'
    else:gen_dic['save_dir']= working_path
    
    #Code directory     
    code_dir = os_system.path.dirname(__file__).split('ANTARESS_launch')[0]

    #Retrieve default or user-defined system properties
    if custom_systems!='':systems_file = import_module(working_path+custom_systems)
    else:systems_file = import_module(code_dir+'ANTARESS_launch/ANTARESS_systems.py')
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
    os_system.chdir(code_dir)
  
    #Run over nominal settings properties
    #    - notebook settings have already been used to overwrite congiguration settings, and are only passed on to overwrite the plot settings if relevant
    if not gen_dic['grid_run']:
        ANTARESS_main(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic, corr_spot_dic,system_params,nbook_dic,custom_plot_settings)
    
    #Run over a grid of properties
    #    - will overwrite default and notebook configuration settings
    else:
        ANTARESS_gridrun(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic, corr_spot_dic,system_params,nbook_dic,custom_plot_settings)

    print('End of workflow')
    return None







