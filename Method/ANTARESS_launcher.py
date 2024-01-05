#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ANTARESS_main import ANTARESS_main
from utils import stop
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
    from ANTARESS_systems import all_system_params

    #Overwrite default system properties
    if ('system' in nbook_dic) and (len(nbook_dic['system'])>0):
        all_system_params.update(nbook_dic['system'])

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
        'IntrProp':{},
        'ResProf':{},
        'IntrProf':{},
        'AtmProp':{},
        'AtmProf':{},
        }  
    
    #Retrieve default settings
    from ANTARESS_settings import ANTARESS_settings
    ANTARESS_settings(gen_dic,plot_dic,corr_spot_dic,data_dic,mock_dic,theo_dic,glob_fit_dic,detrend_prof_dic)
    
    #Overwrite with user settings
    if user!='':
        main_file = importlib.import_module('ANTARESS_settings_'+user)
        main_file.ANTARESS_settings(gen_dic,plot_dic,corr_spot_dic,data_dic,mock_dic,theo_dic,glob_fit_dic,detrend_prof_dic)
    
    #Overwrite with notebook settings
    if ('settings' in nbook_dic) and (len(nbook_dic['settings'])>0):
        if 'gen_dic' in nbook_dic['settings']:gen_dic.update(nbook_dic['settings']['gen_dic'])
        if 'data_dic' in nbook_dic['settings']:
            if 'DI' in nbook_dic['settings']['data_dic']:data_dic['DI'].update(nbook_dic['settings']['data_dic']['DI'])
        if 'mock_dic' in nbook_dic['settings']:mock_dic.update(nbook_dic['settings']['mock_dic'])
        if 'plot_dic' in nbook_dic['settings']:plot_dic.update(nbook_dic['settings']['plot_dic'])
    
    #Moving to ANTARESS directory
    antaress_dir = '/Users/samsonmercier/Desktop/UNIGE/Fall_Semester_2023-2024/antaress'
    gen_dic['save_dir']= '/Users/samsonmercier/Desktop/UNIGE/Fall_Semester_2023-2024/'
    os_system.chdir(antaress_dir)

    
    print('****************************************')
    print('Launching ANTARESS')
    print('****************************************')
    print('')

    #Run over nominal settings properties
    if len(gen_dic['grid_run'])==0:
        ANTARESS_main(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic, corr_spot_dic,all_system_params[gen_dic['star_name']],nbook_dic,user)
    
    #Run over a grid of properties
    else:
        
        #Run the pipeline over individual spectral order
        inst = list(gen_dic['grid_run'].keys())[0]
        for iord in gen_dic['grid_run'][inst]:
            print('--------------------------------------------')
            if inst=='ESPRESSO':
                print('Order :',str(iord),'(slices = ',2*iord,2*iord+1,')')
                gen_dic['orders4ccf'][inst] = [2*iord,2*iord+1]  
            else:
                print('Order :',str(iord))
                gen_dic['orders4ccf'][inst]=[iord] 
            ANTARESS_main(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic, corr_spot_dic,all_system_params[gen_dic['star_name']],nbook_dic,user)

    print('End of workflow')
    return None
