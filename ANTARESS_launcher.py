#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ANTARESS_main import ANTARESS_main
from ANTARESS_settings import ANTARESS_settings
from ANTARESS_systems import all_system_params
from utils import stop
import os as os_system

def ANTARESS_launcher(input_settings = {} ,input_system = {} , user = None):
    r"""**ANTARESS launch routine.**
    
    Runs ANTARESS with default or manual settings.  
    
    Args:
        TBD
    
    Returns:
        None
    
    """ 

    #Overwrite default system properties
    if len(input_system)>0:
        all_system_params.update(input_system)

    #Retrieve default settings
    gen_dic={}
    plot_dic={}
    corr_spot_dic={}
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
    ANTARESS_settings(user,gen_dic,plot_dic,corr_spot_dic,data_dic,mock_dic,theo_dic,glob_fit_dic,detrend_prof_dic)

    #Retrieve user settings
    

    #Moving to ANTARESS directory
    antaress_dir = '/Users/samsonmercier/Desktop/UNIGE/Fall_Semester_2023-2024/antaress'
    gen_dic['save_dir']= '/Users/samsonmercier/Desktop/UNIGE/Fall_Semester_2023-2024/'
    os_system.chdir(antaress_dir)

    #Overwrite default settings
    if len(input_settings)>0: 
        if 'gen_dic' in input_settings:gen_dic.update(input_settings['gen_dic'])
        if 'data_dic' in input_settings:
            if 'DI' in input_settings['data_dic']:data_dic['DI'].update(input_settings['data_dic']['DI'])

    print('****************************************')
    print('Launching ANTARESS')
    print('****************************************')
    print('')
    
    #Run over nominal settings properties
    if len(gen_dic['grid_run'])==0:
        ANTARESS_main(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic, corr_spot_dic,all_system_params[gen_dic['star_name']])
    
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
            ANTARESS_main(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic, corr_spot_dic,all_system_params[gen_dic['star_name']])

    stop('End of workflow')
    return None
