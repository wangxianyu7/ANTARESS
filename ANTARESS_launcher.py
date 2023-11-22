#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ANTARESS_main import ANTARESS_main
from ANTARESS_settings import ANTARESS_settings
from ANTARESS_systems import all_system_params
from utils import stop
import os as os_system

def ANTARESS_launcher(input_dic = {} , user = None):
    r"""**ANTARESS launch routine.**
    
    Runs ANTARESS with default or manual settings.  
    
    Args:
        TBD
    
    Returns:
        None
    
    """ 

    #Overwrite default system properties
    if ('system' in input_dic) and (len(input_dic['system'])>0):
        all_system_params.update(input_dic['system'])

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
    antaress_dir = os_system.path.dirname(__file__).split('Method')[0]
    gen_dic['save_dir']= antaress_dir+'En_cours/'  
    os_system.chdir(antaress_dir+'Method/')

    #Overwrite default settings
    if ('settings' in input_dic) and (len(input_dic['settings'])>0):
        if 'gen_dic' in input_dic['settings']:gen_dic.update(input_dic['settings']['gen_dic'])
        if 'data_dic' in input_dic['settings']:
            if 'DI' in input_dic['settings']['data_dic']:data_dic['DI'].update(input_dic['settings']['data_dic']['DI'])
        if 'mock_dic' in input_dic['settings']:mock_dic.update(input_dic['settings']['mock_dic'])
        if 'plot_dic' in input_dic['settings']:plot_dic.update(input_dic['settings']['plot_dic'])
    
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

    if len(input_dic)==0:stop('End of workflow')
    return None
