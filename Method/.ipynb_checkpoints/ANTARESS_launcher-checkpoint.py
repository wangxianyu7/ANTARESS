#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 22:39:27 2023

@author: V. Bourrier
"""
from ANTARESS_main import ANTARESS_main
from ANTARESS_settings import ANTARESS_settings
from ANTARESS_systems import all_system_params
from utils import stop

def ANTARESS_launcher(input_settings,input_system , user = None):
    
    #Overwrite default system properties
    if len(input_system)>0:
        all_system_params.update(input_system)

    #Retrieve default settings
    plot_dic,gen_dic,data_dic,mock_dic,theo_dic,detrend_prof_dic,glob_fit_dic,corr_spot_dic = ANTARESS_settings(user)
    
    #Overwrite default settings
    if len(input_settings)>0: 
        stop('TODO')

    print('****************************************')
    print('Launching ANTARESS')
    print('     Study of :') 
    for pl_loc in gen_dic['transit_pl'].keys():print('      ',pl_loc)
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

