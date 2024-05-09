#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from ..ANTARESS_process.ANTARESS_main import ANTARESS_main,ANTARESS_settings_overwrite

def ANTARESS_gridrun(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic, corr_spot_dic,system_params,nbook_dic,input_path,custom_plot_settings):
    r"""**ANTARESS grid run launch.**
    
    Runs ANTARESS over a grid of values for a given set of fields in the configuration file.  
    
    Args:
        TBD
    
    Returns:
        None
    
    """ 
    grid_prop = {}

    #Grid over individual spectral orders
    iord_run = np.delete(np.arange(85),[59,68,74,75,76,77,78,82])    #empty CCF orders (tellurics)
    for iord in iord_run: grid_prop['orders4ccf'] += [2*iord,2*iord+1]   
   
    # inst = 'HARPN'
    # iord_run = np.delete(np.arange(69),[54,63,68])     #Old masks 

    # inst = 'HARPN'
    # iord_run = np.delete(np.arange(69),[53,54,60,63,64])     #New masks 

    # for iord in iord_run: grid_prop['orders4ccf'] += [iord] 

    #Instrument
    inst = 'ESPRESSO'


    
    
    #------------------------------------------------------------------------------------------------

    #Grid preparation
    par_names = []
    grid_par = []
    for prop in grid_prop:
        par_names+=[prop]
        grid_par+=[list(grid_prop[prop])]
    grid_par = np.array(grid_par).T

    #Looping on grid 
    gridstep_dic = {
        'orders4ccf':{},    
    }

    #Run the code over grid
    for par_set in grid_par:    
        for par_name,par_val in zip(par_names,par_set):
            root_name = par_name

            #Spectral orders
            if root_name == 'orders4ccf':
                gridstep_dic['orders4ccf'][inst] = par_val 
    
            #----------------------------------------------------------------------------------------------------------------------------------   
            
            #Overwrite current settings        
            ANTARESS_settings_overwrite(gen_dic,plot_dic,corr_spot_dic,data_dic,mock_dic,theo_dic,glob_fit_dic,detrend_prof_dic,gridstep_dic)
            
            #Run code for current parameter set        
            ANTARESS_main(data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic, corr_spot_dic,system_params,nbook_dic,custom_plot_settings)


    print('End of grid')
    return None
