#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os as os_system
import argparse
import json
import logging
from copy import deepcopy
from os import makedirs
from os.path import exists as path_exist
import glob
import shutil
#The following relative imports are necessary to create an executable command
from ..ANTARESS_process.ANTARESS_main import ANTARESS_main,ANTARESS_settings_overwrite
from ..ANTARESS_launch.ANTARESS_gridrun import ANTARESS_gridrun
from ..ANTARESS_general.utils import import_module,stop,dataload_npz,datasave_npz 

def ANTARESS_launcher(sequence = '' , custom_systems = '',custom_settings = '',custom_plot_settings = '',working_path='',nbook_dic = {} , exec_comm = True):
    r"""**ANTARESS launch routine.**
    
    Runs ANTARESS with default or manual settings.  
    
    Args:
        sequence (str): name of custom sequence (default "": default settings are used)        
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
        parser.add_argument("--sequence",      type=str, default='',help = 'Name of custom sequence (default "": default settings are used)')
        parser.add_argument("--custom_systems",      type=str, default='',help = 'Name of custom systems file (default "": default file ANTARESS_systems.py is used)')
        parser.add_argument("--custom_settings",     type=str, default='',help = 'Name of custom settings file (default "": default file ANTARESS_settings.py is used)')
        parser.add_argument("--custom_plot_settings",type=str, default='',help = 'Name of custom plot settings file (default "": default file ANTARESS_plot_settings.py is used)')
        parser.add_argument("--working_path", type=str, default='' ,help = 'Path to user settings files (default "./": user files are retrieved from current directory)')
        parser.add_argument("-d", "--nbook_dic", type=json.loads, default={})
        input_args=parser.parse_args()
        sequence = input_args.sequence
        custom_systems = input_args.custom_systems
        custom_settings = input_args.custom_settings
        custom_plot_settings = input_args.custom_plot_settings
        working_path = input_args.working_path
        nbook_dic = input_args.nbook_dic 

    #Initializes main dictionaries
    gen_dic={'sequence':sequence}
    plot_dic={}
    data_dic={
        'DI':{'fit_prof':{},'mask':{}},
        'Diff':{},
        'PCA':{},
        'Intr':{'fit_prof':{},'mask':{}},
        'Atm':{'fit_prof':{},'mask':{}}}
    mock_dic={}
    theo_dic={}
    detrend_prof_dic={}
    glob_fit_dic={
        'DIProp':{},
        'IntrProp':{},
        'DiffProf':{},
        'IntrProf':{},
        'AtmProp':{},
        'AtmProf':{},
        }  
    input_dics = (data_dic,mock_dic,gen_dic,theo_dic,plot_dic,glob_fit_dic,detrend_prof_dic)
    
    #Retrieve default settings
    from ..ANTARESS_launch.ANTARESS_settings import ANTARESS_settings
    ANTARESS_settings(*input_dics)

    #Overwrite with user settings
    if custom_settings!='':import_module(working_path+custom_settings).ANTARESS_settings(*input_dics)

    #Overwrite with notebook settings
    if ('settings' in nbook_dic) and (len(nbook_dic['settings'])>0):
        ANTARESS_settings_overwrite(*input_dics,nbook_dic)

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
    
    #Sequence
    if len(sequence)>0:
        if sequence=='st_master_tseries':txt_seq = 'computation of stellar master from time-series'
        elif sequence=='night_proc':txt_seq = 'processing of NIGHT data'        
        print('Sequence : '+txt_seq)        
        print('')

    #Moving to code directory
    os_system.chdir(code_dir)
  
    #Run over nominal settings properties
    #    - notebook settings have already been used to overwrite congiguration settings, and are only passed on to overwrite the plot settings if relevant
    if not gen_dic['grid_run']:
        ANTARESS_main(*input_dics,system_params,nbook_dic,custom_plot_settings)
    
    #Run over a grid of properties
    #    - will overwrite default and notebook configuration settings
    else:
        ANTARESS_gridrun(*input_dics,system_params,nbook_dic,custom_plot_settings)

    print('End of workflow')
    return None



def ANTARESS_DACE_launcher(star_name,inst,sub_inst,data_path,working_path,sysvel = 0.,debug_mode = False,del_dir = True):
    r"""**ANTARESS launch routine: master computation**
    
    Runs ANTARESS on a S2D dataset to generate 2D and 1D master spectra with minimal inputs. 
    
    Args:
        star_name (str): name of the star (to be saved in the master outputs, not critical to the computation)
        inst (str): name of the instrument used to acquire the dataset
        sub_inst (str): name of the sub-instrument used to acquire the dataset
        data_path (str): path to the dataset directory
        working_path (str): path to the directory where ANTARESS outputs are stored
        sysvel (float; optional): radial velocity between the stellar and solar system barycenters (default 0 km/s). Masters will be Doppler-shifted by -sysvel with respect to the Solar system barycenter.
    
    Returns:
        None
    
    """

    #Default settings file  
    code_dir = os_system.path.dirname(__file__).split('ANTARESS_launch')[0]
    default_settings_file = code_dir+'ANTARESS_launch/ANTARESS_settings.py'
    with open(default_settings_file,'r') as file:settings_lines_default = file.readlines()

    #-------------

    #Modifying settings file
    #    - defining settings files:
    # + for dataset reduction, 2D processing, and computation of 2D master spectrum
    # + for computation of 1D master spectrum using reduced products    
    settings_lines_reduc2D = deepcopy(settings_lines_default)
    settings_lines_master1D = deepcopy(settings_lines_default)
    for idx_line, line in enumerate(settings_lines_default):
        arr_line = line.split()
        
        #Input data type        
        if ("gen_dic['type']={}" in arr_line):
            settings_lines_reduc2D[idx_line] = '    '+"gen_dic['type'] = {'"+inst+"':'spec2D'}" + '\n'
            settings_lines_master1D[idx_line] = settings_lines_reduc2D[idx_line]

        #Paths to data directory
        if ("gen_dic['data_dir_list']={}" in arr_line):
            settings_lines_reduc2D[idx_line] = '    '+"gen_dic['data_dir_list'] = {'"+inst+"':{'all':'"+data_path+"'}}" + '\n'
            settings_lines_master1D[idx_line] = settings_lines_reduc2D[idx_line]
            
        #Systemic velocity 
        if ("data_dic['DI']['sysvel']={}" in arr_line):
            settings_lines_reduc2D[idx_line] = '    '+"data_dic['DI']['sysvel'] = {'"+inst+"':{'all':"+str(sysvel)+"}}" + '\n'
            settings_lines_master1D[idx_line] = settings_lines_reduc2D[idx_line]
            
        #Deactive 2D/1D conversion and plots for 2D master computation
        if ("gen_dic['spec_1D_DI']=True" in arr_line):settings_lines_reduc2D[idx_line] = '        '+"gen_dic['spec_1D_DI'] = False" + '\n'  
        if ("plot_dic['DIbin']='pdf'" in arr_line):   settings_lines_reduc2D[idx_line] = '        '+"plot_dic['DIbin'] =''" + '\n' 
        
        #Deactivation of modules for 1D master computation
        #    - to avoid reprocessing the 2D data
        if ("gen_dic['calc_proc_data']=True" in arr_line):  settings_lines_master1D[idx_line] = '    '+"gen_dic['calc_proc_data'] = False" + '\n'    
        if ("gen_dic['calc_gcal']=True" in arr_line):       settings_lines_master1D[idx_line] = '    '+"gen_dic['calc_gcal'] = False" + '\n'            
        if ("gen_dic['calc_corr_tell']=True" in arr_line):  settings_lines_master1D[idx_line] = '    '+"gen_dic['calc_corr_tell'] = False" + '\n'     
        if ("gen_dic['calc_glob_mast']=True" in arr_line):  settings_lines_master1D[idx_line] = '    '+"gen_dic['calc_glob_mast'] = False" + '\n'     
        if ("gen_dic['calc_corr_Fbal']=True" in arr_line):  settings_lines_master1D[idx_line] = '    '+"gen_dic['calc_corr_Fbal'] = False" + '\n'     
        if ("gen_dic['calc_cosm']=True" in arr_line):       settings_lines_master1D[idx_line] = '    '+"gen_dic['calc_cosm'] = False" + '\n'     
        if ("gen_dic['calc_align_DI']=True" in arr_line):   settings_lines_master1D[idx_line] = '    '+"gen_dic['calc_align_DI'] = False" + '\n'  
        if ("gen_dic['calc_flux_sc']=True" in arr_line):    settings_lines_master1D[idx_line] = '    '+"gen_dic['calc_flux_sc'] = False" + '\n'              
        if ("gen_dic['calc_DImast']=True" in arr_line):     settings_lines_master1D[idx_line] = '    '+"gen_dic['calc_DImast'] = False" + '\n'         
        
        #Test settings
        if debug_mode:               
            if ("gen_dic['calc_proc_data']=True" in arr_line):  settings_lines_reduc2D[idx_line] = '    '+"gen_dic['calc_proc_data'] = False" + '\n'    
            # if ("gen_dic['del_orders']={}" in arr_line):        settings_lines_reduc2D[idx_line] = '    '+"gen_dic['del_orders']={'ESPRESSO':np.append(np.arange(0,130),np.arange(160,170))}" + '\n'    
            if ("gen_dic['calc_gcal']=True" in arr_line):       settings_lines_reduc2D[idx_line] = '    '+"gen_dic['calc_gcal'] = False" + '\n'            
            if ("gen_dic['calc_corr_tell']=True" in arr_line):  settings_lines_reduc2D[idx_line] = '    '+"gen_dic['calc_corr_tell'] = False" + '\n'     
            if ("gen_dic['calc_glob_mast']=True" in arr_line):  settings_lines_reduc2D[idx_line] = '    '+"gen_dic['calc_glob_mast'] = False" + '\n'     
            if ("gen_dic['calc_corr_Fbal']=True" in arr_line):  settings_lines_reduc2D[idx_line] = '    '+"gen_dic['calc_corr_Fbal'] = False" + '\n'     
            if ("gen_dic['calc_cosm']=True" in arr_line):       settings_lines_reduc2D[idx_line] = '    '+"gen_dic['calc_cosm'] = False" + '\n'      
            if ("gen_dic['calc_align_DI']=True" in arr_line):   settings_lines_reduc2D[idx_line] = '    '+"gen_dic['calc_align_DI'] = False" + '\n'  
            if ("gen_dic['calc_flux_sc']=True" in arr_line):    settings_lines_reduc2D[idx_line] = '    '+"gen_dic['calc_flux_sc'] = False" + '\n'              
            if ("gen_dic['calc_DImast']=True" in arr_line):     settings_lines_reduc2D[idx_line] = '    '+"gen_dic['calc_DImast'] = False" + '\n'           
            if ("gen_dic['calc_DIbin']=True" in arr_line):      settings_lines_reduc2D[idx_line] = '    '+"gen_dic['calc_DIbin'] = False" + '\n'        
            if ("gen_dic['calc_spec_1D_DI']=True" in arr_line): settings_lines_master1D[idx_line] = '        '+"gen_dic['calc_spec_1D_DI'] = False" + '\n' 
            if ("gen_dic['calc_DIbin']=True" in arr_line):      settings_lines_master1D[idx_line] = '    '+"gen_dic['calc_DIbin'] = False" + '\n'               



    #-------------
    #Reduction and 2D master spectrum
    
    #Saving modified settings files
    with open(working_path + "ANTARESS_settings_st_master_tseries_reduc2D.py", 'w') as file:
        file.writelines(settings_lines_reduc2D)

    #Calling ANTARESS
    ANTARESS_launcher(sequence = 'st_master_tseries' , working_path = working_path , custom_settings = 'ANTARESS_settings_st_master_tseries_reduc2D.py' ,exec_comm=False)

    #-------------
    #1D master spectrum
    
    #Saving modified settings files
    with open(working_path + "ANTARESS_settings_st_master_tseries_master1D.py", 'w') as file:
        file.writelines(settings_lines_master1D)

    #Calling ANTARESS
    ANTARESS_launcher(sequence = 'st_master_tseries' , working_path = working_path , custom_settings = 'ANTARESS_settings_st_master_tseries_master1D.py' ,exec_comm=False)

    #-------------
    #Retrieve and store relevant files
    
    #Create storage directory
    store_dir = working_path+'/'+star_name+'/'+sub_inst+'/'
    if (not path_exist(store_dir)):makedirs(store_dir) 
    
    #Open master files
    master2D = dataload_npz(glob.glob(working_path+'/Star_tseries/Saved_data/DIbin_data/ESPRESSO_all_spec2D_time0.npz')[0].split('.npz')[0])    
    master1D = dataload_npz(glob.glob(working_path+'/Star_tseries/Saved_data/DIbin_data/ESPRESSO_all_spec1D_time0.npz')[0].split('.npz')[0])

    #Store complementary information
    master2D['star_name'] = star_name
    master1D['star_name'] = star_name
    
    #Save masters in storage directory
    datasave_npz(store_dir+'Master2D',master2D)  
    datasave_npz(store_dir+'Master1D',master1D)
    
    #Move plots
    os_system.rename(glob.glob(working_path+'/Star_tseries/Plots/Binned_DI_data/ESPRESSO_all_Indiv/Data/Spec_time/idx0_out0_time*')[0],store_dir+'Master1D_plot.pdf')

    #Delete ANTARESS outputs
    if del_dir:shutil.rmtree(working_path+'/Star_tseries/', ignore_errors=True)
    
    return None
