#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
from ..ANTARESS_general.utils import np_where1D,dataload_npz,check_data
from ..ANTARESS_corrections.ANTARESS_tellurics import corr_tell
from ..ANTARESS_corrections.ANTARESS_flux_balance import def_Mstar,corr_Fbal,corr_Ftemp
from ..ANTARESS_corrections.ANTARESS_peaks import corr_cosm,MAIN_permpeak
from ..ANTARESS_corrections.ANTARESS_interferences import MAIN_corr_wig,corr_fring

def red_sp_data_instru(inst,data_dic,plot_dic,gen_dic,data_prop,coord_dic,system_param):
    r"""**Correction sequence for spectral data.**
    
    Calls the successive modules correcting spectra for environmental/instrumental effects
    All datasets from a given instrument are processed together, as some corrections benefit from combining several visits, and from using the full range of the input data. 
    After the correction sequence, spectra can be trimmed to a chosen set of spectral orders and/or spectral bands if the subsequent analysis focuses on a specific region. 
    
    Args:
        TBD
    
    Returns:
        None
    
    """
    #Information
    #    - plots are deactivated if corrections are not required 
    #    - every time data is corrected in a module the path to the 'processed' data is updated, so that later modules will use the most corrected data
    data_inst=data_dic[inst]

    #Save path to uncorrected data
    for vis in data_inst['visit_list']:data_inst[vis]['uncorr_exp_data_paths']=deepcopy(data_inst[vis]['proc_DI_data_paths'])

    #Correcting for tellurics
    if (gen_dic['corr_tell']):corr_tell(gen_dic,data_inst,inst,data_dic,data_prop,coord_dic,plot_dic)  

    #Definition of a global master for the star
    if (gen_dic['glob_mast']):def_Mstar(gen_dic,data_inst,inst,data_prop,plot_dic,data_dic,coord_dic)

    #Correcting for flux balance
    if gen_dic['corr_Fbal'] or gen_dic['corr_FbalOrd']:corr_Fbal(inst,gen_dic,data_inst,plot_dic,data_prop,data_dic)  

    #Correcting for temporal variations
    if (gen_dic['corr_Ftemp']):corr_Ftemp(inst,gen_dic,data_inst,plot_dic,data_prop,coord_dic,data_dic) 

    #Correcting for cosmics
    if (gen_dic['corr_cosm']):corr_cosm(inst,gen_dic,data_inst,plot_dic,data_dic,coord_dic)

    #Masking of persistent features
    if (gen_dic['mask_permpeak']):MAIN_permpeak(inst,gen_dic,data_inst,plot_dic,data_dic,data_prop)

    #Correcting for ESPRESSO wiggles
    if (gen_dic['corr_wig']):MAIN_corr_wig(inst,gen_dic,data_dic,coord_dic,data_prop,plot_dic,system_param)

    #Correcting for fringing
    if (gen_dic['corr_fring']):corr_fring(inst,gen_dic,data_inst,plot_dic,data_dic) 

    #Reducing spectra to the selected spectral ranges
    if (gen_dic['trim_spec']):lim_sp_range(inst,data_dic,gen_dic,data_prop)

    #Save path to corrected data
    for vis in data_inst['visit_list']:data_inst[vis]['corr_exp_data_paths']=deepcopy(data_inst[vis]['proc_DI_data_paths'])

    return None





def lim_sp_range(inst,data_dic,gen_dic,data_prop):
    r"""**Data trimming.**
    
    Limits input spectra to specific spectral ranges and order

     - orders left empty in all visits are removed
    
    Args:
        TBD
    
    Returns:
        None
    
    """
    print('   > Trimming spectra')    
 
    #Calculating data
    data_inst = data_dic[inst]
    if (gen_dic['calc_trim_spec']):
        print('         Calculating data')    
    
        #Upload latest processed data
        #    - data must be put in global tables to perform a global reduction of ranges and orders
        data_dic_exp={}
        data_dic_com={}
        data_com_inst = dataload_npz(gen_dic['save_data_dir']+'Processed_data/'+inst+'_com') 
        for vis in data_inst['visit_list']:    
            data_vis=data_inst[vis]
            data_dic_com[vis] = dataload_npz(data_vis['proc_com_data_paths'])
            data_dic_exp[vis] = {}            
            for key in ['cen_bins','flux','mean_gdet']:data_dic_exp[vis][key]=np.zeros(data_vis['dim_all'], dtype=float)*np.nan
            data_dic_exp[vis]['cond_def']=np.zeros(data_vis['dim_all'], dtype=bool)
            data_dic_exp[vis]['cov'] = np.zeros(data_vis['dim_sp'], dtype=object)
            data_dic_exp[vis]['edge_bins']=np.zeros(data_vis['dim_sp']+[data_vis['nspec']+1], dtype=float)*np.nan
            if data_vis['tell_sp']:data_dic_exp[vis]['tell']=np.zeros(data_vis['dim_all'], dtype=float)
            for iexp in range(data_vis['n_in_visit']):               
                data_exp = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp))                 
                for key in [key_sub for key_sub in data_dic_exp[vis] if key_sub not in ['mean_gdet','tell']]:data_dic_exp[vis][key][iexp] = data_exp[key]
                if data_vis['tell_sp']:data_dic_exp[vis]['tell'][iexp] = dataload_npz(data_vis['tell_DI_data_paths'][iexp])['tell']             
                data_dic_exp[vis]['mean_gdet'][iexp] = dataload_npz(data_vis['mean_gdet_DI_data_paths'][iexp])['mean_gdet']

        #------------------------------------------

        #Processing each visit
        if len(gen_dic['trim_range'])>0:
            if (data_inst['type']=='spec2D'):cond_ord_kept = np.repeat(False,data_inst['nord'])
            for vis in data_inst['visit_list']:    
                data_vis=data_inst[vis]
    
                #Set bins outside selected ranges to nan in flux tables
                for iexp in range(data_vis['n_in_visit']):
                    low_bins_exp = data_dic_exp[vis]['edge_bins'][iexp,:,0:-1]
                    high_bins_exp = data_dic_exp[vis]['edge_bins'][iexp,:,1::]
                    cond_keep_sp = np.zeros(data_vis['dim_exp'],dtype=bool)   
                    for trim_range_loc in gen_dic['trim_range']:
                        cond_keep_sp |= (low_bins_exp>=trim_range_loc[0]) & (high_bins_exp<=trim_range_loc[1])
                    data_dic_exp[vis]['flux'][iexp,~cond_keep_sp] = np.nan         
                    data_dic_exp[vis]['cond_def'][iexp]&=cond_keep_sp                        
               
                #Limiting all exposures to the smallest common defined range
                #    - the range is delimited by the most extreme bins with flux value defined in at least one order and one exposure
                #    - spectral tables remain continuous
                cond_def_vis = (np.sum(data_dic_exp[vis]['cond_def'],axis=(0,1)) > 0.)         
                idx_def_raw_all = np_where1D(cond_def_vis)
                if len(idx_def_raw_all)<data_vis['nspec']:
                    idx_range_kept = np.arange(idx_def_raw_all[0],idx_def_raw_all[-1]+1)  
                    for key in ['cen_bins','flux','cond_def','mean_gdet']:
                        data_dic_exp[vis][key] = np.take(data_dic_exp[vis][key],idx_range_kept,axis=-1)
                    if data_vis['tell_sp']:data_dic_exp[vis]['tell'] = np.take(data_dic_exp[vis]['tell'],idx_range_kept,axis=-1)
                    data_dic_com[vis]['cen_bins'] = np.take(data_dic_com[vis]['cen_bins'],idx_range_kept,axis=-1)
                    data_dic_exp[vis]['edge_bins'] = np.append(data_dic_exp[vis]['edge_bins'][:,:,idx_range_kept],data_dic_exp[vis]['edge_bins'][:,:,idx_range_kept[-1]+1][:,:,None],axis=2)                 
                    data_dic_com[vis]['edge_bins'] = np.append(data_dic_com[vis]['edge_bins'][:,idx_range_kept],data_dic_com[vis]['edge_bins'][:,idx_range_kept[-1]+1][:,None],axis=1)                 
                    for iexp in range(data_vis['n_in_visit']):          
                        for iord in range(data_inst['nord']):data_dic_exp[vis]['cov'][iexp,iord] = data_dic_exp[vis]['cov'][iexp,iord][:,idx_range_kept]
                    data_vis['nspec'] = len(idx_range_kept)
                    data_vis['dim_all'][2] = data_vis['nspec']
                    data_vis['dim_exp'][1] = data_vis['nspec']            
            
                #Limiting all exposures to the smallest common defined orders
                #    - some of the orders might be entirely empty after spectral ranges exclusion
                #    - an order is kept if it contains at least one defined bin in at least one exposure
                if (data_inst['type']=='spec2D'):cond_ord_kept |= (np.sum(data_dic_exp[vis]['cond_def'],axis=(0,2)) > 0.)
    
            #Limiting inst common table to smallest defined range
            cond_keep_sp = np.zeros(data_inst['dim_exp'],dtype=bool)  
            for trim_range_loc in gen_dic['trim_range']:
                cond_keep_sp |= (data_com_inst['edge_bins'][:,0:-1]>=trim_range_loc[0]) & (data_com_inst['edge_bins'][:,1::]<=trim_range_loc[1]) 
            idx_def_raw_inst = np_where1D(np.sum(cond_keep_sp,axis=(0)) > 0.)
            if len(idx_def_raw_inst)<data_inst['nspec']:
                idx_range_kept = np.arange(idx_def_raw_inst[0],idx_def_raw_inst[-1]+1) 
                data_com_inst['cen_bins'] = np.take(data_com_inst['cen_bins'],idx_range_kept,axis=-1)
                data_com_inst['edge_bins'] = np.append(data_com_inst['edge_bins'][:,idx_range_kept],data_com_inst['edge_bins'][:,idx_range_kept[-1]+1][:,None],axis=1)                 
                data_inst['nspec'] = len(idx_range_kept)               
                data_inst['dim_exp'][1] = data_inst['nspec']   
        elif (data_inst['type']=='spec2D'):
            cond_ord_kept = np.repeat(True,data_inst['nord'])    #condition relative to current tables
        
        #Reduce 2D tables to common defined, and selected, orders over all visits
        #    - an order has been kept if it is defined in at least one visit
        if (data_inst['type']=='spec2D'):
            idx_ord_kept = np_where1D(cond_ord_kept)             #indexes relative to current tables
            if (inst in gen_dic['trim_orders']) and (len(gen_dic['trim_orders'][inst])>0):
                idx_ord_kept = np.intersect1d(idx_ord_kept,gen_dic['trim_orders'][inst])
            if len(idx_ord_kept)<data_inst['nord']:
                cond_orders4ccf = np.zeros(data_inst['nord'],dtype=bool)
                cond_orders4ccf[gen_dic[inst]['orders4ccf']] = True
                data_inst['nord'] = len(idx_ord_kept)  
                data_inst['nord_spec']=deepcopy(data_inst['nord'])     
                gen_dic[inst]['orders4ccf'] = np_where1D(cond_orders4ccf[idx_ord_kept]) #index of orders relative to the new reduced tables    
                data_inst['idx_ord_ref'] = np.array(data_inst['idx_ord_ref'])[idx_ord_kept]
                for key in ['cen_bins','edge_bins']:
                    data_com_inst[key] = np.take(data_com_inst[key],idx_ord_kept,axis=0)    
                data_inst['dim_exp'][0] = data_inst['nord']  
                for vis in data_inst['visit_list']: 
                    data_vis=data_inst[vis] 
                    for key in data_dic_exp[vis]:
                        data_dic_exp[vis][key] = np.take(data_dic_exp[vis][key],idx_ord_kept,axis=1) 
                    for key in ['cen_bins','edge_bins']:
                        data_dic_com[vis][key] = np.take(data_dic_com[vis][key],idx_ord_kept,axis=0)
                    data_vis['dim_all'][1] = data_inst['nord']
                    data_vis['dim_exp'][0] = data_inst['nord']
                    data_vis['dim_sp'][1] = data_inst['nord']     

        #Saving modified data and updating paths
        data_inst['proc_com_data_path'] = gen_dic['save_data_dir']+'Corr_data/Trim/'+inst+'_com' 
        np.savez_compressed(data_inst['proc_com_data_path'],data = data_com_inst,allow_pickle=True)    
        np.savez(gen_dic['save_data_dir']+'Corr_data/Trim/DimTrimmed_'+inst,nord = data_inst['nord'],nspec = data_inst['nspec'],dim_exp = data_inst['dim_exp'],orders4ccf = gen_dic[inst]['orders4ccf'],idx_ord_ref=data_inst['idx_ord_ref'],nord_spec=data_inst['nord_spec'])  
        for vis in data_inst['visit_list']:
            data_vis=data_inst[vis]
            data_vis['proc_DI_data_paths'] = gen_dic['save_data_dir']+'Corr_data/Trim/'+inst+'_'+vis+'_'
            if data_vis['tell_sp']:data_vis['tell_DI_data_paths'] = {}
            data_vis['mean_gdet_DI_data_paths'] = {}
            for iexp in range(data_vis['n_in_visit']): 
                if data_vis['tell_sp']:
                    data_vis['tell_DI_data_paths'][iexp] = gen_dic['save_data_dir']+'Corr_data/Trim/'+inst+'_'+vis+'_tell_'+str(iexp)
                    np.savez_compressed(data_vis['tell_DI_data_paths'][iexp],data={'tell':data_dic_exp[vis]['tell'][iexp]},allow_pickle=True)
                data_vis['mean_gdet_DI_data_paths'][iexp] = gen_dic['save_data_dir']+'Corr_data/Trim/'+inst+'_'+vis+'_mean_gdet_'+str(iexp)
                np.savez_compressed(data_vis['mean_gdet_DI_data_paths'][iexp],data={'mean_gdet':data_dic_exp[vis]['mean_gdet'][iexp]},allow_pickle=True)   
                data_sav_exp = {key:data_dic_exp[vis][key][iexp] for key in data_dic_exp[vis] if key not in ['tell','mean_gdet']}                
                np.savez_compressed(data_vis['proc_DI_data_paths']+str(iexp),data=data_sav_exp,allow_pickle=True)
            data_vis['proc_com_data_paths'] = gen_dic['save_data_dir']+'Corr_data/Trim/'+inst+'_'+vis+'_com'          
            np.savez_compressed(data_vis['proc_com_data_paths'],data = data_dic_com[vis],allow_pickle=True)    
            np.savez(gen_dic['save_data_dir']+'Corr_data/Trim/DimTrimmed_'+inst+'_'+vis,
                     nspec = data_vis['nspec'],dim_exp = data_vis['dim_exp'],dim_sp = data_vis['dim_sp'],dim_all = data_vis['dim_all'])     
    
    #Updating path to processed data and checking it has been calculated
    else:
        data_inst['proc_com_data_path'] = gen_dic['save_data_dir']+'Corr_data/Trim/'+inst+'_com'
        data_load = np.load(gen_dic['save_data_dir']+'Corr_data/Trim/DimTrimmed_'+inst+'.npz')
        data_inst['dim_exp'] = list(data_load['dim_exp'])
        for key in ['nord','nspec','idx_ord_ref','nord_spec']:data_inst[key]=data_load[key]
        gen_dic[inst]['orders4ccf'] = data_load['orders4ccf']                 
        for vis in data_inst['visit_list']: 
            data_vis=data_inst[vis]
            data_vis['proc_com_data_paths'] = gen_dic['save_data_dir']+'Corr_data/Trim/'+inst+'_'+vis+'_com'
            check_data({'path':data_vis['proc_com_data_paths']},vis=vis)
            data_vis['proc_DI_data_paths'] = gen_dic['save_data_dir']+'Corr_data/Trim/'+inst+'_'+vis+'_'
            if data_vis['tell_sp']:data_vis['tell_DI_data_paths']={}
            data_vis['mean_gdet_DI_data_paths'] = {}
            for iexp in range(data_vis['n_in_visit']): 
                if data_vis['tell_sp']:
                    data_vis['tell_DI_data_paths'][iexp] = gen_dic['save_data_dir']+'Corr_data/Trim/'+inst+'_'+vis+'_tell_'+str(iexp)
                data_vis['mean_gdet_DI_data_paths'][iexp] = gen_dic['save_data_dir']+'Corr_data/Trim/'+inst+'_'+vis+'_mean_gdet_'+str(iexp)
            data_load = np.load(gen_dic['save_data_dir']+'Corr_data/Trim/DimTrimmed_'+inst+'_'+vis+'.npz')
            for key in ['dim_all','dim_exp','dim_sp']:data_inst[vis][key] = list(data_load[key])
            data_inst[vis]['nspec']=data_load['nspec']  

    return None










