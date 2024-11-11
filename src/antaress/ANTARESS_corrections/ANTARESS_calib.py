#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import lmfit
from lmfit import Parameters
from scipy.interpolate import interp1d
from scipy import stats
from copy import deepcopy
from pathos.multiprocessing import Pool
from numpy.polynomial import Polynomial
import bindensity as bind
import os as os_system
from ..ANTARESS_conversions.ANTARESS_binning import sub_calc_bins,sub_def_bins
from ..ANTARESS_analysis.ANTARESS_inst_resp import return_resolv
from ..ANTARESS_general.utils import dataload_npz,np_where1D,stop,init_parallel_func,check_data,def_edge_tab,gen_specdopshift,datasave_npz
from ..ANTARESS_general.minim_routines import call_lmfit

def calc_gcal(gen_dic,data_dic,inst,plot_dic,coord_dic,data_prop):
    r"""**Main instrumental calibration routine.**
    
    Estimates instrumental calibration (see `weights_bin_prof()`)
    
    Calibration profiles are used in weight profiles and to scale back profiles approximatively to their raw photoelectron counts.
    If blaze files are not available, calibration profiles are estimated from the error and flux tables and we neglect detector noise. This tends to overestimate the true calibration in regions of low count levels.
        
    We fit a model to the calibration profile (measured or estimated) for each exposure, so that it can be extrapolated over a larger common range for all visits, or used to complete the measured profile over undefined pixels.
    
    Weights use the specific calibration profile defined for each exposure.
    Weights are used for temporal binning, and are relevant only if the weight of a given pixel change over time. The calibration is thus not defined for CCFs.
    
    Scaling uses a single calibration profile, constant in time and common to all processed exposures of an instrument, so that the relative color balance between spectra is not modified when converting them back to counts. 
    For the same reason the calibration must be applied uniformely (ie, in the same rest frame) to spectra in different exposures, and their master, so that it does not affect their combinations - even if the original calibration to flux units is applied in the detector rest frame.
    The scaling is applied in particular during CCF calculations, to avoid artifically increasing errors when combining spectral regions with different SNRs.
    Spectra must however be kept to their extracted flux units througout ANTARESS processing, as the stellar spectrum shifts over time in the instrument rest frame.
    The same region of the spectrum thus sees different instrumental responses over time and is measured with different flux levels.
    Shifted stellar spectra with the same profile would thus have a different color balance after being converted in raw count units, preventing in particular the correct calculation of binned spectra. 
    We calculate the median of the extrapolated models or measured completed profiles, interpolate it as a function, and use it to define the common scaling calibration profile over the specific table of each exposure.
    Spectra are typically provided in the solar barycentric rest frame. Calibration profiles measured or estimated in this frame are thus shifted to the Earth barycentric frame (ie, the detector frame) before being averaged
    
    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    print('   > Estimating instrumental calibration')
    data_inst = data_dic[inst] 

    #Calculating data
    if gen_dic['calc_gcal']:
        print('         Calculating data')        
        cal_inputs_dic = {} 
        min_edge_ord_all = np.repeat(1e100,data_dic[inst]['nord'])
        max_edge_ord_all = np.repeat(-1e100,data_dic[inst]['nord'])  
        min_BERV = 1e100
        max_BERV = -1e100
        iexp_glob_groups_vis = {}
        gcal_exp_all = {}
        plot_save = (plot_dic['gcal_all']!='') or (plot_dic['gcal_ord']!='')
        
        #Processing each visit
        for vis in data_dic[inst]['visit_list']:
            print('           Processing '+vis) 
            data_vis=data_dic[inst][vis]
            data_com_vis = dataload_npz(data_vis['proc_com_data_paths'])
            data_gain_all={}
            gcal_exp_all[vis] = np.zeros(data_vis['dim_all'],dtype=float)*np.nan
            min_BERV = np.min((min_BERV,np.min(data_vis['min_BERV'])))
            max_BERV = np.max((max_BERV,np.max(data_vis['max_BERV'])))

            #Path to calibration profile for each exposure
            data_vis['cal_data_paths'] = gen_dic['save_data_dir']+'Processed_data/Calibration/'+inst+'_'+vis+'_'
            
            #Error tables are not available as input
            #    - this cases naturally covers the absence of blaze-derived calibration profiles
            #    - errors have been scaled as
            # EF_meas_scaled(w,t,v)^2 = EF_meas(w,t,v)^2*g_err
            #      thus (see weights_bin_prof())
            # EF_meas_scaled(t,v)^2 / TF_meas(band,t,v) = g_err*gcal(band,t,v) = gcal_scaled(band,t,v)
            #      since we cannot access the blaze we assume it is constant and unity, and thus define 
            # gcal_scaled(band,t,v) = g_err/dw(w)
            #      which we approximate with a constant pixel width per order, using the common table associated with the instrument
            if data_vis['mock'] or (not gen_dic[inst][vis]['flag_err']):
                gcal_blaze_vis = False
                if data_vis['mock']:cst_gain = 1.
                elif (not gen_dic[inst][vis]['flag_err']):cst_gain = gen_dic['g_err'][inst]
                data_com_inst = dataload_npz(data_dic[inst]['proc_com_data_path'])
                dw_mean_all = np.mean(data_com_inst['edge_bins'][:,1::]-data_com_inst['edge_bins'][:,0:-1],axis=1)
                data_gain={'gcal_inputs' : {iord : {'par':None,'args':{'constant':cst_gain/dw_mean_all[iord]}} for iord in range(data_dic[inst]['nord'])}}
                n_glob_groups = data_vis['n_in_visit']
                iexp_glob_groups_vis[vis] = range(n_glob_groups) 
                for iexp in range(data_vis['n_in_visit']):data_gain_all[iexp] = data_gain                
    
            #Error tables are available as input  
            else:
                
                #Calibration measured from blazed spectra
                #    - blaze is retrieved exactly for each exposure, thus there is no need to group them over several exposures
                if (vis in data_inst['gcal_blaze_vis']):
                    gcal_blaze_vis = True                    
                    iexp_gain_groups = list(range(i,min(i+1,data_vis['n_in_visit'])) for i in range(0,data_vis['n_in_visit']))
                    
                #Calibration estimated from sum(s[F]^2)/sum(F^2) summed over larger bins  
                else:
                    gcal_blaze_vis = False
                    iexp_gain_groups = list(range(i,min(i+gen_dic['gcal_binN'],data_vis['n_in_visit'])) for i in range(0,data_vis['n_in_visit'],gen_dic['gcal_binN']))

                #BERV per exposure group
                n_glob_groups = len(iexp_gain_groups)
                BERV_all = np.zeros(n_glob_groups,dtype=float)
                for iexp_glob,iexp_in_group in enumerate(iexp_gain_groups):
                    BERV_all[iexp_glob] = np.mean(data_prop[inst][vis]['BERV'][iexp_in_group])

                #Binning calibration profiles in each exposure for fitting
                #    - measured blaze-derived profiles are binned as well as they are otherwise too heavy to store
                gcal_val_all = np.zeros([n_glob_groups,data_dic[inst]['nord']],dtype=object) 
                data_all_temp = {}
                for iexp_glob,iexp_in_group in enumerate(iexp_gain_groups):
                    for iexp in iexp_in_group:
                        data_all_temp[iexp] = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp))
                        if (gcal_blaze_vis):data_all_temp[iexp]['gcal'] = dataload_npz(data_vis['sing_gcal_DI_data_paths'][iexp])['gcal']                              
                for iord in range(data_dic[inst]['nord']):                    
                    for iexp_glob,iexp_in_group in enumerate(iexp_gain_groups):
                   
                        #Process exposures in current group
                        idx_def_group = np.zeros(0,dtype=int)
                        low_wav_group = np.zeros(0,dtype=float)
                        high_wav_group = np.zeros(0,dtype=float)
                        wav_group = np.zeros(0,dtype=float)
                        if (not gcal_blaze_vis):
                            flux_group = np.zeros(0,dtype=float)
                            var_group = np.zeros(0,dtype=float)
                            gcal_blaze_group=None  
                        else:
                            flux_group = None
                            var_group = None
                            gcal_blaze_group = np.zeros(0,dtype=float)
                        for iexp in iexp_in_group:

                            #Defined bins
                            #    - we further exclude ranges that were found to display abnormal calibration estimates (when estimated from error and flux tables)     
                            cond_def_ord = data_all_temp[iexp]['cond_def'][iord]
                            if gcal_blaze_vis:
                                cond_def_ord &= ~np.isnan(data_all_temp[iexp]['gcal'][iord])
                            else:
                                if inst=='HARPN':
                                    if data_inst['idx_ord_ref'][iord]==51:cond_def_ord[data_all_temp[iexp]['cen_bins'][iord]<5742.] = False
                                    elif data_inst['idx_ord_ref'][iord]==64:cond_def_ord[(data_all_temp[iexp]['cen_bins'][iord]>6561.) & (data_all_temp[iexp]['cen_bins'][iord]<6564.)] = False    
                                elif inst=='HARPS': 
                                    if data_inst['idx_ord_ref'][iord]==66:cond_def_ord[(data_all_temp[iexp]['cen_bins'][iord]>6558.) & (data_all_temp[iexp]['cen_bins'][iord]<6569.)] = False
                            idx_def_exp_ord = np_where1D(cond_def_ord)

                            #Concatenate tables so that they are binned together
                            if len(idx_def_exp_ord)>0.:
                                idx_def_group = np.append(idx_def_group,idx_def_exp_ord)
                                low_wav_group = np.append(low_wav_group,data_all_temp[iexp]['edge_bins'][iord,0:-1])
                                high_wav_group = np.append(high_wav_group,data_all_temp[iexp]['edge_bins'][iord,1::] )
                                wav_group = np.append(wav_group,data_all_temp[iexp]['cen_bins'][iord])
                                if (not gcal_blaze_vis):
                                    flux_group = np.append(flux_group,data_all_temp[iexp]['flux'][iord])
                                    var_group = np.append(var_group,data_all_temp[iexp]['cov'][iord][0])
                                else:
                                    gcal_blaze_group = np.append(gcal_blaze_group,data_all_temp[iexp]['gcal'][iord])

                        #Process (grouped) exposures
                        if np.sum(idx_def_group)>0:

                            #Binned calibration tables                          
                            bin_bd,raw_loc_dic = sub_def_bins(gen_dic['gcal_binw'][inst],idx_def_group,low_wav_group,high_wav_group,high_wav_group-low_wav_group,wav_group,flux_group,var1D_loc=var_group,gcal_blaze = gcal_blaze_group)
                            
                            #Adding progressively bins that will be used to fit the correction
                            #    - calibration values are scaled down temporarily to values closer to unity
                            bin_ord_dic={}
                            for key in ['gcal','cen_bins']:bin_ord_dic[key] = np.zeros(0,dtype=float) 
                            for ibin,(low_bin_loc,high_bin_loc) in enumerate(zip(bin_bd[0:-1],bin_bd[1:])):
                                bin_loc_dic,_ = sub_calc_bins(low_bin_loc,high_bin_loc,raw_loc_dic,0,calc_gcal=True)
                                if len(bin_loc_dic)>0:
                                    bin_ord_dic['gcal'] = np.append( bin_ord_dic['gcal'] , bin_loc_dic['gcal']*1e-3)
                                    bin_ord_dic['cen_bins'] = np.append( bin_ord_dic['cen_bins'] , bin_loc_dic['cen_bins'])
                            gcal_val_all[iexp_glob,iord] = bin_ord_dic
                        else:gcal_val_all[iexp_glob,iord] = None

                    ### End of exposure groups

                ### End of orders

                #Initialize fit structure
                iexp_glob_groups_vis[vis] = range(n_glob_groups) 
                p_start = Parameters()           
                p_start.add_many(
                          ('a1',0., True   , None , None , None),
                          ('a2',0., True   , None , None , None),
                          ('a3',0., True   , None , None , None),
                          ('a4',0., True   , None , None , None),
                          ('b3',0., True   , None , None , None),
                          ('b4',0., True   , None , None , None),
                          ('c0',0., True   , None , None , None),
                          ('c1',0., True   , None , None , None),
                          ('c2',0., True   , None , None , None),
                          ('c3',0., True   , None , None , None),
                          ('c4',0., True   , None , None , None)) 
                for key in ['blue','mid','red']:
                    if (gen_dic['gcal_deg'][key]<2) or (gen_dic['gcal_deg'][key]>4):stop('Degrees must be between 2 and 4')
                for ideg in range(gen_dic['gcal_deg']['mid']+1,5):p_start['b'+str(ideg)].vary = False                             
                for ideg in range(gen_dic['gcal_deg']['blue']+1,5):p_start['a'+str(ideg)].vary = False                                   
                for ideg in range(gen_dic['gcal_deg']['red']+1,5):p_start['c'+str(ideg)].vary = False   
                nfree_gainfit =  gen_dic['gcal_deg']['blue']+gen_dic['gcal_deg']['red']+1+gen_dic['gcal_deg']['mid']-2                                     

                fixed_args={
                    'use_cov':False,
                    'deg_low':gen_dic['gcal_deg']['blue'],
                    'deg_mid':gen_dic['gcal_deg']['mid'],
                    'deg_high':gen_dic['gcal_deg']['red'],
                    'constant':None                                                
                    }                                           

                #Model calibration profile
                #    - fitted over the measured estimates, to get a smooth profile not biased by tellurics and detector noise in case of measured profiles, or to allow extending blaze-derived profiles over the full spectrum range
                #    - the scaling is removed from the best-fit parameters so that the model is comparable to the original profiles
                common_args = (plot_save,data_dic[inst]['nord'],gcal_val_all,inst,gen_dic['gcal_thresh'][inst],gen_dic['gcal_edges'],gen_dic['gcal_nooutedge'],fixed_args,nfree_gainfit,p_start,data_vis['cal_data_paths'],gcal_blaze_vis)
                if (gen_dic['gcal_nthreads']>1) and (gen_dic['gcal_nthreads']<=n_glob_groups):data_gain_all = para_model_gain(model_gain,gen_dic['gcal_nthreads'],n_glob_groups,[iexp_glob_groups_vis[vis],iexp_gain_groups],common_args)                           
                else:data_gain_all = model_gain(iexp_glob_groups_vis[vis],iexp_gain_groups,*common_args)  

                #Exposure-specific profiles
                #    - to be used for weighing
                for iexp_glob,iexp_in_group in enumerate(iexp_gain_groups):
                    cal_inputs_iexp_glob = data_gain_all[iexp_glob]['gcal_inputs']
                    for iexp in iexp_in_group:
    
                        #From blaze-derived profile
                        #    - we use the best-fit model to the measured calibration profile in each exposure to define them at undefined pixels
                        #    - calculated in any case so that an average scaling calibration profile common to all exposures must can be calculated
                        if gcal_blaze_vis: 
                            gcal_exp_all[vis][iexp] = data_all_temp[iexp]['gcal']
                            cond_undef_blaze_exp = np.isnan(data_all_temp[iexp]['gcal'])
                            for iord in range(data_inst['nord']):
                                if np.sum(cond_undef_blaze_exp[iord])>0:
                                    gcal_exp_all[vis][iexp,iord,cond_undef_blaze_exp[iord]] = cal_piecewise_func(cal_inputs_iexp_glob[iord]['par'],data_all_temp[iexp]['cen_bins'][iord,cond_undef_blaze_exp[iord]],args=cal_inputs_iexp_glob[iord]['args'])[0] 
                                    
                            #Saving if weighing is required
                            if gen_dic['cal_weight']:
                                data_gcal_exp = dataload_npz(data_vis['sing_gcal_DI_data_paths'][iexp])
                                data_gcal_exp['gcal'] = gcal_exp_all[vis][iexp]
                                datasave_npz(data_vis['sing_gcal_DI_data_paths'][iexp],data_gcal_exp) 
                            
                            #Deleting if weighing is not required 
                            else:os_system.remove(data_vis['sing_gcal_DI_data_paths'][iexp]) 
                            
    
                        #From calibration profile estimate
                        #    - we use the model only to avoid the spurious features in the estimated profile
                        #    - only calculated for weighing, as the model is directly recomputed on a common grid to define the scaling profile
                        elif gen_dic['cal_weight']:
                            for iord in range(data_inst['nord']):
                                gcal_exp_all[vis][iexp][iord] = cal_piecewise_func(cal_inputs_iexp_glob[iord]['par'],data_all_temp[iexp]['cen_bins'][iord],args=cal_inputs_iexp_glob[iord]['args'])[0] 
    
                            #Saving
                            datasave_npz(data_vis['sing_gcal_DI_data_paths'][iexp],{'gcal':gcal_exp_all[vis][iexp]}) 

                data_all_temp.clear()


            #Storing best-fit calibration parameters
            cal_inputs_dic[vis] = np.zeros([data_dic[inst]['nord'],n_glob_groups],dtype=object)
            for iord in range(data_dic[inst]['nord']): 

                #Widest spectral range over all visits   
                #    - defined in the input rest frame
                min_edge_ord_all[iord] = np.min([min_edge_ord_all[iord],data_com_vis['min_edge_ord'][iord]])
                max_edge_ord_all[iord] = np.max([max_edge_ord_all[iord],data_com_vis['max_edge_ord'][iord]])
                
                #Retrieve function inputs 
                #    - defined in the rest frame of each input spectrum
                for iexp_glob in iexp_glob_groups_vis[vis]:
                    cal_inputs_dic[vis][iord,iexp_glob] = data_gain_all[iexp_glob]['gcal_inputs'][iord] 

        ### End of visit
        
        #------------------------------------------------------------------------------------------------------------
        #Calibration profile for scaling
        
        #Median calibration profile over all exposures in the visit
        #    - we assume the median calibration is smooth enough that it can be captured with an interpolation function
        #    - we assume the spectral tables are always more resolved than the typical variations of the calibration profiles
        #    - the median calibration profile is defined in the detector (Earth) rest frame, over a regular table in log oversampled compared to the instrument resolution (dlnw_inst = dw/w = 1/R)           
        mean_gcal_func = {} 
        gain_grid_dlnw = 0.5/return_resolv(inst)
        minmax_dopp_shift = 1./(np.array([gen_specdopshift(min_BERV),gen_specdopshift(max_BERV)])*(1.+1.55e-8))    
        min_dopp_shift = np.min(minmax_dopp_shift)
        max_dopp_shift = np.max(minmax_dopp_shift)
        if (len(data_inst['gcal_blaze_vis'])>0):
            edge_bins_all_temp={}
            for vis in data_inst['gcal_blaze_vis']:
                edge_bins_all_temp[vis]={}
                for iexp in range(data_dic[inst][vis]['n_in_visit']):
                    edge_bins_all_temp[vis][iexp]={'edge_bins':dataload_npz(data_dic[inst][vis]['proc_DI_data_paths']+str(iexp))['edge_bins']} 
        for iord in range(data_dic[inst]['nord']): 
            
            #Table of definition of mean calibration profile over each order
            #    - in the detector frame
            #    - over the widest range covered by the instrument visits, and at the oversampled instrumental resolution
            #    - tables are uniformely spaced in ln(w)
            #      d[ln(w)] = sc*dw/w = sc*dv/c = sc/R     
            min_edge_ord_Earth = min_edge_ord_all[iord]*min_dopp_shift
            max_edge_ord_Earth = max_edge_ord_all[iord]*max_dopp_shift
            nspec_ord = 1+int( np.ceil(   np.log(max_edge_ord_Earth/min_edge_ord_Earth)/np.log( gain_grid_dlnw + 1. ) )  ) 
            cen_bins_ord = min_edge_ord_Earth*( gain_grid_dlnw + 1. )**np.arange(nspec_ord)
            if (len(data_inst['gcal_blaze_vis'])>0):edge_bins_ord = def_edge_tab(cen_bins_ord,dim=0)
        
            #Median calibration over all exposures in the visit
            #    - the small variations in measured calibration between orders may result in two slices having different profiles
            #      however this is not an issue for the eventual weighing of the flux profiles
            #    - the calibration profiles from each exposure are comparable in the detector rest frame
            med_gcal_allvis = np.zeros(nspec_ord,dtype=float)  
            for ivis,vis in enumerate(data_dic[inst]['visit_list']): 
                mean_gcal_ord = np.zeros([nspec_ord,0],dtype=float)*np.nan 
                for iexp_glob in iexp_glob_groups_vis[vis]:
                    dopp_shift = 1./(gen_specdopshift(BERV_all[iexp_glob])*(1.+1.55e-8))
               
                    #Calculating best-fit model of estimated calibration
                    #     - the model is calculated on the grid defined in detector frame, shifted back to the input rest frame, in which the model coefficients were derived
                    #       since it is defined on the 'cen_bins_ord' grid, it remains defined in the detector frame when saved and used with this grid
                    if (vis not in data_inst['gcal_blaze_vis']):
                        mean_gcal_ord_loc = np.zeros(nspec_ord,dtype=float)*np.nan
                        mean_gcal_ord_loc=cal_piecewise_func(cal_inputs_dic[vis][iord,iexp_glob]['par'],cen_bins_ord/dopp_shift,args=cal_inputs_dic[vis][iord,iexp_glob]['args'])[0] 
                    
                    #Resampling complete blaze-derived calibration profile 
                    #    - no grouped exposures in that case, ie iexp_glob = iexp 
                    #    - calibration profiles are measured in the frame of the input exposure defined in the solar barycentric rest frame
                    #      they are thus shifted to the detector frame (see gen_specdopshift()) to be averaged:
                    # w_Earth = w_solbar / (1+ (BERV/c)) 
                    else:            
                        mean_gcal_ord_loc = bind.resampling(edge_bins_ord, edge_bins_all_temp[vis][iexp_glob]['edge_bins'][iord]*dopp_shift, gcal_exp_all[vis][iexp_glob,iord], kind='cubic')      

                    #Storing
                    mean_gcal_ord = np.append(mean_gcal_ord,mean_gcal_ord_loc[:,None],axis=1)
                    
                #Co-adding median profile over the visit
                cond_def_bins = (np.sum(~np.isnan(mean_gcal_ord),axis=1)>0)
                med_gcal_allvis[cond_def_bins]+=np.nanmedian(mean_gcal_ord[cond_def_bins],axis=1)     

            #Mean over all visits
            #    - we limit the calibration below the chosen global outlier threshold when estimated from flux and error tables
            #    - we do not extrapolate beyond the range of definition of the median calibration profile to avoid spurious behaviour
            #    - we create a function by interpolating the profile at pixels where it is >0
            #      interp1d is more stable at the edges than CubicSpline, and capture better the calibration variations than polynomials
            med_gcal_ord = med_gcal_allvis/gen_dic[inst]['n_visits'] 
            if (len(data_inst['gcal_blaze_vis'])>0):med_gcal_ord[med_gcal_ord>gen_dic['gcal_thresh'][inst]['global']] = gen_dic['gcal_thresh'][inst]['global']
            cond_pos_bins = (med_gcal_ord>0.)
            mean_gcal_func[iord] = interp1d(cen_bins_ord[cond_pos_bins],med_gcal_ord[cond_pos_bins],bounds_error=False,fill_value=(med_gcal_ord[cond_pos_bins][0],med_gcal_ord[cond_pos_bins][-1]))  

        #Store common calibration function
        np.savez_compressed(gen_dic['save_data_dir']+'Processed_data/Calibration/'+inst+'_mean_gcal',data = {'func':mean_gcal_func},allow_pickle=True)  

        #Scaling calibration tables for each exposure
        #    - used for scaling, not for weighing
        #    - the calibration profile calculated over all exposures is defined in the detector rest frame
        #      the exposure profile is thus calculated in the detector rest frame, but over the exposure grid in the input rest frame, where it is defined when called with this grid
        #    - the path is made specific to a visit and to a type of profile so that the calibration function can still be called in the multi-visit routines for any type of profile,
        # even after the type of profile has changed in a given visit
        for vis in data_inst['visit_list']:
            data_vis=data_inst[vis]
            data_vis['mean_gcal_DI_data_paths'] = {}
            for iexp in range(data_vis['n_in_visit']):
                gcal_exp = np.zeros(data_vis['dim_exp'],dtype=float)*np.nan
                data_exp = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp))
                dopp_shift = 1./(gen_specdopshift(data_prop[inst][vis]['BERV'][iexp])*(1.+1.55e-8))
                for iord in range(data_inst['nord']):
                    gcal_exp[iord] = mean_gcal_func[iord](data_exp['cen_bins'][iord]/dopp_shift)
                data_vis['mean_gcal_DI_data_paths'][iexp] = data_vis['proc_DI_data_paths']+'mean_gcal_'+str(iexp)
                np.savez_compressed(data_vis['mean_gcal_DI_data_paths'][iexp], data = {'mean_gcal':gcal_exp},allow_pickle=True) 
           
    else: 
        for vis in data_inst['visit_list']: 
            data_inst[vis]['cal_data_paths'] = gen_dic['save_data_dir']+'Processed_data/Calibration/'+inst+'_'+vis+'_'
            data_inst[vis]['mean_gcal_DI_data_paths'] = {}
            for iexp in range(data_inst[vis]['n_in_visit']):data_inst[vis]['mean_gcal_DI_data_paths'][iexp] = data_inst[vis]['proc_DI_data_paths']+'mean_gcal_'+str(iexp)
            check_data(data_inst[vis]['mean_gcal_DI_data_paths'],vis=vis)                    

    return None





def model_gain(iexp_glob_groups,iexp_gain_groups,plot_save,nord,gcal_val_all,inst,gcal_thresh,gcal_edges,gcal_nooutedge,fixed_args,nfree_gainfit,p_start,cal_data_paths,gcal_blaze_vis):
    r"""**Fit function for instrumental calibration.**

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    data_gain_all = {}
    for iexp_glob,iexp_gain_group in zip(iexp_glob_groups,iexp_gain_groups):
        data_gain={'gcal_inputs':{}}
        if plot_save:
            data_gain['wav_bin_all']=np.zeros(nord,dtype=object)
            data_gain['wav_trans_all']=np.zeros([2,nord],dtype=float)
            data_gain['cond_fit_all']=np.zeros(nord,dtype=object)
            data_gain['gcal_bin_all']=np.zeros(nord,dtype=object)  
        for iord in range(nord):
            
            #Process order if defined
            fit_done = False
            if gcal_val_all[iexp_glob,iord] is not None:
                bin_ord_dic = gcal_val_all[iexp_glob,iord] 
                
                #Cleaning calibration estimates
                #    - not required for blaze-derived profiles
                if (not gcal_blaze_vis):
                   
                    #Fitting positive values below global threshold
                    cond_fit = (bin_ord_dic['gcal']>0.) & (bin_ord_dic['gcal']<gcal_thresh['global']/1e3)
        
                    #Remove extreme outliers
                    med_prop = np.median(bin_ord_dic['gcal'][cond_fit])
                    res = bin_ord_dic['gcal'][cond_fit] - med_prop
                    cond_fit[cond_fit][np.abs(res) > 10.*stats.median_abs_deviation(res)] = False  
    
                else:cond_fit = np.ones(len(bin_ord_dic['gcal']),dtype=bool)
    
                #Generic fit properties
                #    - the fit does not use the covariance matrix, so fitted tables can be limited to non-consecutive pixels and 'idx_fit' covers all fitted pixels 
                low_edge = bin_ord_dic['cen_bins'][0]
                high_edge = bin_ord_dic['cen_bins'][-1]
                drange =  high_edge-low_edge    
                fixed_args.update({'w_lowedge':low_edge+gcal_edges['blue']*drange,
                                   'w_highedge':high_edge-gcal_edges['red']*drange,
                                   'wref':0.5*(low_edge+high_edge)})  
                
                #Fit binned calibration profile and define complete calibration profile
                #    - if enough bins are defined, otherwise a single measured value is used for the order
                if (np.sum(cond_fit)>nfree_gainfit):
    
                    #Imposing that gradient is negative (resp. positive) at the blue (resp. red) edges
                    p_start.add_many(('dPblue_wmin',-1., True   , None , 0. ))                                            
                    p_start.add_many(('wmin',low_edge-fixed_args['wref'], False)) 
                    p_start.add_many(('a1',0, False   , None , None , 'dPblue_wmin-(2*a2*wmin+3*a3*wmin**2.+4*a4*wmin**3.)')) 
                          
                    p_start.add_many(('dPred_wmax',1., True   , 0. , None))                                            
                    p_start.add_many(('wmax',high_edge-fixed_args['wref'], False)) 
                    p_start.add_many(('c1',0, False   , None , None , 'dPred_wmax-(2*c2*wmax+3*c3*wmax**2.+4*c4*wmax**3.)')) 
                                                          
                    #Fitting
                    #    - sigma-clipping applied to the inner parts of the order to prevent excluding edges where calibration can vary sharply
                    cond_check = deepcopy(cond_fit) 
                    if (not gcal_blaze_vis) and (inst in gcal_nooutedge):cond_check = cond_fit & (bin_ord_dic['cen_bins']>=bin_ord_dic['cen_bins'][0]+gcal_nooutedge[inst][0]) & (bin_ord_dic['cen_bins']<=bin_ord_dic['cen_bins'][-1]-gcal_nooutedge[inst][1]) 
                    if (np.sum(cond_check)>nfree_gainfit):

                        #Temporary fit to identify outliers
                        #    - negative values are removed
                        #    - weights are used to prevent point with large calibrations (associated with low fluxes) biasing the simple polynomial fit                        
                        if (not gcal_blaze_vis):
                            var_fit = bin_ord_dic['gcal'][cond_check]
                            fixed_args['idx_fit'] = np.ones(np.sum(cond_check),dtype=bool)
                            _,merit,_ = call_lmfit(p_start,bin_ord_dic['cen_bins'][cond_check],bin_ord_dic['gcal'][cond_check],np.array([var_fit]),cal_piecewise_func,verbose=False,fixed_args=fixed_args)  
                            res_gcal = bin_ord_dic['gcal'][cond_check] - merit['fit']
                            cond_fit[cond_check] = np.abs(res_gcal)<=gcal_thresh['outliers']*np.std(res_gcal)
                        else:merit = {'chi2r':1} 
    
                        #Model fit
                        #    - errors are scaled with the reduced chi2 from the preliminary fit, if relevant                           
                        if (np.sum(cond_fit)>nfree_gainfit):
                            fit_done = True
                            
                            #Fitting
                            var_fit = bin_ord_dic['gcal'][cond_fit]*merit['chi2r'] 
                            fixed_args['idx_fit'] = np.ones(np.sum(cond_fit),dtype=bool)
                            _,merit,p_best = call_lmfit(p_start,bin_ord_dic['cen_bins'][cond_fit],bin_ord_dic['gcal'][cond_fit],np.array([var_fit]),cal_piecewise_func,verbose=False,fixed_args=fixed_args)                                              

                            #Scaling back relevant properties
                            p_final = {par_loc:p_best[par_loc].value for par_loc in p_best}
                            for par_loc in ['a1','a2','a3','a4','b3','b4','c0','c1','c2','c3','c4']:p_final[par_loc]*=1e3

                            #Storing best-fit
                            data_gain['gcal_inputs'][iord] = {'par':deepcopy(p_final),'args':deepcopy(fixed_args)}
    
                #Fit could not be performed
                if not fit_done:
                    data_gain['gcal_inputs'][iord]={'par':None,'args':{'constant':np.median(bin_ord_dic['gcal']*1e3)}}
    
                #Save
                if plot_save:
                    data_gain['wav_bin_all'][iord] = bin_ord_dic['cen_bins']
                    data_gain['wav_trans_all'][:,iord] = [fixed_args['w_lowedge'],fixed_args['w_highedge']]
                    data_gain['cond_fit_all'][iord] = cond_fit
                    data_gain['gcal_bin_all'][iord] = bin_ord_dic['gcal']*1e3
    
            #Order fully undefined
            #    - no calibration is applied
            else:
                data_gain['gcal_inputs'][iord]={'par':None,'args':{'constant':1.}}

        ### End of orders
    
        #Save calibration model properties for each original exposure associated with current exposure group
        data_gain_all[iexp_glob]=data_gain 
        if plot_save:
            for iexp in iexp_gain_group:np.savez_compressed(cal_data_paths+str(iexp),data=data_gain,allow_pickle=True) 
          
    ### End of exposure groups            
          
    return data_gain_all

def para_model_gain(func_input,nthreads,n_elem,y_inputs,common_args): 
    r"""**Multithreading routine for model_gain().**

    Args:
        func_input (function): multi-threaded function
        nthreads (int): number of threads
        n_elem (int): number of elements to thread
        y_inputs (list): threadable function inputs 
        common_args (tuple): common function inputs
    
    Returns:
        y_output (None or specific to func_input): function outputs 
    
    """
    pool_proc = Pool(processes=nthreads)   #cannot be passed through lmfit
    ind_chunk_list=init_parallel_func(nthreads,n_elem)
    chunked_args=[(y_inputs[0][ind_chunk[0]:ind_chunk[1]],y_inputs[1][ind_chunk[0]:ind_chunk[1]])+common_args for ind_chunk in ind_chunk_list]	
    all_results=tuple(tab for tab in pool_proc.starmap(func_input,chunked_args))

    data_gain_all = {}
    for data_gain in tuple(all_results[i] for i in range(nthreads)):data_gain_all.update(data_gain)
    y_output=data_gain_all
    
    pool_proc.close()
    pool_proc.join() 				
    return y_output




def cal_piecewise_func(param_in,wav_in,args=None):
    r"""**Instrumental calibration model.**
    
    Defines joined polynomials to model calibration profiles.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    if isinstance(param_in,lmfit.parameter.Parameters):params={par:param_in[par].value for par in param_in}
    else:params=deepcopy(param_in)                                                   
    #P_low(w) = sum(0:nlow,ai*w^i)
    #P_mid(w) = sum(0:nmid,bi*w^i)                                               
    #P_high(w) = sum(0:nhigh,ci*w^i)
    #   with w = wav-wav_ref
    #
    #Continuity in derivative:
    #P_low'(w1) = P_mid'(w1)  
    #   sum(1:nlow,i*ai*w1^i-1) = b1 + 2*b2*w1 + sum(3:nmid,i*bi*w1^i-1)
    #P_high'(w2) = P_mid'(w2)   
    #   sum(1:nhigh,i*ci*w2^i-1) = b1 + 2*b2*w2 + sum(3:nmid,i*bi*w2^i-1)
    #
    # 2*b2*(w1-w2) = sum(1:nlow,i*ai*w1^i-1) - sum(1:nhigh,i*ci*w2^i-1) + sum(3:nmid,i*bi*w2^i-1) - sum(3:nmid,i*bi*w1^i-1)
    # > b2 = (sum(1:nlow,i*ai*w1^i-1) - sum(1:nhigh,i*ci*w2^i-1) + sum(3:nmid,i*bi*w2^i-1) - sum(3:nmid,i*bi*w1^i-1))/(2*(w1-w2))      
    #      = (sum(0:nlow-1,(j+1)*a[j+1]*w1^j) - sum(0:nhigh-1,(j+1)*c[j+1]*w2^j) + sum(2:nmid-1,(j+1)*b[j+1]*w2^j) - sum(2:nmid-1,(j+1)*b[j+1]*w1^j)  )/(2*(w1-w2))     
    #
    # > b1 = sum(0:nlow-1,(j+1)*a[j+1]*w1^j) - sum(2:nmid,i*bi*w1^i-1)
    #      = sum(0:nlow-1,(j+1)*a[j+1]*w1^j) - sum(1:nmid-1,(j+1)*b[j+1]*w1^j) 
    #
    #Continuity in value:
    #P_low(w1) = P_mid(w1) 
    #   sum(0:nlow,ai*w1^i) = b0 + sum(1:nmid,bi*w1^i)
    #P_high(w2) = P_mid(w2)   
    #   sum(0:nhigh,ci*w2^i) = b0 + sum(1:nmid,bi*w2^i)
    #
    # a0 + sum(1:nlow,ai*w1^i) - sum(0:nhigh,ci*w2^i) = sum(1:nmid,bi*w1^i) - sum(1:nmid,bi*w2^i)  
    # > a0 = sum(1:nmid,bi*w1^i) - sum(1:nmid,bi*w2^i) - sum(1:nlow,ai*w1^i) + sum(0:nhigh,ci*w2^i)
    #
    # > b0 = sum(0:nlow,ai*w1^i) - sum(1:nmid,bi*w1^i)
    if args['constant'] is not None:
        model = np.repeat(args['constant'],len(wav_in))
    else:
        wav = wav_in-args['wref']
        w1 = args['w_lowedge']-args['wref']
        w2 = args['w_highedge']-args['wref'] 
    
        #Derivative continuity
        dPlow = Polynomial([(ideg+1)*params['a'+str(ideg+1)] for ideg in range(args['deg_low'])])
        dPhigh = Polynomial([(ideg+1)*params['c'+str(ideg+1)] for ideg in range(args['deg_high'])])
        dPmid_cut = Polynomial([0.,0.]+[(ideg+1)*params['b'+str(ideg+1)] for ideg in range(2,args['deg_mid'])])
        params['b2'] = (dPlow(w1) - dPhigh(w2) + dPmid_cut(w2) - dPmid_cut(w1))/(2.*(w1-w2)) 
        params['b1'] = dPlow(w1) - Polynomial([0.]+[(ideg+1)*params['b'+str(ideg+1)] for ideg in range(1,args['deg_mid'])])(w1)   

        #Value continuity        
        Phigh = Polynomial([params['c'+str(ideg)] for ideg in range(args['deg_high']+1)])   
        Pmid_cut = Polynomial([0.]+[params['b'+str(ideg)] for ideg in range(1,args['deg_mid']+1)])
        params['a0'] = Pmid_cut(w1) - Pmid_cut(w2) - Polynomial([0]+[params['a'+str(ideg)] for ideg in range(1,args['deg_low']+1)])(w1) + Phigh(w2)
        Plow = Polynomial([params['a'+str(ideg)] for ideg in range(args['deg_low']+1)])
        params['b0'] = Plow(w1) - Pmid_cut(w1)
    
        #Model
        model = np.zeros(len(wav),dtype=float)
        cond_mid = (wav>=w1) & (wav<=w2)
        model[cond_mid] = Polynomial([params['b'+str(ideg)] for ideg in range(args['deg_mid']+1)])(wav[cond_mid])                          
        cond_lowe=wav<w1
        model[cond_lowe] = Plow(wav[cond_lowe])     
        cond_highe=wav>w2
        model[cond_highe] = Phigh(wav[cond_highe])
        
    return model,None

