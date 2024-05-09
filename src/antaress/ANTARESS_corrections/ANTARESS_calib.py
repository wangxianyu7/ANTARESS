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
from ..ANTARESS_conversions.ANTARESS_binning import sub_calc_bins,sub_def_bins
from ..ANTARESS_analysis.ANTARESS_inst_resp import return_resolv
from ..ANTARESS_general.utils import dataload_npz,np_where1D,stop,init_parallel_func,check_data
from ..ANTARESS_general.minim_routines import call_lmfit




def calc_gcal(gen_dic,data_dic,inst,plot_dic,coord_dic):
    r"""**Main instrumental calibration routine.**
    
    Estimates instrumental calibration (see `weights_bin_prof()`)
    
    
    Calibration is used as weight or to scale back profiles approximatively to their raw photoelectron counts during CCF calculations (this allows artifically increasing errors when combining spectral regions with different SNRs).
    Spectra must however be kept to their extracted flux units througout ANTARESS processing, as the stellar spectrum shifts over time in the instrument rest frame.
    The same region of the spectrum thus sees different instrumental responses over time and is measured with different flux levels
    shifted stellar spectra with the same profile would thus have a different color balance after being converted in raw count units, preventing in particular the correct calculation of binned spectra 
    
    Weights are used for temporal binning, and are relevant only if the weight of a given pixel change over time, the calibration is thus not defined for CCFs.
    
    A median calibration profile over all visits of an instrument is calculated in spectral mode. 
    We first fit a model to the calibration for each exposure, so that it can be extrapolated over a larger, common range for all visits 
    then we calculate the median of these extrapolated model, interpolate it as a function, and use it to define the common calibration profile over the specific table of each exposure.
    
    Overall the estimated calibrations are stable between exposures but low count levels ten to yield larger calibrations - hence the independent calculation of calibration per exposure, before taking the median over the visit.
    This may come from additional noise sources.
    If :math:`E = E_\mathrm{white} + E_\mathrm{red} = \sqrt{g_\mathrm{true}} \sqrt{F} + E_\mathrm{red}` then :math:`g_\mathrm{true} = (E-E_\mathrm{red})^2/F < E^2/F = g_\mathrm{meas}` (since :math:`E_\mathrm{red} < 2 E`), so that we underestimate the actual calibration. 
    In this case our assumptions to estimate gdet do not hold anymore, but for the purpose of weighing the exposures and scaling to raw count levels we still use the measured calibration.
    Since :math:`g_\mathrm{meas} = (E_\mathrm{white} + E_\mathrm{red})^2/F` this factor accounts for the true calibration but also for additional noise sources.        
    
    Spectra are typically provided in the solar barycentric rest frame, and are thus not defined here in the detector rest frame, so that the calibration profiles in different epochs may be shifted by the Earth barycentric RV difference.   
    We nonetheless use a single calibration profile, constant in time and common to all processed exposures of an instrument, so that the relative color balance between spectra is not modified when converting them back to counts. 
    For the same reason the calibration must be applied uniformely (ie, in the same rest frame) to spectra in different exposures, and their master, so that it does not affect their combinations - even if the original calibration to flux units is applied in the input rest frame.

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
        minmax_def = {}
        min_edge_ord_all = np.repeat(1e100,data_dic[inst]['nord'])
        max_edge_ord_all = np.repeat(-1e100,data_dic[inst]['nord'])   
        iexp_glob_groups_vis = {}
        for vis in data_dic[inst]['visit_list']:
            print('           Processing '+vis) 
            data_vis=data_dic[inst][vis]
            data_com_vis = dataload_npz(data_vis['proc_com_data_paths'])
            data_gain_all={}

            #Estimate of instrumental calibration
            #    - set to the requested scaling if error tables are not available as input, or derived from sum(s[F]^2)/sum(F^2) summed over larger bins
            data_vis['cal_data_paths'] = gen_dic['save_data_dir']+'Processed_data/Calibration/'+inst+'_'+vis+'_'
            if data_vis['mock'] or (not gen_dic[inst][vis]['flag_err']):
                if data_vis['mock']:cst_gain = 1.
                elif (not gen_dic[inst][vis]['flag_err']):cst_gain = gen_dic['g_err'][inst]
                data_gain={'gdet_inputs' : {iord : {'par':None,'args':{'constant':cst_gain}} for iord in range(data_dic[inst]['nord'])}}
                n_glob_groups = data_vis['n_in_visit']
                iexp_glob_groups_vis[vis] = range(n_glob_groups) 
                for iexp in range(data_vis['n_in_visit']):data_gain_all[iexp] = data_gain                
    
            else:
                
                #Exposure groups
                iexp_gain_groups = list(range(i,min(i+gen_dic['gcal_binN'],data_vis['n_in_visit'])) for i in range(0,data_vis['n_in_visit'],gen_dic['gcal_binN']))
                n_glob_groups = len(iexp_gain_groups)
                iexp_glob_groups_vis[vis] = range(n_glob_groups)  
                gdet_val_all = np.zeros([data_vis['n_in_visit'],data_dic[inst]['nord']],dtype=object)
                minmax_def[vis] = np.zeros([data_vis['n_in_visit'],data_dic[inst]['nord'],2])*np.nan 
                data_all_temp = {}
                for iord in range(data_dic[inst]['nord']):                    
                    for iexp_glob,iexp_in_group in enumerate(iexp_gain_groups):
                   
                        #Process exposures in current group
                        idx_def_group = np.zeros(0,dtype=int)
                        low_wav_group = np.zeros(0,dtype=float)
                        high_wav_group = np.zeros(0,dtype=float)
                        wav_group = np.zeros(0,dtype=float)
                        flux_group = np.zeros(0,dtype=float)
                        var_group = np.zeros(0,dtype=float)
                        for iexp in iexp_in_group:
                            if iexp not in data_all_temp:data_all_temp[iexp] = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp))
                      
                            #Defined bins
                            #    - we further exclude ranges that were found to display abnormal calibration estimates
                            cond_def_ord = data_all_temp[iexp]['cond_def'][iord]
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
                                flux_group = np.append(flux_group,data_all_temp[iexp]['flux'][iord])
                                var_group = np.append(var_group,data_all_temp[iexp]['cov'][iord][0])
               
                                #Save range over which profiles are defined
                                minmax_def[vis][iexp,iord,:]= [data_all_temp[iexp]['edge_bins'][iord,idx_def_exp_ord[0]],data_all_temp[iexp]['edge_bins'][iord,idx_def_exp_ord[-1]]]

                        #Initialize binned tables from grouped exposures
                        if np.sum(idx_def_group)>0:
                            bin_bd,raw_loc_dic = sub_def_bins(gen_dic['gcal_binw'],idx_def_group,low_wav_group,high_wav_group,high_wav_group-low_wav_group,wav_group,flux_group,var1D_loc=var_group)

                            #Adding progressively bins that will be used to fit the correction
                            bin_ord_dic={}
                            for key in ['gdet','cen_bins']:bin_ord_dic[key] = np.zeros(0,dtype=float) 
                            for ibin,(low_bin_loc,high_bin_loc) in enumerate(zip(bin_bd[0:-1],bin_bd[1:])):
                                bin_loc_dic,_ = sub_calc_bins(low_bin_loc,high_bin_loc,raw_loc_dic,0,calc_gdet=True)
                                if len(bin_loc_dic)>0:
                                    bin_ord_dic['gdet'] = np.append( bin_ord_dic['gdet'] , bin_loc_dic['gdet']*1e-3)
                                    bin_ord_dic['cen_bins'] = np.append( bin_ord_dic['cen_bins'] , bin_loc_dic['cen_bins'])
                            gdet_val_all[iexp_glob,iord] = bin_ord_dic
                        else:gdet_val_all[iexp_glob,iord] = None

                    ### End of exposure groups

                ### End of orders 

                #Initialize fit structure
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

                #Calibration for spectral profiles
                common_args = (minmax_def[vis],plot_dic,data_dic[inst]['nord'],gdet_val_all,inst,gen_dic['gcal_thresh'][inst],gen_dic['gcal_edges'],gen_dic['gcal_nooutedge'],fixed_args,nfree_gainfit,p_start,data_vis['cal_data_paths'])
                if gen_dic['gcal_nthreads']>1:data_gain_all = para_model_gain(model_gain,gen_dic['gcal_nthreads'],n_glob_groups,[iexp_glob_groups_vis[vis],iexp_gain_groups],common_args)                           
                else:data_gain_all = model_gain(iexp_glob_groups_vis[vis],iexp_gain_groups,*common_args)  

            #Processing all orders for the visit
            cal_inputs_dic[vis] = np.zeros([data_dic[inst]['nord'],n_glob_groups],dtype=object)
            for iord in range(data_dic[inst]['nord']): 

                #Widest spectral range over all visits   
                #    - defined in the input rest frame
                min_edge_ord_all[iord] = np.min([min_edge_ord_all[iord],data_com_vis['min_edge_ord'][iord]])
                max_edge_ord_all[iord] = np.max([max_edge_ord_all[iord],data_com_vis['max_edge_ord'][iord]])
                
                #Retrieve function inputs 
                #    - defined in the input rest frame
                for iexp_glob in iexp_glob_groups_vis[vis]:
                    cal_inputs_dic[vis][iord,iexp_glob] = data_gain_all[iexp_glob]['gdet_inputs'][iord] 

        #Median calibration profile over all exposures in the visit
        #    - we assume the median calibration is smooth enough that it can be captured with an interpolation function
        #    - we assume the spectral tables are always more resolved than the typical variations of the calibration profiles
        #    - the calibration profile is defined in the input rest frame, over a regular table in log oversampled compared to the instrument resolution (dlnw_inst = dw/w = 1/R)
        mean_gdet_func = {} 
        gain_grid_dlnw = 0.5/return_resolv(inst)
        for iord in range(data_dic[inst]['nord']): 

            #Table of definition of mean calibration profile over each order
            #    - over the widest range covered by the inst visits, and at the oversampled instrumental resolution
            #    - tables are uniformely spaced in ln(w)
            #      d[ln(w)] = sc*dw/w = sc*dv/c = sc/R             
            nspec_ord = 1+int( np.ceil(   np.log(max_edge_ord_all[iord]/min_edge_ord_all[iord])/np.log( gain_grid_dlnw + 1. ) )  ) 
            cen_bins_ord = min_edge_ord_all[iord]*( gain_grid_dlnw + 1. )**np.arange(nspec_ord)     
        
            #Median calibration over all exposures in the visit
            #    - interp1d is more stable at the edges than CubicSpline, and capture better the calibration variations than polynomials
            #    - the small variations in measured calibration between orders may result in two slices having different profiles
            #      however this is not an issue for the eventual weighing of the flux profiles
            med_gdet_allvis = np.zeros(nspec_ord,dtype=float)  
            for ivis,vis in enumerate(data_dic[inst]['visit_list']): 
                mean_gdet_ord = np.zeros([nspec_ord,0],dtype=float)*np.nan 
                for iexp_glob in iexp_glob_groups_vis[vis]:
                    mean_gdet_ord_loc = np.zeros(nspec_ord,dtype=float)*np.nan
                    mean_gdet_ord_loc=cal_piecewise_func(cal_inputs_dic[vis][iord,iexp_glob]['par'],cen_bins_ord,args=cal_inputs_dic[vis][iord,iexp_glob]['args'])      
                    mean_gdet_ord = np.append(mean_gdet_ord,mean_gdet_ord_loc[:,None],axis=1)
                med_gdet_allvis+=np.nanmedian(mean_gdet_ord,axis=1)     

            #Mean over all visits
            #    - we limit the calibration below the chosen global outlier threshold
            #    - we do not extrapolate beyond the range of definition of the median calibration profile to avoid spurious behaviour
            med_gdet_ord = 1e3*med_gdet_allvis/gen_dic[inst]['n_visits'] 
            med_gdet_ord[med_gdet_ord>gen_dic['gcal_thresh'][inst]['global']] = gen_dic['gcal_thresh'][inst]['global']
            med_gdet_ord[med_gdet_ord<=0.]=np.min(med_gdet_ord[med_gdet_ord>0.])
            mean_gdet_func[iord] = interp1d(cen_bins_ord,med_gdet_ord,bounds_error=False,fill_value=(med_gdet_ord[0],med_gdet_ord[-1]))  

        #Store mean calibration function
        np.savez_compressed(gen_dic['save_data_dir']+'Processed_data/Calibration/'+inst+'_mean_gdet',data = {'func':mean_gdet_func},allow_pickle=True)  

        #Define calibration tables for each exposure
        #    - the profile is the same, but defined over the table of the exposure
        #    - the path is made specific to a visit and a type of profile so that the calibration function can still be called in the multi-visit routines for any type of profile,
        # even after the type of profile has changed in a given visit
        for vis in data_inst['visit_list']: 
            data_vis=data_inst[vis]
            data_vis['mean_gdet_DI_data_paths'] = {}
            for iexp in range(data_vis['n_in_visit']):
                gain_exp = np.zeros(data_vis['dim_exp'],dtype=float)*np.nan
                data_exp = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp))
                for iord in range(data_inst['nord']):
                    gain_exp[iord] = mean_gdet_func[iord](data_exp['cen_bins'][iord])
                data_vis['mean_gdet_DI_data_paths'][iexp] = data_vis['proc_DI_data_paths']+'mean_gdet_'+str(iexp)
                np.savez_compressed(data_vis['mean_gdet_DI_data_paths'][iexp], data = {'mean_gdet':gain_exp},allow_pickle=True) 
           
    else: 
        for vis in data_inst['visit_list']: 
            data_inst[vis]['cal_data_paths'] = gen_dic['save_data_dir']+'Processed_data/Calibration/'+inst+'_'+vis+'_'
            data_inst[vis]['mean_gdet_DI_data_paths'] = {}
            for iexp in range(data_inst[vis]['n_in_visit']):data_inst[vis]['mean_gdet_DI_data_paths'][iexp] = data_inst[vis]['proc_DI_data_paths']+'mean_gdet_'+str(iexp)
            check_data(data_inst[vis]['mean_gdet_DI_data_paths'],vis=vis)                    

    return None





def model_gain(iexp_glob_groups,iexp_gain_groups,minmax_def,plot_dic,nord,gdet_val_all,inst,gcal_thresh,gcal_edges,gcal_nooutedge,fixed_args,nfree_gainfit,p_start,cal_data_paths):
    r"""**Fit function for instrumental calibration.**

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    data_gain_all = {}
    for iexp_glob,iexp_gain_group in zip(iexp_glob_groups,iexp_gain_groups):
        data_gain={'gdet_inputs':{}}
        if (plot_dic['gcal']!='') or (plot_dic['gcal_ord']!=''):
            data_gain['wav_bin_all']=np.zeros(nord,dtype=object)
            data_gain['wav_trans_all']=np.zeros([2,nord],dtype=float)
            data_gain['cond_fit_all']=np.zeros(nord,dtype=object)
            data_gain['gdet_bin_all']=np.zeros(nord,dtype=object)  
        for iord in range(nord):
            
            #Process order if defined
            fit_done = False
            if gdet_val_all[iexp_glob,iord] is not None:
               
                #Fitting positive values below global threshold
                bin_ord_dic = gdet_val_all[iexp_glob,iord] 
                cond_fit = (bin_ord_dic['gdet']>0.) & (bin_ord_dic['gdet']<gcal_thresh['global']/1e3)
    
                #Remove extreme outliers
                med_prop = np.median(bin_ord_dic['gdet'][cond_fit])
                res = bin_ord_dic['gdet'][cond_fit] - med_prop
                cond_fit[cond_fit][np.abs(res) > 10.*stats.median_abs_deviation(res)] = False  
    
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
                                                          
                    #Temporary fit to identify outliers
                    #    - sigma-clipping applied to the inner parts of the order to prevent excluding edges where calibration can vary sharply
                    #    - negative values are removed
                    #    - weights are used to prevent point with large calibrations (associated with low fluxes) biasing the simple polynomial fit
                    cond_check = deepcopy(cond_fit) 
                    if inst in gcal_nooutedge:cond_check = cond_fit & (bin_ord_dic['cen_bins']>=bin_ord_dic['cen_bins'][0]+gcal_nooutedge[inst][0]) & (bin_ord_dic['cen_bins']<=bin_ord_dic['cen_bins'][-1]-gcal_nooutedge[inst][1]) 
                    if (np.sum(cond_check)>nfree_gainfit):
                        var_fit = bin_ord_dic['gdet'][cond_check]
                        fixed_args['idx_fit'] = np.ones(np.sum(cond_check),dtype=bool)
                        _,merit,_ = call_lmfit(p_start,bin_ord_dic['cen_bins'][cond_check],bin_ord_dic['gdet'][cond_check],np.array([var_fit]),cal_piecewise_func,verbose=False,fixed_args=fixed_args)  
                        res_gdet = bin_ord_dic['gdet'][cond_check] - merit['fit']
                        cond_fit[cond_check] = np.abs(res_gdet)<=gcal_thresh['outliers']*np.std(res_gdet)
    
                        #Model fit
                        #    - errors are scaled with the reduced chi2 from the preliminary fit                           
                        if (np.sum(cond_fit)>nfree_gainfit):
                            fit_done = True
                            var_fit = bin_ord_dic['gdet'][cond_fit]*merit['chi2r'] 
                            fixed_args['idx_fit'] = np.ones(np.sum(cond_fit),dtype=bool)
                            _,merit,p_best = call_lmfit(p_start,bin_ord_dic['cen_bins'][cond_fit],bin_ord_dic['gdet'][cond_fit],np.array([var_fit]),cal_piecewise_func,verbose=False,fixed_args=fixed_args)                                              
                            data_gain['gdet_inputs'][iord] = {'par':deepcopy(p_best),'args':deepcopy(fixed_args)}
    
                #Fit could not be performed
                if not fit_done:
                    data_gain['gdet_inputs'][iord]={'par':None,'args':{'constant':np.median(bin_ord_dic['gdet'])}}
    
                #Save
                if (plot_dic['gcal']!='') or (plot_dic['gcal_ord']!=''):
                    data_gain['wav_bin_all'][iord] = bin_ord_dic['cen_bins']
                    data_gain['wav_trans_all'][:,iord] = [fixed_args['w_lowedge'],fixed_args['w_highedge']]
                    data_gain['cond_fit_all'][iord] = cond_fit
                    data_gain['gdet_bin_all'][iord] = bin_ord_dic['gdet']
    
            #Order fully undefined
            #    - no calibration is applied
            else:
                data_gain['gdet_inputs'][iord]={'par':None,'args':{'constant':1.}}
    
        #Save calibration for each original exposure associated with current exposure group
        data_gain_all[iexp_glob]=data_gain 
        if (plot_dic['gcal']!='') or (plot_dic['gcal_ord']!=''):
            for iexp in iexp_gain_group:np.savez_compressed(cal_data_paths+str(iexp),data=data_gain,allow_pickle=True) 
            
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
        
    return model

