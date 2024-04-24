#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
from PyAstronomy import pyasl
import bindensity as bind
from pathos.multiprocessing import Pool
from ..ANTARESS_conversions.ANTARESS_sp_cont import calc_spectral_cont
from ..ANTARESS_general.utils import stop,np_where1D,init_parallel_func,dataload_npz,MAIN_multithread,gen_specdopshift,check_data

def corr_cosm(inst,gen_dic,data_inst,plot_dic,data_dic,coord_dic):
    r"""**Main cosmics correction routine.**    

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    print('   > Correcting spectra for cosmics')
    
    #Calculating data
    if (gen_dic['calc_cosm']):
        print('         Calculating data') 
        if (inst not in gen_dic['cosm_thresh']):gen_dic['cosm_thresh'][inst]={}

        #Process each visit independently
        for ivisit,vis in enumerate(data_inst['visit_list']):
            data_vis=data_inst[vis]
            if (vis not in gen_dic['cosm_thresh'][inst]):gen_dic['cosm_thresh'][inst][vis] = 10
            
            #RV used for spectra alignment set to keplerian model
            #    - RV relative to CDM are used since the correction is performed per visit, so that the systemic RV does not need to be set                
            if gen_dic['al_cosm']['mode']=='kep':rv_al_all = coord_dic[inst][vis]['RV_star_stelCDM']                
            
            #RV used for spectra alignment set to measured values          
            elif gen_dic['al_cosm']['mode']=='pip':
                if ('RVpip' not in data_dic['DI'][inst][vis]):stop('Pipeline RVs not available')
                rv_al_all = data_dic['DI'][inst][vis]['RVpip']

            #Exposures and orders to be corrected
            if (inst in gen_dic['cosm_exp_corr']) and (vis in gen_dic['cosm_exp_corr'][inst]) and (len(gen_dic['cosm_exp_corr'][inst][vis])>0):exp_corr_list = gen_dic['cosm_exp_corr'][inst][vis]   
            else:exp_corr_list = range(data_vis['n_in_visit']) 
            if (inst in gen_dic['cosm_ord_corr']) and (vis in gen_dic['cosm_ord_corr'][inst]) and (len(gen_dic['cosm_ord_corr'][inst][vis])>0):ord_corr_list = gen_dic['cosm_ord_corr'][inst][vis]   
            else:ord_corr_list = range(data_inst['nord'])   
            
            #Identify defined bins and orders used to cross-correlate and align spectra
            #    - all exposures are processed, even those not to be corrected, since complementary exposures are required 
            edge_bins_all  = np.zeros(list(data_vis['dim_sp'])+[data_vis['nspec']+1], dtype=float)*np.nan
            if (gen_dic['al_cosm']['mode']=='autom'):
                cen_bins_all = np.zeros(data_vis['dim_all'], dtype=float)*np.nan 
                cond_cc  = np.zeros(data_vis['dim_all'], dtype=bool) 
                err2_all = np.zeros(data_vis['dim_all'], dtype=float)*np.nan
            else:
                cen_bins_all=None
                cond_cc=None
                idx_ord_cc=None 
                err2_all=None
            flux_all = np.zeros(data_vis['dim_all'], dtype=float)*np.nan 
            for iexp in range(data_vis['n_in_visit']): 

                #Upload latest processed data
                data_exp = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp))
                edge_bins_all[iexp] = data_exp['edge_bins']
                flux_all[iexp] = data_exp['flux']
                
                #Bins used for alignment
                if gen_dic['al_cosm']['mode']=='autom':
                    cen_bins_all[iexp] = data_exp['cen_bins']
                    if (data_inst['type']=='spec2D'):
                        for iord in range(data_inst['nord']):err2_all[iexp,iord] = data_exp['cov'][iord][0,:]
                    
                    #Defined bins matrix
                    cond_cc[iexp] = data_exp['cond_def']
                    if (len(gen_dic['al_cosm']['range'])>0):
                        cond_range = False
                        for bd_band_loc in gen_dic['al_cosm']['range']:cond_range|=(data_exp['edge_bins'][:,0:-1]>bd_band_loc[0]) & (data_exp['edge_bins'][:,1::]<bd_band_loc[1])
                        cond_cc[iexp] &= cond_range

            #Selected orders common to all exposures
            if (gen_dic['al_cosm']['mode']=='autom') and (len(gen_dic['al_cosm']['range'])>0):idx_ord_cc = np_where1D(np.sum(cond_cc,axis=(0,2),dtype=bool)) 

            #Number of complementary exposures used for comparison  
            cosm_ncomp = int(np.min( [ np.ceil(gen_dic['cosm_ncomp'] / 2.)*2. , data_vis['n_in_visit'] -1 ]))        
            hcosm_ncomp = int(cosm_ncomp/2)  
            
            #Search and correct for cosmics in each exposure 
            proc_DI_data_paths_new = gen_dic['save_data_dir']+'Corr_data/Cosm/'+inst+'_'+vis+'_'
            
            #Processing all exposures    
            common_args = (data_vis['proc_DI_data_paths'],hcosm_ncomp,data_vis['n_in_visit'],cosm_ncomp,data_vis['dim_all'],gen_dic['al_cosm'],idx_ord_cc,cen_bins_all,cond_cc,flux_all,\
                           gen_dic['pix_size_v'][inst],data_inst['type'],rv_al_all,err2_all,edge_bins_all,ord_corr_list,gen_dic['resamp_mode'],data_vis['dim_exp'],data_vis['nspec'],gen_dic['cosm_thresh'][inst][vis],plot_dic['cosm_corr'],proc_DI_data_paths_new)
            if gen_dic['cosm_nthreads']>1:MAIN_multithread(corr_cosm_vis,gen_dic['cosm_nthreads'],len(exp_corr_list),[exp_corr_list],common_args)                           
            else:corr_cosm_vis(exp_corr_list,*common_args)  
            data_vis['proc_DI_data_paths'] = proc_DI_data_paths_new

    #Updating path to processed data and checking it has been calculated
    else:
        for vis in data_inst['visit_list']:  
            data_vis=data_inst[vis]
            data_vis['proc_DI_data_paths']=gen_dic['save_data_dir']+'Corr_data/Cosm/'+inst+'_'+vis+'_'  
            check_data({'path':data_vis['proc_DI_data_paths']+str(0)},vis=vis)                

    return None    


def corr_cosm_vis(iexp_group,proc_DI_data_paths,hcosm_ncomp,n_in_visit,cosm_ncomp,dim_all,al_cosm,idx_ord_cc,cen_bins_all,cond_cc,flux_all,pix_size_v,data_type,rv_al_all,err2_all,edge_bins_all,ord_corr_list,\
                  resamp_mode,dim_exp,nspec,cosm_thresh,plot_cosm_corr,proc_DI_data_paths_new):
    r"""**Cosmics correction routine per visit.**    

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    for iexp in iexp_group:
        data_exp = dataload_npz(proc_DI_data_paths+str(iexp))
        
        #Indexes of complementary exposures to current exposures
        #    - limited to the requested number
        #    - indexes falling before (resp. after) the time-series are replaced by exposures after (resp. before) the last (resp. the first) complementary one
        idx_around = np.append(np.arange(iexp - hcosm_ncomp , iexp) , np.arange( iexp+1 , iexp + hcosm_ncomp + 1) )
        isub_low = (idx_around<0)
        isub_high = (idx_around>n_in_visit-1)
        if True in isub_low:idx_around[isub_low]+=(cosm_ncomp+1)
        if True in isub_high:idx_around[isub_high]-=(cosm_ncomp+1)
        dim_comp = deepcopy(dim_all)
        dim_comp[0] = len(idx_around)
        
        #Align complementary exposures with current exposure
        #    - we repeat the alignement of complementary exposures for each exposure so as to not interpolate it
        flux_align_comp = np.zeros(dim_comp, dtype=float)
        for iloc,iexp_c in enumerate(idx_around):

            #Determine radial velocity shift relative to current exposure                    
            if (al_cosm['mode']=='autom'):
                rv_shift = 0.
                twrv_ord = 0.
                rv_cc_ord_HR = np.linspace(al_cosm['RVrange_cc'][0],al_cosm['RVrange_cc'][1], 100)
                for iord in idx_ord_cc:
                
                    #Cross-correlate with current spectrum
                    rv_cc_ord, cc_func_ord = pyasl.crosscorrRV(cen_bins_all[iexp_c,iord,cond_cc[iexp_c,iord]],flux_all[iexp_c,iord,cond_cc[iexp_c,iord]], data_exp['cen_bins'][iord,cond_cc[iexp,iord]] , data_exp['flux'][iord,cond_cc[iexp,iord]], al_cosm['RVrange_cc'][0], al_cosm['RVrange_cc'][1], al_cosm['RVrange_cc'][2] , skipedge = int( max( [abs(al_cosm['RVrange_cc'][0]) , abs(al_cosm['RVrange_cc'][1])] )/pix_size_v)+1    )# ,skipedge = int(10/gen_dic['pix_size_v'][inst])+10)
                        
                    #Fit polynomial and retrieve position of cross-correlation maximum
                    rv_shift_ord = rv_cc_ord_HR[np.argmax(np.poly1d(np.polyfit(rv_cc_ord, cc_func_ord, 4 ))(rv_cc_ord_HR))]
                
                    #Weights
                    #    - for 2D spectra with several orders we define the RV shift as the weighted mean of RV shift measured in the orders containing the requested spectral range
                    #      we use the mean error over defined pixels in each order to set weights
                    if (data_type=='spec2D'):  
                        wrv_ord = 1./np.mean(err2_all[iexp_c,iord,cond_cc[iexp_c,iord]])                          
                        rv_shift+=rv_shift_ord*wrv_ord
                        twrv_ord+=wrv_ord
                    else:
                        rv_shift+=rv_shift_ord 
                    
                #Weighted rv shift
                if (data_type=='spec2D'):rv_shift/=twrv_ord                                

            #Set to known value
            else:
                rv_shift = rv_al_all[iexp_c]-rv_al_all[iexp]                        
                
            #Align complementary exposures with current spectrum
            #    - in orders to be corrected for only
            edge_bins_new = edge_bins_all[iexp_c]/gen_specdopshift(rv_shift)  
            for iord in ord_corr_list: 
                flux_align_comp[iloc,iord] = bind.resampling(edge_bins_all[iexp,iord], edge_bins_new[iord], flux_all[iexp_c,iord], kind=resamp_mode)
            
        #Process each order
        com_def_pix = np.all(~np.isnan(flux_align_comp),axis=0) &  data_exp['cond_def']       #common defined pixels over all exposures
        dcen_bins_exp = edge_bins_all[iexp,:,1::] - edge_bins_all[iexp,:,0:-1]
        SNR_diff_exp = np.zeros([2]+dim_exp,dtype=float)*np.nan 
        idx_cosm_exp = {}
        for iord in ord_corr_list: 
 
            #Set complementary exposures to the flux level of processed exposure
            #    - even if the color balance was corrected for, spectra were left to their overall flux level                    
            com_def_pix_ord = com_def_pix[iord]
            com_flux_comp = np.sum(flux_align_comp[:,iord,com_def_pix_ord]*dcen_bins_exp[iord,com_def_pix_ord],axis=1)
            com_flux_proc = np.sum(data_exp['flux'][iord,com_def_pix_ord]*dcen_bins_exp[iord,com_def_pix_ord])
            flux_align_comp[:,iord]*=(com_flux_proc/com_flux_comp[:,None])       

            #Mean spectrum over aligned complementary exposures and standard-deviation
            #    - for each bin with at least two defined complementary exposures, and defined in the current exposure
            mean_sp_loc = np.zeros(nspec, dtype=float)*np.nan
            std_sp_loc = np.zeros(nspec, dtype=float)*np.nan
            cond_def_exp = data_exp['cond_def'][iord] & (np.sum(~np.isnan(flux_align_comp[:,iord]),axis=0)>1)
            if True in cond_def_exp:                         
                mean_sp_loc[cond_def_exp] = np.nanmean(flux_align_comp[:,iord,cond_def_exp],axis=0)
                std_sp_loc[cond_def_exp] = np.nanstd(flux_align_comp[:,iord,cond_def_exp],axis=0)
                
                #Relative difference exposure-master with respect to dispersion of spectra in the night and to current spectrum error
                diff_loc_def = (data_exp['flux'][iord,cond_def_exp] - mean_sp_loc[cond_def_exp])
                SNR_diff_exp[0,iord,cond_def_exp] = diff_loc_def/std_sp_loc[cond_def_exp]
                SNR_diff_exp[1,iord,cond_def_exp] = diff_loc_def/np.sqrt(data_exp['cov'][iord][0,cond_def_exp])

                #Identifying and replacing cosmics
                #    - we flag a bin as affected by a cosmic if its flux deviates from the mean over complementary exposures by more than a threshold
                # + times the standard-deviation of the mean
                # + times the error on the bin flux
                #      this second condition is to account for the current spectrum being much noiser than the complementary exposures
                #    - we attribute the mean and standard deviation to the outlying bins
                #      only the variance is modified, not the covariance (at this stage the covariance matrix should still be 1D)
                cond_cosm_def = (SNR_diff_exp[0,iord,cond_def_exp]>cosm_thresh) & (SNR_diff_exp[1,iord,cond_def_exp]>cosm_thresh) 
                if True in cond_cosm_def:
                    idx_cosm_exp[iord] = np_where1D(cond_def_exp)[cond_cosm_def]     
                    data_exp['flux'][iord,idx_cosm_exp[iord]] = mean_sp_loc[idx_cosm_exp[iord]]
                    data_exp['cov'][iord][0,idx_cosm_exp[iord]] = std_sp_loc[idx_cosm_exp[iord]]**2.                            

        #Save independently correction data
        #    - too heavy to be saved for all exposures
        if (plot_cosm_corr!=''): 
            dic_sav = {'SNR_diff_exp':SNR_diff_exp,'idx_cosm_exp':idx_cosm_exp}
            np.savez_compressed(proc_DI_data_paths_new+str(iexp)+'_add',data=dic_sav,allow_pickle=True)

        #Saving modified data and update paths
        np.savez_compressed(proc_DI_data_paths_new+str(iexp),data = data_exp,allow_pickle=True)     

    return None







def MAIN_permpeak(inst,gen_dic,data_inst,plot_dic,data_dic,data_prop):
    r'''**Main persistent peak routine.**
    
    Identifies and masks spectra for positive features that persists over time.
    
     - this routine identifies and masks hot pixels and telluric lines in emission 
     - bad pixels on the detector and telluric lines are aligned in the Earth rest frame, so that exposures are compared in this referential
     - we use an estimate of the continuum to identify spurious features, using RASSINE approach (Cretignier+2020, A&A, 640, A42)
    
    Args:
        TBD:
    
    Returns:
        TBD:
            
    '''
    print('   > Masking persistent peaks in spectra')
    
    #Calculating data
    if (gen_dic['calc_permpeak']):
        print('         Calculating data')    

        #Excluded ranges
        if (inst in gen_dic['permpeak_range_corr']) and (len(gen_dic['permpeak_range_corr'][inst])>0):range_proc_ord = list(gen_dic['permpeak_range_corr'][inst].keys())
        else:range_proc_ord=list(range(data_inst['nord']) )
        if (inst in gen_dic['permpeak_edges']):permpeak_edges = gen_dic['permpeak_edges'][inst]
        else:permpeak_edges = [0.,0.]

        #Process each visit independently
        for ivisit,vis in enumerate(data_inst['visit_list']):
            data_vis=data_inst[vis]
            
            #Exposures and orders to be corrected
            if (inst in gen_dic['permpeak_exp_corr']) and (vis in gen_dic['permpeak_exp_corr'][inst]) and (len(gen_dic['permpeak_exp_corr'][inst][vis])>0):exp_corr_list = gen_dic['permpeak_exp_corr'][inst][vis]   
            else:exp_corr_list = list(range(data_vis['n_in_visit']))
            nexp_corr_list=len(exp_corr_list)
            if (inst in gen_dic['permpeak_ord_corr']) and (vis in gen_dic['permpeak_ord_corr'][inst]) and (len(gen_dic['permpeak_ord_corr'][inst][vis])>0):ord_corr_list = gen_dic['permpeak_ord_corr'][inst][vis]   
            else:ord_corr_list = list(range(data_inst['nord']) )
            nord_corr_list = len(ord_corr_list)
            
            #Common visit table
            #    - shifted from the solar barycentric (receiver) to the average Earth (source) rest frame for the visit
            data_com = dataload_npz(data_vis['proc_com_data_paths'])
            sp_shift = 1./(gen_specdopshift(np.mean(data_prop[inst][vis]['BERV']))*(1.+1.55e-8))      
            edge_bins_com_Earth = data_com['edge_bins']*sp_shift  
            cen_bins_com_Earth = data_com['cen_bins']*sp_shift

            #Retrieve all exposures for master calculation
            dim_all = [data_vis['n_in_visit'],nord_corr_list,data_com['nspec']]
            flux_Earth_all = np.zeros(dim_all, dtype=float)*np.nan 
            cond_def_all = np.zeros(dim_all, dtype=bool)
            for iexp in range(data_vis['n_in_visit']): 

                #Upload latest processed data
                data_exp = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp))

                #Align exposure in Earth referential for orders to be corrected
                edge_bins_Earth = data_exp['edge_bins']/(gen_specdopshift(data_prop[inst][vis]['BERV'][iexp])*(1.+1.55e-8))   
                for isub_ord,iord in enumerate(ord_corr_list): 
                    flux_Earth_all[iexp,isub_ord] = bind.resampling(edge_bins_com_Earth[iord], edge_bins_Earth[iord], data_exp['flux'][iord], kind=gen_dic['resamp_mode'])
                cond_def_all[iexp] = ~np.isnan(flux_Earth_all[iexp])                    
                
            #Order x pixels where at least one exposure has a defined bin                      
            cond_def_mast = (np.sum(cond_def_all,axis=0)>0)                 
                
            #Save for plotting
            if (plot_dic['permpeak_corr']!='') or (plot_dic['sp_raw']!=''):
                dic_sav = {
                    'tot_Fr_all':np.ones(data_vis['dim_sp'],dtype=float),
                    'count_bad_all':np.zeros( [data_vis['n_in_visit'],data_inst['nord']] , dtype=int)}

            #Calculate stellar continuum of master spectrum
            mean_flux_mast,cont_func_dic,dic_sav = calc_spectral_cont(data_inst['nord'],ord_corr_list,None,cen_bins_com_Earth[ord_corr_list],edge_bins_com_Earth[ord_corr_list],cond_def_mast,flux_Earth_all,cond_def_all,inst,gen_dic['contin_roll_win'][inst]  ,\
                                                             gen_dic['contin_smooth_win'][inst],gen_dic['contin_locmax_win'][inst],gen_dic['contin_stretch'][inst],gen_dic['contin_pinR'][inst],data_com['min_edge_ord'][0],dic_sav,gen_dic['permpeak_nthreads'])

            #Flag pixels with spurious positive flux
            common_args = (nord_corr_list,data_vis['nspec'],data_vis['proc_DI_data_paths'],ord_corr_list,permpeak_edges,inst,gen_dic['permpeak_range_corr'],gen_dic['permpeak_outthresh'],gen_dic['permpeak_peakwin'][inst],range_proc_ord,mean_flux_mast,data_inst['nord'],cont_func_dic)
            if gen_dic['permpeak_nthreads']>1:cond_bad_all,cond_undef_all,tot_Fr_all=multithread_permpeak_flag(permpeak_flag,gen_dic['permpeak_nthreads'],len(exp_corr_list),[exp_corr_list,data_prop[inst][vis]['BERV'][exp_corr_list]],common_args)                           
            else:cond_bad_all,cond_undef_all,tot_Fr_all=permpeak_flag(exp_corr_list,data_prop[inst][vis]['BERV'][exp_corr_list],*common_args) 
            if (plot_dic['permpeak_corr']!='') or (plot_dic['sp_raw']!=''):
                dic_sav['tot_Fr_all'][exp_corr_list] = tot_Fr_all
                dic_sav['cont_func_dic'] = cont_func_dic

            #Mask persistent pixels with spurious positive flux
            nexp_bad = max(gen_dic['permpeak_nbad'],3)
            cond_mask_all = np.zeros(dim_all,dtype=bool)
            for isub_ord in range(nord_corr_list):    

                #Mask permanently bad pixels that are not undefined in all exposures
                cond_undef_ord = (np.sum(cond_undef_all[:,isub_ord],axis=0)<nexp_corr_list)
                count_bad_ord = np.sum(cond_bad_all[:,isub_ord],axis=0)
                cond_bad_pix = (count_bad_ord==nexp_corr_list) & cond_undef_ord
                if (True in cond_bad_pix):cond_mask_all[:,isub_ord,cond_bad_pix] = True 

                #Process remaining pixels
                #    - bad in at least one exposure and not undefined in all exposures 
                for ipix in np_where1D((count_bad_ord>0) & (count_bad_ord<nexp_corr_list) & cond_undef_ord):
                    isub_exp = 0
                    while (isub_exp<nexp_corr_list-nexp_bad):
                        
                        #Pixel is bad 
                        if cond_bad_all[isub_exp,isub_ord,ipix]:
                            nsub = 0
                            cond_loc = True
                            while cond_loc and (isub_exp+nsub+1<nexp_corr_list):
                                nsub+=1
                                
                                #Undefined pixels are counted as potentially bad
                                cond_loc = cond_bad_all[isub_exp+nsub,isub_ord,ipix]| cond_undef_all[isub_exp+nsub,isub_ord,ipix]
                     
                            #Pixel is bad in more than nexp_bad consecutive exposures
                            if nsub+1>=nexp_bad:cond_mask_all[isub_exp:isub_exp+nsub+1,isub_ord,ipix] = True
 
                            #Move to next exposures
                            isub_exp+=nsub+1
                            
                        #Moving to next exposure
                        else:isub_exp+=1

            #Apply masking
            proc_DI_data_paths_new = gen_dic['save_data_dir']+'Corr_data/Permpeak/'+inst+'_'+vis+'_'
            common_args = (data_vis['proc_DI_data_paths'],proc_DI_data_paths_new,ord_corr_list)
            if gen_dic['permpeak_nthreads']>1:MAIN_multithread(permpeak_mask,gen_dic['permpeak_nthreads'],len(exp_corr_list),[exp_corr_list,cond_mask_all],common_args)                           
            else:permpeak_mask(exp_corr_list,cond_mask_all,*common_args)  
            data_vis['proc_DI_data_paths'] = proc_DI_data_paths_new            
            
            #Number of masked pixels in each order
            if (plot_dic['permpeak_corr']!='') or (plot_dic['sp_raw']!=''):
                for isub_exp,iexp in enumerate(exp_corr_list):
                    dic_sav['count_bad_all'][iexp,ord_corr_list] = np.sum(cond_mask_all[isub_exp],axis=1)
                    
            #update paths
            data_vis['proc_DI_data_paths'] = proc_DI_data_paths_new

            #Save for plotting
            if (plot_dic['permpeak_corr']!='') or (plot_dic['sp_raw']!=''):
                np.savez_compressed(gen_dic['save_data_dir']+'Corr_data/Permpeak/'+inst+'_'+vis+'_add',data=dic_sav,allow_pickle=True)

    #Updating path to processed data and checking it has been calculated
    else:
        for vis in data_inst['visit_list']:  
            data_vis=data_inst[vis]
            data_vis['proc_DI_data_paths']=gen_dic['save_data_dir']+'Corr_data/Permpeak/'+inst+'_'+vis+'_'  
            check_data({'path':data_vis['proc_DI_data_paths']+str(0)},vis=vis)                

    return None    



def multithread_permpeak_flag(func_input,nthreads,n_elem,y_inputs,common_args):    
    r"""**Multithreading of permpeak_flag().**
    
    Specific implementation of MAIN_multithread() 

    Args:
        func_input (function): multi-threaded function
        nthreads (int): number of threads
        n_elem (int): number of elements to thread
        y_inputs (list): threadable function inputs 
        common_args (tuple): common function inputs
    
    Returns:
        y_output (specific to func_input): function outputs 
    
    """ 
    pool_proc = Pool(processes=nthreads)   #cannot be passed through lmfit
    ind_chunk_list=init_parallel_func(nthreads,n_elem)
    chunked_args=[tuple(y_inputs[i][ind_chunk[0]:ind_chunk[1]] for i in range(len(y_inputs)))+common_args for ind_chunk in ind_chunk_list]	
    all_results=tuple(tab for tab in pool_proc.starmap(func_input,chunked_args))	
    y_output=(np.concatenate(tuple(all_results[i][0] for i in range(nthreads)),axis=0),np.concatenate(tuple(all_results[i][1] for i in range(nthreads)),axis=0),np.concatenate(tuple(all_results[i][2] for i in range(nthreads)),axis=0))   
    pool_proc.close()
    pool_proc.join() 				
    return y_output



def permpeak_flag(iexp_group,BERV_group,nord_corr_list,nspec,proc_DI_data_paths,ord_corr_list,permpeak_edges,inst,permpeak_range_corr,permpeak_outthresh,permpeak_peakwin,range_proc_ord,mean_flux_mast,nord,cont_func_dic):
    r'''**Persistent peak flagging.**
    
    Flags pixels with spurious positive flux
    
    Args:
        TBD:
    
    Returns:
        TBD:
            
    '''
    dim_all = [len(iexp_group),nord_corr_list,nspec]
    cond_bad_all = np.zeros(dim_all,dtype=bool)
    cond_undef_all = np.zeros(dim_all,dtype=bool)
    tot_Fr_all = np.ones([len(iexp_group),nord],dtype=float)
    for isub_exp,iexp in enumerate(iexp_group):                 
            
        #Upload latest processed data
        data_exp = dataload_npz(proc_DI_data_paths+str(iexp))

        #Shift exposure from the solar barycenter (receiver) to the Earth (source) rest frame
        #    - see gen_specdopshift():
        # w_source = w_receiver / (1+ (rv[s/r]/c))
        # w_Earth = w_solbar / (1+ (BERV/c))      
        sp_shift = 1./(gen_specdopshift(BERV_group[isub_exp])*(1.+1.55e-8))   
        cen_bins_Earth = data_exp['cen_bins']*sp_shift
        edge_bins_Earth = data_exp['edge_bins']*sp_shift

        #Process each order
        for isub_ord,iord in enumerate(ord_corr_list):
            cond_def_ord = data_exp['cond_def'][iord] 
            
            #Excluding edges
            cond_def_ord &= ((edge_bins_Earth[iord,0:-1]>=edge_bins_Earth[iord,0:-1][cond_def_ord][0]+permpeak_edges[0]) & (edge_bins_Earth[iord,1:]<=edge_bins_Earth[iord,1:][cond_def_ord][-1]-permpeak_edges[1])   ) 
     
            #Store undefined pixels
            cond_undef_all[isub_exp,isub_ord,~cond_def_ord] = True
       
            #Processed range
            cond_proc_ord  =cond_def_ord
            if (inst in permpeak_range_corr) and (iord in range_proc_ord):
                cond_range  = False
                for bd_int in permpeak_range_corr[inst][iord]:
                    cond_range |= (edge_bins_Earth[iord,0:-1]>=bd_int[0]) & (edge_bins_Earth[iord,1:]<=bd_int[1])   
                cond_proc_ord  &=cond_range

            #Compute continuum over current spectrum table
            #    - reset to the flux level of current exposure (even if the color balance was corrected for, spectra were left to their overall flux level  )
            dcen_bins = edge_bins_Earth[iord,1::][cond_proc_ord] - edge_bins_Earth[iord,0:-1][cond_proc_ord]
            mean_flux_ord = np.sum(data_exp['flux'][iord,cond_proc_ord]*dcen_bins)/np.sum(dcen_bins)
            tot_Fr_all[isub_exp,iord] = mean_flux_ord/mean_flux_mast[iord]
            cont_ord = cont_func_dic[iord](cen_bins_Earth[iord,cond_proc_ord])*tot_Fr_all[isub_exp,iord]

            #Residuals to continuum
            res_cont_ord = data_exp['flux'][iord,cond_proc_ord] - cont_ord

            #Identify pixels deviating from continuum beyond threshold and chosen range around them
            #    - only for positive deviations, or it would flag telluric and stellar absorption lines
            cond_bad_ord = (res_cont_ord > permpeak_outthresh*np.sqrt(data_exp['cov'][iord][0,cond_proc_ord]))        
            hn_exc = int(np.round(0.5*(permpeak_peakwin/np.mean(dcen_bins))))
            cond_badrange_ord = deepcopy(cond_bad_ord)
            for j in range(1,hn_exc):cond_badrange_ord  |= np.roll(cond_bad_ord,-j) | np.roll(cond_bad_ord,j)
            cond_bad_all[isub_exp,isub_ord,np_where1D(cond_proc_ord)[cond_badrange_ord]] = True 

    return cond_bad_all,cond_undef_all,tot_Fr_all


def permpeak_mask(iexp_group,cond_mask_all,proc_DI_data_paths,proc_DI_data_paths_new,ord_corr_list):
    r'''**Persistent peak masking.**
    
    Masks pixels with spurious positive flux
    
    Args:
        TBD:
    
    Returns:
        TBD:
            
    '''
    for isub_exp,iexp in enumerate(iexp_group):
        data_exp = dataload_npz(proc_DI_data_paths+str(iexp))
        
        #Mask bad pixels
        for isub_ord,iord in enumerate(ord_corr_list):data_exp['flux'][iord,cond_mask_all[isub_exp,isub_ord]] = np.nan
        data_exp['cond_def'] = ~np.isnan(data_exp['flux'])
        
        #Saving modified data
        np.savez_compressed(proc_DI_data_paths_new+str(iexp),data = data_exp,allow_pickle=True) 

    return None




