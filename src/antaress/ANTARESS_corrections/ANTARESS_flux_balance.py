#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import numpy as np
from copy import deepcopy
import bindensity as bind
from scipy.interpolate import UnivariateSpline
from scipy import stats
from ..ANTARESS_general.constant_data import c_light
from ..ANTARESS_conversions.ANTARESS_binning import sub_calc_bins,sub_def_bins
from ..ANTARESS_general.utils import stop,np_where1D,dataload_npz,MAIN_multithread,gen_specdopshift,def_edge_tab,check_data


def def_Mstar(gen_dic,data_inst,inst,data_prop,plot_dic,data_dic,coord_dic):
    """**Stellar master definition.**
    
    Defines master spectra for the star
    
     - used exclusively for flux balance corrections
     - either set to the mean or median of the spectra over visits, or to an input spectrum 
     - the module is ran independently and the master saved for each exposure, so that it can be used independently for the global and local flux balance corrections

    Args:
        TBD
    
    Returns:
        None
    
    """ 
    print('   > Calculating stellar masters')   

    #Calculating data
    if (gen_dic['calc_glob_mast']):
        print('         Calculating data')    

        #Common instrument table, defined in input rest frame
        #    - used to define masters (if a single visit is processed this table will match the visit-specific common table)
        data_com_inst = dataload_npz(data_inst['proc_com_data_path']) 
        wav_Mstar = data_com_inst['cen_bins']
        edge_wav_Mstar = data_com_inst['edge_bins']  
        dim_exp_mast = data_inst['dim_exp']
        Mstar_vis_all = np.zeros([gen_dic[inst]['n_visits']]+dim_exp_mast,dtype=float)*np.nan 
        
        #Calculating visit-specific masters 
        #    - the master can either be calculated using a mean, or a median to avoid introducing spurious variations such as cosmics
        #      the absolute flux level of the master does not matter, as it is used to correct for the relative flux balance correction  
        idx_ord_def_vis = np.zeros(gen_dic[inst]['n_visits'],dtype=object) 
        for ivisit,vis in enumerate(data_inst['visit_list']):
            data_vis = data_inst[vis]

            #Doppler shift used for spectra alignment in the star rest frame (see below)
            #    - beware that systemic RV should be defined or similar between visits if a master common to all visits is used
            specdopshift_star_solbar = 1./(gen_specdopshift(coord_dic[inst][vis]['RV_star_stelCDM'])*gen_specdopshift(data_dic['DI']['sysvel'][inst][vis]))
         
            #Exposure selection    
            if (inst in gen_dic['glob_mast_exp']) and (vis in gen_dic['glob_mast_exp'][inst]) and (gen_dic['glob_mast_exp'][inst][vis]!='all'):
                iexp_mast = gen_dic['glob_mast_exp'][inst][vis]
            else:iexp_mast = range(data_vis['n_in_visit'])
            
            #Processing selected exposures
            flux_vis = np.zeros([len(iexp_mast)]+dim_exp_mast,dtype=float)*np.nan 
            cond_def_vis = np.zeros([len(iexp_mast)]+dim_exp_mast,dtype=bool)
            for isub_exp,iexp in enumerate(iexp_mast):  

                #Upload latest processed data
                data_exp = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp))
                
                #Shifting data from the input rest frame (assumed to be the solar or shifted-solar CDM, receiver) to the star rest frame (source) and resampling on common table
                #    - see gen_specdopshift():
                # w_source = w_receiver / (1+ (rv[s/r]/c))
                # w_star = w_starbar/ (1+ (rv[star/starbar]/c))
                #        = w_solbar / ((1+ (rv[star/starbar]/c)) * (1+ (rv[starbar/solbar]/c)))
                #        = w_solbar / ((1+ (rv_kep/c)) * (1+ (rv_sys/c))) 
                edge_bins_rest = data_exp['edge_bins']*specdopshift_star_solbar[iexp]
                for iord in range(data_inst['nord']): 
                    flux_vis[isub_exp][iord] = bind.resampling(edge_wav_Mstar[iord], edge_bins_rest[iord],data_exp['flux'][iord], kind=gen_dic['resamp_mode'])
                cond_def_vis[isub_exp] = ~np.isnan(flux_vis[isub_exp])

            #Indexes where at least one exposure has a defined bin 
            #    - per order for 2D spectra                       
            cond_def_Mstar_vis = (np.sum(cond_def_vis,axis=0)>0) 

            #Visit master calculated over defined bins
            #    - the co-addition is done over defined bins, to leave at nan those bins where no exposure is defined (otherwise np.nansum returns 0 if only nan are summed)
            #    - bins on the edges of the 1D master over all visits might be undefined if nigthly masters were defined on different ranges  
            #    - we neglect possible changes in instrumental calibration over time, tellurics, and color balance since they are not corrected for yet 
            if gen_dic['glob_mast_mode']=='mean':      

                #Calculate mean of spectra
                #    - the use of the mean naturally gives a stronger weight to spectra with larger flux (and SNR), but it might be biased by strong spurious features
                #    - we do not use weights, as in the present case they would compensate in the weighted mean
                #      after normalizing spectra to the same global flux level, the correct weights would be (see weights_bin_prof):
                # wi = 1/ci, with Fi_norm = ci*Fi 
                #      then Fmast = sum( Fi_norm*wi )/sum(wi) = sum( ci*Fi/ci )/sum(1/ci) = sum(Fi)/sum(ci)
                #      since the overal level of the master does not matter, we can thus simply calculate a mean of the unweighted flux
                for iord in range(data_inst['nord']):
                    for ipix in np_where1D(cond_def_Mstar_vis[iord]):Mstar_vis_all[ivisit][iord,ipix] = np.mean(flux_vis[cond_def_vis[:,iord,ipix],iord,ipix])  
                    
            elif gen_dic['glob_mast_mode']=='med':
                
                #Calculate median of spectra                    
                #     - the use of the median allows mitigating the impact of such features (such as cosmic rays), but it requires that the spectra have comparable flux levels
                #       we apply an empirical normalization to account for all possible source of variations between exposures (duration, stellar variability, earth diffusion, ..)
                for iord in range(data_inst['nord']):

                    #Set each order to same flux level in all exposures, set to average over the visit
                    mean_flux_exp = np.nanmean(flux_vis[:,iord],axis=1)
                    flux_vis[:,iord]*=np.mean(mean_flux_exp)/mean_flux_exp[:,None]
 
                    #Calculate median flux
                    for ipix in np_where1D(cond_def_Mstar_vis[iord]):Mstar_vis_all[ivisit][iord,ipix] = np.median(flux_vis[cond_def_vis[:,iord,ipix],iord,ipix])

            #Defined orders for visit master
            #    - orders might be fully undefined if few bins were defined and spectral tables were different between visits
            #    - we do not remove empty orders to keep the same structure as individual visits
            if data_inst['type']=='spec1D': idx_ord_def_vis[ivisit]=[0]
            elif data_inst['type']=='spec2D':idx_ord_def_vis[ivisit] = np_where1D( np.sum(cond_def_Mstar_vis,axis=1,dtype=bool) > 0)

            #Saving visit-specific master
            if (plot_dic['glob_mast']!='') or (gen_dic[inst]['n_visits']>1):
                np.savez_compressed(gen_dic['save_data_dir']+'Corr_data/Global_Master/'+inst+'_'+vis+'_meas',data = {'cen_bins':wav_Mstar,'flux':Mstar_vis_all[ivisit] ,'cond_def':cond_def_Mstar_vis,'proc_DI_data_paths':data_vis['proc_DI_data_paths'],'specdopshift_star_solbar':specdopshift_star_solbar},allow_pickle=True) 

            #-------------------------------------------------------------------------------------------

            #Shifting and resampling master over the table of each exposure
            #    - to minimize biases in the exposure-to-master ratios 
            #    - even if all exposures and their master are defined over a common table, the master must be resampled after being shifted
            data_mast_exp = {}
            for iexp in range(data_vis['n_in_visit']):

                #Upload latest processed data
                data_exp = np.load(data_vis['proc_DI_data_paths']+str(iexp)+'.npz',allow_pickle=True)['data'].item()     

                #Shifting master from the star rest frame to the exposure rest frame (applying the opposite shift as for the master calculation)
                data_mast_exp['flux'] = np.zeros(data_vis['dim_exp'])*np.nan
                edge_bins_rest = edge_wav_Mstar/specdopshift_star_solbar[iexp]   
                
                #Resampling
                for iord in idx_ord_def_vis[ivisit]:data_mast_exp['flux'][iord] = bind.resampling(data_exp['edge_bins'][iord], edge_bins_rest[iord],Mstar_vis_all[ivisit][iord], kind=gen_dic['resamp_mode'])
                data_mast_exp['cond_def'] = ~np.isnan(data_mast_exp['flux'])                      

                #Saving master for current exposure
                np.savez_compressed(gen_dic['save_data_dir']+'Corr_data/Global_Master/'+inst+'_'+vis+'_'+str(iexp),data = data_mast_exp,allow_pickle=True) 

            ### End of exposures 

            #Theoretical reference master
            #    - two columns: wavelength in star rest frame in A, flux density in arbitrary units  
            if (gen_dic['Fbal_vis']=='theo'):

                #Retrieving input spectrum specific to the visit
                wav_Mstar_theo,Mstar_theo=np.loadtxt(glob.glob(gen_dic['Fbal_refFstar'][inst][vis])[0]).T
                edge_wav_Mstar_theo = def_edge_tab(wav_Mstar_theo,dim=0)
                
                #Check for theoretical spectrum to encompass the full visit range
                min_def_wav_Mstar = np.nanmin(edge_wav_Mstar[0:-1][cond_def_Mstar_vis])
                max_def_wav_Mstar = np.nanmax(edge_wav_Mstar[1::][cond_def_Mstar_vis])
                if (edge_wav_Mstar_theo[0]>min_def_wav_Mstar) or (edge_wav_Mstar_theo[-1]<max_def_wav_Mstar):
                    stop('Extend range of definition of theoretical spectrum to cover visit '+vis)
                
                #Resampling on table of current visit-specific master
                Mstar_ref = np.zeros(dim_exp_mast)*np.nan
                for iord in range(data_inst['nord']): 
                    Mstar_ref[iord] = bind.resampling(edge_wav_Mstar[iord],edge_wav_Mstar_theo,Mstar_theo, kind=gen_dic['resamp_mode']) 
         
                #Saving
                np.savez_compressed(gen_dic['save_data_dir']+'Corr_data/Global_Master/'+inst+'_'+vis+'_theo',data = {'edge_bins':edge_wav_Mstar,'cen_bins':wav_Mstar,'flux':Mstar_ref},allow_pickle=True) 
    
        ### End of visits  

        #Instrument reference master
        #    - all masters may not have the same flux balance, and are thus only co-added at bins where all masters are defined to prevent introducing flux distortions
        #    - if required for multi-visit scaling
        if (gen_dic['Fbal_vis']=='meas') and (gen_dic[inst]['n_visits']>1):
            cond_def_Mstar_vis_all = ~np.isnan(Mstar_vis_all)
            Mstar_ref = np.zeros(dim_exp_mast)*np.nan
            cond_def_Mast = np.zeros(dim_exp_mast,dtype=bool)
            for iord in range(data_inst['nord']):  
                cond_def_Mast[iord] = (np.sum(cond_def_Mstar_vis_all[:,iord,:],axis=0)==gen_dic[inst]['n_visits']) 
                if gen_dic['glob_mast_mode']=='mean':  Mstar_ref[iord,cond_def_Mast[iord]] = np.mean(Mstar_vis_all[:,iord,cond_def_Mast[iord]],axis=0)
                elif gen_dic['glob_mast_mode']=='med': Mstar_ref[iord,cond_def_Mast[iord]] = np.median(Mstar_vis_all[:,iord,cond_def_Mast[iord]],axis=0)
            np.savez_compressed(gen_dic['save_data_dir']+'Corr_data/Global_Master/'+inst+'_meas',data = {'edge_bins':edge_wav_Mstar,'cen_bins':wav_Mstar,'flux':Mstar_ref,'cond_def':cond_def_Mast},allow_pickle=True) 

    #Defining paths and checking data were calculated
    else:
        for vis in data_inst['visit_list']:  
            check_data({iexp:gen_dic['save_data_dir']+'Corr_data/Global_Master/'+inst+'_'+vis+'_'+str(iexp) for iexp in range(data_inst[vis]['n_in_visit'])},vis=vis)                    

    return None
  
    






def corr_Fbal(inst,gen_dic,data_inst,plot_dic,data_prop,data_dic):
    r"""**Main flux balance correction routine.**

    Determines and applies flux balance correction.
    
    If spectra have lost their absolute flux balance, they write as
    
    :math:`F_\mathrm{obs}(w,t) = a(w,t) F(w,t)`
    
    where `a(w,t)` represents instrumental systematics, diffusion by Earth atmosphere, etc.  
    These systematics are assumed to vary slowly with wavelength, so that the shape of the spectra is kept locally.  
    Correcting for these low-frequency variations should then allow us to retrieve the correct shape of planetary and stellar lines.
    
    If we neglect variations due to narrow planetary lines, assumed to be diluted when considering the spectra at low-resolution: 
    
    .. math::
       F_\mathrm{obs}(\mathrm{w \, in \, w_\mathrm{bin}},vis,t) &\sim a(w_\mathrm{bin},t) ( F_\mathrm{\star}(w,vis) LC_\mathrm{tr}(w_\mathrm{bin},vis,t) + F_\mathrm{\star}(w,vis) LC_\mathrm{refl}(w_\mathrm{bin},vis,t) + F_\mathrm{p}^\mathrm{therm}(w_\mathrm{bin},vis,t) )  \\
                             &= a(w_\mathrm{bin},t) F_\mathrm{\star}(w,vis) ( LC_\mathrm{tr}(w_\mathrm{bin},vis,t) + LC_\mathrm{refl}(w_\mathrm{bin},vis,t) + F_\mathrm{p}^\mathrm{therm}(w_\mathrm{bin},vis,t)/F_\mathrm{\star}(w,vis) )  \\                      
                             &= a(w_\mathrm{bin},t) F_\mathrm{\star}(w,vis) \delta_\mathrm{p}(w_\mathrm{bin},vis,t)                        
                          
    where we account for all contributions from the planet. 
    
    We define a reference for the unocculted star :math:`F_\mathrm{ref}(w,vis)`, which can either be a theoretical spectrum, or a measured master spectrum. 
    It is thus possible that :math:`F_\mathrm{ref}(w,vis) = C_\mathrm{ref}(w) F_\mathrm{\star}(w,vis)`, where `A` represents the deviation of the theoretical spectrum from the stellar spectrum, or a   
    combination of the a(w,t) from the spectra used to build the master. Note that the unknown scaling of the flux due to the distance to the star is implicitely included in `A`.  
    The same reference can be used for all spectra obtained with a given instrument, but we obtain better results by calculating references specific to each visit.    
    Assuming that :math:`C_\mathrm{ref}` is dominated by low-frequency variations, we have :math:`F_\mathrm{ref}(w_\mathrm{bin},vis) = C_\mathrm{ref}(w_\mathrm{bin}) F_\mathrm{\star}(w_\mathrm{bin},vis)`

    We bin the data to smooth out noise and high-frequency variations, and to sample `a(w,t)` and the planetary continuum.
    Theoretically we would obtain a cleaner estimate of `a` by first calculating the spectral ratio between each exposure and the reference, and then binning the spectral ratio.   
    In practice however the dispersion of the flux in low-SNR regions leads to spurious variations when calculating this ratio on individual pixels.
    We thus first bin the exposure and master spectra, and then calculate the ratio between those binned spectra.
    
    Rather than binning the data in flux units we first scale it back to raw count units, to avoid artificially increasing the uncertainty and dispersion in a given bin.
    This does not bias the measurement of 'a', considering that:  

    .. math::        
        N_\mathrm{obs}(w_\mathrm{bin},t) &= \sum(w_\mathrm{bin},F_\mathrm{obs}(w,t) dt dw/g_\mathrm{cal}(w))   \\
                     &\sim \sum(w_\mathrm{bin},a(w_\mathrm{bin},t) F_\mathrm{\star}(w,v) dt dw/g_\mathrm{cal}(w))  \\
                     &\sim a(w_\mathrm{bin},t) \delta_\mathrm{p}(w_\mathrm{bin},v,t) dt \sum(w_\mathrm{bin},F_\mathrm{\star}(w,v)/g_\mathrm{cal}(w))     \\    
        N_\mathrm{ref}(w_\mathrm{bin},t) &= \sum(w_\mathrm{bin},F_\mathrm{ref}(w,t) dw/g_\mathrm{cal}(w))        \\
                     &= \sum(w_\mathrm{bin},C_\mathrm{ref}(w_\mathrm{bin}) F_\mathrm{\star}(w,v) dw/g_\mathrm{cal}(w))    \\    
                     &= C_\mathrm{ref}(w_\mathrm{bin}) \sum(w_\mathrm{bin},F_\mathrm{\star}(w,v) dw/g_\mathrm{cal}(w))   
                 
    where we scale the master AFTER its calculation in the star rest frame and shift to the exposure rest frame, to prevent introducing biases
    
    We fit a polynomial to the ratio between the binned exposure spectra and the stellar reference, scaled. :math:`P[w_\mathrm{bin}]` is thus an estimate of: 

    .. math::  
        N_\mathrm{obs}(w_\mathrm{bin},t)/N_\mathrm{ref}(w_\mathrm{bin},t) &= a(w_\mathrm{bin},t) \delta_\mathrm{p}(w_\mathrm{bin},v,t) dt \sum(w_\mathrm{bin},F_\mathrm{\star}(w,v)/g_\mathrm{cal}(w)) / C_\mathrm{ref}(w_\mathrm{bin}) \sum(w_\mathrm{bin},F_\mathrm{\star}(w,v) dw/g_\mathrm{cal}(w))  \\ 
                                  &= a(w_\mathrm{bin},t) \delta_\mathrm{p}(w_\mathrm{bin},vis,t) dt / C_\mathrm{ref}(w_\mathrm{bin}) 
                              
    assuming that :math:`C_\mathrm{ref}` is dominated by low-frequency variations, we obtain 

    :math:`P[w_\mathrm{bin}] ~ a(w_\mathrm{bin},t) \delta_\mathrm{p}(w_\mathrm{bin},vis,t)/C_\mathrm{ref}(w_\mathrm{bin})`
    
    We then extrapolate :math:`P[w_\mathrm{bin}]` at all wavelengths w, and correct spectra using P(w), resulting in :
    
    :math:`F_\mathrm{norm}(w,t) = F_\mathrm{obs}(w,t)/P(w) = F_\mathrm{\star}(w,vis) C_\mathrm{ref}(w) = F_\mathrm{ref}(w,vis)`
    
    low-frequency variations of planetary origins are thus removed together with the systematics, and will be re-injected during the light curve scaling
    
    Finally, we reset the spectra to their original absolute flux level. This is done to remain as close as possible to the measured photon count, which is 
    important in particular to their conversion into stellar CCFs, before it is necessary to rescale the spectra to the exact same flux level, which becomes
    necessary when analyzing the RM effect and atmospheric signals
    
    The correction can be fitted to a smaller spectral band than the full spectrum (eg if the pipeline is set to analyze spectral lines in a given region, or to avoid ranges contaminated by spurious features).
    However the range selected for the fit must be large enough to capture low-frequency variations over the spectral ranges selected for analysis in the pipeline
    
    This correction is applied here, rather than simultaneously with the transit rescaling, because it is necessary to correct for cosmics
    
    When a measured master is used and several visits are processed, we apply first a global correction relative to the master of the visit.
    If requested we then apply the intra-order correction, still using the visit master as reference.
    Finally we apply a global correction based on the ratio between the visit master and a reference, measured at lower resolution than with the visit master.
    This is because deriving directly the correction from the ratio between exposure and instrument master introduces local variations (possibly due to the changes in 
    line shape between epochs) that bias the fitted model. The two-step approach above, where the visit-to-reference correction is measured at low-resolution and high SNR, is more stable
     
    Measuring the stellar continuum on the master and individual spectra, to then do their ratio and measure the flux balance, is not a better solution because to capture well enough the 
    continuum profile the rollin pin has to be small enough, in which case it becomes sensitive to the wiggles. They could then be smoothed out in the continuum ratio, or through a fit with a smoothed or low-order 
    polynomial, but then it becomes equivalent to the present solution    

    Args:
        TBD
    
    Returns:
        None
    
    """     
    txt_print = '   > Correcting spectra for'
    if gen_dic['corr_Fbal']:txt_print+=' global'
    if gen_dic['corr_FbalOrd']:txt_print+=' and intra-order'
    txt_print+=' flux balance'
    print(txt_print)
    if (gen_dic['Fbal_vis']=='meas') and (gen_dic[inst]['n_visits']==1):stop('No need to set flux balance to a reference for a single visit and instrument') 
    if (gen_dic['Fbal_vis'] is None):
        if (gen_dic[inst]['n_visits']>1):stop('Flux balance must be set to a reference for multiple visits')
    else:print('         Final scaling to '+{'meas':'measured','theo':'theoretical'}[gen_dic['Fbal_vis']]+' reference')  
    cond_calc = gen_dic['calc_corr_Fbal'] | gen_dic['calc_corr_FbalOrd'] 

    #Calculating data
    if cond_calc:
        print('         Calculating data')    

        #Global correction
        if gen_dic['corr_Fbal']:

            #Indexes of order to be fitted
            iord_fit_Fbal = range(data_inst['nord_ref'])

            #Deviation between current visit and reference
            if gen_dic['Fbal_vis'] is not None:

                #Calibration profile function
                mean_gdet_func = dataload_npz(gen_dic['save_data_dir']+'Processed_data/Calibration/'+inst+'_mean_gdet')['func'] 

                #Set reference to average of measured visit masters
                if gen_dic['Fbal_vis']=='meas':  
                    data_Mast_ref = dataload_npz(gen_dic['save_data_dir']+'Corr_data/Global_Master/'+inst+'_meas')                     
                    cen_wav_Mstar = data_Mast_ref['cen_bins']
                    low_wav_Mstar = data_Mast_ref['edge_bins'][:,0:-1]
                    high_wav_Mstar = data_Mast_ref['edge_bins'][:,1::]    
                    dwav_Mstar = high_wav_Mstar-low_wav_Mstar

        #Intra-order correction
        if gen_dic['corr_FbalOrd']:        
            range_fit=gen_dic['FbalOrd_range_fit']
            
            #Indexes of order to be fitted
            #    - for "order" mode, the condition also defines orders to be corrected
            iord_fit_Fbal_ord = range(data_inst['nord_ref'])    

        #Default options
        for key in ['Fbal_deg','Fbal_smooth','Fbal_deg_vis','Fbal_smooth_vis','Fbal_deg_ord']:
            if (inst not in gen_dic[key]):gen_dic[key][inst]={}            

        #Process each visit
        corr_func_vis = None
        for ivisit,vis in enumerate(deepcopy(data_inst['visit_list'])):
            data_vis=data_inst[vis]
            data_glob={}

            #Saved paths
            proc_DI_data_paths_new = gen_dic['save_data_dir']+'Corr_data/Fbal/'+inst+'_'+vis+'_' 

            #--------------------------------------------------------
            #Global flux balance
            if gen_dic['corr_Fbal']:

                #Default options
                if (vis not in gen_dic['Fbal_deg_vis'][inst]):gen_dic['Fbal_deg_vis'][inst][vis] = 4
                if (vis not in gen_dic['Fbal_smooth_vis'][inst]):gen_dic['Fbal_smooth_vis'][inst][vis] = 1e-4                  

                #Fitted orders
                if (inst in gen_dic['Fbal_ord_fit']) and (vis in gen_dic['Fbal_ord_fit'][inst]) and (len(gen_dic['Fbal_ord_fit'][inst][vis])>0):
                    iord_fit_list = np.intersect1d(iord_fit_Fbal,gen_dic['Fbal_ord_fit'][inst][vis]) 
                else:iord_fit_list = iord_fit_Fbal                
                
                #Fitted ranges
                if (inst not in gen_dic['Fbal_range_fit']) or ((inst in gen_dic['Fbal_range_fit']) & (vis not in gen_dic['Fbal_range_fit'][inst])):range_fit=[]
                else:range_fit=gen_dic['Fbal_range_fit'][inst][vis] 

                #Measuring deviation between current visit and global instrument master
                #    - see details in corrFbal_vis()
                if gen_dic['Fbal_vis'] is not None:   
                    
                    #Set reference to theoretical master
                    if gen_dic['Fbal_vis']=='theo':                     
                        data_Mast_ref = dataload_npz(gen_dic['save_data_dir']+'Corr_data/Global_Master/'+inst+'_'+vis+'_theo')                     
                        cen_wav_Mstar = data_Mast_ref['cen_bins']
                        low_wav_Mstar = data_Mast_ref['edge_bins'][:,0:-1]
                        high_wav_Mstar = data_Mast_ref['edge_bins'][:,1::]    
                        dwav_Mstar = high_wav_Mstar-low_wav_Mstar    
                        
                    #Retrieve measured visit master
                    data_Mast_vis = dataload_npz(gen_dic['save_data_dir']+'Corr_data/Global_Master/'+inst+'_'+vis+'_meas')
                    
                    #Processing visit and reference masters over their commonly defined pixels, within selected ranges
                    #    - the measured reference for an instrument is only defined in pixels where all visit masters are defined
                    #    - the theoretical reference is defined at all wavelengths
                    if gen_dic['Fbal_vis']=='meas':cond_def_Mast = deepcopy(data_Mast_ref['cond_def'])
                    elif gen_dic['Fbal_vis']=='theo':cond_def_Mast = deepcopy(data_Mast_vis['cond_def'])   
                    if (len(range_fit)>0):
                        cond_sel = np.zeros(data_Mast_ref['flux'].shape,dtype=bool)
                        for bd_band_loc in range_fit:cond_sel|=(low_wav_Mstar>bd_band_loc[0]) & (high_wav_Mstar<bd_band_loc[1])
                        cond_def_Mast &= cond_sel                     
             
                    #Mean flux ratio between visit and reference
                    tot_Fr_vis = corrFbal_totFr(data_inst['nord'],cond_def_Mast,data_dic['DI']['scaling_range'],low_wav_Mstar,high_wav_Mstar,dwav_Mstar,data_Mast_vis['flux'],data_Mast_ref['flux'])
               
                    #Switching fitted data to nu space
                    #    - the scaling must be applied uniformely to the two masters so as not introduce biases
                    low_nu_exp = c_light/high_wav_Mstar[:,::-1]
                    high_nu_exp = c_light/low_wav_Mstar[:,::-1]
                    dnu_exp =  high_nu_exp - low_nu_exp   
                    nu_bins_exp = c_light/cen_wav_Mstar[:,::-1]

                    #Processing requested orders
                    #    - shifting from the star (source) to the solar barycenter (receiver) rest frame
                    #      see gen_specdopshift() :
                    # w_receiver = w_source * (1+ rv[s/r]/c))
                    # w_solbar = w_star * (1+ rv[star/starbar]/c))* (1+ rv[starbar/solbar]/c))
                    #      we neglect the stellar keplerian motion
                    iord_fit_list_def = np.intersect1d(iord_fit_list,np_where1D(np.sum(cond_def_Mast,axis=1)))
                    cen_wav_Mstar_solbar = cen_wav_Mstar*gen_specdopshift(data_dic['DI']['sysvel'][inst][vis])             
                    nu_Mstar_binned = np.zeros(0,dtype=float)
                    Mstar_Fr = np.zeros(0,dtype=float)
                    for iord in iord_fit_list_def:  
                        
                        #Scaling masters back to counts over their common defined pixels in order  
                        #    - median calibration profile must be calculated in solar barycentric rest frame with the function, but defined over the same table that is used in the star rest frame for all masters
                        mean_gdet_Mstar_ord = mean_gdet_func[iord](cen_wav_Mstar_solbar[iord,cond_def_Mast[iord]])[::-1]
                        count_Mstar_vis_ord = data_Mast_vis['flux'][iord,cond_def_Mast[iord]][::-1] /mean_gdet_Mstar_ord                      
                        count_Mstar_ref_ord = data_Mast_ref['flux'][iord,cond_def_Mast[iord]][::-1]/mean_gdet_Mstar_ord
                        
                        #Summing over full order
                        nu_Mstar_binned = np.append(nu_Mstar_binned,np.mean(nu_bins_exp[iord,cond_def_Mast[iord]]))
                        dnu_Mstar_ord = dnu_exp[iord,cond_def_Mast[iord]] 
                        totcount_Mstar_ref = np.sum(count_Mstar_ref_ord*dnu_Mstar_ord)
                        totcount_Mstar_vis = np.sum(count_Mstar_vis_ord*dnu_Mstar_ord) 
                        Mstar_Fr = np.append(Mstar_Fr,totcount_Mstar_vis/totcount_Mstar_ref)

                    #Fitted values
                    id_sort=np.argsort(nu_Mstar_binned)
                    cen_bins_fit = nu_Mstar_binned[id_sort]
                    norm_Fr_fit = Mstar_Fr[id_sort]/tot_Fr_vis   
                    n_bins = len(norm_Fr_fit)
                    cond_fit = np.repeat(True,n_bins)            
           
                    #Fit
                    #    - we neglect uncertainties and assume they are equal for all bins
                    if gen_dic['Fbal_mod']=='pol':corr_func_vis = np.poly1d(np.polyfit(cen_bins_fit[cond_fit], norm_Fr_fit[cond_fit], gen_dic['Fbal_deg_vis'][inst][vis] )) 
                    elif gen_dic['Fbal_mod']=='spline':corr_func_vis = UnivariateSpline(cen_bins_fit[cond_fit], norm_Fr_fit[cond_fit],s=gen_dic['Fbal_smooth_vis'][inst][vis] )

                    #Save correction data for plotting purposes
                    if (plot_dic['Fbal_corr_vis']!=''):
                        data_glob.update({'corr_func_vis':corr_func_vis,'Fbal_wav_bin_vis':c_light/nu_Mstar_binned[::-1],'Fbal_T_binned_vis':Mstar_Fr[::-1],'tot_Fr_vis':tot_Fr_vis}) 
                        data_glob['cond_fit_vis'] = np.zeros(n_bins,dtype=bool)
                        data_glob['cond_fit_vis'][id_sort] = cond_fit[::-1]                             

            #--------------------------------------------------------
            #Intra-order flux balance
            if gen_dic['corr_FbalOrd']:

                #Fitted and corrected orders
                if (inst in gen_dic['FbalOrd_ord_fit']) and (vis in gen_dic['FbalOrd_ord_fit'][inst]) and (len(gen_dic['FbalOrd_ord_fit'][inst][vis])>0):
                    iord_fit_list = np.intersect1d(iord_fit_Fbal_ord,gen_dic['FbalOrd_ord_fit'][inst][vis]) 
                else:iord_fit_list = iord_fit_Fbal_ord

            #Default options       
            if (vis not in gen_dic['Fbal_deg'][inst]):gen_dic['Fbal_deg'][inst][vis] = 4
            if (vis not in gen_dic['Fbal_smooth'][inst]):gen_dic['Fbal_smooth'][inst][vis] = 1e-4     
            if (vis not in gen_dic['Fbal_deg_ord'][inst]):gen_dic['Fbal_deg_ord'][inst][vis] = 4                

            #--------------------------------------------------------
            #Processing all exposures    
            iexp_all = range(data_vis['n_in_visit'])
            common_args = (data_vis['proc_DI_data_paths'],inst,vis,gen_dic['save_data_dir'],range_fit,data_vis['dim_exp'],iord_fit_list,gen_dic['Fbal_bin_nu'],data_dic['DI']['scaling_range'],gen_dic['Fbal_mod'],gen_dic['Fbal_deg'][inst][vis],gen_dic['Fbal_smooth'][inst][vis],gen_dic['Fbal_clip'],data_inst['nord'],data_vis['nspec'],gen_dic['Fbal_range_corr'],\
                            plot_dic['Fbal_corr'],plot_dic['sp_raw'],gen_dic['Fbal_binw_ord'],plot_dic['Fbal_corr_ord'],proc_DI_data_paths_new,gen_dic['Fbal_ord_clip'],gen_dic['resamp_mode'],gen_dic['Fbal_deg_ord'][inst][vis],gen_dic['Fbal_expvar'],data_dic[inst][vis]['mean_gdet_DI_data_paths'],corr_func_vis,gen_dic['corr_Fbal'],gen_dic['corr_FbalOrd'])
            if gen_dic['Fbal_nthreads']>1:tot_Fr_all = MAIN_multithread(corrFbal_vis,gen_dic['Fbal_nthreads'],data_vis['n_in_visit'],[iexp_all],common_args,output = True)                           
            else:tot_Fr_all = corrFbal_vis(iexp_all,*common_args)  
            data_vis['proc_DI_data_paths'] = proc_DI_data_paths_new

            #Save global data 
            if gen_dic['corr_Fbal']:data_glob.update({'tot_Fr_all':tot_Fr_all})
            if gen_dic['corr_FbalOrd']:data_glob['Ord'] = {'iord_corr_list':iord_fit_list}
            np.savez_compressed(proc_DI_data_paths_new+'add',data=data_glob,allow_pickle=True)

    #Updating path to processed data and checking it has been calculated
    else:
        for vis in data_inst['visit_list']:  
            data_vis=data_inst[vis] 
            data_vis['proc_DI_data_paths']=gen_dic['save_data_dir']+'Corr_data/Fbal/'+inst+'_'+vis+'_'         
            check_data({'path':data_vis['proc_DI_data_paths']+str(0)},vis=vis)

    return None


def corrFbal_vis(iexp_group,proc_DI_data_paths,inst,vis,save_data_dir,range_fit,dim_exp,iord_fit_list,Fbal_bin_nu,scaling_range,Fbal_mod,Fbal_deg,Fbal_smooth,Fbal_clip,nord,nspec,Fbal_range_corr,\
                 plot_Fbal_corr,plot_sp_raw,Fbal_binw_ord,plot_Fbal_corr_ord,proc_DI_data_paths_new,Fbal_ord_clip,resamp_mode,Fbal_deg_ord,Fbal_expvar,mean_gdet_DI_data_paths,corr_func_vis,corr_Fbal,corr_FbalOrd):
    r"""**Flux balance correction per visit.**    

    Determines and applies flux balance correction in each visit.    
    
    Args:
        TBD
    
    Returns:
        TBD
    
    """       
    
    #Processing each exposure
    tot_Fr_all = np.zeros(len(iexp_group),dtype=float)
    for isub_exp,iexp in enumerate(iexp_group):
        dic_sav={}
            
        #Upload latest processed data and corresponding visit master
        #    - master has been shifted and resampled onto the exposure table in the input rest frame
        data_exp = dataload_npz(proc_DI_data_paths+str(iexp))
        data_mast_exp = dataload_npz(save_data_dir+'Corr_data/Global_Master/'+inst+'_'+vis+'_'+str(iexp))
        low_wav_exp = data_exp['edge_bins'][:,0:-1]
        high_wav_exp = data_exp['edge_bins'][:,1::] 
        dwav_exp =  high_wav_exp - low_wav_exp                
      
        #Processing master and exposure over their common pixels (defined and outside of selected range)
        cond_fit_all = data_exp['cond_def'] & data_mast_exp['cond_def']
        if (len(range_fit)>0):
            cond_sel = np.zeros(dim_exp,dtype=bool)
            for bd_band_loc in range_fit:cond_sel|=(data_exp['edge_bins'][:,0:-1]>bd_band_loc[0]) & (data_exp['edge_bins'][:,1::]<bd_band_loc[1])
            cond_fit_all &= cond_sel                

        #Retrieve calibration profile associated with current exposure in nu space
        #    - defined in the same frame and over the same table as the exposure spectrum
        mean_gdet_exp = dataload_npz(mean_gdet_DI_data_paths[iexp])['mean_gdet'][:,::-1] 

        #--------------------------------------------------------------------------------
        #Correct flux balance over full spectrum
        if corr_Fbal:

            #Mean flux ratio between exposure and reference
            tot_Fr_all[isub_exp] = corrFbal_totFr(nord,cond_fit_all,scaling_range,low_wav_exp,high_wav_exp,dwav_exp,data_exp['flux'],data_mast_exp['flux'])

            #Switching fitted data to nu space and scaling it from flux to count units
            #    - the scaling must be applied uniformely to the exposure and master spectra (ie, after the master has been calculated) so as not introduce biases
            low_nu_exp = c_light/high_wav_exp[:,::-1]
            high_nu_exp = c_light/low_wav_exp[:,::-1]
            dnu_exp =  high_nu_exp - low_nu_exp   
            nu_bins_exp = c_light/data_exp['cen_bins'][:,::-1]
            count_exp = data_exp['flux'][:,::-1]/mean_gdet_exp
            cond_def_exp =  data_exp['cond_def'][:,::-1]
            count_mast_exp = data_mast_exp['flux'][:,::-1]/mean_gdet_exp 

            #Adding progressively bins of current spectrum and master that will be used to fit the color balance correction
            #    - each order of 2D spectra is processed independently, but contributes to the global binned table  
            bin_exp_dic={}
            for key in ['Fr','varFr','Fmast_tot','cen_bins','low_bins','high_bins','idx_ord_bin']:bin_exp_dic[key] = np.zeros(0,dtype=float) 
            iord_fit_list_def = np.intersect1d(iord_fit_list,np_where1D(np.sum(cond_fit_all,axis=1)))                
            for iord in iord_fit_list_def:
                
                #Force binning of full order at bluest wavelengths
                #    - prevent spline from diverging
                if (inst=='ESPRESSO') and (iord in [0,1]):Fbal_bin_nu_loc = 1000.
                else:Fbal_bin_nu_loc = deepcopy(Fbal_bin_nu)

                #Defining bins
                var_ord = data_exp['cov'][iord][0][::-1]/mean_gdet_exp[iord]**2. 
                bin_bd,raw_loc_dic = sub_def_bins(Fbal_bin_nu_loc,np_where1D(cond_fit_all[iord][::-1]),low_nu_exp[iord],high_nu_exp[iord],dnu_exp[iord],nu_bins_exp[iord],count_exp[iord],Mstar_loc=count_mast_exp[iord],var1D_loc=var_ord)
      
                #Process bins
                nfilled_bins=0
                for ibin,(low_bin_loc,high_bin_loc) in enumerate(zip(bin_bd[0:-1],bin_bd[1:])):
                    bin_loc_dic,nfilled_bins = sub_calc_bins(low_bin_loc,high_bin_loc,raw_loc_dic,nfilled_bins,calc_Fr=True)
                    if len(bin_loc_dic)>0:
                        for key in bin_loc_dic:bin_exp_dic[key] = np.append( bin_exp_dic[key] , bin_loc_dic[key])

                #Store index of order to which the bin belongs to
                bin_exp_dic['idx_ord_bin'] = np.append(bin_exp_dic['idx_ord_bin'],np.repeat(iord,nfilled_bins))

            #Fitted values
            #    - we divide by the exposure-to-master flux ratio C(t), so that only the relative flux balance around the mean exposure flux is corrected for 
            #    - in nu space
            id_sort=np.argsort(bin_exp_dic['cen_bins'])
            cen_bins_fit = bin_exp_dic['cen_bins'][id_sort]
            norm_Fr_fit = bin_exp_dic['Fr'][id_sort]/tot_Fr_all[isub_exp]
            norm_varFr_fit = bin_exp_dic['varFr'][id_sort]/tot_Fr_all[isub_exp]**2.
            n_bins = len(norm_Fr_fit)
            cond_fit = np.repeat(True,n_bins)

            #Uncertainty scaling
            #    - we scale errors because large uncertainties in some parts of the spectra (typically in the blue for ground-based spectra) may bias the polynomial fit in these regions. 
            w_Fr_fit = 1./norm_varFr_fit**Fbal_expvar
            w_Fr_fit = w_Fr_fit/np.mean(w_Fr_fit)

            #Remove extreme outliers
            if Fbal_clip:
                med_prop = np.median(norm_Fr_fit)
                res = norm_Fr_fit - med_prop
                disp_est = stats.median_abs_deviation(res)
                cond_fit[(norm_Fr_fit>med_prop + 10.*disp_est) | (norm_Fr_fit<med_prop-10.*disp_est)] = False               

            #Fit
            if Fbal_mod=='pol':     corr_func = np.poly1d(np.polyfit(cen_bins_fit[cond_fit], norm_Fr_fit[cond_fit], Fbal_deg )) 
            elif Fbal_mod=='spline':corr_func = UnivariateSpline(cen_bins_fit[cond_fit], norm_Fr_fit[cond_fit],s=Fbal_smooth     ,w=w_Fr_fit[cond_fit])

            #Successive fits with automatic identification and exclusion of outliers
            if Fbal_clip:
                thresh = 3.
                for it_res in range(3):
                    
                    #Residuals from previous iteration
                    res = norm_Fr_fit - corr_func(cen_bins_fit)
                    
                    #Sigma-clipping
                    disp_est = np.std(res[cond_fit])
                    cond_fit[(res>thresh*disp_est) | (res<-thresh*disp_est)] = False
         
                    #Fit for current iteration
                    if Fbal_mod=='pol':corr_func = np.poly1d(np.polyfit(cen_bins_fit[cond_fit], norm_Fr_fit[cond_fit],Fbal_deg )) 
                    elif Fbal_mod=='spline':corr_func = UnivariateSpline(cen_bins_fit[cond_fit], norm_Fr_fit[cond_fit],s=Fbal_smooth      ,w=w_Fr_fit[cond_fit])

            #Flatten the order matrix into a 1D table
            n_flat = nord*nspec
            nu_bins_flat = np.reshape(nu_bins_exp,n_flat)
            cond_def_flat = np.reshape(cond_def_exp,n_flat)
            pcorr_T_flat = np.zeros(n_flat,dtype=float)*np.nan

            #Calculate correction 
            #    - the correction is defined over the full range of the original, non-reduced spectra (for plotting purposes) but applied to the reduced spectra 
            pcorr_T_flat[cond_def_flat] = corr_func(nu_bins_flat[cond_def_flat])  

            #Return correction to matrix shape and wavelength dimension
            Fbal_T_fit_all= np.reshape(pcorr_T_flat,[nord,nspec])[:,::-1]
            
            #Define correction range
            #    - we do not correct undefined pixels (correction is set to 1)
            #    - the SNR can be too low in the bluest orders for the correction to be fitted correctly, implying that the noise dominates
            # the flux balance effect. Rather than bias the spectra with the poorly defined polynomial correction, one can chose to leave those bins uncorrected when setting 'Fbal_range_corr'
            #      this will create a sharp transition with corrected bins, but it disappear later on when calculating flux ratios   
            cond_corr = deepcopy(data_exp['cond_def'])
            if len(Fbal_range_corr)>0:
                cond_range=False
                for bd_int in Fbal_range_corr:cond_range |= (low_wav_exp>=bd_int[0]) & (high_wav_exp<=bd_int[1])  
                cond_corr &= cond_range

            #Applying correction
            iord2corr = np_where1D( np.sum( cond_corr,axis=1 )>0 )
            for iord in iord2corr:
                data_corr = np.ones(Fbal_T_fit_all[iord].shape,dtype=float)
                data_corr[cond_corr[iord]] = Fbal_T_fit_all[iord][cond_corr[iord]]                
                data_exp['flux'][iord],data_exp['cov'][iord] = bind.mul_array(data_exp['flux'][iord],data_exp['cov'][iord],1./data_corr)

            #Save independently correction data
            dic_sav.update({'corr_func':corr_func,'corr_func_vis':corr_func_vis})                          
            if (plot_Fbal_corr!='') or (plot_sp_raw!=''):
                dic_sav['Fbal_wav_bin_all'] = np.vstack((c_light/bin_exp_dic['high_bins'][::-1],c_light/bin_exp_dic['cen_bins'][::-1],c_light/bin_exp_dic['low_bins'][::-1]))
                if plot_sp_raw!='':
                    dic_sav['idx_ord_bin'] = bin_exp_dic['idx_ord_bin'][::-1]
                    dic_sav['uncorrected_data_path'] = deepcopy(proc_DI_data_paths+str(iexp)) #save path to data before correction 
                if (plot_Fbal_corr!=''):
                    dic_sav['Fbal_T_binned_all'] = bin_exp_dic['Fr'][::-1]
                    dic_sav['cond_fit'] = np.zeros(n_bins,dtype=bool)
                    dic_sav['cond_fit'][id_sort] = cond_fit[::-1]                       

        #--------------------------------------------------------------------------------
        #Correct flux balance in each order of the latest processed data
        #    - this is a relative flux correction
        #    - we define Fcorr(t) = F(t)*TFr/Fr_mod ~ F(t)*TF(t)*Fmast_mod/(F_mod(t)*TFmast) ~ Fmast_mod*TF(t)/TFmast
        #      so that the corrected spectrum has the same balance as the master stellar spectrum, and keep its original mean flux
        #    - here we assume the bins are small enough that count scaling is not necessary
        if corr_FbalOrd:

            #Processing each order
            Fbal_wav_bin_exp = np.zeros(nord,dtype=object)
            Fbal_T_binned_exp = np.zeros(nord,dtype=object)
            Fbal_T_fit_exp = np.zeros(nord,dtype=object)
            tot_Fr_exp = np.zeros(nord,dtype=float) 
            cond_fit_unsort = np.zeros(nord,dtype=object)
            corr_func={}
            iord_fit_list_def = np.intersect1d(iord_fit_list,np_where1D(np.sum(cond_fit_all,axis=1)))            
            for isub_ord,iord in enumerate(iord_fit_list_def):
                cond_def_sp_exp = data_exp['cond_def'][iord]
                data_corr = np.ones(nspec,dtype=float) 

                #Total flux ratio between exposure and reference, over defined pixels
                idx_def_ord = np_where1D(cond_fit_all[iord])
                bin_dic={'tot_Fr' : np.sum(data_exp['flux'][iord][idx_def_ord]*dwav_exp[iord][idx_def_ord] )/np.sum(data_mast_exp['flux'][iord][idx_def_ord]*dwav_exp[iord][idx_def_ord] ) }   

                #Adding progressively bins that will be used to fit the correction
                #    - we keep this approach because orders with deep telluric lines (for which corrected pixels are undefined)
                # have too many undefined ranges after downsampling with bind.resampling, resulting in poorly defined polynomial corrections
                bin_bd,raw_loc_dic = sub_def_bins(Fbal_binw_ord,cond_fit_all[iord],low_wav_exp[iord],high_wav_exp[iord],dwav_exp[iord],data_exp['cen_bins'][iord],data_exp['flux'][iord]/mean_gdet_exp[iord],Mstar_loc=data_mast_exp['flux'][iord]/mean_gdet_exp[iord])
                for key in ['Fr','Fmast_tot','cen_bins','low_bins','high_bins']:bin_dic[key] = np.zeros(0,dtype=float) 
                for ibin,(low_bin_loc,high_bin_loc) in enumerate(zip(bin_bd[0:-1],bin_bd[1:])):
                    bin_loc_dic,_ = sub_calc_bins(low_bin_loc,high_bin_loc,raw_loc_dic,0,calc_Fr=True)
                    if len(bin_loc_dic)>0:
                        for key in bin_loc_dic:bin_dic[key] = np.append( bin_dic[key] , bin_loc_dic[key])  
                id_sort=np.argsort(bin_dic['cen_bins'])
                for key in ['Fr','Fmast_tot','cen_bins','low_bins','high_bins']:bin_dic[key] = bin_dic[key][id_sort]
                conddef_bin = np.ones(len(bin_dic['cen_bins']),dtype=bool)

                #Fitted values
                #    - we divide by the exposure-to-master flux ratio C(t), so that only the relative flux balance around the mean exposure flux is corrected for 
                cen_bins_fit = bin_dic['cen_bins'][conddef_bin]
                norm_Fr_fit = bin_dic['Fr']/bin_dic['tot_Fr']
                n_bins = len(norm_Fr_fit)
                cond_fit = np.repeat(True,n_bins)
                
                #Remove extreme outliers
                med_prop = np.median(norm_Fr_fit)
                res = norm_Fr_fit - med_prop
                disp_est = stats.median_abs_deviation(res)
                cond_fit[(norm_Fr_fit>med_prop + 10.*disp_est) | (norm_Fr_fit<med_prop-10.*disp_est)] = False               

                #Fit
                corr_func[iord] = np.poly1d(np.polyfit(cen_bins_fit[cond_fit], norm_Fr_fit[cond_fit], Fbal_deg_ord )) 

                #Successive fits with automatic identification and exclusion of outliers
                if Fbal_ord_clip:
                    thresh = 3.
                    for it_res in range(3):
                        
                        #Residuals from previous iteration
                        res = norm_Fr_fit - corr_func[iord](cen_bins_fit)
                        
                        #Sigma-clipping
                        disp_est = np.std(res[cond_fit])
                        cond_fit[(res>thresh*disp_est) | (res<-thresh*disp_est)] = False
             
                        #Fit for current iteration
                        corr_func[iord] = np.poly1d(np.polyfit(cen_bins_fit[cond_fit], norm_Fr_fit[cond_fit], Fbal_deg_ord )) 

                #Calculate and apply relative flux correction
                #    - proxy for F(w,t)/Fstar(w) ~ C(t)*Fbal(w)
                #    - the correction is defined over the full range of the spectra (for plotting purposes), even if then applied to a selected range
                data_corr_rel = np.ones(nspec,dtype=float)
                data_corr_rel[cond_def_sp_exp] = corr_func[iord](data_exp['cen_bins'][iord,cond_def_sp_exp])                     
                data_exp['flux'][iord],data_exp['cov'][iord] = bind.mul_array(data_exp['flux'][iord],data_exp['cov'][iord],1./data_corr_rel)
                data_corr*=data_corr_rel

                #------------------------------------------
                
                #Save independently correction data for plotting purposes
                if plot_Fbal_corr_ord!='':
                    Fbal_wav_bin_exp[iord] = np.vstack((bin_dic['low_bins'],bin_dic['cen_bins'],bin_dic['high_bins']))
                    Fbal_T_binned_exp[iord] = bin_dic['Fr']
                    Fbal_T_fit_exp[iord] = data_corr
                    tot_Fr_exp[iord] = bin_dic['tot_Fr'] 
                    cond_fit_unsort[iord] = np.zeros(n_bins,dtype=bool)       
                    cond_fit_unsort[iord][id_sort] = cond_fit

            #Save independently correction data for plotting purposes
            dic_sav['Ord'] = {'corr_func':corr_func}  
            if plot_Fbal_corr_ord!='':  
                dic_sav['Ord']['Fbal_wav_bin_all'] = Fbal_wav_bin_exp
                dic_sav['Ord']['uncorrected_data_path'] = deepcopy(proc_DI_data_paths+str(iexp)) #save path to data before correction 
                dic_sav['Ord']['Fbal_T_binned_all'] = Fbal_T_binned_exp
                dic_sav['Ord']['Fbal_T_fit_all'] = Fbal_T_fit_exp 
                dic_sav['Ord']['tot_Fr_all'] = tot_Fr_exp       
                dic_sav['Ord']['cond_fit'] = cond_fit_unsort
                dic_sav['Ord']['cen_bins_all'] = data_exp['cen_bins'] 

        ### End of order scaling
        
        #--------------------------------------------------------------------------------
        #Scaling to reference
        #    - performed after the intra-order scaling so that the latter is done with respect to the visit master
        if corr_func_vis is not None:
            pcorr_T_flat = np.zeros(n_flat,dtype=float)*np.nan
            pcorr_T_flat[cond_def_flat]= corr_func_vis(nu_bins_flat[cond_def_flat])
            Fbal_T_fit_all= np.reshape(pcorr_T_flat,[nord,nspec])[:,::-1]
            for iord in iord2corr:
                data_corr = np.ones(Fbal_T_fit_all[iord].shape,dtype=float)
                data_corr[cond_corr[iord]] = Fbal_T_fit_all[iord][cond_corr[iord]]                
                data_exp['flux'][iord],data_exp['cov'][iord] = bind.mul_array(data_exp['flux'][iord],data_exp['cov'][iord],1./data_corr)

        ### End of reference scaling

        #--------------------------------------------------------------------------------
        np.savez_compressed(proc_DI_data_paths_new+str(iexp)+'_add',data=dic_sav,allow_pickle=True)

        #Saving modified data and update paths
        np.savez_compressed(proc_DI_data_paths_new+str(iexp),data = data_exp,allow_pickle=True) 

    return tot_Fr_all



def corrFbal_totFr(nord,cond_def_sp_mast,scaling_range,low_wav,high_wav,dwav,flux_sp,flux_mast):
    r"""    

    Calculates the mean flux ratio between a given spectrum and a reference
    
     - over the same range as requested for the broadband flux scaling, neglecting the small shifts due to alignments
    
    Args:
        TBD
    
    Returns:
        tot_Fr_spec (float): mean flux ratio
    
    """     
    mean_Fsp = 0.
    mean_Fmast = 0.
    dwav_all = 0.
    for iord in range(nord):
        cond_def_ord = deepcopy(cond_def_sp_mast[iord])
        if len(scaling_range)>0:
            cond_def_scal=False 
            for bd_int in scaling_range:cond_def_scal |= (low_wav[iord]>=bd_int[0]) & (high_wav[iord]<=bd_int[1])  
            cond_def_ord &= cond_def_scal 
        dwav_all += np.sum(dwav[iord][cond_def_ord])
        mean_Fsp+=np.sum(flux_sp[iord][cond_def_ord]*dwav[iord][cond_def_ord])
        mean_Fmast+=np.sum(flux_mast[iord][cond_def_ord]*dwav[iord][cond_def_ord])    
    mean_Fsp/=dwav_all
    mean_Fmast/=dwav_all
    tot_Fr_spec = mean_Fsp/mean_Fmast    
    return tot_Fr_spec


















def corr_Ftemp(inst,gen_dic,data_inst,plot_dic,data_prop,coord_dic,data_dic):
    r"""**Temporal flux correction.**    

    Args:
        TBD
    
    Returns:
        TBD
    
    """  
    print('   > Correcting spectra for temporal variations')
    
    #Calculating data
    if (gen_dic['calc_corr_Ftemp']):
        print('         Calculating data')     
    
        #Process each visit independently
        for ivisit,vis in enumerate(data_inst['visit_list']):
            data_vis=data_inst[vis]

            #Scaling range
            if (inst not in gen_dic['Ftemp_range_fit']) or ((inst in gen_dic['Ftemp_range_fit']) & (vis not in gen_dic['Ftemp_range_fit'][inst])):range_fit=[]
            else:range_fit=gen_dic['Ftemp_range_fit'][inst][vis]  

            #Upload common spectral table
            #    - if profiles are defined on different tables they are resampled on this one
            #      if they are already defined on a common table, it is this one, which has been kept the same since the beginning of the routine
            edge_bins_com = (np.load(data_vis['proc_com_data_paths']+'.npz',allow_pickle=True)['data'].item())['edge_bins']
            
            #Global fitting range and tables
            #    - we neglect covariance between bins
            flux_all = np.zeros(data_vis['dim_all'],dtype=float)*np.nan
            sig2_all=np.zeros(data_vis['dim_all'],dtype=float)
            cond_def_fit_all  = np.zeros(data_vis['dim_all'],dtype=bool)
            bjd_all = np.zeros(data_vis['n_in_visit'],dtype=float)
            for iexp in range(data_vis['n_in_visit']): 
                bjd_all[iexp] = coord_dic[inst][vis]['bjd'][iexp]
                
                #Latest processed DI data
                #    - if data were kept on independent tables they need to be resampled on a common one to calculate equivalent fluxes
                data_exp = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp))  
                if (not data_vis['comm_sp_tab']):
                    for iord in range(data_inst['nord']): 
                        flux_all[iexp,iord],cov_temp = bind.resampling(edge_bins_com[iord], data_exp['edge_bins'][iord], data_exp['flux'][iord]  , cov = data_exp['cov'][iord], kind=gen_dic['resamp_mode'])                                                                                                               
                        sig2_all[iexp,iord] = cov_temp[0]
                    cond_def_exp = ~np.isnan(flux_all[iexp])   
                else:
                    flux_all[iexp] = data_exp['flux']
                    sig2_all[iexp]  = data_exp['cov'][iord][0]
                    cond_def_exp = data_exp['cond_def']
    
                #Fitting range, accounting for undefined pixels
                cond_def_fit_all[iexp] = cond_def_exp
                if len(range_fit)>0:
                    cond_range=False
                    for bd_int in range_fit:cond_range |= (edge_bins_com[:,0:-1]>=bd_int[0]) & (edge_bins_com[:,1:]<=bd_int[1])   
                    cond_def_fit_all[iexp] &= cond_range
   
            #Fitting pixels common to all exposures
            cond_def_fit_com  = np.all(cond_def_fit_all,axis=0)   
            
            #Total flux in fitting range
            Tflux_all = np.zeros(data_vis['n_in_visit'],dtype=float)
            Tsig_all = np.zeros(data_vis['n_in_visit'],dtype=float)
            dcen_bin_comm = (edge_bins_com[:,1::] - edge_bins_com[:,0:-1])
            for iord in range(data_inst['nord']):    
                Tflux_all += np.sum(flux_all[:,iord,cond_def_fit_com[iord]]*dcen_bin_comm[iord,cond_def_fit_com[iord]],axis=1)
                Tsig_all += np.sum(sig2_all[:,iord,cond_def_fit_com[iord]]*dcen_bin_comm[iord,cond_def_fit_com[iord]]**2.,axis=1)
            Tsig_all=np.sqrt(Tsig_all)

            #Fitted exposures
            iexp_fit = np_where1D(Tflux_all>0)
            if (inst in gen_dic['idx_nin_Ftemp_fit']) and (vis in gen_dic['idx_nin_Ftemp_fit'][inst]) and (len(gen_dic['idx_nin_Ftemp_fit'][inst][vis])>0): 
                iexp_fit = np.delete(iexp_fit,gen_dic['idx_nin_Ftemp_fit'][inst][vis])

            #Breathing correction
            if inst=='HST':
                
                #faire correction du temps par palier, jutilisais tout le temps ca
                #faire fonction de variation de la phase en polynome
                

                stop('Implement breathing correction from fonction_breath_t_LC.pro and correction_STIS_systematics.pro')
          
            #Temporal correction
            else:
    
                #Fit polynomial as a function of time
                #    - the correction is normalized to maintain average flux over the visit
                corr_func = np.poly1d(np.polyfit(bjd_all[iexp_fit], Tflux_all[iexp_fit]/np.mean(Tflux_all[iexp_fit]) , gen_dic['Ftemp_deg'][inst][vis] , w=1./Tsig_all))

                #Calculate correction over all exposures
                Ftemp_T_fit_all = corr_func(bjd_all)

            #Applying correction to original data
            proc_DI_data_paths_new = gen_dic['save_data_dir']+'Corr_data/Ftemp/'+inst+'_'+vis+'_'                 
            for iexp in range(data_vis['n_in_visit']): 
                data_exp = np.load(data_vis['proc_DI_data_paths']+str(iexp)+'.npz',allow_pickle=True)['data'].item()
                for iord in range(data_inst['nord']): 
                    data_exp['flux'][iord],data_exp['cov'][iord] = bind.mul_array(data_exp['flux'][iord],data_exp['cov'][iord],1./np.repeat(Ftemp_T_fit_all[iexp],data_vis['nspec']))
          
                #Saving modified data and update paths
                np.savez_compressed(proc_DI_data_paths_new+str(iexp),data = data_exp,allow_pickle=True) 
            data_vis['proc_DI_data_paths'] = proc_DI_data_paths_new          
            
            #Save independently correction data                         
            if (plot_dic['Ftemp_corr']!=''):
                dic_sav = {'corr_func':corr_func,'iexp_fit':iexp_fit,'bjd_all':bjd_all,'Tflux_all':Tflux_all,'Tsig_all':Tsig_all}                             
                np.savez_compressed(gen_dic['save_data_dir']+'Corr_data/Ftemp/'+inst+'_'+vis+'_add',data=dic_sav,allow_pickle=True)


    #Updating path to processed data and checking it has been calculated
    else:
        for vis in data_inst['visit_list']:  
            data_vis=data_inst[vis]
            data_vis['proc_DI_data_paths']=gen_dic['save_data_dir']+'Corr_data/Ftemp/'+inst+'_'+vis+'_'  
            check_data({'path':data_vis['proc_DI_data_paths']+str(0)},vis=vis)                

    return None                
            
            
            
            
            
    
    





















