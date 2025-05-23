#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
import bindensity as bind
from scipy.interpolate import interp1d
import pandas as pd
import batman
from ..ANTARESS_grids.ANTARESS_occ_grid import init_surf_shift,def_surf_shift,sub_calc_plocc_ar_prop,retrieve_ar_prop_from_param
from ..ANTARESS_grids.ANTARESS_coord import get_timeorbit,calc_pl_coord,excl_plrange,coord_expos_ar
from ..ANTARESS_conversions.ANTARESS_binning import init_bin_prof,weights_bin_prof,calc_bin_prof,weights_bin_prof_calc
from ..ANTARESS_process.ANTARESS_data_align import align_data
from ..ANTARESS_general.utils import dataload_npz,gen_specdopshift,stop,np_where1D,datasave_npz,np_interp,npint,MAIN_multithread,check_data

################################################################################################## 
#%% Alignment routines
################################################################################################## 
    
#NB: cannot be put in file data_align.py with align_data() due to circular import issues
def align_profiles(data_type_gen,data_dic,inst,vis,gen_dic,coord_dic):
    r"""**Main alignment routine.**    

    Aligns time-series of disk-integrated, intrinsic, and planetary profiles.  
    Profiles used as weights throughout the workflow are shifted in the same way as their associated profiles.

    Args:
        TBD
    
    Returns:
        None
    
    """     
    data_inst = data_dic[inst]  
    data_vis=data_inst[vis]
    print('   > Aligning '+gen_dic['type_name'][data_type_gen]+' profiles') 
    if data_type_gen == 'Atm':data_type = data_dic['Atm']['pl_atm_sign']
    else:data_type=deepcopy(data_type_gen)
    prop_dic = data_dic[data_type_gen]  
    proc_gen_data_paths_new = gen_dic['save_data_dir']+'Aligned_'+data_type_gen+'_data/'+gen_dic['add_txt_path'][data_type_gen]+'/'+inst+'_'+vis+'_'
    if (data_type_gen=='DI') and (data_dic['DI']['sysvel'][inst][vis]==0.):print('         WARNING: sysvel = 0 km/s')
    if data_type_gen=='Intr':proc_gen_data_paths_new+='in'  
    if (data_vis['type']=='spec2D') and ('sing_gcal_'+data_type_gen+'_data_paths' not in data_vis):stop('ERROR : weighing calibration profiles undefined; make sure you activate gen_dic["calc_proc_data"] and gen_dic["calc_gcal"] when running this module.')   
    if (data_type_gen in ['Intr','Atm']) and ('mast_'+data_type_gen+'_data_paths'): proc_mast = True
    else: proc_mast = False   #The master DI profiles has not yet been calculated when calling align_profiles() on DI profiles
    proc_locEst = True if ((data_type_gen=='Atm') and ((data_type=='Absorption') or ((data_type=='Emission')) and data_dic['Intr']['cov_loc_star'])) else False
    
    #1D variance grid
    #    - 1D variance profiles are defined for a given data type if it was converted from 2D to 1D, but may then be associated with later data types
    #      we thus align variance grids from earlier data types up to the present ones, depending on when the conversion was performed    
    var_key = None
    for subtype_gen in gen_dic['earliertypes4var'][data_type_gen]:
        if data_dic[subtype_gen]['spec2D_to_spec1D'][inst] and (data_vis['type']=='spec1d'):
            var_key = gen_dic['type2var'][gen_dic['typegen2type'][subtype_gen]]
            break

    #Resample aligned profiles on the common visit table if relevant
    #    - edge issues can arise when resampling profiles on the common table after they are shifted by
    # + the systemic velocity & Keplerian motion (for disk-integrated profiles)
    # + the stellar surface rv (for intrinsic profiles)
    # + the planetary orbital rv (for atmospheric profiles)
    #      indeed, resampling the shifted profiles back on their original tables mean that part of the profile would be outside of the table and get undefined
    #    - to account for this issue the common table has been in extended on both sides in the initialization routines by the maximum rv shift for the Keplerian, stellar surface, and orbital rv
    #      here we further shift the common table by the systemic velocity when aligning disk-integrated profiles (since it is a global shift common to all profiles)
    #      only the table needs shifting, without resampling the profiles, since the shift is common to all exposures 
    #    - even if spectra are not resampled on the common table we shift and redefined the common table so that it can be used in the shifted frame
    if (data_type_gen=='DI'):
        data_star_com = dataload_npz(data_vis['proc_com_data_paths'])

        #Alignment mode
        if data_dic['DI']['align_mode']=='kep':print('         Aligning with Keplerian model')
        elif data_dic['DI']['align_mode']=='pip':print('         Aligning with DRS RVs')
    
        #Use common systemic velocity to keep all tables the same in case they are common to an instrument
        if (data_inst['comm_sp_tab']) and (data_inst['n_visits_inst']>1):comm_sysvel = data_dic['DI']['sysvel'][inst][data_inst['com_vis']]  
        else:comm_sysvel=data_dic['DI']['sysvel'][inst][vis]       
    
        #Shift table
        if (data_vis['type']=='CCF'): 
            data_star_com['cen_bins'] -= comm_sysvel
            data_star_com['edge_bins'] -= comm_sysvel    
        else:
            spec_dopshift = 1./gen_specdopshift(comm_sysvel)
            data_star_com['cen_bins']*=spec_dopshift
            data_star_com['edge_bins']*=spec_dopshift
    
        #Save updated table
        data_vis['proc_com_star_data_paths'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_'+vis+'_com_star'
        datasave_npz(data_vis['proc_com_star_data_paths'],data_star_com) 
    
    #Resampling upon common table in star rest frame
    if (data_vis['comm_sp_tab']):
        data_com_star = dataload_npz(data_vis['proc_com_star_data_paths'])
        cen_bins_resamp, edge_bins_resamp , dim_exp_resamp , nspec_resamp  = data_com_star['cen_bins'],data_com_star['edge_bins'],data_com_star['dim_exp'] , data_com_star['nspec']
    else:cen_bins_resamp, edge_bins_resamp , dim_exp_resamp = None,None,None   

    #Calculating aligned data
    if gen_dic['calc_align_'+data_type_gen]:
        print('         Calculating data')

        #Define RV shifts
        #    - shifts are saved independently so that they can be used to account for the combined Doppler shifts
        data_comp = {'rv_starbar_solbar':data_dic['DI']['sysvel'][inst][vis]}
        idx_def = prop_dic[inst][vis]['idx_def']
        if data_type_gen=='DI':
            data_comp['star_starbar'] = {}
            data_comp['idx_aligned'] = idx_def
        if data_type_gen=='Intr': 
            data_comp['surf_star'] = {}
            
            #Surface RV 
            #    - indexes relative to in-transit table
            ref_pl,dic_rv,idx_def_rv = init_surf_shift(gen_dic,inst,vis,data_dic,data_dic['Intr']['align_mode'])
            data_comp['idx_aligned'] = np.intersect1d(idx_def,idx_def_rv)
        
            #Remove chromatic RVs
            #    - chromatic deviations from the 'white' average rv of the occulted stellar surface (due to variations in the planet size and stellar intensity) were already 
            # corrected for when extracting the intrinsic profiles
            if ('chrom' in dic_rv):dic_rv.pop('chrom')

        elif data_type_gen=='Atm': 
            data_comp['pl_star'] = {}
            
            #Orbital radial velocity of the planet calculated for each exposure in the star rest frame
            if data_type=='Absorption':data_comp['idx_aligned'] = list(np.array(gen_dic[inst][vis]['idx_in'])[idx_def])
            elif data_type=='Emission':data_comp['idx_aligned'] = idx_def
            rv_shifts = coord_dic[inst][vis][data_dic['Atm']['ref_pl_align']]['rv_pl'][data_comp['idx_aligned']]
            v_orb = coord_dic[inst][vis][data_dic['Atm']['ref_pl_align']]['v_pl'][data_comp['idx_aligned']]

        #Processing each in-transit exposure
        for isub,iexp in enumerate(data_comp['idx_aligned']):    

            #Upload latest processed data
            data_exp = dataload_npz(data_dic[inst][vis]['proc_'+data_type_gen+'_data_paths']+str(iexp))
          
            #Reflex motion and systemic velocity 
            if data_type_gen=='DI':    
                if data_dic['DI']['align_mode']=='kep': 
                    RV_star_stelCDM_exp = coord_dic[inst][vis]['RV_star_stelCDM'][iexp]
                    rv_shift_cen = RV_star_stelCDM_exp + data_dic['DI']['sysvel'][inst][vis]
                elif data_dic['DI']['align_mode']=='pip': 
                    if ('rv_pip' not in data_dic['DI'][inst][vis]):stop('ERROR : Pipeline RVs not available')
                    rv_shift_cen = data_dic['DI'][inst][vis]['rv_pip'][iexp]
                    RV_star_stelCDM_exp = rv_shift_cen - data_dic['DI']['sysvel'][inst][vis]
                spec_dopshift = 1./(gen_specdopshift(RV_star_stelCDM_exp)*gen_specdopshift(data_dic['DI']['sysvel'][inst][vis]))
                data_comp['star_starbar'][iexp] = RV_star_stelCDM_exp

            #Achromatic planet-occulted surface rv in the star rest frame
            elif data_type_gen=='Intr':
                rv_shift_cen = def_surf_shift(prop_dic['align_mode'],dic_rv,iexp,data_exp,ref_pl,data_vis['type'],data_dic['DI']['system_prop'],data_dic[inst][vis]['dim_exp'],data_dic[inst]['nord'],data_dic[inst][vis]['nspec'])[0]
                data_comp['surf_star'][iexp] = rv_shift_cen
                spec_dopshift = 1./gen_specdopshift(rv_shift_cen)
                
            #Orbital radial velocity of the planet in the star rest frame
            elif data_type_gen=='Atm':
                rv_shift_cen = rv_shifts[isub]
                data_comp['pl_star'][iexp] = rv_shifts[isub]
                spec_dopshift = 1./gen_specdopshift(rv_shift_cen , v_s = v_orb)
                
            #Aligning exposure profile and complementary tables
            #    - telluric, calibration, and 1D variance profiles must follow the same shifts as the exposure
            # + calibration profile used for scaling is always defined for S2D
            #   it is common to all exposures of a processed instrument, and is originally sampled over the table of each exposure in the detector rest frame
            # + calibration profile used as weight in temporal binning, or to scale back profiles from flux to count units, is only defined if requested
            if ('spec' in data_vis['type']) and gen_dic['corr_tell']:
                if ('tell_'+data_type_gen+'_data_paths' not in data_vis):stop('ERROR : weighing telluric profiles undefined; make sure you activate gen_dic["calc_proc_data"] and gen_dic["calc_corr_tell"] when running this module.')
                data_exp['tell'] = dataload_npz(data_vis['tell_'+data_type_gen+'_data_paths'][iexp])['tell'] 
            if data_vis['type']=='spec2D':
                data_exp['mean_gcal'] = dataload_npz(data_vis['mean_gcal_'+data_type_gen+'_data_paths'][iexp])['mean_gcal'] 
                data_gcal = dataload_npz(data_vis['sing_gcal_'+data_type_gen+'_data_paths'][iexp])
                data_exp['sing_gcal'] = data_gcal['gcal'] 
                if (vis in data_inst['gcal_blaze_vis']):data_exp['sdet2'] = data_gcal['sdet2'] 
            if var_key is not None:data_exp[var_key] = dataload_npz(data_vis[var_key+'_'+data_type+'_data_paths'][iexp])['var']  
            data_align=align_data(data_exp,data_vis['type'],data_dic[inst]['nord'],dim_exp_resamp,gen_dic['resamp_mode'],cen_bins_resamp, edge_bins_resamp,rv_shift_cen,spec_dopshift)
          
            #Saving aligned exposure and complementary tables
            if ('tell' in data_align):
                datasave_npz(proc_gen_data_paths_new+'_tell'+str(iexp),{'tell':data_align['tell']}) 
                data_align.pop('tell')
            if ('mean_gcal' in data_align):
                datasave_npz(proc_gen_data_paths_new+'_mean_gcal'+str(iexp),{'mean_gcal':data_align['mean_gcal']}) 
                data_align.pop('mean_gcal')
                data_gcal = {'gcal':deepcopy(data_align['sing_gcal'])}
                data_align.pop('sing_gcal')
                if 'sdet2' in data_align:
                    data_gcal['sdet2']=deepcopy(data_align['sdet2'])
                    data_align.pop('sdet2')  
                datasave_npz(proc_gen_data_paths_new+'_sing_gcal'+str(iexp),data_gcal) 
            if (var_key in data_align):
                datasave_npz(proc_gen_data_paths_new+'_'+var_key+str(iexp), {'var':data_align[var_key]})          
                data_align.pop(var_key)
            datasave_npz(proc_gen_data_paths_new+str(iexp),data_align)

            #Aligning weighing master
            #    - called for intrinsic and atmospheric profiles, when they are shifted from the star rest frame to other frames
            #      it is not called for DI profiles, since it is computed from DI profiles after alignment (ie, the disk-integrated master does not need further alignment)
            #    - master is shifted for Intr and Atm types 
            #    - the master is originally defined in the star rest frame, like the differential and then intrinsic profiles, but on the common table for the visit
            # + if profiles are shifted and resampled on the common table, this will also be the case of the associated master
            # + if profiles are shifted but kept on their individual tables, the master will remain defined on the common table without being resampled, and it is this table that is shifted
            #   the master table thus becomes specific to each exposure, but is still different from the table of the exposure
            #    - path to the master associated with current profile is updated
            if proc_mast:
                data_ref = dataload_npz(data_vis['mast_'+data_type_gen+'_data_paths'][iexp])
                data_ref_align=align_data(data_ref,data_vis['type'],data_dic[inst]['nord'],dim_exp_resamp,gen_dic['resamp_mode'],cen_bins_resamp,edge_bins_resamp,rv_shift_cen,spec_dopshift)
                np.savez_compressed(proc_gen_data_paths_new+'_ref'+str(iexp),data={'cen_bins':data_ref_align['cen_bins'],'edge_bins':data_ref_align['edge_bins'],'flux':data_ref_align['flux'],'cov':data_ref_align['cov']},allow_pickle=True)           

            #Retrieve and align estimate of local stellar profile for current exposure, if based on measured profiles
            #    - required to weigh binned atmospheric profiles                
            #    - only defined for in-transit exposures
            if proc_locEst and (iexp in data_vis['LocEst_Atm_data_paths']):
                data_est_loc=np.load(data_vis['LocEst_Atm_data_paths'][iexp]+'.npz',allow_pickle=True)['data'].item() 
                data_est_loc_align=align_data(data_est_loc,data_vis['type'],data_dic[inst]['nord'],dim_exp_resamp,gen_dic['resamp_mode'],cen_bins_resamp, edge_bins_resamp,rv_shift_cen,spec_dopshift,nocov = ~data_dic['Intr']['cov_loc_star']) 
                np.savez_compressed(proc_gen_data_paths_new+'estloc'+str(iexp),data=data_est_loc_align,allow_pickle=True)

        #Updating path to processed data and saving complementary data
        np.savez_compressed(proc_gen_data_paths_new+'_add',data=data_comp,allow_pickle=True)
                            
    #Retrieving data
    else: 
        #Updating path to processed data and checking it has been calculated
        check_data({'path':proc_gen_data_paths_new+'_add'}) 
    data_vis['proc_'+data_type_gen+'_data_paths'] = proc_gen_data_paths_new 
    prop_dic[inst][vis]['idx_def'] = dataload_npz(data_vis['proc_'+data_type_gen+'_data_paths']+'_add')['idx_aligned']

    #Update data dimensions
    if (data_vis['comm_sp_tab']):
        data_vis['nspec'] = nspec_resamp
        data_vis['dim_all'] = [data_vis['n_in_visit'],data_dic[inst]['nord'],data_vis['nspec']]
        data_vis['dim_exp'] = [data_dic[inst]['nord'],data_vis['nspec']]
        data_vis['dim_ord'] = [data_vis['n_in_visit'],data_vis['nspec']]    

    #Updating rest frame
    #    - rest frame of disk-integrated spectra is not updated if systemic velocity is 0
    if (data_type_gen!='DI') or (data_dic['DI']['sysvel'][inst][vis]!=0.):
        data_dic[data_type_gen][inst][vis]['rest_frame'] = {'DI':'star','Intr':'surf','Atm':'pl'}[data_type_gen]
    
    #Updating paths
    if proc_mast:data_vis['mast_'+data_type_gen+'_data_paths']={}
    if proc_locEst:data_vis['LocEst_Atm_data_paths'] = {}
    if ('spec' in data_vis['type']) and gen_dic['corr_tell']:
        data_vis['tell_'+data_type_gen+'_data_paths']={}  
    if data_vis['type']=='spec2D':
        data_vis['mean_gcal_'+data_type_gen+'_data_paths']={}  
        data_vis['sing_gcal_'+data_type_gen+'_data_paths']={}  
    if var_key is not None:data_vis[var_key+'_'+data_type_gen+'_data_paths']={}  
    for iexp in prop_dic[inst][vis]['idx_def']:
        if proc_mast:data_vis['mast_'+data_type_gen+'_data_paths'][iexp]=proc_gen_data_paths_new+'_ref'+str(iexp)
        if proc_locEst and (iexp in data_vis['LocEst_Atm_data_paths']):data_vis['LocEst_Atm_data_paths'][iexp] = proc_gen_data_paths_new+'estloc'+str(iexp) 
        if ('spec' in data_vis['type']) and gen_dic['corr_tell']:
            data_vis['tell_'+data_type_gen+'_data_paths'][iexp] = proc_gen_data_paths_new+'_tell'+str(iexp)
        if data_vis['type']=='spec2D':
            data_vis['mean_gcal_'+data_type_gen+'_data_paths'][iexp] = proc_gen_data_paths_new+'_mean_gcal'+str(iexp)  
            data_vis['sing_gcal_'+data_type_gen+'_data_paths'][iexp] = proc_gen_data_paths_new+'_sing_gcal'+str(iexp)  
        if var_key is not None:data_vis[var_key+'_'+data_type_gen+'_data_paths'][iexp] = proc_gen_data_paths_new+'_'+var_key+str(iexp)    
            
    return None







################################################################################################## 
#%% Flux scaling routines
################################################################################################## 

def rescale_profiles(data_inst,inst,vis,data_dic,coord_dic,exp_dur_d,gen_dic,plot_dic,system_param,theo_dic,ar_dic={}):
    r"""**Main flux scaling routine.**    

    Scales profiles to the correct relative broadband flux level.

    Spectra :math:`F_\mathrm{corr}(\lambda,t,v)` should have been set to the same flux balance as a reference spectrum at low resolution using `corr_Fbal()`.

    .. math::
       F_\mathrm{corr}(\lambda,t,v) = F_{\star}(\lambda,v) C_\mathrm{ref}(\lambda,v) L(t)
      
    Where :math:`C_\mathrm{ref}(\mathrm{\lambda \, in \, band \, B},v) \sim C_\mathrm{ref}(B,v)` represents a possible low-frequency deviation from the true stellar spectrum, and 
    `L(t)` represents the global flux deviation from the true stellar spectrum, not corrected for in previous modules. 
    
    We first convert all spectra from flux to (temporal) flux density units, so that they are equivalent except for flux variations.
    
    We then correct for `L(t)` by dividing the profiles by 

    .. math::   
       F^\mathrm{glob}(v,t) &= \frac{\mathrm{TF}_\mathrm{corr}(v,t)}{\mathrm{median}(t_k, \mathrm{TF}_\mathrm{corr}(v,t_k) )}  \\
       \mathrm{or}&  \\    
       F^\mathrm{glob}(v,t) &= \frac{\mathrm{TF}_\mathrm{corr}(v,t)}{F_\mathrm{sc} \int_{\mathrm{\lambda \, in \, full \, B}}{d\lambda }} 
       
    Where 

    .. math::  
       \mathrm{TF}_\mathrm{corr}(v,t) &= \int_{\mathrm{\lambda \, in \, full \, B}}{F_\mathrm{corr}(\lambda,t,v) d\lambda } \\
                                      &= L(t) C 

    The exact value of the (unocculted) global flux level does not matter within a visit and is set to the median flux of all spectra, 
    unless the user choses to impose a common flux density for all visits (in case disk-integrated data needs to be combined between visits).
          
    We then need to rescale spectrally the data so that 

    .. math::      
       F_\mathrm{sc}(\lambda,t) &= c(\lambda,t) F_\mathrm{corr}(\lambda,t) \\ 
                                &= F(\lambda,t) 
 
    To do so we use broadband spectral light curves, which by definition correspond to the ratio of in- and out-of-transit flux integrated over the band

    .. math::      
       LC(B,v,t) &= \frac{ \int_{\mathrm{\lambda \, in \, B}}{F(\lambda,t) d\lambda }}{\int_{\mathrm{\lambda \, in \, B}}{F_{\star}(\lambda,v) d\lambda } } \\
                    &= \frac{ \mathrm{TF}(B,t) }{ \mathrm{TF}_{\star}(B,v) } 
    
    We assume that :math:`c(\lambda,t)` has low-frequency variations (which should be the case, since the difference between :math:`F_\mathrm{corr}` and `F` comes from the changes in color balance).
    The correction thus implies

    .. math::      
       & \int_{\mathrm{\lambda \, in \, B}}{c(\lambda,t) F_\mathrm{corr}(\lambda,t) } = \int_{\mathrm{\lambda \, in \, B}}{ F(\lambda,t) }       \\             
       & c(B,t) \int_{\mathrm{\lambda \, in \, B}}{F_\mathrm{corr}(\lambda,t) } = \mathrm{TF}(B,t)     \\
       & c(B,t) = \frac{ \mathrm{TF}_{\star}(B,v) \mathrm{LC}(B,v,t) }{ \int_{\mathrm{\lambda \, in \, B}}{F_{\star}(\lambda,v) C_\mathrm{ref}(B,v) }  }   \\
       & c(B,t) = \frac{ \mathrm{TF}_{\star}(B,v) \mathrm{LC}(B,v,t)}{  \mathrm{TF}_{\star}(B,v) C_\mathrm{ref}(B,v)  }  \\
       & c(B,t) = \frac{ \mathrm{LC}(B,v,t) }{ C_\mathrm{ref}(B,v) }
 
    We set :math:`c(B,t) = \mathrm{LC}_\mathrm{theo}(B,t)` and interpolate the chromatic `c(B,t)` at all wavelengths in the spectrum to define :math:`c(\lambda,t)`
    
    .. math::      
       F_\mathrm{sc}(\lambda,t,v) &= \frac{ \mathrm{LC}_\mathrm{theo}(\lambda,t) F_\mathrm{corr}(\lambda,t)}{F^\mathrm{glob}(v,t)   }  \\
                                  &= F(\lambda,t,v) C_\mathrm{ref}(B,v)
      
        
    Care must be taken about :math:`C_\mathrm{ref}(B,v)` when several visits are exploited
    
    (1) if the stellar emission remains the same in all visits, then we can use a single theoretical spectrum / master as reference, and a constant normalized light curve       

        .. math::     
           F_\mathrm{sc}(\lambda,t,v) = F(\lambda,t,v) C_\mathrm{ref}(B)
 
    (2) if the stellar emission changes in amplitude uniformly over the star, the normalized light curve remains the same, and the stellar spectrum keeps the same profile   

        .. math::         
           &F_\mathrm{sc}(\lambda,t,v) = \frac{F(\lambda,t,v) C_\mathrm{ref}(B)}{F_r(v)}   \\
           &\mathrm{where \,} F_r(v) = \frac{F_{\star}(\lambda,v)}{F_{\star}(\lambda,v_\mathrm{ref})}

        The stellar spectrum of one of the visit :math:`F_{\star}(\lambda,v_\mathrm{ref})` is taken as reference, with 
                                                                                   
        .. math:: 
           \mathrm{LC}_\mathrm{theo}(\lambda,t) &= \frac{F(\lambda,v_\mathrm{ref},t)}{F_{\star}(\lambda,v_\mathrm{ref})}  \\
                                             &= \frac{F(\lambda,t,v)}{F_{\star}(\lambda,v)}
 
        Thus 
        
        .. math:: 
           F_\mathrm{sc}(\lambda,t,v) &= \frac{F(\lambda,t,v) C_\mathrm{ref}(B) F_{\star}(\lambda,v_\mathrm{ref})}{F_{\star}(\lambda,v) }     \\        
                                      &= \frac{F(\lambda,v_\mathrm{ref},t) C_\mathrm{ref}(B) F_{\star}(\lambda,v_\mathrm{ref})}{F_{\star}(\lambda,v_\mathrm{ref}) }       \\      
                                      &= F(\lambda,v_\mathrm{ref},t) C_\mathrm{ref}(B)  
     
        Scaled spectra thus keep the same profile, deviating from the true profile by a common :math:`C_\mathrm{ref}(B)` in all visits.

    (3) if the stellar emission changes in amplitude and shape uniformely over the star, the normalized light curve remains the same but the stellar spectrum has not the same profile

        .. math::        
           F_\mathrm{sc}(\lambda,t,v) &= \mathrm{LC}_\mathrm{theo}(\lambda,t) F(\lambda,v) C_\mathrm{ref}(\lambda,v) \\
                                      &= (F(\lambda,v,t)/F_{\star}(\lambda,v)) F_{\star}(\lambda,t) C_\mathrm{ref}(B,v)  \\
                                      &= F(\lambda,t,v) C_\mathrm{ref}(B,v)
            
        By using a reference spectrum specific to each visit when correcting the color balance, with the correct shape and relative flux, we would get        

        .. math::             
           F_\mathrm{sc}(\lambda,t,v) = F(\lambda,t,v) C_\mathrm{ref}

        Where :math:`C_\mathrm{ref}` is the deviation from the absolute flux level, common to all visits.
 
    (4) if local stellar spectra in specific regions of the star change between visits, eg due to (un)occulted spots, then the normalized light curve changes as well.
        Both a reference spectrum and a light curve specific to each visit must be used to remove the stellar contribution :math:`F_{\star}(\lambda,v)` and avoid introducing a different balance to the planet-occulted spectra. 


    If there is emission/reflection from the planet, then the true flux writes as
    
    .. math:: 
       F(\lambda,v,t) &= F_{\star}(\lambda,v) \mathrm{LC}^\mathrm{tr}(\lambda,v,t) + F_{\star}(\lambda,v) \mathrm{LC}^\mathrm{refl}(\lambda,v,t) + F_\mathrm{p}^\mathrm{thermal}(\lambda,v,t)      \\
                      &= F_{\star}(\lambda,v) ( \delta_p^\mathrm{tr}(\lambda,v,t) + \delta_p^\mathrm{refl}(\lambda,v,t) + \delta_p^\mathrm{thermal}(\lambda,v,t) )   \\
                      &= F_{\star}(\lambda,v) \delta_p(\lambda,v,t)   
            
    The correction should still ensure that
      
    .. math::      
       c(B,t) &= \frac{ \mathrm{TF}(B,v,t) }{  \mathrm{TF}_{\star}(B,v) C_\mathrm{ref}(B)  }    \\
                 &= \frac{ \mathrm{LC}(B,v,t) \mathrm{TF}_{\star}(B,v) }{  \mathrm{TF}_{\star}(B,v) C_\mathrm{ref}(B)  }    \\
                 &= \frac{ \mathrm{LC}(B,v,t) }{ C_\mathrm{ref}(B) }
 
    But broadband spectral light curves correspond to
      
    .. math::      
       \mathrm{LC}(B,v,t) &=  \frac{ \mathrm{TF}(B,v,t) }{   \mathrm{TF}_{\star}(B,v) }  \\ 
                             &=  \frac{ \int_{\mathrm{\lambda \, in \, B}}, F_{\star}(\lambda,v) \delta_p(\lambda,v,t) d\lambda ) }{   \mathrm{TF}_{\star}(B,v)   }
    
    So that care must be taken to use light curves that integrate the different spectral contributions and are then normalized by the integrated stellar flux, rather than integrating the pure planet contribution. 
        
      
    For spectra, we can use the flux integrated over bands including the planetary absorption, as we can match it with the same band over which the light curve was measured.
    Ideally the input light curves should be defined with a fine enough resolution to sample the low-frequency variations of the planetary atmosphere and stellar limb-darkening 


    For CCFs, a light curve does not match the cumulated flux of the different spectral regions used to calculate the CCF. 
    If we assume that the signature of the planet and of the RM effect is similar in all lines used to calculate the CCF, and that the light curve is achromatic, then it is similar to the scaling of a spectrum with a single line.
    Thus, as with spectra, CCFs should be scaled using the flux of their full profile and not just the continuum (although there will always be a bias in using CCFs at this stage and the use of spectra should be preferred).


    Since the scaling is imposed by a light curve independent of the spectra, bin by bin, it does not need to be applied to the full spectra like the flux balance color correction, and can be applied to any spectral range. 


    Note that the measured light curve corresponds to 
    
    .. math::    
       \mathrm{LC}(B,t) &= \frac{ \int_{\lambda \, in \, B}{ F_\mathrm{in}(\lambda,t) d\lambda } }{ \int_{\lambda \, in \, B}{ F_{\star}(\lambda,v) d\lambda }} \\
                           &= 1 - \frac{\int_{ \lambda \, in \, B}{ f_p(\lambda) ( S_\mathrm{thick}(B,t) + S_\mathrm{thin}(\lambda,t) ) d\lambda} }{ <F_{\star}(\mathrm{\lambda \, in \, B},v)> }
                      
    The theoretical light curves fitted to the measured one assume a constant, uniform stellar intensity `I` and limb-darkening law, and a constant average radius over the band
    
    .. math::      
       \mathrm{LC}^\mathrm{theo}(B,t) &= \frac{ \mathrm{TF}^\mathrm{theo}(B,t) }{ \mathrm{TF}_{\star}^\mathrm{theo}(B) } \\
                                         &= \frac{ F^\mathrm{theo}(B,t) }{ F_{\star}^\mathrm{theo}(B)  } \\
                                         &= 1 - \frac{ I^\mathrm{theo}(B) \mathrm{LD}(B,t) S_p(B,t)}{\sum_{k}{ I^\mathrm{theo}(B) \mathrm{LD}_k(B) S_k }  } \\
                                         &= 1 - \frac{ \mathrm{LD}(B,t) S_p(B,t)}{\sum_{k}{ \mathrm{LD}_k(B) S_k} } \\
                                         &= 1 - \frac{ \mathrm{LD}(B,t) S_p(B,t)}{S_{\star}^\mathrm{LD}(B) }
                 
    The fact that this is an approximation does not matter as long as the measured light curve is correctly reproduced, in which case the rescaling with the theoretical light curve is correct.
    In the following we assume that the theoretical LD matches the true LD.
        
        
    Out-of-transit data is rescaled as well, to account for possible variations of stellar origin in the disk-integrated flux.  

    Args:
        TBD
    
    Returns:
        None
    
    """  
    print('   > Broadband flux scaling') 
    data_vis=data_inst[vis]
    if data_dic['DI']['rescale_DI']:proc_DI_data_paths_new = gen_dic['save_data_dir']+'Scaled_data/'+inst+'_'+vis+'_'
    else:proc_DI_data_paths_new = deepcopy(data_vis['proc_DI_data_paths'])
    data_vis['scaled_DI_data_paths'] = proc_DI_data_paths_new+'scaling_'   

    #Check
    if (len(data_inst[vis]['studied_pl'])>0):
        cond_tr = True
    else:cond_tr = False
    if (len(data_inst[vis]['studied_ar'])>0):
        cond_ar = True
    else:cond_ar = False
    
    #Light curve
    cond_lc = True
    if (vis not in data_dic['DI']['transit_prop'][inst]):
        if (not cond_ar) and (not cond_tr):cond_lc = False
        else:
            if cond_ar:stop('ERROR: active regions are defined; use imported or simulated transit light curve')
            if cond_tr:
                if len(data_inst[vis]['studied_pl'])>1:stop('ERROR: multiple transiting planets; use imported or simulated transit light curve')
                print('         Default transit light curve model with single planet')
                data_dic['DI']['transit_prop'][inst][vis] = {'mode':'model','dt':np.min(coord_dic[inst][vis]['t_dur']/60.)/5.}
                transit_prop=data_dic['DI']['transit_prop'][inst][vis]
    else:
        transit_prop=data_dic['DI']['transit_prop'][inst][vis]
        if transit_prop['mode']=='imp':
            print('         Using imported light curve')   
        elif transit_prop['mode']=='model':
            if (not cond_ar) and (not cond_tr):cond_lc = False
            else:
                if cond_ar:stop('ERROR: active regions are defined; use imported or simulated transit light curve')
                if cond_tr:
                    if len(data_inst[vis]['studied_pl'])>1:stop('ERROR: multiple transiting planets; use imported or simulated transit light curve')
                    print('         Using transit light curve model with single planet')
        elif transit_prop['mode']=='simu': 
            print('         Using simulated light curve')               
    if not cond_lc:print('         Global scaling only')
    else:print('         Global scaling and light curve scaling')
    
    #Flux scaling application
    #    - if the data has absolute flux level, scaling should not be applied
    #      light curve scaling is however still defined so that weight profiles can be computed (since the relative flux scaling that the data naturally contains is still used in weight computations)
    #      global scaling unity is set to unity
    if not data_dic['DI']['rescale_DI']:
        print('         No scaling is applied. Computing light curve scaling and setting global scaling to 1.')
        
    #Calculating 
    if (gen_dic['calc_flux_sc']):
        print('         Calculating data')
        dic_save={}
    
        #Light curve for all bands and all exposures
        #    - chromatic values are used if provided and if disk-integrated profiles are in spectral mode
        if ('spec' in data_vis['type']) and ('chrom' in data_vis['system_prop']):key_chrom = ['chrom']
        else:key_chrom = ['achrom']
        system_prop = data_vis['system_prop'][key_chrom[0]]
        LC_flux_band_all = np.ones([data_vis['n_in_visit'],system_prop['nw']])    

        #------------------------------------------------------------------------ 
        #Light curve
        if cond_lc:
        
            #Simulated light curves
            if transit_prop['mode']=='simu':
                params_LC = deepcopy(system_param['star'])
                params_LC.update({'rv':0.,'cont':1.}) 
    
                #Include spots in the LC generation
                if ar_dic!={} and ('ar_prop' in ar_dic) and (inst in ar_dic['ar_prop']) and (vis in ar_dic['ar_prop'][inst]):
                    params_LC['use_ar']=True
                    if (data_dic['DI']['ar_prop']=={}):stop('WARNING: spot properties for simulated light curves are not defined')
                else:params_LC['use_ar']=False
    
            #Calculate light curve for plotting        
            if (plot_dic['input_LC']!='') or (plot_dic['prop_Intr']!=''):
                
                #High-resolution time table over visit
                min_bjd = coord_dic[inst][vis]['bjd'][0]
                max_bjd = coord_dic[inst][vis]['bjd'][-1]
                dbjd_HR = plot_dic['dt_LC']/(3600.*24.)
                nbjd_HR = round((max_bjd-min_bjd)/dbjd_HR)
                bjd_HR=min_bjd+dbjd_HR*np.arange(nbjd_HR)
          
                #Corresponding orbital phases and coordinates for each planet
                #    - high-resolution tables are calculated assuming no exposure duration
                coord_HR = {}
                LC_HR=np.ones([nbjd_HR,system_prop['nw']],dtype=float)  
                if transit_prop['mode']=='simu':ecl_all_HR = np.zeros(nbjd_HR,dtype=bool)
                for pl_loc in data_inst[vis]['studied_pl']:
                    pl_params_loc=system_param[pl_loc]
                    coord_HR[pl_loc]={'cen_ph':get_timeorbit(coord_dic[inst][vis][pl_loc]['Tcenter'],bjd_HR,pl_params_loc,None)[1]}   
    
                    #Definition of coordinates for all transiting planets
                    if transit_prop['mode']=='simu': 
                        x_pos_pl,y_pos_pl,z_pos_pl,Dprojp,_,_,_,_,ecl_pl = calc_pl_coord(pl_params_loc['ecc'],pl_params_loc['omega_rad'],pl_params_loc['aRs'],pl_params_loc['inclin_rad'],coord_HR[pl_loc]['cen_ph'],data_dic['DI']['system_prop']['achrom'][pl_loc][0],pl_params_loc['lambda_rad'],system_param['star'])
                        coord_HR[pl_loc].update({'ecl':ecl_pl,'cen_pos':np.vstack((x_pos_pl,y_pos_pl,z_pos_pl))})
        
                        #Exposure considered out-of-transit if no planet at all is transiting
                        ecl_all_HR |= abs(ecl_pl)!=1                      
    
                #Corresponding orbital phases and coordinates for each spot
                if (transit_prop['mode']=='simu') and (params_LC['use_ar']): 
                    ar_prop_HR = retrieve_ar_prop_from_param(ar_dic['ar_prop'][inst][vis], inst, vis)
                    ar_prop_HR['cos_istar']=system_param['star']['cos_istar']
                    for spot in data_inst[vis]['studied_ar']:
                        coord_HR[spot]={}
                        for key in gen_dic['ar_coord_par']:coord_HR[spot][key] = np.zeros([3,len(bjd_HR)],dtype=float)*np.nan
                        coord_HR[spot]['is_visible'] = np.zeros([3,len(bjd_HR)],dtype=bool)
                        for key in ['Tc_ar', 'ang_rad', 'lat_rad', 'fctrst']:coord_dic[inst][vis][spot][key] = ar_prop_HR[spot][key] 
        
                    #Retrieving the spot coordinates for all the times that we have
                    for itstamp, tstamp in enumerate(bjd_HR):
                        for spot in data_inst[vis]['studied_ar']:
                            ar_prop_exp = coord_expos_ar(spot,tstamp,ar_prop_HR,system_param['star'],dbjd_HR,gen_dic['ar_coord_par'])                           
                            for key in ar_prop_exp:coord_HR[spot][key][:, itstamp] = [ar_prop_exp[key][0],ar_prop_exp[key][1],ar_prop_exp[key][2]] 

            #------------------------------------------------------------------------        

            #Light curves from import
            #    - defined over a set of wavelengths that can be different for each visit
            #    - here we import the light curves, so that they can be interpolated for each visit after their exposures have been defined
            if transit_prop['mode']=='imp':
                t_dur_d=coord_dic[inst][vis]['t_dur']/(3600.*24.)
                cen_bjd = coord_dic[inst][vis]['bjd']
              
                #Retrieving light curve
                #    - first column must be absolute time (BJD), to be independent of a specific planet 
                #    - next columns must be normalized stellar flux for all chosen bands, in the same order as data_dic['DI']['system_prop']['chrom']['w']
                ext = transit_prop['path'].split('.')[-1]
                if (ext=='csv'):
                    imp_LC = (pd.read_csv(transit_prop['path'])).values
                elif (ext in ['txt','dat']):
                    imp_LC = np.loadtxt(transit_prop['path']).T          
                else:
                    stop('Light curve path extension TBD') 
                imp_LC[0] -= 2400000. 
                if (plot_dic['input_LC']!=''):dic_save['imp_LC'] = imp_LC
             
                #Average imported light curve within the exposure time windows
                #    - the light curve must be imported with sufficient temporal resolution
                for iexp,(bjd_loc,dt_loc) in enumerate(zip(cen_bjd,t_dur_d)):
    
                    #Imported points within exposure
                    id_impLC=np_where1D( (imp_LC[0]>=bjd_loc-0.5*dt_loc) & (imp_LC[0]<=bjd_loc+0.5*dt_loc))
                  
                    #Normalized flux averaged within exposure
                    if len(id_impLC)>0:LC_flux_band_all[iexp,:]=np.mean(imp_LC[1::,id_impLC],axis=1)
                    else:stop('No LC measurements within exposure')
    
                #Calculate light curve for plotting        
                if (plot_dic['input_LC']!='') or (plot_dic['prop_Intr']!=''):     
                    for iband in range(system_prop['nw']):LC_HR[:,iband] = np_interp(bjd_HR,imp_LC[0],imp_LC[1+iband],left=imp_LC[1+iband,0],right=imp_LC[1+iband,-1])
    
            #------------------------------------------------------------------------
            
            #Model light curve for a single planet
            #    - can be oversampled   
            #    - defined over a set of wavelengths but constant for each visit
            elif transit_prop['mode']=='model':
                pl_vis = data_inst[vis]['studied_pl'][0]
                LC_params = batman.TransitParams()
                LC_pl_params = system_param[pl_vis]
            
                #Phase reference for inferior conjunction
                LC_params.t0 = 0. 
                
                #Orbital period in phase
                LC_params.per = 1. 
                
                #Semi-major axis (in units of stellar radii)
                LC_params.a = LC_pl_params['aRs']
                
                #Orbital inclination (in degrees)
                #    - from the line of sight to the normal to the orbital plane
                LC_params.inc = LC_pl_params['inclination'] 
                
                #Eccentricity
                LC_params.ecc = LC_pl_params['ecc']
                
                #Longitude of periastron (in degrees)
                LC_params.w = LC_pl_params['omega_deg']
                
                #Oversampling 
                if ('dt' not in transit_prop):LC_osamp = np.repeat(10,data_vis['n_in_visit'])
                else:LC_osamp = npint(np.ceil(coord_dic[inst][vis]['t_dur']/(60.*transit_prop['dt'])))
                if np.min(LC_osamp)<2.:print('WARNING: no oversampling of model light curve')
                
                #Calculate white or chromatic light curves
                cen_ph_pl = coord_dic[inst][vis][pl_vis]['cen_ph']
                ph_dur_pl=coord_dic[inst][vis][pl_vis]['ph_dur']
                for iband,wband in enumerate(system_prop['w']):
        
                    #Light curve properties for the band
                    LC_params_band = deepcopy(LC_params)
        
                    #LD law 
                    LD_mod = system_prop['LD'][iband]
            
                    #Limb darkening coefficients in the format required for batman
                    LC_params_band.limb_dark = LD_mod
                    if LD_mod == 'uniform':
                        ld_coeff=[]
                    elif LD_mod == 'linear':
                        ld_coeff=[system_prop['LD_u1'][iband]]
                    elif LD_mod in ['quadratic' ,'squareroot','logarithmic', 'power2' ,'exponential']:
                        ld_coeff=[system_prop['LD_u1'][iband],system_prop['LD_u2'][iband]]
                    elif LD_mod == 'nonlinear':   
                        ld_coeff=[system_prop['LD_u1'][iband],system_prop['LD_u2'][iband],system_prop['LD_u3'][iband],system_prop['LD_u4'][iband]]           
                    else:
                        stop('Limb-darkening not supported by batman')  
                    LC_params_band.u=ld_coeff
            
                    #Planet-to-star radius ratio
                    LC_params_band.rp=system_prop[pl_vis][iband]
    
                    #All exposures have same duration
                    #    - process each band for all exposures together
                    if coord_dic[inst][vis]['cst_tdur']:
                        LC_flux_band_all[:,iband] = batman.TransitModel(LC_params_band, cen_ph_pl, supersample_factor = LC_osamp[0], exp_time = ph_dur_pl[0]).light_curve(LC_params_band)
                        
                    #Exposures have different durations
                    #    - process each band and each exposure
                    else:                      
                        for iexp,(cen_ph_exp,ph_dur_exp,LC_osamp_exp) in enumerate(zip(cen_ph_pl,ph_dur_pl,LC_osamp)):                    
                            LC_flux_band_all[iexp,iband]=float(batman.TransitModel(LC_params_band, np.array([cen_ph_exp]), supersample_factor = LC_osamp_exp, exp_time = np.array([ph_dur_exp])).light_curve(LC_params_band))
                            
                    #Calculate light curve for plotting        
                    if (plot_dic['input_LC']!='') or (plot_dic['prop_Intr']!=''):
                        LC_HR[:,iband] = batman.TransitModel(LC_params_band,coord_HR[pl_vis]['cen_ph']).light_curve(LC_params_band)  
                
            #------------------------------------------------------------------------
         
            #Simulated light curve   
            #    - can account for multiple transiting planets
            elif transit_prop['mode']=='simu':    
                
                #Set out-of-transit values to unity
                #    - values will be redefined if relevant  
                LC_flux_band_all[gen_dic[inst][vis]['idx_out'],:]=1.        
                
                #Oversampling factor, in units of RpRs
                theo_dic_LC= deepcopy(theo_dic)
                theo_dic_LC['d_oversamp_pl']={}
                theo_dic_LC['n_oversamp_ar']={}
                if (transit_prop['n_oversamp']>0.):
                    for pl_loc in data_inst[vis]['studied_pl']:theo_dic_LC['d_oversamp_pl'][pl_loc]=data_dic['DI']['system_prop']['achrom'][pl_loc][0]/transit_prop['n_oversamp'] 
                if params_LC['use_ar']:
                    for spot in data_inst[vis]['studied_ar']:theo_dic_LC['n_oversamp_ar'][spot]=1
    
                #Calculate transit light curves accounting for all planets in the visit
                fixed_args = {}
                if params_LC['use_ar']:
                    fixed_args['ar_coord_par']=gen_dic['ar_coord_par']
                    fixed_args['rout_mode']='Intr_prop'
                plocc_prop,_,common_prop = sub_calc_plocc_ar_prop(key_chrom,fixed_args,[],data_inst[vis]['studied_pl'],[],system_param,theo_dic_LC,data_dic['DI']['system_prop'],params_LC,coord_dic[inst][vis],range(data_vis['n_in_visit']),system_ar_prop_in=data_dic['DI']['ar_prop'],Ftot_star=True) 
                if not params_LC['use_ar']:LC_flux_band_all[gen_dic[inst][vis]['idx_in'],:]=plocc_prop[key_chrom[0]]['Ftot_star'][:, gen_dic[inst][vis]['idx_in']].T
                else:LC_flux_band_all=common_prop[key_chrom[0]]['Ftot_star'].T    
    
                #Calculate light curve for plotting        
                if (plot_dic['input_LC']!='') or (plot_dic['prop_Intr']!=''):
                    theo_dic_LC['d_oversamp_pl']={}
                    if not params_LC['use_ar']:idx_HR = np_where1D(ecl_all_HR)
                    else:idx_HR = np.arange(nbjd_HR)                
                    plocc_prop_HR,_,common_prop_HR = sub_calc_plocc_ar_prop(key_chrom,fixed_args,[],data_inst[vis]['studied_pl'],[],system_param,theo_dic_LC,data_dic['DI']['system_prop'],params_LC,coord_HR,idx_HR,system_ar_prop_in=data_dic['DI']['ar_prop'],Ftot_star=True)
                    if not params_LC['use_ar']:LC_HR[idx_HR,:]=plocc_prop_HR[key_chrom[0]]['Ftot_star'].T
                    else:LC_HR[idx_HR,:]=common_prop_HR[key_chrom[0]]['Ftot_star'].T
    
            #Store for plots
            if (plot_dic['input_LC']!='') or (plot_dic['prop_Intr']!=''):
                dic_save['flux_band_all'] = LC_flux_band_all
                dic_save['coord_HR'] = coord_HR
                dic_save['LC_HR'] = LC_HR

        #------------------------------------------------------------------------

        #Upload common spectral table
        #    - if profiles are defined on different tables they are resampled on this one
        #      if they are already defined on a common table, it is this one, which has been kept the same since the beginning of the routine
        #    - if alignment in the star rest frame was not applied, the common star table points toward the common input table
        data_com = dataload_npz(data_vis['proc_com_star_data_paths'])
        
        #Spectral scaling table and global scaling range
        if (not data_vis['comm_sp_tab']):dim_all = data_com['dim_all']
        else:dim_all = data_vis['dim_all']
        loc_flux_scaling = np.zeros(data_vis['n_in_visit'],dtype=object) 
        flux_all = np.zeros(dim_all,dtype=float)*np.nan
        cond_def_all = np.zeros(dim_all,dtype=bool)
        cond_def_scal_all  = np.zeros(dim_all,dtype=bool)
        null_loc_flux_scaling = np.ones(data_vis['n_in_visit'],dtype=bool)
        for isub,iexp in enumerate(range(data_vis['n_in_visit'])): 
            
            #Latest processed DI data
            data_exp = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp))
            
            #Resampling and conversion to temporal flux density
            #    - if data were kept on independent tables they need to be resampled on a common one to calculate equivalent fluxes
            if (not data_vis['comm_sp_tab']):
                for iord in range(data_inst['nord']): 
                    flux_all[iexp,iord] = bind.resampling(data_com['edge_bins'][iord], data_exp['edge_bins'][iord], data_exp['flux'][iord] , kind=gen_dic['resamp_mode'])                                                        
                cond_def_all[iexp] = ~np.isnan(flux_all[iexp])   
                if isub==0:edge_bins_com = data_com['edge_bins']
            else:
                flux_all[iexp] = data_exp['flux']
                cond_def_all[iexp] = data_exp['cond_def']
                if isub==0:edge_bins_com = data_exp['edge_bins']
            flux_all[iexp]/=coord_dic[inst][vis]['t_dur'][iexp]

            #Spectral scaling table                                        
            #    - scale to the expected flux level at all wavelengths, using the broadband flux interpolated over the full spectrum range, unless a single band is used
            #    - accounts for the potentially chromatic signature of the planet 
            #    - if there is no transiting planet, or active regions, of stellar flux modulations (ie, if LC_flux_band_all = 1 at all times) the 'null_loc_flux_scaling' flag remains set to True
            #      'loc_flux_scaling' is still defined as a function returning 0, for later use 
            if np.max(np.abs(1.-LC_flux_band_all[iexp]))>0.:null_loc_flux_scaling[iexp] = False
            if (system_prop['nw']==1) or (null_loc_flux_scaling[iexp]):loc_flux_scaling[iexp] = np.poly1d([1.-LC_flux_band_all[iexp,0]])
            else:loc_flux_scaling[iexp] = interp1d(system_prop['w'],1.-LC_flux_band_all[iexp],fill_value=(1.-LC_flux_band_all[iexp,0],1.-LC_flux_band_all[iexp,-1]), bounds_error=False)
                
            #Global scaling
            if data_dic['DI']['rescale_DI']:
                
                #Requested scaling range
                if len(data_dic['DI']['scaling_range'])>0:
                    cond_def_scal=False 
                    for bd_int in data_dic['DI']['scaling_range']:cond_def_scal |= (edge_bins_com[:,0:-1]>=bd_int[0]) & (edge_bins_com[:,1:]<=bd_int[1])   
                else:cond_def_scal=True 
    
                #Accounting for undefined pixels in scaling range            
                cond_def_scal_all[iexp] = cond_def_all[iexp]  & cond_def_scal            

        #Defining global scaling values
        #    - used to set all profiles to a common global flux level
        #    - spectral profiles have been trimmed, corrected, and aligned
        #      it might thus be more accurate to rescale them using their own flux rather than the flux of the original spectra
        #      furthermore masters afterward will be calculated from these profiles, scaled, thus they do not need to be set to the level of the original global master
        #      we thus use the total flux summed over the full range of the current profiles, with their median taken as reference
        #    - defined on temporal flux density (not cumulated photoelectrons counts)
        if data_dic['DI']['rescale_DI']:

            #Scaling pixels common to all exposures
            #    - planetary signatures should not be excluded from the range of summation, for the same reason as they are included in the spectral scaling : the light curves used for the scaling include those ranges potentially absorbed by the planet
            #      the same logic applies to CCF: their full range must be used for the scaling, and not just the continuum      
            cond_scal_com  = np.all(cond_def_scal_all,axis=0)
            if np.sum(cond_scal_com)==0.:stop('No pixels in common scaling range')               
            
            #Global scaling
            Tflux_all = np.zeros(data_vis['n_in_visit'],dtype=float)
            dcen_bin_comm = (edge_bins_com[:,1::] - edge_bins_com[:,0:-1])
            Tcen_bin_comm = 0.
            for iord in range(data_inst['nord']):    
                Tflux_all += np.sum(flux_all[:,iord,cond_scal_com[iord]]*dcen_bin_comm[iord,cond_scal_com[iord]],axis=1)
                Tcen_bin_comm += np.sum(dcen_bin_comm[iord,cond_scal_com[iord]])
            if data_dic['DI']['scaling_val'] is None:Tflux_ref = np.median(Tflux_all)
            else:Tflux_ref=Tcen_bin_comm*data_dic['DI']['scaling_val']            
            norm_exp_glob = Tflux_all/Tflux_ref
        else:norm_exp_glob = np.ones(data_vis['n_in_visit'],dtype=float)

        #Scaling each exposure
        #    - only defined bins are scaled (the flux in undefined bins remain set to nan), but the scaling spectrum was calculated at all wavelengths so that it can be used later with data for which different bins are defined or not
        #    - all defined bins remain defined 
        #    - operation depends on condition 'rescale_DI' because flux scaling tables may be required even if data needs not be scaled
        for iexp in range(data_vis['n_in_visit']):  
            
            #Scale and save exposure
            if data_dic['DI']['rescale_DI']: 
                data_exp = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp)) 
                for iord in range(data_inst['nord']): 
                    if null_loc_flux_scaling[iexp]:flux_sc_ord = np.ones(data_vis['nspec'],dtype=float) 
                    else:flux_sc_ord=(1.-loc_flux_scaling[iexp](data_exp['cen_bins'][iord]))
                    data_exp['flux'][iord],data_exp['cov'][iord] = bind.mul_array(data_exp['flux'][iord],data_exp['cov'][iord],flux_sc_ord/(coord_dic[inst][vis]['t_dur'][iexp]*norm_exp_glob[iexp]))
                datasave_npz(proc_DI_data_paths_new+str(iexp),data_exp)
            
            #Save scaling
            #    - must be saved for each exposure because (for some reason) the interp1d function cannot be saved once passed out of multiprocessing 
            #    - in-transit scaling can be used later to manipulate local profiles from the planet-occulted regions 
            data_scaling = {'loc_flux_scaling':loc_flux_scaling[iexp],'glob_flux_scaling':norm_exp_glob[iexp],'null_loc_flux_scaling':null_loc_flux_scaling[iexp]}
            if system_prop['nw']>1:data_scaling['chrom']=True
            else:data_scaling['chrom']=False
            datasave_npz(data_vis['scaled_DI_data_paths']+str(iexp),data_scaling)                
  
        #Saving complementary data
        dic_save['rest_frame']=data_dic['DI'][inst][vis]['rest_frame']
        datasave_npz(proc_DI_data_paths_new+'add',dic_save)           
        
    #Updating path to processed data and checking it has been calculated
    else:   
        check_data({'path':proc_DI_data_paths_new+str(0)})   
    data_vis['proc_DI_data_paths'] = proc_DI_data_paths_new

    return None
    












################################################################################################## 
#%% Differential profiles routines
################################################################################################## 

def extract_diff_profiles(gen_dic,data_dic,inst,vis,data_prop,coord_dic):
    r"""**Main differential profile routine.** 

    Extracts differential profiles in the stellar rest frame.

    The real measured spectrum can be written as 
    
    .. math::    
       F(\lambda,t,v) = F_{\star}(\lambda,t,v) \delta_p(\lambda,t,v) 
 
    It can also be decomposed spatially as 
      
    .. math::
       F(\lambda,t,v) &= (\sum_\mathrm{i \, not \, occulted}{F_i(\lambda,t,v)}) + f_p(\lambda,v) (S_\mathrm{occ}(t) - S_\mathrm{thick}(B,t) - S_\mathrm{thin}(\lambda,t) ) + F_p(\lambda,t)       \\
                      &= F_{\star}(\lambda,t,v) - f_p(\lambda,v) ( S_\mathrm{thick}(B,t) + S_\mathrm{thin}(\lambda,t) ) + F_p(\lambda,t) 
          
    With
     
     - :math:`\lambda` the absolute wavelength in the stellar rest frame, within a given spectral band. 
     - `F` the flux received from the star - planet system :math:`[erg \, s^{-1} \,  A^{-1} \, cm^{-2}]` at a given distance
       :math:`F_i(\lambda,v) = f_i(\lambda,v) S_i` is the flux emitted by the region `i` of surface :math:`S_i`.
     - the surface density flux can be written as :math:`f_i(\lambda,v) = I_i(\lambda) \mathrm{LD}_i(\lambda) \mathrm{GD}_i(\lambda)`. 
       
       + :math:`I_i(\lambda)` is the specific intensity in the direction of the LOS :math:`[erg \, s^{-1} \,  A^{-1} \, cm^{-2} \, sr^{-1}]`
         which can be written as :math:`I_0(\lambda-\lambda_{\star}(t))` if the local stellar emission is constant over the stellar disk, and simply shifted by :math:`\lambda_{\star}(t)` because of surface velocity.
       + :math:`\mathrm{LD}_i(\lambda)` is the spectral limb-darkening law.
       + :math:`\mathrm{GD}_i(\lambda)` the gravity-darkening law.
     - :math:`F_p(\lambda,t)` is the flux emitted or reflected by the planet.                
       The planet and its atmosphere occult a surface :math:`S_\mathrm{occ}(t)`, time-dependent because of partial occultation at ingress/egress.    

       + :math:`S_\mathrm{thick}(B,t)` is the equivalent surface of the planet disk opaque to light in the local spectral band. 
       + :math:`S_\mathrm{thin}(\lambda,t)` is the equivalent surface of the atmospheric annulus optically thin to light in the band, varying at high frequency and null outside of narrow absorption lines.  
       + :math:`F_p(\lambda,t)` and :math:`S_\mathrm{thin}(\lambda,t)` are sensitive to the planet orbital motion, and shifted in the star rest frame by :math:`\lambda_\mathrm{pl}(t)`.
    
    The measured profiles are now defined in the most general case as (see `rescale_profiles()`)
    
    .. math::
       F_\mathrm{sc}(\lambda,t,v) = F(\lambda,t,v) C_\mathrm{ref}(\lambda_\mathrm{B},v)
 
    With :math:`F(\lambda,t,v)` the true spectrum.
    Since all spectra were set to the same balance, corresponding to the stellar spectrum times a low-frequency coefficient, the master out corresponds in a given visit to 

    .. math::    
        F^\mathrm{mast}_{\star}(\lambda,v) = F_{\star}(\lambda,v) C_\mathrm{ref}(\lambda_\mathrm{B},v)  

    Note that the unknown scaling of the flux due to the distance to the star is implicitely included in :math:`C_\mathrm{ref}`.
    We can decompose :math:`F_{\star}` as: 
          
    .. math::
       F_{\star}(\lambda,v) = (\sum_\mathrm{i \, not \, occulted}{F_i(\lambda,v)}) + f_p(\lambda,v) S_\mathrm{occ}(t)  
        
  
    The local differential profiles are calculated as
     
    .. math::
       F_\mathrm{diff}(\lambda,t,v) &= F^\mathrm{mast}_{\star}(\lambda,v) - F_\mathrm{sc}(\lambda,t,v)   \\              
                                   &= F_{\star}(\lambda,v) C_\mathrm{ref}(\lambda_\mathrm{B},v)  - F(\lambda,t,v) C_\mathrm{ref}(\lambda_\mathrm{B},v)   \\
                                   &= ( F_{\star}(\lambda,v) - F(\lambda,t,v) ) C_\mathrm{ref}(\lambda_\mathrm{B},v)   \\
                                   &= ( F_{\star}(\lambda,v) - F_{\star}(\lambda,t,v) + f_p(\lambda,v) ( S_\mathrm{thick}(B,t) + S_\mathrm{thin}(\lambda,t) ) - F_p(\lambda,t)  ) C_\mathrm{ref}(\lambda_\mathrm{B},v)
      
    Here we make the assumption that :math:`F_{\star}(\lambda,t,v) \sim F_{\star}(\lambda,v)`, but care must be taken not to neglect uncertainties on :math:`F_{\star}(\lambda,t,v)` when propagating errors, even if uncertainties on the reference :math:`F_{\star}(\lambda,v)` can be neglected.

    .. math::
       F_\mathrm{diff}(\lambda,t,v) &= ( f_p(\lambda,v) ( S_\mathrm{thick}(B,t) + S_\mathrm{thin}(\lambda,t) ) -  F_p(\lambda,t) ) C_\mathrm{ref}(\lambda_\mathrm{B},v)  \\
                                &= ( f_p(\lambda,v) S_p(\lambda,t) -  F_p(\lambda,t) ) C_\mathrm{ref}(\lambda_\mathrm{B},v)
                                   
    Where :math:`S_p(\lambda,t)` represents the equivalent surface occulted by the opaque planetary disk and its optically thin atmosphere, at each wavelength.


    If there is no contribution from the atmosphere, or its contamination is excluded
     
    .. math::
       F_\mathrm{diff}(\lambda,t,v) = f_p(\lambda,v) S_\mathrm{thick}(B,t) C_\mathrm{ref}(\lambda_\mathrm{B},v) 
 
    CCFs computed on these :math:`F_\mathrm{diff}` have the same contrast, FWHM, and RV between visits, as long as the intrinsic lines are comparable.
    
    
    It is possible that :math:`F_{\star}(\lambda,v)` varies with time, during or outside of the transit, eg if a spot/plage is present on the star and changes during the observations (in particular with the star's rotation).
    In that case, rather than :math:`F^\mathrm{mast}_{\star}(\lambda,v)` we would need to use a :math:`F_{\star}(\lambda,t,v)` representative of the star at the time of each exposure.


    Binned disk-integrated profiles are calculated from spectra resampled directly on the table of each exposure, to avoid blurring spectral features, losing resolution, and introducing spurious features
    when doing differences and ratio between the master and individual spectra (which we found to be the case if using a single master calculated on a common table and then resampled on the table of each exposure).


    If a bin is undefined in the master and/or the exposure, it will be undefined in the local profile.

    Args:
        TBD
    
    Returns:
        None
    
    """  
    print('   > Extracting differential profiles')
    data_inst = data_dic[inst]
    data_vis = data_inst[vis]
    
    #Current rest frame
    if data_dic['DI'][inst][vis]['rest_frame']!='star':print('         WARNING: disk-integrated profiles must be aligned')
    data_dic['Diff'][inst][vis]['rest_frame'] = 'star'

    #Path to initialized local data
    proc_gen_data_paths_new=gen_dic['save_data_dir']+'Diff_data/'+inst+'_'+vis+'_'

    #Exposures for which local profiles will be extracted
    #    - the user can request extraction for in-transit exposures alone (to avoid computing time)
    #      we force the extraction for all exposures if a common master is used for the extraction (ie, when exposures are resampled on a common table) and no time is required to recalculate the master for each exposure
    if data_dic['Diff']['extract_in'] and ('spec' in data_vis['type']) and (not data_vis['comm_sp_tab']):data_dic['Diff'][inst][vis]['idx_to_extract'] = deepcopy(gen_dic[inst][vis]['idx_in'])
    else:data_dic['Diff'][inst][vis]['idx_to_extract'] =  np.arange(data_vis['n_in_visit'],dtype=int) 
    data_dic['Diff'][inst][vis]['idx_def'] = data_dic['Diff'][inst][vis]['idx_to_extract'] 

    #Calculating
    if (gen_dic['calc_diff_data']):
        print('         Calculating data')     

        #Phase range from which original exposures contributing to the master are taken
        #    - we impose that a single master be used    
        bin_prop = {'bin_low':[-0.5],'bin_high':[0.5]}  

        #Binning mode
        #   - using current visit exposures only, or exposures from multiple visits
        if (inst in data_dic['Diff']['vis_in_bin']) and (len(data_dic['Diff']['vis_in_bin'][inst])>0):
            mode='multivis'
            vis_to_bin = data_dic['Diff']['vis_in_bin'][inst]  
            for vis_bin in vis_to_bin:
                if data_dic[inst][vis_bin]['type']!=data_vis['type']:stop('Binned disk-integrated profiles must be of the same type as processed visit')                  
        else:
            mode=''
            vis_to_bin = [vis]   

        #Automatic definition of reference planet for single-transiting planet  
        #    - for multiple planets the reference planet for the phase does not matter, we just require that all exposures are selected and then the selection is done via their indexes
        ref_pl={} 
        for vis_loc in vis_to_bin:
            if ('pl_in_bin' in data_dic['DI']) and (inst in data_dic['DI']['pl_in_bin']) and (vis_loc in data_dic['DI']['pl_in_bin'][inst]):
                ref_pl[vis_loc] = data_dic['DI']['pl_in_bin'][inst][vis_loc]
            else:
                ref_pl[vis_loc] = data_inst[vis_loc]['studied_pl'][0]  
            
        #Check for multiple reference planets
        bin_prop['multi_flag'] = False
        if (mode=='multivis'):
            if (len(np.unique(list(ref_pl.values())))>1):bin_prop['multi_flag'] = True
        
        #Initializing weight calculation conditions
        calc_EFsc2,calc_var_ref2,calc_flux_sc_all,var_key_def = weights_bin_prof_calc('DI','DI',gen_dic,data_dic,inst)     

        #Initialize binning
        #    - output tables contain a single value, associated with the single master (=binned profiles) used for the extraction 
        _,_,_,_,n_in_bin_all,idx_to_bin_all,dx_ov_all,_,idx_bin2orig,idx_bin2vis,idx_to_bin_unik = init_bin_prof('DI',ref_pl,data_dic['Diff']['idx_in_bin'],'phase',coord_dic,inst,vis_to_bin,data_dic,gen_dic,bin_prop)
        scaled_data_paths_vis = {}  
        iexp_no_plrange_vis = {}
        exclu_rangestar_vis = {}
        for vis_bin in vis_to_bin:
            if gen_dic['flux_sc'] and calc_flux_sc_all:scaled_data_paths_vis[vis_bin] = data_dic[inst][vis_bin]['scaled_DI_data_paths']
            else:scaled_data_paths_vis[vis_bin] = None
            if ('DI_Mast' in data_dic['Atm']['no_plrange']):iexp_no_plrange_vis[vis_bin] = data_dic['Atm'][inst][vis_bin]['iexp_no_plrange']
            else:iexp_no_plrange_vis[vis_bin] = {}
            exclu_rangestar_vis[vis_bin] = data_dic['Atm'][inst][vis_bin]['exclu_range_star']

        #Retrieving data that will be used in the binning to define the master disk-integrated profile
        #    - in process_bin_prof() all profiles are resampled on the common table before being binned, thus they can be resampled when uploaded the first time
        #    - here the binned profiles must be defined on the table of each processed exposure, so the components of the weight profile are retrieved here and then either copied or resampled if necessary for each exposure
        #      here a single binned profile (the master) is calculated, thus 'idx_to_bin_unik' is the same as idx_to_bin_all, which contains a single element
        data_to_bin_gen={}    
        resamp_cond = {}
        resamp_cond_all = False
        for iexp_off in idx_to_bin_unik:
            data_to_bin_gen[iexp_off]={}

            #Original index and visit of contributing exposure
            #    - index is relative to the global table
            iexp_glob = idx_bin2orig[iexp_off]
            vis_bin = idx_bin2vis[iexp_off]
            
            #Latest processed disk-integrated data and associated tables
            #    - profiles should have been aligned in the star rest frame and rescaled to their correct flux level, if necessary    
            #    - if profiles were converted into 1D we use directly the variance tables associated with DI profiles
            #      no modifications were applied since the conversion, so no resampling is required   
            data_exp_off = dataload_npz(data_inst[vis_bin]['proc_DI_data_paths']+str(iexp_glob))
            for key in ['cen_bins','edge_bins','flux','cond_def','cov']:data_to_bin_gen[iexp_off][key] = data_exp_off[key]
            if ('spec' in data_vis['type']) and gen_dic['corr_tell'] and calc_EFsc2:
                if ('tell_DI_data_paths' not in data_inst[vis_bin]):stop('ERROR : weighing telluric profiles undefined; make sure you activate gen_dic["calc_proc_data"] and gen_dic["calc_corr_tell"] when running this module.')
                data_to_bin_gen[iexp_off]['tell'] = dataload_npz(data_inst[vis_bin]['tell_DI_data_paths'][iexp_glob])['tell']    
            else:data_to_bin_gen[iexp_off]['tell'] = None
            if (data_vis['type']=='spec2D') and calc_EFsc2:
                if ('sing_gcal_DI_data_paths' not in data_inst[vis_bin]):stop('ERROR : weighing calibration profiles undefined; make sure you activate gen_dic["calc_proc_data"] and gen_dic["calc_gcal"] when running this module.')  
                data_gcal = dataload_npz(data_inst[vis_bin]['sing_gcal_DI_data_paths'][iexp_off])
                data_to_bin_gen[iexp_off]['sing_gcal'] = data_gcal['gcal'] 
                if (vis_bin in data_inst['gcal_blaze_vis']):data_to_bin_gen[iexp_off]['sdet2'] = data_gcal['sdet2'] 
                else:data_to_bin_gen[iexp_off]['sdet2'] = None                
            else:
                data_to_bin_gen[iexp_off]['sing_gcal']=None   
                data_to_bin_gen[iexp_off]['sdet2'] = None  
            if data_dic['DI']['spec2D_to_spec1D'][inst]:data_to_bin_gen[iexp_off]['EFsc2'] = dataload_npz(data_inst[vis_bin]['EFsc2_DI_data_paths'][iexp_glob])['var']      
            else:data_to_bin_gen[iexp_off]['EFsc2'] = None
            
            #Master disk-integrated spectrum for weighing
            #    - profile has been shifted to the same frame as the differential profiles, but is still defined on the common table, not the table of current exposure
            #    - master covariance is not required for DI profile weights
            #    - see process_binned_prof() for details
            if (calc_EFsc2 or calc_var_ref2):        
                if ('mast_DI_data_paths' not in data_dic[inst][vis_bin]):stop('ERROR : weighing DI master undefined; make sure you activate gen_dic["calc_proc_data"] and gen_dic["calc_DImast"] when running this module.')
                data_ref = dataload_npz(data_dic[inst][vis_bin]['mast_DI_data_paths'][iexp_glob])
                data_to_bin_gen[iexp_off]['edge_bins_ref'] = data_ref['edge_bins']
                data_to_bin_gen[iexp_off]['flux_ref'] = data_ref['flux']
            else:data_to_bin_gen[iexp_off]['flux_ref'] = None
                
            #Exposure duration
            data_to_bin_gen[iexp_off]['dt'] = coord_dic[inst][vis_bin]['t_dur'][iexp_glob]

            #Weight profile
            #    - only calculated here on a common table if resampling is not required
            #    - resampling condition is that binned profiles:
            # + come from a single visit, and do not share a common table for the visit
            # + come from multiple visits, do not share a common table for all visits, and visit of the processed exposure is not the one used as reference for the common table of all visits (in which case resampling is not needed)     
            resamp_cond[iexp_off] = ((mode=='') and (not data_inst[vis_bin]['comm_sp_tab'])) or ((mode=='multivis') and (not data_inst['comm_sp_tab']) and (vis_bin!=data_inst['com_vis']))
            resamp_cond_all |= resamp_cond[iexp_off]
            if (not resamp_cond[iexp_off]): 
                data_to_bin_gen[iexp_off]['weight'] = weights_bin_prof(range(data_inst['nord']),scaled_data_paths_vis[vis_bin],inst,vis_bin,gen_dic['corr_Fbal'],gen_dic['corr_FbalOrd'],gen_dic['save_data_dir'],data_inst['nord'],iexp_glob,'DI',data_vis['type'],data_vis['dim_exp'],data_to_bin_gen[iexp_off]['tell'],data_to_bin_gen[iexp_off]['sing_gcal'],data_to_bin_gen[iexp_off]['cen_bins'],
                                                                       data_to_bin_gen[iexp_off]['dt'],data_to_bin_gen[iexp_off]['flux_ref'],None,(calc_EFsc2,calc_var_ref2,calc_flux_sc_all),sdet_exp2 = data_to_bin_gen[iexp_off]['sdet2'],EFsc2_all_in = data_to_bin_gen[iexp_off]['EFsc2'])[0]

        #Processing each exposure of current visit selected for extraction
        iexp_proc = data_dic['Diff'][inst][vis]['idx_to_extract']
        common_args = (data_vis['proc_DI_data_paths'],proc_gen_data_paths_new,idx_to_bin_all[0],n_in_bin_all[0],dx_ov_all[0],idx_bin2orig,idx_bin2vis,data_dic[inst]['nord'],data_vis['dim_exp'],data_vis['nspec'],data_to_bin_gen,gen_dic['resamp_mode'],\
                       scaled_data_paths_vis,inst,iexp_no_plrange_vis,exclu_rangestar_vis,data_vis['type'],gen_dic['type'],gen_dic['corr_Fbal'],gen_dic['corr_FbalOrd'],gen_dic['save_data_dir'],resamp_cond,resamp_cond_all,(calc_EFsc2,calc_var_ref2,calc_flux_sc_all))               
        if gen_dic['nthreads_diff_data']>1:MAIN_multithread(sub_extract_diff_profiles,gen_dic['nthreads_diff_data'],len(iexp_proc),[iexp_proc],common_args)                           
        else:sub_extract_diff_profiles(iexp_proc,*common_args)    

    #Checking that local data has been calculated for all exposures
    else:
        data_paths={iexp:proc_gen_data_paths_new+str(iexp) for iexp in range(data_vis['n_in_visit'])}
        check_data(data_paths)            

    #Path to weighing master and calibration profile
    #    - differential profiles are extracted in the same rest frame as the disk-integrated master, so that it can directly be used
    #    - at this stage a single master has been defined over the common spectral table, it will be resampled in the binning routine
    #    - calibration paths are updated even if they are not used as weights, to be used in flux/count scalings
    data_vis['proc_Diff_data_paths']=proc_gen_data_paths_new
    if ('mast_DI_data_paths' not in data_vis):stop('ERROR : weighing DI master undefined; make sure you activate gen_dic["calc_proc_data"] and gen_dic["calc_DImast"] when running this module.')
    data_vis['mast_Diff_data_paths'] = data_vis['mast_DI_data_paths']
    if ('spec' in data_vis['type']) and gen_dic['corr_tell']:
        if ('tell_DI_data_paths' not in data_vis):stop('ERROR : weighing telluric profiles undefined; make sure you activate gen_dic["calc_proc_data"] and gen_dic["calc_corr_tell"] when running this module.')
        data_vis['tell_Diff_data_paths'] = data_vis['tell_DI_data_paths']
    if gen_dic['flux_sc']:data_vis['scaled_Diff_data_paths'] = data_vis['scaled_DI_data_paths']
    if data_vis['type']=='spec2D':
        data_vis['mean_gcal_Diff_data_paths'] = data_vis['mean_gcal_DI_data_paths']
        if ('sing_gcal_DI_data_paths' not in data_vis):stop('ERROR : weighing calibration profiles undefined; make sure you activate gen_dic["calc_proc_data"] and gen_dic["calc_gcal"] when running this module.')  
        data_vis['sing_gcal_Diff_data_paths'] = data_vis['sing_gcal_DI_data_paths']
    for subtype_gen in gen_dic['earliertypes4var']['Diff']:
        if data_dic[subtype_gen]['spec2D_to_spec1D'][inst]:
            var_key = gen_dic['type2var'][gen_dic['typegen2type'][subtype_gen]]
            data_vis[var_key+'_Diff_data_paths']=data_vis[var_key+'_DI_data_paths']
            break    

    return None



def sub_extract_diff_profiles(iexp_proc,proc_DI_data_paths,proc_gen_data_paths_new,idx_to_bin_mast,n_in_bin_mast,dx_ov_mast,idx_bin2orig,idx_bin2vis,nord,dim_exp,nspec,data_to_bin_gen,resamp_mode,\
                             scaled_data_paths_vis,inst,iexp_no_plrange_vis,exclu_rangestar_vis,vis_type,gen_type,corr_Fbal,corr_FbalOrd,save_data_dir,resamp_cond,resamp_cond_all,calc_cond):            
    r"""**Differential profile extraction.** 

    Calculates differential profiles.
    
    Args:
        TBD
    
    Returns:
        None
    
    """     
    #Processing each exposure of current visit selected for extraction
    #    - extraction can be limited to in-transit exposures to gain computing time, e.g if one only needs to analyze the local stellar profiles 
    for isub,iexp in enumerate(iexp_proc):        
       
        #Upload latest processed DI data from which to extract local profile
        data_exp = dataload_npz(proc_DI_data_paths+str(iexp))
    
        #Calculating master disk-integrated profile
        #    - the master is calculated in a given exposure:
        # + if it is the first one
        # + if it is another one and binned profiles need resampling
        if (isub==0) or resamp_cond_all:                
            data_to_bin={}
            for iexp_off in idx_to_bin_mast:

                #Original index and visit of contributing exposure
                #    - index is relative to the global table
                iexp_glob = idx_bin2orig[iexp_off]
                vis_bin = idx_bin2vis[iexp_off]                    

                #Resampling on common spectral table if required
                #    - data is stored with the same indexes as in idx_to_bin_all
                #    - all exposures must be defined on the same spectral table before being binned
                #    - if multiple visits are used and do not share a common table, they do not need resampling if their table is the one used as reference to set the common table
                if resamp_cond[iexp_off]:
                    data_to_bin[iexp_off]={}
                    
                    #Resampling exposure profile
                    data_to_bin[iexp_off]['flux']=np.zeros(dim_exp,dtype=float)*np.nan
                    data_to_bin[iexp_off]['cov']=np.zeros(nord,dtype=object) 
                    flux_ref_exp=np.zeros(dim_exp,dtype=float)*np.nan if (data_to_bin_gen[iexp_off]['flux_ref'] is not None) else None
                    tell_exp=np.ones(dim_exp,dtype=float) if (data_to_bin_gen[iexp_off]['tell'] is not None) else None
                    sing_gcal_exp=np.ones(dim_exp,dtype=float) if (data_to_bin_gen[iexp_off]['sing_gcal'] is not None) else None
                    sdet2_exp=np.zeros(dim_exp,dtype=float) if (data_to_bin_gen[iexp_off]['sdet2'] is not None) else None
                    EFsc2_exp=np.zeros(dim_exp,dtype=float) if (data_to_bin_gen[iexp_off]['EFsc2'] is not None) else None
                    for iord in range(nord): 
                        data_to_bin[iexp_off]['flux'][iord],data_to_bin[iexp_off]['cov'][iord] = bind.resampling(data_exp['edge_bins'][iord], data_to_bin_gen[iexp_off]['edge_bins'][iord], data_to_bin_gen[iexp_off]['flux'][iord] , cov = data_to_bin_gen[iexp_off]['cov'][iord], kind=resamp_mode)                                                        
                        if flux_ref_exp is not None:flux_ref_exp[iord] = bind.resampling(data_exp['edge_bins'][iord], data_to_bin_gen[iexp_off]['edge_bins_ref'][iord], data_to_bin_gen[iexp_off]['flux_ref'][iord], kind=resamp_mode)                                                        
                        if tell_exp is not None:tell_exp[iord] = bind.resampling(data_exp['edge_bins'][iord], data_to_bin_gen[iexp_off]['edge_bins'][iord], data_to_bin_gen[iexp_off]['tell'][iord] , kind=resamp_mode) 
                        if sing_gcal_exp is not None:sing_gcal_exp[iord] = bind.resampling(data_exp['edge_bins'][iord], data_to_bin_gen[iexp_off]['edge_bins'][iord],data_to_bin_gen[iexp_off]['sing_gcal'][iord], kind=resamp_mode)  
                        if sdet2_exp is not None:sdet2_exp[iord] = bind.resampling(data_exp['edge_bins'][iord], data_to_bin_gen[iexp_off]['edge_bins'][iord],data_to_bin_gen[iexp_off]['sdet2'][iord], kind=resamp_mode)         
                        if EFsc2_exp is not None:EFsc2_exp[iord] = bind.resampling(data_exp['edge_bins'][iord], data_to_bin_gen[iexp_off]['edge_bins'][iord],data_to_bin_gen[iexp_off]['EFsc2'][iord], kind=resamp_mode)            
                    data_to_bin[iexp_off]['cond_def'] = ~np.isnan(data_to_bin[iexp_off]['flux'])   
                    if sdet2_exp is not None:sdet2_exp[np.isnan(sdet2_exp)]=0.
    
                    #Weight definition         
                    data_to_bin[iexp_off]['weight'] = weights_bin_prof(range(nord),scaled_data_paths_vis[vis_bin],inst,vis_bin,corr_Fbal,corr_FbalOrd,save_data_dir,nord,iexp_glob,'DI',vis_type,dim_exp,tell_exp,sing_gcal_exp,data_exp['cen_bins'],data_to_bin[iexp_off]['dt'],flux_ref_exp,None,calc_cond,sdet_exp2 = sdet2_exp,EFsc2_all_in = EFsc2_exp)[0]

                #Weighing components and current exposure are defined on the same table common to the visit 
                else:data_to_bin[iexp_off] = deepcopy(data_to_bin_gen[iexp_off])  

                #Exclude planet-contaminated bins  
                #    - condition that 'DI_Mast' is in 'no_plrange' is included in the definition of 'iexp_no_plrange_vis'
                if (iexp_glob in iexp_no_plrange_vis[vis_bin]):
                    for iord in range(nord):                   
                        data_to_bin[iexp_off]['cond_def'][iord] &=  excl_plrange(data_to_bin[iexp_off]['cond_def'][iord],exclu_rangestar_vis[vis_bin],iexp_off,data_exp['edge_bins'][iord],vis_type)[0]
        
            #Calculate master on current exposure table
            data_mast = calc_bin_prof(idx_to_bin_mast,nord,dim_exp,nspec,data_to_bin,inst,n_in_bin_mast,data_exp['cen_bins'],data_exp['edge_bins'],dx_ov_in = dx_ov_mast)
       
        #Extracting differential stellar profiles  
        #    - the master is defined for each individual exposures if they are defined on different spectral table
        #      otherwise defined on a single common spectral table, in which case we repeat the master to have the same structure as individual exposures          
        data_loc = {'cen_bins':data_exp['cen_bins'],
                    'edge_bins':data_exp['edge_bins'],
                    'flux' : np.zeros(dim_exp, dtype=float),
                    'cov' : np.zeros(nord, dtype=object)}
        for iord in range(nord):
            data_loc['flux'][iord],data_loc['cov'][iord]=bind.add(data_mast['flux'][iord], data_mast['cov'][iord], -data_exp['flux'][iord], data_exp['cov'][iord])                 
        data_loc['cond_def'] = ~np.isnan(data_loc['flux'])       

        #Saving data
        #    - saved for each exposure, as the files are too large otherwise                
        datasave_npz(proc_gen_data_paths_new+str(iexp),data_loc)    
    
    return None








################################################################################################## 
#%% Intrinsic profiles routines
################################################################################################## 

def extract_intr_profiles(data_dic,gen_dic,inst,vis,star_params,coord_dic,theo_dic,plot_dic):
    r"""**Main intrinsic profile routine.** 

    Extracts intrinsic profiles in the stellar rest frame.
    
    The in-transit differential spectra, at wavelengths where planetary contamination was masked, correspond to

    .. math::        
       F_\mathrm{diff}(\mathrm{\lambda \, in \, B},t,v) &= ( f_p(\lambda,v) S(\lambda,t) -  F_p(\lambda,t) ) C_\mathrm{ref}(B,v)   \\
                                                          &= f_p(\lambda,v) S_\mathrm{thick}(B,t) C_\mathrm{ref}(B,v)  
                       
    With 
    
     - :math:`f_p` the surface density flux spectrum, assumed spatially constant over the region occulted by the planet, 
       known to a scaling factor :math:`C_\mathrm{ref}(B,v)`, assumed to be constant over the band, and accounting for the absolute flux level and deviations from the stellar SED.  
     - :math:`S_\mathrm{thick}(B,p)` the effective planet surface occulting the star in the band, assumed spectrally constant over the band, varying spatially during ingress/egress, and set by our choice of light curve. 
    
    The above expression consider only wavelengths :math:`\lambda` where there are no narrow absorption lines from the planetary atmosphere (absorbed wavelengths have been masked).
    
    
    We scale back differential spectra to get back to the intrinsic stellar profiles (ie, without broadband planetary absorption and limb/grav-darkening), assuming that 

    .. math::      
       f_p(\mathrm{\lambda \, in \, B},v) = I(\mathrm{\lambda \, in \, B},t) \mathrm{LD}(B,t) 
       
    i.e. that the limb-darkening has low-frequency variations and does not affect the shape of the local intrinsic spectra.
    The theoretical light curve used to rescale the data writes as (see `rescale_profiles()`) :

    .. math::          
       1 - \mathrm{LC}_\mathrm{theo}(B,t) = \mathrm{LC}_p(B,t) \frac{S_p(B,t)}{S_{\star}^\mathrm{LD}(B) }  
    
    Where the fluxes are constant and spatially uniform over the band and stellar disk, and the planet is described by a constant, mean radius over the band.
    If we normalize the local spectra by this factor, we obtain the intrinsic spectra as

    .. math::  
       F_\mathrm{intr}(\lambda,t,v) &= \frac{F_\mathrm{diff}(\mathrm{\lambda \, in \, B},t,v)}{1 - \mathrm{LC}_\mathrm{theo}(B,t)} \\
                                    &= \frac{f_p(\lambda,v) S_\mathrm{thick}(B,t) C_\mathrm{ref}(B,v)}{1 - \mathrm{LC}_\mathrm{theo}(B,t)}  \\
                                    &= \frac{I(\lambda,t) \mathrm{LD}(B,t) S_\mathrm{thick}(B,t) C_\mathrm{ref}(B,v) S_{\star}^\mathrm{LD}(B)}{LD(B,t) S_p(B,t)}   \\
                                    &= \frac{I(\lambda,t) S_\mathrm{thick}(B,t) C_\mathrm{ref}(B,v) S_{\star}^\mathrm{LD}(B)}{Sp(B,t) }
    
    During the full transit, the ratio :math:`S_r(B) = S_\mathrm{thick}(B,t)/S_p(B,t)` is constant over time. 
    During ingress/egress, we assume that the ratio remains the same, so that
  
    .. math::      
       F_\mathrm{intr}(\lambda,t,v) = I(\lambda,t) F^\mathrm{norm}_\mathrm{ref}(B)
    
    With 

    .. math::     
       F^\mathrm{norm}_\mathrm{ref} = S_r(B) C_\mathrm{ref}(B,v) S_{\star}^\mathrm{LD}(B) 
       
    Only dependent on the band. The normalized local spectra then allow comparing the shape of the intrinsic spectra along the transit chord in a given band.
            
    
    The continuum of the differential profiles is
    
    .. math::  
       F_\mathrm{diff}(B_\mathrm{cont},t,v) &= \sum_{\lambda \, in \, B_\mathrm{cont}}(I_p(\lambda,t) \mathrm{LD}_p(B,t) S_\mathrm{thick}(B,t) C_\mathrm{ref}(B,v) d\lambda(\lambda,v))   \\
                                           &= \sum_{\lambda \, in \, B_\mathrm{cont}}(I_p(\lambda,t) d\lambda(\lambda,v)) \mathrm{LD}_p(B,t) S_\mathrm{thick}(B,t) C_\mathrm{ref}(B,v)   \\ 
                                           &= I_p(B_\mathrm{cont},v) \mathrm{LD}_p(B,t) S_\mathrm{thick}(B,t) C_\mathrm{ref}(B,v)   
                      
    If we define the continuum as a spectral range where the flux only varies due to broadband limb-darkening, we have :math:`I_p(B_\mathrm{cont},v) = I(B_\mathrm{cont},v)`
    
    .. math::  
       F_{\star}(B_\mathrm{cont},v) &= \sum_{ x }(\sum_{\lambda \, in \, B_\mathrm{cont}}(I_x(\lambda,v) d\lambda(\lambda,v)) \mathrm{LD}_x(B,v) S_x )  \\
                                    &= I(B_\mathrm{cont},v) S_{\star}^\mathrm{LD}(B)
                     
    Thus the intrinsic continuum writes as
    
    .. math::  
       F_\mathrm{intr}(B_\mathrm{cont},t,v) &= I(B_\mathrm{cont},v) S_r(B) C_\mathrm{ref}(B,v) S_{\star}^\mathrm{LD}(B)  \\
                                            &= F_{\star}(B_\mathrm{cont},v) S_r(B) C_\mathrm{ref}(B,v) 
                       
    And the differential continuum writes as 

    .. math::             
       F_\mathrm{diff}(B_\mathrm{cont},t,v) &= \frac{F_{\star}(B_\mathrm{cont},v) \mathrm{LD}_p(B,t) S_\mathrm{thick}(B,t) C_\mathrm{ref}(B,v)}{S_{\star}^\mathrm{LD}(B) }   \\                        
                                           &= F_{\star}(B_\mathrm{cont},v) ( 1 - \mathrm{LC}_\mathrm{theo}(B,t) ) S_r(B,t) C_\mathrm{ref}(B,v)
                      
    If :math:`S_r \sim 1` the intrinsic profiles have the same flux as the disk-integrated spectra before the relative broadband flux scaling.
    
    .. math::  
       \sum_{\lambda \, in \, B_\mathrm{cont}}( \frac{F_\mathrm{sc}(\lambda,t,v) d\lambda(\lambda,t,v)}{\mathrm{LC}_\mathrm{theo}(B,t)} ) &= \sum_{\lambda \, in \, B_\mathrm{cont}}( \frac{ F_\mathrm{sc}(\lambda,t,v) d\lambda(\lambda,t,v) }{ \mathrm{LC}_\mathrm{theo}(B,t) } ) \\
                                                                                                                                          &= \sum_{\lambda \, in \, B_\mathrm{cont}}( F_{\star}(\lambda,v) C_\mathrm{ref}(B,v) d\lambda(\lambda,t,v)) \\
                                                                                                                                          &= F_{\star}(B_\mathrm{cont},v) C_\mathrm{ref}(B,v)
                                                       
    Intrinsic profiles thus have the same continuum as the scaled out-of-transit and master disk-integrated profiles, within a range controlled by broadband flux variations (ie, outside of planetary and stellar lines) but this continuum may not be exactly unity.


    Bins affected by the planet absorption must be included in the scaling band (rescale_profiles())), but after this operation is done we set all bins affected by the planetary atmosphere to nan so that the final profiles are purely stellar.
  
    
    Here we do not shift the intrinsic stellar profiles to a common rest wavelength, as they can be later used to derive the velocity of the stellar surface. 


    The approach is the same with CCFs, except that everything is performed with a single band, and the scaling can be carried out in the CCF continuum (thus avoiding potential variations in the line shape)

     - when CCFs are given as input or calculated before the flux scaling, their continuum is normalized to unity outside of the transit
     - when CCFs are calculated from differential spectra the continuum is unknown a priori, as it depends on the reference spectrum to which each DI spectrum corresponds to, times the spectral flux scaling during transit.
       The differential spectra write as :math:`F_\mathrm{diff}(\lambda,t,v) = F_\mathrm{intr}(\lambda,t,v) (1 - \mathrm{LC}_\mathrm{theo}(B,t))`
       when converting them into :math:`CCF_\mathrm{diff}`, we also compute the equivalent :math:`CCF_\mathrm{sc}` of :math:`(1 - \mathrm{LC}_\mathrm{theo}(B,t))`
       :math:`CCF_\mathrm{diff}` are then converted into :math:`CCF_\mathrm{intr}` in this routine by dividing their continuum using :math:`CCF_\mathrm{sc}` 
       this is an approximation, since the spectral scaling cannot be isolated from :math:`F_\mathrm{intr}(\lambda,t,v)` when computing :math:`CCF_\mathrm{diff}`, but it is the same approximation we do when scaling DI CCFs with a white light transit
       we take this approach rather than first calculate :math:`F_\mathrm{intr}` and then its CCF because we need :math:`CCF_\mathrm{diff}` to later derive the atmospheric CCFs
       ideally though, one should keep processing spectra rather than convert differential profiles into CCFs
    
    Args:
        TBD
    
    Returns:
        None
    
    """      
    print('   > Extracting intrinsic stellar profiles')  
    data_vis=data_dic[inst][vis]
    gen_vis = gen_dic[inst][vis]
    
    #Current rest frame
    if data_dic['Diff'][inst][vis]['rest_frame']!='star':print('WARNING: differential profiles must be aligned')
    data_dic['Intr'][inst][vis]['rest_frame'] = 'star'

    #Path to initialized intrinsic data
    proc_gen_data_paths_new = gen_dic['save_data_dir']+'Intr_data/'+inst+'_'+vis

    #Updating paths
    #    - if no shifts are applied the associated profiles remain the same as those of the differential profiles, and their paths are not updated
    #    - calibration paths are updated even if they are not used as weights, to be used in flux/count scalings
    #    - paths are defined for each exposure for associated tables, to avoid copying tables from differential profiles and simply point from in-transit to global differential profiles
    data_vis['proc_Intr_data_paths']=proc_gen_data_paths_new+'_' 
    if gen_dic['flux_sc']:data_vis['scaled_Intr_data_paths'] = data_vis['scaled_Diff_data_paths']
    if 'mast_Diff_data_paths' in data_vis:data_vis['mast_Intr_data_paths'] = {}
    if ('spec' in data_vis['type']) and gen_dic['corr_tell']:
        if ('tell_Diff_data_paths' not in data_vis):stop('ERROR : weighing telluric profiles undefined; make sure you activate gen_dic["calc_proc_data"] and gen_dic["calc_corr_tell"] when running this module.')
        data_vis['tell_Intr_data_paths'] = {}
    if data_vis['type']=='spec2D':
        data_vis['mean_gcal_Intr_data_paths'] = {} 
        if ('sing_gcal_Diff_data_paths' not in data_vis):stop('ERROR : weighing calibration profiles undefined; make sure you activate gen_dic["calc_proc_data"] and gen_dic["calc_gcal"] when running this module.')  
        data_vis['sing_gcal_Intr_data_paths'] = {} 
    var_key = None
    for subtype_gen in gen_dic['earliertypes4var']['Intr']:
        if data_dic[subtype_gen]['spec2D_to_spec1D'][inst]:
            var_key = gen_dic['type2var'][gen_dic['typegen2type'][subtype_gen]]
            data_vis[var_key+'_Intr_data_paths']={}
            break            
    for i_in,iexp in enumerate(gen_vis['idx_in']):
        if 'mast_Intr_data_paths' in data_vis:data_vis['mast_Intr_data_paths'][i_in] = data_vis['mast_Diff_data_paths'][iexp]
        if ('spec' in data_vis['type']) and gen_dic['corr_tell']:data_vis['tell_Intr_data_paths'][i_in] = data_vis['tell_Diff_data_paths'][iexp]
        if var_key is not None:data_vis[var_key+'_Intr_data_paths'][i_in] = data_vis[var_key+'_Diff_data_paths'][iexp]
        if data_vis['type']=='spec2D':
            data_vis['mean_gcal_Intr_data_paths'][i_in] = data_vis['mean_gcal_Diff_data_paths'][iexp]
            data_vis['sing_gcal_Intr_data_paths'][i_in] = data_vis['sing_gcal_Diff_data_paths'][iexp]

    #Correcting for relative chromatic shift    
    if ('spec' in data_vis['type']) and ('chrom' in data_dic['DI']['system_prop']):intr_rv_corr = True
    else:intr_rv_corr=False

    #Processing intrinsic data
    if (gen_dic['calc_intr_data']):
        print('         Calculating data')
        plAtm_vis = data_dic['Atm'][inst][vis]

        #Initialize in-transit indexes of defined intrinsic profiles
        data_dic['Intr'][inst][vis]['idx_def'] = np.arange(data_vis['n_in_tr'],dtype=int) 

        #Correcting for relative chromatic shift
        #    - for spectral data and chromatic occultation: the planet occults region of different size across the spectrum, so that
        # the lines in a given spectral band are shifted by an average RV over the corresponding chromatic surface
        #    - here we correct the intrinsic spectra for the chromatic RV deviation around the nominal planet-occulted RV
        #    - this correction is performed here so that intrinsic profiles can be converted into CCF prior to the alignment module, in which
        # a constant RV shift can then be applied
        if intr_rv_corr:
            ref_pl,dic_rv,idx_aligned = init_surf_shift(gen_dic,inst,vis,data_dic,'theo')
            
            #Resample aligned profiles on the common visit table if relevant
            #    - alignment in the star rest frame must have been applied for intrinsic extraction
            if (data_vis['comm_sp_tab']):
                data_com = dataload_npz(data_vis['proc_com_star_data_paths'])
                cen_bins_resamp, edge_bins_resamp = data_com['cen_bins'],data_com['edge_bins']
            else:cen_bins_resamp, edge_bins_resamp = None,None  

        #Definition of intrinsic stellar profiles
        for i_in,iexp in enumerate(gen_vis['idx_in']):

            #Upload local stellar profile
            data_exp = dataload_npz(data_vis['proc_Diff_data_paths']+str(iexp))
            
            #Upload flux scaling
            data_scaling_exp = dataload_npz(data_vis['scaled_Diff_data_paths']+str(iexp))

            #Rescale local stellar profiles to a common flux level
            #    - correcting for LD variation and planetary occultation
            #    - the scaling spectrum is defined at all wavelengths, thus defined bins are unchanged 
            #      the scaling equals 0 for undefined pixels, thus we set the rescaling to 1 to avoid warnings
            #    - the scaling may be entirely null, if the exposure is not actually overlapping with the stellar disk
            #      in that case the profile is set to nan
            if (data_scaling_exp['null_loc_flux_scaling']):
                
                #Set exposure as undefined
                print('         Profile at idx '+str(iexp),'(global) = '+str(i_in)+' (in-tr) is undefined')
                data_dic['Intr'][inst][vis]['idx_def'] = np.delete( data_dic['Intr'][inst][vis]['idx_def'], data_dic['Intr'][inst][vis]['idx_def']==i_in)
                for iord in range(data_dic[inst]['nord']):
                    data_exp['flux'][iord,:] = np.nan
                    data_exp['cond_def'][iord,:] = False
 
            else:
                for iord in range(data_dic[inst]['nord']):
                    cond_exp_ord = data_exp['cond_def'][iord]
                    resc_ord = np.ones(data_vis['nspec'],dtype=float)
                    resc_ord[cond_exp_ord] = 1./data_scaling_exp['loc_flux_scaling'](data_exp['cen_bins'][iord,cond_exp_ord])
                    data_exp['flux'][iord],data_exp['cov'][iord] = bind.mul_array(data_exp['flux'][iord],data_exp['cov'][iord],resc_ord)

                #Correct for relative chromatic shift
                if intr_rv_corr:
    
                    #Radial velocity shifts set to the opposite of the planet-occulted surface rv associated with current exposure
                    rv_surf_star,rv_surf_star_edge = def_surf_shift('theo_rel',dic_rv,i_in,data_exp,ref_pl,data_vis['type'],data_dic['DI']['system_prop'],data_dic[inst][vis]['dim_exp'],data_dic[inst]['nord'],data_dic[inst][vis]['nspec']) 

                    #Doppler shift
                    spec_dopshift_cen = 1./gen_specdopshift(rv_surf_star)
                    spec_dopshift_edge = 1./gen_specdopshift(rv_surf_star_edge)
    
                    #Spectral RV correction of current exposure and complementary tables
                    if ('spec' in data_vis['type']):
                        if gen_dic['corr_tell']:data_exp['tell'] = dataload_npz(data_vis['tell_Diff_data_paths'][iexp])['tell'] 
                        if data_vis['type']=='spec2D':
                            data_exp['mean_gcal'] = dataload_npz(data_vis['mean_gcal_Diff_data_paths'][iexp] )['mean_gcal'] 
                            data_gcal = dataload_npz(data_vis['sing_gcal_Diff_data_paths'][iexp])
                            data_exp['sing_gcal'] = data_gcal['gcal'] 
                            if (vis in data_dic[inst]['gcal_blaze_vis']):data_exp['sdet2'] = data_gcal['sdet2']    
                    if var_key is not None:data_exp[var_key] = dataload_npz(data_vis[var_key+'_Diff_data_paths'][iexp])['var']               
                    data_exp=align_data(data_exp,data_vis['type'],data_dic[inst]['nord'],data_dic[inst][vis]['dim_exp'],gen_dic['resamp_mode'],cen_bins_resamp,edge_bins_resamp,rv_surf_star,spec_dopshift_cen,rv_shift_edge = rv_surf_star_edge,spec_dopshift_edge=spec_dopshift_edge)
    
                    #Saving aligned exposure and complementary tables
                    if ('spec' in data_vis['type']):
                        if gen_dic['corr_tell']:
                            data_vis['tell_Intr_data_paths'][i_in] = proc_gen_data_paths_new+'_tell'+str(i_in)
                            datasave_npz(data_vis['tell_Intr_data_paths'][i_in],{'tell':data_exp['tell']})
                            data_exp.pop('tell')
                        if data_vis['type']=='spec2D':
                            data_vis['mean_gcal_Intr_data_paths'][i_in] = proc_gen_data_paths_new+'_mean_gcal'+str(i_in)
                            datasave_npz(data_vis['mean_gcal_Intr_data_paths'][i_in],{'mean_gcal':data_exp['mean_gcal']})
                            data_exp.pop('mean_gcal')
                            data_gcal = {'gcal':deepcopy(data_exp['sing_gcal'])}
                            data_exp.pop('sing_gcal')
                            if 'sdet2' in data_exp:
                                data_gcal['sdet2']=deepcopy(data_exp['sdet2'])
                                data_exp.pop('sdet2')  
                            datasave_npz(data_vis['sing_gcal_Intr_data_paths'][i_in],data_gcal) 
                        if var_key is not None:
                            data_vis[var_key+'_Intr_data_paths'][iexp] = proc_gen_data_paths_new+'_'+var_key+str(i_in)    
                            datasave_npz(data_vis[var_key+'_Intr_data_paths'][iexp], {'var':data_exp[var_key]})          
                            data_exp.pop(var_key)
    
                    #Spectral RV correction of weighing master
                    if ('mast_Intr_data_paths' in data_vis):
                        data_ref = dataload_npz(data_vis['mast_Diff_data_paths'][iexp]) 
                        data_ref_align=align_data(data_ref,data_vis['type'],data_dic[inst]['nord'],data_dic[inst][vis]['dim_exp'],gen_dic['resamp_mode'],cen_bins_resamp,edge_bins_resamp,rv_surf_star,spec_dopshift_cen,rv_shift_edge = rv_surf_star_edge,spec_dopshift_edge=spec_dopshift_edge)                  
                        data_vis['mast_Intr_data_paths'][i_in] = proc_gen_data_paths_new+'_ref'+str(i_in)
                        datasave_npz(data_vis['mast_Intr_data_paths'][i_in],data_ref_align)

                #Set to nan planetary ranges
                #    - only if intrinsic spectra are not to be converted into CCFs, in which case the exclusion is applied after their conversion
                if ('Intr' in data_dic['Atm']['no_plrange']) and (not gen_dic['Intr_CCF']) and (iexp in plAtm_vis['iexp_no_plrange']):
                    cond_out_pl = np.ones(data_vis['dim_exp'],dtype=bool)
                    for iord in range(data_dic[inst]['nord']):                        
                        cond_out_pl[iord] &=excl_plrange(data_exp['cond_def'][iord],plAtm_vis['exclu_range_star'],iexp,data_exp['edge_bins'][iord],data_vis['type'])[0]
                    cond_in_pl = ~cond_out_pl
                    data_exp['flux'][cond_in_pl]=np.nan
                    data_exp['cond_def'][cond_in_pl]=False    
    
                    #Saving exclusion flag
                    data_exp['plrange_exc'] = True
                else:data_exp['plrange_exc'] = False

            #Saving data using in-transit indexes              
            np.savez_compressed(proc_gen_data_paths_new+'_'+str(i_in),data=data_exp,allow_pickle=True)
  
        #Save complementary data
        np.savez_compressed(proc_gen_data_paths_new+'_add',data={'idx_def':data_dic['Intr'][inst][vis]['idx_def']},allow_pickle=True)            
  
    else:
        check_data({0:proc_gen_data_paths_new+'_add'})  
        data_dic['Intr'][inst][vis]['idx_def'] = dataload_npz(proc_gen_data_paths_new+'_add')['idx_def']
  
    #Continuum level and correction
    #    - at this stage, profiles in CCF mode always come from input CCF data
    #      continuum is only calculated for spectral data if not converted later on (in which case continuum is calculated later on)
    #    - if applied to intrinsic profiles derived from CCF data, planetary ranges have been excluded if requested
    #      if undefined for spectral data, the continuum has been set to the full order range
    if (data_vis['type']=='CCF') or (('spec' in data_vis['type']) and ((not gen_dic['Intr_CCF']) and (not gen_dic['spec_1D_Intr']))):
        print('         Intrinsic continuum calculations')
        if (data_vis['type']=='CCF'):loc_type = 'CCFIntr_from_CCFDI'
        else:loc_type = 'SpecIntr'
        if data_dic['Intr']['calc_cont']:           
            data_dic['Intr'][inst][vis]['mean_cont'],cont_norm_flag=calc_Intr_mean_cont(data_dic['Intr'][inst][vis]['idx_def'],data_dic[inst]['nord'],data_dic[inst][vis]['nspec'],data_vis['proc_Intr_data_paths'],data_vis['type'],data_dic['Intr']['cont_range'],inst,data_dic['Intr']['cont_norm'],gen_dic[inst][vis]['flag_err'],loc_type)
            np.savez_compressed(data_vis['proc_Intr_data_paths']+'_add',data={'mean_cont':data_dic['Intr'][inst][vis]['mean_cont'],'cont_norm_flag':cont_norm_flag,'type':loc_type},allow_pickle=True)
        else:
            check_flag = check_data({'0':data_vis['proc_Intr_data_paths']+'_add'},silent=True)
            if not check_flag:stop('WARNING: calculate continuum for intrinsic spectra (origin: '+data_vis['type']+')')
            data_add = dataload_npz(data_vis['proc_Intr_data_paths']+'_add')
            if data_add['type']!=loc_type:stop('WARNING: continuum type incompatible with data, run extraction again.')
            data_dic['Intr'][inst][vis]['mean_cont'] = data_add['mean_cont']
            if data_add['cont_norm_flag']:print('         Correcting intrinsic continuum')

    return None



def calc_Intr_mean_cont(idx_in_tr,nord,nspec,proc_Intr_data_paths,data_type,cont_range,inst,cont_norm,flag_err,loc_type):
    r"""**Intrinsic continuum calculation.**  
    
    Calculates common continuum level for intrinsic profiles.
    
    The continuum level is defined as a weighted mean because local CCFs at the limbs can be very poorly defined due to the partial occultation and limb-darkening.
    
    If intrinsic CCFs were calculated from input CCF profiles, their continuum flux should match that of the out-of-transit CCFs (beware however that the scaling of disk-integrated profiles is done over their full range, not just the continuum).
    If differential or intrinsic profiles were converted from spectra, then their continuum is not know a priori.
    As a general approach we thus calculate the continuum value here    
    
    Args:
        TBD
    
    Returns:
        None
    
    """     
    #Continuum determination
    n_in_tr =len(idx_in_tr)
    cond_def_cont_all  = np.zeros([n_in_tr,nord,nspec],dtype=bool)
    for i_in in idx_in_tr:
        data_exp = dataload_npz(proc_Intr_data_paths+str(i_in))  
        for iord in range(nord):
            
            #Setting continuum pixels as true in requested continuum range
            if (inst in cont_range) and (iord in cont_range[inst]):
                for bd_int in cont_range[inst][iord]:cond_def_cont_all[i_in,iord] |= (data_exp['edge_bins'][iord,0:-1]>=bd_int[0]) & (data_exp['edge_bins'][iord,1:]<=bd_int[1])   
                cond_def_cont_all[i_in,iord] &= data_exp['cond_def'][iord]         
                
            #Setting continuum pixels directly to defined pixels
            else:cond_def_cont_all[i_in,iord][data_exp['cond_def'][iord]] = True      
            
    #Common continuum in each order
    cond_cont_com  = np.all(cond_def_cont_all,axis=0) 
    
    #Continuum level
    cont_intr = np.ones([n_in_tr,nord])
    wcont_intr = np.ones([n_in_tr,nord])
    for i_in in idx_in_tr:
        data_exp = dataload_npz(proc_Intr_data_paths+str(i_in))
        for iord in range(nord):
            if np.sum(data_exp['cond_def'][iord])>0:
                cond_cont_com_ord = cond_cont_com[iord]
                if np.sum(cond_cont_com_ord)==0.:stop('No pixels in common continuum') 
                dcen_bins = data_exp['edge_bins'][iord,1:][cond_cont_com_ord] - data_exp['edge_bins'][iord,0:-1][cond_cont_com_ord]
                
                #Continuum flux of the intrinsic CCF and corresponding error
                #    - calculated over the defined bins common to all differential and intrinsic profiles
                #    - we use the covariance diagonal to define a representative weight, unless no errors are defined              
                cont_intr[i_in,iord] = np.sum(data_exp['flux'][iord,cond_cont_com_ord]*dcen_bins)/np.sum(dcen_bins)
                if flag_err:wcont_intr[i_in,iord] = np.sum(dcen_bins**2.)/np.sum( data_exp['cov'][iord][0,cond_cont_com_ord]*dcen_bins**2. )
                
            else:
                cont_intr[i_in,iord] = 0.
                wcont_intr[i_in,iord] = 0.

    #Continuum flux over all in-transit exposures
    mean_cont=np.sum(cont_intr*wcont_intr,axis=0)/np.sum(wcont_intr,axis=0)      

    #Continuum correction
    #    - intrinsic profiles can show deviations from the common continuum level they should have 
    #      here we set manually their continuum to the mean continuum of all intrinsic profiles before the correction
    if cont_norm:    
        print('         Correcting intrinsic continuum')
        if loc_type=='SpecIntr':stop('WARNING : it is unwise to apply continuum correction to 2D intrinsic spectra.')
        for i_in in idx_in_tr:
            data_exp = dataload_npz(proc_Intr_data_paths+str(i_in))
            for iord in range(nord):
                cond_cont_com_ord = cond_cont_com[iord]

                #Continuum of current exposure
                dcen_bins = data_exp['edge_bins'][iord,1:][cond_cont_com_ord] - data_exp['edge_bins'][iord,0:-1][cond_cont_com_ord]
                cont_intr_exp = np.sum(data_exp['flux'][iord][cond_cont_com_ord]*dcen_bins)/np.sum(dcen_bins)
              
                #Correction factor
                corr_exp = mean_cont[iord]/cont_intr_exp
  
                #Overwrite exposure data
                data_exp['flux'][iord],data_exp['cov'][iord] = bind.mul_array(data_exp['flux'][iord],data_exp['cov'][iord],np.repeat(corr_exp,nspec))
            datasave_npz(proc_Intr_data_paths+str(i_in),data_exp)      
        cont_norm_flag = True
    else:cont_norm_flag = False
 
    return mean_cont,cont_norm_flag


















################################################################################################## 
#%% Planetary profiles routines
################################################################################################## 

def extract_pl_profiles(data_dic,inst,vis,gen_dic):
    r"""**Main atmospheric profile routine.**  
    
    Extracts planetary atmospheric profiles in the stellar rest frame.

    Args:
        TBD
    
    Returns:
        None
    
    """  
    print('   > Extracting atmospheric stellar profiles')
    data_vis = data_dic[inst][vis]
    plAtm_vis = data_dic['Atm'][inst][vis]

    #Current rest frame
    if data_dic['Diff'][inst][vis]['rest_frame']!='star':print('WARNING: differential profiles must be aligned')
    data_dic['Atm'][inst][vis]['rest_frame'] = 'star'

    #Indexes of in-transit exposures with defined estimates of local stellar profiles
    idx_est_loc = dataload_npz(gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['opt_loc_prof_est']['corr_mode']+'/'+inst+'_'+vis)['idx_est_loc']

    #Indexes of exposures with retrieved signal
    if (data_dic['Atm']['pl_atm_sign']=='Absorption'):
        #In-transit indexes
        plAtm_vis['idx_def'] = idx_est_loc
        #Corresponding global indexes
        iexp_glob = np.array(gen_dic[inst][vis]['idx_in'])[idx_est_loc]

    elif (data_dic['Atm']['pl_atm_sign']=='Emission'):
        #Global indexes
        plAtm_vis['idx_def'] = list(gen_dic[inst][vis]['idx_out']) + list(np.array(gen_dic[inst][vis]['idx_in'])[idx_est_loc])
        iexp_glob = plAtm_vis['idx_def']
         
    #Initializing path to atmospheric data
    data_vis['proc_Atm_data_paths']=gen_dic['save_data_dir']+'Atm_data/'+data_dic['Atm']['pl_atm_sign']+'/'+inst+'_'+vis+'_'
                
    #Initialize path of weighing profiles for atmospheric exposures
    #    - the weighing profiles include the master disk-integrated profile, and the best estimates of the local stellar profiles (if measured)
    #    - best estimates are only defined for in-transit profiles, and paths must be defined relative to the indexes used to call atmpospheric profiles
    #    - weighing master is defined on the common table for the visit
    if (data_dic['Atm']['pl_atm_sign']=='Absorption') or ((data_dic['Atm']['pl_atm_sign']=='Emission') and data_dic['Intr']['cov_loc_star']):
        data_vis['LocEst_Atm_data_paths'] = {}
        if (data_dic['Atm']['pl_atm_sign']=='Absorption'):iexp_paths = idx_est_loc     #in-transit indexes
        elif (data_dic['Atm']['pl_atm_sign']=='Emission'):iexp_paths = np.array(gen_dic[inst][vis]['idx_in'])[idx_est_loc]    #global indexes, limited to in-transit values
        data_vis['LocEst_Atm_data_paths'] = {iexp:gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['opt_loc_prof_est']['corr_mode']+'/'+inst+'_'+vis+'_'+str(i_in) for iexp,i_in in zip(iexp_paths,idx_est_loc)}
    
    #Calculating
    if (gen_dic['calc_pl_atm']):
        print('         Calculating data')     

        #Data for absorption signal
        if (data_dic['Atm']['pl_atm_sign']=='Absorption'):

            #Properties of planet-occulted regions
            dic_plocc_prop = np.load(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item()     
                     
        #Process all exposures
        for iexp,i_in in zip(range(data_vis['n_in_visit']),gen_dic[inst][vis]['idx_exp2in']):

            #Upload local profile
            #    - the local stellar profiles defined in the star rest frame write as 
            # Fdiff(w,t,vis) = ( fp(w,vis)*(Sthick(band,t) + Sthin(w-wp,t) ) -  Fatm(w-wp,t) )*Cref(band)/Fr(vis) 
            #      we want to retrieve the atmospheric emission signal Fatm, and the atmospheric absorption surface Sthin
            data_loc = np.load(data_vis['proc_Diff_data_paths']+str(iexp)+'.npz',allow_pickle=True)['data'].item()             
                
            #Upload estimate of local stellar profile for in-transit exposures
            #    - we distinguish between theoretical estimates and ones derived from data in the calculation of the covariance below 
            if (i_in>-1) and (i_in in idx_est_loc):data_loc_star = np.load(gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['opt_loc_prof_est']['corr_mode']+'/'+inst+'_'+vis+'_'+str(i_in)+'.npz',allow_pickle=True)['data'].item()   
                        
            #Extraction of emission signal
            #    - out-of-transit we take the opposite of the local profiles to retrieve the emission signal
            # F_em(w,t,vis) = - Fdiff(w,t,vis) 
            #               = Fatm(w-wp,t)*Cref(band)/Fr(vis)  
            #    - in-transit, the estimates of local stellar profiles correspond to 
            # Fstar_loc(w,t) = fp(w,vis)*Sthick(band,t)*Cref(band)/Fr(vis) 
            #      and we define the emission signal as    
            # F_em(w,t,vis) = Fstar_loc(w,t) - Fdiff(w,t,vis) 
            #               = Fatm(w-wp,t)*Cref(band)/Fr(vis) - fp(w,vis)*Sthin(w-wp,t )*Cref(band)/Fr(vis)
            #      thus the emission signal will remain contaminated if there is an absorption signal
            #    - unless the exact stellar SED for each visit was used to correct for the color balance, and to set spectra to the correct overall flux level in each visit, the
            # emission signals will be known to a scaling factor Cref(band)/Fr(vis)                                                                         
            if (data_dic['Atm']['pl_atm_sign']=='Emission') and (iexp in plAtm_vis['idx_def']):
                data_em = {'cen_bins':data_loc['cen_bins'],'edge_bins':data_loc['edge_bins'],'tell':data_loc['tell']}              
                
                #Out-of-transit signal
                if (i_in == -1):
                    data_em['flux'] = -data_loc['flux']
                    for key in ['cov','cond_def']:data_em[key] = data_loc[key]

                #In-transit signal
                #    - in-transit exposures with no estimates of local stellar profiles cannot be retrieved
                elif (i_in in idx_est_loc):
                    if not data_dic['Intr']['cov_loc_star']:
                        data_em['flux'] = data_loc_star['flux']-data_loc['flux']
                        data_em['cov'] = data_loc['cov']
                    else:
                        data_em['flux'] = np.zeros(data_vis['dim_exp'], dtype=float)
                        data_em['cov'] = np.zeros(data_dic[inst]['nord'], dtype=object)
                        for iord in range(data_dic[inst]['nord']):
                            data_em['flux'][iord],data_em['cov'][iord]=bind.add(data_loc_star['flux'][iord], data_loc_star['cov'][iord],-data_loc['flux'][iord], data_loc['cov'][iord])                 
                    data_em['cond_def'] = ~np.isnan(data_em['flux'])                 

                #Saving data               
                np.savez_compressed(data_vis['proc_Atm_data_paths']+str(iexp),data=data_em,allow_pickle=True)
                  
            #------------------------------------------------------------------------------------------------------------

            #Extraction of absorption signal
            #    - the absorption signal in-transit is retrieved as
            # Abs(w,t,vis) = ( F_diff(w,t,vis) - Fstar_loc(w,t,vis) ) / ( Fstar_loc(w,t,vis)/( Sthick(band,t)/Sstar ) )                    
            #      subtracting Fstar_loc(w,t,vis) removes the local stellar profile absorbed by the planetary continuum
            #      dividing by Fstar_loc(w,t,vis) then removes the contribution of the local stellar profile
            #      rescaling by Sthick(band,t)/Sstar finally replaces the scaling of the planet-occulted region surface by the full stellar surface, so that the result is comparable with classical absorption signal
            #              = ( [ ( fp(w,vis)*(Sthick(band,t) + Sthin(w-wp,t) ) -  Fatm(w-wp,t) )*Cref(band)/Fr(vis) ]  - [ fp(w,vis)*Sthick(band,t)*Cref(band)/Fr(vis) ]  ) / ( [ fp(w,vis)*Sthick(band,t)*Cref(band)/Fr(vis) ]/( Sthick(band,t)/Sstar ) )    
            #              = ( ( fp(w,vis)*(Sthick(band,t) + Sthin(w-wp,t) ) -  Fatm(w-wp,t) )  -  fp(w,vis)*Sthick(band,t)  ) / ( fp(w,vis)*Sstar )                     
            #              = ( fp(w,vis)*Sthin(w-wp,t) -  Fatm(w-wp,t) )/ ( fp(w,vis)*Sstar )                     
            #              = Sthin(w-wp,t)/Sstar -  Fatm(w-wp,t)/( fp(w,vis)*Sstar )   
            #    - the calculation sums up as:
            # Abs(w,t,vis) = [ F_diff(w,t,vis)/Fstar_loc(w,t,vis)  - 1 ]*( Sthick(band,t)/Sstar )
            #    - the absorption signal can be contaminated by an emission signal, however the latter likely varies over the planet orbit (thus preventing us to compute an out-of-transit master to be corrected from 
            # in-transit exposures) and is unlikely to be visible during transit in any case as it would arise from nightside emission
            #      if no emission is present, the corrected spectra correspond to:
            # Signal(w,t) = Sthin(w-wp,t )/Sstar
            #    - we use a numerical estimate of Sthick(band,t)/Sstar with a constant Sstar, so that we extract the pure atmospheric spectral surface, normalized
            # by a constant stellar surface to be equivalent to an absorption signal, but unbiased by a spectral stellar surface
            #    - if we consider that there is an absorption signal outside of the transit defined by the input light curve, ie that a region of the planetary atmosphere is absorbing
            # in a specific line but not in the continuum:
            # F_diff(w,t,vis) = ( fp(w,vis)*Sthin(w-wp,t) -  Fatm(w-wp,t) )*Cref(band)/Fr(vis) 
            if (data_dic['Atm']['pl_atm_sign']=='Absorption') and (i_in>-1) and (i_in in idx_est_loc):

                #Planet-to-star surface ratios
                SpSstar_spec = np.zeros([data_dic[inst]['nord'],data_vis['nspec']],dtype=float)   
                                          
                #Achromatic/chromatic planet-to-star radius ratio
                #    - for now a single transiting planet is considered
                if ('spec' in data_vis['type']) and ('chrom' in data_dic['DI']['system_prop']):SpSstar_chrom = True
                else:
                    if len(data_vis['studied_pl'])>1:stop()
                    SpSstar_spec[:,:] = dic_plocc_prop['achrom'][data_vis['studied_pl'][0]]['SpSstar'][0,i_in] 
                    SpSstar_chrom = False
             
                #Processing each order
                data_abs = {'cen_bins':data_loc['cen_bins'],'edge_bins':data_loc['edge_bins'],
                            'flux' : np.zeros(data_vis['dim_exp'], dtype=float),
                            'cov' : np.zeros(data_dic[inst]['nord'], dtype=object)}
                for iord in range(data_dic[inst]['nord']):
                    
                    #Chromatic planet-to-star radius ratio
                    if SpSstar_chrom: 
                        SpSstar_spec[iord] = np_interp(data_loc['cen_bins'][iord],data_dic['DI']['system_prop']['chrom']['w'],dic_plocc_prop['chrom']['SpSstar'][:,i_in],left=dic_plocc_prop['chrom']['SpSstar'][0,i_in],right=dic_plocc_prop['chrom']['SpSstar'][-1,i_in])

                    #Calculation of absorption signal                                            
                    if data_dic['Intr']['cov_loc_star']:dat_temp,cov_temp = bind.div(data_loc['flux'][iord],data_loc['cov'][iord],data_loc_star['flux'][iord], data_loc_star['cov'][iord])
                    else:dat_temp,cov_temp = bind.mul_array(data_loc['flux'][iord],data_loc['cov'][iord],1./data_loc_star['flux'][iord])
                    data_abs['flux'][iord],data_abs['cov'][iord] = bind.mul_array(dat_temp-1.,cov_temp,SpSstar_spec[iord])
                data_abs['cond_def'] = ~np.isnan(data_abs['flux'])                                          
                data_abs['SpSstar_spec'] = SpSstar_spec   
                    
                #Saving data               
                np.savez_compressed(data_vis['proc_Atm_data_paths']+str(i_in),data=data_abs,allow_pickle=True)
                
    #------------------------------------------------------------------------------------------------------------
                
    #Checking that local data has been calculated for all exposures
    else:
        data_paths={iexp:data_vis['proc_Atm_data_paths']+str(iexp) for iexp in plAtm_vis['idx_def']}        
        check_data(data_paths)

    #Path to associated tables
    #    - atmospheric profiles are extracted in the same frame as differential profiles
    #    - indexes may be limited to in-transit indexes if absorption signals are extracted
    if ('mast_Diff_data_paths' in data_vis):data_vis['mast_Atm_data_paths'] = {}
    if ('spec' in data_vis['type']) and gen_dic['corr_tell']:
        if ('tell_Diff_data_paths' not in data_vis):stop('ERROR : weighing telluric profiles undefined; make sure you activate gen_dic["calc_proc_data"] and gen_dic["calc_corr_tell"] when running this module.')
        data_vis['tell_Atm_data_paths'] = {}
    if data_vis['type']=='spec2D':
        data_vis['mean_gcal_Atm_data_paths'] = {}
        if ('sing_gcal_Diff_data_paths' not in data_vis):stop('ERROR : weighing calibration profiles undefined; make sure you activate gen_dic["calc_proc_data"] and gen_dic["calc_gcal"] when running this module.')  
        data_vis['sing_gcal_Atm_data_paths']={} 
    var_key = None
    for subtype_gen in gen_dic['earliertypes4var']['Atm']:
        if data_dic[subtype_gen]['spec2D_to_spec1D'][inst]:
            var_key = gen_dic['type2var'][gen_dic['typegen2type'][subtype_gen]]
            data_vis[var_key+'_'+data_dic['Atm']['pl_atm_sign']+'_data_paths']={}
            break        
    for iexp_atm,iexp in zip(plAtm_vis['idx_def'],iexp_glob):
        if ('mast_Atm_data_paths' in data_vis):data_vis['mast_Atm_data_paths'][iexp_atm] = data_vis['mast_Diff_data_paths'][iexp] 
        if ('tell_Atm_data_paths' in data_vis):data_vis['tell_Atm_data_paths'][iexp_atm] = data_vis['tell_Diff_data_paths'][iexp] 
        if ('mean_gcal_Diff_data_paths' in data_vis):
            data_vis['mean_gcal_Atm_data_paths'][iexp_atm] = data_vis['mean_gcal_Diff_data_paths'][iexp] 
            data_vis['sing_gcal_Atm_data_paths'][iexp_atm] = data_vis['sing_gcal_Diff_data_paths'][iexp] 
        if var_key is not None:data_vis[var_key+'_'+data_dic['Atm']['pl_atm_sign']+'_data_paths'][iexp_atm] = data_vis[var_key+'_Diff_data_paths'][iexp] 

    return None    





################################################################################################## 
#%% EvE output routines
################################################################################################## 
    
def EvE_outputs():
    
    # #EvE output directory
    # if gen_dic['EvE_outputs'] and (not path_exist(gen_dic['save_data_dir']+'EvE_outputs/')):makedirs(gen_dic['save_data_dir']+'EvE_outputs/')      
    
    
    return None
    
    
    
    
    