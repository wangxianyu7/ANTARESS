#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import bindensity as bind
from copy import deepcopy
import glob
from ..ANTARESS_general.utils import stop,np_where1D,dataload_npz,default_func,check_data
from ..ANTARESS_general.constant_data import c_light
from ..ANTARESS_grids.ANTARESS_coord import excl_plrange,calc_pl_coord,conv_phase
from ..ANTARESS_grids.ANTARESS_occ_grid import sub_calc_plocc_spot_prop,retrieve_spots_prop_from_param

def process_bin_prof(mode,data_type_gen,gen_dic,inst,vis_in,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,spot_dic={},masterDI=False):
    r"""**Binning routine**

    Bins series of input spectral profile into a new series along the chosen temporal/spatial dimension.
    
     - for a given visit or between several visits
     - binned profiles are calculated as weighted means with weights specific to the type of profiles
     - binned profiles are used for analysis purposes
     - master profiles used to extract residual profiles from each exposure are calculated in `extract_res_profiles()` 

    Args:
        TBD
    
    Returns:
        TBD
    
    """     
    
    data_inst = data_dic[inst]    
        
    #Identifier for saved file
    if mode=='multivis':vis_save = 'binned'      
    else:vis_save = vis_in 
    prop_dic = deepcopy(data_dic[data_type_gen]) 
    if data_type_gen=='DI':   
        data_type='DI'

        #Set default binning properties to calculate master
        #    - if undefined, or if a single master is requested (overwrites settings in this case, and always calculated for a single visit)
        if masterDI or (inst not in prop_dic['prop_bin']) or (vis_save not in list(prop_dic['prop_bin'][inst].keys())):
            prop_dic['dim_bin']='phase'
            if inst not in prop_dic['prop_bin']:prop_dic['prop_bin'][inst] = {}
            if (vis_save not in prop_dic['prop_bin'][inst]):prop_dic['prop_bin'][inst]={vis_save:{'bin_low':[-0.5],'bin_high':[0.5]}} 
            
        #Calculation of a master spectrum used for weighing
        #    - calculated per visit
        #    - for the purpose of weighing we calculate a single master on the common spectral table of the visit, rather than recalculate the master for every weighing profile
        #      we assume the blurring that will be introduced by the resampling of this master on the table of each exposure is negligible in the weighing process
        if masterDI:
            if (data_dic['DI'][inst][vis_in]['rest_frame']!='star') and (data_dic['DI']['sysvel'][inst][vis_in]!=0.):print('WARNING: disk-integrated profiles must be aligned')
            prop_dic['idx_in_bin']=gen_dic['DImast_idx_in_bin']
       
            #Initialize path of weighing master for disk-integrated exposures
            #    - the paths return to the single master common to all exposures, and for now defined on the same table
            data_dic[inst][vis_in]['mast_'+data_type+'_data_paths'] = {iexp:gen_dic['save_data_dir']+data_type+'_data/Master/'+inst+'_'+vis_in+'_phase' for iexp in range(data_dic[inst][vis_in]['n_in_visit'])}
            save_pref = gen_dic['save_data_dir']+data_type+'_data/Master/'+inst+'_'+vis_in+'_phase'

    elif data_type_gen=='Intr':
        data_type='Intr'
    elif data_type_gen == 'Atm':
        data_type = data_dic['Atm']['pl_atm_sign']     
    if not masterDI:
        print('   > Binning '+gen_dic['type_name'][data_type_gen]+' profiles over '+prop_dic['dim_bin'])
        calc_check=gen_dic['calc_'+data_type_gen+'bin'+mode]
        if mode=='multivis':prop_dic[inst]['binned']={} 
        save_pref = gen_dic['save_data_dir']+data_type_gen+'bin_data/'
        if data_type_gen=='Atm':save_pref+=data_dic['Atm']['pl_atm_sign']+'/'
        save_pref+=inst+'_'+vis_save+'_'+prop_dic['dim_bin']
    else:
        print('   > Calculating master stellar spectrum') 
        calc_check=gen_dic['calc_DImast'] 

    if (calc_check):
        print('         Calculating data') 

        #Set default binning properties to calculate single master
        if (inst not in prop_dic['prop_bin']):prop_dic['prop_bin'][inst] = {}
        if (vis_save not in list(prop_dic['prop_bin'][inst].keys())):
            if prop_dic['dim_bin']=='phase':prop_dic['prop_bin'][inst]={vis_save:{'bin_low':[-0.5],'bin_high':[0.5]}} 
            elif prop_dic['dim_bin']=='r_proj':prop_dic['prop_bin'][inst]={vis_save:{'bin_low':[0.],'bin_high':[1.]}}             
            else:stop('Undefined')
            
        #Bin properties
        bin_prop = prop_dic['prop_bin'][inst][vis_save]
               
        #Several visits are binned together
        #    - common table and dimensions are shared between visits
        #    - new coordinates are relative to a planet chosen as reference for the binned coordinates, which must be present in all binned visits 
        if mode=='multivis':

            #Retrieving table common to all visits
            #    - defined in input rest frame for spectra
            #    - centered on the input systemic velocity for DI CCFs and on 0 for intrinsic/atmospheric CCFs
            data_com = dataload_npz(data_inst['proc_com_data_path'])
            dim_exp_com = data_inst['dim_exp']  
            nspec_com = data_inst['nspec'] 
      
            #Visits to include in the binning
            vis_to_bin = prop_dic['vis_in_bin'][inst] if ((inst in data_dic[data_type_gen]['vis_in_bin']) and (len(data_dic[data_type_gen]['vis_in_bin'][inst])>0)) else data_dic[inst]['visit_list']

            #Planets associated with the binned visits
            data_inst.update({vis_save:{'transit_pl':[]}})
            for vis_bin in vis_to_bin:data_inst[vis_save]['transit_pl']+=data_inst[vis_bin]['transit_pl']
            data_inst[vis_save]['transit_pl'] = list(np.unique(data_inst[vis_save]['transit_pl']))
            
            #Mean systemic velocity 
            sysvel=0.
            for vis_bin in vis_to_bin:sysvel+=data_dic['DI']['sysvel'][inst][vis_bin]
            sysvel/=len(vis_to_bin)
            
            #Common data type
            data_mode = data_inst['type']
            
            #Common rest frame
            rest_frame=[]
            for vis_bin in vis_to_bin:rest_frame+=[data_dic['DI'][inst][vis_bin]['rest_frame']]
            if len(np.unique(rest_frame))>1:stop('Incompatible rest frames') 
            rest_frame = np.unique(rest_frame)[0]
           
            #System properties
            system_prop = data_dic[inst]['system_prop']
            
        #A single visit is processed
        #    - common table and dimensions are specific to this visit
        elif mode=='': 
            data_com = dataload_npz(data_inst[vis_in]['proc_com_data_paths'])
            dim_exp_com = data_inst[vis_in]['dim_exp']
            nspec_com = data_inst['nspec']
            vis_to_bin=[vis_in]
            sysvel = data_dic['DI']['sysvel'][inst][vis_in]
            data_mode = data_inst[vis_in]['type']
            rest_frame = data_dic['DI'][inst][vis_in]['rest_frame']
            system_prop = data_dic[inst][vis_in]['system_prop']
 
        #Store flags
        #    - 'FromAligned' set to True if binned profiles were aligned before
        #    - 'in_inbin' set to True if binned profiles include at least one in-transit profile
        data_glob_new={'FromAligned':gen_dic['align_'+data_type_gen],'in_inbin' : False}

        #Automatic definition of reference planet 
        #    - for single-transiting planet or if undefined    
        if (len(data_inst[vis_save]['transit_pl'])==1) or ('ref_pl' not in bin_prop):bin_prop['ref_pl'] = data_inst[vis_save]['transit_pl'][0]  

        #Initialize binning
        new_x_cen,new_x_low,new_x_high,_,n_in_bin_all,idx_to_bin_all,dx_ov_all,n_bin,idx_bin2orig,idx_bin2vis,idx_to_bin_unik = init_bin_prof(data_type,bin_prop,prop_dic['idx_in_bin'],prop_dic['dim_bin'],coord_dic,inst,vis_to_bin,data_dic,gen_dic)

        #Retrieving data that will be used in the binning
        #    - original data is associated with its original index, so that it can be retrieved easily by the binning routine
        #      for each new bin, the binning routine is called with the list of original index that it will use
        #    - different binned profiles might use the same original exposures, which is why we use 'idx_to_bin_unik' to pre-process only once original exposures 
        data_to_bin={}
        if (data_type_gen=='DI') and (not masterDI):
            data_glob_new['RV_star_solCDM'] = np.zeros(n_bin,dtype=float)
            data_glob_new['RV_star_stelCDM'] = np.zeros(n_bin,dtype=float)
        data_glob_new['vis_iexp_in_bin']={vis_bin:{} for vis_bin in vis_to_bin}
        for iexp_off in idx_to_bin_unik:
            data_to_bin[iexp_off]={}
            
            #Original index and visit of contributing exposure
            #    - iexp is relative to global or in-transit indexes depending on data_type
            iexp = idx_bin2orig[iexp_off]
            vis_bin = idx_bin2vis[iexp_off]            
            data_glob_new['vis_iexp_in_bin'][vis_bin][iexp]={}
            if (data_type not in ['Intr','Absorption']) and (gen_dic[inst][vis_bin]['idx_exp2in'][iexp]!=-1.):data_glob_new['in_inbin']=True

            #Latest processed data
            #    - profiles should have been aligned in the star rest frame and rescaled to their correct flux level, if necessary
            flux_est_loc_exp = None
            cov_est_loc_exp = None
            SpSstar_spec = None
            data_exp = dataload_npz(data_inst[vis_bin]['proc_'+data_type_gen+'_data_paths']+str(iexp))
            data_glob_new['vis_iexp_in_bin'][vis_bin][iexp]['data_path'] = data_inst[vis_bin]['proc_'+data_type_gen+'_data_paths']+str(iexp)
            if gen_dic['flux_sc']:scaled_data_paths = data_dic[inst][vis_bin]['scaled_'+data_type_gen+'_data_paths']
            else:scaled_data_paths = None
            if data_inst[vis_bin]['tell_sp']:
                data_exp['tell'] = dataload_npz(data_inst[vis_bin]['tell_'+data_type_gen+'_data_paths'][iexp])['tell']  
                data_glob_new['vis_iexp_in_bin'][vis_bin][iexp]['tell_path'] = data_inst[vis_bin]['tell_'+data_type_gen+'_data_paths'][iexp]
            else:data_exp['tell'] = None
            if gen_dic['cal_weight'] and data_inst[vis_bin]['mean_gdet']:data_exp['mean_gdet'] = dataload_npz(data_inst[vis_bin]['mean_gdet_'+data_type_gen+'_data_paths'][iexp])['mean_gdet'] 
            else:data_exp['mean_gdet'] = None
            if data_type_gen=='DI': 
                iexp_glob=iexp
                
                #Store Keplerian motion
                if ('RV_star_solCDM' in data_glob_new):
                    data_to_bin[iexp_off]['RV_star_solCDM'] = coord_dic[inst][vis_bin]['RV_star_solCDM'][iexp_glob]
                    data_to_bin[iexp_off]['RV_star_stelCDM'] = coord_dic[inst][vis_bin]['RV_star_stelCDM'][iexp_glob]
                
            else:
                                
                #Intrinsic profiles
                #    - beware that profiles were aligned if binning dimension is not phase
                if data_type=='Intr':
                    iexp_glob = gen_dic[inst][vis_bin]['idx_in2exp'][iexp]
               
                #Atmospheric profiles
                #    - beware that profiles were aligned if binning dimension is not phase
                elif data_type_gen=='Atm':
                    if data_type=='Absorption':
                        iexp_glob = gen_dic[inst][vis_bin]['idx_in2exp'][iexp] 
                        
                        #Planet-to-star radius ratio
                        #    - profile has not been aligned but is varying with low-frequency, and the shifts are thus not critical to the weighing
                        SpSstar_spec = data_exp['SpSstar_spec']
                        
                    elif data_type=='Emission':iexp_glob=iexp

                    #Estimate of local stellar profile for current exposure  
                    #    - defined on the same table as data_exp
                    if (data_type=='Absorption') or ((data_type=='Emission') and data_dic['Intr']['cov_loc_star']): 
                        data_est_loc=np.load(data_dic[inst][vis_bin]['LocEst_Atm_data_paths'][iexp]+'.npz',allow_pickle=True)['data'].item() 
             
            #Resampling on common spectral table if required
            #    - condition is True unless all exposures of 'vis_bin' are defined on a common table, and it is the reference for the binning    
            #    - upon first calculation of the weighing DI master, no DI stellar spectrum is available and it is set to 1 (the weighing DI master must in any case be calculated from stellar spectra aligned in the star rest frame, where stellar lines will not contribute to the weighing)         
            #      the master stellar spectrum is also set to 1 when binning DI profiles in the star rest frame (where stellar lines will not contribute to the weighing) 
            #    - if the resampling condition is not met, then all profiles have been resampled on the common table for the visit, and the master does not need resampling as:
            # + it is still on its original table, which is the common table for the visit
            # + it has been shifted, and resampled on the table of the associated profile, which is also the common table            
            #    
            #    - data is stored with the same indexes as in idx_to_bin_all
            #    - all exposures must be defined on the same spectral table before being binned
            #    - profiles are resampled if :
            # + profiles are defined on their individual tables
            # + several visits are used, profiles have already been resampled within the visit, but not all visits share a common table and visit of binned exposure is not the one used as reference to set the common table
            #    - telluric are set to 1 if unused
            cov_ref_exp = None  
            flux_ref_exp=np.ones(dim_exp_com,dtype=float)
            if masterDI:
                dt_exp = 1.   
            else:
                if len(np.array(glob.glob(data_dic[inst][vis_bin]['mast_'+data_type_gen+'_data_paths'][iexp]+'.npz')))==0:stop('No weighing master found. Activate "gen_dic["DImast_weight"]" and "gen_dic["calc_DImast"]".') 
                data_ref = dataload_npz(data_dic[inst][vis_bin]['mast_'+data_type_gen+'_data_paths'][iexp])         
                dt_exp = coord_dic[inst][vis_bin]['t_dur'][iexp]
            if ((mode=='') and (not data_inst[vis_bin]['comm_sp_tab'])) or ((mode=='multivis') and (not data_inst['comm_sp_tab']) and (vis_bin!=data_inst['com_vis'])):

                #Resampling exposure profile
                data_to_bin[iexp_off]['flux']=np.zeros(dim_exp_com,dtype=float)*np.nan
                data_to_bin[iexp_off]['cov']=np.zeros(data_inst['nord'],dtype=object)
                if not masterDI:
                    if ((data_type_gen=='DI') and (rest_frame!='star')) or (data_type_gen!='DI'):flux_ref_exp=np.zeros(dim_exp_com,dtype=float)*np.nan
                    if data_type_gen!='DI':cov_ref_exp=np.zeros(data_inst['nord'],dtype=object)
                tell_exp=np.ones(dim_exp_com,dtype=float) if data_inst[vis_bin]['tell_sp'] else None
                mean_gdet_exp=np.ones(dim_exp_com,dtype=float) if gen_dic['cal_weight'] and data_inst[vis_bin]['mean_gdet'] else None                
                for iord in range(data_inst['nord']): 
                    data_to_bin[iexp_off]['flux'][iord],data_to_bin[iexp_off]['cov'][iord] = bind.resampling(data_com['edge_bins'][iord], data_exp['edge_bins'][iord], data_exp['flux'][iord] , cov = data_exp['cov'][iord], kind=gen_dic['resamp_mode'])                                                        
                    if not masterDI:
                        if (data_type_gen=='DI') and (rest_frame!='star'):flux_ref_exp[iord] = bind.resampling(data_com['edge_bins'][iord], data_ref['edge_bins'][iord], data_ref['flux'][iord], kind=gen_dic['resamp_mode'])                                                                            
                        elif (data_type_gen!='DI'):flux_ref_exp[iord],cov_ref_exp[iord] = bind.resampling(data_com['edge_bins'][iord], data_ref['edge_bins'][iord], data_ref['flux'][iord] , cov = data_ref['cov'][iord], kind=gen_dic['resamp_mode'])                                                        
                    if data_inst[vis_bin]['tell_sp']:tell_exp[iord] = bind.resampling(data_com['edge_bins'][iord], data_exp['edge_bins'][iord], data_exp['tell'][iord] , kind=gen_dic['resamp_mode']) 
                    if gen_dic['cal_weight'] and data_inst[vis_bin]['mean_gdet'] :mean_gdet_exp[iord] = bind.resampling(data_com['edge_bins'][iord], data_exp['edge_bins'][iord], data_exp['mean_gdet'][iord] , kind=gen_dic['resamp_mode'])                    
                data_to_bin[iexp_off]['cond_def'] = ~np.isnan(data_to_bin[iexp_off]['flux'])  

                #Resample local stellar profile estimate
                if (data_type_gen=='Atm'):
                    flux_est_loc_exp = np.zeros(dim_exp_com,dtype=float)
                    if data_dic['Intr']['cov_loc_star']:
                        cov_est_loc_exp = np.zeros(data_inst['nord'],dtype=object)
                        for iord in range(data_inst['nord']): 
                            flux_est_loc_exp[iord] ,cov_est_loc_exp[iord] = bind.resampling(data_com['edge_bins'][iord], data_exp['edge_bins'][iord], data_est_loc['flux'][iord] , cov = data_est_loc['cov'][iord], kind=gen_dic['resamp_mode'])                                                        
                    else:
                        cov_est_loc_exp = np.zeros([data_inst['nord'],1],dtype=float)
                        for iord in range(data_inst['nord']): 
                            flux_est_loc_exp[iord] = bind.resampling(data_com['edge_bins'][iord], data_exp['edge_bins'][iord], data_est_loc['flux'][iord] , kind=gen_dic['resamp_mode'])                                                        
                                        
            #No resampling required
            #    - if the resampling condition is not met, then all profiles have been resampled on the common table for the visit
            #    - the local stellar profile estimate does not need resampling as:
            # + it is on its original table, which is the table of the associated profile, which is also the common table  
            # + it has been shifted, and resampled on the table of the associated profile, which is also the common table   
            else:                
                if not masterDI:
                    if ((data_type_gen=='DI') and (rest_frame!='star')) or (data_type_gen!='DI'):flux_ref_exp = data_ref['flux']
                    if data_type_gen!='DI':cov_ref_exp = data_ref['cov']   
                for key in ['flux','cond_def','cov']:data_to_bin[iexp_off][key] = data_exp[key] 
                tell_exp=data_exp['tell']
                mean_gdet_exp=data_exp['mean_gdet'] 
                if (data_type_gen=='Atm'):
                    flux_est_loc_exp = data_est_loc['flux']
                    if data_dic['Intr']['cov_loc_star']:cov_est_loc_exp = data_est_loc['cov'] 
                    else:cov_est_loc_exp = np.zeros([data_inst['nord'],1],dtype=float)   

            #Exclude planet-contaminated bins  
            if (data_type_gen=='DI') and ('DI_Mast' in data_dic['Atm']['no_plrange']) and (iexp_glob in data_dic['Atm'][inst][vis_bin]['iexp_no_plrange']):
                for iord in range(data_inst['nord']):   
                    data_to_bin[iexp_off]['cond_def'][iord] &= excl_plrange(data_to_bin[iexp_off]['cond_def'][iord],data_dic['Atm'][inst][vis_bin]['exclu_range_star'],iexp_glob,data_com['edge_bins'][iord],data_mode)[0]

            #Weight definition
            #    - the profiles must be specific to a given data type so that earlier types can still be called in the multi-visit binning, after the type of profile has evolved in a given visit
            #    - at this stage of the pipeline broadband flux scaling has been defined, if requested 
            data_to_bin[iexp_off]['weight'] = weights_bin_prof(range(data_inst['nord']),scaled_data_paths,inst,vis_bin,gen_dic['corr_Fbal'],gen_dic['corr_FbalOrd'],gen_dic['save_data_dir'],gen_dic['type'],data_inst['nord'],iexp_glob,data_type,data_mode,dim_exp_com,tell_exp,mean_gdet_exp,data_com['cen_bins'],dt_exp,flux_ref_exp,cov_ref_exp,flux_est_loc_exp=flux_est_loc_exp,cov_est_loc_exp = cov_est_loc_exp, SpSstar_spec = SpSstar_spec,bdband_flux_sc = gen_dic['flux_sc'])                          

            #Timestamp of exposure
            data_to_bin[iexp_off]['bjd'] = coord_dic[inst][vis_bin]['bjd'][iexp]

            #Duration of exposure
            data_to_bin[iexp_off]['t_dur'] = coord_dic[inst][vis_bin]['t_dur'][iexp]

        #----------------------------------------------------------------------------------------------

        #Preparing an array that will contain the timestamp and duration of each binned exposure
        binned_time = np.zeros(len(idx_to_bin_all),dtype=float)
        binned_t_dur = np.zeros(len(idx_to_bin_all),dtype=float)

        #Processing and analyzing each new exposure 
        for i_new,(idx_to_bin,n_in_bin,dx_ov) in enumerate(zip(idx_to_bin_all,n_in_bin_all,dx_ov_all)):

            #Calculate binned exposure on common spectral table
            data_exp_new = calc_bin_prof(idx_to_bin,data_dic[inst]['nord'],dim_exp_com,nspec_com,data_to_bin,inst,n_in_bin,data_com['cen_bins'],data_com['edge_bins'],dx_ov_in = dx_ov)

            #Keplerian motion relative to the stellar CDM and the Sun (km/s)
            if ('RV_star_solCDM' in data_glob_new):
                RV_star_solCDM = 0.
                RV_star_stelCDM = 0.
                for isub,ibin in enumerate(idx_to_bin):
                    RV_star_solCDM+=dx_ov[isub]*data_to_bin[ibin]['RV_star_solCDM']
                    RV_star_stelCDM+=dx_ov[isub]*data_to_bin[ibin]['RV_star_stelCDM']
                data_glob_new['RV_star_solCDM'][i_new] = RV_star_solCDM/np.sum(dx_ov)
                data_glob_new['RV_star_stelCDM'][i_new] = RV_star_stelCDM/np.sum(dx_ov)

            #Saving new exposure  
            if not masterDI:np.savez_compressed(save_pref+str(i_new),data=data_exp_new,allow_pickle=True)
    
            #Calculate the timestamp (BJD) and duration of the binned exposure(s)
            time_to_bin = np.zeros(len(idx_to_bin),dtype=float)
            dur_to_bin = np.zeros(len(idx_to_bin),dtype=float)
            for loc, indiv_idx in enumerate(idx_to_bin):
                time_to_bin[loc] = data_to_bin[indiv_idx]['bjd']
                dur_to_bin[loc] = data_to_bin[indiv_idx]['t_dur']
            binned_time[i_new] = np.average(time_to_bin)
            binned_t_dur[i_new] = np.average(dur_to_bin)
           
        #Store path to weighing master 
        if masterDI:
            np.savez_compressed(data_dic[inst][vis_in]['mast_'+data_type+'_data_paths'][0],data=data_exp_new,allow_pickle=True) 

        #Store common table of binned profiles
        data_glob_new['cen_bins'] = data_com['cen_bins']
        data_glob_new['edge_bins'] = data_com['edge_bins']
        
        #---------------------------------------------------------------------------
        #Calculating associated properties 
        #    - calculation of theoretical properties of planet-occulted regions is only possible if data binned over phase
        #    - new coordinates are relative to the planet chosen as reference for the binned coordinates 
        #---------------------------------------------------------------------------
        data_glob_new.update({'st_bindim':new_x_low,'end_bindim':new_x_high,'cen_bindim':new_x_cen,'n_exp':n_bin,'dim_all':[n_bin]+dim_exp_com,'dim_exp':dim_exp_com,'nspec':nspec_com,'rest_frame' : rest_frame})
        if (prop_dic['dim_bin'] == 'phase'):  

            #Coordinates of planets for new exposures
            #    - phase is associated with the reference planet, and must be converted into phases of the other planets 
            ecl_all = np.zeros(data_glob_new['n_exp'],dtype=bool)
            data_glob_new['coord'] = {}
            for pl_loc in data_dic[inst][vis_save]['transit_pl']:
                pl_params_loc=system_param[pl_loc]
            
                #Phase conversion
                phase_tab = conv_phase(coord_dic,inst,vis_save,system_param,bin_prop['ref_pl'],pl_loc,np.vstack((new_x_low,new_x_cen,new_x_high)))              

                #Transit center check
                if mode=='multi_vis':
                    for vis_bin in vis_to_bin:
                        if (inst in gen_dic['Tcenter_visits'][pl_loc]) and (vis_bin in gen_dic['Tcenter_visits'][pl_loc][inst]):stop('WARNING: you are binning visits with TTVs')

                #Coordinates
                x_pos_pl,y_pos_pl,z_pos_pl,Dprojp,_,_,_,_,ecl_pl = calc_pl_coord(pl_params_loc['ecc'],pl_params_loc['omega_rad'],pl_params_loc['aRs'],pl_params_loc['inclin_rad'],phase_tab,system_prop['achrom'][pl_loc][0],pl_params_loc['lambda_rad'],system_param['star'])
                data_glob_new['coord'][pl_loc] = {
                    'Tcenter':coord_dic[inst][vis_to_bin[0]][pl_loc]['Tcenter'], 
                    'ecl':ecl_pl,
                    'st_ph':phase_tab[0],'cen_ph':phase_tab[1],'end_ph':phase_tab[2],
                    'st_pos':np.vstack((x_pos_pl[0],y_pos_pl[0],z_pos_pl[0])),
                    'cen_pos':np.vstack((x_pos_pl[1],y_pos_pl[1],z_pos_pl[1])),
                    'end_pos':np.vstack((x_pos_pl[2],y_pos_pl[2],z_pos_pl[2]))}

                #Exposure considered out-of-transit if no planet at all is transiting
                ecl_all |= abs(ecl_pl)!=1            
  
                #Orbital rv of current planet in star rest frame
                if data_dic['Atm']['exc_plrange']:
                    data_glob_new['coord'][pl_loc]['rv_pl'] = np.zeros(len(phase_tab[1]))*np.nan
                    dphases = phase_tab[2] - phase_tab[0]
                    for isub,dph_loc in enumerate(dphases):
                        nt_osamp_RV=max(int(dph_loc/data_dic['Atm']['dph_osamp_RVpl']),2)
                        dph_osamp_loc=dph_loc/nt_osamp_RV
                        ph_osamp_loc = phase_tab[0]+dph_osamp_loc*(0.5+np.arange(nt_osamp_RV))
                        data_glob_new['coord'][pl_loc]['rv_pl'][isub]=system_param['star']['RV_conv_fact']*np.mean(calc_pl_coord(pl_params_loc['ecc'],pl_params_loc['omega_rad'],pl_params_loc['aRs'],pl_params_loc['inclin_rad'],ph_osamp_loc,None,None,None,rv_LOS=True,omega_p=pl_params_loc['omega_p'])[6])

            #In-transit data
            data_glob_new['idx_in']=np_where1D(ecl_all)
            data_glob_new['idx_out']=np_where1D(~ecl_all)
            data_glob_new['n_in_tr'] = len(data_glob_new['idx_in'])
            data_glob_new['idx_exp2in'] = np.zeros(data_glob_new['n_exp'],dtype=int)-1
            data_glob_new['idx_exp2in'][data_glob_new['idx_in']]=np.arange(data_glob_new['n_in_tr'])
            data_glob_new['idx_in2exp'] = np.arange(data_glob_new['n_exp'],dtype=int)[data_glob_new['idx_in']]
            data_glob_new['dim_in'] = [data_glob_new['n_in_tr']]+data_glob_new['dim_exp']
 
            #Binned exposure timestamp and duration
            data_glob_new['coord']['bjd'] = binned_time
            data_glob_new['coord']['t_dur'] = binned_t_dur

            #Properties of planet-occulted and spot-occulted regions 
            params = deepcopy(system_param['star'])
            params.update({'rv':0.,'cont':1.}) 
            params['use_spots']=False
            if (inst in spot_dic):
                if mode=='multivis':
                    print('WARNING: spots properties are not propagated for multiple visits.')
                elif vis_in in spot_dic[inst]:

                    #Trigger spot use
                    params['use_spots']=True

                    #Initialize entries for spot coordinates 
                    for spot in data_dic[inst][vis_in]['transit_sp']:
                        data_glob_new['coord'][spot]={}
                        for key in ['Tcenter', 'ang', 'ang_rad', 'lat', 'ctrst']:data_glob_new['coord'][spot][key] = np.zeros(n_bin,dtype=float)*np.nan
                        for key in ['lat_rad_exp','sin_lat_exp','cos_lat_exp','long_rad_exp','sin_long_exp','cos_long_exp','x_sky_exp','y_sky_exp','z_sky_exp']:data_glob_new['coord'][spot][key] = np.zeros([3,n_bin],dtype=float)*np.nan
                        data_glob_new['coord'][spot]['is_visible'] = np.zeros([3,n_bin],dtype=float)

                    #Retrieve spot coordinates/properties for new exposures
                    spot_dic[inst][vis_in]['cos_istar']=system_param['star']['cos_istar']

                    for i_new in range(n_bin):
                        spots_prop = retrieve_spots_prop_from_param(system_param['star'],spot_dic[inst][vis_in],inst,vis_in,data_glob_new['coord']['bjd'][i_new],exp_dur=data_glob_new['coord']['t_dur'][i_new])
                        for spot in data_dic[inst][vis_in]['transit_sp']:
                            for key in ['Tcenter', 'ang', 'ang_rad', 'lat', 'ctrst']:
                                data_glob_new['coord'][spot][key][i_new] = spots_prop[spot][key]

                            for key in ['lat_rad_exp','sin_lat_exp','cos_lat_exp','long_rad_exp','sin_long_exp','cos_long_exp','x_sky_exp','y_sky_exp','z_sky_exp']:
                                data_glob_new['coord'][spot][key][:, i_new] = [spots_prop[spot][key+'_start'],spots_prop[spot][key+'_center'],spots_prop[spot][key+'_end']]

                            data_glob_new['coord'][spot]['is_visible'][:, i_new]=[spots_prop[spot]['is_start_visible'],spots_prop[spot]['is_center_visible'],spots_prop[spot]['is_end_visible']]

                    spot_dic[inst][vis_in].pop('cos_istar')
                                        
            par_list=['rv','CB_RV','mu','lat','lon','x_st','y_st','SpSstar','xp_abs','r_proj']
            key_chrom = ['achrom']
            if ('spec' in data_mode) and ('chrom' in system_prop):key_chrom+=['chrom']
            data_glob_new['plocc_prop'],data_glob_new['spot_prop'],data_glob_new['common_prop'] = sub_calc_plocc_spot_prop(key_chrom,{},par_list,data_inst[vis_save]['transit_pl'],system_param,theo_dic,system_prop,params,data_glob_new['coord'],range(n_bin),system_spot_prop_in=data_dic['DI']['spots_prop'],out_ranges=True)
            
        #---------------------------------------------------------------------------

        #Saving global tables for new exposures
        if not masterDI:
            if (data_type=='DI'):data_glob_new['sysvel'] = sysvel
        np.savez_compressed(save_pref+'_add',data=data_glob_new,allow_pickle=True)

    #Checking that data were calculated 
    #    - check is performed on the complementary table
    else:
        if masterDI: check_data({'path':data_dic[inst][vis_in]['mast_DI_data_paths'][0]})
        else:check_data({'path':save_pref+'_add'})

    return None


def init_bin_prof(data_type,bin_prop,idx_in_bin,dim_bin,coord_dic,inst,vis_to_bin,data_dic,gen_dic):
    r"""**Binning routine: initialization**

    Initializes `process_bin_prof()`. 

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Concatenate tables from all visits to bin
    #    - indexes of original exposures are arbitrarily offset between visits to be distinguishable
    x_low_vis = np.zeros(0,dtype=float)
    x_high_vis = np.zeros(0,dtype=float)
    idx_orig_vis = np.zeros(0,dtype=int)
    vis_orig=np.zeros(0,dtype='U35')
    idx_to_bin_vis = np.zeros(0,dtype=int)
    vis_shift = 0
    for vis in vis_to_bin:

        #Indexes of exposures requested for binning      
        if (inst in idx_in_bin) and (vis in idx_in_bin[inst]) and len(idx_in_bin[inst][vis])>0:idx_in_bin_vis = np.array(idx_in_bin[inst][vis]) 
        else:idx_in_bin_vis = None
    
        #Indexes of original exposures used as input, and default indexes of those exposures that contribute to the binned profiles
        #    - idx_in_bin indexes are relative to all exposures or only in-transit exposures depending on the case
        #    - for aligned disk-integrated profiles, all exposures are considered if a selection is requested
        #      if no selection is done, only out-of-transit exposures are considered
        #      indexes are relative to global tables
        #    - for aligned intrinsic profiles, corresponds to in-transit indexes of exposures with known surface rvs
        #    - for absorption profiles, indexes are relative to in-transit tables
        #      for emission profiles, indexes are relative to global tables       
        if data_type in ['DI','Res']:
            idx_orig = np.arange(data_dic[inst][vis]['n_in_visit'])
            if idx_in_bin_vis is not None:idx_to_bin = idx_in_bin_vis
            else:idx_to_bin = gen_dic[inst][vis]['idx_out']
        elif data_type=='Intr':
            idx_orig = gen_dic[inst][vis]['idx_in']
            idx_to_bin = data_dic['Intr'][inst][vis]['idx_def']    
            if idx_in_bin_vis is not None:idx_to_bin = np.intersect1d(idx_in_bin_vis,idx_to_bin)
        elif data_type in ['Absorption','Emission']:
            if data_type=='Absorption':idx_orig = gen_dic[inst][vis]['idx_in']
            elif data_type=='Emission':idx_orig = np.arange(data_dic[inst][vis]['n_in_visit'])
            idx_to_bin = data_dic['Atm'][inst][vis]['idx_def']
            if idx_in_bin_vis is not None:idx_to_bin = np.intersect1d(idx_in_bin_vis,idx_to_bin)        
    
        #Coordinate tables of input exposures along chosen bin dimension
        #    - tables are restricted to the range of input exposures
        if dim_bin in ['phase']:     
            coord_vis = coord_dic[inst][vis][bin_prop['ref_pl']]
            x_low = coord_vis['st_ph'][idx_orig]
            x_high = coord_vis['end_ph'][idx_orig]
            x_cen_all = coord_vis['cen_ph'][idx_orig]         #central coordinates of all exposures, required for def_plocc_profiles()
            
        elif dim_bin in ['xp_abs','r_proj']: 
            transit_prop_nom = (np.load(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item())['achrom'][bin_prop['ref_pl']]                           
            x_low = transit_prop_nom[dim_bin+'_range'][0,:,0]
            x_high = transit_prop_nom[dim_bin+'_range'][0,:,1]    
            x_cen_all = transit_prop_nom[dim_bin][0,:]      #central coordinates of all exposures, required for def_plocc_profiles()  
        
        #Selection of input exposures contributing to binned profiles
        if len(idx_to_bin)==0:stop('No remaining exposures after input selection')
        x_low = x_low[idx_to_bin]
        x_high = x_high[idx_to_bin]  
    
        #Append all visits
        x_low_vis = np.append(x_low_vis,x_low)
        x_high_vis = np.append(x_high_vis,x_high)
        idx_orig_vis = np.append(idx_orig_vis,idx_to_bin)
        vis_orig = np.append(vis_orig,np.repeat(vis,len(idx_to_bin)))
        idx_to_bin_vis = np.append(idx_to_bin_vis,idx_to_bin+vis_shift)
        vis_shift+=data_dic[inst][vis]['n_in_visit']
   
    #Dictionary to match indexes in concatenated tables toward original indexes and visits
    idx_bin2orig = dict(zip(idx_to_bin_vis, idx_orig_vis))   
    idx_bin2vis = dict(zip(idx_to_bin_vis, vis_orig)) 

    #Coordinates of binned exposures along bin dimension
    if 'bin_low' in bin_prop:
        new_x_low_in = np.array(bin_prop['bin_low'])
        new_x_high_in = np.array(bin_prop['bin_high'])
    elif 'bin_range' in bin_prop:
        min_x = max([bin_prop['bin_range'][0],min(x_low_vis)])
        max_x = min([bin_prop['bin_range'][1],max(x_high_vis)])
        new_dx =  (max_x-min_x)/bin_prop['nbins']
        new_nx = int((max_x-min_x)/new_dx)
        new_x_low_in = min_x + new_dx*np.arange(new_nx)
        new_x_high_in = new_x_low_in+new_dx 
    
    #Limiting contributing profiles to requested range along bin dimension            
    cond_keep = (x_high_vis>=new_x_low_in[0]) &  (x_low_vis <=new_x_high_in[-1])
    x_low_vis = x_low_vis[cond_keep]
    x_high_vis = x_high_vis[cond_keep]         
    idx_to_bin_vis = idx_to_bin_vis[cond_keep]
    if np.sum(cond_keep)==0:stop('No original exposures in bin range')  
  
    #Limiting binned profiles to the original exposure range
    cond_keep = (new_x_high_in>=min(x_low_vis)) &  (new_x_low_in <=max(x_high_vis))
    new_x_low_in = new_x_low_in[cond_keep]        
    new_x_high_in = new_x_high_in[cond_keep]  
    if np.sum(cond_keep)==0:stop('No binned exposures in original range')     
    
    #Properties of binned profiles along chosen dimension
    idx_to_bin_all = []
    n_in_bin_all = []
    dx_ov_all=[]
    new_x_low = np.zeros(0,dtype=float)
    new_x_high = np.zeros(0,dtype=float)
    new_x_cen = np.zeros(0,dtype=float) 
    for i_new,(new_x_low_in_loc,new_x_high_in_loc) in enumerate(zip(new_x_low_in,new_x_high_in)):

        #Original exposures overlapping with new bin
        #    - we use 'where' rather than searchsorted to allow processing original bins that overlap together
        idx_olap = np_where1D( (x_high_vis>=new_x_low_in_loc) &  (x_low_vis <=new_x_high_in_loc) )
        if len(idx_olap)>0:
            
            #Indexes of contributing exposures relative to the original indexes of input exposures
            #    - offset between visits if relevant
            idx_to_bin_all+=[idx_to_bin_vis[idx_olap]]

            #Number of overlapping original exposures
            n_in_bin = len(idx_olap)
            n_in_bin_all += [n_in_bin]

            #Minimum between overlapping exposure upper boundaries and new exposure upper boundary
            #    - if the exposure upper boundary is beyond the new exposure, then the exposure fraction beyond the new exposure upper boundary will not contribute to the binned flux 
            #    - if the exposure upper boundary is within the new exposure, then the exposure will only contribute to the binned flux up to its own boundary
            x_high_ov=np.minimum(x_high_vis[idx_olap],np.repeat(new_x_high_in_loc,n_in_bin))
        
            #Maximum between overlapping exposure lower boundaries and new exposure lower boundary
            #    - if the exposure lower boundary is beyond the new exposure, then the exposure fraction beyond the new exposure upper boundary will not contribute to the binned flux 
            #    - if the exposure lower boundary is within the new exposure, then the exposure will only contribute to the binned flux up to its own boundary
            x_low_ov=np.maximum(x_low_vis[idx_olap],np.repeat(new_x_low_in_loc,n_in_bin))
            
            #Effective boundaries of the new exposure
            new_x_low_loc = min(x_low_ov)
            new_x_high_loc = max(x_high_ov)

            #Center for new exposure
            #    - defined as the barycenter of all overlaps centers
            new_x_cen_loc = np.mean(0.5*(x_low_ov+x_high_ov))
        
            #Width over which each original exposure contributes to the binned flux
            dx_ov_all+=[x_high_ov-x_low_ov]

            #Store updated bin boundary and center
            new_x_low=np.append(new_x_low,new_x_low_loc)
            new_x_high=np.append(new_x_high,new_x_high_loc)
            new_x_cen = np.append(new_x_cen,new_x_cen_loc)
            
    n_bin = len(new_x_cen)
    
    #Unique list of original exposures that will be used in the binning
    #    - indexes are relative to original exposures, but have been offset in case several visits are binned
    idx_to_bin_unik = np.unique(np.concatenate(idx_to_bin_all))      
    
    return new_x_cen,new_x_low,new_x_high,x_cen_all,n_in_bin_all,idx_to_bin_all,dx_ov_all,n_bin,idx_bin2orig,idx_bin2vis,idx_to_bin_unik








def weights_bin_prof(iord_orig_list,scaled_data_paths,inst,vis,gen_corr_Fbal,gen_corr_Fbal_ord,save_data_dir,gen_type,nord,iexp_glob,data_type,data_mode,dim_exp,tell_exp,mean_gdet,cen_bins,dt,flux_ref_exp,cov_ref_exp,ref_val=0.,flux_est_loc_exp=None,cov_est_loc_exp=None,SpSstar_spec=None,bdband_flux_sc=False,glob_flux_sc=None,corr_Fbal = True):
    r"""**Binning routine: weights**

    Defines weights to be used when binning profiles.
    Weights should only be defined using the inverse squared error if the weighted values are comparable, so that all profiles should have been scaled to comparable flux levels prior to binning.
    Profiles must be defined on the same spectral table to be binned together.

    Args:
        TBD
    
    Returns:
        TBD
    
    """  
    #Weight definition
    #    - in case of photon noise the error on photons counts writes as E(n) = sqrt(N) where N is the number of photons received during a time interval dt over a spectral bin dw
    # + if the same measurement is manipulated, eg with N' = a*N1, then the rules of error propagation apply and E(N') = a*E(N1)
    # + if two independent measurements are made, eg related as N2 = a*N1, then E(N2) = sqrt(a*N1) = sqrt(a)*sqrt(N1) = sqrt(a)*E(N1) 
    #      if measurements are provided as a density, eg over time with n = N/dt
    # + if the same measurement is manipulated, eg with n = N/dt, then the rules of error propagation apply and E(n) = E(N)/dt
    # + if two independent measurements are made, eg with dt2 = a*dt1, then N2 = a*N1 and n2 = a*N1/(a*dt1) = n1 but E[n2] = E[N2/dt2] = E[N2]/dt2 = sqrt(N2)/dt2 = sqrt(a*N1)/(a*dt1) = E[N1]/sqrt(a) 
    #    - standard pipelines return spectral flux density, ie the number of photoelectrons measured over an entire exposure per unit of wavelength (but not per unit of time):
    # + F_meas(t,w) = gcal(band) N_meas(t,w)  
    # + EF_meas(t,w) = E[ F_meas(t,w) ]
    #                = gcal(band) sqrt( N_meas(t,w) )               assuming no other noise sources
    #                = gcal(band) sqrt( F_meas(t,w)/gcal(band) )
    #                = sqrt( gcal(band)*F_meas(t,w))       
    #      where gcal represents corrections of raw measured photoelectrons for the instrumental response (typically the blaze function, ie the transmission of the grating, with gcal = 1/blaze), assumed constant for an instrument (see below) 
    #      if fluxes were converted into temporal densities:
    # + f_meas(t,w) = F_meas(t,w)/dt = gcal(band) N_meas(t,w) / dt
    # + Ef_meas(t,w) = EF_meas(t,w)/dt = gcal(band) sqrt( N_meas(t,w) ) /dt =  sqrt( gcal(band) f_meas(t,w) / dt) 
    #    - our estimate of the true measured flux is F_meas(t,w) = g(band) N_meas(t,w), where N_meas(t,w) follows a Poisson distribution with number of events N_true(t,w)
    #      if enough photons are measured the Poisson distribution on N_meas(t,w) and thus F_meas(t,w) can be approximated by a normal distribution with mean F_true(t,w) and standard deviation EF_true(t,w) 
    #      the issue with using the errors on individual bins to define their weights is that EF_meas(t,w) = sqrt(g(band) F_meas(t,w)) is a biased estimate of the true error
    #      for a bin where F_meas(t,w) < F_true(w) because of statistical variations, we indeed get EF_meas(t,w) < EF_true(w) (and the reverse where F_meas(t,w) > F_true(w)) 
    #      bins with lower flux values than the true flux will have larger weights, resulting in a weighted mean that is underestimated compared to the true flux (mathematically speaking, we do a harmonic mean rather than an arithmetic mean).    
    #      the average flux measured over a large band (such as an order) is closer to the true flux over the band than in the case of a single bin, and thus its error is a better estimate of the true error, so we can assume:
    # EF_meas(t,band) ~ EF_true(t,band)
    #    - for all cases below, bands are defined as orders for e2ds, and as the full spectral range for CCFs and s1d   
    #    - assuming that the flux spectra provided by pipelines are well corrected for this response, we do not want to introduce biases between exposures by using specific calibration files
    #      a given calibration file (eg the blaze) may indeed not include all possible corrections for the instrumental response, and furthermore the flux calibration is not necessarily available for all instruments
    #      we thus use the input flux and error tables to estimate gcal(band,t,v) and derive a profile common to all exposures processed for a given instrument (the goal being to approximate the instrumental response without introducing biases)
    #      at the stage of the workflow where calibration is estimated, fluxes are still in number of photoelectrons, so that:
    #      the flux in a given pixel verifies:
    # F_true(w,t,v) = gcal(w,t,v)*N_true(w,t,v)
    # EF_true(w,t,v) = gcal(w,t,v)*sqrt(N_true(w,t,v))
    #      we estimate gcal over each order for S2D and the full spectral range for S1D
    #      summing the flux densities over a large band yields 
    # TF_meas(t,v) ~ TF_true(t,v) = sum(w, F_true(w,t,v) ) = gcal(band,t,v)*sum(w over band, N_true(w,t,v) )
    #      thus
    # ETF_meas(t,v) ~ E[ TF_true(t,v) ]
    #               = E[ gcal(band,t,v)*sum(w over band, N_true(w,t,v) ) ]
    #               = gcal(band,t,v)*E[ sum(w over band, N_true(w,t,v) ) ]
    #               = gcal(band,t,v)*sqrt( sum(w over band, E[ N_true(w,t,v) )]^2 )
    #               = gcal(band,t,v)*sqrt( sum(w over band, N_true(w,t,v) ) ) 
    #      and thus 
    # gcal(band,t,v) = TF_meas(t,v)*(dt*dw)/sum(w over band, N_true(w,t,v) )
    # gcal(band,t,v) = TF_meas(t,v)*gcal(band,t,v)^2/ETF_meas(t,v)^2
    # gcal(band,t,v) = ETF_meas(t,v)^2 / TF_meas(t,v)
    #      we estimate gcal over each order and all exposures, and define gcal(band) as < t, v :  gcal(band,t,v) > 
    #
    #    - in the presence of red noise associated with the measured counts:
    # ETF_meas(t,v) = gcal(band,t,v)*E[ sum(w over band, N_true(w,t,v) ) ]
    #               = gcal(band,t,v)*sqrt( sum(w over band, E[ N_true(w,t,v) ]^2 + Ered(w)^2 )
    #               = gcal(band,t,v)*sqrt( sum(w over band, N_true(w,t,v) + Ered(w)^2 )    
    #               = gcal(band,t,v)*sqrt( sum(w over band, N_true(w,t,v) ) + TEred(band)^2 )
    #      thus 
    # sum(w over band, N_true(w,t,v) ) = ( ETF_meas(t,v)^2 / gcal(band,t,v)^2 ) - TEred(band)^2
    #      and 
    # gcal(band,t,v) = TF_meas(t,v)/sum(w over band, N_true(w,t,v) )               
    # gcal(band,t,v) = TF_meas(t,v)/(( ETF_meas(t,v)^2 / gcal(band,t,v)^2 ) - TEred(band)^2 )
    # ETF_meas(t,v)^2 / gcal(band,t,v) - gcal(band,t,v)*TEred(band)^2 = TF_meas(t,v)
    # gcal(band,t,v)^2*TEred(band)^2 + TF_meas(t,v)*gcal(band,t,v) - ETF_meas(t,v)^2 = 0
    #      with a = TEred(band)^2
    #           b = TF_meas(t,v)
    #           c = - ETF_meas(t,v)^2
    #           gcal_white(band,t,v) = -c / b
    #      taking the positive solution
    # gcal(band,t,v) = ( -b + sqrt(  b^2 - 4*a*c ) ) / (2*a) 
    #                = ( -TF_meas(t,v) + sqrt(  TF_meas(t,v)^2 + 4*TEred(band)^2*ETF_meas(t,v)^2 ) ) / (2*TEred(band)^2) 
    #                = ( -TF_meas(t,v) + sqrt(  TF_meas(t,v)^2 + 4*TEred(band)^2*gcal_white(band,t,v)*TF_meas(t,v) ) ) / (2*TEred(band)^2) 
    #      so that we could properly estimate the calibration profile in this way, knowing the red noise value
    #      in practive we are still measuring the calibration as :
    # gcal_meas(band,t,v) = ETF_meas(t,v)^2 / TF_meas(t,v)
    #                     = gcal(band,t,v)^2 *( sum(w over band, N_true(w,t,v) ) + TEred(band)^2 ) / TF_meas(t,v)        
    #                     = gcal(band,t,v)^2 *( TF_meas(t,v)/gcal(band,t,v) + TEred(band)^2 ) / TF_meas(t,v)   
    #                     = gcal(band,t,v) + gcal(band,t,v)^2*TEred(band)^2/ TF_meas(t,v)
    #      so that we are overestimating the true calibration profile (especially in region of low flux), and when scaling back to measured counts as N_meas(t,w) = F_meas(t,w)/gcal_meas(band) we are underestimating the true measured counts.
    #    - what matters for the weighing is the change in the precision on the flux over time in a given pixel:
    # + low-frequency variations linked to the overall flux level of the data (eg due to atmospheric diffusion)
    # + high-frequency variations linked to variations in the spectral flux distribution at the specific wavelength of the pixel
    #   for example when averaging the same spectra in their own rest frame, spectral features do not change and there is no need to weigh
    #   however if there are additional features, such as telluric absorption or stellar lines in transmission spectra, then a given pixel can see large flux variations and weighing is required.  
    #    - extreme care must be taken about the rest frame in which the different elements involved in the weighing profiles are defined
    # + low-frequency components are assumed constant over an order and are not aligned / resampled (eg instrumental calibration, global flux scaling, flux balance and light curve scaling)
    # + high-frequency components must have followed the same shifts and resampling as the weighed profile (eg telluric or master stellar spectrum) 
    #    - weights are normalized within the binning function, so that any factor constant over time will not contribute to the weighing
    #    - planetary signatures are not accounted for when binning disk-integrated and intrinsic stellar spectra because they have been excluded from the data or are considered negligible
    #    - weights are necessarily spectral for intrinsic and atmospheric spectra because of the master 
    #    - the weighing due to integration time is accounted for in the binning routine directly, not through errors but through the fraction of an exposure duration that overlaps with the new bin time window 
    weight_spec = np.zeros(dim_exp,dtype=float)*np.nan
    
    #--------------------------------------------------------    
    #Definition of errors on disk-integrated spectra
    #--------------------------------------------------------     
    #    - we calculate the mean of a time-series of disk-integrated spectra while accounting for variations in the noise of the pixels between different exposures
    #    - at the latest processing stage those spectra are defined from rescale_profiles() as:
    # Fsc(w,t,v) = LC_theo(band,t,v)*Fcorr(w,t,v)/(dt*globF(t,v))    
    #      with the corrected spectra linked to the measured spectra as (see spec_corr() function above for the corrections that are included):
    # Fcorr(w,t,v) = F_meas(w,t,v)*Ccorr(w,t,v)  
    #      the measured spectrum is the one read from the input files of the instruments DRS as:
    # F_meas(w,t,v) = gcal(band)*N_meas(w,t,v)
    #      where gcal(band) represents corrections of measured spectral density of photoelectron N_meas for instrumental response (flat field, blaze, etc) assumed to have low-frequency variations
    #            for the purpose of weighing and to avoid biases between the estimated N_meas we assume a constant conversion for all exposures
    #      thus
    # Fsc(w,t,v) = LC_theo(band,t,v)*gcal(band)*N_meas(w,t,v)*Ccorr(w,t,v)/(dt*globF(t,v)) 
    #      or in terms of photoelectons cumulated over the exposure
    # N_sc(w,t,v) = LC_theo(band,t,v)*N_meas(w,t,v)*Ccorr(w,t,v)/globF(t,v) 
    #    - as explained above we want to use as weights the errors on the true measured flux, ie : 
    # EFsc_true(w,t,v) = E( LC_theo(band,t,v)*gcal(band)*N_true(w,t,v)*Ccorr(w,t,v)/(dt*globF(t,v)) )
    #                  = LC_theo(band,t,v)*gcal(band)*Ccorr(w,t,v)*E( N_true(w,t,v) )/(dt*globF(t,v))
    #                  = LC_theo(band,t,v)*gcal(band)*Ccorr(w,t,v)*sqrt( N_true(w,t,v) )/(dt*globF(t,v))
    #      the spectra Fsc_true(w,t,v)/LC_theo(band,t,v) have the same flux density profile as the master of the unocculted star MFstar_true(w,v), so that 
    # gcal(band)*N_true(w,t,v)*Ccorr(w,t,v)/(dt*globF(t,v)) ~ MFstar_true(w,v)
    # N_true(w,t,v) = MFstar_true(w,v)*(dt*globF(t,v))/(gcal(band)*Ccorr(w,t,v))
    #      since the master spectrum of the unocculted star is measured with a high SNR we can assume: 
    # MFstar_true(w,v) ~ MFstar_meas(w,v)
    #      thus
    # sqrt(N_true(w,t,v)) = sqrt( MFstar_meas(w,v)*dt*globF(t,v))/(gcal(band)*Ccorr(w,t,v)) )    
    #      and
    # EFsc_true(w,t,v) = LC_theo(band,t,v)*gcal(band)*Ccorr(w,t,v)*sqrt( MFstar_meas(w,v)*dt*globF(t,v)/(gcal(band)*Ccorr(w,t,v)) )/(dt*globF(t,v))    
    #                  = LC_theo(band,t,v)*sqrt(gcal(band)*Ccorr(w,t,v)* MFstar_meas(w,v)/(dt*globF(t,v)) )    
    #    - final weights are defined as:    
    # weight(w,t,v) = 1/EFsc_true(w,t,v)^2  
    #               = dt*globF(t,v)/(LC_theo(band,t,v)^2*gcal(band)*Ccorr(w,t,v)*MFstar_meas(w,v))      
    #      we note that the spectral normalization of counts by dw is ignored since all profiles are defined over the same spectra table
    #    - in the star rest frame, spectral features from the disk-integrated spectrum remain fixed over time , ie that a given pixel sees no variations in flux from MFstar(w,v) in a given visit and the contribution of the master spectrum can be ignored    
    #      we note that we neglected the differences between the Fsc_true and MFstar profiles that are due to the occulted local stellar lines and planetary atmospheric lines
    #    - even if the calibration remains constant in time and does not contribute to the weighing of disk-integrated spectra, it must always be set to a value consistent with error definitions in the pipeline,
    # so that EFsc_true is comparable with other error tables in the weighing of intrinsic and planetary profiles
    #      calibration profiles are thus always estimated by default by the pipeline, either from the input error tables when available or from the imposed error table for consistency   
    #    - in region of low S/N the noise may not be poisson-dominated anymore. If we consider a constant red noise over time per pixel:
    # EN_true(w,t,v)^2 = N_true(w,t,v) + Ered(w)^2
    #      and thus 
    # EFsc_true(w,t,v)^2 = LC_theo(band,t,v)^2*gcal(band)^2*Ccorr(w,t,v)^2*E( N_true(w,t,v) )^2/(dt*globF(t,v))^2
    #                    = LC_theo(band,t,v)^2*gcal(band)^2*Ccorr(w,t,v)^2*( N_true(w,t,v) + Ered(w)^2 )/(dt*globF(t,v))^2    
    #                    = EFsc_true_white(w,t,v)^2 + ( LC_theo(band,t,v)*gcal(band)*Ccorr(w,t,v)*Ered(w)/(dt*globF(t,v)))^2      
    #       we can consider that the master spectrum has been defined as the weighted sum of n_out exposures, so that :
    # MFstar_meas(t,w) ~ MFstar_true(t,w) = gcal(band,t,v)*sum(n_out, wstar(t_out,w)*Nstar_meas(t_out,w))
    #      and its error is the quadratic sum of errors on each exposure, themselves the quadratic sum of photon-noise and red noise errors: 
    # EMFstar_meas(t,w)^2 ~ EMFstar_true(t,w)^2 = gcal(band)^2*sum(n_out, wstar(t_out,w)^2*( Nstar_meas(t_out,w) + Ered(w)^2 ) )
    #                                           = gcal(band)^2*( sum(n_out, wstar(t_out,w)^2*Nstar_meas(t_out,w)) + sum(n_out, wstar(t_out,w))*Ered(w)^2  )
    #                                           = gcal(band)^2*( sum(n_out, wstar(t_out,w)^2*MFstar_meas(w,v)*(dt_out*globF(t_out,v))/(gcal(band)*Ccorr(w,t_out,v))) + Twstar(w)*Ered(w)^2  ) 
    #                                           = gcal(band)*MFstar_meas(w,v)*sum(n_out, wstar(t_out,w)^2*(dt_out*globF(t_out,v))/Ccorr(w,t_out,v)) + gcal(band)^2*Twstar(w)*Ered(w)^2  
    #                                           = gcal(band)*MFstar_meas(w,v)*Twstar2(w) + gcal(band)^2*Twstar(w)*Ered(w)^2 
    #      so that 
    # Ered(w)^2 = (EMFstar_meas(t,w)^2  -  gcal(band)*MFstar_meas(w,v)*Twstar2(w) )/(gcal(band)^2*Twstar(w))
    #      and 
    # EFsc_true(w,t,v)^2 = EFsc_true_white(w,t,v)^2 + Ered(w)^2*( LC_theo(band,t,v)*gcal(band)*Ccorr(w,t,v)/(dt*globF(t,v)))^2         
    #                    = LC_theo(band,t,v)^2*(gcal(band)*Ccorr(w,t,v)* MFstar_meas(w,v)/(dt*globF(t,v))) + ( LC_theo(band,t,v)*gcal(band)*Ccorr(w,t,v)/(dt*globF(t,v)))^2 *(EMFstar_meas(t,w)^2  -  gcal(band)*MFstar_meas(w,v)*Twstar2(w) )/(gcal(band)^2*Twstar(w))        
    #                    = LC_theo(band,t,v)^2*Ccorr(w,t,v)*[  gcal(band)*MFstar_meas(w,v) + (Ccorr(w,t,v)/(dt*globF(t,v)))*( EMFstar_meas(t,w)^2 - MFstar_meas(t,w)*gcal(band)*Twstar2(w) )/Twstar(w)  ]/(dt*globF(t,v))
    #                    = LC_theo(band,t,v)^2*Ccorr(w,t,v)*[  gcal(band)*MFstar_meas(w,v)    - (Ccorr(w,t,v)/(dt*globF(t,v)))* MFstar_meas(t,w)*gcal(band)*Twstar2(w) /Twstar(w)         + (Ccorr(w,t,v)/(dt*globF(t,v)))*EMFstar_meas(t,w)^2/Twstar(w)  ]/(dt*globF(t,v))
    #                    = LC_theo(band,t,v)^2*Ccorr(w,t,v)*[  MFstar_meas(w,v)*gcal(band)*( 1 - (Ccorr(w,t,v)*Twstar2(w)/(dt*globF(t,v)*Twstar(w)))) + (Ccorr(w,t,v)/(dt*globF(t,v)))*EMFstar_meas(t,w)^2/Twstar(w)  ]/(dt*globF(t,v))
    #                    = LC_theo(band,t,v)^2*Ccorr(w,t,v)*[  MFstar_meas(w,v)*gcal(band)*( Twstar(w)*dt*globF(t,v) - Ccorr(w,t,v)*Twstar2(w)) + Ccorr(w,t,v)*EMFstar_meas(t,w)^2  ]/((dt*globF(t,v))^2*Twstar(w))
    #--------------------------------------------------------     

    #Calculate weights at pixels where the master stellar spectrum is defined and positive
    cond_def_weights = (~np.isnan(flux_ref_exp)) & (flux_ref_exp>ref_val)
    if np.sum(cond_def_weights)==0:stop('Issue with master definition')

    #Spectral corrections
    #    - all corrections that are applied between the input spectra and the spectra processed per visit should be included in this function:
    # > instrumental calibration: uploaded from 'gcal_inputs' for every type of data
    #   for original 2D or 1D spectra, it returns the estimated spectral calibration profiles for each exposure 
    #   for original CCFs or after conversion into CCFs or from 2D/1D it returns a global calibration
    #   if original 2D or 1D spectra were converted back into count-equivalent values and are still in their original format, calibration profiles are rescaled by the mean calibration profile over the visit
    # > tellurics: Ccorr(w,t,v) = 1/T(w,t,v)
    #   with T=1 if not telluric absorption, 0 if maximum absorption
    #   telluric profiles for each exposure are contained in the data upload specific to the exposure, and have been aligned and set to the same rest frame  
    #   they are propagated through 2D/1D conversion but not CCF conversion
    # > flux balance: Ccorr(w,t,v) = ( 1/Pbal(band,t,v) )*( 1/Pbal_ord(band,t,v))
    #   where Pbal and Pbal_ord are the low-frequency polynomial corrections of flux balance variations over the full spectrum or per order
    #   errors on Pbal are neglected and it can be considered a true estimate because it is fitted over a large number of pixels 
    #   the corrections were defined over the spectral in their input rest frame. Given the size of the bands, we neglect spectral shifts and assume the correction can be directly used over any table   
    #   note that SNR(t,order) = N(t,order)/sqrt(N(t,order)) = sqrt(N(t,order))    
    #   since Pbal(band,t,v) is an estimate of F(band,t,v)/MFstar(band,v) times a normalization factor, it is proportional to N(band,t,v) (since the reference is time-independent)
    #   thus Pbal(order,t,v) is proportional to SNR(order,t,v)^2
    #   global normalisation coefficient is not spectral and can be directly applied to any data
    #   global flux balance correction is defined as a function over the full instrument range and can be directly applied to any spectral data, but since it does not change the mean flux it is not propagated through CCF conversion
    #   order flux balance correction is not propagated through 2D/1D conversion or CCF conversion 
    # > cosmics and permanent peaks: ignored in the weighing, as flux values are not scaled but replaced
    # > fringing and wiggles: ignored for now
    if gen_corr_Fbal and ('spec' in data_mode) and corr_Fbal: 
        data_Fbal = dataload_npz(save_data_dir+'Corr_data/Fbal/'+inst+'_'+vis+'_'+str(iexp_glob)+'_add')
        corr_func_glob = data_Fbal['corr_func']
        corr_func_vis = data_Fbal['corr_func_vis']
    else:
        data_Fbal = None
        corr_func_glob=default_func 
        corr_func_vis = None
    if corr_func_vis is not None:corr_func_glob_vis = corr_func_vis
    else:corr_func_glob_vis=default_func 
    if gen_corr_Fbal_ord and ('spec' in data_mode) and (data_mode==gen_type[inst]):
        if data_Fbal is None:data_Fbal = dataload_npz(save_data_dir+'Corr_data/Fbal/'+inst+'_'+vis+'_'+str(iexp_glob)+'_add')
        corr_func_ord = data_Fbal['Ord']['corr_func']
    else:corr_func_ord={iord_orig:default_func for iord_orig in iord_orig_list}
    if (mean_gdet is None):mean_gdet = np.ones(dim_exp,dtype=float)
    if (tell_exp is None):tell_exp = np.ones(dim_exp,dtype=float)
    spec_corr={}
    for iord,iord_orig in enumerate(iord_orig_list):
        if data_mode=='CCF':cen_bins_ord = np.array([1.])
        else:cen_bins_ord = cen_bins[iord,cond_def_weights[iord]]
        nu_bins_ord = c_light/cen_bins_ord[::-1]
        corr_Fbal_glob_ord = (corr_func_glob(nu_bins_ord)*corr_func_glob_vis(nu_bins_ord))[::-1]
        spec_corr[iord] = mean_gdet[iord,cond_def_weights[iord]]/(tell_exp[iord,cond_def_weights[iord]]*corr_func_ord[iord_orig](cen_bins_ord)*corr_Fbal_glob_ord)
     
    #Spectral broadband flux scaling 
    #    - includes broadband contribution, unless overwritten by input 
    #    - flux_sc = 1 with no occultation, 0 with full occultation
    if bdband_flux_sc:     
        data_scaling = dataload_npz(scaled_data_paths+str(iexp_glob))        
        flux_sc = {iord:1. - data_scaling['loc_flux_scaling'](cen_bins[iord,cond_def_weights[iord]]) for iord in range(nord)}
        if (glob_flux_sc is None):glob_flux_sc = data_scaling['glob_flux_scaling']        
    
    #Global flux scaling can still be applied, if provided as input
    else:
        flux_sc = {iord:1. for iord in range(nord)}     
        if (glob_flux_sc is None):glob_flux_sc = 1.

    #Sub-function defining the squared error on scaled disk-integrated profiles
    def calc_EFsc2(flux_sc_ord,flux_ref_ord,spec_corr_ord):
        return (flux_sc_ord**2.)*flux_ref_ord*spec_corr_ord/(dt*glob_flux_sc)

    #Weights on disk-integrated spectra
    if data_type=='DI': 
        for iord in range(nord):
            weight_spec[iord,cond_def_weights[iord]] = 1./calc_EFsc2(flux_sc[iord],flux_ref_exp[iord,cond_def_weights[iord]],spec_corr[iord])  

    else:

        #Function defining the squared error on local stellar profiles
        def calc_EFres2(flux_ref_ord,err_ref_ord2,flux_sc_ord,spec_corr_ord):
            return err_ref_ord2 + calc_EFsc2(flux_sc_ord,flux_ref_ord,spec_corr_ord)

        #--------------------------------------------------------    
        #Definition of errors on residual spectra
        #-------------------------------------------------------- 
        #    - see rescale_profiles(), extract_res_profiles(), the profiles are defined as:
        # Fres(w,t,v) = ( MFstar(w,v) - Fsc(w,t,v) )
        #      where profiles have been scaled to comparable levels and can be seen as (temporal) flux densities
        #    - we want to use as weights the errors on the true residual flux:
        # EFres_true(w,t,v) = E[  MFstar_true(w,v) - Fsc_true(w,t,v) ]
        #                   = sqrt( EMFstar_true(w,v)^2 + EFsc_true(w,t,v)^2 )  
        #      where we assume that the two profiles are independent, and that the error on the master flux averaged over several exposures approximates well the error on the true flux, even within a single bin, so that:          
        # EFres_true(w,t,v) = sqrt( EMFstar_meas(w,v)^2 + EFsc_true(w,t,v)^2 )  
        #    - final weights are defined as:
        # weight(w,t,v) = 1/EFres_true(w,t,v)^2
        #    - we neglect covariance in the uncertainties of the master spectrum
        #    - the binning should be performed in the star rest frame 
        if data_type=='Res': 
            for iord in range(nord):
                weight_spec[iord,cond_def_weights[iord]] = 1. / calc_EFres2(flux_ref_exp[iord,cond_def_weights[iord]],cov_ref_exp[iord][0,cond_def_weights[iord]],flux_sc[iord],spec_corr[iord])

        #--------------------------------------------------------    
        #Definition of errors on intrinsic spectra
        #-------------------------------------------------------- 
        #    - see rescale_profiles(), extract_res_profiles() and proc_intr_data(), the profiles are defined as:
        # Fintr(w,t,v) = Fres(w,t,v)/(1 - LC_theo(band,t))
        #              = ( MFstar(w,v) - Fsc(w,t,v) )/(1 - LC_theo(band,t))            
        #    - we want to use as weights the errors on the true intrinsic flux, ie : 
        # EFintr_true(w,t,v) = EFres_true(w,t,v)/(1 - LC_theo(band,t)) 
        #      we assume that the error on the master flux averaged over several exposures approximates well the error on the true flux, even within a single bin, so that:          
        # EFres_true(w,t,v) = sqrt( EMFstar_meas(w,t,v)^2 + EFsc_true(w,t,v)^2 )  
        #    - final weights are defined as:
        # weight(w,t,v) = 1/EFintr_true(w,t,v)^2             
        #               = (1 - LC_theo(band,t))^2 / EFres_true(w,t,v)^2
        #    - we neglect covariance in the uncertainties of the master spectrum
        #    - intrinsic spectra are extracted in the star rest frame, in which case (Fstar_meas,EFstar_meas) and T must also be in the star rest frame (T must thus have been shifted in the same way as Fsc) 
        #      the binning can be performed in the star rest frame or in the rest frame of the intrinsic stellar lines (ie, in which they are aligned) 
        #      in the latter case, if intrinsic spectra given as input have been shifted (aligned to their local frame or to a given surface rv), then the above components must have been shifted in the same way
        #      for example if the intrinsic line is centered at -rv, then the stellar spectrum is redshifted by +rv when aligning the intrinsic line, so that its blue wing at -rv contributes to the weight at the center of the intrinsic line
        #      the spectral features of the local stellar lines then remain aligned in Fsc over time, but the global stellar line in which they are imprinted shifts, so that a given pixel sees important flux variations
        # weight(w_shifted,t,v) = (1 - LC_theo(band,t))^2 / ( EMFstar_meas(w_shifted,v)^2 + EFsc_true(w_shifted,t,v)^2 )     
        #                       ~ (1 - LC_theo(band,t))^2 / EFsc_true(w_shifted,t,v)^2           
        #                       ~ (1 - LC_theo(band,t))^2 /(LC_theo(band,t)^2*gdet(band,v)*Ccorr(w_shifted,t,v)*MFstar(w_shifted,v)/globF(t,v))  
        #      the weight is inversely proportional to MFstar(w_shifted,v), so that weights in intrinsic line are lower when it is located at high rv in the disk-integrated line, compared to being located at its center
        #      while it may appear counter-intuitive, this is because the disk-integrated stellar line acts only as a background flux (and thus noise) level, so that it brings less noise in its deep core than its wings
        #      another way to see it is as:
        # Fintr_estimated(w,t,v) = (MFstar(w,v) - F(w,t,v))/(1 - LC_theo(band,t))
        #                        = (MFstar(w,v) - (MFstar(w,t,v) - Fintr_meas(w,t,v)))/(1 - LC_theo(band,t))
        #                        = (dFstar(w,t,v) + Fintr_meas(w,t,v)))/(1 - LC_theo(band,t))
        #      if MFstar(w,v) is a good estimate of MFstar(w,t,v) we will retrieve the correct intrinsic profile, but its uncertainties will be affected by the errors on MFstar(w,t,v) - which will dominate the weighing when binning the retrieved 
        # intrinsic profiles in their local rest frame
        elif data_type=='Intr': 
            for iord in range(nord):
                weight_spec[iord,cond_def_weights[iord]] =(1. - flux_sc[iord])**2. / calc_EFres2(flux_ref_exp[iord,cond_def_weights[iord]],cov_ref_exp[iord][0,cond_def_weights[iord]],flux_sc[iord],spec_corr[iord])

        #--------------------------------------------------------        
        #Atmospheric spectra
        #    - in the definition of the weights below we implicitely assume that Fnorm_true(t,w) has the same spectral profile as Fstar_true(t,w)
        #      this is not the case because of the presence of the planetary signature in Fnorm_true(t,w)
        #      however, in the binned atmospheric spectra the planetary signature keeps the same shape and position over time (this is the necessary assumption when binning)
        #      thus differences in errors and weights in a given bin will not come from the planetary signature, but from low-frequency flux variations and from the stellar and telluric lines varying in position
        #      we can thus neglect the presence of the planetary signature
        #--------------------------------------------------------        
                
        #For emission spectra:
        #    - see extract_pl_profiles(), the profiles are defined as:
        # Fem(t,w in band) = Fstar_loc(w,t) - Fres(t,w in band)            
        #      with Fstar_loc null in out-of-transit exposures
        #    - we want to weigh using the true error on the emission flux:    
        # EFem_true(t,w in band)^2 = EFstar_loc_true(w,t)^2 + EFres_true(t,w in band)^2           
        #      EFres_true is defined as above 
        #      local stellar profiles Fstar_loc_true can be defined in different ways (see def_plocc_profiles()). 
        #      we neglect their errors when models are used (mode = 'glob_mod', 'indiv_mod', 'rec_prof') 
        #      otherwise they are defined using the disk-integrated or intrinsic profiles (binned in loc_prof_meas() )
        #      here we assume that enough exposures are used to create these masters that their measured error approximate well enough the true value
        #      EFstar_loc_true(w,t) ~ EFstar_loc_meas(w,t)
        #      final weights are defined as:
        # weight(t,w in band) = 1/( EFstar_loc_meas(w,t)^2 + EFres_true(t,w in band)^2 ) 
        #    - emission spectra are extracted in the star rest frame, in which case EFstar_loc_meas, (Fstar_meas,EFstar_meas) and T involved in Fres must also be in the star rest frame (T must thus have been shifted in the same way as Fnorm)
        #      if emission spectra given as input have been shifted (aligned to their local frame), then the above components  must have been shifted in the same way     
        elif data_type=='Emission': 
            for iord in range(nord):
                weight_spec[iord,cond_def_weights[iord]] = 1./( cov_est_loc_exp[iord][0,cond_def_weights[iord]] + calc_EFres2(flux_ref_exp[iord,cond_def_weights[iord]],cov_ref_exp[iord][0,cond_def_weights[iord]],flux_sc[iord],spec_corr[iord]) ) 
            
        #For absorption spectra: 
        #    - see extract_pl_profiles(), the profiles are defined as:                   
        # Abs(t,w in band) = [ Fres(w,t,vis)/Fstar_loc(w,t,vis)  - 1 ]*( Sthick(band,t)/Sstar )
        #    - we want to weigh using the true error on the absorption signal: 
        # EAbs_true(t,w in band)^2 = E[ Fres_true(t,w)/Fstar_loc_true(t,w) ]^2*( Sthick(band,t)/Sstar )^2
        #                          = ( (Fstar_loc_true(t,w)*EFres_true(t,w))^2 + (Fres_true(t,w)*EFstar_loc_true(t,w))^2 )*( Sthick(band,t)/Sstar )^2
        #      we calculate errors as detailed above for the emission spectra
        # Fres_true(t,w) = Fstar_true(w in band) - LC_theo(t,band)*Fnorm_true(t,w in band)
        #                = Fstar_true(w in band)*(1 - LC_theo(t,band))
        #                ~ Fstar_meas(w in band)*(1 - LC_theo(t,band))       
        #      final weights are defined as:
        # weight(t,w in band) = 1/EAbs_true(t,w in band)^2 
        elif data_type=='Absorption': 
            for iord in range(nord): 
                Floc2_ord = (flux_ref_exp[iord,cond_def_weights[iord]]*(1. - flux_sc[iord]))**2.
                EFloc2_ord = calc_EFres2(flux_ref_exp[iord,cond_def_weights[iord]],cov_ref_exp[iord][0,cond_def_weights[iord]],flux_sc[iord],spec_corr[iord])                
                weight_spec[iord,cond_def_weights[iord]] =  1./( ( flux_est_loc_exp[iord,cond_def_weights[iord]]**2.*EFloc2_ord + Floc2_ord*cov_est_loc_exp[iord][0,cond_def_weights[iord]] )*SpSstar_spec[iord,cond_def_weights[iord]]**2. )               

    return weight_spec




















def calc_bin_prof(idx_to_bin,nord,dim_exp,nspec,data_to_bin_in,inst,n_in_bin,cen_bins_exp,edge_bins_exp,dx_ov_in=None):
    r"""**Spectral profile binning**

    Main routine to bin input spectral profiles, defined over a common spectral table, into a single spectral profile
    
     - propagates covariance matrixes
     - uses spectral weight profiles
     - assumes profiles are independent along the binning dimension

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Clean weights
    #    - in all calls to the routine, exposures contributing to the master are already defined / have been resampled on a common spectral table
    flux_exp_all,cov_exp_all,cond_def_all,glob_weight_all,cond_def_binned = pre_calc_bin_prof(n_in_bin,dim_exp,idx_to_bin,None,dx_ov_in,data_to_bin_in,None,tab_delete=cen_bins_exp)

    #Tables for new exposure
    data_bin={'cen_bins':cen_bins_exp,'edge_bins':edge_bins_exp} 
    data_bin['flux'] = np.zeros(dim_exp,dtype=float)*np.nan
    data_bin['cond_def'] = np.zeros(dim_exp,dtype=bool) 
    data_bin['cov'] = np.zeros(nord,dtype=object)     

    #Calculating new exposures order by order
    for iord in range(nord):
        flux_ord_contr=[]
        cov_ord_contr=[]
        for isub,iexp in enumerate(idx_to_bin):   
 
            #Co-addition to binned profile                       
            #    - binned flux density writes as 
            # fnew(p) = sum(exp i, f(i,p)*w_glob(i,p) )/sum(exp i, w_glob(i,p))
            #      the corresponding error (when there is no covariance) writes as
            # enew(p) = sqrt(sum(exp i,  e(i,p)^2*w_glob(i,p)^2 ) )/sum(exp i, w_glob(i,p))                               
            #      for exposure i, we define the weight on all bins p in a given order as 
            # w_glob(i,p) = weights(i,p)*ov(i)   
            #      where weights are specific to the type of profiles that are binned
            #            ov(i), in units of x, is the fraction covered by exposure i in the new exposure along the bin dimension
            #      the binned flux density can thus be seen as the weighted number of contributing counts from each exposure, f(i,p)*ov(i), divided by the cumulated width of the contributing fractions of bins, sum(ov(i))
            #    - undefined pixels have w(i,p) = 0 so that they do not contribute to the new exposure
            #    - weights approximate the true error on the profiles
            # w_glob(i,p)  = a(p)/e_true(i,p)^2
            #      where a(p) is a factor independent of time that does not contribute to the weighing 
            #      thus we can define here the weight table to be associated to a binned profile as:
            # wnew(p) = 1/enew_true(p)^2
            #         = sum(exp i, w_glob(i,p))^2 /sum(exp i, e_true(i,p)^2*w_glob(i,p)^2 )
            #         = sum(exp i, w_glob(i,p))^2 /sum(exp i, w_glob(i,p) )   
            #         = sum(exp i, w_glob(i,p)) 
            #       where only the variance is considered when defining the spectral weight profiles
            #    - weights account for an original exposure duration (which is independent of the overlap Dxi) through e_true(i,p), ie that the same flux density measured over a longer exposure will weigh more     
            if (True in cond_def_all[isub,iord]):                

                #Weighted profiles
                #    - undefined pixels have null weights and have been set to 0, thus their weighted value is set to 0 and do not contribute to the master
                flux_temp,cov_temp = bind.mul_array(flux_exp_all[isub,iord],cov_exp_all[isub][iord],glob_weight_all[isub,iord])
                flux_ord_contr+=[flux_temp]
                cov_ord_contr+=[cov_temp]
                
                #Defined bins in master
                #    - a bin is defined if at least one bin is defined in any of the contributing exposures
                data_bin['cond_def'][iord] |= cond_def_all[isub,iord]
 
        #Co-addition of weighted profiles from all contributing exposures
        if len(flux_ord_contr)>0:data_bin['flux'][iord],data_bin['cov'][iord] = bind.sum(flux_ord_contr,cov_ord_contr)
        else:data_bin['cov'][iord]=np.zeros([1,nspec])          

        ### End of loop on exposures contributing to master in current order    

    ### End of loop on orders                

    #Set undefined pixels to nan
    #    - a pixel is defined if at least one bin is defined in any of the contributing exposures
    data_bin['flux'][~data_bin['cond_def']]=np.nan

    return data_bin



def pre_calc_bin_prof(n_in_bin,dim_sec,idx_to_bin,resamp_mode,dx_ov_in,data_to_bin,edge_bins_resamp,nocov=False,tab_delete=None):
    r"""**Spectral binning: pre-processing**

    Cleans and normalizes profiles and their weights before binning.

    Args:
        TBD
    
    Returns:
        TBD
    
    """     
    #Pre-processing
    #    - it is necessary to do this operation first to process undefined flux and weight values
    weight_exp_all = np.zeros([n_in_bin]+dim_sec,dtype=float)     
    flux_exp_all = np.zeros([n_in_bin]+dim_sec,dtype=float)*np.nan 
    if not nocov:cov_exp_all = np.zeros(n_in_bin,dtype=object)
    else:cov_exp_all=None
    cond_def_all = np.zeros([n_in_bin]+dim_sec,dtype=bool) 
    cond_undef_weights = np.zeros(dim_sec,dtype=bool)  
    for isub,idx in enumerate(idx_to_bin):
        
        #Resampling
        if resamp_mode is not None:
            if not nocov:flux_exp_all[isub],cov_exp_all[isub] = bind.resampling(edge_bins_resamp, data_to_bin['edge_bins'][isub], data_to_bin['flux'][isub] , cov = data_to_bin['cov'][isub] , kind=resamp_mode)   
            else:flux_exp_all[isub] = bind.resampling(edge_bins_resamp, data_to_bin['edge_bins'][isub], data_to_bin['flux'][isub] , kind=resamp_mode)   
            weight_exp_all[isub] = bind.resampling(edge_bins_resamp, data_to_bin['edge_bins'][isub], data_to_bin['weights'][isub], kind=resamp_mode)  
            cond_def_all[isub] = ~np.isnan(flux_exp_all[isub])            
        else:
            flux_exp_all[isub]= data_to_bin[idx]['flux']
            if not nocov:cov_exp_all[isub]= data_to_bin[idx]['cov']
            weight_exp_all[isub]= data_to_bin[idx]['weight']
            cond_def_all[isub]  = data_to_bin[idx]['cond_def']

        #Set undefined pixels to 0 so that they do not contribute to the binned spectrum
        #    - corresponding weight is set to 0 as well so that it does not mess up the binning if it was undefined
        flux_exp_all[isub,~cond_def_all[isub]] = 0.        
        weight_exp_all[isub,~cond_def_all[isub]] = 0.

        #Pixels where at least one profile has an undefined or negative weight (due to interpolation) for a defined flux value
        cond_undef_weights |= ( (np.isnan(weight_exp_all[isub]) | (weight_exp_all[isub]<0) ) & cond_def_all[isub] )

    #Defined bins in binned spectrum
    #    - a bin is defined if at least one bin is defined in any of the contributing profiles
    cond_def_binned = np.sum(cond_def_all,axis=0)>0  

    #Disable weighing in all binned profiles for pixels validating at least one of these conditions:
    # + 'cond_null_weights' : pixel has null weights at all defined flux values (weight_exp_all is null at undefined flux values, so if its sum is null in a pixel 
    # fulfilling cond_def_binned it implies it is null at all defined flux values for this pixel)
    # + 'cond_undef_weights' : if at least one profile has an undefined weight for a defined flux value, it messes up with the weighted average     
    #    - in both cases we thus set all weights to a common value (arbitrarily set to unity for the pixel), ie no weighing is applied
    #    - pixels with undefined flux values do not matter as their flux has been set to 0, so they can be attributed an arbitrary weight
    cond_null_weights = (np.sum(weight_exp_all,axis=0)==0.) & cond_def_binned
    weight_exp_all[:,cond_undef_weights | cond_null_weights] = 1.

    #Global weight table
    #    - pixels that do not contribute to the binning (eg due to planetary range masking) have null flux and weight values, and thus do not contribute to the total weight
    #    - weight tables only depend on each original exposure but their weight is specific to the new exposures and the original exposures it contains
    dx_ov = np.ones([n_in_bin]+dim_sec,dtype=float) if dx_ov_in is None else dx_ov_in[:,None,None] 
    glob_weight_all = dx_ov*weight_exp_all

    #Total weight per pixel and normalization
    #    - normalization is done along the bin dimension, for each pixel with at least one defined contributing exposure
    glob_weight_tot = np.sum(glob_weight_all,axis=0)
    glob_weight_all[:,cond_def_binned]/=glob_weight_tot[cond_def_binned]

    return flux_exp_all,cov_exp_all,cond_def_all,glob_weight_all,cond_def_binned




 
def resample_func(x_bd_low_in,x_bd_high_in,x_low_in_all,x_high_in_all,flux_in_all,err_in_all,remove_empty=True,dim_bin=0,cond_def_in_all=None,multi=False,adj_xtab=True):
    r"""**General profiles binning**

    Bins input profiles into a new series of profiles, along the chosen dimension
    
     - Propagates variance only
     - Natural weighing by the size of the original bins along the binning dimension
     - Original and final bins can be discontinuous and of different sizes.
       Original bins can overlap and be given as a single input table.
       Input tables must have the same stucture and dimensions, except along the binned dimension .
     - This resampling assumes that the flux density is constant over each pixel, ie that the same number of photons is received by every portion of the pixel.
       In practice this is not the case, as the flux may vary sharply from one pixel to the next, and thus the flux density varies over a pixel (as we would measure at a higher spectral resolution).
       The advantage of the present resampling is that it conserves the flux, which is not necessarily the case for interpolation.
     - nan values can be left in the input tables, so that new pixels that overlap with them will be set conservatively to nan.
       They can also be removed before input to speed up calculation, in which case the new pixel will be defined based on defined, overlapping pixels.
     - If a new pixel is only partially covered by input pixels, its boundaries can be re-adjusted to the maximum overlapping range.
     - The x tables must not contain nan so that they can be sorted.
       The x tables must correspond to the dimension set by dim_bin.
     - Multiprocessing takes longer than the standard loop, even for tens of exposures and the full ESPRESSO spectra.     
     
    Args:
        TBD
    
    Returns:
        TBD
    
    """      
    
    #Input data provided as a single table
    calc_cond_def = True if cond_def_in_all is None else False
    if not False:

        #Shape of input data
        n_all = 1
        dim_in = flux_in_all.shape

        #Artificially creating a single-element list to use the same structure as with multiple input elements
        cond_def_in_all=[cond_def_in_all]
        x_low_in_all = [x_low_in_all]
        x_high_in_all = [x_high_in_all]
        flux_in_all = [flux_in_all]
        err_in_all = [err_in_all]
 
    #Input data provided as separate elements        
    else:
        
        #Shape of input data
        #    - only the dimensions complementary to the binned dimension will be used, and must be common to all input elements
        n_all = len(flux_in_all)
        dim_in = flux_in_all[0].shape

    #Set conditions
    if err_in_all[0] is not None:
        calc_err = True 
    else:
        calc_err = False 
        err_bin_out = None

    #Properties of binned table
    #    - we define the total number of photons in the new table, over all input elements, and the associated error 
    #    - the output binned tables are initialized so that the binned dimension is along the last axis, and the same slices can thus be used for all dimensions within the routine        
    xbin_low_temp = np.tile(x_bd_low_in,[n_all,1])
    xbin_high_temp  = np.tile(x_bd_high_in,[n_all,1])
    dxbin_out=x_bd_high_in-x_bd_low_in
    n_xbins=len(dxbin_out)
    n_dim=len(dim_in)
    if n_dim==1:
        dim_bin=0
        dim_loc_out = [n_xbins]
        ax_trans = None
    elif n_dim==2:
        if dim_bin==0: 
            dim_loc_out = [dim_in[1],n_xbins]
            ax_trans = (1,0)  #transpose from nbin,ny to ny,nbin
            ax_detrans = (1,0)
        elif dim_bin==1:            
            dim_loc_out = [dim_in[0],n_xbins] 
            ax_trans = None
    elif n_dim==3:
        if dim_bin==0:
            dim_loc_out = [dim_in[1],dim_in[2],n_xbins]
            ax_trans = (1,2,0)  #transpose from nbin,ny,nz to ny,nz,nbin
            ax_detrans = (2,0,1)
        else:
            stop('Binning routine to be coded')
    ax_sum = tuple(icomp for icomp in range(n_dim-1))
    dx_ov_cont_tot_bins=np.zeros(n_xbins)
    count_in_bins=np.zeros(dim_loc_out) 
    flux_bin=np.zeros(dim_loc_out)
    flux_bin_out=np.zeros(dim_loc_out)*np.nan 
    if calc_err:
        err2_bin=np.zeros(dim_loc_out) 
        err_bin_out=np.zeros(dim_loc_out)*np.nan 

    #Process each input element separately
    for iloc,(x_low_in_loc,x_high_in_loc,flux_in_loc,err_in_loc,cond_def_in_loc) in enumerate(zip(x_low_in_all,x_high_in_all,flux_in_all,err_in_all,cond_def_in_all)):

        #Transpose input arrays when binned dimension is not along last axis
        if ax_trans is not None:
            flux_in_loc = np.transpose(flux_in_loc,axes=ax_trans)   #from nbin,ny to ny,nbin
            if calc_err==True:err_in_loc = np.transpose(err_in_loc,axes=ax_trans)
            if calc_cond_def == False:cond_def_in_loc = np.transpose(cond_def_in_loc,axes=ax_trans)

        #Sort input data over binned dimension 
        #    - flux_in has been transposed whenever relevant so that the axis to be binned is at the last dimension
        #    - this is necessary for searchsorted to work later on 
        mid_x_in=0.5*(x_low_in_loc+x_high_in_loc) 
        if (True in np.isnan(mid_x_in)):stop('Table along binned dimension must not contain nan')
        id_sort=np.argsort(mid_x_in)
        x_low=x_low_in_loc[id_sort]
        x_high=x_high_in_loc[id_sort]
        flux_in=np.take(flux_in_loc,id_sort,axis=n_dim-1)
        if calc_err:err_in=np.take(err_in_loc,id_sort,axis=n_dim-1) 
        
        #Defined pixels in input table
        #    - we also identify defined pixels along the dimension to be binned
        #      a pixel along this dimension is considered undefined if no pixel along any other dimension is defined
        if calc_cond_def:
            cond_def_in = ~np.isnan(flux_in)  
        else:
            cond_def_in = np.take(cond_def_in_loc,id_sort,axis=n_dim-1)                     
        cond_def_in = np.sum(cond_def_in,axis=ax_sum,dtype=bool)

        #Index of pixels within the boundary of the new table
        #    - we use the first and last defined pixels to keep potentially undefined pixels in between
        #    - beyond those pixels, there are no defined pixels
        cond_sup = x_high[cond_def_in]>=x_bd_low_in[0]
        cond_inf = x_low[cond_def_in]<=x_bd_high_in[-1] 
        if (True in cond_sup) and (True in cond_inf):
            idxcond_def_in = np_where1D(cond_def_in)
            idx_st_withinbin = idxcond_def_in[ np_where1D(cond_sup)[0] ]
            idx_end_withinbin = idxcond_def_in[ np_where1D(cond_inf)[-1] ]
            idx_kept = np.arange(idx_st_withinbin,idx_end_withinbin+1)
            
            #Remove input data beyond boundaries of new table, and beyond the most extreme defined pixels, to speed up calculation  
            #    - flux_in has been transposed whenever relevant so that the axis to be binned is at the last dimension
            x_low=x_low[idx_kept]
            x_high=x_high[idx_kept]
            dx_in=x_high-x_low 
            flux_in=np.take(flux_in,idx_kept,axis=n_dim-1)
            if calc_err==True:err_in2=np.take(err_in,idx_kept,axis=n_dim-1)**2.             

            #Identify bins overlapping with the most extreme defined pixels of the input table
            #    - bins beyond those pixels cannot be defined, and will remain unprocessed/empty
            cond_bin_proc = (x_bd_high_in>=x_low[0]) & (x_bd_low_in<=x_high[-1])
            idx_bin_proc = np.arange(n_xbins)[cond_bin_proc]
            x_bd_low_proc = x_bd_low_in[cond_bin_proc]
            x_bd_high_proc = x_bd_high_in[cond_bin_proc]
       
            #--------------------------------------------------------------------------------------------------------

            #Process new bins
            for isub_bin,(ibin,xbin_low_loc,xbin_high_loc) in enumerate(zip(idx_bin_proc,x_bd_low_proc,x_bd_high_proc)):   
           
                #Indexes of all original pixels overlapping with current bin
                #    - flux_in has been transposed whenever relevant so that the axis to be binned is at the last dimension
                #    - we use 'where' rather than searchsorted to allow processing original bins that overlap together
                idx_overpix = np_where1D( (x_high>=xbin_low_loc) &  (x_low <=xbin_high_loc) )
                
                #Process bins if original pixels overlap
                n_overpix = len(idx_overpix)
                if n_overpix>0:
                    flux_overpix=np.take(flux_in,idx_overpix,axis=n_dim-1)
                    if calc_err==True:err_overpix2=np.take(err_in2,idx_overpix,axis=n_dim-1)
     
                    #Checking that overlapping pixels are all defined
                    #    - the condition is that pixels in the dimensions complementary to that binned, for which all overlapping pixels (along the binned dimension) are defined 
                    #    - if at least one pixel is undefined, we consider conservatively that the new pixel it overlap is undefined as well
                    #      new bins have been initialized to nan, and thus remain undefined unless processed
                    #    - for n_dim>1, the new bin is processed if there is at least one position along the complementary dimensions for which all overlapping pixels are defined
                    #    - this condition is applied for one of the input element, but if at least one input element has defined pixels overlapping with the new pixel, it will be defined
                    #    - the binned axis has been placed as last dimension, and cond_defbin_compdim has thus the dimensions of flux_overpix along the other axis (kept in the original order) 
                    cond_defbin_compdim = (np.prod(~np.isnan(flux_overpix),axis=n_dim-1)==1) 
    
                    #Process if original overlapping pixels are defined
                    #    - cond_defbin_compdim is a single boolean for ndim=1, an array otherwise, so that "if True in cond_defbin_compdim" cannot be used
                    #    - each pixel overlapping with the bin can either be:
                    # + fully within the bin (both pixels boundaries are within the bin)
                    # + fully containing the bin (both pixels boundaries are outside of the bin)
                    # + containing one of the two bin boundaries (the upper - resp lower - pixel boundary is outside the bin)                
                    if np.sum(cond_defbin_compdim)>0:    
    
                        #Minimum between overlapping pixels upper boundaries and new bin upper boundary
                        #    - if the pixel upper boundary is beyond the bin, then the pixel fraction beyond the bin upper boundary will not contribute to the binned flux 
                        #    - if the pixel upper boundary is within the bin, then the pixel will only contribute to the binned flux up to its own boundary
                        x_high_ov_contr=np.minimum(x_high[idx_overpix],np.repeat(xbin_high_loc,n_overpix))
                    
                        #Maximum between overlapping pixels lower boundaries and new bin lower boundary
                        #    - if the pixel lower boundary is beyond the bin, then the pixel fraction beyond the bin upper boundary will not contribute to the binned flux 
                        #    - if the pixel lower boundary is within the bin, then the pixel will only contribute to the binned flux up to its own boundary
                        x_low_ov_contr=np.maximum(x_low[idx_overpix],np.repeat(xbin_low_loc,n_overpix))
                    
                        #Width over which each original pixel contribute to the binned flux
                        dx_ov_cont=x_high_ov_contr-x_low_ov_contr
                        
                        #Co-add total effective overlap of original pixels to current bin
                        #    - overlap are cumulated over successive input elements 
                        #    - this can be done because original pixels do not overlap
                        #    - the effective overlaps of successive input elements are averaged afterwards via count_in_bins
                        dx_ov_cont_tot_bins[ibin]+=np.sum(dx_ov_cont)
                    
                        #Store local bin boundaries
                        #    - if all overlapping pixels have their upper (resp lower) boundary lower (resp higher) than that of the bin, then the bin 
                        # is restricted to the maximum (resp minimum) pixels boundary
                        #    - because this redefinition may vary for the different input elements, we store the updated boundaries for each element and make a final definition afterwards                         
                        xbin_low_temp[iloc,ibin] = min(x_low_ov_contr)
                        xbin_high_temp[iloc,ibin] = max(x_high_ov_contr)
    
                        #Slices suited to all dimensions
                        #    - cond_defbin_compdim equals True for ndim=1, and will not count in the call to the various tables
                        idx_mdim = (cond_defbin_compdim,ibin)
    
                        #Implement counter
                        #    - we set to 1 the new pixels along the complementary dimensions for which all original overlapping pixels are defined
                        #      only those pixels contribute to the new bins, but which pixels contribute can change with the input element
                        #      the counter is thus used to average values of the new bins, along all complementary dimensions, over the input elements that contributed to each dimension
                        count_in_bins[idx_mdim]+=1
    
                        #Add contribution of original overlapping pixels to new pixel
                        #    - flux (=spectral photon density) is assumed to be constant over a given pixel:
                        # F(pix) = gcal* N(pix) / dpix = N(dx)/dx
                        #      where gcal is the instrumental calibration of photoelectrons
                        #    - the flux density in a bin is the total number of photons from the overlapping fractions of all pixels, divided by the effective overlapping surface (see below)
                        # F(bin) = gcal*sum( overlapping pixels, N(pixel overlap) )/sum( overlapping pixels, dx(pixel overlap) )
                        #       since F(pixel) = gcal*N(pixel overlap)/dx(pixel overlap) we obtain
                        # F(bin) = sum( overlapping pixels, F(pixel)*dx(pixel overlap) )/sum( overlapping pixels, dx(pixel overlap) )
                        #    - because there can be gaps between two original pixels, we average their contributed number of photons by the effective width they cover
                        #      imagine a new pixel between x=0 and x=1, covered by two original pixels extending up to x=1/3 for the first one, and starting from x=2/3 for the second one
                        #      both original pixels have a flux of 1, and thus contribute 1/3 photons over their overlapping range
                        #      normalizing by the width of the new pixel would yield an incorrect flux of (1/3+1/3)/1 = 2/3
                        #      we thus normalize by the effective overlap = 2/3 
                        #    - for ndim = 1, flux_overpix[cond_defbin_compdim] = flux_overpix[True] returns an array (1,len(cond_defbin_compdim))
                        #      its product with dx_ov_cont must thus be summed over axis=1 
                        flux_bin[idx_mdim]+=np.sum(flux_overpix[cond_defbin_compdim]*dx_ov_cont,axis=1)
                        if calc_err: err2_bin[idx_mdim]+=np.sum(err_overpix2[cond_defbin_compdim]*dx_ov_cont*dx_in[idx_overpix],axis=1)

    #----------------------------------------------------------------------------      

    #Redefinition of the bin boundaries
    #    - the upper (resp. lower) boundaries are set to the maximum (resp. min) of the boundaries defined for each input element
    if adj_xtab==True:
        xbin_low_out = np.amin(xbin_low_temp,axis=0) 
        xbin_high_out = np.amax(xbin_high_temp,axis=0) 
    else:
        xbin_low_out = x_bd_low_in
        xbin_high_out= x_bd_high_in

    #Average flux over all contributing pixels and input element
    #    - we define new bins only if they are filled in at least one contributing input element
    #      count_in_bins is summed along all dimensions complementary to the binned dimension
    #    - the number of photons in a given bin, obtained by co-adding the number of photons from the overlapping pixels of each successive input element, is 
    # converted into a flux density by normalizing with the total width of overlap from these contributing pixels
    #    - empty bins remain set to nan
    #    - the binned table have dimensions fbin = (n0,n1 .. nn)
    #      the overlap table dx_ov has dimension ni along one of these axes
    #      there is no need to create a new axis to perform fbin/dx_ov if the size of dx_ov is that of the last dimension in fbin
    #    - the 'axes = (i,j,k,..)' field in transpose() means that axe i -> axe 0, axe j -> axe 1  
    cond_filled=(np.sum(count_in_bins,axis=ax_sum)>0.)
    if (True in cond_filled):
        if (n_dim==1):
            flux_bin_out[cond_filled]=flux_bin[cond_filled]/dx_ov_cont_tot_bins[cond_filled]      
            if calc_err==True:err_bin_out[cond_filled]=np.sqrt(err2_bin[cond_filled])/dx_ov_cont_tot_bins[cond_filled]
        else:
     
            #Normalization of defined pixels
            #    - we apply the operation to defined pixels only along the binned dimension (placed along last axis) and to all pixels in complementary dimensions (called with slice())
            ax_slice = tuple(slice(icomp) for icomp in np.array(dim_loc_out)[0:-1])+(cond_filled,)
            flux_bin_out[ax_slice]=flux_bin[ax_slice]/dx_ov_cont_tot_bins[cond_filled]
            if calc_err==True:err_bin_out[ax_slice]=np.sqrt(err2_bin[ax_slice])/dx_ov_cont_tot_bins[cond_filled]                    
                    
            #Transpose back arrays to original dimensions of input arrays, if relevant 
            #    - binned table currently have the binned axis as last dimension
            if ax_trans is not None:
                flux_bin_out = np.transpose(flux_bin_out,axes=ax_detrans)        
                if calc_err==True:err_bin_out = np.transpose(err_bin_out,axes=ax_detrans)

    #Remove empty bins if requested
    #      bins are removed only if undefined along all complementary dimensions
    if (remove_empty==True) and (False in cond_filled):   
        xbin_low_out=xbin_low_out[cond_filled]
        xbin_high_out=xbin_high_out[cond_filled] 
        idx_cond_filled = np_where1D(cond_filled)               
        flux_bin_out=np.take(flux_bin_out,idx_cond_filled,axis=dim_bin)
        if calc_err==True:err_bin_out=np.take(err_bin_out,idx_cond_filled,axis=dim_bin)            
    xbin_out=0.5*(xbin_low_out+xbin_high_out) 
    dxbin_out = xbin_high_out - xbin_low_out

    return xbin_low_out,xbin_high_out,xbin_out,dxbin_out,flux_bin_out,err_bin_out  






def sub_calc_bins(low_bin,high_bin,raw_loc_dic,nfilled_bins,calc_Fr=False,calc_gdet=False,adjust_bins=True):
    r"""**Simplified binning routine**

    Used instead of the resampling function when only the flux is binned and/or the bins are large enough that we can neglect the covariance between them.
    This is also why we can bin the master and exposure over defined pixels (ie, ignoring some of the pixels that might be undefined within a bin).
    Otherwise the covariance matrix would need to be resampled over all consecutive pixels, included undefined ones, and the binned pixels would be set to undefined by the resampling function.
        
    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    bin_loc_dic = {}

    #Indexes of all original bins overlapping with current bin
    #    - searchsorted cannot be used in case bins are not continuous
    #    - the approach below remains faster than tiling a matrix with the original tables to apply in one go the search and sum operations
    idx_overpix = np_where1D( (raw_loc_dic['high_bins']>=low_bin) &  (raw_loc_dic['low_bins'] <=high_bin) )
    if len(idx_overpix)>0:
      
        #Total exposure flux over the selected bins
        Fexp_tot = np.sum(raw_loc_dic['flux'][idx_overpix]*raw_loc_dic['dbins'][idx_overpix] )
        if calc_Fr or (calc_gdet and (Fexp_tot>0.)):                             
            nfilled_bins+=1.
            
            #Ratio binned exposure flux / binned master flux
            if calc_Fr:
                bin_loc_dic['Fmast_tot'] = np.sum(raw_loc_dic['mast_flux'][idx_overpix]*raw_loc_dic['dbins'][idx_overpix] )            
                bin_loc_dic['Fr'] = Fexp_tot/bin_loc_dic['Fmast_tot']
                if 'var' in raw_loc_dic:bin_loc_dic['varFr'] = np.sum(raw_loc_dic['var'][idx_overpix]*raw_loc_dic['dbins'][idx_overpix]**2.)/bin_loc_dic['Fmast_tot']**2.

            #Ratio binned exposure error squared / binned exposure flux
            #    - see weights_bin_prof(): 
            # gdet(band,v) = sum( EF_meas(w,t,v)^2 )  ) / sum( F_meas(w,t,v) )
            if calc_gdet and (Fexp_tot>0.):
                bin_loc_dic['gdet'] = np.sum(raw_loc_dic['var'][idx_overpix]) /np.sum(raw_loc_dic['flux'][idx_overpix])

            #Adjust bin center and boundaries
            if adjust_bins:
                bin_loc_dic['cen_bins'] = np.mean(raw_loc_dic['cen_bins'][idx_overpix])
                bin_loc_dic['low_bins'] = raw_loc_dic['low_bins'][idx_overpix[0]]
                bin_loc_dic['high_bins'] = raw_loc_dic['high_bins'][idx_overpix[-1]]
            else:
                bin_loc_dic['cen_bins'] = 0.5*(low_bin+high_bin)
                bin_loc_dic['low_bins'] = low_bin
                bin_loc_dic['high_bins'] = high_bin
                
    return bin_loc_dic,nfilled_bins


def sub_def_bins(bin_siz,idx_kept_ord,low_pix,high_pix,dpix_loc,pix_loc,sp1D_loc,Mstar_loc=None,var1D_loc=None):
    r"""**Bins definition**

    Defines new bins from input bins

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    raw_loc_dic = {    
        'low_bins': low_pix[idx_kept_ord],
        'high_bins': high_pix[idx_kept_ord],
        'dbins': dpix_loc[idx_kept_ord],
        'cen_bins': pix_loc[idx_kept_ord],
        'flux': sp1D_loc[idx_kept_ord]}
    if Mstar_loc is not None:raw_loc_dic['mast_flux'] = Mstar_loc[idx_kept_ord]  
    if var1D_loc is not None:raw_loc_dic['var'] = var1D_loc[idx_kept_ord]    

    #Defining bins at the requested resolution over the range of original defined bins
    min_pix = np.nanmin(raw_loc_dic['low_bins'])
    max_pix = np.nanmax(raw_loc_dic['high_bins'])
    n_bins_init=int(np.ceil((max_pix-min_pix)/bin_siz))
    bin_siz=(max_pix-min_pix)/n_bins_init
    bin_bd=min_pix+bin_siz*np.arange(n_bins_init+1,dtype=float)                      

    return bin_bd,raw_loc_dic






