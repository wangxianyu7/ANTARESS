#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import griddata
from copy import deepcopy
import bindensity as bind
import glob
from ..ANTARESS_conversions.ANTARESS_binning import calc_bin_prof,weights_bin_prof,init_bin_prof
from ..ANTARESS_grids.ANTARESS_prof_grid import init_custom_DI_prof,theo_intr2loc
from ..ANTARESS_grids.ANTARESS_occ_grid import init_surf_shift,def_surf_shift,sub_calc_plocc_spot_prop,up_plocc_prop
from ..ANTARESS_grids.ANTARESS_coord import excl_plrange
from ..ANTARESS_process.ANTARESS_data_align import align_data
from ..ANTARESS_analysis.ANTARESS_inst_resp import def_st_prof_tab,cond_conv_st_prof_tab,get_FWHM_inst,resamp_st_prof_tab,conv_st_prof_tab
from ..ANTARESS_general.utils import stop,dataload_npz,datasave_npz,closest_arr,np_where1D,gen_specdopshift,check_data

#%% Planet-occulted exposure profiles.

def def_in_plocc_profiles(inst,vis,gen_dic,data_dic,data_prop,coord_dic,system_param,theo_dic,glob_fit_dic,plot_dic):
    r"""**Planet-occulted exposure profiles.**
    
    Calls requested function to define planet-occulted profiles associated with each observed exposure
    
     - local profiles are used to correct residual profiles from stellar contamination
     - intrinsic profiles are used to assess fit quality

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    print('   > Building estimates for planet-occulted stellar profiles')
    opt_dic = data_dic['Intr']['opt_loc_data_corr'] 
    corr_mode = opt_dic['corr_mode']
    text_print={
        'DIbin':    '         Using DI master',
        'Intrbin':  '         Using binned intrinsic profiles',
        'glob_mod': '         Using global model',
        'indiv_mod':'         Using individual models',
        'rec_prof': '         Using reconstruction',
        }[corr_mode]
    print(text_print)  

    #Calculating
    if (gen_dic['calc_loc_data_corr']):
        print('         Calculating data')     
 
        #Using disk-integrated master or binned intrinsic profiles
        if corr_mode in ['DIbin','Intrbin']:
            data_add = plocc_prof_meas(opt_dic,corr_mode,inst,vis,data_dic['Intr'],gen_dic,data_dic,data_prop,data_dic['Atm'],coord_dic)
            
        #Using global profile model
        elif corr_mode=='glob_mod': 
            data_add = plocc_spocc_prof_globmod(opt_dic,corr_mode,inst,vis,gen_dic,data_dic,data_prop,system_param,theo_dic,coord_dic,glob_fit_dic,False)
            
        #Using individual profile models
        elif corr_mode=='indiv_mod': 
            data_add = plocc_prof_indivmod(opt_dic,corr_mode,inst,vis,gen_dic,data_dic)

        #Defining undefined pixels via a polynomial fit to defined pixels in complementary exposures, or via a 2D interpolation over complementary exposures and a narrow spectral band
        elif corr_mode=='rec_prof': 
            data_add = plocc_prof_rec(opt_dic,corr_mode,inst,vis,gen_dic,data_dic,coord_dic)
            
        #Saving complementary data for plots
        if plot_dic['map_Intr_prof_res']!='':
            data_add['loc_data_corr_inpath'] = data_dic[inst][vis]['proc_Intr_data_paths']
            data_add['loc_data_corr_outpath'] = data_dic[inst][vis]['proc_Res_data_paths']
            data_add['rest_frame'] = data_dic['Intr'][inst][vis]['rest_frame']
        datasave_npz(gen_dic['save_data_dir']+'Loc_estimates/'+corr_mode+'/'+inst+'_'+vis+'_add',data_add)
            
    #Checking that local data has been calculated for all exposures
    else:
        idx_est_loc = dataload_npz(gen_dic['save_data_dir']+'Loc_estimates/'+corr_mode+'/'+inst+'_'+vis+'_add')['idx_est_loc']
        data_paths={i_in:gen_dic['save_data_dir']+'Loc_estimates/'+corr_mode+'/'+inst+'_'+vis+'_'+str(i_in) for i_in in idx_est_loc}
        check_data(data_paths)

    return None



def plocc_prof_meas(opt_dic,corr_mode,inst,vis,gen_dic,data_dic,data_prop,coord_dic):
    r"""**Planet-occulted exposure profiles: measured**
    
    Sub-function to define planet-occulted profiles using measured profiles as best estimates of intrinsic profiles
    
    When binned intrinsic profiles are used
    
     - we first define the range and center of the binned profiles, along the bin dimension
     - for each in-transit local profile, we find the nearest binned profile along the bin dimension (via their centers)
     - we use intrinsic profiles that have already been aligned to a null rest frame, first shifting them to the rv of the target local profile, and then binning them
     - we could use the original intrinsic profiles and shift them by their own rv minus that of the local profile, however in this way we can use directly the outputs of `align_Intr()`, and 
       shifting the profiles in two steps is not an issue when they are not resampled until binned
     - we finally bin the aligned profiles, and scale the binned profile to the level of the local one 

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    data_inst = data_dic[inst]
    data_vis = data_inst[vis]
    
    #Initialize surface RV
    ref_pl,dic_rv,idx_aligned = init_surf_shift(gen_dic,inst,vis,data_dic,data_dic['Intr']['align_mode'])  

    #Binning mode
    #   - using current visit exposures only, or exposures from multiple visits
    if (inst in opt_dic['vis_in_bin']) and (len(opt_dic['vis_in_bin'][inst])>0):vis_to_bin = opt_dic['vis_in_bin'][inst]   
    else:vis_to_bin = [vis]           
    
    #Initialize binning
    #    - when disk-integrated profiles are used, the output tables contain a single value, associated with the single master (=binned profile) used for the extraction
    if corr_mode=='DIbin':
        in_type='DI'
        dim_bin = 'phase'
    elif corr_mode=='Intrbin':
        in_type='Intr'
        dim_bin = opt_dic['dim_bin']
        if not gen_dic['align_Intr']:stop('Intrinsic profiles must have been aligned')
    new_x_cen,_,_,x_cen_all,n_in_bin_all,idx_to_bin_all,dx_ov_all,_,idx_bin2orig,idx_bin2vis,idx_to_bin_unik = init_bin_prof(in_type,opt_dic[inst][vis],opt_dic['idx_in_bin'],dim_bin,coord_dic,inst,vis_to_bin,data_dic,gen_dic)

    #Find binned profile closest (along bin dimension ) to each processed in-transit exposure 
    if corr_mode=='Intrbin':idx_bin_closest = closest_arr(new_x_cen, x_cen_all[idx_aligned])
    
    #Processing in-transit exposures for which planet-occulted rv is known
    for isub,i_in in enumerate(idx_aligned):    
    
        #Upload spectral tables from residual or intrinsic profile of current exposure
        if data_dic['Intr']['plocc_prof_type']=='Intr':iexp_eff = i_in
        elif data_dic['Intr']['plocc_prof_type']=='Res':iexp_eff = gen_dic[inst][vis]['idx_in2exp'][i_in]
        data_loc_exp = dataload_npz(data_vis['proc_'+data_dic['Intr']['plocc_prof_type']+'_data_paths']+str(iexp_eff))

        #Index of binned profile associated with current processed exposure
        if corr_mode=='DIbin':i_bin = 0
        elif corr_mode=='Intrbin':i_bin = idx_bin_closest[isub]
                
        #Calculating binned profile associated with current processed exposure
        #    - since the shift is specific to each processed exposure, contributing profiles must be aligned and resampled for each one, either on the table common to all profiles, or on the table of current exposure, before being binned together        
        data_to_bin={}
        for iexp_off in idx_to_bin_all[i_bin]:    
        
            #Original index and visit of contributing exposure
            iexp_bin = idx_bin2orig[iexp_off]
            vis_bin = idx_bin2vis[iexp_off]
            
            #Upload latest processed disk-integrated data         
            if corr_mode=='DIbin':
                iexp_bin_glob = iexp_bin
                data_exp_bin = np.load(data_inst[vis_bin]['proc_DI_data_paths']+str(iexp_bin)+'.npz',allow_pickle=True)['data'].item() 
                if gen_dic['flux_sc']: scaled_data_paths = data_dic[inst][vis_bin]['scaled_DI_data_paths']  
                else:scaled_data_paths=None
                
                #Exclude planet-contaminated bins 
                #    - here we set to nan the flux (rather than just the defined bins) when profiles are still aligned in the star rest frame, to avoid having to shift for every exposure the planet-excluded ranges
                #      after profiles are aligned to the local surface velocity, the defined pixels will account for the exclusion and be used to define the weights in the bin routine
                if ('DI_Mast' in data_dic['Atm']['no_plrange']) and (iexp_bin in data_dic['Atm'][inst][vis_bin]['iexp_no_plrange']):
                    for iord in range(data_inst['nord']):
                        data_exp_bin['flux'][iord][excl_plrange(data_exp_bin['cond_def'][iord],data_dic['Atm'][inst][vis_bin]['exclu_range_star'],iexp_bin,data_loc_exp['edge_bins'][iord],data_vis['type'])[0]] = np.nan
                if data_dic[inst][vis_bin]['tell_sp']:data_exp_bin['tell'] = dataload_npz(data_dic[inst][vis_bin]['tell_DI_data_paths'][iexp_bin])['tell']             
                if data_dic[inst][vis_bin]['mean_gdet']:data_exp_bin['mean_gdet'] = dataload_npz(data_dic[inst][vis_bin]['mean_gdet_DI_data_paths'][iexp_bin])['mean_gdet']             
                data_ref = dataload_npz(data_dic[inst][vis_bin]['mast_DI_data_paths'][iexp_bin])
                
                
            #Upload intrinsic stellar profiles aligned in their local frame
            #    - we use intrinsic profiles already aligned in a null rest frame (still defined on their original tables unless a common table was used)     
            #    - profiles could be retrieved from their original extraction rest frame, and shifted directly to the surface rv associated with current exposure
            #      however in the ideal processing profiles were maintained on their individual tables, which were shifted without resampling to create the aligned profiles, thus doing an additional shift here
            # will not create any correlations 
            elif corr_mode=='Intrbin':
                iexp_bin_glob = gen_dic[inst][vis_bin]['idx_in2exp'][iexp_bin]
                if iexp_bin_glob not in data_dic['Intr'][inst][vis]['idx_def']:stop('Intrinsic exposure at i=',str(iexp_bin),' has not been aligned')
                data_exp_bin = dataload_npz(gen_dic['save_data_dir']+'Aligned_Intr_data/'+inst+'_'+vis_bin+'_'+str(iexp_bin))
                if data_dic[inst][vis_bin]['tell_sp']:data_exp_bin['tell'] = dataload_npz(gen_dic['save_data_dir']+'Aligned_Intr_data/'+inst+'_'+vis_bin+'_in_tell'+str(iexp_bin))['tell']             
                if data_dic[inst][vis_bin]['mean_gdet']:data_exp_bin['mean_gdet'] = dataload_npz(gen_dic['save_data_dir']+'Aligned_Intr_data/'+inst+'_'+vis_bin+'_in_mean_gdet'+str(iexp_bin))['mean_gdet']             
                data_ref = np.load(gen_dic['save_data_dir']+'Aligned_Intr_data/'+inst+'_'+vis_bin+'_in_ref'+str(iexp_bin)+'.npz',allow_pickle=True)['data'].item() 
                if gen_dic['flux_sc']: scaled_data_paths = data_dic[inst][vis_bin]['scaled_Intr_data_paths']  
                else:scaled_data_paths=None
                
            #Radial velocity shifts set to the opposite of the planet-occulted surface rv associated with current exposure
            rv_surf_star,rv_surf_star_edge = def_surf_shift(data_dic['Intr']['align_mode'],dic_rv,i_in,data_exp_bin,ref_pl,data_vis['type'],data_dic[inst]['system_prop'],data_dic[inst][vis_bin]['dim_exp'],data_dic[inst]['nord'],data_dic[inst][vis_bin]['nspec'])
            rv_shift_cen = -rv_surf_star
            spec_dopshift_cen = gen_specdopshift(rv_surf_star)
            if rv_surf_star_edge is not None:
                rv_shift_edge = -rv_surf_star_edge
                spec_dopshift_edge = gen_specdopshift(rv_surf_star_edge)
            else:
                rv_shift_edge = None
                spec_dopshift_edge = None            
                    
            #Aligning contributing profile at current exposure stellar surface rv 
            #    - aligned profiles are here resampled on the table of current exposure, which is common to all exposures if a common table is used        
            #    - complementary tables follow the same shifts
            data_to_bin[iexp_off]=align_data(data_exp_bin,data_vis['type'],data_dic[inst]['nord'],data_dic[inst][vis]['dim_exp'],gen_dic['resamp_mode'],data_loc_exp['cen_bins'],data_loc_exp['edge_bins'],rv_shift_cen,spec_dopshift_cen,spec_dopshift_edge = spec_dopshift_edge,rv_shift_edge = rv_shift_edge)
                
            #Shifting weighing master at current exposure stellar surface rv 
            #    - master will be resampled on the same table as current exposure
            data_ref_align=align_data(data_ref,data_vis['type'],data_dic[inst]['nord'],data_dic[inst][vis]['dim_exp'],gen_dic['resamp_mode'],data_loc_exp['cen_bins'], data_loc_exp['edge_bins'],rv_shift_cen,spec_dopshift_cen,spec_dopshift_edge = spec_dopshift_edge,rv_shift_edge = rv_shift_edge)

            #Weight profile
            data_to_bin[iexp_off]['weight'] = weights_bin_prof(range(data_inst['nord']),scaled_data_paths,inst,vis_bin,gen_dic['corr_Fbal'],gen_dic['corr_FbalOrd'],gen_dic['save_data_dir'],gen_dic['type'],data_inst['nord'],iexp_bin_glob,in_type,data_vis['type'],data_vis['dim_exp'],data_to_bin[iexp_off]['tell'],data_to_bin[iexp_off]['mean_gdet'],data_to_bin[iexp_off]['cen_bins'],coord_dic[inst][vis_bin]['t_dur'][iexp_off],data_ref_align['flux'],data_ref_align['cov'],bdband_flux_sc = gen_dic['flux_sc'])            
            
        #Calculating binned profile
        data_est_loc = calc_bin_prof(idx_to_bin_all[i_bin],data_dic[inst]['nord'],data_vis['dim_exp'],data_vis['nspec'],data_to_bin,inst,n_in_bin_all[i_bin],data_loc_exp['cen_bins'],data_loc_exp['edge_bins'],dx_ov_in = dx_ov_all[i_bin])
        
        #Rescaling measured intrinsic profile to the level of the local profile
        #    - this operation assumes that all exposures used to compute the master-out have not been rescaled with respect to the reference, ie that they all have the same flux balance as current exposure before it was rescaled
        #    - see rescale_profiles() and proc_intr_data() for more details
        #      a given local profile write as 
        #      F_res(w,t,vis) = MFstar(w,vis) - Fsc(w,vis,t)
        #   at low resolution, in the continuum (see rescale_profiles())
        #      Fsc(w,vis,t) = LC(w,t)*Fstar(w,vis_norm)*Cref(w)
        #   assuming that the rescaling light curves are constant outside of the transit, all out-of-transit profiles are equivalent between themselves and thus to the master-out
        #      MFstar(w,vis) ~ Fstar(w,vis_norm)*Cref(w)
        #   thus
        #      F_res(w,t,vis) = Fstar(w,vis_norm)*Cref(w) - LC(w,t)*Fstar(w,vis_norm)*Cref(w)
        #                     = Fstar(w,vis_norm)*Cref(w)*(1 - LC(w,t))
        #                     = MFstar(w,vis)*(1 - LC(w,t))           
        #    - if intrinsic profiles are requested no scaling is applied, since F_res(w,t,vis) = F_intr(w,t,vis)*(1 - LC(w,t))      
        #    - the scaling spectrum is defined at all pixels, and thus does not affect undefined pixels in the master (the covariance matrix cannot be sliced)
        if (data_dic['Intr']['plocc_prof_type']=='Res') and gen_dic['flux_sc']:
            data_scaling = dataload_npz(data_vis['scaled_Intr_data_paths']+str(gen_dic[inst][vis]['idx_in2exp'][i_in]))
            for iord in range(data_inst['nord']):
                loc_flux_scaling_ord = data_scaling['loc_flux_scaling'](data_est_loc['cen_bins'][iord])              
                data_est_loc['flux'][iord],data_est_loc['cov'][iord] = bind.mul_array(data_est_loc['flux'][iord],data_est_loc['cov'][iord],loc_flux_scaling_ord)
            
        #Saving estimate of local profile for current exposure                   
        np.savez_compressed(gen_dic['save_data_dir']+'Loc_estimates/'+corr_mode+'/'+inst+'_'+vis+'_'+str(i_in),data=data_est_loc,allow_pickle=True)

    #Complementary data
    data_add={'idx_est_loc':idx_aligned}
    if corr_mode=='Intrbin':data_add['idx_bin_closest']=idx_bin_closest

    return data_add





def plocc_spocc_prof_globmod(opt_dic,corr_mode,inst,vis,gen_dic,data_dic,data_prop,system_param,theo_dic,coord_dic,glob_fit_dic,spot_on):
    r"""**Planet-occulted / spotted exposure profiles: global model**
    
    Sub-function to define planet-occulted and spotted profiles using line profile models fitted to all
        - intrinsic profiles together in `fit_IntrProf_all()`.
          This model can be based on analytical, measured, or theoretical profiles.        
        - to all differential profiles together with `fit_ResProf_all()`.
          This model is based on analytical profiles.  
    
    Flux scaling is not applied to a global intrinsic profile using the chromatic light curve, but re-calculated for each planet-occulted intrinsic line profile for more flexibility (the cumulated scaling should be equivalent).
    rv used to shift the profiles are similarly re-calculated theoretically. 

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    data_vis = data_dic[inst][vis]
    def_iord = opt_dic['def_iord']
    if (data_dic['Intr']['align_mode']=='meas'):stop('This mode cannot be used with measured RVs')

    #Retrieving selected model properties
    fit_txt = None
    if (not spot_on) and ('IntrProf_prop_path' not in opt_dic):fit_txt = 'Intr'
    if (spot_on) and ('ResProf_prop_path' not in opt_dic):fit_txt = 'Res'
    if fit_txt is not None:
        fit_subdirs = glob.glob(gen_dic['save_data_dir']+'/Joined_fits/'+fit_txt+'Prof/*/')
        if len(fit_subdirs)==0:stop('         No existing fit directory. Run a fit to the '+{'Intr':'intrinsic','Res':'differential'}[fit_txt]+' line profiles.')
        else:
            fit_types = [fit_subdir.split('/')[-2] for fit_subdir in fit_subdirs]
            if 'mcmc' in fit_types:fit_used  ='mcmc'
            else:fit_used  ='chi2'
            print('         Using existing '+fit_used+' fit as default')
            opt_dic[fit_txt+'Prof_prop_path']={inst:{vis:gen_dic['save_data_dir']+'/Joined_fits/'+fit_txt+'Prof/'+fit_used+'/Fit_results'}}  
    if spot_on:prof_type = 'Res'
    else:prof_type='Intr'        
    data_prop = dataload_npz(opt_dic[prof_type+'Prof_prop_path'][inst][vis])
    params=data_prop['p_final'] 
    fixed_args={  
        'mode':opt_dic['mode'],
        'type':data_vis['type'],
        'nord':data_dic[inst]['nord'],
        'nthreads': opt_dic['nthreads'],
        'resamp_mode' : gen_dic['resamp_mode'], 
        'inst':inst,
        'vis':vis,  
        'fit':False,
        'ph_fit':data_prop['ph_fit'],
        'system_param':system_param,
        'genpar_instvis':data_prop['genpar_instvis'],
        'name_prop2input':data_prop['name_prop2input'],
        'fit_orbit':data_prop['fit_orbit'],
        'fit_RpRs':data_prop['fit_RpRs'],
        'var_par_list':data_prop['var_par_list'],
        'system_prop':data_prop['system_prop'],
        'grid_dic':data_prop['grid_dic'],
    } 
    if fixed_args['mode']=='ana':
        fixed_args.update({  
            'mac_mode':theo_dic['mac_mode'], 
            'coeff_line':data_prop['coeff_line_dic'][inst][vis],
            'func_prof_name':data_prop['func_prof_name'][inst]
        })        
        for key in ['coeff_ord2name','pol_mode','coord_line','linevar_par']:fixed_args[key] = data_prop[key]
    if spot_on:  
        fixed_args.update({          
        'fit_spot_ang':data_prop['fit_spot_ang'],
        'fit_spot':data_prop['fit_spot'],
        'system_spot_prop':data_prop['system_spot_prop'],
        'update_crosstime':False,    
        'conv2intr' :False,           
            })
        transit_spots=data_vis['transit_sp']
        iexp_list = range(data_vis['n_in_visit'])
        spots_prop = data_vis['spots_prop']
        plocc_prof_type = 'Res'
        
        #Defining parameters for the clean version of profiles
        if opt_dic['clean_calc']:
            clean_params = deepcopy(params)
            clean_params['use_spots']=False        
        
    else:
        fixed_args.update({          
        'system_spot_prop':{},          
            })
        plocc_prof_type = data_dic['Intr']['plocc_prof_type']
        if data_dic['Intr']['plocc_prof_type']=='Intr':fixed_args['conv2intr'] = True
        else:fixed_args['conv2intr'] = False 
        transit_spots={}
        spots_prop ={}
        iexp_list = range(data_vis['n_in_tr'])
    chrom_mode = data_vis['system_prop']['chrom_mode']
  
    #Activation of spectral conversion and resampling 
    cond_conv_st_prof_tab(theo_dic['rv_osamp_line_mod'],fixed_args,data_vis['type']) 

    #Updating coordinates with the best-fit properties
    ph_rec = {}
    coord_vis = coord_dic[inst][vis]
    for pl_loc in data_vis['transit_pl']:
        ph_rec[pl_loc] = np.vstack((coord_vis[pl_loc]['st_ph'],coord_vis[pl_loc]['cen_ph'],coord_vis[pl_loc]['end_ph']) ) 
    _,coord_pl_sp,_ = up_plocc_prop(inst,vis,fixed_args,params,data_vis['transit_pl'],ph_rec,coord_vis,transit_spots=transit_spots)

    #Processing relevant exposures
    for isub,iexp in enumerate(iexp_list):
        if not spot_on:
            iexp_glob = gen_dic[inst][vis]['idx_in2exp'][iexp]
            if plocc_prof_type=='Intr':iexp_eff = iexp
            else:iexp_eff = iexp_glob
        else:
            iexp_glob =iexp
            iexp_eff = iexp

        #Upload spectral tables from residual profile of current exposure
        data_loc_exp = dataload_npz(data_vis['proc_'+plocc_prof_type+'_data_paths']+str(iexp_eff))
    
        #Limit model table to requested definition range
        if len(opt_dic['def_range'])==0:cond_calc_pix = np.ones(data_vis['nspec'] ,dtype=bool)    
        else:cond_calc_pix = (data_loc_exp['edge_bins'][def_iord][0:-1]>=opt_dic['def_range'][0]) & (data_loc_exp['edge_bins'][def_iord][1:]<=opt_dic['def_range'][1])             
        idx_calc_pix = np_where1D(cond_calc_pix)
        cond_def_full = np.zeros(data_vis['nspec'],dtype=bool)
        cond_def_full[idx_calc_pix] = True
    
        #Saves
        data_store = {'cen_bins':data_loc_exp['cen_bins'],'edge_bins':data_loc_exp['edge_bins'],'cond_def':np.array([cond_def_full])}
    
        #Final table for model line profile
        fixed_args['ncen_bins']=len(idx_calc_pix)
        fixed_args['dim_exp'] = [1,fixed_args['ncen_bins']] 
        fixed_args['cen_bins'] = data_loc_exp['cen_bins'][def_iord,idx_calc_pix]
        fixed_args['edge_bins']=data_loc_exp['edge_bins'][def_iord,idx_calc_pix[0]:idx_calc_pix[-1]+2]
        fixed_args['dcen_bins']=fixed_args['edge_bins'][1::] - fixed_args['edge_bins'][0:-1] 
    
        #Initializing stellar profiles
        #    - can be defined using the first exposure table
        if isub==0:
            fixed_args = init_custom_DI_prof(fixed_args,gen_dic,data_vis['system_prop'],spots_prop,theo_dic,system_param['star'],params)                  
    
            #Effective instrumental convolution
            fixed_args['FWHM_inst'] = get_FWHM_inst(inst,fixed_args,fixed_args['cen_bins'])
    
        #Resampled spectral table for model line profile
        if fixed_args['resamp']:resamp_st_prof_tab(None,None,None,fixed_args,gen_dic,1,theo_dic['rv_osamp_line_mod'])
        
        #Table for model calculation
        args_exp = def_st_prof_tab(None,None,None,fixed_args) 
    
        #Define broadband scaling of intrinsic profiles into local profiles
        if plocc_prof_type=='Res':
            args_exp['Fsurf_grid_spec'] = theo_intr2loc(fixed_args['grid_dic'],fixed_args['system_prop'],args_exp,args_exp['ncen_bins'],fixed_args['grid_dic']['nsub_star']) 
    
        #Planet-occulted properties
        surf_prop_dic,spot_prop_dic,_ = sub_calc_plocc_spot_prop([chrom_mode],args_exp,['line_prof'],data_vis['transit_pl'],deepcopy(system_param),theo_dic,fixed_args['system_prop'],params,coord_pl_sp,[iexp_glob],system_spot_prop_in=fixed_args['system_spot_prop'])

        #With spots
        if spot_on:
            save_path = gen_dic['save_data_dir']+'Diff_estimates/'+corr_mode+'/'+inst+'_'+vis+'_'+str(iexp)

            #Planet-occulted and spotted line profiles - unclean
            pl_line_model = surf_prop_dic[chrom_mode]['line_prof'][:,0]
            sp_line_model = spot_prop_dic[chrom_mode]['line_prof'][:,0]
            line_prof_cons = {'unclean_pl_flux':pl_line_model, 'unclean_sp_flux':sp_line_model}
            
            #Planet-occulted and spotted line profiles - clean
            if opt_dic['clean_calc']:
                
                #Only planet-occulted profiles
                clean_surf_prop_dic,_,_ = sub_calc_plocc_spot_prop([chrom_mode],args_exp,['line_prof'],data_vis['transit_pl'],deepcopy(system_param),theo_dic,fixed_args['system_prop'],clean_params,coord_pl_sp,[iexp])
                
                #Only spotted profiles
                _,clean_spot_prop_dic,_ = sub_calc_plocc_spot_prop([chrom_mode],args_exp,['line_prof'],[],deepcopy(system_param),theo_dic,fixed_args['system_prop'],params,coord_pl_sp,[iexp],system_spot_prop_in=fixed_args['system_spot_prop'])
    
                #Storing
                clean_pl_line_model = clean_surf_prop_dic[chrom_mode]['line_prof'][:,0]
                clean_sp_line_model = clean_spot_prop_dic[chrom_mode]['line_prof'][:,0]
                line_prof_cons.update({'clean_pl_flux':clean_pl_line_model, 'clean_sp_flux':clean_sp_line_model})

        #Without spots
        else:
            save_path = gen_dic['save_data_dir']+'Loc_estimates/'+corr_mode+'/'+inst+'_'+vis+'_'+str(iexp)
    
            #Planet-occulted line profile 
            sp_line_model = surf_prop_dic[chrom_mode]['line_prof'][:,0]
            line_prof_cons = {'flux':sp_line_model}
        
        #Exposure profiles
        for line_prof in list(line_prof_cons.keys()):
           
            #Scaling to fitted intrinsic continuum level
            #    - model profiles have been output with a continuum level unity (through the option 'conv2intr') and are thus scaled to the fitted level
            if plocc_prof_type=='Intr':line_prof_cons['flux']*=params['cont']
         
            #Conversion and resampling 
            flux_loc = conv_st_prof_tab(None,None,None,fixed_args,args_exp,line_prof_cons['flux'],fixed_args['FWHM_inst'])
        
            #Filling full table with defined reconstructed profile
            plocc_prof = np.zeros(data_vis['nspec'],dtype=float)*np.nan
            plocc_prof[idx_calc_pix] = flux_loc
            
            #Store
            data_store.update({line_prof:np.array([plocc_prof])})
        
        #Saving estimate of local profile for current exposure                 
        np.savez_compressed(save_path,data=data_store,allow_pickle=True)

    #Complementary data
    data_add={'idx_est_loc':iexp_list,'cont':params['cont']}

    return data_add



def plocc_prof_indivmod(opt_dic,corr_mode,inst,vis,gen_dic,data_dic):
    r"""**Planet-occulted exposure profiles: individual model**
    
    Sub-function to define planet-occulted profiles using line profile model fitted to individual intrinsic lines.
    These models correspond directly to the measured profile and needs only be rescaled to the level of the local profile.
    This approach only works for exposures in which the stellar line could be fitted correctly after excluding the planet-contaminated range

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    data_vis = data_dic[inst][vis]
    if data_vis['type']!='CCF':stop('Method not valid for spectra')

    #Upload fit results
    data_prop=(np.load(gen_dic['save_data_dir']+'Introrig_prop/'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item())['prof_fit_dic']
    idx_aligned = np_where1D(data_prop['cond_detected'])

    #Processing in-transit exposures
    for i_in in idx_aligned:
     
        #Model intrinsic profile
        data_est_loc={'cen_bins':data_prop[i_in]['cen_bins'],'edge_bins':data_prop[i_in]['edge_bins'],'flux':data_prop[i_in]['flux'][:,None],'cond_def':np.ones(data_vis['dim_exp'],dtype=bool)}

        #Rescaling model intrinsic profile to the level of the local profile
        if data_dic['Intr']['plocc_prof_type']=='Res':
            loc_flux_scaling = dataload_npz(data_vis['scaled_Intr_data_paths']+str(gen_dic[inst][vis]['idx_in2exp'][i_in]))['loc_flux_scaling']
            data_est_loc['flux'][0] *= loc_flux_scaling(data_est_loc['cen_bins'][0]) 

        #Saving estimate of local profile for current exposure                   
        np.savez_compressed(gen_dic['save_data_dir']+'Loc_estimates/'+corr_mode+'/'+inst+'_'+vis+'_'+str(i_in),data=data_est_loc,allow_pickle=True)

    #Complementary data
    data_add={'idx_est_loc':idx_aligned}
      
    return data_add



def plocc_prof_rec(opt_dic,corr_mode,inst,vis,gen_dic,data_dic,coord_dic):
    r"""**Planet-occulted exposure profiles: reconstructed**
    
    Sub-function to define planet-occulted profiles by reconstructing undefined pixels via a polynomial fit to defined pixels in complementary exposures, or via a 2D interpolation over complementary exposures and a narrow spectral band.
    
     - providing the planetary track shifts sufficiently during the transit, we can reconstruct the local spectra at all wavelengths by interpolating the surrounding spectra at the wavelengths masked for planetary absorption.
       this approach allows accounting for changes in the shape of the local stellar spectra between exposures.
     - we reconstruct undefined pixels in the map of intrinsic profiles aligned in the null rest frame and resampled on the common spectral table.
       for each exposure, we then shift the corresponding reconstructed profile to the stellar surface velocity associated with the exposure, and resample it on the exposure table.
       ideally the map should be first aligned at the stellar surface velocity of each exposure, and then resampled on its table, but that would take too much time and the error is likely lower than that introduced by the interpolation.
     - no errors are propagated onto the new pixels.
     - at defined pixels, the local profile and its best estimate for the intrinsic profile will have the same values, resulting in null values in the atmospheric profiles.
       there will be small differences due to the need to resample the estimate on a different table than the corresponding local profile.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    data_inst = data_dic[inst]
    data_vis = data_inst[vis]
    if not gen_dic['align_Intr']:stop('Intrinsic stellar profiles must have been aligned')

    #Upload common spectral table for processed visit
    data_com = np.load(data_inst[vis]['proc_com_data_paths']+'.npz',allow_pickle=True)['data'].item()  

    #Using current visit exposures only, or exposures from multiple visits
    if (inst in opt_dic['vis_in_rec']) and (len(opt_dic['vis_in_rec'][inst])>0):vis_in_rec = opt_dic['vis_in_rec'][inst]     
    else:vis_in_rec = [vis]  

    #Initializing reconstruction
    y_cen_all = np.zeros(0,dtype=float)
    idx_orig_vis = np.zeros(0,dtype=int)
    vis_orig=np.zeros(0,dtype='U35')
    n_in_rec = 0
    for vis_rec in vis_in_rec:

        #Original indexes of aligned intrinsic profiles in current visit, which can potentially contribute to the reconstruction
        #    - relative to in-transit tables
        ref_pl,dic_rv_rec,_ = init_surf_shift(gen_dic,inst,vis_rec,data_dic,data_dic['Intr']['align_mode'])
            
        #Limiting aligned intrinsic profiles to input selection
        idx_in_rec = data_dic['Intr'][inst][vis]['idx_def']
        if (inst in opt_dic['idx_in_rec']) and (vis_rec in opt_dic['idx_in_rec'][inst]):
            idx_in_rec = np.intersect1d(opt_dic['idx_in_rec'][inst][vis_rec],)
            if len(idx_in_rec)==0:stop('No remaining exposures after input selection')  
        n_to_rec = len(idx_in_rec)                   

        #Tables along the chosen fit/interpolation dimension
        if opt_dic['dim_bin']=='phase':    
            y_cen_vis_rec = coord_dic[inst][vis_rec]['cen_ph'][gen_dic[inst][vis_rec]['idx_in']]   
        elif opt_dic['dim_bin'] in ['xp_abs','r_proj']: 
            transit_prop_nom = (np.load(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis_rec+'.npz',allow_pickle=True)['data'].item())['achrom'][ref_pl]                           
            y_cen_vis_rec = transit_prop_nom[opt_dic['dim_bin']][0,:]
        y_cen_all=np.append(y_cen_all,y_cen_vis_rec[idx_in_rec])
        
        #Store properties for visit to reconstruct
        if vis_rec==vis:
            dic_rv=deepcopy(dic_rv_rec)
            idx_aligned=deepcopy(idx_in_rec)
            
            #Indexes of processed visit within data_for_rec tables
            isub_vis_to_rec = n_in_rec+np.arange(n_to_rec)

            #Coordinates of exposures in processed visit
            y_cen_vis = y_cen_vis_rec[idx_in_rec]        

        #Store exposure identifiers
        n_in_rec+=n_to_rec                                          #total number of contributing exposures
        idx_orig_vis = np.append(idx_orig_vis,idx_in_rec)                   #original indexes of contributing exposures in their respective in-transit tables
        vis_orig = np.append(vis_orig,np.repeat(vis_rec,len(idx_in_rec)))   #original visits of contributing exposures

    #Upload contributing and processed intrinsic profiles
    #    - data must be put in global tables to perform the fits and interpolations  
    #    - the fits cannot deal with covariance matrix along the dimension orthogonal to the spectral one, thus we only use the diagonal
    data_for_rec={}
    data_rec={}
    for key in ['flux','err']:data_for_rec[key]=np.zeros([n_in_rec]+data_vis['dim_exp'], dtype=float)*np.nan
    data_for_rec['cond_def']=np.zeros([n_in_rec]+data_vis['dim_exp'], dtype=bool)
    isub_to_rec=0
    for isub_rec,(vis_rec,iexp_rec) in enumerate(zip(vis_orig,idx_orig_vis)):
        data_exp = np.load(gen_dic['save_data_dir']+'Aligned_Intr_data/'+inst+'_'+vis_rec+'_'+str(iexp_rec)+'.npz',allow_pickle=True)['data'].item()    
    
        #Resampling aligned intrinsic profiles on the common spectral table of the processed visit
        #    - if the processed and reconstructed visits are the same and exposures do not share a common table
        #    - if the processed and reconstructed visits are not the same, and visits do not share a common table
        if ((vis_rec==vis) and (not data_vis['comm_sp_tab'])) or ((vis_rec!=vis) and (not data_inst['comm_sp_tab'])): 
            for iord in range(data_inst['nord']):
                data_for_rec['flux'][isub_rec,iord],cov_ord = bind.resampling(data_com['edge_bins'][iord],data_exp['edge_bins'][iord], data_exp['flux'][iord] , cov =  data_exp['cov'][iord], kind=gen_dic['resamp_mode']) 
                data_for_rec['err'][isub_rec,iord] = np.sqrt(cov_ord[0])
            data_for_rec['cond_def'][isub_rec] = ~np.isnan(data_for_rec['flux'][isub_rec])     

        #Aligned intrinsic profiles are already defined on a common table for all visits
        else:
            for key in ['flux','cond_def']:data_for_rec[key][isub_rec] = data_exp[key]
            for iord in range(data_inst['nord']):data_for_rec['err'][isub_rec,iord] = np.sqrt(data_exp['cov'][iord][0])
            
        #Initialize reconstructed profiles table for processed visit               
        if (vis_rec==vis):
            data_rec[isub_to_rec]={
                'cen_bins':data_com['cen_bins'],
                'edge_bins':data_com['edge_bins'],
                'flux':data_for_rec['flux'][isub_rec]}
            isub_to_rec+=1

    #Process each order 
    for iord in range(data_dic[inst]['nord']):
        
        #Identify pixels to reconstruct in at least one exposure of the processed visit
        idx_undef_pix = np_where1D(np.sum(data_for_rec['cond_def'][isub_vis_to_rec,iord,:],axis=0)<n_to_rec) 

        #Process undefined pixels
        for idx_pix in idx_undef_pix: 

            #Exposures for which the pixel is defined in contributing exposures from all visits
            cond_def_pix_exp = data_for_rec['cond_def'][:,iord,idx_pix]

            #Exposures for which the pixel is undefined in exposures of processed visit
            idx_undef_pix_vis = np_where1D( data_for_rec['cond_def'][isub_vis_to_rec,iord,idx_pix] )
            
            #------------------------------------------------------------------------------------------------------------------------
                                         
            #Pixel-per-pixel interpolation
            #    - for each undefined pixel, we perform a polynomial fit along the chosen dimension over defined pixels
            if opt_dic['rec_mode']=='pix_polfit':

                #Fit polynomial along the chosen dimension    
                #    - only if more than two pixels are defined
                if np.sum(cond_def_pix_exp)>2:
                    func_def_pix = np.poly1d(np.polyfit(y_cen_all[cond_def_pix_exp], data_for_rec['flux'][cond_def_pix_exp,iord,idx_pix],opt_dic['pol_deg'],w=1./data_for_rec['err'][cond_def_pix_exp,iord,idx_pix]))

                    #Calculate polynomial in exposures of processed visit for which the pixel is undefined
                    val_def_pix = func_def_pix(y_cen_vis[idx_undef_pix_vis])
                    for isub_to_rec,val_def_pix_loc in zip(idx_undef_pix_vis,val_def_pix):
                        data_rec[isub_to_rec]['flux'][iord,idx_pix] = val_def_pix_loc   

            #------------------------------------------------------------------------------------------------------------------------
                        
            #2D band interpolation
            elif opt_dic['rec_mode']=='band_interp':  
                
                #Pixels to interpolate
                #    - we use the pixels defined over complementary exposures in a band surrounding the current pixel
                #    - tables must be put in 1D
                idx_band = np.arange(max(0,idx_pix-opt_dic['band_pix_hw']),min(data_vis['nspec']-1,idx_pix+opt_dic['band_pix_hw'])+1)
                nx_band = len(idx_band)
                cond_def_band = data_for_rec['cond_def'][:,iord,idx_band].flatten()      #defined pixels in 1D flux table within the band
                if np.sum(cond_def_band)>0:
                    flux_band = (data_for_rec['flux'][:,iord,idx_band].flatten())[cond_def_band]                       
                    x_cen_band = np.repeat(data_com['cen_bins'][iord,idx_band],n_in_rec)[cond_def_band]
                    y_cen_band = np.tile(y_cen_all,[nx_band])[cond_def_band]

                    #Tables must have the same scaling because the interpolation is done in euclidian norm
                    xmin_sc=np.min(x_cen_band)
                    ymin_sc=np.min(y_cen_band)
                    xscale=np.max(x_cen_band)-xmin_sc
                    yscale=np.max(y_cen_band)-ymin_sc
                    if (xscale>0) and (yscale>0):

                        #Pixels to reconstruct in processed visit
                        n_undef_pix_exp_vis = len(idx_undef_pix_vis)
                        x_cen_rec = np.repeat(data_com['cen_bins'][iord,idx_pix],n_undef_pix_exp_vis)
                        y_cen_rec = y_cen_vis[idx_undef_pix_vis]
                        
                        #Scaling
                        x_cen_band_sc=(x_cen_band-xmin_sc)/xscale
                        y_cen_band_sc=(y_cen_band-ymin_sc)/yscale    
                        x_cen_rec_sc=(x_cen_rec-xmin_sc)/xscale
                        y_cen_rec_sc=(y_cen_rec-ymin_sc)/yscale     
 
                        #Interpolating
                        val_def_pix = griddata((x_cen_band_sc,y_cen_band_sc), flux_band, (x_cen_rec_sc, y_cen_rec_sc),method=opt_dic['interp_mode'])
                        for isub_to_rec,val_def_pix_loc in zip(idx_undef_pix_vis,val_def_pix):
                            data_rec[isub_to_rec]['flux'][iord,idx_pix] = val_def_pix_loc

    #------------------------------------------------------------------------------------------------------------------------

    #Processing in-transit exposures with reconstructed intrinsic profiles
    for isub,i_in in enumerate(idx_aligned):

        #Upload residual or intrinsic profile for current exposure to get its spectral tables
        if data_dic['Intr']['plocc_prof_type']=='Intr':iexp_eff = i_in
        else:iexp_eff = gen_dic[inst][vis]['idx_in2exp'][i_in]
        data_loc_exp = dataload_npz(data_vis['proc_'+data_dic['Intr']['plocc_prof_type']+'_data_paths']+str(iexp_eff))  

        #Radial velocity shifts set to the opposite of the planet-occulted surface rv associated with current exposure
        rv_surf_star,rv_surf_star_edge = def_surf_shift(data_dic['Intr']['align_mode'],dic_rv,i_in,data_rec[isub],ref_pl,data_vis['type'],data_vis['system_prop'],data_vis['dim_exp'],data_dic[inst]['nord'],data_vis['nspec'])

        #Aligning reconstructed profile at current exposure stellar surface rv 
        #    - reconstructed profile is already aligned in a null rest frame and resampled on a common table
        #    - aligned profiles are resampled on the table of current exposure, which is common to all exposures if a common table is used 
        rv_shift_cen = -rv_surf_star
        spec_dopshift_cen = gen_specdopshift(rv_surf_star)
        if rv_surf_star_edge is not None:
            rv_shift_edge = -rv_surf_star_edge
            spec_dopshift_edge = gen_specdopshift(rv_surf_star_edge)
        else:
            rv_shift_edge = None
            spec_dopshift_edge = None  
        data_est_loc=align_data(data_rec[isub],data_vis['type'],data_dic[inst]['nord'],data_vis['dim_exp'],gen_dic['resamp_mode'],data_loc_exp['cen_bins'],data_loc_exp['edge_bins'],rv_shift_cen,spec_dopshift_cen,rv_shift_edge = rv_shift_edge,spec_dopshift_edge = spec_dopshift_edge, nocov = True)

        #Rescaling reconstructed intrinsic profile to the level of the local profile
        if data_dic['Intr']['plocc_prof_type']=='Res':
            loc_flux_scaling = dataload_npz(data_vis['scaled_Intr_data_paths']+str(gen_dic[inst][vis]['idx_in2exp'][i_in]))['loc_flux_scaling']
            for iord in range(data_dic[inst]['nord']):data_est_loc['flux'][iord] *=loc_flux_scaling(data_est_loc['cen_bins'][iord]) 

        #Saving estimate of local profile for current exposure                   
        np.savez_compressed(gen_dic['save_data_dir']+'Loc_estimates/'+corr_mode+'/'+inst+'_'+vis+'_'+str(i_in),data=data_est_loc,allow_pickle=True)


    #Complementary data
    data_add={'idx_est_loc':idx_aligned}

    return data_add



#%% Differential exposure profiles.
#    - accounting for planet-occulted and spotted stellar profiles

def def_diff_profiles(inst,vis,gen_dic,data_dic,data_prop,coord_dic,system_param,theo_dic,glob_fit_dic,plot_dic):
    r"""**Planet-occulted and spotted exposure profiles.**
    
    Calls requested function to define planet-occulted and spotted profiles associated with each observed exposure

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    print('   > Building estimates for planet-occulted and spotted stellar profiles')
    if data_dic[inst][vis]['spots_prop']=={}:stop('Spot properties are not provided. We recommended using the separate routine dedicated to the extraction of planet-occulted profiles instead.')
    opt_dic = data_dic['Res']['opt_loc_data_corr'] 
    corr_mode = opt_dic['corr_mode']
    print('         Using global model')  

    #Calculating
    if (gen_dic['calc_res_loc_data_corr']):
        print('         Calculating data')     
             
        #Calculating the clean version of the data
        opt_dic['clean_calc']=False
        if (plot_dic['map_Res_prof_clean_pl_est']!='') or (plot_dic['map_Res_prof_clean_sp_est']!=''):
            opt_dic['clean_calc']=True

        #Using global profile model
        if corr_mode=='glob_mod': 
            data_add = plocc_spocc_prof_globmod(opt_dic,corr_mode,inst,vis,gen_dic,data_dic,data_prop,system_param,theo_dic,coord_dic,glob_fit_dic,False)
        
        else:stop('WARNING: Only joined-fit results can be used at the moment. Set corr_mode to \'glob_mod\'')

        #Saving complementary data for plots
        if (plot_dic['map_Res_prof_clean_sp_res']!='') or (plot_dic['map_Res_prof_clean_pl_res']!='') or (plot_dic['map_Res_prof_unclean_sp_res']!='') or (plot_dic['map_Res_prof_unclean_pl_res']!=''):
            data_add['loc_data_corr_path'] = data_dic[inst][vis]['proc_Res_data_paths']
            data_add['rest_frame'] = data_dic['Res'][inst][vis]['rest_frame']
        datasave_npz(gen_dic['save_data_dir']+'Diff_estimates/'+corr_mode+'/'+inst+'_'+vis+'_add',data_add)

    #Checking that local data has been calculated for all exposures
    else:
        idx_est_loc = dataload_npz(gen_dic['save_data_dir']+'Diff_estimates/'+corr_mode+'/'+inst+'_'+vis+'_add')['idx_est_loc']
        data_paths={i_in:gen_dic['save_data_dir']+'Diff_estimates/'+corr_mode+'/'+inst+'_'+vis+'_'+str(i_in) for i_in in idx_est_loc}
        check_data(data_paths)

    return None







