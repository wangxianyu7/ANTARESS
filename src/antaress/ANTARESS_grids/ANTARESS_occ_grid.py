#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from itertools import product as it_product
from copy import deepcopy
import lmfit
from lmfit import Parameters
from ..ANTARESS_grids.ANTARESS_coord import frameconv_skyorb_to_skystar,frameconv_skystar_to_skyorb,frameconv_skystar_to_star,calc_pl_coord,frameconv_star_to_skystar,calc_zLOS_oblate,coord_expos_ar,is_ar_visible
from ..ANTARESS_process.ANTARESS_data_align import align_data
from ..ANTARESS_analysis.ANTARESS_inst_resp import convol_prof
from ..ANTARESS_grids.ANTARESS_star_grid import calc_CB_RV,get_LD_coeff,calc_st_sky,calc_Isurf_grid,calc_RVrot
from ..ANTARESS_analysis.ANTARESS_model_prof import calc_polymodu,polycoeff_def
from ..ANTARESS_grids.ANTARESS_prof_grid import coadd_loc_line_prof,coadd_loc_gauss_prof,calc_loc_line_prof,init_st_intr_prof,calc_linevar_coord_grid, use_C_coadd_loc_gauss_prof
from ..ANTARESS_general.utils import stop,closest,np_poly,npint,np_interp,np_where1D,datasave_npz,dataload_npz,gen_specdopshift,check_data
from ..ANTARESS_general.constant_data import Rsun,c_light

#%% Common routines

def calc_plocc_ar_prop(system_param,gen_dic,theo_dic,coord_dic,inst,vis,data_dic,calc_pl_atm=False,ar_dic={}):
    r"""**Planet-occulted / active region properties: workflow**

    Calls function to calculate theoretical properties of the regions occulted by all transiting planets and/or active regions. 

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 

    #Check for active regions
    if (ar_dic != {}) and (inst in ar_dic['ar_prop']) and (vis in ar_dic['ar_prop'][inst]):
        txt_ar = ' and active '
        cond_ar = True
    else:
        txt_ar = ' '
        cond_ar = False
    
    print('   > Calculating properties of planet-occulted'+txt_ar+'regions')    
    if gen_dic['calc_theoPlOcc']:
        print('         Calculating data')
        
        #Theoretical properties of active regions
        param = deepcopy(system_param['star'])
        param['use_ar']=cond_ar
        args={'rout_mode':'Intr_prop'}
        if cond_ar:args['ar_coord_par']= gen_dic['ar_coord_par']
        
        #Theoretical properties of planet occulted-regions
        #    - calculated for the nominal and broadband planet properties 
        #    - for the nominal properties we retrieve the range of some properties covered by the planet during each exposures
        #    - chromatic transit required if local profiles are in spectral mode  
        param.update({'rv':0.,'cont':1.})
        par_list=['rv','CB_RV','mu','lat','lon','x_st','y_st','SpSstar','xp_abs','r_proj']
        key_chrom = ['achrom']
        if ('spec' in data_dic[inst][vis]['type']) and ('chrom' in data_dic[inst][vis]['system_prop']):key_chrom+=['chrom']
        
        #Calculate properties
        plocc_prop,ar_prop,common_prop = sub_calc_plocc_ar_prop(key_chrom,args,par_list,data_dic[inst][vis]['studied_pl'],data_dic[inst][vis]['studied_ar'],system_param,theo_dic,data_dic[inst][vis]['system_prop'],param,coord_dic[inst][vis],gen_dic[inst][vis]['idx_in'], system_ar_prop_in = data_dic['DI']['ar_prop'], out_ranges=True)
        
        #Save active region properties
        if cond_ar:
            datasave_npz(gen_dic['save_data_dir']+'Introrig_prop/AR_Prop_'+inst+'_'+vis,ar_prop)    

            #Save properties combined from planet-occulted and active regions
            datasave_npz(gen_dic['save_data_dir']+'Introrig_prop/Common_Prop_'+inst+'_'+vis,common_prop) 

        #Save planet-occulted region properties
        datasave_npz(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis,plocc_prop)

    else:
        check_data({'path':gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis})        

    return None






def up_plocc_arocc_prop(inst,vis,args,param_in,studied_pl,ph_grid,coord_grid, studied_ar=[]):
    r"""**Planet-occulted and active region properties: update**

    Updates properties of the planet-occulted region, planetary orbit, and active regions for fitted step. 

    Args:
        inst (str) : Instrument considered.
        vis (str) : Visit considered. 
        args (dict) : Additional parameters needed to evaluate the fitted function.
        param_in (dict) : Model parameters for the fitted step considered.
        studied_pl (list) : Transiting planets for the instrument and visit considered.
        ph_grid (dict) : Dictionary containing the phase of each planet.
        coord_grid (dict) : Dictionary containing the various coordinates of each planet and active region (e.g., exposure time, exposure x/y/z coordinate).
        studied_ar (list) : Visible active regions present for the instrument and visit considered.
    
    Returns:
        system_param_loc (dict) : System (star+planet+active regions) properties.
        coords (dict) : Updated planet and active region coordinates.
        param (dict) : Model parameter names and values.
    
    """ 
    system_param_loc=deepcopy(args['system_param'])

    #In case param_in is defined as a Parameters structure, retrieve values and define dictionary
    param={}
    if isinstance(param_in,lmfit.parameter.Parameters):
        for par in param_in:param[par]=param_in[par].value
    else:param=param_in

    #Coefficients describing the polynomial variation of spectral line properties as a function of the chosen coordinate
    #    - coefficients can be specific to a given spectral line model
    if (args['mode']=='ana') and (len(args['linevar_par'])>0):
        args['coeff_line'] = {}
        for par_loc in args['linevar_par'][inst][vis]:    
            args['coeff_line'][par_loc] = polycoeff_def(param,args['coeff_ord2name'][inst][vis][par_loc])

    #Instrument or visit-specific line continuum
    if ('cont' in args['genpar_instvis']):param['cont'] = param[args['name_prop2input']['cont__IS'+inst+'_VS'+vis]] 

    #Recalculate coordinates of occulted regions or use nominal values
    #    - the 'fit_X' conditions are only True if at least one parameter is varying, so that param_fit is True if fit_X is True
    coords = deepcopy(coord_grid)
    for pl_loc in studied_pl:

        #Recalculate planet grid if relevant
        if args['fit_RpRs'] and ('RpRs__pl'+pl_loc in args['var_par_list']):
            args['system_prop']['achrom'][pl_loc][0]=param['RpRs__pl'+pl_loc] 
            args['grid_dic']['Ssub_Sstar_pl'][pl_loc],args['grid_dic']['x_st_sky_grid_pl'][pl_loc],args['grid_dic']['y_st_sky_grid_pl'][pl_loc],r_sub_pl2=occ_region_grid(args['system_prop']['achrom'][pl_loc][0],args['grid_dic']['nsub_Dpl'][pl_loc])  
            args['system_prop']['achrom']['cond_in_RpRs'][pl_loc] = [(r_sub_pl2<args['system_prop']['achrom'][pl_loc][0]**2.)]        

        #Recalculate planet coordinates if relevant 
        pl_params_loc = system_param_loc[pl_loc]

        #Update fitted system properties for current step 
        if args['fit_orbit']:
            if ('lambda_rad__pl'+pl_loc in args['genpar_instvis']):lamb_name = 'lambda_rad__pl'+pl_loc+'__IS'+inst+'_VS'+vis 
            else:lamb_name = 'lambda_rad__pl'+pl_loc 
            if (lamb_name in args['var_par_list']):pl_params_loc['lambda_rad'] = param[lamb_name]                     
            if ('inclin_rad__pl'+pl_loc in args['var_par_list']):pl_params_loc['inclin_rad']=param['inclin_rad__pl'+pl_loc]       
            if ('aRs__pl'+pl_loc in args['var_par_list']):pl_params_loc['aRs']=param['aRs__pl'+pl_loc]  
            
        #Calculate coordinates
        #    - start/end phase have been set to None if no oversampling is requested, in which case start/end positions are not calculated
        if args['fit_orbit'] or args['fit_star_pl']:
            if args['grid_dic']['d_oversamp_pl'] is not None:phases = ph_grid[pl_loc]
            else:phases = ph_grid[pl_loc][1]
            x_pos_pl,y_pos_pl,_,_,_,_,_,_,ecl_pl = calc_pl_coord(pl_params_loc['ecc'],pl_params_loc['omega_rad'],pl_params_loc['aRs'],pl_params_loc['inclin_rad'],phases,args['system_prop']['achrom'][pl_loc][0],pl_params_loc['lambda_rad'],system_param_loc['star'])
            coords[pl_loc]={}
            if args['grid_dic']['d_oversamp_pl'] is not None:
                coords[pl_loc]['st_pos'] = np.vstack((x_pos_pl[0],y_pos_pl[0]))
                coords[pl_loc]['cen_pos'] = np.vstack((x_pos_pl[1],y_pos_pl[1]))
                coords[pl_loc]['end_pos'] = np.vstack((x_pos_pl[2],y_pos_pl[2]))
            else:coords[pl_loc]['cen_pos'] = np.vstack((x_pos_pl,y_pos_pl))
            coords[pl_loc]['ecl'] = ecl_pl

    #Process active regions
    if len(studied_ar)>0:

        #Set up properties of active regions for the active region coordinate retrieval in sub_calc_plocc_ar_prop
        for ar_loc in studied_ar:
            
            #Recalculate active region grid if relevant
            if ar_loc in args['fit_ar_ang']:
                args['system_ar_prop']['achrom'][ar_loc][0]=param['ang__IS'+inst+'_VS'+vis+'_AR'+ar_loc] * np.pi/180
                _,args['grid_dic']['Ssub_Sstar_ar'][ar_loc],args['grid_dic']['x_st_sky_grid_ar'][ar_loc],args['grid_dic']['y_st_sky_grid_ar'][ar_loc],_ = occ_region_grid(np.sin(args['system_ar_prop']['achrom'][ar_loc][0]),args['grid_dic']['nsub_Dar'][ar_loc],planet=True)  

            #Update active region crossing time before doing active region parameters' retrieval
            if args['fit_ar']:param['Tc_ar__IS'+inst+'_VS'+vis+'_AR'+ar_loc] += args['bjd_time_shift'][inst][vis]

        #Recalculate active region coordinates if relevant        
        if args['fit_ar'] or (args['fit_ar_ang']!=[]):
    
            #Retrieving the active region coordinates for all the times that we have
            ar_prop = retrieve_ar_prop_from_param(param,inst,vis)
            ar_prop['cos_istar']=system_param_loc['star']['cos_istar']      
            for ar_loc in studied_ar:
                coords[ar_loc] = {}
                for key in args['ar_coord_par']:coords[ar_loc][key] = np.zeros([3,len(coords['bjd'])],dtype=float)*np.nan
                coords[ar_loc]['is_visible'] = np.zeros([3,len(coords['bjd'])],dtype=bool)
                for key in ['Tc_ar', 'ang_rad', 'lat_rad', 'fctrst']:coords[ar_loc][key] = ar_prop[ar_loc][key]
            for ifit_tstamp, fit_tstamp in enumerate(coords['bjd']):                
                for ar_loc in studied_ar:
                    ar_prop_exp = coord_expos_ar(ar_loc,fit_tstamp,ar_prop,system_param_loc['star'],coords['t_dur'][ifit_tstamp],args['ar_coord_par'])                           
                    for key in ar_prop_exp:coords[ar_loc][key][:, ifit_tstamp] = [ar_prop_exp[key][0],ar_prop_exp[key][1],ar_prop_exp[key][2]]     

        #Trigger use of active regions in the function computing the DI profile deviation
        param['use_ar']=True

    #Useful if active regions are present but the active region parameters are fixed
    else:param['use_ar']=False

    return system_param_loc,coords,param




def sub_calc_plocc_ar_prop(key_chrom,args,par_list_gen,studied_pl,studied_ar,system_param,theo_dic,star_I_prop_in,param,coord_in,iexp_list,system_ar_prop_in={},out_ranges=False,Ftot_star=False):
    r"""**Planet-occulted and active region properties: exposure**

    Calculates average theoretical properties of the stellar surface occulted by all transiting planets and/or within active regions during an exposure
    
     - we normalize all quantities by the flux emitted by the occulted regions
     - all positions are in units of :math:`R_\star` 
    
    Args:
        key_chrom (list) : chromatic modes used (either chromatic, 'chrom', achromatic, 'achrom', or both).
        args (dict) : parameters used to generate analytical profiles.
        par_list_gen (list) : parameters whose value we want to calculate over each planet-occulted/active region.
        studied_pl (list) : list of transiting planets in the exposures considered.
        studied_ar (list) : list of visible active regions in the exposures considered.
        system_param (dict) : system (star + planet + active region) properties.
        theo_dic (dict) : parameters used to generate and describe the stellar grid and planet-occulted/active regions grid.
        star_I_prop_in (dict) : stellar intensity properties.
        param (dict) : fitted or fixed star/planet/active region properties.
        coord_in (dict) : dictionary containing the various coordinates of each planet and active region (e.g., exposure time, exposure phase, exposure x/y/z coordinate)
        iexp_list (list) : exposures to process.
        system_ar_prop_in (dict) : optional, active region limb-darkening properties.
        out_ranges (bool) : optional, whether or not to calculate the range of values the parameters of interest (par_list_gen) will take. Turned off by default.
        Ftot_star (bool) : optional, whether or not to calculate the normalized stellar flux after accounting for the active region/planet occultations. Turned off by default.
    
    Returns:
        surf_prop_dic_pl (dict) : average value of all the properties of interest over all the planet-occulted regions, in each exposure and chromatic mode considered.
        surf_prop_dic_ar (dict) : average value of all the properties of interest over all the active regions, in each exposure and chromatic mode considered.
        surf_prop_dic_common (dict) : average value of all the properties of interest considering the contributions from both the planet-occulted and active regions, in each exposure and chromatic mode considered.

    """ 
    star_I_prop = deepcopy(star_I_prop_in)
    system_ar_prop = deepcopy(system_ar_prop_in)
    par_list_in = deepcopy(par_list_gen)
    n_exp = len(iexp_list)
    
    #Active region activation condition
    #    - if active regions are being used, and they are in the param dictionary provided (which is not always the case)  
    if 'use_ar' in param.keys() and param['use_ar']:cond_ar = True
    else:cond_ar = False    

    #Line properties initialization
    if ('linevar_par' in args) and (len(args['linevar_par'])>0):
        args['linevar_par_vis'] = args['linevar_par'][args['inst']][args['vis']]
    else:args['linevar_par_vis'] = []

    #Line profile initialization
    if ('line_prof' in par_list_in):
        par_list_in+=['rv','mu','SpSstar']+args['linevar_par_vis']
        
        #Chromatic / achromatic calculation
        if len(key_chrom)>1:stop('Function can only be called in a single mode to calculate line profiles')
        switch_chrom = False
        if key_chrom==['chrom']:
            
            #Full profile width is smaller than the typical scale of chromatic variations
            #    - mode is switched to closest-achromatic mode, with properties set to those of the closest chromatic band
            if (args['edge_bins'][-1]-args['edge_bins'][0]<star_I_prop['chrom']['med_dw']):
                key_chrom=['achrom']
                switch_chrom = True
                iband_cl = closest(star_I_prop['chrom']['w'],np.median(args['cen_bins']))
                for key in ['w','LD','GD_wmin','GD_wmax','GD_dw']:star_I_prop['achrom'][key] = [star_I_prop['chrom'][key][iband_cl]]
                for pl_loc in studied_pl:
                    star_I_prop['achrom']['cond_in_RpRs'][pl_loc] = [star_I_prop['chrom']['cond_in_RpRs'][pl_loc][iband_cl]] 
                    star_I_prop['achrom'][pl_loc] = [star_I_prop['chrom'][pl_loc][iband_cl]]                

            #Profiles covers a wide spectral band
            #    - requires calculation of achromatic properties                
            else:
                if (args['mode']=='ana'):stop('Analytical model not suited for wide spectral bands') 
                if (theo_dic['precision']=='high'):stop('High precision not possible for wide spectral bands') 
                key_chrom=['achrom','chrom']

    #Calculation of achromatic and/or chromatic values
    surf_prop_dic_pl = {}
    surf_prop_dic_ar = {}
    surf_prop_dic_common = {}
    for subkey_chrom in key_chrom:
        surf_prop_dic_pl[subkey_chrom] = {}        
        surf_prop_dic_ar[subkey_chrom] = {}
        surf_prop_dic_common[subkey_chrom] = {}
    if 'line_prof' in par_list_in:
        for subkey_chrom in key_chrom:
            surf_prop_dic_pl[subkey_chrom]['line_prof']=np.zeros([args['ncen_bins'],n_exp],dtype=float)
            surf_prop_dic_ar[subkey_chrom]['line_prof']=np.zeros([args['ncen_bins'],n_exp],dtype=float)
            if 'corr_ar' in args:surf_prop_dic_pl[subkey_chrom]['corr_supp']=np.zeros([args['ncen_bins'],n_exp],dtype=float)

    #Properties to be calculated
    #    - properties in 'param' have the nominal values from system properties only if the property was not defined in the model property dictionary from settings 
    par_star = deepcopy(param)
    par_list = ['Ftot']
    if ('Rstar' in par_star):Rstar_km = par_star['Rstar']*Rsun
    else:Rstar_km = system_param['star']['Rstar_km']   
    if ('Peq' in par_star):Peq = par_star['Peq']
    else:Peq = system_param['star']['Peq']       
    for par_loc in par_list_in:
        if par_loc=='rv':
            par_list+=['Rot_RV']
            if ('rv_line' in args['linevar_par_vis']):par_list+=['rv_line']
            
            #Updating veq (which is the parameter used to calculate surface rvs) when Peq is the fitted parameter
            if ('Rstar' in par_star) or ('Peq' in par_star):
                par_star['veq'] = 2.*np.pi*Rstar_km/(Peq*24.*3600.)
            
        elif (par_loc not in ['line_prof']):par_list+=[par_loc]
    cos_istar = (par_star['cos_istar']-(1.)) % 2 - 1.   #Reset cos_istar within -1 : 1
    par_star['istar_rad']=np.arccos(cos_istar)
    cb_band_dic = {}
    if cond_ar:cb_band_ar_dic = {}
    for subkey_chrom in key_chrom:

        #Disk-integrated stellar flux
        if Ftot_star:
            surf_prop_dic_pl[subkey_chrom]['Ftot_star']=np.zeros([star_I_prop[subkey_chrom]['nw'],n_exp])*np.nan 
            surf_prop_dic_common[subkey_chrom]['Ftot_star']=np.zeros([star_I_prop[subkey_chrom]['nw'],n_exp])*np.nan 
            if cond_ar:
                surf_prop_dic_ar[subkey_chrom]['Ftot_star']=np.zeros([star_I_prop[subkey_chrom]['nw'],n_exp])*np.nan 

        #Convective blueshift
        #    - physically, it makes sense for us to define different CB coefficients for an active region since they are regions of magnetic suppression and would have different CB.
        #However, we make the simplifying assumption that the c1_CB, c2_CB, and c3_CB coefficient are the same for the active region as for the quiet star regions, with c0_CB being
        #the only coefficient that varies, and which is calculated with the same condition as before. 
        #Even though our assumption is not correct, we think that the RV shift induced by the difference in CB for the active region can be captured in the RV parameter used to describe the
        #line profiles with which the active region is tiled.
        cb_band_dic[subkey_chrom]={}  
        if cond_ar:cb_band_ar_dic[subkey_chrom] = {}  
        if ('CB_RV' in par_list) or ('c0_CB' in par_list):     
            surf_prop_dic_pl[subkey_chrom]['c0_CB']=np.zeros(star_I_prop[subkey_chrom]['nw'])*np.nan
            if cond_ar:surf_prop_dic_ar[subkey_chrom]['c0_CB']=np.zeros(star_I_prop[subkey_chrom]['nw'])*np.nan
            for iband in range(star_I_prop[subkey_chrom]['nw']):
                cb_band_dic[subkey_chrom][iband] = calc_CB_RV(get_LD_coeff(star_I_prop[subkey_chrom],iband),star_I_prop[subkey_chrom]['LD'][iband],par_star['c1_CB'],par_star['c2_CB'],par_star['c3_CB'],par_star) 
                surf_prop_dic_pl[subkey_chrom]['c0_CB'][iband]=cb_band_dic[subkey_chrom][iband][0] 
                if cond_ar:
                    cb_band_ar_dic[subkey_chrom][iband] = calc_CB_RV(get_LD_coeff(system_ar_prop[subkey_chrom],iband),system_ar_prop[subkey_chrom]['LD'][iband],par_star['c1_CB'],par_star['c2_CB'],par_star['c3_CB'],par_star) 
                    surf_prop_dic_ar[subkey_chrom]['c0_CB'][iband]=cb_band_ar_dic[subkey_chrom][iband][0]
        else:
            for iband in range(star_I_prop[subkey_chrom]['nw']):
                cb_band_dic[subkey_chrom][iband] = None
                if cond_ar:cb_band_ar_dic[subkey_chrom][iband] = None
    if 'rv' in par_list_in:par_list+=['rv']  #must be placed after all other RV contributions

    #List of parameters whose range we're interested in
    range_par_list=[]
    if (len(theo_dic['d_oversamp_pl'])>0) and out_ranges:range_par_list = list(np.intersect1d(['mu','lat','lon','x_st','y_st','xp_abs','r_proj'],par_list))
 
    #Initializing active region variables
    #    - must be initialized in anyy case since they will be called later, even if active regions are not activated.
    cond_ar_all = np.zeros([n_exp,1], dtype=bool)
    n_ar = len(studied_ar)

    #Initializing list that will contain the oversampled steps for planets and active regions, if they are oversampled.
    dcoord_exp_in = {'x':{},'y':{},'z':{}}

    #Initialize a list which will tell us the oversampling rate for each exposure
    n_osamp_exp_all_ar = np.repeat(1,n_exp)

    #Define active regions properties
    if cond_ar:
        
        #High precision is required for active regions
        if (theo_dic['precision']!='high'):stop('ERROR: High precision required for active regions')

        #Initialize the dictionary that will contain active region presence
        cond_ar_all = np.zeros([n_exp,n_ar], dtype=bool)

        #Oriented distance covered along each dimension (in Rstar)
        if len(theo_dic['n_oversamp_ar'])>0:
            for ar in studied_ar:
                for key in ['x','y','z']:dcoord_exp_in[key][ar] = coord_in[ar][key+'_sky_exp'][2,iexp_list] - coord_in[ar][key+'_sky_exp'][0,iexp_list]
                
        #Looping over all exposures
        for isub_exp, iexp in enumerate(iexp_list):

            #Check if at least one active region is visible.
            #    - to do so, we need a more precise estimate of the active region location.
            ar_within_grid_all=np.zeros(n_ar, dtype=bool)

            #Go through the active regions and see if they are *roughly* visible.
            for ar_index, ar in enumerate(studied_ar):

                #See if active region is visible at any point during the exposure.
                if (np.sum(coord_in[ar]['is_visible'][:, iexp])>=1):

                    #Need to make a dictionary of active region coordinates which will be used in calc_ar_tiles.
                    mini_ar_dic = {}
                    for par_ar in args['ar_coord_par']:mini_ar_dic[par_ar] = coord_in[ar][par_ar][:, iexp]
    
                    #See if active region is *precisely* visible.
                    ar_within_grid, _ = calc_ar_tiles(mini_ar_dic,coord_in[ar]['ang_rad'],theo_dic['x_st_sky'], theo_dic['y_st_sky'], theo_dic['z_st_sky'], theo_dic,par_star, True)
                    if ar_within_grid:ar_within_grid_all[ar_index]=True

                    #Check if oversampling is turned on for this active region and force all active regions to have same oversampling rate
                    if (ar in theo_dic['n_oversamp_ar']):
                        n_osamp_exp_all_ar[isub_exp] = np.maximum(n_osamp_exp_all_ar[isub_exp], theo_dic['n_oversamp_ar'][ar])

                    #Active region-dependent properties - initialize dictionaries
                    for subkey_chrom in key_chrom:
                        surf_prop_dic_ar[subkey_chrom][ar]={}
                        for par_loc in par_list:
                            surf_prop_dic_ar[subkey_chrom][ar][par_loc]=np.zeros([system_ar_prop[subkey_chrom]['nw'],n_exp])*np.nan        
                        for par_loc in range_par_list:surf_prop_dic_ar[subkey_chrom][ar][par_loc+'_range']=np.zeros([system_ar_prop[subkey_chrom]['nw'],n_exp,2])*np.nan

            #Update cond_ar_all
            cond_ar_all[isub_exp]=ar_within_grid_all
        if (np.sum(cond_ar_all)==0) and (not args['fit']):print('WARNING: no active regions are visible in any exposure for nominal properties')

    #If active regions are not present, need to initialize the active region LD dictionary entry for later purposes
    else:
        for subkey_chrom in key_chrom:system_ar_prop[subkey_chrom]={}

    #Occulted planet zones properties
    n_osamp_exp_all = np.repeat(1,n_exp)
    lambda_rad_pl = {}
    cond_transit_all = np.zeros([n_exp,len(studied_pl)],dtype=bool)
    for ipl,pl_loc in enumerate(studied_pl):

        #Check for planet transit
        if np.sum(np.abs(coord_in[pl_loc]['ecl'][iexp_list])!=1.)>0:
            cond_transit_all[:,ipl]|=(np.abs(coord_in[pl_loc]['ecl'][iexp_list])!=1.)   

            #Obliquities for multiple planets
            #    - for now only defined for a single planet if fitted  
            #    - the nominal lambda has been overwritten in 'system_param[pl_loc]' if fitted
            lambda_rad_pl[pl_loc]=system_param[pl_loc]['lambda_rad']
            
            #Exposure oversampling
            if len(theo_dic['d_oversamp_pl'])>0:
                
                #Oriented distance covered along each dimension (in Rstar)
                for ikey,key in enumerate(['x','y']):dcoord_exp_in[key][pl_loc] = coord_in[pl_loc]['end_pos'][ikey,iexp_list]-coord_in[pl_loc]['st_pos'][ikey,iexp_list]

                #Number of oversampling points for current exposure  
                #    - for each exposure we take the maximum oversampling all planets considered 
                if (pl_loc in theo_dic['d_oversamp_pl']):
                    d_exp_in = np.sqrt(dcoord_exp_in['x'][pl_loc]**2 + dcoord_exp_in['y'][pl_loc]**2)
                    n_osamp_exp_all=np.maximum(n_osamp_exp_all,npint(np.round(d_exp_in/theo_dic['d_oversamp_pl'][pl_loc]))+1)
                    
            #Planet-dependent properties
            for subkey_chrom in key_chrom:
                surf_prop_dic_pl[subkey_chrom][pl_loc]={}
                for par_loc in par_list:
                    surf_prop_dic_pl[subkey_chrom][pl_loc][par_loc]=np.zeros([star_I_prop[subkey_chrom]['nw'],n_exp])*np.nan        
                for par_loc in range_par_list:surf_prop_dic_pl[subkey_chrom][pl_loc][par_loc+'_range']=np.zeros([star_I_prop[subkey_chrom]['nw'],n_exp,2])*np.nan
                if ('line_prof' in par_list_in) and (theo_dic['precision']=='low'):
                    surf_prop_dic_pl[subkey_chrom][pl_loc]['rv_broad']=-1e100*np.ones([star_I_prop[subkey_chrom]['nw'],n_exp])

    #Figuring out which exposures are occulted (by active regions or planets)
    cond_iexp_proc = (np.sum(cond_ar_all, axis=1)>0)|(np.sum(cond_transit_all,axis=1)>0)

    #Enforcing a common oversampling factor to the active regions and planets
    n_osamp_exp_all_total = np.maximum(n_osamp_exp_all, n_osamp_exp_all_ar)

    #Processing each exposure 
    for isub_exp,(iexp,n_osamp_exp) in enumerate(zip(iexp_list,n_osamp_exp_all_total)):

        #Planets in exposure
        studied_pl_exp = np.array(studied_pl)[cond_transit_all[isub_exp]]

        #Active regions in exposure 
        if cond_ar:ar_in_exp = np.array(studied_ar)[cond_ar_all[isub_exp]]
        else:ar_in_exp = {}
   
        #Initialize averaged and range values
        Focc_star_pl={}
        if 'corr_ar' in args:args['Focc_corr']={}
        if cond_ar:Focc_star_ar={}
        sum_prop_dic={}
        coord_reg_dic={}
        range_dic={}
        line_occ_HP={}
        for subkey_chrom in key_chrom:
            Focc_star_pl[subkey_chrom]=np.zeros(star_I_prop[subkey_chrom]['nw'],dtype=float) 
            if 'corr_ar' in args:args['Focc_corr'][subkey_chrom]=np.zeros(star_I_prop[subkey_chrom]['nw'],dtype=float) 
            if cond_ar:Focc_star_ar[subkey_chrom]=np.zeros(system_ar_prop[subkey_chrom]['nw'],dtype=float) 
            sum_prop_dic[subkey_chrom]={}
            coord_reg_dic[subkey_chrom]={}
            range_dic[subkey_chrom]={}
            line_occ_HP[subkey_chrom]={}
      
            #Initializing dictionary entries for planet
            for pl_loc in studied_pl_exp:
                sum_prop_dic[subkey_chrom][pl_loc]={}
                coord_reg_dic[subkey_chrom][pl_loc]={}
                range_dic[subkey_chrom][pl_loc]={}
                for par_loc in par_list:    
                    sum_prop_dic[subkey_chrom][pl_loc][par_loc]=np.zeros(star_I_prop[subkey_chrom]['nw'],dtype=float)
                    coord_reg_dic[subkey_chrom][pl_loc][par_loc]=np.zeros(star_I_prop[subkey_chrom]['nw'],dtype=float)
                    if par_loc in range_par_list:range_dic[subkey_chrom][pl_loc][par_loc+'_range']=np.tile([1e100,-1e100],[star_I_prop[subkey_chrom]['nw'],1])
                sum_prop_dic[subkey_chrom][pl_loc]['nocc']=0. 
                if ('line_prof' in par_list_in):
                    if (theo_dic['precision'] in ['low','medium']):
                        coord_reg_dic[subkey_chrom][pl_loc]['rv_broad']=np.zeros(star_I_prop[subkey_chrom]['nw'],dtype=float)
                    elif (theo_dic['precision']=='high'):
                        sum_prop_dic[subkey_chrom][pl_loc]['line_prof'] = np.zeros(args['ncen_bins'],dtype=float) 
                        if 'corr_ar' in args:sum_prop_dic[subkey_chrom][pl_loc]['corr_supp'] = np.zeros(args['ncen_bins'],dtype=float)
                    
            #Initializing dictionary entries for active regions
            for ar in ar_in_exp:
                sum_prop_dic[subkey_chrom][ar]={}
                coord_reg_dic[subkey_chrom][ar]={}
                range_dic[subkey_chrom][ar]={}
                for par_loc in par_list:    
                    sum_prop_dic[subkey_chrom][ar][par_loc]=np.zeros(system_ar_prop[subkey_chrom]['nw'],dtype=float)
                    coord_reg_dic[subkey_chrom][ar][par_loc]=np.zeros(system_ar_prop[subkey_chrom]['nw'],dtype=float)
                    if par_loc in range_par_list:range_dic[subkey_chrom][ar][par_loc+'_range']=np.tile([1e100,-1e100],[system_ar_prop[subkey_chrom]['nw'],1])
                sum_prop_dic[subkey_chrom][ar]['nocc']=0. 
                if ('line_prof' in par_list_in):sum_prop_dic[subkey_chrom][ar]['line_prof'] = np.zeros(args['ncen_bins'],dtype=float)
                                        
            #Line profile can be calculated over each stellar cell only in achromatic / closest-achromatic mode 
            if ('line_prof' in par_list_in):line_occ_HP[subkey_chrom] = np.repeat(theo_dic['precision'],star_I_prop[subkey_chrom]['nw'])
            else:line_occ_HP[subkey_chrom] = np.repeat('',star_I_prop[subkey_chrom]['nw'])  
            
        #Theoretical properties from active regions or regions occulted by planets, at exposure center       
        if cond_iexp_proc[isub_exp]:
            coord_oversamp = {'x':{},'y':{}}        
            
            #Planet oversampled positions
            for pl_loc in studied_pl_exp:
            
                #No oversampling
                if n_osamp_exp==1:
                    for ikey,key in enumerate(['x','y']):coord_oversamp[key][pl_loc] = [coord_in[pl_loc]['cen_pos'][ikey,iexp]]
    
                #Theoretical properties from regions occulted by each planet, averaged over full exposure duration  
                #    - only if oversampling is effective for this exposure
                else:
                    for ikey,key in enumerate(['x','y']):coord_oversamp[key][pl_loc] = coord_in[pl_loc]['st_pos'][ikey][iexp]+np.arange(n_osamp_exp)*dcoord_exp_in[key][pl_loc][isub_exp]/(n_osamp_exp-1.)  
                        
            #Active region oversampled positions initialization
            if cond_ar:
    
                #Active region oversampled positions
                for ar in ar_in_exp:
                    
                    #No oversampling
                    if n_osamp_exp==1:
                        for key in ['x','y']:coord_oversamp[key][ar] = [coord_in[ar][key+'_sky_exp'][1,iexp]]
                   
                    #If we want to oversample
                    else:
                        for key in ['x','y']:coord_oversamp[key][ar] = coord_in[ar][key+'_sky_exp'][0,iexp] + np.arange(n_osamp_exp)*dcoord_exp_in[key][ar][isub_exp]/(n_osamp_exp-1.)            

            #Variables to keep track of how many oversampled positions in this exposure were occulting the star
            n_osamp_exp_eff_pl = 0
            n_osamp_exp_eff_ar = 0

            #Loop on oversampled exposure positions 
            #    - after coord_oversamp has been defined for all planets
            #    - if oversampling is not active a single central position is processed
            #    - we neglect the potential chromatic variations of the planet radius and corresponding grid 
            #    - if at least one of the processed planet is transiting
            for iosamp in range(n_osamp_exp):

                #Need to define a reduced version of the active region dictionary in this oversampled position. This is required
                #if we want to account for presence of active regions in planet-occulted regions.
                ar_are_visible = False
                reduced_ar_prop_oversamp={}
                for ar in ar_in_exp:
                    reduced_ar_prop_oversamp[ar]={
                        'fctrst':coord_in[ar]['fctrst'],
                        'ang_rad':coord_in[ar]['ang_rad'],
                    }
                    temp_long = np.arcsin(coord_oversamp['x'][ar][iosamp] / np.cos(coord_in[ar]['lat_rad_exp'][1,iexp]))
                    reduced_ar_prop_oversamp[ar]['cos_lat_exp_center'] = np.cos(coord_in[ar]['lat_rad_exp'][1,iexp])
                    reduced_ar_prop_oversamp[ar]['cos_long_exp_center'] = np.cos(temp_long)
                    reduced_ar_prop_oversamp[ar]['sin_lat_exp_center'] = np.sin(coord_in[ar]['lat_rad_exp'][1,iexp])
                    reduced_ar_prop_oversamp[ar]['sin_long_exp_center'] = np.sin(temp_long)
                    ar_are_visible |= is_ar_visible(par_star['istar_rad'], temp_long, coord_in[ar]['lat_rad_exp'][1,iexp], reduced_ar_prop_oversamp[ar]['ang_rad'], par_star['f_GD'], (1-par_star['f_GD']))

                #------------------------------------------------------------
                #Planet-occulted regions

                #Dictionary telling us which planets have been processed in which chromatic mode and band.
                pl_proc={subkey_chrom:{iband:[] for iband in range(star_I_prop[subkey_chrom]['nw'])} for subkey_chrom in key_chrom}
                cond_occ_pl = False
                for pl_loc in studied_pl_exp:   
                    
                    #Frame conversion of planet coordinates from the classical frame perpendicular to the LOS, to the 'inclined star' frame
                    x_st_sky_pos,y_st_sky_pos,_=frameconv_skyorb_to_skystar(lambda_rad_pl[pl_loc],coord_oversamp['x'][pl_loc][iosamp],coord_oversamp['y'][pl_loc][iosamp],None)      
    
                    #Largest possible square grid enclosing the planet shifted to current planet position     
                    x_st_sky_max = x_st_sky_pos+theo_dic['x_st_sky_grid_pl'][pl_loc]
                    y_st_sky_max = y_st_sky_pos+theo_dic['y_st_sky_grid_pl'][pl_loc]

                    #Calculating properties
                    for subkey_chrom in key_chrom:
                        for iband in range(star_I_prop[subkey_chrom]['nw']):
                            Focc_star_pl[subkey_chrom][iband],cond_occ_pl = calc_occ_region_prop(line_occ_HP[subkey_chrom][iband],cond_occ_pl,iband,args,star_I_prop[subkey_chrom],system_ar_prop[subkey_chrom],iosamp,pl_loc,pl_proc[subkey_chrom][iband],theo_dic['Ssub_Sstar_pl'][pl_loc],x_st_sky_max,y_st_sky_max,star_I_prop[subkey_chrom]['cond_in_RpRs'][pl_loc][iband],par_list,theo_dic['Istar_norm_'+subkey_chrom],\
                                                                                  coord_oversamp['x'],coord_oversamp['y'],lambda_rad_pl,par_star,sum_prop_dic[subkey_chrom][pl_loc],coord_reg_dic[subkey_chrom][pl_loc],range_dic[subkey_chrom][pl_loc],range_par_list,Focc_star_pl[subkey_chrom][iband],cb_band_dic[subkey_chrom][iband],theo_dic, ar_occ=ar_are_visible, reduced_ar_prop=reduced_ar_prop_oversamp)
            
                            #Cumulate line profile from planet-occulted cells
                            #    - in high-precision mode there is a single subkey_chrom and achromatic band, but several planets may have been processed
                            if ('line_prof' in par_list_in):
                                if (theo_dic['precision']=='low'):surf_prop_dic_pl[subkey_chrom][pl_loc]['rv_broad'][iband,isub_exp] = np.max([coord_reg_dic[subkey_chrom][pl_loc]['rv_broad'][iband],surf_prop_dic_pl[subkey_chrom][pl_loc]['rv_broad'][iband,isub_exp]])
                                elif (theo_dic['precision']=='high'):
                                    surf_prop_dic_pl[subkey_chrom]['line_prof'][:,isub_exp]+=sum_prop_dic[subkey_chrom][pl_loc]['line_prof']
                                    if 'corr_ar' in args:surf_prop_dic_pl[subkey_chrom]['corr_supp'][:,isub_exp]+=sum_prop_dic[subkey_chrom][pl_loc]['corr_supp']
                    
                #Star was effectively occulted at oversampled position
                if cond_occ_pl:
                    n_osamp_exp_eff_pl+=1
                    
                    #Calculate line profile from planet-occulted region 
                    #    - profile is scaled to the total flux from current occulted region, stored in coord_reg_dic_pl['Ftot']
                    if ('line_prof' in par_list_in) and (theo_dic['precision']=='medium'):
                        idx_w = {'achrom':range(star_I_prop['achrom']['nw'])}
                        if ('chrom' in key_chrom):idx_w['chrom'] = range(star_I_prop['chrom']['nw'])
                        surf_prop_dic_pl[key_chrom[-1]]['line_prof'][:,isub_exp]+=plocc_prof(args,studied_pl_exp,coord_reg_dic,idx_w,star_I_prop,key_chrom,par_star,theo_dic)

                #------------------------------------------------------------
                #Active regions
                #    - calculated only for differential profiles
                if cond_ar and (args['rout_mode']!='IntrProf'):
                    cond_occ_ar = False
                    
                    #Need to make a new dictionary that contains the active region properties for this oversampled exposure
                    ar_prop_oversamp = {}

                    #Building the active region dictionary - we need to extract this information for all the active regions to find their overlap
                    for ar in ar_in_exp:
                  
                        #Make a rough estimate of the active region grid - has a different resolution than the stellar grid - is in inclined star rest frame
                        x_st_sky_max_ar = coord_oversamp['x'][ar][iosamp] + theo_dic['x_st_sky_grid_ar'][ar]
                        y_st_sky_max_ar = coord_oversamp['y'][ar][iosamp] + theo_dic['y_st_sky_grid_ar'][ar]
                        ar_prop_oversamp[ar] = {
                            'fctrst':coord_in[ar]['fctrst'],
                            'ang_rad':coord_in[ar]['ang_rad'],
                            }
                        ar_prop_oversamp[ar]['x_sky_grid'] = x_st_sky_max_ar
                        ar_prop_oversamp[ar]['y_sky_grid'] = y_st_sky_max_ar
                        ar_prop_oversamp[ar]['lat_rad_exp_center'] = coord_in[ar]['lat_rad_exp'][1,iexp]
                        ar_prop_oversamp[ar]['long_rad_exp_center'] = np.arcsin(coord_oversamp['x'][ar][iosamp] / np.cos(coord_in[ar]['lat_rad_exp'][1,iexp]))
                        ar_prop_oversamp[ar]['cos_long_exp_center'] = np.cos(ar_prop_oversamp[ar]['long_rad_exp_center'])
                        ar_prop_oversamp[ar]['sin_long_exp_center'] = np.sin(ar_prop_oversamp[ar]['long_rad_exp_center'])
                        ar_prop_oversamp[ar]['cos_lat_exp_center'] = np.cos(ar_prop_oversamp[ar]['lat_rad_exp_center'])
                        ar_prop_oversamp[ar]['sin_lat_exp_center'] = np.sin(ar_prop_oversamp[ar]['lat_rad_exp_center'])
                        ar_prop_oversamp[ar]['is_visible'] = is_ar_visible(par_star['istar_rad'], ar_prop_oversamp[ar]['long_rad_exp_center'], ar_prop_oversamp[ar]['lat_rad_exp_center'], ar_prop_oversamp[ar]['ang_rad'], par_star['f_GD'], (1-par_star['f_GD']))
    
                    #Retrieving the properties of the region occulted by each active region
                    for iar, ar in enumerate(ar_in_exp):

                        #List telling us which active regions to be processed for active region overlap
                        ar_proc=np.delete(ar_in_exp, iar)

                        #Going over the chromatic modes
                        for subkey_chrom in key_chrom:
                            
                            #Going over the bands in each chromatic mode
                            for iband in range(system_ar_prop[subkey_chrom]['nw']):
                                Focc_star_ar[subkey_chrom][iband], cond_occ_ar = calc_ar_region_prop(line_occ_HP[subkey_chrom][iband], cond_occ_ar, ar_prop_oversamp, iband, star_I_prop[subkey_chrom], 
                                                                system_ar_prop[subkey_chrom], par_star, ar_proc,ar,theo_dic['Ssub_Sstar_ar'][ar], 
                                                                theo_dic['Ssub_Sstar'], theo_dic['Istar_norm_'+subkey_chrom], sum_prop_dic[subkey_chrom][ar], coord_reg_dic[subkey_chrom][ar],
                                                                range_dic[subkey_chrom][ar], Focc_star_ar[subkey_chrom][iband], par_list, range_par_list, args, cb_band_ar_dic[subkey_chrom][iband])
    
                                #Cumulate line profile from active region cells
                                #    - this is a deviation profile, corresponding to Freg(quiet) - Freg(ar) over 
                                #    - in high-precision mode there is a single subkey_chrom and achromatic band, but several active regions may have been processed
                                if ('line_prof' in par_list_in):
                                    surf_prop_dic_ar[subkey_chrom]['line_prof'][:,isub_exp]+=sum_prop_dic[subkey_chrom][ar]['line_prof']

                    #Star was effectively occulted at oversampled position
                    if cond_occ_ar:
                        n_osamp_exp_eff_ar+=1
                    
            #------------------------------------------------------------

            #Averaged values behind all occulted regions during exposure
            #    - with the oversampling, positions at the center of exposure will weigh more in the sum than those at start and end of exposure, like in reality
            #    - parameters are retrieved in both oversampled/not-oversampled case after they are updated within the sum_prop_dic dictionary 
            #    - undefined values remain set to nan, and are otherwise normalized by the flux from the planet-occulted region
            #    - we use a single Itot as condition that the planet occulted the star
            calc_mean_occ_region_prop(studied_pl_exp,surf_prop_dic_pl,n_osamp_exp_eff_pl,sum_prop_dic,key_chrom,par_list,isub_exp,out_ranges,range_par_list,range_dic)     
            if cond_ar and (args['rout_mode']!='IntrProf'):
                calc_mean_occ_region_prop(ar_in_exp,surf_prop_dic_ar,n_osamp_exp_eff_ar,sum_prop_dic,key_chrom,par_list,isub_exp,out_ranges,range_par_list,range_dic)                            
                            
            #Normalized stellar flux after occultation by all planets and by all active regions
            #    - the intensity from each cell is calculated in the same way as that of the full pre-calculated stellar grid
            if Ftot_star:
                for subkey_chrom in key_chrom:
                    surf_prop_dic_pl[subkey_chrom]['Ftot_star'][:,isub_exp] = 1.
                    surf_prop_dic_common[subkey_chrom]['Ftot_star'][:,isub_exp] = 1.
                    if cond_ar and (args['rout_mode']!='IntrProf'):surf_prop_dic_ar[subkey_chrom]['Ftot_star'][:,isub_exp] = 1.

                    #Planets
                    if n_osamp_exp_eff_pl>0:
                        surf_prop_dic_pl[subkey_chrom]['Ftot_star'][:,isub_exp] -= Focc_star_pl[subkey_chrom]/(n_osamp_exp_eff_pl*theo_dic['Ftot_star_'+subkey_chrom])
                        surf_prop_dic_common[subkey_chrom]['Ftot_star'][:,isub_exp] -= Focc_star_pl[subkey_chrom]/(n_osamp_exp_eff_pl*theo_dic['Ftot_star_'+subkey_chrom])
                    
                    #Active regions
                    if cond_ar and (args['rout_mode']!='IntrProf') and (n_osamp_exp_eff_ar>0):
                        surf_prop_dic_ar[subkey_chrom]['Ftot_star'][:,isub_exp] -= (Focc_star_ar[subkey_chrom])/(n_osamp_exp_eff_ar*theo_dic['Ftot_star_'+subkey_chrom])
                        surf_prop_dic_common[subkey_chrom]['Ftot_star'][:,isub_exp] -= (Focc_star_ar[subkey_chrom])/(n_osamp_exp_eff_ar*theo_dic['Ftot_star_'+subkey_chrom])


            #Local line profiles from current exposure
            if ('line_prof' in par_list_in):

                #Stellar line profile from planet-occulted region
                #    - accounting for both quiet and active cells
                if (n_osamp_exp_eff_pl>0):
                    calc_mean_occ_region_line(theo_dic['precision'],star_I_prop,isub_exp,key_chrom,n_osamp_exp_eff_pl,Focc_star_pl,surf_prop_dic_pl,studied_pl_exp,args,par_star,theo_dic)
                    if 'corr_ar' in args:
                        surf_prop_dic_pl[key_chrom[-1]]['corr_supp'][:,isub_exp]/=n_osamp_exp_eff_pl
                        if args['conv2intr']:surf_prop_dic_pl[key_chrom[-1]]['corr_supp'][:,isub_exp] /= (args['Focc_corr'][key_chrom[-1]]/n_osamp_exp_eff_pl)
 
                #Deviation line profile between quiet and active emission, from active region cells outside of planet-occulted regions
                if cond_ar and (args['rout_mode']=='DiffProf') and (n_osamp_exp_eff_ar > 0):
                    calc_mean_occ_region_line(theo_dic['precision'],star_I_prop,isub_exp,key_chrom,n_osamp_exp_eff_ar,Focc_star_ar,surf_prop_dic_ar,ar_in_exp,args,par_star,theo_dic)

    ### end of exposure            
 
    #Output properties in chromatic mode if calculated in closest-achromatic mode
    if ('line_prof' in par_list_in) and switch_chrom:
        surf_prop_dic_pl = {'chrom':surf_prop_dic_pl['achrom']}
        surf_prop_dic_ar = {'chrom':surf_prop_dic_ar['achrom']}
        surf_prop_dic_common = {'chrom':surf_prop_dic_common['achrom']}

    return surf_prop_dic_pl, surf_prop_dic_ar , surf_prop_dic_common



def calc_mean_occ_region_prop(occulters,surf_prop_dic,n_osamp_exp_eff,sum_prop_dic,key_chrom,par_list,i_in,out_ranges,range_par_list,range_dic):
    r"""**Occulted region: average properties**

    Calculates the properties from the cumulated stellar surface regions occulted during an exposure.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    for occ in occulters:
        if (sum_prop_dic[key_chrom[0]][occ]['Ftot'][0]>0.):
            for subkey_chrom in key_chrom:
                for par_loc in par_list:
        
                    #Total surface ratio and flux from occulted region
                    #    - calculated per band so that the chromatic dependence of the radius can be accounted for also at the stellar limbs, which we cannot do with the input chromatic RpRs
                    #    - averaged over oversampled regions
                    if par_loc in ['SpSstar','Ftot']:surf_prop_dic[subkey_chrom][occ][par_loc][:,i_in] = sum_prop_dic[subkey_chrom][occ][par_loc]/n_osamp_exp_eff

                    #Other surface properties
                    #    - defined as sum( oversampled region , sum( cell , xi*fi )  ) / sum( oversampled region , sum( cell , fi )  )
                    else:surf_prop_dic[subkey_chrom][occ][par_loc][:,i_in] = sum_prop_dic[subkey_chrom][occ][par_loc]/sum_prop_dic[subkey_chrom][occ]['Ftot']
                                                   
                    #Range of values covered during exposures    
                    if out_ranges and (par_loc in range_par_list):
                        surf_prop_dic[subkey_chrom][occ][par_loc+'_range'][:,i_in,:] = range_dic[subkey_chrom][occ][par_loc+'_range']
                        
    return None


def calc_mean_occ_region_line(precision,star_I_prop,i_in,key_chrom,n_osamp_exp_eff,Focc_star,surf_prop_dic,occ_in_exp,args,par_star,theo_dic):
    r"""**Occulted region: average line**

    Calculates the line profile from the cumulated stellar surface regions occulted during an exposure.

    Args:
        TBD
    
    Returns:
        TBD
    
    """     
        
    #Profile from averaged properties over exposures
    if (precision=='low'): 
        idx_w = {'achrom':(range(star_I_prop['achrom']['nw']),i_in)}
        if ('chrom' in key_chrom):idx_w['chrom'] = (range(star_I_prop['chrom']['nw']),i_in)          
        surf_prop_dic[key_chrom[-1]]['line_prof'][:,i_in]=plocc_prof(args,occ_in_exp,surf_prop_dic,idx_w,star_I_prop,key_chrom,par_star,theo_dic)
    
    #Averaged profiles behind all occulted regions during exposure   
    #    - the weighing by stellar intensity is naturally included when applying flux scaling 
    elif (precision in ['medium','high']): 
        surf_prop_dic[key_chrom[-1]]['line_prof'][:,i_in]/=n_osamp_exp_eff

        #Normalization into intrinsic profile
        #    - profiles used to tile the planet-occulted regions have mean unity, and are then scaled by the cell achromatic flux
        #      we normalize by the total planet-occulted flux
        #    - high-precision profile is achromatic
        #    - not required for low- and medium-precision because intrinsic profiles are not scaled to local flux upon calculation in plocc_prof()
        if (theo_dic['precision']=='high') and args['conv2intr']:
            Focc_star_achrom=Focc_star[key_chrom[-1]]/n_osamp_exp_eff
            surf_prop_dic[key_chrom[-1]]['line_prof'][:,i_in] /=Focc_star_achrom   

    return None


def calc_occ_region_prop(line_occ_HP_band,cond_occ,iband,args,star_I_prop,system_ar_prop,idx,pl_loc,pl_proc_band,Ssub_Sstar,x_st_sky_max,y_st_sky_max,cond_in_RpRs,par_list,Istar_norm_band,x_pos_pl,y_pos_pl,lambda_rad_pl,par_star,sum_prop_dic_pl,\
                         coord_reg_dic_pl,range_reg,range_par_list,Focc_star_band,cb_band,theo_dic,ar_occ = False,reduced_ar_prop={}):
    r"""**Occulted region: properties**

    Calculates the average and summed properties from a planet-occulted stellar surface region during an exposure.

    Args:
        line_occ_HP_band (str) : The precision with which to process the exposure.
        cond_occ (bool) : Boolean telling us whether there is an occultation by at least one planet in the oversampled exposure considered.
        iband (int) : Index of the band of interest.
        star_I_prop (dict) : Stellar intensity properties.
        system_ar_prop (dict) : Active region limb-darkening properties.
        idx (int) : Index of the oversampled exposure considered.
        pl_loc (str) : Planet considered.
        pl_proc_band (dict) : Dictionary telling us which planets have been processed in the chromatic mode and band considered.
        Ssub_Sstar (float) : Surface ratio of a planet-occulted region grid cell to a stellar grid cell.
        x_st_sky_max (array) : x coordinates of the maximum square planet-occulted region grid.
        x_st_sky_max (array) : y coordinates of the maximum square planet-occulted region grid.
        cond_in_RpRs (array) : Booleans telling us which cells in the maximum square planet-occulted region grid are within the circular planet-occulted region.
        par_list (list) : List of parameters of interest, whose value in sum_prop_dic_pl will be updated.
        Istar_norm_band (float) : total intensity of the star in the band considered.
        x_pos_pl (float) : x coordinate of the planet in the sky-projected orbital frame.
        y_pos_pl (float) : y coordinate of the planet in the sky-projected orbital frame.
        lambda_rad_pl (float) : Spin-orbit angle of the planet.
        par_star (dict) : Variable stellar parameters.
        sum_prop_dic_pl (dict) : dictionary containing the value of all parameters of interest (par_list), summed over the planet-occulted region in the exposure considered, and for the band of interest.
        coord_reg_dic_pl (dict) : dictionary containing the value of all parameters of interest (par_list), averaged over the planet-occulted region in the exposure considered, and for the band of interest.
        range_reg (dict) : dictionary containing the range of average values the parameters of interest (range_par_list) can take during this exposure.
        range_par_list (list) : list of parameters of interest, whose range of values, stored in range_reg_dic_ar, will be updated.
        Focc_star_band (float) : total flux occulted by the active region in the exposure and band considered. 
        cb_band (list) : Polynomial coefficients used to compute thr RV component of the planet-occulted region due to convective blueshift.
        theo_dic (dict) : parameters used to generate and describe the stellar grid and planet-occulted regions grid.
        ar_occ (bool) : Optional, whether active regions are present in the oversampled exposure considered. Default is False.
        reduced_ar_prop (dict) : Optional, active region properties used to account for the possible presence of active region cells in the planet-occulted region. Default is an empty dictionary.
    
    Returns:
        Focc_star_band (float) : the input Focc_star_band updated with the flux occulted by the planet considered.
        cond_occ (bool) : updated version of the input cond_occ. Tells us whether or not the planet occulted the exposure considered.

    """ 

    parameter_list = deepcopy(par_list)
    range_parameter_list = deepcopy(range_par_list)
        
    #Reduce maximum square planet grid to size of planet in current band
    coord_grid = {}
    coord_grid['x_st_sky']=x_st_sky_max[cond_in_RpRs] 
    coord_grid['y_st_sky']=y_st_sky_max[cond_in_RpRs]   

    #Identifying occulted stellar cells in the sky-projected and star rest frame
    n_pl_occ = calc_st_sky(coord_grid,par_star)

    #Star is effectively occulted
    #    - when the expositions are oversampled, some oversampled positions may put the planet beyond the stellar disk, with no points behind the star
    if n_pl_occ>0:
        cond_occ = True
        
        #Removing current planet cells already processed for previous occultations
        if len(pl_proc_band)>0:
            cond_pl_occ_corr = np.repeat(True,n_pl_occ)
            for pl_prev in pl_proc_band:
    
                #Coordinate of previous planet center in the 'inclined star' frame
                x_st_sky_prev,y_st_sky_prev,_=frameconv_skyorb_to_skystar(lambda_rad_pl[pl_prev],x_pos_pl[pl_prev][idx],y_pos_pl[pl_prev][idx],None)

                #Cells occulted by current planet and not previous ones
                #    - condition is that cells must be beyond previous planet grid in this band
                RpRs_prev = star_I_prop[pl_prev][iband]
                cond_pl_occ_corr &= ( (coord_grid['x_st_sky'] - x_st_sky_prev)**2.+(coord_grid['y_st_sky'] - y_st_sky_prev)**2. > RpRs_prev**2. )
            
            #Reduce grid to remaining cells
            for key in coord_grid:coord_grid[key] = coord_grid[key][cond_pl_occ_corr]
            n_pl_occ = np.sum(cond_pl_occ_corr)
      
        #Store planet as processed in current band
        pl_proc_band+=[pl_loc]  
        coord_grid['nsub_star'] = n_pl_occ   

        #--------------------------------
        #Account for active region occultation in planet-occulted region
        cond_eff_ar = False
        
        #Active regions are visible over the stellar disk
        if ar_occ:
            n_grid = len(coord_grid['x_st_sky'])
            
            #Identify the cells in the planet-occulted region that are active
            #    - 'gen_ar_flag_map' is set to True in every cell of the occulted region that is covered by at least one active region
            #    - 'fctrst_map_ar' is set to the contrast value of the active regions and accounts for overlap between active regions (bright wins)
            count_ar_olap = np.zeros(n_grid, dtype=int)
            fctrst_map_ar = np.zeros(n_grid, dtype=float)
            for ar in list(reduced_ar_prop.keys()):

                #Retrieve coordinates of the planet-occulted region in the inclined star frame
                new_x_sky_grid = coord_grid['x_st_sky']
                new_y_sky_grid = coord_grid['y_st_sky']
                new_z_sky_grid = coord_grid['z_st_sky']
                
                #Move coordinates to the (non-inclined) star frame and then the active region reference frame
                x_st_grid, y_st_grid, z_st_grid = frameconv_skystar_to_star(new_x_sky_grid, new_y_sky_grid, new_z_sky_grid, par_star['istar_rad'])
                x_ar_grid = x_st_grid*reduced_ar_prop[ar]['cos_long_exp_center'] - z_st_grid*reduced_ar_prop[ar]['sin_long_exp_center']
                y_ar_grid = y_st_grid*reduced_ar_prop[ar]['cos_lat_exp_center'] - (z_st_grid*reduced_ar_prop[ar]['cos_long_exp_center'] + x_st_grid*reduced_ar_prop[ar]['sin_long_exp_center']) * reduced_ar_prop[ar]['sin_lat_exp_center']
                cond_in_ar = (x_ar_grid**2. + y_ar_grid**2. <= reduced_ar_prop[ar]['ang_rad']**2)
         
                #Updating flag map
                count_ar_olap[cond_in_ar]+=1

                #Updating contrast map (bright regions win)
                fctrst_map_ar[cond_in_ar] = np.maximum(fctrst_map_ar[cond_in_ar], reduced_ar_prop[ar]['fctrst'])

            #Defining independent properties for quiet and active cells
            gen_ar_flag_map = count_ar_olap.astype(bool)
            cond_eff_ar = np.sum(gen_ar_flag_map)
            if cond_eff_ar:
                for key in ['veq','alpha_rot','beta_rot']:
                    coord_grid[key] = np.repeat(par_star[key],n_pl_occ)
                    coord_grid[key][fctrst_map_ar < 1.] = par_star[key+'_spots']
                    coord_grid[key][fctrst_map_ar >= 1.] = par_star[key+'_faculae']

        #No active regions are overlapping with current planet
        if not cond_eff_ar:
            for key in ['veq','alpha_rot','beta_rot']:coord_grid[key] = par_star[key]

        #--------------------------------
        #Local flux grid over current planet-occulted region, in current band
        _,_,mu_grid_star,Fsurf_grid_star,Ftot_star,_ = calc_Isurf_grid([iband],coord_grid['nsub_star'],star_I_prop,coord_grid,par_star,Ssub_Sstar,Istar_norm = Istar_norm_band,region = 'local',Ssub_Sstar_ref = theo_dic['Ssub_Sstar'])
        coord_grid['mu'] = mu_grid_star[:,0]

        #Accounting for the active region emission
        if cond_eff_ar:
            
            #Update flux-grid for active region cells, accounting for their specific limb-darkening and contrast
            _,_,_,Fsurf_ar_emit_grid,_,_ = calc_Isurf_grid([iband],coord_grid['nsub_star'],system_ar_prop,coord_grid,par_star,Ssub_Sstar,Istar_norm = Istar_norm_band,region = 'local',Ssub_Sstar_ref = theo_dic['Ssub_Sstar'])
            Fsurf_grid_star[gen_ar_flag_map, iband] = Fsurf_ar_emit_grid[gen_ar_flag_map, iband] * fctrst_map_ar[gen_ar_flag_map] 

            #Update total flux from occulted region
            Ftot_star = np.sum(Fsurf_grid_star, axis=0)

        #Scale continuum level
        Fsurf_grid_star*=par_star['cont']
        Ftot_star*=par_star['cont']
       
        #Flux and number of cells occulted from all planets, cumulated over oversampled positions
        Focc_star_band+= Ftot_star[0]   
        sum_prop_dic_pl['nocc']+=coord_grid['nsub_star']
        
        #--------------------------------
        #Active region contamination correction
        if 'corr_ar' in args:
            
            #Re-calculate Local flux grid over current planet-occulted region, in current band
            _,_,_,Fsurf_grid_star_corr,_,_ = calc_Isurf_grid([iband],coord_grid['nsub_star'],star_I_prop,coord_grid,par_star,Ssub_Sstar,Istar_norm = Istar_norm_band,region = 'local',Ssub_Sstar_ref = theo_dic['Ssub_Sstar'])

            #Define variables to store the total flux of the overlap between multiple active regions
            temp_ar = 0.

            #Define number of active cells 
            if ar_occ:args['corr_nsub_ov_ar'] = np.sum(gen_ar_flag_map)
            else:args['corr_nsub_ov_ar'] = 0.

            #If planet-occulted region covers active regions
            if cond_eff_ar:
                _,_,_,Fsurf_ar_emit_grid,_,_ = calc_Isurf_grid([iband],coord_grid['nsub_star'],system_ar_prop,coord_grid,par_star,Ssub_Sstar,Istar_norm = Istar_norm_band,region = 'local',Ssub_Sstar_ref = theo_dic['Ssub_Sstar'])
                args['corr_Far_grid'] = Fsurf_ar_emit_grid[gen_ar_flag_map, iband] * fctrst_map_ar[gen_ar_flag_map]
                args['corr_Fstar_grid_ar'] = Fsurf_grid_star_corr[gen_ar_flag_map, iband]
                for key in ['corr_Far_grid','corr_Fstar_grid_ar']:args[key]=args[key].reshape(len(args[key]),1)
                args['ar_flag_map'] = gen_ar_flag_map
                temp_ar = np.sum(args['corr_Far_grid'] - args['corr_Fstar_grid_ar'], axis=0)*par_star['cont']
                for key in ['corr_Far_grid','corr_Fstar_grid_ar']:args[key]*=par_star['cont']

            #Putting everything together
            args['Focc_corr']['achrom'][iband] += temp_ar

        #--------------------------------
        #Co-adding properties from current region to the cumulated values over oversampled planet positions 
        sum_region_prop(line_occ_HP_band,iband,args,parameter_list,Fsurf_grid_star[:,0],None,coord_grid,Ssub_Sstar,cb_band,range_parameter_list,range_reg,sum_prop_dic_pl,coord_reg_dic_pl,par_star,lambda_rad_pl[pl_loc],None)

    return Focc_star_band,cond_occ




def sum_region_prop(line_occ_HP_band,iband,args,par_list,Fsurf_grid_band,Fsurf_grid_emit_band,coord_grid,Ssub_Sstar,cb_band,range_par_list,range_reg,sum_prop_dic,coord_reg_dic,par_star,lambda_rad_pl_loc,ar_contrast):
    r"""**Planet-occulted or active region properties: calculations**

    Calculates the average and summed properties from a local (planet-occulted or active) stellar surface region during an exposure.
    
    The flux emitted by a local element writes
    
    .. math::  
       dF[\nu] =  I[\nu](\cos{\theta})  dA \, \vec{N}(dA).\vec{N}(\mathrm{LOS})
    
    with :math:`dF[\nu]` emitted in the direction :math:`\vec{N}(\mathrm{LOS})` of the LOS, :math:`dA = R_\mathrm{\star}^2 \sin{\theta} d\theta d\phi` the spherical surface element 
    at the surface of the star, and :math:`\vec{N}(dA)` its normal. By definition :math:`\vec{N}(dA).\vec{N}(\mathrm{LOS}) = \cos{\theta} = \mu` so that
    
    .. math::     
       dF[\nu] =   I[\nu](\cos{\theta}) R_\mathrm{\star}^2 \sin{\theta} d\theta d\phi \cos{\theta}
       
    Which can also write, with :math:`dS = dx dy = dA \cos{\theta}` the projection of `dA` onto the plane of sky (where `XY` is the plane perpendicular to the LOS (`Z`))

    .. math::  
       dF[\nu] =   I[\nu](\cos{\theta}) dS[\theta]
    
    Here the fluxes are normalized by the stellar surface, ie :math:`dS[\theta] = d_\mathrm{grid}^2/(\pi R_\mathrm{\star}^2)`, since :math:`d_\mathrm{grid}` is defined in units of :math:`R_\mathrm{\star}`.
    The total flux emitted by the star in the direction of the LOS is then, with :math:`\mu = \cos(\theta)` and :math:`d\mu = -\sin{\theta} d\theta`
    
    .. math:: 
       F_\mathrm{\star}[\nu] &= ( \int_{\phi = 0}^{2 \pi} \int_{\theta = 0}^{\pi / 2} I[\nu](\cos{\theta}) \sin{\theta} d\theta d\phi \cos{\theta} )/\pi   \\
                             &= 2 \int_{\mu = 0}^{1} I[\nu](\mu) \mu d\mu    \\
                             &= 2 I_0[\nu] \int_{\mu = 0}^{1} \mathrm{LD}(\mu) \mu d\mu \\
                             &= 2 I_0[\nu] \mathrm{Int}_0
                
    If there is no limb-darkening then :math:`\mathrm{Int}_0 = 1/2` and :math:`F_\mathrm{\star}[\nu] = I_0[\nu]`      
    
    In the non-oversampled case
    
     - we add the values, even if each index is only updated once, so that the subroutine can be used directly with oversampling
     - all tables have been initialized to 0
    
    In the oversampled case
    
     - average values are co-added over regions oversampling the total region occulted during an exposure.    
       We update values cumulated during an exposure through every passage through the function.
     - the flux emitted by a surface element is
     
       .. math:: 
          dF(\mu) =   I[\nu](\mu) S_\mathrm{sub}/S_\mathrm{\star} = I[\nu](xy) S_\mathrm{sub}/S_\mathrm{\star}
       
       Here we must consider the flux that is emitted by a surface element during the time `T` it is occulted by the planet
       
       .. math:: 
          dF_\mathrm{occ}(xy) = dF(xy) T_\mathrm{occ}(xy)

       If we assume the planet velocity is constant during the exposure, and it has no latitudinal motion, then 

       .. math:: 
          dF_\mathrm{occ}(xy) = dF(xy) d_\mathrm{occ}(xy)/v_\mathrm{pl}

       Where :math:`d_\mathrm{occ}` is the distance between the planet centers from the first to the last moment it occults the surface element

       .. math:: 
          d_\mathrm{occ}(xy) = \sqrt{ R_\mathrm{p}^2 - y_\mathrm{grid}^2 } = d_\mathrm{occ}(y)
            
     - the weighted mean of a quantity `V` during an exposure would thus be (where :math:`xy_\mathrm{grid}` describes the stellar grid cells occulted during an exposure)

       .. math:: 
          <A> &= \sum_{xy_\mathrm{grid}} V(xy) dF_\mathrm{occ}(xy)  / \sum_{xy_\mathrm{grid}} dF_\mathrm{occ}(xy)    \\
          <A> &= \sum_{xy_\mathrm{grid}} V(xy) I[\nu](xy) S_\mathrm{sub}/S_\mathrm{\star} d_\mathrm{occ}(y)/v_\mathrm{pl} / \sum_{xy_\mathrm{grid}} I[\nu](xy) S_\mathrm{sub}/S_\mathrm{\star} d_\mathrm{occ}(y)/v_\mathrm{pl}    \\  
          <A> &= \sum_{xy_\mathrm{grid}} V(xy) I[\nu](xy) d_\mathrm{occ}(y) / \sum_{xy_\mathrm{grid}} I[\nu](xy) d_\mathrm{occ}(y)   
          
     - instead of discretizing the exact surface occulted by the planet during a full exposure, we place the planet
       and its grid at consecutive positions during the exposure. Between two consecutive positions separated by `d_\mathrm{exp}^\mathrm{oversamp}`,
       the planet spent a time :math:`t_\mathrm{exp}^\mathrm{oversamp} = d_\mathrm{exp}^\mathrm{oversamp}/v_\mathrm{pl}`. If a surface element is occulted by the planet during `N(xy)` consecutive positions, then
       we can write  the total occultation time as 

       .. math:: 
          T_\mathrm{occ}(xy) &= N(xy) t_\mathrm{exp}^\mathrm{oversamp}     \\
          dF_\mathrm{occ}(xy) &= dF(xy) N(xy) d_\mathrm{exp}^\mathrm{oversamp}/v_\mathrm{pl}
           
       the weighted mean of a quantity `V` during an exposure is then

       .. math::        
          <A> &= \sum_{xy_\mathrm{grid}} V(xy) dF_\mathrm{occ}(xy) / \sum_{xy_\mathrm{grid}} dF_\mathrm{occ}(xy)            \\
          <A> &= \sum_{xy_\mathrm{grid}} V(xy) (dF(xy) N(xy) d_\mathrm{exp}^\mathrm{oversamp}/v_\mathrm{pl}) / \sum_{xy_\mathrm{grid}} (dF(xy) N(xy) d_\mathrm{exp}^\mathrm{oversamp}/v_\mathrm{pl})   \\  
          <A> &= \sum_{xy_\mathrm{grid}} V(xy) dF(xy) N(xy) / \sum_{xy_\mathrm{grid}} dF(xy) N(xy)   
            
       ie that we 'add' successively the `N(xy)` times a given surface element flux was occulted. 
       The normalization factor corresponds to :math:`F_\mathrm{tot}^\mathrm{oversamp}`.
       To ensure that this approximation is good, `N(xy)` must be high enough, ie :math:`t_\mathrm{exp}^\mathrm{oversamp}` and :math:`d_\mathrm{exp}^\mathrm{oversamp}` small enough 
     
     - note that :math:`S_\mathrm{sub}/S_\mathrm{\star}` is normalized by :math:`R_\mathrm{\star}^2`, since :math:`d_\mathrm{grid}` is defined from :math:`R_\mathrm{p}/R_\mathrm{\star}`

    Args:
        line_occ_HP_band (str) : The precision with which to process the exposure.
        iband (int) : Index of the band of interest.
        args (dict) : Parameters used to generate the intrinsic profiles.
        par_list (list) : List of parameters of interest, whose value in sum_prop_dic will be updated.
        Fsurf_grid_band (array) : Stellar flux grid over local region in the band of interest.
        coord_grid (dict) : Dictionary of coordinates for the local region.
        Ssub_Sstar (float) : Surface ratio of a local region grid cell to a stellar grid cell.
        cb_band (list) : Polynomial coefficients used to compute thr RV component of the planet-occulted region due to convective blueshift.
        range_par_list (list) : List of parameters of interest, whose range of values, stored in range_reg_dic, will be updated.
        range_reg (dict) : Dictionary containing the range of average values the parameters of interest (range_par_list) can take during this exposure.
        sum_prop_dic (dict) : Dictionary containing the value of all parameters of interest (par_list), summed over the local region in the exposure considered, and for the band of interest.
        coord_reg_dic (dict) : Dictionary containing the value of all parameters of interest (par_list), averaged over the local region in the exposure considered, and for the band of interest.
        par_star (dict) : Fixed/variable stellar parameters.
        lambda_rad_pl_loc (float) : Spin-orbit angle of the planet.
        ar_contrast (float) : Contrast level of the active region considered.

    Returns:
        None
    
    """     
    #Distance from projected orbital normal in the sky plane, in absolute value
    if ('xp_abs' in par_list) or (('coord_line' in args) and (args['coord_line']=='xp_abs')):coord_grid['xp_abs'] = frameconv_skystar_to_skyorb(lambda_rad_pl_loc,coord_grid['x_st_sky'],coord_grid['y_st_sky'],coord_grid['z_st_sky'])[0]  

    #Sky-projected distance from star center
    if ('r_proj' in par_list) or (('coord_line' in args) and (args['coord_line']=='r_proj')):coord_grid['r_proj'] = np.sqrt(coord_grid['r2_st_sky'])                   

    #Processing requested properties
    for par_loc in par_list:
        
        #Occultation ratio
        #    - ratio = Sp/S* = sum(x,sp(x))/(pi*Rstar^2) = sum(x,dp(x)^2)/(pi*Rstar^2) = sum(x,d_surfloc(x)*Rstar^2)/(pi*Rstar^2) = sum(x,d_surfloc(x))/pi
        #      since we use a square grid, sum(x,d_surfloc(x)) = nx*d_surfloc
        #      this quantity is not limb-darkening weighted
        if par_loc=='SpSstar':
            sum_prop_dic[par_loc][iband]+=Ssub_Sstar*coord_grid['nsub_star']
            coord_reg_dic[par_loc][iband] = Ssub_Sstar*coord_grid['nsub_star']
         
        else:
        
            #Flux level from region occulted by the planet alone
            #    - set to 1 since it is weighted by flux afterward
            if par_loc=='Ftot':coord_grid[par_loc] = 1.                    

            #Stellar latitude and longitude (degrees)
            #    - associated with X and Y positions in star frame (units of Rstar)
            # sin(lat) = Ystar / Rstar
            # sin(lon) = Xstar / Rstar
            elif par_loc=='lat':coord_grid[par_loc] = np.arcsin(coord_grid['y_st'])*180./np.pi    
            elif par_loc=='lon':coord_grid[par_loc] = np.arcsin(coord_grid['x_st'])*180./np.pi     
    
            #Stellar line properties with polynomial spatial dependence 
            elif (par_loc in args['linevar_par_vis']):  
                linevar_coord_grid = calc_linevar_coord_grid(args['coord_line'],coord_grid)
                coord_grid[par_loc] = calc_polymodu(args['pol_mode'],args['coeff_line'][par_loc],linevar_coord_grid) 
                
            #Stellar-rotation induced radial velocity (km/s)
            elif par_loc=='Rot_RV':
                coord_grid[par_loc] = calc_RVrot(coord_grid['x_st_sky'],coord_grid['y_st'],par_star['istar_rad'],coord_grid['veq'],coord_grid['alpha_rot'],coord_grid['beta_rot'])[0]
                         
            #Disk-integrated-corrected convective blueshift polynomial (km/s)
            elif par_loc=='CB_RV':coord_grid[par_loc] = np_poly(cb_band)(coord_grid['mu'])          
    
            #Full RV (km/s)
            #    - accounting for an additional constant offset to model jitter or global shifts, and for visit-specific offset to model shifts specific to a given transition
            elif par_loc=='rv':
                coord_grid[par_loc] = deepcopy(coord_grid['Rot_RV']) + par_star['rv']
                if 'CB_RV' in par_list:coord_grid[par_loc]+=coord_grid['CB_RV']
                if 'rv_line' in par_list:coord_grid[par_loc]+=coord_grid['rv_line']
                
            #------------------------------------------------
    
            #Sum property over occulted region, weighted by stellar flux
            #    - we use flux rather than intensity, because local flux level depend on the local region grid resolution
            #    - total RVs from local region is set last in par_list to calculate all rv contributions first:
            # + rotational contribution is always included
            # + disk-integrated-corrected convective blueshift polynomial (in km/s)   
            coord_grid[par_loc+'_sum'] = np.sum(coord_grid[par_loc]*Fsurf_grid_band)
            if par_loc=='xp_abs':coord_grid[par_loc+'_sum'] = np.abs(coord_grid[par_loc+'_sum'])
              
            #Cumulate property over successively occulted regions
            sum_prop_dic[par_loc][iband]+=coord_grid[par_loc+'_sum'] 

            #Total flux from current occulted region
            if par_loc=='Ftot':coord_reg_dic['Ftot'][iband] = coord_grid['Ftot_sum']

            #Calculate average property over current occulted region  
            #    - <X> = sum(cell, xcell*fcell)/sum(cell,fcell)           
            else:coord_reg_dic[par_loc][iband] = coord_grid[par_loc+'_sum']/coord_grid['Ftot_sum'] 

            #Range of values covered during the exposures (normalized)
            #    - for spatial-related coordinates
            if par_loc in range_par_list:
                range_reg[par_loc+'_range'][iband][0]=np.min([range_reg[par_loc+'_range'][iband][0],coord_reg_dic[par_loc][iband]])
                range_reg[par_loc+'_range'][iband][1]=np.max([range_reg[par_loc+'_range'][iband][1],coord_reg_dic[par_loc][iband]])
     
    #------------------------------------------------
    #Calculate line profile from average of cell profiles over current region
    #    - this high precision mode is only possible for achromatic or closest-achromatic mode 
    if line_occ_HP_band=='high':    
        
        #Attribute intrinsic profile to each cell 
        init_st_intr_prof(args,coord_grid,par_star)

        #Whether to use the over-simplified grid building function or not
        use_OS_grid=False
        use_C_OS_grid=False
        if 'OS_grid' in args and args['OS_grid']:use_OS_grid=True
        if 'C_OS_grid' in args and args['C_OS_grid']:
            use_OS_grid=False
            use_C_OS_grid=True

        #Calculate individual local line profiles from all region cells
        #    - analytical intrinsic profiles are fully calculated 
        #      theoretical and measured intrinsic profiles have been pre-defined and are just shifted to their position
        #    - in both cases a scaling is then applied to convert them into local profiles
        if use_OS_grid:
            fit_Fsurf_grid_band = np.tile(Fsurf_grid_band, (args['ncen_bins'], 1)).T
            line_prof_grid=coadd_loc_gauss_prof(coord_grid['rv'],fit_Fsurf_grid_band,args)
        elif use_C_OS_grid:
            line_prof_grid = use_C_coadd_loc_gauss_prof(coord_grid['rv'],Fsurf_grid_band,args)
        else:line_prof_grid=coadd_loc_line_prof(coord_grid['rv'],range(coord_grid['nsub_star']),Fsurf_grid_band,args['flux_intr_grid'],coord_grid['mu'],par_star,args)

        #Coadd quiet line profiles from all elementary cells over current occulted region
        #    - cells can be quiet or within active region
        sum_prop_dic['line_prof'] = np.sum(line_prof_grid,axis=0) 

        #Active region correction
        if ('corr_ar' in args) and (ar_contrast is None):

            #Making temporary variables to store the planet-active region overlap contributions to the correction
            temp_line_ar = np.zeros(sum_prop_dic['corr_supp'].shape, dtype=float)

            if args['corr_nsub_ov_ar'] > 0.:
                #Retrieve the quiet star and active region profiles for the planet-active region overlapping region
                if use_OS_grid:
                    fit_Fstar_grid_band = np.tile(args['corr_Fstar_grid_ar'][:,0], (args['ncen_bins'], 1)).T
                    fit_Far_grid_band = np.tile(args['corr_Far_grid'][:,0], (args['ncen_bins'], 1)).T
                    star_line_prof_grid=coadd_loc_gauss_prof(coord_grid['rv'][args['ar_flag_map']],fit_Fstar_grid_band,args)
                    ar_line_prof_grid=coadd_loc_gauss_prof(coord_grid['rv'][args['ar_flag_map']],fit_Far_grid_band,args)
                elif use_C_OS_grid:
                    star_line_prof_grid = use_C_coadd_loc_gauss_prof(coord_grid['rv'][args['ar_flag_map']],args['corr_Fstar_grid_ar'][:,0],args)
                    ar_line_prof_grid = use_C_coadd_loc_gauss_prof(coord_grid['rv'][args['ar_flag_map']],args['corr_Far_grid'][:,0],args)
                else:
                    star_line_prof_grid=coadd_loc_line_prof(coord_grid['rv'][args['ar_flag_map']],range(args['corr_nsub_ov_ar']),args['corr_Fstar_grid_ar'][:,0],args['flux_intr_grid'][args['ar_flag_map']],coord_grid['mu'][args['ar_flag_map']],par_star,args)
                    ar_line_prof_grid=coadd_loc_line_prof(coord_grid['rv'][args['ar_flag_map']],range(args['corr_nsub_ov_ar']),args['corr_Far_grid'][:,0],args['flux_intr_grid'][args['ar_flag_map']],coord_grid['mu'][args['ar_flag_map']],par_star,args)

                #Putting everything together
                temp_line_ar = np.sum(ar_line_prof_grid, axis=0) - np.sum(star_line_prof_grid, axis=0) 

            sum_prop_dic['corr_supp'] = temp_line_ar

        #Remove active region line profiles from all elementary cells over current occulted region
        #    - condition is only fulfilled when calling sum_region_prop() for active region, not planet-occulted regions
        #    - the final profile from a active region is Freg(quiet) - Freg(ar)
        if ar_contrast is not None:
            if use_OS_grid:
                fit_Fsurf_grid_emit_band = np.tile(Fsurf_grid_emit_band*ar_contrast, (args['ncen_bins'], 1)).T
                emit_line_prof_grid = coadd_loc_gauss_prof(coord_grid['rv'],fit_Fsurf_grid_emit_band,args)
            elif use_C_OS_grid:
                emit_line_prof_grid = use_C_coadd_loc_gauss_prof(coord_grid['rv'],Fsurf_grid_emit_band*ar_contrast,args)
            else:emit_line_prof_grid = coadd_loc_line_prof(coord_grid['rv'],range(coord_grid['nsub_star']),Fsurf_grid_emit_band*ar_contrast,args['flux_intr_grid'],coord_grid['mu'],par_star,args)          
            sum_prop_dic['line_prof'] -= np.sum(emit_line_prof_grid,axis=0)

    #Define rotational broadening of planet-occulted region
    elif line_occ_HP_band in ['low','medium']:
        drv_min = coord_reg_dic['rv'][iband]-np.min(coord_grid['rv'])
        drv_max = np.max(coord_grid['rv'])-coord_reg_dic['rv'][iband] 
        coord_reg_dic['rv_broad'][iband] = 0.5*(drv_min+drv_max)       

    return None






def occ_region_grid(RregRs, nsub_Dreg , planet = True):
    r"""**Local region grid** 

    Defines a square x/y/z grid enclosing a local region of the stellar surface in the 'inclined' star frame, with:
      
     - X axis is parallel to the star equator
     - Y axis is the projected spin axis
     - Z axis is along the LOS

    Args:
        RregRs (float) : the radius of the region in the XY plane. 
                         For a planet the projected region keeps a constant radius in the XY plane and its angular aperture increases toward the limbs.
                         For an active region it is the angular aperture that is fixed and the radius of the projection that decreases toward the limb. 
                         `RregRs` is then the sine of the (half) angle defining the chord of the active region, corresponding to the maximum projected radius of the active region as it would be seen at the center of the star, and defines the largest square enclosing the active region.     
        nsub_Dreg (int) : the number of grid cells desired.
        planet (bool) : Default False. Whether or not to perform additional processing for planet-occulted region grids.
    
    Returns:
        x_st_sky_grid (1D dict) : The x-coordinates of the grid cells.
        y_st_sky_grid (1D dict) : The y-coordinates of the grid cells.
        Ssub_Sstar (float) : The surface of each grid cell.
    
    """ 
    
    #Subcell width (in units of Rstar) and surface (in units of Rstar^2 and pi*Rstar^2) 
    d_sub=2.*RregRs/nsub_Dreg
    Ssub_Sstar=d_sub*d_sub/np.pi

    #Coordinates of points discretizing the enclosing square
    cen_sub=-RregRs+(np.arange(-2, nsub_Dreg+2)+0.5)*d_sub       
    xy_st_sky_grid=np.array(list(it_product(cen_sub,cen_sub)))
    
    #Keeping grid points behind the planet-occulted region
    #    - because the planet-occulted region is a disk with constant radius in the XY plane we can reduce the grid to this disk
    if planet:

        #Distance to region center (squared)
        r_sub_pl2=xy_st_sky_grid[:,0]*xy_st_sky_grid[:,0]+xy_st_sky_grid[:,1]*xy_st_sky_grid[:,1]

        #Keeping only grid points behind the planet
        cond_in_pldisk = ( r_sub_pl2 < RregRs*RregRs)           
        x_st_sky_grid=xy_st_sky_grid[cond_in_pldisk,0]
        y_st_sky_grid=xy_st_sky_grid[cond_in_pldisk,1] 
        r_sub_pl2=r_sub_pl2[cond_in_pldisk] 

    else:
        x_st_sky_grid=xy_st_sky_grid[:,0]
        y_st_sky_grid=xy_st_sky_grid[:,1] 
        r_sub_pl2 = None

    return d_sub,Ssub_Sstar,x_st_sky_grid,y_st_sky_grid,r_sub_pl2


#%% Planet-occulted region routines

def plocc_prof(args,studied_pl,coord_dic,idx_w,star_I_prop,key_chrom,param,theo_dic):
    r"""**Planet-occulted line profile**

    Line profiles can be 
    
     - theoretical (calculated with a stellar atmospheric model)
     - measured (from the input data)
     - analytical (calculated in RV or wavelength space, but over a single line).
    
    Spectral dependence
    
     - chromatic mode: planet-to-star radius ratio and/or stellar intensy are chromatic (spectral mode only)
     - closest-achromatic mode: profile width is smaller than the typical scale of chromatic variations (spectral mode only)
     - achromatic mode: white values are used (default in CCF mode, optional in spectral mode)
      
    Profiles can be calculated at three precision levels
    
     - high. Intrinsic spectra are summed over each occulted cells. This option is only possible in chromatic / closest-achromatic mode (intrinsic spectra cannot be summed over individual cells for each 
       chromatic band, since different parts of the spectra are affected differently)
     - medium. Intrinsic profiles are defined for each occulted region, based on the average region properties, and cumulated.
       In chromatic mode each profile is respectively scaled and shifted using the chromatic RV and flux scaling table from the region 
     - low. Intrinsic profiles are defined for each exposure, based on the average exposure properties.
       In chromatic mode each profile is respectively scaled and shifted using the chromatic RV and flux scaling table from the exposure 
                     
    When several planets are transiting the properties are averaged over the complementary regions they occult, in particular the flux scaling, so that the final profile cumulated over all planets 
    should have the flux corresponding to the summed planet-to-star surface ratios 

    Args:
        TBD
    
    Returns:
        TBD
    
    """     
    data_av_pl={'cen_bins':np.array([args['cen_bins']]),'edge_bins':np.array([args['edge_bins']])}
    
    #In chromatic mode 'key_chrom' contains both chrom and achrom, while in other modes it only contains achrom
    if ('chrom' in key_chrom):chrom_calc = 'chrom'
    else:chrom_calc = 'achrom'
    
    #Profile calculation
    line_prof = np.zeros(args['ncen_bins'],dtype=float)
    for pl_loc in studied_pl:  
        
        #Planet transits
        if ~np.isnan(coord_dic[chrom_calc][pl_loc]['Ftot'][idx_w[chrom_calc]]):
            mu_av = coord_dic['achrom'][pl_loc]['mu'][idx_w['achrom']]
    
            #Chromatic scaling based on current planet occultation and stellar intensity
            #    - single value in achromatic mode
            #    - not applied for the calculation of intrinsic profiles, as chromatic broadband variations are corrected for when extracting measured intrinsic profiles
            if args['conv2intr']:flux_sc_spec = 1.
            else:
                flux_sc = coord_dic[chrom_calc][pl_loc]['Ftot'][idx_w[chrom_calc]]
                if chrom_calc=='achrom':flux_sc_spec = flux_sc[0]
                else:flux_sc_spec = np_interp(args['cen_bins'],star_I_prop['chrom']['w'],flux_sc,left=flux_sc[0],right=flux_sc[-1])
         
            #Analytical profile
            #    - only in achromatic / closest-chromatic mode (no chromatic shift is applied)
            if (args['mode']=='ana'):   
            
                #Surface properties with polynomial variation of chosen coordinate, required to calculate spectral line profile
                args['input_cell_all']={}
                for par_loc in args['linevar_par_vis']:     
                    args['input_cell_all'][par_loc] = coord_dic['achrom'][pl_loc][par_loc][idx_w['achrom']]  
                
                #Calculation of planet-occulted line profile
                data_av_pl['flux'] = [calc_loc_line_prof(0,coord_dic['achrom'][pl_loc]['rv'][idx_w['achrom']],flux_sc_spec,None,mu_av,args,param)]
                
            #Theoretical or measured profile
            elif (args['mode'] in ['theo','Intrbin']):     
                data_loc = {}
                
                #Retrieve measured intrinsic profile at average coordinate for current planet
                if args['mode']=='Intrbin':
                    coordline_av = coord_dic['achrom'][pl_loc][args['coord_line']][idx_w['achrom']]
                    ibin_to_cell = closest(args['cen_dim_Intrbin'],coordline_av)
                    Fintr_av_pl=args['flux_Intrbin'][ibin_to_cell]   
                    data_loc['cen_bins'] = np.array([args['cen_bins_Intrbin']])
                    data_loc['edge_bins'] = np.array([args['edge_bins_Intrbin']])
              
                #Interpolate theoretical profile at average coordinate for current planet
                elif args['mode']=='theo':     
                    Fintr_av_pl = theo_dic['sme_grid']['flux_intr_grid'](mu_av[0])                    
                    data_loc['cen_bins'] = np.array([args['cen_bins_intr']])
                    data_loc['edge_bins'] = np.array([args['edge_bins_intr']])

                #Scaling from intrinsic to planet-occulted flux  
                data_loc['flux'] = np.array([Fintr_av_pl*flux_sc_spec])
                ncen_bins_Intr = len(data_loc['cen_bins'][0])
            
                #Shift profile from average occulted photopshere position for current planet (source, in which it is defined) to star rest frame (receiver) 
                #    - 'conv2intr' is True when the routine is used to calculate model intrinsic profiles (flux_sc_spec is then 1) to be compared with measured intrinsic profiles
                #       since the latter were corrected for chromatic deviations upon extraction, the present model profiles are only shifted with the achromatic planet-occulted rv 
                #    - otherwise a chromatic shift is applied
                if args['conv2intr']:dic_rv = {'achrom':{pl_loc:{'rv': np.reshape(coord_dic['achrom'][pl_loc]['rv'][idx_w['achrom']],[star_I_prop['achrom']['nw'],1]) }}}
                else:dic_rv = {chrom_calc:{pl_loc:{'rv': np.reshape(coord_dic[chrom_calc][pl_loc]['rv'][idx_w[chrom_calc]],[star_I_prop[chrom_calc]['nw'],1]) }}}
                rv_surf_star,rv_surf_star_edge = def_surf_shift('theo',dic_rv,0,data_loc,pl_loc,args['type'],star_I_prop,[1,ncen_bins_Intr],1,ncen_bins_Intr)  
                if rv_surf_star_edge is not None:
                    rv_shift_edge = -rv_surf_star_edge
                    spec_dopshift_edge = gen_specdopshift(rv_surf_star_edge)
                else:
                    rv_shift_edge = None
                    spec_dopshift_edge = None
                data_av_pl=align_data(data_loc,args['type'],1,args['dim_exp'],args['resamp_mode'],data_av_pl['cen_bins'],data_av_pl['edge_bins'],-rv_surf_star,gen_specdopshift(rv_surf_star),rv_shift_edge = rv_shift_edge,spec_dopshift_edge = spec_dopshift_edge, nocov = True)

            #Rotational broadening
            #    - for a planet large enough, the distribution of surface RV over the occulted region acts produces rotational broadening
            if coord_dic[chrom_calc][pl_loc]['rv_broad'][idx_w[chrom_calc]][0]>0.:
                FWHM_broad = coord_dic[chrom_calc][pl_loc]['rv_broad'][idx_w[chrom_calc]][0]
                
                #Convert into spectral broadening (in A)                    
                if ('spec' in args['type']):FWHM_broad*=args['ref_conv']/c_light
                    
                #Rotational kernel
                data_av_pl['flux'][0] = convol_prof(data_av_pl['flux'][0],args['cen_bins'],FWHM_broad)  

            #Co-add contribution from current planet
            line_prof+=data_av_pl['flux'][0]
            
    return line_prof



def init_surf_shift(gen_dic,inst,vis,data_dic,align_mode):
    r"""**Planet-occulted rv**

    Returns measured or theoretical planet-occulted rv

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Set surface RV to measured achromatic value  
    #    - only in-transit profiles profiles for which the local stellar line was flagged as detected can be aligned  
    if (align_mode=='meas'):
        dic_rv = dataload_npz(gen_dic['save_data_dir']+'Introrig_prop/'+inst+'_'+vis)
        idx_aligned = np_where1D(dic_rv['cond_detected'])
        ref_pl=None

    #Set surface RVs to model
    #    - all in-transit profiles can be aligned, as rv can be calculated at any phase
    elif (align_mode=='theo'):        
        dic_rv = dataload_npz(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis)
        idx_aligned = np.arange(data_dic[inst][vis]['n_in_tr'])   

        #Reference planet
        #    - theoretical velocities are calculated for all planets transiting in a given visit
        if len(data_dic[inst][vis]['studied_pl'])==1:ref_pl = data_dic[inst][vis]['studied_pl'][0]  
        else:ref_pl=data_dic['Intr']['align_ref_pl'][inst][vis]
    
    return ref_pl,dic_rv,idx_aligned


def def_surf_shift(align_mode,dic_rv,i_in,data_exp,pl_ref,data_type,star_I_prop,dim_exp,nord,nspec):    
    r"""**Planet-occulted rv shifts**

    Returns rv shifts of stellar surface in star rest frame, from measured or theoretical planet-occulted rv

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Set surface RV to measured achromatic value 
    if (align_mode=='meas'):
        rv_surf_star=dic_rv[i_in]['rv']
        rv_surf_star_edge=None
    
    #Set surface RVs to model
    elif ('theo' in align_mode):

        #Chromatic RVs
        #    - theoretical RVs calculated for the broadband RpRs and intensity values provided as inputs are interpolated over the table of each exposure, so that 
        # each bin is shifted with the surface rv corresponding to its wavelength   
        #    - if data has been converted from spectra to CCF, nominal properties will be used
        if ('spec' in data_type) and ('chrom' in dic_rv):
            rv_surf_star=np.zeros(dim_exp,dtype=float)*np.nan
            rv_surf_star_edge=np.zeros([nord,nspec+1],dtype=float)*np.nan
            
            #Absolute or chromatic-to-nominal relative RV
            if align_mode=='theo':RV_shift_pl = dic_rv['chrom'][pl_ref]['rv'][:,i_in]
            elif align_mode=='theo_rel':RV_shift_pl = dic_rv['chrom'][pl_ref]['rv'][:,i_in]-dic_rv['achrom'][pl_ref]['rv'][0,i_in]              
            for iord in range(nord):
                rv_surf_star[iord] = np_interp(data_exp['cen_bins'][iord],star_I_prop['chrom']['w'],RV_shift_pl,left=RV_shift_pl[0],right=RV_shift_pl[-1])
                rv_surf_star_edge[iord] = np_interp(data_exp['edge_bins'][iord],star_I_prop['chrom']['w'],RV_shift_pl,left=RV_shift_pl[0],right=RV_shift_pl[-1])
            
        #Achromatic RV defined for the nominal transit properties
        else:
            rv_surf_star = dic_rv['achrom'][pl_ref]['rv'][0,i_in]  
            rv_surf_star_edge=None

    return rv_surf_star,rv_surf_star_edge

def calc_plocced_tiles(pl_prop, x_sky_grid, y_sky_grid):
    r"""**'Planet-occulted' tiles** 
    
    Args:
        pl_prop (dict) : relevant planet properties.
        x_sky_grid (1D array) : x coordinates of the stellar grid in the inclined star frame. (at st, cen, and end)
        y_sky_grid (1D array) : y coordinates of the stellar grid in the inclined star frame. (at st, cen, and end)
     
    Returns:
        cond_in_pl (1D array) : array of booleans telling us which cells in the stellar grid are occulted by the planet.

    """          
    
    #Cells occulted in at least one exposure
    cond_in_pl = np.zeros(len(x_sky_grid), dtype=bool)

    #Processing all input exposures
    for i in range(len(pl_prop['x_orb_exp'])):

        #Conversion of planet coordinates from sky-projected orbital to sky-projected stellar frame
        x_sky_exp,y_sky_exp,_ = frameconv_skyorb_to_skystar(pl_prop['lambda'],pl_prop['x_orb_exp'][i],pl_prop['y_orb_exp'][i],0)
     
        #Planet-occulted cells in current exposure
        pos_cond_close_to_pl = (x_sky_grid - x_sky_exp)**2 + (y_sky_grid - y_sky_exp)**2 < pl_prop['RpRs']**2  
        cond_in_pl |=   pos_cond_close_to_pl

    return cond_in_pl



#%% Active region routines

def generate_ar_prop(mock_dic, data_dic, gen_dic):
    r"""**Automatic active region generation**
    
    Generates distribution of active regions and updates relevant dictionaries. 
    The properties of each region is randomly drawn from a uniform or gaussian distribution. 
    
    Args:
        mock_dic (dict) : mock dictionary.
        data_dic (dict) : data dictionary.
        gen_dic (dict)  : general dictionary.
    
    Returns:
        None
    
    """  

    #Initializing the mock dictionary if not done previously
    for inst in mock_dic['auto_gen_ar']:
        if inst not in mock_dic['ar_prop']:mock_dic['ar_prop'][inst]={}
        for vis in mock_dic['auto_gen_ar'][inst]:
            if vis not in mock_dic['ar_prop'][inst]:mock_dic['ar_prop'][inst][vis]={}
            auto_gen_dic = mock_dic['auto_gen_ar'][inst][vis]
            
            #Iterating over the number of active regions to update the dictionary
            for iac in range(auto_gen_dic['num']):

                #Making active region name
                ac_reg_name = 'gen_ar'+str(iac+1)

                #Looping over all relevant properties
                for prop in ['fctrst','lat','Tc_ar','ang']:

                    #Retrieve dictionary
                    prop_dic = auto_gen_dic[prop]

                    #Drawing from distributions provided
                    if prop_dic['distrib']=='uf':prop_gen_val = np.random.uniform(low = prop_dic['low'], high = prop_dic['high'])
                    elif prop_dic['distrib']=='gauss':prop_gen_val = np.random.normal(loc = prop_dic['val'], scale = prop_dic['s_val'])
                    else:stop('Unrecognized distribution.')

                    #Updating mock properties
                    mock_dic['ar_prop'][inst][vis].update({
                    prop+'__IS'+inst+'_VS'+vis+'_AR'+ac_reg_name : prop_gen_val,
                        })
        
                #Updating LD properties
                if data_dic['DI']['ar_prop'] == {}:data_dic['DI']['ar_prop']['achrom'] = {}
                if ac_reg_name not in data_dic['DI']['ar_prop']['achrom']:data_dic['DI']['ar_prop']['achrom'][ac_reg_name]=[mock_dic['ar_prop'][inst][vis]['ang__IS'+inst+'_VS'+vis+'_AR'+ac_reg_name] * np.pi/180]  
       
                #Updating triggers
                if ac_reg_name not in gen_dic['studied_ar']:gen_dic['studied_ar'].update({ac_reg_name : {inst : [vis]}}) 

    #Finishing construction of LD dictionary 
    data_dic['DI']['ar_prop']['achrom'].update({'LD':data_dic['DI']['system_prop']['achrom']['LD']})
    for ideg in range(1,5):data_dic['DI']['ar_prop']['achrom'].update({'LD_u'+str(ideg):data_dic['DI']['system_prop']['achrom']['LD_u'+str(ideg)]})

    return None



def retrieve_ar_prop_from_param(param, inst, vis): 
    r"""**Active region parameters: retrieval and formatting**

    Transforms a dictionary with 'raw' active region properties in the format param_ISinstrument_VSvisit_ARarname to a more convenient active region dictionary of the form : 
        
     ar_prop = { arname : {'lat' : , 'Tc_ar' : , ....}}
    
    The formatted dictionary contains the initial active region properties as well as additional derived active region properties, such as the longitude and latitude of the active region as well as 
    its visibility criterion (see `is_ar_visible`).
    We assume active region parameter are never defined as common across multiple visits / instruments.

    Args:
        param (dict) : 'raw' active region properties.
        inst (str) : instrument considered. Should match the instrument in the 'raw' active region parameter name (see format above).
        vis (str) : visit considered. Should match the visit in the 'raw' active region parameter name (see format above).
     
    Returns:
        contamin_prop (dict) : formatted active region dictionary.

    """ 

    #Initializing necessary dictionary/list
    contamin_prop = {}
    for par in param : 

        # Parameter is active region-related and linked to the right visit and instrument
        if (('_IS_' in par) or ('_IS'+inst in par)) and (('_VS'+vis in par) or ('_VS_' in par)) and ('_AR' in par):
            contamin_par = par.split('__IS')[0] 
            contamin_name = par.split('_AR')[1]
            if contamin_name not in contamin_prop : contamin_prop[contamin_name] = {}
            contamin_prop[contamin_name][contamin_par] = param[par]

    #Processing active regions
    contamin_list = list(contamin_prop.keys())
    contamin_prop['ar'] = list(np.unique(contamin_list))
    for contamin_name in contamin_prop['ar']:

        #Active region latitude in radians
        contamin_prop[contamin_name]['lat_rad'] = contamin_prop[contamin_name]['lat']*np.pi/180.

        #Store properties common across the exposure
        contamin_prop[contamin_name]['ang_rad'] = contamin_prop[contamin_name]['ang']*np.pi/180

    return contamin_prop



def calc_ar_tiles(ar_prop, ang_rad, x_sky_grid, y_sky_grid, z_sky_grid, grid_dic, star_param, use_grid_dic = False, disc_exp = True) :
    r"""**'Active region' tiles** 

    Identification of which cells on the provided grid are covered by active regions. Two methods are available: 
        
    - use_grid_dic = False : calculation will be performed on the x_sky_grid, y_sky_grid, z_sky_grid, by moving these grids from the inclined star frame to the star rest frame.
                            This option can be used for identifiying active region stellar tiles, when istar is fitted.
    - use_grid_dic = True : calculation will be performed on the grid contained in grid_dic['x/y/z_st'], which is already in star rest frame (no frame conversion needed). 

    To identify which cells in the provided grid are 'active', we move the grid cells to the active region reference frame and check which cells are within the active region. 
    This is a necessary step to obtain the correct shape for the active regions, as the spherical nature of the stellar surface must be accounted for. The calculation is as follows:
    
    Using the x/y/z grid in the star rest frame (obtained either with a frame conversion function or from the grid dictionary), we rotate the x/y/z grid around the stellar spin axis 
    to the longitude of the active region

    .. math::
        & x_{\mathrm{ar}'} = x_{\mathrm{st}} \mathrm{cos}(long) - z_{\mathrm{st}} \mathrm{sin}(long)
        & y_{\mathrm{ar}'} = y_{\mathrm{st}}
        & z_{\mathrm{ar}'} = x_{\mathrm{st}} \mathrm{sin}(long) + z_{\mathrm{st}} \mathrm{cos}(long)

    We then rotate the new grid to the latitude of the active region to obtain the x/y/z grid in the active region rest frame
    
    .. math::
        & x_{\mathrm{ar}} = x_{\mathrm{ar}'}
        & y_{\mathrm{ar}} = y_{\mathrm{ar}'} \mathrm{cos}(lat) - z_{\mathrm{ar}'} \mathrm{sin}(lat)
        & z_{\mathrm{ar}} = y_{\mathrm{ar}'} \mathrm{sin}(lat) + z_{\mathrm{ar}'} \mathrm{cos}(lat)
    
    Finally, we check which cells are within the active region

    .. math::
        & arctan2(\sqrt{x_{\mathrm{ar}}^{2} + y_{\mathrm{ar}}^{2}}, z_{\mathrm{ar}}) < R_{\mathrm{ar}}

    With R_{\mathrm{ar}} the angular size of the active region in radians.


    Args:
        ar_prop (dict) : formatted active region properties (see retrieve_ar_prop_from_param).
        x_sky_grid (1D array) : x coordinates of the stellar grid in the inclined star frame.
        y_sky_grid (1D array) : y coordinates of the stellar grid in the inclined star frame.
        z_sky_grid (1D array) : z coordinates of the stellar grid in the inclined star frame.
        grid_dic (dict) : dictionary containing the x/y/z grids in various reference frames, including the star rest frame and inclined star frame.
        star_param (dict) : star properties.
        use_grid_dic (bool) : whether or not to use the grid_dic provided to retrieve the x/y/z grids in the star rest frame. Turned off by default.
        disc_exp (bool) : whether we use the start, center and end of the exposures to figure out if and where the star is covered by the active region, or just the center.
     
    Returns:
        ar_within_grid (bool) : a finer estimate of the active region visibility that can be obtained with is_ar_visible. Essentially, it tells us if at least one tile on the grid is 'active'.
        cond_in_ar (1D array) : array of booleans telling us which cells in the original grid are 'active'.
    
    """                                      

    if disc_exp:positions = range(len(ar_prop['x_sky_exp']))
    else:positions = [1]
    cond_in_ar = np.zeros(len(x_sky_grid), dtype=bool)
    for pos in positions:
        if use_grid_dic :
            cond_close_to_ar = (grid_dic['x_st_sky'] - ar_prop['x_sky_exp'][pos])**2 + (grid_dic['y_st_sky'] - ar_prop['y_sky_exp'][pos])**2 < ang_rad**2
            x_st_grid, y_st_grid, z_st_grid = grid_dic['x_st'][cond_close_to_ar], grid_dic['y_st'][cond_close_to_ar], grid_dic['z_st'][cond_close_to_ar]
            
            
        else :  
            cond_close_to_ar = (x_sky_grid - ar_prop['x_sky_exp'][pos])**2 + (y_sky_grid - ar_prop['y_sky_exp'][pos])**2 < ang_rad**2
            x_st_grid, y_st_grid, z_st_grid = frameconv_skystar_to_star(x_sky_grid[cond_close_to_ar],y_sky_grid[cond_close_to_ar],z_sky_grid[cond_close_to_ar],np.arccos(star_param['cos_istar']))
        
        # Retrieve angular coordinates of active region
        cos_long, sin_long, cos_lat, sin_lat = ar_prop['cos_long_exp'][pos], ar_prop['sin_long_exp'][pos], ar_prop['cos_lat_exp'][pos], ar_prop['sin_lat_exp'][pos]
        
        # Calculate coordinates in active region rest frame
        x_ar =                       x_st_grid*cos_long - z_st_grid*sin_long
        y_ar = y_st_grid*cos_lat  - (x_st_grid*sin_long + z_st_grid*cos_long)   *   sin_lat
        z_ar = y_st_grid*sin_lat  + (x_st_grid*sin_long + z_st_grid*cos_long)   *   cos_lat
        
        # Deduce which cells are within the active region
        phi_ar = np.arctan2(np.sqrt(x_ar**2. + y_ar**2.),z_ar)
        pos_cond_in_ar = cond_close_to_ar
        pos_cond_in_ar[cond_close_to_ar] = (phi_ar <= ang_rad)
        cond_in_ar |= pos_cond_in_ar

    # Check if at least one tile is within the active region
    ar_within_grid = (True in cond_in_ar)   

    return ar_within_grid, cond_in_ar





def calc_ar_region_prop(line_occ_HP_band,cond_occ,contamin_prop,iband,star_I_prop,system_contamin_prop, par_star, proc_band, elem_consid, Ssub_Sstar, Ssub_Sstar_ref, Istar_norm_band, sum_prop_dic,\
                                    coord_reg_dic, range_reg, Focc_star_band, par_list, range_par_list, args, cb_band) :
    
    r"""**Active region properties: define and update**
    
    Identify the active region in each exposure provided and calculate its properties. 
    Accounts for the overlap of active regions by making the brightest region win the overlap.
    Update the provided dictionaries which contain the average and sum of the properties of interest over the active region.

    Args:
        line_occ_HP_band (str) : the precision with which to process the exposure.
        cond_occ (bool) : whether there is an occultation by at least one active region in the exposure considered.
        contamin_prop (dict) : formatted active region properties dictionary (see retrieve_ar_prop_from_param).
        iband (int) : index of the band used to retrieve the corresponding planet and active region limb-darkening properties.
        star_I_prop (dict) : quiet star intensity properties.
        system_contamin_prop (dict) : active region limb-darkening properties.        
        par_star (dict) : star properties.
        proc_band (list) : active regions to be processed to account for the overlap of active regions.
        elem_consid (str) : name of the active region being processed.
        Ssub_Sstar (float): surface of grid cells in the active region-occulted region grid.
        Ssub_Sstar_ref (float) : surface of grid cells in the stellar grid.
        Istar_norm_band (float) : total intensity of the star in the band considered.
        sum_prop_dic (dict) : dictionary containing the value of all parameters of interest (par_list), summed over the active region in the exposure considered, and for the band of interest.
        coord_reg_dic (dict) : dictionary containing the value of all parameters of interest (par_list), averaged over the active region in the exposure considered, and for the band of interest.
        range_reg (dict) : dictionary containing the range of average values the parameters of interest (range_par_list) can take during this exposure.
        Focc_star_band (float) : total flux occulted by the active region in the exposure and band considered. 
        par_list (list) : List of parameters of interest, whose value in sum_prop_dic will be updated.
        range_par_list (list) : list of parameters of interest, whose range of values, stored in range_reg, will be updated.
        args (dict) : parameters used to generate the intrinsic profiles.
        cb_band (list) : coefficients used to calculate the convective blueshift RV contribution.  

    Returns:
        Focc_star_band (float) : the input Focc_star_band updated with the flux occulted by the active region considered.
        cond_occ (bool) : updated version of the input cond_occ. Tells us whether or not the active region is visible in the exposure considered.
    
    """ 
    
    parameter_list = deepcopy(par_list)
    range_parameter_list = deepcopy(range_par_list)
    
    #We have as input a grid discretizing the active region.
    #We have a condition to find the cells in the input grid that are in the stellar grid.
    cond_in_star = contamin_prop[elem_consid]['x_sky_grid']**2 + contamin_prop[elem_consid]['y_sky_grid']**2 < 1.

    #We have a condition to figure out which cells in this input grid are occulted.
    ##Take the cells that are in the stellar grid.
    new_x_sky_grid = contamin_prop[elem_consid]['x_sky_grid'][cond_in_star]
    new_y_sky_grid = contamin_prop[elem_consid]['y_sky_grid'][cond_in_star]

    #Retrieve the z-coordinate for the cells.
    new_z_sky_grid = np.sqrt(1 - new_x_sky_grid**2 - new_y_sky_grid**2)

    #Move coordinates to the star reference frame and then the active region reference frame
    x_st_grid, y_st_grid, z_st_grid = frameconv_skystar_to_star(new_x_sky_grid, new_y_sky_grid, new_z_sky_grid, par_star['istar_rad'])
    x_contamin_grid = x_st_grid*contamin_prop[elem_consid]['cos_long_exp_center'] - z_st_grid*contamin_prop[elem_consid]['sin_long_exp_center']
    y_contamin_grid = y_st_grid*contamin_prop[elem_consid]['cos_lat_exp_center'] - (z_st_grid*contamin_prop[elem_consid]['cos_long_exp_center'] + x_st_grid*contamin_prop[elem_consid]['sin_long_exp_center']) * contamin_prop[elem_consid]['sin_lat_exp_center']
    cond_in = x_contamin_grid**2. + y_contamin_grid**2. <= contamin_prop[elem_consid]['ang_rad']**2

    #Making our refined active region grid and calculating the number of points it has
    contamin_x_sky_grid = new_x_sky_grid[cond_in]
    contamin_y_sky_grid = new_y_sky_grid[cond_in]
    contamin_z_sky_grid = new_z_sky_grid[cond_in]
    n_occ = np.sum(cond_in)

    #Accounting for active region - active region overlap -- the region with the highest contrast wins
    for prev in proc_band:
        if contamin_prop[prev]['fctrst'] > contamin_prop[elem_consid]['fctrst']:
        
            #Move current active region coordinates to the star reference frame and then the other active region's reference frame
            updated_x_st_grid, updated_y_st_grid, updated_z_st_grid = frameconv_skystar_to_star(contamin_x_sky_grid, contamin_y_sky_grid, contamin_z_sky_grid, par_star['istar_rad'])
            x_prev_grid = updated_x_st_grid*contamin_prop[prev]['cos_long_exp_center'] - updated_z_st_grid*contamin_prop[prev]['sin_long_exp_center']
            y_prev_grid = updated_y_st_grid*contamin_prop[prev]['cos_lat_exp_center'] - (updated_z_st_grid*contamin_prop[prev]['cos_long_exp_center'] + updated_x_st_grid*contamin_prop[prev]['sin_long_exp_center']) * contamin_prop[prev]['sin_lat_exp_center']
            cond_in_prev = x_prev_grid**2. + y_prev_grid**2 <= system_contamin_prop[prev][iband]**2

            #Remove cells that overlap
            contamin_x_sky_grid = contamin_x_sky_grid[~cond_in_prev]
            contamin_y_sky_grid = contamin_y_sky_grid[~cond_in_prev]
            contamin_z_sky_grid = contamin_z_sky_grid[~cond_in_prev]
            n_occ -= np.sum(cond_in_prev)

    #--------------------------------

    #Figure out the number of cells occulted and store it
    if n_occ > 0:
        cond_occ = True

    #Making the grid of coordinates for the calc_Isurf_grid() function.
    coord_grid = {}
    
    #Getting, x, y, z, sky-projected radius, and number of occulted cells.
    coord_grid['x_st_sky'] = contamin_x_sky_grid
    coord_grid['y_st_sky'] = contamin_y_sky_grid
    coord_grid['z_st_sky'] = contamin_z_sky_grid
    coord_grid['r2_st_sky']=coord_grid['x_st_sky']*coord_grid['x_st_sky']+coord_grid['y_st_sky']*coord_grid['y_st_sky']
    coord_grid['nsub_star'] = n_occ

    #Getting the coordinates in the star rest frame
    coord_grid['x_st'], coord_grid['y_st'], coord_grid['z_st'] = frameconv_skystar_to_star(coord_grid['x_st_sky'], coord_grid['y_st_sky'], coord_grid['z_st_sky'], par_star['istar_rad'])
    
    #Velocity properties
    for key in ['veq','alpha_rot','beta_rot']:
        if contamin_prop[elem_consid]['fctrst'] < 1.:coord_grid[key] = par_star[key+'_spots']
        else:coord_grid[key] = par_star[key+'_faculae']
    
    #--------------------------------    
    
    #Retrieve the quiet stellar flux grids over this local occulted-region grid.
    _,_,mu_grid_occ,Fsurf_grid_occ,Ftot_occ,_ = calc_Isurf_grid([iband], coord_grid['nsub_star'], star_I_prop, coord_grid, par_star, Ssub_Sstar, Istar_norm_band, region='local', Ssub_Sstar_ref=Ssub_Sstar_ref)

    #Retrieve the flux grid for the active region's emission (since active regions have different LD law compared to the 'quiet' stellar surface)
    _,_,_,Fsurf_grid_emit,Ftot_emit,_ = calc_Isurf_grid([iband], coord_grid['nsub_star'], system_contamin_prop, coord_grid, par_star, Ssub_Sstar, Istar_norm_band, region='local', Ssub_Sstar_ref=Ssub_Sstar_ref)

    #Scale the flux grid to the desired level
    Fsurf_grid_occ *= par_star['cont']
    Fsurf_grid_emit *= par_star['cont']
    Ftot_occ *= par_star['cont']
    Ftot_emit *= par_star['cont']

    #--------------------------------

    #Updating the provided dictionaries 
    coord_grid['mu'] = mu_grid_occ[:,iband]
    Focc_star_band += Ftot_occ[0] - (Ftot_emit[0]*contamin_prop[elem_consid]['fctrst'])
    sum_prop_dic['nocc'] += coord_grid['nsub_star']
    
    #Distance from projected orbital normal in the sky plane, in absolute value
    if 'xp_abs' in parameter_list : parameter_list.remove('xp_abs')
    if 'xp_abs' in range_parameter_list : range_parameter_list.remove('xp_abs') 

    #--------------------------------
    #Co-adding properties from current region to the cumulated values over oversampled active region positions 
    sum_region_prop(line_occ_HP_band,iband,args,parameter_list,Fsurf_grid_occ[:,iband],Fsurf_grid_emit[:,iband],coord_grid,Ssub_Sstar,cb_band,range_parameter_list,range_reg,sum_prop_dic,coord_reg_dic,par_star,None,contamin_prop[elem_consid]['fctrst']) 
    
    return Focc_star_band, cond_occ


