#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from itertools import product as it_product
from copy import deepcopy
import lmfit
from lmfit import Parameters
from ..ANTARESS_grids.ANTARESS_coord import frameconv_skyorb_to_skystar,frameconv_skystar_to_skyorb,frameconv_skystar_to_star,calc_pl_coord,frameconv_star_to_skystar,calc_zLOS_oblate
from ..ANTARESS_process.ANTARESS_data_align import align_data
from ..ANTARESS_analysis.ANTARESS_inst_resp import convol_prof
from ..ANTARESS_grids.ANTARESS_star_grid import calc_CB_RV,get_LD_coeff,calc_st_sky,calc_Isurf_grid,calc_RVrot
from ..ANTARESS_analysis.ANTARESS_model_prof import calc_polymodu,polycoeff_def
from ..ANTARESS_grids.ANTARESS_prof_grid import coadd_loc_line_prof,coadd_loc_gauss_prof,calc_loc_line_prof,init_st_intr_prof,calc_linevar_coord_grid, use_C_coadd_loc_gauss_prof
from ..ANTARESS_general.utils import stop,closest,np_poly,npint,np_interp,np_where1D,datasave_npz,dataload_npz,gen_specdopshift,check_data
from ..ANTARESS_general.constant_data import Rsun,c_light


def calc_plocc_spot_prop(system_param,gen_dic,theo_dic,coord_dic,inst,vis,data_dic,calc_pl_atm=False,spot_dic={}):
    r"""**Planet-occulted / spot properties: workflow**

    Calls function to calculate theoretical properties of the regions occulted by all transiting planets and/or spotted. 

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
 
    #Check for spots
    if (spot_dic != {}) and (inst in spot_dic['spots_prop']):
        txt_spot = ' and spotted '
        cond_spot = True
    else:
        txt_spot = ' '
        cond_spot = False
    
    print('   > Calculating properties of planet-occulted'+txt_spot+'regions')    
    if gen_dic['calc_theoPlOcc']:
        print('         Calculating data')
        
        #Theoretical properties of spotted regions
        params = deepcopy(system_param['star'])
        params['use_spots']=cond_spot
            
        #Theoretical properties of planet occulted-regions
        #    - calculated for the nominal and broadband planet properties 
        #    - for the nominal properties we retrieve the range of some properties covered by the planet during each exposures
        #    - chromatic transit required if local profiles are in spectral mode  
        params.update({'rv':0.,'cont':1.})
        par_list=['rv','CB_RV','mu','lat','lon','x_st','y_st','SpSstar','xp_abs','r_proj']
        key_chrom = ['achrom']
        if ('spec' in data_dic[inst][vis]['type']) and ('chrom' in data_dic[inst][vis]['system_prop']):key_chrom+=['chrom']

        #Calculate properties
        plocc_prop,spot_prop,common_prop = sub_calc_plocc_spot_prop(key_chrom,{},par_list,data_dic[inst][vis]['transit_pl'],system_param,theo_dic,data_dic[inst][vis]['system_prop'],params,coord_dic[inst][vis],gen_dic[inst][vis]['idx_in'], system_spot_prop_in = data_dic['DI']['spots_prop'], out_ranges=True)
        
        #Save spot-occulted region properties
        if cond_spot:
            datasave_npz(gen_dic['save_data_dir']+'Introrig_prop/Spot_Prop_'+inst+'_'+vis,spot_prop)    

            #Save properties combined from planet-occulted and spotted regions
            datasave_npz(gen_dic['save_data_dir']+'Introrig_prop/Common_Prop_'+inst+'_'+vis,common_prop) 

        #Save planet-occulted region properties
        datasave_npz(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis,plocc_prop) 

    else:
        check_data({'path':gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis})        

    return None






def up_plocc_prop(inst,vis,args,param_in,transit_pl,ph_grid,coord_grid, transit_spots=[]):
    r"""**Planet-occulted and spotted region properties: update**

    Updates properties of the planet-occulted region, planetary orbit, and spotted region for fitted step. 

    Args:
        inst (str) : Instrument considered.
        vis (str) : Visit considered. 
        args (dict) : Additional parameters needed to evaluate the fitted function.
        param_in (dict) : Model parameters for the fitted step considered.
        transit_pl (list) : Transiting planets for the instrument and visit considered.
        ph_grid (dict) : Dictionary containing the phase of each planet.
        coord_grid (dict) : Dictionary containing the various coordinates of each planet and spot (e.g., exposure time, exposure x/y/z coordinate).
        transit_spots (list) : Spots present for the instrument and visit considered.
    
    Returns:
        system_param_loc (dict) : System (star+planet+spot) properties.
        coords (dict) : Updated planet and spot coordinates.
        param (dict) : Model parameter names and values.
    
    """ 
    system_param_loc=deepcopy(args['system_param'])
   
    #In case param_in is defined as a Parameters structure, retrieve values and define dictionary
    param={}
    if isinstance(param_in,lmfit.parameter.Parameters):
        for par in param_in:param[par]=param_in[par].value
    else:param=param_in

    #Update stellar parameters if necessary
    if 'cos_istar' in args['var_par_list']:
        system_param_loc['star']['cos_istar'] = param['cos_istar']
        system_param_loc['star']['istar_rad'] = np.arccos(param['cos_istar'])

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
    for pl_loc in transit_pl:

        #Recalculate planet grid if relevant
        if args['fit_RpRs'] and ('RpRs__pl'+pl_loc in args['var_par_list']):
            args['system_prop']['achrom'][pl_loc][0]=param['RpRs__pl'+pl_loc] 
            args['grid_dic']['Ssub_Sstar_pl'][pl_loc],args['grid_dic']['x_st_sky_grid_pl'][pl_loc],args['grid_dic']['y_st_sky_grid_pl'][pl_loc],r_sub_pl2=occ_region_grid(args['system_prop']['achrom'][pl_loc][0],args['grid_dic']['nsub_Dpl'][pl_loc])  
            args['system_prop']['achrom']['cond_in_RpRs'][pl_loc] = [(r_sub_pl2<args['system_prop']['achrom'][pl_loc][0]**2.)]        

        #Recalculate planet coordinates if relevant        
        if args['fit_orbit']:
            coords[pl_loc]={}
            pl_params_loc = system_param_loc[pl_loc]
            
            #Update fitted system properties for current step 
            if ('lambda_rad__pl'+pl_loc in args['genpar_instvis']):lamb_name = 'lambda_rad__pl'+pl_loc+'__IS'+inst+'_VS'+vis 
            else:lamb_name = 'lambda_rad__pl'+pl_loc 
            if (lamb_name in args['var_par_list']):pl_params_loc['lambda_rad'] = param[lamb_name]                     
            if ('inclin_rad__pl'+pl_loc in args['var_par_list']):pl_params_loc['inclin_rad']=param['inclin_rad__pl'+pl_loc]       
            if ('aRs__pl'+pl_loc in args['var_par_list']):pl_params_loc['aRs']=param['aRs__pl'+pl_loc]  
            
            #Calculate coordinates
            #    - start/end phase have been set to None if no oversampling is requested, in which case start/end positions are not calculated
            if args['grid_dic']['d_oversamp'] is not None:phases = ph_grid[pl_loc]
            else:phases = ph_grid[pl_loc][1]
            x_pos_pl,y_pos_pl,_,_,_,_,_,_,ecl_pl = calc_pl_coord(pl_params_loc['ecc'],pl_params_loc['omega_rad'],pl_params_loc['aRs'],pl_params_loc['inclin_rad'],phases,args['system_prop']['achrom'][pl_loc][0],pl_params_loc['lambda_rad'],system_param_loc['star'])
            if args['grid_dic']['d_oversamp'] is not None:
                coords[pl_loc]['st_pos'] = np.vstack((x_pos_pl[0],y_pos_pl[0]))
                coords[pl_loc]['cen_pos'] = np.vstack((x_pos_pl[1],y_pos_pl[1]))
                coords[pl_loc]['end_pos'] = np.vstack((x_pos_pl[2],y_pos_pl[2]))
            else:coords[pl_loc]['cen_pos'] = np.vstack((x_pos_pl,y_pos_pl))
            coords[pl_loc]['ecl'] = ecl_pl

    #Process spots
    if len(transit_spots)>0:

        #Set up properties of spotted regions for the spot coordinate retrieval in sub_calc_plocc_spot_prop
        for spot in transit_spots:
            
            #Recalculate spot grid if relevant
            if spot in args['fit_spot_ang']:
                args['system_spot_prop']['achrom'][spot][0]=param['ang__IS'+inst+'_VS'+vis+'_SP'+spot] * np.pi/180
                _,args['grid_dic']['Ssub_Sstar_sp'][spot],args['grid_dic']['x_st_sky_grid_sp'][spot],args['grid_dic']['y_st_sky_grid_sp'][spot],_ = occ_region_grid(args['system_spot_prop']['achrom'][spot][0],args['grid_dic']['nsub_Dspot'][spot],spot=True)  

    
        #Recalculate spot coordinates if relevant        
        if args['fit_spot']:
    
            #Update spot crossing time before doing spot parameters' retrieval
            for par in param:
                if ('Tcenter' in par) and args['update_crosstime']:param[par] += args['spot_crosstime_supp'][inst][vis]

            #Initialize entries for spot coordinates 
            for spot in transit_spots:
                coords[spot] = {}
                for key in ['Tcenter', 'ang', 'ang_rad', 'lat', 'ctrst']:coords[spot][key] = np.zeros(len(coords['bjd']),dtype=float)*np.nan
                for key in ['lat_rad_exp','sin_lat_exp','cos_lat_exp','long_rad_exp','sin_long_exp','cos_long_exp','x_sky_exp','y_sky_exp','z_sky_exp']:coords[spot][key] = np.zeros([3,len(coords['bjd'])],dtype=float)*np.nan
                coords[spot]['is_visible'] = np.zeros([3,len(coords['bjd'])],dtype=bool)
                
            #Retrieving the spot coordinates for all the times that we have
            for ifit_tstamp, fit_tstamp in enumerate(coords['bjd']):                
                spots_prop = retrieve_spots_prop_from_param(system_param_loc['star'],param,inst,vis,fit_tstamp,exp_dur=coords['t_dur'][ifit_tstamp])
                for spot in transit_spots:

                    for key in ['Tcenter', 'ang', 'ang_rad', 'lat', 'ctrst']:
                        coords[spot][key][ifit_tstamp] = spots_prop[spot][key]

                    for key in ['lat_rad_exp','sin_lat_exp','cos_lat_exp','long_rad_exp','sin_long_exp','cos_long_exp','x_sky_exp','y_sky_exp','z_sky_exp']:
                        coords[spot][key][:, ifit_tstamp] = [spots_prop[spot][key+'_start'],spots_prop[spot][key+'_center'],spots_prop[spot][key+'_end']]

                    coords[spot]['is_visible'][:, ifit_tstamp] = [spots_prop[spot]['is_start_visible'],spots_prop[spot]['is_center_visible'],spots_prop[spot]['is_end_visible']]

        #Trigger use of spots in the function computing the DI profile deviation
        param['use_spots']=True

    #Useful if spots are present but the spot parameters are fixed
    else:param['use_spots']=False

    return system_param_loc,coords,param




def sub_calc_plocc_spot_prop(key_chrom,args,par_list_gen,transit_pl,system_param,theo_dic,system_prop_in,param,coord_in,iexp_list,system_spot_prop_in={},out_ranges=False,Ftot_star=False):
    r"""**Planet-occulted and spot properties: exposure**

    Calculates average theoretical properties of the stellar surface occulted by all transiting planets and/or spotted during an exposure
    
     - we normalize all quantities by the flux emitted by the occulted regions
     - all positions are in units of :math:`R_\star` 
    
    Args:
        key_chrom (list) : chromatic modes used (either chromatic, 'chrom', achromatic, 'achrom', or both).
        args (dict) : parameters used to generate analytical profiles.
        par_list_gen (list) : parameters whose value we want to calculate over each planet-occulted/spotted region.
        transit_pl (list) : list of transiting planets in the exposures considered.
        system_param (dict) : system (star + planet + spot) properties.
        theo_dic (dict) : parameters used to generate and describe the stellar grid and planet/spot-occulted regions grid.
        system_prop_in (dict) : planet limb-darkening properties.
        param (dict) : fitted or fixed star/planet/spot properties.
        coord_in (dict) : dictionary containing the various coordinates of each planet and spot (e.g., exposure time, exposure phase, exposure x/y/z coordinate)
        iexp_list (list) : exposures to process.
        system_spot_prop_in (dict) : optional, spot limb-darkening properties.
        out_ranges (bool) : optional, whether or not to calculate the range of values the parameters of interest (par_list_gen) will take. Turned off by default.
        Ftot_star (bool) : optional, whether or not to calculate the normalized stellar flux after accounting for the spot/planet occultations. Turned off by default.
    
    Returns:
        surf_prop_dic_pl (dict) : average value of all the properties of interest over all the planet-occulted regions, in each exposure and chromatic mode considered.
        surf_prop_dic_spot (dict) : average value of all the properties of interest over all the spotted regions, in each exposure and chromatic mode considered.
        surf_prop_dic_common (dict) : average value of all the properties of interest considering the contributions from both the planet-occulted and spotted regions, in each exposure and chromatic mode considered.

    """ 
    system_prop = deepcopy(system_prop_in)
    system_spot_prop = deepcopy(system_spot_prop_in)
    par_list_in = deepcopy(par_list_gen)
    n_exp = len(iexp_list)
    
    #Spot condition
    #    - if spots are being used, and they are in the param dictionary provided (which is not always the case)  
    if 'use_spots' in param.keys() and param['use_spots']:cond_spot = True
    else:cond_spot = False    

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
            if (args['edge_bins'][-1]-args['edge_bins'][0]<system_prop['chrom']['med_dw']):
                key_chrom=['achrom']
                switch_chrom = True
                iband_cl = closest(system_prop['chrom']['w'],np.median(args['cen_bins']))
                for key in ['w','LD','GD_wmin','GD_wmax','GD_dw']:system_prop['achrom'][key] = [system_prop['chrom'][key][iband_cl]]
                for pl_loc in transit_pl:
                    system_prop['achrom']['cond_in_RpRs'][pl_loc] = [system_prop['chrom']['cond_in_RpRs'][pl_loc][iband_cl]] 
                    system_prop['achrom'][pl_loc] = [system_prop['chrom'][pl_loc][iband_cl]]                

            #Profiles covers a wide spectral band
            #    - requires calculation of achromatic properties                
            else:
                if (args['mode']=='ana'):stop('Analytical model not suited for wide spectral bands') 
                if (theo_dic['precision']=='high'):stop('High precision not possible for wide spectral bands') 
                key_chrom=['achrom','chrom']

    #Calculation of achromatic and/or chromatic values
    surf_prop_dic_pl = {}
    surf_prop_dic_spot = {}
    surf_prop_dic_common = {}
    for subkey_chrom in key_chrom:
        surf_prop_dic_pl[subkey_chrom] = {}        
        surf_prop_dic_spot[subkey_chrom] = {}
    if 'line_prof' in par_list_in:
        for subkey_chrom in key_chrom:
            surf_prop_dic_pl[subkey_chrom]['line_prof']=np.zeros([args['ncen_bins'],n_exp],dtype=float)
            surf_prop_dic_spot[subkey_chrom]['line_prof']=np.zeros([args['ncen_bins'],n_exp],dtype=float)

    #Properties to be calculated
    #    - properties in 'param' have the nominal values from system properties only if the property was not defined in the model property dictionary from settings 
    par_star = deepcopy(param)
    par_list = ['Ftot']
    for par_loc in par_list_in:
        if par_loc=='rv':
            par_list+=['Rot_RV']
            if ('rv_line' in args['linevar_par_vis']):par_list+=['rv_line']
            if ('Rstar' in par_star) and ('Peq' in par_star):par_star['veq'] = 2.*np.pi*par_star['Rstar']*Rsun/(par_star['Peq']*24.*3600.)
        elif (par_loc not in ['line_prof']):par_list+=[par_loc]
    cos_istar = (par_star['cos_istar']-(1.)) % 2 - 1.   #Reset cos_istar within -1 : 1
    par_star['istar_rad']=np.arccos(cos_istar)
    cb_band_dic = {}
    if cond_spot:cb_band_spot_dic = {}
    for subkey_chrom in key_chrom:

        #Disk-integrated stellar flux
        if Ftot_star:
            surf_prop_dic_pl[subkey_chrom]['Ftot_star']=np.zeros([system_prop[subkey_chrom]['nw'],n_exp])*np.nan 
            if cond_spot:
                surf_prop_dic_spot[subkey_chrom]['Ftot_star']=np.zeros([system_prop[subkey_chrom]['nw'],n_exp])*np.nan 
                surf_prop_dic_common[subkey_chrom]['Ftot_star']=np.ones([system_prop[subkey_chrom]['nw'],n_exp])

        #Convective blueshift
        #    - physically, it makes sense for us to define different CB coefficients for a spot since spotted regions are regions of magnetic suppression and would have different CB.
        #However, we make the simplifying assumption that the c1_CB, c2_CB, and c3_CB coefficient are the same for the spotted region as for the quiet star regions, with c0_CB being
        #the only coefficient that varies, and which is calculated with the same condition as before. 
        #Even though our assumption is not correct, we think that the RV shift induced by the difference in CB for the spot can be captured in the RV parameter used to describe the
        #line profiles with which the spotted region is tiled.
        cb_band_dic[subkey_chrom]={}  
        if cond_spot:cb_band_spot_dic[subkey_chrom] = {}  
        if ('CB_RV' in par_list) or ('c0_CB' in par_list):     
            surf_prop_dic_pl[subkey_chrom]['c0_CB']=np.zeros(system_prop[subkey_chrom]['nw'])*np.nan
            if cond_spot:surf_prop_dic_spot[subkey_chrom]['c0_CB']=np.zeros(system_prop[subkey_chrom]['nw'])*np.nan
            for iband in range(system_prop[subkey_chrom]['nw']):
                cb_band_dic[subkey_chrom][iband] = calc_CB_RV(get_LD_coeff(system_prop[subkey_chrom],iband),system_prop[subkey_chrom]['LD'][iband],par_star['c1_CB'],par_star['c2_CB'],par_star['c3_CB'],par_star) 
                surf_prop_dic_pl[subkey_chrom]['c0_CB'][iband]=cb_band_dic[subkey_chrom][iband][0] 
                if cond_spot:
                    cb_band_spot_dic[subkey_chrom][iband] = calc_CB_RV(get_LD_coeff(system_spot_prop[subkey_chrom],iband),system_spot_prop[subkey_chrom]['LD'][iband],par_star['c1_CB'],par_star['c2_CB'],par_star['c3_CB'],par_star) 
                    surf_prop_dic_spot[subkey_chrom]['c0_CB'][iband]=cb_band_spot_dic[subkey_chrom][iband][0]
        else:
            for iband in range(system_prop[subkey_chrom]['nw']):
                cb_band_dic[subkey_chrom][iband] = None
                if cond_spot:cb_band_spot_dic[subkey_chrom][iband] = None
    if 'rv' in par_list_in:par_list+=['rv']  #must be placed after all other RV contributions

    #List of parameters whose range we're interested in
    range_par_list=[]
    if (len(theo_dic['d_oversamp'])>0) and out_ranges:range_par_list = list(np.intersect1d(['mu','lat','lon','x_st','y_st','xp_abs','r_proj'],par_list))
 
    #Initializing spot variables
    #    - must be initialized in anyy case since they will be called later, even if spots are not activated.
    cond_spots_all = np.zeros([n_exp,1], dtype=bool)
    list_spot_names = []

    #Initializing list that will contain the oversampled step for spots, if they are oversampled.
    dcoord_exp_in_sp = {'x':{},'y':{},'z':{}}

    #Initialize a list which will tell us the oversampling rate for each exposure
    n_osamp_exp_all_sp = np.repeat(1,n_exp)

    #Define spot properties
    if cond_spot:
        
        #High precision is required for spots
        if (theo_dic['precision']!='high'):stop('High precision required for spots')
        
        #Figure out the number of spots
        n_spots = 0
        for key in coord_in.keys():
            if 'spot' in key:
                n_spots+=1
                list_spot_names.append(key)
        
        #Initialize the dictionary that will contain spot presence
        cond_spots_all = np.zeros([n_exp,n_spots], dtype=bool)

        #Looping over all exposures
        for isub_exp, iexp in enumerate(iexp_list):

            #Check if at least one spot is visible.
            #    - to do so, we need a more precise estimate of the spot location.
            spot_within_grid_all=np.zeros(n_spots, dtype=bool)

            #Go through the spots and see if they are *roughly* visible.
            for spot_index, spot in enumerate(list_spot_names):

                #See if spot is visible at the center of the exposure.
                if (np.sum(coord_in[spot]['is_visible'][:, iexp])>=1):

                    #Need to make a dictionary of spot coordinates which will be used in calc_spotted_tiles.
                    mini_spot_dic = {}
                    mini_spot_dic['x_sky_exp_start'],mini_spot_dic['x_sky_exp_center'],mini_spot_dic['x_sky_exp_end']=coord_in[spot]['x_sky_exp'][:, iexp]
                    mini_spot_dic['y_sky_exp_start'],mini_spot_dic['y_sky_exp_center'],mini_spot_dic['y_sky_exp_end']=coord_in[spot]['y_sky_exp'][:, iexp]
                    mini_spot_dic['ang_rad']=coord_in[spot]['ang_rad'][iexp]
                    mini_spot_dic['cos_long_exp_start'],mini_spot_dic['cos_long_exp_center'],mini_spot_dic['cos_long_exp_end']=coord_in[spot]['cos_long_exp'][:, iexp]
                    mini_spot_dic['sin_long_exp_start'],mini_spot_dic['sin_long_exp_center'],mini_spot_dic['sin_long_exp_end']=coord_in[spot]['sin_long_exp'][:, iexp]
                    mini_spot_dic['cos_lat_exp_start'],mini_spot_dic['cos_lat_exp_center'],mini_spot_dic['cos_lat_exp_end']=coord_in[spot]['cos_lat_exp'][:, iexp]
                    mini_spot_dic['sin_lat_exp_start'],mini_spot_dic['sin_lat_exp_center'],mini_spot_dic['sin_lat_exp_end']=coord_in[spot]['sin_lat_exp'][:, iexp]
    
                    #See if spot is *precisely* visible.
                    spot_within_grid, _ = calc_spotted_tiles(mini_spot_dic,theo_dic['x_st_sky'], theo_dic['y_st_sky'], theo_dic['z_st_sky'], theo_dic,par_star, True)
                    if spot_within_grid:
                        spot_within_grid_all[spot_index]=True

                    #Check if oversampling is turned on for this spot and force all spots to have same oversampling rate
                    if len(theo_dic['n_oversamp_spot'])>0 and theo_dic['n_oversamp_spot'][spot]>0:
                        for key in ['x','y','z']:dcoord_exp_in_sp[key][spot] = coord_in[spot][key+'_sky_exp'][2,iexp] - coord_in[spot][key+'_sky_exp'][0,iexp]
                        n_osamp_exp_all_sp[isub_exp] = np.maximum(n_osamp_exp_all_sp[isub_exp], theo_dic['n_oversamp_spot'][spot])

                    #Spot-dependent properties - initialize dictionaries
                    for subkey_chrom in key_chrom:
                        surf_prop_dic_spot[subkey_chrom][spot]={}
                        for par_loc in par_list:
                            surf_prop_dic_spot[subkey_chrom][spot][par_loc]=np.zeros([system_spot_prop[subkey_chrom]['nw'],n_exp])*np.nan        
                        for par_loc in range_par_list:surf_prop_dic_spot[subkey_chrom][spot][par_loc+'_range']=np.zeros([system_spot_prop[subkey_chrom]['nw'],n_exp,2])*np.nan

            #Update cond_spots_all
            cond_spots_all[isub_exp]=spot_within_grid_all

    #If spots are not present, need to initialize the spot LD dictionary entry for later purposes
    else:
        for subkey_chrom in key_chrom:system_spot_prop[subkey_chrom]={}

    #Occulted planet zones properties
    n_osamp_exp_all = np.repeat(1,n_exp)
    lambda_rad_pl = {}
    dx_exp_in={}
    dy_exp_in={}
    cond_transit_all = np.zeros([n_exp,len(transit_pl)],dtype=bool)
    for ipl,pl_loc in enumerate(transit_pl):
        
        #Check for planet transit
        if np.sum(np.abs(coord_in[pl_loc]['ecl'][iexp_list])!=1.)>0:
            cond_transit_all[:,ipl]|=(np.abs(coord_in[pl_loc]['ecl'][iexp_list])!=1.)   

            #Obliquities for multiple planets
            #    - for now only defined for a single planet if fitted  
            #    - the nominal lambda has been overwritten in 'system_param[pl_loc]' if fitted
            lambda_rad_pl[pl_loc]=system_param[pl_loc]['lambda_rad']
            
            #Exposure distance (Rstar) 
            if len(theo_dic['d_oversamp'])>0:
                dx_exp_in[pl_loc]=abs(coord_in[pl_loc]['end_pos'][0,iexp_list]-coord_in[pl_loc]['st_pos'][0,iexp_list])
                dy_exp_in[pl_loc]=abs(coord_in[pl_loc]['end_pos'][1,iexp_list]-coord_in[pl_loc]['st_pos'][1,iexp_list])
             
                #Number of oversampling points for current exposure  
                #    - for each exposure we take the maximum oversampling all planets considered 
                if pl_loc in theo_dic['d_oversamp']:
                    d_exp_in = np.sqrt(dx_exp_in[pl_loc]**2 + dy_exp_in[pl_loc]**2)
                    n_osamp_exp_all=np.maximum(n_osamp_exp_all,npint(np.round(d_exp_in/theo_dic['d_oversamp'][pl_loc]))+1)
                    
            #Planet-dependent properties
            for subkey_chrom in key_chrom:
                surf_prop_dic_pl[subkey_chrom][pl_loc]={}
                for par_loc in par_list:
                    surf_prop_dic_pl[subkey_chrom][pl_loc][par_loc]=np.zeros([system_prop[subkey_chrom]['nw'],n_exp])*np.nan        
                for par_loc in range_par_list:surf_prop_dic_pl[subkey_chrom][pl_loc][par_loc+'_range']=np.zeros([system_prop[subkey_chrom]['nw'],n_exp,2])*np.nan
                if ('line_prof' in par_list_in) and (theo_dic['precision']=='low'):
                    surf_prop_dic_pl[subkey_chrom][pl_loc]['rv_broad']=-1e100*np.ones([system_prop[subkey_chrom]['nw'],n_exp])

    #Figuring out which exposures are occulted (by spots or planets)
    cond_transit = np.sum(cond_transit_all,axis=1)>0
    cond_spotted = np.sum(cond_spots_all, axis=1)>0
    cond_iexp_proc = cond_spotted|cond_transit

    #Enforcing a common oversampling factor to the spots and planets
    n_osamp_exp_all_total = np.maximum(n_osamp_exp_all, n_osamp_exp_all_sp)

    #Processing each exposure 
    for isub_exp,(iexp,n_osamp_exp) in enumerate(zip(iexp_list,n_osamp_exp_all_total)):
   
        #Planets in exposure
        transit_pl_exp = np.array(transit_pl)[cond_transit_all[isub_exp]]

        #Spots in exposure 
        if cond_spot:spots_in_exp = np.array(list_spot_names)[cond_spots_all[isub_exp]]
        else:spots_in_exp = {}
   
        #Initialize averaged and range values
        Focc_star_pl={}
        if cond_spot:Focc_star_sp={}
        sum_prop_dic={}
        coord_reg_dic={}
        range_dic={}
        line_occ_HP={}
        for subkey_chrom in key_chrom:
            Focc_star_pl[subkey_chrom]=np.zeros(system_prop[subkey_chrom]['nw'],dtype=float) 
            if cond_spot:Focc_star_sp[subkey_chrom]=np.zeros(system_spot_prop[subkey_chrom]['nw'],dtype=float) 
            sum_prop_dic[subkey_chrom]={}
            coord_reg_dic[subkey_chrom]={}
            range_dic[subkey_chrom]={}
            line_occ_HP[subkey_chrom]={}
            
            #Initializing dictionary entries for planet
            for pl_loc in transit_pl_exp:
                sum_prop_dic[subkey_chrom][pl_loc]={}
                coord_reg_dic[subkey_chrom][pl_loc]={}
                range_dic[subkey_chrom][pl_loc]={}
                for par_loc in par_list:    
                    sum_prop_dic[subkey_chrom][pl_loc][par_loc]=np.zeros(system_prop[subkey_chrom]['nw'],dtype=float)
                    coord_reg_dic[subkey_chrom][pl_loc][par_loc]=np.zeros(system_prop[subkey_chrom]['nw'],dtype=float)
                    if par_loc in range_par_list:range_dic[subkey_chrom][pl_loc][par_loc+'_range']=np.tile([1e100,-1e100],[system_prop[subkey_chrom]['nw'],1])
                sum_prop_dic[subkey_chrom][pl_loc]['nocc']=0. 
                if ('line_prof' in par_list_in):
                    if (theo_dic['precision'] in ['low','medium']):
                        coord_reg_dic[subkey_chrom][pl_loc]['rv_broad']=np.zeros(system_prop[subkey_chrom]['nw'],dtype=float)
                    elif (theo_dic['precision']=='high'):
                        sum_prop_dic[subkey_chrom][pl_loc]['line_prof'] = np.zeros(args['ncen_bins'],dtype=float) 
                    
            #Initializing dictionary entries for spots
            for spot in spots_in_exp:
                sum_prop_dic[subkey_chrom][spot]={}
                coord_reg_dic[subkey_chrom][spot]={}
                range_dic[subkey_chrom][spot]={}
                for par_loc in par_list:    
                    sum_prop_dic[subkey_chrom][spot][par_loc]=np.zeros(system_spot_prop[subkey_chrom]['nw'],dtype=float)
                    coord_reg_dic[subkey_chrom][spot][par_loc]=np.zeros(system_spot_prop[subkey_chrom]['nw'],dtype=float)
                    if par_loc in range_par_list:range_dic[subkey_chrom][spot][par_loc+'_range']=np.tile([1e100,-1e100],[system_spot_prop[subkey_chrom]['nw'],1])
                sum_prop_dic[subkey_chrom][spot]['nocc']=0. 
                if ('line_prof' in par_list_in):sum_prop_dic[subkey_chrom][spot]['line_prof'] = np.zeros(args['ncen_bins'],dtype=float)
                                        
            #Line profile can be calculated over each stellar cell only in achromatic / closest-achromatic mode 
            if ('line_prof' in par_list_in):line_occ_HP[subkey_chrom] = np.repeat(theo_dic['precision'],system_prop[subkey_chrom]['nw'])
            else:line_occ_HP[subkey_chrom] = np.repeat('',system_prop[subkey_chrom]['nw'])  
            
        #Theoretical properties from spotted regions or regions occulted by planets, at exposure center       
        if cond_iexp_proc[isub_exp]:        
            x_oversamp_pl={}
            y_oversamp_pl={}
            
            #Planet oversampled positions
            for pl_loc in transit_pl_exp:
            
                #No oversampling
                if n_osamp_exp==1:
                    x_oversamp_pl[pl_loc] = [coord_in[pl_loc]['cen_pos'][0,iexp]]
                    y_oversamp_pl[pl_loc] = [coord_in[pl_loc]['cen_pos'][1,iexp]]
    
                #Theoretical properties from regions occulted by each planet, averaged over full exposure duration  
                #    - only if oversampling is effective for this exposure
                else:
                    x_oversamp_pl[pl_loc] = coord_in[pl_loc]['st_pos'][0][iexp]+np.arange(n_osamp_exp)*dx_exp_in[pl_loc][isub_exp]/(n_osamp_exp-1.)  
                    y_oversamp_pl[pl_loc] = coord_in[pl_loc]['st_pos'][1][iexp]+np.arange(n_osamp_exp)*dy_exp_in[pl_loc][isub_exp]/(n_osamp_exp-1.) 
                    
            #Spot oversampled positions initialization
            if cond_spot:
                coord_oversamp = {'x':{},'y':{},'z':{}}
    
                #Spot oversampled positions
                for spot in spots_in_exp:
                    
                    #No oversampling
                    if n_osamp_exp==1:
                        for key in ['x','y','z']:coord_oversamp[key][spot] = [coord_in[spot][key+'_sky_exp'][1,iexp]]
                   
                    #If we want to oversample
                    else:
                        for key in ['x','y','z']:coord_in[spot][key+'_sky_exp'][0,iexp] + np.arange(n_osamp_exp)*dcoord_exp_in_sp[key][spot]/(n_osamp_exp-1.)                

            #Variables to keep track of how many oversampled positions in this exposure were occulting the star
            n_osamp_exp_eff_pl = 0
            n_osamp_exp_eff_sp = 0

            #Loop on oversampled exposure positions 
            #    - after x_oversamp_pl has been defined for all planets
            #    - if oversampling is not active a single central position is processed
            #    - we neglect the potential chromatic variations of the planet radius and corresponding grid 
            #    - if at least one of the processed planet is transiting
            for iosamp in range(n_osamp_exp):

                #Need to define a reduced version of the spot dictionary in this oversampled position. This is required
                #if we want to account for presence of spots in planet-occulted regions.
                spots_are_visible = False
                reduced_spot_prop_oversamp={}
                for spot in spots_in_exp:
                    reduced_spot_prop_oversamp[spot]={}
                    reduced_spot_prop_oversamp[spot]['ctrst']=coord_in[spot]['ctrst'][iexp]
                    reduced_spot_prop_oversamp[spot]['ang_rad']=coord_in[spot]['ang_rad'][iexp]
                    temp_long = np.arcsin(coord_oversamp['x'][spot][iosamp] / np.cos(coord_in[spot]['lat_rad_exp'][1,iexp]))
                    reduced_spot_prop_oversamp[spot]['cos_lat_exp_center'] = np.cos(coord_in[spot]['lat_rad_exp'][1,iexp])
                    reduced_spot_prop_oversamp[spot]['cos_long_exp_center'] = np.cos(temp_long)
                    reduced_spot_prop_oversamp[spot]['sin_lat_exp_center'] = np.sin(coord_in[spot]['lat_rad_exp'][1,iexp])
                    reduced_spot_prop_oversamp[spot]['sin_long_exp_center'] = np.sin(temp_long)
                    spots_are_visible |= is_spot_visible(par_star['istar_rad'], temp_long, coord_in[spot]['lat_rad_exp'][1,iexp], reduced_spot_prop_oversamp[spot]['ang_rad'], par_star['f_GD'], (1-par_star['f_GD']))


                #------------------------------------------------------------
                #Planet-occulted regions
                
                #Dictionary telling us which planets have been processed in which chromatic mode and band.
                pl_proc={subkey_chrom:{iband:[] for iband in range(system_prop[subkey_chrom]['nw'])} for subkey_chrom in key_chrom}
                cond_occ_pl = False
                for pl_loc in transit_pl_exp:   
                    
                    #Frame conversion of planet coordinates from the classical frame perpendicular to the LOS, to the 'inclined star' frame
                    x_st_sky_pos,y_st_sky_pos,_=frameconv_skyorb_to_skystar(lambda_rad_pl[pl_loc],x_oversamp_pl[pl_loc][iosamp],y_oversamp_pl[pl_loc][iosamp],None)      
    
                    #Largest possible square grid enclosing the planet shifted to current planet position     
                    x_st_sky_max = x_st_sky_pos+theo_dic['x_st_sky_grid_pl'][pl_loc]
                    y_st_sky_max = y_st_sky_pos+theo_dic['y_st_sky_grid_pl'][pl_loc]

                    #Calculating properties
                    for subkey_chrom in key_chrom:
                        for iband in range(system_prop[subkey_chrom]['nw']):
                            Focc_star_pl[subkey_chrom][iband],cond_occ_pl = calc_occ_region_prop(line_occ_HP[subkey_chrom][iband],cond_occ_pl,iband,args,system_prop[subkey_chrom],system_spot_prop[subkey_chrom],iosamp,pl_loc,pl_proc[subkey_chrom][iband],theo_dic['Ssub_Sstar_pl'][pl_loc],x_st_sky_max,y_st_sky_max,system_prop[subkey_chrom]['cond_in_RpRs'][pl_loc][iband],par_list,theo_dic['Istar_norm_'+subkey_chrom],\
                                                                                  x_oversamp_pl,y_oversamp_pl,lambda_rad_pl,par_star,sum_prop_dic[subkey_chrom][pl_loc],coord_reg_dic[subkey_chrom][pl_loc],range_dic[subkey_chrom][pl_loc],range_par_list,Focc_star_pl[subkey_chrom][iband],cb_band_dic[subkey_chrom][iband],theo_dic, spot_occ=spots_are_visible, reduced_spot_prop=reduced_spot_prop_oversamp)
            
                            #Cumulate line profile from planet-occulted cells
                            #    - in high-precision mode there is a single subkey_chrom and achromatic band, but several planets may have been processed
                            if ('line_prof' in par_list_in):
                                if (theo_dic['precision']=='low'):surf_prop_dic_pl[subkey_chrom][pl_loc]['rv_broad'][iband,isub_exp] = np.max([coord_reg_dic[subkey_chrom][pl_loc]['rv_broad'][iband],surf_prop_dic_pl[subkey_chrom][pl_loc]['rv_broad'][iband,isub_exp]])
                                elif (theo_dic['precision']=='high'):surf_prop_dic_pl[subkey_chrom]['line_prof'][:,isub_exp]+=sum_prop_dic[subkey_chrom][pl_loc]['line_prof']
                    
                #Star was effectively occulted at oversampled position
                if cond_occ_pl:
                    n_osamp_exp_eff_pl+=1
                    
                    #Calculate line profile from planet-occulted region 
                    #    - profile is scaled to the total flux from current occulted region, stored in coord_reg_dic_pl['Ftot']
                    if ('line_prof' in par_list_in) and (theo_dic['precision']=='medium'):
                        idx_w = {'achrom':range(system_prop['achrom']['nw'])}
                        if ('chrom' in key_chrom):idx_w['chrom'] = range(system_prop['chrom']['nw'])
                        surf_prop_dic_pl[key_chrom[-1]]['line_prof'][:,isub_exp]+=plocc_prof(args,transit_pl_exp,coord_reg_dic,idx_w,system_prop,key_chrom,par_star,theo_dic)

                #------------------------------------------------------------
                #Spotted regions
                if cond_spot:
                    cond_occ_sp = False
                    
                    #Need to make a new dictionary that contains the spot properties for this oversampled exposure
                    spot_prop_oversamp = {}

                    #Building the spot dictionary - we need to extract this information for all the spots to find their overlap
                    for spot in spots_in_exp:
                        
                        #Make a rough estimate of the spot-occulted grid - has a different resolution than the stellar grid - is in inclined star rest frame
                        x_st_sky_max_sp = coord_oversamp['x'][spot][iosamp] + theo_dic['x_st_sky_grid_sp'][spot]
                        y_st_sky_max_sp = coord_oversamp['y'][spot][iosamp] + theo_dic['y_st_sky_grid_sp'][spot]
                        spot_prop_oversamp[spot] = {}
                        spot_prop_oversamp[spot]['ctrst'] = coord_in[spot]['ctrst'][iexp]
                        spot_prop_oversamp[spot]['x_sky_grid'] = x_st_sky_max_sp
                        spot_prop_oversamp[spot]['y_sky_grid'] = y_st_sky_max_sp
                        spot_prop_oversamp[spot]['x_sky_exp_center'] = coord_oversamp['x'][spot][iosamp]
                        spot_prop_oversamp[spot]['y_sky_exp_center'] = coord_oversamp['y'][spot][iosamp]
                        spot_prop_oversamp[spot]['z_sky_exp_center'] = coord_oversamp['z'][spot][iosamp]
                        spot_prop_oversamp[spot]['lat_rad_exp_center'] = coord_in[spot]['lat_rad_exp'][1,iexp]
                        spot_prop_oversamp[spot]['ang_rad'] = coord_in[spot]['ang_rad'][iexp]
                        spot_prop_oversamp[spot]['long_rad_exp_center'] = np.arcsin(coord_oversamp['x'][spot][iosamp] / np.cos(coord_in[spot]['lat_rad_exp'][1,iexp]))
                        spot_prop_oversamp[spot]['cos_long_exp_center'] = np.cos(spot_prop_oversamp[spot]['long_rad_exp_center'])
                        spot_prop_oversamp[spot]['sin_long_exp_center'] = np.sin(spot_prop_oversamp[spot]['long_rad_exp_center'])
                        spot_prop_oversamp[spot]['cos_lat_exp_center'] = np.cos(spot_prop_oversamp[spot]['lat_rad_exp_center'])
                        spot_prop_oversamp[spot]['sin_lat_exp_center'] = np.sin(spot_prop_oversamp[spot]['lat_rad_exp_center'])
                        spot_prop_oversamp[spot]['is_visible'] = is_spot_visible(par_star['istar_rad'], spot_prop_oversamp[spot]['long_rad_exp_center'], spot_prop_oversamp[spot]['lat_rad_exp_center'], spot_prop_oversamp[spot]['ang_rad'], par_star['f_GD'], (1-par_star['f_GD']))
    
                    #Dictionary telling us which spots remain to be processed in which chromatic mode and band.
                    sp_proc={subkey_chrom:{iband:[] for iband in range(system_spot_prop[subkey_chrom]['nw'])} for subkey_chrom in key_chrom}

                    #Retrieving the properties of the region occulted by each spot
                    for spot in spots_in_exp:
   
                        #Going over the chromatic modes
                        for subkey_chrom in key_chrom:
                            
                            #Going over the bands in each chromatic mode
                            for iband in range(system_spot_prop[subkey_chrom]['nw']):
                                Focc_star_sp[subkey_chrom][iband], cond_occ_sp = calc_spotted_region_prop(line_occ_HP[subkey_chrom][iband], cond_occ_sp, spot_prop_oversamp, iband, system_prop[subkey_chrom], 
                                                                system_spot_prop[subkey_chrom], par_star, sp_proc[subkey_chrom][iband],spot,theo_dic['Ssub_Sstar_sp'][spot], 
                                                                theo_dic['Ssub_Sstar'], theo_dic['Istar_norm_'+subkey_chrom], sum_prop_dic[subkey_chrom][spot], coord_reg_dic[subkey_chrom][spot],
                                                                range_dic[subkey_chrom][spot], Focc_star_sp[subkey_chrom][iband], par_list, range_par_list, args, cb_band_spot_dic[subkey_chrom][iband], 
                                                                pl_loc_x = x_oversamp_pl, pl_loc_y = y_oversamp_pl, oversamp_idx = iosamp, RpRs = system_prop[subkey_chrom], plocc = (n_osamp_exp_eff_pl>=1))
    
                                #Cumulate line profile from spot-occulted cells
                                #    - in high-precision mode there is a single subkey_chrom and achromatic band, but several spots may have been processed
                                if ('line_prof' in par_list_in):
                                    surf_prop_dic_spot[subkey_chrom]['line_prof'][:,isub_exp]+=sum_prop_dic[subkey_chrom][spot]['line_prof']

                                emit_coord_reg_dic = deepcopy(coord_reg_dic)
                                emit_coord_reg_dic[subkey_chrom][spot]['Ftot'][iband] *= (1-spot_prop_oversamp[spot]['ctrst'])

                                emit_surf_prop_dic_spot = deepcopy(surf_prop_dic_spot)
                                emit_surf_prop_dic_spot[subkey_chrom][spot]['Ftot'][iband] *= (1-spot_prop_oversamp[spot]['ctrst'])

                    #Star was effectively occulted at oversampled position
                    if cond_occ_sp:
                        n_osamp_exp_eff_sp+=1
                    
            #------------------------------------------------------------

            #Averaged values behind all occulted regions during exposure
            #    - with the oversampling, positions at the center of exposure will weigh more in the sum than those at start and end of exposure, like in reality
            #    - parameters are retrieved in both oversampled/not-oversampled case after they are updated within the sum_prop_dic dictionary 
            #    - undefined values remain set to nan, and are otherwise normalized by the flux from the planet-occulted region
            #    - we use a single Itot as condition that the planet occulted the star
            calc_mean_occ_region_prop(transit_pl_exp,surf_prop_dic_pl,n_osamp_exp_eff_pl,sum_prop_dic,key_chrom,par_list,isub_exp,out_ranges,range_par_list,range_dic)     
            if cond_spot:                
                calc_mean_occ_region_prop(spots_in_exp,surf_prop_dic_spot,n_osamp_exp_eff_sp,sum_prop_dic,key_chrom,par_list,isub_exp,out_ranges,range_par_list,range_dic)                            
                            
            #Normalized stellar flux after occultation by all planets and by all spots
            #    - the intensity from each cell is calculated in the same way as that of the full pre-calculated stellar grid
            if Ftot_star:
                for subkey_chrom in key_chrom:
                    surf_prop_dic_pl[subkey_chrom]['Ftot_star'][:,isub_exp] = 1.
                    surf_prop_dic_common[subkey_chrom]['Ftot_star'][:,isub_exp] = 1.
                    if cond_spot:surf_prop_dic_spot[subkey_chrom]['Ftot_star'][:,isub_exp] = 1.

                    #Planets
                    if n_osamp_exp_eff_pl>0:
                        surf_prop_dic_pl[subkey_chrom]['Ftot_star'][:,isub_exp] -= Focc_star_pl[subkey_chrom]/(n_osamp_exp_eff_pl*theo_dic['Ftot_star_'+subkey_chrom])
                        surf_prop_dic_common[subkey_chrom]['Ftot_star'][:,isub_exp] -= Focc_star_pl[subkey_chrom]/(n_osamp_exp_eff_pl*theo_dic['Ftot_star_'+subkey_chrom])
                    
                    #Spots
                    if cond_spot and (n_osamp_exp_eff_sp>0):
                        surf_prop_dic_spot[subkey_chrom]['Ftot_star'][:,isub_exp] -= (Focc_star_sp[subkey_chrom])/(n_osamp_exp_eff_sp*theo_dic['Ftot_star_'+subkey_chrom])
                        surf_prop_dic_common[subkey_chrom]['Ftot_star'][:,isub_exp] -= (Focc_star_sp[subkey_chrom])/(n_osamp_exp_eff_sp*theo_dic['Ftot_star_'+subkey_chrom])


            #Local line profiles from current exposure
            if ('line_prof' in par_list_in):

                #Planet-occulted line profile         
                if (n_osamp_exp_eff_pl>0):
                    calc_mean_occ_region_line(theo_dic['precision'],system_prop,isub_exp,key_chrom,n_osamp_exp_eff_pl,Focc_star_pl,surf_prop_dic_pl,transit_pl_exp,args,par_star,theo_dic)
            
                #Spotted line profile 
                if cond_spot and (n_osamp_exp_eff_sp > 0):
                    calc_mean_occ_region_line(theo_dic['precision'],system_prop,isub_exp,key_chrom,n_osamp_exp_eff_sp,Focc_star_sp,surf_prop_dic_spot,spots_in_exp,args,par_star,theo_dic)

    ### end of exposure            
 
    #Output properties in chromatic mode if calculated in closest-achromatic mode
    if ('line_prof' in par_list_in) and switch_chrom:
        surf_prop_dic_pl = {'chrom':surf_prop_dic_pl['achrom']}
        surf_prop_dic_spot = {'chrom':surf_prop_dic_spot['achrom']}
        surf_prop_dic_common = {'chrom':surf_prop_dic_common['achrom']}

    return surf_prop_dic_pl, surf_prop_dic_spot , surf_prop_dic_common



def calc_mean_occ_region_prop(occulters,surf_prop_dic,n_osamp_exp_eff,sum_prop_dic,key_chrom,par_list,i_in,out_ranges,range_par_list,range_dic):
    r"""**Planet-occulted properties: average properties**

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


def calc_mean_occ_region_line(precision,system_prop,i_in,key_chrom,n_osamp_exp_eff,Focc_star,surf_prop_dic,occ_in_exp,args,par_star,theo_dic):
    r"""**Planet-occulted properties: average line**

    Calculates the line profile from the cumulated stellar surface regions occulted during an exposure.

    Args:
        TBD
    
    Returns:
        TBD
    
    """     
        
    #Profile from averaged properties over exposures
    if (precision=='low'): 
        idx_w = {'achrom':(range(system_prop['achrom']['nw']),i_in)}
        if ('chrom' in key_chrom):idx_w['chrom'] = (range(system_prop['chrom']['nw']),i_in)          
        surf_prop_dic[key_chrom[-1]]['line_prof'][:,i_in]=plocc_prof(args,occ_in_exp,surf_prop_dic,idx_w,system_prop,key_chrom,par_star,theo_dic)
    
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


def calc_occ_region_prop(line_occ_HP_band,cond_occ,iband,args,system_prop,system_spot_prop,idx,pl_loc,pl_proc_band,Ssub_Sstar,x_st_sky_max,y_st_sky_max,cond_in_RpRs,par_list,Istar_norm_band,x_pos_pl,y_pos_pl,lambda_rad_pl,par_star,sum_prop_dic_pl,\
                         coord_reg_dic_pl,range_reg,range_par_list,Focc_star_band,cb_band,theo_dic,spot_occ = False,reduced_spot_prop={}):
    r"""**Planet-occulted properties: region**

    Calculates the average and summed properties from a planet-occulted stellar surface region during an exposure.

    Args:
        line_occ_HP_band (str) : The precision with which to process the exposure.
        cond_occ (bool) : Boolean telling us whether there is an occultation by at least one planet in the oversampled exposure considered.
        iband (int) : Index of the band of interest.
        system_prop (dict) : Planet limb-darkening properties.
        system_spot_prop (dict) : Spot limb-darkening properties.
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
        range_par_list (list) : list of parameters of interest, whose range of values, stored in range_reg_dic_spot, will be updated.
        Focc_star_band (float) : total flux occulted by the spot in the exposure and band considered. 
        cb_band (list) : Polynomial coefficients used to compute thr RV component of the planet-occulted region due to convective blueshift.
        theo_dic (dict) : parameters used to generate and describe the stellar grid and planet-occulted regions grid.
        spot_occ (bool) : Optional, whether spotted regions are present in the oversampled exposure considered. Default is False.
        reduced_spot_prop (dict) : Optional, spot properties used to account for the possible presence of spotted cells in the planet-occulted region. Default is an empty dictionary.
        fit_Ftot_star (bool) : Optional, whether to include the continuum in the calculation of Focc_star_band.
    
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

    #Identifying occulted stellar cells in the sky-projected and star star rest frame
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
                RpRs_prev = system_prop[pl_prev][iband]
                cond_pl_occ_corr &= ( (coord_grid['x_st_sky'] - x_st_sky_prev)**2.+(coord_grid['y_st_sky'] - y_st_sky_prev)**2. > RpRs_prev**2. )
            for key in coord_grid:coord_grid[key] = coord_grid[key][cond_pl_occ_corr]
            n_pl_occ = np.sum(cond_pl_occ_corr)
      
        #Store planet as processed in current band
        pl_proc_band+=[pl_loc]     

        #--------------------------------
        #Account for spot occultation in planet-occulted region
        if spot_occ:

            #Making a reference spot to extract the contrast
            ref_spot = list(reduced_spot_prop.keys())[0]
            
            #Identify the cells in the planet-occulted region that are spotted
            for spot in list(reduced_spot_prop.keys()):
                
                #Making individual spot maps
                coord_grid[spot+'_flag_map']=np.zeros(len(coord_grid['x_st_sky']), dtype=bool)

                #Retrieve coordinates of the planet-occulted region in the inclined star frame
                new_x_sky_grid = coord_grid['x_st_sky']
                new_y_sky_grid = coord_grid['y_st_sky']
                new_z_sky_grid = coord_grid['z_st_sky']


                #Move coordinates to the (non-inclined) star frame and then the spot reference frame
                x_st_grid, y_st_grid, z_st_grid = frameconv_skystar_to_star(new_x_sky_grid, new_y_sky_grid, new_z_sky_grid, par_star['istar_rad'])
                x_spot_grid = x_st_grid*reduced_spot_prop[spot]['cos_long_exp_center'] - z_st_grid*reduced_spot_prop[spot]['sin_long_exp_center']
                y_spot_grid = y_st_grid*reduced_spot_prop[spot]['cos_lat_exp_center'] - (z_st_grid*reduced_spot_prop[spot]['cos_long_exp_center'] + x_st_grid*reduced_spot_prop[spot]['sin_long_exp_center']) * reduced_spot_prop[spot]['sin_lat_exp_center']
                cond_in_sp = (x_spot_grid**2. + y_spot_grid**2. <= reduced_spot_prop[spot]['ang_rad']**2)

                #Updating the flag map
                coord_grid[spot+'_flag_map'] |= cond_in_sp

            coord_grid['gen_spot_flag_map'] = coord_grid[ref_spot+'_flag_map']

            #Accounting for possible overlap
            if len(list(reduced_spot_prop.keys()))>1:
                for spot in list(reduced_spot_prop.keys()):
                    coord_grid['gen_spot_flag_map'] |= (coord_grid[spot+'_flag_map'] & ~coord_grid['gen_spot_flag_map'])


        #--------------------------------

        #Local flux grid over current planet-occulted region, in current band
        coord_grid['nsub_star'] = n_pl_occ
        _,_,mu_grid_star,Fsurf_grid_star,Ftot_star,_ = calc_Isurf_grid([iband],coord_grid['nsub_star'],system_prop,coord_grid,par_star,Ssub_Sstar,Istar_norm = Istar_norm_band,region = 'pl',Ssub_Sstar_ref = theo_dic['Ssub_Sstar'])
        coord_grid['mu'] = mu_grid_star[:,0]

        #Accounting for the spots' emission
        if spot_occ:
            
            #Retrieve the flux-grid with the LD coefficients of the spot
            _,_,_,Fsurf_spot_emit_grid,_,_ = calc_Isurf_grid([iband],coord_grid['nsub_star'],system_spot_prop,coord_grid,par_star,Ssub_Sstar,Istar_norm = Istar_norm_band,region = 'pl',Ssub_Sstar_ref = theo_dic['Ssub_Sstar'])
            Fsurf_grid_star[:, iband] = np.where(coord_grid['gen_spot_flag_map'] == True, Fsurf_grid_star[:, iband] - (Fsurf_spot_emit_grid[:, iband] * reduced_spot_prop[ref_spot]['ctrst']), Fsurf_grid_star[:, iband])
            Ftot_star = np.sum(Fsurf_grid_star, axis=0)

        #Scale continuum level
        Fsurf_grid_star*=par_star['cont']
        Ftot_star*=par_star['cont']
       
        #Flux and number of cells occulted from all planets, cumulated over oversampled positions
        Focc_star_band+= Ftot_star[0]   
        sum_prop_dic_pl['nocc']+=coord_grid['nsub_star']
        
        #--------------------------------

        #Co-adding properties from current region to the cumulated values over oversampled planet positions 
        sum_region_prop(line_occ_HP_band,iband,args,parameter_list,Fsurf_grid_star[:,0],None,coord_grid,Ssub_Sstar,cb_band,range_parameter_list,range_reg,sum_prop_dic_pl,coord_reg_dic_pl,par_star,lambda_rad_pl[pl_loc],None)

    return Focc_star_band,cond_occ




def sum_region_prop(line_occ_HP_band,iband,args,par_list,Fsurf_grid_band,Fsurf_grid_emit_band,coord_grid,Ssub_Sstar,cb_band,range_par_list,range_reg,sum_prop_dic,coord_reg_dic,par_star,lambda_rad_pl_loc,spot_contrast):
    r"""**Planet-occulted or spotted region properties: calculations**

    Calculates the average and summed properties from a local (planet-occulted or spotted) stellar surface region during an exposure.
    
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
        spot_contrast (float) : Contrast level of the spot considered.

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
            elif par_loc=='Rot_RV':coord_grid[par_loc] = calc_RVrot(coord_grid['x_st_sky'],coord_grid['y_st'],par_star['istar_rad'],par_star)[0]
                         
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
        if not args['fit']:line_prof_grid=coadd_loc_line_prof(coord_grid['rv'],range(coord_grid['nsub_star']),Fsurf_grid_band,args['flux_intr_grid'],coord_grid['mu'],par_star,args)          
        else:
            if use_OS_grid:
                fit_Fsurf_grid_band = np.tile(Fsurf_grid_band, (args['ncen_bins'], 1)).T
                line_prof_grid=coadd_loc_gauss_prof(coord_grid['rv'],fit_Fsurf_grid_band,args)
            elif use_C_OS_grid:
                line_prof_grid = use_C_coadd_loc_gauss_prof(coord_grid['rv'],Fsurf_grid_band,args)
            else:line_prof_grid=coadd_loc_line_prof(coord_grid['rv'],range(coord_grid['nsub_star']),Fsurf_grid_band,args['flux_intr_grid'],coord_grid['mu'],par_star,args)

        #Coadd line profiles over planet-occulted region
        sum_prop_dic['line_prof'] = np.sum(line_prof_grid,axis=0) 

        #Calculate profile emitted by a spot
        if spot_contrast is not None:
            if not args['fit']:emit_line_prof_grid = coadd_loc_line_prof(coord_grid['rv'],range(coord_grid['nsub_star']),(1-spot_contrast)*Fsurf_grid_emit_band,args['flux_intr_grid'],coord_grid['mu'],par_star,args)          
            else:
                if use_OS_grid:
                    fit_Fsurf_grid_emit_band = np.tile((1-spot_contrast)*Fsurf_grid_emit_band, (args['ncen_bins'], 1)).T
                    emit_line_prof_grid = coadd_loc_gauss_prof(coord_grid['rv'],fit_Fsurf_grid_emit_band,args)
                elif use_C_OS_grid:
                    emit_line_prof_grid = use_C_coadd_loc_gauss_prof(coord_grid['rv'],(1-spot_contrast)*Fsurf_grid_emit_band,args)
                else:emit_line_prof_grid = coadd_loc_line_prof(coord_grid['rv'],range(coord_grid['nsub_star']),(1-spot_contrast)*Fsurf_grid_emit_band,args['flux_intr_grid'],coord_grid['mu'],par_star,args)        

            #Remove line profiles over spotted region
            sum_prop_dic['line_prof'] -= np.sum(emit_line_prof_grid,axis=0)
  
    #Define rotational broadening of planet-occulted region
    elif line_occ_HP_band in ['low','medium']:
        drv_min = coord_reg_dic['rv'][iband]-np.min(coord_grid['rv'])
        drv_max = np.max(coord_grid['rv'])-coord_reg_dic['rv'][iband] 
        coord_reg_dic['rv_broad'][iband] = 0.5*(drv_min+drv_max)       

    return None






def plocc_prof(args,transit_pl,coord_dic,idx_w,system_prop,key_chrom,param,theo_dic):
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
    for pl_loc in transit_pl:  
        
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
                else:flux_sc_spec = np_interp(args['cen_bins'],system_prop['chrom']['w'],flux_sc,left=flux_sc[0],right=flux_sc[-1])
         
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
                if args['conv2intr']:dic_rv = {'achrom':{pl_loc:{'rv': np.reshape(coord_dic['achrom'][pl_loc]['rv'][idx_w['achrom']],[system_prop['achrom']['nw'],1]) }}}
                else:dic_rv = {chrom_calc:{pl_loc:{'rv': np.reshape(coord_dic[chrom_calc][pl_loc]['rv'][idx_w[chrom_calc]],[system_prop[chrom_calc]['nw'],1]) }}}
                rv_surf_star,rv_surf_star_edge = def_surf_shift('theo',dic_rv,0,data_loc,pl_loc,args['type'],system_prop,[1,ncen_bins_Intr],1,ncen_bins_Intr)  
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



def occ_region_grid(RregRs, nsub_Dreg , spot = False):
    r"""**Local region grid** 

    Defines a square x/y/z grid enclosing a local region of the stellar surface in the 'inclined' star frame, with:
      
     - X axis is parallel to the star equator
     - Y axis is the projected spin axis
     - Z axis is along the LOS

    Args:
        RregRs (float) : the radius of the region in the XY plane. 
                         Because it is normalized by the stellar radius, `RregRs` also corresponds to the (half) angle defining the chord of the region along the stellar surface. 
                         For a planet the projected region keeps a constant radius in the XY plane and its angular aperture increases toward the limbs.
                         For a spot the angular aperture is fixed and it is the radius of the projection that decreases toward the limb. The input `RregRs` corresponds to the angular aperture and thus defines the largest square enclosing the spot as it would be seen at the center of the star.   
        nsub_Dreg (int) : the number of grid cells desired.
    
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
    if not spot:

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
        if len(data_dic[inst][vis]['transit_pl'])==1:ref_pl = data_dic[inst][vis]['transit_pl'][0]  
        else:ref_pl=data_dic['Intr']['align_ref_pl'][inst][vis]
    
    return ref_pl,dic_rv,idx_aligned


def def_surf_shift(align_mode,dic_rv,i_in,data_exp,pl_ref,data_type,system_prop,dim_exp,nord,nspec):    
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
                rv_surf_star[iord] = np_interp(data_exp['cen_bins'][iord],system_prop['chrom']['w'],RV_shift_pl,left=RV_shift_pl[0],right=RV_shift_pl[-1])
                rv_surf_star_edge[iord] = np_interp(data_exp['edge_bins'][iord],system_prop['chrom']['w'],RV_shift_pl,left=RV_shift_pl[0],right=RV_shift_pl[-1])
            
        #Achromatic RV defined for the nominal transit properties
        else:
            rv_surf_star = dic_rv['achrom'][pl_ref]['rv'][0,i_in]  
            rv_surf_star_edge=None

    return rv_surf_star,rv_surf_star_edge










#%% Spot-specific routines

def is_spot_visible(istar, long_rad, lat_rad, ang_rad, f_GD, RpoleReq) :
    r"""**Spot visibility**

    Performs a rough estimation of the visibility of a spot, based on star inclination (istar), spot coordinates in star rest frame (long, lat) and spot angular size (ang).
    To do so, we discretize the spot edge and check if at least one point is visible. The visibility criterion is derived as follows. 

    Let :math:`\mathrm{P} = (x_{\mathrm{st}}, y_{\mathrm{st}}, z_{\mathrm{st}})` be a point on the stellar surface, in the star rest frame (i.e. X axis parallel to the star 
    equator and  Y axis along the stellar spin), that is on the edge of a spot.
    
    Expressing P in spherical coordinates gives

    .. math::
        & x_{\mathrm{st}} = \mathrm{sin}(long) \mathrm{cos}(lat)
        & y_{\mathrm{st}} = \mathrm{sin}(lat)
        & z_{\mathrm{st}} = \mathrm{cos}(long) \mathrm{cos}(lat)
    
    Moving P to the 'inclined' star rest frame gives 

    .. math::
        & x_{\mathrm{sky}} = x_{\mathrm{st}}
        & y_{\mathrm{sky}} = \mathrm{sin}(i_*) y_{\mathrm{st}} - \mathrm{cos}(i_*) z_{\mathrm{st}}
        & z_{\mathrm{sky}} = \mathrm{cos}(i_*) y_{\mathrm{st}} + \mathrm{sin}(i_*) z_{\mathrm{st}}
    
    The condition for P to be visible is then :math:`z_{\mathrm{sky}} > 0` or 
    
    .. math::
        & \mathrm{cos}(i_*) \mathrm{sin}(lat) + \mathrm{sin}(i_*) \mathrm{cos}(long) \mathrm{cos}(lat) > 0

    WIP : Doing this with gravity darkening

    Args:
        istar (float) : stellar inclination (in radians)
        long_rad (float) : spot longitude (in radians)
        lat_rad (float) : spot latitude (in radians)
        ang_rad (float) : spot angular size (in radians)
        f_GD (float) : oblateness coefficient.
        RpoleReq (float) : pole to equatoral radius ratio.
     
    Returns:
        spot_visible (bool) : spot visibility criterion.
        
    """ 
    spot_visible = False
    
    for arg in np.linspace(0,2*np.pi, 20) :
        
        #Define the edges of the spots
        long_edge = long_rad + ang_rad * np.sin(arg)
        lat_edge  = lat_rad  + ang_rad * np.cos(arg)

        #Define the corresponding x, y, and z coordinates of the edges of the spots in the un-inclined star rest frame
        x_st_edge = np.sin(long_edge)*np.cos(lat_edge)
        y_st_edge = np.sin(lat_edge)
        z_st_edge = np.cos(long_edge)*np.cos(lat_edge)

        #Move the x, y, and z coordinates of the spot edges into the inclined star rest frame 
        x_sky_edge, y_sky_edge, z_sky_edge = frameconv_star_to_skystar(x_st_edge, y_st_edge, z_st_edge, istar)

        ##### WIP #####
        #Checking the value of the z coordinate for each edge point to see if the point is visible
        if f_GD>0:
            #We take the modulo in case the spot is plotted over a very long time.
            long_edge_deg = (long_edge * 180/np.pi)%360
            #Checking if the spot is in the front - rough estimate that doesn't account for the star inclination or spot latitude
            additive = 90 - (istar * 180/np.pi)
            first_criterion = (-90-additive <= long_edge_deg and long_edge_deg <= 90+additive) or (270-additive <= long_edge_deg and long_edge_deg <=360) or (-360 <= long_edge_deg and long_edge_deg <= -270+additive)
            #Now that we known the spot is in front, combine with the condition "is in stellar photosphere"
            criteria = calc_zLOS_oblate(np.array([x_sky_edge]),np.array([y_sky_edge/(1-f_GD)]),istar, RpoleReq)[2]
            criterion = first_criterion and criteria
        #####    #####
        else:
            criterion = (z_sky_edge > 0)
        
        spot_visible |= criterion

    return spot_visible

def calc_plocced_tiles(pl_prop, x_sky_grid, y_sky_grid):
    r"""**'Planet-occulted' tiles** 
    
    Args:
        pl_prop (dict) : planet properties.
        x_sky_grid (1D array) : x coordinates of the stellar / planetary grid in the inclined star frame. (at st, cen, and end)
        y_sky_grid (1D array) : y coordinates of the stellar / planetary grid in the inclined star frame. (at st, cen, and end)
     
    Returns:
        cond_in_pl (1D array) : array of booleans telling us which cells in the original grid are occulted by the planet.

    """          
    cond_in_pl = np.zeros(len(x_sky_grid), dtype=bool)

    for i in range(len(pl_prop['x_orb_exp'])):

                
   
   
   
        pl_prop['x_sky_exp'],pl_prop['y_sky_exp'],_ = frameconv_skyorb_to_skystar(pl_prop['lambda'],pl_prop['x_orb_exp'][i],pl_prop['y_orb_exp'][i],0)
     
        pos_cond_close_to_pl = (x_sky_grid - pl_prop['x_sky_exp'])**2 + (y_sky_grid - pl_prop['y_sky_exp'])**2 < pl_prop['RpRs']**2  

        cond_in_pl |=   pos_cond_close_to_pl

    return cond_in_pl



def retrieve_spots_prop_from_param(star_params, param, inst, vis, t_bjd, exp_dur=None): 
    r"""**Spot parameters: retrieval and formatting**

    Transforms a dictionary with 'raw' spot properties in the format param_ISinstrument_VSvisit_SPspotname to a more convenient spot dictionary of the form : 
    spot_prop = { spotname : {'lat' : , 'Tcenter' : , ....}}
    The formatted contains the initial spot properties as well as additional derived spot properties, such as the longitude and latitude of the spot as well as 
    its visibility criterion (see is_spot_visible).
    We assume spot parameter are never defined as common across multiple visits / instruments.

    Args:
        star_params (dict) : star properties.
        param (dict) : 'raw' spot properties.
        inst (str) : instrument considered. Should match the instrument in the 'raw' spot parameter name (see format above).
        vis (str) : visit considered. Should match the visit in the 'raw' spot parameter name (see format above).
        t_BJD (float) : timestamp of the exposure considered. Needed to calculate the spot longitude.
        exp_dur (float) : optional, duration of the exposure considered. If left as None, only the spot properties at the center of the exposure will be computed.
                            Otherwise, the spot properties at the start, center and end of exposure will be computed. 
     
    Returns:
        spots_prop (dict) : formatted spot dictionary.

    """ 

    spots_prop = {}
    ctrst_param = []
    for par in param : 
        # Parameter is spot-related and linked to the right visit and instrument
        if ('_SP' in par) and ('_IS'+inst in par) and ('_VS'+vis in par): 
            if 'ctrst' not in par:
                spot_par = par.split('__IS')[0]
                spot_name = par.split('_SP')[1]
                if spot_name not in spots_prop : spots_prop[spot_name] = {}
                spots_prop[spot_name][spot_par] = param[par]
            else:
                ctrst_param.append(param[par])

    #Ensuring that only one contrast is provided for the spots in each visit
    if len(ctrst_param)>1:
        stop('WARNING: All spots in a given visit must share a contrast value. Please provide only one spot contrast parameter.')
    
    #Assigning contrast value to each spot
    for spot_name in spots_prop.keys():
        spots_prop[spot_name]['ctrst'] = ctrst_param[0]
 
    # Retrieve properties, if spots are visible in the exposures considered
    for spot in spots_prop : 
        
        #Finding the times at the center, start and end of each exposure considered
        t_bjd_center = deepcopy(t_bjd)

        # Spot lattitude - constant in time
        lat_rad = spots_prop[spot]['lat']*np.pi/180.
        
        # Spot longitude - varies over time
        sin_lat = np.sin(lat_rad)
        P_spot = 2*np.pi/((1.-star_params['alpha_rot_spots']*sin_lat**2.-star_params['beta_rot_spots']*sin_lat**4.)*star_params['om_eq_spots']*3600.*24.)
        Tcen_sp = spots_prop[spot]['Tcenter'] - 2400000.
        long_rad_center = (t_bjd_center-Tcen_sp)/P_spot * 2*np.pi
        
        # Spot center coordinates in star rest frame
        #Exposure center
        x_st_center = np.sin(long_rad_center)*np.cos(lat_rad)
        y_st_center = np.sin(lat_rad)
        z_st_center = np.cos(long_rad_center)*np.cos(lat_rad)

        # inclined frame
        istar = np.arccos(param['cos_istar'])

        #Exposure center
        x_sky_center,y_sky_center,z_sky_center = frameconv_star_to_skystar(x_st_center,y_st_center,z_st_center,istar)

       
        #Store properties common across the exposure
        spots_prop[spot]['ang_rad'] = spots_prop[spot]['ang'] * np.pi/180
        # Store properties at exposure center
        spots_prop[spot]['lat_rad_exp_center'] = lat_rad
        spots_prop[spot]['sin_lat_exp_center'] = np.sin(lat_rad)
        spots_prop[spot]['cos_lat_exp_center'] = np.cos(lat_rad)
        spots_prop[spot]['long_rad_exp_center'] = long_rad_center
        spots_prop[spot]['sin_long_exp_center'] = np.sin(long_rad_center)
        spots_prop[spot]['cos_long_exp_center'] = np.cos(long_rad_center)
        spots_prop[spot]['x_sky_exp_center'] = x_sky_center
        spots_prop[spot]['y_sky_exp_center'] = y_sky_center
        spots_prop[spot]['z_sky_exp_center'] = z_sky_center
        spots_prop[spot]['is_center_visible'] = is_spot_visible(istar,long_rad_center, lat_rad, spots_prop[spot]['ang_rad'], star_params['f_GD'], star_params['RpoleReq'])

        if exp_dur is not None:
            t_dur_days = exp_dur/(24.*3600.)
            t_bjd_start = t_bjd - t_dur_days/2
            t_bjd_end = t_bjd + t_dur_days/2
            long_rad_start = (t_bjd_start-Tcen_sp)/P_spot * 2*np.pi
            long_rad_end = (t_bjd_end-Tcen_sp)/P_spot * 2*np.pi
            
            #Exposure start positions
            x_st_start = np.sin(long_rad_start)*np.cos(lat_rad)
            y_st_start = np.sin(lat_rad)
            z_st_start = np.cos(long_rad_start)*np.cos(lat_rad)
            x_sky_start,y_sky_start,z_sky_start = frameconv_star_to_skystar(x_st_start,y_st_start,z_st_start,istar)

            #Exposure end position
            x_st_end = np.sin(long_rad_end)*np.cos(lat_rad)
            y_st_end = np.sin(lat_rad)
            z_st_end = np.cos(long_rad_end)*np.cos(lat_rad)
            x_sky_end,y_sky_end,z_sky_end = frameconv_star_to_skystar(x_st_end,y_st_end,z_st_end,istar)

            # Store properties at exposure start
            spots_prop[spot]['lat_rad_exp_start'] = lat_rad
            spots_prop[spot]['sin_lat_exp_start'] = np.sin(lat_rad)
            spots_prop[spot]['cos_lat_exp_start'] = np.cos(lat_rad)
            spots_prop[spot]['long_rad_exp_start'] = long_rad_start
            spots_prop[spot]['sin_long_exp_start'] = np.sin(long_rad_start)
            spots_prop[spot]['cos_long_exp_start'] = np.cos(long_rad_start)
            spots_prop[spot]['x_sky_exp_start'] = x_sky_start
            spots_prop[spot]['y_sky_exp_start'] = y_sky_start
            spots_prop[spot]['z_sky_exp_start'] = z_sky_start
            spots_prop[spot]['is_start_visible'] = is_spot_visible(istar,long_rad_start, lat_rad, spots_prop[spot]['ang_rad'], star_params['f_GD'], star_params['RpoleReq'])

            # Store properties at exposure end
            spots_prop[spot]['lat_rad_exp_end'] = lat_rad
            spots_prop[spot]['sin_lat_exp_end'] = np.sin(lat_rad)
            spots_prop[spot]['cos_lat_exp_end'] = np.cos(lat_rad)
            spots_prop[spot]['long_rad_exp_end'] = long_rad_end
            spots_prop[spot]['sin_long_exp_end'] = np.sin(long_rad_end)
            spots_prop[spot]['cos_long_exp_end'] = np.cos(long_rad_end)
            spots_prop[spot]['x_sky_exp_end'] = x_sky_end
            spots_prop[spot]['y_sky_exp_end'] = y_sky_end
            spots_prop[spot]['z_sky_exp_end'] = z_sky_end
            spots_prop[spot]['is_end_visible'] = is_spot_visible(istar,long_rad_end, lat_rad, spots_prop[spot]['ang_rad'], star_params['f_GD'], star_params['RpoleReq'])

    return spots_prop



def calc_spotted_tiles(spot_prop, x_sky_grid, y_sky_grid, z_sky_grid, grid_dic, star_param, use_grid_dic = False, disc_exp = True) :
    r"""**'Spotted' tiles** 

    Identification of which cells on the provided grid are covered by spots a.k.a. which cells are 'spotted'. Two methods are available: 
        
    - use_grid_dic = False : calculation will be performed on the x_sky_grid, y_sky_grid, z_sky_grid, by moving these grids from the inclined star frame to the star rest frame.
                            This option can be used for identifiying spotted stellar tiles, when istar is fitted.
    - use_grid_dic = True : calculation will be performed on the grid contained in grid_dic['x/y/z_st'], which is already in star rest frame (no frame conversion needed). 


    To identify which cells in the provided grid are 'spotted', we move the grid cells to the spot reference frame and check which cells are within the spot. 
    This is a necessary step to obtain the correct shape for the spots, as the spherical nature of the stellar surface must be accounted for. The calculation is as follows:
    
    Using the x/y/z grid in the star rest frame (obtained either with a frame conversion function or from the grid dictionary), we rotate the x/y/z grid around the stellar spin axis 
    to the longitude of the spot

    .. math::
        & x_{\mathrm{sp}'} = x_{\mathrm{st}} \mathrm{cos}(long) - z_{\mathrm{st}} \mathrm{sin}(long)
        & y_{\mathrm{sp}'} = y_{\mathrm{st}}
        & z_{\mathrm{sp}'} = x_{\mathrm{st}} \mathrm{sin}(long) + z_{\mathrm{st}} \mathrm{cos}(long)

    We then rotate the new grid to the latitude of the spot to obtain the x/y/z grid in the spot rest frame
    
    .. math::
        & x_{\mathrm{sp}} = x_{\mathrm{sp}'}
        & y_{\mathrm{sp}} = y_{\mathrm{sp}'} \mathrm{cos}(lat) - z_{\mathrm{sp}'} \mathrm{sin}(lat)
        & z_{\mathrm{sp}} = y_{\mathrm{sp}'} \mathrm{sin}(lat) + z_{\mathrm{sp}'} \mathrm{cos}(lat)
    
    Finally, we check which cells are within the spot

    .. math::
        & arctan2(\sqrt{x_{\mathrm{sp}}^{2} + y_{\mathrm{sp}}^{2}}, z_{\mathrm{sp}}) < R_{\mathrm{sp}}

    With R_{\mathrm{sp}} the angular size of the spot in radians.


    Args:
        spot_prop (dict) : formatted spot properties (see retrieve_spots_prop_from_param).
        x_sky_grid (1D array) : x coordinates of the stellar / planetary grid in the inclined star frame.
        y_sky_grid (1D array) : y coordinates of the stellar / planetary grid in the inclined star frame.
        z_sky_grid (1D array) : z coordinates of the stellar / planetary grid in the inclined star frame.
        grid_dic (dict) : dictionary containing the x/y/z grids in various reference frames, including the star rest frame and inclined star frame.
        star_param (dict) : star properties.
        use_grid_dic (bool) : whether or not to use the grid_dic provided to retrieve the x/y/z grids in the star rest frame. Turned off by default.
        disc_exp (bool) : whether we use the start, center and end of the exposures to figure out if the star is spotted and where the star is spotted, or just the center.
     
    Returns:
        spot_within_grid (bool) : a finer estimate of the spot visibility that can be obtained with is_spot_visible. Essentially, it tells us if at least one tile on the grid is 'spotted'.
        cond_in_sp (1D array) : array of booleans telling us which cells in the original grid are 'spotted'.
    
    """                                      

    if disc_exp:positions = ['start','center','end']
    else:positions = ['center']
    cond_in_sp = np.zeros(len(x_sky_grid), dtype=bool)

    for pos in positions:

        if use_grid_dic :
            cond_close_to_spot = (grid_dic['x_st_sky'] - spot_prop['x_sky_exp_'+pos])**2 + (grid_dic['y_st_sky'] - spot_prop['y_sky_exp_'+pos])**2 < spot_prop['ang_rad']**2
         
            x_st_grid, y_st_grid, z_st_grid = grid_dic['x_st'][cond_close_to_spot], grid_dic['y_st'][cond_close_to_spot], grid_dic['z_st'][cond_close_to_spot]
            
            
        else :  
            cond_close_to_spot = (x_sky_grid - spot_prop['x_sky_exp_'+pos])**2 + (y_sky_grid - spot_prop['y_sky_exp_'+pos])**2 < spot_prop['ang_rad']**2
        
            x_st_grid, y_st_grid, z_st_grid = frameconv_skystar_to_star(x_sky_grid[cond_close_to_spot],
                                                                                        y_sky_grid[cond_close_to_spot],
                                                                                        z_sky_grid[cond_close_to_spot],
                                                                                        np.arccos(star_param['cos_istar']))
            
            
        
        # Retrieve angular coordinates of spot
        cos_long, sin_long, cos_lat, sin_lat = spot_prop['cos_long_exp_'+pos], spot_prop['sin_long_exp_'+pos], spot_prop['cos_lat_exp_'+pos], spot_prop['sin_lat_exp_'+pos]
        
        # Calculate coordinates in spot rest frame
        x_sp =                       x_st_grid*cos_long - z_st_grid*sin_long
        y_sp = y_st_grid*cos_lat  - (x_st_grid*sin_long + z_st_grid*cos_long)   *   sin_lat
        z_sp = y_st_grid*sin_lat  + (x_st_grid*sin_long + z_st_grid*cos_long)   *   cos_lat
        
        # Deduce which cells are within the spot
        phi_sp = np.arctan2(np.sqrt(x_sp**2. + y_sp**2.),z_sp)
        pos_cond_in_sp = cond_close_to_spot
        pos_cond_in_sp[cond_close_to_spot] = (phi_sp <= spot_prop['ang_rad'])
    
        cond_in_sp |= pos_cond_in_sp

    # Check if at least one tile is within the spot
    spot_within_grid = (True in cond_in_sp)   
    

    return spot_within_grid, cond_in_sp





def calc_spotted_region_prop(line_occ_HP_band,cond_occ,spot_prop,iband,system_prop,system_spot_prop, par_star, sp_proc_band, spot_consid, Ssub_Sstar_sp, Ssub_Sstar_ref, Istar_norm_band, sum_prop_dic_spot,\
                                    coord_reg_dic_spot, range_reg, Focc_star_band, par_list, range_par_list, args, cb_band, pl_loc_x = {}, pl_loc_y = {}, oversamp_idx = None, RpRs = None, plocc = False) :
    
    r"""**Spot-occulted properties: define and update**
    
    Identify the spot-occulted region in each exposure provided and calculate its properties. 
    Update the provided dictionaries which contain the average and sum of the properties of interest over the spot-occulted region.

    Args:
        line_occ_HP_band (str) : the precision with which to process the exposure.
        cond_occ (bool) : whether there is an occultation by at least one spot in the exposure considered.
        spot_prop (dict) : formatted spot properties Dictionary (see retrieve_spots_prop_from_param).
        iband (int) : index of the band used to retrieve the corresponding planet and spot limb-darkening properties.
        system_prop (dict) : quiet star limb-darkening properties.
        system_spot_prop (dict) : spot limb-darkening properties.
        star_params (dict) : star properties.
        sp_proc_band (list) : spots previously processed. Used to account for the overlap of spots.
        spot_consid (str) : name of the spot being processed.
        Ssub_Sstar_sp (float): surface of grid cells in the spot-occulted region grid.
        Ssub_Sstar_ref (float) : surface of grid cells in the stellar grid.
        Istar_norm_band (float) : total intensity of the star in the band considered.
        sum_prop_dic_spot (dict) : dictionary containing the value of all parameters of interest (par_list), summed over the spotted region in the exposure considered, and for the band of interest.
        coord_reg_dic_spot (dict) : dictionary containing the value of all parameters of interest (par_list), averaged over the spotted region in the exposure considered, and for the band of interest.
        range_reg (dict) : dictionary containing the range of average values the parameters of interest (range_par_list) can take during this exposure.
        Focc_star_band (float) : total flux occulted by the spot in the exposure and band considered. 
        par_list (list) : List of parameters of interest, whose value in sum_prop_dict_spot will be updated.
        range_par_list (list) : list of parameters of interest, whose range of values, stored in range_reg, will be updated.
        args (dict) : parameters used to generate the intrinsic profiles.
        cb_band (list) : coefficients used to calculate the convective blueshift RV contribution.
        pl_loc_x (dict) : optional, x coordinates of the planets' oversampled positions in the exposure considered. Used to account for the cells in the spot-occulted region that are occulted by the planet.
        pl_loc_y (dict) : optional, y coordinates of the planets' oversampled positions in the exposure considered. Used to account for the cells in the spot-occulted region that are occulted by the planet.
        oversamp_idx (int) : optional, index of the oversampled exposure being processed.
        RpRs (dict) : optional, planets' limb-darkening properties.
        plocc (bool) : optional, whether the exposure considered is being occulted by planets or not. Turned off by default.

    Returns:
        Focc_star_band (float) : the input Focc_star_band updated with the flux occulted by the spot considered.
        cond_occ (bool) : updated version of the input cond_occ. Tells us whether or not the spot occulted the exposure considered.
    
    """ 
    
    parameter_list = deepcopy(par_list)
    range_parameter_list = deepcopy(range_par_list)
    
    #We have as input a grid discretizing the spot.
    #We have a condition to find the cells in the input grid that are in the stellar grid.
    cond_in_star = spot_prop[spot_consid]['x_sky_grid']**2 + spot_prop[spot_consid]['y_sky_grid']**2 < 1.

    #We have a condition to figure out which cells in this input grid are occulted.
    ##Take the cells that are in the stellar grid.
    new_x_sky_grid = spot_prop[spot_consid]['x_sky_grid'][cond_in_star]
    new_y_sky_grid = spot_prop[spot_consid]['y_sky_grid'][cond_in_star]

    ##Retrieve the z-coordinate for the cells.
    new_z_sky_grid = np.sqrt(1 - new_x_sky_grid**2 - new_y_sky_grid**2)

    ##Move coordinates to the star reference frame and then the spot reference frame
    x_st_grid, y_st_grid, z_st_grid = frameconv_skystar_to_star(new_x_sky_grid, new_y_sky_grid, new_z_sky_grid, par_star['istar_rad'])
    x_spot_grid = x_st_grid*spot_prop[spot_consid]['cos_long_exp_center'] - z_st_grid*spot_prop[spot_consid]['sin_long_exp_center']
    y_spot_grid = y_st_grid*spot_prop[spot_consid]['cos_lat_exp_center'] - (z_st_grid*spot_prop[spot_consid]['cos_long_exp_center'] + x_st_grid*spot_prop[spot_consid]['sin_long_exp_center']) * spot_prop[spot_consid]['sin_lat_exp_center']
    z_spot_grid = y_st_grid*spot_prop[spot_consid]['sin_lat_exp_center'] + (z_st_grid*spot_prop[spot_consid]['cos_long_exp_center'] + x_st_grid*spot_prop[spot_consid]['sin_long_exp_center']) * spot_prop[spot_consid]['cos_lat_exp_center']
    cond_in_sp = x_spot_grid**2. + y_spot_grid**2. <= spot_prop[spot_consid]['ang_rad']**2

    #Making our refined spot-occultation grid and calculating the number of points it has
    spot_x_sky_grid = new_x_sky_grid[cond_in_sp]
    spot_y_sky_grid = new_y_sky_grid[cond_in_sp]
    spot_z_sky_grid = new_z_sky_grid[cond_in_sp]
    n_occ_sp = np.sum(cond_in_sp)

    #--------------------------------
    # Accounting for planet occultation of the spot
    if plocc:
        for planet in list(pl_loc_x.keys()):
            cond_in_planet_disk = (spot_x_sky_grid - pl_loc_x[planet][oversamp_idx])**2 + (spot_y_sky_grid -  pl_loc_y[planet][oversamp_idx])**2 < RpRs[planet][iband]**2
            spot_x_sky_grid = spot_x_sky_grid[~cond_in_planet_disk]
            spot_y_sky_grid = spot_y_sky_grid[~cond_in_planet_disk]
            spot_z_sky_grid = spot_z_sky_grid[~cond_in_planet_disk]
            n_occ_sp -= np.sum(cond_in_planet_disk)

    # Accounting for overlap with other spots - spot's have the same contrast so we can just remove the overlapping cells
    if len(sp_proc_band)>0:
        for sp_prev in sp_proc_band:

            #Move current spot coordinates to the star reference frame and then the other spot's reference frame
            updated_x_st_grid, updated_y_st_grid, updated_z_st_grid = frameconv_skystar_to_star(spot_x_sky_grid, spot_y_sky_grid, spot_z_sky_grid, par_star['istar_rad'])
            x_prev_spot_grid = updated_x_st_grid*spot_prop[sp_prev]['cos_long_exp_center'] - updated_z_st_grid*spot_prop[sp_prev]['sin_long_exp_center']
            y_prev_spot_grid = updated_y_st_grid*spot_prop[sp_prev]['cos_lat_exp_center'] - (updated_z_st_grid*spot_prop[sp_prev]['cos_long_exp_center'] + updated_x_st_grid*spot_prop[sp_prev]['sin_long_exp_center']) * spot_prop[sp_prev]['sin_lat_exp_center']
            cond_in_prev_spot = x_prev_spot_grid**2. + y_prev_spot_grid**2 <= system_spot_prop[sp_prev][iband]**2

            #Remove cells that overlap
            spot_x_sky_grid = spot_x_sky_grid[~cond_in_prev_spot]
            spot_y_sky_grid = spot_y_sky_grid[~cond_in_prev_spot]
            spot_z_sky_grid = spot_z_sky_grid[~cond_in_prev_spot]
            n_occ_sp -= np.sum(cond_in_prev_spot)
    sp_proc_band += [spot_consid]


    #--------------------------------

    #Figure out the number of cells occulted and store it
    if n_occ_sp > 0:
        cond_occ = True

    #Making the grid of coordinates for the calc_Isurf_grid function.
    coord_grid = {}
    #Getting, x, y, z, sky-projected radius, and number of occulted cells.
    coord_grid['x_st_sky'] = spot_x_sky_grid
    coord_grid['y_st_sky'] = spot_y_sky_grid
    coord_grid['z_st_sky'] = spot_z_sky_grid

    coord_grid['r2_st_sky']=coord_grid['x_st_sky']*coord_grid['x_st_sky']+coord_grid['y_st_sky']*coord_grid['y_st_sky']

    coord_grid['nsub_star'] = n_occ_sp

    #Getting the coordinates in the star rest frame
    coord_grid['x_st'], coord_grid['y_st'], coord_grid['z_st'] = frameconv_skystar_to_star(coord_grid['x_st_sky'], coord_grid['y_st_sky'], coord_grid['z_st_sky'], par_star['istar_rad'])
    
    #Retrieve the stellar flux grids over this local occulted-region grid.
    _,_,mu_grid_occ,Fsurf_grid_occ,Ftot_occ,_ = calc_Isurf_grid([iband], coord_grid['nsub_star'], system_prop, coord_grid, par_star, Ssub_Sstar_sp, Istar_norm_band, region='pl', Ssub_Sstar_ref=Ssub_Sstar_ref)

    #Retrieve the flux grid for the spot's emission (since spots have different LD law compared to the 'quiet' stellar surface)
    _,_,_,Fsurf_grid_emit,Ftot_emit,_ = calc_Isurf_grid([iband], coord_grid['nsub_star'], system_spot_prop, coord_grid, par_star, Ssub_Sstar_sp, Istar_norm_band, region='pl', Ssub_Sstar_ref=Ssub_Sstar_ref)

    #Scale the flux grid to the desired level
    Fsurf_grid_occ *= par_star['cont']
    Fsurf_grid_emit *= par_star['cont']
    Ftot_occ *= par_star['cont']
    Ftot_emit *= par_star['cont']

    #--------------------------------

    #Updating the provided dictionaries 
    coord_grid['mu'] = mu_grid_occ[:,0]
    Focc_star_band += (Ftot_occ[0] - (Ftot_emit[0]*(1-spot_prop[spot_consid]['ctrst'])))
    sum_prop_dic_spot['nocc'] += coord_grid['nsub_star']
    
    #Distance from projected orbital normal in the sky plane, in absolute value
    if 'xp_abs' in par_list : par_list.remove('xp_abs')
    if 'xp_abs' in range_par_list : range_par_list.remove('xp_abs')    
    
    #--------------------------------
    #Co-adding properties from current region to the cumulated values over oversampled spot positions 
    sum_region_prop(line_occ_HP_band,iband,args,parameter_list,Fsurf_grid_occ[:,0],Fsurf_grid_emit[:,0],coord_grid,Ssub_Sstar_sp,cb_band,range_parameter_list,range_reg,sum_prop_dic_spot,coord_reg_dic_spot,par_star,None,spot_prop[spot_consid]['ctrst']) 

    return Focc_star_band, cond_occ


