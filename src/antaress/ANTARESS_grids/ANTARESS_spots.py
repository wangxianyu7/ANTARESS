#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
from itertools import product as it_product
from ..ANTARESS_general.utils import np_poly,stop,check_data
from ..ANTARESS_grids.ANTARESS_coord import frameconv_skystar_to_star,frameconv_star_to_skystar
from ..ANTARESS_analysis.ANTARESS_model_prof import poly_prop_calc
from ..ANTARESS_analysis.ANTARESS_inst_resp import convol_prof
from ..ANTARESS_analysis.ANTARESS_ana_comm import par_formatting
from ..ANTARESS_grids.ANTARESS_coord import calc_zLOS_oblate
from ..ANTARESS_grids.ANTARESS_star_grid import calc_GD, calc_LD, calc_RVrot, calc_Isurf_grid
from ..ANTARESS_grids.ANTARESS_prof_grid import init_st_intr_prof, coadd_loc_line_prof
from ..ANTARESS_analysis.ANTARESS_model_prof import calc_linevar_coord_grid, calc_polymodu





'''
Calculation of stellar surface properties from spotted regions
    - we do not pre-calculate a grid to be shifted to the spot location, as with planets, because the projected area of the spot does not remain a disk
'''
def calc_spot_region_prop(bjd_sp,spot_prop,x_st_grid,y_st_grid,z_st_grid,Ssub_Sstar,r2_st_sky,par_star,LD_mod,ld_coeff,cb_band,par_list,range_par_list,range_reg,sum_prop_dic,Ftot_star_grid):

    #Stellar surface period at spot latitude (days)
    #    - accounting for differential rotation: 
    # om(lat) = om_eq*(1-alpha*sin(th_lat)^2)
    # P(lat) = 2*np.pi/(om_eq*(1-alpha*sin(th_lat)^2))
    #    - th_lat is the angular latitude of the spot, counted between the stellar equator and the star-spot center axis  
    th_lat_rad = spot_prop['th_lat']*np.pi/180.
    lat_sp = np.sin(th_lat_rad)
    P_spot = 2*np.pi/((1.-par_star['alpha_rot']*lat_sp**2.-par_star['beta_rot']*lat_sp**4.)*par_star['om_eq']*3600.*24.)

    #Spot phase
    #    - null phase corresponds to the crossing of the projected spin-axis
    Tcen_sp = spot_prop['Tcenter'] - 2400000.  
    ph_rad=(bjd_sp-Tcen_sp)/P_spot

    #Grid coordinates from star rest frame to rotated star frame
    #    - Xstar is the inclined LOS (by istar), Ystar its perpendicular in the star equatorial plane, Zstar the spin axis
    #    - rotation by spot phase angle, so that X_sp_star is the projection of the star-spot axis onto the equatorial plane
    x_sp_star =  x_st_grid*np.cos(ph_rad) + y_st_grid*np.sin(ph_rad)   
    y_sp_star = -x_st_grid*np.sin(ph_rad) + y_st_grid*np.cos(ph_rad)
    z_sp_star =  deepcopy(z_st_grid)

    #Grid coordinates in 'spot rest frame'    
    #    - Xsp axis is the star-spot center axis
    #      Ysp axis is the perpendicular in the plane parallel to the stellar equatorial plane, in the direction of the stellar rotation
    #      Zsp completes the referential
    x_sp =  x_sp_star*np.cos(th_lat_rad) + z_sp_star*np.sin(th_lat_rad) 
    y_sp =  y_sp_star
    z_sp = -x_sp_star*np.sin(th_lat_rad) + z_sp_star*np.cos(th_lat_rad) 

    #Cells within the spot
    #    - ang_sp is the (half) angular size of the spot, so that its projected radius when the spot is seen face-on is Rstar*sin(ang_sp)
    #    - condition is phi(cell) <= ang_sp, with phi counted from the star-spot center axis:
    # phi(cell) = arctan( sqrt(Ysp^2 + Zsp^2)/Xsp ) 
    phi_sp = np.arctan2(np.sqrt(y_sp**2. + z_sp**2.),x_sp)
    cond_in_sp = phi_sp <= spot_prop['ang_sp']*np.pi/180.    
    if True in cond_in_sp:
        region_prop = {}

        #Stellar flux from spot subcells
        #    - assuming a specific intensity at disk center of 1, without LD and GD contributions
        #    - Fr_sp is a flux scaling factor of the spot flux
        flux_grid = np.ones(np.sum(cond_in_sp))*Ssub_Sstar*spot_prop['Fr_sp']

        #Mu coordinate     
        region_prop['mu'] = np.sqrt(1. - r2_st_sky[cond_in_sp]) 
        
        #Limb-darkening 
        flux_grid*= calc_LD(LD_mod,region_prop['mu'],ld_coeff)        
        
        #Modify stellar flux from current spot
        #    - we assume the effect of spots is cumulative
        if Ftot_star_grid is not None:
            Ftot_star_grid[cond_in_sp]*=spot_prop['Fr_sp']
                         
        #Processing requested properties
        #    - obliquity is set to 0 as spots are assumed to evolve in planes parallel to the equator
        sum_region_prop(par_list,flux_grid,region_prop,Ssub_Sstar,cb_band,range_par_list,range_reg,sum_prop_dic,par_star,0.,'')


######## appeler avant calcul des prop planetaires
######## updater et sauver la grille stellaire pour chaque exposition
####### ensuite dans la routine planetaire, occulter la grill de l'expo, et faire gaffe a recalculer le flux total par expo aussi


    
    return Ftot_star_grid



"""

Fonction which decides if a spot is visible or not, based on star inclination (istar), spot coordinates in star rest frame (long, lat) and spot angular size (ang).

The fonction discretizes the spot circle and test for each point if it visible, thanks this calculation :

    Let's P = (x_st, y_st, z_st) be a point of stellar surface (star rest frame), with X axis along stellar equator and Y axis along stellar spin.

    Writing P in spherical coordinateed in star rest frame gives :

    x_st = sin(long)cos(lat)
    y_st = sin(lat)
    z_st = cos(long)cos(lat)

    We rotate this vector by angle (pi/2-istar) around X axis, moving it in the 'inclined' star frame (with Z now along the LOS, and Y the projected stellar spin):

    x_sky = x_st
    y_sky = sin(istar)y_st - cos(istar)z_st
    z_sky = cos(istar)y_st + sin(istar)z_st,

    with Y_sky axis along the line of sight and X_sky still along the star equator.

    The condition for P to be visible then reads z_sky > 0, yielding :

                                            cos(istar)sin(lat) + sin(istar)cos(long)cos(lat) > 0


    With gravity darkening : we replace y_st by sin(lat)*(1-f), which yields :

                                          cos(istar)sin(lat)(1-f) + sin(istar)cos(long)cos(lat) > 0

"""


def is_spot_visible(istar, long_rad, lat_rad, ang_rad, f_GD, RpoleReq) :
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









"""

Fucntion which transforms a parameters list with spots properties (eg, lat__ISinst_VSvis_SPspot) into a more convenient dictionary of the form : 

# spot_prop = {
#   'spot1' : {lat : 0,   ang : 0,   Tcenter : 0,   flux : 0,   lat_rad_exp_center : 0,   sin_lat_exp_center : 0,    ... }
#   'spot2  : { ... }
#    }


# We assume spot parameters are never defined as common for all visit or all inst.

"""

def retrieve_spots_prop_from_param(star_params, param, inst, vis, t_bjd): 

    spots_prop = {}
    for par in param : 
        # Parameter is spot-related and linked to the right visit and instrument
        if ('_SP' in par) and ('_IS'+inst in par) and ('_VS'+vis in par) : 
            spot_name = par.split('_SP')[1]
            spot_par = par.split('__IS')[0]
            if spot_name not in spots_prop : spots_prop[spot_name] = {}
            spots_prop[spot_name][spot_par] = param[par]
            
    # Retrieve if spots are visible (at exposure center) 
    for spot in spots_prop : 
    
        # Spot lattitude
        lat_rad = spots_prop[spot]['lat']*np.pi/180.
        
        # Spot longitude
        sin_lat = np.sin(lat_rad)
        P_spot = 2*np.pi/((1.-param['alpha_rot']*sin_lat**2.-param['beta_rot']*sin_lat**4.)*star_params['om_eq']*3600.*24.)
        Tcen_sp = spots_prop[spot]['Tcenter'] - 2400000.
        long_rad = (t_bjd-Tcen_sp)/P_spot * 2*np.pi
        
        # Spot center coordinates in star rest frame
        x_st = np.sin(long_rad)*np.cos(lat_rad)
        y_st = np.sin(lat_rad)
        z_st = np.cos(long_rad)*np.cos(lat_rad)
    
        # inclined frame
        istar = np.arccos(param['cos_istar'])
        x_sky,y_sky,z_sky = frameconv_star_to_skystar(x_st,y_st,z_st,istar)
       
        # Store properties at exposure center
        spots_prop[spot]['lat_rad_exp_center'] = lat_rad
        spots_prop[spot]['sin_lat_exp_center'] = np.sin(lat_rad)
        spots_prop[spot]['cos_lat_exp_center'] = np.cos(lat_rad)
        spots_prop[spot]['long_rad_exp_center'] = long_rad
        spots_prop[spot]['sin_long_exp_center'] = np.sin(long_rad)
        spots_prop[spot]['cos_long_exp_center'] = np.cos(long_rad)
        spots_prop[spot]['x_sky_exp_center'] = x_sky
        spots_prop[spot]['y_sky_exp_center'] = y_sky
        spots_prop[spot]['z_sky_exp_center'] = z_sky
        spots_prop[spot]['ang_rad'] = spots_prop[spot]['ang'] * np.pi/180
        spots_prop[spot]['is_visible'] = is_spot_visible(istar,long_rad, lat_rad, spots_prop[spot]['ang_rad'], star_params['f_GD'], star_params['RpoleReq'])

    return spots_prop
             
            
            
            
            
            
            
            


"""

Function which calculates which tiles of the input sky grid are spotted

2 options : 

    + use_grid_dic = False : calculation will be performed on the   x_sky_grid, y_sky_grid, z_sky_grid   args (can be either a planetary or stellar grid),
                             by moving these grids from inclined star to star rest frame
    + use_grid_dic = True : calculation will be performed on the star grid contained in grid_dic['x/y/z_st'], which is already in star rest frame (no frame conversion needed). 
                            This option is used for calculating spotted stellar tiles, when istar is not fitted.
                            
                            
Calculation is straighforward : 

 - We rotate the star grid by the longitude of the spot around stellar spin : 
 
    x_sp_star =  x_st_grid*np.cos(long_rad) - z_st_grid*np.sin(long_rad)
    y_sp_star =  deepcopy(y_st_grid)
    z_sp_star =  x_st_grid*np.sin(long_rad) + z_st_grid*np.cos(long_rad)
    
    
 - We rotate the new grid by the lattitude of the spot, moving it to the spot rest frame
 
    x_sp =   deepcopy(x_sp_star)
    y_sp =   y_sp_star*np.cos(lat_rad) - z_sp_star*np.sin(lat_rad)
    z_sp =   y_sp_star*np.sin(lat_rad) + z_sp_star*np.cos(lat_rad)
    
    
 - We then check wich cells are within the spot by evaluing : 
 
                                            np.arctan2(np.sqrt(x_sp**2. + y_sp**2.),z_sp)     <     ang_sp
    
     
"""

def calc_spotted_tiles(spot_prop, x_sky_grid, y_sky_grid, z_sky_grid, grid_dic, param, use_grid_dic = False) :

    #Samson: check whether the zstar and sskystar coordinates need to be adapted in this routine                                            

    if use_grid_dic :
        cond_close_to_spot = (grid_dic['x_st_sky'] - spot_prop['x_sky_exp_center'])**2 + (grid_dic['y_st_sky'] - spot_prop['y_sky_exp_center'])**2 < spot_prop['ang_rad']**2
     
        x_st_grid, y_st_grid, z_st_grid = grid_dic['x_st'][cond_close_to_spot], grid_dic['y_st'][cond_close_to_spot], grid_dic['z_st'][cond_close_to_spot]
        
        
    else :  
        cond_close_to_spot = (x_sky_grid - spot_prop['x_sky_exp_center'])**2 + (y_sky_grid - spot_prop['y_sky_exp_center'])**2 < spot_prop['ang_rad']**2
    
        x_st_grid, y_st_grid, z_st_grid = frameconv_skystar_to_star(x_sky_grid[cond_close_to_spot],
                                                                                    y_sky_grid[cond_close_to_spot],
                                                                                    z_sky_grid[cond_close_to_spot],
                                                                                    np.arccos(param['cos_istar']))
        
        
    
    # Retrieve angular coordinates of spot
    cos_long, sin_long, cos_lat, sin_lat =  spot_prop['cos_long_exp_center'],  spot_prop['sin_long_exp_center'],                             spot_prop['cos_lat_exp_center' ],  spot_prop['sin_lat_exp_center' ]
    
    # Calculate coordinates in spot rest frame
    x_sp =                         x_st_grid*cos_long - z_st_grid*sin_long
    y_sp = y_st_grid*cos_lat  - (x_st_grid*sin_long + z_st_grid*cos_long)   *   sin_lat
    z_sp = y_st_grid*sin_lat  + (x_st_grid*sin_long + z_st_grid*cos_long)   *   cos_lat
   
    # Deduce which cells are within the spot
    phi_sp = np.arctan2(np.sqrt(x_sp**2. + y_sp**2.),z_sp)
    cond_in_sp = cond_close_to_spot
    cond_in_sp[cond_close_to_spot] = (phi_sp <= spot_prop['ang_rad'])
        
    # Check if at least one tile is within the spot
    spot_within_grid = (True in cond_in_sp)   
    

    return spot_within_grid, cond_in_sp







"""

Function which calculates the properties of spot-occulted stellar cells 

   + For each spot, we check if it is visible (with spots_prop[spot]['is_visible']), and if so, we use the previous function to calculate which cells of the stellar grid it occults
   + We store one global list of all spotted cells of the star (cond_in_sp), and their base flux level, calculated as the product of spot flux (flux_emitted_all_tiles_sp)
   + We then deduce the absorbed flux, as well as all the other properties of spotted tiles (RV, mu, ctrst, ...), exactly like in get_planet_disk_prop.

"""

def calc_spotted_region_prop(spots_prop, grid_dic, star_params, LD_law, ld_coeff, gd_band, cb_band, param, coeff_ord2name, dim, func_prof_name, var_par_list, pol_mode) :
    
    #Samson: check whether the zstar and sskystar coordinates need to be adapted in this routine; the region_prop['y_st_sp'] may not be calculated properly
    
    # Nombre de cases de l'étoile
    n_tiles = len(grid_dic['x_st_sky'])

    # On stocke la liste des cases stellaires occultées par au moins 1 spot, et la liste des flux occultés par les spots sur chaque case. On cummule l'occultation des spots si 2 ou plus overlappent. 


    flux_emitted_all_tiles_sp = np.ones(n_tiles, dtype = float)
    cond_in_sp = np.zeros(n_tiles, dtype = bool)
    spot_within_grid_all = False
    for spot in spots_prop :
        if spots_prop[spot]['is_visible'] : 
            if 'cos_istar' in var_par_list : use_grid_dic = False
            else : use_grid_dic = True
            spot_within_grid, cond_in_one_sp = calc_spotted_tiles(spots_prop[spot],
                                    grid_dic['x_st_sky'], grid_dic['y_st_sky'], grid_dic['z_st_sky'], grid_dic,
                                    star_params, use_grid_dic)
            if spot_within_grid:
                spot_within_grid_all = True
                flux_emitted_all_tiles_sp[cond_in_one_sp] *= spots_prop[spot]['flux']
                cond_in_sp |= cond_in_one_sp
                
    flux_occulted_all_tiles_sp = 1 - flux_emitted_all_tiles_sp
    
    region_prop = {}
    
    # Star is effectively affected by (at least) one spot
    if spot_within_grid_all :


        ## Coordinates calculation

        # Coordonnées x, y, z, r des régions spottées, référentiel incliné
        region_prop['r_proj2_st_sky_sp']  = grid_dic['r2_st_sky'][cond_in_sp]
        region_prop['x_st_sky_sp']        = grid_dic['x_st_sky'][cond_in_sp]
        region_prop['y_st_sky_sp']        = grid_dic['y_st_sky'][cond_in_sp]
        region_prop['z_st_sky_sp']        = grid_dic['z_st_sky'][cond_in_sp]

        #Frame conversion from the inclined star frame to the 'star' frame
        region_prop['x_st_sp'],region_prop['y_st_sp'],region_prop['z_st_sp'] = frameconv_skystar_to_star(region_prop['x_st_sky_sp'],
                                                                                                                      region_prop['y_st_sky_sp'], 
                                                                                                                      region_prop['z_st_sky_sp'], 
                                                                                                                      np.arccos(param['cos_istar']))
        


        ## Flux calculation


        # On garde uniquement les cases spottées, en gardant pour chaque case le flux du spot le plus sombre qui la recouvre (cf flux_occulted_all_tiles_sp)
        region_prop['flux_sp'] = flux_occulted_all_tiles_sp[cond_in_sp]*grid_dic['Ssub_Sstar']    #Vincent to Samson: since flux_occulted_all_tiles_sp is a flux ratio (= 1 - x) I think the flux does need to be multiplied by Ssub_Sstar, as Ftot_star (and thus Ftot_star_achrom, which we use below to normalize flux_sp) in calc_Isurf_grid()

        # If GD is on AND istar is fitted, then we need to recalculate mu, LD and GD
        if   (gd_band is not None)    and    ('cos_istar' in var_par_list)   :
            
            gd_sp, mu_sp = calc_GD(region_prop['x_st_sp'],
                                   region_prop['y_st_sp'],
                                   region_prop['z_st_sp'], 
                                   star_params, gd_band, 
                                   region_prop['x_st_sky_sp'],
                                   region_prop['y_st_sky_sp'], 
                                   np.arccos(param['cos_istar']))
                                   
            ld_sp = calc_LD(LD_law, mu_sp, ld_coeff) 
        
        # Otherwise it's ok, we can retrieve mu, LD and GD from those contained in grid_dic
        else: 
            mu_sp = grid_dic['mu_grid_star_achrom'][cond_in_sp][:,0]
            gd_sp = grid_dic['gd_grid_star_achrom'][cond_in_sp][:,0]
            ld_sp = grid_dic['ld_grid_star_achrom'][cond_in_sp][:,0]
            

        region_prop['flux_sp'] *= ld_sp
        region_prop['flux_sp'] *= gd_sp
        region_prop['mu_sp']    = mu_sp

        # Renormalisation to take into account that sum(Ftile) < 1 :
        region_prop['flux_sp'] /= grid_dic['Ftot_star_achrom'][0]
        

        ## Radial velocity calculation

        # Rotation speed
        region_prop['RV_sp'] = calc_RVrot(region_prop['x_st_sky_sp'],region_prop['y_st_sp'],star_params['istar_rad'],param)[0]

        # Systemic velocity
        region_prop['RV_sp'] += param['rv']

        # Convectivd blueshift
        CB_sp = np_poly(cb_band)(region_prop['mu_sp'])
        region_prop['RV_sp'] += CB_sp


        ## Other properties calculation : FW, ctrst, ...


        # We store the coordinates associated with the chosen dimension( mu, r_proj,.. Add more possible coord choice ? )
        if dim == 'mu'        : coord_prop = region_prop['mu_sp']
        if dim == 'r_proj'    : coord_prop = np.sqrt(region_prop['r_proj2_st_sky_sp'])
        
        # FW et ctrst : always useful
        region_prop['FWHM_sp']     = poly_prop_calc(param,coord_prop,coeff_ord2name['FWHM' ], pol_mode) 
        region_prop['ctrst_sp']    = poly_prop_calc(param,coord_prop,coeff_ord2name['ctrst'], pol_mode)

        # Cas à deux gaussiennes
        if func_prof_name == 'dgauss' :
            region_prop['amp_l2c_sp'] = poly_prop_calc(param,coord_prop,coeff_ord2name['amp_l2c'], pol_mode)
            region_prop['rv_l2c_sp'] = poly_prop_calc(param,coord_prop,coeff_ord2name['rv_l2c'], pol_mode)
            region_prop['FWHM_l2c_sp'] = poly_prop_calc(param,coord_prop,coeff_ord2name['FWHM_l2c'], pol_mode)

        # Cas profil 'voigt'
        if func_prof_name == 'voigt' :
            region_prop['a_damp_pl'] = poly_prop_calc(param,coord_prop,coeff_ord2name['a_damp'], pol_mode)

        #On retire les champs inutiles dans region_prop :
        for key in ['r_proj2_st_sky_sp', 'x_st_sky_sp', 'y_st_sky_sp', 'z_st_sky_sp', 'x_st_sp', 'y_st_sp', 'z_st_sp', 'mu_sp'] :
           region_prop.pop(key)
    
    return cond_in_sp, (   spot_within_grid_all and np.any(region_prop['flux_sp'] > 0)   ), region_prop









"""

Final function which compute the deviation of an exposure from the 'normal' stellar CCF

+ We first calculate the properties of spot-occulted and planet-occulted cells 
+ We then compute the corresponding profiles, with the chosen calculation mode (see args['precision']).
+ Note that spot properties are assumed constant throughout the exposure

"""

def compute_deviation_profile(args, param, inst, vis, iexp,gen_dic,theo_dic,data_dic,coord_dic,system_param) :
    
    # Retrieve spot parameters
    spot_within_grid_all = False
    if inst in args['spots_prop']:
        spots_prop = retrieve_spots_prop_from_param(args['star_params'], param, inst, vis, args['t_exp_bjd'][inst][vis][iexp])

      
    

        # Spots occulted tiles
        
        # Properties are stores in a single dic : 
        
        # - tab_prop_spot['rv'][i_tile]   -> RV of the spotted tile number 'itile'
        # - tab_prop_spot['flux'][i_tile] -> flux level of the spotted tile number 'itile'
        # etc...

        cond_sp, spot_within_grid_all, tab_prop_sp = calc_spotted_region_prop(
                                                        spots_prop,
                                                        args['grid_dic'],
                                                        args['t_exp_bjd'][inst][vis][iexp],
                
                                                        args['star_params'],
                                                        LD_law,
                                                        ld_coeff,
                                                        gd_band,
                                                        cb_band,
                                                        param,
                                                        args['coeff_ord2name'][inst][vis],
                                                        args['coord_line'],
                                                        args['func_prof_name'][inst],
                                                        args['var_par_list'],
                                                        args['pol_mode'])    
      
    else:spots_prop={}


    #-----------------------------------------------    

    prof_deviation = np.zeros(len(args['cen_bins']))


  
  
    #Retrieve planetary coordinates
    # for pl_loc in gen_dic['studied_pl']:
        # x_oversamp_pl, y_oversamp_pl, n_disk = calc_pl_coord_sky(args, param, pl_loc, inst, vis, iexp,theo_dic)
        

        # Planet occulted tile
        
        # Properties are stores in a single dic : 
        # - tab_prop_pl['rv'][i_disk][i_tile] -> RV of the tile number 'itile' of the disk number 'idisk'
    
        # n_disk_in_transit = 0
        # tab_prop_pl = {}
        # if args['calc_pl_flux'] : 
        #     for idisk in range(n_disk) :
        
        #         disk_in_transit, region_prop_pl = get_planet_disk_prop(
        #                                         spots_prop,
        #                                         pl_loc,
        #                                         theo_dic,
        #                                         args['system_prop'],
        #                                         x_oversamp_pl[idisk],
        #                                         y_oversamp_pl[idisk],
        #                                         star_params,
        #                                         LD_law,
        #                                         ld_coeff,
        #                                         gd_band,
        #                                         cb_band,
        #                                         param,
        #                                         args['coeff_ord2name'][inst][vis],
        #                                         args['func_prof_name'],
        #                                         args['system_prop'][pl_loc][0],
        #                                         args['var_par_list'],
        #                                         args['pol_mode'],args)
        
        #         if disk_in_transit :
        #             n_disk_in_transit += 1
        #             for prop in region_prop_pl :
        #                 if prop not in tab_prop_pl : tab_prop_pl[prop] = []
        #                 tab_prop_pl[prop].append(region_prop_pl[prop])



    # We then compute the deviation profile

    # - For spots, we compute one profile per tile, we the computed properties
    # - For the planet, the calculation mode depends on args['precision']

    # Spot contribution
    if spots_prop != {} and spot_within_grid_all :

        for itile in range(len(tab_prop_sp['flux_sp'])) :

            input_exp = {'cont'  :  tab_prop_sp['flux_sp'] [itile],
                         'rv'    :  tab_prop_sp['RV_sp']   [itile],
                         'ctrst' :  tab_prop_sp['ctrst_sp'][itile],
                         'FWHM'  :  tab_prop_sp['FWHM_sp'] [itile]
                        }

            # Cas demi-gaussienne :
            if args['func_prof_name'] == 'dgauss' :
                input_exp['rv_l2c'  ] = tab_prop_sp['rv_l2c_sp']   [itile]
                input_exp['amp_l2c' ] = tab_prop_sp['amp_l2c']     [itile]
                input_exp['FWHM_l2c'] = tab_prop_sp['FWHM_l2c_sp'] [itile]


            # Cas voigt :
            if args['func_prof_name'] == 'voigt' :
                input_exp['a_damp'] = tab_prop_sp ['a_damp_sp'][itile]

            # On ajoute le profil de la case au profil de la déviation
            prof_deviation +=  args['func_prof'][inst](input_exp, args['cen_bins'][inst][vis])[0]       # On l'ajoute au profil local du disque









    # # We store spot occulted flux, useful for the corr_spot module
    # spot_occulted_prof = deepcopy(prof_deviation)
    
    
    # # Retrieving total occulted flux (ie, expected continuum) and mean RV of planet-occulted region : 
    # occulted_flux = 0 
    # mean_prop = {'rv' : 0, 'FWHM' : 0, 'ctrst' : 0, 'mu' : 0}    #j'ai mis mu temporarieemnt
    
    # # From spots
    # if spots_prop != {} and spot_within_grid_all : occulted_flux += np.sum(tab_prop_sp['flux_sp'])
    # spot_occulted_flux = occulted_flux
    
    # # From the planet
    # if n_disk_in_transit >= 1 : 
    #     for idisk in range(n_disk_in_transit) : 
        
    #         # Flux
    #         occulted_flux += np.sum(tab_prop_pl['flux_pl'][idisk])/n_disk
            
    #         # Mean prop
    #         if args['calc_pl_mean_prop'] : 
    #             mean_prop['rv']    += np.sum(    tab_prop_pl['flux_pl'][idisk] * tab_prop_pl['RV_pl'][idisk]     /   np.sum(tab_prop_pl['flux_pl'][idisk])   )   /   n_disk
    #             mean_prop['FWHM']  += np.sum(    tab_prop_pl['flux_pl'][idisk] * tab_prop_pl['FWHM_pl'][idisk]   /   np.sum(tab_prop_pl['flux_pl'][idisk])   )   /   n_disk
    #             mean_prop['ctrst'] += np.sum(    tab_prop_pl['flux_pl'][idisk] * tab_prop_pl['ctrst_pl'][idisk]  /   np.sum(tab_prop_pl['flux_pl'][idisk])   )   /   n_disk
    #             # if args['coord_line'] == 'mu':
    #             mean_prop['mu'] += np.sum(   tab_prop_pl['flux_pl'][idisk] * tab_prop_pl['mu_pl'][idisk]     /   np.sum(tab_prop_pl['flux_pl'][idisk])   )   /   n_disk
    #             # if args['coord_line'] == 'r_proj' : 
    #             #     mean_prop['r_proj'] += np.sum(    tab_prop_pl['flux_pl'][idisk] * np.sqrt(tab_prop_pl['r_proj2_sky_pl'][idisk])  /   np.sum(tab_prop_pl['flux_pl'][idisk])   )   /   n_disk

    return prof_deviation


#Function to calculate the properties of the region occulted by a single spot. The previous function (calc_spotted_region_prop) did it for all the spots considered.
def new_calc_spotted_region_prop(spot_prop, grid_dic, star_params, LD_law, ld_coeff, gd_band, cb_band, param, coeff_ord2name, dim, func_prof_name, var_par_list, pol_mode) :
    
    #Samson: check whether the zstar and sskystar coordinates need to be adapted in this routine; the region_prop['y_st_sp'] may not be calculated properly
    
    # Nombre de cases de l'étoile
    n_tiles = len(grid_dic['x_st_sky'])

    # On stocke la liste des cases stellaires occultées par au moins 1 spot, et la liste des flux occultés par les spots sur chaque case. On cummule l'occultation des spots si 2 ou plus overlappent. 
    flux_emitted_all_tiles_sp = np.ones(n_tiles, dtype = float)
    cond_in_sp = np.zeros(n_tiles, dtype = bool)
    spot_within_grid = False
    if spot_prop['is_center_visible'] : 
        if 'cos_istar' in var_par_list : use_grid_dic = False
        else : use_grid_dic = True
        spot_within_grid, cond_in_one_sp = calc_spotted_tiles(spot_prop,
                                grid_dic['x_st_sky'], grid_dic['y_st_sky'], grid_dic['z_st_sky'], grid_dic,
                                star_params, use_grid_dic)
        if spot_within_grid:
            flux_emitted_all_tiles_sp[cond_in_one_sp] *= spot_prop['atten']
            cond_in_sp |= cond_in_one_sp
                
    flux_occulted_all_tiles_sp = 1 - flux_emitted_all_tiles_sp
    
    region_prop = {}
    
    # Star is effectively affected by (at least) one spot
    if spot_within_grid :


        ## Coordinates calculation

        # Coordonnées x, y, z, r des régions spottées, référentiel incliné
        region_prop['r_proj2_st_sky_sp']  = grid_dic['r2_st_sky'][cond_in_sp]
        region_prop['x_st_sky_sp']        = grid_dic['x_st_sky'][cond_in_sp]
        region_prop['y_st_sky_sp']        = grid_dic['y_st_sky'][cond_in_sp]
        region_prop['z_st_sky_sp']        = grid_dic['z_st_sky'][cond_in_sp]

        #Frame conversion from the inclined star frame to the 'star' frame
        region_prop['x_st_sp'],region_prop['y_st_sp'],region_prop['z_st_sp'] = frameconv_skystar_to_star(region_prop['x_st_sky_sp'],
                                                                                                                      region_prop['y_st_sky_sp'], 
                                                                                                                      region_prop['z_st_sky_sp'], 
                                                                                                                      np.arccos(param['cos_istar']))
        


        ## Flux calculation


        # On garde uniquement les cases spottées, en gardant pour chaque case le flux du spot le plus sombre qui la recouvre (cf flux_occulted_all_tiles_sp)
        region_prop['flux_sp'] = flux_occulted_all_tiles_sp[cond_in_sp]#*grid_dic['Ssub_Sstar']

        # If GD is on AND istar is fitted, then we need to recalculate mu, LD and GD
        if   (gd_band is not None)    and    ('cos_istar' in var_par_list)   :
            
            gd_sp, mu_sp = calc_GD(region_prop['x_st_sp'],
                                   region_prop['y_st_sp'],
                                   region_prop['z_st_sp'], 
                                   star_params, gd_band, 
                                   region_prop['x_st_sky_sp'],
                                   region_prop['y_st_sky_sp'], 
                                   np.arccos(param['cos_istar']))
                                   
            ld_sp = calc_LD(LD_law, mu_sp, ld_coeff) 
        
        # Otherwise it's ok, we can retrieve mu, LD and GD from those contained in grid_dic
        else: 
            mu_sp = grid_dic['mu_grid_star_achrom'][cond_in_sp][:,0]
            gd_sp = grid_dic['gd_grid_star_achrom'][cond_in_sp][:,0]
            ld_sp = grid_dic['ld_grid_star_achrom'][cond_in_sp][:,0]
            

        region_prop['flux_sp'] *= ld_sp
        region_prop['flux_sp'] *= gd_sp
        region_prop['mu_sp']    = mu_sp


        #Limb-Darkening coefficient at mu
        # region_prop['flux_sp'] *= LD_mu_func(LD_law,region_prop['mu_sp'],ld_coeff)

        # Renormalisation to take into account that sum(Ftile) < 1 :
        region_prop['flux_sp'] /= grid_dic['Ftot_star_achrom'][0]
        

        ## Radial velocity calculation
        # Rotation speed
        region_prop['RV_sp'] = calc_RVrot(region_prop['x_st_sky_sp'],region_prop['y_st_sp'],star_params['istar_rad'],param)

        # Systemic velocity
        region_prop['RV_sp'] += param['rv']

        # Convectivd blueshift
        CB_sp = np_poly(cb_band)(region_prop['mu_sp'])
        region_prop['RV_sp'] += CB_sp


        ## Other properties calculation : FW, ctrst, ...


        # We store the coordinates associated with the chosen dimension( mu, r_proj,.. Add more possible coord choice ? )
        if dim == 'mu'        : coord_prop = region_prop['mu_sp']
        if dim == 'r_proj'    : coord_prop = np.sqrt(region_prop['r_proj2_st_sky_sp'])
        
        # FW et ctrst : always useful
        region_prop['FWHM_sp']     = poly_prop_calc(param,coord_prop,coeff_ord2name['FWHM' ], pol_mode) 
        region_prop['ctrst_sp']    = poly_prop_calc(param,coord_prop,coeff_ord2name['ctrst'], pol_mode)

        # Cas à deux gaussiennes
        if func_prof_name == 'dgauss' :
            region_prop['amp_l2c_sp'] = poly_prop_calc(param,coord_prop,coeff_ord2name['amp_l2c'], pol_mode)
            region_prop['rv_l2c_sp'] = poly_prop_calc(param,coord_prop,coeff_ord2name['rv_l2c'], pol_mode)
            region_prop['FWHM_l2c_sp'] = poly_prop_calc(param,coord_prop,coeff_ord2name['FWHM_l2c'], pol_mode)

        # Cas profil 'voigt'
        if func_prof_name == 'voigt' :
            region_prop['a_damp_pl'] = poly_prop_calc(param,coord_prop,coeff_ord2name['a_damp'], pol_mode)

        #On retire les champs inutiles dans region_prop :
        for key in ['r_proj2_st_sky_sp', 'x_st_sky_sp', 'y_st_sky_sp', 'z_st_sky_sp', 'x_st_sp', 'y_st_sp', 'z_st_sp', 'mu_sp'] :
           region_prop.pop(key)
    
    return cond_in_sp, (   spot_within_grid and np.any(region_prop['flux_sp'] > 0)   ), region_prop



   
   
   
"""

Fucntion which transforms a parameters list with spots properties (eg, lat__ISinst_VSvis_SPspot) into a more convenient dictionary of the form : 

# spot_prop = {
#   'spot1' : {lat : 0,   ang : 0,   Tcenter : 0,   flux : 0,   lat_rad_exp_center : 0,   sin_lat_exp_center : 0,    ... }
#   'spot2  : { ... }
#    }


# We assume spot parameters are never defined as common for all visit or all inst. 

#This function also now returns the spot position at the start, center and end of exposure (really useful for oversampling later).

"""

def new_retrieve_spots_prop_from_param(star_params, param, inst, vis, t_bjd, exp_dur): 

    spots_prop = {}
    for par in param : 
        # Parameter is spot-related and linked to the right visit and instrument
        if ('_SP' in par) and ('_IS'+inst in par) and ('_VS'+vis in par) : 
            spot_name = par.split('_SP')[1]
            spot_par = par.split('__IS')[0]
            if spot_name not in spots_prop : spots_prop[spot_name] = {}
            spots_prop[spot_name][spot_par] = param[par]
            
    # Retrieve properties, if spots are visible in the exposures considered
    for spot in spots_prop : 
        #Finding the times at the center, start and end of each exposure considered
        t_dur_days = exp_dur/(24.*3600.)
        t_bjd_center = t_bjd
        t_bjd_start = t_bjd - t_dur_days/2
        t_bjd_end = t_bjd + t_dur_days/2
        # Spot lattitude - constant in time
        lat_rad = spots_prop[spot]['lat']*np.pi/180.
        
        # Spot longitude - varies over time
        sin_lat = np.sin(lat_rad)
        P_spot = 2*np.pi/((1.-param['alpha_rot']*sin_lat**2.-param['beta_rot']*sin_lat**4.)*star_params['om_eq']*3600.*24.)
        Tcen_sp = spots_prop[spot]['Tcenter'] - 2400000.
        long_rad_center = (t_bjd_center-Tcen_sp)/P_spot * 2*np.pi
        long_rad_start = (t_bjd_start-Tcen_sp)/P_spot * 2*np.pi
        long_rad_end = (t_bjd_end-Tcen_sp)/P_spot * 2*np.pi
        
        # Spot center coordinates in star rest frame
        #Exposure center
        x_st_center = np.sin(long_rad_center)*np.cos(lat_rad)
        y_st_center = np.sin(lat_rad)
        z_st_center = np.cos(long_rad_center)*np.cos(lat_rad)
        #Exposure start
        x_st_start = np.sin(long_rad_start)*np.cos(lat_rad)
        y_st_start = np.sin(lat_rad)
        z_st_start = np.cos(long_rad_start)*np.cos(lat_rad)
        #Exposure end
        x_st_end = np.sin(long_rad_end)*np.cos(lat_rad)
        y_st_end = np.sin(lat_rad)
        z_st_end = np.cos(long_rad_end)*np.cos(lat_rad)

        # inclined frame
        istar = np.arccos(param['cos_istar'])
        #Exposure center
        x_sky_center,y_sky_center,z_sky_center = frameconv_star_to_skystar(x_st_center,y_st_center,z_st_center,istar)
        #Exposure start
        x_sky_start,y_sky_start,z_sky_start = frameconv_star_to_skystar(x_st_start,y_st_start,z_st_start,istar)
        #Exposure end
        x_sky_end,y_sky_end,z_sky_end = frameconv_star_to_skystar(x_st_end,y_st_end,z_st_end,istar)
       
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



def spot_occ_region_grid(RspRs, nsub_Dsp):
    r"""**Spot grid** 

    Defines grid discretizing the spot-occulted region
      
     - X axis is parallel to the star equator
     - Y axis is the projected spin axis
     - Z axis is the LOS

    Args:
        RspRs : The angular radius of the spot.
        nsub_Dsp : The number of grid cells desired.
    
    Returns:
        x_st_sky_grid : The x-coordinate of the grid cells.
        y_st_sky_grid : The y-coordinate of the grid cells.
        Ssub_Sstar : The surface of the grid cells.
    
    """ 
    #Subcell width (in units of Rstar) and surface (in units of Rstar^2 and pi*Rstar^2) 
    d_sub=2.*RspRs/nsub_Dsp
    Ssub_Sstar=d_sub*d_sub/np.pi

    #Coordinates of points discretizing the enclosing square
    cen_sub=-RspRs+(np.arange(nsub_Dsp)+0.5)*d_sub            
    xy_st_sky_grid=np.array(list(it_product(cen_sub,cen_sub)))

    # #Keeping only grid points behind the spot
    x_st_sky_grid=xy_st_sky_grid[:,0]
    y_st_sky_grid=xy_st_sky_grid[:,1] 

    return x_st_sky_grid,y_st_sky_grid, Ssub_Sstar



def new_new_calc_spotted_region_prop(line_occ_HP_band, cond_occ, spot_prop, iband, system_spot_prop, star_params, Ssub_Sstar_sp, Ssub_Sstar_ref, Istar_norm_band, par_star, sum_prop_dic_spot,\
                                    coord_reg_dic_spot, range_reg_dic_spot, Focc_star_band, par_list, range_par_list, args, cb_band, pl_loc_x = None, pl_loc_y = None, RpRs = None, plocc = False) :
    
    r"""**Spot-occulted properties: define and update**
    
    Refine the spot-occulted grid and calculate the properties of the spot-occulted region.
    Updates the average and summed properties from a spot-occulted stellar surface region during an exposure.

    Args:
        line_occ_HP_band : string telling us the precision with which to process the exposure in this particular band.
        cond_occ : Boolean telling us whether there is an occultation in the (oversampled) exposure and band considered.
        spot_prop : Dictionary (formatted with new_retrieve_spots_prop_from_param) containing the spot properties.
        iband : The index of the band used to calculate the spot LD properties.
        system_spot_prop : Dictionary containing the spot LD properties.
        star_params : Dictionary containing the stellar parameters.
        Ssub_Sstar_sp : The cell surface from the grid used to discretize the spot-occulted region.
        Ssub_Sstar_ref : The cell surface from the grid used to discretize the star.
        Istar_norm_band : The total intensity of the star in the given band.
        par_star : ???
        sum_prop_dic_spot : Dictionary containing the value of all parameters of interest (par_list), summed over the region occulted by the spot of interest in the exposure considered, 
                            and for the band of interest.
        coord_reg_dic_spot : Dictionary containing the value of all parameters of interest (par_list), averaged over the region occulted by the spot of interest in the exposure considered, 
                            and for the band of interest.
        range_reg_dic_spot : Dictionary containing the range of average values the parameters of interest (range_par_list) can take during this exposure.
        Focc_star_band : Float storing the total flux occulted by the spot considered in the exposure considered, in the band considered. 
        par_list : List of parameters of interest, whose value in sum_prop_dict_spot will be updated.
        range par_list : List of parameters of interest, whose range of values, stored in range_reg_dic_spot, will be updated.
        args : Dictionary containing the parameters used to generate the intrinsic profiles (not sure this is all it contains).
        cb_band : List containing the coefficients to calculate the convective blueshift RV contribution.
    Returns:
        Focc_star_band : The input Focc_star_band updated with the flux occulted by the spot in the exposure being processed.
    
    """ 
    
    #Samson: check whether the zstar and sskystar coordinates need to be adapted in this routine
    
    parameter_list = deepcopy(par_list)
    range_parameter_list = deepcopy(range_par_list)
    #We have as input a grid discretizing the spot.
    #We have a condition to find the cells in the input grid that are in the stellar grid.
    cond_in_star = spot_prop['x_sky_grid']**2 + spot_prop['y_sky_grid']**2 < 1.

    #We have a condition to figure out which cells in this input grid are occulted.
    ##Take the cells that are in the stellar grid.
    new_x_sky_grid = spot_prop['x_sky_grid'][cond_in_star]
    new_y_sky_grid = spot_prop['y_sky_grid'][cond_in_star]

    ##Retrieve the z-coordinate for the cells.
    new_z_sky_grid = np.sqrt(1 - new_x_sky_grid**2 - new_y_sky_grid**2)

    ##Move coordinates to the star reference frame and then the spot reference frame
    x_st_grid, y_st_grid, z_st_grid = frameconv_skystar_to_star(new_x_sky_grid, new_y_sky_grid, new_z_sky_grid, star_params['istar_rad'])

    x_spot_grid = x_st_grid*spot_prop['cos_long_exp_center'] - z_st_grid*spot_prop['sin_long_exp_center']

    y_spot_grid = y_st_grid*spot_prop['cos_lat_exp_center'] - (z_st_grid*spot_prop['cos_long_exp_center'] + x_st_grid*spot_prop['sin_long_exp_center']) * spot_prop['sin_lat_exp_center']

    z_spot_grid = y_st_grid*spot_prop['sin_lat_exp_center'] + (z_st_grid*spot_prop['cos_long_exp_center'] + x_st_grid*spot_prop['sin_long_exp_center']) * spot_prop['cos_lat_exp_center']

    cond_in_sp = x_spot_grid**2. + y_spot_grid**2. <= spot_prop['ang_rad']**2

    #--------------------------------
    # Accounting for planet occultation of the spot
    if plocc:
        cond_in_planet_disk = (new_x_sky_grid[cond_in_sp] - pl_loc_x)**2 + (new_y_sky_grid[cond_in_sp] - pl_loc_y)**2 < RpRs**2
        spot_x_sky_grid = new_x_sky_grid[cond_in_sp][~cond_in_planet_disk]
        spot_y_sky_grid = new_x_sky_grid[cond_in_sp][~cond_in_planet_disk]
        spot_z_sky_grid = new_x_sky_grid[cond_in_sp][~cond_in_planet_disk]
        n_occ_sp = np.sum(cond_in_sp) - np.sum(cond_in_planet_disk)
    else:
        spot_x_sky_grid = new_x_sky_grid[cond_in_sp]
        spot_y_sky_grid = new_y_sky_grid[cond_in_sp]
        spot_z_sky_grid = new_z_sky_grid[cond_in_sp]
        n_occ_sp = np.sum(cond_in_sp)

    #--------------------------------

    #Figure out the number of cells occulted and store it - account for overlap when using oversampling
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
    coord_grid['x_st'], coord_grid['y_st'], coord_grid['z_st'] = frameconv_skystar_to_star(coord_grid['x_st_sky'], coord_grid['y_st_sky'], coord_grid['z_st_sky'], star_params['istar_rad'])
    
    #Retrieve the stellar flux grids over this local occulted-region grid.
    ld_grid_occ, gd_grid_occ, mu_grid_occ, Fsurf_grid_occ, Ftot_occ, Istar_occ = calc_Isurf_grid([iband], coord_grid['nsub_star'], system_spot_prop, coord_grid, star_params, Ssub_Sstar_sp, Istar_norm_band, region='pl', Ssub_Sstar_ref=Ssub_Sstar_ref)

    #Scale the flux grid to the desired level
    Fsurf_grid_occ *= par_star['cont']
    Ftot_occ *= par_star['cont']

    #--------------------------------

    #Updating the provided dictionaries 

    coord_grid['mu'] = mu_grid_occ[:,0]
    Focc_star_band += Ftot_occ[0]
    sum_prop_dic_spot['nocc'] += coord_grid['nsub_star']
    
    #Remove xp_abs from the list if it's in there
    if 'xp_abs' in parameter_list : parameter_list.remove('xp_abs')
    if 'xp_abs' in range_parameter_list : range_parameter_list.remove('xp_abs')

    #Sky-projected distance from star center
    if ('r_proj' in parameter_list) or (('coord_line' in args) and (args['coord_line']=='r_proj')):coord_grid['r_proj'] = np.sqrt(coord_grid['r2_st_sky'])                   

    for par_loc in parameter_list:

        #Ratio of the occulted surface to the star surface
        if par_loc=='SpSstar':
            sum_prop_dic_spot[par_loc][iband] += Ssub_Sstar_sp*coord_grid['nsub_star']
            coord_reg_dic_spot[par_loc][iband] = Ssub_Sstar_sp*coord_grid['nsub_star']

        else:
            #Flux level from region occulted by the planet alone
            #    - set to 1 since it is weighted by flux afterward
            if par_loc=='Ftot':coord_grid[par_loc] = 1.                    

            #Longitude and Latitude of occulted region - in degrees
            # sin(lat) = Ystar / Rstar
            # sin(lon) = Xstar / Rstar
            if par_loc=='lat':coord_grid[par_loc] = np.arcsin(coord_grid['y_st'])*180./np.pi    
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
                if 'CB_RV' in parameter_list:coord_grid[par_loc]+=coord_grid['CB_RV']
                if 'rv_line' in parameter_list:coord_grid[par_loc]+=coord_grid['rv_line'] 

    #--------------------------------

            #Sum property over occulted region, weighted by stellar flux
            #    - we use flux rather than intensity, because local flux level depend on the spot grid resolution
            #    - total RVs from spot-occulted region is set last in par_list to calculate all rv contributions first:
            # + rotational contribution is always included
            # + disk-integrated-corrected convective blueshift polynomial (in km/s)   
            coord_grid[par_loc+'_sum'] = np.sum(coord_grid[par_loc]*Fsurf_grid_occ[:,0])

            #Cumulate property over successively occulted regions
            sum_prop_dic_spot[par_loc][iband]+=coord_grid[par_loc+'_sum'] 

            #Total flux from current occulted region
            if par_loc=='Ftot':coord_reg_dic_spot['Ftot'][iband] = coord_grid['Ftot_sum']

            #Calculate average property over current occulted region  
            #    - <X> = sum(cell, xcell*fcell)/sum(cell,fcell)           
            else:coord_reg_dic_spot[par_loc][iband] = coord_grid[par_loc+'_sum']/coord_grid['Ftot_sum'] 

            #Range of values covered during the exposures (normalized)
            #    - for spatial-related coordinates
            if par_loc in range_parameter_list:
                range_reg_dic_spot[par_loc+'_range'][iband][0]=np.min([range_reg_dic_spot[par_loc+'_range'][iband][0],coord_reg_dic_spot[par_loc][iband]])
                range_reg_dic_spot[par_loc+'_range'][iband][1]=np.max([range_reg_dic_spot[par_loc+'_range'][iband][1],coord_reg_dic_spot[par_loc][iband]])

    #------------------------------------------------    
    #Calculate line profile from average of cell profiles over current region
    #    - this high precision mode is only possible for achromatic or closest-achromatic mode 
    if line_occ_HP_band=='high':    
        
        #Attribute intrinsic profile to each cell 
        init_st_intr_prof(args,coord_grid,par_star)

        #Calculate individual local line profiles from all region cells
        #    - analytical intrinsic profiles are fully calculated 
        #      theoretical and measured intrinsic profiles have been pre-defined and are just shifted to their position
        #    - in both cases a scaling is then applied to convert them into local profiles
        line_prof_grid=coadd_loc_line_prof(coord_grid['rv'],range(coord_grid['nsub_star']),Fsurf_grid_occ[:,0],args['flux_intr_grid'],coord_grid['mu'],par_star,args)          
        
        #Calculate line profile emitted by the spot
        emit_line_prof_grid = coadd_loc_line_prof(coord_grid['rv'],range(coord_grid['nsub_star']),(1-spot_prop['atten'])*Fsurf_grid_occ[:,0],args['flux_intr_grid'],coord_grid['mu'],par_star,args)          
        
        #Coadd line profiles over spot-occulted region
        sum_prop_dic_spot['line_prof'] = np.sum((np.array(line_prof_grid)-np.array(emit_line_prof_grid)),axis=0) 

    #Define rotational broadening of planet-occulted region
    elif line_occ_HP_band in ['low','medium']:
        drv_min = coord_reg_dic_spot['rv'][iband]-np.min(coord_grid['rv'])
        drv_max = np.max(coord_grid['rv'])-coord_reg_dic_spot['rv'][iband] 
        coord_reg_dic_spot['rv_broad'][iband] = 0.5*(drv_min+drv_max)       

    return Focc_star_band, cond_occ












    
    
    
    
    



                
   
   
   






"""

Routine to correction DI profiles from spot contamination
Profile are assumed to be aligned in star rest frame 

"""

def corr_spot(corr_spot_dic, coord_dic,inst,vis,data_dic,data_prop,gen_dic, theo_dic,system_param) :
    star_params = system_param['star']
    
    
    print('   > Correcting DI CCF from spot contamination' )
    proc_gen_data_paths_new = gen_dic['save_data_dir']+'Spot_corr_DI_data/'+gen_dic['add_txt_path']['DI']+'/'+inst+'_'+vis+'_'

    if gen_dic['calc_correct_spots']:
        print('         Calculating data')
    
        data_vis = data_dic[inst][vis]
        data_DI_prop_vis = np.load(gen_dic['save_data_dir']+'DIorig_prop/'+inst+'_'+vis+'.npz',  allow_pickle=True)['data'].item()
        pl_loc = list(gen_dic['transit_pl'])[0]
    
    
    
        for iexp in range(data_vis['n_in_visit']):    
    
            # Load exp profile
            data_exp = np.load(data_dic[inst][vis]['proc_DI_data_paths']+str(iexp)+'.npz',allow_pickle=True)['data'].item()
    
            if corr_spot_dic['spots_prop'][inst][vis] != {} : 
            
                if iexp == 0 :
                    
                    # Initialize args for 'compute_deviation_profile' function, common to all exposures
                    fixed_args = {}
                    fixed_args['inst_list']=[inst]
                    fixed_args['inst_vis_list']={inst:[vis]}
                    fixed_args['pl_loc'] = pl_loc
                    fixed_args['cen_bins'] = {inst : {vis : data_exp['cen_bins'][0]}}
                    fixed_args['dcen_bin'] = {inst : {vis : data_exp['cen_bins'][0][1]  -  data_exp['cen_bins'][0][0]}}
                    fixed_args['system_param'] = system_param
                    fixed_args['grid_dic'] = theo_dic     
                    fixed_args['grid_dic']['cond_in_RpRs'] = {pl_loc : data_dic['DI']['system_prop']['achrom']['cond_in_RpRs'][pl_loc]}
                    fixed_args['system_prop'] = data_dic['DI']['system_prop']['achrom']
                    fixed_args['coord_line'] = corr_spot_dic['coord_line']
                    fixed_args['func_prof'] = {inst : dispatch_func_prof(corr_spot_dic['intr_prof']['func_prof_name'][inst])}
                    fixed_args['func_prof_name'] =  corr_spot_dic['intr_prof']['func_prof_name']
                    fixed_args['pol_mode'] = corr_spot_dic['intr_prof']['pol_mode']
                    fixed_args['phase']     = {inst : {vis :  [coord_dic[inst][vis][pl_loc][m] for m in ['st_ph','cen_ph','end_ph']]}}
                    fixed_args['t_exp_bjd'] = {inst : { vis : coord_dic[inst][vis]['bjd']   }}
                    fixed_args['precision'] = corr_spot_dic['precision']
                    fixed_args['var_par_list'] = []
                    fixed_args['print_exp'] = False
                    fixed_args['calc_pl_mean_prop'] = False
                    
                    # Paramètres susceptibles d'être fittés (voir gen_dic['fit_ResProf'])
                    params = {'rv' : 0}
                    for par in ['veq','alpha_rot','beta_rot','cos_istar','c1_CB','c2_CB','c3_CB'] : params[par]=system_param['star'][par]
                    params['lambda_rad__pl'+pl_loc] = system_param[pl_loc]['lambda_rad']
                    params['aRs__pl'+pl_loc] = system_param[pl_loc]['aRs']
                    params['inclin_rad__pl'+pl_loc] = system_param[pl_loc]['inclin_rad']
                        
                    # Propriétés du profils stellaire
                    params = par_formatting(params,corr_spot_dic['intr_prof']['mod_prop'],None,None,fixed_args,inst,vis,line_type) 
                    params_without_spot = deepcopy(params)
                    
                    # Propriétés des spots   
                    par_formatting(params,corr_spot_dic['spots_prop'][inst][vis],None,None,fixed_args,inst,vis,line_type)
                    params_with_spot = deepcopy(params)
                
                
                # Load exposure continuum
                cont_exp = data_DI_prop_vis[iexp]['cont']
        
                # Version with overlapping took into account
                if corr_spot_dic['overlap'] :
                    fixed_args['calc_pl_flux'] = True
                    tot_occulted_prof, tot_occulted_flux = compute_deviation_profile(fixed_args, params_with_spot,    inst, vis, iexp,star_params,gen_dic,theo_dic,data_dic,coord_dic)[0:2]
                    pl_occulted_prof , pl_occulted_flux  = compute_deviation_profile(fixed_args, params_without_spot, inst, vis, iexp,star_params,gen_dic,theo_dic,data_dic,coord_dic)[0:2]
                    spot_occulted_prof, spot_occulted_flux = tot_occulted_prof - pl_occulted_prof  ,  tot_occulted_flux - pl_occulted_flux
                
                # Version which assumes that spots and planet are always separated. 
                else : 
                    fixed_args['calc_pl_flux'] = False
                    spot_occulted_prof, spot_occulted_flux = compute_deviation_profile(fixed_args, params_with_spot,    inst, vis, iexp,star_params,gen_dic,theo_dic,data_dic,coord_dic)[2:4]
                    fixed_args['calc_pl_flux'] = True
                    pl_occulted_flux                       = compute_deviation_profile(fixed_args, params_without_spot, inst, vis, iexp,star_params,gen_dic,theo_dic,data_dic,coord_dic)[1]
                
                # Set the DI profile continuum to the expected value given planetary and spot occultation
                data_exp['flux'][0] *= (1 - spot_occulted_flux - pl_occulted_flux)  /  cont_exp
                
                # Add the profile occulted by spots, taking into account instrumental dispersion :
                data_exp['flux'][0] += convol_prof(spot_occulted_prof, fixed_args['cen_bins'][inst][vis], calc_FWHM_inst(inst))
                
                # Reset the exposure continuum to the initial value (might be important if the profiles are not fitted again, especially in the Joined Residual profiles fitting). 
                data_exp['flux'][0] *= cont_exp  /  (1 - pl_occulted_flux)
            
            # Save exp data
            np.savez_compressed(proc_gen_data_paths_new+str(iexp),data=data_exp,allow_pickle=True)

        # updating path to DI data
        data_vis['proc_DI_data_paths'] = proc_gen_data_paths_new

            
        
    else :
        check_data({'path':proc_gen_data_paths_new+'_add'}) 
        data_vis['proc_DI_data_paths'] = proc_gen_data_paths_new


    return None









  

'''
Sub-function to calculate theoretical properties of spots
'''
def calc_spots_prop(gen_dic,star_params,theo_dic,inst,data_dic):
    print('   > Calculating properties of spots')    
    if star_params['f_GD']>0.:stop('Spot processing undefined for oblate stars')

    if (gen_dic['calc_theo_spots']):
        print('         Calculating data')


        
        #Process spots in each requested visit
        for vis in np.intersect1d(theo_dic['spots_prop'][inst],data_dic[inst]['visit_list']):
            for spot in theo_dic['spots_prop'][inst][vis]:
                spot_prop = theo_dic['spots_prop'][inst][vis][spot]
    

# faire une boucle d'oversampling et dedans, pour chaque expo, boucler sur les spots

               # calc_spot_region_prop(bjd_osamp,spot_prop,theo_dic['x_st'],theo_dic['y_st'],theo_dic['z_st'],theo_dic['Ssub_Sstar'],theo_dic['r2_st_sky'],par_star,LD_law,ld_coeff)
    

    
    ##boucler sur tous les spots d'abord
    ##et ensuite boucler sur oversamp 
    
    
                # #Coordinates of spot contour in 'spot rest frame'
                # #    - Xsp axis is the star-spot center axis
                # #      Ysp axis is the perpendicular in the plane parallel to the stellar equatorial plane, in the direction of the stellar rotation
                # #      Zsp completes the referential
                # nlimb = 501
                # th_limb = 2*np.pi*np.arange(nlimb)/(nlimb-1.)
                # x_sp = np.repeat(np.cos(ang_sp_rad) , nlimb)
                # y_sp = np.sin( ang_sp_rad )*np.cos(th_limb)
                # z_sp = np.sin( ang_sp_rad )*np.sin(th_limb)
    



###attention inclure ce calcul dans une sous-routine d'oversampling


        ####  je pense que la meilleure approche c de faire comme pour les contacts du transit oblate
        ####  je definis l'enveloppe HR du spot dans le ref ou il est aligne avec la LOS. Depend que de sa taille
        ## ensuite je shifte les coord avec l'offset en phase (definir pr chaque spot un T0 comme les TTV)
        ## ensuite je shifte l'offset en latitude
        ## ensuite je peux identifier quelles cellules dans le ref stellaire (spin axis dans plan ciel) sont occultees
        ## avec ca je peux utiliser mes routines deja existantes pour calculer les props des spots (RV et Ftot surtout)
        
        ## dans le custom model, je ferai pareil : dans une expo donnee je sais ou est le spot, j'identifie ses cellules, et au lieu de la CCF normale
        ## je peux y mettre une CCF * facteur de scaling du flux (qui serait une autre des props des spots)

### refechir a implementer spots avec transits: je prends les cells occultee par une planete, je checke si elles sont dans tous les spots, et si oui je change
### le flux de ces cellules 


        stop('spots')

    
    return None




"""

Function which calculates the properties of the stellar surface elements occulted by the planetary grid (x_st_sky_pl , y_st_sky_pl)

"""


def get_planet_disk_prop(spots_prop, pl_loc, grid_dic,system_prop, x_pos_pl, y_pos_pl, star_params, LD_law, ld_coeff, gd_band, cb_band, param, coeff_ord2name, func_prof_name, Rp_Rs, var_par_list, pol_mode,args) :

    #Samson: check whether the zstar and sskystar coordinates need to be adapted in this routine

    # #Shift planet grid to current planet position
    # x_st_sky_pl = x_pos_pl+grid_dic['x_st_sky_grid_pl'][pl_loc][  system_prop['cond_in_RpRs'][pl_loc][0]  ]
    # y_st_sky_pl = y_pos_pl+grid_dic['y_st_sky_grid_pl'][pl_loc][  system_prop['cond_in_RpRs'][pl_loc][0]  ]

    # #Distance to star center, squared
    # r_proj2_sky_pl=x_st_sky_pl*x_st_sky_pl+y_st_sky_pl*y_st_sky_pl

    # #Identifying planet subcells occulting the star
    # #    - see model_star() for details on oblate star
    # if star_params['f_GD']==0 :cond_pl_occ = ( r_proj2_sky_pl <=1. )
    # else : z_st_sky_pl, cond_pl_occ = calc_zLOS_oblate(x_st_sky_pl, y_st_sky_pl, np.arccos(param['cos_istar']), star_params['RpoleReq'])[1:3]



    #Star is effectively occulted
    region_prop = {}

    if True in cond_pl_occ:

        ## Coordinates calculation

        # # x, y, z, r of occulted cells, in the 'inclined' star frame
        # region_prop['r_proj2_sky_pl']  = r_proj2_sky_pl[cond_pl_occ]
        # region_prop['x_st_sky_pl'] = x_st_sky_pl[cond_pl_occ]
        # region_prop['y_st_sky_pl'] = y_st_sky_pl[cond_pl_occ]
        # if star_params['f_GD']>0.:region_prop['z_st_sky_pl']=z_st_sky_pl[cond_pl_occ]
        # else:region_prop['z_st_sky_pl']=np.sqrt(1.-region_prop['r_proj2_sky_pl'])

        # #Frame conversion from the inclined star frame to the 'star' frame
        # region_prop['x_st_pl'],region_prop['y_st_pl'],region_prop['zstar_pl'] = frameconv_skystar_to_star(region_prop['x_st_sky_pl'],
        #                             region_prop['y_st_sky_pl'], region_prop['z_st_sky_pl'], np.arccos(param['cos_istar']))

        ## Flux calculation

        # # Size of disk cells
        # region_prop['flux_pl'] = np.ones(np.sum(cond_pl_occ))*grid_dic['Ssub_Sstar_pl'][pl_loc]


        # # Mu coordinate and gravity-darkening
        # #     - mu = cos(theta)), from 1 at the center of the disk to 0 at the limbs), with theta angle between LOS and local normal
        # if gd_band is not None :
        #     gd_grid,region_prop['mu_pl']=calc_GD(region_prop['x_st_pl'],
        #                                         region_prop['y_st_pl'],
        #                                         region_prop['zstar_pl'], 
        #                                         star_params,gd_band,
        #                                         region_prop['x_st_sky_pl'],
        #                                         region_prop['y_st_sky_pl'], 
        #                                         np.arccos([param['cos_istar']]))
                                                
        #     region_prop['flux_pl'] *= gd_grid   # We correct the flux from gravity darkening effect (larger flux at the pole)

        # else: region_prop['mu_pl'] = np.sqrt(1. - region_prop['r_proj2_sky_pl']  )


        # # Limb-Darkening coefficient at mu
        # region_prop['flux_pl'] *= LD_mu_func(LD_law,region_prop['mu_pl'],ld_coeff)

        # # Renormalisation to take into account that sum(Ftile) < 1 :
        # region_prop['flux_pl'] /= grid_dic['Ftot_star_achrom'][0]


        ## Spot effect on flux 
        
        # 'Base' flux level of the tiles : 1 if off-spot, and product of the spot flux if the tiles belongs to one or more spots (spot effect are assumed cumulative) :
        spot_atenuation = np.ones(np.sum(cond_pl_occ))  
        
        for spot in spots_prop :
            # Check if the spot is visible and 'close' to the planet center (in inclined star frame) : 
            x_sp_sky, y_sp_sky, ang_sp = spots_prop[spot]['x_sky_exp_center'],spots_prop[spot]['y_sky_exp_center'],spots_prop[spot]['ang_rad']
            if spots_prop[spot]['is_visible'] and (x_sp_sky - x_pos_pl)**2 + (y_sp_sky - y_pos_pl)**2 < (ang_sp + Rp_Rs)**2:
                
                spot_within_grid, spotted_tiles = calc_spotted_tiles(spots_prop[spot], region_prop['x_st_sky_pl'], region_prop['y_st_sky_pl'], region_prop['z_st_sky_pl'], {},
                                                                     star_params, param, use_grid_dic = False)
                                                                     
                # Multiply spotted tiles flux by the spot flux                                                
                if spot_within_grid :
                    for i in range(np.sum(cond_pl_occ)):
                        if spotted_tiles[i] : spot_atenuation[i] *= spots_prop[spot]['flux']  
                        
        region_prop['flux_pl'] *= spot_atenuation


        ## Radial velocity calculation

        # Vitesse de rotation
        region_prop['RV_pl'] = calc_RVrot(region_prop['x_st_sky_pl'],region_prop['y_st_pl'],np.arccos(param['cos_istar']),param)

        # Vitesse systémique
        region_prop['RV_pl'] += param['rv']

        # Convective blueshift
        CB_pl = np_poly(cb_band)(region_prop['mu_pl'])
        region_prop['RV_pl'] += CB_pl


        ## Other properties calculation : FW, ctrst, ...
        
        # We store the coordinates associated with the chosen dimension( mu, r_proj,.. add more possible coord choice ? )
        # if dim == 'mu'        : coord_prop = region_prop['mu_pl']
        # if dim == 'r_proj'    : coord_prop = np.sqrt(region_prop['r_proj2_sky_pl'])
        
        coord_prop = args['linevar_coord_grid']
        
        # Contrast and FWHM, always used
        region_prop['FWHM_pl']     = poly_prop_calc(param,coord_prop,coeff_ord2name['FWHM' ], pol_mode)
        region_prop['ctrst_pl']    = poly_prop_calc(param,coord_prop,coeff_ord2name['ctrst'], pol_mode)

        # Cas à deux gaussiennes
        if func_prof_name == 'dgauss' :
            region_prop['amp_l2c_pl'] = poly_prop_calc(param,coord_prop,coeff_ord2name['amp_l2c'], pol_mode)
            region_prop['rv_l2c_pl'] = poly_prop_calc(param,coord_prop,coeff_ord2name['rv_l2c'], pol_mode)
            region_prop['FWHM_l2c_pl'] = poly_prop_calc(param,coord_prop,coeff_ord2name['FWHM_l2c'], pol_mode)

        # Cas d'un profil voigt
        if func_prof_name == 'voigt' :
            region_prop['a_damp_pl'] = poly_prop_calc(param,coord_prop,coeff_ord2name['a_damp'], pol_mode)

        # On supprime les champs inutiles
        for key in ['x_st_sky_pl', 'y_st_sky_pl', 'z_st_sky_pl', 'x_st_pl', 'y_st_pl', 'zstar_pl'] :
            region_prop.pop(key)

    # On renvoie si le disque recouvre l'étoile, et les propriétés de la région occultée.
    return (   (True in cond_pl_occ) and np.any(region_prop['flux_pl'] > 0)   ), region_prop





