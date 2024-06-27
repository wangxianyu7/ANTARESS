#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
from itertools import product as it_product
from ..ANTARESS_general.utils import np_poly,stop
from ..ANTARESS_grids.ANTARESS_coord import frameconv_skystar_to_star,frameconv_star_to_skystar,frameconv_skyorb_to_skystar,calc_zLOS_oblate
from ..ANTARESS_grids.ANTARESS_star_grid import calc_RVrot, calc_Isurf_grid
from ..ANTARESS_grids.ANTARESS_prof_grid import init_st_intr_prof, coadd_loc_line_prof, OS_coadd_loc_line_prof, use_C_OS_coadd_loc_line_prof
from ..ANTARESS_analysis.ANTARESS_model_prof import calc_linevar_coord_grid, calc_polymodu


def is_spot_visible(istar, long_rad, lat_rad, ang_rad, f_GD, RpoleReq) :
    r"""**Spot visibility**

    Performs a rough estimation of the visibility of a spot, based on star inclination (istar), spot coordinates in star rest frame (long, lat) and spot angular size (ang).
    To do so, we discretize the spot edge and check if at least one point is visible. The visibilty criterion is derived as follows: 

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
    #Samson: check whether the zstar and sskystar coordinates need to be adapted in this routine
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




def spot_occ_region_grid(RspRs, nsub_Dsp):
    r"""**Spot grid** 

    Defines x/y/z grid discretizing the spot-occulted region with:

     - X axis is parallel to the star equator
     - Y axis is the projected spin axis
     - Z axis is along the LOS

    Args:
        RspRs (int) : the angular radius of the spot.
        nsub_Dsp (int) : the number of grid cells desired.
    
    Returns:
        x_st_sky_grid (1D dict) : The x-coordinates of the grid cells.
        y_st_sky_grid (1D dict) : The y-coordinates of the grid cells.
        Ssub_Sstar (float) : The surface of each grid cell.
    
    """ 
    #Subcell width (in units of Rstar) and surface (in units of Rstar^2 and pi*Rstar^2) 
    d_sub=2.*RspRs/nsub_Dsp
    Ssub_Sstar=d_sub*d_sub/np.pi

    #Coordinates of points discretizing the enclosing square
    cen_sub=-RspRs+(np.arange(-2, nsub_Dsp+2)+0.5)*d_sub            
    xy_st_sky_grid=np.array(list(it_product(cen_sub,cen_sub)))

    # #Keeping only grid points behind the spot
    x_st_sky_grid=xy_st_sky_grid[:,0]
    y_st_sky_grid=xy_st_sky_grid[:,1] 

    return x_st_sky_grid,y_st_sky_grid, Ssub_Sstar




def new_new_calc_spotted_region_prop(line_occ_HP_band, cond_occ, spot_prop, iband, system_prop, system_spot_prop, star_params, sp_proc_band, spot_consid, Ssub_Sstar_sp, Ssub_Sstar_ref, Istar_norm_band, sum_prop_dic_spot,\
                                    coord_reg_dic_spot, range_reg_dic_spot, Focc_star_band, par_list, range_par_list, args, cb_band, pl_loc_x = {}, pl_loc_y = {}, oversamp_idx = None, RpRs = None, plocc = False) :
    #Samson: check whether the zstar and sskystar coordinates need to be adapted in this routine; the region_prop['y_st_sp'] may not be calculated properly
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
        star_params (dict) : nominal star properties.
        sp_proc_band (list) : spots previously processed. Used to account for the overlap of spots.
        spot_consid (str) : name of the spot being processed.
        Ssub_Sstar_sp (float): surface of grid cells in the spot-occulted region grid.
        Ssub_Sstar_ref (float) : surface of grid cells in the stellar grid.
        Istar_norm_band (float) : total intensity of the star in the band considered.
        sum_prop_dic_spot (dict) : dictionary containing the value of all parameters of interest (par_list), summed over the spotted region in the exposure considered, and for the band of interest.
        coord_reg_dic_spot (dict) : dictionary containing the value of all parameters of interest (par_list), averaged over the spotted region in the exposure considered, and for the band of interest.
        range_reg_dic_spot (dict) : dictionary containing the range of average values the parameters of interest (range_par_list) can take during this exposure.
        Focc_star_band (float) : total flux occulted by the spot in the exposure and band considered. 
        par_list (list) : List of parameters of interest, whose value in sum_prop_dict_spot will be updated.
        range_par_list (list) : list of parameters of interest, whose range of values, stored in range_reg_dic_spot, will be updated.
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
    x_st_grid, y_st_grid, z_st_grid = frameconv_skystar_to_star(new_x_sky_grid, new_y_sky_grid, new_z_sky_grid, star_params['istar_rad'])

    x_spot_grid = x_st_grid*spot_prop[spot_consid]['cos_long_exp_center'] - z_st_grid*spot_prop[spot_consid]['sin_long_exp_center']

    y_spot_grid = y_st_grid*spot_prop[spot_consid]['cos_lat_exp_center'] - (z_st_grid*spot_prop[spot_consid]['cos_long_exp_center'] + x_st_grid*spot_prop[spot_consid]['sin_long_exp_center']) * spot_prop[spot_consid]['sin_lat_exp_center']

    z_spot_grid = y_st_grid*spot_prop[spot_consid]['sin_lat_exp_center'] + (z_st_grid*spot_prop[spot_consid]['cos_long_exp_center'] + x_st_grid*spot_prop[spot_consid]['sin_long_exp_center']) * spot_prop[spot_consid]['cos_lat_exp_center']

    cond_in_sp = x_spot_grid**2. + y_spot_grid**2. <= spot_prop[spot_consid]['ang_rad']**2

    #--------------------------------
    # Accounting for planet occultation of the spot

    #Making our refined spot-occultation grid and calculating the number of points it has
    spot_x_sky_grid = new_x_sky_grid[cond_in_sp]
    spot_y_sky_grid = new_y_sky_grid[cond_in_sp]
    spot_z_sky_grid = new_z_sky_grid[cond_in_sp]
    n_occ_sp = np.sum(cond_in_sp)

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
            updated_x_st_grid, updated_y_st_grid, updated_z_st_grid = frameconv_skystar_to_star(spot_x_sky_grid, spot_y_sky_grid, spot_z_sky_grid, star_params['istar_rad'])
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
    coord_grid['x_st'], coord_grid['y_st'], coord_grid['z_st'] = frameconv_skystar_to_star(coord_grid['x_st_sky'], coord_grid['y_st_sky'], coord_grid['z_st_sky'], star_params['istar_rad'])
    
    #Retrieve the stellar flux grids over this local occulted-region grid.
    _,_,mu_grid_occ,Fsurf_grid_occ,Ftot_occ,_ = calc_Isurf_grid([iband], coord_grid['nsub_star'], system_prop, coord_grid, star_params, Ssub_Sstar_sp, Istar_norm_band, region='pl', Ssub_Sstar_ref=Ssub_Sstar_ref)

    #Retrieve the flux grid for the spot's emission (since spots have different LD law compared to the 'quiet' stellar surface)
    _,_,_,Fsurf_grid_emit,Ftot_emit,_ = calc_Isurf_grid([iband], coord_grid['nsub_star'], system_spot_prop, coord_grid, star_params, Ssub_Sstar_sp, Istar_norm_band, region='pl', Ssub_Sstar_ref=Ssub_Sstar_ref)

    #Scale the flux grid to the desired level
    Fsurf_grid_occ *= star_params['cont']
    Fsurf_grid_emit *= star_params['cont']
    Ftot_occ *= star_params['cont']
    Ftot_emit *= star_params['cont']

    #--------------------------------

    #Updating the provided dictionaries 
    coord_grid['mu'] = mu_grid_occ[:,0]
    Focc_star_band += (Ftot_occ[0] - (Ftot_emit[0]*(1-spot_prop[spot_consid]['ctrst'])))
    sum_prop_dic_spot['nocc'] += coord_grid['nsub_star']
    
    #--------------------------------
    #Co-adding properties from current region to the cumulated values over oversampled spot positions
    sum_region_spot_prop(line_occ_HP_band,iband,args,parameter_list,Fsurf_grid_occ[:,0],Fsurf_grid_emit[:,0],coord_grid,Ssub_Sstar_sp,cb_band,range_parameter_list,range_reg_dic_spot,sum_prop_dic_spot,coord_reg_dic_spot,star_params,spot_prop[spot_consid]['ctrst'])     

    return Focc_star_band, cond_occ



def sum_region_spot_prop(line_occ_HP_band,iband,args,par_list,Fsurf_grid_band,Fsurf_grid_emit_band,coord_grid,Ssub_Sstar,cb_band,range_par_list,range_reg,sum_prop_dic,coord_reg_dic,star_params,spot_contrast):
    r"""**Spotted region properties: calculations**

    Calculates the average and summed properties from a spotted stellar surface region during an exposure.

    Args:
        line_occ_HP_band (str) : The precision with which to process the exposure.
        iband (int) : Index of the band of interest.
        args (dict) : Parameters used to generate the intrinsic profiles.
        par_list (list) : List of parameters of interest, whose value in sum_prop_dic will be updated.
        Fsurf_grid_band (array) : Stellar flux grid over spotted region in the band of interest.
        Fsurf_grid_emit_band (array) : Stellar flux grid over the spotted region, using the LD coefficients of the spot in the band of interest.
        coord_grid (dict) : Dictionary of coordinates for the spotted region.
        Ssub_Sstar (float) : Surface ratio of a spotted region grid cell to a stellar grid cell.
        cb_band (list) : Polynomial coefficients used to compute thr RV component of the planet-occulted region due to convective blueshift.
        range_par_list (list) : List of parameters of interest, whose range of values, stored in range_reg_dic_spot, will be updated.
        range_reg (dict) : Dictionary containing the range of average values the parameters of interest (range_par_list) can take during this exposure.
        sum_prop_dic (dict) : Dictionary containing the value of all parameters of interest (par_list), summed over the spotted region in the exposure considered, and for the band of interest.
        coord_reg_dic (dict) : Dictionary containing the value of all parameters of interest (par_list), averaged over the spotted region in the exposure considered, and for the band of interest.
        star_params : nominal star properties.
        spot_contrast (float) : Contrast level of the spot considered.
    Returns:
        None
    
    """
    #Distance from projected orbital normal in the sky plane, in absolute value
    if 'xp_abs' in par_list : par_list.remove('xp_abs')
    if 'xp_abs' in range_par_list : range_par_list.remove('xp_abs')

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
            elif par_loc=='Rot_RV':coord_grid[par_loc] = calc_RVrot(coord_grid['x_st_sky'],coord_grid['y_st'],star_params['istar_rad'],star_params)[0]
                         
            #Disk-integrated-corrected convective blueshift polynomial (km/s)
            elif par_loc=='CB_RV':coord_grid[par_loc] = np_poly(cb_band)(coord_grid['mu'])          
    
            #Full RV (km/s)
            #    - accounting for an additional constant offset to model jitter or global shifts, and for visit-specific offset to model shifts specific to a given transition
            elif par_loc=='rv':
                coord_grid[par_loc] = deepcopy(coord_grid['Rot_RV']) + star_params['rv']
                if 'CB_RV' in par_list:coord_grid[par_loc]+=coord_grid['CB_RV']
                if 'rv_line' in par_list:coord_grid[par_loc]+=coord_grid['rv_line']
                
            #------------------------------------------------
    
            #Sum property over occulted region, weighted by stellar flux
            #    - we use flux rather than intensity, because local flux level depend on the planet grid resolution
            #    - total RVs from planet-occulted region is set last in par_list to calculate all rv contributions first:
            # + rotational contribution is always included
            # + disk-integrated-corrected convective blueshift polynomial (in km/s)   
            coord_grid[par_loc+'_sum'] = np.sum(coord_grid[par_loc]*Fsurf_grid_band)
              
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
        init_st_intr_prof(args,coord_grid,star_params)
        
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
        if not args['fit']:line_prof_grid=coadd_loc_line_prof(coord_grid['rv'],range(coord_grid['nsub_star']),Fsurf_grid_band,args['flux_intr_grid'],coord_grid['mu'],star_params,args)          
        else:
            if use_OS_grid:
                fit_Fsurf_grid_band = np.tile(Fsurf_grid_band, (args['ncen_bins'], 1)).T
                line_prof_grid=OS_coadd_loc_line_prof(coord_grid['rv'],fit_Fsurf_grid_band,args)
            elif use_C_OS_grid:
                line_prof_grid = use_C_OS_coadd_loc_line_prof(coord_grid['rv'],Fsurf_grid_band,args)
            else:line_prof_grid=coadd_loc_line_prof(coord_grid['rv'],range(coord_grid['nsub_star']),Fsurf_grid_band,args['flux_intr_grid'],coord_grid['mu'],star_params,args) 


        #Calculate profile emitted by the spot
        if not args['fit']:emit_line_prof_grid = coadd_loc_line_prof(coord_grid['rv'],range(coord_grid['nsub_star']),(1-spot_contrast)*Fsurf_grid_emit_band,args['flux_intr_grid'],coord_grid['mu'],star_params,args)          
        else:
            if use_OS_grid:
                fit_Fsurf_grid_emit_band = np.tile((1-spot_contrast)*Fsurf_grid_emit_band, (args['ncen_bins'], 1)).T
                emit_line_prof_grid = OS_coadd_loc_line_prof(coord_grid['rv'],fit_Fsurf_grid_emit_band,args)
            elif use_C_OS_grid:
                emit_line_prof_grid = use_C_OS_coadd_loc_line_prof(coord_grid['rv'],(1-spot_contrast)*Fsurf_grid_emit_band,args)
            else:emit_line_prof_grid = coadd_loc_line_prof(coord_grid['rv'],range(coord_grid['nsub_star']),(1-spot_contrast)*Fsurf_grid_emit_band,args['flux_intr_grid'],coord_grid['mu'],star_params,args)        

        #Coadd line profiles over spotted region
        sum_prop_dic['line_prof'] = np.sum((np.array(line_prof_grid)-np.array(emit_line_prof_grid)),axis=0)

    #Define rotational broadening of planet-occulted region
    elif line_occ_HP_band in ['low','medium']:
        drv_min = coord_reg_dic['rv'][iband]-np.min(coord_grid['rv'])
        drv_max = np.max(coord_grid['rv'])-coord_reg_dic['rv'][iband] 
        coord_reg_dic['rv_broad'][iband] = 0.5*(drv_min+drv_max)       

    return None


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