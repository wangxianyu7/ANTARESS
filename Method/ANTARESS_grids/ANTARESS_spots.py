import numpy as np
from copy import deepcopy
from utils import np_poly,stop,check_data
from itertools import product as it_product
from ANTARESS_grids.ANTARESS_coord import frameconv_skystar_to_star,frameconv_star_to_skystar
from ANTARESS_analysis.ANTARESS_model_prof import poly_prop_calc
from ANTARESS_analysis.ANTARESS_inst_resp import convol_prof
# from ANTARESS_analysis.ANTARESS_ana_comm import par_formatting
from ANTARESS_grids.ANTARESS_coord import calc_zLOS_oblate
from ANTARESS_grids.ANTARESS_star_grid import calc_GD, calc_LD, calc_RVrot, calc_Isurf_grid
from ANTARESS_grids.ANTARESS_prof_grid import init_st_intr_prof, coadd_loc_line_prof
from ANTARESS_analysis.ANTARESS_model_prof import calc_linevar_coord_grid, calc_polymodu


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









def calc_spotted_tiles(spot_prop, x_sky_grid, y_sky_grid, z_sky_grid, grid_dic, star_param, use_grid_dic = False) :
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
     
    Returns:
        spot_within_grid (bool) : a finer estimate of the spot visibility that can be obtained with is_spot_visible. Essentially, it tells us if at least one tile on the grid is 'spotted'.
        cond_in_sp (1D array) : array of booleans telling us which cells in the original grid are 'spotted'.
    
    """                                      

    if use_grid_dic :
        cond_close_to_spot = (grid_dic['x_st_sky'] - spot_prop['x_sky_exp_center'])**2 + (grid_dic['y_st_sky'] - spot_prop['y_sky_exp_center'])**2 < spot_prop['ang_rad']**2
     
        x_st_grid, y_st_grid, z_st_grid = grid_dic['x_st'][cond_close_to_spot], grid_dic['y_st'][cond_close_to_spot], grid_dic['z_st'][cond_close_to_spot]
        
        
    else :  
        cond_close_to_spot = (x_sky_grid - spot_prop['x_sky_exp_center'])**2 + (y_sky_grid - spot_prop['y_sky_exp_center'])**2 < spot_prop['ang_rad']**2
    
        x_st_grid, y_st_grid, z_st_grid = frameconv_skystar_to_star(x_sky_grid[cond_close_to_spot],
                                                                                    y_sky_grid[cond_close_to_spot],
                                                                                    z_sky_grid[cond_close_to_spot],
                                                                                    np.arccos(star_param['cos_istar']))
        
        
    
    # Retrieve angular coordinates of spot
    cos_long, sin_long, cos_lat, sin_lat = spot_prop['cos_long_exp_center'], spot_prop['sin_long_exp_center'], spot_prop['cos_lat_exp_center' ], spot_prop['sin_lat_exp_center' ]
    
    # Calculate coordinates in spot rest frame
    x_sp =                       x_st_grid*cos_long - z_st_grid*sin_long
    y_sp = y_st_grid*cos_lat  - (x_st_grid*sin_long + z_st_grid*cos_long)   *   sin_lat
    z_sp = y_st_grid*sin_lat  + (x_st_grid*sin_long + z_st_grid*cos_long)   *   cos_lat
    
    # Deduce which cells are within the spot
    phi_sp = np.arctan2(np.sqrt(x_sp**2. + y_sp**2.),z_sp)
    cond_in_sp = cond_close_to_spot
    cond_in_sp[cond_close_to_spot] = (phi_sp <= spot_prop['ang_rad'])
        
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




def new_new_calc_spotted_region_prop(line_occ_HP_band, cond_occ, spot_prop, iband, system_spot_prop, star_params, Ssub_Sstar_sp, Ssub_Sstar_ref, Istar_norm_band, par_star, sum_prop_dic_spot,\
                                    coord_reg_dic_spot, range_reg_dic_spot, Focc_star_band, par_list, range_par_list, args, cb_band, pl_loc_x = {}, pl_loc_y = {}, oversamp_idx = None, RpRs = None, plocc = False) :
    
    r"""**Spot-occulted properties: define and update**

    Identify the spot-occulted region in each exposure provided and calculate its properties. 
    Update the provided dictionaries which contain the average and sum of the properties of interest over the spot-occulted region.

    Args:
        line_occ_HP_band (str) : the precision with which to process the exposure.
        cond_occ (bool) : whether there is an occultation by at least one spot in the exposure considered.
        spot_prop (dict) : formatted spot properties Dictionary (see retrieve_spots_prop_from_param).
        iband (int) : index of the band used to retrieve the corresponding planet and spot limb-darkening properties.
        system_spot_prop (dict) : spot limb-darkening properties.
        star_params (dict) : star properties.
        Ssub_Sstar_sp (float): surface of grid cells in the spot-occulted region grid.
        Ssub_Sstar_ref (float) : surface of grid cells in the stellar grid.
        Istar_norm_band (float) : total intensity of the star in the band considered.
        par_star : ???
        sum_prop_dic_spot (dict) : dictionary containing the value of all parameters of interest (par_list), summed over the spot-occulted region in the exposure considered, and for the band of interest.
        coord_reg_dic_spot (dict) : dictionary containing the value of all parameters of interest (par_list), averaged over the spot-occulted region in the exposure considered, and for the band of interest.
        range_reg_dic_spot (dict) : dictionary containing the range of average values the parameters of interest (range_par_list) can take during this exposure.
        Focc_star_band (float) : total flux occulted by the spot in the exposure and band considered. 
        par_list (list) : List of parameters of interest, whose value in sum_prop_dict_spot will be updated.
        range par_list (list) : list of parameters of interest, whose range of values, stored in range_reg_dic_spot, will be updated.
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

    #Making our refined spot-occultation grid and calculating the number of points it has
    spot_x_sky_grid = new_x_sky_grid[cond_in_sp]
    spot_y_sky_grid = new_y_sky_grid[cond_in_sp]
    spot_z_sky_grid = new_z_sky_grid[cond_in_sp]
    n_occ_sp = np.sum(cond_in_sp)

    if plocc:
        for planet in list(pl_loc_x.keys()):
            cond_in_planet_disk = (new_x_sky_grid[cond_in_sp] - pl_loc_x[planet][oversamp_idx])**2 + (new_y_sky_grid[cond_in_sp] - pl_loc_y[planet][oversamp_idx])**2 < RpRs[planet][iband]**2
            spot_x_sky_grid = spot_x_sky_grid[~cond_in_planet_disk]
            spot_y_sky_grid = spot_y_sky_grid[~cond_in_planet_disk]
            spot_z_sky_grid = spot_z_sky_grid[~cond_in_planet_disk]
            n_occ_sp -= np.sum(cond_in_planet_disk)

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
        emit_line_prof_grid = coadd_loc_line_prof(coord_grid['rv'],range(coord_grid['nsub_star']),(1-spot_prop['ctrst'])*Fsurf_grid_occ[:,0],args['flux_intr_grid'],coord_grid['mu'],par_star,args)          
        
        #Coadd line profiles over spot-occulted region
        sum_prop_dic_spot['line_prof'] = np.sum((np.array(line_prof_grid)-np.array(emit_line_prof_grid)),axis=0) 

    #Define rotational broadening of planet-occulted region
    elif line_occ_HP_band in ['low','medium']:
        drv_min = coord_reg_dic_spot['rv'][iband]-np.min(coord_grid['rv'])
        drv_max = np.max(coord_grid['rv'])-coord_reg_dic_spot['rv'][iband] 
        coord_reg_dic_spot['rv_broad'][iband] = 0.5*(drv_min+drv_max)       

    return Focc_star_band, cond_occ







########################################################################################################################
########################################################################################################################
#################################################### UNUSED FUNCTIONS  #################################################
########################################################################################################################
########################################################################################################################



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



"""

Function which calculates the properties of spot-occulted stellar cells 

   + For each spot, we check if it is visible (with spots_prop[spot]['is_visible']), and if so, we use the previous function to calculate which cells of the stellar grid it occults
   + We store one global list of all spotted cells of the star (cond_in_sp), and their base flux level, calculated as the product of spot flux (flux_emitted_all_tiles_sp)
   + We then deduce the absorbed flux, as well as all the other properties of spotted tiles (RV, mu, ctrst, ...), exactly like in get_planet_disk_prop.

"""

def calc_spotted_region_prop(spots_prop, grid_dic, star_params, LD_law, ld_coeff, gd_band, cb_band, param, coeff_ord2name, dim, func_prof_name, var_par_list, pol_mode) :
    
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
    

    return cond_in_sp, (   spot_within_grid and np.any(region_prop['flux_sp'] > 0)   ), region_prop
