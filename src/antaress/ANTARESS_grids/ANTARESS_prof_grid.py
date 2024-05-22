#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
from copy import deepcopy
import astropy.convolution.convolve as astro_conv
import bindensity as bind
from pysme import sme as SME
from pysme.linelist.vald import ValdFile
from pysme.abund         import Abund
from pysme.synthesize import synthesize_spectrum
import lmfit
from ctypes import CDLL,c_double,c_int,c_void_p,cast,POINTER
import os as os_system
from ..ANTARESS_analysis.ANTARESS_model_prof import pol_cont,dispatch_func_prof,polycoeff_def,calc_polymodu,calc_linevar_coord_grid
from ..ANTARESS_grids.ANTARESS_star_grid import up_model_star,calc_RVrot,calc_CB_RV,get_LD_coeff
from ..ANTARESS_general.utils import closest,np_poly,np_interp,gen_specdopshift,closest_arr,MAIN_multithread,stop,def_edge_tab

def def_Cfunc_prof():
    r"""**C profile calculation**

    Defines the C function and its parameters used in the optimization

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 

    code_dir = os_system.path.dirname(__file__).split('ANTARESS_grids')[0]
    myfunctions = CDLL(code_dir+'/ANTARESS_analysis/C_grid/Gauss_star_grid.so')
    fun_to_use = myfunctions.C_coadd_loc_gauss_prof
    fun_to_free = myfunctions.free_gaussian_line_grid
    fun_to_use.argtypes = [np.ctypeslib.ndpointer(c_double),
                    np.ctypeslib.ndpointer(c_double),
                    np.ctypeslib.ndpointer(c_double),
                    np.ctypeslib.ndpointer(c_double),
                    np.ctypeslib.ndpointer(c_double),
                    c_int,
                    c_int]
    fun_to_use.restype = c_void_p
    fun_to_free.argtypes = [np.ctypeslib.ndpointer(c_double)]
    fun_to_free.restype = None
    
    return fun_to_use,fun_to_free



def custom_DI_prof(param,x,args=None):
    r"""**Disk-integrated profile: model function**

    Calculates custom disk-integrated stellar profile.
    The disk-integrated profile is built upon a discretized grid of the stellar surface to allow accounting for any type of intensity and velocity field.
    This routine is separated from init_custom_DI_prof() so that it can be used in fitting routines with varying parameters.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    
    #--------------------------------------------------------------------------------
    #Updating stellar grid and line profile grid
    #    - only if routine is called in fit mode, otherwise the default pipeline stellar grid and the scaling from init_custom_DI_prof() are used
    #--------------------------------------------------------------------------------
    if args['fit']:    

        #Updating stellar grid
        #    - if stellar grid is different from the default one 
        if args['var_star_grid'] and (args['unquiet_star'] is None):

            #Update variable stellar properties and stellar grid
            up_model_star(args,param)

            #Update broadband scaling of intrinsic profiles into local profiles
            #    - only necessary if stellar grid is updated
            args['Fsurf_grid_spec'] = theo_intr2loc(args['grid_dic'],args['system_prop'],args,args['ncen_bins'],args['grid_dic']['nsub_star'])     
    
        #--------------------------------------------------------------------------------        
        #Updating intrinsic line profiles
        #    - if the line properties or the stellar grid cells to which they are attributed to vary
        if args['var_line'] or args['var_star_grid']:
            init_st_intr_prof(args,args['grid_dic'],param)
            
    #--------------------------------------------------------------------------------
    #Radial velocities of the stellar surface (km/s)
    #    - an offset is allowed to account for the star/input frame velocity when the model is used on raw data 
    #--------------------------------------------------------------------------------
    rv_surf_star_grid = calc_RVrot(args['grid_dic']['x_st_sky'],args['grid_dic']['y_st'],args['star_params']['istar_rad'],param)[0] + param['rv']
    cb_band = calc_CB_RV(get_LD_coeff(args['system_prop']['achrom'],0),args['system_prop']['achrom']['LD'][0],param['c1_CB'], param['c2_CB'], param['c3_CB'],param)
    if np.max(np.abs(cb_band))!=0.:rv_surf_star_grid += np_poly(cb_band)(args['grid_dic']['mu']).flatten()

    #--------------------------------------------------------------------------------        
    #Coadding local line profiles over stellar disk
    #--------------------------------------------------------------------------------
    icell_list = np.arange(args['grid_dic']['nsub_star'])
    
    #Reducing grid to quiet cells
    if args['unquiet_star'] is not None:
        cond_quiet_star = ~args['unquiet_star']
        rv_surf_star_grid=rv_surf_star_grid[cond_quiet_star]
        args['grid_dic']['mu']=args['grid_dic']['mu'][cond_quiet_star]
        args['flux_intr_grid']=args['flux_intr_grid'][cond_quiet_star]
        icell_list=icell_list[cond_quiet_star]
        args['Fsurf_grid_spec']=args['Fsurf_grid_spec'][cond_quiet_star]
        nsub_star=len(icell_list)
    else:
        nsub_star = len(icell_list)
        cond_quiet_star = np.repeat(True,nsub_star)

    #Set up properties for fast line profile grid calculation
    use_OS_grid=False
    use_C_OS_grid=False
    if 'OS_grid' in args and args['OS_grid']:use_OS_grid=True
    if 'C_OS_grid' in args and args['C_OS_grid']:
        use_OS_grid=False
        use_C_OS_grid=True
    
    #Multithreading
    #    - disabled with theoretical profiles, there seems to be an incompatibility with sme
    if (args['nthreads']>1) and (args['mode']!='theo') and ('prof_grid' not in args['unthreaded_op']) :
        simplified_args={}
        for key in ['mode', 'mac_mode', 'input_cell_all', 'func_prof', 'cen_bins', 'nthreads']:simplified_args[key]=args[key]
        flux_DI_sum=MAIN_multithread(coadd_loc_line_prof,args['nthreads'],nsub_star,[rv_surf_star_grid,icell_list,args['Fsurf_grid_spec'],args['flux_intr_grid'],args['grid_dic']['mu']],(param,simplified_args,),output = True)              

    #Direct call
    else:
        if use_OS_grid:
            for pol_par in args['input_cell_all']:args['input_cell_all'][pol_par]=args['input_cell_all'][pol_par][cond_quiet_star]
            flux_DI_sum=coadd_loc_gauss_prof(rv_surf_star_grid,args['Fsurf_grid_spec'],args)
        elif use_C_OS_grid:
            for pol_par in args['input_cell_all']:args['input_cell_all'][pol_par]=args['input_cell_all'][pol_par][cond_quiet_star]
            Fsurf_grid_spec = args['Fsurf_grid_spec'][:, 0]
            flux_DI_sum = use_C_coadd_loc_gauss_prof(rv_surf_star_grid,Fsurf_grid_spec,args)
        else:flux_DI_sum=coadd_loc_line_prof(rv_surf_star_grid,icell_list,args['Fsurf_grid_spec'],args['flux_intr_grid'],args['grid_dic']['mu'],param,args)
    
    #Co-adding profiles
    DI_flux_norm = np.sum(flux_DI_sum,axis=0)

    #Scaling disk-integrated profile to requested continuum
    #    - DI_flux_norm is returned by the function normalized to unity  
    DI_flux_cont = param['cont']*DI_flux_norm

    #Polynomial continuum level
    #    - P(x) = cont*(1 + a1*rv + a2*rv^2 + a3*rv^3 ... )
    #      defined as x = rv or x=w-wref to provide less leverage to the fit 
    cen_bins_ref = args['cen_bins'] - args['cen_bins_polref']
    DI_flux_mod=DI_flux_cont*pol_cont(cen_bins_ref,args,param)    

    return DI_flux_mod,DI_flux_cont,DI_flux_norm




def init_custom_DI_prof(fixed_args,gen_dic,system_prop,system_spot_prop,theo_dic,star_params,param_in):   
    r"""**Disk-integrated profile: grid initialization**

    Initializes stellar and intrinsic profile grids.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    if isinstance(param_in,lmfit.parameter.Parameters):params={par:param_in[par].value for par in param_in}
    else:params=deepcopy(param_in) 

    #Store various properties potentially overwritten in the fitting procedure
    fixed_args['resamp_mode'] = gen_dic['resamp_mode'] 
    fixed_args['system_prop'] = deepcopy(system_prop) 
    fixed_args['star_params'] = deepcopy(star_params)  
    fixed_args['grid_dic'] = deepcopy(theo_dic)
    fixed_args['system_spot_prop'] = deepcopy(system_spot_prop)

    #------------------------------------------------------------------------
    #Identify variable stellar grid
    #    - condition is that one property controlling the stellar grid is variable OR has a value different from the one in theo_dic use to define the default stellar grid 
    #    - if condition is true, the sky-projected stellar grid and corresponding broadband emission are re-calculated at each step of the minimization for the step parameters
    #    - this option cannot be used with chromatic intensity variations
    #    - this option is not required if only veq is varying 
    #------------------------------------------------------------------------
    stargrid_prop = ['veq','alpha_rot','beta_rot','c1_CB','c2_CB','c3_CB','cos_istar','f_GD','beta_GD','Tpole','A_R','ksi_R','A_T','ksi_T','eta_R','eta_T']
    
    #Fit mode
    #    - grid will be updated in custom_DI_prof() if one of the properties vary
    if fixed_args['fit']:
        
        #Update condition
        fixed_args['var_star_grid']=False
        for par in params:
            if ((par in stargrid_prop) and (params[par] != star_params[par])) or ((par in ['LD_u1','LD_u2','LD_u3','LD_u4']) and (params[par] != system_prop['achrom'][par][0])):fixed_args['var_star_grid']=True    
    
    #Forward mode
    #    - only if properties differ from default ones
    else:
        up_grid = False
        for par in params:
            if ((par in stargrid_prop) and (params[par] != star_params[par])) or ((par in ['LD_u1','LD_u2','LD_u3','LD_u4']) and (params[par] != system_prop['achrom'][par][0])):up_grid = True
        if up_grid:up_model_star(fixed_args,params)

    #------------------------------------------------------------------------
    #Intrinsic line profile grid
    #    - in foward mode: profiles are updated here under condition, and are always used to tile the stellar grid
    #      in fit mode: profiles are updated here if they do not vary during the fit
    #------------------------------------------------------------------------
        
    #Line profile variations
    #    - by default profiles are fixed on initialized values 
    fixed_args['var_line'] = False

    #Theoretical profiles 
    if fixed_args['mode']=='theo':
        fixed_args['abund_sp']=[] 

        #No covariance associated with intrinsic profile
        fixed_args['cov_loc_star']=False
    
        #Spectral table    
        #    - theoretical profiles are defined on a common table in the star rest frame                                                      
        fixed_args['cen_bins_intr'] = np.array(theo_dic['sme_grid']['wave'])  
        fixed_args['edge_bins_intr'] = theo_dic['sme_grid']['edge_bins']                                                                    

        #Update profiles in forward mode
        #    - profiles are only updated if abundances differ from default ones, but are in any case attributed to the stellar grid
        if (not fixed_args['fit']):
            for par in params:
                if 'abund' in par:
                    sp_abund = par.split('_')[1]
                    if np.abs(params[par] - theo_dic['sme_grid']['abund'][sp_abund])>1e-6:fixed_args['abund_sp']+=[sp_abund] 
            init_st_intr_prof(fixed_args,fixed_args['grid_dic'],params)        

        #Fit mode
        #    - profiles are updated during the fit if abundances vary
        else:
            cond_update = False
            for par in params:
                if 'abund' in par:
                    sp_abund = par.split('_')[1]
                    fixed_args['abund_sp']+=[sp_abund] 
                    
                    #Abundance varies
                    if param_in[par].vary:fixed_args['var_line'] = True
                    
                    #Abundance is fixed and differs from initialization
                    elif np.abs(params[par] - theo_dic['sme_grid']['abund'][sp_abund])>1e-6:cond_update = True

            #Update profiles and attribute them to stellar grid if they remain fixed, and the profile and grid are the same as initialization
            if cond_update and (not fixed_args['var_star_grid']):init_st_intr_prof(fixed_args,fixed_args['grid_dic'],params)
          
    #----------------------------------
    #Analytical profiles
    elif (fixed_args['mode']=='ana'):

        #No covariance associated with intrinsic profile
        fixed_args['cov_loc_star']=False    

        #Model function
        #    - calculated directly on disk-integrated spectral table
        #    - if requested, we convolve the final model disk-integrated line with the instrumental LSF before comparison with the measured disk-integrated profile
        #    - the mean flux level of intrinsic profiles is assumed to be unity (see function)   
        if type(fixed_args['func_prof_name'])==str:fixed_args['func_prof'] = dispatch_func_prof(fixed_args['func_prof_name'])
        else:fixed_args['func_prof'] = {inst:dispatch_func_prof(fixed_args['func_prof_name'][inst]) for inst in fixed_args['func_prof_name']}

        #Define profiles in forward mode
        if (not fixed_args['fit']):init_st_intr_prof(fixed_args,fixed_args['grid_dic'],params)
     
        #Fit mode
        #    - there is no default grid of analytical profiles
        #      profiles are calculated directly in each cell, but their properties can be pre-calculated either here (if not fitted) or during the fit (if fitted)
        else:           
            for par in params:
                if (('ctrst_ord' in par) or ('FWHM_ord' in par)) and (param_in[par].vary):fixed_args['var_line'] = True
                
            #Define properties and attribute them to stellar grid if the properties and grid remain fixed
            if (not fixed_args['var_line']) and (not fixed_args['var_star_grid']):init_st_intr_prof(fixed_args,fixed_args['grid_dic'],params)

    #----------------------------------
    #Measured intrinsic profile
    elif fixed_args['mode']=='Intrbin':

        #Covariance associated with intrinsic profile
        fixed_args['cov_loc_star']=True
        
        #Spectral table
        #    - binned profiles are defined on a common table in the star rest frame
        fixed_args['edge_bins_intr'] = fixed_args['edge_bins_Intrbin']
        
        #Define profiles in forward mode
        if (not fixed_args['fit']):init_st_intr_prof(fixed_args,fixed_args['grid_dic'],params)
       
        #Fit mode
        #    - the profiles themselves are fixed, and will only be updated if the stellar grid is updated
        else:

            #Attribute profiles to stellar grid if the latter remains fixed
            if not fixed_args['var_star_grid']:init_st_intr_prof(fixed_args,fixed_args['grid_dic'],params)

    #Reference point for polynomial continuum 
    #    - if several visits are processed we assume their tables are roughly similar
    if (fixed_args['type']=='CCF') or fixed_args['spec2rv']:fixed_args['cen_bins_polref'] = 0.
    else:
        if type(fixed_args['ncen_bins'])==int:fixed_args['cen_bins_polref'] = fixed_args['cen_bins'][int(fixed_args['ncen_bins']/2)]  
        else:
            inst_ref = list(fixed_args['ncen_bins'].keys())[0]
            vis_ref = list(fixed_args['ncen_bins'][inst_ref].keys())[0] 
            cen_pix = int(fixed_args['ncen_bins'][inst_ref][vis_ref]/2) 
            fixed_args['cen_bins_polref'] = fixed_args['cen_bins'][inst_ref][vis_ref][0][cen_pix]  

    return fixed_args



def init_custom_DI_par(fixed_args,gen_dic,system_prop,star_params,params,RV_guess_tab):  
    r"""**Disk-integrated profile: parameter initialization**

    Initializes stellar parameters controlling either disk-integrated or local stellar grids. 
    Fit parameters are initialized to default stellar properties.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Stellar grid properties
    #    - all stellar properties are initialized to default stellar values
    #      those defined as variable properties through the settings will be overwritten in 'par_formatting'
    for key,vary,bd_min,bd_max in zip(['veq','veq_spots','alpha_rot','alpha_rot_spots','beta_rot','beta_rot_spots','c1_CB','c2_CB','c3_CB','c1_CB_spots','c2_CB_spots','c3_CB_spots','cos_istar','f_GD','beta_GD','Tpole','A_R','ksi_R','A_T','ksi_T','eta_R','eta_T'],
                                      [False,   False,      False,         False,        False,        False,        False, False,  False,   False,         False,          False,   False,    False,  False,   False, False, False, False, False,  False,   False],
                                      [0.,      0.,         None,          None,         None,         None,         None,  None,   None,    None,          None,           None,     -1.,       0.,     0.,      0.,     0. ,   0.,    0. ,   0.,     0. ,      0.],
                                      [1e4,     1e4,        None,          None,         None,         None,         None,  None,   None,    None,          None,           None,      1.,       1.,     1.,      1e5,    1.,   1e5,    1e5,  1e5,    100.,    100.]):
        if key in star_params:params.add_many((key, star_params[key],   vary,    bd_min,bd_max,None))

    #Properties specific to disk-integrated profiles
    if fixed_args['DI_grid']:
        for ideg in range(1,5):params.add_many(('LD_u'+str(ideg),  system_prop['achrom']['LD_u'+str(ideg)][0],              False,    None,None,None))   

    #Line model properties
    params.add_many(('cont',      fixed_args['flux_cont'],                          False,    None,             None,               None))
    params.add_many(('rv',        RV_guess_tab[0],                                  False,    RV_guess_tab[1],  RV_guess_tab[2],    None)) 
    for ideg in range(1,5):params.add_many(('c'+str(ideg)+'_pol',          0.,              False,    None,None,None)) 

    return params    





def init_st_intr_prof(args,grid_dic,param):
    r"""**Intrinsic stellar profiles: initialization**

    Initializes intrinsic profile grid and properties.
    Called upon initialization of the disk-integrated stellar profile grid, or in fit mode if the stellar grid or the intrinsic profile grid vary

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Theoretical intrinsic profiles
    if (args['mode']=='theo'):

        #Update line profile series
        sme_grid = deepcopy(args['grid_dic']['sme_grid'])
        if (len(args['abund_sp'])>0):
            for sp in args['abund_sp']:sme_grid['abund'][sp]=param['abund_'+sp]
            gen_theo_intr_prof(sme_grid)
            
        #Interpolating profiles defined over mu grid over full stellar mu grid
        args['flux_intr_grid'] = sme_grid['flux_intr_grid'](grid_dic['mu'])
     
    #Analytic intrinsic profile
    #    - coefficients describing surface properties with polynomial variation of chosen coordinate, required to calculate spectral line profile
    elif (args['mode']=='ana'): 
        
        #Set coordinate grid
        grid_dic['linevar_coord_grid'] = calc_linevar_coord_grid(args['coord_line'],grid_dic)          

        #Set line profile properties
        args['flux_intr_grid'] = np.zeros(grid_dic['nsub_star'])*np.nan   
        args['input_cell_all']={}
        args['coeff_line'] = {}
        inst_list = [args['inst']] if ('inst' in args) else list(args['linevar_par'].keys())
        for inst in inst_list:
            vis_list = [args['vis']] if ('vis' in args) else list(args['linevar_par'][inst].keys())
            for vis in vis_list:  
                for par_loc in args['linevar_par'][inst][vis]:     
                    args['coeff_line'][par_loc] = polycoeff_def(param,args['coeff_ord2name'][inst][vis][par_loc])
                    args['input_cell_all'][par_loc] = calc_polymodu(args['pol_mode'],args['coeff_line'][par_loc],grid_dic['linevar_coord_grid']) 
                                        
    #Measured intrinsic profiles
    #    - attributing to each stellar cell the measured profile with closest coordinate
    elif (args['mode']=='Intrbin'):
        ibin_to_cell = closest_arr(args['cen_dim_Intrbin'], grid_dic[args['coord_line']])
        args['flux_intr_grid'] = np.zeros([grid_dic['nsub_star'],len(args['flux_Intrbin'][0])],dtype=float)
        for icell in range(grid_dic['nsub_star']):args['flux_intr_grid'][icell]=args['flux_Intrbin'][ibin_to_cell[icell]] 
            
    return None




def gen_theo_atm(st_atm,star_params):
    r"""**Theoretical stellar atmosphere**

    Initializes SME grid to generate intrinsic stellar profiles 

    Args:
        TBD
    
    Returns:
        TBD
    
    """

    #Atmosphere structure
    sme_grid = SME.SME_Structure()
    
    #Stellar atmosphere model
    #    - we keep the defaults for sme.atmo.method = "grid" and sme.atmo.geom = "PP"
    sme_grid.atmo.source = st_atm['atm_model']+'.sav'
    
    #NLTE departure
    if ('nlte' in st_atm) and (len(st_atm['nlte'])>0):
        for sp_nlte in st_atm['nlte']:sme_grid.nlte.set_nlte(sp_nlte,st_atm['nlte'][sp_nlte]+'.grd')   
    
    #Set nominal model properties
    #    - vsini is set to 0 so that intrinsic profiles are kept aligned
    sme_grid['teff'] = star_params['Tpole']    
    sme_grid['logg'] = star_params['logg']                
    sme_grid['vsini'] = 0.
    sme_grid['vmic'] = 0.            
    sme_grid['vmac'] = 0.            

    #Wavelength and mu grid of the synthetic spectra
    #    - in A, in the star rest frame
    sme_grid['n_wav'] = int((st_atm['wav_max']-st_atm['wav_min'])/st_atm['dwav'])
    sme_grid_wave = np.linspace(st_atm['wav_min'],st_atm['wav_max'],sme_grid['n_wav']) 
    sme_grid['edge_bins'] = def_edge_tab(sme_grid_wave[None,:][None,:])[0,0]
    sme_grid['wave'] = sme_grid_wave
    
    #Add extreme mu values if not set to prevent extrapolation issues
    id_sort = np.argsort(st_atm['mu_grid'])
    st_atm['mu_grid'] = st_atm['mu_grid'][id_sort]
    if st_atm['mu_grid'][0]>0.:st_atm['mu_grid'] = np.append(0.,st_atm['mu_grid'])
    if st_atm['mu_grid'][-1]<1.:st_atm['mu_grid'] = np.append(st_atm['mu_grid'],1.)
    sme_grid['mu_grid'] = st_atm['mu_grid']
    sme_grid['n_mu'] = len(st_atm['mu_grid'])
    
    #Retrieve linelist and limit to range of spectral grid
    sme_grid['linelist'] = ValdFile(st_atm['linelist'])
    wlcent = sme_grid['linelist']['wlcent']
    cond_within = (wlcent>=st_atm['wav_min']-5.) & (wlcent<=st_atm['wav_max']+5.)
    if True not in cond_within:stop('No VALD transitions within requested range')
    sme_grid['linelist'] = sme_grid['linelist'][cond_within]

    #Abundances
    #    - set by default to solar, from Asplund+2009
    #    - specific abundances are defined as A(X) = log10( N(X)/N(H) ) + 12 
    #    - overall metallicity yields A(X) = Anominal(X) + [M/H] for X != H and He   
    sme_grid['abund'] = Abund.solar()
    if ('abund' in st_atm) and (len(st_atm['abund'])>0):
        for sp_abund in st_atm['abund']:sme_grid['abund'][sp_abund]=st_atm['abund'][sp_abund]
    sme_grid['monh'] = st_atm['MovH'] if 'MovH' in st_atm else 0    
    
    #Intrinsic profile grid
    gen_theo_intr_prof(sme_grid)    
    
    return sme_grid


def gen_theo_intr_prof(sme_grid):
    r"""**Theoretical intrinsic stellar profiles**

    Returns grid of theoretical intrinsic profiles  as a function of :math:`\mu`

    Args:
        TBD
    
    Returns:
        TBD
    
    """    
    #Initialize grid of synthetic spectra
    flux_intr_grid = np.zeros([sme_grid['n_mu'],sme_grid['n_wav']],dtype=float)
    
    #Processing reqested mu
    for imu,mu in enumerate(sme_grid['mu_grid']):
        sme_grid['mu'] = [mu]
    
        #Synthetize spectrum
        sme_spec = synthesize_spectrum(sme_grid)
        flux_intr_grid[imu] = sme_spec.synth[0]

    #Interpolator for profile grid
    sme_grid['flux_intr_grid'] = interp1d(sme_grid['mu_grid'],flux_intr_grid,axis=0)
    
    return None






def theo_intr2loc(grid_dic,system_prop,fixed_args,ncen_bins,nsub_star):
    r"""**Intrinsic to local intensity scaling**

    Returns the scaling values or profiles to scale stellar line profiles from intrinsic to local in each cell.
    
    Intrinsic spectra do not necessarily have the same flux as the disk-integrated profiles, but have been set to the same continuum or total flux (see `proc_intr_data()` and `loc_prof_DI_mast()`).
    Now that the intrinsic and disk-integrated profiles are equivalent in terms of flux we have :math:`F_\mathrm{intr}(w,t,vis) \sim F_\mathrm{DI}(w,vis)`.
    This is valid for broadband fluxes or the continuum of CCFs (see `rescale_profiles()`).
    
    The model disk-integrated profile is defined as 
    
    .. math::    
       F_\mathrm{DI,mod}(w,vis) &= (\sum_{\mathrm{all \, stellar \, cells}} F_\mathrm{intr}(w - w_\mathrm{shift},x,vis) \mathrm{LD}(cell,w) dS ) / A  \\
                                & \sim (\sum_{\mathrm{all \, stellar \, cells}} F_\mathrm{DI}(w - w_\mathrm{shift},vis) \mathrm{LD}(cell,w) dS ) / A
                               
    Where the shift accounts for the stellar surface velocity field. If we neglect these shifts (for spectra) or consider the continuum range (for CCF) then

    .. math::    
       F_\mathrm{DI,mod}(w,vis) \sim F_\mathrm{DI}(w,vis) (\sum_{\mathrm{all \, stellar \, cells}} \mathrm{LD}(cell,w) dS ) / A
       
    Where A is a spectrum   

    .. math::     
       A(w) = \sum_{\mathrm{all \, stellar \, cells}}(\mathrm{LD}(cell,w) dS )

    If the line profile is calculated directly within the fit function we store the scaling profiles.

    Args:
        TBD
    
    Returns:
        TBD
    
    """   
    
    #Grid of broadband intensity variations, at the resolution of current grid
    #    - condition is validated if data is spectral and intensity input is chromatic
    if ('spec' in fixed_args['type']) and ('chrom' in system_prop):

        #Grid of broadband flux variations, at the spectral resolution of the input chromatic properties
        Fsurf_grid_band = grid_dic['Fsurf_grid_star_chrom']
        
        #Grid of broadband flux variations, at the spectral resolution of the local profiles
        #    - interpolation over the profile table if the full range of the profile is larger than the scale of chromatic variations, otherwise closest definition point
        Fsurf_grid_spec = np.zeros([nsub_star,ncen_bins])
        if (fixed_args['edge_bins'][-1]-fixed_args['edge_bins'][0]>system_prop['chrom']['med_dw']):
            for icell in range(nsub_star):
                Fsurf_grid_spec[icell,:] = np_interp(fixed_args['cen_bins'],system_prop['chrom']['w'],Fsurf_grid_band[icell],left=Fsurf_grid_band[icell,0],right=Fsurf_grid_band[icell,-1])
        else:
            iband = closest(system_prop['chrom']['w'],np.median(fixed_args['cen_bins']))
            Fsurf_grid_spec = Fsurf_grid_band[:,iband][:,None]
    else:
        Fsurf_grid_spec = np.tile(grid_dic['Fsurf_grid_star_achrom'][:,0],[ncen_bins,1]).T       

    return Fsurf_grid_spec







def calc_loc_line_prof(icell,rv_surf_star,Fsurf_cell_spec,flux_loc_cell,mu_cell,args,param):
    r"""**Local line profile calculation**

    Calculates local profile from a given cell of the stellar disk.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 

    #Calculation of analytical intrinsic line profile
    #    - model is always calculated in RV space, and later converted back to wavelength space if relevant
    #    - the model is directly calculated over the RV table at its requested position, rather than being pre-calculated and shifted
    if args['mode']=='ana':
        input_cell = {'cont':1. , 'rv':rv_surf_star,'c1_pol' : 0.,'c2_pol' : 0.,'c3_pol' : 0.,'c4_pol' : 0.}
        for pol_par in args['input_cell_all']:
            input_cell[pol_par] = args['input_cell_all'][pol_par][icell]
        flux_intr=args['func_prof'](input_cell,args['cen_bins'] )[0]
        
        #Convolve intrinsic profile with macroturbulence kernel
        if args['mac_mode'] is not None:
    
            #Mactroturbulence kernel table     
            #    - centered spectral table with same pixel widths as the band spectrum the kernel is associated to
            #    - a range of 3.15 times the FWHM already contains 99.98% of a Gaussian LSF integral, we thus use 5 times its value 
            #    - G(rv) = exp(- (rv/ sqrt(2)*sig )^2  ) with  = FWHM/(2 sqrt(2ln(2)))   
            #      with eta = sqrt(2)*sig
            #           sig = FWHM/(2 sqrt(2ln(2)))  
            #      thus FWHM = eta*(2 sqrt(ln(2)))  
            #      we assume a conservative FWHM = 50km/s for macroturbulence
            dbins = args['cen_bins'][1]-args['cen_bins'][0]        
            hnkern=np.int(np.ceil(2.125*50./dbins)+1)
            cen_bins_kernel=dbins*np.arange(-hnkern,hnkern+1)
            
            #Calculate local macroturbulence kernel
            cos_mu_cell = mu_cell
            sin_mu_cell = np.sqrt(1. - mu_cell**2.)
            macro_kern_loc = args['mac_mode_func'](cen_bins_kernel,param,cos_mu_cell,sin_mu_cell)
            macro_kern_loc=macro_kern_loc/np.sum(macro_kern_loc) 
    
            #Convolution
            #    - bins must have the same size in a given table
            flux_intr=astro_conv(flux_intr,macro_kern_loc,boundary='extend')          
        
    #Shift stored intrinsic profile from the photosphere (source) to the star (receiver) rest frame     
    #    - see gen_specdopshift():
    # w_receiver = w_source * (1+ (rv[s/r]/c))
    # w_star = w_photo * (1+ (rv[photo/star]/c))
    #        = w_photo * (1+ (rv_surf/c))
    #    - models are pre-calculated and then shifted, since the line profile and the shift are independent
    elif (args['mode'] in ['theo','Intrbin']):
        if ('spec' in args['type']):edge_bins_surf = args['edge_bins_intr']*gen_specdopshift(rv_surf_star)        
        elif (args['type']=='CCF'):edge_bins_surf = args['edge_bins_intr'] + rv_surf_star
        flux_intr = bind.resampling(args['edge_bins'],edge_bins_surf,flux_loc_cell, kind=args['resamp_mode'])            
    
    #Continuum scaling into local line profiles
    #    - default continuum of intrinsic profiles is set to 1 so that it can be modulated chromatically here
    flux_loc=flux_intr*Fsurf_cell_spec
    
    return flux_loc




def coadd_loc_line_prof(rv_surf_star_grid,icell_list,Fsurf_grid_spec,flux_intr_grid,mu_grid,param,args):
    r"""**Local line profile co-addition**

    Cumulates local profiles from each cell of the stellar disk.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    flux_DI_sum=[]
    for isub,(icell,rv_surf_star,flux_intr_cell,mu_cell) in enumerate(zip(icell_list,rv_surf_star_grid,flux_intr_grid,mu_grid)):
        flux_DI_sum+=[calc_loc_line_prof(icell,rv_surf_star,Fsurf_grid_spec[isub],flux_intr_cell,mu_cell,args,param)]
    return flux_DI_sum




def coadd_loc_gauss_prof(rv_surf_star_grid, Fsurf_grid_spec, args):
    r"""**Local Gaussian line co-addition**

    Oversimplified way of cumulating the local profiles from each cell of the stellar disk. 
    This version assumes gaussian line profiles in each cell.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Define necessay grids    
    true_rv_surf_star_grid = np.tile(rv_surf_star_grid, (args['ncen_bins'], 1)).T
    model_table = np.ones((Fsurf_grid_spec.shape[0], args['ncen_bins']), dtype=float) * args['cen_bins']
    cont_grid = np.ones((Fsurf_grid_spec.shape[0], args['ncen_bins']))
    sqrt_log2 = np.sqrt(np.log(2.))
    ctrst_grid = np.tile(args['input_cell_all']['ctrst'], (args['ncen_bins'], 1)).T
    FWHM_grid = np.tile(args['input_cell_all']['FWHM'], (args['ncen_bins'], 1)).T
    
    #Make grid of profiles    
    gaussian_line_grid = cont_grid*(1.-ctrst_grid*np.exp(-(2.*sqrt_log2*(model_table-true_rv_surf_star_grid)/FWHM_grid)**2))

    gaussian_line_grid *= Fsurf_grid_spec

    return gaussian_line_grid



def use_C_coadd_loc_gauss_prof(rv_surf_star_grid, Fsurf_grid_spec, args):
    r"""**C++ local Gaussian line co-addition**

    C++ implementation of `coadd_loc_gauss_prof()`.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    Fsurf_grid_spec_shape0 = len(Fsurf_grid_spec)
    ncen_bins = args['ncen_bins']
    gauss_grid_ptr = args['fun_to_use'](rv_surf_star_grid, args['input_cell_all']['ctrst'], args['input_cell_all']['FWHM'], 
        args['cen_bins'], Fsurf_grid_spec*10, ncen_bins, Fsurf_grid_spec_shape0)
    gauss_grid = np.frombuffer(cast(gauss_grid_ptr, POINTER(c_double * (ncen_bins * Fsurf_grid_spec_shape0))).contents, dtype=np.float64)
    truegauss_grid = gauss_grid.reshape((Fsurf_grid_spec_shape0, ncen_bins))/10
    args['fun_to_free'](gauss_grid)
    return truegauss_grid






