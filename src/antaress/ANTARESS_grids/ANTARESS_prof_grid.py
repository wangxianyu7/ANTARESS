#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d
from scipy import special
from copy import deepcopy
import astropy.convolution.convolve as astro_conv
import bindensity as bind
import lmfit
from ctypes import CDLL,c_double,c_int,c_float
import os as os_system
from ..ANTARESS_analysis.ANTARESS_model_prof import pol_cont,dispatch_func_prof,polycoeff_def,calc_polymodu,calc_linevar_coord_grid
from ..ANTARESS_grids.ANTARESS_star_grid import up_model_star,calc_RVrot,calc_CB_RV,get_LD_coeff
from ..ANTARESS_general.utils import closest,np_poly,np_interp,gen_specdopshift,closest_arr,MAIN_multithread,stop,def_edge_tab,get_pw10,np_where1D

class CFunctionWrapper:
    r"""**C profile calculation**

    Defines the C function used in the optimization.
    The implementation of a class and the getstate and setstate functions is necessary to use the C function when pickling is used in emcee (when multiprocessing is used).

    """ 
    def __init__(self):
        self._initialize_functions()
        self.current_function_name = None

    def _initialize_functions(self):
        code_dir = os_system.path.dirname(__file__).split('ANTARESS_grids')[0]
        self.myfunctions = CDLL(code_dir + '/ANTARESS_analysis/C_grid/C_star_grid.so')

        #Load all functions we want to use
        #Gauss
        self.C_coadd_loc_gauss_prof = self.myfunctions.C_coadd_loc_gauss_prof

        self.C_coadd_loc_gauss_prof.argtypes = [
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            c_int,
            c_int,
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS")
        ]
        self.C_coadd_loc_gauss_prof.restype = None

        #Voigt
        self.C_coadd_loc_voigt_prof = self.myfunctions.C_coadd_loc_voigt_prof

        self.C_coadd_loc_voigt_prof.argtypes = [
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            c_int,
            c_int,
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS")
        ]
        self.C_coadd_loc_voigt_prof.restype = None

        #Dgauss
        self.C_coadd_loc_dgauss_prof = self.myfunctions.C_coadd_loc_dgauss_prof

        self.C_coadd_loc_dgauss_prof.argtypes = [
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            c_int,
            c_int,
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS")
        ]
        self.C_coadd_loc_dgauss_prof.restype = None

        #Cgauss
        self.C_coadd_loc_cgauss_prof = self.myfunctions.C_coadd_loc_cgauss_prof

        self.C_coadd_loc_cgauss_prof.argtypes = [
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            c_float,
            c_float,
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            c_int,
            c_int,
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS")
        ]
        self.C_coadd_loc_cgauss_prof.restype = None

        #Pgauss
        self.C_coadd_loc_pgauss_prof = self.myfunctions.C_coadd_loc_pgauss_prof

        self.C_coadd_loc_pgauss_prof.argtypes = [
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS"),
            c_int,
            c_int,
            np.ctypeslib.ndpointer(c_double, flags="C_CONTIGUOUS")
        ]
        self.C_coadd_loc_pgauss_prof.restype = None

    def coadd_loc_gauss_prof_with_C(self, rv_surf_star_grid, ctrst_grid, FWHM_grid, args_cen_bins, Fsurf_grid_spec, args_ncen_bins, Fsurf_grid_spec_shape_0, gauss_grid):
        self.C_coadd_loc_gauss_prof(rv_surf_star_grid, ctrst_grid, FWHM_grid, args_cen_bins, Fsurf_grid_spec, args_ncen_bins, Fsurf_grid_spec_shape_0, gauss_grid)
    
    def coadd_loc_voigt_prof_with_C(self, rv_surf_star_grid, ctrst_grid, FWHM_grid, a_damp_grid, args_cen_bins, Fsurf_grid_spec, args_ncen_bins, Fsurf_grid_spec_shape_0, voigt_grid):
        self.C_coadd_loc_voigt_prof(rv_surf_star_grid, ctrst_grid, FWHM_grid, a_damp_grid, args_cen_bins, Fsurf_grid_spec, args_ncen_bins, Fsurf_grid_spec_shape_0, voigt_grid)
    
    def coadd_loc_dgauss_prof_with_C(self, rv_surf_star_grid, ctrst_grid, FWHM_grid, rv_l2c_grid, FWHM_l2c_grid, amp_l2c_grid, args_cen_bins, Fsurf_grid_spec, args_ncen_bins, Fsurf_grid_spec_shape_0, dgauss_grid):
        self.C_coadd_loc_dgauss_prof(rv_surf_star_grid, ctrst_grid, FWHM_grid, rv_l2c_grid, FWHM_l2c_grid, amp_l2c_grid, args_cen_bins, Fsurf_grid_spec, args_ncen_bins, Fsurf_grid_spec_shape_0, dgauss_grid)
    
    def coadd_loc_cgauss_prof_with_C(self, rv_surf_star_grid, ctrst_grid, FWHM_grid, skew, kurt, args_cen_bins, Fsurf_grid_spec, args_ncen_bins, Fsurf_grid_spec_shape_0, cgauss_grid):
        self.C_coadd_loc_cgauss_prof(rv_surf_star_grid, ctrst_grid, FWHM_grid, skew, kurt, args_cen_bins, Fsurf_grid_spec, args_ncen_bins, Fsurf_grid_spec_shape_0, cgauss_grid)
    
    def coadd_loc_pgauss_prof_with_C(self, rv_surf_star_grid, ctrst_grid, FWHM_grid, c4_pol_grid, c6_pol_grid, dRV_joint_grid, args_cen_bins, Fsurf_grid_spec, args_ncen_bins, Fsurf_grid_spec_shape_0, pgauss_grid):
        self.C_coadd_loc_pgauss_prof(rv_surf_star_grid, ctrst_grid, FWHM_grid, c4_pol_grid, c6_pol_grid, dRV_joint_grid, args_cen_bins, Fsurf_grid_spec, args_ncen_bins, Fsurf_grid_spec_shape_0, pgauss_grid)

    def __getstate__(self):
        # When pickling, we remove the ctypes function pointer
        state = self.__dict__.copy()
        del state['myfunctions']
        del state['C_coadd_loc_gauss_prof']
        del state['C_coadd_loc_voigt_prof']
        del state['C_coadd_loc_cgauss_prof']
        del state['C_coadd_loc_pgauss_prof']
        del state['C_coadd_loc_dgauss_prof']
        return state

    def __setstate__(self, state):
        # When unpickling, we re-initialize the ctypes function pointer
        self.__dict__.update(state)
        self._initialize_functions()







def var_stellar_prop(fixed_args,theo_dic,system_prop,system_ar_prop,star_params,param_in):   
    r"""**Stellar properties: variables**

    Defines variable stellar properties.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    if isinstance(param_in,lmfit.parameter.Parameters):params={par:param_in[par].value for par in param_in}
    else:params=deepcopy(param_in) 

    #Store nominal properties potentially overwritten in the fitting procedure
    fixed_args['grid_dic'] = deepcopy(theo_dic) 
    fixed_args['system_prop'] = deepcopy(system_prop) 
    fixed_args['system_ar_prop'] = deepcopy(system_ar_prop)
    
    #List of stellar properties potentially modified as model parameter
    stargrid_prop_nom = np.array(['veq','alpha_rot','beta_rot','c1_CB','c2_CB','c3_CB','cos_istar','f_GD','beta_GD','Tpole','A_R','ksi_R','A_T','ksi_T','eta_R','eta_T'])
    if len(system_ar_prop)>0:
        stargrid_prop_spots_nom = np.array(['veq','alpha_rot','beta_rot'])
        stargrid_prop_faculae_nom = np.array(['veq','alpha_rot','beta_rot'])
    else:
        stargrid_prop_spots_nom=[]    
        stargrid_prop_faculae_nom=[]

    #Define stellar rotational property
    #    - at this stage 'params' contains 'veq' by default and only contains 'Peq' if it was requested as a model parameter
    #      in this case 'veq' is removed from the model parameters and its value is updated in the stellar property dictionary (here in forward mode, or at each time step of the fit)
    if 'Peq' in params:
        print('       Switching veq for Peq as model parameter')
        stargrid_prop_nom[np_where1D(stargrid_prop_nom=='veq')]='Peq'
        if (len(stargrid_prop_spots_nom)>0):stargrid_prop_spots_nom[np_where1D(stargrid_prop_spots_nom=='veq')]='Peq'
        if (len(stargrid_prop_faculae_nom)>0):stargrid_prop_faculae_nom[np_where1D(stargrid_prop_faculae_nom=='veq')]='Peq'

    #--------------------------------------------------------------------------------------------

    #Check for model parameters with different values than nominal stellar ones  
    #    - 'diff_star_grid' is set True if at least one property controlling the stellar grid has a different input model value than the nominal stellar one  
    #      in that case the stellar grid is updated here, so that it accounts for model parameters being different from nominal ones even if they are not modified during a fit
    #      if no model is fitted then the stellar grid only needs to be updated once here since the stellar properties are not updated as part of a fit later on
    #    - limb-darkening is not variable but can be set to a different value than the nominal one, and is thus checked for here  
    fixed_args['var_star_grid'] = False
    fixed_args['var_stargrid_prop'] = []  
    fixed_args['var_stargrid_prop_spots']=[]
    fixed_args['var_stargrid_prop_faculae']=[]
    fixed_args['var_stargrid_bulk']=False 
    fixed_args['var_stargrid_I']=False        
    for par in params:
        if ((par in stargrid_prop_nom) and (params[par] != star_params[par])):
            fixed_args['var_star_grid'] = True
            fixed_args['var_stargrid_prop']+=[par]
            if par=='f_GD':
                fixed_args['var_stargrid_I']=True
                fixed_args['var_stargrid_bulk']=True
            if par=='cos_istar':fixed_args['var_stargrid_bulk'] = True
        if ((par in ['LD_u1','LD_u2','LD_u3','LD_u4']) and (params[par] != system_prop['achrom'][par][0])):
            fixed_args['var_star_grid'] = True
            fixed_args['var_stargrid_prop']+=[par]
            fixed_args['var_stargrid_I']=True    
        if par in stargrid_prop_spots_nom:
            par_spot = par+'_spots'
            if (params[par_spot] != star_params[par_spot]):
                fixed_args['var_star_grid'] = True
                fixed_args['var_stargrid_prop_spots']+=[par]
            if (params[par_spot] != params[par]):print('WARNING: quiet and spot values for '+par+' are different.')
        if par in stargrid_prop_faculae_nom:
            par_facula = par+'_faculae'
            if (params[par_facula] != star_params[par_facula]):
                fixed_args['var_star_grid'] = True
                fixed_args['var_stargrid_prop_faculae']+=[par]
            if (params[par_facula] != params[par]):print('WARNING: quiet and facula values for '+par+' are different.')

    #Update stellar grid
    if fixed_args['var_star_grid']:up_model_star(fixed_args,params)

    #--------------------------------------------------------------------------------------------

    #Initializing update condition
    #    - 'var_star_grid' is set True if the model is fitted and at least one property controlling the stellar grid is variable 
    #      in that case the sky-projected stellar grid and corresponding broadband emission are re-calculated at each step of the minimization for the step parameters
    #      this option cannot be used with chromatic intensity variations
    fixed_args['var_star_grid']=False

    #Fit mode
    #    - if one of the properties vary the stellar grid will be updated in the call to custom_DI_prof() at every step of the fit
    if fixed_args['fit']:
        
        #Properties to update are stored in these lists
        fixed_args['var_stargrid_prop'] = []   
        fixed_args['var_stargrid_prop_spots']=[]  
        fixed_args['var_stargrid_prop_faculae']=[]  
        fixed_args['var_stargrid_I']=False   
        fixed_args['var_stargrid_bulk']=False    
        
        #Processing parameters
        for par in params:

            #Check stellar properties
            if (par in stargrid_prop_nom) and (param_in[par].vary):
                fixed_args['var_star_grid']=True 
                fixed_args['var_stargrid_prop']+=[par]
                if par=='f_GD':
                    fixed_args['var_stargrid_I']=True
                    fixed_args['var_stargrid_bulk']= True
                if par=='cos_istar':fixed_args['var_stargrid_bulk'] = True
                
            #Check spot properties
            # - We distinguish three cases for properties common to the spots and quiet star:
            #   1 - Both quiet and spot properties are fit. In this case, both properties will behave independently.
            #   2 - Both quiet and spot properties are fixed. In this case, we issue a warning to inform the user if these properties have different values.
            #       It is important to stress that, in this case, the user is responsible for controlling the fixed values used.
            #   3 - Fixed/fit quiet properties and fit/fixed spot properties. In this case, the user is responsible for controlling the value of the fixed quantity,
            #       while the fit quantity behaves by itself.
            if (par in stargrid_prop_spots_nom):
                par_spot = par+'_spots'
                if (param_in[par_spot].vary):
                    fixed_args['var_star_grid']=True 
                    fixed_args['var_stargrid_prop_spots']+=[par]
                elif (params[par_spot] != params[par]) and (not param_in[par].vary) and (not param_in[par_spot].vary):print('WARNING: quiet and spot values for '+par+' are different.')

            #Check faculae properties
            if (par in stargrid_prop_faculae_nom):
                par_facula = par+'_faculae'
                if (param_in[par_facula].vary):
                    fixed_args['var_star_grid']=True 
                    fixed_args['var_stargrid_prop_faculae']+=[par]
                elif (params[par_facula] != params[par]) and (not param_in[par].vary) and (not param_in[par_facula].vary):print('WARNING: quiet and facula values for '+par+' are different.')

    return fixed_args






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
        if args['var_star_grid']:

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
    #    - velocity properties are stored in grid_dic to allow disinguishing between quiet and active cells
    #--------------------------------------------------------------------------------
    rv_surf_star_grid = calc_RVrot(args['grid_dic']['x_st_sky'],args['grid_dic']['y_st'],args['system_param']['star']['istar_rad'],args['grid_dic']['veq'],args['grid_dic']['alpha_rot'],args['grid_dic']['beta_rot'])[0] + param['rv']
    cb_band = calc_CB_RV(get_LD_coeff(args['system_prop']['achrom'],0),args['system_prop']['achrom']['LD'][0],param['c1_CB'], param['c2_CB'], param['c3_CB'],param)
    if np.max(np.abs(cb_band))!=0.:rv_surf_star_grid += np_poly(cb_band)(args['grid_dic']['mu']).flatten()

    #--------------------------------------------------------------------------------        
    #Coadding local line profiles over stellar disk
    #--------------------------------------------------------------------------------
    icell_list = np.arange(args['grid_dic']['nsub_star'])
    nsub_star = len(icell_list)

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
        if use_OS_grid:flux_DI_sum=coadd_loc_OS_prof(rv_surf_star_grid,args['Fsurf_grid_spec'],args)
        elif use_C_OS_grid:
            Fsurf_grid_spec = args['Fsurf_grid_spec'][:, 0]
            flux_DI_sum = use_C_coadd_loc_OS_prof(rv_surf_star_grid,Fsurf_grid_spec,args)
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



def init_custom_DI_prof(fixed_args,gen_dic,param_in):   
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
        sme_grid = fixed_args['grid_dic']['sme_grid']

        #No covariance associated with intrinsic profile
        fixed_args['cov_loc_star']=False
    
        #Spectral table    
        #    - theoretical profiles are defined on a common table in the star rest frame                                                      
        fixed_args['cen_bins_intr'] = np.array(sme_grid['wave'])  
        fixed_args['edge_bins_intr'] = sme_grid['edge_bins']                                                                    

        #Update profiles in forward mode
        #    - profiles are only updated if abundances differ from default ones, but are in any case attributed to the stellar grid
        if (not fixed_args['fit']):
            for par in params:
                if 'abund' in par:
                    sp_abund = par.split('_')[1]
                    if np.abs(params[par] - sme_grid['abund'][sp_abund])>1e-6:fixed_args['abund_sp']+=[sp_abund] 
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
                    elif np.abs(params[par] - sme_grid['abund'][sp_abund])>1e-6:cond_update = True

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
        if type(fixed_args['model'])==str:fixed_args['func_prof'] = dispatch_func_prof(fixed_args['model'])
        else:fixed_args['func_prof'] = {inst:dispatch_func_prof(fixed_args['model'][inst]) for inst in fixed_args['model']}

        #Define profiles in forward mode
        if (not fixed_args['fit']):init_st_intr_prof(fixed_args,fixed_args['grid_dic'],params)
     
        #Fit mode
        #    - there is no default grid of analytical profiles
        #      profiles are calculated directly in each cell, but their properties can be pre-calculated either here (if not fitted) or during the fit (if fitted)
        else:           
            for par in params:
                if (('ctrst__ord' in par) or ('FWHM__ord' in par)) and (param_in[par].vary):fixed_args['var_line'] = True
                
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
    #    - see var_stellar_prop() for details about the processing of 'veq' and 'Peq'
    for key,vary,bd_min,bd_max in zip(['veq','alpha_rot','beta_rot','c1_CB','c2_CB','c3_CB','cos_istar','f_GD', 'beta_GD','Tpole','A_R','ksi_R','A_T','ksi_T','eta_R','eta_T'],
                                      [False, False,      False,     False,  False,  False,  False,     False,  False,   False,  False, False, False, False,  False,  False],
                                      [1.,    None,       None,      None,   None,   None,   -1.,       0.,     0.,      0.,     0. ,   0.,    0. ,   0.,     0. ,    0.],
                                      [1e4,   None,       None,      None,   None,   None,    1.,       1.,     1.,      1e5,    1.,    1e5,   1e5,   1e5,    100.,   100.]):
        if key in star_params:params.add_many((key, star_params[key],   vary,    bd_min,bd_max,None))

    #Active region properties
    if fixed_args['cond_studied_ar']:
        for key,vary,bd_min,bd_max in zip(['veq_spots','alpha_rot_spots','beta_rot_spots','c1_CB_spots','c2_CB_spots','c3_CB_spots','veq_faculae','alpha_rot_faculae','beta_rot_faculae','c1_CB_faculae','c2_CB_faculae','c3_CB_faculae'],
                                          [False,      False,            False,            False,         False,          False,       False,           False,            False,            False,         False,          False],
                                          [1.,         None,             None,             None,          None,           None,         1.,             None,             None,             None,           None,          None],
                                          [1e4,        None,             None,             None,          None,           None,         1e4,            None,             None,             None,           None,          None]):
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
               
    #Import pySME 
    #    - the package raises issues on some operating system, so it is just retrieved if needed
    from pysme import sme as SME    
    from pysme.linelist.vald import ValdFile    
    from pysme.abund         import Abund
    
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
    from pysme.synthesize import synthesize_spectrum
    
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

def coadd_loc_OS_prof(rv_surf_star_grid, Fsurf_grid_spec, args):
    prof_funcs = {
        'gauss_intr_prof': coadd_loc_gauss_prof,
        'voigt':           coadd_loc_voigt_prof,
        'gauss_herm_lin':  coadd_loc_cgauss_prof,
        'gauss_poly':      coadd_loc_pgauss_prof,
        'dgauss':          coadd_loc_dgauss_prof
    }
    return prof_funcs[args['func_prof'].__name__](rv_surf_star_grid, Fsurf_grid_spec, args)



def coadd_loc_gauss_prof(rv_surf_star_grid, Fsurf_grid_spec, args):
    r"""**Local Gaussian line co-addition**

    Oversimplified way of cumulating the local profiles from each cell of the stellar disk. 
    This version assumes gaussian line profiles in each cell.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Define necessary grids    
    true_rv_surf_star_grid = np.tile(rv_surf_star_grid, (args['ncen_bins'], 1)).T
    model_table = np.ones((Fsurf_grid_spec.shape[0], args['ncen_bins']), dtype=float) * args['cen_bins']
    sqrt_log2 = np.sqrt(np.log(2.))
    ctrst_grid = np.tile(args['input_cell_all']['ctrst'], (args['ncen_bins'], 1)).T
    FWHM_grid = np.tile(args['input_cell_all']['FWHM'], (args['ncen_bins'], 1)).T
    
    #Make grid of profiles    
    gaussian_line_grid = 1.-ctrst_grid*np.exp(-(2.*sqrt_log2*(model_table-true_rv_surf_star_grid)/FWHM_grid)**2)
    gaussian_line_grid *= Fsurf_grid_spec

    return gaussian_line_grid


def coadd_loc_voigt_prof(rv_surf_star_grid, Fsurf_grid_spec, args):
    r"""**Local Voigt line co-addition**

    Oversimplified way of cumulating the local profiles from each cell of the stellar disk. 
    This version assumes voigt line profiles in each cell.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Define necessary grids    
    true_rv_surf_star_grid = np.tile(rv_surf_star_grid, (args['ncen_bins'], 1)).T
    model_table = np.ones((Fsurf_grid_spec.shape[0], args['ncen_bins']), dtype=float) * args['cen_bins']
    sqrt_log2 = np.sqrt(np.log(2.))
    ctrst_grid = np.tile(args['input_cell_all']['ctrst'], (args['ncen_bins'], 1)).T
    FWHM_grid = np.tile(args['input_cell_all']['FWHM'], (args['ncen_bins'], 1)).T
    a_damp_grid = np.tile(args['input_cell_all']['a_damp'], (args['ncen_bins'], 1)).T
    
    #Make grid of profiles   
    z_tab_grid =  2.*sqrt_log2*(model_table - true_rv_surf_star_grid)/FWHM_grid +  1j*a_damp_grid
    voigt_peak_grid = special.wofz(1j*a_damp_grid).real
    voigt_mod_grid = 1. - (ctrst_grid/voigt_peak_grid)*special.wofz(z_tab_grid).real
    voigt_mod_grid *= Fsurf_grid_spec

    return voigt_mod_grid


def coadd_loc_cgauss_prof(rv_surf_star_grid, Fsurf_grid_spec, args):
    r"""**Local skewed Gaussian line co-addition**

    Oversimplified way of cumulating the local profiles from each cell of the stellar disk. 
    This version assumes skewed gaussian line profiles in each cell.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Define necessary grids    
    true_rv_surf_star_grid = np.tile(rv_surf_star_grid, (args['ncen_bins'], 1)).T
    model_table = np.ones((Fsurf_grid_spec.shape[0], args['ncen_bins']), dtype=float) * args['cen_bins']
    factor_grid = np.ones((Fsurf_grid_spec.shape[0], args['ncen_bins']), dtype=float)
    sqrt2_log2 = np.sqrt(2.*np.log(2.))
    skew1 = args['input_cell_all']['skewA'][0]
    kurt1 = args['input_cell_all']['kurtA'][0]
    ctrst_grid = np.tile(args['input_cell_all']['ctrst'], (args['ncen_bins'], 1)).T
    FWHM_grid = np.tile(args['input_cell_all']['FWHM'], (args['ncen_bins'], 1)).T
    
    #Make grid of profiles   
    x_tab_grid =  2.*sqrt2_log2*(model_table - true_rv_surf_star_grid)/FWHM_grid

    #Skewness and kurtosis
    c = np.array([np.sqrt(6.)/4., -np.sqrt(3.), -np.sqrt(6.), 2./np.sqrt(3.), np.sqrt(6.)/3.])        
    if skew1 != 0:factor_grid+=skew1*(c[1]*x_tab_grid+c[3]*x_tab_grid**3.)
    if kurt1 != 0:factor_grid+=kurt1*(c[0]+c[2]*x_tab_grid**2.+c[4]*x_tab_grid**4.)   
 
    #Skewd gaussian profile
    sk_gauss_grid = 1.- factor_grid*ctrst_grid*np.exp(-0.5*x_tab_grid**2.)
    sk_gauss_grid *= Fsurf_grid_spec

    return sk_gauss_grid

def coadd_loc_dgauss_prof(rv_surf_star_grid, Fsurf_grid_spec, args):
    r"""**Local double Gaussian line co-addition**

    Oversimplified way of cumulating the local profiles from each cell of the stellar disk. 
    This version assumes double gaussian line profiles in each cell.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Define necessary grids 
    true_rv_surf_star_grid = np.tile(rv_surf_star_grid, (args['ncen_bins'], 1)).T
    model_table = np.ones((Fsurf_grid_spec.shape[0], args['ncen_bins']), dtype=float) * args['cen_bins']
    sqrt_log2 = np.sqrt(np.log(2.))
    ctrst_grid = np.tile(args['input_cell_all']['ctrst'], (args['ncen_bins'], 1)).T
    rv_l2c_grid = np.tile(args['input_cell_all']['rv_l2c'], (args['ncen_bins'], 1)).T
    FWHM_l2c_grid = np.tile(args['input_cell_all']['FWHM_l2c'], (args['ncen_bins'], 1)).T
    amp_l2c_grid = np.tile(args['input_cell_all']['amp_l2c'], (args['ncen_bins'], 1)).T
    FWHM_grid = np.tile(args['input_cell_all']['FWHM'], (args['ncen_bins'], 1)).T
    
    #Reduced contrast grid
    red_ctrst_grid = ctrst_grid/(1. - amp_l2c_grid)

    #Inverted gaussian core
    y_gausscore_grid=Fsurf_grid_spec*(1. - red_ctrst_grid*np.exp(-np.power(2.*sqrt_log2*(model_table-true_rv_surf_star_grid)/FWHM_grid,2))) 
    
    #Gaussian lobes
    cen_RV_lobes_grid=true_rv_surf_star_grid+rv_l2c_grid 
    FWHM_lobes_grid = FWHM_grid*FWHM_l2c_grid   
    y_gausslobes_grid=Fsurf_grid_spec*(1. + red_ctrst_grid*amp_l2c_grid*np.exp(-np.power(2.*sqrt_log2*(model_table-cen_RV_lobes_grid)/FWHM_lobes_grid,2)))     

    return y_gausscore_grid + y_gausslobes_grid - Fsurf_grid_spec


def coadd_loc_pgauss_prof(rv_surf_star_grid, Fsurf_grid_spec, args):
    r"""**Local sidelobed Gaussian line co-addition**

    Oversimplified way of cumulating the local profiles from each cell of the stellar disk. 
    This version assumes sidelobed gaussian line profiles in each cell.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Define necessary grids   
    true_rv_surf_star_grid = np.tile(rv_surf_star_grid, (args['ncen_bins'], 1)).T
    model_table = np.ones((Fsurf_grid_spec.shape[0], args['ncen_bins']), dtype=float) * args['cen_bins']
    y_polylobe_grid = np.ones((Fsurf_grid_spec.shape[0], args['ncen_bins']), dtype=float)
    sqrt_log2 = np.sqrt(np.log(2.))
    ctrst_grid = np.tile(args['input_cell_all']['ctrst'], (args['ncen_bins'], 1)).T
    c4_pol_grid = np.tile(args['input_cell_all']['c4_pol'], (args['ncen_bins'], 1)).T
    c6_pol_grid = np.tile(args['input_cell_all']['c6_pol'], (args['ncen_bins'], 1)).T
    dRV_joint_grid = np.tile(args['input_cell_all']['dRV_joint'], (args['ncen_bins'], 1)).T
    FWHM_grid = np.tile(args['input_cell_all']['FWHM'], (args['ncen_bins'], 1)).T
    
    #Gaussian with baseline set to continuum value
    y_gausscore_grid=1.-ctrst_grid*np.exp(-np.power( 2.*sqrt_log2*(model_table-true_rv_surf_star_grid)/FWHM_grid  ,2.  )) 
    ymodel_grid = y_gausscore_grid*Fsurf_grid_spec
    
    #Polynomial
    RV_joint_high = true_rv_surf_star_grid + dRV_joint_grid       
    RV_joint_low  = true_rv_surf_star_grid - dRV_joint_grid
    cond_lobes = (model_table >= RV_joint_low) & (model_table <= RV_joint_high) 
    y_polylobe_grid[cond_lobes] *= c4_pol_grid[cond_lobes]*dRV_joint_grid[cond_lobes]**4. + 2.*c6_pol_grid[cond_lobes]*dRV_joint_grid[cond_lobes]**6. - dRV_joint_grid[cond_lobes]**2.*np.power(model_table[cond_lobes]-true_rv_surf_star_grid[cond_lobes],2.)*(2.*c4_pol_grid[cond_lobes]+ 3.*c6_pol_grid[cond_lobes]*dRV_joint_grid[cond_lobes]**2.) + c4_pol_grid[cond_lobes]*np.power(model_table[cond_lobes]-true_rv_surf_star_grid[cond_lobes],4.) + c6_pol_grid[cond_lobes]*np.power(model_table[cond_lobes]-true_rv_surf_star_grid[cond_lobes],6.)
    ymodel_grid[cond_lobes]*=y_polylobe_grid[cond_lobes]    

    return ymodel_grid


def use_C_coadd_loc_OS_prof(rv_surf_star_grid, Fsurf_grid_spec, args):
    prof_funcs = {
        'gauss_intr_prof': use_C_coadd_loc_gauss_prof,
        'voigt':           use_C_coadd_loc_voigt_prof,
        'gauss_herm_lin':  use_C_coadd_loc_cgauss_prof,
        'gauss_poly':      use_C_coadd_loc_pgauss_prof,
        'dgauss':          use_C_coadd_loc_dgauss_prof
    }
    return prof_funcs[args['func_prof'].__name__](rv_surf_star_grid, Fsurf_grid_spec, args)



def use_C_coadd_loc_gauss_prof(rv_surf_star_grid, Fsurf_grid_spec, args):
    r"""**C++ local Gaussian line co-addition**

    C++ implementation of `coadd_loc_gauss_prof()`.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    if np.mean(Fsurf_grid_spec)==0.:sc_10 = 1
    else: sc_10 = get_pw10(np.abs(np.mean(Fsurf_grid_spec)))
    Fsurf_grid_spec_shape0 = len(Fsurf_grid_spec)
    ncen_bins = args['ncen_bins']
    gauss_grid = np.zeros((Fsurf_grid_spec_shape0, ncen_bins), dtype=np.float64).flatten()
    c_function_wrapper = args['c_function_wrapper']
    c_function_wrapper.coadd_loc_gauss_prof_with_C(rv_surf_star_grid,args['input_cell_all']['ctrst'],args['input_cell_all']['FWHM'],args['cen_bins'],Fsurf_grid_spec / sc_10,ncen_bins,Fsurf_grid_spec_shape0,gauss_grid)
    truegauss_grid = gauss_grid.reshape((Fsurf_grid_spec_shape0, ncen_bins)) * sc_10
    return truegauss_grid

def use_C_coadd_loc_voigt_prof(rv_surf_star_grid, Fsurf_grid_spec, args):
    r"""**C++ local Voigt line co-addition**

    C++ implementation of `coadd_loc_voigt_prof()`.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    if np.mean(Fsurf_grid_spec)==0.:sc_10 = 1
    else: sc_10 = get_pw10(np.abs(np.mean(Fsurf_grid_spec)))
    Fsurf_grid_spec_shape0 = len(Fsurf_grid_spec)
    ncen_bins = args['ncen_bins']
    voigt_grid = np.zeros((Fsurf_grid_spec_shape0, ncen_bins), dtype=np.float64).flatten()
    c_function_wrapper = args['c_function_wrapper']
    c_function_wrapper.coadd_loc_voigt_prof_with_C(rv_surf_star_grid,args['input_cell_all']['ctrst'],args['input_cell_all']['FWHM'],args['input_cell_all']['a_damp'],args['cen_bins'],Fsurf_grid_spec / sc_10,ncen_bins,Fsurf_grid_spec_shape0,voigt_grid)
    truevoigt_grid = voigt_grid.reshape((Fsurf_grid_spec_shape0, ncen_bins)) * sc_10

    return truevoigt_grid

def use_C_coadd_loc_cgauss_prof(rv_surf_star_grid, Fsurf_grid_spec, args):
    r"""**C++ local skewed Gaussian line co-addition**

    C++ implementation of `coadd_loc_cgauss_prof()`.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    if np.mean(Fsurf_grid_spec)==0.:sc_10 = 1
    else: sc_10 = get_pw10(np.abs(np.mean(Fsurf_grid_spec)))
    Fsurf_grid_spec_shape0 = len(Fsurf_grid_spec)
    ncen_bins = args['ncen_bins']
    cgauss_grid = np.zeros((Fsurf_grid_spec_shape0, ncen_bins), dtype=np.float64).flatten()
    factor_grid = np.ones((Fsurf_grid_spec_shape0, ncen_bins), dtype=np.float64).flatten()
    c_function_wrapper = args['c_function_wrapper']
    c_function_wrapper.coadd_loc_cgauss_prof_with_C(rv_surf_star_grid,args['input_cell_all']['ctrst'],args['input_cell_all']['FWHM'],args['input_cell_all']['skewA'][0],args['input_cell_all']['kurtA'][0],args['cen_bins'],Fsurf_grid_spec / sc_10,ncen_bins,Fsurf_grid_spec_shape0,cgauss_grid)
    truecgauss_grid = cgauss_grid.reshape((Fsurf_grid_spec_shape0, ncen_bins)) * sc_10

    return truecgauss_grid

def use_C_coadd_loc_dgauss_prof(rv_surf_star_grid, Fsurf_grid_spec, args):
    r"""**C++ local double Gaussian line co-addition**

    C++ implementation of `coadd_loc_dgauss_prof()`.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    if np.mean(Fsurf_grid_spec)==0.:sc_10 = 1
    else: sc_10 = get_pw10(np.abs(np.mean(Fsurf_grid_spec)))
    Fsurf_grid_spec_shape0 = len(Fsurf_grid_spec)
    ncen_bins = args['ncen_bins']
    dgauss_grid = np.zeros((Fsurf_grid_spec_shape0, ncen_bins), dtype=np.float64).flatten()
    c_function_wrapper = args['c_function_wrapper']
    c_function_wrapper.coadd_loc_dgauss_prof_with_C(rv_surf_star_grid,args['input_cell_all']['ctrst'],args['input_cell_all']['FWHM'],args['input_cell_all']['rv_l2c'],args['input_cell_all']['FWHM_l2c'],args['input_cell_all']['amp_l2c'],args['cen_bins'],Fsurf_grid_spec / sc_10,ncen_bins,Fsurf_grid_spec_shape0,dgauss_grid)
    truedgauss_grid = dgauss_grid.reshape((Fsurf_grid_spec_shape0, ncen_bins)) * sc_10

    return truedgauss_grid

def use_C_coadd_loc_pgauss_prof(rv_surf_star_grid, Fsurf_grid_spec, args):
    r"""**C++ local sidelobed Gaussian line co-addition**

    C++ implementation of `coadd_loc_pgauss_prof()`.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    if np.mean(Fsurf_grid_spec)==0.:sc_10 = 1
    else: sc_10 = get_pw10(np.abs(np.mean(Fsurf_grid_spec)))
    Fsurf_grid_spec_shape0 = len(Fsurf_grid_spec)
    ncen_bins = args['ncen_bins']
    pgauss_grid = np.zeros((Fsurf_grid_spec_shape0, ncen_bins), dtype=np.float64).flatten()
    c_function_wrapper = args['c_function_wrapper']
    c_function_wrapper.coadd_loc_pgauss_prof_with_C(rv_surf_star_grid,args['input_cell_all']['ctrst'],args['input_cell_all']['FWHM'],args['input_cell_all']['c4_pol'],args['input_cell_all']['c6_pol'],args['input_cell_all']['dRV_joint'],args['cen_bins'],Fsurf_grid_spec / sc_10,ncen_bins,Fsurf_grid_spec_shape0,pgauss_grid)
    truepgauss_grid = pgauss_grid.reshape((Fsurf_grid_spec_shape0, ncen_bins)) * sc_10

    return truepgauss_grid
