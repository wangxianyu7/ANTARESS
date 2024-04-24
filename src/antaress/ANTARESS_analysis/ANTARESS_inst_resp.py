#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import astropy.convolution.convolve as astro_conv
from copy import deepcopy
import bindensity as bind
from ..ANTARESS_general.utils import npint,stop,gen_specdopshift
from ..ANTARESS_general.constant_data import c_light


   
def return_pix_size(): 
    r"""**Spectrograph sampling**

    Returns width of detector pixel in rv space (km/s)

    Args:
        TBD
    
    Returns:
        TBD
    
    """             
    return {
            
        #Sophie HE mod: pix_size = 0.0275 A ~ 1.4 km/s at 5890 A 
        'SOPHIE_HE':1.4,  

        #CORALIE:
        #    ordre 10:  deltaV = 1.7240 km/s
        #    ordre 35:  deltaV = 1.7315 km/s
        #    ordre 60:  deltaV = 1.7326 km/s
        #    resolving power = 55000 -> deltav_instru = 5.45 km/s          
        'CORALIE':1.73,

        #HARPS-N or HARPS: pix_size = 0.016 A ~ 0.8 km/s at 5890 A         
        #    - resolving power = 120000 -> deltav_instru = 2.6km/s         
        'HARPN':0.82,
        'HARPS':0.82,
        
        #STIS E230M
        #    - size varies in wavelength but is roughly constant in velocity:
        # 0.0496 A at 3021A (=4.920 km/s)
        # 0.0375 A at 2274A (=4.944 km/s)   
        #      we take pix_size ~ 4.93 km/s    
        #    - with resolving power = 30000 -> deltav_instru = 9.9 km/s (2 bins)
        'STIS_E230M':4.93,   
        
        #STIS G750L
        #    - size varies in radial velocity and remains roughly constant in wavelength:
        # 195 km/s at 5300 A 
        # 275 km/s at 7500 A
        # ie about 4.87 A 
        #      we take an average pix_size ~ 235 km/s       
        'STIS_G750L':235.,  
        
        #ESPRESSO in HR mode
        #    - pixel size = 0.5 km/s
        # 0.01 A at 6000A
        #    - resolving power = 140000 -> deltav_instru = 2.1km/s           
        'ESPRESSO':0.5,

        #ESPRESSO in MR mode
        'ESPRESSO_MR':1.,
        
        #CARMENES   
        #    optical resolving power = 93400 -> deltav_instru = 3.2 km/s   
        #    - 2.8 pixel / FWHM, so that pixel size = 1.1317 km/s             
        'CARMENES_VIS':1.1317,
        #    near-infrared resolving power = 80400 -> deltav_instru = 3.72876 km/s   
        #    - 2.3 pixel / FWHM, so that pixel size = 1.62 km/s   
        'CARMENES_NIR':1.1317,
        
        'NIRPS_HA':0.93,
        'NIRPS_HE':0.93,  
        
        'EXPRES':0.5,
        
        'IRD':2.08442,
        
        'GIANO':3.1280284
        
    }     


def resamp_st_prof_tab(inst,vis,isub,fixed_args,gen_dic,nexp,rv_osamp_line_mod):
    r"""**Resampled spectral profile table**

    Defines resampled spectral grid for line profile calculations.
    Theoretical profiles are directly calculated at the requested resolution, measured profiles are extracted at their native resolution.

    Args:
        inst (str) : Instrument considered.
        vis (str) : Visit considered.
        isub (int) : Index of the exposure considered.
        fixed_args (dict) : Parameters of the profiles considered.
        gen_dic (dict) : General dictionary.
        nexp (int) : Number of exposures in the visit considered.
        rv_osamp_line_mode (float) : RV-space oversampling factor.
    
    Returns:
        TBD
    
    """
    if inst is None:edge_bins = fixed_args['edge_bins']
    else:edge_bins = fixed_args['edge_bins'][inst][vis][isub]

    #Resampled model table
    #    - defined in RV space if relevant for spectral data
    rv_resamp = deepcopy(rv_osamp_line_mod)
    if fixed_args['spec2rv']:
        edge_bins_RV = c_light*((edge_bins/fixed_args['line_trans']) - 1.) 
        if rv_resamp is None:rv_resamp = np.mean(edge_bins_RV[1::]-edge_bins_RV[0:-1]    )
        min_x = edge_bins_RV[0]
        max_x = edge_bins_RV[-1]    
    else:
        min_x = edge_bins[0]
        max_x = edge_bins[-1]    
    delta_x = (max_x-min_x)
    
    #Extend definition range to allow for convolution
    min_x-=0.05*delta_x
    max_x+=0.05*delta_x
    ncen_bins_HR = int(np.ceil(round((max_x-min_x)/rv_resamp)))
    dx_HR=(max_x-min_x)/ncen_bins_HR

    #Define and attribute table for current exposure
    dic_exp = {}
    dic_exp['edge_bins_HR']=min_x + dx_HR*np.arange(ncen_bins_HR+1)
    dic_exp['cen_bins_HR']=0.5*(dic_exp['edge_bins_HR'][1::]+dic_exp['edge_bins_HR'][0:-1])  
    dic_exp['dcen_bins_HR']= dic_exp['edge_bins_HR'][1::]-dic_exp['edge_bins_HR'][0:-1]   
    if inst is None: 
        for key in dic_exp:fixed_args[key] = dic_exp[key]
        fixed_args['ncen_bins_HR']=ncen_bins_HR
        fixed_args['dim_exp_HR']=deepcopy(fixed_args['dim_exp'])
        fixed_args['dim_exp_HR'][1] = ncen_bins_HR
    else:
        for key in ['cen_bins_HR','edge_bins_HR','dcen_bins_HR','dim_exp_HR']:
            if key not in fixed_args:fixed_args[key]={inst:{vis:np.zeros(nexp,dtype=object)}}
        for key in dic_exp:fixed_args[key][inst][vis][isub]  = dic_exp[key]   
        if 'ncen_bins_HR' not in fixed_args:
            fixed_args['ncen_bins_HR']={inst:{vis:ncen_bins_HR}}
        if 'dim_exp_HR' not in fixed_args:
            fixed_args['dim_exp_HR']={inst:{vis:fixed_args['dim_exp'][inst][vis]}} 
            fixed_args['dim_exp_HR'][inst][vis][1] = ncen_bins_HR        

    return None


def def_st_prof_tab(inst,vis,isub,args):
    r"""**Spectral profile table attribution**

    Attributes original or resampled spectral grid for line profile calculations.

    Args:
        TBD
    
    Returns:
        TBD
    
    """  
    args_exp = deepcopy(args)
    if args['resamp']:suff='_HR'
    else:suff=''
    if (inst is None):
        for key in ['edge_bins','cen_bins','dcen_bins','ncen_bins','dim_exp']:args_exp[key] = args[key+suff]
    else:
        for key in ['edge_bins','cen_bins','dcen_bins']:args_exp[key] = args[key+suff][inst][vis][isub]
        for key in ['ncen_bins','dim_exp']:args_exp[key] = args[key+suff][inst][vis]
        if (args['mode']=='ana'):args_exp['func_prof'] = args['func_prof'][inst]
        
    return args_exp



def return_resolv(inst): 
    r"""**Spectral resolving power**

    Returns resolving power of a given spectrograph.

    Args:
        inst (str) : Instrument / spectrograph considered.
    
    Returns:
        inst_res (float) : Resolving power of the spectrograph.
    
    """
    inst_res = {        
        'SOPHIE_HR':75000.,  
        'SOPHIE_HE':40000.,  
        'CORALIE':55000.,
        'HARPN':120000.,
        'HARPS':120000.,
        'STIS_E230M':30000.,     
        'STIS_G750L':1280.,         
        'ESPRESSO':140000.,
        'ESPRESSO_MR':70000.,
        'CARMENES_NIR':80400.,
        'CARMENES_VIS':94600.,
        'NIRPS_HE':75000.,
        'NIRPS_HA':88000.,
        'EXPRES':137500.,
        'NIRSPEC':25000.,
        'IRD':70000.,
        'GIANO':50000.,
    }[inst]  
    return inst_res

def calc_FWHM_inst(inst,w_c):
    r"""**Spectral resolution**

    Returns FWHM of a Gaussian approximating the LSF for a given resolving power, in rv or wavelength space
    
    .. math:: 
       \Delta v &= c / R   \\
       \Delta \lambda &= \lambda_\mathrm{ref}/R = \lambda_\mathrm{ref} \Delta v/c 
     
    Args:
        TBD
    
    Returns:
        TBD
    
    """
    FWHM_inst = w_c/return_resolv(inst)
    return FWHM_inst
  
    
def get_FWHM_inst(inst,fixed_args,cen_bins):
    r"""**Effective spectral resolution**

    Returns FWHM relevant to convolve the processed data 
    
     - in rv space for analytical profiles
     - in wavelength space for theoretical profiles
     - disabled if measured profiles as used as proxy for intrinsic profiles

    Args:
        TBD
    
    Returns:
        TBD
    
    """     
    #Reference point
    if (fixed_args['mode']=='ana') or fixed_args['spec2rv']:fixed_args['ref_conv'] = c_light
    elif fixed_args['mode']=='theo':fixed_args['ref_conv'] = cen_bins[int(len(cen_bins)/2)]    
    
    #Instrumental response 
    if (fixed_args['mode']=='Intrbin'):FWHM_inst = None
    else:FWHM_inst = calc_FWHM_inst(inst,fixed_args['ref_conv'])      
    
    return FWHM_inst
        

def convol_prof(prof_in,cen_bins,FWHM):
    r"""**Instrumental convolution**

    Convolves input profile with spectrograph LSF.
    Profile must be defined on a uniform spectral grid.

    Args:
        prof_in (array, float) : original spectral profile.
        cen_bins (array, float) : wavelength grid over which `prof_in` is defined.
        FWHM (float) : width of the Gaussian LSF used to convolve `prof_in`.
    
    Returns:
        prof_conv (array, float) : convolved spectral profile.
    
    """  
    
    #Half number of pixels in the kernel table at the resolution of the band spectrum
    #    - a range of 3.15 x FWHM ( = 3.15*2*sqrt(2*ln(2)) sigma = 7.42 sigma ) contains 99.98% of a Gaussian LSF integral
    #      we conservatively use a kernel covering 4.25 x FWHM / dbin pixels, ie 2.125 FWHM or 5 sigma on each side   
    dbins = cen_bins[1]-cen_bins[0]
    hnkern=npint(np.ceil(2.125*FWHM/dbins)+1)
    
    #Centered spectral table with same pixel widths as the band spectrum the kernel is associated to
    cen_bins_kernel=dbins*np.arange(-hnkern,hnkern+1)

    #Discrete Gaussian kernel 
    gauss_psf=np.exp(-np.power(  2.*np.sqrt(np.log(2.))*cen_bins_kernel/FWHM   ,2.))

    #Normalization
    gauss_kernel=gauss_psf/np.sum(gauss_psf)        

    #Convolution by the instrumental LSF   
    #    - bins must have the same size in a given table
    prof_conv=astro_conv(prof_in,gauss_kernel,boundary='extend')

    return prof_conv


def cond_conv_st_prof_tab(rv_osamp_line_mod,fixed_args,data_type):
    r"""**Spectral conversion and resampling**

    Enables/disables operations.

    Args:
        TBD
    
    Returns:
        TBD
    
    """  
    
    #Spectral oversampling
    if (fixed_args['mode']=='ana') and (rv_osamp_line_mod is not None):fixed_args['resamp'] = True
    else:fixed_args['resamp'] = False
    
    #Activate RV mode for analytical models of spectral profiles
    #    - theoretical profiles are processed in wavelength space
    #      measured profiles are processed in their space of origin
    #      analytical profiles are processed in RV space, and needs conversion back to wavelength space if data is in spectral mode  
    #    - since spectral tables will not have constant pixel size (required for model computation) in RV space, we activate the resampling mode so that all models will be calculated on this table and then resampled in spectral space,
    # rather than resampling the exposure in RV space
    if ('spec' in data_type) and (fixed_args['mode']=='ana'):
        fixed_args['spec2rv'] = True
        fixed_args['resamp'] = True 
        if fixed_args['line_trans'] is None:stop('Define "line_trans" to fit spectral data with "mode = ana"')
    else:fixed_args['spec2rv'] = False
    
    return None


def conv_st_prof_tab(inst,vis,isub,args,args_exp,line_mod_in,FWHM_inst):
    r"""**Spectral convolution, conversion, and resampling**

    Applies operations.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    
    #Convolve with instrumental response 
    #    - performed on table with constant bin size
    if FWHM_inst is None:line_mod_out = line_mod_in
    else:line_mod_out =  convol_prof( line_mod_in,args_exp['cen_bins'],FWHM_inst)

    #Convert table from RV to spectral space if relevant
    #    - w = w0*(1+rv/c)
    if args['spec2rv']:
        args_exp['edge_bins'] = args['line_trans']*gen_specdopshift(args_exp['edge_bins'])  
        args_exp['cen_bins'] = args['line_trans']*gen_specdopshift(args_exp['cen_bins'])  

    #Resample model on observed table if oversampling
    if args['resamp']:
        if inst is None:edge_bins_mod_out = args['edge_bins']
        else:edge_bins_mod_out = args['edge_bins'][inst][vis][isub]
        line_mod_out = bind.resampling(edge_bins_mod_out,args_exp['edge_bins'],line_mod_out, kind=args['resamp_mode'])       

    return line_mod_out

