#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import numpy as np
import lmfit
from lmfit import Parameters
from itertools import product as it_product
from copy import deepcopy
import os as os_system 
import bindensity as bind
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d,CubicSpline
from astropy.timeseries import LombScargle
from scipy import stats
import itertools
from scipy.signal import savgol_filter
from matplotlib.ticker import MultipleLocator
from ..ANTARESS_plots.utils_plots import autom_tick_prop,custom_axis
from ..ANTARESS_conversions.ANTARESS_binning import calc_bin_prof,resample_func,sub_calc_bins,sub_def_bins,weights_bin_prof
from ..ANTARESS_grids.ANTARESS_coord import get_timeorbit,calc_tr_contacts
from ..ANTARESS_analysis.ANTARESS_ana_comm import par_formatting
from ..ANTARESS_general.utils import stop,np_where1D,is_odd,closest,dataload_npz,gen_specdopshift,check_data,datasave_npz
from ..ANTARESS_general.constant_data import c_light
from ..ANTARESS_general.minim_routines import init_fit,fit_merit,call_lmfit


def MAIN_corr_wig(inst,gen_dic,data_dic,coord_dic,data_prop,plot_dic,system_param):
    r"""**Main wiggle correction routine.**    

    Determines and applies correction for ESPRESSO wiggles
    
    Wiggles are processed in wave_number space.
    We use :math:`\nu_\mathrm{ana} = c[m s^{-1}]/w[A]` because it is on the order of 10, with :math:`\nu[s^{-1}] = 10^{10} \nu_\mathrm{ana}`
    
    Periodogram frequencies :math:`F_{\nu,\mathrm{ana}}` correspond to wiggle periods 

    .. math::    
       P_{\nu}[s^{-1}] &= 10^{10} P_{\nu,\mathrm{ana}} \\
                    &= 10^{10} / F_{\nu,\mathrm{ana}}
                    
    with :math:`F_{\nu}[s] = 10^{-10} F_{\nu,\mathrm{ana}}`. 
    The corresponding period in wavelength space is
    
    .. math:: 
       P_{w}[A] &= c[A s^{-1}] P_{\nu}[s^{-1}]/\nu[s^{-1}]^2    \\
                &= c[A s^{-1}]/(F_{\nu}[s^{-1}] \nu[s^{-1}]^2)    \\
                &= c[m s^{-1}]/(F_{\nu,\mathrm{ana}} \nu_\mathrm{ana}^2)    \\
                &= w[A]^2/(F_{\nu,\mathrm{ana}} c[m s^{-1}])
  
    at this stage spectra can be written as (see rescale_profiles())
    
    .. math:: 
       F_\mathrm{corr}(w,t,v) = F_{\star}(w,v) C_\mathrm{ref}(w,v) L(t) W(w,t,v)
    
    where `W(w,t)` represents the wiggles and we can ignore :math:`C_\mathrm{ref}(w,v)` and `L(t)` as they will be removed when normalizing :math:`F_\mathrm{corr}(w,t,v)` by the visit master
  
    The approach is the same as with the flux balance module: spectra are first scaled back to count units and binned to smooth out noise, with a bin size short enough that it does not dilute the wiggles.
    When spectra are shifted into the star rest frame their calibration profile is shifted as well, so that a given exposure and its master are scaled using the same profile.
    In this way the contribution from the stellar spectrum and calibration profile is removed when calculating the ratio between the binned exposure and master count spectra, and 
    all low-resolution transmission spectra are comparable, even if they involved calibration profiles shifted to different positions.

    Args:
        TBD
    
    Returns:
        None
    
    """     
    print('   > Correcting ESPRESSO spectra for wiggles') 
    if inst!='ESPRESSO':stop('Wiggles are an ESPRESSO issue !')
    data_inst = data_dic[inst] 
    if (data_inst['type']!='spec2D'):stop('Data must be S2D')
    if len(gen_dic['wig_vis'])==0:gen_dic['wig_vis']=data_inst['visit_list']

    #Calculating data
    if gen_dic['calc_wig'] :
        print('         Fitting/correcting wiggles') 
        fixed_args = {'rasterized':True}

        #Optional arguments to be passed to the fit functions
        #    - common to all steps in the module
        fixed_args['use_cov'] = False 

        #Wiggle filter
        if gen_dic['wig_exp_filt']['mode']: 
            if gen_dic['wig_norm_ord']:print("WARNING: it is advised to disable gen_dic['wig_norm_ord'] with filter mode")
            fit_dic = {}

            #Disable analytical steps
            for key in ['wig_exp_samp','wig_exp_nu_ana','wig_exp_fit','wig_exp_point_ana','wig_vis_fit']:gen_dic[key]['mode'] = False

        #Analytical wiggle model
        else:

            #Reference nu for frequency calculations
            fixed_args['nu_ref'] = c_light/6000. 
        
            #Maximum number of fit evaluations
            fixed_args['max_nfev'] = 10000

            #Maximum degree of polynomial frequency variations
            #    - 2 is required over the full ESPRESSO range, but may be reduced to 1 for dataset with wiggles undetectable in the blue orders
            fixed_args['deg_Freq'] = gen_dic['wig_deg_Freq'] 

            #Maximum degree of polynomial amplitude variations
            #    - defined for each component
            fixed_args['deg_Amp'] = gen_dic['wig_deg_Amp'] 

            #Component colors
            color_comps={1:'dodgerblue',2:'orange',3:'limegreen',4:'magenta'}

            #Parameters generic names
            pref_names_amp={}
            suf_names_amp={}
            pref_names_freq={}
            suf_names_freq={}
            pref_names={}
            suf_names={}
            for comp_id in range(1,6):
                pref_names_amp[comp_id] = ['AmpGlob' for ideg in range(fixed_args['deg_Amp'][comp_id]+1)] 
                suf_names_amp[comp_id] =  ['_c'+str(ideg) for ideg in range(fixed_args['deg_Amp'][comp_id]+1)]
                pref_names_freq[comp_id] = ['Freq' for i in range(fixed_args['deg_Freq'][comp_id]+1)]
                suf_names_freq[comp_id] = ['_c'+str(ideg) for ideg in range(fixed_args['deg_Freq'][comp_id]+1)]
                pref_names[comp_id] = pref_names_amp[comp_id]+pref_names_freq[comp_id]+['Phi']
                suf_names[comp_id] =  suf_names_amp[comp_id]+suf_names_freq[comp_id]+['']
            suf_hyper = ['_off','_dx_east','_dy_east','_dz_east','_dz_west','_doff']    
    
            #Generic fit dictionary
            fit_prop_dic = {'calc_quant' : False}
            fit_dic={'uf_bd':{}}    
    
        #Resolution of plot model (1e-10 s-1)
        fixed_args['dnu_HR'] = gen_dic['wig_bin']/4.

        #Account for data noise in periodograms
        fixed_args['lb_with_err'] = False
        
        #FAP levels
        fixed_args['sampling_fap'] = [0.1, 0.05, 0.01]

        #Condition for spectral processing in visit fit
        if gen_dic['wig_vis_fit']['mode']:
            cond_exp_proc_vis = True 
            if gen_dic['wig_vis_fit']['fixed'] and not (gen_dic['wig_vis_fit']['plot_mod'] | gen_dic['wig_vis_fit']['plot_rms'] | gen_dic['wig_vis_fit']['plot_hist']| gen_dic['wig_vis_fit']['plot_par_chrom']):cond_exp_proc_vis = False  
        else:cond_exp_proc_vis = False  
        cond_exp_proc = gen_dic['wig_exp_filt']['mode'] | gen_dic['wig_exp_init']['mode'] | gen_dic['wig_exp_samp']['mode'] | gen_dic['wig_exp_fit']['mode'] 
        
        #Indexes of order to be fitted
        iord_fit_ref = range(data_inst['nord_ref'])

        #Sampling initialization
        if gen_dic['wig_exp_samp']['mode']:             

            #Sampled components
            fixed_args['comp_ids'] = gen_dic['wig_exp_samp']['comp_ids']
            fixed_args['comp_id_max'] = np.max(fixed_args['comp_ids'])
            
            #Guess frequency for definition of sampling bands
            freq_params_samp = {}
            for comp_id in fixed_args['comp_ids']:
                for ideg in range(fixed_args['deg_Freq'][comp_id]+1):freq_params_samp['Freq'+str(comp_id)+'_c'+str(ideg)+'_off'] = gen_dic['wig_exp_samp']['freq_guess'][comp_id]['c'+str(ideg)]

        #------------------------------

        #Process each visit
        for ivisit,vis in enumerate(gen_dic['wig_vis']):
            print('           Processing visit ',vis)
            data_vis=data_inst[vis]
            data_prop_vis = data_prop[inst][vis]
            
            #Indexes of orders to be fitted
            if (vis in gen_dic['wig_ord_fit']) and (len(gen_dic['wig_ord_fit'][vis])>0):
                iord_fit_list = np.intersect1d(iord_fit_ref,gen_dic['wig_ord_fit'][vis]) 
            else:iord_fit_list = iord_fit_ref
            
            #Original indexes of exposures to be included in the master calculation 
            if (vis not in gen_dic['wig_exp_mast']) or (gen_dic['wig_exp_mast'][vis]=='all'):iexp_mast_list = np.arange(data_vis['n_in_visit'])
            else:iexp_mast_list = gen_dic['wig_exp_mast'][vis]

            #Original indexes of exposures to be fitted 
            if (vis not in gen_dic['wig_exp_in_fit']) or (gen_dic['wig_exp_in_fit'][vis]=='all'):iexp_fit_list = np.arange(data_vis['n_in_visit'])
            else:iexp_fit_list = gen_dic['wig_exp_in_fit'][vis]
            nexp_fit_list = len(iexp_fit_list)            
            
            #RV used for spectra alignment set to keplerian model
            #    - RV relative to CDM are used since the correction is performed per visit, so that the systemic RV does not need to be set                
            rv_al_all = coord_dic[inst][vis]['RV_star_stelCDM'] 

            #Target cartesian coordinates
            #    - coordinates in the plane of horizon, with the y axis pointing toward the north:
            # x = cos(th) = sin(az)
            # y = sin(th) = cos(az) 
            #      with th = pi/2 - az, assuming az = 0 at the north and positive toward the east
            #    - coordinate in the perpendicular plane, along the zenith axis
            # z = sin(alt)     
            fixed_args['z_mer']  = data_dic[inst][vis]['z_mer'] 
            tel_coord_vis={
                't_dur' : coord_dic[inst][vis]['t_dur'], 
                'cen_ph' : coord_dic[inst][vis][gen_dic['studied_pl'][0]]['cen_ph'],
                'az': data_prop_vis['az'],
                'x_az' : np.sin(data_prop_vis['az']*np.pi/180.),
                'y_az' : np.cos(data_prop_vis['az']*np.pi/180.), 
                'z_alt' : np.sin(data_prop_vis['alt']*np.pi/180.),  
                'cond_eastmer' : np.repeat(False,data_vis['n_in_visit'])}
            tel_coord_vis['cond_eastmer'][data_vis['idx_eastmer']] = True    
            tel_coord_vis['cond_westmer'] = ~tel_coord_vis['cond_eastmer']
          
            #Check for guide star change during night, unless disabled
            #    - the wiggle are modeled in three phases, as wiggle behaviour changes with guide star:
            #      if the guide star changed before meridian crossing, the model is shift / A / B
            #      if the guide star changed after meridian crossing, the model is A / B / shift
            #      where A and B are models east or west of the meridian  
            if not gen_dic['wig_exp_filt']['mode']:suf_hyper_vis = deepcopy(suf_hyper)
            fixed_args['iexp_guidchange'] = 1e10
            cen_ph_guid = None
            shift_group = None
            tel_coord_vis['cond_shift'] = np.repeat(False,data_vis['n_in_visit'])
            if vis not in gen_dic['wig_no_guidchange']:
                delta_guid_RA = data_prop_vis['guid_coord'][1:,0] - data_prop_vis['guid_coord'][0:-1,0]    
                delta_guid_DEC = data_prop_vis['guid_coord'][1:,1] - data_prop_vis['guid_coord'][0:-1,1]
                iexp_guidchange =  np_where1D((np.abs(delta_guid_RA)>1e-2) | (np.abs(delta_guid_DEC)>1e-2)) 
                if len(iexp_guidchange)>0:
                    if len(iexp_guidchange)>1:stop('             Check guide star coordinates')
                    print('             Guide star changed during visit: enabling offset')
                    fixed_args['iexp_guidchange'] = iexp_guidchange[0] 
                    cen_ph_guid = 0.5*(tel_coord_vis['cen_ph'][fixed_args['iexp_guidchange']]+tel_coord_vis['cen_ph'][fixed_args['iexp_guidchange']+1])
                    
                    #Exposures before (resp. after) guide star change are set as 'shifted' if the change occurs before (resp. after) meridian crossing 
                    if fixed_args['iexp_guidchange']<data_vis['idx_mer']:
                        shift_group = 'pre'
                        tel_coord_vis['cond_shift'] = (np.arange(data_vis['n_in_visit']) <= fixed_args['iexp_guidchange'])             
                    else:
                        shift_group = 'post'                        
                        tel_coord_vis['cond_shift'] = (np.arange(data_vis['n_in_visit']) > fixed_args['iexp_guidchange'])  
            
                    #Removing shift exposures from other models
                    tel_coord_vis['cond_eastmer'][tel_coord_vis['cond_shift']] = False
                    tel_coord_vis['cond_westmer'][tel_coord_vis['cond_shift']] = False
                    
                    #Hyperparameter suffix
                    if not gen_dic['wig_exp_filt']['mode']:
                        suf_hyper_vis+=['_doff_sh','_dx_shift','_dy_shift','_dz_shift']

            #Maximum edges of fitted spectral tables
            glob_min_bins = 1e100
            glob_max_bins = -1e100    
  
            #High-resolution model for plotting
            if (not gen_dic['wig_exp_filt']['mode']) and (((gen_dic['wig_exp_point_ana']['mode']) and (gen_dic['wig_exp_point_ana']['plot'])) or ((gen_dic['wig_vis_fit']['mode']) and (gen_dic['wig_vis_fit']['plot_chrompar_point']))):
            
                #High-res model
                #    - cubic spline captures better the altitude variations
                min_bjd = coord_dic[inst][vis]['bjd'][0]
                max_bjd = coord_dic[inst][vis]['bjd'][-1]
                dbjd_HR = 1./(3600.*24.)
                nbjd_HR = round((max_bjd-min_bjd)/dbjd_HR)
                bjd_HR=min_bjd+dbjd_HR*np.arange(nbjd_HR)
                cen_ph_HR = get_timeorbit(gen_dic['studied_pl'][0],coord_dic[inst][vis],bjd_HR, system_param[gen_dic['studied_pl'][0]], 0.)[1]
                az_HR = CubicSpline(coord_dic[inst][vis]['bjd'],data_prop[inst][vis]['az'])(bjd_HR)
                alt_HR = CubicSpline(coord_dic[inst][vis]['bjd'],data_prop[inst][vis]['alt'])(bjd_HR) 
                cen_ph_mer = cen_ph_HR[closest(az_HR,180.)]                 
                tel_coord_HR={
                    'az' : az_HR,
                    'alt' : alt_HR,
                    'x_az' : np.sin(az_HR*np.pi/180.),
                    'y_az' : np.cos(az_HR*np.pi/180.),
                    'z_alt' : np.sin(alt_HR*np.pi/180.),
                    'cond_eastmer' : az_HR < 180.,
                    'cond_westmer' : az_HR > 180.,
                    'cond_shift' : np.repeat(False,nbjd_HR)}
                if shift_group == 'pre':tel_coord_HR['cond_shift'][cen_ph_HR<=cen_ph_guid] = True
                elif shift_group == 'post':tel_coord_HR['cond_shift'][cen_ph_HR>cen_ph_guid] = True
                tel_coord_HR['cond_eastmer'][tel_coord_HR['cond_shift']] = False
                tel_coord_HR['cond_westmer'][tel_coord_HR['cond_shift']] = False
                
                #Contact phases for main planet
                contact_phases=calc_tr_contacts(data_vis['system_prop']['achrom'][gen_dic['studied_pl'][0]][0],system_param[gen_dic['studied_pl'][0]],plot_dic['stend_ph'],system_param['star'])

            #------------------------------------------------------------------

            #Save directories
            path_dic={}
            for step,dir_name in zip(['wig_exp_init','wig_exp_filt','wig_exp_samp','wig_exp_nu_ana','wig_exp_fit','wig_exp_point_ana'],['Init','Filter','Sampling','Chrom','Global','Coord']):
                path_dic['datapath_'+dir_name] = gen_dic['save_data_dir']+'Corr_data/Wiggles/Exp_fit/'+inst+'_'+vis+'/'+dir_name+'/' 
                if gen_dic['wig_exp_ana']:                    
                    path_dic['plotpath_'+dir_name] = gen_dic['save_dir']+gen_dic['main_pl_text']+'_Plots/Spec_raw/Wiggles/Exp_fit/'+inst+'_'+vis+'/'+dir_name+'/'
                    if gen_dic[step]['mode']:             
                        if not os_system.path.exists(path_dic['datapath_'+dir_name]):os_system.makedirs(path_dic['datapath_'+dir_name])  
                        if step=='wig_exp_samp':
                            path_sampling_fit = path_dic['datapath_Sampling']+'/Comp'+str(fixed_args['comp_id_max'])+'/'
                            if not os_system.path.exists(path_sampling_fit):os_system.makedirs(path_sampling_fit) 
                        key_step = list(gen_dic[step].keys())
                        for key in key_step:
                            if ('plot' in key) and gen_dic[step][key] and (not os_system.path.exists(path_dic['plotpath_'+dir_name])):
                                os_system.makedirs(path_dic['plotpath_'+dir_name])  
            
            #Initialize default model structure for chromatic exposure fit
            #    - those parameters define the model in a current exposure (they define the chromatic component of hyperparameters)
            #    - amplitude is too dependent on the dataset and has no default initialization
            #      frequency of wiggle components is stable between datasets and can be set to default values, used as initial guesses or fixed models for the global fit (unless overwritten by the harmonic fit) 
            #    - only parameters effectively used to define an exposure model are set to vary, to initialize their guesses and priors
            if gen_dic['wig_exp_fit']['mode'] or gen_dic['wig_exp_samp']['mode'] :
                
                #Initialization of the parameter structure
                #    - well-defined priors, through inputs, are critical for the good convergence of the global fit
                mod_prop_exp={}
                varpar_priors_exp = {}
                for comp_id in range(1,6):
                    comp_str=str(comp_id) 
                    
                    #Model properties and default values                             
                    for pref,suf in zip(pref_names_amp[comp_id]+['Phi'],suf_names_amp[comp_id]+['']):
                        mod_prop_exp[pref+comp_str+suf+'_off']  = {'vary':True,'guess':0.}
                    for pref,suf in zip(pref_names_freq[comp_id],suf_names_freq[comp_id]):
                        mod_prop_exp[pref+comp_str+suf+'_off']  = {'vary':True,'guess':1.}

                    #Default priors
                    varpar_priors_exp.update({            
                        'AmpGlob'+comp_str+'_c0_off':{'low':-1e-2 ,'high':1e-2},'AmpGlob'+comp_str+'_c1_off':{'low':-1e-3,'high':1e-3},'AmpGlob'+comp_str+'_c2_off':{'low':-1e-4,'high':1e-4},'AmpGlob'+comp_str+'_c3_off':{'low':-1e-5 ,'high':1e-5},'AmpGlob'+comp_str+'_c4_off':{'low':-1e-6 ,'high':1e-6},        
                        'Freq'+comp_str+'_c0_off':{'low':0.,'high':10.},'Freq'+comp_str+'_c1_off':{'low':-1.,'high':1.},'Freq'+comp_str+'_c2_off':{'low':-1.,'high':1.},'Freq'+comp_str+'_c2_off':{'low':-1.,'high':1.}, 
                        'Phi'+comp_str+'_off':{'low':-100.,'high':100.}})                    

                #Chromatic initialization of exposure fits
                if gen_dic['wig_exp_fit']['mode']:
                    if (gen_dic['wig_exp_fit']['init_chrom']) and (len(glob.glob(path_dic['datapath_Chrom']+'/Fit_results_iexpGroup*.npz'))>0):
                        print('               Initialization from chromatic fit')
                        init_chrom = True
                    else:
                        print('               Default initialization')
                        init_chrom = False    
                       
                        #Initialize frequency model to requested values
                        for comp_id in gen_dic['wig_exp_fit']['freq_guess']:
                            for ideg in range(fixed_args['deg_Freq'][comp_id]+1):mod_prop_exp['Freq'+str(comp_id)+'_c'+str(ideg)+'_off']['guess'] = gen_dic['wig_exp_fit']['freq_guess'][comp_id]['c'+str(ideg)]

                #Overwrite default prior
                if (vis in gen_dic['wig_exp_fit']['prior_par']): 
                    for par in gen_dic['wig_exp_fit']['prior_par'][vis]:
                        if (par+'_off') in varpar_priors_exp:
                            if ('guess' in gen_dic['wig_exp_fit']['prior_par'][vis][par]):varpar_priors_exp[par+'_off']['guess'] = gen_dic['wig_exp_fit']['prior_par'][vis][par]['guess']
                            if ('low' in gen_dic['wig_exp_fit']['prior_par'][vis][par]):varpar_priors_exp[par+'_off']['low'] = gen_dic['wig_exp_fit']['prior_par'][vis][par]['low']
                            if ('high' in gen_dic['wig_exp_fit']['prior_par'][vis][par]):varpar_priors_exp[par+'_off']['high'] = gen_dic['wig_exp_fit']['prior_par'][vis][par]['high']
                
                #Default variable properties 
                stable_pointpar_exp = {}
                for par in mod_prop_exp:
                    stable_pointpar_exp[par.split('off')[0]] = False
                    if mod_prop_exp[par]['vary']:
                        if par in varpar_priors_exp:varpar_priors_exp[par]['mod']='uf'              

            #-----------------------------------------------------

            #Initialize default model structure for global visit fit
            #    - those parameters define the model over a complete visit (they define the temporal and chromatic components of hyperparameters)            
            #    - only parameters effectively used to define an exposure model are set to vary, to initialize their guesses and priors    
            if cond_exp_proc_vis:
                mod_prop_vis={}  
                varpar_priors_vis = {}
                stable_pointpar_vis = {}
                for comp_id in gen_dic['wig_vis_fit']['comp_ids']:
                    comp_str=str(comp_id)
                    for pref,suf in zip(pref_names[comp_id],suf_names[comp_id]):
                        stable_pointpar_vis[pref+comp_str+suf+'_'] = False                        
                        for kcoord in suf_hyper_vis:
                            mod_prop_vis[pref+comp_str+suf+kcoord]  = {'vary':True,'guess':1.}
                            varpar_priors_vis[pref+comp_str+suf+kcoord]  = {'low':-1e4,'high':1e4}

                fixed_args['var_par_vis'] = []
                for par in mod_prop_vis:
                    if mod_prop_vis[par]['vary']:
                        fixed_args['var_par_vis']+=[par]    
                        if par in varpar_priors_vis:
                            varpar_priors_vis[par]['mod']='uf'    
     
            #----------------------------------------------------------------------                         

            #Initialize exposure-per-exposure analysis
            if gen_dic['wig_exp_ana']:
                print('             Exposure analysis')
                if gen_dic['wig_exp_init']['mode']:     print('             - Initialization')
                if gen_dic['wig_exp_filt']['mode']:     print('             - Filtering')
                if gen_dic['wig_exp_samp']['mode']:     print('             - Periodogram sampling')
                if gen_dic['wig_exp_nu_ana']['mode']:   print('             - Chromatic sampling analysis')
                if gen_dic['wig_exp_fit']['mode']:      print('             - Global exposure fit')
                if gen_dic['wig_exp_point_ana']['mode']:print('             - Temporal analysis')                    

                #Number of grouped exposures and indexes of exposures making each group
                if (vis in gen_dic['wig_exp_groups']) and (len(gen_dic['wig_exp_groups'][vis])>0):
                    wig_exp_groups = gen_dic['wig_exp_groups'][vis]
                    n_exp_groups = len(gen_dic['wig_exp_groups'][vis])
                else:
                    wig_exp_groups = None
                    n_exp_groups = nexp_fit_list     
                
                #Analytical model
                fit_dic['save_dir'] = gen_dic['save_data_dir']+'/Corr_data/Wiggles/Exp_fit/'
                if not gen_dic['wig_exp_filt']['mode']:
                    fit_dic.update({
                        'merit':{},                    
                        'fit_mode' : 'chi2'})
    
                #Equivalent tables for grouped exposures
                tel_coord_expgroup = {
                    'az' : np.zeros(n_exp_groups,dtype=float)*np.nan,
                    'x_az' : np.zeros(n_exp_groups,dtype=float)*np.nan,
                    'y_az' : np.zeros(n_exp_groups,dtype=float)*np.nan,
                    'z_alt' : np.zeros(n_exp_groups,dtype=float)*np.nan,
                    'cond_eastmer' : np.repeat(False,n_exp_groups),
                    'cond_westmer' : np.repeat(False,n_exp_groups),
                    'cond_shift' : np.repeat(False,n_exp_groups),
                    'cen_ph' : np.zeros(n_exp_groups,dtype=float)*np.nan}

                #RMS
                if gen_dic['wig_exp_fit']['mode'] | gen_dic['wig_exp_filt']['mode']:
                    rms_exp_raw = np.zeros(n_exp_groups,dtype=float)*np.nan
                    median_err = np.zeros(n_exp_groups,dtype=float)*np.nan
                rms_exp_fit = np.zeros(n_exp_groups,dtype=float)*np.nan

            #Initialize global dictionary to store binned transmission spectra
            bin_dic={}
            data_glob={}
            
            #------------------------------------------------------
        
            #Spectral processing
            if cond_exp_proc or cond_exp_proc_vis:
           
                #Retrieve common visit table 
                if not gen_dic['wig_indiv_mast'] :
                    data_com = np.load(data_inst[vis]['proc_com_data_paths']+'.npz',allow_pickle=True)['data'].item() 
                    dim_exp_com = data_inst[vis]['dim_exp']
                    nspec_com = data_inst['nspec']   
            
                #Calculate pre/post-meridian masters
                if gen_dic['wig_merid_diff']:
                    if not gen_dic['wig_indiv_mast']:
                        data_mast = {'eastmer':{},'westmer':{}} 
                        data_to_bin={'eastmer':{},'westmer':{}} 
                        
                    #Identify meridian crossing
                    idx_to_bin_mer={
                        'eastmer':np.intersect1d(iexp_mast_list,data_vis['idx_eastmer']),
                        'westmer':np.intersect1d(iexp_mast_list,data_vis['idx_westmer'])}
                    idx_to_bin = list(idx_to_bin_mer['eastmer'])+list(idx_to_bin_mer['westmer'])
                    iexp2mer = np.append(np.repeat('eastmer',len(data_vis['eastmer'])) , np.repeat('westmer',len(data_vis['westmer'])) )
    
                #Calculate global master
                else:
                    idx_to_bin = deepcopy(iexp_mast_list)
                    idx_to_bin_mer={'glob':idx_to_bin}
                    n_in_bin = len(idx_to_bin)
                    if not gen_dic['wig_indiv_mast']:
                        data_mast={'glob':{}}                    
                        data_to_bin={'glob':{}}
                    iexp2mer = np.repeat('glob',data_vis['n_in_visit'])

                #No flux scaling and master are defined at this stage for the weighing
                #    - the wiggle master spectrum is in any case processed in the star rest frame, so that the stellar line do not contribute to weighing 
                flux_ref = np.ones(data_vis['dim_exp'])
    
                #Retrieving data that will be used in the binning to calculate a master profile
                #    - in the process_bin_prof() routine dedicated to the analysis of binned profiles, original exposures are resampled on a common table after 
                # upload if relevant. Here the original data need to be resampled for each new exposure, thus here we just upload and store the profiles
                #    - data is processed here in wavelength space
                iexp_proc_list = np.unique(list(iexp_mast_list)+list(iexp_fit_list)) 
                for isub_exp,iexp in enumerate(iexp_proc_list):

                    #Latest processed disk-integrated data and associated tables
                    #    - associated tables are defined in the same frame and over the same table as the exposure spectrum
                    data_exp = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp))
                    data_glob[iexp] = {}
                    for key in ['cen_bins','edge_bins','flux','cond_def','cov']:data_glob[iexp][key] = data_exp[key]
                    if data_vis['tell_sp']:data_glob[iexp]['tell'] = dataload_npz(data_dic[inst][vis]['tell_DI_data_paths'][iexp])['tell']             
                    else:data_glob[iexp]['tell'] = None
                    data_glob[iexp]['mean_gdet'] = dataload_npz(data_dic[inst][vis]['mean_gdet_DI_data_paths'][iexp])['mean_gdet']             
           
                    #Aligning exposures, shifting them from the solar system barycentric rest frame (source) into a common frame (receiver)
                    #      see gen_specdopshift() :            
                    # w_receiver = w_source * (1+ (rv[s/r]/c))
                    # w_common = w_solbar * (1- rv[common/solbar]/c))
                    #      no need to align them in the star rest frame: profiles need to be aligned in a common frame to calculate transmission spectra, 
                    # and are then shifted back
                    dopp_fact = 1./gen_specdopshift(rv_al_all[iexp])  
                    
                    #Only the exposure table is modified if data do not share a common table
                    #    - data will be resampled along with the master in a later stage
                    if (not data_vis['comm_sp_tab']):
                        data_glob[iexp]['edge_bins'] *= dopp_fact
                        data_glob[iexp]['cen_bins'] *= dopp_fact                        
                    
                    #Exposure and associated tables are resampled on the common table otherwise
                    else:
                        for iord in iord_fit_list: 
                            edge_bins_rest = data_glob[iexp]['edge_bins'][iord]*dopp_fact
                            data_glob[iexp]['flux'][iord],data_glob[iexp]['cov'][iord] = bind.resampling(data_com['edge_bins'][iord],edge_bins_rest, data_glob[iexp]['flux'][iord], cov = data_glob[iexp]['cov'][iord], kind=gen_dic['resamp_mode'])
                            data_glob[iexp]['mean_gdet'][iord] = bind.resampling(data_com['edge_bins'][iord],edge_bins_rest, data_glob[iexp]['mean_gdet'][iord] , kind=gen_dic['resamp_mode']) 
                            if data_vis['tell_sp']:data_glob[iexp]['tell'][iord] = bind.resampling(data_com['edge_bins'][iord],edge_bins_rest, data_glob[iexp]['tell'][iord] , kind=gen_dic['resamp_mode']) 
                        data_glob[iexp]['cond_def'] = ~np.isnan(data_glob[iexp]['flux'])
                        data_glob[iexp]['edge_bins']=   data_com['edge_bins']
                        data_glob[iexp]['cen_bins'] =   data_com['cen_bins']                   
                           
                    #Convert spectral tables in 10-10 s-1
                    #    - reorderd at a later step
                    data_glob[iexp]['cen_nu'] = c_light/data_glob[iexp]['cen_bins']
                    data_glob[iexp]['edge_nu'] = c_light/data_glob[iexp]['edge_bins']
                    
                    #Normalize to global flux unity
                    #    - even if the color balance was corrected for, spectra were left to their overall flux level
                    #    - they must be set to the same level before the master is calculated
                    #      while not necessary for exposures not used in the master, it will make it easier to compare all normalized exposures
                    #    - we neglect possible differences in defined pixels between exposures
                    #    - we neglect this scaling in weights
                    dcen_wav = data_glob[iexp]['edge_bins'][:,1::] - data_glob[iexp]['edge_bins'][:,0:-1]
                    cond_def = data_glob[iexp]['cond_def']
                    flux_glob = 0.
                    for iord in iord_fit_list:
                        flux_glob+= np.sum(dcen_wav[iord][cond_def[iord]])/np.sum(data_glob[iexp]['flux'][iord][cond_def[iord]]*dcen_wav[iord][cond_def[iord]])
                    for iord in iord_fit_list:
                        data_glob[iexp]['flux'][iord],data_glob[iexp]['cov'][iord] = bind.mul_array(data_glob[iexp]['flux'][iord] , data_glob[iexp]['cov'][iord],np.repeat(flux_glob,data_vis['nspec']))
                    
                    #Exposures used in master calculations
                    if iexp in idx_to_bin:
                        if gen_dic['wig_merid_diff']:idmer = iexp2mer[iexp]  
                        else:idmer = 'glob'

                        #Weight definition 
                        #    - at this stage of the pipeline no broadband flux scaling was applied
                        #    - weights with at least one undefined pixels are set to 1 for all binned exposures (ie, no weighing is applied) within calc_bin_prof()    
                        weight_mean_gdet_exp = data_glob[iexp]['mean_gdet'] if gen_dic['cal_weight'] else None 
                        data_glob[iexp]['weight'] = weights_bin_prof(range(data_inst['nord']),None,inst,vis,gen_dic['corr_Fbal'],gen_dic['corr_FbalOrd'],gen_dic['save_data_dir'],gen_dic['type'],data_inst['nord'],iexp,'DI',data_inst['type'],data_vis['dim_exp'],data_glob[iexp]['tell'],weight_mean_gdet_exp,data_glob[iexp]['cen_bins'],1.,flux_ref,None,glob_flux_sc=1./flux_glob)            
    
                        #Resampling if a common master is used
                        #    - if exposures do not share a common table they were kept on individual exposures; here we only resample those involved in the master calculation
                        #    - exposures sharing a common table have already been resampled
                        if (not gen_dic['wig_indiv_mast']):
                            if (not data_vis['comm_sp_tab']):                    
                                data_to_bin[idmer][iexp]={
                                    'flux':np.zeros(data_vis['dim_exp'],dtype=float)*np.nan,
                                    'cov':np.zeros(data_inst['nord'],dtype=object)}
                                data_to_bin[idmer][iexp]['weight']=np.zeros(data_vis['dim_exp'],dtype=float)*np.nan
                                for iord in iord_fit_list: 
                                    data_to_bin[idmer][iexp]['flux'][iord],data_to_bin[idmer][iexp]['cov'][iord] = bind.resampling(data_com['edge_bins'][iord], data_glob[iexp]['edge_bins'][iord], data_glob[iexp]['flux'][iord] , cov = data_glob[iexp]['cov'][iord], kind=gen_dic['resamp_mode'])                                                        
                                    data_to_bin[idmer][iexp]['weight'][iord] = bind.resampling(data_com['edge_bins'][iord], data_glob[iexp]['edge_bins'][iord], data_glob[iexp]['weight'][iord] ,kind=gen_dic['resamp_mode'])   
                                data_to_bin[idmer][iexp]['cond_def'] = ~np.isnan(data_to_bin[idmer][iexp]['flux'])   
                            else:
                                data_to_bin[idmer][iexp]={}
                                for key in ['flux','cond_def','weight','cov']:data_to_bin[idmer][iexp][key] = data_glob[iexp][key]

                #Normalized weights
                #    - to calculate the weighted wiggles in the same way as the master
                #    - ideally we should convert the order-per-order weight tables of each exposure on a 1D table, so that they can then be multiplied directly to the wiggles models of the corresponding exposure later on
                #      this would however be extremely heavy, and we approximate the weights by the mean SNR over all orders
                if cond_exp_proc_vis: 
                    mean_SNR = data_prop[inst][vis]['mean_SNR']
                    norm_weight_mast={}
                    for idmer in idx_to_bin_mer:norm_weight_mast[idmer] = mean_SNR[idx_to_bin_mer[idmer]]/np.sum(mean_SNR[idx_to_bin_mer[idmer]])
                    
                #Calculate common master
                #    - we do not use the global master of the flux balance correction because it is defined over all visits 
                if (not gen_dic['wig_indiv_mast']):
                    for idmer in idx_to_bin_mer:                    
                        data_mast[idmer] = calc_bin_prof(idx_to_bin_mer[idmer],data_dic[inst]['nord'],dim_exp_com,nspec_com,data_to_bin[idmer],inst,len(idx_to_bin_mer[idmer]),data_com['cen_bins'],data_com['edge_bins'])

            #------------------------------------------------------

            #Limits for periodograms depending on bin size
            if gen_dic['wig_bin']<0.008:
                fixed_args['min_x_glob_perio'] = 0.2
                fixed_args['max_x_glob_perio'] = 200.   
                fixed_args['min_y_glob_perio'] = 1e-5    
                fixed_args['log_glob_perio'] = True
            else:
                fixed_args['min_x_glob_perio'] = 0.1   
                fixed_args['max_x_glob_perio'] = 8.  #10.  
                fixed_args['min_y_glob_perio'] = 1e-4    
                fixed_args['log_glob_perio'] = False  

            #Initialize common periodogram for component search
            if gen_dic['wig_exp_init']['plot_hist'] or gen_dic['wig_vis_fit']['plot_hist']:

                #Search range for periodograms 
                comm_freq_comp={}
                if gen_dic['wig_bin']<0.008:comm_freq_comp['glob'] = fixed_args['min_x_glob_perio']*10**(np.arange(10000.)*(np.log10(fixed_args['max_x_glob_perio'])-np.log10(fixed_args['min_x_glob_perio']))/(10000.-1)),      #regularly spaced in log10 space    
                else:comm_freq_comp['glob'] = np.linspace(fixed_args['min_x_glob_perio'],fixed_args['max_x_glob_perio'], 1000)            
                comm_freq_comp.update({          
                    3:np.linspace(1.1-0.5,1.1+0.5,400),
                    2:np.linspace(2.-1.,2.+1.,400),
                    1:np.linspace(3.9-2,3.9+2,400)
                    })
                if gen_dic['wig_bin']<0.008:comm_freq_comp[4] = np.linspace(100.,200.,400)
                if gen_dic['wig_exp_init']['plot_hist']:
                    comm_power_comp = {}
                    for comp_id in comm_freq_comp:comm_power_comp[comp_id]=np.zeros(len(comm_freq_comp[comp_id]),dtype=float)

            #Initialize filter results
            if gen_dic['wig_exp_filt']['mode']:
                dic_filter = {}

            #Processing each exposure
            init_ord = False
            init_exp = False
            iexp_group_list = []    
            iexp2glob = {}
            for isub_exp,iexp in enumerate(iexp_fit_list):
                
                #Process exposure 
                if gen_dic['wig_exp_ana']:

                    #Index of next exposure is set to highest value to meet fit condition if current exposure is the last one processed
                    iexp_next = iexp_fit_list[isub_exp+1] if (isub_exp < nexp_fit_list-1) else data_vis['n_in_visit']
                   
                    #Identify the global exposure group to which current and next exposures belong to
                    if wig_exp_groups is not None:
                        exp_glob_bin = [ibin for ibin in range(n_exp_groups) if iexp in wig_exp_groups[ibin]]   #indexes of exposure groups to which current exposure belongs to
                        iexp_glob_bin = exp_glob_bin[0] if len(exp_glob_bin)==1 else None
                        
                        exp_glob_bin_next = [ibin for ibin in range(n_exp_groups) if iexp_next in wig_exp_groups[ibin]]  #indexes of exposure groups to which next exposure belongs to
                        iexp_glob_bin_next = exp_glob_bin_next[0] if len(exp_glob_bin_next)==1 else n_exp_groups

                        #Store exposures effectively included in current exposure group
                        if (not init_exp):
                            init_exp = True
                            wig_exp_group = [iexp]
                            iexp_group_list +=[iexp_glob_bin]
                        else:wig_exp_group+=[iexp]

                    #Set group orders to global orders if group are undefined
                    else:
                        iexp_glob_bin = iexp
                        iexp_glob_bin_next = iexp_next
                        iexp_group_list +=[iexp]
                        
                    #Tracing exposure group
                    if gen_dic['wig_exp_filt']['mode']:iexp2glob[iexp] = deepcopy(iexp_glob_bin)

                #Processing spectral ratio
                if cond_exp_proc or cond_exp_proc_vis:
                    idmer = iexp2mer[iexp]
                    idx_to_bin = idx_to_bin_mer[idmer]

                    #Using common master disk-integrated profile for current exposure
                    if (not gen_dic['wig_indiv_mast']):
                        
                        #Resampling common master on current exposure table
                        if (not data_vis['comm_sp_tab']):
                            data_mast_exp={'flux':np.zeros(data_vis['dim_exp'],dtype=float)*np.nan,
                                           'cov':np.zeros(data_inst['nord'],dtype=object)}
                            for iord in iord_fit_list: 
                                data_mast_exp['flux'][iord],data_mast_exp['cov'][iord] = bind.resampling(data_glob[iexp]['edge_bins'][iord],data_com['edge_bins'][iord],data_mast[idmer]['flux'][iord] , cov = data_mast[idmer]['cov'][iord], kind=gen_dic['resamp_mode'])                                                        
                            data_mast_exp['cond_def'] = ~np.isnan(data_mast_exp['flux'])  
    
                        #Master table is shared by all exposures
                        else:
                            data_mast_exp = deepcopy(data_mast[idmer])
                    
                    #Calculating master disk-integrated profile on current exposure table
                    #    - the master is calculated in a given exposure:
                    # + if it is the first one
                    # + if it is another one and exposures do not share a common table in current visit                        
                    elif (isub_exp==0) or (not data_vis['comm_sp_tab']):
                        data_to_bin_extract={}
                        for iexp_off in idx_to_bin:

                            #Resampling on current exposure spectral table if required
                            #    - data is stored with the same indexes as in idx_to_bin
                            #    - all exposures must be defined on the same spectral table before being binned
                            #    - profiles are resampled if defined on their individual tables
                            if (not data_vis['comm_sp_tab']):
                                data_to_bin_extract[iexp_off]={
                                    'flux':np.zeros(data_vis['dim_exp'],dtype=float)*np.nan,
                                    'cov':np.zeros(data_inst['nord'],dtype=object)}
                                data_to_bin_extract[iexp_off]['weight']=np.zeros(data_vis['dim_exp'],dtype=float)*np.nan
                                for iord in iord_fit_list: 
                                    data_to_bin_extract[iexp_off]['flux'][iord],data_to_bin_extract[iexp_off]['cov'][iord] = bind.resampling(data_glob[iexp]['edge_bins'][iord], data_glob[iexp_off]['edge_bins'][iord], data_glob[iexp_off]['flux'][iord] , cov = data_glob[iexp_off]['cov'][iord], kind=gen_dic['resamp_mode'])                                                        
                                    data_to_bin_extract[iexp_off]['weight'][iord] = bind.resampling(data_glob[iexp]['edge_bins'][iord], data_glob[iexp_off]['edge_bins'][iord], data_glob[iexp_off]['weight'][iord] ,kind=gen_dic['resamp_mode'])   
                                data_to_bin_extract[iexp_off]['cond_def'] = ~np.isnan(data_to_bin_extract[iexp_off]['flux'])   
                            else:
                                for key in ['flux','cond_def','cov','weight']:data_to_bin_extract[iexp_off][key] = data_glob[iexp_off][key]

                        #Calculate master on current exposure table
                        data_mast_exp = calc_bin_prof(idx_to_bin,data_dic[inst]['nord'],data_vis['dim_exp'],data_vis['nspec'],data_to_bin_extract,inst,n_in_bin,data_glob[iexp]['cen_bins'],data_glob[iexp]['edge_bins'])
    
                    #Processing master and exposure, now defined over common pixels and within selected range
                    #    - from here on data is processed in nu space
                    #    - master and exposure are still defined per order
                    cond_kept_all = data_glob[iexp]['cond_def'] & data_mast_exp['cond_def']
                    if (vis in gen_dic['wig_range_fit']) and (len(gen_dic['wig_range_fit'][vis])>0):
                        cond_sel = np.zeros(data_vis['dim_exp'],dtype=bool)
                        for bd_band_loc in gen_dic['wig_range_fit'][vis]:cond_sel|=(data_glob[iexp]['edge_nu'][:,0:-1]>bd_band_loc[0]) & (data_glob[iexp]['edge_nu'][:,1::]<bd_band_loc[1])
                        cond_kept_all &= cond_sel

                    #Visit fit initialization
                    if cond_exp_proc_vis:
                        bin_dic[iexp]={}
                        for key in ['Fr','varFr','Fmast_tot','nu','low_nu','high_nu']:bin_dic[iexp][key] = np.zeros(0,dtype=float)

                    #Defined and fitted orders
                    iord_def = np.intersect1d(iord_fit_list,np_where1D(np.sum(cond_kept_all,axis=1)>0))

                    #Filter
                    #    - reshaping orders into a single array so that they are binned together
                    #    - this is necessary because a single contiguous table must be passed to the filter
                    #    - spectra are not rescaled into count units so that they are comparable in order overlaps and can be integrated together within new bins 
                    if gen_dic['wig_exp_filt']['mode']:
                        adjust_bins=False
                        cond_kept_exp = np.array([cond_kept_all[iord_def,::-1].flatten()])
                        count_exp = np.array([data_glob[iexp]['flux'][iord_def,::-1].flatten()])
                        var_exp=np.zeros(0,dtype=float)
                        for iord in iord_def:var_exp = np.append(var_exp,data_glob[iexp]['cov'][iord][0][::-1])
                        var_exp = np.array([var_exp])
                        count_mast_exp = np.array([data_mast_exp['flux'][iord_def,::-1].flatten()])
                        cen_nu_exp = np.array([data_glob[iexp]['cen_nu'][iord_def,::-1].flatten()])
                        edge_nu_exp=data_glob[iexp]['edge_nu'][iord_def,::-1]
                        low_bins_nu_exp = np.array([edge_nu_exp[:,0:-1].flatten()])
                        high_bins_nu_exp = np.array([edge_nu_exp[:,1::].flatten()])
                        iord_def = [0]
                        min_nu_exp = np.min(low_bins_nu_exp)
                        max_nu_exp = np.max(high_bins_nu_exp)
 
                    #Analytical model
                    #    - scaling exposure spectrum and master back to count units
                    #    - the scaling must be applied uniformely to the exposure and master spectra (ie, after the master has been calculated) so as not introduce biases
                    else:
                        adjust_bins = True
                        cond_kept_exp = cond_kept_all[iord_def,::-1]                        
                        mean_gdet_exp = data_glob[iexp]['mean_gdet'][iord_def,::-1]
                        count_exp = data_glob[iexp]['flux'][iord_def,::-1]/mean_gdet_exp
                        var_exp = np.zeros(np.shape(count_exp),dtype=float)
                        for isub_ord,iord in enumerate(iord_def):var_exp[isub_ord] = data_glob[iexp]['cov'][iord][0][::-1]/mean_gdet_exp[isub_ord]**2. 
                        count_mast_exp = data_mast_exp['flux'][iord_def,::-1]/mean_gdet_exp    
                        cen_nu_exp=data_glob[iexp]['cen_nu'][iord_def,::-1]
                        edge_nu_exp=data_glob[iexp]['edge_nu'][iord_def,::-1]
                        low_bins_nu_exp=edge_nu_exp[:,0:-1]
                        high_bins_nu_exp=edge_nu_exp[:,1::]

                    #Grouping progressively bins of current spectrum 
                    #    - each order of 2D spectra is processed independently, but contributes to the global binned table
                    #      this is done in this way to keep track of each order, and because there is no need to bin orders together to fit the model
                    for isub_ord,iord in zip(np.arange(len(iord_def))[::-1],iord_def[::-1]):

                        #Resample spectra
                        low_bins_nu=low_bins_nu_exp[isub_ord]
                        high_bins_nu=high_bins_nu_exp[isub_ord]
                        bin_bd,raw_loc_dic = sub_def_bins(gen_dic['wig_bin'],cond_kept_exp[isub_ord],low_bins_nu,high_bins_nu,high_bins_nu-low_bins_nu,cen_nu_exp[isub_ord],count_exp[isub_ord],Mstar_loc=count_mast_exp[isub_ord],var1D_loc=var_exp[isub_ord])                       
                    
                        #Initialize tables for fit per order
                        bin_ord_dic={}
                        for key in ['Fr','varFr','Fmast_tot','nu','low_nu','high_nu']:bin_ord_dic[key] = np.zeros(0,dtype=float)
                        
                        #Process spectral bins
                        nfilled_bins=0
                        for ibin,(low_bin_loc,high_bin_loc) in enumerate(zip(bin_bd[0:-1],bin_bd[1:])):
                            bin_loc_dic,nfilled_bins = sub_calc_bins(low_bin_loc,high_bin_loc,raw_loc_dic,nfilled_bins,calc_Fr=True,adjust_bins=adjust_bins)
                    
                            #Add bin to order table
                            if len(bin_loc_dic)>0:
                                for key_loc,key in zip(['Fr','varFr','cen_bins','low_bins','high_bins'],['Fr','varFr','nu','low_nu','high_nu']):
                                    bin_ord_dic[key] = np.append( bin_ord_dic[key] , bin_loc_dic[key_loc])
                            
                            #When wiggles are filtered bins need to be contiguous and defined
                            #    - we set undefined bins to unity (so that the filter will not correct the data) and attribute a larger error of 5% 
                            elif gen_dic['wig_exp_filt']['mode']:
                                for val_loc,key in zip([1.,0.05,0.5*(low_bin_loc+high_bin_loc),low_bin_loc,high_bin_loc],['Fr','varFr','nu','low_nu','high_nu']):
                                    bin_ord_dic[key] = np.append( bin_ord_dic[key] , val_loc)
                                                                  
                        #Set binned ratio over current order to a constant level unity
                        #    - in cases the color balance is not perfectly corrected at order level
                        #      we assume the measurement of the total flux is precise enough that it does not need fitting
                        #    - this normalization assumes that the wiggle does not modify the average flux over the slice, which is not true if the slice contains few wiggle periods
                        #      however small differences in the level of the fitted flux do not bias the derived sinusoidal parameters  
                        if gen_dic['wig_norm_ord']:                            
                            delta_bin_ord = (bin_ord_dic['high_nu']-bin_ord_dic['low_nu'])
                            corr_Fr = np.sum(delta_bin_ord)/np.sum(bin_ord_dic['Fr']*delta_bin_ord)  
                            bin_ord_dic['Fr']*=corr_Fr
                            bin_ord_dic['varFr']*=corr_Fr**2.

                        #Shift transmission spectrum from common frame (receiver) to Earth (source) rest frame
                        #    - see gen_specdopshift():                            
                        # w_source = w_receiver / (1+ (rv[s/r]/c))
                        # w_Earth = w_solbar / (1+ (rv[Earth/solbar]/c))
                        #         = w_common / ((1+ (rv[Earth/solbar]/c))*(1+ (rv[solbar/common]/c)) )
                        #         = w_common / ((1+ (BERV/c))*(1- (rv[common/solbar]/c)) )
                        #    - once stellar lines have been removed by dividing exposure and master in the star rest frame, we align transmission spectra in the 
                        # frame of the telescope expected to trace more directly the wiggle
                        dopp_star2earth = 1./(gen_specdopshift(data_prop[inst][vis]['BERV'][iexp])*(1.+1.55e-8)*gen_specdopshift(-rv_al_all[iexp]))
                        bin_ord_dic['high_nu']*= dopp_star2earth
                        bin_ord_dic['low_nu']*= dopp_star2earth
                        bin_ord_dic['nu']*= dopp_star2earth

                        #Append binned ratio over current order to global table for current exposure
                        if cond_exp_proc_vis:
                            for key in ['Fr','varFr','nu','low_nu','high_nu']:
                                bin_dic[iexp][key] = np.append( bin_dic[iexp][key] , bin_ord_dic[key])  
                                    
                        #Process slice for fit per exposure if exposure is in groups
                        if cond_exp_proc and (iexp_glob_bin is not None):

                            #Initialize dictionaries storing all orders over all exposures in current group
                            n_bins_ord = len(bin_ord_dic['Fr'])
                            if (not init_ord):
                                init_ord = True
                                ibin2ord_fit = np.repeat(iord,n_bins_ord)     #slices to which binned datapoints belong to 
                                ibin2exp_fit = np.repeat(iexp,n_bins_ord)     #exposures to which binned datapoints belong to 
                                Fr_bin_fit = deepcopy(bin_ord_dic)   

                            #Append to fit table
                            else:
                                ibin2ord_fit = np.append(ibin2ord_fit,np.repeat(iord,n_bins_ord))
                                ibin2exp_fit = np.append(ibin2exp_fit,np.repeat(iexp,n_bins_ord))
                                for key in ['Fr','varFr','nu','low_nu','high_nu']:
                                    Fr_bin_fit[key] = np.append( Fr_bin_fit[key] , bin_ord_dic[key]) 

                    ### End of orders for current exposure
                    if (not gen_dic['wig_indiv_mast']):data_glob.pop(iexp)

                    #Update maximum spectral boundaries and order tables
                    if cond_exp_proc:
                        wsort = np.argsort(Fr_bin_fit['nu'])
                        for key in ['Fr','varFr','nu','low_nu','high_nu']:Fr_bin_fit[key] = Fr_bin_fit[key][wsort]                   
                    if cond_exp_proc_vis:
                        glob_min_bins = np.min([glob_min_bins,bin_dic[iexp]['low_nu'][0]])
                        glob_max_bins = np.max([glob_max_bins,bin_dic[iexp]['high_nu'][-1]])
                        wsort = np.argsort(bin_dic[iexp]['nu'])
                        for key in ['Fr','varFr','nu','low_nu','high_nu']:bin_dic[iexp][key] = bin_dic[iexp][key][wsort]                

                ### End of processing spectral ratio

                #Perform fit when all exposures in current group have been joined
                #    - if the group exposure of next exposure is larger than the group exposure of current exposure
                #      if current exposure is the last one processed, the group exposure of next exposure has been set to index 'n' and the condition is met     
                if gen_dic['wig_exp_ana'] and (iexp_glob_bin_next>iexp_glob_bin):
                    isub_group = len(iexp_group_list)-1
                  
                    #Reset group initialisation
                    init_exp = False
                    init_ord = False

                    #Store equivalent values for current grouped exposures
                    #    - calculated as a time-weighted average
                    if wig_exp_groups is not None:
                        t_dur_tot = np.sum(tel_coord_vis['t_dur'][wig_exp_group])
                        for key in ['cen_ph','az','x_az','y_az','z_alt']:tel_coord_expgroup[key][isub_group] = np.sum(tel_coord_vis['t_dur'][wig_exp_group]*tel_coord_vis[key][iexp_glob_bin])/t_dur_tot 
                        n_shift_expgroups = np.sum(tel_coord_vis['cond_shift'][iexp_glob_bin]) 
                        n_unshift_expgroups = len(iexp_glob_bin) - n_shift_expgroups
                        tel_coord_expgroup['cond_shift'][isub_group] = True if n_shift_expgroups>n_unshift_expgroups else False                        
                        if tel_coord_expgroup['cond_shift'][isub_group]:
                            tel_coord_expgroup['cond_eastmer'][isub_group] = False
                            tel_coord_expgroup['cond_westmer'][isub_group] = False
                        else:
                            n_eastmer_expgroups = np.sum(tel_coord_vis['cond_eastmer'][iexp_glob_bin]) 
                            n_westmer_expgroups = len(iexp_glob_bin) - n_eastmer_expgroups
                            tel_coord_expgroup['cond_eastmer'][isub_group] = True if n_eastmer_expgroups>n_westmer_expgroups else False
                            tel_coord_expgroup['cond_westmer'][isub_group] = ~tel_coord_expgroup['cond_eastmer'][isub_group]                                         
                    else:
                        for key in ['cen_ph','az','x_az','y_az','z_alt','cond_eastmer','cond_westmer','cond_shift']:tel_coord_expgroup[key][isub_group] = tel_coord_vis[key][iexp_glob_bin]
                                                             
                    #Exposure processing
                    if cond_exp_proc:
                        y_range = None
                        print('                Analyzing iexp[group] = ',str(iexp_glob_bin))

                        #Store wavelength information
                        min_plot=Fr_bin_fit['low_nu'][0] 
                        max_plot=Fr_bin_fit['high_nu'][-1]

                        #RMS and median error of uncorrected data
                        if gen_dic['wig_exp_fit']['mode'] | gen_dic['wig_exp_filt']['mode']:
                            rms_exp_raw[isub_group] = Fr_bin_fit['Fr'].std()
                            median_err[isub_group] = np.median(np.sqrt(Fr_bin_fit['varFr']))

                        #-------------------
    
                        #Constant fit and global periodogram
                        #    - run first, to identify ranges/orders to exclude and calculate RMS
                        if gen_dic['wig_exp_init']['mode']:
                            
                            #Fit transmission spectra with constant level unity
                            p_start = Parameters()
                            p_start.add_many(('level',1., True & False  , None , None , None)) 

                            #Fitting
                            fit_func_wig = wig_mod_cst
                            fixed_args['idx_fit'] = np.ones(len(Fr_bin_fit['nu']),dtype=bool)
                            _,merit ,p_ord_best = call_lmfit(p_start,Fr_bin_fit['nu'],Fr_bin_fit['Fr'],np.array([Fr_bin_fit['varFr']]),fit_func_wig,verbose=False,fixed_args=fixed_args)
    
                            #Save results of sampling fit for current exposure
                            fit_results={'RMS':merit['rms']}
                            np.savez_compressed(path_dic['datapath_Init']+'/Fit_results_iexpGroup'+str(iexp_glob_bin),data=fit_results,allow_pickle=True)                         

                            #Cumulate periodogram from current exposure
                            #    - over global search range, and over the windows of typical components
                            if gen_dic['wig_exp_init']['plot_hist']:
                                if fixed_args['lb_with_err']:ls = LombScargle(Fr_bin_fit['nu'],Fr_bin_fit['Fr'],np.sqrt(Fr_bin_fit['varFr']))
                                else:ls = LombScargle(Fr_bin_fit['nu'],Fr_bin_fit['Fr'])
                                for comp_id in comm_freq_comp:
                                    power_comp = ls.power(comm_freq_comp[comp_id])
                                    comm_power_comp[comp_id]+=power_comp

                            #Plot fit
                            #    - this is done internally to the function to avoid saving all ratios and fits for the plot routine
                            if gen_dic['wig_exp_init']['plot_spec']:
                                plot_screening(ibin2exp_fit,ibin2ord_fit,min_plot,max_plot,gen_dic,Fr_bin_fit,merit['rms'],path_dic['plotpath_Init']+'ExpGroup'+str(iexp_glob_bin),fixed_args)
                         
                        #-------------------
                        #Filter
                        if gen_dic['wig_exp_filt']['mode']:                                 

                            #Smoothing window
                            dnu_tab = Fr_bin_fit['high_nu']-Fr_bin_fit['low_nu']
                            dnu_med = np.median(dnu_tab)
                            n_smooth_win = int(gen_dic['wig_exp_filt']['win']/dnu_med)
                            if not is_odd(n_smooth_win):n_smooth_win+=1
                            
                            #Savitzky-Golay filtering
                            filtered_flux = savgol_filter(Fr_bin_fit['Fr'], n_smooth_win,gen_dic['wig_exp_filt']['deg'])  

                            #Save filtered spectrum as a function
                            #    - filling in the edges with unity values to prevent sharp variations
                            nlow = int((Fr_bin_fit['nu'][0]-min_nu_exp)/dnu_tab[0])+2
                            nhigh = int((max_nu_exp-Fr_bin_fit['nu'][-1])/dnu_tab[-1])+2
                            nu_interp = np.concatenate((Fr_bin_fit['nu'][0] - dnu_tab[0]*np.arange(1,nlow+1)[::-1],Fr_bin_fit['nu'],Fr_bin_fit['nu'][-1] + dnu_tab[-1]*np.arange(1,nhigh+1)))
                            flux_interp = np.concatenate((np.ones(nlow),filtered_flux,np.ones(nhigh)))
                            dic_filter[iexp_glob_bin] = CubicSpline(nu_interp,flux_interp)

                            #Residuals between binned spectrum and filtered spectrum
                            flux_res = 1. + Fr_bin_fit['Fr'] - filtered_flux
                
                            #RMS and median error of corrected data
                            rms_exp_fit[isub_group] = flux_res.std()

                            #------------------------------        
                            #Plot 
                            if gen_dic['wig_exp_filt']['plot']: 
                                fixed_args_loc = deepcopy(fixed_args) 
                                min_nuplot=Fr_bin_fit['nu'][0]-0.3
                                max_nuplot=Fr_bin_fit['nu'][-1]+0.3
                                min_max_plot = [min_nuplot,max_nuplot]
                                
                                #Periodogram in log mode
                                fixed_args_loc['perio_log'] = True & False

                                #Model
                                n_nu,nu_plot = def_wig_tab(min_nuplot,max_nuplot,fixed_args_loc['dnu_HR'])
                                mod_plot_glob={'all':dic_filter[iexp_glob_bin](nu_plot)}

                                #Plot
                                plot_wig_glob(Fr_bin_fit['low_nu'],Fr_bin_fit['high_nu'],Fr_bin_fit['nu'],Fr_bin_fit['Fr'],np.sqrt(Fr_bin_fit['varFr']),nu_plot,mod_plot_glob,Fr_bin_fit['low_nu'],Fr_bin_fit['high_nu'],Fr_bin_fit['nu'],flux_res,np.sqrt(Fr_bin_fit['varFr']),fixed_args_loc,min_max_plot,'_',path_dic['plotpath_Filter']+'ExpGroup'+str(iexp_glob_bin),bin_spec=True,filter_mode=True)

                        #-------------------
                        #Periodogram sampling
                        if gen_dic['wig_exp_samp']['mode']: 
                            fit_dic['run_name'] = '_'+inst+'_'+vis+'_'+str(iexp_glob_bin)
                            fixed_args_loc = deepcopy(fixed_args) 
                            fixed_args_loc['nsamp'] = gen_dic['wig_exp_samp']['nsamp'] 
                            fixed_args_loc['idx_fit'] = np.ones(len(Fr_bin_fit['nu']),dtype=bool)
                                            
                            #Fit iterations
                            #    - the minimization fails to converge in a single run, possibly stopping after considering that the improvement in fit quality is sufficient enough (decreasing the fit tolerance does not help)
                            #      running successive iterations that start from the previous fit allows the minimization to eventually converge to a truly good fit                 
                            fixed_args_loc['nit'] = gen_dic['wig_exp_samp']['nit']           
            
                            #Fix frequency of components to best periodogram frequency or let it be fitted with sine function
                            fix_freq2perio = { comp_id:True for comp_id in range(1,6)}
    
                            #Fit component only if FAP below threshold (in %)
                            fixed_args_loc['fap_thresh'] = gen_dic['wig_exp_samp']['fap_thresh']     
     
                            #Possibility to modify guesses and priors before local exposure fit
                            fit_prop_dic['mod_prop'] = deepcopy(mod_prop_exp)
                            fit_prop_dic['varpar_priors'] = deepcopy(varpar_priors_exp)
                            fixed_args_loc['stable_pointpar'] = stable_pointpar_exp
    
                            #Fix all properties by default
                            for par in fit_prop_dic['mod_prop']:fit_prop_dic['mod_prop'][par]['vary'] = False 

                            #Fix component frequency using the fit results from 'wig_exp_point_ana' or 'wig_vis_fit'
                            comp_freqfixed = []
                            if (len(gen_dic['wig_exp_samp']['fix_freq2expmod'])>0) or (vis in gen_dic['wig_exp_samp']['fix_freq2vismod']):                             
                                if len(gen_dic['wig_exp_samp']['fix_freq2expmod'])>0:     
                                    comp_freqfixed = gen_dic['wig_exp_samp']['fix_freq2expmod']
                                    coord_fit = np.load(path_dic['datapath_Coord']+'/Fit_results.npz',allow_pickle=True)['data'].item()                                
                                if (vis in gen_dic['wig_exp_samp']['fix_freq2vismod']):    
                                    comp_freqfixed = gen_dic['wig_exp_samp']['fix_freq2vismod']['comps']                
                                    coord_fit = (np.load(gen_dic['wig_exp_samp']['fix_freq2vismod'][vis],allow_pickle=True)['data'].item())['p_best']

                                #Exposure properties
                                args_loc={**fixed_args_loc,**{key:np.array([tel_coord_expgroup[key][isub_group]]) for key in tel_coord_expgroup}}                                 
                                
                                #Component frequency
                                for comp_id in comp_freqfixed:
                                    for ideg in range(fixed_args_loc['deg_Freq'][comp_id]+1):
                                        par = 'Freq'+str(comp_id)+'_c0'
                                      
                                        #Calculate parameter value in current exposure
                                        args_loc['par_name'] = par+'_'
                                        if len(gen_dic['wig_exp_samp']['fix_freq2expmod'])>0: 
                                            params_par = {par+sub:coord_fit[par+sub][0] for sub in suf_hyper_vis}
                                        if (vis in gen_dic['wig_exp_samp']['fix_freq2vismod']): 
                                            params_par = {par+sub:coord_fit[par+sub].value for sub in suf_hyper_vis}
                                        fit_prop_dic['mod_prop'][par+'_off']['guess'] = wig_submod_coord_discont(1,params_par,args_loc)[0]

                            #Model parameters
                            p_start = Parameters()  
                            par_formatting(p_start,fit_prop_dic['mod_prop'],fit_prop_dic['varpar_priors'],fit_dic,fixed_args_loc,'','',None)
                            init_fit(fit_dic,fixed_args_loc,p_start,fit_prop_dic,model_par_names,model_par_units)                    

                            #Initialize band fit results
                            #    - we only store constant coefficients of the amplitude and frequency hyper-parameters, for the highest fitted component
                            #    - set main amplitude and frequency coefficients as constant fit properties
                            fit_results = {'par_list' : [] }   
                            for par in ['AmpGlob'+str(fixed_args_loc['comp_id_max'])+'_c0_off','Freq'+str(fixed_args_loc['comp_id_max'])+'_c0_off']:
                                if not (('Freq' in par) and (fixed_args_loc['comp_id_max'] in comp_freqfixed)):
                                    fit_results['par_list']+=[par] 
                                    fit_results[par] = {'nu':np.zeros(0,dtype=float),'val':np.zeros([2,0],dtype=float)}                            
                                
                            #Tables
                            samp_fit_dic = {'nu':{},'flux':{},'var':{},'err':{},'res':{},'nu_all':{},'flux_all':{},'var_all':{},'res_all':{} }
                            nu_amp = {}
                            count = {1:0,2:0,3:0,4:0,5:0}
                            ishift_comp = {}
                            fixed_args_loc['perio_log'] = False
    
                            #Initialize global model for plotting  
                            path_sampling_plot = path_dic['plotpath_Sampling']+'/Comp'+str(fixed_args_loc['comp_id_max'])+'/ExpGroup'+str(iexp_glob_bin)+'/'
                            if not os_system.path.exists(path_sampling_plot):os_system.makedirs(path_sampling_plot)          
                            fixed_args_loc['n_nu_HR'],fixed_args_loc['nu_plot_glob'] = def_wig_tab(Fr_bin_fit['nu'][0],Fr_bin_fit['nu'][-1],fixed_args_loc['dnu_HR'])   
                            samp_fit_dic['mod_plot_glob'] = {comp_id:np.ones(fixed_args_loc['n_nu_HR']) for comp_id in fixed_args_loc['comp_ids'] }              

                            #Oversampling of sampling bands (in 1e-10 s-1) 
                            sampbands_shifts = deepcopy(gen_dic['wig_exp_samp']['sampbands_shifts'])

                            #Process a single shift for lower component than the last one
                            for comp_id in gen_dic['wig_exp_samp']['comp_ids']:
                                if (fixed_args_loc['comp_id_max']>comp_id):sampbands_shifts[comp_id] = [gen_dic['wig_exp_samp']['direct_samp'][comp_id+1]]
                            
                            #Sampling component 1  
                            comp_id=1
                            comp_id_proc = []
                            for ishift,shift_off in enumerate(sampbands_shifts[1]):
                                wig_perio_sampling(comp_id_proc,gen_dic['wig_exp_samp']['plot'],samp_fit_dic,shift_off,ishift_comp,ishift,comp_id,Fr_bin_fit['nu'],Fr_bin_fit['Fr'],Fr_bin_fit['varFr'],count,fixed_args_loc,p_start,nu_amp,gen_dic['wig_exp_samp']['src_perio'][comp_id],comp_freqfixed,pref_names,suf_names,fix_freq2perio,fit_results,path_sampling_plot,freq_params_samp,fixed_args)
                            comp_id_proc += [1]
                            
                            #Sampling higher components
                            for comp_id in [comp_id_loc for comp_id_loc in gen_dic['wig_exp_samp']['comp_ids'] if comp_id_loc!=1]:
                                for ishift,shift_off in enumerate(sampbands_shifts[comp_id]): 
                                    wig_perio_sampling(comp_id_proc,gen_dic['wig_exp_samp']['plot'],samp_fit_dic,shift_off,ishift_comp,ishift,comp_id,samp_fit_dic['nu_all'][comp_id_proc[-1]],samp_fit_dic['res_all'][comp_id_proc[-1]],samp_fit_dic['var_all'][comp_id_proc[-1]],count,fixed_args_loc,p_start,nu_amp,gen_dic['wig_exp_samp']['src_perio'][comp_id],comp_freqfixed,pref_names,suf_names,fix_freq2perio,fit_results,path_sampling_plot,freq_params_samp,fixed_args)
                                
                                #Store last processed component
                                comp_id_proc+=[comp_id]
              
                            #Save results for highest component
                            np.savez_compressed(path_dic['datapath_Sampling']+'/Comp'+str(fixed_args_loc['comp_id_max'])+'/Fit_results_iexpGroup'+str(iexp_glob_bin),data=fit_results,allow_pickle=True)                         

                        #------------------------------------------------------------------------------------
                        #Fit with full model
                        if gen_dic['wig_exp_fit']['mode']: 
                            fixed_args_loc = deepcopy(fixed_args) 
                
                            #Amplitude model fixed
                            fixed_args_loc['fixed_amp'] = {1:False,2:False,3:False,4:False,5:False}
                            
                            #Decreasing frequencies
                            #    - comp_id set to True imposes that Freq[comp_id]_c0 <  Freq[comp_id_proc]_c0
                            #    - amplitudes of components are not necessarily decreasing between components
                            #    - remove if priors are more constraining
                            fixed_args_loc['cond_dec_f'] = {2:False,3:False,4:False,5:False}
                          
                            #Frequency model fixed
                            fixed_args_loc['fixed_freq'] = {1:False,2:False,3:False,4:False,5:False}    
         
                            #---------------------------
                            
                            #Components to include
                            fixed_args_loc['comp_mod']=deepcopy(gen_dic['wig_exp_fit']['comp_ids'])
                            
                            #Possibility to modify guesses and priors before local exposure fit
                            fit_prop_dic['mod_prop'] = deepcopy(mod_prop_exp)
                            fit_prop_dic['varpar_priors'] = deepcopy(varpar_priors_exp)
                            fixed_args_loc['stable_pointpar'] = deepcopy(stable_pointpar_exp)

                            #Initialize coefficients of hyper-parameters by their chromatic fits to initialize the global fit
                            #    - default initialization unless chromatic fit was not run
                            #    - taking the closest exposure if current one was not processed
                            if init_chrom:
                                exp_path = path_dic['datapath_Chrom']+'/Fit_results_iexpGroup'+str(iexp_glob_bin)+'.npz'
                                if os_system.path.exists(exp_path):
                                    iexp_init = iexp_glob_bin
                                    data_path = exp_path
                                else:
                                    isub_low = iexp_glob_bin - np.arange(1,iexp_glob_bin+1)
                                    isub_high = iexp_glob_bin + np.arange(1,nexp_fit_list-1-iexp_glob_bin+1)
                                    isub_alternate = [x for x in itertools.chain.from_iterable(itertools.zip_longest(isub_low,isub_high)) if x is not None]
                                    for isub in isub_alternate:
                                        exp_path = path_dic['datapath_Chrom']+'/Fit_results_iexpGroup'+str(isub)+'.npz'
                                        if os_system.path.exists(exp_path):
                                            data_path = exp_path
                                            break
                                    iexp_init = isub
                                print('                  Init. from chromatic exp '+str(iexp_init))
                                hyperpar_chrom_fit = np.load(data_path,allow_pickle=True)
                                hyperpar_chrom_results = hyperpar_chrom_fit['data'].item()   
                                hyperpar_chrom_args = hyperpar_chrom_fit['args'].item()  
                                for comp_id in gen_dic['wig_exp_fit']['comp_ids']:
                                    if hyperpar_chrom_args['deg_Freq'][comp_id]!=fixed_args_loc['deg_Freq'][comp_id]:stop('Run chromatic analysis with same Freq degree')
                                    if hyperpar_chrom_args['deg_Amp'][comp_id]!=fixed_args_loc['deg_Amp'][comp_id]:stop('Run chromatic analysis with same Amp degree')
                                for par in hyperpar_chrom_results:
                                    if par in fit_prop_dic['mod_prop']:fit_prop_dic['mod_prop'][par]['guess'] = hyperpar_chrom_results[par][0]

                            #Fix or bound properties using results from temporal hyperparameter fit
                            if (vis in gen_dic['wig_exp_fit']['model_par']):model_par = list(gen_dic['wig_exp_fit']['model_par'][vis].keys())
                            else:model_par = {}
                            if (vis in gen_dic['wig_exp_fit']['fixed_pointpar']):fixed_pointpar = gen_dic['wig_exp_fit']['fixed_pointpar'][vis]
                            else:fixed_pointpar = []
                            if (len(model_par)>0) or (len(fixed_pointpar)>0):
                            
                                #Retrieve hyperparameter model                            
                                hyperpar_coord_fit = np.load(path_dic['datapath_Coord']+'/Fit_results.npz',allow_pickle=True)['data'].item()   

                                #Exposure properties
                                args_loc={**fixed_args_loc,**{key:np.array([tel_coord_expgroup[key][isub_group]]) for key in tel_coord_expgroup}}   
                                
                                #Process requested properties
                                for par in np.unique(model_par+fixed_pointpar):
                                    if par+'_off' not in fit_prop_dic['mod_prop']:stop('Undefined parameter'+par)
                                    
                                    #Calculate parameter value in current exposure
                                    cond_hyperfit = True
                                    for sub in suf_hyper_vis:cond_hyperfit&=(par+sub in hyperpar_coord_fit)
                                    if not cond_hyperfit:print('                    ',par,' not in coordinate fit')
                                    else:
                                        params_par = {par+sub:hyperpar_coord_fit[par+sub][0] for sub in suf_hyper_vis}   
                                        args_loc['par_name'] = par+'_'                              
                                        par_val = wig_submod_coord_discont(1,params_par,args_loc)[0]
                                        fit_prop_dic['mod_prop'][par+'_off']['guess'] = par_val
                                    
                                        #Set prior
                                        if par in model_par:
                                            prior_range = gen_dic['wig_exp_fit']['model_par'][vis][par]   
                                            fit_prop_dic['varpar_priors'][par+'_off']['low'] = par_val  - prior_range[0] 
                                            fit_prop_dic['varpar_priors'][par+'_off']['high'] = par_val + prior_range[1]   
                                
                                        #Fix property
                                        if par in fixed_pointpar:
                                            fit_prop_dic['mod_prop'][par+'_off']['vary'] = False
    
                            #Change variable status
                            #    - provides additional flexibility in addition to 'comp_mod'
                            # fit_prop_dic['mod_prop']['Freq3_c1_off']['vary'] = False
                            # fit_prop_dic['mod_prop']['Freq3_c2_off']['vary'] = False   

                            #Define each model component
                            for comp_id in range(1,6):     
                                comp_str = str(comp_id)   
                    
                                #Nullify and fix component
                                if comp_id not in gen_dic['wig_exp_fit']['comp_ids']:
                                    for pref,suf in zip(pref_names[comp_id],suf_names[comp_id]):
                                        fit_prop_dic['mod_prop'].pop(pref+comp_str+suf+'_off')

                                else:    
                                    
                                    #Amplitude model fixed
                                    if fixed_args_loc['fixed_amp'][comp_id]:
                                        for pref,suf in zip(pref_names_amp[comp_id],suf_names_amp[comp_id]):fit_prop_dic['mod_prop'][pref+comp_str+suf+'_off']['vary']  = False                 

                                    #Frequency model fixed
                                    if fixed_args_loc['fixed_freq'][comp_id]:
                                        for pref,suf in zip(pref_names_freq[comp_id],suf_names_freq[comp_id]):fit_prop_dic['mod_prop'][pref+comp_str+suf+'_off']['vary']  = False                                          

                                    #Decreasing frequencies
                                    #    - we assume the frequencies do not vary with nu so much that their curve overlap, and we thus use the zero-th order coefficient of each component frequency as constraint
                                    elif (comp_id>1) and fixed_args_loc['cond_dec_f'][comp_id]:
                                        delta_freq ='delta_f'+str(comp_id-1)+comp_str 
                                        fit_prop_dic['mod_prop'][delta_freq]={'guess':p_start['Freq'+str(comp_id-1)+'_c0_off'].value - p_start['Freq'+comp_str+'_c0_off'].value,'vary':True}
                                        fit_prop_dic['varpar_priors'][delta_freq]={'low':0.,'high':5.}         
                                        fit_prop_dic['mod_prop'].pop('Freq'+comp_str+'_c0_off')
                                        fit_prop_dic['mod_prop']['Freq'+comp_str+'_c0_off']={'guess':np.nan,'vary':False,'expr':'Freq'+str(comp_id-1)+'_c0_off - '+delta_freq}
                                        fit_prop_dic['varpar_priors']['Freq'+comp_str+'_c0_off']={'low':0.,'high':5.} 
                       
                            #Model parameters
                            #    - initialized with generic parameter values
                            p_start = Parameters()  
                            par_formatting(p_start,fit_prop_dic['mod_prop'],fit_prop_dic['varpar_priors'],fit_dic,fixed_args_loc,'','',None)
                            init_fit(fit_dic,fixed_args_loc,p_start,fit_prop_dic,model_par_names,model_par_units)     

                            #------------------------------
    
                            #Fit model to data
                            p_best = p_start
                            if gen_dic['wig_exp_fit']['use']:
                      
                                #Run fit over several iteration to converge, using the robust 'nelder' method
                                fixed_args_loc['idx_fit'] = np.ones(len(Fr_bin_fit['nu']),dtype=bool)
                                for it in range(gen_dic['wig_exp_fit']['nit']):
                                    _,_ ,p_best = call_lmfit(p_best,Fr_bin_fit['nu'],Fr_bin_fit['Fr'],np.array([Fr_bin_fit['varFr']]),FIT_calc_wig_mod_nu,verbose=False ,fixed_args=fixed_args_loc,maxfev = fixed_args_loc['max_nfev'],method=gen_dic['wig_exp_fit']['fit_method'])  

                                #Determine uncertainties by running LM fit using Nelder-Mead solution as starting point
                                _,_ ,p_best = call_lmfit(p_best,Fr_bin_fit['nu'],Fr_bin_fit['Fr'],np.array([Fr_bin_fit['varFr']]),FIT_calc_wig_mod_nu,verbose=False ,fixed_args=fixed_args_loc,maxfev = fixed_args_loc['max_nfev'],method='leastsq')

                                #Store properties
                                globexpfit_results = {}
                                for subpar in p_best:globexpfit_results[subpar] = [p_best[subpar].value,p_best[subpar].stderr,p_best[subpar].vary]       
                                np.savez_compressed(path_dic['datapath_Global']+'/Fit_results_iexpGroup'+str(iexp_glob_bin),data=globexpfit_results,allow_pickle=True)            

                            #Retrieve results from existing fit
                            else:
                                globexpfit_results = dataload_npz(path_dic['datapath_Global']+'/Fit_results_iexpGroup'+str(iexp_glob_bin))        
                                for subpar in globexpfit_results:
                                    p_best[subpar].value = globexpfit_results[subpar][0] 
                                    
                            #Residuals between fitted spectrum and best-fit model
                            flux_res = 1. + Fr_bin_fit['Fr'] - calc_wig_mod_nu(Fr_bin_fit['nu'],p_best,fixed_args_loc)[0] 
                
                            #RMS and median error of corrected data
                            rms_exp_fit[isub_group] = flux_res.std()
    
                            #------------------------------        
                            #Plot 
                            if gen_dic['wig_exp_fit']['plot']:             
                                min_nuplot=Fr_bin_fit['nu'][0]-0.3
                                max_nuplot=Fr_bin_fit['nu'][-1]+0.3
                                min_max_plot = [min_nuplot,max_nuplot]
                                
                                #Periodogram in log mode
                                fixed_args_loc['perio_log'] = True & False

                                #Model
                                n_nu,nu_plot = def_wig_tab(min_nuplot,max_nuplot,fixed_args_loc['dnu_HR'])
                                best_mod_HR,comp_mod_HR = calc_wig_mod_nu(nu_plot,p_best,fixed_args_loc) 
                                mod_plot_glob={'all':best_mod_HR}
                                for comp_id in comp_mod_HR:mod_plot_glob[comp_id]=1.+comp_mod_HR[comp_id]
    
                                #Plot
                                plot_wig_glob(None,None,Fr_bin_fit['nu'],Fr_bin_fit['Fr'],np.sqrt(Fr_bin_fit['varFr']),nu_plot,mod_plot_glob,None,None,Fr_bin_fit['nu'],flux_res,np.sqrt(Fr_bin_fit['varFr']),fixed_args_loc,min_max_plot,'_',path_dic['plotpath_Global']+'ExpGroup'+str(iexp_glob_bin))

                    ### end of processing for current exposure
           
                    #-----------------------------------------------------------                                  
                        
                    #Automatic fit of sampled properties 
                    if gen_dic['wig_exp_nu_ana']['mode']: 

                        #Number of fit iterations
                        nit_hyper=4
  
                        #Custom fit ranges
                        fit_range_dic={
                            # 'Freq1_c0_off':[[0.,0.00021]], 
                            # 'AmpGlob1_maxAmp_off':[[0.,0.00021]],  
                            # 'Freq2_c0_off':[[0.000127,0.000175]], 
                            # 'AmpGlob3_maxAmp_off':[[0.000131,0.00021]], 
                            # 'Freq3_c0_off':[[0.0001305,0.000208]], 
                            }    
                        
                        #Force axis ranges
                        x_range_var={
                            'AmpGlob1_c0_off':[37.5,59.5],  #HD209 ANTARESS I
                            'Freq1_c0_off':[37.5,59.5],
                            'AmpGlob2_c0_off':[37.5,59.5],
                            'Freq2_c0_off':[37.5,59.5],
                            # 'AmpGlob1_c0_off':[37.,64.],  #WASP76
                            # 'Freq1_c0_off':[37.,64.],  #WASP76
                            # 'AmpGlob1_maxAmp_off':[37.,76.],  #HD29291
                            # 'Freq1_c0_off':[37.,76.],   #HD29291
                            # 'AmpGlob2_c0_off':[37.,60.],    #WASP76
                            # 'Freq2_c0_off':[37.,60.],     #WASP76
                            # 'Freq2_c0_off':[37.,76.],   #HD29291
                            }
                        x_range_res={}
                        y_range_var={
                            'AmpGlob1_c0_off':[1.5e-4,2.6e-3],  #HD209 ANTARESS I
                            'Freq1_c0_off':[3.6,3.85],
                            'AmpGlob2_c0_off':[1e-4,8.5e-4],
                            'Freq2_c0_off':[1.9,2.4],                            
                            # 'AmpGlob1_c0_off':[0,0.006],
                            # 'Freq1_c0_off':[3.2,4.2],  #WASP76
                            # 'AmpGlob1_maxAmp_off':[0,0.01],  #HD29291
                            # 'Freq1_c0_off':[3.4,4.3],    #HD29291
                            # 'AmpGlob2_c0_off':[0,0.0025],
                            # 'Freq2_c0_off':[1.7,2.5],
                            # 'Freq2_c0_off':[0.,3.],   #HD29291

                        #     'AmpGlob3_maxAmp_off':[0.,0.00070],
                        #     'Freq3_c0_off':[0.4,1.5],
                        #     'AmpGlob4_maxAmp_off':[0.0001,0.0004],
                        #     'Freq4_c0_off':[0.25,0.5],
                            }
                        y_range_res={
                            'AmpGlob1_c0_off':[-5e-4,5e-4],  #HD209 ANTARESS I
                            'Freq1_c0_off':[-0.15,0.15],
                            'AmpGlob2_c0_off':[-2.5e-4,2.5e-4],
                            'Freq2_c0_off':[-0.2,0.2], 
                        #     'AmpGlob3_maxAmp_off':[-0.0002,0.0002],
                        #     'Freq3_c0_off':[-0.5,0.5], 
                        #     'AmpGlob4_maxAmp_off':[-0.0001,0.0001],
                        #     'Freq4_c0_off':[-0.1,0.1], 
                            } 

                        #-------------------------------------------------------------------------------------
                        hyperpar_chrom_fit = {}

                        #Fit chromatic dependence of constant coefficients for amplitude and frequency
                        #    - for properties fitted via band analysis
                        file_save = open(path_dic['datapath_Chrom']+'/Fit_results_iexpGroup'+str(iexp_glob_bin),'w+')
                        for comp_id in gen_dic['wig_exp_nu_ana']['comp_ids']:
                            fixed_args['comp_id'] = comp_id
                            fixed_args['comp_str'] = str(comp_id)
                            fit_results = np.load(path_dic['datapath_Sampling']+'/Comp'+fixed_args['comp_str']+'/Fit_results_iexpGroup'+str(iexp_glob_bin)+'.npz',allow_pickle=True)['data'].item()   

                            for par in fit_results['par_list']:
                   
                                #Property
                                nu_samp = fit_results[par]['nu']
                                if len(nu_samp)>0:
                                    prop_samp = fit_results[par]['val'][0]
                                    eprop_samp = fit_results[par]['val'][1]   
                                    wsort = np.argsort(nu_samp)
                                    nu_samp = nu_samp[wsort].astype(float)
                                    prop_samp = prop_samp[wsort].astype(float)
                                    eprop_samp = eprop_samp[wsort].astype(float)
                             
                                    #Fit parameters        
                                    params_fit = Parameters()
        
                                    #Wiggle frequencies
                                    if 'Freq' in par:
                                        fit_func = FIT_wig_freq_nu  
                                        params_fit.add_many(('Freq'+fixed_args['comp_str']+'_c0_off',1., True  , None, None))
                                        for ideg in range(1,3): 
                                            if fixed_args['deg_Freq'][comp_id]>=ideg:params_fit.add_many(('Freq'+fixed_args['comp_str']+'_c'+str(ideg)+'_off',0., True  , None, None))

                                    #Wiggle amplitudes                    
                                    elif 'Amp' in par:
                                        fit_func = FIT_wig_amp_nu_poly
                                        params_fit.add_many(('AmpGlob'+fixed_args['comp_str']+'_c0_off',1e-3, True      , None , None, None))
                                        for ideg in range(1,5): 
                                            if fixed_args['deg_Amp'][comp_id]>=ideg:params_fit.add_many(('AmpGlob'+fixed_args['comp_str']+'_c'+str(ideg)+'_off',0., True  , None, None))

                                        #Amplitude can be negative due to the phase offset
                                        prop_samp[prop_samp<0.]*=-1
        
                                    #--------------------------------
    
                                    #Fitted values                               
                                    if par in fit_range_dic:
                                        cond_fit_all = np.repeat(False,len(nu_samp))
                                        for bd_band in fit_range_dic[par]:cond_fit_all|=(nu_samp>bd_band[0]) & (nu_samp<bd_band[1])                       
                                    else:cond_fit_all = np.repeat(True,len(nu_samp))
                                    cond_fit_all &= ~np.isnan(prop_samp)
        
                                    #Remove extreme outliers
                                    med_prop = np.median(prop_samp[cond_fit_all])
                                    res = prop_samp[cond_fit_all] - med_prop
                                    disp_est = stats.median_abs_deviation(res)
                                    cond_fit_all[(prop_samp>med_prop + 10.*disp_est) | (prop_samp<med_prop-10.*disp_est)] = False               
        
                                    #Attributing constant errors if null or undefined
                                    nan_err = (True in np.isnan(eprop_samp[cond_fit_all])) | (np.nanmax(eprop_samp[cond_fit_all])==0.)
                                    if nan_err:var_fit = np.array([np.repeat(np.median(prop_samp[cond_fit_all]),np.sum(cond_fit_all))])
                                    else:var_fit = np.array([eprop_samp[cond_fit_all]])**2.
                         
                                    #Fit
                                    p_best = params_fit   
                                    fit_prop = False
                                    fixed_args['idx_fit'] = np.ones(np.sum(cond_fit_all),dtype=bool)
                                    if np.sum(cond_fit_all)>6:   
                                        fit_prop = True
                                        _,_,p_best = call_lmfit(p_best,nu_samp[cond_fit_all],prop_samp[cond_fit_all],var_fit,fit_func,verbose=False ,fixed_args=fixed_args)
                                          
                                        #Successive fits with automatic identification and exclusion of outliers
                                        for it_res in range(nit_hyper):
                                            
                                            #Model from previous fit iteration
                                            mod = fit_func(p_best,nu_samp,args = fixed_args)
                                            
                                            #Residuals
                                            res = prop_samp - mod
                                            
                                            #Sigma-clipping
                                            if  (gen_dic['wig_exp_nu_ana']['thresh'] is not None): 
                                                disp_est = np.std(res[cond_fit_all])
                                                cond_fit_all[(res>gen_dic['wig_exp_nu_ana']['thresh']*disp_est) | (res<-gen_dic['wig_exp_nu_ana']['thresh']*disp_est)] = False
                                 
                                            #Variance based on cleaned residuals
                                            if nan_err:var_fit = np.array([np.repeat(np.std(res[cond_fit_all]),np.sum(cond_fit_all))])
                                            else:var_fit = np.array([eprop_samp[cond_fit_all]])**2.
                                    
                                            #Fit for current iteration
                                            if (len(nu_samp[cond_fit_all])>5) and (np.max(np.abs(res[cond_fit_all]))>0):
                                                if nan_err:var_fit = np.array([np.repeat(np.std(res[cond_fit_all]),np.sum(cond_fit_all))])                                                  
                                                else:var_fit = np.array([eprop_samp[cond_fit_all]])**2. 
                                                fixed_args['idx_fit'] = np.ones(np.sum(cond_fit_all),dtype=bool)
                                                _,merit,p_best = call_lmfit(p_best,nu_samp[cond_fit_all],prop_samp[cond_fit_all],var_fit,fit_func,verbose=False ,fixed_args=fixed_args)
                                  
                                        #Save results of current hyperparameter fit
                                        #    - only for variable properties, so that global fits remain initialized to generic parameter values if not fitted here
                                        np.savetxt(file_save,[['Parameter '+par]],fmt=['%s'])
                                        for subpar in p_best:
                                            if p_best[subpar].vary:
                                                hyperpar_chrom_fit[subpar] = [p_best[subpar].value,p_best[subpar].stderr]       
                                                np.savetxt(file_save,[['  '+subpar+' = '+str(p_best[subpar].value)+'+-'+str(p_best[subpar].stderr)]],fmt=['%s']) 
                                            else:
                                                np.savetxt(file_save,[['  '+subpar+' = '+str(p_best[subpar].value)+' (fixed)']],fmt=['%s']) 
                            
                            
                                    #--------------------------------
                                    #Plotting
                                    if gen_dic['wig_exp_nu_ana']['plot']: 
                                        path_hyper_plot = path_dic['plotpath_Chrom']+'ExpGroup'+str(iexp_glob_bin)+'/'
                                        if not os_system.path.exists(path_hyper_plot):os_system.makedirs(path_hyper_plot)  
                                        plt.ioff()        
                                        fig, ax = plt.subplots(2, 1, figsize=(20, 10),gridspec_kw = {'height_ratios':[70.,30.]})
                                        fontsize = 35
                                        
                                        #Plot data
                                        x_plot = nu_samp
                                        x_range_plot = [nu_samp[0]-0.3,nu_samp[-1]+0.3]
                                        ax[1].set_xlabel(r'$\nu$ (10$^{-10}$ s$^{-1}$)',fontsize=fontsize) 
                                        if 'Amp' in par:sc_fact = 1e3
                                        else:sc_fact = 1.
                                        ax[0].plot(x_plot,sc_fact*prop_samp,marker='o',linestyle='',color='dodgerblue',zorder=0)                                    
                                        
                                        #Plot fitted points
                                        if fit_prop:ax[0].plot(x_plot[cond_fit_all],sc_fact*prop_samp[cond_fit_all],marker='o',linestyle='',color='red',zorder=1)
                            
                                        #Plot best fit / fixed  model
                                        n_nu_HR,nu_HR = def_wig_tab(nu_samp[0],nu_samp[-1],fixed_args['dnu_HR'])        
                                        best_mod_HR = fit_func(p_best,nu_HR,args = fixed_args)
                                        ax[0].plot(nu_HR,sc_fact*best_mod_HR,linestyle='-',color='black',zorder=2)
                                    
                                        #Frame
                                        if par in x_range_var:x_range_plot = x_range_var[par]        
                                        if par in y_range_var:y_range_plot = sc_fact*np.array(y_range_var[par])
                                        else:
                                            dy_range_plot = max(prop_samp)-min(prop_samp)
                                            y_range_plot = [ min(prop_samp) - 0.05*dy_range_plot , max(prop_samp) + 0.05*dy_range_plot ]  
                                        ax[0].set_xlim(x_range_plot)  
                                        ax[0].set_ylim(y_range_plot)
                                        if 'Amp' in par:ytitle = r'Amplitude Comp.'+str(comp_id)+' (x 10$^{3}$)'
                                        elif 'Freq' in par:ytitle=r'Frequency Comp.'+str(comp_id)+' (10$^{10}$ s)'
                                        else:ytitle = par
                                        ax[0].set_ylabel(ytitle,fontsize=fontsize) 
                                        ax[0].tick_params('x',labelsize=fontsize)
                                        ax[0].tick_params('y',labelsize=fontsize)
                                        for axis_side in ['bottom','top','left','right']:ax[0].spines[axis_side].set_linewidth(1.5)
                                        
                                        #------------------------------
                                    
                                        #Residuals
                                        best_mod = fit_func(p_best,nu_samp,args = fixed_args)             
                                        res_prop = sc_fact*(prop_samp - best_mod)   
                                        ax[1].plot(x_plot,res_prop,marker='o',linestyle='',color='dodgerblue',zorder=0)                
                                        if fit_prop:ax[1].plot(x_plot[cond_fit_all],res_prop[cond_fit_all],marker='o',linestyle='',color='red',zorder=0)  
                                        ax[1].plot([x_plot[0],x_plot[-1]],[0.,0.],linestyle='--',color='black',zorder=0)  
                                        if par in x_range_res:x_range_plot = x_range_res[par]     
                                        if par in y_range_res:y_range_plot = sc_fact*np.array(y_range_res[par])
                                        else:
                                            dy_range_plot = max(res_prop)-min(res_prop)
                                            y_range_plot = [ min(res_prop) - 0.05*dy_range_plot , max(res_prop) + 0.05*dy_range_plot ]  
                                        
                                        #Merit
                                        if fit_prop:
                                            if 'Amp' in par:rms_txt = 'RMS = '+"{0:.2e}".format(merit['rms']*1e6)+' ppm'
                                            else:rms_txt = 'RMS = '+"{0:.2e}".format(merit['rms'])
                                            ax[1].text(x_range_plot[0]+0.1*(x_range_plot[1]-x_range_plot[0]),y_range_plot[1]-0.3*(y_range_plot[1]-y_range_plot[0]),
                                                   rms_txt,verticalalignment='bottom', horizontalalignment='left',fontsize=20,zorder=4,color='green')                                        
                                        
                                        ax[1].set_xlim(x_range_plot)  
                                        ax[1].set_ylim(y_range_plot)  
                                        if 'Amp' in par:ytitle = r'Res. (x 10$^{3}$)'
                                        else:ytitle = 'Res.'                                       
                                        ax[1].set_ylabel(ytitle,fontsize=fontsize) 
                                        ax[1].tick_params('x',labelsize=fontsize)
                                        ax[1].tick_params('y',labelsize=fontsize)
                                        for axis_side in ['bottom','top','left','right']:ax[1].spines[axis_side].set_linewidth(1.5)
                                        
                                        plt.savefig(path_hyper_plot+par+'.png')                  
                                        plt.close()   
                            
                            #--------------------------------                       

                        #Save fit results over all hyperparameters for current exposure
                        if len(hyperpar_chrom_fit)>0:
                            np.savez_compressed(path_dic['datapath_Chrom']+'/Fit_results_iexpGroup'+str(iexp_glob_bin),data=hyperpar_chrom_fit,args={'deg_Freq':fixed_args['deg_Freq'],'deg_Amp':fixed_args['deg_Amp']},allow_pickle=True)      
                                

            ### end of all exposure processing
            n_expgroup = len(iexp_group_list)                 

            #-----------------------------------------------------------    
            #Save filter results
            if gen_dic['wig_exp_filt']['mode']:
                save_dir_vis = gen_dic['save_data_dir']+'/Corr_data/Wiggles/Vis_fit/'+inst+'_'+vis+'/'
                if (not os_system.path.exists(save_dir_vis)):os_system.makedirs(save_dir_vis)
                dic_filter['iexp2glob'] = iexp2glob
                datasave_npz(save_dir_vis+'Outputs_final',dic_filter)

            #-----------------------------------------------------------                                  
            #RMS over all exposures                 
            if (gen_dic['wig_exp_fit']['mode'] and gen_dic['wig_exp_fit']['plot']):  plot_rms_wig(tel_coord_expgroup['cen_ph'],rms_exp_raw,rms_exp_fit,median_err,path_dic['plotpath_Global'])
            if (gen_dic['wig_exp_filt']['mode'] and gen_dic['wig_exp_filt']['plot']):plot_rms_wig(tel_coord_expgroup['cen_ph'],rms_exp_raw,rms_exp_fit,median_err,path_dic['plotpath_Filter'])

            #-----------------------------------------------------------                                  
            #Periodogram from all exposures
            #    - use to identify wiggle components
            if gen_dic['wig_exp_init']['mode'] and gen_dic['wig_exp_init']['plot_hist']:
                plot_global_perio(fixed_args,comm_freq_comp,comm_power_comp,nexp_fit_list,path_dic['plotpath_Init'],color_comps)

            #-------------------------------------------------------------------------------  
            #Analysis of hyperparameter variations
            #    - fitting coordinate-dependent variations of hyperparameters from sampling or global fits, over grouped exposures 
            #-------------------------------------------------------------------------------                    
            if gen_dic['wig_exp_point_ana']['mode']: 
                fixed_args_loc = deepcopy(fixed_args)
                
                #Retrieve model properties
                hyper_par_tab = {}
                fitted_par =  {}
                cond_init_par = False
                for isub_group,iexp_group in enumerate(iexp_group_list):
                    fitted_exp = True

                    #Hyperparameters from sampling or global fit
                    if gen_dic['wig_exp_point_ana']['source']=='samp':
                        exp_path = path_dic['datapath_Chrom']+'/Fit_results_iexpGroup'+str(iexp_group)+'.npz'
                        if os_system.path.exists(exp_path):hyper_fit_exp = np.load(exp_path,allow_pickle=True)['data'].item()  
                        else:fitted_exp = False
                    elif gen_dic['wig_exp_point_ana']['source']=='glob':
                        hyper_fit_exp = np.load(path_dic['datapath_Global']+'/Fit_results_iexpGroup'+str(iexp_group)+'.npz',allow_pickle=True)['data'].item() 
                    if fitted_exp: 
                        if not cond_init_par:
                            for par in hyper_fit_exp:
                                fitted_par[par] = False
                                hyper_par_tab[par] = np.zeros([2,n_expgroup])*np.nan                            
                            cond_init_par = True
                        
                        for par in hyper_fit_exp:
                            if par=='delta_f12':   
                                v_par = hyper_fit_exp['Freq1_c0_off'][0] - hyper_fit_exp['delta_f12'][0]
                                s_par=0.
                                if hyper_fit_exp['Freq1_c0_off'][1] is not None:s_par+=hyper_fit_exp['Freq1_c0_off'][1]**2.
                                if hyper_fit_exp['delta_f12'][1] is not None:s_par+=hyper_fit_exp['delta_f12'][1]**2. 
                                s_par = np.sqrt(s_par)
                                fit_par = [v_par,s_par]
                                par = 'Freq2_c0_off'
                            else:                            
                                fit_par = hyper_fit_exp[par][0:2]  
                            fitted_par[par] = hyper_fit_exp[par][2] 
                            hyper_par_tab[par][:,isub_group] = fit_par


                #Number of fit iterations
                nit_hyper_coord=4

                #Fit storage
                hyperpar_coord_fit = {}

                       
                #Generic plot properties
                if gen_dic['wig_exp_point_ana']['plot']: 
                    
                    #Force axis ranges
                    x_range_var={
                        # 'AmpGlob1_maxAmp_off':[0.000125,0.000213],
                        # 'AmpGlob2_maxAmp_off':[0.000125,0.000213],
                        }
                    x_range_res={}
                    y_range_var={
                        # 'AmpGlob2_c0':[0.,0.0005],   #ANTARESS I   
                        # 'Freq1_c0':[3.695,3.715],   #ANTARESS I    
                        # 'Freq2_c0':[2.01,2.125],   #ANTARESS I                        
                        # 'Freq1_c0':[3.7145,3.731 ],   #ANTARESS I   
                        # 'AmpGlob1_maxAmp_off':[0,0.007],
                    #     'Freq1_c0_off':[3.55,3.9],
                        # 'AmpGlob2_maxAmp_off':[0.,0.007],
                    #     'Freq2_c0_off':[1.7,2.5],
                    #     'AmpGlob3_maxAmp_off':[0.,0.00070],
                    #     'Freq3_c0_off':[0.4,1.5],
                    #     'AmpGlob4_maxAmp_off':[0.0001,0.0004],
                    #     'Freq4_c0_off':[0.25,0.5],
                        }
                    y_range_res={
                        # 'AmpGlob1_c0':[-0.00034,0.00034],   #ANTARESS I 
                        # 'AmpGlob2_c0':[-0.0003,0.0003],   #ANTARESS I                          
                        # 'Freq1_c0':[-0.006,0.006],   #ANTARESS I
                        # 'Freq2_c0':[-0.07,0.07],   #ANTARESS I
                        # 'Phi1':[-0.5,0.5],   #ANTARESS I
                        # 'Phi2':[-1.2,1.2],   #ANTARESS I

                        # 'AmpGlob1_c0':[-0.0003,0.0003],   #ANTARESS I 
                        # 'Freq1_c0':[-0.004,0.004],   #ANTARESS I 
                        # 'Phi1':[-0.25,0.25],   #ANTARESS I
                    #     'AmpGlob1_maxAmp_off':[-0.00025,0.00025],
                    #     'Freq1_c0_off':[-0.1,0.1],
                    #     'AmpGlob2_maxAmp_off':[-0.00015,0.00015],
                    #     'Freq2_c0_off':[-0.3,0.3],
                    #     'AmpGlob3_maxAmp_off':[-0.0002,0.0002],
                    #     'Freq3_c0_off':[-0.5,0.5], 
                    #     'AmpGlob4_maxAmp_off':[-0.0001,0.0001],
                    #     'Freq4_c0_off':[-0.1,0.1], 
                        } 

            
                    #Plot errors 
                    plot_err_var = True
    
                    #Plot symbols 
                    symb_tab = np.repeat('s',n_exp_groups)
                    symb_tab[~tel_coord_expgroup['cond_eastmer']]='d'
                    symb_tab[tel_coord_expgroup['cond_shift']]='o'
            
                    #Plot data with no errors as empty symbols 
                    plot_empty=True

                    #Colors
                    cmap = plt.get_cmap('rainbow') 
                    col_visit=cmap( np.arange(n_expgroup)/(n_expgroup-1.)) 

                #Convert negative amplitude and shift Phi
                #    - condition is that amplitude model at the reddest wavelengths (where wiggles are strong) must be positive
                #    - phase is shifted arbitrarily by pi
                if gen_dic['wig_exp_point_ana']['conv_amp_phase']:
                    for comp_id in range(1,6):
                        comp_str = str(comp_id)
                        AmpGlob_par = [par for par in hyper_par_tab.keys() if 'AmpGlob'+comp_str in par] 
                        if len(AmpGlob_par)==fixed_args_loc['deg_Amp'][comp_id]+1:
                            for isub_group,iexp_group in enumerate(iexp_group_list):
                                amp_par_exp = {par:hyper_par_tab[par][0,isub_group] for par in AmpGlob_par}
                                amp_red = wig_amp_nu_poly(comp_id,np.array([37.]),amp_par_exp,fixed_args_loc)
                                if amp_red<0.:
                                    hyper_par_tab['Phi'+comp_str+'_off'][0,isub_group]+=np.pi  
                                    for ideg in range(fixed_args_loc['deg_Amp'][comp_id]+1):
                                        hyper_par_tab['AmpGlob'+str(comp_id)+'_c'+str(ideg)+'_off'][0,isub_group]*=-1.           

                #Process properties
                #    - we fit the temporal/coordinate dependence of each hyperparameter
                #    - we fit all properties, including fixed ones, so that the associated coordinate properties are derived and stored for use as input to further fits
                file_save = open(path_dic['datapath_Coord']+'/Fit_results','w+')
                fixed_args_loc['stable_pointpar'] = {}
                for par in hyper_par_tab:

                    #Root name
                    par_root = par.split('_off')[0]
                    fixed_args_loc['par_name'] = par_root+'_'
                    fixed_args_loc['stable_pointpar'][fixed_args_loc['par_name']] = False

                    #Fitted property
                    y_var = deepcopy(hyper_par_tab[par][0])
                    sy_var = deepcopy(hyper_par_tab[par][1])
                    
                    #Fit parameters        
                    params_fit = Parameters()

                    #Generic model
                    fit_func = FIT_wig_submod_coord_discont 

                    #Wiggle offset
                    if 'Phi' in par_root:
                        comp_str = str(par_root.split('Phi')[1])
                        ytitle= r'Phase Comp. '+comp_str
                        sc_fact = 1.

                    #Wiggle frequency
                    elif 'Freq' in par_root:
                        comp_str = str(par_root.split('Freq')[1].split('_')[0])
                        deg_str = str(par_root.split('Freq')[1].split('_c')[1])
                        ytitle= r'Freq$_{'+deg_str+'}$ Comp. '+comp_str+' (10$^{10}$ s)'
                        sc_fact = 1.
                      
                    #Wiggle amplitude
                    elif 'AmpGlob' in par_root:
                        comp_str = str(par_root.split('AmpGlob')[1].split('_')[0])
                        deg_str = str(par_root.split('AmpGlob')[1].split('_c')[1])
                        if deg_str=='0':sc_deg = 3.
                        elif deg_str=='1':sc_deg = 4.
                        elif deg_str=='2':sc_deg = 5.
                        ytitle= r'Amp$_{'+deg_str+'}$ Comp. '+comp_str+' (x 10$^{'+str(int(sc_deg))+'}$)'
                        sc_fact = 10**sc_deg

                    #Fit properties
                    for suff in suf_hyper:
                        if suff=='_off':val_guess = np.nanmedian(y_var)
                        else:val_guess = 0.
                        params_fit.add_many((par_root+suff,val_guess, True    , None, None))                             
                    if fixed_args_loc['iexp_guidchange']<1e10:
                        for suff in ['_doff_sh','_dx_shift','_dy_shift','_dz_shift']:params_fit.add_many((par_root+suff,0., True    , None, None))                             
                        if np.sum(tel_coord_expgroup['cond_westmer'])==0:params_fit.add_many((fixed_args_loc['par_name']+'dz_west',np.nan, False   , None, None))

                    #Properties kept stable during a night
                    #    - all associated coordinate properties are set to 0 and kept fixed, except for the constant component 
                    if len(gen_dic['wig_exp_point_ana']['stable_pointpar'])>0:
                        var_suf_hyper_vis = deepcopy(suf_hyper_vis)
                        var_suf_hyper_vis.remove('_off')
                        if fixed_args_loc['iexp_guidchange']<1e10:var_suf_hyper_vis.remove('_doff_sh')
                        if par_root in gen_dic['wig_exp_point_ana']['stable_pointpar']:
                            fixed_args_loc['stable_pointpar'][fixed_args_loc['par_name']] = True
                            for suf_coord in var_suf_hyper_vis:
                                params_fit[par_root+suf_coord].value = 0.
                                params_fit[par_root+suf_coord].vary = 0.
                 
                    # #Correlation tests
                    # params_fit = Parameters()
                    # fit_func = FIT_wig_corre     
                    # params_fit.add_many(
                    #     (fixed_args_loc['par_name']+'a0',0., True    , None, None) ,
                    #     (fixed_args_loc['par_name']+'a1',0., True    , None, None), 
                    #     (fixed_args_loc['par_name']+'a2',0., False    , None, None), 
                    #     (fixed_args_loc['par_name']+'a3',0., False    , None, None) )            

                    #--------------------------------



                    ##Temporary modification
                    # if 'AmpGlob1' in par:
                    #     y_var[hyper_par_tab['AmpGlob1_c1_off'][0]>0.00004]*=-1.
                    # if 'AmpGlob2' in par:
                    #     y_var[hyper_par_tab['AmpGlob2_c0_off'][0]<0.]*=-1.
                    # if par_root=='Freq1_c0':
                    #     if vis=='20180902':
                    #         y_var[(tel_coord_expgroup['cen_ph'] <-0.061)] = np.nan        
                    # if vis=='20180902':
                    #     sy_var[(tel_coord_expgroup['cen_ph'] <-0.055)] = np.nan  
                    #     sy_var[(tel_coord_expgroup['cen_ph'] >0.07)] = np.nan   
                    # if vis=='20181030':
                    #     # sy_var[(tel_coord_expgroup['cen_ph'] >0.06)] = np.nan  
                    #     # sy_var[hyper_par_tab['Freq2_c1_off'][0] > 0.01] = np.nan  
                    #     # sy_var[hyper_par_tab['Freq2_c0_off'][0] < 1.99] = np.nan  
                    #     sy_var[hyper_par_tab['Freq2_c0_off'][0] > 2.15] = np.nan 

                    #     if par_root=='Freq2_c0':
                    #         sy_var[(tel_coord_expgroup['cen_ph'] >0.062)] = np.nan  
                    #     if par_root=='Freq2_c1':
                    #         sy_var[(tel_coord_expgroup['cen_ph'] >0.062)] = np.nan  
                        
                    # if par_root=='Phi1':
                    # # #     if vis=='20190720':
                    # # #         y_var[(y_var<0.)]+=2.*np.pi
                    # # #         y_var[(y_var>10.)]-=2.*np.pi
                    # #         # y_var[(tel_coord_expgroup['cen_ph']<0.018) & (y_var<0.)]+=2.*np.pi
                    #     if vis=='20180902':
                    # #         y_var[y_var<-1]+=2.*np.pi
                    # #         y_var[(tel_coord_expgroup['cen_ph']>0.02) & (y_var>3.)]-=np.pi  
                    #         y_var[(tel_coord_expgroup['cen_ph'] < -0.009)]+=2.*np.pi   
                    #         y_var[y_var<-4.] += 2.*np.pi   
                    # #     # if vis=='20181030':
                    # #     #     y_var[(tel_coord_expgroup['cen_ph']>0.002)]-=np.pi        
                    # # # # #     if vis=='20210124':
                    # # # # #         y_var[y_var>3.]-=2.*np.pi  

                    # # if vis=='20190911':
                    #         y_var[(tel_coord_expgroup['cen_ph'] <-0.01) & (hyper_par_tab['Freq2_c0_off'][0] < 2.04)] = np.nan                    
                    #         y_var[(tel_coord_expgroup['cen_ph'] <-0.02) & (hyper_par_tab['Freq2_c1_off'][0] > 0.005)] = np.nan           
                    #         y_var[(hyper_par_tab['Freq2_c1_off'][0] > 0.01)] = np.nan   
                            
                    # if vis=='20181030':
                    # #     y_var[  (tel_coord_expgroup['cen_ph'] >-0.05) & (hyper_par_tab['Phi2_off'][0] > 10.5)  ] = np.nan                
                    #     y_var[  (tel_coord_expgroup['cen_ph'] >0.02) & (hyper_par_tab['Phi2_off'][0] > 2.)  ] = np.nan    
                    #     y_var[  (tel_coord_expgroup['cen_ph'] >0.05) & (hyper_par_tab['Phi2_off'][0] > -1.)  ] = np.nan    
                    # #     y_var[  (tel_coord_expgroup['cen_ph'] <0.07) & (hyper_par_tab['Freq2_c1_off'][0] > 0.02)  ] = np.nan  
                    # #     y_var[  (tel_coord_expgroup['cen_ph'] >0.065) & (hyper_par_tab['Freq2_c1_off'][0] < 0.007)  ] = np.nan                           
                    
                    # if par_root=='Phi2':
                    # #     if vis=='20190720':
                    # #         y_var[y_var>10.]-= np.pi
                    # # #         y_var[(tel_coord_expgroup['cen_ph'] < -0.003)]+=2.*np.pi  
                    # # #         y_var[(tel_coord_expgroup['cen_ph'] > 0.015)  & (y_var>-1.)]-=2.*np.pi   
                    # # #         y_var[(tel_coord_expgroup['cen_ph'] > -0.02) & (y_var>9.)]-=2.*np.pi    
                    # # #         y_var[(tel_coord_expgroup['cen_ph'] < -0.02) & (y_var>8.)]-=np.pi   
                    # # #         y_var[(tel_coord_expgroup['cen_ph'] < -0.02) & (y_var>8.)]-=np.pi   
                        
                            
                    #     if vis=='20190911':
                    #         y_var[y_var>10.] = np.nan 
                    #         y_var[y_var<-2.6] = np.nan 
                    # # #     #     y_var[(tel_coord_expgroup['cen_ph'] < 0.00) & (y_var<4)]+=2.*np.pi    
                    # # #     #     y_var[(tel_coord_expgroup['cen_ph'] > 0.018) & (y_var>2)]-=2.*np.pi 
                    # # #     # #     y_var[(tel_coord_expgroup['cen_ph'] < -0.02) & (y_var<7.)]+=2.*np.pi 
                            


                        # if vis=='20180902':     
                        # #     y_var[y_var>90.] -= 15*2.*np.pi   
                        # #     y_var[y_var>4.] -= 2.*np.pi     
                        # #     y_var[(tel_coord_expgroup['cen_ph'] <0.)]+=2.*np.pi 
                        # #     y_var[y_var<-5.] += 2.*np.pi  
                        # #     y_var[(tel_coord_expgroup['cen_ph'] <0.03)]+=2.*np.pi 
                        # #     y_var[(tel_coord_expgroup['cen_ph'] <0.04) & (y_var<0)]+=2.*np.pi   
                        #     sy_var[(tel_coord_expgroup['cen_ph'] <-0.043)]=np.nan 
                        #     sy_var[(tel_coord_expgroup['cen_ph'] >0.075)]=np.nan 
                
                            
                            
                            
                        # if vis=='20181030':    
                        #     y_var[(tel_coord_expgroup['cen_ph'] > -0.03) & (y_var>10) ]-=np.pi  
                        #     y_var[(tel_coord_expgroup['cen_ph'] > 0.01) ]-=np.pi  
                        #     y_var[(tel_coord_expgroup['cen_ph'] > 0.001) ]+=np.pi  
                        #     y_var[(tel_coord_expgroup['cen_ph'] > 0.011) ]-=np.pi   
                        #     y_var[(tel_coord_expgroup['cen_ph'] > 0.047) ]+=np.pi   
                        #     sy_var[(tel_coord_expgroup['cen_ph'] > 0.065) ] = np.nan
                        # #     y_var[ ((tel_coord_expgroup['cen_ph'] > 0.002)) & ((tel_coord_expgroup['cen_ph'] < 0.019)) ]+=np.pi         
                        # #     y_var[y_var>15]-=3.*np.pi                                
                            
                    if par_root=='Phi4':
                        if vis=='20190720':
                            y_var[(tel_coord_expgroup['cen_ph'] < 0.015)  & (y_var<0.)]+=np.pi 
                            
                    # sy_var =  np.abs(y_var/hyper_par_tab['Freq1_c0_off'][0])*np.sqrt( (hyper_par_tab['Freq1_c0_off'][1]/hyper_par_tab['Freq1_c0_off'][0])**2. + (sy_var/y_var)**2. )  
                    # y_var/=hyper_par_tab['Freq1_c0_off'][0] 
                    
                    #Variable
                    x_var = tel_coord_expgroup['cen_ph']
                    # x_var = deepcopy(hyper_par_tab['Freq1_c0_off'][0])
                    # x_var = deepcopy(hyper_par_tab['Freq2_c0_off'][0])
                    # x_var = deepcopy(hyper_par_tab['AmpGlob1_c0_off'][0])
                    # x_var = deepcopy(hyper_par_tab['AmpGlob1_c1_off'][0])
                    # x_var[x_var<0.]*=-1.                               
                    # x_var = deepcopy(hyper_par_tab['Phi1_off'][0])
                    # if vis=='20190720':
                    #     x_var[(x_var<0.)]+=2.*np.pi
                    # if vis=='20180902':
                    #     x_var[(cen_ph_expgroups<-0.02)]+=2.*np.pi                    

                    # if vis=='20210121':
                    # # if vis=='20210124':                            
                    #     # plt.plot(cen_ph_vis,x_az_vis,marker='o',linestyle='')
                    #     # plt.plot(cen_ph_vis,y_az_vis,marker='o',linestyle='')
                    #     # plt.plot(cen_ph_vis,z_alt_vis,marker='o',linestyle='')                        
                    #     # plt.plot(cen_ph_vis,data_prop_vis['az'],marker='o',linestyle='')
                    #     plt.plot(cen_ph_vis,data_prop_vis['alt'],marker='o',linestyle='')
                    #     plt.show()
                    #     stop()
  
                    #Fitting   
                    if (vis in gen_dic['wig_exp_point_ana']['fit_range']) and (par_root in gen_dic['wig_exp_point_ana']['fit_range'][vis]):
                        cond_fit_all = np.repeat(False,len(x_var))
                        for bd_band in gen_dic['wig_exp_point_ana']['fit_range'][vis][par_root]:cond_fit_all|=(x_var>bd_band[0]) & (x_var<bd_band[1])                       
                    else:cond_fit_all = np.repeat(True,len(x_var))
                    cond_fit_all &= ~np.isnan(y_var) 
                    if not gen_dic['wig_exp_point_ana']['fit_undef']:cond_fit_all &= ~np.isnan(sy_var)

                    #Generic sub-function
                    def sub_prep(fixed_args_loc,cond_keep,y_var,sy_var):
                        
                        #Telescope coordinates in fitted exposures
                        for key in tel_coord_expgroup:fixed_args_loc[key] = tel_coord_expgroup[key][cond_keep]
             
                        #Attributing constant errors if undefined or null
                        cst_err = (sy_var[cond_keep]==0.)
                        if gen_dic['wig_exp_point_ana']['fit_undef']:cst_err |= np.isnan(sy_var[cond_keep]) 
                        if (np.sum(cst_err) == np.sum(cond_keep)) or (np.nanmax(np.abs(sy_var[cond_keep]))==0.):
                            var_fit = np.array([np.repeat(np.median(y_var[cond_keep]),np.sum(cond_keep))])
                        else:
                            var_fit = sy_var[cond_keep]
                            var_fit[cst_err] = np.median(var_fit[~cst_err])  
                            var_fit = np.array([var_fit])
                        
                        return var_fit,fixed_args_loc

                    #Fit
                    p_best = params_fit 
                    nfree = len([par_loc for par_loc in p_best if p_best[par_loc].vary])
                    var_fit,fixed_args_loc = sub_prep(fixed_args_loc,cond_fit_all,y_var,sy_var)
       
                    #Fitting
                    fixed_args_loc['idx_fit'] = np.ones(np.sum(cond_fit_all),dtype=bool)
                    _,merit,p_best = call_lmfit(p_best,x_var[cond_fit_all],y_var[cond_fit_all],var_fit**2.,fit_func,verbose=False ,fixed_args=fixed_args_loc)
                      
                    #Successive fits with automatic identification and exclusion of outliers
                    if  (gen_dic['wig_exp_point_ana']['thresh'] is not None) and (fitted_par[par]):
                        for it_res in range(nit_hyper_coord):
                            
                            #Model from previous fit iteration
                            var_fit,fixed_args_loc = sub_prep(fixed_args_loc,np.repeat(True,n_exp_groups),y_var,sy_var)                               
                            mod = fit_func(p_best,x_var,args = fixed_args_loc)
                            
                            #Residuals
                            res = y_var - mod
                            
                            #Sigma-clipping
                            disp_est = np.std(res)
                            cond_sub = (res> gen_dic['wig_exp_point_ana']['thresh']*disp_est) | (res<- gen_dic['wig_exp_point_ana']['thresh']*disp_est)

                            #Fit for current iteration
                            if np.sum(cond_fit_all[~cond_sub])>nfree:
                                cond_fit_all[cond_sub]=False
                                var_fit,fixed_args_loc = sub_prep(fixed_args_loc,cond_fit_all,res,sy_var)
                                fixed_args_loc['idx_fit'] = np.ones(np.sum(cond_fit_all),dtype=bool)
                                _,merit,p_best = call_lmfit(p_best,x_var[cond_fit_all],y_var[cond_fit_all],var_fit**2.,fit_func,verbose=False ,fixed_args=fixed_args_loc)
              
                    #Save results of current hyperparameter fit
                    np.savetxt(file_save,[['Parameter '+par_root]],fmt=['%s']) 
                    for subpar in p_best:
                        hyperpar_coord_fit[subpar] = [p_best[subpar].value,p_best[subpar].stderr,p_best[subpar].vary]
                        if p_best[subpar].vary:  
                            np.savetxt(file_save,[['  '+subpar+' = '+str(p_best[subpar].value)+'+-'+str(p_best[subpar].stderr)]],fmt=['%s']) 
                        else:
                            np.savetxt(file_save,[['  '+subpar+' = '+str(p_best[subpar].value)+' (fixed)']],fmt=['%s']) 

                    #--------------------------------
                    #Plotting
                    if gen_dic['wig_exp_point_ana']['plot']:  
                        plt.ioff()        
                        fig, ax = plt.subplots(2, 1, figsize=(20, 10),gridspec_kw = {'wspace':0, 'hspace':0.05, 'height_ratios':[70,30.]})
                        fontsize = 25
                        plot_contact = False
                        plot_modHR = True #& False
                    
                        #Ranges
                        dx_range = np.max(x_var)-np.min(x_var)
                        x_range_plot = [np.min(x_var)-0.05*dx_range,np.max(x_var)+0.05*dx_range]
                        
                        #------------------------------
                        #Values
                        y_var_plot = sc_fact*deepcopy(y_var)
                        sy_var_plot = sc_fact*deepcopy(sy_var)
                        sy_var_plot[np.isnan(sy_var)] = 0.
                        for isub_group in range(n_exp_groups): 
                            if plot_empty and (sy_var_plot[isub_group]==0.):markerfacecolor = 'none'
                            else:markerfacecolor=col_visit[isub_group]                  
                            if plot_err_var:ax[0].errorbar(x_var[isub_group],y_var_plot[isub_group],yerr=sy_var_plot[isub_group]  ,linestyle='',color=col_visit[isub_group],zorder=0,alpha=0.5,marker='')                                                          
                            ax[0].plot(x_var[isub_group],y_var_plot[isub_group],linestyle='',color=col_visit[isub_group],zorder=1,marker=symb_tab[isub_group],markersize=4,markerfacecolor=markerfacecolor)    

                            #Shade unfitted points
                            if ~cond_fit_all[isub_group]:ax[0].plot(x_var[isub_group],y_var_plot[isub_group],linestyle='',color='black',zorder=1,marker=symb_tab[isub_group],markersize=4,markerfacecolor=markerfacecolor)    
                             
                        #Plotting best-fit
                        ax[0].plot(x_var[cond_fit_all],sc_fact*merit['fit'],linestyle='--',color='black',zorder=1,lw=1.5) 

                        #Plotting best fit at high-resolution
                        if plot_modHR:
                            fixed_args_loc.update(tel_coord_HR)
                            mod_HR = fit_func(p_best,np.zeros(len(cen_ph_HR)),args=fixed_args_loc)
                            ax[0].plot(cen_ph_HR,sc_fact*mod_HR,linestyle=':',color='black',zorder=1,lw=1.5)   

                        #Range 
                        if par_root in y_range_var:y_range_plot = sc_fact*np.array(y_range_var[par_root])
                        else:  
                            dy_range_plot = np.nanmax(y_var_plot)-np.nanmin(y_var_plot)
                            y_range_plot = [ np.nanmin(y_var_plot) - 0.05*dy_range_plot , np.nanmax(y_var_plot) + 0.05*dy_range_plot ]  

                        #Main planet contacts
                        if plot_contact:
                            for cont_ph in contact_phases:ax[0].plot([cont_ph,cont_ph],y_range_plot,color='black',linestyle=':',zorder=0,lw=1.5)

                        #Guide shift and meridian crossing
                        ax[0].plot([cen_ph_mer,cen_ph_mer],y_range_plot,linestyle=':',color='black',zorder=1) 
                        if cen_ph_guid is not None:ax[0].plot([cen_ph_guid,cen_ph_guid],y_range_plot,linestyle='--',color='black',zorder=1,lw=1.5)  

                        #Frame  
                        ax[0].set_xlim(x_range_plot)  
                        ax[0].set_ylim(y_range_plot)
                        ax[0].set_ylabel(ytitle,fontsize=fontsize) 
                        ax[0].tick_params('x',labelbottom=False)
                        ax[0].tick_params('y',labelsize=fontsize)
                        for axis_side in ['bottom','top','left','right']:ax[0].spines[axis_side].set_linewidth(1.5)

                        #------------------------------
                    
                        #Residuals
                        if (fitted_par[par]):
                            for key in tel_coord_expgroup:fixed_args_loc[key] = tel_coord_expgroup[key]      
                            res_prop = sc_fact*(y_var - fit_func(p_best,x_var,args = fixed_args_loc) )             
                            for isub_group in range(n_exp_groups): 
                                if plot_empty and (sy_var_plot[isub_group]==0.):markerfacecolor = 'none'
                                else:markerfacecolor=col_visit[isub_group]                  
                                if plot_err_var:ax[1].errorbar(x_var[isub_group],res_prop[isub_group],yerr=sy_var_plot[isub_group]  ,linestyle='',color=col_visit[isub_group],zorder=0,alpha=0.5,marker='')                                                          
                                ax[1].plot(x_var[isub_group],res_prop[isub_group],linestyle='',color=col_visit[isub_group],zorder=1,marker=symb_tab[isub_group],markersize=4,markerfacecolor=markerfacecolor)    
    
                                #Shade unfitted points
                                if ~cond_fit_all[isub_group]:ax[1].plot(x_var[isub_group],res_prop[isub_group],linestyle='',color='black',zorder=1,marker=symb_tab[isub_group],markersize=4,markerfacecolor=markerfacecolor)    
                                
                            #Level
                            ax[1].plot(x_range_plot,[0.,0.],ls='--',color='black',lw=1.5)
                                
                            #Range       
                            if par_root in y_range_res:y_range_plot = sc_fact*np.array(y_range_res[par_root])
                            else:  
                                dy_range_plot = np.nanmax(res_prop)-np.nanmin(res_prop)
                                y_range_plot = [ np.nanmin(res_prop) - 0.05*dy_range_plot , np.nanmax(res_prop) + 0.05*dy_range_plot ]                              

                            #Main planet contacts
                            if plot_contact:
                                for cont_ph in contact_phases:ax[1].plot([cont_ph,cont_ph],y_range_plot,color='black',linestyle=':',lw=1.5,zorder=0)
    
                            #Guide shift and meridian crossing
                            cen_ph_mer = cen_ph_HR[closest(az_HR,180.)]
                            ax[1].plot([cen_ph_mer,cen_ph_mer],y_range_plot,linestyle=':',color='black',zorder=1,lw=1.5) 
                            if cen_ph_guid is not None:ax[0].plot([cen_ph_guid,cen_ph_guid],y_range_plot,linestyle='--',color='black',zorder=1,lw=1.5) 
    
                            #Merit
                            ax[1].text(x_range_plot[0]+0.1*(x_range_plot[1]-x_range_plot[0]),y_range_plot[1]-0.1*(y_range_plot[1]-y_range_plot[0]),
                                        'RMS = '+"{0:.2e}".format(sc_fact*merit['rms']),verticalalignment='bottom', horizontalalignment='left',fontsize=10,zorder=4,color='green') 
    
                            #Frame                              
                            ax[1].set_xlim(x_range_plot)  
                            ax[1].set_ylim(y_range_plot)  
                            ax[1].set_ylabel('Res.',fontsize=fontsize) 
                            ax[1].tick_params('x',labelsize=fontsize)
                            ax[1].tick_params('y',labelsize=fontsize)
                            for axis_side in ['bottom','top','left','right']:ax[1].spines[axis_side].set_linewidth(1.5)
                            
                        ax[1].set_xlabel('Orbital phase',fontsize=fontsize) 
                        plt.savefig(path_dic['plotpath_Coord']+par_root+'.png') 
                        plt.close()                                 
                            

                #Save fit results
                np.savez_compressed(path_dic['datapath_Coord']+'/Fit_results',data=hyperpar_coord_fit,allow_pickle=True)      
                file_save.close()                   

            #-------------------------------------------------------------------------------  
            #Global fit over visit
            #-------------------------------------------------------------------------------  
            if gen_dic['wig_vis_fit']['mode']: 
                print('           - Global visit fit') 
                
                #Fit dictionary
                fixed_args_loc = deepcopy(fixed_args)

                #Settings for fit with wiggle ratio 
                # n_conv = 50
                n_conv = 10
                nmax_loop = 50   #2000
                n_plotchain=1  #10

                #----------------------------------------------------------

                #Initializations
                fit_dic={'save_dir':gen_dic['save_data_dir']+'/Corr_data/Wiggles/Vis_fit/'+inst+'_'+vis+'/','merit':{}}
                if (not os_system.path.exists(fit_dic['save_dir'])):os_system.makedirs(fit_dic['save_dir'])
                if gen_dic['wig_vis_fit']['fixed']:fit_dic['fit_mode'] = ''
                else:fit_dic['fit_mode'] = 'chi2'   
                    
                #Join tables for global model
                for key in ['az','x_az','y_az','z_alt','cond_eastmer','cond_westmer','cond_shift']:fixed_args_loc[key] = tel_coord_vis[key][iexp_fit_list] 
                fixed_args_loc['iexp_bounds'] = np.zeros([2,0],dtype=int)    
                      
                #Spectral processing required
                if cond_exp_proc_vis:
                    fixed_args_loc['iexp_list'] = iexp_fit_list 
                    fixed_args_loc['nexp_list'] = len(iexp_fit_list) 
                    
                    #Components to include
                    fixed_args_loc['comp_mod']=deepcopy(gen_dic['wig_vis_fit']['comp_ids'])
  
                    #Initializations
                    fit_prop_dic={'save_outputs':False}   #To prevent creation of generic output file
    
                    #Join tables for global model
                    nu_all = np.zeros(0,dtype=float)*np.nan
                    Fr_all = np.zeros(0,dtype=float)*np.nan
                    varFr_all = np.zeros(0,dtype=float)*np.nan
                    for iexp in iexp_fit_list:
                        istart = len(nu_all)
                        nu_all = np.append(nu_all,bin_dic[iexp]['nu']) 
                        Fr_all = np.append(Fr_all,bin_dic[iexp]['Fr']) 
                        varFr_all = np.append(varFr_all,bin_dic[iexp]['varFr']) 
                        iend = len(nu_all)
                        fixed_args_loc['iexp_bounds'] = np.append(fixed_args_loc['iexp_bounds'],[[istart],[iend]],axis=1)
    
                    #Model and fit properties 
                    fixed_args_loc.update({
                        'par_list':[],
                        'fit_func':FIT_calc_wig_mod_nu_t,
                        'inside_fit':False, 
                        'x_val':nu_all,
                        'y_val':Fr_all ,  
                        'cov_val':np.array([varFr_all]) , 
                        })
                    fit_dic['nx_fit'] = len(nu_all)

                    #-----------------------------------
    
                    #Possibility to modify guesses and priors before local exposure fit
                    fit_prop_dic['mod_prop'] = deepcopy(mod_prop_vis)
                    fit_prop_dic['varpar_priors'] = deepcopy(varpar_priors_vis)
                    fixed_args_loc['stable_pointpar'] = deepcopy(stable_pointpar_vis)

                    #Replace chromatic coefficients of hyper-parameters by their temporal fits to initialize the global fit
                    #    - default initialization
                    hyperpar_coord_fit = np.load(path_dic['datapath_Coord']+'/Fit_results.npz',allow_pickle=True)['data'].item() 
                    for par in hyperpar_coord_fit:
                        if par in fit_prop_dic['mod_prop']:fit_prop_dic['mod_prop'][par]['guess'] = hyperpar_coord_fit[par][0]
    
                    #Disable west-meridian component if all west-meridian exposures are switched to the shifted guide star model
                    if np.sum(tel_coord_vis['cond_westmer'][iexp_fit_list])==0:
                        for par in fit_prop_dic['mod_prop']:
                            if 'dz_west' in par:
                                fit_prop_dic['mod_prop'][par]['guess'] = 0.
                                fit_prop_dic['mod_prop'][par]['vary'] = False                            
    
                    #Fix specific hyperparameters
                    fixed_par = gen_dic['wig_vis_fit']['fixed_par']
                    for hyperpar in fixed_par:
                        for suf_coord in suf_hyper_vis:
                            fit_prop_dic['mod_prop'][hyperpar+suf_coord]['vary'] = False
    
                    #Fix specific properties
                    fixed_pointpar = gen_dic['wig_vis_fit']['fixed_pointpar']
                    for par in fixed_pointpar:
                        if par not in fit_prop_dic['mod_prop']:stop('Undefined parameter',par)
                        fit_prop_dic['mod_prop'][par]['vary'] = False
    
                    #Properties kept stable during a night
                    #    - all associated coordinate properties are set to 0 and kept fixed, except for the constant component 
                    if len(gen_dic['wig_vis_fit']['stable_pointpar'])>0:
                        var_suf_hyper_vis = deepcopy(suf_hyper_vis)
                        var_suf_hyper_vis.remove('_off')
                        if '_doff_sh' in var_suf_hyper_vis:var_suf_hyper_vis.remove('_doff_sh')
                        for par_root in gen_dic['wig_vis_fit']['stable_pointpar']:
                            fixed_args_loc['stable_pointpar'][par_root+'_'] = True
                            for suf_coord in var_suf_hyper_vis:
                                fit_prop_dic['mod_prop'][par_root+suf_coord]['guess'] = 0.
                                fit_prop_dic['mod_prop'][par_root+suf_coord]['vary'] = False

                    #Manually change variable status
                    #    - provides additional flexibility
                    # fit_prop_dic['mod_prop']['Freq3_c1_off']['vary'] = False
                    # fit_prop_dic['mod_prop']['Freq3_c2_off']['vary'] = False   
    
                    #Define each component
                    #    - by default 'p_start' only contains the coefficients of requested components, up to the degree requested for amplitude and frequency
                    for comp_id in gen_dic['wig_vis_fit']['comp_ids']:     
                        comp_str = str(comp_id)   
    
                        #Amplitude model fixed
                        if comp_id in gen_dic['wig_vis_fit']['fixed_amp']:
                            for pref,suf in zip(pref_names_amp[comp_id],suf_names_amp[comp_id]):
                                for suf_coord in suf_hyper_vis:fit_prop_dic['mod_prop'][pref+comp_str+suf+suf_coord]['vary']  = False                 

                        #Frequency model fixed
                        if comp_id in gen_dic['wig_vis_fit']['fixed_freq']:
                            for pref,suf in zip(pref_names_freq[comp_id],suf_names_freq[comp_id]):
                                for suf_coord in suf_hyper_vis:fit_prop_dic['mod_prop'][pref+comp_str+suf+suf_coord]['vary']  = False                                          
    
    
                    #Parameter initialization
                    p_start = Parameters()  
                    par_formatting(p_start,fit_prop_dic['mod_prop'],fit_prop_dic['varpar_priors'],fit_dic,fixed_args_loc,'','',None)
                    init_fit(fit_dic,fixed_args_loc,p_start,fit_prop_dic,model_par_names,model_par_units)     
                
                #Retrieve previous fit
                if vis in gen_dic['wig_vis_fit']['reuse']:
                    globvisfit_results = np.load(gen_dic['wig_vis_fit']['reuse'][vis],allow_pickle=True)['data'].item()
                    p_start = globvisfit_results['p_best']   

                #Parameter initialization
                fit_dic['save_outputs']=True
                var_par_list=[par for par in p_start if p_start[par].vary] 

                #Oversampled model spectral table
                #    - required to compute the master wiggle and wiggle-to-master ratios 
                dnu_HR = 0.02
                if cond_exp_proc_vis:
                    min_nu_HR = np.min(nu_all)-10.*dnu_HR
                    max_nu_HR = np.max(nu_all)+10.*dnu_HR
                else:
                    min_nu_HR=37.
                    max_nu_HR=76.
                min_max_plot = [min_nu_HR,max_nu_HR]
                n_nu_HR,fixed_args_loc['nu_HR']  = def_wig_tab(min_nu_HR,max_nu_HR,dnu_HR)

                #Definition of master wiggle
                def sub_wig_mast(p_loc):
                    wig_mast = np.zeros(n_nu_HR,dtype=float) 
                    calc_chrom_coord(p_loc,fixed_args_loc)
                    for isub,iexp_off in enumerate(idx_to_bin): 
                        wig_mast+=norm_weight_mast[iexp2mer[iexp_off]][isub]*calc_wig_mod_nu_t(fixed_args_loc['nu_HR'],p_loc,{**fixed_args_loc,'icoord':isub})[0]
                    return wig_mast

                #Fixed model
                if fit_dic['fit_mode'] == '':
                    p_best = deepcopy(p_start)

                #Fitted model 
                elif cond_exp_proc_vis:

                    #No convergence loop if no wiggle ratio
                    if (not gen_dic['wig_vis_fit']['wig_fit_ratio']):nmax_loop=1

                    #Initialize save file
                    globvisfit_results = {
                        'var_par_fit':{subpar:np.zeros([nmax_loop,2],dtype=float)*np.nan for subpar in var_par_list},
                        'rms':np.zeros(nmax_loop,dtype=float)*np.nan,
                        'iexp_fit_list':iexp_fit_list,'var_par_list':var_par_list,'nmax_loop':nmax_loop,
                        'stable_pointpar':fixed_args_loc['stable_pointpar']
                    }  
                    
                    #Initialize master wiggle
                    p_best_pre = deepcopy(p_start)
                    if (gen_dic['wig_vis_fit']['wig_fit_ratio']) and (vis in gen_dic['wig_vis_fit']['reuse']):fixed_args_loc['weighted_wig_mod_HR']=sub_wig_mast(p_best_pre)
                    else:fixed_args_loc['weighted_wig_mod_HR'] = np.repeat(1.,n_nu_HR)
                    
                    #Fit all exposures until reaching convergence on the model parameters
                    #    - the binned transmission ratios are equivalent to 
                    # F(t)/Fmast = F(t)/sum(t in mast , norm_weight(t)*F(t))
                    #            ~ wig(t)*Fstar / sum(t in mast) , norm_weight(t)*wig(t)*Fstar)
                    #            = wig(t) / sum(t in mast) , norm_weight(t)*wig(t))
                    #            = wig(t) / mast_wig(t)
                    #      where norm_weight(t) = weight(t)/sum(t in mast , weight(t)) 
                    #      see calc_bin_profile() for the definition of Fmast
                    #    - in a given iteration, we fit the wiggles for all exposures using the master wiggle calculated at the previous iteration
                    #      we then iterate until reaching convergence on all model parameters, defined as (par[i]/par[i-1]) - 1 < threshold
                    #    - if global fit is performed and wiggle normalization is not applied, there is no need for convergence
                    cond_conv_pre = False
                    iloop = 0
                    i_conv = 0
                    while (i_conv < n_conv-1) and (iloop < nmax_loop):
                        if gen_dic['wig_vis_fit']['wig_fit_ratio']:print('             Loop/convergence:',iloop,i_conv)
                        cond_conv = True
                 
                        #Define model sum of all wiggles over fitted exposures
                        #    - set to unity at first loop
                        #    - then set to weighted sum of best-fit models from previous exposure, or kept to unity if model is not normalized
                        #    - defined over the common, oversampled spectral table
                        #    - if wiggle ratio is not used the master wiggle remains set to unity
                        if (gen_dic['wig_vis_fit']['wig_fit_ratio']) and (iloop>0):
                            fixed_args_loc['weighted_wig_mod_HR']=sub_wig_mast(p_best_pre)

                        #Run fit over several iteration to converge
                        p_best_curr = deepcopy(p_best_pre)
                        fixed_args_loc['idx_fit'] = np.ones(len(nu_all),dtype=bool)
                        for it in range(gen_dic['wig_vis_fit']['nit']):
                            print('               Iteration:',it+1,'/',gen_dic['wig_vis_fit']['nit'])
                            p_best_curr = call_lmfit(p_best_curr,nu_all,Fr_all,np.array([varFr_all]),fixed_args_loc['fit_func'],verbose=True,fixed_args=fixed_args_loc,maxfev = fixed_args_loc['max_nfev'],method=gen_dic['wig_vis_fit']['fit_method'])[2]

                            #Save results every n iterations
                            if it % gen_dic['wig_vis_fit']['n_save_it'] ==0:
                                globvisfit_results['iloop_end']=iloop
                                globvisfit_results['it']=it
                                globvisfit_results['p_best'] = p_best_curr
                                np.savez_compressed(fit_dic['save_dir']+'Outputs_loop'+str(iloop)+'_it'+str(it),data=globvisfit_results,allow_pickle=True)
                                fit_dic['file_save']=open(fit_dic['save_dir']+'Outputs_loop'+str(iloop)+'_it'+str(it),'w+')
                                fit_merit(p_best_curr,fixed_args_loc,fit_dic,False)                
                                fit_dic['file_save'].close() 
                            
                        #Determine uncertainties by running LM fit using iterative solution as starting point
                        _,merit,p_best_curr = call_lmfit(p_best_curr,nu_all,Fr_all,np.array([varFr_all]),fixed_args_loc['fit_func'],verbose=True ,fixed_args=fixed_args_loc,maxfev = fixed_args_loc['max_nfev'],method='leastsq')

                        #Store best fit for current loop
                        for subpar in var_par_list:globvisfit_results['var_par_fit'][subpar][iloop,:] = [p_best_curr[subpar].value,p_best_curr[subpar].stderr] 
                        globvisfit_results['rms'][iloop] = merit['rms']

                        #Convergence check
                        #    - not required if wiggle normalisation is not applied in global fit 
                        if gen_dic['wig_vis_fit']['wig_fit_ratio']:

                            #Save results every n iterations
                            if iloop % n_plotchain ==0:
                                globvisfit_results['iloop_end']=iloop
                                globvisfit_results['p_best'] = p_best_curr
                                np.savez_compressed(fit_dic['save_dir']+'Outputs_loop'+str(iloop),data=globvisfit_results,allow_pickle=True)
                                fit_dic['file_save']=open(fit_dic['save_dir']+'Outputs_loop'+str(iloop),'w+')
                                fit_merit(p_best_curr,fixed_args_loc,fit_dic,False)                
                                fit_dic['file_save'].close() 
                                                         
                            #Assess convergence on all variable parameters
                            #    - condition must be satisfied over the n_conv last iterations
                            for par in var_par_list:cond_conv &= ( abs((p_best_curr[par].value/p_best_pre[par].value)-1.) < gen_dic['wig_vis_fit']['wig_conv_rel_thresh'] )   
    
                            #Update convergence condition over last iterations
                            if not cond_conv:  i_conv = 0      #convergence not reached this iteration (iconv maintainted or reset to 0)
                            elif cond_conv_pre:i_conv+=1       #convergence reached this iteration and over previous iterations (iconv updated)
                            cond_conv_pre = deepcopy(cond_conv)
                        
                            #Update best fit from previous exposure 
                            p_best_pre = deepcopy(p_best_curr)
                        
                        ### End of exposures  
                        iloop+=1                    
    
                    ### End of loops
                    if gen_dic['wig_vis_fit']['wig_fit_ratio']:
                        if iloop==nmax_loop:print('    Reached maximum iterations')
                        else:print('    Reached convergence in ',str(iloop),' iterations')
                    p_best = deepcopy(p_best_curr)

                    #Save final fit results
                    #    - wiggle master is calculated with the last iteration
                    globvisfit_results['iloop_end']:iloop-1
                    globvisfit_results['p_best'] = p_best
                    np.savez_compressed(fit_dic['save_dir']+'Outputs_final',data=globvisfit_results,allow_pickle=True) 
                    fit_dic['file_save']=open(fit_dic['save_dir']+'Outputs_final','w+')
                    fit_merit(p_best,fixed_args_loc,fit_dic,False)                
                    fit_dic['file_save'].close() 

                #Analyse and plot results
                if gen_dic['wig_vis_fit']['plot_mod'] or gen_dic['wig_vis_fit']['plot_rms'] or gen_dic['wig_vis_fit']['plot_hist']:
                
                    #Frequency ranges within which periodograms are calculated for exposure residuals (in 1e-10 s-1)                             
                    if gen_dic['wig_vis_fit']['plot_rms']:
                        rms_prepost_corr = np.zeros([2,nexp_fit_list])*np.nan  
                        median_err = np.zeros(nexp_fit_list)*np.nan  
                    if gen_dic['wig_vis_fit']['plot_hist']:
                        comm_power_comp = {}
                        for comp_id in comm_freq_comp:comm_power_comp[comp_id]=np.zeros(len(comm_freq_comp[comp_id]),dtype=float)
                    
                    #Save path
                    path_fit = gen_dic['save_dir']+gen_dic['main_pl_text']+'_Plots/Spec_raw/Wiggles/Vis_fit/'+inst+'_'+vis+'/Exposures/'
                    if not os_system.path.exists(path_fit):os_system.makedirs(path_fit) 

                    #Periodograms in log mode
                    fixed_args_loc['perio_log'] = True & False

                    #------------------------------ 
                    #Wiggle master
                    #    - calculated with the last iteration
                    weighted_wig_mod_HR = sub_wig_mast(p_best)
                    
                    #Plot
                    plt.ioff()
                    plt.figure(figsize=(100,5)) 
                    ax=plt.gca()
                    x_range_plot = [min_max_plot[0]-0.3,min_max_plot[1]+0.3]
                    ax.set_xlabel(r'$\nu$ (10$^{-10}$ s$^{-1}$)',fontsize=10) 
                    ax.xaxis.set_major_locator(MultipleLocator(1))
                    ax.xaxis.set_major_formatter('{x:.0f}')
                    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
                    max_delta = np.max([ np.max(weighted_wig_mod_HR-1.),np.max(1.-weighted_wig_mod_HR) ])
                    y_range_plot = [1.-max_delta*1.1,1.+max_delta*1.1]
                    ax.set_xlim(x_range_plot)
                    ax.set_ylim(y_range_plot)   
                    ax.set_ylabel('Wiggle master',fontsize=10)  
                    plt.plot(fixed_args_loc['nu_HR'],weighted_wig_mod_HR,linestyle='-',color='red',lw=2,rasterized=fixed_args_loc['rasterized'])      
                    plt.plot(x_range_plot,[1.,1.],linestyle=':',color='black') 
                    plt.savefig(path_fit+'Wiggle_master.png')                  
                    plt.close()   

                    #Store if required
                    if gen_dic['wig_vis_fit']['wig_fit_ratio']:fixed_args_loc['weighted_wig_mod_HR']=weighted_wig_mod_HR
                    else:fixed_args_loc['weighted_wig_mod_HR'] = np.repeat(1.,n_nu_HR) 

                    #Process each exposure
                    calc_chrom_coord(p_best,fixed_args_loc)
                    for isub,(iexp,istart,iend) in enumerate(zip(iexp_fit_list,fixed_args_loc['iexp_bounds'][0],fixed_args_loc['iexp_bounds'][1])): 
    
                        #HR model and components
                        best_mod_HR,comp_mod_HR=calc_wig_mod_nu_t(fixed_args_loc['nu_HR'],p_best,{**fixed_args_loc,'icoord':isub})
                        mod_plot_glob={'all':best_mod_HR/fixed_args_loc['weighted_wig_mod_HR']}
                        for comp_id in comp_mod_HR:mod_plot_glob[comp_id]=(1.+comp_mod_HR[comp_id])/fixed_args_loc['weighted_wig_mod_HR']

                        #Best fit model and residuals
                        best_mod=interp1d(fixed_args_loc['nu_HR'],mod_plot_glob['all'],fill_value='extrapolate')(nu_all[istart:iend]) 
                        flux_res = 1.+Fr_all[istart:iend]-best_mod

                        #Plot
                        if gen_dic['wig_vis_fit']['plot_mod']:
                            plot_wig_glob(None,None,nu_all[istart:iend],Fr_all[istart:iend],np.array([varFr_all[istart:iend]]),fixed_args_loc['nu_HR'],mod_plot_glob,None,None,nu_all[istart:iend],flux_res,np.array([varFr_all[istart:iend]]),fixed_args_loc,min_max_plot,'_',path_fit+'Exp'+str(iexp)+'_')
                     
                        #RMS and median error
                        if gen_dic['wig_vis_fit']['plot_rms']:
                            rms_prepost_corr[0,isub] = Fr_all[istart:iend].std() 
                            rms_prepost_corr[1,isub] = flux_res.std()  
                            median_err[isub] = np.median(np.sqrt(varFr_all[istart:iend]))
                        
                        #Cumulate periodogram from current exposure
                        #    - over global search range, and over the windows of typical components
                        if gen_dic['wig_vis_fit']['plot_hist']:
                            if fixed_args_loc['lb_with_err']:ls = LombScargle(nu_all[istart:iend],flux_res,np.sqrt(varFr_all[istart:iend]))
                            else:ls = LombScargle(nu_all[istart:iend],flux_res)
                            for comp_id in comm_freq_comp:
                                power_comp = ls.power(comm_freq_comp[comp_id])
                                comm_power_comp[comp_id]+=power_comp                        
                    
                    #RMS over all exposures    
                    if gen_dic['wig_vis_fit']['plot_rms']:                
                        plot_rms_wig(tel_coord_vis['cen_ph'][iexp_fit_list],rms_prepost_corr[0],rms_prepost_corr[1],median_err,path_fit)
                               
                    #Periodogram from all exposures
                    #    - use to check quality of fit and possible residual components
                    if gen_dic['wig_vis_fit']['plot_hist']:
                        plot_global_perio(fixed_args,comm_freq_comp,comm_power_comp,nexp_fit_list,path_fit,color_comps)

                #------------------------------------------------------------------------------------------------------------------------
                #Plot chromatic parameter models as a function of pointing coordinates
                if gen_dic['wig_vis_fit']['plot_chrompar_point']:
                    lw_plot = 0.5
                    markersize = 3.

                    #------------------------------                           
                    #Create directory if required
                    path_loc = gen_dic['save_dir']+gen_dic['main_pl_text']+'_Plots/Spec_raw/Wiggles/Vis_fit/'+inst+'_'+vis+'/Chrompar_coord/'
                    if not os_system.path.exists(path_loc):os_system.makedirs(path_loc) 

                    #Colors
                    n_expplot = data_vis['n_in_visit']
                    cmap = plt.get_cmap('rainbow') 
                    col_visit=cmap( np.arange(n_expplot)/(n_expplot-1.))         

                    #Plot symbols 
                    symb_tab = np.repeat('s',n_expplot)
                    symb_tab[~tel_coord_vis['cond_eastmer']]='d'
                    symb_tab[tel_coord_vis['cond_shift']]='o'

                    #Horizontal range
                    dx_range = np.max(tel_coord_vis['cen_ph'])-np.min(tel_coord_vis['cen_ph'])
                    x_range_plot = [np.min(tel_coord_vis['cen_ph'])-0.05*dx_range,np.max(tel_coord_vis['cen_ph'])+0.05*dx_range]    

                    #Plotting each parameter                    
                    for comp_id in gen_dic['wig_vis_fit']['comp_ids']:
                        comp_str=str(comp_id)
                        for pref,suf in zip(pref_names[comp_id],suf_names[comp_id]):
                            chrompar = pref+comp_str+suf

                            #Model at observed exposures and high-resolution
                            mod_coord = wig_submod_coord_discont(n_expplot,p_best,{**fixed_args_loc,'par_name':chrompar+'_',**{key:tel_coord_vis[key] for key in tel_coord_vis}}) 
                            mod_coord_HR = wig_submod_coord_discont(nbjd_HR,p_best,{**fixed_args_loc,'par_name':chrompar+'_',**{key:tel_coord_HR[key] for key in tel_coord_HR}})   

                            #Plot
                            plt.ioff()  
                            plt.figure(figsize=(20,6))
                            for cen_ph,mod_exp,col_exp in zip(tel_coord_vis['cen_ph'],mod_coord,col_visit):
                                plt.plot(cen_ph,mod_exp,color=col_exp,rasterized=fixed_args_loc['rasterized'],zorder=1,marker='o',markersize=markersize)
                            plt.plot(cen_ph_HR,mod_coord_HR,color='black',rasterized=fixed_args_loc['rasterized'],zorder=0,linestyle='-',lw=lw_plot)

                            #Guide shift and meridian crossing
                            dy_range = np.max(mod_coord)-np.min(mod_coord)
                            y_range_plot = [np.min(mod_coord)-0.05*dy_range,np.max(mod_coord)+0.05*dy_range] 
                            plt.plot([cen_ph_mer,cen_ph_mer],y_range_plot,linestyle=':',color='black',zorder=1)  
                            if cen_ph_guid is not None:plt.plot([cen_ph_guid,cen_ph_guid],y_range_plot,linestyle='--',color='black',zorder=1)  

                            #Main planet contacts
                            for cont_ph in contact_phases:plt.plot([cont_ph,cont_ph],y_range_plot,color='black',linestyle=':',lw=0.5,zorder=0)

                            #Frame
                            xmajor_int,xminor_int,xmajor_form=autom_tick_prop(x_range_plot[1]-x_range_plot[0]) 
                            ymajor_int,yminor_int,ymajor_form=autom_tick_prop(y_range_plot[1]-y_range_plot[0]) 
                            custom_axis(plt,position=[0.15,0.15,0.95,0.7],
                                        x_range=x_range_plot,xmajor_int=xmajor_int,xminor_int=xminor_int,xmajor_form=xmajor_form,
                                        y_range=y_range_plot,ymajor_int=ymajor_int,yminor_int=yminor_int,ymajor_form=ymajor_form,dir_y='out',
                                        x_title='Orbital phase',y_title=chrompar,font_size=14,xfont_size=14,yfont_size=14)
                            plt.savefig(path_loc+chrompar+'.png')                  
                            plt.close() 

                                



                #------------------------------------------------------------------------------------------------------------------------
                #Plot parameter models as a function of nu, for all exposures
                if gen_dic['wig_vis_fit']['plot_par_chrom']:
                    lw_plot = 0.5

                    #------------------------------                           
                    #Create directory if required
                    path_loc = gen_dic['save_dir']+gen_dic['main_pl_text']+'_Plots/Spec_raw/Wiggles/Vis_fit/'+inst+'_'+vis+'/Par_chrom/'
                    if not os_system.path.exists(path_loc):os_system.makedirs(path_loc) 
                    
                    #Colors
                    cmap = plt.get_cmap('rainbow') 
                    col_visit=cmap( np.arange(fixed_args_loc['nexp_list'])/(fixed_args_loc['nexp_list']-1.)) 

                    #Calculate coordinate variations of chromatic parameters and offset
                    calc_chrom_coord(p_best,fixed_args_loc)
      
                    #Plotting properties for each component
                    x_range_plot = [min_max_plot[0]-0.3,min_max_plot[1]+0.3]
                    for comp_id in gen_dic['wig_vis_fit']['comp_ids']:
                        comp_str = str(comp_id)
                 
                        def sub_plot(func_name,prop_name):
                            plt.ioff()  
                            plt.figure(figsize=(20,6))
                            y_min=1e100
                            y_max=-1e100
                            
                            #Model
                            if prop_name=='Freq':deg_loc = fixed_args_loc['deg_Freq'][comp_id]
                            elif prop_name=='AmpGlob':deg_loc = fixed_args_loc['deg_Amp'][comp_id]
                            for icoord,col_exp in enumerate(col_visit):
                                chrom_par = {prop_name+comp_str+'_c'+str(ideg)+'_off':fixed_args_loc[prop_name+comp_str+'_c'+str(ideg)+'_coord'][icoord] for ideg in range(deg_loc+1)}
                                par_mod = func_name(comp_id,fixed_args_loc['nu_HR'],chrom_par,fixed_args_loc)                                
                                
                                #Plot model for current exposure
                                plt.plot(fixed_args_loc['nu_HR'],par_mod,linestyle='-',lw=lw_plot,color = col_exp)
                                y_min=np.min([np.min(par_mod),y_min])
                                y_max=np.max([np.max(par_mod),y_max])   
                            
                            #Frame
                            y_range_plot=np.array([y_min,y_max]) 
                            ymajor_int,yminor_int,ymajor_form=autom_tick_prop(y_range_plot[1]-y_range_plot[0]) 
                            custom_axis(plt,position=[0.15,0.15,0.95,0.7],
                                        x_range=x_range_plot,xmajor_int=5,xminor_int=1.,
                                        y_range=y_range_plot,ymajor_int=ymajor_int,yminor_int=yminor_int,ymajor_form=ymajor_form,dir_y='out',
                                        xmajor_form='%i',x_title=r'$\nu$ (10$^{-10}$ s$^{-1}$)',y_title=prop_name,font_size=14,xfont_size=14,yfont_size=14)
                            
                            plt.savefig(path_loc+prop_name+'_comp'+str(comp_id)+'.png')                  
                            plt.close()                             
                            
                            return None
                        
                        #Chromatic amplitude over selected exposures
                        sub_plot(wig_amp_nu_poly,'AmpGlob')

                        #Chromatic frequency over selected exposures                            
                        sub_plot(wig_freq_nu,'Freq')

                #------------------------------------------------------------------------------------------------------------------------
                #Parameter convergence
                if gen_dic['wig_vis_fit']['plot_pointpar_conv'] and gen_dic['wig_vis_fit']['wig_fit_ratio']:     
                    lw_plot=0.5
                    markersize=3.
                    fontsize = 12
                    
                    #Relative threshold (%)
                    rel_thresh=0.2

                    #Bornes du plot  
                    y_range={} 
                    y_range_rel={} 
                    # y_range_var = {'p_0':[0.,0.0063],'per':[-1.,73.]}
                    # y_range_var = {'a_0':[-0.005,0.01]}

                    #------------------------------                           
                    #Create directory if required
                    path_loc = gen_dic['save_dir']+gen_dic['main_pl_text']+'_Plots/Spec_raw/Wiggles/Vis_fit/'+inst+'_'+vis+'/Par_convergence/'
                    if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  

                    #Loop or iteration values
                    iloop_end = globvisfit_results['iloop_end']
                    x_range_plot = np.array([-1,iloop_end+1])
                    xplot = np.arange(iloop_end+1)
                    
                    # #Test correlation
                    # par = 'AmpGlob1_c1_dx_east'
                    # xplot = globvisfit_results['var_par_fit'][par][0:iloop_end+1,0]
                    # x_range_plot = [np.min(xplot),np.max(xplot)]

                    #------------------------------   
                    #RMS over all exposures, as a function of loops
                    plt.ioff() 
                    plt.figure(figsize=(10,6))
                    ax=plt.gca()

                    #RMS
                    yplot = 1e6*globvisfit_results['rms'][0:iloop_end+1]

                    #Mean RMS pre-correction
                    plt.plot(xplot,yplot,linestyle='-',color='dodgerblue',rasterized=fixed_args_loc['rasterized'],zorder=1,marker='o',markersize=markersize,lw=lw_plot)
                        
                    #Frame
                    dy_range = np.max(yplot)-np.min(yplot)
                    y_range_plot = [np.min(yplot)-0.05*dy_range,np.max(yplot)+0.05*dy_range]
                    ax.set_xlim(x_range_plot)  
                    ax.set_xlabel('Step',fontsize=fontsize) 
                    ax.xaxis.set_major_formatter('{x:.0f}')
                    ax.set_ylim(y_range_plot) 
                    ax.set_ylabel('RMS (ppm)',fontsize=fontsize)  
                    plt.savefig(path_loc+'RMS_loops.png') 
                    plt.close() 

                    #------------------------------  
                    #Plot each fitted property
                    for par in var_par_list:
                        plt.ioff()        
                        axd = plt.figure(constrained_layout=True,figsize=(10, 6)).subplot_mosaic([['00'],['10']],gridspec_kw={"bottom": 0.1,"top": 0.99,"left": 0.07,"right": 0.98})            
                        
                        #---------------------------------
                        #Absolute values
                        yplot = globvisfit_results['var_par_fit'][par][0:iloop_end+1,0]
                        splot = globvisfit_results['var_par_fit'][par][0:iloop_end+1,1]
                        n_sdef = np.sum( ~np.isnan(splot) )
                  
                        #Plot property
                        axd['00'].plot(xplot,yplot,linestyle='-',color='dodgerblue',rasterized=fixed_args_loc['rasterized'],zorder=1,marker='o',markersize=markersize,lw=lw_plot)
                        if n_sdef>0:axd['00'].errorbar(xplot,yplot,yerr=splot,color='dodgerblue',rasterized=fixed_args_loc['rasterized'],markeredgecolor='None',markerfacecolor='None',marker='o',markersize=markersize,linestyle='',zorder=0,alpha=0.5) 

                        #Value from last loop
                        axd['00'].plot(x_range_plot,np.repeat(yplot[iloop_end],2),linestyle='--',color='limegreen',rasterized=fixed_args_loc['rasterized'],zorder=1,lw=lw_plot)

                        #Value from first iteration
                        #    - equivalent to assuming the wiggles average out in the master
                        axd['00'].plot(x_range_plot,np.repeat(yplot[0],2),linestyle='--',color='red',rasterized=fixed_args_loc['rasterized'],zorder=1,lw=lw_plot)
                        
                        #Frame
                        if par in y_range:y_range_plot = y_range[par]
                        else:  
                            dy_range_plot = max(yplot)-min(yplot)
                            y_range_plot = [ min(yplot) - 0.05*dy_range_plot , max(yplot) + 0.05*dy_range_plot ] 
                        axd['00'].set_ylim(y_range_plot) 
                        axd['00'].set_ylabel(par,fontsize=fontsize)  
                            
                        #---------------------------------
                        #Relative values
                        #    - we plot the relative variations between a given step and the previous ones
                        xplot_rel=xplot[1::]        
                        if n_sdef>0:
                            splot_num = globvisfit_results['var_par_fit'][par][1:iloop_end+1,1]
                            splot_den = globvisfit_results['var_par_fit'][par][0:iloop_end,1]
                            splot = 100.*np.abs(yplot[1::]/yplot[0:-1])*np.sqrt( (splot_num/yplot[1::])**2. + (splot_den/yplot[0:-1])**2. )
                        yplot = 100.*((yplot[1::]/yplot[0:-1]) - 1.)
                        
                        #Plot property
                        axd['10'].plot(xplot_rel,yplot,linestyle='-',color='dodgerblue',rasterized=fixed_args_loc['rasterized'],zorder=1,marker='o',markersize=markersize,lw=lw_plot)
                        if n_sdef>0:axd['10'].errorbar(xplot_rel,yplot,yerr=splot,color='dodgerblue',rasterized=fixed_args_loc['rasterized'],markeredgecolor='None',markerfacecolor='None',marker='o',markersize=markersize,linestyle='',zorder=0,alpha=0.5) 
                        
                        #Plot relative threshold
                        axd['10'].plot(x_range_plot,[rel_thresh,rel_thresh],linestyle=':',color='black',rasterized=fixed_args_loc['rasterized'],zorder=1,lw=lw_plot)
                        axd['10'].plot(x_range_plot,[0.,0.],linestyle='--',color='black',rasterized=fixed_args_loc['rasterized'],zorder=1,lw=lw_plot)
                        axd['10'].plot(x_range_plot,[-rel_thresh,-rel_thresh],linestyle=':',color='black',rasterized=fixed_args_loc['rasterized'],zorder=1,lw=lw_plot)

                        #Frame
                        if par in y_range_rel:y_range_plot = y_range_rel[par]
                        else:  
                            dy_range_plot = max(yplot)-min(yplot)
                            y_range_plot = [ min(yplot) - 0.05*dy_range_plot , max(yplot) + 0.05*dy_range_plot ] 
                        axd['10'].set_ylim(y_range_plot) 
                        axd['10'].set_ylabel('(y[i]-y[i-1])/y[i-1] %',fontsize=fontsize)  
                        
                        #---------------------------------
                                                
                        #Plot frame  
                        for ax_key in ['00','10']:
                            axd[ax_key].set_xlim(x_range_plot)                              
                            axd[ax_key].set_xlabel('Step',fontsize=fontsize) 
                            # axd[ax_key].xaxis.set_major_formatter('{x:.0f}')
                        plt.savefig(path_loc+par+'.png') 
                        plt.close() 

      
                     
            #-------------------------------------------------------------------------------  
            #Correcting data
            #-------------------------------------------------------------------------------  
            if gen_dic['wig_corr']['mode']:  
                print('           - Correcting data')  
                fixed_args_loc = deepcopy(fixed_args)
                
                #Original indexes of exposures to be corrected
                if (vis in gen_dic['wig_corr']['exp_list']):iexp_corr_list = gen_dic['wig_corr']['exp_list'][vis]
                else:iexp_corr_list = np.arange(data_vis['n_in_visit'])                
                
                #Correction
                if (vis in gen_dic['wig_corr']['path']):corr_path = gen_dic['wig_corr']['path'][vis]
                else:corr_path = gen_dic['save_data_dir']+'/Corr_data/Wiggles/Vis_fit/'+inst+'_'+vis+'/Outputs_final'
                data_corr = dataload_npz(corr_path)
                
                #Analytical model
                if ('iexp2glob' not in data_corr):
                    p_corr = data_corr['p_best']
                    fixed_args_loc.update({
                        'comp_mod':gen_dic['wig_corr']['comp_ids'],
                        'nexp_list':len(iexp_corr_list),
                        'stable_pointpar':data_corr['stable_pointpar']
                       })
                    for key in ['az','x_az','y_az','z_alt','cond_eastmer','cond_westmer','cond_shift']:fixed_args_loc[key] = tel_coord_vis[key][iexp_corr_list] 
                    calc_chrom_coord(p_corr,fixed_args_loc)          

                #Processing exposures
                proc_DI_data_paths_new = gen_dic['save_data_dir']+'Corr_data/Wiggles/Data/'+inst+'_'+vis+'_' 
                for isub,iexp in enumerate(iexp_corr_list):
               
                    #Latest processed disk-integrated data
                    data_exp = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp))                     
                    
                    #Flatten the order matrix into a 1D table
                    #    - correction must be applied to sorted array for when interpolator is applied
                    n_flat = data_inst['nord']*data_vis['nspec']
                    cen_bins_flat = np.reshape(data_exp['cen_bins'],n_flat)
                    nu_bins_flat = c_light/cen_bins_flat[::-1]
                    isort = np.argsort(nu_bins_flat)
                    nu_bins_flat = nu_bins_flat[isort]
                
                    #Shift spectral table from solar barycentric rest frame to earth rest frame, in which model is defined
                    #    - see gen_specdopshift():
                    # w_source = w_receiver / (1+ (rv[s/r]/c))
                    # w_Earth = w_solbar / (1+ (BERV/c))   
                    #    - since flux values remain associated to the original pixels, there is no need to shift back the model after definition
                    nu_bins_flat*=1./(gen_specdopshift(data_prop[inst][vis]['BERV'][iexp])*(1.+1.55e-8))  
                            
                    #Filter
                    if ('iexp2glob' in data_corr):
                        iexp2glob = data_corr['iexp2glob']
                        
                        #Exposure group to which exposure belongs to 
                        if iexp in iexp2glob:
                            
                            #Exposure was included in wiggle analysis
                            iexp_glob_bin = iexp2glob[iexp]   
                            
                        else:
                            
                            #Find closest exposure included in wiggle analysis and its group 
                            isub_low = iexp - np.arange(1,iexp+1)
                            isub_high = iexp + np.arange(1,data_vis['n_in_visit']-1-iexp+1)
                            isub_alternate = [x for x in itertools.chain.from_iterable(itertools.zip_longest(isub_low,isub_high)) if x is not None]
                            for isub in isub_alternate:
                                if isub in iexp2glob:
                                    iexp_glob_bin = iexp2glob[isub]   
                                    break

                        #Retrieve filter from current exposure group
                        if iexp_glob_bin in data_corr:
                            iexp_filt = iexp

                        #Retrieve filter from closest exposure group
                        else:
                            isub_low = iexp_glob_bin - np.arange(1,iexp+1)
                            isub_high = iexp_glob_bin + np.arange(1,nexp_fit_list-1-iexp_glob_bin+1)
                            isub_alternate = [x for x in itertools.chain.from_iterable(itertools.zip_longest(isub_low,isub_high)) if x is not None]
                            for isub in isub_alternate:
                                if isub in data_corr:
                                    iexp_filt = isub
                                    break

                        #Calculate correction model 
                        wig_mod_corr=data_corr[iexp_filt](nu_bins_flat)
                        
                    #Analytical model
                    else:
                    
                        #Calculate correction model
                        #    - the model is calculated without oversampling, as wiggles spectral scales are larger than bin widths  
                        wig_mod_corr=calc_wig_mod_nu_t(nu_bins_flat,p_corr,{**fixed_args_loc,'icoord':isub})[0]
                    
                    #Unsort correction array and return to wavelength space 
                    iunsort = np.argsort( isort )
                    wig_mod_corr = wig_mod_corr[iunsort][::-1]
                    
                    #Restrict correction array to defined bins
                    pcorr_flat = np.ones(n_flat,dtype=float)
                    cond_def_flat = np.reshape(data_exp['cond_def'],n_flat)
                    pcorr_flat[cond_def_flat] = wig_mod_corr[cond_def_flat]       

                    #Return correction to matrix shape
                    #    - the correction is defined as a modulation around unity and will not change the overall flux level of the spectra
                    pcorr= np.reshape(pcorr_flat,[data_inst['nord'],data_vis['nspec']])                
        
                    #Define correction range
                    #    - we do not correct undefined pixels (correction is set to 1)  
                    cond_corr = data_exp['cond_def']
                    if (vis in gen_dic['wig_corr']['range']) and (len(gen_dic['wig_corr']['range'][vis])>0):
                        cond_range = False
                        for bd_int in gen_dic['wig_corr']['range'][vis]:cond_range |= (data_exp['edge_bins'][:,0:-1]>=bd_int[0]) & (data_exp['edge_bins'][:,1::]<=bd_int[1])  
                        cond_corr &= cond_range

                    #Applying correction
                    for iord in np_where1D( np.sum( cond_corr,axis=1 )>0 ):
                        wig_fit_ord = np.ones(pcorr[iord].shape,dtype=float)
                        wig_fit_ord[cond_corr[iord]] = pcorr[iord][cond_corr[iord]]
                        data_exp['flux'][iord],data_exp['cov'][iord] = bind.mul_array(data_exp['flux'][iord],data_exp['cov'][iord],1./wig_fit_ord)

                    #Saving modified data and update paths
                    datasave_npz(proc_DI_data_paths_new+str(iexp),data_exp) 

                #Save for plotting purpose
                dic_plot = {'wig_range_fit':gen_dic['wig_range_fit'],'corr_path':corr_path,'iexp_mast_list':iexp_mast_list,'uncorrected_data_path':deepcopy(data_vis['proc_DI_data_paths']),'corrected_data_path':proc_DI_data_paths_new}
                if ('iexp2glob' not in data_corr):
                    dic_plot.update({
                        'comp_ids':gen_dic['wig_corr']['comp_ids'],'nu_ref':fixed_args['nu_ref'],'tel_coord_vis':tel_coord_vis,'deg_Freq':fixed_args['deg_Freq'],'deg_Amp':fixed_args['deg_Amp']})
                datasave_npz(gen_dic['save_data_dir']+'Corr_data/Wiggles/Vis_fit/'+inst+'_'+vis+'_add',dic_plot)      
                data_vis['proc_DI_data_paths'] = proc_DI_data_paths_new                        
                               
    #------------------------------------------------------------------------------------------------------------
                        
    #Updating path to processed data and checking it has been calculated
    else:
        if gen_dic['wig_corr']['mode']: 
            for vis in gen_dic['wig_vis']:     
                data_vis=data_inst[vis]
                data_vis['proc_DI_data_paths']=gen_dic['save_data_dir']+'Corr_data/Wiggles/Data/'+inst+'_'+vis+'_'    
                check_data({'path':data_vis['proc_DI_data_paths']+str(0)},vis=vis)  

    return None





'''
Wiggle sampling functions
'''

def def_wig_tab(min_nu,max_nu,dnu):
    r"""**Wiggle table definition**

    Defines wiggle table in :math:`\nu` space   

    Args:
        min_nu (float): Start value of the table in :math:`\nu`.
        max_nu (float): End value of the table in :math:`\nu`.
        dnu (float): Resolution of the table in :math:`\nu`.
    
    Returns:
        n_nu (float): Number of table values.
        nu_tab (1D array): :math:`\nu` table.
    
    """      
    n_nu = int((max_nu-min_nu)/dnu)+1
    nu_tab = min_nu+dnu*np.arange(n_nu)  
    return n_nu,nu_tab

def wig_mod_cst(params,x_in,args=None):
    r"""**Flat wiggle model**

    Defines wiggle model with constant value.

    Args:
        TBD
    
    Returns:
        TBD
    
    """  
    return np.repeat(params['level'],len(x_in))



def wig_def_sampbands(comp_id,nu,shift_off,args,p_start_gen):
    r"""**Wiggle sampling bands**

    Defines sampling bands of the wiggles.
    
     - sampling size is based on the period component
     - the shift to oversample is applied to the starting wavelength, so that the sampling width remains correctly adjusted with wavelength
     
    Args:
        TBD
    
    Returns:
        TBD
    
    """  
    if shift_off is not None:

        #Return period of wiggle in nu space
        def samp_per(nu,comp_id_in):
            return 1./wig_freq_nu(comp_id,nu,p_start_gen,args)
        
        #Smallest frequency is at low nu (smallest period is at low wavelength / high nu)
        min_bandwidth = args['nsamp'][comp_id]*samp_per(nu[-1],comp_id)
        nsamp_sub = int(shift_off/min_bandwidth)
        max_samp = nu[-1] + shift_off - (nsamp_sub+1)*min_bandwidth
        min_samp = nu[0]
        low_edge = max_samp
        high_edge = max_samp
        sampbands_edges = np.array([high_edge],dtype=float)
        while low_edge > min_samp: 
            low_edge-=args['nsamp'][comp_id]*samp_per(low_edge,comp_id)
            sampbands_edges = np.append(low_edge,sampbands_edges)
            high_edge = low_edge 

        #Shifted sample bands still within the spectrum range
        low_theoband_edges = sampbands_edges[0:-1]
        high_theoband_edges = sampbands_edges[1:]
        idx_samp_in = np_where1D((low_theoband_edges <= nu[-1]) &  (high_theoband_edges >= nu[0]))
        low_theoband_edges = low_theoband_edges[idx_samp_in]
        high_theoband_edges = high_theoband_edges[idx_samp_in]
        n_bands = len(idx_samp_in)
    else:
        low_theoband_edges = [nu[0]]
        high_theoband_edges = [nu[-1]]
        n_bands = 1

    return low_theoband_edges,high_theoband_edges,n_bands


def wig_perio_gen(calc_range,src_range,nu_in,flux_in,err_in,args,plot=False):
    r"""**Wiggle periodogram**

    Calculates wiggle periodogram over a current spectral band.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Periodogram on current band
    #    - increase 'samples_per_peak' if derived frequencies are not precise enough
    if args['lb_with_err']:ls = LombScargle(nu_in,flux_in,err_in)
    else:ls = LombScargle(nu_in,flux_in)
    frequency, power = ls.autopower(minimum_frequency=src_range[0],maximum_frequency=src_range[1],samples_per_peak=100)
    good_ls = (~np.isnan(power)) & (~np.isinf(power))
    frequency = frequency[good_ls]
    power = power[good_ls]
   
    #Peak search
    max_power = power.max()
    
    #FAP
    fap_max = 100.*(ls.false_alarm_probability(max_power)) 

    #Define guess for period of main component
    freq_guess = frequency[np.argmax(power)]

    #Guesses for amplitude and offset
    #    - over a given band, with a constant period, the wiggle model writes as:
    # wig(w) = 0.5*maxA*sin( 2pi*f*w - Phi )
    #      which can be rewritten as:
    # wig(w) = th1*sin(2pifw) + th2*cos(2pifw) = 0.5*maxA*sin(2pifw)*cos(Phi) - 0.5*maxA*cos(2pifw)*sin(Phi)
    # th1 = maxA*cos(Phi)/2
    # th2 = -maxA*sin(Phi)/2
    # A = 2.*sqrt(th1^2 + th2^2) 
    # phi = arctan(-th2/th1)  
    theta = ls.model_parameters(freq_guess)
    amp_guess = 2.*np.sqrt(theta[2]**2.+theta[1]**2.)
    phi_guess = np.arctan2(-theta[2],theta[1])
    if np.sign(np.cos(phi_guess))==-1:phi_guess+=np.pi

    #Calculate periodogram on large range for visibility
    if (calc_range[0]<src_range[0]) or (calc_range[1]>src_range[1]):
        frequency, power = ls.autopower(minimum_frequency=calc_range[0],maximum_frequency=calc_range[1],samples_per_peak=10)
    
    return amp_guess,freq_guess,phi_guess,fap_max,ls,frequency, power,max_power

def plot_bin_spec(ax,low_bins,high_bins,val_bins,min_plot,max_plot,dbin_plot):
    r"""**Wiggle plot : binned spectrum**

    Plot the binned transmission spectrum of an exposure.
    We use resample_func() rather than bind.resampling because input tables are not necessarily continuous, and because the calculations here are less generic than those in process_bin_prof()

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    nbin_plot = int((max_plot-min_plot)/dbin_plot)
    if nbin_plot<2:nbin_plot=2
    dbin_plot = (max_plot-min_plot)/nbin_plot
    x_bd_low_in = min_plot + dbin_plot*np.arange(nbin_plot)
    x_bd_high_in = x_bd_low_in+dbin_plot 
    _,_,wav_bin_plot,_,Fr_bin_plot,_ = resample_func(x_bd_low_in,x_bd_high_in,low_bins,high_bins,val_bins,None)
    ax.plot(wav_bin_plot,Fr_bin_plot,linestyle='-',marker='o',markersize=5,color='goldenrod',zorder=1)
    
    return None

def plot_screening(ibin2exp_fit,ibin2ord_fit,min_plot,max_plot,gen_dic,Fr_bin_fit,rms,plot_path,fixed_args):
    r"""**Wiggle plot : screening**

    Plot the transmission spectrum of an exposure with the associated periodogram.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 

    #Unique list of processed exposures and orders
    expbin_list = np.unique(ibin2exp_fit)
    ordbin_list = np.unique(ibin2ord_fit)
    
    #Frame
    plt.ioff()        
    plt.figure(figsize=(  min( 10*max([2,int(len(ordbin_list)/3)]) , 100 )  , 6))

    #Set fixed range to better visualize wiggles
    x_range = [min_plot,max_plot]
    if gen_dic['wig_exp_init']['y_range'] is None:y_range = 1. + np.array([-1.,1])*np.std(Fr_bin_fit['Fr'])*10.
    else:y_range = gen_dic['wig_exp_init']['y_range']
  
    #Plot each exposure and slices independently
    for (iexp_plt,iord_plt) in it_product(expbin_list,ordbin_list):
        
        #Binned points belonging to current exposure and slice
        ibin_group = ((ibin2exp_fit==iexp_plt) & (ibin2ord_fit==iord_plt))
        plt.plot(Fr_bin_fit['nu'][ibin_group],Fr_bin_fit['Fr'][ibin_group],linestyle='-',zorder=0)
            
        #Index of global order
        if iexp_plt==expbin_list[0]:
            dsign_txt = 1. if is_odd(iord_plt) else -1.
            plt.text(np.mean(Fr_bin_fit['nu'][ibin_group]),y_range[1]+(0.1+dsign_txt*0.05)*(y_range[1]-y_range[0]),str(iord_plt),verticalalignment='center', horizontalalignment='center',fontsize=10.,zorder=15,color='black') 
 
    #Binned data
    #    - we use resample_func() rather than bind.resampling because input tables are not necessarily continuous, and because the calculations here are less generic than those in process_bin_prof()
    plot_bin_spec(plt.gca(),Fr_bin_fit['low_nu'],Fr_bin_fit['high_nu'],Fr_bin_fit['Fr'],min_plot,max_plot,0.27/5.)

    #Constant unity level
    plt.plot(x_range,[1.,1.],linestyle=':',color='black',zorder=-1,rasterized=fixed_args['rasterized'])
 
    #Dispersion of fitted spectrum
    dx_range = x_range[1]-x_range[0]
    dy_range = y_range[1]-y_range[0]
    plt.text(x_range[1]-0.1*dx_range,y_range[1]-0.2*dy_range,'RMS ='+"{0:.4e}".format(rms),verticalalignment='center', horizontalalignment='left',fontsize=9.,zorder=10,color='black')     
 
    #Frame
    xmajor_int,xminor_int,xmajor_form = autom_tick_prop(x_range[1]-x_range[0])
    ymajor_int,yminor_int,ymajor_form = autom_tick_prop(y_range[1]-y_range[0])  
    xmajor_int = 1.
    xminor_int = 0.5
    custom_axis(plt,position=[0.15,0.15,0.95,0.7],
                x_range=x_range,xmajor_int=xmajor_int,xminor_int=xminor_int,
                y_range=y_range,ymajor_int=ymajor_int,yminor_int=yminor_int,ymajor_form=ymajor_form,dir_y='out',
                xmajor_form=xmajor_form,x_title=r'$\nu$ (10$^{-10}$ s$^{-1}$)',y_title='Flux ratio',font_size=16,xfont_size=16,yfont_size=16)
    plt.savefig(plot_path+'.png') 
    plt.close()  

    return None    


def plot_sampling_perio(fixed_args,ax,best_freq,src_range,calc_perio,ls,ls_freq,ls_pow,freq_guess,max_pow,fap_max,probas,log=False,fontsize=10,plot_xlab=True,plot_ylab=True): 
    r"""**Wiggle plot : periodogram**

    Plot the sampled wiggle periodogram.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Periodogram
    ax.plot(ls_freq,ls_pow,color='black',zorder=1,rasterized=fixed_args['rasterized'])
    ax.plot([freq_guess,freq_guess] , [0.,max_pow],color='dodgerblue',linestyle='--',zorder=5,rasterized=fixed_args['rasterized'])     
    if best_freq is not None:
        ax.plot([best_freq,best_freq] , [0.,max_pow],color='red',linestyle='--',zorder=5)
   
    #Range searched for frequency
    if src_range[0]>calc_perio[0]:ax.axvspan(calc_perio[0],src_range[0], hatch='/',zorder=4,facecolor='none',edgecolor = 'grey')        
    if src_range[1]<calc_perio[1]:ax.axvspan(src_range[1],calc_perio[1], hatch='/',zorder=4,facecolor='none',edgecolor = 'grey')             
        
    #FAP
    fap_levels = ls.false_alarm_level(probas)  
    for fap_lev in fap_levels:ax.plot(calc_perio,[fap_lev,fap_lev],color='black',linestyle=':',zorder=1)
   
    #FAP of maximum power
    plot_txt = False
    if plot_txt:
        ax.text(calc_perio[0]+0.6*(calc_perio[1]-calc_perio[0]),0.8*max_pow,'FAP = '+"{0:.3e}".format(fap_max)+'%',verticalalignment='bottom', horizontalalignment='left',fontsize=12,zorder=4,color='black') 
        ax.text(calc_perio[0]+0.6*(calc_perio[1]-calc_perio[0]),0.6*max_pow,'F = '+"{0:.3f}".format(freq_guess),verticalalignment='bottom', horizontalalignment='left',fontsize=12,zorder=4,color='black') 
                     
    #Ranges
    ax.set_xlim(calc_perio)  
    if plot_xlab:ax.set_xlabel(r'10$^{10}$ s',fontsize=fontsize)
    if plot_ylab:ax.set_ylabel('Power',fontsize=fontsize)
    ymax = np.max([np.max(fap_levels),max_pow])
    if log:
        if fixed_args['log_glob_perio']:ax.set_xscale('log') 
        ax.set_yscale('log')
    ax.set_ylim([fixed_args['min_y_glob_perio'],1.05*ymax])   
    ax.tick_params('x',labelsize=fontsize)
    ax.tick_params('y',labelsize=fontsize)
    
    return None


def plot_global_perio(fixed_args,comm_freq_comp,comm_power_comp,nexp,path_gen,color_comps): 
    r"""**Wiggle plot : global periodogram**

    Plots wiggle periodogram calculated over all exposures.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    if 4 in comm_freq_comp:
        grid = [['00','00','00','00'],['10','11','12','13']]
        ax_keys = ['00','10','11','12','13']
        comp_list = ['glob',3,2,1,4]
    else:
        grid = [['00','00','00'],['10','11','12']]
        ax_keys = ['00','10','11','12']        
        comp_list = ['glob',3,2,1]
        
    plt.ioff() 
    axd = plt.figure(constrained_layout=True,figsize=(40,15)).subplot_mosaic(grid,gridspec_kw={"bottom": 0.1,"top": 0.99,"left": 0.05,"right": 0.99})
    fontsize=35

    #Plot each periodogram
    for comp_id,ax_key in zip(comp_list,ax_keys):
        
        #Normalize by number of exposures
        power_comp = comm_power_comp[comp_id]/nexp
        
        #Find maximum power
        max_pow = power_comp.max()
        freq_guess = comm_freq_comp[comp_id][np.argmax(power_comp)]                      

        #Plot periodogram
        if (ax_key=='00'):col_loc = 'black'
        else:col_loc = color_comps[comp_id]
        axd[ax_key].plot(comm_freq_comp[comp_id],power_comp,color='black',zorder=0,rasterized=fixed_args['rasterized'])
        axd[ax_key].plot([freq_guess,freq_guess] , [0.,max_pow],color=col_loc,linestyle='--',zorder=5,rasterized=fixed_args['rasterized'])     
        axd[ax_key].text(comm_freq_comp[comp_id][0]+0.2*(comm_freq_comp[comp_id][-1]-comm_freq_comp[comp_id][0]),0.6*max_pow,'F = '+"{0:.3f}".format(freq_guess),verticalalignment='bottom', horizontalalignment='left',fontsize=30,zorder=2,color=col_loc) 
                    
        #Frame
        x_range_plot=[comm_freq_comp[comp_id][0],comm_freq_comp[comp_id][-1]]
        y_range_plot = [fixed_args['min_y_glob_perio'],1.05*max_pow]
        if (ax_key=='00') and (fixed_args['log_glob_perio']):
            axd[ax_key].set_xscale('log')            
            axd[ax_key].set_yscale('log')             
            axd[ax_key].set_ylabel('Log Power',fontsize=fontsize)  
        else:
            axd[ax_key].set_ylabel('Power',fontsize=fontsize)  
        axd[ax_key].set_xlim(x_range_plot) 
        axd[ax_key].set_xlabel(r'Freq (10$^{10}$ s)',fontsize=fontsize)  
        axd[ax_key].set_ylim(y_range_plot) 
        axd[ax_key].tick_params('x',labelsize=fontsize)
        axd[ax_key].tick_params('y',labelsize=fontsize)
        for axis_side in ['bottom','top','left','right']:axd[ax_key].spines[axis_side].set_linewidth(2)
      
    #Save
    plt.savefig(path_gen+'Perio_ExpAll.png') 
    plt.close() 
    
    return None



def plot_sampling_spec(ax,x_range_plot_in,nu_fit,var_plot,nu_plot,mod_HR,mean_wav,mean_nu,fixed_args,fontsize=10,plot_xlab=True,plot_ylab=True):
    r"""**Wiggle plot : spectra**

    Plot transmission spectra in sampled bands.

    Args:
        TBD
    
    Returns:
        TBD
    
    """         
    x_fit = nu_fit
    x_plot = nu_plot
    x_range_plot = x_range_plot_in
    x_cen = mean_nu
    if plot_xlab:ax.set_xlabel(r'$\nu$ (10$^{-10}$ s$^{-1}$)',fontsize=fontsize)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        
    #Fitted spectrum
    ax.plot(x_fit,var_plot,linestyle='-',color='black',zorder=1,rasterized=fixed_args['rasterized'])  
    max_delta = np.max([ np.max(var_plot-1.),np.max(1.-var_plot) ])
    y_range_plot = [1.-max_delta*1.1,1.+max_delta*1.1]             

    #Wiggle model
    if np.max(mod_HR)>0.:ax.plot(x_plot,mod_HR,linestyle='--',color='red',zorder=2,rasterized=fixed_args['rasterized'])      
    
    #Dispersion of fitted spectrum
    plot_rms = True & False
    if plot_rms:
        rms = (var_plot-1.).std()
        ax.text(x_range_plot[0]+0.1*(x_range_plot[1]-x_range_plot[0]),
                y_range_plot[1]-0.1*(y_range_plot[1]-y_range_plot[0]),
                'RMS = '+"{0:.0f}".format(rms*1e6)+' ppm',verticalalignment='bottom', horizontalalignment='left',fontsize=fontsize,zorder=4,color='green') 

    #Reference level and mean in band
    ax.plot(x_range_plot,[1.,1.],linestyle=':',color='black')    
    if x_cen is not None:ax.plot([x_cen,x_cen],y_range_plot,linestyle=':',color='black') 
    
    ax.set_xlim(x_range_plot)
    ax.set_ylim(y_range_plot)  
    if plot_ylab:ax.set_ylabel('Flux ratio',fontsize=fontsize)   
    ax.tick_params('x',labelsize=fontsize)
    ax.tick_params('y',labelsize=fontsize)

    return None



def wig_perio_sampling(comp_id_proc,plot_samp,samp_fit_dic,shift_off,ishift_comp,ishift,comp_id,nu_in,flux_in,var_in,count,args,p_start_gen,nu_amp,src_perio_comp,comp_freqfixed,pref_names,suf_names,fix_freq2perio,fit_results,path_sampling_plot,freq_params_samp,fixed_args):
    r"""**Wiggle sampling**

    Main routine to sample wiggle properties chromatically

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    ishift_comp[comp_id] = ishift
    comp_str = str(comp_id)

    #Sampling bands
    low_theoband_edges,high_theoband_edges,n_bands = wig_def_sampbands(comp_id,nu_in,shift_off,args,freq_params_samp)

    #Reset model for current component
    samp_fit_dic['mod_plot_glob'][comp_id][:] = 1.    

    #Initialize plot
    if plot_samp:                
        plt.ioff() 
        if n_bands==1:
            nsub_col = 1 
            figsize = (40, 7)
        else:
            figsize = (40, 15)
            if n_bands>10:nsub_col = 4
            elif n_bands>5:nsub_col = 2
            else:nsub_col = 1    
        nsub_rows = int(np.ceil(n_bands/nsub_col))
        mosaic = []
        for irow in range(nsub_rows):mosaic+=[ [str(irow)+str(icol) for icol in range(2*nsub_col)] ]
        axd = plt.figure(constrained_layout=True,figsize=figsize).subplot_mosaic(
            mosaic,width_ratios=np.tile([70.,30.],nsub_col),gridspec_kw={"bottom": 0.1,"top": 0.97,"left": 0.07,"right": 0.97})
        min_max_plot = [1e10,-1e10] 

    #Sampling
    nu_amp[comp_id] = np.empty([3,0],dtype=float)
    fap_guess = {}
    src_range = {}
    args['comp_mod'] = [comp_id]
    for iband,(low_theoband_edge,high_theoband_edge) in enumerate(zip(low_theoband_edges,high_theoband_edges)):
        irow = int(iband/nsub_col)
        icol = iband%nsub_col 

        #Sampled spectrum
        cond_in_theoband = (nu_in>=low_theoband_edge) & (nu_in<=high_theoband_edge)
        if np.sum(cond_in_theoband)>5:
            nu_band = nu_in[cond_in_theoband]
            samp_fit_dic['nu'][comp_id] = nu_band
            samp_fit_dic['flux'][comp_id] = flux_in[cond_in_theoband]
            samp_fit_dic['var'][comp_id] = var_in[cond_in_theoband]
            samp_fit_dic['err'][comp_id] = np.sqrt(samp_fit_dic['var'][comp_id])
            nu_min = samp_fit_dic['nu'][comp_id][0]
            nu_max = samp_fit_dic['nu'][comp_id][-1]
    
            #Bands sampled by higher component falling in current band
            if comp_id>1:
                condhigh_inband = ((nu_amp[comp_id_proc[-1]][0]>nu_min) & (nu_amp[comp_id_proc[-1]][0]<nu_max))    
                nhigh_inband = np.sum(condhigh_inband)
            else:nhigh_inband = 0

            #Periodogram calculation and search range
            #    - 'calc_perio_comp': frequency ranges within which periodograms are calculated for each component (in 1e-10 s-1)
            #                         setting 'calc_perio' to a larger range than 'src_perio' is useful to check for other peaks in the vicinity
            if src_perio_comp['mod'] is None:
                calc_perio_comp = [fixed_args['min_x_glob_perio'],fixed_args['max_x_glob_perio']]
                src_range[comp_id] = deepcopy(calc_perio_comp)
            elif (src_perio_comp['mod']=='slide'):          
                f_cen = np.mean(wig_freq_nu(comp_id,nu_band,freq_params_samp,args))   
                calc_perio_comp = [0.2*f_cen,1.8*f_cen]
                src_range[comp_id] = np.array([max([calc_perio_comp[0],f_cen-src_perio_comp['range'][0]]),min([f_cen+src_perio_comp['range'][1],calc_perio_comp[1]])])
 
            if ('up_bd' in src_perio_comp) and src_perio_comp['up_bd'] and (nhigh_inband>0):
                src_range[comp_id][1] = min([src_range[comp_id][1],np.max(nu_amp[comp_id_proc[-1]][1][condhigh_inband])])  
            if src_range[comp_id][1]<src_range[comp_id][0]:stop('Issue with periodogram search range')
        
            #Periodogram
            amp_guess,freq_guess,phi_guess,fap_guess[comp_id],ls,ls_freq,ls_pow,max_pow = wig_perio_gen(calc_perio_comp,src_range[comp_id],samp_fit_dic['nu'][comp_id],samp_fit_dic['flux'][comp_id],samp_fit_dic['err'][comp_id],args,plot=False)
        
            #Plot properties
            if plot_samp:
                min_max_plot = [min([min_max_plot[0],nu_min]),max([min_max_plot[1],nu_max])]
                min_nuplot= nu_min - 0.3
                max_nuplot= nu_max + 0.3  
                n_nu,nu_plot = def_wig_tab(min_nuplot,max_nuplot,args['dnu_HR'])
                x_range_plot = np.array([min_nuplot,max_nuplot]  )
                mean_nu =  np.mean(samp_fit_dic['nu'][comp_id])
                mean_wav =  np.mean(c_light/samp_fit_dic['nu'][comp_id])
            
            #Fit component if frequency is fixed or FAP is below threshold                     
            if (comp_id in comp_freqfixed) or (fap_guess[comp_id]<=args['fap_thresh']):
                
                #Model parameters
                #    - the generic parameter structure is used because we call the full chromatic model with constant amplitude and frequency 
                #    - the values and priors in 'p_start_gen' are not used
                p_temp_best = deepcopy(p_start_gen)    
        
                #Nullify other components
                #    - parameters are fixed by default
                for comp_null_id in [comp_loc for comp_loc in args['comp_ids'] if comp_loc!=comp_id]:
                    for pref,suf in zip(pref_names[comp_null_id],suf_names[comp_null_id]):p_temp_best[pref+str(comp_null_id)+suf+'_off'].value  = 0.
                    
                #Fitting amplitude and phase of current component
                for ideg in range(1,args['deg_Amp'][comp_id]+1):
                    p_temp_best['AmpGlob'+comp_str+'_c'+str(ideg)+'_off'].value  = 0.   
                amp_key = 'AmpGlob'+comp_str+'_c0_off'
                p_temp_best[amp_key].value  = amp_guess
                p_temp_best[amp_key].vary  = True  
                p_temp_best['Phi'+comp_str+'_off'].value  = phi_guess
                p_temp_best['Phi'+comp_str+'_off'].vary  = True          

                #Prior on amplitude  
                #    - sometimes the fit otherwise converges toward a flat line
                p_temp_best[amp_key].min = 0.5*amp_guess
                p_temp_best[amp_key].max = 10.*amp_guess
                
                #Fitting constant period for current component  
                #    - parameters are fixed by default             
                if comp_id not in comp_freqfixed: 
                    
                    #Nullify higher order frequency coefficients
                    for ideg in range(1,args['deg_Freq'][comp_id]+1):
                        p_temp_best['Freq'+comp_str+'_c'+str(ideg)+'_off'].value  = 0.   
                    
                    #Fix/fit value to peak periodogram frequency
                    #    - if fitted, frequency is still initialized to peak periodogram frequency
                    p_temp_best['Freq'+comp_str+'_c0_off'].value  = freq_guess   
                    if fix_freq2perio[comp_id]:p_temp_best['Freq'+comp_str+'_c0_off'].vary  = False 
                    else:                
                        p_temp_best['Freq'+comp_str+'_c0_off'].vary  = True  
        
                        #Prior set to the 'src_range' range 
                        p_temp_best['Freq'+comp_str+'_c0_off'].min = src_range[0]
                        p_temp_best['Freq'+comp_str+'_c0_off'].max = src_range[1]

                #Run fit over several iteration to converge, using the robust 'nelder' method
                args['idx_fit'] = np.ones(len(samp_fit_dic['nu'][comp_id]),dtype=bool)
                for it in range(args['nit']):
                    _,_ ,p_temp_best = call_lmfit(p_temp_best,samp_fit_dic['nu'][comp_id],samp_fit_dic['flux'][comp_id],np.array([samp_fit_dic['var'][comp_id]]),FIT_calc_wig_mod_nu,verbose=False ,fixed_args=args,maxfev = args['max_nfev'],method='nelder')

                #Determine uncertainties by running LM fit using Nelder-Mead solution as starting point
                _, merit,p_temp_best = call_lmfit(p_temp_best,samp_fit_dic['nu'][comp_id],samp_fit_dic['flux'][comp_id],np.array([samp_fit_dic['var'][comp_id]]),FIT_calc_wig_mod_nu,verbose=False ,fixed_args=args,maxfev = args['max_nfev'],method='leastsq')
                
                #Store best fit for variable parameters if current component is fitted
                if comp_id == args['comp_id_max']:
                    for par in fit_results['par_list']:
                        fit_results[par]['nu'] = np.append(fit_results[par]['nu'],mean_nu)
                        fit_results[par]['val'] = np.append(fit_results[par]['val'],[[p_temp_best[par].value],[p_temp_best[par].stderr]],axis=1)  
        
                #Plot models
                if plot_samp:             
                    best_mod_plot = calc_wig_mod_nu(nu_plot,p_temp_best,args)[0] 
                
                    #Adding contribution from current component over processed band
                    condHR_in_theoband = (args['nu_plot_glob']>=nu_min) & (args['nu_plot_glob']<=nu_max)  
                    samp_fit_dic['mod_plot_glob'][comp_id][condHR_in_theoband] += calc_wig_mod_nu(args['nu_plot_glob'][condHR_in_theoband],p_temp_best,args)[0]-1.  
      
                #Best-fit frequency
                #    - assuming the component approximates to a sine with constant period:
                # sin( 2pi*( sum(i=1,d+1 : c[i-1]*(w-wref)^i )  ) - Phi ) ~ sin(2pi(w-wref)*f_approx - Phi)    
                # f_approx = sum(i=1,d+1 : c[i-1]*(w-wref)^(i-1) ) 
                # f_approx = sum(i=0,d : c[i]*(w-wref)^i ) 
                # f_approx ~ < sum(i=0,d : c[i]*(w-wref)^i ) >
                best_freq = np.mean(wig_freq_nu(comp_id,nu_band,p_temp_best,args))                  
 
                #Store frequency and amplitude      
                nu_amp[comp_id] = np.append(nu_amp[comp_id],[[mean_nu],[best_freq],[np.abs(p_temp_best[amp_key].value)]],axis=1)                 
                        
            else:
                best_freq = None
                best_mod_plot = np.zeros(len(nu_plot))
                merit = {'fit':1.}
                
            #Residuals    
            samp_fit_dic['res'][comp_id] = 1. + samp_fit_dic['flux'][comp_id] - merit['fit']   
           
            #Plot of spectra and periodograms
            if plot_samp:
                fontsize = 20
                
                #Axis
                if irow==nsub_rows-1:plot_xlab = True
                else:plot_xlab = False
                if icol==0:plot_ylab = True
                else:plot_ylab = False
                    
                #Spectra
                idx_plot = str(irow)+str(2*icol)      
                var_plot = samp_fit_dic['flux'][comp_id]
                plot_sampling_spec(axd[idx_plot],x_range_plot,samp_fit_dic['nu'][comp_id],var_plot,nu_plot,best_mod_plot,mean_wav,mean_nu,args,fontsize=fontsize,plot_xlab=plot_xlab,plot_ylab=plot_ylab)
    
                #Periodogram plot
                idx_plot= str(irow)+str(2*icol+1)           
                plot_sampling_perio(fixed_args,axd[idx_plot],best_freq,src_range[comp_id],calc_perio_comp,ls,ls_freq,ls_pow,freq_guess,max_pow,fap_guess[comp_id],args['sampling_fap'],fontsize=fontsize,plot_xlab=plot_xlab,plot_ylab=plot_ylab)  
        
            #Store full spectra
            if count[comp_id]==0:
                for key in ['nu','res','var']:samp_fit_dic[key+'_all'][comp_id]  = deepcopy(samp_fit_dic[key][comp_id])   
                if (comp_id==1) and plot_samp:samp_fit_dic['flux_all'][comp_id]  = deepcopy(samp_fit_dic['flux'][comp_id])  
            else:
                for key in ['nu','res','var']:samp_fit_dic[key+'_all'][comp_id]  = np.append(samp_fit_dic[key+'_all'][comp_id] ,samp_fit_dic[key][comp_id]    )         
                if (comp_id==1) and plot_samp:samp_fit_dic['flux_all'][comp_id]  = np.append(samp_fit_dic['flux_all'][comp_id] ,samp_fit_dic['flux'][comp_id]    )   
                    
            #Empty data
            for key in ['nu','flux','res','var','err']:samp_fit_dic[key].pop(comp_id)       
                
            #Implement number of processed bands
            count[comp_id]+=1

    #Save figure
    if plot_samp:
        pref_plot = '_'
        for comp_loc in comp_id_proc+[comp_id]:
            if ishift_comp[comp_loc] is not None:pref_plot = '_Comp'+str(comp_loc)+'shift'+"{0:.1f}".format(ishift_comp[comp_loc])+pref_plot
            else:pref_plot = '_Comp'+str(comp_loc)+pref_plot
        plt.savefig(path_sampling_plot+'BandFit'+pref_plot+'minw'+"{0:.1f}".format(min_max_plot[0])+'_maxw'+"{0:.1f}".format(min_max_plot[-1])+'.png')                   
        plt.close() 
    count[comp_id] = 0   
    
    #-----------------------------------------------------------
    #Process final residuals
    #    - when last component has been processed
    if (comp_id==args['comp_id_max']):

        #Full model
        samp_fit_dic['mod_plot_glob']['all'] = np.ones(args['n_nu_HR'],dtype=float)
        for comp_loc in args['comp_ids']:
            samp_fit_dic['mod_plot_glob']['all']+=samp_fit_dic['mod_plot_glob'][comp_loc]-1.
         
        #Plot
        if plot_samp:
            plot_wig_glob(None,None,samp_fit_dic['nu_all'][1],samp_fit_dic['flux_all'][1],np.sqrt(samp_fit_dic['var_all'][1]),args['nu_plot_glob'],samp_fit_dic['mod_plot_glob'],None,None,samp_fit_dic['nu_all'][comp_id],samp_fit_dic['res_all'][comp_id],np.sqrt(samp_fit_dic['var_all'][comp_id]),args,min_max_plot,pref_plot,path_sampling_plot)
        
        #Empty global dictionaries 
        for key in ['nu_all','var_all','res_all']:samp_fit_dic[key].pop(comp_id) 
                
    return None



def plot_wig_glob(low_nu_fit,high_nu_fit,nu_fit,flux_fit,err_fit,nu_mod,mod_plot_glob,low_nu_res,high_nu_res,nu_res,flux_res,err_res,fixed_args,min_max_plot,pref_plot,path_sampling_plot,bin_spec=False,filter_mode=False):
    r"""**Wiggle plot: full spectrum**

    Plots full wiggle transmission spectrum (measured and model) before and after correction, with associated periodogram.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    plt.ioff()
    axd = plt.figure(constrained_layout=True,figsize=(100, 10)).subplot_mosaic(
          [['00','01'],['10','11']],width_ratios=np.tile([90.,10.],1),gridspec_kw={"bottom": 0.1,"top": 0.99,"left": 0.07,"right": 0.97})   
    fontsize=35       
    
    #Spectral tables
    x_fit = nu_fit
    x_mod = nu_mod
    x_range_plot = [min_max_plot[0]-0.3,min_max_plot[1]+0.3]
    x_res = nu_res
    for ax_key in ['00','10']:
        axd[ax_key].set_xlabel(r'$\nu$ (10$^{-10}$ s$^{-1}$)',fontsize=fontsize) 
        axd[ax_key].xaxis.set_major_locator(MultipleLocator(1))
        axd[ax_key].xaxis.set_major_formatter('{x:.0f}')
        axd[ax_key].xaxis.set_minor_locator(MultipleLocator(0.5))
    perio_range = [fixed_args['min_x_glob_perio'],fixed_args['max_x_glob_perio']]
        
    #-----------------------------------    
    
    #Fitted uncorrected spectrum
    axd['00'].plot(x_fit,flux_fit,linestyle='-',color='black',zorder=1,rasterized=fixed_args['rasterized'])
    rms = (flux_fit-1.).std()
    max_delta = np.max([ np.max(flux_fit-1.),np.max(1.-flux_fit) ])
    y_range_plot = [1.-max_delta*1.1,1.+max_delta*1.1]
    axd['00'].text(x_range_plot[0]+0.1*(x_range_plot[1]-x_range_plot[0]),
                   y_range_plot[1]-0.15*(y_range_plot[1]-y_range_plot[0]),
                   'RMS = '+"{0:.0f}".format(rms*1e6)+' ppm',verticalalignment='bottom', horizontalalignment='left',fontsize=25,zorder=4,color='green') 

    #Plot binned spectrum
    if bin_spec:plot_bin_spec(axd['00'],low_nu_fit,high_nu_fit,flux_fit,x_range_plot[0],x_range_plot[1],0.27/5.)

    #Plot full model and components
    #    - in bands where the component model was not fitted it remains set to zero (and the global model to one)
    axd['00'].plot(x_mod,mod_plot_glob['all'],linestyle='--',color='red',zorder=5,lw=2,rasterized=fixed_args['rasterized'])      
    if not filter_mode:
        for comp_loc in fixed_args['comp_mod']:
            axd['00'].plot(x_mod,mod_plot_glob[comp_loc],linestyle='-',color={1:'dodgerblue',2:'orange',3:'limegreen',4:'magenta'}[comp_loc],zorder=comp_loc,lw=1.5,rasterized=True) 

    #Reference level 
    axd['00'].plot(x_range_plot,[1.,1.],linestyle=':',color='black') 

    #Periodogram
    _,freq_guess,_,fap_guess_loc,ls,ls_freq,ls_pow,max_pow = wig_perio_gen(perio_range,perio_range,nu_fit,flux_fit,err_fit,fixed_args,plot=False)
    plot_sampling_perio(fixed_args,axd['01'],None,perio_range,perio_range,ls,ls_freq,ls_pow,freq_guess,max_pow,fap_guess_loc,fixed_args['sampling_fap'],log=fixed_args['perio_log'],fontsize=fontsize)         

    #-----------------------------------

    #Fitted corrected spectrum
    axd['10'].plot(x_res,flux_res,linestyle='-',color='black',zorder=0,rasterized=fixed_args['rasterized'])
    rms = (flux_res-1.).std()
    axd['10'].text(x_range_plot[0]+0.1*(x_range_plot[1]-x_range_plot[0]),
                   y_range_plot[1]-0.15*(y_range_plot[1]-y_range_plot[0]),
                   'RMS = '+"{0:.0f}".format(rms*1e6)+' ppm',verticalalignment='bottom', horizontalalignment='left',fontsize=25,zorder=4,color='green') 
    
    #Plot binned spectrum
    if bin_spec:plot_bin_spec(axd['10'],low_nu_res,high_nu_res,flux_res,x_range_plot[0],x_range_plot[1],0.27/5.)    
    
    #Reference level 
    axd['10'].plot(x_range_plot,[1.,1.],linestyle=':',color='black') 

    #Periodogram 
    _,freq_guess,_,fap_guess_loc,ls,ls_freq,ls_pow,max_pow = wig_perio_gen(perio_range,perio_range,nu_res,flux_res,err_res,fixed_args,plot=False)
    plot_sampling_perio(fixed_args,axd['11'],None,perio_range,perio_range,ls,ls_freq,ls_pow,freq_guess,max_pow,fap_guess_loc,fixed_args['sampling_fap'],log=fixed_args['perio_log'],fontsize=fontsize)         

    #-----------------------------------
         
    for ax_key in ['00','10']:
        axd[ax_key].set_xlim(x_range_plot)
        axd[ax_key].set_ylim(y_range_plot)   
        axd[ax_key].set_ylabel('Flux ratio',fontsize=fontsize)  
        axd[ax_key].tick_params('x',labelsize=fontsize)
        axd[ax_key].tick_params('y',labelsize=fontsize)
        
    plt.savefig(path_sampling_plot+'BandFit_All'+pref_plot+'minw'+"{0:.1f}".format(min_max_plot[0])+'_maxw'+"{0:.1f}".format(min_max_plot[-1])+'.png')                  
    plt.close()        

    return None






def plot_rms_wig(cen_ph_plot,rms_precorr,rms_postcorr,median_err,save_path):
    r"""**Wiggle plot: RMS**

    Plots global RMS of wiggle transmission spectra

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    plt.ioff()        
    fig, ax = plt.subplots(2, 1, figsize=(10, 9),squeeze=True,gridspec_kw={"bottom": 0.15,"top": 0.7,"left": 0.15,"right": 0.95, 'height_ratios':[2./3.,1./3.]}) 

    #Automatic range
    dx_range = np.max(cen_ph_plot)-np.min(cen_ph_plot)
    x_range_plot = [np.min(cen_ph_plot)-0.05*dx_range,np.max(cen_ph_plot)+0.05*dx_range]

    #------------------------------------------------------------------------------
    fontsize = 14

    #Automatic range
    sc_fact=1e6
    dy_range = np.max(sc_fact*rms_precorr)-np.min(sc_fact*median_err)
    y_range_plot = [np.min(sc_fact*median_err)-0.05*dy_range,np.max(sc_fact*rms_precorr)+0.05*dy_range]
    
    #RMS pre-correction
    ax[0].plot(cen_ph_plot,sc_fact*rms_precorr,linestyle='',color='red',zorder=0,marker='o',markersize=4,markerfacecolor='None') 
    ax[0].text(x_range_plot[0]+0.3*dx_range,y_range_plot[1]-0.1*dy_range,'RMS pre-corr. = '+"{0:.0f}".format(np.mean(sc_fact*rms_precorr)),verticalalignment='center',horizontalalignment='left',fontsize=fontsize-1,zorder=1,color='red') 
    
    #RMS post-correction
    ax[0].plot(cen_ph_plot,sc_fact*rms_postcorr,linestyle='',color='dodgerblue',zorder=0,marker='o',markersize=4,markerfacecolor='None')                     
    ax[0].text(x_range_plot[0]+0.3*dx_range,y_range_plot[1]-0.2*dy_range,'RMS post-corr. = '+"{0:.0f}".format(np.mean(sc_fact*rms_postcorr)),verticalalignment='center',horizontalalignment='left',fontsize=fontsize-1,zorder=1,color='dodgerblue') 

    #Median error
    ax[0].plot(cen_ph_plot,sc_fact*median_err,linestyle='',color='limegreen',zorder=0,marker='s',markersize=4,markerfacecolor='None')                     
    ax[0].text(x_range_plot[0]+0.3*dx_range,y_range_plot[1]-0.3*dy_range,'Median error = '+"{0:.0f}".format(np.mean(sc_fact*median_err)),verticalalignment='center',horizontalalignment='left',fontsize=fontsize-1,zorder=1,color='limegreen') 
    
    #Frame
    ax[0].set_ylabel('RMS (ppm)',fontsize=fontsize)  
    ax[0].set_xlim(x_range_plot)  
    ax[0].set_ylim(y_range_plot)
    ax[0].tick_params('x',labelsize=fontsize)
    ax[0].tick_params('y',labelsize=fontsize)

    #------------------------------------------------------------------------------    
    
    #Ratio RMS/error
    rms2err_precorr = rms_precorr/median_err
    rms2err_postcorr = rms_postcorr/median_err    

    #Automatic range
    dy_range = np.max(rms2err_precorr)-np.min(rms2err_postcorr)
    y_range_plot = [np.min(rms2err_postcorr)-0.05*dy_range,np.max(rms2err_precorr)+0.05*dy_range]

    #Ratio pre-correction
    ax[1].plot(cen_ph_plot,rms2err_precorr,linestyle='',color='red',zorder=0,marker='o',markersize=4,markerfacecolor='None') 
    ax[1].text(x_range_plot[0]+0.3*dx_range,y_range_plot[1]-0.1*dy_range,'RMS/<e> pre-corr. = '+"{0:.2f}".format(np.mean(rms2err_precorr)),verticalalignment='center',horizontalalignment='left',fontsize=fontsize-1,zorder=1,color='red') 
    
    #Ratio post-correction
    ax[1].plot(cen_ph_plot,rms2err_postcorr,linestyle='',color='dodgerblue',zorder=0,marker='o',markersize=4,markerfacecolor='None')                     
    ax[1].text(x_range_plot[0]+0.3*dx_range,y_range_plot[1]-0.3*dy_range,'RMS/<e> post-corr. = '+"{0:.2f}".format(np.mean(rms2err_postcorr)),verticalalignment='center',horizontalalignment='left',fontsize=fontsize-1,zorder=1,color='dodgerblue') 

    #Frame
    ax[1].set_ylabel('RMS/<e>',fontsize=fontsize) 
    ax[1].set_xlim(x_range_plot)  
    ax[1].set_ylim(y_range_plot)
    ax[1].tick_params('x',labelsize=fontsize)
    ax[1].tick_params('y',labelsize=fontsize)
    
    #------------------------------------------------------------------------------    
    
    #Frame
    ax[1].set_xlabel('Orbital phase',fontsize=fontsize) 
    plt.savefig(save_path+'RMS_all.png')                  
    plt.close()      

    return None













def FIT_wig_submod_coord_discont(param_in,x_in,args=None):
    r"""**Wiggle fit function: generic model for hyper-parameters**

    Calls corresponding model function for optimization

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    if isinstance(param_in,lmfit.parameter.Parameters):params={par:param_in[par].value for par in param_in}
    else:params=param_in        
    mod_exp = wig_submod_coord_discont(len(x_in),params,args)
    return mod_exp



def wig_submod_coord_discont(nexp,params,args): 
    r"""**Wiggle model function: generic model for hyper-parameters**

    Defines the following model for wiggle hyper-parameters
    
    :math:`f(t) = U + \alpha_\mathrm{x} \sin(az(t)) + \alpha_\mathrm{y} \cos(az(t)) + \alpha_\mathrm{z} \sin(alt(t))`
        
    where `U`, :math:`\alpha_\mathrm{x}`, :math:`\alpha_\mathrm{y}`, :math:`\alpha_\mathrm{z}` are different before/after the meridian but linked through continuity of values at the meridian (:math:`az = \pi`)
    
    .. math::
        U - \alpha_\mathrm{y}^\mathrm{east} + \alpha_\mathrm{z}^\mathrm{east} \sin(alt^\mathrm{mer}) &= U + dU - \alpha_\mathrm{y}^\mathrm{west} + \alpha_\mathrm{z}^\mathrm{west} \sin(alt^\mathrm{mer}) \\
        \alpha_\mathrm{y}^\mathrm{west} &= dU + \alpha_\mathrm{y}^\mathrm{east} + \sin(alt^\mathrm{mer}) (\alpha_\mathrm{z}^\mathrm{west} - \alpha_\mathrm{z}^\mathrm{east}) 
        
    and through continuity of derivative at the meridian (:math:`az = \pi`, :math:`dalt/dt = 0`)

    .. math::
        f'(t) &= az'(t) (\alpha_\mathrm{x} \cos(az(t)) - \alpha_\mathrm{y} \sin(az(t))) + \alpha_\mathrm{z} alt'(t) \cos(alt(t))    \\
        -\alpha_\mathrm{x}^\mathrm{east} az'(mer) &= -\alpha_\mathrm{x}^\mathrm{west} az'(mer) \\
        \alpha_\mathrm{x}^\mathrm{west} &= \alpha_\mathrm{x}^\mathrm{east}      
        
    If the guide star changes during the night, it changes the wiggle properties and breaks continuity.
    Exposure pre-shift (if the change occured before meridian crossing) or post-shift (if the change occured after meridian crossing) are then modelled differently.        
        
    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    par_name = args['par_name'] 
    cond_westmer = args['cond_westmer']
    cond_shift = args['cond_shift']
    mod = np.repeat(params[par_name+'off'],nexp)
    
    #Property kept stable over each phase
    if args['stable_pointpar'][par_name]:
        if True in cond_shift:mod[cond_shift] =      params[par_name+'doff_sh'].value 
    
    #Property varies with telescope coordinates
    else:        

        #Continuity in derivative
        dx_west = params[par_name+'dx_east']  
    
        #Continuity in value
        dy_west = params[par_name+'doff'] + params[par_name+'dy_east'] + args['z_mer']*(params[par_name+'dz_west']-params[par_name+'dz_east'])   
           
        #Telescope coordinates
        x_az = args['x_az']
        y_az = args['y_az']
        z_alt = args['z_alt']
    
        #Model
        cond_eastmer = args['cond_eastmer']
        if True in cond_eastmer:mod[cond_eastmer] +=                               params[par_name+'dx_east']*x_az[cond_eastmer] +  params[par_name+'dy_east']*y_az[cond_eastmer] +  params[par_name+'dz_east']*z_alt[cond_eastmer]
        if True in cond_westmer:mod[cond_westmer] += params[par_name+'doff']+                         dx_west*x_az[cond_westmer] +                     dy_west*y_az[cond_westmer] +  params[par_name+'dz_west']*z_alt[cond_westmer]
        if True in cond_shift:  mod[cond_shift]    = params[par_name+'doff_sh'] + params[par_name+'dx_shift']*x_az[cond_shift]   + params[par_name+'dy_shift']*y_az[cond_shift]   + params[par_name+'dz_shift']*z_alt[cond_shift]

    return mod  

# #Test function    
# def FIT_wig_corre(param_in,x_in,args=None):
#     if isinstance(param_in,lmfit.parameter.Parameters):params={par:param_in[par].value for par in param_in}
#     else:params=param_in    
#     mod_exp = params[args['par_name']+'a0'] + params[args['par_name']+'a1']*x_in + params[args['par_name']+'a2']*x_in*2. + params[args['par_name']+'a3']*x_in**3.
#     return mod_exp





def FIT_wig_amp_nu_poly(param_in,nu_in,args=None):
    r"""**Wiggle fit function: chromatic amplitude**

    Calls corresponding model function for optimization

    Args:
        TBD
    
    Returns:
        TBD
    
    """
    if isinstance(param_in,lmfit.parameter.Parameters):params={par:param_in[par].value for par in param_in}
    else:params=param_in    
    mod_exp = wig_amp_nu_poly(args['comp_id'],nu_in,params,args)
    return mod_exp


def wig_amp_nu_poly(comp_id,nu_in,params,args):
    r"""**Wiggle model function: chromatic amplitude**

    Defines the model for wiggle chromatic amplitude

    Args:
        TBD
    
    Returns:
        TBD
    
    """        
    coeff_pol_dec = [params['AmpGlob'+str(comp_id)+'_c'+str(ideg)+'_off'] for ideg in range(args['deg_Amp'][comp_id]+1)]      
    mod = np.poly1d(coeff_pol_dec[::-1])(nu_in-args['nu_ref'])
    
    #Chromatic model forced to remain positive
    #    - due to the degeneracy with the phase offset the model can be negative
    #      we thus force the model to null at nu>nu_cross, with Amp(nu>nu_cross) < 0 if Amp(nu_min) > 0 or Amp(nu>nu_cross) > 0 if Amp(nu_min) < 0
    cond_null = np.sign(mod[0])*mod < 0
    if True in cond_null:mod[np_where1D(cond_null)[0]:len(mod)+1] = 0.
    
    return mod 







def FIT_wig_freq_nu(param_in,nu_in,args=None):
    r"""**Wiggle fit function: chromatic frequency**

    Calls corresponding model function for optimization

    Args:
        TBD
    
    Returns:
        TBD
    
    """
    if isinstance(param_in,lmfit.parameter.Parameters):params={par:param_in[par].value for par in param_in}
    else:params=param_in    
    mod_exp = wig_freq_nu(args['comp_id'],nu_in,params,args)
    return mod_exp

def wig_freq_nu(comp_id,nu_in,params,args):
    r"""**Wiggle model function: chromatic frequency**

    Defines the model for wiggle chromatic frequency

    Args:
        TBD
    
    Returns:
        TBD
    
    """   
    coeff_pol_dec = [params['Freq'+str(comp_id)+'_c'+str(ideg)+'_off'] for ideg in range(args['deg_Freq'][comp_id]+1)]      
    mod_par = np.poly1d(coeff_pol_dec[::-1])(nu_in-args['nu_ref'])
    return mod_par  





def wig_intfreq_nu(comp_id,nu_in,params,args):
    r'''**Wiggle sub-function: integrated frequency**
    
    Defines the integral of the wiggle frequency. A sinusoid frequency is the rate-of-change (= derivative) of the sinusoid phase. 
    A given frequency profile `f(t)` needs to be integrated to compute the desired sinusoid phase (up to a constant additive term), from which one can then compute the desired waveform.
    
        - `<https://stackoverflow.com/questions/64958186/numpy-generate-sine-wave-signal-with-time-varying-frequency>`_
        - `<https://www.mathworks.com/matlabcentral/answers/217746-implementing-a-sine-wave-with-linearly-changing-frequency>`_

    Assuming the frequency profile varies as a polynomial of :math:`\nu`  

    .. math::    
        W(\nu) &= \sin(       \int_{0}^{\nu} 2 \pi f(\nu) d\nu  - \Phi_\mathrm{0} )    \\
                 &= \sin( 2 \pi \int_{0}^{\nu} \sum_{i=0}^{d}( a[i] (\nu-\nu_\mathrm{ref})^i) d\nu - \Phi_\mathrm{0} )    \\
                 &= \sin( 2 \pi \sum_{i=0}^{d}( \int_{0}^{\nu} a[i] (\nu-\nu_\mathrm{ref})^i d\nu ) - \Phi_\mathrm{0} ) \\
                 &= \sin( 2 \pi \sum_{i=0}^{d}( a[i] \frac{(\nu-\nu_\mathrm{ref})^{i+1}}{i+1} ) - \Phi )             \\
                 &= \sin( 2 \pi \sum_{i=1}^{d+1}( a[i-1] \frac{(\nu-\nu_\mathrm{ref})^i}{i} ) - \Phi ) \\
                 &= \sin( 2 \pi \sum_{i=0}^{d+1}( d[i] (\nu-\nu_\mathrm{ref})^i ) - \Phi ) \\
                 &= \sin( 2 \pi f_\mathrm{Int}(\nu) - \Phi )  
    
    with `d[0] = 0` et `d[i>0] = a[i-1]/i` in :math:`A^{-1}`.      
                
    We set :math:`W(\nu_\mathrm{ref}) = \sin( -\Phi )` so that :math:`\Phi` can be estimated in a given spectrum by taking the model fit at :math:`\nu = \nu_\mathrm{ref}`.
    
    Locally we can assume a constant frequency :math:`f_\mathrm{0}` so that :math:`a_\mathrm{0}` can be directly measured through
    
    .. math::     
        W(\nu) = \sin( 2 \pi a_\mathrm{0} (\nu-\nu_\mathrm{ref}) - \Phi ) = \sin( 2 \pi f_\mathrm{0} w - \Phi_\mathrm{0} ) 

    '''    
    coeff_pol_dec= [0.]+[params['Freq'+str(comp_id)+'_c'+str(ideg)+'_off']/(ideg+1.) for ideg in range(args['deg_Freq'][comp_id]+1)]      
    intfreq = np.poly1d(coeff_pol_dec[::-1])(nu_in-args['nu_ref'])  
    return intfreq






def FIT_calc_wig_mod_nu(param_in,nu_in,args=None):
    r"""**Wiggle fit function: global chromatic model**

    Calls corresponding model function for optimization

    Args:
        TBD
    
    Returns:
        TBD
    
    """
    if isinstance(param_in,lmfit.parameter.Parameters):params={par:param_in[par].value for par in param_in}
    else:params=param_in    
    mod_exp = calc_wig_mod_nu(nu_in,params,args)[0]
    return mod_exp

def calc_wig_mod_nu(nu_in,params,args):
    r"""**Wiggle model function: global chromatic model**

    Defines the chromatic model for the wiggles in a given exposure as

    .. math::  
        W(\nu) = 1 + A(\nu) \sin( 2 \pi f_\mathrm{Int}(\nu) - \Phi )


    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    comp_mod = {}
    full_mod = 1.
    for comp_id in args['comp_mod']:         
        
        #Amplitude
        amp_nu = wig_amp_nu_poly(comp_id,nu_in,params,args)
    
        #Integrated frequency
        intfreq_nu = wig_intfreq_nu(comp_id,nu_in,params,args)
    
        #Global component
        comp_mod[comp_id] = amp_nu*np.sin(2.*np.pi*intfreq_nu  - params['Phi'+str(comp_id)+'_off']  )          
        full_mod += comp_mod[comp_id]
        
    return full_mod, comp_mod







def FIT_calc_wig_mod_nu_t(param_in,nu_all,args=None):
    r"""**Wiggle fit function: global chromato-temporal model**

    Calls corresponding model function for optimization

    Args:
        TBD
    
    Returns:
        TBD
    
    """
    #In case param_in is defined as a Parameters structure, retrieve values and define dictionary
    if isinstance(param_in,lmfit.parameter.Parameters):params={par:param_in[par].value for par in param_in}
    else:params=param_in    
   
    #Calculate coordinate variations of chromatic parameters and offset
    calc_chrom_coord(params,args)
    
    #Join wiggle models over all fitted exposures
    mod_all = np.zeros(0,dtype=float)
    args_loc = deepcopy(args)
    for isub,(istart,iend) in enumerate(zip(args['iexp_bounds'][0],args['iexp_bounds'][1])):
  
        #High-resolution model of transmission wiggle  
        args_loc['icoord'] = isub
        trans_wig_mod_HR = calc_wig_mod_nu_t(args['nu_HR'],params,args_loc)[0]/args['weighted_wig_mod_HR']  

        #Wiggle model over current exposure table
        mod_exp=interp1d(args['nu_HR'],trans_wig_mod_HR,fill_value='extrapolate')(nu_all[istart:iend])     
    
        #Join to global table
        mod_all = np.append( mod_all , mod_exp   )
    
    return mod_all

def calc_wig_mod_nu_t(nu_in,params,args):
    r"""**Wiggle model function: global chromato-temporal model**
    
    The model for a given visit is defined as 

    .. math::   
        W(\nu,c(t),it) = 1 + A(\nu,c(t),it) \sin( 2 \pi f_\mathrm{Int}(\nu,c(t)) - \Phi(c(t)) )
          
    With `c(t)` the pointing coordinates of the telescope at time `t`.
          
    The model for the corresponding transmission spectrum is defined as

    .. math::    
        W_\mathrm{norm}(\nu,c(t),it) = \frac{W(\nu,c(t),it)}{< t , W(\nu,c(t),it-1) >}

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    comp_mod = {}
    full_mod = 1.
    for comp_id in args['comp_mod']: 
        comp_str = str(comp_id)
        icoord = args['icoord']
        
        #Chromatic amplitude at current time
        chrom_par = {'AmpGlob'+comp_str+'_c'+str(ideg)+'_off':args['AmpGlob'+comp_str+'_c'+str(ideg)+'_coord'][icoord] for ideg in range(args['deg_Amp'][comp_id]+1)}
        amp_nu_t = wig_amp_nu_poly(comp_id,nu_in,chrom_par,args)

        #Chromatic integrated frequency at current time
        chrom_par = {'Freq'+comp_str+'_c'+str(ideg)+'_off':args['Freq'+comp_str+'_c'+str(ideg)+'_coord'][icoord] for ideg in range(args['deg_Freq'][comp_id]+1)}
        intfreq_nu_t = wig_intfreq_nu(comp_id,nu_in,chrom_par,args)      

        #Global component
        comp_mod[comp_id] = amp_nu_t*np.sin(2.*np.pi*intfreq_nu_t  - args['Phi'+comp_str+'_coord'][icoord] )          
        full_mod += comp_mod[comp_id]

    return full_mod, comp_mod

def calc_chrom_coord(params,args):
    r"""**Wiggle model function: properties**

    Calculates coordinate variations of chromatic parameters (amplitude and frequency) and offset

    Args:
        TBD
    
    Returns:
        TBD
    
    """    
    args_loc = deepcopy(args)
    for comp_id in args_loc['comp_mod']: 
        comp_str = str(comp_id)
        
        #Offset
        args_loc['par_name'] = 'Phi'+comp_str+'_'
        args['Phi'+comp_str+'_coord'] = wig_submod_coord_discont(args_loc['nexp_list'],params,args_loc) 
   
        #Frequency
        for ideg in range(args_loc['deg_Freq'][comp_id]+1):       
            args_loc['par_name'] = 'Freq'+comp_str+'_c'+str(ideg)+'_'
            args['Freq'+comp_str+'_c'+str(ideg)+'_coord'] = wig_submod_coord_discont(args_loc['nexp_list'],params,args_loc)

        #Amplitude
        for ideg in range(args_loc['deg_Amp'][comp_id]+1):       
            args_loc['par_name'] = 'AmpGlob'+comp_str+'_c'+str(ideg)+'_'
            args['AmpGlob'+comp_str+'_c'+str(ideg)+'_coord'] = wig_submod_coord_discont(args_loc['nexp_list'],params,args_loc)    
    
    return None































'''
Fringing correction  
'''

def corr_fring(inst,gen_dic,data_inst,plot_dic,data_dic):
    r"""**Main fringing correction routine.**   

    Determines and applies correction for near-infrared spectrograph fringes.

    Args:
        TBD
    
    Returns:
        TBD
    
    """    
    print('   > Correcting spectra for fringing')
    
    #Calculating data
    if (gen_dic['calc_fring']):
        print('         Calculating data')    
    
    
        stop('Adapter selon la meme approche que wiggles; notamment scaler par cal.')
        print('reflechir a faire guess adapte automatiquement a chaque ordre')
        print('faire selection de modeles selon inst (GIARPS, STIS, ..)')
      

        #1D spectra
        if (data_inst['type']=='spec1D'):
            stop('Defringing TBD')
            
        #2D spectra
        #    - method from Guilluy+2020 : 
        #    - fringing affects most the red part of an order, least the blue part 
        #      by comparing the overlap between these two orders we can derive a fringing model for each order
        elif (data_inst['type']=='spec2D'):
    
            #Next order used for comparison
            #    - the next order is at index + 2 for ESPRESSO
            iord_next = 2 if inst=='ESPRESSO' else 1

            #Initialize model parameters
            #    - see model definition in the corresponding function
            p_use = Parameters()

            #Fit parameters
            p_use.add_many(('fnom',np.nan, True  , None , None, None),
                            ('pol_1',np.nan, True  , None , None, None),
                            ('pol_2',np.nan, False , None , None, None),   
                            ('pol_3',np.nan, False  , None , None, None),
                            ('pol_4',np.nan, False  , None , None, None),
                            ('sc_att',np.nan, False  , None , None, None),
                            ('w_att',np.nan, False  , None , None, None),
                            ('w_sin',np.nan, False  , None , None, None),
                            ('wper_sin',np.nan, False  , None , None, None))

            #Optional arguments to be passed to the fit functions
            fixed_args={'use_cov':gen_dic['use_cov']}

            #Degrees and coefficient names of polynomial function
            #    - the number in the coefficient name corresponds to its degree
            #    - coefficients can be defined in any order in p_use
            fixed_args['coeff_pol']={}     
            for par in p_use:
                if 'pol_' in par:
                    temp=par.split('_')[1]
                    deg=int(temp.split('_')[0])
                    fixed_args['coeff_pol'][deg]=par
            fixed_args['deg_pol']=len(fixed_args['coeff_pol'])
            
            #Fit function
            fit_func=FIT_fring_mod             
            
            
            
                            
            #------------------------------------------------------------
            
            #Process each visit
            for ivisit,vis in enumerate(data_inst['visit_list']):
                data_vis=data_inst[vis]  

                #Process each exposure
                proc_DI_data_paths_new = gen_dic['save_data_dir']+'Corr_data/Fring/'+inst+'_'+vis+'_'
                for iexp in range(data_vis['n_in_visit']):
                    data_exp = np.load(data_vis['proc_DI_data_paths']+str(iexp)+'.npz',allow_pickle=True)['data'].item() 
    
                    #Plot dictionary
                    if (plot_dic['fring_corr']!=''):                    
                        dic_fit = {}
                        dic_fit['exp_ord_defring'] = np.zeros(data_inst['nord'],dtype=bool)
                        for key in ['idx_ov_ord_def','fring_mod','data_fit','cov_fit']:dic_fit[key] = np.zeros(data_inst['nord'], dtype=object)                    
    
                    #Identify orders to be defringed
                    #    - condition is that the order overlaps with the requested range
                    #    - even if the spectral tables change with exposures we set the condition with the first exposure
                    #    - we first remove the last order, as it cannot be corrected with this method
                    #      for ESPRESSO we remove the last two orders, as two successive orders are defined over the same range 
                    low_edge_bins = data_exp['edge_bins'][:,0:-1]
                    high_edge_bins = data_exp['edge_bins'][:,1::]
                    if iexp==0:
                        idx_ord_defring = np.arange(data_inst['nord'])
                        if inst=='ESPRESSO':
                            idx_ord_defring = idx_ord_defring[:-2]
                        else:
                            idx_ord_defring = idx_ord_defring[:-1]
                        idx_ord_defring = idx_ord_defring[(low_edge_bins[idx_ord_defring,0]<gen_dic['fring_range'][1]) & (high_edge_bins[idx_ord_defring,-1]>gen_dic['fring_range'][0])]
                        if len(idx_ord_defring)==0:stop('No orders overlap with the requested range')

                    #Process each order to be defringed
                    for iord in idx_ord_defring:

                        #Overlapping bins between current and next order
                        idx_ov_loc_ord = np_where1D( (low_edge_bins[iord,:]<high_edge_bins[iord+iord_next,-1]) & (high_edge_bins[iord,:]>low_edge_bins[iord+iord_next,0]) & data_exp['cond_def'][iord,:] )         
                        idx_ov_next_ord = np_where1D( (low_edge_bins[iord+iord_next,:]<high_edge_bins[iord,-1]) & (high_edge_bins[iord+iord_next,:]>low_edge_bins[iord,0]) & data_exp['cond_def'][iord+iord_next,:] )                                
                        if (len(idx_ov_loc_ord)>0) and (len(idx_ov_next_ord)>0):

                            #Resample next order over current order table
                            #    - required only if exposures have not been already resampled on a common table
                            if (not data_vis['comm_sp_tab']):
                                
                                #The full overlapping ranges are used for the resampling, as the function needs contiguous bins to work
                                idx_ov_loc_ord = np.arange(idx_ov_loc_ord[0],idx_ov_loc_ord[-1]+1)
                                idx_ov_next_ord = np.arange(idx_ov_next_ord[0],idx_ov_next_ord[-1]+1)                               
                                edge_bins_ord = np.append(data_exp['edge_bins'][iord,idx_ov_loc_ord],data_exp['edge_bins'][iord,idx_ov_loc_ord[-1]+1])                 
                                edge_bins_next_ord = np.append(data_exp['edge_bins'][iord+iord_next,idx_ov_next_ord],data_exp['edge_bins'][iord+iord_next,idx_ov_next_ord[-1]+1])                 
                                sp_ov_next,cov_ov_next = bind.resampling(edge_bins_ord, edge_bins_next_ord, data_exp['flux'][iord+iord_next,idx_ov_next_ord] , cov = data_exp['cov'][iord+iord_next][:,idx_ov_next_ord], kind=gen_dic['resamp_mode'])

                            else:
                                sp_ov_next  = data_exp['flux'][iord+iord_next,idx_ov_next_ord]
                                cov_ov_next = data_exp['cov'][iord+iord_next][:,idx_ov_next_ord]
                                
                            #Defined bins in overlap
                            sp_ov_loc = data_exp['flux'][iord,idx_ov_loc_ord]
                            cond_def_ov = (~np.isnan(sp_ov_next)) & (~np.isnan(sp_ov_loc))
                            if True in cond_def_ov:
                                
                                #Ratio between current (most affected by fringing in its overlapping red part) and next (least affected by fringing in its overlapping blue part) order over overlap       
                                ratio_ov,cov_ratio_ov = bind.div(sp_ov_loc , data_exp['cov'][iord][:,idx_ov_loc_ord], sp_ov_next , cov_ov_next)
                        
                                #-------------------------------------------------------------------------------  
    
                                #Set parameter guess for current order
                                p_use['fnom'].value = 1.                            
                                p_use['pol_1'].value =-1e-5                            
                                p_use['pol_2'].value =0                              
                                p_use['pol_3'].value =0.                            
                                p_use['pol_4'].value =0.                            
                                p_use['sc_att'].value =5e9                            
                                p_use['w_att'].value =1094.                            
                                p_use['w_sin'].value =1090.                            
                                p_use['wper_sin'].value =1e6
    
                                #Fitting
                                wav_ov = data_exp['cen_bins'][iord]
                                args = {}
                                args['idx_fit'] = idx_ov_loc_ord[cond_def_ov]
                                result, merit,p_best= call_lmfit(p_use,wav_ov,ratio_ov,cov_ratio_ov,fit_func,verbose=False,fixed_args=fixed_args)
                                if (plot_dic['fring_corr']!=''):
                                    dic_fit['exp_ord_defring_all'][iord]=True
                                    dic_fit['idx_ov_ord_def_all'][iord] = idx_ov_loc_ord[cond_def_ov]
                                    dic_fit['fring_mod_all'][iord] = merit['fit']
                                    dic_fit['data_fit_all'][iord] = ratio_ov
                                    dic_fit['sig_fit_all'][iord] = np.sqrt(cov_ratio_ov[0,:])                                   
                            
                                #Correcting current order over full spectral range
                                #    - correction is set to 1 for undefined pixels (the full table must be given to bind so as not to mess the covariance)
                                cond_def_curr = data_exp['cond_def'][iord]
                                corr_mod = np.repeat(1.,data_vis['nspec'])
                                corr_mod[cond_def_curr] = 1./fit_func(p_best,data_exp['cen_bins'][iord,cond_def_curr],args=fixed_args)
                                data_exp['flux'][iord],data_exp['cov'][iord] = bind.mul_array(data_exp['flux'][iord] , data_exp['cov'][iord],corr_mod)

                    #-------------------------------------------------------------------------------------------------                    
                        
                    #Saving modified data and updating paths
                    np.savez_compressed(proc_DI_data_paths_new+str(iexp),data = data_exp,allow_pickle=True) 
    
                    #Save independently correction data
                    if (plot_dic['fring_corr']!=''): 
                        np.savez_compressed(gen_dic['save_data_dir']+'Corr_data/Fring/'+inst+'_'+vis+'_'+str(iexp)+'_add',data=dic_fit,allow_pickle=True)
                data_vis['proc_DI_data_paths']=proc_DI_data_paths_new
                
    #Updating path to processed data and checking it has been calculated
    else:
        for vis in data_inst['visit_list']:  
            data_vis=data_inst[vis]
            data_vis['proc_DI_data_paths']=gen_dic['save_data_dir']+'Corr_data/Fring/'+inst+'_'+vis+'_'     
            check_data({'path':data_vis['proc_DI_data_paths']+str(0)},vis=vis) 

    return None






def FIT_fring_mod(param,x,args=None):
    r"""**Fringing fit function**

    Calls corresponding model function for optimization

    Args:
        TBD
    
    Returns:
        TBD
    
    """
    ymodel=fring_mod(param,x,args)[0]
    return ymodel

def fring_mod(param,x,args=None):
    r"""**Fringing model function**
    
    Defined as the product of
    
        - nominal flux : should be close to 1 as we model the ratio between two overlapping orders
        - 4th order polynomial function, function of wavelength: model low-frequency variations, such as the blaze if not removed
        - amplifying sinusoidal perturbation, function of wavenumber : periodic in wavenumber, amplitude is a gaussian increasing with wavenumber

    Args:
        TBD
    
    Returns:
        TBD
    
    """
    #Wavenumber (A-1)
    k = 1./x
    
    #Polynomial variation (no unit)
    #    - 'coeff_pol' is a dictionary that contains the coefficients defined in p_use
    #      keys are the coefficients degrees, values their names
    #      they can be defined in any order, but must be defined above degree one
    #    - degrees can be missing
    #    - the polynomial is assumed to be a modulation around the nominal flux, and is defined as 1 + p1*x + ...
    coeff_pol=args['coeff_pol']
    deg_max=args['deg_pol']
    coeff_pol_dec=[1.]+[param[coeff_pol[ideg]] if ideg in coeff_pol else 0. for ideg in range(1,deg_max+1)]
    pol_func=np.poly1d(coeff_pol_dec[::-1])
    pol_mod =  pol_func(x)   
    
    #Amplifying sinusoidal amplitude (no unit) 
    k_att = 1./param['w_att'].value
    amp_att = np.exp(-param['sc_att'].value*((k-k_att)**2.))

    #Sinusoidal variation (no unit)
    k_sin = 1./param['w_sin'].value 
    kper_sin = 1./param['wper_sin'].value 
    sin_att = (1.+   amp_att*np.sin(2.*np.pi*(k - k_sin)/kper_sin))
    
    #Complete model
    fnom = param['fnom'].value 
    mod =  fnom*pol_func(x)*sin_att
    
    return mod , fnom*pol_mod , sin_att














