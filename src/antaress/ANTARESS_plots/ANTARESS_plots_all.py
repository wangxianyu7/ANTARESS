#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import importlib
import os as os_system
from lmfit import Parameters
from copy import deepcopy
from math import pi,cos,sin,sqrt
from pathos.multiprocessing import Pool
import numpy.ma as ma
import bindensity as bind
from itertools import product as it_product
from matplotlib.ticker import MultipleLocator,MaxNLocator
import matplotlib as mpl
import copy
from astropy.io import fits
import glob
import imageio
from ..ANTARESS_general.constant_data import c_light
from ..ANTARESS_general.minim_routines import call_lmfit
from ..ANTARESS_general.utils import closest,stop,np_where1D,closest_Ndim,np_interp,init_parallel_func,is_odd,dataload_npz,air_index,gen_specdopshift,import_module
from ..ANTARESS_plots.utils_plots import custom_axis,autom_tick_prop,stackrel,scaled_title,autom_range_ext,plot_shade_range
from ..ANTARESS_conversions.ANTARESS_binning import resample_func,calc_bin_prof,weights_bin_prof
from ..ANTARESS_analysis.ANTARESS_inst_resp import calc_FWHM_inst,return_pix_size
from ..ANTARESS_analysis.ANTARESS_model_prof import gauss_intr_prop,dgauss,cust_mod_true_prop,voigt
from ..ANTARESS_corrections.ANTARESS_detrend import detrend_prof_gen
from ..ANTARESS_corrections.ANTARESS_interferences import def_wig_tab,calc_chrom_coord,calc_wig_mod_nu_t
from ..ANTARESS_grids.ANTARESS_coord import calc_pl_coord_plots,calc_pl_coord,calc_rv_star_HR,frameconv_skyorb_to_skystar,frameconv_skystar_to_skyorb,get_timeorbit,\
    calc_zLOS_oblate,frameconv_star_to_skystar,calc_tr_contacts
from ..ANTARESS_corrections.ANTARESS_calib import cal_piecewise_func
from ..ANTARESS_grids.ANTARESS_star_grid import get_LD_coeff,calc_CB_RV,calc_RVrot,calc_Isurf_grid,calc_st_sky
from ..ANTARESS_grids.ANTARESS_occ_grid import occ_region_grid,sub_calc_plocc_spot_prop,retrieve_spots_prop_from_param, calc_spotted_tiles


def ANTARESS_plot_functions(system_param,plot_dic,data_dic,gen_dic,coord_dic,theo_dic,data_prop,glob_fit_dic,mock_dic,nbook_dic,custom_plot_settings):
    print()
    print('-----------------------------------')
    print('Plots')  
    print('-----------------------------------')

    import matplotlib
    if gen_dic['non_int_back']:matplotlib.use('Agg')	
    import matplotlib.pyplot as plt    

    #Default instruments and visits to plot
    plot_dic['visits_to_plot'] = {inst:[vis for vis in data_dic[inst]['visit_list']] for inst in data_dic['instrum_list']}

    #Retrieve default settings
    plot_settings={}
    from ..ANTARESS_plots.ANTARESS_plot_settings import ANTARESS_plot_settings
    ANTARESS_plot_settings(plot_settings,plot_dic,gen_dic,data_dic,glob_fit_dic,theo_dic)   
    
    #Overwrite with user settings
    if custom_plot_settings!='':import_module(gen_dic['save_dir']+custom_plot_settings).ANTARESS_plot_settings(plot_settings,plot_dic,gen_dic,data_dic,glob_fit_dic,theo_dic)
    
    #Overwrite with notebook settings
    if ('plots' not in nbook_dic):nbook_dic['plots'] = {}
    for key_plot in plot_settings:
        if (key_plot in nbook_dic['plots']):
            if key_plot=='prop_DI_ordin':
                plot_settings[key_plot] = nbook_dic['plots'][key_plot]
                for plot_prop in nbook_dic['plots'][key_plot]:plot_settings['prop_DI_'+plot_prop].update(nbook_dic['plots']['prop_DI_'+plot_prop])
            elif key_plot=='prop_Intr_ordin':
                plot_settings[key_plot] = nbook_dic['plots'][key_plot]
                for plot_prop in nbook_dic['plots'][key_plot]:plot_settings['prop_Intr_'+plot_prop].update(nbook_dic['plots']['prop_Intr_'+plot_prop])
            else:plot_settings[key_plot].update(nbook_dic['plots'][key_plot])

    ################################################################################################################    
    #%% Generic plot functions
    ################################################################################################################ 

    '''
    Calculating high-resolution model of planet-occulted regions along a chosen orbital range
        - the HR phase table is common to all models
        - the coordinates and stellar surface properties calculated here use the nominal system properties given as input of the pipeline, unless otherwise specified
    '''
    if (plot_dic['occulted_regions']!='') or any('map_' in key for key in list(plot_dic.keys())) or (plot_dic['prop_Intr']!='') or (plot_dic['prop_DI']!='') or (plot_dic['input_LC']!='') or (plot_dic['pca_ana']!=''): 

        #Contacts
        contact_phases={}
        for pl_loc in gen_dic['studied_pl']:
            contact_phases[pl_loc]=calc_tr_contacts(data_dic['DI']['system_prop']['achrom'][pl_loc][0],system_param[pl_loc],plot_dic['stend_ph'],system_param['star'])
            system_param[pl_loc]['T14_num'] = (contact_phases[pl_loc][3]-contact_phases[pl_loc][0])*system_param[pl_loc]['period']
            print('Numerical T14['+str(pl_loc)+']='+"{0:.6f}".format(system_param[pl_loc]['T14_num']*24.)+' h')                  

            #Stellar mass derived from orbital motion of main planets
            print('Numerical Kp_orb='+"{0:.2f}".format(system_param[pl_loc]['Kp_orb'])+' km/s (Mstar = '+"{0:.3f}".format(system_param[pl_loc]['Mstar_orb'])+' Msun)')  
    
        #Function to calculate coordinates of planets and occulted regions in a given visit
        #    - in achromatic mode
        def calc_occ_plot(data_bin,theo_dic_loc,inst,vis,genpar_instvis,param_loc,args,system_param_loc,iband=0,par_list = ['rv','CB_RV','mu','xp_abs','r_proj','y_st','lat']):

            #Generate high-resolution time table covering all planet transits   
            min_bjd = 1e100
            max_bjd = -1e100
            stend_ph = 1.3
            if vis=='binned':coord_vis = data_bin['coord'] 
            else:coord_vis = coord_dic[inst][vis]
            for pl_loc in gen_dic['studied_pl']:      
                min_bjd = np.min([min_bjd,coord_vis[pl_loc]['Tcenter']+stend_ph*contact_phases[pl_loc][0]*system_param[pl_loc]['period']])        
                max_bjd = np.max([max_bjd,coord_vis[pl_loc]['Tcenter'] +stend_ph*contact_phases[pl_loc][3]*system_param[pl_loc]['period']])   
            bjd_HR=min_bjd+ ((max_bjd-min_bjd)/(plot_dic['nph_HR']-1.))*np.arange(plot_dic['nph_HR']) - 2400000.

            #-------------------------------------
            #High-resolution planet model
            #-------------------------------------

            coord_pl_in={}
            theo_dic_loc['d_oversamp']={}
            system_prop_loc = deepcopy(data_dic['DI']['system_prop'])
            cond_occ_HR = np.zeros(plot_dic['nph_HR'],dtype=bool)
            for pl_loc in gen_dic['studied_pl']:

                #High-resolution phase table
                phase_pl = get_timeorbit(pl_loc,coord_vis,bjd_HR,system_param_loc[pl_loc],None)[1]   

                #Overwrite default values     
                if ('lambda_rad__pl'+pl_loc in genpar_instvis):lamb_name = 'lambda_rad__pl'+pl_loc+'__IS'+inst+'_VS'+vis 
                else:lamb_name = 'lambda_rad__pl'+pl_loc 
                if (lamb_name in param_loc):system_param_loc[pl_loc]['lambda_rad'] = param_loc[lamb_name]             
                if ('inclin_rad__pl'+pl_loc in param_loc):system_param_loc[pl_loc]['inclin_rad']=param_loc['inclin_rad__pl'+pl_loc]     
                if ('aRs__pl'+pl_loc in param_loc):system_param_loc[pl_loc]['aRs']=param_loc['aRs__pl'+pl_loc]  
                if 'RpRs__pl'+pl_loc in param_loc:
                    system_prop_loc['achrom'][pl_loc][0]=param_loc['RpRs__pl'+pl_loc] 
                    theo_dic_loc['Ssub_Sstar_pl'][pl_loc],theo_dic_loc['x_st_sky_grid_pl'][pl_loc],theo_dic_loc['y_st_sky_grid_pl'][pl_loc],r_sub_pl2=occ_region_grid(system_prop_loc['achrom'][pl_loc][0],theo_dic_loc['nsub_Dpl'][pl_loc])  
                    system_prop_loc['achrom']['cond_in_RpRs'][pl_loc] = [(r_sub_pl2<system_prop_loc['achrom'][pl_loc][0]**2.)]
                
                #Calculate coordinates
                #    - start/end phase have been set to None if no oversampling is requested, in which case start/end positions are not calculated
                xp_HR,yp_HR,_,Dprojplanet_HR,_,_,_,_,ecl_pl_HR = calc_pl_coord(system_param_loc[pl_loc]['ecc'],system_param_loc[pl_loc]['omega_rad'],system_param_loc[pl_loc]['aRs'],system_param_loc[pl_loc]['inclin_rad'],phase_pl,system_prop_loc['achrom'][pl_loc][0],system_param_loc[pl_loc]['lambda_rad'],system_param_loc['star'])
            
                #Coordinates and properties of planet-occulted regions
                coord_pl_in[pl_loc] = {'ecl':ecl_pl_HR,'cen_pos':np.vstack((xp_HR,yp_HR)),'phase':phase_pl}
                
                #Keep exposures where at least one planet transits
                #    - coordinates are normalized by Rstar, corresponding to the equatorial radius (and thus largest for an oblate star), so that the condition below is always conservative 
                cond_occ_HR |= Dprojplanet_HR <= (1.+data_dic['DI']['system_prop']['RpRs_max'][pl_loc])
        
            #Planet-occulted properties
            coord_pl_in['nph_HR'] =  np.sum(cond_occ_HR)   
            bjd_HR = bjd_HR[cond_occ_HR]
            args['inst'] = inst
            args['vis'] = vis
            for pl_loc in gen_dic['studied_pl']:
                coord_pl_in[pl_loc]['cen_pos'] = coord_pl_in[pl_loc]['cen_pos'][:,cond_occ_HR]
                coord_pl_in[pl_loc]['phase'] = coord_pl_in[pl_loc]['phase'][cond_occ_HR]     
                coord_pl_in[pl_loc]['ecl'] = coord_pl_in[pl_loc]['ecl'][cond_occ_HR] 
            surf_prop_dic, surf_prop_dic_spot, surf_prop_dic_common = sub_calc_plocc_spot_prop(['achrom'],args,par_list,gen_dic['studied_pl'],system_param_loc,theo_dic_loc,system_prop_loc,param_loc,coord_pl_in,range(coord_pl_in['nph_HR']))
            theo_HR_prop_plocc = surf_prop_dic['achrom']
            theo_HR_prop_plocc['nph_HR'] = coord_pl_in['nph_HR']
            for pl_loc in gen_dic['studied_pl']:
                theo_HR_prop_plocc[pl_loc].update(coord_pl_in[pl_loc])
                
                #Reduce to chosen band
                cond_def_HR = ~np.isnan(theo_HR_prop_plocc[pl_loc]['Ftot'][iband])
                theo_HR_prop_plocc[pl_loc]['phase']=theo_HR_prop_plocc[pl_loc]['phase'][cond_def_HR]
                for par in list(set(['rv','mu','lat','lon','x_st','y_st','xp_abs','r_proj','Rot_RV','CB_RV']+par_list)):
                    if par in theo_HR_prop_plocc[pl_loc]:theo_HR_prop_plocc[pl_loc][par]=theo_HR_prop_plocc[pl_loc][par][iband,cond_def_HR] 
            
            return theo_HR_prop_plocc
    
    













    '''
    Sub-functions for 2D map
    '''
    if any('map_' in key for key in list(plot_dic.keys())):

        def doppler_track_plots(key_track,line_mask,rest_frame,col_loc,cond_track,cond_range,lw_mod,ls_loc,pl_list,pl_ref,theo_HR,line_range,iexp_plot,iexp_range,reverse_2D,data_type,x_range_loc,y_range_loc):
            r"""**Orbital and stellar tracks.**
            
            Plots the Doppler track (time vs RV or wavelength) of the planet orbital trajectory or transit chord.

            Args:
                TBD
            
            Returns:
                TBD
            
            """

            #Masks
            for pl_loc in pl_list:
                
                #Spectral line ranges
                #    - from planet rest frame to planet line range frame, or from surface rest frame to planet line range frame
                if cond_range and (line_mask is not None):range_dopshift = gen_specdopshift(line_range)[:,None]      

                #Model track in their rest frame
                #    - independent of the planets, but defined for each planet to use a common structure in plot functions
                if key_track==rest_frame:
                    if cond_range: 
                        
                        #Spectral space
                        if (line_mask is not None):   
                            
                            #Line position
                            lines_HR = np.tile(line_mask,(theo_HR['nph_HR'],1))
                            
                            #Minimum/maximum position of each line over the chosen phase range
                            #    - constant in planet rest frame
                            min_lines_HR = line_mask
                            max_lines_HR = line_mask
                            
                            #Lower/upper boundary of excluded spectral range, for each line
                            #    - common to all lines in rv but not in wavelength
                            low_lines_HR = line_mask[:,None] * range_dopshift[0,:]  
                            high_lines_HR = line_mask[:,None] * range_dopshift[1,:]                                
                            
                            #Minimum lower/maximum upper boundary of excluded spectral range, over all lines
                            min_low_lines_HR = np.min(low_lines_HR,axis=1)
                            max_high_lines_HR = np.max(high_lines_HR,axis=1) 
                            
                        #RV space
                        else:
                            
                            #Minimum lower/maximum upper boundary of excluded rv range 
                            min_low_lines_HR = line_range[0]
                            max_high_lines_HR = line_range[1]                
                    
                #High-resolution orbital or surface RV model in star rest frame (km/s)
                #    - must be shorter than an exposure duration for exposures with exclusion to be plotted properly
                #    - for each planet
                #    - for each requested line (by default atmospheric mask line) in spectral mode  
                #    - shift from the planet or surface to the star rest frame
                elif rest_frame=='star':

                    #Spectral space
                    #    - dimensions nHR x nlines
                    if line_mask is not None:
                        if key_track=='pl':main_dopshift = gen_specdopshift(theo_HR[pl_loc]['rv'],v_s = theo_HR[pl_loc]['v'])  
                        elif key_track=='surf':main_dopshift = gen_specdopshift(theo_HR[pl_loc]['rv'])  
                        
                        #Line position
                        lines_HR = line_mask*main_dopshift[:,None]
                        
                        #Minimum/maximum position of each line over the chosen phase range
                        min_lines_HR = np.min(lines_HR,axis=0)
                        max_lines_HR = np.max(lines_HR,axis=0) 
                    
                        #Excluded planetary ranges
                        if cond_range:
                            
                            #Lower/upper wavelength boundaries of line ranges for each line
                            low_lines_HR = line_mask[:,None]*main_dopshift[:,None]*range_dopshift
                            high_lines_HR = line_mask[:,None]*main_dopshift[:,None]*range_dopshift                     
                            
                            #Minimum lower/maximum upper boundary of spectral line range over all lines
                            min_low_lines_HR = np.min(low_lines_HR,axis=1)
                            max_high_lines_HR = np.max(high_lines_HR,axis=1)
                         
                    #RV space
                    elif cond_range:
                            
                            #Minimum lower/maximum upper boundary of rv line range 
                            range_star_HR = line_range[:,None] + theo_HR[pl_loc]['rv']
                            max_high_lines_HR = range_star_HR[0]
                            max_high_lines_HR = range_star_HR[1]

                #Order high-resolution curves
                if reverse_2D:w_sorted=theo_HR[pl_loc]['phase'].argsort()
                else:w_sorted=theo_HR[pl_ref]['rv'].argsort() 
    
                #In rv space
                if (data_type == 'CCF'):                            
    
                    #High-resolution model
                    if cond_track:   
                        if (rest_frame=='star'):rv_mod_loc = theo_HR[pl_loc]['rv']
                        elif (key_track==rest_frame):rv_mod_loc = np.repeat(0.,theo_HR['nph_HR'])
                        if reverse_2D:plt.plot(theo_HR[pl_loc]['phase'][w_sorted],rv_mod_loc[w_sorted],color=col_loc,linestyle=ls_loc,lw=lw_mod,zorder=10) 
                        else:plt.plot(rv_mod_loc[w_sorted],theo_HR[pl_loc]['phase'][w_sorted] ,color=col_loc,linestyle=ls_loc,lw=lw_mod,zorder=10)                              
    
                    #Excluded ranges for each/all planetary mask line
                    if cond_range: 
                        if reverse_2D:plt.fill_between(theo_HR[pl_loc]['phase'][w_sorted],max_high_lines_HR[w_sorted], max_high_lines_HR[w_sorted],zorder=9,color='white',alpha=0.2)
                        else:plt.fill_betweenx(theo_HR[pl_loc]['phase'][w_sorted],max_high_lines_HR[w_sorted], max_high_lines_HR[w_sorted],zorder=9,color='white',alpha=0.2)
    
                #In spectral space
                #    - for each studied line in plot range
                elif ('spec' in data_type): 
    
                    #High-resolution model 
                    if cond_track: 
                        if reverse_2D:
                            cond_lines_in = (max_lines_HR>=y_range_loc[0]) & (min_lines_HR<=y_range_loc[1])  
                            lines_HR_plot = lines_HR[:,cond_lines_in][w_sorted]
                            for iline in range(np.sum(cond_lines_in)):plt.plot(theo_HR[pl_loc]['phase'][w_sorted],lines_HR_plot[:,iline],color=col_loc,linestyle=ls_loc,lw=lw_mod,zorder=10,alpha=1)
                        else:  
                            cond_lines_in = (max_lines_HR>=x_range_loc[0]) & (min_lines_HR<=x_range_loc[1]) 
                            lines_HR_plot = lines_HR[:,cond_lines_in][w_sorted]
                            for iline in range(np.sum(cond_lines_in)):plt.plot(lines_HR_plot[:,iline],theo_HR[pl_loc]['phase'][w_sorted],color=col_loc,linestyle=ls_loc,lw=lw_mod,zorder=10,alpha=1) 
    
                    #Line ranges for each line
                    if cond_range:  
    
                        #Create condition for high-resolution points within exposures without exclusion
                        #    - by setting to false those points without exposures with exclusion, so that they are not masked
                        cond_exp_excl = np.ones([len(line_mask),theo_HR['nph_HR']],dtype=bool)
                        if iexp_range is not None:
                            for iexp in np.intersect1d(iexp_range,iexp_plot):
                                sub_cond_HR = (theo_HR[pl_loc]['phase']>=coord_dic[inst][vis]['st_ph'][iexp]) & (theo_HR[pl_loc]['phase']<=coord_dic[inst][vis]['end_ph'][iexp])
                                if True in sub_cond_HR:cond_exp_excl[:,sub_cond_HR]=False

                        if reverse_2D:cond_lines_in = (max_high_lines_HR>=y_range_loc[0]) & (min_low_lines_HR<=y_range_loc[1]) 
                        else:cond_lines_in = (max_high_lines_HR>=x_range_loc[0]) & (min_low_lines_HR<=x_range_loc[1]) 
    
                        wlow_lines_HR_plot = low_lines_HR[cond_lines_in][:,w_sorted]
                        whigh_lines_HR_plot = high_lines_HR[cond_lines_in][:,w_sorted]
                        cond_exp_excl = cond_exp_excl[cond_lines_in][:,w_sorted]
                        wlow_lines_HR_plot = ma.masked_where(cond_exp_excl, wlow_lines_HR_plot)
                        whigh_lines_HR_plot = ma.masked_where(cond_exp_excl, whigh_lines_HR_plot) 
                        
                        if reverse_2D:
                            for iline in range(np.sum(cond_lines_in)):
                                plt.fill_between(theo_HR[pl_loc]['phase'][w_sorted],wlow_lines_HR_plot[iline,:], whigh_lines_HR_plot[iline,:],zorder=9,color=col_loc,alpha=0.3)
                        else:                                  
                            for iline in range(np.sum(cond_lines_in)):
                                plt.fill_betweenx(theo_HR[pl_loc]['phase'][w_sorted], wlow_lines_HR_plot[iline,:], whigh_lines_HR_plot[iline,:],zorder=9,color=col_loc,alpha=0.3)

            return None


        def sub_2D_map(plot_mod,save_res_map,plot_options):
            r"""**2D map.**
            
            Generic function to plot 2D map of flux or flux ratios as a function of time and spectral dimension.
            
             - it is advised to plot spectral maps as png due to their weight.
            
            Args:
                TBD
            
            Returns:
                None
            
            """            
           
            #Options
            sc_fact=10**plot_options['sc_fact10']            

            if plot_mod in ['map_Res_prof_clean_pl_est','map_Res_prof_clean_sp_est','map_Res_prof_unclean_sp_est','map_Res_prof_unclean_pl_est',
                        'map_Res_prof_clean_sp_res','map_Res_prof_clean_pl_res','map_Res_prof_unclean_sp_res','map_Res_prof_unclean_pl_res']:
        
                #Defining whether we are plotting the planet-occulted or spotted profiles and if they are clean or uncleaned
                supp_name = plot_mod.split('_')[4]
                corr_plot_mod = plot_mod.split('_')[3]

            #Plotting separate visits on different plots because of the possible overlap between exposures
            for inst in np.intersect1d(data_dic['instrum_list'],list(plot_options['visits_to_plot'].keys())):             
                print('   - Instrument :',inst)
    
                #Data to plot
                maink_list = []
                data_list = []
                plot_options['plot_pre']=None
                maink_list=['post']
                data_list=['all']     
        
                #Spectral options
                if ('spec' in gen_dic['type'][inst]): 
    
                    #HITRAN telluric lines
                    wave_tell_lines = get_tell_lines(plot_options['plot_tell_HITRANS'],gen_dic)    
    
                #Generic init function
                data_type_gen,data_mode,data_type,add_txt_path,txt_aligned=sub_plot_prof_init(plot_mod,plot_options,inst)        

                #Frame properties
                title_name=''   

                #Plot for each visit
                for ivis,vis in enumerate(np.intersect1d(list(data_dic[inst].keys())+['binned'],plot_options['visits_to_plot'][inst])):                 
                    print('     - Visit :',vis)
                    data_inst=data_dic[inst]
                    data_vis = data_inst[vis]
                    
                    #Data
                    # data_com = dataload_npz(data_vis['proc_com_data_paths'])   
                    fixed_args_loc = {}
                    pl_ref,txt_conv,iexp_plot,iexp_orig,prof_fit_vis,fit_results,data_path_all,rest_frame,data_path_dic,nexp_plot,inout_flag,path_loc,iexp_mast_list,nord_data,data_bin = sub_plot_prof_dir(inst,vis,plot_options,data_mode,'Map',add_txt_path,plot_mod,txt_aligned,data_type,data_type_gen)
                    
                    #Order list
                    order_list = plot_options['orders_to_plot'] if len(plot_options['orders_to_plot'])>0 else range(nord_data) 
                    idx_sel_ord = order_list
                    
                    #Frame
                    if plot_options['aligned']:title_name='Aligned '+title_name
                    xt_str={'input':'heliocentric','star':'star','surf':'surface','pl':'planet'}[rest_frame]
                    ordi_olap = False
                    if 'map' in plot_mod: 
                        ordi_name = plot_options['dim_plot'] 
                        if ('bin' not in plot_mod) and ordi_name in ['xp_abs','r_proj']:
                            transit_prop_nom = dataload_npz(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis)['achrom'][0]                       
                            ordi_olap=True
                    else:
                        ordi_name = 'phase'
                    time_title={
                            'phase':'Orbital phase',
                            'xp_abs':'Distance from normal',
                            'r_proj':'Distance from center'}[ordi_name]
                    if plot_options['x_range'] is not None:x_range_loc = plot_options['x_range']
                    if (data_type == 'CCF'):
                        sp_title='Velocity in '+xt_str+' rest frame (km s$^{-1}$)'
                    elif ('spec' in data_type):
                        if plot_options['sp_var'] == 'nu' :
                            if plot_options['x_range'] is None:sp_range_loc = [c_light/9000.,c_light/3000.]
                            sp_title = r'$\nu$ in '+xt_str+' rest frame (10$^{-10}$s$^{-1}$)'  
                        elif plot_options['sp_var'] == 'wav' :
                            if plot_options['x_range'] is None:sp_range_loc = [3000.,9000.] 
                            sp_title = r'Wavelength in '+xt_str+' rest frame (A)'                                            
                    if plot_options['reverse_2D']:
                        x_title=time_title      
                        y_title=sp_title 
                    else:    
                        x_title=sp_title       
                        y_title=time_title  

                    #Colors    
                    cmap_2D = copy.copy(plt.get_cmap(plot_options['cmap'])) 

                    #Process selected ranges and orders
                    nord_proc = len(idx_sel_ord)
                    if nord_proc==0:stop('No orders left')
                    if vis=='binned':nspec = data_inst['nspec']
                    else:nspec = data_vis['nspec']
                    dim_exp_proc = [nord_proc,nspec]
                    dim_all_proc = [nexp_plot]+dim_exp_proc
                    # cen_bins_com = data_com['cen_bins'][idx_sel_ord]
                    # edge_bins_com = data_com['edge_bins'][idx_sel_ord] 

                    #Initializing tables   
                    low_sp_map = np.zeros(dim_all_proc)*np.nan
                    high_sp_map = np.zeros(dim_all_proc)*np.nan
                    var_map = np.zeros(dim_all_proc)*np.nan
                    cond_def_map = np.zeros(dim_all_proc,dtype=bool)
                    low_ordi_tab = np.zeros(nexp_plot)*np.nan
                    high_ordi_tab = np.zeros(nexp_plot)*np.nan

                    #High-resolution surface RV model 
                    #    - achromatic
                    #    - rv of stellar surface region in star rest frame
                    params = deepcopy(system_param['star'])
                    params.update({'rv':0.,'cont':1.})  
                    if plot_options['theoRV_HR']:
                        theo_HR_prop_plocc = calc_occ_plot(data_bin,deepcopy(theo_dic),inst,vis,{},params,{},deepcopy(system_param))
                    if plot_options['theoRV_HR_align']:
                        system_param_align = deepcopy(system_param)
                        system_param_align[pl_ref]['lambda_rad'] = 0.
                        theo_HR_prop_plocc_align = calc_occ_plot(data_bin,deepcopy(theo_dic),inst,vis,{},params,{},system_param_align)
                        
                    #High-resolution orbital phase model
                    #    - rv of planet in star rest frame
                    if plot_options['theoRVpl_HR']:
                        theo_HR_pl = {}
                        min_ph_HRpl=-0.3
                        max_ph_HRpl=0.3        
                        dph_HRpl=0.0001
                        nph_HRpl=int((max_ph_HRpl-min_ph_HRpl)/dph_HRpl)
                        dph_HRpl=(max_ph_HRpl-min_ph_HRpl)/(nph_HRpl-1.)
                        ph_tab_HRpl=min_ph_HRpl+dph_HRpl*np.arange(nph_HRpl) 
                        theo_HR_pl['nph_HR'] = nph_HRpl
                        for pl_loc in gen_dic['studied_pl']:
                            theo_HR_pl[pl_loc]={}
                            theo_HR_pl[pl_loc]['phase'] = ph_tab_HRpl
                            rv_HR,v_HR=calc_pl_coord(system_param[pl_loc]['ecc'],system_param[pl_loc]['omega_rad'],system_param[pl_loc]['aRs'],system_param[pl_loc]['inclin_rad'],ph_tab_HRpl,None,None,None,rv_LOS=True,omega_p=system_param[pl_loc]['omega_p'])[6:8]     
                            theo_HR_pl[pl_loc]['rv']=rv_HR*system_param['star']['RV_conv_fact']
                            theo_HR_pl[pl_loc]['v']=v_HR*system_param['star']['RV_conv_fact']  

                    #Processing exposures
                    for isub,(iexp,iexp_or,data_path_exp) in enumerate(zip(iexp_plot,iexp_orig,data_path_all)):
                        
                        #Upload data
                        if data_path_exp is not None:data_exp = dataload_npz(data_path_exp)

                        #PC-based noise model
                        if (plot_mod=='map_pca_prof'):
                            var_map[isub] = data_exp['flux']*sc_fact
                            cond_def_map[isub] = ~np.isnan(var_map[isub])   
                            
                        #Data
                        else:     
                            
                            if plot_mod in ['map_Res_prof_clean_pl_est','map_Res_prof_clean_sp_est','map_Res_prof_unclean_sp_est','map_Res_prof_unclean_pl_est',
                                            'map_Res_prof_clean_sp_res','map_Res_prof_clean_pl_res','map_Res_prof_unclean_sp_res','map_Res_prof_unclean_pl_res']:
                                
                                cond_def_map[isub] = data_exp['cond_def']
                                #Retrieving flux for these regions
                                if '_est' in plot_mod:
                                    var_map[isub] = data_exp[corr_plot_mod+'_'+supp_name+'_flux'] 
                                #Building residuals
                                elif '_res' in plot_mod:
                                    data_exp_est = dataload_npz(gen_dic['save_data_dir']+'Spot_Loc_estimates/'+plot_options['mode_loc_data_corr']+'/'+inst+'_'+vis+'_'+str(iexp)) 
                                    var_map[isub] = data_exp['flux'] - data_exp_est[corr_plot_mod+'_'+supp_name+'_flux']
                            
                            elif plot_mod in ['map_BF_Res_prof', 'map_BF_Res_prof_re']:
                                cond_def_map[isub] = data_exp['cond_def_fit']
                                
                                flux_2_use = data_exp['flux']
                                if plot_mod == 'map_BF_Res_prof_re':
                                    raw_prof_loc = gen_dic['save_data_dir']+'Res_data/'+inst+'_'+vis+'_'+str(iexp)
                                    raw_prof = dataload_npz(raw_prof_loc)
                                    flux_2_use -= raw_prof['flux']
                                
                                var_map[isub] = flux_2_use
                            
                            elif plot_mod in ['map_Intr_prof_est','map_Intr_prof_res']: 


                                #Check that model exists for in-transit profiles 
                                if (gen_dic[inst][vis]['idx_exp2in'][iexp_or]==-1.) or \
                                  ((gen_dic[inst][vis]['idx_exp2in'][iexp_or]>-1) and \
                                        (plot_options['cont_only'] or \
                                        ((plot_options['line_model']=='fit') and (iexp in prof_fit_vis)) or \
                                        ((plot_options['line_model']=='rec') and (iexp in prof_fit_vis['idx_est_loc'])))): 
                               
                                    #Estimates for intrinsic stellar profiles
                                    #    - models for local stellar profiles are scaled as F_intr(w,t,v) = F_res(w,t,v)/(1 - LC_theo(band,t))
                                    #      where loc_flux_scaling = 1 - LC_theo
                                    if (plot_mod=='map_Intr_prof_est'):                                                                        
                                        if data_dic['Intr']['plocc_prof_type']=='Res':
                                            data_scaling  = dataload_npz(data_vis['scaled_Intr_data_paths']+str(iexp_or))
                                            loc_flux_scaling_plot = dataload_npz(data_vis['scaled_Intr_data_paths']+str(iexp_or))['loc_flux_scaling'] 
                                        for iord in range(dim_exp_proc[0]):                                
                                            var_map[isub,iord] = data_exp['flux'][iord] 
                                            if (data_dic['Intr']['plocc_prof_type']=='Res') and (not data_scaling['null_loc_flux_scaling']):var_map[isub,iord]/=loc_flux_scaling_plot(data_exp['cen_bins'][iord])    
                                    
                                    #Residual maps
                                    #    - see details in pc_analysis()
                                    #    - out-of-transit residual profiles correspond to :
                                    # F_res(w,t out,v) = ( Fstar(w,v) - F(w,t,v) )*Cref(wband,v) = - Pert(w,t,v)*Cref(wband,v)
                                    #      where Pert correspond to systematic deviations from the master disk-integrated profile
                                    #    - in-transit residual profiles must be set to a comparable flux level, through:
                                    # F_res(w,t in, v) = (F_intr(w,t,v) - F_intr(cont,t,v))*(1 - LC_theo(band,t))/LC_theo(band,t) 
                                    #                  = - Pert(w,t,v)*Cref(wband,v)      
                                    #      this can be seen as scaling F_intr by the brightness ratio between intrinsic spectra and disk-integrated spectra, with (1 - LC_theo(band,t) the scaling between intrinsic and local, and LC_theo(band,t) the scaling between local and disk-integrated
                                    elif (plot_mod=='map_Intr_prof_res'):
                                
                                        #In-transit profile
                                        if (inout_flag[isub]=='in'):
                                            data_scaling  = dataload_npz(data_vis['scaled_Intr_data_paths']+str(iexp_or))
                                            
                                            #Exosure is effectively in-transit
                                            if not data_scaling['null_loc_flux_scaling']:
                                                if not plot_options['cont_only']:data_exp_est = dataload_npz(gen_dic['save_data_dir']+'Loc_estimates/'+plot_options['mode_loc_data_corr']+'/'+inst+'_'+vis+'_'+str(iexp)) 
                                                intr_flux_est = np.zeros(dim_exp_proc)*np.nan 
                                                intr2res_sc = np.zeros(dim_exp_proc)*np.nan 
                                                for iord in range(nord_proc):
    
                                                    #Conversion from intrinsic to differential flux levels
                                                    loc_flux_scaling_exp_ord =  data_scaling['loc_flux_scaling'](data_exp['cen_bins'][iord])
                                                    intr2res_sc[iord] = loc_flux_scaling_exp_ord/(1. - loc_flux_scaling_exp_ord)                                                
    
                                                    #Correct for local continuum level only
                                                    if plot_options['cont_only']: 
    
                                                        #Broadband continuum profile
                                                        #    - for spectral data, if continuum was measured
                                                        if ('spec' in data_type) and (plot_options['st_cont'] is not None):
                                                            cont_func_dic = dataload_npz(gen_dic['save_data_dir']+'Stellar_cont_'+plot_options['st_cont']+'/'+inst+'_'+vis+'/St_cont')['cont_func_dic']
                                                            intr_flux_est[iord] = cont_func_dic(data_exp['cen_bins'][iord])
                                                            
                                                        #Mean intrinsic continuum level                              
                                                        else:
                                                            intr_flux_est[iord] = data_dic['Intr'][inst][vis]['mean_cont'][iord]
      
                                                    #Correct for full local stellar profile
                                                    else:
                                                        
                                                        #Model intrinsic profiles
                                                        if data_dic['Intr']['plocc_prof_type']=='Intr': 
                                                            intr_flux_est[iord] = data_exp_est['flux'][iord]
                                                        
                                                        #Conversion from model differential to intrinsic profiles
                                                        elif data_dic['Intr']['plocc_prof_type']=='Res':          
                                                            intr_flux_est[iord] = data_exp_est['flux'][iord]/loc_flux_scaling_exp_ord
        
                                                #Differential profiles
                                                var_map[isub] =  (data_exp['flux'] - intr_flux_est)*intr2res_sc

                                            #Exosure is effectively out-transit
                                            else:var_map[isub]=data_exp['flux']                                                
                                                
                                        #Out-of-transit differential profile
                                        else:var_map[isub]=data_exp['flux']

                                #Plot scaling
                                var_map[isub]*=sc_fact
                                cond_def_map[isub] = ~np.isnan(var_map[isub])                             
    
                            else:
                                var_map[isub] = sc_fact*data_exp['flux']
                                cond_def_map[isub] = data_exp['cond_def']

                        #Additional tables
                        low_edges = data_exp['edge_bins'][:,0:-1]
                        high_edges = data_exp['edge_bins'][:,1::]
                        if plot_options['sp_var'] == 'nu' :  
                            low_sp_map[isub] = c_light/low_edges
                            high_sp_map[isub] = c_light/high_edges                     
                        elif plot_options['sp_var'] == 'wav' :
                            low_sp_map[isub] = low_edges
                            high_sp_map[isub]= high_edges

                    ### End of exposure processing
         
                    #Ordina tables in the visit along chosen dimension
                    if 'map' in plot_mod: 
                        if ('bin' in plot_mod):
                            low_ordi_tab=data_bin['st_bindim']
                            high_ordi_tab=data_bin['end_bindim']  
                        else:                        
                            if ordi_name=='phase':     
                                low_ordi_tab=coord_dic[inst][vis][pl_ref]['st_ph'][iexp_orig]
                                high_ordi_tab=coord_dic[inst][vis][pl_ref]['end_ph'][iexp_orig]
                            elif ordi_name in ['xp_abs','r_proj']: 
                                transit_prop_nom = dataload_npz(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis)['achrom'][0]                           
                                low_ordi_tab =  transit_prop_nom[ordi_name+'_range'][iexp_plot,0]
                                high_ordi_tab = transit_prop_nom[ordi_name+'_range'][iexp_plot,1]  
                    else:
                        low_ordi_tab=coord_dic[inst][vis][pl_ref]['st_ph'][iexp_orig]
                        high_ordi_tab=coord_dic[inst][vis][pl_ref]['end_ph'][iexp_orig]
   
                   
                    #Order ordina tables by increasing values
                    mid_phase_tab = 0.5*(low_ordi_tab+high_ordi_tab)
                    isort=mid_phase_tab.argsort() 
                    low_ordi_tab = low_ordi_tab[isort]
                    high_ordi_tab = high_ordi_tab[isort]
                    var_map = var_map[isort]
                    cond_def_map = cond_def_map[isort]
                    low_sp_map = low_sp_map[isort]
                    high_sp_map=high_sp_map[isort] 
                    if 'bin' not in plot_mod:iexp_orig = np.array(iexp_orig)[isort] 
              
                    #Modify ordina tables to avoid overlap if necessary
                    #    - we shift exposures so that they are plotted successively, thus losing their absolute coordinates butavoiding overlaps
                    if ordi_olap:
                        low_ordi_tab_ref = deepcopy(low_ordi_tab)
                        high_ordi_tab_ref = deepcopy(high_ordi_tab)
                        delta_ordi_tab = high_ordi_tab[0:-1]-low_ordi_tab[1::]
                        for iexp_olap,delta_exp in zip( range(1,nexp_plot),delta_ordi_tab):
                            if delta_exp>0:
                                low_ordi_tab[iexp_olap::]+=delta_exp
                                high_ordi_tab[iexp_olap::]+=delta_exp
                                   
                    ordi_range = np.array([np.min(low_ordi_tab),np.max(high_ordi_tab)])
                    if plot_options['verbose']:print('      range for '+ordi_name+' table:',"{0:.5f}".format(ordi_range[0]),' ; ',"{0:.5f}".format(ordi_range[1]))
                    
                    #Processing each order 
                    for iord in order_list:

                        #Data to be plotted
                        low_ordi_tab_loc = deepcopy(low_ordi_tab)
                        high_ordi_tab_loc = deepcopy(high_ordi_tab)
                        low_sp_loc = low_sp_map[:,iord,:]
                        high_sp_loc = high_sp_map[:,iord,:]
                        cond_def_sp = np.any(cond_def_map[:,iord,:],axis=0)
                        var_loc = var_map[:,iord,:]

                        #Automatic ranges detection
                        sp_range = np.array([np.nanmin(low_sp_loc[:,cond_def_sp]),np.nanmax(high_sp_loc[:,cond_def_sp])])

                        #Axis range definition
                        #    - reverse_2D = True : Horizontal axis is phase dimension, vertical axis is spectral dimension (common to all visits)
                        #      reverse_2D = False: Horizontal axis is spectral dimension (common to all visits), vertical axis is phase dimension  
                        #    - we limit tables to the plot ranges if defined, to lighten the plots
                        cond_ph_in=[True]
                        cond_sp_in=[True]
                        if plot_options['reverse_2D']:
                            if (inst in plot_options['x_range_all']) and (vis in plot_options['x_range_all'][inst]):
                                x_range_loc=np.array(plot_options['x_range_all'][inst][vis])
                                cond_ph_in=(high_ordi_tab>x_range_loc[0]) & (low_ordi_tab<x_range_loc[-1])   
                            else:x_range_loc=ordi_range
                            if plot_options['y_range'] is not None:
                                y_range_loc = deepcopy(plot_options['y_range'])
                                cond_sp_in=(high_sp_loc>y_range_loc[0]) & (low_sp_loc<y_range_loc[-1])
                            else:y_range_loc=sp_range                           
                        else:
                            if plot_options['x_range'] is not None:
                                x_range_loc=deepcopy(plot_options['x_range'])
                                cond_sp_in=(high_sp_loc>x_range_loc[0]) & (low_sp_loc<x_range_loc[-1])
                            else:x_range_loc=sp_range  
                            if (inst in plot_options['y_range_all']) and (vis in plot_options['y_range_all'][inst]) and (plot_options['y_range_all'][inst][vis] is not None):
                                y_range_loc=np.array(plot_options['y_range_all'][inst][vis])
                                cond_ph_in=(high_ordi_tab>y_range_loc[0]) & (low_ordi_tab<y_range_loc[-1])   
                            else:y_range_loc=ordi_range

                        #Limiting tables to plot ranges, if imposed
                        cond_plot = True
                        if False in cond_ph_in:
                            if (True not in cond_ph_in):cond_plot = False
                            else:
                                var_loc = var_loc[cond_ph_in,:]
                                low_ordi_tab_loc = low_ordi_tab_loc[cond_ph_in]
                                high_ordi_tab_loc=high_ordi_tab_loc[cond_ph_in]
                                low_sp_loc=low_sp_loc[cond_ph_in,:]
                                high_sp_loc=high_sp_loc[cond_ph_in,:]
                        if False in cond_sp_in: 
                            if (True not in cond_sp_in):cond_plot = False
                            else:
                                cond_sp_in = np.sum(cond_sp_in,axis=0,dtype=bool) 
                                var_loc = var_loc[:,cond_sp_in]
                                low_sp_loc=low_sp_loc[:,cond_sp_in]
                                high_sp_loc=high_sp_loc[:,cond_sp_in] 

                        #------------------------------------                            
                        if cond_plot:

                            #Axis                           
                            plt.ioff()  
                            if plot_options['reverse_2D']: plt.figure(figsize=plot_options['fig_size'])
                            else:plt.figure(figsize=(10, 9))                                                
                            ax=plt.gca() 
                            fig = plt.gcf()  

                            #Normalisation
                            #    - to set profiles to comparable levels for the plot, to a mean unity 
                            if plot_options['norm_prof'] and ('Intr' in plot_mod):
                                mean_flux = data_dic['Intr'][inst][vis]['mean_cont'][iord] 
                                var_loc/=mean_flux
                             
                            #Automatic ranges detection
                            if (inst in plot_options['v_range_all']) and (vis in plot_options['v_range_all'][inst]) and (plot_options['v_range_all'][inst][vis] is not None):
                                v_range=np.array(plot_options['v_range_all'][inst][vis])*sc_fact
                            else:
                                v_range=np.array([np.nanmin(var_loc),np.nanmax(var_loc)])
                                if (data_type == 'CCF') and (plot_options['verbose']):print('      range for color table:',"{0:.5f}".format(v_range[0]),' ; ',"{0:.5f}".format(v_range[1]))

                            #Plot data for each exposure   
                            for iexp,(var_exp,low_phase_exp,high_phase_exp) in enumerate(zip(var_loc,low_ordi_tab_loc,high_ordi_tab_loc)):
                                low_sp_exp=low_sp_loc[iexp]
                                high_sp_exp = high_sp_loc[iexp]
                                cond_def_exp = ~np.isnan(var_exp) 
                                
                                #Normalisation
                                #    - applied to individual exposures for disk-integrated profiles
                                if plot_options['norm_prof'] and ('Intr' not in plot_mod):
                                    cond_def_scal = True
                                    if ('DI' in plot_mod) and (len(data_dic['DI']['scaling_range'])>0):
                                        cond_def_scal=False 
                                        for bd_int in data_dic['DI']['scaling_range']:cond_def_scal |= (low_sp_exp>=bd_int[0]) & (high_sp_exp<=bd_int[1])                                     
                                    cond_def_scal&=cond_def_exp
                                    dcen_bins = high_sp_exp - low_sp_exp
                                    mean_flux=np.nansum(var_exp[cond_def_scal]*dcen_bins[cond_def_scal])/np.sum(dcen_bins[cond_def_scal]) 
                                    var_exp/=mean_flux

                                #Defined pixels
                                ng_exp = 2*np.sum(cond_def_exp)
                                if ng_exp>0:
                                
                                    #Creating continuous grid in X and Y
                                    #    - artificial, empty bins added in between the upper boundary of a bin and the lower boundary of the next bin, are set to nan
                                    #    - see online how grid must be defined for pcolormesh                                              
                                    if plot_options['reverse_2D']: #Xplot = phase, Yplot = spectral, i = Yplot, j = Xplot
                                        x_grid = np.hstack( (np.tile(low_phase_exp,[ng_exp,1]),np.tile(high_phase_exp,[ng_exp,1])) ) 
                                        y_grid = np.zeros([ng_exp,2])*np.nan
                                        y_grid[0::2,:]=low_sp_exp[cond_def_exp]
                                        y_grid[1::2,:]=high_sp_exp[cond_def_exp]                              
                                    else:                #Xplot = spectral, Yplot = phase, i = Yplot, j = Xplot
                                        x_grid = np.zeros([2,ng_exp])*np.nan
                                        x_grid[:,0::2]=low_sp_exp[cond_def_exp]
                                        x_grid[:,1::2]=high_sp_exp[cond_def_exp]    
                                        y_grid = np.vstack( (np.repeat(low_phase_exp,ng_exp),np.repeat(high_phase_exp,ng_exp)) )  
                                    val_grid = np.zeros(ng_exp-1)*np.nan                                       
                                    val_grid[0::2] = var_exp[cond_def_exp]
                                      
                                    #Mask nan values
                                    val_grid_mask=[np.ma.masked_invalid(val_grid)]
                                    cmap = plt.get_cmap(cmap_2D)       
                                    cmap.set_bad(color = 'black')       
                                    
                                    #Plot current row or column
                                    plt.pcolormesh(x_grid, y_grid, val_grid_mask,cmap=cmap,vmin=v_range[0], vmax=v_range[1],rasterized=True,zorder=0,linewidth=0.5,edgecolors='face')
                           
                                else:print('No data in order ',iord,' for exposure ',iexp)

                            #------------------------------------                            
    
                            #Levels
                            if plot_options['plot_zermark']:
                                if plot_options['reverse_2D']:
                                    if ordi_name == 'phase':plt.plot(x_range_loc,[0,0],color='black',linestyle='--',lw=0.8,zorder=10)
                                    if (data_type == 'CCF') :plt.plot([0,0],y_range_loc,color='black',linestyle='--',lw=0.8,zorder=10)
                                else:
                                    if ordi_name == 'phase':plt.plot([0,0],y_range_loc,color='black',linestyle='--',lw=0.8,zorder=10)
                                    if (data_type == 'CCF') :plt.plot(x_range_loc,[0,0],color='black',linestyle='--',lw=0.8,zorder=10)                            

                            #Shaded ranges
                            if (inst in plot_options['shade_ranges']) and (vis in plot_options['shade_ranges'][inst]):
                                if not plot_options['reverse_2D']:
                                    plot_shade_range(ax,plot_options['shade_ranges'][inst][vis],x_range_loc,None,mode='span',compl=True,zorder=100,alpha=0.4,facecolor='dodgerblue')
                                
                                        
                            #Contacts for planets transiting in the visit
                            if ordi_name == 'phase':
                                lw_cont = 2.5
                                if plot_options['cmap'] in ['afmhot_r']:
                                    if plot_mod in ['map_Intr_prof_res','map_pca_prof']:
                                        col_loc='lime'
                                        if plot_options['reverse_2D']:Twin = x_range_loc[1]-x_range_loc[0]
                                        else:Twin = y_range_loc[1]-y_range_loc[0]
                                        T14_Twin = system_param[pl_ref]['TLength']/(Twin*system_param[pl_ref]['period'])
                                        if T14_Twin<0.5:lw_cont = 2   
                                        else:lw_cont = 3   
                                    else:col_loc='limegreen'
                                else:col_loc='white'
                                for ipl,pl_loc in enumerate(data_dic[inst][vis]['transit_pl']):
                                    if pl_loc==pl_ref:contact_phases_vis = contact_phases[pl_ref]
                                    else:
                                        contact_times = coord_dic[inst][vis][pl_loc]['Tcenter']+contact_phases[pl_loc]*system_param[pl_loc]["period"]
                                        contact_phases_vis = (contact_times-coord_dic[inst][vis][pl_ref]['Tcenter'])/system_param[pl_ref]["period"]
                                    ls_pl = {0:':',1:'--'}[ipl]
                                    for cont_ph in contact_phases_vis:
                                        if plot_options['reverse_2D']:plt.plot([cont_ph,cont_ph],y_range_loc,color=col_loc,linestyle='--',lw=lw_cont,zorder=10)
                                        else:plt.plot(x_range_loc,[cont_ph,cont_ph],color=col_loc,linestyle=ls_pl,lw=lw_cont,zorder=10)
    
                            #Plot global and in-transit indexes
                            if plot_options['plot_idx']:
                                for isub,iexp in enumerate(iexp_orig):
                                    plt.text(x_range_loc[0]+0.05*(x_range_loc[1]-x_range_loc[0]),0.5*(low_ordi_tab+high_ordi_tab)[isub],str(iexp),verticalalignment='center', horizontalalignment='left',fontsize=10.,zorder=15,color='black') 
                                    plt.text(x_range_loc[0]+0.1*(x_range_loc[1]-x_range_loc[0]),0.5*(low_ordi_tab+high_ordi_tab)[isub],str(gen_dic[inst][vis]['idx_exp2in'][iexp]),verticalalignment='center', horizontalalignment='left',fontsize=10.,zorder=15,color='black') 

                            #------------------------------------  
                      
                            #Planetary and stellar tracks
                            if (ordi_name == 'phase'):
                                
                                #Planet track
                                if (rest_frame in ['star','pl']) and (plot_mod in ['map_DI_prof','map_Res_prof','map_Intr_prof','map_Intr_prof_est','map_Intr_prof_res','map_Atm_prof']) and (plot_options['theoRVpl_HR'] or plot_options['plot_plexc']):
                                    
                                    #Line mask
                                    if ('spec' in data_type):
                                        if (len(plot_options['pl_lines_wav'])>0):line_mask = plot_options['pl_lines_wav']
                                        else:
                                            if ('CCF_mask_wav' not in data_dic['Atm']):stop('Define atmospheric CCF mask or plot mask')
                                            line_mask = data_dic['Atm']['CCF_mask_wav']
                                    else:line_mask = None 
                                    
                                    #Settings
                                    if plot_mod=='map_Atm_prof': col_loc='limegreen'
                                    else:col_loc='magenta'
                                    lw_mod=1.5
                                    ls_loc='-'
                                    
                                    cond_track = plot_options['theoRVpl_HR']
                                    cond_range = (plot_options['plot_plexc']) and (data_dic['Atm']['exc_plrange'])
                                    if cond_range:
                                        line_range = data_dic['Atm']['plrange']
                                        iexp_range = data_dic['Atm'][inst][vis]['iexp_no_plrange']
                                    else:
                                        line_range=None
                                        iexp_range=None
                                    doppler_track_plots('pl',line_mask,rest_frame,col_loc,cond_track,cond_range,lw_mod,ls_loc,data_dic[inst][vis]['transit_pl'],pl_ref,theo_HR_pl,line_range,iexp_plot,iexp_range,plot_options['reverse_2D'],data_type,x_range_loc,y_range_loc)

                                #Stellar track
                                if (rest_frame in ['star','surf']) and ((plot_mod in ['map_Res_prof','map_Intr_prof_est','map_Intr_prof_res','map_Intrbin']) or ((plot_mod in ['map_Intr_prof']) and (not plot_options['aligned']))) and (plot_options['theoRV_HR'] or plot_options['theoRV_HR_align']): 
                                    
                                    #Line mask
                                    if ('spec' in data_type):
                                        if (len(plot_options['st_lines_wav'])>0):line_mask = plot_options['st_lines_wav']
                                        else:
                                            if ('CCF_mask_wav' not in gen_dic) and (inst not in gen_dic['CCF_mask_wav']):stop('Define stellar CCF mask or plot mask')
                                            line_mask = gen_dic['CCF_mask_wav'][inst]
                                    else:line_mask = None 

                                    #Settings
                                    if plot_options['cmap'] in ['afmhot_r']:
                                        col_loc='limegreen'
                                        ls_loc='-'
                                        lw_mod = 2 
                                    else:
                                        col_loc='black'  #'white'
                                        ls_loc='--'   
                                        lw_mod = 2  
                                        
                                    cond_range = (plot_options['plot_stexc']) and (inst in data_dic['DI']['occ_range'])
                                    if cond_range:line_range = data_dic['DI']['occ_range'][inst]
                                    else:line_range=None
                                    iexp_range=None

                                    #Nominal orbit
                                    if plot_options['theoRV_HR']:
                                        cond_track = plot_options['theoRVpl_HR']
                                        doppler_track_plots('surf',line_mask,rest_frame,col_loc,cond_track,cond_range,lw_mod,ls_loc,data_dic[inst][vis]['transit_pl'],pl_ref,theo_HR_prop_plocc,line_range,iexp_plot,iexp_range,plot_options['reverse_2D'],data_type,x_range_loc,y_range_loc)

                                    #Aligned orbit
                                    if plot_options['theoRV_HR_align']:
                                        cond_track = plot_options['theoRVpl_HR_align']
                                        doppler_track_plots('surf',line_mask,rest_frame,col_loc,cond_track,cond_range,lw_mod,ls_loc,data_dic[inst][vis]['transit_pl'],pl_ref,theo_HR_prop_plocc_align,line_range,iexp_plot,iexp_range,plot_options['reverse_2D'],data_type,x_range_loc,y_range_loc)

                            #------------------------------------  
                                      
                            #Modelled and measured local stellar surface RVs
                            if (data_type == 'CCF'):
    
                                #Local differential and intrinsic profiles
                                if (plot_mod in ['map_Res_prof','map_Intr_prof_est','map_Intr_prof_res','map_Intrbin']) or ((plot_mod in ['map_Intr_prof']) and (not plot_options['aligned'])):
                                
                                    #Model at each exposure                        
                                    if plot_options['plot_theoRV']:
                                        RV_plocc = dataload_npz(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis)['achrom']['rv'][0]
                                        ph_theoRV=np.vstack((coord_dic[inst][vis][pl_ref]['cen_ph'][gen_dic[inst][vis]['idx_in']],RV_plocc))
                                        if plot_options['reverse_2D']:
                                            w_sorted=ph_theoRV[0].argsort() 
                                            xtheoRV=ph_theoRV[0,w_sorted]
                                            ytheoRV=ph_theoRV[1,w_sorted]
                                        else:  
                                            w_sorted=ph_theoRV[1].argsort() 
                                            xtheoRV=ph_theoRV[1,w_sorted]
                                            ytheoRV=ph_theoRV[0,w_sorted]
                                        plt.plot(xtheoRV,ytheoRV,color='black',linestyle='-',lw=1.,zorder=10,marker='o',markersize=1.5)
                       
                                    #High-resolution model 
                                    if plot_options['theoRV_HR'] or plot_options['theoRV_HR_align']:
                               
                                        for pl_loc in data_dic[inst][vis]['transit_pl']:
                                            if plot_options['theoRV_HR']:
                                                ph_HR_loc = theo_HR_prop_plocc[pl_ref]['phase']
                                                rv_HR_loc = theo_HR_prop_plocc[pl_loc]['rv']
                                            elif plot_options['theoRV_HR_align']:
                                                ph_HR_loc = theo_HR_prop_plocc_align[pl_ref]['phase']
                                                rv_HR_loc = theo_HR_prop_plocc_align[pl_loc]['rv']
                                            if plot_options['reverse_2D']:
                                                w_sorted=ph_HR_loc.argsort() 
                                                xtheoRV_HR=ph_HR_loc[w_sorted]
                                                if plot_options['theoRV_HR']:
                                                    ytheoRV_HR=rv_HR_loc[w_sorted]
                                                    plt.plot(xtheoRV_HR,ytheoRV_HR,color=col_loc,linestyle=ls_loc,lw=lw_mod,zorder=10) 
                                                if plot_options['theoRV_HR_align']:
                                                    plt.plot(xtheoRV_HR,theo_HR_prop_plocc_align[pl_ref]['rv'][w_sorted],color=col_loc,linestyle='--',lw=lw_mod,zorder=10) 
                                            else:  
                                                w_sorted=rv_HR_loc.argsort() 
                                                ytheoRV_HR=ph_HR_loc[w_sorted] 
                                                if plot_options['theoRV_HR']:
                                                    xtheoRV_HR=rv_HR_loc[w_sorted]
                                                    plt.plot(xtheoRV_HR,ytheoRV_HR,color=col_loc,linestyle=ls_loc,lw=lw_mod,zorder=10) 
                                                if plot_options['theoRV_HR_align']:
                                                    plt.plot(theo_HR_prop_plocc_align[pl_ref]['rv'][w_sorted] ,ytheoRV_HR,color=col_loc,linestyle='--',lw=lw_mod,zorder=10) 
    
                                        
    
                                        # #Save/replot manually  
                                        # save_path = '/Users/bourrier/Travaux/ANTARESS/Ongoing/HAT_P33b_Plots/Intr_data/Maps/RMR_fits/HAT_P33_RVsurf_CB1.dat'
                                        # # np.savetxt(save_path, np.column_stack((xtheoRV_HR,ytheoRV_HR)),fmt=('%15.10f','%15.10f') )
                                        # xtheoRV_HR_sav,ytheoRV_HR_sav=np.loadtxt(save_path).T
                                        # plt.plot(xtheoRV_HR_sav,ytheoRV_HR_sav,color=col_loc,linestyle=':',lw=lw_mod,zorder=10)                                      
                                        
                                        
                                #Measured RV 
                                #    - plot_measRV = 1 : only detected CCFs are plotted
                                #    - plot_measRV = 2 : all CCFs are plotted, but non-detected CCFs are plotted with empty symbols
                                if (plot_options['plot_measRV'] in ['det','all']):
                                    msize_RV=4
                                    mew_RV=0.8
                                    i_in=0
                                    for iexp in range(data_dic[inst][vis]['n_in_visit']):
                                        col_face_symb='magenta'  #'black'
                                        col_symb='magenta'
                                        RV_rest = 0.
                                        plot_rv = False
                                        
                                        #Local differential profiles
                                        if (plot_mod=='map_Res_prof') and ('prof_fit_dic' in data_dic['Res'][inst][vis]) and (iexp in gen_dic[inst][vis]['idx_in']): 
                                            CCF_fit_loc = data_dic['Res'][inst][vis]['prof_fit_dic'][i_in] 
                                            plot_rv = True
                                        
                                        #Local atmospheric profiles
                                        if plot_mod=='map_Atm_prof':
                                            if (data_dic['Atm']['pl_atm_sign']=='Absorption') and (iexp in gen_dic[inst][vis]['idx_in']):CCF_fit_loc = data_dic['Atm'][inst][vis]['prof_fit_dic'][i_in]
                                            if data_dic['Atm']['pl_atm_sign']=='Emission':CCF_fit_loc = data_dic['Atm'][inst][vis]['prof_fit_dic'][iexp]
                                            if (rest_frame=='star'):RV_rest = coord_dic[inst][vis]['rv_pl'][iexp] 
                                            plot_rv = True
                                        if (iexp in gen_dic[inst][vis]['idx_in']):i_in+=1 
                                                                        
                                        #Plot
                                        if (plot_rv == True) and (((plot_options['plot_measRV']=='det') and (CCF_fit_loc['detected'])) or (plot_options['plot_measRV']=='all')):
                                            ph_loc=coord_dic[inst][vis][pl_ref]['cen_ph'][iexp]
                                            RV_loc=CCF_fit_loc['rv']+RV_rest
                                            eRV_loc=[CCF_fit_loc['err_rv'][0],CCF_fit_loc['err_rv'][1]]
                                            if (plot_options['plot_measRV']=='all') and (not CCF_fit_loc['detected']):col_face_symb='none' 
                                            if plot_options['reverse_2D']:plt.errorbar(ph_loc,RV_loc,yerr=eRV_loc,color=col_symb,linestyle='',markeredgewidth=mew_RV,zorder=15,marker='|',markersize=msize_RV,markerfacecolor=col_face_symb,markeredgecolor=col_symb,elinewidth=1)
                                            else: plt.errorbar(RV_loc,ph_loc,xerr=eRV_loc,color=col_symb,linestyle='',markeredgewidth=mew_RV,zorder=15,marker='|',markersize=msize_RV,markerfacecolor=col_face_symb,markeredgecolor=col_symb,elinewidth=1)
                            
                            #----------------------------------------------
                                        
                            #Plot frame 
                            xmajor_int,xminor_int,xmajor_form=autom_tick_prop(x_range_loc[1]-x_range_loc[0])
                            ymajor_int,yminor_int,ymajor_form=autom_tick_prop(y_range_loc[1]-y_range_loc[0])                        
    
                            #Suppress ticks if exposures were plotted sequentially
                            no_xticks,no_yticks=False,False
                            if (plot_mod in ['map_Intr_prof','map_Intr_1D','map_Intr_prof_est','map_Intr_prof_res','map_pca_prof','map_Atm_prof','map_Atm_1D']) and (ordi_name!='phase'):
                                if plot_options['reverse_2D']:no_xticks=True
                                else:
                                    no_yticks=True
                                    for isub,iexp in enumerate(iexp_orig):
                                        plt.text(x_range_loc[0]-0.2*(x_range_loc[1]-x_range_loc[0]),0.5*(low_ordi_tab+high_ordi_tab)[isub],'['+"{0:.2f}".format(low_ordi_tab_ref[isub])+' ; '+"{0:.2f}".format(high_ordi_tab_ref[isub])+']',verticalalignment='center', horizontalalignment='left',fontsize=10.,zorder=15) 
                                  
                            #Frame    
                            custom_axis(plt,ax=ax,position=plot_options['margins'],x_range=x_range_loc,y_range=y_range_loc,
                                        dir_x='out',dir_y='out',xmajor_length=6,ymajor_length=6,
                                        xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,
                                        xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                                        x_title=x_title,y_title=y_title,no_xticks=no_xticks,no_yticks=no_yticks,
                                        colback='black',
                                        font_size=plot_options['font_size'],xfont_size=plot_options['font_size'],yfont_size=plot_options['font_size'])
    
                            #----------------------------------------------	
                            #Colorbar
                            #    - position in add_axes are left, bottom (between 0 and 1), width, height in fraction of the plot size
                            if plot_options['reverse_2D']:cbar_pos=fig.add_axes([plot_options['margins'][0],plot_options['margins'][3]+0.01,(plot_options['margins'][2]-plot_options['margins'][0]),0.03])   
                            else:cbar_pos=fig.add_axes([plot_options['margins'][2]+0.01,plot_options['margins'][1],0.015,(plot_options['margins'][3]-plot_options['margins'][1])])   
                        						
                            #Values	
                            cb = mpl.cm.ScalarMappable(cmap=cmap_2D,norm=plt.Normalize(vmin=v_range[0], vmax=v_range[1]))										
                            cb.set_array(v_range) 	
                        
                            if plot_mod in ['map_DIbin','map_DI_prof','map_Res_prof','map_Intr_prof','map_BF_Res_prof','map_BF_Res_prof_re','map_Intr_prof_est','map_Intr_prof_res','map_pca_prof','map_Intrbin',
                                            'map_Intr_1D','map_Res_prof_clean_pl_est','map_Res_prof_clean_sp_est','map_Res_prof_unclean_sp_est','map_Res_prof_unclean_pl_est','map_Res_prof_clean_sp_res',
                                            'map_Res_prof_clean_pl_res','map_Res_prof_unclean_sp_res','map_Res_prof_unclean_pl_res']:cbar_txt='flux'
                            elif plot_mod in ['map_Atm_prof','map_Atmbin','map_Atm_1D']:cbar_txt=plot_options['pl_atm_sign']
                            cbar_txt = scaled_title(plot_options['sc_fact10'],cbar_txt)  
                            if plot_options['reverse_2D']:
                                cbar=fig.colorbar(cb,cax=cbar_pos,format="%.1f", orientation='horizontal')
                                cbar.set_label(cbar_txt,labelpad=-55,fontsize=plot_options['font_size']-2)	
                                cbar.ax.tick_params(labelsize=plot_options['font_size']-2,labeltop='on')
                                cbar.ax.xaxis.set_ticks_position('top')
                        
            #                        #Fixed labels
            #                        labels = np.array([-0.5e-2,0.,0.5e-2,1e-2,1.5e-2,2e-2])
            #                        cbar.set_ticks(labels)
            #                        cbar.ax.set_xticklabels(labels)
                            else:	
                                cbar=fig.colorbar(cb,cax=cbar_pos) #,format="%i")															
                                cbar.set_label(cbar_txt,rotation=270,labelpad=25,fontsize=plot_options['font_size']-2)	
                                cbar.ax.tick_params(labelsize=plot_options['font_size']-2)
                                			
                            #----------------------------------------------
                            #Saving plot
                            plt.gcf().set_rasterized(plot_options['rasterized'])
                            add_str = 'iord'+str(iord)   
                            if plot_mod in ['map_Intr_prof_est','map_Intr_prof_res']:
                                add_str+='_'+plot_options['mode_loc_data_corr']+'_'+plot_options['line_model'] 
                                if plot_mod=='map_Intr_prof_res':add_str+='_res'
                            elif plot_mod in ['map_BF_Res_prof', 'map_BF_Res_prof_re']:
                                add_str += 'BestFit'
                                if plot_mod=='map_BF_Res_prof_re': add_str += 'Differential'
                            elif plot_mod in ['map_Res_prof_clean_pl_est','map_Res_prof_clean_sp_est','map_Res_prof_unclean_sp_est','map_Res_prof_unclean_pl_est',
                                            'map_Res_prof_clean_sp_res','map_Res_prof_clean_pl_res','map_Res_prof_unclean_sp_res','map_Res_prof_unclean_pl_res']:
                                prof_typ = plot_mod.split('_')[-1]
                                add_str += '_'+corr_plot_mod+'_'+supp_name+'_'+prof_typ                            
                            if ('bin' in plot_mod):add_str+='_'+plot_options['dim_plot'] 
                            plt.savefig(path_loc+'/'+add_str+'.'+save_res_map)                        
                            plt.close() 
               
                            ### End of order                          
           
            return None



    '''
    Generic sub-functions
    '''
    def sub_plot_prof_init(plot_mod,plot_options,inst):

        #Data type
        plot_options['add_txt_path'] = {'DI':'','Intr':'','Res':'','Atm':plot_options['pl_atm_sign']+'/'}
        if 'DI' in plot_mod:data_type_gen = 'DI'
        elif 'Res' in plot_mod:data_type_gen = 'Res'
        elif 'Intr' in plot_mod:data_type_gen = 'Intr'
        elif 'Atm' in plot_mod:data_type_gen = 'Atm'  
        data_type = data_dic[data_type_gen]['type'][inst] 
        if ('bin' in plot_mod):data_mode = 'bin'
        else:data_mode = 'orig'   
        add_txt_path = plot_options['add_txt_path'][data_type_gen]
        
        #Plotting aligned data
        if plot_options['aligned']:
            txt_aligned = 'Aligned_'
            
            #Models are fitted to original intrinsic /atmospheric data before alignment
            if (data_mode == 'orig') and ( ('_res' in plot_mod) or (plot_options['plot_line_model'] and (data_type_gen in ['Intr','Atm'])) ):stop('Remove "aligned" option to overplot model')
        else:txt_aligned=''        
        
        return data_type_gen,data_mode,data_type,add_txt_path,txt_aligned

    def get_data_path(plot_mod,data_type,inst,vis):
        if (plot_mod=='DI_prof') and ('spec' in data_type):  
            data_path_dic = {
                'raw':data_dic[inst][vis]['uncorr_exp_data_paths'],
                'all':data_dic[inst][vis]['corr_exp_data_paths'],
                'tell':gen_dic['save_data_dir']+'Corr_data/Tell/'+inst+'_'+vis+'_', 
                'fbal':gen_dic['save_data_dir']+'Corr_data/Fbal/'+inst+'_'+vis+'_',
                'cosm':gen_dic['save_data_dir']+'Corr_data/Cosm/'+inst+'_'+vis+'_',
                'permpeak':gen_dic['save_data_dir']+'Corr_data/Permpeak/'+inst+'_'+vis+'_',
                'wig':gen_dic['save_data_dir']+'Corr_data/Wiggles/Data/'+inst+'_'+vis+'_',
                }
        else:data_path_dic={}        
        return data_path_dic

    def sub_plot_prof_dir(inst,vis,plot_options,data_mode,series,add_txt_path,plot_mod,txt_aligned,data_type,data_type_gen):
        inout_flag=None

        #Data at chosen steps
        data_path_dic = get_data_path(plot_mod,data_type,inst,vis)

        #Reference planet for the visit
        pl_ref=plot_options['pl_ref'][inst][vis] 
        
        #Upload model
        prof_fit_vis = None
        fit_results = None
        if plot_options['plot_line_model'] or plot_options['plot_line_model_HR'] or (('_res' in plot_mod) & (not plot_options['cont_only'])):
        
            #Planet-occulted models from reconstruction
            if plot_options['line_model']=='rec':
                if 'Intr' in plot_mod:prof_fit_vis = dataload_npz(gen_dic['save_data_dir']+'Loc_estimates/'+plot_options['mode_loc_data_corr']+'/'+inst+'_'+vis+'_add')
                elif 'Res' in plot_mod:prof_fit_vis = dataload_npz(gen_dic['save_data_dir']+'Spot_Loc_estimates/'+plot_options['mode_loc_data_corr']+'/'+inst+'_'+vis+'_add')
        
            #Line profile from best-fit
            else:
                if (plot_options['fit_type']=='indiv'):
                    if not os_system.path.exists(gen_dic['save_data_dir']+data_type_gen+data_mode+'_prop/'+add_txt_path+inst+'_'+vis+'.npz'):stop('No existing fit results')
                    prof_fit_vis=dataload_npz(gen_dic['save_data_dir']+data_type_gen+data_mode+'_prop/'+add_txt_path+inst+'_'+vis)  
                elif (plot_options['fit_type']=='global'):
                    if (data_type_gen in ['Intr','Atm']) and (inst in glob_fit_dic['IntrProf']['idx_in_fit']) and (vis in glob_fit_dic['IntrProf']['idx_in_fit'][inst]):
                        if not os_system.path.exists(gen_dic['save_data_dir']+'Joined_fits/IntrProf/IntrProf_fit_'+add_txt_path+inst+'_'+vis+'.npz'):stop('No existing fit results')
                        prof_fit_vis=dataload_npz(gen_dic['save_data_dir']+'Joined_fits/IntrProf/IntrProf_fit_'+add_txt_path+inst+'_'+vis)['prof_fit_dic']  
                        fit_results =dataload_npz(gen_dic['save_data_dir']+'Joined_fits/IntrProf/Fit_results')
   
        #Directories
        txt_mod=deepcopy(add_txt_path)
        if '_pca' in plot_mod:
            if txt_mod=='':txt_mod+='PCA'
            else:txt_mod+='_PCA'      
        main_path_txt = txt_aligned+data_type_gen+'_data/'+txt_mod+'/'+inst+'_'+vis+'_'+series+'/'
        if '_est' in plot_mod:main_path_txt+='Model'
        elif ('_res' in plot_mod):main_path_txt+= 'Res' 
        else:main_path_txt+= 'Data' 
        main_path_txt+='/'
        if ('spec' in data_type):main_path_txt+='Spec'
        elif ('CCF' in data_type):main_path_txt+='CCF'
        if (gen_dic['type'][inst]=='spec2D') and ('_1D' in plot_mod):main_path_txt+='1Dfrom2D'
        if ('spec' in gen_dic['type'][inst]) and (data_type=='CCF'):
            txt_conv='CCFfromSpec'        
            main_path_txt+='fromSpec'        
        else:txt_conv = ''
        data_bin=None
        if '_pca' in plot_mod: 
            if plot_mod=='map_pca_prof':
                path_loc = gen_dic['save_plot_dir']+'Res_data/PCA/Maps_theo/'  
                data_pca = dataload_npz(gen_dic['save_data_dir']+'PCA_results/'+inst+'_'+vis)
                iexp_plot = data_pca['idx_corr']   
        else:
            if (data_mode == 'bin'):
                path_loc = gen_dic['save_plot_dir']+'Binned_'+main_path_txt+'_'+plot_options['dim_plot'] 
                data_bin = dataload_npz(gen_dic['save_data_dir']+data_type_gen+'bin_data/'+add_txt_path+inst+'_'+vis+'_'+plot_options['dim_plot']+'_add')
                iexp_plot = np.arange(data_bin['n_exp'],dtype=int) 
            elif (data_mode == 'orig'):  
                if plot_mod=='DImast': 
                    path_loc = gen_dic['save_plot_dir']+'Weighing_master/'+inst+'_'+vis
                    data_bin = dataload_npz(gen_dic['save_data_dir']+'DI_data/Master/'+inst+'_'+vis+'_phase_add')
                    iexp_plot = [0]                
                else:
                    path_loc = gen_dic['save_plot_dir']+main_path_txt

                    #Specific options
                    if (plot_mod=='DI_prof') and ('spec' in data_type):   
    
                        #Cosmics
                        if (plot_options['plot_pre']!='cosm') and (plot_options['plot_post']!='cosm'):plot_options['det_cosm']=False
    
                        #Persistent peak correction data
                        if (plot_options['plot_pre']=='permpeak') or (plot_options['plot_post']=='permpeak'):
                            plot_options['data_permpeak'] = dataload_npz(gen_dic['save_data_dir']+'Corr_data/Permpeak/'+inst+'_'+vis+'_add')
                        else:
                            plot_options['data_permpeak'] = None
                            plot_options['det_permpeak']=False                                

                    #Exposures to plot and rest frames              
                    if (gen_dic['type'][inst]=='spec2D') and ('_1D' in plot_mod): 
                        data_add = dataload_npz(gen_dic['save_data_dir']+data_type_gen+'_data/1Dfrom2D/'+add_txt_path+'/'+inst+'_'+vis+'_add')
                        iexp_plot = data_add['iexp_conv']                  
                    else:
                        if data_type_gen=='DI':iexp_plot=range(data_dic[inst][vis]['n_in_visit'])
                        elif data_type_gen=='Res':iexp_plot = data_dic[data_type_gen][inst][vis]['idx_to_extract']
                        elif data_type_gen in ['Intr','Atm']:iexp_plot = data_dic[data_type_gen][inst][vis]['idx_def']
                    if ('iexp_plot' in plot_options) and (inst in plot_options['iexp_plot']) and (vis in plot_options['iexp_plot'][inst]) and (len(plot_options['iexp_plot'][inst][vis])>0):
                        iexp_plot = np.intersect1d(iexp_plot,plot_options['iexp_plot'][inst][vis])   
        if (not os_system.path.exists(path_loc)):os_system.makedirs(path_loc)  


        #Original exposure index
        nexp_plot = len(iexp_plot)
        if (vis=='binned') or (('DI' in plot_mod) or ('Res' in plot_mod) or (plot_mod=='map_pca_prof') or (('Atm' in plot_mod) and (plot_options['pl_atm_sign']=='Emission'))):iexp_orig = iexp_plot
        elif ('Intr' in plot_mod) or (('Atm' in plot_mod) and (plot_options['pl_atm_sign']=='Absorption')):iexp_orig = gen_dic[inst][vis]['idx_in2exp'][iexp_plot]
            
    
        #Indexes of master exposures
        if ('DI' in plot_mod) and (vis!='binned'):
            if (inst in plot_options['iexp_mast_list']) and (vis in plot_options['iexp_mast_list'][inst]):
                if plot_options['iexp_mast_list'][inst][vis]=='all':iexp_mast_list = np.arange(data_dic[inst][vis]['n_in_visit'])
                else:iexp_mast_list = plot_options['iexp_mast_list'][inst][vis]
            else:iexp_mast_list = gen_dic[inst][vis]['idx_out']        
        else:iexp_mast_list=None   
        
        #----------------------------------------
        #Path of retrieved data
        #    - for each type of data we define the rest frame and number of spectral orders (because the generic field is overwritten with the latest value we get if from the data structure itself)
        #----------------------------------------
        
        #PCA results
        if plot_mod=='map_pca_prof':
            data_path_all = [gen_dic['save_data_dir']+'PCA_results/'+inst+'_'+vis+'_model'+str(iexp) for iexp in iexp_plot] 
            rest_frame = 'star'
              
        #Reconstructed line profiles
        elif '_est' in plot_mod:
            if 'Intr' in plot_mod:data_path_all = [gen_dic['save_data_dir']+'Loc_estimates/'+plot_options['mode_loc_data_corr']+'/'+inst+'_'+vis+'_'+str(iexp) for iexp in iexp_plot]
            elif 'Res' in plot_mod:data_path_all = [gen_dic['save_data_dir']+'Spot_Loc_estimates/'+plot_options['mode_loc_data_corr']+'/'+inst+'_'+vis+'_'+str(iexp) for iexp in iexp_plot]
            rest_frame = 'star'
         
        #Residual maps from Intrinsic and out-of-transit Residual profiles
        #    - data_path_all contains the path to Intrinsic profiles, from which will later be subtracted the model profiles
        #      if out-of-transit residuals are requested, we add the path to out-of-transit Residual profiles
        elif (plot_mod=='map_Intr_prof_res'): 
            inout_flag=np.repeat('in',len(iexp_plot))  
            if plot_options['cont_only']:
                data_path_all = [data_dic[inst][vis]['proc_Intr_data_paths']+str(iexp) for iexp in iexp_plot]
                rest_frame = data_dic['Intr'][inst][vis]['rest_frame']                  
            else:
                data_path_all = [prof_fit_vis['loc_data_corr_inpath']+str(iexp) for iexp in iexp_plot]
                rest_frame = prof_fit_vis['rest_frame']                   
 
            #Calculate out-of-transit residuals
            if plot_options['show_outres']:
                iexp_out = np.setdiff1d(data_dic['Res'][inst][vis]['idx_to_extract'],iexp_orig)   #original indexes of local profiles not processed as intrinsic profiles
                inout_flag=np.append(inout_flag,np.repeat('out',len(iexp_out)))
                iexp_plot=np.append(iexp_plot,iexp_out) 
                nexp_plot+=len(iexp_out)
                iexp_orig=np.append(iexp_orig,iexp_out)
                if plot_options['cont_only']:
                    data_path_all+=[data_dic[inst][vis]['proc_Res_data_paths']+str(iexp) for iexp in iexp_out]
                else:
                    data_path_all+=[prof_fit_vis['loc_data_corr_outpath']+str(iexp) for iexp in iexp_out]

        #Residual maps from Residual profiles
        #    - data_path_all contains the path to Residual profiles, from which will later be subtracted the model profiles
        elif plot_mod in ['map_Res_prof_clean_sp_res','map_Res_prof_clean_pl_res','map_Res_prof_unclean_sp_res','map_Res_prof_unclean_pl_res']:
            data_path_all = [prof_fit_vis['loc_data_corr_path']+str(iexp) for iexp in iexp_plot]
            rest_frame = prof_fit_vis['rest_frame']              
        elif (plot_mod in ['map_BF_Res_prof', 'map_BF_Res_prof_re']):
            #Retrieving the bin mode
            if vis+'_bin' in data_dic['Res']['idx_in_bin'][inst]:bin_mode='_bin'
            else:bin_mode = ''
            #Retrieving the best-fit exposures
            data_path_all = [gen_dic['save_data_dir']+'Joined_fits/ResProf/'+glob_fit_dic['ResProf']['fit_mode']+'/'+inst+'/'+vis+'/'+'BestFit'+bin_mode+'_'+str(iexp) for iexp in iexp_plot]
            rest_frame = 'star'
        
        #Measured data
        else:         
            if (data_mode == 'bin'):
                data_path_all = [gen_dic['save_data_dir']+data_type_gen+'bin_data/'+add_txt_path+'/'+inst+'_'+vis+'_'+plot_options['dim_plot']+str(iexp) for iexp in iexp_plot]
                rest_frame = data_bin['rest_frame']
            elif (data_mode == 'orig'): 
                if (gen_dic['type'][inst]=='spec2D') and ('_1D' in plot_mod):         
                    data_path_all = [gen_dic['save_data_dir']+data_type_gen+'_data/1Dfrom2D/'+add_txt_path+'/'+inst+'_'+vis+'_'+str(iexp) for iexp in iexp_plot]
                    rest_frame = data_add['rest_frame'] 
                else: 
                    
                    #Disk-integrated profiles
                    if 'DI' in plot_mod:

                        #Master built from original data 
                        if plot_mod=='DImast':    
                            data_path_all = [gen_dic['save_data_dir']+data_type_gen+'_data/Master/'+inst+'_'+vis+'_phase']
                            rest_frame = data_bin['rest_frame']
                            
                        #Disk-integrated profiles
                        elif ('DI_prof' in plot_mod): 
    
                            #Spectral profiles
                            if ('spec' in data_type):         
                                
                                #Over correction steps
                                if ('plot_pre' in plot_options) or ('plot_post' in plot_options):
                                    if plot_options['plot_post'] is not None:data_path_all = [data_path_dic[plot_options['plot_post']]+str(iexp) for iexp in iexp_plot]
                                    else:data_path_all = [None for iexp in iexp_plot]
                                    rest_frame='input'   
                                
                                #Over processing steps
                                else:
                                    if plot_options['step']=='sp_corr':
                                        data_path = data_dic[inst][vis]['corr_exp_data_paths']
                                        rest_frame='input'   
                                    elif plot_options['step']=='detrend':
                                        data_path = gen_dic['save_data_dir']+'Detrend_prof/'+inst+'_'+vis+'_'  
                                        rest_frame='input'                
                                    elif plot_options['step']=='aligned':
                                        data_path = gen_dic['save_data_dir']+'Aligned_DI_data/'+inst+'_'+vis+'_' 
                                        rest_frame='star'   
                                    elif plot_options['step']=='scaled':
                                        data_path = gen_dic['save_data_dir']+'Scaled_data/'+inst+'_'+vis+'_' 
                                        rest_frame = dataload_npz(data_path+'add')['rest_frame']                                   
                                    data_path_all = [data_path+str(iexp) for iexp in iexp_plot] 
                                
                            #CCF profiles
                            elif (data_type=='CCF'):
                                if plot_options['step']=='raw':
                                    data_path = data_dic[inst][vis]['raw_exp_data_paths']
                                    rest_frame = 'input'
                                elif plot_options['step']=='conv':    
                                    data_path = gen_dic['save_data_dir']+'DI_data/CCFfromSpec/'+inst+'_'+vis+'_'  
                                    rest_frame = 'input'
                                elif plot_options['step']=='scaled':
                                    data_path = gen_dic['save_data_dir']+'Scaled_data/'+inst+'_'+vis+'_' 
                                    rest_frame = dataload_npz(data_path+'add')['rest_frame']
                                elif plot_options['step']=='latest':
                                    data_path = data_dic[inst][vis]['proc_DI_data_paths']
                                    rest_frame = dataload_npz(data_path+'add')['rest_frame']                                    
                                data_path_all = [data_path+str(iexp) for iexp in iexp_plot]                  
                          
                    #Other types of profiles
                    else:  
                        data_path_all = [gen_dic['save_data_dir']+txt_aligned+data_type_gen+'_data/'+add_txt_path+'/'+txt_conv+'/'+inst+'_'+vis+'_'+str(iexp) for iexp in iexp_plot]   
                        if 'Res' in plot_mod:rest_frame='star'   
                        elif 'Intr' in plot_mod:
                            if plot_options['aligned']:rest_frame='surf' 
                            else:rest_frame='star'   
                        if 'Atm' in plot_mod:
                            if plot_options['aligned']:rest_frame='pl' 
                            else:rest_frame='star'   
            
        #Data dimensions
        if data_path_all[0] is not None:
            if ('Res_prof' in plot_mod) and ('_est' in plot_mod):
                flux_supp = plot_mod.split('_')[4]
                corr_plot_mod = plot_mod.split('_')[3]
                flux_name=corr_plot_mod+'_'+flux_supp+'_flux'
            else:flux_name='flux'
            dim_exp_data = list(np.shape(dataload_npz(data_path_all[0])[flux_name]))
            nord_data = dim_exp_data[0] 
        else:
            nord_data = None
                            
        return pl_ref,txt_conv,iexp_plot,iexp_orig,prof_fit_vis,fit_results,data_path_all,rest_frame,data_path_dic,nexp_plot,inout_flag,path_loc,iexp_mast_list,nord_data,data_bin
    
    
    

    
  
    '''
    Sub-function to get telluric transitions in their rest frame
    '''    
    def get_tell_lines(tell_HITRANS,gen_dic):
        wave_tell_lines = {}
        if len(tell_HITRANS)>0:
            for molec in tell_HITRANS:
                
                #Full line list for considered species 
                #    - defined in vacuum and at rest
                static_file_full_range = fits.open('Telluric_processing/Static_model/'+inst+'/Static_hitran_qt_'+molec+'.fits')   
                wave_number_tellL = (static_file_full_range[1].data)['wave_number'] # [cm-1]
                wave_tell_lines[molec] = 1e8/wave_number_tellL #[A]
    
                #Spectral conversion from vacuum to air
                if gen_dic['sp_frame']=='air': 
                    wave_tell_lines[molec]/=air_index(wave_tell_lines[molec], t=15., p=760.)    
    
        return wave_tell_lines

    '''
    Generic sub-function to plot individual profiles
    '''
    def sub_plot_prof(plot_options,plot_mod,plot_ext):
        
        #Options
        sc_fact=10**plot_options['sc_fact10']
        hide_yticks=False  

        #Plot for each instrument        
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_options['visits_to_plot'].keys())): 
            print('   - Instrument :',inst)
            for key in ['color_dic','color_dic_sec','color_dic_bin','color_dic_bin_sec']:
                if inst not in plot_options[key]:plot_options[key][inst]={}

            #Data to plot
            maink_list = []
            data_list = []
            if plot_options['plot_pre'] is not None:
                maink_list+=['pre']
                data_list+=[plot_options['plot_pre']]
            if plot_options['plot_post'] is not None:
                maink_list+=['post']
                data_list+=[plot_options['plot_post']]  

            #Spectral options
            if ('spec' in gen_dic['type'][inst]): 

                #HITRAN telluric lines
                wave_tell_lines = get_tell_lines(plot_options['plot_tell_HITRANS'],gen_dic)

            #Generic init function
            data_type_gen,data_mode,data_type,add_txt_path,txt_aligned=sub_plot_prof_init(plot_mod,plot_options,inst)

            #Frame properties
            if 'DI' in plot_mod:
                y_title='Flux'
                if ('DIbin' in plot_mod) or ('DImast' in plot_mod):title_name='Binned disk-integrated'
                else:title_name='Raw disk-integrated'                                                
            elif ('Res' in plot_mod):
                y_title='Flux'
                title_name='Local'                          
            elif ('Intr' in plot_mod):   
                y_title='Flux'
                if 'bin' in plot_mod:title_name='Binned intrinsic'
                else:title_name='Raw Intrinsic'                                                        
            elif ('Atm' in plot_mod):                      
                if plot_options['pl_atm_sign']=='Absorption':y_title='Absorption'
                elif plot_options['pl_atm_sign']=='Emission':y_title='Flux'
                if 'bin' in plot_mod:title_name='Binned atmospheric '+plot_options['pl_atm_sign'] 
                else:title_name='Raw atmospheric '+plot_options['pl_atm_sign']                                                
            if ('1D' in plot_mod):title_name='1D '+title_name                                                                   
            if ('_res' in plot_mod):
                title_name+=' (residuals)'   
                y_title='Residuals'
            y_title = scaled_title(plot_options['sc_fact10'],y_title)
    
            if (plot_mod in ['DIbin','DIbin_res','DImast']):                                        
                ref_name='phase'
            elif (plot_mod in ['Intrbin','Intrbin_res','Atmbin','Atmbin_res','Intrbin','Atmbin']):
                ref_name = plot_options['dim_plot']
            # elif (plot_mod in ['sp_Intr_1D','sp_Atm_1D']) and (plot_options['dim_plot']=='bin'):
            #     ref_name = dim_bin_1D
            else:
                ref_name = 'phase'  

            #Plot for each visit
            for vis in np.intersect1d(list(data_dic[inst].keys())+['binned'],plot_options['visits_to_plot'][inst]): 
                print('     - Visit :',vis)
                data_inst=data_dic[inst]

                #Data
                if vis=='binned':data_com = dataload_npz(data_inst['proc_com_data_path'])  
                else:data_com = dataload_npz(data_inst[vis]['proc_com_data_paths'])  
                fixed_args_loc = {}
                pl_ref,txt_conv,iexp_plot,iexp_orig,prof_fit_vis,fit_results,data_path_all,rest_frame,data_path_dic,nexp_plot,inout_flag,path_loc,iexp_mast_list,nord_data,data_bin = sub_plot_prof_dir(inst,vis,plot_options,data_mode,'Indiv',add_txt_path,plot_mod,txt_aligned,data_type,data_type_gen)
                
                #Order list
                if data_type=='CCF':order_list=[0]
                else:
                    order_list = plot_options['orders_to_plot'] if len(plot_options['orders_to_plot'])>0 else range(nord_data)  
                idx_sel_ord = order_list
                if len(order_list)==1:plot_options['multi_ord']=False

                #Spectral variable
                if plot_options['aligned']:title_name='Aligned '+title_name
                xt_str={'input':'heliocentric','star':'star','surf':'surface','pl':'planet'}[rest_frame]
                if plot_options['x_range'] is not None:x_range_loc = plot_options['x_range']
                if 'spec' in data_type:
                    if plot_options['sp_var'] == 'nu' :
                        if plot_options['x_range'] is None:x_range_loc = [c_light/9000.,c_light/3000.]
                        x_title = r'$\nu$ in '+xt_str+' rest frame (10$^{-10}$s$^{-1}$)'
                        
                    elif plot_options['sp_var'] == 'wav' :
                        if plot_options['x_range'] is None:x_range_loc = [3000.,9000.] 
                        x_title = r'Wavelength in '+xt_str+' rest frame (A)'                        
                else:x_title='Velocity in '+xt_str+' rest frame (km s$^{-1}$)'                    

                #Colors
                if vis not in plot_options['color_dic'][inst]:plot_options['color_dic'][inst][vis] = np.repeat('dodgerblue',nexp_plot)
                else:
                    if plot_options['color_dic'][inst][vis]=='jet':
                        cmap = plt.get_cmap('jet') 
                        plot_options['color_dic'][inst][vis]=np.array([cmap(0)]) if nexp_plot==1 else cmap( np.arange(nexp_plot)/(nexp_plot-1.))         
                    else:plot_options['color_dic'][inst][vis] = np.repeat(plot_options['color_dic'][inst][vis],nexp_plot)
                if vis not in plot_options['color_dic_sec'][inst]:plot_options['color_dic_sec'][inst][vis] =np.repeat('red',nexp_plot)   
                else:
                    if plot_options['color_dic_sec'][inst][vis]=='jet':
                        cmap = plt.get_cmap('jet') 
                        plot_options['color_dic_sec'][inst][vis]=np.array([cmap(0)]) if nexp_plot==1 else cmap( np.arange(nexp_plot)/(nexp_plot-1.))         
                    else:plot_options['color_dic_sec'][inst][vis] = np.repeat(plot_options['color_dic_sec'][inst][vis],nexp_plot)

                #Process selected ranges and orders
                nord_proc = len(idx_sel_ord)
                if nord_proc==0:stop('No orders left')
                if vis=='binned':
                    nspec_eff = data_inst['nspec']
                    dim_exp_proc = [nord_proc,nspec_eff]
                else:
                    nspec_eff = data_inst[vis]['nspec']
                    dim_exp_proc = [nord_proc,nspec_eff]
                cen_bins_com = data_com['cen_bins'][idx_sel_ord]
                edge_bins_com = data_com['edge_bins'][idx_sel_ord]                

                #Pre-processing exposures  
                if (plot_mod=='DI_prof') and ('spec' in data_type) and (not gen_dic['mock_data']):  
                    data_proc,data_mod,data4mast = pre_proc_exp(plot_options,inst,vis,maink_list,iexp_plot,iexp_mast_list,data_inst,data_inst[vis],data_path_dic,idx_sel_ord,cen_bins_com,edge_bins_com,nord_proc,dim_exp_proc,data_list,fixed_args_loc)
                else:
                    data_proc={}
                    data_mod={}
                    data4mast={}
                
                #Stellar continuum
                if ('spec' in data_type) and (plot_options['st_cont'] is not None):
                    cont_func_dic = dataload_npz(gen_dic['save_data_dir']+'Stellar_cont_'+plot_options['st_cont']+'/'+inst+'_'+vis+'/St_cont')['cont_func_dic']

                #Multiple exposures per plot
                images_to_make_GIF = None
                if plot_options['multi_exp']:
                    
                    #Single order per plot
                    if (not plot_options['multi_ord']): 
                        all_figs = {}
                        all_ax= {}
                        x_range_loc ={}
                        y_range_loc = {}
                        
                        #GIF is generated over series of orders with multiple exposures in each plot
                        if plot_options['GIF_generation']:images_to_make_GIF = []   
    
                    #Multiple orders per plot
                    elif plot_options['multi_exp']: 
                        plt.ioff() 
                        key_frame = ('all','all')
                        all_figs = {key_frame : plt.figure(figsize=plot_options['fig_size'])}
                        all_ax= {key_frame : plt.gca()}
                        if plot_options['x_range'] is None:x_range_loc= {key_frame :[1e100,-1e100] }
                        else:x_range_loc={}
                        if plot_options['y_range'] is not None:y_range_loc= {key_frame :sc_fact*np.array(plot_options['y_range'])}
                        else:y_range_loc= {key_frame :[1e100,-1e100] }
         
                #Plot profile for each exposure
                for isub,(iexp,data_path_exp) in enumerate(zip(iexp_plot,data_path_all)): 
                    iexp_or=iexp_orig[isub]     
                 
                    #Colors
                    col_exp = plot_options['color_dic'][inst][vis][isub]
                    col_exp_sec = plot_options['color_dic_sec'][inst][vis][isub]        
              
                    #Upload data
                    if data_path_exp is not None:data_exp = dataload_npz(data_path_exp)
                    
                    if (data_mode == 'orig'):       

                        #Specific options
                        if (plot_mod=='DI_prof') and ('spec' in data_type):  
                                  
                            #Data at chosen steps
                            if plot_options['plot_pre'] is not None:
                                data_precorr = dataload_npz(data_path_dic[plot_options['plot_pre']]+str(iexp))

                            #Flux balance correction data
                            if (plot_options['plot_mast']):
                                if (gen_dic['corr_Fbal']):data_Fbal_exp = dataload_npz(gen_dic['save_data_dir']+'Corr_data/Fbal/'+inst+'_'+vis+'_'+str(iexp)+'_add')
                                
                            #Cosmics correction data
                            if plot_options['det_cosm']:
                                data_cosm = dataload_npz(gen_dic['save_data_dir']+'Corr_data/Cosm/'+inst+'_'+vis+'_'+str(iexp)+'_add')
    
                            #Plot HITRAN telluric lines
                            if len(plot_options['plot_tell_HITRANS'])>0:
                                wave_tellL_exp = {}
        
                                #Shift from Earth (source) to solar barycentric (receiver) rest frame
                                #    - see gen_specdopshift():
                                # w_receiver = w_source * (1+ (rv[s/r]/c))
                                # w_solbar = w_Earth * (1+ (rv[Earth/solbar]/c))
                                # w_solbar = w_Earth * (1+ (BERV/c))
                                for molec in wave_tell_lines:
                                    wave_tellL_exp[molec] = wave_tell_lines[molec]*gen_specdopshift(data_prop[inst][vis]['BERV'][iexp])*(1.+1.55e-8)  
                                    
                            #Telluric spectrum
                            if plot_options['plot_tell']:
                                tell_exp = dataload_npz(gen_dic['save_data_dir']+'Corr_data/Tell/'+inst+'_'+vis+'_'+'tell_'+str(iexp))['tell']                                    


                    #Models
                    do_plot=True
                    cond_mod = False
                    if vis=='binned':idx_exp2in = data_bin['idx_exp2in']
                    else:idx_exp2in = gen_dic[inst][vis]['idx_exp2in']
                    if ('Res' in plot_mod):iexp_mod = idx_exp2in[iexp]
                    else:iexp_mod = iexp
                    i_in = idx_exp2in[iexp_or]
                    if (prof_fit_vis is not None) and \
                        ((plot_mod in ['DI_prof','DI_prof_res']) or \
                         ('Intr' in plot_mod) and ((i_in==-1.) or \
                                                   ((i_in>-1) and (((plot_options['line_model']=='fit') and (iexp_mod in prof_fit_vis)) or \
                                                                                                        ((plot_options['line_model']=='rec') and (iexp_mod in prof_fit_vis['idx_est_loc'])))))): 
                        cond_mod = True
                        if plot_options['line_model']=='fit':
                            
                            #Best-fit model stored as array
                            if  (plot_options['fit_type'] in ['indiv','global']): 
                                mod_prop_exp=prof_fit_vis[iexp_mod]                    
                                if (plot_options['fit_type']=='indiv'):  
                                    if (plot_mod in ['DI_prof','DI_prof_res']):idx_excl_bd_ranges=prof_fit_vis['idx_excl_bd_ranges'][iexp_mod]
                                    idx_trim_kept = mod_prop_exp['idx_mod']
                                    cond_fit_exp_raw = np.zeros(nspec_eff,dtype=bool)
                                    cond_fit_exp_raw[idx_trim_kept] =  prof_fit_vis['cond_def_fit_all'][iexp_mod][idx_trim_kept]                 #trimmed and fitted condition in individual table, defined only over model pixels
                                    cond_cont_exp_raw = np.zeros(nspec_eff,dtype=bool)
                                    cond_cont_exp_raw[idx_trim_kept] = prof_fit_vis['cond_def_cont_all'][iexp_mod][idx_trim_kept]                #trimmed and fitted condition in individual table, defined only over model pixels
                                    flux_mod_exp_fit = mod_prop_exp['flux']   
                                elif (plot_options['fit_type']=='global'): 
                                    idx_trim_kept = fit_results['idx_trim_kept'][inst][vis]     #trimmed indexes of profiles before fit
                                    cond_fit_exp_trim = mod_prop_exp['cond_def_fit']            #fitted indexes in trimmed tables                      
                                    cond_fit_exp_raw = np.zeros(nspec_eff,dtype=bool)
                                    cond_fit_exp_raw[idx_trim_kept] = cond_fit_exp_trim         #trimmed and fitted condition in global tables
                                    cond_cont_exp_raw = np.zeros(nspec_eff,dtype=bool)
                                    cond_cont_exp_raw[idx_trim_kept] = mod_prop_exp['cond_def_cont']     #trimmed and continuum condition in global tables
                                    flux_mod_exp_fit = mod_prop_exp['flux'][cond_fit_exp_trim]
                                cond_cont_exp_fit  = cond_cont_exp_raw[idx_trim_kept]  #trimmed and continuum condition in individual table, reduced to model pixels 
                                    
                                cen_bins_mod_exp =  mod_prop_exp['cen_bins']
                                flux_mod_exp = mod_prop_exp['flux']
                                
                        #Reconstruction stored with data structure
                        elif plot_options['line_model']=='rec':
                            data_exp_est = dataload_npz(gen_dic['save_data_dir']+'Loc_estimates/'+plot_options['mode_loc_data_corr']+'/'+inst+'_'+vis+'_'+str(i_in))
                            cond_def_mod = data_exp_est['cond_def'][0]     
                            cond_fit_exp_raw = cond_def_mod
                            cen_bins_mod_exp =  data_exp_est['cen_bins'][0,cond_def_mod]
                            flux_mod_exp = data_exp_est['flux'][0,cond_def_mod]
                            flux_mod_exp_fit = flux_mod_exp
                            loc_flux_scaling_plot = dataload_npz(data_vis['scaled_Intr_data_paths']+str(iexp_or))['loc_flux_scaling']       
                            loc_flux_scaling_exp_ord = loc_flux_scaling_plot(cen_bins_mod_exp)
                            intr2res_sc = loc_flux_scaling_exp_ord/(1. - loc_flux_scaling_exp_ord)
                            if ('Res' in plot_mod) and (data_dic['Intr']['plocc_prof_type']=='Intr'):flux_mod_exp*=intr2res_sc
                            if ('Int' in plot_mod) and (data_dic['Intr']['plocc_prof_type']=='Res'):flux_mod_exp/=intr2res_sc 
                        idx_def_fit_raw = np_where1D(cond_fit_exp_raw)                            
                            
                    elif ('_res' in plot_mod):do_plot=False    
                    
                    #Plot exposure only if fitted
                    if plot_options['fitted_exp']:do_plot&=cond_mod
                   
                    #Plot current exposure  
                    if do_plot:

                        #Overplot multiple orders for each plotted exposure
                        if (plot_options['multi_ord']) and (not plot_options['multi_exp']): 
                            plt.ioff()
                            key_frame = (iexp,'all')
                            all_figs[key_frame] = plt.figure(figsize=plot_options['fig_size'])
                            all_ax[key_frame] = plt.gca()
                            if plot_options['x_range'] is None:x_range_loc= {key_frame :[1e100,-1e100] }   
                            else:x_range_loc={}
                            if plot_options['y_range'] is not None:y_range_loc[key_frame]=sc_fact*np.array(plot_options['y_range'])
                            else:y_range_loc[key_frame]=[1e100,-1e100]                    
     
                        #Plot each order 
                        for isub_ord,iord in enumerate(order_list):                    
                           
                            #Only plot order if it overlaps with the requested window
                            plot_ord=True
                            cen_bins = data_exp['cen_bins'][iord]
                            cond_def = data_exp['cond_def'][iord]
                            if np.sum(cond_def)>0:
                                if plot_options['x_range'] is not None:
                                    if (cen_bins[cond_def][-1]<plot_options['x_range'][0]) or (cen_bins[cond_def][0]>plot_options['x_range'][1]):plot_ord&=False
                                    else:cond_def[ (cen_bins<plot_options['x_range'][0]) | (cen_bins>plot_options['x_range'][1]) ]=False

                                #Conditions on disk-integrated spectra
                                if (plot_mod=='DI_prof') and ('spec' in data_type):
                                    
                                    #Only plot order if cosmics were detected
                                    if plot_options['det_cosm'] and (len(data_cosm['idx_cosm_exp'][iord])==0):plot_ord &= False     
            
                                    #Only plot order if cosmics were detected
                                    if plot_options['det_permpeak'] and (plot_options['data_permpeak']['count_bad_all'][iexp,iord]==0):plot_ord &= False  
                                
                                #Plot order
                                if plot_ord:  

                                    #One order per plot
                                    if not plot_options['multi_ord']: 

                                        #One exposure per plot
                                        if not plot_options['multi_exp']:                                          
                                            plt.ioff()   
                                            key_frame = (iexp,iord)
                                            all_figs = {key_frame:plt.figure(figsize=plot_options['fig_size'])}
                                            all_ax = {key_frame:plt.gca()}
                                            if plot_options['x_range'] is not None:x_range_loc= {key_frame:np.array(plot_options['x_range'])}
                                            else:x_range_loc = {key_frame:[1e100,-1e100] }                                            
                                            if plot_options['y_range'] is not None:y_range_loc={key_frame:sc_fact*np.array(plot_options['y_range'])}
                                            else:y_range_loc={key_frame:[1e100,-1e100]} 
                                            
                                            #GIF is generated over series of exposures for current order
                                            if plot_options['GIF_generation'] and (len(iexp_plot)>1) and (images_to_make_GIF is None):images_to_make_GIF = {iord : []}
         
                                        #Multiple exposures per plot
                                        else:
                                            key_frame = ('all',iord)
                                            if key_frame not in all_figs:
                                                all_figs[key_frame] = plt.figure(figsize=plot_options['fig_size'])
                                                all_ax[key_frame] = plt.gca()
                                                if plot_options['x_range'] is not None:x_range_loc[key_frame]=np.array(plot_options['x_range'])
                                                else:x_range_loc[key_frame]=[1e100,-1e100] 
                                                if plot_options['y_range'] is not None:y_range_loc[key_frame]=sc_fact*np.array(plot_options['y_range'])
                                                else:y_range_loc[key_frame]=[1e100,-1e100] 

                                    #Frame        
                                    if plot_options['x_range'] is None:
                                        gap = 0.  #0.05
                                        xmin = cen_bins[cond_def][0]
                                        xmax = cen_bins[cond_def][-1]
                                        dx_range = xmax-xmin
                                        x_range_ord = [xmin-gap*dx_range,xmax+gap*dx_range]
                                        if plot_options['multi_ord']:
                                            xmin = min(xmin,x_range_loc[key_frame][0])
                                            xmax = max(xmax,x_range_loc[key_frame][1])
                                        dx_range = xmax-xmin
                                        x_range_loc[key_frame] = [xmin-gap*dx_range,xmax+gap*dx_range]  
                                    else:x_range_ord = plot_options['x_range']
                                    dx_range=x_range_loc[key_frame][1]-x_range_loc[key_frame][0]                                    
    

                                    #----------------------------------------
                                    #Data
                                    #----------------------------------------
           
                                    #Resampling table
                                    if plot_options['resample'] is not None:
                                        n_reg = int(np.ceil((x_range_ord[1]-x_range_ord[0])/plot_options['resample']))
                                        edge_bins_reg = np.linspace(x_range_ord[0],x_range_ord[1],n_reg)
                                        cen_bins_reg = 0.5*(edge_bins_reg[0:-1]+edge_bins_reg[1::])   
    
                                    #Flux and error
                                    edge_bins = data_exp['edge_bins'][iord]
                                    flux_exp = data_exp['flux'][iord]
                                    cond_def_exp = data_exp['cond_def'][iord]
                                    err_exp = np.sqrt(data_exp['cov'][iord][0,:])                      
        
                                    #Normalisation
                                    #    - to set profiles to comparable levels for the plot and to a mean unity  
                                    if plot_options['norm_prof']:
                                        if ('Intr' in plot_mod):
                                            if cond_mod:
                                                if (plot_options['line_model']=='fit'):
                                                    if (plot_options['fit_type']=='indiv'):mean_flux = prof_fit_vis[iexp]['cont'] 
                                                    elif (plot_options['fit_type']=='global'):mean_flux = fit_results['p_final']['cont'] 
                                                elif (plot_options['line_model']=='rec'):mean_flux = prof_fit_vis['cont']  
                                            else:mean_flux = data_dic['Intr'][inst][vis]['mean_cont'][iord] 
                                        else:
                                            cond_def_scal = True
                                            if ('DI' in plot_mod) and (len(data_dic['DI']['scaling_range'])>0):
                                                cond_def_scal=False 
                                                for bd_int in data_dic['DI']['scaling_range']:cond_def_scal |= (edge_bins[0:-1]>=bd_int[0]) & (edge_bins[1:]<=bd_int[1])                                     
                                            cond_def_scal&=cond_def_exp
                                            dcen_bins = edge_bins[1:] - edge_bins[0:-1]
                                            mean_flux=np.nansum(flux_exp[cond_def_scal]*dcen_bins[cond_def_scal])/np.sum(dcen_bins[cond_def_scal])
                                    else:mean_flux=1.
                                    
                                    #Plot DI spectra at various steps of the spectral corrections
                                    if (plot_mod=='DI_prof') and ('spec' in data_type): 
                                        
                                        #Plot spectrum before correction   
                                        if plot_options['plot_pre'] is not None:
                                            if plot_options['norm_prof']:
                                                dcen_bins_raw = data_precorr['edge_bins'][iord][1:] - data_precorr['edge_bins'][iord][0:-1]
                                                mean_flux_raw=np.sum(data_precorr['flux'][iord,data_precorr['cond_def'][iord]]*dcen_bins_raw[data_precorr['cond_def'][iord]])/np.sum(dcen_bins_raw[data_precorr['cond_def'][iord]])
                                            else:mean_flux_raw=1.     
                                            var_loc = sc_fact*data_precorr['flux'][iord]/mean_flux_raw   
                                            if plot_options['plot_err']:
                                                col_exp_err =  'lightblue'   
                                                col_exp_err = col_exp
                                                all_ax[key_frame].errorbar(data_precorr['cen_bins'][iord],var_loc,yerr = sc_fact*np.sqrt(data_precorr['cov'][iord][0,:])/mean_flux_raw,color=col_exp_err,linestyle='',lw=plot_options['lw_plot'],rasterized=plot_options['rasterized'],zorder=0,alpha=plot_options['alpha_err'],figure = all_figs[key_frame]) 
                                            all_ax[key_frame].plot(data_precorr['cen_bins'][iord],var_loc,color=col_exp,linestyle='-',lw=plot_options['lw_plot'],rasterized=plot_options['rasterized'],zorder=1,alpha=plot_options['alpha_symb'],drawstyle=plot_options['drawstyle'],figure = all_figs[key_frame])   
                                            if plot_options['y_range'] is None:y_range_loc[key_frame] = [min(np.nanmin(var_loc),y_range_loc[key_frame][0]),max(np.nanmax(var_loc),y_range_loc[key_frame][1])]
        
                                            #Resampling
                                            if plot_options['resample'] is not None:
                                                var_resamp = bind.resampling(edge_bins_reg, data_precorr['edge_bins'][iord],var_loc, kind=gen_dic['resamp_mode'])   
                                                all_ax[key_frame].plot(cen_bins_reg,var_resamp,color=col_exp,linestyle='-',lw=plot_options['lw_plot'],rasterized=plot_options['rasterized'],zorder=2,drawstyle=plot_options['drawstyle'],figure = all_figs[key_frame])                      
                                    
        
                                        #Plot spectrum after correction   
                                        if plot_options['plot_post'] is not None:        
                                            var_loc = sc_fact*data_exp['flux'][iord]/mean_flux
                                            if plot_options['plot_err']:
                                                col_exp_err =    'orange'   
                                                col_exp_err = col_exp_sec                                           
                                                all_ax[key_frame].errorbar(cen_bins,var_loc,yerr = sc_fact*np.sqrt(data_exp['cov'][iord][0,:])/mean_flux,color=col_exp_err,linestyle='',lw=plot_options['lw_plot'],rasterized=plot_options['rasterized'],zorder=2,alpha=plot_options['alpha_err'],figure = all_figs[key_frame]) 
                                            all_ax[key_frame].plot(cen_bins,var_loc,color=col_exp_sec,linestyle='-',lw=plot_options['lw_plot'],rasterized=plot_options['rasterized'],zorder=3,drawstyle=plot_options['drawstyle'],figure = all_figs[key_frame])   
                                            if plot_options['y_range'] is None:y_range_loc[key_frame] = [min(np.nanmin(var_loc),y_range_loc[key_frame][0]),max(np.nanmax(var_loc),y_range_loc[key_frame][1])]
        
                                            #Resampling
                                            if plot_options['resample'] is not None:
                                                var_resamp = bind.resampling(edge_bins_reg, data_exp['edge_bins'][iord],var_loc, kind=gen_dic['resamp_mode'])   
                                                all_ax[key_frame].plot(cen_bins_reg,var_resamp,color=col_exp_sec,linestyle='-',lw=plot_options['lw_plot'],rasterized=plot_options['rasterized'],zorder=2,drawstyle=plot_options['drawstyle'],figure = all_figs[key_frame])                      
                                                                  
                                            #Telluric spectrum
                                            if plot_options['plot_tell']:
                                                tell_loc = tell_exp[iord]
                                                
                                                #Scale telluric spectrum (between tell_min and tell_max) to min/max of plotted spectrum
                                                tell_min = np.min(tell_loc[cond_def_exp])
                                                tell_max = np.max(tell_loc[cond_def_exp])
                                                min_f = np.min(var_loc[cond_def_exp])
                                                max_f = np.max(var_loc[cond_def_exp])
                                                def flux2tell(x):return (x -min_f )*(tell_max-tell_min)/(max_f-min_f) + tell_min 
                                                def tell2flux(x):return (x - tell_min)*(max_f-min_f)/(tell_max-tell_min) +min_f 
                                                all_ax[key_frame].plot(cen_bins,tell2flux(tell_loc),color='green',linestyle='-',lw=plot_options['lw_plot'],rasterized=plot_options['rasterized'],zorder=-10,drawstyle=plot_options['drawstyle'],figure = all_figs[key_frame])   
                                   
                                                #Secondary axis
                                                secax = all_ax[key_frame].secondary_yaxis('right', functions=(flux2tell, tell2flux))
                                                secax.tick_params('y', which='major',direction='in',labelsize=plot_options['font_size'])
                                                secax.set_ylabel('Telluric spectrum',fontsize=plot_options['font_size'])
                                                hide_yticks = True
        
                                            #HITRAN telluric lines
                                            for molec in plot_options['plot_tell_HITRANS']:
                                                cond_in_plot = (wave_tellL_exp[molec]>x_range_ord[0]) & (wave_tellL_exp[molec]<x_range_ord[1])
                                                for wave_tell_loc in wave_tellL_exp[molec][cond_in_plot]:
                                                    if plot_options['plot_tell']:
                                                        idx_tell = closest(cen_bins,wave_tell_loc)
                                                        if (1.-tell_loc[idx_tell])>plot_options['telldepth_min']:
                                                            plt.plot([wave_tell_loc,wave_tell_loc],y_range_loc[key_frame],linestyle='--',color='limegreen',lw=plot_options['lw_plot'])
                                                            plt.text(wave_tell_loc,y_range_loc[key_frame][1]+0.02*(y_range_loc[key_frame][1]-y_range_loc[key_frame][0]),str(molec),verticalalignment='center', horizontalalignment='center',fontsize=6.,zorder=10,color='green')
        
                                        #Plot flux balance master
                                        #    - reset to the level of the current spectrum for comparison
                                        if (gen_dic['corr_Fbal']):
                                            if (plot_options['plot_mast']): 
                                                all_ax[key_frame].plot(cen_bins,var_loc,color='black',linestyle='-',lw=1,rasterized=plot_options['rasterized'],zorder=-1,drawstyle=plot_options['drawstyle'],figure = all_figs[key_frame])                                 
                                            
                                            #Position of binned pixels used for the color balance fit
                                            if (plot_options['plot_bins']):
                                                Fbal_wav_bin_exp = data_Fbal_exp['Fbal_wav_bin_all']
                                                cond_plot_bins = (Fbal_wav_bin_exp[2]>x_range_ord[0]) & (Fbal_wav_bin_exp[0]<x_range_ord[1]) & (data_Fbal_exp['idx_ord_bin']==iord)
                                                if True in cond_plot_bins:   
                                                    for low_wbin,wbin,high_wbin in zip(Fbal_wav_bin_exp[0,cond_plot_bins],Fbal_wav_bin_exp[1,cond_plot_bins],Fbal_wav_bin_exp[2,cond_plot_bins]):
                                                        all_ax[key_frame].plot([low_wbin,low_wbin],y_range_loc[key_frame],linestyle='-',color='grey',figure = all_figs[key_frame])
                                                        all_ax[key_frame].plot([high_wbin,high_wbin],y_range_loc[key_frame],linestyle='-',color='grey',figure = all_figs[key_frame])
                                        
                                        #Plot continuum for persistent peak masking
                                        if (plot_options['data_permpeak'] is not None) and plot_options['plot_contmax']: 
                                            mean_flux_cont=np.sum(plot_options['data_permpeak']['cont_func_dic'][iord](cen_bins[cond_def_exp])*dcen_bins[cond_def_exp])/np.sum(dcen_bins[cond_def_exp])
                                            if plot_options['norm_prof']:var_loc=sc_fact*plot_options['data_permpeak']['cont_func_dic'][iord](cen_bins)/mean_flux_cont
                                            else:var_loc = sc_fact*plot_options['data_permpeak']['cont_func_dic'][iord](cen_bins)*mean_flux/mean_flux_cont
                                            all_ax[key_frame].plot(cen_bins,var_loc,color='black',linestyle='-',lw=1,rasterized=plot_options['rasterized'],zorder=10,figure = all_figs[key_frame])                                 
                                                                            
                                    #Plot other types of profiles
                                    else:       
                                        
                                        #-------------------------
                                        #Residuals between data and models 
                                        if ('_res' in plot_mod):
                                            edge_loc = edge_bins[idx_def_fit_raw[0]:idx_def_fit_raw[-1]+2]
                                            x_loc = cen_bins[cond_fit_exp_raw]
                                            var_loc=sc_fact*(flux_exp[cond_fit_exp_raw]-flux_mod_exp_fit)   
                                            evar_loc = sc_fact*err_exp[cond_fit_exp_raw]
                                            if plot_options['y_range'] is None:y_range_loc[key_frame] = [min(np.nanmin(var_loc),y_range_loc[key_frame][0]),max(np.nanmax(var_loc),y_range_loc[key_frame][1])]
                                            dy_range=y_range_loc[key_frame][1]-y_range_loc[key_frame][0]
                
                                            #Calculate mean and dispersion of the residuals 
                                            if plot_options['plot_prop']:  
                                                xtxt = x_range_ord[0]+0.1*dx_range
    
                                                #Over the fitted pixels (if fit) or all defined pixels (for reconstruction)   
                                                mean_res=np.mean(var_loc)          
                                                disp_from_mean=var_loc.std()   
                                                plt.plot([cen_bins[0],cen_bins[-1]],[mean_res,mean_res],linestyle='--',color='black',lw=plot_options['lw_plot'])
                                                plt.text(xtxt,y_range_loc[key_frame][1]-0.1*dy_range,'Mean in fitted range ='+"{0:.2e}".format(mean_res)+'+-'+"{0:.2e}".format(disp_from_mean),verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=4) 
                                                 
                                                #Residual from fit
                                                if plot_options['line_model']=='fit':
                                                    
                                                    #Over the continuum pixels
                                                    mean_res_cont=np.mean(var_loc[cond_cont_exp_fit])           
                                                    disp_from_mean_cont=(var_loc[cond_cont_exp_fit]).std()   
                                                    plt.plot([cen_bins[0],cen_bins[-1]],[mean_res_cont,mean_res_cont],linestyle=':',color='black',lw=plot_options['lw_plot'])                 
                                                    plt.text(xtxt,y_range_loc[key_frame][1]-0.2*dy_range,'Mean in continuum ='+"{0:.2e}".format(mean_res_cont)+'+-'+"{0:.2e}".format(disp_from_mean_cont),verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=4) 
                                                       
                                                    #Over the fitted range outside of the continuum
                                                    mean_res_nocont=np.mean(var_loc[~cond_cont_exp_fit])           
                                                    disp_from_mean_nocont=(var_loc[~cond_cont_exp_fit]).std()   
                                                    plt.plot([cen_bins[0],cen_bins[-1]],[mean_res_nocont,mean_res_nocont],linestyle=':',color='black',lw=plot_options['lw_plot'])                 
                                                    plt.text(xtxt,y_range_loc[key_frame][1]-0.3*dy_range,'Mean out continuum ='+"{0:.2e}".format(mean_res_nocont)+'+-'+"{0:.2e}".format(disp_from_mean_nocont),verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=4) 
                                    
    
                                        #-------------------------
                                        #Data
                                        else:
                                            edge_loc = edge_bins
                                            x_loc = cen_bins
                                            var_loc = sc_fact*flux_exp/mean_flux
                                            evar_loc = sc_fact*err_exp/mean_flux  
                                            if plot_options['y_range'] is None:y_range_loc[key_frame] = [min(np.nanmin(var_loc),y_range_loc[key_frame][0]),max(np.nanmax(var_loc),y_range_loc[key_frame][1])]
                                            dy_range=y_range_loc[key_frame][1]-y_range_loc[key_frame][0]
                                       
                                            #Continuum level
                                            if plot_options['plot_cont_lev'] and ('Intr' in plot_mod):
                                                if cond_mod:
                                                    if (plot_options['line_model']=='fit'):
                                                        if (plot_options['fit_type']=='indiv'):cont_lev = prof_fit_vis[iexp]['cont'] 
                                                        elif (plot_options['fit_type']=='global'):cont_lev = fit_results['p_final']['cont'] 
                                                    elif (plot_options['line_model']=='rec'):cont_lev = prof_fit_vis['cont']                                                     
                                                intr_lev=sc_fact*cont_lev/mean_flux
                                                plt.plot(x_range_ord,[intr_lev,intr_lev],linestyle='-',color='black',lw=plot_options['lw_plot'])                  
    
                                            #Plot measurements 
                                            if plot_options['plot_biss'] and (('DI' in plot_mod) or ('Intr' in plot_mod)):                          
                                                plt.plot(mod_prop_exp['RV_biss'],sc_fact*mod_prop_exp['F_biss']/mean_flux,color='black',linestyle='--',lw=plot_options['lw_plot'])    
    
                                            #Model
                                            if cond_mod:
                                                if (plot_options['plot_line_model'] or plot_options['plot_line_model_HR']): 
                                                    col_mod = 'black'
                                                    # col_mod = col_exp_sec
                                                    
                                                    #Plot model profile
                                                    #    - specific to a given order (single tables are defined)
                                                    var_mod = sc_fact*flux_mod_exp/mean_flux
                                                    if gen_dic['flux_sc'] and ('bin' not in plot_mod) and (plot_mod=='Res_prof'):
                                                        loc_flux_scaling = dataload_npz(data_dic[inst][vis]['scaled_'+data_type_gen+'_data_paths']+str(iexp_or))['loc_flux_scaling']   
                                                        var_mod*=loc_flux_scaling(cen_bins_mod_exp) 
                                                    if plot_options['plot_line_model']: 
                                                        all_ax[key_frame].plot(cen_bins_mod_exp,var_mod,color=col_mod,linestyle='--',lw=plot_options['lw_plot'],zorder=-1+15)                                    
                                                    if plot_options['plot_line_model_HR']:
                                                        var_mod_HR = sc_fact*mod_prop_exp['flux_HR']/mean_flux                      
                                                        all_ax[key_frame].plot(mod_prop_exp['cen_bins_HR'],var_mod_HR,color=col_mod,linestyle='-',lw=plot_options['lw_plot'],zorder=-1) 
                                                 
                                                    #Plot fitted pixels
                                                    if plot_options['plot_fitpix']: 
                                                        # all_ax[key_frame].plot(cen_bins_mod_exp[cond_fit_exp], var_mod[cond_fit_exp],color='black',linestyle='',lw=0.5,marker='o',markersize=1) 
                                                        all_ax[key_frame].plot(cen_bins[cond_fit_exp_raw], var_loc[cond_fit_exp_raw],color='black',linestyle='',lw=0.5,marker='o',markersize=3) 
                                                        
                                                    #Plot continuum pixels
                                                    #    - specific to the exposure, used to measure dispersion
                                                    if plot_options['plot_cont_exp']: 
                                                        # all_ax[key_frame].plot(cen_bins_mod_exp[cond_cont_exp], var_mod[cond_cont_exp],markeredgecolor='black',linestyle='',lw=0.5,marker='d',markersize=3,markerfacecolor='none')                  
                                                        all_ax[key_frame].plot(cen_bins[cond_cont_exp_raw], var_loc[cond_cont_exp_raw],markeredgecolor='black',linestyle='',lw=0.5,marker='d',markersize=3,markerfacecolor='none')                  
    
                                                    #Oplot continuum pixels from fits
                                                    if plot_options['plot_cont']:
                                                        cond_cont_com  = np.all(prof_fit_vis['cond_def_cont_all'],axis=0)
                                                        plt.plot(cen_bins[cond_cont_com],var_loc[cond_cont_com],color='black',linestyle='',lw=plot_options['lw_plot'],marker='d',markersize=2,zorder=10)  
    
                                                    #Plot individual model components
                                                    if (plot_options['plot_line_model_compo']):
                                                        if ('gauss_core' in mod_prop_exp):plt.plot(cen_bins,sc_fact*mod_prop_exp['gauss_core']/mean_flux,color='black',linestyle='--',lw=plot_options['lw_plot'])     
                                                        if ('gauss_lobe' in mod_prop_exp):plt.plot(cen_bins,sc_fact*mod_prop_exp['gauss_lobe']/mean_flux,color='black',linestyle='--',lw=plot_options['lw_plot'])   
                                                        if ('poly_lobe' in mod_prop_exp): plt.plot(cen_bins,sc_fact*mod_prop_exp['poly_lobe']/mean_flux,color='black',linestyle='--',lw=plot_options['lw_plot'])    
                                                        if ('core' in mod_prop_exp): plt.plot(cen_bins,sc_fact*mod_prop_exp['core']/mean_flux,color='black',linestyle='--',lw=plot_options['lw_plot'])    
    
                                        #-------------------------
                                        #Plotting original data                                      
                                        if (not plot_options['no_orig']):
                                            if plot_options['plot_err']:all_ax[key_frame].errorbar(x_loc,var_loc,yerr = evar_loc ,color=col_exp,linestyle='-',lw=plot_options['lw_plot'],marker=None,rasterized=plot_options['rasterized'],zorder=1,alpha=plot_options['alpha_err'],figure = all_figs[key_frame]) 
                                            all_ax[key_frame].plot(x_loc,var_loc,color=col_exp,linestyle=plot_options['ls_plot'],lw=plot_options['lw_plot'],marker=None,alpha=plot_options['alpha_symb'],rasterized=plot_options['rasterized'],zorder=2,drawstyle=plot_options['drawstyle'],figure = all_figs[key_frame])                                           
                                            if plot_options['y_range'] is None: y_range_loc[key_frame] = [min(0.9*np.nanmin(var_loc),y_range_loc[key_frame][0]),max(1.1*np.nanmax(var_loc),y_range_loc[key_frame][1])]
                                            else:y_range_loc[key_frame] = sc_fact*np.array(plot_options['y_range'])                                       
    
                                        #Resampling
                                        if plot_options['resample'] is not None:
                                            var_resamp = bind.resampling(edge_bins_reg,edge_loc,var_loc, kind=gen_dic['resamp_mode'])   
                                            all_ax[key_frame].plot(cen_bins_reg,var_resamp,color=col_exp,linestyle='-',lw=plot_options['lw_plot'],rasterized=plot_options['rasterized'] & False,zorder=2,drawstyle=plot_options['drawstyle'],figure = all_figs[key_frame])                      

                                    #Plot stellar continuum
                                    if ('spec' in data_type) and (plot_options['st_cont'] is not None) and (('DI' in plot_mod) or ('Intr' in plot_mod)):                      
                                        all_ax[key_frame].plot(x_loc,sc_fact*cont_func_dic(x_loc)/mean_flux ,color='black',linestyle='-',lw=plot_options['lw_plot']-0.5,zorder=-1) 
                                 
                       
                                    #Single order options
                                    if (not plot_options['multi_ord']):
        
                                        #----------------------------------------
                                        #Complementary features
                                        #----------------------------------------
                                        dy_range=y_range_loc[key_frame][1]-y_range_loc[key_frame][0]
                                        
                                        if (not plot_options['multi_exp'] or isub==1):
        
                                            #Reference null level
                                            if ('_res' in plot_mod):
                                                all_ax[key_frame].axhline(0.,color='black', lw=plot_options['lw_plot'],linestyle='--') 
        
                                            #Shade range requested in plot
                                            if (inst in plot_options['shade_ranges']) and (vis in plot_options['shade_ranges'][inst]):
                                                plot_shade_range(all_ax[key_frame],plot_options['shade_ranges'][inst][vis],x_range_ord,None,mode='span',compl=True,zorder=100,alpha=0.3)
        
                                            #Shade effective continuum range used in fit
                                            if plot_options['shade_cont'] and (prof_fit_vis is not None):
                                                for i_int,bd_int in enumerate(prof_fit_vis['cont_range']):
                                                    all_ax[key_frame].axvspan(bd_int[0],bd_int[1], facecolor='dodgerblue', alpha=0.1) 
        
                                            #Oplot reference velocity
                                            if plot_options['plot_refvel'] and ('CCF' in plot_mod):
                                                if (plot_mod in ['DI_prof','DI_prof_res']):refvel=data_dic['DI']['sysvel'][inst][vis]
                                                else:refvel=0.
                                                all_ax[key_frame].plot([refvel,refvel], y_range_loc[key_frame],color='black',lw=plot_options['lw_plot'],linestyle=':') 
        
                                        #Print measurements 
                                        if plot_options['print_mes']:                              
                                            plt.text(x_range_ord[0]+0.1*dx_range,y_range_loc[1]-0.1*dy_range,'Mean signal='+"{0:.2f}".format(1e6*plot_options['data_meas']['int_sign'][iexp])+'+-'+"{0:.2f}".format(1e6*plot_options['data_meas']['e_int_sign'][iexp])+' ppm ('+"{0:.2f}".format(plot_options['data_meas']['R_sign'][iexp])+'$\sigma$)' ,
                                                    verticalalignment='center', horizontalalignment='left',fontsize=10.,zorder=4,color='black') 
        
        
        
                                        #----------------------------------------
                                        #Fit properties
                                        if cond_mod and (plot_options['line_model']=='fit'):
                                            
                                            #Shade area not included within fit
                                            #    - using range from settings
                                            if plot_options['shade_unfit']:
                                                plot_shade_range(all_ax[key_frame],prof_fit_vis['fit_range'],x_range_ord,None,mode='span',compl=True)
                                                # plot_shade_range(all_ax[key_frame],prof_fit_vis['fit_range'],x_range_ord,y_range_loc[key_frame],mode='hatch',compl=True)
        
                                            #Plot measured centroid
                                            #    - corresponds to rv(star/sun) for raw CCFs, and to rv(region/star) for local CCFs
                                            if plot_options['plot_line_fit_rv'] and ('CCF' in plot_mod):
                                                plt.plot([mod_prop_exp['rv'],mod_prop_exp['rv']], y_range_loc[key_frame],color='black',lw=plot_options['lw_plot'],linestyle=':')                             
        
                                            #Print fit properties
                                            if plot_options['plot_prop']:
                                                xtxt = x_range_ord[0]+0.02*(x_range_ord[1]-x_range_ord[0])
                                                ytxt = y_range_loc[key_frame][0]+0.3*dy_range
                                                ygap = -0.1*dy_range
                                                
                                                if (plot_options['fit_type']=='global'):
                                                    plt.text(xtxt,ytxt+ygap,'BIC = '+"{0:.5f}".format(fit_results['merit']['BIC'])+' ($\chi^2_r$ = '+"{0:.5f}".format(fit_results['merit']['red_chi2'])+')',verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40)  
                                                    disp_res=(flux_exp[cond_fit_exp_raw]-flux_mod_exp_fit).std()
                                                    plt.text(xtxt,ytxt+2*ygap,'RMS (res.) = '+"{0:.5f}".format(disp_res),verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40)  
          
                                                if (plot_mod in ['DI_prof','DIbin','Intr_prof','Intrbin','Atm_prof','Atmbin']):                    
                        
                                                    #Intrinsic and atmospheric CCFs
                                                    ytxt = y_range_loc[key_frame][1]
                                                    if (plot_mod in ['Intr_prof','Intrbin','Atm_prof','Atmbin']): 
                                                    
                                                        #Oplot original index
                                                        if (plot_mod in ['Intr_prof','Atm_prof']): 
                                                            plt.text(x_range_ord[0]+0.1*dx_range,ytxt-0.3*dy_range,'i$_\mathrm{all}$ ='+str(iexp_or),verticalalignment='center', horizontalalignment='left',fontsize=15.,zorder=40) 
                            
                                                        #Indicate if individual CCF is detected
                                                        if (plot_options['fit_type']=='indiv'):
                                                            txt_detection='Forced ' if (mod_prop_exp['forced_det']) else ''                     
                                                            if mod_prop_exp['detected']:txt_detection+='Detected' 
                                                            else: txt_detection+='Undetected' 
                                                            if ('crit_area' in mod_prop_exp):txt_detection+=' (R$_\mathrm{ctrst}$='+"{0:.2f}".format(mod_prop_exp['crit_area'])+'$\sigma$)'
                                                            plt.text(x_range_ord[0]+0.1*dx_range,ytxt-0.4*dy_range,txt_detection,verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40) 
                                        
                                                    #Main fit properties
                                                    if (plot_mod in ['DI_prof','DIbin','Intr_prof','Intrbin','Atm_prof','Atmbin']):
                                                        xtxt = np.mean(x_range_ord)+0.05*dx_range
                                                        ygap = -0.1*dy_range
                                                        if (plot_options['fit_type']=='indiv'):    
                                                            plt.text(xtxt,ytxt+ygap,'BIC = '+"{0:.5f}".format(mod_prop_exp['BIC'])+' ($\chi^2_r$ = '+"{0:.5f}".format(mod_prop_exp['red_chi2'])+')',verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40)  
                                                            plt.text(xtxt,ytxt+2*ygap,'RV ='+stackrel(mod_prop_exp['rv'],mod_prop_exp['err_rv'][0],mod_prop_exp['err_rv'][1],"0:.4f")+' km s$^{-1}$',verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40)                                                 
                                                            if ('ctrst' in mod_prop_exp):plt.text(xtxt,ytxt+3*ygap,'C ='+stackrel(mod_prop_exp['ctrst'],mod_prop_exp['err_ctrst'][0],mod_prop_exp['err_ctrst'][1],"0:.4f"),verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40) 
                                                            if ('FWHM' in mod_prop_exp):plt.text(xtxt,ytxt+4*ygap,'FWHM ='+stackrel(mod_prop_exp['FWHM'],mod_prop_exp['err_FWHM'][0],mod_prop_exp['err_FWHM'][1],"0:.4f")+' km s$^{-1}$',verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40) 
                                                            if ('FWHM_LOR' in mod_prop_exp):plt.text(xtxt,ytxt+5*ygap,'FWHM[Lor] ='+stackrel(mod_prop_exp['FWHM_LOR'],mod_prop_exp['err_FWHM_LOR'][0],mod_prop_exp['err_FWHM_LOR'][1],"0:.4f")+' km s$^{-1}$',verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40)                                                                         
                                                            if ('FWHM_voigt' in mod_prop_exp):plt.text(xtxt,ytxt+6*ygap,'FWHM[Voigt] ='+stackrel(mod_prop_exp['FWHM_voigt'],mod_prop_exp['err_FWHM_voigt'][0],mod_prop_exp['err_FWHM_voigt'][1],"0:.4f")+' km s$^{-1}$',verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40)                                     
                                                            if ('cont_amp' in mod_prop_exp):
                                                                disp_from_mean_cont=((flux_exp - mod_prop_exp['flux'])[cond_cont_exp_raw]).std()
                                                                if (plot_mod in  ['Intr_prof','Intrbin']) and (data_dic['Intr']['model'][inst]=='dgauss'):
                                                                    plt.text(xtxt,ytxt+6.*ygap,'cont. amp ='+stackrel(mod_prop_exp['cont_amp'],mod_prop_exp['err_cont_amp'][0],mod_prop_exp['err_cont_amp'][1],"0:.2e"),verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40) 
                                                                    plt.text(xtxt,ytxt+6.5*ygap,'           '+"{0:.2f}".format(mod_prop_exp['cont_amp']/np.mean(mod_prop_exp['err_cont_amp']))+'$\sigma$ ; '+"{0:.2f}".format(mod_prop_exp['cont_amp']/disp_from_mean_cont)+'$\sigma_\mathrm{cont}$',verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40)     
                                                                plt.text(xtxt,ytxt+5.*ygap,'amp ='+stackrel(mod_prop_exp['amp'],mod_prop_exp['err_amp'][0],mod_prop_exp['err_amp'][1],"0:.2e"),verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40) 
                                                                plt.text(xtxt,ytxt+5.5*ygap,'           '+"{0:.2f}".format(mod_prop_exp['amp']/np.mean(mod_prop_exp['err_amp']))+'$\sigma$ ; '+"{0:.2f}".format(mod_prop_exp['cont_amp']/disp_from_mean_cont)+'$\sigma_\mathrm{cont}$',verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40)                                 
                                                        elif (plot_options['fit_type']=='global') and (plot_mod in ['Intr_prof','Atm_prof']):
                                                            plt.text(xtxt,ytxt+2*ygap,'RV (intr.) ='+"{0:.4f}".format(mod_prop_exp['rv'])+' km s$^{-1}$',verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40)                                                      
                                                            plt.text(xtxt,ytxt+3*ygap,'ctrst (intr.) ='+"{0:.4f}".format(mod_prop_exp['ctrst']),verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40) 
                                                            plt.text(xtxt,ytxt+4*ygap,'FWHM (intr.) ='+"{0:.4f}".format(mod_prop_exp['FWHM'])+' km s$^{-1}$',verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40) 
                                                            if ('cont_amp' in mod_prop_exp):
                                                                disp_from_mean_cont=((flux_exp - mod_prop_exp['flux'])[cond_cont_exp_raw]).std()
                                                                if (plot_mod == 'Intr_prof') and (data_dic['Intr']['model'][inst]=='dgauss'):
                                                                    plt.text(xtxt,ytxt+6.*ygap,'cont. amp ='+stackrel(mod_prop_exp['cont_amp'],mod_prop_exp['err_cont_amp'][0],mod_prop_exp['err_cont_amp'][1],"0:.2e"),verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40) 
                                                                    plt.text(xtxt,ytxt+6.5*ygap,'           '+"{0:.2f}".format(mod_prop_exp['cont_amp']/np.mean(mod_prop_exp['err_cont_amp']))+'$\sigma$ ; '+"{0:.2f}".format(mod_prop_exp['cont_amp']/disp_from_mean_cont)+'$\sigma_\mathrm{cont}$',verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40)     
                                                                plt.text(xtxt,ytxt+5.*ygap,'amp ='+"{0:.2e}".format(mod_prop_exp['amp'])+'+-'+"{0:.2e}".format(mod_prop_exp['err_amp']),verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40) 
                                                                plt.text(xtxt,ytxt+5.5*ygap,'           '+"{0:.2f}".format(mod_prop_exp['amp']/mod_prop_exp['err_amp'])+'$\sigma$ ; '+"{0:.2f}".format(mod_prop_exp['cont_amp']/disp_from_mean_cont)+'$\sigma_\mathrm{cont}$',verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40) 
        
            
                                        #----------------------------------------    
                                        #Shade range of planetary signal excluded for relevant exposures
                                        if (data_dic['Atm']['exc_plrange']) and (plot_options['plot_plexc']) and (iexp in data_dic['Atm'][inst][vis]['iexp_no_plrange']):
        
                                            #Disk-integrated profiles
                                            #    - ranges excluded from profile fit
                                            if (plot_mod in ['DI_prof','DI_prof_res']) and (idx_excl_bd_ranges is not None):
                                                for bd_int in idx_excl_bd_ranges:
                                                    low_bd = data_exp['edge_bins'][0,0:-1][bd_int[0]]
                                                    high_bd = data_exp['edge_bins'][0,1::][bd_int[1]]
                                                    if (low_bd<x_range_ord[1]) and (high_bd>x_range_ord[0]):
                                                        plt.axvspan(low_bd,high_bd, facecolor='red', alpha=0.1,zorder=10) 
                                                        
                                            # elif (plot_mod=='Res_prof'):
                                            #     for pl_loc in data_dic[inst][vis]['transit_pl']:
                                            #         bd_int=data_dic['Atm']['plrange'] + coord_dic[inst][vis][pl_loc]['rv_pl'][iexp_or]
                                            #         if (bd_int[0]<1e10):plt.axvspan(bd_int[0],bd_int[1], facecolor='red', alpha=0.1,zorder=10) 
                                            # elif (plot_mod in ['Atm_prof','Atm_prof_res','Atmbin','Atmbin_res']):
                                            #     bd_int=data_dic['Atm']['plrange']
                                            #     if (bd_int[0]<1e10):plt.axvspan(bd_int[0],bd_int[1], facecolor='red', alpha=0.1,zorder=10) 
                                            
                                            elif (plot_mod=='Res'): 
                                                wrange_line = data_dic['Atm'][inst][vis]['exclu_range_star'][:,:,iexp]
                                            elif (plot_mod in ['Atm','Atmbin','sp_Atm_1D']):
                                                wrange_line= data_dic['Atm']['CCF_mask_wav'][:,None] * np.sqrt(1. + (data_dic['Atm']['plrange']/c_light) )/np.sqrt(1. - (data_dic['Atm']['plrange']/c_light) ) 
                                            cond_lines_in = (wrange_line[:,0]>=x_range_ord[0]) & (wrange_line[:,1]<=x_range_ord[1]) 
                                            for iline in range(np.sum(cond_lines_in)):
                                                all_ax[key_frame].axvspan(wrange_line[iline,0],wrange_line[iline,1], facecolor='red', alpha=0.1,zorder=10) 
        
                                        #----------------------------------------
                                        #Plot frame 
                                        #    - end frame for current order if :
                                        # + a single order per plot is plotted
                                        # + a single exposure per order is plotted, or last exposure has been processed
                                        #----------------------------------------
                                        if ((not plot_options['multi_exp']) or (isub==len(iexp_plot)-1)): 
                                            filename = end_plot_prof(pl_ref,inst,vis,all_figs[key_frame],all_ax[key_frame],x_range_loc[key_frame],y_range_loc[key_frame],hide_yticks,plot_options,iexp_or,plot_mod,title_name,ref_name,iord,plot_ext,x_title,y_title,path_loc,data_bin)
     
                                            #Store image for GIF generation
                                            if (images_to_make_GIF is not None): 
                                                if (not plot_options['multi_exp']):images_to_make_GIF[iord].append(imageio.v2.imread(filename))   
                                                else:images_to_make_GIF.append(imageio.v2.imread(filename))   
                
                        ### End of loop on orders         

                        #----------------------------------------
                        #Plot frame 
                        #    - end frame after current exposure and all orders have been processed if :
                        # + a single exposure per order is plotted, or last exposure has been processed
                        # + all orders are plotted together
                        #----------------------------------------
                        if plot_options['multi_ord'] and ((not plot_options['multi_exp']) or (isub==len(iexp_plot)-1)): 
                            filename = end_plot_prof(pl_ref,inst,vis,all_figs[key_frame],all_ax[key_frame],x_range_loc[key_frame],y_range_loc[key_frame],hide_yticks,plot_options,iexp_or,plot_mod,title_name,ref_name,None,plot_ext,x_title,y_title,path_loc,data_bin)    

                    ### End of condition to plot current exposure

                ### End of loop on exposures

                #----------------------------------------
                #Plot frame 
                #    - end frame after all exposures and orders have been processed if :
                # + all exposures are plotted together
                # + all orders are plotted together
                #----------------------------------------
                if plot_options['multi_ord'] and plot_options['multi_exp']:
                    filename = end_plot_prof(pl_ref,inst,vis,all_figs[key_frame],all_ax[key_frame],x_range_loc[key_frame],y_range_loc[key_frame],hide_yticks,plot_options,None,plot_mod,title_name,ref_name,None,plot_ext,x_title,y_title,path_loc,data_bin)                

                #Generate GIF
                if (images_to_make_GIF is not None):
                    
                    #For each order, over exposure series
                    if (not plot_options['multi_exp']):
                        for iord in images_to_make_GIF:
                            if (plot_mod in ['DI_prof','DImast','Res','Intr','atm']) and (data_dic['Res']['type'][inst]=='spec2D'):str_add='_iord'+str(iord)
                            else:str_add=''                            
                            imageio.mimsave(path_loc+'/'+ref_name+str_add+'.gif', images_to_make_GIF[iord],duration=(1000 * 1/plot_options['fps']))
                    
                    #Over order series, with multiple exposures per plot
                    else:imageio.mimsave(path_loc+'/multi_exp_'+ref_name+'.gif', images_to_make_GIF,duration=(1000 * 1/plot_options['fps']))

               
        return None

    def pre_proc_exp(plot_options,inst,vis,maink_list,iexp_plot,iexp_mast_list,data_inst,data_vis,data_path_dic,idx_sel_ord,cen_bins_com,edge_bins_com,nord_proc,dim_exp_proc,data_list,fixed_args_loc):
        print('           Pre-processing exposures')
        data_proc = {} 
        data4mast = {} 
        for maink in maink_list:
            data_proc[maink] = {} 
            data4mast[maink] = {}                            
        data_mod = {}
        iexp_proc_list = np.unique(list(iexp_mast_list)+list(iexp_plot)) 
        flux_ref = np.ones(data_vis['dim_exp']) 
        for isub_exp,iexp in enumerate(iexp_proc_list):

            #Aligning profiles in star rest frame (source), shifting them from the solar system barycentric (receiver) rest frame
            #    - see gen_specdopshift():
            # w_source = w_receiver / (1+ (rv[s/r]/c))
            # w_star = w_starbar / (1+ (rv[star/starbar]/c))
            # w_star = w_solbar / ((1+ (rv[star/starbar]/c))*(1+ (rv[starbar/solbar]/c)))
            dopp_fact = 1./(gen_specdopshift(coord_dic[inst][vis]['RV_star_stelCDM'][iexp])*gen_specdopshift(data_dic['DI']['sysvel'][inst][vis]))  
       
            #Retrieve data
            for maink in maink_list:
                data_proc[maink][iexp] = dataload_npz(data_path_dic[plot_options['plot_'+maink]]+str(iexp)) 
                for key in ['flux','cov','cond_def','cen_bins','edge_bins']:data_proc[maink][iexp][key] = data_proc[maink][iexp][key][idx_sel_ord]
                if data_vis['tell_sp']:data_proc[maink][iexp]['tell'] = dataload_npz(data_vis['tell_DI_data_paths'][iexp])['tell'][idx_sel_ord]
                else:data_proc[maink][iexp]['tell'] = None
                if gen_dic['cal_weight']:data_proc[maink][iexp]['mean_gdet'] = dataload_npz(data_vis['mean_gdet_DI_data_paths'][iexp])['mean_gdet'] 
                else:data_proc[maink][iexp]['mean_gdet']=None   
             
                #Only the exposure table is modified if data do not share a common table
                #    - data will be resampled along with the master in a later stage
                if (not data_vis['comm_sp_tab']):
                    data_proc[maink][iexp]['edge_bins']*=dopp_fact
                    data_proc[maink][iexp]['cen_bins']*=dopp_fact                        

                #Exposure is resampled on common table otherwise
                else:
                    for isub,iord in enumerate(idx_sel_ord): 
                        data_proc[maink][iexp]['flux'][isub],data_proc[maink][iexp]['cov'][isub] = bind.resampling(edge_bins_com[isub], data_proc[maink][iexp]['edge_bins'][isub]*dopp_fact, data_proc[maink][iexp]['flux'][isub], cov = data_proc[maink][iexp]['cov'][isub], kind=gen_dic['resamp_mode'])
                    if data_vis['tell_sp']:data_proc[maink][iexp]['tell'] = bind.resampling(edge_bins_com[isub], data_proc[maink][iexp]['edge_bins'][isub]*dopp_fact, data_proc[maink][iexp]['tell'][iord] , kind=gen_dic['resamp_mode']) 
                    if data_inst['cal_weight']:data_proc[maink][iexp]['mean_gdet'] = bind.resampling(edge_bins_com[isub], data_proc[maink][iexp]['edge_bins'][isub]*dopp_fact, data_proc[maink][iexp]['mean_gdet'][iord] , kind=gen_dic['resamp_mode']) 
                    data_proc[maink][iexp]['cond_def'] = ~np.isnan(data_proc[maink][iexp]['flux'])
                    data_proc[maink][iexp]['edge_bins']=   edge_bins_com
                    data_proc[maink][iexp]['cen_bins'] =   cen_bins_com                    
                
                #Normalize to global flux unity
                #    - so that spectra are at a comparable level 
                dcen_wav = data_proc[maink][iexp]['edge_bins'][:,1::] - data_proc[maink][iexp]['edge_bins'][:,0:-1]
                cond_def = data_proc[maink][iexp]['cond_def']
                flux_glob = 0.
                for isub in range(nord_proc):
                    flux_glob+= np.sum(dcen_wav[isub][cond_def[isub]])/np.sum(data_proc[maink][iexp]['flux'][isub][cond_def[isub]]*dcen_wav[isub][cond_def[isub]])
                for isub in range(nord_proc):
                    data_proc[maink][iexp]['flux'][isub],data_proc[maink][iexp]['cov'][isub] = bind.mul_array(data_proc[maink][iexp]['flux'][isub] , data_proc[maink][iexp]['cov'][isub],np.repeat(flux_glob,data_vis['nspec']))
   
                #Exposures used in master calculations
                if iexp in iexp_mast_list:
                    data4mast[maink][iexp]={}
                    for key in ['cen_bins','edge_bins','flux','cov','cond_def']:data4mast[maink][iexp][key]=deepcopy(data_proc[maink][iexp][key])
                    
                    #Weight definition   
                    #    - at this stage, no broadband flux scaling has been applied to the data
                    data4mast[maink][iexp]['weight'] = weights_bin_prof(idx_sel_ord,None,inst,vis,gen_dic['corr_Fbal'],gen_dic['corr_FbalOrd'],gen_dic['save_data_dir'],gen_dic['type'],nord_proc,iexp,'DI',data_inst['type'],dim_exp_proc,data_proc[maink][iexp]['tell'],data_proc[maink][iexp]['mean_gdet'],data_proc[maink][iexp]['cen_bins'],1.,flux_ref,None,glob_flux_sc = 1./flux_glob)                       
       
                    #Resampling if exposures do not share a common table
                    if (not data_vis['comm_sp_tab']):   
                        for isub in range(nord_proc):
                            data4mast[maink][iexp]['flux'][isub],data4mast[maink][iexp]['cov'][isub] = bind.resampling(edge_bins_com[isub], data4mast[maink][iexp]['edge_bins'][isub], data_proc[maink][iexp]['flux'][isub] , cov = data_proc[maink][iexp]['cov'][isub], kind=gen_dic['resamp_mode'])                                                        
                            data4mast[maink][iexp]['weight'][isub] = bind.resampling(edge_bins_com[isub], data4mast[maink][iexp]['edge_bins'][isub], data4mast[maink][iexp]['weight'][isub] ,kind=gen_dic['resamp_mode'])   
                        data4mast[maink][iexp]['cond_def'] = ~np.isnan(data4mast[maink][iexp]['flux']) 
                        data4mast[maink][iexp]['cen_bins'] = cen_bins_com
                        data4mast[maink][iexp]['edge_bins'] = edge_bins_com
              
            #Calculating wiggle model for current exposure and shifting it from the telluric (source) to the star (receiver) rest frame
            #    - see gen_specdopshift():
            # w_receiver = w_source * (1+ (rv[s/r]/c))
            # w_star = w_starbar * (1+ (rv[starbar/star]/c))
            #        = w_solbar * (1- (rv_kep/c)) * (1+ (rv[solbar/starbar]/c))
            #        = w_Earth * (1- (rv_kep/c)) * (1- (rv_sys/c)) * (1+ (rv[Earth/solbar]/c))
            #        = w_Earth * (1- (rv_kep/c)) * (1- (rv_sys/c)) * (1+ (BERV/c))
            if ('wiggle' in data_list) and plot_options['plot_model']:
                data_mod[iexp] = {}
                data_mod[iexp]['wig_HR']=calc_wig_mod_nu_t(fixed_args_loc['nu_HR'],p_best,{**fixed_args_loc,'icoord':iexp})[0]
                data_mod[iexp]['nu_HR'] = fixed_args_loc['nu_HR']*gen_specdopshift(-coord_dic[inst][vis]['RV_star_stelCDM'][iexp])*gen_specdopshift(-data_dic['DI']['sysvel'][inst][vis])*gen_specdopshift(data_prop[inst][vis]['BERV'][iexp])*(1.+1.55e-8)  
     
        return data_proc,data_mod,data4mast



    def end_plot_prof(pl_ref,inst,vis,fig_frame,ax_frame,x_range_frame,y_range_frame,hide_yticks,plot_options,iexp_or,plot_mod,title_name,ref_name,iord,plot_ext,x_title,y_title,path_loc,data_bin):
        if (not plot_options['multi_exp']):
            if (plot_mod in ['DIbin','DIbin_res','DImast']):                                        
                ref_val=data_bin['cen_bindim'][iexp_or]
            elif (plot_mod in ['Intrbin','Intrbin_res','Atmbin','Atmbin_res','Intrbin','Atmbin']):
                ref_val=data_bin['cen_bindim'][iexp_or]
            # elif (plot_mod in ['sp_Intr_1D','sp_Atm_1D']) and (plot_options['dim_plot']=='bin'):
            #     ref_val=data_bin['cen_bindim'][iexp]
            else:
                ref_val=coord_dic[inst][vis][pl_ref]['cen_ph'][iexp_or]  
        else:ref_val=None
        if plot_options['title']:
            title = title_name
            if 'spec' in data_type:title+=' spectrum'
            else:title+=' CCF'
            if ref_val is not None:title+=' (visit ='+vis+'; '+ref_name+' ='+"{0:.5f}".format(ref_val)+')'
            else:title+=' (visit ='+vis+')'
            ax_frame.title(title,fontsize=plot_options['font_size'])                      
        if plot_options['x_range'] is None:
            dx_range=x_range_frame[1]-x_range_frame[0]
            x_range_frame = [x_range_frame[0]-0.05*dx_range,x_range_frame[1]+0.05*dx_range]
        else:x_range_frame = plot_options['x_range']
        if plot_options['y_range'] is None:y_range_frame = [0.95*y_range_frame[0],1.05*y_range_frame[1]]
        else:y_range_frame = plot_options['y_range']
        dx_range=x_range_frame[1]-x_range_frame[0]
        dy_range=y_range_frame[1]-y_range_frame[0]
        xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
        ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)
        custom_axis(plt,fig = fig_frame,ax=ax_frame,position=plot_options['margins'],x_range=x_range_frame,y_range=y_range_frame,
                    xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,
                    xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                    hide_yticks = hide_yticks,dir_x = 'out',
                    # dir_y='out',
                    # axis_thick=plot_options['axis_thick']
                    hide_axis = plot_options['hide_axis'],
                    x_title=x_title,y_title=y_title,
                    font_size=plot_options['font_size'],xfont_size=plot_options['font_size'],yfont_size=plot_options['font_size'])
  
        if not plot_options['multi_ord']:
            if (plot_mod in ['DI_prof','DImast','Res','Intr','atm']) and (data_dic['Res']['type'][inst]=='spec2D'):str_add='_iord'+str(iord)
            else:str_add=''
        else:str_add='multi_ord'
        if not plot_options['multi_exp']:
            str_idx = 'idx'+"{:d}".format(iexp_or)
            loc_str='_'   
            if vis=='binned':
                idx_exp2in = data_bin['idx_exp2in']
                idx_in = data_bin['idx_in']
                idx_out = data_bin['idx_out']
            else:
                idx_exp2in = gen_dic[inst][vis]['idx_exp2in']
                idx_in = gen_dic[inst][vis]['idx_in']
                idx_out = gen_dic[inst][vis]['idx_out']                
            if iexp_or in idx_in:loc_str='_in'+str(idx_exp2in[iexp_or])
            elif iexp_or in idx_out:loc_str='_out'+str(iexp_or)                                         
            str_mid=str_idx+loc_str 
        else:    
            str_mid='multiexp'
        filename = path_loc+'/'+str_mid+'_'+ref_name
        if ref_val is not None:filename += "{0:.5f}".format(ref_val)
        filename+=str_add+'.'+plot_ext
        fig_frame.savefig(filename,transparent=plot_options['transparent'])                        
        plt.close() 

        return filename





    '''
    Sub-functions to plot individual transmission profiles 
        - for disk-integrated data in the star rest frame
        - a common master is used
    '''
    def sub_plot_DI_trans(plot_options,plot_mod,plot_ext):
        
        #Plot for each instrument        
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_options['visits_to_plot'].keys())): 
            print('   - Instrument :',inst)
            for key in ['color_dic','color_dic_sec','color_dic_bin','color_dic_bin_sec']:
                if inst not in plot_options[key]:plot_options[key][inst]={}
                data_type = data_dic['DI']['type'][inst]
            
            #Data to plot
            maink_list = []
            data_list = []
            if plot_options['plot_pre'] is not None:
                maink_list+=['pre']
                data_list+=[plot_options['plot_pre']]
            if plot_options['plot_post'] is not None:
                maink_list+=['post']
                data_list+=[plot_options['plot_post']]       

            #Frame properties
            title_name=''
            
            #Plot for each visit
            for vis in np.intersect1d(list(data_dic[inst].keys()),plot_options['visits_to_plot'][inst]): 
                print('     - Visit :',vis)
                if vis not in plot_options['color_dic'][inst]:plot_options['color_dic'][inst][vis] ='red' 
                if vis not in plot_options['color_dic_bin'][inst]:plot_options['color_dic_bin'][inst][vis] ='limegreen' 
                if vis not in plot_options['color_dic_sec'][inst]:plot_options['color_dic_sec'][inst][vis] ='dodgerblue' 
                if vis not in plot_options['color_dic_bin_sec'][inst]:plot_options['color_dic_bin_sec'][inst][vis] ='orange' 
  
                #Create directory if required
                path_loc = gen_dic['save_plot_dir']+'Spec_raw/DI_trans/'+inst+'_'+vis+'/'
                if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  

                #Data
                data_inst=data_dic[inst]
                data_vis = data_inst[vis]
                data_com = dataload_npz(data_vis['proc_com_data_paths'])
                
                rest_frame='star'
                fixed_args_loc = {}
                if 'wiggle' in data_list:
                    data_com_wig = np.load(gen_dic['save_data_dir']+'/Corr_data/Wiggles/Vis_fit/'+inst+'_'+vis+'_add.npz',allow_pickle=True)['data'].item()   

                    #Correction model
                    if (vis in plot_options['wig_path_corr']):data_fit = (np.load(plot_options['wig_path_corr'][vis],allow_pickle=True)['data'].item())
                    else:data_fit = dataload_npz(data_com_wig['corr_path'])
                    if ('iexp2glob' not in data_fit):p_best = data_fit['p_best']
 
                #Spectral variable
                if plot_options['aligned']:title_name='Aligned '+title_name
                xt_str={'input':'heliocentric','star':'star','surf':'surface','pl':'planet'}[rest_frame]
                if plot_options['x_range'] is not None:x_range_loc = plot_options['x_range']
                if 'spec' in data_type:
                    if plot_options['sp_var'] == 'nu' :
                        if plot_options['x_range'] is None:x_range_loc = [c_light/9000.,c_light/3000.]
                        x_title = r'$\nu$ in '+xt_str+' rest frame (10$^{-10}$s$^{-1}$)'
                        
                    elif plot_options['sp_var'] == 'wav' :
                        if plot_options['x_range'] is None:x_range_loc = [3000.,9000.] 
                        x_title = r'Wavelength in '+xt_str+' rest frame (A)'
                        
                else:x_title='Velocity in '+xt_str+' rest frame (km s$^{-1}$)'     
                
                #Selected ranges and orders
                cond_sel = np.zeros(data_vis['dim_exp'],dtype=bool)
                if plot_options['sp_var'] == 'nu' :edge_bins_var = c_light/data_com['edge_bins'][:,::-1]    
                elif plot_options['sp_var'] == 'wav' :edge_bins_var = data_com['edge_bins']
                cond_sel|=(edge_bins_var[:,0:-1]>x_range_loc[0]) & (edge_bins_var[:,1::]<x_range_loc[1])
                idx_sel_ord = np_where1D(np.sum( cond_sel,axis=1 )>0 )
                if len(plot_options['orders_to_plot'])>0:idx_sel_ord=np.intersect1d(idx_sel_ord,plot_options['orders_to_plot'])
                cond_sel_proc = cond_sel[idx_sel_ord]
                nord_proc = len(idx_sel_ord)
                if nord_proc==0:stop('No orders left')
                dim_exp_proc = [nord_proc,data_vis['nspec']]
                edge_bins_var = edge_bins_var[idx_sel_ord]
                cen_bins_com = data_com['cen_bins'][idx_sel_ord]
                edge_bins_com = data_com['edge_bins'][idx_sel_ord]
                if plot_options['x_range'] is None:x_range_loc = [np.max([x_range_loc[0],np.min(edge_bins_var)]),np.min([x_range_loc[1],np.max(edge_bins_var)])]
                dx_range = x_range_loc[1]-x_range_loc[0]

                #Retrieve wiggle model
                if 'wiggle' in data_list:

                    #Oversampled model spectral table
                    if plot_options['plot_model']:
                        dnu_HR = 0.02
                        if plot_options['sp_var'] == 'nu' :min_nu,max_nu = x_range_loc[0],x_range_loc[1]
                        else:min_nu,max_nu = c_light/x_range_loc[1],c_light/x_range_loc[0]
                        min_nu_HR = min_nu-10.*dnu_HR
                        max_nu_HR = max_nu+10.*dnu_HR
                        n_nu_HR,fixed_args_loc['nu_HR']  = def_wig_tab(min_nu_HR,max_nu_HR,dnu_HR)            
              
                    #Pointing coordinate series
                    if ('iexp2glob' not in data_fit):
                        fixed_args_loc.update({
                            'z_mer' : data_vis['z_mer'],
                            'deg_Freq':data_com_wig['deg_Freq'], 
                            'deg_Amp':data_com_wig['deg_Amp'], 
                            'nu_ref':data_com_wig['nu_ref'],
                            'comp_mod':data_com_wig['comp_ids'],
                            'nexp_list':len(range(data_vis['n_in_visit'])),
                            'stable_pointpar':data_fit['stable_pointpar']})                
                        for key in ['az','x_az','y_az','z_alt','cond_eastmer','cond_westmer','cond_shift']:fixed_args_loc[key] = data_com_wig['tel_coord_vis'][key]                 
                        calc_chrom_coord(p_best,fixed_args_loc)

                #Data at chosen steps
                data_path_dic = get_data_path('DI_prof','spec',inst,vis)

                #Exposures to process
                if (inst in plot_options['iexp_plot']) and (vis in plot_options['iexp_plot'][inst]):
                    iexp_plot = plot_options['iexp_plot'][inst][vis]
                else:iexp_plot = range(data_vis['n_in_visit'])   

                #Indexes of master exposures
                if (inst in plot_options['iexp_mast_list']) and (vis in plot_options['iexp_mast_list'][inst]):
                    if plot_options['iexp_mast_list'][inst][vis]=='all':iexp_mast_list = np.arange(data_vis['n_in_visit'])
                    else:iexp_mast_list = plot_options['iexp_mast_list'][inst][vis]
                else:iexp_mast_list = gen_dic[inst][vis]['idx_out']
                
                #Pre-process all exposures
                data_proc,data_mod,data4mast = pre_proc_exp(plot_options,inst,vis,maink_list,iexp_plot,iexp_mast_list,data_inst,data_vis,data_path_dic,idx_sel_ord,cen_bins_com,edge_bins_com,nord_proc,dim_exp_proc,data_list,fixed_args_loc)  
     
                #Calculate master for requested data steps
                data_mast={}
                disp_dic={}
                for maink in maink_list:
                
                    #Calculate common master
                    #    - defined over common table if exposures are defined over independent tables, or over common table shared by all exposures                       
                    data_mast[maink] = calc_bin_prof(iexp_mast_list,nord_proc,dim_exp_proc,data_dic[inst]['nspec'],data4mast[maink],inst,len(iexp_mast_list),cen_bins_com,edge_bins_com)

                    #Dispersion tables
                    if plot_options['print_disp']:disp_dic[maink] = np.zeros([2,len(iexp_plot)])*np.nan 
                
                #Calculating and plotting transmission spectra
                #    - all exposures are now defined in the star rest frame
                print('           Processing and plotting exposures')
                for isub_exp,iexp in enumerate(iexp_plot):
                    plt.ioff()        
                    fig = plt.figure(figsize=plot_options['fig_size'])
            
                    #Vertical range
                    y_min=1e100
                    y_max=-1e100

                    #Adding progressively bins of current spectrum and master
                    for maink,datak in zip(maink_list,data_list):
               
                        #Resampling common master on current exposure table
                        if (not data_vis['comm_sp_tab']):
                            data_mast_exp={'flux':np.zeros(dim_exp_proc,dtype=float)*np.nan,'cov':np.zeros(nord_proc,dtype=object)}
                            for isub in range(nord_proc): 
                                data_mast_exp['flux'][isub],data_mast_exp['cov'][isub] = bind.resampling(data_proc[maink][iexp]['edge_bins'][isub],edge_bins_com[isub],data_mast[maink]['flux'][isub] , cov = data_mast[maink]['cov'][isub], kind=gen_dic['resamp_mode'])                                                       
                            data_mast_exp['cond_def'] = ~np.isnan(data_mast_exp['flux'])  
    
                        #Master table is shared by all exposures
                        else:data_mast_exp = deepcopy(data_mast[maink])

                        #Defined pixels in exposure and master over selected range
                        cond_kept_all = data_proc[maink][iexp]['cond_def'] & data_mast_exp['cond_def'] & cond_sel_proc
                        idx_proc_ord = np_where1D(np.sum( cond_kept_all,axis=1 )>0 )
                        if len(idx_proc_ord)==0:stop('No orders left')
                        if len(maink_list)==1:
                            off_sign = 1.
                            yoff=0.
                        else:
                            off_sign = {'pre':1.,'post':-1.}[maink] 
                            yoff=0.5*plot_options['gap_exp']*off_sign
                        raw_Fr_all = np.empty(0,dtype=float)
                        bin_Fr_all = np.empty(0,dtype=float)
                        for isub_ord in idx_proc_ord[::-1]:
                            cond_kept_ord = cond_kept_all[isub_ord]
                            mast_flux_ord=data_mast[maink]['flux'][isub_ord]
                            flux_ord=data_proc[maink][iexp]['flux'][isub_ord]
                            cov_ord=data_proc[maink][iexp]['cov'][isub_ord]
                            
                            #Spectral grid
                            if plot_options['sp_var'] == 'nu' :                
                                cen_bins_ord = c_light/data_proc[maink][iexp]['cen_bins'][isub_ord,::-1]
                                edge_bins_ord = c_light/data_proc[maink][iexp]['edge_bins'][isub_ord,::-1]    
                                
                                #Order tables with nu 
                                cond_kept_ord = cond_kept_ord[::-1]
                                mast_flux_ord=mast_flux_ord[::-1]
                                flux_ord=flux_ord[::-1]
                                cov_ord=cov_ord[:,::-1]   
                                    
                            elif plot_options['sp_var'] == 'wav' :
                                cen_bins_ord = data_proc[maink][iexp]['cen_bins'][isub_ord]
                                edge_bins_ord = data_proc[maink][iexp]['edge_bins'][isub_ord]                            

                            #Defining bins at the requested resolution over the range of original defined bins
                            min_pix = np.nanmin(edge_bins_ord[0:-1][cond_kept_ord])
                            max_pix = np.nanmax(edge_bins_ord[1::][cond_kept_ord])
                            n_bins_init=int(np.ceil((max_pix-min_pix)/plot_options['bin_width']))
                            bin_siz=(max_pix-min_pix)/n_bins_init
                            bin_bd=np.append(min_pix+bin_siz*np.arange(n_bins_init,dtype=float),max_pix)  
                            bin_dic={}
                            bin_dic['cen_bins'] = 0.5*(bin_bd[0:-1]+bin_bd[1::])
                            bin_dic['dcen_bins'] = (bin_bd[1::]-bin_bd[0:-1])

                            #Resampling at lower resolution
                            flux_rb,cov_rb = bind.resampling(bin_bd, edge_bins_ord, flux_ord , cov = cov_ord, kind=gen_dic['resamp_mode'])
                            mast_rb = bind.resampling(bin_bd, edge_bins_ord, mast_flux_ord , kind=gen_dic['resamp_mode'])

                            #New bins have constant width so that flux and flux density are equivalent
                            bin_dic['Fr'] = flux_rb/mast_rb
                            bin_dic['varFr'] = cov_rb[0]/mast_rb**2.
                            conddef_bin = ~np.isnan(bin_dic['Fr'] )
                            for key in ['Fr','varFr','cen_bins','dcen_bins']:bin_dic[key]=bin_dic[key][conddef_bin]

                            #Set binned ratio over current order to a constant level unity   
                            if plot_options['force_unity']:                     
                                corr_Fr = np.sum(bin_dic['dcen_bins'])/np.sum(bin_dic['Fr']*bin_dic['dcen_bins'])  
                                bin_dic['Fr']*=corr_Fr
                                bin_dic['varFr']*=corr_Fr**2.
                            else:corr_Fr=1.

                            #Plot binned data
                            bin_Fr_all = np.append(bin_Fr_all,bin_dic['Fr'])
                            if plot_options['plot_bin']:
                                if maink=='pre':col_loc = plot_options['color_dic_bin'][inst][vis]
                                elif maink=='post':col_loc = plot_options['color_dic_bin_sec'][inst][vis]
                                plt.plot(bin_dic['cen_bins'],yoff+bin_dic['Fr'],color=col_loc,linestyle='-',lw=plot_options['lw_plot'],marker=plot_options['marker'],markersize=plot_options['markersize'],drawstyle=plot_options['drawstyle'],zorder=1)   
                                if plot_options['plot_bin_err']:plt.errorbar(bin_dic['cen_bins'],yoff+bin_dic['Fr'],yerr = np.sqrt(bin_dic['varFr']),color=col_loc,linestyle='',lw=plot_options['lw_plot'],marker=None,alpha=plot_options['alpha_err'],zorder=1) 

                            #Plot transmission spectrum at original resolution
                            if plot_options['plot_data']:
                                cond_def_raw = (~np.isnan(flux_ord)) & (~np.isnan(mast_flux_ord))
                                raw_Fr = corr_Fr*flux_ord[cond_def_raw]/mast_flux_ord[cond_def_raw]   
                                raw_Fr_all = np.append(raw_Fr_all,raw_Fr)
                                if maink=='pre':col_loc = plot_options['color_dic'][inst][vis]
                                elif maink=='post':col_loc = plot_options['color_dic_sec'][inst][vis]
                                plt.plot(cen_bins_ord[cond_def_raw], yoff+raw_Fr,color=col_loc,linestyle='-',lw=plot_options['lw_plot'],marker=plot_options['marker'],markersize=plot_options['markersize'],drawstyle=plot_options['drawstyle'],zorder=0,alpha = 0.5)   
                                if plot_options['plot_err']:
                                    raw_varFr = corr_Fr**2.*cov_ord[0][cond_def_raw]/mast_flux_ord[cond_def_raw]**2.    
                                    plt.errorbar(cen_bins_ord[cond_def_raw], yoff+raw_Fr,yerr = np.sqrt(raw_varFr),color=col_loc,linestyle='-',lw=plot_options['lw_plot'],marker=None,markersize=plot_options['markersize'],zorder=0,alpha = 0.3)   

                        #Range
                        if plot_options['y_range'] is None:
                            if plot_options['plot_data']:
                                y_max=max(np.max(yoff+raw_Fr_all),y_max)
                                y_min=min(np.min(yoff+raw_Fr_all),y_min)
                            elif plot_options['plot_bin']:
                                y_max=max(np.max(yoff+bin_Fr_all),y_max)
                                y_min=min(np.min(yoff+bin_Fr_all),y_min)
                     
                        #Dispersions
                        if plot_options['print_disp']:
                            ytxt = 1. 
                            if maink=='pre':xtxt =x_range_loc[0]+0.05*dx_range 
                            if maink=='post':xtxt =x_range_loc[1]-0.4*dx_range
                            rms_txt = 'RMS['+datak+'.]'
                            if plot_options['plot_data']:
                                rms_loc = (raw_Fr_all-1.).std()
                                rms_txt += " = {0:.2e}".format(rms_loc)+' (raw)'
                                disp_dic[maink][0,isub_exp]=rms_loc
                            if plot_options['plot_bin']:
                                rms_loc = (bin_Fr_all-1.).std()
                                rms_txt += " = {0:.2e}".format(rms_loc)+' (bin)'
                                disp_dic[maink][1,isub_exp]=rms_loc
                            plt.text(xtxt,ytxt,rms_txt,verticalalignment='center', horizontalalignment='left',fontsize=8.,zorder=10,color=col_loc)

                    #Plot model
                    if ('wiggle' in data_list) and plot_options['plot_model']:
                        if plot_options['sp_var']=='nu':x_HR = data_mod[iexp]['nu_HR']
                        elif plot_options['sp_var']=='wav':x_HR = c_light/data_mod[iexp]['nu_HR']
                        plt.plot(x_HR,0.5*plot_options['gap_exp']+data_mod[iexp]['wig_HR'],color='red',linestyle='-',lw=plot_options['lw_plot'],zorder=2)   

                    #Plot level
                    if len(maink_list)==1:
                        plt.plot(x_range_loc,np.repeat(1.,2),color='limegreen',linestyle='--',lw=plot_options['lw_plot'],zorder=3)   
                    else:
                        plt.plot(x_range_loc,np.repeat(1.-0.5*plot_options['gap_exp'],2),color=col_loc,linestyle='--',lw=plot_options['lw_plot'],zorder=3)   
                        plt.plot(x_range_loc,np.repeat(1.+0.5*plot_options['gap_exp'],2),color=col_loc,linestyle='--',lw=plot_options['lw_plot'],zorder=3)   
                    
                    #Plot order index
                    y_range_loc = plot_options['y_range'] if plot_options['y_range'] is not None else [y_min,y_max]
                    dy_range=y_range_loc[1]-y_range_loc[0]
                    if plot_options['plot_idx_ord']:
                        if plot_options['sp_var']=='nu':x_ord = c_light/gen_dic['wav_ord_inst'][inst]
                        elif plot_options['sp_var']=='wav':x_ord = gen_dic['wav_ord_inst'][inst]
                        if dx_range<2:dord=1
                        elif dx_range<10:dord=2
                        else:dord=4  
                        for iord in np.arange(0,len(x_ord),dord):
                            if (x_ord[iord]>x_range_loc[0]) and (x_ord[iord]<x_range_loc[1]):
                                if is_odd(iord):delt_txt = 0.03
                                else:delt_txt = 0.06
                                plt.text(x_ord[iord],y_range_loc[1]+delt_txt*dy_range,str(iord),verticalalignment='center', horizontalalignment='center',fontsize=6.,zorder=15,color='black') 

                    #Shade ranges not included in the fit
                    if ('wiggle' in data_list) and (plot_options['shade_unfit']) and (vis in data_com_wig['wig_range_fit']):
                        plot_shade_range(plt.gca(),data_com_wig['wig_range_fit'][vis],x_range_loc,y_range_loc,mode='fill',facecolor='dodgerblue',compl=True)                  

                    #Frame
                    xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                    ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)
                    if plot_options['sp_var']=='nu':x_title=r'Nu (10$^{-10}$s$^{-1}$)'
                    elif plot_options['sp_var']=='wav':x_title=r'Wavelength (A)'
                    custom_axis(plt,position=plot_options['margins'],x_range=x_range_loc,y_range=y_range_loc,xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                                x_title=x_title,y_title='Flux ratio',font_size=plot_options['font_size'],xfont_size=plot_options['font_size'],yfont_size=plot_options['font_size'])
                    plt.savefig(path_loc+'idx'+str(iexp)+'.'+plot_dic['trans_sp']) 
                    plt.close()      
                 

                #Dispersion plots
                if plot_options['print_disp']!=[] & plot_options['plot_disp']:
                
                    #Frame
                    plt.ioff() 
                    fig = plt.figure(figsize=plot_options['fig_size'])
                    x_min=1e100
                    x_max=-1e100
                    if (inst in plot_options['y_range_disp']) and (vis in plot_options['y_range_disp'][inst]):y_range_loc =  plot_options['y_range_disp'][inst][vis]
                    else:
                        y_range_loc=None
                        y_min=1e100
                        y_max=-1e100                            
                    
                    #Plot for requested step
                    mean_disp = {}
                    bin_types=[]
                    if plot_options['plot_data']:bin_types+=['raw']
                    if plot_options['plot_bin']:bin_types+=['bin']
                    for maink in disp_dic:
                        mean_disp[maink]={}
                        for bin_type in bin_types:
                            if bin_type=='raw':
                                ibin=0
                                if maink=='pre':col_loc = plot_options['color_dic'][inst][vis]
                                elif maink=='post':col_loc = plot_options['color_dic_sec'][inst][vis]
                            elif bin_type=='bin':
                                ibin=1
                                if maink=='pre':col_loc = plot_options['color_dic_bin'][inst][vis]
                                elif maink=='post':col_loc = plot_options['color_dic_bin_sec'][inst][vis]
                            cond_def = ~np.isnan(disp_dic[maink][ibin,isub_exp])
                            cen_ph_vis = coord_dic[inst][vis][plot_options['pl_ref'][inst][vis]]['cen_ph'][cond_def]
                            disp_maink = 1e6*disp_dic[maink][ibin,cond_def]
                            plt.plot(cen_ph_vis,disp_maink,color=col_loc,linestyle='',zorder=1,marker='o',markersize=1.5)
    
                            #Boundaries
                            x_min=np.min([np.nanmin(cen_ph_vis),x_min])
                            x_max=np.max([np.nanmax(cen_ph_vis),x_max]) 
                            dx_range = x_max-x_min
                            x_range_loc = [x_min-0.05*dx_range,x_max+0.05*dx_range]     
                            if (y_range_loc is None):
                                y_min=np.min([np.nanmin(disp_maink),y_min])
                                y_max=np.max([np.nanmax(disp_maink),y_max]) 

                            #Mean dispersion
                            mean_disp[maink][bin_type] = np.mean(disp_maink)
                            plt.plot(x_range_loc,[mean_disp[maink][bin_type],mean_disp[maink][bin_type]],color=col_loc,linestyle='-',zorder=1,lw=0.5)

                    #Ranges
                    dx_range=x_range_loc[1]-x_range_loc[0]   
                    if (y_range_loc is None):
                        dy_range = y_max-y_min
                        y_range_loc = [y_min-0.05*dy_range,y_max+0.05*dy_range]     
                    dy_range=y_range_loc[1]-y_range_loc[0] 

                    #Print on-screen   
                    for bin_type in bin_types:     
                        if bin_type=='raw':ytxt = y_range_loc[0]+0.6*dy_range   
                        elif bin_type=='bin':ytxt = y_range_loc[0]+0.4*dy_range   
                        for maink,datak in zip(maink_list,data_list): 
                            if maink=='pre':
                                xtxt =x_range_loc[0]+0.05*dx_range 
                                if bin_type=='raw':col_loc = plot_options['color_dic'][inst][vis]
                                elif bin_type=='bin':col_loc = plot_options['color_dic_bin'][inst][vis]
                            if maink=='post':
                                xtxt =x_range_loc[1]-0.4*dx_range
                                if bin_type=='raw':col_loc = plot_options['color_dic_sec'][inst][vis]
                                elif bin_type=='bin':col_loc = plot_options['color_dic_bin_sec'][inst][vis] 
                            rms_txt = 'RMS['+datak+'.]'+ " = {0:.2e}".format(mean_disp[maink][bin_type])+' ('+bin_type+')'
                            plt.text(xtxt,ytxt,rms_txt,verticalalignment='center', horizontalalignment='left',fontsize=8.,zorder=10,color=col_loc)
            
                    #Frame                                                 
                    xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                    ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)
                    custom_axis(plt,position=plot_options['margins'],x_range=x_range_loc,y_range=y_range_loc,xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                                x_title=r'Orbital Phase',y_title='RMS (ppm)',font_size=plot_options['font_size'],xfont_size=plot_options['font_size'],yfont_size=plot_options['font_size'])
                    plt.savefig(path_loc+'Dispersion.'+plot_dic['trans_sp']) 
                    plt.close() 

        return None


















    '''
    Sub-function to plot all profiles from a given visit together
        - for disk-integrated and intrinsic data
    '''
    def sub_plot_all_prof(plot_options,plot_mod,plot_ext):
        sc_fact=10**plot_options['sc_fact10']

        #Plot frame  
        if plot_mod=='glob_mast':
            xt_str='input'
            title_name='disk-integrated'
            y_title='Flux'
        else:
            if plot_mod=='DI':
                xt_str='star'
                title_name='disk-integrated'
                y_title='Flux'
            elif plot_mod=='Intr':           
                xt_str='local'  
                title_name='intrinsic' 
                y_title='Flux' 
            elif plot_mod=='Atm':           
                xt_str='planet'  
                title_name='atmospheric '+plot_options['pl_atm_sign']   
                if plot_options['pl_atm_sign']=='Absorption':y_title='Absorption'
                elif plot_options['pl_atm_sign']=='Emission':y_title='Flux'
        plot_options['add_txt_path'] = {'DI':'','Intr':'','Atm':plot_options['pl_atm_sign']+'/'}
        if plot_mod!='glob_mast':add_txt_path = plot_options['add_txt_path'][plot_mod]
        suff_txt_path = {'DI':'_','glob_mast':'','Intr':'_','Atm':'_'}[plot_mod]
        
        #Plot for each instrument
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_options['visits_to_plot'].keys())):   
            if plot_mod=='glob_mast':data_type = gen_dic['type'][inst]   
            else:data_type = data_dic[plot_mod]['type'][inst]
        
            #Plot for each visit
            for vis in np.intersect1d(list(data_dic[inst].keys()),plot_options['visits_to_plot'][inst]): 
                print('     - visit '+vis)
                data_vis = data_dic[inst][vis]
                if plot_mod=='glob_mast':
                    path_loc = gen_dic['save_plot_dir']+'Spec_raw/FluxBalance/Global_master/'+inst+'_'+vis+'/' 
                    al_txt = 'Aligned '
                    sav_txt = ''
                elif plot_mod in ['DI','Intr','Atm']:
                    if plot_options['aligned']:
                        sav_txt = 'Aligned_'
                        al_txt = 'Aligned '
                    else:
                        sav_txt = ''
                        al_txt = '' 
                    path_loc = gen_dic['save_plot_dir']+sav_txt+plot_mod+'_data/'+add_txt_path+inst+'_'+vis+'_All/'                     
                if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)   
 
                #Exposures to plot
                if (plot_mod=='glob_mast'):
                    if plot_options['plot_input']:iexp_plot = range(data_vis['n_in_visit'])
                    else:iexp_plot=[0]           
                else:
                    if plot_mod=='DI':iexp_plot = range(data_vis['n_in_visit'])
                    elif plot_mod=='Intr':iexp_plot = data_dic[plot_mod][inst][vis]['idx_def']          
                    elif plot_mod=='Atm': iexp_plot = data_dic[plot_mod][inst][vis]['idx_def']             
                if ('iexp_plot' in plot_options) and (inst in plot_options['iexp_plot']) and (vis in plot_options['iexp_plot'][inst]):
                    iexp_plot = np.intersect1d(iexp_plot,plot_options['iexp_plot'][inst][vis])
                nexp_plot=len(iexp_plot)                
                
                #Visit colors                
                if (inst in plot_options['color_dic']) and (vis in plot_options['color_dic'][inst]) and not ((plot_mod=='glob_mast') and ('spec' in data_type)):
                    col_visit=np.repeat(plot_options['color_dic'][inst][vis],nexp_plot)
                else:
                    cmap = plt.get_cmap('jet') 
                    col_visit=np.array([cmap(0)]) if nexp_plot==1 else cmap( np.arange(nexp_plot)/(nexp_plot-1.))

                #------------------------------------------------------------------------------------------
                        
                #Paths
                if plot_mod in['DI','Intr','Atm']:path_exp = gen_dic['save_data_dir']+sav_txt+plot_mod+'_data/'+add_txt_path+inst+'_'+vis+suff_txt_path    

                #Plotting CCF profiles
                if data_type=='CCF':
                    plt.ioff()                    
                    plt.figure(figsize=plot_options['fig_size'])
                    
                    #Plotting each exposure
                    y_range_loc=sc_fact*np.array(plot_options['y_range']) if plot_options['y_range'] is not None else [1e100,-1e100] 
                    for isub,iexp in enumerate(iexp_plot):
                        
                        #Retrieving data
                        data_exp=np.load(path_exp+str(iexp)+'.npz',allow_pickle=True)['data'].item()   
                        var_loc=sc_fact*data_exp['flux'][0]
                 
                        #Approximate normalization to set CCFs to comparable levels for the plot or to a mean unity                       
                        if plot_mod=='DI':           
                            if len(data_dic['DI']['scaling_range'])>0:
                                cond_def_scal=False 
                                for bd_int in data_dic['DI']['scaling_range']:cond_def_scal |= (data_exp['edge_bins'][0,0:-1]>=bd_int[0]) & (data_exp['edge_bins'][0,1:]<=bd_int[1])   
                            else:cond_def_scal=True                                               
                            cond_def_scal&=data_exp['cond_def'][0]
                            mean_flux=np.mean(var_loc[cond_def_scal])                         
                 
                        #Normalizing unscaled DI CCF so that they can be comparable
                        if plot_mod=='DI':var_loc/=mean_flux
                        plt.plot(data_exp['cen_bins'][0],var_loc ,color=col_visit[isub],linestyle='-',lw=plot_options['lw_plot'],rasterized=plot_options['rasterized'])    
                        if plot_options['y_range'] is None:y_range_loc = [0.99*min(np.nanmin(var_loc),y_range_loc[0]),1.01*max(np.nanmax(var_loc),y_range_loc[1])]
                        if isub==0:                            
                            x_range_loc=np.array(plot_options['x_range']) if plot_options['x_range'] is not None else np.array([data_exp['cen_bins'][0,0]-5.,data_exp['cen_bins'][0,-1]+5.])                       
                            
                            #Oplot pixels effectively used as continuum
                            #    - only pixels within the requested range common to all exposures after planetary ranges are excluded are kept
                            # if plot_options['plot_cont']:plt.plot(data_exp['cen_bins'][0][cond_cont],var_loc[cond_cont],color='black',linestyle='',marker='d',markersize=1,zorder=10,rasterized=plot_options['rasterized'])      
                            
                    #Plot frame               
                    if plot_options['title']:plt.title(al_txt+title_name+' CCFs in '+xt_str+' rest frame',fontsize=plot_options['font_size'])
                    y_title = scaled_title(plot_options['sc_fact10'],y_title)                       
                    dx_range=x_range_loc[1]-x_range_loc[0]
                    dy_range=y_range_loc[1]-y_range_loc[0]
                    xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                    ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range) 
                    custom_axis(plt,position=plot_options['margins'],x_range=x_range_loc,y_range=y_range_loc,
                                xmajor_int=xmajor_int,xminor_int=xminor_int,
                                ymajor_int=ymajor_int,yminor_int=yminor_int,
                                xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                                x_title='Velocity in '+xt_str+' rest frame (km s$^{-1}$)',y_title=y_title,
                                font_size=plot_options['font_size'],xfont_size=plot_options['font_size'],yfont_size=plot_options['font_size'])
                  						

                    plt.savefig(path_loc+data_type+'.'+plot_ext) 
                    plt.close()    
                                
                #------------------------------------------------------------------------------------------
                        
                #Plotting spectral profiles
                elif 'spec' in data_type:
                        
                    #Retrieving data
                    if plot_mod=='glob_mast':
                        data_mast_vis = np.load(gen_dic['save_data_dir']+'Corr_data/Global_Master/'+inst+'_'+vis+'_meas.npz',allow_pickle=True)['data'].item() 
                        path_exp = data_mast_vis['proc_DI_data_paths']
                        if plot_options['glob_mast_all']=='meas':data_mast_vis_all = np.load(gen_dic['save_data_dir']+'Corr_data/Global_Master/'+inst+'_meas.npz',allow_pickle=True)['data'].item()  
                        elif plot_options['glob_mast_all']=='theo':data_mast_vis_all = np.load(gen_dic['save_data_dir']+'Corr_data/Global_Master/'+inst+'_'+vis+'_theo.npz',allow_pickle=True)['data'].item()        
                    cond_def_exp_all={}
                    wav_exp_all={}
                    edge_bins_exp_all={}
                    sp_exp_all={}
                    for iexp in iexp_plot:                    
                        data_exp=np.load(path_exp+str(iexp)+'.npz',allow_pickle=True)['data'].item()    
                        cond_def_exp_all[iexp] = data_exp['cond_def']
                        wav_exp_all[iexp] = data_exp['cen_bins']
                        edge_bins_exp_all[iexp] = data_exp['edge_bins']
                        sp_exp_all[iexp] = data_exp['flux']    

                        #Shift individual exposures to the star rest frame in which master is defined
                        if plot_mod=='glob_mast': 
                            dop_sh = data_mast_vis['specdopshift_star_solbar'][iexp]    
                            wav_exp_all[iexp]*=dop_sh
                            edge_bins_exp_all[iexp]*=dop_sh

                    #Order list
                    order_list = plot_options['orders_to_plot'] if len(plot_options['orders_to_plot'])>0 else range(data_dic[inst]['nord']) 
                        
                    #Plot each order of spectra for all exposures
                    for iord in order_list:

                        #Plot order if it overlaps with the requested window
                        plot_ord=False
                        min_wav = np.min([edge_bins_exp_all[iexp][iord][0:-1][cond_def_exp_all[iexp][iord]][0] for iexp in iexp_plot])
                        max_wav = np.max([edge_bins_exp_all[iexp][iord][1::][cond_def_exp_all[iexp][iord]][-1] for iexp in iexp_plot])
                        if plot_options['x_range'] is not None:
                            if (max_wav>=plot_options['x_range'][0]) and (min_wav<=plot_options['x_range'][1]):plot_ord=True
                            x_range_loc = deepcopy(plot_options['x_range'])
                        else:
                            drange = max_wav-min_wav
                            x_range_loc = [min_wav-0.02*drange,max_wav+0.02*drange]     
                            plot_ord=True
                        if plot_ord:                                         
                            plt.ioff()                    
                            plt.figure(figsize=plot_options['fig_size'])
                            y_range_loc=sc_fact*np.array(plot_options['y_range']) if plot_options['y_range'] is not None else [1e100,-1e100] 

                            #Resampling
                            if plot_options['resample'] is not None:
                                n_reg = int(np.ceil((x_range_loc[1]-x_range_loc[0])/plot_options['resample']))
                                edge_bins_reg = np.linspace(x_range_loc[0],x_range_loc[1],n_reg)
                                cen_bins_reg = 0.5*(edge_bins_reg[0:-1]+edge_bins_reg[1::])

                            #Plot all exposures
                            if plot_options['plot_input']:
                                for isub,iexp in enumerate(iexp_plot):
                                    cond_def_tab = cond_def_exp_all[iexp][iord] & (wav_exp_all[iexp][iord]>x_range_loc[0]) & (wav_exp_all[iexp][iord]<x_range_loc[1])
                                    wav_tab = wav_exp_all[iexp][iord][cond_def_tab]
                                    var_loc=sc_fact*sp_exp_all[iexp][iord][cond_def_tab]
                                    plt.plot(wav_tab,var_loc,color=col_visit[isub],linestyle='-',lw=plot_options['lw_plot'],rasterized=plot_options['rasterized'],zorder=1,alpha=plot_options['alpha_symb']) 
                                    if plot_options['y_range'] is None:y_range_loc = [0.99*min(sc_fact*np.nanmin(var_loc),y_range_loc[0]),1.01*max(sc_fact*np.nanmax(var_loc),y_range_loc[1])]                        

                                    #Resampling
                                    if plot_options['resample'] is not None:
                                        var_resamp = bind.resampling(edge_bins_reg, edge_bins_exp_all[iexp][iord], sc_fact*sp_exp_all[iexp][iord], kind=gen_dic['resamp_mode'])   
                                        plt.plot(cen_bins_reg,var_resamp,color=col_visit[isub],linestyle='-',lw=plot_options['lw_plot'],rasterized=plot_options['rasterized'],zorder=2)                            
                            
                            
                            #Plot master
                            if plot_mod=='glob_mast':

                                #Global master of current visit
                                if plot_options['glob_mast_vis']:
                                    cond_in_plot = (data_mast_vis['cen_bins'][iord]>x_range_loc[0]) & (data_mast_vis['cen_bins'][iord]<x_range_loc[1])
                                    plt.plot(data_mast_vis['cen_bins'][iord,cond_in_plot],sc_fact*data_mast_vis['flux'][iord,cond_in_plot],color='black',linestyle='-',lw=plot_options['lw_plot']+0.5,rasterized=False,zorder=2) 
                                    if plot_options['y_range'] is None:y_range_loc = [0.99*min(sc_fact*np.nanmin(data_mast_vis['flux'][iord,cond_in_plot]),y_range_loc[0]),1.01*max(sc_fact*np.nanmax(data_mast_vis['flux'][iord,cond_in_plot]),y_range_loc[1])] 
                                                  
                                #Global master over all visits
                                if plot_options['glob_mast_all'] is not None:
                                    cond_in_plot = (data_mast_vis_all['cen_bins'][iord]>x_range_loc[0]) & (data_mast_vis_all['cen_bins'][iord]<x_range_loc[1])
                                    plt.plot(data_mast_vis_all['cen_bins'][iord,cond_in_plot],sc_fact*data_mast_vis_all['flux'][iord,cond_in_plot],color='dimgrey',linestyle='-',lw=plot_options['lw_plot']+1,rasterized=False,zorder=2) 
                                    if plot_options['y_range'] is None:y_range_loc = [0.99*min(sc_fact*np.nanmin(data_mast_vis_all['flux'][iord,cond_in_plot]),y_range_loc[0]),1.01*max(sc_fact*np.nanmax(data_mast_vis_all['flux'][iord,cond_in_plot]),y_range_loc[1])] 

                            #Overplot stellar lines
                            if len(plot_options['st_lines_wav'])>0:
                                st_lines_wav = np.array(plot_options['st_lines_wav'])
                                cond_in_plot = (st_lines_wav>x_range_loc[0]) & (st_lines_wav<x_range_loc[1])
                                for st_line_wav in st_lines_wav[cond_in_plot]:plt.axvline(x=st_line_wav,color='black',ls='--',lw=1,alpha=0.6)  

                            #Plot frame  
                            if data_type=='spec1D':str_add=''
                            elif data_type=='spec2D':str_add=' (iord = '+str(iord)+')'                        
                            if plot_options['title']:plt.title(al_txt+title_name+' spectra in '+xt_str+' rest frame'+str_add)
                            y_title='Flux'
                            y_title = scaled_title(plot_options['sc_fact10'],y_title)
                            dx_range=x_range_loc[1]-x_range_loc[0]
                            dy_range=y_range_loc[1]-y_range_loc[0]
                            xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                            ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)
                            custom_axis(plt,position=plot_options['margins'],x_range=x_range_loc,y_range=y_range_loc,
                                        xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,
                                        xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                                        x_title='Wavelength in '+xt_str+' rest frame (A)',y_title=y_title,
                                        font_size=plot_options['font_size'],xfont_size=plot_options['font_size'],yfont_size=plot_options['font_size'])
            
                            if (data_type=='spec2D'):str_add='_iord'+str(iord)
                            else:str_add=''
                            plt.savefig(path_loc+data_type+str_add+'.'+plot_ext) 
                            plt.close() 
      
        return None    





    '''
    Sub-function to compare series of binned disk-integrated and intrinsic profiles
        - for aligned disk-integrated and intrinsic data
    '''
    def sub_plot_DI_Intr_binprof(plot_options,plot_ext):
        sc_fact=10**plot_options['sc_fact10']

        #Plot for each instrument
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_options['visits_to_plot'].keys())):   
            print('  > Instrument: '+inst)
                
            #Plot for each visit
            for vis in np.intersect1d(list(data_dic[inst].keys())+['binned'],plot_options['visits_to_plot'][inst]): 
                print('     - Visit : '+vis)
                path_loc = gen_dic['save_plot_dir']+'Compa_DI_intr_series/'+inst+'_'+vis+'/' 
                if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)   
   
                #Exposures to plot
                data_DIbin = np.load(gen_dic['save_data_dir']+'DIbin_data/'+inst+'_'+vis+'_'+plot_options['dim_plot_DI']+'_add.npz',allow_pickle=True)['data'].item()
                iexp_plot_DI = np.arange(data_DIbin['n_exp'],dtype=int) 
                nexp_plot_DI=len(iexp_plot_DI) 

                data_Intrbin = np.load(gen_dic['save_data_dir']+'Intrbin_data/'+inst+'_'+vis+'_'+plot_options['dim_plot_intr']+'_add.npz',allow_pickle=True)['data'].item()                    
                iexp_plot_intr = np.arange(data_Intrbin['n_exp'],dtype=int) 
                nexp_plot_intr=len(iexp_plot_intr) 

                #Visit colors  
                if 'DI' in plot_options['color_dic']:col_DI=np.repeat(plot_options['color_dic']['DI'],nexp_plot_DI)
                else:col_DI=np.array([cmap(0)]) if nexp_plot_DI==1 else cmap( np.arange(nexp_plot_DI)/(nexp_plot_DI-1.))
                if 'Intr' in plot_options['color_dic']:col_intr=np.repeat(plot_options['color_dic']['Intr'],nexp_plot_intr)
                else:col_intr=np.array([cmap(0)]) if nexp_plot_intr==1 else cmap( np.arange(nexp_plot_intr)/(nexp_plot_intr-1.))
    
                #------------------------------------------------------------------------------------------
                        
                #Plotting CCF profiles
                if plot_options['data_type']=='CCF':
                    plt.ioff()                    
                    plt.figure(figsize=plot_options['fig_size'])
                    
                    #Plotting each exposure
                    y_range_loc=sc_fact*np.array(plot_options['y_range']) if plot_options['y_range'] is not None else [1e100,-1e100] 
                    for isub,iexp in enumerate(iexp_plot_DI):             
                        data_exp=np.load(gen_dic['save_data_dir']+'DIbin_data/'+inst+'_'+vis+'_'+plot_options['dim_plot_DI']+str(iexp)+'.npz' ,allow_pickle=True)['data'].item()    
                        var_loc=sc_fact*data_exp['flux'][0]
                        plt.plot(data_exp['cen_bins'][0],var_loc ,color=col_DI[isub],linestyle='-',lw=plot_options['lw_plot'],rasterized=plot_options['rasterized'])    
                        if plot_options['y_range'] is None:y_range_loc = [0.99*min(np.nanmin(var_loc),y_range_loc[0]),1.01*max(np.nanmax(var_loc),y_range_loc[1])]
                    for isub,iexp in enumerate(iexp_plot_intr):                
                        data_exp=np.load(gen_dic['save_data_dir']+'Intrbin_data/'+inst+'_'+vis+'_'+plot_options['dim_plot_intr']+str(iexp)+'.npz' ,allow_pickle=True)['data'].item()    
                        var_loc=sc_fact*data_exp['flux'][0]
                        plt.plot(data_exp['cen_bins'][0],var_loc ,color=col_intr[isub],linestyle='-',lw=plot_options['lw_plot'],rasterized=plot_options['rasterized'])    
                        if plot_options['y_range'] is None:y_range_loc = [0.99*min(np.nanmin(var_loc),y_range_loc[0]),1.01*max(np.nanmax(var_loc),y_range_loc[1])]

                    #Central null velocity
                    plt.plot([0,0],y_range_loc,linestyle=':',lw=plot_options['lw_plot'],color='black')
    
                    #Reference level
                    x_range_loc=np.array(plot_options['x_range']) if plot_options['x_range'] is not None else np.array([data_exp['cen_bins'][0,0]-5.,data_exp['cen_bins'][0,-1]+5.])     
                    plt.plot(x_range_loc,[1.,1.],linestyle='-',lw=0.5,color='black')
               
                    #Plot frame
                    if plot_options['title']:plt.title('Binned disk-integrated and intrinsic CCFs',fontsize=plot_options['font_size'])
                    y_title='Flux'
                    y_title = scaled_title(plot_options['sc_fact10'],y_title)                   
                    dx_range=x_range_loc[1]-x_range_loc[0]
                    dy_range=y_range_loc[1]-y_range_loc[0]
                    xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                    ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range) 
                    custom_axis(plt,position=plot_options['margins'],x_range=x_range_loc,y_range=y_range_loc,
                                xmajor_int=xmajor_int,xminor_int=xminor_int,
                                ymajor_int=ymajor_int,yminor_int=yminor_int,
                                xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                                x_title='Velocity (km s$^{-1}$)',y_title=y_title,
                                font_size=plot_options['font_size'],xfont_size=plot_options['font_size'],yfont_size=plot_options['font_size'])                 						
                    plt.savefig(path_loc+'CCF.'+plot_ext) 
                    plt.close()    
                                
                #------------------------------------------------------------------------------------------
                        
                #Plotting spectral profiles
                elif 'spec' in plot_options['data_type']:
                        
                    #Retrieving data
                    cond_def_exp_DI={}
                    wav_exp_DI={}
                    sp_exp_DI={}
                    for iexp in iexp_plot_DI:
                        data_exp=np.load(gen_dic['save_data_dir']+'DIbin_data/'+inst+'_'+vis+'_'+plot_options['dim_plot_DI']+str(iexp)+'.npz' ,allow_pickle=True)['data'].item() 
                        cond_def_exp_DI[iexp] = data_exp['cond_def']
                        wav_exp_DI[iexp] = data_exp['cen_bins']
                        sp_exp_DI[iexp] = data_exp['flux']                                
                    cond_def_exp_intr={}
                    wav_exp_intr={}
                    sp_exp_intr={}
                    for iexp in iexp_plot_intr:
                        data_exp=np.load(gen_dic['save_data_dir']+'Intrbin_data/'+inst+'_'+vis+'_'+plot_options['dim_plot_intr']+str(iexp)+'.npz' ,allow_pickle=True)['data'].item()                             
                        cond_def_exp_intr[iexp] = data_exp['cond_def']
                        wav_exp_intr[iexp] = data_exp['cen_bins']
                        sp_exp_intr[iexp] = data_exp['flux']  
                        
                    #Plot each order of spectra for all exposures
                    for iord in range(data_dic[inst]['nord']):
                        plt.ioff()                    
                        plt.figure(figsize=plot_options['fig_size'])
                        y_range_loc=sc_fact*plot_options['y_range'] if plot_options['y_range'] is not None else [1e100,-1e100] 
    
                        for isub,iexp in enumerate(iexp_plot_DI):
                            cond_def_tab = cond_def_exp_DI[iexp][iord]
                            wav_tab = wav_exp_DI[iexp][iord][cond_def_tab]
                            var_loc=sc_fact*sp_exp_DI[iexp][iord][cond_def_tab]
                            plt.plot(wav_tab,var_loc,color=col_DI[iexp],linestyle='-',lw=plot_options['lw_plot'],rasterized=plot_options['rasterized'],zorder=1) 
                            if plot_options['y_range'] is None:y_range_loc = [0.99*min(np.nanmin(var_loc),y_range_loc[0]),1.01*max(np.nanmax(var_loc),y_range_loc[1])]                        
                        for isub,iexp in enumerate(iexp_plot_intr):
                            cond_def_tab = cond_def_exp_intr[iexp][iord]
                            wav_tab = wav_exp_intr[iexp][iord][cond_def_tab]
                            var_loc=sc_fact*sp_exp_intr[iexp][iord][cond_def_tab]
                            plt.plot(wav_tab,var_loc,color=col_intr[iexp],linestyle='-',lw=plot_options['lw_plot'],rasterized=plot_options['rasterized'],zorder=1) 
                            if plot_options['y_range'] is None:y_range_loc = [0.99*min(np.nanmin(var_loc),y_range_loc[0]),1.01*max(np.nanmax(var_loc),y_range_loc[1])]                        
                        
                        #Plot frame  
                        if isub==0: x_range_loc=np.array(plot_options['x_range']) if plot_options['x_range'] is not None else np.array([wav_tab[0]-0.02,wav_tab[-1]+0.02])  
                        if plot_options['data_type']=='spec1D':str_add=''
                        elif plot_options['data_type']=='spec2D':str_add=' (iord = '+str(iord)+')'                        
                        if plot_options['title']==True:plt.title('Binned disk-integrated and intrinsic '+str_add)
                        y_title='Flux'
                        y_title = scaled_title(plot_options['sc_fact10'],y_title) 
                        dx_range=x_range_loc[1]-x_range_loc[0]
                        dy_range=y_range_loc[1]-y_range_loc[0]
                        xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                        ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)
                        custom_axis(plt,position=plot_options['margins'],x_range=x_range_loc,y_range=y_range_loc,
                                    xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,
                                    xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                                    x_title='Wavelength (A)',y_title=y_title,
                                    font_size=plot_options['font_size'],xfont_size=plot_options['font_size'],yfont_size=plot_options['font_size'])
                        str_add=''
                        if (plot_options['data_type']=='spec2D'):str_add='_iord'+str(iord)
                        plt.savefig(path_loc+'spectra'+str_add+'.'+plot_ext) 
                        plt.close() 
      
        return None 


    '''
    Sub-function to plot chi2 of fitted property series
    '''
    def sub_plot_chi2_prop(plot_options,plot_ext):
        path_loc = gen_dic['save_plot_dir']+'Intr_prop/' 
        if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)                                        
        plt.ioff()                    
        plt.figure(figsize=plot_options['fig_size'])

        #Ranges
        x_min=1e100
        x_max=-1e100
        y_max=-1e100  

        #Upload fit results
        if plot_options['IntrProp_path'] is None:stop('Define path to fit')
        data_upload =dataload_npz(plot_options['IntrProp_path']+plot_options['prop']+'/Fit_results')

        #Plot independently each visit
        for inst in np.intersect1d(list(data_upload['prop_fit'].keys()),list(plot_options['visits_to_plot'].keys())): 
            if inst not in plot_options['color_dic']:plot_options['color_dic'][inst]={}
            for vis in np.intersect1d(list(data_upload['prop_fit'][inst].keys()),plot_options['visits_to_plot'][inst]):    
                if vis not in plot_options['color_dic'][inst]:plot_options['color_dic'][inst][vis]='dodgerblue'
                
                #Chi2 values
                chi2_vis= ( (data_upload['prop_fit'][inst][vis] - data_upload['prop_mod'][inst][vis])/data_upload['err_prop_fit'][inst][vis])**2.
              
                #X table
                x_plot=data_upload['coord_mod'][inst][vis]
               
                #Outliers
                if (plot_options['chi2_thresh'] is not None):
                    cond_outliers=(chi2_vis>plot_options['chi2_thresh'])
                    if (True in cond_outliers):print('     Outliers in '+inst+' : '+vis+' at phase =',x_plot[cond_outliers])
        
                #Update min/max for the plot
                x_min=min(x_min,min(x_plot))
                x_max=max(x_max,max(x_plot))  
                y_max=max(y_max,max(chi2_vis))  
                                        
                #Plot chi2 values        
                plt.plot(x_plot,chi2_vis,marker='o',linestyle='',markersize=plot_options['markersize'],markerfacecolor=plot_options['color_dic'][inst][vis],markeredgecolor=plot_options['color_dic'][inst][vis])
            
        #---------------------------------------------------------------------------------------- 

        #Plot frame 
        x_range_loc=autom_range_ext(plot_options['x_range'],x_min,x_max)
        y_range_loc=autom_range_ext(plot_options['y_range'],0.,y_max)
        if plot_options['title']:plt.title('$\Chi^2$ per exposure',fontsize=plot_options['font_size'])     
        xmajor_int,xminor_int,xmajor_form=autom_tick_prop(x_range_loc[1]-x_range_loc[0])
        ymajor_int,yminor_int,ymajor_form=autom_tick_prop(y_range_loc[1]-y_range_loc[0])                          
        custom_axis(plt,position=plot_options['margins'],x_range=x_range_loc,y_range=y_range_loc,
    		        xmajor_int=xmajor_int,xminor_int=xminor_int,xmajor_form=xmajor_form,
                    ymajor_form=ymajor_form,ymajor_int=ymajor_int,yminor_int=yminor_int,     
        		    x_title=data_upload['coord_line'],y_title='$\chi^2$',
                    font_size=plot_options['font_size'],xfont_size=plot_options['font_size'],yfont_size=plot_options['font_size'])
        plt.savefig(path_loc+'Chi2_per_exp.'+plot_ext)                        
        plt.close() 






    '''
    Generic function for CCF properties plots
    '''
    def sub_func_bin(bin_val_loc,dic_val,x_min,x_max,y_min,y_max,plot_options):
        if 'dbin' in bin_val_loc:
            nbin=int((bin_val_loc['bin_max']-bin_val_loc['bin_min'])/bin_val_loc['dbin'])
            dbin_eff=(bin_val_loc['bin_max']-bin_val_loc['bin_min'])/nbin
            x_bd_low_eff=bin_val_loc['bin_min']+dbin_eff*np.arange(nbin)
            x_bd_high_eff=x_bd_low_eff+dbin_eff 
        else:
            x_bd_low_eff = bin_val_loc['x_bd_low'] 
            x_bd_high_eff = bin_val_loc['x_bd_high'] 
        if (not plot_options['plot_err']) and (not plot_options['plot_HDI']):
            _,_,mid_x_bin,_,val_bin,_ = resample_func(x_bd_low_eff,x_bd_high_eff,dic_val['st_x_all'],dic_val['end_x_all'],dic_val['val_all'],None,remove_empty=True,dim_bin=0,cond_olap=1e-14)    
            plt.plot(mid_x_bin,val_bin,color=bin_val_loc['color'],rasterized=plot_options['rasterized'],markeredgecolor=bin_val_loc['color'],markerfacecolor='None',marker='d',markersize=plot_options['markersize']+2,linestyle='',zorder=10,alpha=bin_val_loc['alpha_bin']) 
        else:
            if plot_options['plot_err']:
                _,_,mid_x_bin,_,val_bin,eval_bin = resample_func(x_bd_low_eff,x_bd_high_eff,dic_val['st_x_all'],dic_val['end_x_all'],dic_val['val_all'],np.mean(dic_val['eval_all'],axis=0),remove_empty=True,dim_bin=0)    
                plt.errorbar(mid_x_bin,val_bin,yerr=eval_bin,color=bin_val_loc['color'],rasterized=plot_options['rasterized'],markeredgecolor=bin_val_loc['color'],markerfacecolor='None',marker='d',markersize=plot_options['markersize']+2,linestyle='',zorder=10,alpha=bin_val_loc['alpha_bin']) 
            if plot_options['plot_HDI']:
                _,_,mid_x_bin,_,val_bin,eval_bin = resample_func(x_bd_low_eff,x_bd_high_eff,dic_val['st_x_all'],dic_val['end_x_all'],dic_val['val_all'],np.mean(dic_val['HDI_all'],axis=0),remove_empty=True,dim_bin=0)    
                plt.errorbar(mid_x_bin,val_bin,yerr=eval_bin,color='black',rasterized=plot_options['rasterized'],markeredgecolor='black',markerfacecolor='None',marker='d',markersize=plot_options['markersize']+2,linestyle='',zorder=10,alpha=plot_options['alpha_symb']) 

        
        #Update min/max for the plot
        x_min=min(x_min,min(mid_x_bin))
        x_max=max(x_max,max(mid_x_bin))
        if plot_options['plot_err']:
            y_min=min(y_min,np.min((val_bin-eval_bin)))
            y_max=max(y_max,np.max((val_bin+eval_bin)))
        else:
            y_min=min(y_min,np.min(val_bin))
            y_max=max(y_max,np.max(val_bin)) 

        return x_min,x_max,y_min,y_max

    def sub_plot_CCF_prop(prop_mode,plot_options,data_mode): 
        path_loc = gen_dic['save_plot_dir']+data_mode+'_prop/'  
        if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)

        plt.ioff()        
        plt.figure(figsize=plot_options['fig_size'])            
        
        #Fit sub-function     
        def fit_pol_sin(param,x_unused,args=None):
            corr_prop = {}
            
            #Constant correction level
            fit_tab=np.repeat(param['a0'].value,len(x_unused))

            #Variable contributions
            for iprop,prop_fit in enumerate(args['prop_fit']): 
                x_prop = args['var_fit'][prop_fit]
                apply_corr =False

                #Sinusoidal variation 
                if prop_fit in fixed_args['sin_prop']:
                    corr_prop['sin'] = [ param[prop_fit+corr_val].value for corr_val in ['_amp', '_off', '_per'] ]
                    apply_corr = True
                
                #Polynomial variation
                if (prop_fit in fixed_args['pol_prop']) and (args['deg_pol'][prop_fit]>0):
                    corr_prop['pol']= [param[prop_fit+'_c'+str(ideg)].value for ideg in  range(1,args['deg_pol'][prop_fit]+1)   ]   
                    apply_corr = True

                #Variations are defined
                if apply_corr:
                    if prop_mode in ['rv','rv_res']:fit_tab = detrend_prof_gen(  corr_prop, x_prop, fit_tab , 'add')  
                    else:fit_tab = detrend_prof_gen(  corr_prop, x_prop, fit_tab , 'modul')  
            
            return fit_tab     
     
        #Horizontal range
        x_min=1e100
        x_max=-1e100
       
        #Vertical range
        y_min=1e100
        y_max=-1e100
        
        #Symbols for in/out transit data
        mark_tr='s'
        mark_otr='s'
        if plot_options['use_diff_symb']:mark_tr='o'

        #Store values over all instruments
        dic_all = {}
        for key in ['x_all','st_x_all','end_x_all','val_all']:dic_all[key] = np.empty(0,dtype=float)
        dic_all['eval_all'] = np.empty([2,0],dtype=float)
        dic_all['HDI_all'] = np.empty([2,0],dtype=float)

        #Uploading best-fit data
        if data_mode=='Intr':    
            data_fit_prop=None  
            data_fit_prof=None
            if prop_mode in ['rv','rv_res','FWHM','ctrst','true_FWHM','true_FWHM','a_damp']:
                if plot_options['IntrProp_path'] is not None:
                    if prop_mode=='rv_res':prop_mode_get = 'rv'
                    else:prop_mode_get = prop_mode
                    data_fit_prop = dataload_npz(plot_options['IntrProp_path']+prop_mode_get+'/Fit_results')    
                if plot_options['IntrProf_path'] is not None:
                    data_fit_prof = dataload_npz(plot_options['IntrProf_path']+'Fit_results')    
            
        #Plot for each instrument
        i_visit=-1
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_options['visits_to_plot'].keys())): 
            if inst not in plot_options['color_dic']:plot_options['color_dic'][inst]={}
            if inst not in plot_options['idx_noplot']:plot_options['idx_noplot'][inst]={}
            print('    ',inst)
            dic_inst = {}
            for key in ['x_all','st_x_all','end_x_all','val_all']:dic_inst[key] = np.empty(0,dtype=float)
            dic_inst['eval_all'] = np.empty([2,0],dtype=float)
            dic_inst['HDI_all'] = np.empty([2,0],dtype=float)
                   
            #Plot for each visit
            if data_mode=='DI':vis_list = np.intersect1d(list(data_dic['DI'][inst].keys())+['binned'],plot_options['visits_to_plot'][inst])
            elif data_mode=='Intr':vis_list = plot_options['visits_to_plot'][inst]
            for vis in vis_list: 
        
            
                # if (gen_dic['star_name']=='WASP76'):   #ANTARESS I
                #     print('DELETE')
                #     if vis=='20180902':
                #         mark_tr = 'o'
                #         mark_otr = 'o'
                #     if vis=='20181030':
                #         mark_tr = 's'
                #         mark_otr = 's'
                
                #Identifying original or binned data
                if data_mode=='DI':
                    orig_vis = vis
                    if 'bin' in vis:data_type = data_mode+'bin'
                    else:data_type = data_mode+'orig'
                    cond_vis = True
                elif data_mode=='Intr':
                    orig_vis = vis.split('_bin')[0]
                    if 'bin' in vis:data_type = data_mode+'bin'
                    else:data_type = data_mode+'orig'
                    cond_vis = orig_vis in list(data_dic['Res'][inst].keys())

                #Reference planet for the visit                    
                pl_ref = plot_options['pl_ref'][inst][orig_vis]

                #High-resolution RV models from nominal system properties
                #    - achromatic
                if (data_mode=='Intr') and (prop_mode in ['rv','rv_res']) and plot_options['theo_HR_nom'] : 
                    params = deepcopy(system_param['star'])
                    params.update({'rv':0.,'cont':1.})                 
                    theo_HR_prop_plocc = calc_occ_plot(None,deepcopy(theo_dic),inst,vis,{},params,{},deepcopy(system_param))
                    if plot_options['prop_'+data_mode+'_absc']=='phase':xvar_HR=deepcopy(theo_HR_prop_plocc[pl_ref]['phase'])  
                    elif plot_options['prop_'+data_mode+'_absc'] in ['mu','lat','lon','x_st','y_st','xp_abs','r_proj']:xvar_HR=deepcopy(theo_HR_prop_plocc[pl_ref][plot_options['prop_'+data_mode+'_absc']])  
                    elif plot_options['prop_'+data_mode+'_absc']=='y_st2':xvar_HR=theo_HR_prop_plocc[pl_ref]['y_st']**2.  
                    elif plot_options['prop_'+data_mode+'_absc']=='abs_y_st':xvar_HR=np.abs(theo_HR_prop_plocc[pl_ref]['y_st'])
                    wsort=theo_HR_prop_plocc[pl_ref]['phase'].argsort()
       
                    #Nominal high-resolution model and associated components     
                    x_theo_HR_nom = xvar_HR[wsort]
                    y_theo_HR_nom = theo_HR_prop_plocc[pl_ref]['rv'][wsort]
                    if prop_mode=='rv':plt.plot(x_theo_HR_nom,y_theo_HR_nom,color='black',linestyle='-',lw=plot_options['lw_plot'],zorder=-1)   
                    
                    #Solid-body model
                    if (len(plot_options['contrib_theo_HR'])>0) or ((prop_mode=='rv_res') and (plot_options['mod_compos'] == 'SB')):
                        params['alpha_rot'] = 0.
                        params['beta_rot'] = 0.
                        rv_sb_theo_HR_nom = calc_occ_plot(None,deepcopy(theo_dic),inst,vis,{},params,{},deepcopy(system_param))[pl_ref]['Rot_RV'][wsort]
                      

                    # #Save/replot manually 
                    # # save_path = '/Travaux/ANTARESS/Ongoing/Fit/SB_aligned.dat'
                    # # save_path = '/Travaux/ANTARESS/Ongoing/RVloc_model_WASP121_Vincent.dat'
                    # # save_path = '/Users/bourrier/Travaux/ANTARESS/Ongoing/GJ436_b_Plots/Intr_prop/RRM_RV_model.dat' 
                    # # save_path = '/Users/bourrier/Travaux/ANTARESS/Ongoing/TIC257527578b_Plots/Intr_prop/RRM_RV_model_DR0.2_lambda0.dat' 
                    # save_path = '/Users/bourrier/Travaux/ANTARESS/Ongoing/TIC257527578b_Plots/Intr_prop/RRM_RV_model_DR0.2_lambda90.dat' 
                    # # np.savetxt(save_path, np.column_stack((xvar_HR[wsort],theo_HR_prop_plocc[pl_ref]['rv'][wsort])),fmt=('%15.10f','%15.10f') )
                    # xvar_al,RV_stsurf_al=np.loadtxt(save_path).T
                    # plt.plot(xvar_al,RV_stsurf_al,color='black',linestyle=':',lw=plot_options['lw_plot'],zorder=-1) 

                    x_min=min(x_min,np.nanmin(xvar_HR))
                    x_max=max(x_max,np.nanmax(xvar_HR))
                    y_min=min(y_min,np.nanmin(theo_HR_prop_plocc[pl_ref]['rv']))
                    y_max=max(y_max,np.nanmax(theo_HR_prop_plocc[pl_ref]['rv']))

                    #Surface RV model      
                    if (prop_mode=='rv'):

                        #Contributions to the model    
                        #    - when DR is activated, 'Rot_RV' accounts for it
                        if len(plot_options['contrib_theo_HR'])>0:
                            if 'SB' in plot_options['contrib_theo_HR']:
                                plt.plot(xvar_HR[wsort],rv_sb_theo_HR_nom,color='orange',linestyle='--',lw=plot_options['lw_plot'],zorder=-1)  
                            if 'CB' in plot_options['contrib_theo_HR']:
                                plt.plot(xvar_HR[wsort],rv_sb_theo_HR_nom+theo_HR_prop_plocc[pl_ref]['CB_RV'][wsort],color='dodgerblue',linestyle='--',lw=plot_options['lw_plot'],zorder=-1)   
                            if 'DR' in plot_options['contrib_theo_HR']:
                                plt.plot(xvar_HR[wsort],theo_HR_prop_plocc[pl_ref]['Rot_RV'][wsort],color='magenta',linestyle='--',lw=plot_options['lw_plot'],zorder=-1)   
    
                        #----------------------------------------------------------------------
                
                        #Calculate samples of the PDF for the best-fit model
                        #    - calculated over the HR phase table 
                        #    - nominal system properties must be set to the best-fit for the HR model to correspond to these samples
                        if (plot_options['calc_envMCMC_theo_HR_nom']!='') or (plot_options['calc_sampMCMC_theo_HR_nom']!=''):
                            
                            #Use multi-threading to compute distributions
                            nthreads=1# 10
        
                            #Multi-threading
                            pool_proc = Pool(processes=nthreads) if (nthreads>1) else None  

                            #Generic functions
                            #    - fixed parameters do not need to be input as they are defined at the definition of the function
                            def sub_calc_plocc_spot_prop_threads(par_subsample,pl_loc):
                                nsamp=len(par_subsample[0])
                                RV_stsurf_HR_thread=np.empty([nsamp,theo_HR_prop_plocc['nph_HR']])
                                
                                #For each sample, calculate and overwrite surface rv alone
                                coord_pl_in_samp = deepcopy(theo_HR_prop_plocc)
                                theo_dic_samp = deepcopy(theo_dic)
                                theo_dic_samp['d_oversamp'] = []
                                for isamp in range(nsamp):
                                    surf_prop_dic,_,_ = sub_calc_plocc_spot_prop(['achrom'],{},['rv'],[pl_loc],system_param,theo_dic_samp,data_dic['DI']['system_prop'],par_subsample[0][isamp],coord_pl_in_samp,range(theo_HR_prop_plocc['nph_HR']))        
                                    RV_stsurf_HR_thread[isamp,:] =surf_prop_dic['achrom'][pl_loc]['rv'][0,:]                                
                                return RV_stsurf_HR_thread
                            
                            def sub_calc_plocc_spot_prop_par(pool_proc,func_input,nthreads,n_elem,y_inputs,common_args):     
                                ind_chunk_list=init_parallel_func(nthreads,n_elem)
                                chunked_args=[(y_inputs[0][ind_chunk[0]:ind_chunk[1]],)+common_args for ind_chunk in ind_chunk_list]	
                                all_results=tuple(tab for tab in pool_proc.map(func_input,chunked_args))			
                                #Outputs: dictionary with keys the planet for which transit is modeled, and values array with dimensions nsamp_thread x n_HR to be concatenated along the first axis   					
                                y_output=np.concatenate(tuple(all_results[i] for i in range(nthreads)),axis=0)
                                return y_output
                
                            #-----------------------------------------------------
                          
                            #Calculate all model light curves within +-1 sigma range of the parameters and retrieve the envelope
                            if (plot_options['calc_envMCMC_theo_HR_nom']!=''):
                                print('     Retrieve 1-sigma sample')
                                                 
                                #Load samples
                                npzfile = np.load(plot_options['calc_envMCMC_theo_HR_nom'],allow_pickle=True)
                                	 
                                par_sample_sig1=npzfile['par_sample_sig1'][0]  
                                if nthreads>1:                    
                                    common_args=()
                                    chunkable_args=[par_sample_sig1]
                                    RV_stsurf_HR_sig1_all=sub_calc_plocc_spot_prop_par(pool_proc,sub_calc_plocc_spot_prop_threads,nthreads,len(par_sample_sig1),chunkable_args,common_args)                           
                                else:        
                                    RV_stsurf_HR_sig1_all=sub_calc_plocc_spot_prop_threads([par_sample_sig1],pl_ref)
                                RV_stsurf_HR_sig1=np.vstack((np.repeat(1e100,theo_HR_prop_plocc['nph_HR']),np.repeat(-1e100,theo_HR_prop_plocc['nph_HR'])))   
                                for isamp in range(len(par_sample_sig1)):
                                    RV_stsurf_HR_sig1[0,:]=np.minimum(RV_stsurf_HR_sig1[0,:],RV_stsurf_HR_sig1_all[isamp])
                                    RV_stsurf_HR_sig1[1,:]=np.maximum(RV_stsurf_HR_sig1[1,:],RV_stsurf_HR_sig1_all[isamp])
        
                                #Plot 1-sigma envelope
                                plt.gca().fill_between(xvar_HR[wsort], RV_stsurf_HR_sig1[0,wsort], RV_stsurf_HR_sig1[1,wsort], color="lightgrey", alpha=0.4,zorder=-5)
                                plt.plot(xvar_HR[wsort], RV_stsurf_HR_sig1[0,wsort], color="darkgrey", linestyle='-',lw=1,zorder=-5)
                                plt.plot(xvar_HR[wsort], RV_stsurf_HR_sig1[1,wsort], color="darkgrey", linestyle='-',lw=1, zorder=-5)
                                                
                            #-----------------------------------------------------
                        
                            #Calculate all models from the random sample retrieved in MCMC fit routine
                            if (plot_options['calc_sampMCMC_theo_HR_nom']!=''):
                                print('     Retrieve random sample')

                                #Load samples
                                npzfile = np.load(plot_options['calc_sampMCMC_theo_HR_nom'],allow_pickle=True)                                
                                
                                par_sample=npzfile['par_sample'][0]      
                                if nthreads>1:
                                    common_args=()
                                    chunkable_args=[par_sample]
                                    RV_stsurf_HR_sample=sub_calc_plocc_spot_prop_par(pool_proc,sub_calc_plocc_spot_prop_threads,nthreads,len(par_sample),chunkable_args,common_args)                           
                                else:
                                    RV_stsurf_HR_sample=sub_calc_plocc_spot_prop_threads([par_sample],pl_ref)
                               
                                #Plot random sample
                                for isamp in range(len(RV_stsurf_HR_sample)):
                                    plt.plot(xvar_HR[wsort],RV_stsurf_HR_sample[isamp][wsort],color='darkgrey',linestyle='-',lw=0.1,alpha=0.3,zorder=-4)

                    #Residuals from surface RVs      
                    elif (prop_mode=='rv_res'):

                        #Contributions to the model    
                        #    - when DR is activated, 'Rot_RV' accounts for it                    
                        if len(plot_options['contrib_theo_HR'])>0:
                            if 'CB' in plot_options['contrib_theo_HR']:
                                plt.plot(xvar_HR[wsort],1e3*theo_HR_prop_plocc[pl_ref]['CB_RV'][wsort],color='dodgerblue',linestyle='--',lw=plot_options['lw_plot'],zorder=-1)   
                            if 'DR' in plot_options['contrib_theo_HR']:
                                plt.plot(xvar_HR[wsort],1e3*(theo_HR_prop_plocc[pl_ref]['rv'][wsort] - rv_sb_theo_HR_nom - theo_HR_prop_plocc[pl_ref]['CB_RV'][wsort]) ,color='magenta',linestyle='--',lw=plot_options['lw_plot'],zorder=-1)   
                            if 'CB_DR' in plot_options['contrib_theo_HR']:
                                plt.plot(xvar_HR[wsort],1e3*(theo_HR_prop_plocc[pl_ref]['rv'][wsort] - rv_sb_theo_HR_nom),color='black',linestyle='--',lw=plot_options['lw_plot'],zorder=-1)   
    
        
                ################################################################################################                
                if cond_vis:                
                    print('      '+vis)
                    i_visit+=1
                    data_vis = data_dic[inst][vis]
                    prof_fit_vis = None
                    data_bin = None
                    if vis not in plot_options['color_dic'][inst]:plot_options['color_dic'][inst][vis]='dodgerblue'                 
                    if plot_options['color_dic'][inst][vis]=='jet':col_loc='dodgerblue'  
                    else:col_loc = plot_options['color_dic'][inst][vis]
                    if data_mode=='DI':
                        if vis=='binned':
                            if plot_options['prop_'+data_mode+'_absc']!='phase':stop('Use correct binning dimension')
                            data_bin = dataload_npz(gen_dic['save_data_dir']+'DIbin_data/'+inst+'_'+vis+'_phase')
                            if gen_dic['fit_DIbin'] or gen_dic['fit_DIbinmultivis']:prof_fit_vis=dataload_npz(gen_dic['save_data_dir']+'DIbin_prop/'+inst+'_'+vis)
                        else:
                            coord_vis = coord_dic[inst][vis][pl_ref]
                            data_prop_vis = data_prop[inst][vis]
                            # if gen_dic['fit_DI'] or gen_dic['fit_DI_1D']:prof_fit_vis=np.load(gen_dic['save_data_dir']+'DIorig_prop/'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item()
                            prof_fit_vis=dataload_npz(gen_dic['save_data_dir']+'DIorig_prop/'+inst+'_'+vis)
                        transit_prop_nom = dataload_npz(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis)['achrom'][pl_ref]

                    elif data_mode=='Intr':
                        if data_type=='Introrig':
                            coord_vis = coord_dic[inst][vis][pl_ref]
                            if plot_options['plot_data']:prof_fit_vis=dataload_npz(gen_dic['save_data_dir']+'Introrig_prop/'+inst+'_'+vis)
                            transit_prop_nom = (np.load(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item())['achrom'][pl_ref]                                                      
                        elif data_type=='Intrbin': 
                            if plot_options['plot_data']:
                                prof_fit_vis=np.load(gen_dic['save_data_dir']+'Intrbin_prop/'+inst+'_'+orig_vis+'.npz',allow_pickle=True)['data'].item()
                                data_bin = np.load(gen_dic['save_data_dir']+'Intrbin_data/'+inst+'_'+orig_vis+'_'+plot_options['dim_plot']+'_add.npz',allow_pickle=True)['data'].item()
                            transit_prop_nom = data_bin['plocc_prop']['achrom'][pl_ref]      

                        #Models from fits 
                        if (data_fit_prop is not None) or (data_fit_prof is not None): 
                            
                            #High-resolution models
                            if plot_options['theo_HR_prop'] or plot_options['theo_HR_prof']:  
                                def sub_plot_HR(mode_loc,input_dic,col_loc):
                                    if mode_loc =='from_prop':data_fit_loc=data_fit_prop
                                    if mode_loc =='from_prof':data_fit_loc=data_fit_prof
                                    
                                    #Coordinates and properties associated with planet-occulted regions
                                    #    - for the purpose of the plot properties are calculated in low-precision mode, as they cannot be easily extracted from the model line profiles
                                    par_list_HR = ['mu','xp_abs','r_proj','y_st','lat']
                                    theo_dic_in = deepcopy(theo_dic)
                                    theo_dic_in['precision'] = 'low'
                                    if (inst in data_fit_loc['coeff_line_dic']) and (vis in data_fit_loc['coeff_line_dic'][inst]):data_fit_loc['coeff_line'] =  data_fit_loc['coeff_line_dic'][inst][vis]
                                    if prop_mode in ['rv','rv_res']:
                                        par_list_HR+=['rv','CB_RV']
                                        if (inst in data_fit_loc['linevar_par']) and (vis in data_fit_loc['linevar_par'][inst]) and ('rv_line' in data_fit_loc['linevar_par'][inst][vis]):par_list_HR+=['rv_line']
                                    elif prop_mode in ['ctrst','FWHM','a_damp']: 
                                        if plot_options['theo_HR_prop']:par_list_HR+=[prop_mode]
                                        elif plot_options['theo_HR_prof']:par_list_HR+=['ctrst','FWHM']
                                    
                                    theo_HR_prop_loc = calc_occ_plot(data_bin,theo_dic_in,inst,vis,data_fit_loc['genpar_instvis'],data_fit_loc['p_final'],data_fit_loc,deepcopy(system_param),par_list = par_list_HR)[pl_ref]
                                    if plot_options['prop_'+data_mode+'_absc']=='phase':xvar_HR_loc=deepcopy(theo_HR_prop_loc['phase'])  
                                    elif plot_options['prop_'+data_mode+'_absc'] in ['mu','lat','lon','x_st','y_st','xp_abs','r_proj']:xvar_HR_loc=deepcopy(theo_HR_prop_loc[plot_options['prop_'+data_mode+'_absc']])  
                                    elif plot_options['prop_'+data_mode+'_absc']=='y_st2':xvar_HR_loc=theo_HR_prop_loc['y_st']**2.  
                                    elif plot_options['prop_'+data_mode+'_absc']=='abs_y_st':xvar_HR_loc=np.abs(theo_HR_prop_loc['y_st'])
                                    wdefHR = np_where1D(~np.isnan(xvar_HR_loc))
                                    wsort_sub=xvar_HR_loc[wdefHR].argsort() 
                                    wsort = wdefHR[wsort_sub]   
                                    xvar_HR_loc = xvar_HR_loc[wsort]

                                    #Property
                                    if prop_mode in ['rv','rv_res']:
                                        yvar_HR_loc = theo_HR_prop_loc['rv'][wsort]
                                        
                                        #Solid-body model
                                        if (len(plot_options['contrib_theo_HR'])>0) or ((prop_mode=='rv_res') and (plot_options['mod_compos'] == 'SB')):
                                            params = deepcopy(data_fit_loc['p_final'])
                                            params['alpha_rot'] = 0.
                                            params['beta_rot'] = 0.
                                            rv_sb_theo_HR = calc_occ_plot(data_bin,theo_dic_in,inst,vis,data_fit_loc['genpar_instvis'],params,data_fit_loc,deepcopy(system_param),par_list = par_list_HR)[pl_ref]['Rot_RV'][wsort]

                                        #Model components
                                        if len(plot_options['contrib_theo_HR'])>0:
                                            if (prop_mode=='rv'):
                                                if 'SB' in plot_options['contrib_theo_HR']:
                                                    plt.plot(xvar_HR_loc,rv_sb_theo_HR,color='orange',linestyle='--',lw=plot_options['lw_plot'],zorder=-1)  
                                                if 'CB' in plot_options['contrib_theo_HR']:
                                                    plt.plot(xvar_HR_loc,theo_HR_prop_loc['CB_RV'][wsort],color='dodgerblue',linestyle='--',lw=plot_options['lw_plot'],zorder=-1)                               
                                                if 'DR' in plot_options['contrib_theo_HR']:                                           
                                                    plt.plot(xvar_HR_loc,(theo_HR_prop_loc['Rot_RV'][wsort] - rv_sb_theo_HR),color='magenta',linestyle='--',lw=plot_options['lw_plot'],zorder=-1)   
                                            
                                            elif (prop_mode=='rv_res'):
                                                if 'CB' in plot_options['contrib_theo_HR']:
                                                    plt.plot(xvar_HR_loc,1e3*theo_HR_prop_loc['CB_RV'][wsort],color='dodgerblue',linestyle='--',lw=plot_options['lw_plot'],zorder=-1)   
                                                if 'DR' in plot_options['contrib_theo_HR']: 
                                                    plt.plot(xvar_HR_loc,1e3*(theo_HR_prop_loc['Rot_RV'][wsort] - rv_sb_theo_HR),color='magenta',linestyle='--',lw=plot_options['lw_plot'],zorder=-1)   
                                                if 'CB_DR' in plot_options['contrib_theo_HR']:
                                                    plt.plot(xvar_HR_loc,1e3*(theo_HR_prop_loc['CB_RV'][wsort] + theo_HR_prop_loc['Rot_RV'][wsort] - rv_sb_theo_HR),color='black',linestyle='--',lw=plot_options['lw_plot'],zorder=-1)   
                        
                                        #Replace rv model with chosen component to calculate residuals
                                        if (prop_mode=='rv_res'):        
                                            if plot_options['mod_compos'] == 'SB':yvar_HR_loc = rv_sb_theo_HR  

                                    else:
                                    
                                        #FWHM and contrast of single property measurements 
                                        if (mode_loc =='from_prop'):
                                            yvar_HR_loc = theo_HR_prop_loc[prop_mode][wsort] 
                                      
                                        if mode_loc =='from_prof':
                                            
                                            #FWHM and contrast of unconvolved intrinsic stellar profiles / of single property measurements 
                                            raw_FWHM_mod_HR = theo_HR_prop_loc['FWHM'][wsort] 
                                            raw_ctrst_mod_HR = theo_HR_prop_loc['ctrst'][wsort]

                                            #FWHM and contrast equivalent to observed profiles
                                            #    - the derivation of FWHM and contrast from a HR profile is only relevant when observed individual profiles have been fitted with a gaussian model estimating their contrast and FWHM, 
                                            #      if those measured properties are not the ones controlling the joint profile model, we must estimate them on the HR profile
                                            #      if the same model profile was however used for the fit to individual and joint profiles, the properties can be directlyy compared
                                            if (data_fit_loc['func_prof_name'][inst]=='gauss') or ((prop_mode in ['ctrst','FWHM']) and ~plot_options['inst_conv']):
                                                if plot_options['inst_conv']:ctrst_mod_HR,FWHM_mod_HR = gauss_intr_prop(raw_ctrst_mod_HR,raw_FWHM_mod_HR,calc_FWHM_inst(inst,c_light)) 
                                                else:ctrst_mod_HR,FWHM_mod_HR = raw_ctrst_mod_HR,raw_FWHM_mod_HR
                                            else:
                                                if plot_options['inst_conv']:input_dic['FWHM_inst'] = calc_FWHM_inst(inst,c_light)
                                                else:input_dic['FWHM_inst'] = None
                                                if data_fit_loc['func_prof_name'][inst]=='dgauss':
                                                    ctrst_mod_HR = np.zeros(len(wsort),dtype=float)  
                                                    FWHM_mod_HR = np.zeros(len(wsort),dtype=float)
                                                    for isub_HR,(raw_ctrst_loc,raw_FWHM_loc) in enumerate(zip(raw_ctrst_mod_HR,raw_FWHM_mod_HR)):
                                                        input_dic.update({'ctrst':raw_ctrst_loc,  'FWHM':raw_FWHM_loc ,'fit_func_gen':dgauss})
                                                        ctrst_mod_HR[isub_HR],FWHM_mod_HR[isub_HR] = cust_mod_true_prop(input_dic,input_dic['vel'],input_dic)[0:2]                            
                                                elif data_fit_loc['func_prof_name'][inst]=='voigt':
                                                    ctrst_mod_HR = np.zeros(len(wsort),dtype=float)  
                                                    FWHM_mod_HR = np.zeros(len(wsort),dtype=float)
                                                    for isub_HR,(raw_ctrst_loc,raw_FWHM_loc) in enumerate(zip(raw_ctrst_mod_HR,raw_FWHM_mod_HR)):
                                                        input_dic.update({'ctrst':raw_ctrst_loc,  'FWHM':raw_FWHM_loc,'fit_func_gen':voigt,'slope':0.})
                                                        ctrst_mod_HR[isub_HR],FWHM_mod_HR[isub_HR] = cust_mod_true_prop(input_dic,input_dic['vel'],input_dic)[0:2]                            
                                            if (prop_mode in ['true_FWHM','FWHM','FWHM_voigt']):yvar_HR_loc = FWHM_mod_HR   
                                            if (prop_mode in ['true_ctrst','ctrst']):yvar_HR_loc = ctrst_mod_HR
                  
                                    #Plot
                                    if prop_mode != 'rv_res':plt.plot(xvar_HR_loc,yvar_HR_loc,color=col_loc,linestyle='--',lw=1,zorder=-1) 
                                    
                                    return xvar_HR_loc,yvar_HR_loc
                                
                        #-------------------------------------------------------                             
                        #Model from property fit
                        if (data_fit_prop is not None):
                        
                            #Data-equivalent model
                            if plot_options['theo_obs_prop']:
                                if (prop_mode in ['rv','rv_res'] and plot_options['prop_'+data_mode+'_absc']!='phase') or (data_fit_prop['coord_line']!=plot_options['prop_'+data_mode+'_absc']):stop('Plot and fit coordinates must match')
                                plt.plot(data_fit_prop['coord_mod'][inst][vis],data_fit_prop['prop_mod'][inst][vis],color='green',linestyle='',lw=1,marker='s',markersize=plot_options['markersize'],zorder=-1)

                            #High-resolution model
                            if plot_options['theo_HR_prop']:
                                xvar_HR_loc,yvar_HR_loc = sub_plot_HR('from_prop',None,'orange')

                        #------------------------------------------------------- 
                        #Model from profile fit
                        if (data_fit_prof is not None):                                
                            input_dic = {}
                            if ((data_fit_prof['func_prof_name'][inst]=='dgauss') & (prop_mode in ['true_FWHM','true_ctrst'])) | ((data_fit_prof['func_prof_name'][inst]=='voigt') & (prop_mode in ['FWHM_voigt','ctrst'])):                        
                                if 'best_mod_tab' not in plot_options:
                                    dx =return_pix_size()[inst]/4.
                                    min_x = -100.
                                    max_x = 100.                             
                                else:
                                    dx =plot_options['dx']
                                    min_x = plot_options['min_x']
                                    max_x = plot_options['max_x']
                                nHR_mod =  int((max_x-min_x)/dx) 
                                dx_mod = (max_x-min_x)/nHR_mod
                                edgeHR_mod = min_x+np.arange(nHR_mod+1)*dx_mod    
                                cenHR_mod = 0.5*(edgeHR_mod[0:-1]+edgeHR_mod[1::])       
                                input_dic.update({'cont':1.,'rv':0.,'vel':cenHR_mod})
                                if data_fit_prof['func_prof_name'][inst]=='dgauss':
                                    for par_loc in ['amp_l2c','rv_l2c','FWHM_l2c']:input_dic[par_loc]=data_fit_prof['p_final'][data_fit_prof['name_prop2input'][par_loc+'__IS'+inst+'_VS'+vis]]
                                elif data_fit_prof['func_prof_name'][inst]=='voigt':
                                    input_dic['a_damp']=data_fit_prof['p_final'][data_fit_prof['name_prop2input']['a_damp_ord0__IS'+inst+'_VS'+vis]]
                                    
                            #Data-equivalent model                 
                            if plot_options['theo_obs_prof']:                            
                                CCF_jointfit_vis=(np.load(gen_dic['save_data_dir']+'Joined_fits/Intr_prof/IntrProf_fit_'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item())['prof_fit_dic']                            
                                for isub,i_in in enumerate(glob_fit_dic['IntrProf']['idx_in_fit'][inst][vis]):
                                    if data_fit_prof['func_prof_name'][inst]=='gauss':
                                        ctrst_mod_exp,FWHM_mod_exp = gauss_intr_prop(CCF_jointfit_vis[i_in]['ctrst'],CCF_jointfit_vis[i_in]['FWHM'],calc_FWHM_inst(inst,c_light)) 
                                    elif data_fit_prof['func_prof_name'][inst]=='dgauss':
                                        input_dic.update({'ctrst':CCF_jointfit_vis[i_in]['ctrst'],  'FWHM':CCF_jointfit_vis[i_in]['FWHM']})
                                        ctrst_mod_exp,FWHM_mod_exp = cust_mod_true_prop(input_dic,input_dic['vel'],calc_FWHM_inst(inst,c_light))[0:3]                            
                                    elif data_fit_prof['func_prof_name'][inst]=='voigt':
                                        input_dic.update({'ctrst':CCF_jointfit_vis[i_in]['ctrst'],  'FWHM':CCF_jointfit_vis[i_in]['FWHM'],'a_damp':CCF_jointfit_vis[i_in]['a_damp']})
                                        ctrst_mod_exp,FWHM_mod_exp = cust_mod_true_prop(input_dic,input_dic['vel'],calc_FWHM_inst(inst,c_light))[0:3]                                        
                                    if (prop_mode=='FWHM'):val_mod  = FWHM_mod_exp
                                    if (prop_mode=='ctrst'):val_mod = ctrst_mod_exp
                                    if data_fit_prof['coord_line']!=plot_options['prop_'+data_mode+'_absc']:stop('Plot and fit coordinates must match')
                                    plt.plot(data_fit_prop['coord_mod'][inst][vis][isub],val_mod,color='limegreen',linestyle='',lw=1,marker='s',markersize=plot_options['markersize'],zorder=-1)
                       
                            #High-resolution model
                            if plot_options['theo_HR_prof']:  
                                col_mod_prof = 'black'
                                col_mod_prof = col_loc
                                xvar_HR_loc,yvar_HR_loc = sub_plot_HR('from_prof',input_dic,col_mod_prof)
                                
                    if vis=='binned':
                        n_exp_vis = data_bin['n_exp']
                        idx_in =data_bin['idx_in']
                        idx_out=data_bin['idx_out']
                    else:
                        if data_mode=='DI':n_exp_vis = data_vis['n_in_visit']
                        elif data_mode=='Intr':n_exp_vis = data_vis['n_in_tr']                      
                        idx_in =gen_dic[inst][vis]['idx_in']
                        idx_out =gen_dic[inst][vis]['idx_out']
            
                    #SNR 
                    if data_mode=='DI':iexp_plot = range(n_exp_vis)
                    elif data_mode=='Intr':iexp_plot = gen_dic[inst][vis]['idx_in']  
                    for isub,iexp in enumerate(iexp_plot): 
                        SNRS_loc = data_prop[inst][vis]['SNRs'][iexp]
                        if isub==0:SNR_obs = np.zeros([len(iexp_plot)]+list(SNRS_loc.shape))*np.nan
                        SNR_obs[isub] = SNRS_loc
                    
                    #Horizontal property
                    #    - values are put in tables covering all exposures if necessary  
                    x_obs = np.zeros(n_exp_vis)*np.nan
                    st_x_obs = np.zeros(n_exp_vis)*np.nan
                    end_x_obs = np.zeros(n_exp_vis)*np.nan
                    if plot_options['prop_'+data_mode+'_absc']=='phase':
                        if ((data_mode=='DI') and (vis=='binned')) or ((data_mode=='Intr') and (data_type=='Intrbin')):
                            x_obs=data_bin['cen_bindim']
                            st_x_obs=data_bin['st_bindim']
                            end_x_obs=data_bin['end_bindim']                        
                        else:
                            x_obs=coord_vis['cen_ph'][iexp_plot] 
                            st_x_obs=coord_vis['st_ph'][iexp_plot] 
                            end_x_obs=coord_vis['end_ph'][iexp_plot] 
                    elif plot_options['prop_'+data_mode+'_absc'] in ['mu','lat','lon','x_st','y_st','xp_abs','r_proj','y_st2','abs_y_st']:  
                        if data_mode=='DI':iexp_in = gen_dic[inst][vis]['idx_in'] 
                        elif data_mode=='Intr':iexp_in = range(n_exp_vis)                    
                        if plot_options['prop_'+data_mode+'_absc'] in ['mu','lat','lon','x_st','y_st','xp_abs','r_proj']: 
                            x_obs[iexp_in] = transit_prop_nom[plot_options['prop_'+data_mode+'_absc']][0,:]
                            st_x_obs[iexp_in],end_x_obs[iexp_in]  = transit_prop_nom[plot_options['prop_'+data_mode+'_absc']+'_range'][0,:,0],transit_prop_nom[plot_options['prop_'+data_mode+'_absc']+'_range'][0,:,1]
                        elif plot_options['prop_'+data_mode+'_absc'] in ['y_st2','abs_y_st']:
                            if plot_options['prop_'+data_mode+'_absc']=='y_st2':x_obs[iexp_in] = transit_prop_nom['y_st'][0,:]**2.
                            elif plot_options['prop_'+data_mode+'_absc']=='abs_y_st':x_obs[iexp_in] = np.abs(transit_prop_nom['y_st'][0,:])
                            if 'y_st_range' in transit_prop_nom:
                                st_x_obs[iexp_in],end_x_obs[iexp_in] = transit_prop_nom['y_st_range'][0,:,0],transit_prop_nom['y_st_range'][0,:,1]                            
                                cond_cross = ( (st_x_obs<=0.) & (end_x_obs>=0.))
                                bd_x_obs = np.vstack((st_x_obs**2.,end_x_obs**2.))
                                st_x_obs=np.min(bd_x_obs,axis=0)
                                end_x_obs=np.max(bd_x_obs,axis=0)
                                if True in cond_cross:
                                    end_x_obs[cond_cross] = np.max(np.vstack((st_x_obs[cond_cross],end_x_obs[cond_cross])),axis=0)
                                    st_x_obs[cond_cross] = 0.
                            else:st_x_obs,end_x_obs = x_obs,x_obs                  
                    elif plot_options['prop_'+data_mode+'_absc'] in ['ctrst','FWHM']:
                        x_obs=np.array([prof_fit_vis[idx_loc][plot_options['prop_'+data_mode+'_absc']] for idx_loc in range(n_exp_vis)])
                        st_x_obs,end_x_obs = x_obs,x_obs 
                    elif plot_options['prop_'+data_mode+'_absc'] in ['snr','snr_quad']: 
                        if plot_options['prop_'+data_mode+'_absc']=='snr':x_obs =np.mean(SNR_obs[:,plot_options['idx_SNR'][inst]],axis=1)
                        elif plot_options['prop_'+data_mode+'_absc']=='snr_quad':x_obs =np.sqrt(np.sum(SNR_obs[:,plot_options['idx_SNR'][inst]]**2.,axis=1))
                        st_x_obs,end_x_obs = x_obs,x_obs                                          
                    elif plot_options['prop_'+data_mode+'_absc']=='snr_R':                        
                        x_obs=np.mean(data_prop_vis['SNRs'][:,plot_options['idx_num_SNR'][inst]],axis=1)/np.mean(data_prop_vis['SNRs'][:,plot_options['idx_den_SNR'][inst]],axis=1)
                        st_x_obs,end_x_obs = x_obs,x_obs                             
                    elif plot_options['prop_'+data_mode+'_absc'] in ['AM','seeing','RVdrift','colcorrmin','colcorrmax','satur_check','alt','az']:            
                        x_obs=data_prop_vis[plot_options['prop_'+data_mode+'_absc']][iexp_plot]   
                        st_x_obs,end_x_obs = x_obs,x_obs 
                    elif plot_options['prop_'+data_mode+'_absc'] == 'flux_airmass':
                        #Flux decreases as 0.7^(AM^0.678), see Meinel+1976 
                        x_obs=0.7**(data_prop_vis['AM'][iexp_plot]**0.678)
                        st_x_obs,end_x_obs = x_obs,x_obs 
                    elif plot_options['prop_'+data_mode+'_absc']=='colcorrR':            
                        x_obs=data_prop_vis['colcorrmax'][iexp_plot]/data_prop_vis['colcorrmin'][iexp_plot]   
                        st_x_obs,end_x_obs = x_obs,x_obs  
                    elif plot_options['prop_'+data_mode+'_absc'] in ['colcorr450','colcorr550','colcorr650']:
                        idx_prop = {'colcorr450':0,'colcorr550':1,'colcorr650':2}[plot_options['prop_'+data_mode+'_absc']]
                        x_obs = data_prop_vis['colcorr_vals'][iexp_plot,idx_prop]  
                        st_x_obs,end_x_obs = x_obs,x_obs    
                    elif plot_options['prop_'+data_mode+'_absc'] in ['PSFx','PSFy','PSFr','PSFang']:
                        idx_prop = {'PSFx':0,'PSFy':1,'PSFr':2,'PSFang':3}[plot_options['prop_'+data_mode+'_absc']]   
                        x_obs = data_prop_vis['PSF_prop'][iexp_plot,idx_prop] 
                        st_x_obs,end_x_obs = x_obs,x_obs 
                    elif plot_options['prop_'+data_mode+'_absc'] in ['ha','na','ca','s','rhk']:            
                        x_obs=data_prop_vis[plot_options['prop_'+data_mode+'_absc']][iexp_plot,0]   
                        x_obs_err = data_prop_vis[plot_options['prop_'+data_mode+'_absc']][iexp_plot,1]   
                        st_x_obs,end_x_obs = x_obs-x_obs_err   ,x_obs+x_obs_err  
                    elif (plot_options['prop_'+data_mode+'_absc'] in ['ADC1 POS','ADC1 RA','ADC1 DEC','ADC2 POS','ADC2 RA','ADC2 DEC']):    
                        if plot_options['prop_'+data_mode+'_absc']=='ADC1 POS': x_obs=data_prop_vis['adc_prop'][iexp_plot,0]                                                                
                        elif plot_options['prop_'+data_mode+'_absc']=='ADC1 RA': x_obs=data_prop_vis['adc_prop'][iexp_plot,1]    
                        elif plot_options['prop_'+data_mode+'_absc']=='ADC1 DEC': x_obs=data_prop_vis['adc_prop'][iexp_plot,2]    
                        elif plot_options['prop_'+data_mode+'_absc']=='ADC2 POS': x_obs=data_prop_vis['adc_prop'][iexp_plot,3]    
                        elif plot_options['prop_'+data_mode+'_absc']=='ADC2 RA': x_obs=data_prop_vis['adc_prop'][iexp_plot,4]    
                        elif plot_options['prop_'+data_mode+'_absc']=='ADC2 DEC': x_obs=data_prop_vis['adc_prop'][iexp_plot,5] 
                        st_x_obs,end_x_obs = x_obs,x_obs
                    
                    #Vertical property
                    #    - values are put in tables covering all exposures if necessary 
                    if vis not in plot_options['idx_noplot'][inst]:plot_options['idx_noplot'][inst][vis]=[]
                    dic_val_unit = {'rv':'km/s','rv_pip':'km/s','rv_res':'m/s','rv_pip_res':'m/s','FWHM':'km/s','ctrst':''}
                    if prop_mode not in dic_val_unit:dic_val_unit[prop_mode] = ''
                    val_unit = dic_val_unit[prop_mode]
                    if plot_options['plot_data']:
                        val_obs = np.zeros(n_exp_vis)*np.nan
                        eval_obs= np.zeros([2,n_exp_vis])
                        rv_mod_obs = np.zeros(n_exp_vis)*np.nan
                        if prop_mode in ['rv','rv_res','rv_pip_res','FWHM','ctrst','amp','rv_l2c','amp_l2c','FWHM_l2c','RV_lobe','amp_lobe','FWHM_lobe','true_ctrst','cont','c1_pol','c2_pol','c3_pol','c4_pol','vsini',\
                                         'ctrst_ord0__IS__VS_','FWHM_ord0__IS__VS_','FWHM_voigt','area','EW','biss_span','a_damp']:
                            if (prop_mode=='rv_res') and (((data_mode=='DI') and (vis=='binned')) or ((data_mode=='Intr') and (plot_options['theo_HR_prop'] or plot_options['theo_HR_prof'] or plot_options['theo_HR_nom']))):                            
                                prop_loc = 'rv'
                                err_prop_loc = 'err_rv'                            
                            else:
                                prop_loc = prop_mode
                                if prop_mode in ['area','biss_span']:err_prop_loc = None  
                                else:err_prop_loc = 'err_'+prop_mode
                            for idx_loc in range(n_exp_vis):
                                if idx_loc in prof_fit_vis:
                                    val_obs[idx_loc]=prof_fit_vis[idx_loc][prop_loc]
                                    if err_prop_loc is not None:eval_obs[:,idx_loc]=[prof_fit_vis[idx_loc][err_prop_loc][0],prof_fit_vis[idx_loc][err_prop_loc][1]]
                            if (prop_mode in ['rv_res','rv_pip_res']):                               
                                eval_obs*=1e3
                                val_obs*=1e3
                                if (data_mode=='DI'):
                                    for idx_loc in range(n_exp_vis):rv_mod_obs[idx_loc] = prof_fit_vis[idx_loc]['RVmod']*1e3
                                    
                                #Redefine residual from surface RVs using model
                                elif (data_mode=='Intr') and (plot_options['theo_HR_prop'] or plot_options['theo_HR_prof'] or plot_options['theo_HR_nom']):
                                    if plot_options['theo_HR_nom']:
                                        x_theo_HR = x_theo_HR_nom
                                        if plot_options['mod_compos'] == 'full':y_theo_HR = y_theo_HR_nom
                                        elif plot_options['mod_compos'] == 'SB':y_theo_HR = rv_sb_theo_HR_nom                                        
                                    else:
                                        x_theo_HR = xvar_HR_loc
                                        y_theo_HR = yvar_HR_loc                                       
                                    for idx_loc in range(n_exp_vis):
                                        rv_mod_obs[idx_loc] = np.nanmean(y_theo_HR[ (x_theo_HR>=st_x_obs[idx_loc]) & (x_theo_HR<=end_x_obs[idx_loc])])*1e3
                                    val_obs -= rv_mod_obs 

                        elif prop_mode in ['rv_pip','FWHM_pip','ctrst_pip']:
                            val_obs = data_dic['DI'][inst][vis][prop_mode] 
                            eval_obs = np.tile(data_dic['DI'][inst][vis]['e'+prop_mode],[2,n_exp_vis]) 
                        elif prop_mode in ['snr','snr_quad']: 
                            if prop_mode=='snr':
                                val_obs =np.mean(SNR_obs[:,plot_options['idx_SNR'][inst]],axis=1)
                            elif prop_mode=='snr_quad':val_obs =np.sqrt(np.sum(SNR_obs[:,plot_options['idx_SNR'][inst]]**2.),axis=1)
                        elif prop_mode=='snr_R':                      
                            val_obs=np.mean(data_prop_vis['SNRs'][:,plot_options['idx_ord_SNR'][inst]],axis=1)/np.mean(data_prop_vis['SNRs'][:,plot_options['idx_ord_SNR'][inst]],axis=1)                        
                        elif prop_mode in ['AM','seeing','RVdrift','colcorrmin','colcorrmax','satur_check','alt','az']:  
                            val_obs = data_prop_vis[prop_mode].astype(float)  
                        elif prop_mode in ['colcorr450','colcorr550','colcorr650']:
                            idx_prop = {'colcorr450':0,'colcorr550':1,'colcorr650':2}[prop_mode]
                            val_obs = data_prop_vis['colcorr_vals'][:,idx_prop] 
                        elif prop_mode=='glob_flux_sc':
                            val_obs = np.zeros(n_exp_vis,dtype=float)*np.nan
                            for iexp in range(n_exp_vis):
                                val_obs[iexp] = dataload_npz(data_vis['scaled_DI_data_paths']+str(iexp))['glob_flux_scaling'] 
                        elif prop_mode in ['wig_p_0', 'wig_p_1', 'wig_wref', 'wig_a_0', 'wig_a_1', 'wig_a_2', 'wig_a_3', 'wig_a_4']:
                            par_name = prop_mode.split('wig_')[1]
                            data_fit = (np.load(gen_dic['save_data_dir']+'Corr_data/Wiggles/'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item())
                            val_obs = data_fit['p_best_all'][par_name][:,data_fit['iloop_end']]
                        elif prop_mode in ['PC0', 'PC1']:
                            par_name = prop_mode.split('PC')[1]
                            data_fit = (np.load(gen_dic['save_data_dir']+'PCA_results/'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item())
                            val_obs = np.zeros(n_exp_vis,dtype=float)*np.nan
                            for iexp in range(n_exp_vis):
                                if iexp in data_fit['p_best_all']:
                                    val_obs[iexp] = data_fit['p_best_all'][iexp]['a'+par_name]
                        elif prop_mode in ['ha','na','ca','s','rhk']:            
                            val_obs=data_prop_vis[prop_mode][:,0]   
                            eval_obs = np.tile(data_prop_vis[prop_mode][:,1] ,[2,n_exp_vis]) 
                        elif (prop_mode in ['ADC1 POS','ADC1 RA','ADC1 DEC','ADC2 POS','ADC2 RA','ADC2 DEC']):    
                            if prop_mode=='ADC1 POS': val_obs=data_prop_vis['adc_prop'][:,0]                                                                
                            elif prop_mode=='ADC1 RA': val_obs=data_prop_vis['adc_prop'][:,1]    
                            elif prop_mode=='ADC1 DEC': val_obs=data_prop_vis['adc_prop'][:,2]    
                            elif prop_mode=='ADC2 POS': val_obs=data_prop_vis['adc_prop'][:,3]    
                            elif prop_mode=='ADC2 RA': val_obs=data_prop_vis['adc_prop'][:,4]    
                            elif prop_mode=='ADC2 DEC': val_obs=data_prop_vis['adc_prop'][:,5]                         
                        elif (prop_mode in ['TILT1 VAL1','TILT1 VAL2','TILT2 VAL1','TILT2 VAL2']):    
                            if prop_mode=='TILT1 VAL1': val_obs=data_prop_vis['piezo_prop'][:,0]                                                                
                            elif prop_mode=='TILT1 VAL2': val_obs=data_prop_vis['piezo_prop'][:,1]    
                            elif prop_mode=='TILT2 VAL1': val_obs=data_prop_vis['piezo_prop'][:,2]    
                            elif prop_mode=='TILT2 VAL2': val_obs=data_prop_vis['piezo_prop'][:,3]                                              
                        else:stop(prop_mode+' not recognized')

                        #Save residual RVs before screening
                        if (prop_mode=='rv_res') and plot_options['save_RVres']:
                            np.savetxt(path_loc+inst+'_'+vis+'_'+prop_mode+'_'+plot_options['prop_'+data_mode+'_absc']+'.dat', np.column_stack((x_obs,val_obs,np.mean(eval_obs,axis=0))),fmt=('%15.10f','%15.10f','%15.10f') )                                       
          
                        #Points to plot   
                        if data_mode=='DI':
                            idx_in_plot=[iexp for iexp in range(n_exp_vis) if (iexp not in plot_options['idx_noplot'][inst][vis]) and (np.isnan(val_obs[iexp])==False)]
                        elif data_mode=='Intr':
                            idx_in_plot=[i_in for i_in in range(n_exp_vis) if (i_in not in plot_options['idx_noplot'][inst][vis]) and (np.isnan(val_obs[i_in])==False) and   (     ((not plot_options['plot_det'])) or (plot_options['plot_det']  and (prof_fit_vis[i_in]['detected']))        ) ]
                        if len(idx_in_plot)==0:stop('No points to plot')
                        x_obs=x_obs[idx_in_plot]
                        st_x_obs=st_x_obs[idx_in_plot]
                        end_x_obs=end_x_obs[idx_in_plot] 
                        val_obs=val_obs[idx_in_plot]
                        eval_obs=eval_obs[:,idx_in_plot]
                        rv_mod_obs = rv_mod_obs[idx_in_plot]
                        marker_obs=np.repeat(mark_tr,len(idx_in_plot))
                        if prof_fit_vis is None:idx_in_plot_det = np.repeat(True,len(idx_in_plot))
                        else:idx_in_plot_det=prof_fit_vis['cond_detected'][idx_in_plot]
                            
                    else:idx_in_plot = range(len(x_obs))
                    n_exp_vis=len(idx_in_plot)
    
                    #Colors             
                    if plot_options['color_dic'][inst][vis]=='jet':
                        cmap = plt.get_cmap('jet') 
                        col_visit=np.array([cmap(0)]) if n_exp_vis==1 else cmap( np.arange(n_exp_vis)/(n_exp_vis-1.))  
                    else:
                        col_visit=np.repeat(plot_options['color_dic'][inst][vis],n_exp_vis)
            
                    #-------------------------------------------------------

                    #Normalisation
                    if data_mode=='DI':

                        #Out-of-transit normalisation value
                        isub_out_plot=[isub for isub in range(len(val_obs)) if idx_in_plot[isub] in idx_out] 
                        if True in ~np.isnan(val_obs[isub_out_plot]):
                            val_out=np.mean(val_obs[isub_out_plot])
                            marker_obs[isub_out_plot]=mark_otr 
                        else:val_out=np.nan

                        #Normalise values
                        if ('rv' not in prop_mode) and (plot_options['norm_ref']) and (not np.isnan(val_out)):
                            val_obs/=val_out
                            eval_obs/=val_out            

                    #Plot in-transit value with different color or with empty symbols
                    if plot_options['color_dic'][inst][vis]!='jet':
                        isub_in_plot=[isub for isub in range(len(val_obs)) if idx_in_plot[isub] in idx_in] 
                        col_obs=np.zeros(len(idx_in_plot),dtype='U30')
                        col_face_obs=np.zeros(len(idx_in_plot),dtype='U30')
                        col_obs[:] = plot_options['color_dic'][inst][vis]
                        col_face_obs[:] = plot_options['color_dic'][inst][vis]  
                        col_loc = plot_options['color_dic'][inst][vis]

                        #Plot in or all or undetected symbols empty
                        if data_mode=='DI':
                            if plot_options['col_in']!='':col_obs[isub_in_plot]=plot_options['col_in']
                            if plot_options['empty_in']:col_face_obs[isub_in_plot]='white' 
                            if plot_options['col_in']=='none':col_face_obs[isub_in_plot]='none' 
                        if plot_options['empty_all']:col_face_obs[:]='white'  
                        if plot_options['empty_det']:col_face_obs[idx_in_plot_det]='none'  
                        
                    else:
                        col_obs=deepcopy(col_visit)
                        col_face_obs=deepcopy(col_visit)
                        col_loc = 'black'
                    
                    #-------------------------------------------------------
                    #Plot value
                    if (not plot_options['no_orig']) and (plot_options['plot_data']): 
                        HDIval_obs = np.empty([2,0],dtype=float)
                        for i_loc,iexp_eff in enumerate(idx_in_plot):
                            if plot_options['plot_err']:
                                plt.errorbar(x_obs[i_loc],val_obs[i_loc],yerr=[[eval_obs[0,i_loc]],[eval_obs[1,i_loc]]],color=col_obs[i_loc],markeredgecolor=col_obs[i_loc],markerfacecolor=col_face_obs[i_loc],marker='',markersize=plot_options['markersize'],linestyle='',zorder=0,alpha=plot_options['alpha_err'])
                            if plot_options['plot_HDI']:                            
                                data_exp = dataload_npz(gen_dic['save_data_dir']+data_type+'_prop/'+inst+'_'+vis+'_mcmc/iexp'+str(iexp_eff)+'/merged_deriv_chains_walk'+str(plot_options['nwalkers'])+'_steps'+str(plot_options['nsteps']))
                                if prop_mode=='rv_res':ipar = np_where1D(data_exp['var_par_list']=='rv')[0] 
                                else:ipar = np_where1D(data_exp['var_par_list']==prop_mode)[0] 
                                yerr_sub_minmax = [1e100,-1e100]
                                for HDI_sub in data_exp['HDI_interv'][ipar]:
                                    yerr_sub = [HDI_sub[0],HDI_sub[1]]
                                    if prop_mode=='rv_res':yerr_sub = np.array(yerr_sub)*1e3-rv_mod_obs[i_loc]
                                    plt.plot([x_obs[i_loc],x_obs[i_loc]],yerr_sub,color=col_obs[i_loc],marker='',linestyle='-',zorder=0,alpha=plot_options['alpha_err'])
                                    yerr_sub_minmax = [np.min([yerr_sub_minmax[0],yerr_sub[0]]),np.max([yerr_sub_minmax[1],yerr_sub[1]])]                                  
                                HDIval_obs=np.append(HDIval_obs,[[val_obs[i_loc]-yerr_sub_minmax[0]],[yerr_sub_minmax[1]-val_obs[i_loc]]],axis=1)
                            if plot_options['plot_xerr']:plt.errorbar(x_obs[i_loc],val_obs[i_loc],xerr=[[x_obs[i_loc]-st_x_obs[i_loc]],[end_x_obs[i_loc]-x_obs[i_loc]]],color=col_obs[i_loc],markeredgecolor=col_obs[i_loc],markerfacecolor=col_face_obs[i_loc],marker='',markersize=plot_options['markersize'],linestyle='',zorder=0,alpha=plot_options['alpha_err'])                    
                            plt.plot(x_obs[i_loc],val_obs[i_loc],color=col_obs[i_loc],markeredgecolor=col_obs[i_loc],markerfacecolor=col_face_obs[i_loc],marker=marker_obs[i_loc],markersize=plot_options['markersize'],linestyle='',zorder=0,alpha=plot_options['alpha_symb'])                                                
    
                        #Update min/max for the plot
                        x_min=min(x_min,min(x_obs))
                        x_max=max(x_max,max(x_obs))
                        if plot_options['plot_err']:
                            y_min=min(y_min,np.min((val_obs[None,:]-eval_obs)))
                            y_max=max(y_max,np.max((val_obs[None,:]+eval_obs)))
                        else:
                            y_min=min(y_min,np.min(val_obs[None,:]))
                            y_max=max(y_max,np.max(val_obs[None,:]))                    
    
                    # #ANTARESS I - delete afterward
                    # if (gen_dic['star_name']=='WASP76'):
                    #     prof_fit_add=dataload_npz('/Users/bourrier/Travaux/ANTARESS/Ongoing/WASP76b_Saved_data/DIorig_prop/chi2_noplexc/ESPRESSO_'+vis)
                    #     for i_loc in range(len(coord_vis['cen_ph'])):
                    #         val_loc = prof_fit_add[i_loc][prop_mode]
                    #         if (prop_mode=='rv_res'):  val_loc*=1e3
                    #         plt.plot(coord_vis['cen_ph'][i_loc],val_loc,markeredgecolor='black',markerfacecolor='none',marker=mark_tr,markersize=3,linestyle='',zorder=10)                                                
                            
    
    
                    #Store visit values  
                    dic_all['x_all']=np.append(dic_all['x_all'],x_obs)
                    dic_all['st_x_all']=np.append(dic_all['st_x_all'],st_x_obs)
                    dic_all['end_x_all']=np.append(dic_all['end_x_all'],end_x_obs)
                    dic_all['val_all']=np.append(dic_all['val_all'],val_obs)
                    dic_all['eval_all']=np.append(dic_all['eval_all'],eval_obs,axis=1)
                    dic_all['HDI_all']=np.append(dic_all['HDI_all'],HDIval_obs,axis=1)
                    dic_inst['x_all']=np.append(dic_inst['x_all'],x_obs)
                    dic_inst['st_x_all']=np.append(dic_inst['st_x_all'],st_x_obs)
                    dic_inst['end_x_all']=np.append(dic_inst['end_x_all'],end_x_obs)
                    dic_inst['val_all']=np.append(dic_inst['val_all'],val_obs)
                    dic_inst['eval_all']=np.append(dic_inst['eval_all'],eval_obs,axis=1)
                    dic_inst['HDI_all']=np.append(dic_inst['HDI_all'],HDIval_obs,axis=1)
    
                    #-------------------------------------------------------
                    #Print dispersion of residuals to a reference 
                    #    - for in-transit selection we use the ou-of-transit mean as reference                                            
                    if (plot_options['print_disp']!=[] or plot_options['plot_disp']) and (not plot_options['no_orig']):                        
                        if plot_options['disp_mod']=='all':idisp=range(len(val_obs))
                        elif plot_options['disp_mod']=='det':idisp=idx_in_plot_det 
                        elif data_mode=='DI':
                            if plot_options['disp_mod']=='out':idisp=isub_out_plot
                            elif plot_options['disp_mod']=='in':idisp=isub_in_plot    
                        if np.sum(eval_obs[:,idisp])==0.:                        
                            mean_val_plot = np.mean(val_obs[idisp])
                            eval_1D = 0.
                        else:
                            eval_1D = np.mean(eval_obs[:,idisp],axis=0)                        
                            mean_val_plot = np.sum(val_obs[idisp]*(1/eval_1D**2.))/np.sum(1/eval_1D**2.)
                        
                        #Ratio of dispersion to mean error over selected points
                        disp_from_mean=(val_obs[idisp]-mean_val_plot).std()
                        if np.min(eval_1D)>0.:
                            eval_mean = np.mean(eval_1D)
                            disp_err_R = disp_from_mean/eval_mean
                        else:
                            eval_mean = 0.
                            disp_err_R=np.nan
                        dytxt=i_visit*0.1
    
                        #Plot mean value over selected points
                        if plot_options['plot_disp'] and ( ((data_mode=='DI') and (prop_mode not in ['rv','rv_pip'])) or ((data_mode=='Intr') and (prop_mode not in ['rv']))):
                            x_tab = plot_options['x_range'] if plot_options['x_range'] is not None else [min(x_obs),max(x_obs)]
                            plt.plot(x_tab,[mean_val_plot,mean_val_plot],color=col_loc,linestyle='--',lw=plot_options['lw_plot']+0.2,zorder=0) 
                        
                        #Print data quality information in the log and on the figure                       
                        if (plot_options['print_disp']!=[]) and ( (data_mode=='DI') or ((data_mode=='Intr') and (prop_mode not in ['rv']))):
                            if (prop_mode in ['rv_res','rv_pip_res']):
                                # sc_txt = 100.
                                # units = ' (cm/s)'
                                sc_txt = 1.
                                units = ' (m/s)'
                            else:
                                sc_txt = 1.
                                units = ''
                            
                            if 'fig' in plot_options['print_disp']:
                                plt.text(0.2,1.1+dytxt,'w. mean['+prop_mode+'] ='+"{0:.5e}".format(mean_val_plot)+' +-'+"{0:.2e}".format(disp_from_mean)+' '+val_unit,verticalalignment='center', horizontalalignment='center',fontsize=10.,zorder=10,color=col_loc,transform=plt.gca().transAxes) 
                                if ~np.isnan(disp_err_R):plt.text(0.6,1.1+dytxt,'$\sigma$/<e> ='+"{0:.5e}".format(disp_err_R),verticalalignment='center', horizontalalignment='left',fontsize=10.,zorder=10,color=col_loc,transform=plt.gca().transAxes) 
                            if 'plot' in plot_options['print_disp']:
                                if ~np.isnan(disp_err_R):plt.text(0.8,0.9-dytxt,'$\sigma$/<e> ='+"{0:.2e}".format(disp_err_R),verticalalignment='center', horizontalalignment='left',fontsize=12.,zorder=10,color=col_loc,transform=plt.gca().transAxes) 
                            if 'log' in plot_options['print_disp']:                            
                                print('       wm =',"{0:.5f}".format(sc_txt*mean_val_plot)+units)
                                print('       <e> =',"{0:.5e}".format(sc_txt*eval_mean)+units)
                                print('       std =',"{0:.5e}".format(sc_txt*disp_from_mean)+units)
                                if (prop_mode not in ['rv_res','rv_pip_res']):
                                    print('       e_r =',"{0:.5f}".format(1e6*eval_mean/mean_val_plot)+' (ppm)')
                                    print('       std_r =',"{0:.5f}".format(1e6*disp_from_mean/mean_val_plot)+' (ppm)')
                                if ~np.isnan(disp_err_R):print('       std/e =',"{0:.5e}".format(disp_err_R))
                                  
                        #Save dispersion values for analysis in external routine
                        if (data_mode=='DI') and plot_options['save_disp']:
                            data_save = {'disp_err_R':disp_err_R,'disp_from_mean':disp_from_mean,'emean':eval_mean,'delta_mean':np.mean(val_obs[isub_in_plot])-np.mean(val_obs[isub_out_plot])}
                            np.savez(gen_dic['save_dir']+'Dispersions/'+gen_dic['main_pl_text']+'_Raw_'+prop_mode,data=data_save,allow_pickle=True)    
    
                        #Plot results from common fit to local CCFs
                        if ('plot_fit_comm' in plot_options) and (inst in plot_options['plot_fit_comm']) and (vis in plot_options['plot_fit_comm'][inst]):
                            fit_allCCF_results=data_dic['Res'][inst][vis]['fit_allCCF_results']
                            plot_options['font_size_txt']=12
                            xtxt_left=-0.02
                            ytxt_ref=2.
                            ygap=0.4
                            plt.text(xtxt_left,ytxt_ref-0.*ygap,'v$_\mathrm{eq}$ = '+"{0:.4f}".format(fit_allCCF_results['veq'])+'+-'+"{0:.4f}".format(fit_allCCF_results['err_veq']),verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40) 
                            plt.text(xtxt_left,ytxt_ref-1.*ygap,'$\lambda$ = '+"{0:.4f}".format(fit_allCCF_results['lamb'])+'+-'+"{0:.4f}".format(fit_allCCF_results['err_lamb']),verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40) 
                            plt.text(xtxt_left,ytxt_ref-2.*ygap,'i$_{*}$ = '+"{0:.4f}".format(fit_allCCF_results['istar'])+'+-'+"{0:.4f}".format(fit_allCCF_results['err_istar']),verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40) 
                            plt.text(xtxt_left,ytxt_ref-3.*ygap, r'$\alpha_{DR}$ = '+"{0:.4f}".format(fit_allCCF_results['alpha'])+'+-'+"{0:.4f}".format(fit_allCCF_results['err_alpha']),verticalalignment='center', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=40) 
                        
    
    
                    #-------------------------------------------------------
                    #Theoretical radial velocity of star relative to the Sun (in km/s)
                    #    - at high temporal resolution, between min/max bjd for the visit
                    if (data_mode=='DI') and (prop_mode in ['rv','rv_pip','RV_lobe']) and (plot_options['prop_'+data_mode+'_absc']=='phase') and plot_options['theoRV']:   
                        phase_RV_star,RV_star_solCDM = calc_rv_star_HR(pl_ref,system_param,gen_dic['kepl_pl'],coord_dic,inst,vis,data_dic)
                        shift_RV=np.nanmean([prof_fit_vis[idx_loc]['rv_l2c'] for idx_loc in idx_out]) if prop_mode=='RV_lobe'  else 0. 
                        plt.plot(phase_RV_star,RV_star_solCDM+shift_RV,color=col_loc,linestyle='-',lw=plot_options['lw_plot'],zorder=0)
    
                    #-------------------------------------------------------
                    #Fit the selected ordina property as a function of abscissa property
                    cond_plot_fit = True if ((inst in plot_options['idx_fit']) and (vis in plot_options['idx_fit'][inst])) else False
                    if cond_plot_fit and (((inst in plot_options['deg_prop_fit']) and (vis in plot_options['deg_prop_fit'][inst])) or (((inst in plot_options['fit_sin']) and (vis in plot_options['fit_sin'][inst])))):                      
                        print('       Fit versus')
    
                        #Absolute index for fit, in the plot-reduced table
                        #    - by default, out-of-transit points for disk-integrated properties, all points for intrinsic properties
                        if len(plot_options['idx_fit'][inst][vis])==0:
                            if (data_mode=='DI'):plot_options['idx_fit'][inst][vis] = isub_out_plot
                            if (data_mode=='Intr'):plot_options['idx_fit'][inst][vis] = range(len(val_obs))
                        idx_fit_loc=plot_options['idx_fit'][inst][vis]
                        npts_fit=len(idx_fit_loc)
    
                        #Use dispersion on residuals to set the error on properties
                        #    - we perform a preliminary fit to set the dispersion from the residuals   
                        if plot_options['set_err'] is None:
                            err_disp=np.repeat((val_obs[idx_fit_loc]-np.mean(val_obs[idx_fit_loc])).std(),npts_fit)
                        else:
                            eval_1D = np.mean(eval_obs,axis=0)
                            err_disp=plot_options['set_err']*eval_1D[idx_fit_loc]    
                            if np.max(err_disp)==0.:
                                print('No errors defined on ',prop_mode,': set to constant value')
                                err_disp = np.repeat((val_obs[idx_fit_loc]-np.mean(val_obs[idx_fit_loc])).std(),npts_fit)
                        
                        #Define fit properties
                        fixed_args={'use_cov':False,'deg_pol':{},'var_prop':{},'var_fit':{}}
                        if ((inst in plot_options['fit_sin']) and (vis in plot_options['fit_sin'][inst])):fixed_args['sin_prop'] = [plot_options['fit_sin'][inst][vis]] 
                        else:fixed_args['sin_prop'] = []   
                        if ((inst in plot_options['deg_prop_fit']) and (vis in plot_options['deg_prop_fit'][inst])):fixed_args['pol_prop'] = list(plot_options['deg_prop_fit'][inst][vis].keys()) 
                        else:fixed_args['pol_prop'] = []                    
                        fixed_args['prop_fit'] = list(set(fixed_args['sin_prop']+fixed_args['pol_prop']))
                        
                        #Global correction functions
                        #    - defined as F(var) = a0*F(var1)*F(var2)*..
                        #              or F(var) = a0+F(var1)+F(var2)*..
                        p_guess = Parameters()
                        if prop_mode in ['rv','rv_res']:p_guess.add_many(('a0', 0., True,None,None,  None)) 
                        else:p_guess.add_many(('a0', np.mean(val_obs[idx_fit_loc]), True,None,None,  None))                        
                        for iprop,prop_fit in enumerate(fixed_args['prop_fit']):
                            if prop_fit=='phase':var_prop = coord_vis['cen_ph'][iexp_plot]
                            elif prop_fit=='snr':var_prop = np.mean(SNR_obs[:,plot_options['idx_SNR'][inst]],axis=1)
                            elif prop_fit=='snr_quad':var_prop = np.sqrt(np.sum(SNR_obs[:,plot_options['idx_SNR'][inst]]**2.,axis=1))
                            else:var_prop = x_obs
                            fixed_args['var_prop'][prop_fit] = var_prop[idx_in_plot]
                            fixed_args['var_fit'][prop_fit] = var_prop[idx_in_plot][idx_fit_loc]
                            
                            #Sine function
                            #    - F(var) = (1 + A*sin(2*pi*((var-var0)/P)))
                            if prop_fit in fixed_args['sin_prop']:
                                p_guess.add_many((prop_fit+'_amp',(np.max(val_obs) - np.min(val_obs)) / 2,True,  0.,None,None),
                                                 (prop_fit+'_off',0.,True,  None,None,None),
                                                 (prop_fit+'_per',(np.max(x_obs) - np.min(x_obs)) / 4,True, 0.,None,None))  
                            #Polynomial variation
                            #    - F(var) = (1 + c1*x + c2*x^2 + ...)
                            #   or F(var) =  a1*x + a2*x^2 + ...
                            #    - if several trends are combined, a single a0 is required
                            if prop_fit in fixed_args['pol_prop']:
                                fixed_args['deg_pol'][prop_fit]=plot_options['deg_prop_fit'][inst][vis][prop_fit]
                                if fixed_args['deg_pol'][prop_fit]>0:
                                    for ideg in range(1,plot_options['deg_prop_fit'][inst][vis][prop_fit]+1):  
                                        p_guess.add_many((prop_fit+'_c'+str(ideg), 0., True,None,None,  None))                                 
    
                        #Fitting
                        nfree = 0.
                        for par in p_guess:
                            if p_guess[par].vary:nfree+=1.
                        fixed_args['idx_fit'] = np.ones(npts_fit,dtype=bool)
                        result_loc,merit,p_best = call_lmfit(p_guess,np.zeros(npts_fit),val_obs[idx_fit_loc],np.array([err_disp**2.]),fit_pol_sin,fixed_args=fixed_args)
                       
                        #Best fit    
                        p_obs = fit_pol_sin(p_best,np.zeros(npts_fit),args=fixed_args) 
                        print('          a0'+'='+"{0:.6e}".format(p_best['a0'].value)) 
                        for prop_fit in fixed_args['prop_fit']:
                            print('         ',prop_fit)
                            if prop_fit in fixed_args['sin_prop']:
                                print('       Asin ='+"{0:.6e}".format(p_best[prop_fit+'_amp'].value)) 
                                print('       X0sin ='+"{0:.6e}".format(p_best[prop_fit+'_off'].value)) 
                                print('       Psin ='+"{0:.6e}".format(p_best[prop_fit+'_per'].value))  
                            if (prop_fit in fixed_args['pol_prop']) and (fixed_args['deg_pol'][prop_fit]>0):
                                for ideg in range(1,plot_options['deg_prop_fit'][inst][vis][prop_fit]+1): 
                                    print('           c'+str(ideg)+'='+"{0:.6e}".format(p_best[prop_fit+'_c'+str(ideg)].value)) 
                                        
          
                        #Fit quality
                        bestchi2=np.sum( ( (val_obs[idx_fit_loc] - p_obs)/err_disp )**2.)
                        print('       Chi2 = '+str(bestchi2))
                        print('       Reduced Chi2 ='+str(bestchi2/(npts_fit-nfree)))
                        print('       BIC ='+str(bestchi2+nfree*np.log(npts_fit)))  
                        print('       Dispersion :')
                        print('         pre-fit :'+str((val_obs[idx_fit_loc]-np.mean(val_obs[idx_fit_loc])).std()))  
                        res_obs = val_obs[idx_fit_loc] - p_obs 
                        print('         post-fit :'+str((res_obs-np.mean(res_obs)).std()))                              
                     
                        #Best-fit at high-resolution  
                        #    - this is only possible if there is a single variable, otherwise we would need to create a HR multi-dimensional table by interpolating the observed one, as a given observed variable matches a set of other ones
                        if len(fixed_args['prop_fit'])==1:
                            nx_HR = 100
                            prop_fit=fixed_args['prop_fit'][0]                        
                            x_mod = deepcopy(fixed_args['var_prop'][prop_fit])  
                            dx_HR=(max(x_mod)-min(x_mod))/nx_HR
                            fixed_args['var_fit'][prop_fit] = min(x_mod) + dx_HR*np.arange(nx_HR)
                            p_mod=fit_pol_sin(p_best,np.zeros(nx_HR),args=fixed_args)
                            plt.plot(fixed_args['var_fit'][plot_options['prop_'+data_mode+'_absc']],p_mod,color='darkgrey',linestyle='-',lw=1,zorder=50)                            
                        else:
                            for prop_fit in fixed_args['prop_fit']:
                                fixed_args['var_fit'][prop_fit]  = fixed_args['var_prop'][prop_fit]                          
                            p_mod=fit_pol_sin(p_best,np.zeros(len(fixed_args['var_fit'][prop_fit])),args=fixed_args)
                            wsort = np.argsort(fixed_args['var_fit'][plot_options['prop_'+data_mode+'_absc']])
                            plt.plot(fixed_args['var_fit'][plot_options['prop_'+data_mode+'_absc']][wsort],p_mod[wsort],color='darkgrey',linestyle='-',lw=1,zorder=50)

                        #Plot fit as a function of current variable   
                        if plot_options['plot_err']:plt.errorbar(x_obs[idx_fit_loc],val_obs[idx_fit_loc],yerr=err_disp,color=col_loc,markeredgecolor=col_loc,marker='',markersize=plot_options['markersize'],linestyle='',zorder=50,alpha=0.5)                       
    
                      
                    #-------------------------------------------------------           
                    #Master property 
                    if (prop_mode in ['rv_res','rv_pip_res','FWHM','ctrst','amp','rv_l2c','amp_l2c','FWHM_l2c','amp_lobe','FWHM_lobe','area']) and (not plot_options['no_orig']) and \
                        ( (plot_options['plot_Mout'] and (data_dic['DI']['type'][inst]=='CCF')) or ( (plot_options['plot_Mloc'] and (data_dic['Res']['type'][inst]=='CCF')))):                                                
                        
                        if (data_mode=='DI') and (plot_options['norm_ref'] or ('rv' in prop_mode)):
                            val_ref = val_out
                        else:
                            if plot_options['plot_Mout']:
                                Bin_prof_fit_vis=(np.load(gen_dic['save_data_dir']+'DIbin_prop/'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item())[0]
                            if plot_options['plot_Mloc']:
                                Bin_prof_fit_vis=(np.load(gen_dic['save_data_dir']+'Intrbin_prop/'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item())[0] 
                            val_ref=Bin_prof_fit_vis[prop_loc]
                        plt.plot([min(x_obs),max(x_obs)],np.repeat(val_ref,2),color=col_loc,linestyle=':',lw=plot_options['lw_plot'],zorder=0)
    
    
                    #Contacts
                    if plot_options['prop_'+data_mode+'_absc']=='phase':
                        for ipl,pl_loc in enumerate(data_dic[inst][vis]['transit_pl']):
                            if (i_visit==0) or ((pl_loc in gen_dic['Tcenter_visits']) and (inst in gen_dic['Tcenter_visits'][pl_loc]) and (vis in gen_dic['Tcenter_visits'][pl_loc][inst])): 
                                if pl_loc==pl_ref:
                                    cen_ph = 0.
                                    contact_phases_vis = contact_phases[pl_ref]
                                else:
                                    contact_times = coord_dic[inst][vis][pl_loc]['Tcenter']+contact_phases[pl_loc]*system_param[pl_loc]["period"]
                                    contact_phases_vis = (contact_times-coord_dic[inst][vis][pl_ref]['Tcenter'])/system_param[pl_ref]["period"]  
                                    cen_ph = (coord_dic[inst][vis][pl_loc]['Tcenter']-coord_dic[inst][vis][pl_ref]['Tcenter'])/system_param[pl_ref]["period"] 
                                ls_pl = {0:':',1:'--'}[ipl]
                                for cont_ph in contact_phases_vis:
                                    plt.plot([cont_ph,cont_ph],[-1e6,1e6],color='black',linestyle=ls_pl,lw=plot_options['lw_plot'],zorder=0)
                
                                #Overplot transit duration from system properties
                                if (data_mode=='DI') and plot_options['plot_T14']:
                                    T14_phase = system_param[pl_loc]['TLength']/(system_param[pl_ref]['period'])
                                    plt.plot([cen_ph-0.5*T14_phase,cen_ph-0.5*T14_phase],[-1e6,1e6],color='black',linestyle='--',lw=plot_options['lw_plot'],zorder=0)
                                    plt.plot([cen_ph+0.5*T14_phase,cen_ph+0.5*T14_phase],[-1e6,1e6],color='black',linestyle='--',lw=plot_options['lw_plot'],zorder=0)                              

                        #Use main planet contact as plot range if undefined
                        if (plot_options['x_range'] is None) and (data_mode=='Intr'):
                            delt_range = 0.05*system_param[pl_ref]['T14_num']/system_param[pl_ref]["period"]
                            plot_options['x_range'] = np.array([contact_phases[pl_ref][0]-delt_range,contact_phases[pl_ref][3]+delt_range])   
                                              
                    #-------------------------------------------------------
                    #Predicted local RVs measurements from nominal system properties
                    if (data_mode=='Intr') and ('predic' in plot_options) and (len(plot_options['predic'])>0) and (prop_mode=='rv') and (plot_options['prop_'+data_mode+'_absc']=='phase'):                 
                  
                        #Sub-function predicting errors on local RVs (in km/s)
                        #    - the error on the RV measurement derived from a CCF writes as (see Boisse et al.  (2010), A&A 523, 88)
                        # eRV(t) = 1 / Q_CCF * sqrt( sum( i, CCF[i,t] ) )
                        # eRV(t) = sqrt( sum(i, CCF[i,t] ) )  / [ sqrt( sum(i,  dCCFdv[i,t]^2 / sigCCF[i,t]^2    )  ) * sqrt(Nscale) * sqrt( sum(i, CCF[i,t] ) ) ]
                        # eRV(t) = 1  / [ sqrt( sum(i,  dCCFdv[i,t]^2 / sigCCF[i,t]^2    )  ) * sqrt(Nscale) ]
                        #      where dCCFdv[i,t] = ( CCF[i+1,t] - CCF[i,t] )/dRV
                        #            Nscale = dRV/pix_size       
                        #      here we assume that dRV = pix_size    
                        #      we assume that the RV is mostly constrained by a given region of the CCF where the slope is strongest (which we take at the FWHM), so that :            
                        # eRV(t) = 1  / sqrt( dCCFdv(t)^2 / sigCCF(t)^2    )              
                        # eRV(t) = sigCCF(t)  / dCCFdv(t)
                        #      assuming that the local CCF is a gaussian, it writes as
                        # CCF_Res(rv,t) = CCF_Res_cont(t)*(1 - CT_loc*exp( -(2*sqrt(ln2)*(rv-rv0)/FWHM_loc)^2 )) 
                        #      and its slope for rv = rv0+FWHM_loc is proportional to
                        # dCCFdv(t) ~ CCF_Res_cont(t)*CT_loc/FWHM_loc = CCFref_cont*(1 - LC(t))*CT_loc/FWHM_loc  
                        #      where LC(t) is the normalized light curve (1 outside of transit, 0 if full absorption), CT_loc and FWHM_loc the contrast and FWHM_loc of the local CCF
                        # eRV(t) = C*sigCCF(t)  / ( CCFref_cont*(1 - LC(t))*CT_loc/FWHM_loc  )
                        #      where C is a factor in which we will include all contributions that are not dependent on time, the instrument, and the system   
                        #      see weights_bin_prof(), fluxes are defined as F = g_inst*N for a given instrument, and the error on the local CCF flux approximates as:
                        # sigCCF(t) = LC(t)*sqrt( g_inst*CCFref(rv) )
                        #      so that
                        # eRV(t) = C*LC(t)*sqrt( g_inst^2*CCF_Nref(rv) )  / ( g_inst*CCF_Nref_cont*(1 - LC(t))*CT_loc/FWHM_loc  )                                
                        # eRV(t) = C*LC(t)*sqrt( CCF_Nref_cont )  / ( CCF_Nref_cont*(1 - LC(t))*CT_loc/FWHM_loc  )                                     
                        #      where we neglected the errors on CCFref and the effects of Earth atmosphere, and the flux and error variations of the master over the rv range
                        #      here we also have to account for the fact that the number of measured photons depends on the telescope size and efficiency of the instrument, so that:
                        # N = C_inst*Ntrue
                        #      and 
                        # eRV(t) = C*LC(t)*sqrt( CCF_Ntrue_ref_cont )  / ( sqrt(C_inst)*CCF_Ntrue_ref_cont*(1 - LC(t))*CT_loc/FWHM_loc  )                                 
                        # eRV(t) = C * ( LC(t)/(1 - LC(t)) )*sqrt( 1/C_inst )*(FWHM_loc/CT_loc) / sqrt(CCF_Ntrue_ref_cont)          
                        #      the flux on the master-out CCF (here, the number of measured photon) scales with the stellar magnitude and the exposure time
                        # eRV(t) = C * ( LC(t)/(1 - LC(t)) )*sqrt( 1/C_inst )*(FWHM_loc/CT_loc) / sqrt(10^(-V/2.5)*texp)               
                        # eRV(t) = C * ( LC(t)/(1 - LC(t)) )*sqrt( 1/C_inst )*(FWHM_loc/CT_loc)*10^(V/5)*sqrt(1/texp)  
                        #    - we define an estimate for the RM Revolutions signal, as the SNR of the amplitude of the local stellar line:
                        # SNR(t) = A/sigA(t)
                        #      with A = cont - min = C*cont
                        #        = C*cont/sqrt( sig_cont^2 + sig_min^2 )
                        #      we assume sig_cont ~ sig_min ~ sigCCF(t)
                        # SNR(t) = C*CCF_Res_cont(t)/sigCCF(t)
                        #        = C*CCFref_cont*(1 - LC(t))/(LC(t)*sqrt( g_inst*CCFref(rv) ))
                        #        = C*CCF_Nref_cont*(1 - LC(t))/(LC(t)*sqrt( g_inst*CCF_Nref(rv) ))
                        #        = C*CCF_Nref_cont*(1 - LC(t))/(LC(t)*sqrt( g_inst*CCF_Nref_cont ))
                        #        = C*C_inst*CCF_Ntrue_ref_cont*(1 - LC(t))/(LC(t)*sqrt( g_inst*C_inst*CCF_Ntrue_ref_cont ))
                        #        = C*sqrt(C_inst*CCF_Ntrue_ref_cont)*(1 - LC(t))/(LC(t)*sqrt(g_inst))
                        #        = C*sqrt(C_inst*10^(-V/2.5)*texp)*(1 - LC(t))/(LC(t)*sqrt(g_inst))
                        #      we can consider that g_inst is included in C_inst, so that
                        # SNR(t) = C*sqrt(C_inst)*10^(-V/5)*sqrt(texp)*(1 - LC(t))/LC(t)
                        #      over the transit, if we consider that the line amplitude remains constant, the SNR of the global signal writes as:
                        # SNRtr = A/sigA_tr
                        #       = A/sqrt( sum( sigA(t)^2  ) )
                        #       = A/sqrt( sum( A^2/SNR(t)^2  ) )
                        #       = 1/sqrt( sum( 1/SNR(t)^2  ) )
                        #    - the SNR and local RV errors are linked as:
                        # SNR(t) * eRV(t) = C^2*(FWHM_loc/CT_loc)/LC(t)    
                        def sub_def_err_RVloc(const_fact,FWHM_loc,CT_loc,LC_exp,C_inst,texp):
                            return const_fact*( LC_exp/(1. - LC_exp) )*sqrt(1./C_inst)*(FWHM_loc/CT_loc)*10.**(system_param['star']['mag']/5.)*sqrt(1./texp)
                    
                        #High-resolution light curve
                        data_upload = dataload_npz(gen_dic['save_data_dir']+'Scaled_data/'+inst+'_'+vis+'_add')
                        ph_LC_HR =  data_upload['coord_HR'][pl_ref]['cen_ph']
                        LC_HR = data_upload['LC_HR'][:,0]

                        #-------------------------------------------                        
                        #Predictions of RM signal
                        RVpred_tab = np.zeros([4,0],dtype=float)
                        SNRpred_tab = np.zeros(0,dtype=float)
                        
                        #Flux gain between the considered instrument and VLT/ESPRESSO
                        C_inst_dic = {
                            'ESPRESSO':1.,      
                            'HARPS':1./6.,    
                            'NIRPS':1./5.,     #from C. Lovis, efficiency roughly similar to ESPRESSO, thus flux ratio scales as mirror size ratio 
                            }
                        if inst not in C_inst_dic:stop('Define '+inst+'/ESPRESSO flux gain')
                        
                        #Processing exposures
                        for low_ph,high_ph in zip(st_x_obs[idx_in_plot],end_x_obs[idx_in_plot]):
                            cond_in = (xvar_HR[wsort]>low_ph) & (xvar_HR[wsort]<=high_ph)
                            cond_in_LC = (ph_LC_HR>low_ph) & (ph_LC_HR<=high_ph)
                            if (True in cond_in) and (True in cond_in_LC):
                                ph_RV_exp = np.mean(xvar_HR[wsort][cond_in])
                                RV_exp = np.mean(theo_HR_prop_plocc[pl_ref]['rv'][wsort][cond_in])
                                LC_exp = np.mean(LC_HR[cond_in_LC])
                                if (~np.isnan(RV_exp)) and (LC_exp<1.):
                                    texp = (high_ph-low_ph)*system_param[pl_ref]['period_s']
                                    eRV_exp = sub_def_err_RVloc(plot_options['predic']['C'],plot_options['predic']['FWHM'],plot_options['predic']['ctrst'],LC_exp,C_inst_dic[inst],texp)                                    
                                    if plot_options['predic']['rand']:RV_plot = np.random.normal(loc=RV_exp, scale=eRV_exp)
                                    else:RV_plot=RV_exp 
                                    plt.errorbar(ph_RV_exp,RV_plot,xerr=[[ph_RV_exp-low_ph],[high_ph-ph_RV_exp]],yerr=eRV_exp,color='black',markeredgecolor='black',markerfacecolor='black',marker='o',linestyle='',zorder=0,alpha=plot_options['alpha_err'])
                                    
                                    #Store surface RV
                                    RVpred_tab=np.append(RVpred_tab,[[RV_exp],[RV_plot],[eRV_exp],[ph_RV_exp]],axis=1) 
                                    
                                    #Store SNR on the contrast of the local stellar line
                                    #    - scales as cst*sqrt(C_inst)*C*(1 - LC(t))/LC(t)*10^(-V/5)*sqrt(texp)
                                    SNRpred_tab=np.append(SNRpred_tab,plot_options['predic']['C_RMR']*sqrt(C_inst_dic[inst])*plot_options['predic']['ctrst']*((1 - LC_exp)/LC_exp)*10.**(-system_param['star']['mag']/5.) *np.sqrt(texp))

                        #-------------------------------------------  
                                    
                        #Calculate deltachi2 with null hypothesis
                        chi2_mod = np.sum(((RVpred_tab[1]-RVpred_tab[0])/RVpred_tab[2])**2.)   #prediction randomized vs model
                        chi2_null = np.sum(((RVpred_tab[1]-0.)/RVpred_tab[2])**2.)             #prediction randomized vs null hypothesis
                        print('     chi2[null]-chi2[mod] =',chi2_null- chi2_mod)
    
                        # #Calculate deltachi2 with nominal and uploaded model
                        # RVmod_up = np_interp(RVpred_tab[3],xvar_al,RV_stsurf_al)
                        # chi2_mod_up = np.sum(((RVpred_tab[1]-RVmod_up)/RVpred_tab[2])**2.)  
                        # print('     chi2[mod]-chi2[sec. mod] =',np.abs(chi2_mod_up- chi2_mod))
    
                        #Predictions of RM Revolutions signal
                        SNR_RMR = 1./np.sqrt( np.sum( 1./SNRpred_tab**2.  ) )
                        print('     SNR(RMR) =',SNR_RMR)
    
            ### end of visit                           
 
            #-------------------------------------------------------
            #Plot data binned over all visit datasets for current instrument
            if (inst in plot_options['bin_val']) and (plot_options['plot_data']): 
                x_min,x_max,y_min,y_max=sub_func_bin(plot_options['bin_val'][inst],dic_inst,x_min,x_max,y_min,y_max,plot_options)   

        ### end of instrument

        if (gen_dic['star_name']=='WASP76'):   #ANTARESS I
            plt.gca().axvspan(-0.002 ,0.014, facecolor='lightgrey', alpha=0.3,zorder=-10)
            plt.gca().fill([-0.002,0.014,0.014,-0.002],[plot_options['y_range'][0],plot_options['y_range'][0],plot_options['y_range'][1],plot_options['y_range'][1]], fill=False, hatch='\\',color='grey',zorder=-10)         


        #-------------------------------------------------------
        #Plot data binned over all instrument datasets
        if ('all' in plot_options['bin_val']) and (plot_options['plot_data']): 
            x_min,x_max,y_min,y_max=sub_func_bin(plot_options['bin_val']['all'],dic_all,x_min,x_max,y_min,y_max,plot_options) 

        #-------------------------------------------------------
        #Ranges   
        if plot_options['plot_bounds']:
            print('     Max X range =',np.array([x_min,x_max])  )
            print('     Max Y range =',np.array([y_min,y_max])  )                               
        if plot_options['x_range'] is not None:x_range_loc=plot_options['x_range']
        else:   
            delt_range = 0.05*( x_max-x_min )
            x_range_loc = np.array([x_min-delt_range,x_max+delt_range])   
        if plot_options['y_range'] is not None:y_range_loc=plot_options['y_range'] 
        else:
            dy_range = y_max-y_min
            y_range_loc = np.array([y_min-0.05*dy_range,y_max+0.05*dy_range])  

        #Reference level
        if plot_options['plot_ref']:
           if ((data_mode=='DI') and (prop_mode in ['rv_res','rv_pip_res'])) or ((data_mode=='Intr') and (prop_mode in ['rv','rv_res'])):val_ref = 0.
           elif (data_mode=='DI') and plot_options['norm_ref']:val_ref = 1.
           else:val_ref = None
           plt.plot(x_range_loc,[val_ref,val_ref],color='black',linestyle=':',lw=plot_options['lw_plot'],zorder=0)    

        #Meridian
        if prop_mode=='az':plt.plot(x_range_loc,[180.,180.],color='black',linestyle=':',lw=plot_options['lw_plot'],zorder=0)

        #Abscissa properties 
        xmajor_int=None
        xminor_int=None    
        x_title_dic={ 
            'phase':'Orbital phase',
            'mu':'$\mu$',
            'lat':'Stellar latitude ($^{\circ}$)'   ,
            'lon':'Stellar longitude ($^{\circ}$)'  ,  
            'x_st':'Stellar X position (R$_{*}$)' ,
            'y_st':'Stellar Y position (R$_{*}$)' ,
            'y_st2':'Stellar Y$^{2}$ position (R$_{*}^{2}$)', 
            'AM':'Airmass' ,
            'flux_airmass':'Abs[Airmass]' ,
            'seeing':'Seeing'  ,
            'snr':'snr'     ,
            'snr_quad':'snr',
            'snr_R':'snr ratio',
            'RVdrift':'RV drift (m s$^{-1}$)',
            'abs_y_st':'Stellar |Y| position (R$_{*}$)', 
            'cstr_loc':'Local contrast',
            'FWHM_loc':'Local FWHM (km s)', 
            'rv_loc':'Local RV',
            'xp_abs':'Distance from normal',
            'r_proj':'Distance from star center',
            'colcorrmin':'Min color coefficient',
            'colcorrmax':'Max color coefficient',
            'colcorrR':'Max/min color coefficient',
            'satur_check':'Saturation flag',
            'ctrst':'Contrast',
            'FWHM':'FWHM (km s$^{-1}$)',
            'colcorr450':'Color coeff at 450nm',
            'colcorr550':'Color coeff at 550nm' ,
            'colcorr650':'Color coeff at 650nm' ,
            'PSFx':'PSFx',
            'PSFy':'PSFy',
            'PSFr':'PSFr',
            'PSFang':'PSFang',
            'alt':'Tel. altitude ($^{\circ}$)',
            'az':'Tel. azimuth ($^{\circ}$)',
            'ha':r'H$_{\alpha}$ index',
            'na':'Na index',
            'ca':'Ca index',
            's':'S index',
            'rhk':'R$_{hk}$ index'
            }
        if plot_options['prop_'+data_mode+'_absc'] not in x_title_dic:x_title_dic[plot_options['prop_'+data_mode+'_absc']]=plot_options['prop_'+data_mode+'_absc']
            
        y_title_dic={
                'rv_pip':'RV pip. (km s$^{-1}$)',
                'rv_pip_res':'RV pip. res (m s$^{-1}$)',
                'FWHM':'FWHM (km s$^{-1}$)',
                'FWHM_voigt':'FWHM(Voigt) (km s$^{-1}$)',
                'FWHM_ord0__IS__VS_':'Local FWHM (deg 0) (km s$^{-1}$)',
                'FWHM_pip':'FWHM pip. (km s$^{-1}$)',
                'vsini':'v$_\mathrm{eq}$sin$\,$i$_{\star}$ (km s$^{-1}$)',
                'amp':'Amplitude',
                'cont':'Continuum level',                    
                'c1_pol':'Pol. cont. deg 1',                    
                'c2_pol':'Pol. cont. deg 2',                     
                'c3_pol':'Pol. cont. deg 3',                     
                'c4_pol':'Pol. cont. deg 4',  
                'ctrst':'Contrast',
                'ctrst_ord0__IS__VS_':'Local contrast (deg 0)',
                'ctrst_pip':'Contrast pip.',
                'true_ctrst':'Contrast$_\mathrm{true}$',
                'AM':'Airmass',
                'seeing':'Seeing',
                'snr':'SNR' ,
                'snr_quad':'SNR' ,
                'rv_l2c':'RV$_\mathrm{lobe}$-RV$_\mathrm{core}$ (km s$^{-1}$)',
                'amp_l2c':'A$_\mathrm{lobe}$/A$_\mathrm{core}$' ,
                'FWHM_l2c':'FWHM$_\mathrm{lobe}$/FWHM$_\mathrm{core}$' ,
                'area':'Area',
                'RV_lobe':'RV$_\mathrm{lobe}$ (km s$^{-1}$)',
                'FWHM_lobe':'FWHM$_\mathrm{lobe}$ (km s$^{-1}$)',
                'amp_lobe':'A$_\mathrm{lobe}$',
                'snr_R':'SNR ratio',
                'RVdrift':'RV drift (m s$^{-1}$)',
                'colcorrmin':'Min color coeff',
                'colcorrmax':'Max color coeff', 
                'colcorr450': 'Color coeff at 450nm' ,
                'colcorr550':'Color coeff at 550nm',
                'colcorr650':'Color coeff at 650nm',
                'glob_flux_sc':'Global flux scaling',
                'satur_check':'Saturation flag',
                'wig_p_0':'Wiggle p0 (A)', 'wig_p_1':'Wiggle p1',
                'wig_wref':'Wiggle w$_\mathrm{ref}$ (A)',
                'wig_a_0':'Wiggle a0 (flux)', 'wig_a_1':'Wiggle a1 (flux A$^{-1}$)', 'wig_a_2':'Wiggle a2 (flux A$^{-2}$)', 'wig_a_3':'Wiggle a3 (flux A$^{-3}$)', 'wig_a_4':'Wiggle a4 (flux A$^{-4}$)',                   
                'alt':'Tel. altitude ($^{\circ}$)',
                'az':'Tel. azimuth ($^{\circ}$)',
                'ha':r'H$_{\alpha}$ index',
                'na':'Na index',
                'ca':'Ca index',
                's':'S index',
                'rhk':'R$_{hk}$ index',
                'true_FWHM':'FWHM$_\mathrm{true}$ (km s$^{-1}$)',
                'true_ctrst':'Contrast$_\mathrm{true}$'                    
                }
        if data_mode=='DI':
            y_title_dic['rv'] = 'RV (km s$^{-1}$)'
            y_title_dic['rv_res'] = 'RV res. (m s$^{-1}$)'
        elif data_mode=='Intr':
            y_title_dic['rv'] = 'Surface RV (km s$^{-1}$)'
            y_title_dic['rv_res'] =  'Surface RV residuals (m s$^{-1}$)'           
        if prop_mode not in y_title_dic:y_title_dic[prop_mode]=prop_mode
            
        if plot_options['title']:
            plot_title_dic={}
            for key in y_title_dic:plot_title_dic[key] = y_title_dic[key]
            if data_mode=='DI':sub_key = 'DI'
            elif data_mode=='Intr':sub_key = 'intrinsic'
            plot_title_dic['rv'] = 'RV of '+sub_key+' CCFs'
            plot_title_dic['rv_pip'] = 'RV pip. of '+sub_key+' CCFs'
            plot_title_dic['rv_res'] = 'Residuals from RV of '+sub_key+' CCFs'
            plot_title_dic['rv_pip_res'] = 'RV pip. residuals of '+sub_key+' CCFs'
            plot_title_dic['FWHM'] = 'FWHM of '+sub_key+' CCFs'
            plot_title_dic['FWHM_pip'] = 'FWHM pip. of '+sub_key+' CCFs'
            plot_title_dic['amp'] = 'Amplitude of '+sub_key+' CCFs'
            plot_title_dic['ctrst'] = 'Contrast of '+sub_key+' CCFs'
            plot_title_dic['ctrst_pip'] = 'Contrast pip. of '+sub_key+' CCFs'
            plot_title_dic['rv_l2c'] = 'Lobes/core RV'
            plot_title_dic['amp_l2c'] = 'Lobes/core amplitude'
            plot_title_dic['FWHM_l2c'] = 'Lobes/core FWHM'
            plot_title_dic['area'] = 'Area under continuum'
            plot_title_dic['RV_lobe'] = 'Lobe RV of '+sub_key+' CCFs'
            plot_title_dic['FWHM_lobe'] = 'Lobe FWHM of '+sub_key+' CCFs'
            plot_title_dic['amp_lobe'] = 'Lobe amplitude of '+sub_key+' CCFs'
            plot_title_dic['true_FWHM'] = 'True FWHM of '+sub_key+' CCFs'
            plot_title_dic['true_ctrst'] = 'True contrast of '+sub_key+' CCFs'
            plt.title(plot_title_dic[prop_mode])

        #Invert horizontal axis
        if plot_options['retro_orbit']:x_range_loc=-np.array(x_range_loc)

        #Frame
        xmajor_int,xminor_int,xmajor_form=autom_tick_prop(x_range_loc[1]-x_range_loc[0])
        ymajor_int,yminor_int,ymajor_form=autom_tick_prop(y_range_loc[1]-y_range_loc[0])       
        custom_axis(plt,position=plot_options['margins'],x_range=x_range_loc,y_range=y_range_loc,
                    dir_x = 'out',dir_y = 'out',
    		        xmajor_int=xmajor_int,xminor_int=xminor_int,xmajor_form=xmajor_form,
                    ymajor_form=ymajor_form,ymajor_int=ymajor_int,yminor_int=yminor_int,                
                    x_title=x_title_dic[plot_options['prop_'+data_mode+'_absc']],y_title=y_title_dic[prop_mode],
                    font_size=plot_options['font_size'],xfont_size=plot_options['font_size'],yfont_size=plot_options['font_size'])
        if data_mode=='DI':sub_key = 'DI'
        elif data_mode=='Intr':sub_key = 'Intr'
        plt.savefig(path_loc+prop_mode+'_'+plot_options['prop_'+data_mode+'_absc']+'.'+plot_dic['prop_'+sub_key]) 
        plt.close()
    		
        return None   






    def sub_plot_propCCF_mcmc_PDFs(plot_options,plot_prop,plot_type):    
        path_loc = gen_dic['save_plot_dir']+plot_options['data_mode']+'_prop/MCMC/'
        if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  
                    
        #Plot for each instrument
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_options['visits_to_plot'].keys())):                            
            vis_list = np.intersect1d(list(data_dic[inst].keys()),plot_options['visits_to_plot'][inst])
            for ivis,vis in enumerate(vis_list): 
                
                #Identifying original or binned data
                orig_vis = vis.split('_bin')[0]
                if 'bin' in vis:data_type = plot_options['data_mode']+'bin'
                else:data_type = plot_options['data_mode']+'orig'
                
                if orig_vis in list(data_dic[plot_options['data_dic_idx']][inst].keys()):
    
                    #Common data
                    data_vis  = dataload_npz(gen_dic['save_data_dir']+data_type+'_prop/'+inst+'_'+vis)
    
                    #Exposures to plot
                    if (inst in plot_options['iexp_plot']) and (vis in plot_options['iexp_plot'][inst]) and (len(plot_options['iexp_plot'][inst][vis])>0):
                        iexp_eff = np.intersect1d(range(data_vis['n_exp']),plot_options['iexp_plot'][inst][vis])
                    else:iexp_eff = range(data_vis['n_exp'])
                    nexp_eff = len(iexp_eff)
    
                    #Create figure
                    for plot_prop in plot_options['plot_prop_list']:
                        plt.ioff() 
                        nsub_rows = int(np.ceil(nexp_eff/plot_options['nsub_col']))
                        fig, axes = plt.subplots(nsub_rows, plot_options['nsub_col'], figsize=plot_options['fig_size'])
                        fig.subplots_adjust(left=plot_options['margins'][0], bottom=plot_options['margins'][1], right=plot_options['margins'][2], top=plot_options['margins'][3],wspace=plot_options['wspace'] , hspace=plot_options['hspace']  )
                        nall = nsub_rows*plot_options['nsub_col']   #number of possible subplots
                        comp_ax = (nall>nexp_eff)   #all possible subplots are not filled
                        
                        #Process chosen exposures
                        for isub,iexp in enumerate(iexp_eff):
    
                            #Upload chains and properties
                            data_exp =np.load(gen_dic['save_data_dir']+data_type+'_prop/'+inst+'_'+vis+'_mcmc/iexp'+str(iexp)+'/merged_deriv_chains_walk'+str(plot_options['nwalkers'])+'_steps'+str(plot_options['nsteps'])+'.npz',allow_pickle=True)['data'].item() 
                            ipar = np_where1D(data_exp['var_par_list']==plot_prop)[0]
                            xchain = data_exp['merged_chain'][:,ipar]
                            
                            #Set row & column index of current exposure
                            irow = int(isub/plot_options['nsub_col'])
                            icol = isub%plot_options['nsub_col'] 
    
                            #Single/multiple subplots
                            if nexp_eff == 1:ax = axes
                            else:
                                if nsub_rows==1:ax = axes[icol]
                                else:ax = axes[irow, icol]
                            
                            #Horzontal range
                            if plot_prop in plot_options['x_range_all']:x_range_loc = plot_options['x_range_all'][plot_prop]
                            else:x_range_loc = [np.min(xchain),np.max(xchain)]
    
                            #Plot histogram of PDF
                            n, _, _ = ax.hist(xchain, bins=plot_options['bins_par'], range=x_range_loc, **{"color":"k","histtype":"step",'lw':0.8*plot_options['lw_plot']})
                            
                            #Vertical range
                            if plot_options['y_range'] is None:y_range_loc = [0,1.1 * np.max(n)]
                            else:y_range_loc = plot_options['y_range']                        
                    
                            #Plot best-fit value
                            ax.axvline(data_exp['med_parfinal'][ipar], color="#4682b4",lw=plot_options['lw_plot'])
                    
                            #Plot confidence intervals
                            if 'quant' in plot_options['plot_conf_mode']:
                                ax.axvline(data_exp['sig_parfinal_val'][0,ipar], ls="dashed", color='darkorange',lw=plot_options['lw_plot']) 
                                ax.axvline(data_exp['sig_parfinal_val'][1,ipar], ls="dashed", color='darkorange',lw=plot_options['lw_plot'])                                
                            if 'HDI' in plot_options['plot_conf_mode']:
                                for HDI_sub in data_exp['HDI_interv'][ipar]:
                                    ax.axvline(HDI_sub[0], ls="dashed", color='green',lw=plot_options['lw_plot']) 
                                    ax.axvline(HDI_sub[1], ls="dashed", color='green',lw=plot_options['lw_plot'])
                                    
                            #Exposures index
                            if plot_options['plot_expid']:ax.text(x_range_loc[0]+0.07*(x_range_loc[1]-x_range_loc[0]),y_range_loc[0]+0.8*(y_range_loc[1]-y_range_loc[0]),str(iexp),verticalalignment='bottom', horizontalalignment='center',fontsize=0.8*plot_options['font_size'],zorder=40,color='black') 
                    
                            #Set up the axes
                            ax.set_xlim(x_range_loc)
                            ax.set_ylim(y_range_loc)
                            ax.set_yticklabels([])
                            ax.yaxis.set_ticks_position('none') 
                            
                            #Interval between ticks	
                            if plot_prop not in plot_options['xmajor_int_all']:ax.xaxis.set_major_locator(MaxNLocator(5, prune="lower"))
                            else:ax.xaxis.set_major_locator(MultipleLocator(plot_options['xmajor_int_all'][plot_prop]))
                            if plot_prop in plot_options['xminor_int_all']:ax.xaxis.set_minor_locator(MultipleLocator(plot_options['xminor_int_all'][plot_prop]))
                            
                            #Add x ticks in last bottom row
                            if (plot_prop in plot_options['x_range_all']):
                                  if (irow==nsub_rows-1) or ((irow==nsub_rows-2) & (comp_ax and (icol>((nexp_eff-1)%plot_options['nsub_col'] )))):ax.tick_params('x',labelsize=plot_options['font_size'])
                                  else:ax.set_xticklabels([]) 
                            
                            #Add x ticks in all subplots
                            else:ax.tick_params('x',labelsize=plot_options['font_size'])
                                    
                            #Show x label and ticks in last row, or next-to-last row if last row is not fully filled
                            if (irow==nsub_rows-1) or ((irow==nsub_rows-2) & (comp_ax and (icol>((nexp_eff-1)%plot_options['nsub_col'] )))):
                                ax.set_xlabel(data_exp['var_par_names'][ipar],fontsize=plot_options['font_size'])
                                    
                        #--------------------------------------------------------
                        #Fill remaining space with empty axes
                        for isub_comp in range(isub+1,nall):
                            irow = int(isub_comp/plot_options['nsub_col'])
                            icol = isub_comp%plot_options['nsub_col'] 
                            if nsub_rows==1:axes[icol].axis('off')
                            else:axes[irow, icol].axis('off')
                            
                        plt.savefig(path_loc+inst+'_'+vis+'_'+plot_prop+'.'+plot_type)                        
                        plt.close()  

        return None  
    
    

    '''
    Plot distributions of line properties for mask generation
    '''
    def sub_dist_CCFmasks(dist_info,plot_info,plot_options,ax_loc,var,var_thresh,var_thresh_test,prop_type,y_title,range_loc,ax_name,mode='2D'):
        if plot_options[ax_name+'_range_hist'] is not None:range_hist = plot_options[ax_name+'_range_hist']
        else:range_hist = None
        if dist_info =='hist':
            if ax_name=='x':
                orientation="vertical"
                if mode=='2D':
                    ax_loc.set_yticklabels([])
                    ax_loc.yaxis.set_ticks_position('none') 
            elif ax_name=='y':
                orientation="horizontal"
                ax_loc.set_xticklabels([])
                ax_loc.xaxis.set_ticks_position('none')                 
            var_dist, _, _ = ax_loc.hist(var, bins=plot_options[ax_name+'_bins_par'], range=range_loc,orientation=orientation, **{"color":"k","histtype":"step",'lw':plot_options['lw_plot'],"log":plot_options[ax_name+'_log_hist']})
            if plot_options[ax_name+'_log_hist']:
                if plot_options[ax_name+'_range_hist'] is None:range_hist = [0.9*np.min(var_dist[var_dist>0]),1.1*np.max(var_dist)]
                log_txt=' (log)'
            else:
                if plot_options[ax_name+'_range_hist'] is None:range_hist = [0,1.1 * np.max(var_dist)]               
                log_txt = ''
            ax_loc.set_ylim(range_hist)
            if ax_name=='x':
                ax_loc.axvline(var_thresh, color='magenta',lw=plot_options['lw_plot'],linestyle='--')
                if prop_type=='ld':ax_loc.axvline(plot_info['linedepth_cont_max'], color='magenta',lw=plot_options['lw_plot'],linestyle='--')
                if var_thresh_test is not None:
                    ax_loc.axvline(var_thresh_test, color='gold',lw=plot_options['lw_plot'],linestyle='--')
                    if prop_type=='ld':ax_loc.axvline(plot_options['linedepth_cont_max'], color='gold',lw=plot_options['lw_plot'],linestyle='--')   
            elif ax_name=='y':
                ax_loc.axhline(var_thresh, color='magenta',lw=plot_options['lw_plot'],linestyle='--')
                if prop_type=='ld':ax_loc.axhline(plot_info['linedepth_cont_max'], color='magenta',lw=plot_options['lw_plot'],linestyle='--')
                if var_thresh_test is not None:
                    ax_loc.axhline(var_thresh_test, color='gold',lw=plot_options['lw_plot'],linestyle='--')
                    if prop_type=='ld':ax_loc.axhline(plot_options['linedepth_cont_max'], color='gold',lw=plot_options['lw_plot'],linestyle='--')   
                 
        elif dist_info =='cum_w':
            idx_sort = np.argsort(var)
            weight_rv_plt = plot_info['weight_rv_'+prop_type][idx_sort]
            cumsum = np.nancumsum(weight_rv_plt)
            var_dist = (cumsum-np.min(cumsum))/(np.max(cumsum)-np.min(cumsum))   
            x_var_dist = var[idx_sort]
            if plot_options[ax_name+'_range_hist'] is None:range_hist = [0,1]
            if ax_name=='x':
                x_loc,y_loc = x_var_dist, var_dist  
                ax_loc.set_ylim(range_hist)
            elif ax_name=='y':
                x_loc,y_loc = var_dist,x_var_dist  
                ax_loc.set_xlim(range_hist)
            ax_loc.plot(x_loc,y_loc,color='black',linestyle='-',lw=plot_options['lw_plot'],drawstyle='steps-mid',zorder=10) 
            log_txt = ''
            y_var_dist_loc = var_dist[closest(x_var_dist,var_thresh)]
            if ax_name=='x':x_loc,y_loc = [var_thresh,var_thresh] ,[0.,y_var_dist_loc]
            elif ax_name=='y':x_loc,y_loc = [0.,y_var_dist_loc] ,[var_thresh,var_thresh]            
            ax_loc.plot(x_loc,y_loc,color='magenta',lw=plot_options['lw_plot'],linestyle='--')
            if ax_name=='x':x_loc,y_loc = [range_loc[0],var_thresh],[y_var_dist_loc,y_var_dist_loc] 
            elif ax_name=='y':x_loc,y_loc = [y_var_dist_loc,y_var_dist_loc],[range_loc[0],var_thresh] 
            ax_loc.plot(x_loc,y_loc,color='magenta',lw=plot_options['lw_plot'],linestyle='--')
            if prop_type=='ld':    
                y_var_dist_loc = var_dist[closest(x_var_dist,plot_info['linedepth_cont_max'])]
                if ax_name=='x':x_loc,y_loc = [plot_info['linedepth_cont_max'],plot_info['linedepth_cont_max']],[0.,y_var_dist_loc]
                elif ax_name=='y':x_loc,y_loc = [0.,y_var_dist_loc],[plot_info['linedepth_cont_max'],plot_info['linedepth_cont_max']]
                ax_loc.plot(x_loc,y_loc ,color='magenta',lw=plot_options['lw_plot'],linestyle='--')
                if ax_name=='x':x_loc,y_loc = [range_loc[0],plot_info['linedepth_cont_max']],[y_var_dist_loc,y_var_dist_loc]
                elif ax_name=='y':x_loc,y_loc = [y_var_dist_loc,y_var_dist_loc] ,[range_loc[0],plot_info['linedepth_cont_max']]     
                ax_loc.plot(x_loc,y_loc ,color='magenta',lw=plot_options['lw_plot'],linestyle='--')
            if var_thresh_test is not None:
                y_var_dist_loc = var_dist[closest(x_var_dist,var_thresh_test)]
                if ax_name=='x':x_loc,y_loc = [var_thresh_test,var_thresh_test],[0.,y_var_dist_loc]
                elif ax_name=='y':x_loc,y_loc = [0.,y_var_dist_loc],[var_thresh_test,var_thresh_test]  
                ax_loc.plot( x_loc,y_loc,color='gold',lw=plot_options['lw_plot'],linestyle='--')
                if ax_name=='x':x_loc,y_loc = [range_loc[0],var_thresh_test],[y_var_dist_loc,y_var_dist_loc]
                elif ax_name=='y':x_loc,y_loc = [y_var_dist_loc,y_var_dist_loc],[range_loc[0],var_thresh_test]                  
                ax_loc.plot( x_loc,y_loc,color='gold',lw=plot_options['lw_plot'],linestyle='--')
            if (prop_type=='ld') and ('linedepth_cont_max' in plot_options):     
                y_var_dist_loc = var_dist[closest(x_var_dist,plot_options['linedepth_cont_max'])]  
                if ax_name=='x':x_loc,y_loc = [plot_options['linedepth_cont_max'],plot_options['linedepth_cont_max']],[0.,y_var_dist_loc] 
                elif ax_name=='y':x_loc,y_loc = [0.,y_var_dist_loc],[plot_options['linedepth_cont_max'],plot_options['linedepth_cont_max']]                       
                ax_loc.plot(x_loc,y_loc,color='gold',lw=plot_options['lw_plot'],linestyle='--')
                if ax_name=='x':x_loc,y_loc = [range_loc[0],plot_options['linedepth_cont_max']],[y_var_dist_loc,y_var_dist_loc] 
                elif ax_name=='y':x_loc,y_loc = [y_var_dist_loc,y_var_dist_loc] , [range_loc[0],plot_options['linedepth_cont_max']]                     
                ax_loc.plot(x_loc,y_loc,color='gold',lw=plot_options['lw_plot'],linestyle='--')

        #Set up the axes
        if ax_name=='x':
            ax_loc.set_ylabel(y_title+log_txt,fontsize=plot_options['font_size'])
            ax_loc.set_xlim(range_loc)
            if mode=='2D':
                ax_loc.set_xticklabels([])
                ax_loc.xaxis.set_ticks_position('none') 
        elif ax_name=='y':
            ax_loc.set_xlabel(y_title+log_txt,fontsize=plot_options['font_size'])
            ax_loc.set_ylim(range_loc)
            ax_loc.set_yticklabels([])
            ax_loc.yaxis.set_ticks_position('none')
            
        return range_hist
    
    def dist1D_stlines_CCFmasks(dist_info,plot_options,key_plot,plot_ext):
        data_type_gen=key_plot.split('mask')[0]
        prop_type = key_plot.split('mask_')[1]
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_options['visits_to_plot'].keys())): 
            vis_det=list(data_dic[inst].keys()) if data_dic[inst]['n_visits_inst']==1 else 'binned'
            data_paths = 'CCF_masks_'+data_type_gen+'/'+gen_dic['add_txt_path'][data_type_gen]+'/'+inst+'_'+vis_det+'/'

            #Create directory if required
            path_loc = gen_dic['save_plot_dir']+data_paths
            if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)                 
            plt.ioff()        
            fig = plt.figure(figsize=plot_options['fig_size'])
            fig.subplots_adjust(left=plot_options['margins'][0],bottom=plot_options['margins'][1],right=plot_options['margins'][2],top=plot_options['margins'][3]) 
            ax = plt.gca()

            #Retrieve plot dictionary
            plot_info = dataload_npz(gen_dic['save_data_dir']+data_paths+'Plot_info')

            #Property before selection 
            x_var_name = {
                'RVdisp':'disp_RV_lines',
                'RVdev_fit':'abs_RVdev_fit',
                'tellcont':'rel_contam',
                'tellcont_final':'rel_contam_final'}[prop_type]            
            var = plot_info[x_var_name]
            
            #Plot frame 
            if plot_options['x_range'] is not None:x_range_loc=plot_options['x_range'] 
            else:x_range_loc = np.array([np.min(var),np.max(var)])
            dx_range=x_range_loc[1]-x_range_loc[0]
            if plot_options['x_range'] is None:
                if prop_type not in ['RVdev_fit']:x_range_loc[0]-=0.05*dx_range
                x_range_loc[1]+=0.05*dx_range
                dx_range=x_range_loc[1]-x_range_loc[0]

            #Thresholds
            x_thresh_name = {
                'RVdisp':'RVdisp_max',
                'RVdev_fit':'abs_RVdev_fit_max',
                'tellcont':'tell_star_depthR_max',
                'tellcont_final':'tell_star_depthR_max_final'}[prop_type]   
            if x_thresh_name in plot_options:var_thresh_test = plot_options[x_thresh_name]
            else:var_thresh_test = None

            #Histogram 
            if dist_info =='hist':y_title='Occurences'
            elif dist_info =='cum_w':
                y_title='Cumulated weights'
                plot_options['x_log_hist'] = False
            y_range_loc = sub_dist_CCFmasks(dist_info,plot_info,plot_options,ax,var,plot_info[x_thresh_name],var_thresh_test,prop_type,y_title,x_range_loc,'x',mode='1D')
            if plot_options['x_log_hist']:dy_range=np.log10(y_range_loc[1])-np.log10(y_range_loc[0])
            else:dy_range=y_range_loc[1]-y_range_loc[0]

            #Print number of lines before selection, after pipeline selection, after test selection
            n_all = len(var)
            cond_sel = (var<plot_info[x_thresh_name])
            n_sel = np.sum(cond_sel)
            xpos_txt = x_range_loc[1]-0.5*dx_range
            ytxt = 10**(np.log10(y_range_loc[0])+0.8*dy_range)     if plot_options['x_log_hist'] else y_range_loc[0]+0.8*dy_range
            plt.text(xpos_txt,ytxt,'All lines = '+str(n_all),verticalalignment='bottom', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=4,color='black') 
            ytxt = 10**(np.log10(y_range_loc[0])+0.7*dy_range)     if plot_options['x_log_hist'] else y_range_loc[0]+0.7*dy_range
            plt.text(xpos_txt,ytxt,'Selected = '+str(n_sel),verticalalignment='bottom', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=4,color='magenta') 
            if x_thresh_name in plot_options:
                cond_sel_test = (var<plot_options[x_thresh_name] )  
                n_test = np.sum(cond_sel_test)  
                ytxt = 10**(np.log10(y_range_loc[0])+0.6*dy_range)     if plot_options['x_log_hist'] else y_range_loc[0]+0.6*dy_range
                plt.text(xpos_txt,ytxt,'Trial = '+str(n_test),verticalalignment='bottom', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=4,color='gold') 

            #Frame
            x_lab_name = {
                'RVdisp':r'RV dispersion (m/s)',
                'RVdev_fit':r'|RV$_{\rm fit}$| deviation',
                'tellcont':r'Telluric/Line depth',
                'tellcont_final':r'Telluric/Line depth'}[prop_type]
            ax.set_xlabel(x_lab_name,fontsize=plot_options['font_size'])
            ax.tick_params('x',labelsize=plot_options['font_size'])
            ax.tick_params('y',labelsize=plot_options['font_size'])
            if prop_type=='RVdisp':prop_type+='1D'
            plt.savefig(path_loc+prop_type+'_'+dist_info+'.'+plot_ext)                       
            plt.close() 
          
        return None



    def dist2D_stlines_CCFmasks(dist_info,plot_options,key_plot,plot_ext):
        data_type_gen=key_plot.split('mask')[0]
        prop_type = key_plot.split('mask_')[1]
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_options['visits_to_plot'].keys())): 
            vis_det=list(data_dic[inst].keys()) if data_dic[inst]['n_visits_inst']==1 else 'binned'
            data_paths = 'CCF_masks_'+data_type_gen+'/'+gen_dic['add_txt_path'][data_type_gen]+'/'+inst+'_'+vis_det+'/'
    
            #Create directory if required
            path_loc = gen_dic['save_plot_dir']+data_paths
            if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)                 
            plt.ioff()        
            fig, axes = plt.subplots(2,2,figsize=plot_options['fig_size'])
            fig.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.9,wspace=plot_options['wspace'] , hspace=plot_options['hspace']  )
    
            #Retrieve plot dictionary
            plot_info = dataload_npz(gen_dic['save_data_dir']+data_paths+'Plot_info')
    
            #Properties of lines before selection 
            x_var_name = {
                'ld':'line_depth_cont',
                'ld_lw':'log10_min_line_width',
                'morphasym':'diff_continuum_rel',
                'morphshape':'width_kms',
                'RVdisp':'abs_av_RV_lines'}[prop_type]
            y_var_name = {
                'ld':'line_depth',
                'ld_lw':'log10_min_line_depth',
                'morphasym':'abs_asym_ddflux_norm',
                'morphshape':'diff_depth',
                'RVdisp':'disp_err_RV_lines'}[prop_type]
            x_var = plot_info[x_var_name]
            y_var = plot_info[y_var_name] 
  
            #Plot frame 
            if plot_options['x_range'] is not None:x_range_loc=plot_options['x_range'] 
            else:x_range_loc = np.array([np.min(x_var),np.max(x_var)])
            if plot_options['y_range'] is not None:y_range_loc=plot_options['y_range'] 
            else:y_range_loc = np.array([np.min(y_var),np.max(y_var)])
            dx_range=x_range_loc[1]-x_range_loc[0]
            if plot_options['x_range'] is None:
                x_range_loc[0]-=0.05*dx_range
                x_range_loc[1]+=0.05*dx_range
                dx_range=x_range_loc[1]-x_range_loc[0]
            dy_range=y_range_loc[1]-y_range_loc[0]
            if plot_options['y_range'] is None:
                y_range_loc[0]-=0.05*dy_range
                y_range_loc[1]+=0.05*dy_range
                dy_range=y_range_loc[1]-y_range_loc[0]
         
            #Thresholds
            x_thresh_name = {
                'ld':'linedepth_cont_min',
                'ld_lw':'line_width_logmin',
                'morphasym':'diff_cont_rel_max',
                'morphshape':'width_max',
                'RVdisp':'absRV_max'}[prop_type]
            y_thresh_name = {
                'ld':'linedepth_min',
                'ld_lw':'line_depth_logmin',
                'morphasym':'asym_ddflux_max',
                'morphshape':'diff_depth_min',
                'RVdisp':'RVdisp2err_max'}[prop_type]
    
            #Plot pipeline selection thresholds    
            x_var_thresh = plot_info[x_thresh_name]      
            y_var_thresh = plot_info[y_thresh_name]        
            axes[1,0].plot(x_range_loc,np.repeat(y_var_thresh,2),color='magenta',linestyle='--',lw=plot_options['lw_plot'],zorder=10)               
            axes[1,0].plot(np.repeat(x_var_thresh,2),y_range_loc,color='magenta',linestyle='--',lw=plot_options['lw_plot'],zorder=10) 
            if prop_type=='ld':             
                axes[1,0].plot(np.repeat(plot_info['linedepth_cont_max'] ,2),y_range_loc,color='magenta',linestyle='--',lw=plot_options['lw_plot'],zorder=10)
                axes[1,0].plot(x_range_loc,np.repeat(plot_info['linedepth_max'] ,2),color='magenta',linestyle='--',lw=plot_options['lw_plot'],zorder=10) 
                if plot_info['linedepth_contdepth'] is not None: 
                    linedepth_rel = plot_info['linedepth_contdepth'][0]*x_var+plot_info['linedepth_contdepth'][1]
                    axes[1,0].plot(x_range_loc,plot_info['linedepth_contdepth'][0]*x_range_loc+plot_info['linedepth_contdepth'][1],color='magenta',linestyle='--',lw=plot_options['lw_plot'],zorder=10)                  
                else:linedepth_rel=10.
        
            #Plot test selection threshold
            thresh_test = 1
            x_var_thresh_test = None
            y_var_thresh_test = None
            if x_thresh_name in plot_options:
                x_var_thresh_test = plot_options[x_thresh_name]                 
                axes[1,0].plot(np.repeat(x_var_thresh_test,2),y_range_loc,color='gold',linestyle='--',lw=plot_options['lw_plot'],zorder=10) 
                if prop_type=='ld':             
                    axes[1,0].plot(np.repeat(plot_options['linedepth_cont_max'] ,2),y_range_loc,color='gold',linestyle='--',lw=plot_options['lw_plot'],zorder=10) 
                    axes[1,0].plot(x_range_loc,np.repeat(plot_options['linedepth_max'] ,2),color='gold',linestyle='--',lw=plot_options['lw_plot'],zorder=10) 
                    if plot_options['linedepth_contdepth'] is not None: 
                        linedepth_rel_test = plot_options['linedepth_contdepth'][0]*x_var+plot_options['linedepth_contdepth'][1]
                        axes[1,0].plot(x_range_loc,plot_options['linedepth_contdepth'][0]*x_range_loc+plot_options['linedepth_contdepth'][1],color='gold',linestyle='--',lw=plot_options['lw_plot'],zorder=10) 
                    else:linedepth_rel_test = 10.
            else:thresh_test = None
            if y_thresh_name in plot_options:
                y_var_thresh_test = plot_options[y_thresh_name]
                axes[1,0].plot(x_range_loc,np.repeat(y_var_thresh_test,2),color='gold',linestyle='--',lw=plot_options['lw_plot'],zorder=10)  
            else:thresh_test = None
     
            #Plot line properties before/after pipeline selection
            if prop_type in ['ld']:
                cond_sel = (x_var>x_var_thresh)&(x_var<plot_info['linedepth_cont_max'])&(y_var>y_var_thresh)&(y_var<plot_info['linedepth_max'])&(y_var<linedepth_rel)
                if (thresh_test is not None):cond_sel_test = (x_var>x_var_thresh_test)&(x_var<plot_options['linedepth_cont_max'])&(y_var>y_var_thresh_test)&(y_var<plot_options['linedepth_cont_min'])&(y_var<linedepth_rel_test)
            elif prop_type in ['ld_lw']:
                cond_sel = (x_var>x_var_thresh)&(y_var>y_var_thresh)
                if (thresh_test is not None):cond_sel_test = (x_var>x_var_thresh_test)&(y_var>y_var_thresh_test)
            elif prop_type in ['morphasym','RVdisp']:
                cond_sel = (x_var<x_var_thresh)&(y_var<y_var_thresh)
                if (thresh_test is not None):cond_sel_test = (x_var<x_var_thresh_test)&(y_var<y_var_thresh_test)   
            elif prop_type in ['morphshape']:                
                cond_sel = (x_var<x_var_thresh)&(y_var>y_var_thresh)
                if (thresh_test is not None):cond_sel_test = (x_var<x_var_thresh_test)&(y_var>y_var_thresh_test)                 
            axes[1,0].plot(x_var[~cond_sel],y_var[~cond_sel],markersize=plot_options['markersize'],color='red',markeredgecolor='white',zorder=0,marker='o',ls='',markeredgewidth=0.5,rasterized = plot_options['rasterized'])
            axes[1,0].plot(x_var[cond_sel],y_var[cond_sel],markersize=plot_options['markersize'],color='limegreen',markeredgecolor='white',zorder=0,marker='o',ls='',markeredgewidth=0.5,rasterized = plot_options['rasterized'])
    
            #Print number of lines before selection, aftr pipeline selection, after test selection
            n_all = len(x_var)
            n_sel = np.sum(cond_sel)
            axes[1,0].text(x_range_loc[1]+0.2*dx_range,y_range_loc[1]+0.5*dy_range,'All lines = '+str(n_all),verticalalignment='bottom', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=4,color='black') 
            axes[1,0].text(x_range_loc[1]+0.2*dx_range,y_range_loc[1]+0.3*dy_range,'Selected = '+str(n_sel),verticalalignment='bottom', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=4,color='magenta') 
            if (thresh_test is not None):
                n_test = np.sum(cond_sel_test)             
                axes[1,0].text(x_range_loc[1]+0.2*dx_range,y_range_loc[1]+0.1*dy_range,'Trial = '+str(n_test),verticalalignment='bottom', horizontalalignment='left',fontsize=plot_options['font_size_txt'],zorder=4,color='gold') 
            
            #Set up the axes
            axes[1,0].set_xlim(x_range_loc)
            axes[1,0].set_ylim(y_range_loc)
            xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
            ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)
            axes[1,0].xaxis.set_major_locator(MultipleLocator(xmajor_int))
            axes[1,0].yaxis.set_major_locator(MultipleLocator(ymajor_int))
            x_lab_name = {
                'ld':'Continuum depth',
                'ld_lw':r'$log_{10}$(Width)',
                'morphasym':'Continuum difference',
                'morphshape':'Width (km/s)',
                'RVdisp':r'$|\Delta$RV|'}[prop_type]
            y_lab_name = {
                'ld':'Depth',
                'ld_lw':r'$log_{10}$(Depth)',
                'morphasym':'Continuum asymmetry',
                'morphshape':'Depth',
                'RVdisp':r'$\sigma_{RV}/<e_{RV}>$'}[prop_type]            
            axes[1,0].set_xlabel(x_lab_name,fontsize=plot_options['font_size'])
            axes[1,0].set_ylabel(y_lab_name,fontsize=plot_options['font_size'])
    
            #------------------------------------------------------------
            if dist_info =='hist':y_title='Occurences'
            elif dist_info =='cum_w':y_title='Cumulated weights'
            #------------------------------------------------------------
            
            #Histogram of X
            sub_dist_CCFmasks(dist_info,plot_info,plot_options,axes[0,0],x_var,x_var_thresh,x_var_thresh_test,prop_type,y_title,x_range_loc,'x')

            #------------------------------------------------------------

            #Histograms of Y
            sub_dist_CCFmasks(dist_info,plot_info,plot_options,axes[1,1],y_var,y_var_thresh,y_var_thresh_test,prop_type,y_title,y_range_loc,'y')            

            #--------------------------------------------------------
            #Fill remaining space with empty ax
            axes[0,1].axis('off')
            plt.savefig(path_loc+prop_type+'_'+dist_info+'.'+plot_ext)                        
            plt.close() 

        return None



























































    





















    
    ################################################################################################################    
    #%% Weighing master
    ################################################################################################################  
    if 'DImast' in plot_settings:
        key_plot = 'DImast'

        print('-----------------------------------')
        print('+ Weighing master')
        sub_plot_prof(plot_settings[key_plot],key_plot,plot_dic[key_plot])   





    ##################################################################################################
    #%% Global spectral corrections
    ################################################################################################## 

    ################################################################################################################    
    #%%% Instrumental calibration estimates
    ################################################################################################################ 
    if ('gcal' in plot_settings):
        key_plot = 'gcal'
        plot_set_key = plot_settings[key_plot]

        print('-----------------------------------')
        print('+ Instrumental calibration estimates')     
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())): 

            #Mean calibration profile
            #    - as calculated from input data, before any shifts
            if plot_set_key['mean_gdet'] and (not plot_set_key['norm_exp']):
                mean_gdet_func = (np.load(gen_dic['save_data_dir']+'Processed_data/Calibration/'+inst+'_mean_gdet.npz', allow_pickle=True)['data'].item())['func']  

            for vis in np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst]): 
                data_vis = data_dic[inst][vis]

                #Create directory if required
                path_loc = gen_dic['save_plot_dir']+'General/gcal/'+inst+'/'+vis+'/'
                if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  

                #Orders to process
                if ('orders_to_plot' in plot_set_key) and (len(plot_set_key['orders_to_plot'])>0):order_list = plot_set_key['orders_to_plot']
                else:order_list=range(data_dic[inst]['nord']) 
                nord_list = len(order_list)
            
                #Exposures to plot
                if ('iexp_plot' in plot_set_key) and (inst in plot_set_key['iexp_plot']) and (len(plot_set_key['iexp_plot'][inst][vis])>0):iexp_plot = plot_set_key['iexp_plot'][inst][vis]
                else:iexp_plot=range(data_dic[inst][vis]['n_in_visit'])
         
                #Visit color
                nexp_plot=len(iexp_plot)
                cmap = plt.get_cmap('jet') 
                col_visit=np.array([cmap(0)]) if nexp_plot==1 else cmap( np.arange(nexp_plot)/(nexp_plot-1.))

                #Upload data
                if ('spec' in data_dic['DI']['type'][inst]):
                    gdet_cen_binned_all = np.empty([nord_list,nexp_plot],dtype=object)
                    gdet_meas_binned_all =  np.empty([nord_list,nexp_plot],dtype=object)
                    cond_fit_all =  np.empty([nord_list,nexp_plot],dtype=object)
                    wav_trans_all =  np.empty([2,nord_list,nexp_plot],dtype=object)
                gdet_cen_binned_mean =  np.empty([nord_list,nexp_plot],dtype=float)
                gdet_meas_binned_mean =  np.empty([nord_list,nexp_plot],dtype=float)                
                cen_bins_all =  np.empty([nord_list,nexp_plot],dtype=object)
                gdet_bins_all =   np.empty([nord_list,nexp_plot],dtype=object)      
                cen_bins_mean =  np.empty([nord_list,nexp_plot],dtype=float)
                gdet_bins_mean =  np.empty([nord_list,nexp_plot],dtype=float)                         
                for isub_iexp,iexp in enumerate(iexp_plot):
                    data_load = np.load(data_vis['cal_data_paths']+str(iexp)+'.npz', allow_pickle=True)['data'].item()  
                    data_exp = np.load(gen_dic['save_data_dir']+'Processed_data/'+inst+'_'+vis+'_'+str(iexp)+'.npz', allow_pickle=True)['data'].item() 
                    cond_def_exp =  data_exp['cond_def']
                    for isub_ord,iord in enumerate(order_list):
                        if ('spec' in data_dic['DI']['type'][inst]):
                            gdet_cen_binned_all[isub_ord,isub_iexp] = data_load['wav_bin_all'][iord]
                            gdet_meas_binned_all[isub_ord,isub_iexp] = data_load['gdet_bin_all'][iord]
                            cond_fit_all[isub_ord,isub_iexp] = data_load['cond_fit_all'][iord]
                            wav_trans_all[:,isub_ord,isub_iexp] = data_load['wav_trans_all'][:,iord]
                        if len(cond_def_exp[iord]):
                            cen_bins_all[isub_ord,isub_iexp] = data_exp['cen_bins'][iord][cond_def_exp[iord]]
                            gdet_bins_all[isub_ord,isub_iexp] = cal_piecewise_func(data_load['gdet_inputs'][iord]['par'],cen_bins_all[isub_ord,isub_iexp],args=data_load['gdet_inputs'][iord]['args'])    

                        #Mean calibration per order (from measurements)
                        if (plot_dic[key_plot]!=''):
                            if ('spec' in data_dic['DI']['type'][inst]):
                                gdet_cen_binned_mean[isub_ord,isub_iexp] = np.mean(gdet_cen_binned_all[isub_ord,isub_iexp][cond_fit_all[isub_ord,isub_iexp]])
                                gdet_meas_binned_mean[isub_ord,isub_iexp] = np.mean(gdet_meas_binned_all[isub_ord,isub_iexp][cond_fit_all[isub_ord,isub_iexp]])
                                cen_bins_mean[isub_ord,isub_iexp] = gdet_cen_binned_mean[isub_ord,isub_iexp]
                            else:
                                cen_bins_mean[isub_ord,isub_iexp] = gen_dic['wav_ord_inst'][inst][iord]
                            gdet_bins_mean[isub_ord,isub_iexp] = np.mean(cal_piecewise_func(data_load['gdet_inputs'][iord]['par'],data_load['wav_bin_all'][iord],args=data_load['gdet_inputs'][iord]['args']))   
                                
                #Mean calibration per order (from model, for normalization)
                if plot_set_key['norm_exp']:norm_gdet = gdet_bins_mean
                else:norm_gdet=np.ones([nord_list,nexp_plot])
 
                #---------------------------------------------------------------------------------------------------------------
                    
                #Plot all orders for all exposures
                if (plot_dic[key_plot]!=''):
                    plt.ioff()        
                    fig = plt.figure(figsize=plot_set_key['fig_size'])
   
                    #Vertical range
                    y_min=1e100
                    y_max=-1e100

                    #Mean calibration over each order, for all orders and all exposures 
                    for isub_iexp in range(len(iexp_plot)):
                        if ('spec' in data_dic['DI']['type'][inst]):plt.plot(gdet_cen_binned_mean[:,isub_iexp],gdet_meas_binned_mean[:,isub_iexp],linestyle='',marker='o',markerfacecolor=col_visit[isub_iexp],markeredgecolor=col_visit[isub_iexp],markersize=plot_set_key['markersize'],rasterized=plot_set_key['rasterized'],color = col_visit[isub_iexp])
                        plt.plot(cen_bins_mean[:,isub_iexp],gdet_bins_mean[:,isub_iexp],linestyle='-',marker='.',markerfacecolor=col_visit[isub_iexp],markeredgecolor=col_visit[isub_iexp],markersize=plot_set_key['markersize'],rasterized=plot_set_key['rasterized'],color = col_visit[isub_iexp])            
                    x_range_loc = plot_set_key['x_range'] if plot_set_key['x_range'] is not None else [np.nanmin(cen_bins_mean),np.nanmax(cen_bins_mean)] 
                    y_range_loc = plot_set_key['y_range'] if plot_set_key['y_range'] is not None else [np.nanmin(gdet_bins_mean),np.nanmax(gdet_bins_mean)] 
                   
                    #Plot frame  
                    if plot_set_key['title']:plt.title('Calibration for visit '+vis+' in '+inst)
                    dx_range=x_range_loc[1]-x_range_loc[0]
                    dy_range=y_range_loc[1]-y_range_loc[0]
                    xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                    ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)                    
                    custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,dir_y='out', 
                                xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,
                                xmajor_form=xmajor_form,ymajor_form=ymajor_form,hide_axis=plot_set_key['hide_axis'],
                                x_title='Wavelength (A)',y_title=r'10$^{-5}$ Calibration',
                                font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                    plt.savefig(path_loc+'Global'+'.'+plot_dic[key_plot],transparent=plot_set_key['transparent']) 
                    plt.close()                     

                #---------------------------------------------------------------------------------------------------------------

                #Plot current order for all exposures
                if (plot_dic['gcal_ord']!='') and ('spec' in data_dic['DI']['type'][inst]):
                    for isub_ord,iord in enumerate(order_list):
                        plt.ioff()        
                        fig = plt.figure(figsize=plot_set_key['fig_size']) #,dpi=10)
    
                        #Calibration estimate (measured and modelled)
                        y_min=1e100
                        y_max=-1e100
                        for isub_iexp in range(len(iexp_plot)):
                            x_range_loc = plot_set_key['x_range_ord'] if plot_set_key['x_range_ord'] is not None else [np.nanmin(cen_bins_all[isub_ord,isub_iexp]),np.nanmax(cen_bins_all[isub_ord,isub_iexp])] 
                                
                            #Mesured calibration estimate
                            #    - all calculated bins are shown (ie, with positive integrated flux) are shown
                            #    - bins used for the fit are filled
                            if len(gdet_meas_binned_all[isub_ord,isub_iexp])>0:
                                cond_fit_ord = cond_fit_all[isub_ord,isub_iexp]
                                plt.plot(gdet_cen_binned_all[isub_ord,isub_iexp][cond_fit_ord],gdet_meas_binned_all[isub_ord,isub_iexp][cond_fit_ord]/norm_gdet[isub_ord,isub_iexp],linestyle='',marker='o',markerfacecolor=col_visit[isub_iexp],markeredgecolor=col_visit[isub_iexp],markersize=plot_set_key['markersize'],zorder=0,rasterized=plot_set_key['rasterized'])
                                plt.plot(gdet_cen_binned_all[isub_ord,isub_iexp][~cond_fit_ord],gdet_meas_binned_all[isub_ord,isub_iexp][~cond_fit_ord]/norm_gdet[isub_ord,isub_iexp],linestyle='',marker='o',markerfacecolor='None',markeredgecolor=col_visit[isub_iexp],markersize=plot_set_key['markersize'],zorder=0,rasterized=plot_set_key['rasterized'])
                                if np.sum(cond_fit_ord):
                                    y_min=np.min([np.nanmin(gdet_meas_binned_all[isub_ord,isub_iexp][cond_fit_ord]/norm_gdet[isub_ord,isub_iexp]),y_min])
                                    y_max=np.max([np.nanmax(gdet_meas_binned_all[isub_ord,isub_iexp][cond_fit_ord]/norm_gdet[isub_ord,isub_iexp]),y_max]) 
                               
                            #Best-fit calculated over the defined spectral table
                            if plot_set_key['plot_best_exp'] and (gdet_bins_all[isub_ord,isub_iexp] is not None):
                                plt.plot(cen_bins_all[isub_ord,isub_iexp],gdet_bins_all[isub_ord,isub_iexp]/norm_gdet[isub_ord,isub_iexp],linestyle='-',color=col_visit[isub_iexp],lw=0.5,rasterized=plot_set_key['rasterized'],zorder=0,alpha=0.5) 
                                y_min=np.min([np.min(gdet_bins_all[isub_ord,isub_iexp]/norm_gdet[isub_ord,isub_iexp]),y_min])
                                y_max=np.max([np.max(gdet_bins_all[isub_ord,isub_iexp]/norm_gdet[isub_ord,isub_iexp]),y_max]) 
                            
                        #Mean calibration profile over all exposures
                        #    - shown on table of first exposure, as calculated from input data
                        #    - this is the profile used to rescale homogeneously all exposures from flux to count values
                        if plot_set_key['mean_gdet'] and (not plot_set_key['norm_exp']):
                            plt.plot(cen_bins_all[isub_ord,0],mean_gdet_func[iord](cen_bins_all[isub_ord,0])*1e-3,linestyle='--',color='black',lw=1,rasterized=plot_set_key['rasterized'],zorder=10) 

                        #Transitions of polynomials
                        y_range_loc=plot_set_key['y_range_ord'] if plot_set_key['y_range_ord'] is not None else np.array([y_min,y_max])
                        for isub_iexp in range(len(iexp_plot)):
                            plt.plot([wav_trans_all[0,isub_ord,isub_iexp],wav_trans_all[0,isub_ord,isub_iexp]],y_range_loc,linestyle='--',color=col_visit[isub_iexp],lw=0.5,rasterized=plot_set_key['rasterized'])
                            plt.plot([wav_trans_all[1,isub_ord,isub_iexp],wav_trans_all[1,isub_ord,isub_iexp]],y_range_loc,linestyle='--',color=col_visit[isub_iexp],lw=0.5,rasterized=plot_set_key['rasterized'])
                                                        
                        #Plot frame                          
                        if plot_set_key['title']:plt.title('Calibration for visit '+vis+' in '+inst+' (order '+str(iord)+')')
                        dx_range=x_range_loc[1]-x_range_loc[0]
                        dy_range=y_range_loc[1]-y_range_loc[0]
                        xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                        ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)
                        custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,dir_y='out', 
                                    xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,
                                    xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                                    x_title='Wavelength (A)',y_title=r'10$^{-3}$ Calibration',hide_axis=plot_set_key['hide_axis'],
                                    font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                        plt.savefig(path_loc+'ord'+str(iord)+'.'+plot_dic['gcal_ord'],transparent=plot_set_key['transparent']) 
                        plt.close() 






    ################################################################################################################    
    #%%%% Telluric CCF
    ################################################################################################################  
    if ('tell_CCF' in plot_settings):
        key_plot = 'tell_CCF'
        plot_set_key = plot_settings[key_plot]
          
        print('-----------------------------------')
        print('+ Telluric CCF')

        #Process original visits   
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())): 
            print('   - Instrument :',inst)
            if inst not in plot_set_key['color_dic']:plot_set_key['color_dic'][inst]={}
            if inst not in plot_set_key['color_dic_sec']:plot_set_key['color_dic_sec'][inst]={}
            for vis in np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst]): 
                print('     - Visit :',vis)
                if vis not in plot_set_key['color_dic'][inst]:plot_set_key['color_dic'][inst][vis]='red'
                if vis not in plot_set_key['color_dic_sec'][inst]:plot_set_key['color_dic_sec'][inst][vis]='dodgerblue'
                
                #Create directory if required
                path_loc = gen_dic['save_plot_dir']+'Spec_raw/Tell_corr/'+inst+'_'+vis+'/Tell_CCF/'
                if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  

                #Exposures to plot
                if (inst in plot_set_key['iexp_plot']) and (vis in plot_set_key['iexp_plot'][inst]) and (len(plot_set_key['iexp_plot'][inst][vis])>0):iexp_plot_vis = deepcopy(plot_set_key['iexp_plot'][inst][vis])
                else:iexp_plot_vis=np.arange(data_dic[inst][vis]['n_in_visit'])
                if plot_set_key['print_disp'] is not None:
                    disp_dic = {}
                    for molec in plot_set_key['tell_species']: disp_dic[molec] = np.zeros(len(iexp_plot_vis),dtype=float)*np.nan
                for isub_exp,iexp in enumerate(iexp_plot_vis):

                    #Check if exposure was fitted
                    data_path = gen_dic['save_data_dir']+'Corr_data/Tell/'+inst+'_'+vis+'_'+str(iexp)+'_add.npz'
                    data_path = np.array(glob.glob(data_path))
                    if len(data_path)>0:
                        data_exp = np.load(data_path[0],allow_pickle=True)['data'].item()
                        
                        #Loop on requested molecules
                        for molec in plot_set_key['tell_species']:                           
                            if molec in data_exp:
                                
                                #Frame
                                plt.ioff() 
                                fig = plt.figure(figsize=plot_set_key['fig_size'])
                                x_min=1e100
                                x_max=-1e100
                                y_min=1e100
                                y_max=-1e100

                                #Plot measured, model, and corrected telluric CCFs
                                plt.plot(data_exp['velccf'],data_exp[molec]['ccf_uncorr_master'],color=plot_set_key['color_dic'][inst][vis],linestyle='-',lw=plot_set_key['lw_plot'],zorder=0)   
                                plt.plot(data_exp['velccf'],data_exp[molec]['ccf_model_conv_master'],color='green',linestyle='-',lw=plot_set_key['lw_plot'],zorder=1)   
                                plt.plot(data_exp['velccf'],data_exp[molec]['ccf_corr_master'],color=plot_set_key['color_dic_sec'][inst][vis],linestyle='-',lw=plot_set_key['lw_plot'],zorder=1)   
                    
                                if plot_set_key['x_range'] is None:
                                    x_min=np.min([np.min(data_exp['velccf']),x_min])
                                    x_max=np.max([np.max(data_exp['velccf']),x_max]) 
                                if plot_set_key['y_range'] is None:
                                    y_min=np.min([np.nanmin(data_exp[molec]['ccf_uncorr_master']),np.nanmin(data_exp[molec]['ccf_model_conv_master']),np.nanmin(data_exp[molec]['ccf_corr_master']),y_min])
                                    y_max=np.max([np.nanmax(data_exp[molec]['ccf_uncorr_master']),np.nanmax(data_exp[molec]['ccf_model_conv_master']),np.nanmax(data_exp[molec]['ccf_corr_master']),y_max])                                        
                                
                                #Plot dispersion  
                                if plot_set_key['print_disp'] is not None:
                                    cond_disp = (~np.isnan(data_exp[molec]['ccf_corr_master'])) & (data_exp['velccf']>=plot_set_key['print_disp'][0]) & (data_exp['velccf']<=plot_set_key['print_disp'][1])
                                    disp_CCF =  np.std(data_exp[molec]['ccf_corr_master'][cond_disp])
                                    plt.text(0.05,1.1,'$\sigma$ ='+"{0:.2e}".format(disp_CCF),verticalalignment='center', horizontalalignment='left',fontsize=10.,zorder=10,color='black',transform=plt.gca().transAxes) 
                                    disp_dic[molec][isub_exp] = disp_CCF
                                             
                                #Frame
                                x_range_loc=plot_set_key['x_range'] if plot_set_key['x_range'] is not None else np.array([x_min,x_max])
                                if (plot_set_key['y_range'] is not None) and (inst in plot_set_key['y_range']) and (vis in plot_set_key['y_range'][inst]) and (molec in plot_set_key['y_range'][inst][vis]):
                                    y_range_loc = plot_set_key['y_range'][inst][vis][molec]
                                else:y_range_loc=np.array([y_min,y_max])                            
                                dx_range=x_range_loc[1]-x_range_loc[0]        
                                dy_range=y_range_loc[1]-y_range_loc[0]                         
                                xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                                ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)
                                custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                                            x_title=r'Velocity in telluric rest frame (km s$^{-1}$)',y_title='Flux',font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                                plt.savefig(path_loc+molec+'_idx'+str(iexp)+'.'+plot_dic['tell_CCF']) 
                                plt.close()  

                #Dispersion plots
                if plot_set_key['print_disp'] is not None:
                    for molec in plot_set_key['tell_species']:  
                    
                        #Frame
                        plt.ioff() 
                        fig = plt.figure(figsize=plot_set_key['fig_size'])
                        x_min=1e100
                        x_max=-1e100
                        y_min=1e100
                        y_max=-1e100                            
                        
                        #Plot defined exposures
                        cond_def = ~np.isnan(disp_dic[molec][isub_exp])
                        cen_ph_vis = coord_dic[inst][vis][plot_set_key['pl_ref'][inst][vis]]['cen_ph'][cond_def]
                        disp_molec = disp_dic[molec][cond_def]
                        plt.plot(cen_ph_vis,disp_molec,color='dodgerblue',linestyle='',zorder=1,marker='o',markersize=1.5)

                        #Boundaries
                        x_min=np.min([np.nanmin(cen_ph_vis),x_min])
                        x_max=np.max([np.nanmax(cen_ph_vis),x_max])                      
                        y_min=np.min([np.nanmin(disp_molec),y_min])
                        y_max=np.max([np.nanmax(disp_molec),y_max]) 
                        dx_range = x_max-x_min
                        x_range_loc = [x_min-0.05*dx_range,x_max+0.05*dx_range] 
                        dy_range = y_max-y_min
                        y_range_loc = [y_min-0.05*dy_range,y_max+0.05*dy_range]                              
                        
                        #Mean dispersion
                        mean_disp = np.mean(disp_molec)
                        plt.plot(x_range_loc,[mean_disp,mean_disp],color='dodgerblue',linestyle='-',zorder=1)
                        plt.text(0.05,1.1,'$\sigma$ ='+"{0:.2e}".format(mean_disp),verticalalignment='center', horizontalalignment='left',fontsize=10.,zorder=10,color='black',transform=plt.gca().transAxes) 
                        
                        #Frame                         
                        dx_range=x_range_loc[1]-x_range_loc[0]        
                        dy_range=y_range_loc[1]-y_range_loc[0]                         
                        xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                        ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)
                        custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                                    x_title=r'Orbital Phase',y_title='Dispersion',font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                        plt.savefig(path_loc+molec+'_disp.'+plot_dic['tell_CCF']) 
                        plt.close() 





    ################################################################################################################    
    #%%%% Telluric properties
    ################################################################################################################ 
    if ('tell_prop' in plot_settings):
        key_plot = 'tell_prop'
        plot_set_key = plot_settings[key_plot]    
    
        print('-----------------------------------')
        print('+ Telluric properties')

        #Process original visits   
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())): 
            print('   - Instrument :',inst)
            if inst not in plot_set_key['color_dic']:plot_set_key['color_dic'][inst]={}
            
            for vis in np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst]): 
                print('     - Visit :',vis)
                if vis not in plot_set_key['color_dic'][inst]:plot_set_key['color_dic'][inst][vis]='dodgerblue'
                data_vis = data_dic[inst][vis]
                cen_ph_vis = coord_dic[inst][vis][plot_set_key['pl_ref'][inst][vis]]['cen_ph']
                
                #Create directory if required
                path_loc = gen_dic['save_plot_dir']+'Spec_raw/Tell_corr/'+inst+'_'+vis+'/Tell_prop/'
                if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  

                #Exposures to plot
                if (inst in plot_set_key['iexp_plot']) and (vis in plot_set_key['iexp_plot'][inst]) and (len(plot_set_key['iexp_plot'][inst][vis])>0):iexp_plot_vis = deepcopy(plot_set_key['iexp_plot'][inst][vis])
                else:iexp_plot_vis=np.arange(data_dic[inst][vis]['n_in_visit'])

                #Retrieve fitted data
                for isub_exp,iexp in enumerate(iexp_plot_vis):

                    #Check if exposure was fitted
                    data_path = gen_dic['save_data_dir']+'Corr_data/Tell/'+inst+'_'+vis+'_'+str(iexp)+'_add.npz'
                    data_path = np.array(glob.glob(data_path))
                    if len(data_path)>0:
                        data_exp = np.load(data_path[0],allow_pickle=True)['data'].item()

                        #Initialize
                        if isub_exp==0:
                            tell_prop = {}
                            for molec in plot_set_key['tell_species']:
                                if molec in data_exp:
                                    tell_prop[molec]={}  
                                    for key in ['Temperature','IWV_LOS','Pressure_ground']:tell_prop[molec][key] = np.zeros([2,data_vis['n_in_visit']])*np.nan                            

                        #Store exposure values
                        for molec in tell_prop:
                            p_best = data_exp[molec]['p_best']
                            for key in ['Temperature','IWV_LOS','Pressure_ground']:
                                if p_best[key].vary:
                                    if plot_settings[key_plot]['plot_err']:tell_prop[molec][key][:,iexp] = [p_best[key].value,p_best[key].stderr]
                                    else:tell_prop[molec][key][:,iexp] = [p_best[key].value,0.]
                        
                                         
                #Loop on requested molecules
                for molec in plot_set_key['tell_species']:
                    for key in ['Temperature','IWV_LOS','Pressure_ground']:
                        cond_def = ~np.isnan(tell_prop[molec][key][0])
                        if True in cond_def:

                            #Frame
                            plt.ioff() 
                            fig = plt.figure(figsize=plot_set_key['fig_size'])
                            
                            #Data
                            if plot_settings[key_plot]['plot_err']:
                                sub_cond_def = ~np.isnan(tell_prop[molec][key][1][cond_def])
                                plt.errorbar(cen_ph_vis[cond_def][sub_cond_def],tell_prop[molec][key][0][cond_def][sub_cond_def],yerr=tell_prop[molec][key][1][cond_def][sub_cond_def],linestyle='',color=plot_set_key['color_dic'][inst][vis],rasterized=plot_set_key['rasterized'],zorder=0,alpha=plot_set_key['alpha_err'])                       
                            else:sub_cond_def = np.repeat(True,np.sum(cond_def))
                            plt.plot(cen_ph_vis[cond_def],tell_prop[molec][key][0][cond_def],color=plot_set_key['color_dic'][inst][vis],linestyle='',zorder=1,marker='o',markersize=1.5)
                     
                            #Frame
                            if plot_set_key['x_range'] is not None:
                                x_range_loc=plot_set_key['x_range']
                            else:
                                xmin = cen_ph_vis[cond_def][0]
                                xmax = cen_ph_vis[cond_def][-1]
                                dx_range = xmax-xmin
                                x_range_loc = [xmin-0.05*dx_range,xmax+0.05*dx_range]  
                            if (plot_set_key['y_range'] is not None) and (inst in plot_set_key['y_range']) and (vis in plot_set_key['y_range'][inst]) and (molec in plot_set_key['y_range'][inst][vis]) and (key in plot_set_key['y_range'][inst][vis][molec]):
                                y_range_loc = plot_set_key['y_range'][inst][vis][molec][key]
                            else:
                                ymin = np.min(tell_prop[molec][key][0][cond_def][sub_cond_def]-tell_prop[molec][key][1][cond_def][sub_cond_def])
                                ymax = np.max(tell_prop[molec][key][0][cond_def][sub_cond_def]+tell_prop[molec][key][1][cond_def][sub_cond_def])
                                dy_range = ymax-ymin
                                y_range_loc = [ymin-0.05*dy_range,ymax+0.05*dy_range]  
                            dx_range=x_range_loc[1]-x_range_loc[0]        
                            dy_range=y_range_loc[1]-y_range_loc[0]     
                            xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                            ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)
                            y_title = {'Temperature':'Temperature (K)','IWV_LOS':'Column density (cm$^{-2}$)','Pressure_ground':'Pressure (atm)'}[key]
                            custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                                        x_title='Orbital phase',y_title=y_title,font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                            plt.savefig(path_loc+molec+'_'+key+'.'+plot_dic['tell_prop']) 
                            plt.close()  
            
            






    ##################################################################################################
    #%%% Flux balance corrections
    ##################################################################################################    

    ################################################################################################################    
    #%%%% Global scaling masters        
    ################################################################################################################
    if ('glob_mast' in plot_settings):
        key_plot = 'glob_mast'

        print('-----------------------------------')
        print('+ Global scaling master')
        sub_plot_all_prof(plot_settings[key_plot],key_plot,plot_dic[key_plot])   



    ################################################################################################################    
    #%%%% Global flux balance (exposures)
    ################################################################################################################
    if ('Fbal_corr' in plot_settings):
        key_plot = 'Fbal_corr'
        plot_set_key = plot_settings[key_plot] 

        print('-----------------------------------')
        print('+ Global flux balance (exposures)')
        
        #Create directory if required
        path_loc = gen_dic['save_plot_dir']+'Spec_raw/FluxBalance/'
        if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  

        #Process original visits        
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())): 
            
            #Process each visit
            for vis in np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst]): 
                data_vis = data_dic[inst][vis]
                data_Fbal_gen = np.load(gen_dic['save_data_dir']+'Corr_data/Fbal/'+inst+'_'+vis+'_add.npz', allow_pickle=True)['data'].item()
                if plot_set_key['gap_exp']==0.:plot_settings[key_plot]['margins']=[0.15,0.15,0.95,0.7] 
                else:plot_set_key['margins']=[0.1,0.02,0.95,0.95] 
                    
                #Visit color
                n_in_visit=data_vis['n_in_visit']
                if (inst in plot_set_key['color_dic']) and (vis in plot_set_key['color_dic'][inst]):
                    col_visit=np.repeat(plot_set_key['color_dic'][inst][vis],n_in_visit)
                else:
                    cmap = plt.get_cmap('jet') 
                    col_visit=np.array([cmap(0)]) if n_in_visit==1 else cmap( np.arange(n_in_visit)/(n_in_visit-1.))

                plt.ioff()        
                if plot_set_key['gap_exp']==0.:fig = plt.figure(figsize=plot_set_key['fig_size'])
                else:fig = plt.figure(figsize=(10, 60))

                #Vertical range
                y_min=1e100
                y_max=-1e100

                #Horizontal range
                if plot_set_key['x_range'] is not None:x_range_loc = deepcopy(plot_set_key['x_range'])
                else:
                    data_com = np.load(data_vis['proc_com_data_paths']+'.npz',allow_pickle=True)['data'].item() 
                    if plot_set_key['sp_var'] == 'nu' :x_range_loc = [c_light/data_com['edge_bins'][-1,-1],c_light/data_com['edge_bins'][0,0]]
                    elif plot_set_key['sp_var'] == 'wav' :x_range_loc = [data_com['edge_bins'][0,0],data_com['edge_bins'][-1,-1]]
                if plot_set_key['plot_model']:
                    if plot_set_key['sp_var'] == 'wav' :
                        min_wav,max_wav = x_range_loc[0],x_range_loc[1]
                    elif plot_set_key['sp_var'] == 'nu' :
                        min_wav,max_wav = c_light/x_range_loc[1],c_light/x_range_loc[0] 
                    nspec_plot = int( np.floor(   np.log(max_wav/min_wav)/np.log( plot_set_key['dlnw_plot'] + 1. ) )  ) 
                    cen_bin_plot = min_wav*( plot_set_key['dlnw_plot'] + 1. )**(0.5+np.arange(nspec_plot))  
                    nu_bin_plot = c_light/cen_bin_plot[::-1]

                #Binned ratio of spectrum with stellar reference, measured and fitted
                if (inst in plot_set_key['iexp_plot']) and (vis in plot_set_key['iexp_plot'][inst]) and (len(plot_set_key['iexp_plot'][inst][vis])>0):iexp_plot_vis = deepcopy(plot_set_key['iexp_plot'][inst][vis])
                else:iexp_plot_vis=np.arange(data_vis['n_in_visit'])
                for isub,iexp in enumerate(iexp_plot_vis):
                    lev_exp = isub*plot_set_key['gap_exp'] 

                    #Upload flux balance correction data
                    data_Fbal = np.load(gen_dic['save_data_dir']+'Corr_data/Fbal/'+inst+'_'+vis+'_'+str(iexp)+'_add.npz', allow_pickle=True)['data'].item()
                    if (inst in plot_set_key['ibin_plot']) and (vis in plot_set_key['ibin_plot'][inst]) and (len(plot_set_key['ibin_plot'][inst][vis])>0):ibin_exp = list(plot_set_key['ibin_plot'][inst][vis])
                    else:ibin_exp=range(len(data_Fbal['Fbal_wav_bin_all'][1])) 

                    #Mesured ratio binned exposure/binned master
                    if plot_set_key['plot_data']:
                        cond_fit_plot = data_Fbal['cond_fit'][ibin_exp]
                        Fbal_T_binned_plot = data_Fbal['Fbal_T_binned_all'][ibin_exp]/data_Fbal_gen['tot_Fr_all'][iexp]
                        if plot_set_key['sp_var'] == 'nu' :Fbal_cen_bin_plot = c_light/data_Fbal['Fbal_wav_bin_all'][1,ibin_exp]
                        elif plot_set_key['sp_var'] == 'wav' :Fbal_cen_bin_plot = data_Fbal['Fbal_wav_bin_all'][1,ibin_exp]
                        y_min=np.min([np.min(lev_exp+Fbal_T_binned_plot),y_min])
                        y_max=np.max([np.max(lev_exp+Fbal_T_binned_plot),y_max]) 
                        for cen_bin,val_bin,cond_bin in zip(Fbal_cen_bin_plot,Fbal_T_binned_plot,cond_fit_plot):
                            markerfacecolor = col_visit[iexp] if cond_bin else 'None'
                            plt.plot(cen_bin,lev_exp+val_bin,linestyle='',marker='o',markerfacecolor=markerfacecolor  ,markeredgecolor=col_visit[iexp],markersize=plot_set_key['markersize'],rasterized=plot_set_key['rasterized'])

                    #Best-fit model
                    if plot_set_key['plot_model']:
                        if plot_set_key['sp_var'] == 'nu' :
                            mod_cen_bin_plot = nu_bin_plot
                            y_mod = data_Fbal['corr_func'](nu_bin_plot)
                        elif plot_set_key['sp_var'] == 'wav' :
                            mod_cen_bin_plot = cen_bin_plot
                            y_mod = data_Fbal['corr_func'](nu_bin_plot[::-1])
                        plt.plot(mod_cen_bin_plot,lev_exp+y_mod,linestyle='-',color=col_visit[iexp],lw=1) 

                    #Plot exposure indexes
                    if plot_set_key['plot_expid']:
                        if plot_set_key['plot_data']:ytxt = np.median(Fbal_T_binned_plot)
                        if plot_set_key['plot_model']: ytxt = np.median(y_mod)
                        plt.text(x_range_loc[1]+0.01*(x_range_loc[1]-x_range_loc[0]),lev_exp+ytxt,str(iexp),verticalalignment='center', horizontalalignment='left',fontsize=5.,zorder=15,color=col_visit[iexp]) 

                #Shade area not included within fit
                if plot_set_key['shade_unfit']:
                    if (inst not in gen_dic['Fbal_range_fit']) or ((inst in gen_dic['Fbal_range_fit']) & (vis not in gen_dic['Fbal_range_fit'][inst])):fit_range=[x_range_loc]
                    else:
                        if plot_set_key['sp_var'] == 'nu' :fit_range=c_light/gen_dic['Fbal_range_fit'][inst][vis]  
                        elif plot_set_key['sp_var'] == 'wav' :fit_range=gen_dic['Fbal_range_fit'][inst][vis]       
                    plot_shade_range(plt.gca(),fit_range,x_range_loc,y_range_loc,mode='span',zorder=-10,alpha=0.1,compl=True)

                #Strip range used for correction
                y_range_loc=plot_set_key['y_range'] if plot_set_key['y_range'] is not None else np.array([y_min,y_max])
                if plot_set_key['strip_corr']:
                    if len(gen_dic['Fbal_range_corr'])==0:Fbal_range_corr=[x_range_loc]
                    else:
                        if plot_set_key['sp_var'] == 'nu' :Fbal_range_corr=c_light/gen_dic['Fbal_range_corr']
                        elif plot_set_key['sp_var'] == 'wav' :Fbal_range_corr=gen_dic['Fbal_range_corr']                    
                    for bd_band_loc in Fbal_range_corr:
                        plt.fill([bd_band_loc[0],bd_band_loc[1],bd_band_loc[1],bd_band_loc[0]],[y_range_loc[0],y_range_loc[0],y_range_loc[1],y_range_loc[1]], fill=False, hatch='\\',color='grey',zorder=-5)

                #Plot order index
                if y_range_loc[0]<-100000.:y_range_loc[0]=-100000.
                if y_range_loc[1]>100000.:y_range_loc[1]=100000.
                dy_range=y_range_loc[1]-y_range_loc[0]
                if plot_set_key['plot_idx_ord']:
                    if plot_set_key['sp_var'] == 'nu' :cen_ord = c_light/gen_dic['wav_ord_inst'][inst]
                    elif plot_set_key['sp_var'] == 'wav' :cen_ord = gen_dic['wav_ord_inst'][inst]                
                    delt_txt = 0.03 if plot_set_key['gap_exp'] == 0. else 0.005
                    # ord_freq = 4
                    # fontsize_ord = 6.
                    ord_freq = 8
                    fontsize_ord = 11.
                    for iord in np.arange(0,len(cen_ord),ord_freq):
                        plt.text(cen_ord[iord],y_range_loc[1]+delt_txt*dy_range,str(iord),verticalalignment='center', horizontalalignment='center',fontsize=fontsize_ord,zorder=15,color='black') 
                        plt.plot([cen_ord[iord],cen_ord[iord]],y_range_loc,linestyle='--',zorder=-15,color='darkgrey',lw=1) 
            
                #Plot frame  
                if plot_set_key['title']:plt.title('Global flux balance for visit '+vis+' in '+inst)               
                dx_range=x_range_loc[1]-x_range_loc[0]
                xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)     
                if plot_set_key['sp_var'] == 'nu' :x_title=r'$\nu$ (10$^{-10}$s$^{-1}$)'
                elif plot_set_key['sp_var'] == 'wav' :x_title='Wavelength (A)'
                custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,dir_y='out', 
                            xmajor_int=xmajor_int,xminor_int=xminor_int,xmajor_form=xmajor_form,
                            ymajor_int=ymajor_int,yminor_int=yminor_int,ymajor_form=ymajor_form,
                            # x_mode='log',
                            x_title=x_title,y_title='Flux ratio',
                            font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                plt.savefig(path_loc+inst+'_'+vis+'_global.'+plot_dic['Fbal_corr']) 
                plt.close() 




    ################################################################################################################    
    #%%%% Global DRS flux balance (exposures)
    ################################################################################################################
    if ('Fbal_corr_DRS' in plot_settings):
        key_plot = 'Fbal_corr_DRS'
        plot_set_key = plot_settings[key_plot] 

        print('-----------------------------------')
        print('   > Plotting flux balance correction from DRS')
                        
        #Create directory if required
        path_loc = gen_dic['save_plot_dir']+'Spec_raw/FluxBalance/'
        if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  
        
        #Process original visits        
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())): 
            
            #Wavelength centers of the instrument orders
            if (inst in gen_dic['del_orders']):wav_ord_inst = gen_dic[inst]['wav_ord_inst'] 
            else: wav_ord_inst = gen_dic['wav_ord_inst'][inst]

            #Process each visit
            for vis in np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst]):
                plt.ioff()        
                fig = plt.figure(figsize=plot_set_key['fig_size'])
                data_vis = data_dic[inst][vis]

                #Vertical range
                y_min=1e100
                y_max=-1e100

                #Visit color
                n_in_visit=data_vis['n_in_visit']
                if (inst in plot_set_key['color_dic']) and (vis in plot_set_key['color_dic'][inst]):
                    col_visit=np.repeat(plot_set_key['color_dic'][inst][vis],n_in_visit)
                else:
                    cmap = plt.get_cmap('jet') 
                    col_visit=np.array([cmap(0)]) if n_in_visit==1 else cmap( np.arange(n_in_visit)/(n_in_visit-1.))

                #Ratio of binned spectra with stellar reference, measured and fitted
                for iexp in range(n_in_visit):

                    #Flux balance correction data
                    Fbal_T_exp = data_prop[inst][vis]['colcorr_ord'][iexp] 
                    plt.plot(wav_ord_inst,Fbal_T_exp,linestyle='-',marker='s',markerfacecolor=col_visit[iexp],markeredgecolor=col_visit[iexp],markersize=plot_set_key['markersize'] )
                    y_min=np.min([np.min(Fbal_T_exp),y_min])
                    y_max=np.max([np.max(Fbal_T_exp),y_max]) 
                
                #Plot frame  
                y_range_loc=plot_set_key['y_range'] if plot_set_key['y_range'] is not None else np.array([y_min,y_max])
                if plot_set_key['title']:plt.title('Flux balance correction for visit '+vis+' in '+inst)
                custom_axis(plt,position=plot_set_key['margins'],x_range=plot_set_key['x_range'],y_range=y_range_loc,dir_y='out', 
                            xmajor_int=1000.,xminor_int=100.,
                            xmajor_form='%.1f',ymajor_form='%.1f',
                            x_title='Wavelength (A)',y_title='Flux ratio',
                            font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                plt.savefig(path_loc+inst+'_'+vis+'_DRS.'+plot_dic['Fbal_corr_DRS']) 
                plt.close() 
        




    ################################################################################################################    
    #%%%% Global flux balance (visits)
    ################################################################################################################
    if ('Fbal_corr_vis' in plot_settings):  
        key_plot = 'Fbal_corr_vis'
        plot_set_key = plot_settings[key_plot] 

        print('-----------------------------------')
        print('+ Global flux balance (visits)')
        
        #Create directory if required
        path_loc = gen_dic['save_plot_dir']+'Spec_raw/FluxBalance/'
        if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  

        #Process original visits        
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())): 
            if plot_set_key['gap_exp']==0.:plot_set_key['margins']=[0.15,0.15,0.95,0.7] 
            else:plot_set_key['margins']=[0.1,0.02,0.95,0.95] 

            plt.ioff()        
            if plot_set_key['gap_exp']==0.:fig = plt.figure(figsize=(10, 4))
            else:fig = plt.figure(figsize=(10, 10))

            #Vertical range
            y_min=1e100
            y_max=-1e100

            #Horizontal range
            if plot_set_key['x_range'] is not None:x_range_loc = deepcopy(plot_set_key['x_range'])
            else:
                data_com = np.load(data_dic[inst]['proc_com_data_path']+'.npz',allow_pickle=True)['data'].item() 
                if plot_set_key['sp_var'] == 'nu' :x_range_loc = [c_light/data_com['edge_bins'][-1,-1],c_light/data_com['edge_bins'][0,0]]
                elif plot_set_key['sp_var'] == 'wav' :x_range_loc = [data_com['edge_bins'][0,0],data_com['edge_bins'][-1,-1]]
            if plot_set_key['plot_model']:
                if plot_set_key['sp_var'] == 'wav' :min_wav,max_wav = x_range_loc[0],x_range_loc[1]
                elif plot_set_key['sp_var'] == 'nu' :min_wav,max_wav = c_light/x_range_loc[1],c_light/x_range_loc[0] 
                nspec_plot = int( np.floor(   np.log(max_wav/min_wav)/np.log( plot_set_key['dlnw_plot'] + 1. ) )  ) 
                cen_bin_plot = min_wav*( plot_set_key['dlnw_plot'] + 1. )**(0.5+np.arange(nspec_plot))  
                nu_bin_plot = c_light/cen_bin_plot[::-1]
            
            #Process each visit
            for isub,vis in enumerate(np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst])): 
                data_vis = data_dic[inst][vis]
  
                #Visit color
                if (inst in plot_set_key['color_dic']) and (vis in plot_set_key['color_dic'][inst]):
                    col_visit=plot_set_key['color_dic'][inst][vis]
                else:
                    col_visit='dodgerblue'

                #Binned ratio of spectrum with stellar reference, measured and fitted
                lev_exp = isub*plot_set_key['gap_exp'] 

                #Upload flux balance correction data
                data_Fbal = np.load(gen_dic['save_data_dir']+'Corr_data/Fbal/'+inst+'_'+vis+'_add.npz', allow_pickle=True)['data'].item()              
                if (inst in plot_set_key['ibin_plot']) and (vis in plot_set_key['ibin_plot'][inst]) and (len(plot_set_key['ibin_plot'][inst][vis])>0):ibin_exp = list(plot_set_key['ibin_plot'][inst][vis])
                else:ibin_exp=range(len(data_Fbal['Fbal_wav_bin_vis'])) 

                #Mesured ratio binned exposure/binned master
                if plot_set_key['plot_data']:
                    cond_fit_plot = data_Fbal['cond_fit_vis'][ibin_exp]
                    Fbal_T_binned_plot = data_Fbal['Fbal_T_binned_vis'][ibin_exp]/data_Fbal['tot_Fr_vis']
                    if plot_set_key['sp_var'] == 'nu' :Fbal_cen_bin_plot = c_light/data_Fbal['Fbal_wav_bin_vis'][ibin_exp]
                    elif plot_set_key['sp_var'] == 'wav' :Fbal_cen_bin_plot = data_Fbal['Fbal_wav_bin_vis'][ibin_exp]
                    y_min=np.min([np.min(lev_exp+Fbal_T_binned_plot),y_min])
                    y_max=np.max([np.max(lev_exp+Fbal_T_binned_plot),y_max]) 
                    for cen_bin,val_bin,cond_bin in zip(Fbal_cen_bin_plot,Fbal_T_binned_plot,cond_fit_plot):
                        markerfacecolor = col_visit if cond_bin else 'None'
                        plt.plot(cen_bin,lev_exp+val_bin,linestyle='',marker='o',markerfacecolor=markerfacecolor  ,markeredgecolor=col_visit,markersize=plot_set_key['markersize'],rasterized=plot_set_key['rasterized'])

                #Best-fit model
                if plot_set_key['plot_model']:
                    if plot_set_key['sp_var'] == 'nu' :
                        mod_cen_bin_plot = nu_bin_plot
                        y_mod = data_Fbal['corr_func_vis'](nu_bin_plot)
                    elif plot_set_key['sp_var'] == 'wav' :
                        mod_cen_bin_plot = cen_bin_plot
                        y_mod = data_Fbal['corr_func_vis'](nu_bin_plot)[::-1]
                    plt.plot(mod_cen_bin_plot,lev_exp+y_mod,linestyle='-',color=col_visit,lw=1) 

                #Plot visit names
                if plot_set_key['plot_expid']:
                    if plot_set_key['plot_data']:ytxt = np.median(Fbal_T_binned_plot)
                    if plot_set_key['plot_model']: ytxt = np.median(y_mod)
                    plt.text(x_range_loc[1]+0.01*(x_range_loc[1]-x_range_loc[0]),lev_exp+ytxt,vis,verticalalignment='center', horizontalalignment='left',fontsize=5.,zorder=15,color=col_visit) 

            #Shade area not included within fit
            if plot_set_key['shade_unfit']:
                if (inst not in gen_dic['Fbal_range_fit']) or ((inst in gen_dic['Fbal_range_fit']) & (vis not in gen_dic['Fbal_range_fit'][inst])):fit_range=[x_range_loc]
                else:
                    if plot_set_key['sp_var'] == 'nu' :fit_range=c_light/gen_dic['Fbal_range_fit'][inst][vis]  
                    elif plot_set_key['sp_var'] == 'wav' :fit_range=gen_dic['Fbal_range_fit'][inst][vis]       
                plot_shade_range(plt.gca(),fit_range,x_range_loc,y_range_loc,mode='span',zorder=-10,alpha=0.1,compl=True)

            #Strip range used for correction
            y_range_loc=plot_set_key['y_range'] if plot_set_key['y_range'] is not None else np.array([y_min,y_max])
            if plot_set_key['strip_corr']:
                if len(gen_dic['Fbal_range_corr'])==0:Fbal_range_corr=[x_range_loc]
                else:
                    if plot_set_key['sp_var'] == 'nu' :Fbal_range_corr=c_light/gen_dic['Fbal_range_corr']
                    elif plot_set_key['sp_var'] == 'wav' :Fbal_range_corr=gen_dic['Fbal_range_corr']                    
                for bd_band_loc in Fbal_range_corr:
                    plt.fill([bd_band_loc[0],bd_band_loc[1],bd_band_loc[1],bd_band_loc[0]],[y_range_loc[0],y_range_loc[0],y_range_loc[1],y_range_loc[1]], fill=False, hatch='\\',color='grey',zorder=-5)

            #Plot order index
            if y_range_loc[0]<-100000.:y_range_loc[0]=-100000.
            if y_range_loc[1]>100000.:y_range_loc[1]=100000.
            dy_range=y_range_loc[1]-y_range_loc[0]
            if plot_set_key['plot_idx_ord']:
                if plot_set_key['sp_var'] == 'nu' :cen_ord = c_light/gen_dic['wav_ord_inst'][inst]
                elif plot_set_key['sp_var'] == 'wav' :cen_ord = gen_dic['wav_ord_inst'][inst]                
                delt_txt = 0.03 if plot_set_key['gap_exp'] == 0. else 0.005
                ord_freq = 8
                fontsize_ord = 11.
                for iord in np.arange(0,len(cen_ord),ord_freq):
                    plt.text(cen_ord[iord],y_range_loc[1]+delt_txt*dy_range,str(iord),verticalalignment='center', horizontalalignment='center',fontsize=fontsize_ord,zorder=15,color='black') 
                    plt.plot([cen_ord[iord],cen_ord[iord]],y_range_loc,linestyle='--',zorder=-15,color='darkgrey',lw=1) 
        
            #Plot frame  
            if plot_set_key['title']:plt.title('Global flux balance for instrument '+inst)               
            dx_range=x_range_loc[1]-x_range_loc[0]
            xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
            ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)     
            if plot_set_key['sp_var'] == 'nu' :x_title=r'$\nu$ (10$^{-10}$s$^{-1}$)'
            elif plot_set_key['sp_var'] == 'wav' :x_title='Wavelength (A)'
            custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,dir_y='out', 
                        xmajor_int=xmajor_int,xminor_int=xminor_int,xmajor_form=xmajor_form,
                        ymajor_int=ymajor_int,yminor_int=yminor_int,ymajor_form=ymajor_form,
                        # x_mode='log',
                        x_title=x_title,y_title='Flux ratio',
                        font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
            plt.savefig(path_loc+inst+'_global_vis.'+plot_dic['Fbal_corr_vis']) 
            plt.close() 





    ################################################################################################################ 
    #%%%% Intra-order flux balance
    ################################################################################################################ 
    if ('Fbal_corr_ord' in plot_settings):  
        key_plot = 'Fbal_corr_ord'
        plot_set_key = plot_settings[key_plot]     
    
        print('-----------------------------------')
        print('   > Plotting intra-order flux balance correction')
        
        #Create directory if required
        path_loc = gen_dic['save_plot_dir']+'Spec_raw/FluxBalance/Orders/'
        if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  
        
        #Process original visits        
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())): 

            #Process each visit
            for vis in np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst]): 
                data_vis = data_dic[inst][vis]
                data_Fbal_gen = dataload_npz(gen_dic['save_data_dir']+'Corr_data/Fbal/'+inst+'_'+vis+'_add')['Ord']

                #Orders to process
                order_list = data_Fbal_gen['iord_corr_list']
                if len(plot_set_key['orders_to_plot'])>0:order_list = np.intersect1d(order_list,data_Fbal_gen['iord_corr_list'])
                nord_list = len(order_list)

                #Visit color
                n_in_visit=data_vis['n_in_visit']
                if (inst in plot_set_key['color_dic']) and (vis in plot_set_key['color_dic'][inst]):
                    col_visit=np.repeat(plot_set_key['color_dic'][inst][vis],n_in_visit)
                else:
                    cmap = plt.get_cmap('jet') 
                    col_visit=np.array([cmap(0)]) if n_in_visit==1 else cmap( np.arange(n_in_visit)/(n_in_visit-1.))

                #Upload data
                Fbal_wav_bin_all = np.empty(nord_list,dtype=object)
                Fbal_T_binned_all =  np.empty(nord_list,dtype=object)
                cen_bins_all =  np.empty(nord_list,dtype=object)
                Fbal_T_fit_all =  np.empty(nord_list,dtype=object) 
                tot_Fr_all = np.empty(nord_list,dtype=object) 
                for isub_ord in range(nord_list):
                    Fbal_wav_bin_all[isub_ord] = np.empty(n_in_visit,dtype=object)
                    Fbal_T_binned_all[isub_ord] = np.empty(n_in_visit,dtype=object)
                    cen_bins_all[isub_ord] = np.empty(n_in_visit,dtype=object)
                    Fbal_T_fit_all[isub_ord] = np.empty(n_in_visit,dtype=object)
                    tot_Fr_all[isub_ord] = np.empty(n_in_visit,dtype=object)                               
                for iexp in range(n_in_visit):
                    data_Fbal = dataload_npz(gen_dic['save_data_dir']+'Corr_data/Fbal/'+inst+'_'+vis+'_'+str(iexp)+'_add')['Ord']  
                    for isub_ord,iord in enumerate(order_list):
                        Fbal_wav_bin_all[isub_ord][iexp] = data_Fbal['Fbal_wav_bin_all'][iord][1]
                        Fbal_T_binned_all[isub_ord][iexp] = data_Fbal['Fbal_T_binned_all'][iord]
                        cen_bins_all[isub_ord][iexp] = data_Fbal['cen_bins_all'][iord]
                        Fbal_T_fit_all[isub_ord][iexp] = data_Fbal['Fbal_T_fit_all'][iord]
                        tot_Fr_all[isub_ord][iexp] = data_Fbal['tot_Fr_all'][iord]

                #Plot current order for all exposures
                for isub_ord,iord in enumerate(order_list):
    
                    #Global flux balance
                    plt.ioff()        
                    fig = plt.figure(figsize=plot_set_key['fig_size'])

                    #Ratio of binned spectra with stellar reference, measured and fitted
                    y_min=1e100
                    y_max=-1e100
                    for iexp in range(n_in_visit):
                        
                        #Mesured ratio binned exposure/binned master
                        var_plot = Fbal_T_binned_all[isub_ord][iexp]/tot_Fr_all[isub_ord][iexp]
                        y_min=np.min([np.min(var_plot),y_min])
                        y_max=np.max([np.max(var_plot),y_max]) 
                        plt.plot(Fbal_wav_bin_all[isub_ord][iexp],var_plot,linestyle='',marker='o',markerfacecolor=col_visit[iexp],markeredgecolor=col_visit[iexp],markersize=plot_set_key['markersize'])
    
                        #Best-fit calculated over the full spectral table
                        cond_def = ~np.isnan(Fbal_T_fit_all[isub_ord][iexp])
                        plt.plot(cen_bins_all[isub_ord][iexp][cond_def],Fbal_T_fit_all[isub_ord][iexp][cond_def],linestyle='-',color=col_visit[iexp],lw=1,rasterized=plot_set_key['rasterized']) 

                    #Plot frame  
                    if plot_set_key['title']:plt.title('Intra-order flux balance for visit '+vis+' in '+inst+' (order '+str(iord)+')')
                    y_range_loc=plot_set_key['y_range'] if plot_set_key['y_range'] is not None else np.array([y_min,y_max])
                    custom_axis(plt,position=plot_set_key['margins'],x_range=plot_set_key['x_range'],y_range=y_range_loc,dir_y='out', 
                                xmajor_int=10.,xminor_int=1.,
                                xmajor_form='%.1f',ymajor_form='%.3f',
                                x_title='Wavelength (A)',y_title='Flux ratio',
                                font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                    plt.savefig(path_loc+inst+'_'+vis+'_ord'+str(iord)+'.'+plot_dic['Fbal_corr_ord']) 
                    plt.close() 








    ################################################################################################################ 
    #%%%% Temporal flux balance
    ################################################################################################################
    if ('Ftemp_corr' in plot_settings): 
        key_plot = 'Ftemp_corr'
        plot_set_key=plot_settings[key_plot]

        print('-----------------------------------')
        print('   > Plotting temporal flux correction')
            
        #Create directory if required
        path_loc = gen_dic['save_plot_dir']+'Spec_raw/Ftemp_corr/'
        if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  
        
        #Process original visits        
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())): 
            
            #Process each visit
            for vis in np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst]): 
                data_vis = data_dic[inst][vis]
                data_Ftemp = np.load(gen_dic['save_data_dir']+'Corr_data/Ftemp/'+inst+'_'+vis+'_add.npz', allow_pickle=True)['data'].item()
                    
                #Visit color
                n_in_visit=data_vis['n_in_visit']
                if (inst in plot_set_key['color_dic']) and (vis in plot_set_key['color_dic'][inst]):
                    col_visit=np.repeat(plot_set_key['color_dic'][inst][vis],n_in_visit)
                else:
                    cmap = plt.get_cmap('jet') 
                    col_visit=np.array([cmap(0)]) if n_in_visit==1 else cmap( np.arange(n_in_visit)/(n_in_visit-1.))

                plt.ioff()  
                fig = plt.figure(figsize=plot_set_key['fig_size'])

                #Measured values
                for iexp in range(data_vis['n_in_visit']):
                    if iexp in data_Ftemp['iexp_fit']:marker='o'
                    else:marker='s'
                    if plot_set_key['plot_err']:plt.errorbar(data_Ftemp['bjd_all'],data_Ftemp['Tflux_all'],yerr=data_Ftemp['Tsig_all'],linestyle='',color=col_visit[iexp],rasterized=plot_set_key['rasterized'],zorder=0,alpha=plot_set_key['alpha_err'])                       
                    plt.plot(data_Ftemp['bjd_all'],data_Ftemp['Tflux_all'],linestyle='',marker='o',markerfacecolor=col_visit[iexp]  ,markeredgecolor=col_visit[iexp],markersize=plot_set_key['markersize'])

                #Best-fit 
                plt.plot(data_Ftemp['bjd_all'],data_Ftemp['corr_func'](data_Ftemp['bjd_all']),linestyle='-',color=col_visit[iexp],lw=1,rasterized=plot_set_key['rasterized']) 

                #Plot frame  
                x_range_loc = plot_set_key['x_range'] if plot_set_key['x_range'] is not None else [np.min(data_Ftemp['bjd_all']),np.max(data_Ftemp['bjd_all'])] 
                y_range_loc = plot_set_key['y_range'] if plot_set_key['y_range'] is not None else [np.min(data_Ftemp['Tflux_all']-data_Ftemp['Tsig_all']),np.max(data_Ftemp['Tflux_all']+data_Ftemp['Tsig_all'])]          
                if plot_set_key['title']:plt.title('Temporal flux correction for visit '+vis+' in '+inst)               
                custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,dir_y='out', 
                            # xmajor_int=1000.,xminor_int=100.,
                            # xmajor_form='%.1f',ymajor_form='%.1f',
                            # x_title='Wavelength (A)',y_title='Flux ratio',
                            font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                plt.savefig(path_loc+inst+'_'+vis+'.'+plot_dic['Ftemp_corr']) 
                plt.close() 













    ################################################################################################################ 
    #%%% Cosmics correction
    ################################################################################################################ 
    if ('cosm_corr' in plot_settings): 
        key_plot = 'cosm_corr'
        plot_set_key=plot_settings[key_plot]

        print('-----------------------------------')
        print('+ Plotting cosmics correction')
        
        #Process original visits
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())): 
            print('  > Instrument: '+inst)

            #Process each visit
            for vis in np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst]): 
                print('     - Visit : '+vis)
                path_loc = gen_dic['save_plot_dir']+'Spec_raw/Cosmics/'+inst+'_'+vis+'/'
                if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)                 
                data_vis=data_dic[inst][vis]

                #Selected exposures
                if (inst in plot_set_key['iexp_plot']) and (vis in plot_set_key['iexp_plot'][inst]) and (len(plot_set_key['iexp_plot'][inst][vis])>0):iexp_plot_vis = deepcopy(plot_set_key['iexp_plot'][inst][vis])
                else:iexp_plot_vis=np.arange(data_vis['n_in_visit'])
                nexp_list = len(iexp_plot_vis)

                #Orders to plot
                order_list = plot_set_key['orders_to_plot'] if len(plot_set_key['orders_to_plot'])>0 else range(data_dic[inst]['nord_ref']) 
                if (inst in gen_dic['cosm_ord_corr']) and (vis in gen_dic['cosm_ord_corr'][inst]) and (len(gen_dic['cosm_ord_corr'][inst][vis])>0):
                    order_list = np.intersect1d(order_list,gen_dic['cosm_ord_corr'][inst][vis])                        
                nord_list = len(order_list)
                
                #Number of cosmics per exposure and order
                n_cosmics_vis = np.zeros([nexp_list,nord_list],dtype=int)

                #Plot each order for each exposure
                for isub,iexp in enumerate(iexp_plot_vis):                

                    #Upload correction data
                    data_cosm = np.load(gen_dic['save_data_dir']+'Corr_data/Cosm/'+inst+'_'+vis+'_'+str(iexp)+'_add.npz', allow_pickle=True)['data'].item()

                    #Upload corrected data
                    data_exp = np.load(gen_dic['save_data_dir']+'Corr_data/Cosm/'+inst+'_'+vis+'_'+str(iexp)+'.npz',allow_pickle=True)['data'].item()

                    #Orders to plot
                    for isub_ord,iord in enumerate(order_list):
                     
                        #Detected cosmics
                        if iord in data_cosm['idx_cosm_exp']:
                            idx_cosm_tab=data_cosm['idx_cosm_exp'][iord]
                            n_cosmics_vis[isub,isub_ord] = len(idx_cosm_tab)
                        else:n_cosmics_vis[isub,isub_ord]=0
                        if (not plot_set_key['detcosm']) or (n_cosmics_vis[isub,isub_ord]>0):                         
                            wav_tab=data_exp['cen_bins'][iord]     
                            plt.ioff()        
                            plt.figure(figsize=plot_set_key['fig_size'])
                                
                            #Plot boundaries
                            x_range_loc = plot_set_key['x_range'] if plot_set_key['x_range'] is not None else [wav_tab[0],wav_tab[-1]]
                            y_range_loc = plot_set_key['y_range'] if plot_set_key['y_range'] is not None else [0.,np.nanmax(data_cosm['SNR_diff_exp'][:,iord])]
                            dy_range=y_range_loc[1]-y_range_loc[0]
                                
                            #Relative difference exposure-master over :
                            #    - dispersion of complementary spectra (blue)
                            #    - error of processed spectrum (green)
                            # If current exposure has a larger noise level than complementary spectra, the threshold might be exceeded for the blue spectrum (simply because of the dispersion intrinsic to this spectrum) but not for the green spectrum 
                            plt.plot(wav_tab,data_cosm['SNR_diff_exp'][0,iord],linestyle='-',color='black',rasterized=plot_set_key['rasterized'],zorder=0,drawstyle='steps-mid',lw=plot_set_key['lw_plot'])
                            plt.plot(wav_tab,data_cosm['SNR_diff_exp'][1,iord],linestyle='-',color='limegreen',rasterized=plot_set_key['rasterized'],zorder=1,drawstyle='steps-mid',lw=plot_set_key['lw_plot'])
                 
                            #Cosmics
                            if n_cosmics_vis[isub,isub_ord]>0:
                                if plot_set_key['markcosm']:plt.plot(wav_tab[idx_cosm_tab],data_cosm['SNR_diff_exp'][0,iord][idx_cosm_tab],linestyle='',color='red',marker='o',markersize=plot_set_key['markersize'],zorder=2)
                                if plot_set_key['ncosm']:plt.text(x_range_loc[0]+0.1*(x_range_loc[1]-x_range_loc[0]),y_range_loc[0]+0.9*dy_range,'N(cosmics) ='+"{:d}".format(n_cosmics_vis[isub,isub_ord]),verticalalignment='center', horizontalalignment='left',fontsize=15.,zorder=4) 
                                
                            #Threshold
                            plt.plot(x_range_loc,[0,0],linestyle='--',color='grey',lw=1)
                            plt.plot(x_range_loc,[gen_dic['cosm_thresh'][inst][vis],gen_dic['cosm_thresh'][inst][vis]],linestyle='--',color='red',lw=1)
                            
                            #Plot frame  
                            if plot_set_key['title']:plt.title('Cosmics detections for visit '+vis+' in '+inst)
                            dx_range=x_range_loc[1]-x_range_loc[0]
                            xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                            ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)  
                            custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,dir_y='out', 
                                        xmajor_int=xmajor_int,xminor_int=xminor_int,xmajor_form=xmajor_form,
                                        ymajor_int=ymajor_int,yminor_int=yminor_int,ymajor_form=ymajor_form,
                                        x_title='Wavelength in input rest frame (A)',y_title='(F-<F$_\mathrm{compl}$>)/$\sigma$',
                                        font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                            if gen_dic['type'][inst]=='spec1D':str_add=''
                            elif  gen_dic['type'][inst]=='spec2D':str_add='_iord'+str(iord)
                            plt.savefig(path_loc+'idx'+str(iexp)+str_add+'.'+plot_dic['cosm_corr'])
                            plt.close()                     
                        
                #Plot number of comics vs order
                if plot_set_key['cosm_vs_ord']:
                    plt.ioff()        
                    plt.figure(figsize=plot_set_key['fig_size'])
                    cmap = plt.get_cmap('jet') 
                    col_tab = cmap( np.arange(nexp_list)/(nexp_list-1.))
                    
                    #Plot boundaries
                    x_range_loc = plot_set_key['x_range'] if plot_set_key['x_range'] is not None else [gen_dic['wav_ord_inst'][inst][order_list][0],gen_dic['wav_ord_inst'][inst][order_list][-1]]
                    y_range_loc = plot_set_key['y_range'] if plot_set_key['y_range'] is not None else [0.,np.max(n_cosmics_vis)]
                            
                    #Plot number of cosmics for each exposure and the total count (scaled to the axis range)
                    for isub,iexp in enumerate(iexp_plot_vis): 
                        plt.plot(gen_dic['wav_ord_inst'][inst][order_list],n_cosmics_vis[isub],linestyle='-',color=col_tab[isub],rasterized=plot_set_key['rasterized'],zorder=0,drawstyle='steps-mid',lw=plot_set_key['lw_plot'])
                    n_cosmics_vis_tot = np.sum(n_cosmics_vis,axis = 0)
                    n_cosmics_vis_tot=n_cosmics_vis_tot*y_range_loc[1]/np.max(n_cosmics_vis_tot)
                    plt.plot(gen_dic['wav_ord_inst'][inst][order_list],n_cosmics_vis_tot,linestyle='-',color='black',rasterized=plot_set_key['rasterized'],zorder=0,drawstyle='steps-mid',lw=plot_set_key['lw_plot'])
                    
                    #Plot frame  
                    if plot_set_key['title']:plt.title('Cosmics detections for visit '+vis+' in '+inst)
                    dx_range=x_range_loc[1]-x_range_loc[0]
                    dy_range=y_range_loc[1]-y_range_loc[0]
                    xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                    ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)  
                    custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,dir_y='out', 
                                xmajor_int=xmajor_int,xminor_int=xminor_int,xmajor_form=xmajor_form,
                                ymajor_int=ymajor_int,yminor_int=yminor_int,ymajor_form=ymajor_form,
                                x_title='Wavelength in input rest frame (A)',y_title='Cosmics occurrence',
                                font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                    plt.savefig(path_loc+'Occurrences.'+plot_dic['cosm_corr'])
                    plt.close()  









    ################################################################################################################ 
    #%%% Persistent peaks master
    ################################################################################################################ 
    if ('permpeak_corr' in plot_settings): 
        key_plot = 'permpeak_corr'
        plot_set_key = plot_settings[key_plot]

        print('-----------------------------------')
        print('+ Plotting master for persistent peaks correction')
        
        #Process original visits
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())): 

            #Process each visit
            for vis in np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst]): 
                
                #Create directory if required
                path_loc = gen_dic['save_plot_dir']+'Spec_raw/'+inst+'_'+vis+'_Permpeaks/'
                if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)                 
                data_vis=data_dic[inst][vis]

                #Upload data
                data_corr = np.load(gen_dic['save_data_dir']+'Corr_data/Permpeak/'+inst+'_'+vis+'_add.npz', allow_pickle=True)['data'].item()

                #Orders to plot
                order_list = plot_set_key['orders_to_plot'] if len(plot_set_key['orders_to_plot'])>0 else range(data_dic[inst]['nord']) 
                if (inst in gen_dic['permpeak_ord_corr']) and (vis in gen_dic['permpeak_ord_corr'][inst]) and (len(gen_dic['permpeak_ord_corr'][inst][vis])>0):
                    order_list = np.intersect1d(order_list,gen_dic['permpeak_ord_corr'][inst][vis])                        
                nord_list = len(order_list)
                for isub_ord,iord in enumerate(order_list):
                    plt.ioff()        
                    plt.figure(figsize=plot_set_key['fig_size'])

                    data_ord = data_corr[iord]
                    
                    #Master spectrum
                    plt.plot(data_ord['wav_mast'],data_ord['flux_mast'],linestyle='-',color='dodgerblue',rasterized=plot_set_key['rasterized'],zorder=0,lw=plot_set_key['lw_plot'] )

                    #Anchor points 
                    plt.plot(data_ord['wav_max'],data_ord['flux_max'],linestyle='',color='black',rasterized=plot_set_key['rasterized'],zorder=1,marker='o',markersize=plot_set_key['markersize'] )
                    
                    #Continuum
                    plt.plot(data_ord['wav_mast'],data_corr['cont_func_dic'][iord](data_ord['wav_mast']),linestyle='-',color='red',rasterized=plot_set_key['rasterized'],zorder=1,lw=plot_set_key['lw_plot'] )
                                        
                    #Plot boundaries
                    if plot_set_key['x_range'] is not None:x_range_loc = plot_set_key['x_range']
                    else:
                        xmin = data_ord['wav_mast'][0]
                        xmax = data_ord['wav_mast'][-1]
                        dx_range = xmax-xmin
                        x_range_loc = [xmin-0.05*dx_range,xmax+0.05*dx_range]  
                    if plot_set_key['y_range'] is not None:y_range_loc = plot_set_key['y_range'] 
                    else:
                        ymin = np.nanmin(data_ord['flux_max'])
                        ymax = np.nanmax(data_ord['flux_max'])
                        dy_range = ymax-ymin
                        y_range_loc = [ymin-0.05*dy_range,ymax+0.1*dy_range]                      
                    dx_range=x_range_loc[1]-x_range_loc[0]
                    dy_range=y_range_loc[1]-y_range_loc[0]
                      
                    #Plot frame  
                    if plot_set_key['title']:plt.title('Master for persistent peaks correction for visit '+vis+' in '+inst)
                    xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                    ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)  
                    custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,dir_y='out', 
                                xmajor_int=xmajor_int,xminor_int=xminor_int,xmajor_form=xmajor_form,
                                ymajor_int=ymajor_int,yminor_int=yminor_int,ymajor_form=ymajor_form,
                                x_title='Wavelength in Earth rest frame (A)',y_title='Flux',
                                font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                    if  gen_dic['type'][inst]=='spec1D':str_add=''
                    elif  gen_dic['type'][inst]=='spec2D':str_add='iord'+str(iord)
                    plt.savefig(path_loc+str_add+'.'+plot_dic['permpeak_corr'])
                    plt.close()                     
                






    ################################################################################################################ 
    #%%% Fringing correction
    ################################################################################################################ 
    if ('fring_corr' in plot_settings): 
        key_plot = 'fring_corr'
        plot_set_key = plot_settings[key_plot] 
        
        print('-----------------------------------')
        print('   > Plotting fringing correction')
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())): 
            for vis in np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst]): 
                
                #Create directory if required
                path_loc = gen_dic['save_plot_dir']+'Spec_raw/'+inst+'_'+vis+'_Fringing/'
                if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)                 
                data_vis=data_dic[inst][vis]

                #Plot each order for each exposure
                for iexp in range(data_vis['n_in_visit']):
                    data_corr = np.load(gen_dic['save_data_dir']+'Corr_data/Fring/'+ gen_dic['type'][inst]+'_'+inst+'_'+vis+'_'+str(iexp)+'_add.npz', allow_pickle=True)
                    data_exp = np.load(data_vis['proc_DI_data_paths']+str(iexp)+'.npz',allow_pickle=True)['data'].item()     
                    for iord in range(data_dic[inst]['nord_ref']):
                        if data_corr['exp_ord_defring'][iord]==True:
                            idxdef_ov = data_corr['idx_ov_ord_def'][iord]
                            low_wav_ov = data_exp['edge_bins'][iord,idxdef_ov]
                            high_wav_ov=data_exp['edge_bins'][iord,idxdef_ov+1]
                            wav_ov=data_exp['cen_bins'][iord,idxdef_ov]
                                                        
                            plt.ioff()        
                            fig = plt.figure(figsize=plot_set_key['fig_size'])
                                
                            #Plot boundaries
                            x_range_loc = plot_set_key['x_range'] if plot_set_key['x_range'] is not None else [low_wav_ov[0],high_wav_ov[-1]]
                                
                            #Ratio between two successive orders over their overlapping range
                            if plot_set_key['plot_err']:plt.errorbar(wav_ov,data_corr['data_fit'][iord],yerr=data_corr['data_fit'][iord],linestyle='-',color='dodgerblue',rasterized=plot_set_key['rasterized'],zorder=0)
                            else:plt.plot(wav_ov,data_corr['data_fit'][iord],linestyle='-',color='dodgerblue',rasterized=plot_set_key['rasterized'],zorder=0)
                            
                            #Plot binned data
                            if (inst in plot_set_key['bin_data']):
                                nw_bin=int((high_wav_ov[-1]-low_wav_ov[0])/plot_set_key['bin_data'][inst])+1
                                dw_bin = (high_wav_ov[-1]-low_wav_ov[0])/nw_bin
                                low_w_bin =low_wav_ov[0]+dw_bin*np.arange(nw_bin)   
                                high_w_bin = np.append(low_w_bin[1::],low_w_bin[-1]+dw_bin)
                                _,_,w_bin,_,ratio_ov_bin,eratio_ov_bin = resample_func(low_w_bin,high_w_bin,low_wav_ov,high_wav_ov,data_corr['data_fit'][iord],data_corr['sig_fit'][iord],remove_empty=True,dim_bin=0,cond_olap=1e-14)
                                if plot_set_key['plot_err']:plt.errorbar(w_bin,ratio_ov_bin,yerr=eratio_ov_bin,linestyle='',color='black',rasterized=plot_set_key['rasterized'],zorder=1,marker='o',markersize=2)
                                else:plt.plot(w_bin,ratio_ov_bin,linestyle='',color='black',rasterized=plot_set_key['rasterized'],zorder=1,marker='o',markersize=2)
                            
                            #Best-fit model
                            plt.plot(wav_ov,data_corr['fring_mod'][iord],linestyle='-',color='red',rasterized=plot_set_key['rasterized'],zorder=1)    
                                     
                            #Plot frame  
                            if plot_set_key['title']:plt.title('Fringing correction for visit '+vis+' in '+inst)
                            custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=plot_set_key['y_range'],dir_y='out', 
                        		    # xmajor_int=1000,xminor_int=100.,
            #                        ymajor_int=0.005,yminor_int=0.001,
                    #    		    xmajor_form='%.3f',ymajor_form='%.4f',
                        		    # xmajor_form='%.4f',ymajor_form='%i',
                        		    x_title='Wavelength (A)',y_title='(F/F$_\mathrm{next}$)',
                                        font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                            if  gen_dic['type'][inst]=='spec1D':str_add=''
                            elif  gen_dic['type'][inst]=='spec2D':str_add='_iord'+str(iord)
                            plt.savefig(path_loc+'idx'+str(iexp)+str_add+'.'+plot_dic['fring_corr'])
                            plt.close()                     
           







    ##################################################################################################
    #%% Disk-integrated profiles 
    ##################################################################################################   

    ##################################################################################################
    #%%% 2D maps
    ##################################################################################################  

    ##################################################################################################
    #%%%% Original profiles
    ##################################################################################################
    if ('map_DI_prof' in plot_settings):  
        key_plot = 'map_DI_prof'
        
        print('-----------------------------------')
        print('+ 2D map : disk-integrated stellar profiles')   

        #Plot map
        sub_2D_map(key_plot,plot_dic[key_plot],plot_settings[key_plot])




    ##################################################################################################
    #%%%% Binned profiles
    ##################################################################################################
    if ('map_DIbin' in plot_settings):
        key_plot = 'map_DIbin'

        print('-----------------------------------')
        print('+ 2D map : binned disk-integrated stellar profiles') 

        #Plot map
        sub_2D_map(key_plot,plot_dic[key_plot],plot_settings[key_plot])
                   



    ##################################################################################################
    #%%%% 1D converted spectra
    ##################################################################################################
    if ('map_DI_1D' in plot_settings):
        key_plot = 'map_DI_1D'
        
        print('-----------------------------------')
        print('+ 2D map : 1D disk-integrated stellar profiles') 

        #Plot map
        sub_2D_map(key_plot,plot_dic[key_plot],plot_settings[key_plot])
                   






    ##################################################################################################
    #%%% Individual profiles
    ##################################################################################################   

    ##################################################################################################
    #%%%% Original spectra (correction steps) 
    ##################################################################################################
    if gen_dic['specINtype'] and (plot_dic['sp_raw']!=''):
        key_plot = 'DI_prof'
        
        print('-----------------------------------')
        print('+ Individual disk-integrated spectra')
        sub_plot_prof(plot_settings[key_plot],key_plot,plot_dic['sp_raw'])               
                        



    ##################################################################################################
    #%%%% Original transmission spectra (correction steps)
    ##################################################################################################
    if ('trans_sp' in plot_settings):
        key_plot = 'trans_sp'
        
        print('-----------------------------------')
        print('+ Individual disk-integrated transmission spectra')
        sub_plot_DI_trans(plot_settings[key_plot],key_plot,plot_dic[key_plot])               
                        








    ##################################################################################################
    #%%%% Original profiles (processing steps) 
    ##################################################################################################
    for key_plot in ['DI_prof','DI_prof_res']:
        if (key_plot in plot_settings):

            ##############################################################################
            #%%%%% Raw profile and its fit in their original rest frame (typically heliocentric)
            if (key_plot=='DI_prof'):
                print('-----------------------------------')
                print('+ Individual disk-integrated profiles')
    
            ##############################################################################
            #%%%%% Residuals between the raw CCFs and their fit
            if (key_plot=='DI_prof_res') and (gen_dic['fit_DI']):
                print('-----------------------------------')
                print('+ Individual residuals from disk-integrated profiles')
    
            ##############################################################################
            #%%%%% Plot       
            sub_plot_prof(plot_settings[key_plot],key_plot,plot_dic[key_plot])











    ##################################################################################################
    #%%%% Binned profiles
    ##################################################################################################
    for key_plot in ['DIbin','DIbin_res']:
        if (key_plot in plot_settings):

            ##############################################################################
            #%%%%% Plot each disk-integrated profiles and its fit
            if (key_plot == 'DIbin'):               
                print('-----------------------------------')
                print('+ Individual binned disk-integrated profiles')
                
            ##############################################################################
            #%%%%% Plot residuals between the binned disk-integrated profiles and their fit
            if (key_plot == 'DIbin_res'):   
                print('-----------------------------------')
                print('+ Individual residuals from binned disk-integrated profiles')
    
            ##############################################################################
            #%%%%% Plot  
            sub_plot_prof(plot_settings[key_plot],key_plot,plot_dic[key_plot])









    ##################################################################################################
    #%%%% 1D converted spectra
    ##################################################################################################
    if ('sp_DI_1D' in plot_settings):
        key_plot = 'sp_DI_1D' 
      
        print('-----------------------------------')
        print('+ Individual 1D disk-integrated spectra')
        sub_plot_prof(plot_settings[key_plot],key_plot,plot_dic[key_plot])             
          



    ##################################################################################################
    #%%%% Aligned profiles (all)
    ##################################################################################################   
    if ('all_DI_data' in plot_settings):  
        key_plot = 'all_DI_data'

        print('-----------------------------------')
        print('+ Profiles aligned in star rest frame')
        sub_plot_all_prof(plot_settings[key_plot],'DI',plot_dic['all_DI_data'])                 
                                            
                    
            





    
    ##################################################################################################
    #%%% Properties of raw data and disk-integrated profiles
    ##################################################################################################
    if (plot_dic['prop_DI']!=''):
        print('-----------------------------------')
        print('+ Properties of raw data and disk-integrated profiles')

        #%%%% Processing properties
        for plot_prop in plot_settings['prop_DI_ordin']:
            key_plot = 'prop_DI_'+plot_prop 
            txt_print = {'rv':'RV','rv_pip':'RV pipeline','rv_res':'RV residuals','rv_pip_res':'RV pipeline residuals','RVdrift':'RV drift','rv_l2c':'RV lobe-to-core difference','RV_lobe':'Lobe RV',
                         'FWHM':'FWHM','FWHM_pip':'FWHM pipeline','FWHM_voigt':'FWHM of Voigt profile','FWHM_l2c':'FWHM lobe-to-core ratio','FWHM_lobe':'Lobe FWHM','FWHM_ord0__IS__VS_':'Local FWHM (deg. 0)','EW':'Equivalent width','vsini':'Projected rotational velocity',
                         'ctrst':'Contrast','ctrst_pip':'Contrast pipeline','true_ctrst':'True contrast','ctrst_ord0__IS__VS_':'Local contrast (deg. 0)','amp':'Amplitude','amp_l2c':'Amplitude lobe-to-core ratio','amp_lobe':'Lobe amplitude','area':'Area',
                         'cont':'Continuum level','biss_span':'Bissector span','ha':'Ha index','na':'Na index','ca':'Ca index','s':'S index','rhk':'Rhk index',
                         'phase':'Phase','mu':'Center-to-Limb angle','lat':'Stellar latitude','lon':'Stellar longitude','x_st':'Stellar X coordinate','y_st':'Stellar Y coordinate',
                         'AM':'Airmass','flux_airmass':'Airmass absorption','seeing':'Seeing','snr':'SNR','snr_quad':'Quadratic SNR (ESPRESSO)','snr_R':'SNR ratio','colcorrmin':r'C$_\mathrm(corr}^\mathrm{min}$', 'colcorrmax':r'C$_\mathrm(corr}^\mathrm{max}$', 'colcorrR':r'C$_\mathrm(corr}^\mathrm{ratio}$','colcorr450':r'C$_\mathrm(corr}^\mathrm{450nm}$', 'colcorr550':r'C$_\mathrm(corr}^\mathrm{550nm}$', 'colcorr650':r'C$_\mathrm(corr}^\mathrm{650nm}$',\
                         'PSFx':'', 'PSFy':'','PSFr':'','PSFang':'',
                         'glob_flux_sc':'Global flux scaling','satur_check':'Saturation check','ADC1 POS':'ESPRESSO ADC1 Position','ADC1 RA':'ESPRESSO ADC1 Right Ascension','ADC1 DEC':'ESPRESSO ADC1 Declination','ADC2 POS':'ESPRESSO ADC2 Position','ADC2 RA':'ESPRESSO ADC2 Right Ascension','ADC2 DEC':'ESPRESSO ADC2 Declination',
                         'alt':'Telescope altitude','az':'Azimuth angle','TILT1 VAL1':'ESPRESSO TILT1 VAL1','TILT1 VAL2':'ESPRESSO TILT1 VAL2','TILT2 VAL1':'ESPRESSO TILT2 VAL1','TILT2 VAL2':'ESPRESSO TILT2 VAL2'}
            for ideg in range(1,5):txt_print['c'+str(ideg)+'_pol']=r'Polynomial continuum (deg. '+str(ideg)+')'
            for ideg in range(0,11):txt_print['PC'+str(ideg)]=r'Coefficient PCA (deg. '+str(ideg)+')'
            print('   ---------------')
            print('   > '+txt_print[plot_prop])

            #%%%%% Plot   
            sub_plot_CCF_prop(plot_prop,plot_settings[key_plot],'DI')  
            		 
                
            
        



















    ##################################################################################################
    #%% Stellar CCF mask
    ################################################################################################## 

    ##################################################################################################
    #%%% Spectrum
    ##################################################################################################
    for key_plot in ['DImask_spectra','Intrmask_spectra']:
        if key_plot in plot_settings:    
            plot_set_key = plot_settings[key_plot]  

            ##############################################################################
            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_spectra'):       
        
                print('-----------------------------------')
                print('+ Disk-integrated mask : spectra')
                
            ##############################################################################
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_spectra'):

                print('-----------------------------------')
                print('+ Intrinsic mask : spectra')
            
            ##############################################################################
            #%%%% Plot 
            data_type_gen=key_plot.split('mask')[0]
            key_step = plot_set_key['step']
            for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())): 
                vis_det=list(data_dic[inst].keys()) if data_dic[inst]['n_visits_inst']==1 else 'binned'
                data_paths = 'CCF_masks_'+data_type_gen+'/'+gen_dic['add_txt_path'][data_type_gen]+'/'+inst+'_'+vis_det+'/'
    
                #Create directory if required
                path_loc = gen_dic['save_plot_dir']+data_paths
                if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)                 
                plt.ioff()        
                fig = plt.figure(figsize=plot_set_key['fig_size'])
                ax=plt.gca()
    
                #Retrieve plot dictionary
                plot_info = dataload_npz(gen_dic['save_data_dir']+data_paths+'Plot_info')
    
                #Plot frame 
                if plot_set_key['x_range'] is not None:
                    x_range_loc=plot_set_key['x_range'] 
                    cond_in_range = (plot_info['cen_bins_reg']>=x_range_loc[0]) & (plot_info['cen_bins_reg']<=x_range_loc[1]) 
                else:
                    cond_def = (plot_info['flux_norm_reg']!=0.)
                    x_range_loc = np.array([plot_info['cen_bins_reg'][cond_def][0],plot_info['cen_bins_reg'][cond_def][-1]])
                    cond_in_range = np.repeat(True,len(plot_info['cen_bins_reg']))
                if plot_set_key['y_range'] is not None:
                    y_range_loc=plot_set_key['y_range'] 
                else:
                    y_range_loc = np.array([np.min(plot_info['flux_norm_reg'][cond_in_range]),np.max(plot_info['flux_norm_reg'][cond_in_range])])
                dx_range=x_range_loc[1]-x_range_loc[0]
                if plot_set_key['x_range'] is None:
                    x_range_loc[0]-=0.05*dx_range
                    x_range_loc[1]+=0.05*dx_range
                    dx_range=x_range_loc[1]-x_range_loc[0]
                dy_range=y_range_loc[1]-y_range_loc[0]
                if plot_set_key['y_range'] is None:
                    y_range_loc[0]-=0.05*dy_range
                    y_range_loc[1]+=0.05*dy_range
                    dy_range=y_range_loc[1]-y_range_loc[0]
    
                #Resampling
                if plot_set_key['resample'] is not None:
                    n_reg = int(np.ceil((x_range_loc[1]-x_range_loc[0])/plot_set_key['resample']))
                    edge_bins_reg = np.linspace(x_range_loc[0],x_range_loc[1],n_reg)
                    cen_bins_reg = 0.5*(edge_bins_reg[0:-1]+edge_bins_reg[1::]) 
    
                #Smoothed regular normalized spectrum
                if plot_set_key['plot_norm_reg']:
                    ax.plot(plot_info['cen_bins_reg'][cond_in_range],plot_info['flux_norm_reg'][cond_in_range],color='black',linestyle='-',lw=plot_set_key['lw_plot'],zorder=10,rasterized=plot_set_key['rasterized'],alpha = plot_set_key['alpha_symb'])  
    
                    #Resampled spectrum
                    if plot_set_key['resample'] is not None:
                        dbins_loc = plot_info['cen_bins_reg'][1]-plot_info['cen_bins_reg'][0]
                        edge_bins_loc = np.concatenate(([plot_info['cen_bins_reg'][0]-0.5*dbins_loc], 0.5*(plot_info['cen_bins_reg'][0:-1]+plot_info['cen_bins_reg'][1::]),[plot_info['cen_bins_reg'][-1]+0.5*dbins_loc])) 
                        var_resamp = bind.resampling(edge_bins_reg,edge_bins_loc,plot_info['flux_norm_reg'], kind=gen_dic['resamp_mode'])   
                        ax.plot(cen_bins_reg,var_resamp,color='black',linestyle='-',lw=plot_set_key['lw_plot'],rasterized=plot_set_key['rasterized'],zorder=11,drawstyle=plot_set_key['drawstyle'])                      
                              
                
                #--------------------------------------------------------                            
                #Continuum normalization
                if key_step=='cont':
       
                    #Original spectrum with continuum
                    if plot_set_key['plot_raw']:
                        data_mast = dataload_npz(gen_dic['save_data_dir']+data_type_gen+'bin_data/'+gen_dic['add_txt_path'][data_type_gen]+inst+'_'+vis_det+'_'+data_dic[data_type_gen]['dim_bin']+str(0))
                        cond_def = data_mast['cond_def'][0]
                        cen_bins = data_mast['cen_bins'][0,cond_def]
                        flux_mast = data_mast['flux'][0,cond_def]
                        ax.plot(cen_bins,flux_mast,color='dodgerblue',linestyle='-',lw=plot_set_key['lw_plot'],zorder=0,rasterized=plot_set_key['rasterized'],alpha = plot_set_key['alpha_symb'])    
    
                        #Resampled spectrum
                        if plot_set_key['resample'] is not None:
                            var_resamp = bind.resampling(edge_bins_reg, data_mast['edge_bins'][0],data_mast['flux'][0], kind=gen_dic['resamp_mode'])   
                            ax.plot(cen_bins_reg,var_resamp,color='dodgerblue',linestyle='-',lw=plot_set_key['lw_plot'],rasterized=plot_set_key['rasterized'],zorder=1,drawstyle=plot_set_key['drawstyle'])                      
                                                    
                        #Continuum
                        cont_norm = dataload_npz(gen_dic['save_data_dir']+'Stellar_cont_DI/'+inst+'_'+vis_det+'/St_cont')['cont_func_dic'](data_mast['cen_bins'][0])
                        ax.plot(cen_bins,cont_norm[cond_def],color='red',linestyle='-',lw=plot_set_key['lw_plot']+0.5,zorder=7,rasterized=plot_set_key['rasterized']) 
    
                    #Normalized spectrum
                    if plot_set_key['plot_norm']:
                        flux_mast_norm=flux_mast/cont_norm[cond_def]
                        ax.plot(cen_bins,flux_mast_norm,color='grey',linestyle='-',lw=plot_set_key['lw_plot'],zorder=3,rasterized=plot_set_key['rasterized'],alpha = plot_set_key['alpha_symb'])              
    
                        #Resampled spectrum
                        if plot_set_key['resample'] is not None:
                            var_resamp = bind.resampling(edge_bins_reg, data_mast['edge_bins'][0],data_mast['flux'][0]/cont_norm, kind=gen_dic['resamp_mode'])   
                            ax.plot(cen_bins_reg,var_resamp,color='grey',linestyle='-',lw=plot_set_key['lw_plot'],rasterized=plot_set_key['rasterized'],zorder=4,drawstyle=plot_set_key['drawstyle'])                      
                              
                #--------------------------------------------------------
                else:
                    plot_info_step = plot_info[key_step]
        
                    #Number of lines in current iteration              
                    if plot_set_key['print_nl']:
                        ax.text(x_range_loc[0]+0.1*dx_range,y_range_loc[0]+0.1*dy_range,'Mask lines pre/post ='+str(plot_info_step['nl_mask_pre'])+'/'+str(plot_info_step['nl_mask_post']),verticalalignment='bottom', horizontalalignment='left',fontsize=plot_set_key['font_size']-2.,zorder=4,color='black') 
    
                    #Extrema
                    cond_in_plot = (plot_info_step['w_maxima_left']>x_range_loc[0])  & (plot_info_step['w_maxima_right']<x_range_loc[1])
                    if True in cond_in_plot:
                        ax.plot(plot_info_step['w_lines'][cond_in_plot],plot_info_step['f_minima'][cond_in_plot],markersize=plot_set_key['markersize'],color='green',markeredgecolor='white',zorder=100,marker='o',ls='',markeredgewidth=0.5,rasterized=plot_set_key['rasterized'])
                        ax.plot(plot_info_step['w_maxima_left'][cond_in_plot],plot_info_step['f_maxima_left'][cond_in_plot],markersize=plot_set_key['markersize'],color='dodgerblue',markeredgecolor='white',zorder=100,marker='o',ls='',markeredgewidth=0.5,rasterized=plot_set_key['rasterized'])
                        ax.plot(plot_info_step['w_maxima_right'][cond_in_plot],plot_info_step['f_maxima_right'][cond_in_plot],markersize=plot_set_key['markersize'],color='red',markeredgecolor='white',zorder=102,marker='o',ls='',markeredgewidth=0.5,rasterized=plot_set_key['rasterized'])
                
                    #Line selection with depth and width criteria
                    if key_step=='sel1':
    
                        #Plot line depth selection range
                        ax.plot(x_range_loc,np.repeat(1.-plot_info_step['linedepth_cont_min'],2),color='magenta',linestyle='--',lw=plot_set_key['lw_plot']+1,zorder=10,rasterized=plot_set_key['rasterized'])               
                        ax.plot(x_range_loc,np.repeat(1.-plot_info_step['linedepth_cont_max'],2),color='magenta',linestyle='--',lw=plot_set_key['lw_plot']+1,zorder=10,rasterized=plot_set_key['rasterized'])               
    
                    #Line selection with telluric contamination
                    elif key_step=='sel3':
    
                        #Plot minimum telluric depth to be considered
                        if plot_set_key['tell_depth_min']:
                            ax.plot(x_range_loc,np.repeat(1.-plot_info_step['tell_depth_min'],2),color='magenta',linestyle='--',lw=plot_set_key['lw_plot']+1,zorder=10,rasterized=plot_set_key['rasterized'])               
                        
                        #Telluric spectrum at minimum / maximum positions
                        ax.plot(plot_info['cen_bins_reg']*plot_info_step['min_dopp_shift'],plot_info_step['spectre_t'],color='dodgerblue',lw=plot_set_key['lw_plot'],zorder=10,rasterized=plot_set_key['rasterized'])    
                        ax.plot(plot_info['cen_bins_reg']*plot_info_step['max_dopp_shift'],plot_info_step['spectre_t'],color='red',lw=plot_set_key['lw_plot'],zorder=10,rasterized=plot_set_key['rasterized'])    
    
                        #Contaminating telluric lines
                        for wave_t,flux_t in zip(plot_info_step['w_tell_contam'],plot_info_step['f_tell_contam']):
                            ax.plot(wave_t,flux_t,marker='o',markersize=plot_set_key['markersize'],color='grey',markeredgecolor='white',markeredgewidth=0.5,rasterized=plot_set_key['rasterized'])
                            ax.plot([wave_t,wave_t],y_range_loc,linestyle=':',color='grey')        
    
                    #Final line selection (RV dispersion and telluric contaminatin)
                    elif key_step=='sel6': 
                        
                        #Line ranges
                        if plot_set_key['line_ranges']:
                            for iline,(wl_loc,hrange_loc) in enumerate(zip(plot_info_step['w_lines'],plot_info_step['line_hrange'])):
                                ax.axvline(x=wl_loc,color='limegreen',alpha=0.6,ls='--',lw=1)
                                ax.axvspan(xmin=wl_loc-hrange_loc,xmax=wl_loc+hrange_loc,color='grey',alpha=0.1,ls='')     
    
                        #Matching VALD lines
                        if plot_set_key['vald_sp']:
                            f_minima_vald = 1.-plot_info['depth_vald_corr']
                            cond_match = (~np.isnan(plot_info['wave_vald'])) & (plot_info['wave_vald']>x_range_loc[0]) & (plot_info['wave_vald']<x_range_loc[1])
                            for wave_vald,f_min_vald,spec_vald in zip(plot_info['wave_vald'][cond_match],f_minima_vald[cond_match],plot_info['species'][cond_match]):
                                
                                #Closest line
                                #    - matching was performed before last selection steps
                                idx_stl = closest(plot_info_step['w_lines'],wave_vald)
                                if abs(wave_vald-plot_info_step['w_lines'][idx_stl])<plot_info_step['line_hrange'][idx_stl]:
                                    ax.plot(wave_vald,f_min_vald,marker='o',markersize=plot_set_key['markersize'],color='goldenrod',markeredgecolor='white',markeredgewidth=0.5,rasterized=plot_set_key['rasterized'])
                                    sp_name = spec_vald if type(spec_vald)==str else '?'
                                    ax.text(wave_vald,1.03,sp_name,verticalalignment='center', horizontalalignment='center',fontsize=plot_set_key['font_size']-1.,zorder=100,color='goldenrod',rasterized=plot_set_key['rasterized']) 
                                                        
    
                #--------------------------------------------------------  
                #Plot rejection ranges
                if plot_set_key['line_rej_range']:
                    plot_shade_range(ax,plot_info['line_rej_range'],x_range_loc,y_range_loc,mode='fill',zorder=4)    
    
                #Plot unity level
                ax.axhline(1.,color='black', lw=plot_set_key['lw_plot'],linestyle='--',zorder=10,rasterized=plot_set_key['rasterized'])              
    
                #Frame
                xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range) 
                if data_type_gen=='DI':txt_rest='star'                 
                elif data_type_gen=='Intr':txt_rest='surface'    
                custom_axis(plt,ax=ax,position=plot_set_key['margins'] ,x_range=x_range_loc,y_range=y_range_loc,dir_y='out',dir_x='out', 
                		    xmajor_int=xmajor_int,xminor_int=xminor_int,xmajor_form=xmajor_form,
                		    ymajor_int=ymajor_int,yminor_int=yminor_int,ymajor_form=ymajor_form,
                            x_title='Wavelength in '+txt_rest+' rest frame (A)',y_title='Normalized flux',
                         font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                plt.savefig(path_loc+'Spectrum_'+plot_set_key['step']+'.'+plot_dic[data_type_gen+'mask_spectra'])                        
                plt.close()   
      

    ##################################################################################################
    #%%% Line depth range selection
    ##################################################################################################
    for key_plot in ['DImask_spectra','Intrmask_spectra']:
        if key_plot in plot_settings:
            plot_set_key = plot_settings[key_plot]  
            
            ##############################################################################
            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_ld'):      
            
                print('-----------------------------------')
                print('+ Disk-integrated mask : line depth range selection')
                
            ##############################################################################
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_ld'):
    
                print('-----------------------------------')
                print('+ Intrinsic mask : line depth range selection')
    
            #%%%% Plot
            for dist_info in plot_settings[key_plot]['dist_info']:dist2D_stlines_CCFmasks(dist_info,plot_settings[key_plot],key_plot,plot_dic[key_plot])  
            
            
            
        
    ##################################################################################################
    #%%% Line depth and width selection
    ##################################################################################################
    for key_plot in ['DImask_ld_lw','Intrmask_ld_lw']:
        if key_plot in plot_settings:    

            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_ld_lw'):
                print('-----------------------------------')
                print('+ Disk-integrated mask : line depth and width selection')

            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_ld_lw'):
                print('-----------------------------------')
                print('+ Intrinsic mask : line depth and width selection')
                
            #%%%% Plot
            for dist_info in plot_settings[key_plot]['dist_info']:dist2D_stlines_CCFmasks(dist_info,plot_settings[key_plot],key_plot,plot_dic[key_plot])        
            




    ##################################################################################################
    #%%% Line position selection
    ##################################################################################################
    for key_plot in ['DImask_RVdev_fit','Intrmask_RVdev_fit']:
        if key_plot in plot_settings: 

            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_RVdev_fit'):              
                print('-----------------------------------')
                print('+ Disk-integrated mask : line position selection')
                
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_RVdev_fit'):
                print('-----------------------------------')
                print('+ Intrinsic mask : line position selection')
                
            #%%%% Plot
            for dist_info in plot_settings[key_plot]['dist_info']:dist1D_stlines_CCFmasks(dist_info,plot_settings[key_plot],key_plot,plot_dic[key_plot])



    ##################################################################################################
    #%%% Telluric selection
    ##################################################################################################
    for key_plot in ['DImask_tellcont','Intrmask_tellcont']:
        if key_plot in plot_settings:      

            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_tellcont'):
                print('-----------------------------------')
                print('+ Disk-integrated mask : telluric selection')
                
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_tellcont'):
                print('-----------------------------------')
                print('+ Intrinsic mask : telluric selection')

            #%%%% Plot
            for dist_info in plot_settings[key_plot]['dist_info']:
                dist1D_stlines_CCFmasks(dist_info,plot_settings[key_plot],key_plot,plot_dic[key_plot])
                dist1D_stlines_CCFmasks(dist_info,plot_settings[key_plot],key_plot+'_final',plot_dic[key_plot])
            
        

    ##################################################################################################
    #%%% VALD line depth correction
    ##################################################################################################
    for key_plot in ['DImask_vald_depthcorr','Intrmask_vald_depthcorr']:
        if key_plot in plot_settings:
            plot_set_key = plot_settings[key_plot]   

            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_vald_depthcorr'):
                print('-----------------------------------')
                print('+ Disk-integrated mask : VALD selection')
                
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_vald_depthcorr'):
                print('-----------------------------------')
                print('+ Intrinsic mask : VALD selection')
                
            #%%%% Plot
            data_type_gen=key_plot.split('mask')[0]
            for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())): 
                vis_det=list(data_dic[inst].keys()) if data_dic[inst]['n_visits_inst']==1 else 'binned'
                data_paths = 'CCF_masks_'+data_type_gen+'/'+gen_dic['add_txt_path'][data_type_gen]+'/'+inst+'_'+vis_det+'/'
    
                #Create directory if required
                path_loc = gen_dic['save_plot_dir']+data_paths
                if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)                 
                
                #Retrieve plot dictionary
                plot_info = dataload_npz(gen_dic['save_data_dir']+data_paths+'Plot_info')['vald_depthcorr']
                
                #Species to plot
                vald_spec = list(plot_info.keys())
                if len(plot_set_key['spec_to_plot'])>0:vald_spec = np.intersect1d(plot_set_key['spec_to_plot'],vald_spec) 
                n_species = len(vald_spec)
                
                #Frame
                plt.ioff()
                if n_species==1:
                    nsub_col = 1 
                    nsub_rows = 1 
                else:
                    nsub_col = 2 
                    nsub_rows = int(np.ceil(n_species/nsub_col))
                nall = nsub_rows*nsub_col   #number of possible subplots
                fig, axes = plt.subplots(nsub_rows,nsub_col, figsize=plot_set_key['fig_size'])
                fig.subplots_adjust(top=plot_set_key['margins'][3],left=plot_set_key['margins'][0],right=plot_set_key['margins'][2],bottom=plot_set_key['margins'][1],wspace=0.45,hspace=0.55)
                for idx_plot,elem in enumerate(vald_spec):
                    irow = int(idx_plot/nsub_col)
                    icol = idx_plot%nsub_col
                    if n_species == 1:ax_loc = axes
                    else:
                        if nsub_rows==1:ax_loc = axes[icol]
                        else:ax_loc = axes[irow, icol]
                    ax_loc.title.set_text(elem)
                    plot_info_elem = plot_info[elem]
    
                    #Plotting difference in depth for fitted VALD lines
                    ax_loc.errorbar(plot_info_elem['vald_depth_fit'],plot_info_elem['line_minus_vald_depth_fit'],yerr=plot_info_elem['line_minus_vald_depth_err_fit'],linestyle='',color='black',rasterized=plot_set_key['rasterized'],zorder=1,marker='o',markersize=plot_set_key['rasterized'])      
     
                    #Depth correction model
                    depth_mod = np.linspace(0,1,50)
                    delta_depth_mod = np.polyval(plot_info_elem['deltadepth_mod_coeff'],depth_mod)
                    ax_loc.plot(depth_mod,delta_depth_mod,color='black')               
    
                    #Plotting difference in depth for single VALD lines matching one stellar line of current species 
                    #    - colored with increasing wavelengths
                    ax_loc.scatter(plot_info_elem['vald_depth'],plot_info_elem['line_minus_vald_depth'],c=plot_info_elem['w_lines'],cmap='jet',zorder=0)
                    
                    #No deviation level 
                    ax_loc.plot([0,1],[0,0],color='gray',alpha=0.5)
                    
                    #Frame
                    ax_loc.set_xlim([0,1])
                    if plot_set_key['y_range'] is not None:y_range_loc=plot_set_key['y_range'] 
                    else:
                        y_range_loc = np.array([np.min(plot_info_elem['line_minus_vald_depth_fit']-plot_info_elem['line_minus_vald_depth_err_fit']),np.max(plot_info_elem['line_minus_vald_depth_fit']+plot_info_elem['line_minus_vald_depth_err_fit'])])
                        dy_range=y_range_loc[1]-y_range_loc[0]
                        y_range_loc = [y_range_loc[0]-0.05*dy_range,y_range_loc[1]+0.05*dy_range]     
                    ax_loc.set_ylim(y_range_loc)
    
                #--------------------------------------------------------
                #Fill remaining space with empty axes
                for isub_comp in range(idx_plot+1,nall):
                    irow = int(isub_comp/nsub_col)
                    icol = isub_comp%nsub_col
                    if nsub_rows==1:axes[icol].axis('off')
                    else:axes[irow, icol].axis('off')
    
                #Frame
                fig.text(0.5,0.04,'VALD line_depth', ha='center')
                fig.text(0.04,0.5,r'$\Delta$ Line depth', va='center', rotation='vertical')
    
                plt.savefig(path_loc+'VALDdepthcorr'+'.'+plot_dic[data_type_gen+'mask_vald_depthcorr'])                        
                plt.close() 

   

    ##################################################################################################
    #%%% Morphological (asymmetry) selection
    ##################################################################################################
    for key_plot in ['DImask_morphasym','Intrmask_morphasym']:
        if key_plot in plot_settings:     
    
            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_morphasym'):
                print('-----------------------------------')
                print('+ Disk-integrated mask : morphological (asymmetry) selection')
                
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_morphasym'):
                print('-----------------------------------')
                print('+ Intrinsic mask : morphological (asymmetry) selection')
                
            #%%%% Plot
            for dist_info in plot_settings[key_plot]['dist_info']:dist2D_stlines_CCFmasks(dist_info,plot_settings[key_plot],key_plot,plot_dic[key_plot])
            


    ##################################################################################################
    #%%% Morphological (shape) selection
    ##################################################################################################
    for key_plot in ['DImask_morphshape','Intrmask_morphshape']:
        if key_plot in plot_settings:     

            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_morphshape'):
                print('-----------------------------------')
                print('+ Disk-integrated mask : morphological (shape) selection')
                
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_morphshape'):
                print('-----------------------------------')
                print('+ Intrinsic mask : morphological (shape) selection')
                
            #%%%% Plot
            for dist_info in plot_settings[key_plot]['dist_info']:dist2D_stlines_CCFmasks(dist_info,plot_settings[key_plot],key_plot,plot_dic[key_plot])
    


    ##################################################################################################
    #%%% RV dispersion selection
    ##################################################################################################
    for key_plot in ['DImask_RVdisp','Intrmask_RVdisp']:
        if key_plot in plot_settings:      

            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_RVdisp'):
                print('-----------------------------------')
                print('+ Disk-integrated mask : RV dispersion selection')
                
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_RVdisp'):
                print('-----------------------------------')
                print('+ Intrinsic mask : RV dispersion selection')
    
            #%%%% Plot
            for dist_info in plot_settings[key_plot]['dist_info']:
                dist2D_stlines_CCFmasks(dist_info,plot_settings[key_plot],key_plot,plot_dic[key_plot])
                dist1D_stlines_CCFmasks(dist_info,plot_settings[key_plot],key_plot,plot_dic[key_plot])
    
    





 

        
        


    ################################################################################################################  
    #%% Light curves
    ################################################################################################################         
    
    ################################################################################################################    
    #%%% Input light curves
    ################################################################################################################ 
    if ('input_LC' in plot_settings):
        key_plot = 'input_LC'
        plot_set_key = plot_settings[key_plot]   

        print('-----------------------------------')
        print('+ Input light curves')
                     
        path_loc = gen_dic['save_plot_dir']+'Light_curves/Input/'
        if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  

        #Plot for each spectral band
        if ('chrom' in data_dic['DI']['system_prop_sc']) and (not plot_set_key['achrom']):
            system_prop = data_dic['DI']['system_prop_sc']['chrom']
            idx_bands_plot = plot_set_key['idx_bands'] if len(plot_set_key['idx_bands'])>0 else range(system_prop['nw']) 
            chrom_mode = True
        else:
            system_prop = data_dic['DI']['system_prop_sc']['achrom']
            idx_bands_plot = [0]
            chrom_mode = False  
  
        for iband in idx_bands_plot:
            if chrom_mode:print('   > Band w = '+str(system_prop['w'][iband]))
            else:print('   > Achromatic band')
           
            #Plot for each instrument
            for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())):                           
                plt.ioff()                    
                fig = plt.figure(figsize=plot_set_key['fig_size'])
          
                #Plot all visits
                x_min=1.
                x_max=-1.
                y_min=1e100
                y_max=-1e100
                if (inst not in plot_set_key['color_dic'] ):plot_set_key['color_dic'][inst]={}
                for ivis,vis in enumerate(np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst])): 

                    #Visit color
                    if (inst!='binned') and (vis in plot_set_key['color_dic'][inst]):col_vis = plot_set_key['color_dic'][inst][vis]
                    else:col_vis = 'black'

                    #Reference planet for the visit
                    pl_ref=plot_set_key['pl_ref'][inst][vis]    
                    coord_vis = coord_dic[inst][vis][pl_ref]    
                    Tdepth = (system_prop[pl_ref][iband])**2.
                    if plot_set_key['lc_gap'] is None:vis_shift = ivis*0.3*Tdepth
                    else:vis_shift = ivis*plot_set_key['lc_gap']

                    #Upload light curve model
                    data_upload = dataload_npz(gen_dic['save_data_dir']+'Scaled_data/'+inst+'_'+vis+'_add')           
                        
                    #Imported light curve
                    if plot_set_key['plot_LC_imp'] and (data_dic['DI']['transit_prop'][inst][vis]['mode']=='imp'):
                        ph_imp=get_timeorbit(pl_ref ,coord_dic[inst][vis], data_upload['imp_LC'][0], system_param[pl_ref], 0.)[1]
                        plt.plot(ph_imp,data_upload['imp_LC'][iband]-vis_shift,color=col_vis,linestyle='--',lw=plot_set_key['lw_plot'])  
                        
                    #HR light curve
                    if plot_set_key['plot_LC_HR']:
                        ph_plot = data_upload['coord_HR'][pl_ref]['cen_ph']
                        x_min=min(np.min(ph_plot),x_min)
                        x_max=max(np.max(ph_plot),x_max)
                        var_plot = data_upload['LC_HR'][:,iband]-vis_shift
                        y_min=min(np.min(var_plot),y_min)
                        y_max=max(np.max(var_plot),y_max)
                        plt.plot(ph_plot,var_plot,color=col_vis,linestyle='-',lw=plot_set_key['lw_plot'])  
               
                        #Print visit names
                        if plot_set_key['plot_vis']:
                            plt.text(0.,np.min(var_plot)+0.3*(np.max(var_plot)-np.min(var_plot)),vis,verticalalignment='center', horizontalalignment='center',fontsize=10.,zorder=4,color=col_vis) 
                  
                    #Time averaged light-curve within each exposure
                    #    - circles are used for in-transit exposures
                    #    - squares are used for out-of-transit exposures
                    #    - plot with empty symbols indicate exposures with bad fit of local stellar profiles
                    if plot_set_key['plot_LC_exp']: 
                        x_min=min(np.min(coord_vis['st_ph']),x_min)
                        x_max=max(np.max(coord_vis['end_ph']),x_max)
                        LC_flux_band_all = data_upload['flux_band_all']
                        y_min=min(np.min(LC_flux_band_all[:,iband]-vis_shift),y_min)
                        y_max=max(np.max(LC_flux_band_all[:,iband]-vis_shift),y_max)
                        i_in=0
                        for iexp,(ph_loc,ph_dur_loc,flux_loc) in enumerate(zip(coord_vis['cen_ph'],coord_vis['ph_dur'],LC_flux_band_all[:,iband])):
    
                            #Exposures indexes (general above)
                            if plot_set_key['plot_expid']:                   
                                plt.text(ph_loc,flux_loc-vis_shift+0.1*Tdepth,str(iexp),verticalalignment='bottom', horizontalalignment='center',fontsize=4.,zorder=4,color=col_vis) 
    
                            #Observed exposure
                            markerfacecolor='white'
                            marker='s'
                            if iexp in gen_dic[inst][vis]['idx_in']:
                                marker='o'
                                if ('prof_fit_dic' in data_dic['Res'][inst][vis]) and (data_dic['Res'][inst][vis]['prof_fit_dic'][i_in]['detected']):markerfacecolor=col_vis
                                
                                #Exposures indexes (in-transit below)
                                if plot_set_key['plot_expid']:plt.text(ph_loc,flux_loc-vis_shift-0.1*Tdepth,str(i_in),verticalalignment='bottom', horizontalalignment='center',fontsize=4.,zorder=4,color=col_vis) 
                                i_in+=1                               
                            plt.errorbar(ph_loc,flux_loc-vis_shift,xerr=0.5*ph_dur_loc,color=col_vis,marker=marker,markersize=plot_set_key['markersize'],linestyle='',markerfacecolor=markerfacecolor)                

                #Axis ranges
                x_range_loc=plot_set_key['x_range'] if plot_set_key['x_range'] is not None else np.array([x_min-0.005,x_max+0.005])
                if plot_set_key['y_range'] is not None:y_range_loc=plot_set_key['y_range'] 
                else:y_range_loc=np.array([y_min-0.2*Tdepth,y_max+0.2*Tdepth]) 
    
                #Contacts for planets transiting in the visit
                for ivis,vis in enumerate(np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst])):
                    for ipl,pl_loc in enumerate(data_dic[inst][vis]['transit_pl']):
                        if pl_loc==pl_ref:contact_phases_vis = contact_phases[pl_ref]
                        else:
                            contact_times = coord_dic[inst][vis][pl_loc]['Tcenter']+contact_phases[pl_loc]*system_param[pl_loc]["period"]
                            contact_phases_vis = (contact_times-coord_dic[inst][vis][pl_ref]['Tcenter'])/system_param[pl_ref]["period"]
                        ls_pl = {0:':',1:'--'}[ipl]
                        for cont_ph in contact_phases_vis:
                            plt.plot([cont_ph,cont_ph],y_range_loc,color='black',linestyle=ls_pl,lw=plot_set_key['lw_plot'])

                #Plot frame                 
                if plot_set_key['title']:plt.title('Input light curves')
                dx_range=x_range_loc[1]-x_range_loc[0]
                dy_range=y_range_loc[1]-y_range_loc[0]
                xmajor_int,xminor_int,xmajor_form = autom_tick_prop(dx_range)
                ymajor_int,yminor_int,ymajor_form = autom_tick_prop(dy_range)               
                custom_axis(plt,position=plot_set_key['margins'] ,x_range=x_range_loc,y_range=y_range_loc,dir_y='out', 
                		     xmajor_int=xmajor_int,xminor_int=xminor_int,xmajor_form=xmajor_form,
                		     ymajor_int=ymajor_int,yminor_int=yminor_int,ymajor_form=ymajor_form,
                         x_title='Orbital phase',y_title='Flux',
                         font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                
                save_path = path_loc+inst
                if chrom_mode:save_path+='_idxband'+str(iband)
                else:save_path+='_achrom'
                plt.savefig(save_path+'.'+plot_dic['input_LC'])                        
                plt.close()  


    ##################################################################################################
    #%%% Effective scaling light curves     
    ##################################################################################################
    if ('spectral_LC' in plot_settings):
        key_plot = 'spectral_LC'
        plot_set_key = plot_settings[key_plot] 

        print('-----------------------------------')
        print('+ Effective spectral light curves')
        
        path_loc = gen_dic['save_plot_dir']+'Light_curves/Spectral/'
        if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  

        #Visit color
        n_wLC=len(plot_set_key['wav_LC'])
        if (len(plot_set_key['color_dic'])==0):
            cmap = plt.get_cmap('jet') 
            if n_wLC==1:color_dic={0:cmap(0)} 
            else:
                col_tab = cmap( np.arange(n_wLC)/(n_wLC-1.))
                color_dic={iw:col_tab[iw] for iw in range(n_wLC)}
        else:color_dic = plot_set_key['color_dic']

        #Plot for each instrument
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())):   

            #Plot for each visit
            for ivis,vis in enumerate(np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst])): 
                coord_vis = coord_dic[inst][vis]
                data_vis = data_dic[inst][vis]                           
                plt.ioff()                    
                fig = plt.figure(figsize=plot_set_key['fig_size'])

                #Reference planet for the visit
                pl_ref=plot_set_key['pl_ref'][inst][vis]

                #Plot each in-transit observed exposure
                y_min=1e100
                y_max=-1e100
                for i_exp,(ph_loc,ph_dur_loc) in enumerate(zip(coord_vis[pl_ref]['cen_ph'],coord_vis[pl_ref]['ph_dur'])):
                    scaled_data = dataload_npz(gen_dic['save_data_dir']+'Scaled_data/'+inst+'_'+vis+'_'+str(i_exp))
                    scaling_data = dataload_npz(gen_dic['save_data_dir']+'Scaled_data/'+inst+'_'+vis+'_scaling_'+str(i_exp))
                
                    #Plot flux scaling for each requested wavelength
                    for iwLC,wLC_loc in enumerate(plot_set_key['wav_LC']):
                        
                        #Wavelength closest to request in table of current exposure
                        idx_wLC_loc = closest_Ndim(scaled_data['cen_bins'],wLC_loc)

                        #Theoretical flux value at retrieved wavelength
                        flux_loc = 1. - scaling_data['loc_flux_scaling'](scaled_data['cen_bins'][idx_wLC_loc])    
                        plt.errorbar(ph_loc,flux_loc,xerr=0.5*ph_dur_loc,color=color_dic[iwLC],marker='s',markersize=plot_set_key['markersize'],linestyle='',markerfacecolor='white')                
                        y_min=min(flux_loc,y_min)
                        y_max=max(flux_loc,y_max)

                #Axis ranges
                x_range_loc=plot_set_key['x_range'] if plot_set_key['x_range'] is not None else np.array([np.min(coord_vis[pl_ref]['st_ph'])-0.005,np.max(coord_vis[pl_ref]['end_ph'])+0.005])
                y_range_loc=plot_set_key['y_range'] if plot_set_key['y_range'] is not None else np.array([0.995*y_min,1.005*y_max]) 
    
                #Plot requested wavelengths
                for iwLC,wLC_loc in enumerate(plot_set_key['wav_LC']):
                    dx_range = x_range_loc[1]-x_range_loc[0]
                    dy_range = y_range_loc[1]-y_range_loc[0]
                    plt.text(x_range_loc[0]+0.05*dx_range,y_range_loc[0]+(1.+iwLC)*dy_range/(n_wLC+1.),"{0:.0f}".format(wLC_loc),verticalalignment='center', horizontalalignment='left',fontsize=10.,zorder=4,color=color_dic[iwLC]) 
        
                #Contacts
                for cont_ph in contact_phases[pl_ref]:
                    plt.plot([cont_ph,cont_ph],y_range_loc,color='black',linestyle=':',lw=1.)                
                
                #Plot frame                 
                if plot_set_key['title']:plt.title('Spectral light curves')
                custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,dir_y='out', 
                		    xmajor_int=0.05,xminor_int=0.01,
                		    ymajor_int=0.01,yminor_int=0.001,
                		  xmajor_form='%.2f',ymajor_form='%.2f',
                         x_title='Orbital phase',y_title='Flux',
                         font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                plt.savefig(path_loc+inst+'_'+vis+'.'+plot_dic['spectral_LC'])                        
                plt.close()  




    ################################################################################################################  
    #%% Differential profiles
    ################################################################################################################        
        
    ################################################################################################################  
    #%%% 2D maps
    ################################################################################################################  

    ################################################################################################################  
    #%%%% Original profiles 
    ################################################################################################################  
    if ('map_Res_prof' in plot_settings):
        key_plot = 'map_Res_prof'

        print('-----------------------------------')
        print('+ 2D map : differential profiles')    

        #Plot map
        sub_2D_map(key_plot,plot_dic[key_plot],plot_settings[key_plot])
                   

                
        

    ################################################################################################################  
    #%%%%% Best-fit profiles 
    ################################################################################################################  
    if ('map_BF_Res_prof' in plot_settings):
        key_plot = 'map_BF_Res_prof'
        print('-----------------------------------')
        print('+ 2D map : best-fit differential profiles')    
        
        #Plot map
        sub_2D_map(key_plot,plot_dic[key_plot],plot_settings[key_plot])
        
        
        
        
    ################################################################################################################  
    #%%%%% Residual from best-fit profiles 
    ################################################################################################################  
    if ('map_BF_Res_prof_re' in plot_settings):
        key_plot = 'map_BF_Res_prof_re'
        
        print('-----------------------------------')
        print('+ 2D map : residuals from best-fit differential profiles') 
        
        #Plot map
        sub_2D_map(key_plot,plot_dic[key_plot],plot_settings[key_plot])


        
    ################################################################################################################ 
    #%%%%% Estimates and residual profiles
    ################################################################################################################ 
    for key_plot in ['map_Res_prof_clean_pl_est','map_Res_prof_clean_sp_est','map_Res_prof_unclean_sp_est','map_Res_prof_unclean_pl_est',
                     'map_Res_prof_clean_sp_res','map_Res_prof_clean_pl_res','map_Res_prof_unclean_sp_res','map_Res_prof_unclean_pl_res']:
        if (key_plot in plot_settings):
            ##############################################################################
            #%%%%%% Un-cleaned estimates
            if key_plot == 'map_Res_prof_unclean_sp_est':
                print('-----------------------------------')
                print('+ 2D map: un-cleaned theoretical spotted profiles') 
            
            if key_plot == 'map_Res_prof_unclean_pl_est':
                print('-----------------------------------')
                print('+ 2D map: un-cleaned theoretical planet-occulted profiles') 
            ##############################################################################
            #%%%%%% Cleaned estimates
            if key_plot == 'map_Res_prof_clean_sp_est':
                print('-----------------------------------')
                print('+ 2D map: cleaned theoretical spotted profiles') 
            
            if key_plot == 'map_Res_prof_clean_pl_est':
                print('-----------------------------------')
                print('+ 2D map: cleaned theoretical planet-occulted profiles') 
            ##############################################################################
            #%%%%%% Residuals    
            if key_plot == 'map_Res_prof_clean_sp_res':
                print('-----------------------------------')
                print('+ 2D map: residuals from cleaned spotted profiles') 
            if key_plot == 'map_Res_prof_clean_pl_res':
                print('-----------------------------------')
                print('+ 2D map: residuals from cleaned planet-occulted profiles') 
            if key_plot == 'map_Res_prof_unclean_sp_res':
                print('-----------------------------------')
                print('+ 2D map: residuals from un-cleaned spotted profiles') 
            if key_plot == 'map_Res_prof_unclean_pl_res':
                print('-----------------------------------')
                print('+ 2D map: residuals from un-cleaned planet-occulted profiles') 
            
            #Plot map
            sub_2D_map(key_plot,plot_dic[key_plot],plot_settings[key_plot])  
        
        
        
        
        
        
        
        
        
        



    ################################################################################################################ 
    #%%% Individual profiles
    ################################################################################################################  

    ################################################################################################################ 
    #%%%% Original profiles 
    ################################################################################################################ 
    if ('Res_prof' in plot_settings):
        key_plot = 'Res_prof'

        print('-----------------------------------')
        print('   > Individual residual profiles')
        sub_plot_prof(plot_settings[key_plot],key_plot,plot_dic[key_plot])







    ##################################################################################################
    #%%% PCA results
    ##################################################################################################
    if ('pca_ana' in plot_settings):        
        key_plot = 'pca_ana'
        plot_set_key = plot_settings[key_plot] 

        print('-----------------------------------')
        print('   > Plotting PCA')
      
        #Create directory if required
        path_loc = gen_dic['save_plot_dir']+'Res_data/PCA/'
        if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  
        
        #Process original visits        
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())): 
            
            #Process each visit
            for vis in np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst]):
                
                #Retrieve data
                data_upload = (np.load(gen_dic['save_data_dir']+'PCA_results/'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item())                           
                
                #------------------------------------------------                         
                #PC variances
                if plot_set_key['pc_var']:
                    plt.ioff()        
                    fig = plt.figure(figsize=plot_set_key['fig_size'])
                    y_min=1e100
                    y_max=-1e100
                    
                    #Colors
                    col_type = {'pre':'dodgerblue','post':'red','out':'black'}
    
                    #Fractions of data variance explained by each PC
                    #    - from PCA on residual matrix and noise matrix
                    for key in plot_set_key['var_list']:
                        if key in data_upload['eig_val_res']:
                            n_pc = len(data_upload['eig_val_res'][key])
                            var_res_explained = 100.*data_upload['eig_val_res'][key]/np.sum(data_upload['eig_val_res'][key])
                            plt.plot(np.arange(n_pc), var_res_explained,marker = 'o',color=col_type[key],lw=plot_set_key['lw_plot'],markersize=plot_set_key['markersize'],linestyle='-')
                            y_min=np.min([np.min(var_res_explained),y_min])
                            y_max=np.max([np.max(var_res_explained),y_max])                 
                            
                            var_noise_explained = 100.*data_upload['eig_val_noise'][key]/np.sum(data_upload['eig_val_noise'][key])
                            plt.plot(np.arange(n_pc), var_noise_explained,color=col_type[key],lw=plot_set_key['lw_plot'],linestyle='--')
                            y_min=np.min([np.min(var_noise_explained),y_min])
                            y_max=np.max([np.max(var_noise_explained),y_max]) 

                    #Plot frame  
                    x_range_loc=plot_set_key['x_range_var'] if plot_set_key['x_range_var'] is not None else np.array([0.,n_pc])
                    y_range_loc=plot_set_key['y_range_var'] if plot_set_key['y_range_var'] is not None else np.array([y_min,y_max])
                    if plot_set_key['title']:plt.title('PC variance for visit '+vis+' in '+inst)
                    xmajor_int,xminor_int,xmajor_form=autom_tick_prop(x_range_loc[1]-x_range_loc[0])
                    ymajor_int,yminor_int,ymajor_form=autom_tick_prop(y_range_loc[1]-y_range_loc[0]) 
                    custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,dir_y='out', 
                                xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,
                                xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                                x_title='PC',y_title='Variance explained (%)',
                                font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                    plt.savefig(path_loc+inst+'_'+vis+'_PCvar.'+plot_dic['pca_ana']) 
                    plt.close()                 

                #------------------------------------------------                         
                #RMS pre/post correction on fitted residual profile 
                if plot_set_key['pc_rms']:
                    plt.ioff()        
                    fig = plt.figure(figsize=plot_set_key['fig_size'])
                    y_min=1e100
                    y_max=-1e100
                    pl_ref=data_dic[inst][vis]['transit_pl'][0]
                    cen_ph=coord_dic[inst][vis][pl_ref]['cen_ph'] 

                    #Print RMS
                    plt.text(0.1,1.1,'RMS[fit] = '+"{0:.3e}".format(np.mean(data_upload['rms_full_res_matr'][0][data_upload['idx_pca']]))+' -> '+"{0:.3e}".format(np.mean(data_upload['rms_full_res_matr'][1][data_upload['idx_pca']])),verticalalignment='center', horizontalalignment='left',fontsize=10.,zorder=10,color='black',transform=plt.gca().transAxes)
                    cond_corr_in = (cen_ph[data_upload['idx_corr']]>contact_phases[pl_ref][0]) & (cen_ph[data_upload['idx_corr']]<contact_phases[pl_ref][3])
                    idx_corr_in = data_upload['idx_corr'][cond_corr_in]
                    idx_corr_out = data_upload['idx_corr'][~cond_corr_in]
                    plt.text(0.6,1.1,'RMS[in-tr] = '+"{0:.3e}".format(np.mean(data_upload['rms_full_res_matr'][0][idx_corr_in]))+' -> '+"{0:.3e}".format(np.mean(data_upload['rms_full_res_matr'][1][idx_corr_in])),verticalalignment='center', horizontalalignment='left',fontsize=10.,zorder=10,color='black',transform=plt.gca().transAxes)
                    
                    #Measured RMS
                    plt.plot(cen_ph[data_upload['idx_corr']][idx_corr_out], data_upload['rms_full_res_matr'][0][idx_corr_out],markerfacecolor='dodgerblue',marker = 's',color='dodgerblue',ls='',markersize=plot_set_key['markersize'])
                    plt.plot(cen_ph[data_upload['idx_corr']][idx_corr_out], data_upload['rms_full_res_matr'][1][idx_corr_out],markerfacecolor='red',marker = 's',color='red',ls='',markersize=plot_set_key['markersize'])
                    plt.plot(cen_ph[idx_corr_in], data_upload['rms_full_res_matr'][0][idx_corr_in],markerfacecolor='dodgerblue',marker = 'o',color='dodgerblue',ls='',markersize=plot_set_key['markersize'])
                    plt.plot(cen_ph[idx_corr_in], data_upload['rms_full_res_matr'][1][idx_corr_in],markerfacecolor='red',marker = 'o',color='red',ls='',markersize=plot_set_key['markersize'])
                    y_min=np.min([np.nanmin(data_upload['rms_full_res_matr']),y_min])
                    y_max=np.max([np.nanmax(data_upload['rms_full_res_matr']),y_max]) 
                    

                    #Contacts
                    y_range_loc=plot_set_key['y_range_rms'] if plot_set_key['y_range_rms'] is not None else np.array([y_min,y_max])
                    for cont_ph in contact_phases[pl_ref]:plt.plot([cont_ph,cont_ph],y_range_loc,color='black',linestyle=':',lw=plot_set_key['lw_plot'],zorder=0)

                    #Plot frame  
                    x_range_loc=plot_set_key['x_range_rms'] if plot_set_key['x_range_rms'] is not None else np.array([np.min(cen_ph),np.max(cen_ph)])
                    if plot_set_key['title']:plt.title('RMS for visit '+vis+' in '+inst)
                    xmajor_int,xminor_int,xmajor_form=autom_tick_prop(x_range_loc[1]-x_range_loc[0])
                    ymajor_int,yminor_int,ymajor_form=autom_tick_prop(y_range_loc[1]-y_range_loc[0]) 
                    custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,dir_y='out', 
                                xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,
                                xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                                x_title='Orbital phase',y_title='RMS',
                                font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                    plt.savefig(path_loc+inst+'_'+vis+'_PCrms.'+plot_dic['pca_ana']) 
                    plt.close()           

                #------------------------------------------------                         
                #Delta-BIC between best-fit PC model and null model
                #    - do not compare in-transit and fit data, as the number of fitted points is not the same
                if plot_set_key['pc_bic']:
                    plt.ioff()        
                    fig = plt.figure(figsize=plot_set_key['fig_size'])
                    y_min=1e100
                    y_max=-1e100
                    pl_ref=data_dic[inst][vis]['transit_pl'][0]
                    cen_ph=coord_dic[inst][vis][pl_ref]['cen_ph'] 
                    dBIC = data_upload['BIC_tab'] - data_upload['chi2null_tab'] 

                    #Print delta-BIC of out-of-transit exposures used for the PCA, and on corrected in-transit exposure
                    plt.text(0.1,1.1,'dBIC[fit] = '+"{0:.3f}".format(np.mean(dBIC[data_upload['idx_pca']])),verticalalignment='center', horizontalalignment='left',fontsize=10.,zorder=10,color='black',transform=plt.gca().transAxes)
                    cond_corr_in = (cen_ph[data_upload['idx_corr']]>contact_phases[pl_ref][0]) & (cen_ph[data_upload['idx_corr']]<contact_phases[pl_ref][3])
                    plt.text(0.6,1.1,'dBIC[in-tr] = '+"{0:.3f}".format(np.mean(dBIC[data_upload['idx_corr'][cond_corr_in]])),verticalalignment='center', horizontalalignment='left',fontsize=10.,zorder=10,color='black',transform=plt.gca().transAxes)
                    
                    #Measured RMS
                    for isub,iexp in enumerate(data_upload['idx_corr']):
                        if iexp in data_upload['idx_pca']:col_loc = 'red'
                        else:col_loc = 'dodgerblue'
                        if cond_corr_in[isub]:marker_loc = 'o'
                        else:marker_loc = 's'
                        plt.plot(cen_ph[iexp], dBIC[iexp],markerfacecolor=col_loc,marker = marker_loc,color=col_loc,ls='',markersize=plot_set_key['markersize'])
                    y_min=np.min([np.nanmin(dBIC),y_min])
                    y_max=np.max([np.nanmax(dBIC),y_max]) 
                    

                    #Contacts
                    x_range_loc=plot_set_key['x_range_bic'] if plot_set_key['x_range_bic'] is not None else np.array([np.min(cen_ph),np.max(cen_ph)])
                    y_range_loc=plot_set_key['y_range_bic'] if plot_set_key['y_range_bic'] is not None else np.array([y_min,y_max])
                    for cont_ph in contact_phases[pl_ref]:plt.plot([cont_ph,cont_ph],y_range_loc,color='black',linestyle=':',lw=plot_set_key['lw_plot'],zorder=0)
                    plt.plot(x_range_loc,[0.,0.],color='black',linestyle='--',lw=plot_set_key['lw_plot'],zorder=0)

                    #Plot frame  
                    if plot_set_key['title']:plt.title('Delta-BIC for visit '+vis+' in '+inst)
                    xmajor_int,xminor_int,xmajor_form=autom_tick_prop(x_range_loc[1]-x_range_loc[0])
                    ymajor_int,yminor_int,ymajor_form=autom_tick_prop(y_range_loc[1]-y_range_loc[0]) 
                    custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,dir_y='out', 
                                xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,
                                xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                                x_title='Orbital phase',y_title='Delta-BIC',
                                font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                    plt.savefig(path_loc+inst+'_'+vis+'_PCdBIC.'+plot_dic['pca_ana']) 
                    plt.close()               

                #------------------------------------------------                         
                #Histograms of corrected residual profiles
                if plot_set_key['pc_hist']:
                    y_min=1e100
                    y_max=-1e100

                    plt.ioff() 
                    fig, axes = plt.subplots(1, 3, figsize=plot_set_key['fig_size'])
                    fig.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.6,wspace=0.05 , hspace=0.05 )
                    y_range_loc = [0,1.1] 
                    sc_fact10 = 4
                    x_title = scaled_title(sc_fact10,'Residuals') 
                    sc_fact=10**sc_fact10
                    
                    
                    #---------------------------------------------------                    
                    #Histogram on pre-transit residuals
                    hist_data = data_upload['hist_corr_res_pre']
                    if not np.isnan(hist_data[0][0]):
                        ax = axes[0]
                        edge_bins = hist_data[1]*sc_fact
                        ax.stairs(hist_data[0]/sc_fact,edges = edge_bins,color='blue')
                        ax.axvline(0., color='red',lw=1,ls='--')
                        cen_bins = 0.5*(edge_bins[0:-1]+edge_bins[1::])
                        gauss_mod = np.exp( -cen_bins**2./(2.*(data_upload['std_corr_res_pre']*sc_fact)**2.) )
                        gauss_norm = gauss_mod/np.sum(gauss_mod*(edge_bins[1::]-edge_bins[0:-1]))
                        ax.plot(cen_bins,gauss_norm,color='red',lw=1)
    
                        if plot_set_key['x_range_hist'] is not None:x_range_loc = plot_set_key['x_range_hist']
                        else:x_range_loc = [np.min(cen_bins),np.max(cen_bins)]    
                        if plot_set_key['y_range_hist'] is not None:y_range_loc = plot_set_key['y_range_hist']
                        else:y_range_loc = [0,1.1 * np.max(gauss_norm)]                      
                        ax.text(x_range_loc[0]+0.5*(x_range_loc[1]-x_range_loc[0]),y_range_loc[1]+0.01*(y_range_loc[1]-y_range_loc[0]),'Pre-transit',verticalalignment='bottom', horizontalalignment='center',fontsize=0.8*plot_set_key['font_size'],zorder=4,color='black') 
                        ax.set_xlim(x_range_loc)
                        ax.set_ylim(y_range_loc)
                        ax.set_yticklabels([])
                        ax.yaxis.set_ticks_position('none')          
                        ax.set_xlabel(x_title,fontsize=plot_set_key['font_size'])
                    
                    #---------------------------------------------------                    
                    #Histogram on post-transit residuals
                    hist_data = data_upload['hist_corr_res_post']
                    if not np.isnan(hist_data[0][0]):
                        ax = axes[1]
                        edge_bins = hist_data[1]*sc_fact
                        ax.stairs(hist_data[0]/sc_fact,edges = edge_bins,color='blue')
    
                        ax.axvline(0., color='red',lw=1,ls='--')
                        cen_bins = 0.5*(edge_bins[0:-1]+edge_bins[1::])
                        gauss_mod = np.exp( -cen_bins**2./(2.*(data_upload['std_corr_res_post']*sc_fact)**2.) )
                        gauss_norm = gauss_mod/np.sum(gauss_mod*(edge_bins[1::]-edge_bins[0:-1]))
                        ax.plot(cen_bins,gauss_norm,color='red',lw=1)
    
                        if plot_set_key['x_range_hist'] is not None:x_range_loc = plot_set_key['x_range_hist']
                        else:x_range_loc = [np.min(cen_bins),np.max(cen_bins)]    
                        if plot_set_key['y_range_hist'] is not None:y_range_loc = plot_set_key['y_range_hist']
                        else:y_range_loc = [0,1.1 * np.max(gauss_norm)]                      
                        ax.text(x_range_loc[0]+0.5*(x_range_loc[1]-x_range_loc[0]),y_range_loc[1]+0.01*(y_range_loc[1]-y_range_loc[0]),'Post-transit',verticalalignment='bottom', horizontalalignment='center',fontsize=0.8*plot_set_key['font_size'],zorder=4,color='black') 
                        ax.set_xlim(x_range_loc)
                        ax.set_ylim(y_range_loc)
                        ax.set_yticklabels([])
                        ax.yaxis.set_ticks_position('none')          
                        ax.set_xlabel(x_title,fontsize=plot_set_key['font_size'])
                        
                    #---------------------------------------------------                    
                    #Histogram on out-transit residuals
                    ax = axes[2]

                    hist_data = data_upload['hist_corr_res']
                    edge_bins = hist_data[1]*sc_fact
                    ax.stairs(hist_data[0]/sc_fact,edges = edge_bins,color='blue')

                    ax.axvline(0., color='red',lw=1,ls='--')
                    cen_bins = 0.5*(edge_bins[0:-1]+edge_bins[1::])
                    gauss_mod = np.exp( -cen_bins**2./(2.*(data_upload['std_corr_res']*sc_fact)**2.) )
                    gauss_norm = gauss_mod/np.sum(gauss_mod*(edge_bins[1::]-edge_bins[0:-1]))
                    ax.plot(cen_bins,gauss_norm,color='red',lw=1)

                    if plot_set_key['x_range_hist'] is not None:x_range_loc = plot_set_key['x_range_hist']
                    else:x_range_loc = [np.min(cen_bins),np.max(cen_bins)]    
                    if plot_set_key['y_range_hist'] is not None:y_range_loc = plot_set_key['y_range_hist']
                    else:y_range_loc = [0,1.1 * np.max(gauss_norm)]                      
                    ax.text(x_range_loc[0]+0.5*(x_range_loc[1]-x_range_loc[0]),y_range_loc[1]+0.01*(y_range_loc[1]-y_range_loc[0]),'Out-transit',verticalalignment='bottom', horizontalalignment='center',fontsize=0.8*plot_set_key['font_size'],zorder=4,color='black') 
                    ax.set_xlim(x_range_loc)
                    ax.set_ylim(y_range_loc)
                    ax.set_yticklabels([])
                    ax.yaxis.set_ticks_position('none')                     
                    ax.set_xlabel(x_title,fontsize=plot_set_key['font_size'])

                    plt.savefig(path_loc+inst+'_'+vis+'_PChist.'+plot_dic['pca_ana'])                  
                    plt.close()  

              
                

                                
                
                #------------------------------------------------                
                #PC profiles
                if plot_set_key['pc_prof']:
                    plt.ioff()        
                    fig = plt.figure(figsize=plot_set_key['fig_size'])
                    y_min=1e100
                    y_max=-1e100
    
                    #PC 
                    for isubpc,ipc in enumerate(plot_set_key['pc_list']):
                        plt.plot(data_upload['cen_bins'], data_upload['eig_res_matr'][ipc],color=plot_set_key['pc_col'][isubpc],lw=plot_set_key['lw_plot'])
                        y_min=np.min([np.min(data_upload['eig_res_matr'][ipc]),y_min])
                        y_max=np.max([np.max(data_upload['eig_res_matr'][ipc]),y_max])                      
    
                    #Plot frame  
                    x_range_loc=plot_set_key['x_range_pc'] if plot_set_key['x_range_pc'] is not None else np.array([np.min(data_upload['cen_bins']),np.max(data_upload['cen_bins'])])
                    plt.plot(x_range_loc,[0,0],color='black',linestyle='--')
                    y_range_loc=plot_set_key['y_range_pc'] if plot_set_key['y_range_pc'] is not None else np.array([y_min,y_max])
                    if plot_set_key['title']:plt.title('PC profiles for visit '+vis+' in '+inst)
                    xmajor_int,xminor_int,xmajor_form=autom_tick_prop(x_range_loc[1]-x_range_loc[0])
                    ymajor_int,yminor_int,ymajor_form=autom_tick_prop(y_range_loc[1]-y_range_loc[0]) 
                    custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,dir_y='out', 
                                xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,
                                xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                                x_title='Velocity in star rest frame (km/s)',y_title='PC profile',
                                font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                    plt.savefig(path_loc+inst+'_'+vis+'_PCprof.'+plot_dic['pca_ana']) 
                    plt.close()  



                #------------------------------------------------                
                #FFT analysis
                if plot_set_key['fft_prof']:
                    x_range_loc=plot_set_key['x_range_fft'] if plot_set_key['x_range_fft'] is not None else np.array([np.min(data_upload['cen_bins_res_mat']),np.max(data_upload['cen_bins_res_mat'])])
                    for fft_loc in plot_set_key['fft_list']:
                        plt.ioff()        
                        fig = plt.figure(figsize=plot_set_key['fig_size'])
                        y_min=1e100
                        y_max=-1e100

                        #FFT over phase at each velocity 
                        if fft_loc=='res':
                            var_plot = data_upload['fft1D_res_matr']
                            plt.text(0.1,1.15,'max(| FFT2D |)^2 = '+"{0:.3e}".format(np.mean(data_upload['max_fft2_res_matr'][0]))+' (preTR) / '+"{0:.3e}".format(np.mean(data_upload['max_fft2_res_matr'][1]))+' (postTR)',verticalalignment='center', horizontalalignment='left',fontsize=10.,zorder=10,color='black',transform=plt.gca().transAxes)                    
                        elif fft_loc=='corr':
                            var_plot = data_upload['fft1D_corr_res_matr']
                            plt.text(0.1,1.15,'max(| FFT2D |)^2 = '+"{0:.3e}".format(np.mean(data_upload['max_fft2_corr_res_matr'][0]))+' (preTR) / '+"{0:.3e}".format(np.mean(data_upload['max_fft2_corr_res_matr'][1]))+' (postTR)',verticalalignment='center', horizontalalignment='left',fontsize=10.,zorder=10,color='black',transform=plt.gca().transAxes)                    
                        elif fft_loc=='boot':
                            var_plot = data_upload['fft1D_boot_res_matr']
                            plt.text(0.1,1.15,'max(| FFT2D |)^2 = '+"{0:.3e}".format(np.mean(data_upload['max_fft2_boot_res_matr'][0]))+' (preTR) / '+"{0:.3e}".format(np.mean(data_upload['max_fft2_boot_res_matr'][1]))+' (postTR)',verticalalignment='center', horizontalalignment='left',fontsize=10.,zorder=10,color='black',transform=plt.gca().transAxes) 
                        plt.plot(data_upload['cen_bins_res_mat'], var_plot[0],lw=plot_set_key['lw_plot'],marker='o',markersize=1.5,color='dodgerblue')
                        plt.plot(data_upload['cen_bins_res_mat'], var_plot[1],lw=plot_set_key['lw_plot'],marker='o',markersize=1.5,color='red')
                        y_min=np.min([np.nanmin(var_plot),y_min])
                        y_max=np.max([np.nanmax(var_plot),y_max])                      
                 
                        #Plot frame  
                        y_range_loc=plot_set_key['y_range_fft'] if plot_set_key['y_range_fft'] is not None else np.array([y_min,y_max])
                        if plot_set_key['title']:
                            if fft_loc=='res':txt = 'residual profiles'
                            elif fft_loc=='corr':txt = 'corr. residual profiles'
                            elif fft_loc=='boot':txt = 'boot. corr. residual profiles'
                            plt.title('FFT from '+txt+' for visit '+vis+' in '+inst)
                        xmajor_int,xminor_int,xmajor_form=autom_tick_prop(x_range_loc[1]-x_range_loc[0])
                        ymajor_int,yminor_int,ymajor_form=autom_tick_prop(y_range_loc[1]-y_range_loc[0]) 
                        custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,dir_y='out', 
                                    xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,
                                    xmajor_form=xmajor_form,ymajor_form=ymajor_form,
                                    x_title='Velocity in star rest frame (km/s)',y_title='max(| FFT |)^2',
                                    font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                        plt.savefig(path_loc+inst+'_'+vis+'_FFTprof_'+fft_loc+'.'+plot_dic['pca_ana']) 
                        plt.close()  







 


    ##################################################################################################
    #%%% Residual dispersion
    ##################################################################################################
    if ('scr_search' in plot_settings):
        key_plot = 'scr_search'
        plot_set_key = plot_settings[key_plot]  

        print('-----------------------------------')
        print('+ Plot stdev vs binsize for out CCFs in each visit')
    
        #Plot for each instrument
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())): 
            
            #Plot for each visit
            for vis in np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst]): 
                Res_data_vis = data_dic['Res'][inst][vis]
                path_loc = gen_dic['save_plot_dir']+'Screen_search'+inst+'_'+vis+'/'
                if (not os_system.path.exists(path_loc)):os_system.makedirs(path_loc) 
    
                #--------------------------------------------------------------------
                #Residual CCF out-transit for current visit
                for isub,iexp in enumerate(gen_dic[inst][vis]['idx_out']):
                    plt.ioff()                    
                    fig = plt.figure(figsize=plot_set_key['fig_size'])
                
                    #Vertical range
                    y_min=1e100
                    y_max=-1e100
          
                    #Plot measured stdev
                    plt.plot(gen_dic['scr_srch_nperbins'],Res_data_vis['corr_search']['meas'][isub],marker='o',markersize=4,linestyle='')
                    y_min=min(min(Res_data_vis['corr_search']['meas'][isub]),y_min)
                    y_max=max(max(Res_data_vis['corr_search']['meas'][isub]),y_max)
    
                    #Plot fit and its components
                    plt.plot(gen_dic['scr_srch_nperbins'],Res_data_vis['corr_search']['fit'][isub,:]) 
                    uncorr_fit=Res_data_vis['corr_search']['sig_uncorr'][isub]/np.sqrt(gen_dic['scr_srch_nperbins'])
                    corr_fit=np.ones(len(gen_dic['scr_srch_nperbins']))*Res_data_vis['corr_search']['sig_corr'][isub]
                    plt.plot(gen_dic['scr_srch_nperbins'],uncorr_fit) 
                    plt.plot(gen_dic['scr_srch_nperbins'],corr_fit) 
                    y_min=min(min(uncorr_fit),y_min)
                    y_max=max(max(uncorr_fit),y_max)
                           
                    #Plot frame  
                    plt.title('Standard deviation vs binsize (visit ='+vis+'; phase ='+"{0:.5f}".format(coord_dic[inst][vis][pl_ref]['cen_ph'][iexp])+')')
                    y_range_loc=plot_set_key['y_range'] if plot_set_key['y_range'] is not None else np.array([0.9*y_min,1.1*y_max])
                    custom_axis(plt,position=plot_set_key['margins'],x_range=[1,gen_dic['scr_srch_nperbins'][-1]],y_range=y_range_loc,
                		    xmajor_form='%i',ymajor_form='%.2e',x_mode='log',y_mode='log',
                		    x_title='Number of points per bin',y_title='Standard-deviation',
                          font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                    plt.savefig(path_loc+'OutCCF_stdev_idx'+str(iexp)+"{0:.4f}".format(coord_dic[inst][vis][pl_ref]['cen_ph'][iexp])+'.'+plot_dic['scr_search']) 
                    plt.close()
    
    
















    ##################################################################################################
    #%% Intrinsic profiles
    ##################################################################################################        

    ##################################################################################################
    #%%% 2D maps
    ################################################################################################## 
        
    ##################################################################################################
    #%%%% Original profiles
    ##################################################################################################
    if ('map_Intr_prof' in plot_settings):
        key_plot = 'map_Intr_prof'

        print('-----------------------------------')
        print('+ 2D map: intrinsic stellar profiles')
        
        #Plot map        
        sub_2D_map(key_plot,plot_dic[key_plot],plot_settings[key_plot])
                   



    ##################################################################################################
    #%%%% Binned profiles
    ##################################################################################################
    if ('map_Intrbin' in plot_settings):
        key_plot = 'map_Intrbin'

        print('-----------------------------------')
        print('+ 2D map: binned intrinsic stellar profiles') 

        #Plot map
        sub_2D_map(key_plot,plot_dic[key_plot],plot_settings[key_plot])
                   




    ##################################################################################################
    #%%%% 1D converted spectra
    ##################################################################################################
    if ('map_Intr_1D' in plot_settings):
        key_plot = 'map_Intr_1D'

        print('-----------------------------------')
        print('+ 2D map: 1D intrinsic stellar profiles') 

        #Plot map
        sub_2D_map(key_plot,plot_dic[key_plot],plot_settings[key_plot])
                   


    ################################################################################################################ 
    #%%%% Estimate and residual profiles
    ################################################################################################################ 
    for key_plot in ['map_Intr_prof_est','map_Intr_prof_res']:
        if (key_plot in plot_settings):

            ##############################################################################
            #%%%%% Estimates
            if key_plot == 'map_Intr_prof_est':
                print('-----------------------------------')
                print('+ 2D map: theoretical intrinsic stellar profiles') 

            ##############################################################################
            #%%%%% Residuals    
            if key_plot == 'map_Intr_prof_res':
                print('-----------------------------------')
                print('+ 2D map: residuals from intrinsic stellar profiles') 

            #Plot map
            sub_2D_map(key_plot,plot_dic[key_plot],plot_settings[key_plot])    






    ################################################################################################################ 
    #%%%% PC-based noise model 
    ################################################################################################################   
    if ('map_pca_prof' in plot_settings): 
        key_plot = 'map_pca_prof'
        
        print('-----------------------------------')
        print('+ Plotting 2D map of PC-based noise model')     

        #Plot map
        sub_2D_map(key_plot,plot_dic[key_plot],plot_settings[key_plot])

    




    ##################################################################################################
    #%%% Individual profiles
    ##################################################################################################   

    ##################################################################################################
    #%%%% Original profiles (grouped)
    ##################################################################################################
    if ('all_intr_data' in plot_settings):
        key_plot = 'all_intr_data'

        print('-----------------------------------')
        print('+ All intrinsic profiles')
        sub_plot_all_prof(plot_settings[key_plot],'Intr',plot_dic[key_plot])             
                        





    ##################################################################################################
    #%%%% Original profiles
    ##################################################################################################
    for key_plot in ['Intr_prof','Intr_prof_res']:
        if (key_plot in plot_settings):
    
            ##############################################################################
            #%%%%% Flux profiles
            if (key_plot=='Intr_prof'):
                print('-----------------------------------')
                print('+ Individual intrinsic profiles')
    
            ##############################################################################
            #%%%%% Residuals profiles
            if (key_plot=='Intr_prof_res'):
                print('-----------------------------------')
                print('+ Individual residuals from intrinsic profiles')

            #%%%%% Plot        
            sub_plot_prof(plot_settings[key_plot],key_plot,plot_dic[key_plot])    





    ##################################################################################################
    #%%%% Binned profiles
    ##################################################################################################
    for key_plot in ['Intrbin','Intrbin_res']:
        if (key_plot in plot_settings):

            ##############################################################################
            #%%%%% Profile and its fit
            if (key_plot=='Intrbin'):
                print('-----------------------------------')
                print('+ Individual binned intrinsic profiles')
                
            ##############################################################################
            #%%%%% Residuals between the profile and its fit
            if (key_plot=='Intrbin_res'):
                print('-----------------------------------')
                print('+ Individual residuals from binned intrinsic profiles')

            ##############################################################################
            #%%%%% Plot  
            sub_plot_prof(plot_settings[key_plot],key_plot,plot_dic[key_plot])








    ################################################################################################################   
    #%%%% 1D converted spectra
    ################################################################################################################   
    for key_plot in ['sp_Intr_1D','sp_Intr_1D_res']:
        if (key_plot in plot_settings):

            ##############################################################################
            #%%%%% Profile and its fit
            if (key_plot=='sp_Intr_1D'):
                print('-----------------------------------')
                print('+ Individual 1D intrinsic profiles')
                
            ##############################################################################
            #%%%%% Residuals between the profile and its fit
            if (key_plot=='sp_Intr_1D_res'):
                print('-----------------------------------')
                print('+ Individual residuals from 1D intrinsic profiles')

            ##############################################################################
            #%%%%% Plot  
            sub_plot_prof(plot_settings[key_plot],key_plot,plot_dic[key_plot])

                







    ##################################################################################################
    #%%% Chi2 over intrinsic property series
    ##################################################################################################
    if ('chi2_fit_IntrProp' in plot_settings):
        key_plot = 'chi2_fit_IntrProp'

        print('-----------------------------------')
        print('+ Chi2 over intrinsic property series')
        sub_plot_chi2_prop(plot_settings[key_plot],plot_dic['chi2_fit_IntrProp'])             




    ################################################################################################################ 
    #%%% Range of planet-occulted properties
    ################################################################################################################ 
    if ('plocc_ranges' in plot_settings):
        key_plot = 'plocc_ranges'
        plot_set_key = plot_settings[key_plot] 

        print('-----------------------------------')
        print('+ Range of planet-occulted regions')
        
        #Directory
        path_loc = gen_dic['save_plot_dir']+'Plocc_prop/' 
        if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  
                    
        #Plot for each instrument
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())):                            
            plt.ioff()                    
            fig = plt.figure(figsize=plot_set_key['fig_size'])

            #New bin ranges
            if inst in plot_set_key['prop_bin']:
                bin_prop = plot_set_key['prop_bin'][inst]
                if ('bin_low' in bin_prop):
                    new_x_low = np.array(bin_prop['bin_low'])
                    new_x_high = np.array(bin_prop['bin_high'])   
                elif 'bin_range' in bin_prop:
                    min_x = bin_prop['bin_range'][0]
                    max_x = bin_prop['bin_range'][1]
                    new_dx =  (max_x-min_x)/bin_prop['nbins']
                    new_nx = int((max_x-min_x)/new_dx)
                    new_x_low = min_x + new_dx*np.arange(new_nx)
                    new_x_high = new_x_low+new_dx 
            
            #Plot all visits
            vis_list = np.intersect1d(list(data_dic[inst].keys()),plot_set_key['visits_to_plot'][inst])
            y_min=-5.*plot_set_key['y_gap']
            y_max=0.
            if plot_set_key['x_range'] is None:stop('Define plot range')
            for ivis,vis in enumerate(vis_list): 

                #Visit colors    
                n_in_tr = data_dic[inst][vis]['n_in_tr']
                if (inst in plot_set_key['color_dic']) and (vis in plot_set_key['color_dic'][inst]):
                    col_visit=np.repeat(plot_set_key['color_dic'][inst][vis],n_in_tr)
                else:
                    cmap = plt.get_cmap('jet') 
                    col_visit=np.array([cmap(0)]) if n_in_tr==1 else cmap( np.arange(n_in_tr)/(n_in_tr-1.))

                #Reference planet for the visit                    
                pl_ref = plot_set_key['pl_ref'][inst][vis]
                    
                #Plot chosen property
                transit_prop_nom = dataload_npz(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis)['achrom'][pl_ref]    
                x_obs = transit_prop_nom[plot_set_key['x_prop']][0,:]
                st_x_obs,end_x_obs = transit_prop_nom[plot_set_key['x_prop']+'_range'][0,:,0],transit_prop_nom[plot_set_key['x_prop']+'_range'][0,:,1]
                if plot_set_key['plot_expid']:
                    hx_range = 0.5*(np.max(x_obs)-np.min(x_obs))
                    mean_x = 0.5*(np.max(x_obs)+np.min(x_obs)) 
                    idx_conj = closest(x_obs,np.min(x_obs)) 

                for i_in,(x_obs_loc,st_x_obs_loc,end_x_obs_loc,col_exp) in enumerate(zip(x_obs,st_x_obs,end_x_obs,col_visit)):
                    plt.errorbar(x_obs_loc, y_max+ plot_set_key['y_gap']*i_in,xerr=[[x_obs_loc-st_x_obs_loc],[end_x_obs_loc-x_obs_loc]],color=col_exp,marker='o',markersize=plot_set_key['markersize'],linestyle='',markerfacecolor=col_exp,lw=plot_set_key['lw_plot'])                

                    #Exposures indexes
                    if plot_set_key['plot_expid']:
                        x_txt = x_obs_loc #-  np.sign(mean_x-x_obs_loc)*hx_range*0.03
                        y_txt = y_max+ plot_set_key['y_gap']*i_in - np.sign(idx_conj-i_in)*2*plot_set_key['y_gap']
                        plt.text(x_txt,y_txt,str(i_in),verticalalignment='center', horizontalalignment='center',fontsize=plot_set_key['font_size']-3.,zorder=4,color='black') 

                #Visit name
                if plot_set_key['plot_visid']:
                    plt.text(0.5*(plot_set_key['x_range'][0]+plot_set_key['x_range'][1]),y_max+ plot_set_key['y_gap']*0.5*data_dic[inst][vis]['n_in_tr'],str(vis),verticalalignment='bottom', horizontalalignment='center',fontsize=plot_set_key['font_size']-2.,zorder=4,color='black') 
                y_max+=n_in_tr*plot_set_key['y_gap'] + 5.*plot_set_key['y_gap'] 

                #New bin ranges
                if inst in plot_set_key['prop_bin']:
                    for new_x_low_loc,new_x_high_loc in zip(new_x_low,new_x_high):
                        plt.gca().axvspan(xmin=new_x_low_loc,xmax=new_x_high_loc,color='black',alpha=0.1,linestyle='--',lw=plot_set_key['lw_plot']-1,zorder=-1)

            #Plot frame  
            x_title={'mu':r'$\mu$','xp_abs':r'|x$_p$|','r_proj':r'r$_{\rm sky}$'}[plot_set_key['x_prop']]
            if plot_set_key['title']:plt.title('Planet-occulted range for '+x_title,fontsize=plot_set_key['font_size'])
            if plot_set_key['x_range'] is not None:x_range_loc=plot_set_key['x_range']
            else:
                dx_range = x_max-x_min
                x_range_loc = np.array([x_min-0.05*dx_range,x_max+0.05*dx_range])   
            if plot_set_key['y_range'] is not None:y_range_loc=plot_set_key['y_range'] 
            else:
                dy_range = y_max-y_min
                y_range_loc = np.array([y_min-0.05*dy_range,y_max+0.05*dy_range])                          
            xmajor_int,xminor_int,xmajor_form=autom_tick_prop(x_range_loc[1]-x_range_loc[0])
            ymajor_int,yminor_int,ymajor_form=autom_tick_prop(y_range_loc[1]-y_range_loc[0]) 
            custom_axis(plt,position=plot_set_key['margins'],x_range=x_range_loc,y_range=y_range_loc,dir_y='out', 
            		     xmajor_int=xmajor_int,xminor_int=xminor_int,xmajor_form=xmajor_form,
            		     ymajor_int=None,yminor_int=yminor_int,ymajor_form=ymajor_form,
                         x_title=x_title,
                         y_title='Exposures',
                         font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
            plt.savefig(path_loc+inst+'_'+plot_set_key['x_prop']+'.'+plot_dic['plocc_ranges'])                        
            plt.close()  







    ################################################################################################################
    #%%% 1D PDFs from analysis of individual profiles
    ################################################################################################################
    for key_plot in ['prop_DI_mcmc_PDFs','prop_Intr_mcmc_PDFs']:
        if (key_plot in plot_settings):

            ##############################################################################
            #%%%% Disk-integrated profiles
            if (key_plot=='prop_DI_mcmc_PDFs'):
                print('-----------------------------------')
                print('+ 1D PDFs of disk-integrated properties')
        
            ##############################################################################
            #%%%% Intrinsic profiles            
            if (key_plot=='prop_Intr_mcmc_PDFs'):
                print('-----------------------------------')
                print('+ 1D PDFs of intrinsic properties')

            #%%%% Plot
            sub_plot_propCCF_mcmc_PDFs(plot_settings[key_plot],key_plot,plot_dic[key_plot])
        
        
        
        
        
        


 
    ##################################################################################################
    #%%% Properties of intrinsic profiles
    ##################################################################################################
    if (plot_dic['prop_Intr']!=''):
        print('-----------------------------------')
        print('+ Properties of intrinsic stellar profiles')

        #%%%% Processing properties      
        for plot_prop in plot_settings['prop_Intr_ordin']:
            key_plot = 'prop_Intr_'+plot_prop 
            txt_print = {'rv':'RV','rv_res':'RV residuals','rv_l2c':'RV lobe-to-core difference','RV_lobe':'Lobe RV','FWHM':'FWHM','FWHM_voigt':'','FWHM_l2c':'FWHM lobe-to-core ratio','FWHM_lobe':'Lobe FWHM','true_FWHM':'True FWHM',
                         'ctrst':'Contrast','true_ctrst':'True contrast','amp':'Amplitude','amp_l2c':'Amplitude lobe-to-core ratio','amp_lobe':'Lobe amplitude','area':'Area','a_damp':'Damping coefficient'}
            print('   ---------------')
            print('   > '+txt_print[plot_prop])

            #%%%%% Plot routine   
            sub_plot_CCF_prop(plot_prop,plot_settings[key_plot],'Intr')                  


        


    ################################################################################################################ 
    #%% Binned disk-integrated and intrinsic profile series
    ################################################################################################################  
    if ('binned_DI_Intr' in plot_settings):
        key_plot = 'binned_DI_Intr'

        print('-----------------------------------')
        print('+ Plotting binned disk-integrated and intrinsic profiles')
             
        #Plot
        sub_plot_DI_Intr_binprof(plot_settings,plot_dic[key_plot])     

 



        
        
    ##################################################################################################
    #%% System view
    ##################################################################################################        
      
    ################################################################################################## 
    #%%% Occulted stellar regions
    ##################################################################################################  
    if ('occulted_regions' in plot_settings):
        key_plot = 'occulted_regions' 
        plot_set_key = plot_settings[key_plot]  

        print('-----------------------------------')
        print('+ Planet-occulted regions')

        path_loc = gen_dic['save_plot_dir']+'Plocc_prop/' 
        if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  

        #Plot CCFs per instrument
        cmap = plt.get_cmap('jet')
        size_y = 10 #Enter here the size you want for y axis (figsize will take this size value and proportionally adjust the x axis with dif_x_over_y)
        dif_x_over_y = (plot_set_key['x_range'][1]-plot_set_key['x_range'][0])/(plot_set_key['y_range'][1]-plot_set_key['y_range'][0])
        size_x = size_y*dif_x_over_y
        if ('chrom' in data_dic['DI']['system_prop']):system_prop = data_dic['DI']['system_prop']['chrom']
        else:system_prop = data_dic['DI']['system_prop']['achrom'] 
        for inst in np.intersect1d(data_dic['instrum_list'],list(plot_set_key['visits_to_plot'].keys())): 
            for vis in np.intersect1d(list(data_dic[inst].keys())+['binned'],plot_set_key['visits_to_plot'][inst]): 
                pl_ref = plot_set_key['pl_ref'][inst][vis]
                RpRs_band = system_prop[pl_ref][plot_set_key['iband']]
        
                #Identifying original or binned data
                orig_vis = vis.split('_bin')[0]
                if 'bin' in vis:data_type = 'Intrbin'
                else:data_type = 'Introrig'           
                if orig_vis in list(data_dic['Res'][inst].keys()):
                    plt.ioff()        
                    fig = plt.figure(figsize=(size_x,size_y))

                    #Oblate star
                    #    - see definition of star boundary in 'system_view'        
                    if system_param['star']['f_GD']>0.: 
                        #Oblate star parameters
                        star_params = system_param['star']
                        Rpole = star_params['RpoleReq']
                        mRp2 = (1. - Rpole**2.)
                        istar_rad=star_params['istar_rad']
                        ci = cos(istar_rad)
                        si = sin(istar_rad)  
                        #X position initialization
                        r_inner=1.
                        r_outer=1.7
                        x_annulus = np.linspace(-r_outer, r_outer, 3000, endpoint=True)
                        #Oblate star
                        #there is no simple expression of the projection of the photosphere envelope in the plane of the sky
                        #for each x value of the grid, we thus find the condition for y values on a grid to belong to the photosphere (ie, having defined z values), and we take the maximum of the y values fulfilling the condition   
                        ygrid = np.linspace(-1., 1., 3000, endpoint=True)
                        y_grid_all = np.tile(ygrid,(3000,1)) 
                        Aquad =  1. - si**2.*mRp2 
                        r_grid2 = x_annulus[:,None]**2.+ y_grid_all**2. 
                        Bquad = 2.*y_grid_all*ci*si*mRp2        
                        Cquad_in = y_grid_all**2.*si**2.*mRp2 + Rpole**2.*(r_grid2 - r_inner**2.)   
                        det_in = Bquad**2.-4.*Aquad*Cquad_in
                        cond_in = det_in<0.
                        y_grid_in = deepcopy(y_grid_all)                
                        y_grid_in[cond_in] = 0.         
                        yin_up = np.nanmax(y_grid_in,axis=1)
                        yin_down = -yin_up            
                        #Finding the spot where we the yin_up values go to 0, so as to keep one of the zeros for plotting.
                        #Finding indexes where yin_up is zero.
                        up_zeros_indexes = np.where(yin_up==0)
                        #Finding indexes where the zero to non-zero value switch occurs (happens at two places).
                        zero_shift_index_low = np.where(np.diff(up_zeros_indexes[0])>1)[0][0]
                        zero_shift_index_high = up_zeros_indexes[0][zero_shift_index_low+1]
                        #The spot where we swap from zero to non-zero values is the same in yin_up and yin_down.
                        #Plotting the oblate star outline
                        plt.plot(x_annulus[zero_shift_index_low:zero_shift_index_high+1], yin_up[zero_shift_index_low:zero_shift_index_high+1], color="black",zorder=0,lw=1)
                        plt.plot(x_annulus[zero_shift_index_low:zero_shift_index_high+1], yin_down[zero_shift_index_low:zero_shift_index_high+1], color="black",zorder=0,lw=1)  

                    #Spherical star
                    else: 
                        star_circle=plt.Circle((0.,0.),1.,color='black',fill=False,lw=1,zorder=0)   
                        fig.gca().add_artist(star_circle)

                    #Planet orbit
                    #    - 'coord_orbit' is defined in the Sky-projected orbital frame: Xsky,Ysky,Zsky	
                    if plot_set_key['plot_orb']:
                        coord_orbit = calc_pl_coord_plots(plot_dic['npts_orbit'],system_param[pl_ref])
                        x_orbit_view=coord_orbit[0]
                        y_orbit_view=coord_orbit[1]
                        w_noorb=np.where( ( (np.power(x_orbit_view,2.)+np.power(y_orbit_view,2.) ) < 1. ) & (coord_orbit[2] < 0.) )[0]
                        x_orbit_view[w_noorb]=np.nan
                        y_orbit_view[w_noorb]=np.nan
                        plt.plot(x_orbit_view,y_orbit_view,zorder=1, color='green',lw=1)
        
                    #Coordinate of planet-occulted regions during transit
                    if data_type=='Introrig':
                        idx_in = gen_dic[inst][vis]['idx_in']
                        x_pos_in=coord_dic[inst][vis][pl_loc]['cen_pos'][0,idx_in]
                        y_pos_in=coord_dic[inst][vis][pl_loc]['cen_pos'][1,idx_in]
                        st_pos_in=coord_dic[inst][vis][pl_loc]['st_pos']
                        end_pos_in=coord_dic[inst][vis][pl_loc]['end_pos']
                    elif data_type=='Intrbin':
                        data_bin = np.load(gen_dic['save_data_dir']+'Intrbin_data/'+inst+'_'+vis+'_'+plot_set_key['dim_plot']+'_add.npz',allow_pickle=True)['data'].item()
                        x_pos_in=data_bin['cen_pos'][0]
                        y_pos_in=data_bin['cen_pos'][1]
                        st_pos_in=data_bin['st_pos']
                        end_pos_in=data_bin['end_pos']
                    n_pos=len(x_pos_in)
                    color_pos=cmap( np.arange(n_pos)/(n_pos-1.))
                    
                    #Positions in x,y,z in front of the stellar disk
                    #    - x along the node line, y along the projection of the normal to the orbital plane
                    for i_in,(x_pos,y_pos,st_x_pos,st_y_pos,end_x_pos,end_y_pos) in enumerate(zip(x_pos_in,y_pos_in,st_pos_in[0,idx_in],st_pos_in[1,idx_in],end_pos_in[0,idx_in],end_pos_in[1,idx_in])):
        
                        #Planet center                
                        plt.plot(x_pos,y_pos,color=color_pos[i_in],marker='o',markersize=3,zorder=2+i_in)  
                        
                        #Occulted regions during full exposure
                        #    - because of the blur, the regions occulted by the planet go from its position from the start to the end of the exposure    
                        #    - to plot exactly the occulted region one should rotate the plot below so that its horizontal axis is the tangent to the orbital plane
                        # but during the transit that's close to the horizontal
                        ang_left=(pi/2.)+pi*np.arange(101)/100.
                        plt.plot((st_x_pos+RpRs_band*np.cos(ang_left)),(st_y_pos+RpRs_band*np.sin(ang_left)),color=color_pos[i_in],lw = 0.5,zorder=2+i_in)         
                        ang_right=(-pi/2.)+pi*np.arange(101)/100.
                        plt.plot((end_x_pos+RpRs_band*np.cos(ang_right)),(end_y_pos+RpRs_band*np.sin(ang_right)),color=color_pos[i_in],lw = 0.5,zorder=2+i_in)         
                        plt.plot([st_x_pos,end_x_pos],[st_y_pos+RpRs_band,end_y_pos+RpRs_band],color=color_pos[i_in],lw = 0.5,zorder=2+i_in)
                        plt.plot([st_x_pos,end_x_pos],[st_y_pos-RpRs_band,end_y_pos-RpRs_band],color=color_pos[i_in],lw = 0.5,zorder=2+i_in)
        
                    #Plot frame  
                    custom_axis(plt,position=[0.15,0.15,0.95,0.95],x_range=plot_set_key['x_range'],y_range=plot_set_key['y_range'],
                		    xmajor_int=0.5,xminor_int=0.1,ymajor_int=0.5,yminor_int=0.1,
                		    xmajor_form='%.1f',ymajor_form='%.1f',
                		    x_title='Position (R$_{*}$)',y_title='Position (R$_{*}$)',
                              font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
                    plt.savefig(path_loc+'Star_occ_'+inst+'_'+vis+'.'+plot_dic['occulted_regions']) 
                    plt.close()
        
        



    ################################################################################################################   
    #%%% Planetary system architecture
    ################################################################################################################ 
    if ('system_view' in plot_settings):
        key_plot = 'system_view'
        plot_set_key = plot_settings[key_plot] 

        print('-----------------------------------')
        print('+ Plotting planetary system architecture')

        path_loc = gen_dic['save_plot_dir']+'System_view/' 
        if not os_system.path.exists(path_loc):os_system.makedirs(path_loc)  
         
        #--------------------------------------------
        #Coordinates
        #---------
        #Sky-projected orbital frame: Xsky,Ysky,Zsky
        #    - Xsky = node line of orbital plane
        #    - Ysky = projection of the orbital plane normal
        #    - Zsky = LOS
    
        #Sky-projected stellar frame: Xsky_st,Ysky_st,Zsky_st
        #    - Xsky_st = projection of stellar equator
        #    - Ysky_st = projection of stellar spin axis
        #    - Zsky_st = LOS
    
        #Stellar frame: Xst,Yst,Zst
        #    - Xstar = stellar equator
        #    - Ystar = stellar spin axis
        #    - Zstar = completes the frame
        #---------
        #Relations Sky-projected orbital frame / Sky-projected stellar frame 
        #    - rotation by lambda around the Zsky axis, with lambda counted positive counterclokwise from Xsky_st to Xsky
        #    - Xsky_st = cos(lambda) Xsky - sin(lambda) Ysky
        #    - Ysky_st = sin(lambda) Xsky + cos(lambda) Ysky
        #    - Zsky_st = Zsky
        # or 
        #    - Xsky =  cos(lambda) Xsky_st + sin(lambda) Ysky_st
        #    - Ysky = -sin(lambda) Xsky_st + cos(lambda) Ysky_st
        #    - Zsky = Zsky_st
    
        #Relations Sky-projected stellar frame / stellar frame
        #    - rotation by istar around the Xsky_st axis, with istar counted positive clockwise from Zsky_st to Zstar
        #    - Xstar = Xsky_st
        #    - Ystar =  sin(istar) Ysky_st + cos(istar) Zsky_st
        #    - Zstar = -cos(istar) Ysky_st + sin(istar) Zsky_st
        # or 
        #    - Xsky_st = Xstar 
        #    - Ysky_st = sin(istar) Ystar - cos(istar) Zstar  
        #    - Zsky_st = cos(istar) Ystar + sin(istar) Zstar
        #---------
        #    - the sky-projected obliquity also represents the longitude of the ascending node, if the ascending node of the stellar equatorial plane is taken as reference
        #    - if the frame is set to 'sky_orb', ie it refers to the main planet, then other planets' coordinates must be projected onto it as:
        # Xsky = cos(Dlambda) Xsky[pl_loc] - sin(Dlambda) Ysky[pl_loc]
        # Ysky = sin(Dlambda) Xsky[pl_loc] + cos(Dlambda) Ysky[pl_loc]
        # Zsky = Zsky[pl_loc]
        #      avec Dlambda = lambda[pl_loc] - lambda
        #    - if the frame is set to 'sky_ste', the coordinates Xsky_st, Ysky_st, Zsky_st can be obtained as for the main planet using the local planet properties
        #-------------------------------------------
        star_params = system_param['star']
        Rpole = star_params['RpoleReq']
        mRp2 = (1. - Rpole**2.)
        lref=system_param[plot_set_key['pl_ref']]['lambda_rad']
        istar_rad=star_params['istar_rad']
        ci = cos(istar_rad)
        si = sin(istar_rad)  
        iband = plot_set_key['iband']
            
        #---------------------- 
        #Star boundary
        #---------------------- 
        r_inner=1.
        r_outer=1.7
        x_annulus = np.linspace(-r_outer, r_outer, 1000, endpoint=True)

        #Oblate star
        #      there is no simple expression of the projection of the photosphere envelope in the plane of the sky
        #      for each x value of the grid, we thus find the condition for y values on a grid to belong to the photosphere (ie, having defined z values), and we take the maximum of the y values fulfilling the condition   
        if star_params['f_GD']>0.:
            ygrid = np.linspace(-1., 1., 1000, endpoint=True)
            y_grid_all = np.tile(ygrid,(1000,1)) 
            Aquad =  1. - si**2.*mRp2 
            r_grid2 = x_annulus[:,None]**2.+ y_grid_all**2. 
            if plot_set_key['conf_system']=='sky_ste':
                #    - see calc_zLOS_oblate() for details on the calculations in the inclined star frame              
                Bquad = 2.*y_grid_all*ci*si*mRp2        
                Cquad_out = y_grid_all**2.*si**2.*mRp2 + Rpole**2.*(r_grid2 - r_outer**2.)   
                Cquad_in = y_grid_all**2.*si**2.*mRp2 + Rpole**2.*(r_grid2 - r_inner**2.)   

            if plot_set_key['conf_system']=='sky_orb': 
                #    - the photosphere expressed in the sky-projected star frame is :
                # xsky_st^2 + (ysky_st*si + zsky_st*ci)^2 / Rpole^2 + (-ysky_st*ci + zsky_st*si)^2 = Req_norm^2
                #      coordinates in the 'orbital' sky-projected frame verify:
                # xsky_st = xsky*cos(lambd) - ysky*sin(lambd) = xsky*cl - ysky*sl
                # ysky_st = xsky*sin(lambd) + ysky*cos(lambd) = xsky*sl + ysky*cl
                # zsky_st = zsky
                #      thus: 
                # (xsky*cl - ysky*sl)^2*Rpole^2 + ((xsky*sl + ysky*cl)*si + zsky*ci)^2 + (-(xsky*sl + ysky*cl)*ci + zsky*si)^2*Rpole^2 = Req_norm^2*Rpole^2
                #
                # xsky^2*cl^2*Rpole^2 - 2*xsky*cl*ysky*sl*Rpole^2 + ysky^2*sl^2*Rpole^2 
                #+ (xsky*sl + ysky*cl)^2*si^2 + 2*(xsky*sl + ysky*cl)*si*zsky*ci + zsky^2*ci^2
                #+ (xsky*sl + ysky*cl)^2*ci^2*Rpole^2 - 2*(xsky*sl + ysky*cl)*ci*zsky*si*Rpole^2 + zsky^2*si^2*Rpole^2 = Req_norm^2*Rpole^2
                #
                # xsky^2*cl^2*Rpole^2 - 2*xsky*cl*ysky*sl*Rpole^2 + ysky^2*sl^2*Rpole^2 
                #+ xsky^2*sl^2*si^2 + 2*xsky*sl*ysky*cl*si^2 + ysky^2*cl^2*si^2 + 2*xsky*sl*si*zsky*ci + 2*ysky*cl*si*zsky*ci + zsky^2*ci^2
                #+ xsky^2*sl^2*ci^2*Rpole^2 + ysky^2*cl^2*ci^2*Rpole^2 + 2*xsky*sl*ysky*cl*ci^2*Rpole^2 - 2*xsky*sl*ci*zsky*si*Rpole^2 - 2*ysky*cl*ci*zsky*si*Rpole^2 + zsky^2*si^2*Rpole^2 - Req_norm^2*Rpole^2 = 0               
                #
                # zsky^2*ci^2  + zsky^2*si^2*Rpole^2 
                # + 2*xsky*sl*si*zsky*ci  + 2*ysky*cl*si*zsky*ci - 2*xsky*sl*ci*zsky*si*Rpole^2  - 2*ysky*cl*ci*zsky*si*Rpole^2
                # + xsky^2*cl^2*Rpole^2 - 2*xsky*cl*ysky*sl*Rpole^2 + ysky^2*sl^2*Rpole^2 + xsky^2*sl^2*si^2 + 2*xsky*sl*ysky*cl*si^2 + ysky^2*cl^2*si^2 + xsky^2*sl^2*ci^2*Rpole^2 + ysky^2*cl^2*ci^2*Rpole^2 + 2*xsky*sl*ysky*cl*ci^2*Rpole^2  - Req_norm^2*Rpole^2 = 0               
                #
                # zsky^2*(ci^2  + si^2*Rpole^2) 
                # + 2*zsky*si*ci*(xsky*sl  + ysky*cl - xsky*sl*Rpole^2  - ysky*cl*Rpole^2)
                # + xsky^2*(cl^2*Rpole^2 + sl^2*si^2 + sl^2*ci^2*Rpole^2)   + 2*ysky*xsky*sl*cl*(si^2  + ci^2*Rpole^2 - Rpole^2)  + ysky^2*(cl^2*ci^2*Rpole^2  + sl^2*Rpole^2 + cl^2*si^2) - Req_norm^2*Rpole^2 = 0               
                #
                # zsky^2*(1 - si^2*(1-Rpole^2)) 
                # + 2*zsky*si*ci*(1-Rpole^2)*(xsky*sl + ysky*cl)
                # + xsky^2*(Rpole^2 + sl^2*si^2*(1-Rpole^2) ) + ysky^2*(Rpole^2 + cl^2*si^2*(1-Rpole^2))  + 2*ysky*xsky*sl*cl*si^2*(1-Rpole^2)  - Req_norm^2*Rpole^2 = 0               
                #
                # zsky^2*(1 - si^2*(1-Rpole^2)) 
                # + 2*zsky*si*ci*(1-Rpole^2)*(xsky*sl + ysky*cl)
                # + (xsky^2 + ysky^2 - Req_norm^2)*Rpole^2  + (xsky*sl + ysky*cl)^2*si^2*(1-Rpole^2) = 0   
                cl = cos(lref)
                sl = sin(lref)
                cross_term = (x_annulus[:,None]*sl + y_grid_all*cl)
                Bquad = 2.*si*ci*mRp2*cross_term
                Cquad_out = cross_term**2.*si**2.*mRp2 + (r_grid2 - r_outer**2.)*Rpole**2.
                Cquad_in = cross_term**2.*si**2.*mRp2  + (r_grid2 - r_inner**2.)*Rpole**2. 

            det_out = Bquad**2.-4.*Aquad*Cquad_out
            det_in = Bquad**2.-4.*Aquad*Cquad_in
            cond_out = det_out<0.
            cond_in = det_in<0.
            y_grid_out = deepcopy(y_grid_all)
            y_grid_in = deepcopy(y_grid_all)                
            y_grid_out[cond_out] = 0.
            y_grid_in[cond_in] = 0.         
            yout_up = np.nanmax(y_grid_out,axis=1)
            yin_up = np.nanmax(y_grid_in,axis=1)

            if plot_set_key['conf_system']=='sky_ste':
                yout_down = -yout_up
                yin_down = -yin_up
            if plot_set_key['conf_system']=='sky_orb':          
                yout_down = np.nanmin(y_grid_out,axis=1)
                yin_down = np.nanmin(y_grid_in,axis=1)                 

        else:
            yout_up = r_outer*np.sin(np.arccos(x_annulus/r_outer)) 
            yin_up = np.zeros(len(x_annulus))
            wdef_yin = np.abs((x_annulus/r_inner))<=1.
            yin_up[wdef_yin] = r_inner*np.sin(np.arccos(x_annulus[wdef_yin]/r_inner)) 
            yout_down = -yout_up
            yin_down = -yin_up

        #----------------------     
        #Stellar equator
        #    - stellar frame
        # Xstar = Rstar*cos(th)
        # Ystar = 0
        # Zstar = Rstar*sin(th)
        #    - in the Sky-projected stellar frame
        # Xsky_st = Rstar*cos(th)
        # Ysky_st = -cos(istar) Rstar*sin(th)  
        # Zsky_st =  sin(istar) Rstar*sin(th)
        #    - in the Sky-projected orbital frame
        # Xsky =  cos(lambda) Xsky_st + sin(lambda) Ysky_st
        # Ysky = -sin(lambda) Xsky_st + cos(lambda) Ysky_st
        # Zsky =  Zsky_st
        #      inflexion point in X vs Y (vertical) :
        # Xsky =  cos(lambda) Rstar*cos(th) - sin(lambda) cos(istar) Rstar*sin(th)
        # Ysky = -sin(lambda) Rstar*cos(th) - cos(lambda) cos(istar) Rstar*sin(th)
        # dXsky = -cos(lambda) Rstar*sin(th)*dth - sin(lambda) cos(istar) Rstar*cos(th)*dth
        # dXsky = 0 <-> -cos(lambda) sin(th) = sin(lambda) cos(istar) cos(th) 
        #               th_infl = arctan(-tan(lambda) cos(istar) ) in between -pi/2 and pi/2         
        #    - this is valid also if the star is oblate
        #----------------------   
    
        #Stellar equator:
        #    - theta and x,z coordinates defined to go along equator counter-clockwise, in the direction of the star motion, from 0 to -2pi 
        #      theta starts at 0 at the star center, then go to the eastern node at pi/2, then beyond the star, then through the western node at 3*pi/2, then back on the visible hemisphere.
        if plot_set_key['plot_equ_vis'] or plot_set_key['plot_equ_hid']: 
            n_pts_eq=1000.
            th_eq=2.*pi*np.arange(n_pts_eq+1.)/n_pts_eq      #in between 0 to -2pi                            
            x_eqst_st = np.sin(th_eq)
            y_eqst_st = 0.
            z_eqst_st = np.cos(th_eq)            
            x_eqst_sky_st,y_eqst_sky_st,z_eqst_sky_st=frameconv_star_to_skystar(x_eqst_st,y_eqst_st,z_eqst_st,istar_rad)
            if plot_set_key['conf_system']=='sky_orb': 
                x_eqst_view,y_eqst_view,z_eqst_view=frameconv_skystar_to_skyorb(lref,x_eqst_sky_st,y_eqst_sky_st,z_eqst_sky_st) 
                
                #Inflection points:
                th_infl = np.arctan(-np.tan(lref)*cos(istar_rad))
                if (th_infl>0.) & (th_infl<np.pi):
                    th_infl_vis = th_infl
                    th_infl_hid = th_infl-np.pi
                if (th_infl>-np.pi) & (th_infl<0):
                    th_infl_vis = th_infl-np.pi
                    th_infl_hid = th_infl
                ysky_infl_vis = -sin(lref)*cos(th_infl_vis) - cos(lref)*cos(istar_rad)*sin(th_infl_vis)
                ysky_infl_hid = -sin(lref)*cos(th_infl_hid) - cos(lref)*cos(istar_rad)*sin(th_infl_hid)

            if plot_set_key['conf_system']=='sky_ste':         
                x_eqst_view=x_eqst_sky_st
                y_eqst_view=y_eqst_sky_st
                z_eqst_view=z_eqst_sky_st 

            #Add arrow in the direction of the stellar rotation
            shaft_width_eq=0.001     #inches
            headwidth_eq=30          #units of shaft_width
            headlength_eq=25         #units of shaft_width
            headaxislength_eq=10     #units of shaft_width  

        #----------------------     
        #Stellar spin axis
        #    - stellar frame
        # + for a spherical star :
        #       Xstar=0
        #       Ystar = +- fact*Rstar 
        #       Zstar = 0 
        # + for an oblate star :
        #   the photosphere is defined by Xstar^2 + (Ystar/(1-f))^2 + Zstar^2 = 1, so that
        #       Xstar=0
        #       Ystar = +- fact*(1-f)*Rstar 
        #       Zstar = 0
        #    - in the Sky-projected stellar frame
        #       Xsky_st = 0
        #       Ysky_st = sin(istar) Ystar
        #       Zsky_st = cos(istar) Ystar
        #    - in the Sky-projected orbital frame
        #       Xsky = sin(lambda) sin(istar) Ystar
        #       Ysky = cos(lambda) sin(istar) Ystar
        #       Zsky = cos(istar) Ystar
        #    - we plot separately the axis parts going from the north or south poles 
        #----------------------   
        if plot_set_key['plot_stspin']:
        
            #Normalisation factor of the spin axis
            #    - in unit of Rstar 
#            st_spin_norm=1.1  #1.33   #1.08   
            st_spin_norm=1.3
            
            #Spin up and down coordinates
            if plot_set_key['conf_system']=='sky_orb':st_spin_up=st_spin_norm*Rpole*np.array([sin(lref)*sin(istar_rad),cos(lref)*sin(istar_rad),cos(istar_rad)])
            if plot_set_key['conf_system']=='sky_ste':st_spin_up=st_spin_norm*Rpole*np.array([0.,sin(istar_rad) ,cos(istar_rad)])
            st_spin_down=-st_spin_up
                
            #Resolution of the projected spin axis along x and y
            dst_spin=1./200.
            dx_spin=abs(st_spin_up[0]-st_spin_down[0])
            dy_spin=abs(st_spin_up[1]-st_spin_down[1])
            if dx_spin>dy_spin:nst_spin=round(dx_spin/dst_spin)
            else:nst_spin=round(dy_spin/dst_spin)
            if nst_spin<1:stop('Issue with stellar spin axis')
            
            #Coordinates of subpoints along the projected spin up and down axis, in chosen sky-projected frame
            dst_spin_x=(st_spin_up[0]-st_spin_down[0])/nst_spin
            st_spin_x=st_spin_down[0]+dst_spin_x*np.arange(nst_spin)
            dst_spin_y=(st_spin_up[1]-st_spin_down[1])/nst_spin
            st_spin_y=st_spin_down[1]+dst_spin_y*np.arange(nst_spin)
            dst_spin_z=(st_spin_up[2]-st_spin_down[2])/nst_spin
            st_spin_z=st_spin_down[2]+dst_spin_z*np.arange(nst_spin)

            #Coordinates of spin up and down axis, in sky-projected star frame   
            if plot_set_key['conf_system']=='sky_orb':st_spin_x_st,st_spin_y_st,st_spin_z_st=frameconv_skyorb_to_skystar(lref,st_spin_x,st_spin_y,st_spin_z) 
            elif plot_set_key['conf_system']=='sky_ste':st_spin_x_st,st_spin_y_st,st_spin_z_st = deepcopy(st_spin_x),deepcopy(st_spin_y),deepcopy(st_spin_z)
        
            #Keep parts of the axis not occulted by the star, and seen coming toward the observer, or moving away, from the observer
            #    - in the stellar frame the spin axis is always the vertical axis  
            lw_spin = 2.5
            if star_params['f_GD']>0.:
                
                #Points away from us, in LOS that do not intersect the projected photosphere
                idx_behind = np_where1D(st_spin_z_st < 0.)
                z_st_sky_behind,_,cond_in_stphot_behind=calc_zLOS_oblate(st_spin_x_st[idx_behind],st_spin_y_st[idx_behind],star_params['istar_rad'],star_params['RpoleReq'])                  
                w_vis_far = idx_behind[~cond_in_stphot_behind]
                               
                #Points away from us, in LOS that intersect the projected photosphere, outside of the photosphere   
                w_unvis_far = sorted(list(idx_behind[cond_in_stphot_behind][st_spin_z_st[idx_behind[cond_in_stphot_behind]] <=  z_st_sky_behind[cond_in_stphot_behind] ]))
                
                #Points toward us, in LOS that do not intersect the projected photosphere or in front of it
                idx_front = np_where1D(st_spin_z_st >= 0.)
                _,z_photo_front,cond_in_stphot_front=calc_zLOS_oblate(st_spin_x_st[idx_front],st_spin_y_st[idx_front],star_params['istar_rad'],star_params['RpoleReq'])                 
                w_vis_close = sorted(list(idx_front[~cond_in_stphot_front])+list(idx_front[cond_in_stphot_front][st_spin_z_st[idx_front[cond_in_stphot_front]] >=  z_photo_front[cond_in_stphot_front] ]))

                #Plot hidden spin axis
                if plot_set_key['plot_stspin_hid']:
                    w_vis_in= sorted(list(idx_behind[cond_in_stphot_behind][st_spin_z_st[idx_behind[cond_in_stphot_behind]]>  z_st_sky_behind[cond_in_stphot_behind] ])+ list(idx_front[cond_in_stphot_front][st_spin_z_st[idx_front[cond_in_stphot_front]] <  z_photo_front[cond_in_stphot_front] ]))

            else:
                r_st_spin2 = st_spin_x_st**2.+ st_spin_y_st**2.
                d_stspin_star2=r_st_spin2  +np.power(st_spin_z,2.)
                w_vis_far= np_where1D(( (d_stspin_star2>=1.) & (st_spin_z<0) & (r_st_spin2 >= 1.))  )    #part of the axis outside of the star, away from us, and outside the projected star disk
                w_unvis_far= np_where1D(( (d_stspin_star2>=1.) & (st_spin_z<0) & (r_st_spin2 < 1.))  )    #part of the axis outside of the star, away from us, and inside the projected star disk     
                w_vis_close= np_where1D(( (d_stspin_star2>=1.) & (st_spin_z>=0))  )    

                #Plot hidden spin axis
                if plot_set_key['plot_stspin_hid']:
                    w_vis_in= np_where1D( (d_stspin_star2<1.) | ((d_stspin_star2>=1.) & (st_spin_z<0) & (r_st_spin2 <= 1.))   ) #part of the axis within the star


            #---------------------- 
            #Plot spin vector head (emerging from north pole)						
            shaft_width_spin=0.003*2 
            headwidth_spin=10*2  
            headlength_spin=10
            if plot_set_key['conf_system']=='sky_ste':x_shaft_spin=0.

            #Last point of stellar spin is pointing toward us  
            if (nst_spin-1) in w_vis_close:
                w_vis_last = w_vis_close[-1]
                w_vis_first = w_vis_close[0]
                if plot_set_key['conf_system']=='sky_orb':x_shaft_spin = st_spin_x[w_vis_last]-st_spin_x[w_vis_first]
                y_shaft_spin = st_spin_y[w_vis_last]-st_spin_y[w_vis_first]

            #Last point of stellar spin is pointing away from us and is visible or not outside of the projected photosphere              
            elif ((nst_spin-1) in w_vis_far) or (plot_set_key['plot_hidden_pole'] and ((nst_spin-1) in w_unvis_far)):
                if ((nst_spin-1) in w_vis_far): 
                    w_vis_last = w_vis_far[-1]
                    w_vis_first = w_vis_far[0] 
                    ls_spin =  '-'
                    alph_spin = 1.
                elif ((nst_spin-1) in w_unvis_far):
                    w_vis_last = w_unvis_far[-1]
                    w_vis_first = w_unvis_far[0]   
                    ls_spin = '--'
                    alph_spin = 0.3
                if plot_set_key['conf_system']=='sky_orb':x_shaft = st_spin_x[w_vis_last]-st_spin_x[w_vis_first]
                y_shaft_spin = st_spin_y[w_vis_last]-st_spin_y[w_vis_first]

                  
        #-------------------------------------------------------               
        #Stellar poles
        #-------------------------------------------------------   
        if plot_set_key['plot_poles']:  

            #North and south poles coordinates
            if plot_set_key['conf_system']=='sky_orb':coord_Npole=Rpole*np.array([sin(lref)*sin(istar_rad),cos(lref)*sin(istar_rad),cos(istar_rad)])
            if plot_set_key['conf_system']=='sky_ste':coord_Npole=Rpole*np.array([0.,sin(istar_rad) ,cos(istar_rad)])
            coord_Spole=-coord_Npole 
            

        #-------------------------------------------------------
        #Limb-darkening and radial velocity at the surface of the star
        #-------------------------------------------------------
        coord_grid = {}

        #Coordinates of points discretizing the stellar disk
        #    - this grid must be perpendicular to the LOS, but is either considered as defined in the sky-projected orbital frame ('sky_orb') or in the sky-projected stellar frame  ('sky_ste')
        #    - between -Rstar_norm+0.5*d_stcell and Rstar_norm-0.5*d_stcell
        d_stcell,Ssub_Sstar,coord_grid['x_st_sky'],coord_grid['y_st_sky'],_=occ_region_grid(1.,plot_set_key['n_stcell'])

        #Grid coordinates in the sky-projected stellar frame      
        if plot_set_key['conf_system']=='sky_orb':  
            coord_grid['x_st_sky'],coord_grid['y_st_sky'],_=frameconv_skyorb_to_skystar(lref,coord_grid['x_st_sky'],coord_grid['y_st_sky'],None)          

        #Coordinates in the sky-projected star rest frame
        #    - vertical axis is the stellar spin
        #    - yprimperp = y_st_sky_grid si istar = 90
        nsub_star = calc_st_sky(coord_grid,star_params)

        #Flux emitted by each stellar cell 
        #    - assuming a maximum flux unity for plotting purposed (ie, dS = 1 and GD_max = 1 at the pole)
        if ('chrom' in data_dic['DI']['system_prop']):system_prop = data_dic['DI']['system_prop']['chrom']
        else:system_prop = data_dic['DI']['system_prop']['achrom']        
        ld_grid_star,gd_grid_star,mu_grid_star,Fsurf_grid_star,_,_ = calc_Isurf_grid(range(system_prop['nw']),nsub_star,system_prop,coord_grid,star_params,Ssub_Sstar)
        
        #Stellar surface radial velocity 
        RVstel = calc_RVrot(coord_grid['x_st_sky'],coord_grid['y_st'],istar_rad,star_params)[0]
        cb_band = calc_CB_RV(get_LD_coeff(system_prop,iband),system_prop['LD'][iband],star_params['c1_CB'],star_params['c2_CB'],star_params['c3_CB'],star_params) 
        for icb in range(4):RVstel+=cb_band[icb]*np.power(mu_grid_star[:,iband],icb)

        #--------------------------------------------
        #Orbits
        #--------------------------------------------
        def plot_orb_func(ax_plot,npts_orbits_plot,pl_params_plot,pl_ref,col_orb_plot,alph_front,alph_back,zord_plot,lw):
            
            #Orbit coordinates
            coord_orbit_loc=calc_pl_coord_plots(npts_orbits_plot,pl_params_plot)
            x_orbit_sky=coord_orbit_loc[0]
            y_orbit_sky=coord_orbit_loc[1]
            z_orbit_sky=coord_orbit_loc[2]            
            if plot_set_key['conf_system']=='sky_ste': ang_orb = pl_params_plot['lambda_rad']
            if plot_set_key['conf_system']=='sky_orb': ang_orb = pl_params_plot['lambda_rad']-lref    
            x_orbit_plot,y_orbit_plot,z_orbit_plot = frameconv_skyorb_to_skystar(ang_orb,x_orbit_sky,y_orbit_sky,z_orbit_sky)
            
            #Hidding part of the orbit behind the projected star
            #    - for a given (x,y) of the orbit at a negative z, in the sky-projected frame, we check if the photosphere determinant is positive, and if so if z_orb is lower than the lower z value of the photosphere at this (x,y) 
            #    - see calc_zLOS_oblate() for details on the calculations in the inclined star frame 
            if star_params['f_GD']>0.:
                cond_behind = (z_orbit_plot < 0.)
                z_photo_behind,_,cond_in_stphot=calc_zLOS_oblate(x_orbit_plot[cond_behind],y_orbit_plot[cond_behind],star_params['istar_rad'],star_params['RpoleReq'])    
                cond_novis = np.repeat(False,len(x_orbit_sky))
                idx_behind = np_where1D(cond_behind)
                idx_behind_sub = np_where1D(cond_in_stphot)[     z_orbit_plot[cond_behind][cond_in_stphot] <  z_photo_behind[cond_in_stphot] ] 
                cond_novis[ idx_behind[idx_behind_sub]   ] = True
            else:
                cond_novis = ( x_orbit_plot**2.+ y_orbit_plot**2. < 1. ) & (z_orbit_plot < 0.) 
            
            #Plotting differently parts of the orbit toward/aways from observer
            #    - the back orbit cannot be hidden by the plotted star via zorder because of the white annulus around the star
            dx_range = plot_set_key['x_range'][1]-plot_set_key['x_range'][0]
            dy_range = plot_set_key['y_range'][1]-plot_set_key['y_range'][0]
            cond_in_plot = (x_orbit_plot>=plot_set_key['x_range'][0]-0.1*dx_range) & (x_orbit_plot<=plot_set_key['x_range'][1]+0.1*dx_range) &  (y_orbit_plot>=plot_set_key['y_range'][0]-0.1*dy_range) & (y_orbit_plot<=plot_set_key['y_range'][1]+0.1*dy_range)
            w_front = (z_orbit_plot >= 0.) & cond_in_plot
            isort=np.argsort(x_orbit_plot[w_front])
            ax_plot.plot(x_orbit_plot[w_front][isort],y_orbit_plot[w_front][isort],zorder=zord_plot, color=col_orb_plot,lw=plot_set_key['lw_plot'],alpha=alph_front)

            w_back = (z_orbit_plot < 0.) & cond_in_plot
            isort=np.argsort(x_orbit_plot[w_back])
            x_orbit_back = x_orbit_plot[w_back][isort]
            y_orbit_back = y_orbit_plot[w_back][isort]
            cond_novis_back = cond_novis[w_back][isort]
            x_orbit_back[cond_novis_back]=np.nan
            y_orbit_back[cond_novis_back]=np.nan           
            ax_plot.plot(x_orbit_back,y_orbit_back,zorder=zord_plot, color=col_orb_plot,lw=plot_set_key['lw_plot'],alpha=alph_back)

            return x_orbit_plot,y_orbit_plot,z_orbit_plot            

        def sem_norm_dist(par_mean,par_elow,par_ehigh):
            n_dist = 1000
            rand_draw_right = np.random.normal(loc=par_mean, scale=par_ehigh, size=int(2.5*n_dist))
            rand_draw_right = rand_draw_right[rand_draw_right>par_mean]
            rand_draw_right = rand_draw_right[0:n_dist]
            rand_draw_left = np.random.normal(loc=par_mean, scale=par_elow, size=int(2.5*n_dist))
            rand_draw_left = rand_draw_left[rand_draw_left<=par_mean]
            rand_draw_left = rand_draw_left[0:n_dist]
            chain_par = np.append(rand_draw_left,rand_draw_right)
            hist_val, bin_edg_val = np.histogram(chain_par, bins=1000,density=True)
            grid_val = 0.5*(bin_edg_val[0:-1]+bin_edg_val[1:])
            cdf_val = np.cumsum(hist_val)
            cdf_val = (cdf_val-np.min(cdf_val))/(np.max(cdf_val)-np.min(cdf_val))            
            return grid_val,cdf_val             
    
        
    
    
    
        
        #--------------------------------------------
        #Plot series
        if plot_set_key['t_BJD'] is None:plot_series=['']
        else:plot_series = plot_set_key['t_BJD']['t']-2400000. 

        #Making a GIF if we have multiple exposures of the system
        if (plot_set_key['t_BJD'] is not None) and plot_set_key['GIF_generation']:
            if plot_dic['system_view']!='png':stop('GIF generation requires png format')

            #Initialize a list to store the images used to make the GIF.
            images_to_make_GIF = [] 

            #Initializing a list to store the image paths.
            filenames = []
            
        else:images_to_make_GIF = None
        
        #Processing exposures
        for idx_pl,plot_t in enumerate(plot_series):
            
            #--------------------------------------------
            #Frame axis
            #    - we only plot the sky-projected X and Y axis
            #---------
            plt.ioff()        
            fig = plt.figure(figsize=(plot_set_key['width'],plot_set_key['width']))
            ax1=fig.add_axes([plot_set_key['margins'][0],plot_set_key['margins'][1],plot_set_key['margins'][2]-plot_set_key['margins'][0],plot_set_key['margins'][3]-plot_set_key['margins'][1]])
        
            if plot_set_key['axis_overlay']:
                
                #Axis properties
                axis_norm=20.
                axis_shaft_width=0.003 
                axis_headwidth=10  
                axis_headlength=10
            
                #Axis
                ax1.quiver(0,0,axis_norm,0.,color='gray',angles='xy',scale_units='xy',scale=1,zorder=-10,width=axis_shaft_width,headwidth=axis_headwidth, headlength=axis_headlength)
                ax1.quiver(0,0,0.,axis_norm,color='gray',angles='xy',scale_units='xy',scale=1,zorder=-10,width=axis_shaft_width,headwidth=axis_headwidth, headlength=axis_headlength)

            #---------------------- 
            #Star boundary
            #---------------------- 
            fig.gca().fill_between(x_annulus, yin_up, yout_up, color="white",zorder=0,lw=0)
            fig.gca().fill_between(x_annulus, yout_down, yin_down, color="white",zorder=0,lw=0)   
            
            #Plot all required planets
            n_pl = len(plot_set_key['pl_to_plot'])
            for ipl,pl_loc in enumerate(plot_set_key['pl_to_plot']):
          
                #Planet orbit	
                #    - 'coord_orbit' is defined in the Sky-projected orbital frame: Xsky,Ysky,Zsky	
                x_orbit_view,y_orbit_view,z_orbit_view  = plot_orb_func(ax1,plot_set_key['npts_orbits'][ipl],system_param[pl_loc],pl_loc,plot_set_key['col_orb'][ipl],1.,0.4,40.+2*ipl+1,2)
                
                #Uncertainty of planet orbit
                if (pl_loc in plot_set_key['lambdeg_err']) or (pl_loc in plot_set_key['aRs_err']) or (pl_loc in plot_set_key['ip_err']):
                    if (pl_loc in plot_set_key['lambdeg_err']):
                        grid_val,cdf_val = sem_norm_dist(system_param[pl_loc]['lambda_rad'],plot_set_key['lambdeg_err'][pl_loc][0]*np.pi/180.,plot_set_key['lambdeg_err'][pl_loc][1]*np.pi/180.)
                        rand_draw = np.random.uniform(low=0.0, high=1.0, size=plot_set_key['norb'])    
                        lambdarad_tab = np_interp(rand_draw,cdf_val,grid_val)
                        lambdeg_range = [system_param[pl_loc]['lambda_rad']-plot_set_key['lambdeg_err'][pl_loc][0]*np.pi/180.,system_param[pl_loc]['lambda_rad']+plot_set_key['lambdeg_err'][pl_loc][1]*np.pi/180.]
                    else:
                        lambdarad_tab = np.repeat(system_param[pl_loc]['lambda_rad'],plot_set_key['norb'])
                        lambdeg_range=[-1e10,1e10]
                    if (pl_loc in plot_set_key['aRs_err']):
                        grid_val,cdf_val = sem_norm_dist(system_param[pl_loc]['aRs'],plot_set_key['aRs_err'][pl_loc][0],plot_set_key['aRs_err'][pl_loc][1])
                        rand_draw = np.random.uniform(low=0.0, high=1.0, size=plot_set_key['norb'])    
                        aRs_tab = np_interp(rand_draw,cdf_val,grid_val)    
                        aRs_range = [system_param[pl_loc]['aRs']-plot_set_key['aRs_err'][pl_loc][0],system_param[pl_loc]['aRs']+plot_set_key['aRs_err'][pl_loc][1]]               
                    else:
                        aRs_tab = np.repeat(system_param[pl_loc]['aRs'],plot_set_key['norb'])
                        aRs_range=[0.,1e10]
                    if (pl_loc in plot_set_key['ip_err']):
                        grid_val,cdf_val = sem_norm_dist(system_param[pl_loc]['inclin_rad'],plot_set_key['ip_err'][pl_loc][0]*np.pi/180.,plot_set_key['ip_err'][pl_loc][1]*np.pi/180.)
                        rand_draw = np.random.uniform(low=0.0, high=1.0, size=plot_set_key['norb'])    
                        ip_tab = np_interp(rand_draw,cdf_val,grid_val)
                        ip_range = [system_param[pl_loc]['inclin_rad']-plot_set_key['ip_err'][pl_loc][0]*np.pi/180.,system_param[pl_loc]['inclin_rad']+plot_set_key['ip_err'][pl_loc][1]*np.pi/180.]  
                    else:
                        ip_tab = np.repeat(system_param[pl_loc]['inclin_rad'],plot_set_key['norb'])
                        ip_range=[-1e10,1e10]
                    b_tab = np.abs(aRs_tab*np.cos(ip_tab))
                    if (pl_loc in plot_set_key['b_range_all']):b_range=plot_set_key['b_range_all'][pl_loc]
                    else:b_range=[-1e10,1e10]
                 
                    #Keep orbits within selected 1 sigma ranges
                    cond_keep = ((lambdarad_tab>=lambdeg_range[0]) & ((lambdarad_tab<=lambdeg_range[1]))) &\
                        ((aRs_tab>=aRs_range[0]) & ((aRs_tab<=aRs_range[1]))) &\
                        ((ip_tab>=ip_range[0]) & ((ip_tab<=ip_range[1])))    &\
                        ((b_tab>=b_range[0]) & ((b_tab<=b_range[1])))                   
                    lambdarad_tab=lambdarad_tab[cond_keep]
                    aRs_tab=aRs_tab[cond_keep]
                    ip_tab=ip_tab[cond_keep]
                    print('   - plotting ',np.sum(cond_keep),' orbits for ',pl_loc)
                    for iorb,(lambdarad_loc,aRs_loc,ip_loc) in enumerate(zip(lambdarad_tab,aRs_tab,ip_tab)):
                        pl_params_orb = deepcopy(system_param[pl_loc])                
                        pl_params_orb['lambda_rad'] = lambdarad_loc
                        pl_params_orb['aRs'] = aRs_loc
                        pl_params_orb['inclin_rad'] = ip_loc
                        _,_,_ = plot_orb_func(ax1,plot_set_key['npts_orbits'][ipl],pl_params_orb,pl_loc,plot_settings[key_plot]['col_orb_samp'][ipl],0.2,0.03,40.+2*ipl,1)
             
                #Planet at given position along the orbit
                RpRs = plot_set_key['RpRs_pl'][ipl]
                if plot_set_key['t_BJD'] is not None:
                    phase_pl=get_timeorbit(pl_loc,coord_dic[plot_set_key['t_BJD']['inst']][plot_set_key['t_BJD']['vis']],plot_t,system_param[pl_loc],0.)[1]  
                    x_pl_sky,y_pl_sky,z_pl_sky= calc_pl_coord(system_param[pl_loc]['ecc'],system_param[pl_loc]['omega_rad'],system_param[pl_loc]['aRs'],system_param[pl_loc]['inclin_rad'],phase_pl,None,None,None)[0:3]               
                    if plot_set_key['conf_system']=='sky_ste': ang_orb = system_param[pl_loc]['lambda_rad']
                    if plot_set_key['conf_system']=='sky_orb': ang_orb = system_param[pl_loc]['lambda_rad']-lref    
                    x_pl_plot,y_pl_plot,_ = frameconv_skyorb_to_skystar(ang_orb,x_pl_sky,y_pl_sky,z_pl_sky)       
                else:
                    w_visible=np.where((z_orbit_view>0) & (x_orbit_view>=plot_set_key['xorp_pl'][ipl][0]) & (x_orbit_view<=plot_set_key['xorp_pl'][ipl][1]))[0]
                    if len(w_visible)==0.:stop('No visible orbit in requested range')
                    w_pl=closest(y_orbit_view[w_visible],plot_set_key['yorb_pl'][ipl])
                    x_pl_plot = x_orbit_view[w_visible][w_pl] 
                    y_pl_plot = y_orbit_view[w_visible][w_pl] 
                col_pl = 'black'
                alph_pl = 1
                # col_pl = plot_settings[key_plot]['col_orb'][ipl]
                # alph_pl = 0.8
                plt.fill((x_pl_plot+RpRs*np.cos(2*pi*np.arange(101)/100.)),(y_pl_plot+RpRs*np.sin(2*pi*np.arange(101)/100.)),col_pl,zorder=40+2*n_pl+1,lw = 0,alpha=alph_pl)         
                ax1.add_artist(plt.Circle((x_pl_plot,y_pl_plot),RpRs,color=plot_set_key['col_orb'][ipl],fill=False,lw=1,zorder=40+2*n_pl+1)   )
                        
                #Overlaying planet grid cell boundaries
                if plot_set_key['pl_grid_overlay']:       
                    col_grid = plot_settings[key_plot]['col_orb'][ipl]
                    lw_st_grid=1.
                    coord_pl_grid = {}      
                    d_sub,_,x_st_sky_grid_pl,y_st_sky_grid_pl,r_sub_pl2=occ_region_grid(RpRs,plot_set_key['n_plcell'][pl_loc])
                    cond_in_RpRs = (r_sub_pl2<RpRs**2.)
                    coord_pl_grid['x_st_sky'] = x_pl_plot+x_st_sky_grid_pl[cond_in_RpRs] 
                    coord_pl_grid['y_st_sky'] = y_pl_plot+y_st_sky_grid_pl[cond_in_RpRs] 
                    if plot_set_key['conf_system']=='sky_orb':
                        coord_pl_grid['z_st_sky'] = np.sqrt(1.-coord_pl_grid['x_st_sky']**2.-coord_pl_grid['y_st_sky']**2.)
                        coord_pl_grid['x_st_sky'],coord_pl_grid['x_st_sky'],coord_pl_grid['z_st_sky']=frameconv_skyorb_to_skystar(lref,coord_pl_grid['x_st_sky'],coord_pl_grid['y_st_sky'],coord_pl_grid['z_st_sky']) 
                    n_pl_occ = calc_st_sky(coord_pl_grid,star_params)  
                    if n_pl_occ>0:
                        if plot_set_key['conf_system']=='sky_orb':                    
                            coord_pl_grid['x_st_sky'],coord_pl_grid['y_st_sky'],coord_pl_grid['z_st_sky']=frameconv_skystar_to_skyorb(lref,coord_pl_grid['x_st_sky'],coord_pl_grid['y_st_sky'],coord_pl_grid['z_st_sky']) 
                        else:
                            coord_pl_grid['x_st_sky'],coord_pl_grid['y_st_sky'],coord_pl_grid['z_st_sky'] = coord_pl_grid['x_st_sky'],coord_pl_grid['y_st_sky'],coord_pl_grid['z_st_sky']
                        for xcell,ycell in zip(coord_pl_grid['x_st_sky'],coord_pl_grid['y_st_sky']):
                            ax1.add_artist(plt.Rectangle((xcell-0.5*d_sub,ycell-0.5*d_sub), d_sub, d_sub, fc='none',ec=col_grid,lw=lw_st_grid,zorder=40+2*n_pl+1))
                    
                #Add arrow in the direction of the planet orbital motion
                shaft_width_orb=0.001     #inches
                headwidth_orb=30          #units of shaft_width
                headlength_orb=25         #units of shaft_width
                headaxislength_orb=10     #units of shaft_width  
    
                w_visible=np.where((z_orbit_view>0) & (x_orbit_view>=plot_set_key['xorb_dir'][ipl][0]) & (x_orbit_view<=plot_set_key['xorb_dir'][ipl][1]))[0]
                if len(w_visible)==0.:stop('No visible orbit in requested range')
                w_orien=closest(y_orbit_view[w_visible],plot_set_key['yorb_dir'][ipl])    
                x_shaft = 40.*(x_orbit_view[w_visible][w_orien]-x_orbit_view[w_visible][w_orien-1]) 
                y_shaft = 40.*(y_orbit_view[w_visible][w_orien]-y_orbit_view[w_visible][w_orien-1]) 
                ax1.quiver(x_orbit_view[w_visible][w_orien],y_orbit_view[w_visible][w_orien],0.2*np.sign(x_shaft),0.2*np.abs(y_shaft/x_shaft)*np.sign(y_shaft),color=plot_set_key['col_orb'][ipl],
                           angles='xy',scale_units='inches',scale=1,zorder=40+2*n_pl,width=shaft_width_orb,headwidth=headwidth_orb, headlength=headlength_orb,headaxislength=headaxislength_orb,pivot='tip')  #head
    
                #-------------------------------------------------------
                #Normal to the orbital planes
                #    - coordinates in the orbital frame:
                # Xorb = 0
                # Yorb = fact Rstar
                # Zorb = 0 
                #    - coordinates in the Sky-projected orbital frame:
                #      rotation by Inclin (angle from the LOS Zsky to the normal to the orbital plane Ysky) around the Xorb axis:
                # Xsky = 0
                # Ysky = -Zorb*cos(Inclin)+Yorb*sin(Inclin) = fact Rstar sin(Inclin) 
                # Zsky =  Zorb*sin(Inclin)+Yorb*cos(Inclin) = fact Rstar cos(Inclin) 
                #    - coordinates in the Sky-projected stellar frame:
                # Xsky_st = - sin(lambda) fact Rstar sin(Inclin)
                # Ysky_st =   cos(lambda) fact Rstar sin(Inclin)
                # Zsky_st =   fact Rstar cos(Inclin)
                #---------
                #For additional planets
                #    - coordinates in the Sky-projected orbital frame (ie it refers to the main planet):
                # Xsky[2nd_pl] = 0
                # Ysky[2nd_pl] = fact Rstar sin(Inclin[2nd_pl]) 
                # Zsky[2nd_pl] = fact Rstar cos(Inclin[2nd_pl])     
                #      thus
                # Xsky = cos(Dlambda) Xsky[2nd_pl] - sin(Dlambda) Ysky[2nd_pl] = - fact Rstar sin(Inclin[2nd_pl]) sin(Dlambda)
                # Ysky = sin(Dlambda) Xsky[2nd_pl] + cos(Dlambda) Ysky[2nd_pl] =   fact Rstar sin(Inclin[2nd_pl]) cos(Dlambda) 
                # Zsky = Zsky[2nd_pl] = fact Rstar cos(Inclin[2nd_pl])
                #      avec Dlambda = lambda[2nd_pl] - lambda 
                #    - coordinates in the Sky-projected stellar frame:
                # same as for the main planet, with the second planet properties
                #--------------------------------------------   
               
                orb_spin_norm=1.1  #1.3    #plot units
                shaft_width_norm=0.005    #inches
                headwidth_norm=12         #units of shaft_width
                headlength_norm=7         #units of shaft_width
                headaxislength_norm=4     #units of shaft_width       
                
                #Coordinates of axis point in sky-projected star frame
                if plot_set_key['conf_system']=='sky_ste':ang_normal =  system_param[pl_loc]['lambda_rad']   
                if plot_set_key['conf_system']=='sky_orb':ang_normal =  system_param[pl_loc]['lambda_rad']-lref   
                orb_spin_xyzview=orb_spin_norm*np.array([-sin(ang_normal)*sin(system_param[pl_loc]['inclin_rad']),cos(ang_normal)*sin(system_param[pl_loc]['inclin_rad']),cos(system_param[pl_loc]['inclin_rad'])]) 
        
                #Plotting visible shaft
                #    - here we show the orbital plane normal if it is at z>0, or at z<0 outside / in front of the projected photosphere
                n_shaft = 500
                x_shaft = np.linspace(0,0.98*orb_spin_xyzview[0],n_shaft)
                y_shaft = np.linspace(0,0.98*orb_spin_xyzview[1],n_shaft)
                z_shaft = np.linspace(0,0.98*orb_spin_xyzview[2],n_shaft)
                if star_params['f_GD']>0.:
                    idx_behind = np_where1D(z_shaft < 0.)
                    z_photo_behind,_,cond_in_stphot=calc_zLOS_oblate(x_shaft[idx_behind],y_shaft[idx_behind],star_params['istar_rad'],star_params['RpoleReq'])    
                    w_novis = idx_behind[cond_in_stphot][z_shaft[idx_behind[cond_in_stphot]] <  z_photo_behind[cond_in_stphot] ]                
                else:
                    w_novis=np_where1D( ( x_shaft**2.+ y_shaft**2.  < 1. ) & (z_shaft < 0.) )    
                x_shaft[w_novis]=np.nan
                y_shaft[w_novis]=np.nan                       
                plt.plot(x_shaft,y_shaft,lw=3,color=plot_set_key['col_orb'][ipl],zorder=30+2*n_pl)    #shaft             
                
                #Plotting visible axis head
                r_head2 = orb_spin_xyzview[0]**2.+ orb_spin_xyzview[1]**2.
                if star_params['f_GD']>0.:
                    if (orb_spin_xyzview[2]>=0):cond_vis=True
                    else:
                        Bquad = 2.*orb_spin_xyzview[1]*ci*si*mRp2        
                        Cquad = orb_spin_xyzview[1]**2.*si**2.*mRp2 + Rpole**2.*(r_head2  - 1.)                   
                        det = Bquad**2.-4.*Aquad*Cquad
                        if det<0:cond_vis=True
                        else:cond_vis = orb_spin_xyzview[2] >=  (-Bquad+np.sqrt(det))/(2.*Aquad)         
                else:cond_vis= ( r_head2 >= 1. ) | (orb_spin_xyzview[2] >= 0.)          
                if cond_vis:
                    if abs(orb_spin_xyzview[0])<2e-2:
                        xdir_quiv = 0.
                        ydir_quiv = 0.05*np.sign(orb_spin_xyzview[1])
                    else:
                        xdir_quiv = 0.05*np.sign(orb_spin_xyzview[0])
                        ydir_quiv = 0.05*np.abs(orb_spin_xyzview[1]/orb_spin_xyzview[0])*np.sign(orb_spin_xyzview[1])   
                    ax1.quiver(orb_spin_xyzview[0],orb_spin_xyzview[1],xdir_quiv,ydir_quiv,color=plot_set_key['col_orb'][ipl],
                                angles='xy',scale_units='inches',scale=1,minlength=0,minshaft=0,zorder=30+2*n_pl,width=shaft_width_norm,headwidth=headwidth_norm, headlength=headlength_norm,headaxislength=headaxislength_norm,pivot='tip')  #head


            #----------------------           
            #Stellar equator
            #----------------------   
            if plot_set_key['plot_equ_vis'] or plot_set_key['plot_equ_hid']: 
                 
                #Visible part  
                if plot_set_key['plot_equ_vis']:
                    cond_vis_up=( z_eqst_view >= 0. )
                    cond_vis_down = (z_eqst_view >= 0.)
                    if plot_set_key['conf_system']=='sky_orb':
                        cond_vis_up &= (y_eqst_view > ysky_infl_vis) 
                        cond_vis_down  &=  (y_eqst_view <= ysky_infl_vis) 
                        isort_down=np.argsort(x_eqst_view[cond_vis_down])
                        plt.plot(x_eqst_view[cond_vis_down][isort_down],y_eqst_view[cond_vis_down][isort_down],zorder=25., color='black',lw=1.,alpha=1)  
                    isort_up=np.argsort(x_eqst_view[cond_vis_up])
                    plt.plot(x_eqst_view[cond_vis_up][isort_up],y_eqst_view[cond_vis_up][isort_up],zorder=25., color='black',lw=1.,alpha=1)   

                    #Add arrow in the direction of the stellar rotation
                    w_vis=np.where(cond_vis_up & (x_eqst_view>=plot_set_key['xst_dir'][0]) & (x_eqst_view<=plot_set_key['xst_dir'][1]))[0]
                    w_orien=closest(y_eqst_view[w_vis],plot_set_key['yst_dir'])   
                    if (plot_set_key['conf_system']=='sky_orb') and ((w_orien==0) or (w_orien==np.sum(cond_vis_up)-1)):
                        w_vis=np.where(cond_vis_down & (x_eqst_view>=plot_set_key['xst_dir'][0]) & (x_eqst_view<=plot_set_key['xst_dir'][1]))[0]
                        w_orien=closest(y_eqst_view[w_vis],plot_set_key['yst_dir']) 
                    x_shaft = 40.*(x_eqst_view[w_vis[w_orien]]-x_eqst_view[w_vis[w_orien]-1]) 
                    y_shaft = 40.*(y_eqst_view[w_vis[w_orien]]-y_eqst_view[w_vis[w_orien]-1]) 
                    ax1.quiver(x_eqst_view[w_vis][w_orien],y_eqst_view[w_vis][w_orien],0.2*np.sign(x_shaft),0.2*np.abs(y_shaft/x_shaft)*np.sign(y_shaft),color='black',
                               angles='xy',scale_units='inches',scale=1,zorder=25,width=shaft_width_eq,headwidth=headwidth_eq, headlength=headlength_eq,headaxislength=headaxislength_eq,pivot='tip')  #head

                #Hidden part
                if plot_set_key['plot_equ_hid']:
                    if plot_set_key['conf_system']=='sky_orb':
                        w_hid_up=np.where( (z_eqst_view < 0.) & (y_eqst_view > ysky_infl_hid) )[0]  
                        w_hid_down=np.where( (z_eqst_view < 0.) & (y_eqst_view <= ysky_infl_hid) )[0]
                        isort_down=np.argsort(x_eqst_view[w_hid_down])
                        plt.plot(x_eqst_view[w_hid_down][isort_down],y_eqst_view[w_hid_down][isort_down],zorder=25., color='black',lw=1.,linestyle='--',alpha=0.5)                  
                    if plot_set_key['conf_system']=='sky_ste':  
                        w_hid_up=np.where( (z_eqst_view < 0.))[0]
                    isort_up=np.argsort(x_eqst_view[w_hid_up])
                    plt.plot(x_eqst_view[w_hid_up][isort_up],y_eqst_view[w_hid_up][isort_up],zorder=25., color='black',lw=1.,linestyle='--',alpha=0.5)   



            #----------------------     
            #Stellar spin axis
            #----------------------   
            if plot_set_key['plot_stspin']:

                #Plotting visible parts of spin axis    
                plt.plot(st_spin_x[w_vis_close],st_spin_y[w_vis_close],color='black',lw=lw_spin,linestyle='-')
                plt.plot(st_spin_x[w_vis_far],st_spin_y[w_vis_far],color='black',lw=lw_spin,linestyle='-')        
                if plot_set_key['plot_hidden_pole']:plt.plot(0.99*st_spin_x[w_unvis_far],0.99*st_spin_y[w_unvis_far],color='black',lw=lw_spin,linestyle='--',alpha=0.3)  
        
                #Plotting spin axis within the star
                if plot_set_key['plot_stspin_hid']:
                    plt.plot(st_spin_x[w_vis_in],st_spin_y[w_vis_in],color='black',lw=lw_spin-0.5,linestyle='--')        
        
                #---------------------- 
                #Plot spin vector head (emerging from north pole)						

                #Last point of stellar spin is pointing toward us  
                if (nst_spin-1) in w_vis_close:
                    ax1.quiver(st_spin_x[w_vis_last],st_spin_y[w_vis_last],0.2*np.sign(x_shaft_spin)*np.abs(x_shaft_spin/y_shaft_spin),0.2*np.sign(y_shaft_spin),
                                color='black',angles='xy',scale_units='inches',scale=1,zorder=10,width=shaft_width_spin,headwidth=headwidth_spin, headlength=headlength_spin,linestyle='-',pivot='tip')
                
                #Last point of stellar spin is pointing away from us and is visible or not outside of the projected photosphere              
                elif ((nst_spin-1) in w_vis_far) or (plot_set_key['plot_hidden_pole'] and ((nst_spin-1) in w_unvis_far)):
                    ax1.quiver(st_spin_x[w_vis_last],st_spin_y[w_vis_last],0.2*np.sign(x_shaft_spin)*np.abs(x_shaft_spin/y_shaft_spin),0.2*np.sign(y_shaft_spin),alpha=alph_spin,
                                color='black',angles='xy',scale_units='inches',scale=1,zorder=10,width=shaft_width_spin,headwidth=headwidth_spin, headlength=headlength_spin,linestyle=ls_spin,pivot='tip')
    
                
                      
            #-------------------------------------------------------               
            #Stellar poles
            #-------------------------------------------------------   
            if plot_set_key['plot_poles']:  

                #Plot North pole
                if (coord_Npole[2]>0.) or (plot_set_key['plot_hidden_pole']): 
                    if coord_Npole[2]>0.:alph_loc = 1.
                    else:alph_loc = 0.3
                    plt.plot(coord_Npole[0],coord_Npole[1],zorder=4., color='black',marker='o',markersize=6,alpha = alph_loc,markerfacecolor='black') 
    
                #Plot South pole
                if (coord_Spole[2]>0.) or (plot_set_key['plot_hidden_pole']):  
                    if coord_Spole[2]>0.:alph_loc = 1.
                    else:alph_loc = 0.3
                    plt.plot(coord_Spole[0],coord_Spole[1],zorder=4., color='black',marker='o',markersize=6,alpha = alph_loc,markerfacecolor='black') 
                

            #-------------------------------------------------------
            #Spotted cells 
            #-------------------------------------------------------
            if plot_set_key['plot_spots']:

                #Custom spot properties
                if len(plot_set_key['custom_spot_prop'])>0:
                    if idx_pl==0: print('   + With custom spot properties')

                    # Initialize params to use the retrieve_spots_prop_from_param function.
                    params = {'cos_istar' : star_params['cos_istar'], 'alpha_rot' : star_params['alpha_rot'], 'beta_rot' : star_params['beta_rot'] }        
                    for spot in plot_set_key['custom_spot_prop'] : 
                        params['lat__IS__VS__SP'+spot]     = plot_set_key['custom_spot_prop'][spot]['lat']
                        params['ang__IS__VS__SP'+spot]     = plot_set_key['custom_spot_prop'][spot]['ang']
                        params['Tcenter__IS__VS__SP'+spot] = plot_set_key['custom_spot_prop'][spot]['Tcenter']
                        params['ctrst__IS__VS__SP'+spot]    = plot_set_key['custom_spot_prop'][spot]['ctrst']
                    
                #Mock dataset spot properties
                elif plot_set_key['mock_spot_prop']:
                    if idx_pl==0: print('   + With mock dataset spot properties')
                    if (mock_dic['spots_prop'] != {}):
                            inst_to_use = plot_set_key['inst_to_plot'][0]
                            vis_to_use = plot_set_key['visits_to_plot'][inst_to_use][0]
                            
                            #Retrieve the spot parameters from the mock dictionary
                            params = deepcopy(mock_dic['spots_prop'][inst_to_use][vis_to_use])
                            params['cos_istar'] = star_params['cos_istar'] 
                            params['alpha_rot'] = star_params['alpha_rot']
                            params['beta_rot'] = star_params['beta_rot']   
                    else:stop('Mock spot properties undefined for this system.')   
                    
                #Fitted spot properties
                elif plot_set_key['fit_spot_prop']:
                    if idx_pl==0: print('   + With fitted spot properties')
                    inst_to_use = plot_set_key['inst_to_plot'][0]
                    vis_to_use = plot_set_key['visits_to_plot'][inst_to_use][0]
                    if plot_set_key['fit_results_file'] !='':fit_res = dataload_npz(plot_set_key['fit_results_file'])
                    else:stop('No best-fit output file provided.')
                    params = fit_res['p_final']
                else:stop('System view unavailable : Spot generation initialized with no spot properties provided')
  
                #Spot Equatorial rotation rate (rad/s)
                if 'veq_spots' in star_params:star_params['om_eq_spots']=star_params['veq_spots']/star_params['Rstar_km']
                else:star_params['om_eq_spots']=star_params['om_eq']

                #Calculate the flux of star grid, at all the exposures considered
                star_flux_before_spot = deepcopy(Fsurf_grid_star[:,iband])

                #Check if we have provided times for the plotting
                if plot_set_key['t_BJD'] is not None:
                    
                    #Accounting for spot overlap
                    shared_spotted_tiles = np.zeros(len(coord_grid['x_st_sky']), dtype=bool)
                                        
                    #Defining a new flux array, such that we don't see the previous spot when plotting the next spot
                    Flux_for_nonredundant_spot_plotting = deepcopy(Fsurf_grid_star[:,iband])
                    if len(plot_set_key['custom_spot_prop'])>0:
                        spots_prop = retrieve_spots_prop_from_param(star_params, params, '_', '_', plot_t) 
                    else:
                        spots_prop = retrieve_spots_prop_from_param(star_params, params, inst_to_use, vis_to_use, plot_t) 

                    #Defining a reference spot contrast - since all spots share the same contrast
                    ref_ctrst = spots_prop[list(spots_prop.keys())[0]]['ctrst']

                    for spot in spots_prop :
                        if spots_prop[spot]['is_center_visible']: 
                            _, spotted_tiles = calc_spotted_tiles(spots_prop[spot], coord_grid['x_st_sky'], coord_grid['y_st_sky'], coord_grid['z_st_sky'], 
                                                                   {}, params, use_grid_dic = False, disc_exp = False)

                            #Accounting for spot overlap - figure out which cells of the stellar grid are spotted
                            shared_spotted_tiles |= spotted_tiles
                            
                    #Accounting for spot overlap - all spots share the same contrast. As such we figure out which cells are spotted by either spot1, or spot2, or...
                    #Once we have all the spotted cells we scale their flux with the shared contrast. This prevents us of counting shared cells multiple times and 
                    #lowering their flux more than necessary.
                    star_flux_before_spot[shared_spotted_tiles] *=  (1-ref_ctrst)

                    
                    if plot_set_key['spot_overlap']:      
                        Fsurf_grid_star[:,iband] = np.minimum(Fsurf_grid_star[:,iband], star_flux_before_spot)
                    else:
                        Flux_for_nonredundant_spot_plotting = np.minimum(Fsurf_grid_star[:,iband], star_flux_before_spot)
                #If no times provided for the plotting, then generate some
                else:
                    # Compute BJD of images 
                    if plot_set_key['plot_spot_all_Peq'] :
                        #Getting reference spot and latitude
                        for par in params:
                            if 'lat' in par:
                                ref_lat = params[par]
                                break
                        sin_lat = np.sin(ref_lat) 
                        t_all_spot = np.linspace(0,2*np.pi/((1.-star_params['alpha_rot_spots']*sin_lat**2.-star_params['beta_rot_spots']*sin_lat**4.)*star_params['om_eq_spots']*3600.*24.),plot_set_key['n_image_spots'])
                    else:
                        dbjd =  (plot_set_key['time_range_spot'][1]-plot_set_key['time_range_spot'][0])/plot_set_key['n_image_spots']
                        n_in_visit = int((plot_set_key['time_range_spot'][1]-plot_set_key['time_range_spot'][0])/dbjd)
                        bjd_exp_low = plot_set_key['time_range_spot'][0] + dbjd*np.arange(n_in_visit)
                        bjd_exp_high = bjd_exp_low+dbjd      
                        t_all_spot = 0.5*(bjd_exp_low+bjd_exp_high) - 2400000.
                    for t_exp in t_all_spot :
                        star_flux_exp = deepcopy(star_flux_before_spot)
                        #Accounting for spot overlap
                        shared_spotted_tiles = np.zeros(len(coord_grid['x_st_sky']), dtype=bool)
                        if len(plot_set_key['custom_spot_prop'])>0:
                            spots_prop = retrieve_spots_prop_from_param(star_params, params, '_', '_', t_exp) 
                        else:
                            spots_prop = retrieve_spots_prop_from_param(star_params, params, inst_to_use, vis_to_use, t_exp) 
                            
                        #Defining a reference spot contrast - since all spots share the same contrast
                        ref_ctrst = spots_prop[list(spots_prop.keys())[0]]['ctrst']
                            
                        for spot in spots_prop :
                            if spots_prop[spot]['is_center_visible']:
                                _, spotted_tiles = calc_spotted_tiles(spots_prop[spot], coord_grid['x_st_sky'], coord_grid['y_st_sky'], coord_grid['z_st_sky'], 
                                                                       {}, params, use_grid_dic = False, disc_exp = False)

                                #Accounting for spot overlap - figure out which cells of the stellar grid are spotted
                                shared_spotted_tiles |= spotted_tiles
                        
                        star_flux_exp[shared_spotted_tiles] *=  (1-ref_ctrst)
                        Fsurf_grid_star[:,iband] = np.minimum(Fsurf_grid_star[:,iband], star_flux_exp)


        

    
            #------------------------------------------------------------          
            #Color table (from 0 to 1)
            #    - put '_r' at the end to reverse color table 
            #    - min_col/max_col limit the color range to be used in the color table
            # 	   min_val/max_val control the range of values to be attributed colors (values beyond will be saturated to min_col/max_col)	  						
            if plot_set_key['disk_color']=='RV':
                val_disk=RVstel
                cmap = plt.get_cmap('bwr')
            elif plot_set_key['disk_color']=='LD':
                val_disk=ld_grid_star[:,iband] 
                cmap = plt.get_cmap('YlOrRd_r')
            elif plot_set_key['disk_color']=='GD':
                val_disk=gd_grid_star[:,iband]
                cmap = plt.get_cmap('GnBu_r')
            elif plot_set_key['disk_color']=='F':
                val_disk=Fsurf_grid_star[:,iband]  
                
                if plot_set_key['plot_spots'] and (not plot_set_key['spot_overlap']) and (plot_set_key['t_BJD'] is not None):val_disk = Flux_for_nonredundant_spot_plotting

                
                # cmap = plt.get_cmap('GnBu_r')
                # cmap = plt.get_cmap('jet')
                cmap = plt.get_cmap('rainbow')
                sc_fact10 = -np.floor(np.log10(np.abs(np.median(val_disk))))
                val_disk*=10**sc_fact10
            min_col=0
            max_col=1
            if plot_set_key['val_range'] is None:
                min_val=np.nanmin(val_disk)
                max_val=np.nanmax(val_disk)
            else:
                min_val=plot_set_key['val_range'][0]
                max_val=plot_set_key['val_range'][1]
            if min_val==max_val:stop('No variations in colored values')
            color_tab=cmap( (min_col+ (val_disk-min_val)*(max_col-min_col)/ (max_val-min_val)) )
            
            #Settings cells outside the stellar disk to white											
            ax1.set_facecolor('white')  
     
            #Intensity shade
            if (plot_set_key['disk_color']=='RV') and (plot_set_key['shade_overlay']): 
                cmap_f = plt.get_cmap('Greys_r')
                min_f=np.nanmin(Fsurf_grid_star[:,iband])
                max_f=np.nanmax(Fsurf_grid_star[:,iband])
                color_f=cmap_f( (min_col+ (Fsurf_grid_star[:,iband]-min_f)*(max_col-min_col)/ (max_f-min_f)) )

                if plot_set_key['plot_spots'] and (not plot_set_key['spot_overlap']) and (plot_set_key['t_BJD'] is not None): 
                    min_f=np.nanmin(Flux_for_nonredundant_spot_plotting)
                    max_f=np.nanmax(Flux_for_nonredundant_spot_plotting)
                    color_f=cmap_f( (min_col+ (Flux_for_nonredundant_spot_plotting-min_f)*(max_col-min_col)/ (max_f-min_f)) )
        

            #Disk colored with RV or specific intensity
            cond_in_plot = (coord_grid['x_st_sky']+0.5*d_stcell>=plot_set_key['x_range'][0]) & (coord_grid['x_st_sky']-0.5*d_stcell<=plot_set_key['x_range'][1])\
                         & (coord_grid['y_st_sky']+0.5*d_stcell>=plot_set_key['y_range'][0]) & (coord_grid['y_st_sky']-0.5*d_stcell<=plot_set_key['y_range'][1])
            for i_in,x_pos,y_pos in zip(np.arange(len(val_disk))[cond_in_plot],coord_grid['x_st_sky'][cond_in_plot],coord_grid['y_st_sky'][cond_in_plot]):  
                edgecolor_stcell='none' if plot_set_key['st_grid_overlay'] else color_tab[i_in]
                rect = plt.Rectangle(( x_pos-0.5*d_stcell,y_pos-0.5*d_stcell), d_stcell,d_stcell, facecolor=color_tab[i_in],edgecolor=edgecolor_stcell,lw=0.1,zorder=-2)
                ax1.add_artist(rect)
                if (plot_set_key['disk_color']=='RV') and (plot_set_key['shade_overlay']): 
                    rect = plt.Rectangle(( x_pos-0.5*d_stcell,y_pos-0.5*d_stcell), d_stcell,d_stcell, facecolor=color_f[i_in],edgecolor='none',lw=0.1,zorder=-1,alpha=0.2)
                    ax1.add_artist(rect)
    
            #Equi-radial velocities
            if (plot_set_key['n_equi'] is not None) and (plot_set_key['conf_system']=='sky_ste'): 
                
                #Range of RV from -veq sin(istar) to veq sin(istar) are discretized with 'n_equi' lines
                dequiRV=2.*star_params['vsini']/plot_set_key['n_equi']
                equiRV_range=-star_params['vsini']+0.5*dequiRV+dequiRV*np.arange(plot_set_key['n_equi'])
                        
                #Unique vertical cell coordinates
                y_st_sky_grid_unique=np.unique(coord_grid['y_st_sky'])
                nrow_cells=len(y_st_sky_grid_unique)        
           
                #Initialize X and Y coordinates of each equiRV curve
                #    - the curves have one position per unique Y grid position
                x_equiRV=np.zeros([plot_set_key['n_equi'],nrow_cells])*np.nan
                y_equiRV=np.zeros([plot_set_key['n_equi'],nrow_cells])*np.nan
    
                #Loop on vertical cell rows
                for irow_cell,y_st_sky_grid_loc in enumerate(y_st_sky_grid_unique):
                    
                    #Retrieve X coordinates and RV of cells at current Y position
                    cond_row_cells=(coord_grid['y_st_sky']==y_st_sky_grid_loc)
                    xrow_cells=coord_grid['x_st_sky'][cond_row_cells]
                    RVrow_cells=RVstel[cond_row_cells]   
    
                    #Find X cell with RV closest from each equiRV
                    for i_equi,equiRVloc in enumerate(equiRV_range):
                        if np.abs(equiRVloc)<=np.max(np.abs(RVrow_cells)):
                            x_equiRV[i_equi,irow_cell]=xrow_cells[closest(RVrow_cells,equiRVloc)]
                            y_equiRV[i_equi,irow_cell]=y_st_sky_grid_loc
    
                #Plot equiRV
                for i_equi in range(plot_set_key['n_equi']): 
                    cond_kept = ~np.isnan(x_equiRV[i_equi])
                    if True in cond_kept:
                        isort=y_equiRV[i_equi,cond_kept].argsort()
                        plt.plot(x_equiRV[i_equi,cond_kept][isort],y_equiRV[i_equi,cond_kept][isort],color='black',lw=1.5,zorder=-1,linestyle=':')
                    
            #Overlaying stellar grid cell boundaries
            if plot_set_key['st_grid_overlay']:       
                col_grid = 'black'
                lw_st_grid=0.5
                ls_grid = ':'
                for i in range(int(plot_set_key['n_stcell']+1)):
                    plt.plot(np.array([-1.,1.]), np.array([2.*i/plot_set_key['n_stcell']-1.,2.*i/plot_set_key['n_stcell']-1.]), color=col_grid,lw=lw_st_grid,zorder=-1,ls=ls_grid)
                    plt.plot(np.array([2.*i/plot_set_key['n_stcell']-1.,2.*i/plot_set_key['n_stcell']-1.]), np.array([-1.,1.]), color=col_grid,lw=lw_st_grid,zorder=-1,ls=ls_grid)        
            
            #-----------------------	
            #Colorbar
            if plot_set_key['plot_colbar']:
                x2=0.87  #0.90							
                cbar_pos=fig.add_axes([plot_set_key['margins'][2],plot_set_key['margins'][1],x2-plot_set_key['margins'][2],plot_set_key['margins'][3]-plot_set_key['margins'][1]])   
                if plot_set_key['disk_color']=='RV':
                    colbar_title='Radial velocity (km s$^{-1}$)'		
                    format_cbar="%.1f"										
                elif plot_set_key['disk_color']=='LD':
                    colbar_title='Limb-Darkening'		
                    format_cbar="%.1f"	
                elif plot_set_key['disk_color']=='GD':
                    colbar_title='Gravity-Darkening'		
                    format_cbar="%.2f"	
                elif plot_set_key['disk_color']=='F':
                    colbar_title=scaled_title(sc_fact10,'Flux')	
                    format_cbar="%.2f"	         								
                cb = mpl.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin=min_val, vmax=max_val))										
                cb.set_array(val_disk)		
                cbar=fig.colorbar(cb,cax=cbar_pos,format=format_cbar)	
                cbar.ax.tick_params(labelsize=plot_set_key['font_size']) 
                cbar.set_label(colbar_title, rotation=270,labelpad=25+2,size=plot_set_key['font_size'])	
        
            #-----------------------
            #Main axis 
            xmajor_int,xminor_int,xmajor_form = autom_tick_prop(plot_set_key['x_range'][1]-plot_set_key['x_range'][0])
            ymajor_int,yminor_int,ymajor_form = autom_tick_prop(plot_set_key['y_range'][1]-plot_set_key['y_range'][0])
            custom_axis(plt,ax=ax1,x_range=plot_set_key['x_range'],y_range=plot_set_key['y_range'],
                        xmajor_int=xmajor_int,xminor_int=xminor_int,ymajor_int=ymajor_int,yminor_int=yminor_int,
                        # xmajor_int=0.5,xminor_int=0.1,ymajor_int=0.5,yminor_int=0.1,
        #                xmajor_length=8,xminor_length=4,ymajor_length=8,yminor_length=4,
            		    xmajor_form='%.1f',ymajor_form='%.1f',
                        top_xticks='on',
                        x_title='Distance (R$_{*}$)',y_title='Distance (R$_{*}$)',
                        font_size=plot_set_key['font_size'],xfont_size=plot_set_key['font_size'],yfont_size=plot_set_key['font_size'])
            
            #-----------------------	
            filename = path_loc+'System'+str(idx_pl)+'_'+str(plot_t)+'.'+plot_dic['system_view']	
            if images_to_make_GIF is not None:filenames += [filename]		
            plt.savefig(filename) 
            plt.close()
            
            #Store image for GIF generation
            if images_to_make_GIF is not None:images_to_make_GIF.append(imageio.v2.imread(filename))

        ### End of loop on plotted timesteps    

        #Produce and store the GIF.
        if images_to_make_GIF is not None:
            imageio.mimsave(path_loc+'System.gif', images_to_make_GIF,duration=(1000 * 1/plot_set_key['fps']))
            for film in filenames:os_system.remove(film)














    ##################################################################################################
    #%% Atmospheric profiles
    ##################################################################################################        

    ##################################################################################################
    #%%% 2D maps
    ################################################################################################## 
  
    ##################################################################################################
    #%%%% Original profiles
    ##################################################################################################
    if ('map_Atm_prof' in plot_settings):
        key_plot = 'map_Atm_prof'  

        print('-----------------------------------')
        print('+ Plotting 2D map of atmospheric profiles') 

        #Plot
        sub_2D_map(key_plot,plot_dic[key_plot],plot_settings[key_plot])
                   
        

        

        
    ##################################################################################################
    #%%%% Binned profiles
    ##################################################################################################
    if ('map_Atmbin' in plot_settings):
        key_plot = 'map_Atmbin'   

        print('-----------------------------------')
        print('+ Plotting 2D map of binned atmospheric profiles') 

        #Plot
        sub_2D_map(key_plot,plot_dic[key_plot],plot_settings[key_plot])        
        









    ##################################################################################################
    #%%%% 1D converted spectra
    ##################################################################################################
    if ('map_Atm_1D' in plot_settings):
        key_plot = 'map_Atm_1D' 

        print('-----------------------------------')
        print('+ Plotting 2D map of 1D atmospheric profiles') 

        #Plot
        sub_2D_map(key_plot,plot_dic[key_plot],plot_settings[key_plot])
                   







    ##################################################################################################
    #%%% Individual profiles
    ##################################################################################################        
    

    ##################################################################################################
    #%%%% Original profiles (grouped)
    ################################################################################################## 
    if ('all_atm_data' in plot_settings):
        key_plot = 'all_atm_data'

        print('-----------------------------------')
        print('+ Plotting all atmospheric profiles in each visit')
        
        #Plot        
        sub_plot_all_prof(plot_settings,'atm',plot_settings[key_plot])          
        




    ##################################################################################################
    #%%%% Original profiles
    ##################################################################################################
    for key_plot in ['Atm_prof','Atm_prof_res']:
        if (key_plot in plot_settings):
    
            ##############################################################################
            #%%%%% Flux profiles
            if (key_plot=='Atm_prof'):
                print('-----------------------------------')
                print('+ Individual atmospheric profiles')
    
            ##############################################################################
            #%%%%% Residuals profiles
            if (key_plot=='Atm_prof_res'):
                print('-----------------------------------')
                print('+ Individual residuals from atmospheric profiles')

            #%%%%% Plot        
            sub_plot_prof(plot_settings[key_plot],key_plot,plot_dic[key_plot])     
    
      
            
            
    ##################################################################################################
    #%%%% Binned profiles
    ##################################################################################################
    for key_plot in ['Atmbin','Atmbin_res']:
        if (key_plot in plot_settings):

            ##############################################################################
            #%%%%% Profile and its fit
            if (key_plot=='Atmbin'):
                print('-----------------------------------')
                print('+ Individual binned atmospheric profiles')
                
            ##############################################################################
            #%%%%% Residuals between the profile and its fit
            if (key_plot=='Atmbin_res'):
                print('-----------------------------------')
                print('+ Individual residuals from binned atmospheric profiles')

            ##############################################################################
            #%%%%% Plot  
            sub_plot_prof(plot_settings[key_plot],key_plot,plot_dic[key_plot])
          
                        




            
            
    ################################################################################################################   
    #%%%% 1D converted spectra
    ################################################################################################################   
    for key_plot in ['sp_Atm_1D','sp_Atm_1D_res']:
        if (key_plot in plot_settings):

            ##############################################################################
            #%%%%% Profile and its fit
            if (key_plot=='sp_Atm_1D'):
                print('-----------------------------------')
                print('+ Individual 1D intrinsic profiles')
                
            ##############################################################################
            #%%%%% Residuals between the profile and its fit
            if (key_plot=='sp_Atm_1D_res'):
                print('-----------------------------------')
                print('+ Individual residuals from 1D intrinsic profiles')

            ##############################################################################
            #%%%%% Plot  
            sub_plot_prof(plot_settings[key_plot],key_plot,plot_dic[key_plot])           
                        




    ##################################################################################################
    #%%% Chi2 over atmospheric property series
    ##################################################################################################
    if ('chi2_fit_AtmProp' in plot_settings):
        key_plot = 'chi2_fit_AtmProp'

        print('-----------------------------------')
        print('+ Chi2 over atmospheric property series')
        sub_plot_chi2_prop(plot_settings[key_plot],plot_dic['chi2_fit_AtmProp'])  









    return None



























