#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
from antaress.ANTARESS_general.utils import stop

def gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic):
    r"""**Default plot settings.**
    
    Set default values for the following fields. 
    Manual values can be given to all fields through `ANTARESS_plot_settings.py`, depending on their relevance for the chosen plot function. 
    
     - `fig_size = (width,height)` : figure size.
     - `margins = [x_left,y_low,x_width,y_width]` : margins of plot in the figure. 
     - `wspace = width` : horizontal separation between subplots.      
       `hspace = height` : vertical separation between subplots. 
     - `title = bool` : show plot title.
     - `font_size = float` : size for figure font.     
     - `font_size_txt = float` : size for text font within plot.
     - `lw_plot = float` : linewidth.
     - `ls_plot = str` : linestyle.
     - `col_contacts = str` : color for transit contacts.
     - `axis_thick = float` : thickness for plot axis.
     - `marker` : general marker type.

    Args:
        plot_settings (dic) : dictionary for all generic plot settings
        key_plot (str) : identifier of current plot 
    
    Returns:
        plot_settings (dic) : input, initialized with default settings according to the plot identifier
    
    """
    plot_settings[key_plot]={}
    plot_options = plot_settings[key_plot]
    
    #Figure size
    plot_options['fig_size'] = (10,6)
    
    #Margins
    plot_options['margins']=[0.15,0.15,0.95,0.7] 
    
    #Separation spaces in subplots 
    plot_options['wspace'] = 0.06
    plot_options['hspace'] = 0.06
    
    #Plot title
    plot_options['title']= False        
    
    #Font size
    plot_options['font_size']=14
    plot_options['font_size_txt'] = deepcopy(plot_options['font_size'])

    #Linewidth
    plot_options['lw_plot']=0.5
    
    #Linestyle
    plot_options['ls_plot']='-'
    
    #Color for transit contacts
    plot_options['col_contacts']='black'

    #Axis thickness
    plot_options['axis_thick']=1  
    
    #Marker
    plot_options['marker'] = 'o'
    
    #Symbol size
    plot_options['markersize']=2.5

    #Hide axis
    plot_options['hide_axis'] = False
    
    #Rasterize datapoints
    plot_options['rasterized'] = True

    #Transparent background
    plot_options['transparent'] = False
    
    #Curve style
    plot_options['drawstyle']=None
 
    #Plot abscissa range or errors
    plot_options['plot_xerr']=False  

    #Transparency of symbols (0 = void)
    plot_options['alpha_symb']=1. 
  
    #Transparency of error bars (0 = void)
    plot_options['alpha_err']=0.5

    #Spectral resampling
    plot_options['resample'] = None
    
    #List of stellar lines to plot in spectral mode
    #    - default to stellar CCF mask
    plot_options['st_lines_wav'] = []

    #List of planet lines to plot in spectral mode
    #    - default to planet CCF mask
    plot_options['pl_lines_wav'] = []

    #Make GIF from plot series.
    plot_options['GIF_generation'] = False
    
    #FPS for gif
    plot_options['fps'] = 5
    
    #Print information
    plot_options['verbose']=True 

    #Plot legend figure
    plot_options['legend']=False 
    plot_options['legend_to_plot']={} 

    #--------------------------------------

    #Instruments to plot
    plot_options['inst_to_plot']=list(plot_dic['visits_to_plot'].keys())
    
    #Visits to plot
    #    - add '_bin' to the name of a visit to plot properties derived from intrinsic profiles binned within a visit
    #      use 'binned' as visit name to plot properties derived from intrinsic profiles binned over several visits
    plot_options['visits_to_plot']=deepcopy(plot_dic['visits_to_plot'])
    
    #Indexes of exposures to be plotted 
    plot_options['iexp2plot']={}

    #Indexes of exposures to be removed from the plot
    plot_options['idx_noplot'] = {}

    #Do not plot original data
    plot_options['no_orig']= False

    #Plot data
    plot_options['plot_data'] = True

    #Plot data errors
    plot_options['plot_err']=True 

    #Plot models 
    plot_options['plot_model'] = True

    #Indexes of orders to be plotted         
    #    - leave empty to plot all orders
    plot_options['iord2plot']=[]    
    
    #Plot order indexes
    plot_settings[key_plot]['plot_idx_ord'] = False
    
    #Colors
    plot_options['color_dic']={}  
    plot_options['color_dic_sec']={}
    plot_options['color_dic_bin']={}  
    plot_options['color_dic_bin_sec']={}
    
    #Scaling factor 
    #    - in power of ten, ie flux are multiplied by 10**sc_fact10)
    #    - set to None for automatic determination
    plot_options['sc_fact10'] = 0.
    
    #Spectral variable
    #    - wavelength in A
    plot_options['sp_var'] = 'wav'
    
    #Bornes du plot  
    plot_options['x_range']=None
    plot_options['y_range']=None 

    #Overplot excluded planetary ranges
    plot_options['plot_plexc']=False

    #Overplot intrinsic stellar line ranges
    plot_options['plot_stexc']=False
    
    #Reference planet for each visit
    #    - set to first transit planet if it exists, or to first planet overall otherwise
    plot_options['pl_ref']={}
    for inst in plot_options['visits_to_plot']:
        plot_options['pl_ref'][inst]={'binned':gen_dic['studied_pl_list'][0]}
        for vis in plot_options['visits_to_plot'][inst]:
            if len(data_dic[inst][vis]['studied_pl'])>0:plot_options['pl_ref'][inst][vis]=data_dic[inst][vis]['studied_pl'][0]
            else:plot_options['pl_ref'][inst][vis]=gen_dic['studied_pl_list'][0]

    #Shade range not used for fitting
    plot_options['shade_unfit']=False
    
    #Shade selected ranges
    plot_options['shade_ranges']={}
    
    #Plot continuum pixels common to all exposures
    plot_options['plot_cont']=False

    #Shade continuum range requested as input
    plot_options['shade_cont']=False     

    #Plot continuum level
    plot_options['plot_cont_lev']=False
    
    #Plot continuum pixels specific to each exposure
    plot_options['plot_cont_exp']=False   
    
    #Plot residuals from continuum alone
    plot_options['cont_only'] = False

    #Plot fitted pixels
    plot_options['plot_fitpix']=False

    #Plot fitted line profile
    plot_options['plot_line_model']=False

    #Plot model line profile
    plot_options['plot_line_model_HR']=False

    #Plot individual model components for line profile
    plot_options['plot_line_model_compo']=False
    
    #Print fit properties on plot
    plot_options['plot_prop']=True  

    #Plot fitted line centroid
    plot_options['plot_line_fit_rv']=False

    #Plot bissector
    plot_options['plot_biss']=False
            
    #Print dispersions
    plot_options['print_disp'] = []  
    
    #Plot HITRAN telluric lines for requested molecules
    plot_options['plot_tell_HITRANS']=[]
    plot_options['telldepth_min'] = 0.
    
    #Plot master used as reference for flux balance correction
    plot_options['plot_mast'] = False    

    #Plot stellar continuum
    plot_options['st_cont']=None    
    
    #Plot spectra at two chosen steps of the correction process
    #    - set to None, or chose amongst:
    # + 'raw' : before any correction
    # + 'all' : after all requested corrections
    # + 'tell' : after telluric correction 
    # + 'fbal' : after flux balance correction  
    # + 'cosm' : after cosmics correction  
    # + 'permpeak' : after persistent peak correction 
    # + 'wig' : after wiggle correction 
    plot_options['plot_pre']='all'
    plot_options['plot_post']=None        
 
    #Plot all exposures on the same plot
    plot_options['multi_exp']= False

    #Plot all order on the same plot
    plot_options['multi_ord']= False    

    #Normalize spectra to integrated flux unity
    #    - to allow for comparison
    plot_options['norm_prof'] = False  

    #Absorption signal type
    plot_options['pl_atm_sign']='Absorption'  

    #Histograms bin number
    plot_options['bins_par'] = 40
    
    #Exposures used in master out
    plot_options['iexp_mast_list']={}

    #--------------------------------------   

    #Aligned profiles
    plot_options['aligned']=False

    #Measured values
    plot_options['print_mes']=False
    
    #Plot reference level
    plot_options['plot_reflev']=False

    #Plot reference velocity
    plot_options['plot_refvel']=True  

    #Fit type
    plot_options['fit_type']='indiv'

    #Plot fitted exposures only
    plot_options['fitted_exp'] = False

    #Planet-occulted line model from best-fit ('fit') or reconstruction ('rec')
    plot_options['line_model']='fit'
    
    #Ranges for continuum definition in plots
    plot_options['cont_range']={}

    #--------------------------------------  
    if (key_plot in ['glob_mast','all_DI_data','all_intr_data','all_atm_data']):
        
        #Plot spectra used for measured master calculation
        plot_options['plot_input']=True
        
    #--------------------------------------              
    if (key_plot in ['Fbal_corr','Fbal_corr_vis','input_LC','plocc_ranges','prop_DI_mcmc_PDFs','prop_Intr_mcmc_PDFs']):

        #Plot exposure indexes
        plot_options['plot_expid'] = True
        
    #--------------------------------------   
    #Binned profiles settings     
    if ('map_' in key_plot) or ('bin' in key_plot) or ('prop_' in key_plot) or (key_plot in ['occulted_regions']):

        #Bin dimension
        #    - for 2D maps:
        # + 'phase' : see details and routine
        #             coordinates are only available if binning was performed over 'phase'
        #    - for binned profiles:
        # + 'phase'
        # + 'xp_abs'
        # + 'r_proj' 
        #    - if not phase, exposures are plotted successively without respecting their actual positions, because of overlaps 
        plot_options['dim_plot']='phase' 

    #--------------------------------------   
    if ('prop_' in key_plot):

        #Abscissa
        plot_options['prop_DI_absc']='phase'                
        plot_options['prop_Intr_absc']='phase'
        
        #Invert horizontal axis
        #    - useful for retrograde orbit to follow the orbit from left to right (and decreasing phase)
        plot_options['retro_orbit']=False  

        #Plot values for detected CCFs only
        plot_options['plot_det']=False 

        #Print min/max values (to adjust plot ranges)
        plot_options['plot_bounds']=False

        #Print and plot mean value and dispersion 
        plot_options['plot_disp']=True    

        #Plot master property
        plot_options['plot_Mout']=False
        plot_options['plot_Mloc']=False

        #Plot HDI subintervals, if available
        plot_options['plot_HDI']= False    

        #Use different symbols for transits (disks vs squares)
        plot_options['use_diff_symb']= False    
        
        #Empty symbols
        plot_options['empty_in'] = True
        plot_options['empty_all']= False
        plot_options['empty_det']= False   
 
        #Bin properties 
        plot_options['bin_val'] = {}

        #Orders for SNR
        plot_options['idx_SNR']={'HARPS':[49],'HARPN':[46],'CORALIE':[46],'ESPRESSO':[102,103],'ESPRESSO_MR':[39],'CARMENES_VIS':[40],'NIRPS_HE':[57]}
        plot_options['idx_num_SNR']={'HARPN':[46]}
        plot_options['idx_den_SNR']={'HARPN':[46]}              
 
        #Save a text file of residual RVs vs phase
        plot_options['save_RVres'] = False

    #--------------------------------------
    if 'atm' in key_plot:
      
        #Scaling factor
        plot_options['sc_fact10'] = 6.

    #--------------------------------------
    if (key_plot in ['occulted_regions','system_view']):
        
        #Choice of spectral band for intensity
        #    - from the main planet transit properties
        plot_options['iband']=0 

    #--------------------------------------
    if 'map_' in key_plot:

        #Atmospheric signal type 
        if 'map_Atm' in key_plot:plot_options['pl_atm_sign']='Absorption'            

        #Margins   
        plot_options['margins']=[0.15,0.3,0.84,0.95]  

        #Font size
        plot_options['font_size']=18  

        #Reverse image
        #    - if set to True, map is orientated so that velocity is the vertical axis, and phase the horizontal axis
        plot_options['reverse_2D']=False

        #Overplot surface RV model along the full transit chord
        plot_options['theoRV_HR']=False

        #Overplot surface RV model along the full transit chord for an aligned orbit
        plot_options['theoRV_HR_align'] = False 

        #Overplot surface RV model at the phases of the observations 
        plot_options['plot_theoRV']=False

        #Overplot RV(pl/star) model 
        plot_options['theoRVpl_HR']=False
        plot_options['theoRVpl_HR_align']=False

        #Plot zero line markers
        plot_options['plot_zermark']= True 
        
        #Color map
        if 'map_DI' in key_plot:plot_options['cmap']="jet" 
        if 'map_Diff' in key_plot:plot_options['cmap']="jet"             
        elif 'map_Intr' in key_plot:plot_options['cmap']="afmhot_r" 
        elif 'map_Intr_prof_res' in key_plot:plot_options['cmap']="afmhot_r" 
        elif 'map_Atm' in key_plot:plot_options['cmap']="winter"             
        
        #Map color range
        plot_options['v_range_all']={}
        
        #Ranges
        plot_options['x_range_all']={}
        plot_options['y_range_all']={}
        
        #Overplot measured RVs
        plot_options['plot_measRV'] = 'none'       

        #Plot global and in-transit indexes
        plot_options['plot_idx']=False              

    return plot_settings





  
def ANTARESS_plot_settings(plot_settings,plot_dic,gen_dic,data_dic,glob_fit_dic,theo_dic):
    r"""**ANTARESS default plot settings.**
    
    Initializes ANTARESS plot settings with default values.  
    
    Args:
        TBD
    
    Returns:
        None
    
    """   
    
    ################################################################################################################    
    #%% Weighing master
    ################################################################################################################  
    if (plot_dic['DImast']!='') and gen_dic['specINtype']:
        key_plot = 'DImast'
        
        #%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)


    ################################################################################################################  
    #%% Global spectral corrections
    ################################################################################################################          
    if gen_dic['specINtype']:

        ################################################################################################################    
        #%%% Instrumental calibration estimates
        ################################################################################################################ 
        if (plot_dic['gcal_all']!='') or (plot_dic['gcal_ord']!='') or (plot_dic['noises_ord']!=''):
            key_plot = 'gcal'
            
            #%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)
    
            #%%%% Plot measured binned profiles in each order
            plot_settings[key_plot]['plot_meas_exp'] = True 

            #%%%% Plot blaze-derived profiles in each order
            plot_settings[key_plot]['plot_gcal_blaze'] = True 
    
            #%%%% Plot best-fit profile model in each order
            plot_settings[key_plot]['plot_best_exp'] = True
    
            #%%%% Normalize calibration profiles over all exposures
            plot_settings[key_plot]['norm_exp'] = False
            
            #%%%% Plot mean calibration profile if calculated
            plot_settings[key_plot]['mean_gcal']  = True
            
            #%%%% Scaling for calibration and variance
            plot_settings[key_plot]['sc_fact10'] = -3
            plot_settings[key_plot]['sc_fact10_var'] = -9
    
            #%%%% Range per order
            #    - unfitted points are not shown in automatic ranges
            plot_settings[key_plot]['x_range_ord'] = None
            plot_settings[key_plot]['y_range_ord'] = None          
            
           
    
        ##################################################################################################
        #%%% Tellurics
        ##################################################################################################        
            
        ################################################################################################################    
        #%%%% Telluric CCF
        ################################################################################################################  
        if (plot_dic['tell_CCF']!=''):
            key_plot = 'tell_CCF'
    
            #%%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)
    
            #%%%% Plot axis boundaries
            plot_settings[key_plot]['margins']=[0.15,0.15,0.85,0.88]  
    
            #%%%%% CCF to plot
            #    - plot CCF computed over strong linelist used in the fit, and full linelist
            plot_settings[key_plot]['tell_CCF_lines'] = ['master','full']
    
            #%%%%% Molecules to plot
            plot_settings[key_plot]['tell_species'] = gen_dic['tell_species']
            
            #%%%%% Plot dispersion  
            plot_settings[key_plot]['print_disp']=None 
    
    
    
    
        ################################################################################################################    
        #%%%% Telluric properties
        ################################################################################################################   
        if (plot_dic['tell_prop']!=''):
            key_plot = 'tell_prop'
            
            #%%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)
    
            #%%%%% Molecules to plot
            plot_settings[key_plot]['tell_species'] = gen_dic['tell_species']
      
    
    
    
    
    
        ##################################################################################################
        #%%% Flux balance corrections
        ##################################################################################################        
              
        ################################################################################################################    
        #%%%% Global scaling masters        
        #    - plotting global measured masters used in the flux balance correction and the flux scaling of the spectra
        #    - all raw spectra are plotted by default, before they are corrected
        #    - the measured master built from these spectra is also plotted by default
        ################################################################################################################
        if (plot_dic['glob_mast']!=''):
            key_plot = 'glob_mast'
    
            #%%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)        
    
            #%%%%% Plot global master over all visits    
            plot_settings[key_plot]['glob_mast_all']=None 
    
            #%%%%% Plot master of each visit    
            plot_settings[key_plot]['glob_mast_vis']=True 
    
            
    
    
        ################################################################################################################    
        #%%%% Global flux balance (exposures)
        #    - relative to the mean level of each profile
        ################################################################################################################
        if (plot_dic['Fbal_corr']!=''):   
            key_plot = 'Fbal_corr'
    
            #%%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)         
    
            #%%%%% Plot order indexes
            plot_settings[key_plot]['plot_idx_ord'] = True
            
            #%%%%% Indexes of exposures and bins to be plotted 
            plot_settings[key_plot]['ibin_plot'] = {}
              
            #%%%%% Overplot all exposures or offset them
            plot_settings[key_plot]['gap_exp']=0.        
    
            #%%%%% Strip range used for correction
            plot_settings[key_plot]['strip_corr'] = False
    
            #%%%%% Model spectral resolution 
            #    - in dlnw = dw/w, if x_range is defined
            plot_settings[key_plot]['dlnw_plot'] = 0.002
        
    
        
    
    
        ################################################################################################################    
        #%%%% Global DRS flux balance (exposures)
        ################################################################################################################
        if (plot_dic['Fbal_corr_DRS']!=''):
            key_plot = 'Fbal_corr_DRS'
    
            #%%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 
            
      
        
      
        ################################################################################################################    
        #%%%% Global flux balance (visits)
        #    - relative to the mean level of each profile
        ################################################################################################################
        if (plot_dic['Fbal_corr_vis']!=''):   
            key_plot = 'Fbal_corr_vis'
    
            #%%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)         
    
            #%%%%% Plot order indexes
            plot_settings[key_plot]['plot_idx_ord'] = True
            
            #%%%%% Indexes of bins to be plotted 
            plot_settings[key_plot]['ibin_plot'] = {}
              
            #%%%%% Overplot all profiles or offset them
            plot_settings[key_plot]['gap_exp']=0.        
    
            #%%%%% Strip range used for correction
            plot_settings[key_plot]['strip_corr'] = False
    
            #%%%%% Model spectral resolution 
            #    - in dlnw = dw/w, if x_range is defined
            plot_settings[key_plot]['dlnw_plot'] = 0.002   
    
        
        
        
        
    
        ################################################################################################################ 
        #%%%% Intra-order flux balance
        ################################################################################################################ 
        if (plot_dic['Fbal_corr_ord']!=''):
            key_plot = 'Fbal_corr_ord' 
    
            #%%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)          
    
    
    
            
            
            
            
            
            
        ################################################################################################################ 
        #%%%% Temporal flux balance
        ################################################################################################################ 
        if (plot_dic['Ftemp_corr']!=''):
            key_plot = 'Ftemp_corr'
    
            #%%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)  
            
    
    
    
    
                     
    
    
    
        ################################################################################################################ 
        #%%% Cosmics correction
        ################################################################################################################ 
        if (plot_dic['cosm_corr']!=''):
            key_plot = 'cosm_corr'
    
            #%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)         
    
            #%%%% Cosmic marker
            plot_settings[key_plot]['markcosm'] = True  
            plot_settings[key_plot]['ncosm'] = True  
    
            #%%%% Plot orders with detected cosmics only  
            plot_settings[key_plot]['det_cosm'] = True
    
            #%%%% Plot number of cosmics per order
            plot_settings[key_plot]['cosm_vs_ord']=True         
    
    
    
    
    
    
    
    
    
    
    
        ################################################################################################################ 
        #%%% Persistent peaks master
        ################################################################################################################ 
        if (plot_dic['permpeak_corr']!=''):
            key_plot = 'permpeak_corr'
    
            #%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)          
            
            
            
            
            
            
            
            
            
    
    
    
    
        ################################################################################################################ 
        #%%% Fringing correction
        ################################################################################################################ 
        if (plot_dic['fring_corr']!=''):
            key_plot = 'fring_corr'
    
            #%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)
            
            
            
        
        



        
    ##################################################################################################
    #%% Disk-integrated profiles 
    ##################################################################################################        
        
    ##################################################################################################
    #%%% 2D maps
    ##################################################################################################     
    
    ##################################################################################################
    #%%%% Original profiles
    #    - allows visualizing the exposures used to build the master, and the ranges excluded because of planetary contamination
    ##################################################################################################
    if (plot_dic['map_DI_prof']!=''):  
        key_plot = 'map_DI_prof'

        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)

        #%%%%% Profiles to plot
        #    - 'sp_corr' : spectral profiles have been corrected
        #    - 'detrend' : profiles have been detrended
        #    - 'aligned' : profiles have been aligned, after being corrected if requested
        #    - 'scaled' : profiles have been scaled, after being corrected and aligned if requested
        plot_settings[key_plot]['step']='scaled'
        

        
        
        
        
    ##################################################################################################
    #%%%% Binned profiles
    ##################################################################################################
    if (plot_dic['map_DIbin']!=''):
        key_plot = 'map_DIbin'

        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)
     
        
     
        


    ##################################################################################################
    #%%%% 1D converted profiles
    ##################################################################################################
    if (plot_dic['map_DI_1D']!='') and (any('spec' in s for s in data_dic['DI']['type'].values())):
        key_plot = 'map_DI_1D'

        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)
        
        
        

        


  
    ##################################################################################################
    #%%% Individual profiles
    ##################################################################################################       

    ##################################################################################################
    #%%%% Original spectra (correction steps) 
    #    - before and after spectral corrections 
    #    - in their input rest frame (typically heliocentric)
    ##################################################################################################
    if gen_dic['specINtype'] and (plot_dic['flux_sp']!=''):
        key_plot = 'DI_prof_corr' #key must be different from 'DI_prof' to differentiate from original profile plot

        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)        

        #%%%%% Plot spectra at two chosen steps of the correction process
        plot_settings[key_plot]['plot_pre']='raw'
        plot_settings[key_plot]['plot_post']='all'

        #%%%%% Plot binned data and errors
        plot_settings[key_plot]['plot_bin'] = True
        plot_settings[key_plot]['plot_bin_err'] = True

        #%%%%% Data bin size (in A)
        plot_settings[key_plot]['bin_width'] = 10.

        #%%%%% Only plot exposures & orders with detected cosmics         
        plot_settings[key_plot]['det_cosm']= False
     
        #%%%%% Only plot exposures & orders with masked persistent pixels 
        plot_settings[key_plot]['det_permpeak']= False 

        #%%%%% Plot telluric spectrum
        #    - the one associated with the 'plot_post' spectra
        plot_settings[key_plot]['plot_tell'] = False

        #%%%%% Plot binned pixels used for the color balance fit
        plot_settings[key_plot]['plot_bins']= False

        #%%%%% Plot continuum used for persistent peak masking
        plot_settings[key_plot]['plot_contmax']=True 

        #%%%%% Plot order indexes
        plot_settings[key_plot]['plot_idx_ord'] = True






    ##################################################################################################
    #%%%% Original transmission spectra (correction steps)
    #    - in the star rest frame, offset or not by the systemic velocity
    ##################################################################################################
    if gen_dic['specINtype'] and (plot_dic['trans_sp']!=''):
        key_plot = 'trans_sp'

        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 
        
        #%%%%% Plot spectra at two chosen steps of the correction process
        plot_settings[key_plot]['plot_pre']='raw'
        plot_settings[key_plot]['plot_post']='all'

        #%%%%% Plot binned data and errors
        plot_settings[key_plot]['plot_bin'] = True
        plot_settings[key_plot]['plot_bin_err'] = True

        #%%%%% Data bin size (in A)
        plot_settings[key_plot]['bin_width'] = 10.

        #%%%%% Force order level to unity
        plot_settings[key_plot]['force_unity'] = False
        
        #%%%%% Offset between uncorrected / corrected data
        plot_settings[key_plot]['gap_exp'] = 0.03
        
        #%%%%% Path to correction
        #          - leave empty to use last result from 'wig_vis_fit'
        plot_settings[key_plot]['wig_path_corr'] = {}  

        #%%%%% Plot mean value over exposures
        plot_settings[key_plot]['plot_disp'] = True   

        #%%%%% Vertical range for dispersion plot
        plot_settings[key_plot]['y_range_disp'] = {}

        #%%%%% Plot order indexes
        plot_settings[key_plot]['plot_idx_ord'] = True



    




    ##################################################################################################
    #%%%% Original profiles (processing steps) 
    #    - in their original rest frame (typically heliocentric), with their model fit
    ##################################################################################################
    for key_plot in ['DI_prof','DI_prof_res']:
        if plot_dic[key_plot]!='':
    
            #%%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)                 

            #%%%%% Choose profiles to plot
            #    - 'raw' : input profiles
            plot_settings[key_plot]['step']='raw'  

            ##############################################################################
            #%%%%% Profile and its fit
            #    - ccfs are continuum-normalized for the purpose of the plot
            if (key_plot=='DI_prof'):  
                pass
            
            ##############################################################################
            #%%%%% Residuals between the profile and its fit
            if (key_plot=='DI_prof_res'):
                if (not gen_dic['fit_DI']):stop('Activate "gen_dic["fit_DI"]" to plot "plot_dic["DI_prof_res"]"')     
                



    ##################################################################################################
    #%%%% Binned profiles
    ##################################################################################################
    for key_plot in ['DIbin','DIbin_res']:
        if plot_dic[key_plot]!='':
    
            #%%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)  

            ##############################################################################
            #%%%%% Profile and its fit
            if (key_plot=='DIbin'):
                pass

            ##############################################################################
            #%%%%% Residuals between the profile and its fit
            if (key_plot=='DIbin_res'):
                if (not gen_dic['fit_DI_gen']):stop('Activate "gen_dic["fit_DI_gen"]" to plot "plot_dic["DIbin_res"]"') 


    
        
        

        

        
        
        
        
        
    ##################################################################################################
    #%%%% 1D converted spectra
    ##################################################################################################
    if (plot_dic['sp_DI_1D']!='') and (any('spec' in s for s in data_dic['DI']['type'].values())):
        key_plot = 'sp_DI_1D' 

        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)

 

    
  
    
  
    
        
        

  
    
  

    ##################################################################################################
    #%%%% Aligned profiles (all)
    #    - in star rest frame
    #    - all profiles from a given visit together
    #    - spectra may have been corrected for color balance or not
    #     CCF have not yet been scaled in flux, which is done within the routine to allow better comparing them (his requires the transit scaling routine to have been run so that continuum pixels are defined)
    ##################################################################################################   
    if (plot_dic['all_DI_data']!=''):  
        key_plot = 'all_DI_data'

        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)        

        #%%%%% Data type
        plot_settings[key_plot]['data_type']='CCF' 
        
            
  
    
  
    
  
        
  
    
        
        
    ##################################################################################################
    #%% Stellar CCF mask
    ##################################################################################################        
        
    ##################################################################################################
    #%%% Spectrum
    #    - plotting spectrum used for CCF mask generation and associated properties
    ##################################################################################################
    for key_plot in ['DImask_spectra','Intrmask_spectra']:
        if plot_dic[key_plot]!='':

            #%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)  

            #%%%% Plot master spectrum and mask at chosen step
            # + 'cont': continuum-normalization
            # + 'sel0': line selection with default telluric ranges and exclusion ranges (use to adjust regular spectrum and extrema search)
            # + 'sel1': line selection with depth and width criteria
            # + 'sel2': line selection with delta-position criterion
            # + 'sel3': line selection with telluric contamination
            # + 'sel4': line selection with morphological clipping (delta-maxima/line depth and asymetry parameter)
            # + 'sel5': line selection with morphological clipping (depth and width criteria)
            # + 'sel6': line selection with RV dispersion and telluric contamination
            plot_settings[key_plot]['step']='cont'      
            
            #%%%% Plot various spectra
            plot_settings[key_plot]['plot_raw'] = True        #Original spectrum (blue)
            plot_settings[key_plot]['plot_norm'] = True       #Normalized spectrum (grey)
            plot_settings[key_plot]['plot_norm_reg'] = True   #Smoothed regular normalized spectrum (black, used in mask generation)

            #%%%% Print number of line selected in step
            plot_settings[key_plot]['print_nl']=True   

            #%%%% Plot minimum telluric depth to be considered
            plot_settings[key_plot]['tell_depth_min'] = False

            ##############################################################################
            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_spectra'):
                pass

            ##############################################################################
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_spectra'):
                pass




    ##################################################################################################
    #%%% Line depth range selection
    ##################################################################################################
    for key_plot in ['DImask_ld','Intrmask_ld']:
        if plot_dic[key_plot]!='':

            #%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)  

            #%%%% Number of bins in histograms
            plot_settings[key_plot]['dist_info'] = ['hist','cum_w']
            plot_settings[key_plot]['x_bins_par'] = 40
            plot_settings[key_plot]['x_log_hist'] = False
            plot_settings[key_plot]['y_bins_par'] = 40    
            plot_settings[key_plot]['y_log_hist'] = False
            plot_settings[key_plot]['x_range_hist'] = None
            plot_settings[key_plot]['y_range_hist'] = None      
            
            ##############################################################################
            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_ld'):
                pass
    
            ##############################################################################
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_ld'):
                pass




    ##################################################################################################
    #%%% Line depth and width selection
    ##################################################################################################
    for key_plot in ['DImask_ld_lw','Intrmask_ld_lw']:
        if plot_dic[key_plot]!='':

            #%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)  

            #%%%% Number of bins in histograms
            plot_settings[key_plot]['dist_info'] = ['hist','cum_w']
            plot_settings[key_plot]['x_bins_par'] = 40
            plot_settings[key_plot]['x_log_hist'] = False
            plot_settings[key_plot]['y_bins_par'] = 40    
            plot_settings[key_plot]['y_log_hist'] = False
            plot_settings[key_plot]['x_range_hist'] = None
            plot_settings[key_plot]['y_range_hist'] = None
            
            ##############################################################################
            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_ld_lw'):
                pass
    
            ##############################################################################
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_ld_lw'):
                pass    

        

        




    ##################################################################################################
    #%%% Line position selection
    ##################################################################################################
    for key_plot in ['DImask_RVdev_fit','Intrmask_RVdev_fit']:
        if plot_dic[key_plot]!='':

            #%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)  

            #%%%% Number of bins in histograms
            plot_settings[key_plot]['dist_info'] =['hist','cum_w']
            plot_settings[key_plot]['x_bins_par'] = 40
            plot_settings[key_plot]['x_log_hist'] = False
            plot_settings[key_plot]['x_range_hist'] = None
            
            ##############################################################################
            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_RVdev_fit'):
                pass
    
            ##############################################################################
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_RVdev_fit'):
                pass    






    ##################################################################################################
    #%%% Telluric selection
    #    - plotting relative depth between telluric and stellar lines for CCF mask generation
    ##################################################################################################
    for key_plot in ['DImask_tellcont','Intrmask_tellcont']:
        if plot_dic[key_plot]!='':

            #%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 

            #%%%% Number of bins in histograms
            plot_settings[key_plot]['dist_info'] =['hist','cum_w']
            plot_settings[key_plot]['x_bins_par'] = 40
            plot_settings[key_plot]['x_log_hist'] = False
            plot_settings[key_plot]['x_range_hist'] = None

            ##############################################################################
            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_tellcont'):
                pass
    
            ##############################################################################
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_tellcont'):
                pass   
        

        


    ##################################################################################################
    #%%% VALD line depth correction
    ##################################################################################################
    for key_plot in ['DImask_vald_depthcorr','Intrmask_vald_depthcorr']:
        if plot_dic[key_plot]!='':

            #%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 

            #%%%% Figure size
            plot_settings[key_plot]['fig_size'] = (12,15)

            ##############################################################################
            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_vald_depthcorr'):
                pass
    
            ##############################################################################
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_vald_depthcorr'):
                pass   







    ##################################################################################################
    #%%% Morphological (asymmetry) selection
    ##################################################################################################
    for key_plot in ['DImask_morphasym','Intrmask_morphasym']:
        if plot_dic[key_plot]!='':

            #%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)        

            #%%%% Number of bins in histograms
            plot_settings[key_plot]['dist_info'] =['hist','cum_w']
            plot_settings[key_plot]['x_bins_par'] = 40
            plot_settings[key_plot]['x_log_hist'] = False
            plot_settings[key_plot]['y_bins_par'] = 40    
            plot_settings[key_plot]['y_log_hist'] = False
            plot_settings[key_plot]['x_range_hist'] = None
            plot_settings[key_plot]['y_range_hist'] = None

            ##############################################################################
            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_morphasym'):
                pass
    
            ##############################################################################
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_morphasym'):
                pass   
    



    ##################################################################################################
    #%%% Morphological (shape) selection
    ##################################################################################################
    for key_plot in ['DImask_morphshape','Intrmask_morphshape']:
        if plot_dic[key_plot]!='':

            #%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)       

            #%%%% Number of bins in histograms
            plot_settings[key_plot]['dist_info'] =['hist','cum_w']
            plot_settings[key_plot]['x_bins_par'] = 40
            plot_settings[key_plot]['x_log_hist'] = False
            plot_settings[key_plot]['y_bins_par'] = 40    
            plot_settings[key_plot]['y_log_hist'] = False
            plot_settings[key_plot]['x_range_hist'] = None
            plot_settings[key_plot]['y_range_hist'] = None

            ##############################################################################
            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_morphshape'):
                pass
    
            ##############################################################################
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_morphshape'):
                pass   

        

        

    ##################################################################################################
    #%%% RV dispersion selection
    ##################################################################################################
    for key_plot in ['DImask_RVdisp','Intrmask_RVdisp']:
        if plot_dic[key_plot]!='':

            #%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)        

            #%%%% Number of bins in histograms
            plot_settings[key_plot]['dist_info'] =['hist','cum_w']
            plot_settings[key_plot]['x_bins_par'] = 40
            plot_settings[key_plot]['x_log_hist'] = False
            plot_settings[key_plot]['y_bins_par'] = 40    
            plot_settings[key_plot]['y_log_hist'] = False
            plot_settings[key_plot]['x_range_hist'] = None
            plot_settings[key_plot]['y_range_hist'] = None

            ##############################################################################
            #%%%% Disk-integrated profiles
            if (key_plot=='DImask_RVdisp'):
                pass
    
            ##############################################################################
            #%%%% Intrinsic profile
            if (key_plot=='Intrmask_RVdisp'):
                pass   















    ##################################################################################################
    #%% Properties of raw data and disk-integrated CCFs
    #    - possibility to plot them as a function of many variable to search for correlations
    ##################################################################################################
    if (plot_dic['prop_DI']!=''):
        
        #%%% Ordina properties
        #    - choose values to plot in ordina (list of properties)
        #    - we set all properties to be plotted as default so that their associated plot properties can be defined here
        #    - properties:
        # + 'rv' : centroid of the raw CCFs in heliocentric rest frame (in km/s)
        # + 'FWHM': width of raw CCFs (in km/s)
        # + 'ctrst': contrast of raw CCFs
        # + 'rv_l2c': RV(lobe)-RV(core) of double gaussian components
        # + 'FWHM_l2c': FWHM(lobe)/FWHM(core) of double gaussian components
        # + 'amp_l2c': contrast(lobe)/contrast(core) of double gaussian components
        # + 'rv_res' residuals from Keplerian curve (m/s)  
        # + 'RVdrift' : RV drift of the spectrograph, derived from the Fabry-Perot (m/s)
        # + 'phase' : orbital phase
        # + 'mu' : mu
        # + 'lat' : stellar latitude 
        # + 'lon' : stellar longitude 
        # + projected position in the stellar frame
        #   x (along the equator) : 'x_st'
        #   y (along the spin axis) : 'y_st' 
        # + 'AM' : airmass : 
        # + 'seeing' : seeing : 
        # + 'snr' : SNR : 
        # in that case select the indices of orders over which average the SNR
        # + 'snr_R' : SNR ratio 
        # in that case select the indices of orders over which average the SNR for both numerator and denominator
        # + 'colcorrmin', 'colcorrmax', 'colcorrR' : min/max color correction coefficients and ratio max/min 
        #   'colcorr450', 'colcorr550', 'colcorr650' : correction coefficients at the corresponding wavelengths (nm)
        # + 'glob_flux_sc': ratio of total flux in each exposure profile, to their mean value, used to scale all profiles to the same global flux level
        # + 'satur_check': check of saturation on detector
        # + 'PSFx', 'PSFy' : sizes of PSF on detector (?)
        #   'PSFr' : average (quadratic) size
        #   'PSFang' : angle y/x in degrees  
        # + coefficients of wiggles laws: wig_p_0, wig_p1, wig_wref, wig_ai[i=0,4]
        # + 'alt': telescope altitude angle (deg)
        # + 'ha','na','ca','s','rhk': activity indexes 
        # + 'ADC1 POS','ADC1 RA','ADC1 DEC','ADC2 POS','ADC2 RA','ADC2 DEC': ESPRESSO ADC intel
        # + 'TILT1 VAL1','TILT1 VAL2','TILT2 VAL1','TILT2 VAL2': ESPRESSO piezo intel
        # + 'EW': equivalent width
        # + 'biss_span': bissector span
        plot_settings['prop_DI_ordin']=['rv','rv_pip','rv_res','rv_pip_res','RVdrift','rv_l2c','RV_lobe',\
                                        'FWHM','FWHM_pip','FWHM_voigt','FWHM_l2c','FWHM_lobe','FWHM_ord0__IS__VS_','EW','vsini',\
                                        'ctrst','ctrst_pip','true_ctrst','ctrst_ord0__IS__VS_','amp','amp_l2c','amp_lobe','area',\
                                        'cont','c1_pol','c2_pol','c3_pol','c4_pol',\
                                        'biss_span','ha','na','ca','s','rhk',\
                                        'PC0','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10',\
                                        'phase','mu','lat','lon','x_st','y_st',\
                                        'AM','flux_airmass','seeing','snr','snr_quad','snr_R','colcorrmin', 'colcorrmax', 'colcorrR','colcorr450', 'colcorr550', 'colcorr650',\
                                        'PSFx', 'PSFy','PSFr','PSFang','alt',\
                                        'glob_flux_sc','satur_check','ADC1 POS','ADC1 RA','ADC1 DEC','ADC2 POS','ADC2 RA','ADC2 DEC',\
                                        'TILT1 VAL1','TILT1 VAL2','TILT2 VAL1','TILT2 VAL2']     
        
        #%%% Settings for selected properties
        for plot_prop in plot_settings['prop_DI_ordin']:
            key_plot = 'prop_DI_'+plot_prop 

            #%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)               

            #%%%% Plot data-equivalent model from property fit 
            plot_settings[key_plot]['theo_obs_prop'] = False

            #%%%% Plot high-resolution model from property fit
            plot_settings[key_plot]['theo_HR_prop'] = False

            #%%%% Print and plot mean value and dispersion 
            plot_settings[key_plot]['disp_mod']='out' 
            
            #%%%% General path to the best-fit model to property series
            plot_settings[key_plot]['DIProp_path']=None            

            #%%%% Plot high-resolution model from property fit
            plot_settings[key_plot]['theo_HR_prop'] = False
            
            #%%%% Save dispersion values for analysis in external routine
            plot_settings[key_plot]['save_disp']=False
            
            #%%%% Plot in-transit symbols in a different color
            plot_settings[key_plot]['col_in']=''

            #%%%% Overplot transit duration from system properties 
            plot_settings[key_plot]['plot_T14'] = False
  
            #%%%% Normalisation of values by their out-of-transit mean
            plot_settings[key_plot]['norm_ref']=False
            
            #Normalisation of values by their model modulation
            plot_settings[key_plot]['norm_mod']=False     
    
            #%%%% RV plot
            if (plot_prop=='rv'):

                #%%%%% Plot theoretical RV curve
                plot_settings[key_plot]['theoRV'] = True                
    
            #%%%% RV pipeline plot
            if (plot_prop=='rv_pip'):pass 
            
            #%%%% RV residual plot 
            if (plot_prop=='rv_res'):pass

            #%%%% RV pipeline residual plot
            if (plot_prop=='rv_pip_res'):pass
                
            #%%%% FWHM plot
            if (plot_prop=='FWHM'):pass
            
            #%%%% FWHM pipeline plot
            if (plot_prop=='FWHM_pip'):pass
            
            #%%%% Contrast plot
            if (plot_prop=='ctrst'):pass
            
            #%%%% Contrast pipeline plot
            if (plot_prop=='ctrst_pip'):pass
            
            #%%%% Amplitude plot
            if (plot_prop=='amp'):pass

            #%%%% Area plot
            if (plot_prop=='area'):pass
            
            #%%%% Airmass plot
            if (plot_prop=='AM'):pass
            
            #%%%% Seeing plot
            if (plot_prop=='seeing'):pass
            
            #%%%% SNR plot
            if (plot_prop=='snr') or (plot_prop=='snr_quad'):pass
  
            #%%%% RV drift plot
            if (plot_prop=='RVdrift'):pass
            
            #%%%% Color correction coefficients
            if (plot_prop=='colcorrmin'):pass                  
            if (plot_prop=='colcorrmax'):pass
            if (plot_prop=='colcorr450'):pass
            if (plot_prop=='colcorr550'):pass
            if (plot_prop=='colcorr650'):pass

            #%%%% Ratio of lobe FWHM to core FWHM
            if (plot_prop=='FWHM_l2c'):pass
            
            #%%%% Ratio of lobe contrast to core amplitude
            if (plot_prop=='amp_l2c'):pass

            #%%%% RV shift bewtween lobe and core gaussian RV centroid
            if (plot_prop=='rv_l2c'):pass
            
            #%%%% Lobe FWHM 
            if (plot_prop=='FWHM_lobe'):pass

            #%%%% Lobe amplitude
            if (plot_prop=='amp_lobe'):pass

            #%%%% Lobe RV
            if (plot_prop=='RV_lobe'):pass

            #%%%% True contrast
            if (plot_prop=='true_ctrst'):pass
    

    
    
        
        
        
        

        
        
    
    ################################################################################################################  
    #%% Light curves
    ################################################################################################################         
    
    ################################################################################################################    
    #%%% Input light curves
    #    - used to rescale the flux
    #    - one plot for each band, each instrument, all visits
    #    - overplotting all visits together is useful to quickly compare the exposure ranges, the detection of their local stellar profiles, and determine binning windows
    ################################################################################################################ 
    if (plot_dic['input_LC']!=''):
        key_plot = 'input_LC'

        #%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 
        
        #%%%% Plot exposure-averaged light curves used for scaling
        plot_settings[key_plot]['plot_LC_exp'] = True
        
        #%%%% Plot HR input light curves
        plot_settings[key_plot]['plot_LC_HR'] = True
        
        #%%%% Plot raw imported light curve, if available
        plot_settings[key_plot]['plot_LC_imp'] = True

        #%%%% Print visit names
        plot_settings[key_plot]['plot_vis']=True

        #%%%% Indexes of bands to plot
        #     - default is all
        plot_settings[key_plot]['idx_bands']=[]
        
        #%%%% Plot achromatic light curve
        #     - if chromatic light curves were used, will overwrite the plotting of the chromatic light curves over the 'idx_bands' band
        plot_settings[key_plot]['achrom']=False        

        #%%%% Gap between visits light curves  
        plot_settings[key_plot]['lc_gap']=None    




    ##################################################################################################
    #%%% Effective scaling light curves     
    #   - plotting effective light curves used to rescale the flux
    #   - for input spectra only
    #   - one plot for each instrument, each visit, for all wavelengths in a given list
    #   - these scaling light curves are only defined at the phase of observed exposures, and for the corresponding wavelength, which might differ slightly from one exposure to the next
    ##################################################################################################
    if (plot_dic['spectral_LC']!=''):
        key_plot = 'spectral_LC'
        
        #%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 

        #%%%% Wavelengths to plot (A)
        plot_settings[key_plot]['wav_LC']=[5500.]
        
        
        
        
    



        
    ################################################################################################################  
    #%% Differential profiles
    ################################################################################################################        
        
    ################################################################################################################  
    #%%% 2D maps
    ################################################################################################################  

    ################################################################################################################  
    #%%%% Original profiles 
    ################################################################################################################  
    if (plot_dic['map_Diff_prof']!=''):
        key_plot = 'map_Diff_prof'
        
        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 






    ################################################################################################################ 
    #%%% Individual profiles
    ################################################################################################################ 

    ################################################################################################################ 
    #%%%% Original profiles 
    #    - in the star rest frame 
    ################################################################################################################ 
    if (plot_dic['Diff_prof']!=''):
        key_plot = 'Diff_prof'
        
        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 

        #%%%%% Overplot estimates for local stellar profiles
        plot_settings[key_plot]['estim_loc']= False 

        #%%%%% Default mode for estimates of local stellar profiles
        plot_settings[key_plot]['mode_loc_prof_est'] = 'glob_mod'

        #%%%%% Model from the global fit to all CCFs ('global')
        plot_settings[key_plot]['fit_type']='global'      
 


    ##################################################################################################
    #%%% PCA results
    ##################################################################################################
    if (plot_dic['pca_ana']!=''):        
        key_plot = 'pca_ana'

        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 
        
        #%%%%% Subplots
        plot_settings[key_plot]['pc_var'] = True 
        for key in ['pc_rms','pc_bic','pc_hist','pc_prof','fft_prof']:plot_settings[key_plot][key] = False

        #%%%%% PC variances to overplot
        plot_settings[key_plot]['var_list'] = ['pre','post','out']

        #%%%%% Principal components profiles to plot
        plot_settings[key_plot]['pc_list'] = [0]
        plot_settings[key_plot]['pc_col'] = ['dodgerblue']

        #%%%%% FFT profiles to plot
        #    - from original differential profiles ('diff'), corrected differential profiles ('corr'), or bootstrapped corrected differential profiles ('boot') 
        plot_settings[key_plot]['fft_list'] = ['diff','corr','boot']

        #%%%%% Bornes du plot  
        for key in ['x_range_var','y_range_var','x_range_rms','y_range_rms','x_range_bic','y_range_bic','x_range_hist','y_range_hist','x_range_pc','y_range_pc','x_range_pc','y_range_pc','x_range_fft','y_range_fft']:plot_settings[key_plot][key]=None
         







    ##################################################################################################
    #%%% Residual dispersion
    #    - standard deviation with bin size for out-of-transit differential CCFs
    #    - one plot per exposure
    ##################################################################################################
    if (plot_dic['scr_search']!=''):
        key_plot = 'scr_search'
        
        #%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 









            
            
            
            
        
    ##################################################################################################
    #%% Intrinsic profiles
    ##################################################################################################        

    ##################################################################################################
    #%%% 2D maps
    ################################################################################################## 
        
    ##################################################################################################
    #%%%% Original profiles
    ##################################################################################################
    if (plot_dic['map_Intr_prof']!=''):
        key_plot = 'map_Intr_prof'

        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 

        #%%%%% Plot aligned profiles
        plot_settings[key_plot]['aligned']=False


            

    ##################################################################################################
    #%%%% Binned profiles
    ##################################################################################################
    if (plot_dic['map_Intrbin']!=''):
        key_plot = 'map_Intrbin'

        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 


    ##################################################################################################
    #%%%% 1D converted spectra
    ##################################################################################################
    if (plot_dic['map_Intr_1D']!=''):
        key_plot = 'map_Intr_1D'

        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 


    ##################################################################################################
    #%%%% Estimates and residuals
    ##################################################################################################
    for key_plot in ['map_Intr_prof_est','map_Intr_prof_res']:
        if plot_dic[key_plot]!='':

            #%%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 

            #%%%%% Model to retrieve
            plot_settings[key_plot]['mode_loc_prof_est'] = 'glob_mod'

            ##############################################################################
            #%%%%% Estimates
            if key_plot=='map_Intr_prof_est':

                #%%%%%% Model always required
                plot_settings[key_plot]['plot_line_model'] = True
        
            ##############################################################################
            #%%%%% Residuals
            if key_plot=='map_Intr_prof_res':

                #%%%%%% Include out-of-transit residual profiles to the plot
                plot_settings[key_plot]['show_outres']=True
    
                #%%%%%% Correct only for continuum level
                plot_settings[key_plot]['cont_only']=True
                plot_settings[key_plot]['plot_line_model'] = False
                
        

 
    ################################################################################################################ 
    #%%%% PC-based noise model 
    ################################################################################################################   
    if (plot_dic['map_pca_prof']!=''):
        key_plot = 'map_pca_prof'

        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)         

        #%%%%% Colormap
        plot_settings[key_plot]['cmap'] = 'Spectral'









            
    ##################################################################################################
    #%%% Individual profiles
    ##################################################################################################              
            
    ##################################################################################################
    #%%%% Original profiles (grouped)
    #    - for a given visit
    ##################################################################################################
    if (plot_dic['all_intr_data']!=''):
        key_plot = 'all_intr_data'

        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)         

        #%%%%% Data type
        plot_settings[key_plot]['data_type']='CCF' 







    ##################################################################################################
    #%%%% Original profiles
    ##################################################################################################
    for key_plot in ['Intr_prof','Intr_prof_res']:
        if plot_dic[key_plot]!='':

            #%%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)                    

            #%%%%% Model from the global fit to all profiles ('global')
            plot_settings[key_plot]['fit_type']='global'   

            ##############################################################################
            #%%%%% Flux profiles
            if (key_plot=='Intr_prof'):
                
                #%%%%%% Plot aligned profiles
                plot_settings[key_plot]['aligned']=False

            ##############################################################################
            #%%%%% Residuals profiles
            if (key_plot=='Intr_prof_res'):pass
  
                   
   



    ##################################################################################################
    #%%%% Binned profiles
    ##################################################################################################
    for key_plot in ['Intrbin','Intrbin_res']:
        if plot_dic[key_plot]!='':
    
            #%%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)  

            #%%%%% Plot reference level
            plot_settings[key_plot]['plot_reflev'] = False

            ##############################################################################
            #%%%%% Profile and its fit
            if (key_plot=='Intrbin'):pass

            ##############################################################################
            #%%%%% Residuals between the profile and its fit
            if (key_plot=='Intrbin_res'):pass


    


 

    ################################################################################################################   
    #%%%% 1D converted spectra
    ################################################################################################################   
    if ((plot_dic['sp_Intr_1D']!='') or (plot_dic['sp_Intr_1D_res']!='')) and (any('spec' in s for s in data_dic['Intr']['type'].values())):
        for key_plot in ['sp_Intr_1D','sp_Intr_1D_res']:
            if plot_dic[key_plot]!='':

                #%%%%% Generic settings
                plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)  
    
                ##############################################################################
                #%%%%% Profile and its fit
                if (key_plot=='sp_Intr_1D'):pass

                ##############################################################################
                #%%%%% Residuals between the profile and its fit
                if (key_plot=='sp_Intr_1D_res'):pass









    ##################################################################################################
    #%%% Chi2 over intrinsic property series
    ##################################################################################################
    if (plot_dic['chi2_fit_IntrProp']!=''):
        key_plot = 'chi2_fit_IntrProp'

        #%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)  

        #%%%% Threshold to identify and print outliers
        plot_settings[key_plot]['chi2_thresh']=3.  

        #%%%% Property
        plot_settings[key_plot]['prop'] = 'rv'

        #%%%% General path to the best-fit model to property series
        plot_settings[key_plot]['IntrProp_path']=None






    ################################################################################################################ 
    #%%% Range of planet-occulted properties
    ################################################################################################################ 
    if (plot_dic['plocc_ranges']!=''):
        key_plot = 'plocc_ranges'

        #%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)  

        #%%%% Choose values to plot among
        plot_settings[key_plot]['x_prop']='r_proj' 
        
        #%%%% Gap between exposures
        plot_settings[key_plot]['y_gap'] = 0.2
        
        #%%%% New bin ranges
        plot_settings[key_plot]['prop_bin']={}
        
        #%%%% Visit names
        plot_settings[key_plot]['plot_visid'] = True










    ################################################################################################################
    #%%% 1D PDFs from analysis of individual profiles
    ################################################################################################################
    for key_plot in ['prop_DI_mcmc_PDFs','prop_Intr_mcmc_PDFs']:
        if plot_dic[key_plot]!='':

            #%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)  

            #%%%% Figure size
            plot_settings[key_plot]['fig_size']=(10,5)

            #%%%% Choose property to plot
            plot_settings[key_plot]['plot_prop_list']=['rv']
            
            #%%%% Default MCMC settings
            plot_settings[key_plot]['nwalkers'] = 50
            plot_settings[key_plot]['nsteps'] = 1000
    
            #%%%% Number of subplots per row (>=1)
            plot_settings[key_plot]['nsub_col'] = 5

            #%%%% Spacing between subplots
            plot_settings[key_plot]['wspace'] = 0.05
            plot_settings[key_plot]['hspace'] = 0.05

            #%%%% Plot 1 sigma HDI or confidence intervals
            plot_settings[key_plot]['plot_conf_mode']=['HDI']

            #%%%% Tick intervals
            plot_settings[key_plot]['xmajor_int_all'] = {}
            plot_settings[key_plot]['xminor_int_all'] = {}

            #%%%% Common ranges
            plot_settings[key_plot]['x_range_all'] = {}
    
            ##############################################################################
            #%%%% Disk-integrated profiles
            if (key_plot=='prop_DI_mcmc_PDFs'):
                plot_settings[key_plot]['data_mode'] = 'DI'
                plot_settings[key_plot]['data_dic_idx'] = 'DI'

            ##############################################################################
            #%%%% Intrinsic profiles            
            if (key_plot=='prop_Intr_mcmc_PDFs'):
                plot_settings[key_plot]['data_mode'] = 'Intr'
                plot_settings[key_plot]['data_dic_idx'] = 'Diff'
                



        




    ##################################################################################################
    #%%% Properties of planet-occulted stellar CCFs
    ##################################################################################################
    if (plot_dic['prop_Intr']!=''):

        #%%% Ordina properties
        #    - properties:
        # + 'rv' : centroid of the local stellar CCFs in the star rest frame (in km/s)
        # + 'rv_res' residuals from their RRM model (in km/s)        
        # + 'FWHM': width of the local CCFs (in km/s)
        # + 'ctrst': contrast of the local CCFs
        # + 'rv_l2c': RV(lobe)-RV(core) of double gaussian components
        # + 'FWHM_l2c': FWHM(lobe)/FWHM(core) of double gaussian components
        # + 'amp_l2c': contrast(lobe)/contrast(core) of double gaussian components
        plot_settings['prop_Intr_ordin']=['rv','rv_res','rv_l2c','RV_lobe','FWHM','FWHM_voigt','FWHM_l2c','FWHM_lobe','true_FWHM','ctrst','true_ctrst','amp','amp_l2c','amp_lobe','area','a_damp']

        #%%% Settings for selected properties
        for plot_prop in plot_settings['prop_Intr_ordin']:
            key_plot = 'prop_Intr_'+plot_prop 

            #%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 

            #%%%% Print and plot mean value and dispersion 
            plot_settings[key_plot]['disp_mod']='all'  

            #%%%% General path to the best-fit model to property series
            plot_settings[key_plot]['IntrProp_path']=None
                    
            #%%%% General path to the best-fit model to profile series
            plot_settings[key_plot]['IntrProf_path']=None
                
            #%%%% Default MCMC settings
            plot_settings[key_plot]['nwalkers'] = 50
            plot_settings[key_plot]['nsteps'] = 1000

            #%%%% Plot data-equivalent model from property fit 
            #    - this can also be used to check which exposures were fitted, as the best-fit model was only calculated over them in the fitting routine
            plot_settings[key_plot]['theo_obs_prop'] = False

            #%%%% Plot data-equivalent model from profile fit 
            plot_settings[key_plot]['theo_obs_prof'] = False

            #%%%% Plot high-resolution model from property fit
            plot_settings[key_plot]['theo_HR_prop'] = False

            #%%%% Plot high-resolution model from profile fit
            plot_settings[key_plot]['theo_HR_prof'] = False

            #%%%% Print system properties derived from common fit to all exposures (if relevant)
            plot_settings[key_plot]['plot_fit_comm']={}

            #%%%% RV plot
            if (plot_prop=='rv' ):
                
                #%%%%% Plot high-resolution model from nominal values in ANTARESS_systems.py
                plot_settings[key_plot]['theo_HR_nom'] = False            
                
                #%%%%% Overplot the different contributions
                plot_settings[key_plot]['contrib_theo_HR']= []
    
                #%%%%% Calculate model envelope from MCMC results (calculation of models at +-1 sigma range of the parameters)
                plot_settings[key_plot]['calc_envMCMC_theo_HR_nom']=''
            
                #%%%%% Calculate model sample from MCMC results (distribution of models following the PDF of the parameters)
                plot_settings[key_plot]['calc_sampMCMC_theo_HR_nom']=''   
                
                #%%%%% Predict local RVs measurements from nominal model with errors, and SNR of RMR signal
                plot_settings[key_plot]['predic']={}   
                
            #%%%% RV residual plot 
            if (plot_prop=='rv_res' ):

                #%%%%% Overplot the different contributions
                plot_settings[key_plot]['contrib_theo_HR']= []
    
                #%%%%% Subtract full model
                plot_settings[key_plot]['mod_compos'] = 'full'  

            #%%%% FWHM plot
            if (plot_prop=='FWHM' ):pass

            #%%%% Contrast plot
            if (plot_prop=='ctrst' ):pass
    
            #%%%% Damping coefficient
            if (plot_prop=='a_damp' ):pass
                
            #%%%% Lobe-core properties ratios
            if (plot_prop=='dgauss' in data_dic['Intr']['model'].values()):
        
                #%%%%% Ratio of lobe FWHM to core FWHM
                if (plot_prop=='FWHM_l2c' ):pass

                #%%%%% Ratio of lobe contrast to core amplitude
                if (plot_prop=='amp_l2c' ):pass

                #%%%%% RV shift between lobe and core gaussian RV centroid
                if (plot_prop=='rv_l2c' ):pass

            #%%%% True properties for complex models
            if (plot_prop!='gauss' in data_dic['Intr']['model'].values()):
                    
                #%%%%% True FWHM
                if (plot_prop=='true_FWHM' ):pass
    
                #%%%%% True contrast
                if (plot_prop=='true_ctrst' ):pass





    ################################################################################################################ 
    #%% Binned disk-integrated and intrinsic profile series
    #    - plotted together for comparison
    ################################################################################################################  
    if (plot_dic['binned_DI_Intr']!=''):
        key_plot = 'binned_DI_Intr'

        #%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 

        #%%%% Data type
        plot_settings[key_plot]['data_type']='CCF' 

        #%%%% Choose bin dimension for disk-integrated profiles
        plot_settings[key_plot]['dim_plot_DI']='phase' 

        #%%%% Choose bin dimension for intrinsic profiles
        plot_settings[key_plot]['dim_plot_intr']='r_proj' 











    ##################################################################################################
    #%% System view
    ##################################################################################################        
      
    ################################################################################################## 
    #%%% Occulted stellar regions
    #    - for a given system configuration, with RV of the planet-occulted regions
    ##################################################################################################  
    if (plot_dic['occulted_regions']!=''):
        key_plot = 'occulted_regions'

        #%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)         

        #%%%% Plot full orbit
        plot_settings[key_plot]['plot_orb']=True        

        #%%%% Ranges
        plot_settings[key_plot]['x_range'] = np.array([-1.15,1.15])
        plot_settings[key_plot]['y_range'] = np.array([-1.15,1.15])





    ################################################################################################################   
    #%%% Planetary system architecture
    #   - orbits of all planets in the system can be plotted in 3D
    #   - different views can be chosen
    #   - the stellar surface can show limb-darkening or limb-darkening-weighted radial velocity field
    ################################################################################################################ 
    if (plot_dic['system_view']!=''):
        key_plot = 'system_view'

        #%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)

        #%%%% Default planets to plot
        plot_settings[key_plot]['pl_to_plot'] = gen_dic['def_pl']

        #%%%% Number of points in the planet orbits
        plot_settings[key_plot]['npts_orbits'] = np.repeat(10000,len(plot_settings[key_plot]['pl_to_plot'])) 

        #%%%% Number of points in the spot orbits
        plot_settings[key_plot]['npts_orbits_sp'] = 10000

        #%%%% Position of planets along their orbit
        plot_settings[key_plot]['t_BJD'] = None
        plot_settings[key_plot]['GIF_generation'] = False
        plot_settings[key_plot]['xorp_pl'] = np.tile([[-0.5],[0.5]],len(plot_settings[key_plot]['pl_to_plot'])).T
        plot_settings[key_plot]['yorb_pl'] = np.repeat(0.5,len(plot_settings[key_plot]['pl_to_plot']))  

        #%%%% Position of oriented arrow along planet orbits
        plot_settings[key_plot]['xorb_dir'] = np.tile([[-2.],[2.]],len(plot_settings[key_plot]['pl_to_plot'])).T        
        plot_settings[key_plot]['yorb_dir'] = np.repeat(-0.5,len(plot_settings[key_plot]['pl_to_plot'])) 

        #%%%% Apparent size of the planet
        plot_settings[key_plot]['RpRs_pl'] = {pl_loc:data_dic['DI']['system_prop']['achrom'][pl_loc][0] for pl_loc in plot_settings[key_plot]['pl_to_plot']}
       
        #%%%% Orbit colors
        plot_settings[key_plot]['col_orb'] = np.repeat('forestgreen',len(plot_settings[key_plot]['pl_to_plot']))
        plot_settings[key_plot]['col_orb_samp'] = np.repeat('forestgreen',len(plot_settings[key_plot]['pl_to_plot']))
            
        #%%%% Spot trajectory color
        plot_settings[key_plot]['col_orb_sp'] = 'greenyellow'
                
        #%%%% Number of orbits drawn randomly
        plot_settings[key_plot]['norb']=np.repeat(100,len(plot_settings[key_plot]['pl_to_plot'])) 

        #%%%% Ranges of orbital parameters
        plot_settings[key_plot]['lambdeg_err']={}
        plot_settings[key_plot]['aRs_err']={}
        plot_settings[key_plot]['ip_err']={}
        plot_settings[key_plot]['b_range_all']={}            

        #%%%% Plot width
        plot_settings[key_plot]['width']=10.         

        #%%%% Choose view
        plot_settings[key_plot]['conf_system']='sky_ste'

        #%%%% Reference planet for the 'sky_orb' configuration
        plot_settings[key_plot]['pl_ref'] = plot_settings[key_plot]['pl_to_plot'][0]
        
        #%%%% Overlaying grid cell boundaries
        plot_settings[key_plot]['st_grid_overlay']=False
        plot_settings[key_plot]['pl_grid_overlay']=False
        
        #%%%% Number of cells on a diameter of the star and planets (must be odd)
        plot_settings[key_plot]['n_stcell']=theo_dic['nsub_Dstar']
        plot_settings[key_plot]['n_plcell']={}
        for pl_loc in plot_settings[key_plot]['pl_to_plot']:plot_settings[key_plot]['n_plcell'][pl_loc] = theo_dic['nsub_Dpl'][pl_loc]

        #%%%% Color stellar disk with RV, with limb-darkened specific intensity, with gravity-darkened specific intensity, or total flux
        plot_settings[key_plot]['disk_color']='RV'

        #%%%% Plot colorbar
        plot_settings[key_plot]['plot_colbar']=True 

        #%%%% Plot visible stellar equator
        plot_settings[key_plot]['plot_equ_vis']=True  
    
        #%%%% Position of oriented arrow along stellar equator
        plot_settings[key_plot]['xst_dir'] = [-0.1,0.1]  
        plot_settings[key_plot]['yst_dir'] = 0.
    
        #%%%% Plot hidden equator
        plot_settings[key_plot]['plot_equ_hid']=  False

        #%%%% Plot hidden stellar spin
        plot_settings[key_plot]['plot_stspin_hid']=  False
        
        #%%%% Plot stellar spin
        plot_settings[key_plot]['plot_stspin']=True 
        
        #%%%% Plot stellar poles
        plot_settings[key_plot]['plot_poles']=True  
        plot_settings[key_plot]['plot_hidden_pole']= False 

        #%%%% Source for spots
        #    - spot properties can come from three sources for this plot:
        # + the mock dataset (mock_spot_prop) - from mock_dic
        # + fitted spot properties (fit_spot_prop) - from glob_fit_dic
        # + custom user-specified properties (custom_spot_prop) - parameterized below
        plot_settings[key_plot]['mock_spot_prop'] = False
        plot_settings[key_plot]['fit_spot_prop'] = False
        plot_settings[key_plot]['custom_spot_prop'] = {}
        
        #%%%% Path to the file storing the best-fit spot results to plot
        plot_settings[key_plot]['fit_results_file'] = ''
        
        #%%%% Number of positions of the spots to be plotted, equally distributed within the given time range.
        plot_settings[key_plot]['n_image_spots'] = 15

        #%%%% Use stellar rotation period to distribute the positions, instead of time
        plot_settings[key_plot]['plot_spot_all_Peq'] = True  

        #%%%%  Whether we want to show the track of the spots (spot_overlap=True) or the location of the spots (spot_overlap=False).
        # Only activated if we plot multiple exposures.
        plot_settings[key_plot]['spot_overlap'] = False
    
        #%%%% Overlay to the RV-colored disk a shade controlled by flux
        plot_settings[key_plot]['shade_overlay']=True       
        
        #%%%% Number of equi-RV curves
        plot_settings[key_plot]['n_equi']=None     
    
        #%%%% Color range
        plot_settings[key_plot]['val_range'] = None       
    
        #%%%% Overlay axis of selected frame
        plot_settings[key_plot]['axis_overlay']=False      
    
        #%%%% Plot axis boundaries
        plot_settings[key_plot]['margins']=[0.12,0.15,0.85,0.88]    
    
        #%%%% Plot boundaries
        plot_settings[key_plot]['x_range'] = np.array([-1.5,1.5])   
        plot_settings[key_plot]['y_range'] = np.array([-1.5,1.5]) 
        
        #%%%% RV range
        plot_settings[key_plot]['rv_range'] = None

        #%%%% GIF generation when using multiple exposures
        plot_settings[key_plot]['GIF_generation'] = False
    













    ##################################################################################################
    #%% Atmospheric profiles
    ##################################################################################################        

    ##################################################################################################
    #%%% 2D maps
    ################################################################################################## 
  
    ##################################################################################################
    #%%%% Original profiles
    ##################################################################################################
    if (plot_dic['map_Atm_prof']!=''):
        key_plot = 'map_Atm_prof'  

        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 

        #%%%%% Plot profiles in star (False) or planet rest frame (True)
        plot_settings[key_plot]['aligned']=False






    ##################################################################################################
    #%%%% Binned profiles
    ##################################################################################################
    if (plot_dic['map_Atmbin']!=''):
        key_plot = 'map_Atmbin'  

        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)         


            
            

    ##################################################################################################
    #%%%% 1D converted spectra
    ##################################################################################################
    if (plot_dic['map_Atm_1D']!=''):
        key_plot = 'map_Atm_1D' 
        
        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 



    
    
    
    
    
    ##################################################################################################
    #%%% Individual profiles
    ##################################################################################################        
    

    ##################################################################################################
    #%%%% Original profiles (grouped)
    #    - for a given visit
    ################################################################################################## 
    if (plot_dic['all_atm_data']!=''):
        key_plot = 'all_atm_data'
                
        #%%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic) 

        #%%%%% Data type
        plot_settings[key_plot]['data_type']='CCF' 
        





    ##################################################################################################
    #%%%% Original profiles
    ##################################################################################################
    for key_plot in ['Atm_prof','Atm_prof_res']:
        if plot_dic[key_plot]!='':

            #%%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)                    

            #%%%%% Model from the global fit to all profiles ('global')
            plot_settings[key_plot]['fit_type']='global'   

            ##############################################################################
            #%%%%% Atmospheric profiles
            if (key_plot=='Atm_prof'):
                
                #%%%%%% Plot aligned profiles
                plot_settings[key_plot]['aligned']=False

            ##############################################################################
            #%%%%% Residuals profiles
            if (key_plot=='Atm_prof_res'):
                if not ((gen_dic['fit_Atm']) or (gen_dic['fit_AtmProf'])): 
                    if (not gen_dic['fit_Atm']):stop('Activate "gen_dic["fit_Atm"]" to plot "plot_dic["Atm_prof_res"]"')     
                    if (not gen_dic['fit_AtmProf']):stop('Activate "gen_dic["fit_AtmProf"]" to plot "plot_dic["Atm_prof_res"]"')   
                        







    ##################################################################################################
    #%%%% Binned profiles
    ##################################################################################################
    for key_plot in ['Atmbin','Atmbin_res']:
        if plot_dic[key_plot]!='':
    
            #%%%%% Generic settings
            plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)  

            ##############################################################################
            #%%%%% Profile and its fit
            if (key_plot=='Atmbin'):pass

            ##############################################################################
            #%%%%% Residuals between the profile and its fit
            if (key_plot=='Atmbin_res'):pass








    ################################################################################################################   
    #%%%% 1D converted spectra
    ################################################################################################################   
    if ((plot_dic['sp_Atm_1D']!='') or (plot_dic['sp_Atm_1D_res']!='')) and (any('spec' in s for s in data_dic['Atm']['type'].values())):
        for key_plot in ['sp_Atm_1D','sp_Atm_1D_res']:
            if plot_dic[key_plot]!='':

                #%%%%% Generic settings
                plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)  
    
                ##############################################################################
                #%%%%% Profile and its fit
                if (key_plot=='sp_Atm_1D'):pass

                ##############################################################################
                #%%%%% Residuals between the profile and its fit
                if (key_plot=='sp_Atm_1D_res'):pass







    ##################################################################################################
    #%%% Chi2 over atmospheric property series
    ##################################################################################################
    if (plot_dic['chi2_fit_AtmProp']!=''):
        key_plot = 'chi2_fit_AtmProp'

        #%%%% Generic settings
        plot_settings=gen_plot_default(plot_settings,key_plot,plot_dic,gen_dic,data_dic)  

        #%%%% Threshold to identify and print outliers
        plot_settings[key_plot]['chi2_thresh']=3.  

        #%%%% Property
        plot_settings[key_plot]['prop'] = 'rv'

        #%%%% General path to the best-fit model to property series
        plot_settings[key_plot]['AtmProp_path']=None        
        
        
        
        
        


        
    return plot_settings