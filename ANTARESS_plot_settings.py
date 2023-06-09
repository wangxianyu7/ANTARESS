#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 14:27:38 2022

@author: V. Bourrier
"""
import numpy as np
from constant_data import c_light

  
def ANTARESS_plot_settings(plot_dic,gen_dic,data_dic):
    plot_settings={}


    ################################################################################################################    
    #%% Detector gain estimates
    ################################################################################################################ 
    if (plot_dic['det_gain']!='') or (plot_dic['det_gain_ord']!=''):
        key_plot = 'det_gain'
        plot_settings[key_plot]={}

        #Margins
        if gen_dic['star_name']=='WASP76':plot_settings[key_plot]['margins']=[0.15,0.15,0.95,0.7] 

        #Font size
        plot_settings[key_plot]['font_size']=18

        #Raster
        plot_settings[key_plot]['rasterized'] = False

        # #Exposures to plot
        # #    - indexes are relative to global tables
        # if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['iexp_plot']={'ESPRESSO':{
        #         '20181030':[30],           #ANTARESS I
        #         '20180902':[30]}}    
    
        #Choice of orders to plot
        #    - leave empty to plot all orders
        plot_settings[key_plot]['orders_to_plot']=[]
        plot_settings[key_plot]['orders_to_plot']=[]
        # if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['orders_to_plot']=[140]   #ANTARESS I
        #     # plot_settings[key_plot]['orders_to_plot']=np.delete(np.arange(170),[90,91]) 
        #     # plot_settings[key_plot]['orders_to_plot']=[106]  
        # if gen_dic['star_name']=='HD209458':
        #     plot_settings[key_plot]['orders_to_plot']=[110,112,146,148,162,164,166]   

        
        #Plot best-fit exposure models in each order plot
        plot_settings[key_plot]['plot_best_exp'] = True  #& False        
        
        #Normalize gain profiles over all exposures
        #    - to compare profile changes in the 'det_gain_ord' plot
        plot_settings[key_plot]['norm_exp'] = False        

        #Plot mean gain if calculated
        #    - in the 'det_gain_ord' plot
        plot_settings[key_plot]['mean_gdet']  = True    

        #Range over time
        plot_settings[key_plot]['x_range']=[3500.,8000.] 
        # if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['x_range']=[3780.,4200.]             
        #     # plot_settings[key_plot]['y_range']=[0.,500.] 
        # if gen_dic['star_name']=='GJ3090':
        #     plot_settings[key_plot]['y_range'] = [0,1000.]
        if gen_dic['star_name']=='55Cnc':
            plot_settings[key_plot]['x_range']=[4800.,7500.]   #EXPRES
    
        #Range per order
        #    - unfitted points are not shown in automatic ranges
        # if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['x_range_ord']=[5190.,5280.]
        #     plot_settings[key_plot]['x_range_ord']=[6603.,6706.]   #ANTARESS I
        #     plot_settings[key_plot]['y_range_ord']=[1.6,10.5]   
        
            
     
        
     
        
     
        







    ################################################################################################################    
    #%% Weighing master
    ################################################################################################################
    if gen_dic['specINtype'] and (plot_dic['DImast']!=''):
        key_plot = 'DImast'
        plot_settings[key_plot]={} 

        #Margins
        if gen_dic['star_name']=='WASP76':plot_settings[key_plot]['margins']=[0.1,0.15,0.95,0.7]

        #Raster
        plot_settings[key_plot]['rasterized'] = False    #ANTARESS I, cont
        # plot_settings[key_plot]['rasterized'] = True    #ANTARESS I, all spec

        #Font size
        plot_settings[key_plot]['font_size']=18

        #Line width for individual exposures
        plot_settings[key_plot]['lw_plot'] = 1.

        #Instruments and visits to plot
        if gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']}  
        elif gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20180902','20181030']} 
            
        #Choice of orders to plot
        #    - leave empty to plot all orders
        # plot_settings[key_plot]['orders_to_plot']=[10,11,12,13]
        plot_settings[key_plot]['orders_to_plot']=[]
        if gen_dic['studied_pl']==['HD3167_b']:
            plot_settings[key_plot]['orders_to_plot']=range(76,111)
        # if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['orders_to_plot']=[140]   #ANTARESS I
    
        #Scaling factor
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['sc_fact10'] = -3     #ANTARESS I

        #Plot boundaries in wav
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['x_range']=[6603.,6706.]     #ANTARESS I
          
        #Plot boundaries in flux
        if gen_dic['studied_pl']==['HD3167_b']:
            plot_settings[key_plot]['y_range']=[-2000.,11000.]    #deblazed
            plot_settings[key_plot]['y_range']=[-1000.,5000.]    #blazed
        # if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['y_range']=[0.,18000.]   #ANTARESS I    V1
        #     # plot_settings[key_plot]['y_range']=[200.,9000.]   #ANTARESS I    V2
        #     plot_settings[key_plot]['y_range']=[1500.,12500.]   #ANTARESS I,cont V1
        #     # plot_settings[key_plot]['y_range']=[500.,8000.]   #ANTARESS I,cont V2
            
        #Colors
        if gen_dic['studied_pl']==['HD3167_b']:
            plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-10-09':'dodgerblue'}}   



            





    ################################################################################################################    
    #%% Telluric CCF
    ################################################################################################################  
    if gen_dic['specINtype'] and (plot_dic['tell_CCF']!=''):
        key_plot = 'tell_CCF'
        plot_settings[key_plot]={}

        #Font size
        plot_settings[key_plot]['font_size']=18

        #Margins
        if gen_dic['star_name']=='WASP76':plot_settings[key_plot]['margins']=[0.15,0.15,0.95,0.7]    

        #Instruments and visits to plot
        # elif gen_dic['star_name']=='HD209458':
        #     plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20190911']} 

        #Exposures to plot
        plot_settings[key_plot]['iexp_plot'] = {}
        # if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['iexp_plot'] = {'ESPRESSO':{'20180902':[30],'20181030':[0]}}

        #Linewidth
        plot_settings[key_plot]['lw_plot']=1.

        #Ranges
        if gen_dic['star_name']=='HD209458':
            plot_settings[key_plot]['y_range'] = {'ESPRESSO':{'20190720':{'H2O':[0.6,1.1],'O2':[0.55,1.1]},'20190911':{'H2O':[0.45,1.1],'O2':[0.55,1.1]}}}
        # elif gen_dic['star_name']=='WASP76':
        #     # plot_settings[key_plot]['y_range'] = {'ESPRESSO':{'20180902':{'H2O':[0.5,1.1],'O2':[0.5,1.1]},'20181030':{'H2O':[0.15,1.1],'O2':[0.5,1.1]}}}
        #     plot_settings[key_plot]['x_range'] = [-25.,25.]   #ANTARESS I, tell
        #     plot_settings[key_plot]['y_range'] = {'ESPRESSO':{'20180902':{'H2O':[0.7,1.08]}}}   

        #Molecules to plot
        # plot_settings[key_plot]['tell_species'] = []     
        
        #Plot dispersion 
        #    - set to None, or to requested range
        plot_settings[key_plot]['print_disp']=[-30.,30.]







    ################################################################################################################    
    #%% Telluric properties
    ################################################################################################################   
    if gen_dic['specINtype'] and (plot_dic['tell_prop']!=''):
        key_plot = 'tell_prop'
        plot_settings[key_plot]={}

        #Font size
        plot_settings[key_plot]['font_size']=18

        #Plot errors
        plot_settings[key_plot]['plot_err'] = True #False

        #Margins
        if gen_dic['star_name']=='WASP76':plot_settings[key_plot]['margins']=[0.15,0.15,0.95,0.7]       

        #Molecules to plot
        # plot_settings[key_plot]['tell_species'] = []    

        #Ranges
        # if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['y_range'] = {'ESPRESSO':{'20180902':{'O2':{'Pressure_ground':[0.,0.2]}}}}
  



  


        
        













     
    ################################################################################################################    
    #%% Global scaling masters        
    #    - measured master used in the flux scaling of the spectra
    ################################################################################################################
    if gen_dic['specINtype'] and (plot_dic['glob_mast']!=''):
        key_plot = 'glob_mast'
        plot_settings[key_plot]={} 

        #Margins
        if gen_dic['star_name']=='WASP76':plot_settings[key_plot]['margins']=[0.1,0.15,0.95,0.7]

        #Raster
        plot_settings[key_plot]['rasterized'] = True    #ANTARESS I, vis master + all spec


        #Font size
        plot_settings[key_plot]['font_size']=18

        #Line width for individual exposures
        plot_settings[key_plot]['lw_plot'] = 0.1

        #Plot visit-specific masters 
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['glob_mast_vis']=True    #ANTARESS I, vis master + all spec   

        #Plot reference master
        #    - set to the measured multi-visit master ('meas'), or to a theoretical reference ('theo'), if available
        #    - set to None to prevent
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['glob_mast_all']=None    #ANTARESS I, vis master + all spec   
            plot_settings[key_plot]['glob_mast_all']='meas'     #ANTARESS I, vis master + inst master

        #Instruments and visits to plot
        if gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']}  
        # elif gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20180902','20181030']} 

        #Plot all exposures
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['plot_input'] = True    #ANTARESS I, vis master + all spec
            # plot_settings[key_plot]['plot_input'] = False    #ANTARESS I, vis master + inst master
            # plot_settings[key_plot]['plot_input'] = False #ANTARESS I, cont

        #Exposures to plot
        #    - indexes are relative to global tables
        # if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['iexp_plot']={'ESPRESSO':{
        #         '20180902':[0],          
        #         # '20181030':[52,53],         
        #         },    
        #     }   

        #Choice of orders to plot
        #    - leave empty to plot all orders
        plot_settings[key_plot]['orders_to_plot']=[10,11,12,13]
        plot_settings[key_plot]['orders_to_plot']=[0,1,2,3]
        if gen_dic['studied_pl']==['HD3167_b']:
            plot_settings[key_plot]['orders_to_plot']=range(76,111)
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['orders_to_plot']=[140]   #ANTARESS I
    
        #Scaling factor
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['sc_fact10'] = -5     #ANTARESS I

        #Plot boundaries in wav
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['x_range']=[6603.,6706.]     #ANTARESS I
          

        #Plot boundaries in flux
        if gen_dic['studied_pl']==['HD3167_b']:
            plot_settings[key_plot]['y_range']=[-2000.,11000.]    #deblazed
            plot_settings[key_plot]['y_range']=[-1000.,5000.]    #blazed
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['y_range']=[2e5,3.1e6]   #ANTARESS I    V1, vis-masters, all spec
            plot_settings[key_plot]['y_range']=[3e5,1.7e6]   #ANTARESS I    V2, vis-masters, all spec
            
        #Colors
        if gen_dic['studied_pl']==['HD3167_b']:
            plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-10-09':'dodgerblue'}}             
            
            





    ################################################################################################################    
    #%% Global flux balance (exposures)
    ################################################################################################################
    if gen_dic['specINtype'] and (plot_dic['Fbal_corr']!=''):   
        key_plot = 'Fbal_corr'
        plot_settings[key_plot]={}     
   
        #Font size
        plot_settings[key_plot]['font_size']=18        
           
        #Plot exposure indexes
        plot_settings[key_plot]['plot_expid'] = False  
        
        #Raster
        plot_settings[key_plot]['rasterized'] = True

        #Plot as a function of nu = c/w
        plot_settings[key_plot]['sp_var'] = 'nu'    
    
        #Overplot all exposures or offset them
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['gap_exp'] = 0.05    #exploration
            plot_settings[key_plot]['gap_exp'] = 0.   #ANTARESS I
        elif gen_dic['star_name']=='HD209458':
            plot_settings[key_plot]['gap_exp'] = 0.1    #exploration
            plot_settings[key_plot]['gap_exp'] = 0.05
            # plot_settings[key_plot]['gap_exp'] = 0.
        elif gen_dic['star_name']=='WASP107':
            plot_settings[key_plot]['gap_exp'] = 0.5    #exploration
            plot_settings[key_plot]['gap_exp'] = 0.
        elif gen_dic['star_name']=='HAT_P11':
            plot_settings[key_plot]['gap_exp'] = 0.5    #exploration
            # plot_settings[key_plot]['gap_exp'] = 0.
        elif gen_dic['star_name']=='WASP156':
            plot_settings[key_plot]['gap_exp'] = 0.5    #exploration
            # plot_settings[key_plot]['gap_exp'] = 0.
        elif gen_dic['star_name']=='55Cnc':
            plot_settings[key_plot]['gap_exp'] = 0.5    #exploration
            plot_settings[key_plot]['gap_exp'] = 0.2    #EXPRESSO
            plot_settings[key_plot]['gap_exp'] = 0.05   #EXPRES
        elif gen_dic['star_name']=='GJ3090':
            plot_settings[key_plot]['gap_exp'] = 5.    
            plot_settings[key_plot]['gap_exp'] = 0.5 
        elif gen_dic['star_name']=='HD29291':   
            plot_settings[key_plot]['gap_exp'] = 0.05     
    
        #Colors
        # if gen_dic['studied_pl']=='WASP76b':
        #     plot_settings[key_plot]['color_dic']={'2018-10-31':'dodgerblue','2018-09-03':'red'}      

        #Indexes of exposures and bins to be plotted 
        #    - use this option to identify bins biasing the fit
        #    - indexes must match the tables used for the fit
        # if gen_dic['star_name']=='WASP76':
        # #     # plot_settings[key_plot]['ibin_plot']=np.delete(np.arange(85),[44,45,64,73,81,82])    #2018-09-03
        # #     plot_settings[key_plot]['ibin_plot']=np.delete(np.arange(85),[44,45,73,77,81,82])    #2018-10-31
        # #     plot_settings[key_plot]['iexp_plot']={'ESPRESSO':{'2018-10-31':[16],'2018-09-03':[0,1,2]}}  
        #     plot_settings[key_plot]['ibin_plot']={'ESPRESSO':{
        #         '20180902':np.delete(np.arange(170),[146,147,162,163,164,165]),
        #         '20181030':np.delete(np.arange(170),[146,147,162,163,164,165])}}           
        # if gen_dic['star_name']=='HD209458':
        #     # plot_settings[key_plot]['ibin_plot']={'ESPRESSO':{'20190720':np.delete(np.arange(170),[78,79,80,81,82,83,84,85,86,87,88,89,   146,147,  162,163,  164,165])}}   
        #     # plot_settings[key_plot]['ibin_plot']['ESPRESSO']['20190911']=plot_settings[key_plot]['ibin_plot']['ESPRESSO']['20190720']        
        #     plot_settings[key_plot]['ibin_plot']={'ESPRESSO':{
        #         '20190720':np.delete(np.arange(170),[146,147,162,163,164,165]),
        #         '20190911':np.delete(np.arange(170),[146,147,162,163,164,165])}}        

    
        #Bornes du plot  
        plot_settings[key_plot]['x_range']=[3500.,8000.] 
        plot_settings[key_plot]['x_range']=[3783.,7880.]   #ESPRESSO range
        plot_settings[key_plot]['x_range'] = None
        plot_settings[key_plot]['y_range']=None 
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['x_range']=[3700.,7950.]    
      
            plot_settings[key_plot]['x_range']=[c_light/7880.,c_light/3783.]   #ANTARESS I
            # plot_settings[key_plot]['y_range']=[0.52,1.15]

        if gen_dic['star_name']=='HD209458':
            plot_settings[key_plot]['x_range']=[c_light/7880.,c_light/3783.]   #ANTARESS I
            # plot_settings[key_plot]['y_range']=[0.72,1.15]

        if gen_dic['studied_pl']==['HD3167_b']:
            plot_settings[key_plot]['y_range']=[0.4,2.] 
        elif gen_dic['star_name'] in ['WASP107']:   
            plot_settings[key_plot]['x_range']=[5000.,11000.]             
            plot_settings[key_plot]['x_range']=[5000.,10000.]              
            plot_settings[key_plot]['x_range']=[5190.,9950.]      #final trimmed range         
        elif gen_dic['star_name'] in ['HAT_P11']:   
            plot_settings[key_plot]['x_range']=[5000.,11000.]              
            plot_settings[key_plot]['x_range']=[5100.,10000.]      #final trimmed range                 
        elif gen_dic['star_name'] in ['WASP156']:                
            plot_settings[key_plot]['x_range']=[5100.,10000.]      #final trimmed range   
        # elif gen_dic['star_name'] in ['GJ3090']:                
        #     plot_settings[key_plot]['y_range']=[-2.,195.]    #V1      
        #     # plot_settings[key_plot]['y_range']=[-5.,320.]    #V2   
    

    



    ##################################################################################################
    #%% DRS global flux balance correction
    ##################################################################################################
    if gen_dic['specINtype'] and (plot_dic['Fbal_corr_DRS']!=''):
        key_plot = 'Fbal_corr_DRS'
        plot_settings[key_plot]={}
        
        
        #Visits to plot
        if gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']}           
        
        #Bornes du plot  
        plot_settings[key_plot]['x_range']=[3500.,8000.] 
        
        #Colors
#        if gen_dic['studied_pl']=='WASP76b':
#            plot_settings[key_plot]['color_dic']={'2018-10-31':'dodgerblue','2018-09-03':'red'}          
    
    

    ################################################################################################################    
    #%% Global flux balance (visits)
    ################################################################################################################
    if gen_dic['specINtype'] and (plot_dic['Fbal_corr_vis']!=''):   
        key_plot = 'Fbal_corr_vis'
        plot_settings[key_plot]={}     
   
        #Font size
        plot_settings[key_plot]['font_size']=18        

        #Raster
        plot_settings[key_plot]['rasterized'] = True

        #Plot as a function of nu = c/w
        plot_settings[key_plot]['sp_var'] = 'nu'    
    
        #Overplot all exposures or offset them
        plot_settings[key_plot]['gap_exp'] = 0.
            
        #Colors
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['color_dic']={'ESPRESSO':{'20180902':'dodgerblue','20181030':'limegreen'}}      

        #Indexes of bins to be plotted 
        #    - use this option to identify bins biasing the fit
        #    - indexes must match the tables used for the fit

           
        #Plot visit indexes
        plot_settings[key_plot]['plot_expid'] = True   & False
        
        
        #Bornes du plot  
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['x_range']=[3700.,7950.]    
      
            plot_settings[key_plot]['x_range']=[c_light/7880.,c_light/3783.]   #ANTARESS I
            plot_settings[key_plot]['y_range']=[0.88,1.08]

        if gen_dic['star_name']=='HD209458':
            plot_settings[key_plot]['x_range']=[c_light/7880.,c_light/3783.]   #ANTARESS I
            # plot_settings[key_plot]['y_range']=[0.72,1.15]




    
    
    
    




    ##################################################################################################
    #%% Intra-order flux balance
    ##################################################################################################
    if gen_dic['specINtype'] and (plot_dic['Fbal_corr_ord']!=''):
        key_plot = 'Fbal_corr_ord' 
        plot_settings[key_plot]={}   


        #Visits to plot    
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}         
        elif gen_dic['studied_pl']==['HD3167_b']:
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']}  
    
        #Choice of orders to plot
        #    - leave empty to plot all orders
        # if gen_dic['studied_pl']==['HD3167_b']:
        #     plot_settings[key_plot]['orders_to_plot']=range(76,111)
        
        #Colors
#        if gen_dic['studied_pl']=='WASP76b':
#            plot_settings[key_plot]['color_dic']={'2018-10-31':'dodgerblue','2018-09-03':'red'}          
        
        
        
        
        
        
        
        
        
        
    '''
    Temporal flux correction
    '''
    if gen_dic['specINtype'] and (plot_dic['Ftemp_corr']!=''):
        key_plot = 'Ftemp_corr'
        plot_settings[key_plot]={}

        #Visits to be plotted
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}         
             
        #Colors
#        if gen_dic['studied_pl']=='WASP76b':
#            plot_settings[key_plot]['color_dic']={'2018-10-31':'dodgerblue','2018-09-03':'red'}       
        




                 


    '''
    Plotting cosmics correction
    '''
    if gen_dic['specINtype'] and (plot_dic['cosm_corr']!=''):
        key_plot = 'cosm_corr'
        plot_settings[key_plot]={}
        
        #Margins
        if gen_dic['star_name']=='WASP76':plot_settings[key_plot]['margins'] = [0.13,0.15,0.95,0.75]    #ANTARESS I
        
        #Line width
        plot_settings[key_plot]['lw_plot'] = 0.8
    

        #Visits to plot
        # if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20181030']}

        #Indexes of exposures to be plotted 
        # if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['iexp_plot']={'ESPRESSO':{'20181030':[5]}}  
        #     # plot_settings[key_plot]['iexp_plot']={'ESPRESSO':{'20181030':[68]}}
        # # if gen_dic['star_name']=='WASP107':
        # #     plot_settings[key_plot]['iexp_plot']={'CARMENES_VIS':{'20180224':[10]}}  

        #Choice of orders to plot
        #    - leave empty to plot all orders 
        # if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['orders_to_plot']=[131]
        #     # plot_settings[key_plot]['orders_to_plot']=[121]
        # # elif gen_dic['star_name']=='WASP107':
        # #     plot_settings[key_plot]['orders_to_plot']=[2]


        #Plot boundaries
        #    - set to None to plot full range
        # if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['x_range']=[6321.5,6324.5]  #ANTARESS I, cosm
        #     plot_settings[key_plot]['y_range']=[-5.,75.] 
        #     # plot_settings[key_plot]['x_range']=[5985.5,5989.5]      #ANTARESS I, false cosm           
        #     # plot_settings[key_plot]['y_range']=[-5.,20.]             
        # if gen_dic['star_name']=='HD209458':     
        #     plot_settings[key_plot]['y_range']=[-5.,75.] 

        #Plot number of cosmics per order
        plot_settings[key_plot]['cosm_vs_ord']=True 










    '''
    Plotting master for persistent peaks correction
    '''
    if gen_dic['specINtype'] and (plot_dic['permpeak_corr']!=''):
        key_plot = 'permpeak_corr'
        plot_settings[key_plot]={}
        
        
        
        
        
        
        
        
        












    '''
    Fringing correction
    '''
    if gen_dic['specINtype'] and (plot_dic['fring_corr']!=''):
        key_plot = 'fring_corr'
        plot_settings[key_plot]={}

        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}         

        #Bin datapoints
        #    - set binsize (A) for each instrument
        #    - leave empty not to be bin
        plot_settings[key_plot]['bin_data'] = {'ESPRESSO':1.}
        
        #Ranges       
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['x_range']=[3500.,8000.] 
            plot_settings[key_plot]['y_range']=[0.,2.]      




    '''
    Plotting individual disk-integrated spectra in their input rest frame
    '''
    if gen_dic['specINtype'] and (plot_dic['sp_raw']!=''):
        key_plot = 'DI_raw'
        plot_settings[key_plot]={}

        #Font size
        plot_settings[key_plot]['font_size']=18

        #Raster
        plot_settings[key_plot]['rasterized'] = False
        
        #Linewidth
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['lw_plot']=0.5     #ANTARESS I, gain, tell
            # plot_settings[key_plot]['lw_plot']=0.8     #ANTARESS I, cosm
        if gen_dic['star_name']=='WASP156':
            plot_settings[key_plot]['lw_plot']=0.5     #ANTARESS I, persistent peaks


        #Margins
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['margins']=[0.15-0.05,0.15,0.95-0.05,0.8]         #ANTARESS I, gain, tell
            # plot_settings[key_plot]['margins'] = [0.08,0.15,0.9,0.75]    #ANTARESS I, cosm
        if gen_dic['star_name']=='WASP156':
            plot_settings[key_plot]['margins'] = [0.1,0.15,0.9,0.75]     #ANTARESS I, persistent peaks


        #Instruments and visits to plot
        if gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']}               
        # elif gen_dic['star_name']=='WASP76':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20181030']}            
        elif gen_dic['star_name']=='WASP156':plot_settings[key_plot]['visits_to_plot']={'CARMENES_VIS':['20191025']}   
        # elif gen_dic['star_name']=='HD209458':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20190911']}  

        #Exposures to plot
        #    - indexes are relative to global tables
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['iexp_plot']={'ESPRESSO':{
                # '20180902':[0],            #ANTARESS I, gain
                # '20181030':[53],   
                '20181030':[30],           #ANTARESS I, tell
                '20180902':[30],
                # '20181030':[5],     #ANTARESS I, cosm
                # '20181030':[68],     #ANTARESS I, false cosm    
                }}   
        if gen_dic['star_name']=='WASP107':
            plot_settings[key_plot]['iexp_plot']={'CARMENES_VIS':{'20180224':[0,10,18]}}  
        # elif gen_dic['star_name']=='HAT_P11':
        #     plot_settings[key_plot]['iexp_plot']={'CARMENES_VIS':{'20170807':[0,30,59],'20170812':[0,30,57]}}  
        # elif gen_dic['star_name']=='WASP156':
        #     plot_settings[key_plot]['iexp_plot']={'CARMENES_VIS':{'20191025':[0]}}           #ANTARESS I, persistent peaks 1  



        #Plot all exposures on the same plot
        if gen_dic['star_name']=='WASP156':plot_settings[key_plot]['multi_exp']=True  # #ANTARESS I, persistent peaks 2 et 3  
        elif gen_dic['star_name']=='HD209458':plot_settings[key_plot]['multi_exp']=True   & False

        #Choice of orders to plot
        #    - leave empty to plot all orders
        plot_settings[key_plot]['orders_to_plot']=[10,11,12,13]
        plot_settings[key_plot]['orders_to_plot']=[0,1,2,3]
        plot_settings[key_plot]['orders_to_plot']=[]
        if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['orders_to_plot']=range(100,120)
            # plot_settings[key_plot]['orders_to_plot']=[140]   #ANTARESS I, gain
            plot_settings[key_plot]['orders_to_plot']=[157]   #ANTARESS I, tell
        #     plot_settings[key_plot]['orders_to_plot']=[131]   #ANTARESS I, cosm
        #     plot_settings[key_plot]['orders_to_plot']=[121]   #ANTARESS I, false cosm 
            # plot_settings[key_plot]['orders_to_plot']=[110,111,112,113,146,147,148,149,162,163,164,165,166,167]     
            # plot_settings[key_plot]['orders_to_plot']=[164]        
        
        elif gen_dic['star_name']=='HD209458':            
            # plot_settings[key_plot]['orders_to_plot']=[133]       
            # plot_settings[key_plot]['orders_to_plot']=[110,111,112,113,146,147,148,149,162,163,164,165,166,167]       
            plot_settings[key_plot]['orders_to_plot']=[116,117]  
            
        if gen_dic['star_name']=='WASP107':
            plot_settings[key_plot]['orders_to_plot']=[8]
        elif gen_dic['star_name']=='HAT_P11':
            plot_settings[key_plot]['orders_to_plot']=[8]
            plot_settings[key_plot]['orders_to_plot']=[25]
            plot_settings[key_plot]['orders_to_plot']=[49]
        elif gen_dic['star_name']=='WASP156':
            plot_settings[key_plot]['orders_to_plot']=[46]         #ANTARESS I, persistent peaks 1, 2
            plot_settings[key_plot]['orders_to_plot']=[8]      #ANTARESS I, persistent peaks 3  

        #Only plot exposures & orders with masked persistent pixels        
        if gen_dic['star_name']=='WASP156':plot_settings[key_plot]['det_permpeak']=True      #ANTARESS I, persistent peaks  
        # elif gen_dic['star_name']=='HD209458':plot_settings[key_plot]['det_permpeak']=True      #ANTARESS I, persistent peaks  


        #Plot HITRAN telluric lines for requested molecules
        #    - only for line with depth (%) beyond threshold
        # plot_settings[key_plot]['plot_tell_HITRANS'] = ['O2','H2O']  
        plot_settings[key_plot]['telldepth_min'] = 0.05  


        #Plot telluric spectrum
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['plot_tell'] = True    #ANTARESS I, tell


        #Plot spectra at two chosen steps of the correction process
        #    - set to None, or chose amongst:
        # + 'raw' : before any correction
        # + 'all' : after all requested corrections
        # + 'tell' : after telluric correction 
        # + 'count' : after flux-to-count scaling
        # + 'fbal' : after flux balance correction 
        # + 'cosm' : after cosmics correction  
        # + 'permpeak' : after persistent peak correction 
        if gen_dic['star_name']=='WASP76':    
            plot_settings[key_plot]['color_dic']={'ESPRESSO':{'20180902':'red','20181030':'red'}} 
            plot_settings[key_plot]['color_dic_sec']={'ESPRESSO':{'20180902':'dodgerblue','20181030':'dodgerblue'}}            
    
            plot_settings[key_plot]['plot_pre']='raw'     #ANTARESS I, tell   
            plot_settings[key_plot]['plot_post']='tell'

        # plot_settings[key_plot]['plot_pre']='raw'   #ANTARESS I, global fbal
        # plot_settings[key_plot]['plot_post']='fbal'

        # plot_settings[key_plot]['plot_pre']='fbal'   #ANTARESS I, cosmic corr
        # plot_settings[key_plot]['plot_post']='cosm'

        # plot_settings[key_plot]['plot_pre']=None    
        # plot_settings[key_plot]['plot_post']='count'
        
        # plot_settings[key_plot]['plot_pre']='count'    
        # plot_settings[key_plot]['plot_post']='cosm'

        # plot_settings[key_plot]['plot_pre']='fbal'    
        # plot_settings[key_plot]['plot_post']='cosm'

        if gen_dic['star_name']=='WASP156':    #ANTARESS I, persistent peaks  
            plot_settings[key_plot]['plot_pre']='cosm'                             
            plot_settings[key_plot]['plot_post']='permpeak' 
            # plot_settings[key_plot]['color_dic']={'CARMENES_VIS':{'20191025':'red'}} #ANTARESS I, persistent peaks   1
            # plot_settings[key_plot]['color_dic_sec']={'CARMENES_VIS':{'20191025':'dodgerblue'}}

       # plot_settings[key_plot]['plot_pre']='cosm'    
       # plot_settings[key_plot]['plot_post']='permpeak'     
        
        # plot_settings[key_plot]['plot_pre']='fbal'    
        # plot_settings[key_plot]['plot_post']='permpeak'  

        if gen_dic['star_name']=='HD209458':  
            plot_settings[key_plot]['plot_pre']='raw'                             
            plot_settings[key_plot]['plot_post']=None 
#            plot_settings[key_plot]['plot_pre']='count'                             
#            plot_settings[key_plot]['plot_post']='tell' 
            # plot_settings[key_plot]['plot_pre']='cosm'                             
            # plot_settings[key_plot]['plot_post']='permpeak' 


        #Plot continuum used for persistent peak masking
        if gen_dic['star_name']=='WASP156':plot_settings[key_plot]['plot_contmax']=True      #ANTARESS I, persistent peaks  
        if gen_dic['star_name']=='HD209458':plot_settings[key_plot]['plot_contmax']=True       #ANTARESS I, persistent peaks  
        
        #Normalize spectra to integrated flux unity
        #    - to allow for comparison
        if gen_dic['star_name']=='WASP156':plot_settings[key_plot]['norm_prof']=True        #ANTARESS I, persistent peaks  
        if gen_dic['star_name']=='HD209458':plot_settings[key_plot]['norm_prof']=True & False        #ANTARESS I, persistent peaks  

        #Scaling factor
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['sc_fact10'] = -5     #ANTARESS I, tell, gain, cosm    

        #Plot error
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['plot_err'] = True   #ANTARESS I, gain, fbal, cosm
            plot_settings[key_plot]['plot_err'] = False   #ANTARESS I, tell

        #Plot boundaries in wav
        if gen_dic['star_name']=='WASP76':
            # plot_settings[key_plot]['x_range']=[7665.,7715.] 
            # plot_settings[key_plot]['x_range']=[6603.,6706.]   #ANTARESS I, gain
            plot_settings[key_plot]['x_range']=[7250.,7280.]   #ANTARESS I, tell
        #     plot_settings[key_plot]['x_range']=[6321.5,6324.5]  #ANTARESS I, cosm
        #     plot_settings[key_plot]['x_range']=[5985.5,5989.5]      #ANTARESS I, false cosm

        if gen_dic['star_name']=='WASP107':
            plot_settings[key_plot]['x_range']=[5575.,5590.] 
            plot_settings[key_plot]['x_range']=[6570.,6571.]
            plot_settings[key_plot]['x_range']=[6590.,6600.]
            plot_settings[key_plot]['x_range']=[7278.3,7279.5]
            plot_settings[key_plot]['x_range']=[7752.,7755.]
            plot_settings[key_plot]['x_range']=[7915.,7918.]
            plot_settings[key_plot]['x_range']=[7966.,7969.]
            plot_settings[key_plot]['x_range']=[8285.,8305.]
            plot_settings[key_plot]['x_range']=[8298.7,8302.]
            plot_settings[key_plot]['x_range']=[8346.5,8348.]
            plot_settings[key_plot]['x_range']=[8400.,8405.]
            plot_settings[key_plot]['x_range']=[8417.,8419.]
            plot_settings[key_plot]['x_range']=[8432.,8434.]
            plot_settings[key_plot]['x_range']=[8454.,8457.]  
            plot_settings[key_plot]['x_range']=[8467.,8469.]   
            plot_settings[key_plot]['x_range']=[8507.,8509.]    
            plot_settings[key_plot]['x_range']=[8761.4,8762.]          
            plot_settings[key_plot]['x_range']=[8764.,8766.]         
            plot_settings[key_plot]['x_range']=[8769.,8771.5]        
            plot_settings[key_plot]['x_range']=[8793.5,8794.5]  
            plot_settings[key_plot]['x_range']=[9307.,9313.]               
            plot_settings[key_plot]['x_range']=None            

        elif gen_dic['star_name']=='HAT_P11':
            plot_settings[key_plot]['x_range']=[5578.,5580.5]  
            plot_settings[key_plot]['x_range']=[6569.,6571.]           
            plot_settings[key_plot]['x_range']=None  

        # elif gen_dic['star_name']=='WASP156':
        #     plot_settings[key_plot]['x_range']=[9376.,9379.]  
            

        #Plot boundaries in flux
        if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['y_range']=[1e3,19e3]     #ANTARESS I, gain
            plot_settings[key_plot]['y_range']=[5.4e5,9.7e5]     #ANTARESS I, tell
        #     plot_settings[key_plot]['y_range']=[2000.,12000.]     #ANTARESS I, cosm
        #     plot_settings[key_plot]['y_range']=[1500.,5700.]     #ANTARESS I, false cosm
        if gen_dic['star_name']=='WASP156': 
            # plot_settings[key_plot]['x_range']=[8420.,8564.]    #ANTARESS I, persistent peaks plot 1  
            # plot_settings[key_plot]['y_range']=[0.,2.5]                     
            # plot_settings[key_plot]['x_range']=[8466.,8470.]    #ANTARESS I, persistent peaks plot 2
            # plot_settings[key_plot]['y_range']=[0.6,2.4]    
            plot_settings[key_plot]['x_range']=[5578,5580.]    #ANTARESS I, persistent peaks plot 3
            plot_settings[key_plot]['y_range']=[0.5,14]                
            

        if gen_dic['star_name']=='HD209458': 
            plot_settings[key_plot]['x_range']=[5870.,5915.]    #sodium







    ##################################################################################################
    #%% Individual disk-integrated profiles
    #    - in their original rest frame, with their model fit
    ##################################################################################################
    if ((plot_dic['DI_prof']!='') or (plot_dic['DI_prof_res']!='')):
        for key_plot in ['DI_prof','DI_prof_res']:
            if plot_dic[key_plot]!='':
                plot_settings[key_plot]={} 

                #Margins
                plot_settings[key_plot]['margins']=[0.15,0.15,0.75,0.85]  
                plot_settings[key_plot]['margins']=[0.15,0.15,0.95,0.9]   #ANTARESS I, mock, multi-tr

                #Font size
                plot_settings[key_plot]['font_size']=18   #ANTARESS I, mock, multi-tr

                #Linewidth
                plot_settings[key_plot]['lw_plot']=1.   #ANTARESS I, mock, multi-tr
            
                #Visits to plot
                if gen_dic['studied_pl']=='55Cnc_e':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-02-05']}            
                elif gen_dic['studied_pl']=='WASP121b':
                    plot_settings[key_plot]['visits_to_plot']={           
        #                'HARPS':['14-01-18','09-01-18','31-12-17'],   
                        'HARPS':['14-01-18'], 
                    }  
                    plot_settings[key_plot]['visits_to_plot']={'ESPRESSO_MR':['2018-12-01'],'ESPRESSO':['2019-01-07']}   
                elif gen_dic['studied_pl']=='Kelt9b':plot_settings[key_plot]['visits_to_plot']={'HARPN':['31-07-2017']}  
                elif gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03','binned']}  
                elif gen_dic['studied_pl']=='WASP127b':plot_settings[key_plot]['visits_to_plot']={'HARPS':['2017-03-20','2018-03-31','2018-02-13','2017-02-28']}        
                elif gen_dic['studied_pl']=='HD209458b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-07-20','2019-09-11']}  
                elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']}
                elif gen_dic['studied_pl']==['HD3167_c']:plot_settings[key_plot]['visits_to_plot']={'HARPN':['2016-10-01']} 
                elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-20']} 
                elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-03-18']} 
                elif gen_dic['studied_pl']==['GJ436_b']:plot_settings[key_plot]['visits_to_plot']['ESPRESSO']=['binned']
                
                
                #Bornes du plot en RV  
                if gen_dic['studied_pl']=='WASP_8b':
                    plot_settings[key_plot]['x_range']=[-23.,20.] 
                elif gen_dic['star_name']=='GJ436':
                    plot_settings[key_plot]['x_range']=[-35.,55.]     #GJ436b - plage complete            
                    # plot_settings[key_plot]['x_range']=[-16.3,35.7]   #GJ436b - plage du continu
                elif gen_dic['star_name']=='55Cnc':
                    plot_settings[key_plot]['x_range']=[6.,50.] 
                    plot_settings[key_plot]['x_range']=[27.4-85.,27.4+85.]    
                    plot_settings[key_plot]['x_range']=[-90.,90.]           
                elif gen_dic['studied_pl']=='WASP121b':
                    plot_settings[key_plot]['x_range']=[-15,90.]                 
                    plot_settings[key_plot]['x_range']=[-60,140.]    #mask F
                    plot_settings[key_plot]['x_range']=[-59.7,136.7]    #plot papier
                elif gen_dic['studied_pl']=='Kelt9b':
                    plot_settings[key_plot]['x_range']=[-300.,300.]   
                elif gen_dic['studied_pl']=='WASP76b':
                    plot_settings[key_plot]['x_range']=[-150.,150.] 
                elif gen_dic['studied_pl']=='WASP127b':plot_settings[key_plot]['x_range']=[-32.,12.] 
                elif gen_dic['star_name']=='HD209458':
                    plot_settings[key_plot]['x_range']=[-40.,20.] 
                    plot_settings[key_plot]['x_range']=[5889.2,5890.7]   #ANTARESS I, mock, multi-tr                     
                    
                elif gen_dic['studied_pl']==['HD3167_b']:
                    plot_settings[key_plot]['x_range']=[-80.,120.] 
                    plot_settings[key_plot]['x_range']=[19.4-260.,19.4+260.] 
                    plot_settings[key_plot]['x_range']=[-80.,80.] 
                elif gen_dic['studied_pl']==['HD3167_c']:
                    plot_settings[key_plot]['x_range']=[19.5-70.,19.5+70.]                 
                    # plot_settings[key_plot]['x_range']=[19.5-260.,19.5+260.] 
                    plot_settings[key_plot]['x_range']=[-72.,72.] 
                elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['x_range']=[10.,53.] 
                elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['x_range']=[-90.,-48.] 
                elif gen_dic['star_name']=='GJ9827': plot_settings[key_plot]['x_range']=[10.,54.] 
                elif gen_dic['star_name']=='TOI858':plot_settings[key_plot]['x_range']=[33.,98.]      #CORALIE        
                elif gen_dic['star_name']=='HD189733':plot_settings[key_plot]['x_range']=[-50.,50.]               
        
                #Plot continuum pixels specific to each exposure 
                plot_settings[key_plot]['plot_cont_exp']=True & False  
                plot_settings[key_plot]['plot_cont']=True 
                
                #Shade area not included in fit
                plot_settings[key_plot]['plot_nofit']=True   
        
        #---------------------------------
        #Raw profiles
        if (plot_dic['DI_prof']!=''):  
            key_plot='DI_prof'
        
            #Bornes du plot
            if gen_dic['star_name']=='HD209458':
                plot_settings[key_plot]['y_range']=[0.89,1.06]    #ANTARESS I, mock, multi-tr
                
            elif gen_dic['studied_pl']=='WASP_8b':     
                plot_settings[key_plot]['y_range']=[0.4,1.1]
            elif gen_dic['star_name']=='GJ436':
                plot_settings[key_plot]['y_range']=[0.70,1.15]   #HARPS/HARPN
                plot_settings[key_plot]['y_range']=[0.60,1.15]   #ESPRESSO
            elif gen_dic['studied_pl']=='55Cnc_e':
                plot_settings[key_plot]['y_range']=[0.35,1.05]
                plot_settings[key_plot]['y_range']=[0.25,1.05]   #ESPRESSO
                                
            elif gen_dic['studied_pl']==['HD3167_c']:
                plot_settings[key_plot]['y_range']=[0.4,1.05]
                plot_settings[key_plot]['y_range']=[0.32,1.05]                
            elif gen_dic['studied_pl']=='WASP121b':
                plot_settings[key_plot]['y_range']=[0.8,1.03]            
                plot_settings[key_plot]['y_range']=[0.7,1.1] 
            elif gen_dic['studied_pl']=='Kelt9b':
                plot_settings[key_plot]['y_range']=[0.96,1.02]  
            elif gen_dic['studied_pl']=='WASP76b':
                plot_settings[key_plot]['y_range']=[0.35,1.05] 
            elif gen_dic['studied_pl']=='WASP127b':plot_settings[key_plot]['y_range']=[0.5,1.05]    
            elif gen_dic['studied_pl']==['HD3167_b']:
                plot_settings[key_plot]['y_range']=[0.25,1.05]      
                plot_settings[key_plot]['y_range']=[0.4,1.05]
            elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['y_range']=[0.2,1.2] 
            elif gen_dic['studied_pl']=='GJ9827d':
                plot_settings[key_plot]['y_range']=[0.3,1.1]      #ESPRESSO
                plot_settings[key_plot]['y_range']=[0.5,1.1]      #HARPS
            elif gen_dic['studied_pl']=='GJ9827b':
                plot_settings[key_plot]['y_range']=[0.5,1.1]      #HARPS
            elif gen_dic['studied_pl']==['TOI858b']:
                plot_settings[key_plot]['pl_plot']='TOI858b'
                plot_settings[key_plot]['y_range']=[0.55,1.05]      #CORALIE          

            #Plot fitted line profile
            plot_settings[key_plot]['plot_line_model']=True & False
            if gen_dic['star_name'] in ['HD189733','WASP43','L98_59','GJ1214']:plot_settings[key_plot]['plot_line_model']=True 

            #Print fit properties on plot
            plot_settings[key_plot]['plot_prop']= False  
            if gen_dic['star_name'] in ['HD189733','WASP43','L98_59','GJ1214']:plot_settings[key_plot]['plot_prop']=True             
        
        #---------------------------------
        #Residuals profiles
        if (plot_dic['DI_prof_res']!='') and (gen_dic['fit_DI']):     
            key_plot='DI_prof_res'

            #Bornes du plot en RV 
            if gen_dic['studied_pl']=='WASP_8b':     
                plot_settings[key_plot]['y_range']=[0.4,1.1]
            if gen_dic['star_name']=='GJ436':          
                plot_settings[key_plot]['y_range']=[-1e-2,1e-2]  #None
                plot_settings[key_plot]['y_range']=[-3e-2,7e-2]  #None            
                plot_settings[key_plot]['y_range']=[-2.5e-2,2.5e-2]            
            if gen_dic['studied_pl']=='55Cnc_e':
                plot_settings[key_plot]['y_range']=[-2.5e-2,2.5e-2]
            if gen_dic['studied_pl']==['HD3167_c']:
                plot_settings[key_plot]['y_range']=[-3.5e-2,2.8e-2]               
            if gen_dic['studied_pl']=='WASP121b':
                plot_settings[key_plot]['y_range']=[-2e-2,2e-2]                  
                plot_settings[key_plot]['y_range']=[-1.5e-2,1.5e-2]         
            elif gen_dic['studied_pl']=='HD209458b':plot_settings[key_plot]['y_range']=[-1.5e-2,1.5e-2]    
            elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['y_range']=[-2.5e-2,2.5e-2]  
            elif gen_dic['star_name']=='GJ9827': 
                plot_settings[key_plot]['y_range']=[-0.015,0.015]      #HARPS
            elif gen_dic['studied_pl']==['TOI858b']:plot_settings[key_plot]['y_range']=[-1.5e-2,1.5e-2]      #CORALIE  











    ##################################################################################################
    #%% Individual disk-integrated transmission spectra
    #    - in the star rest frame, offset or not by the systemic velocity
    ##################################################################################################
    if gen_dic['specINtype'] and (plot_dic['trans_sp']!=''):
        key_plot = 'trans_sp'
        plot_settings[key_plot]={}
        
        #Margins
        # plot_settings[key_plot]['margins'] = ...
        
        #Figure size
        plot_settings[key_plot]['fig_size'] = (10,6)
        plot_settings[key_plot]['fig_size'] = (30,6)
        # plot_settings[key_plot]['fig_size'] = (40,6)
        
        #Visits to plot    
        # if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20180902']}

        #Path to wiggle model
        #    - leave empty to use last result from 'wig_vis_fit'
        plot_settings[key_plot]['wig_path_corr'] = {}    

        #Plot as a function of nu = c/w
        plot_settings[key_plot]['sp_var'] = 'nu'     
        # plot_settings[key_plot]['sp_var'] = 'wav'    


        #Data bin size
        #    - in 'sp_var' units
        plot_settings[key_plot]['bin_width'] = 0.0166*2
        plot_settings[key_plot]['bin_width'] = 0.2     #nu
        # plot_settings[key_plot]['bin_width'] = 0.08     #wav
        # plot_settings[key_plot]['bin_width'] = 25.     #wav



        #Horizontal range for the plot
        # plot_settings[key_plot]['x_range'] = [c_light/5700.,c_light/5500.]
        # plot_settings[key_plot]['x_range'] = [c_light/6300.,c_light/5700.]
        plot_settings[key_plot]['x_range'] = [45.,55.]
        plot_settings[key_plot]['x_range'] = [38.,64.]
        # plot_settings[key_plot]['x_range'] = [48.,50.]        
        # plot_settings[key_plot]['x_range'] = [5883.,5902.]
        # plot_settings[key_plot]['x_range'] = [c_light/53.,c_light/49.]    #bump de Fbal dans region Na, HD209    
        
        #Vertical range
        # plot_settings[key_plot]['y_range'] = [0.98,1.02]

        #Plot dispersion over exposures
        plot_settings[key_plot]['plot_disp'] = True    #   & False 

        #Vertical range for dispersion plot (in ppm)
        if gen_dic['star_name']=='HD209458':plot_settings[key_plot]['y_range_disp'] =  {'ESPRESSO':{'20190720': [200.,1500.],'20190911': [0.,1500.]}}
        if gen_dic['star_name']=='WASP76':plot_settings[key_plot]['y_range_disp'] =  {'ESPRESSO':{'20180902': [200.,3000.],'20181030': [0.,3000.]}}

        #Choice of orders to plot
        #    - leave empty to plot all orders
        # plot_settings[key_plot]['orders_to_plot']=[116,117,118,119,120,121]       

        #Plot order indexes
        plot_settings[key_plot]['plot_idx_ord'] = True

        #Exposures to plot
        plot_settings[key_plot]['iexp_plot'] = {}
        # plot_settings[key_plot]['iexp_plot'] = {'ESPRESSO':{'20190720':[6,81]}}

        #Plot raw data and errors
        plot_settings[key_plot]['plot_data'] = True & False
        plot_settings[key_plot]['plot_err'] = True   & False

        #Plot binned data and errors
        plot_settings[key_plot]['plot_bin'] = True #& False
        plot_settings[key_plot]['plot_bin_err'] = True   & False

        #Force order level to unity
        plot_settings[key_plot]['force_unity'] = True & False

        #Select plot mode
        plot_settings[key_plot]['drawstyle']='steps-mid'
        # plot_settings[key_plot]['drawstyle']='default'

        #Offset between uncorrected / corrected data
        plot_settings[key_plot]['gap_exp'] = 0.02

        #Exposures used in master calculation
        #    - out-of-transit exposures if left undefined
        #    - set to list, or to 'all'
        if gen_dic['star_name']=='HD209458':
            plot_settings[key_plot]['iexp_mast_list'] = {'ESPRESSO':{'20190720':'all','20190911':'all'}}  
        #     # plot_settings[key_plot]['iexp_mast_list'] = {'ESPRESSO':{'20190720':[0,1],'20190911':[0,1]}}      
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['iexp_mast_list'] = {'ESPRESSO':{'20180902':'all','20181030':'all'}}       



        #Plot spectra at two chosen steps of the correction process
        #    - set to None, or chose amongst:
        # + 'raw' : before any correction
        # + 'all' : after all requested corrections
        # + 'tell' : after telluric correction 
        # + 'count' : after flux-to-count scaling
        # + 'fbal' : after flux balance correction 
        # + 'cosm' : after cosmics correction  
        # + 'permpeak' : after persistent peak correction 
        # + 'wiggle' : after ESPRESSO wiggle correction
        #              calculations are performed here as ratios are not calculated within the routine after correction   
        #              for speed and simplicity in the plot we resample all spectra only once on their common table

        plot_settings[key_plot]['plot_pre']='tell'
        plot_settings[key_plot]['plot_post']='fbal'

        # plot_settings[key_plot]['plot_pre']='raw'     #ANTARESS I, tell   
        # plot_settings[key_plot]['plot_post']='tell'

        # plot_settings[key_plot]['plot_pre']='tell'   #ANTARESS I, gain
        # plot_settings[key_plot]['plot_post']='count'

        # plot_settings[key_plot]['plot_pre']='count'   #ANTARESS I, global fbal
        # plot_settings[key_plot]['plot_post']='fbal'

        # plot_settings[key_plot]['plot_pre']='fbal'   #ANTARESS I, cosmic corr
        # plot_settings[key_plot]['plot_post']='cosm'

        # plot_settings[key_plot]['plot_pre']=None    
        # plot_settings[key_plot]['plot_post']='count'
        
        # plot_settings[key_plot]['plot_pre']='count'    
        # plot_settings[key_plot]['plot_post']='cosm'

        # plot_settings[key_plot]['plot_pre']=None    
        # plot_settings[key_plot]['plot_post']='all'

        if gen_dic['star_name']=='WASP156':    #ANTARESS I, persistent peaks  
            plot_settings[key_plot]['plot_pre']='cosm'                             
            plot_settings[key_plot]['plot_post']='permpeak' 
            # plot_settings[key_plot]['color_dic']={'CARMENES_VIS':{'20191025':'red'}} #ANTARESS I, persistent peaks   1
            # plot_settings[key_plot]['color_dic_sec']={'CARMENES_VIS':{'20191025':'dodgerblue'}}

       # plot_settings[key_plot]['plot_pre']='cosm'    
       # plot_settings[key_plot]['plot_post']='permpeak'     
        
        # plot_settings[key_plot]['plot_pre']='fbal'    
        # plot_settings[key_plot]['plot_post']='permpeak'  


        # plot_settings[key_plot]['color_dic']={'ESPRESSO':{'20190720':'dodgerblue','20190911':'dodgerblue'}} 
        # plot_settings[key_plot]['color_dic_sec']={'ESPRESSO':{'20190720':'red','20190911':'red'}}
        # plot_settings[key_plot]['color_dic_bin']={'ESPRESSO':{'20190720':'dodgerblue','20190911':'dodgerblue'}} 
        # plot_settings[key_plot]['color_dic_bin_sec']={'ESPRESSO':{'20190720':'red','20190911':'red'}}



    















    ##################################################################################################
    #%% Individual binned disk-integrated spectral profiles
    ##################################################################################################
    if (any('spec' in s for s in data_dic['DI']['type'].values())) and (plot_dic['sp_DIbin']!=''):
        key_plot = 'sp_DIbin'
        plot_settings[key_plot]={} 

        #Scaling factor (in power of ten, ie flux are multiplied by 10**sc_fact10)
        # plot_settings[key_plot]['sc_fact10']=-5.
        
        #Errors
        plot_settings[key_plot]['plot_err']=False

        #Instruments and visits to plot
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}   
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['binned']}   
        if gen_dic['star_name']=='HD209458':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['binned']}
            
        #Colors
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'}} 
            
        #Plot boundaries in wav
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['x_range']=[3500.,8000.] 
            plot_settings[key_plot]['x_range']=[5880.,5905.] 
            plot_settings[key_plot]['x_range']=[6850.,6950.] 
            # plot_settings[key_plot]['x_range']=None

        #Plot boundaries in flux
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['y_range']=[0.,2.] 



    ##################################################################################################
    #%% Individual binned disk-integrated CCF profiles
    ##################################################################################################
    if ('CCF' in data_dic['DI']['type'].values()) and ((plot_dic['CCF_DIbin']!='') or (plot_dic['CCF_DIbin_res']!='')):
        for key_plot in ['CCF_DIbin','CCF_DIbin_res']:
            if plot_dic[key_plot]!='':
                plot_settings[key_plot]={} 

                #Visits to plot
                if gen_dic['studied_pl']==['HD3167_c']:plot_settings[key_plot]['visits_to_plot']={'HARPN':['2016-10-01']}  
                elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']} 
                # elif gen_dic['studied_pl']==['GJ436_b']:plot_settings[key_plot]['visits_to_plot']['ESPRESSO']=['binned'] 
                elif gen_dic['star_name']=='WASP107':plot_settings[key_plot]['visits_to_plot']['HARPS']+=['binned'] 
                elif gen_dic['star_name']=='WASP166':plot_settings[key_plot]['visits_to_plot']['HARPS']+=['binned'] 
                elif gen_dic['star_name']=='HD106315':plot_settings[key_plot]['visits_to_plot']['HARPS']+=['binned'] 
                elif gen_dic['star_name']=='HAT_P11':
                    if 'HARPN' in plot_settings[key_plot]['visits_to_plot']:plot_settings[key_plot]['visits_to_plot']['HARPN']+=['binned'] 
                    if 'CARMENES_VIS' in plot_settings[key_plot]['visits_to_plot']:plot_settings[key_plot]['visits_to_plot']['CARMENES_VIS']+=['binned'] 
                elif gen_dic['star_name']=='55Cnc':
                    if 'ESPRESSO' in plot_settings[key_plot]['visits_to_plot']:plot_settings[key_plot]['visits_to_plot']['ESPRESSO']+=['binned']      
        
        
                #Color dictionary
                if gen_dic['studied_pl']==['HD3167_c']:
                    plot_settings[key_plot]['color_dic']={'HARPN':{'2016-10-01':'dodgerblue'}}
                elif gen_dic['studied_pl']==['HD3167_b']:
                    plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-10-09':'dodgerblue'}}
        
        
                #Bornes du plot en RV
                if gen_dic['studied_pl']==['HD3167_c']:
                    plot_settings[key_plot]['x_range']=[-101.,101.]  
                    # plot_settings[key_plot]['x_range']=[-31.,31.]  
                elif gen_dic['studied_pl']==['HD3167_b']:
                    plot_settings[key_plot]['x_range']=np.array([-71.,71.])+19.4 
                    # plot_settings[key_plot]['x_range']=[-100.,420.] 
                    # plot_settings[key_plot]['x_range']=[-420.,100.] 
                    # plot_settings[key_plot]['x_range']=np.array([-31.,31.])
                    # plot_settings[key_plot]['x_range']=np.array([-31.,31.])-160.
                    # plot_settings[key_plot]['x_range']=np.array([-31.,31.])+160.
                elif gen_dic['studied_pl']==['GJ436_b']:
                    plot_settings[key_plot]['x_range']=np.array([-45.,45.])+9.7 
                    plot_settings[key_plot]['x_range']=np.array([-30.,30.])+9.7    #papier
                    if plot_settings[key_plot]['visits_to_plot']['ESPRESSO']==['binned']:plot_settings[key_plot]['x_range']=np.array([-45.,45.])   
                elif gen_dic['star_name']=='MASCARA1':
                    plot_settings[key_plot]['x_range']=[-300.,320.]              
                if gen_dic['star_name']=='55Cnc':
                    plot_settings[key_plot]['x_range']=[-80.,80.]    #ESPRESSO          
                    plot_settings[key_plot]['x_range']=[-30.,30.]    #Zoom    
                if gen_dic['star_name']=='GJ1214':
                    plot_settings[key_plot]['x_range']=[-5.,17.]    #Zoom   

                #Plot continuum pixels specific to each exposure 
                plot_settings[key_plot]['plot_cont']=True 
                
                #Shade area not included in fit
                plot_settings[key_plot]['plot_nofit']=True   

                #Overplot fit
                plot_settings[key_plot]['plot_line_model']=True  


        #-----------------------------------
        #Plot each disk-integrated CCF and its fit
        if (plot_dic['CCF_DIbin']!=''):
            key_plot='CCF_DIbin'
            
            #Bornes du plot
            #    - true fluxes, before scaling factor  
            if gen_dic['studied_pl']==['HD3167_c']:
                plot_settings[key_plot]['y_range']=[0.4,1.1]   
            elif gen_dic['studied_pl']==['HD3167_b']: 
                plot_settings[key_plot]['y_range']=[0.2,1.1]    
            elif gen_dic['star_name']=='MASCARA1':                
                plot_settings[key_plot]['y_range']=[0.945,1.03] 
                # plot_settings[key_plot]['y_range']=[0.9,1.06]                 
                plot_settings[key_plot]['y_range']=[0.94,1.08]        
                plot_settings[key_plot]['y_range']=[0.95,1.1]             
            
        #-----------------------------------
        #Plot each disk-integrated CCF and its fit
        if (plot_dic['CCF_DIbin_res']!='') and (gen_dic['fit_DIbin']): 
            key_plot='CCF_DIbin_res'         
            
            #Bornes du plot en RV 
            if gen_dic['studied_pl']==['HD3167_c']:
                plot_settings[key_plot]['y_range']=[-1.5e-2,1.5e-2]          
        
        
        
        
        
        
    ##################################################################################################
    #%% Individual 1D disk-integrated profiles
    ##################################################################################################
    if (any('spec' in s for s in data_dic['DI']['type'].values())) and (plot_dic['sp_DI_1D']!=''):
        key_plot = 'sp_DI_1D' 
        plot_settings[key_plot]={} 

 
        
        #Scaling factor (in power of ten, ie flux are multiplied by 10**sc_fact10)
        # plot_settings[key_plot]['sc_fact10']=-3.

        #Instruments and visits to plot
        # if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}            

        #Colors
        # if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'}} 
            
        #Plot boundaries in wav
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['x_range']=[3500.,8000.] 
            plot_settings[key_plot]['x_range']=[5880.,5905.] 
#            plot_settings[key_plot]['x_range']=[6200.,6300.] 
            # plot_settings[key_plot]['x_range']=None

        #Plot boundaries in signal
        if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['y_range']=[0.,6000.] 

            plot_settings[key_plot]['y_range']=[0.,1.5]             
        #     plot_settings[key_plot]['norm_prof']=True
        
        
        
        
        
        
        
        
        
    ##################################################################################################
    #%% Stellar CCF mask: spectrum
    #    - plotting spectrum used for CCF mask generation and associated properties
    ##################################################################################################
    if ((plot_dic['DImask_spectra']!='') or (plot_dic['Intrmask_spectra']!='')):
        for key_plot in ['DImask_spectra','Intrmask_spectra']:
            if plot_dic[key_plot]!='':
                plot_settings[key_plot]={}  
                
                #Rasterized
                plot_settings[key_plot]['rasterized']=True 

                #Instrument to plot
                #    - mask is always built from all available visits
                # plot_settings[key_plot]['inst_to_plot']=[]

                #Plot master spectrum and mask at chosen step
                # + 'cont': continuum-normalization
                # + 'sel1': line selection with depth and width criteria
                # + 'sel2': line selection with delta-position criterion
                # + 'sel3': line selection with telluric contamination
                # + 'sel4': line selection with morphological clipping (delta-maxima/line depth and asymetry parameter)
                # + 'sel5': line selection with morphological clipping (depth and width criteria)
                # + 'sel6': line selection with RV dispersion and telluric contamination
                plot_settings[key_plot]['step']='sel1'           
        
                #Overplot resampled spectra
                if plot_settings[key_plot]['step']=='cont': 
                    if gen_dic['star_name']=='HD209458':        
                        plot_settings[key_plot]['resample'] = 0.2#2.    
                        plot_settings[key_plot]['alpha_symb'] = 0.2 
        
                #Print number of lines selected in step
                plot_settings[key_plot]['print_nl']=True                 

                #Plot exclusion ranges
                plot_settings[key_plot]['line_rej_range']=True 

                #Plot minimum telluric depth to be considered
                plot_settings[key_plot]['tell_depth_min'] = False

                #Print VALD species in final plot 
                plot_settings[key_plot]['vald_sp']=True  #  & False 

                #Plot line ranges in final plot 
                plot_settings[key_plot]['line_ranges']=True  #  & False 
        
                #Spectral range
                # plot_settings[key_plot]['x_range']=[3700.,4000.]   
                # plot_settings[key_plot]['x_range']=[6250.,6350.]   #tell range   
                # plot_settings[key_plot]['x_range']=[6850.,6980.]   #excl range 
                # plot_settings[key_plot]['x_range']=[7100.,7400.]   #excl range 
                # plot_settings[key_plot]['x_range']=[7670.,7800.]   #tell range  
                # plot_settings[key_plot]['x_range']=[6200.,6500.]   #tell exc

                
                # plot_settings[key_plot]['x_range']=[5500.,5700.]    
                # plot_settings[key_plot]['x_range']=[7200.,7300.] 

                #Vertical range
                plot_settings[key_plot]['y_range']=None
                
                
                #Marker size
                plot_settings[key_plot]['markersize'] = 3.                
                
                #ANTARES I
                if gen_dic['star_name']=='HD209458':
                    plot_settings[key_plot]['rasterized'] = False  
                    if plot_settings[key_plot]['step']=='cont':                    
                        plot_settings[key_plot]['plot_norm'] = False
                        # plot_settings[key_plot]['plot_norm_reg'] = False
                        
                           
                #         # plot_settings[key_plot]['y_range']=[-2.5,3.05]    
                #         plot_settings[key_plot]['y_range']=[-0.15,1.17]    
                #         # plot_settings[key_plot]['x_range']=[3770.,4500.] 
                #         # plot_settings[key_plot]['x_range']=[4220.,4230.] 
                #         plot_settings[key_plot]['x_range']=[3920.,3980.] 
                #         plot_settings[key_plot]['x_range']=[3944.,3948.] 
                #         plot_settings[key_plot]['x_range']=[7500.,7900.] 

                    # plot_settings[key_plot]['x_range']=[3770.,7890.] 
                    # plot_settings[key_plot]['y_range']=[0.,1.5] 
                    # plot_settings[key_plot]['y_range']=[-2.,3.] 

                    plot_settings[key_plot]['x_range']=[4430.,4450.]   
                    plot_settings[key_plot]['y_range']=[0.,1.5] 


                        
                        
                if gen_dic['star_name']=='WASP76':
                    plot_settings[key_plot]['markersize'] = 5.  
                    if plot_settings[key_plot]['step']=='cont':
                        plot_settings[key_plot]['x_range']=[3770.,7890.] 
                        plot_settings[key_plot]['y_range']=[0.,1.5] 
                    elif plot_settings[key_plot]['step']=='sel3':
                        plot_settings[key_plot]['x_range']=[7214.+5.5,7234.-5.5] 
                    elif plot_settings[key_plot]['step']=='sel6':
                        # plot_settings[key_plot]['x_range']=[3770.,7890.] 
                        # plot_settings[key_plot]['y_range']=[0.,1.05] 
                        # plot_settings[key_plot]['print_nl']=False 
                        plot_settings[key_plot]['x_range']=[4430.,4450.]     
                        plot_settings[key_plot]['y_range']=[0.2,1.1]                     

        
        #---------------------------------
        #Disk-integrated profiles
        if (plot_dic['DImask_spectra']!=''):  
            key_plot='DImask_spectra'        
        
        
        #---------------------------------
        #Intrinsic profiles
        if (plot_dic['Intrmask_spectra']!=''):  
            key_plot='Intrmask_spectra'


    ##################################################################################################
    #%% Stellar CCF mask: line depth range selection
    ##################################################################################################
    if ((plot_dic['DImask_ld']!='') or (plot_dic['Intrmask_ld']!='')):
        for key_plot in ['DImask_ld','Intrmask_ld']:
            if plot_dic[key_plot]!='':
                plot_settings[key_plot]={}         

                #Figure size
                plot_settings[key_plot]['fig_size'] = [6,6]
                
                #Linewidth
                plot_settings[key_plot]['lw_plot'] = 1.

                #Instrument to plot
                #    - mask is always built from all available visits
                # plot_settings[key_plot]['inst_to_plot']=[]

                #Display the following distribution information:
                #    - 'hist' : histogram of line number
                #    - 'cum_w' : cumulative of line weights (normalized)
                plot_settings[key_plot]['dist_info'] = 'cum_w' 
                
                #Width range
                plot_settings[key_plot]['x_range']=None

                #Depth range
                plot_settings[key_plot]['y_range']=None
                
                #Marker size
                plot_settings[key_plot]['markersize'] = 3.

                #Test threshold on line depth range
                #    - leave undefined to prevent 
                plot_settings[key_plot]['linedepth_cont_min'] =  0.05   
                plot_settings[key_plot]['linedepth_cont_max'] = 0.98  
                plot_settings[key_plot]['linedepth_min'] = 0.01 
        
                #Number of bins in histograms
                plot_settings[key_plot]['x_bins_par'] = 100
                plot_settings[key_plot]['y_bins_par'] = 100
                
                #Log scales
                plot_settings[key_plot]['x_log_hist'] = True
                plot_settings[key_plot]['y_log_hist'] = True                
        
        #---------------------------------
        #Disk-integrated profiles
        if (plot_dic['DImask_ld']!=''):  
            key_plot='DImask_ld'        
        
        
        #---------------------------------
        #Intrinsic profiles
        if (plot_dic['Intrmask_ld']!=''):  
            key_plot='Intrmask_ld'



    ##################################################################################################
    #%% Stellar CCF mask: line depth and width selection
    ##################################################################################################
    if ((plot_dic['DImask_ld_lw']!='') or (plot_dic['Intrmask_ld_lw']!='')):
        for key_plot in ['DImask_ld_lw','Intrmask_ld_lw']:
            if plot_dic[key_plot]!='':
                plot_settings[key_plot]={}         

                #Figure size
                plot_settings[key_plot]['fig_size'] = [6,6]
                
                #Linewidth
                plot_settings[key_plot]['lw_plot'] = 1.

                #Instrument to plot
                #    - mask is always built from all available visits
                # plot_settings[key_plot]['inst_to_plot']=[]

                #Display the following distribution information:
                #    - 'hist' : histogram of line number
                #    - 'cum_w' : cumulative of line weights (normalized)
                plot_settings[key_plot]['dist_info'] = 'hist' 

                #Width range
                plot_settings[key_plot]['x_range']=None

                #Depth range
                plot_settings[key_plot]['y_range']=None
                
                #Marker size
                plot_settings[key_plot]['markersize'] = 3.

                # #Test threshold on minimum line depth and half-width to be kept (value > 10^(crit)) 
                plot_settings[key_plot]['line_width_logmin'] = -1.6
                plot_settings[key_plot]['line_depth_logmin'] = -2.5
        
                #Number of bins in histograms
                plot_settings[key_plot]['x_bins_par'] = 50
                plot_settings[key_plot]['y_bins_par'] = 60        
        
        #---------------------------------
        #Disk-integrated profiles
        if (plot_dic['DImask_ld_lw']!=''):  
            key_plot='DImask_ld_lw'        
        
        
        #---------------------------------
        #Intrinsic profiles
        if (plot_dic['Intrmask_ld_lw']!=''):  
            key_plot='Intrmask_ld_lw'





    ##################################################################################################
    #%% Stellar CCF mask: line position selection
    ##################################################################################################
    if ((plot_dic['DImask_RVdev_fit']!='') or (plot_dic['Intrmask_RVdev_fit']!='')):
        for key_plot in ['DImask_RVdev_fit','Intrmask_RVdev_fit']:
            if plot_dic[key_plot]!='':
                plot_settings[key_plot]={}         

                #Figure size
                plot_settings[key_plot]['fig_size'] = [6,6]

                #Linewidth
                plot_settings[key_plot]['lw_plot'] = 1.

                #Instrument to plot
                #    - mask is always built from all available visits
                # plot_settings[key_plot]['inst_to_plot']=[]

                #Display the following distribution information:
                #    - 'hist' : histogram of line number
                #    - 'cum_w' : cumulative of line weights (normalized)
                plot_settings[key_plot]['dist_info'] = 'cum_w' 

                #RV deviation range
                plot_settings[key_plot]['x_range']=None

                #Test threshold on RV deviation in line position
                plot_settings[key_plot]['abs_RVdev_fit_max'] = 500.
        
                #Number of bins in histograms
                plot_settings[key_plot]['x_bins_par'] = 60         
        
        #---------------------------------
        #Disk-integrated profiles
        if (plot_dic['DImask_RVdev_fit']!=''):  
            key_plot='DImask_RVdev_fit'        
        
        
        #---------------------------------
        #Intrinsic profiles
        if (plot_dic['Intrmask_RVdev_fit']!=''):  
            key_plot='Intrmask_RVdev_fit'






    ##################################################################################################
    #%% Stellar CCF mask: telluric selection
    ##################################################################################################
    if ((plot_dic['DImask_tellcont']!='') or (plot_dic['Intrmask_tellcont']!='')):
        for key_plot in ['DImask_tellcont','Intrmask_tellcont']:
            if plot_dic[key_plot]!='':
                plot_settings[key_plot]={}         

                #Figure size
                plot_settings[key_plot]['fig_size'] = [6,6]
                plot_settings[key_plot]['margins'] = [0.2,0.2,0.9,0.9] 

                #Linewidth
                plot_settings[key_plot]['lw_plot'] = 1.

                #Instrument to plot
                #    - mask is always built from all available visits
                # plot_settings[key_plot]['inst_to_plot']=[]

                #Display the following distribution information:
                #    - 'hist' : histogram of line number
                #    - 'cum_w' : cumulative of line weights (normalized)
                plot_settings[key_plot]['dist_info'] = 'hist' 
                
                #Depth ratio range
                plot_settings[key_plot]['x_range']=None

                #Histogram range
                if plot_settings[key_plot]['dist_info'] == 'cum_w' :
                    plot_settings[key_plot]['x_range_hist']=[0.95,1]

                #Test thresholds on ratio between telluric and stellar line
                plot_settings[key_plot]['tell_star_depthR_max'] = 0.1
                plot_settings[key_plot]['tell_star_depthR_max_final'] = 0.1                
                
                #Number of bins in histograms
                plot_settings[key_plot]['x_bins_par'] = 80         

                #Log scales
                plot_settings[key_plot]['x_log_hist'] = True
        
        #---------------------------------
        #Disk-integrated profiles
        if (plot_dic['DImask_tellcont']!=''):  
            key_plot='DImask_tellcont'        
        
        
        #---------------------------------
        #Intrinsic profiles
        if (plot_dic['Intrmask_tellcont']!=''):  
            key_plot='Intrmask_tellcont'





    ##################################################################################################
    #%% Stellar CCF mask: VALD line depth correction
    ##################################################################################################
    if ((plot_dic['DImask_vald_depthcorr']!='') or (plot_dic['Intrmask_vald_depthcorr']!='')):
        for key_plot in ['DImask_vald_depthcorr','Intrmask_vald_depthcorr']:
            if plot_dic[key_plot]!='':
                plot_settings[key_plot]={}         

                #Figure size
                plot_settings[key_plot]['fig_size'] = (12,6)

                #Linewidth
                plot_settings[key_plot]['lw_plot'] = 1.

                #Instrument to plot
                #    - mask is always built from all available visits
                # plot_settings[key_plot]['inst_to_plot']=[]

                #Species to plot
                #    - leave empty to plot all detected species
                plot_settings[key_plot]['spec_to_plot']=[]

                #Depth ratio range
                plot_settings[key_plot]['x_range']=None


        #---------------------------------
        #Disk-integrated profiles
        if (plot_dic['DImask_vald_depthcorr']!=''):  
            key_plot='DImask_vald_depthcorr'        
        
        
        #---------------------------------
        #Intrinsic profiles
        if (plot_dic['Intrmask_vald_depthcorr']!=''):  
            key_plot='Intrmask_vald_depthcorr'





    ##################################################################################################
    #%% Stellar CCF mask: morphological (asymmetry) selection
    ##################################################################################################
    if ((plot_dic['DImask_morphasym']!='') or (plot_dic['Intrmask_morphasym']!='')):
        for key_plot in ['DImask_morphasym','Intrmask_morphasym']:
            if plot_dic[key_plot]!='':
                plot_settings[key_plot]={}         

                #Figure size
                plot_settings[key_plot]['fig_size'] = [6,6]
                
                #Linewidth
                plot_settings[key_plot]['lw_plot'] = 1.
                
                #Display the following distribution information:
                #    - 'hist' : histogram of line number
                #    - 'cum_w' : cumulative of line weights (normalized)
                plot_settings[key_plot]['dist_info'] = 'cum_w'                

                #Instrument to plot
                #    - mask is always built from all available visits
                # plot_settings[key_plot]['inst_to_plot']=[]

                #Ratio between normalized continuum difference and relative line depth
                plot_settings[key_plot]['x_range']=None

                #Normalized asymetry parameter
                plot_settings[key_plot]['y_range']=None
                
                #Marker size
                plot_settings[key_plot]['markersize'] = 3.

                #Test thresholds (value < crit) 
                plot_settings[key_plot]['diff_cont_rel_max'] = 5. #1.3
                plot_settings[key_plot]['asym_ddflux_max'] = 0.6 #0.3
        
                #Number of bins in histograms
                # plot_settings[key_plot]['x_bins_par'] = 30  #50
                # plot_settings[key_plot]['y_bins_par'] = 40  #50        
        
        #---------------------------------
        #Disk-integrated profiles
        if (plot_dic['DImask_morphasym']!=''):  
            key_plot='DImask_morphasym'        
        
        
        #---------------------------------
        #Intrinsic profiles
        if (plot_dic['Intrmask_morphasym']!=''):  
            key_plot='Intrmask_morphasym'



    ##################################################################################################
    #%% Stellar CCF mask: morphological (shape) selection
    ##################################################################################################
    if ((plot_dic['DImask_morphshape']!='') or (plot_dic['Intrmask_morphshape']!='')):
        for key_plot in ['DImask_morphshape','Intrmask_morphshape']:
            if plot_dic[key_plot]!='':
                plot_settings[key_plot]={}         

                #Figure size
                plot_settings[key_plot]['fig_size'] = [6,6]
                
                #Linewidth
                plot_settings[key_plot]['lw_plot'] = 1.

                #Instrument to plot
                #    - mask is always built from all available visits
                # plot_settings[key_plot]['inst_to_plot']=[]

                #Display the following distribution information:
                #    - 'hist' : histogram of line number
                #    - 'cum_w' : cumulative of line weights (normalized)
                plot_settings[key_plot]['dist_info'] = 'hist'  

                #Width
                plot_settings[key_plot]['x_range']=None

                #Minimum to mean maxima difference
                plot_settings[key_plot]['y_range']=None
                
                #Marker size
                plot_settings[key_plot]['markersize'] = 3.

                #Test thresholds (x value < crit and y value > crit) 
                plot_settings[key_plot]['width_max'] = 15.
                plot_settings[key_plot]['diff_depth_min'] = 0.05
        
                #Number of bins in histograms
                # plot_settings[key_plot]['x_bins_par'] = 30  #50
                # plot_settings[key_plot]['y_bins_par'] = 40  #50        
        
        #---------------------------------
        #Disk-integrated profiles
        if (plot_dic['DImask_morphshape']!=''):  
            key_plot='DImask_morphshape'        
        
        
        #---------------------------------
        #Intrinsic profiles
        if (plot_dic['Intrmask_morphshape']!=''):  
            key_plot='Intrmask_morphshape'

    ##################################################################################################
    #%% Stellar CCF mask: RV dispersion selection
    ##################################################################################################
    if ((plot_dic['DImask_RVdisp']!='') or (plot_dic['Intrmask_RVdisp']!='')):
        for key_plot in ['DImask_RVdisp','Intrmask_RVdisp']:
            if plot_dic[key_plot]!='':
                plot_settings[key_plot]={}         

                #Figure size
                plot_settings[key_plot]['fig_size'] = [6,6]
                plot_settings[key_plot]['margins'] = [0.2,0.2,0.9,0.9]
                
                #Linewidth
                plot_settings[key_plot]['lw_plot'] = 1.

                #Instrument to plot
                #    - mask is always built from all available visits
                # plot_settings[key_plot]['inst_to_plot']=[]

                #Display the following distribution information:
                #    - 'hist' : histogram of line number
                #    - 'cum_w' : cumulative of line weights (normalized)
                plot_settings[key_plot]['dist_info'] = 'hist'  

                #X range
                plot_settings[key_plot]['x_range']=None
                plot_settings[key_plot]['x_range']=[0.,25.]

                #Y range
                plot_settings[key_plot]['y_range']=None
                plot_settings[key_plot]['y_range']=[0.,2.]

                #Histogram ranges
                if plot_settings[key_plot]['dist_info'] == 'cum_w' : 
                    plot_settings[key_plot]['x_range_hist']=[0.9,1]
                
                #Marker size
                plot_settings[key_plot]['markersize'] = 3.

                #Test thresholds (value < crit) 
                # plot_settings[key_plot]['absRV_max'] = 10.
                # plot_settings[key_plot]['RVdisp2err_max'] = 2.
        
                #Number of bins in histograms
                # plot_settings[key_plot]['x_bins_par'] = 30  #50
                # plot_settings[key_plot]['y_bins_par'] = 40  #50        
        
        #---------------------------------
        #Disk-integrated profiles
        if (plot_dic['DImask_RVdisp']!=''):  
            key_plot='DImask_RVdisp'        
        
        
        #---------------------------------
        #Intrinsic profiles
        if (plot_dic['Intrmask_RVdisp']!=''):  
            key_plot='Intrmask_RVdisp'
















        
        
        
        
        
        
        
        
        
        
        
        

    ##################################################################################################
    #%% Properties of raw data and disk-integrated CCFs
    #    - possibility to plot them as a function of many variable to search for correlations
    ##################################################################################################
    if (plot_dic['prop_raw']!=''):
        
        #Choose values to plot in ordina (list of properties) and abscissa
        #    - properties:
        # + 'rv' : centroid of the raw CCFs in heliocentric rest frame (in km/s)
        # + 'FWHM': width of raw CCFs (in km/s)
        # + 'ctrst': contrast of raw CCFs
        # + 'rv_l2c': RV(lobe)-RV(core) of double gaussian components
        # + 'FWHM_l2c': FWHM(lobe)/FWHM(core) of double gaussian components
        # + 'amp_l2c': contrast(lobe)/contrast(core) of double gaussian components
        # + 'RVres' residuals from Keplerian curve (m/s)  
        # + 'RVdrift' : RV drift of the spectrograph, derived from the Fabry-Perot (m/s)
        #    + 'phase' : orbital phase
        #    + 'mu' : mu
        #    + 'lat' : stellar latitude 
        #    + 'lon' : stellar longitude 
        #    + projected position in the stellar frame
        # x (along the equator) : 'x_st'
        # y (along the spin axis) : 'y_st' 
        #    + 'AM' : airmass : 
        #    + 'seeing' : seeing : 
        #    + 'snr' : SNR : 
        # in that case select the indices of orders over which average the SNR
        #    + 'snr_R' : SNR ratio 
        # in that case select the indices of orders over which average the SNR for both numerator and denominator
        #    + 'colcorrmin', 'colcorrmax', 'colcorrR' : min/max color correction coefficients and ratio max/min 
        #      'colcorr450', 'colcorr550', 'colcorr650' : correction coefficients at the corresponding wavelengths (nm)
        #    + 'glob_flux_sc': ratio of total flux in each exposure profile, to their mean value, used to scale all profiles to the same global flux level
        #    + 'satur_check': check of saturation on detector
        #    + 'PSFx', 'PSFy' : sizes of PSF on detector (?)
        #      'PSFr' : average (quadratic) size
        #      'PSFang' : angle y/x in degrees  
        #    + coefficients of wiggles laws: wig_p_0, wig_p1, wig_wref, wig_ai[i=0,4]
        #    + 'alt': telescope altitude angle (deg)
        #    + 'ha','na','ca','s','rhk': activity indexes 
        #    + 'ADC1 POS','ADC1 RA','ADC1 DEC','ADC2 POS','ADC2 RA','ADC2 DEC': ESPRESSO ADC intel
        #    + 'TILT1 VAL1','TILT1 VAL2','TILT2 VAL1','TILT2 VAL2': ESPRESSO piezo intel
        plot_settings['prop_ordin']=['rv','FWHM','ctrst','AM', 'snr','RVres','RVdrift']
        plot_settings['prop_ordin']=['rv','RVres','FWHM','ctrst','RVpip','RVpipres','FWHMpip','Ctrstpip']
        plot_settings['prop_ordin']=['AM', 'snr']        
        # plot_settings['prop_ordin']=['AM', 'snr_quad','seeing','ha','na','ca','s','rhk']  #ESPRESSO
        # plot_settings['prop_ordin']=['AM', 'snr','seeing','ha','na','ca','s']    #HARPS
        # plot_settings['prop_ordin']=['AM', 'snr','ha','na','ca','s']    #HARPS-N
        plot_settings['prop_ordin']=['rv','RVres','FWHM','ctrst']
        # plot_settings['prop_ordin']=['RVres','FWHM','ctrst']
        # plot_settings['prop_ordin']=['RVpip']
        # plot_settings['prop_ordin']=['RVres','FWHM_voigt','ctrst']
        # plot_settings['prop_ordin']=['RVres','ctrst']
        # plot_settings['prop_ordin']=['ctrst','FWHM']
        # plot_settings['prop_ordin']=['rv','RVres','vsini','cont','ctrst_ord0__IS__VS_','FWHM_ord0__IS__VS_']
        # plot_settings['prop_ordin']=['RVres','FWHM','ctrst']
        # plot_settings['prop_ordin']=['rv_l2c','FWHM_l2c','amp_l2c']
        # plot_settings['prop_ordin']=['rv','RVres','RVpip','RVpipres']  
        # plot_settings['prop_ordin']=['snr']
        # plot_settings['prop_ordin']=['rv'] 
        # plot_settings['prop_ordin']=['ctrst'] 
        # plot_settings['prop_ordin']=['RVres'] 
        # plot_settings['prop_ordin'] = ['colcorrmin','colcorrmax']
        # plot_settings['prop_ordin'] += ['snr','AM']        
        # plot_settings['prop_ordin'] += ['seeing']     
        # plot_settings['prop_ordin'] += ['satur_check']
    #    plot_settings['prop_ordin']=['colcorr450','colcorr550','colcorr650']  
#        plot_settings['prop_ordin']=['RVres','rv_l2c','FWHM_l2c','amp_l2c']     
#        plot_settings['prop_ordin']=['RVres','FWHM_l2c','amp_l2c']  
#        plot_settings['prop_ordin']=['rv','RV_lobe','amp','amp_lobe','FWHM','FWHM_lobe']          
        # plot_settings['prop_ordin']=['wig_p_0', 'wig_p_1', 'wig_wref', 'wig_a_0', 'wig_a_1', 'wig_a_2', 'wig_a_3', 'wig_a_4']
        # plot_settings['prop_ordin']=['RVres']
        # plot_settings['prop_ordin']=['AM']        
        # plot_settings['prop_ordin']+=['az']  
        # plot_settings['prop_ordin']=['ha','na']  
        # plot_settings['prop_ordin']=['ADC1 POS','ADC1 RA','ADC1 DEC','ADC2 POS','ADC2 RA','ADC2 DEC']
        # plot_settings['prop_ordin']=['TILT1 VAL1','TILT1 VAL2','TILT2 VAL1','TILT2 VAL2']        
        


        #Settings for selected properties
        for plot_prop in plot_settings['prop_ordin']:
            key_plot = 'prop_'+plot_prop 
            plot_settings[key_plot]={} 

            #Margins
            plot_settings[key_plot]['margins']=[0.15,0.15,0.95,0.95]   #regular size 
            plot_settings[key_plot]['margins']=[0.15,0.12,0.95,0.55]    #elongated 
            plot_settings[key_plot]['margins']=[0.15,0.12,0.8,0.55]    #DREAM paper
            plot_settings[key_plot]['margins']=[0.15,0.12,0.85,0.55]   #ANTARESS I, mock, multi-tr

            #Font size
            plot_settings[key_plot]['font_size']=18   #ANTARESS I, mock, multi-tr

            #Abscissa
            plot_settings[key_plot]['prop_absc']='phase'
            # plot_settings[key_plot]['prop_absc']='ctrst'
            # plot_settings[key_plot]['prop_absc']='FWHM'
            # plot_settings[key_plot]['prop_absc']='snr'
            # plot_settings[key_plot]['prop_absc']='snr_quad'
            # plot_settings[key_plot]['prop_absc']='AM'
            # plot_settings[key_plot]['prop_absc']='flux_airmass'
            # plot_settings[key_plot]['prop_absc']='seeing'
            # plot_settings[key_plot]['prop_absc']='colcorrmin'
            # plot_settings[key_plot]['prop_absc']='PSFang'
            # plot_settings[key_plot]['prop_absc']='alt'
            # plot_settings[key_plot]['prop_absc'] = 's'
            # plot_settings[key_plot]['prop_absc'] = 'ca'
            # plot_settings[key_plot]['prop_absc'] = 'ha'
            # plot_settings[key_plot]['prop_absc'] = 'na'

            #Visits to plot
            plot_settings[key_plot]['visits_to_plot']={}
            if gen_dic['star_name']=='55Cnc':
                plot_settings[key_plot]['visits_to_plot'].update({'ESPRESSO':['20200205','20210121','20210124']} )
                # plot_settings[key_plot]['visits_to_plot']['ESPRESSO']+=['binned']
                # plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20200205']} 
                # plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20210121']} 
                plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20210124']} 
                
                # plot_settings[key_plot]['visits_to_plot'].update({'HARPS':['20120127','20120213','20120227','20120315']} )
                # plot_settings[key_plot]['visits_to_plot']['HARPS']+=['binned']
                # plot_settings[key_plot]['visits_to_plot']={'HARPS':['20120127']} 
                # plot_settings[key_plot]['visits_to_plot']={'HARPS':['20120213']} 
                # plot_settings[key_plot]['visits_to_plot']={'HARPS':['20120227']} 
                # plot_settings[key_plot]['visits_to_plot']={'HARPS':['20120315']} 
    
                # plot_settings[key_plot]['visits_to_plot'].update({'HARPN':['20131114','20131128','20140101','20140126','20140226','20140329']} )
                # plot_settings[key_plot]['visits_to_plot']={'HARPN':['20131114']}  #V2
                # plot_settings[key_plot]['visits_to_plot']={'HARPN':['20131128']}  #V3
                # plot_settings[key_plot]['visits_to_plot']={'HARPN':['20140101']}  #V4
                # plot_settings[key_plot]['visits_to_plot']={'HARPN':['20140126']}  #V5 
                # plot_settings[key_plot]['visits_to_plot']={'HARPN':['20140226']}  #V6
                # plot_settings[key_plot]['visits_to_plot']={'HARPN':['20140329']}  #V7

                # plot_settings[key_plot]['visits_to_plot'].update({'EXPRES':['20220131','20220406']} )                
                # plot_settings[key_plot]['visits_to_plot']={'EXPRES':['20220131']} 
                plot_settings[key_plot]['visits_to_plot']={'EXPRES':['20220406']}                 
                
    
            if gen_dic['star_name']=='GJ436':
                plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20190228','20190429'],'HARPN':['20160318','20160411'],'HARPS':['20070509']}     
                # plot_settings[key_plot]['visits_to_plot']={'HARPN':['20160318','20160411']}   
                # plot_settings[key_plot]['visits_to_plot']={'HARPS':['20070509']}     
            
            elif gen_dic['studied_pl']=='WASP121b':
                plot_settings[key_plot]['visits_to_plot']={'HARPS':['09-01-18','31-12-17','14-01-18']}  
                plot_settings[key_plot]['visits_to_plot']={'ESPRESSO_MR':['2018-12-01'],'ESPRESSO':['2019-01-07']}  
            elif gen_dic['star_name']=='WASP76':
                plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20180902','20181030']}  
                plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20181030']}  
            elif gen_dic['studied_pl']=='WASP127b':plot_settings[key_plot]['visits_to_plot']={'HARPS':['2017-03-20','2018-03-31','2018-02-13','2017-02-28']} 
            elif gen_dic['star_name']=='HD209458':
                plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20190720','20190911']} 
                plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20190720']} 
                plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20190911']} 
                plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['mock_vis']}                 
                
            elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-20']} 
            elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-03-18']}  
            elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']} 
            elif gen_dic['studied_pl']==['HD3167_c']:plot_settings[key_plot]['visits_to_plot']={'HARPN':['2016-10-01']}        
            # elif gen_dic['star_name']=='WASP156':plot_settings[key_plot]['visits_to_plot']={'CARMENES_VIS':['20191210']}    
            elif gen_dic['star_name']=='HAT_P11':
                plot_settings[key_plot]['visits_to_plot']={'HARPN':['20150913','20151101']}   #RM paper, C corr    
            if gen_dic['star_name']=='HD189733':
                plot_settings[key_plot]['visits_to_plot'].update({'ESPRESSO':['20210810','20210830']} )
            elif gen_dic['star_name']=='GJ3090':
                plot_settings[key_plot]['visits_to_plot'] = {'NIRPS_HA':['20221202']} 
                plot_settings[key_plot]['visits_to_plot'] = {'NIRPS_HE':['20221201']} 
            if gen_dic['star_name']=='WASP43':
                plot_settings[key_plot]['visits_to_plot'].update({'NIRPS_HE':['20230119']} )
            if gen_dic['star_name']=='L98_59':
                plot_settings[key_plot]['visits_to_plot'].update({'NIRPS_HE':['20230411']} )
            if gen_dic['star_name']=='GJ1214':
                plot_settings[key_plot]['visits_to_plot'].update({'NIRPS_HE':['20230407']} )






            #Select indexes of points to be removed from the plot
            #    - leave empty otherwise
            #    - indexes are relative to all exposures
            
        #    plot_settings[key_plot]['idx_noplot']=range(9)+range(51,74)   #WASP8b
        
        #    plot_settings[key_plot]['idx_noplot']={'HARPS':{'2007-05-09':range(7)+[14,15,16]+range(17,35)},   #GJ436b, en excluant les CCFs qui ne remplissent pas le critere sur le contraste
        #                'HARPN':{'2016-03-18':range(63)+[63]+range(71,76),                        
        #                           '2016-04-11':range(20)+[20]+range(28,69)}}

            if gen_dic['studied_pl']=='WASP121b':    
                plot_settings[key_plot]['idx_noplot']={'HARPS':{
                        '31-12-17':np.arange(10,26,dtype=int),   #Exclusion points en transit           
                        '09-01-18':np.arange(8,29,dtype=int),            
                        '14-01-18':np.arange(19,39,dtype=int)}
        #
        #                '31-12-17':np.arange(10,35,dtype=int),   #Exclusion points en transit + post-tr           
        #                '09-01-18':np.arange(8,55,dtype=int),            
        #                '14-01-18':np.arange(19,50,dtype=int)}    
        #
        ##                '31-12-17':np.arange(0,26,dtype=int),   #Exclusion points pre + en transit           
        ##                '09-01-18':np.arange(0,29,dtype=int),            
        ##                '14-01-18':np.arange(0,39,dtype=int)}
        #    
            }
                
            # plot_settings[key_plot]['idx_noplot'] = {'HARPN':{ '20200730':np.arange(90)}}
            elif gen_dic['star_name']=='GJ3090':
                plot_settings[key_plot]['idx_noplot']={'NIRPS_HA':{'20221202':[3,4,5,6,7,8,47,48,49,50,51,52]},'NIRPS_HE':{'20221201':[0,1,2,3,4,5]+list(range(50,83))}}      #exclusion de CCF sans raie detectee        
                plot_settings[key_plot]['idx_noplot']={'NIRPS_HA':{'20221202':[0,1,2,3,4,5,6,7,8,9,10,11,47,48,49,50,51,52,53,54]},'NIRPS_HE':{'20221201':[0,1,2,3,4,5,6,7,8]+list(range(50,83))}}      #exclusion de CCF sans raie detectee + outliers              
        

            #Print and plot mean value and dispersion
            #    - relative to 'all', 'out', 'in' data
            plot_settings[key_plot]['print_disp']=True  # &   False
            plot_settings[key_plot]['disp_mod']='out'   #'out'
            plot_settings[key_plot]['plot_disp']=True # &  False

            #Save dispersion values for analysis in external routine
            plot_settings[key_plot]['save_disp']=False

            #Plot master out property
            plot_settings[key_plot]['plot_Mout']=True   & False
            
            #Reference level
            plot_settings[key_plot]['plot_ref'] = True #  & False

            #Plot HDI subintervals, if available
            plot_settings[key_plot]['plot_HDI']=True   & False    

            #Use different symbols for transits (disks vs squares)
            plot_settings[key_plot]['use_diff_symb']=True # &  False

            #Empty symbols
            plot_settings[key_plot]['empty_in'] = True & False
            plot_settings[key_plot]['empty_all']=False
            
            #Plot in-transit symbols in a different color
            #    - set the requested color ('none' to remove points, '' to use the same color as other points)
            plot_settings[key_plot]['col_in']=''
            if gen_dic['star_name'] in ['WASP43','L98_59','GJ1214']:plot_settings[key_plot]['col_in']='red'
            # if gen_dic['star_name'] in ['HD189733']:plot_settings[key_plot]['col_in']='none'

    
            #Print min/max values (to adjust plot ranges)
            plot_settings[key_plot]['plot_bounds']=True #& False
            
            #Save a text file of residual RVs vs phase
            plot_settings[key_plot]['save_RVres'] = True  & False

            #Plot errorbars / abscissa windows (if available)
            plot_settings[key_plot]['plot_xerr']=True  &  False

            #Transparency of symbols (0 = void)
            plot_settings[key_plot]['alpha_symb']=1. #0.6
          
            #Transparency of error bars (0 = void)
            plot_settings[key_plot]['alpha_err']=0.5 #0.2

            #Overplot transit duration from system properties
            #    - use to check that orbital properties match transit properties  
            plot_settings[key_plot]['plot_T14'] = True #  & False


            #Color table
            if gen_dic['star_name']=='55Cnc':
                plot_settings[key_plot]['color_dic']={
                    'ESPRESSO':{'20200205':'rainbow','20210121':'rainbow','20210124':'rainbow'},
                    'HARPS':{'20120127':'rainbow','20120213':'rainbow','20120227':'rainbow','20120315':'rainbow'}}
    
                plot_settings[key_plot]['color_dic']={
                    'ESPRESSO':{'20200205':'dodgerblue','20210121':'limegreen','20210124':'red','binned':'black'},
                    'HARPS':{'20120127':'dodgerblue','20120213':'limegreen','20120227':'orange','20120315':'red','binned':'black'},
                    'HARPN':{'20131114':'magenta','20131128':'dodgerblue','20140101':'limegreen','20140126':'gold','20140226':'orange','20140329':'red'},                  
                    }
    
                # plot_settings[key_plot]['color_dic']={
                #     'ESPRESSO':{'binned':'red'},
                #     'HARPS':{'binned':'dodgerblue'},                
                #     }
    
            elif gen_dic['star_name']=='GJ436':
                plot_settings[key_plot]['color_dic']={'ESPRESSO':{'20190228':'dodgerblue','20190429':'red'},'HARPN':{'20160318':'orange','20160411':'limegreen'},'HARPS':{'20070509':'magenta'}}     
            elif gen_dic['studied_pl']=='WASP121b':   
                plot_settings[key_plot]['color_dic']={'HARPS':{'09-01-18':'green','14-01-18':'dodgerblue','31-12-17':'red'},
                           'binned':{'HARPS-binned':'black'}}
                plot_settings[key_plot]['color_dic']={'ESPRESSO_MR':{'2018-12-01':'dodgerblue'},'ESPRESSO':{'2019-01-07':'red'}} 
            elif gen_dic['studied_pl']=='Kelt9b':   
                plot_settings[key_plot]['color_dic']={'HARPN':{'31-07-2017':'dodgerblue','20-07-2018':'red'},
                           'binned':{'HARPS-binned':'black'}}            
            elif gen_dic['star_name']=='WASP76':   
                plot_settings[key_plot]['color_dic']={'ESPRESSO':{'20180902':'rainbow','20181030':'rainbow'},
                            'binned':{'ESP_binned':'black'}}              
            elif gen_dic['studied_pl']=='WASP127b':plot_settings[key_plot]['color_dic']={'HARPS':{'2017-03-20':'dodgerblue','2018-03-31':'green','2018-02-13':'orange','2017-02-28':'red'}} 
            elif gen_dic['star_name']=='HD209458':
                plot_settings[key_plot]['color_dic']={'ESPRESSO':{'20190720':'rainbow','20190911':'rainbow'}} 
            elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-10-09':'dodgerblue'}} 
            elif gen_dic['studied_pl']==['HD3167_c']:plot_settings[key_plot]['color_dic']={'HARPN':{'2016-10-01':'dodgerblue'}}   
            elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-02-20':'dodgerblue'}}  
            elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2020-03-18':'dodgerblue'}}  
            elif gen_dic['studied_pl']=='GJ9827d':
                plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-08-25':'dodgerblue'},
                           'HARPS':{'2018-08-18':'dodgerblue','2018-09-18':'red'}}  
            elif gen_dic['studied_pl']=='GJ9827b':
                plot_settings[key_plot]['color_dic']={'HARPS':{'2018-08-04':'dodgerblue','2018-08-15':'limegreen','2018-09-18':'orange','2018-09-19':'red'}}
            elif gen_dic['star_name']=='HIP41378':plot_settings[key_plot]['color_dic']={'HARPN':{'20191218':'dodgerblue','20220401':'red'}}   
            elif gen_dic['star_name']=='TOI858':plot_settings[key_plot]['color_dic']={'CORALIE':{'20191205':'dodgerblue','20210118':'red'}}            
            elif gen_dic['star_name']=='MASCARA1':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'20190714':'dodgerblue','20190811':'red'}}  
            elif gen_dic['star_name']=='V1298tau':plot_settings[key_plot]['color_dic']={'HARPN':{'20200128':'dodgerblue','20201207':'red'}}  
            # elif gen_dic['star_name']=='HAT_P3':plot_settings[key_plot]['color_dic']={'HARPN':{'20190415':'dodgerblue','20200130':'red'}}         
            elif gen_dic['star_name']=='WASP107':plot_settings[key_plot]['color_dic']={'HARPS':{'20140406':'dodgerblue','20180201':'red','20180313':'limegreen'},'CARMENES_VIS':{'20180224':'orange'}}          
            elif gen_dic['star_name']=='WASP166':plot_settings[key_plot]['color_dic']={'HARPS':{'20170114':'dodgerblue','20170304':'red','20170315':'limegreen'}}          
            elif gen_dic['star_name']=='HAT_P11':plot_settings[key_plot]['color_dic']={'HARPN':{'20150913':'dodgerblue','20151101':'red'},'CARMENES_VIS':{'20170807':'orange','20170812':'purple'}}          
            elif gen_dic['star_name']=='WASP156':plot_settings[key_plot]['color_dic']={'CARMENES_VIS':{'20190928':'orange','20191025':'purple','20191210':'darkred'}}          
            elif gen_dic['star_name']=='HD106315':plot_settings[key_plot]['color_dic']={'HARPS':{'20170309':'dodgerblue','20170330':'red','20180323':'limegreen'}}          
            elif gen_dic['star_name']=='GJ3090':plot_settings[key_plot]['color_dic']={'NIRPS_HA':{'20221202':'dodgerblue'},'NIRPS_HE':{'20221201':'red'}}          
            elif gen_dic['star_name']=='HD189733':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'20210810':'dodgerblue','20210830':'red'}} 


                                                                                     

            #Bin properties 
            plot_settings[key_plot]['bin_val'] = {}
            if 1==0:
                plot_settings[key_plot]['bin_val']['alpha_bin']=1.
                plot_settings[key_plot]['alpha_symb']=0.5
                plot_settings[key_plot]['alpha_err']=0.2
                if plot_settings[key_plot]['prop_absc']=='phase':
                    if gen_dic['star_name']=='GJ436':dbin,bin_min,bin_max=0.003 ,-0.05  ,0.05
                    elif gen_dic['star_name']=='HAT_P3':dbin,bin_min,bin_max=0.008 ,-0.034  ,0.057
                    elif gen_dic['star_name']=='HAT_P11':dbin,bin_min,bin_max=0.0035 ,-0.045  ,0.045
                    elif gen_dic['star_name']=='HAT_P49':dbin,bin_min,bin_max=0.0045 ,-0.06  ,0.064
                    elif gen_dic['star_name']=='HAT_P33':dbin,bin_min,bin_max=0.005 ,-0.047  ,0.0357
                    elif gen_dic['star_name']=='K2_105':dbin,bin_min,bin_max=0.002 ,-0.0145  ,0.017
                    elif gen_dic['star_name']=='HD89345':dbin,bin_min,bin_max=0.001 ,-0.018  ,0.0165
                    elif gen_dic['star_name']=='HD106315':dbin,bin_min,bin_max=0.00075 ,-0.0107  ,0.0104
                    elif gen_dic['star_name']=='Kepler25':dbin,bin_min,bin_max=0.00105 ,-0.011  ,0.0115
                    elif gen_dic['star_name']=='Kepler63':dbin,bin_min,bin_max=0.0020 ,-0.004  ,0.011
                    elif gen_dic['star_name']=='Kepler68':dbin,bin_min,bin_max=0.002 ,-0.031  ,0.02  
                    elif gen_dic['star_name']=='WASP107':dbin,bin_min,bin_max=0.0022 ,-0.038  ,0.024
                    elif gen_dic['star_name']=='WASP156':dbin,bin_min,bin_max=0.006 ,-0.038  ,0.055
                    elif gen_dic['star_name']=='WASP166':dbin,bin_min,bin_max=0.002 ,-0.028  ,0.035
                    elif gen_dic['star_name']=='HD106315':dbin,bin_min,bin_max=0.001 ,-0.05  ,0.05
                    elif gen_dic['star_name']=='WASP47':
                        dbin,bin_min,bin_max=0.0025 ,-0.008  ,0.016
                        dbin=None
                        x_bd_low= np.array([-0.008,-0.005,-0.002,0.001,0.004,0.007,0.010,0.013])
                        x_bd_high=np.array([-0.005,-0.002, 0.001,0.004,0.007,0.010,0.013,0.016])

                    elif gen_dic['star_name']=='55Cnc':
                        plot_settings[key_plot]['bin_val']['ESPRESSO'] = {'dbin':0.006,'bin_min':-0.16,'bin_max':0.16,'color':'dodgerblue'}   
                        plot_settings[key_plot]['bin_val']['HARPS'] = {'dbin':0.006,'bin_min':-0.06,'bin_max':0.105,'color':'red'}   
                        plot_settings[key_plot]['bin_val']['HARPN'] = {'dbin':0.006,'bin_min':-0.155,'bin_max':0.125,'color':'black'}   
                        # plot_settings[key_plot]['bin_val']['all'] = {'dbin':0.006,'bin_min':-0.16,'bin_max':0.16,'color':'black'}  
            
                if plot_settings[key_plot]['prop_absc']=='snr':
                    dbin,bin_min,bin_max=3 ,30  ,70   

            #Do not plot original data
            #    - useful to show only binned date
            plot_settings[key_plot]['no_orig']=True & False

            #Do not plot any data
            #    - useful to show only models
            plot_settings[key_plot]['no_data']=True & False

            #Normalisation of values by their out-of-transit mean
            #    - can be useful for comparison between visits
            plot_settings[key_plot]['norm_out']=True  &  False   
            if gen_dic['star_name']=='HD189733':  plot_settings[key_plot]['norm_out'] = True 
    
            #Bornes du plot       
            if gen_dic['studied_pl']=='WASP_8b':
                if plot_settings[key_plot]['prop_absc']=='phase':            
                    plot_settings[key_plot]['x_range']=[-0.018,0.022]          
                if plot_settings[key_plot]['prop_absc']=='snr':            
                    plot_settings[key_plot]['x_range']=[25.,55.]   
                if plot_settings[key_plot]['prop_absc']=='AM':            
                    plot_settings[key_plot]['x_range']=[0.9,1.7] 
            if gen_dic['star_name']=='55Cnc':
                if plot_settings[key_plot]['prop_absc']=='phase':            
                    if 'ESPRESSO' in plot_settings[key_plot]['visits_to_plot']:
                        plot_settings[key_plot]['x_range']=[-0.165,0.165]
                        if plot_settings[key_plot]['visits_to_plot']['ESPRESSO']==['20200205']:plot_settings[key_plot]['x_range']=[-0.1,0.165]
                        if plot_settings[key_plot]['visits_to_plot']['ESPRESSO']==['20210121']:plot_settings[key_plot]['x_range']=[-0.17,0.12]
                        if plot_settings[key_plot]['visits_to_plot']['ESPRESSO']==['20210124']:plot_settings[key_plot]['x_range']=[-0.12,0.17]
                    if 'HARPS' in plot_settings[key_plot]['visits_to_plot']:
                        if plot_settings[key_plot]['visits_to_plot']['HARPS']==['20120127']:plot_settings[key_plot]['x_range']=[-0.067,0.105]
                        if plot_settings[key_plot]['visits_to_plot']['HARPS']==['20120213']:plot_settings[key_plot]['x_range']=[-0.032,0.112]
                        if plot_settings[key_plot]['visits_to_plot']['HARPS']==['20120227']:plot_settings[key_plot]['x_range']=[-0.065,0.065]
                        if plot_settings[key_plot]['visits_to_plot']['HARPS']==['20120315']:plot_settings[key_plot]['x_range']=[-0.062,0.085]
                    
                    
                    
                    
            if gen_dic['studied_pl']==['HD3167_c']:      
                if plot_settings[key_plot]['prop_absc']=='phase':
                    plot_settings[key_plot]['x_range']=[-0.0075,0.0075] 
                if plot_settings[key_plot]['prop_absc']=='snr':
                    plot_settings[key_plot]['x_range']=[30,130]     
        
            if gen_dic['studied_pl']=='WASP121b':      
                if plot_settings[key_plot]['prop_absc']=='phase':
                    plot_settings[key_plot]['x_range']=[-0.16,0.19] 
        
                if plot_settings[key_plot]['prop_absc']=='snr':
                    plot_settings[key_plot]['x_range']=[24.,55.]     #SNR50
        
                if plot_settings[key_plot]['prop_absc']=='AM':
                    plot_settings[key_plot]['x_range']=[0.8,2.]     
        
            if gen_dic['studied_pl']=='Kelt9b':      
                if plot_settings[key_plot]['prop_absc']=='phase':
                    plot_settings[key_plot]['x_range']=[-0.15,0.11]     
    
            if gen_dic['star_name']=='WASP76b':      
                if plot_settings[key_plot]['prop_absc']=='phase':
                    plot_settings[key_plot]['x_range']=[-0.08,0.09]
    
            elif gen_dic['studied_pl']=='WASP127b':
                plot_settings[key_plot]['x_range']=[-0.05,0.04] 
    
            if gen_dic['star_name']=='GJ436':
                if plot_settings[key_plot]['prop_absc']=='phase': 
                    plot_settings[key_plot]['x_range']=[-0.13,0.085]  
                    # plot_settings[key_plot]['x_range']=[-0.04,0.05]   #ESPRESSO alone / binned data  
                if plot_settings[key_plot]['prop_absc']=='snr':
                    plot_settings[key_plot]['x_range']=[20.,41.]   #HARPS-N
                    plot_settings[key_plot]['x_range']=[13.,33.]   #HARPS                
                if plot_settings[key_plot]['prop_absc']=='snr_quad':
                    plot_settings[key_plot]['x_range']=[31.,71.]   #ESPRESSO
    
                
            if gen_dic['studied_pl']=='HD209458b':
                if plot_settings[key_plot]['prop_absc']=='phase': plot_settings[key_plot]['x_range']=[-0.031,0.041]       
            if gen_dic['studied_pl']==['HD3167_b']:
                if plot_settings[key_plot]['prop_absc']=='phase':
                    plot_settings[key_plot]['x_range']=[-0.08,0.11]    #[-0.05,0.13] 
                    plot_settings[key_plot]['x_range']=[-0.075,0.115]    #[-0.05,0.13] 
                if plot_settings[key_plot]['prop_absc']=='snr': plot_settings[key_plot]['x_range']=[30.,70.] 
                if plot_settings[key_plot]['prop_absc']=='snr_quad': plot_settings[key_plot]['x_range']=[47.,99.] 
            elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['x_range']=[-0.08,0.16]  
            elif gen_dic['studied_pl']=='Nu2Lupi_c':
                plot_settings[key_plot]['x_range']=[-0.0062,  0.0042]      
                if plot_settings[key_plot]['prop_absc']=='ctrst':plot_settings[key_plot]['x_range']=[0.5755,0.577]  
                if plot_settings[key_plot]['prop_absc']=='FWHM':plot_settings[key_plot]['x_range']=[7.133,7.141]   
            elif gen_dic['studied_pl']=='GJ9827d':
                plot_settings[key_plot]['x_range']=[-0.019,  0.022]     #ESPRESSO
                plot_settings[key_plot]['x_range']=[-0.04 ,  0.02 ]     #HARPS
                if plot_settings[key_plot]['prop_absc']=='AM':plot_settings[key_plot]['x_range'] = [1.,2.5]
                if plot_settings[key_plot]['prop_absc']=='seeing':plot_settings[key_plot]['x_range'] = [0.3,1.9] 
            elif gen_dic['studied_pl']=='GJ9827b':
                if plot_settings[key_plot]['prop_absc']=='phase':plot_settings[key_plot]['x_range'] = [-0.16,0.1]
                if plot_settings[key_plot]['prop_absc']=='AM':plot_settings[key_plot]['x_range'] = [1.1,2.4]
                if plot_settings[key_plot]['prop_absc']=='seeing':plot_settings[key_plot]['x_range'] = [0.4,1.8] 
            if gen_dic['star_name']=='TOI858':    
                if plot_settings[key_plot]['prop_absc']=='phase':plot_settings[key_plot]['x_range'] = [-0.035,0.05]            
            if gen_dic['star_name']=='HIP41378':    
                if plot_settings[key_plot]['prop_absc']=='phase':
                    plot_settings[key_plot]['x_range'] = [0.,1.3e-3] 
                    plot_settings[key_plot]['x_range'] = [-0.001796223476066,0.001796223476066] 
            if gen_dic['star_name']=='HD15337':    
                if plot_settings[key_plot]['prop_absc']=='snr':
                    plot_settings[key_plot]['x_range'] = [95,185]    
    
    
            #Orders for SNR
            #    - specific to each instrument because the same spectral range may be associated with different orders
            #    - beware: order 50 that contains 550 nm is at index 49 in HARPS,n 46 in HARPS-N
            #              for CARMENES we use order 40 (centered at 7840 A), which contains the maximum flux for a K-type star
            #              for NIRPS we use order 57 (centered at 16285 A), little affected by tellurics 
        #     plot_settings[key_plot]['idx_SNR']={
        # #                  'HARPN':range(35)
        # #                  'HARPN':range(35,69)
        #                   # 'ESPRESSO':[102], 
        #                   # 'NIRPS_HE':[57],
        #                   }
            if gen_dic['star_name']=='55Cnc':              
                plot_settings[key_plot]['idx_SNR']['EXPRES']=[14]   #562 nm, index specific to the chosen order removal
            plot_settings[key_plot]['idx_num_SNR']={'HARPN':range(28,31)   #range(10)
                         }
            plot_settings[key_plot]['idx_den_SNR']={'HARPN':[32]}   #range(59,69)
            if gen_dic['star_name']=='TOI858':  
                plot_settings[key_plot]['idx_ord_SNR']={'CORALIE':[46]}    
    

            #-------------------------------------------------
            #Degree of polynomial to fit ordina property vs abscissa property
            #    - set to {} to prevent fit
            #    - detrending is performed with respect to the multiplied/added polynomials associated to each requested property
            #    - structure is deg_prop_fit > inst > vis > {prop1 : deg , prop2 : deg ..}   
            plot_settings[key_plot]['deg_prop_fit']={}
            deg_corr = 0
            for inst in plot_settings[key_plot]['visits_to_plot']:
                plot_settings[key_plot]['deg_prop_fit'][inst]={}
                for vis in plot_settings[key_plot]['visits_to_plot'][inst]:plot_settings[key_plot]['deg_prop_fit'][inst][vis]={plot_settings[key_plot]['prop_absc']:deg_corr}

    
            # if gen_dic['star_name']=='WASP156':
            #     plot_settings[key_plot]['deg_prop_fit']={'phase':3,'snr':2}            
            #     # plot_settings[key_plot]['deg_prop_fit']={'phase':2}          
            if gen_dic['star_name']=='HAT_P11':
                plot_settings[key_plot]['deg_prop_fit']['HARPN']['20150913']['snr']= 2
                plot_settings[key_plot]['deg_prop_fit']['HARPN']['20151101']['snr']= 1
            if gen_dic['star_name']=='55Cnc':
                # plot_settings[key_plot]['deg_prop_fit']['ESPRESSO'][
                #     '20200205']={'phase':1,'snr_quad':2}  
                #     #'20210121']={'phase':1,'snr_quad':1} 
                #     #'20210124']={'phase':2,'snr_quad':3}
                # plot_settings[key_plot]['deg_prop_fit']['HARPS'][
                #     '20120127']={'phase':0,'snr':0}  
                #     #'20120213']={'phase':1,'snr':1} 
                #     #'20120227']={'phase':2,'snr':3}  
                #     #'20120315']={'phase':1,'snr':1}             
               plot_settings[key_plot]['deg_prop_fit']['EXPRES'][
                    '20220131']={'phase':0,'snr':0}  
                    # '20220406']={'phase':0,'snr':0}              
            if gen_dic['star_name']=='WASP43':
                plot_settings[key_plot]['deg_prop_fit']['NIRPS_HE']['20230119']['phase']= 1     
                # plot_settings[key_plot]['deg_prop_fit']['NIRPS_HE']['20230119']['snr']= 1   

            if gen_dic['star_name']=='L98_59':   
                plot_settings[key_plot]['deg_prop_fit']['NIRPS_HE']['20230411']['phase']= 0       
                # plot_settings[key_plot]['deg_prop_fit']['NIRPS_HE']['20230411']['snr']= 1  
                
            if gen_dic['star_name']=='GJ1214':    
                plot_settings[key_plot]['deg_prop_fit']['NIRPS_HE']['20230407']['phase']= 1
                # plot_settings[key_plot]['deg_prop_fit']['NIRPS_HE']['20230407']['snr']= 1        
                
                  
            if gen_dic['star_name']=='HD189733':
                plot_settings[key_plot]['deg_prop_fit']['ESPRESSO']['20210810']['phase']= 0 
                plot_settings[key_plot]['deg_prop_fit']['ESPRESSO']['20210830']['phase']= 0
                # plot_settings[key_plot]['deg_prop_fit']['ESPRESSO']['20210810']['snr_quad']= 0
                # plot_settings[key_plot]['deg_prop_fit']['ESPRESSO']['20210830']['snr_quad']= 0 

                    
            plot_settings[key_plot]['deg_prop_fit']={}




            #Multiplication/addition of sinusoidal component to the fit 
            # if gen_dic['star_name']=='55Cnc':
            #     plot_settings[key_plot]['fit_sin']['ESPRESSO']={'20200205':'phase','20210121':'phase','20210124':'phase'}
            
            
            #Scaling value for errors on fitted data
            #    - set to None to be set to dispersion
            # plot_settings[key_plot]['set_err']=  sqrt(8.999011582452807)            
            
            #Points to be fitted
            #    - set instrument and visit to an empty list for its out-of-transit exposures to be fitted automatically, otherwise indicate specific exposures
            plot_settings[key_plot]['idx_fit']={}
            for inst in plot_settings[key_plot]['visits_to_plot']:
                plot_settings[key_plot]['idx_fit'][inst]={}
                for vis in plot_settings[key_plot]['visits_to_plot'][inst]:plot_settings[key_plot]['idx_fit'][inst][vis]=[]  
            
            if gen_dic['studied_pl']=='WASP_8b':
                if plot_prop==['ctrst']:
                    plot_settings[key_plot]['idx_fit']={
                        'HARPS':{'2008-10-04':range(7)+range(56,74)}
                        }
                
            if gen_dic['studied_pl']=='55Cnc_e':
            #     if plot_prop==['RVres']:
            #         plot_settings[key_plot]['idx_fit']={
            #             'HARPN':{
            #              '2014-03-29':range(10)+range(24,27),  
            #              '2014-01-01':range(13)+range(27,33), 
            # #             '2014-01-01':range(9,14)+range(27,33), #pas de transit ni premiers points stables
            #              '2012-12-25':range(7,27),            #expos sans debut + transit 
            # #             '2012-12-25':[0]+range(7,27),            
            #              '2013-11-28':range(26)+range(45,61), 
            #              '2013-11-14':range(25)+range(46,53), 
            #              '2014-01-26':range(3)+range(17,30),  
            #              '2014-02-26':range(7)+range(21,30),  
            #             }, 
            #         'HARPS':{
            #             '2012-02-27':range(6)+range(31,36),
            #             '2012-03-15':range(4)+range(30,41),
            #             '2012-02-13':range(29,55),
            #             '2012-01-27':range(6)+range(31,47),
            #         }            
            #         }
            #     if plot_prop==['ctrst'] or plot_prop==['FWHM']:
            #         plot_settings[key_plot]['idx_fit']={
            #             'HARPN':{
            #              '2014-03-29':range(10)+range(24,27),  #best-fit : deg1 
            #              '2014-01-01':range(13)+range(27,33),  #best-fit : deg1 
            #              '2012-12-25':[0]+range(7,27),         #best-fit : deg0            
            #              '2013-11-28':range(26)+range(45,61),  #best-fit : deg0  
            #              '2013-11-14':range(25)+range(46,53),  #best-fit : deg0  
            #              '2014-01-26':range(3)+range(17,30),   #best-fit : deg0  
            #              '2014-02-26':range(7)+range(21,30),   #best-fit : deg0  
            #             } , 
            #         'HARPS':{
            #             '2012-02-27':range(6)+range(31,36),
            #             '2012-03-15':range(4)+range(30,41),
            #             '2012-02-13':range(29,55),
            #             '2012-01-27':range(6)+range(31,47),
            #         }   
            #         }
                plot_settings[key_plot]['idx_fit']={'ESPRESSO':{'2020-02-05':list(range(0,22))+list(range(55,97))}}         
            
            elif gen_dic['studied_pl']=='WASP121b':    #en ayant ote les points in-transit
                plot_settings[key_plot]['idx_fit']={'HARPS':{'31-12-17':range(19)}} 
            elif gen_dic['studied_pl']=='Nu2Lupi_c':    
                plot_settings[key_plot]['idx_fit']={'ESPRESSO':{'2020-03-18':range(16,68)}}    
                plot_settings[key_plot]['idx_fit']={'ESPRESSO':{'2020-03-18':list(range(16,24))+list(range(59,68))}}   
        
            elif gen_dic['star_name']=='GJ436':
                #Exclusion in-transit values
                plot_settings[key_plot]['idx_fit']={'ESPRESSO':{'20190228':list(range(17))+list(range(27,49)),'20190429':list(range(16))+list(range(26,49))},
                         'HARPN':{'20160318':list(range(63))+list(range(72,76)),'20160411':list(range(20))+list(range(29,70))},
                         'HARPS':{'20070509':list(range(10))+list(range(20,44))}}
                    
        
            elif gen_dic['star_name']=='HD15337':    
                plot_settings[key_plot]['idx_fit']={'ESPRESSO_MR':{'20191122':list(range(91))+list(range(116,127))}}
            elif gen_dic['star_name']=='HAT_P3':    
                plot_settings[key_plot]['idx_fit']={'HARPN':{'20200130':[]}}             
            
            
    
            #-----------------------------------------------------
            #RV plot
            if (plot_prop=='rv'):

                #Plot theoretical RV curve
                if gen_dic['star_name']=='HIP41378':plot_settings[key_plot]['theoRV'] = True
    
                #Bornes du plot
                plot_settings[key_plot]['y_range']=None
                if gen_dic['studied_pl']=='GJ436_b':
                    plot_settings[key_plot]['y_range']=[9.761,9.829] 
                    plot_settings[key_plot]['y_range']=[-0.0195,0.075]   #avec erreurs 
                    plot_settings[key_plot]['y_range']=[-0.01,0.07]   #sans erreurs 
                    plot_settings[key_plot]['y_range']=[9.705,9.7305]   
                if gen_dic['star_name']=='55Cnc':    
                    # if ('HARPS' in data_dic['DI']) and (gen_dic['n_instru']==1):plot_settings[key_plot]['y_range']=[27.34,27.55]    
                    if ('HARPN' in data_dic['DI']) and (gen_dic['n_instru']==1):
                        if data_dic['DI']['HARPN'].keys()==['2012-12-25']:plot_settings[key_plot]['y_range']=[27.41,27.45] 
                        if data_dic['DI']['HARPN'].keys()==['2013-11-14']:plot_settings[key_plot]['y_range']=[27.34,27.4]     
                        if data_dic['DI']['HARPN'].keys()==['2013-11-28']:plot_settings[key_plot]['y_range']=[27.36,27.42]    
                        if data_dic['DI']['HARPN'].keys()==['2014-01-01']:plot_settings[key_plot]['y_range']=[27.40,27.43]    
                        if data_dic['DI']['HARPN'].keys()==['2014-01-26']:plot_settings[key_plot]['y_range']=[27.38,27.42]    
                        if data_dic['DI']['HARPN'].keys()==['2014-02-26']:plot_settings[key_plot]['y_range']=[27.375,27.395]  
                        if data_dic['DI']['HARPN'].keys()==['2014-03-29']:plot_settings[key_plot]['y_range']=[27.385,27.395]  
    
                if gen_dic['studied_pl']==['HD3167_b']:
                    plot_settings[key_plot]['y_range']=[19.40 ,19.415]
                    plot_settings[key_plot]['y_range']=[19.40+0.021 ,19.415+0.021] 
                    plot_settings[key_plot]['y_range']=[19.40994586 ,19.43827671] 
                    plot_settings[key_plot]['y_range']=None
                    
                if gen_dic['studied_pl']==['HD3167_c']:
                    plot_settings[key_plot]['y_range']=[19.493,19.512] 
                    plot_settings[key_plot]['y_range']=np.array([19.3669 , 19.3813])+ 0.0175
                    plot_settings[key_plot]['y_range']=None
                if gen_dic['studied_pl']=='WASP121b':
                    plot_settings[key_plot]['y_range']=[38.15,38.6] 
                    plot_settings[key_plot]['y_range']=[38.25,38.7]  
                    plot_settings[key_plot]['y_range']=[38.1,38.65]    
                if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['y_range']=[-1.35,-1.]
                if gen_dic['studied_pl']=='WASP127b':plot_settings[key_plot]['y_range']=[-9.24,-9.17]
                elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['y_range']=[31.03,31.08]
                elif gen_dic['studied_pl']=='Nu2Lupi_c':
                    plot_settings[key_plot]['y_range']=[-68.814, -68.804]
                    plot_settings[key_plot]['y_range']=[-68.814+4e-3, -68.804+4e-3]   #blue detector
                    plot_settings[key_plot]['y_range']=[-68.814-13e-3, -68.804-13e-3]   #red detector
                elif gen_dic['studied_pl']=='GJ9827d':
                    plot_settings[key_plot]['y_range']=[31.954,31.966]
                    plot_settings[key_plot]['y_range'] = None
                elif gen_dic['star_name']=='HD15337':                           
                    plot_settings[key_plot]['y_range']=[76.158,76.164] 
                    plot_settings[key_plot]['y_range']=[76.158-3e-3,76.164-3e-3]     #C corr, RV corr
    
                elif gen_dic['star_name']=='V1298tau': 
                    plot_settings[key_plot]['y_range']=[14.3,15.3]
    
            #-----------------------------------------------------
            #RV pipeline plot
            if (plot_prop=='RVpip'):      
    
                #Bornes du plot
                if gen_dic['studied_pl']=='55Cnc_e':    
                    if ('HARPS' in data_dic['DI']) and (gen_dic['n_instru']==1):plot_settings[key_plot]['y_range']=[27.34,27.55]    
                    if ('HARPN' in data_dic['DI']) and (gen_dic['n_instru']==1):
                        if data_dic['DI']['HARPN'].keys()==['2012-12-25']:plot_settings[key_plot]['y_range']=[27.41,27.45] 
                        if data_dic['DI']['HARPN'].keys()==['2013-11-14']:plot_settings[key_plot]['y_range']=[27.34,27.4]     
                        if data_dic['DI']['HARPN'].keys()==['2013-11-28']:plot_settings[key_plot]['y_range']=[27.36,27.42]    
                        if data_dic['DI']['HARPN'].keys()==['2014-01-01']:plot_settings[key_plot]['y_range']=[27.40,27.43]    
                        if data_dic['DI']['HARPN'].keys()==['2014-01-26']:plot_settings[key_plot]['y_range']=[27.38,27.42]    
                        if data_dic['DI']['HARPN'].keys()==['2014-02-26']:plot_settings[key_plot]['y_range']=[27.375,27.395]  
                        if data_dic['DI']['HARPN'].keys()==['2014-03-29']:plot_settings[key_plot]['y_range']=[27.385,27.395]         
                if gen_dic['studied_pl']==['HD3167_c']:
                    plot_settings[key_plot]['y_range']=[19.493,19.512] 
                    plot_settings[key_plot]['y_range']=None
                if gen_dic['studied_pl']=='WASP121b':
                    plot_settings[key_plot]['y_range']=[38.15,38.6] 
                if gen_dic['star_name']=='TOI858':plot_settings[key_plot]['y_range']=[64.22,64.48]     

            #-----------------------------------------------------
            #RV residual plot (m/s)
            if (plot_prop=='RVres'):
            
                #Bornes du plot
                if gen_dic['studied_pl']=='GJ436_b':
                    plot_settings[key_plot]['y_range']=[-10.,10.] 
                    plot_settings[key_plot]['y_range']=[-4.,4.] 
    
                if gen_dic['studied_pl']=='WASP121b':
                    plot_settings[key_plot]['y_range']=[-100.,50]
                    plot_settings[key_plot]['y_range']=[140.,300]
                    plot_settings[key_plot]['y_range']=[0.,220]
                if gen_dic['studied_pl']=='Kelt9b':plot_settings[key_plot]['y_range']=[-300.,300]     
                if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['y_range']=[-50.,50.]     
                elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['y_range']=[-20.,20.]
                elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['y_range']=[-50.,50.] 
                elif gen_dic['studied_pl']=='GJ9827d':
                    plot_settings[key_plot]['y_range']=[-5.,10.]  
                    plot_settings[key_plot]['y_range'] = None        
                if gen_dic['studied_pl']==['HD3167_b']:
                    plot_settings[key_plot]['y_range']=[-5.,5.]     #HD3167 paper
                if gen_dic['studied_pl']==['HD3167_c']:
                    plot_settings[key_plot]['y_range']=[-10.,10.]
                if gen_dic['star_name']=='HIP41378':
                    plot_settings[key_plot]['y_range']=[-10.,10.]
                if gen_dic['star_name']=='55Cnc':
                    if ('ESPRESSO' in data_dic['DI']) and (gen_dic['n_instru']==1):plot_settings[key_plot]['y_range']=[-2.,2.]        
                if gen_dic['star_name']=='HD189733':
                    if plot_settings[key_plot]['col_in']=='none':plot_settings[key_plot]['y_range']=[-3.,3.]


            #-----------------------------------------------------
            #RV pipeline residual plot (m/s)
            if (plot_prop=='RVpipres'):
                
                #Bornes du plot
                if gen_dic['studied_pl']=='55Cnc_e':    
                    if ('HARPS' in data_dic['DI']) and (gen_dic['n_instru']==1):
                        plot_settings[key_plot]['y_range']=[-5.,5.]     #avec vsys mesuree   
                        plot_settings[key_plot]['y_range']=[-25.,-5.]     #avec vsys connue  
                    if ('HARPN' in data_dic['DI']) and (gen_dic['n_instru']==1):
                        plot_settings[key_plot]['y_range']=[-10.,10.] 
                        if data_dic['DI']['HARPN'].keys()==['2012-12-25']:
                            plot_settings[key_plot]['y_range']=[-5.,5.]     #avec vsys mesuree
        #                    plot_settings[key_plot]['y_range']=[0.,10.]     #avec vsys connue
                        if data_dic['DI']['HARPN'].keys()==['2013-11-14']:plot_settings[key_plot]['y_range']=[-5.,5.]     
                        if data_dic['DI']['HARPN'].keys()==['2013-11-28']:plot_settings[key_plot]['y_range']=[-5.,5.]    
                        if data_dic['DI']['HARPN'].keys()==['2014-01-01']:plot_settings[key_plot]['y_range']=[-5.,5.]    
                        if data_dic['DI']['HARPN'].keys()==['2014-01-26']:plot_settings[key_plot]['y_range']=[-5.,5.]    
                        if data_dic['DI']['HARPN'].keys()==['2014-02-26']:plot_settings[key_plot]['y_range']=[-5.,5.] 
                        if data_dic['DI']['HARPN'].keys()==['2014-03-29']:plot_settings[key_plot]['y_range']=[-5.,5.] 
        
                if gen_dic['studied_pl']==['HD3167_c']:
                    plot_settings[key_plot]['y_range']=[-10.,10.] 
                if gen_dic['studied_pl']=='WASP121b':
                    plot_settings[key_plot]['y_range']=[-100.,50]         


            #-----------------------------------------------------
            #FWHM plot
            if (plot_prop=='FWHM'):
            
                #Bornes du plot
                if gen_dic['studied_pl']=='GJ436_b':
                    plot_settings[key_plot]['y_range']=[2.85,3.10]    #   single gauss, full range  
                #    plot_settings[key_plot]['y_range']=[3.,3.5]    #   single gauss, range excluded    
                    plot_settings[key_plot]['y_range']=[4.53,4.77]    #   double gauss, core FWHM, avec erreurs
                    plot_settings[key_plot]['y_range']=[4.551,4.749]    #   double gauss, core FWHM, sans erreurs
                    plot_settings[key_plot]['y_range']=[6.93,6.99]    #   double gauss, core FWHM, ESPRESSO
    
                if gen_dic['studied_pl']=='WASP121b':    
                    plot_settings[key_plot]['y_range']=[18.85,19.7]  
                    plot_settings[key_plot]['y_range']=[18.85,19.4]     #out-tr seul single gaussian
                    plot_settings[key_plot]['y_range']=[24.5,26.2]           #out-tr seul, double-gaussian fixed  
    
                if gen_dic['studied_pl']=='Kelt9b':    
                    plot_settings[key_plot]['y_range']=[140.,190.]   
                if gen_dic['studied_pl']=='WASP76b':
                    plot_settings[key_plot]['y_range']=[9.6,10.]  
                    # plot_settings[key_plot]['y_range']=[8.,9.]       
                elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['y_range']=[7.46,7.55]
                elif gen_dic['studied_pl']=='Nu2Lupi_c':
                    plot_settings[key_plot]['y_range']=[7.132,7.15]
                    plot_settings[key_plot]['y_range']=[7.132-0.045,7.15-0.045]    #blue detector
                    plot_settings[key_plot]['y_range']=[7.132+0.1544,7.15+0.1544]  #red detector
                elif gen_dic['studied_pl']=='GJ9827d':
                    plot_settings[key_plot]['y_range']=[6.43 ,6.48]
                    plot_settings[key_plot]['y_range'] = None            
    
    
                if gen_dic['studied_pl']==['HD3167_b']:    
                    plot_settings[key_plot]['y_range']=[7.565,7.605]    #ESPRESSO     #HD3167 paper
                    plot_settings[key_plot]['y_range']=[7.18, 7.22]    #ESPRESSO  orders removed      #HD3167 paper
                    plot_settings[key_plot]['y_range']=[8.01, 8.08]    #ESPRESSO  microtell corr      #HD3167 paper 
                    plot_settings[key_plot]['y_range']=None    #ESPRESSO  microtell corr      #HD3167 paper 
                if gen_dic['studied_pl']==['HD3167_c']:    
                    plot_settings[key_plot]['y_range']=[6.795,6.82]    #mask G2
                    plot_settings[key_plot]['y_range']=[6.22,6.27]     #mask K5  
                    plot_settings[key_plot]['y_range']=[6.52, 6.545]     #newred  
                    plot_settings[key_plot]['y_range']=None      
                    # plot_settings[key_plot]['y_range']=[7.895, 7.94]     #newred          
                if gen_dic['star_name']==['HD15337']:    
                    plot_settings[key_plot]['y_range']=[7.539,7.551]
    
                if gen_dic['star_name']=='55Cnc':    
                    if ('ESPRESSO' in data_dic['DI']) and (gen_dic['n_instru']==1) and plot_settings[key_plot]['norm_out']:plot_settings[key_plot]['y_range']=[0.9993,1.0007] 
                if gen_dic['star_name']=='HD189733':
                    if plot_settings[key_plot]['col_in']=='none':plot_settings[key_plot]['y_range']=[0.999,1.001]


            #-----------------------------------------------------
            #FWHM pipeline plot
            if (plot_prop=='FWHMpip'):
            
                #Bornes du plot
                if gen_dic['studied_pl']==['HD3167_b']:    
                    plot_settings[key_plot]['y_range']=[7.565,7.605]    #ESPRESSO     
 
            #-----------------------------------------------------
            #Contrast plot
            if (plot_prop=='ctrst'):
            
                #Bornes du plot  
                if gen_dic['star_name']=='GJ436':
                    plot_settings[key_plot]['y_range']=[0.24,0.26]    #   single gauss, full range, HARPS-N  
                #    plot_settings[key_plot]['y_range']=[0.22,0.25]    #   single gauss, full range, HARPS
                #    plot_settings[key_plot]['y_range']=[0.215,0.24]    #   single gauss, range excluded
                    plot_settings[key_plot]['y_range']=[0.208,0.238]    #   double gauss
                    plot_settings[key_plot]['y_range']=[0.34,0.345]    #   double gauss, ESPRESSO
    
                    # plot_settings[key_plot]['y_range']=[0.320,0.3238]    #HARPS-N           
                    # plot_settings[key_plot]['y_range']=[0.312,0.319]    #HARPS 
                    plot_settings[key_plot]['y_range']=[0.306,0.308]    #ESPRESSO
    
                
    
                elif gen_dic['studied_pl']=='WASP121b':    
                    plot_settings[key_plot]['y_range']=[0.15,0.1565] 
                    plot_settings[key_plot]['y_range']=[0.1515,0.1565]     #out-tr seul, single gaussian
                    plot_settings[key_plot]['y_range']=[0.240,0.247]           #out-tr seul, double-gaussian fixed    
    
                elif gen_dic['studied_pl']=='Kelt9b':  
                    plot_settings[key_plot]['y_range']=[0.024,0.034]  
                    
                if gen_dic['studied_pl']=='WASP76b':
                    plot_settings[key_plot]['y_range']=[0.595,0.615] 
                    # plot_settings[key_plot]['y_range']=[0.4,0.6]    
                elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['y_range']=[0.652,0.659]
                elif gen_dic['studied_pl']=='Nu2Lupi_c':
                    plot_settings[key_plot]['y_range']=[0.573 ,0.577]
                    plot_settings[key_plot]['y_range']=[0.573+0.0225 ,0.577+0.0225]    #blue detector
                    plot_settings[key_plot]['y_range']=[0.573-0.0575 ,0.577-0.0575]    #red detector
                elif gen_dic['studied_pl']=='GJ9827d':
                    plot_settings[key_plot]['y_range']=[0.5705,0.5725]
                    plot_settings[key_plot]['y_range'] = None
         
                if gen_dic['studied_pl']==['HD3167_b']:    
                    plot_settings[key_plot]['y_range']=[0.6717,0.6738]    #ESPRESSO         #HD3167 paper 
                    plot_settings[key_plot]['y_range']=[6.67597e-01-0.001,6.67597e-01+0.001]    #ESPRESSO orders removed               #HD3167 paper 
                    plot_settings[key_plot]['y_range']=[0.708 , 0.711]    #G8, ESPRESSO no microtell corr               #HD3167 paper 
                    # plot_settings[key_plot]['y_range']=[0.7033, 0.7065]    #G8 ESPRESSO microtell corr               #HD3167 paper 
                    plot_settings[key_plot]['y_range']=None    #mask DRS ESPRESSO microtell corr               #HD3167 paper 
                    
                    
                elif gen_dic['studied_pl']==['HD3167_c']:  
                    plot_settings[key_plot]['y_range']=[0.5475,0.5485]    #mask G2
                    plot_settings[key_plot]['y_range']=[0.4214,0.4223]         #mask K5
                    plot_settings[key_plot]['y_range']=[0.425 , 0.427]         #new red
                    plot_settings[key_plot]['y_range']=None
                    
                elif gen_dic['star_name']=='HD15337':                           
                    plot_settings[key_plot]['y_range']=[0.568,0.5708]    #uncorr        
                    plot_settings[key_plot]['y_range']=[0.568-0.01,0.5708-0.01]    #corr C  
                
                elif gen_dic['star_name']=='HAT_P11': 
                    plot_settings[key_plot]['y_range']=[0.5194,0.5231]                         
    
                if gen_dic['star_name']=='55Cnc':    
                    if ('ESPRESSO' in data_dic['DI']) and (gen_dic['n_instru']==1) and plot_settings[key_plot]['norm_out']:
                        plot_settings[key_plot]['y_range']=[0.9987,1.0021]                  #uncorr
                        plot_settings[key_plot]['y_range']=[0.9996,1.0004]                 #corr
                if gen_dic['star_name']=='HD189733':
                    if plot_settings[key_plot]['col_in']=='none':plot_settings[key_plot]['y_range']=[0.999,1.001]    
    
    
            #-----------------------------------------------------
            #Contrast pipeline plot
            if (plot_prop=='Ctrstpip'):
            
                #Bornes du plot    
                if gen_dic['studied_pl']==['HD3167_c']:  
                    plot_settings[key_plot]['y_range']=[0.5475,0.5485]
                    

            #-----------------------------------------------------
            #Amplitude plot
            if (plot_prop=='amp'):
            
                #Bornes du plot  
                if gen_dic['studied_pl']=='WASP121b':      
                    plot_settings[key_plot]['y_range']=[0.475,0.56] 
    #                plot_settings[key_plot]['y_range']=[0.54,0.56]   #blue
    #                plot_settings[key_plot]['y_range']=[0.515,0.535]   #green
    #                plot_settings[key_plot]['y_range']=[0.475,0.495]  #red   
                    plot_settings[key_plot]['y_range']=[0.4,0.65]    #DG free             
            

        
            #-----------------------------------------------------
            #Area plot
            if (plot_prop=='area'):
            
                #Bornes du plot  
                if gen_dic['studied_pl']=='55Cnc_e':    
                    if ('HARPS' in data_dic['DI']) and (gen_dic['n_instru']==1):print('TODO')  
                    if ('HARPN' in data_dic['DI']) and (gen_dic['n_instru']==1):
                        if data_dic['DI']['HARPN'].keys()==['2012-12-25']:plot_settings[key_plot]['y_range']=[4.54,4.55] 
                        if data_dic['DI']['HARPN'].keys()==['2013-11-14']:plot_settings[key_plot]['y_range']=[4.55,4.56] 
                        if data_dic['DI']['HARPN'].keys()==['2013-11-28']:plot_settings[key_plot]['y_range']=[4.547,4.557]    
                        if data_dic['DI']['HARPN'].keys()==['2014-01-01']:plot_settings[key_plot]['y_range']=[4.55,4.56]   
                        if data_dic['DI']['HARPN'].keys()==['2014-01-26']:plot_settings[key_plot]['y_range']=[4.56,4.59] 
            

    
        
            #-----------------------------------------------------
            #Airmass plot
            if (plot_prop=='AM'):
            
                #Bornes du plot  
                if gen_dic['studied_pl']=='55Cnc_e':    
                    if ('HARPS' in data_dic['DI']) and (gen_dic['n_instru']==1):plot_settings[key_plot]['y_range']=[1.65,2.25]    
                    if ('HARPN' in data_dic['DI']) and (gen_dic['n_instru']==1):
                        if data_dic['DI']['HARPN'].keys()==['2012-12-25']:plot_settings[key_plot]['y_range']=[0.8,1.6] 
                        if data_dic['DI']['HARPN'].keys()==['2013-11-14']:plot_settings[key_plot]['y_range']=[0.8,2.]  
                        if data_dic['DI']['HARPN'].keys()==['2013-11-28']:plot_settings[key_plot]['y_range']=[0.8,1.6]    
                        if data_dic['DI']['HARPN'].keys()==['2014-01-01']:plot_settings[key_plot]['y_range']=[0.8,1.6]   
                        if data_dic['DI']['HARPN'].keys()==['2014-01-26']:plot_settings[key_plot]['y_range']=[0.8,1.6]   
                        if data_dic['DI']['HARPN'].keys()==['2014-02-26']:plot_settings[key_plot]['y_range']=[0.8,1.6] 
                        if data_dic['DI']['HARPN'].keys()==['2014-03-29']:plot_settings[key_plot]['y_range']=[0.8,1.6]  
                if gen_dic['studied_pl']=='WASP121b':    
                    plot_settings[key_plot]['y_range']=[0.,2.5]  


            #-----------------------------------------------------
            #Seeing plot
            if (plot_prop=='seeing'):
            
                #Bornes du plot  
                if gen_dic['studied_pl']=='55Cnc_e':    
                    plot_settings[key_plot]['y_range']=[-1.5,1.5]  
                if ('HARPN' in data_dic['DI']) and (gen_dic['n_instru']==1):
                    plot_settings[key_plot]['y_range']=[-1.5,1.5]  
                if gen_dic['studied_pl']=='WASP121b':    
                    plot_settings[key_plot]['y_range']=[0.,2.5]  
                    

        
        
            #-----------------------------------------------------
            #SNR plot
            if (plot_prop=='snr') or (plot_prop=='snr_quad'):
            
                #Bornes du plot  
                if gen_dic['studied_pl']=='55Cnc_e':    
                    if ('HARPS' in data_dic['DI']) and (gen_dic['n_instru']==1):
                        plot_settings[key_plot]['y_range']=[10.,60.]     #HARPS, SNR10 
                        plot_settings[key_plot]['y_range']=[35.,185.]    #HARPS, SNR50   
                        plot_settings[key_plot]['y_range']=[30.,220.]    #HARPS, SNR60 
                    if ('HARPN' in data_dic['DI']) and (gen_dic['n_instru']==1):
                        if data_dic['DI']['HARPN'].keys()==['2012-12-25']:
                            plot_settings[key_plot]['y_range']=[70.,140.]    #SNR10  
        #                   plot_settings[key_plot]['y_range']=[160.,310.]   #SNR50                   
        #                   plot_settings[key_plot]['y_range']=[140.,260.]    #SNR60 
                        if data_dic['DI']['HARPN'].keys()==['2013-11-14']:
                            plot_settings[key_plot]['y_range']=[100.,350.]   #SNR50 
                        if data_dic['DI']['HARPN'].keys()==['2013-11-28']:
                            plot_settings[key_plot]['y_range']=[80.,250.]   #SNR50    
                        if data_dic['DI']['HARPN'].keys()==['2014-01-01']:
                            plot_settings[key_plot]['y_range']=[80.,150.]   #SNR10
                           # plot_settings[key_plot]['y_range']=[250.,400.]  #SNR50   
                        if data_dic['DI']['HARPN'].keys()==['2014-01-26']:
                            plot_settings[key_plot]['y_range']=[70.,200.]    #SNR50   
                        if data_dic['DI']['HARPN'].keys()==['2014-02-26']:
                            plot_settings[key_plot]['y_range']=[50.,180.]    #SNR50 
                        if data_dic['DI']['HARPN'].keys()==['2014-03-29']:
                            plot_settings[key_plot]['y_range']=[200.,450.]    #SNR50
                    plot_settings[key_plot]['y_range']=[110.,220.]    #SNR102 ESPRESSO                        
                            
                if gen_dic['studied_pl']=='WASP121b':
                    plot_settings[key_plot]['y_range']=[10.,70.]    #SNR50
                elif gen_dic['studied_pl']=='GJ436_b':
                    plot_settings[key_plot]['y_range']=[27.,53.]    #SNR102
                elif gen_dic['studied_pl']=='Nu2Lupi_c':
                    plot_settings[key_plot]['y_range']=[150.,600.]  #SNR102
                # elif gen_dic['studied_pl']=='GJ9827d':
                #     plot_settings[key_plot]['y_range']=[20.,70.]    #SNR102   ESPRESSO
                elif gen_dic['studied_pl']==['HD3167_b']:
                    plot_settings[key_plot]['y_range']=[30,70.]    #SNR102+103
                elif gen_dic['studied_pl']==['HD3167_c']:
                    plot_settings[key_plot]['y_range']=[33,126.]    #SNR49
                    


            #-----------------------------------------------------
            #RV drift plot
            if (plot_prop=='RVdrift'):
            
                #Bornes du plot  
                if gen_dic['studied_pl']=='55Cnc_e': 
                    if ('HARPN' in data_dic['DI']) and (gen_dic['n_instru']==1):
                        if data_dic['DI']['HARPN'].keys()==['2012-12-25']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2013-11-14']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2013-11-28']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2014-01-01']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2014-01-26']:plot_settings[key_plot]['y_range']=[0.4,1.9]
                        if data_dic['DI']['HARPN'].keys()==['2014-02-26']:plot_settings[key_plot]['y_range']=[1.,2.3]
                        if data_dic['DI']['HARPN'].keys()==['2014-03-29']:plot_settings[key_plot]['y_range']=[-0.5,0.5]
					
        
            #-----------------------------------------------------
            #Color correction coefficients
            if (plot_prop=='colcorrmin'):
                    
                #Bornes du plot  
                if gen_dic['studied_pl']=='55Cnc_e': 
                    if ('HARPN' in data_dic['DI']) and (gen_dic['n_instru']==1):
                        if data_dic['DI']['HARPN'].keys()==['2012-12-25']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2013-11-14']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2013-11-28']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2014-01-01']:plot_settings[key_plot]['y_range']=[0.54,0.85]
                        if data_dic['DI']['HARPN'].keys()==['2014-01-26']:plot_settings[key_plot]['y_range']=[0.4,1.9]
                        if data_dic['DI']['HARPN'].keys()==['2014-02-26']:plot_settings[key_plot]['y_range']=[1.,2.3]
                        if data_dic['DI']['HARPN'].keys()==['2014-03-29']:plot_settings[key_plot]['y_range']=[-0.5,0.5]
        
        
            #----------------------------------------
            if (plot_prop=='colcorrmax'):
        
                #Bornes du plot  
                if gen_dic['studied_pl']=='55Cnc_e': 
                    if ('HARPN' in data_dic['DI']) and (gen_dic['n_instru']==1):
                        if data_dic['DI']['HARPN'].keys()==['2012-12-25']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2013-11-14']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2013-11-28']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2014-01-01']:plot_settings[key_plot]['y_range']=[1.,1.4]
                        if data_dic['DI']['HARPN'].keys()==['2014-01-26']:plot_settings[key_plot]['y_range']=[0.4,1.9]
                        if data_dic['DI']['HARPN'].keys()==['2014-02-26']:plot_settings[key_plot]['y_range']=[1.,2.3]
                        if data_dic['DI']['HARPN'].keys()==['2014-03-29']:plot_settings[key_plot]['y_range']=[-0.5,0.5]

            #----------------------------------------
            if (plot_prop=='colcorr450'):
        
                #Bornes du plot  
                if gen_dic['studied_pl']=='55Cnc_e': 
                    if ('HARPN' in data_dic['DI']) and (gen_dic['n_instru']==1):
                        if data_dic['DI']['HARPN'].keys()==['2012-12-25']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2013-11-14']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2013-11-28']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2014-01-01']:plot_settings[key_plot]['y_range']=[0.8,1.2]
                        if data_dic['DI']['HARPN'].keys()==['2014-01-26']:plot_settings[key_plot]['y_range']=[0.4,1.9]
                        if data_dic['DI']['HARPN'].keys()==['2014-02-26']:plot_settings[key_plot]['y_range']=[1.,2.3]
                        if data_dic['DI']['HARPN'].keys()==['2014-03-29']:plot_settings[key_plot]['y_range']=[-0.5,0.5]

            #----------------------------------------
            if (plot_prop=='colcorr550'):
        
                #Bornes du plot  
                if gen_dic['studied_pl']=='55Cnc_e': 
                    if ('HARPN' in data_dic['DI']) and (gen_dic['n_instru']==1):
                        if data_dic['DI']['HARPN'].keys()==['2012-12-25']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2013-11-14']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2013-11-28']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2014-01-01']:plot_settings[key_plot]['y_range']=[1.03,1.1]
                        if data_dic['DI']['HARPN'].keys()==['2014-01-26']:plot_settings[key_plot]['y_range']=[0.4,1.9]
                        if data_dic['DI']['HARPN'].keys()==['2014-02-26']:plot_settings[key_plot]['y_range']=[1.,2.3]
                        if data_dic['DI']['HARPN'].keys()==['2014-03-29']:plot_settings[key_plot]['y_range']=[-0.5,0.5]
                

            #----------------------------------------
            if (plot_prop=='colcorr650'):
        
                #Bornes du plot  
                if gen_dic['studied_pl']=='55Cnc_e': 
                    if ('HARPN' in data_dic['DI']) and (gen_dic['n_instru']==1):
                        if data_dic['DI']['HARPN'].keys()==['2012-12-25']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2013-11-14']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2013-11-28']:plot_settings[key_plot]['y_range']=[-1.,1.]
                        if data_dic['DI']['HARPN'].keys()==['2014-01-01']:plot_settings[key_plot]['y_range']=[0.6,1.4]
                        if data_dic['DI']['HARPN'].keys()==['2014-01-26']:plot_settings[key_plot]['y_range']=[0.4,1.9]
                        if data_dic['DI']['HARPN'].keys()==['2014-02-26']:plot_settings[key_plot]['y_range']=[1.,2.3]
                        if data_dic['DI']['HARPN'].keys()==['2014-03-29']:plot_settings[key_plot]['y_range']=[-0.5,0.5]
                

    
            #-----------------------------------------------------
            #Ratio of lobe FWHM to core FWHM
            if (plot_prop=='FWHM_l2c'):
            
                #Bornes du plot
        #        plot_settings[key_plot]['y_range']=[1.5,2.2]    
                plot_settings[key_plot]['y_range']=[1.655,1.995]    #avec erreurs
                plot_settings[key_plot]['y_range']=[1.705,1.945]    #sans erreurs
                plot_settings[key_plot]['y_range']=[1.15,1.45]    #sans erreurs
    
                if gen_dic['studied_pl']=='GJ436_b':
                    plot_settings[key_plot]['y_range']=[1.0015,1.0035]  
                    

            #-----------------------------------------------------
            #Ratio of lobe contrast to core amplitude
            if (plot_prop=='amp_l2c'):
            
                #Bornes du plot  
        #        plot_settings[key_plot]['y_range']=[0.3,0.7]      
                plot_settings[key_plot]['y_range']=[0.476,0.569]       #avec erreurs  
                plot_settings[key_plot]['y_range']=[0.481,0.559]       #sans erreurs
                plot_settings[key_plot]['y_range']=[0.42,0.65]       #sans erreurs
    
                if gen_dic['studied_pl']=='GJ436_b':
                    plot_settings[key_plot]['y_range']=[0.9945,0.9975] 
                    
                    
                
            #-----------------------------------------------------
            #RV shift bewtween lobe and core gaussian RV centroid
            if (plot_prop=='rv_l2c'):
            
                #Bornes du plot  
        #        plot_settings[key_plot]['y_range']=[0.3,0.7]      
                plot_settings[key_plot]['y_range']=[-0.059,0.059]      #avec erreurs
                plot_settings[key_plot]['y_range']=[-0.039,0.039]      #sans erreurs
                plot_settings[key_plot]['y_range']=[0.1,0.55]      #sans erreurs
    
                if gen_dic['studied_pl']=='GJ436_b':
                    plot_settings[key_plot]['y_range']=[-0.001,-0.0004] 
                    
    
            #-----------------------------------------------------
            #Lobe FWHM 
            if (plot_prop=='FWHM_lobe'):
            
                #Bornes du plot
                plot_settings[key_plot]['y_range']=[32.9,34.5]   

    
            #-----------------------------------------------------
            #Lobe amplitude
            if (plot_prop=='amp_lobe'):
            
                #Bornes du plot  
                plot_settings[key_plot]['y_range']=[0.225,0.32]     
    #                plot_settings[key_plot]['y_range']=[0.295,0.312]   #blue
    #                plot_settings[key_plot]['y_range']=[0.27,0.29]   #green
    #                plot_settings[key_plot]['y_range']=[0.23,0.25]   #red
                plot_settings[key_plot]['y_range']=[0.15,0.45]    #DG free

            #-----------------------------------------------------
            #Lobe RV
            if (plot_prop=='RV_lobe'):
            
                #Bornes du plot  
                plot_settings[key_plot]['y_range']=[38.5,39.1]      

            #-----------------------------------------------------
            #True contrast
            if (plot_prop=='true_ctrst'):
    
                #Bornes du plot         
                if gen_dic['studied_pl']==['GJ436_b']:
                    plot_settings[key_plot]['y_range']=[0.15,0.45]
                    # plot_settings[key_plot]['y_range']=[0.,1.]
    

    
    
    
    '''
    Plotting aligned disk-integrated profiles
        - in star rest frame
        - all profiles from a given visit together
        - spectra may have been corrected for color balance or not
          CCF have not yet been scaled in flux, which is done within the routine to allow better comparing them (his requires the transit scaling routine to have been run so that continuum pixels are defined)
    '''    
    if (plot_dic['all_DI_data']!=''):  
        key_plot = 'all_DI_data'
        plot_settings[key_plot]={}
        
        #Data type
        plot_settings[key_plot]['data_type']='CCF' 
        
        #Oplot continuum pixels
        plot_settings[key_plot]['plot_cont']=True # &  False

        #Visits to plot
        if gen_dic['studied_pl']=='55Cnc_e':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-02-05']}
        elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-28','2019-04-29']} 
        elif gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}     
        elif gen_dic['studied_pl']=='HD209458b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-07-20','2019-09-11']}
        elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']}  
        elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-20']} 
        elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-03-18']} 

        #Exposures to plot
        #    - indexes are relative to global tables
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['iexp_plot']={'ESPRESSO':{
                '2018-10-31':np.arange(2,37,dtype=int),
                '2018-09-03':np.arange(1,20,dtype=int)}}

        #Choice of orders to plot
        #    - for 2D spectra (leave empty to plot all orders)
        plot_settings[key_plot]['orders_to_plot']=[10,11,12,13]
        # plot_settings[key_plot]['orders_to_plot']=[0,1,2,3]
        # plot_settings[key_plot]['orders_to_plot']=[]

        #Bornes du plot
        if gen_dic['studied_pl']=='55Cnc_e':
            plot_settings[key_plot]['x_range']=[6,50]
            if ('HARPS' in data_dic['DI']) and (gen_dic['n_instru']==1):plot_settings[key_plot]['y_range']=[0.,2e7]        
            if ('HARPN' in data_dic['DI']) and (gen_dic['n_instru']==1):
                if data_dic['DI']['HARPN'].keys()==['2012-12-25']:plot_settings[key_plot]['y_range']=[5e6,6e7]   
            if ('ESPRESSO' in data_dic['DI']) and (gen_dic['n_instru']==1):
                plot_settings[key_plot]['x_range']=[-20,20]                    
                plot_settings[key_plot]['y_range']=[0.,1.1]  
        if gen_dic['studied_pl']==['HD3167_c']:
            plot_settings[key_plot]['x_range']=[-4,42.] 
            plot_settings[key_plot]['x_range']=[-25,25.]   
            plot_settings[key_plot]['y_range']=[0.4,1.05]                 
        elif gen_dic['studied_pl']=='WASP121b':
            plot_settings[key_plot]['x_range']=[-15.,90.]  
            plot_settings[key_plot]['y_range']=[0.84,1.05]   
        elif gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['x_range']=[-150.,150.]  
            plot_settings[key_plot]['y_range']=[0.35,1.05]   
        elif gen_dic['studied_pl']=='HD209458b':
            plot_settings[key_plot]['x_range']=[-30.,30.]  
            plot_settings[key_plot]['y_range']=[0.4,1.05]       
        elif gen_dic['studied_pl']=='GJ436_b':
            plot_settings[key_plot]['x_range']=[-20.,20.]  
            plot_settings[key_plot]['y_range']=[0.6,1.2]  
        elif gen_dic['studied_pl']==['HD3167_b']: plot_settings[key_plot]['y_range']=[0.3,1.1]
        elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['y_range']=[0.3,1.1]
        elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['y_range']=[0.35,1.1]
        elif gen_dic['studied_pl']=='GJ9827d':
            plot_settings[key_plot]['y_range']=[0.3,1.1]      #ESPRESSO
            plot_settings[key_plot]['y_range']=[0.55,1.05]      #HARPS        
        
        
        
        
        
        
    
    
    
    
    ################################################################################################################    
    #%% Input light curves
    #    - used to rescale the flux
    #    - one plot for each band, each instrument, all visits
    #    - overplotting all visits together is useful to quickly compare the exposure ranges, the detection of their local stellar profiles, and determine binning windows
    ################################################################################################################ 
    if (plot_dic['input_LC']!=''):
        key_plot = 'input_LC'
        plot_settings[key_plot]={}
        
        #Margins
        plot_settings[key_plot]['margins']=[0.15,0.12,0.8,0.6]   #ANTARESS I, HD209 oblate        
        
        #Margins        
        plot_settings[key_plot]['font_size']=18   #ANTARESS I, mock, multi-tr
        plot_settings[key_plot]['font_size']=14   #ANTARESS I, HD209 oblate        
        
        #Line width
        plot_settings[key_plot]['lw_plot'] = 1.

        #Plot exposure-averaged light curves used for scaling
        plot_settings[key_plot]['plot_LC_exp'] = True & False
        
        #Plot HR input light curves
        plot_settings[key_plot]['plot_LC_HR'] = True
        
        #Plot raw imported light curve, if available
        plot_settings[key_plot]['plot_LC_imp'] = True

        #Print exposure indexes
        plot_settings[key_plot]['plot_expid']=True & False

        #Print visit names
        plot_settings[key_plot]['plot_vis']=True & False

        #Indexes of bands to plot
        #    - default is all
        plot_settings[key_plot]['idx_bands']=[]        

        #Gap between visits light curves
        # plot_settings[key_plot]['lc_gap']=1e-2
        # plot_settings[key_plot]['lc_gap']=7e-4        
        
        #Visits to plot
        if gen_dic['studied_pl']=='55Cnc_e':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-02-05']}
        if gen_dic['studied_pl']==['HD3167_c']:
            plot_settings[key_plot]['visits_to_plot']={
    #        'HARPS':['2012-01-27','2012-02-27','2012-02-13','2012-03-15'],
    #        'HARPN':['2012-12-25','2013-11-14','2013-11-28','2014-01-26','2014-02-26','2014-03-29'],
    #        'HARPN':['2014-01-26'],
    #         'SOPHIE':['2012-02-02','2012-02-03','2012-02-05','2012-02-17','2012-02-19','2012-02-22','2012-02-25','2012-03-02','2012-03-24','2012-03-27','2013-03-03']
    #        'SOPHIE':['2012-02-02']
            
            'HARPN':['2016-10-01'],    #HD3167c 
    
            
    #        'binned':['all_HARPSS']
    #        'binned':['all_HARPS_adj']
    #        'binned':['all_HARPS_adj','all_HARPS_adj2'],
    #        'binned':['2012-01-27_binned','2012-02-27_binned','2012-02-13_binned','2012-03-15_binned']
    #        'binned':['good_HARPSN','good_HARPSN_adj']
    #        'binned':['good_HARPSN_adj']
    #        'binned':['HARPS_HARPSN_binHARPS']
    #        'binned':['HARPS_HARPSN_binHARPSN']
    #        'binned':['best_HARPSN_adj']
    #        'binned':['best_HARPSN_adj_short','best_HARPSN_adj_long']
    
                }

        elif gen_dic['studied_pl']=='Kelt9b':
            plot_settings[key_plot]['visits_to_plot']={           
                'HARPN':['31-07-2017']
            }      
        elif gen_dic['studied_pl']=='WASP121b':
            plot_settings[key_plot]['visits_to_plot']={           
                'HARPS':['14-01-18','09-01-18','31-12-17'],   
#                'HARPS':['31-12-17'], 
                'binned':['HARPS-binned']
            }              
        elif gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03'],'binned':['ESP_binned']}      

        elif gen_dic['studied_pl']=='WASP127b':
            plot_settings[key_plot]['visits_to_plot']={'HARPS':['2017-03-20','2018-03-31','2018-02-13','2017-02-28']}
            
        elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-28','2019-04-29']}
        elif gen_dic['star_name']=='HD209458':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20190720']} 
        elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']}
        elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-20']} 
        elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-03-18']} 
        
        
        
        #Color dictionary for each original visit
        #    - binned visits are in black
        if gen_dic['studied_pl']=='55Cnc_e':
            # plot_settings[key_plot]['color_dic={
            #     'all_HARPSS':'dodgerblue',            
            #     'all_HARPS_adj':'dodgerblue',
            #     'all_HARPS_adj2':'cyan',            
            #     'best_HARPSN_adj':'red',
            #     'good_HARPSN_adj':'orange',
            #     'HARPS_HARPSN_binHARPS':'black',
            #     'HARPS_HARPSN_binHARPSN':'black',
            #     'best_HARPSN_adj_short':'lime',
            #     'best_HARPSN_adj_long':'red', 
            #     }
            plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2020-02-05':'blue'}}
        elif gen_dic['studied_pl']=='Kelt9b':
            plot_settings[key_plot]['color_dic']={
                  '31-07-2017':'dodgerblue'
                    }    
        elif gen_dic['studied_pl']=='WASP121b':
            plot_settings[key_plot]['color_dic']={
                  '14-01-18':'dodgerblue',
                  '09-01-18':'green',
                  '31-12-17':'red',
                  'HARPS-binned':'black'
                    }
        elif gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'}}  
        elif gen_dic['studied_pl']=='WASP127b':
            plot_settings[key_plot]['color_dic']={'HARPS':{'2017-03-20':'blue','2018-03-31':'green','2018-02-13':'orange','2017-02-28':'red'}} 

        elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-02-28':'dodgerblue','2019-04-29':'red'}}   
        elif gen_dic['studied_pl']=='HD209458b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-07-20':'dodgerblue','2019-09-11':'red'}}
        elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-10-09':'dodgerblue'}} 
        elif gen_dic['studied_pl']==['HD3167_c']:plot_settings[key_plot]['color_dic']={'HARPN':{'2016-10-01':'dodgerblue'}}
        elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-02-20':'dodgerblue'}} 
        elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2020-03-18':'dodgerblue'}} 
        elif gen_dic['studied_pl']=='GJ9827d':
            plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-08-25':'dodgerblue'},'HARPS':{'2018-08-18':'dodgerblue','2018-09-18':'red'}} 
        
        
        #Bornes du plot
        if gen_dic['studied_pl']=='GJ436_b':
            plot_settings[key_plot]['x_range']=[-0.13,0.09]   #GJ436b    
            plot_settings[key_plot]['x_range']=[-0.01,0.01]   #zoom transit   
            plot_settings[key_plot]['y_range']=[0.993,1.005]  #None
            
            plot_settings[key_plot]['x_range']=[-0.036,0.048]   #ESPRESSO 
            plot_settings[key_plot]['y_range']=[0.99,1.0015]  
            plot_settings[key_plot]['lc_gap']=2e-3
            
        if gen_dic['studied_pl']=='55Cnc_e':
            
            #HARPS
            plot_settings[key_plot]['x_range']=[-0.07,0.12]
            plot_settings[key_plot]['y_range']=[0.9994,1.0002]   #single visit
            
            #HARPS-N
            plot_settings[key_plot]['x_range']=[-0.07,0.24]      #'2012-12-25'       
            plot_settings[key_plot]['x_range']=[-0.18,0.09]      #'2013-11-14'       
            plot_settings[key_plot]['x_range']=[-0.16,0.12]      #'2013-11-28'       
            plot_settings[key_plot]['x_range']=[-0.14,0.1]       #'2014-01-01'       
            plot_settings[key_plot]['x_range']=[-0.07,0.13]      #'2014-01-26'       
            plot_settings[key_plot]['x_range']=[-0.09,0.1]       #'2014-02-26'       
            plot_settings[key_plot]['x_range']=[-0.11,0.07]      #'2014-03-29' 
            
            plot_settings[key_plot]['x_range']=[-0.18,0.24]      #7 nuits ensemble
            plot_settings[key_plot]['y_range']=[0.99955,1.0007]
            
            plot_settings[key_plot]['x_range']=[-0.16,0.13]       #5 bonnes nuits HARPS-N
            plot_settings[key_plot]['y_range']=[0.99955,1.0005] 
          
            #SOPHIE, all visits  
            plot_settings[key_plot]['x_range']=[-0.12,0.2]
            plot_settings[key_plot]['y_range']=[0.99955,1.0015]
            
            #ESPRESSO
            plot_settings[key_plot]['x_range']=[-0.11,0.16]
            plot_settings[key_plot]['y_range']=[0.99952,1.0002]
            
        if gen_dic['studied_pl']==['HD3167_c']:
            plot_settings[key_plot]['x_range']=[-0.0075,0.0075]    
            plot_settings[key_plot]['y_range']=[0.99885,1.0002]         

        if gen_dic['studied_pl']=='Kelt9b':
            plot_settings[key_plot]['x_range']=[-0.16,0.19]    
            plot_settings[key_plot]['y_range']=[0.975,1.02]    
        if gen_dic['studied_pl']=='WASP121b':
            plot_settings[key_plot]['x_range']=[-0.16,0.19]    
            plot_settings[key_plot]['y_range']=[0.975,1.02] 
        elif gen_dic['studied_pl']=='WASP76b':
        #     plot_settings[key_plot]['x_range']=[-0.16,0.19]   
            plot_settings[key_plot]['y_range']=[0.965,1.005]  
            plot_settings[key_plot]['lc_gap']=5e-3
        elif gen_dic['studied_pl']=='WASP127b':
            plot_settings[key_plot]['x_range']=[-0.05,0.05]   
            plot_settings[key_plot]['y_range']=[0.95,1.005] 
        elif gen_dic['studied_pl']=='HD209458b':
            plot_settings[key_plot]['x_range']=[-0.03,0.042]   
            plot_settings[key_plot]['y_range']=[0.975,1.005] 
            plot_settings[key_plot]['lc_gap']=3e-3
        elif gen_dic['studied_pl']==['HD3167_b']:
            plot_settings[key_plot]['x_range']=[-0.075,0.112]             
            plot_settings[key_plot]['y_range']=[0.999,1.001]   
        elif gen_dic['studied_pl']=='Corot7b':
            plot_settings[key_plot]['x_range']=[-0.08,0.17]             
            plot_settings[key_plot]['y_range']=[0.999,1.001]  
        elif gen_dic['studied_pl']=='Nu2Lupi_c':
            plot_settings[key_plot]['x_range']=[-0.007,0.007]             
            plot_settings[key_plot]['y_range']=[0.999,1.0003]  
        elif gen_dic['studied_pl']=='GJ9827d':
            plot_settings[key_plot]['x_range']=[-0.025,0.025]         #ESPRESSO      
            plot_settings[key_plot]['y_range']=[0.999,1.00025]  
            plot_settings[key_plot]['x_range'] = [-0.04 ,  0.02 ]
            plot_settings[key_plot]['y_range']=[0.9983,1.00025]  
        elif gen_dic['star_name']=='HIP41378':
            plot_settings[key_plot]['x_range']=[-0.001,0.0015]        
            plot_settings[key_plot]['y_range']=[0.998,1.00025]  
        elif gen_dic['star_name']=='Altair':  
            plot_settings[key_plot]['x_range']=[-0.03426535088,0.03426535088]
            plot_settings[key_plot]['y_range']=[0.995,1.000]  
        elif gen_dic['star_name']=='HD89345':  
            plot_settings[key_plot]['x_range']=[-0.012,0.017]  

    




    ##################################################################################################
    #%% Effective scaling light curves     
    #   - plotting effective light curves used to rescale the flux
    #   - for input spectra only
    #   - one plot for each instrument, each visit, for all wavelengths in a given list
    #   - these scaling light curves are only defined at the phase of observed exposures, and for the corresponding wavelength, which might differ slightly from one exposure to the next
    ##################################################################################################
    if (plot_dic['spectral_LC']!=''):
        key_plot = 'spectral_LC'
        plot_settings[key_plot]={}

        #Margins
        plot_settings[key_plot]['margins']=[0.15,0.12,0.8,0.6]   #ANTARESS I, WASP76
        
        #Wavelengths to plot (A)
        #    - light curve will be plotted for the closest wavelength
        plot_settings[key_plot]['wav_LC']=[4000.,5000.,6000.,7000.]

        #Instruments and visits to plot
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03'],'binned':['ESP_binned']}      

        #Color dictionary for each wavelength
        #    - colors are associated with the wavelength index in wav_LC
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['color_dic']={0:'black',1:'dodgerblue',2:'limegreen',3:'orange',4:'red'}  

   	
        #Bornes du plot
        if gen_dic['star_name']=='WASP76':
        #     plot_settings['x_range']=[-0.16,0.19]   
            plot_settings[key_plot]['y_range']=[0.981,1.003]          
        
        
        
        
        
        
        
        
        
        
        
        
    '''
    2D maps of disk-integrated profiles in star rest frame 
        - allows visualizing the exposures used to build the master, and the ranges excluded because of planetary contamination
    '''
    if (plot_dic['map_DI_prof']!=''):  
        key_plot = 'map_DI_prof'
        plot_settings[key_plot]={}      
        
        #Reverse image
        plot_settings[key_plot]['reverse_2D']=False        
        
        #Choose profiles to plot
        #    - 'corr' : profiles have been corrected
        #    - 'aligned' : profiles have been aligned, after being corrected if requested
        #    - 'scaled' : profiles have been scaled, after being corrected and aligned if requested
        plot_settings[key_plot]['step']='scaled'        
        
        #Choice of visits to be plotted
        if gen_dic['studied_pl']=='55Cnc_e':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-02-05']} 
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']} 
        elif gen_dic['studied_pl']=='HD209458b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-07-20','2019-09-11']} 
        elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-28','2019-04-29']} 
        elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']} 
        elif gen_dic['studied_pl']==['HD3167_c']:plot_settings[key_plot]['visits_to_plot']={'HARPN':['2016-10-01']} 
        elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-20']} 
        elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-03-18']} 
            
        #Choice of orders to plot
        #    - for 2D spectra (leave empty to plot all orders)
        # plot_settings[key_plot]['orders_to_plot']=[10,11,12,13]
        # plot_settings[key_plot]['orders_to_plot']=[0,1,2,3]
       
        #Overplot RV(pl/star) model 
        plot_settings[key_plot]['theoRVpl_HR']=True       &   False         

        #Overplot excluded planetary ranges
        plot_settings[key_plot]['plot_plexc']=True  #&  False        

        #Color range    
        # if gen_dic['studied_pl']=='WASP76b':
        #     v_range_comm=[-0.002  ,  0.02]
        #     plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2018-10-31':v_range_comm,'2018-09-03':v_range_comm},'binned':{'ESP_binned':v_range_comm}}         
        if gen_dic['star_name']=='HAT_P11':
            v_range_comm=[0.98,1.02]
            plot_settings[key_plot]['v_range_all']={'CARMENES_VIS':{'20170807':v_range_comm,'20170812':v_range_comm}} 
        elif gen_dic['star_name']=='WASP156':
            v_range_comm=[0.99,1.04]
            plot_settings[key_plot]['v_range_all']={'CARMENES_VIS':{'20190928':v_range_comm,'20191025':v_range_comm,'20191210':v_range_comm}} 

        
        #Reversed image 
        if plot_settings[key_plot]['reverse_2D']:   
            plot_settings[key_plot]['margins']=[0.15,0.12,0.95,0.84]          
                
            if gen_dic['studied_pl']=='WASP76b':
                plot_settings[key_plot]['x_range_all']={'ESPRESSO':{'2018-10-31':[-0.007,0.0061]}}
                plot_settings[key_plot]['y_range']=[-150.,150.]
            
        else:        
            if gen_dic['studied_pl']=='WASP76b':
                # plot_settings[key_plot]['x_range']=[-150.,150.]
                y_range_comm=[-0.08  ,  0.09]
                plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2018-10-31':y_range_comm,'2018-09-03':y_range_comm},'binned':{'ESP_binned':y_range_comm}}         
            elif gen_dic['studied_pl']=='55Cnc_e':
                y_range_comm=[-0.1  ,  0.15]
                plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2020-02-05':y_range_comm}} 
            # elif gen_dic['studied_pl']=='GJ436_b':
            #     y_range_comm=[-0.1  ,  0.15]
            #     visits_to_plot={'ESPRESSO':{'2019-02-28':y_range_comm,'2019-04-29':y_range_comm}}        
            elif gen_dic['studied_pl']==['HD3167_c']:plot_settings[key_plot]['y_range_all']={'HARPN':{'2016-10-01':[-0.0075  ,  0.006]}}          
        
        
        
        
        
        
    '''
    2D maps of binned disk-integrated stellar profiles
    '''
    if (plot_dic['map_DIbin']!=''):
        key_plot = 'map_DIbin'
        plot_settings[key_plot]={} 

        #Choice of visits to be plotted
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03','binned']}  

        #Choice of orders to plot
        #    - for 2D spectra (leave empty to plot all orders)
        plot_settings[key_plot]['orders_to_plot']=[113,114]   #sodium doublet

        #Color range 
        if gen_dic['studied_pl']=='WASP76b':
            v_range_comm=[3,7]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2018-10-31':v_range_comm,'2018-09-03':v_range_comm}}  

        #Ranges
        if gen_dic['studied_pl']=='WASP76b':
            # plot_settings[key_plot]['x_range']=[5880.,5905.]
            # plot_settings[key_plot]['x_range']=[-150,150]
            plot_settings[key_plot]['x_range']=[-100.,100]
            y_range_comm=[-0.048  ,  0.048]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2018-10-31':y_range_comm,'2018-09-03':y_range_comm}}   

                
        #------------------------------------
     
        
     
        
     
        
     
        
     
        





    '''
    2D maps of 1D disk-integrated stellar profiles
    '''
    if (plot_dic['map_DI_1D']!=''):
        key_plot = 'map_DI_1D'
        plot_settings[key_plot]={}

        #Choice of visits to be plotted
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}  

        #Color range
        if gen_dic['studied_pl']=='WASP76b':
            v_range_comm=[3,7]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2018-10-31':v_range_comm,'2018-09-03':v_range_comm}} 

        #Ranges
        if gen_dic['studied_pl']=='WASP76b':
            # plot_settings[key_plot]['x_range']=[5880.,5905.]
            # plot_settings[key_plot]['x_range']=[-150,150]
            plot_settings[key_plot]['x_range']=[-100.,100]
            y_range_comm=[-0.048  ,  0.048]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2018-10-31':y_range_comm,'2018-09-03':y_range_comm}}         
        
     

    





    ################################################################################################################  
    #%% 2D maps of residual profiles 
    #    - in stellar rest frame
    ################################################################################################################  
    if (plot_dic['map_res_prof']!=''):
        key_plot = 'map_res_prof'
        plot_settings[key_plot]={}

        #Reverse image
        plot_settings[key_plot]['reverse_2D']=False  

        #Choice of visits to be plotted
        if gen_dic['studied_pl']=='WASP_8b':
            plot_settings[key_plot]['visits_to_plot']={'HARPS':['2008-10-04']}
        elif gen_dic['star_name']=='55Cnc':
            plot_settings[key_plot]['visits_to_plot']={
                #            'HARPN':['2014-01-01','2012-12-25','2014-02-26','2014-03-29','2013-11-28','2013-11-14','2014-01-26']
    #            'HARPN':['2014-03-29','2013-11-28','2013-11-14','2014-01-26']
              'SOPHIE':['2012-02-02','2012-02-03','2012-02-05','2012-02-17','2012-02-19','2012-02-22','2012-02-25','2012-03-02','2012-03-24','2012-03-27','2013-03-03']
    #        'SOPHIE':['2012-02-02']
    
    #            'binned':['best_HARPSN_adj_short','best_HARPSN_adj_long']
    #            'binned':['all_HARPSS','all_HARPS_adj','all_HARPS_adj2','good_HARPSN','good_HARPSN_adj','HARPS_HARPSN_binHARPS','best_HARPSN_adj']
            }
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20200205','20210121','20210124']}

        elif gen_dic['studied_pl']=='WASP121b':
            plot_settings[key_plot]['visits_to_plot']={'HARPS':['14-01-18','09-01-18','31-12-17'],
                            'binned':['HARPS-binned']}
#            plot_settings[key_plot]['visits_to_plot']={'HARPS':['14-01-18','09-01-18'],
#                            'binned':['HARPS-binned-2018']}
#            plot_settings[key_plot]['visits_to_plot']={'HARPS':['14-01-18','09-01-18','31-12-17']}
#            plot_settings[key_plot]['visits_to_plot']={'HARPS':['31-12-17']}
        elif gen_dic['studied_pl']=='Kelt9b':
            plot_settings[key_plot]['visits_to_plot']={'HARPN':['31-07-2017']}    
        elif gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03'],'binned':['ESP_binned']} 
        elif gen_dic['studied_pl']=='WASP127b':plot_settings[key_plot]['visits_to_plot']={'HARPS':['2017-03-20','2018-03-31','2018-02-13','2017-02-28']} 
        elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-28','2019-04-29']} 
        elif gen_dic['studied_pl']=='HD209458b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-07-20','2019-09-11']}   
        elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']}
        elif gen_dic['studied_pl']==['HD3167_c']:plot_settings[key_plot]['visits_to_plot']={'HARPN':['2016-10-01']}
        elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-20']} 
        elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-03-18']} 
        
        
        #Choice of orders to plot
        #    - for 2D spectra (leave empty to plot all orders)
        # plot_settings[key_plot]['orders_to_plot']=[10,11,12,13]
        # plot_settings[key_plot]['orders_to_plot']=[50,80,100,120,150]
        # plot_settings[key_plot]['orders_to_plot']=[105,107,109,111,113,115]
        # plot_settings[key_plot]['orders_to_plot']=[113]   #sodium doublet
        # if gen_dic['studied_pl']==['HD3167_b']:
        #     plot_settings[key_plot]['orders_to_plot']=range(76,111)        
        
        #Overplot RV model at the phases of the observations 
        plot_settings[key_plot]['plot_theoRV']=False
    
        #Overplot RV model along the full transit chord (CCF only) 
        plot_settings[key_plot]['theoRV_HR']=True #  &   False
    
        #Overplot RV(pl/star) model 
        plot_settings[key_plot]['theoRVpl_HR']=True       &   False         


        #Overplot lines of stellar CCF mask
        plot_settings[key_plot]['CCF_mask_star']=True  &  False

        #Plot zero line markers
        plot_settings[key_plot]['plot_zermark']= True #  False

        #Overplot measured RVs
        #    - 'none': no plot, 'det': detected CCFs, 'all': all CCFs
        plot_settings[key_plot]['plot_measRV']='det'

        #Color range
        if gen_dic['studied_pl']=='WASP_8b':
            plot_settings[key_plot]['v_range_all']={
                'HARPS':{'2008-10-04':[-0.005  ,  0.018]}
                }
#        elif gen_dic['studied_pl']=='GJ436_b':
#            v_range=[-0.001  ,  0.0075] #GJ436b
#            v_range=[-0.0013  ,  0.0075] #GJ436b
        elif gen_dic['star_name']=='55Cnc':
            v_range_comm=[-0.00075,  0.00075]     #midrange
    #        v_range_comm=[-0.0003,  0.0007]       #smallrange
            v_range_comm=[-0.0002  ,  0.0006]    #common range, toutes nuits et instruments
            
            plot_settings[key_plot]['v_range_all']={
                'HARPS':{
                    '2012-01-27':v_range_comm,'2012-02-27':v_range_comm,'2012-02-13':v_range_comm,'2012-03-15':v_range_comm
                    },
                'HARPN':{
                      '2012-12-25':[-0.00043  ,  0.00091],
                      '2013-11-14':[-0.00062  ,  0.00070],
                      '2013-11-28':[-0.00059  ,  0.00086],
    #                 '2014-01-01':[-0.00033  ,  0.00065],
                      '2014-01-26':[-0.00089  ,  0.00111],
    #                 '2014-02-26':[-0.00084  ,  0.00146],
                      '2014-02-26':[-0.00084  ,  0.00084], 
                      '2014-03-29':[-0.00039  ,  0.00068]                 
                }, 
                'SOPHIE':{
                        '2012-02-02':v_range_comm,'2012-02-03':v_range_comm,'2012-02-05':v_range_comm,'2012-02-17':v_range_comm,'2012-02-19':v_range_comm,'2012-02-22':v_range_comm,
                        '2012-02-25':v_range_comm,'2012-03-02':v_range_comm,'2012-03-24':v_range_comm, '2012-03-27':v_range_comm, '2013-03-03':v_range_comm,
                    },                
                'binned':{
                    'all_HARPSS':v_range_comm,'all_HARPS_adj':v_range_comm,'all_HARPS_adj2':v_range_comm, '2012-01-27_binned':v_range_comm,
                    '2012-02-27_binned':v_range_comm, '2012-02-13_binned':v_range_comm, '2012-03-15_binned':v_range_comm,'good_HARPSN': v_range_comm,
                    'good_HARPSN_adj':v_range_comm,'HARPS_HARPSN_binHARPS':v_range_comm,'best_HARPSN_adj':v_range_comm,'best_HARPSN_adj_short':v_range_comm,
                    'best_HARPSN_adj_long':v_range_comm,'HARPS-binned-2018':v_range_comm,                    
                }}
            
            v_range_comm = [-0.0002,0.0006]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'20200205':v_range_comm,'20210121':v_range_comm,'20210124':v_range_comm}}
            

        elif gen_dic['studied_pl']=='WASP121b':
            v_range_comm=[-0.0035  ,  0.025]
#            v_range_comm=[-0.0035  ,  0.0035]     #affichage dispersion horst-transit
            plot_settings[key_plot]['v_range_all']={'HARPS':{'14-01-18':v_range_comm,
                                  '09-01-18':v_range_comm,
                                  '31-12-17':v_range_comm},
                          'binned':{'HARPS-binned':v_range_comm,'HARPS-binned-2018':v_range_comm}                
                    }
        elif gen_dic['studied_pl']=='Kelt9b':
            v_range_comm=[-0.0025  ,  0.009]
            plot_settings[key_plot]['v_range_all']={'HARPN':{'31-07-2017':v_range_comm}}    
        elif gen_dic['studied_pl']=='WASP76b':
            v_range_comm=[-0.01  ,  0.018]
            v_range_comm=[-0.0012  ,  0.0012]
            # v_range_comm=[-1000.,1500.]  
            # v_range_comm=[-500.,500.]   #Na doublet
            plot_settings[key_plot]['sc_fact']=1e-5
            v_range_comm=[-1.35e5,2.51e5]     #CCFs manuelles  
            # v_range_comm=[-0.2e5,0.2e5]     #zoom residus
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2018-10-31':v_range_comm,'2018-09-03':v_range_comm},'binned':{'ESP_binned':v_range_comm}}         
            plot_settings[key_plot]['v_range_all']={}
        elif gen_dic['studied_pl']=='WASP127b':
            v_range_comm = [-0.0025  ,  0.009]
            # plot_settings[key_plot]['v_range_all']={'HARPS':{'2017-03-20':v_range_comm,'2018-03-31':v_range_comm,'2018-02-13':v_range_comm,'2017-02-28':v_range_comm}} 
        elif gen_dic['studied_pl']=='GJ436_b':
            v_range_comm = [-0.001  ,  0.0073]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2019-02-28':v_range_comm,'2019-04-29':v_range_comm}} 
            plot_settings[key_plot]['sc_fact']=1e3
        elif gen_dic['star_name']=='HD209458':
            v_range_comm = [-0.02  , 0.03]
            # v_range_comm = [-1e-3  ,  1e-3]    #residuals
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'20190720':v_range_comm,'20190911':v_range_comm}} 
            plot_settings[key_plot]['sc_fact10']=2
        elif gen_dic['studied_pl']=='Corot7b':
            v_range_comm = [-0.005,  0.004]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2019-02-20':v_range_comm}} 
            plot_settings[key_plot]['sc_fact']=1e3
        elif gen_dic['studied_pl']=='GJ9827d':
            # v_range_comm = [-5e-4,  5e-4]    #résidus
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2019-08-25':[-5e-4,  1.3e-3]},'HARPS':{'2018-08-18':[-3e-3,  4e-3],'2018-09-18':[-3e-3,  4e-3]}} 
            plot_settings[key_plot]['sc_fact']=1e3 
        elif gen_dic['studied_pl']=='GJ9827b':
            plot_settings[key_plot]['v_range_all']={'HARPS':{vis:[-3e-3,  3e-3] for vis in ['2018-08-04','2018-08-15','2018-09-18','2018-09-19']}} 
            plot_settings[key_plot]['sc_fact']=1e3             
        elif gen_dic['studied_pl']=='Nu2Lupi_c':
            v_range_comm = [-3e-4,  8e-4]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2020-03-18':v_range_comm}} 
            plot_settings[key_plot]['sc_fact']=1e4        
        elif gen_dic['studied_pl']==['HD3167_b']:
            if data_dic['Res']['type'][inst]=='CCF':
                v_range_comm = [-1.5e-3,2.e-3]
                v_range_comm = [-1e-3,1.3e-3]
                # v_range_comm = [-1e-3,1e-3]     #zoom residus
            else:
                v_range_comm = [-100.,100.]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2019-10-09':v_range_comm}}  
            # plot_settings[key_plot]['v_range_all']={}
        elif gen_dic['studied_pl']==['HD3167_c']:
    #        plot_settings[key_plot]['v_range_all']={'HARPN':{'2016-10-01':[-0.0015  ,  0.0018]}}
    #        plot_settings[key_plot]['v_range_all']={'HARPN':{'2016-10-01':[-0.001  ,  0.0016]}}
            plot_settings[key_plot]['v_range_all']={'HARPN':{'2016-10-01':[-1e-3  ,  1.6e-3]}}

        elif gen_dic['star_name']=='V1298tau':
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20200128':[-0.01,0.02],'20201207':[-0.01,0.02], 'mock_vis' : [-0.01,0.02]}}   #pour voir in et out
        
        #Reversed image 
        if plot_settings[key_plot]['reverse_2D']:   
            plot_settings[key_plot]['margins']=[0.15,0.12,0.95,0.84]   

            #Bornes du plot
            plot_settings[key_plot]['x_range']=[-0.015,0.021] #WASP8b    
            plot_settings[key_plot]['y_range']=[-20.,20.] 
            
            plot_settings[key_plot]['x_range']=[-0.05,0.05] #GJ436b 
            plot_settings[key_plot]['y_range']=[-26.,26.]  #plage du continu
            plot_settings[key_plot]['y_range']=[-20.,20.] 
    
            plot_settings[key_plot]['x_range_all']={'HARPN':{'2016-10-01':[-0.007,0.0061]}}
            plot_settings[key_plot]['y_range']=[-20.,20.]
 
     
            if gen_dic['studied_pl']=='WASP76b':
                plot_settings[key_plot]['x_range_all']={'ESPRESSO':{'2018-10-31':[-0.007,0.0061]}}
                plot_settings[key_plot]['y_range']=[-150.,150.]
           
        else:
    
       
    
            #Bornes du plot
            if gen_dic['studied_pl']=='WASP_8b':
                plot_settings[key_plot]['x_range']=[-20.,20.] 
                plot_settings[key_plot]['y_range_all']={'HARPS':{'2008-10-04':[-0.015,0.021]}}
                        
            #Bornes du plot
            elif gen_dic['studied_pl']=='GJ436_b':
                plot_settings[key_plot]['x_range']=[-21.,21.]   
                plot_settings[key_plot]['y_range']=[-0.15  ,  0.25]
                plot_settings[key_plot]['y_range']=[-0.15  ,  0.15]
                plot_settings[key_plot]['y_range']=[-0.0153,0.0285]
        
                plot_settings[key_plot]['y_range']=[-0.03,0.045] #GJ436b 
                plot_settings[key_plot]['x_range']=[-19.95,19.95] 

                plot_settings[key_plot]['x_range']=[-22.,22.] 
                plot_settings[key_plot]['y_range']=[-0.036,0.048] #GJ436b        
               
            elif gen_dic['star_name']=='55Cnc':
                plot_settings[key_plot]['x_range']=[-31.,31.]
                plot_settings[key_plot]['x_range']=[-122.5,177.5]
                plot_settings[key_plot]['x_range']=[-41.,41]

                
                y_range_comm=[-0.065  ,  0.11]
                y_range_comm=[-0.155,0.121]  #HARPS-N
                y_range_comm=[-0.17,0.23]   #HARPS, HARPS-N
                y_range_comm=[-0.16,0.13]   #HARPS, HARPS-N bonnes nuits
                plot_settings[key_plot]['y_range_all']={
                    'HARPS':{
                        '2012-01-27':y_range_comm,
                        '2012-02-27':y_range_comm,
                        '2012-02-13':y_range_comm,
                        '2012-03-15':y_range_comm
                        },
                    'HARPN':{
                          '2012-12-25':[-0.06357  ,  0.22984],
                          '2013-11-14':[-0.16  ,  0.08],
                          '2013-11-28':[-0.17  ,  0.13],
        #                 '2014-01-01':[-0.14  ,  0.1],
                          '2014-01-26':[-0.07,0.13],
                          '2014-02-26':[-0.09,0.1],
                          '2014-03-29':[-0.11,0.07]                
                    }, 
                    'SOPHIE':{
                        '2012-02-02':[-0.1,0.13],
                        '2012-02-03':[-0.05,0.2],
                        '2012-02-05':[-0.11,0.12],
                        '2012-02-17':[-0.12,0.15],
                        '2012-02-19':[-0.1,0.1],
                        '2012-02-22':[-0.1,0.11],
                        '2012-02-25':[-0.1,0.13],
                        '2012-03-02':[-0.05,0.12],
                        '2012-03-24':[-0.11,0.1],
                        '2012-03-27':[-0.1,0.11],
                        '2013-03-03':[-0.11,0.12],                   
                    },                
                    'binned':{
                        'all_HARPSS':y_range_comm,
                        'all_HARPS_adj':y_range_comm,
                        'all_HARPS_adj2':y_range_comm,
                        '2012-01-27_binned':y_range_comm,
                        '2012-02-27_binned':y_range_comm,
                        '2012-02-13_binned':y_range_comm,
                        '2012-03-15_binned':y_range_comm,
                        'good_HARPSN': y_range_comm,
                        'good_HARPSN_adj': y_range_comm,
                        'HARPS_HARPSN_binHARPS': y_range_comm,
                        'best_HARPSN_adj':y_range_comm,
                        'best_HARPSN_adj_short':y_range_comm,
                        'best_HARPSN_adj_long':y_range_comm
                    },
                    'ESPRESSO':{'2020-02-05':[-0.11,0.16]}
                }
    
               

            elif gen_dic['studied_pl']=='WASP121b':
                plot_settings[key_plot]['x_range']=[-50.,50.]
    #            plot_settings[key_plot]['y_range_all']={'HARPS':{'14-01-18':[-0.15,0.11],
    #                                  '09-01-18':[-0.1,0.18],
    #                                  '31-12-17':[-0.12,0.11]}}
                y_range_comm=[-0.15,0.18]
                plot_settings[key_plot]['y_range_all']={'HARPS':{'14-01-18':y_range_comm,'09-01-18':y_range_comm,'31-12-17':y_range_comm},
                              'binned':{'HARPS-binned':y_range_comm,'HARPS-binned-2018':y_range_comm}   }

            elif gen_dic['studied_pl']=='Kelt9b':
                plot_settings[key_plot]['x_range']=[-300.,300.]
                y_range_comm=[-0.15,0.1]
                plot_settings[key_plot]['y_range_all']={'HARPN':{'31-07-2017':y_range_comm}}    

            elif gen_dic['studied_pl']=='WASP76b':
                plot_settings[key_plot]['x_range']=[-150.,150.]
                y_range_comm=[-0.08  ,  0.09]
                plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2018-10-31':y_range_comm,'2018-09-03':y_range_comm},'binned':{'ESP_binned':y_range_comm}}         
      
                plot_settings[key_plot]['x_range']=[5880.,5905.]
                # plot_settings[key_plot]['x_range']=[-150,150]
                # plot_settings[key_plot]['x_range']=[-200.,200]
                plot_settings[key_plot]['x_range']= None
                y_range_comm=[-0.08  ,  0.09]
                plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2018-10-31':y_range_comm,'2018-09-03':y_range_comm},'binned':{'ESP_binned':y_range_comm}}         
          
            elif gen_dic['studied_pl']=='WASP127b':
                plot_settings[key_plot]['x_range']=[-22.,22.]
                y_range_comm=[-0.05  ,  0.04]
                plot_settings[key_plot]['y_range_all']={'HARPS':{'2017-03-20':y_range_comm,'2018-03-31':y_range_comm,'2018-02-13':y_range_comm,'2017-02-28':y_range_comm}} 
            elif gen_dic['star_name']=='HD209458':
                # plot_settings[key_plot]['x_range']= [-19.  , 19.]  
                plot_settings[key_plot]['x_range']= [5893.-6.1  ,5893.+6.1 ]   #Na doublet
                y_range_comm = [-0.03  , 0.04] 
                plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'20190720':y_range_comm,'20190911':y_range_comm}} 
            elif gen_dic['studied_pl']=='Corot7b':
                plot_settings[key_plot]['x_range']= [-21.,21.]  
                y_range_comm = [-0.07  , 0.15] 
                plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2019-02-20':y_range_comm}} 
            elif gen_dic['studied_pl']=='GJ9827d':
                plot_settings[key_plot]['x_range']= [-21.,21.]  
                plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2019-08-25':[-0.028 , 0.005] },'HARPS':{'2018-08-18':[-0.025,0.01],'2018-09-18':[-0.035,0.02]}} 
            elif gen_dic['studied_pl']=='GJ9827b':
                plot_settings[key_plot]['x_range']= [-21.,21.]  
                plot_settings[key_plot]['y_range_all']={'HARPS':{'2018-08-04':[-0.16,0.1],'2018-08-15':[-0.16,0.1],'2018-09-18':[-0.16,0.1],'2018-09-19':[-0.16,0.1]}} 
            elif gen_dic['studied_pl']=='Nu2Lupi_c':
                plot_settings[key_plot]['x_range']= [-21.,21.]  
                y_range_comm = [-0.0061 , 0.0041] 
                plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2020-03-18':y_range_comm}} 

            elif gen_dic['studied_pl']==['HD3167_c']:
                plot_settings[key_plot]['x_range']=[-20.,20.]
                plot_settings[key_plot]['y_range_all']={'HARPN':{'2016-10-01':[-0.0075,0.0075]}} 
                # plot_settings[key_plot]['x_range']=[-70.,70.]
                plot_settings[key_plot]['y_range_all']={'HARPN':{'2016-10-01':[-0.0075,0.006]}} 

    
            elif gen_dic['studied_pl']==['HD3167_b']:
                if data_dic['Res']['type'][inst]=='CCF':
                    plot_settings[key_plot]['x_range']= [-70.,70.]  
                    plot_settings[key_plot]['x_range']= [-51.,51.]  
                else:plot_settings[key_plot]['x_range']= None
                y_range_comm = [-0.07 , 0.108] 
                plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2019-10-09':y_range_comm}} 

            elif gen_dic['star_name']=='V1298tau':
                plot_settings[key_plot]['x_range']= None
                plot_settings[key_plot]['y_range_all']={'HARPN':{'20200128':[-0.013,-0.0027],'20201207':[-0.005,0.012]}} 
        
        


        






    '''
    Plotting individual residual spectral profiles in the star rest frame 
    '''
    if any('spec' in s for s in data_dic['Res']['type'].values()) and (plot_dic['sp_loc']!=''):
        key_plot = 'sp_loc'
        plot_settings[key_plot]={}  

        #Choice of orders to plot
        #    - leave empty to plot all orders
        plot_settings[key_plot]['orders_to_plot']=[10,11,12,13]
        plot_settings[key_plot]['orders_to_plot']=[0,1,2,3]
        if gen_dic['studied_pl']==['HD3167_b']:
            plot_settings[key_plot]['orders_to_plot']=range(76,111)

        #Scaling factor (in power of ten, ie flux are multiplied by 10**sc_fact10)
        plot_settings[key_plot]['sc_fact10']=-3.
        
        #Transparency
        plot_settings[key_plot]['alpha_err']=0.2

        #Overplot estimates for local stellar profiles
        plot_settings[key_plot]['estim_loc']=True    &  False 

        #Choose mode for estimates of local stellar profiles
        #    - data must have been calculated for the requested mode:
        # 'DIbin', 'Intrbin', 'glob_mod', 'indiv_mod', 'rec_prof', 'theo'
        plot_settings[key_plot]['mode_loc_data_corr'] = 'DIbin'
        
        #Instruments and visits to plot
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']} 
        elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-28','2019-04-29']}   
        elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']}             

        #Colors
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'}} 
        elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-02-28':'dodgerblue','2019-04-29':'red'}} 
        elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-10-09':'dodgerblue'}} 
            
        #Plot boundaries in wav
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['x_range']=[3500.,8000.] 
            plot_settings[key_plot]['x_range']=[5880.,5905.] 
            plot_settings[key_plot]['x_range']=[5887.,5898.] 
#            plot_settings[key_plot]['x_range']=[6200.,6300.] 
        # elif gen_dic['studied_pl']==['HD3167_b']:
        #     plot_settings[key_plot]['x_range']=[5000.,5700.]   
            
        #Plot boundaries in flux
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['y_range']=[-1e3,4e3] 
        elif gen_dic['studied_pl']==['HD3167_b']:
            plot_settings[key_plot]['y_range']=[-1.5e3,1.5e3] 
            
            
            
            
            
            
    '''
    Plotting individual residual CCF profiles in the star rest frame 
        - fit options relate to the model adjusted to the intrinsic stellar CCFs
    '''
    if ('CCF' in data_dic['Res']['type'].values()) and (plot_dic['CCFloc']!=''):
        key_plot = 'CCFloc'
        plot_settings[key_plot]={}  

        #Scaling factor (in power of ten, ie flux are multiplied by 10**sc_fact10)
        plot_settings[key_plot]['sc_fact10']=0.  #-5.



        #Visits to plot
        if gen_dic['studied_pl']=='WASP121b':
            plot_settings[key_plot]['visits_to_plot']={'HARPS':['14-01-18','09-01-18','31-12-17'],
                            'binned':['HARPS-binned']}          
    #        plot_settings[key_plot]['visits_to_plot']={'HARPS':['31-12-17']}     
        elif gen_dic['studied_pl']=='Kelt9b':plot_settings[key_plot]['visits_to_plot']={'HARPN':['31-07-2017']}  
        elif gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}              
            # plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03'],'binned':['ESP_binned']} 
        elif gen_dic['studied_pl']=='WASP127b':plot_settings[key_plot]['visits_to_plot']={'HARPS':['2017-03-20','2018-03-31','2018-02-13','2017-02-28']} 
        elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-28','2019-04-29']}   
        elif gen_dic['studied_pl']=='HD209458b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-07-20','2019-09-11']} 
        elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-20']} 
        elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-03-18']} 
        elif gen_dic['studied_pl']=='55Cnc_e':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-02-05']} 
        elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']}  
        elif gen_dic['studied_pl']==['HD3167_c']:plot_settings[key_plot]['visits_to_plot']={'HARPN':['2016-10-01']}  

            
        #Color dictionary
        if gen_dic['studied_pl']=='GJ436_b':
                    plot_settings[key_plot]['color_dic']=['dodgerblue','orange']  #deux nuits HARPS-N
                    plot_settings[key_plot]['color_dic']=['red','green']  #HARPS + HARPS-N binned
                    plot_settings[key_plot]['color_dic']=['red','red']  #HARPS + HARPS-N binned
            #        plot_settings[key_plot]['color_dic']=['red']  #HAR1PS
                    plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-02-28':'dodgerblue','2019-04-29':'red'}}
    
        if gen_dic['studied_pl']=='55Cnc_e':
    
            if data_dic['instrum_list']==['binned','HARPS']:
                if gen_dic['n_visits_tot']==5:plot_settings[key_plot]['color_dic']=['orange','purple','dodgerblue','limegreen','red']   #4 nuits binned
            if data_dic['instrum_list']==['HARPN']:plot_settings[key_plot]['color_dic']=['dodgerblue']
        
            plot_settings[key_plot]['color_dic']={
            
                #SOPHIE  
                '2012-02-02':'dodgerblue',
            
                #Nuits binned        
                'all_HARPSS':'dodgerblue',            
                'all_HARPS_adj':'dodgerblue',
                'all_HARPS_adj2':'cyan',            
                'best_HARPSN_adj':'red',
                'good_HARPSN_adj':'orange',
                'HARPS_HARPSN_binHARPS':'black',
                'HARPS_HARPSN_binHARPSN':'black',
                'best_HARPSN_adj_short':'lime',
                'best_HARPSN_adj_long':'red',            
                }        
            
            plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2020-02-05':'dodgerblue'}}
    
        elif gen_dic['studied_pl']==['HD3167_c']:
            plot_settings[key_plot]['color_dic']={'HARPN':{'2016-10-01':'dodgerblue'}}
    
        elif gen_dic['studied_pl']=='WASP121b':
            plot_settings[key_plot]['color_dic']={'09-01-18':'green','14-01-18':'dodgerblue','31-12-17':'red',
                        'HARPS-binned':'orange','HARPS-binned-2018':'orange'}      
    
        elif gen_dic['studied_pl']=='Kelt9b':
            plot_settings[key_plot]['color_dic']={'31-07-2017':'dodgerblue'}
        elif gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'},'binned':{'ESP_binned':'black'}} 
        elif gen_dic['studied_pl']=='WASP127b':plot_settings[key_plot]['color_dic']={'HARPS':{'2017-03-20':'dodgerblue','2018-03-31':'green','2018-02-13':'orange','2017-02-28':'red'}} 
        elif gen_dic['studied_pl']=='HD209458b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-07-20':'dodgerblue','2019-09-11':'red'}}



        #Overplot estimates for local stellar profiles
        plot_settings[key_plot]['estim_loc']=True     &  False 
        
        #Choose mode to retrieve for estimates
        #    - data must have been calculated for the requested mode:
        # 'DIbin', 'Intrbin', 'glob_mod', 'indiv_mod', 'rec_prof', 'theo'
        plot_settings[key_plot]['mode_loc_data_corr'] = 'DIbin'


        #Overplot continuum pixels
        plot_settings[key_plot]['plot_cont']=True   &   False

        #Overplot range of planetary signal excluded
        plot_settings[key_plot]['plot_plexc']=True   &  False

        #Shade area not included in fit
        plot_settings[key_plot]['plot_nofit']=True    & False

        #Overplot fit
        plot_settings[key_plot]['plot_line_model']=True    &   False

        #Choose model to use
        #    - from the fit to individual CCFs ('indiv') or from the global fit to all CCFs ('global')
        plot_settings[key_plot]['fit_type']='indiv'  

        #Print CCFs fit properties on plot
        plot_settings[key_plot]['plot_prop']=True    #  &   False
        
        #Plot measured centroid
        plot_settings[key_plot]['plot_line_fit_rv']=True    &   False

        #Plot fitted pixels 
        plot_settings[key_plot]['plot_fitpix']=True    &   False

        #Plot continuum pixels specific to each exposure 
        plot_settings[key_plot]['plot_cont_exp']=True      &   False
        
        #Plot stellar rest velocity
        plot_settings[key_plot]['plot_refvel']=True
        

        #Bornes du plot en RV
        if gen_dic['studied_pl']=='WASP_8b':
            plot_settings[key_plot]['x_range']=[-23.,20.] 
        elif gen_dic['studied_pl']=='GJ436_b':
            plot_settings[key_plot]['x_range']=[-21.,21.]
            plot_settings[key_plot]['x_range']=[-26.,26.]  #GJ436b, plage du continu
            plot_settings[key_plot]['x_range']=[-21.,21.]   
            plot_settings[key_plot]['x_range']=[-26.,26.]  #GJ436b, plage du continu
            plot_settings[key_plot]['x_range']=[-20.,20.] 
            plot_settings[key_plot]['x_range']=[-22.,22.] 
        elif gen_dic['studied_pl']=='55Cnc_e':
            plot_settings[key_plot]['x_range']=[-50.,50.]               
        elif gen_dic['studied_pl']=='WASP121b':
            plot_settings[key_plot]['x_range']=[-50.,50.]
            plot_settings[key_plot]['x_range']=[-90.,90.]    #Mask F
        elif gen_dic['studied_pl']=='Kelt9b':
            plot_settings[key_plot]['x_range']=[-300.,300.]    #Mask F   
        elif gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['x_range']=[-110.,110.] 
            plot_settings[key_plot]['x_range']=[-160.,160.] 
        elif gen_dic['studied_pl']=='WASP127b':plot_settings[key_plot]['x_range']=[-22.,22.] 
        elif gen_dic['star_name']=='HD209458':
            plot_settings[key_plot]['x_range']=[-23.,10.] 
        elif gen_dic['studied_pl']==['HD3167_b']:
            plot_settings[key_plot]['x_range']=[-70.,70.]  
            plot_settings[key_plot]['x_range']=[-51.,51.]         
        elif gen_dic['studied_pl']==['HD3167_c']:
            plot_settings[key_plot]['x_range']=[-20.,20.]
            plot_settings[key_plot]['x_range']=[-50.,50.]
        elif gen_dic['studied_pl']=='Corot7b':
            plot_settings[key_plot]['x_range']=[-21.,21.] 
            plot_settings[key_plot]['x_range']=[-21.+14.,21.+14.] 
        elif gen_dic['studied_pl']=='GJ9827d':
            plot_settings[key_plot]['x_range']=[-21.,21.] 
        elif gen_dic['star_name']=='HIP41378':
            plot_settings[key_plot]['x_range']=[-51.,51.]       

    

        #Bornes du plot
        #    - true fluxes, before scaling factor
        plot_settings[key_plot]['y_range']=None
        if gen_dic['studied_pl']=='GJ436_b':
             #Bornes du plot en RV
        #     plot_settings[key_plot]['y_range']=[-0.4,1.]
            plot_settings[key_plot]['y_range']=[-1.5e-3,1.5e-3]  #None
            plot_settings[key_plot]['y_range']=[-8e-4,8e-4]      #None
    
            plot_settings[key_plot]['y_range']=[-0.005,0.018]
    
            plot_settings[key_plot]['y_range']=[-2  ,  9]
    #       plot_settings[key_plot]['y_range']=[-0.5  ,  7.5] #binned HARPSN visits
            plot_settings[key_plot]['y_range']=[-0.001,0.008]
            plot_settings[key_plot]['sc_fact10']=3.
    
        elif gen_dic['studied_pl']=='55Cnc_e':
            plot_settings[key_plot]['y_range']=[-4,8]     
            plot_settings[key_plot]['y_range']=[-5e-4,8e-4]  
            plot_settings[key_plot]['sc_fact10']=4.   
        elif gen_dic['studied_pl']=='WASP121b':
            plot_settings[key_plot]['y_range']=[-3.,25.] 
            plot_settings[key_plot]['y_range']=[-3.,25.] 
        elif gen_dic['studied_pl']=='Kelt9b':
            plot_settings[key_plot]['y_range']=[-2.42  ,  9.23]          
        elif gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['y_range']=[-1.,19.]
            plot_settings[key_plot]['y_range']=[-4e4,3e5]
            plot_settings[key_plot]['y_range'] = None
        elif gen_dic['studied_pl']=='WASP127b':
            plot_settings[key_plot]['y_range']=[-0.003,0.015]
        elif gen_dic['studied_pl']=='HD209458b':
            plot_settings[key_plot]['y_range']=[-0.001,17e-3]
            plot_settings[key_plot]['y_range'] = None
            plot_settings[key_plot]['sc_fact10']=3.
        elif gen_dic['studied_pl']==['HD3167_b']:            
            plot_settings[key_plot]['y_range']=[-0.0015,2.e-3]
            plot_settings[key_plot]['y_range']=[-2.5e-3,3e-3]
            plot_settings[key_plot]['y_range']=[-1.5e-3,2e-3]
            plot_settings[key_plot]['sc_fact10']=3.
        elif gen_dic['studied_pl']==['HD3167_c']:
            plot_settings[key_plot]['y_range']=[-0.0015,2.e-3]
            plot_settings[key_plot]['sc_fact10']=3.            
        elif gen_dic['studied_pl']=='Corot7b':
            plot_settings[key_plot]['y_range']=[-0.007,0.007]
            plot_settings[key_plot]['sc_fact10']=3.
        elif gen_dic['studied_pl']=='Nu2Lupi_c':
            plot_settings[key_plot]['y_range']=[-3e-4,8e-4]
            plot_settings[key_plot]['sc_fact10']=4.
        elif gen_dic['studied_pl']=='GJ9827d':
            plot_settings[key_plot]['y_range']=[-1e-3,2e-3]   #ESPRESSO
            # plot_settings[key_plot]['y_range']=[-7e-3,7e-3]   #HARPS
            plot_settings[key_plot]['sc_fact10']=3.
        elif gen_dic['studied_pl']=='GJ9827b':
            plot_settings[key_plot]['y_range']=[-5e-3,5e-3]   #HARPS
            plot_settings[key_plot]['sc_fact10']=3.
        elif gen_dic['star_name']=='HIP41378':
            plot_settings[key_plot]['y_range']=[-2e-3,5e-3]   #HARPS
            plot_settings[key_plot]['sc_fact10']=3.










    ##################################################################################################
    #%% Standard deviation with bin size for out-of-transit residual CCFs
    #    - one plot per exposure
    ##################################################################################################
    if (plot_dic['scr_search']!=''):
        key_plot = 'scr_search'
        plot_settings[key_plot]={} 









            
            
            
            
        
        
        
    ##################################################################################################
    #%% 2D maps: intrinsic profiles
    ##################################################################################################
    if (plot_dic['map_Intr_prof']!=''):
        key_plot = 'map_Intr_prof'
        plot_settings[key_plot]={}      

        #Normalize CCFs
        plot_settings[key_plot]['norm_prof']=True        
        
        #Margins
        if gen_dic['star_name']=='GJ436':plot_settings[key_plot]['margins']=[0.15,0.3,0.75,0.95]          
        # if gen_dic['star_name']=='HD209458':plot_settings[key_plot]['margins']=[0.15,0.3,0.85,0.9]   #ANTARESS I, mock, multi-tr        

        #Font size
        if gen_dic['star_name']=='HD209458':plot_settings[key_plot]['font_size']=18   #ANTARESS I, mock, multi-tr
        if gen_dic['star_name']=='WASP76':plot_settings[key_plot]['font_size']=18   #ANTARESS I, CCF intr
        
        #Choice of visits to be plotted
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}  
        elif gen_dic['star_name']=='HD3167':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09'],'HARPN':['2016-10-01']}  
        elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-20']} 
        elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-03-18']}         
        
        
        #Choice of orders to plot
        #    - for 2D spectra (leave empty to plot all orders)  
        # plot_settings[key_plot]['orders_to_plot']=[113,114]   #sodium doublet
        # plot_settings[key_plot]['orders_to_plot']=[8,9]   #sodium doublet        
        
        #Choose dimension
        #    - 'phase', 'xp_abs', 'r_proj' 
        #    - if not phase, exposures are plotted successively without respecting their actual positions, because of overlaps 
        plot_settings[key_plot]['dim_plot']='phase'         
        
        #Plot aligned profiles
        plot_settings[key_plot]['aligned']=False

        #Overplot surface RV model along the full transit chord
        #    - CCF only, if not aligned 
        plot_settings[key_plot]['theoRV_HR']=True   &   False

        #Overplot surface RV model along the full transit chord for an aligned orbit
        plot_settings[key_plot]['theoRV_HR_align'] = True & False

        #Overplot surface RV model at the phases of the observations 
        plot_settings[key_plot]['plot_theoRV']=False

        #Overplot RV(pl/star) model 
        plot_settings[key_plot]['theoRVpl_HR']=True    #   &   False

        #Plot global and in-transit indexes
        plot_settings[key_plot]['plot_idx']=True    &   False  

        #Color range 
        plot_settings[key_plot]['v_range_all']={}
        if gen_dic['star_name']=='WASP76':
            v_range_comm=[0.4,1.15]   #ANTARESS I 
            # v_range_comm=[-5e3,3e4]   #sodium 
            plot_settings[key_plot]['sc_fact10']=0
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'20181030':v_range_comm,'20180902':v_range_comm}} 
            # plot_settings[key_plot]['v_range_all']={}
        elif gen_dic['star_name']=='GJ436': 
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'20190228':[0.65,1.1],'20190429':[0.65,1.1]},'HARPS':{'20070509':[0.6,1.3]},'HARPN':{'20160318':[0.6,1.2],'20160411':[0.6,1.2]}}  
            # plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'20190228':[0.6,1.3],'20190429':[0.6,1.3]},'HARPS':{'20070509':[0.6,1.3]},'HARPN':{'20160318':[0.6,1.3],'20160411':[0.6,1.3]}}  
        
        
        elif gen_dic['studied_pl']=='Corot7b':
            v_range_comm=[-5.,10.]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2019-02-20':v_range_comm}}    
        elif gen_dic['studied_pl']=='Nu2Lupi_c':
            v_range_comm=[0.,1.5]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2020-03-18':v_range_comm}}  
        elif gen_dic['studied_pl']=='GJ9827d':
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2019-08-25':[-0.5,2.5]},'HARPS':{'2018-08-18':[-3.,5.],'2018-09-18':[-3.,5.]}}   
        elif gen_dic['studied_pl']=='GJ9827b':
            plot_settings[key_plot]['v_range_all']={'HARPS':{vis:[-3., 4.] for vis in ['2018-08-04','2018-08-15','2018-09-18','2018-09-19']}}  
        elif gen_dic['star_name']=='55Cnc':
            v_range_comm=[0.2,1.3]
            plot_settings[key_plot]['v_range_all']['ESPRESSO']={'20200205':v_range_comm,'20210121':v_range_comm,'20210124':v_range_comm}  
            v_range_comm=[-0.4,2.]
            plot_settings[key_plot]['v_range_all']['HARPS']={'20120127':v_range_comm,'20120213':v_range_comm,'20120227':v_range_comm,'20120315':v_range_comm} 
            v_range_comm=[-0.4,2.]
            plot_settings[key_plot]['v_range_all']['HARPN']={'20131114':v_range_comm,'20131128':v_range_comm,'20140101':v_range_comm,'20140126':v_range_comm,'20140226':v_range_comm,'20140329':v_range_comm} 
            v_range_comm=[-0.1,1.6]
            plot_settings[key_plot]['v_range_all']['EXPRES']={'20220131':v_range_comm,'20220406':v_range_comm}  


        elif gen_dic['star_name']=='HD3167': 
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2019-10-09':[-2.5,4.]},'HARPN':{'2016-10-01':[0.,1.5]}}  
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2019-10-09':[-1.5,3.5]},'HARPN':{'2016-10-01':[0.,1.5]}}
        elif gen_dic['star_name']=='TOI858': 
            plot_settings[key_plot]['v_range_all']={'CORALIE':{'20191205':[0.,1.5],'20210118':[0.,1.5]}} 
        elif gen_dic['star_name']=='HIP41378': 
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20191218':[0.,2.],'20220401':[0.,2.]}}  
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20191218':[-1.,3.],'20220401':[-1.,3.]}} 
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20191218':[-0.5,1.5]}} 
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20191218':[-0.5,2.]}} 
            
        elif gen_dic['star_name']=='MASCARA1': 
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'20190714':[0.4,1.2],'20190811':[0.4,1.2]}}             
        elif gen_dic['star_name']=='HD209458': 
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'mock_vis':[0.4,1.2]}}
            v_range_comm = [-0.1,1.1]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'20190720':v_range_comm,'20190911':v_range_comm}}
            plot_settings[key_plot]['sc_fact10']=0

        #RM survey
        if gen_dic['star_name']=='HAT_P3':
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20200130':[0.3,1.3]}}  
        elif gen_dic['star_name']=='HAT_P11':
            v_range_comm = [0.35,1.15]              
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20150913':v_range_comm,'20151101':v_range_comm},'CARMENES_VIS':{'20170807':v_range_comm,'20170812':v_range_comm}}
        elif gen_dic['star_name']=='HAT_P33':
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20191204':[0.4,1.4]}}             
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20191204':[0.3,1.3]}}  
        elif gen_dic['star_name']=='HAT_P49':
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20200730':[0.4,1.4]}}             
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20200730':[0.3,1.3]}}  
        elif gen_dic['star_name']=='HD89345':
            plot_settings[key_plot]['v_range_all']={'HARPN':{
                # '20200202':[-0.3,2.5]}
                '20200202':[0.,1.8]}} 
        elif gen_dic['star_name']=='HD106315':
            v_range_comm = [0.1,1.5]            
            v_range_comm = [0.,1.6]   
            plot_settings[key_plot]['v_range_all']={'HARPS':{'20170309':v_range_comm,'20170330':v_range_comm,'20180323':v_range_comm}}  
        elif gen_dic['star_name']=='K2_105':
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20200118':[-1,2.5]}}  
        elif gen_dic['star_name']=='Kepler25':
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20190614':[-1,3]}} 
        elif gen_dic['star_name']=='Kepler63':
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20200513':[0.,1.8]}} 
        elif gen_dic['star_name']=='Kepler68':
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20190803':[-3,5]}}   
        elif gen_dic['star_name']=='WASP107':
            v_range_comm = [0.4,1.1]     
            plot_settings[key_plot]['v_range_all']={'HARPS':{'20140406':v_range_comm,'20180201':v_range_comm,'20180313':v_range_comm},'CARMENES_VIS':{'20180224':v_range_comm}}
        elif gen_dic['star_name']=='WASP47':   
            plot_settings[key_plot]['v_range_all']={}
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20210730':[-3.,5.]}}   
        elif gen_dic['star_name']=='WASP166':
            v_range_comm = [0.4,1.3]
            plot_settings[key_plot]['v_range_all']={'HARPS':{'20170114':v_range_comm,'20170304':v_range_comm,'20170315':v_range_comm}}       
        elif gen_dic['star_name']=='WASP156'  :
            v_range_comm = [0.3,1.3]            
            v_range_comm = [0.2,1.4] 
            plot_settings[key_plot]['v_range_all']={'CARMENES_VIS':{'20190928':v_range_comm,'20191025':v_range_comm,'20191210':v_range_comm}}
        elif gen_dic['star_name']=='WASP43':
            plot_settings[key_plot]['v_range_all']['NIRPS_HE']={'20230119':[0.,2.]}             
        elif gen_dic['star_name']=='L98_59':
            plot_settings[key_plot]['v_range_all']['NIRPS_HE']={'20230411':[-1.,3.]} 
        elif gen_dic['star_name']=='GJ1214':
            plot_settings[key_plot]['v_range_all']['NIRPS_HE']={'20230407':[0.6,1.4]}               
            
            
          
            
        #Ranges  
        plot_settings[key_plot]['y_range_all']={}
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['x_range']=[5880.,5905.]
            plot_settings[key_plot]['x_range']=[5883.,5902.]
            # plot_settings[key_plot]['x_range']=[-150,150]
            plot_settings[key_plot]['x_range']=[-95.,95]   #ANTARESS I
            y_range_comm=[-0.046  ,  0.046]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'20181030':y_range_comm,'20180902':y_range_comm},'binned':{'ESP_binned':y_range_comm}}         

        elif gen_dic['star_name']=='GJ436': 
            plot_settings[key_plot]['x_range']=[-31.,31.]
            plot_settings[key_plot]['x_range']=[-21.,21.]
            y_range_comm=[-0.009  ,  0.009]     
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'20190228':y_range_comm,'20190429':y_range_comm},'HARPS':{'20070509':y_range_comm},'HARPN':{'20160318':y_range_comm,'20160411':y_range_comm}}  
            
        elif gen_dic['studied_pl']=='Corot7b':
            plot_settings[key_plot]['x_range']=[-21.,21.]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2019-02-20':[-0.03  ,  0.03]}}    
        elif gen_dic['studied_pl']=='Nu2Lupi_c':
            plot_settings[key_plot]['x_range']=[-21.,21.]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2020-03-18':[-0.0028  ,  0.0028]}}    
        elif gen_dic['studied_pl']=='GJ9827d':
            plot_settings[key_plot]['x_range']=[-21.,21.]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2019-08-25':[-0.0045  ,  0.0045]},'HARPS':{'2018-08-18':[-0.005,0.005],'2018-09-18':[-0.005,0.005]}}   
        elif gen_dic['studied_pl']=='GJ9827b':
            plot_settings[key_plot]['x_range']=[-21.,21.]
            plot_settings[key_plot]['y_range_all']={'HARPS':{vis:[-0.027,0.027] for vis in ['2018-08-04','2018-08-15','2018-09-18','2018-09-19']}}     
        elif gen_dic['star_name']=='55Cnc':
            # plot_settings[key_plot]['x_range']=[-122.5,177.5]
            plot_settings[key_plot]['x_range']=[-41.,41]
            # plot_settings[key_plot]['x_range']=[-31.,31]
            
            y_range_comm = [-0.045  ,  0.045]
            plot_settings[key_plot]['y_range_all']['ESPRESSO']={'20200205':[-0.043  ,  0.043],'20210121':[-0.043  ,  0.043],'20210124':[-0.043  ,  0.043]}            
            plot_settings[key_plot]['y_range_all']['HARPS']={'20120127':y_range_comm,'20120213':y_range_comm,'20120227':y_range_comm,'20120315':y_range_comm}                       
            plot_settings[key_plot]['y_range_all']['HARPN']={'20131114':y_range_comm,'20131128':y_range_comm,'20140101':y_range_comm,'20140126':y_range_comm,'20140226':y_range_comm,'20140329':y_range_comm}       
            plot_settings[key_plot]['y_range_all']['EXPRES']={'20220131':y_range_comm,'20220406':y_range_comm}   
            
        elif gen_dic['star_name']=='HD3167': 
            plot_settings[key_plot]['x_range']=[-31.,31.]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2019-10-09':[-0.039  ,  0.039]},'HARPN':{'2016-10-01':[-0.004  ,  0.004]}}     
            if gen_dic['studied_pl']==['HD3167_b']:
                plot_settings[key_plot]['x_range']=[-71.,71.]
                plot_settings[key_plot]['x_range']=[-31.,31.]
                plot_settings[key_plot]['x_range']=[-31.,31.] 
            elif gen_dic['studied_pl']==['HD3167_c']:
                plot_settings[key_plot]['x_range']=[-21.,21.]
                plot_settings[key_plot]['x_range']=[-31.,31.]
        elif gen_dic['star_name']=='TOI858':
            plot_settings[key_plot]['x_range']=[-29.,29.]
            plot_settings[key_plot]['y_range_all']={'CORALIE':{'20191205':[-0.025,0.025],'20210118':[-0.025,0.025]}} 
        elif gen_dic['star_name']=='HIP41378':              
            plot_settings[key_plot]['x_range']=[-101.,101.]
            plot_settings[key_plot]['x_range']=[-51.,51.]
            plot_settings[key_plot]['x_range']=[-31.,31.]
            plot_settings[key_plot]['x_range']=[-41.,41.]
            plot_settings[key_plot]['y_range_all']={'HARPN':{'20191218':[0.00005,0.001],'20220401':[-0.00065, 0.0005]}}  
            plot_settings[key_plot]['y_range_all']={'HARPN':{'20191218':[0.0002,0.00095]}}            
        elif gen_dic['star_name']=='MASCARA1': 
            plot_settings[key_plot]['x_range']=[-301.,301.]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'20190714':[-0.043,0.043],'20190811':[-0.043,0.043]}}  
        elif gen_dic['star_name']=='HD209458':
            plot_settings[key_plot]['x_range']= [5893.-6.1  ,5893.+6.1 ]   #Na doublet
            y_range_comm = [-0.019  , 0.019] 
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'20190720':y_range_comm,'20190911':y_range_comm}} 
            
        #RM survey
        if gen_dic['star_name']=='HAT_P3':
            plot_settings[key_plot]['x_range']=[-125.,80.]   #full range
            plot_settings[key_plot]['x_range']=[-41.,41.]    #zoom, paper
            plot_settings[key_plot]['y_range_all']={'HARPN':{'20200130':[-0.018,0.018]}}  
        elif gen_dic['star_name']=='HAT_P11':
            plot_settings[key_plot]['x_range']=[-41.,41.]     #zoom 
            plot_settings[key_plot]['x_range']=[-36.,36.]     #papier
            ph_range_comm = [-0.0115  ,  0.0115 ]
            plot_settings[key_plot]['y_range_all']={'CARMENES_VIS':{'20170807':ph_range_comm,'20170812':ph_range_comm},'HARPN':{'20150913':ph_range_comm,'20151101':ph_range_comm}}               
        elif gen_dic['star_name']=='HAT_P33':
            plot_settings[key_plot]['x_range']=[-51.,51.]     #zoom  
            plot_settings[key_plot]['y_range_all']={'HARPN':{'20191204':[-0.028,0.028]}} 
        elif gen_dic['star_name']=='HAT_P49':
            plot_settings[key_plot]['x_range']=[-41.,41.]     #zoom 
            plot_settings[key_plot]['x_range']=[-51.,51.]     #zoom, paper 
            plot_settings[key_plot]['y_range_all']={'HARPN':{'20200730':[-0.034,0.034]}}             
        elif gen_dic['star_name']=='HD89345':
            plot_settings[key_plot]['x_range']=[-41.,41.]     #zoom 
            plot_settings[key_plot]['y_range_all']={'HARPN':{'20200202':[-0.01,0.01]}}                          
        elif gen_dic['star_name']=='HD106315':
            plot_settings[key_plot]['x_range']=[-41.,41.]     #zoom 
            plot_settings[key_plot]['x_range']=[-51.,51.]     #papier
            ph_range_comm = [-0.005  ,  0.005 ]          
            plot_settings[key_plot]['y_range_all']={'HARPS':{'20170309':ph_range_comm,'20170330':ph_range_comm,'20180323':ph_range_comm}}  
        elif gen_dic['star_name']=='K2_105':
            plot_settings[key_plot]['x_range']=[-41.,41.]     #zoom 
            plot_settings[key_plot]['y_range_all']={'HARPN':{'20200118':[-0.0092,0.0092]}} 
        elif gen_dic['star_name']=='Kepler25':            
            plot_settings[key_plot]['x_range']=[-150.,150.]   #full range   
            plot_settings[key_plot]['x_range']=[-41.,41.]     #zoom              
            plot_settings[key_plot]['y_range_all']={'HARPN':{'20190614':[-0.005,0.005]}}    
        elif gen_dic['star_name']=='Kepler63':
            plot_settings[key_plot]['x_range']=[-41.,41.]     #zoom 
            plot_settings[key_plot]['y_range_all']={'HARPN':{'20200513':[-0.0042,0.0067]}}             
            plot_settings[key_plot]['y_range_all']={'HARPN':{'20200513':[-0.0067,0.0067]}}               
        elif gen_dic['star_name']=='Kepler68':              
            plot_settings[key_plot]['x_range']=[-122.,82.]   #full range
            plot_settings[key_plot]['x_range']=[-41.,41.]     #zoom  
            plot_settings[key_plot]['y_range_all']={'HARPN':{'20190803':[-0.014,0.014]}} 
        elif gen_dic['star_name']=='WASP107':
            plot_settings[key_plot]['x_range']=[-61.,61.]     #zoom 
            plot_settings[key_plot]['x_range']=[-51.,51.]     #paper 
            ph_range_comm = [-0.011  ,  0.011 ]  
            plot_settings[key_plot]['y_range_all']={'HARPS':{'20140406':ph_range_comm,'20180201':ph_range_comm,'20180313':ph_range_comm},'CARMENES_VIS':{'20180224':ph_range_comm}}            
        elif gen_dic['star_name']=='WASP166':
            plot_settings[key_plot]['x_range']=[-41.,41.]     #zoom 
            plot_settings[key_plot]['x_range']=[-46.,46.]     #paper             
            ph_range_comm = [-0.0145  ,  0.0145 ]
            plot_settings[key_plot]['y_range_all']={'HARPS':{'20170114':ph_range_comm,'20170304':ph_range_comm,'20170315':ph_range_comm}}   
        elif gen_dic['star_name']=='WASP47':
            plot_settings[key_plot]['x_range']=None  
            plot_settings[key_plot]['x_range']= [-41.,41.]     #zoom 
            plot_settings[key_plot]['y_range_all']={'HARPN':{'20210730':[-0.0085,0.0105]}}              
            plot_settings[key_plot]['y_range_all']={'HARPN':{'20210730':[-0.0105,0.0105]}}              
        elif gen_dic['star_name']=='WASP156'  :
            plot_settings[key_plot]['x_range']= [-41.,41.]     #zoom 
            yrange_comm = [-0.017,0.017]
            plot_settings[key_plot]['y_range_all']={'CARMENES_VIS':{'20190928':yrange_comm,'20191025':yrange_comm,'20191210':yrange_comm}}
        elif gen_dic['star_name']=='WASP43':
            # plot_settings[key_plot]['x_range']=[-17.,17] 
            plot_settings[key_plot]['x_range']=[-10.,10] 
            plot_settings[key_plot]['y_range_all']['NIRPS_HE']={'20230119':[-0.035,0.035]}
        elif gen_dic['star_name']=='L98_59':
            plot_settings[key_plot]['x_range']=[-10.,10] 
            plot_settings[key_plot]['y_range_all']['NIRPS_HE']={'20230411':[-0.008,0.025]}
        elif gen_dic['star_name']=='GJ1214':
            plot_settings[key_plot]['x_range']=[-15.,15] 
            plot_settings[key_plot]['x_range']=[-7.,7] 
            plot_settings[key_plot]['y_range_all']['NIRPS_HE']={'20230407':[-0.013,0.013]}




            















    ##################################################################################################
    #%% 2D maps: best estimates for intrinsic stellar profiles and residuals
    #    - in star rest frame
    ##################################################################################################
    if (plot_dic['map_Intr_prof_est']!='') or (plot_dic['map_Intr_prof_res']!=''):
        for key_plot in ['map_Intr_prof_est','map_Intr_prof_res']:
            if plot_dic[key_plot]!='':
                plot_settings[key_plot]={} 

                #Choose mode to retrieve
                #    - estimates for the local stellar profiles must have been calculated for the requested mode:
                # 'DIbin', 'Intrbin', 'glob_mod', 'indiv_mod', 'rec_prof', 'theo'
                #    - not required for residual map if only continuum level is subtracted
                plot_settings[key_plot]['mode_loc_data_corr'] = 'glob_mod'

                #Choose dimension
                #    - 'phase', 'xp_abs', 'r_proj' 
                #    - if not phase, exposures are plotted successively without respecting their actual positions, because of overlaps 
                plot_settings[key_plot]['dim_plot']='phase' 

                #Choice of visits to be plotted
                if gen_dic['studied_pl']=='WASP76b':
                    plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}  
                elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-28','2019-04-29']}  
                if gen_dic['star_name']=='HD3167':
                    plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09'],'HARPN':['2016-10-01']}  

                #Choice of orders to plot
                #    - for 2D spectra (leave empty to plot all orders)
                # plot_settings[key_plot]['orders_to_plot']=[113,114]   #sodium doublet
                # plot_settings[key_plot]['orders_to_plot']=[8,9]   

                #Overplot surface RV model along the full transit chord
                #    - CCF only, if not aligned 
                plot_settings[key_plot]['theoRV_HR']=True & False 
                if gen_dic['star_name'] in ['Kepler68','WASP47']:plot_settings[key_plot]['theoRV_HR']=False

                #Overplot surface RV model along the full transit chord for an aligned orbit
                plot_settings[key_plot]['theoRV_HR_align'] = True 
                
                #Overplot RV(pl/star) model 
                if gen_dic['star_name']=='HAT_P49':plot_settings[key_plot]['theoRVpl_HR']=True  

                #Ranges
                plot_settings[key_plot]['y_range_all']={}
                plot_settings[key_plot]['v_range_all']={}
                if gen_dic['studied_pl']=='WASP76b':
                    # plot_settings[key_plot]['x_range']=[5880.,5905.]
                    plot_settings[key_plot]['x_range']=[5883.,5902.]
                    # plot_settings[key_plot]['x_range']=[-150,150]
                    # plot_settings[key_plot]['x_range']=[-100.,100]
                    # y_range_comm=[-0.048  ,  0.048]
                    # plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2018-10-31':y_range_comm,'2018-09-03':y_range_comm},'binned':{'ESP_binned':y_range_comm}} 
                elif gen_dic['star_name']=='GJ436':
                    plot_settings[key_plot]['x_range']=[-21.,21.]
                    plot_settings[key_plot]['x_range']=[-30.,30.]
                    y_range_comm=[-0.036  ,  0.046]
                    
                    plot_settings[key_plot]['x_range']=[-31.,31.]
                    plot_settings[key_plot]['x_range']=[-21.,21.]
                    y_range_comm =[-0.009  ,  0.009]  
                    plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'20190228':y_range_comm,'20190429':y_range_comm}}   
        
            
                if gen_dic['star_name']=='HD3167':
                    plot_settings[key_plot]['x_range']=[-31.,31.]      
                    plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2019-10-09':[-0.037  ,  0.037]},'HARPN':{'2016-10-01':[-0.0037  ,  0.0037]}}     
                    plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2019-10-09':[-0.07  ,  0.107]},'HARPN':{'2016-10-01':[-0.0074  ,  0.0057]}}    
                    if gen_dic['studied_pl']==['HD3167_b']:
                        plot_settings[key_plot]['x_range']=[-71.,71.]
                        plot_settings[key_plot]['x_range']=[-31.,31.]  
                    elif gen_dic['studied_pl']==['HD3167_c']:
                        plot_settings[key_plot]['x_range']=[-71.,71.]
                        plot_settings[key_plot]['x_range']=[-31.,31.]           
                if gen_dic['star_name']=='TOI858':
                    plot_settings[key_plot]['x_range']=[-29.,29.]   
                    plot_settings[key_plot]['y_range_all']={'CORALIE':{'20191205':[-0.03  ,  0.05],'20210118':[-0.03  ,  0.05]}}    
                elif gen_dic['star_name']=='HIP41378'  :
                    plot_settings[key_plot]['x_range']=[-41.,41.]      
                    plot_settings[key_plot]['y_range_all']={'HARPN':{'20191218':[0.0002,0.00095]}}   
                    plot_settings[key_plot]['y_range_all']={'HARPN':{'20191218':[0.0002,0.00135]}}   
                    
                #RM survey
                elif gen_dic['star_name']=='HAT_P3':
                    plot_settings[key_plot]['x_range']=[-41.,41.]    #zoom
                    plot_settings[key_plot]['y_range_all']={'HARPN':{'20200130':[-0.016,0.016]}}
                    plot_settings[key_plot]['y_range_all']={'HARPN':{'20200130':[-0.035,0.06]}}
                elif gen_dic['star_name']=='HAT_P11':
                    plot_settings[key_plot]['x_range']=[-36.,36.]     #papier
                    plot_settings[key_plot]['y_range_all']={'CARMENES_VIS':{'20170807':[-0.045  ,  0.023 ],'20170812':[-0.026  ,  0.045 ]},'HARPN':{'20150913':[-0.04 ,  0.032 ],'20151101':[-0.02  ,  0.023 ]}}   
                elif gen_dic['star_name']=='HAT_P33':
                    plot_settings[key_plot]['x_range']=[-51.,51.]     #zoom  
                    plot_settings[key_plot]['y_range_all']={'HARPN':{'20191204':[-0.0293,0.0293]}}    
                    plot_settings[key_plot]['y_range_all']={'HARPN':{'20191204':[-0.049,0.039]}}  
                elif gen_dic['star_name']=='HAT_P49'  :      
                    plot_settings[key_plot]['x_range']=[-51.,51.]               
                    plot_settings[key_plot]['y_range_all']={'HARPN':{'20200730':[-0.061,0.061]}} 
                elif gen_dic['star_name']=='HD89345':
                    plot_settings[key_plot]['x_range']=[-41.,41.]     #zoom     
                    plot_settings[key_plot]['y_range_all']={'HARPN':{'20200202':[-0.0124,0.017]}}   
                elif gen_dic['star_name']=='HD106315':
                    plot_settings[key_plot]['x_range']=[-51.,51.]       
                    plot_settings[key_plot]['y_range_all']={'HARPS':{'20170309':[-0.0058,0.0107],'20170330':[-0.011,0.0062],'20180323':[-0.0084,0.0089]}}               
                elif gen_dic['star_name']=='K2_105':
                    plot_settings[key_plot]['x_range']=[-41.,41.]     #zoom 
                    plot_settings[key_plot]['y_range_all']={'HARPN':{'20200118':[-0.0089,0.0089]}}  
                    plot_settings[key_plot]['y_range_all']={'HARPN':{'20200118':[-0.015,0.0175]}} 
                elif gen_dic['star_name']=='Kepler25':            
                    plot_settings[key_plot]['x_range']=[-41.,41.]     #zoom              
                    plot_settings[key_plot]['y_range_all']={'HARPN':{'20190614':[-0.0048,0.0048]}}             
                    plot_settings[key_plot]['y_range_all']={'HARPN':{'20190614':[-0.0115,0.012]}} 
                elif gen_dic['star_name']=='Kepler63'  :      
                    plot_settings[key_plot]['x_range']=[-41.,41.]                 
                    plot_settings[key_plot]['y_range_all']={'HARPN':{'20200513':[-0.007,0.0115]}}  
                elif gen_dic['star_name']=='Kepler68'  :      
                    plot_settings[key_plot]['x_range']=[-41.,41.]                 
                    plot_settings[key_plot]['y_range_all']={'HARPN':{'20190803':[-0.032,0.021]}}               
                elif gen_dic['star_name']=='WASP107':
                    plot_settings[key_plot]['x_range']=[-61.,61.]     #zoom 
                    plot_settings[key_plot]['x_range']=[-51.,51.]     #paper
                    plot_settings[key_plot]['y_range_all']={'HARPS':{'20140406':[-0.0105,0.0105],'20180201':[-0.0105,0.0105],'20180313':[-0.0105,0.0105]},'CARMENES_VIS':{'20180224':[-0.0105,0.0105]}}                        
                    plot_settings[key_plot]['y_range_all']={'HARPS':{'20140406':[-0.0245,0.0215],'20180201':[-0.017,0.0255],'20180313':[-0.04,0.021]},'CARMENES_VIS':{'20180224':[-0.0245,0.0185]}}                        
                elif gen_dic['star_name']=='WASP47'  :      
                    plot_settings[key_plot]['x_range']=None             
                    plot_settings[key_plot]['x_range']=[-41.,41.]       
                    plot_settings[key_plot]['y_range_all']={'HARPN':{'20210730':[-0.0105,0.0165]}}  
                elif gen_dic['star_name']=='WASP156'  :
                    plot_settings[key_plot]['x_range']=[-101.,101.]     
                    plot_settings[key_plot]['x_range']=[-41.,41.]                          
                    plot_settings[key_plot]['y_range_all']={'CARMENES_VIS':{'20190928':[-0.038,0.042],'20191025':[-0.023,0.057]}}              
                elif gen_dic['star_name']=='WASP166'  :   
                    plot_settings[key_plot]['x_range']=[-41.,41.]                          
                    plot_settings[key_plot]['y_range_all']={'HARPS':{'20170114':[-0.029,0.024],'20170304':[-0.018,0.022],'20170315':[-0.02,0.036]}}  
        
                elif gen_dic['star_name']=='55Cnc'  :   
                    plot_settings[key_plot]['x_range']=[-125.,180.]                          
                    plot_settings[key_plot]['x_range']=[-41.,41.] 
                    plot_settings[key_plot]['y_range_all']['ESPRESSO']={'20200205':[-0.093  ,  0.16],'20210121':[-0.161  ,  0.098],'20210124':[-0.11  ,  0.155]}   
                    plot_settings[key_plot]['y_range_all']['HARPS']={'20120127':[-0.063,0.1],'20120213':[-0.045,0.109],'20120227':[-0.063,0.0615],'20120315':[-0.06,0.082]}
                    # plot_settings[key_plot]['x_range']=[-200.,200.] 
                    plot_settings[key_plot]['y_range_all']['EXPRES']={'20220131':[-0.08,0.095],'20220406':[-0.15,0.15]}          
                elif gen_dic['star_name']=='WASP43':
                    plot_settings[key_plot]['x_range']=[-17.,17] 
                    plot_settings[key_plot]['x_range']=[-10.,10] 
                    plot_settings[key_plot]['y_range_all']['NIRPS_HE']={'20230119':[-0.07,0.11]} 
                elif gen_dic['star_name']=='L98_59':
                    plot_settings[key_plot]['x_range']=[-10.,10] 
                    plot_settings[key_plot]['y_range_all']['NIRPS_HE']={'20230411':[-0.013,0.032]} 
                elif gen_dic['star_name']=='GJ1214':
                    plot_settings[key_plot]['x_range']=[-15.,15] 
                    plot_settings[key_plot]['x_range']=[-7.,7] 
                    plot_settings[key_plot]['y_range_all']['NIRPS_HE']={'20230407':[-0.02,0.025]} 



        #---------------------------------
        if (plot_dic['map_Intr_prof_est']!=''):  
            key_plot='map_Intr_prof_est'

            #Color range 
            if gen_dic['studied_pl']=='WASP76b':
                v_range_comm=[3,7]   #CCF
                v_range_comm=[-5e3,3e4]   #sodium 
                plot_settings[key_plot]['sc_fact10']=-3
                plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2018-10-31':v_range_comm,'2018-09-03':v_range_comm}} 
                # plot_settings[key_plot]['v_range_all']={}
            elif gen_dic['star_name']=='GJ436':
                v_range_comm=[0.4,1.3]
                v_range_comm=[0.65,1.1]
                plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'20190228':v_range_comm,'20190429':v_range_comm}}  
                
            elif gen_dic['studied_pl']==['HD3167_b']:
                v_range_comm=[-3.,5.]
                plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2019-10-09':v_range_comm}}  
            elif gen_dic['studied_pl']==['HD3167_c']:
                plot_settings[key_plot]['v_range_all']={'HARPN':{'2016-10-01':[0.4,1.]}}       
            elif gen_dic['star_name']=='HIP41378':
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20191218':[-0.5,1.5]}}
                
                
            #RM survey                
            elif gen_dic['star_name']=='HAT_P3':
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20200130':[0.3,1.3]}}                   
            elif gen_dic['star_name']=='Kepler25':
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20190614':[-1,3]}}            
            elif gen_dic['star_name']=='HAT_P33':
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20191204':[0.4,1.4]}}       
            elif gen_dic['star_name']=='K2_105':
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20200118':[-1,2.5]}}
            elif gen_dic['star_name']=='WASP107':
                plot_settings[key_plot]['v_range_all']={'HARPS':{'20140406':[0.5,1.1],'20180201':[0.5,1.1],'20180313':[0.5,1.1]},'CARMENES_VIS':{'20180224':[0.5,1.1]}}                 
                




        #---------------------------------
        if (plot_dic['map_Intr_prof_res']!=''):  
            key_plot='map_Intr_prof_res'

            #Include out-of-transit residual profiles to the plot
            plot_settings[key_plot]['show_outres']=True # & False

            #Correct only for continuum level
            plot_settings[key_plot]['cont_only']=True   & False
            if gen_dic['star_name'] in ['Kepler68','WASP47']:plot_settings[key_plot]['cont_only']=True  #non-detection
            if gen_dic['star_name'] in ['WASP43','L98_59','GJ1214']:plot_settings[key_plot]['cont_only']=True  #non-detection


            #Color range 
            if gen_dic['studied_pl']=='WASP76b':
                v_range_comm=[-1e4,1e4]  
                plot_settings[key_plot]['sc_fact10']=-4
                plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2018-10-31':v_range_comm,'2018-09-03':v_range_comm}} 
                # plot_settings[key_plot]['v_range_all']={}            
                
            elif gen_dic['star_name']=='GJ436':
                v_range_comm=[-8e-4,8e-4]  
                plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'20190228':v_range_comm,'20190429':v_range_comm}}  
                plot_settings[key_plot]['sc_fact10']=4
                
                
                
            if gen_dic['star_name']=='HD3167':
                plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2019-10-09':[-3.,5.]},'HARPN':{'2016-10-01':[-5e-4,5e-4]}}
                plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2019-10-09':[-7e-4,7e-4]},'HARPN':{'2016-10-01':[-8e-4,5e-4]}} 
                plot_settings[key_plot]['sc_fact10']=4   
            if gen_dic['star_name']=='TOI858':
                plot_settings[key_plot]['v_range_all']={'CORALIE':{'20191205':[-10e-3,6e-3],'20210118':[-10e-3,6e-3]}}      #cont only
                plot_settings[key_plot]['v_range_all']={'CORALIE':{'20191205':[-7e-3,7e-3],'20210118':[-7e-3,7e-3]}}      #cont and model
                plot_settings[key_plot]['sc_fact10']=4
            if gen_dic['star_name']=='MASCARA1':
                plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'20190714':[-1.5e-3,1.5e-3],'20190811':[-1.5e-3,1.5e-3]}}      #cont only
            elif gen_dic['star_name']=='HIP41378':
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20191218':[-1.5e-3,1.5e-3]}}


            #RM survey
            if gen_dic['star_name']=='HAT_P3':
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20200130':[-3e-3,3e-3]}}  
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20200130':[-2.5e-3,2.5e-3]}}    
            elif gen_dic['star_name']=='HAT_P11':
                v_range_comm = [-2e-3,2e-3]              
                v_range_comm = [-1.5e-3,1.5e-3] 
                v_range_comm = [-2.5e-3,1e-3]   #test corr C
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20150913':v_range_comm,'20151101':v_range_comm},'CARMENES_VIS':{'20170807':v_range_comm,'20170812':v_range_comm}}
            elif gen_dic['star_name']=='HAT_P33':
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20191204':[-2e-3,2e-3]}}  
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20191204':[-4e-3,4e-3]}}  
            elif gen_dic['star_name']=='HAT_P49':
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20200730':[-3e-3,3e-3]}} 
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20200730':[-2.5e-3,2.5e-3]}}    #comparaison corrections
                # plot_settings[key_plot]['v_range_all']={'HARPN':{'20200730':[-1.5e-3,1.5e-3]}}             
            elif gen_dic['star_name']=='HD89345':
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20200202':[-1.5e-3,1.5e-3]}} 
            elif gen_dic['star_name']=='HD106315':
                v_range_comm = [-2e-3,2e-3]   
                v_range_comm = [-1.5e-3,1.5e-3] 
                plot_settings[key_plot]['v_range_all']={'HARPS':{'20170309':v_range_comm,'20170330':v_range_comm,'20180323':v_range_comm}}                 
            elif gen_dic['star_name']=='K2_105':
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20200118':[-2.5e-3,2.5e-3]}}                  
            elif gen_dic['star_name']=='Kepler25':
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20190614':[-2.5e-3,2.5e-3]}}  
            elif gen_dic['star_name']=='Kepler63'  :                     
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20200513':[-4e-3,4e-3]}}  
            elif gen_dic['star_name']=='Kepler68':   
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20190803':[-2.5e-3,2.5e-3]}}              
            elif gen_dic['star_name']=='WASP47':   
                plot_settings[key_plot]['v_range_all']={'HARPN':{'20210730':[-3.5e-3,3.5e-3]}} 
                
                
            elif gen_dic['star_name']=='WASP107':
                v_range_comm = [-3.5e-3,3.5e-3]  
                v_range_comm = [-4e-3,4e-3]  
                plot_settings[key_plot]['v_range_all']={'HARPS':{'20140406':v_range_comm,'20180201':v_range_comm,'20180313':v_range_comm},'CARMENES_VIS':{'20180224':v_range_comm}}                 
            elif gen_dic['star_name']=='WASP166':
                v_range_comm = [-2e-3,2e-3]  
                plot_settings[key_plot]['v_range_all']={'HARPS':{'20170114':v_range_comm,'20170304':v_range_comm,'20170315':v_range_comm}}     
            elif gen_dic['star_name']=='WASP156'  :        
                plot_settings[key_plot]['v_range_all']={'CARMENES_VIS':{'20190928':[-2e-3,2e-3],'20191025':[-4e-3,4e-3],'20191210':[-2e-3,4e-3]}}
                plot_settings[key_plot]['v_range_all']={'CARMENES_VIS':{'20190928':[-3e-3,3e-3],'20191025':[-3e-3,3e-3],'20191210':[-3e-3,3e-3]}}
  
            elif gen_dic['star_name']=='55Cnc':        
                v_range_comm = [-0.0002,0.0002]
                # v_range_comm = [-0.0004,0.0004]
                plot_settings[key_plot]['v_range_all']['ESPRESSO']={'20200205':v_range_comm,'20210121':v_range_comm,'20210124':v_range_comm}
                v_range_comm = [-0.0006,0.0006]
                plot_settings[key_plot]['v_range_all']['HARPS']={'20120127':v_range_comm,'20120213':v_range_comm,'20120227':v_range_comm,'20120315':v_range_comm}        
                v_range_comm=[-0.00035,0.00035]
                plot_settings[key_plot]['v_range_all']['EXPRES']={'20220131':v_range_comm,'20220406':v_range_comm}  
            elif gen_dic['star_name']=='WASP43':
                plot_settings[key_plot]['sc_fact10']=2.
                plot_settings[key_plot]['v_range_all']['NIRPS_HE']={'20230119':[-0.01,0.01]} 
            elif gen_dic['star_name']=='L98_59':
                plot_settings[key_plot]['sc_fact10']=2.
                plot_settings[key_plot]['v_range_all']['NIRPS_HE']={'20230411':[-0.008,0.008]} 
            elif gen_dic['star_name']=='GJ1214':
                plot_settings[key_plot]['sc_fact10']=2.
                plot_settings[key_plot]['v_range_all']['NIRPS_HE']={'20230407':[-0.008,0.008]} 









    '''
    2D maps of PC-based noise model 
    '''
    if (plot_dic['map_pca_prof']!=''):
        key_plot = 'map_pca_prof'
        plot_settings[key_plot]={}           

        #Overplot surface RV model along the full transit chord
        #    - CCF only, if not aligned 
        plot_settings[key_plot]['theoRV_HR']=True  

        #Ranges
        if gen_dic['star_name']=='WASP166'  :   
            plot_settings[key_plot]['x_range']=[-41.,41.]                          
            plot_settings[key_plot]['y_range_all']={'HARPS':{'20170114':[-0.029,0.024],'20170304':[-0.018,0.022],'20170315':[-0.02,0.036]}}  

        #Color range 
        if gen_dic['star_name']=='WASP107':
            plot_settings[key_plot]['v_range_all']={'HARPS':{'20140406':[0.5,1.1],'20180201':[0.5,1.1],'20180313':[0.5,1.1]},'CARMENES_VIS':{'20180224':[0.5,1.1]}}                 
               










    '''
    2D maps of binned intrinsic stellar profiles
    '''
    if (plot_dic['map_Intrbin']!=''):
        key_plot = 'map_Intrbin'
        plot_settings[key_plot]={}  

        #Choice of visits to be plotted
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}  
        elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-28','2019-04-29']} 
        elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']}  
        elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-20']} 
        elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-03-18']} 
        elif gen_dic['studied_pl']=='GJ9827d':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-08-25'],'HARPS':'binned'} 
        elif gen_dic['studied_pl']=='GJ9827b':plot_settings[key_plot]['visits_to_plot']={'HARPS':'binned'} 
        elif gen_dic['star_name']=='HAT_P11':
            plot_settings[key_plot]['visits_to_plot']['HARPN']=['binned']
            plot_settings[key_plot]['visits_to_plot']['CARMENES_VIS']=['binned']
        elif gen_dic['star_name']=='HD106315':
            plot_settings[key_plot]['visits_to_plot']['HARPS']=['binned']    
        elif gen_dic['star_name']=='WASP107':
            plot_settings[key_plot]['visits_to_plot']['HARPS']=['binned']    
        # elif gen_dic['star_name']=='WASP166':
        #     plot_settings[key_plot]['visits_to_plot']['HARPS']=['binned']               
        elif gen_dic['star_name']=='55Cnc':
            plot_settings[key_plot]['visits_to_plot']['ESPRESSO']=['binned']             
            plot_settings[key_plot]['visits_to_plot']['HARPS']=['binned']  
            
        #Choice of orders to plot
        #    - for 2D spectra (leave empty to plot all orders)
        # plot_settings[key_plot]['orders_to_plot']=[113,114]   #sodium doublet

        #Overplot surface RV model along the full transit chord
        #    - CCF only, if not aligned 
        plot_settings[key_plot]['theoRV_HR']=True #  &   False

        #Choose bin dimension
        #    - 'phase', 'xp_abs', 'r_proj' (see details and routine)
        plot_settings[key_plot]['dim_plot']='phase'         

 
        #Color range
        if gen_dic['studied_pl']=='WASP76b':
            v_range_comm=[3,7]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2018-10-31':v_range_comm,'2018-09-03':v_range_comm}}  
        elif gen_dic['studied_pl']=='GJ436_b':
            v_range_comm=[0.4,1.3]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2019-02-28':v_range_comm,'2019-04-29':v_range_comm}}   
        elif gen_dic['studied_pl']=='Corot7b':
            v_range_comm=[-5.,10.]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2019-02-20':v_range_comm}}  
        elif gen_dic['studied_pl']=='Nu2Lupi_c':
            v_range_comm=[0.,1.5]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2020-03-18':v_range_comm}}  
        elif gen_dic['studied_pl']=='GJ9827d':
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2019-08-25':[-0.5,2.5]},'HARPS':{'binned':[-3.,5.]}}  
        elif gen_dic['studied_pl']=='GJ9827b':
            plot_settings[key_plot]['v_range_all']={'HARPS':{'binned':[-0.5,2.]}}   
        elif gen_dic['star_name']=='55Cnc':
            v_range_comm=[0.3,1.3]
            v_range_comm=[0.2,1.3]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'binned':v_range_comm}}        
            v_range_comm=[-0.4,2.]
            v_range_comm=[-0.2,1.8]
            v_range_comm=[0.,1.5]
            plot_settings[key_plot]['v_range_all']={'HARPS':{'binned':v_range_comm}}  
        elif gen_dic['studied_pl']==['HD3167_b']:
            v_range_comm=[-3.,5.]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2019-10-09':v_range_comm}}         
        #RM survey
        elif gen_dic['star_name']=='HAT_P11':
            v_range_comm = [0.35,1.15]              
            plot_settings[key_plot]['v_range_all']={'HARPN':{'binned':v_range_comm},'CARMENES_VIS':{'binned':v_range_comm}}
        elif gen_dic['star_name']=='HD106315':
            v_range_comm = [0.2,1.4]      
            plot_settings[key_plot]['v_range_all']={'HARPS':{'binned':v_range_comm}}                           
        elif gen_dic['star_name']=='WASP107':
            plot_settings[key_plot]['v_range_all']={'HARPS':{'binned':[0.4,1.1]}}
        elif gen_dic['star_name']=='WASP166':
            plot_settings[key_plot]['v_range_all']={'HARPS':{'binned':[0.3,1.3],'20170114':[0.3,1.3],'20170304':[0.3,1.3],'20170315':[0.3,1.3]}}
            
        #Ranges
        if gen_dic['studied_pl']=='WASP76b':
            # plot_settings[key_plot]['x_range']=[5880.,5905.]
            # plot_settings[key_plot]['x_range']=[-150,150]
            plot_settings[key_plot]['x_range']=[-100.,100]
            y_range_comm=[-0.048  ,  0.048]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2018-10-31':y_range_comm,'2018-09-03':y_range_comm},'binned':{'ESP_binned':y_range_comm}}         
        elif gen_dic['studied_pl']=='GJ436_b':
            plot_settings[key_plot]['x_range']=[-21.,21.]
            y_range_comm=[-0.009  ,  0.009]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2019-02-28':y_range_comm,'2019-04-29':y_range_comm}}                       
 
        elif gen_dic['studied_pl']=='Corot7b':
            plot_settings[key_plot]['x_range']=[-21.,21.]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2019-02-20':[-0.03  ,  0.03]}}    
        elif gen_dic['studied_pl']=='Nu2Lupi_c':
            plot_settings[key_plot]['x_range']=[-21.,21.]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2020-03-18':[-0.0028  ,  0.0028]}}    
        elif gen_dic['studied_pl']=='GJ9827d':
            plot_settings[key_plot]['x_range']=[-21.,21.]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2019-08-25':[-0.009  ,  0.009]},'HARPS':{'binned':[-0.005,0.005]}}   
        elif gen_dic['studied_pl']=='GJ9827b':
            plot_settings[key_plot]['x_range']=[-21.,21.]
            plot_settings[key_plot]['y_range_all']={'HARPS':{'binned':[-0.03,0.03]}}     
        elif gen_dic['star_name']=='55Cnc':
            plot_settings[key_plot]['x_range']=[-125.,180.]
            plot_settings[key_plot]['x_range']=[-41.,41.]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'binned':[-0.043  ,  0.043]},'HARPS':{'binned':[-0.043  ,  0.043]}}   
        elif gen_dic['studied_pl']==['HD3167_b']:
            plot_settings[key_plot]['x_range']=[-71.,71.]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2019-10-09':[-0.037  ,  0.037]}}  
        #RM survey
        elif gen_dic['star_name']=='HAT_P11':
            plot_settings[key_plot]['x_range']=[-36.,36.]     #zoom 
            ph_range_comm =[-0.0115  ,  0.0115 ]
            plot_settings[key_plot]['y_range_all']={'HARPN':{'binned':ph_range_comm},'CARMENES_VIS':{'binned':ph_range_comm}}           
        elif gen_dic['star_name']=='HD106315':
            plot_settings[key_plot]['x_range']=[-51.,51.]     #papier
            plot_settings[key_plot]['y_range_all']={'HARPS':{'binned':[-0.005  ,  0.005 ] }}             
        elif gen_dic['star_name']=='WASP107':
            plot_settings[key_plot]['x_range']=[-51.,51.]     #paper 
            plot_settings[key_plot]['y_range_all']={'HARPS':{'binned':[-0.011  ,  0.011 ]  }}            
        elif gen_dic['star_name']=='WASP166':
            plot_settings[key_plot]['x_range']=[-46.,46.]     #paper 
            plot_settings[key_plot]['y_range_all']={'HARPS':{'binned':[-0.0145  ,  0.0145 ],'20170114':[-0.0145  ,  0.0145 ],'20170304':[-0.0145  ,  0.0145 ],'20170315':[-0.0145  ,  0.0145 ]  }}                     
            
                        
            
            
            
            
    ##################################################################################################
    #%% Plotting all individual intrinsic profiles
    #    - for a given visit
    ##################################################################################################
    if (plot_dic['all_intr_data']!=''):
        key_plot = 'all_intr_data'
        plot_settings[key_plot]={}  

        #Margins
        plot_settings[key_plot]['margins']=[0.15,0.15,0.85,0.7]

        #Data type
        plot_settings[key_plot]['data_type']='CCF' 
        if gen_dic['star_name']=='HD209458':plot_settings[key_plot]['data_type']='spec2D' 
        
        #Plot profiles aligned or not
        plot_settings[key_plot]['aligned'] = False

        #Scaling factor (in power of ten, ie flux are multiplied by 10**sc_fact10)
        plot_settings[key_plot]['sc_fact10']=0.

        #Plot continuum pixels
        #    - for CCFs only
        plot_settings[key_plot]['plot_cont']= True & False

        #Instruments and visits to plot
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}   
        elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-28','2019-04-29']}  
        
        #Exposures to plot
        #    - indexes are relative to in-transit tables
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['iexp_plot']={'ESPRESSO':{
                '2018-10-31':np.arange(2,37,dtype=int),
                '2018-09-03':np.arange(1,20,dtype=int)}}
        if gen_dic['star_name']=='HD209458':
            plot_settings[key_plot]['iexp_plot']={'ESPRESSO':{
                '20190720':[7,39],     #phase -0.01261 (iexp 21 i_in 7) ; -0.00019 (iexp 37 i_in 23) ; 0.01225 (iexp 53 i_in 39) ; 
                '20190911':[7,39]}}     #phase -0.01196 (iexp 22 i_in 8) ; -0.00027 (iexp 37 i_in 23) ; 0.01217 (iexp 53 i_in 39) ; 
            
            
        #Colors
        # if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'}} 
        # elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-02-28':'dodgerblue','2019-04-29':'red'}} 
            
        #Plot boundaries
#         if gen_dic['studied_pl']=='WASP76b':
#             x_range=[3500.,8000.] 
#             x_range=[5880.,5905.] 
# #            x_range=[6200.,6300.] 
#             # x_range=None

        #Plot boundaries in flux
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['x_range']=[-120.,120.] 
            plot_settings[key_plot]['y_range']=[3e3,7e3]
            plot_settings[key_plot]['sc_fact10']=-3.
        elif gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['x_range']=[-25.,25.] 
            # y_range']=[3e3,7e3]
            plot_settings[key_plot]['sc_fact10']=0.

        if gen_dic['star_name']=='HD209458':
            plot_settings[key_plot]['x_range']=[5893.-6.1  ,5893.+6.1 ]   #Na doublet
            # plot_settings[key_plot]['x_range']=[5889.95094-2  ,5889.95094+2 ]   #D2 5889.95094
            # plot_settings[key_plot]['x_range']=[5895.92424-2  ,5895.92424+2 ]   #D1 5895.92424
            plot_settings[key_plot]['y_range']=[-0.3,1.9] 
            plot_settings[key_plot]['y_range']=[-0.5,2.] 
            plot_settings[key_plot]['sc_fact10']=0.
            
        #Overplot resampled spectra
        if gen_dic['star_name']=='HD209458':        
            plot_settings[key_plot]['resample'] = 0.08
            plot_settings[key_plot]['alpha_symb'] = 0.2
            
        #Overplot stellar lines
        if gen_dic['star_name']=='HD209458':          
            plot_settings[key_plot]['st_lines_wav'] = [5889.95094,5895.92424]      
        






    ##################################################################################################
    #%% Individual intrinsic spectral profiles
    ##################################################################################################
    if any('spec' in s for s in data_dic['Res']['type'].values()) and (plot_dic['sp_intr']!=''):
        key_plot = 'sp_intr'
        plot_settings[key_plot]={} 


        #Scaling factor (in power of ten, ie flux are multiplied by 10**sc_fact10)
        plot_settings[key_plot]['sc_fact10']=0.

        #Instruments and visits to plot
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}            

        #Colors
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'}} 
        elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-02-28':'dodgerblue','2019-04-29':'red'}} 
            
        #Plot errors on flux
        plot_settings[key_plot]['plot_err'] = False
        
        #Orders to plot
        # plot_settings[key_plot]['orders_to_plot'] = []
        # if gen_dic['star_name']=='WASP76':
        #     plot_settings[key_plot]['orders_to_plot'] = [47]            
            
        #Plot boundaries in wav
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['x_range']=[3500.,8000.] 
            plot_settings[key_plot]['x_range']=[5880.,5905.] 
#            plot_settings[key_plot]['x_range']=[6200.,6300.] 
            # plot_settings[key_plot]['x_range']=None
            plot_settings[key_plot]['x_range']=[4430.,4450.]
            plot_settings[key_plot]['x_range']=[7800.,7805.]
        if gen_dic['star_name']=='HD209458':
            plot_settings[key_plot]['x_range']=[5893.-6.1  ,5893.+6.1 ] 



        #Plot boundaries in flux
#        if gen_dic['studied_pl']=='WASP76b':
#            plot_settings[key_plot]['y_range']=[0.35,1.05] 
        if gen_dic['star_name']=='HD209458':
            plot_settings[key_plot]['y_range']=[-0.3,1.9] 








    ##################################################################################################
    #%% Individual intrinsic CCF profiles
    ##################################################################################################
    if ('CCF' in data_dic['Intr']['type'].values()) and ((plot_dic['CCFintr']!='') or (plot_dic['CCFintr_res']!='')):
        for key_plot in ['CCFintr','CCFintr_res']:
            if plot_dic[key_plot]!='':
                plot_settings[key_plot]={} 

                #Overplot continuum pixels
                plot_settings[key_plot]['plot_cont']=True  &  False

                #Normalize CCFs
                plot_settings[key_plot]['norm_prof']=True        
                    
                #Plot errors
                plot_settings[key_plot]['plot_err'] = True & False                
                
                #Choose model to use
                #    - from the fit to individual CCFs ('indiv') or from the global fit to all CCFs ('global')
                plot_settings[key_plot]['fit_type']='indiv'           
                # plot_settings[key_plot]['fit_type']='global' 
        
                #Shade area not included in fit
                plot_settings[key_plot]['plot_nofit']=True    & False
        
                #Scaling factor (in power of ten, ie flux are multiplied by 10**sc_fact10)
                plot_settings[key_plot]['sc_fact10']=-3.
        
                #Visits to plot
                if gen_dic['studied_pl']==['HD3167_c']:plot_settings[key_plot]['visits_to_plot']={'HARPN':['2016-10-01']}  
                elif gen_dic['studied_pl']=='WASP121b':
                    plot_settings[key_plot]['visits_to_plot']={'HARPS':['14-01-18','09-01-18','31-12-17'],
                                    'binned':['HARPS-binned']}          
            #        plot_settings[key_plot]['visits_to_plot']={'HARPS':['31-12-17']}     
                elif gen_dic['studied_pl']=='Kelt9b':plot_settings[key_plot]['visits_to_plot']={'HARPN':['31-07-2017']}  
                elif gen_dic['studied_pl']=='WASP76b':
                    plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}              
                    # plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03'],'binned':['ESP_binned']} 
                elif gen_dic['studied_pl']=='WASP127b':plot_settings[key_plot]['visits_to_plot']={'HARPS':['2017-03-20','2018-03-31','2018-02-13','2017-02-28']}   
                elif gen_dic['studied_pl']=='HD209458b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-07-20','2019-09-11']} 
                elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']} 
                elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-20']} 
                elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-03-18']} 
                elif gen_dic['studied_pl']=='55Cnc_e':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-02-05']} 
        
                    
                #Color dictionary
                if gen_dic['star_name']=='GJ436':
                            # plot_settings[key_plot]['color_dic']=['dodgerblue','orange']  #deux nuits HARPS-N
                            # plot_settings[key_plot]['color_dic']=['red','green']  #HARPS + HARPS-N binned
                            # plot_settings[key_plot]['color_dic']=['red','red']  #HARPS + HARPS-N binned
                    #        plot_settings[key_plot]['color_dic']=['red']  #HAR1PS
                    plot_settings[key_plot]['color_dic']={'ESPRESSO':{'20190228':'dodgerblue','20190429':'red'},'HARPN':{'20160318':'orange','20160411':'limegreen'},'HARPS':{'20070509':'magenta'}}     
            
                if gen_dic['studied_pl']=='55Cnc_e':
            
                    if data_dic['instrum_list']==['binned','HARPS']:
                        if gen_dic['n_visits_tot']==5:plot_settings[key_plot]['color_dic']=['orange','purple','dodgerblue','limegreen','red']   #4 nuits binned
                    if data_dic['instrum_list']==['HARPN']:plot_settings[key_plot]['color_dic']=['dodgerblue']
                
                    plot_settings[key_plot]['color_dic']={
                    
                        #SOPHIE  
                        '2012-02-02':'dodgerblue',
                    
                        #Nuits binned        
                        'all_HARPSS':'dodgerblue',            
                        'all_HARPS_adj':'dodgerblue',
                        'all_HARPS_adj2':'cyan',            
                        'best_HARPSN_adj':'red',
                        'good_HARPSN_adj':'orange',
                        'HARPS_HARPSN_binHARPS':'black',
                        'HARPS_HARPSN_binHARPSN':'black',
                        'best_HARPSN_adj_short':'lime',
                        'best_HARPSN_adj_long':'red',            
                        }        
                    plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2020-02-05':'dodgerblue'}}
            
                elif gen_dic['studied_pl']==['HD3167_c']:
                    plot_settings[key_plot]['color_dic']={'HARPN':{'2016-10-01':'dodgerblue'}}
            
                elif gen_dic['studied_pl']=='WASP121b':
                    plot_settings[key_plot]['color_dic']={'09-01-18':'green','14-01-18':'dodgerblue','31-12-17':'red',
                                'HARPS-binned':'orange','HARPS-binned-2018':'orange'}      
            
                elif gen_dic['studied_pl']=='Kelt9b':
                    plot_settings[key_plot]['color_dic']={'31-07-2017':'dodgerblue'}
                elif gen_dic['studied_pl']=='WASP76b':
                    plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'},'binned':{'ESP_binned':'black'}} 
                elif gen_dic['studied_pl']=='WASP127b':plot_settings[key_plot]['color_dic']={'HARPS':{'2017-03-20':'dodgerblue','2018-03-31':'green','2018-02-13':'orange','2017-02-28':'red'}} 
                elif gen_dic['star_name']=='HD209458':
                    plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-07-20':'dodgerblue','2019-09-11':'red'}}
                    plot_settings[key_plot]['color_dic']={'ESPRESSO':{'mock_vis':'dodgerblue'}}            
                
        
                #Bornes du plot en RV
                if gen_dic['studied_pl']=='WASP_8b':
                    plot_settings[key_plot]['x_range']=[-23.,20.] 
                elif gen_dic['star_name']=='GJ436':
                    plot_settings[key_plot]['x_range']=[-21.,21.]
                    plot_settings[key_plot]['x_range']=[-26.,26.]  #GJ436b, plage du continu
                    plot_settings[key_plot]['x_range']=[-21.,21.]   
                    plot_settings[key_plot]['x_range']=[-26.,26.]  #GJ436b, plage du continu
                    plot_settings[key_plot]['x_range']=[-20.,20.] 
                    plot_settings[key_plot]['x_range']=[-40.,40.] 
                elif gen_dic['studied_pl']==['HD3167_c']:
                    plot_settings[key_plot]['x_range']=[-20.,20.]    
                    plot_settings[key_plot]['x_range']=[-50.,50.]            
                elif gen_dic['studied_pl']=='WASP121b':
                    plot_settings[key_plot]['x_range']=[-50.,50.]
                    plot_settings[key_plot]['x_range']=[-90.,90.]    #Mask F
                elif gen_dic['studied_pl']=='Kelt9b':
                    plot_settings[key_plot]['x_range']=[-300.,300.]    #Mask F   
                elif gen_dic['studied_pl']=='WASP76b':
                    plot_settings[key_plot]['x_range']=[-110.,110.] 
                    # plot_settings[key_plot]['x_range']=[-160.,160.] 
                elif gen_dic['studied_pl']=='WASP127b':plot_settings[key_plot]['x_range']=[-22.,22.] 
                elif gen_dic['star_name']=='HD209458':
                    plot_settings[key_plot]['x_range']=[-20.+4.,12.-4.] 
                    
                elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['x_range']=[-71.,71.] 
                elif gen_dic['studied_pl']=='Corot7b':plot_settings[key_plot]['x_range']=[-21.,21.]  
                elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['x_range']=[-21.,21.] 
                elif gen_dic['studied_pl']=='GJ9827d':plot_settings[key_plot]['x_range']=[-21.,21.]
                elif gen_dic['studied_pl']==['TOI858b']:plot_settings[key_plot]['x_range']=[-30.,30] 
                elif gen_dic['star_name']=='HIP41378':
                    plot_settings[key_plot]['x_range']=[-51.,51] 
                    plot_settings[key_plot]['x_range']=[-101.,101] 
                elif gen_dic['star_name']=='MASCARA1':
                    plot_settings[key_plot]['x_range']=[-130.,130] 
                elif gen_dic['star_name']=='WASP47':
                    plot_settings[key_plot]['x_range']=[-51.,51]         
                elif gen_dic['star_name']=='WASP43':plot_settings[key_plot]['x_range']=[-17.,17]  
                elif gen_dic['star_name']=='L98_59':plot_settings[key_plot]['x_range']=[-19.,19.]
                elif gen_dic['star_name']=='GJ1214':plot_settings[key_plot]['x_range']=[-15.,15]  
                
                
                

                #Line width
                plot_settings[key_plot]['lw_plot'] = 1.5
                
                #Linestyle
                # plot_settings[key_plot]['ls_plot'] = '--'                
                # plot_settings[key_plot]['ls_plot'] = ':'  
                
                #Transparent background
                plot_settings[key_plot]['transparent'] = True & False

                #Font size
                plot_settings[key_plot]['font_size']=18
        
                #Hide axis
                plot_settings[key_plot]['hide_axis'] = True& False




        #Plot each intrinsic CCF and its fit
        if (plot_dic['CCFintr']!=''):
            key_plot = 'CCFintr'


            #Plot aligned CCFs
            plot_settings[key_plot]['aligned']=False
            
            #Overplot fit (aligned option must not be requested)
            plot_settings[key_plot]['plot_line_model']=True    &   False

            #Print CCFs fit properties on plot
            plot_settings[key_plot]['plot_prop']=True   #   &   False
            
            #Plot measured centroid position
            plot_settings[key_plot]['plot_line_fit_rv']=True

            #Plot stellar rest velocity
            plot_settings[key_plot]['plot_refvel']=True    &   False
            
            #Plot fitted pixels 
            plot_settings[key_plot]['plot_fitpix']=True    &   False
    
            #Plot continuum level
            plot_settings[key_plot]['plot_cont_lev']=True  #   &   False

            #Plot stellar rest velocity
            plot_settings[key_plot]['plot_refvel']=True    &   False



            #Bornes du plot
            #    - true fluxes, before scaling factor
            plot_settings[key_plot]['sc_fact10']=0. 
            if gen_dic['star_name']=='GJ436':
                 #Bornes du plot en RV
            #     plot_settings[key_plot]['y_range']=[-0.4,1.]
                plot_settings[key_plot]['y_range']=[-1.5e-3,1.5e-3]  #None
                plot_settings[key_plot]['y_range']=[-8e-4,8e-4]      #None
        
                plot_settings[key_plot]['y_range']=[-0.005,0.018]
        
                plot_settings[key_plot]['y_range']=[-2  ,  9]
        #       plot_settings[key_plot]['y_range']=[-0.5  ,  7.5] #binned HARPSN visits
                plot_settings[key_plot]['y_range']=[-0.001,0.008]
                
                plot_settings[key_plot]['y_range']=[0.6,1.3]         #ESPRESSO profiles        
                plot_settings[key_plot]['y_range']=[0.2,1.8]         #All         
        
            elif gen_dic['studied_pl']=='55Cnc_e':
                plot_settings[key_plot]['y_range']=[-4,8]     
                plot_settings[key_plot]['y_range']=[-3.,6.] 
            elif gen_dic['studied_pl']==['HD3167_c']:
                plot_settings[key_plot]['y_range']=[-1.5,2.5]   
            elif gen_dic['studied_pl']=='WASP121b':
                plot_settings[key_plot]['y_range']=[-3.,25.] 
                plot_settings[key_plot]['y_range']=[-3.,25.] 
            elif gen_dic['studied_pl']=='Kelt9b':
                plot_settings[key_plot]['y_range']=[-2.42  ,  9.23]          
            elif gen_dic['studied_pl']=='WASP76b':
                plot_settings[key_plot]['y_range']=[-1.,19.]
                plot_settings[key_plot]['y_range']=[3e3,7.5e3]
                plot_settings[key_plot]['sc_fact10']=-3.                
                # plot_settings[key_plot]['y_range'] = None
            elif gen_dic['studied_pl']=='WASP127b':
                plot_settings[key_plot]['y_range']=[-0.003,0.015]
            elif gen_dic['star_name']=='HD209458':
                # plot_settings[key_plot]['y_range']=[-0.001,17e-3]
                # plot_settings[key_plot]['y_range'] = None
                # plot_settings[key_plot]['sc_fact10']=3.
                plot_settings[key_plot]['y_range']=[0.35,1.15]   #ANTARESS I precision


            elif gen_dic['studied_pl']=='Corot7b':
                plot_settings[key_plot]['y_range']=[-10.,15.] 
                plot_settings[key_plot]['sc_fact10']=0.
            elif gen_dic['studied_pl']=='Nu2Lupi_c':
                plot_settings[key_plot]['y_range']=[0.,1.5] 
                plot_settings[key_plot]['sc_fact10']=0.                
            elif gen_dic['studied_pl']=='GJ9827d':
                plot_settings[key_plot]['y_range']=[-1.,3.]    #ESPRESSO
                # plot_settings[key_plot]['y_range']=[-5.,7.]    #HARPS               
            elif gen_dic['studied_pl']=='GJ9827b':
                plot_settings[key_plot]['y_range']=[-15.,15.]    #HARPS              
            elif gen_dic['studied_pl']==['HD3167_b']:
                plot_settings[key_plot]['y_range']=[-6.,8.]               
            elif gen_dic['studied_pl']==['TOI858b']:
                plot_settings[key_plot]['y_range']=[-0.5,2.] 
            elif gen_dic['star_name']=='HIP41378':
                plot_settings[key_plot]['y_range']=[-5.,6.]     
            elif gen_dic['star_name']=='MASCARA1':
                plot_settings[key_plot]['y_range']=[0.3,1.3]   
                
            #RM survey
            if gen_dic['star_name']=='Kepler68':plot_settings[key_plot]['y_range']=[-8.,8.]  
            elif gen_dic['star_name']=='HAT_P33':plot_settings[key_plot]['y_range']=[0.,2.] 
            elif gen_dic['star_name']=='K2_105':plot_settings[key_plot]['y_range']=[-3.,5.]  
            elif gen_dic['star_name']=='HD89345':plot_settings[key_plot]['y_range']=[-1.5,4.]   
            elif gen_dic['star_name']=='HAT_P49':plot_settings[key_plot]['y_range']=[-0.5,3.]   
            elif gen_dic['star_name']=='Kepler63':plot_settings[key_plot]['y_range']=[-0.5,3.]   
            elif gen_dic['star_name']=='WASP107':plot_settings[key_plot]['y_range']=[0.,1.5]   
            elif gen_dic['star_name']=='WASP166':plot_settings[key_plot]['y_range']=[0.,2.]  
            elif gen_dic['star_name']=='HAT_P11':plot_settings[key_plot]['y_range']=[0.2,1.5]    
            elif gen_dic['star_name']=='WASP156':plot_settings[key_plot]['y_range']=[0.,2.]     
            elif gen_dic['star_name']=='HD106315':plot_settings[key_plot]['y_range']=[-2.,5.]   
            elif gen_dic['star_name']=='WASP47':plot_settings[key_plot]['y_range']=[-4.,6.]     
            elif gen_dic['star_name']=='WASP43':plot_settings[key_plot]['y_range']=[-0.5,2.5]  
            elif gen_dic['star_name']=='L98_59':plot_settings[key_plot]['y_range']=[-2.,5.]            
            elif gen_dic['star_name']=='GJ1214':plot_settings[key_plot]['y_range']=[0.3,1.5]     



        #Plot residuals between the intrinsic CCFs and their fit
        if (plot_dic['CCFintr_res']!='') and ((gen_dic['fit_Intr']) or (gen_dic['fit_IntrProf'])):
            key_plot = 'CCFintr_res'

            #Print dispersions of residuals in various ranges
            plot_settings[key_plot]['plot_prop']=True #  &   False

            #Bornes du plot en RV 
            if gen_dic['studied_pl']=='WASP_8b':     
                plot_settings[key_plot]['y_range']=[0.4,1.1]
            if gen_dic['star_name']=='GJ436':         
                plot_settings[key_plot]['y_range']=[-1e-2,1e-2]  #None
                plot_settings[key_plot]['y_range']=[-3e-2,7e-2]  #None            
                plot_settings[key_plot]['y_range']=[-2.5e-2,2.5e-2]  #None   
                
            if gen_dic['studied_pl']=='55Cnc_e':
                plot_settings[key_plot]['y_range']=[-2.5e-2,2.5e-2]
            if gen_dic['studied_pl']==['HD3167_c']:
                plot_settings[key_plot]['y_range']=[-1.5e-2,1.5e-2]               
            if gen_dic['studied_pl']=='WASP121b':
                plot_settings[key_plot]['y_range']=[-2e-2,2e-2]                  
                plot_settings[key_plot]['y_range']=[-1.5e-2,1.5e-2]              
            elif gen_dic['studied_pl']=='WASP76b':  
                plot_settings[key_plot]['y_range']=[-400.,400.]
                plot_settings[key_plot]['sc_fact10']=-3.
                # plot_settings[key_plot]['y_range']=None

   




    ##################################################################################################
    #%% Individual binned intrinsic spectra
    ##################################################################################################
    if any('spec' in s for s in data_dic['Res']['type'].values()) and (plot_dic['sp_Intrbin']!=''):
        key_plot = 'sp_Intrbin'
        plot_settings[key_plot]={}  

        #Choose bin dimension
        #    - 'phase', 'xp_abs', 'r_proj' (see details and routine)
        plot_settings[key_plot]['dim_plot']='r_proj' 
     
        #Plot errors
        plot_settings[key_plot]['plot_err']=True  &  False   


        #Scaling factor (in power of ten, ie flux are multiplied by 10**sc_fact10)
        plot_settings[key_plot]['sc_fact10']=0.

        #Overplot resampled spectra
        if gen_dic['star_name'] in ['HD209458','WASP76']:        
            plot_settings[key_plot]['resample'] = 0.08
            plot_settings[key_plot]['alpha_symb'] = 0.2

        #Instruments and visits to plot
        if gen_dic['star_name'] in ['HD209458','WASP76']:
            plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['binned']}            

        #Colors
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'}} 
            
        #Plot boundaries in wav
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['x_range']=[3500.,8000.] 
            plot_settings[key_plot]['x_range']=[5880.,5905.]   #Na doublet
            plot_settings[key_plot]['x_range']=[5500.,5600.] 
            # plot_settings[key_plot]['x_range']=[6095.,6110.] 
            # plot_settings[key_plot]['x_range'] = None

        #Plot boundaries in flux
#        if gen_dic['studied_pl']=='WASP76b':
#            plot_settings[key_plot]['y_range']=[0.35,1.05] 
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['y_range']=[-3.2,12.] 
            plot_settings[key_plot]['y_range']=[-0.5,2.5] 
            plot_settings[key_plot]['y_range']=None 








    '''
    Plotting individual binned intrinsic CCF profiles
    '''
    if ('CCF' in data_dic['Res']['type'].values()) and ((plot_dic['CCF_Intrbin']!='') or (plot_dic['CCF_Intrbin_res']!='')):
        for key_plot in ['CCF_Intrbin','CCF_Intrbin_res']:
            if plot_dic[key_plot]!='':       
        
                #Overplot continuum pixels
                plot_settings[key_plot]['plot_cont']=True  &  False                

                #Plot reference level
                plot_settings[key_plot]['ref_level'] = True&  False

                #Plot null velocity
                plot_settings[key_plot]['plot_refvel'] = True   &  False    
        
                #Plot errors
                plot_settings[key_plot]['plot_err']=True  # &  False        

                #Choose bin dimension
                #    - 'phase', 'xp_abs', 'r_proj' (see details and routine)
                plot_settings[key_plot]['dim_plot']='r_proj'   
                plot_settings[key_plot]['dim_plot']='phase'  

                #Shade area not included in fit
                plot_settings[key_plot]['plot_nofit']=True    & False        

                #Visits to plot
                #    - use 'binned' as visit name to plot profiles binned over several visits
                if gen_dic['star_name']=='HD3167':
                    plot_settings[key_plot]['visits_to_plot']={'HARPN':['2016-10-01'],'ESPRESSO':['2019-10-09']} 
                elif gen_dic['studied_pl']=='GJ9827b':plot_settings[key_plot]['visits_to_plot']={'HARPS':['binned']} 
                elif gen_dic['star_name']=='GJ436':plot_settings[key_plot]['visits_to_plot']['ESPRESSO']+=['binned'] 
                elif gen_dic['star_name']=='WASP166':plot_settings[key_plot]['visits_to_plot']['HARPS']+=['binned'] 
                elif gen_dic['star_name']=='HD106315':plot_settings[key_plot]['visits_to_plot']['HARPS']+=['binned'] 
                elif gen_dic['star_name']=='HAT_P11':
                    if 'HARPN' in plot_settings[key_plot]['visits_to_plot']:plot_settings[key_plot]['visits_to_plot']['HARPN']+=['binned'] 
                    if 'CARMENES_VIS' in plot_settings[key_plot]['visits_to_plot']:plot_settings[key_plot]['visits_to_plot']['CARMENES_VIS']+=['binned'] 
                elif gen_dic['star_name']=='WASP107':plot_settings[key_plot]['visits_to_plot']['HARPS']+=['binned'] 
                elif gen_dic['star_name']=='55Cnc':plot_settings[key_plot]['visits_to_plot']['ESPRESSO']+=['binned'] 
                
                #Color dictionary
                # if gen_dic['star_name']=='WASP107':plot_settings[key_plot]['color_dic']['HARPS']={'binned':'orange'} 
                if gen_dic['star_name']=='WASP107':plot_settings[key_plot]['color_dic']['HARPS']={'binned':'purple'} 
                    
                #Bornes du plot en RV
                plot_settings[key_plot]['x_range']=[-21.,21.] 
                if gen_dic['star_name']=='HD3167':
                    plot_settings[key_plot]['x_range']=[-41.,41.]             
                    if gen_dic['studied_pl']==['HD3167_c']:
                        plot_settings[key_plot]['x_range']=[-71.,71.] 
                        # plot_settings[key_plot]['x_range']=[-31.,31.]  
                    elif gen_dic['studied_pl']==['HD3167_b']:
                        plot_settings[key_plot]['x_range']=[-71.,71.] 
                        plot_settings[key_plot]['x_range']=[-100.,420.] 
                        # plot_settings[key_plot]['x_range']=[-420.,100.] 
                        plot_settings[key_plot]['x_range']=[-31.,31.]
                        # plot_settings[key_plot]['x_range']=np.array([-31.,31.])-160.
                        # plot_settings[key_plot]['x_range']=np.array([-31.,31.])+160.
                elif gen_dic['star_name']=='HIP41378':
                    plot_settings[key_plot]['x_range']=[-41.,41.]  
                #RM survey
                elif gen_dic['star_name'] in ['HAT_P3','HD89345','K2_105','Kepler25','Kepler63','Kepler68','WASP47']:      
                    plot_settings[key_plot]['x_range']=[-41.,41.] 
                elif gen_dic['star_name']=='HAT_P11':
                    plot_settings[key_plot]['x_range']=[-36.,36.]    #mask is cut in HARPS-N
                elif gen_dic['star_name'] in ['HAT_P33','HD106315','HAT_P49','WASP107']:
                    plot_settings[key_plot]['x_range']=[-51.,51.]                 
                elif gen_dic['star_name']=='WASP166':
                    plot_settings[key_plot]['x_range']=[-46.,46.] 
                elif gen_dic['star_name']=='WASP156':
                    plot_settings[key_plot]['x_range']=[-41.,41.]              
                elif gen_dic['star_name']=='55Cnc': 
                    plot_settings[key_plot]['x_range']=[-81.,81.]     
                    plot_settings[key_plot]['x_range']=[-36.,36.] 

        
        
        #Plot each intrinsic CCF and its fit
        if (plot_dic['CCF_Intrbin']!=''):
            key_plot = 'CCF_Intrbin'        

            #Overplot fit 
            plot_settings[key_plot]['plot_line_model']=True  #  &   False        


            #Print CCFs fit properties on plot
            plot_settings[key_plot]['plot_prop']=True      &   False
            
            #Plot measured centroid
            plot_settings[key_plot]['plot_line_fit_rv']=True    #& False

            #Plot fitted pixels 
            plot_settings[key_plot]['plot_fitpix']=True    &   False
    
            #Plot continuum pixels specific to each exposure 
            plot_settings[key_plot]['plot_cont_exp']=True     &   False


            #Bornes du plot
            #    - true fluxes, before scaling factor
            if gen_dic['star_name']=='HD3167': 
                plot_settings[key_plot]['y_range']=[-0.3,2.]  
                if gen_dic['studied_pl']==['HD3167_c']:
                    plot_settings[key_plot]['y_range']=[0.1,1.3]
                elif gen_dic['studied_pl']==['HD3167_b']:
                    plot_settings[key_plot]['y_range']=[-6.,8.] 
            elif gen_dic['studied_pl']=='GJ9827b':
                plot_settings[key_plot]['y_range']=[-5.,5.]                
            elif gen_dic['star_name']=='HIP41378':
                plot_settings[key_plot]['y_range']=[-0.5,2.]
            #RM survey
            elif gen_dic['star_name']=='HAT_P3':
                plot_settings[key_plot]['y_range']=[0.4,1.2]
            elif gen_dic['star_name']=='HAT_P11':
                plot_settings[key_plot]['y_range']=[0.3,1.4]                
                plot_settings[key_plot]['y_range']=[0.4,1.15] #paper
            elif gen_dic['star_name']=='HAT_P33':
                plot_settings[key_plot]['y_range']=[0.4,1.2]   
            elif gen_dic['star_name']=='HAT_P49':
                plot_settings[key_plot]['y_range']=[0.45,1.25]   
            elif gen_dic['star_name']=='HD89345':
                plot_settings[key_plot]['y_range']=[0.3,1.5] 
            elif gen_dic['star_name']=='HD106315':
                plot_settings[key_plot]['y_range']=[0.3,1.35]  
            elif gen_dic['star_name']=='K2_105':
                plot_settings[key_plot]['y_range']=[-0.3,2.2] 
            elif gen_dic['star_name']=='Kepler25':
                plot_settings[key_plot]['y_range']=[-0.5,2.4]            
            elif gen_dic['star_name']=='Kepler63':
                plot_settings[key_plot]['y_range']=[0.2,1.7]  
            elif gen_dic['star_name']=='Kepler68':
                plot_settings[key_plot]['y_range']=[-1.5,3.5]                  
            elif gen_dic['star_name']=='WASP107':
                plot_settings[key_plot]['y_range']=[0.4,1.15]  
                plot_settings[key_plot]['y_range']=[0.45,1.1]   #compa
            elif gen_dic['star_name']=='WASP166':
                plot_settings[key_plot]['y_range']=[0.4,1.2]  
            elif gen_dic['star_name']=='WASP47':
                plot_settings[key_plot]['y_range']=[-2.,3.1]  
            elif gen_dic['star_name']=='WASP156':
                plot_settings[key_plot]['y_range']=[0.4,1.4]  
            elif gen_dic['star_name']=='55Cnc':            
                plot_settings[key_plot]['y_range']=[0.25,1.4]     
                        
        
        #Plot each residual
        if (plot_dic['CCF_Intrbin_res']!=''):
            key_plot = 'CCF_Intrbin_res'           

            #Print dispersions of residuals in various ranges
            plot_settings[key_plot]['plot_prop']=True #  &   False
            
            #Bornes du plot en RV 
            if gen_dic['studied_pl']==['HD3167_c']:
                plot_settings[key_plot]['y_range']=[-1.5e-2,1.5e-2]               
            
            
            
            
            

    ################################################################################################################   
    #%% Individual 1D intrinsic profiles
    ################################################################################################################   
    if (any('spec' in s for s in data_dic['Intr']['type'].values())) and (plot_dic['sp_Intr_1D']!=''):  
        key_plot = 'sp_Intr_1D'
        plot_settings[key_plot]={}            
        
        #Plot errors
        plot_settings[key_plot]['plot_err'] = False


        #Instruments and visits to plot
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}            

        #Colors
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'}} 

        #Overplot resampled spectra
        if gen_dic['star_name'] in ['HD209458','WASP76']:         
            plot_settings[key_plot]['resample'] = 0.08*10
            plot_settings[key_plot]['alpha_symb'] = 0.2
            
        #Plot boundaries in wav
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['x_range']=[3500.,8000.] 
            plot_settings[key_plot]['x_range']=[5880.,5905.] 
            plot_settings[key_plot]['x_range']=[3800.,7900.] 
            # plot_settings[key_plot]['x_range=None

        #Plot boundaries in signal
        if gen_dic['star_name']=='WASP76':
            plot_settings[key_plot]['y_range']=[-2.,3.]     
        if gen_dic['star_name']=='HD209458':
            plot_settings[key_plot]['y_range']=[-0.5,2.]      
    
            
        



    '''
    Plotting chi2 values for each fitted local property
    '''
    if (plot_dic['chi2_fit_loc_prop']!=''):
        key_plot = 'chi2_fit_loc_prop'
        plot_settings[key_plot]={}   

        #Ranges specific to each band
        #    - set to None for automatic determination
        # plot_settings[key_plot]['x_range']=[-0.021,0.021]

        #Threshold to identify and print outliers
        #    - set to None to prevent
        plot_settings[key_plot]['chi2_thresh']=3.  

        #Property
        plot_settings[key_plot]['prop'] = 'rv'

        #General path to the best-fit model to property series
        plot_settings[key_plot]['IntrProp_path']=None
        if gen_dic['star_name']=='HAT_P3':plot_settings[key_plot]['IntrProp_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/HAT_P3b_Saved_data/Joined_fits/Intr_prop/Orig/chi2/'
        elif gen_dic['star_name']=='HAT_P33':plot_settings[key_plot]['IntrProp_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/HAT_P33b_Saved_data/Joined_fits/Intr_prop/Orig/chi2/'

        #Visits to plot
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']} 
        elif gen_dic['studied_pl']=='HD209458b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-07-20','2019-09-11']} 
        elif gen_dic['studied_pl']==['HD3167_c']:plot_settings[key_plot]['visits_to_plot']={'HARPN':['2016-10-01']}

        #Colors
        if gen_dic['studied_pl']=='WASP76b':            
            plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'}}
        elif gen_dic['studied_pl']=='HD209458b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-07-20':'dodgerblue','2019-09-11':'red'}}
        elif gen_dic['studied_pl']=='GJ9827d':plot_settings[key_plot]['color_dic']={'ESPRESSO':['2019-08-25']} 
        




    '''
    Plotting range of properties covered by the planet in each exposure 
    '''
    if (plot_dic['plocc_ranges']!=''):
        key_plot = 'plocc_ranges'
        plot_settings[key_plot]={} 

        #Visits to plot
        if gen_dic['studied_pl']=='WASP76b':        plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']} 

        #Print exposure indexes
        #    - relative to in-transit tables
        plot_settings[key_plot]['plot_expid']=True     

        #Choose values to plot among:
        #    - 'mu', 'lat', 'lon', 'x_st', 'y_st', 'xp_abs', 'r_proj'
        plot_settings[key_plot]['x_prop']='r_proj' 
        
        #Gap between exposures
        plot_settings[key_plot]['y_gap'] = 0.2

        #Abscissa ranges
        if gen_dic['studied_pl']=='WASP76b':
           if plot_settings[key_plot]['x_prop']=='mu':plot_settings[key_plot]['x_range']=[-0.1,1.1]    
           if plot_settings[key_plot]['x_prop']=='xp_abs':plot_settings[key_plot]['x_range']=[-0.02,1.02]  
           if plot_settings[key_plot]['x_prop']=='r_proj':plot_settings[key_plot]['x_range']=[0.03,1.02]







    '''
    Plotting 1D PDFs from analysis of individual profiles
    '''
    if (plot_dic['propCCF_DI_mcmc_PDFs']!='') or (plot_dic['propCCFloc_mcmc_PDFs']!=''):
        for key_plot in ['propCCF_DI_mcmc_PDFs','propCCFloc_mcmc_PDFs']:
            if plot_dic[key_plot]!='':
                plot_settings[key_plot]={} 

                #Figure size and margins       
                if gen_dic['star_name']=='HD3167':       
                    plot_settings[key_plot]['fig_size']=(10,3.)
                    plot_settings[key_plot]['margins']=[0.02,0.15,0.98,0.98] 
                if gen_dic['star_name']=='TOI858':       
                    plot_settings[key_plot]['fig_size']=(10,2.)
                    plot_settings[key_plot]['margins']=[0.02,0.3,0.98,0.9] 
                if gen_dic['star_name']=='GJ436':       
                    plot_settings[key_plot]['fig_size']=(10,3.)
                    plot_settings[key_plot]['fig_size']=(5,3.)     #ESP   
                    plot_settings[key_plot]['fig_size']=(10,3.)     #HARPS   
                    plot_settings[key_plot]['margins']=[0.02,0.2,0.98,0.9] 
                if gen_dic['star_name']=='HD89345':       
                    plot_settings[key_plot]['fig_size']=(10,15.)

                #Visits to plot
                #    - add '_bin' to the name of a visit to plot properties derived from intrinsic profiles binned within a visit
                #    - use 'binned' as visit name to plot properties derived from intrinsic profiles binned over several visits
                if gen_dic['studied_pl']==['HD3167_c']:plot_settings[key_plot]['visits_to_plot']={'HARPN':['2016-10-01']} 
                elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']} 
                elif gen_dic['studied_pl']==['TOI858b']:plot_settings[key_plot]['visits_to_plot']={'CORALIE':['20191205','20210118']} 

                #Choose property to plot
                #    - 'rv', 'FWHM', 'ctrst', 'true_amp', 'true_ctrst', 'true_FWHM'
                plot_settings[key_plot]['plot_prop_list']=['rv','ctrst','FWHM']
        
                #Indexes of exposures to be plotted
                #    - all exposures are plotted if left empty
                # if gen_dic['studied_pl']=='WASP76b':
                #     plot_settings[key_plot]['iexp_plot']={'ESPRESSO':{'2018-10-31':[0],'2018-09-03':[28]}}
                # if gen_dic['star_name']=='GJ436':
                #     plot_settings[key_plot]['iexp_plot']={'ESPRESSO':{'20190228':range(1,9),'20190429':range(1,9)}}        
                if gen_dic['star_name']=='HAT_P3':plot_settings[key_plot]['iexp_plot']={'HARPN':{'20200130':range(1,8)}}  
                elif gen_dic['star_name']=='Kepler25':plot_settings[key_plot]['iexp_plot']={'HARPN':{'20190614':range(1,19)}}  
                elif gen_dic['star_name']=='HAT_P33':plot_settings[key_plot]['iexp_plot']={'HARPN':{'20191204':range(1,33)}}          
                elif gen_dic['star_name']=='HD89345':plot_settings[key_plot]['iexp_plot']={'HARPN':{'20200202':range(2,93)}}            
                elif gen_dic['star_name']=='Kepler63':plot_settings[key_plot]['iexp_plot']={'HARPN':{'20200513':range(9)}}            
                elif gen_dic['star_name']=='HAT_P49':plot_settings[key_plot]['iexp_plot']={'HARPN':{'20200730':range(3,71)}}         
                elif gen_dic['star_name']=='WASP107':plot_settings[key_plot]['iexp_plot']={'CARMENES_VIS':{'20180224':range(1,9)},'HARPS':{'20140406':range(1,11),'20180201':range(1,12),'20180313':range(1,12)}}        
                elif gen_dic['star_name']=='WASP166':plot_settings[key_plot]['iexp_plot']={'HARPS':{'20170114':range(1,39),'20170304':range(1,38),'20170315':range(1,33)}}        
                elif gen_dic['star_name']=='HAT_P11':plot_settings[key_plot]['iexp_plot']={'HARPN':{'20150913':range(2,26),'20151101':range(2,25)},'CARMENES_VIS':{'20170807':range(1,17),'20170812':range(2,18)}}                 
                elif gen_dic['star_name']=='WASP156':plot_settings[key_plot]['iexp_plot']={'CARMENES_VIS':{'20190928':range(1,7),'20191025':range(1,6)}}                 
                elif gen_dic['star_name']=='HD106315':
                    # plot_settings[key_plot]['iexp_plot']={'HARPS':{'20170309':range(2,39),'20170330':range(2,24),'20180323':range(2,23)}}                  
                    plot_settings[key_plot]['iexp_plot']={'HARPS':{'20170309':range(3,44),'20170330':range(1,26),'20180323':range(1,27)}} 
    
                #Retrieve mcmc run for a given number of walkers and steps
                if any(x in gen_dic['studied_pl'] for x in ['HD3167_c','HD3167_b','TOI858b','GJ436_b']):
                    plot_settings[key_plot]['nwalkers'] = 100
                    plot_settings[key_plot]['nsteps'] = 2000
                elif gen_dic['star_name'] in ['HAT_P3','Kepler25','Kepler68','HAT_P33','K2_105','HD89345','WASP107','WASP166','HAT_P11','WASP156','HD106315','Kepler63','HIP41378','WASP47']:
                    plot_settings[key_plot]['nwalkers'] = 50
                    plot_settings[key_plot]['nsteps'] = 1000
                elif gen_dic['star_name'] in ['HAT_P49']:
                    plot_settings[key_plot]['nwalkers'] = 100
                    plot_settings[key_plot]['nsteps'] = 1500  

                #Number of subplots per row (>=1)
                if gen_dic['studied_pl']==['HD3167_c']:plot_settings[key_plot]['nsub_col'] = 10
                elif gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['nsub_col'] = 9
                elif gen_dic['studied_pl']==['TOI858b']:plot_settings[key_plot]['nsub_col'] = 7
                elif gen_dic['studied_pl']==['GJ436_b']:
                    # plot_settings[key_plot]['nsub_col'] = 4  #ESP
                    # plot_settings[key_plot]['nsub_col'] = 8  #HARPS
                    plot_settings[key_plot]['nsub_col'] = 3  #HARPS-N
                elif gen_dic['star_name'] in ['HIP41378']:plot_settings[key_plot]['nsub_col']=4
                elif gen_dic['star_name'] in ['HAT_P3','WASP156','Kepler63']:plot_settings[key_plot]['nsub_col']=4
                elif gen_dic['star_name'] in ['Kepler25','HAT_P11']:plot_settings[key_plot]['nsub_col']=9
                elif gen_dic['star_name'] in ['Kepler68','HAT_P33','K2_105','HD89345','HAT_P49','WASP166','HD106315']:plot_settings[key_plot]['nsub_col']=10
                elif gen_dic['star_name'] in ['WASP107','WASP47']:plot_settings[key_plot]['nsub_col']=7
                        
                #Spacing between subplots
                if gen_dic['studied_pl']==['GJ436_b']:
                    plot_settings[key_plot]['wspace'] = 0.06
                    plot_settings[key_plot]['hspace'] = 0.06                
                
                #Plot 1 sigma HDI or confidence intervals
                plot_settings[key_plot]['plot_conf_mode']=['HDI','quant']
                plot_settings[key_plot]['plot_conf_mode']=['HDI']
                
                #Number of bins in histograms
                plot_settings[key_plot]['bins_par'] = 30  #50
                plot_settings[key_plot]['bins_par'] = 40  #50
        
                #Common ranges
                #    - set to None for automatic determination in individual subplots
                plot_settings[key_plot]['xrange_all']= {}
                if any(x in gen_dic['studied_pl'] for x in ['HD3167_c','HD3167_b']):   
                    plot_settings[key_plot]['xrange_all']={'rv': [-5,5],
                                'ctrst': [-1.99,1.99],  #[0.,0.99]
                                'FWHM': [0.,19.99]}
                if gen_dic['studied_pl']==['TOI858b']:
                    plot_settings[key_plot]['xrange_all']={'rv': [-2.,8.],
                                'FWHM': [0.,16.],
                                'ctrst': [-0.5,2.]}
                if gen_dic['studied_pl']==['GJ436_b']:
                    plot_settings[key_plot]['xrange_all']={'rv': [-0.99,0.99],
                                'FWHM':[0.,10.],
                                # 'FWHM':[2.,7.]   #paper, ESP
                                # 'FWHM':[1.,7.]   #paper, HARPS
                                # 'FWHM':[2.,7.]   #paper, HARPS-N
                                'ctrst':[0.,1.],
                                # 'ctrst':[0.101,0.499],   #paper, ESP
                                # 'ctrst':[0.,0.8],   #paper, HARPS
                                # 'ctrst':[0.,0.7]   #paper, HARPS-N
                                }
                    plot_settings[key_plot]['xrange_all']['true_FWHM'] = plot_settings[key_plot]['xrange_all']['FWHM']
                    plot_settings[key_plot]['xrange_all']['true_ctrst'] = plot_settings[key_plot]['xrange_all']['ctrst']
                plot_settings[key_plot]['yrange'] = [0,1e4]         
        
                #Tick intervals
                if any(x in gen_dic['studied_pl'] for x in ['HD3167_c','HD3167_b']):   
                    plot_settings[key_plot]['xmajor_int_all']={
                        'rv':4.,
                        'ctrst':1.,
                        'FWHM': 5,
                        }
                    plot_settings[key_plot]['xminor_int_all']={
                        'rv':1.,
                        'ctrst':0.5,
                        'FWHM': 1,
                        }
        
                if gen_dic['studied_pl']==['GJ436_b']:
                    plot_settings[key_plot]['xmajor_int_all']={
                        'rv':0.5,
                        # 'ctrst':0.1       #ESP
                        # 'FWHM': 1.      #ESP
                        'FWHM': 2.      #HARPS,HARPS-N
                        }
                    plot_settings[key_plot]['xminor_int_all']={
                        'rv':0.1,
                        # 'ctrst':0.05
                        # 'FWHM': 1.   #ESP
                        'FWHM': 1.     #HARPS,HARPS-N
                        }            
                    plot_settings[key_plot]['xmajor_int_all']['true_FWHM'] = plot_settings[key_plot]['xmajor_int_all']['FWHM']
                    plot_settings[key_plot]['xminor_int_all']['true_FWHM'] = plot_settings[key_plot]['xminor_int_all']['FWHM']
                        
                
                
                







    ##################################################################################################
    #%% 2D maps: 1D intrinsic profiles
    ##################################################################################################
    if (plot_dic['map_Intr_1D']!=''):
        key_plot = 'map_Intr_1D'
        plot_settings[key_plot]={}  

        #Choice of visits to be plotted
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}  
        
        #Color range
        if gen_dic['studied_pl']=='WASP76b':
            v_range_comm=[3,7]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2018-10-31':v_range_comm,'2018-09-03':v_range_comm}}          
        
        #Ranges
        if gen_dic['studied_pl']=='WASP76b':
            # plot_settings[key_plot]['x_range']=[5880.,5905.]
            # plot_settings[key_plot]['x_range']=[-150,150]
            plot_settings[key_plot]['x_range']=[-100.,100]
            y_range_comm=[-0.048  ,  0.048]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2018-10-31':y_range_comm,'2018-09-03':y_range_comm}}           
       
        
        
        
        
        
    '''
    Plotting results of PCA analysis
    '''
    if (plot_dic['pca_ana']!=''):        
        key_plot = 'pca_ana'
        plot_settings[key_plot]={}          
        
        #Visits to plot         
        # if gen_dic['star_name']=='55Cnc':
        #     plot_settings[key_plot]['visits_to_plot']={
        #         'HARPS':['20120213']}        
        
        #Subplots
        plot_settings[key_plot]['pc_var'] = True #& False
        plot_settings[key_plot]['pc_rms'] = True #& False
        plot_settings[key_plot]['pc_bic'] = True #  & False
        plot_settings[key_plot]['pc_hist'] = True # & False
        plot_settings[key_plot]['pc_prof'] = True  & False
        plot_settings[key_plot]['fft_prof'] = True  & False

        #PC variances to overplot
        plot_settings[key_plot]['var_list'] = ['pre','post','out']

        #Principal components profiles to plot
        plot_settings[key_plot]['pc_list'] = [0,1]
        plot_settings[key_plot]['pc_col'] = ['dodgerblue','green','red']

        #FFT profiles to plot
        #    - from original residual profiles ('res'), corrected residual profiles ('corr'), or bootstrapped corrected residual profiles ('boot') 
        plot_settings[key_plot]['fft_list'] = ['res','corr','boot']

        #Bornes du plot  
        plot_settings[key_plot]['x_range_var']=None 
        plot_settings[key_plot]['y_range_var']=None 
        plot_settings[key_plot]['x_range_rms']=None 
        plot_settings[key_plot]['y_range_rms']=None 
        plot_settings[key_plot]['x_range_bic']=None 
        plot_settings[key_plot]['y_range_bic']=None 
        plot_settings[key_plot]['x_range_hist']=None 
        plot_settings[key_plot]['y_range_hist']=None 
        plot_settings[key_plot]['x_range_pc']=None 
        plot_settings[key_plot]['y_range_pc']=None 
        plot_settings[key_plot]['x_range_pc']=None 
        plot_settings[key_plot]['y_range_pc']=None         
        plot_settings[key_plot]['x_range_fft']=None 
        plot_settings[key_plot]['y_range_fft']=None          
        
        #Colors
#        if gen_dic['main_pl']=='WASP76b':
#            plot_settings[key_plot]['color_dic']={'2018-10-31':'dodgerblue','2018-09-03':'red'}  











    '''
    Plotting properties of intrinsic stellar CCFs
    '''
    if (plot_dic['prop_loc']!=''):

        #Choose values to plot in ordina (list of properties) 
        #    - properties:
        # + 'rv' : centroid of the local stellar CCFs in the star rest frame (in km/s)
        # + 'RVres' residuals from their RRM model (in km/s)        
        # + 'FWHM': width of the local CCFs (in km/s)
        # + 'ctrst': contrast of the local CCFs
        # + 'rv_l2c': RV(lobe)-RV(core) of double gaussian components
        # + 'FWHM_l2c': FWHM(lobe)/FWHM(core) of double gaussian components
        # + 'amp_l2c': contrast(lobe)/contrast(core) of double gaussian components
        plot_settings['prop_ordin']=['rv','RVres','FWHM','ctrst'] 
        plot_settings['prop_ordin']=['rv','FWHM','ctrst'] 
        plot_settings['prop_ordin']=['ctrst'] 
        # plot_settings['prop_ordin']=['rv']
        # plot_settings['prop_ordin']=['ctrst']
        # plot_settings['prop_ordin']=['FWHM']
        # plot_settings['prop_ordin']=['ctrst','FWHM']
        # plot_settings['prop_ordin']=['rv','RVres']
        # plot_settings['prop_ordin']=['rv_l2c','FWHM_l2c','amp_l2c'] 
        # plot_settings['prop_ordin']=['rv','true_FWHM','true_ctrst'] 

        #Settings for selected properties
        for plot_prop in plot_settings['prop_ordin']:
            key_plot = 'prop_'+plot_prop 
            plot_settings[key_plot]={} 

            #Plot margins
            plot_settings[key_plot]['margins']=[0.15,0.15,0.95,0.95]   #regular size 
            plot_settings[key_plot]['margins']=[0.15,0.12,0.95,0.7]    #elongated
            plot_settings[key_plot]['margins']=[0.15,0.12,0.95,0.65]    #elongated

            #Choose bin dimension to be plotted
            #    - 'phase', 'xp_abs', 'r_proj' (see details and routine)
            plot_settings[key_plot]['dim_plot']='r_proj'   
            plot_settings[key_plot]['dim_plot']='phase'  

            #Choose values to plot in absissa 
            #    - orbital phase, mu, stellar latitude, stellar longitude, projected position in the stellar frame x (along the equator) and y (along the spin axis) 
            #    - choose: 
            # + 'phase', 'cstr_loc', 'FWHM_loc', 'rv_loc' (defined for all exposures)
            # + 'mu', 'lat', 'lon', 'x_st', 'abs_y_st', 'y_st', 'y_st2', 'xp_abs', 'r_proj' (defined for in-transit exposures only, and for binned data only if binned over phase)
            plot_settings[key_plot]['prop_absc']='phase'
            plot_settings[key_plot]['prop_absc']='r_proj'
            # plot_settings[key_plot]['prop_absc']='y_st2'
    
            #Visits to plot
            #    - add '_bin' to the name of a visit to plot properties derived from intrinsic profiles binned within a visit
            #    - use 'binned' as visit name to plot properties derived from intrinsic profiles binned over several visits
            if gen_dic['studied_pl']=='WASP_8b':plot_settings[key_plot]['visits_to_plot']={'HARPS':['2008-10-04']} 
            elif gen_dic['studied_pl']=='55Cnc_e':plot_settings[key_plot]['visits_to_plot']={'binned':['all_HARPSS','all_HARPS_adj','all_HARPS_adj2','best_HARPSN_adj','best_HARPSN_adj_short','best_HARPSN_adj_long','good_HARPSN_adj','HARPS_HARPSN_binHARPS','HARPS_HARPSN_binHARPSN']}
            if gen_dic['star_name']=='HD3167': 
                plot_settings[key_plot]['visits_to_plot']={'HARPN':['2016-10-01'],'ESPRESSO':['2019-10-09']} 
            elif gen_dic['studied_pl']=='Kelt9b':plot_settings[key_plot]['visits_to_plot']={ 'HARPN':['31-07-2017']}       
            elif gen_dic['star_name']=='GJ436':
                plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['20190228','20190429'],'HARPN':['20160318','20160411'],'HARPS':['20070509']} 
            elif gen_dic['studied_pl']=='WASP121b':
                plot_settings[key_plot]['visits_to_plot']={
        #            'HARPS':['14-01-18','09-01-18','31-12-17'],'
        #            'HARPS':['14-01-18','09-01-18','31-12-17'],
                    'binned':['HARPS-binned']
                }
            elif gen_dic['studied_pl']=='WASP76b':
                plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03','2018-10-31_bin','2018-09-03_bin'],'binned':['ESP_binned']} 
            elif gen_dic['studied_pl']=='WASP127b':plot_settings[key_plot]['visits_to_plot']={'HARPS':['2017-03-20','2018-03-31','2018-02-13','2017-02-28']} 
            # elif gen_dic['studied_pl']=='HD209458b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-07-20','2019-09-11']}  
            elif gen_dic['studied_pl']=='GJ9827d':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-08-25']} 
            elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2020-03-18']} 
            # elif gen_dic['star_name']=='TOI858':
            #     plot_settings[key_plot]['visits_to_plot']={'CORALIE':['20191205']} 
            #     plot_settings[key_plot]['visits_to_plot']={'CORALIE':['20210118']}         
       
            elif gen_dic['star_name']=='WASP107':plot_settings[key_plot]['visits_to_plot']={'HARPS':['20140406','20180201','20180313']}
            elif gen_dic['star_name']=='HAT_P49':plot_settings[key_plot]['visits_to_plot']={'HARPN':['20200730_bin']}


            #Select indexes of points to be removed from the plot
            #    - leave empty otherwise
            #    - indexes are relative to in-transit / binned exposures
            if gen_dic['studied_pl']=='WASP_8b':
                plot_settings[key_plot]['idx_noplot']={'HARPS':{'2008-10-04':range(9)+range(51,74)}}  
            elif gen_dic['star_name']=='GJ436':
            #     plot_settings[key_plot]['idx_noplot']={'HARPN':{'2016-03-18':range(63)+[63]+range(72,76),          #GJ436b, en excluant seulement celles clairement pas detectees
            #                             '2016-04-11':range(20)+[20]+range(29,69)},
            #                  'HARPS':{'2007-05-09':range(6)+[14,16]+range(17,35)}}
            #     plot_settings[key_plot]['idx_noplot']={'HARPS':{'2007-05-09':range(7)+[14,15,16]+range(17,35)},   #GJ436b, en excluant les CCFs qui ne remplissent pas le critere sur le contraste
            #                 'HARPN':{'2016-03-18':range(63)+[63]+range(71,76),                        
            #                            '2016-04-11':range(20)+[20]+range(28,69)},
            #                 'binned':{'HARPSN-binned':[0,1,9,10]}}   #CCFs considered as undetected
            
                plot_settings[key_plot]['idx_noplot']={'ESPRESSO':{'20190228':[0,9],'20190429':[0,9]}, 
                            'HARPS':{'20070509':[0,1,2,9,10,11]}, 
                            'HARPN':{'20160318':[0,8],'20160411':[0,8]}}      
                
            elif gen_dic['studied_pl']=='55Cnc_e': 
                plot_settings[key_plot]['idx_noplot']={
            #        'HARPS':{'2012-02-27':range(36),'2012-01-27':range(47),'2012-03-15':range(41),'2012-02-13':range(55)},  #uncomment to plot only the binned visits
                    'binned':{  
                        'all_HARPSS':[6,7,8,32,35,36],     #exposures with no detection
                        'all_HARPS_adj':[2,3,5,26,27,28],
                        'all_HARPS_adj2':[2,3,15,16],
                        '2012-01-27_binned':[3,4,6,8,9,16,18],        
                        '2012-02-27_binned':[3,4,6,7,8,9,11,12,13,16,17,18],         
                        '2012-02-13_binned':[7,8,11,12],      
                        '2012-03-15_binned':[3,13,14,16,17,18],
                        'good_HARPSN':[26,27,44,45,46,47],
                        'best_HARPSN_adj':[26,27,40,41],
                        'good_HARPSN_adj':[26,27,39,40,41],
                        'HARPS_HARPSN_binHARPS':[18,19,31,32],
                        'HARPS_HARPSN_binHARPSN':[26,27,40,41],
                        'best_HARPSN_adj_long':[11,22,23,24],
                        'best_HARPSN_adj_short':[26,45,46],    
                            }            
                        }  
    
            elif gen_dic['star_name']=='HD3167':
                plot_settings[key_plot]['idx_noplot']={'ESPRESSO':{'2019-10-09':[16]}}    
    
                # plot_settings[key_plot]['idx_noplot']={'HARPN':{'2016-10-01':[]}}     
            #    plot_settings[key_plot]['idx_noplot']={'HARPN':{'2016-10-01':[9,24,28,29]}}      #CCF non-detecte, conservatif
            #    plot_settings[key_plot]['idx_noplot']={'HARPN':{'2016-10-01':[9,24,27,28,29]}}      #CCF non-detecte, conservatif
                # plot_settings[key_plot]['idx_noplot']={'HARPN':{'2016-10-01':[0,19,20]}}      #CCF aux limbes
                plot_settings[key_plot]['idx_noplot'].update({'HARPN':{'2016-10-01':[17,18,19]}})      #CCF 3 last expos
                
                
        #    elif gen_dic['studied_pl']=='WASP121b':
        #        plot_settings[key_plot]['idx_noplot']={'HARPS':{'14-01-18':[19,38],'09-01-18':[8,9,27,28],'31-12-17':[10,25]},
        #                    'binned':{'HARPS-binned':[15,30,31],'HARPS-binned-2018':[]}}  
    
            elif gen_dic['star_name']=='MASCARA1':
                plot_settings[key_plot]['idx_noplot']={'ESPRESSO':{'20190714':[0,1,2,69,70],'20190811':[0,1,68,69,70]}}
    
            #RM survey
            elif gen_dic['star_name']=='HAT_P3':plot_settings[key_plot]['idx_noplot']={'HARPN':{'20200130':[0,8]}}
            elif gen_dic['star_name']=='Kepler25':plot_settings[key_plot]['idx_noplot']={'HARPN':{'20200130':[0,19]}}
            elif gen_dic['star_name']=='HAT_P33':plot_settings[key_plot]['idx_noplot']={'HARPN':{'20191204':[0,33,34]}}
            elif gen_dic['star_name']=='HD89345':plot_settings[key_plot]['idx_noplot']={'HARPN':{'20200202':[0,1,93,94,95]}}
            elif gen_dic['star_name']=='HAT_P49':plot_settings[key_plot]['idx_noplot']={'HARPN':{'20200730':[0,1,2,71,72,73]}}
            elif gen_dic['star_name']=='WASP107':plot_settings[key_plot]['idx_noplot']={'CARMENES_VIS':{'20180224':[0,9]},'HARPS':{'20140406':[0,11],'20180201':[0,12],'20180313':[0,12]}}
            elif gen_dic['star_name']=='WASP166':plot_settings[key_plot]['idx_noplot']={'HARPS':{'20170114':[0,39],'20170304':[0,37,38],'20170315':[0,33]}}
            elif gen_dic['star_name']=='HAT_P11':plot_settings[key_plot]['idx_noplot']={'HARPN':{'20150913':[0,1,26],'20151101':[0,1,25,26]},'CARMENES_VIS':{'20170807':[0,17,18],'20170812':[0,1,18]}}                 
            elif gen_dic['star_name']=='WASP156':plot_settings[key_plot]['idx_noplot']={'CARMENES_VIS':{'20190928':[0,1,7],'20191025':[0,6]}}        
            elif gen_dic['star_name']=='HD106315':plot_settings[key_plot]['idx_noplot']={'HARPS':{'20170309':[0,1,39],'20170330':[0,1,24],'20180323':[0,1,23]}}

            #Plot observational data
            if gen_dic['star_name'] in ['Altair','TOI-3362','Nu2Lupi','K2-139','TIC257527578']:
                plot_settings[key_plot]['plot_data']=False

            #Print and plot mean value and dispersion
            #    - relative to all plotted points ('all') or restricted to those with detected stellar line ('det')
            plot_settings[key_plot]['print_disp']=True   #&   False
            plot_settings[key_plot]['disp_mod']='all'  
            plot_settings[key_plot]['plot_disp']=True &  False

            #Plot HDI subintervals, if available
            plot_settings[key_plot]['plot_HDI']=True   & False    
            if gen_dic['star_name'] in ['HD3167','TOI858','GJ436']: 
                plot_settings[key_plot]['nwalkers'] = 100
                plot_settings[key_plot]['nsteps'] = 2000
            #RM survey
            elif gen_dic['star_name'] in ['HAT_P3','Kepler_25','Kepler_68','HAT_P33','K2_105','HD89345','WASP107','WASP166','HAT_P11','WASP156','HIP41378']:
                plot_settings[key_plot]['nwalkers'] = 50
                plot_settings[key_plot]['nsteps'] = 1000
            elif gen_dic['star_name'] in ['HAT_P49']:
                plot_settings[key_plot]['nwalkers'] = 100
                plot_settings[key_plot]['nsteps'] = 1500

            #Plot errorbars / abscissa windows (if available)
            plot_settings[key_plot]['plot_xerr']=True # &  False

            #Print min/max values (to adjust plot ranges)
            plot_settings[key_plot]['plot_bounds']=True & False

            #Transparency of symbols (0 = void)
            plot_settings[key_plot]['alpha_symb']=1. #0.6
        
            #Transparency of error bars (0 = void)
            plot_settings[key_plot]['alpha_err']=0.5  #0.2
            # plot_settings[key_plot]['alpha_err']=1.

    
            #Colors
            #    - set to rainbow to define colors as a function of orbital phase 
            if gen_dic['studied_pl']=='55Cnc_e':
            #    #HARPS seule
            #    plot_settings[key_plot]['color_dic']={'all_HARPSS':'black',
            #               '2012-01-27':'purple',
            #               '2012-02-27':'dodgerblue',
            #               '2012-02-13':'limegreen',
            #               '2012-03-15':'red',      
            #               '2012-01-27_binned':'purple',
            #               '2012-02-27_binned':'dodgerblue',
            #               '2012-02-13_binned':'limegreen',
            #               '2012-03-15_binned':'red'}
            
            #        #Toutes nuits
            #        plot_settings[key_plot]['color_dic']={
            #             '2012-02-27':'black',
            #             '2012-01-27':'grey',
            #             '2012-03-15':'magenta',
            #             '2012-02-13':'blue',
            #             
            #             '2014-03-29':'deepskyblue',
            #             '2014-01-01':'cyan',
            #             '2012-12-25':'green',
            #             '2013-11-28':'lime',
            #             '2013-11-14':'gold',
            #             '2014-02-26':'darkorange',
            #             '2014-01-26':'red'} 
            
                    #Nuits binned
                plot_settings[key_plot]['color_dic']={
                        'all_HARPSS':'dodgerblue',            
                        'all_HARPS_adj':'dodgerblue',
                        'all_HARPS_adj2':'cyan',            
                        'best_HARPSN_adj':'red',
                        'good_HARPSN_adj':'orange',
                        'HARPS_HARPSN_binHARPS':'black',
                        'HARPS_HARPSN_binHARPSN':'black',
                        'best_HARPSN_adj_short':'lime',
                        'best_HARPSN_adj_long':'red',            
                        }            
                plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2020-02-05':'dodgerblue'}}
            elif gen_dic['star_name']=='GJ436':
                plot_settings[key_plot]['color_dic']=['dodgerblue','orange','red']  #trois nuits GJ436b
                plot_settings[key_plot]['color_dic']=['limegreen','red']  #nuits binnees HARPSN + nuit HARPS GJ436b   
                plot_settings[key_plot]['color_dic']={'2019-02-28':'dodgerblue','2019-04-29':'red'} 
                plot_settings[key_plot]['color_dic']={'ESPRESSO':{'20190228':'dodgerblue','20190429':'red'}} 
                plot_settings[key_plot]['color_dic']={'ESPRESSO':{'20190228':'dodgerblue','20190429':'red'},'HARPN':{'20160318':'orange','20160411':'limegreen'},'HARPS':{'20070509':'magenta'}} 
                
                
            elif gen_dic['star_name']=='HD3167':    
                plot_settings[key_plot]['color_dic']={'HARPN':{'2016-10-01':'orange'},'ESPRESSO':{'2019-10-09':'limegreen'}} 
                if gen_dic['studied_pl']==['HD3167_b']:  
                    plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-10-09':'rainbow'}}  
                if gen_dic['studied_pl']==['HD3167_c']:
                    plot_settings[key_plot]['color_dic']={'HARPN':{'2016-10-01':'dodgerblue'}}   
                    plot_settings[key_plot]['color_dic']={'HARPN':{'2016-10-01':'rainbow'}}   
            elif gen_dic['studied_pl']=='WASP121b':         
                plot_settings[key_plot]['color_dic']={'09-01-18':'green','14-01-18':'dodgerblue','31-12-17':'red','HARPS-binned':'black','HARPS-binned-2018':'black'} 
                plot_settings[key_plot]['color_dic']={'HARPS':{'09-01-18':'green','14-01-18':'dodgerblue','31-12-17':'red'},
                           'binned':{'HARPS-binned':'black'}}
            elif gen_dic['studied_pl']=='Kelt9b': 
                plot_settings[key_plot]['color_dic']={'31-07-2017':'dodgerblue'}  
                plot_settings[key_plot]['color_dic']={'HARPN':{'31-07-2017':'dodgerblue','20-07-2018':'red'},
                           'binned':{'HARPS-binned':'black'}}            
            elif gen_dic['studied_pl']=='WASP76b':            
                plot_settings[key_plot]['color_dic']={'2018-10-31':'dodgerblue','2018-09-03':'red','ESP_binned':'black'}
                plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-09-03':'dodgerblue','2018-10-31':'red'},
                           'binned':{'ESP_binned':'black'}}              
            elif gen_dic['studied_pl']=='WASP127b':plot_settings[key_plot]['color_dic']={'HARPS':{'2017-03-20':'dodgerblue','2018-03-31':'green','2018-02-13':'orange','2017-02-28':'red'}} 
            elif gen_dic['studied_pl']=='HD209458b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-07-20':'dodgerblue','2019-09-11':'red'}}
            elif gen_dic['studied_pl']=='GJ9827d':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-08-25':'dodgerblue'}} 
            elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2020-03-18':'dodgerblue'}}
            elif gen_dic['studied_pl']==['TOI858b']:plot_settings[key_plot]['color_dic']={'CORALIE':{'20191205':'dodgerblue','20210118':'red'}}
            elif gen_dic['star_name']=='MASCARA1':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'20190714':'dodgerblue','20190811':'red'}}
            
            #RM survey
            elif gen_dic['star_name']=='HAT_P3':
                plot_settings[key_plot]['color_dic']={'HARPN':{'20200130':'rainbow'}}          
                plot_settings[key_plot]['color_dic']={'HARPN':{'20200130':'dodgerblue'}}  
            elif gen_dic['star_name']=='WASP107':plot_settings[key_plot]['color_dic']={'HARPS':{'20140406':'dodgerblue','20180201':'red','20180313':'limegreen'},'CARMENES_VIS':{'20180224':'orange'}}          
            elif gen_dic['star_name']=='WASP166':plot_settings[key_plot]['color_dic']={'HARPS':{'20170114':'dodgerblue','20170304':'red','20170315':'limegreen'}}          
            elif gen_dic['star_name']=='HAT_P11':plot_settings[key_plot]['color_dic']={'CARMENES_VIS':{'20170807':'dodgerblue','20170812':'limegreen'},'HARPN':{'20150913':'orange','20151101':'purple'}}          
            elif gen_dic['star_name']=='WASP156':plot_settings[key_plot]['color_dic']={'CARMENES_VIS':{'20190928':'dodgerblue','20191025':'red'}}          
            elif gen_dic['star_name']=='HD106315':plot_settings[key_plot]['color_dic']={'HARPS':{'20170309':'dodgerblue','20170330':'limegreen','20180323':'red'}}          
            
            #Plot values for detected CCFs only
            if gen_dic['studied_pl']=='Kelt9b': plot_settings[key_plot]['plot_det']=False     
            elif gen_dic['studied_pl']=='WASP76b': plot_settings[key_plot]['plot_det']=True # & False  

            #Bin values 
            #    - define boundaries of the bins within which the exposures will be binned
            plot_settings[key_plot]['bin_val'] = {}
            if 1==0:
                if plot_settings[key_plot]['prop_absc']=='phase':
                    dbin,bin_min,bin_max=0.01 ,-0.15  ,0.18
                    if gen_dic['star_name']=='HAT_P11': 
                        dbin,bin_min,bin_max=0.0015 ,-0.01  ,0.01
                elif plot_settings[key_plot]['prop_absc']=='r_proj':
                    dbin,bin_min,bin_max=0.05 ,0.15  ,0.98 
                    if gen_dic['star_name']=='WASP107': 
                        dbin=None
                        x_bd_low= np.array([0.15,0.22,0.30,0.39,0.50,0.65,0.80])
                        x_bd_high=np.array([0.22,0.30,0.39,0.50,0.65,0.80,0.98])

            #Abscissa boundaries
            if gen_dic['studied_pl']=='WASP_8b':
                plot_settings[key_plot]['x_range']=[-0.015,0.02]   #phase 
                plot_settings[key_plot]['x_range']=[-0.015,0.015]   #phase 
            if gen_dic['studied_pl']=='55Cnc_e':plot_settings[key_plot]['x_range']=[-0.05,0.05]   #phase      
            #plot_settings[key_plot]['x_range']=[0.95,0.4]          #mu
            #plot_settings[key_plot]['x_range']=[-10.,80.]       #st_lat
            #plot_settings[key_plot]['x_range']=[-80.,30.]       #st_lon
            #plot_settings[key_plot]['x_range']=[-1.,0.4]       #st_x
            #plot_settings[key_plot]['x_range']=[-0.1,1.]       #st_y
            #plot_settings[key_plot]['x_range']=[0.4,1.3]          #contraste
            #plot_settings[key_plot]['x_range']=[3.,12.]          #FWHM
            if gen_dic['star_name']=='HD3167': 
                if plot_settings[key_plot]['prop_absc'] in ['abs_y_st','y_st2']:
                    plot_settings[key_plot]['x_range']=[0.,1.]    
                    plot_settings[key_plot]['x_range']=[-0.02,0.75]    #paper
                if plot_settings[key_plot]['prop_absc']=='abs_lat':plot_settings[key_plot]['x_range']=[0.,180.]  
                if plot_settings[key_plot]['prop_absc']=='mu':plot_settings[key_plot]['x_range']=[-0.02,0.92]  
                    
                if gen_dic['studied_pl']==['HD3167_c']:
                    if plot_settings[key_plot]['prop_absc']=='phase':plot_settings[key_plot]['x_range']=[-0.004,0.004]     
                    if plot_settings[key_plot]['prop_absc']=='mu':plot_settings[key_plot]['x_range']=[0.05,0.9]     
                    if plot_settings[key_plot]['prop_absc']=='y_st':plot_settings[key_plot]['x_range']=[-1.,1.]
                    if plot_settings[key_plot]['prop_absc']=='lat':plot_settings[key_plot]['x_range']=[-180.,180.]  
                if gen_dic['studied_pl']==['HD3167_b']:
                    if plot_settings[key_plot]['prop_absc']=='phase':plot_settings[key_plot]['x_range']=[-0.04,0.04]    
                    if plot_settings[key_plot]['prop_absc']=='mu':plot_settings[key_plot]['x_range']=[-0.02,0.92]              
                    if plot_settings[key_plot]['prop_absc']=='xp_abs':plot_settings[key_plot]['x_range']=[0.,0.9]  
                    if plot_settings[key_plot]['prop_absc']=='r_proj':plot_settings[key_plot]['x_range']=[0.2,1.]      
                    if plot_settings[key_plot]['prop_absc']=='y_st':plot_settings[key_plot]['x_range']=[-1.,1.]
                    if plot_settings[key_plot]['prop_absc']=='lat':plot_settings[key_plot]['x_range']=[-180.,180.]
                
    
            elif gen_dic['studied_pl']=='WASP121b':plot_settings[key_plot]['x_range']=[-0.05,0.05]     #phase 
            elif gen_dic['studied_pl']=='Kelt9b':plot_settings[key_plot]['x_range']=[-0.065,0.065]     #phase     
            elif gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['x_range']=[-0.05,0.05]     #phase     
            elif gen_dic['studied_pl']==['HD209458b']:plot_settings[key_plot]['x_range']=[-0.02,0.02]     #phase  
            elif gen_dic['studied_pl']=='GJ9827d':plot_settings[key_plot]['x_range']=[-0.005,0.005]     #phase  
            elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['x_range']=[-0.0028,0.0028]     #phase  
            elif gen_dic['studied_pl']==['TIC61024636b']:plot_settings[key_plot]['x_range']=[-0.0023,0.0023]     #phase  
            elif gen_dic['studied_pl']==['TOI858b']:plot_settings[key_plot]['x_range']=[-0.025,0.025]     #phase         
            if gen_dic['star_name']=='GJ436':
                if plot_settings[key_plot]['prop_absc']=='phase':plot_settings[key_plot]['x_range']=[-0.0085,0.0085]              
                if plot_settings[key_plot]['prop_absc']=='mu':plot_settings[key_plot]['x_range']=[0.2,0.505]     
            elif gen_dic['star_name']=='Altair':plot_settings[key_plot]['x_range']=[-0.04,0.04]               
            elif gen_dic['star_name']=='TOI-3362':plot_settings[key_plot]['x_range']=[-0.0035,0.0035]  
            elif gen_dic['star_name']=='Nu2Lupi':plot_settings[key_plot]['x_range']=[-0.0018,0.0018]  
            elif gen_dic['star_name']=='K2-139':plot_settings[key_plot]['x_range']=[-0.0036,0.0036]                 
            elif gen_dic['star_name']=='TIC257527578':plot_settings[key_plot]['x_range']=[-0.0019,0.0019]          
            elif gen_dic['star_name']=='HIP41378':plot_settings[key_plot]['x_range']=[0.0002,0.0011]   
            elif gen_dic['star_name']=='MASCARA1':
                    if plot_settings[key_plot]['prop_absc']=='phase':plot_settings[key_plot]['x_range']=[-0.045,0.045] 
                    if plot_settings[key_plot]['prop_absc']=='mu':plot_settings[key_plot]['x_range']=[0.,1.]                
                    if plot_settings[key_plot]['prop_absc']=='y_st':plot_settings[key_plot]['x_range']=[-1.,1.]                    
                    if plot_settings[key_plot]['prop_absc']=='y_st2':plot_settings[key_plot]['x_range']=[0.,1.]                    
                    if plot_settings[key_plot]['prop_absc']=='lat':plot_settings[key_plot]['x_range']=[-180.,180.]  
    
            #RM survey
            elif gen_dic['star_name']=='HAT_P3':
                if plot_settings[key_plot]['prop_absc']=='phase':plot_settings[key_plot]['x_range']=[-0.016,0.016]   
                elif plot_settings[key_plot]['prop_absc']=='mu':plot_settings[key_plot]['x_range']=[0.2,0.8]   
            elif gen_dic['star_name']=='HAT_P33':
                if plot_settings[key_plot]['prop_absc']=='phase':plot_settings[key_plot]['x_range']=[-0.029,0.029]  
                elif plot_settings[key_plot]['prop_absc']=='mu':plot_settings[key_plot]['x_range']=[0.3,1.]                     
            elif gen_dic['star_name']=='HAT_P49':
                if plot_settings[key_plot]['prop_absc']=='phase':plot_settings[key_plot]['x_range']=[-0.035,0.035]         
            elif gen_dic['star_name']=='WASP107':
                if plot_settings[key_plot]['prop_absc']=='phase':plot_settings[key_plot]['x_range']=[-0.011,0.011]  
                elif plot_settings[key_plot]['prop_absc']=='mu':plot_settings[key_plot]['x_range']=[0.2,1.]
                elif plot_settings[key_plot]['prop_absc']=='r_proj':plot_settings[key_plot]['x_range']=[0.15,0.99]
            elif gen_dic['star_name']=='WASP166':
                if plot_settings[key_plot]['prop_absc']=='phase':plot_settings[key_plot]['x_range']=[-0.0145,0.0145]   
                elif plot_settings[key_plot]['prop_absc']=='mu':plot_settings[key_plot]['x_range']=[0.,1.]       
            elif gen_dic['star_name']=='HAT_P11':
                if plot_settings[key_plot]['prop_absc']=='phase':plot_settings[key_plot]['x_range']=[-0.012,0.012]  
                elif plot_settings[key_plot]['prop_absc']=='mu':plot_settings[key_plot]['x_range']=[0.2,1.]           
            elif gen_dic['star_name']=='WASP156':
                if plot_settings[key_plot]['prop_absc']=='phase':plot_settings[key_plot]['x_range']=[-0.013,0.013]  
                elif plot_settings[key_plot]['prop_absc']=='mu':plot_settings[key_plot]['x_range']=[0.,1.]   
            elif gen_dic['star_name']=='HD106315':
                if plot_settings[key_plot]['prop_absc']=='phase':plot_settings[key_plot]['x_range']=[-0.0045,0.0045] 

    
            #General path to the best-fit model to property series
            if gen_dic['star_name']=='MASCARA1':
                plot_settings[key_plot]['IntrProp_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/MASCARA1b_Saved_data/Joined_fits/Intr_prop/Orig/chi2/'
            #RM survey
            elif gen_dic['star_name']=='HAT_P3':
                plot_settings[key_plot]['IntrProp_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/HAT_P3b_Saved_data/Joined_fits/Intr_prop/Orig/chi2/'
                plot_settings[key_plot]['IntrProp_path'] = None
            elif gen_dic['star_name']=='HAT_P33':
                plot_settings[key_plot]['IntrProp_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/HAT_P33b_Saved_data/Joined_fits/Intr_prop/Orig/chi2/'
                # plot_settings[key_plot]['IntrProp_path'] = None
            elif gen_dic['star_name']=='WASP107':plot_settings[key_plot]['IntrProp_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/WASP107b_Saved_data/Joined_fits/Intr_prop/Orig/chi2/'
            elif gen_dic['star_name']=='WASP166':plot_settings[key_plot]['IntrProp_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/WASP166b_Saved_data/Joined_fits/Intr_prop/Orig/chi2/'
            # elif gen_dic['star_name']=='HAT_P11':plot_settings[key_plot]['IntrProp_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/HAT_P11b_Saved_data/Joined_fits/Intr_prop/Orig/chi2/'
            elif gen_dic['star_name']=='WASP156':plot_settings[key_plot]['IntrProp_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/WASP156b_Saved_data/Joined_fits/Intr_prop/Orig/chi2/'
    
                    
            #General path to the best-fit model to profile series
            if gen_dic['studied_pl']==['TOI858b']:
                plot_settings[key_plot]['IntrProf_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/TOI858b_Saved_data/Joined_fits/Intr_prof/mcmc/'
                plot_settings[key_plot]['IntrProf_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/TOI858b_Saved_data/Joined_fits/Intr_prof/mcmc/Visits12_indiv_osamp5/'
            # if gen_dic['studied_pl']==['GJ436_b']:
            #     plot_settings[key_plot]['IntrProf_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/GJ436_b_Saved_data/Joined_fits/Intr_prof/mcmc/ESPRESSO/DG_CLfromDI_proppervis_osamp5/'
            #RM survey
            elif gen_dic['star_name']=='HAT_P3':
                plot_settings[key_plot]['IntrProf_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/HAT_P3b_Saved_data/Joined_fits/Intr_prof/mcmc/Oversamp5_n51/'
            # elif gen_dic['star_name']=='HAT_P33':
            #     plot_settings[key_plot]['IntrProf_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/HAT_P33b_Saved_data/Joined_fits/Intr_prof/mcmc/Oversamp5_n51/'
            elif gen_dic['star_name']=='WASP107':                                   
                plot_settings[key_plot]['IntrProp_path'] = None
                plot_settings[key_plot]['IntrProf_path']='/Users/bourrier/Travaux/ANTARESS/En_cours/WASP107b_Saved_data/Joined_fits/Intr_prof/mcmc/Oversamp3_n31_voigt_scaled_FINAL/'

            #-----------------------------------------------------
            #RV plot
            if (plot_prop=='rv' ):
                
                #Plot null reference
                plot_settings[key_plot]['plot_ref']=True
    
                #Plot data-equivalent model from property fit 
                plot_settings[key_plot]['theo_obs_prop'] = True & False
    
                #Plot data-equivalent model from profile fit 
                plot_settings[key_plot]['theo_obs_prof'] = False
    
                #Plot high-resolution model from property fit
                plot_settings[key_plot]['theo_HR_prop'] = True    & False
    
                #Plot high-resolution model from profile fit
                plot_settings[key_plot]['theo_HR_prof'] = True    # &  False
                
                #Plot high-resolution model from nominal values in ANTARESS_systems.py
                plot_settings[key_plot]['theo_HR_nom'] = True  # &   False            
                
                #Overplot the different contributions
                plot_settings[key_plot]['contrib_theo_HR_nom']=True       &   False
    
                #Calculate model envelope from MCMC results (calculation of models at +-1 sigma range of the parameters)
                plot_settings[key_plot]['calc_envMCMC_theo_HR_nom']=False
            
                #Calculate model sample from MCMC results (distribution of models following the PDF of the parameters)
                plot_settings[key_plot]['calc_sampMCMC_theo_HR_nom']=False   
                
                #Predict local RVs measurements from nominal model with errors, and SNR of RMR signal
                #    - leave empty, or define:
                # + a phase table for the mock exposures over which the high-res RV model will be binned
                # + the factor 'C' that should remain always the same (scaled from previous measurements)
                # + the flux gain 'C_inst' between the considered instrument and the one used to define C
                # + the FWHM and contrast of the local CCFs
                # + rand: set to True to draw measurements from a gaussian with mean the theoretical RV value and sigma its estimated error
                #    - for example, one can use RV errors obtained for a given star with a given instrument to estimate C

                # plot_settings[key_plot]['predic']={
                #     'C_inst':{
                #         'ESPRESSO':1.,      
                #         'HARPS':1./6.,    
                #         'NIRPS':1./5.,     #from C. Lovis, efficiency roughly similar to ESPRESSO, thus flux ratio scales as mirror size ratio 
                #         },
                #     'C':1.2e-5,     #determined roughly for ESPRESSO, HD209458b, FWHM=8.3, C = 0.6, texp=175 s
                #     # 'FWHM':6.1,'ctrst':0.45,     #valeurs pour une K0, pris de Cegla+2016 pour HD189
                #     # 'FWHM':10.,'ctrst':0.4,       #valeurs pour une F, HAT-P-41, pris de l'analyse de Omar en RRM
                #     # 'FWHM':8.5,'ctrst':0.6,       #valeurs pour une G0V, HD209 de mon analyse ESPRESSO
                #     'FWHM':6.,'ctrst':0.7,       #valeurs pour une G4V, Nu2_Lupi, de mon analyse ESPRESSO
                #     'rand':True
    
                #     'C_RMR':155000.,     #determined roughly for ESPRESSO, HD3167
                #     }
    
    
                #Print system properties derived from common fit to all exposures (if relevant)
                # plot_settings[key_plot]['plot_fit_comm']={'binned':['ESP_binned']}
    
                #Boundaries
                if gen_dic['studied_pl']=='WASP_8b':
                    plot_settings[key_plot]['y_range']=[-3.,1.5]
                elif gen_dic['studied_pl']=='55Cnc_e':
                    plot_settings[key_plot]['y_range']=[-5.,5.]
                elif gen_dic['studied_pl']==['HD3167_b']:
                    plot_settings[key_plot]['y_range']=[-5.,5.]  #MCMC
                elif gen_dic['studied_pl']==['HD3167_c']:
                    plot_settings[key_plot]['y_range']=[-7.,3.]
                    plot_settings[key_plot]['y_range']=[-3.5,2.]    #MCMC
                elif gen_dic['studied_pl']=='WASP121b':
                    plot_settings[key_plot]['y_range']=[-1.,4.]
        #            plot_settings[key_plot]['y_range']=[-3.,6.5]
                elif gen_dic['studied_pl']=='Kelt9b':
                    plot_settings[key_plot]['y_range']=[-30.,5.]     
                elif gen_dic['studied_pl']=='WASP76b':
                    plot_settings[key_plot]['y_range']=[-1.5,1.5]  
                elif gen_dic['studied_pl']==['HD209458b']:plot_settings[key_plot]['y_range']=[-4.5,4.5] 
                elif gen_dic['studied_pl']=='GJ9827d':plot_settings[key_plot]['y_range']=[-2.5,2.5] 
                elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['y_range']=[-2.5,2.5] 
                elif gen_dic['studied_pl']==['TIC61024636b']:plot_settings[key_plot]['y_range']=[-0.7,0.7] 
                elif gen_dic['star_name']=='Altair':plot_settings[key_plot]['y_range']=[-200.,200.] 
                elif gen_dic['studied_pl']==['TOI858b']:plot_settings[key_plot]['y_range']=[-2.,8.] 
                elif gen_dic['studied_pl']==['GJ436_b']:
                    plot_settings[key_plot]['y_range']=[-1.3,1.3]  
                    plot_settings[key_plot]['y_range']=[-0.9,0.9]  #ESP
                    plot_settings[key_plot]['y_range']=[-1.,1.]  #all
                elif gen_dic['star_name']=='TOI-3362':
                    plot_settings[key_plot]['y_range']=[-4.,4.]  
                    plot_settings[key_plot]['y_range']=[-21.,21.]  
                elif gen_dic['star_name']=='Nu2Lupi':
                    plot_settings[key_plot]['y_range']=[-2.3,2.3]  
                elif gen_dic['star_name']=='K2-139':
                    plot_settings[key_plot]['y_range']=[-2.9,2.9]  
                elif gen_dic['star_name']=='TIC257527578':
                    # plot_settings[key_plot]['y_range']=[-6.,6.]    #lambda 0
                    plot_settings[key_plot]['y_range']=[4.,8.5]    #lambda 90
                elif gen_dic['star_name']=='MASCARA1':
                    plot_settings[key_plot]['y_range']=[-40.,50.]  
                elif gen_dic['star_name']=='HIP41378':plot_settings[key_plot]['y_range']=[-10.,10.]  
                elif gen_dic['star_name']=='HAT_P3':
                    plot_settings[key_plot]['y_range']=[-1.3,1.3]   
                elif gen_dic['star_name']=='HAT_P33':
                    plot_settings[key_plot]['y_range']=[-30.,30.] 
                elif gen_dic['star_name']=='WASP107':
                    plot_settings[key_plot]['y_range']=[-2.5,2.5] 
                elif gen_dic['star_name']=='HAT_P49':
                    plot_settings[key_plot]['y_range']=[-10.,5.]  
    
    

            #-----------------------------------------------------
            #RV residual plot (m/s)
            if (plot_prop=='RVres' ):

                #Plot null reference
                plot_settings[key_plot]['plot_ref']=True
            
                #Bornes du plot
                plot_settings[key_plot]['y_range']=[-1.495,1.495]
                plot_settings[key_plot]['y_range']=[-100.,100.]
                plot_settings[key_plot]['y_range']=[-500.,500.]         
                if gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['y_range']=[-500.,500.]  
                elif gen_dic['studied_pl']=='GJ9827d':plot_settings[key_plot]['y_range']=[-2500.,2500.]  
                elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['y_range']=[-2500.,2500.] 
    
            #-----------------------------------------------------
            #FWHM plot
            if (plot_prop=='FWHM' ):
    
                #Plot data-equivalent model from property fit 
                plot_settings[key_plot]['theo_obs_prop'] = True & False
    
                #Plot data-equivalent model from profile fit 
                plot_settings[key_plot]['theo_obs_prof'] = False
    
                #Plot high-resolution model from property fit
                plot_settings[key_plot]['theo_HR_prop'] = True  & False
    
                #Plot high-resolution model from profile fit
                plot_settings[key_plot]['theo_HR_prof'] = True   #&  False
    
                #Bornes du plot
                plot_settings[key_plot]['y_range']=None
                if gen_dic['studied_pl']=='WASP_8b':
                    plot_settings[key_plot]['y_range']=[4.,12.]
                elif gen_dic['studied_pl']=='55Cnc_e':
                    plot_settings[key_plot]['y_range']=[0.,15.]
                elif gen_dic['studied_pl']==['HD3167_b']:
                    plot_settings[key_plot]['y_range']=[0.,20.]
                elif gen_dic['studied_pl']==['HD3167_c']:
                    plot_settings[key_plot]['y_range']=[2.,16.]
                    plot_settings[key_plot]['y_range']=[1.5,16.]  #mask K5
                    plot_settings[key_plot]['y_range']=[0.,13.]  #MCMC
                elif gen_dic['studied_pl']=='WASP121b':
                    plot_settings[key_plot]['y_range']=[4.,15.]
                    plot_settings[key_plot]['y_range']=[5.,20.]    #mask F
                elif gen_dic['studied_pl']=='WASP76b':
                    plot_settings[key_plot]['y_range']=[6.,15.]      
                elif gen_dic['studied_pl']=='HD209458b':plot_settings[key_plot]['y_range']=[7.3,12.2]     
                elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['y_range']=[5.5,8.6]  
                elif gen_dic['studied_pl']=='GJ9827d':plot_settings[key_plot]['y_range']=[0.,15.]
                elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['y_range']=[4.5,9.] 
                elif gen_dic['studied_pl']==['TOI858b']:plot_settings[key_plot]['y_range']=[2.,15.] 
                elif gen_dic['star_name']=='MASCARA1':
                    plot_settings[key_plot]['y_range']=[0.,30.]  
                    
            
         
                    

            #-----------------------------------------------------
            #Contrast plot
            if (plot_prop=='ctrst' ):
    
                #Plot data-equivalent model from property fit 
                plot_settings[key_plot]['theo_obs_prop'] = True  & False
    
                #Plot data-equivalent model from profile fit 
                plot_settings[key_plot]['theo_obs_prof'] = False
    
                #Plot high-resolution model from property fit
                plot_settings[key_plot]['theo_HR_prop'] = True  & False
    
                #Plot high-resolution model from profile fit
                plot_settings[key_plot]['theo_HR_prof'] = True   #&  False

                #Bornes du plot  
                if gen_dic['studied_pl']=='WASP_8b':
                    plot_settings[key_plot]['y_range']=[0.3,0.9]
                elif gen_dic['studied_pl']=='55Cnc_e':
                    plot_settings[key_plot]['y_range']=[0.2,1.5]
                if gen_dic['star_name']=='HD3167':
                    plot_settings[key_plot]['y_range']=[-0.8,1.8]                
                    if gen_dic['studied_pl']==['HD3167_b']:
                        plot_settings[key_plot]['y_range']=[-1.5,2.]
                    if gen_dic['studied_pl']==['HD3167_c']:
                        plot_settings[key_plot]['y_range']=[0.3,1.8]
                        plot_settings[key_plot]['y_range']=[0.3,1.2]   #detectees
                        plot_settings[key_plot]['y_range']=[0.1,1.]    #detectees mask K5
                        plot_settings[key_plot]['y_range']=[0.,1.1]  #MCMC              
    
                elif gen_dic['studied_pl']=='WASP121b':
                    plot_settings[key_plot]['y_range']=[0.25,0.65]
                elif gen_dic['studied_pl']=='WASP76b':
                    plot_settings[key_plot]['y_range']=[0.3,0.55]  
                elif gen_dic['studied_pl']=='HD209458b':plot_settings[key_plot]['y_range']=[0.5,0.78]    
                elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['y_range']=[0.23,0.39]  
                elif gen_dic['studied_pl']=='GJ9827d':plot_settings[key_plot]['y_range']=[0.3,1.7]
                elif gen_dic['studied_pl']=='Nu2Lupi_c':plot_settings[key_plot]['y_range']=[0.45,0.9] 
                elif gen_dic['studied_pl']==['TOI858b']:plot_settings[key_plot]['y_range']=[0.,1.7] 
                elif gen_dic['star_name']=='MASCARA1':plot_settings[key_plot]['y_range']=[0.,1.]  
                # elif gen_dic['star_name']=='WASP107':plot_settings[key_plot]['y_range']=[0.28,0.72] 

            #-----------------------------------------------------
            #Plot of lobe-core properties ratios
            if (plot_prop=='dgauss' in data_dic['DI']['model'].values()):
        
                #Ratio of lobe FWHM to core FWHM
                if (plot_prop=='FWHM_l2c' ):

                    #Bornes du plot
                    if gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['y_range']=[0.,2.]
        
                #-----------------------------------------------------
                #Ratio of lobe contrast to core amplitude
                if (plot_prop=='amp_l2c' ):

                    #Bornes du plot  
                    if gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['y_range']=[-1.,3.]
        
                #-----------------------------------------------------
                #RV shift bewtween lobe and core gaussian RV centroid
                if (plot_prop=='rv_l2c' ):

                    #Bornes du plot  
                    if gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['y_range']=[-2.,2.]

                #-----------------------------------------------------
                #True FWHM
                if (plot_prop=='true_FWHM' ):

                    #Bornes du plot            
                    if gen_dic['studied_pl']==['GJ436_b']:
                        plot_settings[key_plot]['y_range']=[2.5,6.5] 
                        # plot_settings[key_plot]['y_range']=[0.,7.] 
        
                #-----------------------------------------------------
                #True contrast
                if (plot_prop=='true_ctrst' ):

                    #Bornes du plot    
                    if gen_dic['studied_pl']==['GJ436_b']:
                        plot_settings[key_plot]['y_range']=[0.15,0.45]
                        # plot_settings[key_plot]['y_range']=[0.,1.]



















    '''
    Plot occulted stellar regions, for a given system configuration, with RV of the planet-occulted regions
    '''
    if (plot_dic['occulted_regions']!=''):
        key_plot = 'occulted_regions'
        plot_settings[key_plot]={} 

        #Visits to plot
        if gen_dic['studied_pl']==['HD3167_b']:plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-10-09']} 

        #Planet
        plot_settings[key_plot]['pl_ref'] = {'inst':{'vis':'HD3167_c'}}
        plot_settings[key_plot]['pl_ref'] = {'inst':{'vis':'GJ436_b'}}
        plot_settings[key_plot]['pl_ref'] = {'inst':{'vis':'HD15337c'}}
















    '''
    Plotting planetary system architecture
        - orbits of all planets in the system can be plotted in 3D
        - different views can be chosen
        - the stellar surface can show limb-darkening or limb-darkening-weighted radial velocity field
    '''
    if (plot_dic['system_view']!=''):
        key_plot = 'system_view'
        plot_settings[key_plot]={} 

        #Planets to plot
        if gen_dic['star_name']=='HD3167':
            plot_settings[key_plot]['pl_to_plot']=['HD3167_b','HD3167_c','HD3167_d']
            plot_settings[key_plot]['pl_to_plot']=['HD3167_b','HD3167_c']            
        if gen_dic['star_name']=='TOI178':
            plot_settings[key_plot]['pl_to_plot']=['TOI178b','TOI178c','TOI178d','TOI178e','TOI178f','TOI178g']
        if gen_dic['star_name']=='TOI858':
            plot_settings[key_plot]['pl_to_plot']=['TOI858b'] 
        if gen_dic['star_name']=='HD15337':
            plot_settings[key_plot]['pl_to_plot']=['HD15337c'] 
        if gen_dic['star_name']=='HD209458':
            plot_settings[key_plot]['pl_to_plot']=['HD209458b'] 
            # plot_settings[key_plot]['pl_to_plot']=['HD209458b','HD209458c'] 
        if gen_dic['star_name']=='Altair':
            plot_settings[key_plot]['pl_to_plot']=['Altair_b'] 
        if gen_dic['star_name']=='GJ436':
            plot_settings[key_plot]['pl_to_plot']=['GJ436_b'] 
        if gen_dic['star_name']=='MASCARA1':plot_settings[key_plot]['pl_to_plot']=['MASCARA1b'] 
        elif gen_dic['star_name']=='HAT_P3':plot_settings[key_plot]['pl_to_plot']=['HAT_P3b'] 
        elif gen_dic['star_name']=='HAT_P11':plot_settings[key_plot]['pl_to_plot']=['HAT_P11b'] 
        elif gen_dic['star_name']=='HAT_P33':plot_settings[key_plot]['pl_to_plot']=['HAT_P33b'] 
        elif gen_dic['star_name']=='HAT_P49':plot_settings[key_plot]['pl_to_plot']=['HAT_P49b'] 
        elif gen_dic['star_name']=='HD89345':plot_settings[key_plot]['pl_to_plot']=['HD89345b'] 
        elif gen_dic['star_name']=='HD106315':plot_settings[key_plot]['pl_to_plot']=['HD106315c'] 
        elif gen_dic['star_name']=='K2_105':plot_settings[key_plot]['pl_to_plot']=['K2_105b'] 
        elif gen_dic['star_name']=='Kepler25':plot_settings[key_plot]['pl_to_plot']=['Kepler25c']
        elif gen_dic['star_name']=='Kepler63':plot_settings[key_plot]['pl_to_plot']=['Kepler63b']
        elif gen_dic['star_name']=='WASP107':plot_settings[key_plot]['pl_to_plot']=['WASP107b']
        elif gen_dic['star_name']=='WASP156':plot_settings[key_plot]['pl_to_plot']=['WASP156b']
        elif gen_dic['star_name']=='WASP166':plot_settings[key_plot]['pl_to_plot']=['WASP166b']
        elif gen_dic['star_name'] == 'V1298tau' :
            plot_settings[key_plot]['pl_to_plot']=['V1298tau_b']


        #Number of points in the planet orbits
        if gen_dic['star_name']=='GJ436':plot_settings[key_plot]['npts_orbits'] = np.repeat(5000,len(plot_settings[key_plot]['pl_to_plot'])) 
        elif gen_dic['star_name']=='MASCARA1':plot_settings[key_plot]['npts_orbits'] = np.repeat(5000,len(plot_settings[key_plot]['pl_to_plot'])) 
        elif gen_dic['star_name']=='WASP166':plot_settings[key_plot]['npts_orbits'] = np.repeat(20000,len(plot_settings[key_plot]['pl_to_plot'])) 

        #Position of planets along their orbit
        #    - alternatively set an absolute time list in BJD through plot_settings[key_plot]['t_BJD'] 
        if gen_dic['star_name']=='HD3167':
            plot_settings[key_plot]['yorb_pl'] = [-0.495,0.]    #config istar
            # plot_settings[key_plot]['yorb_pl'] = [0.495,0.]    #config pi-istar
        elif gen_dic['star_name']=='TOI858':plot_settings[key_plot]['yorb_pl'] = [0.,-0.4] 
        elif gen_dic['star_name']=='GJ436':plot_settings[key_plot]['yorb_pl'] = [0.] 
        elif gen_dic['star_name']=='MASCARA1':plot_settings[key_plot]['yorb_pl'] = [0.]  
        elif gen_dic['star_name']=='HAT_P3':plot_settings[key_plot]['yorb_pl'] = [-0.55]            
        elif gen_dic['star_name']=='HAT_P11':plot_settings[key_plot]['yorb_pl'] = [-0.1]            
        elif gen_dic['star_name']=='HAT_P33':plot_settings[key_plot]['yorb_pl'] = [-0.1]            
        elif gen_dic['star_name']=='HAT_P49':plot_settings[key_plot]['yorb_pl'] = [0.3]          
        elif gen_dic['star_name']=='HD89345':plot_settings[key_plot]['yorb_pl'] = [-0.3]     
        elif gen_dic['star_name']=='HD106315':plot_settings[key_plot]['yorb_pl'] = [-0.555]   
        elif gen_dic['star_name']=='K2_105':plot_settings[key_plot]['yorb_pl'] = [0.25]  
        elif gen_dic['star_name']=='Kepler25':
            plot_settings[key_plot]['xorp_pl'] = np.array([[-1.,  0.]])
            plot_settings[key_plot]['yorb_pl'] = [-0.88]    
        elif gen_dic['star_name']=='Kepler63':plot_settings[key_plot]['yorb_pl'] = [0.8]  
        elif gen_dic['star_name']=='WASP107':plot_settings[key_plot]['yorb_pl'] = [0.3]  
        elif gen_dic['star_name']=='WASP156':plot_settings[key_plot]['yorb_pl'] = [-0.3] 
        elif gen_dic['star_name']=='WASP166':
            plot_settings[key_plot]['xorp_pl'] = np.array([[-1.,  0.]])
            plot_settings[key_plot]['yorb_pl'] = [-0.394] 
        elif gen_dic['star_name']=='HD209458':   #ANTARESS I, arbitrary plot
            plot_settings[key_plot]['xorp_pl'] = np.array([[0.,  1.]])
            plot_settings[key_plot]['yorb_pl'] = [-0.3]

            # plot_settings[key_plot]['xorp_pl'] = np.array([[0.,  1.],[0.,  1.]])
            # plot_settings[key_plot]['yorb_pl'] = [-0.3,-0.3]
            
        #Absolute time of the plot (BJD - 2400000)
        #    - overwrites 'yorb_pl' if not set to None
        plot_settings[key_plot]['t_BJD'] = None
        # if gen_dic['star_name']=='HD209458':   #ANTARESS I, mock, multi-pl
        #     plot_settings[key_plot]['t_BJD'] = { 'inst':'ESPRESSO','vis':'mock_vis','t':54560.806755574+np.array([-0.5,-0.2,0.,0.2,0.5])/24. }
            
            
            
            
        #Vertical position of oriented arrow along planet orbits  
        if gen_dic['star_name']=='HD3167':
            plot_settings[key_plot]['yorb_dir'] = [-0.45,-1.,1.]        
            plot_settings[key_plot]['yorb_dir'] = [-0.55,-0.9]      #config istar
            # plot_settings[key_plot]['yorb_dir'] = [0.55,0.9]      #config pi-istar            
        if gen_dic['star_name']=='TOI858':plot_settings[key_plot]['yorb_dir'] = [0.,0.4]   
        if gen_dic['star_name']=='GJ436':plot_settings[key_plot]['yorb_dir'] = [0.5]   
        if gen_dic['star_name']=='MASCARA1':plot_settings[key_plot]['yorb_dir'] = [0.5] 
        elif gen_dic['star_name']=='HAT_P3':plot_settings[key_plot]['yorb_dir'] = [-0.7]             
        elif gen_dic['star_name']=='HAT_P11':plot_settings[key_plot]['yorb_dir'] = [0.5]              
        elif gen_dic['star_name']=='HAT_P33':plot_settings[key_plot]['yorb_dir'] = [-0.2]           
        elif gen_dic['star_name']=='HAT_P49':plot_settings[key_plot]['yorb_dir'] = [-0.3]           
        elif gen_dic['star_name']=='HD89345':plot_settings[key_plot]['yorb_dir'] = [0.3]           
        elif gen_dic['star_name']=='HD106315':plot_settings[key_plot]['yorb_dir'] = [-0.59]  
        elif gen_dic['star_name']=='K2_105':plot_settings[key_plot]['yorb_dir'] = [-0.25]   
        elif gen_dic['star_name']=='Kepler25':
            plot_settings[key_plot]['xorb_dir'] = np.array([[0.,  1.]])
            plot_settings[key_plot]['yorb_dir'] = [-0.89]    
        elif gen_dic['star_name']=='Kepler63':plot_settings[key_plot]['yorb_dir'] = [0.25]
        elif gen_dic['star_name']=='WASP107':plot_settings[key_plot]['yorb_dir'] = [-0.05]  
        elif gen_dic['star_name']=='WASP156':plot_settings[key_plot]['yorb_dir'] = [0.4]  
        elif gen_dic['star_name']=='WASP166':
            plot_settings[key_plot]['xorb_dir'] = np.array([[0.,  1.]])
            plot_settings[key_plot]['yorb_dir'] = [-0.404]              
        elif gen_dic['star_name']=='HD209458':   
            plot_settings[key_plot]['xorb_dir'] = np.array([[0.,  1.]])
            plot_settings[key_plot]['yorb_dir'] = [-0.05]

            plot_settings[key_plot]['xorb_dir'] = np.array([[0.,  1.],[0.,  1.]])    #ANTARESS I, multi-pl plot full
            plot_settings[key_plot]['yorb_dir'] = [-0.05,-0.05]
            # plot_settings[key_plot]['xorb_dir'] = np.array([[0.,  0.6],[0.,  0.4]])    #ANTARESS I, multi-pl plot zoom
            # plot_settings[key_plot]['yorb_dir'] = [-0.44,-0.7]
            
            
            
        #Apparent size of the planet
        #    - provide the radius of each planet (relative to the star) that will be used for the plot
        #    - leave empty to use maximum radisu by default
        # elif gen_dic['star_name']=='HD209458':
        #     plot_settings[key_plot]['RpRs_pl'] = [0.12086]
        #     plot_settings[key_plot]['RpRs_pl'] = [0.12086,0.12086*2.]
          
            
            

        #Orbit colors
        if len(plot_settings[key_plot]['pl_to_plot'])==1:
            plot_settings[key_plot]['col_orb'] = ['forestgreen'] 
            plot_settings[key_plot]['col_orb_samp'] = ['limegreen'] 
        elif len(plot_settings[key_plot]['pl_to_plot'])==2:
            plot_settings[key_plot]['col_orb'] = ['forestgreen','peru'] 
            plot_settings[key_plot]['col_orb_samp'] = ['limegreen','orange'] 
        elif len(plot_settings[key_plot]['pl_to_plot'])==3:
            plot_settings[key_plot]['col_orb'] = ['limegreen','orange','purple'] 
        elif len(plot_settings[key_plot]['pl_to_plot'])==6:
            plot_settings[key_plot]['col_orb'] = ['darkred','orange','gold','limegreen','cyan','purple']             

        #Number of orbits drawn randomly
        if gen_dic['star_name']=='HD3167':plot_settings[key_plot]['norb']=4000  #paper HD3167, to get 1000 orbits
        elif gen_dic['star_name']=='TOI858':plot_settings[key_plot]['norb']=2000                  
        elif gen_dic['star_name']=='GJ436':plot_settings[key_plot]['norb']=2500  
        elif gen_dic['star_name']=='HAT_P3':plot_settings[key_plot]['norb']=5000        #to get 500 orbits     
        elif gen_dic['star_name']=='HAT_P11':plot_settings[key_plot]['norb']=2300       #to get 550 orbits  
        elif gen_dic['star_name']=='HAT_P33':plot_settings[key_plot]['norb']=1000       #to get 260 orbits 
        elif gen_dic['star_name']=='HAT_P49':plot_settings[key_plot]['norb']=1000       #to get 300 orbits 
        elif gen_dic['star_name']=='HD89345':plot_settings[key_plot]['norb']=2500       #to get 600 orbits 
        elif gen_dic['star_name']=='HD106315':plot_settings[key_plot]['norb']=2500       #to get 540 orbits 
        elif gen_dic['star_name']=='K2_105':plot_settings[key_plot]['norb']=3000       #to get 950 orbits 
        elif gen_dic['star_name']=='Kepler25':plot_settings[key_plot]['norb']=6000       #to get 260 orbits 
        elif gen_dic['star_name']=='Kepler63':plot_settings[key_plot]['norb']=4000       #to get 600 orbits 
        elif gen_dic['star_name']=='WASP107':plot_settings[key_plot]['norb']=2500       #to get 800 orbits 
        elif gen_dic['star_name']=='WASP156':plot_settings[key_plot]['norb']=1500       #to get 430 orbits 
        elif gen_dic['star_name']=='WASP166':plot_settings[key_plot]['norb']=1500       #to get 430 orbits 




            
        #Ranges of orbital parameters
        #    - will be called if one of the dictionaries contains an entry
        #    - values are used as lower / upper errors around the nominal property
        #    - the impact parameter range will not be used directly, but to remove configurations for (a/Rs,ip) that yield values beyond the range
        #      in this case the range should be in absolute values
        if gen_dic['star_name']=='HD3167':
            plot_settings[key_plot]['aRs_err'] = {'HD3167_b':[0.986,0.464],'HD3167_c':[12.622,5.549]}
            plot_settings[key_plot]['lambdeg_err'] = {'HD3167_b':[7.9,6.6],'HD3167_c':[5.5,5.4]}   #nominal config        
            plot_settings[key_plot]['ip_err'] = {'HD3167_b':[7.7,4.6],'HD3167_c':[0.96,0.5]}          
            plot_settings[key_plot]['b_range_all'] = {'HD3167_b':[0.47-0.32,0.47+0.31],'HD3167_c':[0.50-0.33,0.50+0.31]}
        
            # plot_settings[key_plot]['lambdeg_err'] = {'HD3167_b':[6.6,7.9],'HD3167_c':[5.4,5.5]}   #alternative config
            # plot_settings[key_plot]['ip_err'] = {'HD3167_b':[4.6,7.7],'HD3167_c':[0.5,0.96]}  
            # plot_settings[key_plot]['b_range_all'] = {'HD3167_b':[0.47-0.31,0.47+0.32],'HD3167_c':[0.50-0.31,0.50+0.33]}   

        elif gen_dic['star_name']=='TOI858':
            plot_settings[key_plot]['aRs_err'] = {'TOI858b':[0.15,0.16]}
            plot_settings[key_plot]['lambdeg_err'] = {'TOI858b':[3.65981,3.79777]}    
            plot_settings[key_plot]['ip_err'] = {'TOI858b':[0.44,0.50]}          
            plot_settings[key_plot]['b_range_all'] = {'TOI858b':[0.397-0.055,0.397+0.046]}        
        elif gen_dic['star_name']=='GJ436':
            plot_settings[key_plot]['aRs_err'] = {'GJ436_b':[0.08,0.08]}   
            plot_settings[key_plot]['lambdeg_err'] = {'GJ436_b':[17.3,23.3]}          
            plot_settings[key_plot]['ip_err'] = {'GJ436_b':[0.0306,0.0306]}          
            plot_settings[key_plot]['b_range_all'] = {'GJ436_b':[0.802-0.012,0.802+0.012]} 
        elif gen_dic['star_name']=='HAT_P3':
            plot_settings[key_plot]['aRs_err'] = {'HAT_P3b':[0.2667,0.2667]}   
            plot_settings[key_plot]['lambdeg_err'] = {'HAT_P3b':[13.06753547,17.42564114]}          
            plot_settings[key_plot]['ip_err'] = {'HAT_P3b':[0.19,0.19]}          
            plot_settings[key_plot]['b_range_all'] = {'HAT_P3b':[0.615-0.012 ,0.615+0.012]} 
        elif gen_dic['star_name']=='HAT_P11':
            plot_settings[key_plot]['aRs_err'] = {'HAT_P11b':[0.18,0.18]}   
            plot_settings[key_plot]['lambdeg_err'] = {'HAT_P11b':[8.3,7.1]}          
            plot_settings[key_plot]['ip_err'] = {'HAT_P11b':[0.09,0.15]}          
            plot_settings[key_plot]['b_range_all'] = {'HAT_P11b':[16.5*np.cos(89.05*np.pi/180.)-0.032 ,16.5*np.cos(89.05*np.pi/180.)+0.019]}  #   le b ne matche pas la combinaison de aRs et i de Allart/Huber, ou meme de Huber seul, je prends les memes erreurs mais avec le aRs*cos(i) utilise
        elif gen_dic['star_name']=='HAT_P33':
            plot_settings[key_plot]['aRs_err'] = {'HAT_P33b':[0.59,0.58]}   
            plot_settings[key_plot]['lambdeg_err'] = {'HAT_P33b':[4.1,4.1]}          
            plot_settings[key_plot]['ip_err'] = {'HAT_P33b':[1.3,1.2]}          
            plot_settings[key_plot]['b_range_all'] = {'HAT_P33b':[5.69*np.cos(88.2*np.pi/180.)-0.098 ,5.69*np.cos(88.2*np.pi/180.)+0.1]} # {'HAT_P33b':[0.151-0.098 ,0.151+0.1]}    #aRscosi = 5.69*np.cos(88.2*np.pi/180.) = 0.1787 ne matche pas celui donné
        elif gen_dic['star_name']=='HAT_P49':
            plot_settings[key_plot]['aRs_err'] = {'HAT_P49b':[0.30,0.19]}   
            plot_settings[key_plot]['lambdeg_err'] = {'HAT_P49b':[1.78,1.80]}          
            plot_settings[key_plot]['ip_err'] = {'HAT_P49b':[1.7,1.7]}          
            plot_settings[key_plot]['b_range_all'] = {'HAT_P49b':[0.34-0.141 ,0.34+0.119]}    #aRscosi = 5.13*np.cos(86.2*np.pi/180.) = 0.34 ok
        elif gen_dic['star_name']=='HD89345':
            plot_settings[key_plot]['aRs_err'] = {'HD89345b':[0.027,0.027]}   
            plot_settings[key_plot]['lambdeg_err'] = {'HD89345b':[32.5,33.6]}          
            plot_settings[key_plot]['ip_err'] = {'HD89345b':[0.1,0.1]}          
            plot_settings[key_plot]['b_range_all'] = {'HD89345b':[0.552-0.017 ,0.552+0.017]}    #aRscosi = 13.625*np.cos(87.68*np.pi/180.) = 0.552 different du b de VanEylen+2018, je prends les memes erreurs mais avec le aRs*cos(i) utilise
        elif gen_dic['star_name']=='HD106315':
            plot_settings[key_plot]['aRs_err'] = {'HD106315c':[4.2,5.7]}   
            plot_settings[key_plot]['lambdeg_err'] = {'HD106315c':[2.6,2.7]}          
            plot_settings[key_plot]['ip_err'] = {'HD106315c':[0.51,0.69]}          
            plot_settings[key_plot]['b_range_all'] = {'HD106315c':[0.5714-0.2 ,0.5714+0.2]}    #aRscosi = 29.5*np.cos(88.89*np.pi/180.) = 0.5714 different des valeurs publiees. Je prends des barres d'erreurs de 0.2 parce que typiques de tous les autres papiers 
        elif gen_dic['star_name']=='K2_105':
            plot_settings[key_plot]['aRs_err'] = {'K2_105b':[0.19,0.19]}   
            plot_settings[key_plot]['lambdeg_err'] = {'K2_105b':[47.,50.]}          
            plot_settings[key_plot]['ip_err'] = {'K2_105b':[0.1,0.1]}          
            plot_settings[key_plot]['b_range_all'] = {'K2_105b':[0.42-0.03,0.42+0.03]}    #aRscosi = 17.39*np.cos(88.62*np.pi/180.) = 0.42 ok
        elif gen_dic['star_name']=='Kepler25':
            plot_settings[key_plot]['aRs_err'] = {'Kepler25c':[0.27,0.27]}   
            plot_settings[key_plot]['lambdeg_err'] = {'Kepler25c':[9.7,9.1]}          
            plot_settings[key_plot]['ip_err'] = {'Kepler25c':[0.042,0.039]}          
            plot_settings[key_plot]['b_range_all'] = {'Kepler25c':[0.8842-0.0018,0.8842+0.0018]}    #aRscosi = 18.336*np.cos(87.236*np.pi/180.) = 0.88420 different de Benomar2014 pour le meme aRs et ip, mais proche, donc je prends meme erreurs
        elif gen_dic['star_name']=='Kepler63':
            plot_settings[key_plot]['aRs_err'] = {'Kepler63b':[0.08,0.08]}   
            plot_settings[key_plot]['lambdeg_err'] = {'Kepler63b':[26.8,21.2]}          
            plot_settings[key_plot]['ip_err'] = {'Kepler63b':[0.019,0.018]}          
            plot_settings[key_plot]['b_range_all'] = {'Kepler63b':[0.73197-0.003,0.73197+0.003]}    #aRscosi = 19.12*np.cos(87.806*np.pi/180.) = 0.73197 ok
        elif gen_dic['star_name']=='WASP107':
            plot_settings[key_plot]['aRs_err'] = {'WASP107b':[0.27,0.27]}   
            plot_settings[key_plot]['lambdeg_err'] = {'WASP107b':[18.5,15.2]}          
            plot_settings[key_plot]['ip_err'] = {'WASP107b':[0.078,0.078]}          
            plot_settings[key_plot]['b_range_all'] = {'WASP107b':[0.138382-0.024,0.138382+0.024]}    #aRscosi = 18.02*np.cos(89.56*np.pi/180.) = 0.138382 ok
        elif gen_dic['star_name']=='WASP156':
            plot_settings[key_plot]['aRs_err'] = {'WASP156b':[0.027,0.025]}   
            plot_settings[key_plot]['lambdeg_err'] = {'WASP156b':[14.4,14.0]}          
            plot_settings[key_plot]['ip_err'] = {'WASP156b':[0.028,0.033]}          
            plot_settings[key_plot]['b_range_all'] = {'WASP156b':[0.24428-0.0073,0.24428+0.0061]}    #aRscosi = 12.748*np.cos(88.902*np.pi/180.) = 0.24428 ok
        elif gen_dic['star_name']=='WASP166':
            plot_settings[key_plot]['aRs_err'] = {'WASP166b':[0.50,0.42]}   
            plot_settings[key_plot]['lambdeg_err'] = {'WASP166b':[1.6,1.6]}          
            plot_settings[key_plot]['ip_err'] = {'WASP166b':[0.59,0.62]}          
            plot_settings[key_plot]['b_range_all'] = {'WASP166b':[0.398-0.111,0.398+0.093]}    #aRscosi = 11.14*np.cos(87.95*np.pi/180.) = 0.39849 ok


        #Choose view
        #    - sky_orb : X axis is the node line of the main planet, Y axis the projection of its orbital plane normal, Z axis the LOS
        #    - sky_ste : X axis is the sky-projected stellar equator, Y axis the sky-projected stellar spin axis, Z axis the LOS
        #    - we do not define a stellar mode because the Z axis upon which the radial velocities are defined is then shifted
        plot_settings[key_plot]['conf_system']='sky_orb'
        plot_settings[key_plot]['conf_system']='sky_ste'

        #Reference planet for the 'sky_orb' configuration
        plot_settings[key_plot]['pl_ref'] = plot_settings[key_plot]['pl_to_plot'][0]
        
        #Orbit width
        plot_settings[key_plot]['lw_plot'] = 2

        #Overlaying stellar grid cell boundaries
        plot_settings[key_plot]['st_grid_overlay']=True & False

        #Overlaying planets grid cell boundaries
        plot_settings[key_plot]['pl_grid_overlay']=True & False

        #Color stellar disk with RV, with limb-darkened specific intensity, with gravity-darkened specific intensity, or total flux
        #    - disk_color = 'RV', 'LD', 'GD', 'F'
        plot_settings[key_plot]['disk_color']='RV'  #classic GJ436
        # plot_settings[key_plot]['disk_color']='LD'  #dezoom GJ436
        # plot_settings[key_plot]['disk_color']='GD'  
        # plot_settings[key_plot]['disk_color']='F'
        if gen_dic['star_name']=='HD209458':
            plot_settings[key_plot]['disk_color'] = 'F'    #ANTARESS I, oblate view

        #Choice of spectral band for intensity
        #    - from the main planet transit properties
        plot_settings[key_plot]['iband']=0

        #Plot colorbar
        plot_settings[key_plot]['plot_colbar']=True #classic GJ436
    #    plot_settings[key_plot]['plot_colbar']=False #dezoom GJ436
    
        #Plot visible stellar equator
        if gen_dic['star_name'] in ['HAT_P33','HAT_P49','HD106315','K2_105','WASP156']:plot_settings[key_plot]['plot_equ_vis']=False       
    
        #Plot hidden equator
        plot_settings[key_plot]['plot_equ_hid']=True  &  False
        
        #Plot stellar spin
        plot_settings[key_plot]['plot_stspin']=True  #  &  False        
        
        #Plot stellar poles
        plot_settings[key_plot]['plot_hidden_pole']=True & False
        if gen_dic['star_name'] in ['HAT_P33','HAT_P49','HD106315','K2_105','WASP156']:plot_settings[key_plot]['plot_poles']=False
        # if gen_dic['star_name']=='HD209458':  #ANTARESS I multi-pl  
        #     plot_settings[key_plot]['plot_equ_vis']=False  
        #     plot_settings[key_plot]['plot_stspin']=False  

        #Number of cells on a diameter of the star (must be odd)
        #    - leave undefined for default settings to be used
        # plot_settings[key_plot]['n_stcell']=11. 
        # plot_settings[key_plot]['n_stcell']=201. #classic GJ436
        # plot_settings[key_plot]['n_stcell']=101.
        # plot_settings[key_plot]['n_stcell']=301. #espresso paper
    #    plot_settings[key_plot]['n_stcell']=201  #dezoom GJ436
    #    plot_settings[key_plot]['n_stcell']=1001.   #to get a smooth plot of the equiRV. The plot must be in png not to be too heavy
        # if gen_dic['star_name'] in ['HAT_P3','HAT_P11','HAT_P33','HAT_P49','HD89345','HD106315','K2_105','Kepler25','Kepler63','WASP107','WASP156','WASP166']:plot_settings[key_plot]['n_stcell']=201.     
        # if gen_dic['star_name']=='HD209458':   #ANTARESS I multi-pl           
        #     plot_settings[key_plot]['n_stcell']=41.             
        
        
        #Number of cells on a diameter of planets (must be odd)
        #    - leave undefined for default settings to be used       
        # if gen_dic['star_name']=='HD209458':   #ANTARESS I multi-pl
        #     plot_settings[key_plot]['n_plcell']['HD209458b'] = 15
        #     plot_settings[key_plot]['n_plcell']['HD209458c'] = 17  #21
        
        
        
        # Stage Théo : Plot stellar spots
        plot_settings[key_plot]['stellar_spot'] = {}
        #         plot_settings[key_plot]['stellar_spot']['spot1'] = {'lat' :  30, 'Tcenter' : 2458877.6306 - 12/24, 'ang' : 10, 'flux' : 0.6}
        #         plot_settings[key_plot]['stellar_spot']['spot2'] = {'lat' : -40, 'Tcenter' : 2458877.6306  + 5/24, 'ang' : 7, 'flux' : 0.6}
        
        # Number of positions of the spots to be plotted, equally distributed within the given time range.
        plot_settings[key_plot]['n_image_spots'] = 15
        
        # time range (BJD) for ploting spots
        plot_settings[key_plot]['time_range_spot'] = 2458877.6306   + np.array([-7.5/24, -1.5/24])
        
        # Use stellar rotation period to distribute the positions, instead of time
        plot_settings[key_plot]['plot_spot_all_Peq'] = True    
                                
        
        
                                
        #Overlay to the RV-colored disk a shade controlled by flux
        plot_settings[key_plot]['shade_overlay']=True      
    
        #Number of equi-RV curves
        #    - must be even not to overplot the stellar spin axis
        #    - coded in 'sky_ste' mode only
        #    - set to None to prevent
        plot_settings[key_plot]['n_equi']=None   #40   #None   #10    
        # if gen_dic['star_name']=='HD209458':
        #     plot_settings[key_plot]['n_equi'] = 15     #ANTARESS I   
    
        #RV range
        plot_settings[key_plot]['val_range'] = None    
        # if gen_dic['star_name']=='HD209458':
        #     plot_settings[key_plot]['val_range'] = [-3.5,3.5]       #ANTARESS I, RV view   
    
        #Overlay axis of selected frame
        plot_settings[key_plot]['axis_overlay']=False   #classic GJ436
    #    plot_settings[key_plot]['axis_overlay']=True   #dezoom GJ436    

        #Plot axis boundaries
        #    - x0,y0,x1,y1
        plot_settings[key_plot]['margins']=[0.12,0.15,0.85,0.88]

        #Font size
        plot_settings[key_plot]['font_size']=22   #14
        #plot_settings[key_plot]['font_size']=18	
        plot_settings[key_plot]['font_size']=27   #RM survey

        #Plot boundaries
        #    - if modified, the overlay of the pixelled star edge with a white annulus must be adjusted 
        if gen_dic['star_name']=='HD3167':
            plot_settings[key_plot]['x_range'] = np.array([-2.,2.])   
            plot_settings[key_plot]['y_range'] = np.array([-2.,2.])  
            if gen_dic['studied_pl']==['HD3167_c']:
                plot_settings[key_plot]['x_range'] = np.array([-2.,2.])   
                plot_settings[key_plot]['y_range'] = np.array([-2.,2.])       
            # plot_settings[key_plot]['x_range'] = np.array([-3.5,3.5])     
            # plot_settings[key_plot]['y_range'] = np.array([-3.5,3.5])      
    
        elif gen_dic['star_name']=='TOI858':
            plot_settings[key_plot]['x_range'] = np.array([-2.,2.])   
            plot_settings[key_plot]['y_range'] = np.array([-2.,2.])     
    
        elif gen_dic['star_name']=='HD15337':
            plot_settings[key_plot]['x_range'] = np.array([-10.,10.])   
            plot_settings[key_plot]['y_range'] = np.array([-10.,10.])       

        # elif gen_dic['star_name']=='GJ436':            
        #    plot_settings[key_plot]['x_range'] = np.array([-22.,22.])    #GJ436 inset dezoomed
        #    plot_settings[key_plot]['y_range'] = np.array([-22.,22.])   
    
        elif gen_dic['star_name']=='MASCARA1':
            plot_settings[key_plot]['x_range'] = np.array([-1.1,1.1])   
            plot_settings[key_plot]['y_range'] = np.array([-1.1,1.1])      

        # elif gen_dic['star_name']=='HD209458':
        #     plot_settings[key_plot]['x_range'] = np.array([-0.5+0.3,0.5+0.3])      #ANTARESS multi-pl
        #     plot_settings[key_plot]['y_range'] = np.array([-1.05,-0.05])  

















    '''
    Plotting series of binned disk-integrated and intrinsic profiles together for comparison
    '''
    if (plot_dic['binned_DI_Intr']!=''):
        key_plot = 'binned_DI_Intr'


        #Data type
        plot_settings[key_plot]['data_type']='CCF' 

        #Choose bin dimension for disk-integrated profiles
        #    - 'phase' 
        plot_settings[key_plot]['dim_plot_DI']='phase' 

        #Choose bin dimension for intrinsic profiles
        #    - 'xp_abs', 'r_proj' (see details and routine)
        plot_settings[key_plot]['dim_plot_intr']='r_proj' 
    

        #Instruments and visits to plot
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}   
        elif gen_dic['star_name']=='GJ436':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-28','2019-04-29','binned']}  
        
        #Exposures to plot for each series
        # if gen_dic['studied_pl']=='WASP76b':
        #     plot_settings[key_plot]['iexp_plot']={'ESPRESSO':{
        #         'DIbin':np.arange(2,37,dtype=int),
        #         'Intrbin':np.arange(1,20,dtype=int)}}

        #Colors
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['color_dic']={'DI':'dodgerblue','Intr':'red'} 
        elif gen_dic['star_name']=='GJ436':plot_settings[key_plot]['color_dic']={'DI':'dodgerblue','Intr':'red'} 
        elif gen_dic['star_name']=='HAT_P3':plot_settings[key_plot]['color_dic']={'DI':'dodgerblue','Intr':'red'} 

            
        #Plot boundaries
#         if gen_dic['studied_pl']=='WASP76b':
#             plot_settings[key_plot]['x_range']=[3500.,8000.] 
#             plot_settings[key_plot]['x_range']=[5880.,5905.] 
# #            plot_settings[key_plot]['x_range']=[6200.,6300.] 
#             # plot_settings[key_plot]['x_range']=None

        #Plot boundaries in flux
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['y_range']=[-120.,120.] 
            plot_settings[key_plot]['y_range']=[3e3,7e3]
        elif gen_dic['star_name']=='GJ436':
            plot_settings[key_plot]['x_range']=[-35.,35.] 







    '''
    Plotting all individual atmospheric profiles from a given visit together
    '''
    if (plot_dic['all_atm_data']!=''):
        key_plot = 'all_atm_data'
        plot_settings[key_plot]={}   

        #Signal type 
        #    - 'Absorption' or 'Emission'
        plot_settings[key_plot]['pl_atm_sign']='Absorption'
        
        #Data type
        plot_settings[key_plot]['data_type']='CCF'        

        #Plot continuum pixels
        #    - for CCFs only
        plot_settings[key_plot]['plot_cont']= True & False

        #Instruments and visits to plot
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}   
        elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2019-02-28','2019-04-29']}  
        
        #Exposures to plot
        #    - indexes are relative to in-transit tables
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['iexp_plot']={'ESPRESSO':{
                '2018-10-31':np.arange(2,37,dtype=int),
                '2018-09-03':np.arange(1,20,dtype=int)}}

        #Colors
        # if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'}} 
        # elif gen_dic['studied_pl']=='GJ436_b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2019-02-28':'dodgerblue','2019-04-29':'red'}} 
            
        #Plot boundaries
#         if gen_dic['studied_pl']=='WASP76b':
#             x_range=[3500.,8000.] 
#             x_range=[5880.,5905.] 
# #            x_range=[6200.,6300.] 

        #Plot boundaries in flux
        if gen_dic['studied_pl']=='WASP76b':
            x_range=[-120.,120.] 
            y_range=[3e3,7e3]
        elif gen_dic['studied_pl']=='WASP76b':
            x_range=[-25.,25.] 
            # y_range=[3e3,7e3]












    '''
    Plotting individual atmospheric spectral profiles
    '''
    if any('spec' in s for s in data_dic['Atm']['type'].values()) and (plot_dic['sp_atm']!=''):    
        key_plot = 'sp_atm'
        plot_settings[key_plot]={}   

        #Signal type 
        #    - 'Absorption' or 'Emission'
        plot_settings[key_plot]['pl_atm_sign']='Absorption'

        #Plot aligned profiles
        plot_settings[key_plot]['aligned']=True  &  False

        #Instruments and visits to plot
        if gen_dic['studied_pl']=='WASP76b':visits_to_plot={'ESPRESSO':['2018-10-31','2018-09-03']}            

        #Colors
        if gen_dic['studied_pl']=='WASP76b':color_dic={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'}} 
            
        #Plot boundaries in wav
        if gen_dic['studied_pl']=='WASP76b':
            x_range=[3500.,8000.] 
            x_range=[5880.,5905.] 
#            x_range=[6200.,6300.] 

        #Plot boundaries in flux
        if gen_dic['studied_pl']=='WASP76b':
            y_range=[-0.05,0.1] 








    '''
    Plotting individual atmospheric CCF profiles
    '''
    if ('CCF' in data_dic['Atm']['type'].values()) and ((plot_dic['CCFatm']!='') or (plot_dic['CCFatm_res']!='')):
        for key_plot in ['CCFatm','CCFatm_res']:
            if plot_dic[key_plot]!='':
                plot_settings[key_plot]={} 

                #Signal type 
                #    - 'Absorption' or 'Emission'
                plot_settings[key_plot]['pl_atm_sign']='Absorption'        
                
                #Overplot continuum pixels
                plot_settings[key_plot]['plot_cont']=True  &  False

                #Plot errors
                plot_settings[key_plot]['plot_err']=True   &  False
                    
                #Choose model to use
                #    - from the fit to individual CCFs ('indiv') or from the global fit to all CCFs ('global')
                plot_settings[key_plot]['fit_type']='global'           
                        
                #Shade area not included in fit
                plot_settings[key_plot]['plot_nofit']=True    & False

                #Plot planet rest velocity
                plot_settings[key_plot]['plot_refvel']=True    &   False

                #visits to plot
                if gen_dic['studied_pl']=='WASP76b':
                    plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}              
                if gen_dic['studied_pl']=='WASP121b':
            #        plot_settings[key_plot]['visits_to_plot']={'HARPS':['14-01-18','09-01-18','31-12-17'],
            #                        'binned':['HARPS-binned']}
        #            plot_settings[key_plot]['visits_to_plot']={'HARPS':['14-01-18','09-01-18'],
        #                            'binned':['HARPS-binned-2018']}
                    plot_settings[key_plot]['visits_to_plot']={'binned':['HARPS-binned']}
                elif gen_dic['studied_pl']=='Kelt9b':
                    plot_settings[key_plot]['visits_to_plot']={'HARPN':['31-07-2017']}
        
                #Color dictionary
                if gen_dic['studied_pl']=='WASP76b':
                    color_dic={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'}} 
                if gen_dic['studied_pl']=='Kelt9b':
                    color_dic={'31-07-2017':'dodgerblue'}   
                elif gen_dic['studied_pl']=='WASP121b':
                    color_dic={'HARPS-binned':'dodgerblue'}   
                    
                #Bornes du plot en RV
                plot_settings[key_plot]['x_range']=[5.,50.]   
                if gen_dic['studied_pl']=='WASP76b':
                    plot_settings[key_plot]['x_range']=[-110.,110.] 
        

        #-------------------------------------
        #Plot each atmospheric CCF and its fit
        if (plot_dic['CCFatm']!=''):
            key_plot='CCFatm'

            #Plot aligned CCFs
            plot_settings[key_plot]['aligned']=False
            
            #Overplot fit (aligned option must not be requested)
            plot_settings[key_plot]['plot_line_model']=True   # &   False

            #Print CCFs fit properties on plot
            plot_settings[key_plot]['plot_prop']=True      &   False
            
            #Plot measured centroid
            plot_settings[key_plot]['plot_line_fit_rv']=True

            #Plot fitted pixels 
            plot_settings[key_plot]['plot_fitpix']=True    &   False
    
            #Plot continuum pixels specific to each exposure 
            plot_settings[key_plot]['plot_cont_exp']=True     &   False


            #Bornes du plot
            #    - true values, before scaling factor     
            if gen_dic['studied_pl']=='WASP76b':
                y_range=[-0.05,0.1]   

        #-------------------------------------
        #Plot residuals between the atmospheric CCFs and their fit
        if (plot_dic['CCFatm_res']!='') and ((gen_dic['ana_atm']) or (gen_dic['fit_atm_all'])):
            key_plot='CCFatm_res'

            #Print dispersions of residuals in various ranges
            plot_settings[key_plot]['plot_prop']=True #  &   False
            
            #Bornes du plot en RV       
            if gen_dic['studied_pl']=='WASP76b':  
                y_range=[-400.,400.]

                
            








    '''
    Plotting individual binned atmospheric spectral profiles
    '''
    if any('spec' in s for s in data_dic['Atm']['type'].values()) and (plot_dic['sp_Atmbin']!=''):     
        key_plot = 'sp_Atmbin'
        plot_settings[key_plot]={} 

        #Signal type 
        #    - 'Absorption' or 'Emission'
        plot_settings[key_plot]['pl_atm_sign']='Absorption'

        #Choose bin dimension
        #    - 'phase' (see details and routine)
        plot_settings[key_plot]['dim_plot']='phase' 
     

        #Instruments and visits to plot
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}            

        #Colors
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'}} 
            
        #Plot boundaries in wav
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['x_range']=[3500.,8000.] 
            plot_settings[key_plot]['x_range']=[5880.,5905.] 
#            x_range=[6200.,6300.] 
            # x_range=None

        #Plot boundaries in signal
#        if gen_dic['studied_pl']=='WASP76b':
#            y_range=[0.35,1.05] 










    '''
    Plotting individual binned atmospheric CCF profiles
    '''
    if ('CCF' in data_dic['Atm']['type'].values()) and ((plot_dic['CCF_Atmbin']!='') or (plot_dic['CCF_Atmbin_res']!='')):
        for key_plot in ['CCF_Atmbin','CCF_Atmbin_res']:
            if plot_dic[key_plot]!='':
                plot_settings[key_plot]={} 

                #Signal type 
                #    - 'Absorption' or 'Emission'
                plot_settings[key_plot]['pl_atm_sign']='Absorption'

                #Overplot continuum pixels
                plot_settings[key_plot]['plot_cont']=True  &  False
                        
                #Shade area not included in fit
                plot_settings[key_plot]['plot_nofit']=True    & False

                #Select plot mode
                plot_settings[key_plot]['drawstyle']='steps-mid'
                
                #Choose bin dimension
                #    - 'phase' (see details and routine)
                plot_settings[key_plot]['dim_plot']='phase'    
        
                #Plot planetary exclusion range
                plot_settings[key_plot]['plot_plexc']=True
        
                #Scaling factor (in power of ten, ie flux are multiplied by 10**sc_fact10)
                plot_settings[key_plot]['sc_fact10']=6.


                #visits to plot
                if gen_dic['studied_pl']=='WASP121b':
            #        plot_settings[key_plot]['visits_to_plot']={'HARPS':['14-01-18','09-01-18','31-12-17'],
            #                        'binned':['HARPS-binned']}
                    plot_settings[key_plot]['visits_to_plot']={'HARPS':['14-01-18','09-01-18'],
                                    'binned':['HARPS-binned-2018']}
                    plot_settings[key_plot]['visits_to_plot']={'binned':['HARPS-binned']}
                if gen_dic['studied_pl']=='Kelt9b':
                    plot_settings[key_plot]['visits_to_plot']={'HARPN':['31-07-2017']}
                elif gen_dic['studied_pl']=='WASP76b':
                    plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03'],'binned':['ESP_binned']} 
        
        
                #Color dictionary
                if gen_dic['studied_pl']=='Kelt9b':
                    plot_settings[key_plot]['color_dic']={'HARPN':{'31-07-2017':'dodgerblue'}} 
                if gen_dic['studied_pl']=='WASP121b':
                    plot_settings[key_plot]['color_dic']={'binned':{'HARPS-binned':'dodgerblue'}}
                elif gen_dic['studied_pl']=='WASP76b':
                    plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'},'binned':{'ESP_binned':'black'}}  
                    
                #Bornes du plot en RV
                plot_settings[key_plot]['x_range']=[-21.,21.] 
                plot_settings[key_plot]['x_range']=[-150.,150.] 
                plot_settings[key_plot]['x_range']=[-110.,110.] 
                plot_settings[key_plot]['y_range']=[-0.4e-3  ,  0.9e-3] 
                plot_settings[key_plot]['x_range'] = None
                plot_settings[key_plot]['y_range'] = None
                if gen_dic['studied_pl']=='Kelt9b':
                    x_range=[-150.,150.] 
                    y_range=[-0.5  ,  1.] 
                elif gen_dic['studied_pl']=='WASP76b':
                    x_range=[-100.,100.] 
                    y_range=[-0.3e-3  ,  0.7e-3] 

        #---------------------------------
        #Individual binned atmospheric CCF profiles
        if (plot_dic['CCF_Atmbin']!=''):  
            key_plot='CCF_Atmbin'

            #Overplot fit 
            plot_settings[key_plot]['plot_line_model']=True   # &   False

            #Print CCFs fit properties on plot
            plot_settings[key_plot]['plot_prop']=True      &   False
            
            #Plot measured centroid
            plot_settings[key_plot]['plot_line_fit_rv']=True

            #Plot fitted pixels 
            plot_settings[key_plot]['plot_fitpix']=True    &   False
    
            #Plot continuum pixels specific to each exposure 
            plot_settings[key_plot]['plot_cont_exp']=True     &   False            
            
        #---------------------------------
        #Residual from binned atmospheric CCF profiles fit
        if (plot_dic['CCF_Atmbin_res']!=''):  
            key_plot='CCF_Atmbin_res'            
            








    '''
    Plotting individual 1D atmospheric spectral profiles
    '''
    if any('spec' in s for s in data_dic['Atm']['type'].values()) and (plot_dic['sp_1D_atm']!=''):
        key_plot = 'sp_1D_atm'  
        plot_settings[key_plot]={} 

        #Signal type 
        #    - 'Absorption' or 'Emission'
        plot_settings[key_plot]['pl_atm_sign']='Absorption'
  
        #Scaling factor (in power of ten, ie flux are multiplied by 10**sc_fact10)
        plot_settings[key_plot]['sc_fact10']=-5.

        #Instruments and visits to plot
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']}            

        #Colors
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'}} 
            
        #Plot boundaries in wav
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['x_range']=[3500.,8000.] 
            plot_settings[key_plot]['x_range']=[5880.,5905.] 
#            plot_settings[key_plot]['x_range']=[6200.,6300.] 
            # plot_settings[key_plot]['x_range']=None

        #Plot boundaries in signal
#        if gen_dic['studied_pl']=='WASP76b':
#            plot_settings[key_plot]['y_range']=[0.35,1.05] 






    '''
    Plotting chi2 values for the properties of the atmospheric profiles fitted in each exposure
    '''
    if (plot_dic['chi2_fit_atm_prop']!=''):
        key_plot = 'chi2_fit_atm_prop'  
        plot_settings[key_plot]={} 
        
        #Ranges
        #    - set to None for automatic determination
        plot_settings[key_plot]['x_range']=[-0.021,0.021]
        
        #Threshold to identify and print outliers
        #    - set to None to prevent
        plot_settings[key_plot]['chi2_thresh']=3.  

        #Visits to plot
        if gen_dic['studied_pl']=='WASP76b':plot_settings[key_plot]['visits_to_plot']={'ESPRESSO':['2018-10-31','2018-09-03']} 

        #Colors
        if gen_dic['studied_pl']=='WASP76b':            
            plot_settings[key_plot]['color_dic']={'ESPRESSO':{'2018-10-31':'dodgerblue','2018-09-03':'red'}}        
        
        
        
        
        
        
        
    '''
    2D maps of atmospheric profiles
    '''
    if (plot_dic['map_Atm_prof']!=''):
        key_plot = 'map_Atm_prof'  
        plot_settings[key_plot]={}  
              
        #Plot profiles in star (False) or planet rest frame (True)
        plot_settings[key_plot]['aligned']=False        
        
        #Color range     
        if gen_dic['studied_pl']=='WASP121b':
            v_range_comm=[-0.0035  ,  0.0035]
            v_range_comm=[-0.002  ,  0.002]
#                    v_range_comm=[-0.4e-3  ,  0.9e-3]
#                    v_range_comm=[-1.5e-3  ,  1.5e-3]
            plot_settings[key_plot]['v_range_all']={'HARPS':{'14-01-18':v_range_comm,
                                  '09-01-18':v_range_comm,
                                  '31-12-17':v_range_comm},
                         'binned':{'HARPS-binned':v_range_comm,'HARPS-binned-2018':v_range_comm}                
                    }
        elif gen_dic['studied_pl']=='Kelt9b':
            v_range_comm=[-0.003  ,  0.003]
            plot_settings[key_plot]['v_range_all']={'HARPN':{'31-07-2017':v_range_comm}}  
        elif gen_dic['studied_pl']=='WASP76b':
            v_range_comm=[-0.0005  ,  0.0015]
            v_range_comm=[-0.02  ,  0.1]
            v_range_comm=[-0.02  ,  0.02]    #Na
            # v_range_comm=[-5.  ,  5.]    #Fe
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2018-10-31':v_range_comm,
                                     '2018-09-03':v_range_comm},
                         'binned':{'ESP_binned':v_range_comm}}
            # plot_settings[key_plot]['v_range_all']={}
        elif gen_dic['star_name']=='HAT_P49':
            plot_settings[key_plot]['v_range_all']={'HARPN':{'20200730':[-3e-3,3e-3]}}

        #Ranges
        if gen_dic['studied_pl']=='WASP121b':
            plot_settings[key_plot]['x_range']=[-80.,80.]
#                        plot_settings[key_plot]['x_range']=[-100.,100.]
#            plot_settings[key_plot]['y_range_all']={'HARPS':{'14-01-18':[-0.15,0.11],
#                                  '09-01-18':[-0.1,0.18],
#                                  '31-12-17':[-0.12,0.11]}}
            if plot_settings[key_plot]['pl_atm_sign']=='Emission':
                y_range_comm=[-0.15,0.18]
            if plot_settings[key_plot]['pl_atm_sign']=='Absorption':
                y_range_comm=[-0.07,0.07]
#                            y_range_comm=[-0.15,0.18]

                
            plot_settings[key_plot]['y_range_all']={'HARPS':{'14-01-18':y_range_comm,'09-01-18':y_range_comm,'31-12-17':y_range_comm},
                         'binned':{'HARPS-binned':y_range_comm,'HARPS-binned-2018':y_range_comm}   }

        elif gen_dic['studied_pl']=='Kelt9b':
            plot_settings[key_plot]['x_range']=[-300.,300.]
            if plot_settings[key_plot]['pl_atm_sign']=='Emission':
                y_range_comm=[-0.15,0.18]
            if plot_settings[key_plot]['pl_atm_sign']=='Absorption':
                y_range_comm=[-0.065,0.065]                           
            plot_settings[key_plot]['y_range_all']={'HARPN':{'31-07-2017':y_range_comm}}

        elif gen_dic['studied_pl']=='WASP76b':
            
            plot_settings[key_plot]['x_range']=[-100,100]
            plot_settings[key_plot]['orders_to_plot']=[8,9]
            plot_settings[key_plot]['x_range']=[5885.,5901.]
            # 
            # plot_settings[key_plot]['x_range']=[-100.,100]
            y_range_comm=[-0.05,0.05]
            plot_settings[key_plot]['sc_fact10']=0.
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2018-10-31':y_range_comm,'2018-09-03':y_range_comm},'binned':{'ESP_binned':y_range_comm}}                            
        elif gen_dic['star_name']=='HAT_P49': 
            plot_settings[key_plot]['margins']=[0.15,0.15,0.85,0.85] 
            plot_settings[key_plot]['x_range']=[-150,150]













    '''
    2D maps of binned atmospheric profiles
    '''
    if (plot_dic['map_Atmbin']!=''):
        key_plot = 'map_Atmbin'  
        plot_settings[key_plot]={}  

        #Color range
        if gen_dic['studied_pl']=='WASP76b':
            v_range_comm=[-0.0005  ,  0.0015]
            v_range_comm=[-0.02  ,  0.1]
            v_range_comm=[-0.02  ,  0.02]    #Na
            # v_range_comm=[-5.  ,  5.]    #Fe
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2018-10-31':v_range_comm,
                                     '2018-09-03':v_range_comm},
                         'binned':{'ESP_binned':v_range_comm}}
            # plot_settings[key_plot]['v_range_all']={}       
             
        #Ranges
        if gen_dic['studied_pl']=='WASP76b':
            plot_settings[key_plot]['x_range']=[-100,100]
            plot_settings[key_plot]['orders_to_plot']=[8,9]
            plot_settings[key_plot]['x_range']=[5885.,5901.]
            # 
            # plot_settings[key_plot]['x_range']=[-100.,100]
            y_range_comm=[-0.05,0.05]
            plot_settings[key_plot]['sc_fact10']=0.
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2018-10-31':y_range_comm,'2018-09-03':y_range_comm},'binned':{'ESP_binned':y_range_comm}}                            
                            
            
            
            
            
            
            


    '''
    2D maps of 1D atmospheric profiles
    '''
    if (plot_dic['map_Atm_1D']!=''):
        key_plot = 'map_Atm_1D'  
        plot_settings[key_plot]={}  

        #Color range
        if gen_dic['studied_pl']=='WASP76b':
            v_range_comm=[3,7]
            plot_settings[key_plot]['v_range_all']={'ESPRESSO':{'2018-10-31':v_range_comm,'2018-09-03':v_range_comm}}  
        
        #Ranges
        if gen_dic['studied_pl']=='WASP76b':
            # plot_settings[key_plot]['x_range']=[5880.,5905.]
            # plot_settings[key_plot]['x_range']=[-150,150]
            plot_settings[key_plot]['x_range']=[-100.,100]
            y_range_comm=[-0.048  ,  0.048]
            plot_settings[key_plot]['y_range_all']={'ESPRESSO':{'2018-10-31':y_range_comm,'2018-09-03':y_range_comm}}





        
    return plot_settings