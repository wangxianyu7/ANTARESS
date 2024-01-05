import numpy as np
from itertools import product as it_product
from copy import deepcopy
from utils import stop,closest,np_poly,npint,np_interp,np_where1D,datasave_npz,dataload_npz,gen_specdopshift,check_data
from constant_data import Rsun,c_light
import lmfit
from lmfit import Parameters
from ANTARESS_grids.ANTARESS_coord import frameconv_LOS_to_InclinedStar,frameconv_InclinedStar_to_LOS,calc_pl_coord
from ANTARESS_routines.ANTARESS_data_align import align_data
from ANTARESS_analysis.ANTARESS_inst_resp import convol_prof
from ANTARESS_grids.ANTARESS_star_grid import calc_CB_RV,get_LD_coeff,calc_st_sky,calc_Isurf_grid,calc_RVrot
from ANTARESS_analysis.ANTARESS_model_prof import calc_polymodu,polycoeff_def
from ANTARESS_grids.ANTARESS_prof_grid import coadd_loc_line_prof,calc_loc_line_prof,init_st_intr_prof,calc_linevar_coord_grid
from ANTARESS_grids.ANTARESS_spots import is_spot_visible, calc_spotted_tiles, new_calc_spotted_region_prop, new_retrieve_spots_prop_from_param, new_new_calc_spotted_region_prop

def calc_plocc_prop(system_param,gen_dic,theo_dic,coord_dic,inst,vis,data_dic,calc_pl_atm=False,mock_dic={}):
    r"""**Planet-occulted properties: workflow**

    Calls function to calculate theoretical properties of the regions occulted by all transiting planets. 

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    print('   > Calculating properties of planet-occulted regions')    
    if (gen_dic['calc_theoPlOcc']):
        print('         Calculating data')

        #Theoretical RV of the planet occulted-regions
        #    - calculated for the nominal and broadband planet properties 
        #    - for the nominal properties we retrieve the range of some properties covered by the planet during each exposures
        #    - chromatic transit required if local profiles are in spectral mode  
        params = deepcopy(system_param['star'])
        params.update({'rv':0.,'cont':1.})
        
        #Add spot properties
        params['use_spots']=False

        if mock_dic!={} and mock_dic['use_spots']:
            params['use_spots']=True
            for spot_param in mock_dic['spots_prop'][inst][vis].keys():
                params[spot_param]=mock_dic['spots_prop'][inst][vis][spot_param]
            #Figuring out the number of spots
            num_spots = 0
            for par in params:
                if 'lat__IS'+inst+'_VS'+vis+'_SP' in par:
                    num_spots +=1
            params['num_spots']=num_spots
            params['inst']=inst
            params['vis']=vis

        par_list=['rv','CB_RV','mu','lat','lon','x_st','y_st','SpSstar','xp_abs','r_proj']
        key_chrom = ['achrom']
        if ('spec' in data_dic[inst][vis]['type']) and ('chrom' in data_dic['DI']['system_prop']):key_chrom+=['chrom']

        if params['use_spots']:
            plocc_prop, spotocc_prop = sub_calc_plocc_prop(key_chrom,{},par_list,data_dic[inst][vis]['transit_pl'],system_param,theo_dic,data_dic['DI']['system_prop'],params,coord_dic[inst][vis],gen_dic[inst][vis]['idx_in'], system_spot_prop_in = data_dic['DI']['spots_prop'], out_ranges=True)
            
            #Save spot-occulted region properties
            datasave_npz(gen_dic['save_data_dir']+'Introrig_prop/SpotOcc_Prop_'+inst+'_'+vis,spotocc_prop)    

        else:
            plocc_prop = sub_calc_plocc_prop(key_chrom,{},par_list,data_dic[inst][vis]['transit_pl'],system_param,theo_dic,data_dic['DI']['system_prop'],params,coord_dic[inst][vis],gen_dic[inst][vis]['idx_in'], out_ranges=True)

        #Save planet-occulted region properties
        datasave_npz(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis,plocc_prop)    

    else:
        check_data({'path':gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis})        

    return None


def up_plocc_prop(inst,vis,par_list,args,param_in,transit_pl,nexp_fit,ph_fit,coord_pl_fit, transit_spots=[], coord_spot_fit={}):
    r"""**Planet-occulted properties: update**

    Updates properties of the planet-occulted region and planetary orbit for fitted step. 

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    system_param_loc=deepcopy(args['system_param'])
   
    #In case param_in is defined as a Parameters structure, retrieve values and define dictionary
    param={}
    if isinstance(param_in,lmfit.parameter.Parameters):
        for par in param_in:param[par]=param_in[par].value
    else:param=param_in

    #Coefficients describing the polynomial variation of spectral line properties as a function of the chosen coordinate
    #    - coefficients can be specific to a given spectral line model
    if (args['mode']=='ana') and (len(args['linevar_par'])>0):
        args['coeff_line'] = {}
        for par_loc in args['linevar_par'][inst][vis]:    
            args['coeff_line'][par_loc] = polycoeff_def(param,args['coeff_ord2name'][inst][vis][par_loc])
          
    #Recalculate coordinates of occulted regions or use nominal values
    #    - the 'fit_X' conditions are only True if at least one parameter is varying, so that param_fit is True if fit_X is True
    if args['fit_orbit']:coord_pl = {}
    else:coord_pl = deepcopy(coord_pl_fit)
    for pl_loc in transit_pl:

        #Recalculate planet grid if relevant
        if args['fit_RpRs'] and ('RpRs__pl'+pl_loc in args['var_par_list']):
            args['system_prop']['achrom'][pl_loc][0]=param['RpRs__pl'+pl_loc] 
            args['grid_dic']['Ssub_Sstar_pl'][pl_loc],args['grid_dic']['x_st_sky_grid_pl'][pl_loc],args['grid_dic']['y_st_sky_grid_pl'][pl_loc],r_sub_pl2=occ_region_grid(args['system_prop']['achrom'][pl_loc][0],args['grid_dic']['nsub_Dpl'][pl_loc])  
            args['system_prop']['achrom'],['cond_in_RpRs'][pl_loc] = [(r_sub_pl2<args['system_prop']['achrom'],[pl_loc][0]**2.)]        

        #Recalculate planet coordinates if relevant        
        if args['fit_orbit']:
            coord_pl[pl_loc]={}
            pl_params_loc = system_param_loc[pl_loc]
            
            #Update fitted system properties for current step 
            if ('lambda_rad__pl'+pl_loc in args['genpar_instvis']):lamb_name = 'lambda_rad__pl'+pl_loc+'__IS'+inst+'_VS'+vis 
            else:lamb_name = 'lambda_rad__pl'+pl_loc 
            if (lamb_name in args['var_par_list']):pl_params_loc['lambda_rad'] = param[lamb_name]                     
            if ('inclin_rad__pl'+pl_loc in args['var_par_list']):pl_params_loc['inclin_rad']=param['inclin_rad__pl'+pl_loc]       
            if ('aRs__pl'+pl_loc in args['var_par_list']):pl_params_loc['aRs']=param['aRs__pl'+pl_loc]  
            
            #Calculate coordinates
            #    - start/end phase have been set to None if no oversampling is requested, in which case start/end positions are not calculated
            if args['grid_dic']['d_oversamp'] is not None:phases = ph_fit[pl_loc]
            else:phases = ph_fit[pl_loc][1]
            x_pos_pl,y_pos_pl,_,_,_,_,_,_,ecl_pl = calc_pl_coord(pl_params_loc['ecc'],pl_params_loc['omega_rad'],pl_params_loc['aRs'],pl_params_loc['inclin_rad'],phases,args['system_prop']['achrom'][pl_loc][0],pl_params_loc['lambda_rad'],system_param_loc['star'])
            if args['grid_dic']['d_oversamp'] is not None:
                coord_pl[pl_loc]['st_pos'] = np.vstack((x_pos_pl[0],y_pos_pl[0]))
                coord_pl[pl_loc]['cen_pos'] = np.vstack((x_pos_pl[1],y_pos_pl[1]))
                coord_pl[pl_loc]['end_pos'] = np.vstack((x_pos_pl[2],y_pos_pl[2]))
            else:coord_pl[pl_loc]['cen_pos'] = np.vstack((x_pos_pl,y_pos_pl))
            coord_pl[pl_loc]['ecl'] = ecl_pl

    #Attempt at updating the spot-occulted region properties
    # #Recalculate coordinates of occulted regions or use nominal values
    # #    - the 'fit_X' conditions are only True if at least one parameter is varying, so that param_fit is True if fit_X is True
    # if args['fit_orbit']:coord_spot = {}
    # else:coord_spot = deepcopy(coord_spot_fit)
    # for spot in transit_spots:

    #     #Recalculate planet grid if relevant
    #     if args['fit_spot_ang'] and ('spot_ang'+spot in args['var_par_list']):
    #         args['system_spot_prop']['achrom'][spot][0]=param['spot_ang'+spot] 
    #         args['grid_dic']['x_st_sky_grid_spot'][spot],args['grid_dic']['y_st_sky_grid_spot'][spot],args['grid_dic']['Ssub_Sstar_spot'][spot] = spot_occ_region_grid(args['system_spot_prop']['achrom'][spot][0],args['grid_dic']['nsub_Dspot'][spot])  

    #     #Recalculate spot coordinates if relevant        
    #     if args['fit_orbit']:
    #         coord_spot[spot]={}
    #         pl_params_loc = system_param_loc[spot]
            
    #         #Update fitted system properties for current step 
    #         if ('lambda_rad__pl'+pl_loc in args['genpar_instvis']):lamb_name = 'lambda_rad__pl'+pl_loc+'__IS'+inst+'_VS'+vis 
    #         else:lamb_name = 'lambda_rad__pl'+pl_loc 
    #         if (lamb_name in args['var_par_list']):pl_params_loc['lambda_rad'] = param[lamb_name]                     
    #         if ('inclin_rad__pl'+pl_loc in args['var_par_list']):pl_params_loc['inclin_rad']=param['inclin_rad__pl'+pl_loc]       
    #         if ('aRs__pl'+pl_loc in args['var_par_list']):pl_params_loc['aRs']=param['aRs__pl'+pl_loc]  
            
    #         #Calculate coordinates
    #         #    - start/end phase have been set to None if no oversampling is requested, in which case start/end positions are not calculated
    #         if args['grid_dic']['d_oversamp'] is not None:phases = ph_fit[pl_loc]
    #         else:phases = ph_fit[pl_loc][1]
    #         x_pos_pl,y_pos_pl,_,_,_,_,_,_,ecl_pl = calc_pl_coord(pl_params_loc['ecc'],pl_params_loc['omega_rad'],pl_params_loc['aRs'],pl_params_loc['inclin_rad'],phases,args['system_prop']['achrom'][pl_loc][0],pl_params_loc['lambda_rad'],system_param_loc['star'])
    #         if args['grid_dic']['d_oversamp'] is not None:
    #             coord_pl[pl_loc]['st_pos'] = np.vstack((x_pos_pl[0],y_pos_pl[0]))
    #             coord_pl[pl_loc]['cen_pos'] = np.vstack((x_pos_pl[1],y_pos_pl[1]))
    #             coord_pl[pl_loc]['end_pos'] = np.vstack((x_pos_pl[2],y_pos_pl[2]))
    #         else:coord_pl[pl_loc]['cen_pos'] = np.vstack((x_pos_pl,y_pos_pl))
    #         coord_pl[pl_loc]['ecl'] = ecl_pl

    return system_param_loc,coord_pl,param






def sub_calc_plocc_prop(key_chrom,args,par_list_gen,transit_pl,system_param,theo_dic,system_prop_in,param,coord_pl_in,iexp_list, system_spot_prop_in = {}, out_ranges=False,Ftot_star=False):
    r"""**Planet-occulted properties: exposure**

    Calculates average theoretical properties of the stellar surface occulted by all transiting planets during an exposure
    
     - we normalize all quantities by the flux emitted by the occulted regions
     - all positions are in units of Rstar 
    
    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    system_prop = deepcopy(system_prop_in)  
    system_spot_prop = deepcopy(system_spot_prop_in)
    par_list_in = deepcopy(par_list_gen)
    n_exp = len(iexp_list)

    #Line properties initialization
    if ('linevar_par' in args) and (len(args['linevar_par'])>0):
        args['linevar_par_vis'] = args['linevar_par'][args['inst']][args['vis']]
    else:args['linevar_par_vis'] = []

    #Line profile initialization
    if ('line_prof' in par_list_in):
        par_list_in+=['rv','mu','SpSstar']+args['linevar_par_vis']
        
        #Chromatic / achromatic calculation
        if len(key_chrom)>1:stop('Function can only be called in a single mode to calculate line profiles')
        switch_chrom = False
        if key_chrom==['chrom']:
            
            #Full profile width is smaller than the typical scale of chromatic variations
            #    - mode is switched to closest-achromatic mode, with properties set to those of the closest chromatic band
            if (args['edge_bins'][-1]-args['edge_bins'][0]<system_prop['chrom']['med_dw']):
                key_chrom=['achrom']
                switch_chrom = True
                iband_cl = closest(system_prop['chrom']['w'],np.median(args['cen_bins']))
                for key in ['w','LD','GD_wmin','GD_wmax','GD_dw']:system_prop['achrom'][key] = [system_prop['chrom'][key][iband_cl]]
                for pl_loc in transit_pl:
                    system_prop['achrom']['cond_in_RpRs'][pl_loc] = [system_prop['chrom']['cond_in_RpRs'][pl_loc][iband_cl]] 
                    system_prop['achrom'][pl_loc] = [system_prop['chrom'][pl_loc][iband_cl]]                

            #Profiles covers a wide spectral band
            #    - requires calculation of achromatic properties                
            else:
                if (args['mode']=='ana'):stop('Analytical model not suited for wide spectral bands') 
                if (theo_dic['precision']=='high'):stop('High precision not possible for wide spectral bands') 
                key_chrom=['achrom','chrom']

    #Calculation of achromatic and/or chromatic values
    surf_prop_dic = {}
    surf_prop_dic_spot = {}
    for subkey_chrom in key_chrom:
        surf_prop_dic[subkey_chrom] = {}
        surf_prop_dic_spot[subkey_chrom] = {}
    if 'line_prof' in par_list_in:
        for subkey_chrom in key_chrom:
            surf_prop_dic[subkey_chrom]['line_prof']=np.zeros([args['ncen_bins'],n_exp],dtype=float)
            surf_prop_dic_spot[subkey_chrom]['line_prof']=np.zeros([args['ncen_bins'],n_exp],dtype=float)

    #Properties to be calculated
    par_star = deepcopy(param)
    star_params = system_param['star']
    par_list = ['Ftot']
    for par_loc in par_list_in:
        if par_loc=='rv':
            par_list+=['Rot_RV']
            if ('rv_line' in args['linevar_par_vis']):par_list+=['rv_line']
            if ('Rstar' in param) and ('Peq' in param):par_star['veq'] = 2.*np.pi*param['Rstar']*Rsun/(param['Peq']*24.*3600.)
        elif (par_loc not in ['line_prof']):par_list+=[par_loc]
    par_star['istar_rad']=np.arccos(par_star['cos_istar'])
    cb_band_dic = {}
    for subkey_chrom in key_chrom:
        if Ftot_star:
            surf_prop_dic[subkey_chrom]['Ftot_star']=np.zeros([system_prop[subkey_chrom]['nw'],n_exp])*np.nan 
            surf_prop_dic_spot[subkey_chrom]['Ftot_star']=np.zeros([system_prop[subkey_chrom]['nw'],n_exp])*np.nan 
        cb_band_dic[subkey_chrom]={}  
        if ('CB_RV' in par_list) or ('c0_CB' in par_list):     
            surf_prop_dic[subkey_chrom]['c0_CB']=np.zeros(system_prop[subkey_chrom]['nw'])*np.nan
            surf_prop_dic_spot[subkey_chrom]['c0_CB']=np.zeros(system_prop[subkey_chrom]['nw'])*np.nan
            for iband in range(system_prop[subkey_chrom]['nw']):
                cb_band_dic[subkey_chrom][iband] = calc_CB_RV(get_LD_coeff(system_prop[subkey_chrom],iband),system_prop[subkey_chrom]['LD'][iband],par_star['c1_CB'],par_star['c2_CB'],par_star['c3_CB'],par_star) 
                surf_prop_dic[subkey_chrom]['c0_CB'][iband]=cb_band_dic[subkey_chrom][iband][0] 
                surf_prop_dic_spot[subkey_chrom]['c0_CB'][iband]=cb_band_dic[subkey_chrom][iband][0]
        else:
            for iband in range(system_prop[subkey_chrom]['nw']):cb_band_dic[subkey_chrom][iband] = None
    if 'rv' in par_list_in:par_list+=['rv']  #must be placed after all other RV contributions

    #Define the list of parameters whose range we're interested in
    range_par_list=[]
    if (len(theo_dic['d_oversamp'])>0) and out_ranges:range_par_list = list(np.intersect1d(['mu','lat','lon','x_st','y_st','xp_abs','r_proj'],par_list))


    #Condition for spot calculation
    #Initializing the necessary dictionaries. We have to initialize them to something since they 
    #will be called later, even if spots are not activated.
    #Initialize the dictionary that will contain spot presence - has to be initialized here, in case where there are no spots
    cond_spots_all = np.zeros([n_exp,1], dtype=bool)

    #Define dictionary that will contain the spot properties for all the exposures
    spots_prop_all_exp = {}
    
    #Initialize this list since it's used elsewhere
    list_spot_names = []
    
    #Initializing list that will contain the oversampled step for spots, if they are oversampled.
    dx_exp_in_sp={}
    dy_exp_in_sp={}
    dz_exp_in_sp={}

    #Initialize a list which will tell us the oversampling rate for each exposure
    n_osamp_exp_all_sp = np.repeat(1,n_exp)

    # #If spots are being used, and they are in the param dictionary provided (which is not always the case)  
    if 'use_spots' in param.keys() and param['use_spots']:
        #Figure out the number of spots
        n_spots = param['num_spots']
        #Initialize the dictionary that will contain spot presence
        cond_spots_all = np.zeros([n_exp,n_spots], dtype=bool)

        #Looping over all exposures
        for isub_exp, iexp in enumerate(iexp_list):
            
            #Get the time in BJD of the exposure we're considering.
            exp_time = coord_pl_in['bjd'][iexp]

            #Extract spot properties for the exposure we're considering.
            spots_prop = new_retrieve_spots_prop_from_param(star_params, param, param['inst'], param['vis'], exp_time, coord_pl_in['t_dur'][iexp])

            #Check if at least one spot is visible.
            #To do so, we need a more precise estimate of the spot location.
            spot_within_grid_all=np.zeros(len(spots_prop.keys()), dtype=bool)

            #Go through the spots and see if they are *roughly* visible.
            for spot_index, spot in enumerate(spots_prop):
                if spots_prop[spot]['is_center_visible']:
                    #See if spot is *precisely* visible.
                    spot_within_grid, _ = calc_spotted_tiles(spots_prop[spot],
                                    theo_dic['x_st_sky'], theo_dic['y_st_sky'], theo_dic['z_st_sky'], theo_dic,
                                    star_params, True)
                    if spot_within_grid:
                        spot_within_grid_all[spot_index]=True

                    #Check if oversampling is turned on for this spot
                    if len(theo_dic['n_oversamp_spot'])>0 and theo_dic['n_oversamp_spot'][spot]>0:
                        dx_exp_in_sp[spot] = spots_prop[spot]['x_sky_exp_end'] - spots_prop[spot]['x_sky_exp_start']
                        dy_exp_in_sp[spot] = spots_prop[spot]['y_sky_exp_end'] - spots_prop[spot]['y_sky_exp_start']
                        dz_exp_in_sp[spot] = spots_prop[spot]['z_sky_exp_end'] - spots_prop[spot]['z_sky_exp_start']
                        n_osamp_exp_all_sp[isub_exp] = np.maximum(n_osamp_exp_all_sp[isub_exp], theo_dic['n_oversamp_spot'][spot])

                    #Spot-dependent properties - initialize dictionaries
                    for subkey_chrom in key_chrom:
                        surf_prop_dic_spot[subkey_chrom][spot]={}
                        for par_loc in par_list:
                            surf_prop_dic_spot[subkey_chrom][spot][par_loc]=np.zeros([system_spot_prop[subkey_chrom]['nw'],n_exp])*np.nan        
                        for par_loc in range_par_list:surf_prop_dic_spot[subkey_chrom][spot][par_loc+'_range']=np.zeros([system_spot_prop[subkey_chrom]['nw'],n_exp,2])*np.nan
                        if ('line_prof' in par_list_in) and (theo_dic['precision']=='low'):
                            surf_prop_dic_spot[subkey_chrom][spot]['rv_broad']=-1e100*np.ones([system_spot_prop[subkey_chrom]['nw'],n_exp])

            # Update cond_spots_all
            cond_spots_all[isub_exp]=spot_within_grid_all

            # Put an entry in the dictionary storing spot properties over all exposures
            spots_prop_all_exp[iexp] = spots_prop
     
        #Storing the 'names' of the spots for later use.
        list_spot_names = list(spots_prop.keys())

    n_osamp_exp_all = np.repeat(1,n_exp)
    lambda_rad_pl = {}
    dx_exp_in={}
    dy_exp_in={}
    cond_transit_all = np.zeros([n_exp,len(transit_pl)],dtype=bool)
    for ipl,pl_loc in enumerate(transit_pl):
        
        #Check for planet transit
        if np.sum(np.abs(coord_pl_in[pl_loc]['ecl'][iexp_list])!=1.)>0:
            cond_transit_all[:,ipl]|=(np.abs(coord_pl_in[pl_loc]['ecl'][iexp_list])!=1.)   

            #Obliquities for multiple planets
            #    - for now only defined for a single planet if fitted  
            #    - the nominal lambda has been overwritten in 'system_param[pl_loc]' if fitted
            lambda_rad_pl[pl_loc]=system_param[pl_loc]['lambda_rad']
            
            #Exposure distance (Rstar) 
            if len(theo_dic['d_oversamp'])>0:
                dx_exp_in[pl_loc]=abs(coord_pl_in[pl_loc]['end_pos'][0,iexp_list]-coord_pl_in[pl_loc]['st_pos'][0,iexp_list])
                dy_exp_in[pl_loc]=abs(coord_pl_in[pl_loc]['end_pos'][1,iexp_list]-coord_pl_in[pl_loc]['st_pos'][1,iexp_list])
             
                #Number of oversampling points for current exposure  
                #    - for each exposure we take the maximum oversampling all planets considered 
                if pl_loc in theo_dic['d_oversamp']:
                    d_exp_in = np.sqrt(dx_exp_in[pl_loc]**2 + dy_exp_in[pl_loc]**2)
                    n_osamp_exp_all=np.maximum(n_osamp_exp_all,npint(np.round(d_exp_in/theo_dic['d_oversamp'][pl_loc]))+1)
                    
            #Planet-dependent properties
            for subkey_chrom in key_chrom:
                surf_prop_dic[subkey_chrom][pl_loc]={}
                for par_loc in par_list:
                    surf_prop_dic[subkey_chrom][pl_loc][par_loc]=np.zeros([system_prop[subkey_chrom]['nw'],n_exp])*np.nan        
                for par_loc in range_par_list:surf_prop_dic[subkey_chrom][pl_loc][par_loc+'_range']=np.zeros([system_prop[subkey_chrom]['nw'],n_exp,2])*np.nan
                if ('line_prof' in par_list_in) and (theo_dic['precision']=='low'):
                    surf_prop_dic[subkey_chrom][pl_loc]['rv_broad']=-1e100*np.ones([system_prop[subkey_chrom]['nw'],n_exp])


    #Processing each exposure 
    cond_transit = np.sum(cond_transit_all,axis=1)>0
    cond_spots = np.sum(cond_spots_all, axis=1)>0

    cond_iexp_proc = cond_spots|cond_transit
    for i_in,(iexp,n_osamp_exp, n_osamp_exp_sp) in enumerate(zip(iexp_list,n_osamp_exp_all, n_osamp_exp_all_sp)):
        #Planets in exposure
        transit_pl_exp = np.array(transit_pl)[cond_transit_all[i_in]]
        #Spots in exposure 
        if spots_prop_all_exp != {}:
            spots_in_exp = np.array(list_spot_names)[cond_spots_all[i_in]]
        else:
            spots_in_exp = {}

        #Initialize averaged and range values
        Focc_star_pl={}
        if 'use_spots' in param.keys() and param['use_spots']:Focc_star_sp={}
        sum_prop_dic={}
        coord_reg_dic={}
        range_dic={}
        line_occ_HP={}
        for subkey_chrom in key_chrom:
            Focc_star_pl[subkey_chrom]=np.zeros(system_prop[subkey_chrom]['nw'],dtype=float)  
            if 'use_spots' in param.keys() and param['use_spots']:Focc_star_sp[subkey_chrom]=np.zeros(system_spot_prop[subkey_chrom]['nw'],dtype=float)            
            sum_prop_dic[subkey_chrom]={}
            coord_reg_dic[subkey_chrom]={}
            range_dic[subkey_chrom]={}
            line_occ_HP[subkey_chrom]={}

            #Initializing dictionary entries for planet
            for pl_loc in transit_pl_exp:
                sum_prop_dic[subkey_chrom][pl_loc]={}
                coord_reg_dic[subkey_chrom][pl_loc]={}
                range_dic[subkey_chrom][pl_loc]={}
                for par_loc in par_list:    
                    sum_prop_dic[subkey_chrom][pl_loc][par_loc]=np.zeros(system_prop[subkey_chrom]['nw'],dtype=float)
                    coord_reg_dic[subkey_chrom][pl_loc][par_loc]=np.zeros(system_prop[subkey_chrom]['nw'],dtype=float)
                    if par_loc in range_par_list:range_dic[subkey_chrom][pl_loc][par_loc+'_range']=np.tile([1e100,-1e100],[system_prop[subkey_chrom]['nw'],1])
                sum_prop_dic[subkey_chrom][pl_loc]['nocc']=0. 
                if ('line_prof' in par_list_in):
                    if (theo_dic['precision'] in ['low','medium']):
                        coord_reg_dic[subkey_chrom][pl_loc]['rv_broad']=np.zeros(system_prop[subkey_chrom]['nw'],dtype=float)
                    elif (theo_dic['precision']=='high'):
                        sum_prop_dic[subkey_chrom][pl_loc]['line_prof'] = np.zeros(args['ncen_bins'],dtype=float)
            
            #Initializing dictionary entries for spots
            for spot in spots_in_exp:
                sum_prop_dic[subkey_chrom][spot]={}
                coord_reg_dic[subkey_chrom][spot]={}
                range_dic[subkey_chrom][spot]={}
                for par_loc in par_list:    
                    sum_prop_dic[subkey_chrom][spot][par_loc]=np.zeros(system_spot_prop[subkey_chrom]['nw'],dtype=float)
                    coord_reg_dic[subkey_chrom][spot][par_loc]=np.zeros(system_spot_prop[subkey_chrom]['nw'],dtype=float)
                    if par_loc in range_par_list:range_dic[subkey_chrom][spot][par_loc+'_range']=np.tile([1e100,-1e100],[system_spot_prop[subkey_chrom]['nw'],1])
                sum_prop_dic[subkey_chrom][spot]['nocc']=0. 
                if ('line_prof' in par_list_in):
                    if (theo_dic['precision'] in ['low','medium']):
                        coord_reg_dic[subkey_chrom][spot]['rv_broad']=np.zeros(system_spot_prop[subkey_chrom]['nw'],dtype=float)
                    elif (theo_dic['precision']=='high'):
                        sum_prop_dic[subkey_chrom][spot]['line_prof'] = np.zeros(args['ncen_bins'],dtype=float)
                    
            #Line profile can be calculated over each stellar cell only in achromatic / closest-achromatic mode 
            if ('line_prof' in par_list_in):line_occ_HP[subkey_chrom] = np.repeat(theo_dic['precision'],system_prop[subkey_chrom]['nw'])
            else:line_occ_HP[subkey_chrom] = np.repeat('',system_prop[subkey_chrom]['nw'])  
        

        #Theoretical properties from regions occulted by each planet or spot, at exposure center    
        if cond_iexp_proc[i_in]:
            #Planet oversampled positions initialization        
            x_oversamp_pl={}
            y_oversamp_pl={}
        
            #Planet oversampled positions
            for pl_loc in transit_pl_exp:
            
                #No oversampling
                if n_osamp_exp==1:
                    x_oversamp_pl[pl_loc] = [coord_pl_in[pl_loc]['cen_pos'][0,iexp]]
                    y_oversamp_pl[pl_loc] = [coord_pl_in[pl_loc]['cen_pos'][1,iexp]]
    
                #Theoretical properties from regions occulted by each planet, averaged over full exposure duration  
                #    - only if oversampling is effective for this exposure
                else:
                    x_oversamp_pl[pl_loc] = coord_pl_in[pl_loc]['st_pos'][0][iexp]+np.arange(n_osamp_exp)*dx_exp_in[pl_loc][i_in]/(n_osamp_exp-1.)  
                    y_oversamp_pl[pl_loc] = coord_pl_in[pl_loc]['st_pos'][1][iexp]+np.arange(n_osamp_exp)*dy_exp_in[pl_loc][i_in]/(n_osamp_exp-1.) 
            
            #Spot oversampled positions initialization
            x_oversamp_sp={}
            y_oversamp_sp={}
            z_oversamp_sp={}

            #Spot oversampled positions
            for spot in spots_in_exp:
                #No oversampling
                if n_osamp_exp_sp==1:
                    x_oversamp_sp[spot] = [spots_prop_all_exp[iexp][spot]['x_sky_exp_center']]
                    y_oversamp_sp[spot] = [spots_prop_all_exp[iexp][spot]['y_sky_exp_center']]
                    z_oversamp_sp[spot] = [spots_prop_all_exp[iexp][spot]['z_sky_exp_center']]
                
                #If we want to oversample
                else:
                    x_oversamp_sp[spot] = spots_prop_all_exp[iexp][spot]['x_sky_exp_start'] + np.arange(n_osamp_exp_sp)*dx_exp_in_sp[spot]/(n_osamp_exp_sp-1.)  
                    y_oversamp_sp[spot] = spots_prop_all_exp[iexp][spot]['y_sky_exp_start'] + np.arange(n_osamp_exp_sp)*dy_exp_in_sp[spot]/(n_osamp_exp_sp-1.) 
                    z_oversamp_sp[spot] = spots_prop_all_exp[iexp][spot]['z_sky_exp_start'] + np.arange(n_osamp_exp_sp)*dz_exp_in_sp[spot]/(n_osamp_exp_sp-1.) 

            #Loop on oversampled exposure positions - planets
            #    - after x_oversamp_pl has been defined for all planets
            #    - if oversampling is not active a single central position is processed
            #    - we neglect the potential chromatic variations of the planet radius and corresponding grid 
            #    - if at least one of the processed planet is transiting
            n_osamp_exp_eff = 0
            for iosamp in range(n_osamp_exp):
                #Dictionary telling us which planets have been processed in which chromatic mode and band.
                pl_proc={subkey_chrom:{iband:[] for iband in range(system_prop[subkey_chrom]['nw'])} for subkey_chrom in key_chrom}

                #Initializing variable in case it is not defined
                cond_occ_pl = False
                for pl_loc in transit_pl_exp:   
                    
                    #Frame conversion of planet coordinates from the classical frame perpendicular to the LOS, to the 'inclined star' frame
                    x_st_sky_pos,y_st_sky_pos,_=frameconv_LOS_to_InclinedStar(lambda_rad_pl[pl_loc],x_oversamp_pl[pl_loc][iosamp],y_oversamp_pl[pl_loc][iosamp],None)      
    
                    #Largest possible square grid enclosing the planet shifted to current planet position     
                    x_st_sky_max = x_st_sky_pos+theo_dic['x_st_sky_grid_pl'][pl_loc]
                    y_st_sky_max = y_st_sky_pos+theo_dic['y_st_sky_grid_pl'][pl_loc]
                    
                    #Calculating properties
                    cond_occ_pl = False
                    for subkey_chrom in key_chrom:
                        for iband in range(system_prop[subkey_chrom]['nw']):
                            Focc_star_pl[subkey_chrom][iband],cond_occ_pl = calc_occ_region_prop(line_occ_HP[subkey_chrom][iband],cond_occ_pl,iband,args,system_prop[subkey_chrom],iosamp,pl_loc,pl_proc[subkey_chrom][iband],theo_dic['Ssub_Sstar_pl'][pl_loc],x_st_sky_max,y_st_sky_max,system_prop[subkey_chrom]['cond_in_RpRs'][pl_loc][iband],par_list,par_star,theo_dic['Istar_norm_'+subkey_chrom],\
                                                                                  x_oversamp_pl,y_oversamp_pl,lambda_rad_pl,par_star,sum_prop_dic[subkey_chrom][pl_loc],coord_reg_dic[subkey_chrom][pl_loc],range_dic[subkey_chrom][pl_loc],range_par_list,Focc_star_pl[subkey_chrom][iband],star_params,cb_band_dic[subkey_chrom][iband],theo_dic)
         
                            #Cumulate line profile from planet-occulted cells
                            #    - in high-precision mode there is a single subkey_chrom and achromatic band, but several planets may have been processed
                            if ('line_prof' in par_list_in):
                                if (theo_dic['precision']=='low'):surf_prop_dic[subkey_chrom][pl_loc]['rv_broad'][iband,i_in] = np.max([coord_reg_dic[subkey_chrom][pl_loc]['rv_broad'][iband],surf_prop_dic[subkey_chrom][pl_loc]['rv_broad'][iband,i_in]])
                                elif (theo_dic['precision']=='high'):surf_prop_dic[subkey_chrom]['line_prof'][:,i_in]+=sum_prop_dic[subkey_chrom][pl_loc]['line_prof']
                 
                #Star was effectively occulted at oversampled position
                if cond_occ_pl:
                    n_osamp_exp_eff+=1
                    
                    #Calculate line profile from planet-occulted region 
                    #    - profile is scaled to the total flux from current occulted region, stored in coord_reg_dic_pl['Ftot']
                    if ('line_prof' in par_list_in) and (theo_dic['precision']=='medium'):
                        idx_w = {'achrom':range(system_prop['achrom']['nw'])}
                        if ('chrom' in key_chrom):idx_w['chrom'] = range(system_prop['chrom']['nw'])
                        surf_prop_dic[key_chrom[-1]]['line_prof'][:,i_in]+=plocc_prof(args,transit_pl_exp,coord_reg_dic,idx_w,system_prop,key_chrom,par_star,theo_dic)
            
            n_osamp_exp_eff_sp = 0
            #Loop on oversampled exposure positions - spots
            for iosamp in range(n_osamp_exp_sp):

                cond_occ_sp = False

                #Retrieving the properties of the region occulted by each spot
                for spot in spots_in_exp:
                    
                    #Make a rough estimate of the spot-occulted grid - has a different resolution than the stellar grid - is in inclined star rest frame
                    x_st_sky_max_sp = x_oversamp_sp[spot][iosamp] + theo_dic['x_st_sky_grid_sp'][spot]
                    y_st_sky_max_sp = y_oversamp_sp[spot][iosamp] + theo_dic['y_st_sky_grid_sp'][spot]

                    #Need to make a new dictionary that contains the spot properties for this oversampled exposure
                    spot_prop_oversamp = {}
                    #Setting the properties in spot_prop_oversamp to those of the oversampled exposure
                    spot_prop_oversamp['atten'] = spots_prop_all_exp[iexp][spot]['atten']
                    spot_prop_oversamp['x_sky_grid'] = x_st_sky_max_sp
                    spot_prop_oversamp['y_sky_grid'] = y_st_sky_max_sp
                    spot_prop_oversamp['x_sky_exp_center'] = x_oversamp_sp[spot][iosamp]
                    spot_prop_oversamp['y_sky_exp_center'] = y_oversamp_sp[spot][iosamp]
                    spot_prop_oversamp['z_sky_exp_center'] = z_oversamp_sp[spot][iosamp]
                    spot_prop_oversamp['lat_rad_exp_center'] = spots_prop_all_exp[iexp][spot]['lat_rad_exp_center']
                    spot_prop_oversamp['ang_rad'] = spots_prop_all_exp[iexp][spot]['ang_rad']
                    spot_prop_oversamp['long_rad_exp_center'] = np.arcsin(x_oversamp_sp[spot][iosamp] / np.cos(spots_prop_all_exp[iexp][spot]['lat_rad_exp_center']))
                    spot_prop_oversamp['cos_long_exp_center'] = np.cos(spot_prop_oversamp['long_rad_exp_center'])
                    spot_prop_oversamp['sin_long_exp_center'] = np.sin(spot_prop_oversamp['long_rad_exp_center'])
                    spot_prop_oversamp['cos_lat_exp_center'] = np.cos(spot_prop_oversamp['lat_rad_exp_center'])
                    spot_prop_oversamp['sin_lat_exp_center'] = np.sin(spot_prop_oversamp['lat_rad_exp_center'])
                    spot_prop_oversamp['is_visible'] = is_spot_visible(star_params['istar_rad'], spot_prop_oversamp['long_rad_exp_center'], spot_prop_oversamp['lat_rad_exp_center'], spot_prop_oversamp['ang_rad'], star_params['f_GD'], star_params['RpoleReq'])

                    cond_occ_sp = False

                    #Going over the chromatic modes
                    for subkey_chrom in key_chrom:
                        
                        #Going over the bands in each chromatic mode
                        for iband in range(system_spot_prop[subkey_chrom]['nw']):
                            
                            Focc_star_sp[subkey_chrom][iband], cond_occ_sp = new_new_calc_spotted_region_prop(line_occ_HP[subkey_chrom][iband], cond_occ_sp, spot_prop_oversamp, iband, system_spot_prop[subkey_chrom], star_params, theo_dic['Ssub_Sstar_sp'][spot], 
                                                            theo_dic['Ssub_Sstar'], theo_dic['Istar_norm_'+subkey_chrom], par_star, sum_prop_dic[subkey_chrom][spot], coord_reg_dic[subkey_chrom][spot], 
                                                            range_dic[subkey_chrom][spot], Focc_star_sp[subkey_chrom][iband], par_list, range_par_list, args, cb_band_dic[subkey_chrom][iband], 
                                                            pl_loc_x = coord_pl_in[pl_loc]['cen_pos'][0,iexp], pl_loc_y = coord_pl_in[pl_loc]['cen_pos'][1,iexp], RpRs = system_prop[subkey_chrom][pl_loc][iband], plocc = (n_osamp_exp_eff>=1))

                            #Cumulate line profile from spot-occulted cells
                            #    - in high-precision mode there is a single subkey_chrom and achromatic band, but several spots may have been processed
                            if ('line_prof' in par_list_in):
                                if (theo_dic['precision']=='low'):surf_prop_dic_spot[subkey_chrom][spot]['rv_broad'][iband,i_in] = np.max([coord_reg_dic[subkey_chrom][spot]['rv_broad'][iband],surf_prop_dic_spot[subkey_chrom][spot]['rv_broad'][iband,i_in]])
                                elif (theo_dic['precision']=='high'):surf_prop_dic_spot[subkey_chrom]['line_prof'][:,i_in]+=sum_prop_dic[subkey_chrom][spot]['line_prof']
                            
                            emit_coord_reg_dic = deepcopy(coord_reg_dic)
                            emit_coord_reg_dic[subkey_chrom][spot]['Ftot'][iband] *= (1-spot_prop_oversamp['atten'])

                            emit_surf_prop_dic_spot = deepcopy(surf_prop_dic_spot)
                            emit_surf_prop_dic_spot[subkey_chrom][spot]['Ftot'][iband] *= (1-spot_prop_oversamp['atten'])

                #Star was effectively occulted at oversampled position
                if cond_occ_sp:
                    n_osamp_exp_eff_sp+=1
                    
                    #Calculate line profile from spot-occulted region 
                    #    - profile is scaled to the total flux from current occulted region, stored in coord_reg_dic_pl['Ftot']
                    if ('line_prof' in par_list_in) and (theo_dic['precision']=='medium'):
                        idx_w = {'achrom':range(system_spot_prop['achrom']['nw'])}
                        if ('chrom' in key_chrom):idx_w['chrom'] = range(system_spot_prop['chrom']['nw'])
                        
                        #Line profile of region occulted by the spot
                        surf_prop_dic_spot[key_chrom[-1]]['line_prof'][:,i_in]+=plocc_prof(args,spots_in_exp,coord_reg_dic,idx_w,system_spot_prop,key_chrom,par_star,theo_dic)
                        
                        #Line profile emitted by the spot
                        surf_prop_dic_spot[key_chrom[-1]]['line_prof'][:,i_in]-=plocc_prof(args,spots_in_exp,emit_coord_reg_dic,idx_w,system_spot_prop,key_chrom,par_star,theo_dic)


            #------------------------------------------------------------

            #Averaged values behind all occulted regions during exposure
            #    - with the oversampling, positions at the center of exposure will weigh more in the sum than those at start and end of exposure, like in reality
            #    - parameters are retrieved in both oversampled/not-oversampled case after they are updated within the sum_prop_dic dictionary 
            #    - undefined values remain set to nan, and are otherwise normalized by the flux from the planet-occulted region
            #    - we use a single Itot as condition that the planet occulted the star
            for pl_loc in transit_pl_exp:
                if (sum_prop_dic[key_chrom[0]][pl_loc]['Ftot'][0]>0.):
                    for subkey_chrom in key_chrom:
                        for par_loc in par_list:
                
                            #Total occulted surface ratio and flux from planet-occulted region
                            #    - calculated per band so that the chromatic dependence of the radius can be accounted for also at the stellar limbs, which we cannot do with the input chromatic RpRs
                            #    - averaged over oversampled regions
                            if par_loc in ['SpSstar','Ftot']:surf_prop_dic[subkey_chrom][pl_loc][par_loc][:,i_in] = sum_prop_dic[subkey_chrom][pl_loc][par_loc]/n_osamp_exp_eff

                            #Other surface properties
                            #    - defined as sum( oversampled region , sum( cell , xi*fi )  ) / sum( oversampled region , sum( cell , fi )  )
                            else:surf_prop_dic[subkey_chrom][pl_loc][par_loc][:,i_in] = sum_prop_dic[subkey_chrom][pl_loc][par_loc]/sum_prop_dic[subkey_chrom][pl_loc]['Ftot']
                                                           
                            #Range of values covered during exposures    
                            if out_ranges and (par_loc in range_par_list):
                                surf_prop_dic[subkey_chrom][pl_loc][par_loc+'_range'][:,i_in,:] = range_dic[subkey_chrom][pl_loc][par_loc+'_range']
            
            for spot in spots_in_exp:
               if (sum_prop_dic[key_chrom[0]][spot]['Ftot'][0]>0.):
                    for subkey_chrom in key_chrom:
                        for par_loc in par_list:

                           #Total occulted surface ratio and flux from spot-occulted region
                            #    - calculated per band so that the chromatic dependence of the radius can be accounted for also at the stellar limbs, which we cannot do with the input chromatic RpRs
                            #    - averaged over oversampled regions
                            if par_loc in ['SpSstar','Ftot']:surf_prop_dic_spot[subkey_chrom][spot][par_loc][:,i_in] = sum_prop_dic[subkey_chrom][spot][par_loc]/n_osamp_exp_eff_sp

                            #Other surface properties
                            #    - defined as sum( oversampled region , sum( cell , xi*fi )  ) / sum( oversampled region , sum( cell , fi )  )
                            else:surf_prop_dic_spot[subkey_chrom][spot][par_loc][:,i_in] = sum_prop_dic[subkey_chrom][spot][par_loc]/sum_prop_dic[subkey_chrom][spot]['Ftot']
                                                           
                            #Range of values covered during exposures    
                            if out_ranges and (par_loc in range_par_list):
                                surf_prop_dic_spot[subkey_chrom][spot][par_loc+'_range'][:,i_in,:] = range_dic[subkey_chrom][spot][par_loc+'_range']


            #Normalized stellar flux after occultation by all planets and by all spots
            #    - the intensity from each cell is calculated in the same way as that of the full pre-calculated stellar grid
            if Ftot_star:
                for subkey_chrom in key_chrom:

                    surf_prop_dic[subkey_chrom]['Ftot_star'][:,i_in] = 1.

                    #Planets
                    if n_osamp_exp_eff>0:
                        surf_prop_dic[subkey_chrom]['Ftot_star'][:,i_in] -= Focc_star_pl[subkey_chrom]/(n_osamp_exp_eff*theo_dic['Ftot_star_'+subkey_chrom])
                    #Spots
                    if n_osamp_exp_eff_sp>0:
                        surf_prop_dic_spot[subkey_chrom]['Ftot_star'][:,i_in] -= Focc_star_sp[subkey_chrom]/(n_osamp_exp_eff_sp*theo_dic['Ftot_star_'+subkey_chrom])

            #Planet-occulted line profile from current exposure
            if ('line_prof' in par_list_in) and (n_osamp_exp_eff>0):

                #Profile from averaged properties over exposures
                if (theo_dic['precision']=='low'): 
                    idx_w = {'achrom':(range(system_prop['achrom']['nw']),i_in)}
                    if ('chrom' in key_chrom):idx_w['chrom'] = (range(system_prop['chrom']['nw']),i_in)
                    surf_prop_dic[key_chrom[-1]]['line_prof'][:,i_in]=plocc_prof(args,transit_pl_exp,surf_prop_dic,idx_w,system_prop,key_chrom,par_star,theo_dic)
    
                #Averaged profiles behind all occulted regions during exposure   
                #    - the weighing by stellar intensity is naturally included when applying flux scaling 
                elif (theo_dic['precision'] in ['medium','high']):
                    surf_prop_dic[key_chrom[-1]]['line_prof'][:,i_in]/=n_osamp_exp_eff

                    #Normalization into intrinsic profile
                    #    - profiles used to tile the planet-occulted regions have mean unity, and are then scaled by the cell achromatic flux
                    #      we normalize by the total planet-occulted flux
                    #    - high-precision profils is achromatic
                    #    - not required for low- and medium-precision because intrinsic profiles are not scaled to local flux upon calculation in plocc_prof()
                    if (theo_dic['precision']=='high') and args['conv2intr']:
                        Focc_star_achrom=Focc_star_pl[key_chrom[-1]]/n_osamp_exp_eff
                        surf_prop_dic[key_chrom[-1]]['line_prof'][:,i_in] /=Focc_star_achrom


            #Spot-occulted line profile from current exposure
            if ('line_prof' in par_list_in) and (n_osamp_exp_eff_sp > 0):

                #Profile from averaged properties over exposures
                if (theo_dic['precision']=='low'): 
                    idx_w = {'achrom':(range(system_prop['achrom']['nw']),i_in)}
                    if ('chrom' in key_chrom):idx_w['chrom'] = (range(system_prop['chrom']['nw']),i_in)
                    surf_prop_dic_spot[key_chrom[-1]]['line_prof'][:,i_in]=(plocc_prof(args,spots_in_exp,surf_prop_dic_spot,idx_w,system_spot_prop,key_chrom,par_star,theo_dic) - plocc_prof(args,spots_in_exp,emit_surf_prop_dic_spot,idx_w,system_spot_prop,key_chrom,par_star,theo_dic))

                #Averaged profiles behind all occulted regions during exposure   
                #    - the weighing by stellar intensity is naturally included when applying flux scaling 
                elif (theo_dic['precision'] in ['medium','high']):
                    surf_prop_dic_spot[key_chrom[-1]]['line_prof'][:,i_in]/=n_osamp_exp_eff_sp

                    #Normalization into intrinsic profile
                    #    - profiles used to tile the planet-occulted regions have mean unity, and are then scaled by the cell achromatic flux
                    #      we normalize by the total planet-occulted flux
                    #    - high-precision profils is achromatic
                    #    - not required for low- and medium-precision because intrinsic profiles are not scaled to local flux upon calculation in plocc_prof()
                    if (theo_dic['precision']=='high') and args['conv2intr']:
                        Focc_star_achrom_sp = Focc_star_sp[key_chrom[-1]]/n_osamp_exp_eff_sp
                        surf_prop_dic_spot[key_chrom[-1]]['line_prof'][:,i_in] /=Focc_star_achrom_sp
   

    ### end of exposure            
 
    #Output properties in chromatic mode if calculated in closest-achromatic mode
    if ('line_prof' in par_list_in) and switch_chrom:
        surf_prop_dic = {'chrom':surf_prop_dic['achrom']}
        surf_prop_dic_spot = {'chrom':surf_prop_dic_spot['achrom']}

    if 'use_spots' in param.keys():
        return surf_prop_dic, surf_prop_dic_spot
    else:
        return surf_prop_dic



def calc_occ_region_prop(line_occ_HP_band,cond_occ,iband,args,system_prop,idx,pl_loc,pl_proc_band,Ssub_Sstar,x_st_sky_max,y_st_sky_max,cond_in_RpRs,par_list,param,Istar_norm_band,x_pos_pl,y_pos_pl,lambda_rad_pl,par_star,sum_prop_dic_pl,\
                         coord_reg_dic_pl,range_reg_pl,range_par_list,Focc_star_band,star_params,cb_band,theo_dic):
    r"""**Planet-occulted properties: region**

    Calculates the average and summed properties from a planet-occulted stellar surface region during an exposure.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
        
    #Reduce maximum square planet grid to size of planet in current band
    coord_grid = {}
    coord_grid['x_st_sky']=x_st_sky_max[cond_in_RpRs] 
    coord_grid['y_st_sky']=y_st_sky_max[cond_in_RpRs]   

    #Identifying occulted stellar cells in the sky-projected and star star rest frame
    n_pl_occ = calc_st_sky(coord_grid,star_params)
    
    #Star is effectively occulted
    #    - when the expositions are oversampled, some oversampled positions may put the planet beyond the stellar disk, with no points behind the star
    if n_pl_occ>0:
        cond_occ = True
        
        #Removing current planet cells already processed for previous occultations
        if len(pl_proc_band)>0:
            cond_pl_occ_corr = np.repeat(True,n_pl_occ)
            for pl_prev in pl_proc_band:
    
                #Coordinate of previous planet center in the 'inclined star' frame
                x_st_sky_prev,y_st_sky_prev,_=frameconv_LOS_to_InclinedStar(lambda_rad_pl[pl_prev],x_pos_pl[pl_prev][idx],y_pos_pl[pl_prev][idx],None)

                #Cells occulted by current planet and not previous ones
                #    - condition is that cells must be beyond previous planet grid in this band
                RpRs_prev = system_prop[pl_prev][iband]
                cond_pl_occ_corr &= ( (coord_grid['x_st_sky'] - x_st_sky_prev)**2.+(coord_grid['y_st_sky'] - y_st_sky_prev)**2. > RpRs_prev**2. )
            for key in coord_grid:coord_grid[key] = coord_grid[key][cond_pl_occ_corr]
            n_pl_occ = np.sum(cond_pl_occ_corr)
      
            #Store planet as processed in current band
            pl_proc_band+=[pl_loc]     

        #--------------------------------

        #Local flux grid over current planet-occulted region, in current band
        coord_grid['nsub_star'] = n_pl_occ
        _,_,mu_grid_star,Fsurf_grid_star,Ftot_star,_ = calc_Isurf_grid([iband],coord_grid['nsub_star'],system_prop,coord_grid,star_params,Ssub_Sstar,Istar_norm = Istar_norm_band,region = 'pl',Ssub_Sstar_ref = theo_dic['Ssub_Sstar'])
        coord_grid['mu'] = mu_grid_star[:,0]

        #Scale continuum level
        Fsurf_grid_star*=param['cont']
        Ftot_star*=param['cont']

        #Flux and number of cells occulted from all planets, cumulated over oversampled positions
        Focc_star_band+= Ftot_star[0]   
        sum_prop_dic_pl['nocc']+=coord_grid['nsub_star']
    
        #--------------------------------

        #Co-adding properties from current region to the cumulated values over oversampled planet positions 
        sum_region_prop(line_occ_HP_band,iband,args,system_prop,par_list,Fsurf_grid_star[:,0],coord_grid,Ssub_Sstar,cb_band,range_par_list,range_reg_pl,sum_prop_dic_pl,coord_reg_dic_pl,par_star,lambda_rad_pl[pl_loc],theo_dic,param)

    return Focc_star_band,cond_occ




def sum_region_prop(line_occ_HP_band,iband,args,system_prop,par_list,Fsurf_grid_band,coord_grid,Ssub_Sstar,cb_band,range_par_list,range_reg_pl,sum_prop_dic_pl,coord_reg_dic_pl,par_star,lambda_rad_pl_loc,theo_dic,param):
    r"""**Planet-occulted properties: calculations**

    Calculates the average and summed properties from a planet-occulted stellar surface region during an exposure.
    
    The flux emitted by a local element writes
    
    .. math::  
       dF[\nu] =  I[\nu](\cos{\theta})  dA \, \vec{N}(dA).\vec{N}(\mathrm{LOS})
    
    with :math:`dF[\nu]` emitted in the direction :math:`\vec{N}(\mathrm{LOS})` of the LOS, :math:`dA = R_\mathrm{\star}^2 \sin{\theta} d\theta d\phi` the spherical surface element 
    at the surface of the star, and :math:`\vec{N}(dA)` its normal. By definition :math:`\vec{N}(dA).\vec{N}(\mathrm{LOS}) = \cos{\theta} = \mu` so that
    
    .. math::     
       dF[\nu] =   I[\nu](\cos{\theta}) R_\mathrm{\star}^2 \sin{\theta} d\theta d\phi \cos{\theta}
       
    Which can also write, with :math:`dS = dx dy = dA \cos{\theta}` the projection of `dA` onto the plane of sky (where `XY` is the plane perpendicular to the LOS (`Z`))

    .. math::  
       dF[\nu] =   I[\nu](\cos{\theta}) dS[\theta]
    
    Here the fluxes are normalized by the stellar surface, ie :math:`dS[\theta] = d_\mathrm{grid}^2/(\pi R_\mathrm{\star}^2)`, since :math:`d_\mathrm{grid}` is defined in units of :math:`R_\mathrm{\star}`.
    The total flux emitted by the star in the direction of the LOS is then, with :math:`\mu = \cos(\theta)` and :math:`d\mu = -\sin{\theta} d\theta`
    
    .. math:: 
       F_\mathrm{\star}[\nu] &= ( \int_{\phi = 0}^{2 \pi} \int_{\theta = 0}^{\pi / 2} I[\nu](\cos{\theta}) \sin{\theta} d\theta d\phi \cos{\theta} )/\pi   \\
                             &= 2 \int_{\mu = 0}^{1} I[\nu](\mu) \mu d\mu    \\
                             &= 2 I_0[\nu] \int_{\mu = 0}^{1} \mathrm{LD}(\mu) \mu d\mu \\
                             &= 2 I_0[\nu] \mathrm{Int}_0
                
    If there is no limb-darkening then :math:`\mathrm{Int}_0 = 1/2` and :math:`F_\mathrm{\star}[\nu] = I_0[\nu]`      
    
    In the non-oversampled case
    
     - we add the values, even if each index is only updated once, so that the subroutine can be used directly with oversampling
     - all tables have been initialized to 0
    
    In the oversampled case
    
     - average values are co-added over regions oversampling the total region occulted during an exposure.    
       We update values cumulated during an exposure through every passage through the function.
     - the flux emitted by a surface element is
     
       .. math:: 
          dF(\mu) =   I[\nu](\mu) S_\mathrm{sub}/S_\mathrm{\star} = I[\nu](xy) S_\mathrm{sub}/S_\mathrm{\star}
       
       Here we must consider the flux that is emitted by a surface element during the time `T` it is occulted by the planet
       
       .. math:: 
          dF_\mathrm{occ}(xy) = dF(xy) T_\mathrm{occ}(xy)

       If we assume the planet velocity is constant during the exposure, and it has no latitudinal motion, then 

       .. math:: 
          dF_\mathrm{occ}(xy) = dF(xy) d_\mathrm{occ}(xy)/v_\mathrm{pl}

       Where :math:`d_\mathrm{occ}` is the distance between the planet centers from the first to the last moment it occults the surface element

       .. math:: 
          d_\mathrm{occ}(xy) = \sqrt{ R_\mathrm{p}^2 - y_\mathrm{grid}^2 } = d_\mathrm{occ}(y)
            
     - the weighted mean of a quantity `V` during an exposure would thus be (where :math:`xy_\mathrm{grid}` describes the stellar grid cells occulted during an exposure)

       .. math:: 
          <A> &= \sum_{xy_\mathrm{grid}} V(xy) dF_\mathrm{occ}(xy)  / \sum_{xy_\mathrm{grid}} dF_\mathrm{occ}(xy)    \\
          <A> &= \sum_{xy_\mathrm{grid}} V(xy) I[\nu](xy) S_\mathrm{sub}/S_\mathrm{\star} d_\mathrm{occ}(y)/v_\mathrm{pl} / \sum_{xy_\mathrm{grid}} I[\nu](xy) S_\mathrm{sub}/S_\mathrm{\star} d_\mathrm{occ}(y)/v_\mathrm{pl}    \\  
          <A> &= \sum_{xy_\mathrm{grid}} V(xy) I[\nu](xy) d_\mathrm{occ}(y) / \sum_{xy_\mathrm{grid}} I[\nu](xy) d_\mathrm{occ}(y)   
          
     - instead of discretizing the exact surface occulted by the planet during a full exposure, we place the planet
       and its grid at consecutive positions during the exposure. Between two consecutive positions separated by `d_\mathrm{exp}^\mathrm{oversamp}`,
       the planet spent a time :math:`t_\mathrm{exp}^\mathrm{oversamp} = d_\mathrm{exp}^\mathrm{oversamp}/v_\mathrm{pl}`. If a surface element is occulted by the planet during `N(xy)` consecutive positions, then
       we can write  the total occultation time as 

       .. math:: 
          T_\mathrm{occ}(xy) &= N(xy) t_\mathrm{exp}^\mathrm{oversamp}     \\
          dF_\mathrm{occ}(xy) &= dF(xy) N(xy) d_\mathrm{exp}^\mathrm{oversamp}/v_\mathrm{pl}
           
       the weighted mean of a quantity `V` during an exposure is then

       .. math::        
          <A> &= \sum_{xy_\mathrm{grid}} V(xy) dF_\mathrm{occ}(xy) / \sum_{xy_\mathrm{grid}} dF_\mathrm{occ}(xy)            \\
          <A> &= \sum_{xy_\mathrm{grid}} V(xy) (dF(xy) N(xy) d_\mathrm{exp}^\mathrm{oversamp}/v_\mathrm{pl}) / \sum_{xy_\mathrm{grid}} (dF(xy) N(xy) d_\mathrm{exp}^\mathrm{oversamp}/v_\mathrm{pl})   \\  
          <A> &= \sum_{xy_\mathrm{grid}} V(xy) dF(xy) N(xy) / \sum_{xy_\mathrm{grid}} dF(xy) N(xy)   
            
       ie that we 'add' successively the `N(xy)` times a given surface element flux was occulted. 
       The normalization factor corresponds to :math:`F_\mathrm{tot}^\mathrm{oversamp}`.
       To ensure that this approximation is good, `N(xy)` must be high enough, ie :math:`t_\mathrm{exp}^\mathrm{oversamp}` and :math:`d_\mathrm{exp}^\mathrm{oversamp}` small enough 
     
     - note that :math:`S_\mathrm{sub}/S_\mathrm{\star}` is normalized by :math:`R_\mathrm{\star}^2`, since :math:`d_\mathrm{grid}` is defined from :math:`R_\mathrm{p}/R_\mathrm{\star}`

    Args:
        TBD
    
    Returns:
        TBD
    
    """     
    #Distance from projected orbital normal in the sky plane, in absolute value
    if ('xp_abs' in par_list) or (('coord_line' in args) and (args['coord_line']=='xp_abs')):coord_grid['xp_abs'] = frameconv_InclinedStar_to_LOS(lambda_rad_pl_loc,coord_grid['x_st_sky'],coord_grid['y_st_sky'],coord_grid['z_st_sky'])[0]  

    #Sky-projected distance from star center
    if ('r_proj' in par_list) or (('coord_line' in args) and (args['coord_line']=='r_proj')):coord_grid['r_proj'] = np.sqrt(coord_grid['r2_st_sky'])                   

    #Processing requested properties
    for par_loc in par_list:
     
        #Occultation ratio
        #    - ratio = Sp/S* = sum(x,sp(x))/(pi*Rstar^2) = sum(x,dp(x)^2)/(pi*Rstar^2) = sum(x,d_surfloc(x)*Rstar^2)/(pi*Rstar^2) = sum(x,d_surfloc(x))/pi
        #      since we use a square grid, sum(x,d_surfloc(x)) = nx*d_surfloc
        #      this quantity is not limb-darkening weighted
        if par_loc=='SpSstar':
            sum_prop_dic_pl[par_loc][iband]+=Ssub_Sstar*coord_grid['nsub_star']
            coord_reg_dic_pl[par_loc][iband] = Ssub_Sstar*coord_grid['nsub_star']
         
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
            elif par_loc=='Rot_RV':coord_grid[par_loc] = calc_RVrot(coord_grid['x_st_sky'],coord_grid['y_st'],par_star['istar_rad'],par_star)[0]
                         
            #Disk-integrated-corrected convective blueshift polynomial (km/s)
            elif par_loc=='CB_RV':coord_grid[par_loc] = np_poly(cb_band)(coord_grid['mu'])          
    
            #Full RV (km/s)
            #    - accounting for an additional constant offset to model jitter or global shifts, and for visit-specific offset to model shifts specific to a given transition
            elif par_loc=='rv':
                coord_grid[par_loc] = deepcopy(coord_grid['Rot_RV']) + param['rv']
                if 'CB_RV' in par_list:coord_grid[par_loc]+=coord_grid['CB_RV']
                if 'rv_line' in par_list:coord_grid[par_loc]+=coord_grid['rv_line']
                
            #------------------------------------------------
    
            #Sum property over occulted region, weighted by stellar flux
            #    - we use flux rather than intensity, because local flux level depend on the planet grid resolution
            #    - total RVs from planet-occulted region is set last in par_list to calculate all rv contributions first:
            # + rotational contribution is always included
            # + disk-integrated-corrected convective blueshift polynomial (in km/s)   
            coord_grid[par_loc+'_sum'] = np.sum(coord_grid[par_loc]*Fsurf_grid_band)
            if par_loc=='xp_abs':coord_grid[par_loc+'_sum'] = np.abs(coord_grid[par_loc+'_sum'])
              
            #Cumulate property over successively occulted regions
            sum_prop_dic_pl[par_loc][iband]+=coord_grid[par_loc+'_sum'] 

            #Total flux from current occulted region
            if par_loc=='Ftot':coord_reg_dic_pl['Ftot'][iband] = coord_grid['Ftot_sum']

            #Calculate average property over current occulted region  
            #    - <X> = sum(cell, xcell*fcell)/sum(cell,fcell)           
            else:coord_reg_dic_pl[par_loc][iband] = coord_grid[par_loc+'_sum']/coord_grid['Ftot_sum'] 

            #Range of values covered during the exposures (normalized)
            #    - for spatial-related coordinates
            if par_loc in range_par_list:
                range_reg_pl[par_loc+'_range'][iband][0]=np.min([range_reg_pl[par_loc+'_range'][iband][0],coord_reg_dic_pl[par_loc][iband]])
                range_reg_pl[par_loc+'_range'][iband][1]=np.max([range_reg_pl[par_loc+'_range'][iband][1],coord_reg_dic_pl[par_loc][iband]])
     
    #------------------------------------------------    
    #Calculate line profile from average of cell profiles over current region
    #    - this high precision mode is only possible for achromatic or closest-achromatic mode 
    if line_occ_HP_band=='high':    
        
        #Attribute intrinsic profile to each cell 
        init_st_intr_prof(args,coord_grid,param)

        #Calculate individual local line profiles from all region cells
        #    - analytical intrinsic profiles are fully calculated 
        #      theoretical and measured intrinsic profiles have been pre-defined and are just shifted to their position
        #    - in both cases a scaling is then applied to convert them into local profiles
        line_prof_grid=coadd_loc_line_prof(coord_grid['rv'],range(coord_grid['nsub_star']),Fsurf_grid_band,args['flux_intr_grid'],coord_grid['mu'],param,args)          
      
        #Coadd line profiles over planet-occulted region
        sum_prop_dic_pl['line_prof'] = np.sum(line_prof_grid,axis=0) 
  
    #Define rotational broadening of planet-occulted region
    elif line_occ_HP_band in ['low','medium']:
        drv_min = coord_reg_dic_pl['rv'][iband]-np.min(coord_grid['rv'])
        drv_max = np.max(coord_grid['rv'])-coord_reg_dic_pl['rv'][iband] 
        coord_reg_dic_pl['rv_broad'][iband] = 0.5*(drv_min+drv_max)       

    return None






def plocc_prof(args,transit_pl,coord_dic,idx_w,system_prop,key_chrom,param,theo_dic):
    r"""**Planet-occulted line profile**

    Line profiles can be 
    
     - theoretical (calculated with a stellar atmospheric model)
     - measured (from the input data)
     - analytical (calculated in RV or wavelength space, but over a single line).
    
    Spectral dependence
    
     - chromatic mode: planet-to-star radius ratio and/or stellar intensy are chromatic (spectral mode only)
     - closest-achromatic mode: profile width is smaller than the typical scale of chromatic variations (spectral mode only)
     - achromatic mode: white values are used (default in CCF mode, optional in spectral mode)
      
    Profiles can be calculated at three precision levels
    
     - high. Intrinsic spectra are summed over each occulted cells. This option is only possible in chromatic / closest-achromatic mode (intrinsic spectra cannot be summed over individual cells for each 
       chromatic band, since different parts of the spectra are affected differently)
     - medium. Intrinsic profiles are defined for each occulted region, based on the average region properties, and cumulated.
       In chromatic mode each profile is respectively scaled and shifted using the chromatic RV and flux scaling table from the region 
     - low. Intrinsic profiles are defined for each exposure, based on the average exposure properties.
       In chromatic mode each profile is respectively scaled and shifted using the chromatic RV and flux scaling table from the exposure 
                     
    When several planets are transiting the properties are averaged over the complementary regions they occult, in particular the flux scaling, so that the final profile cumulated over all planets 
    should have the flux corresponding to the summed planet-to-star surface ratios 

    Args:
        TBD
    
    Returns:
        TBD
    
    """     
    data_av_pl={'cen_bins':np.array([args['cen_bins']]),'edge_bins':np.array([args['edge_bins']])}
    
    #In chromatic mode 'key_chrom' contains both chrom and achrom, while in other modes it only contains achrom
    if ('chrom' in key_chrom):chrom_calc = 'chrom'
    else:chrom_calc = 'achrom'
    
    #Profile calculation
    line_prof = np.zeros(args['ncen_bins'],dtype=float)
    for pl_loc in transit_pl:  
        
        #Planet transits
        if ~np.isnan(coord_dic[chrom_calc][pl_loc]['Ftot'][idx_w[chrom_calc]]):
            mu_av = coord_dic['achrom'][pl_loc]['mu'][idx_w['achrom']]
    
            #Chromatic scaling based on current planet occultation and stellar intensity
            #    - single value in achromatic mode
            #    - not applied for the calculation of intrinsic profiles, as chromatic broadband variations are corrected for when extracting measured intrinsic profiles
            if args['conv2intr']:flux_sc_spec = 1.
            else:
                flux_sc = coord_dic[chrom_calc][pl_loc]['Ftot'][idx_w[chrom_calc]]
                if chrom_calc=='achrom':flux_sc_spec = flux_sc[0]
                else:flux_sc_spec = np_interp(args['cen_bins'],system_prop['chrom']['w'],flux_sc,left=flux_sc[0],right=flux_sc[-1])
         
            #Analytical profile
            #    - only in achromatic / closest-chromatic mode (no chromatic shift is applied)
            if (args['mode']=='ana'):   
            
                #Surface properties with polynomial variation of chosen coordinate, required to calculate spectral line profile
                args['input_cell_all']={}
                for par_loc in args['linevar_par_vis']:     
                    args['input_cell_all'][par_loc] = coord_dic['achrom'][pl_loc][par_loc][idx_w['achrom']]  
                
                #Calculation of planet-occulted line profile
                data_av_pl['flux'] = [calc_loc_line_prof(0,coord_dic['achrom'][pl_loc]['rv'][idx_w['achrom']],flux_sc_spec,None,mu_av,args,param)]
                
            #Theoretical or measured profile
            elif (args['mode'] in ['theo','Intrbin']):     
                data_loc = {}
                
                #Retrieve measured intrinsic profile at average coordinate for current planet
                if args['mode']=='Intrbin':
                    coordline_av = coord_dic['achrom'][pl_loc][args['coord_line']][idx_w['achrom']]
                    ibin_to_cell = closest(args['cen_dim_Intrbin'],coordline_av)
                    Fintr_av_pl=args['flux_Intrbin'][ibin_to_cell]   
                    data_loc['cen_bins'] = np.array([args['cen_bins_Intrbin']])
                    data_loc['edge_bins'] = np.array([args['edge_bins_Intrbin']])
              
                #Interpolate theoretical profile at average coordinate for current planet
                elif args['mode']=='theo':     
                    Fintr_av_pl = theo_dic['sme_grid']['flux_intr_grid'](mu_av[0])                    
                    data_loc['cen_bins'] = np.array([args['cen_bins_intr']])
                    data_loc['edge_bins'] = np.array([args['edge_bins_intr']])

                #Scaling from intrinsic to planet-occulted flux  
                data_loc['flux'] = np.array([Fintr_av_pl*flux_sc_spec])
                ncen_bins_Intr = len(data_loc['cen_bins'][0])
            
                #Shift profile from average occulted photopshere position for current planet (source, in which it is defined) to star rest frame (receiver) 
                #    - 'conv2intr' is True when the routine is used to calculate model intrinsic profiles (flux_sc_spec is then 1) to be compared with measured intrinsic profiles
                #       since the latter were corrected for chromatic deviations upon extraction, the present model profiles are only shifted with the achromatic planet-occulted rv 
                #    - otherwise a chromatic shift is applied
                if args['conv2intr']:dic_rv = {'achrom':{pl_loc:{'rv': np.reshape(coord_dic['achrom'][pl_loc]['rv'][idx_w['achrom']],[system_prop['achrom']['nw'],1]) }}}
                else:dic_rv = {chrom_calc:{pl_loc:{'rv': np.reshape(coord_dic[chrom_calc][pl_loc]['rv'][idx_w[chrom_calc]],[system_prop[chrom_calc]['nw'],1]) }}}
                rv_surf_star,rv_surf_star_edge = def_surf_shift('theo',dic_rv,0,data_loc,pl_loc,args['type'],system_prop,[1,ncen_bins_Intr],1,ncen_bins_Intr)  
                if rv_surf_star_edge is not None:
                    rv_shift_edge = -rv_surf_star_edge
                    spec_dopshift_edge = gen_specdopshift(rv_surf_star_edge)
                else:
                    rv_shift_edge = None
                    spec_dopshift_edge = None
                data_av_pl=align_data(data_loc,args['type'],1,args['dim_exp'],args['resamp_mode'],data_av_pl['cen_bins'],data_av_pl['edge_bins'],-rv_surf_star,gen_specdopshift(rv_surf_star),rv_shift_edge = rv_shift_edge,spec_dopshift_edge = spec_dopshift_edge, nocov = True)

            #Rotational broadening
            #    - for a planet large enough, the distribution of surface RV over the occulted region acts produces rotational broadening
            if coord_dic[chrom_calc][pl_loc]['rv_broad'][idx_w[chrom_calc]][0]>0.:
                FWHM_broad = coord_dic[chrom_calc][pl_loc]['rv_broad'][idx_w[chrom_calc]][0]
                
                #Convert into spectral broadening (in A)                    
                if ('spec' in args['type']):FWHM_broad*=args['ref_conv']/c_light
                    
                #Rotational kernel
                data_av_pl['flux'][0] = convol_prof(data_av_pl['flux'][0],args['cen_bins'],FWHM_broad)  

            #Co-add contribution from current planet
            line_prof+=data_av_pl['flux'][0]
            
    return line_prof








def occ_region_grid(RpRs,nsub_Dpl):
    r"""**Planet grid**

    Defines grid discretizing planet disk, in the 'inclined' star frame
      
     - X axis is parallel to the star equator
     - Y axis is the projected spin axis
     - Z axis is the LOS

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    
    #Subcell width (in units of Rstar) and surface (in units of Rstar^2 and pi*Rstar^2) 
    d_sub=2.*RpRs/nsub_Dpl
    Ssub_Sstar=d_sub*d_sub/np.pi

    #Coordinates of points discretizing the enclosing square
    cen_sub=-RpRs+(np.arange(nsub_Dpl)+0.5)*d_sub            
    xy_st_sky_grid=np.array(list(it_product(cen_sub,cen_sub)))

    #Distance to planet center (squared)
    r_sub_pl2=xy_st_sky_grid[:,0]*xy_st_sky_grid[:,0]+xy_st_sky_grid[:,1]*xy_st_sky_grid[:,1]

    #Keeping only grid points behind the planet
    cond_in_pldisk = ( r_sub_pl2 < RpRs*RpRs)           
    x_st_sky_grid=xy_st_sky_grid[cond_in_pldisk,0]
    y_st_sky_grid=xy_st_sky_grid[cond_in_pldisk,1] 
    r_sub_pl2=r_sub_pl2[cond_in_pldisk] 

    return d_sub,Ssub_Sstar,x_st_sky_grid,y_st_sky_grid,r_sub_pl2



def init_surf_shift(gen_dic,inst,vis,data_dic,align_mode):
    r"""**Planet-occulted rv**

    Returns measured or theoretical planet-occulted rv

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Set surface RV to measured achromatic value  
    #    - only in-transit profiles profiles for which the local stellar line was flagged as detected can be aligned  
    if (align_mode=='meas'):
        dic_rv = dataload_npz(gen_dic['save_data_dir']+'Introrig_prop/'+inst+'_'+vis)
        idx_aligned = np_where1D(dic_rv['cond_detected'])
        ref_pl=None

    #Set surface RVs to model
    #    - all in-transit profiles can be aligned, as rv can be calculated at any phase
    elif (align_mode=='theo'):        
        dic_rv = dataload_npz(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis)
        idx_aligned = np.arange(data_dic[inst][vis]['n_in_tr'])   

        #Reference planet
        #    - theoretical velocities are calculated for all planets transiting in a given visit
        if len(data_dic[inst][vis]['transit_pl'])==1:ref_pl = data_dic[inst][vis]['transit_pl'][0]  
        else:ref_pl=data_dic['Intr']['align_ref_pl'][inst][vis]
    
    return ref_pl,dic_rv,idx_aligned


def def_surf_shift(align_mode,dic_rv,i_in,data_exp,pl_ref,data_type,system_prop,dim_exp,nord,nspec):    
    r"""**Planet-occulted rv shifts**

    Returns rv shifts of stellar surface in star rest frame, from measured or theoretical planet-occulted rv

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    #Set surface RV to measured achromatic value 
    if (align_mode=='meas'):
        rv_surf_star=dic_rv[i_in]['rv']
        rv_surf_star_edge=None
    
    #Set surface RVs to model
    elif ('theo' in align_mode):

        #Chromatic RVs
        #    - theoretical RVs calculated for the broadband RpRs and intensity values provided as inputs are interpolated over the table of each exposure, so that 
        # each bin is shifted with the surface rv corresponding to its wavelength   
        #    - if data has been converted from spectra to CCF, nominal properties will be used
        if ('spec' in data_type) and ('chrom' in dic_rv):
            rv_surf_star=np.zeros(dim_exp,dtype=float)*np.nan
            rv_surf_star_edge=np.zeros([nord,nspec+1],dtype=float)*np.nan
            
            #Absolute or chromatic-to-nominal relative RV
            if align_mode=='theo':RV_shift_pl = dic_rv['chrom'][pl_ref]['rv'][:,i_in]
            elif align_mode=='theo_rel':RV_shift_pl = dic_rv['chrom'][pl_ref]['rv'][:,i_in]-dic_rv['achrom'][pl_ref]['rv'][0,i_in]              
            for iord in range(nord):
                rv_surf_star[iord] = np_interp(data_exp['cen_bins'][iord],system_prop['chrom']['w'],RV_shift_pl,left=RV_shift_pl[0],right=RV_shift_pl[-1])
                rv_surf_star_edge[iord] = np_interp(data_exp['edge_bins'][iord],system_prop['chrom']['w'],RV_shift_pl,left=RV_shift_pl[0],right=RV_shift_pl[-1])
            
        #Achromatic RV defined for the nominal transit properties
        else:
            rv_surf_star = dic_rv['achrom'][pl_ref]['rv'][0,i_in]  
            rv_surf_star_edge=None

    return rv_surf_star,rv_surf_star_edge



