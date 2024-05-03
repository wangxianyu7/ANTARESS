#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from lmfit import Parameters
from copy import deepcopy
import bindensity as bind
from astropy.io import fits
from scipy import special
import scipy.linalg
from ..ANTARESS_conversions.ANTARESS_conv import new_compute_CCF,check_CCF_mask_lines
from ..ANTARESS_analysis.ANTARESS_inst_resp import convol_prof,calc_FWHM_inst,return_resolv
from ..ANTARESS_general.utils import stop,np_where1D,npint,dataload_npz,MAIN_multithread,air_index,gen_specdopshift,def_edge_tab,check_data
from ..ANTARESS_general.constant_data import N_avo,c_light_m,k_boltz,h_planck
from ..ANTARESS_general.minim_routines import call_lmfit




def corr_tell(gen_dic,data_inst,inst,data_dic,data_prop,coord_dic,plot_dic):
    r"""**Main telluric correction routine.**
    
    Initializes and runs the selected telluric correction.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    print('   > Correcting spectra for tellurics')
    
    #Calculating data
    if (gen_dic['calc_corr_tell']):
        print('         Calculating data')   
        
        #Static model directories
        if inst in ['NIRPS_HE','NIRPS_HA']:inst_dir = 'NIRPS'
        else: inst_dir=inst

        #Automatic telluric correction
        if gen_dic['calc_tell_mode']=='autom':
            fixed_args={'tell_thresh_corr':gen_dic['tell_thresh_corr'],'inst':inst}
            
            #HITRAN properties
            static_model_path = 'ANTARESS_corrections/Telluric_processing/Static_model/'
            tell_mol_dic={}
            for key in ['range_mol_prop','fit_range_mol_prop','lines_fit_molecules','qt_molec','temp_offset_molecules','iwv_value_molecules','temp_bool_molecules' ]:tell_mol_dic[key] = {}
            for molec in gen_dic['tell_species']:

                #Full line list for considered species 
                static_file_full_range = fits.open(static_model_path+inst_dir+'/Static_hitran_qt_'+molec+'.fits')
       
                #Full molecules properties
                tell_mol_dic['range_mol_prop'][molec]  = static_file_full_range[1].data
                
                #Total partition sum value at temperature T
                tell_mol_dic['qt_molec'][molec] = static_file_full_range[2].data

                #Strong selection line list for fitting
                static_file_lines_fit  = fits.open(static_model_path+inst_dir+'/Static_hitran_strongest_lines_'+molec+'.fits')

                #Molecules line list and properties                
                tell_mol_dic['fit_range_mol_prop'][molec] = static_file_lines_fit[1].data     
                tell_mol_dic['lines_fit_molecules'][molec]   = static_file_lines_fit[2].data 

            #Molecule properties
            tell_mol_dic['Nx_molec']={ # [molec*cm^-3]
                'H2O':3.3427274952610645e+22,       
                'CH4':4.573e+13,
                'CO2':9.8185e+15,
                'O2':5.649e+18}     
            tell_mol_dic['M_mol_molec']={# [g*mol^-1]
                'H2O':18.01528,
                'CH4':16.04,
                'CO2':44.01,
                'O2':31.9988 }
            if inst in ['HARPS','SOPHIE','ESPRESSO','NIRPS_HA','NIRPS_HE','HARPSN','CARMENES_VIS']:   #Fix temperature if it is known from in-situ measurements
                tell_mol_dic['temp_bool_molecules']={
                    'H2O':False,
                    'CH4':False,                  
                    'CO2':False,
                    'O2':False}
            else:
                tell_mol_dic['temp_bool_molecules']={
                    'H2O':True,
                    'CH4':True,                  
                    'CO2':True,
                    'O2':True}
            tell_mol_dic['temp_offset_molecules']={
                'H2O':0.,
                'CH4':20.,
                'CO2':20.,
                'O2':20.}
                    
            tell_mol_dic['iwv_value_molecules']={# [cm]
                'H2O':None,
                'CH4':2163379.,
                'CO2':951782.,
                'O2':660128.}

            #Telluric CCF grid
            drv = gen_dic['pix_size_v'][inst]
            if gen_dic['tell_def_range'] is None:min_def_ccf,max_def_ccf = -40.,40.001
            else:min_def_ccf,max_def_ccf = gen_dic['tell_def_range'][0],gen_dic['tell_def_range'][1]
            fixed_args['velccf']= np.arange(min_def_ccf,max_def_ccf,gen_dic['pix_size_v'][inst])
            fixed_args['edge_velccf'] = np.append(fixed_args['velccf']-0.5*drv,fixed_args['velccf'][-1]+0.5*drv)
            fixed_args['n_ccf'] = len(fixed_args['velccf'])     
            
            #Continuum of model CCF
            fixed_args['CCFcont_deg']={}
            for molec in gen_dic['tell_species']:    
                if (inst in gen_dic['tell_CCFcont_deg']) and (molec in gen_dic['tell_CCFcont_deg'][inst]):fixed_args['CCFcont_deg'][molec] = gen_dic['tell_CCFcont_deg'][inst][molec]
                else:fixed_args['CCFcont_deg'][molec] = 0
        
            #Function arguments
            fixed_args['c2']= 1e2*c_light_m*h_planck/k_boltz  #in [cm*K]
            fixed_args['R_mod']= 3000000 # model resolution,
            fixed_args['resamp_mode']=gen_dic['resamp_mode']
            fixed_args['use_cov']=False 
            fixed_args['use_cov_eff'] = gen_dic['use_cov']

            #Mock table for chi2 calculation
            fixed_args['y_val'] = np.zeros(fixed_args['n_ccf'],dtype=float)
            fixed_args['cov_val'] = np.array([np.ones(fixed_args['n_ccf'],dtype=float)**2.])  

            #Telluric CCF minimization model
            fixed_args['fit_func'] = FIT_CCF_telluric_model
            fixed_args['inside_fit'] = False 
            
        #Process each visit independently
        iord_list_ref = range(data_inst['nord_ref'])
        for ivisit,vis in enumerate(data_inst['visit_list']):
            data_vis=data_inst[vis]
            data_prop_vis = data_prop[inst][vis]

            #Exposures to be corrected
            if (inst in gen_dic['tell_exp_corr']) and (vis in gen_dic['tell_exp_corr'][inst]) and (len(gen_dic['tell_exp_corr'][inst][vis])>0):
                iexp_corr_list = gen_dic['tell_exp_corr'][inst][vis]
            else:iexp_corr_list = np.arange(data_vis['n_in_visit'])

            #Orders to be corrected
            if (inst in gen_dic['tell_ord_corr']) and (vis in gen_dic['tell_ord_corr'][inst]) and (len(gen_dic['tell_ord_corr'][inst][vis])>0):
                iord_corr_list = np.intersect1d(iord_list_ref,gen_dic['tell_ord_corr'][inst][vis]) 
            else:iord_corr_list = iord_list_ref 

            #Automatic correction
            if gen_dic['calc_tell_mode']=='autom':
                
                #Resolution map or constant resolution for instrumental response
                if (data_inst['type']=='spec2D') and (inst in ['ESPRESSO','NIRPS_HE','NIRPS_HA']):
                    print('         Using resolution map for instrumental response')
                    fixed_args['fixed_res'] = False
                    mean_bjd = np.mean(coord_dic[inst][vis]['bjd'])
                    fixed_args['resolution_map'] = open_resolution_map(inst,mean_bjd,data_prop[inst][vis]['ins_mod'],data_prop[inst][vis]['det_binx'])
                else:
                    print('         Using constant resolution for instrumental response ')
                    fixed_args['fixed_res'] = True
                    fixed_args['resolution_map'] = calc_FWHM_inst

                #Overwrite fit properties
                if (inst in gen_dic['tell_mod_prop']) and (vis in gen_dic['tell_mod_prop'][inst]):fixed_args['tell_mod_prop'] = gen_dic['tell_mod_prop'][inst][vis]
                else:fixed_args['tell_mod_prop']={}

                #Orders to be fitted
                if (inst in gen_dic['tell_ord_fit']) and (vis in gen_dic['tell_ord_fit'][inst]) and (len(gen_dic['tell_ord_fit'][inst][vis])>0):
                    fixed_args['iord_fit_list'] = np.intersect1d(iord_list_ref,gen_dic['tell_ord_fit'][inst][vis]) 
                else:fixed_args['iord_fit_list'] = iord_list_ref

                #Continuum and fitted RV range for telluric CCFs 
                if (inst in gen_dic['tell_fit_range']) and (vis in gen_dic['tell_fit_range'][inst]):fixed_args['tell_fit_range'] = gen_dic['tell_fit_range'][inst][vis]
                else:fixed_args['tell_fit_range'] = {}
                if (inst in gen_dic['tell_cont_range']) and (vis in gen_dic['tell_cont_range'][inst]):fixed_args['tell_cont_range'] = gen_dic['tell_cont_range'][inst][vis]
                else:fixed_args['tell_cont_range'] = {}

            #Processing all exposures    
            proc_DI_data_paths_new = gen_dic['save_data_dir']+'Corr_data/Tell/'+inst+'_'+vis+'_'
            iexp_all = range(data_vis['n_in_visit'])
            common_args = (data_vis['proc_DI_data_paths'],iexp_corr_list,inst,vis,gen_dic['tell_range_corr'],iord_corr_list,gen_dic['calc_tell_mode'],fixed_args,gen_dic['tell_species'],tell_mol_dic,gen_dic['sp_frame'],gen_dic['resamp_mode'],data_inst[vis]['dim_exp'],plot_dic['tell_CCF'],plot_dic['tell_prop'],proc_DI_data_paths_new,data_inst[vis]['mean_gdet_DI_data_paths'])
            if gen_dic['tell_nthreads']>1:MAIN_multithread(corr_tell_vis,gen_dic['tell_nthreads'],data_vis['n_in_visit'],[iexp_all,data_prop_vis['AM'],data_prop_vis['IWV_AM'],data_prop_vis['TEMP'],data_prop_vis['PRESS'],data_prop_vis['BERV']],common_args)                           
            else:corr_tell_vis(iexp_all,data_prop_vis['AM'],data_prop_vis['IWV_AM'],data_prop_vis['TEMP'],data_prop_vis['PRESS'],data_prop_vis['BERV'],*common_args)  
            data_vis['proc_DI_data_paths'] = proc_DI_data_paths_new
            data_vis['tell_DI_data_paths'] = {}
            for iexp in range(data_vis['n_in_visit']):data_vis['tell_DI_data_paths'][iexp] = proc_DI_data_paths_new+'tell_'+str(iexp)

    #Updating path to processed data and checking it has been calculated
    else:
        for vis in data_inst['visit_list']:  
            data_vis=data_inst[vis]
            data_vis['proc_DI_data_paths']=gen_dic['save_data_dir']+'Corr_data/Tell/'+inst+'_'+vis+'_'  
            check_data({'path':data_vis['proc_DI_data_paths']+str(0)},vis=vis) 
            data_vis['tell_DI_data_paths'] = {}
            for iexp in range(data_vis['n_in_visit']):data_vis['tell_DI_data_paths'][iexp] = data_vis['proc_DI_data_paths']+'tell_'+str(iexp)
            
    return None   


def corr_tell_vis(iexp_group,airmass_group,IWV_airmass_group,temp_group,press_group,BERV_group,proc_DI_data_paths,iexp_corr_list,inst,vis,tell_range_corr,iord_corr_list,calc_tell_mode,fixed_args,tell_species,tell_mol_dic,sp_frame,resamp_mode,dim_exp,tell_CCF,tell_prop,proc_DI_data_paths_new,mean_gdet_DI_data_paths):
    """**Telluric correction per visit.**

    Applies the chosen telluric correction in a given visit.

    Args:
        TBD
    
    Returns:
        None
    
    """
    for iexp,airmass_exp,IWV_airmass_exp,temp_exp,press_exp,BERV_exp in zip(iexp_group,airmass_group,IWV_airmass_group,temp_group,press_group,BERV_group):
        data_exp = dataload_npz(proc_DI_data_paths+str(iexp))
        if iexp in iexp_corr_list:
            edge_bins = deepcopy(data_exp['edge_bins'])
            cen_bins = deepcopy(data_exp['cen_bins'])
            
            #Retrieve calibration profile
            #    - defined in the same frame and over the same table as the exposure spectrum
            mean_gdet_exp = dataload_npz(mean_gdet_DI_data_paths[iexp])['mean_gdet'] 

            #Define correction range
            cond_corr = data_exp['cond_def']
            if (inst in tell_range_corr) and (vis in tell_range_corr[inst]) and (len(tell_range_corr[inst][vis])>0):
                cond_range=False
                low_wav_exp = edge_bins[:,0:-1]
                high_wav_exp = edge_bins[:,1::] 
                for bd_int in tell_range_corr:cond_range |= (low_wav_exp>=bd_int[0]) & (high_wav_exp<=bd_int[1])  
                cond_corr &= cond_range                
                iord_corr_exp = np.intersect(iord_corr_list,np_where1D( np.sum( cond_corr,axis=1 )>0 ))  
            else:iord_corr_exp = iord_corr_list
            
            #Generate telluric spectrum 
            #    - defined over the exposure spectral table, in the telluric rest frame and vacuum
            #      since the pixels of the shifted spectral table still match the original exposure ones, no conversion needs to be applied to the final telluric spectrum 
            if calc_tell_mode=='autom': 
                tell_exp,data_exp['flux'],data_exp['cov'],outputs = Run_ATC(airmass_exp,IWV_airmass_exp,temp_exp,press_exp,BERV_exp,edge_bins,cen_bins,data_exp['flux'],data_exp['cov'],data_exp['cond_def'],mean_gdet_exp,inst,tell_species,tell_mol_dic,sp_frame,resamp_mode,dim_exp,iord_corr_exp,tell_CCF,tell_prop,fixed_args)
        
                #Save correction data for plotting purposes
                dic_sav = {}  
                if (tell_CCF!='') or (tell_prop!=''):
                    dic_sav.update(outputs)
                np.savez_compressed(proc_DI_data_paths_new+str(iexp)+'_add',data=dic_sav,allow_pickle=True)
        
            #Applying input correction
            #    - input telluric spectrum is defined over input table
            else:    
                for iord in iord_corr_exp:
                    data_exp['flux'][iord],data_exp['cov'][iord] = bind.mul_array(data_exp['flux'][iord],data_exp['cov'][iord],1./tell_exp[iord])
    
            #Defined pixels
            data_exp['cond_def'] = ~np.isnan(data_exp['flux'])
    
        #No telluric correction
        else:tell_exp = np.ones(dim_exp,dtype=float)
    
        #Saving corrected data and associated tables
        np.savez_compressed(proc_DI_data_paths_new+str(iexp),data = data_exp,allow_pickle=True) 
        np.savez_compressed(proc_DI_data_paths_new+'tell_'+str(iexp), data = {'tell':tell_exp},allow_pickle=True) 

    return None


def Run_ATC(airmass_exp,IWV_airmass_exp,temp_exp,press_exp,BERV_exp,edge_bins,cen_bins,flux_exp,cov_exp,cond_def_exp,mean_gdet_exp,inst,tell_species,tell_mol_dic,sp_frame,resamp_mode,dim_exp,iord_corr_list,tell_CCF,tell_prop,fixed_args):
    """**ATC function.**
    
    Adaptation of the Automatic Telluric Correction routine by R. Allart.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    fixed_args['iord_corr_list']=iord_corr_list

    #Spectral conversion 
    #    - all telluric spectral tables are assumed to be defined in vacuum
    #      if data are defined air, they are temporarily shifted to vacuum
    if sp_frame=='air': 
        edge_bins*=air_index(edge_bins, t=15., p=760.)
        cen_bins*=air_index(cen_bins, t=15., p=760.)

    #Align spectra back from the solar barycentric (receiver) to the telluric (source) rest frame 
    #    - see gen_specdopshift():
    # w_source = w_receiver / (1+ (rv[s/r]/c))
    # w_Earth = w_solbar / (1+ (rv[Earth/solbar]/c))
    # w_Earth = w_solbar / (1+ (BERV/c))
    sc_shift = 1./(gen_specdopshift(BERV_exp)*(1.+1.55e-8))
    fixed_args['edge_bins_earth'] = edge_bins*sc_shift
    fixed_args['cen_bins_earth'] = cen_bins*sc_shift   

    #Data    
    fixed_args['flux'] =  flux_exp
    fixed_args['cov'] =  cov_exp
    fixed_args['cond_def'] = cond_def_exp
    fixed_args['mean_gdet'] =  mean_gdet_exp
    fixed_args['ccf_corr'] = False
    
    #Store for plotting
    outputs={}
    if tell_CCF!='':
        outputs['velccf'] = fixed_args['velccf']

    #Fitting each telluric species
    param_molecules={}
    for molec in tell_species:
        outputs[molec] = {}
        
        #Fit properties
        params = Parameters()

        #Temperature (K)
        if np.isnan(temp_exp):temp_guess = 280.
        else:temp_guess = temp_exp-tell_mol_dic['temp_offset_molecules'][molec]
        params.add('Temperature',     value= temp_guess,         min=150., max=313.,    vary=tell_mol_dic['temp_bool_molecules'][molec] )
        
        #Guess for pressure and integrated species vapour toward zenith
        if molec=='H2O':
            if np.isnan(IWV_airmass_exp):IWV_zenith_exp = 0.1
            else:IWV_zenith_exp = IWV_airmass_exp/ 10.0 # /10. to put in cm
            if np.isnan(press_exp):
                press_guess = 0.5
                press_max = 10.
            else:
                press_guess = press_exp/2.
                press_max = 10.*press_exp

        #Other molecules
        else:
            IWV_zenith_exp = tell_mol_dic['iwv_value_molecules'][molec]               
            if np.isnan(press_exp):
                press_guess = 0.5
                press_max = 10.
            else:
                press_guess = press_exp/4.
                press_max = 10.*press_exp
        
        #Integrated species vapour along the LOS
        if IWV_zenith_exp==0.:IWV_zenith_exp = 0.1
        params.add('IWV_LOS',   value= IWV_zenith_exp*airmass_exp,  min=0., vary=True  )            
        
        #Pressure
        #    - this parameter represents an average pressure over the layers occupied by the species
        #    - pressure should always be smaller than the pressure measured at the facility at ground level, but for safety we take twice the value
        if press_guess==0.:press_guess = 0.5
        params.add('Pressure_ground', value= press_guess,     min=0., max=press_max,  vary=True  )
            
        #Overwrite fit properties
        if molec in fixed_args['tell_mod_prop']:
            for par in params:
                if par in fixed_args['tell_mod_prop'][molec]:
                    if ('vary' in fixed_args['tell_mod_prop'][molec][par]):params[par].vary = fixed_args['tell_mod_prop'][molec][par]['vary']
                    if ('value' in fixed_args['tell_mod_prop'][molec][par]):params[par].value = fixed_args['tell_mod_prop'][molec][par]['value']
                    if ('min' in fixed_args['tell_mod_prop'][molec][par]):params[par].min = fixed_args['tell_mod_prop'][molec][par]['min']
                    if ('max' in fixed_args['tell_mod_prop'][molec][par]):params[par].max = fixed_args['tell_mod_prop'][molec][par]['max']
                
        #Molecules properties
        fixed_args['M_mol_molec'] = {molec:tell_mol_dic['M_mol_molec'][molec]}
        fixed_args['molec'] = molec
        fixed_args['range_mol_prop'] = {molec:tell_mol_dic['fit_range_mol_prop'][molec]}
        fixed_args['CCF_lines'] = tell_mol_dic['lines_fit_molecules'][molec]['CCF_lines_position_wavelength']
        idx_CCF_lines = np.intersect1d(fixed_args['CCF_lines'],1e8/tell_mol_dic['fit_range_mol_prop'][molec]['wave_number'],return_indices=True)[2]
        idx_CCF_lines = idx_CCF_lines[np.argsort(idx_CCF_lines)]
        fixed_args['CCF_deltas'] = tell_mol_dic['fit_range_mol_prop'][molec]['delta'][idx_CCF_lines]       
        fixed_args['qt_molec'] = {molec:tell_mol_dic['qt_molec'][molec]}
        fixed_args['Nx_molec'] = {molec:tell_mol_dic['Nx_molec'][molec]}
    
        #Continuum and fit range
        if (molec in fixed_args['tell_cont_range']):fixed_args['mol_cont_range'] = fixed_args['tell_cont_range'][molec]
        else:fixed_args['mol_cont_range'] = [-15.,15.]
        if (molec in fixed_args['tell_fit_range']) and (len(fixed_args['tell_fit_range'][molec])>0):cond_def_fit = np_where1D((fixed_args['edge_velccf'][0:-1]>fixed_args['tell_fit_range'][molec][0]) & (fixed_args['edge_velccf'][1::]<=fixed_args['tell_fit_range'][molec][1]))
        else:cond_def_fit = np.arange(fixed_args['n_ccf'],dtype=int)
        idx_def_fit = np_where1D(cond_def_fit)
        fixed_args['idx_mod'] = range(idx_def_fit[0],idx_def_fit[-1]+1)
        fixed_args['idx_sub_fit'] = np_where1D(cond_def_fit[fixed_args['idx_mod']])
 
        #Continuum of CCF   
        if fixed_args['CCFcont_deg'][molec]>0:
            for ideg_CCFcont in range(1,fixed_args['CCFcont_deg'][molec]+1):
                params.add('cont'+str(ideg_CCFcont), value= 0.,  vary=True  )

        #Fit minimization for current molecule, through comparison between telluric CCF from data and model
        #    - unfitted pixels are removed from the chi2 table passed to residual() , so that they are then summed over the full tables
        fixed_args['idx_fit'] = np.ones(len(fixed_args['idx_mod']),dtype=bool)   
        param_molecules[molec] = call_lmfit(params,fixed_args['velccf'][fixed_args['idx_mod']],fixed_args['y_val'][fixed_args['idx_mod']],np.array([fixed_args['cov_val'][0,fixed_args['idx_mod']]]),fixed_args['fit_func'],verbose=False,fixed_args=fixed_args)[2]
                
        #Store results for plotting
        if (tell_prop!=''):
            if (tell_prop!=''):outputs[molec] = {'p_best':param_molecules[molec]}

    #Calculate best telluric spectrum over the full exposure 2D spectrum
    #    - defined on the exposure table, shifted into the Earth rest frame and in vacuum
    tell_spec_exp,corr_flux_exp,corr_cov_exp = full_telluric_model(flux_exp,cov_exp,fixed_args,param_molecules,tell_species,tell_mol_dic,tell_mol_dic['range_mol_prop'],tell_mol_dic['qt_molec'],tell_mol_dic['M_mol_molec'],tell_mol_dic['Nx_molec'],fixed_args['resolution_map'],inst,resamp_mode,dim_exp)

    #Compute telluric CCF for plotting purposes  
    if (tell_CCF!=''):
        fixed_args['ccf_corr'] = True
        fixed_args['flux_corr'] = corr_flux_exp
        for molec in tell_species:       
            fixed_args['molec'] = molec
            fixed_args['M_mol_molec'] = {molec:tell_mol_dic['M_mol_molec'][molec]}
            fixed_args['qt_molec'] = {molec:tell_mol_dic['qt_molec'][molec]}
            fixed_args['Nx_molec'] = {molec:tell_mol_dic['Nx_molec'][molec]}
            fixed_args['idx_mod'] = np.arange(fixed_args['n_ccf']+1,dtype=int)
            
            #CCF on strong lines used for the fit     
            fixed_args['range_mol_prop'] = {molec:tell_mol_dic['fit_range_mol_prop'][molec]}
            fixed_args['CCF_lines'] = tell_mol_dic['lines_fit_molecules'][molec]['CCF_lines_position_wavelength']
            idx_CCF_lines = np.intersect1d(fixed_args['CCF_lines'],1e8/tell_mol_dic['fit_range_mol_prop'][molec]['wave_number'],return_indices=True)[2]
            idx_CCF_lines = idx_CCF_lines[np.argsort(idx_CCF_lines)]
            fixed_args['CCF_deltas'] = tell_mol_dic['fit_range_mol_prop'][molec]['delta'][idx_CCF_lines] 
            outputs[molec]['ccf_uncorr_master'],_,outputs[molec]['ccf_model_conv_master'],outputs[molec]['ccf_corr_master']=CCF_telluric_model(param_molecules[molec],fixed_args['velccf'],args=fixed_args)
            
            #CCF on full line list
            fixed_args['range_mol_prop'] = {molec:tell_mol_dic['range_mol_prop'][molec]}
            fixed_args['CCF_lines'] = 10**8./fixed_args['range_mol_prop'][molec]['wave_number']  
            fixed_args['CCF_deltas'] = tell_mol_dic['range_mol_prop'][molec]['delta'] 
            outputs[molec]['ccf_uncorr_full'],_,outputs[molec]['ccf_model_conv_full'],outputs[molec]['ccf_corr_full']=CCF_telluric_model(param_molecules[molec],fixed_args['velccf'],args=fixed_args)

    return tell_spec_exp,corr_flux_exp,corr_cov_exp,outputs

def voigt_em(x, HWHM=0.5, gamma=1., center=0.):
    """**Emission Voigt.**

    Returns Voigt profile in emission.

    Args:
        x (array): Spectral table over which Voigt is computed.
        HWHM (float): HWHM of the Voigt.
        gamma (float): Damping coefficient of the Voigt.   
        center (float): Central value of the Voigt over x.
    
    Returns:
        V (array): Voigt profile.
    
    """
    sigma = HWHM / np.sqrt(2.*np.log(2))
    z = (x - center + 1j * gamma) / (sigma * np.sqrt(2.))
    V = special.wofz(z).real / (np.sqrt(2. * np.pi) * sigma)
    return V


def open_resolution_map(instrument,time_science,ins_mod,bin_x):
    """**Retrieve static resolution map.**
    
    The map depends on the instrumental mode of the science frame and on the epoch where it was acquired ==> technical intervention
    
    Adapted from the ATC code (author: R. Allart)
    
    Args:
        TBD:
    
    Returns:
        resolution_map (2D array): Resolution map per order and pixel.
    
    """
    static_resol_path = 'ANTARESS_corrections/Telluric_processing/Static_resolution/'
    if instrument =='ESPRESSO':
        if ins_mod == 'SINGLEUHR':
            if time_science < 58421.5:
                instrumental_function = fits.open(static_resol_path+'ESPRESSO/r.ESPRE.2018-09-09T12:18:49.369_resolution_map.fits')
                #period = 'UHR_1x1_pre_october_2018'
            elif (time_science > 58421.5) and (time_science < 58653.1):
                instrumental_function = fits.open(static_resol_path+'ESPRESSO/r.ESPRE.2019-04-06T11:52:27.477_resolution_map.fits')
                #period = 'UHR_1x1_post_october_2018'
            elif time_science > 58653.1:
                instrumental_function = fits.open(static_resol_path+'ESPRESSO/r.ESPRE.2019-11-06T11:06:36.913_resolution_map.fits')
                #period = 'UHR_1x1_post_june_2019'
            
        elif (ins_mod == 'MULTIMR') and (bin_x == 4):
            if time_science < 58421.5:
                instrumental_function = fits.open(static_resol_path+'ESPRESSO/r.ESPRE.2018-07-08T14:51:53.873_resolution_map.fits')
                #period = 'MR_4x2_pre_october_2018'
            elif (time_science > 58421.5) and (time_science < 58653.1):
                instrumental_function = fits.open(static_resol_path+'r.ESPRE.2018-12-01T11:25:27.377_resolution_map.fits')
                #period = 'MR_4x2_post_october_2018'
            elif time_science > 58653.1:
                instrumental_function = fits.open(static_resol_path+'ESPRESSO/r.ESPRE.2019-11-05T12:23:45.139_resolution_map.fits')
                #period = 'MR_4x2_post_june_2019'
        
        elif (ins_mod == 'MULTIMR') and (bin_x == 8):
            if time_science < 58421.5:
                instrumental_function = fits.open(static_resol_path+'ESPRESSO/r.ESPRE.2018-07-06T11:48:22.862_resolution_map.fits')
                #period = 'MR_8x4_pre_october_2018'
            elif (time_science > 58421.5) and (time_science < 58653.1):
                instrumental_function = fits.open(static_resol_path+'ESPRESSO/r.ESPRE.2018-10-24T14:16:52.394_resolution_map.fits')
                #period = 'MR_8x4_post_october_2018'
            elif time_science > 58653.1:
                instrumental_function = fits.open(static_resol_path+'ESPRESSO/r.ESPRE.2019-11-05T12:59:57.138_resolution_map.fits')
                #period = 'MR_8x4_post_june_2019'
        
        elif (ins_mod == 'SINGLEHR') and (bin_x == 1):
            if time_science < 58421.5:
                instrumental_function = fits.open(static_resol_path+'ESPRESSO/r.ESPRE.2018-07-04T21:28:53.759_resolution_map.fits')
                #period = 'HR_1x1_pre_october_2018'
            elif (time_science > 58421.5) and (time_science < 58653.1):
                instrumental_function = fits.open(static_resol_path+'ESPRESSO/r.ESPRE.2019-01-05T19:53:55.501_resolution_map.fits')
                #period = 'HR_1x1_post_october_2018'
            elif time_science > 58653.1:
                instrumental_function = fits.open(static_resol_path+'ESPRESSO/r.ESPRE.2019-11-19T10:10:12.384_resolution_map.fits')
                #period = 'HR_1x1_post_june_2019'
        
        elif (ins_mod == 'SINGLEHR') and (bin_x == 2):
            if time_science < 58421.5:
                instrumental_function = fits.open(static_resol_path+'ESPRESSO/r.ESPRE.2018-09-05T14:01:58.063_resolution_map.fits')
                #period = 'HR_2x1_pre_october_2018'
            elif (time_science > 58421.5) and (time_science < 58653.1):
                instrumental_function = fits.open(static_resol_path+'ESPRESSO/r.ESPRE.2019-03-30T21:28:32.060_resolution_map.fits')
                #period = 'HR_2x1_post_october_2018'
            elif time_science > 58653.1:
                instrumental_function = fits.open(static_resol_path+'ESPRESSO/r.ESPRE.2019-09-26T11:06:01.271_resolution_map.fits')
                #period = 'HR_2x1_post_june_2019'
        
    elif instrument in ['NIRPS_HE','NIRPS_HA']:
        if ins_mod == 'HA':
            if time_science < 59850.5:
                instrumental_function = fits.open(static_resol_path+'NIRPS/r.NIRPS.2022-06-15T18_09_53.175_resolution_map.fits')
            elif (time_science > 59850.5):
                instrumental_function = fits.open(static_resol_path+'NIRPS/r.NIRPS.2022-11-28T21_06_04.212_resolution_map.fits')
        elif ins_mod == 'HE':
            if time_science < 59850.5:
                instrumental_function = fits.open(static_resol_path+'NIRPS/r.NIRPS.2022-06-15T17_52_36.533_resolution_map.fits')
            elif (time_science > 59850.5):
                instrumental_function = fits.open(static_resol_path+'NIRPS/r.NIRPS.2022-11-28T20_47_51.815_resolution_map.fits')         

    #Resolution map
    #    - defined as a function of detector pixels
    resolution_map = instrumental_function[1].data
    nord,npix = resolution_map.shape
    for iord in range(nord):
        cond_undef = np.isnan(resolution_map[iord])
        if True in cond_undef:
            if np.sum(cond_undef)==npix:resolution_map[iord,:] = return_resolv(instrument)
            else:
                idx_def = np_where1D(~cond_undef)
                idx_def_min = idx_def[0]
                if idx_def_min>0:resolution_map[iord,0:idx_def_min-1]=resolution_map[iord,idx_def_min]
                idx_def_max = idx_def[-1]
                if idx_def_max<npix-1:resolution_map[iord,idx_def_max+1::]=resolution_map[iord,idx_def_max]                    
    
    return resolution_map
   

def init_tell_molec(tell_species,params_molec,range_mol_prop,qt_molec,M_mol_molec,c2):
    """**Molecule initialization.**

    Initializes molecule line properties.
    
    Adapted from the ATC code (author: R. Allart)
    
    Args:
        TBD:
    
    Returns:
        TBD:
    
    """    

    #Processing requested molecules
    hwhm_scaled_dic = {'lor':{},'gauss':{}}
    intensity_scaled_dic = {}
    nu_scaled_dic = {}
    for molec in tell_species:
        temp        = params_molec[molec]['Temperature'].value
        P_0         = params_molec[molec]['Pressure_ground'].value
        mol_prop = range_mol_prop[molec]
        
        #Lines wavenumber [cm-1]
        nu_rest_mod   = mol_prop['wave_number']  

        #Pressure shift correction of lines position
        #    - P_0 in atm (1 atm = 101325 Pa)
        #    - delta = air-pressure shift, in cm-1 atm-1
        #    - nu in [cm-1]
        delta_air_mod = mol_prop['delta']
        nu_scaled_dic[molec] = nu_rest_mod + delta_air_mod * P_0

        #Total partition sum value at temperature T
        cond_line_temp = qt_molec[molec]['Temperature']==round(temp)
        qt_used = qt_molec[molec]['Qt'][cond_line_temp][0]

        #Lines intensity [cm−1/(molecule · cm−2)]
        Epp_mol = mol_prop['Epp']
        intensity_scaled_dic[molec] = mol_prop['Intensity'] * (174.5813 / qt_used) * (np.exp(- c2 * Epp_mol / temp)) / (np.exp(- c2 * Epp_mol / 296.0)) * (1. - np.exp(- c2 * nu_rest_mod / temp)) / (1. - np.exp(- c2 * nu_rest_mod / 296.0))

        #Lines HWHM
        n_air_mod = mol_prop['N']
        hwhm_scaled_dic['lor'][molec] = (296.0 / temp)**n_air_mod * mol_prop['gamma_air'] * P_0  # assuming P_mol=0                
        if molec !='H2O':
            hwhm_scaled_dic['gauss'][molec] = nu_rest_mod / c_light_m * np.sqrt(2. * N_avo * k_boltz * temp * np.log(2.) / (10 ** -3 * M_mol_molec[molec]))
        
    return hwhm_scaled_dic,intensity_scaled_dic,nu_scaled_dic




def calc_tell_model(tell_species,range_mol_prop,nu_sel_min,nu_sel_max,intensity_scaled_dic,nu_scaled_dic,params,Nx_molec,hwhm_scaled_dic,nu_mod,nbins_mod,edge_mod):
    """**Telluric model.**

    Calculates model telluric spectrum. 
    
    Adapted from the ATC code (author: R. Allart)
    
    Args:
        TBD:
    
    Returns:
        telluric_spectrum (1D array): Model telluric spectrum.
    
    """  
    
    #Processing requested molecules
    tell_op_mod = np.zeros(nbins_mod,dtype=float)
    for molec in tell_species:

        #Selects lines in a broader range than the spectrum range
        #    - we keep lines over the full model range so that their wings can contribute over the spectrum range
        cond_sel =   (nu_scaled_dic[molec] > nu_sel_min ) &  (nu_scaled_dic[molec] < nu_sel_max)
        if np.sum(cond_sel)>0:
            nu_scaled_mod_ord       = nu_scaled_dic[molec][cond_sel]
      
            #Opacity matrix
            #    - model spectral table has dimension nx
            #      line tables have dimension nl
            #    - calculating an opacity matrix with dimension nl x nx is much longer than looping over the lines with reduced ranges
            hwhm_lor_scaled_mod_ord = hwhm_scaled_dic['lor'][molec][cond_sel]
            if molec!='H2O':hwhm_gauss_scaled_mod_ord   = hwhm_scaled_dic['gauss'][molec][cond_sel]
            intensity_scaled_mod_ord = intensity_scaled_dic[molec][cond_sel]
            for iline in range(np.sum(cond_sel)):                
            
                #Pixels covered by current line profile
                wav_scaled_mod_ord_line = 1e8/nu_scaled_mod_ord[iline]
                nu_min_line = 10**8/(wav_scaled_mod_ord_line+8.*edge_mod)
                nu_max_line = 10**8/(wav_scaled_mod_ord_line-8.*edge_mod)
                idx_bins_li = np.arange(np.searchsorted(nu_mod,nu_min_line),np.searchsorted(nu_mod,nu_max_line+1))
                nu_mod_i = nu_mod[idx_bins_li]
                if molec == 'H2O': #lorentzian profile
                    line_profile_i = (1. / np.pi) * hwhm_lor_scaled_mod_ord[iline] / ( hwhm_lor_scaled_mod_ord[iline]**2. + ( nu_mod_i - nu_scaled_mod_ord[iline] )**2. )
                else:
                    line_profile_i = voigt_em(nu_mod_i, HWHM=hwhm_gauss_scaled_mod_ord[iline], gamma=hwhm_lor_scaled_mod_ord[iline], center=nu_scaled_mod_ord[iline])
                
                #Intensity profile
                intensity_i = line_profile_i * intensity_scaled_mod_ord[iline] 
          
                #Co-addition to optical depth spectrum
                #    - the column density is defined as IWV_LOS * Nx_molec, where Nx_molec is the species number density (molec cm^-3) in Earth atmosphere
                tell_op_mod[idx_bins_li]+=intensity_i*params[molec]['IWV_LOS'] * Nx_molec[molec]

    #end of molecules

    #Model telluric spectrum 
    #    - reordered as a function of wavelength
    telluric_spectrum = np.exp( - tell_op_mod)[::-1]   

    return telluric_spectrum


def var_convol_tell_sp_slow(telluric_spectrum,nbins,nbins_mod,cen_bins,cen_bins_mod,resolution_map_ord,dwav_mod    ,nedge_mod):
    """**Convolution of telluric spectrum (slow approach).**

    Adapted from the ATC code (author: R. Allart)
    
    Args:
        TBD:
    
    Returns:
        telluric_spectrum_conv (1D array): Convolved telluric spectrum.
    
    """      
    telluric_spectrum_conv = deepcopy(telluric_spectrum)

    #Index of observed wavelength closest to each selected model wavelength
    index_pixel = np.searchsorted(cen_bins,cen_bins_mod)
    index_pixel[index_pixel==-1]=0
    index_pixel[index_pixel==nbins]=nbins-1

    #Measured instrumental response (dw = w/R thus FWHM = w/R) at each model wavelength
    fwhm_angstrom_pixel = cen_bins_mod/resolution_map_ord[index_pixel]  
    
    #Convolution kernel half-width
    #    - a range of 3.15 x FWHM ( = 3.15*2*sqrt(2*ln(2)) sigma = 7.42 sigma ) contains 99.98% of a Gaussian LSF integral
    #      we conservatively use a kernel covering 4.25 x FWHM / dbin pixels, ie 2.125 FWHM or 5 sigma on each side  
    hnkern = npint(np.ceil(2.125*fwhm_angstrom_pixel/dwav_mod))  

    #Loop on model pixels with telluric absorption between step_convolution and n-step_convolution-1
    idx_tell_abs = np_where1D((telluric_spectrum<0.999) & (np.arange(nbins_mod)>=hnkern) & (np.arange(nbins_mod)<=nbins_mod-hnkern-1))

    #Process each pixel
    for p,hnkern_p,fwhm_loc in zip(idx_tell_abs,hnkern[idx_tell_abs],fwhm_angstrom_pixel[idx_tell_abs]):
        
        #Model spectrum contributes to the convolved spectrum at 'p' over the model pixel range 'range_psf'
        range_psf = np.arange(p-hnkern_p,p+hnkern_p+1)

        #Centered spectral table with same pixel widths as the band spectrum the kernel is associated to
        cen_bins_kernel=dwav_mod*np.arange(-hnkern_p,hnkern_p+1)

        #Discrete Gaussian kernel 
        gaussian_psf        = np.exp( - ((2.*np.sqrt(np.log(2.)))*cen_bins_kernel/fwhm_loc)**2.   ) 
        gaussian_psf_norm   = gaussian_psf/np.sum(gaussian_psf)

        #Convolving model spectrum using local resolution and attributing value at the center of the convolved range, ie at wavelength of pixel p
        telluric_spectrum_conv[p] = np.convolve(telluric_spectrum[range_psf],gaussian_psf_norm,mode='same')[hnkern_p]

    return telluric_spectrum_conv

def var_convol_tell_sp(telluric_spectrum,edge_bins_ord,edge_bins_mod,resolution_map_ord,nbins_mod,cen_bins_mod):
    """**Convolution of telluric spectrum (fast approach).**
    
    Adapted from a code by E. Artigau
    
    Args:
        TBD:
    
    Returns:
        telluric_spectrum_conv (1D array): Convolved telluric spectrum.
    
    """ 
    
    #Resample resolution map on spectrum table
    res_map = bind.resampling(edge_bins_mod, edge_bins_ord,resolution_map_ord, kind='cubic')
    idx_def = np_where1D(~np.isnan(res_map))
    if idx_def[0]>0:res_map[0:idx_def[0]]=res_map[idx_def[0]]
    if idx_def[-1]<nbins_mod:res_map[idx_def[-1]+1:nbins_mod]=res_map[idx_def[-1]]
    
    #Fill in internal nan ranges
    mask = np.isnan(res_map)
    if True in mask:res_map[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), res_map[~mask])

    #Measured instrumental response (FWHM = c/R) at each model wavelength
    fwhm_c = 1./res_map

    #Contributing width (in pixels) of the instrumental response, at 3 times the FWHM
    range_scan =  3.*np.max(fwhm_c)/(np.median(np.gradient(cen_bins_mod)/cen_bins_mod))
    range_scan = int(np.ceil(range_scan))

    #Scaling factor within the instrumental gaussian response
    ew = (fwhm_c /2.) / np.sqrt(np.log(2))

    #Convolving
    tot_weight = np.zeros(nbins_mod)
    telluric_spectrum_conv = np.zeros(nbins_mod)    
    for offset in range(-range_scan,range_scan):
        dv = (cen_bins_mod/np.roll(cen_bins_mod,offset)-1)
        
        #Super-gaussian profile
        weight = np.exp(-(dv/ew)**2.)

        #Adding contribution from current pixel
        telluric_spectrum_conv+=np.roll(telluric_spectrum,offset)*weight
        tot_weight+=weight

    #Normalization
    telluric_spectrum_conv/=tot_weight
    
    return telluric_spectrum_conv
   




def FIT_CCF_telluric_model(param,velccf,args=None):
    r"""**Main minimization function for CCF telluric model.**
    
    Calculates the merit factor of the telluric fit

    Args:
        param (Parameter): input parameters
        velccf (1D array): rv table
        args (dict): optional parameters
    
    Returns:
        chi (1D array): Table of :math:`\chi` values.
    
    """ 
    
    #Models over full spectral range
    ccf_uncorr_master,cov_uncorr,ccf_model_conv_master,_=CCF_telluric_model(param,velccf,args=args)
    
    #Merit table
    res = ccf_uncorr_master-ccf_model_conv_master
    if args['use_cov_eff']:
        L_mat = scipy.linalg.cholesky_banded(cov_uncorr, lower=True)
        chi = scipy.linalg.blas.dtbsv(L_mat.shape[0]-1, L_mat, res, lower=True)
        chi = chi[args['idx_sub_fit']]
    else:
        chi = res[args['idx_sub_fit']]/np.sqrt( cov_uncorr[0][args['idx_sub_fit']]) 

    return chi    




def CCF_telluric_model(params,velccf,args=None):
    """**Telluric CCF function.**
    
    Calculates telluric CCF and associated covariance matrix from measured and model spectra.  
    
    Adapted from the ATC code (author: R. Allart)
        
    Args:
        param (Parameter): input parameters
        velccf (1D array): rv table
        args (dict): optional parameters
    
    Returns:
        ccf_uncorr_master (1D array): telluric CCF of original spectrum
        cov_uncorr (ndarray): covariance matrix of **ccf_uncorr_master**
        ccf_model_conv_master (1D array): telluric CCF of model spectrum
        ccf_corr_master (ndarray): telluric CCF of corrected spectrum
    
    """ 
    
    #Fit parameters
    param_molecules = {args['molec']:params}
    tell_species = deepcopy([args['molec']])
    
    #Processing current molecule
    hwhm_scaled_dic,intensity_scaled_dic,nu_scaled_dic = init_tell_molec(tell_species,param_molecules,args['range_mol_prop'],args['qt_molec'],args['M_mol_molec'],args['c2'])
  
    #Telluric CCFs
    n_ccf = len(velccf)
    ccf_uncorr     = np.zeros(n_ccf,dtype=float)
    if args['ccf_corr']:ccf_corr     = np.zeros(n_ccf,dtype=float)
    nord_coadd = len(args['iord_fit_list'])
    cov_uncorr_ord = np.zeros(nord_coadd,dtype=object)
    nd_cov_uncorr_ord=np.zeros(nord_coadd,dtype=int) 
    ccf_model_conv = np.zeros(n_ccf,dtype=float)

    #Processing requested orders
    norm_cont = 0.
    ord_coadd_eff = []
    for isub_ord,iord in enumerate(args['iord_fit_list']):
        idx_def_ord = np_where1D(args['cond_def'][iord])

        #Order table reduced to minimum defined range       
        cen_bins_ord = args['cen_bins_earth'][iord,idx_def_ord[0]:idx_def_ord[-1]+1]        
        edge_bins_ord = args['edge_bins_earth'][iord,idx_def_ord[0]:idx_def_ord[-1]+2]

        #High-resolution telluric model
        #    - resolution set at order center: dw = w/R
        #    - model defined over a broader range than defined spectrum
        #    - model defined in nu [cm-1]
        dwav_mod      = np.mean(cen_bins_ord)/args['R_mod']
        nedge_mod = 40
        edge_mod =  nedge_mod*dwav_mod
        cen_bins_mod = np.arange(edge_bins_ord[0]-edge_mod,edge_bins_ord[-1]+edge_mod,dwav_mod)  
        nbins_mod = len(cen_bins_mod)
        nu_mod          = 1e8/cen_bins_mod
        nu_mod=nu_mod[np.argsort(nu_mod)]

        #Selects molecule lines in a narrower range than the spectrum range
        nu_sel_min = 1./((edge_bins_ord[-1]+4*dwav_mod)*10**-8) 
        nu_sel_max = 1./((edge_bins_ord[0]-4*dwav_mod)*10**-8)                 
            
        #Model telluric spectrum
        telluric_spectrum = calc_tell_model(tell_species,args['range_mol_prop'],nu_sel_min,nu_sel_max,intensity_scaled_dic,nu_scaled_dic,param_molecules,args['Nx_molec'],hwhm_scaled_dic,nu_mod,nbins_mod,edge_mod) 
 
       	#Convolution and binning  
        if np.min(telluric_spectrum)<1:

            #Convolution
            edge_bins_mod = def_edge_tab(cen_bins_mod,dim=0)
            if args['fixed_res']:
                telluric_spectrum_conv = convol_prof(telluric_spectrum,cen_bins_mod,args['resolution_map'](args['inst'],cen_bins_mod[int(len(cen_bins_mod)/2)]))
            else:
                resolution_map_ord = args['resolution_map'][iord,idx_def_ord[0]:idx_def_ord[-1]+1]        
                telluric_spectrum_conv = var_convol_tell_sp(telluric_spectrum,edge_bins_ord,edge_bins_mod,resolution_map_ord,nbins_mod,cen_bins_mod)

            #CCF linelist
            #    - CCF linelist is provided at rest, and must be corrected for pressure shift (see init_tell_molec())
            CCF_line_nu_rest = 1e8/args['CCF_lines'] 
            CCF_line_nu_scaled = CCF_line_nu_rest + args['CCF_deltas'] * param_molecules[args['molec']]['Pressure_ground'].value
            CCF_line_wav_scaled = 1e8/CCF_line_nu_scaled

            #Keep lines fully within spectrum range without nan
            flux_ord = args['flux'][iord,idx_def_ord[0]:idx_def_ord[-1]+1]        
            cov_ord = args['cov'][iord][:,idx_def_ord[0]:idx_def_ord[-1]+1] 
            gdet_ord = args['mean_gdet'][iord,idx_def_ord[0]:idx_def_ord[-1]+1] 
            if args['ccf_corr']:flux_corr_ord = args['flux_corr'][iord][idx_def_ord[0]:idx_def_ord[-1]+1] 
            cond_def_ord = args['cond_def'][iord,idx_def_ord[0]:idx_def_ord[-1]+1]        
            idx_maskL_kept = check_CCF_mask_lines(1,np.array([edge_bins_ord]),np.array([cond_def_ord]),CCF_line_wav_scaled,args['edge_velccf'])
            if len(idx_maskL_kept)>0:
                ord_coadd_eff+=[isub_ord]
          
                #Weights unity
                sij_ccf       = np.ones(len(idx_maskL_kept))
                wave_line_ccf = CCF_line_wav_scaled[idx_maskL_kept]

                #Compute CCFs
                #    - multiprocessing not efficient for these calculations
                edge_velccf_fit = args['edge_velccf'][args['idx_mod'][0]:args['idx_mod'][-1]+2]
                ccf_uncorr_ord,    cov_ccf_uncorr_ord      = new_compute_CCF(edge_bins_ord,flux_ord,cov_ord,args['resamp_mode'],edge_velccf_fit,sij_ccf,wave_line_ccf,1,cal = gdet_ord)[0:2]
                cov_uncorr_ord[isub_ord] = cov_ccf_uncorr_ord 
                nd_cov_uncorr_ord[isub_ord] = np.shape(cov_ccf_uncorr_ord)[0]
                if args['ccf_corr']:ccf_corr_ord = new_compute_CCF(edge_bins_ord,flux_corr_ord,None,args['resamp_mode'],edge_velccf_fit,sij_ccf,wave_line_ccf,1,cal = gdet_ord)[0]
                ccf_model_conv_ord = new_compute_CCF(edge_bins_mod,telluric_spectrum_conv,None,args['resamp_mode'],edge_velccf_fit,sij_ccf,wave_line_ccf,1)[0]
               
                #Normalize CCFs and co-add
                #    - data-based CCF is kept to its original level to naturally account for SNR and noise variations between orders
                cond_cont_ccf = (edge_velccf_fit[0:-1] < args['mol_cont_range'][0] ) | (edge_velccf_fit[1::] > args['mol_cont_range'][1] )
                cont_data = np.nanmedian(ccf_uncorr_ord[cond_cont_ccf])
                norm_cont+=cont_data
                ccf_uncorr+=ccf_uncorr_ord
                if args['ccf_corr']:ccf_corr+=ccf_corr_ord
                ccf_model_conv+=ccf_model_conv_ord*cont_data/np.nanmedian(ccf_model_conv_ord[cond_cont_ccf])

    #Check for absence of telluric absorption
    if len(ord_coadd_eff)==0:stop('No telluric absorption calculated: check starting values')

    #Maximum dimension of covariance matrix in current exposure from all contributing orders
    nd_cov_uncorr = np.amax(nd_cov_uncorr_ord)   

    #Co-adding contributions from orders
    cov_uncorr = np.zeros([nd_cov_uncorr,n_ccf])
    for isub_ord in ord_coadd_eff:
        cov_uncorr[0:nd_cov_uncorr_ord[isub_ord],:] +=  cov_uncorr_ord[isub_ord]   
        
    #Mean CCF over all orders
    ccf_uncorr_master=ccf_uncorr/norm_cont
    cov_uncorr/=norm_cont**2.
    if args['ccf_corr']:ccf_corr_master=ccf_corr/norm_cont
    else:ccf_corr_master = None
    ccf_model_conv_master=ccf_model_conv/norm_cont
  
    #Continuum of CCF   
    if args['CCFcont_deg'][args['molec']]>0:
        for ideg_CCFcont in range(1,args['CCFcont_deg'][args['molec']]+1):
            ccf_model_conv_master+=params['cont'+str(ideg_CCFcont)]*velccf**ideg_CCFcont

    return ccf_uncorr_master,cov_uncorr,ccf_model_conv_master,ccf_corr_master

    



def full_telluric_model(flux_exp,cov_exp,args,param_molecules,tell_species,tell_mol_dic,range_mol_prop,qt_molec,M_mol_molec,N_x_molecules,resolution_map,inst,resamp_mode,dim_exp):
    """**Telluric spectrum function.**
    
    Calculates telluric spectral model over full spectral range and uses it to correct measured spectrum.
    
    Adapted from the ATC code (author: R. Allart)
        
    Args:
        TBD
    
    Returns:
        tell_spec_exp (nord,nspec array): model telluric spectrum
        corr_flux_exp (nord,nspec array): corrected measured spectrum
        corr_cov_exp (nord,nspec array): covariance matrix of **corr_flux_exp**
    
    """ 
    
    #Processing requested molecules
    hwhm_scaled_dic,intensity_scaled_dic,nu_scaled_dic = init_tell_molec(tell_species,param_molecules,range_mol_prop,qt_molec,M_mol_molec,args['c2'])
    
    #Telluric spectrum
    tell_spec_exp = np.ones(dim_exp,dtype=float)  

    #Processing requested orders
    corr_flux_exp = deepcopy(flux_exp)
    corr_cov_exp = deepcopy(cov_exp)
    for isub,iord in enumerate(args['iord_corr_list']):
        idx_def_ord = np_where1D(args['cond_def'][iord])

        #Order table reduced to minimum defined range   
        cen_bins_ord = args['cen_bins_earth'][iord,idx_def_ord[0]:idx_def_ord[-1]+1]        
        edge_bins_ord = args['edge_bins_earth'][iord,idx_def_ord[0]:idx_def_ord[-1]+2]

        #High-resolution telluric model
        #    - resolution set at order center
        #    - model defined over a broader range than defined spectrum
        #    - model defined in nu [cm-1]
        dwav_mod      = np.mean(cen_bins_ord)/args['R_mod']
        nedge_mod=40
        edge_mod = nedge_mod*dwav_mod
        cen_bins_mod = np.arange(edge_bins_ord[0]-edge_mod,edge_bins_ord[-1]+edge_mod,dwav_mod) 
        nbins_mod = len(cen_bins_mod)
        nu_mod    = 1e8/cen_bins_mod   
        nu_mod=nu_mod[np.argsort(nu_mod)]

        #Selects molecule lines in a broader range than the spectrum range
        #    - we keep lines over the full model range so that their wings can contribute over the spectrum range
        nu_sel_min = nu_mod[0]
        nu_sel_max = nu_mod[-1]

        #Telluric spectrum
        telluric_spectrum = calc_tell_model(tell_species,range_mol_prop,nu_sel_min,nu_sel_max,intensity_scaled_dic,nu_scaled_dic,param_molecules,N_x_molecules,hwhm_scaled_dic,nu_mod,nbins_mod,edge_mod)        

       	#Convolution and binning  
        if np.min(telluric_spectrum)<1:
    
            #Convolution
            if args['fixed_res']:
                telluric_spectrum_conv = convol_prof(telluric_spectrum,cen_bins_mod,args['resolution_map'](args['inst'],cen_bins_mod[int(len(cen_bins_mod)/2)]))
            else:
                edge_bins_mod = def_edge_tab(cen_bins_mod,dim=0)
                telluric_spectrum_conv = var_convol_tell_sp(telluric_spectrum,edge_bins_ord,edge_bins_mod,resolution_map[iord,idx_def_ord[0]:idx_def_ord[-1]+1],nbins_mod,cen_bins_mod)
                
            #Resampling
            edge_bins_mod = def_edge_tab(cen_bins_mod,dim=0)
            tell_spec_exp[iord] = bind.resampling(args['edge_bins_earth'][iord], edge_bins_mod,telluric_spectrum_conv, kind=resamp_mode)                         

        #Correct data in current order
        corr_flux_exp[iord],corr_cov_exp[iord] = bind.mul_array(flux_exp[iord],cov_exp[iord],1./tell_spec_exp[iord])

        #Telluric threshold
        #    - flux values where telluric contrast is deeper than this threshold are set to nan, as we consider the telluric-absorbed flux is too low and noisy to be retrieved
        cond_deep_tell = (1.-tell_spec_exp[iord]>args['tell_thresh_corr'])
        if True in cond_deep_tell:corr_flux_exp[iord,cond_deep_tell] = np.nan

    return tell_spec_exp,corr_flux_exp,corr_cov_exp
