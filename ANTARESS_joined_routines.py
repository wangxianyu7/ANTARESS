#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 00:05:29 2020

@author: V. Bourrier
"""

from utils import stop,np_where1D,npint,np_interp
from ANTARESS_routines import sub_calc_plocc_prop,calc_pl_coord,return_FWHM_inst,convol_prof,dispatch_func_prof,model_par_names,polycoeff_def,def_st_prof_tab,conv_st_prof_tab,cond_conv_st_prof_tab,model_st_prof_tab,\
                            par_formatting,conv_cosistar,conv_CF_intr_meas,occ_region_grid,calc_polymodu,compute_deviation_profile, calc_binned_prof,init_custom_DI_prof,init_custom_DI_par,ref_inst_convol
from copy import deepcopy
from lmfit import Parameters
import lmfit
import numpy as np
from minim_routines import init_fit,call_MCMC,postMCMCwrapper_1,postMCMCwrapper_2,save_fit_results,fit_merit,ln_prob_func_lmfit,fit_minimization
import scipy.linalg
from constant_data import Rsun,c_light
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d



'''
Wrap-up for definition of complex prior functions for MCMC runs
'''
#Prior on sin(istar)
#    - use only if istar is fitted
#    - assumes isotropic distribution of stellar orientations (equivalent to draw a uniform distribution on cosistar)
#    - pr(x) = sin(istar(x))/2, normalized by integral of sin(istar) over the definition space of istar (0:180), which is 2 
#      ln(pr(x)) = ln(0.5*sin(istar(x)))
#    - sin(istar) defined directly from cos(istar)    
def prior_sini_geom(p_step_loc,fixed_args,prior_func_prop):
    sin_istar=np.sqrt(1.-p_step_loc['cos_istar']**2.)   
    return np.log(0.5*sin_istar) 

#Prior on 0<ip<90degrees
#    - when letting free the orbital inclination
def prior_cosi(p_step_loc,fixed_args,prior_func_prop): 
    ln_p = 0.
    for pl_loc in fixed_args['inclin_rad_pl']:
        if np.cos(p_step_loc['inclin_rad__pl'+pl_loc]) < 0:
            ln_p = -np.inf
            break
    return ln_p

def prior_sini(p_step_loc,fixed_args,prior_func_prop): 
    return -0.5*( (     np.sin(p_step_loc['inclin_rad__pl'+prior_func_prop['pl']]) - prior_func_prop['val'])/prior_func_prop['sig'])**2.                

#Prior on the impact parameter b<1
#    - when letting free the orbital inclination and/or semi-major axis
#    - if only one is free to vary, the other should still be defined as a constant parameter so that this prior can be used
def prior_b(p_step_loc,fixed_args,prior_func_prop):  
    ln_p = 0.
    for pl_loc in fixed_args['b_pl']:
        if np.abs(p_step_loc['aRs__pl'+pl_loc]*np.cos(p_step_loc['inclin_rad__pl'+pl_loc])) > 1:
            ln_p = -np.inf
            break
    return ln_p    

#If both differential rotation coefficients are defined, their sum cannot be larger than 1 if the star rotates in the same direction at all latitudes
def prior_DR(p_step_loc,fixed_args,prior_func_prop):
    if (p_step_loc['alpha_rot']+p_step_loc['beta_rot']>1.):return -np.inf	
    else: return 0.

#Prior on veq*sin(istar)
#    - relevant when veq and istar are fitted independently, otherwise set a simple prior on veq
#    - gaussian prior
#    ln(pr(x)) = - chi2_x / 2.
#        with chi2_x = ( (x - x_constraint)/s_constraint  )^2
def prior_vsini(p_step_loc,fixed_args,prior_func_prop):
    vsini = p_step_loc['veq']*np.sqrt(1.-p_step_loc['cos_istar']**2.)
    return -0.5*( (vsini - prior_func_prop['val'])/prior_func_prop['sig'])**2.                

def prior_vsini_deriv(p_step_loc,fixed_args,prior_func_prop):
    vsini = 2.*np.pi*p_step_loc['Rstar']*Rsun*np.sqrt(1.-p_step_loc['cos_istar']**2.)/(p_step_loc['Peq']*24.*3600.)
    return -0.5*( (vsini - prior_func_prop['val'])/prior_func_prop['sig'])**2.    

#Prior on intrinsic contrast
#    - the contrast of (intrinsic, and a fortiori measured) stellar lines should be between 0 and 1
#    - if any of the fitted exposures has its contrast go beyond these boundaries the ln_p is set to -inf
#    - see details in joined_CCFs()
#    - function is coded assuming a single planet per visit transits
def prior_contrast(p_step_loc,args_in,prior_func_prop):
    ln_p_loc = 0.
    args = deepcopy(args_in)
    args['precision']='low'
    for inst in args['inst_list']:
        args['inst']=inst
        for vis in args['inst_vis_list'][inst]:   
            args['vis']=vis
            pl_vis = args['transit_pl'][inst][vis][0]
            system_param_loc,coord_pl = calc_plocc_coord(inst,vis,[args['coord_line']],args,p_step_loc,[pl_vis],args['nexp_fit_all'][inst][vis],args['ph_fit'][inst][vis],args['coord_pl_fit'][inst][vis])
            surf_prop_dic = sub_calc_plocc_prop(args['chrom_mode'],args,[args['coord_line']],[pl_vis],system_param_loc,args['theo_dic'],args['system_prop'],p_step_loc,args['coord_pl_fit'][inst][vis],args['nexp_fit_all'][inst][vis])
            ctrst_vis = surf_prop_dic[pl_vis]['ctrst'][0]       
            break_cond = (ctrst_vis<0.) | (ctrst_vis>1.)
            if True in break_cond:
                ln_p_loc+= -np.inf	
                break 
    return ln_p_loc

#Prior on the line width
#    - we assume that the local line cannot be larger than the disk-integrated one, after rotational broadening
#      this sets an upper limit on the local FWHM, considering there can be other broadening contributions 
#    - the local line must have constant width
def prior_FWHM_vsini(p_step_loc,args,prior_func_prop):
    ln_p_loc = 0.
    vsini = p_step_loc['veq']*np.sqrt(1.-p_step_loc['cos_istar']**2.)
    for inst in args['inst_list']:
        for vis in args['inst_vis_list'][inst]:  

            #Width of disk-integrated profile, rotationally broadened, after instrumental convolution
            FWHM_intr = p_step_loc[args['name_prop2input']['FWHM_ord0__IS'+inst+'_VS'+vis]]
            FWHM_DI_mod2 = FWHM_intr**2. + return_FWHM_inst(inst,c_light)**2. + vsini**2.
            
            #Width must be smaller than width of measured disk-integrated profile
            if FWHM_DI_mod2>prior_func_prop['FWHM_DI']**2.:
                ln_p_loc+= -np.inf	
                break 

    return ln_p_loc


#Dictionary to store prior functions
prior_functions={
    'sinistar_geom':prior_sini_geom,
    'cosi':prior_cosi,
    'sini':prior_sini,
    'b':prior_b,
    'DR':prior_DR,
    'vsini':prior_vsini,
    'contrast':prior_contrast,
    'FWHM_vsini':prior_FWHM_vsini
}

#Global prior function
def global_ln_prior_func(p_step_loc,fixed_args):
    ln_p_loc = 0.
    for key in ['sinistar_geom','cosi','sini','b','DR','vsini','contrast','FWHM_vsini']:
        if key in fixed_args['prior_func']:ln_p_loc+=prior_functions[key](p_step_loc,fixed_args,fixed_args['prior_func'][key])
    return ln_p_loc












'''
Post-processing routine
'''
def gen_hrand_chain(par_med,epar_low,epar_high,n_throws):
    if epar_high==epar_low:
        hrand_chain = np.random.normal(par_med, epar_high, n_throws)
    else:
        if n_throws>1:
            if n_throws<20:n_throws_half = 10*n_throws
            else:n_throws_half = 2*n_throws
            rand_draw_right = np.random.normal(loc=par_med, scale=epar_high, size=n_throws_half)
            rand_draw_right = rand_draw_right[rand_draw_right>par_med]
            rand_draw_right = rand_draw_right[0:int(n_throws/2)]
            rand_draw_left = np.random.normal(loc=par_med, scale=epar_low, size=n_throws_half)
            rand_draw_left = rand_draw_left[rand_draw_left<=par_med]
            rand_draw_left = rand_draw_left[0:n_throws-int(n_throws/2)]
            hrand_chain = np.append(rand_draw_left,rand_draw_right)
        else:
            if np.random.normal(loc=0., scale=1., size=1)>0:hrand_chain = np.random.normal(loc=par_med, scale=epar_high, size=1)
            else:hrand_chain = np.random.normal(loc=par_med, scale=epar_low, size=1)
    return hrand_chain

def post_proc_func(p_final,fixed_args,fit_dic,merged_chain,fit_prop_dic,gen_dic):

    #Call to specific post-processing
    #    - Fit from chi2
    # + combination/modification of best-fit results to add new parameters
    #    - Fit from mcmc
    # + combination/modification of MCMC chains to add new parameters
    # + new parameters are not used for model
    # + we calculate median and errors after chain are added   
    if (fit_dic['fit_mod'] in ['chi2','mcmc']) and ('modif_list' in fit_prop_dic) and len(fit_prop_dic['modif_list'])>0:
        print('     ----------------------------------')
        print('     Post-processing')
        modif_list = fit_prop_dic['modif_list']
    
        #Convert Rstar and Peq into veq
        #    - veq = (2*pi*Rstar/Peq)
        if 'veq_from_Peq_Rstar' in modif_list:
            print('     + Deriving veq from Req and Rstar ')
            
            if fit_dic['fit_mod']=='chi2': 
                p_final['veq']=(2.*np.pi*p_final['Peq']*Rsun)/(p_final['Peq']*3600.*24.)
                sig_loc=np.nan
                fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))
            elif fit_dic['fit_mod']=='mcmc':   
                iPeq = np_where1D(fixed_args['var_par_list']=='Peq')
                iRstar = np_where1D(fixed_args['var_par_list']=='Rstar')
                chain_loc=(2.*np.pi*np.squeeze(merged_chain[:,iRstar])*Rsun)/(np.squeeze(merged_chain[:,iPeq])*3600.*24.)
                merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)  
            fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'veq')
            fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],'v$_{eq}$(km/s)')
    
    
        #Convert veq into veq*sin(istar) to be comparable with solid-body values
        #    - must be done before modification of istar chain
        if 'vsini' in modif_list:
            print('     + Converting veq to vsini')
            if 'veq' in fixed_args['var_par_list']:iveq=np_where1D(fixed_args['var_par_list']=='veq')
            else:stop('Activate veq_from_Peq_Rstar')
            iistar = np_where1D(fixed_args['var_par_list']=='cos_istar')
            if fit_dic['fit_mod']=='chi2':             
                #    - vsini = veq*sin(i)
                #      dvsini = vsini*sqrt( (dveq/veq)^2 + (dsini/sini)^2 )
                #      d[sini] = cos(i)*di
                sin_istar = np.sqrt(1.-p_final['cos_istar']**2.)
                p_final['vsini'] = p_final['veq']*sin_istar
                if ('cos_istar' in fixed_args['var_par_list']):
                    dsini = p_final['cos_istar']*fit_dic['sig_parfinal_err']['1s'][0,iistar]/ sin_istar
                    sig_temp = p_final['vsini']*np.sqrt( (fit_dic['sig_parfinal_err']['1s'][0,iveq]/p_final['veq'])**2. + (dsini/sin_istar)**2. )  
                else:
                    sig_temp = fit_dic['sig_parfinal_err']['1s'][:,iveq]*sin_istar            
                fit_dic['sig_parfinal_err']['1s'][:,iveq] = sig_temp
            elif fit_dic['fit_mod']=='mcmc':                  
                if ('cos_istar' in fixed_args['var_par_list']):
                    cosistar_chain=merged_chain[:,iistar]  
                else:
                    #Stellar inclination is fixed
                    cosistar_chain=p_final['cos_istar']
                veq_chain=deepcopy(merged_chain[:,iveq])            
                merged_chain[:,iveq]=veq_chain*np.sqrt(1.-cosistar_chain*cosistar_chain)
            fixed_args['var_par_list'][iveq]='vsini'            
            fixed_args['var_par_names'][iveq]='v$_\mathrm{eq}$sin i$_{*}$ (km/s)'          
    
            
        #-------------------------------------------------            

        #Replace cos(istar[rad]) by istar[deg]
        if ('istar_deg_conv' in modif_list) or ('istar_deg_add' in modif_list):
            print('     + Converting cos(istar) to istar')
            conv_cosistar(modif_list,fixed_args,fit_dic,p_final,merged_chain)

        #-------------------------------------------------            

        #Folding istar[deg] around 90
        #    - only use if all other parameters are degenerate with istar
        #    - by default we fold over 0-90
        if ('fold_istar' in modif_list):        
            print('     + Folding istar')
            iistar = np_where1D(fixed_args['var_par_list']=='istar_deg')
            istar_temp=np.squeeze(merged_chain[:,iistar])
            w_gt_90=(istar_temp > 90.)
            if True in w_gt_90:merged_chain[w_gt_90,iistar]=180.-istar_temp[w_gt_90]

        #-------------------------------------------------
        #Add istar using the value derived from vsini and independent measurement of Peq and Rstar
        #    - prefer the use of Peq and Rstar as fit parameters
        #    - vsini = veq*sin(istar) = (2*pi*Rstar/Peq)*sin(istar)
        #      istar = np.arcsin( vsini*peq/(2*pi*Rstar) )
        if ('istar_Peq' in modif_list) or ('istar_Peq_vsini' in modif_list):
            if ('cos_istar' in fixed_args['var_par_list']):stop('    istar has been fitted')
            print('     + Deriving istar from vsini and Peq')
            
            #Nominal values and 1s errors for Rstar (Rsun) and Peq (d)
            # Rstar_med = 0.850       #HD3167
            # Rstar_err = 0.020
            # Peq_med = 23.52         #HD3167
            # Peq_err = 2.87
            # Rstar_med = 0.438       #GJ436
            # Rstar_err = 0.013
            # Peq_med = 44.09       
            # Peq_err = 0.08
            if gen_dic['star_name']=='HAT_P3':
                Rstar_med = 0.85       
                Rstar_err = 0.021
                Peq_med = 19.9 
                Peq_elow = 1.5
                Peq_ehigh = 1.5
            elif gen_dic['star_name']=='HAT_P11':
                Rstar_med = 0.74       
                Rstar_err = 0.01
                Peq_med = 30.5       
                Peq_elow = 3.2
                Peq_ehigh = 4.1
            # Rstar_med = 0.901       #Kepler-63
            # Rstar_err = 0.0245
            # Peq_med = 5.401
            # Peq_err =   0.014  
            elif gen_dic['star_name']=='WASP107':            
                Rstar_med = 0.67       
                Rstar_err = 0.02
                Peq_med = 17.1
                Peq_elow = 1.
                Peq_ehigh = 1.  
            # Rstar_med = 1.273       #HIP 41378
            # Rstar_err = 0.015
            # Peq_med = 6.4
            # Peq_err = 0.8  
            elif gen_dic['star_name']=='Kepler25':
                Rstar_med = 1.316       
                Rstar_err = 0.016
                Peq_med = 23.147         
                Peq_elow = 0.039
                Peq_ehigh = 0.039
            elif gen_dic['star_name']=='WASP47':
                Rstar_med = 1.137       
                Rstar_err = 0.013
                Peq_med = 39.4         
                Peq_elow = 4.5
                Peq_ehigh = 2.2
            elif gen_dic['star_name']=='WASP166':
                Rstar_med = 1.22       
                Rstar_err = 0.06
                Peq_med = 12.3      
                Peq_elow = 1.9
                Peq_ehigh = 1.9                
                
            #Conversions
            Rstar_med*=Rsun
            Rstar_err*=Rsun
            Peq_med*=24.*3600.
            Peq_elow*=24.*3600.
            Peq_ehigh*=24.*3600.
            
            if fit_dic['fit_mod']=='chi2':             
                p_final['istar_deg']=np.arcsin(p_final['vsini']*Peq_med/(2.*np.pi*Rstar_med))*180./np.pi
                sig_loc=np.nan
                fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))
            elif fit_dic['fit_mod']=='mcmc':   
                n_chain = len(merged_chain[:,0])
                if 'istar_Peq_vsini' in modif_list:
                    print('       Using external vsini')
                    if gen_dic['star_name']=='WASP47':
                        vsini_med = 1.80         
                        vsini_elow = 0.16
                        vsini_ehigh = 0.24   
                    vsini_chain = gen_hrand_chain(vsini_med,vsini_elow,vsini_ehigh,n_chain)
                elif 'istar_Peq' in modif_list:
                    print('       Using derived vsini')
                    vsini_chain = np.squeeze(merged_chain[:,np_where1D(fixed_args['var_par_list']=='vsini')])

                #Generate gaussian distribution for Rstar and Peq
                Rstar_chain = np.random.normal(Rstar_med, Rstar_err, n_chain)
                Peq_chain = gen_hrand_chain(Peq_med,Peq_elow,Peq_ehigh,n_chain)

                #Calculate sin(istar) chain
                sinistar_chain = vsini_chain*Peq_chain/(2.*np.pi*Rstar_chain)
                               
                #Replace non-physical values
                cond_good = np.abs(sinistar_chain)<=1.
                n_good = np.sum(cond_good)
                while n_good<n_chain:
                    Rstar_add = np.random.normal(Rstar_med, Rstar_err, n_chain-n_good)
                    Peq_add = gen_hrand_chain(Peq_med,Peq_elow,Peq_ehigh,n_chain-n_good)
                    if 'istar_Peq_vsini' in modif_list:vsini_add = gen_hrand_chain(vsini_med,vsini_elow,vsini_ehigh,n_chain-n_good)
                    elif 'istar_Peq' in modif_list:vsini_add = np.random.choice(vsini_chain,n_chain-n_good)
                    sinistar_chain = np.append(sinistar_chain[cond_good],vsini_add*Peq_add/(2.*np.pi*Rstar_add))
                    cond_good = np.abs(sinistar_chain)<=1.
                    n_good = np.sum(cond_good)
                
                #istar chain
                chain_loc=np.arcsin( sinistar_chain )*180./np.pi 
                
                # #pi-istar chain
                # chain_loc=180.-chain_loc   
                
                merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)  
            fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'istar_deg')
            fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],'i$_{*}(^{\circ}$)')
    
        #-------------------------------------------------
        #Add Peq using the value derived from veq and independent measurement of Rstar
        #    - Peq = (2*pi*Rstar/veq)
        if 'Peq_veq' in modif_list:
            if ('cos_istar' in fixed_args['var_par_list']):stop('    istar has been fitted')
            print('     + Deriving Peq from veq ')
            
            #Nominal values and 1s errors for Rstar (Rsun) and Peq (d)
            Rstar_med = 0.850       #HD3167
            Rstar_err = 0.020
    
            #Conversions
            Rstar_med*=Rsun
            Rstar_err*=Rsun
            
            if fit_dic['fit_mod']=='chi2': 
                p_final['Peq_d']=(2.*np.pi*Rstar_med)/(p_final['veq']*3600.*24.)
                sig_loc=np.nan
                fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))
            elif fit_dic['fit_mod']=='mcmc':   
            
                #Generate gaussian distribution for Rstar
                Rstar_chain = np.random.normal(Rstar_med, Rstar_err, len(merged_chain[:,0]))
                
                #Calculate Peq chain
                iveq = np_where1D(fixed_args['var_par_list']=='veq')
                chain_loc=(2.*np.pi*Rstar_chain)/(np.squeeze(merged_chain[:,iveq])*3600.*24.)
                merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)  
            fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'Peq')
            fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],'P$_{eq}$(d)')
    
        #-------------------------------------------------
        #Add Peq using the value derived from vsini and independent measurement of Rstar and istar
        #    - Peq = (2*pi*Rstar*sin(istar)/vsini)
        if 'Peq_vsini' in modif_list:
            print('     + Deriving Peq from vsini')
            
            #Nominal values and 1s errors for Rstar (Rsun) 
            if gen_dic['star_name']=='Kepler25':
                Rstar_med = 1.316       
                Rstar_err = 0.016

                
            #Generate distribution for istar
            if gen_dic['star_name']=='Kepler25':
                istar_mean = 66.7*np.pi/180.    
                istar_high = 12.1*np.pi/180.
                istar_low  = 7.4*np.pi/180.
    
            #Conversions
            Rstar_med*=Rsun
            Rstar_err*=Rsun

            if fit_dic['fit_mod']=='chi2': 
                stop('TBD')
            elif fit_dic['fit_mod']=='mcmc':  
                n_chain = len(merged_chain[:,0])
            
                #Generate gaussian distribution for Rstar
                Rstar_chain = np.random.normal(Rstar_med, Rstar_err,n_chain )
                
                #Generate distribution for sin(istar) 
                sinistar_chain = np.sin(gen_hrand_chain(istar_mean,istar_low,istar_high,n_chain))
                
                #Calculate Peq chain
                if 'vsini' not in fixed_args['var_par_list']:stop('Add vsini chain')
                ivsini = np_where1D(fixed_args['var_par_list']=='vsini')
                chain_loc=(2.*np.pi*Rstar_chain*sinistar_chain)/(np.squeeze(merged_chain[:,ivsini])*3600.*24.)
                merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)  
            fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'Peq')
            fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],'P$_{eq}$(d)')    
    
        #-------------------------------------------------
                
        #Add true obliquity
        #    - psi = acos(sin(istar)*cos(lamba)*sin(ip) + cos(istar)*cos(ip))
        #    - must be done before modification of istar and lambda chains
        if ('psi' in modif_list) or ('psi_lambda' in modif_list):
            print('     + Adding true obliquity')
            for pl_loc in fixed_args['lambda_rad_pl']:              
                lambda_rad_pl = 'lambda_rad__pl'+pl_loc
                if fit_dic['fit_mod']=='chi2':              
                    istar = p_final['istar_deg']*np.pi/180.
                    if lambda_rad_pl in fixed_args['genpar_instvis']:
                        for inst in fixed_args['genpar_instvis'][lambda_rad_pl]:
                            for vis in fixed_args['genpar_instvis'][lambda_rad_pl][inst]:
                                p_final['Psi__pl'+pl_loc+'__IS'+inst+'_VS'+vis]=np.arccos(np.sin(istar)*np.cos(p_final[lambda_rad_pl+'__IS'+inst+'_VS'+vis])*np.sin(p_final['inclin_rad__pl'+pl_loc]) + np.cos(istar)*np.cos(p_final['inclin_rad__pl'+pl_loc]))*180./np.pi
                                fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[np.nan],[np.nan]]))                    
                    else:        
                        p_final['Psi__pl'+pl_loc]=np.arccos(np.sin(istar)*np.cos(p_final[lambda_rad_pl])*np.sin(p_final['inclin_rad__pl'+pl_loc]) + np.cos(istar)*np.cos(p_final['inclin_rad__pl'+pl_loc]))*180./np.pi
                        fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[np.nan],[np.nan]]))
                elif fit_dic['fit_mod']=='mcmc':    
                    n_chain = len(merged_chain[:,0])
                    
                    #Obliquity 
                    if 'psi_lambda' in modif_list:
                        print('       Using external lambda')
                        if gen_dic['star_name']=='WASP47':
                            lambda_med = 0.         
                            lambda_elow = 24.
                            lambda_ehigh = 24.  
                            
                        lambda_med*=np.pi/180.
                        lambda_elow*=np.pi/180.
                        lambda_ehigh*=np.pi/180.
                        lamb_chain = {'':{'':gen_hrand_chain(lambda_med,lambda_elow,lambda_ehigh,n_chain)}}
                    elif 'psi' in modif_list:
                        print('       Using derived lambda')
                        if lambda_rad_pl in fixed_args['genpar_instvis']:
                            lamb_chain={}
                            for inst in fixed_args['genpar_instvis'][lambda_rad_pl]:
                                lamb_chain[inst] = {}
                                for vis in fixed_args['genpar_instvis'][lambda_rad_pl][inst]:      
                                    lamb_chain[inst][vis] = np.squeeze(merged_chain[:,np_where1D(fixed_args['var_par_list']==lambda_rad_pl+'__IS'+inst+'_VS'+vis)]  ) 
                        else:
                            lamb_chain = {'':{'':np.squeeze(merged_chain[:,np_where1D(fixed_args['var_par_list']==lambda_rad_pl)]  )}} 
                    
                    #Stellar inclination
                    if ('istar_deg' in fixed_args['var_par_list']):
                        print('       Using derived istar')
                        istarN_chain=np.squeeze(merged_chain[:,np_where1D(fixed_args['var_par_list']=='istar_deg')]    )*np.pi/180. 
                        istarS_chain=np.pi-istarN_chain
                    else:
                        print('       Using external istar')
                        
                        #Complex PDF on istar
                        if pl_loc=='HD89345b':                        
                            istar_mean = 37.*np.pi/180. 
                            frac_chain = 0.75
                            rand_draw_right = np.random.uniform(low=istar_mean, high=90.*np.pi/180., size=4*n_chain)
                            rand_draw_right = rand_draw_right[rand_draw_right>istar_mean]
                            rand_draw_right = rand_draw_right[0:int(frac_chain*n_chain)]    #cf Fig 5 Van Eylen+2018, 68% contenu dans la partie uf   
                            rand_draw_left = np.random.normal(loc=istar_mean, scale=15.*np.pi/180., size=4*n_chain)
                            rand_draw_left = rand_draw_left[(rand_draw_left<=istar_mean) & (rand_draw_left>=0.)]
                            rand_draw_left = rand_draw_left[0:n_chain-len(rand_draw_right)]   
                            istarN_chain = np.append(rand_draw_left,rand_draw_right)
                    
                            # #Check distribution
                            # hist_val, bin_edg_val = np.histogram(istarN_chain, bins=50,density=True)
                            # grid_val = 0.5*(bin_edg_val[0:-1]+bin_edg_val[1:])
                            # cdf_val = np.cumsum(hist_val)
                            # cdf_val = (cdf_val-np.min(cdf_val))/(np.max(cdf_val)-np.min(cdf_val))                            
                            # rand_draw = np.random.uniform(low=0.0, high=1.0, size=len(istarN_chain))
                            # irand_pts = np_interp(rand_draw,cdf_val,grid_val)  
                            # hist_itest, bin_edg_itest = np.histogram(irand_pts, bins=50,density=True)
                            # grid_itest = 0.5*(bin_edg_itest[0:-1]+bin_edg_itest[1:])
                            # plt.plot(grid_itest*180./np.pi,hist_itest,drawstyle='steps-mid',color='orange')
                            # plt.show()
                            # stop()                            

                        #Gaussian or half-gaussian PDFs on istar
                        else:                    
    
                            #Generate distribution for istar
                            if pl_loc=='Kepler25c':
                                istar_mean = 66.7*np.pi/180.    
                                istar_high = 12.1*np.pi/180.
                                istar_low  = 7.4*np.pi/180.
                            elif pl_loc=='Kepler63b':
                                istar_mean = 138.*np.pi/180.    
                                istar_high = 7.*np.pi/180.
                                istar_low  = 7.*np.pi/180.                                
                            elif pl_loc=='HAT_P11b':
                                # istar_mean = 80.*np.pi/180.    
                                # istar_high = 5.*np.pi/180.
                                # istar_low  = 3.*np.pi/180.

                                istar_mean = 160.*np.pi/180.    
                                istar_high = 9.*np.pi/180.
                                istar_low  = 19.*np.pi/180.
                            istarN_chain = gen_hrand_chain(istar_mean,istar_low,istar_high,n_chain)

                        #Symmetrical PDF around 90°
                        istarS_chain=np.pi-istarN_chain
                    
                    
                    
                    
                    
                    #Orbital inclination
                    if ('inclin_rad__pl'+pl_loc in fixed_args['var_par_list']):
                        inclin_rad_chain=np.squeeze(merged_chain[:,np_where1D(fixed_args['var_par_list']=='inclin_rad__pl'+pl_loc)])
                    else:
                        
                        #Generate distribution for ip
                        if pl_loc=='HAT_P3b':
                            ip_mean = 86.31*np.pi/180.    
                            ip_high = 0.19*np.pi/180.
                            ip_low  = 0.19*np.pi/180.
                        elif pl_loc=='HAT_P11b':
                            ip_mean = 89.05*np.pi/180.    
                            ip_high = 0.15*np.pi/180.
                            ip_low  = 0.09*np.pi/180.
                        elif pl_loc=='Kepler25c':
                            ip_mean = 87.236*np.pi/180.    
                            ip_high = 0.039*np.pi/180.
                            ip_low  = 0.042*np.pi/180.
                        elif pl_loc=='HD89345b':
                            ip_mean = 87.68*np.pi/180.    
                            ip_high = 0.1*np.pi/180.
                            ip_low  = 0.1*np.pi/180.
                        elif pl_loc=='Kepler63b':
                            ip_mean = 87.806*np.pi/180.    
                            ip_high = 0.018*np.pi/180.
                            ip_low  = 0.019*np.pi/180.                            
                        elif pl_loc=='WASP107b':
                            ip_mean = 89.56*np.pi/180.    
                            ip_high = 0.078*np.pi/180.
                            ip_low  = 0.078*np.pi/180.  
                        elif pl_loc=='HIP41378d':
                            ip_mean = 89.80*np.pi/180.     
                            ip_high = 0.02*np.pi/180.
                            ip_low  = 0.02*np.pi/180.  
                        elif pl_loc=='WASP47d':
                            ip_mean = 89.55*np.pi/180.     
                            ip_high = 0.30*np.pi/180.
                            ip_low  = 0.27*np.pi/180.  
                        elif pl_loc=='WASP166b':
                            ip_mean = 87.95*np.pi/180.     
                            ip_high = 0.59*np.pi/180.
                            ip_low  = 0.62*np.pi/180.  
                            
                        n_chain = len(merged_chain[:,0])
                        inclin_rad_chain = gen_hrand_chain(ip_mean,ip_low,ip_high,n_chain)
                    
                    for inst in lamb_chain:
                        for vis in lamb_chain[inst]:
    
                            PsiN_chain=np.arccos(np.sin(istarN_chain)*np.cos(lamb_chain[inst][vis])*np.sin(inclin_rad_chain) + np.cos(istarN_chain)*np.cos(inclin_rad_chain))*180./np.pi
                            PsiS_chain=np.arccos(np.sin(istarS_chain)*np.cos(lamb_chain[inst][vis])*np.sin(inclin_rad_chain) + np.cos(istarS_chain)*np.cos(inclin_rad_chain))*180./np.pi
                              
                            #Combined Psi for istar and pi-istar, assumed equiprobable
                            Psi_chain = 0.5*( PsiN_chain + PsiS_chain   ) 
                        
                            merged_chain=np.concatenate((merged_chain,PsiN_chain[:,None]),axis=1)   
                            merged_chain=np.concatenate((merged_chain,PsiS_chain[:,None]),axis=1)   
                            merged_chain=np.concatenate((merged_chain,Psi_chain[:,None]),axis=1)   
                
                if lambda_rad_pl in fixed_args['genpar_instvis']:  
                    for inst in fixed_args['genpar_instvis'][lambda_rad_pl]:
                        for vis in fixed_args['genpar_instvis'][lambda_rad_pl][inst]:
                            fixed_args['var_par_list']=np.concatenate((fixed_args['var_par_list'],['PsiN__pl'+pl_loc+'__IS'+inst+'_VS'+vis,'PsiS__pl'+pl_loc+'__IS'+inst+'_VS'+vis]))
                            fixed_args['var_par_names']=np.concatenate((fixed_args['var_par_names'],[pl_loc+'_$\psi_{N}$['+inst+']('+vis+')',pl_loc+'_$\psi_{S}$',pl_loc+'_$\psi$['+inst+']('+vis+')']))          
                else:
                    fixed_args['var_par_list']=np.concatenate((fixed_args['var_par_list'],['PsiN__pl'+pl_loc,'PsiS__pl'+pl_loc]))
                    fixed_args['var_par_names']=np.concatenate((fixed_args['var_par_names'],[pl_loc+'_$\psi_{N}$',pl_loc+'_$\psi_{S}$',pl_loc+'_$\psi$']))   
             
        #-------------------------------------------------
            
        #Add argument of ascending node
        #    - Om = np.arctan( -sin(lambda)*tan(ip) )
        #    - must be done before modification of ip and lambda chains
        if 'om' in modif_list:
            print('     + Add argument of ascending node')
            for pl_loc in fixed_args['lambda_rad_pl']:
                lambda_rad_pl = 'lambda_rad__pl'+pl_loc
                if ('inclin_rad__'+pl_loc not in fixed_args['var_par_list']):ip_loc=fixed_args['planets_params'][pl_loc]['inclin_rad'] 
                if fit_dic['fit_mod']=='chi2':  
                    if ('inclin_rad__pl'+pl_loc in fixed_args['var_par_list']):ip_loc=p_final['inclin_rad__pl'+pl_loc]
                    if lambda_rad_pl in fixed_args['genpar_instvis']:
                        for inst in fixed_args['genpar_instvis'][lambda_rad_pl]:
                            for vis in fixed_args['genpar_instvis'][lambda_rad_pl][inst]:
                                p_final['Omega__pl'+pl_loc+'__IS'+inst+'_VS'+vis]=np.arctan( -np.sin(p_final[lambda_rad_pl+'__IS'+inst+'_VS'+vis])*np.tan(ip_loc) )*180./np.pi
                                fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[np.nan],[np.nan]]))
                    else: 
                        p_final['Omega__pl'+pl_loc]=np.arctan( -np.sin(p_final[lambda_rad_pl])*np.tan(ip_loc) )*180./np.pi
                        fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[np.nan],[np.nan]]))
                elif fit_dic['fit_mod']=='mcmc': 
                    if ('inclin_rad__pl'+pl_loc in fixed_args['var_par_list']):ip_loc=merged_chain[:,np_where1D(fixed_args['var_par_list']=='inclin_rad__pl'+pl_loc)]
                    if lambda_rad_pl in fixed_args['genpar_instvis']:
                        lamb_chain={}
                        for inst in fixed_args['genpar_instvis'][lambda_rad_pl]:
                            lamb_chain[inst] = {}
                            for vis in fixed_args['genpar_instvis'][lambda_rad_pl][inst]:      
                                lamb_chain[inst][vis] = np.squeeze(merged_chain[:,np_where1D(fixed_args['var_par_list']==lambda_rad_pl+'__IS'+inst+'_VS'+vis)]  ) 
                    else:
                        lamb_chain = {'':{'':np.squeeze(merged_chain[:,np_where1D(fixed_args['var_par_list']==lambda_rad_pl)]  )}} 
                    for inst in lamb_chain:
                        for vis in lamb_chain[inst]:   
                            chain_loc=np.arctan( -np.sin(lamb_chain[inst][vis])*np.tan(ip_loc) )*180./np.pi
                            merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)   

                if lambda_rad_pl in fixed_args['genpar_instvis']:  
                    for inst in fixed_args['genpar_instvis'][lambda_rad_pl]:
                        for vis in fixed_args['genpar_instvis'][lambda_rad_pl][inst]:
                            fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'Omega__pl'+pl_loc+'__IS'+inst+'_VS'+vis)
                            fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],pl_loc+'_$\Omega$['+inst+']('+vis+')') 
                else:
                    fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'Omega__pl'+pl_loc)
                    fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],pl_loc+'_$\Omega$')               

        #-------------------------------------------------
            
        #Add impact parameter
        #    - b = aRs*cos(ip)
        if ('b' in modif_list) and (fixed_args['fit_orbit']):
            print('     + Adding impact parameter')
            for pl_loc in fixed_args['lambda_rad_pl']:
                if ('inclin_rad__pl'+pl_loc not in fixed_args['var_par_list']):ip_loc=fixed_args['planets_params'][pl_loc]['inclin_rad'] 
                else:iip=np_where1D(fixed_args['var_par_list']=='inclin_rad__pl'+pl_loc)[0]
                if ('aRs__pl'+pl_loc not in fixed_args['var_par_list']):aRs_loc=fixed_args['planets_params'][pl_loc]['aRs']   
                else:iaRs = np_where1D(fixed_args['var_par_list']=='aRs__pl'+pl_loc)[0]
                if fit_dic['fit_mod']=='chi2':  
                    #    - db = b*sqrt( (daRs/aRs)^2 + (dcos(ip)/cos(ip))^2 )
                    #      dcos(ip) = sin(ip)*dip 
                    #    - db = b*sqrt( (daRs/aRs)^2 + (tan(ip)*dip)^2 )                
                    if ('inclin_rad__pl'+pl_loc in fixed_args['var_par_list']):
                        ip_loc=p_final['inclin_rad__pl'+pl_loc]
                        dip_loc = fit_dic['sig_parfinal_err']['1s'][:,iip] 
                    else:dip_loc = 0.
                    if ('aRs__pl'+pl_loc in fixed_args['aRs__pl'+pl_loc]):
                        aRs_loc=p_final['aRs__pl'+pl_loc] 
                        daRs = fit_dic['sig_parfinal_err']['1s'][:,iaRs]   
                    else:daRs=0.
                    p_final['b']=aRs_loc*np.abs(np.cos(ip_loc))
                    sig_loc=p_final['b']*np.sqrt( (daRs/aRs_loc)**2. + (np.tan(ip_loc)*dip_loc)**2. )  
                    fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]])    )            
                    
                elif fit_dic['fit_mod']=='mcmc':             
                    if ('inclin_rad__pl'+pl_loc in fixed_args['var_par_list']):ip_loc=merged_chain[:,iip]
                    if ('aRs__pl'+pl_loc in fixed_args['var_par_list']):aRs_loc=merged_chain[:,iaRs]          
                    chain_loc=aRs_loc*np.abs(np.cos(ip_loc))
                    merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)   
                fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],pl_loc+'_b')
                fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],pl_loc+'_b')
    
        #-------------------------------------------------            
            
        #Orbital inclination
        #    - convert ip[rad] to ip[deg]
        if ('ip' in modif_list) and (fixed_args['fit_orbit']):
            print('     + Converting ip in degrees')
            for pl_loc in fixed_args['lambda_rad_pl']:
                iip=np_where1D(fixed_args['var_par_list']=='inclin_rad__pl'+pl_loc)
                if fit_dic['fit_mod']=='chi2':   
                    p_final['ip_deg']=p_final['inclin_rad__pl'+pl_loc]*180./np.pi                     
                elif fit_dic['fit_mod']=='mcmc':                      
                    merged_chain[:,iip]*=180./np.pi   
                    
                    #Fold ip over 0-90
                    print('     + Folding ip')
                    ip_temp=np.squeeze(merged_chain[:,iip])
                    w_gt_90=(ip_temp > 90.)
                    if True in w_gt_90:
                        merged_chain[w_gt_90,iip]=np.mod(ip_temp[w_gt_90],90.)
                    w_lt_0=(ip_temp < 0.)
                    if True in w_lt_0:
                        i_mod=npint(np.abs(ip_temp[w_lt_0])/90.)+1.
                        merged_chain[w_lt_0,iip] = ip_temp[w_lt_0]+i_mod*90.                    
                    
                fixed_args['var_par_list'][iip]='ip_deg__pl'+pl_loc            
                fixed_args['var_par_names'][iip]='i$_\mathrm{p}$['+pl_loc+']($^{\circ}$)'  
    
        #-------------------------------------------------            
            
        #Convert lambda[rad] to lambda[deg]
        if 'lambda_deg' in modif_list:  
            print('     + Converting lambda in degrees')       
            for pl_loc in fixed_args['lambda_rad_pl']:
                lambda_rad_pl = 'lambda_rad__pl'+pl_loc
                lambda_deg_pl = 'lambda_deg__pl'+pl_loc
                if fit_dic['fit_mod']=='chi2': 
                    def sub_func(lamb_name,new_lamb_name,new_lamb_name_txt):
                        mid_shift = -180.
                        ilamb=np_where1D(fixed_args['var_par_list']==lamb_name)                    
                        lambda_temp = (p_final[lamb_name]*180./np.pi) - mid_shift
                        if lambda_temp>360.:lambda_temp = np.mod(lambda_temp,360.) + mid_shift
                        elif lambda_temp<0.:
                            i_mod=npint(np.abs(lambda_temp)/360.)+1.
                            lambda_temp += i_mod*360.+mid_shift  
                        else:lambda_temp += mid_shift
                        p_final[new_lamb_name] = lambda_temp
                        sig_temp  = fit_dic['sig_parfinal_err']['1s'][0,ilamb]*180./np.pi 
                        fit_dic['sig_parfinal_err']['1s'][:,ilamb] = sig_temp  
                        fixed_args['var_par_list'][ilamb]=new_lamb_name       
                        fixed_args['var_par_names'][ilamb]= new_lamb_name_txt                           
                        return None
                    
                    if lambda_rad_pl in fixed_args['genpar_instvis']:
                        for inst in fixed_args['genpar_instvis'][lambda_rad_pl]:
                            for vis in fixed_args['genpar_instvis'][lambda_rad_pl][inst]:
                                sub_func(lambda_rad_pl+'__IS'+inst+'_VS'+vis,lambda_deg_pl+'__IS'+inst+'_VS'+vis,'$\lambda$['+pl_loc+']['+inst+']('+vis+')($^{\circ}$)')
                    else:sub_func(lambda_rad_pl,lambda_deg_pl,'$\lambda$['+pl_loc+']($^{\circ}$)')   
                
                    
                elif fit_dic['fit_mod']=='mcmc':   
                    def sub_func(lamb_name,new_lamb_name,new_lamb_name_txt):  
                        ilamb=np_where1D(fixed_args['var_par_list']==lamb_name)                     
                        merged_chain[:,ilamb]*=180./np.pi  
            
                        #Fold lambda over x+[-180;180]
                        #    - choose final range so that the best-fit is not close to a boundary
                        #    - we want lambda in x+[-180;180] ie lambda-x+180 in 0;360
                        #      we fold over 0;360 and then get back to the final range
                        x_mid=np.median(merged_chain[:,ilamb])
                        # x_mid=110. 
                        if pl_loc=='GJ436_b':x_mid=70.
                        elif pl_loc=='HD3167_b':x_mid=80.
                        elif pl_loc=='HD3167_c':x_mid=-110.                
                        elif pl_loc=='HAT_P3b':x_mid=0.               
                        elif pl_loc=='K2_105b':x_mid=-80.         
                        elif pl_loc=='HD89345b':x_mid=70. 
                        elif pl_loc=='WASP107b':x_mid=-160. 
                        elif pl_loc=='Kepler25c':x_mid=80. 
                        elif pl_loc=='WASP156b':x_mid=80. 
                        elif pl_loc=='55Cnc_e' and ('20140226' in lamb_name):
                            x_mid = -100.
        
                        print('     + Folding '+lamb_name+' around ',x_mid)
                        mid_shift = x_mid-180.
                        lambda_temp=np.squeeze(merged_chain[:,ilamb])-mid_shift
                        w_gt_360=(lambda_temp > 360.)
                        if True in w_gt_360:
                            merged_chain[w_gt_360,ilamb]=np.mod(lambda_temp[w_gt_360],360.)+mid_shift
                        w_lt_0=(lambda_temp < 0.)
                        if True in w_lt_0:
                            i_mod=npint(np.abs(lambda_temp[w_lt_0])/360.)+1.
                            merged_chain[w_lt_0,ilamb] = lambda_temp[w_lt_0]+i_mod*360.+mid_shift
                
                        fixed_args['var_par_list'][ilamb]=new_lamb_name
                        fixed_args['var_par_names'][ilamb]=new_lamb_name_txt       
                        
                        return None
    
                    if lambda_rad_pl in fixed_args['genpar_instvis']:
                        for inst in fixed_args['genpar_instvis'][lambda_rad_pl]:
                            for vis in fixed_args['genpar_instvis'][lambda_rad_pl][inst]:
                                sub_func(lambda_rad_pl+'__IS'+inst+'_VS'+vis,lambda_deg_pl+'__IS'+inst+'_VS'+vis,'$\lambda$['+pl_loc+']['+inst+']('+vis+')($^{\circ}$)')
                    else:sub_func(lambda_rad_pl,lambda_deg_pl,'$\lambda$['+pl_loc+']($^{\circ}$)')       
    
    
        #-------------------------------------------------            
         
        #Retrieve zeroth order CB coefficient
        #    - independent of visits or planets
        if 'c0' in modif_list: 
            print('     + Adding c0(CB)')
            if fit_dic['fit_mod']=='chi2': 
                theo_par = sub_calc_plocc_prop(['achrom'],None,['c0_CB'],None,fixed_args['system_prop'],p_final,None,None,None,0)            
                p_final['c0_CB']=theo_par['c0_CB'][0,:]
                sig_loc=np.nan  
                fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]) )     
                          
            elif fit_dic['fit_mod']=='mcmc':   
                chain_loc=np.empty(0,dtype=float)            
                p_final_loc={}
                p_final_loc.update(fixed_args['fixed_par_val'])
                for istep in range(fit_dic['nsteps_pb_all']): 
                    for ipar,par in enumerate(fixed_args['var_par_list']):
                        p_final_loc[par]=merged_chain[istep,ipar]     
                        if len(fixed_args['linked_par_expr'])>0:exec(str(par)+'='+str(p_final_loc[par]))
                    for par in fixed_args['linked_par_expr']:
                        p_final_loc[par]=eval(fixed_args['linked_par_expr'][par])
                    theo_par = sub_calc_plocc_prop(['achrom'],None,['c0_CB'],None,fixed_args['system_prop'],p_final,None,None,None,0)                      
                    chain_loc=np.append(chain_loc,theo_par['c0_CB'][0,:])
                merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)   
            fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'c0_CB')
            fixed_args['var_par_names']+=['CB$_{0}$']           
    
    
        #-------------------------------------------------            
         
        #Convert CB coefficients from km/s to m/s
        if 'CB_ms' in modif_list: 
            print('     + Converting CB coefficients to m/s') 
            ipar_mult_loc=[]
            for ipar_name in ['c0_CB','c1_CB','c2_CB','c3_CB']:
                ipar_loc=np_where1D(fixed_args['var_par_list']==ipar_name)
                if len(ipar_loc)>0:ipar_mult_loc+=[ipar_loc]
                if ipar_name=='c0_CB':fixed_args['var_par_names'][ipar_loc]='CB$_{0}$ (m s$^{-1}$)'
                if ipar_name=='c1_CB':fixed_args['var_par_names'][ipar_loc]='CB$_{1}$ (m s$^{-1}$)'
                if ipar_name=='c2_CB':fixed_args['var_par_names'][ipar_loc]='CB$_{2}$ (m s$^{-1}$)'
                if ipar_name=='c3_CB':fixed_args['var_par_names'][ipar_loc]='CB$_{3}$ (m s$^{-1}$)'             
                if fit_dic['fit_mod']=='chi2': 
                    p_final[ipar_name]*=1e3 
                    fit_dic['sig_parfinal_err']['1s'][:,ipar_loc]*=1e3  
                elif fit_dic['fit_mod']=='mcmc':  
                    merged_chain[:,ipar_loc]*=1e3 
    
        #-------------------------------------------------            
            
        #Convert FWHM and contrast of true intrinsic stellar profiles into values for observed profiles
        #    - only for constant FWHM and contrast with no variation in mu
        #    - for double-gaussian profiles the values can be converted into the 'true' contrast and FWHM, either of the intrinsic or of the observed profile
        if ('CF0_meas_add' in modif_list) or ('CF0_meas_conv' in modif_list):  
            print('     + Converting intrinsic ctrst and FWHM to measured values') 
            merged_chain = conv_CF_intr_meas(modif_list,fixed_args['inst_list'],fixed_args['inst_vis_list'],fixed_args,merged_chain,gen_dic,p_final,fit_dic,fit_prop_dic)
        if ('CF0_DG_add' in modif_list) or ('CF0_DG_conv' in modif_list):  
            print('     + Converting DG ctrst and FWHM to true intrinsic values') 
            merged_chain = conv_CF_intr_meas(modif_list,fixed_args['inst_list'],fixed_args['inst_vis_list'],fixed_args,merged_chain,gen_dic,p_final,fit_dic,fit_prop_dic)
            
    #---------------------------------------------------------------            
    #Process:
    #    - best-fit values and confidence interval for the final parameters, potentially modified
    #    - correlation diagram plot
    if fit_dic['fit_mod']=='mcmc':   
        p_final=postMCMCwrapper_2(fit_dic,fixed_args,merged_chain)

    #----------------------------------------------------------
    #Save derived parameters
    #----------------------------------------------------------
    save_fit_results('derived',fixed_args,fit_dic,fit_dic['fit_mod'],p_final)

    #Close save file
    fit_dic['file_save'].close() 

    print('     ----------------------------------')    
  

    return None












'''
Wrap-up function to process intrinsic stellar profiles
'''
def fit_intr_funcs(glob_fit_dic,system_param,theo_dic,data_dic,gen_dic,plot_dic,coord_dic):
        
    #Fitting stellar surface properties with a linked model
    if gen_dic['fit_loc_prop']:
        fit_CCFintr_prop(glob_fit_dic['PropLoc'],gen_dic,system_param,theo_dic,plot_dic,coord_dic,data_dic)    
    
    #Fitting intrinsic stellar lines with joined model
    if gen_dic['fit_IntrProf']:
        fit_IntrProf_all(data_dic,gen_dic,system_param,glob_fit_dic['IntrProf'],theo_dic,plot_dic,coord_dic)    

    #Fitting local stellar lines with joined model, including spots in the fitted parameters
    if gen_dic['fit_ResProf']:
        fit_CCFRes_all(data_dic,gen_dic,system_param,glob_fit_dic['ResProf'],theo_dic,plot_dic,coord_dic)    

    return None


'''
Wrap-up function to process atmospheric profiles
'''
def fit_atm_funcs(PropAtm_fit_dic,gen_dic):

    #Fitting intrinsic stellar CCFs with joined model
    if gen_dic['fit_atm_all']:
        fit_CCFatm_all()    
        
    #Fitting stellar surface properties with a linked model
    if gen_dic['fit_atm_prop']:
        fit_CCFatm_prop()    
    

    return None


'''
Wrap-up function to perform the model fit and calculation
'''
def common_fit_rout(rout_mode,fit_dic,fixed_args,fit_prop_dic,gen_dic,data_dic,theo_dic):

    #Model parameters
    p_start = Parameters()      

    #------------------------------------------------------------------------------------------------------------------------------------------------

    #Model parameters
    #    - we define here parameters common to the different fit routines, but they can be updated and specific parameters defined in the routines later
    #
    #------------------------------------------------------------
    #    - each parameter in p_start is defined by its name, guess/fixed value, fit flag, lower and upper boundary of explored range, expression 
    #    - some parameters can be left undefined and will be set to default values:
    # + alpha, beta : 0
    # + CB coefficients: 0
    # + cos_istar : 0.
    # + aRs, inclin_rad : values defined in ANTARESS_settings 
    #    - parameters are ordered
    #    - use 'expression' to set properties to the same value :
    # define par1, then define par2 and set it to expr='par1'
    #      do not include in the expression of a parameter another parameter linked with an expression
    #    - all parameter options are valid for both chi2 and mcmc, except for boundary conditions that must be defined differently for the mcmc
    #    - model parameters are :
    # + lambda_rad : sky-projected obliquity (rad) 
    # + veq : true equatorial rotational velocity (km/s)    
    # + cos(istar) : inclination of stellar spin axis (rad)
    #                for the fit we use cos(istar), natural variable in the model and linked with istar through bijection in the allowed range
    # + alpha_rot, beta_rot : parameters of DR law
    # + c1, c2 and c3 : coefficients of the mu-dependent CB velocity polynomial (km/s) 
    # + aRs, inclin_rad: in some cases these properties can be better constrained by the fit to the local RVs
    #                    since they control the orbital trajectory of the planet, fitting these parameters will make the code recalculate the coordinates of the planet-occulted regions    
    #      for more details see calc_plocc_prop()  
    #    - parameters specific to a given planet should be defined as 'parname__plX', where X is the name of the planet used throughout the pipeline
    #    - the model uses the nominal RpRs and LD coefficients, which must be suited to the spectral band from which the local stellar properties were derived
    #------------------------------------------------------------
    #Priors on variable model parameters
    #    - see MCMC_routines > ln_prior_func() for the possible priors
    #    - priors must be defined for all variable parameters 
    #    - there is a general, uniform prior on cos(istar) between -1 and 1 that allows us to limit istar in 0:180 
    # since there is a bijection between istar and cos(istar) over this range
    #    - lambda is defined over the entire angular space, however it might need to be limited to a fixed range to prevent the 
    # mcmc to switch from one best-fit region defined in [-180;180] to the next in x+[-180;180]. By default we take -2*180:2*180. 
    #      this range might need to be shifted if the best-fit is at +-180
    #      the posterior distribution is folded over x+[-180;180] in the end
    #    - we use the same approach for the orbital inclination, which must be limited in the post-processing to the range [0-90]°
    #
    #------------------------------------------------------------
    #Starting points of walkers for variable parameters
    #    - they must be set to different values for each walker
    #    - for simplicity we define randomly for each walker 
    #    - parameters must be defined in the same order as in 'p_start'
    fixed_args,p_start = init_custom_DI_par(fixed_args,gen_dic,data_dic['DI']['system_prop'],theo_dic,fixed_args['system_param']['star'],p_start,[0.,None,None],True)
    fixed_args = init_custom_DI_prof(fixed_args,gen_dic,data_dic['DI']['system_prop'],theo_dic,fixed_args['system_param']['star'],p_start)
                        
    #Condition to calculate CB
    if ('c1_CB' in fit_prop_dic['mod_prop']) or ('c2_CB' in fit_prop_dic['mod_prop'])  or ('c3_CB' in fit_prop_dic['mod_prop']):fixed_args['par_list']+=['CB_RV']

    #Parameter initialization
    p_start = par_formatting(p_start,fit_prop_dic['mod_prop'],fit_prop_dic['priors'],fit_dic,fixed_args,'','')

    #Determine if orbital and light curve properties are fitted or whether nominal values are used
    #    - this depends on whether parameters required to calculate coordinates of planet-occulted regions are fitted  
    par_orb=['inclin_rad','aRs','lambda_rad']
    par_LC=['RpRs']    
    for par in par_orb+par_LC:fixed_args[par+'_pl']=[]
    fixed_args['fit_orbit']=False
    fixed_args['fit_RpRs']=False
    for par in p_start:
        
        #Check if rootname of orbital/LC properties is one of the parameters left free to vary for a given planet    
        #    - if so, store name of planet for this property
        for par_check in par_orb:
            if (par_check in par) and (p_start[par].vary):
                if ('__IS' in par):pl_name = (par.split('__pl')[1]).split('__IS')[0]                  
                else:pl_name = (par.split('__pl')[1])  
                fixed_args[par_check+'_pl']+= [pl_name]
                fixed_args['fit_orbit']=True 
        for par_check in par_LC:
            if (par_check in par) and (p_start[par].vary):
                fixed_args[par_check+'_pl'] += [par.split('__pl')[1]]
                fixed_args['fit_RpRs']=True 

    #Unique list of planets with variable properties                
    for par in par_orb:fixed_args[par+'_pl'] = list(np.unique(fixed_args[par+'_pl']))
    for par in par_LC:fixed_args[par+'_pl'] = list(np.unique(fixed_args[par+'_pl']))
    fixed_args['b_pl'] = list(np.unique(fixed_args['inclin_rad_pl']+fixed_args['aRs_pl']))

    # Stage Théo : on utilise des valeurs par défaut pour les propriétés suivantes (si elles ne sont pas fittées) : 
    #    - CB coef, 'alpha_rot', 'beta_rot', 'cos_istar' , 'veq' : les valeurs définies dans fixed_args['star_params']
    #    - aRs, inclin_rad, lambda_rad : les valeurs définies dans fixed_args['planet_params']
    #    - 'rv' : la vitesse systémique des CCF_DI, pas important, fixée à 0 dans tous les cas. 
    #    - 'slope' : demandée par Di_prof_from_intr, fixée à 0 si pas fitté
    #
    #           -> Philosophie : mettre dans param tout ce qui est susceptible d'être fitté, en jouant avec le champ 'vary'
    #           -> param a la même tête quelque soit le fit mené, aux champs 'vary' près.
        
    if rout_mode == 'ResProf'    or    (rout_mode == 'IntrProf' and fit_prop_dic['use_version2'])  :

        for par in par_orb: 
            if par+'__pl'+fixed_args['pl_loc'] not in p_start : 
                p_start.add(par+'__pl'+fixed_args['pl_loc'] ,value=fixed_args['planet_params'][par], vary=False , min=None , max=None)
            
   
            
    #Fit initialization
    init_fit(fit_dic,fixed_args,p_start,model_par_names(),fit_prop_dic)     
    merged_chain = None
    
    ########################################################################################################   

    #Fit by chi2 minimization
    if fit_dic['fit_mod']=='chi2':
        print('     ----------------------------------')
        print('     Chi2 calculation')   
        p_final = fit_minimization(ln_prob_func_lmfit,p_start,fixed_args['x_val'],fixed_args['y_val'],fixed_args['cov_val'],fixed_args['fit_func'],verbose=fit_prop_dic['verbose'],fixed_args=fixed_args)[2]
        print('     ----------------------------------')
 
    ########################################################################################################    
    #Fit par emcmc 
    elif fit_prop_dic['fit_mod']=='mcmc':  
        print('     ----------------------------------')
        print('     MCMC calculation')

        #MCMC walkers setting 
        fit_dic['nwalkers'] = fit_prop_dic['mcmc_set']['nwalkers']
        fit_dic['nsteps'] = fit_prop_dic['mcmc_set']['nsteps']
        fit_dic['nburn'] = fit_prop_dic['mcmc_set']['nburn']    

        #Run MCMC
        if fit_prop_dic['mcmc_run_mode']=='use':
            print('     Applying MCMC') 

            #Complex prior function
            if (len(fixed_args['prior_func'])>0):fixed_args['global_ln_prior_func']=global_ln_prior_func
            
            #Call to MCMC
            walker_chains=call_MCMC(fit_prop_dic['nthreads'],fixed_args,fit_dic,run_name=fit_dic['run_name'])
               
        #---------------------------------------------------------------  
       
        #Reuse MCMC
        elif fit_prop_dic['mcmc_run_mode']=='reuse':
            print('     Retrieving MCMC') 
            if len(fit_prop_dic['mcmc_reuse'])==0:
                walker_chains=np.load(fit_dic['save_dir']+'raw_chains_walk'+np.str(fit_dic['nwalkers'])+'_steps'+np.str(fit_dic['nsteps'])+fit_dic['run_name']+'.npz')['walker_chains']  #(nwalkers, nsteps, n_free)
            else:
                walker_chains = np.empty([fit_dic['nwalkers'],0,fit_dic['n_free'] ],dtype=float)
                fit_dic['nsteps'] = 0
                fit_dic['nburn'] = 0
                for mcmc_path,nburn in zip(fit_prop_dic['mcmc_reuse']['paths'],fit_prop_dic['mcmc_reuse']['nburn']):
                     walker_chains_loc=np.load(mcmc_path)['walker_chains'][:,nburn::,:] 
                     fit_dic['nsteps']+=(walker_chains_loc.shape)[1]
                     walker_chains = np.append(walker_chains,walker_chains_loc,axis=1)
                    
                    
    
        #Excluding parts of the chains
        if fit_dic['exclu_walk']:
            print('     Excluding walkers manually')
            
            #Joined fit
            if gen_dic['fit_IntrProf']:
            
                # ipar_loc=np_where1D(fixed_args['var_par_list']=='lambda_rad__plHD3167_c')
                # # wgood=np_where1D(np.min(walker_chains[:,:,ipar_loc],axis=1)>-1.)
                # # wgood=np_where1D(np.median(walker_chains[:,:,ipar_loc],axis=1)>-2.5)
                # wgood=np_where1D(np.median(walker_chains[:,:,ipar_loc],axis=1)<-1.)
                # wgood=np_where1D(np.median(walker_chains[:,:,ipar_loc],axis=1)<-1.5)
    
                # ipar_loc=np_where1D(fixed_args['var_par_list']=='lambda_rad__plTOI858b')
                # wgood=np_where1D(np.median(walker_chains[:,:,ipar_loc],axis=1)<5.)
    
                if gen_dic['star_name'] == 'GJ436':
                    # wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='amp_l2c__ISESPRESSO_VS20190228')],axis=1)<1.) &\
                    #                  (np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='amp_l2c__ISESPRESSO_VS20190429')],axis=1)<1.)&\
                    #                  (np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='FWHM_ord0__ISESPRESSO_VS20190429')],axis=1)>2.))
    
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='amp_l2c__IS__VS_')],axis=1)<1.)&\
                                      (np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='FWHM_ord0__ISESPRESSO_VS20190228')],axis=1)>2.)&\
                                      (np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='FWHM_ord0__ISESPRESSO_VS20190429')],axis=1)>2.))

                elif gen_dic['star_name'] == 'HAT_P3':
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='FWHM_ord0__ISHARPN_VS20200130')],axis=1)>0.))

                elif gen_dic['star_name'] == 'HAT_P33':
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='FWHM_ord0__ISHARPN_VS20191204')],axis=1)>0.))


                elif gen_dic['star_name'] == 'HAT_P49':
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='FWHM_ord0__ISHARPN_VS20200730')],axis=1)<20.))

                elif gen_dic['star_name'] == 'WASP107':
                    # wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='lambda_rad__plWASP107b')],axis=1)<0.))
                    # wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='lambda_rad__plWASP107b')],axis=1)>0.))
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='a_damp__ISCARMENES_VIS_VS20180224')],axis=1)<10.))

                elif gen_dic['star_name'] == 'HIP41378':
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='lambda_rad__plHIP41378d')],axis=1)<4.))

                elif gen_dic['star_name'] == 'WASP156':
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='a_damp__ISCARMENES_VIS_VS20190928')],axis=1)<2.))

                elif gen_dic['star_name'] == 'WASP166':
                    # wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='FWHM_ord0__IS__VS_')],axis=1)>5.))
                    # wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='ctrst_ord0__ISHARPS_VS_')],axis=1)<0.63) )
                    # wgood=np_where1D((np.min(walker_chains[:,250:750,np_where1D(fixed_args['var_par_list']=='veq')],axis=1)>5.) )
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='veq')],axis=1)>4.5) )
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='Rstar')],axis=1)<1.45) )

                elif gen_dic['star_name'] == 'Kepler25':
                    ilamb = np_where1D(fixed_args['var_par_list']=='lambda_rad__plKepler25c')
                    for iwalk in range(fit_dic['nwalkers']):
                        lambda_temp=np.squeeze(walker_chains[iwalk,:,ilamb])+np.pi
                        w_gt_360=(lambda_temp > 2.*np.pi)
                        if True in w_gt_360:
                            walker_chains[iwalk,w_gt_360,ilamb]=np.mod(lambda_temp[w_gt_360],2.*np.pi)-np.pi
                        w_lt_0=(lambda_temp < 0.)
                        if True in w_lt_0:
                            i_mod=npint(np.abs(lambda_temp[w_lt_0])/(2.*np.pi))+1.
                            walker_chains[iwalk,w_lt_0,ilamb] = lambda_temp[w_lt_0]+i_mod*2.*np.pi-np.pi
                        
                    wgood=np_where1D( (np.max(walker_chains[:,:,ilamb],axis=1)<80.*np.pi/180.) & (np.min(walker_chains[:,:,ilamb],axis=1)>-80.*np.pi/180.) )


                elif gen_dic['star_name'] == '55Cnc':
                    wgood=np_where1D((np.median(walker_chains[:,:,np_where1D(fixed_args['var_par_list']=='veq')],axis=1)>0.5) )







            #Surface RV fit
            if gen_dic['fit_loc_prop']:
    
                ipar_loc=np_where1D(fixed_args['var_par_list']=='lambda_rad__plGJ436_b')
                wgood=np_where1D(np.min(walker_chains[:,:,ipar_loc],axis=1)>-1.)    
    
                # ipar_loc=np_where1D(fixed_args['var_par_list']=='veq')
                # wgood=np_where1D(walker_chains[:,-1,ipar_loc]>2.)                    
                # ipar_loc=np_where1D(fixed_args['var_par_list']=='cos_istar')
                # wgood=np_where1D(walker_chains[:,-1,ipar_loc]<0.)   
    
                ipar_loc=np_where1D(fixed_args['var_par_list']=='inclin_rad__plGJ436_b')
                wgood=np_where1D(walker_chains[:,-1,ipar_loc]<=0.5*np.pi)  


            print('   ',len(wgood),' walkers kept / ',fit_dic['nwalkers'])
            walker_chains=np.take(walker_chains,wgood,axis=0)     
            fit_dic['nwalkers']=len(wgood)  
               
        #------------------------------------------------------------------------------------------------            
 
        #Processing:
        #    - best-fit parameters for model calculation
        #    - 1-sigma and envelope samples for plot
        #    - plot of model parameter chains
        #    - save file 
        p_final,merged_chain,par_sample_sig1,par_sample=postMCMCwrapper_1(fit_dic,fixed_args,walker_chains,fit_prop_dic['nthreads'],fixed_args['par_names'])

    ########################################################################################################  
    #No fit is performed: guess parameters are kept
    else:
        print('     ----------------------------------')
        print('     Fixed model')
        p_final = deepcopy(p_start)   
    
    ########################################################################################################      

    #Merit values     
    p_final=fit_merit(p_final,fixed_args,fit_dic,fit_prop_dic['verbose'])                
    
    
    return merged_chain,p_final







    
 
    



'''
Initialization common to the different fit routines
'''
def init_joined_routines(gen_dic,system_param,theo_dic,data_dic,save_dir,fit_prop_dic):


    #Fit dictionary
    fit_dic={
        'fit_mod':fit_prop_dic['fit_mod'],
        'uf_bd':{},
        'nx_fit':0,
        'run_name':'_'+gen_dic['main_pl_text'],
        'save_dir' : save_dir+fit_prop_dic['fit_mod']+'/'}
    
    #--------------------------------------------------------------------------------

    #Arguments to be passed to the fit function
    fixed_args={
            
        #Global model properties
        'fit_star_grid':False,
        'DI_grid':False,
        'coord_line':fit_prop_dic['dim_fit'],
        'pol_mode':fit_prop_dic['pol_mode'],
        'theo_dic':{'Ssub_Sstar_pl':theo_dic['Ssub_Sstar_pl'],'x_st_sky_grid_pl':theo_dic['x_st_sky_grid_pl'],'y_st_sky_grid_pl':theo_dic['y_st_sky_grid_pl'],'nsub_Dpl':theo_dic['nsub_Dpl'],'d_oversamp':theo_dic['d_oversamp']},

        #Fit parameters
        'par_list':[]
        }

    return fixed_args,fit_dic










'''
Routine to fit intrinsic stellar profiles with a joined model
    - we use simple analytical profiles to describe the profiles, or theoretical models
    - positions of the profiles along the transit chord are linked by the stellar surface RV model
    - shapes of the profiles, when analytical, are linked across the transit chord by polynomial laws as a function of a chosen dimension
      polynomial coefficients can depend on the visit and their associated instrument, to account for possible variations in the line shape between visits
    - stellar line profiles are defined before instrumental convolution, so that data from all instruments and visits can be fitted together
'''
def fit_IntrProf_all(data_dic,gen_dic,system_param,fit_prop_dic,theo_dic,plot_dic,coord_dic):
    print('   > Fitting joined intrinsic stellar CCFs')

    #Path for file saving/retrieving
    save_dir=gen_dic['save_data_dir']+'/Joined_fits/Intr_prof/' 

    #Initializations
    fixed_args,fit_dic = init_joined_routines(gen_dic,system_param,theo_dic,data_dic,save_dir,fit_prop_dic)

    ######################################################################################################## 

    #Arguments to be passed to the fit function
    fixed_args.update({
        'func_prof' :{}, 
        'func_prof_name' : fit_prop_dic['func_prof_name'],  
        'prior_func':fit_prop_dic['prior_func'],  
        'cen_bins':{},
        'dcen_bins':{},
        'cond_fit':{},
        'cond_def':{},
        'flux':{},
        'cov':{},
        'nexp_fit':0,
        'nexp_fit_all':{},
        'FWHM_inst':{},
        'inst_list':[],
        'inst_vis_list':{},
        'mean_cont':{},
        'transit_pl':{},
        'idx_in_fit':{},
        'bin_mode':{},
        'n_pc':{},
        'chrom_mode':[data_dic['DI']['system_prop']['chrom_mode']],
        'conv2intr':True,
        'mac_mode':theo_dic['mac_mode'],
        })
    if len(fit_prop_dic['PC_model'])>0:
        fixed_args.update({
            'eig_res_matr':{},
            'nx_fit_PCout':0.,
            'n_free_PCout':0.,
            'chi2_PCout':0.,
            })
        
    #Stellar surface coordinate required to calculate spectral line profiles
    #    - other required properties are automatically added in the sub_calc_plocc_prop() function
    fixed_args['par_list']+=['line_prof']
    if fit_prop_dic['mode']=='glob_mod':
        if fit_prop_dic['dim_fit'] in ['abs_y_st','y_st2']:fixed_args['coord_line']='y_st'    
        else:fixed_args['coord_line']=fit_prop_dic['dim_fit']
        fixed_args['par_list']+=[fixed_args['coord_line']]
    else:fixed_args['par_list']+=['mu']

    #Activation of spectral conversion and resampling 
    cond_conv_st_prof_tab(gen_dic,fixed_args,data_dic[data_dic['instrum_list'][0]]['type'],fit_prop_dic)

    #Construction of the fit tables
    for par in ['coord_pl_fit','ph_fit']:fixed_args[par]={}
    for inst in np.intersect1d(data_dic['instrum_list'],list(fit_prop_dic['idx_in_fit'].keys())):          
        fit_prop_dic[inst]={}
        fixed_args['inst_list']+=[inst]
        fixed_args['inst_vis_list'][inst]=[]            
        for key in ['coord_pl_fit','ph_fit','transit_pl','cen_bins','edge_bins','dcen_bins','cond_fit','flux','cov','nexp_fit_all','mean_cont','cond_def','idx_in_fit','bin_mode','n_pc','dim_exp','ncen_bins']:fixed_args[key][inst]={}
        if len(fit_prop_dic['PC_model'])>0:fixed_args['eig_res_matr'][inst]={}
        cont_range = fit_prop_dic['cont_range'][inst]
        trim_range = fit_prop_dic['trim_range'][inst] if (inst in fit_prop_dic['trim_range']) else None 
        fixed_args['func_prof'][inst] = dispatch_func_prof(fit_prop_dic['func_prof_name'][inst])
        for vis in data_dic[inst]['visit_list']:
            
            #Identify whether visit is fitted over original or binned exposures
            #    - for simplicity we then use the original visit name in all fit dictionaries, as a visit will not be fitted at the same time in its original and binned format
            if (vis in fit_prop_dic['idx_in_fit'][inst]) and (len(fit_prop_dic['idx_in_fit'][inst][vis])>0):fixed_args['bin_mode'][inst][vis]=''
            elif (vis+'_bin' in fit_prop_dic['idx_in_fit'][inst]) and (len(fit_prop_dic['idx_in_fit'][inst][vis+'_bin'])>0):fixed_args['bin_mode'][inst][vis]='_bin'
            else:vis=None
         
            #Visit is fitted
            if vis is not None:              
                fit_prop_dic[inst][vis]={}
                fixed_args['transit_pl'][inst][vis]=data_dic[inst][vis]['transit_pl'] 
                fixed_args['dim_exp'][inst][vis] = data_dic[inst][vis]['dim_exp']
     
                #Binned/original visit
                if fixed_args['bin_mode'][inst][vis]=='_bin':
                    data_vis_bin = np.load(gen_dic['save_data_dir']+'Intrbin_data/'+inst+'_'+vis+'_phase_add.npz',allow_pickle=True)['data'].item()
                    n_in_tr = data_vis_bin['n_in_tr']
                else:
                    n_in_tr = data_dic[inst][vis]['n_in_tr']
               
                #Fitted exposures
                fixed_args['idx_in_fit'][inst][vis] = fit_prop_dic['idx_in_fit'][inst][vis+fixed_args['bin_mode'][inst][vis]]
                fixed_args['inst_vis_list'][inst]+=[vis]
                if fit_prop_dic['idx_in_fit'][inst][vis+fixed_args['bin_mode'][inst][vis]]=='all':fixed_args['idx_in_fit'][inst][vis]=range(n_in_tr)
                else:fixed_args['idx_in_fit'][inst][vis]=np.intersect1d(fixed_args['idx_in_fit'][inst][vis],range(n_in_tr))
                fixed_args['nexp_fit_all'][inst][vis]=len(fixed_args['idx_in_fit'][inst][vis])        

                #Trimming data
                #    - the trimming is applied to the common table, so that all processed profiles keep the same dimension after trimming
                iord_sel = 0
                if (trim_range is not None):
                    data_com = np.load(data_dic[inst][vis]['proc_com_data_paths']+'.npz',allow_pickle=True)['data'].item()  
                    idx_range_kept = np_where1D((data_com['edge_bins'][iord_sel,0:-1]>=trim_range[0]) & (data_com['edge_bins'][iord_sel,1::]<=trim_range[1]))
                    ncen_bins = len(idx_range_kept)
                    if ncen_bins==0:stop('Empty trimmed range')   
                else:
                    ncen_bins = data_dic[inst][vis]['nspec']
                    idx_range_kept = np.arange(ncen_bins,dtype=int)

                #Enable PC noise model
                if (inst in fit_prop_dic['PC_model']) and (vis in fit_prop_dic['PC_model'][inst]):
                    if fixed_args['bin_mode'][inst][vis]=='_bin':stop('PC correction not available for binned data')
                    data_pca = np.load(fit_prop_dic['PC_model'][inst][vis]['PC_path'],allow_pickle=True)['data'].item()  

                    #Accounting for PCA fit in merit values
                    #    - PC have been fitted in the PCA module to the out-of-transit data
                    if len(fit_prop_dic['PC_model'][inst][vis]['idx_out'])>0:
                        if fit_prop_dic['PC_model'][inst][vis]['idx_out']=='all':idx_pc_out =  np.intersect1d(gen_dic[inst][vis]['idx_out'],data_pca['idx_corr'])
                        else:idx_pc_out =  np.intersect1d(fit_prop_dic['PC_model'][inst][vis]['idx_out'],data_pca['idx_corr'])
                    
                        #Increment :
                        #    + number of fitted datapoints = number of pixels fitted per exposure in the PCA module
                        #    + number of free parameters = number of PC fitted in the PCA module
                        #    + chi2: chi2 for current exposure from the correction applied in the PCA module
                        for idx_out in idx_pc_out:
                            fixed_args['nx_fit_PCout']+=data_pca['nfit_tab'][idx_out]
                            if fit_prop_dic['PC_model'][inst][vis]['noPC']:
                                fixed_args['n_free_PCout']+=0. 
                                fixed_args['chi2_PCout']+=data_pca['chi2null_tab'][idx_out]                                   
                            else:
                                fixed_args['n_free_PCout']+=data_pca['n_pc']
                                fixed_args['chi2_PCout']+=data_pca['chi2_tab'][idx_out] 
                                
                    #Number of PC used for the fit
                    #    - set by the correction applied in the PCA module, for consistency
                    if fit_prop_dic['PC_model'][inst][vis]['noPC']:
                        fixed_args['n_pc'][inst][vis] = None
                    else:
                        fixed_args['n_pc'][inst][vis]=data_pca['n_pc']
                        fixed_args['eig_res_matr'][inst][vis] = {} 

                else:fixed_args['n_pc'][inst][vis] = None
                
                #Fit tables
                #    - models must be calculated over the full, continuous spectral tables to allow for convolution
                #      the fit is then performed on defined pixels only
                for key in ['dcen_bins','cen_bins','edge_bins','cond_fit','flux','cov','cond_def','ncen_bins']:fixed_args[key][inst][vis]=np.zeros(fixed_args['nexp_fit_all'][inst][vis],dtype=object)
                fit_prop_dic[inst][vis]['cond_def_fit_all']=np.zeros([fixed_args['nexp_fit_all'][inst][vis],ncen_bins],dtype=bool)
                fit_prop_dic[inst][vis]['cond_def_cont_all'] = np.zeros([fixed_args['nexp_fit_all'][inst][vis],ncen_bins],dtype=bool)     
                for isub,i_in in enumerate(fixed_args['idx_in_fit'][inst][vis]):
         
                    #Upload latest processed intrinsic data
                    if fixed_args['bin_mode'][inst][vis]=='_bin':
                        data_exp = np.load(gen_dic['save_data_dir']+'Intrbin_data/'+inst+'_'+vis+'_phase'+str(i_in)+'.npz',allow_pickle=True)['data'].item()                   
                    else:
                        data_exp = np.load(data_dic[inst][vis]['proc_Intr_data_paths']+str(i_in)+'.npz',allow_pickle=True)['data'].item()
                    
                    #Trimming profile         
                    for key in ['cen_bins','flux','cond_def']:fixed_args[key][inst][vis][isub] = data_exp[key][iord_sel,idx_range_kept]
                    fixed_args['ncen_bins'][inst][vis][isub] = ncen_bins
                    fixed_args['edge_bins'][inst][vis][isub] = data_exp['edge_bins'][iord_sel,idx_range_kept[0]:idx_range_kept[-1]+1]   
                    fixed_args['dcen_bins'][inst][vis][isub] = fixed_args['edge_bins'][inst][vis][isub][1::]-fixed_args['edge_bins'][inst][vis][isub][0:-1]  
                    fixed_args['cov'][inst][vis][isub] = data_exp['cov'][iord_sel][:,idx_range_kept]  # *0.6703558343325438

                    # print('ATTENTION')

                    #Line profile model table
                    model_st_prof_tab(inst,vis,isub,fixed_args,gen_dic,fixed_args['nexp_fit_all'][inst][vis])

                    #Initializing ranges in the relevant rest frame
                    if len(cont_range)==0:fit_prop_dic[inst][vis]['cond_def_cont_all'][isub] = True    
                    else:
                        for bd_int in cont_range:fit_prop_dic[inst][vis]['cond_def_cont_all'][isub] |= (fixed_args['edge_bins'][inst][vis][isub][0:-1]>=bd_int[0]) & (fixed_args['edge_bins'][inst][vis][isub][1:]<=bd_int[1])         
                    if len(fit_prop_dic['fit_range'][inst][vis])==0:fit_prop_dic[inst][vis]['cond_def_fit_all'][isub] = True    
                    else:
                        for bd_int in fit_prop_dic['fit_range'][inst][vis]:fit_prop_dic[inst][vis]['cond_def_fit_all'][isub] |= (fixed_args['edge_bins'][inst][vis][isub][0:-1]>=bd_int[0]) & (fixed_args['edge_bins'][inst][vis][isub][1:]<=bd_int[1])        

                    #Accounting for undefined pixels
                    fit_prop_dic[inst][vis]['cond_def_cont_all'][isub] &= fixed_args['cond_def'][inst][vis][isub]           
                    fit_prop_dic[inst][vis]['cond_def_fit_all'][isub] &= fixed_args['cond_def'][inst][vis][isub]          
                    fit_dic['nx_fit']+=np.sum(fit_prop_dic[inst][vis]['cond_def_fit_all'][isub])
                    fixed_args['cond_fit'][inst][vis][isub] = fit_prop_dic[inst][vis]['cond_def_fit_all'][isub]

                    #Initialize PCs 
                    if fixed_args['n_pc'][inst][vis] is not None:
                    
                        #PC matrix interpolated on current exposure table
                        fixed_args['eig_res_matr'][inst][vis][i_in] = np.zeros([fixed_args['n_pc'][inst][vis],fixed_args['ncen_bins'][inst][vis][isub]],dtype=float)
                    
                        #Process each PC
                        for i_pc in range(fixed_args['n_pc'][inst][vis]):
                            
                            #PC profile
                            fixed_args['eig_res_matr'][inst][vis][i_in][i_pc] = interp1d(data_pca['cen_bins'],data_pca['eig_res_matr'][i_pc],fill_value='extrapolate')(fixed_args['cen_bins'][inst][vis][isub])       
                            
                            #PC free amplitude
                            pc_name = 'aPC_idxin'+str(i_in)+'_ord'+str(i_pc)+'__IS'+inst+'_VS'+vis
                            fit_prop_dic['mod_prop'][pc_name]={'vary':True,'guess':0.} 
                            if i_pc==0:fit_prop_dic['mod_prop'][pc_name]['bd'] = [-4.,4.]
                            elif i_pc==1:fit_prop_dic['mod_prop'][pc_name]['bd'] = [-3.,2.]
                            else:fit_prop_dic['mod_prop'][pc_name]['bd'] = [-1.,1.]
                            fit_prop_dic['priors'][pc_name]={'low':-100. ,'high':100.,'mod':'uf'}
                            
                #Reference point for instrumental convolution
                #    - in RV space for analytical model, in wavelength space for theoretical profiles
                if ('ref_conv' not in fixed_args):ref_inst_convol(fixed_args,fixed_args['cen_bins'][inst][vis][0])

                #Continuum common to all processed profiles
                #    - collapsed along temporal axis
                cond_cont_com  = np.all(fit_prop_dic[inst][vis]['cond_def_cont_all'],axis=0)
                if np.sum(cond_cont_com)==0.:stop('No pixels in common continuum')  
                                          
                #Common continuum flux in intrinsic profiles
                cont_intr = np.zeros(fixed_args['nexp_fit_all'][inst][vis])*np.nan
                wcont_intr = np.zeros(fixed_args['nexp_fit_all'][inst][vis])*np.nan
                for isub in range(fixed_args['nexp_fit_all'][inst][vis]): 
                    cont_intr[isub] = np.mean(fixed_args['flux'][inst][vis][isub][cond_cont_com])
                    wcont_intr[isub] = 1./np.sum(fixed_args['cov'][inst][vis][isub][0,cond_cont_com] )        
                fixed_args['mean_cont'][inst][vis]=np.sum(cont_intr*wcont_intr)/np.sum(wcont_intr)       

                #Number of fitted exposures
                fixed_args['nexp_fit']+=fixed_args['nexp_fit_all'][inst][vis]
                
                #Store coordinates of fitted exposures in global table
                if fixed_args['bin_mode'][inst][vis]=='_bin':
                    sub_idx_in_fit = fixed_args['idx_in_fit'][inst][vis]
                    coord_vis = data_vis_bin['coord']
                else:
                    sub_idx_in_fit = gen_dic[inst][vis]['idx_in'][fixed_args['idx_in_fit'][inst][vis]]
                    coord_vis = coord_dic[inst][vis]
                for par in ['coord_pl_fit','ph_fit']:fixed_args[par][inst][vis]={}
                for pl_loc in data_dic[inst][vis]['transit_pl']:
                    fixed_args['ph_fit'][inst][vis][pl_loc] = np.vstack((coord_vis[pl_loc]['st_ph'][sub_idx_in_fit],coord_vis[pl_loc]['cen_ph'][sub_idx_in_fit],coord_vis[pl_loc]['end_ph'][sub_idx_in_fit]) ) 
                    fixed_args['coord_pl_fit'][inst][vis][pl_loc] = {}
                    for key in ['cen_pos','st_pos','end_pos']:fixed_args['coord_pl_fit'][inst][vis][pl_loc][key] = coord_vis[pl_loc][key][:,sub_idx_in_fit]


    #Artificial observation table
    #    - covariance condition is set to False so that chi2 values calculated here are not further modified within the residual() function
    fixed_args['x_val']=range(fit_dic['nx_fit'])
    fixed_args['y_val'] = np.zeros(fit_dic['nx_fit'],dtype=float)  
    fixed_args['s_val'] = np.ones(fit_dic['nx_fit'],dtype=float)          
    fixed_args['cov_val'] = np.array([fixed_args['s_val']**2.])  
    fixed_args['use_cov'] = False
    fixed_args['use_cov_eff'] = gen_dic['use_cov']
    fixed_args['fit_func'] = MAIN_joined_CCFs

    #Model fit and calculation
    fit_save={}
    fixed_args['fit'] = True
    merged_chain,p_final = common_fit_rout('IntrProf',fit_dic,fixed_args,fit_prop_dic,gen_dic,data_dic,theo_dic)            

    #PC correction
    if len(fit_prop_dic['PC_model'])>0:
        fit_save.update({'eig_res_matr':fixed_args['eig_res_matr'],'n_pc':fixed_args['n_pc'] })
       
        #Accouting for out-of-transit PCA fit in merit values
        #    - PC have been fitted in the PCA module to the out-of-transit data
        #      we include their merit values to those from RMR + PC fit 
        if fixed_args['nx_fit_PCout']>0:
            fit_dic_PCout = {}
            fit_dic_PCout['nx_fit'] = fit_dic['nx_fit'] + fixed_args['nx_fit_PCout']
            fit_dic_PCout['n_free']  = fit_dic['n_free'] + fixed_args['n_free_PCout']
            fit_dic_PCout['dof'] = fit_dic_PCout['nx_fit'] - fit_dic_PCout['n_free']
            fit_dic_PCout['chi2'] = fit_dic['chi2'] + fixed_args['chi2_PCout']          
            fit_dic_PCout['red_chi2'] = fit_dic_PCout['chi2']/fit_dic_PCout['dof']
            fit_dic_PCout['BIC']=fit_dic_PCout['chi2']+fit_dic_PCout['n_free']*np.log(fit_dic_PCout['nx_fit'])      
            fit_dic_PCout['AIC']=fit_dic_PCout['chi2']+2.*fit_dic_PCout['n_free']  
            
            np.savetxt(fit_dic['file_save'],[['----------------------------------']],fmt=['%s'])
            np.savetxt(fit_dic['file_save'],[['Merit values with out-of-transit PC fit']],fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['----------------------------------']],fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['Npts : '+str(fit_dic_PCout['nx_fit'])]],delimiter='\t',fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['Nfree : '+str(fit_dic_PCout['n_free'])]],delimiter='\t',fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['d.o.f : '+str(fit_dic_PCout['dof'])]],delimiter='\t',fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['Best chi2 : '+str(fit_dic_PCout['chi2'])]],delimiter='\t',fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['Reduced chi2 : '+str(fit_dic_PCout['red_chi2'])]],delimiter='\t',fmt=['%s'])
            np.savetxt(fit_dic['file_save'],[['BIC : '+str(fit_dic_PCout['BIC'])]],delimiter='\t',fmt=['%s'])
            np.savetxt(fit_dic['file_save'],[['AIC : '+str(fit_dic_PCout['AIC'])]],delimiter='\t',fmt=['%s'])
            np.savetxt(fit_dic['file_save'],[['----------------------------------']],fmt=['%s']) 
            np.savetxt(fit_dic['file_save'],[['']],fmt=['%s']) 


    #Best-fit model and derived properties
    fixed_args['fit'] = False
    mod_dic,coeff_line_dic,mod_prop_dic = joined_CCFs(p_final,fixed_args)

    #Save best-fit properties
    #    - with same structure as fit to individual profiles 
    fit_save.update({'p_final':p_final,'coeff_line_dic':coeff_line_dic,'func_prof_name':fixed_args['func_prof_name'],'func_prof':fixed_args['func_prof'],'name_prop2input':fixed_args['name_prop2input'],'coord_line':fixed_args['coord_line'],
                     'pol_mode':fit_prop_dic['pol_mode'],'idx_in_fit':fixed_args['idx_in_fit'],'genpar_instvis':fixed_args['genpar_instvis'],'linevar_par':fixed_args['linevar_par']})

    np.savez(fit_dic['save_dir']+'Fit_results',data=fit_save,allow_pickle=True)
    if (plot_dic['CCFintr']!='') or (plot_dic['CCFintr_res']!='') or (plot_dic['propCCFloc']!=''):
        for inst in fixed_args['inst_list']:
            for vis in fixed_args['inst_vis_list'][inst]:
                prof_fit_dic={}
                for isub,i_in in enumerate(fixed_args['idx_in_fit'][inst][vis]):
                    prof_fit_dic[i_in]={
                        'cen_bins':fixed_args['cen_bins'][inst][vis][isub],
                        'flux':mod_dic[inst][vis][isub],
                        'cond_def_fit':fit_prop_dic[inst][vis]['cond_def_fit_all'][isub],
                        'cond_def_cont':fit_prop_dic[inst][vis]['cond_def_cont_all'][isub]
                        }
                    for prop_loc in mod_prop_dic[inst][vis]:prof_fit_dic[i_in][prop_loc] = mod_prop_dic[inst][vis][prop_loc][isub]
                np.savez_compressed(fit_dic['save_dir']+'IntrProf_fit_'+inst+'_'+vis+fixed_args['bin_mode'][inst][vis],data={'prof_fit_dic':prof_fit_dic},allow_pickle=True)

    #Post-processing    
    fit_dic['p_null'] = deepcopy(p_final)
    for par in [ploc for ploc in fit_dic['p_null'] if 'ctrst' in ploc]:fit_dic['p_null'][par] = 0.    
    post_proc_func(p_final,fixed_args,fit_dic,merged_chain,fit_prop_dic,gen_dic)

    return None







'''
Generic function to calculate the merit factor of the joined fit
'''
def MAIN_joined_CCFs(param,x_tab,args=None):

    #Models over fitted spectral ranges
    mod_dic=joined_CCFs(param,args)[0]
    
    #Merit table
    #    - because exposures are specific to each visit, defined on different bins, and stored as objects we define the output table as :
    # chi = concatenate( exp, (obs(exp)-mod(exp))/err(exp)) ) or the equivalent with the covariance matrix
    #      so that the merit function will compare chi to a table of same size filled with 0 and with errors of 1 in the residual() function (where the condition to use covariance has been set to False for this purpose)
    #    - observed intrinsic profiles may have gaps, but due to the convolution the model must be calculated over the continuous table and then limited to fitted bins
    chi = np.zeros(0,dtype=float)
    if args['use_cov_eff']:
        for inst in args['inst_list']:
            for vis in args['inst_vis_list'][inst]:    
                for isub in range(args['nexp_fit_all'][inst][vis]):
                    cond_fit = args['cond_fit'][inst][vis][isub]
                    L_mat = scipy.linalg.cholesky_banded(args['cov'][inst][vis][isub], lower=True)
                    res = args['flux'][inst][vis][isub][cond_fit]-mod_dic[inst][vis][isub][cond_fit]
                    chi = np.append( chi, scipy.linalg.blas.dtbsv(L_mat.shape[0]-1, L_mat, res, lower=True)) 
    else:
        for inst in args['inst_list']:
            for vis in args['inst_vis_list'][inst]:    
                for isub in range(args['nexp_fit_all'][inst][vis]):
                    cond_fit = args['cond_fit'][inst][vis][isub]
                    res = args['flux'][inst][vis][isub][cond_fit]-mod_dic[inst][vis][isub][cond_fit]
                    chi = np.append( chi, res/np.sqrt( args['cov'][inst][vis][isub][0][cond_fit]) )
          
    return chi


'''
Time-series of planet-occulted intrinsic profiles
'''
def joined_CCFs(param,args):
    mod_dic = {}
    mod_prop_dic = {}
    coeff_line_dic = {}
    for inst in args['inst_list']:
        args['inst']=inst
        mod_dic[inst]={}
        coeff_line_dic[inst]={}
        mod_prop_dic[inst]={}
        for vis in args['inst_vis_list'][inst]:   
            args['vis']=vis

            #Calculate coordinates of occulted regions or use imported values
            system_param_loc,coord_pl = calc_plocc_coord(inst,vis,deepcopy(args['par_list']),args,param,args['transit_pl'][inst][vis],args['nexp_fit_all'][inst][vis],args['ph_fit'][inst][vis],args['coord_pl_fit'][inst][vis])

            #-----------------------------------------------------------
            #Variable line model for each exposure 
            #    - the intrinsic stellar line profile is convolved with the LSF kernel specific to each instrument
            #    - we assume a flat continuum, set after the PC component so that intrinsic profiles models later on can be defined with a continuum unity
            #-----------------------------------------------------------
            mod_dic[inst][vis]=np.zeros(args['nexp_fit_all'][inst][vis],dtype=object)
            for isub,i_in in enumerate(args['idx_in_fit'][inst][vis]):
                
                #Table for model calculation
                args_exp = def_st_prof_tab(inst,vis,isub,args)

                #Intrinsic profile for current exposure
                surf_prop_dic = sub_calc_plocc_prop(args['chrom_mode'],args_exp,args['par_list'],args['transit_pl'][inst][vis],system_param_loc,args['theo_dic'],args['system_prop'],param,coord_pl,1)
                sp_line_model = surf_prop_dic[args['chrom_mode']]['line_prof'][:,isub]

                #Conversion and resampling 
                mod_dic[inst][vis][isub] = conv_st_prof_tab(inst,vis,isub,args,args_exp,sp_line_model,return_FWHM_inst(inst,args['ref_conv']))

                #Add PC noise model
                #    - added to the convolved profiles since PC are derived from observed data
                if args['n_pc'][inst][vis] is not None:
                    for i_pc in range(args['n_pc'][inst][vis]):mod_dic[inst][vis][isub]+=param[args['name_prop2input']['aPC_idxin'+str(i_in)+'_ord'+str(i_pc)+'__IS'+inst+'_VS'+vis]]*args['eig_res_matr'][inst][vis][i_in][i_pc]

                #Set to continuum level
                mod_dic[inst][vis][isub]*=args['mean_cont'][inst][vis]


            #-----------------------------------------------------------
            #Outputs
            if not args['fit']:
                
                #Coefficients describing the polynomial variation of spectral line properties as a function of the chosen coordinate
                coeff_line_dic[inst][vis] = args['coeff_line']

                #Properties of all planet-occulted regions used to calculate spectral line profiles
                mod_prop_dic[inst][vis]={} 
                for pl_loc in args['transit_pl'][inst][vis]:
                    mod_prop_dic[inst][vis][pl_loc]={}                     
                    for prop_loc in ['rv']+args['linevar_par'][inst][vis]:mod_prop_dic[inst][vis][pl_loc][prop_loc] = surf_prop_dic[pl_loc][prop_loc][0] 

    return mod_dic,coeff_line_dic,mod_prop_dic






'''
Routine to fit atmospheric CCFs with a joined model
    - we use simple analytical profiles to describe the CCFs
      their shape and position are linked using polynomial laws as a function of phase
    - contrast and FWHM correspond to the atmospheric line before instrumental convolution, so that data from all instruments and visits can be fitted together
    - polynomial coefficients can nonetheless depend on the visit and their associated instrument, to account for possible variations in the line shape between visits
'''
def fit_CCFatm_all():
    print('   > Fitting atmospheric CCFs with linked model')


    return None








































'''
Routine to fit a given stellar surface property from planet-occulted regions with a common model over instrument/visit
'''
def fit_CCFintr_prop(fit_prop_dic,gen_dic,system_param,theo_dic,plot_dic,coord_dic,data_dic):
    print('   > Fitting single stellar surface property')
    
    #Path for file saving/retrieving
    save_dir=gen_dic['save_data_dir']+'/Joined_fits/Intr_prop/'+fit_prop_dic['data']+'/'
 
    #Initializations
    fixed_args,fit_dic = init_joined_routines(gen_dic,system_param,theo_dic,data_dic,save_dir,fit_prop_dic)    
    fit_dic['save_dir']+=fit_prop_dic['prop']+'/'        

    #Arguments to be passed to the fit function
    #    - fit is performed on achromatic average properties
    #    - the full stellar line profile are not calculated, since we only fit the average properties of the occulted regions
    fixed_args.update({
        'coord_line':fit_prop_dic['dim_fit'],
        'inst_list':[],
        'prior_func':fit_prop_dic['prior_func'], 
        'inst_vis_list':{},
        'nexp_fit_all':{},
        'transit_pl':{},
        'chrom_mode':['achrom'],
        'precision':'low',
        'prop_fit':fit_prop_dic['prop']})

    #Coordinate and property to calculate for the fit
    #    - surface RV model defined as a function of orbital phase  
    fixed_args['par_list']+=[fixed_args['prop_fit']]
    if fixed_args['prop_fit']=='rv':fixed_args['par_list']+=['rv']       
    else:
        if fit_prop_dic['dim_fit'] in ['abs_y_st','y_st2']:fixed_args['coord_line']='y_st'    
        else:fixed_args['coord_line']=fixed_args['dim_fit']
        fixed_args['par_list']+=[fixed_args['coord_line']]

    #Construction of the fit tables
    for par in ['s_val','y_val']:fixed_args[par]=np.zeros(0,dtype=float)
    for par in ['coord_pl_fit','ph_fit']:fixed_args[par]={}
    idx_fit2vis={}
    for inst in np.intersect1d(data_dic['instrum_list'],list(fit_prop_dic['idx_in_fit'].keys())):    
    
        #Instrument is fitted
        fit_prop_dic[inst]={}
        fixed_args['inst_list']+=[inst]
        fixed_args['inst_vis_list'][inst]=[]  
        for key in ['coord_pl_fit','ph_fit','nexp_fit_all','transit_pl']:fixed_args[key][inst]={}
        idx_fit2vis[inst] = {}
            
        for vis in data_dic[inst]['visit_list']:
            
            #Identify whether visit is fitted over original or binned exposures
            #    - for simplicity we then use the original visit name in all fit dictionaries, as a visit will not be fitted at the same time in its original and binned format
            if (vis in fit_prop_dic['idx_in_fit'][inst]) and (len(fit_prop_dic['idx_in_fit'][inst][vis])>0):fixed_args['bin_mode'][inst][vis]=''
            elif (vis+'_bin' in fit_prop_dic['idx_in_fit'][inst]) and (len(fit_prop_dic['idx_in_fit'][inst][vis+'_bin'])>0):fixed_args['bin_mode'][inst][vis]='_bin'
            else:vis=None
         
            #Visit is fitted
            if vis is not None: 
                fit_prop_dic[inst][vis]={}
                data_vis=data_dic[inst][vis]
                
                #Check for multi-transits
                #    - if two planets are transiting the properties derived from the fits to intrinsic profiles cannot be fitted, as the model only contains a single line profile
                if len(data_vis['transit_pl'])>1:stop('Multi-planet transit must be modelled with full intrinsic profiles')
                fixed_args['transit_pl'][inst][vis]=[data_vis['transit_pl'][0]] 
                
                #Upload surface property
                if fit_prop_dic['data']=='orig':
                    data_load = np.load(gen_dic['save_data_dir']+'/Introrig_prop/'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item()
                
                elif fit_prop_dic['data']=='bin':
                    data_load = np.load(gen_dic['save_data_dir']+'/Intrbin_prop/'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item()
                    data_load = data_load.update(np.load(gen_dic['save_data_dir']+'/Intrbin_data/'+inst+'_'+vis+'_'+data_dic['Intr']['dim_bin']+'_add.npz',allow_pickle=True)['data'].item())
            
                #Fitted exposures
                fixed_args['inst_vis_list'][inst]+=[vis]
                if fit_prop_dic['idx_in_fit'][inst][vis]=='all':fit_prop_dic['idx_in_fit'][inst][vis]=range(data_vis['n_in_tr'])
                else:fit_prop_dic['idx_in_fit'][inst][vis]=np.intersect1d(fit_prop_dic['idx_in_fit'][inst][vis],range(data_vis['n_in_tr']))
                fixed_args['nexp_fit_all'][inst][vis]=len(fit_prop_dic['idx_in_fit'][inst][vis])  
     
                #Fit tables
                idx_fit2vis[inst][vis] = range(fit_dic['nx_fit'],fit_dic['nx_fit']+len(fit_prop_dic['idx_in_fit'][inst][vis]))
                fit_dic['nx_fit']+=len(fit_prop_dic['idx_in_fit'][inst][vis])
                for i_in in fit_prop_dic['idx_in_fit'][inst][vis]:    
                    fixed_args['y_val'] = np.append(fixed_args['y_val'],data_load[i_in][fixed_args['prop_fit']])
                    fixed_args['s_val'] = np.append(fixed_args['s_val'],np.mean(data_load[i_in]['err_'+fixed_args['prop_fit']]))
    
                #Store coordinates of fitted exposures in global table
                sub_idx_in_fit = gen_dic[inst][vis]['idx_in'][fit_prop_dic['idx_in_fit'][inst][vis]]
                coord_vis = coord_dic[inst][vis]
                for par in ['coord_pl_fit','ph_fit']:fixed_args[par][inst][vis]={}
                for pl_loc in data_vis['transit_pl']:
                    fixed_args['ph_fit'][inst][vis][pl_loc] = np.vstack((coord_vis[pl_loc]['st_ph'][sub_idx_in_fit],coord_vis[pl_loc]['cen_ph'][sub_idx_in_fit],coord_vis[pl_loc]['end_ph'][sub_idx_in_fit]) ) 
                    fixed_args['coord_pl_fit'][inst][vis][pl_loc] = {}
                    for key in ['cen_pos','st_pos','end_pos']:fixed_args['coord_pl_fit'][inst][vis][pl_loc][key] = coord_vis[pl_loc][key][:,sub_idx_in_fit]

    fixed_args['nexp_fit'] = fit_dic['nx_fit']
    fixed_args['x_val']=range(fixed_args['nexp_fit'])
    fixed_args['fit_func'] = MAIN_joined_prop
    
    #Uncertainties on the property are given a covariance matrix structure for consistency with the fit routine 
    fixed_args['cov_val'] = np.array([fixed_args['s_val']**2.])
    fixed_args['use_cov'] = False   

    #Model fit and calculation
    fixed_args['fit'] = True
    merged_chain,p_final = common_fit_rout('IntrProp',fit_dic,fixed_args,fit_prop_dic,gen_dic,data_dic,theo_dic)   

    #Best-fit model and properties
    fit_save={}
    fixed_args['fit'] = False
    mod_tab,coeff_line_dic,fit_save['prop_mod'],fit_save['coord_mod']= joined_prop(p_final,fixed_args)
  
    #Save best-fit properties
    fit_save.update({'p_final':p_final,'coeff_line_dic':coeff_line_dic,'name_prop2input':fixed_args['name_prop2input'],'coord_line':fit_prop_dic['dim_fit'],'pol_mode':fit_prop_dic['pol_mode'],'genpar_instvis':fixed_args['genpar_instvis'],'linevar_par':fixed_args['linevar_par']})
    if (plot_dic['propCCFloc']!='') or (plot_dic['chi2_fit_loc_prop']!=''):
        key_list = ['prop_fit','err_prop_fit']
        for key in key_list:fit_save[key] = {}
        for inst in fixed_args['inst_list']:
            for key in key_list:fit_save[key][inst] = {}
            for vis in fixed_args['inst_vis_list'][inst]:
                fit_save['prop_fit'][inst][vis] = fixed_args['y_val'][idx_fit2vis[inst][vis]]
                fit_save['err_prop_fit'][inst][vis] = fixed_args['s_val'][idx_fit2vis[inst][vis]]
    np.savez(fit_dic['save_dir']+'Fit_results',data=fit_save,allow_pickle=True)
    
    #Post-processing
    fit_dic['p_null'] = deepcopy(p_final)
    if fit_prop_dic['prop']=='rv':fit_dic['p_null']['veq'] = 0.
    else:
        for par in fit_dic['p_null']:fit_dic['p_null'][par]=0.
    post_proc_func(p_final,fixed_args,fit_dic,merged_chain,fit_prop_dic,gen_dic)

  
    return None





 


'''
Wrap-up function for calculation of model surface property
'''
def MAIN_joined_prop(param,x_tab,args=None):
    return joined_prop(param,args)[0]

def joined_prop(param,args):
    mod_tab=np.zeros(0,dtype=float)
    mod_prop_dic = {}
    coeff_line_dic = {}
    mod_coord_dic={}
    for inst in args['inst_list']:
        mod_prop_dic[inst]={}
        coeff_line_dic[inst]={}
        mod_coord_dic[inst]={}
        for vis in args['inst_vis_list'][inst]:  

            #Calculate coordinates and properties of occulted regions 
            system_param_loc,coord_pl = calc_plocc_coord(inst,vis,args['par_list'],args,param,args['transit_pl'][inst][vis],args['nexp_fit_all'][inst][vis],args['ph_fit'][inst][vis],args['coord_pl_fit'][inst][vis])
            surf_prop_dic = sub_calc_plocc_prop(args['chrom_mode'],args,args['par_list'],args['transit_pl'][inst][vis],system_param_loc,args['theo_dic'],args['system_prop'],param,args['coord_pl_fit'][inst][vis],args['nexp_fit_all'][inst][vis])

            #Properties associated with the transiting planet in the visit 
            theo_vis = surf_prop_dic[args['transit_pl'][inst][vis][0]]      

            #Fit coordinate
            #    - only used for plots
            if not args['fit']:
                coeff_line_dic[inst][vis] = args['coeff_line']
                if args['prop_fit']=='rv':mod_coord_dic[inst][vis] = args['ph_fit'][inst][vis][args['transit_pl'][inst][vis][0]][1]
                else:
                    if args['coord_line']=='abs_y_st':mod_coord_dic[inst][vis] = np.abs(mod_coord_dic[inst][vis])
                    elif args['coord_line']=='y_st2':mod_coord_dic[inst][vis] = mod_coord_dic[inst][vis]**2.  
                    else:mod_coord_dic[inst][vis] = theo_vis[args['coord_line']][0] 

            #Model property for the visit 
            mod_prop_dic[inst][vis] = theo_vis[args['prop_fit']][0] 
                
            #Appending over all visits
            mod_tab=np.append(mod_tab,mod_prop_dic[inst][vis])
 
    return mod_tab,coeff_line_dic,mod_prop_dic,mod_coord_dic
    




"""
Sub-routine to calculate system properties
"""
def calc_plocc_coord(inst,vis,par_list,args,param_in,transit_pl,nexp_fit,ph_fit,coord_pl_fit):
    system_param_loc=deepcopy(args['system_param'])
   
    #In case param_in is defined as a Parameters structure, retrieve values and define dictionary
    param={}
    if isinstance(param_in,lmfit.parameter.Parameters):
        for par in param_in:param[par]=param_in[par].value
    else:param=param_in

    #Coefficients describing the polynomial variation of spectral line properties as a function of the chosen coordinate
    #    - coefficients can be specific to a given spectral line model
    if args['mode']=='glob_mod':
        args['coeff_line'] = {}
        for par_loc in args['linevar_par'][inst][vis]:    
            args['coeff_line'][par_loc] = polycoeff_def(param,args['coeff_ord2name'][inst][vis][par_loc])

    #Recalculate coordinates of occulted regions or use nominal values
    #    - the 'fit_X' conditions are only True if at least one parameter is varying, so that param_fit is True if fit_X is True
    if args['fit_orbit']:coord_pl = {}
    else:coord_pl = deepcopy(coord_pl_fit)
    for pl_loc in transit_pl: 
        coord_pl[pl_loc]={}
        if args['fit_orbit']:
            pl_params_loc = system_param_loc[pl_loc]
            
            #Update fitted system properties for current step 
            if ('lambda_rad__pl'+pl_loc in args['genpar_instvis']):lamb_name = 'lambda_rad__pl'+pl_loc+'__IS'+inst+'_VS'+vis 
            else:lamb_name = 'lambda_rad__pl'+pl_loc 
            if (lamb_name in args['var_par_list']):pl_params_loc['lambda_rad'] = param[lamb_name]                     
            if ('inclin_rad__pl'+pl_loc in args['var_par_list']):pl_params_loc['inclin_rad']=param['inclin_rad__pl'+pl_loc]       
            if ('aRs__pl'+pl_loc in args['var_par_list']):pl_params_loc['aRs']=param['aRs__pl'+pl_loc]  
            
            #Calculate coordinates
            #    - start/end phase have been set to None if no oversampling is requested, in which case start/end positions are not calculated
            if args['theo_dic']['d_oversamp'] is not None:phases = ph_fit[pl_loc]
            else:phases = ph_fit[pl_loc][1]
            x_pos_pl_all,y_pos_pl_all = calc_pl_coord(pl_params_loc['ecc'],pl_params_loc['omega_rad'],pl_params_loc['aRs'],pl_params_loc['inclin_rad'],phases,None,None,None)[0:2]
            if args['theo_dic']['d_oversamp'] is not None:
                coord_pl[pl_loc]['st_pos'] = np.vstack((x_pos_pl_all[0],y_pos_pl_all[0]))
                coord_pl[pl_loc]['cen_pos'] = np.vstack((x_pos_pl_all[1],y_pos_pl_all[1]))
                coord_pl[pl_loc]['end_pos'] = np.vstack((x_pos_pl_all[2],y_pos_pl_all[2]))
            else:coord_pl[pl_loc]['cen_pos'] = np.vstack((x_pos_pl_all,y_pos_pl_all))

        #Recalculate planet grid if relevant
        if args['fit_RpRs'] and ('RpRs__pl'+pl_loc in args['var_par_list']):
            args['system_prop']['achrom'],[pl_loc][0]=param['RpRs__pl'+pl_loc] 
            args['theo_dic']['Ssub_Sstar_pl'][pl_loc],args['theo_dic']['x_st_sky_grid_pl'][pl_loc],args['theo_dic']['y_st_sky_grid_pl'][pl_loc],r_sub_pl2=occ_region_grid(args['system_prop']['achrom'],[pl_loc][0],args['theo_dic']['nsub_Dpl'][pl_loc])  
            args['system_prop']['achrom'],['cond_in_RpRs'][pl_loc] = [(r_sub_pl2<args['system_prop']['achrom'],[pl_loc][0]**2.)]

    return system_param_loc,coord_pl









'''
Routine to fit atmospheric CCF properties with a common model
'''
def fit_CCFatm_prop():
    print('   > Fitting atmospheric properties with linked model')


    return None

















"""
Stage Théo
Routine to fit joined residual profiles, with a fixed number of spots. 

fit_prop_dic  =  glob_fit_dic['ResProf']
"""

def fit_CCFRes_all(data_dic,gen_dic,system_param,fit_prop_dic,theo_dic,plot_dic,coord_dic):
    
    print('   > Fitting joined residual stellar CCFs, including spots')

    #Path for file saving/retrieving
    save_dir=gen_dic['save_data_dir']+'/Joined_fits/Residual_prof/' 

    #Initializations
    fixed_args,fit_dic = init_joined_routines(gen_dic,system_param,theo_dic,data_dic,save_dir,fit_prop_dic)


    pl_loc = list(gen_dic['transit_pl'])[0]
     

    #Arguments to be passed to the fit function
    fixed_args.update({
        'func_prof' :{}, 
        'func_prof_name' : fit_prop_dic['func_prof_name'],  
        'prior_func':fit_prop_dic['prior_func'],  
        'cen_bins' :{},
        'dcen_bins' :{},
        'cond_def' :{},
        'cond_fit' :{},
        'cond_model':{},
        't_exp_bjd' : {},
        'flux':{},
        'cov' :{},
        'nexp_fit':0,
        'nexp_fit_all':{},
        'FWHM_inst':{},
        'inst_list':[],
        'inst_vis_list':{},
        'coord_line': fit_prop_dic['dim_fit'],
        'transit_pl':{},
        'pl_loc' : pl_loc,
        'phase'     : {},
        'precision' : fit_prop_dic['precision'],      
        'idx_in_fit' : {},
        'idx_calc'   : {},
        'n_in_visit' : {},
        'data_mast'  : {},
        'cont_DI_obs' : {},
        'rescaling' : {},
        'grid_dic'  : theo_dic,
        'planet_params' : system_param[pl_loc],
        'print_exp':False, 
        'calc_pl_flux' : True,
        'calc_pl_mean_prop' : False,
        'pol_mode': fit_prop_dic['pol_mode'],
        })
        
    fixed_args['grid_dic']['cond_in_RpRs'] = {pl_loc : data_dic['DI']['system_prop']['achrom']['cond_in_RpRs'][pl_loc]}
        
    
    #Coordinate to calculate for the fit : Mock Théo : only use 'r_prok' and 'mu'
    fixed_args['coord_fit']=fit_prop_dic['dim_fit']
    fixed_args['par_list']+=['rv',fixed_args['coord_fit']]

    
    #Construction of the fit tables
    for inst in data_dic['instrum_list']:
        
        #Instrument is fitted
        if (inst in fit_prop_dic['idx_in_fit']):
            fit_prop_dic[inst]={}
            fixed_args['inst_list']+=[inst]
            fixed_args['inst_vis_list'][inst]=[]            
            for key in ['phase','cen_bins','dcen_bins','flux','cov','nexp_fit_all','cond_def', 'cond_fit','cond_model','t_exp_bjd',  'idx_in_fit', 'rescaling','cont_DI_obs', 'n_in_visit', 'data_mast', 'idx_calc'] : 
                fixed_args[key][inst]={}
            fixed_args['FWHM_inst'][inst]=return_FWHM_inst(inst)
            fixed_args['func_prof'][inst] = dispatch_func_prof(fit_prop_dic['func_prof_name'][inst])
            for vis in data_dic[inst]['visit_list']:
                
                #Identify whether visit is fitted over original or binned exposures
                #    - for simplicity we then use the original visit name in all fit dictionaries, as a visit will not be fitted at the same time in its original and binned format
                if (vis in fit_prop_dic['idx_in_fit'][inst]) and (len(fit_prop_dic['idx_in_fit'][inst][vis])>0):fixed_args['bin_mode'][inst][vis]=''
                elif (vis+'_bin' in fit_prop_dic['idx_in_fit'][inst]) and (len(fit_prop_dic['idx_in_fit'][inst][vis+'_bin'])>0):fixed_args['bin_mode'][inst][vis]='_bin'
                else:vis=None
             
                #Visit is fitted
                if vis is not None: 
                    fit_prop_dic[inst][vis]={}
                    data_vis=data_dic[inst][vis]
                    fixed_args['inst_vis_list'][inst]+=[vis]
                    
                    for key in ['phase','flux','cov','cond_def', 't_exp_bjd', 'cont_DI_obs', 'rescaling'] :
                        fixed_args[key][inst][vis] = np.zeros(data_vis['n_in_visit'],dtype=object)
                    
                    #Fitted exposures (by default, we use all in-transit AND out-transit data. 
                    if fit_prop_dic['idx_in_fit'][inst][vis]=='all'    :    expo_fit = range(data_vis['n_in_visit'])
                    else    :   expo_fit = np.intersect1d(fit_prop_dic['idx_in_fit'][inst][vis],range(data_vis['n_in_visit']))
                    fixed_args['idx_in_fit'][inst][vis] = expo_fit
                    fixed_args['n_in_visit'][inst][vis] = data_vis['n_in_visit']
                    
                    # Number of fitted exposures during visit
                    fixed_args['nexp_fit_all'][inst][vis] = len(expo_fit) 

                    # Total number of fitted exposures 
                    fixed_args['nexp_fit'] += fixed_args['nexp_fit_all'][inst][vis]
                           
                    # Store phase coordinates and BJD times of all visit exposure
                    coord_vis = coord_dic[inst][vis]
                    fixed_args['phase'][inst][vis] = np.vstack((coord_vis[pl_loc]['st_ph'],coord_vis[pl_loc]['cen_ph'],coord_vis[pl_loc]['end_ph'])) 
                    fixed_args['t_exp_bjd'][inst][vis] = coord_vis['bjd']

                    # Load master-out data for current vis
                    data_mast_vis = np.load(data_dic[inst][vis]['mast_DI_data_paths'][0]+'.npz',allow_pickle=True)['data'].item()

                    # Store which exposures must be calculated (fitted exposures + exposures used in the master-out)
                    fixed_args['idx_calc'][inst][vis] = np.union1d(   fixed_args['idx_in_fit'][inst][vis]  ,  data_mast_vis['idx_to_bin']  )
                    
                    #Mock Théo : We forget about triming range or idk
                    iord_sel = 0
                    nspec = data_vis['nspec']
                    
                    # Retrieving fitted exposures data :
                    for i_count, iexp in enumerate(  fixed_args['idx_calc'][inst][vis]  ) :
                    
                        # Load light curve and DI prop data
                        data_LC_vis = dataload_npz(data_vis['scaled_data_paths']+str(iexp))
                        data_DI_prop_vis = np.load(gen_dic['save_data_dir']+'DIorig_prop/'+inst+'_'+vis+'.npz',  allow_pickle=True)['data'].item()
                                                
                        #Upload latest processed intrinsic data
                        data_exp = np.load(data_vis['proc_Res_data_paths']+str(iexp)+'.npz',allow_pickle=True)['data'].item()
                                                
                        # Retrieve defined bins : 
                        if i_count == 0 : 
                            cen_bins  = data_exp['cen_bins']  [iord_sel,:]
                            edge_bins = data_exp['edge_bins'] [iord_sel,:]
                            dcen_bins  = cen_bins[1] - cen_bins[0]
                            
                            # Detecting bins over which models are calculated and fitted  (we calculate the model over the range it is fitted, + a 3*FW_inst margin for convolution)
                            #    -> All tables will be relative to cen_bins[cond_in_model]   /!\  /!\  /!\
                            if len(fit_prop_dic['fit_range'][inst][vis])==0 : 
                                cond_in_fit   = np.ones(nspec,dtype=bool)
                                cond_in_model = np.ones(nspec,dtype=bool)
                            else:
                                cond_in_fit   = np.zeros(nspec,dtype=bool)
                                cond_in_model = np.zeros(nspec,dtype=bool)
                                FW_inst = fixed_args['FWHM_inst'][inst]
                                for bd_int in fit_prop_dic['fit_range'][inst][vis] : 
                                    cond_in_fit   |= (edge_bins[0:-1]>=bd_int[0]             )    &    (edge_bins[1:]<=bd_int[1]             )
                                    cond_in_model |= (edge_bins[0:-1]>=bd_int[0] - 3*FW_inst )    &    (edge_bins[1:]<=bd_int[1] + 3*FW_inst )
                                                                
                            # Store bins properties for the whole visit (edge bins are requested as arg of calc_binned_prof, but uselesss)
                            fixed_args['cen_bins'] [inst][vis]  =  cen_bins [cond_in_model] 
                            fixed_args['dcen_bins'] [inst][vis]  =  dcen_bins
                            fixed_args['cond_fit'] [inst][vis]  =  cond_in_fit[cond_in_model]
                                                
                        # Retrieve profile and defined bins  (relative to cond_model)    ->    only for fitted exposures
                        if iexp in  fixed_args['idx_in_fit'][inst][vis] : 
                            for key in ['flux','cond_def']:fixed_args[key][inst][vis][iexp] = data_exp[key][iord_sel,cond_in_model]
                            fixed_args['cov'][inst][vis][iexp] = data_exp['cov'][iord_sel][:,cond_in_model] 
                            fit_dic['nx_fit']+=np.sum(fixed_args['cond_def'][inst][vis][iexp]  &  fixed_args['cond_fit'][inst][vis] )

                        # Continuum level of the raw DI observations
                        fixed_args['cont_DI_obs'] [inst][vis][iexp] = data_DI_prop_vis[iexp]['cont']
                        
                        # Rescaling factor calculated in the 'broadband flux scaling' module, to go from DI CCF to rescaled CCF
                        LC_exp, scaling_exp  =  1-data_LC_vis['loc_flux_scaling'][iexp,0](cen_bins)  ,  data_LC_vis['glob_flux_scaling'][iexp]
                        fixed_args['rescaling'][inst][vis][iexp] = (LC_exp / scaling_exp) [cond_in_model]

                    # Store useful data for computing the master-out (in particular, weights !)
                    fixed_args['data_mast'][inst][vis] = {}
                    fixed_args['data_mast'][inst][vis]['idx_to_bin']     = data_mast_vis['idx_to_bin']
                    fixed_args['data_mast'][inst][vis]['dx_ov']          = data_mast_vis['dx_ov']
                    fixed_args['data_mast'][inst][vis]['cond_def']       = data_mast_vis['cond_def'][iord_sel,cond_in_model]
                    fixed_args['data_mast'][inst][vis]['weight']         = {}
                    for iexp_off in data_mast_vis['idx_to_bin'] : 
                        fixed_args['data_mast'][inst][vis]['weight'][iexp_off] = data_mast_vis['weight'][iexp_off][iord_sel,cond_in_model]
        
                    
    #Artificial observation table
    #    - covariance condition is set to False so that chi2 values calculated here are not further modified within the residual() function
    fixed_args['x_val']=range(fit_dic['nx_fit'])
    fixed_args['y_val'] = np.zeros(fit_dic['nx_fit'],dtype=float)  
    fixed_args['s_val'] = np.ones(fit_dic['nx_fit'],dtype=float)          
    fixed_args['cov_val'] = np.array([fixed_args['s_val']**2.])  
    fixed_args['use_cov'] = False
    fixed_args['use_cov_eff'] = gen_dic['use_cov']
    fixed_args['fit_func'] = MAIN_joined_CCFs_res
    
    #Model fit and calculation
    fixed_args['fit'] = True
    merged_chain,p_final = common_fit_rout('ResProf',fit_dic,fixed_args,fit_prop_dic,gen_dic,data_dic,theo_dic)     
    

    #Best-fit model and properties
    fit_save={}

    #Save best-fit properties
    #    - with same structure as fit to individual CCFs 
    fit_save.update({'p_final':p_final,'func_prof_name':fixed_args['func_prof_name'],'func_prof':fixed_args['func_prof'],'name_prop2input':fixed_args['name_prop2input'],'dim_fit':fit_prop_dic['dim_fit'],'pol_mode':fit_prop_dic['pol_mode'],'genpar_instvis':fixed_args['genpar_instvis']})
    np.savez(fit_dic['save_dir']+'Fit_results',data=fit_save,allow_pickle=True)
   
    #Post-processing    
    fit_dic['p_null'] = deepcopy(p_final)
    for par in [ploc for ploc in fit_dic['p_null'] if 'ctrst' in ploc]:fit_dic['p_null'][par] = 0.    
    post_proc_func(p_final,fixed_args,fit_dic,merged_chain,fit_prop_dic,gen_dic)
    
    
    return None

    

'''
Generic function to calculate the merit factor of the joined fit
'''
def MAIN_joined_CCFs_res(param,x_tab,args=None):
    
    #Models over fitted spectral ranges
    model_prof=joined_CCFs_res(param,args)
    
    chi = np.zeros(0,dtype=float)
    if args['use_cov_eff']:
        for inst in args['inst_list']:
            for vis in args['inst_vis_list'][inst]:    
                for iexp in args['idx_in_fit'][inst][vis]:
                    cond_fit = args['cond_def'][inst][vis][iexp] & args['cond_fit'] [inst][vis]
                    L_mat = scipy.linalg.cholesky_banded(args['cov'][inst][vis][iexp], lower=True)
                    res = args['flux'][inst][vis][iexp][cond_fit]-model_prof[inst][vis][iexp][cond_fit]
                    chi = np.append( chi, scipy.linalg.blas.dtbsv(L_mat.shape[0]-1, L_mat, res, lower=True)) 
    else:
        for inst in args['inst_list']:
            for vis in args['inst_vis_list'][inst]:    
                for iexp in args['idx_in_fit'][inst][vis]:
                    cond_fit = args['cond_def'][inst][vis][iexp] & args['cond_fit'] [inst][vis]
                    res = args['flux'][inst][vis][iexp][cond_fit]-model_prof[inst][vis][iexp][cond_fit]
                    chi = np.append( chi, res/np.sqrt( args['cov'][inst][vis][iexp][0][cond_fit]) )
                    
                        
    return chi



##
    
"""
Function which computes the model for residual profiles

3 steps : 

    1. We calculate all DI profile of the star (fitted exposures + exposures that contributed to the master-out), and we scale 
       them at the same value as after the Broadband flux Scaling module

    2. We compute the master out, with same weights as those used in the corresponding module
    
    3. We extract residual profile with   CCFres = master-out - CCFscal

"""

def joined_CCFs_res(param,args):
    
    model_prof = {}
    for inst in args['inst_list']:
        model_prof[inst]={}
        for vis in args['inst_vis_list'][inst]: 
        
        
          
              
            #Retrieve DI flux profiles
            DI_data = {}
            DI_data['flux'] = np.zeros(args['n_in_visit'][inst][vis], dtype = object)
            
            
            new_args = deepcopy(args)
            fit_properties = {
                'brband_w':None,
                'func_prof_name':mock_dic['intr_prof'][inst]['func_prof_name'],
                'flux_cont':mock_dic['intensity'][inst][vis]['I0']}
            fit_properties.update(new_args['intr_prof'][inst])
            new_args,param = init_custom_DI_prof(new_args,fit_properties,gen_dic,data_dic['DI']['system_prop'],theo_dic,inst,vis,new_args['system_param']['star'],param,[rv_mock,None,None],False)
            base_DI_prof = custom_DI_prof(param,None,args=new_args)[0]            
            
            for iexp in args['idx_calc'][inst][vis]:
                                
                # Deviation profile and light curve depth
                deviation_prof, occulted_flux = compute_deviation_profile(args, param, inst, vis, iexp,star_params,gen_dic,theo_dic,data_dic,coord_dic)[0:2]
                DI_prof_exp = base_DI_prof - deviation_prof
                cont_exp = 1 - occulted_flux
                
                # Set the continuum to the same as the corresponding exposure
                DI_prof_exp *= args['cont_DI_obs'] [inst][vis][iexp] / cont_exp
                
                # Rescaling exposure at the same level as in the 'broadband flux scaling' module
                DI_prof_exp *= args['rescaling'][inst][vis][iexp]
                
                # Convolving with instrumental FWHM
                DI_prof_exp= convol_prof (DI_prof_exp, args['cen_bins'][inst][vis], args['FWHM_inst'][inst])
                
                # Store exposure DI flux
                DI_data['flux'][iexp] = DI_prof_exp
                
                
                
                
            # Retrieve master_out from DI data 
            data_to_bin = {}
            for iexp_off in args['data_mast'][inst][vis]['idx_to_bin'] : 
                
                data_to_bin[iexp_off] = {}
                data_to_bin[iexp_off]['flux']     = np.array([   DI_data['flux'][iexp_off]   ])
                data_to_bin[iexp_off]['cond_def'] = np.array([   args['cond_def'][inst][vis][iexp]   ])
                data_to_bin[iexp_off]['weight']   = np.array([   args['data_mast'][inst][vis]['weight'][iexp_off]     ])
                data_to_bin[iexp_off]['cov']      = np.ones(   (1, 1, len(args['cen_bins'][inst][vis])),   dtype = float)
                
                
            nspec = len( args['cen_bins'] [inst][vis])
            master_out_flux = calc_binned_prof(args['data_mast'][inst][vis]['idx_to_bin'],   
                                               1,   
                                               [1,nspec] ,
                                               nspec,  
                                               data_to_bin, 
                                               inst,      
                                               len(args['data_mast'][inst][vis]['idx_to_bin']),  
                                               args['cen_bins'] [inst][vis],   
                                               {},   
                                               args['data_mast'][inst][vis]['dx_ov']    
                                               )['flux'][0]
                
                

                
            # Calculate residual profiles
            model_prof[inst][vis]=np.zeros(args['n_in_visit'][inst][vis], dtype = object)
            for iexp in args['idx_in_fit'][inst][vis]:
                
                # Extracting the residual profile
                model_prof[inst][vis][iexp] = master_out_flux - DI_data['flux'][iexp]
                
                
            
                if iexp > 60 : 
                    plt.plot(args['cen_bins'][inst][vis],  model_prof  [inst][vis][iexp], color = 'green')
                    plt.plot(args['cen_bins'][inst][vis],  args['flux'][inst][vis][iexp], color = 'red')
                    plt.show()
                

            
    return model_prof
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


'''
Stage Théo : new version of the fit_IntrProf_all function, which allows to modify the way intrinsic profile are calculated.
'''
def fit_IntrProf_all2(data_dic,gen_dic,system_param,fit_prop_dic,theo_dic,plot_dic,coord_dic):
    

    #Arguments to be passed to the fit function
    fixed_args.update({

        
        # Théo : new fields
        
        't_exp_bjd' : {},
        'phase'     : {},
        'pl_loc' : pl_loc,
        'precision' : fit_prop_dic['precision'],  
        'grid_dic'  : theo_dic,
        'planet_params' : system_param[pl_loc],
        'print_exp':False, 
        'calc_pl_flux' : True,
        'calc_pl_mean_prop' : False,
        'pol_mode': fit_prop_dic['pol_mode']
        })
        
        
    
    """ Théo : Add the list of cells within planetary disk """
    fixed_args['grid_dic']['cond_in_RpRs'] = {pl_loc : data_dic['DI']['system_prop']['achrom']['cond_in_RpRs'][pl_loc]}
        
    
    #Construction of the fit tables
    for inst in data_dic['instrum_list']:
        
        #Instrument is fitted
        if (inst in fit_prop_dic['idx_in_fit']):
            for vis in data_dic[inst]['visit_list']:
                if vis is not None: 
                
                    for isub,i_in in enumerate(fit_prop_dic['idx_in_fit'][inst][vis]):
                        
                        #Upload latest processed intrinsic data
                        data_exp = np.load(data_vis['proc_Intr_data_paths']+str(i_in)+'.npz',allow_pickle=True)['data'].item()
                        
                        """ Théo : we assume bin range is common to all the visit (to match with the 'compute_deviation_profile' args structure) """
                        if isub == 0 :
                            fixed_args['cen_bins'][inst][vis]  = data_exp['cen_bins']  [iord_sel,idx_range_kept]
                            fixed_args['dcen_bins'][inst][vis]  = fixed_args['cen_bins'][inst][vis][1] - fixed_args['cen_bins'][inst][vis][0]

                    """ Théo : we store phase and time coordinates of fitted exposures """
                    sub_idx_in_fit = gen_dic[inst][vis]['idx_in'][fit_prop_dic['idx_in_fit'][inst][vis]]
                    coord_vis = coord_dic[inst][vis]
                    fixed_args['phase'][inst][vis] = np.vstack((coord_vis[pl_loc]['st_ph' ][sub_idx_in_fit],
                                                                coord_vis[pl_loc]['cen_ph'][sub_idx_in_fit],
                                                                coord_vis[pl_loc]['end_ph'][sub_idx_in_fit])) 
                    fixed_args['t_exp_bjd'][inst][vis] = coord_vis['bjd'][sub_idx_in_fit]

                    
                    
                      

    #Best-fit model and properties
    """ Théo : we change the mode so that compute_deviation_profile calculates the mean properties of planet-occulted regions """
    fit_save={}
    fixed_args['calc_pl_mean_prop'] = True
    fixed_args['fit'] = False
    mod_dic,coeff_line_dic,mod_prop_dic = joined_CCFs2(p_final,fixed_args)


    return None

    
    
    
    




  
  
  

