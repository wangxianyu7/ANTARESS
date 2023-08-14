import glob
import numpy as np
from scipy.optimize import newton
from scipy import interpolate,special
import lmfit
from lmfit import Parameters
import batman # Fast Mandel-Agol (last release from L. Kreidberg)
from utils import stop,closest,np_where1D,npint,np_interp,closest_arr,init_parallel_func,planck,np_poly,is_odd,dataload_npz,datasave_npz
from scipy.signal import savgol_filter
from astropy.io import fits #to read .fits
from itertools import product as it_product
from constant_data import G_usi,Mjup,Msun,c_light,AU_1,Rsun,Rjup,c_light_A
from copy import deepcopy
import os as os_system 
import astropy.convolution.convolve as astro_conv
import bindensity as bind
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import griddata,UnivariateSpline,interp1d,CubicSpline
import scipy.linalg
from minim_routines import call_MCMC,init_fit,postMCMCwrapper_1,fit_merit,postMCMCwrapper_2,save_fit_results,fit_minimization,ln_prob_func_lmfit
import pandas as pd
from pathos.multiprocessing import cpu_count,Pool
from dace_query.spectroscopy import Spectroscopy
from scipy import stats
from numpy.polynomial import Polynomial
from pysme.synthesize import synthesize_spectrum
from KitCat_main import kitcat_mask

##########################
#General values
##########################


"""
Names of all possible model parameters
    - used in plots
"""
def model_par_names():

    return {
        'veq':'v$_\mathrm{eq}$ (km s$^{-1}$)','vsini':'v$_\mathrm{eq}$sin i$_{*}$ (km/s)',
        'Peq':'P$_\mathrm{eq}$ (d)',
        'alpha_rot':r'$\alpha_\mathrm{rot}$',   
        'beta_rot':r'$\beta_\mathrm{rot}$',       
        'cos_istar':r'cos(i$_{*}$)','istar_deg':'i$_{*}(^{\circ}$)',
        'lambda_rad':'$\lambda$', 
        'c1_CB':'CB$_{1}$','c2_CB':'CB$_{2}$','c3_CB':'CB$_{3}$',  
        'inclination':'i$_\mathrm{p}$ ($^{\circ}$)',
        'inclin_rad':'i$_\mathrm{p}$ (rad)',
        'aRs':'a/R$_{*}$',
        'Rstar':'R$_{*}$',
        'ctrst':'C','ctrst_ord0':'C$_{0}$','ctrst_ord1':'C$_{1}$','ctrst_ord2':'C$_{2}$','ctrst_ord3':'C$_{3}$','ctrst_ord4':'C$_{4}$', 
        'FWHM_ord0':'FWHM$_{0}$','FWHM_ord1':'FWHM$_{1}$','FWHM_ord2':'FWHM$_{2}$','FWHM_ord3':'FWHM$_{3}$','FWHM_ord4':'FWHM$_{4}$',
        'FWHM_LOR':'FWHM$_\mathrm{Lor}$',
        'a_damp':'a$_\mathrm{damp}$',
        'amp':'Amp',
        'rv':'RV (km/s)',
        'FWHM':'FWHM (km/s)',
        'rv_l2c':'RV$_{l}$-RV$_{c}$','amp_l2c':'A$_{l}$/A$_{c}$','FWHM_l2c':'FWHM$_{l}$/FWHM$_{c}$',
        'cont':'P$_\mathrm{cont}$','offset':'F$_\mathrm{off}$',
        'c1_pol':'c$_1$','c2_pol':'c$_2$','c3_pol':'c$_3$','c4_pol':'c$_4$',
        'LD_u1':'LD$_1$','LD_u2':'LD$_2$','LD_u3':'LD$_3$','LD_u4':'LD$_4$',
        'f_GD':'f$_{\rm GD}$','beta_GD':'$\beta_{\rm GD}$','Tpole':'T$_{\rm pole}$',
        'eta_R':r'$\eta_{\rm R}$','eta_T':r'$\eta_{\rm T}$','ksi_R':r'\Ksi$_\mathrm{R}$','ksi_T':r'\Ksi$_\mathrm{T}$',

        # Stage Théo
        'Tcenter' : 'T$_{sp}$', 'ang' : r'$\alpha_{sp}$', 'lat' : 'lat$_{sp}$', 'flux' : 'F$_{sp}$'
        }  

'''
Parameter conversion routines
'''
def conv_cosistar(modif_list,fixed_args_in,fit_dic_in,p_final_in,merged_chain_in):
    if 'cos_istar' in fixed_args_in['var_par_list']:                    
        iistar=np_where1D(fixed_args_in['var_par_list']=='cos_istar')                    
        if fit_dic_in['fit_mod']=='chi2':                     
            #    - dcosi = sin(i)*di
            #      di = dcosi/sin(i)                      
            p_final_in['istar_deg']= np.arccos(p_final_in['cos_istar'])*180./np.pi   
            sig_loc= (180./np.pi)*fit_dic_in['sig_parfinal_err']['1s'][0,iistar] / np.sqrt(1.-p_final_in['cos_istar']**2.)   
            if ('istar_deg_conv') in modif_list:fit_dic_in['sig_parfinal_err']['1s'][:,iistar] = [[sig_loc],[sig_loc]]
            elif ('istar_deg_add') in modif_list:fit_dic_in['sig_parfinal_err']['1s'] = np.hstack((fit_dic_in['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]])  )               
        elif fit_dic_in['fit_mod']=='mcmc':
            chain_loc = np.arccos(merged_chain_in[:,iistar])*180./np.pi     
            if ('istar_deg_conv') in modif_list:merged_chain_in[:,iistar]= chain_loc
            elif ('istar_deg_add') in modif_list:merged_chain_in=np.concatenate((merged_chain_in,chain_loc[:,None]),axis=1)  
        if ('istar_deg_conv') in modif_list:    
            fixed_args_in['var_par_list'][iistar]='istar_deg'            
            fixed_args_in['var_par_names'][iistar]='i$_{*}(^{\circ}$)'   
        elif ('istar_deg_add') in modif_list: 
            fixed_args_in['var_par_list']=np.append(fixed_args_in['var_par_list'],'istar_deg')
            fixed_args_in['var_par_names']=np.append(fixed_args_in['var_par_names'],'i$_{*}(^{\circ}$)') 
    else:
        if fit_dic_in['fit_mod']=='chi2':p_final_in['istar_deg']=np.arccos(p_final_in['cos_istar'])*180./np.pi  
        elif fit_dic_in['fit_mod']=='mcmc':fixed_args_in['fixed_par_val']['istar_deg']=np.arccos(p_final_in['cos_istar'])*180./np.pi                          
    return merged_chain_in

def conv_CF_intr_meas(modif_list,inst_list,inst_vis_list,fixed_args,merged_chain,gen_dic,p_final,fit_dic,fit_prop_dic):

    #HR RV table to calculate FWHM and contrast at high enough precision                       
    if fixed_args['func_prof_name']=='dgauss':
        fixed_args['velccf_HR'] = theo_dic['RVstart_HR_mod'] + theo_dic['dRV_HR_mod']*np.arange(  int((theo_dic['RVend_HR_mod']-theo_dic['RVstart_HR_mod'])/theo_dic['dRV_HR_mod'])  )
    for inst in inst_list:               
        fixed_args_loc = deepcopy(fixed_args)
        if ('CF0_DG_add' in modif_list) or ('CF0_DG_conv' in modif_list):fixed_args_loc['FWHM_inst']=None
        else:fixed_args_loc['FWHM_inst'] = return_FWHM_inst(inst,c_light)   
        for vis in inst_vis_list[inst]:
            if fixed_args['func_prof_name']=='gauss':
                if any('ctrst_ord1' in par_loc for par_loc in fixed_args['var_par_list']):stop('    Contrast must be constant')
                if any('FWHM_ord1' in par_loc for par_loc in fixed_args['var_par_list']):stop('    FWHM must be constant')                            
            ctrst0_name = fixed_args['name_prop2input']['ctrst_ord0__IS'+inst+'_VS'+vis]
            varC=any('ctrst_ord0' in par_loc for par_loc in fixed_args['var_par_list'])
            FWHM0_name = fixed_args['name_prop2input']['FWHM_ord0__IS'+inst+'_VS'+vis]   
            varF=any('FWHM_ord0' in par_loc for par_loc in fixed_args['var_par_list'])                      
            if fixed_args['func_prof_name']=='dgauss': 
                p_final_loc=deepcopy(p_final)
                p_final_loc['rv'] = 0.    #the profile position does not matter
                p_final_loc['ctrst'] = p_final_loc[ctrst0_name]  
                p_final_loc['FWHM'] = p_final_loc[FWHM0_name]                           
                for par_sub in ['amp_l2c','FWHM_l2c','rv_l2c']:p_final_loc[par_sub]=p_final_loc[fixed_args['name_prop2input'][par_sub+'__IS'+inst+'_VS'+vis]]   
                p_final_loc['cont'] = 1.  #the profile continuum does not matter                  
            if fit_dic['fit_mod']=='chi2': 
                if fixed_args['func_prof_name']=='gauss':
                    p_final['ctrst0__IS'+inst+'_VS'+vis],p_final['FWHM0__IS'+inst+'_VS'+vis] = gauss_intr_prop(p_final[ctrst0_name],p_final[FWHM0_name],fixed_args_loc['FWHM_inst']) 
                elif fixed_args['func_prof_name']=='dgauss':
                    p_final['ctrst0__IS'+inst+'_VS'+vis],p_final['FWHM0__IS'+inst+'_VS'+vis]=cust_mod_true_prop(p_final_loc,fixed_args['velccf_HR'],fixed_args_loc)[0:3]   
                sig_loc=np.nan 
                if varC:fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]])) 
                if varF:fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))                                                      
            elif fit_dic['fit_mod']=='mcmc':  
                if varC:ictrst_loc = np_where1D(fixed_args['var_par_list']==ctrst0_name)[0]
                if varF:iFWHM_loc = np_where1D(fixed_args['var_par_list']==FWHM0_name)[0] 
                if fixed_args['func_prof_name']=='gauss':
                    if varC:chain_ctrst0 = np.squeeze(merged_chain[:,ictrst_loc])
                    else:chain_ctrst0=np.repeat(p_final[ctrst0_name],fit_dic['nsteps_pb_all'])
                    if varF:chain_FWHM0 = np.squeeze(merged_chain[:,iFWHM_loc])
                    else:chain_FWHM0=np.repeat(p_final[FWHM0_name],fit_dic['nsteps_pb_all'])                             
                    chain_ctrst_temp,chain_FWHM_temp = gauss_intr_prop(chain_ctrst0,chain_FWHM0,fixed_args_loc['FWHM_inst'])                        
                if fixed_args['func_prof_name']=='dgauss': 
                    if varC:fixed_args_loc['var_par_list'][ictrst_loc]='ctrst'
                    if varF:fixed_args_loc['var_par_list'][iFWHM_loc]='FWHM'
                    for par_sub in ['amp_l2c','FWHM_l2c','rv_l2c']:
                        if any(par_sub in par_loc for par_loc in fixed_args['var_par_list']):fixed_args_loc['var_par_list'][fixed_args['var_par_list']==fixed_args['name_prop2input'][par_sub+'__IS'+inst+'_VS'+vis]]=par_sub       
                    if fit_prop_dic['nthreads']>1:chain_loc=para_cust_mod_true_prop(proc_cust_mod_true_prop,fit_prop_dic['nthreads'],fit_dic['nsteps_pb_all'],[merged_chain],(fixed_args_loc,p_final_loc,))                           
                    else:  chain_loc=proc_cust_mod_true_prop(merged_chain,fixed_args_loc,p_final_loc)   
                    chain_ctrst_temp = chain_loc[0]
                    chain_FWHM_temp = chain_loc[1]     
                if ('CF0_meas_add' in modif_list) or ('CF0_DG_add' in modif_list):   #add parameters
                    if varC:merged_chain=np.concatenate((merged_chain,chain_ctrst_temp[:,None]),axis=1) 
                    if varF:merged_chain=np.concatenate((merged_chain,chain_FWHM_temp[:,None]),axis=1) 
                elif ('CF0_meas_conv' in modif_list) or ('CF0_DG_conv' in modif_list):   #replace parameters
                    if varC:merged_chain[:,ictrst_loc] = chain_ctrst_temp
                    if varF:merged_chain[:,iFWHM_loc] = chain_FWHM_temp              
            if ('CF0_meas_add' in modif_list) or ('CF0_DG_add' in modif_list):
                if varC:
                    fixed_args['var_par_list']=np.concatenate((fixed_args['var_par_list'],[fixed_args['var_par_list'][ictrst_loc]+'_'+inst]))
                    fixed_args['var_par_names']=np.concatenate((fixed_args['var_par_names'],['Contrast$_\mathrm{'+inst+'}$']))
                if varF:
                    fixed_args['var_par_list']=np.concatenate((fixed_args['var_par_list'][fixed_args['var_par_list'][iFWHM_loc]+'_'+inst]))
                    fixed_args['var_par_names']=np.concatenate((fixed_args['var_par_names']['FWHM$_\mathrm{'+inst+'}$(km/s)']))
            elif ('CF0_meas_conv' in modif_list) or ('CF0_DG_conv' in modif_list):
                if varC:
                    fixed_args['var_par_list'][ictrst_loc] = fixed_args['var_par_list'][ictrst_loc]+'_'+inst
                    fixed_args['var_par_names'][ictrst_loc] = 'Contrast$_\mathrm{'+inst+'}$'
                if varF:                                
                    fixed_args['var_par_list'][iFWHM_loc] = fixed_args['var_par_list'][iFWHM_loc]+'_'+inst
                    fixed_args['var_par_names'][iFWHM_loc] = 'FWHM$_\mathrm{'+inst+'}$(km/s)'

    return merged_chain           

   



"""Compute the refraction index n of the air (n_vacuum=1.)
From C.Lovis in wavelength_calibration.py

wl_air = wl_vacuum/n

keyword argument:
l -- the wavelength in Angström
t -- air temperature in Celsius
p -- air pressure in millimeter of mercury

called in:
    -phys.read_model

"""
def air_index(l, t=15., p=760.):

    n = 1e-6 * p * (1 + (1.049-0.0157*t)*1e-6*p) / 720.883 / (1 + 0.003661*t) * (64.328 + 29498.1/(146-(1e4/l)**2) + 255.4/(41-(1e4/l)**2))
    n = n + 1
    return n





'''
Default function returning 1
'''
def default_func(*x):return np.array([1.])




'''
Function to check that pipeline data is saved on disk
'''
def check_data(file_paths,vis=None):
    check_flag = True
    suff = '' if vis is None else ' for '+vis
    for key in file_paths:
        check_flag&=(Path(file_paths[key]+'.npz').is_file())    
    if check_flag:print('         Retrieving data'+suff) 
    else:
        print('         Data not available'+suff)
        stop()    
    return None






# Orbit
def Kepler_func(Ecc_anom,Mean_anom,ecc):
    """
    Find the Eccentric anomaly using the mean anomaly and eccentricity 
        - M = E - e sin(E)
    """
    delta=Ecc_anom-ecc*np.sin(Ecc_anom)-Mean_anom
    return delta 


    
'''
Return mean anomany at mid-transit
'''    
def Mean_anom_TR_calc(ecc,omega_bar):    
    
    #True anomaly of the planet at mid-transit (in rad):
    #    - angle counted from 0 at the periastron, to the star/Earth LOS
    #    - >0 counterclockwise, possibly modulo 2pi
    #    - with omega_bar counted from ascending node to periastron
    True_anom_TR=(np.pi/2.)-omega_bar

    #Mean anomaly at the time of the transit
    #    - corresponds to 'dt_transit' (in years), time from periapsis to transit center 
    #    - atan(X) is in -pi/2 ; pi/2 
    Ecc_anom_TR=2.*np.arctan( np.tan(True_anom_TR/2.)*np.sqrt((1.-ecc)/(1.+ecc)) )
    Mean_anom_TR=Ecc_anom_TR-ecc*np.sin(Ecc_anom_TR)
    if (Mean_anom_TR<0.):Mean_anom_TR=Mean_anom_TR+2.*np.pi    
    
    return Mean_anom_TR
    

'''
Radial velocity of stellar surface from rotation
    - in km/s
    - negative toward the observer
    - v = 2pi Om Rstar
      rv = 2pi x sinistar Om 
      rv = 2pi x sinistar Om_eq (1-alpha_rot*ylat^2-beta_rot*ylat^4)
      rv = x_norm veq sinistar (1-alpha_rot*ylat^2-beta_rot*ylat^4)      
'''
def calc_RVrot(x_sky_st,y_st,istar_rad,st_par):
    return x_sky_st*st_par['veq']*np.sin(istar_rad)*(1.-st_par['alpha_rot']*y_st**2.-st_par['beta_rot']*y_st**4.)



'''
Define planetary orbite coordinates for plot and contact times
    - 'coord_orbit' coordinates are defined in the Sky-projected orbital frame: Xsky,Ysky,Zsky
 Xsky = node line of orbital plane
 Ysky = projection of the orbital plane normal
 Zsky = LOS
    - 'inclin' is the inclination from the LOS to the normal to the orbital plane
'''    
def def_plotorbite(n_pts_orbit,pl_params):  
    ecc=pl_params['ecc']
    aRs=pl_params['aRs']
    Inclin=pl_params['inclin_rad']
    omega_bar=pl_params['omega_rad']
  
    #Elliptic orbit 
    if (ecc > 1e-4):
        
        #Time resolution of the orbit
        #    - we set a high resolution around the periastron   
        #    - see planet_coord for details  
        n_pts_horbit=int(n_pts_orbit/2.)
        ph_plot=-(0.05)+(0.1/n_pts_horbit)*np.arange(n_pts_horbit+1)
        ph_plot=np.append(ph_plot,(0.05)+((1-0.1 )/n_pts_horbit)*np.arange(n_pts_horbit+1))
        Mean_anom_plot=2.*np.pi*ph_plot
        Ecc_anom_plot=np.array([newton(Kepler_func,Mean_anom_plot[i],args=(Mean_anom_plot[i],ecc,)) for i in range(len(Mean_anom_plot))])
    
        #Coord. in the orbit plane with semi-major axis as X axis    
        X0_plot=aRs*(np.cos(Ecc_anom_plot)-ecc)
        Y0_plot=aRs*np.sqrt(1.-pow(ecc,2.))*np.sin(Ecc_anom_plot)

        #Coord. in the XYZ simulation referential     
        coord_orbit = [-X0_plot*np.cos(omega_bar) +  Y0_plot*np.sin(omega_bar),                  #Xsky
                      -( X0_plot*np.sin(omega_bar) +  Y0_plot*np.cos(omega_bar) )*np.cos(Inclin),   #Ysky
                       ( X0_plot*np.sin(omega_bar) +  Y0_plot*np.cos(omega_bar) )*np.sin(Inclin)]   #Zsky
    
    #------------------------------------------          
    #Circular orbit   
    else:
        n_pts_orbit=int(n_pts_orbit)
        ph_plot=np.arange(n_pts_orbit+1.)/n_pts_orbit
        X0_plot= aRs*np.cos(2.*np.pi*ph_plot)
        Y0_plot= aRs*np.sin(2.*np.pi*ph_plot)
        coord_orbit=[Y0_plot,                     #Xsky
                     -X0_plot*np.cos(Inclin),        #Ysky 
                      X0_plot*np.sin(Inclin)]        #Zsky 

    return coord_orbit

'''  
Contacts 
    - we use a very high phase resolution close to the transit center
'''
def def_contacts(RpRs,pl_params,stend_ph,star_params):  
    ecc=pl_params['ecc']
    aRs=pl_params['aRs']
    omega_bar=pl_params['omega_rad']
    Inclin=pl_params['inclin_rad']
    
    contact_phases=np.zeros(4,dtype=float)*np.nan
    n_pts_contacts=int(5000)   
    ph_ineg=np.arcsin(1./aRs)/(2.*np.pi)   #phase approximative de l'ingress
    ph_st=-stend_ph*ph_ineg
    ph_end=stend_ph*ph_ineg
    ph_contacts=ph_st+((ph_end-ph_st)/n_pts_contacts)*np.arange(n_pts_contacts+1.)
    
    if (ecc > 1e-4):
        if 'Mean_anom_TR' not in pl_params:pl_params['Mean_anom_TR'] = Mean_anom_TR_calc(ecc,omega_bar) 
        Mean_anom_plot=2.*np.pi*ph_contacts+pl_params['Mean_anom_TR']
        Ecc_anom_plot=np.array([newton(Kepler_func,Mean_anom_plot[i],args=(Mean_anom_plot[i],ecc,)) for i in range(len(Mean_anom_plot))])   
        X0_plot=aRs*(np.cos(Ecc_anom_plot)-ecc)
        Y0_plot=aRs*np.sqrt(1.-pow(ecc,2.))*np.sin(Ecc_anom_plot)
        xp=-X0_plot*np.cos(omega_bar) +  Y0_plot*np.sin(omega_bar)
        yp=-( X0_plot*np.sin(omega_bar) +  Y0_plot*np.cos(omega_bar) )*np.cos(Inclin) 
        zp=( X0_plot*np.sin(omega_bar) +  Y0_plot*np.cos(omega_bar) )*np.sin(Inclin) 
    else:
        X0_plot= aRs*np.cos(2.*np.pi*ph_contacts)
        Y0_plot= aRs*np.sin(2.*np.pi*ph_contacts)
        xp=Y0_plot
        yp=-X0_plot*np.cos(Inclin)
        zp=X0_plot*np.sin(Inclin)
        
    #Points before transit, front of the star
    w_bef=np.where((xp<0) & (zp>0))[0]        

    #Points before transit, front of the star
    w_aft=np.where((xp>0) & (zp>0))[0]

    #Oblate star    
    if star_params['f_GD']>0.:

        #Frame conversion of planet coordinates from the classical frame perpendicular to the LOS, to the 'inclined star' frame
        idx_front = np_where1D(zp>0)
        xp_st_sk,yp_st_sk,_=conv_Losframe_to_inclinedStarFrame(pl_params['lambda_rad'],xp[idx_front],yp[idx_front],None)              

        #Number of planet limb points within the projected stellar photosphere
        nlimb = 501
        nlimb_in_ph = oblate_ecl(nlimb,RpRs,xp_st_sk,yp_st_sk,star_params)

        #First and fourth contacts: start of ingress / end of egress
        #    nlimb_in_ph >0 for the first / last time
        w_first=np_where1D(nlimb_in_ph[w_bef]>0)[0]
        w_fth=np_where1D(nlimb_in_ph[w_aft]>0)[-1]
        
        #Second and third contacts: end of ingress / start of egress
        #    nlimb_in_ph = nlimb for the first / last time
        w_scnd=np_where1D(nlimb_in_ph[w_bef]==nlimb)[0]
        w_thrd=np_where1D(nlimb_in_ph[w_aft]==nlimb)[-1]


    #Spherical star:
    else: 
        
        #Distance star - planet centers in the plane of sky
        Dprojplanet=np.sqrt(np.power(xp,2.) + np.power(yp,2.))
    
        #First contact: start of ingress
        #    dist_p = Rs+Rp
        w_first=closest(Dprojplanet[w_bef],(1.+RpRs))
    
        #Second contact: end of ingress
        #    dist_p = Rs-Rp
        w_scnd=closest(Dprojplanet[w_bef],(1.-RpRs))
    
        #Third contact: start of egress
        #    dist_p = Rs-Rp
        w_thrd=closest(Dprojplanet[w_aft],(1.-RpRs))

    
        #Fourth contact: end of egress
        #    dist_p = Rs+Rp
        w_fth=closest(Dprojplanet[w_aft],(1.+RpRs))
    
    #Contacts
    if w_bef[w_first]==0:stop('Decrease start phase for contact determination')
    if w_aft[w_fth]==n_pts_contacts:stop('Increase end phase for contact determination')
    contact_phases[0]=ph_contacts[w_bef][w_first]      
    contact_phases[1]=ph_contacts[w_bef][w_scnd]     
    contact_phases[2]=ph_contacts[w_aft][w_thrd]     
    contact_phases[3]=ph_contacts[w_aft][w_fth]    
        
    return contact_phases   











'''
Initializations
'''
def init_prop(data_dic,mock_dic,gen_dic,system_param,theo_dic,plot_dic,glob_fit_dic,PropAtm_fit_dic,detrend_prof_dic):

    #Multi-threading
    print(str(cpu_count())+' threads available for multi-threading')
  
    #Positional dictionary
    coord_dic={}
    
    #Instrument root names
    gen_dic['inst_root']={
        'SOPHIE_HE':'SOPHIE',
        'SOPHIE_HR':'SOPHIE',
        'CORALIE':'CORALIE',        
        'HARPN':'HARPN',
        'HARPS':'HARPS',
        'ESPRESSO':'ESPRESSO',
        'ESPRESSO_MR':'ESPRESSO',
        'CARMENES_VIS':'CARMENES',
        'EXPRES':'EXPRES',
        'NIRPS_HA':'NIRPS',
        'NIRPS_HE':'NIRPS',
    }     

    #Number of physical orders per instrument
    gen_dic['norders_instru']={
        'SOPHIE_HE':39,
        'SOPHIE_HR':39,
        'CORALIE':69,           
        'HARPN':69,
        'HARPS':71,
        'ESPRESSO':170,
        'ESPRESSO_MR':85,
        'CARMENES_VIS':61,
        'EXPRES':86,
        'NIRPS_HA':71,
        'NIRPS_HE':71,
    } 

    #Return flag that errors on input spectra are defined or not for each instrument   
    gen_dic['flag_err_inst']={          
        'SOPHIE_HE':False,
        'SOPHIE_HR':False,
        'CORALIE':False,         
        'HARPN':True   ,
        'HARPS':True,
        'ESPRESSO':True,
        'ESPRESSO_MR':True,
        'CARMENES_VIS':True,
        'EXPRES':True,
        'NIRPS_HA':True,'NIRPS_HE':True} 
    
    #Central wavelengths of orders for known instruments
    gen_dic['wav_ord_inst']={     
        'ESPRESSO':10.*np.repeat(np.flip([784.45 ,774.52, 764.84, 755.40, 746.19, 737.19, 728.42 ,719.85, 711.48, 703.30, 695.31, 687.50, 679.86, 672.39, 665.08,
                                      657.93 ,650.93 ,644.08, 637.37, 630.80, 624.36, 618.05, 611.87, 605.81, 599.87, 594.05, 588.34, 582.74, 577.24, 571.84,
                                      566.55, 561.35, 556.25, 551.24, 546.31, 541.48, 536.73, 532.06, 527.48, 522.97, 522.97, 518.54, 514.18, 509.89, 505.68,
                                      501.53, 497.46, 493.44, 489.50 ,485.61, 481.79 ,478.02, 474.32, 470.67, 467.08, 463.54 ,460.05, 456.62, 453.24, 449.91,
                                      446.62, 443.39, 440.20, 437.05, 433.95 ,430.90, 427.88 ,424.91, 421.98, 419.09, 416.24, 413.43, 410.65, 407.91, 405.21,
                                      402.55, 399.92, 397.32, 394.76, 392.23, 389.73, 387.26, 384.83, 382.42, 380.04]),2),
        'HARPN':np.array([3896.9385, 3921.912 , 3947.2075, 3972.8315, 3998.7905, 4025.0913,4051.7397, 4078.7437, 4106.11  , 4133.8457, 4161.9595, 4190.458 ,
                         4219.349 , 4248.641 , 4278.3433, 4308.464 , 4339.011 , 4369.9946,4401.424 , 4433.3086, 4465.6587, 4498.484 , 4531.796 , 4565.6045,
                         4599.922 , 4634.759 , 4670.127 , 4706.0396, 4742.509 , 4779.548 ,4817.169 , 4855.388 , 4894.2188, 4933.675 , 4973.773 , 5014.5273,
                         5055.955 , 5098.074 , 5140.9004, 5184.452 , 5228.747 , 5273.8066,5319.6494, 5366.297 , 5413.7686, 5462.088 , 5511.2773, 5561.3613,
                         5612.3633, 5664.311 , 5717.2285, 5771.1436, 5826.086 , 5882.084 ,5939.1685, 5997.3726, 6056.7285, 6117.271 , 6179.0366, 6242.062 ,
                         6306.3867, 6372.0503, 6439.0957, 6507.568 , 6577.511 , 6648.9746,6722.0083, 6796.6646, 6872.9976]),
        'HARPS': np.array([3824.484, 3848.533, 3872.886, 3897.549, 3922.529, 3947.831, 3973.461,
                           3999.426, 4025.733, 4052.388, 4079.398, 4106.771, 4134.515, 4162.635, 4191.141,
                           4220.039, 4249.338, 4279.048, 4309.175, 4339.73 , 4370.722, 4402.16 , 4434.053,
                           4466.411, 4499.245, 4532.564, 4566.384, 4600.709, 4635.555, 4670.933, 4706.854,
                           4743.333, 4780.382, 4818.015, 4856.243, 4895.084, 4934.552, 4974.66 , 5015.426,
                           5056.866, 5098.997, 5141.835, 5185.398, 5229.708, 5274.78 , 5367.266, 5414.749,
                           5463.079, 5512.279, 5562.374, 5613.388, 5665.346, 5718.276, 5772.203, 5827.157,
                           5883.167, 5940.266, 5998.481, 6057.852, 6118.408, 6180.188, 6243.229, 6307.567,
                           6373.246, 6440.308, 6508.795, 6578.756, 6650.235, 6723.287, 6797.961, 6874.313]),                
        'CARMENES_VIS': np.array([5185.683800259293,5230.002514675239,5275.085301397565,5320.952091123464,5367.623513832012,
                                  5415.120929724138,5463.466461819996,5512.683030318242,5562.794388829332,5613.825162603174,5665.800888880403,5718.748059506192,5772.694165955971,
                                  5827.667746933892,5883.6984387171215,5940.817028432618,5999.055510467573,6058.447146230714,6119.026527498995,6180.829643603155,6243.893952726336,
                                  6308.2584576125455,6373.9637860064695,6441.052276173239,6509.5680678764,6579.557199224864,6651.067709835355,6724.149750796099,6798.855701960672,
                                  6875.24029714847,6953.36075788067,7033.276936338339,7115.051468293246,7198.749936832545,7284.4410477767,7372.196817776685,7462.09277617269,
                                  7554.208181803426,7648.626256074004,7745.4344337228085,7844.724632875495,7946.5935461392,8051.142954674568,8158.480067389824,8268.717887632938,
                                  8381.975610018162,8498.379050316224,8618.061111667314,8741.162290748554,8867.83122794844,8998.225306077602,9132.51130268578,9270.866101669406,
                                  9413.477470553611,9560.54491063029,9712.280588045605,9868.910354974394,10030.674871216968,10197.83083793165,10370.652356804127,10549.432429789025]),    
        'NIRPS_HA': np.array([   9793.31830725 , 9859.95301593 , 9927.49931744 , 9995.97622395,10065.40380975, 10135.80171767, 10207.19048503, 10279.59121586,
                                10353.02577856, 10427.51660496, 10503.0864558 , 10579.75913144,10657.5589196 , 10736.51096413, 10816.64116324, 10897.97569896,
                                10980.54199941, 11064.36827222, 11149.48372245, 11235.91830631,11323.70300177, 11412.86989695, 11503.45207046, 11595.48314928,
                                11688.99793186, 11784.03276228, 11880.62559736, 11978.81474198,12078.63947148, 12180.1413448 , 12283.36401067, 12388.35148928,
                                12495.14844764, 12603.80175443, 12714.36064267, 12826.87609104,12941.40020952, 13057.98722241, 13176.69424781, 13297.58039703,
                                13420.70477935, 13546.12854527, 13673.91830748, 14072.1783805,14210.13779528, 14350.82710154, 14494.33106043, 14640.73394106,
                                14790.12380228, 14942.59417032, 15098.24089619, 15257.16474828,15419.46907017, 15585.26310369, 15754.66203085, 15927.78376522,
                                16104.75265749, 16285.69620041, 16470.75172016, 16660.06255469,16853.77645383, 17052.04937274, 17255.0418695 , 17462.92163421,
                                17675.87345983, 17894.08476988, 18117.75049316, 18347.07795686,18582.28588236, 18823.6030827 , 19071.27048963]),
        'EXPRES': np.array([3827.18212341, 3851.28596202, 3875.69435186, 3900.41312958 ,3925.44828033,3950.80594266 ,3976.49241363 ,4002.51415404 ,4028.87779393 ,4055.59013823,
                             4082.65817269 ,4110.08906999 ,4137.89019612 ,4166.069117   ,4194.63360538,4223.591648   ,4252.95145307 ,4282.72145806 ,4312.91033776 ,4343.52701277,
                             4374.58065826 ,4406.08071317 ,4438.03688981 ,4470.45918379 ,4503.3578845,4536.74358599 ,4570.62719834 ,4605.01995956 ,4639.93344803 ,4675.37959548,
                             4711.3707006  ,4747.91944324 ,4785.03889938 ,4822.74255663 ,4861.04433066,4899.95858228 ,4939.50013539 ,4979.68429586 ,5020.52687121 ,5062.04419143,
                             5104.2531307  ,5147.17113026 ,5190.81622248 ,5235.20705609 ,5280.3629228,5326.3037852  ,5373.05030624 ,5420.62388022 ,5469.04666543 ,5518.34161856,
                             5568.53253104 ,5619.6440673  ,5671.70180522 ,5724.73227878 ,5778.76302323,5833.82262268 ,5889.94076065 ,5947.14827335 ,6005.47720621 ,6064.96087378,
                             6125.63392315 ,6187.53240127 ,6250.69382638 ,6315.15726382 ,6380.96340664,6448.15466126 ,6516.77523859 ,6586.87125105 ,6658.49081593 ,6731.68416551,
                             6806.50376456 ,6883.00443578 ,6961.24349375 ,7041.2808881  ,7123.17935679,7207.00459002 ,7292.82540601 ,7380.71393943 ,7470.74584357 ,7563.00050759,
                             7657.56128994 ,7754.51576965 ,7853.9560168  ,7955.97888421 ,8060.68632206,8168.18571772])


}
    gen_dic['wav_ord_inst']['NIRPS_HE'] = gen_dic['wav_ord_inst']['NIRPS_HA'] 

    #Data type
    if gen_dic['mock_data']: 
        print('Running with artificial data')  
        data_dic['instrum_list']=list(mock_dic['visit_def'].keys())
        gen_dic['tell_weight']=False    
    
    else: 
        print('Running with observational data')  
        
        #Instruments to which are associated the different datasets
        data_dic['instrum_list'] = list(gen_dic['data_dir_list'].keys())

    #Used visits
    for inst in data_dic['instrum_list'] :
        if inst not in gen_dic['unused_visits']:gen_dic['unused_visits'][inst]=[]
        
    #Total number of processed instruments
    gen_dic['n_instru'] = len(data_dic['instrum_list'])

    #All processed types
    #    - if not set, assumed to be CCF by default
    #    - gen_dic['type'] traces the original types of datasets throughout the pipeline 
    gen_dic['all_types'] = []
    for inst in data_dic['instrum_list']:    
        if (inst not in gen_dic['type']):gen_dic['type'][inst]='CCF'
        gen_dic['all_types']+=[gen_dic['type'][inst]]
    gen_dic['all_types'] = list(np.unique(gen_dic['all_types']))
    gen_dic['specINtype'] = any('spec' in s for s in gen_dic['all_types'])
    gen_dic['ccfINtype'] = ('CCF' in gen_dic['all_types'])   

    #Deactivate conditions
    if (not gen_dic['specINtype']):
        gen_dic['DI_CCF']=False
        detrend_prof_dic['full_spec'] = False
    if (not gen_dic['specINtype']) or (gen_dic['DI_CCF']):        
        gen_dic['Intr_CCF']=False     
        
    #Deactivate all calculation options
    if not gen_dic['calc_all']:
        for key in gen_dic:
            if ('calc_' in key) and (gen_dic[key]):gen_dic[key] = False 

    #Deactivate spectral corrections in CCF mode
    sp_corr_list = ['corr_tell','corr_FbalOrd','corr_Fbal','corr_Ftemp','corr_cosm','mask_permpeak','corr_wig','corr_fring','trim_spec','glob_mast','cal_weight','gcal']
    if (not gen_dic['specINtype']):
        for key in sp_corr_list:gen_dic[key]=False
    else:
        #Activate spectral calibration calculation
        gen_dic['gcal'] = True
        
        #Deactivate spectral balance correction over orders if not 2D spectra
        if ('spec2D' not in gen_dic['all_types']): 
            gen_dic['corr_FbalOrd']=False
            plot_dic['wig_order_fit']=''

        #Activate global master calculation if required for flux balance corrections
        if (gen_dic['corr_Fbal']) or (gen_dic['corr_FbalOrd']):gen_dic['glob_mast']=True
        
        #Deactivate arbitrary scaling if global scaling not required
        if (not gen_dic['corr_Fbal']):gen_dic['Fbal_vis']=None
        
        #Check for arbitrary scaling        
        elif (gen_dic['Fbal_vis']!='theo') and (gen_dic['n_instru']>1):stop('Flux balance must be set to a theoretical reference for multiple instruments')

        #Deactivate transit scaling if temporal flux correction is applied
        if gen_dic['corr_Ftemp']: gen_dic['flux_sc']=False
            
        #Activate correction flag
        gen_dic['corr_data'] = False
        for key in sp_corr_list: gen_dic['corr_data'] |=gen_dic[key]

    #Set general condition for CCF conversions
    gen_dic['CCF_from_sp'] = gen_dic['DI_CCF'] | gen_dic['Intr_CCF'] | gen_dic['Atm_CCF']
    if gen_dic['CCF_from_sp']:gen_dic['ccfINtype'] = True
        
    #Set general condition for 2D/1D conversion
    gen_dic['spec_1D'] = gen_dic['spec_1D_DI'] | gen_dic['spec_1D_Intr'] | gen_dic['spec_1D_Atm']
    gen_dic['calc_spec_1D'] = gen_dic['calc_spec_1D_DI'] | gen_dic['calc_spec_1D_Intr'] | gen_dic['calc_spec_1D_Atm']

    #Set final data mode for each type of profiles
    #    - the general instrument and visit dictionaries contain a field type that represents the mode of the data at the current stage of the pipeline
    #    - here we set the final mode for each type of profile, so that they can be retrieved in their original mode for plotting
    #      eg, if profiles are converted into CCF at the Residual stage, we keep the information that disk-integrated profiles were in spectral mode
    for key in ['DI','Res','Intr','Atm']:data_dic[key]['type'] = {inst:deepcopy(gen_dic['type'][inst]) for inst in gen_dic['type']}
    for inst in gen_dic['type']:
        if 'spec' in gen_dic['type'][inst]: 
        
            #Set general condition that spectral data is converted into CCFs
            if gen_dic['DI_CCF']:
                data_dic['DI']['type'][inst]='CCF'
            if gen_dic['DI_CCF'] or gen_dic['Intr_CCF']:
                data_dic['Res']['type'][inst]='CCF'          
                data_dic['Intr']['type'][inst]='CCF'
            if gen_dic['DI_CCF'] or gen_dic['Atm_CCF']:
                data_dic['Atm']['type'][inst]='CCF'

            #Set general condition that 2D spectral data is converted into 1D spectral data
            if gen_dic['spec_1D_DI']:
                data_dic['DI']['type'][inst]='spec1D'
            if gen_dic['spec_1D_DI'] or gen_dic['spec_1D_Intr']:
                data_dic['Res']['type'][inst]='spec1D'
                data_dic['Intr']['type'][inst]='spec1D'
            if gen_dic['spec_1D_DI'] or gen_dic['spec_1D_Atm']:
                data_dic['Atm']['type'][inst]='spec1D'

    #Set general condition to activate binning modules
    for mode in ['','multivis']:
        gen_dic['bin'+mode]=False
        gen_dic['calc_bin'+mode]=False
        for data_type_gen in ['DI','Intr','Atm']:
            gen_dic['bin'+mode] |= ( gen_dic[data_type_gen+'bin'+mode] | gen_dic['fit_'+data_type_gen+'bin'+mode])
            gen_dic['calc_bin'+mode] |= ( gen_dic['calc_'+data_type_gen+'bin'+mode] | gen_dic['calc_fit_'+data_type_gen+'bin'+mode])

    #Set general condition for profiles fits
    for key in ['DI','Intr','Atm']:
        gen_dic['fit_'+key+'_gen'] = gen_dic['fit_'+key] | gen_dic['fit_'+key+'bin'] | gen_dic['fit_'+key+'binmultivis']

    #Automatic continuum and fit range
    for key in ['DI','Intr','Atm']:
        if gen_dic['fit_'+key+'_gen']:
            if ('cont_range' not in data_dic[key]):data_dic[key]['cont_range']={}
            if ('fit_range' not in data_dic[key]):data_dic[key]['fit_range']={}
    if (gen_dic['res_data']) and ('cont_range' not in data_dic['Res']):data_dic['Res']['cont_range']={}

    #Automatic activation/deactivation
    if gen_dic['pca_ana']:gen_dic['intr_data'] = True
    if gen_dic['intr_data']:
        if not gen_dic['res_data']:
            print('    Automatic activation of residual profile extraction')
            gen_dic['res_data'] = True
        if not gen_dic['flux_sc']:
            print('    Automatic activation of flux scaling calculation')
            gen_dic['flux_sc'] = True

    else:
        for key in ['map_Intr_prof','all_intr_data','sp_intr','CCF_Intr']:plot_dic[key]=''

    #Deactivate conditions
    if gen_dic['DIbin']==False:
        data_dic['DI']['fit_MCCFout']=False
    if (not gen_dic['specINtype']) or (gen_dic['DI_CCF']):plot_dic['spectral_LC']=''
    if (not gen_dic['res_data']):
        for key in ['map_Res_prof','sp_loc','CCF_Res']:plot_dic[key]=''    
    if not gen_dic['pl_atm']:
        for key in ['map_Atm_prof','sp_atm','CCFatm']:plot_dic[key]=''  
    else:
        if gen_dic['Intr_CCF']:stop('Atmospheric extraction cannot be performed after Res./Intr. CCF conversion')
    if gen_dic['Intr_CCF'] and (gen_dic['pl_atm']) and (any('spec' in s for s in data_dic['Atm']['type'].values())) and (data_dic['Intr']['mode_loc_data_corr'] in ['Intrbin','rec_prof']):stop('Intrinsic profiles cannot be converted into CCFs if also requested for planetary spectra extraction)')
    
    #Telluric condition
    if (not gen_dic['specINtype']):
        gen_dic['tell_weight']=False
        gen_dic['corr_tell']=False
    else:
        gen_dic['tell_weight'] &= gen_dic['corr_tell']
    
    #Set general condition to calculate master spectrum of the disk-integrated star and use it in weighted averages
    #    - the master needs to be calculated if weighing is needed for one of the modules below
    if gen_dic['DImast_weight']:gen_dic['DImast_weight'] |= (gen_dic['res_data'] | (gen_dic['loc_data_corr'] &  (data_dic['Intr']['mode_loc_data_corr'] in ['DIbin','Intrbin'])) | gen_dic['spec_1D'] | gen_dic['bin'] | gen_dic['binmultivis'])
    if gen_dic['DImast_weight'] and gen_dic['calc_DImast']:gen_dic['calc_DImast'] =  gen_dic['calc_res_data'] | (gen_dic['calc_loc_data_corr'] &  (data_dic['Intr']['mode_loc_data_corr'] in ['DIbin','Intrbin'])) | gen_dic['calc_spec_1D'] | gen_dic['calc_bin'] | gen_dic['calc_binmultivis']
  
    #Set general conditions to activate multi-instrument modules     # Stage Théo 
    if gen_dic['fit_IntrProf'] or gen_dic['fit_IntrProp'] or gen_dic['fit_AtmProf'] or gen_dic['fit_AtmProp'] or gen_dic['fit_ResProf'] : gen_dic['multi_inst']=True
    else:gen_dic['multi_inst']=False

    #Import bin size dictionary
    gen_dic['pix_size_v']=return_pix_size()    

    #Additional properties
    data_prop={}

    #Standard-deviation curves with bin size for the out-of-transit residual CCFs
    #    - defining the maximum size of the binning window, and the binning size for the sliding window (we will average the bins in windows of width bin_size from 1 to 40 (arbitrary))
    if gen_dic['scr_search']:
        gen_dic['scr_srch_max_binwin']=40.     
        gen_dic['scr_srch_nperbins'] = 1.+np.arange(gen_dic['scr_srch_max_binwin'])


    #Default options for continuum calculation
    if gen_dic['mask_permpeak'] or gen_dic['DI_stcont'] or gen_dic['Intr_stcont']:
        for inst in data_dic['instrum_list'] :
            
            #Size of rolling window for peak exclusion
            #    - in A
            if (inst not in gen_dic['contin_roll_win']):gen_dic['contin_roll_win'][inst] = 2.
    
            #Size of smoothing window
            #    - in A
            if (inst not in gen_dic['contin_smooth_win']):gen_dic['contin_smooth_win'][inst] = 0.5
    
            #Size of local maxima window
            #    - in A
            if (inst not in gen_dic['contin_locmax_win']):gen_dic['contin_locmax_win'][inst] = 0.5
    
            #Flux/wavelength stretching
            if (inst not in gen_dic['contin_stretch']):gen_dic['contin_stretch'][inst] = 10. 
    
            #Rolling pin radius
            #    - value corresponds to the bluest wavelength of the processed spectra
            if (inst not in gen_dic['contin_pinR']):gen_dic['contin_pinR'][inst] = 5.


    #------------------------------------------------------------------------------
    #Star
    #------------------------------------------------------------------------------
    
    #Conversions and calculations
    star_params=system_param['star']
    star_params['Rstar_km'] = star_params['Rstar']*Rsun
    
    #Spherical star
    if ('f_GD' not in star_params):
        star_params['f_GD']=0.
        star_params['RpoleReq'] = 1.
        
    #Oblate star
    elif star_params['f_GD']>0.:
        print('Star is oblate')
        star_params['RpoleReq']=1.-star_params['f_GD']

    #Stellar equatorial rotation rate (rad/s)
    #    - om = 2*pi/P
    star_params['om_eq'] = star_params['veq']/star_params['Rstar_km']

    #No GD
    if ('beta_GD' not in star_params):star_params['beta_GD']=0.
    if ('Tpole' not in star_params):star_params['Tpole']=0.

    #Conversions
    star_params['istar_rad']=star_params['istar']*np.pi/180.
    star_params['cos_istar']=np.cos(star_params['istar_rad'])
    star_params['vsini']=star_params['veq']*np.sin(star_params['istar_rad'])    #km/s
    
    #Default parameters
    for key in ['alpha_rot','beta_rot','c1_CB','c2_CB','c3_CB','c1_pol','c2_pol','c3_pol','c4_pol']:
        if key not in star_params:star_params[key] = 0.

    #Conversion factor from the LOS velocity output (/Rstar/h) to RV in km/s
    star_params['RV_conv_fact']=-star_params['Rstar_km']/3600.

    #Macroturbulence
    if theo_dic['mac_mode'] is not None:
        if 'rt' in theo_dic['mac_mode']:
            theo_dic['mac_mode_func'] = calc_macro_ker_rt
            if theo_dic['mac_mode']=='rt_iso':
                star_params['A_T']=star_params['A_R']
                star_params['ksi_T']=star_params['ksi_R']   
        elif 'anigauss' in theo_dic['mac_mode']:
            theo_dic['mac_mode_func'] = calc_macro_ker_anigauss
            if theo_dic['mac_mode']=='anigauss_iso':
                star_params['eta_T']=star_params['eta_R']

    #------------------------------------------------------------------------------
    #Planets
    #------------------------------------------------------------------------------
    
    #Planets in the system
    gen_dic['all_pl'] = [pl_loc for pl_loc in system_param.keys() if pl_loc!='star']

    #Planets considered for transit
    gen_dic['studied_pl'] = list(gen_dic['transit_pl'].keys()) 

    #Keplerian motion
    if ('kepl_pl' not in gen_dic):gen_dic['kepl_pl']='all'
    if (gen_dic['kepl_pl']=='all'):
        print('Accounting for Keplerian motion from all planets')
        gen_dic['kepl_pl']=deepcopy(gen_dic['all_pl'])
    
    #Planet properties    
    for pl_loc in list(set(gen_dic['studied_pl']+gen_dic['kepl_pl'])):
        
        #Checking if there is a "-" in a target name
        if '-' in pl_loc:stop('Invalid target name: {}. Target names should not contain a hyphen.'.format(pl_loc))        
        
        PlParam_loc=system_param[pl_loc]
        PlParam_loc['omega_rad']=PlParam_loc['omega_deg']*np.pi/180.
        if ('a' not in PlParam_loc) and ('aRs' in PlParam_loc) and ('Rstar' in star_params):
            PlParam_loc['a']=PlParam_loc['aRs']*star_params['Rstar_km']/AU_1  # in au
        PlParam_loc['period_s'] = PlParam_loc['period']*24.*3600.
        if 'inclination' in PlParam_loc:
            PlParam_loc['inclin_rad']=PlParam_loc['inclination']*np.pi/180.

            #Semi-amplitude of planet orbital motion around the star (approximated with Mp << Mstar) in km/s
            if ('a' in PlParam_loc):
                PlParam_loc['Kp_orb'] = (2.*np.pi/PlParam_loc['period_s'])*np.sin(PlParam_loc['inclin_rad'])*PlParam_loc['a']*AU_1/np.sqrt(1.-PlParam_loc['ecc']**2.)

                #Stellar mass derived from semi-orbital motion around the star
                #    - assuming Mp << Ms : 
                # P^2/a^3 = 4*pi^2/G*Ms 
                # Ms = 4*pi^2*a^3/(G*P^2)
                PlParam_loc['Mstar_orb'] = 4.*np.pi**2.*(PlParam_loc['a']*AU_1*1e3)**3./(Msun*G_usi*PlParam_loc['period_s']**2.)

        if pl_loc in gen_dic['studied_pl']:PlParam_loc['lambda_rad']=PlParam_loc['lambda_proj']*np.pi/180.

        #Calculation of transit center (or time of cunjunction for a non-transiting planet)
        #    - when T0 is not given but Tperiastron is known
        #    - we could calculate directly the true anomaly from Tperi, but all phases in the routine
        # are defined relatively with the transit center
        #    - mean anomaly is counted from pericenter as M(t) = 2pi*(t - Tperi)/P
        # M(Tcenter) = 2pi*(Tcenter - Tperi)/P 
        # Tcenter = Tperi + M(Tcenter)*P/2pi         
        if ('TCenter' not in PlParam_loc):    
            PlParam_loc['TCenter']=PlParam_loc['Tperi']
            if (PlParam_loc['ecc']>0):
                PlParam_loc['Mean_anom_TR'] = Mean_anom_TR_calc(PlParam_loc['ecc'],PlParam_loc['omega_rad']) 
                PlParam_loc['TCenter']+=(PlParam_loc['Mean_anom_TR']*PlParam_loc["period"]/(2.*np.pi))
  
        #Keplerian semi-amplitude from the studied planet (km/s) 
        PlParam_loc['Kstar_kms']=PlParam_loc['Kstar']/1000. if 'Kstar' in PlParam_loc else calc_Kpl(PlParam_loc,star_params)/1000.

        #Orbital frequency, in year-1
        PlParam_loc['omega_p']=2.*np.pi*365.2425/PlParam_loc['period']

    #Calculating theoretical properties of the planet-occulted regions
    if (not gen_dic['theoPlOcc']) and ((('CCF' in data_dic['Res']['type'].values()) and (gen_dic['fit_Intr'])) or (gen_dic['align_Intr']) or (gen_dic['calc_pl_atm'])):
        gen_dic['theoPlOcc']=True

    
    #Oversampling factor for values from planet-occulted regions
    #    - uses the nominal planet-to-star radius ratios, which must correspond to the band from which local properties are derived
    theo_dic['d_oversamp']={}
    for pl_loc in theo_dic['n_oversamp']:
        if (theo_dic['n_oversamp'][pl_loc]>0.):
            theo_dic['d_oversamp'][pl_loc] = data_dic['DI']['system_prop']['achrom'][pl_loc][0]/theo_dic['n_oversamp'][pl_loc]
         
    #Set flag for errors on estimates for local stellar profiles (depending on whether they are derived from data or models)
    if data_dic['Intr']['mode_loc_data_corr'] in ['DIbin','Intrbin']:data_dic['Intr']['cov_loc_star']=True
    else:data_dic['Intr']['cov_loc_star']=False    

    #Transit and stellar surfce chromatic properties
    #    - must be defined for data processing, transit scaling, calculation of planet properties
    #    - we remove the 'chrom' dictionary if no spectral data is used as input or if a single band is defined
    for ideg in range(2,5):
        if 'LD_u'+str(ideg) not in data_dic['DI']['system_prop']['achrom']:data_dic['DI']['system_prop']['achrom']['LD_u'+str(ideg)] = [0.]
    if ('GD_dw' in data_dic['DI']['system_prop']['achrom']):
        star_params['GD']=True
        print('Star is gravity-darkened')
    else:star_params['GD']=False
    data_dic['DI']['system_prop']['achrom']['w']=[None]
    data_dic['DI']['system_prop']['achrom']['nw']=1
    data_dic['DI']['system_prop']['achrom']['cond_in_RpRs']={}
    data_dic['DI']['system_prop']['chrom_mode'] = 'achrom'
    if ('chrom' in data_dic['DI']['system_prop']):
        if (not gen_dic['specINtype']) or (len(data_dic['DI']['system_prop']['chrom']['w'])==1):data_dic['DI']['system_prop'].pop('chrom')
        else:
            data_dic['DI']['system_prop']['chrom_mode'] = 'chrom'
            data_dic['DI']['system_prop']['chrom']['w'] = np.array(data_dic['DI']['system_prop']['chrom']['w'])
            data_dic['DI']['system_prop']['chrom']['nw']=len(data_dic['DI']['system_prop']['chrom']['w'])
            data_dic['DI']['system_prop']['chrom']['cond_in_RpRs']={}
            
            #Typical scale of chromatic variations
            w_edge = def_edge_tab(data_dic['DI']['system_prop']['chrom']['w'][None,:][None,:])[0,0]    
            data_dic['DI']['system_prop']['chrom']['dw'] = w_edge[1::]-w_edge[0:-1]
            data_dic['DI']['system_prop']['chrom']['med_dw'] = np.median(data_dic['DI']['system_prop']['chrom']['dw'])

        
    #Definition of grids discretizing planets disk to calculate planet-occulted properties
    theo_dic['x_st_sky_grid_pl']={}
    theo_dic['y_st_sky_grid_pl']={}
    theo_dic['Ssub_Sstar_pl']={}
    data_dic['DI']['system_prop']['RpRs_max']={}
    if ('nsub_Dpl' not in theo_dic):theo_dic['nsub_Dpl']={}
    for pl_loc in gen_dic['studied_pl']:

        #Largest possible planet size
        if ('chrom' in data_dic['DI']['system_prop']):data_dic['DI']['system_prop']['RpRs_max'][pl_loc] = np.max(data_dic['DI']['system_prop']['achrom'][pl_loc]+data_dic['DI']['system_prop']['chrom'][pl_loc])
        else:data_dic['DI']['system_prop']['RpRs_max'][pl_loc] = data_dic['DI']['system_prop']['achrom'][pl_loc][0]

        #Default grid scaled from a 51x51 grid for a hot Jupiter transiting a solar-size star
        #    - dsub_ref = (2*Rjup/Rsun)*(1/51)
        #      nsub_Dpl = int(2*RpRs/dsub_ref) = int( 51*RpRs*Rsun/Rjup )        
        if (pl_loc not in theo_dic['nsub_Dpl']):
            theo_dic['nsub_Dpl'][pl_loc] =int( 51.*data_dic['DI']['system_prop']['RpRs_max'][pl_loc]*Rsun/Rjup ) 
            print('Default nsub_Dpl = '+str(theo_dic['nsub_Dpl'][pl_loc])+' for ',pl_loc)

        #Corresponding planet grid
        _,theo_dic['Ssub_Sstar_pl'][pl_loc],theo_dic['x_st_sky_grid_pl'][pl_loc],theo_dic['y_st_sky_grid_pl'][pl_loc],r_sub_pl2=occ_region_grid(data_dic['DI']['system_prop']['RpRs_max'][pl_loc],theo_dic['nsub_Dpl'][pl_loc])  
        
        #Identification of cells within the nominal and chromatic planet radii
        data_dic['DI']['system_prop']['achrom']['cond_in_RpRs'][pl_loc] = [(r_sub_pl2<data_dic['DI']['system_prop']['achrom'][pl_loc][0]**2.)]
        if ('chrom' in data_dic['DI']['system_prop']):
            data_dic['DI']['system_prop']['chrom']['cond_in_RpRs'][pl_loc]={}
            for iband in range(data_dic['DI']['system_prop']['chrom']['nw']):
                data_dic['DI']['system_prop']['chrom']['cond_in_RpRs'][pl_loc][iband] = (r_sub_pl2<data_dic['DI']['system_prop']['chrom'][pl_loc][iband]**2.)

    #------------------------------------------------------------------------------
    #Generic path names
    gen_dic['main_pl_text'] = ''
    for pl_loc in gen_dic['studied_pl']:gen_dic['main_pl_text']+=pl_loc
    gen_dic['save_data_dir'] = gen_dic['save_dir']+gen_dic['main_pl_text']+'_Saved_data/'
    gen_dic['save_plot_dir'] = gen_dic['save_dir']+gen_dic['main_pl_text']+'_Plots/'
    gen_dic['add_txt_path']={'DI':'','Intr':'','Res':'','Atm':data_dic['Atm']['pl_atm_sign']+'/'}
    gen_dic['data_type_gen']={'DI':'DI','Res':'Res','Intr':'Intr','Absorption':'Atm','Emission':'Atm'}
    gen_dic['type_name']={'DI':'disk-integrated','Res':'residual','Intr':'intrinsic','Atm':'atmospheric','Absorption':'absorption','Emission':'emission'}    


    #------------------------------------------------------------------------------------------------------------------------
    #Model star
    #------------------------------------------------------------------------------------------------------------------------
    grid_type=[]
    if any('spec' in s for s in data_dic['DI']['type'].values()):grid_type+=['spec']
    if ('CCF' in data_dic['DI']['type'].values()):grid_type+=['ccf']    

    #Calculation of total stellar flux for use in simulated light curves
    if gen_dic['calc_flux_sc'] and (data_dic['DI']['transit_prop']['nsub_Dstar'] is not None): 
        model_star('Ftot',theo_dic,grid_type,data_dic['DI']['system_prop'],data_dic['DI']['transit_prop']['nsub_Dstar'],star_params) 
                    
    #Definition of model stellar grid to calculate local or disk-integrated properties
    #    - used througout the pipeline, unless stellar properties are fitted
    if gen_dic['theoPlOcc'] or (gen_dic['theo_spots']) or (gen_dic['fit_DI_gen'] and (('custom' in data_dic['DI']['model'].values()) or ('RT_ani_macro' in data_dic['DI']['model'].values()))) or gen_dic['mock_data'] \
        or gen_dic['fit_ResProf'] or gen_dic['correct_spots'] or gen_dic['fit_IntrProf'] or gen_dic['loc_data_corr']:
            
        #Stellar grid
        model_star('grid',theo_dic,grid_type,data_dic['DI']['system_prop'],theo_dic['nsub_Dstar'],star_params) 
       
        #Theoretical atmosphere
        cond_st_atm = False
        if gen_dic['mock_data']:
            for inst in mock_dic['intr_prof']:
                if (mock_dic['intr_prof'][inst]['mode']=='theo' ):cond_st_atm = True
        if (gen_dic['fit_DI_gen'] and ('custom' in data_dic['DI']['model'].values())):
            for inst in data_dic['DI']['mod_def']:
                if (data_dic['DI']['mod_def'][inst]['mode'] == 'theo'):cond_st_atm = True
        if gen_dic['fit_IntrProf'] and (glob_fit_dic['IntrProf']['mode'] == 'theo'):cond_st_atm = True 
        if gen_dic['loc_data_corr'] and (data_dic['Intr']['mode_loc_data_corr'] in ['glob_mod','indiv_mod']):
            if data_dic['Intr']['opt_loc_data_corr'][data_dic['Intr']['mode_loc_data_corr']]['mode']=='theo':cond_st_atm = True  
        if cond_st_atm:
            
            #Calculate grid
            if theo_dic['st_atm']['calc']:
                
                #Imports
                from pysme import sme as SME
                from pysme.linelist.vald import ValdFile
                from pysme.abund         import Abund
                
                #Atmosphere structure
                theo_dic['sme_grid'] = SME.SME_Structure()
                
                #Stellar atmosphere model
                #    - we keep the defaults for sme.atmo.method = "grid" and sme.atmo.geom = "PP"
                theo_dic['sme_grid'].atmo.source = theo_dic['st_atm']['atm_model']+'.sav'
                
                #NLTE departure
                if ('nlte' in theo_dic['st_atm']) and (len(theo_dic['st_atm']['nlte'])>0):
                    for sp_nlte in theo_dic['st_atm']['nlte']:theo_dic['sme_grid'].nlte.set_nlte(sp_nlte,theo_dic['st_atm']['nlte'][sp_nlte]+'.grd')   
                
                #Set nominal model properties
                #    - vsini is set to 0 so that intrinsic profiles are kept aligned
                theo_dic['sme_grid']['teff'] = star_params['Tpole']    
                theo_dic['sme_grid']['logg'] = star_params['logg']                
                theo_dic['sme_grid']['vsini'] = 0.
                theo_dic['sme_grid']['vmic'] = 0.            
                theo_dic['sme_grid']['vmac'] = 0.            
    
                #Wavelength and mu grid of the synthetic spectra
                #    - in A, in the star rest frame
                theo_dic['sme_grid']['n_wav'] = int((theo_dic['st_atm']['wav_max']-theo_dic['st_atm']['wav_min'])/theo_dic['st_atm']['dwav'])
                sme_grid_wave = np.linspace(theo_dic['st_atm']['wav_min'],theo_dic['st_atm']['wav_max'],theo_dic['sme_grid']['n_wav']) 
                theo_dic['sme_grid']['edge_bins'] = def_edge_tab(sme_grid_wave[None,:][None,:])[0,0]
                theo_dic['sme_grid']['wave'] = sme_grid_wave
                theo_dic['sme_grid']['mu_grid'] = theo_dic['st_atm']['mu_grid']
                theo_dic['sme_grid']['n_mu'] = len(theo_dic['st_atm']['mu_grid'])
                
                #Retrieve linelist and limit to range of spectral grid
                theo_dic['sme_grid']['linelist'] = ValdFile(theo_dic['st_atm']['linelist'])
                wlcent = theo_dic['sme_grid']['linelist']['wlcent']
                cond_within = (wlcent>=theo_dic['st_atm']['wav_min']-5.) & (wlcent<=theo_dic['st_atm']['wav_max']+5.)
                if True not in cond_within:stop('No VALD transitions within requested range')
                theo_dic['sme_grid']['linelist'] = theo_dic['sme_grid']['linelist'][cond_within]
    
                #Abundances
                #    - set by default to solar, from Asplund+2009
                #    - specific abundances are defined as A(X) = log10( N(X)/N(H) ) + 12 
                #    - overall metallicity yields A(X) = Anominal(X) + [M/H] for X != H and He   
                theo_dic['sme_grid']['abund'] = Abund.solar()
                if ('abund' in theo_dic['st_atm']) and (len(theo_dic['st_atm']['abund'])>0):
                    for sp_abund in theo_dic['st_atm']['abund']:theo_dic['sme_grid']['abund'][sp_abund]=theo_dic['st_atm']['abund'][sp_abund]
                theo_dic['sme_grid']['monh'] = theo_dic['st_atm']['MovH'] if 'MovH' in theo_dic['st_atm'] else 0
    
                #Intrinsic profile grid
                gen_theo_intr_prof(theo_dic['sme_grid'])

                #Save grid
                datasave_npz(gen_dic['save_data_dir']+'Introrig_prop/IntrProf_grid',{'sme_grid':theo_dic['sme_grid']})

            #Retrieving grid
            else:theo_dic['sme_grid'] = dataload_npz(gen_dic['save_data_dir']+'Introrig_prop/IntrProf_grid')['sme_grid']

    #------------------------------------------------------------------------------------------------------------------------

    #Mask used to compute CCF on stellar lines
    if len(gen_dic['CCF_mask'])>0:
        gen_dic['CCF_mask_wav']={}
        gen_dic['CCF_mask_wgt']={}

    #Condition to exclude ranges contaminated by planetary absorption
    #    - independent of atmospheric signal extraction
    if len(data_dic['Atm']['no_plrange'])>0:
        data_dic['Atm']['exc_plrange']=True
        data_dic['Atm']['plrange']=np.array(data_dic['Atm']['plrange'])
    else:data_dic['Atm']['exc_plrange']=False

    #Properties associated with planetary signal extraction
    if gen_dic['calc_pl_atm']:
  
        #Deactivate signal reduction for input CCFs
        if (not gen_dic['specINtype']):gen_dic['atm_CCF']=False

    #Mask used to compute atmospheric CCFs and/or exclude atmospheric planetary ranges
    #    - weights set to 1 unless requested
    if data_dic['Atm']['CCF_mask'] is not None:
        
        #Upload CCF mask
        ext = data_dic['Atm']['CCF_mask'].split('.')[-1]
        if (ext=='fits'):
            hdulist = fits.open(data_dic['Atm']['CCF_mask'])
            data_loc = hdulist[1].data
            data_dic['Atm']['CCF_mask_wav'] = data_loc['lambda']
            data_dic['Atm']['CCF_mask_wgt'] = data_loc['contrast'] if data_dic['Atm']['use_maskW'] == True else np.repeat(1.,len(data_dic['Atm']['CCF_mask_wav']))        
            hdulist.close()
        elif (ext=='csv'):
            data_loc = np.genfromtxt(data_dic['Atm']['CCF_mask'], delimiter=',', names=True)
            data_dic['Atm']['CCF_mask_wav'] = data_loc['wave']
            data_dic['Atm']['CCF_mask_wgt'] = data_loc['contrast'] if data_dic['Atm']['use_maskW'] == True else np.repeat(1.,len(data_dic['Atm']['CCF_mask_wav'])) 
        elif (ext in ['txt','dat']):
            data_loc = np.loadtxt(data_dic['Atm']['CCF_mask']).T            
            data_dic['Atm']['CCF_mask_wav'] = data_loc[0]
            data_dic['Atm']['CCF_mask_wgt'] = data_loc[1]        
        else:
            stop('CCF mask extension TBD') 

    #------------------------------------------------------------------------------------------------------------------------

    #Create directories where data is saved/restored
    if (not os_system.path.exists(gen_dic['save_data_dir']+'Processed_data/Global/')):os_system.makedirs(gen_dic['save_data_dir']+'Processed_data/Global/')  
    if gen_dic['specINtype']:
        if gen_dic['gcal'] and (not os_system.path.exists(gen_dic['save_data_dir']+'Processed_data/Calibration/')):os_system.makedirs(gen_dic['save_data_dir']+'Processed_data/Calibration/')  
        if gen_dic['CCF_from_sp'] and (not os_system.path.exists(gen_dic['save_data_dir']+'Processed_data/CCFfromSpec/')):os_system.makedirs(gen_dic['save_data_dir']+'Processed_data/CCFfromSpec/')  
        if (gen_dic['corr_data']):
            corr_path = gen_dic['save_data_dir']+'Corr_data/'
            if (not os_system.path.exists(corr_path)):os_system.makedirs(corr_path)
            if (gen_dic['corr_tell']) and (not os_system.path.exists(corr_path+'Tell/')):os_system.makedirs(corr_path+'Tell/')         
            if (gen_dic['glob_mast']) and (not os_system.path.exists(corr_path+'Global_Master/')):os_system.makedirs(corr_path+'Global_Master/')    
            if (gen_dic['corr_Fbal']) and (not os_system.path.exists(corr_path+'Fbal/')):os_system.makedirs(corr_path+'Fbal/')
            if (gen_dic['corr_FbalOrd']) and (not os_system.path.exists(corr_path+'Fbal/Orders/')):os_system.makedirs(corr_path+'Fbal/Orders/')            
            if (gen_dic['corr_Ftemp']) and (not os_system.path.exists(corr_path+'Ftemp/')):os_system.makedirs(corr_path+'Ftemp/')
            if (gen_dic['corr_cosm']) and (not os_system.path.exists(corr_path+'Cosm/')):os_system.makedirs(corr_path+'Cosm/')
            if (gen_dic['mask_permpeak']) and (not os_system.path.exists(corr_path+'Permpeak/')):os_system.makedirs(corr_path+'Permpeak/')
            if (gen_dic['corr_wig']): 
                if not os_system.path.exists(corr_path+'Wiggles/'):os_system.makedirs(corr_path+'Wiggles/')
                
                #Condition for exposure analysis
                gen_dic['wig_exp_ana'] = gen_dic['wig_exp_init']['mode'] | gen_dic['wig_exp_samp']['mode'] | gen_dic['wig_exp_nu_ana']['mode'] | gen_dic['wig_exp_fit']['mode'] | gen_dic['wig_exp_point_ana']['mode'] 

                if gen_dic['wig_exp_ana']  and (not os_system.path.exists(corr_path+'Wiggles/Exp_fit/')):os_system.makedirs(corr_path+'Wiggles/Exp_fit/') 
                if gen_dic['wig_vis_fit']['mode'] and (not os_system.path.exists(corr_path+'Wiggles/Vis_fit/')):os_system.makedirs(corr_path+'Wiggles/Vis_fit/')
                if gen_dic['wig_corr']['mode'] and (not os_system.path.exists(corr_path+'Wiggles/Data/')):os_system.makedirs(corr_path+'Wiggles/Data/')
            if (gen_dic['corr_fring']) and (not os_system.path.exists(corr_path+'Fring/')):os_system.makedirs(corr_path+'Fring/')        
            if (gen_dic['trim_spec']) and (not os_system.path.exists(corr_path+'Trim/')):os_system.makedirs(corr_path+'Trim/')         
    if (gen_dic['detrend_prof']) and (not os_system.path.exists(gen_dic['save_data_dir']+'Detrend_prof/')):os_system.makedirs(gen_dic['save_data_dir']+'Detrend_prof/') 
    if (gen_dic['flux_sc']) and (not os_system.path.exists(gen_dic['save_data_dir']+'Scaled_data/')):os_system.makedirs(gen_dic['save_data_dir']+'Scaled_data/')
    if gen_dic['DImast_weight'] and (not os_system.path.exists(gen_dic['save_data_dir']+'DI_data/Master/')):os_system.makedirs(gen_dic['save_data_dir']+'DI_data/Master/')
    if (gen_dic['res_data']) and (not os_system.path.exists(gen_dic['save_data_dir']+'Res_data/')):os_system.makedirs(gen_dic['save_data_dir']+'Res_data/')
    if gen_dic['pca_ana'] and (not os_system.path.exists(gen_dic['save_data_dir']+'PCA_results/')):os_system.makedirs(gen_dic['save_data_dir']+'PCA_results/')   
    if (gen_dic['intr_data']) and (not os_system.path.exists(gen_dic['save_data_dir']+'Intr_data/')):os_system.makedirs(gen_dic['save_data_dir']+'Intr_data/')
    if gen_dic['loc_data_corr']:
        if (not os_system.path.exists(gen_dic['save_data_dir']+'Loc_estimates/')):os_system.makedirs(gen_dic['save_data_dir']+'Loc_estimates/')        
        if (not os_system.path.exists(gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['mode_loc_data_corr']+'/')):os_system.makedirs(gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['mode_loc_data_corr']+'/')  
    if (gen_dic['pl_atm']):
        if (not os_system.path.exists(gen_dic['save_data_dir']+'Atm_data/')):os_system.makedirs(gen_dic['save_data_dir']+'Atm_data/')        
        if (not os_system.path.exists(gen_dic['save_data_dir']+'Atm_data/'+data_dic['Atm']['pl_atm_sign']+'/')):os_system.makedirs(gen_dic['save_data_dir']+'Atm_data/'+data_dic['Atm']['pl_atm_sign']+'/')
    
    for data_type in ['DI','Intr','Atm']:
        if gen_dic['align_'+data_type] and (not os_system.path.exists(gen_dic['save_data_dir']+'Aligned_'+data_type+'_data/'+gen_dic['add_txt_path'][data_type])):os_system.makedirs(gen_dic['save_data_dir']+'Aligned_'+data_type+'_data/'+gen_dic['add_txt_path'][data_type])    
        if gen_dic['spec_1D_'+data_type]:
            if (not os_system.path.exists(gen_dic['save_data_dir']+data_type+'_data/1Dfrom2D/'+gen_dic['add_txt_path'][data_type])):os_system.makedirs(gen_dic['save_data_dir']+data_type+'_data/1Dfrom2D/'+gen_dic['add_txt_path'][data_type])   
            if (data_type=='Intr') and (not os_system.path.exists(gen_dic['save_data_dir']+'Res_data/1Dfrom2D/'+gen_dic['add_txt_path'][data_type])):os_system.makedirs(gen_dic['save_data_dir']+'Res_data/1Dfrom2D/'+gen_dic['add_txt_path'][data_type])  
        if (gen_dic[data_type+'bin'] or gen_dic[data_type+'binmultivis']) and (not os_system.path.exists(gen_dic['save_data_dir']+data_type+'bin_data/'+gen_dic['add_txt_path'][data_type])):os_system.makedirs(gen_dic['save_data_dir']+data_type+'bin_data/'+gen_dic['add_txt_path'][data_type])
        if (gen_dic['fit_'+data_type+'bin'] or gen_dic['fit_'+data_type+'binmultivis']) and (not os_system.path.exists(gen_dic['save_data_dir']+data_type+'bin_prop/'+gen_dic['add_txt_path'][data_type])):os_system.makedirs(gen_dic['save_data_dir']+data_type+'bin_prop/'+gen_dic['add_txt_path'][data_type])    
        if gen_dic['def_'+data_type+'masks'] and (not os_system.path.exists(gen_dic['save_data_dir']+'CCF_masks_'+data_type+'/'+gen_dic['add_txt_path'][data_type])):os_system.makedirs(gen_dic['save_data_dir']+'CCF_masks_'+data_type+'/'+gen_dic['add_txt_path'][data_type])    
        if gen_dic[data_type+'_CCF']:
            if (not os_system.path.exists(gen_dic['save_data_dir']+data_type+'_data/CCFfromSpec/'+gen_dic['add_txt_path'][data_type])):os_system.makedirs(gen_dic['save_data_dir']+data_type+'_data/CCFfromSpec/'+gen_dic['add_txt_path'][data_type])    
            if (data_type=='Intr') and (not os_system.path.exists(gen_dic['save_data_dir']+'Res_data/CCFfromSpec/'+gen_dic['add_txt_path'][data_type])):os_system.makedirs(gen_dic['save_data_dir']+'Res_data/CCFfromSpec/'+gen_dic['add_txt_path'][data_type]) 

    if (gen_dic['fit_DI'] or gen_dic['sav_keywords']) and (not os_system.path.exists(gen_dic['save_data_dir']+'DIorig_prop/')):os_system.makedirs(gen_dic['save_data_dir']+'DIorig_prop/')        
    if ((gen_dic['fit_Intr']) or (gen_dic['theoPlOcc'])) and (not os_system.path.exists(gen_dic['save_data_dir']+'Introrig_prop/')):os_system.makedirs(gen_dic['save_data_dir']+'Introrig_prop/')
    if (gen_dic['fit_Atm']) and (not os_system.path.exists(gen_dic['save_data_dir']+'Atmorig_prop/'+data_dic['Atm']['pl_atm_sign']+'/')):os_system.makedirs(gen_dic['save_data_dir']+'Atmorig_prop/'+data_dic['Atm']['pl_atm_sign']+'/')

    for key in ['IntrProp','IntrProf','AtmProf','AtmProp']:
        if (gen_dic['fit_'+key]) and (not os_system.path.exists(gen_dic['save_data_dir']+'Joined_fits/'+key+'/')):os_system.makedirs(gen_dic['save_data_dir']+'Joined_fits/'+key+'/') 
    
    # # Stage Théo
    # if gen_dic['correct_spots'] and (not os_system.path.exists(gen_dic['save_data_dir']+'Spot_corr_DI_data/'+gen_dic['add_txt_path']['DI'])) :
    #     os_system.makedirs(gen_dic['save_data_dir']+'Spot_corr_DI_data/'+gen_dic['add_txt_path']['DI'])   

    return coord_dic,data_prop











'''
Definition of grid discretizing planet disk
    - defined in the 'inclined' star frame
      X axis is parallel to the star equator
      Y axis is the projected spin axis
      Z axis is the LOS
'''
def occ_region_grid(RpRs,nsub_Dpl):

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





'''
Functions to define model stellar grid
'''
def model_star(mode,grid_dic,grid_type,system_prop_in,nsub_Dstar,star_params):
    coord_grid = {}
    
    #Subcell width (in units of Rstar) and surface (in units of Rstar^2 and pi*Rstar^2) 
    d_sub=2./nsub_Dstar      
    Ssub_Sstar=d_sub*d_sub/np.pi    

    #Coordinates of cells discretizing the square enclosing the star
    #    - coordinates are in the sky-projected star frame ('inclined' star frame)
    #    - X axis = projection of stellar equator in the plane of sky
    #      Y axis = projection of the normal to the stellar spin axis
    #      Z axis = line of sight 
    #    - in units of stellar radius
    #    - we first define a regular grid in the sky-projected star frame, rather than in the star frame, because it is the one that is seen during transit and the conversion from star->star inclined would make it irregular in y
    cen_sub=-1.+(np.arange(nsub_Dstar)+0.5)*d_sub            
    xy_st_sky_grid=np.array(list(it_product(cen_sub,cen_sub))) 
    coord_grid['x_st_sky'] = xy_st_sky_grid[:,0] 
    coord_grid['y_st_sky'] = xy_st_sky_grid[:,1]

    #Coordinates in the sky-projected star rest frame
    nsub_star = calc_st_sky(coord_grid,star_params)

    #Storing for model fits
    if mode=='grid':
        grid_dic.update({'Ssub_Sstar' : Ssub_Sstar,'nsub_star' : nsub_star})
        for key in ['x','y','z','r2']: grid_dic[key+'_st_sky'] = coord_grid[key+'_st_sky'] 
        for key in ['x','y','z']: grid_dic[key+'_st'] = coord_grid[key+'_st'] 
        grid_dic['r_proj'] = np.sqrt(coord_grid['r2_st_sky'])  
 
    #Spectral grid
    #    - the grid is used to model disk-integrated profile and thus only need to be chromatic if they are not in CCF format
    #    - all tables nonetheless have the same structure in ncell x nband
    grid_type_eff = ['achrom']
    if ('spec' in grid_type) and ('chrom' in system_prop_in):grid_type_eff+=['chrom']
    for key_type in grid_type_eff:

        #Intensity grid
        ld_grid_star,gd_grid_star,mu_grid_star,_,Fsurf_grid_star,Ftot_star,Istar_norm = calc_Isurf_grid(range(system_prop_in[key_type]['nw']),nsub_star,system_prop_in[key_type],coord_grid,star_params,Ssub_Sstar)
    
        #Storing for model fits
        if mode=='grid':
            grid_dic.update({'Istar_norm_'+key_type:Istar_norm,
                             'ld_grid_star_'+key_type : ld_grid_star,
                             'gd_grid_star_'+key_type : gd_grid_star,
                             'Fsurf_grid_star_'+key_type : Fsurf_grid_star,
                             'mu_grid_star_'+key_type :  mu_grid_star})

        #Total flux
        #    - correspond to sum(F) = n*dl^2/(pi*Rstar)^2
        elif mode=='Ftot':grid_dic['Ftot_star_'+key_type] = Ftot_star   

    if mode=='grid':grid_dic['mu'] = grid_dic['mu_grid_star_achrom'][:,0]

    return None

'''
Function to update stellar grid
'''
def up_model_star(args,param):

    #Update potentially variable stellar properties
    for ideg in range(1,5):
        if ('LD_u'+str(ideg)) in param:args['system_prop']['achrom'].update({'LD_u'+str(ideg):[param['LD_u'+str(ideg)]]}) 
    args['star_params'].update({'f_GD':param['f_GD'],'RpoleReq':1.-param['f_GD'],'beta_GD':param['beta_GD'],'Tpole':param['Tpole'],
                                'istar_rad':np.arccos(param['cos_istar']),'om_eq' : param['veq']/args['star_params']['Rstar_km']})

    #Update stellar grid
    #    - physical sky-projected grid and broadband intensity variations            
    model_star('grid',args['grid_dic'],[args['type']],args['system_prop'],args['grid_dic']['nsub_Dstar'],args['star_params'])

    return None


'''
Function returning coordinates of stellar occulting cells in the sky-projected star rest frame
'''
def calc_st_sky(coord_grid,star_params):
    
    #Distance of subcells to star center (squared)
    coord_grid['r2_st_sky']=coord_grid['x_st_sky']*coord_grid['x_st_sky']+coord_grid['y_st_sky']*coord_grid['y_st_sky']

    #Oblate star
    #    - the condition is that the 2nd order equation yielding zsky_st for a given (xsky_st,ysky_st) has at least one solution
    if star_params['f_GD']>0.:
        coord_grid['z_st_sky'],cond_in_stphot=calc_zLOS_oblate(coord_grid['x_st_sky'],coord_grid['y_st_sky'],star_params['istar_rad'],star_params['RpoleReq'])[1:3]
        if True in cond_in_stphot:
            for key in ['x','y','z','r2']:coord_grid[key+'_st_sky'] = coord_grid[key+'_st_sky'][cond_in_stphot] 
              
    #Spherical star
    #    - mu and z_st_sky are equivalent in this case
    else:
        cond_in_stphot = ( coord_grid['r2_st_sky'] < 1.) 
        if True in cond_in_stphot:
            for key in ['x','y','r2']:coord_grid[key+'_st_sky'] = coord_grid[key+'_st_sky'][cond_in_stphot] 
            coord_grid['z_st_sky'] = np.sqrt(1.- coord_grid['r2_st_sky'])      

    #Frame conversion from the inclined star frame to the 'star' frame 
    #    - positions in star rest frame (in units of stellar radius)
    nsub_star = np.sum(cond_in_stphot)
    if (nsub_star>0):coord_grid['x_st'],coord_grid['y_st'],coord_grid['z_st']=conv_inclinedStarFrame_to_StarFrame(coord_grid['x_st_sky'],coord_grid['y_st_sky'],coord_grid['z_st_sky'],star_params['istar_rad'])
    
    return nsub_star


'''
Function returning stellar intensity grid
'''
def calc_Isurf_grid(iband_list,ngrid_star,system_prop,coord_grid,star_params,Ssub_Sstar,Istar_norm=1.):

    #Limb-darkening grid and flux emitted by the star
    #    - calculating over the grid as a function of mu, and for each chromatic wavelength given as input  
    #    - mu = cos(theta)), from 1 at the center of the disk to 0 at the limbs), with theta angle between LOS and local normal
    nw = len(iband_list)
    ld_grid_star = np.ones([ngrid_star,nw],dtype=float)
    gd_grid_star = np.ones([ngrid_star,nw],dtype=float)
    mu_grid_star = np.zeros([ngrid_star,nw],dtype=float)
    gd_band={}
    for isubband,iband in enumerate(iband_list):

        #Oblate star with gravity-darkening
        if (star_params['f_GD']>0.):
            if star_params['GD']:gd_band = {'wmin':system_prop['GD_wmin'][iband],'wmax':system_prop['GD_wmax'][iband],'dw':system_prop['GD_dw'][iband]}
            gd_grid_star[:,isubband],mu_grid_star[:,isubband] = calc_GD(coord_grid['x_st'],coord_grid['y_st'],coord_grid['z_st'],star_params,gd_band,coord_grid['x_st_sky'],coord_grid['y_st_sky'])         

        #Spherical star
        else:mu_grid_star[:,isubband] = coord_grid['z_st_sky']

        #Limb-darkening coefficients over the grid
        ld_grid_star[:,isubband]=LD_mu_func(system_prop['LD'][iband],mu_grid_star[:,isubband],LD_coeff_func(system_prop,iband))

    #Intensity and fluxes from cells over the grid
    #    - the chromatic intensity is normalized so that the mean flux integrated over the defined spectral bands, and summed over the full stellar disk, is unity
    # mean(w , sum(cell,F(w,cell))) = 1 
    # mean(w , sum(cell,I(w,cell))) = 1/SpSs 
    #    - specific intensity at disk center is 1 without LD and GD contributions
    #    - normalized by stellar surface, ie F = I0*Scell/Sstar = 1*dl^2/(pi*Rstar)^2 = (dl/Rstar)^2*(1/pi)
    Isurf_grid_star=ld_grid_star*gd_grid_star
    if Istar_norm==1.:
        Itot_star_chrom = np.sum(Isurf_grid_star,axis=0)
        if system_prop['nw']>1:meanItot_star = np.sum(system_prop['dw'][None,:]*Itot_star_chrom)/np.sum(system_prop['dw'])
        else:meanItot_star = np.mean(Itot_star_chrom)
        Istar_norm = (Ssub_Sstar*meanItot_star)
    Isurf_grid_star/=Istar_norm
    Fsurf_grid_star = Isurf_grid_star*Ssub_Sstar  
    
    #Total flux over the full star in each band
    Ftot_star = np.sum(Fsurf_grid_star,axis=0)
   
    return ld_grid_star,gd_grid_star,mu_grid_star,Isurf_grid_star,Fsurf_grid_star,Ftot_star,Istar_norm

'''
Function returning theoretical intrinsic profile grid
'''
def gen_theo_intr_prof(sme_grid):
    
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








'''
Resolving power R
'''    
def return_resolv(inst):                 
    return {        
        'SOPHIE_HR':75000.,  
        'SOPHIE_HE':40000.,  
        'CORALIE':55000.,
        'HARPN':120000.,
        'HARPS':120000.,
        'STIS_E230M':30000.,     
        'STIS_G750L':1280.,         
        'ESPRESSO':140000.,
        'ESPRESSO_MR':70000.,
        'CARMENES_VIS':94600.,
        'NIRPS_HE':75000.,
        'NIRPS_HA':88000.,
        'EXPRES':137500.,
    }[inst]     
'''
Instrumental resolution 
    - corresponds to 
 deltav_instru = c / R
 deltaw_instru = lambda_0/R = lambda_0*deltav_instru/c 
    - equivalent to the FWHM of a Gaussian approximating the LSF 
      call the function with either w or c_light to get the FWHM in the correct space
''' 
def return_FWHM_inst(inst,w_c):
    return w_c/return_resolv(inst)
  
    
'''
Return instrumental bin size (km/s)
'''    
def return_pix_size():             
    return {
            
        #Sophie HE mod: pix_size = 0.0275 A ~ 1.4 km/s at 5890 A 
        'SOPHIE_HE':1.4,  

        #CORALIE:
        #    ordre 10:  deltaV = 1.7240 km/s
        #    ordre 35:  deltaV = 1.7315 km/s
        #    ordre 60:  deltaV = 1.7326 km/s
        #    pouvoir_resol = 55000 -> deltav_instru = 5.45 km/s          
        'CORALIE':1.73,

        #HARPS-N or HARPS: pix_size = 0.016 A ~ 0.8 km/s at 5890 A         
        #    - pouvoir_resol = 120000 -> deltav_instru = 2.6km/s         
        'HARPN':0.82,
        'HARPS':0.82,
        
        #STIS E230M
        #    - size varies in wavelength but is roughly constant in velocity:
        # 0.0496 A at 3021A (=4.920 km/s)
        # 0.0375 A at 2274A (=4.944 km/s)   
        #      we take pix_size ~ 4.93 km/s    
        #    - with pouvoir_resol = 30000 -> deltav_instru = 9.9 km/s (2 bins)
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
        #    - pouvoir_resol = 140000 -> deltav_instru = 2.1km/s           
        'ESPRESSO':0.5,

        #ESPRESSO in MR mode
        'ESPRESSO_MR':1.,
        
        #CARMENES   
        #    pouvoir_resol = 93400 -> deltav_instru = 3.2 km/s   
        #    - 2.8 pixel / FWHM, so that pixel size = 1.1317 km/s             
        'CARMENES_VIS':1.1317,
        
        'NIRPS_HA':1.,
        'NIRPS_HE':1.,  
        
        'EXPRES':0.5
    }      
        
    





#Definition of sub-function to define new bins
#    - here we do not use the resampling function because only the flux is binned and/or the bins are large enough that we can neglect the covariance between them
#      this is also why we can bin the master and exposure over defined pixels (ie, ignoring some of the pixels that might be undefined within a bin)
#      otherwise the covariance matrix would need to be resampled over all consecutive pixels, included undefined ones, and the binned pixels would be set to undefined by the resampling function
def sub_def_bins(bin_siz,idx_kept_ord,low_pix,high_pix,dpix_loc,pix_loc,sp1D_loc,Mstar_loc=None,var1D_loc=None):
    raw_loc_dic = {    
        'low_bins': low_pix[idx_kept_ord],
        'high_bins': high_pix[idx_kept_ord],
        'dbins': dpix_loc[idx_kept_ord],
        'cen_bins': pix_loc[idx_kept_ord],
        'flux': sp1D_loc[idx_kept_ord]}
    if Mstar_loc is not None:raw_loc_dic['mast_flux'] = Mstar_loc[idx_kept_ord]  
    if var1D_loc is not None:raw_loc_dic['var'] = var1D_loc[idx_kept_ord]    

    #Defining bins at the requested resolution over the range of original defined bins
    min_pix = np.nanmin(raw_loc_dic['low_bins'])
    max_pix = np.nanmax(raw_loc_dic['high_bins'])
    n_bins_init=int(np.ceil((max_pix-min_pix)/bin_siz))
    bin_siz=(max_pix-min_pix)/n_bins_init
    bin_bd=np.append(min_pix+bin_siz*np.arange(n_bins_init,dtype=float),raw_loc_dic['high_bins'][-1])                             

    return bin_bd,raw_loc_dic


def sub_calc_bins(low_bin,high_bin,raw_loc_dic,nfilled_bins,calc_Fr=False,calc_gdet=False):
    bin_loc_dic = {}

    #Indexes of all original bins overlapping with current bin
    #    - searchsorted cannot be used in case bins are not continuous
    #    - the approach below remains faster than tiling a matrix with the original tables to apply in one go the search and sum operations
    idx_overpix = np_where1D( (raw_loc_dic['high_bins']>=low_bin) &  (raw_loc_dic['low_bins'] <=high_bin) )
    if len(idx_overpix)>0:
      
        #Total exposure flux over the selected bins
        Fexp_tot = np.sum(raw_loc_dic['flux'][idx_overpix]*raw_loc_dic['dbins'][idx_overpix] )
        if calc_Fr or (calc_gdet and (Fexp_tot>0.)):                             
            nfilled_bins+=1.
            
            #Ratio binned exposure flux / binned master flux
            if calc_Fr:
                bin_loc_dic['Fmast_tot'] = np.sum(raw_loc_dic['mast_flux'][idx_overpix]*raw_loc_dic['dbins'][idx_overpix] )            
                bin_loc_dic['Fr'] = Fexp_tot/bin_loc_dic['Fmast_tot']
                if 'var' in raw_loc_dic:bin_loc_dic['varFr'] = np.sum(raw_loc_dic['var'][idx_overpix]*raw_loc_dic['dbins'][idx_overpix]**2.)/bin_loc_dic['Fmast_tot']**2.

            #Ratio binned exposure error squared / binned exposure flux
            #    - see def_weights_spatiotemp_bin(): 
            # gdet(band,v) = sum( EF_meas(w,t,v)^2 )  ) / sum( F_meas(w,t,v) )
            if calc_gdet and (Fexp_tot>0.):
                bin_loc_dic['gdet'] = np.sum(raw_loc_dic['var'][idx_overpix]) /np.sum(raw_loc_dic['flux'][idx_overpix])

            #Adjust bin center and boundaries
            bin_loc_dic['cen_bins'] = np.mean(raw_loc_dic['cen_bins'][idx_overpix])
            bin_loc_dic['low_bins'] = raw_loc_dic['low_bins'][idx_overpix[0]]
            bin_loc_dic['high_bins'] = raw_loc_dic['high_bins'][idx_overpix[-1]]

    return bin_loc_dic,nfilled_bins









'''
Initialisation of current instrument tables and properties
'''
def init_data_instru(mock_dic,inst,gen_dic,data_dic,theo_dic,data_prop,coord_dic,system_param,plot_dic):
    if (gen_dic['type'][inst]=='CCF'):inst_data='CCFs'
    elif (gen_dic['type'][inst]=='spec1D'):inst_data='1D spectra'    
    elif (gen_dic['type'][inst]=='spec2D'):inst_data='2D echelle spectra'
    print('  Reading and initializing '+inst_data)    

    #Initialize dictionaries for current instrument
    for key_dic in [gen_dic,coord_dic,data_dic['DI'],data_dic,data_dic['Res'],data_dic['Intr'],data_dic['Atm'],data_prop,theo_dic]:key_dic[inst]={} 
    DI_data_inst = data_dic['DI'][inst]
    if (inst not in gen_dic['scr_lgth']):gen_dic['scr_lgth'][inst]={}

    #Facility and reduction id
    facil_inst = {
        'SOPHIE':'OHP',
        'HARPS':'ESO',
        'HARPN':'TNG',   
        'CARMENES_VIS':'CAHA',
        'CORALIE':'ESO','ESPRESSO':'ESO','ESPRESSO_MR':'ESO',
        'EXPRES':'DCT','NIRPS_HE':'ESO','NIRPS_HA':'ESO'
        }[inst]

    #Error definition
    if not gen_dic['mock_data']:
        if (not gen_dic['flag_err_inst'][inst]) and gen_dic['gcal']:stop('Error table must be available to estimate calibration')
        if (inst in gen_dic['force_flag_err']):gen_dic['flag_err_inst'][inst]=False
        if gen_dic['flag_err_inst'][inst]:print('   > Errors propagated from raw data')
        else:print('   > Beware: custom definition of errors')

    #Mask used to compute CCF on stellar lines
    if (gen_dic['CCF_from_sp']) and (inst in gen_dic['CCF_mask']):
        
        #Upload CCF mask
        ext = gen_dic['CCF_mask'][inst].split('.')[-1]
        if (ext=='fits'):
            hdulist = fits.open(gen_dic['CCF_mask'][inst])
            data_loc = hdulist[1].data
            gen_dic['CCF_mask_wav'][inst] = data_loc['lambda']
            gen_dic['CCF_mask_wgt'][inst] = data_loc['contrast']          
            hdulist.close()

        elif (ext=='csv'):
            data_loc = np.genfromtxt(gen_dic['CCF_mask'][inst], delimiter=',', names=True)
            gen_dic['CCF_mask_wav'][inst] = data_loc['wave']
            gen_dic['CCF_mask_wgt'][inst] = data_loc['contrast']  

        elif (ext in ['txt','dat']):
            data_loc = np.loadtxt(gen_dic['CCF_mask'][inst]).T            
            gen_dic['CCF_mask_wav'][inst] = data_loc[0]
            gen_dic['CCF_mask_wgt'][inst] = data_loc[1]       
        else:
            stop('CCF mask extension TBD') 
            
        #Same convention as the ESPRESSO DRS, which takes the 'contrast' column from mask files and use its squares as weights
        #    - masks weights should have been normalized so that mean( weights ) = 1, where weight = CCF_mask_wgt^2
        gen_dic['CCF_mask_wgt'][inst]=gen_dic['CCF_mask_wgt'][inst]**2.

    #Resampling exposures of a given visit on a common table or not
    #    - imposed for CCFs
    if gen_dic['type'][inst]=='CCF':gen_dic['comm_sp_tab'][inst]=True
    elif (inst not in gen_dic['comm_sp_tab']):gen_dic['comm_sp_tab'][inst]=False
    if (not gen_dic['comm_sp_tab'][inst]):print('   > Data processed on individual spectral tables for each exposure')      
    else:print('   > Data resampled on a common spectral table')



    ##############################################################################################################################
    #Retrieval and pre-processing of data
    ##############################################################################################################################    

    #Data is calculated and not retrieved
    if (gen_dic['calc_proc_data']):
        print('         Calculating data')

        #Initialize dictionaries for current instrument
        data_inst={'visit_list':[]}
        data_dic[inst]=data_inst  

        #Initialize current data type for the instrument
        #    - data need to be processed in their input mode at the start of each new instrument or visit
        #    - if mode is switched to CCFs/s1d at some point in the reduction of a visit, the visit type will be switched but the instrument type will remain the same, so that the next visits of the instrument are processed in their original mode
        #      only after all visits have been processed is the instrument type switched
        data_inst['type']=deepcopy(gen_dic['type'][inst])
        
        #Total number of order for current instrument
        #    - we define an artificial order that contains the CCF or 1D spectrum, so that the pipeline can process in the same way as with 2D spectra
        #    - if orders are fully removed from an instrument its default structure is updated
        #      if orders are trimmed after the spectral reduction the default structure is kept
        #    - mock data created with a single order
        if gen_dic['mock_data']: 
            data_inst['nord'] = 1
            gen_dic[inst]['norders_instru'] = 1
        else:
            data_inst['idx_ord_ref']=deepcopy(np.arange(gen_dic['norders_instru'][inst]))      #to keep track of the original orders
            idx_ord_kept = list(np.arange(gen_dic['norders_instru'][inst]))
            if (data_inst['type'] in ['spec1D','CCF']):
                data_inst['nord'] = 1
                gen_dic[inst]['wav_ord_inst'] = gen_dic['wav_ord_inst'][inst]
                gen_dic[inst]['norders_instru']=gen_dic['norders_instru'][inst]
            elif (data_inst['type']=='spec2D'):
                if inst in gen_dic['del_orders']:
                    idx_ord_kept = list(np.delete(np.arange(gen_dic['norders_instru'][inst]),gen_dic['del_orders'][inst]))
                    gen_dic[inst]['wav_ord_inst'] = gen_dic['wav_ord_inst'][inst][idx_ord_kept]
                    data_inst['idx_ord_ref'] = data_inst['idx_ord_ref'][idx_ord_kept]
                gen_dic[inst]['norders_instru'] = len(idx_ord_kept)
                data_inst['nord']=deepcopy(len(idx_ord_kept))   
        data_inst['nord_spec']=deepcopy(data_inst['nord'])                   #to keep track of the number of orders after spectra are trimmed
        data_inst['nord_ref']=deepcopy(gen_dic[inst]['norders_instru'])      #to keep track of the original number of orders
   
        #Orders contributing to calculation of spectral CCFs
        #    - set automatically in case of S1D to use the generic pipeline structure, even though S1D have no orders
        #    - indexes are made relative to order list after initial selection
        if data_inst['type']=='spec1D':gen_dic[inst]['orders4ccf']=[0]
        elif data_inst['type']=='spec2D':
            if (inst in gen_dic['orders4ccf']): gen_dic[inst]['orders4ccf'] = np.intersect1d(gen_dic['orders4ccf'][inst],idx_ord_kept,return_indices=True)[2]     
            else:gen_dic[inst]['orders4ccf'] = np.arange(gen_dic[inst]['norders_instru'],dtype=int)

        #Telluric spectrum condition
        if ('spec' in data_inst['type']) and gen_dic['tell_weight'] or gen_dic['corr_tell']:data_inst['tell_sp'] = True
        else:data_inst['tell_sp'] = False

        #Calibration condition
        #    - must be calculated even if not needed for weights, but can be deactivated if conversion into CCFs or 2D->1D
        if ('spec' in data_inst['type']):data_inst['mean_gdet'] = True
        else:data_inst['mean_gdet'] = False

        #Initialize flag that exposures in all visits of the instrument share a common spectral table
        data_inst['comm_sp_tab'] = True 
        
        #Calibration settings
        if (inst not in gen_dic['gcal_nooutedge']):gen_dic['gcal_nooutedge'][inst] = [0.,0.]
        
        #Processing each visit
        if gen_dic['mock_data']:vis_list=list(mock_dic['visit_def'][inst].keys())
        else:vis_list=list(gen_dic['data_dir_list'][inst].keys())
        ivis=-1
        for vis in vis_list:
            
            #Artificial data
            if gen_dic['mock_data']:
                
                #Generate time table
                if 'bin_low' in mock_dic['visit_def'][inst][vis]:
                    bjd_exp_low = np.array(mock_dic['visit_def'][inst][vis]['bin_low'])
                    bjd_exp_high = np.array(mock_dic['visit_def'][inst][vis]['bin_high'])
                    n_in_visit = len(bjd_exp_low)
                elif 'exp_range' in mock_dic['visit_def'][inst][vis]:
                    dbjd =  (mock_dic['visit_def'][inst][vis]['exp_range'][1]-mock_dic['visit_def'][inst][vis]['exp_range'][0])/mock_dic['visit_def'][inst][vis]['nexp']
                    n_in_visit = int((mock_dic['visit_def'][inst][vis]['exp_range'][1]-mock_dic['visit_def'][inst][vis]['exp_range'][0])/dbjd)
                    bjd_exp_low = mock_dic['visit_def'][inst][vis]['exp_range'][0] + dbjd*np.arange(n_in_visit)
                    bjd_exp_high = bjd_exp_low+dbjd      
                bjd_exp_all = 0.5*(bjd_exp_low+bjd_exp_high)        

            #Observational data        
            else:
                vis_path = gen_dic['data_dir_list'][inst][vis]
        
                #List of all exposures for current instrument
                if inst in ['SOPHIE']:
                    vis_path+= {
                        'CCF':'*ccf*',
                        'spec2D':'*e2ds_',
                        }[data_inst['type']]
                elif inst in ['CORALIE']:
                    vis_path+= {
                        'CCF':'*ccf*',
                        'spec2D':'*S2D_',
                        }[data_inst['type']]
                elif inst in ['CARMENES_VIS']:
                    vis_path+= {
                        'spec2D':'*sci-allr-vis_',
                        }[data_inst['type']]                    
                elif inst in ['ESPRESSO','ESPRESSO_MR','HARPN','HARPS']:
                    vis_path+= {
                        'CCF':'*CCF_',
                        'spec1D':'*S1D_',
                        'spec2D':'*S2D_',
                        }[data_inst['type']] 
                elif inst in ['NIRPS_HA','NIRPS_HE']:
                    vis_path+= {
                        'CCF':'*CCF_TELL_CORR_',       #CCF from telluric-corrected spectra
                        'spec1D':'*S1D_',
                        'spec2D':'*S2D_',
                        }[data_inst['type']] 
                elif inst in ['EXPRES']:
                    vis_path+= {
                        'spec2D':'*',
                        }[data_inst['type']]                      
                else:stop('Instrument undefined')
                
                #Use blazed data
                if (inst in gen_dic['blazed']) and (data_inst['type']=='spec2D'):
                    if (inst in ['ESPRESSO','ESPRESSO_MR','HARPN','HARPS','NIRPS_HA','NIRPS_HE']):vis_path+='BLAZE_'
                    else:stop('Define blaze file names')

                #Use sky-corrected data
                if (inst in gen_dic['fibB_corr']) and (vis in gen_dic['fibB_corr'][inst]) and (len(gen_dic['fibB_corr'][inst][vis])>0):
                    if inst in ['ESPRESSO','ESPRESSO_MR','HARPS','HARPN','NIRPS_HA','NIRPS_HE']:vis_path_skysub=vis_path+'SKYSUB_A'
                    else:vis_path_skysub=vis_path+'C'   
                    vis_path_skysub_exp = np.array(glob.glob(vis_path_skysub+'.fits'))
                    if len(vis_path_skysub_exp)==0:stop('No sky-sub data found. Check path.') 
                    
                    #Orders to be replaced
                    if gen_dic['fibB_corr'][inst][vis]=='all':idx_ord_skysub = 'all'
                    else:idx_ord_skysub = gen_dic['fibB_corr'][inst][vis]
                    
                else:vis_path_skysub_exp = None

                #Path of visits exposures
                if inst not in ['EXPRES']:vis_path+='A'
                vis_path_exp = np.array(glob.glob(vis_path+'.fits'))
                n_in_visit=len(vis_path_exp) 
                if n_in_visit==0:stop('No data found. Check path.')

                
            #Remove/keep visits
            if (vis in gen_dic['unused_visits'][inst]):   
                print('       > Visit '+vis+' is removed')
            else:
                print('         Initializing visit '+vis)
                data_dic[inst]['visit_list']+=[vis]     
                ivis+=1

                #Print visit date if available 
                #    - taking the day at the start of the night, rather than the day at the star of the exposure series
                if (not gen_dic['mock_data']):
                    if (inst in ['CORALIE','ESPRESSO','ESPRESSO_MR','HARPS','HARPN','CARMENES_VIS','EXPRES','NIRPS_HA','NIRPS_HE']):    
                        vis_day_exp=[]
                        vis_hour_exp=[]
                        bjd_exp = []
                        for file_path in vis_path_exp:
                            hdulist =fits.open(file_path)
                            hdr =hdulist[0].header 
                            if inst=='CORALIE':
                                vis_day_exp+= [int(hdr['HIERARCH ESO CORA SHUTTER START DATE'][6:8])]
                                bjd_exp  += [ hdr['HIERARCH ESO DRS BJD'] - 2400000. ]
                                stop('Define time for CORALIE')
                            elif inst in ['ESPRESSO','ESPRESSO_MR','HARPS','HARPN','CARMENES_VIS','NIRPS_HA','NIRPS_HE']:
                                vis_day_exp+= [int(hdr['DATE-OBS'].split('T')[0].split('-')[2]) ]  
                                vis_hour_exp+=[int(hdr['DATE-OBS'].split('T')[1].split(':')[0])]
                                if inst in ['ESPRESSO','ESPRESSO_MR','HARPS','HARPN','NIRPS_HA','NIRPS_HE']:bjd_exp +=[  hdr['HIERARCH '+facil_inst+' QC BJD'] - 2400000. ]
                                elif inst=='CARMENES_VIS':bjd_exp  += [  hdr['CARACAL BJD']  ]                        
                            elif inst=='EXPRES':
                                vis_day_exp+= [int(hdr['DATE-OBS'].split(' ')[0].split('-')[2]) ]  
                                vis_hour_exp+=[int(hdr['DATE-OBS'].split(' ')[1].split(':')[0])]
                        if inst=='CORALIE':
                            vis_yr = hdr['HIERARCH ESO CORA SHUTTER START DATE'][0:4]
                            vis_mt = hdr['HIERARCH ESO CORA SHUTTER START DATE'][4:6]
                        elif inst=='EXPRES':
                            vis_yr = hdr['DATE-OBS'].split(' ')[0].split('-')[0]
                            vis_mt = hdr['DATE-OBS'].split(' ')[0].split('-')[1]                            
                        else:
                            vis_yr = hdr['DATE-OBS'].split('T')[0].split('-')[0]
                            vis_mt = hdr['DATE-OBS'].split('T')[0].split('-')[1]
    
                        #Take the day before if all exposures are past midnight (ie, no exposures between 12 and 23) 
                        vis_day = np.min(vis_day_exp)
                        vis_hour_exp = np.array(vis_hour_exp)
                        bjd_vis = np.mean(bjd_exp)
                        if np.sum((vis_hour_exp>=12) & (vis_hour_exp<=23)  )==0:
                            if vis_day==1:stop('Adapt date retrieval')
                            vis_day-=1
                        vis_day_txt = '0'+str(vis_day) if vis_day<10 else str(vis_day)                    
                        print('           Date :',vis_yr,'/',vis_mt,'/',vis_day_txt)
                
                else:
                    bjd_vis = np.mean(bjd_exp_all) - 2400000.                   
                        
                #Initializing dictionaries for visit
                theo_dic[inst][vis]={}
                data_dic['Atm'][inst][vis]={}   
                data_inst[vis] = {'n_in_visit':n_in_visit,'transit_pl':[],'comm_sp_tab':True} 
                coord_dic[inst][vis] = {}
                for pl_loc in gen_dic['studied_pl']:
                    if (inst in gen_dic['transit_pl'][pl_loc]) and (vis in gen_dic['transit_pl'][pl_loc][inst]):data_inst[vis]['transit_pl']+=[pl_loc]
                data_prop[inst][vis] = {}
                data_dic_temp={}
                gen_dic[inst][vis] = {}
                DI_data_inst[vis] = {}  
                         
                #Associating telluric spectrum with each exposure
                if not gen_dic['mock_data']:data_inst[vis]['tell_DI_data_paths']={}
                
                #Initialize current data type and conditions for the visit
                for key in ['type','tell_sp','mean_gdet']:data_inst[vis][key]=deepcopy(data_inst[key])                
                
                #Initialize exposure tables
                #    - spatial positions in x/y in front of the stellar disk
                # + x position along the node line
                #  y position along the projection of the normal to the orbital plane
                # +calculated also for the out-of-transit positions, as in case of exposures binning some of the 
                #      ingress/egress exposures may contribute to the binned exposures average positions            
                #    - phase of in/out exposures, per instrument and per visit   
                #    - radial velocity of planet in star rest frame  
                for key in ['bjd','t_dur','RV_star_solCDM','RV_star_stelCDM']:coord_dic[inst][vis][key] = np.zeros(n_in_visit,dtype=float)*np.nan
                for pl_loc in list(set(gen_dic['studied_pl']+gen_dic['kepl_pl'])):
                    coord_dic[inst][vis][pl_loc]={} 
                    if pl_loc in data_inst[vis]['transit_pl']:
                        for key in ['ecl','cen_ph','st_ph','end_ph','ph_dur','rv_pl']:coord_dic[inst][vis][pl_loc][key] = np.zeros(n_in_visit,dtype=float)*np.nan
                        for key in ['cen_pos','st_pos','end_pos']:coord_dic[inst][vis][pl_loc][key] = np.zeros([3,n_in_visit],dtype=float)*np.nan
    
                    #Definition of mid-transit times for each planet associated with the visit 
                    if (pl_loc in gen_dic['Tcenter_visits']) and (inst in gen_dic['Tcenter_visits'][pl_loc]) and (vis in gen_dic['Tcenter_visits'][pl_loc][inst]):
                        coord_dic[inst][vis][pl_loc]['Tcenter'] = gen_dic['Tcenter_visits'][pl_loc][inst][vis]
                    else:
                        norb = round((bjd_vis+2400000.-system_param[pl_loc]['TCenter'])/system_param[pl_loc]["period"])
                        coord_dic[inst][vis][pl_loc]['Tcenter']  = system_param[pl_loc]['TCenter']+norb*system_param[pl_loc]["period"]

                #Observation properties
                for key in ['AM','IWV_AM','TEMP','PRESS','seeing','colcorrmin','colcorrmax','mean_SNR','alt','az','BERV']:data_prop[inst][vis][key] = np.zeros(n_in_visit,dtype=float)*np.nan        
                if inst=='ESPRESSO_MR':
                    for key in ['AM_UT','seeing_UT']:data_prop[inst][vis][key] = np.zeros(n_in_visit,dtype=object)*np.nan 
                data_prop[inst][vis]['PSF_prop'] = np.zeros([n_in_visit,4],dtype=float)
                data_prop[inst][vis]['SNRs']=np.zeros([n_in_visit,data_inst['nord_ref']],dtype=float)*np.nan 
                if inst in ['ESPRESSO','HARPN','HARPS','NIRPS_HA','NIRPS_HE']:
                    data_prop[inst][vis]['colcorr_ord']=np.zeros([n_in_visit,data_inst['nord_ref']],dtype=float)*np.nan
                    for key in ['BLAZE_A','BLAZE_B','WAVE_MATRIX_THAR_FP_A','WAVE_MATRIX_THAR_FP_B']:data_prop[inst][vis][key]=np.empty([n_in_visit],dtype='U35')
                    data_prop[inst][vis]['satur_check']=np.zeros([n_in_visit],dtype=int)*np.nan
                    data_prop[inst][vis]['adc_prop']=np.zeros([n_in_visit,6],dtype=float)*np.nan
                    data_prop[inst][vis]['piezo_prop']=np.zeros([n_in_visit,4],dtype=float)*np.nan
                if inst=='ESPRESSO':
                    data_prop[inst][vis]['guid_coord']=np.zeros([n_in_visit,2],dtype=float)*np.nan
                    
                #Set error flag per visit
                #    - same for a given instrument, but this allows dealing with binned visits
                if gen_dic['mock_data']:gen_dic[inst][vis]['flag_err']=False
                else:gen_dic[inst][vis]['flag_err']=deepcopy(gen_dic['flag_err_inst'][inst])     
        
                #Raw CCF properties
                if inst in ['HARPN','HARPS','CORALIE','SOPHIE','ESPRESSO','ESPRESSO_MR','NIRPS_HA','NIRPS_HE']:
                    for key in ['RVdrift','rv_pip','FWHM_pip','ctrst_pip']:DI_data_inst[vis][key] =  np.zeros(n_in_visit,dtype=float)*np.nan  
                    for key in ['erv_pip','eFWHM_pip','ectrst_pip']:DI_data_inst[vis][key] =  np.zeros(n_in_visit,dtype=float)  

                #Activity indexes
                data_inst[vis]['act_idx'] = []
                if inst in ['HARPN','HARPS','CORALIE','SOPHIE','ESPRESSO','ESPRESSO_MR','NIRPS_HA','NIRPS_HE'] and gen_dic['DACE_sp']:
                    DACE_sp = True

                    #Corresponding DACE star name, instrument and visit
                    dace_name =  hdr['HIERARCH '+facil_inst+' OBS TARG NAME']
                    DACE_sp = Spectroscopy.get_timeseries(dace_name, sorted_by_instrument=False)

                    #Condition to retrieve current visit
                    dace_red_inst = hdr['INSTRUME']
                    if dace_red_inst=='ESPRESSO': 
                        if bjd_vis<=58649.5:dace_red_inst+='18'
                        else:dace_red_inst+='19'  
                    elif dace_red_inst=='HARPS': 
                        if bjd_vis<=57167.5:dace_red_inst+='03'
                        else:dace_red_inst+='15'  
                    cond_dace_vis = (np.asarray(DACE_sp['ins_name']) == dace_red_inst) & (np.asarray(DACE_sp['date_night']) == vis_yr+'-'+vis_mt+'-'+vis_day_txt)
                    if True not in cond_dace_vis:
                        print('         Visit not found in DACE')                    
                    else:

                        #Retrieve activity indexes and bjd
                        DACE_idx = {}
                        DACE_idx['bjd'] =  np.array(DACE_sp['rjd'],dtype=float)[cond_dace_vis]
                        data_inst[vis]['act_idx'] = ['ha','na','ca','s','rhk']
                        for key in data_inst[vis]['act_idx']:
                            data_prop[inst][vis][key] = np.zeros([n_in_visit,2],dtype=float)*np.nan
                            if key=='rhk':dace_key = 'rhk'
                            else:dace_key = key+'index'
                            DACE_idx[key] = np.array(DACE_sp[dace_key],dtype=float)[cond_dace_vis]
                            DACE_idx[key+'_err'] = np.array(DACE_sp[dace_key+'_err'],dtype=float)[cond_dace_vis]                    
                else:
                    DACE_sp = False
                    if inst=='EXPRES': 
                        data_inst[vis]['act_idx'] = ['ha','s']
                        for key in data_inst[vis]['act_idx']:data_prop[inst][vis][key] = np.zeros([n_in_visit,2],dtype=float)*np.nan
                    
                #------------------------------------------------------------------------------------------------------------     
   
                #Process all exposures in visit
                for isub_exp,iexp in enumerate(range(n_in_visit)):

                    #Artificial data            
                    if gen_dic['mock_data']:                
                        coord_dic[inst][vis]['bjd'][iexp] = bjd_exp_all[iexp]  - 2400000.  
                        coord_dic[inst][vis]['t_dur'][iexp] = (bjd_exp_high[iexp]-bjd_exp_low[iexp])*24.*3600.
                
                    #Observational data
                    else:
                    
                        #Data of the fits file 
                        hdulist =fits.open(vis_path_exp[iexp])
                        if vis_path_skysub_exp is not None:hdulist_skysub =fits.open(vis_path_skysub_exp[iexp])
                
                        #Header 
                        hdr =hdulist[0].header 

                        #Retrieve exposure bjd in instrument/visit table
                        if inst in ['HARPS','HARPN','ESPRESSO','ESPRESSO_MR','NIRPS_HA','NIRPS_HE']:
                            bjd_exp =  hdr['HIERARCH '+facil_inst+' QC BJD'] - 2400000.      
                        elif inst in ['CORALIE']:bjd_exp =  hdr['HIERARCH ESO DRS BJD'] - 2400000.  
                        elif inst in ['SOPHIE']:bjd_exp =  hdr['HIERARCH OHP DRS BJD'] - 2400000.  
                        elif inst in ['CARMENES_VIS']:bjd_exp =  hdr['CARACAL BJD']   
                        elif inst=='EXPRES':bjd_exp = hdulist[1].header['BARYMJD'] +0.5
                        else:stop('Undefined')
                        coord_dic[inst][vis]['bjd'][iexp] = bjd_exp
                      
                        #Exposure time (s)    
                        if inst=='SOPHIE':             
                            
                            #start time
                            st_hmns=(hdr['HIERARCH OHP OBS DATE START'].split('T')[1]).split(':')
                            h_start=float(st_hmns[0])            
                            mn_start=float(st_hmns[1]) 
                            s_start=float(st_hmns[2]) 
                            
                            #end time
                            end_hmns=(hdr['HIERARCH OHP OBS DATE END'].split('T')[1]).split(':')  
                            h_end=float(end_hmns[0])
                            if h_end<h_start:h_end+=24.
                            mn_end=float(end_hmns[1]) 
                            s_end=float(end_hmns[2])
                    
                            #exposure time (s)
                            coord_dic[inst][vis]['t_dur'][iexp]=( (h_end-h_start)*3600. + (mn_end-mn_start)*60. + (s_end-s_start) )

                        elif inst=='EXPRES':coord_dic[inst][vis]['t_dur'][iexp] = hdr['AEXPTIME']  
                        else:coord_dic[inst][vis]['t_dur'][iexp] = hdr['EXPTIME']  
                
                        #RV drift  
                        #    - drift is measured from the calibration lamps or FP 
                        #    - drift is corrected automatically in new reductions for ESPRESSO and kin spectrographs, direcly included in the wavelength solution
                        if inst in ['CORALIE']:
                            DI_data_inst[vis]['RVdrift'][iexp]=hdr['HIERARCH ESO DRS DRIFT SPE RV']  #in m/s  
                        elif inst=='SOPHIE':
                            DI_data_inst[vis]['RVdrift'][iexp]=hdr['HIERARCH OHP DRS DRIFT RV']      #in m/s                 
                        elif inst in ['HARPS','HARPN','ESPRESSO','ESPRESSO_MR','NIRPS_HA','NIRPS_HE']:
                            if 'HIERARCH '+facil_inst+' QC DRIFT DET0 MEAN' in hdr:
                                DI_data_inst[vis]['RVdrift'][iexp]=hdr['HIERARCH '+facil_inst+' QC DRIFT DET0 MEAN']*gen_dic['pix_size_v'][inst]*1e3    #in pix -> km/s
                    
                    #Orbital coordinates for each studied planet
                    for pl_loc in data_inst[vis]['transit_pl']:
                        coord_dic[inst][vis][pl_loc]['cen_pos'][:,iexp],coord_dic[inst][vis][pl_loc]['st_pos'][:,iexp],coord_dic[inst][vis][pl_loc]['end_pos'][:,iexp],coord_dic[inst][vis][pl_loc]['ecl'][iexp],coord_dic[inst][vis][pl_loc]['rv_pl'][iexp],\
                        coord_dic[inst][vis][pl_loc]['st_ph'][iexp],coord_dic[inst][vis][pl_loc]['cen_ph'][iexp],coord_dic[inst][vis][pl_loc]['end_ph'][iexp],coord_dic[inst][vis][pl_loc]['ph_dur'][iexp]=coord_expos(pl_loc,coord_dic,inst,vis,system_param['star'],
                                            system_param[pl_loc],coord_dic[inst][vis]['bjd'][iexp],coord_dic[inst][vis]['t_dur'][iexp],data_dic,data_dic['DI']['system_prop']['achrom'][pl_loc][0])                    
                    
                    #--------------------------------------------------------------------------------------------------
        
                    #Initialize data at first exposure
                    if isub_exp==0:
 
                        #Artificial data 
                        if gen_dic['mock_data']: 
                            fixed_args = {}
                            fixed_args.update(mock_dic['intr_prof'][inst])
                            data_inst[vis]['mock'] = True

                            #Activation of spectral conversion and resampling 
                            cond_conv_st_prof_tab(theo_dic['rv_osamp_line_mod'],fixed_args,data_inst[vis]['type'])
                                                         
                            #Mock data spectral table
                            #    - fixed for all exposures, defined in the star rest frame
                            #    - the model table is defined in the star rest frame, so that models calculated / defined on this table remain centered independently of the systemic and keplerian RV
                            #      models for each exposure are then resampled on the original spectral table if required, and the table is only then shifted / scaled by the systemic and keplerian RV
                            #    - theoretical models are defined in wavelength space
                            #      analytical models are defined in RV space, and their table converted into spectral space afterwards if relevant
                            data_inst[vis]['nspec'] = int(np.ceil((mock_dic['DI_table']['x_end']-mock_dic['DI_table']['x_start'])/mock_dic['DI_table']['dx']))
                            delta_cen_bins = mock_dic['DI_table']['x_end'] - mock_dic['DI_table']['x_start']
                            dcen_bins = delta_cen_bins/data_inst[vis]['nspec'] 
                            fixed_args['cen_bins']=mock_dic['DI_table']['x_start']+dcen_bins*(0.5+np.arange(data_inst[vis]['nspec']))
                            fixed_args['edge_bins']=def_edge_tab(fixed_args['cen_bins'][None,:][None,:])[0,0]
                            fixed_args['ncen_bins']=data_inst[vis]['nspec']
                            fixed_args['dcen_bins']=fixed_args['edge_bins'][1::] - fixed_args['edge_bins'][0:-1]   
                            fixed_args['dim_exp']=[data_inst['nord'],fixed_args['ncen_bins']]                           
                            
                            #Resampled spectral table for model line profile
                            if fixed_args['resamp']:resamp_model_st_prof_tab(None,None,None,fixed_args,gen_dic,1,theo_dic['rv_osamp_line_mod'])

                            #Effective instrumental convolution
                            fixed_args['FWHM_inst'] = ref_inst_convol(inst,fixed_args,fixed_args['cen_bins'])

                            #Initialize intrinsic profile properties   
                            params_mock = deepcopy(system_param['star']) 
                            if inst not in mock_dic['flux_cont']:mock_dic['flux_cont'][inst]={}
                            if inst not in mock_dic['flux_cont'][inst]:mock_dic['flux_cont'][inst][vis] = 1.
                            params_mock.update({'rv':0.,'cont':mock_dic['flux_cont'][inst][vis]})  
                            params_mock = par_formatting(params_mock,fixed_args['mod_prop'],None,None,fixed_args,inst,vis) 
             
                            #Generic properties required for model calculation
                            if inst not in mock_dic['sysvel']:mock_dic['sysvel'][inst]={}
                            if inst not in mock_dic['sysvel'][inst]:mock_dic['sysvel'][inst][vis] = 0.
                            fixed_args.update({ 
                                'mac_mode':theo_dic['mac_mode'],
                                'type':data_inst[vis]['type'],  
                                'nord':data_inst['nord'],
                                'nthreads':mock_dic['nthreads'], 
                                'resamp_mode' : gen_dic['resamp_mode'], 
                                'conv2intr':False,
                                'inst':inst,
                                'vis':vis, 
                                'fit':False,
                                })

           
                            # # Spots properties 
                            # if fixed_args['use_spots']:
                            #     fixed_args['t_exp_bjd'] = {inst : {vis : coord_dic[inst][vis]['bjd'] }}
                            #     fixed_args['print_exp'] = True
                            # for pl_loc in data_inst[vis]['transit_pl']:
                            #     fixed_args['phase'] = {inst:{vis:[coord_dic[inst][vis][pl_loc][m] for m in ['st_ph','cen_ph','end_ph']] }}
                            #     par_formatting(params_mock,mock_dic['spots_prop'][inst][vis],None,None,fixed_args,inst,vis) 

                        #Observational data            
                        else:   
                            data_inst[vis]['mock'] = False                      
    
                            #VLT UT used by ESPRESSO
                            if inst=='ESPRESSO':
                                tel_inst = {
                                    'ESO-VLT-U1':'1',
                                    'ESO-VLT-U2':'2',
                                    'ESO-VLT-U3':'3',
                                    'ESO-VLT-U4':'4'
                                    }[hdr['TELESCOP']]                                

                            #Instrumental mode
                            if inst in ['ESPRESSO','NIRPS_HA','NIRPS_HE']: 
                                data_prop[inst][vis]['ins_mod'] = hdr['HIERARCH ESO INS MODE']    
                                data_prop[inst][vis]['det_binx'] = hdr['HIERARCH ESO DET BINX']
                                
                            #Radial velocity tables for input CCFs
                            #    - assumed to be common for all CCFs of the visit dataset, and thus calculated only once
                            if (data_inst['type']=='CCF'):
                                    
                                if inst in ['HARPN','HARPS','ESPRESSO','ESPRESSO_MR','NIRPS_HA','NIRPS_HE']:
                                    start_rv =  hdr['HIERARCH '+facil_inst+' RV START']    # first velocity
                                    delta_rv = hdr['HIERARCH '+facil_inst+' RV STEP']     # delta vel step
                                    n_vel_vis = (hdulist[1].header)['NAXIS1']                                              
                                else:                        
                                    start_rv =  hdr['CRVAL1']    # first wavelength/velocity
                                    delta_rv = hdr['CDELT1']     # delta lambda/vel step
                                    n_vel_vis = hdr['NAXIS1']
                                    
                                #Velocity table and resolution
                                velccf = start_rv+delta_rv*np.arange(n_vel_vis)  

                                #Screening
                                #    - CCFs are screened (ie, keeping one point every specified length) with a length defined as input, or the instrumental bin length 
                                #      the goal is to keep only uncorrelated points, on which we can measure the dispersion as a white noise error 
                                #    - we remove scr_lgth-1 points between two kept points, ie we keep one point in scr_lgth+1
                                #      if the original table is x0,x1,x2,x3,x4.. and scr_lgth=3 we keep x0,x3.. 
                                #    - we define in 'scr_lgth' the number of CCF bins to be removed depending on the chosen length, and the CCF resolution 
                                #    - we assume the same value can be used for a given visit:
                                # + if the bin size is chosen as typical length, then it will not depend on the visits but the CCFs may have been computed with a 
                                # different resolution for each visit. In that case the number of screened bins is calculated once for the visit as:
                                #      scr_lgth = bin size/CCF_step
                                #   with CCF typically oversampled, CCF_step is lower than the bin size and we keep about one CCF point every bin size   
                                # + if the correlation length is chosen, then it may depend on the visit conditions but should be constant for a given visit. In that case 
                                # the number of screened bins has already been set in 'scr_lgth' for the visit, using the empirical estimation on the residuals CCFs            
                                if vis not in gen_dic['scr_lgth'][inst]:
                                    gen_dic[inst][vis]['scr_lgth']=int(round(gen_dic['pix_size_v'][inst]/delta_rv))
                                    if gen_dic[inst][vis]['scr_lgth']<=1:gen_dic[inst][vis]['scr_lgth']=1  
                                    if gen_dic[inst][vis]['scr_lgth']==1:print('           No CCF screening required')
                                    else:print('           Screening by '+str(gen_dic[inst][vis]['scr_lgth'])+' pixels')
                                if gen_dic[inst][vis]['scr_lgth']>1:
                                    idx_scr_bins=np.arange(n_vel_vis,dtype=int)[gen_dic['ist_scr']::gen_dic[inst][vis]['scr_lgth']]
                                    velccf = velccf[idx_scr_bins]
            
                                #Table dimensions                             
                                data_inst[vis]['nspec'] = len(velccf) 
    
                            #----------------------------------------
                            
                            #Number of wavelength bins for spectra 
                            elif ('spec' in data_inst['type']):
                
                                #1D spectra
                                if (data_inst['type']=='spec1D'):
                                    if inst in ['ESPRESSO','ESPRESSO_MR','HARPN']:data_inst[vis]['nspec']  = (hdulist[1].header)['NAXIS2']
                                    else:stop('TBD') 
            
                                #2D spectra
                                elif (data_inst['type']=='spec2D'):
                                    if inst in ['ESPRESSO','ESPRESSO_MR','HARPN','HARPS','CARMENES_VIS','NIRPS_HA','NIRPS_HE']:data_inst[vis]['nspec']  = (hdulist[1].header)['NAXIS1']
                                    elif inst=='EXPRES':data_inst[vis]['nspec'] = 7920
                                    else:stop('TBD')

                        #Table dimensions
                        data_inst[vis]['dim_all'] = [n_in_visit,data_inst['nord'],data_inst[vis]['nspec']]
                        data_inst[vis]['dim_exp'] = [data_inst['nord'],data_inst[vis]['nspec']]
                        data_inst[vis]['dim_sp'] = [n_in_visit,data_inst['nord']]
                        data_inst[vis]['dim_ord'] = [n_in_visit,data_inst[vis]['nspec']]

                        #Initialize spectral table, flux table, banded covariance matrix                   
                        data_dic_temp['cen_bins'] = np.zeros(data_inst[vis]['dim_all'], dtype=float)
                        data_dic_temp['flux'] = np.zeros(data_inst[vis]['dim_all'], dtype=float)*np.nan
                        data_dic_temp['cov'] = np.zeros(data_inst[vis]['dim_sp'], dtype=object)

                        #Telluric spectrum
                        if data_inst[vis]['tell_sp'] and (gen_dic['calc_tell_mode']=='input'):data_dic_temp['tell'] = np.ones(data_inst[vis]['dim_all'], dtype=float)

                    ### End of first exposure            
        
                    #------------------------------------------------------------------------------------
                    #Retrieve data for current exposure
                    #------------------------------------------------------------------------------------

                    #-----------------------------------                      
                    #Artificial data  
                    #-----------------------------------
                    if gen_dic['mock_data']: 
                        print('            building exposure '+str(iexp)+'/'+str(n_in_visit - 1))
                        param_exp = deepcopy(params_mock) 

                        #Table for model calculation
                        args_exp = def_st_prof_tab(None,None,None,fixed_args)

                        #Initializing stellar profiles
                        args_exp = init_custom_DI_prof(args_exp,gen_dic,data_dic['DI']['system_prop'],theo_dic,system_param['star'],param_exp)
                        
                        #Initializing broadband scaling of intrinsic profiles into local profiles
                        #    - defined in forward mode at initialization, or defined in fit mode only if the stellar grid is not updated through the fit
                        #    - there are no default pipeline tables for this scaling because they depend on the local spectral tables of the line profiles
                        theo_intr2loc(args_exp['grid_dic'],args_exp['system_prop'],args_exp,args_exp['ncen_bins'],theo_dic['nsub_star'])                        
                        
                        #Add jitter to the intrinsic profile properties (simulating stellar activity)
                        if (fixed_args['mode']=='ana') and (inst in mock_dic['drift_intr']) and (vis in mock_dic['drift_intr'][vis]) and (len(mock_dic['drift_intr'][inst][vis]>0)):
                            for par_drift in mock_dic['drift_intr'][inst][vis] : 
                                if par_drift in param_exp:
                                    if (par_drift=='rv'):param_exp[par_drift] += mock_dic['drift_intr'][inst][vis][par_drift][iexp]
                                    else:param_exp[par_drift] *= mock_dic['drift_intr'][inst][vis][par_drift][iexp]

                        #Disk-integrated stellar line     
                        base_DI_prof = custom_DI_prof(param_exp,None,args=args_exp)[0]

                        #Deviation from nominal stellar profile
                        surf_prop_dic = sub_calc_plocc_prop([data_dic['DI']['system_prop']['chrom_mode']],args_exp,['line_prof'],data_dic[inst][vis]['transit_pl'],deepcopy(system_param),theo_dic,args_exp['system_prop'],param_exp,coord_dic[inst][vis],[iexp],False)
                        
                        #Correcting the disk-integrated profile for planet and spot contributions
                        DI_prof_exp = base_DI_prof - surf_prop_dic[data_dic['DI']['system_prop']['chrom_mode']]['line_prof'][:,0]
                       
                        #Instrumental response 
                        #    - in RV space for analytical model, in wavelength space for theoretical profiles
                        #    - resolution can be modified to model systematic variations from the instrument or atmosphere
                        #    - disabled if measured profiles as used as proxy for the intrinsic profiles
                        if (fixed_args['mode']!='Intrbin') and (inst in mock_dic['drift_post']) and (vis in mock_dic['drift_post'][vis]) and ('resol' in mock_dic['drift_post'][inst][vis]):
                            fixed_args['FWHM_inst'] = fixed_args['ref_conv']/mock_dic['drift_post'][inst][vis]['resol'][iexp]

                        #Convolution, conversion and resampling 
                        DI_prof_exp = conv_st_prof_tab(None,None,None,fixed_args,args_exp,DI_prof_exp,fixed_args['FWHM_inst'])

                        #Define number of photoelectrons extracted during the exposure
                        #   - the model is a density of photoelectrons per unit of time, with continuum set to the input mean flux density
                        if (inst in mock_dic['gcal']):mock_gcal = mock_dic['gcal'][inst]
                        else:mock_gcal = 1.
                        DI_prof_exp_Ftrue = mock_gcal*DI_prof_exp*coord_dic[inst][vis]['t_dur'][iexp]   

                        #Keplerian motion and systemic shift of the disk-integrated profile 
                        #    - including systematic variations if requested
                        kepl_rv = calc_orb_motion(coord_dic,inst,vis,system_param,gen_dic, coord_dic[inst][vis]['bjd'][iexp],coord_dic[inst][vis]['t_dur'][iexp],mock_dic['sysvel'][inst][vis])[1]
                        rv_mock = mock_dic['sysvel'][inst][vis] + kepl_rv
                        if (inst in mock_dic['drift_post']) and (vis in mock_dic['drift_post'][vis]) and ('rv' in mock_dic['drift_post'][inst][vis]):rv_mock+=mock_dic['drift_post'][inst][vis]['rv'][iexp]
                        if ('spec' in data_inst[vis]['type']):data_dic_temp['cen_bins'][iexp,0]=fixed_args['cen_bins']*spec_dopshift(-rv_mock)
                        else:data_dic_temp['cen_bins'][iexp,0] = fixed_args['cen_bins'] + rv_mock        
                   
                        #Defining flux and error table     
                        #    - see def_weights_bin(), the measured (total, not density) flux can be defined as:
                        # F_meas(t,w) = gcal(band) Nmeas(t,w)
                        #      where Nmeas(t,w), drawn from a Poisson distribution with number of events Ntrue(t,w), is the number of photo-electrons measured durint texp
                        #      the estimate of the error is
                        # EF_meas(t,w) = sqrt(gcal(band) F_meas(t,w))
                        #      which is a biased estimate of the true error but corresponds to what is returned by DRS
                        #    - the S/N of the mock profiles is F_meas(t,w)/EF_meas(t,w) = sqrt(F_meas(t,w)/gcal(band)) = sqrt(Nmeas(t,w)) proportional to sqrt(texp*cont)
                        #      errors in the continuum of the normalized profiles are proportional to 1/sqrt(texp*cont) 
                        #      beware that this error does not necessarily match the flux dispersion
                        if (inst in mock_dic['set_err']) and (vis in mock_dic['set_err'][inst]) and mock_dic['set_err'][inst][vis]:
                            DI_prof_exp_Fmeas = np.array(list(map(np.random.poisson, DI_prof_exp_Ftrue,  data_inst[vis]['nspec']*[1]))).flatten()
                            DI_err_exp_Emeas = np.sqrt(mock_gcal*DI_prof_exp_Fmeas)
                        else:
                            DI_prof_exp_Fmeas = DI_prof_exp_Ftrue
                            DI_err_exp_Emeas = np.zeros(data_inst[vis]['nspec'],dtype=float)
                        data_dic_temp['flux'][iexp,0] = DI_prof_exp_Fmeas                      
                        data_dic_temp['cov'][iexp,0] = (DI_err_exp_Emeas**2.)[None,:]                        
          
                    #-----------------------------------
                    #Observational data
                    #-----------------------------------
                    else: 
                        cond_bad_exp = np.zeros(data_inst[vis]['dim_exp'],dtype=bool)
                    
                        #Retrieve BERV
                        #    - in km/s
                        if inst=='EXPRES':
                            data_prop[inst][vis]['BERV'][iexp] = hdulist[2].header['HIERARCH wtd_mdpt_bc']*c_light
                        else:
                            if inst=='CARMENES_VIS': reduc_txt = 'CARACAL' 
                            elif inst in ['HARPN','HARPS','ESPRESSO','ESPRESSO_MR','NIRPS_HA','NIRPS_HE']:reduc_txt = facil_inst+' QC'
                            else:stop('Define BERV retrieval')
                            data_prop[inst][vis]['BERV'][iexp] = hdr['HIERARCH '+reduc_txt+' BERV']

                        #Retrieve CCFs
                        #    - HARPS 'hdu' has 73 x n_velocity size
                        # + 72 CCFs for each of the spectrograph order
                        # + last element is the total CCF over all orders (n_velocity size) 
                        #    - HARPN 'hdu' has 70 x n_velocity size  
                        # + 69 CCFs for each of the spectrograph order
                        # + last element is the total CCF over all orders (n_velocity size)   
                        #    - SOPHIE 'hdu' has 40 x n_velocity size
                        # + 39 CCFs for each of the spectrograph order
                        # + last element is the total CCF over all orders (n_velocity size)                           
                        if data_inst['type']=='CCF': 
            
                            #Bin centers
                            #    - we associate the velocity table to all exposures to keep the same structure as spectra
                            data_dic_temp['cen_bins'][iexp,0] = velccf
         
                            #CCF tables per order 
                            if (vis_path_skysub_exp is not None) and (gen_dic['fibB_corr'][inst][vis]=='all'):hdulist_dat= hdulist_skysub
                            else:hdulist_dat = hdulist
                            if inst in ['HARPS','ESPRESSO','ESPRESSO_MR','HARPN','NIRPS_HA','NIRPS_HE']:
                                all_ccf  = hdulist_dat[1].data  
                                all_eccf = hdulist_dat[2].data 
                                if (vis_path_skysub_exp is not None) and (gen_dic['fibB_corr'][inst][vis]!='all'):
                                    all_ccf[idx_ord_skysub] = (hdulist_skysub[1].data)[idx_ord_skysub]
                                    all_eccf[idx_ord_skysub] = (hdulist_skysub[2].data)[idx_ord_skysub]
                            else:                      
                                all_ccf =hdulist_dat[0].data
                                if (vis_path_skysub_exp is not None) and (gen_dic['fibB_corr'][inst][vis]!='all'):
                                    all_ccf[idx_ord_skysub] = (hdulist_skysub[0].data)[idx_ord_skysub]           
                          
                            #Construction of the total CCF
                            #    - we re-calculate the CCF rather than taking the total CCF stored in the last column of the matrix to be able to account for sky-correction in specific orders
                            #    - we use the input "gen_dic['orders4ccf'][inst]" rather than "gen_dic[inst]['orders4ccf']", which is used for CCF computed from spectral data 
                            if inst not in gen_dic['orders4ccf']:flux_raw = np.sum(all_ccf[ range(gen_dic['norders_instru'][inst])],axis=0)     
                            else:flux_raw = np.sum(all_ccf[gen_dic['orders4ccf'][inst]],axis=0)             

                            #Error table
                            if gen_dic[inst][vis]['flag_err']:
                                if inst not in gen_dic['orders4ccf']:err_raw = np.sqrt(np.sum(all_eccf[range(gen_dic['norders_instru'][inst])]**2.,axis=0))      
                                else:err_raw = np.sqrt(np.sum(all_eccf[gen_dic['orders4ccf'][inst]]**2.,axis=0))                              

                            #Screening CCF
                            if gen_dic[inst][vis]['scr_lgth']>1:
                                flux_raw = flux_raw[idx_scr_bins]
                                err_raw = err_raw[idx_scr_bins]
                            err_raw = np.tile(err_raw,[data_inst['nord'],1])
                            data_dic_temp['flux'][iexp,0] = flux_raw

                        #-------------------------------------------------------------------------------------------------------------------------------------------
                                
                        #Retrieve spectral data
                        elif ('spec' in data_inst['type']):
                                              
                            #1D spectra
                            if data_inst['type']=='spec1D':
                                if inst in ['HARPS','ESPRESSO','ESPRESSO_MR','HARPN','NIRPS_HA','NIRPS_HE']:
                                    
                                    #Replacing single order with sky-corrected data
                                    if (vis_path_skysub_exp is not None):data_loc = hdulist_skysub[1].data         
                                    else:data_loc = hdulist[1].data                                    

                                    #Bin centers
                                    if gen_dic['sp_frame']=='air':data_dic_temp['cen_bins'][iexp] = data_loc['wavelength_air'] 
                                    elif gen_dic['sp_frame']=='vacuum':data_dic_temp['cen_bins'][iexp] = data_loc['wavelength']                             
                                    
                                    #Spectra
                                    #    - flux and errors are per unit of wavelength but not per unit of time, ie that they correspond to the number of photoelectrons measured during the exposure
                                    data_dic_temp['flux'][iexp] = data_loc['flux']
                                    err_raw = data_loc['error']

                                    #Pixels quality
                                    qualdata_exp = data_loc['quality']    

                                else:
                                    stop('Spectra upload TBD for this instrument') 
        
                            #------------------------------
            
                            #2D spectra
                            elif data_inst['type']=='spec2D':
                                    
                                #Sky-corrected data
                                if (vis_path_skysub_exp is not None):
                                    if (gen_dic['fibB_corr'][inst][vis]=='all'):hdulist_dat= hdulist_skysub
                                    else:
                                        cond_skysub = np.repeat(False,gen_dic['norders_instru'][inst])     #Conditions on original order indexes 
                                        cond_skysub[idx_ord_skysub] = True  
                                        idxsub_ord_skysub = np_where1D(cond_skysub[idx_ord_kept])          #Indexes in reduced tables 
                                else:hdulist_dat = hdulist
                                    
                                if inst in ['HARPS','ESPRESSO','ESPRESSO_MR','HARPN','NIRPS_HA','NIRPS_HE']:

                                    #Bin centers
                                    #    - dimension norder x nbins
                                    if gen_dic['sp_frame']=='air':ikey_wav = 5
                                    elif gen_dic['sp_frame']=='vacuum':ikey_wav = 4
                                    data_dic_temp['cen_bins'][iexp] = (hdulist_dat[ikey_wav].data)[idx_ord_kept]
                                    dll = (hdulist_dat[ikey_wav+2].data)[idx_ord_kept]
                                    if (vis_path_skysub_exp is not None) and (gen_dic['fibB_corr'][inst][vis]!='all'):
                                        data_dic_temp['cen_bins'][iexp][idxsub_ord_skysub] = (hdulist_skysub[ikey_wav].data)[idx_ord_kept][idxsub_ord_skysub]                                 
                                        dll[idxsub_ord_skysub] = (hdulist_skysub[ikey_wav+2].data)[idx_ord_kept][idxsub_ord_skysub]   
                                        
                                    #Spectra   
                                    #    - the overall flux level is changed in the sky-corrected data, messing up with the flux balance corrections if only some orders were corrected
                                    #      we thus roughly rescale the flux in the corrected orders to their original level (assuming constant resolution over the order to calculate the mean flux)   
                                    #    - the flux is per unit of pixel, and must be divided by the pixel spectral size stored in dll to be passed in units of wavelength
                                    #    - the flux is not per unit of time, but corresponds to the number of photoelectrons measured during the exposure corrected for the blaze function
                                    data_dic_temp['flux'][iexp] = (hdulist_dat[1].data)[idx_ord_kept]/dll
                                    err_raw = (hdulist_dat[2].data)[idx_ord_kept]/dll  
                                    if (vis_path_skysub_exp is not None) and (gen_dic['fibB_corr'][inst][vis]!='all'):
                                        dll_ord_skysub = dll[idxsub_ord_skysub]
                                        mean_Fraw_ord_skysub = np.nansum(data_dic_temp['flux'][iexp][idxsub_ord_skysub]*dll_ord_skysub,axis=1)/np.nansum(dll_ord_skysub,axis=1)
                                        data_dic_temp['flux'][iexp][idxsub_ord_skysub] = (hdulist_skysub[1].data)[idx_ord_kept][idxsub_ord_skysub]/dll_ord_skysub
                                        mean_Fcorr_ord_skysub = np.nansum(data_dic_temp['flux'][iexp][idxsub_ord_skysub]*dll_ord_skysub,axis=1)/np.nansum(dll_ord_skysub,axis=1)
                                        skysub_corrfact = (mean_Fraw_ord_skysub/mean_Fcorr_ord_skysub)[:,None]
                                        data_dic_temp['flux'][iexp][idxsub_ord_skysub]*=skysub_corrfact
                                        err_raw[idxsub_ord_skysub] = ((hdulist_skysub[2].data)[idx_ord_kept][idxsub_ord_skysub]/dll_ord_skysub)*skysub_corrfact

                                    #Pixels quality
                                    #    - unextracted pixels (on each side of the shorter blue orders and in between the blue/red chips in ESPRESSO) are considered as undefined
                                    qualdata_exp = (hdulist_dat[3].data)[idx_ord_kept]
                                    if (vis_path_skysub_exp is not None) and (gen_dic['fibB_corr'][inst][vis]!='all'):
                                        qualdata_exp[idxsub_ord_skysub] = (hdulist_skysub[3].data)[idx_ord_kept][idxsub_ord_skysub]
                                    cond_bad_exp |= (qualdata_exp == 16384)
                                    
                                    #Remove pixels at the edge of the red detector
                                    #    - usually too poorly defined even for bright stars
                                    if inst=='ESPRESSO':
                                        for iord_bad in [90,91]: 
                                            isub_ord = np_where1D(np.array(idx_ord_kept)==iord_bad)
                                            if len(isub_ord)>0:
                                                data_dic_temp['flux'][iexp][isub_ord,0:30] = np.nan
                                                qualdata_exp[isub_ord,0:30] = 1                                   

                                elif inst=='CARMENES_VIS':             
                                    if (vis_path_skysub_exp is not None):stop('No sky-corrected data available')

                                    #Bin centers
                                    #    - dimension norder x nbins
                                    #    - in vacuum
                                    data_dic_temp['cen_bins'][iexp] = (hdulist_dat['WAVE'].data)[idx_ord_kept]
                                    if gen_dic['sp_frame']=='air':data_dic_temp['cen_bins'][iexp]/=air_index(data_dic_temp['cen_bins'][iexp], t=15., p=760.)

                                    #Spectra   
                                    data_dic_temp['flux'][iexp] = (hdulist_dat['SPEC'].data)[idx_ord_kept]
                                    err_raw = (hdulist_dat['SIG'].data)[idx_ord_kept]                            

                                    #BERV correction to shift exposures from the Earth to the solar System barycenter
                                    #    - the BERV is the radial velocity of the Earth with respect to the solar System barycenter
                                    #    - see align_data:
                                    # w_source/w_received   = (1 + v_receiver/c )/(1 - v_source/c )  
                                    #      the source is the Earth, and the receiver the barycenter             
                                    #           v_receiver = 0
                                    #           v_source = - BERV since BERV < 0 when Earth is moving toward the barycenter         
                                    #           w_earth/w_bary   = 1 /(1 + BERV/c )                  
                                    #           w_bary = w_earth*(1 + BERV/c )  
                                    #           and using the more precise relativistic formula:
                                    #           w_bary = w_earth*sqrt(1 + BERV/c )/sqrt(1 - BERV/c )                                      
                                    #    - we also include a relativistic correction for the rotation of the Earth
                                    data_dic_temp['cen_bins'][iexp]*=spec_dopshift(-data_prop[inst][vis]['BERV'][iexp])*(1.+1.55e-8)

                                elif inst=='EXPRES':              
                                    if (vis_path_skysub_exp is not None):stop('No sky-corrected data available')
                                    
                                    #Bin centers
                                    #    - dimension norder x nbins
                                    #    - we use the chromatic-barycentric-corrected Excalibur wavelengths 
                                    #      Excalibur is a non-parametric, hierarchical method for wavelength calibration. These wavelengths suffer less from systematic or instrumental errors, but cover a smaller spectral range
                                    #      at undefined Excalibur wavelengths we nonetheless set the regular barycentric-corrected wavelength, as ANTARESS requires fully defined spectral tables
                                    #      since flux values are set to nan at undefined Excalibur wavelengths this has no impact on the results
                                    data_dic_temp['cen_bins'][iexp] = ((hdulist_dat[1].data)['bary_wavelength'])[idx_ord_kept]     
                                    bary_excalibur = ((hdulist_dat[1].data)['bary_excalibur'])[idx_ord_kept]
                                    cond_def_wav = (hdulist_dat[1].data)['excalibur_mask'][idx_ord_kept]
                                    data_dic_temp['cen_bins'][iexp][cond_def_wav] = bary_excalibur[cond_def_wav]                                                                        
                                    if gen_dic['sp_frame']=='air':data_dic_temp['cen_bins'][iexp]/=air_index(data_dic_temp['cen_bins'][iexp], t=15., p=760.)

                                    #Spectra 
                                    #    - the flux is not per unit of time, but corresponds to the number of photoelectrons measured during the exposure corrected for the blaze function   
                                    data_dic_temp['flux'][iexp] = (hdulist_dat[1].data)['spectrum'][idx_ord_kept]
                                    err_raw = (hdulist_dat[1].data)['uncertainty'][idx_ord_kept]

                                    #Set flux to nan at undefined Excalibur wavelengths
                                    #    - for consistency with the way pixels are usually defined
                                    #    - excalibur_mask: gives a mask that excludes pixels without Excalibur wavelengths, which have been set to NaNs. These are pixels that do not fall between calibration lines in the order, and so are only on the edges of each order, or outside the range of LFC lines, as described above.
                                    data_dic_temp['flux'][iexp][~cond_def_wav]=np.nan

                                    #Pixels quality
                                    #    - bad pixels are set to False originally
                                    #    - pixel_mask: pixels with too low signal to return a proper extracted value. This encompasses largely pixels on the edges of orders
                                    qualdata_exp = np.zeros(data_dic_temp['flux'][iexp].shape)
                                    qualdata_exp[~(hdulist_dat[1].data)['pixel_mask'][idx_ord_kept]] = 1.
                                    
                                    #Telluric spectrum
                                    if data_inst[vis]['tell_sp'] and (gen_dic['calc_tell_mode']=='input'):
                                        data_dic_temp['tell'][iexp] = (hdulist_dat[1].data)['tellurics'][idx_ord_kept]  

                                else:
                                    stop('Spectra upload TBD for this instrument') 
                        
                        #Set bad quality pixels to nan
                        #    - bad pixels have flag > 0
                        #    - unless requested by the user those pixels are not set to undefined, because we cannot propagate detailed quality information when a profile is resampled / modified and leaving them undefined 
                        # will 'bleed' to adjacent pixels when resampling and will limit some calculations
                        if gen_dic['bad2nan']:cond_bad_exp |= (qualdata_exp > 0.) 
                        if True in cond_bad_exp:data_dic_temp['flux'][iexp][cond_bad_exp] = np.nan

                        #Set undefined errors to square-root of flux as first-order approximation
                        #    - errors will be redefined for local CCF using their continuum dispersion
                        #    - pixels with negative values are set to undefined, as there is no simple way to attribute errors to them
                        if not gen_dic[inst][vis]['flag_err']:
                            cond_pos_exp = data_dic_temp['flux'][iexp]>=0.
                            data_dic_temp['flux'][iexp][~cond_pos_exp] = np.nan
                            err_raw = np.zeros(data_inst[vis]['nspec'])
                            err_raw[cond_pos_exp]=np.sqrt(data_dic_temp['flux'][iexp][cond_pos_exp]) 


                        #------------------------------

                        #Processing orders
                        for iord in range(data_inst['nord']):
                            
                            #Set pixels with null flux on order sides to undefined    
                            #    - those pixels contain no information and can be removed to speed calculations
                            #    - we do not set to undefined all null pixels in case some are within the spectra
                            if ('spec' in data_inst['type']):
                                cumsum = np.nancumsum(data_dic_temp['flux'][iexp][iord])
                                if (True in (cumsum>0.)):
                                    idx_null_left = np_where1D(cumsum>0.)[0]-1
                                    if (idx_null_left>=0) and (np.nansum(data_dic_temp['flux'][iexp][iord][0:idx_null_left+1])==0.):data_dic_temp['flux'][iexp][iord][0:idx_null_left+1] = np.nan
                                if (True in (cumsum<cumsum[-1])):                                
                                    idx_null_right = np_where1D(cumsum<cumsum[-1])[-1]+2                      
                                    if (idx_null_right<data_inst[vis]['nspec']) and (np.nansum(data_dic_temp['flux'][iexp][iord][idx_null_right::])==0.):data_dic_temp['flux'][iexp][iord][idx_null_right::] = np.nan

                            #Banded covariance matrix
                            #    - dimensions (nd+1,nsp) for a given (exposure,order)
                            #    - here we have to assume there are no correlations in the uploaded profile, and store its error table in the matrix diagonal
                            data_dic_temp['cov'][iexp,iord] = (err_raw[iord]**2.)[None,:]

                    #-----------------------------------------------------
                    #Observational properties associated with each exposure
                    #-----------------------------------------------------
                    if not gen_dic['mock_data']: 
                        
                        #Airmass (mean between start and end of exposure)
                        #    - AM = sec(z) = 1/cos(z) with z the zenith angle
                        if inst in ['HARPN','CARMENES_VIS']:data_prop[inst][vis]['AM'][iexp]=hdr['AIRMASS']
                        elif inst=='CORALIE':data_prop[inst][vis]['AM'][iexp]=hdr['HIERARCH ESO OBS TARG AIRMASS']
                        elif inst=='EXPRES':data_prop[inst][vis]['AM'][iexp]=hdr['AIRMASS']
                        else:
                            if inst in ['HARPS','SOPHIE','ESPRESSO','NIRPS_HA','NIRPS_HE']:
                                if inst in ['SOPHIE','HARPS','NIRPS_HA','NIRPS_HE']:pre_txt=''                             
                                elif inst=='ESPRESSO':pre_txt = tel_inst
                                data_prop[inst][vis]['AM'][iexp]=0.5*(hdr['HIERARCH '+facil_inst+' TEL'+pre_txt+' AIRM START']+hdr['HIERARCH '+facil_inst+' TEL'+pre_txt+' AIRM START'] )                                                                
                            elif inst=='ESPRESSO_MR':
                                data_prop[inst][vis]['AM_UT'][iexp]=[]
                                for tel_inst_loc in ['TEL1','TEL2','TEL3','TEL4']:
                                    pre_txt='HIERARCH ESO '+tel_inst_loc+' AIRM'
                                    data_prop[inst][vis]['AM_UT'][iexp]+=[0.5*(hdr[pre_txt+' START']+hdr[pre_txt+' END'] )]    
                                data_prop[inst][vis]['AM'][iexp]=np.mean(data_prop[inst][vis]['AM_UT'][iexp])
                                
                        #Integrated water vapor at zenith (mm)
                        if inst=='ESPRESSO':
                            data_prop[inst][vis]['IWV_AM'][iexp]=0.5*(hdr['HIERARCH '+facil_inst+' TEL'+tel_inst+' AMBI IWV START']+hdr['HIERARCH '+facil_inst+' TEL'+tel_inst+' AMBI IWV START'] )                                                             
                        #No radiometer available at La Silla, TNG and CAHA
                        elif inst in ['CARMENES_VIS','HARPN','HARPS','SOPHIE','NIRPS_HA','NIRPS_HE']:
                            data_prop[inst][vis]['IWV_AM'][iexp] = 1.
                            
                        #Temperature (K)
                        if inst in ['HARPS','SOPHIE','ESPRESSO','NIRPS_HA','NIRPS_HE']:
                            if inst in ['SOPHIE','HARPS','NIRPS_HA','NIRPS_HE']:pre_txt=''                             
                            elif inst=='ESPRESSO':pre_txt = tel_inst
                            data_prop[inst][vis]['TEMP'][iexp] = hdr['HIERARCH '+facil_inst+' TEL'+pre_txt+' AMBI TEMP'] + 273.15
                        elif inst=='HARPN':   #Validated by Rosario Cosentino (harpsn instrument scientist)
                            pre_txt=''
                            data_prop[inst][vis]['TEMP'][iexp] = hdr['HIERARCH '+facil_inst+' METEO TEMP10M'] + 273.15
                        elif inst=='CARMENES_VIS':
                            pre_txt=''
                            data_prop[inst][vis]['TEMP'][iexp] = hdr['HIERARCH '+facil_inst+' GEN AMBI TEMPERATURE'] + 273.15
                            
                        #Pressure (atm)
                        if inst in ['HARPS','SOPHIE','ESPRESSO','NIRPS_HA','NIRPS_HE']:
                            if inst in ['SOPHIE','HARPS','NIRPS_HA','NIRPS_HE']:pre_txt=''                             
                            elif inst=='ESPRESSO':pre_txt = tel_inst
                            data_prop[inst][vis]['PRESS'][iexp]=0.5*(hdr['HIERARCH '+facil_inst+' TEL'+pre_txt+' AMBI PRES START']+hdr['HIERARCH '+facil_inst+' TEL'+pre_txt+' AMBI PRES START'] )/1013.2501                                                             
                        elif inst=='HARPN':    #Validated by Rosario Cosentino (harpsn instrument scientist)
                            pre_txt=''
                            data_prop[inst][vis]['PRESS'][iexp] = hdr['HIERARCH '+facil_inst+' METEO PRESSURE'] / 1013.2501
                        elif inst=='CARMENES_VIS':
                            pre_txt=''
                            data_prop[inst][vis]['PRESS'][iexp] = hdr['HIERARCH '+facil_inst+' GEN AMBI PRESSURE'] / 1013.2501
                            
                        #Seeing (mean between start and end of exposure)
                        #    - not saved for HARPSN data
                        if inst=='CARMENES_VIS':
                            data_prop[inst][vis]['seeing'][iexp]=hdr['HIERARCH CAHA GEN AMBI SEEING']  
                        elif inst=='HARPS':
                            pre_txt='HIERARCH ESO TEL AMBI FWHM' 
                            if (pre_txt+' START' in hdr):data_prop[inst][vis]['seeing'][iexp]=0.5*(hdr[pre_txt+' START']+hdr[pre_txt+' END'] )  
                        elif inst=='SOPHIE':
                            data_prop[inst][vis]['seeing'][iexp]=hdr['HIERARCH OHP GUID SEEING']  
                        elif inst=='ESPRESSO_MR':
                            data_prop[inst][vis]['seeing_UT'][iexp]=[]
                            for tel_inst_loc in ['TEL1','TEL2','TEL3','TEL4']:
                                data_prop[inst][vis]['seeing_UT'][iexp]+=[hdr['HIERARCH ESO '+tel_inst_loc+' IA FWHM']]  
                            data_prop[inst][vis]['seeing'][iexp]=np.mean(data_prop[inst][vis]['seeing_UT'][iexp])                            
                        elif inst=='ESPRESSO':
                            data_prop[inst][vis]['seeing'][iexp]=hdr['HIERARCH ESO TEL'+tel_inst+' IA FWHM']      #seeing corrected for airmass                 
            
                        #Shape of PSF
                        #      first and second values: size in X and Y
                        #      third value : average (quadratic) size
                        #      fourth value : angle      
                        if inst in ['CORALIE','SOPHIE','HARPS','ESPRESSO','ESPRESSO_MR','HARPN']:
                            if inst=='HARPN':
                                pre_txt='HIERARCH '+facil_inst+' AG ACQ'                    
                                data_prop[inst][vis]['PSF_prop'][iexp,0]=hdr[pre_txt+' FWHMX']
                                data_prop[inst][vis]['PSF_prop'][iexp,1]=hdr[pre_txt+' FWHMY']
                            elif inst in ['ESPRESSO','ESPRESSO_MR']:
                                pre_txt='HIERARCH '+facil_inst+' OCS ACQ3'                    
                                if pre_txt+' FWHMX ARCSEC' in hdr:
                                    data_prop[inst][vis]['PSF_prop'][iexp,0]=hdr[pre_txt+' FWHMX ARCSEC']
                                    data_prop[inst][vis]['PSF_prop'][iexp,1]=hdr[pre_txt+' FWHMY ARCSEC']                
                                    data_prop[inst][vis]['PSF_prop'][iexp,2]=np.sqrt(data_prop[inst][vis]['PSF_prop'][iexp,0]**2.+data_prop[inst][vis]['PSF_prop'][iexp,1]**2.)
                            if data_prop[inst][vis]['PSF_prop'][iexp,0]==0.:
                                data_prop[inst][vis]['PSF_prop'][iexp,3]=np.nan
                            else:
                                data_prop[inst][vis]['PSF_prop'][iexp,3]=np.arctan2(data_prop[inst][vis]['PSF_prop'][iexp,1],data_prop[inst][vis]['PSF_prop'][iexp,0])*180./np.pi
            
                        #Coefficient of color correction for Earth atmosphere
                        #    - a polynomial is fitted to the ratio between the spectrum averaged over each order, and a stellar template
                        #    - the coefficient below give the minimum and maximum values of this ratio (they may thus correspond to any order)
                        #      a low value corresponds to a strong flux decrease
                        if inst in ['CORALIE','SOPHIE','HARPS','ESPRESSO','ESPRESSO_MR','HARPN']:
                            if inst in ['CORALIE']:reduc_txt='ESO DRS '      
                            elif inst=='SOPHIE':reduc_txt='OHP DRS '     
                            elif inst in ['HARPS','ESPRESSO','ESPRESSO_MR']:reduc_txt='ESO QC '
                            elif inst=='HARPN':reduc_txt='ESO QC ' if ('HIERARCH ESO QC FLUX CORR') in hdr else 'TNG QC '
                            data_prop[inst][vis]['colcorrmin'][iexp]=hdr['HIERARCH '+reduc_txt+'FLUX CORR MIN'] 
                            data_prop[inst][vis]['colcorrmax'][iexp]=hdr['HIERARCH '+reduc_txt+'FLUX CORR MAX'] 
        
                            #Polynomial for color correction   
                            #    - retrieved for each order
                            if inst in ['HARPS','ESPRESSO','HARPN']:
                                for iord in range(data_inst['nord']):
                                    data_prop[inst][vis]['colcorr_ord'][iexp,iord]=hdr['HIERARCH '+reduc_txt+'ORDER'+str(iord+1)+' FLUX CORR'] 
                                
                        #Saturation check
                        if inst in ['HARPS','ESPRESSO','HARPN']:
                            data_prop[inst][vis]['satur_check'][iexp]=hdr['HIERARCH '+reduc_txt+'SATURATION CHECK']

                        #ADC and PIEZO properties
                        if inst in ['ESPRESSO']:
                            ref_adc = {'ESO-VLT-U1':'','ESO-VLT-U2':'2','ESO-VLT-U3':'3','ESO-VLT-U4':'4'}[hdr['TELESCOP']]
                            data_prop[inst][vis]['adc_prop'][iexp,0]=hdr['HIERARCH ESO INS'+ref_adc+' ADC1 POSANG']
                            data_prop[inst][vis]['adc_prop'][iexp,3]=hdr['HIERARCH ESO INS'+ref_adc+' ADC2 POSANG']
                            
                            #Conversion from hhmmss.ssss to deg
                            data_prop[inst][vis]['adc_prop'][iexp,1]=hdr['HIERARCH ESO INS'+ref_adc+' ADC1 RA']
                            data_prop[inst][vis]['adc_prop'][iexp,2]=hdr['HIERARCH ESO INS'+ref_adc+' ADC1 DEC']
                            data_prop[inst][vis]['adc_prop'][iexp,4]=hdr['HIERARCH ESO INS'+ref_adc+' ADC2 RA']
                            data_prop[inst][vis]['adc_prop'][iexp,5]=hdr['HIERARCH ESO INS'+ref_adc+' ADC2 DEC']
                            for ikey in [1,2,4,5]:
                                str_split = str(data_prop[inst][vis]['adc_prop'][iexp,ikey]).split('.')
                                h_key = float(str_split[0][-6:-4])
                                min_key = float(str_split[0][-4:-2])
                                sec_key = float(str_split[0][-2:]+'.'+str_split[1])
                                data_prop[inst][vis]['adc_prop'][iexp,ikey] = 15.*h_key+15.*min_key/60.+15.*sec_key/3600.

                            #PIEZO properties
                            data_prop[inst][vis]['piezo_prop'][iexp,0]=hdr['HIERARCH ESO INS'+ref_adc+' TILT1 VAL1']
                            data_prop[inst][vis]['piezo_prop'][iexp,1]=hdr['HIERARCH ESO INS'+ref_adc+' TILT1 VAL2']
                            data_prop[inst][vis]['piezo_prop'][iexp,2]=hdr['HIERARCH ESO INS'+ref_adc+' TILT2 VAL1']
                            data_prop[inst][vis]['piezo_prop'][iexp,3]=hdr['HIERARCH ESO INS'+ref_adc+' TILT2 VAL2']   

                        #Guide star coordinates
                        if inst in ['ESPRESSO']:
                            data_prop[inst][vis]['guid_coord'][iexp,0]=hdr['HIERARCH ESO ADA'+tel_inst+' GUID RA']
                            data_prop[inst][vis]['guid_coord'][iexp,1]=hdr['HIERARCH ESO ADA'+tel_inst+' GUID DEC']

                        #Properties derived by the pipeline
                        if ('rv_pip' in DI_data_inst[vis]):
                            if inst in ['CORALIE']:
                                pre_txt='HIERARCH ESO DRS CCF'      
                                DI_data_inst[vis]['rv_pip'][iexp]=hdr[pre_txt+' RVC'] #Baryc RV (drift corrected) (km/s)
                                DI_data_inst[vis]['erv_pip'][iexp]=hdr['HIERARCH ESO DRS DVRMS']/1e3                           
                            elif inst=='SOPHIE':
                                pre_txt='HIERARCH OHP DRS CCF'               
                                DI_data_inst[vis]['rv_pip'][iexp]=hdr[pre_txt+' RV']  #Baryc RV (drift corrected ?? ) (km/s) 
                                DI_data_inst[vis]['erv_pip'][iexp]=hdr[pre_txt+' ERR']
                            elif inst in ['HARPS','ESPRESSO','ESPRESSO_MR','HARPN','NIRPS_HA','NIRPS_HE']:
                                reduc_txt=facil_inst+' QC '
                                pre_txt='HIERARCH '+reduc_txt+'CCF'        
                                DI_data_inst[vis]['rv_pip'][iexp]=hdr[pre_txt+' RV'] #Baryc RV (drift corrected) (km/s)   
                                DI_data_inst[vis]['erv_pip'][iexp]=hdr[pre_txt+' RV ERROR']
                            DI_data_inst[vis]['FWHM_pip'][iexp]=hdr[pre_txt+' FWHM'] 
                            DI_data_inst[vis]['ctrst_pip'][iexp]=hdr[pre_txt+' CONTRAST'] 
                            if inst in ['HARPS','SOPHIE','ESPRESSO','ESPRESSO_MR','HARPN','NIRPS_HA','NIRPS_HE']:
                                DI_data_inst[vis]['eFWHM_pip'][iexp]=hdr[pre_txt+' FWHM ERROR'] 
                                DI_data_inst[vis]['ectrst_pip'][iexp]=hdr[pre_txt+' CONTRAST ERROR'] 

                        #SNRs in all orders
                        #    - SNRs tables are kept associated with original orders
                        if inst in ['SOPHIE','CORALIE']: 
                            if inst in ['CORALIE']:pre_txt='SPE'
                            elif inst=='SOPHIE':pre_txt='CAL'  
                            for iorder in range(data_inst['nord_ref']):
                                data_prop[inst][vis]['SNRs'][iexp,iorder]=hdr['HIERARCH '+facil_inst+' DRS '+pre_txt+' EXT SN'+str(iorder)]        
                        elif inst=='CARMENES_VIS':
                            for iorder in range(data_inst['nord_ref']):
                                data_prop[inst][vis]['SNRs'][iexp,iorder]=hdr['HIERARCH CARACAL FOX SNR '+str(iorder)]  
                        elif inst in ['HARPS','ESPRESSO','ESPRESSO_MR','HARPN','NIRPS_HA','NIRPS_HE']:
                            for iorder in range(data_inst['nord_ref']):
                                data_prop[inst][vis]['SNRs'][iexp,iorder]=hdr['HIERARCH '+facil_inst+' QC ORDER'+str(iorder+1)+' SNR']  
                        elif inst=='EXPRES':
                            #Correspondance channel : original orders
                            #   1 : 28 ; 2 : 37 ; 3 : 45 ; 4 : 51 ; 5 : 58 ; 6 : 63 ; 7 : 68 ; 8 : 72
                            SNR_exp = np.zeros(86,dtype=float)*np.nan 
                            for ichannel,iord in zip(range(1,9),[28,37,45,51,58,63,68,72]): 
                                SNR_exp[iord] = (hdulist[2].header)['HIERARCH channel'+str(ichannel)+'_SNR']
                            data_prop[inst][vis]['SNRs'][iexp] = SNR_exp    

                        #Blaze and wavelength calibration files
                        if inst=='ESPRESSO':
                            for ii in range(1,34):
                                if hdr['HIERARCH ESO PRO REC1 CAL%i CATG'%(ii)] == 'BLAZE_A':
                                    data_prop[inst][vis]['BLAZE_A'][iexp]=hdr['HIERARCH ESO PRO REC1 CAL%i DATAMD5'%(ii)]
                                elif hdr['HIERARCH ESO PRO REC1 CAL%i CATG'%(ii)] == 'BLAZE_B':
                                    data_prop[inst][vis]['BLAZE_B'][iexp]=hdr['HIERARCH ESO PRO REC1 CAL%i DATAMD5'%(ii)]
                                elif hdr['HIERARCH ESO PRO REC1 CAL%i CATG'%(ii)] == 'WAVE_MATRIX_THAR_FP_A':
                                    data_prop[inst][vis]['WAVE_MATRIX_THAR_FP_A'][iexp]=hdr['HIERARCH ESO PRO REC1 CAL%i DATAMD5'%(ii)]                            
                                elif hdr['HIERARCH ESO PRO REC1 CAL%i CATG'%(ii)] == 'WAVE_MATRIX_FP_THAR_B':
                                    data_prop[inst][vis]['WAVE_MATRIX_THAR_FP_B'][iexp]=hdr['HIERARCH ESO PRO REC1 CAL%i DATAMD5'%(ii)]
                                                                    
                        #Telescope altitude angle (degres)
                        if inst=='ESPRESSO':
                            data_prop[inst][vis]['alt'][iexp]=hdr['HIERARCH ESO TEL'+tel_inst+' ALT'] 

                        #Telescope azimuth angle (degrees)
                        if inst=='ESPRESSO':
                            data_prop[inst][vis]['az'][iexp]=hdr['HIERARCH ESO TEL'+tel_inst+' AZ']                                         
                        
                        #Activity indexes
                        if DACE_sp:
                            dace_idexp = closest( DACE_idx['bjd'] ,bjd_exp )
                            for key in data_inst[vis]['act_idx']:
                                data_prop[inst][vis][key][iexp,:] = [DACE_idx[key][dace_idexp],DACE_idx[key+'_err'][dace_idexp]]
                        elif inst=='EXPRES':
                            data_prop[inst][vis]['ha'][iexp,:] = [(hdulist[1].header)['HALPHA'],1.879E-2]
                            data_prop[inst][vis]['s'][iexp,:] = [(hdulist[1].header)['S-VALUE'], 6.611E-3]
                            

                    ### End of current exposure    
                    
                ### End of exposures in visit
           
                #------------------------------------------------------------------------------------

                #Scaling of errors
                if (inst not in gen_dic['g_err']):gen_dic['g_err'][inst]=1.
                elif gen_dic['g_err'][inst]!=1.:
                    print('           Uncertainties are scaled by sqrt('+str(gen_dic['g_err'][inst])+')')
                    data_dic_temp['cov']*=gen_dic['g_err'][inst]

                #Sort exposures by increasing time
                #    - remove exposures if requested, using their index after time-sorting 
                if (inst in gen_dic['used_exp']) and (vis in gen_dic['used_exp'][inst]) and (len(gen_dic['used_exp'][inst][vis])>0):remove_exp=True
                else:remove_exp=False
                w_sorted=coord_dic[inst][vis]['bjd'].argsort()    
                for pl_loc in data_inst[vis]['transit_pl']:
                    for key in ['ecl','cen_ph','st_ph','end_ph','ph_dur','rv_pl']:
                        coord_dic[inst][vis][pl_loc][key]=coord_dic[inst][vis][pl_loc][key][w_sorted]                        
                        if remove_exp:coord_dic[inst][vis][pl_loc][key] = coord_dic[inst][vis][pl_loc][key][gen_dic['used_exp'][inst][vis]]
                    for key in ['cen_pos','st_pos','end_pos']:
                        coord_dic[inst][vis][pl_loc][key]=coord_dic[inst][vis][pl_loc][key][:,w_sorted] 
                        if remove_exp:coord_dic[inst][vis][pl_loc][key] = coord_dic[inst][vis][pl_loc][key][:,gen_dic['used_exp'][inst][vis]] 
                for key in ['bjd','t_dur','RV_star_solCDM','RV_star_stelCDM']:
                    coord_dic[inst][vis][key]=coord_dic[inst][vis][key][w_sorted]
                    if remove_exp:coord_dic[inst][vis][key] = coord_dic[inst][vis][key][gen_dic['used_exp'][inst][vis]]
                            
                for key in data_prop[inst][vis]:
                    if type(data_prop[inst][vis][key]) not in [str,int]:
                        if len(data_prop[inst][vis][key].shape)==1:
                            data_prop[inst][vis][key]=data_prop[inst][vis][key][w_sorted]
                            if remove_exp:data_prop[inst][vis][key] = data_prop[inst][vis][key][gen_dic['used_exp'][inst][vis]]
                        else:
                            data_prop[inst][vis][key]=data_prop[inst][vis][key][w_sorted,:] 
                            if remove_exp:data_prop[inst][vis][key] = data_prop[inst][vis][key][gen_dic['used_exp'][inst][vis],:]
                for key in data_dic_temp:
                    data_dic_temp[key]= data_dic_temp[key][w_sorted]
                    if remove_exp:data_dic_temp[key] = data_dic_temp[key][gen_dic['used_exp'][inst][vis]]
                for key in DI_data_inst[vis]:
                    DI_data_inst[vis][key]=DI_data_inst[vis][key][w_sorted]   
                    if remove_exp:DI_data_inst[vis][key] = DI_data_inst[vis][key][gen_dic['used_exp'][inst][vis]] 
                if remove_exp:
                    n_in_visit = len(gen_dic['used_exp'][inst][vis])
                    data_inst[vis]['n_in_visit'] = n_in_visit
                    data_inst[vis]['dim_all'][0] = n_in_visit
                    data_inst[vis]['dim_sp'][0] = n_in_visit
                    data_inst[vis]['dim_ord'][0] = n_in_visit
             
                #Check if exposures have same duration
                coord_dic[inst][vis]['t_dur_d'] = coord_dic[inst][vis]['t_dur']/(3600.*24.) 
                if (coord_dic[inst][vis]['t_dur']==coord_dic[inst][vis]['t_dur'][0]).all():coord_dic[inst][vis]['cst_tdur']=True
                else:coord_dic[inst][vis]['cst_tdur']=False   

                #Mean SNR
                #    - used to weigh CCFs (over contributing orders)
                #    - used to weigh spectra (over all orders)
                if (not gen_dic['mock_data']):
                    if ('CCF' in data_inst['type']):  
                        if inst not in gen_dic['orders4ccf']:data_prop[inst][vis]['mean_SNR'] = np.nanmean(data_prop[inst][vis]['SNRs'],axis=1)  
                        else:data_prop[inst][vis]['mean_SNR'] = np.nanmean(data_prop[inst][vis]['SNRs'][:,gen_dic['orders4ccf'][inst]],axis=1)
                    elif ('spec' in data_inst['type']):
                        data_prop[inst][vis]['mean_SNR'] = np.nanmean(data_prop[inst][vis]['SNRs'],axis=1)

                #Identify exposures west/east of meridian
                #    - celestial meridian is at azimuth = 180, with azimuth counted clockwise from the celestial north
                #      exposures at az > 180 are to the west of the meridian, exposures at az < 180 to the east
                if (inst=='ESPRESSO') and (not gen_dic['mock_data']):
                    cond_eastmer = data_prop[inst][vis]['az'] < 180.
                    data_inst[vis]['idx_eastmer'] = np_where1D(cond_eastmer)
                    data_inst[vis]['idx_westmer'] = np_where1D(~cond_eastmer) 
                    data_inst[vis]['idx_mer'] = closest(data_prop[inst][vis]['az'] , 180.)               
                    
                    #Estimate telescope altitude (deg) at meridian
                    min_bjd = coord_dic[inst][vis]['bjd'][0]
                    max_bjd = coord_dic[inst][vis]['bjd'][-1]
                    dbjd_HR = 1./(3600.*24.)
                    nbjd_HR = round((max_bjd-min_bjd)/dbjd_HR)
                    bjd_HR=min_bjd+dbjd_HR*np.arange(nbjd_HR)
                    az_HR = CubicSpline(coord_dic[inst][vis]['bjd'],data_prop[inst][vis]['az'])(bjd_HR)
                    idx_mer_HR = closest(az_HR,180.)
                    alt_HR = CubicSpline(coord_dic[inst][vis]['bjd'],data_prop[inst][vis]['alt'])(bjd_HR)
                    data_inst[vis]['alt_mer'] = alt_HR[idx_mer_HR]  
                    data_inst[vis]['z_mer']  = np.sin(alt_HR[idx_mer_HR]*np.pi/180.) 
                     
                
                #------------------------------------------------------------------------------------
               
                #Undefined bins
                data_dic_temp['cond_def'] = ~np.isnan(data_dic_temp['flux'])
                    
                #Definition of edge spectral table
                #    - spectral bins must be continuous for the resampling routine of the pipeline to work
                #    - bin width are included in ESPRESSO files, but to have exactly the same lower/upper boundary between successive bins we redefine them manually  
                data_dic_temp['edge_bins'] = def_edge_tab(data_dic_temp['cen_bins'])
    
                #Common spectral tables
                #    - the common table is set to the table of the first exposure, defined in the input rest frame
                #    - for spectra it will be used indifferently to resample spectra defined in the input or the stellar rest frame
                #    - for CCF it will be shifted to the star rest frame in the alignmen tmodule, if called 
                #      we give the table the same structure as for spectra, with dimensions nord x nspec
                data_inst[vis]['proc_com_data_paths'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_'+vis+'_com'
                data_com = {'cen_bins':data_dic_temp['cen_bins'][0],'edge_bins':data_dic_temp['edge_bins'][0],'dim_exp':data_inst[vis]['dim_exp'],'nspec':data_inst[vis]['nspec']}

                #Exposures in current visit are defined on different tables
                if ('spec' in data_inst['type']) and (np.nanmax(np.abs(data_dic_temp['edge_bins']-data_com['edge_bins'][None,:,:]))>1e-6):
                    data_inst[vis]['comm_sp_tab'] = False

                    #Resampling data on wavelength table common to the visit if requested
                    if gen_dic['comm_sp_tab'][inst]:
                        print('           Resampling on common table')
                        for iexp in range(n_in_visit):
                            for iord in range(data_inst['nord']): 
                                data_dic_temp['flux'][iexp,iord],data_dic_temp['cov'][iexp][iord] = bind.resampling(data_com['edge_bins'][iord], data_dic_temp['edge_bins'][iexp,iord], data_dic_temp['flux'][iexp,iord] , cov = data_dic_temp['cov'][iexp][iord], kind=gen_dic['resamp_mode']) 
                                
                                #Resampling input telluric spectrum
                                if data_inst[vis]['tell_sp'] and (gen_dic['calc_tell_mode']=='input'):
                                    data_dic_temp['tell'][iexp,iord] = bind.resampling(data_com['edge_bins'][iord], data_dic_temp['edge_bins'][iexp,iord], data_dic_temp['tell'][iexp,iord], kind=gen_dic['resamp_mode'])         
                                  
                        data_dic_temp['cond_def'] = ~np.isnan(data_dic_temp['flux'])
                        
                        #Set flag that all exposures in the visit are now defined on a common table
                        data_inst[vis]['comm_sp_tab'] = True

                #Save table common to all visits
                #    - set to the common table of the first instrument visit
                if ivis==0:
                    data_dic[inst]['proc_com_data_path'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_com'
                    data_dic[inst]['dim_exp'] = deepcopy(data_inst[vis]['dim_exp'])
                    data_dic[inst]['nspec'] = deepcopy(data_inst[vis]['nspec'])
                    data_dic[inst]['com_vis'] = vis   
                    data_com_inst = deepcopy(data_com)
                    np.savez_compressed(data_dic[inst]['proc_com_data_path'],data = data_com_inst,allow_pickle=True)

                #Check wether the common table of current visit is the same as the common table for the instrument
                #    - flag set to False if :
                # + exposures of current visit are still defined on individual table 
                # + exposures of current visit are defined on a common table, but it is different from the instrument table (only possible if current visit is not the first, used as reference)   
                if (not data_inst[vis]['comm_sp_tab']) or ((ivis>0) and ((data_com_inst['nspec']!=data_com['nspec']) or (np.nanmax(np.abs(data_com_inst['edge_bins']-data_com['edge_bins']))>1e-6))):
                     data_dic[inst]['comm_sp_tab'] = False

                #------------------------------------------------------------------------------------

                #Final processing of exposures
                data_inst[vis]['proc_DI_data_paths']=gen_dic['save_data_dir']+'Processed_data/'+inst+'_'+vis+'_'
                for iexp in range(n_in_visit):

                    #Set masked pixels to nan
                    #    - masked orders and positions are assumed to be the same for all exposures, but positions have to be specific to a given order as orders can overlap
                    if (inst in gen_dic['masked_pix']) and (vis in gen_dic['masked_pix'][inst]):
                        mask_exp = gen_dic['masked_pix'][inst][vis]['exp_list']
                        if (len(mask_exp)==0) or (iexp in gen_dic['masked_pix'][inst][vis]):
                            for iord in gen_dic['masked_pix'][inst][vis]['ord_list']:
                                mask_bd = gen_dic['masked_pix'][inst][vis]['ord_list'][iord]
                                cond_mask  = np.zeros(data_inst[vis]['nspec'],dtype=bool)
                                for bd_int in mask_bd:
                                    cond_mask |= (data_dic_temp['edge_bins'][iexp,iord,0:-1]>=bd_int[0]) & (data_dic_temp['edge_bins'][iexp,iord,1:]<=bd_int[1])
                                data_dic_temp['cond_def'][iexp ,iord , cond_mask ] = False    
                                data_dic_temp['flux'][iexp ,iord , cond_mask ]  = np.nan                            

                    #Telluric spectrum
                    #    - if defined with input data
                    #    - path is made specific to the exposure to be able to point from in-transit to global profiles without copying them to disk
                    if data_inst[vis]['tell_sp'] and (gen_dic['calc_tell_mode']=='input'):
                        data_inst[vis]['tell_DI_data_paths'][iexp] = data_inst[vis]['proc_DI_data_paths']+'tell_'+str(iexp)
                        np.savez_compressed(data_inst[vis]['tell_DI_data_paths'][iexp], data = {'tell':data_dic_temp['tell'][iexp]},allow_pickle=True) 
                        data_dic_temp.pop('tell')    
                        
                    #Saving path to initialized raw data
                    #    - saving data per exposure to prevent size issue with npz files 
                    data_exp = {key:data_dic_temp[key][iexp] for key in ['cen_bins','edge_bins','flux','cov','cond_def']}                    
                    np.savez_compressed(data_inst[vis]['proc_DI_data_paths']+str(iexp),data=data_exp,allow_pickle=True)   
                               
                #Check for empty orders
                idx_ord_empty = np_where1D(np.sum(data_dic_temp['cond_def'],axis=(0,2)) == 0.)
                if len(idx_ord_empty)>0:
                    txt_str = '['
                    for iord in idx_ord_empty[0:-1]:txt_str+=str(iord)+','
                    txt_str+=str(idx_ord_empty[-1])+']'
                    print('           Empty orders : '+txt_str)

                #Saving defined edges of common spectral table
                low_bins_def = deepcopy(data_dic_temp['edge_bins'][:,:,0:-1])
                high_bins_def = deepcopy(data_dic_temp['edge_bins'][:,:,1::])
                low_bins_def[~data_dic_temp['cond_def']] = np.nan
                high_bins_def[~data_dic_temp['cond_def']] = np.nan
                data_com['min_edge_ord'] = np.nanmin(low_bins_def,axis=(0,2))             
                data_com['max_edge_ord'] = np.nanmax(high_bins_def,axis=(0,2))
                np.savez_compressed(data_inst[vis]['proc_com_data_paths'],data = data_com,allow_pickle=True)
                    
                #Saving dictionary elements defined within the routine
                np.savez_compressed(gen_dic['save_data_dir']+'Processed_data/Global/'+inst+'_'+vis,data_add=data_inst[vis],coord_add=coord_dic[inst][vis],data_prop_add=data_prop[inst][vis],gen_add=gen_dic[inst][vis],DI_data_add=DI_data_inst[vis],allow_pickle=True)
              
                #Saving useful keyword file        
                if gen_dic['sav_keywords']:
                    tup_save=(np.append('index',[str(iexp) for iexp in range(n_in_visit)]),)
                    form_save=('%-6s',)
                    for key,form in zip(['BLAZE_A','BLAZE_B','WAVE_MATRIX_THAR_FP_A','WAVE_MATRIX_THAR_FP_B'],
                                        ['%-35s'  ,'%-35s'  ,'%-35s'                ,'%-35s'   ]):
                        if key in data_prop[inst][vis]:
                            form_save+=(form,)
                            col_loc = np.append(key,[str(data_prop[inst][vis][key][iexp]) for iexp in range(n_in_visit)])
                            tup_save+=(col_loc,)
                    np.savetxt(gen_dic['save_data_dir']+'DIorig_prop/'+inst+'_'+vis+'_keywords.dat', np.column_stack(tup_save),fmt=form_save) 

                #------------------------------------------------------------------------------------
                      
                #Automatic continuum and fit range
                for key in ['DI','Intr','Atm']:
                    if gen_dic['fit_'+key+'_gen']:
                        autom_cont = True if (inst not in data_dic[key]['cont_range']) else False
                        autom_fit = True if ((inst not in data_dic[key]['fit_range']) or (vis not in data_dic[key]['fit_range'][inst])) else False
                        if autom_cont or autom_fit:
                            if (data_dic[key]['type'][inst]=='CCF'):
                                RVcen_bins = data_dic_temp['cen_bins'][:,0,:]
                                RVedge_bins = data_dic_temp['edge_bins'][:,0,:]
                            else:
                                iord_fit = data_dic[key]['fit_prof']['order'][inst]
                                RVcen_bins = c_light*((data_dic_temp['cen_bins'][:,iord_fit,:]/data_dic[key]['line_trans']) - 1.)
                                RVedge_bins = c_light*((data_dic_temp['edge_bins'][:,iord_fit,:]/data_dic[key]['line_trans']) - 1.)

                            #Estimate of systemic velocity for disk-integrated profiles
                            #    - as median of the RV corresponding to the CCF minima over the visit
                            if key=='DI':
                                idx_min_all = np.argmin(data_dic_temp['flux'][:,0,:],axis=1)
                                cen_RV = np.nanmedian(RVcen_bins[np.arange(n_in_visit),idx_min_all])
                                
                            #Intrinsic and atmospheric profiles are fitted in the star rest frame
                            else:cen_RV = 0.
    
                            #Minimum/maximum velocity of the CCF range
                            min_bin = np.max(np.nanmin(RVedge_bins))
                            max_bin = np.min(np.nanmax(RVedge_bins))                        
    
                            #Excluded range for disk-integrated and intrinsic profiles 
                            #    - we assume +-3 vsini accounts for both rotational and thermal broadening of DI profiles, and for rotational shift + thermal broade
                            if key in ['DI','Intr']:
                                min_exc = np.min([cen_RV - 3.*system_param['star']['vsini'] -5.,min_bin+5.])
                                max_exc = np.max([cen_RV + 3.*system_param['star']['vsini'] +5.,max_bin-5.])
                                
                            #Excluded range for atmospheric profiles
                            #    - we account for a width of 20km/s over the range of studied orbital velocities, for all planets considered for atmospheric signals
                            elif key=='Atm':
                                min_pl_RV = 1e100
                                max_pl_RV = -1e100
                                if data_dic['Atm']['pl_atm_sign']=='Absorption':idx_atm = gen_dic[inst][vis]['idx_in']                                 
                                elif data_dic['Atm']['pl_atm_sign']=='Emission': idx_atm = data_dic[inst][vis]['n_in_tr']
                                for pl_atm in data_dic[inst][vis]['pl_with_atm']:
                                    min_pl_RV = np.min([min_pl_RV,np.min(coord_dic[inst][vis][pl_atm]['rv_pl'][idx_atm])])
                                    max_pl_RV = np.max([max_pl_RV,np.max(coord_dic[inst][vis][pl_atm]['rv_pl'][idx_atm])])  
                                min_exc = np.min(min_pl_RV -20.,min_bin+5.)
                                max_exc = np.max(max_pl_RV +20.,max_bin-5.)    

                            #Outer boundaries of continuum and fitted ranges
                            min_contfit = np.max([min_bin, min_exc - 50.])
                            max_contfit = np.min([max_bin, max_exc + 50.])

                            #RV/wave conversion
                            if (data_dic[key]['type'][inst]!='CCF'):
                                min_contfit = data_dic[key]['line_trans']*(1.+(min_contfit/c_light))
                                max_contfit = data_dic[key]['line_trans']*(1.+(max_contfit/c_light))
                                min_exc = data_dic[key]['line_trans']*(1.+(min_exc/c_light))
                                max_exc = data_dic[key]['line_trans']*(1.+(max_exc/c_light))

                            #Continuum range
                            if autom_cont:
                                data_dic[key]['cont_range'][inst] = [[min_contfit,min_exc],[max_exc,max_contfit]]
                        
                            #Fitting range
                            if autom_fit:      
                                if inst not in data_dic[key]['fit_range']:data_dic[key]['fit_range'][inst]={}
                                data_dic[key]['fit_range'][inst][vis] = [[min_contfit,max_contfit]]

            ### End of visits

        #Saving general information
        np.savez_compressed(gen_dic['save_data_dir']+'Processed_data/Global/'+inst,gen_inst_add=gen_dic[inst],data_inst_add=data_dic[inst],allow_pickle=True) 
        
    #Retrieving data
    else:
        check_data({'path':gen_dic['save_data_dir']+'Processed_data/Global/'+inst})    
        data_load = np.load(gen_dic['save_data_dir']+'Processed_data/Global/'+inst+'.npz',allow_pickle=True)
        data_dic[inst]=data_load['data_inst_add'].item()   
        gen_dic[inst]= data_load['gen_inst_add'].item() 

        #Remove visits
        #    - this allows removing visits after processing all of them once via gen_dic['calc_proc_data']
        for vis in gen_dic['unused_visits'][inst]:
            if vis in data_dic[inst]['visit_list']:data_dic[inst]['visit_list'].remove(vis)

        #Retrieve/initialize visit dictionaries
        for vis in data_dic[inst]['visit_list']:     
            data_load = np.load(gen_dic['save_data_dir']+'Processed_data/Global/'+inst+'_'+vis+'.npz',allow_pickle=True)
            data_dic[inst][vis]=data_load['data_add'].item()
            coord_dic[inst][vis]=data_load['coord_add'].item()
            data_prop[inst][vis]=data_load['data_prop_add'].item() 
            gen_dic[inst][vis]=data_load['gen_add'].item()
            data_dic['DI'][inst][vis]=data_load['DI_data_add'].item()
            theo_dic[inst][vis]={}
            data_dic['Atm'][inst][vis]={}  
            
    #Final processing
    if (not data_dic[inst]['comm_sp_tab']):print('         Visits do not share a common spectral table')      
    else:print('         All visits share a common spectral table')    
    for vis in data_dic[inst]['visit_list']:   
        data_vis = data_dic[inst][vis]
        coord_vis = coord_dic[inst][vis]
        gen_vis = gen_dic[inst][vis] 
        if (not data_vis['comm_sp_tab']):print('           Exposures in '+vis+' do not share a common spectral table')      
        else:print('           All exposures in '+vis+' share a common spectral table')   

        #Define CCF resolution, range, and sysvel here so that spectral data needs not be uploaded again after processing
        if gen_dic['CCF_from_sp'] or gen_dic['Intr_CCF']:
          
            #Velocity table in original frame
            if gen_dic['dRV'] is None:dvelccf= gen_dic['pix_size_v'][inst]
            else:dvelccf= gen_dic['dRV']  
            n_vel = int((gen_dic['end_RV']-gen_dic['start_RV'])/dvelccf)+1
            data_vis['nvel'] = n_vel
            data_vis['velccf'] = gen_dic['start_RV']+dvelccf*np.arange(n_vel)

            #Bin edges in velocity space
            data_vis['edge_velccf'] = np.append(data_vis['velccf']-0.5*dvelccf,data_vis['velccf'][-1]+0.5*dvelccf)

            #Shifting the CCFs velocity table by RV(CDM_star/CDM_sun)
            for key in ['velccf','edge_velccf']:data_vis[key+'_star']=data_vis[key]-data_dic['DI']['sysvel'][inst][vis]  
    
        #Keplerian motion relative to the stellar CDM and the Sun (km/s)
        for iexp in range(data_dic[inst][vis]['n_in_visit']):
            coord_vis['RV_star_stelCDM'][iexp],coord_vis['RV_star_solCDM'][iexp] = calc_orb_motion(coord_dic,inst,vis,system_param,gen_dic,coord_dic[inst][vis]['bjd'][iexp],coord_dic[inst][vis]['t_dur'][iexp],data_dic['DI']['sysvel'][inst][vis])

        #Using sky-corrected data for current visit
        if (inst in gen_dic['fibB_corr']) and (vis in gen_dic['fibB_corr'][inst]):print('          ',vis,'is sky-corrected') 

        #Indices of in/out exposures in global tables
        print('   > '+str(data_vis['n_in_visit'])+' exposures')  
        if ('in' in data_dic['DI']['idx_ecl']) and (inst in data_dic['DI']['idx_ecl']['in']) and (vis in data_dic['DI']['idx_ecl']['in'][inst]):
            for iexp in data_dic['DI']['idx_ecl']['in'][inst][vis]:
                for pl_loc in data_vis['transit_pl']:coord_vis[pl_loc]['ecl'][iexp] = 3*np.sign(coord_vis[pl_loc]['ecl'][iexp]) 
        if ('out' in data_dic['DI']['idx_ecl']) and (inst in data_dic['DI']['idx_ecl']['out']) and (vis in data_dic['DI']['idx_ecl']['out'][inst]):
            for iexp in data_dic['DI']['idx_ecl']['out'][inst][vis]:
                for pl_loc in data_vis['transit_pl']:coord_vis[pl_loc]['ecl'][iexp] = 1*np.sign(coord_vis[pl_loc]['ecl'][iexp]) 
        cond_in = np.zeros(data_vis['n_in_visit'],dtype=bool)
        cond_pre = np.ones(data_vis['n_in_visit'],dtype=bool) 
        cond_post = np.ones(data_vis['n_in_visit'],dtype=bool) 
        for pl_loc in data_vis['transit_pl']:
            cond_in|=(np.abs(coord_vis[pl_loc]['ecl'])>1) 
            cond_pre&=(coord_vis[pl_loc]['ecl']==-1.)
            cond_post&=(coord_vis[pl_loc]['ecl']==1.)
        gen_vis['idx_in']=np_where1D(cond_in)
        gen_vis['idx_out']=np_where1D(~cond_in)  
        gen_vis['idx_pretr']=np_where1D(cond_pre)  
        gen_vis['idx_posttr']=np_where1D(cond_post) 
        data_vis['n_in_tr'] = len(gen_vis['idx_in'])
        data_vis['n_out_tr'] = len(gen_vis['idx_out'])
        gen_vis['idx_exp2in'] = np.zeros(data_vis['n_in_visit'],dtype=int)-1
        gen_vis['idx_exp2in'][gen_vis['idx_in']]=np.arange(data_vis['n_in_tr'])
        gen_vis['idx_in2exp'] = np.arange(data_vis['n_in_visit'],dtype=int)[gen_vis['idx_in']]        
        
    #Total number of visits for current instrument
    gen_dic[inst]['n_visits'] = len(data_dic[inst]['visit_list'])
    data_dic[inst]['n_visits_inst']=len(data_dic[inst]['visit_list'])
 
    #Defining 1D table for conversion from 2D spectra
    #    - uniformly spaced in ln(w)
    #    - we impose d[ln(w)] = dw/w = ( w(i+1)-w(i) ) / w(i) 
    #      thus the table can be built with:
    # d[ln(w)]*w(i)  = ( w(i+1) - w(i) )  
    # w(i+1) = ( d[ln(w)] + 1 )*w(i) 
    # w(i+1) = ( d[ln(w)] + 1 )^2*w(i-1)
    # w(i) = ( d[ln(w)] + 1 )^i*w(0)
    #       the table ends at the largest n so that :
    # w(n) <= wmax < w(n+1)
    # ( d[ln(w)] + 1 )^n <= wmax/w(0) < ( d[ln(w)] + 1 )^(n+1)
    # n*ln( d[ln(w)] + 1 ) <= ln(wmax/w(0)) < (n+1)*ln( d[ln(w)] + 1 )
    # n <= ln(wmax/w(0))/ln( d[ln(w)] + 1 ) < (n+1)
    # n = E( ln(wmax/w(0))/ln( d[ln(w)] + 1 ) )
    #    - we do not set an automatic definition because tables for intrinsic and atmospheric profiles will be shifted compared to the original tables
    if gen_dic['spec_1D']:
        def def_1D_bins(prop_dic):
            spec_1D_prop = prop_dic['spec_1D_prop'][inst]  
            spec_1D_prop['nspec'] = int( np.floor(   np.log(spec_1D_prop['w_end']/spec_1D_prop['w_st'])/np.log( spec_1D_prop['dlnw'] + 1. ) )  ) 
            spec_1D_prop['edge_bins'] = spec_1D_prop['w_st']*( spec_1D_prop['dlnw'] + 1. )**np.arange(spec_1D_prop['nspec']+1)    
            spec_1D_prop['cen_bins'] = 0.5*(spec_1D_prop['edge_bins'][0:-1]+spec_1D_prop['edge_bins'][1::])      
            return None
        for key in ['DI','Intr','Atm']:
            if gen_dic['spec_1D_'+key]:def_1D_bins(data_dic[key])
        
    return None  








'''
Estimate of instrumental calibration 
    - see def_weights_spatiotemp_bin()
    - the calibration is used as weight or to scale back profiles approximatively to their raw photoelectron counts during CCF calculations (this allows artifically increasing errors when combining spectral regions with different SNRs)
      spectra must however be kept to their extracted flux units througout ANTARESS processing, as the stellar spectrum shifts over time in the instrument rest frame
      the same region of the spectrum thus sees different instrumental responses over time and is measured with different flux levels
      shifted stellar spectra with the same profile would thus have a different color balance after being converted in raw count units, preventing in particular the correct calculation of binned spectra 
    - weights are used for temporal binning, and are relevant only if the weight of a given pixel change over time, the calibration is thus not defined for CCFs
    - a median calibration profile over all visits of an instrument is calculated in spectral mode 
      we first fit a model to the calibration for each exposure, so that it can be extrapolated over a larger, common range for all visits 
      then we calculate the median of these extrapolated model, interpolate it as a function, and use it to define the common calibration profile over the specific table of each exposure
    - overall the estimated calibrations are stable between exposures but low count levels ten to yield larger calibrations - hence the independent calculation of calibration per exposure, before taking the median over the visit
      this may come from additional noise sources: if E = Ewhite + Ered = sqrt(gdet_true)*sqrt(F) + Er then gdet_true = (E-Er)^2/F < E^2/F = gdet_meas (since Er < 2*E), so that we underestimate the actual calibration. 
      in this case our assumptions to estimate gdet do not hold anymore, but for the purpose of weighing the exposures and scaling to raw count levels we still use the measured calibration
      since g_meas = (Ew + Er)^2/F this factor accounts for the true calibration but also for additional noise sources        
    - spectra are typically provided in the solar barycentric rest frame, and are thus not defined here in the detector rest frame, so that the calibration profiles in different epochs may be shifted by the Earth barycentric RV difference   
      we nonetheless use a single calibration profile, constant in time and common to all processed exposures of an instrument, so that the relative color balance between spectra is not modified when converting them back to counts 
      for the same reason the calibration must be applied uniformely (ie, in the same rest frame) to spectra in different exposures, and their master, so that it does not affect their combinations - even if the original calibration to flux units is applied in the input rest frame
'''
def calc_gcal(gen_dic,data_dic,inst,plot_dic,coord_dic):
    print('   > Estimating instrumental calibration')
    data_inst = data_dic[inst] 
        
    #Calculating data
    if gen_dic['calc_gcal']:
        print('         Calculating data')        
        cal_inputs_dic = {} 
        minmax_def = {}
        min_edge_ord_all = np.repeat(1e100,data_dic[inst]['nord'])
        max_edge_ord_all = np.repeat(-1e100,data_dic[inst]['nord'])   
        iexp_glob_groups_vis = {}
        for vis in data_dic[inst]['visit_list']:
            print('           Processing '+vis) 
            data_vis=data_dic[inst][vis]
            data_com_vis = np.load(data_vis['proc_com_data_paths']+'.npz',allow_pickle=True)['data'].item()
            data_gain_all={}

            #Estimate of instrumental calibration
            #    - set to the requested scaling if error tables are not available as input, or derived from sum(s[F]^2)/sum(F^2) summed over larger bins
            data_vis['cal_data_paths'] = gen_dic['save_data_dir']+'Processed_data/Calibration/'+inst+'_'+vis+'_'
            if data_vis['mock'] or (not gen_dic[inst][vis]['flag_err']):
                if data_vis['mock']:cst_gain = 1.
                elif (not gen_dic[inst][vis]['flag_err']):cst_gain = gen_dic['g_err'][inst]
                data_gain={'gdet_inputs' : {iord : {'par':None,'args':{'constant':cst_gain}} for iord in range(data_dic[inst]['nord'])}}
                n_glob_groups = data_vis['n_in_visit']
                iexp_glob_groups_vis[vis] = range(n_glob_groups) 
                for iexp in range(data_vis['n_in_visit']):data_gain_all[iexp] = data_gain                
    
            else:
                
                #Exposure groups
                iexp_gain_groups = list(range(i,min(i+gen_dic['gcal_binN'],data_vis['n_in_visit'])) for i in range(0,data_vis['n_in_visit'],gen_dic['gcal_binN']))
                n_glob_groups = len(iexp_gain_groups)
                iexp_glob_groups_vis[vis] = range(n_glob_groups)  
                gdet_val_all = np.zeros([data_vis['n_in_visit'],data_dic[inst]['nord']],dtype=object)
                minmax_def[vis] = np.zeros([data_vis['n_in_visit'],data_dic[inst]['nord'],2])*np.nan 
                data_all_temp = {}
                for iord in range(data_dic[inst]['nord']):                    
                    for iexp_glob,iexp_in_group in enumerate(iexp_gain_groups):
                   
                        #Process exposures in current group
                        idx_def_group = np.zeros(0,dtype=int)
                        low_wav_group = np.zeros(0,dtype=float)
                        high_wav_group = np.zeros(0,dtype=float)
                        wav_group = np.zeros(0,dtype=float)
                        flux_group = np.zeros(0,dtype=float)
                        var_group = np.zeros(0,dtype=float)
                        for iexp in iexp_in_group:
                            if iexp not in data_all_temp:data_all_temp[iexp] = np.load(data_vis['proc_DI_data_paths']+str(iexp)+'.npz',allow_pickle=True)['data'].item()
                      
                            #Defined bins
                            #    - we further exclude ranges that were found to display abnormal calibration estimates
                            cond_def_ord = data_all_temp[iexp]['cond_def'][iord]
                            if inst=='HARPN':
                                if data_inst['idx_ord_ref'][iord]==51:cond_def_ord[data_all_temp[iexp]['cen_bins'][iord]<5742.] = False
                                elif data_inst['idx_ord_ref'][iord]==64:cond_def_ord[(data_all_temp[iexp]['cen_bins'][iord]>6561.) & (data_all_temp[iexp]['cen_bins'][iord]<6564.)] = False    
                            elif inst=='HARPS': 
                                if data_inst['idx_ord_ref'][iord]==66:cond_def_ord[(data_all_temp[iexp]['cen_bins'][iord]>6558.) & (data_all_temp[iexp]['cen_bins'][iord]<6569.)] = False
                            idx_def_exp_ord = np_where1D(cond_def_ord)
                    
                            #Concatenate tables so that they are binned together
                            if len(idx_def_exp_ord)>0.:
                                idx_def_group = np.append(idx_def_group,idx_def_exp_ord)
                                low_wav_group = np.append(low_wav_group,data_all_temp[iexp]['edge_bins'][iord,0:-1])
                                high_wav_group = np.append(high_wav_group,data_all_temp[iexp]['edge_bins'][iord,1::] )
                                wav_group = np.append(wav_group,data_all_temp[iexp]['cen_bins'][iord])
                                flux_group = np.append(flux_group,data_all_temp[iexp]['flux'][iord])
                                var_group = np.append(var_group,data_all_temp[iexp]['cov'][iord][0])
               
                                #Save range over which profiles are defined
                                minmax_def[vis][iexp,iord,:]= [data_all_temp[iexp]['edge_bins'][iord,idx_def_exp_ord[0]],data_all_temp[iexp]['edge_bins'][iord,idx_def_exp_ord[-1]]]

                        #Initialize binned tables from grouped exposures
                        if np.sum(idx_def_group)>0:
                            bin_bd,raw_loc_dic = sub_def_bins(gen_dic['gcal_binw'],idx_def_group,low_wav_group,high_wav_group,high_wav_group-low_wav_group,wav_group,flux_group,var1D_loc=var_group)

                            #Adding progressively bins that will be used to fit the correction
                            bin_ord_dic={}
                            for key in ['gdet','cen_bins']:bin_ord_dic[key] = np.zeros(0,dtype=float) 
                            for ibin,(low_bin_loc,high_bin_loc) in enumerate(zip(bin_bd[0:-1],bin_bd[1:])):
                                bin_loc_dic,_ = sub_calc_bins(low_bin_loc,high_bin_loc,raw_loc_dic,0,calc_gdet=True)
                                if len(bin_loc_dic)>0:
                                    bin_ord_dic['gdet'] = np.append( bin_ord_dic['gdet'] , bin_loc_dic['gdet']*1e-3)
                                    bin_ord_dic['cen_bins'] = np.append( bin_ord_dic['cen_bins'] , bin_loc_dic['cen_bins'])
                            gdet_val_all[iexp_glob,iord] = bin_ord_dic
                        else:gdet_val_all[iexp_glob,iord] = None

                    ### End of exposure groups

                ### End of orders 

                #Initialize fit structure
                p_start = Parameters()           
                p_start.add_many(
                          ('a1',0., True   , None , None , None),
                          ('a2',0., True   , None , None , None),
                          ('a3',0., True   , None , None , None),
                          ('a4',0., True   , None , None , None),
                          ('b3',0., True   , None , None , None),
                          ('b4',0., True   , None , None , None),
                          ('c0',0., True   , None , None , None),
                          ('c1',0., True   , None , None , None),
                          ('c2',0., True   , None , None , None),
                          ('c3',0., True   , None , None , None),
                          ('c4',0., True   , None , None , None)) 
                for key in ['blue','mid','red']:
                    if (gen_dic['gcal_deg'][key]<2) or (gen_dic['gcal_deg'][key]>4):stop('Degrees must be between 2 and 4')
                for ideg in range(gen_dic['gcal_deg']['mid']+1,5):p_start['b'+str(ideg)].vary = False                             
                for ideg in range(gen_dic['gcal_deg']['blue']+1,5):p_start['a'+str(ideg)].vary = False                                   
                for ideg in range(gen_dic['gcal_deg']['red']+1,5):p_start['c'+str(ideg)].vary = False   
                nfree_gainfit =  gen_dic['gcal_deg']['blue']+gen_dic['gcal_deg']['red']+1+gen_dic['gcal_deg']['mid']-2                                     

                fixed_args={
                    'use_cov':False,
                    'deg_low':gen_dic['gcal_deg']['blue'],
                    'deg_mid':gen_dic['gcal_deg']['mid'],
                    'deg_high':gen_dic['gcal_deg']['red'],
                    'constant':None                                                
                    }                                           

                #Calibration for spectral profiles
                common_args = (minmax_def[vis],plot_dic,data_dic[inst]['nord'],gdet_val_all,inst,gen_dic['gcal_thresh'][inst],gen_dic['gcal_edges'],gen_dic['gcal_nooutedge'],fixed_args,nfree_gainfit,p_start,data_vis['cal_data_paths'])
                if gen_dic['gcal_nthreads']>1:data_gain_all = para_model_gain(model_gain,gen_dic['gcal_nthreads'],n_glob_groups,[iexp_glob_groups_vis[vis],iexp_gain_groups],common_args)                           
                else:data_gain_all = model_gain(iexp_glob_groups_vis[vis],iexp_gain_groups,*common_args)  

            #Processing all orders for the visit
            cal_inputs_dic[vis] = np.zeros([data_dic[inst]['nord'],n_glob_groups],dtype=object)
            for iord in range(data_dic[inst]['nord']): 

                #Widest spectral range over all visits   
                #    - defined in the input rest frame
                min_edge_ord_all[iord] = np.min([min_edge_ord_all[iord],data_com_vis['min_edge_ord'][iord]])
                max_edge_ord_all[iord] = np.max([max_edge_ord_all[iord],data_com_vis['max_edge_ord'][iord]])
                
                #Retrieve function inputs 
                #    - defined in the input rest frame
                for iexp_glob in iexp_glob_groups_vis[vis]:
                    cal_inputs_dic[vis][iord,iexp_glob] = data_gain_all[iexp_glob]['gdet_inputs'][iord] 

        #Median calibration profile over all exposures in the visit
        #    - we assume the median calibration is smooth enough that it can be captured with an interpolation function
        #    - we assume the spectral tables are always more resolved than the typical variations of the calibration profiles
        #    - the calibration profile is defined in the input rest frame, over a regular table in log oversampled compared to the instrument resolution (dlnw_inst = dw/w = 1/R)
        mean_gdet_func = {} 
        gain_grid_dlnw = 0.5/return_resolv(inst)
        for iord in range(data_dic[inst]['nord']): 

            #Table of definition of mean calibration profile over each order
            #    - over the widest range covered by the inst visits, and at the oversampled instrumental resolution
            #    - tables are uniformely spaced in ln(w)
            #      d[ln(w)] = sc*dw/w = sc*dv/c = sc/R             
            nspec_ord = 1+int( np.ceil(   np.log(max_edge_ord_all[iord]/min_edge_ord_all[iord])/np.log( gain_grid_dlnw + 1. ) )  ) 
            cen_bins_ord = min_edge_ord_all[iord]*( gain_grid_dlnw + 1. )**np.arange(nspec_ord)     
        
            #Median calibration over all exposures in the visit
            #    - interp1d is more stable at the edges than CubicSpline, and capture better the calibration variations than polynomials
            #    - the small variations in measured calibration between orders may result in two slices having different profiles
            #      however this is not an issue for the eventual weighing of the flux profiles
            med_gdet_allvis = np.zeros(nspec_ord,dtype=float)  
            for ivis,vis in enumerate(data_dic[inst]['visit_list']): 
                mean_gdet_ord = np.zeros([nspec_ord,0],dtype=float)*np.nan 
                for iexp_glob in iexp_glob_groups_vis[vis]:
                    mean_gdet_ord_loc = np.zeros(nspec_ord,dtype=float)*np.nan
                    mean_gdet_ord_loc=cal_piecewise_func(cal_inputs_dic[vis][iord,iexp_glob]['par'],cen_bins_ord,args=cal_inputs_dic[vis][iord,iexp_glob]['args'])      
                    mean_gdet_ord = np.append(mean_gdet_ord,mean_gdet_ord_loc[:,None],axis=1)
                med_gdet_allvis+=np.nanmedian(mean_gdet_ord,axis=1)     

            #Mean over all visits
            #    - we limit the calibration below the chosen global outlier threshold
            #    - we do not extrapolate beyond the range of definition of the median calibration profile to avoid spurious behaviour
            med_gdet_ord = 1e3*med_gdet_allvis/gen_dic[inst]['n_visits'] 
            med_gdet_ord[med_gdet_ord>gen_dic['gcal_thresh'][inst]['global']] = gen_dic['gcal_thresh'][inst]['global']
            med_gdet_ord[med_gdet_ord<=0.]=np.min(med_gdet_ord[med_gdet_ord>0.])
            mean_gdet_func[iord] = interp1d(cen_bins_ord,med_gdet_ord,bounds_error=False,fill_value=(med_gdet_ord[0],med_gdet_ord[-1]))  

        #Store mean calibration function
        np.savez_compressed(gen_dic['save_data_dir']+'Processed_data/Calibration/'+inst+'_mean_gdet',data = {'func':mean_gdet_func},allow_pickle=True)  

        #Define calibration tables for each exposure
        #    - the profile is the same, but defined over the table of the exposure
        #    - the path is made specific to a visit and a type of profile so that the calibration function can still be called in the multi-visit routines for any type of profile,
        # even after the type of profile has changed in a given visit
        for vis in data_inst['visit_list']: 
            data_vis=data_inst[vis]
            data_vis['mean_gdet_DI_data_paths'] = {}
            for iexp in range(data_vis['n_in_visit']):
                gain_exp = np.zeros(data_vis['dim_exp'],dtype=float)*np.nan
                data_exp = np.load(data_vis['proc_DI_data_paths']+str(iexp)+'.npz',allow_pickle=True)['data'].item()
                for iord in range(data_inst['nord']):
                    gain_exp[iord] = mean_gdet_func[iord](data_exp['cen_bins'][iord])
                data_vis['mean_gdet_DI_data_paths'][iexp] = data_vis['proc_DI_data_paths']+'mean_gdet_'+str(iexp)
                np.savez_compressed(data_vis['mean_gdet_DI_data_paths'][iexp], data = {'mean_gdet':gain_exp},allow_pickle=True) 
           
    else: 
        for vis in data_inst['visit_list']: 
            data_inst[vis]['cal_data_paths'] = gen_dic['save_data_dir']+'Processed_data/Calibration/'+inst+'_'+vis+'_'
            data_inst[vis]['mean_gdet_DI_data_paths'] = {}
            for iexp in range(data_inst[vis]['n_in_visit']):data_inst[vis]['mean_gdet_DI_data_paths'][iexp] = data_inst[vis]['proc_DI_data_paths']+'mean_gdet_'+str(iexp)
            check_data(data_inst[vis]['mean_gdet_DI_data_paths'],vis=vis)                    

    return None




'''
Calibration function fitting routines
'''
def model_gain(iexp_glob_groups,iexp_gain_groups,minmax_def,plot_dic,nord,gdet_val_all,inst,gcal_thresh,gcal_edges,gcal_nooutedge,fixed_args,nfree_gainfit,p_start,cal_data_paths):
    data_gain_all = {}
    for iexp_glob,iexp_gain_group in zip(iexp_glob_groups,iexp_gain_groups):
        data_gain={'gdet_inputs':{}}
        if (plot_dic['gcal']!='') or (plot_dic['gcal_ord']!=''):
            data_gain['wav_bin_all']=np.zeros(nord,dtype=object)
            data_gain['wav_trans_all']=np.zeros([2,nord],dtype=float)
            data_gain['cond_fit_all']=np.zeros(nord,dtype=object)
            data_gain['gdet_bin_all']=np.zeros(nord,dtype=object)  
        for iord in range(nord):
            
            #Process order if defined
            fit_done = False
            if gdet_val_all[iexp_glob,iord] is not None:
               
                #Fitting positive values below global threshold
                bin_ord_dic = gdet_val_all[iexp_glob,iord] 
                cond_fit = (bin_ord_dic['gdet']>0.) & (bin_ord_dic['gdet']<gcal_thresh['global']/1e3)
    
                #Remove extreme outliers
                med_prop = np.median(bin_ord_dic['gdet'][cond_fit])
                res = bin_ord_dic['gdet'][cond_fit] - med_prop
                cond_fit[cond_fit][np.abs(res) > 10.*stats.median_abs_deviation(res)] = False  
    
                #Generic fit properties
                #    - the fit does not use the covariance matrix, so fitted tables can be limited to non-consecutive pixels and 'idx_fit' covers all fitted pixels 
                low_edge = bin_ord_dic['cen_bins'][0]
                high_edge = bin_ord_dic['cen_bins'][-1]
                drange =  high_edge-low_edge    
                fixed_args.update({'w_lowedge':low_edge+gcal_edges['blue']*drange,
                                   'w_highedge':high_edge-gcal_edges['red']*drange,
                                   'wref':0.5*(low_edge+high_edge)})  
                
                #Fit binned calibration profile and define complete calibration profile
                #    - if enough bins are defined, otherwise a single measured value is used for the order
                if (np.sum(cond_fit)>nfree_gainfit):
    
                    #Imposing that gradient is negative (resp. positive) at the blue (resp. red) edges
                    p_start.add_many(('dPblue_wmin',-1., True   , None , 0. ))                                            
                    p_start.add_many(('wmin',low_edge-fixed_args['wref'], False)) 
                    p_start.add_many(('a1',0, False   , None , None , 'dPblue_wmin-(2*a2*wmin+3*a3*wmin**2.+4*a4*wmin**3.)')) 
                          
                    p_start.add_many(('dPred_wmax',1., True   , 0. , None))                                            
                    p_start.add_many(('wmax',high_edge-fixed_args['wref'], False)) 
                    p_start.add_many(('c1',0, False   , None , None , 'dPred_wmax-(2*c2*wmax+3*c3*wmax**2.+4*c4*wmax**3.)')) 
                                                          
                    #Temporary fit to identify outliers
                    #    - sigma-clipping applied to the inner parts of the order to prevent excluding edges where calibration can vary sharply
                    #    - negative values are removed
                    #    - weights are used to prevent point with large calibrations (associated with low fluxes) biasing the simple polynomial fit
                    cond_check = deepcopy(cond_fit) 
                    if inst in gcal_nooutedge:cond_check = cond_fit & (bin_ord_dic['cen_bins']>=bin_ord_dic['cen_bins'][0]+gcal_nooutedge[inst][0]) & (bin_ord_dic['cen_bins']<=bin_ord_dic['cen_bins'][-1]-gcal_nooutedge[inst][1]) 
                    if (np.sum(cond_check)>nfree_gainfit):
                        var_fit = bin_ord_dic['gdet'][cond_check]
                        fixed_args['idx_fit'] = np.ones(np.sum(cond_check),dtype=bool)
                        _,merit,_ = fit_minimization(ln_prob_func_lmfit,p_start,bin_ord_dic['cen_bins'][cond_check],bin_ord_dic['gdet'][cond_check],np.array([var_fit]),cal_piecewise_func,verbose=False,fixed_args=fixed_args)  
                        res_gdet = bin_ord_dic['gdet'][cond_check] - merit['fit']
                        cond_fit[cond_check] = np.abs(res_gdet)<=gcal_thresh['outliers']*np.std(res_gdet)
    
                        #Model fit
                        #    - errors are scaled with the reduced chi2 from the preliminary fit                           
                        if (np.sum(cond_fit)>nfree_gainfit):
                            fit_done = True
                            var_fit = bin_ord_dic['gdet'][cond_fit]*merit['chi2r'] 
                            fixed_args['idx_fit'] = np.ones(np.sum(cond_fit),dtype=bool)
                            _,merit,p_best = fit_minimization(ln_prob_func_lmfit,p_start,bin_ord_dic['cen_bins'][cond_fit],bin_ord_dic['gdet'][cond_fit],np.array([var_fit]),cal_piecewise_func,verbose=False,fixed_args=fixed_args)                                              
                            data_gain['gdet_inputs'][iord] = {'par':deepcopy(p_best),'args':deepcopy(fixed_args)}
    
                #Fit could not be performed
                if not fit_done:
                    data_gain['gdet_inputs'][iord]={'par':None,'args':{'constant':np.median(bin_ord_dic['gdet'])}}
    
                #Save
                if (plot_dic['gcal']!='') or (plot_dic['gcal_ord']!=''):
                    data_gain['wav_bin_all'][iord] = bin_ord_dic['cen_bins']
                    data_gain['wav_trans_all'][:,iord] = [fixed_args['w_lowedge'],fixed_args['w_highedge']]
                    data_gain['cond_fit_all'][iord] = cond_fit
                    data_gain['gdet_bin_all'][iord] = bin_ord_dic['gdet']
    
            #Order fully undefined
            #    - no calibration is applied
            else:
                data_gain['gdet_inputs'][iord]={'par':None,'args':{'constant':1.}}
    
        #Save calibration for each original exposure associated with current exposure group
        data_gain_all[iexp_glob]=data_gain 
        if (plot_dic['gcal']!='') or (plot_dic['gcal_ord']!=''):
            for iexp in iexp_gain_group:np.savez_compressed(cal_data_paths+str(iexp),data=data_gain,allow_pickle=True) 
            
    return data_gain_all

def para_model_gain(func_input,nthreads,n_elem,y_inputs,common_args): 
    pool_proc = Pool(processes=nthreads)   #cannot be passed through lmfit
    ind_chunk_list=init_parallel_func(nthreads,n_elem)
    chunked_args=[(y_inputs[0][ind_chunk[0]:ind_chunk[1]],y_inputs[1][ind_chunk[0]:ind_chunk[1]])+common_args for ind_chunk in ind_chunk_list]	
    all_results=tuple(tab for tab in pool_proc.starmap(func_input,chunked_args))

    data_gain_all = {}
    for data_gain in tuple(all_results[i] for i in range(nthreads)):data_gain_all.update(data_gain)
    y_output=data_gain_all
    
    pool_proc.close()
    pool_proc.join() 				
    return y_output



'''
Joined polynomials to model calibration functions
'''
def cal_piecewise_func(param_in,wav_in,args=None):
    if isinstance(param_in,lmfit.parameter.Parameters):params={par:param_in[par].value for par in param_in}
    else:params=deepcopy(param_in)                                                   
    #P_low(w) = sum(0:nlow,ai*w^i)
    #P_mid(w) = sum(0:nmid,bi*w^i)                                               
    #P_high(w) = sum(0:nhigh,ci*w^i)
    #   with w = wav-wav_ref
    #
    #Continuity in derivative:
    #P_low'(w1) = P_mid'(w1)  
    #   sum(1:nlow,i*ai*w1^i-1) = b1 + 2*b2*w1 + sum(3:nmid,i*bi*w1^i-1)
    #P_high'(w2) = P_mid'(w2)   
    #   sum(1:nhigh,i*ci*w2^i-1) = b1 + 2*b2*w2 + sum(3:nmid,i*bi*w2^i-1)
    #
    # 2*b2*(w1-w2) = sum(1:nlow,i*ai*w1^i-1) - sum(1:nhigh,i*ci*w2^i-1) + sum(3:nmid,i*bi*w2^i-1) - sum(3:nmid,i*bi*w1^i-1)
    # > b2 = (sum(1:nlow,i*ai*w1^i-1) - sum(1:nhigh,i*ci*w2^i-1) + sum(3:nmid,i*bi*w2^i-1) - sum(3:nmid,i*bi*w1^i-1))/(2*(w1-w2))      
    #      = (sum(0:nlow-1,(j+1)*a[j+1]*w1^j) - sum(0:nhigh-1,(j+1)*c[j+1]*w2^j) + sum(2:nmid-1,(j+1)*b[j+1]*w2^j) - sum(2:nmid-1,(j+1)*b[j+1]*w1^j)  )/(2*(w1-w2))     
    #
    # > b1 = sum(0:nlow-1,(j+1)*a[j+1]*w1^j) - sum(2:nmid,i*bi*w1^i-1)
    #      = sum(0:nlow-1,(j+1)*a[j+1]*w1^j) - sum(1:nmid-1,(j+1)*b[j+1]*w1^j) 
    #
    #Continuity in value:
    #P_low(w1) = P_mid(w1) 
    #   sum(0:nlow,ai*w1^i) = b0 + sum(1:nmid,bi*w1^i)
    #P_high(w2) = P_mid(w2)   
    #   sum(0:nhigh,ci*w2^i) = b0 + sum(1:nmid,bi*w2^i)
    #
    # a0 + sum(1:nlow,ai*w1^i) - sum(0:nhigh,ci*w2^i) = sum(1:nmid,bi*w1^i) - sum(1:nmid,bi*w2^i)  
    # > a0 = sum(1:nmid,bi*w1^i) - sum(1:nmid,bi*w2^i) - sum(1:nlow,ai*w1^i) + sum(0:nhigh,ci*w2^i)
    #
    # > b0 = sum(0:nlow,ai*w1^i) - sum(1:nmid,bi*w1^i)
    if args['constant'] is not None:
        model = np.repeat(args['constant'],len(wav_in))
    else:
        wav = wav_in-args['wref']
        w1 = args['w_lowedge']-args['wref']
        w2 = args['w_highedge']-args['wref'] 
    
        #Derivative continuity
        dPlow = Polynomial([(ideg+1)*params['a'+str(ideg+1)] for ideg in range(args['deg_low'])])
        dPhigh = Polynomial([(ideg+1)*params['c'+str(ideg+1)] for ideg in range(args['deg_high'])])
        dPmid_cut = Polynomial([0.,0.]+[(ideg+1)*params['b'+str(ideg+1)] for ideg in range(2,args['deg_mid'])])
        params['b2'] = (dPlow(w1) - dPhigh(w2) + dPmid_cut(w2) - dPmid_cut(w1))/(2.*(w1-w2)) 
        params['b1'] = dPlow(w1) - Polynomial([0.]+[(ideg+1)*params['b'+str(ideg+1)] for ideg in range(1,args['deg_mid'])])(w1)   

        #Value continuity        
        Phigh = Polynomial([params['c'+str(ideg)] for ideg in range(args['deg_high']+1)])   
        Pmid_cut = Polynomial([0.]+[params['b'+str(ideg)] for ideg in range(1,args['deg_mid']+1)])
        params['a0'] = Pmid_cut(w1) - Pmid_cut(w2) - Polynomial([0]+[params['a'+str(ideg)] for ideg in range(1,args['deg_low']+1)])(w1) + Phigh(w2)
        Plow = Polynomial([params['a'+str(ideg)] for ideg in range(args['deg_low']+1)])
        params['b0'] = Plow(w1) - Pmid_cut(w1)
    
        #Model
        model = np.zeros(len(wav),dtype=float)
        cond_mid = (wav>=w1) & (wav<=w2)
        model[cond_mid] = Polynomial([params['b'+str(ideg)] for ideg in range(args['deg_mid']+1)])(wav[cond_mid])                          
        cond_lowe=wav<w1
        model[cond_lowe] = Plow(wav[cond_lowe])     
        cond_highe=wav>w2
        model[cond_highe] = Phigh(wav[cond_highe])
        
    return model





'''
Definition of edge spectral table
'''
def def_edge_tab(cen_bins,dim = 2):
    if dim==0:
        mid_bins = 0.5*(cen_bins[0:-1]+cen_bins[1::]) 
        low_bins_st =cen_bins[0] - (mid_bins[0] - cen_bins[0])
        high_bins_end = cen_bins[-1] + (cen_bins[-1]-mid_bins[-1])  
        edge_bins =  np.concatenate(([low_bins_st], mid_bins,[high_bins_end]))        
    elif dim==2:
        mid_bins = 0.5*(cen_bins[:,:,0:-1]+cen_bins[:,:,1::]) 
        low_bins_st =cen_bins[:,:,0] - (mid_bins[:,:,0] - cen_bins[:,:,0])
        high_bins_end = cen_bins[:,:,-1] + (cen_bins[:,:,-1]-mid_bins[:,:,-1])  
        edge_bins =  np.concatenate((low_bins_st[:,:,None] , mid_bins,high_bins_end[:,:,None]),axis=2)        
    else:stop('Upgrade def_edge_tab()')                           
    return edge_bins
    





'''
Calculating CCFs from input spectra
'''
def CCF_from_spec(data_type_gen,inst,vis,data_dic,gen_dic,prop_dic):
    data_vis=data_dic[inst][vis]
    gen_vis=gen_dic[inst][vis]
    dir_save = {}
    iexp_conv,data_type_key,_ = init_conversion(data_type_gen,gen_dic,prop_dic,inst,vis,'CCFfromSpec',dir_save)

    #New paths
    #    - intrinsic and out-of-transit residual profiles are stored separately, contrary to global tables  
    flux_sc = False
    if data_type_gen in ['Intr','Atm']:
        dir_mast={}
        for gen in dir_save:dir_mast[gen] = {iexp_eff:dir_save[gen]+'ref_'+str(iexp_eff) for iexp_eff in data_vis['mast_'+gen+'_data_paths']}
        if gen_dic['flux_sc']:flux_sc = True
    proc_com_data_paths_new = gen_dic['save_data_dir']+'Processed_data/CCFfromSpec/'+inst+'_'+vis+'_com'

    #Calculating data
    if gen_dic['calc_'+data_type_gen+'_CCF']:
        print('         Calculating data')
        if data_vis['type']=='CCF':stop('Data must be spectral')         

        #Orders that are used to compute the CCFs
        ord_coadd = gen_dic[inst]['orders4ccf'] 
        nord_coadd = len(ord_coadd)

        #Calibration profile
        if data_vis['mean_gdet']:
            data_com = dataload_npz(data_vis['proc_com_data_paths'])
            mean_gdet_com = np.zeros([nord_coadd,data_com['dim_exp'][1]],dtype=float)
        else:gdet_ord=None
    
        #Upload data from all exposures 
        if flux_sc:data_scaling_all={}
        data_proc={}
        n_exp = len(iexp_conv) 
        for iexp_sub,iexp in enumerate(iexp_conv):
            gen = deepcopy(data_type_gen)
            iexp_eff = deepcopy(iexp)     #Effective index (relative to global or in-transit tables)
            if (gen=='Intr'):
                if (iexp in gen_vis['idx_in']):iexp_eff = gen_vis['idx_exp2in'][iexp]    
                else:gen = 'Res'
            data_exp = dataload_npz(data_vis['proc_'+gen+'_data_paths']+str(iexp_eff))
         
            #Check that planetary ranges were not yet excluded
            if (data_type_gen=='Intr') and (iexp in gen_vis['idx_in']) and ('Intr' in data_dic['Atm']['no_plrange']) and (iexp in data_dic['Atm'][inst][vis]['iexp_no_plrange']) and data_exp['plrange_exc']:
                stop('    Planetary ranges excluded too soon: re-run gen_dic["intr_data"] with gen_dic["Intr_CCF"]')
                
            #Upload data
            if iexp_sub==0:
                if flux_sc:data_proc['cen_bins'] = np.zeros([n_exp,nord_coadd,data_vis['nspec']],dtype=float)
                data_proc['edge_bins'] = np.zeros([n_exp,nord_coadd,data_vis['nspec']+1],dtype=float)
                data_proc['flux'] = np.zeros([n_exp,nord_coadd,data_vis['nspec']],dtype=float)
                data_proc['cov'] = np.zeros([n_exp,nord_coadd],dtype=object)
                data_proc['cond_def'] = np.zeros([n_exp,nord_coadd,data_vis['nspec']],dtype=bool)             
            for key in ['edge_bins','flux','cov','cond_def']:
                data_proc[key][iexp_sub] = data_exp[key][ord_coadd]  

            #Upload flux scaling
            #    - we compute the equivalent CCF of the broadband scaling for the propagation of broadband spectral scaling on disk-integrated profiles into weights
            #    - global flux scaling is not modified  
            if flux_sc:
                data_proc['cen_bins'][iexp_sub] = data_exp['cen_bins'][ord_coadd]          
                data_scaling_all[iexp]=dataload_npz(data_vis['scaled_'+gen+'_data_paths']+str(iexp_eff)) 

            #Mean calibration profile over processed exposures
            #    - due to the various shifts of the processed spectra from the input rest frame, calibration profiles are not equivalent for a given line between exposure
            #      to maintain the relative flux balance between lines when computing CCFs, we calculate a common calibration profile to all processes exposures
            if data_vis['mean_gdet']:
                mean_gdet_exp = dataload_npz(data_vis['mean_gdet_'+gen+'_data_paths'][iexp_eff])['mean_gdet'] 
                for isub_ord,iord in enumerate(ord_coadd):
                    mean_gdet_com_ord=bind.resampling(data_com['edge_bins'][iord], data_proc['edge_bins'][iexp_sub,isub_ord],mean_gdet_exp[iord], kind=gen_dic['resamp_mode'])/n_exp 
                    idx_def_loc = np_where1D(~np.isnan(mean_gdet_com_ord))
                    mean_gdet_com_ord[0:idx_def_loc[0]+1]=mean_gdet_com_ord[idx_def_loc[0]]
                    mean_gdet_com_ord[idx_def_loc[-1]:]=mean_gdet_com_ord[idx_def_loc[-1]]
                    mean_gdet_com[isub_ord]+=mean_gdet_com_ord
                    
            #Upload weighing disk-integrated master 
            #    - the master always remains either defined on the common table, or on a specific table different from the table of its associated exposure
            #    - the master is computed after DI spectra have been converted into CCFs, and thus need conversion only for later profile types
            if data_type_gen in ['Intr','Atm']:
                data_ref = dataload_npz(data_vis['mast_'+gen+'_data_paths'][iexp_eff])
                if iexp_sub==0:
                    nspec_mast = (data_ref['cen_bins'].shape)[1]
                    data_proc['edge_bins_ref'] = np.zeros([n_exp,nord_coadd,nspec_mast+1],dtype=float)
                    data_proc['flux_ref'] = np.zeros([n_exp,nord_coadd,nspec_mast],dtype=float)
                    data_proc['cov_ref'] = np.zeros([n_exp,nord_coadd],dtype=object)                
                for key in ['edge_bins','flux','cov']:
                    data_proc[key+'_ref'][iexp_sub] = data_ref[key][ord_coadd]               

        #Initialize CCF tables
        CCF_all = np.zeros([n_exp,1,data_vis['nvel']],dtype=float)
        cov_exp_ord = np.zeros([n_exp,nord_coadd],dtype=object)
        nd_cov_exp_ord=np.zeros([n_exp,nord_coadd],dtype=int) 
        if data_type_gen in ['DI','Intr']:
            CCF_mask_wav = gen_dic['CCF_mask_wav'][inst]
            CCF_mask_wgt = gen_dic['CCF_mask_wgt'][inst]
        elif data_type_gen=='Atm':
            CCF_mask_wav = data_dic['Atm']['CCF_mask_wav']
            CCF_mask_wgt = data_dic['Atm']['CCF_mask_wgt']
        if data_type_gen in ['Intr','Atm']:
            CCF_ref = np.zeros([n_exp,1,data_vis['nvel']],dtype=float)
            cov_ref_ord = np.zeros([n_exp,nord_coadd],dtype=object)            
            nd_cov_ref_ord=np.zeros([n_exp,nord_coadd],dtype=int)
        
        #Velocity tables and initializations
        #    - we create an artificial order in velocity table to keep the same structure as spectra
        if data_type_gen=='DI':   
            velccf = data_vis['velccf']
            edge_velccf = data_vis['edge_velccf']          
        elif data_type_gen in ['Intr','Atm']:
            velccf = data_vis['velccf_star']
            edge_velccf = data_vis['edge_velccf_star']
        cen_bins = np.tile(velccf,[1,1])
        edge_bins = np.tile(edge_velccf,[1,1])  
        
        #Flux scaling
        if flux_sc: 
            norm_loc_flux_scaling_CCF = np.zeros(n_exp,dtype=float)                  
            loc_flux_scaling_CCF = np.zeros([n_exp,data_vis['nvel']],dtype=float)   
            
        #Calculate CCF over requested orders in each exposure
        #    - the covariance of CCFs calculated over different orders may have different dimensions, thus we first store them independently before they can be co-added
        #    - the structure below works for both s1d and e2ds
        ord_coadd_eff = []
        for isub,iord in enumerate(ord_coadd):
 
            #Identify lines that can contribute to all exposures for current order
            idx_maskL_kept = check_CCF_mask_lines(n_exp,data_proc['edge_bins'][:,isub],data_proc['cond_def'][:,isub],CCF_mask_wav,edge_velccf)

            #Calculating CCF for current order in each exposure with contributing lines
            #    - parallelisation is disabled, as it is inefficient given the size of the tables to process
            if len(idx_maskL_kept)>0:
                ord_coadd_eff+=[isub]
                if data_vis['mean_gdet']:
                    gdet_ord = bind.resampling(data_proc['edge_bins'][iexp_sub,isub],data_com['edge_bins'][iord],mean_gdet_com[isub], kind=gen_dic['resamp_mode'])  
                    idx_def_loc = np_where1D(~np.isnan(gdet_ord))
                    gdet_ord[0:idx_def_loc[0]+1]=gdet_ord[idx_def_loc[0]]
                    gdet_ord[idx_def_loc[-1]:]=gdet_ord[idx_def_loc[-1]]
                for iexp_sub,iexp in enumerate(iexp_conv):                      
                    flux_ord,cov_ord = new_compute_CCF(data_proc['edge_bins'][iexp_sub,isub],data_proc['flux'][iexp_sub,isub],data_proc['cov'][iexp_sub,isub],gen_dic['resamp_mode'],edge_velccf,CCF_mask_wgt[idx_maskL_kept],CCF_mask_wav[idx_maskL_kept],1.,cal = gdet_ord)[0:2]
                    CCF_all[iexp_sub,0]+=flux_ord
                    cov_exp_ord[iexp_sub,isub] = cov_ord 
                    nd_cov_exp_ord[iexp_sub,isub] = np.shape(cov_ord)[0]

                    #Compute CCF of spectral scaling
                    #    - loc_flux_scaling = 1 - LC
                    #      loc_flux_scaling_CCF = X - CCF_LC
                    #       we thus calculate the CCF of 1 to get X and normalize the scaling profile
                    if flux_sc and (not data_scaling_all[iexp]['null_loc_flux_scaling']):                
                        loc_flux_scaling_CCF_exp,_,norm_loc_flux_scaling_CCF_exp = new_compute_CCF(data_proc['edge_bins'][iexp_sub,isub],data_scaling_all[iexp]['loc_flux_scaling'](data_proc['cen_bins'][iexp_sub,isub]),None,gen_dic['resamp_mode'],edge_velccf,CCF_mask_wgt[idx_maskL_kept],CCF_mask_wav[idx_maskL_kept],1.,cal = gdet_ord)
                        loc_flux_scaling_CCF[iexp_sub] +=loc_flux_scaling_CCF_exp
                        norm_loc_flux_scaling_CCF[iexp_sub] += norm_loc_flux_scaling_CCF_exp

                    #Computing CCF of master disk-integrated spectrum
                    #    - so that it can be used in the weighing profiles
                    if data_type_gen in ['Intr','Atm']:
                        flux_temp,cov_temp = new_compute_CCF(data_proc['edge_bins_ref'][iexp_sub,isub],data_proc['flux_ref'][iexp_sub,isub],data_proc['cov_ref'][iexp_sub,isub],gen_dic['resamp_mode'],edge_velccf,CCF_mask_wgt[idx_maskL_kept],CCF_mask_wav[idx_maskL_kept],1.,cal = gdet_ord)[0:2]
                        CCF_ref[iexp_sub,0]+=flux_temp
                        cov_ref_ord[iexp_sub,isub] = cov_temp 
                        nd_cov_ref_ord[iexp_sub,isub] = np.shape(cov_temp)[0]

        #Check
        if len(ord_coadd_eff)==0:stop('         No lines contribute to all exposures')

        #Computing final covariance matrix in artificial new order
        for iexp_sub,iexp in enumerate(iexp_conv):
            gen = deepcopy(data_type_gen)
            iexp_eff = deepcopy(iexp)
            cond_def_exp = (~np.isnan(CCF_all[iexp_sub,0]))[None,:] 
            data_CCF_exp={}
            if (data_type_gen=='Intr'):
                if (iexp in gen_vis['idx_in']):
                    iexp_eff = gen_vis['idx_exp2in'][iexp] 
                    
                    #Set to nan planetary ranges in intrinsic CCFs
                    #    - this is not done for disk-integrated and residual CCFs, as planetary signals need to be kept in them to be later extracted (planetary ranges are temporarily excluded when analyzing CCFs from those profiles)
                    if ('Intr' in data_dic['Atm']['no_plrange']) and (iexp in data_dic['Atm'][inst][vis]['iexp_no_plrange']):
                        cond_in_pl = ~( np.ones(data_vis['nvel'],dtype=bool) & excl_plrange(cond_def_exp[0],data_dic['Atm'][inst][vis]['exclu_range_star'],iexp,edge_bins[0],'CCF')[0])
                        CCF_all[iexp_sub,0,cond_in_pl]=np.nan
                        cond_def_exp[0,cond_in_pl]=False  
                        data_CCF_exp['plrange_exc'] = True
                    else:data_CCF_exp['plrange_exc'] = False
                        
                else:gen = 'Res'

            #Maximum dimension of covariance matrix in current exposure from all contributing orders
            nd_cov_exp = np.amax(nd_cov_exp_ord[iexp_sub,:])
            if data_type_gen in ['Intr','Atm']:nd_cov_exp = np.max([nd_cov_exp,np.amax(nd_cov_ref_ord[iexp_sub,:])])  
           
            #Co-adding contributions from orders
            cov_exp = np.zeros(1,dtype=object)
            cov_exp[0] = np.zeros([nd_cov_exp,data_vis['nvel']])
            if data_type_gen in ['Intr','Atm']:
                cov_ref = np.zeros(1,dtype=object)
                cov_ref[0] = np.zeros([nd_cov_exp,data_vis['nvel']])
            for isub in ord_coadd_eff:
                cov_exp[0][0:nd_cov_exp_ord[iexp_sub,isub],:] +=  cov_exp_ord[iexp_sub,isub]   
                if data_type_gen in ['Intr','Atm']:cov_ref[0][0:nd_cov_ref_ord[iexp_sub,isub],:] +=  cov_ref_ord[iexp_sub,isub]   
      
            #Saving data for each exposure
            #    - CCF are stored independently of input spectra, so that both can be retrieved
            data_CCF_exp.update({'cen_bins':cen_bins,'edge_bins':edge_bins,'flux':CCF_all[iexp_sub],'cond_def':cond_def_exp,'cov':cov_exp,'nd_cov':nd_cov_exp})              
            datasave_npz(dir_save[gen]+str(iexp_eff),data_CCF_exp)
            
            #Processing disk-integrated masters
            if data_type_gen in ['Intr','Atm']:
                datasave_npz(dir_mast[gen][iexp_eff],{'cen_bins':cen_bins,'edge_bins':edge_bins,'flux':CCF_ref[iexp_sub],'cov':cov_ref})

            #Redefine spectral scaling table
            if flux_sc:
                if (not data_scaling_all[iexp]['null_loc_flux_scaling']):loc_flux_scaling_CCF[iexp_sub,:]/=norm_loc_flux_scaling_CCF[iexp_sub]
                if not data_scaling_all[iexp]['chrom']:loc_flux_scaling_exp = np.poly1d(np.mean(loc_flux_scaling_CCF[iexp_sub]))                
                else:loc_flux_scaling_exp = interp1d(cen_bins[0],loc_flux_scaling_CCF[iexp_sub],fill_value=(loc_flux_scaling_CCF[iexp_sub,0],loc_flux_scaling_CCF[iexp_sub,-1]), bounds_error=False)
                data_scaling_all[iexp]['loc_flux_scaling'] = loc_flux_scaling_exp
                data_scaling_all[iexp]['chrom'] = False
                datasave_npz(dir_save[gen]+'_scaling_'+str(iexp),data_scaling_all[iexp])

        #Update common tables
        #    - set to the table in the star rest frame, as it is the one that will be used in later operations                
        datasave_npz(proc_com_data_paths_new,{'dim_exp':[1,data_vis['nvel']],'nspec':data_vis['nvel'],'cen_bins':np.tile(data_vis['velccf_star'],[1,1]),'edge_bins':np.tile(data_vis['edge_velccf_star'],[1,1])})

    else:
        check_data({'path':proc_com_data_paths_new})

    #Updating path to processed data and checking it has been calculated
    data_vis['proc_com_data_paths'] = proc_com_data_paths_new
    for gen in dir_save:
        data_vis['proc_'+gen+'_data_paths'] = dir_save[gen]  
        if flux_sc:data_vis['scaled_'+gen+'_data_paths'] = dir_save[gen]+'_scaling_'
        if gen in ['Intr','Atm']:data_vis['mast_'+gen+'_data_paths'] = dir_mast[gen]

    #Convert spectral mode 
    #    - all operations afterwards will be performed on CCFs
    #    - tellurics are not propagated to calculate weights in CCF mode 
    #    - no spectral calibration is applied
    print('         ANTARESS switched to CCF processing')
    data_vis['comm_sp_tab']=True
    data_vis['tell_sp'] = False 
    data_vis['mean_gdet'] = False 
    data_vis['type']='CCF'
    data_vis['nspec'] = data_vis['nvel']
    data_dic[inst]['nord'] = 1
    data_vis['dim_all'] = [data_vis['n_in_visit'],data_dic[inst]['nord'],data_vis['nvel']]
    data_vis['dim_exp'] = [data_dic[inst]['nord'],data_vis['nvel']]
    data_vis['dim_ord'] = [data_vis['n_in_visit'],data_vis['nvel']]
    if ('chrom' in data_dic['DI']['system_prop']):
        data_dic['DI']['system_prop']['chrom_mode'] = 'achrom'
        data_dic['DI']['system_prop'].pop('chrom')

    return None    
    
 
'''
Wrap-up for conversion of out-of-transit residual and intrinsic spectra into CCFs
'''
def ResIntr_CCF_from_spec(inst,vis,data_dic,gen_dic):
    data_inst = data_dic[inst]
    data_vis = data_inst[vis]
    gen_vis = gen_dic[inst][vis]

    #Calculating CCFs
    #    - if there is atmospheric contamination, excluding them from spectra before the conversion of intrinsic profiles would create empty ranges that shift between exposures
    # due to the planet orbital motion, thus potentially excluding mask lines from some exposures and not others, and resulting in CCFs that are not equivalent between all exposures
    #      the exclusion is thus applied after the conversion, internally to the routine 
    CCF_from_spec('Intr',inst,vis,data_dic,gen_dic,data_dic['Intr'])

    #Continuum pixels over all exposures
    #    - exclusion of planetary ranges is not required for intrinsic profiles if already applied to their definition, and if not already applied contamination is either negligible or neglected  
    cond_def_cont_all  = np.zeros(data_inst[vis]['dim_ord'],dtype=bool)
    for i_in,iexp in zip(gen_vis['idx_exp2in'],range(data_vis['n_in_visit'])): 
        if i_in ==-1:
            gen = 'Res'
            iexp_eff = iexp
        else:
            gen='Intr'
            iexp_eff = i_in
        data_exp = dataload_npz(data_vis['proc_'+gen+'_data_paths']+str(iexp_eff))

        #Continuum ranges
        #    - if planetary contamination is excluded from residual out-of-transit profiles, define a large enough initial continuum range if the planet has a wide velocimetric motion          
        if len(data_dic[gen]['cont_range'][inst])==0:cond_def_cont_all[iexp,:] = True
        else:
            for bd_int in data_dic[gen]['cont_range'][inst]:
                cond_def_cont_all[iexp] |= (data_exp['edge_bins'][0,0:-1]>=bd_int[0]) & (data_exp['edge_bins'][0,1:]<=bd_int[1])        
            cond_def_cont_all[iexp] &= data_exp['cond_def'][0]
        if (gen=='Res') and ('Res_prof' in data_dic['Atm']['no_plrange']) and (iexp in data_dic['Atm'][inst][vis]['iexp_no_plrange']):     
            cond_def_cont_all[iexp] &= excl_plrange(cond_def_cont_all[iexp],data_dic['Atm'][inst][vis]['exclu_range_star'],iexp,data_exp['edge_bins'][0],'CCF')[0]

    #Definition of continuum pixels and errors
    if data_dic['Intr']['disp_err']:print('         Setting errors on intrinsic CCFs to continuum dispersion')
    for i_in,iexp in zip(gen_vis['idx_exp2in'],range(data_vis['n_in_visit'])): 
        if i_in ==-1:
            gen = 'Res'
            iexp_eff = iexp
        else:
            gen='Intr'
            iexp_eff = i_in
        data_exp = dataload_npz(data_vis['proc_'+gen+'_data_paths']+str(iexp_eff))
    
        #Definition of errors on CCFs based on the dispersion in their continuum
        #    - attributing constant error to all CCFs points, if requested
        #    - if atmospheric profiles are extracted this operation must be done on residual CCFs, and not intrinsic ones, so that errors can then be propagated         
        if data_dic['Intr']['disp_err']: 

            #Continuum dispersion
            disp_cont=data_exp['flux'][0,cond_def_cont_all[iexp]].std() 

            #Error table
            #    - scaled if requested
            err_tab = np.sqrt(gen_dic['g_err'][inst])*np.repeat(disp_cont,data_vis['nspec'])
            data_exp['cov'][0] = (err_tab*err_tab)[None,:]

            #Overwrite exposure data
            np.savez_compressed(data_vis['proc_'+gen+'_data_paths']+str(iexp_eff),data=data_exp,allow_pickle=True)  

    #Updating/correcting continuum level          
    data_dic['Intr'][inst][vis]['mean_cont']=calc_Intr_mean_cont(data_vis['n_in_tr'],data_dic[inst]['nord'],data_vis['nspec'],data_vis['proc_Intr_data_paths'],data_vis['type'],data_dic['Intr']['cont_range'],inst,data_dic['Intr']['cont_norm'])

    #Determine the correlation length for the visit
    if gen_dic['scr_search']:
        corr_length_determination(data_dic['Res'][inst][vis],data_vis,gen_dic[inst][vis]['scr_search'],inst,vis,gen_dic)    

    return None







'''
Identify mask lines that should be used to compute CCFs
    - we keep lines for which the CCF RV table (once converted into wavelength space) 
 + is fully within the spectral range of the input spectrum
 + contains no nan from the input spectrum  
    - the check is made for the time-series of spectra given as input as a whole, as CCFs should be calculated from the same mask lines in all exposures 
    - tables must have dimension [n_time x n_spec] as input
'''
def check_CCF_mask_lines(n_exp,edge_wav_all,cond_def_all,wav_mask,edge_velccf):

    #Conversion from wavelengths in star rest rame (lines transitions) to wavelength in the spectrum rest frame
    #    - what we convert is the radial velocity table of the CCF (the edges of its bins)
    #      the new table has dimension n_lines*(n_RV+1)
    #      we later loop on each line, so that the sub-table is centered on the expected line transition, assuming the mask is centered on each successive pixel
    #    - nu_received/nu_source = (1 + v_receiver/c )/(1 - v_source/c )
    #  w_source/w_received   = (1 + v_receiver/c )/(1 - v_source/c )
    #  with v_receiver > 0 if receiver moving toward the source
    #  with v_source > 0   if source moving toward the receiver
    #    - with the receiver at rest in the frame where the spectrum was measured:
    #  v_receiver = 0.
    #  v_source = - rv_star_spframe since by definition rv < 0 when moving toward us
    #  -> wav_rstar/w_spframe = 1./(1 + rv_star_spframe/c )
    #  -> w_spframe = wav_rstar*(1 + rv_star_spframe/c ) 
    #    - dimension n_libes x n_RV
    edge_mask_lines=wav_mask[:,None]*(1.+(edge_velccf/c_light))

    #First start wavelength and last end wavelength of the RV table relative to each mask line
    #    - dimension n_lines
    wstart_mask_table = edge_mask_lines[:,0]
    wstop_mask_table = edge_mask_lines[:,-1]    
    
    #Cheking all input exposures
    n_lines = len(wav_mask)
    idx_lines_kept = np.arange(n_lines)
    for iexp in range(n_exp):

        #Check for discontinuous input table
        low_pix_wav = edge_wav_all[iexp,0:-1]
        high_pix_wav = edge_wav_all[iexp,1::]

        #Identifying the first and last original pixels overlapping with the RV table of each line in wavelength space
        #    - we exploit searchsorted(x,y) which returns i_k in x for each y[k] so that 
        # x_low[i_k-1]  <= y_low[k]  <  x_low[i_k]  (side = right)
        # x_high[i_k-1] <  y_high[k] <= x_high[i_k] (side = left)
        #    - with x the lower boundary of original pixels and y the lower boundary of new bins, we get i_k-1 the index of the first original pixel overlapping with the new bin
        #      with x the upper boundary of original pixels and y the upper boundary of new bins, we get i_k the index of the last original pixel overlapping with the new bin
        #    - only defined pixels must be given to "searchsorted"
        #    - y values beyond the first & last values in x returns indexes -1 or n
        #    - because the old and new bins must be continuous here we could use the edge tables
        #      however this would make searchsorted find the position of all new bins in the mask table, while here we only need to find the start and end pixel
        idx_first_overpix =np.searchsorted(low_pix_wav,wstart_mask_table,side='right')-1
        idx_last_overpix =np.searchsorted(high_pix_wav,wstop_mask_table,side='left')        
        
        #Keep lines whose range is fully within that of the input spectrum
        #    - condition can be invalid for lines at the edges of the spectral range and wide RV tables
        idx_keepL = np_where1D( (idx_first_overpix>=0) & (idx_last_overpix<=len(low_pix_wav)-1) )
    
        #Keep lines whose range does not contain nan values from the input spectrum
        cond_keepL_sub = np.ones(len(idx_keepL),dtype=bool)
        for iline,(idx_first_overpix_line,idx_last_overpix_line) in enumerate(zip(idx_first_overpix[idx_keepL],idx_last_overpix[idx_keepL])):
            cond_keepL_sub[iline] &= (np.sum(~cond_def_all[iexp,idx_first_overpix_line:idx_last_overpix_line+1]) == 0)
    
        #Updates indexes of lines to be kept
        idx_keepL=idx_keepL[cond_keepL_sub]           #indexes relative to reduced table
        idx_lines_kept = idx_lines_kept[idx_keepL]    #indexes relative to reduced table but containting original line indexes in mask
        
        #Updates mask tables to speed the check
        wstart_mask_table = wstart_mask_table[idx_keepL]
        wstop_mask_table = wstop_mask_table[idx_keepL]

    return idx_lines_kept





'''
Compute the Cross Correlation Function with covariance
'''
def new_compute_CCF(edge_wav,flux,cov,resamp_mode,edge_velccf,wght_mask,wav_mask,nthreads,cal = None):

    #Check for discontinuous input table
    low_pix_wav = edge_wav[0:-1]
    high_pix_wav = edge_wav[1::]
    if np.any(low_pix_wav[1:]-high_pix_wav[0:-1]):
        stop('Spectral bins must be continuous')
    
    #Line wavelength at each requested RV
    #    - for each line, the rest wavelength of its transition is shifted to all trial wavelengths of the spectrum rest frame associated with the CCF RVs
    n_RV=len(edge_velccf)-1
    edge_mask_lines=wav_mask[:,None]*spec_dopshift(-edge_velccf)

    #Call to parallelized function   
    if (nthreads>1) and (nthreads<=len(wght_mask)):
        pool_proc = Pool(processes=nthreads)                
        common_args=(n_RV,edge_wav,flux,cov,cal,resamp_mode)
        chunkable_args=[edge_mask_lines,wght_mask]
        fluxCCF,covCCF_line,nd_covCCF_line,contCCF=parallel_new_compute_CCF(pool_proc,sub_new_compute_CCF,nthreads,len(wght_mask),chunkable_args,common_args)                           
        pool_proc.close()
        pool_proc.join()
        
    #Regular routine
    else:
        fluxCCF,covCCF_line,nd_covCCF_line,contCCF=sub_new_compute_CCF(edge_mask_lines,wght_mask,n_RV,edge_wav,flux,cov,cal,resamp_mode)

    #Computing final covariance matrix
    #    - maximum dimension of covariance matrix from all contributing orders
    if cov is not None:
        nd_covCCF = np.amax(nd_covCCF_line) 
        covCCF = np.zeros([nd_covCCF,n_RV],dtype=float)
        for isub,wght_mask_line in enumerate(wght_mask):
            covCCF[0:nd_covCCF_line[isub],:] +=  covCCF_line[isub]         
    else:
        covCCF = None

    return fluxCCF,covCCF,contCCF


def sub_new_compute_CCF(edge_mask,wght_mask,n_RV,edge_wav,flux,cov,cal,resamp_mode):
  
    #Loop on selected lines
    #    - for each line in the line list, we co-add the contribution to the CCF
    #    - the bins have constant spectral width (which is the same for all lines in RV space, but changes with the line in wavelength space)
    nL_kept = len(wght_mask)
    fluxCCF = np.zeros(n_RV,dtype=float)
    contCCF = 0.
    covCCF_line = np.zeros(nL_kept,dtype=object)
    nd_covCCF_line = np.zeros(nL_kept,dtype=int)
    for isub,(edge_mask_line,wght_mask_line) in enumerate(zip(edge_mask,wght_mask)):

        #Spectrum around current line brought back from extracted to raw count units
        #    - if a calibration profile is provided as input we take its mean value in the local line range and use it to scale the input spectrum
        #      this is to get the spectrum as close as possible to its original count level, so that regions of the spectrum with comparable flux levels but different count levels do not contribute in the same way to the CCF
        #      the use of a constant estimated calibration rather than the actual profile is to keep the color balance of the spectrum intact, and avoid biasing the CCF
        if cal is not None:
            idxCCF_sub = np_where1D((edge_wav>=edge_mask_line[0]) & (edge_wav<=edge_mask_line[-1]))    #indexes where spectrum falls within mask line range
            idxCCF_sub_max = min([len(edge_wav)-1,idxCCF_sub[-1]])                                     
            mean_gainCCF_sub = np.mean(cal[idxCCF_sub[0]:idxCCF_sub_max+1])
        else:mean_gainCCF_sub =  1.

        #Spectrum around current line resampled on the CCF table
        if cov is None:
            fluxCCF_sub = bind.resampling(edge_mask_line,edge_wav, flux, kind=resamp_mode)/mean_gainCCF_sub                        
        else:
            fluxCCF_sub,covCCF_sub = bind.resampling(edge_mask_line,edge_wav, flux/mean_gainCCF_sub , cov = cov/mean_gainCCF_sub**2., kind=resamp_mode)   
            covCCF_line[isub]=(wght_mask_line**2.)*covCCF_sub
            nd_covCCF_line[isub] = np.shape(covCCF_line[isub])[0]                      

        #Add the weighted contribution of current line to the CCF
        fluxCCF += wght_mask_line*fluxCCF_sub
        contCCF += wght_mask_line/mean_gainCCF_sub

    return fluxCCF,covCCF_line,nd_covCCF_line,contCCF


def parallel_new_compute_CCF(pool_proc,func_input,nthreads,n_elem,y_inputs,common_args):
    
    #Indexes of chunks to be processed by each core       
    ind_chunk_list=init_parallel_func(nthreads,n_elem)

    #2 arrays with dimensions n_lines x n and n_lines
    chunked_args=[(y_inputs[0][ind_chunk[0]:ind_chunk[1],:],y_inputs[1][ind_chunk[0]:ind_chunk[1]])+common_args for ind_chunk in ind_chunk_list]				
         
    #------------------------------------------------------------------------------------	     					
    #Return the results from all cores as elements of a tuple
    #    - arguments could be given whole to map(), and be automatically divided using an input 'chunksize', however for long arrays it takes 
    # less time to do the chunking before
    all_results=tuple(tab for tab in pool_proc.starmap(func_input,chunked_args))

    #------------------------------------------------------------------------------------	 			
    #Outputs:
    #    - flux table co-added between processors	
    #    - covariance matrix and dimension table are appended between processors
    fluxCCF = np.sum(tuple(all_results[i][0] for i in range(nthreads)),axis=0)
    covCCF_line = np.concatenate(tuple(all_results[i][1] for i in range(nthreads)))
    nd_covCCF_line = np.concatenate(tuple(all_results[i][2] for i in range(nthreads)))
    contCCF = np.sum(tuple(all_results[i][3] for i in range(nthreads)))

    return fluxCCF,covCCF_line,nd_covCCF_line,contCCF














    












'''
Initialisation of dataset for a given visit
'''
def init_visit(data_prop,data_dic,vis,coord_dic,inst,system_param,gen_dic):
    
    #Reset data mode to input
    data_dic[inst]['nord'] = deepcopy(data_dic[inst]['nord_spec'])    
    
    #Dictionaries initialization
    data_vis=data_dic[inst][vis]
    coord_vis = coord_dic[inst][vis] 
    data_dic['Res'][inst][vis]={}
    gen_vis = gen_dic[inst][vis] 
    data_dic['Intr'][inst][vis]={}    

    #Current rest frame
    data_dic['DI'][inst][vis]['rest_frame'] = 'input'    
    
    #Indices of in/out exposures in global tables
    print('   > '+str(data_vis['n_in_visit'])+' exposures')  
    print('        ',data_vis['n_in_tr'],'in-transit')
    print('        ',data_vis['n_out_tr'],'out-of-transit ('+str(len(gen_vis['idx_pretr']))+' pre / '+str(len(gen_vis['idx_posttr']))+' post)')
    data_vis['dim_in'] = [data_vis['n_in_tr']]+data_vis['dim_exp']

    #Definition of CCF continuum, and ranges contributing to master out
    #    - the continuum of a given CCFs account for the ranges selected as input, the planetary exclusion ranges, and the bins defined in the CCFs
    #    - the final continuum is common to all CCFs
    #      if at least one bin is undefined in one exposure it is undefined in the continuum
    plAtm_vis = data_dic['Atm'][inst][vis]    

    #Generic definition of planetary range exclusion
    plAtm_vis['exclu_range_input']={'CCF':{},'spec':{}}
    plAtm_vis['exclu_range_star']={'CCF':{},'spec':{}}
    if (data_dic['Atm']['exc_plrange']):

        #Exposures to be processed
        #    - ranges are excluded from in-transit data only if left undefined as input, since most of the time only in-transit absorption signals are analyzed
        #      beware if an emission or absorption signal is present outside of the transit, in this case out-of-transit should be included manually
        if (inst in data_dic['Atm']['iexp_no_plrange']) and (vis in data_dic['Atm']['iexp_no_plrange'][inst]):plAtm_vis['iexp_no_plrange'] = data_dic['Atm']['iexp_no_plrange'][inst][vis]
        else:plAtm_vis['iexp_no_plrange'] = gen_vis['idx_in']
        iexp_no_plrange = plAtm_vis['iexp_no_plrange']
        
        #Planetary range in star rest frame
        #    - the excluded range 'plrange' is defined in the planet rest frame, ie as RV(plrange/pl)
        #    - raw profiles are in their original sun barycentric rest frame, and thus defined over RV(M/star) + RV(star/CDM_sun)  
        plrange_star={}
        for pl_loc in data_vis['transit_pl']:
            plrange_star[pl_loc] = np.vstack((np.repeat(1e10,data_vis['n_in_visit']),np.repeat(-1e10,data_vis['n_in_visit'])))
            plrange_star[pl_loc][:,iexp_no_plrange] = data_dic['Atm']['plrange'][:,None] + coord_vis[pl_loc]['rv_pl'][iexp_no_plrange] 
            if (True in np.isnan(coord_vis[pl_loc]['rv_pl'])):stop('  Run gen_dic["calc_proc_data"] again to calculate "rv_pl"')
            
            #Planet exclusion range
            #    - we define the lower/upper wavelength boundaries of excluded ranges for each planetary mask line in each requested exposure
            #    - table has dimension (nline,2,nexp) in spectral mode, (2,nexp) in RV mode, 
            #    - if input data is in spectral mode, ranges are defined in both spectral and RV space so that they are available in case of CCF conversion
            #      if input data is in CCF mode, ranges are only defined in RV   
            plAtm_vis['exclu_range_star']['CCF'][pl_loc] = plrange_star[pl_loc]
            plAtm_vis['exclu_range_input']['CCF'][pl_loc] =  plrange_star[pl_loc] + coord_vis['RV_star_solCDM'][None,:]
            if ('spec' in data_dic[inst]['type']):

                plAtm_vis['exclu_range_star']['spec'][pl_loc] = np.zeros([len(data_dic['Atm']['CCF_mask_wav']),2,data_vis['n_in_visit']])*np.nan
                plAtm_vis['exclu_range_star']['spec'][pl_loc][:,:,iexp_no_plrange] = data_dic['Atm']['CCF_mask_wav'][:,None,None]*spec_dopshift(-plrange_star[pl_loc][:,iexp_no_plrange])  

                plAtm_vis['exclu_range_input']['spec'][pl_loc] = np.zeros([len(data_dic['Atm']['CCF_mask_wav']),2,data_vis['n_in_visit']])*np.nan
                plAtm_vis['exclu_range_input']['spec'][pl_loc][:,:,iexp_no_plrange] = data_dic['Atm']['CCF_mask_wav'][:,None,None]*spec_dopshift(- (plrange_star[pl_loc][:,iexp_no_plrange]+coord_vis['RV_star_solCDM'][iexp_no_plrange]))  
 
    return None




'''
Updating instrument settings once all visits have been processed
    - common instrument table is set to the table of the visit taken as reference
'''
def update_data_inst(data_dic,inst,gen_dic):
    data_inst = data_dic[inst]
    data_com_ref = dataload_npz(data_inst[data_inst['com_vis']]['proc_com_data_paths'])      
    if gen_dic['spec_1D']:
        data_inst['type']='spec1D' 
        data_inst['mean_gdet'] = False     
        data_inst['proc_com_data_path'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_com' 
    elif gen_dic['CCF_from_sp']:
        data_inst['type']='CCF'                 
        data_inst['tell_sp'] = False 
        data_inst['mean_gdet'] = False     
        data_inst['proc_com_data_path'] = gen_dic['save_data_dir']+'Processed_data/CCFfromSpec/'+inst+'_com' 
    else:
        data_inst['proc_com_data_path'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_com' 
    np.savez_compressed(data_inst['proc_com_data_path'],data = data_com_ref,allow_pickle=True)  
    data_inst['dim_exp'] = deepcopy(data_com_ref['dim_exp'])
    data_inst['nspec'] = deepcopy(data_com_ref['nspec'])
    data_inst['nord'] = 1
    data_inst['comm_sp_tab']=True
        
    return None





'''
Sub-routine to identify spectral pixels contaminated by the planet
    - spectra are defined over RV(M/star) 
      the excluded range is defined in the planet rest frame, ie as RV(plrange/pl) 
      to correspond to the spectra we shift this range as :
    RV(plrange/pl) + RV(pl/star) + RV(CDM_star/CDM_sun)
  = RV(plrange/star) 
    - the range is excluded for each line in the planet atmosphere mask
      in its own rest frame an atom absorbs the light at nu_received = nu0 
      If we consider an absorbing atom in the planet as the source, then the light observed in the stellar rest frame is:
      w_0/w_star   = (1 + v_star/c )/(1 - v_pl/c )                                
           with v_star =0 as we place ourselves in the star rest frame
           with v_pl > 0   if the planet is moving toward the star, ie
                v_pl = rv(pl/star) since rv(pl/star) < 0 when moving toward us (away from the star)         
           thus :   
      w_star = w_0 / (1 - rv(pl/star)/c )             
    - we use the more precise relativistic formula:
      w_star = w_0 * sqrt(1 + rv(pl/star)/c )/sqrt(1 - rv(pl/star)/c ) 
    - since we consider a range of velocities in the planet rest frame, it is RV(plrange/star) instead of RV(pl/star) 
'''
def excl_plrange(cond_def,range_star_in,iexp,edge_bins,data_type):
    cond_kept = np.ones(cond_def.shape,dtype=bool)
    idx_excl_bd_ranges = []
    
    if data_type=='CCF':
        range_star = range_star_in['CCF']
        for pl_loc in range_star:
            idx_excl = np_where1D((edge_bins[0:-1]>=range_star[pl_loc][0,iexp]) & (edge_bins[1:]<=range_star[pl_loc][1,iexp]))
            if len(idx_excl)>0:
                idx_excl_bd_ranges+=[[idx_excl[0],idx_excl[-1]]]
                cond_kept[idx_excl] = False 

    elif 'spec' in data_type:
        range_star = range_star_in['spec']
        
        #Defined bins in spectrum
        #    - we do not check pixels already undefined to calibration time
        idx_def_exp_pl = np_where1D(cond_def)                             
        n_idx_def_loc = len(idx_def_exp_pl)
        
        #Process lines that overlap at least partially with spectrum range
        #    - we use the spectral table of the target exposure, since current exposure has either been resampled over it or share the same table 
        for pl_loc in range_star:
            cond_keepL = (range_star[pl_loc][:,1,iexp]>=edge_bins[idx_def_exp_pl[0]]) &  (range_star[pl_loc][:,0,iexp]<=edge_bins[idx_def_exp_pl[-1]+1])
            if np.sum(cond_keepL)>0:

                #Indexes of first and last defined bins in the exposure table that overlap with the line range
                #    - see method in resample_func()
                idx_first_overpix =np.searchsorted(edge_bins[idx_def_exp_pl],range_star[pl_loc][cond_keepL,0,iexp],side='right')-1
                idx_last_overpix =np.searchsorted(edge_bins[idx_def_exp_pl+1],range_star[pl_loc][cond_keepL,1,iexp],side='left')
                idx_first_overpix[idx_first_overpix==-1]=0        
                idx_last_overpix[idx_last_overpix==n_idx_def_loc]=n_idx_def_loc-1         
        
                #Exclude progressively the velocity range relative to each line 
                #    - bins are relative to idx_def_exp_pl
                for idx_first_overpix_loc,idx_last_overpix_loc in zip(idx_first_overpix,idx_last_overpix):
                    idx_undef_sub = np.arange(idx_first_overpix_loc,idx_last_overpix_loc+1)
                    idx_undef = idx_def_exp_pl[idx_undef_sub]
                    idx_excl_bd_ranges+=[[idx_undef[0],idx_undef[-1]]]
                    cond_kept[idx_undef] = False
        
    return cond_kept,idx_excl_bd_ranges












'''
Routine for correcting disk-integrated profiles for abnormal variations
    - variations of the contrast, FWHM, and centroid of disk-integrated CCFs are to be fitted within the plot_dic['prop_raw'] and ana_propCCFraw sub-routine
    - we obtain better results by correcting first for the constrast, before the RV
      FWHM must be corrected after RVs, so that line profiles are aligned
    - the routine is called before 'fit_DI' so that corrected profiles can then be re-analyzed
    - the routine requires that aligned profiles have already been calculated, as the RV model has been fitted to profiles corrected for the star motion
      after correction the routine resets the path of current (and aligned) disk-integrated data to corrected profiles, which will not go through 'align_data' again 
    - for modulations the coefficient of degree 0 is not required, as models are normalized to the mean
'''
def detrend_prof_gen(corr_prop_in,var,corr_in,mode):
    corr_out = deepcopy(corr_in)
    
    #Modulated variation
    if mode=='modul':
        if ('sin' in corr_prop_in):
            loc_corr = (1. + corr_prop_in['sin'][0]*np.sin(2*np.pi*((var-corr_prop_in['sin'][1])/corr_prop_in['sin'][2])))
            corr_out*=loc_corr
        if ('pol' in corr_prop_in) and (len(corr_prop_in['pol'])>0):
            loc_corr = 1.
            for ideg in range(1,len(corr_prop_in['pol'])+1):loc_corr+=corr_prop_in['pol'][ideg-1]*(var**ideg)
            corr_out*=loc_corr

    #Cumulative variation
    elif mode=='add':
        if ('sin' in corr_prop_in):
            loc_corr = corr_prop_in['sin'][0]*np.sin(2*np.pi*((var-corr_prop_in['sin'][1])/corr_prop_in['sin'][2]))
            corr_out+=loc_corr
        if ('pol' in corr_prop_in) and (len(corr_prop_in['pol'])>0):
            loc_corr = 0.
            for ideg in range(1,len(corr_prop_in['pol'])+1):loc_corr+=corr_prop_in['pol'][ideg-1]*(var**ideg)
            corr_out+=loc_corr
        
    return corr_out

def detrend_prof(detrend_prof_dic,data_dic,coord_dic,inst,vis,CCF_dic,data_prop,gen_dic,plot_dic):
    cond_trend = (detrend_prof_dic['corr_trend'] and (inst in detrend_prof_dic['prop']) and (vis in detrend_prof_dic['prop'][inst]))
    cond_PC = (detrend_prof_dic['corr_PC'] and (inst in detrend_prof_dic['PC_model']) and (vis in detrend_prof_dic['PC_model'][inst]))

    #Specific instrumental correction
    if (inst=='EXPRES') and (gen_dic['star_name']=='55Cnc'):cond_custom = True
    else:cond_custom = False

    if cond_trend or cond_PC or cond_custom:
        data_vis=data_dic[inst][vis]
        print('   > Correcting disk-integrated line profiles')
        
        #Calculating aligned data
        if gen_dic['calc_detrend_prof']:
            print('         Calculating data')
            data_prop_vis=data_prop[inst][vis]
            pl_loc = data_vis['transit_pl'][0]
            
            #Correct spectrum
            if ('spec' in data_dic[inst][vis]['type']):
                
                #Single line, processed in RV space
                if detrend_prof_dic['line_trans'] is not None:
                    single_l = True
                    iord_line = detrend_prof_dic['iord_line']
                    nord_eff = 1
                    dim_exp_eff=[nord_eff,data_dic[inst][vis]['nspec']]  
                    
                #Correct full spectrum
                else:
                    single_l = False
                    iord_line = None
                    nord_eff = data_dic[inst]['nord']
                    dim_exp_eff = data_dic[inst][vis]['dim_exp']
                 
            #Correct CCF
            else:
                single_l = True
                iord_line = 0
                nord_eff = data_dic[inst]['nord']
                dim_exp_eff = data_dic[inst][vis]['dim_exp']
                
            #------------------------------------------
            #Trend corrections
            if cond_trend:
                print('           Correcting for trends')
                corr_prop = detrend_prof_dic['prop'][inst][vis]
            
                #Default orders for SNR
                if (inst in detrend_prof_dic['SNRorders']):SNRorders_inst = detrend_prof_dic['SNRorders'][inst]
                else:SNRorders_inst = {'HARPS':[49],'HARPN':[46],'ESPRESSO_MR':[39],'ESPRESSO':[102,103],'CARMENES_VIS':[40],
                                       'NIRPS_HA':[57],'NIRPS_HE':[57],    #H band, 1.63 mic, order not affected by tellurics thus stable for SNR measurement
                                       'EXPRES':[14]}[inst]                #562 nm

                #Initialize corrections
                #    - FWHM corrections can only be applied to single lines
                corr_list = list(corr_prop.keys())
                prop_corr = [corr_loc.split('_')[0] for corr_loc in corr_list]
                var_corr = [corr_loc.split('_')[1] for corr_loc in corr_list]
             
                if np.any([corr_loc=='RV' for corr_loc in prop_corr]):corr_RV=True
                else:corr_RV=False
                if np.any([corr_loc=='ctrst' for corr_loc in prop_corr]):
                    corr_ctrst=True
                    cond_def_cont_all  = np.zeros(data_vis['dim_ord'],dtype=bool)
                else:corr_ctrst=False            
                if np.any([corr_loc=='FWHM' for corr_loc in prop_corr]) and single_l:corr_FWHM=True
                else:corr_FWHM=False
                
                #Defining corrections
                glob_corr_ctrst = np.ones(data_vis['n_in_visit'],dtype=float)
                glob_corr_FWHM = np.ones(data_vis['n_in_visit'],dtype=float)
                glob_corr_RV = np.zeros(data_vis['n_in_visit'],dtype=float)
                for iexp in range(data_vis['n_in_visit']):
    
                    #SNR in chosen spectral orders
                    #    - indexes relative to original orders
                    if np.any(['snr' in corr_loc for corr_loc in var_corr]):
                        SNR_exp_ord = data_prop_vis['SNRs'][iexp] 
                        SNR_exp =np.mean(SNR_exp_ord[SNRorders_inst])
                        if np.any([corr_loc=='snrQ' for corr_loc in var_corr]):SNRQ_exp =np.sqrt(np.sum(SNR_exp_ord[SNRorders_inst]**2.))                 
                        
                    #Orbital phase
                    if np.any([corr_loc=='phase' for corr_loc in var_corr]):
                        cen_ph_exp=coord_dic[inst][vis][pl_loc]['cen_ph'][iexp] 
                 
                    #Airmass and activity indexes
                    env_prop = {}
                    if np.any([corr_loc=='AM' for corr_loc in var_corr]):
                        env_prop['AM']=data_prop_vis['AM'][iexp]     
                    for key in data_vis['act_idx']:
                        if np.any([corr_loc==key for corr_loc in var_corr]):
                            env_prop[key]=data_prop_vis[key][iexp,0]               
                
                    #----------------------------------------------------------------------------------------                    
                    #Polynomial corrections for contrast variation
                    #    - only the modulation around the constant value is corrected for
                    #---------------------------------------------------------------------------------------- 
                    if corr_ctrst:            
                        
                        #Upload latest processed DI data
                        data_exp = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp))
  
                        #Continuum for single line
                        if single_l:
                          
                            #Initializing continuum ranges in the input rest frame to defined pixels over requested ranges 
                            for bd_int in data_dic['DI']['cont_range'][inst]:
                                cond_def_cont_all[iexp] |= (data_exp['edge_bins'][0,0:-1]>=bd_int[0]) & (data_exp['edge_bins'][0,1:]<=bd_int[1])    
                            cond_def_cont_all[iexp] &= data_exp['cond_def'][0]
                            
                            #Exclusion of planetary ranges
                            if ('DI_prof' in data_dic['Atm']['no_plrange']) and (iexp in data_dic['Atm'][inst][vis]['iexp_no_plrange']):  
                                cond_def_cont_all[iexp] &= excl_plrange(data_exp['cond_def'][0],data_dic['Atm'][inst][vis]['exclu_range_input'],iexp,data_exp['edge_bins'][0],data_dic[inst][vis]['type'])[0]
    
                        #------------------------
    
                        #With phase   
                        if 'ctrst_phase' in corr_list:glob_corr_ctrst[iexp] = detrend_prof_gen(corr_prop['ctrst_phase'],cen_ph_exp,glob_corr_ctrst[iexp],'modul')
                        
                        #With SNR           
                        if 'ctrst_snr' in corr_list:glob_corr_ctrst[iexp] = detrend_prof_gen(corr_prop['ctrst_snr'],SNR_exp,glob_corr_ctrst[iexp],'modul')
                        elif 'ctrst_snrQ' in corr_list:glob_corr_ctrst[iexp] = detrend_prof_gen(corr_prop['ctrst_snrQ'],SNRQ_exp,glob_corr_ctrst[iexp],'modul')
    
                        #With airmass and indexes
                        for key in env_prop:
                            if 'ctrst_'+key in corr_list:glob_corr_ctrst[iexp] = detrend_prof_gen(corr_prop['ctrst_'+key],env_prop[key],glob_corr_ctrst[iexp],'modul')
                            
                    #----------------------------------------------------------------------------------------                    
                    #Polynomial corrections for FWHM variation
                    #    - only the modulation around the constant value is corrected for
                    #---------------------------------------------------------------------------------------- 
                    if corr_FWHM:                   
    
                        #With phase   
                        if 'FWHM_phase' in corr_list:glob_corr_FWHM[iexp] = detrend_prof_gen(corr_prop['FWHM_phase'],cen_ph_exp,glob_corr_FWHM[iexp],'modul')
                        
                        #With SNR           
                        if 'FWHM_snr' in corr_list:glob_corr_FWHM[iexp] = detrend_prof_gen(corr_prop['FWHM_snr'],SNR_exp,glob_corr_FWHM[iexp],'modul')
                        elif 'FWHM_snrQ' in corr_list:glob_corr_FWHM[iexp] = detrend_prof_gen(corr_prop['FWHM_snrQ'],SNRQ_exp,glob_corr_FWHM[iexp],'modul')
                        
                        #With airmass and indexes
                        for key in env_prop:
                            if 'FWHM_'+key in corr_list:glob_corr_FWHM[iexp] = detrend_prof_gen(corr_prop['FWHM_'+key],env_prop[key],glob_corr_FWHM[iexp],'modul')
    
                    #----------------------------------------------------------------------------------------                 
                    #Polynomial correction for deviation to Keplerian RV curve (km/s)
                    #    - the full polynomial variation is corrected for
                    #----------------------------------------------------------------------------------------       
                    if corr_RV:    
                       
                        #With phase
                        if 'RV_phase' in corr_list:
                            glob_corr_RV[iexp] = detrend_prof_gen(corr_prop['RV_phase'],cen_ph_exp,glob_corr_RV[iexp],'add')
                            
                        #With SNR
                        if 'RV_snr' in corr_list:
                            glob_corr_RV[iexp] = detrend_prof_gen(corr_prop['RV_snr'],SNR_exp,glob_corr_RV[iexp],'add')
    
                        elif 'RV_snrQ' in corr_list:
                            glob_corr_RV[iexp] = detrend_prof_gen(corr_prop['RV_snrQ'],SNRQ_exp,glob_corr_RV[iexp],'add')
    
                        #With airmass and indexes
                        for key in env_prop:
                            if 'RV_'+key in corr_list:glob_corr_RV[iexp] = detrend_prof_gen(corr_prop['RV_'+key],env_prop[key],glob_corr_RV[iexp],'add')
    
                #Normalizing around the mean
                glob_corr_ctrst/=np.mean(glob_corr_ctrst)                    
                glob_corr_FWHM/=np.mean(glob_corr_FWHM)                     
               
                #Continuum for contrast corrections
                if corr_ctrst:
                
                    #Continuum common to all input single line profiles 
                    if single_l:
                        cond_cont_com  = np.all(cond_def_cont_all,axis=0)
                        if np.sum(cond_cont_com)==0.:stop('No pixels in common continuum')
                        cont_func_dic= None
                    
                    #Continuum of stellar spectrum
                    else:
                        cond_cont_com=None
                        cont_func_dic = dataload_npz(gen_dic['save_data_dir']+'Stellar_cont_DI/'+inst+'_'+vis+'/')['cont_func_dic']

            #------------------------------------------
            #Custom correction
            if cond_custom:
                print('           Applying custom correction')
                cust_corr_RV = np.zeros(data_vis['n_in_visit'],dtype=float)
                corr_data = ((pd.read_csv('/Users/bourrier/Travaux/Exoplanet_systems/Divers/55_cancri/RM/RMR/Data/EXPRES/220728_55CnceTransit_detrend.csv')).values).T
                if vis=='20220131':JD_corr = corr_data[1]-2400000. 
                if vis=='20220406':JD_corr = corr_data[1]-2400000. 
                RV_corr = corr_data[2]*1e-3   
                if vis=='20220406':RV_corr*=-1. 
                for iexp in range(data_vis['n_in_visit']):
                    icorr = closest(JD_corr,coord_dic[inst][vis]['bjd'][iexp])
                    cust_corr_RV[iexp]=RV_corr[icorr]
         
            #------------------------------------------
            #PC corrections
            if cond_PC:
                print('           Correcting for PC')
                
                #PCA results
                pca_results = np.load(detrend_prof_dic['PC_model'][inst][vis]['all'],allow_pickle=True)['data'].item() 

                #Joint intrinsic fit
                if 'in' in detrend_prof_dic['PC_model'][inst][vis]:
                    jointfit_results = np.load(detrend_prof_dic['PC_model'][inst][vis]['in'],allow_pickle=True)['data'].item() 
                    if pca_results['n_pc']!=jointfit_results['n_pc'][inst][vis]:stop('Number of fitted PC must match')
                else:jointfit_results=None
                
                #PC profiles fitted to the residual and intrinsic data
                eig_res_matr = pca_results['eig_res_matr'][0:pca_results['n_pc']]
                
                #PC profiles selected to generate the model
                if (inst in detrend_prof_dic['idx_PC']) and (vis in detrend_prof_dic['idx_PC'][inst]):idx_pc = detrend_prof_dic['idx_PC'][inst][vis]
                else:idx_pc = range(pca_results['n_pc'])
                
            #----------------------------------------------------------------------------------------               

            #Resample aligned profiles on the common visit table if relevant
            if (data_vis['comm_sp_tab']):
                data_com = dataload_npz(data_vis['proc_com_data_paths'])
                cen_bins_resamp, edge_bins_resamp = data_com['cen_bins'],data_com['edge_bins']
            else:cen_bins_resamp, edge_bins_resamp = None,None 

            #Correct each exposure  
            proc_DI_data_paths_new = gen_dic['save_data_dir']+'Detrend_prof/'+inst+'_'+vis+'_'                          
            for iexp in range(data_vis['n_in_visit']):
        
                #Upload latest processed DI data
                data_exp = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp))                  
 
                #Switch spectrum into RV space
                #    - upon radial velocity table associated with chosen transition
                if ('spec' in data_dic[inst][vis]['type']) and (detrend_prof_dic['line_trans'] is not None):
                    data_corr = {'flux' : np.array([data_exp['flux'][iord_line]]),
                                 'cov' : np.array([data_exp['cov'][iord_line]]),
                                 'edge_bins' : np.array([c_light*( (data_exp['edge_bins'][iord_line]/detrend_prof_dic['line_trans']) - 1.)]),
                                 'cen_bins' : np.array([c_light*( (data_exp['cen_bins'][iord_line]/detrend_prof_dic['line_trans']) - 1.)])}
 
                #Correct CCF or full spectrum
                else:data_corr = data_exp
                 
                #---------------------------------  
                #Trend corrections
                if cond_trend:

                    #Correcting for RV variation    
                    if corr_RV:
                        data_corr = align_data(data_corr,data_dic[inst][vis]['type'],nord_eff,dim_exp_eff,gen_dic['resamp_mode'],cen_bins_resamp, edge_bins_resamp,glob_corr_RV[iexp])

                    #Correcting for contrast variation 
                    if corr_ctrst:                 
                        data_corr = detrend_prof_ctrst(data_corr,single_l,nord_eff,data_dic[inst][vis]['nspec'],glob_corr_ctrst[iexp],cond_cont_com,cont_func_dic,coord_dic[inst][vis]['RV_star_solCDM'][iexp],
                                                         gen_dic['save_data_dir']+'Scaled_data/'+inst+'_'+vis+'_scaling_'+str(iexp),coord_dic[inst][vis]['t_dur'][iexp])

                    #Correcting for FWHM variation 
                    #    - for CCF and single spectral lines 
                    if corr_FWHM and single_l:                 
                        glob_corr=np.repeat(glob_corr_FWHM[iexp],data_vis['nspec']+1)
                        data_corr = detrend_prof_FWHM(data_corr,glob_corr,coord_dic[inst][vis]['RV_star_solCDM'][iexp],gen_dic,comm_sp_tab=data_vis['comm_sp_tab'])

                #---------------------------------  
                #Custom correction
                if cond_custom:
                    data_corr['cen_bins'][0] -= cust_corr_RV[iexp]
                    data_corr['edge_bins'][0] -= cust_corr_RV[iexp]
      
                #---------------------------------      
                #PC correction
                #    - see PCA module for details
                if cond_PC: 
                    i_in = gen_dic[inst][vis]['idx_exp2in'][iexp]
                    
                    #Correct exposure if included in PCA module or joint intrinsic fit
                    if (iexp in pca_results['idx_corr']) or ((jointfit_results is not None) and (i_in in jointfit_results['idx_in_fit'][inst][vis])):
        
                        #Upload flux scaling 
                        data_scaling = dataload_npz(gen_dic['save_data_dir']+'Scaled_data/'+inst+'_'+vis+'_scaling_'+str(iexp))

                        #Switch PCA spectrum into RV space
                        if ('spec' in data_dic[inst][vis]['type']) and (detrend_prof_dic['line_trans'] is not None):
                            bins_edge_PCA = c_light*( (pca_results['edge_bins'][0]/detrend_prof_dic['line_trans']) - 1.)  
                        else:bins_edge_PCA = pca_results['edge_bins'][0]                           

                        #Residual PC model from intrinsic profile fit
                        pc_mod_exp = np.zeros(data_vis['nspec'],dtype=float)
                        if ((jointfit_results is not None) and (i_in in jointfit_results['idx_in_fit'][inst][vis])):
                     
                            #Intrinsic PC model
                            for i_pc in idx_pc:pc_mod_exp+=jointfit_results['p_final']['aPC_idxin'+str(i_in)+'_ord'+str(i_pc)+'__IS'+inst+'_VS'+vis]*eig_res_matr[i_pc]
                            
                            #Scaling to level of residual data 
                            #    - see PCA module, the intrinsic PC model corresponds to :
                            # Pfit = -Pert(w,t,v)*LC_theo(band,t)*Cref(band,v)/(1 - LC_theo(band,t)) 
                            #      which can be scaled to the level of the disk-integrated flux by
                            # globF(v,t)*(1 - LC_theo(band,t))/LC_theo(band,t)
                            LC_theo_exp = 1. - data_scaling['loc_flux_scaling'][iexp](bins_edge_PCA)
                            pc_mod_exp*= (1. - LC_theo_exp)/LC_theo_exp
                      
                        #Residual PC model from PCA analysis
                        elif (iexp in pca_results['idx_corr']):
                            for i_pc in idx_pc:pc_mod_exp+=pca_results['p_final']['aPC_idx'+str(iexp)+'_ord'+str(i_pc)+'__IS'+inst+'_VS'+vis]*eig_res_matr[i_pc]
               
                        #Save residual model for plotting purposes
                        if plot_dic['map_pca_prof']!='':
                            np.savez_compressed(gen_dic['save_data_dir']+'PCA_results/'+inst+'_'+vis+'_model'+str(iexp) ,data = {'flux':np.array([pc_mod_exp]),'edge_bins' :np.array([bins_edge_PCA]),'cond_def':np.ones(data_dic[inst][vis]['dim_exp'],dtype=bool) },allow_pickle=True)
                
                        #Scale to the level of disk-integrated flux density
                        pc_mod_exp*=data_scaling['glob_flux_scaling'][iexp]
  
                        #Temporary exposure table centered in star rest frame
                        edge_bins_shift = data_corr['edge_bins'][0] - coord_dic[inst][vis]['RV_star_solCDM'][iexp] 
                      
                        #PC model interpolated over exposure table
                        pc_mod_exp = bind.resampling(edge_bins_shift,bins_edge_PCA, pc_mod_exp, kind=gen_dic['resamp_mode'])  
                     
                        #Correction
                        data_corr['flux'][0]+=pc_mod_exp
    
                #-----------------------------------------------------
                #Saving corrected data
                #----------------------------------------------------- 
    
                #Switch spectrum back into wavelength space
                if ('spec' in data_dic[inst][vis]['type']) and (detrend_prof_dic['line_trans'] is not None):
                    data_exp['flux'][iord_line] = data_corr['flux'][0]
                    data_exp['cov'][iord_line] = data_corr['cov'][0]
                    data_exp['edge_bins'][iord_line] = detrend_prof_dic['line_trans']*spec_dopshift(-data_corr['edge_bins'][0])     
                    data_exp['cen_bins'][iord_line]  = detrend_prof_dic['line_trans']*spec_dopshift(-data_corr['edge_bins'][0])                      
                else:data_exp = data_corr
    
                #Updating defined bins
                data_exp['cond_def'] = ~np.isnan(data_exp['flux'])
    
                #Saving corrected data and updating paths 
                np.savez_compressed(proc_DI_data_paths_new+str(iexp) ,data = data_exp,allow_pickle=True)
            
            ### end of exposure    
            data_vis['proc_DI_data_paths'] = proc_DI_data_paths_new
         
        else: 
            #Updating path to processed data and checking it has been calculated
            data_vis['proc_DI_data_paths'] = gen_dic['save_data_dir']+'Detrend_prof/'+inst+'_'+vis+'_' 
            check_data({'path':data_vis['proc_DI_data_paths']+str(0)})

    return None

'''
Sub-function to correct line profile contrast
    - line profiles are temporarily set to a null continuum so that their contrast can be 'stretched'
 detrend_prof = ( ((in_CCF/cont_CCF)-1)/corr_val  + 1)*cont_CCF 
                = (in_CCF-cont_CCF)/corr_val  + cont_CCF 
                = (in_CCF/corr_val) + cont_CCF*(1 - (1/corr_val)) 
    - we assume that the contrast variation arises from a modification of the measured profile akin to a change in LSF
      all profiles (including in-transit) are thus set to a common vertical scale in which the line contrast is equivalent, by normalizing them with the stellar continuum x the profile global flux level   
      both must have been defined from a first processing of the data         
'''
def detrend_prof_ctrst(data_corr,single_l,nord_eff,nspec,glob_corr_ctrst_exp,cond_cont_com,cont_func_dic,rv_shift_cen,scaled_DI_data_paths_exp,t_dur_exp):

    #Correcting single line profile   
    if single_l:

        #CCF continuum flux
        #    - defined in the same way as when fitting CCFs, to be consistent with the definition of the contrast
        cont_CCF=np.mean(data_corr['flux'][0,cond_cont_com])
                 
        #Correcting for contrast variation
        #    - the covariance is only modified by the correction factor
        corr_val = np.repeat(glob_corr_ctrst_exp,nspec) 
        data_corr['flux'][0],data_corr['cov'][0] = bind.mul_array(data_corr['flux'][0],data_corr['cov'][0],1./corr_val)        
        data_corr['flux'][0] += cont_CCF*(1. - (1./corr_val))
        
    #Correcting full spectrum
    else:    
        
        #Align spectra in star rest frame
        #    - the profile is only temporarily shifted and does not need resampling
        cen_bins_rest = data_corr['cen_bins']*spec_dopshift(rv_shift_cen)  

        #Processing each exposure                   
        glob_flux_scaling = dataload_npz(scaled_DI_data_paths_exp)['glob_flux_scaling']   
        glob_sc = np.repeat(1./(t_dur_exp*glob_flux_scaling),nspec) 
        for iord in range(nord_eff):

            #Set spectrum to common vertical range (set to common flux level and correct spectrum for stellar continuum), stretch/correct/unstretch, reset to original range
            #    - broadband flux scaling is not applied to maintain all profiles to the same flux balance
            comm_sc = glob_sc/cont_func_dic(cen_bins_rest[iord])
            data_corr['flux'][iord],data_corr['cov'][iord] = bind.mul_array(data_corr['flux'][iord],data_corr['cov'][iord],comm_sc/glob_corr_ctrst_exp)
            data_corr['flux'][iord] += (1. - (1./glob_corr_ctrst_exp))
            data_corr['flux'][iord],data_corr['cov'][iord] = bind.mul_array(data_corr['flux'][iord],data_corr['cov'][iord],1./comm_sc)

    return data_corr


'''
Sub-function to force a line FWHM to an input value 
    - we perform a stretch of the velocity tables
      modifying the width of a line profile to get FWHM/corr is equivalent to defining the CCF on its velocity table divided by the FWHM correction
    - corrected data are resampled on common visit table if relevant (if a common table is used then all original tables point toward this common table)
    - this operation must be performed on velocity tables symmetrical with respect to the CCF center, ie for CCFs that have been aligned on the null velocity    
'''
def detrend_prof_FWHM(data_corr,FWHM_corr,RV_star_solCDM,gen_dic,comm_sp_tab=False):

    #Temporary spectral table
    #    - stretched by the FWHM correction
    #    - shifted so that all CCFs are aligned to the same RV when stretched
    edge_bins_shift = data_corr['edge_bins'][0] - RV_star_solCDM 
    edge_bins_stretch = edge_bins_shift/FWHM_corr

    #Resampling on common table
    if comm_sp_tab:detrend_prof,corr_cov = bind.resampling(edge_bins_shift, edge_bins_stretch, data_corr['flux'][0] , cov = data_corr['cov'][0], kind=gen_dic['resamp_mode'])       
    else:
        data_corr['edge_bins'][0]=edge_bins_stretch+RV_star_solCDM
        data_corr['cen_bins'][0]=0.5*(data_corr['edge_bins'][0][0:-1]+data_corr['edge_bins'][0][1::])
                     
    return data_corr







































"""
Generic function to align disk-integrated, intrinsic, and planetary profiles
    - profiles used as weights throughout the pipeline must follow the same shifts as their associated profiles
"""
def align_profiles(data_type,data_dic,inst,vis,gen_dic,coord_dic):
    data_vis=data_dic[inst][vis]
    print('   > Aligning '+gen_dic['type_name'][data_type]+' profiles') 
    prop_dic = data_dic[data_type]  
    proc_gen_data_paths_new = gen_dic['save_data_dir']+'Aligned_'+data_type+'_data/'+gen_dic['add_txt_path'][data_type]+'/'+inst+'_'+vis+'_'
    if (data_type=='DI') and (data_dic['DI']['sysvel'][inst][vis]==0.):print('WARNING: sysvel = 0 km/s')
    if data_type=='Intr':proc_gen_data_paths_new+='in'  
    proc_mast = True if ((gen_dic['DImast_weight']) and (data_type in ['Intr','Atm'])) else False
    proc_locEst = True if ((data_type=='Atm') and ((data_dic['Atm']['pl_atm_sign']=='Absorption') or ((data_dic['Atm']['pl_atm_sign']=='Emission')) and data_dic['Intr']['cov_loc_star'])) else False

    #Resample aligned profiles on the common visit table if relevant
    #    - for CCFs the common table is shifted by the systemic velocity to be centered in the star rest frame, as operations using this table will now be performed in the star rest frame, and the 
    # table can be short enough that resampling on the original table would lead to lost pixels
    #      the input velocity table is usually centered around the CCF, ie around RV(CDM_star/CDM_sun)
    #      resampling the CCFs corrected for this motion onto the input table would put them on a side, or even outside the table
    #      we thus shift the table itself to keep the CCFs roughly centered in the table (there is no need to interpolate since this shift is common to all exposures)    
    if (data_vis['comm_sp_tab']):
        data_com = dataload_npz(data_vis['proc_com_data_paths'])
        if (data_type=='DI') and (data_vis['type']=='CCF'): 
            data_com['cen_bins'] -= data_dic['DI']['sysvel'][inst][vis] 
            data_com['edge_bins'] -= data_dic['DI']['sysvel'][inst][vis] 
            data_dic[inst][vis]['proc_com_data_paths'] = gen_dic['save_data_dir']+'Processed_data/'+inst+'_'+vis+'_com_star'
            datasave_npz(data_dic[inst][vis]['proc_com_data_paths'],data_com)       
        cen_bins_resamp, edge_bins_resamp , dim_exp_resamp = data_com['cen_bins'],data_com['edge_bins'],data_com['dim_exp'] 
    else:cen_bins_resamp, edge_bins_resamp , dim_exp_resamp = None,None,None   

    #Calculating aligned data
    if gen_dic['calc_align_'+data_type]:
        print('         Calculating data')

        #Define RV shifts
        #    - reflex motion and systemic velocity for disk-integrated profiles
        #    - surface RV for intrinsic profiles
        #    - for atmospheric profiles we use directly the orbital radial velocity of the planet calculated for each exposure in the star rest frame
        if data_type=='DI': 
            prop_dic[inst][vis]['idx_def'] = range(data_vis['n_in_visit'])
            rv_shifts = coord_dic[inst][vis]['RV_star_solCDM'][prop_dic[inst][vis]['idx_def']]
        elif data_type=='Intr': 
            ref_pl,dic_rv,prop_dic[inst][vis]['idx_def'] = init_surf_shift(gen_dic,inst,vis,data_dic,data_dic['Intr']['align_mode'])
        
            #Remove chromatic RVs
            #    - chromatic deviations from the 'white' average rv of the occulted stellar surface (due to variations in the planet size and stellar intensity) were already 
            # corrected for when extracting the intrinsic profiles
            if ('chrom' in dic_rv):dic_rv.pop('chrom')

        elif data_type=='Atm': 
            if data_dic['Atm']['pl_atm_sign']=='Absorption':rv_shifts = coord_dic[inst][vis][data_dic['Atm']['ref_pl_align']]['rv_pl'][list(np.array(gen_dic[inst][vis]['idx_in'])[prop_dic[inst][vis]['idx_def']])]
            elif data_dic['Atm']['pl_atm_sign']=='Emission':rv_shifts = coord_dic[inst][vis][data_dic['Atm']['ref_pl_align']]['rv_pl'][prop_dic[inst][vis]['idx_def']]

        #Processing each in-transit exposure
        rv_shift_mean = {}
        for isub,iexp in enumerate(prop_dic[inst][vis]['idx_def']):    

            #Upload latest processed data
            data_exp = dataload_npz(data_dic[inst][vis]['proc_'+data_type+'_data_paths']+str(iexp))

            #Achromatic planet-occulted surface rv
            if data_type=='Intr':rv_shift_cen = def_surf_shift(prop_dic['align_mode'],dic_rv,iexp,data_exp,ref_pl,data_vis['type'],data_dic['DI']['system_prop'],data_dic[inst][vis]['dim_exp'],data_dic[inst]['nord'],data_dic[inst][vis]['nspec'])[0]
            elif data_type in ['DI','Atm']:rv_shift_cen = rv_shifts[isub]
            rv_shift_mean[iexp] = np.nanmean(rv_shift_cen)

            #Aligning exposure profile and complementary tables
            #    - the calibration profile is common to all exposures of a processed instrument, and is originally sampled over the table of each exposure in the detector rest frame
            #    - the calibration profile is then used as weight in temporal binning, or to scale back profiles from flux to count units
            #      in both cases the calibration profile must follow the same shifts as the exposure
            if data_vis['tell_sp']:data_exp['tell'] = dataload_npz(data_vis['tell_'+data_type+'_data_paths'][iexp])['tell'] 
            if data_vis['mean_gdet']:data_exp['mean_gdet'] = dataload_npz(data_vis['mean_gdet_'+data_type+'_data_paths'][iexp])['mean_gdet'] 
            data_align=align_data(data_exp,data_vis['type'],data_dic[inst]['nord'],dim_exp_resamp,gen_dic['resamp_mode'],cen_bins_resamp, edge_bins_resamp,rv_shift_cen)

            #Saving aligned exposure and complementary tables
            if data_vis['tell_sp']:
                np.savez_compressed(proc_gen_data_paths_new+'_tell'+str(iexp), data = {'tell':data_align['tell']},allow_pickle=True) 
                data_align.pop('tell')
            if data_vis['mean_gdet']:
                np.savez_compressed(proc_gen_data_paths_new+'_mean_gdet'+str(iexp), data = {'mean_gdet':data_align['mean_gdet']},allow_pickle=True) 
                data_align.pop('mean_gdet')
            np.savez_compressed(proc_gen_data_paths_new+str(iexp),data=data_align,allow_pickle=True)

            #Aligning weighing master
            #    - called for intrinsic and atmospheric profiles, when they are shifted from the star rest frame to other frames
            #      it is not called for DI profiles, since it is computed from DI profiles after alignment (ie, the disk-integrated master does not need further alignment)
            #    - master is shifted for Intr and Atm types 
            #    - the master is originally defined in the star rest frame, like the residual and then intrinsic profiles, but on the common table for the visit
            # + if profiles are shifted and resampled on the common table, this will also be the case of the associated master
            # + if profiles are shifted but kept on their individual tables, the master will remain defined on the common table without being resampled, and it is this table that is shifted
            #   the master table thus becomes specific to each exposure, but is still different from the table of the exposure
            #    - path to the master associated with current profile is updated
            if proc_mast:
                data_ref = dataload_npz(data_vis['mast_'+data_type+'_data_paths'][iexp])
                data_ref_align=align_data(data_ref,data_vis['type'],data_dic[inst]['nord'],dim_exp_resamp,gen_dic['resamp_mode'],cen_bins_resamp,edge_bins_resamp,rv_shift_cen)
                np.savez_compressed(proc_gen_data_paths_new+'_ref'+str(iexp),data={'edge_bins':data_ref_align['edge_bins'],'flux':data_ref_align['flux'],'cov':data_ref_align['cov']},allow_pickle=True)           

            #Retrieve and align estimate of local stellar profile for current exposure, if based on measured profiles
            #    - required to weigh binned atmospheric profiles                
            #    - only defined for in-transit exposures
            if proc_locEst and (iexp in data_vis['LocEst_Atm_data_paths']):
                data_est_loc=np.load(data_vis['LocEst_Atm_data_paths'][iexp]+'.npz',allow_pickle=True)['data'].item() 
                data_est_loc_align=align_data(data_est_loc,data_vis['type'],data_dic[inst]['nord'],dim_exp_resamp,gen_dic['resamp_mode'],cen_bins_resamp, edge_bins_resamp,rv_shift_cen,nocov = ~data_dic['Intr']['cov_loc_star']) 
                np.savez_compressed(proc_gen_data_paths_new+'estloc'+str(iexp),data=data_est_loc_align,allow_pickle=True)

        #Updating path to processed data and saving complementary data
        np.savez_compressed(proc_gen_data_paths_new+'_add',data={'idx_aligned':prop_dic[inst][vis]['idx_def'],'rv_shift_mean':rv_shift_mean},allow_pickle=True)
                            
    #Retrieving data
    else: 
        #Updating path to processed data and checking it has been calculated
        check_data({'path':proc_gen_data_paths_new+'_add'}) 
    data_vis['proc_'+data_type+'_data_paths'] = proc_gen_data_paths_new 
    prop_dic[inst][vis]['idx_def'] = dataload_npz(data_vis['proc_'+data_type+'_data_paths']+'_add')['idx_aligned']
    
    #Updating rest frame
    #    - rest frame of disk-integrated spectra is not updated if systemic velocity is 0
    if (data_type!='DI') or (data_dic['DI']['sysvel'][inst][vis]!=0.):
        data_dic[data_type][inst][vis]['rest_frame'] = {'DI':'star','Intr':'surf','Atm':'pl'}[data_type]
    if proc_mast:data_vis['mast_'+data_type+'_data_paths']={}
    if proc_locEst:data_vis['LocEst_Atm_data_paths'] = {}
    if data_vis['tell_sp']:data_vis['tell_'+data_type+'_data_paths']={}  
    if data_vis['mean_gdet']:data_vis['mean_gdet_'+data_type+'_data_paths']={}  
    for iexp in prop_dic[inst][vis]['idx_def']:
        if proc_mast:data_vis['mast_'+data_type+'_data_paths'][iexp]=proc_gen_data_paths_new+'_ref'+str(iexp)
        if proc_locEst and (iexp in data_vis['LocEst_Atm_data_paths']):data_vis['LocEst_Atm_data_paths'][iexp] = proc_gen_data_paths_new+'estloc'+str(iexp) 
        if data_vis['tell_sp']:data_vis['tell_'+data_type+'_data_paths'][iexp] = proc_gen_data_paths_new+'_tell'+str(iexp)
        if data_vis['mean_gdet']:data_vis['mean_gdet_'+data_type+'_data_paths'][iexp] = proc_gen_data_paths_new+'_mean_gdet'+str(iexp)  
            
    return None




"""
Sub-function to return radial velocities of planet-occulted regions
"""
def init_surf_shift(gen_dic,inst,vis,data_dic,align_mode):

    #Set surface RV to measured achromatic value  
    #    - only in-transit profiles profiles for which the local stellar line was flagged as detected can be aligned  
    if (align_mode=='meas'):
        dic_rv = np.load(gen_dic['save_data_dir']+'Introrig_prop/'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item()
        idx_aligned = np_where1D(dic_rv['cond_detected'])
        ref_pl=None

    #Set surface RVs to model
    #    - all in-transit profiles can be aligned, as rv can be calculated at any phase
    elif (align_mode=='theo'):        
        dic_rv = np.load(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item()     
        idx_aligned = np.arange(data_dic[inst][vis]['n_in_tr'])   

        #Reference planet
        #    - theoretical velocities are calculated for all planets transiting in a given visit
        if len(data_dic[inst][vis]['transit_pl'])==1:ref_pl = data_dic[inst][vis]['transit_pl'][0]  
        else:ref_pl=data_dic['Intr']['align_ref_pl'][inst][vis]
    
    return ref_pl,dic_rv,idx_aligned


def def_surf_shift(align_mode,dic_rv,i_in,data_exp,pl_ref,data_type,system_prop,dim_exp,nord,nspec):    

    #Set surface RV to measured achromatic value 
    if (align_mode=='meas'):
        surf_shifts=dic_rv[i_in]['rv']
        surf_shifts_edge=None
    
    #Set surface RVs to model
    elif ('theo' in align_mode):

        #Chromatic RVs
        #    - theoretical RVs calculated for the broadband RpRs and intensity values provided as inputs are interpolated over the table of each exposure, so that 
        # each bin is shifted with the surface rv corresponding to its wavelength   
        #    - if data has been converted from spectra to CCF, nominal properties will be used
        if ('spec' in data_type) and ('chrom' in dic_rv):
            surf_shifts=np.zeros(dim_exp,dtype=float)*np.nan
            surf_shifts_edge=np.zeros([nord,nspec+1],dtype=float)*np.nan
            
            #Absolute or chromatic-to-nominal relative RV
            if align_mode=='theo':RV_shift_pl = dic_rv['chrom'][pl_ref]['rv'][:,i_in]
            elif align_mode=='theo_rel':RV_shift_pl = dic_rv['chrom'][pl_ref]['rv'][:,i_in]-dic_rv['achrom'][pl_ref]['rv'][0,i_in]              
            for iord in range(nord):
                surf_shifts[iord] = np_interp(data_exp['cen_bins'][iord],system_prop['chrom']['w'],RV_shift_pl,left=RV_shift_pl[0],right=RV_shift_pl[-1])
                surf_shifts_edge[iord] = np_interp(data_exp['edge_bins'][iord],system_prop['chrom']['w'],RV_shift_pl,left=RV_shift_pl[0],right=RV_shift_pl[-1])
            
        #Achromatic RV defined for the nominal transit properties
        else:
            surf_shifts = dic_rv['achrom'][pl_ref]['rv'][0,i_in]  
            surf_shifts_edge=None

    return surf_shifts,surf_shifts_edge






'''
Aligning profiles 
'''
def spec_dopshift(rv_shift):
    return np.sqrt(1. - (rv_shift/c_light) )/np.sqrt(1. + (rv_shift/c_light) )

def align_data(data_exp,rout_mode,nord,dim_exp_resamp,resamp_mode,cen_bins_resamp, edge_bins_resamp,rv_shift_cen,rv_shift_edge = None , nocov = False):

    #Shift wavelength tables to a common rest frame
    #    - disk-integrated spectra, defined in the observer rest frame, are corrected for the motion of the star relative to the sun (ie the orbital motion of star around the stellar
    # system barycenter + systemic velocity) and aligned in the star rest frame
    #      intrinsic spectra, defined in the star rest frame, are corrected for the motion of the stellar surface relative to the star rest velocity, and aligned into their local rest frame 
    #      out-of-transit profiles or aligned intrinsic profiles used as estimates for a given local profile, can be seen as defined in a null rest frame, and shifted back to the stellar surface velocity relative to the star rest velocity
    #      atmospheric profiles, defined in the star rest frame, are corrected for the orbital motion of the planet relative to the star rest velocity, and aligned into the planet rest frame  
    #    - Doppler effect
    # nu_received/nu_source = (1 + v_receiver/c )/(1 - v_source/c ) 
    # w_source/w_received   = (1 + v_receiver/c )/(1 - v_source/c )  
    #      with v_receiver > 0 if receiver moving toward the source
    #      with v_source   > 0 if source moving toward the receiver
    #      for disk-integrated spectra the source is the star, and the receiver the observer             
    #           v_receiver = 0
    #           v_source = - rv_star_obs since rv_star_obs < 0 when moving toward us             
    #           w_star/w_observer   = (1 + v_observer/c )/(1 - v_star/c )                  
    #           w_star = w_observer/(1 + rv_star_obs/c )
    #           and using the more precise relativistic formula:
    #           w_star = w_observer*sqrt(1 - rv_star_obs/c )/sqrt(1 + rv_star_obs/c )                  
    #
    #      for intrinsic spectra the source is the planet-occulted region, and the receiver the star                    
    #           v_receiver = 0                
    #           v_source = - rv_surf_star since rv_surf_star < 0 when moving toward us  
    #           w_region/w_star   = 1/(1 + rv_surf_star/c )        
    #           w_region = w_star/(1 + rv_surf_star/c ) 
    #           w_region = w_star*sqrt(1 - rv_surf_star/c )/sqrt(1 + rv_surf_star/c )  
    #
    #      for aligned profiles used as estimate of local profiles, the source can be seen as the star and the receiver the planet-occulted region 
    #           w_star = w_region/(1 + rv_star_surf/c )  
    #           w_star = w_region*sqrt(1 - rv_star_surf/c )/sqrt(1 + rv_star_surf/c ) 
    #           w_star = w_region*sqrt(1 - (-rv_surf_star)/c )/sqrt(1 + (-rv_surf_star)/c ) 
    #
    #      for atmospheric spectra the source is the planet, and the receiver the star                    
    #           v_receiver = 0                
    #           v_source = - rv_pl_star since rv_pl_star < 0 when moving toward us  
    #           w_pl/w_star   = 1/(1 + rv_pl_star/c )        
    #           w_pl = w_star/(1 + rv_pl_star/c ) 
    #           w_pl = w_star*sqrt(1 - rv_pl_star/c )/sqrt(1 + rv_pl_star/c )      
    if ('spec' in rout_mode):
    
        #Achromatic shift
        if (rv_shift_edge is None):
            dop_shift = spec_dopshift(rv_shift_cen) 
            edge_bins_rest = data_exp['edge_bins']*dop_shift
            cen_bins_rest = data_exp['cen_bins']*dop_shift                         
         
        #Chromatic shift
        #    - in this case the bin edges and center are shifted using the rv calculated at their exact wavelength, to keep the new bins contiguous
        else:
            edge_bins_rest = data_exp['edge_bins']*spec_dopshift(rv_shift_edge)
            cen_bins_rest = data_exp['cen_bins']*spec_dopshift(rv_shift_cen)  

            
    #Shift velocity tables to chosen rest frame
    #    - for disk-integrated data: RV(M/star) = RV(M/sun) - RV(CDM_star/sun) - RV(star/CDM_star)
    #    - for intrinsic data:       RV(M/region) = RV(M/star) - RV(region/star)
    #    - for master out data:      RV(M/star) = RV(M/region) + RV(region/star)
    #    - for atmospheric data:     RV(M/pl) = RV(M/star) - RV(pl/star)
    elif (rout_mode=='CCF'):
        edge_bins_rest = data_exp['edge_bins'] - rv_shift_cen
        cen_bins_rest = data_exp['cen_bins'] - rv_shift_cen        

    #----------------------------------------------------------------

    #Port data from previous processing
    data_align = deepcopy(data_exp) 
    
    #Data is resampled on the common table given as input
    #    - for spectra we neglect the few bins lost by resampling the data (defined on tables shifted to the star rest frame) on the common table defined in the input rest frame
    if cen_bins_resamp is not None:

        #Initialize aligned data
        data_align['edge_bins']=deepcopy(edge_bins_resamp)
        data_align['cen_bins']=deepcopy(cen_bins_resamp) 
        data_align['flux']=np.zeros(dim_exp_resamp, dtype=float)*np.nan
  
        #Aligning each order
        if nocov:
            for iord in range(nord):
                data_align['flux'][iord] = bind.resampling(data_align['edge_bins'][iord], edge_bins_rest[iord], data_exp['flux'][iord], kind=resamp_mode)            
        else:
            data_align['cov']=np.zeros(nord, dtype=object)          
            for iord in range(nord):
                data_align['flux'][iord],data_align['cov'][iord] = bind.resampling(data_align['edge_bins'][iord], edge_bins_rest[iord], data_exp['flux'][iord] , cov = data_exp['cov'][iord], kind=resamp_mode) 
      
        #Processing telluric spectrum
        if ('tell' in data_exp):
            data_align['tell']=np.zeros(dim_exp_resamp, dtype=float)*np.nan
            for iord in range(nord):data_align['tell'][iord] = bind.resampling(data_align['edge_bins'][iord], edge_bins_rest[iord], data_exp['tell'][iord], kind=resamp_mode)            
        
        #Processing calibration profile
        if ('mean_gdet' in data_exp):
            data_align['mean_gdet']=np.zeros(dim_exp_resamp, dtype=float)*np.nan
            for iord in range(nord):data_align['mean_gdet'][iord] = bind.resampling(data_align['edge_bins'][iord], edge_bins_rest[iord], data_exp['mean_gdet'][iord], kind=resamp_mode)             
            
        #Defined bins
        data_align['cond_def'] = ~np.isnan(data_align['flux'])
 
    #Data remain defined on independent tables for each exposure
    #    - we do not resample the data tables of each exposure and keep them defined on their shifted spectral tables
    else:      
        data_align['cen_bins'] = cen_bins_rest   
        data_align['edge_bins'] = edge_bins_rest                

    return data_align



































'''
Scale data to the correct flux level
    - spectra Fcorr(w,t,v) should have been set to the same flux balance as a reference spectrum at low resolution
      Fcorr(w,t,v) = Fstar(w,v)*Cref(w,v)*L(t)
      where Cref(w in band,v) ~ Cref(band,v) represents a possible low-frequency deviation from the true stellar spectrum
            L(t) represents the global flux deviation from the true stellar spectrum, not corrected for in previous modules 
    - we first convert all spectra from flux to (temporal) flux density units, so that they are equivalent except for flux variations 
    - we then correct for L(t) by dividing the profiles by globF(v,t) = TFcorr(v,t)/med(tk, TFcorr(v,tk) ) or TFcorr(v,t)/(Fsc*int(w in full_band, dw )) 
      where TFcorr(v,t) = int(w in full_band, Fcorr(w,t,v)*dw ) = C*L(t) 
      the exact value of the (unocculted) global flux level does not matter within a visit and is set to the median flux of all spectra, unless the user choses to impose a common flux density for all visits (in case disk-integrated data needs to be combined between visits)          
    - we then need to rescale spectrally the data so that 
 Fsc(w,t) = c(w,t)*Fcorr(w,t) = F(w,t) 
      to do so we use broadband spectral light curves, which by definition correspond to the ratio of in- and out-of-transit flux integrated over the band:
 LC(band,vis,t) = int(w in band, F(w,t)*dw )/int(w in band, Fstar(w,vis)*dw )  = TF(band,t)/TFstar(band,vis)  
      we assume that c(w,t) has low-frequency variations (which should be the case, since the difference between Fcorr and F comes from the changes in color balance)
      the correction thus implies
 int(w in band, c(w,t)*Fcorr(w,t) ) = int(w in band, F(w,t) )                 
 c(band,t)*int(w in band, Fcorr(w,t) ) = TF(band,t)   
 c(band,t) = TFstar(band,vis)*LC(band,vis,t) / int(w in band, Fstar(w,vis)*Cref(band,v) )   
 c(band,t) = TFstar(band,vis)*LC(band,vis,t) / ( TFstar(band,vis)*Cref(band,v) ) 
 c(band,t) = LC(band,vis,t) / Cref(band,v) 
      we set c(band,t) = LC_theo(band,t) and interpolate the chromatic c(band,t) at all wavelengths in the spectrum to define c(w,t)
 Fsc(w,t,v) = LC_theo(w,t)*Fcorr(w,t)/globF(v,t)
            = F(w,t,v)*Cref(band,v)
      
    - care must be taken about Cref(band,v) when several visits are exploited:
 (1) if the stellar emission remains the same in all visits, then we can use a single theoretical spectrum / master as reference, and a constant normalized light curve       
 Fsc(w,t,v) = F(w,t,v)*Cref(band)
 
 (2) if the stellar emission changes in amplitude uniformely over the star, the normalized light curve remains the same, and the stellar spectrum keeps the same profile:   
 Fsc(w,t,v) = F(w,t,v)*Cref(band)/Fr(v)  
     where Fr(v) = Fstar(w,v)/Fstar(w,v_ref) 
           Fstar(w,v_ref) the stellar spectrum of one of the visit taken as reference
           LC_theo(w,t) = F(w,v_ref,t)/Fstar(w,v_ref) = F(w,t,v)/Fstar(w,v)
 Fsc(w,t,v) = F(w,t,v)*Cref(band)*Fstar(w,v_ref)/Fstar(w,v)           
            = F(w,v_ref,t)*Cref(band)*Fstar(w,v_ref)/Fstar(w,v_ref)        
            = F(w,v_ref,t)*Cref(band)  
     scaled spectra thus keep the same profile, deviating from the true profile by a common Cref(band) in all visits)

 (3) if the stellar emission changes in amplitude and shape uniformely over the star, the normalized light curve remains the same but the stellar spectrum has not the same profile:
 Fsc(w,t,v) = LC_theo(w,t)*F(w,v)*Cref(w,v)
            = (F(w,v,t)/Fstar(w,v))*Fstar(w,t)*Cref(band,v) 
            = F(w,t,v)*Cref(band,v)
     by using a reference spectrum specific to each visit when correcting the color balance, with the correct shape and relative flux, we would get:        
 Fsc(w,t,v) = F(w,t,v)*Cref
     where Cref is the deviation from the absolute flux level, common to all visits
 
 (4) if local stellar spectra in specific regions of the star change between visits, eg due to (un)occulted spots, then the normalized light curve changes as well
     both a reference spectrum and a light curve specific to each visit must be used to remove the stellar contribution Fstar(w,v) and avoid introducing a different balance to the planet-occulted spectra 

      
    - if there is emission/reflection from the planet, then the true flux writes as:
 F(w,vis,t) = ( Fstar(w,vis)*LCtr(w,vis,t) + Fstar(w,vis)*LCrefl(w,vis,t) + Fp_thermal(w,vis,t) )
            = Fstar(w,vis) * ( delt_p_tr(w,vis,t) + delt_p_refl(w,vis,t) + delt_p_th(w,vis,t) )
            = Fstar(w,vis) * delt_p(w,vis,t)
      the correction should still ensure that:
 c(band,t) = TF(band,vis,t) / ( TFstar(band,vis)*Cref(band) ) 
 c(band,t) = LC(band,vis,t)*TFstar(band,vis) / ( TFstar(band,vis)*Cref(band) ) 
 c(band,t) = LC(band,vis,t) / Cref(band) 
      but broadband spectral light curves correspond to:
  LC(band,vis,t) = TF(band,vis,t)/TFstar(band,vis) = int(w in band, Fstar(w,vis) * delt_p(w,vis,t) *dw )/TFstar(band,vis)       
      so that care must be taken to use light curves that integrate the different spectral contributions and are then normalized by the integrated stellar flux, rather than integrating the pure planet contribution 
        
    - for spectra, we can use the flux integrated over bands including the planetary absorption, as we can match it with the same band over which the light curve was measured
      ideally the input light curves should be defined with a fine enough resolution to sample the low-frequency variations of the planetary atmosphere and stellar limb-darkening 
    - for CCFs, a light curve does not match the cumulated flux of the different spectral regions used to calculate the CCF 
      if we assume that the signature of the planet and of the RM effect is similar in all lines used to calculate the CCF, and that the light curve is achromatic, then it is similar to the scaling of a spectrum with a single line
      thus, as with spectra, CCFs should be scaled using the flux of their full profile and not just the continuum (although there will always be a bias in using CCFs at this stage and the use of spectra should be preferred)
    - since the scaling is imposed by a light curve independent of the spectra, bin by bin, it does not need to be applied to the full spectra like the flux balance color correction, and can be applied to any spectral range 

    - note that the measured light curve corresponds to 
 LC(band,t) = int(w over band, Fin(w,t)*dw )/int(w over band, Fstar(w,vis)*dw )
            = 1 - int(w over band, fp(w)*( Sthick(band,t) + Sthin(w,t) )/<Fstar(w in band,vis)> 
      the theoretical light curves fitted to the measured one assume a constant, uniform stellar flux and limb-darkening law, and a constant average radius over the band
 LC_theo(band,t) = TF_theo_band(t) / TFstar_theo(band) 
                 = F_theo_band(t) / Fstar_theo(band) 
                 = 1 - I_theo(band)*LD(band,t)*Sp(band,t)/sum(k , I_theo(band)*LDk(band)*Sk) 
                 = 1 - LD(band,t)*Sp(band,t)/sum(k , LDk(band)*Sk) 
                 = 1 - LD(band,t)*Sp(band,t)/Sstar_LD(band)
        the fact that this is an approximation does not matter as long as the measured light curve is correctly reproduced, in which case the rescaling with the theoretical light curve is correct
        in the following we assume that the theoretical LD matches the true LD
    - out-of-transit data is rescaled as well, to account for possible variations of stellar origin in the disk-integrated flux  
'''
def rescale_data(data_inst,inst,vis,data_dic,coord_dic,exp_dur_d,gen_dic,plot_dic,system_param,theo_dic):

    print('   > Broadband flux scaling') 
    data_vis=data_inst[vis]
    proc_DI_data_paths_new = gen_dic['save_data_dir']+'Scaled_data/'+inst+'_'+vis+'_'
    data_vis['scaled_DI_data_paths'] = proc_DI_data_paths_new+'scaling_'         

    #Calculating rescaled data
    if (gen_dic['calc_flux_sc']):
        print('         Calculating data')
        transit_prop=data_dic['DI']['transit_prop'][inst][vis]
        dic_save={}
    
        #Light curve for all bands and all exposures
        #    - chromatic values are used if provided and if disk-integrated profiles are in spectral mode
        if ('spec' in data_vis['type']) and ('chrom' in data_dic['DI']['system_prop']):key_chrom = ['chrom']
        else:key_chrom = ['achrom']
        system_prop = data_dic['DI']['system_prop'][key_chrom[0]]
        LC_flux_band_all = np.zeros([data_vis['n_in_visit'],system_prop['nw']])*np.nan    

        #Simulated light curves
        if transit_prop['mode']=='simu':
            params_LC = deepcopy(system_param['star'])
            params_LC.update({'rv':0.,'cont':1.}) 

        #Calculate light curve for plotting        
        if (plot_dic['input_LC']!='') or (plot_dic['prop_Intr']!=''):
            
            #High-resolution time table over visit
            min_bjd = coord_dic[inst][vis]['bjd'][0]
            max_bjd = coord_dic[inst][vis]['bjd'][-1]
            dbjd_HR = plot_dic['dt_LC']/(3600.*24.)
            nbjd_HR = round((max_bjd-min_bjd)/dbjd_HR)
            bjd_HR=min_bjd+dbjd_HR*np.arange(nbjd_HR)
      
            #Corresponding orbital phases and coordinates for each planet
            #    - high-resolution tables are calculated assuming no exposure duration
            coord_pl_HR = {}
            LC_HR=np.ones([nbjd_HR,system_prop['nw']],dtype=float)  
            if transit_prop['mode']=='simu':ecl_all_HR = np.zeros(nbjd_HR,dtype=bool)
            for pl_loc in data_inst[vis]['transit_pl']:
                pl_params_loc=system_param[pl_loc]
                coord_pl_HR[pl_loc]={'cen_ph':get_timeorbit(pl_loc,coord_dic,inst,vis,bjd_HR,pl_params_loc,None)[1]}   

                #Definition of coordinates for all transiting planets
                if transit_prop['mode']=='simu': 
                    x_pos_pl,y_pos_pl,z_pos_pl,Dprojp,_,_,_,ecl_pl = calc_pl_coord(pl_params_loc['ecc'],pl_params_loc['omega_rad'],pl_params_loc['aRs'],pl_params_loc['inclin_rad'],coord_pl_HR[pl_loc]['cen_ph'],data_dic['DI']['system_prop']['achrom'][pl_loc][0],pl_params_loc['lambda_rad'],system_param['star'])
                    coord_pl_HR[pl_loc].update({'ecl':ecl_pl,'cen_pos':np.vstack((x_pos_pl,y_pos_pl,z_pos_pl))})
    
                    #Exposure considered out-of-transit if no planet at all is transiting
                    ecl_all_HR |= abs(ecl_pl)!=1                      

        #------------------------------------------------------------------------        
        
        #Light curves from import
        #    - defined over a set of wavelengths that can be different for each visit
        #    - here we import the light curves, so that they can be interpolated for each visit after their exposures have been defined
        if transit_prop['mode']=='imp':
            t_dur_d=coord_dic[inst][vis]['t_dur']/(3600.*24.)
            cen_bjd = coord_dic[inst][vis]['bjd']
            
            #Retrieving light curve
            #    - first column must be absolute time (BJD), to be independent of any planet body 
            #    - next columns must be normalized stellar flux for all chosen bands
            ext = transit_prop['path'].split('.')[-1]
            if (ext=='csv'):
                imp_LC = (pd.read_csv(transit_prop['path'])).values
            elif (ext in ['txt','dat']):
                imp_LC = np.loadtxt(transit_prop['path']).T          
            else:
                stop('Light curve path extension TBD') 
            imp_LC[0] -= 2400000. 
            if (plot_dic['input_LC']!=''):dic_save['imp_LC'] = imp_LC

            #Average imported light curve within the exposure time windows
            #    - the light curve must be imported with sufficient temporal resolution
            for iexp,(bjd_loc,dt_loc) in enumerate(zip(cen_bjd,t_dur_d)):
                
                #Imported points within exposure
                id_impLC=np_where1D( (imp_LC[0]>=bjd_loc-0.5*dt_loc) & (imp_LC[0]<=bjd_loc+0.5*dt_loc))
                
                #Normalized flux averaged within exposure
                if len(id_impLC)>0:LC_flux_band_all[iexp,:]=np.mean(imp_LC[1::,id_impLC],axis=1)
                else:stop('No LC measurements within exposure')

            #Calculate light curve for plotting        
            if (plot_dic['input_LC']!='') or (plot_dic['prop_Intr']!=''):     
                for iband in range(system_prop['nw']):LC_HR[:,iband] = np_interp(bjd_HR,imp_LC[0],imp_LC[1+iband],left=imp_LC[1+iband,0],right=imp_LC[1+iband,-1])

        #------------------------------------------------------------------------
        
        #Model light curve for a single planet
        #    - can be oversampled   
        #    - defined over a set of wavelengths but constant for each visit
        elif transit_prop['mode']=='model':
            if len(data_inst[vis]['transit_pl'])>1:stop('Multiple planets transiting')
            pl_vis = data_inst[vis]['transit_pl'][0]
            LC_params = batman.TransitParams()
            LC_pl_params = system_param[pl_vis]
        
            #Phase reference for inferior conjunction
            LC_params.t0 = 0. 
            
            #Orbital period in phase
            LC_params.per = 1. 
            
            #Semi-major axis (in units of stellar radii)
            LC_params.a = LC_pl_params['aRs']
            
            #Orbital inclination (in degrees)
            #    - from the line of sight to the normal to the orbital plane
            LC_params.inc = LC_pl_params['inclination'] 
            
            #Eccentricity
            LC_params.ecc = LC_pl_params['ecc']
            
            #Longitude of periastron (in degrees)
            LC_params.w = LC_pl_params['omega_deg']
            
            #Oversampling 
            if ('dt' not in transit_prop):LC_osamp = 10
            else:LC_osamp = npint(np.ceil(coord_dic[inst][vis]['t_dur']/(60.*transit_prop['dt'])))

            #Calculate white or chromatic light curves
            cen_ph_pl = coord_dic[inst][vis][pl_vis]['cen_ph']
            ph_dur_pl=coord_dic[inst][vis][pl_vis]['ph_dur']
            for iband,wband in enumerate(system_prop['w']):
    
                #Light curve properties for the band
                LC_params_band = deepcopy(LC_params)
    
                #LD law 
                LD_mod = system_prop['LD'][iband]
        
                #Limb darkening coefficients in the format required for batman
                LC_params_band.limb_dark = LD_mod
                if LD_mod == 'uniform':
                    ld_coeff=[]
                elif LD_mod == 'linear':
                    ld_coeff=[system_prop['LD_u1'][iband]]
                elif LD_mod in ['quadratic' ,'squareroot','logarithmic', 'power2' ,'exponential']:
                    ld_coeff=[system_prop['LD_u1'][iband],system_prop['LD_u2'][iband]]
                elif LD_mod == 'nonlinear':   
                    ld_coeff=[system_prop['LD_u1'][iband],system_prop['LD_u2'][iband],system_prop['LD_u3'][iband],system_prop['LD_u4'][iband]]           
                else:
                    stop('Limb-darkening not supported by batman')  
                LC_params_band.u=ld_coeff
        
                #Planet-to-star radius ratio
                LC_params_band.rp=system_prop[pl_vis][iband]

                #All exposures have same duration
                #    - process each band for all exposures together
                if coord_dic[inst][vis]['cst_tdur']:
                    LC_flux_band_all[:,iband] = batman.TransitModel(LC_params_band, cen_ph_pl, supersample_factor = LC_osamp[0], exp_time = ph_dur_pl[0]).light_curve(LC_params_band)
                    
                #Exposures have different durations
                #    - process each band and each exposure
                else:                      
                    for iexp,(cen_ph_exp,ph_dur_exp,LC_osamp_exp) in enumerate(zip(cen_ph_pl,ph_dur_pl,LC_osamp)):                    
                        LC_flux_band_all[iexp,iband]=float(batman.TransitModel(LC_params_band, np.array([cen_ph_exp]), supersample_factor = LC_osamp_exp, exp_time = np.array([ph_dur_exp])).light_curve(LC_params_band))
             
                #Calculate light curve for plotting        
                if (plot_dic['input_LC']!='') or (plot_dic['prop_Intr']!=''):
                    LC_HR[:,iband] = batman.TransitModel(LC_params_band,coord_pl_HR[pl_vis]['cen_ph']).light_curve(LC_params_band)  
            
        #------------------------------------------------------------------------
     
        #Simulated light curve   
        #    - can account for multiple transiting planets
        elif transit_prop['mode']=='simu':    
            
            #Set out-of-transit values to unity
            LC_flux_band_all[gen_dic[inst][vis]['idx_out'],:]=1.      
            
            #Oversampling factor, in units of RpRs
            theo_dic_LC= deepcopy(theo_dic)
            theo_dic_LC['d_oversamp']={}
            if (transit_prop['n_oversamp']>0.):
                for pl_loc in data_inst[vis]['transit_pl']:theo_dic_LC['d_oversamp'][pl_loc]=data_dic['DI']['system_prop']['achrom'][pl_loc][0]/transit_prop['n_oversamp'] 

            #Calculate transit light curves accounting for all planets in the visit
            plocc_prop = sub_calc_plocc_prop(key_chrom,{},[],data_inst[vis]['transit_pl'],system_param,theo_dic_LC,data_dic['DI']['system_prop'],params_LC,coord_dic[inst][vis],gen_dic[inst][vis]['idx_in'],False,Ftot_star=True) 
            LC_flux_band_all[gen_dic[inst][vis]['idx_in'],:]=plocc_prop[key_chrom[0]]['Ftot_star'].T   
            
            #Calculate light curve for plotting        
            if (plot_dic['input_LC']!='') or (plot_dic['prop_Intr']!=''):
                theo_dic_LC['d_oversamp']={}
                idx_in_HR = np_where1D(ecl_all_HR)
                plocc_prop_HR = sub_calc_plocc_prop(key_chrom,{},[],data_inst[vis]['transit_pl'],system_param,theo_dic_LC,data_dic['DI']['system_prop'],params_LC,coord_pl_HR,idx_in_HR,False,Ftot_star=True) 
                LC_HR[idx_in_HR,:]=plocc_prop_HR[key_chrom[0]]['Ftot_star'].T   

        #Store for plots
        if (plot_dic['input_LC']!='') or (plot_dic['prop_Intr']!=''):
            dic_save['flux_band_all'] = LC_flux_band_all
            dic_save['coord_HR'] = coord_pl_HR
            dic_save['LC_HR'] = LC_HR


        #Save for plots
        if len(dic_save)>0:datasave_npz(proc_DI_data_paths_new+'add',dic_save)        

        #------------------------------------------------------------------------

        #Upload common spectral table
        #    - if profiles are defined on different tables they are resampled on this one
        #      if they are already defined on a common table, it is this one, which has been kept the same since the beginning of the routine
        edge_bins_com = (np.load(data_vis['proc_com_data_paths']+'.npz',allow_pickle=True)['data'].item())['edge_bins']

        #Spectral scaling table and global scaling range
        loc_flux_scaling = np.zeros(data_vis['n_in_visit'],dtype=object) 
        flux_all = np.zeros(data_vis['dim_all'],dtype=float)*np.nan
        cond_def_all = np.zeros(data_vis['dim_all'],dtype=bool)
        cond_def_scal_all  = np.zeros(data_vis['dim_all'],dtype=bool)
        null_loc_flux_scaling = np.zeros(data_vis['n_in_visit'],dtype=bool)
        for iexp in range(data_vis['n_in_visit']): 
            
            #Latest processed DI data
            data_exp = dataload_npz(data_vis['proc_DI_data_paths']+str(iexp))

            #Resampling and conversion to temporal flux density
            #    - if data were kept on independent tables they need to be resampled on a common one to calculate equivalent fluxes
            if (not data_vis['comm_sp_tab']):
                for iord in range(data_inst['nord']): 
                    flux_all[iexp,iord] = bind.resampling(edge_bins_com[iord], data_exp['edge_bins'][iord], data_exp['flux'][iord] , kind=gen_dic['resamp_mode'])                                                        
                cond_def_all[iexp] = ~np.isnan(flux_all[iexp])   
            else:
                flux_all[iexp] = data_exp['flux']
                cond_def_all[iexp] = data_exp['cond_def']
            flux_all[iexp]/=coord_dic[inst][vis]['t_dur'][iexp]

            #Spectral scaling table                                        
            #    - scale to the expected flux level at all wavelengths, using the broadband flux interpolated over the full spectrum range, unless a single band is used
            #    - accounts for the potentially chromatic signature of the planet 
            if np.max(np.abs(1.-LC_flux_band_all[iexp]))>0.:null_loc_flux_scaling[iexp] = False
            if (system_prop['nw']==1):loc_flux_scaling[iexp] = np.poly1d([1.-LC_flux_band_all[iexp,0]])
            else:loc_flux_scaling[iexp] = interp1d(system_prop['w'],1.-LC_flux_band_all[iexp],fill_value=(1.-LC_flux_band_all[iexp,0],1.-LC_flux_band_all[iexp,-1]), bounds_error=False)
                
            #Requested scaling range
            if len(data_dic['DI']['scaling_range'])>0:
                cond_def_scal=False 
                for bd_int in data_dic['DI']['scaling_range']:cond_def_scal |= (edge_bins_com[:,0:-1]>=bd_int[0]) & (edge_bins_com[:,1:]<=bd_int[1])   
            else:cond_def_scal=True 

            #Accounting for undefined pixels in scaling range            
            cond_def_scal_all[iexp] = cond_def_all[iexp]  & cond_def_scal            

        #Scaling pixels common to all exposures
        #    - planetary signatures should not be excluded from the range of summation, for the same reason as they are included in the spectral scaling : the light curves used for the scaling include those ranges potentially absorbed by the planet
        #      the same logic applies to CCF: their full range must be used for the scaling, and not just the continuum      
        cond_scal_com  = np.all(cond_def_scal_all,axis=0)
        if np.sum(cond_scal_com)==0.:stop('No pixels in common scaling range')    
      
        #Defining global scaling values
        #    - used to set all profiles to a common global flux level
        #    - spectral profiles have been trimmed, corrected, and aligned
        #      it might thus be more accurate to rescale them using their own flux rather than the flux of the original spectra
        #      furthermore masters afterward will be calculated from these profiles, scaled, thus they do not need to be set to the level of the original global master
        #      we thus use the total flux summed over the full range of the current profiles, with their median taken as reference
        #    - defined on temporal flux density (not cumulated photoelectrons counts)
        Tflux_all = np.zeros(data_vis['n_in_visit'],dtype=float)
        dcen_bin_comm = (edge_bins_com[:,1::] - edge_bins_com[:,0:-1])
        Tcen_bin_comm = 0.
        for iord in range(data_inst['nord']):    
            Tflux_all += np.sum(flux_all[:,iord,cond_scal_com[iord]]*dcen_bin_comm[iord,cond_scal_com[iord]],axis=1)
            Tcen_bin_comm += np.sum(dcen_bin_comm[iord,cond_scal_com[iord]])
        if data_dic['DI']['scaling_val'] is None:Tflux_ref = np.median(Tflux_all)
        else:Tflux_ref=Tcen_bin_comm*data_dic['DI']['scaling_val']
        norm_exp_glob = Tflux_all/Tflux_ref
            
        #Scaling each exposure
        #    - only defined bins are scaled (the flux in undefined bins remain set to nan), but the scaling spectrum was calculated at all wavelengths so that it can be used later with data for which different bins are defined or not
        #    - all defined bins remain defined 
        #    - operation depends on condition 'rescale_DI' because flux scaling tbales may be required even if data needs not be scaled
        if data_dic['DI']['rescale_DI']:          
            for iexp in range(data_vis['n_in_visit']):  
                
                #Save exposure
                data_exp = np.load(data_vis['proc_DI_data_paths']+str(iexp)+'.npz',allow_pickle=True)['data'].item() 
                for iord in range(data_inst['nord']): 
                    LC_exp_spec_ord = 1.-loc_flux_scaling[iexp](data_exp['cen_bins'][iord])
                    data_exp['flux'][iord],data_exp['cov'][iord] = bind.mul_array(data_exp['flux'][iord],data_exp['cov'][iord],LC_exp_spec_ord/(coord_dic[inst][vis]['t_dur'][iexp]*norm_exp_glob[iexp]))
                datasave_npz(proc_DI_data_paths_new+str(iexp),data_exp)
                
                #Save scaling
                #    - must be saved for each exposure because (for some reason) the interp1d function cannot be saved once passed out of multiprocessing 
                #    - in-transit scaling can be used later to manipulate local profiles from the planet-occulted regions 
                data_scaling = {'loc_flux_scaling':loc_flux_scaling[iexp],'glob_flux_scaling':norm_exp_glob[iexp],'null_loc_flux_scaling':null_loc_flux_scaling[iexp]}
                if system_prop['nw']>1:data_scaling['chrom']=True
                else:data_scaling['chrom']=False
                datasave_npz(data_vis['scaled_DI_data_paths']+str(iexp),data_scaling)                
        
        #Saving complementary data
        np.savez_compressed(proc_DI_data_paths_new+'add',data={'rest_frame':data_dic['DI'][inst][vis]['rest_frame']},allow_pickle=True)           
        
    #Updating path to processed data and checking it has been calculated
    else:   
        check_data({'path':proc_DI_data_paths_new+str(0)})   
    data_vis['proc_DI_data_paths'] = proc_DI_data_paths_new

    return None
    













"""
Sub-function to calculate a single binned profile from a sample of input profiles, over a fixed spectral table
"""
def pre_calc_binned_prof(n_in_bin,dim_sec,idx_to_bin,resamp_mode,dx_ov_in,data_to_bin,edge_bins_resamp,nocov=False,tab_delete=None):
    
    #Pre-processing
    #    - it is necessary to do this operation first to process undefined flux and weight values
    weight_exp_all = np.zeros([n_in_bin]+dim_sec,dtype=float)     
    flux_exp_all = np.zeros([n_in_bin]+dim_sec,dtype=float)*np.nan 
    if not nocov:cov_exp_all = np.zeros(n_in_bin,dtype=object)
    else:cov_exp_all=None
    cond_def_all = np.zeros([n_in_bin]+dim_sec,dtype=bool) 
    cond_undef_weights = np.zeros(dim_sec,dtype=bool)  
    for isub,idx in enumerate(idx_to_bin):
        
        #Resampling
        if resamp_mode is not None:
            if not nocov:flux_exp_all[isub],cov_exp_all[isub] = bind.resampling(edge_bins_resamp, data_to_bin['edge_bins'][isub], data_to_bin['flux'][isub] , cov = data_to_bin['cov'][isub] , kind=resamp_mode)   
            else:flux_exp_all[isub] = bind.resampling(edge_bins_resamp, data_to_bin['edge_bins'][isub], data_to_bin['flux'][isub] , kind=resamp_mode)   
            weight_exp_all[isub] = bind.resampling(edge_bins_resamp, data_to_bin['edge_bins'][isub], data_to_bin['weights'][isub], kind=resamp_mode)  
            cond_def_all[isub] = ~np.isnan(flux_exp_all[isub])            
        else:
            flux_exp_all[isub]= data_to_bin[idx]['flux']
            if not nocov:cov_exp_all[isub]= data_to_bin[idx]['cov']
            weight_exp_all[isub]= data_to_bin[idx]['weight']
            cond_def_all[isub]  = data_to_bin[idx]['cond_def']

        #Set undefined pixels to 0 so that they do not contribute to the binned spectrum
        #    - corresponding weight is set to 0 as well so that it does not mess up the binning if it was undefined
        flux_exp_all[isub,~cond_def_all[isub]] = 0.        
        weight_exp_all[isub,~cond_def_all[isub]] = 0.

        #Pixels where at least one profile has an undefined or negative weight (due to interpolation) for a defined flux value
        cond_undef_weights |= ( (np.isnan(weight_exp_all[isub]) | (weight_exp_all[isub]<0) ) & cond_def_all[isub] )

    #Defined bins in binned spectrum
    #    - a bin is defined if at least one bin is defined in any of the contributing profiles
    cond_def_binned = np.sum(cond_def_all,axis=0)>0  

    #Disable weighing in all binned profiles for pixels validating at least one of these conditions:
    # + 'cond_null_weights' : pixel has null weights at all defined flux values (weight_exp_all is null at undefined flux values, so if its sum is null in a pixel 
    # fulfilling cond_def_binned it implies it is null at all defined flux values for this pixel)
    # + 'cond_undef_weights' : if at least one profile has an undefined weight for a defined flux value, it messes up with the weighted average     
    #    - in both cases we thus set all weights to a common value (arbitrarily set to unity for the pixel), ie no weighing is applied
    #    - pixels with undefined flux values do not matter as their flux has been set to 0, so they can be attributed an arbitrary weight
    cond_null_weights = (np.sum(weight_exp_all,axis=0)==0.) & cond_def_binned
    weight_exp_all[:,cond_undef_weights | cond_null_weights] = 1.

    #Global weight table
    #    - pixels that do not contribute to the binning (eg due to planetary range masking) have null flux and weight values, and thus do not contribute to the total weight
    #    - weight tables only depend on each original exposure but their weight is specific to the new exposures and the original exposures it contains
    dx_ov = np.ones([n_in_bin]+dim_sec,dtype=float) if dx_ov_in is None else dx_ov_in[:,None,None] 
    glob_weight_all = dx_ov*weight_exp_all

    #Total weight per pixel and normalization
    #    - normalization is done along the bin dimension, for each pixel with at least one defined contributing exposure
    glob_weight_tot = np.sum(glob_weight_all,axis=0)
    glob_weight_all[:,cond_def_binned]/=glob_weight_tot[cond_def_binned]

    return flux_exp_all,cov_exp_all,cond_def_all,glob_weight_all,cond_def_binned

def calc_binned_prof(idx_to_bin,nord,dim_exp,nspec,data_to_bin_in,inst,n_in_bin,cen_bins_exp,edge_bins_exp,dx_ov_in=None):

    #Clean weights
    #    - in all calls to the routine, exposures contributing to the master are already defined / have been resampled on a common spectral table
    flux_exp_all,cov_exp_all,cond_def_all,glob_weight_all,cond_def_binned = pre_calc_binned_prof(n_in_bin,dim_exp,idx_to_bin,None,dx_ov_in,data_to_bin_in,None,tab_delete=cen_bins_exp)

    #Tables for new exposure
    data_bin={'cen_bins':cen_bins_exp,'edge_bins':edge_bins_exp} 
    data_bin['flux'] = np.zeros(dim_exp,dtype=float)*np.nan
    data_bin['cond_def'] = np.zeros(dim_exp,dtype=bool) 
    data_bin['cov'] = np.zeros(nord,dtype=object)     

    #Calculating new exposures order by order
    for iord in range(nord):
        flux_ord_contr=[]
        cov_ord_contr=[]
        for isub,iexp in enumerate(idx_to_bin):   
 
            #Co-addition to binned profile                       
            #    - binned flux density writes as 
            # fnew(p) = sum(exp i, f(i,p)*w_glob(i,p) )/sum(exp i, w_glob(i,p))
            #      the corresponding error (when there is no covariance) writes as
            # enew(p) = sqrt(sum(exp i,  e(i,p)^2*w_glob(i,p)^2 ) )/sum(exp i, w_glob(i,p))                               
            #      for exposure i, we define the weight on all bins p in a given order as 
            # w_glob(i,p) = weights(i,p)*ov(i)   
            #      where weights are specific to the type of profiles that are binned
            #            ov(i), in units of x, is the fraction covered by exposure i in the new exposure along the bin dimension
            #      the binned flux density can thus be seen as the weighted number of contributing counts from each exposure, f(i,p)*ov(i), divided by the cumulated width of the contributing fractions of bins, sum(ov(i))
            #    - undefined pixels have w(i,p) = 0 so that they do not contribute to the new exposure
            #    - weights approximate the true error on the profiles
            # w_glob(i,p)  = a(p)/e_true(i,p)^2
            #      where a(p) is a factor independent of time that does not contribute to the weighing 
            #      thus we can define here the weight table to be associated to a binned profile as:
            # wnew(p) = 1/enew_true(p)^2
            #         = sum(exp i, w_glob(i,p))^2 /sum(exp i, e_true(i,p)^2*w_glob(i,p)^2 )
            #         = sum(exp i, w_glob(i,p))^2 /sum(exp i, w_glob(i,p) )   
            #         = sum(exp i, w_glob(i,p)) 
            #       where only the variance is considered when defining the spectral weight profiles
            #    - weights account for an original exposure duration (which is independent of the overlap Dxi) through e_true(i,p), ie that the same flux density measured over a longer exposure will weigh more     
            if (True in cond_def_all[isub,iord]):                

                #Weighted profiles
                #    - undefined pixels have null weights and have been set to 0, thus their weighted value is set to 0 and do not contribute to the master
                flux_temp,cov_temp = bind.mul_array(flux_exp_all[isub,iord],cov_exp_all[isub][iord],glob_weight_all[isub,iord])
                flux_ord_contr+=[flux_temp]
                cov_ord_contr+=[cov_temp]
                
                #Defined bins in master
                #    - a bin is defined if at least one bin is defined in any of the contributing exposures
                data_bin['cond_def'][iord] |= cond_def_all[isub,iord]
 
        #Co-addition of weighted profiles from all contributing exposures
        if len(flux_ord_contr)>0:data_bin['flux'][iord],data_bin['cov'][iord] = bind.sum(flux_ord_contr,cov_ord_contr)
        else:data_bin['cov'][iord]=np.zeros([1,nspec])          

        ### End of loop on exposures contributing to master in current order    

    ### End of loop on orders                

    #Set undefined pixels to nan
    #    - a pixel is defined if at least one bin is defined in any of the contributing exposures
    data_bin['flux'][~data_bin['cond_def']]=np.nan

    return data_bin



'''
Resample table values upon a new table
    - original and final bins can be discontinuous and of different sizes
      original bins can overlap and be given as a single input table,
      input tables must have the same stucture and dimensions, except along the binned dimension 
    - this resampling assumes that the flux density is constant over each pixel, ie that the same number of photons is received by every portion of the pixel
      in practice this is not the case, as the flux may vary sharply from one pixel to the next, and thus the flux density varies over a pixel (as we would measure at a higher spectral resolution)
      the advantage of the present resampling is that it conserves the flux, which is not necessarily the case for interpolation
    - nan values can be let in the input tables, so that new pixels that overlap with them will be set conservatively to nan
      they can also be removed before input to speed up calculation, in which case the new pixel will be defined based on defined, overlapping pixels
    - if a new pixel is only partially covered by input pixels, its boundaries can be re-adjusted to the maximum overlapping range
    - the x tables must not contain nan so that they can be sorted
      the x tables must correspond to the dimension set by dim_bin
    - multiprocessing takes longer than the standard loop, even for tens of exposures and the full ESPRESSO spectra
'''  
def resample_func(x_bd_low_in,x_bd_high_in,x_low_in_all,x_high_in_all,flux_in_all,err_in_all,remove_empty=True,dim_bin=0,cond_olap=1e-14,cond_def_in_all=None,multi=False,adj_xtab=True):
    
    #Input data provided as a single table
    calc_cond_def = True if cond_def_in_all is None else False
    if not False:

        #Shape of input data
        n_all = 1
        dim_in = flux_in_all.shape

        #Artificially creating a single-element list to use the same structure as with multiple input elements
        cond_def_in_all=[cond_def_in_all]
        x_low_in_all = [x_low_in_all]
        x_high_in_all = [x_high_in_all]
        flux_in_all = [flux_in_all]
        err_in_all = [err_in_all]
 
    #Input data provided as separate elements        
    else:
        
        #Shape of input data
        #    - only the dimensions complementary to the binned dimension will be used, and must be common to all input elements
        n_all = len(flux_in_all)
        dim_in = flux_in_all[0].shape

    #Set conditions
    if err_in_all[0] is not None:
        calc_err = True 
    else:
        calc_err = False 
        err_bin_out = None

    #Properties of binned table
    #    - we define the total number of photons in the new table, over all input elements, and the associated error 
    #    - the output binned tables are initialized so that the binned dimension is along the last axis, and the same slices can thus be used for all dimensions within the routine        
    xbin_low_temp = np.tile(x_bd_low_in,[n_all,1])
    xbin_high_temp  = np.tile(x_bd_high_in,[n_all,1])
    dxbin_out=x_bd_high_in-x_bd_low_in
    n_xbins=len(dxbin_out)
    n_dim=len(dim_in)
    if n_dim==1:
        dim_bin=0
        dim_loc_out = [n_xbins]
        ax_trans = None
    elif n_dim==2:
        if dim_bin==0: 
            dim_loc_out = [dim_in[1],n_xbins]
            ax_trans = (1,0)  #transpose from nbin,ny to ny,nbin
            ax_detrans = (1,0)
        elif dim_bin==1:            
            dim_loc_out = [dim_in[0],n_xbins] 
            ax_trans = None
    elif n_dim==3:
        if dim_bin==0:
            dim_loc_out = [dim_in[1],dim_in[2],n_xbins]
            ax_trans = (1,2,0)  #transpose from nbin,ny,nz to ny,nz,nbin
            ax_detrans = (2,0,1)
        else:
            stop('Binning routine to be coded')
    ax_sum = tuple(icomp for icomp in range(n_dim-1))
    dx_ov_cont_tot_bins=np.zeros(n_xbins)
    count_in_bins=np.zeros(dim_loc_out) 
    flux_bin=np.zeros(dim_loc_out)
    flux_bin_out=np.zeros(dim_loc_out)*np.nan 
    if calc_err:
        err2_bin=np.zeros(dim_loc_out) 
        err_bin_out=np.zeros(dim_loc_out)*np.nan 

    #Process each input element separately
    for iloc,(x_low_in_loc,x_high_in_loc,flux_in_loc,err_in_loc,cond_def_in_loc) in enumerate(zip(x_low_in_all,x_high_in_all,flux_in_all,err_in_all,cond_def_in_all)):

        #Transpose input arrays when binned dimension is not along last axis
        if ax_trans is not None:
            flux_in_loc = np.transpose(flux_in_loc,axes=ax_trans)   #from nbin,ny to ny,nbin
            if calc_err==True:err_in_loc = np.transpose(err_in_loc,axes=ax_trans)
            if calc_cond_def == False:cond_def_in_loc = np.transpose(cond_def_in_loc,axes=ax_trans)

        #Sort input data over binned dimension 
        #    - flux_in has been transposed whenever relevant so that the axis to be binned is at the last dimension
        #    - this is necessary for searchsorted to work later on 
        mid_x_in=0.5*(x_low_in_loc+x_high_in_loc) 
        if (True in np.isnan(mid_x_in)):stop('Table along binned dimension must not contain nan')
        id_sort=np.argsort(mid_x_in)
        x_low=x_low_in_loc[id_sort]
        x_high=x_high_in_loc[id_sort]
        flux_in=np.take(flux_in_loc,id_sort,axis=n_dim-1)
        if calc_err:err_in=np.take(err_in_loc,id_sort,axis=n_dim-1) 
        
        #Defined pixels in input table
        #    - we also identify defined pixels along the dimension to be binned
        #      a pixel along this dimension is considered undefined if no pixel along any other dimension is defined
        if calc_cond_def:
            cond_def_in = ~np.isnan(flux_in)  
        else:
            cond_def_in = np.take(cond_def_in_loc,id_sort,axis=n_dim-1)                     
        cond_def_in = np.sum(cond_def_in,axis=ax_sum,dtype=bool)

        #Index of pixels within the boundary of the new table
        #    - we use the first and last defined pixels to keep potentially undefined pixels in between
        #    - beyond those pixels, there are no defined pixels
        cond_sup = x_high[cond_def_in]>=x_bd_low_in[0]
        cond_inf = x_low[cond_def_in]<=x_bd_high_in[-1] 
        if (True in cond_sup) and (True in cond_inf):
            idxcond_def_in = np_where1D(cond_def_in)
            idx_st_withinbin = idxcond_def_in[ np_where1D(cond_sup)[0] ]
            idx_end_withinbin = idxcond_def_in[ np_where1D(cond_inf)[-1] ]
            idx_kept = np.arange(idx_st_withinbin,idx_end_withinbin+1)
            
            #Remove input data beyond boundaries of new table, and beyond the most extreme defined pixels, to speed up calculation  
            #    - flux_in has been transposed whenever relevant so that the axis to be binned is at the last dimension
            x_low=x_low[idx_kept]
            x_high=x_high[idx_kept]
            dx_in=x_high-x_low 
            flux_in=np.take(flux_in,idx_kept,axis=n_dim-1)
            if calc_err==True:err_in2=np.take(err_in,idx_kept,axis=n_dim-1)**2.             

            #Identify bins overlapping with the most extreme defined pixels of the input table
            #    - bins beyond those pixels cannot be defined, and will remain unprocessed/empty
            cond_bin_proc = (x_bd_high_in>=x_low[0]) & (x_bd_low_in<=x_high[-1])
            idx_bin_proc = np.arange(n_xbins)[cond_bin_proc]
            x_bd_low_proc = x_bd_low_in[cond_bin_proc]
            x_bd_high_proc = x_bd_high_in[cond_bin_proc]
       
            #--------------------------------------------------------------------------------------------------------

            #Process new bins
            for isub_bin,(ibin,xbin_low_loc,xbin_high_loc) in enumerate(zip(idx_bin_proc,x_bd_low_proc,x_bd_high_proc)):   
           
                #Indexes of all original pixels overlapping with current bin
                #    - flux_in has been transposed whenever relevant so that the axis to be binned is at the last dimension
                #    - we use 'where' rather than searchsorted to allow processing original bins that overlap together
                idx_overpix = np_where1D( (x_high>=xbin_low_loc) &  (x_low <=xbin_high_loc) )
                
                #Process bins if original pixels overlap
                n_overpix = len(idx_overpix)
                if n_overpix>0:
                    flux_overpix=np.take(flux_in,idx_overpix,axis=n_dim-1)
                    if calc_err==True:err_overpix2=np.take(err_in2,idx_overpix,axis=n_dim-1)
     
                    #Checking that overlapping pixels are all defined
                    #    - the condition is that pixels in the dimensions complementary to that binned, for which all overlapping pixels (along the binned dimension) are defined 
                    #    - if at least one pixel is undefined, we consider conservatively that the new pixel it overlap is undefined as well
                    #      new bins have been initialized to nan, and thus remain undefined unless processed
                    #    - for n_dim>1, the new bin is processed if there is at least one position along the complementary dimensions for which all overlapping pixels are defined
                    #    - this condition is applied for one of the input element, but if at least one input element has defined pixels overlapping with the new pixel, it will be defined
                    #    - the binned axis has been placed as last dimension, and cond_defbin_compdim has thus the dimensions of flux_overpix along the other axis (kept in the original order) 
                    cond_defbin_compdim = (np.prod(~np.isnan(flux_overpix),axis=n_dim-1)==1) 
    
                    #Process if original overlapping pixels are defined
                    #    - cond_defbin_compdim is a single boolean for ndim=1, an array otherwise, so that "if True in cond_defbin_compdim" cannot be used
                    #    - each pixel overlapping with the bin can either be:
                    # + fully within the bin (both pixels boundaries are within the bin)
                    # + fully containing the bin (both pixels boundaries are outside of the bin)
                    # + containing one of the two bin boundaries (the upper - resp lower - pixel boundary is outside the bin)                
                    if np.sum(cond_defbin_compdim)>0:    
    
                        #Minimum between overlapping pixels upper boundaries and new bin upper boundary
                        #    - if the pixel upper boundary is beyond the bin, then the pixel fraction beyond the bin upper boundary will not contribute to the binned flux 
                        #    - if the pixel upper boundary is within the bin, then the pixel will only contribute to the binned flux up to its own boundary
                        x_high_ov_contr=np.minimum(x_high[idx_overpix],np.repeat(xbin_high_loc,n_overpix))
                    
                        #Maximum between overlapping pixels lower boundaries and new bin lower boundary
                        #    - if the pixel lower boundary is beyond the bin, then the pixel fraction beyond the bin upper boundary will not contribute to the binned flux 
                        #    - if the pixel lower boundary is within the bin, then the pixel will only contribute to the binned flux up to its own boundary
                        x_low_ov_contr=np.maximum(x_low[idx_overpix],np.repeat(xbin_low_loc,n_overpix))
                    
                        #Width over which each original pixel contribute to the binned flux
                        dx_ov_cont=x_high_ov_contr-x_low_ov_contr
                        
                        #Co-add total effective overlap of original pixels to current bin
                        #    - overlap are cumulated over successive input elements 
                        #    - this can be done because original pixels do not overlap
                        #    - the effective overlaps of successive input elements are averaged afterwards via count_in_bins
                        dx_ov_cont_tot_bins[ibin]+=np.sum(dx_ov_cont)
                    
                        #Store local bin boundaries
                        #    - if all overlapping pixels have their upper (resp lower) boundary lower (resp higher) than that of the bin, then the bin 
                        # is restricted to the maximum (resp minimum) pixels boundary
                        #    - because this redefinition may vary for the different input elements, we store the updated boundaries for each element and make a final definition afterwards                         
                        xbin_low_temp[iloc,ibin] = min(x_low_ov_contr)
                        xbin_high_temp[iloc,ibin] = max(x_high_ov_contr)
    
                        #Slices suited to all dimensions
                        #    - cond_defbin_compdim equals True for ndim=1, and will not count in the call to the various tables
                        idx_mdim = (cond_defbin_compdim,ibin)
    
                        #Implement counter
                        #    - we set to 1 the new pixels along the complementary dimensions for which all original overlapping pixels are defined
                        #      only those pixels contribute to the new bins, but which pixels contribute can change with the input element
                        #      the counter is thus used to average values of the new bins, along all complementary dimensions, over the input elements that contributed to each dimension
                        count_in_bins[idx_mdim]+=1
    
                        #Add contribution of original overlapping pixels to new pixel
                        #    - flux (=spectral photon density) is assumed to be constant over a given pixel:
                        # F(pix) = gcal* N(pix) / dpix = N(dx)/dx
                        #      where gcal is the instrumental calibration of photoelectrons
                        #    - the flux density in a bin is the total number of photons from the overlapping fractions of all pixels, divided by the effective overlapping surface (see below)
                        # F(bin) = gcal*sum( overlapping pixels, N(pixel overlap) )/sum( overlapping pixels, dx(pixel overlap) )
                        #       since F(pixel) = gcal*N(pixel overlap)/dx(pixel overlap) we obtain
                        # F(bin) = sum( overlapping pixels, F(pixel)*dx(pixel overlap) )/sum( overlapping pixels, dx(pixel overlap) )
                        #    - because there can be gaps between two original pixels, we average their contributed number of photons by the effective width they cover
                        #      imagine a new pixel between x=0 and x=1, covered by two original pixels extending up to x=1/3 for the first one, and starting from x=2/3 for the second one
                        #      both original pixels have a flux of 1, and thus contribute 1/3 photons over their overlapping range
                        #      normalizing by the width of the new pixel would yield an incorrect flux of (1/3+1/3)/1 = 2/3
                        #      we thus normalize by the effective overlap = 2/3 
                        #    - for ndim = 1, flux_overpix[cond_defbin_compdim] = flux_overpix[True] returns an array (1,len(cond_defbin_compdim))
                        #      its product with dx_ov_cont must thus be summed over axis=1 
                        flux_bin[idx_mdim]+=np.sum(flux_overpix[cond_defbin_compdim]*dx_ov_cont,axis=1)
                        if calc_err: err2_bin[idx_mdim]+=np.sum(err_overpix2[cond_defbin_compdim]*dx_ov_cont*dx_in[idx_overpix],axis=1)

    #----------------------------------------------------------------------------      

    #Redefinition of the bin boundaries
    #    - the upper (resp. lower) boundaries are set to the maximum (resp. min) of the boundaries defined for each input element
    if adj_xtab==True:
        xbin_low_out = np.amin(xbin_low_temp,axis=0) 
        xbin_high_out = np.amax(xbin_high_temp,axis=0) 
    else:
        xbin_low_out = x_bd_low_in
        xbin_high_out= x_bd_high_in

    #Average flux over all contributing pixels and input element
    #    - we define new bins only if they are filled in at least one contributing input element
    #      count_in_bins is summed along all dimensions complementary to the binned dimension
    #    - the number of photons in a given bin, obtained by co-adding the number of photons from the overlapping pixels of each successive input element, is 
    # converted into a flux density by normalizing with the total width of overlap from these contributing pixels
    #    - empty bins remain set to nan
    #    - the binned table have dimensions fbin = (n0,n1 .. nn)
    #      the overlap table dx_ov has dimension ni along one of these axes
    #      there is no need to create a new axis to perform fbin/dx_ov if the size of dx_ov is that of the last dimension in fbin
    #    - the 'axes = (i,j,k,..)' field in transpose() means that axe i -> axe 0, axe j -> axe 1  
    cond_filled=(np.sum(count_in_bins,axis=ax_sum)>0.)
    if (True in cond_filled):
        if (n_dim==1):
            flux_bin_out[cond_filled]=flux_bin[cond_filled]/dx_ov_cont_tot_bins[cond_filled]      
            if calc_err==True:err_bin_out[cond_filled]=np.sqrt(err2_bin[cond_filled])/dx_ov_cont_tot_bins[cond_filled]
        else:
     
            #Normalization of defined pixels
            #    - we apply the operation to defined pixels only along the binned dimension (placed along last axis) and to all pixels in complementary dimensions (called with slice())
            ax_slice = tuple(slice(icomp) for icomp in np.array(dim_loc_out)[0:-1])+(cond_filled,)
            flux_bin_out[ax_slice]=flux_bin[ax_slice]/dx_ov_cont_tot_bins[cond_filled]
            if calc_err==True:err_bin_out[ax_slice]=np.sqrt(err2_bin[ax_slice])/dx_ov_cont_tot_bins[cond_filled]                    
                    
            #Transpose back arrays to original dimensions of input arrays, if relevant 
            #    - binned table currently have the binned axis as last dimension
            if ax_trans is not None:
                flux_bin_out = np.transpose(flux_bin_out,axes=ax_detrans)        
                if calc_err==True:err_bin_out = np.transpose(err_bin_out,axes=ax_detrans)

    #Remove empty bins if requested
    #      bins are removed only if undefined along all complementary dimensions
    if (remove_empty==True) and (False in cond_filled):   
        xbin_low_out=xbin_low_out[cond_filled]
        xbin_high_out=xbin_high_out[cond_filled] 
        idx_cond_filled = np_where1D(cond_filled)               
        flux_bin_out=np.take(flux_bin_out,idx_cond_filled,axis=dim_bin)
        if calc_err==True:err_bin_out=np.take(err_bin_out,idx_cond_filled,axis=dim_bin)            
    xbin_out=0.5*(xbin_low_out+xbin_high_out) 
    dxbin_out = xbin_high_out - xbin_low_out

    return xbin_low_out,xbin_high_out,xbin_out,dxbin_out,flux_bin_out,err_bin_out  















'''
Routine to extract local residual stellar profiles
    - defined in the stellar rest frame 
      if a bin is undefined in the master and/or the exposure, it will be undefined in the local profile

    - the real measured spectrum can be written as 
 F(w,t,v) = Fstar(w,t,v) * delt_p(w,t,v)  
      but it can also be decomposed spatially as 
 F(w,t,v) = sum(i non occulted, Fi(w,t,v)) + fp(w,v)*(Socc(t) - Sthick(band,t) - Sthin(w,t) ) + Fp(w,t)      
          = Fstar(w,t,v) - fp(w,v)*( Sthick(band,t) + Sthin(w,t) ) + Fp(w,t) 
      with :
 + w the absolute wavelength in the stellar rest frame, within a given spectral band 
 + F the flux received from the star - planet system [erg s-1 A-1 cm-2] at a given distance
   Fi(w,v) = fi(w,v)*Si is the flux emitted by the region i of surface Si
 + the surface density flux can be written as fi(w,v) = Ii(w)*LDi(w)*GDi(w) 
   Ii(w) is the specific intensity in the direction of the LOS [erg cm-2 s-1 A-1 sr-1]
         can be written as I0(w-wstar(t)) if the local stellar emission is constant over the stellar disk, and simply shifted by wstar(t) because of surface velocity
   LDi(w) is the spectral limb-darkening law
   GDi(w) the gravity-darkening law
 + Fp(w,t) is the flux emitted or reflected by the planet                
   the planet and its atmosphere occult a surface Socc(t), time-dependent because of partial occultation at ingress/egress    
   Sthick(band,t) is the equivalent surface of the planet disk opaque to light in the local spectral band 
   Sthin(w,t) is the equivalent surface of the atmospheric annulus optically thin to light in the band, varying at high frequency and null outside of narrow absorption lines  
   Fp(w,t) and Sthin(w,t) are sensitive to the planet orbital motion, and shifted in the star rest frame by wpl(t)
  
    - the measured profiles are now defined in the most general case as (see rescale_data) : 
 Fsc(w,t,v) = F(w,t,v)*Cref(wband,v)
      with F(w,t,v) the true spectrum
      since all spectra were set to the same balance, corresponding to the stellar spectrum times a low-frequency coefficient, the master out corresponds in a given visit to 
 MFstar(w,v) = Fstar(w,v)*Cref(wband,v)  
      note that the unknown scaling of the flux due to the distance to the star is implicitely included in Cref
      where we can decompose Fstar as: 
 Fstar(w,v) = sum(i non occulted, Fi(w,v)) + fp(w,v)*Socc(t)  
          
    - the local residual profiles are calculated as :
 F_res(w,t,v) = MFstar(w,v) - Fsc(w,t,v)                
              = Fstar(w,v)*Cref(wband,v)  - F(w,t,v)*Cref(wband,v)
              = ( Fstar(w,v) - F(w,t,v) )*Cref(wband;v)
              = ( Fstar(w,v) - Fstar(w,t,v) + fp(w,v)*( Sthick(band,t) + Sthin(w,t) ) - Fp(w,t)  )*Cref(wband;v)
      here we make the assumption that Fstar(w,t,v) ~ Fstar(w,v), but care must be taken not to neglect uncertainties on Fstar(w,t,v) when propagating errors, even if uncertainties on the reference Fstar(w,v) can be neglected
              = ( fp(w,v)*( Sthick(band,t) + Sthin(w,t) ) -  Fp(w,t) )*Cref(wband,v)
              = ( fp(w,v)*Sp(w,t) -  Fp(w,t) )*Cref(wband,v)
      where Sp(w,t) represent the equivalent surface occulted by the opaque planetary disk and its optically thin atmosphere, at each wavelength

    - if there is no contribution from the atmosphere, or its contamination is excluded :
 F_res(w,t,v) = fp(w,v)*Sthick(band,t)*Cref(wband,v) 
      CCFs computed on these F_res have the same contrast, FWHM, and RV between visits, as long as the intrinsic lines are comparable
    
    - it is possible that Fstar(w,v) varies with time, during or outside of the transit, eg if a spot/plage is present on the star and changes during the observations (in particular with the star's rotation)
      in that case, rather than MFstar(w,v) we would need to use a Fstar(w,t,v) representative of the star at the time of each exposure

    - binned disk-integrated profiles are calculated from spectra resampled directly on the table of each exposure, to avoid blurring spectral features, losing resolution, and introducing spurious features
 when doing differences and ratio between the master and individual spectra (which we found to be the case if using a single master calculated on a common table and then resampled on 
 the table of each exposure)
'''
def extract_res_profiles(gen_dic,data_dic,inst,vis,data_prop,coord_dic):

    print('   > Extracting residual profiles')
    data_inst = data_dic[inst]
    data_vis = data_inst[vis]
    
    #Current rest frame
    if data_dic['DI'][inst][vis]['rest_frame']!='star':print('WARNING: disk-integrated profiles must be aligned')
    data_dic['Res'][inst][vis]['rest_frame'] = 'star'

    #Path to initialized local data
    proc_gen_data_paths_new=gen_dic['save_data_dir']+'Res_data/'+inst+'_'+vis+'_'

    #Exposures for which local profiles will be extracted
    #    - the user can request extraction for in-transit exposures alone (to avoid computing time)
    #      we force the extraction for all exposures if a common master is used for the extraction (ie, when exposures are resampled on a common table) and no time is required to recalculate the master for each exposure
    if data_dic['Res']['extract_in'] and ('spec' in data_dic['Res']['type'][inst]) and (not data_vis['comm_sp_tab']):data_dic['Res'][inst][vis]['idx_to_extract'] = deepcopy(gen_dic[inst][vis]['idx_in'])
    else:data_dic['Res'][inst][vis]['idx_to_extract'] =  np.arange(data_vis['n_in_visit'],dtype=int) 

    data_dic['Res'][inst][vis]['idx_to_extract'] = np.arange(27,89)
    print('ATTENTION')

    #Calculating
    if (gen_dic['calc_res_data']):
        print('         Calculating data')     

        #Phase range from which original exposures contributing to the master are taken
        #    - we impose that a single master be used    
        bin_prop = {'bin_low':[-0.5],'bin_high':[0.5]}  

        #Binning mode
        #   - using current visit exposures only, or exposures from multiple visits
        if (inst in data_dic['Res']['vis_in_bin']) and (len(data_dic['Res']['vis_in_bin'][inst])>0):
            mode='multivis'
            vis_to_bin = data_dic['Res']['vis_in_bin'][inst]  
            
            #Planets associated with the binned visits
            transit_pl = []
            for vis_bin in vis_to_bin:
                transit_pl+=data_inst[vis_bin]['transit_pl']
                if data_dic[inst][vis_bin]['type']!=data_vis['type']:stop('Binned disk-integrated profiles must be of the same type as processed visit')
            transit_pl = list(np.unique(transit_pl))            
            
        else:
            mode=''
            vis_to_bin = [vis]   
            transit_pl=data_vis['transit_pl']

        #Automatic definition of reference planet for single-transiting planet  
        #    - for multiple planets the reference planet for the phase does not matter, we just requie that all exposures are selected and then the selection is done via their indexes
        bin_prop['ref_pl'] = transit_pl[0]  
        if (len(transit_pl)>1) and (('pl_in_bin' in data_dic['DI']) and (inst in data_dic['DI']['pl_in_bin'])):bin_prop['ref_pl'] = data_dic['DI']['pl_in_bin'][inst]

        #Initialize binning
        #    - output tables contain a single value, associated with the single master (=binned profiles) used for the extraction 
        _,_,_,_,n_in_bin_all,idx_to_bin_all,dx_ov_all,_,idx_bin2orig,idx_bin2vis,idx_to_bin_unik = init_bin_rout('DI',bin_prop,data_dic['Res']['idx_in_bin'],'phase',coord_dic,inst,vis_to_bin,data_dic,gen_dic)
        scaled_data_paths_vis = {}  
        iexp_no_plrange_vis = {}
        exclu_rangestar_vis = {}
        for vis_bin in vis_to_bin:
            if gen_dic['flux_sc']:scaled_data_paths_vis[vis_bin] = data_dic[inst][vis_bin]['scaled_DI_data_paths']
            else:scaled_data_paths_vis[vis_bin] = None
            if ('DI_Mast' in data_dic['Atm']['no_plrange']):iexp_no_plrange_vis[vis_bin] = data_dic['Atm'][inst][vis_bin]['iexp_no_plrange']
            else:iexp_no_plrange_vis[vis_bin] = {}
            exclu_rangestar_vis[vis_bin] = data_dic['Atm'][inst][vis_bin]['exclu_range_star']
        
        #Retrieving data that will be used in the binning to define the master disk-integrated profile
        #    - in process_bin_prof() all profiles are resampled on the common table before being binned, thus they can be resampled when uploaded the first time
        #    - here the binned profiles must be defined on the table of each processed exposure, so the components of the weight profile are retrieved here and then either copied or resampled if necessary for each exposure
        #      here a single binned profile (the master) is calculated, thus 'idx_to_bin_unik' is the same as idx_to_bin_all, which contains a single element
        data_to_bin_gen={}    
        gdet4weight = gen_dic['cal_weight'] & data_vis['mean_gdet']
        for iexp_off in idx_to_bin_unik:
            data_to_bin_gen[iexp_off]={}

            #Original index and visit of contributing exposure
            #    - index is relative to the global table
            iexp_glob = idx_bin2orig[iexp_off]
            vis_bin = idx_bin2vis[iexp_off]

            #Latest processed disk-integrated data and associated tables
            #    - profiles should have been aligned in the star rest frame and rescaled to their correct flux level, if necessary          
            data_exp_off = dataload_npz(data_inst[vis_bin]['proc_DI_data_paths']+str(iexp_glob))
            for key in ['cen_bins','edge_bins','flux','cond_def','cov']:data_to_bin_gen[iexp_off][key] = data_exp_off[key]
            if data_vis['tell_sp']:data_to_bin_gen[iexp_off]['tell'] = dataload_npz(data_inst[vis_bin]['tell_DI_data_paths'][iexp_glob])['tell']    
            else:data_to_bin_gen[iexp_off]['tell'] = None
            if gdet4weight:data_to_bin_gen[iexp_off]['mean_gdet'] = dataload_npz(data_inst[vis_bin]['mean_gdet_DI_data_paths'][iexp_glob])['mean_gdet'] 
            else:data_to_bin_gen[iexp_off]['mean_gdet'] = None              
            
            #Master disk-integrated spectrum for weighing
            #    - profile has been shifted to the same frame as the residual profiles, but is still defined on the common table, not the table of current exposure
            #    - see process_binned_prof() for details
            data_ref = dataload_npz(data_dic[inst][vis_bin]['mast_DI_data_paths'][iexp_glob])
            data_to_bin_gen[iexp_off]['edge_bins_ref'] = data_ref['edge_bins']
            data_to_bin_gen[iexp_off]['flux_ref'] = data_ref['flux']

            #Weight profile
            #    - only calculated here on a common table if:
            # + binned profiles come from a single visit, defined on a common table for the visit
            # + binned profiles come from multiple visits, defined on a common table for all visits    
            #    - the master spectrum should be processed in the star rest frame, so that the stellar lines do not contribute to weighing         
            if ((mode=='') and data_vis['comm_sp_tab']) or ((mode=='multivis') and data_inst['comm_sp_tab']):
                flux_ref_exp = np.ones(data_dic[inst][vis_bin]['dim_exp'])  
                data_to_bin_gen[iexp_off]['weight'] = def_weights_spatiotemp_bin(range(data_inst['nord']),scaled_data_paths_vis[vis_bin],inst,vis_bin,gen_dic['corr_Fbal'],gen_dic['corr_FbalOrd'],gen_dic['save_data_dir'],gen_dic['type'],data_inst['nord'],iexp_glob,'DI',data_dic[inst]['type'],data_vis['dim_exp'],data_to_bin_gen[iexp_off]['tell'],data_to_bin_gen[iexp_off]['mean_gdet'],data_to_bin_gen[iexp_off]['cen_bins'],1.,flux_ref_exp,None,bdband_flux_sc = gen_dic['flux_sc'])
 
        #Processing each exposure of current visit selected for extraction
        iexp_proc = data_dic['Res'][inst][vis]['idx_to_extract']
        common_args = (data_vis['proc_DI_data_paths'],mode,data_vis['comm_sp_tab'],data_inst['comm_sp_tab'],proc_gen_data_paths_new,idx_to_bin_all[0],n_in_bin_all[0],dx_ov_all[0],idx_bin2orig,idx_bin2vis,data_inst['com_vis'],data_dic[inst]['nord'],data_vis['dim_exp'],data_vis['tell_sp'],data_vis['nspec'],gen_dic['flux_sc'],gdet4weight,data_to_bin_gen,gen_dic['resamp_mode'],\
                       scaled_data_paths_vis,inst,iexp_no_plrange_vis,exclu_rangestar_vis,data_dic[inst]['type'],gen_dic['type'],gen_dic['corr_Fbal'],gen_dic['corr_FbalOrd'],gen_dic['save_data_dir'])               
        if gen_dic['nthreads_res_data']>1:para_sub_extract_res_profiles(sub_extract_res_profiles,gen_dic['nthreads_res_data'],len(iexp_proc),[iexp_proc],common_args)                           
        else:sub_extract_res_profiles(iexp_proc,*common_args)    

    #Checking that local data has been calculated for all exposures
    else:
        data_paths={iexp:proc_gen_data_paths_new+str(iexp) for iexp in range(data_vis['n_in_visit'])}
        check_data(data_paths)            

    #Path to weighing master and calibration profile
    #    - residual profiles are extracted in the same rest frame as the disk-integrated master, so that it can directly be used
    #    - at this stage a single master has been defined over the common spectral table, it will be resampled in the binning routine
    #    - calibration paths are updated even if they are not used as weights, to be used in flux/count scalings
    data_vis['proc_Res_data_paths']=proc_gen_data_paths_new
    if gen_dic['DImast_weight']:data_vis['mast_Res_data_paths'] = data_vis['mast_DI_data_paths']
    if data_vis['tell_sp']:data_vis['tell_Res_data_paths'] = data_vis['tell_DI_data_paths']
    if gen_dic['flux_sc']:data_vis['scaled_Res_data_paths'] = data_vis['scaled_DI_data_paths']
    if data_vis['mean_gdet']:data_vis['mean_gdet_Res_data_paths'] = data_vis['mean_gdet_DI_data_paths']

    return None



def sub_extract_res_profiles(iexp_proc,proc_DI_data_paths,mode,comm_sp_tab_vis,comm_sp_tab_inst,proc_gen_data_paths_new,idx_to_bin_mast,n_in_bin_mast,dx_ov_mast,idx_bin2orig,idx_bin2vis,com_vis,nord,dim_exp,tell_sp,nspec,flux_sc,gdet4weight,data_to_bin_gen,resamp_mode,\
                             scaled_data_paths_vis,inst,iexp_no_plrange_vis,exclu_rangestar_vis,vis_type,gen_type,corr_Fbal,corr_FbalOrd,save_data_dir):

    #Processing each exposure of current visit selected for extraction
    #    - extraction can be limited to in-transit exposures to gain computing time, e.g if one only needs to analyze the local stellar profiles 
    for isub,iexp in enumerate(iexp_proc):        
       
        #Upload latest processed DI data from which to extract local profile
        data_exp = dataload_npz(proc_DI_data_paths+str(iexp))

        #Calculating master disk-integrated profile
        #    - the master is calculated in a given exposure:
        # + if it is the first one
        # + if it is another one and binned profiles 
        #       come from a single visit, and do not share a common table for the visit
        #       come from multiple visits, do not share a common table for all visits, and visit of the processed exposure is not the one used as reference for the common table of all visits (in which case resampling is not needed)
        if (isub==0) or ((mode=='') and (not comm_sp_tab_vis)) or ((mode=='multivis') and (not comm_sp_tab_inst)):                
            data_to_bin={}
            for iexp_off in idx_to_bin_mast:

                #Original index and visit of contributing exposure
                #    - index is relative to the global table
                iexp_glob = idx_bin2orig[iexp_off]
                vis_bin = idx_bin2vis[iexp_off]                    

                #Resampling on common spectral table if required
                #    - data is stored with the same indexes as in idx_to_bin_all
                #    - all exposures must be defined on the same spectral table before being binned
                #    - if multiple visits are used and do not share a common table, they do not need resampling if their table is the one used as reference to set the common table
                if ((mode=='') and (not comm_sp_tab_vis)) or ((mode=='multivis') and (not comm_sp_tab_inst) and (vis_bin!=com_vis)):
                    data_to_bin[iexp_off]={}
                    
                    #Resampling exposure profile
                    data_to_bin[iexp_off]['flux']=np.zeros(dim_exp,dtype=float)*np.nan
                    data_to_bin[iexp_off]['cov']=np.zeros(nord,dtype=object) 
                    tell_exp=np.ones(dim_exp,dtype=float) if tell_sp else None
                    mean_gdet_exp=np.ones(dim_exp,dtype=float) if gdet4weight else None
                    for iord in range(nord): 
                        data_to_bin[iexp_off]['flux'][iord],data_to_bin[iexp_off]['cov'][iord] = bind.resampling(data_exp['edge_bins'][iord], data_to_bin_gen[iexp_off]['edge_bins'][iord], data_to_bin_gen[iexp_off]['flux'][iord] , cov = data_to_bin_gen[iexp_off]['cov'][iord], kind=resamp_mode)                                                        
                        if tell_sp:tell_exp[iord] = bind.resampling(data_exp['edge_bins'][iord], data_to_bin_gen[iexp_off]['edge_bins'][iord], data_to_bin_gen[iexp_off]['tell'][iord] , kind=resamp_mode) 
                        if gdet4weight:mean_gdet_exp[iord] = bind.resampling(data_exp['edge_bins'][iord], data_to_bin_gen[iexp_off]['edge_bins'][iord],data_to_bin_gen[iexp_off]['mean_gdet'][iord], kind=resamp_mode)   
                        data_to_bin[iexp_off]['cond_def'] = ~np.isnan(data_to_bin[iexp_off]['flux'])   

                    #Weight definition         
                    flux_ref_exp = np.ones(dim_exp,dtype=float)
                    data_to_bin[iexp_off]['weight'] = def_weights_spatiotemp_bin(range(nord),scaled_data_paths_vis[vis_bin],inst,vis_bin,corr_Fbal,corr_FbalOrd,save_data_dir,gen_type,nord,iexp_glob,'DI',vis_type,dim_exp,tell_exp,mean_gdet_exp,data_exp['cen_bins'],1.,flux_ref_exp,None,bdband_flux_sc = flux_sc)

                #Weighing components and current exposure are defined on the same table common to the visit 
                else:data_to_bin[iexp_off] = deepcopy(data_to_bin_gen[iexp_off])  

                #Exclude planet-contaminated bins  
                #    - condition that 'DI_Mast' is in 'no_plrange' is included in the definition of 'iexp_no_plrange_vis'
                if (iexp_glob in iexp_no_plrange_vis[vis_bin]):
                    for iord in range(nord):                   
                        data_to_bin[iexp_off]['cond_def'][iord] &=  excl_plrange(data_to_bin[iexp_off]['cond_def'][iord],exclu_rangestar_vis[vis_bin],iexp_off,data_exp['edge_bins'][iord],vis_type)[0]

            #Calculate master on current exposure table
            data_mast = calc_binned_prof(idx_to_bin_mast,nord,dim_exp,nspec,data_to_bin,inst,n_in_bin_mast,data_exp['cen_bins'],data_exp['edge_bins'],dx_ov_in = dx_ov_mast)

        #Extracting residual stellar profiles  
        #    - the master is defined for each individual exposures if they are defined on different spectral table
        #      otherwise defined on a single common spectral table, in which case we repeat the master to have the same structure as individual exposures          
        data_loc = {'cen_bins':data_exp['cen_bins'],
                    'edge_bins':data_exp['edge_bins'],
                    'flux' : np.zeros(dim_exp, dtype=float),
                    'cov' : np.zeros(nord, dtype=object)}
        for iord in range(nord):
            data_loc['flux'][iord],data_loc['cov'][iord]=bind.add(data_mast['flux'][iord], data_mast['cov'][iord], -data_exp['flux'][iord], data_exp['cov'][iord])                 
        data_loc['cond_def'] = ~np.isnan(data_loc['flux'])       

        #Saving data
        #    - saved for each exposure, as the files are too large otherwise                
        np.savez_compressed(proc_gen_data_paths_new+str(iexp),data=data_loc,allow_pickle=True)    
    
    return None


def para_sub_extract_res_profiles(func_input,nthreads,n_elem,y_inputs,common_args):   
    pool_proc = Pool(processes=nthreads)   #cannot be passed through lmfit
    ind_chunk_list=init_parallel_func(nthreads,n_elem)
    chunked_args=[(y_inputs[0][ind_chunk[0]:ind_chunk[1]],)+common_args for ind_chunk in ind_chunk_list]	
    all_results=tuple(tab for tab in pool_proc.starmap(func_input,chunked_args))	
    pool_proc.close()
    pool_proc.join() 
    return None









'''
Routine to process individual intrinsic stellar profiles
     - the in-transit residual spectra, at wavelengths where planetary contamination was masked, correspond to:  
  F_res(w in band,t,v) = ( fp(w,v)*S(w,t) -  Fp(w,t) )*Cref(band,v)  
                       = fp(w,v)*Sthick(band,t)*Cref(band,v)   
       with fp the surface density flux spectrum, assumed spatially constant over the region occulted by the planet
               known to a scaling factor Cref(band,v), assumed to be constant over the band, and accounting for the absolute flux level and deviations from the stellar SED
       Sthick(band,p) the effective planet surface occulting the star in the band, assumed spectrally constant over the band, varying spatially during ingress/egress, and set by our choice of light curve 
       the above expression consider only wavelengths w where there are no narrow absorption lines from the planetary atmosphere (absorbed wavelengths have been masked)
    - we scale back residual spectra to get back to the intrinsic stellar profiles (ie, without broadband planetary absorption and limb/grav-darkening), assuming that 
  fp(w in band,vis) = I(w in band,t)*LD(band,t) 
       ie that the limb-darkening has low-frequency variations and does not affect the shape of the local intrinsic spectra
       the theoretical light curve used to rescale the data writes as (see rescale_data()) :
  1 - LC_theo(band,t) = LDp(band,t)*Sp(band,t)/Sstar_LD(band)   
       where the fluxes are constant and spatially uniform over the band and stellar disk, and the planet is described by a constant, mean radius over the band
       if we normalize the local spectra by this factor, we obtain the intrinsic spectra as
  F_intr(w,t,vis) = F_res(w in band,t,vis)/(1 - LC_theo(band,t))
                  = fp(w,vis)*Sthick(band,t)*Cref(band,v)/(1 - LC_theo(band,t))
                  = I(w,t)*LD(band,t)*Sthick(band,t)*Cref(band,v)*Sstar_LD(band)/(LD(band,t)*Sp(band,t))  
                  = I(w,t)*Sthick(band,t)*Cref(band,v)*Sstar_LD(band)/Sp(band,t) 
       during the full transit, the ratio Sr(band) = Sthick(band,t)/Sp(band,t) is constant over time. During ingress/egress, we assume that the ratio remains the same, so that :
  F_intr(w,t,v) = I(w,t)*Fnorm_ref(band)
       with Fnorm_ref(band) = Sr(band)*Cref(band,v)*Sstar_LD(band) only dependent on the band
       and the normalized local spectra then allow comparing the shape of the intrinsic spectra along the transit chord in a given band
            
     - the continuum of the residual profiles is :   
  F_res(cont,t,v) = sum(w in cont , Ip(w,t)*LDp(band,t)*Sthick(band,t)*Cref(band,v)*dw(w,v))   
                  = sum(w in cont , Ip(w,t)*dw(w,v)) * LDp(band,t)*Sthick(band,t)*Cref(band,v) 
                  = Ip(cont,v)*LDp(band,t)*Sthick(band,t)*Cref(band,v)   
        if we define the continuum as a spectral range where the flux only varies due to broadband limb-darkening, we have Ip(cont,v) = I(cont,v):
  F_star(cont,v) = sum( x , sum(w in cont, Ix(w,v)*dw(w,v)) *LDx(band,v)*Sx ) 
                 = I(cont,v)*Sstar_LD(band)            
        thus the intrinsic continuum writes as:
  F_intr(cont,t,v) = I(cont,v)*Sr(band)*Cref(band,v)*Sstar_LD(band)
                   = F_star(cont,v)*Sr(band)*Cref(band,v) 
        and the residual continuum writes as            
  F_res(cont,t,v) = F_star(cont,v)*LDp(band,t)*Sthick(band,t)*Cref(band,v)/Sstar_LD(band)                            
                  = F_star(cont,v)*( 1 - LC_theo(band,t) )*Sr(band,t)*Cref(band,v)
        if Sr~1 the intrinsic profiles have the same flux as the disk-integrated spectra before the relative broadband flux scaling:
  sum(w in cont, Fsc(w,t,v)*dw(w,t,v)/LC_theo(band,t)) = sum(w in cont, Fsc(w,t,v)*dw(w,t,v)/LC_theo(band,t)) 
                                                       = sum(w in cont, Fstar(w,v)*Cref(band,v)*dw(w,t,v)) 
                                                       = F_star(cont,v)*Cref(band,v)
        intrinsic profiles thus have the same continuum as the scaled out-of-transit and master disk-integrated profiles, within a range controlled by broadband flux variations (ie, outside of planetary and stellar lines)
        but this continuum may not be exactly unity

     - bins affected by the planet absorption must be included in the scaling band (rescale_data())), but after this operation is done we set all bins affected by the planetary atmosphere to nan so that the
  final profiles are purely stellar
     - here we do not shift the intrinsic stellar profiles to a common rest wavelength, as they can be later used to derive the velocity of the stellar surface 
     - the approach is the same with CCFs, except that everything is performed with a single band, and the scaling can be carried out in the CCF continuum (thus avoiding potential variations in the line shape)
  + when CCFs are given as input or calculated before the flux scaling, their continuum is normalized to unity outside of the transit
  + when CCFs are calculated from residual spectra the continuum is unknown a priori, as it depends on the reference spectrum to which each DI spectrum corresponds to, times the spectral flux scaling during transit
    the residual spectra write as F_res(w,t,vis) = F_intr(w,t,vis)*(1 - LC_theo(band,t))
    when converting them into CCF_Res, we also compute the equivalent CCFscal of (1 - LC_theo(band,t))
    CCF_Res are then converted into CCF_Intr in this routine by dividing their continuum using CCFscal 
    this is an approximation, since the spectral scaling cannot be isolated from F_intr(w,t,vis) when computing CCF_Res, but it is the same approximation we do when scaling DI CCFs with a white light transit
    we take this approach rather than first calculate Fintr and then its CCF because we need CCF_Res to later derive the atmospheric CCFs
    ideally though, one should keep processing spectra rather than convert residual profiles into CCFs
'''
def extract_intr_profiles(data_dic,gen_dic,inst,vis,star_params,coord_dic,theo_dic,plot_dic):
    
    print('   > Extracting intrinsic stellar profiles')  
    data_vis=data_dic[inst][vis]
    gen_vis = gen_dic[inst][vis]
    
    #Current rest frame
    if data_dic['Res'][inst][vis]['rest_frame']!='star':print('WARNING: residual profiles must be aligned')
    data_dic['Intr'][inst][vis]['rest_frame'] = 'star'

    #Path to initialized intrinsic data
    proc_gen_data_paths_new = gen_dic['save_data_dir']+'Intr_data/'+inst+'_'+vis

    #Updating paths
    #    - if no shifts are applied the associated profiles remain the same as those of the residual profiles, and their paths are not updated
    #    - calibration paths are updated even if they are not used as weights, to be used in flux/count scalings
    #    - paths are defined for each exposure for associated tables, to avoid copying tables from residual profiles and simply point from in-transit to global residual profiles
    data_vis['proc_Intr_data_paths']=proc_gen_data_paths_new+'_' 
    if gen_dic['flux_sc']:data_vis['scaled_Intr_data_paths'] = data_vis['scaled_Res_data_paths']
    if gen_dic['DImast_weight']:data_vis['mast_Intr_data_paths'] = {}
    if data_vis['tell_sp']:data_vis['tell_Intr_data_paths'] = {}
    if data_vis['mean_gdet']:data_vis['mean_gdet_Intr_data_paths'] = {} 
    for i_in,iexp in enumerate(gen_vis['idx_in']):
        if gen_dic['DImast_weight']:data_vis['mast_Intr_data_paths'][i_in] = data_vis['mast_Res_data_paths'][iexp]
        if data_vis['tell_sp']:data_vis['tell_Intr_data_paths'][i_in] = data_vis['tell_Res_data_paths'][iexp]
        if data_vis['mean_gdet']:data_vis['mean_gdet_Intr_data_paths'][i_in] = data_vis['mean_gdet_Res_data_paths'][iexp]

    #Initialize in-transit indexes of defined intrinsic profiles
    data_dic['Intr'][inst][vis]['idx_def'] = np.arange(data_vis['n_in_tr'],dtype=int) 
    
    #Correcting for relative chromatic shift    
    if ('spec' in data_vis['type']) and ('chrom' in data_dic['DI']['system_prop']):intr_rv_corr = True
    else:intr_rv_corr=False

    #Processing intrinsic data
    if (gen_dic['calc_intr_data']):
        print('         Calculating data')
        plAtm_vis = data_dic['Atm'][inst][vis]

        #Correcting for relative chromatic shift
        #    - for spectral data and chromatic occultation: the planet occults region of different size across the spectrum, so that
        # the lines in a given spectral band are shifted by an average RV over the corresponding chromatic surface
        #    - here we correct the intrinsic spectra for the chromatic RV deviation around the nominal planet-occulted RV
        #    - this correction is performed here so that intrinsic profiles can be converted into CCF prior to the alignment module, in which
        # a constant RV shift can then be applied
        if intr_rv_corr:
            ref_pl,dic_rv,idx_aligned = init_surf_shift(gen_dic,inst,vis,data_dic,'theo')
            
            #Resample aligned profiles on the common visit table if relevant
            #    - for CCFs the common table has been centered on the systemic velocity 
            if (data_vis['comm_sp_tab']):
                data_com = dataload_npz(data_vis['proc_com_data_paths'])
                cen_bins_resamp, edge_bins_resamp = data_com['cen_bins'],data_com['edge_bins']
            else:cen_bins_resamp, edge_bins_resamp = None,None  

        #Definition of intrinsic stellar profiles
        for i_in,iexp in enumerate(gen_vis['idx_in']):

            #Upload local stellar profile
            data_exp = dataload_npz(data_vis['proc_Res_data_paths']+str(iexp))
        
            #Upload flux scaling
            data_scaling_exp = dataload_npz(data_vis['scaled_Res_data_paths']+str(iexp))
        
            #Rescale local stellar profiles to a common flux level
            #    - correcting for LD variation and planetary occultation
            #    - the scaling spectrum is defined at all wavelengths, thus defined bins are unchanged 
            #      the scaling equals 0 for undefined pixels, thus we set the rescaling to 1 to avoid warnings
            for iord in range(data_dic[inst]['nord']):
                cond_exp_ord = data_exp['cond_def'][iord]
                resc_ord = np.ones(data_vis['nspec'],dtype=float)
                resc_ord[cond_exp_ord] = 1./data_scaling_exp['loc_flux_scaling'](data_exp['cen_bins'][iord,cond_exp_ord])
                data_exp['flux'][iord],data_exp['cov'][iord] = bind.mul_array(data_exp['flux'][iord],data_exp['cov'][iord],resc_ord)
 
            #Correct for relative chromatic shift
            if intr_rv_corr:

                #Radial velocity shifts set to the opposite of the planet-occulted surface rv associated with current exposure
                surf_shifts,surf_shifts_edge = def_surf_shift('theo_rel',dic_rv,i_in,data_exp,ref_pl,data_vis['type'],data_dic['DI']['system_prop'],data_dic[inst][vis]['dim_exp'],data_dic[inst]['nord'],data_dic[inst][vis]['nspec']) 

                #Spectral RV correction of current exposure and complementary tables
                if data_vis['tell_sp']:data_exp['tell'] = dataload_npz(data_vis['tell_Res_data_paths'][iexp])['tell'] 
                if data_vis['mean_gdet']:data_exp['mean_gdet'] = dataload_npz(data_vis['mean_gdet_Res_data_paths'][iexp] )['mean_gdet'] 
                data_exp=align_data(data_exp,data_vis['type'],data_dic[inst]['nord'],data_dic[inst][vis]['dim_exp'],gen_dic['resamp_mode'],cen_bins_resamp,edge_bins_resamp,surf_shifts,rv_shift_edge = surf_shifts_edge)

                #Saving aligned exposure and complementary tables
                if ('spec' in data_vis['type']):
                    if data_vis['tell_sp']:
                        data_vis['tell_Intr_data_paths'][i_in] = proc_gen_data_paths_new+'_tell'+str(i_in)
                        np.savez_compressed(data_vis['tell_Intr_data_paths'][i_in], data = {'tell':data_exp['tell']},allow_pickle=True) 
                        data_exp.pop('tell')
                    if data_vis['mean_gdet']:
                        data_vis['mean_gdet_Intr_data_paths'][i_in] = proc_gen_data_paths_new+'_mean_gdet'+str(i_in)
                        np.savez_compressed(data_vis['mean_gdet_Intr_data_paths'][i_in], data = {'mean_gdet':data_exp['mean_gdet']},allow_pickle=True) 
                        data_exp.pop('mean_gdet')

                #Spectral RV correction of weighing master
                if gen_dic['DImast_weight']:
                    data_ref = np.load(data_vis['mast_Res_data_paths'][iexp]+'.npz',allow_pickle=True)['data'].item() 
                    data_ref_align=align_data(data_ref,data_vis['type'],data_dic[inst]['nord'],data_dic[inst][vis]['dim_exp'],gen_dic['resamp_mode'],cen_bins_resamp,edge_bins_resamp,surf_shifts,rv_shift_edge = surf_shifts_edge)                  
                    data_vis['mast_Intr_data_paths'][i_in] = proc_gen_data_paths_new+'_ref'+str(i_in)
                    np.savez_compressed(data_vis['mast_Intr_data_paths'][i_in],data=data_ref_align,allow_pickle=True)

            #Set to nan planetary ranges
            #    - only if intrinsic spectra are not to be converted into CCFs, in which case the exclusion is applied after their conversion
            if ('Intr' in data_dic['Atm']['no_plrange']) and (not gen_dic['Intr_CCF']) and (iexp in plAtm_vis['iexp_no_plrange']):
                cond_out_pl = np.ones(data_vis['dim_exp'],dtype=bool)
                for iord in range(data_dic[inst]['nord']):                        
                    cond_out_pl[iord] &=excl_plrange(data_exp['cond_def'][iord],plAtm_vis['exclu_range_star'],iexp,data_exp['edge_bins'][iord],data_vis['type'])[0]
                cond_in_pl = ~cond_out_pl
                data_exp['flux'][cond_in_pl]=np.nan
                data_exp['cond_def'][cond_in_pl]=False    

                #Saving exclusion flag
                data_exp['plrange_exc'] = True
            else:data_exp['plrange_exc'] = False

            #Saving data using in-transit indexes              
            np.savez_compressed(proc_gen_data_paths_new+'_'+str(i_in),data=data_exp,allow_pickle=True)

    else:
        data_paths={i_in:proc_gen_data_paths_new+'_'+str(i_in) for i_in in data_dic['Intr'][inst][vis]['idx_def']}
        check_data(data_paths)  

    #Continuum level and correction
    #    - at this stage, profiles in CCF mode always come from input CCF data
    #      continuum is only calculated for spectral data if not converted later on (in which case continuum is calculated later on)
    #    - if applied to intrinsic profiles derived from CCF data, planetary ranges have been excluded if requested
    #      for spectral data the full order range is taken as continuum
    if (data_vis['type']=='CCF') or (('spec' in data_vis['type']) and ((not gen_dic['Intr_CCF']) and (not gen_dic['spec_1D_Intr']))):           
        data_dic['Intr'][inst][vis]['mean_cont']=calc_Intr_mean_cont(data_vis['n_in_tr'],data_dic[inst]['nord'],data_vis['nspec'],data_vis['proc_Intr_data_paths'],data_vis['type'],data_dic['Intr']['cont_range'],inst,data_dic['Intr']['cont_norm'])

    return None


'''
Calculation of common continuum level for intrinsic level
    - defined as a weighted mean because local CCFs at the limbs can be very poorly defined due to the partial occultation and limb-darkening
    - if intrinsic CCFs were calculated from input CCF profiles, their continuum flux should match that of the out-of-transit CCFs (beware however that the scaling of disk-integrated profiles 
is done over their full range, not just the continuum)
      if residual or intrinsic profiles were converted from spectra, then their continuum is not know a priori
      a a general approach we thus calculate the continuum value here
'''
def calc_Intr_mean_cont(n_in_tr,nord,nspec,proc_Intr_data_paths,data_type,cont_range,inst,cont_norm):
    cond_def_cont_all  = np.zeros([n_in_tr,nord,nspec],dtype=bool)
    for i_in in range(n_in_tr):
        data_exp = dataload_npz(proc_Intr_data_paths+str(i_in))  
        for iord in range(nord):
            if (inst in cont_range) and (iord in cont_range[inst]):
                for bd_int in cont_range[inst][iord]:cond_def_cont_all[i_in,iord] |= (data_exp['edge_bins'][0,0:-1]>=bd_int[0]) & (data_exp['edge_bins'][0,1:]<=bd_int[1])     
            else:cond_def_cont_all[i_in,:] = True       
            cond_def_cont_all[i_in,iord] &= data_exp['cond_def'][iord]            
    cond_cont_com  = np.all(cond_def_cont_all,axis=0) 
    cont_intr = np.ones([n_in_tr,nord])
    wcont_intr = np.ones([n_in_tr,nord])
    for i_in in range(n_in_tr):
        data_exp = dataload_npz(proc_Intr_data_paths+str(i_in))
        for iord in range(nord):
            cond_cont_com_ord = cond_cont_com[iord]
            if np.sum(cond_cont_com_ord)==0.:stop('No pixels in common continuum') 
            dcen_bins = data_exp['edge_bins'][iord,1:][cond_cont_com_ord] - data_exp['edge_bins'][iord,0:-1][cond_cont_com_ord]
            
            #Continuum flux of the intrinsic CCF and corresponding error
            #    - calculated over the defined bins common to all residual and intrinsic profiles
            #    - we use the covariance diagonal to define a representative weight              
            cont_intr[i_in,iord] = np.sum(data_exp['flux'][iord,cond_cont_com_ord]*dcen_bins)/np.sum(dcen_bins)
            wcont_intr[i_in,iord] = np.sum(dcen_bins**2.)/np.sum( data_exp['cov'][iord][0,cond_cont_com_ord]*dcen_bins**2. )            
    
    #Continuum flux over all in-transit exposures
    mean_cont=np.sum(cont_intr*wcont_intr,axis=0)/np.sum(wcont_intr,axis=0)      

    #Continuum correction
    #    - intrinsic profiles can show deviations from the common continuum level they should have 
    #      here we set manually their continuum to the mean continuum of all intrinsic profiles before the correction
    if cont_norm:    
        print('         Correcting intrinsic continuum')
        for i_in in range(n_in_tr):
            data_exp = dataload_npz(proc_Intr_data_paths+str(i_in))
            for iord in range(nord):
                cond_cont_com_ord = cond_cont_com[iord]
                
                #Continuum of current exposure
                cont_intr_exp = np.mean(data_exp['flux'][iord][cond_cont_com_ord])
                
                #Correction factor
                corr_exp = mean_cont[iord]/cont_intr_exp
    
                #Overwrite exposure data
                data_exp['flux'][iord],data_exp['cov'][iord] = bind.mul_array(data_exp['flux'][iord],data_exp['cov'][iord],np.repeat(corr_exp,nspec))
            datasave_npz(proc_Intr_data_paths+str(i_in),data_exp)       

    return mean_cont






'''
Routine to apply PCA
    - we define the perturbation as F(w,t,v) = Fstar(w,v) + Pert(w,t,v)
      where Pert(w,t,v) = sum( k , a(t,v)*PC(k,w) ) is the physical perturbation from the star
      we assume that the perturbation is roughly uniform over the stellar disk:
 Pert(w,t,v) ~ sum(k, Ipert(w,t,v)*LDk(band)*Sk) = Ipert(w,t,v)*sum(i, LDk(band)*Sk) = Ipert(w,t,v)*Sstar_LD(band)
      
    - we apply the correction to the raw disk-integrated profiles F(w,t,v)*Cref(band,v)*globF(v,t) in detrend_prof(), so that the correction model is: 
 Pcorr_pca(w,t,v) = Pert(w,t,v)*Cref(band,v)*globF(v,t) 
 
    - residual profiles are calculated from the scaled spectra:
 Fsc(w,t) = F(w,t,v)*Cref(w,v)
 
    - out-of-transit:
 F_res(w,t,v) = Fstar_sc(w,v) - F_sc(w,t,v) 
              = - Pert(w,t,v)*Cref(band,v)              
      we thus fit F_res(w,t,v)*globF(v,t)     

    - in-transit, outside of the planetary lines (see proc_intr_data() ): 
 F_intr(w,t,v) = F_res(w,t,v)/(1 - LC_theo(band,t))
                = ( Fstar_sc(w,v) - F_sc(w,t,v) )/(1 - LC_theo(band,t))
                = ( sum(k, Ik(w,t,v)*LDk(band)*Sk)   - (sum(k unocc, Ik(w,t,v)*LDk(band)*Sk) + sum(k unocc, Ipert(w,t,v)*LDk(band)*Sk)))*Cref(band,v)/(1 - LC_theo(band,t))
                = F_intr_pl(w,t,v) - Ipert(w,t,v)*sum(k unocc,LDk(band)*Sk)*Cref(band,v)/(1 - LC_theo(band,t))
                = F_intr_pl(w,t,v) - (Pert(w,t,v)/Sstar_LD(band))*(Sstar_LD(band) - LD(band,t)*Sp(band,t))*Cref(band,v)/(1 - LC_theo(band,t))
                = F_intr_pl(w,t,v) - Pert(w,t,v)*(1 - (LD(band,t)*Sp(band,t)/Sstar_LD(band)))*Cref(band,v)/(1 - LC_theo(band,t))               
                = F_intr_pl(w,t,v) - Pert(w,t,v)*LC_theo(band,t)*Cref(band,v)/(1 - LC_theo(band,t))                 
      outside of the planet-occulted stellar lines the continuum of F_intr_pl is constant, and the continuum of Ppert is assumed to be null, so that:                
 F_intr(w,t,v) - F_intr(cont,t,v) = - Pert(w,t,v)*LC_theo(band,t)*Cref(band,v)/(1 - LC_theo(band,t))
      we thus fit (F_intr(w,t,v) - F_intr(cont,t,v))*globF(v,t)*(1 - LC_theo(band,t))/LC_theo(band,t)
            
'''
def pc_model(params,x,args = None):
    mod = np.zeros(len(x),dtype=float)
    for i_pc in range(args['n_pc']):mod+=params['aPC_idx'+args['iexp']+'_ord'+str(i_pc)+args['suff']]*args['eig_res_matr'][i_pc]
    return mod

def pc_analysis(gen_dic,data_dic,inst,vis,data_prop,coord_dic):
    if (inst in data_dic['PCA']['vis_list']) and ((vis in data_dic['PCA']['vis_list'][inst]) or (data_dic['PCA']['vis_list'][inst]=='all')):
        print('   > Applying PCA to OT residual profiles') 
        
        #Processing data
        if (gen_dic['calc_pca_ana']):
            print('         Calculating data')        
            data_vis=data_dic[inst][vis]
            if (data_vis['type']!='CCF'):stop('Only coded for CCFs')
            gen_vis = gen_dic[inst][vis]

            #Exposure selection
            idx_pca = gen_vis['idx_out']
            if (inst in data_dic['PCA']['idx_pca']) and (vis in data_dic['PCA']['idx_pca'][inst][vis]) and (len(data_dic['PCA']['idx_pca'][inst][vis])>0):
                idx_pca = np.intersect1d(data_dic['PCA']['idx_pca'][inst][vis],idx_pca)
            nexp_pca = len(idx_pca)
       
            #Analysis and fit range 
            #    - currently we assume that all exposures are defined over the same spectral table
            data_com = np.load(data_vis['proc_com_data_paths']+'.npz',allow_pickle=True)['data'].item()
            cond_out_fit_range = False   
            for bd_int in data_dic['PCA']['ana_range'][inst][vis]:
                cond_out_fit_range |= (data_com['edge_bins'][0,0:-1]>=bd_int[0]) & (data_com['edge_bins'][0,1:]<=bd_int[1])   

            #Store residual profiles as a matrix
            n_rep = 100
            full_res_matr = np.zeros([nexp_pca,data_vis['nspec']],dtype=float)*np.nan
            full_noise_matr = np.zeros([nexp_pca,data_vis['nspec'],n_rep],dtype=float)*np.nan
            cond_fit_range_matr = np.zeros([nexp_pca,data_vis['nspec']],dtype=bool)
            iexp2ipca = np.zeros(data_vis['n_in_visit'],dtype=int)-1
            isub_pca_pretr = []
            isub_pca_posttr = []
            for isub,iexp in enumerate(idx_pca):
                iexp2ipca[iexp] = isub
                if iexp in gen_vis['idx_pretr']:isub_pca_pretr+=[isub]
                else:isub_pca_posttr+=[isub]
                    
                #PCA based on out-of-transit data alone
                data_exp = np.load(data_vis['proc_Res_data_paths']+str(iexp)+'.npz',allow_pickle=True)['data'].item()
                cond_fit_range_matr[isub] = cond_out_fit_range & data_exp['cond_def'][0] 
                full_res_matr[isub] = data_exp['flux'][0]

                #Exclude planetary ranges
                if ('PCA_corr' in data_dic['Atm']['no_plrange']) and (iexp in data_dic['Atm'][inst][vis]['iexp_no_plrange']):
                    cond_in_pl = ~(np.ones(data_vis['nvel'],dtype=bool) & excl_plrange(data_exp['cond_def'][0],data_dic['Atm'][inst][vis]['exclu_range_star'],iexp,data_exp['edge_bins'][0],data_dic[inst][vis]['type'])[0])
                    cond_fit_range_matr[isub,cond_in_pl] = False

                #Generate 'noise' matrix 
                #    - we create n_rep realisations using the error array of current exposure
                err_exp = np.sqrt(data_exp['cov'][0][0])
                full_noise_matr[isub] = list(map(np.random.normal, np.zeros(data_vis['nspec']), err_exp, [n_rep] * data_vis['nspec']))

            #Reduce matrixes to analysis range and center each pixel on null level over time (ie the mean value over selected exposures is subtracted)
            cond_def_all = np.all(cond_fit_range_matr,axis=0)
            nspec_res_mat = np.sum(cond_def_all)
            n_pca_pretr = len(isub_pca_pretr)
            n_pca_posttr = len(isub_pca_posttr)
            nexp_pca = {'out':n_pca_pretr+n_pca_posttr}
            res_matr = {'out': full_res_matr[:,cond_def_all]}
            noise_matr = {'out': full_noise_matr[:,cond_def_all]}
            eig_vals = {}
            eig_val_noise = {}
            if n_pca_pretr>0:
                nexp_pca['pre'] =n_pca_pretr 
                res_matr['pre']=full_res_matr[isub_pca_pretr][:,cond_def_all]
                noise_matr['pre']=full_noise_matr[isub_pca_pretr][:,cond_def_all]
            if n_pca_posttr>0:
                nexp_pca['post'] =n_pca_posttr
                res_matr['post']=full_res_matr[isub_pca_posttr][:,cond_def_all]
                noise_matr['post']=full_noise_matr[isub_pca_posttr][:,cond_def_all]
            for key in res_matr:
                res_matr[key]-=np.nanmean(res_matr[key],axis=0)
                noise_matr[key]-=np.nanmean(noise_matr[key],axis=0)

                #Eigen decomposition
                #    - eigenvalues have dimension nexp, and are already sorted in descending order
                #      eigen_vecs have dimension nexp x nexp, where eig_vecs_matr[i,:] is the ith vector corresponding to the ith eigenvalue
                _, eig_vals_sqrt, eig_vecs_matr = np.linalg.svd(res_matr[key])
                eig_vals[key] = eig_vals_sqrt**2.
                eig_vecs_matr = eig_vecs_matr[0:nexp_pca[key]]

                #Store eigenvectors into null matrix with same dimension as original data
                if key=='out':
                    eig_res_matr = np.zeros([nexp_pca[key],data_vis['nspec']],dtype=float)
                    eig_res_matr[:,cond_def_all] = eig_vecs_matr

                #Calculate eigenvalues for each realization and average them
                eig_val_noise[key] = np.zeros(nexp_pca[key],dtype=float)
                for iboot in range(n_rep):
                    _, eig_vals_noise_sqrt, _ = np.linalg.svd(noise_matr[key][:,:,iboot])
                    eig_val_noise[key]+= eig_vals_noise_sqrt**2.
                eig_val_noise[key]/=n_rep
            
            #--------------------------------------------------------------------
            #Fit PC to all exposures
            fixed_args = {'use_cov':False,'suff':'__IS'+inst+'_VS'+vis}
            
            #PC to use
            fixed_args['n_pc'] = data_dic['PCA']['n_pc'][inst][vis]
            eig_res_matr_fit = eig_res_matr[0:fixed_args['n_pc']]
                
            #Exposures to correct
            idx_corr = np.arange(data_vis['n_in_visit'])
            if (inst in data_dic['PCA']['idx_corr']) and (vis in data_dic['PCA']['idx_corr'][inst][vis]) and (len(data_dic['PCA']['idx_corr'][inst][vis])>0):
                idx_corr = np.intersect1d(data_dic['PCA']['idx_corr'][inst][vis],idx_corr)

            #Fit range
            if (inst not in data_dic['PCA']['fit_range']) or (vis not in data_dic['PCA']['fit_range'][inst]):
                fit_range = data_dic['PCA']['ana_range'][inst][vis]
            else:fit_range=data_dic['PCA']['fit_range'][inst][vis]

            #Process selected exposures
            rms_full_res_matr = np.zeros([2,data_vis['n_in_visit']],dtype=float)*np.nan
            corr_res_matr = np.zeros([nexp_pca['out'],nspec_res_mat],dtype=float)
            chi2null_tab = np.zeros(data_vis['n_in_visit'],dtype=float)*np.nan
            chi2_tab = np.zeros(data_vis['n_in_visit'],dtype=float)*np.nan
            BIC_tab = np.zeros(data_vis['n_in_visit'],dtype=float)*np.nan
            nfit_tab = np.zeros(data_vis['n_in_visit'],dtype=float)*np.nan
            p_final = {}   
            for isub,iexp in enumerate(idx_corr): 

                #Initialise fit parameters 
                p_start = Parameters()
                for i_pc in range(fixed_args['n_pc']):p_start.add_many(('aPC_idx'+str(iexp)+'_ord'+str(i_pc)+'__IS'+inst+'_VS'+vis, 0.,  True, None,None,  None))                
                fixed_args['iexp'] = str(iexp)

                #Fitted range and continuum level
                #    - see explanations in function comments
                data_exp = np.load(data_vis['proc_Res_data_paths']+str(iexp)+'.npz',allow_pickle=True)['data'].item() 
                cond_fit_range = False
                for bd_int in fit_range:cond_fit_range |= (data_exp['edge_bins'][0,0:-1]>=bd_int[0]) & (data_exp['edge_bins'][0,1:]<=bd_int[1])  
                cond_fit_range &=  data_exp['cond_def'][0]

                #Fit PCs
                fixed_args['idx_fit'] = np_where1D(cond_fit_range)
                fixed_args['eig_res_matr'] = eig_res_matr_fit[:,cond_fit_range]
                result, merit ,p_best = fit_minimization(ln_prob_func_lmfit,p_start,data_exp['cen_bins'][0],data_exp['flux'][0],data_exp['cov'][0],pc_model,verbose=True,fixed_args=fixed_args)
                for key in p_best:p_final[key] = p_best[key].value

                #BIC for null model (=chi2) and best-fit model
                flux_fit = data_exp['flux'][0,cond_fit_range]
                cov_fit = data_exp['cov'][0][:,cond_fit_range] 
                chi2null_tab[iexp] = np.sum( flux_fit**2. / cov_fit[0])
                nfit_tab[iexp] = np.sum(cond_fit_range)
                chi2_tab[iexp] = merit['chi2']
                BIC_tab[iexp] = merit['BIC']

                #Measure RMS pre/post correction on fitted residual profile
                rms_full_res_matr[0,iexp] = np.std(flux_fit)
                rms_full_res_matr[1,iexp] = np.std(flux_fit - merit['fit'])
                
                #Measure RMS pre/post correction on defined residual profile used for the PCA
                if iexp2ipca[iexp]>-1:
                    isub_pca = iexp2ipca[iexp]
                    res_exp = full_res_matr[isub_pca,cond_def_all]
                  
                    fixed_args['eig_res_matr'] = eig_res_matr_fit[:,cond_def_all]
                    corr_res_exp = res_exp - pc_model(p_best,data_com['cen_bins'][0,cond_def_all],args = fixed_args)                  
                
                    #Store corrected residual profile used for the PCA
                    #    - OT profiles are fully defined over 'cond_def_all'
                    corr_res_matr[isub_pca] = corr_res_exp

            #--------------------------------------------------------------------
            #Quality assessment
            #    - apply FFT to defined residual profiles used for the PCA
            #    - we separate pre- and post-transit exposures to avoid introducing correlation, if the noise is not white

            #Max fft of residual profiles
            #    - mean-centering does not change the result   
            fft_pre = np.nan if n_pca_pretr==0. else np.max(np.abs(scipy.fft.fft2(res_matr['pre']))**2.)
            fft_post = np.nan if n_pca_posttr==0. else np.max(np.abs(scipy.fft.fft2(res_matr['post']))**2.)
            max_fft2_res_matr = [ fft_pre , fft_post ]
                            
            #Max fft of corrected residual profiles used for PCA
            fft_pre = np.nan if n_pca_pretr==0. else np.max(np.abs(scipy.fft.fft2(corr_res_matr[isub_pca_pretr]))**2.)
            fft_post = np.nan if n_pca_posttr==0. else np.max(np.abs(scipy.fft.fft2(corr_res_matr[isub_pca_posttr]))**2.)
            max_fft2_corr_res_matr = [ fft_pre , fft_post ]
            
            #Distribution of residuals, post-correction
            hist_corr_res_pre = np.histogram(corr_res_matr[isub_pca_pretr],bins=data_dic['PCA']['nbins'],density=True)
            std_corr_res_pre = np.std(corr_res_matr[isub_pca_pretr])
            hist_corr_res_post = np.histogram(corr_res_matr[isub_pca_posttr],bins=data_dic['PCA']['nbins'],density=True)
            std_corr_res_post = np.std(corr_res_matr[isub_pca_posttr])
            hist_corr_res = np.histogram(corr_res_matr,bins=data_dic['PCA']['nbins'],density=True)
            std_corr_res = np.std(corr_res_matr)

            #Loop over velocities
            fft1D_res_matr = np.zeros([2,nspec_res_mat],dtype=float)*np.nan
            fft1D_corr_res_matr = np.zeros([2,nspec_res_mat],dtype=float)*np.nan
            fft1D_boot_res_matr = np.zeros([2,nspec_res_mat],dtype=float)
            if n_pca_pretr>0:boot_res_matr_pretr = np.zeros([n_pca_pretr,nspec_res_mat,data_dic['PCA']['nboot']],dtype=float) 
            if n_pca_posttr>0:boot_res_matr_posttr = np.zeros([n_pca_posttr,nspec_res_mat,data_dic['PCA']['nboot']],dtype=float) 
            for ipix in range(nspec_res_mat):
                
                #Pre-transit data
                if n_pca_pretr>0:
                    
                    #FFT on current velocity column
                    fft1D_res_matr[0,ipix] = np.max(np.abs(scipy.fft.fft(res_matr['pre'][:,ipix]))**2.)
                    fft1D_corr_res_matr[0,ipix] = np.max(np.abs(scipy.fft.fft(corr_res_matr[isub_pca_pretr,ipix]))**2.)
                    for iboot in range(data_dic['PCA']['nboot']):
                        boot_res_matr_pretr[:,ipix,iboot] = np.random.choice(corr_res_matr[isub_pca_pretr,ipix],n_pca_pretr,replace=False)                        
                        fft1D_boot_res_matr[0,ipix] += np.max(np.abs(scipy.fft.fft(boot_res_matr_pretr[:,ipix,iboot]))**2.)
                    fft1D_boot_res_matr[0,ipix]/=n_pca_pretr

                #Post-transit data
                if n_pca_posttr>0:

                    #FFT on current velocity column
                    fft1D_res_matr[1,ipix] = np.max(np.abs(scipy.fft.fft(res_matr['post'][:,ipix]))**2.)
                    fft1D_corr_res_matr[1,ipix] = np.max(np.abs(scipy.fft.fft(corr_res_matr[isub_pca_posttr,ipix]))**2.)
                    for iboot in range(data_dic['PCA']['nboot']):
                        boot_res_matr_posttr[:,ipix,iboot] = np.random.choice(corr_res_matr[isub_pca_posttr,ipix],n_pca_posttr,replace=False) 
                        fft1D_boot_res_matr[1,ipix] += np.max(np.abs(scipy.fft.fft(boot_res_matr_posttr[:,ipix,iboot]))**2.)
                    fft1D_boot_res_matr[1,ipix]/=n_pca_posttr

            #Max fft of corrected residual profiles used for PCA, bootstrapped                        
            max_fft2_boot_res_matr = np.zeros(2,dtype=float)
            for iboot in range(data_dic['PCA']['nboot']):
                if n_pca_pretr>0:max_fft2_boot_res_matr[0] +=  np.max(np.abs(scipy.fft.fft2(boot_res_matr_pretr[:,:,iboot]))**2.)
                if n_pca_posttr>0:max_fft2_boot_res_matr[1] +=  np.max(np.abs(scipy.fft.fft2(boot_res_matr_posttr[:,:,iboot]))**2.)
            if n_pca_pretr>0:
                max_fft2_boot_res_matr[0]/=n_pca_pretr
            else:max_fft2_boot_res_matr[0] = np.nan
            if n_pca_posttr>0:max_fft2_boot_res_matr[1]/=n_pca_posttr
            else:max_fft2_boot_res_matr[1] = np.nan                

            #--------------------------------------------------------------------

            #Save PCA results
            data_save = {'eig_val_res':eig_vals,'eig_val_noise':eig_val_noise,'eig_res_matr':eig_res_matr,'idx_pca':idx_pca,'isub_pca':np_where1D(iexp2ipca>-1),'idx_corr':idx_corr,'cen_bins':data_com['cen_bins'][0],
                         'n_pc':fixed_args['n_pc'],'rms_full_res_matr':rms_full_res_matr,'BIC_tab':BIC_tab,'chi2_tab':chi2_tab,'chi2null_tab':chi2null_tab,'nfit_tab':nfit_tab,
                         'edge_bins' : data_com['edge_bins'],'p_final':p_final,
                         'max_fft2_res_matr':max_fft2_res_matr,'max_fft2_corr_res_matr':max_fft2_corr_res_matr,'max_fft2_boot_res_matr':max_fft2_boot_res_matr,
                         'cen_bins_res_mat':data_com['cen_bins'][0,cond_def_all],'fft1D_res_matr':fft1D_res_matr,'fft1D_corr_res_matr':fft1D_corr_res_matr,'fft1D_boot_res_matr':fft1D_boot_res_matr,
                         'hist_corr_res_pre':hist_corr_res_pre,'hist_corr_res_post':hist_corr_res_post,'hist_corr_res':hist_corr_res,
                         'std_corr_res_pre':std_corr_res_pre,'std_corr_res_post':std_corr_res_post,'std_corr_res':std_corr_res}
            np.savez_compressed(gen_dic['save_data_dir']+'PCA_results/'+inst+'_'+vis,data=data_save,allow_pickle=True)   
    
        else:
            data_paths={'0':gen_dic['save_data_dir']+'PCA_results/'+inst+'_'+vis}
            check_data(data_paths)  

    return None












'''
Routine to analyze line profiles
    - function is applied to:
# + profiles in their input rest frame, original exposures, for all formats
# + profiles in their input or star (if aligned) rest frame, original exposures, converted from 2D->1D 
# + profiles in their input or star (if aligned) rest frame, binned exposures, all formats
# + profiles in the star rest frame, binned exposures, all formats        
    - formats are CCF profiles, 1D spectra, or a specific 2D spectral order
    - each profile is analyzed independently from the others
    - the module can be used to fit the disk-integrated master using best estimates for the intrinsic stellar profiles (measured, theoretical, imported)    
'''
def ana_prof(vis_mode,data_type,data_dic,gen_dic,inst,vis,coord_dic,theo_dic,plot_dic,star_param):

    #Analyzing profiles
    if 'orig' in data_type:   #unbinned profiles
        bin_mode=''
        txt_print=' '
        vis_det=vis  
        data_mode = data_dic[inst][vis]['type']
    elif 'bin' in data_type:  #binned profiles
        bin_mode='bin'   
        txt_print=' binned '
        if vis_mode=='':            #from single visit
            vis_det=vis
            data_mode = data_dic[inst][vis]['type']    
        elif vis_mode=='multivis':  #from multiple visits
            vis_det='binned'
            data_mode = data_dic[inst]['type'] 
            txt_print+=' (multi-vis.) '
    
    if ('DI' in data_type) or ('Intr' in data_type):
        if ('DI' in data_type):data_type_gen='DI' 
        elif ('Intr' in data_type):data_type_gen='Intr'               
        print('   > Analyzing'+txt_print+gen_dic['type_name'][data_type_gen]+' stellar profiles')                 
    elif ('Atm' in data_type):   
        data_type_gen='Atm' 
        print('   > Analyzing'+txt_print+gen_dic['type_name'][data_type_gen]+data_dic['Atm']['pl_atm_sign']+' profiles') 

    save_path = gen_dic['save_data_dir']+data_type+'_prop/'+gen_dic['add_txt_path'][data_type_gen]+inst+'_'+vis_det    
    cond_calc = gen_dic['calc_fit_'+data_type_gen+bin_mode+vis_mode]
    prop_dic = deepcopy(data_dic[data_type_gen])   
    if prop_dic['cst_err'+bin_mode]:print('         Using constant error')
    if (inst in prop_dic['sc_err']) and (vis_det in prop_dic['sc_err'][inst]): print('         Scaling data error')

    #Analyzing
    if cond_calc:
        print('         Calculating data') 
        fit_dic={}
        data_inst = data_dic[inst]

        #Generic fit options
        fit_properties={'type':data_mode,'nthreads':gen_dic['fit_prof_nthreads'],'rv_osamp_line_mod':theo_dic['rv_osamp_line_mod'],'use_cov':gen_dic['use_cov'],'varpar_priors':{},'jitter':False,'inst_list':[inst],'inst_vis_list':{inst:[vis_det]},
                        'save_dir':save_path+'_'+prop_dic['fit_mod']+'/','conv_model' : prop_dic['conv_model'],'resamp_mode' : gen_dic['resamp_mode'],'line_trans':prop_dic['line_trans']}
        fit_properties.update({**star_param})
        
        #Chromatic intensity grid
        if ('spec' in data_mode) and ('chrom' in data_dic['DI']['system_prop']):
            fit_properties['chrom'] = data_dic['DI']['system_prop']['chrom']
        else:
            fit_properties['chrom'] = None        
               
        #Exposure-specific properties        
        if ('mod_def' in prop_dic) and (inst in prop_dic['mod_def']):fit_properties.update(prop_dic['mod_def'][inst])
        
        #Continuum and fitted ranges
        #    - spectral tables are defined in:
        # > input frame for original and binned unaligned disk-integrated data
        #   star rest frame for binned and aligned disk-integrated data
        # > star frame for original / binned intrinsic data
        # > star frame for original / binned planetary data
        #    - continuum and fit ranges are defined in:
        # > input frame for disk-integrated data
        # > star frame for intrinsic data
        # > star frame for planetary data 
        fit_range = prop_dic['fit_range'][inst][vis_det]
        cont_range = prop_dic['cont_range'][inst]
        trim_range = prop_dic['fit_prof']['trim_range'][inst] if (inst in prop_dic['fit_prof']['trim_range']) else None 
        
        #MCMC fit default options
        if prop_dic['fit_mod']=='mcmc': 
            if ('mcmc_set' not in prop_dic):prop_dic['mcmc_set']={}
            for key in ['nwalkers','nsteps','nburn']:
                if key not in prop_dic['mcmc_set']:prop_dic['mcmc_set'][key] = {}
                if (inst not in prop_dic['mcmc_set'][key]):prop_dic['mcmc_set'][key][inst] = {}
            if (vis not in prop_dic['mcmc_set']['nwalkers'][inst]):fit_dic['nwalkers'] = 50
            if (vis not in prop_dic['mcmc_set']['nsteps'][inst]):fit_dic['nsteps'] = 1000
            if (vis not in prop_dic['mcmc_set']['nburn'][inst]):fit_dic['nburn'] = 200
       
        #Default model
        if ('model' not in prop_dic):prop_dic['model']={}
        if (inst not in prop_dic['model']):prop_dic['model'][inst]='gauss'
        
        #Binned data
        if bin_mode=='bin':   
            data_bin = np.load(gen_dic['save_data_dir']+data_type_gen+'bin_data/'+gen_dic['add_txt_path'][data_type_gen]+inst+'_'+vis_det+'_'+prop_dic['dim_bin']+'_add.npz',allow_pickle=True)['data'].item()
            fit_dic['n_exp'] = data_bin['n_exp']
            rest_frame = data_bin['rest_frame']
     
            #Defined fitted bins for each exposure     
            dim_exp = data_bin['dim_exp']
            nspec = dim_exp[1]       

        #Defined fitted bins for each exposure
        elif bin_mode=='':
            dim_exp = data_inst[vis]['dim_exp']
            nspec = data_inst[vis]['nspec']  
            rest_frame = data_dic[data_type_gen][inst][vis]['rest_frame']

        #Upload analytical surface RVs
        if (('DI' in data_type) and (inst in prop_dic['occ_range'])) or ('Intr' in data_type):
            if len(data_inst[vis]['transit_pl'])>1:stop('Adapt model to multiple planets')
            else:
                ref_pl = data_inst[vis]['transit_pl'][0]
                if bin_mode=='':surf_rv_mod = dataload_npz(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis)['achrom'][ref_pl]['rv'][0]
                else:surf_rv_mod = data_bin['achrom'][ref_pl]['rv'][0]
 
        #Disk-integrated profiles
        if 'DI' in data_type:
        
            #Original profiles
            if data_type=='DIorig':
                fit_dic['n_exp'] = data_inst[vis]['n_in_visit'] 
                idx_exp2in = gen_dic[inst][vis]['idx_exp2in']
                idx_in = gen_dic[inst][vis]['idx_in'] 

            #Binned profiles
            elif data_type=='DIbin':
                idx_exp2in = data_bin['idx_exp2in']
                idx_in = data_bin['idx_in'] 
           
                #Shift ranges (defined in input rest frame) into the star rest frame
                if rest_frame=='star':
                    fit_range = np.array(fit_range)
                    cont_range = np.array(cont_range)
                    if ('spec' in data_mode):
                        conv_fact = spec_dopshift(data_bin['sysvel'])
                        fit_range*=conv_fact
                        cont_range*=conv_fact
                        if (trim_range is not None):trim_range = np.array(trim_range)*conv_fact
                    elif (data_mode=='CCF'):
                        fit_range-=data_bin['sysvel']
                        cont_range-=data_bin['sysvel']
                        if (trim_range is not None):trim_range = np.array(trim_range)-data_bin['sysvel']
                    if ('rv' in prop_dic['mod_prop']) and (inst in prop_dic['mod_prop']['rv']) and (vis in prop_dic['mod_prop']['rv'][inst]) and (vis_mode==''):prop_dic['mod_prop']['rv'][inst][vis]['guess']-=data_bin['sysvel']


            #Defined exposures
            iexp_def = range(fit_dic['n_exp'])

            #Definition of ranges covered by occulted stellar lines / for signal integration in the star rest frame
            #    - we center requested range around surface RVs in original rest frame
            if (inst in prop_dic['occ_range']) or ('EW' in prop_dic['meas_prop']) or ('biss' in prop_dic['meas_prop']):
                if rest_frame=='star':rv_shift_frame = np.repeat(0.,fit_dic['n_exp'])
                else:
                    if 'orig' in data_type:rv_shift_frame = coord_dic[inst][vis]['RV_star_solCDM']
                    elif 'bin' in data_type:rv_shift_frame = data_bin['RV_star_solCDM']
                if (inst in prop_dic['occ_range']):
                    occ_exclu_range = np.array(prop_dic['occ_range'][inst])[:,None] + (surf_rv_mod+rv_shift_frame[idx_in])   
                    line_range = np.array(prop_dic['line_range'][inst])[:,None] + rv_shift_frame[idx_in]                     
                    if ('spec' in data_mode):
                        occ_exclu_range = prop_dic['line_trans']*spec_dopshift(-occ_exclu_range)
                        line_range = prop_dic['line_trans']*spec_dopshift(-line_range)
                if ('EW' in prop_dic['meas_prop']):
                    prop_dic['EW_range_frame'] = np.array(prop_dic['meas_prop']['EW']['rv_range'])[:,None] + rv_shift_frame
                if ('biss' in prop_dic['meas_prop']):
                    prop_dic['biss_range_frame'] = np.array(prop_dic['meas_prop']['biss']['rv_range'])[:,None] + rv_shift_frame

        #Intrinsic profiles
        elif 'Intr' in data_type:

            #Original profiles
            if 'orig' in data_type:
                fit_dic['n_exp'] = data_inst[vis]['n_in_tr']
                iexp_def = prop_dic[inst][vis]['idx_def'] 

            #Binned profiles
            elif 'bin' in data_type:iexp_def = range(fit_dic['n_exp'])

            #Definition of ranges for signal integration in the surface rest frame
            #    - we center requested range around surface RVs in the star or surface rest frame
            if ('EW' in prop_dic['meas_prop']) or ('biss' in prop_dic['meas_prop']):
                if rest_frame=='surf':rv_shift_frame = np.repeat(0.,fit_dic['n_exp'])
                else:rv_shift_frame = surf_rv_mod
                if ('EW' in prop_dic['meas_prop']):prop_dic['EW_range_frame'] = np.array(prop_dic['meas_prop']['EW']['rv_range'])[:,None] + rv_shift_frame
                if ('biss' in prop_dic['meas_prop']):prop_dic['biss_range_frame'] = np.array(prop_dic['meas_prop']['biss']['rv_range'])[:,None] + rv_shift_frame

        #Atmospheric profiles
        elif 'Atm' in data_type:               

            #Original profiles / converted 2D->1D 
            if 'orig' in data_type:
                if len(data_inst[vis]['transit_pl'])>1:stop('Adapt model to multiple planets')
                
                #Defined exposures
                if prop_dic['pl_atm_sign']=='Absorption':fit_dic['n_exp'] = data_inst[vis]['n_in_tr']
                elif prop_dic['pl_atm_sign']=='Emission':fit_dic['n_exp'] = data_inst[vis]['n_in_visit']
                iexp_def = prop_dic[inst][vis]['idx_def']

            #Binned profiles
            elif 'bin' in data_type:iexp_def = range(fit_dic['n_exp'])

            #Definition of ranges for signal integration in the planet rest frame
            #    - we center requested range around planet RVs in the star or planet rest frame
            if ('int_sign' in prop_dic['meas_prop']) or ('EW' in prop_dic['meas_prop']):
                if rest_frame=='pl':rv_shift_frame = np.repeat(0.,fit_dic['n_exp'])
                else:
                    if 'orig' in data_type:
                        if prop_dic['pl_atm_sign']=='Absorption':rv_shift_frame = coord_dic[inst][vis]['rv_pl'][idx_in]
                        elif prop_dic['pl_atm_sign']=='Emission':rv_shift_frame = coord_dic[inst][vis]['rv_pl']
                    elif 'bin' in data_type:rv_shift_frame = data_bin[ref_pl]['rv_pl']
                if ('int_sign' in prop_dic['meas_prop']):prop_dic['int_sign_range_frame'] = np.array(prop_dic['meas_prop']['int_sign']['rv_range'])[:,None] + rv_shift_frame
                if ('EW' in prop_dic['meas_prop']):prop_dic['EW_range_frame'] = np.array(prop_dic['meas_prop']['EW']['rv_range'])[:,None] + rv_shift_frame

        #Stellar line model
        if ('DI' in data_type) or ('Intr' in data_type):        

            #Custom model
            if (prop_dic['model'][inst]=='custom'):

                #Retrieve binned intrinsic profiles if required for custom model                
                if fit_properties['mode']=='Intrbin':
                    
                    #Check that profiles were aligned
                    if fit_properties['vis']=='':vis_Intrbin = vis
                    elif fit_properties['vis']=='binned':vis_Intrbin = 'binned'
                    data_Intrbin = dataload_npz(gen_dic['save_data_dir']+'Intrbin_data/'+gen_dic['add_txt_path'][data_type_gen]+inst+'_'+vis_Intrbin+'_'+fit_properties['dim_bin']+'_add')
                    if not data_Intrbin['FromAligned']:stop('Intrinsic profiles must be aligned before binning')
    
                    #Central coordinate of binned profiles along chosen dimension
                    fit_properties['cen_dim_Intrbin'] =data_Intrbin['cen_bindim']

            #Analytical model
            #    - intrinsic profile mode is set to analytical to activate conditions, even if intrinsic profiles are not used
            else:
                fit_properties['mode']='ana'

        #Order selection
        #    - for 2D spectra we select a specific order to perform the comparison, as it is too heavy otherwise
        #    - we remove the order structure from CCF and 1D spectra to have one-dimensional tables
        if data_mode in ['CCF','spec1D']:iord_sel = 0
        elif data_mode=='spec2D':
            if inst not in prop_dic['fit_prof']['order']:stop('Define fitted order')
            else:iord_sel = prop_dic['fit_prof']['order'][inst] 

        #Trimming data
        #    - the trimming is applied to the common table, so that all processed profiles keep the same dimension after trimming
        if (trim_range is not None):
            
            #Common spectral table
            if vis_mode=='':data_com = dataload_npz(data_inst[vis]['proc_com_data_paths'])  
            elif vis_mode=='binned':data_com = dataload_npz(data_inst['proc_com_data_path'])
             
            #Trimmed range
            idx_range_kept = np_where1D((data_com['edge_bins'][iord_sel,0:-1]>=trim_range[0]) & (data_com['edge_bins'][iord_sel,1::]<=trim_range[1]))
            nspec = len(idx_range_kept)
            if nspec==0:stop('Empty trimmed range')
            
        else:
            idx_range_kept = np.arange(nspec,dtype=int)

        #Preparation
        fit_dic['cond_def_cont_all']= np.zeros([fit_dic['n_exp'],nspec],dtype=bool)
        fit_dic['cond_def_fit_all']=np.zeros([fit_dic['n_exp'],nspec],dtype=bool)
        fit_dic['idx_excl_bd_ranges']={}
        data_fit = {}
        for isub,iexp in enumerate(iexp_def):
            
            #Upload profile               
            if bin_mode=='':     data_fit_loc = dataload_npz(data_inst[vis]['proc_'+data_type_gen+'_data_paths']+str(iexp))
            elif bin_mode=='bin':data_fit_loc = np.load(gen_dic['save_data_dir']+data_type_gen+'bin_data/'+gen_dic['add_txt_path'][data_type_gen]+inst+'_'+vis_det+'_'+prop_dic['dim_bin']+str(iexp)+'.npz',allow_pickle=True)['data'].item()    
            data_fit[isub] = {}

            #Trimming profile         
            for key in ['cen_bins','flux','cond_def']:data_fit[isub][key] = data_fit_loc[key][iord_sel,idx_range_kept]
            data_fit[isub]['edge_bins'] = np.append(data_fit_loc['edge_bins'][iord_sel,idx_range_kept],data_fit_loc['edge_bins'][iord_sel,idx_range_kept[-1]+1]) 
            data_fit[isub]['dcen_bins'] = data_fit[isub]['edge_bins'][1::] - data_fit[isub]['edge_bins'][0:-1]          
            data_fit[isub]['cov'] = data_fit_loc['cov'][iord_sel][:,idx_range_kept]
            
            #Initializing ranges in the relevant rest frame
            if len(cont_range)==0:fit_dic['cond_def_cont_all'][isub] = True    
            else:
                for bd_int in cont_range:fit_dic['cond_def_cont_all'][isub] |= (data_fit[isub]['edge_bins'][0:-1]>=bd_int[0]) & (data_fit[isub]['edge_bins'][1:]<=bd_int[1])        
            if len(fit_range)==0:fit_dic['cond_def_fit_all'][isub] = True    
            else:
                for bd_int in fit_range:fit_dic['cond_def_fit_all'][isub] |= (data_fit[isub]['edge_bins'][0:-1]>=bd_int[0]) & (data_fit[isub]['edge_bins'][1:]<=bd_int[1])        
            
            #Accounting for undefined pixels
            fit_dic['cond_def_cont_all'][isub] &= data_fit[isub]['cond_def']            
            fit_dic['cond_def_fit_all'][isub] &= data_fit[isub]['cond_def']   
            
            #Exclusion of planetary ranges
            #    - not required for intrinsic profiles if already applied to their definition, and if not already applied contamination is either negligible or neglected
            #    - not required for binned disk-integrated profiles, as planetary ranges can be excluded from their construction
            #    - not required for atmospheric profiles for obvious reasons
            if (data_type=='DIorig') and ('DI_prof' in data_dic['Atm']['no_plrange']) and (iexp in data_dic['Atm'][inst][vis]['iexp_no_plrange']):   
                cond_kept_plrange,idx_excl_bd_ranges = excl_plrange(data_fit[isub]['cond_def'],data_dic['Atm'][inst][vis]['exclu_range_'+rest_frame],iexp,data_fit[isub]['edge_bins'],data_mode)
                fit_dic['idx_excl_bd_ranges'][isub] = idx_excl_bd_ranges
                fit_dic['cond_def_cont_all'][isub] &= cond_kept_plrange
                fit_dic['cond_def_fit_all'][isub]  &= cond_kept_plrange
            else:fit_dic['idx_excl_bd_ranges'][isub]=None
 
            #Exclusion of occulted stellar line range
            #    - relevant for disk-integrated profiles, in-transit
            #    - occulted lines never cover ranges outside of the disk-integrated stellar line
            if (data_type_gen=='DI') and (inst in data_dic['DI']['occ_range']) and (idx_exp2in[iexp]>-1): 
                i_in = idx_exp2in[iexp]
                cond_occ = (data_fit[isub]['edge_bins'][0:-1]>=np.max([occ_exclu_range[0,i_in],line_range[0,i_in]])) & (data_fit[isub]['edge_bins'][1:]<=np.min([occ_exclu_range[1,i_in],line_range[1,i_in]]))
                fit_dic['cond_def_cont_all'][isub,cond_occ] = False
                fit_dic['cond_def_fit_all'][isub,cond_occ] = False
       
        #Continuum common to all processed profiles
        #    - collapsed along temporal axis
        cond_cont_com  = np.all(fit_dic['cond_def_cont_all'],axis=0)
        if np.sum(cond_cont_com)==0.:stop('No pixels in common continuum')   

        #Common continuum flux in fitted intrinsic profiles
        #    - calculated over the defined bins common to all processed profiles
        #    - defined as a weighted mean because intrinsic profiles at the limbs can be very poorly defined due to the partial occultation and limb-darkening
        #    - we use the covariance diagonal to define a representative weight
        if (data_type_gen=='Intr'):
            cont_intr = np.zeros(fit_dic['n_exp'])*np.nan
            wcont_intr = np.zeros(fit_dic['n_exp'])*np.nan
            for isub in range(fit_dic['n_exp']): 
                dw_sum = np.sum(data_fit[isub]['dcen_bins'][cond_cont_com])
                cont_intr[isub] = np.sum(data_fit[isub]['flux'][cond_cont_com]*data_fit[isub]['dcen_bins'][cond_cont_com])/dw_sum
                wcont_intr[isub] = dw_sum**2./np.sum(data_fit[isub]['cov'][0,cond_cont_com]*data_fit[isub]['dcen_bins'][cond_cont_com]**2.)
            flux_cont=np.sum(cont_intr*wcont_intr)/np.sum(wcont_intr)
            
        #------------------------------------------------------------------------------------------

        #Retrieving binned intrinsic profiles for disk-integrated profile fitting
        #    - all binned profiles are defined over the same table
        if (data_type_gen=='DI') and (prop_dic['model'][inst]=='custom') and (fit_properties['mode']=='Intrbin'):
            fit_properties['edge_bins_Intrbin'] = np.append(data_Intrbin['edge_bins'][iord_sel,idx_range_kept],data_Intrbin['edge_bins'][iord_sel,idx_range_kept[-1]+1]) 
            dcen_bins_Intrbin = (fit_properties['edge_bins_Intrbin'][1::] - fit_properties['edge_bins_Intrbin'][0:-1])
            fit_properties['flux_Intrbin'] = np.zeros([data_Intrbin['n_exp'],nspec],dtype=float)
            cont_Intrbin = np.zeros(data_Intrbin['n_exp'])*np.nan
            wcont_Intrbin = np.zeros(data_Intrbin['n_exp'])*np.nan            
            for isub,iexp in enumerate(data_Intrbin['n_exp']):
                data_Intrbin_loc = np.load(gen_dic['save_data_dir']+'Intrbin_data/'+inst+'_'+vis_Intrbin+'_'+fit_properties['dim_bin']+str(iexp)+'.npz',allow_pickle=True)['data'].item()         
                if False in data_Intrbin_loc['cond_def'][iord_sel,idx_range_kept] :stop('Binned intrinsic profiles must be fully defined to be used in the reconstruction')    
                fit_properties['flux_Intrbin'][isub] = data_Intrbin_loc['flux'][iord_sel,idx_range_kept]
                cov_loc = data_Intrbin_loc['cov'][iord_sel][:,idx_range_kept]
                dw_sum = np.sum(dcen_bins_Intrbin[cond_cont_com])                
                cont_Intrbin[isub] = np.sum(fit_properties['flux_Intrbin'][isub][cond_cont_com]*dcen_bins_Intrbin[cond_cont_com])/dw_sum
                wcont_Intrbin[isub] = dw_sum**2./np.sum(cov_loc[0,cond_cont_com]*dcen_bins_Intrbin[cond_cont_com]**2.)

            #Scaling binned intrinsic profiles to a continuum unity
            flux_cont_Intrbin = np.sum(cont_Intrbin*wcont_Intrbin)/np.sum(wcont_Intrbin)
            fit_properties['flux_Intrbin']/=flux_cont_Intrbin

        #------------------------------------------------------------------------------------------

        #Process exposures    
        key_det = 'idx_force_det'+bin_mode+vis_mode
        fit_dic['cond_detected'] = np.repeat(True,fit_dic['n_exp'])
        for isub,iexp in enumerate(iexp_def):            

            #Disk-integrated profile
            if data_type_gen=='DI':                    
                if data_type=='DIorig':iexp_orig = iexp

                #Continuum flux
                flux_cont = np.sum(data_fit[isub]['flux'][cond_cont_com]*data_fit[isub]['dcen_bins'][cond_cont_com])/np.sum(data_fit[isub]['dcen_bins'][cond_cont_com])
              
                #Set constant error to the sqrt() of the continuum flux, ie covariance to the mean continuum flux
                #    - this is in case errors are not defined on disk-integrated profiles
                #    - the important is to use a constant error over the fitted range, its value can then be scaled using sc_err
                if prop_dic['cst_err'+bin_mode]:cov_exp = np.tile(flux_cont*gen_dic['g_err'][inst],[1,nspec])                         
                else:cov_exp = data_fit[isub]['cov']   
                
                #Estimate of CCF centroid
                if data_type=='DIorig':fit_properties['RV_cen'] = coord_dic[inst][vis]['RV_star_solCDM'][iexp_orig]
                elif data_type=='DIbin':
                    if gen_dic['align_DI']:fit_properties['RV_cen'] = 0. 
                    else:fit_properties['RV_cen']=data_bin['sysvel']
                
            #Intrinsic profile
            #    - see proc_intr_data(), intrinsic CCFs have the same continuum flux level, reset to the mean over all intrinsic profiles
            #      we here fix the continuum level of the model to this value, rather than fit it in each exposure 
            #    - errors are defined, so that if the fit is to be performed with a constant error we set it to the mean error over the continuum
            elif data_type_gen=='Intr':                                  
                if data_type=='Introrig':iexp_orig = gen_dic[inst][vis]['idx_in'][iexp]
                if prop_dic['cst_err'+bin_mode]:cov_exp = np.tile(np.mean(data_fit[isub]['cov'][0])*gen_dic['g_err'][inst],[1,nspec])
                else:cov_exp = data_fit[isub]['cov']

            #Atmospheric profile
            elif data_type_gen=='Atm':   
                flux_cont = 0.                  
                if data_type=='Atmorig': 
                    if prop_dic['pl_atm_sign']=='Absorption':iexp_orig = gen_dic[inst][vis]['idx_in'][iexp]
                    elif prop_dic['pl_atm_sign']=='Emission':iexp_orig = iexp
                    fit_properties['RV_cen']=coord_dic[inst][vis]['rv_pl'][iexp_orig]     #guess
                if prop_dic['cst_err'+bin_mode]:cov_exp = np.tile(np.mean(data_fit[isub]['cov'][0]),[1,nspec])
                else:cov_exp = data_fit[isub]['cov']

            #Scaling data errors
            if (inst in prop_dic['sc_err']) and (vis_det in prop_dic['sc_err'][inst]):cov_exp*=prop_dic['sc_err'][inst][vis]**2. 


            #Forced detection
            if (inst in prop_dic[key_det]) and (vis_det in prop_dic[key_det][inst]) and (iexp in prop_dic[key_det][inst][vis_det]):idx_force_det=prop_dic[key_det][inst][vis_det][iexp]
            else:idx_force_det=None 
          
            #-------------------------------------------------

            #Perform analysis of individual profile
            #    - profiles are fitted on their original table, converted in RV space for single spectral line fitted with analytical models
            #    - analytical models can be calculated on an oversampled table and resampled before the fit
            fit_properties.update({'iexp':iexp,'flux_cont':flux_cont})
            fit_dic[iexp]=ana_prof_func(isub,iexp,inst,data_dic,vis_det,prop_dic,gen_dic,prop_dic['verbose'],fit_dic['cond_def_fit_all'][isub],fit_dic['cond_def_cont_all'][isub] ,data_type_gen,data_fit[isub]['edge_bins'],data_fit[isub]['cen_bins'],data_fit[isub]['flux'],cov_exp,
                                        idx_force_det,theo_dic,star_param,fit_properties,prop_dic['line_fit_priors'],prop_dic['model'][inst],prop_dic['mod_prop'],data_mode)

            #-------------------------------------------------

            #Detection flag
            fit_dic['cond_detected'][iexp] = fit_dic[iexp]['detected']

            #Calculating residuals from Keplerian (km/s)
            #    - we report the errors on the raw velocities 
            if data_type=='DIorig':
                fit_dic[iexp]['RVmod']=coord_dic[inst][vis]['RV_star_solCDM'][iexp]
                if ('rv_pip' in prop_dic[inst][vis]):
                    fit_dic[iexp]['rv_pip_res']=prop_dic[inst][vis]['rv_pip'][iexp]-fit_dic[iexp]['RVmod']
                    fit_dic[iexp]['err_rv_pip_res']=np.repeat(prop_dic[inst][vis]['erv_pip'][iexp],2)  
                if prop_dic['model'][inst]=='dgauss':
                    fit_dic[iexp]['RV_lobe_res']=fit_dic[iexp]['RV_lobe']-fit_dic[iexp]['RVmod']
                    fit_dic[iexp]['err_RV_lobe_res']=fit_dic[iexp]['err_RV_lobe']  

            elif data_type=='DIbin':    
                fit_dic[iexp]['RVmod']=data_bin['RV_star_solCDM'][iexp]

            #Calculating residuals from RRM model (km/s)
            #    - we report the errors on the local velocities
            elif data_type=='Introrig':
                fit_dic[iexp]['RVmod']=surf_rv_mod[iexp] 
               
            #Calculating residuals from orbital RVs (km/s)
            elif data_type=='Atmorig': 
                fit_dic[iexp]['RVmod']=coord_dic[inst][vis]['rv_pl'][iexp_orig]
            
            #Errors are reported from measured velocities
            fit_dic[iexp]['rv_res']=fit_dic[iexp]['rv']-fit_dic[iexp]['RVmod']
            fit_dic[iexp]['err_rv_res']=fit_dic[iexp]['err_rv']

        #Printing out measured systemic velocity (rv(CDM/sun) in km/s)
        #    - for CCFs, the disk-integrated master is centered in the CDM rest frame, hence its fit returns the systemic velocity         
        #      we assume the master is defined well enough that the uncertainty on the velocity v(CDM/sun) is negligible
        if (data_type=='DIbin') and (fit_dic['n_exp']==1):
            rv_sys,erv_sys=0.,0.
            for iexp in iexp_def:
                rv_sys+=fit_dic[iexp]['rv']
                erv_sys+=np.mean(fit_dic[iexp]['err_rv'])**2.
            rv_sys/=fit_dic['n_exp']
            erv_sys = np.sqrt(erv_sys)/fit_dic['n_exp']
            print('         Measured systemic velocity =',"{0:.6f}".format(rv_sys),'+-',"{0:.6e}".format(erv_sys),'km/s')

        #Saving data
        fit_dic['cont_range'] = cont_range
        fit_dic['fit_range'] = fit_range        
        np.savez_compressed(save_path,data=fit_dic,allow_pickle=True)

    #Checking data has been calculated
    else:
        data_paths={'path':save_path}      
        check_data(data_paths)    
                                  
    return None













'''
Sub-function to initalize temporal/spatial resampling routine
'''
def init_bin_rout(data_type,bin_prop,idx_in_bin,dim_bin,coord_dic,inst,vis_to_bin,data_dic,gen_dic):

    #Concatenate tables from all visits to bin
    #    - indexes of original exposures are arbitrarily offset between visits to be distinguishable
    x_low_vis = np.zeros(0,dtype=float)
    x_high_vis = np.zeros(0,dtype=float)
    idx_orig_vis = np.zeros(0,dtype=int)
    vis_orig=np.zeros(0,dtype='U35')
    idx_to_bin_vis = np.zeros(0,dtype=int)
    vis_shift = 0
    for vis in vis_to_bin:

        #Indexes of exposures requested for binning      
        if (inst in idx_in_bin) and (vis in idx_in_bin[inst]) and len(idx_in_bin[inst][vis])>0:idx_in_bin_vis = np.array(idx_in_bin[inst][vis]) 
        else:idx_in_bin_vis = None
    
        #Indexes of original exposures used as input, and default indexes of those exposures that contribute to the binned profiles
        #    - idx_in_bin indexes are relative to all exposures or only in-transit exposures depending on the case
        #    - for aligned disk-integrated profiles, all exposures are considered if a selection is requested
        #      if no selection is done, only out-of-transit exposures are considered
        #      indexes are relative to global tables
        #    - for aligned intrinsic profiles, corresponds to in-transit indexes of exposures with known surface rvs
        #    - for absorption profiles, indexes are relative to in-transit tables
        #      for emission profiles, indexes are relative to global tables       
        if data_type in ['DI','Res']:
            idx_orig = np.arange(data_dic[inst][vis]['n_in_visit'])
            if idx_in_bin_vis is not None:idx_to_bin = idx_in_bin_vis
            else:idx_to_bin = gen_dic[inst][vis]['idx_out']
        elif data_type=='Intr':
            idx_orig = gen_dic[inst][vis]['idx_in']
            idx_to_bin = data_dic['Intr'][inst][vis]['idx_def']    
            if idx_in_bin_vis is not None:idx_to_bin = np.intersect1d(idx_in_bin_vis,idx_to_bin)
        elif data_type in ['Absorption','Emission']:
            if data_type=='Absorption':idx_orig = gen_dic[inst][vis]['idx_in']
            elif data_type=='Emission':idx_orig = np.arange(data_dic[inst][vis]['n_in_visit'])
            idx_to_bin = data_dic['Atm'][inst][vis]['idx_def']
            if idx_in_bin_vis is not None:idx_to_bin = np.intersect1d(idx_in_bin_vis,idx_to_bin)        
    
        #Coordinate tables of input exposures along chosen bin dimension
        #    - tables are restricted to the range of input exposures
        if dim_bin in ['phase']:     
            coord_vis = coord_dic[inst][vis][bin_prop['ref_pl']]
            x_low = coord_vis['st_ph'][idx_orig]
            x_high = coord_vis['end_ph'][idx_orig]
            x_cen_all = coord_vis['cen_ph'][idx_orig]         #central coordinates of all exposures, required for def_plocc_profiles()
            
        elif dim_bin in ['xp_abs','r_proj']: 
            transit_prop_nom = (np.load(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item())['achrom'][bin_prop['ref_pl']]                           
            x_low = transit_prop_nom[dim_bin+'_range'][0,:,0]
            x_high = transit_prop_nom[dim_bin+'_range'][0,:,1]    
            x_cen_all = transit_prop_nom[dim_bin][0,:]      #central coordinates of all exposures, required for def_plocc_profiles()  
        
        #Selection of input exposures contributing to binned profiles
        if len(idx_to_bin)==0:stop('No remaining exposures after input selection')
        x_low = x_low[idx_to_bin]
        x_high = x_high[idx_to_bin]  
    
        #Append all visits
        x_low_vis = np.append(x_low_vis,x_low)
        x_high_vis = np.append(x_high_vis,x_high)
        idx_orig_vis = np.append(idx_orig_vis,idx_to_bin)
        vis_orig = np.append(vis_orig,np.repeat(vis,len(idx_to_bin)))
        idx_to_bin_vis = np.append(idx_to_bin_vis,idx_to_bin+vis_shift)
        vis_shift+=data_dic[inst][vis]['n_in_visit']
   
    #Dictionary to match indexes in concatenated tables toward original indexes and visits
    idx_bin2orig = dict(zip(idx_to_bin_vis, idx_orig_vis))   
    idx_bin2vis = dict(zip(idx_to_bin_vis, vis_orig)) 

    #Coordinates of binned exposures along bin dimension
    if 'bin_low' in bin_prop:
        new_x_low_in = np.array(bin_prop['bin_low'])
        new_x_high_in = np.array(bin_prop['bin_high'])
    elif 'bin_range' in bin_prop:
        min_x = max([bin_prop['bin_range'][0],min(x_low_vis)])
        max_x = min([bin_prop['bin_range'][1],max(x_high_vis)])
        new_dx =  (max_x-min_x)/bin_prop['nbins']
        new_nx = int((max_x-min_x)/new_dx)
        new_x_low_in = min_x + new_dx*np.arange(new_nx)
        new_x_high_in = new_x_low_in+new_dx 
    
    #Limiting contributing profiles to requested range along bin dimension            
    cond_keep = (x_high_vis>=new_x_low_in[0]) &  (x_low_vis <=new_x_high_in[-1])
    x_low_vis = x_low_vis[cond_keep]
    x_high_vis = x_high_vis[cond_keep]         
    idx_to_bin_vis = idx_to_bin_vis[cond_keep]
    if np.sum(cond_keep)==0:stop('No original exposures in bin range')  
  
    #Limiting binned profiles to the original exposure range
    cond_keep = (new_x_high_in>=min(x_low_vis)) &  (new_x_low_in <=max(x_high_vis))
    new_x_low_in = new_x_low_in[cond_keep]        
    new_x_high_in = new_x_high_in[cond_keep]  
    if np.sum(cond_keep)==0:stop('No binned exposures in original range')     
    
    #Properties of binned profiles along chosen dimension
    idx_to_bin_all = []
    n_in_bin_all = []
    dx_ov_all=[]
    new_x_low = np.zeros(0,dtype=float)
    new_x_high = np.zeros(0,dtype=float)
    new_x_cen = np.zeros(0,dtype=float) 
    for i_new,(new_x_low_in_loc,new_x_high_in_loc) in enumerate(zip(new_x_low_in,new_x_high_in)):

        #Original exposures overlapping with new bin
        #    - we use 'where' rather than searchsorted to allow processing original bins that overlap together
        idx_olap = np_where1D( (x_high_vis>=new_x_low_in_loc) &  (x_low_vis <=new_x_high_in_loc) )
        if len(idx_olap)>0:
            
            #Indexes of contributing exposures relative to the original indexes of input exposures
            #    - offset between visits if relevant
            idx_to_bin_all+=[idx_to_bin_vis[idx_olap]]

            #Number of overlapping original exposures
            n_in_bin = len(idx_olap)
            n_in_bin_all += [n_in_bin]

            #Minimum between overlapping exposure upper boundaries and new exposure upper boundary
            #    - if the exposure upper boundary is beyond the new exposure, then the exposure fraction beyond the new exposure upper boundary will not contribute to the binned flux 
            #    - if the exposure upper boundary is within the new exposure, then the exposure will only contribute to the binned flux up to its own boundary
            x_high_ov=np.minimum(x_high_vis[idx_olap],np.repeat(new_x_high_in_loc,n_in_bin))
        
            #Maximum between overlapping exposure lower boundaries and new exposure lower boundary
            #    - if the exposure lower boundary is beyond the new exposure, then the exposure fraction beyond the new exposure upper boundary will not contribute to the binned flux 
            #    - if the exposure lower boundary is within the new exposure, then the exposure will only contribute to the binned flux up to its own boundary
            x_low_ov=np.maximum(x_low_vis[idx_olap],np.repeat(new_x_low_in_loc,n_in_bin))
            
            #Effective boundaries of the new exposure
            new_x_low_loc = min(x_low_ov)
            new_x_high_loc = max(x_high_ov)

            #Center for new exposure
            #    - defined as the barycenter of all overlaps centers
            new_x_cen_loc = np.mean(0.5*(x_low_ov+x_high_ov))
        
            #Width over which each original exposure contributes to the binned flux
            dx_ov_all+=[x_high_ov-x_low_ov]

            #Store updated bin boundary and center
            new_x_low=np.append(new_x_low,new_x_low_loc)
            new_x_high=np.append(new_x_high,new_x_high_loc)
            new_x_cen = np.append(new_x_cen,new_x_cen_loc)
            
    n_bin = len(new_x_cen)
    
    #Unique list of original exposures that will be used in the binning
    #    - indexes are relative to original exposures, but have been offset in case several visits are binned
    idx_to_bin_unik = np.unique(np.concatenate(idx_to_bin_all))      
    
    return new_x_cen,new_x_low,new_x_high,x_cen_all,n_in_bin_all,idx_to_bin_all,dx_ov_all,n_bin,idx_bin2orig,idx_bin2vis,idx_to_bin_unik



'''
Sub-routine to return phase of other planet than the reference
'''
def conv_phase(coord_dic,inst,vis,system_param,ref_pl,pl_loc,phase_tab_in):
    if (ref_pl is None) or (pl_loc==ref_pl):     
        phase_tab_out=phase_tab_in
    else:
        phase_tab_out = deepcopy(phase_tab_in) 
        Tcenter_ref = coord_dic[inst][vis][ref_pl]['Tcenter']
        Tcenter_loc = coord_dic[inst][vis][pl_loc]['Tcenter']   
        for iph in range(phase_tab_in.shape[0]):   

            #Absolute time table and corresponding phases   
            cen_bjd = Tcenter_ref+phase_tab_in[iph]*system_param[ref_pl]["period"]
            phase_temp=(cen_bjd-Tcenter_loc)/system_param[pl_loc]["period"]
            phase_tab_out[iph] = (phase_temp - npint(phase_temp+np.sign(phase_temp)*0.5))        

    return phase_tab_out






'''    
Calculate stellar continuum of master spectrum
'''
def calc_spectral_cont(nord,iord_proc_list,flux_mast_in,cen_bins_mast,edge_bins_mast,cond_def_mast,flux_Earth_all_in,cond_def_all_in,inst,roll_win,smooth_win,locmax_win,par_stretch,\
                       contin_pinR,min_edge_ord,dic_sav,nthreads):
    mean_flux_mast = np.zeros(nord,dtype=float)*np.nan
    if flux_mast_in is None:
        flux_mast=np.repeat(None,len(iord_proc_list))
        flux_Earth_all = deepcopy(flux_Earth_all_in)   
        cond_def_all = deepcopy(cond_def_all_in)    
    else:
        flux_mast = deepcopy(flux_mast_in)
        flux_Earth_all=np.tile(None,[1,len(iord_proc_list)])
        cond_def_all=np.tile(None,[1,len(iord_proc_list)])
    common_args = (inst,roll_win,smooth_win,locmax_win,par_stretch,contin_pinR,min_edge_ord)
    if nthreads>1:mean_flux_mast[iord_proc_list],dic_sav_loc = para_sub_calc_spectral_cont(sub_calc_spectral_cont,nthreads,len(iord_proc_list),[iord_proc_list,cond_def_mast,cen_bins_mast,flux_mast,edge_bins_mast,flux_Earth_all,cond_def_all],common_args)                           
    else:mean_flux_mast[iord_proc_list],dic_sav_loc = sub_calc_spectral_cont(iord_proc_list,cond_def_mast,cen_bins_mast,flux_mast,edge_bins_mast,flux_Earth_all,cond_def_all,*common_args)  
    dic_sav.update(dic_sav_loc)

    #----------------------------------------------------
    #Define continuum function for current order
    #    - performed outside of the parrallelized loop because for some reason it raises issues with 'cont_func_dic'
    cont_func_dic={}
    for isub_ord,iord in enumerate(iord_proc_list):
        cont_func_dic[iord] = interp1d(dic_sav[iord]['wav_max'],dic_sav[iord]['flux_max'],fill_value='extrapolate')  

    return mean_flux_mast,cont_func_dic,dic_sav


def para_sub_calc_spectral_cont(func_input,nthreads,n_elem,y_inputs,common_args):    
    pool_proc = Pool(processes=nthreads)   #cannot be passed through lmfit
    ind_chunk_list=init_parallel_func(nthreads,n_elem)
    chunked_args=[tuple(y_inputs[i][ind_chunk[0]:ind_chunk[1]] for i in range(5))+(y_inputs[5][:,ind_chunk[0]:ind_chunk[1]],y_inputs[6][:,ind_chunk[0]:ind_chunk[1]],)+common_args for ind_chunk in ind_chunk_list]	
    all_results=tuple(tab for tab in pool_proc.starmap(func_input,chunked_args))	

    mean_flux_mast = np.concatenate(tuple(all_results[i][0] for i in range(nthreads)),axis=0)
    dic_sav = {}
    for dic_save_proc in tuple(all_results[i][1] for i in range(nthreads)):dic_sav.update(dic_save_proc)
    y_output=(mean_flux_mast,dic_sav) 
    
    pool_proc.close()
    pool_proc.join() 				
    return y_output

def sub_calc_spectral_cont(iord_proc_list,cond_def_mast,cen_bins_mast,flux_mast,edge_bins_mast,flux_Earth_all,cond_def_all,inst,roll_win,smooth_win,locmax_win,par_stretch,contin_pinR,min_edge_ord):
    mean_flux_mast = np.zeros(len(iord_proc_list),dtype=float)*np.nan
    dic_sav={}
    low_edge_bins = edge_bins_mast[:,0:-1]
    high_edge_bins = edge_bins_mast[:,1::]
    dcen_bins = high_edge_bins - low_edge_bins  
    for isub_ord,iord in enumerate(iord_proc_list):

        #Master tables over order
        cond_def = cond_def_mast[isub_ord]
        cen_bins_ord = cen_bins_mast[isub_ord,cond_def]
        low_edge_bins_ord = low_edge_bins[isub_ord][cond_def]
        high_edge_bins_ord = high_edge_bins[isub_ord][cond_def]
        dcen_bins_ord = dcen_bins[isub_ord][cond_def]
        
        #Calculate master from median flux if undefined
        if flux_mast[isub_ord] is None:
            flux_mast_ord = np.zeros(np.sum(cond_def))*np.nan                 
            for ipix_sub,ipix in enumerate(np_where1D(cond_def)):flux_mast_ord[ipix_sub] = np.median(flux_Earth_all[cond_def_all[:,isub_ord,ipix],isub_ord,ipix])
        else:
            flux_mast_ord = flux_mast[isub_ord,cond_def]
       
        #Mean master flux in order
        mean_flux_mast[isub_ord] = np.sum(flux_mast_ord*dcen_bins_ord)/np.sum(dcen_bins_ord)

        #Mean bin size over order (A)                
        dwav_mean = np.mean(dcen_bins_ord)

        #-------------------------------------------------------
        #Preliminary peak exclusion
        
        #Half-integer range over which peaks are excluded
        #    - exclusion range set to 3*fwhm of the inst
        fwhm_wav = return_FWHM_inst(inst,np.mean(cen_bins_ord))  
        hn_exc = int(np.round(0.5*(3*fwhm_wav /dwav_mean)))

        #Apply successive sigma-clipping to remove spurious peaks
        #    - this step should not be necessary if the cosmic correction module was applied
        pd_spectre = pd.DataFrame(flux_mast_ord)
        n_roll = int(np.round(roll_win/dwav_mean))
        for iteration in range(5): 
            
            #Envelopes of the spectrum at different quantile
            maxi_roll_fast = np.ravel(pd_spectre.rolling(5*n_roll,min_periods=1,center=True).quantile(0.99))
            Q3_fast = np.ravel(pd_spectre.rolling(n_roll,min_periods=1,center=True).quantile(0.75)) 
            median = np.ravel(pd_spectre.rolling(n_roll,min_periods=1,center=True).quantile(0.50))

            #Condition of peak removal
            sup_fast = Q3_fast+3*(Q3_fast-median)  
            mask = (flux_mast_ord>sup_fast) & (flux_mast_ord>maxi_roll_fast)
            
            #Replace peaks and the chosen range around them
            mask_range = deepcopy(mask)
            for j in range(1,hn_exc):mask_range  |= np.roll(mask,-j) | np.roll(mask,j) 
            flux_mast_ord[mask_range] = median[mask_range]

        #-------------------------------------------------------
        #Smoothing
        #    - using Savitzky-Golay filter   
        n_smooth_win = int(smooth_win/dwav_mean)
        if not is_odd(n_smooth_win):n_smooth_win+=1
        smooth_flux = savgol_filter(flux_mast_ord, n_smooth_win, 3)

        #Suppress the smoothing of sharp peaks that create sinc-like wiggle
        spectre_backup = flux_mast_ord.copy()
        median = np.median(abs(spectre_backup-smooth_flux))
        IQ = np.percentile(abs(spectre_backup-smooth_flux),75) - median
        mask_out = np.where(abs(spectre_backup-smooth_flux)>(median+20*IQ))[0]
        mask_out = np.unique(mask_out+np.arange(-n_smooth_win,n_smooth_win+1,1)[:,np.newaxis])
        mask_out = mask_out[(mask_out>=0)&(mask_out<len(cen_bins_ord))].astype('int')
        smooth_flux[mask_out] = spectre_backup[mask_out] 

        #Set negative flux values to 0
        smooth_flux[smooth_flux<0] = 0. 

        #-------------------------------------------------------
        #Identification of local maxima 
        hn_locmax = int(np.round(0.5*locmax_win/dwav_mean))
        smooth_flux_noedge = smooth_flux[hn_locmax:-hn_locmax]
        maxima = np.ones(len(smooth_flux_noedge))
        for k in range(1,hn_locmax):
            maxima *= 0.5*(1+np.sign(smooth_flux_noedge - smooth_flux[hn_locmax-k:-hn_locmax-k]))*0.5*(1+np.sign(smooth_flux_noedge - smooth_flux[hn_locmax+k:-hn_locmax+k]))
        index = np_where1D(maxima==1)+hn_locmax
        maxima_flux = smooth_flux[index]
        maxima_cen_bins = cen_bins_ord[index]

        #Dealing with edge issues
        if maxima_flux[0] < smooth_flux[0]:
            maxima_cen_bins = np.insert(maxima_cen_bins,0,cen_bins_ord[0])
            maxima_flux = np.insert(maxima_flux,0,smooth_flux[0])
        if maxima_flux[-1] < smooth_flux[-1]:
            maxima_cen_bins = np.hstack([maxima_cen_bins,cen_bins_ord[-1]])
            maxima_flux = np.hstack([maxima_flux,smooth_flux[-1]])

        #Removing cosmics
        pd_max_flux = pd.DataFrame(maxima_flux)
        median = np.ravel(pd_max_flux.rolling(10,center=True).quantile(0.50))
        IQ = np.ravel(pd_max_flux.rolling(10,center=True).quantile(0.75)) - median
        IQ[np.isnan(IQ)] = smooth_flux.max()
        median[np.isnan(median)] = smooth_flux.max()
        cond_nopeaks = (maxima_flux <= median + 20 * IQ)
        maxima_cen_bins = maxima_cen_bins[cond_nopeaks]
        maxima_flux = maxima_flux[cond_nopeaks]

        #-------------------------------------------------------
        #Stretching
        #    - Y-axis stretching value
        #    - to scale the x and y axis    
        normalisation = (np.max(flux_mast_ord) - np.min(flux_mast_ord))/(high_edge_bins_ord[-1]  - low_edge_bins_ord[0])
        maxima_flux = maxima_flux/par_stretch/normalisation
        normalisation = normalisation*par_stretch

        #----------------------------------------------------
        #Rolling pin
        waves = maxima_cen_bins - maxima_cen_bins[:,np.newaxis]
        distance = np.sign(waves)*np.sqrt((waves)**2+(maxima_flux - maxima_flux[:,np.newaxis])**2)
        distance[distance<0] = 0

        #Chromatic radius, constant in velocity space
        radius = np.repeat(contin_pinR,len(maxima_cen_bins))*maxima_cen_bins/min_edge_ord
        radius[0] = radius[0]/1.5
        
        #Applying rolling pin
        numero = np.arange(len(distance)).astype('int')
        j = 0
        keep = [0]
        while (len(maxima_cen_bins)-j>3): 
            par_R = float(radius[j]) #take the radius from the penality law
            mask = (distance[j,:]>0)&(distance[j,:]<2.*par_R) #recompute the points closer than the diameter if radius changed with the penality      
            while np.sum(mask)==0:
                par_R *=1.5
                mask = (distance[j,:]>0)&(distance[j,:]<2.*par_R) #recompute the points closer than the diameter if radius changed with the penality      
            p1 = np.array([maxima_cen_bins[j],maxima_flux[j]]).T #vector of all the local maxima 
            p2 = np.array([maxima_cen_bins[mask],maxima_flux[mask]]).T #vector of all the maxima in the diameter zone
            delta = p2 - p1 # delta x delta y
            c = np.sqrt(delta[:,0]**2+delta[:,1]**2) # euclidian distance 
            h = np.sqrt(par_R**2-0.25*c**2) 
            cx = p1[0] + 0.5*delta[:,0] - h/c*delta[:,1] #x coordinate of the circles center
            cy = p1[1] + 0.5*delta[:,1] + h/c*delta[:,0] #y coordinates of the circles center
            cond1 = (cy-p1[1])>=0
            thetas = cond1*(-1*np.arccos((cx - p1[0])/par_R)+np.pi) + (1-1*cond1)*(-1*np.arcsin((cy - p1[1])/par_R) + np.pi)
            j2 = thetas.argmin()
            j = numero[mask][j2] #take the numero of the local maxima falling in the diameter zone
            keep.append(j)
        maxima_flux = maxima_flux[keep] #we only keep the local maxima with the rolling pin condition
        maxima_cen_bins = maxima_cen_bins[keep]

        #----------------------------------------------------
        #Removing outlying maxima
        diff_deri = abs(np.diff(np.diff(maxima_flux)/np.diff(maxima_cen_bins)))
        mask_out = diff_deri  > (np.percentile(diff_deri,99.5))
        mask_out = np.array([False]+mask_out.tolist()+[False])
        maxima_cen_bins = maxima_cen_bins[~mask_out]
        maxima_flux = maxima_flux[~mask_out]
        norm_maxima_flux = maxima_flux*normalisation

        #Save
        dic_sav[iord]={'wav_max':maxima_cen_bins,'flux_max':norm_maxima_flux}  
        if flux_mast[isub_ord] is None:
            dic_sav[iord].update({'wav_mast':cen_bins_ord,'flux_mast':flux_mast_ord})       

    return mean_flux_mast,dic_sav










'''
Routine to combine profiles in new bins along a chosen dimension
    - for a given visit or between several visits
    - binned profiles are calculated as means weighted by specific weights depending on the type of profiles
    - for analysis purpose or use outside the pipeline
      masters used to extract local stellar profiles from each exposure are calculated in extract_res_profiles() 
'''
def process_bin_prof(mode,data_type_gen,gen_dic,inst,vis_in,data_dic,coord_dic,data_prop,system_param,theo_dic,plot_dic,masterDI=False):
    data_inst = data_dic[inst]    
        
    #Identifier for saved file
    if mode=='multivis':vis_save = 'binned'      
    else:vis_save = vis_in 

    if data_type_gen=='DI':   
        data_type='DI'
        prop_dic = deepcopy(data_dic['DI']) 

        #Set default binning properties to calculate master
        #    - if undefined, or if a single master is requested (overwrites settings in this case, and always calculated for a single visit)
        if masterDI or (inst not in prop_dic['prop_bin']) or (vis_save not in list(prop_dic['prop_bin'][inst].keys())):
            prop_dic['dim_bin']='phase'
            if inst not in prop_dic['prop_bin']:prop_dic['prop_bin'][inst] = {}
            if (vis_save not in prop_dic['prop_bin'][inst]):prop_dic['prop_bin'][inst]={vis_save:{'bin_low':[-0.5],'bin_high':[0.5]}} 
            
        #Calculation of a master spectrum used for weighing
        #    - calculated per visit
        #    - for the purpose of weighing we calculate a single master on the common spectral table of the visit, rather than recalculate the master for every weighing profile
        #      we assume the blurring that will be introduced by the resampling of this master on the table of each exposure is negligible in the weighing process
        if masterDI:
            calc_check=gen_dic['calc_DImast']
            print('   > Calculating master stellar spectrum')  
            if (data_dic['DI'][inst][vis_in]['rest_frame']!='star') and (data_dic['DI']['sysvel'][inst][vis_in]!=0.):print('WARNING: disk-integrated profiles must be aligned')
            prop_dic['idx_in_bin']=gen_dic['DImast_idx_in_bin']
       
            #Initialize path of weighing master for disk-integrated exposures
            #    - the paths return to the single master common to all exposures, and for now defined on the same table
            data_dic[inst][vis_in]['mast_'+data_type+'_data_paths'] = {iexp:gen_dic['save_data_dir']+data_type+'_data/Master/'+inst+'_'+vis_in+'_phase' for iexp in range(data_dic[inst][vis_in]['n_in_visit'])}
            save_pref = gen_dic['save_data_dir']+data_type+'_data/Master/'+inst+'_'+vis_in+'_phase'
        else:
            print('   > Binning disk-integrated profiles over '+prop_dic['dim_bin']) 

    elif data_type_gen=='Intr':
        data_type='Intr'
        prop_dic = deepcopy(data_dic['Intr'])
        print('   > Binning intrinsic profiles over '+prop_dic['dim_bin'])
    elif data_type_gen == 'Atm':
        data_type = data_dic['Atm']['pl_atm_sign']
        prop_dic = deepcopy(data_dic['Atm'])
        print('   > Binning atmospheric profiles over '+prop_dic['dim_bin'])       
    if not masterDI:
        calc_check=gen_dic['calc_'+data_type_gen+'bin'+mode]
        if mode=='multivis':prop_dic[inst]['binned']={} 
        save_pref = gen_dic['save_data_dir']+data_type_gen+'bin_data/'
        if data_type_gen=='Atm':save_pref+=data_dic['Atm']['pl_atm_sign']+'/'
        save_pref+=inst+'_'+vis_save+'_'+prop_dic['dim_bin']

    if (calc_check):
        print('         Calculating data') 

        #Set default binning properties to calculate single master
        if (inst not in prop_dic['prop_bin']):prop_dic['prop_bin'][inst] = {}
        if (vis_save not in list(prop_dic['prop_bin'][inst].keys())):
            if prop_dic['dim_bin']=='phase':prop_dic['prop_bin'][inst]={vis_save:{'bin_low':[-0.5],'bin_high':[0.5]}} 
            elif prop_dic['dim_bin']=='r_proj':prop_dic['prop_bin'][inst]={vis_save:{'bin_low':[0.],'bin_high':[1.]}}             
            else:stop('Undefined')
            
        #Bin properties
        bin_prop = prop_dic['prop_bin'][inst][vis_save]
               
        #Several visits are binned together
        #    - common table and dimensions are shared between visits
        #    - new coordinates are relative to a planet chosen as reference for the binned coordinates, which must be present in all binned visits 
        if mode=='multivis':

            #Retrieving table common to all visits
            #    - defined in input rest frame for spectra, and centered on the input systemic velocity for CCFs
            data_com = np.load(data_inst['proc_com_data_path']+'.npz',allow_pickle=True)['data'].item()      
            dim_exp_com = data_inst['dim_exp']  
            nspec_com = data_inst['nspec'] 
   
            #Visits to include in the binning
            vis_to_bin = prop_dic['vis_in_bin'][inst] if ((inst in data_dic[data_type_gen]['vis_in_bin']) and (len(data_dic[data_type_gen]['vis_in_bin'][inst])>0)) else data_dic[inst]['visit_list']

            #Planets associated with the binned visits
            data_inst.update({vis_save:{'transit_pl':[]}})
            for vis_bin in vis_to_bin:data_inst[vis_save]['transit_pl']+=data_inst[vis_bin]['transit_pl']
            data_inst[vis_save]['transit_pl'] = list(np.unique(data_inst[vis_save]['transit_pl']))
            
            #Mean systemic velocity 
            sysvel=0.
            for vis_bin in vis_to_bin:sysvel+=data_dic['DI']['sysvel'][inst][vis_bin]
            sysvel/=len(vis_to_bin)
            
            #Common data type
            data_mode = data_inst['type']
            
            #Common rest frame
            rest_frame=[]
            for vis_bin in vis_to_bin:rest_frame+=[data_dic['DI'][inst][vis_bin]['rest_frame']]
            if len(np.unique(rest_frame))>1:stop('Incompatible rest frames') 
            rest_frame = np.unique(rest_frame)[0]
           
        #A single visit is processed
        #    - common table and dimensions are specific to this visit
        elif mode=='': 
            data_com = np.load(data_inst[vis_in]['proc_com_data_paths']+'.npz',allow_pickle=True)['data'].item() 
            dim_exp_com = data_inst[vis_in]['dim_exp']
            nspec_com = data_inst['nspec']
            vis_to_bin=[vis_in]
            sysvel = data_dic['DI']['sysvel'][inst][vis_in]
            data_mode = data_inst[vis_in]['type']
            rest_frame = data_dic['DI'][inst][vis_in]['rest_frame']
 
        #Store flags
        #    - 'FromAligned' set to True if binned profiles were aligned before
        #    - 'in_inbin' set to True if binned profiles include at least one in-transit profile
        data_glob_new={'FromAligned':gen_dic['align_'+data_type_gen],'in_inbin' : False}
        
        #Automatic definition of reference planet 
        #    - for single-transiting planet or if undefined    
        if (len(data_inst[vis_save]['transit_pl'])==1) or ('ref_pl' not in bin_prop):bin_prop['ref_pl'] = data_inst[vis_save]['transit_pl'][0]  

        #Initialize binning
        new_x_cen,new_x_low,new_x_high,_,n_in_bin_all,idx_to_bin_all,dx_ov_all,n_bin,idx_bin2orig,idx_bin2vis,idx_to_bin_unik = init_bin_rout(data_type,bin_prop,prop_dic['idx_in_bin'],prop_dic['dim_bin'],coord_dic,inst,vis_to_bin,data_dic,gen_dic)

        #Retrieving data that will be used in the binning
        #    - original data is associated with its original index, so that it can be retrieved easily by the binning routine
        #      for each new bin, the binning routine is called with the list of original index that it will use
        #    - different binned profiles might use the same original exposures, which is why we use 'idx_to_bin_unik' to pre-process only once original exposures 
        data_to_bin={}
        if (data_type_gen=='DI') and (not masterDI):data_glob_new['RV_star_solCDM'] = np.zeros(n_bin,dtype=float)
        data_glob_new['vis_iexp_in_bin']={vis_bin:{} for vis_bin in vis_to_bin}
        for iexp_off in idx_to_bin_unik:
            data_to_bin[iexp_off]={}
            
            #Original index and visit of contributing exposure
            #    - iexp is relative to global or in-transit indexes depending on data_type
            iexp = idx_bin2orig[iexp_off]
            vis_bin = idx_bin2vis[iexp_off]            
            data_glob_new['vis_iexp_in_bin'][vis_bin][iexp]={}
            if (data_type not in ['Intr','Absorption']) and (gen_dic[inst][vis_bin]['idx_exp2in'][iexp]!=-1.):data_glob_new['in_inbin']=True
            
            #Latest processed data
            #    - profiles should have been aligned in the star rest frame and rescaled to their correct flux level, if necessary
            flux_est_loc_exp = None
            cov_est_loc_exp = None
            SpSstar_spec = None
            data_exp = dataload_npz(data_inst[vis_bin]['proc_'+data_type_gen+'_data_paths']+str(iexp))
            data_glob_new['vis_iexp_in_bin'][vis_bin][iexp]['data_path'] = data_inst[vis_bin]['proc_'+data_type_gen+'_data_paths']+str(iexp)
            if gen_dic['flux_sc']:scaled_data_paths = data_dic[inst][vis_bin]['scaled_'+data_type_gen+'_data_paths']
            else:scaled_data_paths = None
            if data_inst[vis_bin]['tell_sp']:
                data_exp['tell'] = dataload_npz(data_inst[vis_bin]['tell_'+data_type_gen+'_data_paths'][iexp])['tell']  
                data_glob_new['vis_iexp_in_bin'][vis_bin][iexp]['tell_path'] = data_inst[vis_bin]['tell_'+data_type_gen+'_data_paths'][iexp]
            else:data_exp['tell'] = None
            if gen_dic['cal_weight'] and data_inst[vis_bin]['mean_gdet']:data_exp['mean_gdet'] = dataload_npz(data_inst[vis_bin]['mean_gdet_'+data_type_gen+'_data_paths'][iexp])['mean_gdet'] 
            else:data_exp['mean_gdet'] = None
            if data_type_gen=='DI': 
                iexp_glob=iexp
                
                #Store Keplerian motion
                if ('RV_star_solCDM' in data_glob_new):data_to_bin[iexp_off]['RV_star_solCDM'] = coord_dic[inst][vis_bin]['RV_star_solCDM'][iexp_glob]
                
            else:
                                
                #Intrinsic profiles
                #    - beware that profiles were aligned if binning dimension is not phase
                if data_type=='Intr':
                    iexp_glob = gen_dic[inst][vis_bin]['idx_in2exp'][iexp]
               
                #Atmospheric profiles
                #    - beware that profiles were aligned if binning dimension is not phase
                elif data_type_gen=='Atm':
                    if data_type=='Absorption':
                        iexp_glob = gen_dic[inst][vis_bin]['idx_in2exp'][iexp] 
                        
                        #Planet-to-star radius ratio
                        #    - profile has not been aligned but is varying with low-frequency, and the shifts are thus not critical to the weighing
                        SpSstar_spec = data_exp['SpSstar_spec']
                        
                    elif data_type=='Emission':iexp_glob=iexp

                    #Estimate of local stellar profile for current exposure  
                    #    - defined on the same table as data_exp
                    if (data_type=='Absorption') or ((data_type=='Emission') and data_dic['Intr']['cov_loc_star']): 
                        data_est_loc=np.load(data_dic[inst][vis_bin]['LocEst_Atm_data_paths'][iexp]+'.npz',allow_pickle=True)['data'].item() 
             
            #Resampling on common spectral table if required
            #    - condition is True unless all exposures of 'vis_bin' are defined on a common table, and it is the reference for the binning             
            #    - if the resampling condition is not met, then all profiles have been resampled on the common table for the visit, and the master does not need resampling as:
            # + it is still on its original table, which is the common table for the visit
            # + it has been shifted, and resampled on the table of the associated profile, which is also the common table            
            #    
            #    - data is stored with the same indexes as in idx_to_bin_all
            #    - all exposures must be defined on the same spectral table before being binned
            #    - profiles are resampled if :
            # + profiles are defined on their individual tables
            # + several visits are used, profiles have already been resampled within the visit, but not all visits share a common table and visit of binned exposure is not the one used as reference to set the common table
            #    - telluric are set to 1 if unused
            #    - upon first calculation of the weighing DI master, no DI stellar spectrum is available and it is set to 1
            #      the master must in any case be calculated from stellar spectra aligned in the star rest frame, where stellar lines will not contribute to the weighing
            if masterDI:
                flux_ref_exp=np.ones(dim_exp_com,dtype=float)
                cov_ref_exp = None   
                dt_exp = 1.   
            else:
                data_ref = dataload_npz(data_dic[inst][vis_bin]['mast_'+data_type_gen+'_data_paths'][iexp])         
                dt_exp = coord_dic[inst][vis_bin]['t_dur'][iexp]
            if ((mode=='') and (not data_inst[vis_bin]['comm_sp_tab'])) or ((mode=='multivis') and (not data_inst['comm_sp_tab']) and (vis_bin!=data_inst['com_vis'])):

                #Resampling exposure profile
                data_to_bin[iexp_off]['flux']=np.zeros(dim_exp_com,dtype=float)*np.nan
                data_to_bin[iexp_off]['cov']=np.zeros(data_inst['nord'],dtype=object)
                if not masterDI:
                    flux_ref_exp=np.zeros(dim_exp_com,dtype=float)*np.nan
                    cov_ref_exp=np.zeros(data_inst['nord'],dtype=object)
                tell_exp=np.ones(dim_exp_com,dtype=float) if data_inst[vis_bin]['tell_sp'] else None
                mean_gdet_exp=np.ones(dim_exp_com,dtype=float) if gen_dic['cal_weight'] and data_inst[vis_bin]['mean_gdet'] else None                
                for iord in range(data_inst['nord']): 
                    data_to_bin[iexp_off]['flux'][iord],data_to_bin[iexp_off]['cov'][iord] = bind.resampling(data_com['edge_bins'][iord], data_exp['edge_bins'][iord], data_exp['flux'][iord] , cov = data_exp['cov'][iord], kind=gen_dic['resamp_mode'])                                                        
                    if not masterDI:
                        if data_type_gen=='DI':flux_ref_exp[iord] = bind.resampling(data_com['edge_bins'][iord], data_ref['edge_bins'][iord], data_ref['flux'][iord], kind=gen_dic['resamp_mode'])                                                                            
                        else:flux_ref_exp[iord],cov_ref_exp[iord] = bind.resampling(data_com['edge_bins'][iord], data_ref['edge_bins'][iord], data_ref['flux'][iord] , cov = data_ref['cov'][iord], kind=gen_dic['resamp_mode'])                                                        
                    if data_inst[vis_bin]['tell_sp']:tell_exp[iord] = bind.resampling(data_com['edge_bins'][iord], data_exp['edge_bins'][iord], data_exp['tell'][iord] , kind=gen_dic['resamp_mode']) 
                    if gen_dic['cal_weight'] and data_inst[vis_bin]['mean_gdet'] :mean_gdet_exp[iord] = bind.resampling(data_com['edge_bins'][iord], data_exp['edge_bins'][iord], data_exp['mean_gdet'][iord] , kind=gen_dic['resamp_mode'])                    
                data_to_bin[iexp_off]['cond_def'] = ~np.isnan(data_to_bin[iexp_off]['flux'])  
                
                #Resample local stellar profile estimate
                if (data_type_gen=='Atm'):
                    flux_est_loc_exp = np.zeros(dim_exp_com,dtype=float)
                    if data_dic['Intr']['cov_loc_star']:
                        cov_est_loc_exp = np.zeros(data_inst['nord'],dtype=object)
                        for iord in range(data_inst['nord']): 
                            flux_est_loc_exp[iord] ,cov_est_loc_exp[iord] = bind.resampling(data_com['edge_bins'][iord], data_exp['edge_bins'][iord], data_est_loc['flux'][iord] , cov = data_est_loc['cov'][iord], kind=gen_dic['resamp_mode'])                                                        
                    else:
                        cov_est_loc_exp = np.zeros([data_inst['nord'],1],dtype=float)
                        for iord in range(data_inst['nord']): 
                            flux_est_loc_exp[iord] = bind.resampling(data_com['edge_bins'][iord], data_exp['edge_bins'][iord], data_est_loc['flux'][iord] , kind=gen_dic['resamp_mode'])                                                        
                                        
            #No resampling required
            #    - if the resampling condition is not met, then all profiles have been resampled on the common table for the visit
            #    - the local stellar profile estimate does not need resampling as:
            # + it is on its original table, which is the table of the associated profile, which is also the common table  
            # + it has been shifted, and resampled on the table of the associated profile, which is also the common table   
            else:                
                if not masterDI:
                    flux_ref_exp = data_ref['flux']
                    cov_ref_exp = data_ref['cov']   
                for key in ['flux','cond_def','cov']:data_to_bin[iexp_off][key] = data_exp[key] 
                tell_exp=data_exp['tell']
                mean_gdet_exp=data_exp['mean_gdet'] 
                if (data_type_gen=='Atm'):
                    flux_est_loc_exp = data_est_loc['flux']
                    if data_dic['Intr']['cov_loc_star']:cov_est_loc_exp = data_est_loc['cov'] 
                    else:cov_est_loc_exp = np.zeros([data_inst['nord'],1],dtype=float)   

            #Exclude planet-contaminated bins  
            if (data_type_gen=='DI') and ('DI_Mast' in data_dic['Atm']['no_plrange']) and (iexp_glob in data_dic['Atm'][inst][vis_bin]['iexp_no_plrange']):
                for iord in range(data_inst['nord']):   
                    data_to_bin[iexp_off]['cond_def'][iord] &= excl_plrange(data_to_bin[iexp_off]['cond_def'][iord],data_dic['Atm'][inst][vis_bin]['exclu_range_star'],iexp_glob,data_com['edge_bins'][iord],data_mode)[0]
                 
            #Weight definition
            #    - the profiles must be specific to a given data type so that earlier types can still be called in the multi-visit binning, after the type of profile has evolved in a given visit
            #    - at this stage of the pipeline broadband flux scaling has been defined, if requested 
            data_to_bin[iexp_off]['weight'] = def_weights_spatiotemp_bin(range(data_inst['nord']),scaled_data_paths,inst,vis_bin,gen_dic['corr_Fbal'],gen_dic['corr_FbalOrd'],gen_dic['save_data_dir'],gen_dic['type'],data_inst['nord'],iexp_glob,data_type,data_mode,dim_exp_com,tell_exp,mean_gdet_exp,data_com['cen_bins'],dt_exp,flux_ref_exp,cov_ref_exp,flux_est_loc_exp=flux_est_loc_exp,cov_est_loc_exp = cov_est_loc_exp, SpSstar_spec = SpSstar_spec,bdband_flux_sc = gen_dic['flux_sc'])                          

        #----------------------------------------------------------------------------------------------

        #Processing and analyzing each new exposure 
        for i_new,(idx_to_bin,n_in_bin,dx_ov) in enumerate(zip(idx_to_bin_all,n_in_bin_all,dx_ov_all)):

            #Calculate binned exposure on common spectral table
            data_exp_new = calc_binned_prof(idx_to_bin,data_dic[inst]['nord'],dim_exp_com,nspec_com,data_to_bin,inst,n_in_bin,data_com['cen_bins'],data_com['edge_bins'],dx_ov_in = dx_ov)

            #Keplerian motion relative to the stellar CDM and the Sun (km/s)
            if ('RV_star_solCDM' in data_glob_new):
                RV_star_solCDM = 0.
                for isub,ibin in enumerate(idx_to_bin):RV_star_solCDM+=dx_ov[isub]*data_to_bin[ibin]['RV_star_solCDM']
                data_glob_new['RV_star_solCDM'][i_new] = RV_star_solCDM/np.sum(dx_ov)

            #Saving new exposure  
            if not masterDI:np.savez_compressed(save_pref+str(i_new),data=data_exp_new,allow_pickle=True)
           
           
            # # Stage Théo : Saving extra data for the module 'fit_ResProf'
            
            # if (gen_dic['fit_ResProf'] and masterDI) : 
            #     data_exp_new['idx_to_bin'] = idx_to_bin
            #     data_exp_new['weight'] = {}
            #     for iexp_off in idx_to_bin : data_exp_new['weight'][iexp_off] = data_to_bin[iexp_off]['weight']
            #     data_exp_new['dx_ov'] = dx_ov




        #Store path to weighing master 
        if masterDI:
            np.savez_compressed(data_dic[inst][vis_in]['mast_'+data_type+'_data_paths'][0],data=data_exp_new,allow_pickle=True) 

        #Store common table of binned profiles
        data_glob_new['cen_bins'] = data_com['cen_bins']
        data_glob_new['edge_bins'] = data_com['edge_bins']
 
        #---------------------------------------------------------------------------
        #Calculating associated properties 
        #    - calculation of theoretical properties of planet-occulted regions is only possible if data binned over phase
        #    - new coordinates are relative to the planet chosen as reference for the binned coordinates 
        #---------------------------------------------------------------------------
        data_glob_new.update({'st_bindim':new_x_low,'end_bindim':new_x_high,'cen_bindim':new_x_cen,'n_exp':n_bin,'dim_all':[n_bin]+dim_exp_com,'dim_exp':dim_exp_com,'nspec':nspec_com,'rest_frame' : rest_frame })
        if (prop_dic['dim_bin'] == 'phase'):  

            #Coordinates of planets for new exposures
            #    - phase is associated with the reference planet, and must be converted into phases of the other planets 
            ecl_all = np.zeros(data_glob_new['n_exp'],dtype=bool)
            data_glob_new['coord'] = {}
            for pl_loc in data_dic[inst][vis_save]['transit_pl']:
                pl_params_loc=system_param[pl_loc]
            
                #Phase conversion
                phase_tab = conv_phase(coord_dic,inst,vis_save,system_param,bin_prop['ref_pl'],pl_loc,np.vstack((new_x_low,new_x_cen,new_x_high)))              
            
                #Coordinates
                x_pos_pl,y_pos_pl,z_pos_pl,Dprojp,_,_,_,ecl_pl = calc_pl_coord(pl_params_loc['ecc'],pl_params_loc['omega_rad'],pl_params_loc['aRs'],pl_params_loc['inclin_rad'],phase_tab,data_dic['DI']['system_prop']['achrom'][pl_loc][0],pl_params_loc['lambda_rad'],system_param['star'])
                data_glob_new['coord'][pl_loc] = {
                    'ecl':ecl_pl,
                    'st_ph':phase_tab[0],'cen_ph':phase_tab[1],'end_ph':phase_tab[2],
                    'st_pos':np.vstack((x_pos_pl[0],y_pos_pl[0],z_pos_pl[0])),
                    'cen_pos':np.vstack((x_pos_pl[1],y_pos_pl[1],z_pos_pl[1])),
                    'end_pos':np.vstack((x_pos_pl[2],y_pos_pl[2],z_pos_pl[2]))}

                #Exposure considered out-of-transit if no planet at all is transiting
                ecl_all |= abs(ecl_pl)!=1            
  
                #Orbital rv of current planet in star rest frame
                if data_dic['Atm']['exc_plrange']:
                    data_glob_new['coord'][pl_loc]['rv_pl'] = np.zeros(len(phase_tab[1]))*np.nan
                    dphases = phase_tab[2] - phase_tab[0]
                    for isub,dph_loc in enumerate(dphases):
                        nt_osamp_RV=max(int(dph_loc/data_dic['Atm']['dph_osamp_RVpl']),2)
                        dph_osamp_loc=dph_loc/nt_osamp_RV
                        ph_osamp_loc = phase_tab[0]+dph_osamp_loc*(0.5+np.arange(nt_osamp_RV))
                        data_glob_new['coord'][pl_loc]['rv_pl'][isub]=system_param['star']['RV_conv_fact']*np.mean(calc_pl_coord(pl_params_loc['ecc'],pl_params_loc['omega_rad'],pl_params_loc['aRs'],pl_params_loc['inclin_rad'],ph_osamp_loc,None,None,None,rv_LOS=True,omega_p=pl_params_loc['omega_p'])[6])

            #In-transit data
            data_glob_new['idx_in']=np_where1D(ecl_all)
            data_glob_new['idx_out']=np_where1D(~ecl_all)
            data_glob_new['n_in_tr'] = len(data_glob_new['idx_in'])
            data_glob_new['idx_exp2in'] = np.zeros(data_glob_new['n_exp'],dtype=int)-1
            data_glob_new['idx_exp2in'][data_glob_new['idx_in']]=np.arange(data_glob_new['n_in_tr'])
            data_glob_new['idx_in2exp'] = np.arange(data_glob_new['n_exp'],dtype=int)[data_glob_new['idx_in']]
            data_glob_new['dim_in'] = [data_glob_new['n_in_tr']]+data_glob_new['dim_exp']

            #Properties of planet occulted-regions
            params = deepcopy(system_param['star'])
            params.update({'rv':0.,'cont':1.})  
            par_list=['rv','CB_RV','mu','lat','lon','x_st','y_st','SpSstar','xp_abs','r_proj']
            key_chrom = ['achrom']
            if ('spec' in data_mode) and ('chrom' in data_dic['DI']['system_prop']):key_chrom+=['chrom']
            data_glob_new['plocc_prop'] = sub_calc_plocc_prop(key_chrom,{},par_list,data_inst[vis_save]['transit_pl'],system_param,theo_dic,data_dic['DI']['system_prop'],params,data_glob_new['coord'],range(n_bin),False,out_ranges=True)            

        #---------------------------------------------------------------------------

        #Saving global tables for new exposures
        if not masterDI:
            if (data_type=='DI'):data_glob_new['sysvel'] = sysvel
        np.savez_compressed(save_pref+'_add',data=data_glob_new,allow_pickle=True)

    #Checking that data were calculated 
    #    - check is performed on the complementary table
    else:
        if masterDI: check_data({'path':data_dic[inst][vis_in]['mast_DI_data_paths'][0]})
        else:check_data({'path':save_pref+'_add'})

    return None








'''
Function to define weights when binning profiles temporally/spatially
    - weights should only be defined using the inverse squared error if the weighted values are comparable, so that all spectra should have been scaled to comparable flux levels prior to binning
    - spectra must be defined on the same spectral table to be averaged in time/space     
'''
def def_weights_spatiotemp_bin(iord_orig_list,scaled_data_paths,inst,vis,gen_corr_Fbal,gen_corr_Fbal_ord,save_data_dir,gen_type,nord,iexp_glob,data_type,data_mode,dim_exp,tell_exp,mean_gdet,cen_bins,dt,flux_ref_exp,cov_ref_exp,flux_est_loc_exp=None,cov_est_loc_exp=None,SpSstar_spec=None,bdband_flux_sc=False,glob_flux_sc=None,corr_Fbal = True):

    #Weight definition
    #    - in case of photon noise the error on photons counts writes as E(n) = sqrt(N) where N is the number of photons received during a time interval dt over a spectral bin dw
    # + if the same measurement is manipulated, eg with N' = a*N1, then the rules of error propagation apply and E(N') = a*E(N1)
    # + if two independent measurements are made, eg related as N2 = a*N1, then E(N2) = sqrt(a*N1) = sqrt(a)*sqrt(N1) = sqrt(a)*E(N1) 
    #      if measurements are provided as a density, eg over time with n = N/dt
    # + if the same measurement is manipulated, eg with n = N/dt, then the rules of error propagation apply and E(n) = E(N)/dt
    # + if two independent measurements are made, eg with dt2 = a*dt1, then N2 = a*N1 and n2 = a*N1/(a*dt1) = n1 but E[n2] = E[N2/dt2] = E[N2]/dt2 = sqrt(N2)/dt2 = sqrt(a*N1)/(a*dt1) = E[N1]/sqrt(a) 
    #    - standard pipelines return spectral flux density, ie the number of photoelectrons measured over an entire exposure per unit of wavelength (but not per unit of time):
    # + F_meas(t,w) = gcal(band) N_meas(t,w)  
    # + EF_meas(t,w) = E[ F_meas(t,w) ]
    #                = gcal(band) sqrt( N_meas(t,w) )               assuming no other noise sources
    #                = gcal(band) sqrt( F_meas(t,w)/gcal(band) )
    #                = sqrt( gcal(band)*F_meas(t,w))       
    #      where gcal represents corrections of raw measured photoelectrons for the instrumental response (typically the blaze function, ie the transmission of the grating, with gcal = 1/blaze), assumed constant for an instrument (see below) 
    #      if fluxes were converted into temporal densities:
    # + f_meas(t,w) = F_meas(t,w)/dt = gcal(band) N_meas(t,w) / dt
    # + Ef_meas(t,w) = EF_meas(t,w)/dt = gcal(band) sqrt( N_meas(t,w) ) /dt =  sqrt( gcal(band) f_meas(t,w) / dt) 
    #    - our estimate of the true measured flux is F_meas(t,w) = g(band) N_meas(t,w), where N_meas(t,w) follows a Poisson distribution with number of events N_true(t,w)
    #      if enough photons are measured the Poisson distribution on N_meas(t,w) and thus F_meas(t,w) can be approximated by a normal distribution with mean F_true(t,w) and standard deviation EF_true(t,w) 
    #      the issue with using the errors on individual bins to define their weights is that EF_meas(t,w) = sqrt(g(band) F_meas(t,w)) is a biased estimate of the true error
    #      for a bin where F_meas(t,w) < F_true(w) because of statistical variations, we indeed get EF_meas(t,w) < EF_true(w) (and the reverse where F_meas(t,w) > F_true(w)) 
    #      bins with lower flux values than the true flux will have larger weights, resulting in a weighted mean that is underestimated compared to the true flux (mathematically speaking, we do a harmonic mean rather than an arithmetic mean).    
    #      the average flux measured over a large band (such as an order) is closer to the true flux over the band than in the case of a single bin, and thus its error is a better estimate of the true error, so we can assume:
    # EF_meas(t,band) ~ EF_true(t,band)
    #    - for all cases below, bands are defined as orders for e2ds, and as the full spectral range for CCFs and s1d   
    #    - assuming that the flux spectra provided by pipelines are well corrected for this response, we do not want to introduce biases between exposures by using specific calibration files
    #      a given calibration file (eg the blaze) may indeed not include all possible corrections for the instrumental response, and furthermore the flux calibration is not necessarily available for all instruments
    #      we thus use the input flux and error tables to estimate gcal(band,t,v) and derive a profile common to all exposures processed for a given instrument (the goal being to approximate the instrumental response without introducing biases)
    #      at the stage of the workflow where calibration is estimated, fluxes are still in number of photoelectrons, so that:
    #      the flux in a given pixel verifies:
    # F_true(w,t,v) = gcal(w,t,v)*N_true(w,t,v)
    # EF_true(w,t,v) = gcal(w,t,v)*sqrt(N_true(w,t,v))
    #      summing the fluxes over a large band yields 
    # TF_meas(t,v) ~ TF_true(t,v) = sum(w, F_true(w,t,v) ) = gcal(band,t,v)*sum(w over band, N_true(w,t,v) )
    #      thus
    # ETF_meas(t,v) ~ E[ TF_true(t,v) ]
    #               = E[ gcal(band,t,v)*sum(w over band, N_true(w,t,v) ) ]
    #               = gcal(band,t,v)*E[ sum(w over band, N_true(w,t,v) ) ]
    #               = gcal(band,t,v)*sqrt( sum(w over band, E[ N_true(w,t,v) )]^2 )
    #               = gcal(band,t,v)*sqrt( sum(w over band, N_true(w,t,v) ) ) 
    #      and thus 
    # gcal(band,t,v) = TF_meas(t,v)*(dt*dw)/sum(w over band, N_true(w,t,v) )
    # gcal(band,t,v) = TF_meas(t,v)*gcal(band,t,v)^2/ETF_meas(t,v)^2
    # gcal(band,t,v) = ETF_meas(t,v)^2 / TF_meas(t,v)
    #      we estimate gcal over each order for S2D and the full spectral range for S1D
    #      summing the flux densities over a large band yields 
    # TF_meas(t,v) ~ TF_true(t,v) = sum(w, F_true(w,t,v) ) = gcal(band,t,v)*sum(w over band, N_true(w,t,v) ) 
    #      thus
    # ETF_meas(t,v) ~ E[ TF_true(t,v) ]
    #               = E[ gcal(band,t,v)*sum(w over band, N_true(w,t,v) ) ]
    #               = gcal(band,t,v)*E[ sum(w over band, N_true(w,t,v) ) ]
    #               = gcal(band,t,v)*sqrt( sum(w over band, E[ N_true(w,t,v) )]^2 )
    #               = gcal(band,t,v)*sqrt( sum(w over band, N_true(w,t,v) ) )  
    #      and thus 
    # gcal(band,t,v) = TF_meas(t,v)/sum(w over band, N_true(w,t,v) )
    # gcal(band,t,v) = TF_meas(t,v)*gcal(band,t,v)^2/ETF_meas(t,v)^2
    # gcal(band,t,v) = ETF_meas(t,v)^2 / TF_meas(t,v)
    #      we estimate gcal over each order and all exposures, and define gcal(band) as < t, v :  gcal(band,t,v) > 
    #    - what matters for the weighing is the change in the precision on the flux over time in a given pixel:
    # + low-frequency variations linked to the overall flux level of the data (eg due to atmospheric diffusion)
    # + high-frequency variations linked to variations in the spectral flux distribution at the specific wavelength of the pixel
    #   for example when averaging the same spectra in their own rest frame, spectral features do not change and there is no need to weigh
    #   however if there are additional features, such as telluric absorption or stellar lines in transmission spectra, then a given pixel can see large flux variations and weighing is required.  
    #    - extreme care must be taken about the rest frame in which the different elements involved in the weighing profiles are defined
    # + low-frequency components are assumed constant over an order and are not aligned / resampled (eg instrumental calibration, global flux scaling, flux balance and light curve scaling)
    # + high-frequency components must have followed the same shifts and resampling as the weighed profile (eg telluric or master stellar spectrum) 
    #    - weights are normalized within the binning function, so that any factor constant over time will not contribute to the weighing
    #    - planetary signatures are not accounted for when binning disk-integrated and intrinsic stellar spectra because they have been excluded from the data or are considered negligible
    #    - weights are necessarily spectral for intrinsic and atmospheric spectra because of the master 
    #    - the weighing due to integration time is accounted for in the binning routine directly, not through errors but through the fraction of an exposure duration that overlaps with the new bin time window 
    weight_spec = np.zeros(dim_exp,dtype=float)*np.nan
    
    #--------------------------------------------------------    
    #Definition of errors on disk-integrated spectra
    #--------------------------------------------------------     
    #    - we calculate the mean of a time-series of disk-integrated spectra while accounting for variations in the noise of the pixels between different exposures
    #    - at the latest processing stage those spectra are defined from rescale_data() as:
    # Fsc(w,t,v) = LC_theo(band,t,v)*Fcorr(w,t,v)/(dt*globF(t,v))    
    #      with the corrected spectra linked to the measured spectra as (see spec_corr() function above for the corrections that are included):
    # Fcorr(w,t,v) = F_meas(w,t,v)*Ccorr(w,t,v)  
    #      the measured spectrum is the one read from the input files of the instruments DRS as:
    # F_meas(w,t,v) = gcal(band)*N_meas(w,t,v)
    #      where gcal(band) represents corrections of measured spectral density of photoelectron N_meas for instrumental response (flat field, blaze, etc) assumed to have low-frequency variations
    #            for the purpose of weighing and to avoid biases between the estimated N_meas we assume a constant conversion for all exposures
    #      thus
    # Fsc(w,t,v) = LC_theo(band,t,v)*gcal(band)*N_meas(w,t,v)*Ccorr(w,t,v)/(dt*globF(t,v)) 
    #      or in terms of photoelectons cumulated over the exposure
    # N_sc(w,t,v) = LC_theo(band,t,v)*N_meas(w,t,v)*Ccorr(w,t,v)/globF(t,v) 
    #    - as explained above we want to use as weights the errors on the true measured flux, ie : 
    # EFsc_true(w,t,v) = E( LC_theo(band,t,v)*gcal(band)*N_true(w,t,v)*Ccorr(w,t,v)/(dt*globF(t,v)) )
    #                  = LC_theo(band,t,v)*gcal(band)*Ccorr(w,t,v)*E( N_true(w,t,v) )/(dt*globF(t,v))
    #                  = LC_theo(band,t,v)*gcal(band)*Ccorr(w,t,v)*sqrt( N_true(w,t,v) )/(dt*globF(t,v))
    #      the spectra Fsc_true(w,t,v)/LC_theo(band,t,v) have the same flux density profile as the master of the unocculted star MFstar_true(w,v), so that 
    # gcal(band)*N_true(w,t,v)*Ccorr(w,t,v)/(dt*globF(t,v)) ~ MFstar_true(w,v)
    # N_true(w,t,v) = MFstar_true(w,v)*(dt*globF(t,v))/(gcal(band)*Ccorr(w,t,v))
    #      since the master spectrum of the unocculted star is measured with a high SNR we can assume: 
    # MFstar_true(w,v) ~ MFstar_meas(w,v)
    #      thus
    # sqrt(N_true(w,t,v)) = sqrt( MFstar_meas(w,v)*dt*globF(t,v))/(gcal(band)*Ccorr(w,t,v)) )    
    #      and
    # EFsc_true(w,t,v) = LC_theo(band,t,v)*gcal(band)*Ccorr(w,t,v)*sqrt( MFstar_meas(w,v)*dt*globF(t,v)/(gcal(band)*Ccorr(w,t,v)) )/(dt*globF(t,v))    
    #                  = LC_theo(band,t,v)*sqrt(gcal(band)*Ccorr(w,t,v)* MFstar_meas(w,v)/(dt*globF(t,v)) )    
    #    - final weights are defined as:    
    # weight(w,t,v) = 1/EFsc_true(w,t,v)^2  
    #               = dt*globF(t,v)/(LC_theo(band,t,v)^2*gcal(band)*Ccorr(w,t,v)*MFstar_meas(w,v))      
    #      we note that the spectral normalization of counts by dw is ignored since all profiles are defined over the same spectra table
    #    - in the star rest frame, spectral features from the disk-integrated spectrum remain fixed over time , ie that a given pixel sees no variations in flux from MFstar(w,v) in a given visit and the contribution of the master spectrum can be ignored    
    #      we note that we neglected the differences between the Fsc_true and MFstar profiles that are due to the occulted local stellar lines and planetary atmospheric lines
    #    - even if the calibration remains constant in time and does not contribute to the weighing of disk-integrated spectra, it must always be set to a value consistent with error definitions in the pipeline,
    # so that EFsc_true is comparable with other error tables in the weighing of intrinsic and planetary profiles
    #      calibration profiles are thus always estimated by default by the pipeline, either from the input error tables when available or from the imposed error table for consistency    
    #--------------------------------------------------------     

    #Calculate weights at pixels where the master stellar spectrum is defined and positive
    cond_def_weights = (~np.isnan(flux_ref_exp)) & (flux_ref_exp>0.)
    if np.sum(cond_def_weights)==0:stop('Issue with master definition')

    #Spectral corrections
    #    - all corrections that are applied between the input spectra and the spectra processed per visit should be included in this function:
    # > instrumental calibration: uploaded from 'gcal_inputs' for every type of data
    #   for original 2D or 1D spectra, it returns the estimated spectral calibration profiles for each exposure 
    #   for original CCFs or after conversion into CCFs or from 2D/1D it returns a global calibration
    #   if original 2D or 1D spectra were converted back into count-equivalent values and are still in their original format, calibration profiles are rescaled by the mean calibration profile over the visit
    # > tellurics: Ccorr(w,t,v) = 1/T(w,t,v)
    #   with T=1 if not telluric absorption, 0 if maximum absorption
    #   telluric profiles for each exposure are contained in the data upload specific to the exposure, and have been aligned and set to the same rest frame  
    #   they are propagated through 2D/1D conversion but not CCF conversion
    # > flux balance: Ccorr(w,t,v) = ( 1/Pbal(band,t,v) )*( 1/Pbal_ord(band,t,v))
    #   where Pbal and Pbal_ord are the low-frequency polynomial corrections of flux balance variations over the full spectrum or per order
    #   errors on Pbal are neglected and it can be considered a true estimate because it is fitted over a large number of pixels 
    #   the corrections were defined over the spectral in their input rest frame. Given the size of the bands, we neglect spectral shifts and assume the correction can be directly used over any table   
    #   note that SNR(t,order) = N(t,order)/sqrt(N(t,order)) = sqrt(N(t,order))    
    #   since Pbal(band,t,v) is an estimate of F(band,t,v)/MFstar(band,v) times a normalization factor, it is proportional to N(band,t,v) (since the reference is time-independent)
    #   thus Pbal(order,t,v) is proportional to SNR(order,t,v)^2
    #   global normalisation coefficient is not spectral and can be directly applied to any data
    #   global flux balance correction is defined as a function over the full instrument range and can be directly applied to any spectral data, but since it does not change the mean flux it is not propagated through CCF conversion
    #   order flux balance correction is not propagated through 2D/1D conversion or CCF conversion 
    # > cosmics and permanent peaks: ignored in the weighing, as flux values are not scaled but replaced
    # > fringing and wiggles: ignored for now
    if gen_corr_Fbal and ('spec' in data_mode) and corr_Fbal: 
        data_Fbal = dataload_npz(save_data_dir+'Corr_data/Fbal/'+inst+'_'+vis+'_'+str(iexp_glob)+'_add')
        corr_func_glob = data_Fbal['corr_func']
        corr_func_vis = data_Fbal['corr_func_vis']
    else:
        data_Fbal = None
        corr_func_glob=default_func 
        corr_func_vis = None
    if corr_func_vis is not None:corr_func_glob_vis = corr_func_vis
    else:corr_func_glob_vis=default_func 
    if gen_corr_Fbal_ord and ('spec' in data_mode) and (data_mode==gen_type[inst]):
        if data_Fbal is None:data_Fbal = dataload_npz(save_data_dir+'Corr_data/Fbal/'+inst+'_'+vis+'_'+str(iexp_glob)+'_add')
        corr_func_ord = data_Fbal['Ord']['corr_func']
    else:corr_func_ord={iord_orig:default_func for iord_orig in iord_orig_list}
    if (mean_gdet is None):mean_gdet = np.ones(dim_exp,dtype=float)
    if (tell_exp is None):tell_exp = np.ones(dim_exp,dtype=float)
    spec_corr={}
    for iord,iord_orig in enumerate(iord_orig_list):
        if data_mode=='CCF':cen_bins_ord = np.array([1.])
        else:cen_bins_ord = cen_bins[iord,cond_def_weights[iord]]
        nu_bins_ord = c_light/cen_bins_ord[::-1]
        corr_Fbal_glob_ord = (corr_func_glob(nu_bins_ord)*corr_func_glob_vis(nu_bins_ord))[::-1]
        spec_corr[iord] = mean_gdet[iord,cond_def_weights[iord]]/(tell_exp[iord,cond_def_weights[iord]]*corr_func_ord[iord_orig](cen_bins_ord)*corr_Fbal_glob_ord)
     
    #Spectral broadband flux scaling 
    #    - includes broadband contribution, unless overwritten by input 
    #    - flux_sc = 1 with no occultation, 0 with full occultation
    if bdband_flux_sc:     
        data_scaling = dataload_npz(scaled_data_paths+str(iexp_glob))        
        flux_sc = {iord:1. - data_scaling['loc_flux_scaling'](cen_bins[iord,cond_def_weights[iord]]) for iord in range(nord)}
        if (glob_flux_sc is None):glob_flux_sc = data_scaling['glob_flux_scaling']        
    
    #Global flux scaling can still be applied, if provided as input
    else:
        flux_sc = {iord:1. for iord in range(nord)}     
        if (glob_flux_sc is None):glob_flux_sc = 1.

    #Sub-function defining the squared error on scaled disk-integrated profiles
    def calc_EFsc2(flux_sc_ord,flux_ref_ord,spec_corr_ord):
        return (flux_sc_ord**2.)*flux_ref_ord*spec_corr_ord/(dt*glob_flux_sc)

    #Weights on disk-integrated spectra
    if data_type=='DI': 
        for iord in range(nord):
            weight_spec[iord,cond_def_weights[iord]] = 1./calc_EFsc2(flux_sc[iord],flux_ref_exp[iord,cond_def_weights[iord]],spec_corr[iord])  

    else:

        #Function defining the squared error on local stellar profiles
        def calc_EFres2(flux_ref_ord,err_ref_ord2,flux_sc_ord,spec_corr_ord):
            return err_ref_ord2 + calc_EFsc2(flux_sc_ord,flux_ref_ord,spec_corr_ord)

        #--------------------------------------------------------    
        #Definition of errors on residual spectra
        #-------------------------------------------------------- 
        #    - see rescale_data(), extract_res_profiles(), the profiles are defined as:
        # Fres(w,t,v) = ( MFstar(w,v) - Fsc(w,t,v) )
        #      where profiles have been scaled to comparable levels and can be seen as (temporal) flux densities
        #    - we want to use as weights the errors on the true residual flux:
        # EFres_true(w,t,v) = E[  MFstar_true(w,v) - Fsc_true(w,t,v) ]
        #                   = sqrt( EMFstar_true(w,v)^2 + EFsc_true(w,t,v)^2 )  
        #      where we assume that the two profiles are independent, and that the error on the master flux averaged over several exposures approximates well the error on the true flux, even within a single bin, so that:          
        # EFres_true(w,t,v) = sqrt( EMFstar_meas(w,v)^2 + EFsc_true(w,t,v)^2 )  
        #    - final weights are defined as:
        # weight(w,t,v) = 1/EFres_true(w,t,v)^2
        #    - we neglect covariance in the uncertainties of the master spectrum
        #    - the binning should be performed in the star rest frame 
        if data_type=='Res': 
            for iord in range(nord):
                weight_spec[iord,cond_def_weights[iord]] = 1. / calc_EFres2(flux_ref_exp[iord,cond_def_weights[iord]],cov_ref_exp[iord][0,cond_def_weights[iord]],flux_sc[iord],spec_corr[iord])

        #--------------------------------------------------------    
        #Definition of errors on intrinsic spectra
        #-------------------------------------------------------- 
        #    - see rescale_data(), extract_res_profiles() and proc_intr_data(), the profiles are defined as:
        # Fintr(w,t,v) = Fres(w,t,v)/(1 - LC_theo(band,t))
        #              = ( MFstar(w,v) - Fsc(w,t,v) )/(1 - LC_theo(band,t))            
        #    - we want to use as weights the errors on the true intrinsic flux, ie : 
        # EFintr_true(w,t,v) = EFres_true(w,t,v)/(1 - LC_theo(band,t)) 
        #      we assume that the error on the master flux averaged over several exposures approximates well the error on the true flux, even within a single bin, so that:          
        # EFres_true(w,t,v) = sqrt( EMFstar_meas(w,t,v)^2 + EFsc_true(w,t,v)^2 )  
        #    - final weights are defined as:
        # weight(w,t,v) = 1/EFintr_true(w,t,v)^2             
        #               = (1 - LC_theo(band,t))^2 / EFres_true(w,t,v)^2
        #    - we neglect covariance in the uncertainties of the master spectrum
        #    - intrinsic spectra are extracted in the star rest frame, in which case (Fstar_meas,EFstar_meas) and T must also be in the star rest frame (T must thus have been shifted in the same way as Fsc) 
        #      the binning can be performed in the star rest frame or in the rest frame of the intrinsic stellar lines (ie, in which they are aligned) 
        #      in the latter case, if intrinsic spectra given as input have been shifted (aligned to their local frame or to a given surface rv), then the above components must have been shifted in the same way
        #      for example if the intrinsic line is centered at -rv, then the stellar spectrum is redshifted by +rv when aligning the intrinsic line, so that its blue wing at -rv contributes to the weight at the center of the intrinsic line
        #      the spectral features of the local stellar lines then remain aligned in Fsc over time, but the global stellar line in which they are imprinted shifts, so that a given pixel sees important flux variations
        # weight(w_shifted,t,v) = (1 - LC_theo(band,t))^2 / ( EMFstar_meas(w_shifted,v)^2 + EFsc_true(w_shifted,t,v)^2 )     
        #                       ~ (1 - LC_theo(band,t))^2 / EFsc_true(w_shifted,t,v)^2           
        #                       ~ (1 - LC_theo(band,t))^2 /(LC_theo(band,t)^2*gdet(band,v)*Ccorr(w_shifted,t,v)*MFstar(w_shifted,v)/globF(t,v))  
        #      the weight is inversely proportional to MFstar(w_shifted,v), so that weights in intrinsic line are lower when it is located at high rv in the disk-integrated line, compared to being located at its center
        #      while it may appear counter-intuitive, this is because the disk-integrated stellar line acts only as a background flux (and thus noise) level, so that it brings less noise in its deep core than its wings
        #      another way to see it is as:
        # Fintr_estimated(w,t,v) = (MFstar(w,v) - F(w,t,v))/(1 - LC_theo(band,t))
        #                        = (MFstar(w,v) - (MFstar(w,t,v) - Fintr_meas(w,t,v)))/(1 - LC_theo(band,t))
        #                        = (dFstar(w,t,v) + Fintr_meas(w,t,v)))/(1 - LC_theo(band,t))
        #      if MFstar(w,v) is a good estimate of MFstar(w,t,v) we will retrieve the correct intrinsic profile, but its uncertainties will be affected by the errors on MFstar(w,t,v) - which will dominate the weighing when binning the retrieved 
        # intrinsic profiles in their local rest frame
        elif data_type=='Intr': 
            for iord in range(nord):
                weight_spec[iord,cond_def_weights[iord]] =(1. - flux_sc[iord])**2. / calc_EFres2(flux_ref_exp[iord,cond_def_weights[iord]],cov_ref_exp[iord][0,cond_def_weights[iord]],flux_sc[iord],spec_corr[iord])

        #--------------------------------------------------------        
        #Atmospheric spectra
        #    - in the definition of the weights below we implicitely assume that Fnorm_true(t,w) has the same spectral profile as Fstar_true(t,w)
        #      this is not the case because of the presence of the planetary signature in Fnorm_true(t,w)
        #      however, in the binned atmospheric spectra the planetary signature keeps the same shape and position over time (this is the necessary assumption when binning)
        #      thus differences in errors and weights in a given bin will not come from the planetary signature, but from low-frequency flux variations and from the stellar and telluric lines varying in position
        #      we can thus neglect the presence of the planetary signature
        #--------------------------------------------------------        
                
        #For emission spectra:
        #    - see extract_pl_profiles(), the profiles are defined as:
        # Fem(t,w in band) = Fstar_loc(w,t) - Fres(t,w in band)            
        #      with Fstar_loc null in out-of-transit exposures
        #    - we want to weigh using the true error on the emission flux:    
        # EFem_true(t,w in band)^2 = EFstar_loc_true(w,t)^2 + EFres_true(t,w in band)^2           
        #      EFres_true is defined as above 
        #      local stellar profiles Fstar_loc_true can be defined in different ways (see def_plocc_profiles()). 
        #      we neglect their errors when models are used (mode_loc_data_corr = 'glob_mod', 'indiv_mod', 'rec_prof') 
        #      otherwise they are defined using the disk-integrated or intrinsic profiles (binned in loc_prof_meas() )
        #      here we assume that enough exposures are used to create these masters that their measured error approximate well enough the true value
        #      EFstar_loc_true(w,t) ~ EFstar_loc_meas(w,t)
        #      final weights are defined as:
        # weight(t,w in band) = 1/( EFstar_loc_meas(w,t)^2 + EFres_true(t,w in band)^2 ) 
        #    - emission spectra are extracted in the star rest frame, in which case EFstar_loc_meas, (Fstar_meas,EFstar_meas) and T involved in Fres must also be in the star rest frame (T must thus have been shifted in the same way as Fnorm)
        #      if emission spectra given as input have been shifted (aligned to their local frame), then the above components  must have been shifted in the same way     
        elif data_type=='Emission': 
            for iord in range(nord):
                weight_spec[iord,cond_def_weights[iord]] = 1./( cov_est_loc_exp[iord][0,cond_def_weights[iord]] + calc_EFres2(flux_ref_exp[iord,cond_def_weights[iord]],cov_ref_exp[iord][0,cond_def_weights[iord]],flux_sc[iord],spec_corr[iord]) ) 
            
        #For absorption spectra: 
        #    - see extract_pl_profiles(), the profiles are defined as:                   
        # Abs(t,w in band) = [ Fres(w,t,vis)/Fstar_loc(w,t,vis)  - 1 ]*( Sthick(band,t)/Sstar )
        #    - we want to weigh using the true error on the absorption signal: 
        # EAbs_true(t,w in band)^2 = E[ Fres_true(t,w)/Fstar_loc_true(t,w) ]^2*( Sthick(band,t)/Sstar )^2
        #                          = ( (Fstar_loc_true(t,w)*EFres_true(t,w))^2 + (Fres_true(t,w)*EFstar_loc_true(t,w))^2 )*( Sthick(band,t)/Sstar )^2
        #      we calculate errors as detailed above for the emission spectra
        # Fres_true(t,w) = Fstar_true(w in band) - LC_theo(t,band)*Fnorm_true(t,w in band)
        #                = Fstar_true(w in band)*(1 - LC_theo(t,band))
        #                ~ Fstar_meas(w in band)*(1 - LC_theo(t,band))       
        #      final weights are defined as:
        # weight(t,w in band) = 1/EAbs_true(t,w in band)^2 
        elif data_type=='Absorption': 
            for iord in range(nord): 
                Floc2_ord = (flux_ref_exp[iord,cond_def_weights[iord]]*(1. - flux_sc[iord]))**2.
                EFloc2_ord = calc_EFres2(flux_ref_exp[iord,cond_def_weights[iord]],cov_ref_exp[iord][0,cond_def_weights[iord]],flux_sc[iord],spec_corr[iord])                
                weight_spec[iord,cond_def_weights[iord]] =  1./( ( flux_est_loc_exp[iord,cond_def_weights[iord]]**2.*EFloc2_ord + Floc2_ord*cov_est_loc_exp[iord][0,cond_def_weights[iord]] )*SpSstar_spec[iord,cond_def_weights[iord]]**2. )               

    return weight_spec





'''
Function to calculate generic stellar continuum from binned master spectrum
'''
def process_spectral_cont(vis_mode,data_type_gen,inst,data_dic,gen_dic,vis):
    print('   > Defining stellar continuum on '+gen_dic['type_name'][data_type_gen]+' master') 
    data_inst = data_dic[inst]   

    #Using master from several visits
    if vis_mode=='multivis':
        vis_det='binned'
        if data_inst['type']!='spec1D':stop('Spectra must be 1D') 

    #Using master from single visit
    elif vis_mode=='':
        vis_det=vis   
        if data_inst[vis]['type']!='spec1D':stop('Spectra must be 1D')    

    #Processing data    
    save_data_paths = gen_dic['save_data_dir']+'Stellar_cont_'+data_type_gen+'/'+inst+'_'+vis_det+'/St_cont'
    if not os_system.path.exists(save_data_paths):os_system.makedirs(save_data_paths)  
    if (gen_dic['calc_'+data_type_gen+'_stcont']):
        print('         Calculating data')
        prop_dic = deepcopy(data_dic[data_type_gen]) 
    
        #Retrieve binning information
        data_bin = dataload_npz(gen_dic['save_data_dir']+data_type_gen+'bin_data/'+gen_dic['add_txt_path'][data_type_gen]+inst+'_'+vis_det+'_'+prop_dic['dim_bin']+'_add')
    
        #Retrieve master spectrum
        if data_bin['n_exp']>1:stop('Bin data into a single master spectrum')
        data_mast = dataload_npz(gen_dic['save_data_dir']+data_type_gen+'bin_data/'+gen_dic['add_txt_path'][data_type_gen]+inst+'_'+vis_det+'_'+prop_dic['dim_bin']+str(0))

        #Check for alignment
        if (not gen_dic['align_'+data_type_gen]) or ((data_type_gen=='DI') and (not gen_dic['align_DI'] and data_bin['sysvel']==0.)):
            stop('Data must have been aligned in the stellar rest frame')
            
        #Check for in-transit contamination
        if (data_type_gen=='DI') and data_bin['in_inbin']:
            stop('Disk-integrated master contain in-transit profiles')    
    
        #Limit master to minimum definition range
        idx_def_mast = np_where1D(data_mast['cond_def'][0])
        flux_mast = data_mast['flux'][:,idx_def_mast[0]:idx_def_mast[-1]+1]
        cen_bins_mast = data_mast['cen_bins'][:,idx_def_mast[0]:idx_def_mast[-1]+1]
        cond_def_mast = data_mast['cond_def'][:,idx_def_mast[0]:idx_def_mast[-1]+1]
        edge_bins_mast = data_mast['edge_bins'][:,idx_def_mast[0]:idx_def_mast[-1]+2]
        nspec = len(flux_mast[0])
        
        #Stellar continuum
        min_edge_ord = cen_bins_mast[0,0]
        dic_sav = {}
        _,cont_func_dic,_ = calc_spectral_cont(1,[0],flux_mast,cen_bins_mast,edge_bins_mast,cond_def_mast,None,None,inst,gen_dic['contin_roll_win'][inst],gen_dic['contin_smooth_win'][inst],gen_dic['contin_locmax_win'][inst],\
                                                         gen_dic['contin_stretch'][inst],gen_dic['contin_pinR'][inst],min_edge_ord,dic_sav,1)
            
        #Saving data     
        datasave_npz(save_data_paths,{'cont_func_dic': cont_func_dic[0]})

    else:
        data_paths={'path':save_data_paths}
        check_data(data_paths)          
        
    return None




'''
Function to generate CCF binary masks from processed stellar spectrum 
    - 2D spectra must have been aligned in the star (for disk-integrated profiles), common (for intrinsic profiles), or planet (for atmospheric profiles) rest frame, converted into 1D profiles, and binned into a master spectrum
    - disk-integrated masks are shifted from the star rest frame to the rest frame of the input data using the systemic velocity associated with the master spectrum
      they can then be used to generate CCFs from disk-integrated spectra in their input rest frame
    - intrinsic masks are left defined in the common rest frame
      they can be used to generate CCFs from intrinsic spectra in the star rest frame
    - atmospheric masks are left defined in the common rest frame
      they can be used to generate CCFs from atmospheric spectra in the planet rest frame
'''
def def_masks(vis_mode,gen_dic,data_type_gen,inst,vis,data_dic,plot_dic,system_param,data_prop):
    data_inst = data_dic[inst]
    print('   > Defining CCF mask for '+gen_dic['type_name'][data_type_gen]+' profiles') 
    if data_inst['type']!='spec1D':stop('Spectra must be 1D')
    
    #Using master from several visits
    if vis_mode=='multivis':vis_det='binned'

    #Using master from single visit
    elif vis_mode=='':vis_det=vis    
    
    print('         Calculating data')
    save_data_paths = gen_dic['save_data_dir']+'CCF_masks_'+data_type_gen+'/'+gen_dic['add_txt_path'][data_type_gen]+'/'+inst+'_'+vis_det+'/'
    if not os_system.path.exists(save_data_paths):os_system.makedirs(save_data_paths)  
    prop_dic = deepcopy(data_dic[data_type_gen]) 
    mask_dic = prop_dic['mask']

    #Retrieve binning information
    data_bin = dataload_npz(gen_dic['save_data_dir']+data_type_gen+'bin_data/'+gen_dic['add_txt_path'][data_type_gen]+inst+'_'+vis_det+'_'+prop_dic['dim_bin']+'_add')

    #Retrieve master spectrum
    if data_bin['n_exp']>1:stop('Bin data into a single master spectrum')
    data_mast = dataload_npz(gen_dic['save_data_dir']+data_type_gen+'bin_data/'+gen_dic['add_txt_path'][data_type_gen]+inst+'_'+vis_det+'_'+prop_dic['dim_bin']+str(0))

    #Check for alignment
    if (not gen_dic['align_'+data_type_gen]) or ((data_type_gen=='DI') and (not gen_dic['align_DI'] and data_bin['sysvel']==0.)):
        stop('Data must have been aligned in the stellar rest frame')
        
    #Check for in-transit contamination
    if (data_type_gen=='DI') and data_bin['in_inbin']:
        stop('Disk-integrated master contain in-transit profiles')
        
    #CCF masks for disk-integrated and intrinsic spectra
    dic_sav = {}
    if (data_type_gen in ['DI','Intr']):
        if data_dic['DI']['mask']['verbose']:print('           Continuum-normalization')

        #Limit master to minimum definition range
        idx_def_mast = np_where1D(data_mast['cond_def'][0])
        flux_mast = data_mast['flux'][:,idx_def_mast[0]:idx_def_mast[-1]+1]
        cen_bins_mast = data_mast['cen_bins'][:,idx_def_mast[0]:idx_def_mast[-1]+1]
        cond_def_mast = data_mast['cond_def'][:,idx_def_mast[0]:idx_def_mast[-1]+1]
        edge_bins_mast = data_mast['edge_bins'][:,idx_def_mast[0]:idx_def_mast[-1]+2]
        nspec = len(flux_mast[0])
        
        #Continuum-normalisation
        cont_func_dic = dataload_npz(gen_dic['save_data_dir']+'Stellar_cont_DI/'+inst+'_'+vis_det+'/')['cont_func_dic']
        flux_mast_norm = flux_mast[0]
        flux_mast_norm[~cond_def_mast[0]] = 0.
        flux_mast_norm[cond_def_mast[0]] /=cont_func_dic[0](cen_bins_mast[0,cond_def_mast[0]])

        #---------------------------------------------------------------------------------------------------------------------
        #Telluric contamination
        if data_dic['DI']['mask']['verbose']:print('           Mean telluric spectrum')
        tell_spec = np.zeros(nspec,dtype=float) if data_inst['tell_sp'] else None
        rv_tell_inbin = []
        nexp_in_bin = np.zeros(nspec,dtype=float) 
        for vis_bin in data_bin['vis_iexp_in_bin']:
            
            #Retrieve mean RV shifts used to align data
            RV_star_solCDM = dataload_npz(gen_dic['save_data_dir']+'Aligned_DI_data/'+inst+'_'+vis_bin+'__add')['rv_shift_mean']
            if (data_type_gen=='Intr'):RV_surf_star = dataload_npz(gen_dic['save_data_dir']+'Aligned_Intr_data/'+inst+'_'+vis_bin+'_in_add')['rv_shift_mean']

            #Define mean telluric spectrum over all exposures used in the master stellar spectrum
            #    - iexp is relative to global or in-transit indexes depending on data_type                
            for iexp in data_bin['vis_iexp_in_bin'][vis_bin]:
                if data_type_gen=='Intr':iexp_orig = gen_dic[inst][vis_bin]['idx_in'][iexp]
                else:iexp_orig = iexp

                #Align and resample telluric spectra
                #    - disk-integrated spectra used in the binning were aligned in the star rest frame, so we shift them to the Earth rest frame as
                # rv(tell/earth) = rv(tell/star) - (BERV - RV_star_solCDM)
                #    - intrinsic spectra used in the binning were aligned in their common rest frame, so we shift them to the Earth rest frame as
                # rv(tell/earth) = rv(tell/surf) - (BERV - RV_star_solCDM - rv(surf/star))      
                #    - the resulting master spectrum is aligned in the Earth rest frame
                rv_earth_mast_exp = data_prop[inst][vis_bin]['BERV'][iexp_orig] - RV_star_solCDM[iexp_orig]
                if (data_type_gen=='Intr'):rv_earth_mast_exp-=RV_surf_star[iexp]
                rv_tell_inbin+=[rv_earth_mast_exp]
                if data_inst['tell_sp']:
                    
                    #Retrieve the 1D telluric spectrum associated with the exposure
                    #    - we retrieve the spectrum associated with the 1D exposure before it was binned
                    data_exp = dataload_npz(data_bin['vis_iexp_in_bin'][vis_bin][iexp]['data_path'])
                    cond_def_exp = data_exp['cond_def'][0]
                    tell_exp = dataload_npz(data_bin['vis_iexp_in_bin'][vis_bin][iexp]['tell_path'])['tell'][0]      
                    tell_exp[~cond_def_exp] = np.nan
                    edge_bins_earth=data_exp['edge_bins'][0]*spec_dopshift(rv_earth_mast_exp)/(1.+1.55e-8)                    
                    tell_exp = bind.resampling(edge_bins_mast[0],edge_bins_earth,tell_exp, kind=gen_dic['resamp_mode'])
                
                    #Co-add
                    cond_def_tell = ~np.isnan(tell_exp)
                    tell_spec[cond_def_tell]+=tell_exp[cond_def_tell]
                    nexp_in_bin[cond_def_tell]+=1.

        #Average telluric spectrum
        if data_inst['tell_sp']:
            cond_def_tell = nexp_in_bin>0.
            tell_spec[cond_def_tell]/=nexp_in_bin[cond_def_tell]
            tell_spec[~cond_def_tell] = 1.
            
        #Min/max telluric RV in the rest frame of the stellar master spectrum 
        min_rv_earth_mast = np.min(rv_tell_inbin)
        max_rv_earth_mast = np.max(rv_tell_inbin)            

        #Mask generation
        #    - defined in the stellar (for disk-integrated profiles) or surface (for intrinsic profiles) rest frames
        mask_waves,mask_weights,mask_info = kitcat_mask(mask_dic,mask_dic['fwhm_ccf'],cen_bins_mast[0],inst,edge_bins_mast[0],flux_mast_norm,gen_dic,save_data_paths,tell_spec,data_dic['DI']['sysvel'][inst][vis_bin],min_rv_earth_mast,
                                                        max_rv_earth_mast,system_param['star']['Tpole'],dic_sav,plot_dic[data_type_gen+'mask_spectra'],plot_dic[data_type_gen+'mask_ld'],plot_dic[data_type_gen+'mask_ld_lw'],plot_dic[data_type_gen+'mask_RVdev_fit'],cont_func_dic,
                                                        data_bin['vis_iexp_in_bin'],data_type_gen,data_dic,plot_dic[data_type_gen+'mask_tellcont'],plot_dic[data_type_gen+'mask_vald_depthcorr'],plot_dic[data_type_gen+'mask_morphasym'],
                                                        plot_dic[data_type_gen+'mask_morphshape'],plot_dic[data_type_gen+'mask_RVdisp'])

        #Save mask in format readable by ESPRESSO-like DRSs
        #    - the convention for those DRS is that masks are defined in air
        if gen_dic['sp_frame']=='vacuum':
            n_refr=air_index(mask_waves, t=15., p=760.)
            DRS_waves=mask_waves/n_refr
        else:DRS_waves = mask_waves
    
        #Create data columns
        c1 = fits.Column(name='lambda', array=DRS_waves, format='1D')
        c2 = fits.Column(name='contrast', array=mask_weights, format='1E')
        t = fits.BinTableHDU.from_columns([c1, c2])
        
        #Create header columns
        hdr = fits.Header()
        hdr['HIERARCH ESO PRO CATG'] = 'MASK_TABLE'
        inst_ref = inst.split('_')[0]
        hdr['INSTRUME'] = inst_ref
        p = fits.PrimaryHDU(header=hdr)
        
        #Saving
        hdulist = fits.HDUList([p,t])
        mask_name = 'CCF_mask_'+data_type_gen+'_'+gen_dic['star_name']+'_'+inst_ref+mask_info
        hdulist.writeto(save_data_paths+mask_name+'.fits',overwrite=True,output_verify='ignore')
        np.savetxt(save_data_paths+mask_name+'_'+gen_dic['sp_frame']+'.txt', np.column_stack((mask_waves,mask_weights)),fmt=('%15.10f','%15.10f') )

    else:
        flux_mast_norm = flux_mast[0]

        stop('Code Atmospheric mask routine based on expected species in the planet')

    #Save for plotting
    if (plot_dic[data_type_gen+'mask_spectra']!='') | (plot_dic[data_type_gen+'mask_ld_lw']!='') | (plot_dic[data_type_gen+'mask_RVdev_fit']!=''):
        datasave_npz(save_data_paths+'Plot_info',dic_sav)

    return None











'''
Function to calculate positional properties of planet in each exposure
'''
def coord_expos(pl_loc,coord_dic,inst,vis,star_params,pl_params,bjd_inst,exp_time,data_dic,RpRs):

    #Orbital phase and corresponding times (days) of current exposure relative to transit center  
    st_phases,phases,end_phases=get_timeorbit(pl_loc,coord_dic,inst,vis,bjd_inst, pl_params, exp_time)[0:3]   

    #Exposure duration in phase
    ph_dur=end_phases-st_phases

    #Return values for start, mid, end exposure    
    xp_all,yp_all,zp_all,_,_,_,_,ecl= calc_pl_coord(pl_params['ecc'],pl_params['omega_rad'],pl_params['aRs'],pl_params['inclin_rad'],np.vstack((st_phases,phases,end_phases)),RpRs,pl_params['lambda_rad'],star_params)
    eclipse = ecl[0]
    st_positions= [xp_all[0][0],yp_all[0][0],zp_all[0][0]]
    positions= [xp_all[1][0],yp_all[1][0],zp_all[1][0]]
    end_positions= [xp_all[2][0],yp_all[2][0],zp_all[2][0]]

    #Calculation of planet orbital radial velocity in the star rest frame (km/s)
    if data_dic['Atm']['exc_plrange']:
  
        #Number of oversampling RV bins in exposure
        dph_loc=end_phases-st_phases
        nt_osamp_RV=max(int(dph_loc/data_dic['Atm']['dph_osamp_RVpl']),2)

        #Adjusted resolution
        dph_osamp_loc=dph_loc/nt_osamp_RV
        
        #Table of oversampling bins 
        #    - oversampling bins are centered in phase
        ph_osamp_loc = st_phases+dph_osamp_loc*(0.5+np.arange(nt_osamp_RV))  

        #Average velocity over oversampled values
        #    - converted from velocity along the LOS in /Rstar/h to km/s
        rv_pl_all=star_params['RV_conv_fact']*np.mean(calc_pl_coord(pl_params['ecc'],pl_params['omega_rad'],pl_params['aRs'],pl_params['inclin_rad'],ph_osamp_loc,None,None,None,rv_LOS=True,omega_p=pl_params['omega_p'])[6])        
                
    else:rv_pl_all=None
    
    return positions,st_positions,end_positions,eclipse,rv_pl_all,st_phases,phases,end_phases,ph_dur


'''
Definition of transit phase
    Identify planet position with respect to stellar disk
        - -1/1 : pre/post-tr; -2/2: ingress/egress; -3/3 full transit
          signs indicate pre/post conjunction
        - to be conservative, an exposure is considered out of the transit if it is out during the entire exposure
          similarly an exposure is considered fully in-transit if it is in during the entire exposure
    Input coordinates must either be central values, or (start,central,end) values in each exposure
'''
def eclipse_def(Dprojp_all,RpRs,lambda_rad,star_params,xp_sk_all,yp_sk_all):
    dim_input = np.shape(xp_sk_all)
    
    #Oblate star
    if star_params['f_GD']>0.:
        nlimb = 501
        if len(dim_input)==1:
            nexp = len(xp_sk_all)
            x_st_sky_all,y_st_sky_all,_=conv_Losframe_to_inclinedStarFrame(lambda_rad,xp_sk_all,yp_sk_all,None) 
            nlimb_in_ph = np.array([ oblate_ecl(nlimb,RpRs,[x_st_sky],[y_st_sky],star_params)[0] for x_st_sky,y_st_sky in zip(x_st_sky_all,y_st_sky_all) ])   
        else:
            nexp = len(xp_sk_all[0])

            #Frame conversion of planet coordinates from the classical frame perpendicular to the LOS, to the 'inclined star' frame
            start_x_st_sky_all,start_y_st_sky_all,_=conv_Losframe_to_inclinedStarFrame(lambda_rad,xp_sk_all[0],yp_sk_all[0],None)           
            end_x_st_sky_all,end_y_st_sky_all,_=conv_Losframe_to_inclinedStarFrame(lambda_rad,xp_sk_all[2],yp_sk_all[2],None) 
    
            #Number of planet limb points within the projected stellar photosphere
            nlimb_in_ph = np.zeros(nexp,dtype=float)
            for isub,(start_x_st_sky,start_y_st_sky,end_x_st_sky,end_y_st_sky) in enumerate(zip(start_x_st_sky_all,start_y_st_sky_all,end_x_st_sky_all,end_y_st_sky_all)):
                start_nlimb_in_ph = oblate_ecl(nlimb,RpRs,[start_x_st_sky],[start_y_st_sky],star_params)[0]
                end_nlimb_in_ph = oblate_ecl(nlimb,RpRs,[end_x_st_sky],[end_y_st_sky],star_params)[0]
                nlimb_in_ph[isub] = start_nlimb_in_ph+end_nlimb_in_ph
    
        #Planetary disk is outside the projected stellar photosphere
        cond_out = (nlimb_in_ph==0)

        #Planet is entirely in front of the disk  
        cond_in = (nlimb_in_ph==2*nlimb)
            
    #Spherical star    
    else:
        if len(dim_input)==1:
            nexp = len(Dprojp_all)
            cond_out = (Dprojp_all > (1.+RpRs)) 
            cond_in =  (Dprojp_all <= (1.-RpRs))  
        else:
            nexp = len(Dprojp_all[0])
            cond_out = (Dprojp_all[0] > (1.+RpRs)) & (Dprojp_all[2] > (1.+RpRs))
            cond_in = (Dprojp_all[0] <= (1.-RpRs)) & (Dprojp_all[2] <= (1.-RpRs))
         
    #Default (ingress/egress phase)
    eclipse = np.repeat(2.,nexp)
    
    #Planetary disk is outside the projected stellar photosphere
    eclipse[cond_out] = 1.

    #Planet is entirely in front of the disk  
    eclipse[cond_in] = 3.

    #Pre-transit
    if (len(dim_input)==1):eclipse[xp_sk_all<0.]*=-1.
    else:eclipse[0.5*(xp_sk_all[0]+xp_sk_all[2])<0.]*=-1.
                                    
    return eclipse

'''
Number of planet limb points within the projected stellar photosphere
    - for a given planet position we check how many planetary limb points are in front of the projected photosphere
'''
def oblate_ecl(nlimb,RpRs,xp_st_sk,yp_st_sk,star_params):
    xlimb = RpRs*np.cos(2*np.pi*np.arange(nlimb)/(nlimb-1.))
    ylimb = RpRs*np.sin(2*np.pi*np.arange(nlimb)/(nlimb-1.))
    nlimb_in_ph = np.zeros(len(xp_st_sk),dtype=int)
    for iloc,(xp_st_sk_loc,yp_st_sk_loc) in enumerate(zip(xp_st_sk,yp_st_sk)):
        x_st_sk_limb = xp_st_sk_loc+xlimb
        y_st_sk_limb = yp_st_sk_loc+ylimb
        cond_in_stphot=calc_zLOS_oblate(x_st_sk_limb,y_st_sk_limb,star_params['istar_rad'],star_params['RpoleReq'])[2] 
        nlimb_in_ph[iloc] = np.sum( cond_in_stphot )    
    return nlimb_in_ph















'''
Return True anomaly and orbital position for given position
------------------------
    - references:
 http://www.gehirn-und-geist.de/sixcms/media.php/370/Leseprobe.406372.pdf
 http://www.relativitycalculator.com/pdfs/RV_Derivation.pdf

    - radial velocity is negative when planet is coming toward us, thus in the same frame as the observed velocity tables

    - in the pipeline, we calculate rv(pl/star) directly from the 3D coordinates of the planet in the star rest frame
 rv(pl/star) = - K[pl/star]*( cos(nu+omega_bar)+ecc*cos(omega_bar) )    
      with K[pl/star] = ( 2*pi*a*sin(ip)/(P*sqrt(1-e^2)) )

    - Kepler third's law gives 
 P^2/a^3 = 4*pi^2/G*(Mp+Mstar)
 a = ( G*P^2*(Mstar+Mp) / (4*pi^2) )^(1/3) 
      thus we can rewrite
 K[pl/star] = (2*pi*G/P)^(1/3)*(Mstar+Mp)^(1/3)*sin(ip)/(sqrt(1-e^2))        

    - we then have 
 rv(pl/star) = rv(pl/CMD_star) - rv(star/CDM_star)
      and because the barycenter is fixed in its own frame:
 Mstar*rv(star/CDM_star) +  Mpl*rv(pl/CDM_star) = 0         
 Mstar*rv(star/CDM_star) +  Mpl*(rv(pl/star)+rv(star/CDM_star)) = 0         
 rv(star/CDM_star)= - rv(pl/star)*Mpl/(Mstar+Mp) 
      thus 
 rv(star/CDM_star) = K[star/CDM_star]*( cos(nu+omega_bar)+ecc*cos(omega_bar) )  
      with K[star/CDM_star] = K[pl/star]*Mpl/(Mstar+Mp)
           K[star/CDM_star] = (2*pi*G/P)^(1/3)*Mpl*sin(ip)/((Mstar+Mp)^(2/3) * sqrt(1-e^2)) 
      with K in km/s, G in m3 kg-1 s-2, P in s, Mp and Mstar in kg  
           [ (2*!dpi*6.67300e-11)^(1./3.) * 1.9891001e+030]/[(24.*3600)^(1./3.) *(1.9891001e+030)^(2./3.)]/1000. = 212.91918458020422
           K[star/CDM_star] = (212.91918458020422/P_days^(1./3.))*Mp_sun*sin(ip)/((Mstar_sun+Mp_sun)^(2./3.)*sqrt(1-e^2))
           K[star/CDM_star] = (0.02843112300449059/P_yr^(1./3.))*Mp_sun*sin(ip)/((Mstar_sun+Mp_sun)^(2./3.)*sqrt(1-e^2))           
          
    - we can also write   
 rv(pl/CDM_star) =  rv(pl/star) + rv(star/CDM_star)
                 =  ( - K[pl/star] + K[pl/star]*Mpl/(Mstar+Mp) )*( cos(nu+omega_bar)+ecc*cos(omega_bar) )    
                 =  K[pl/CDM_star]*( cos(nu+omega_bar)+ecc*cos(omega_bar) )  
      with K[pl/CDM_star] = K[pl/star]*( Mpl/(Mstar+Mp)  - 1 )
                          = K[pl/star]*( (Mpl-Mstar)/(Mstar+Mp) )
                          = - (2*pi*G/P)^(1/3)*(Mstar-Mpl)*sin(ip)/(sqrt(1-e^2)*(Mstar+Mp)^(1/3) )

    - here we calculate the perifocal coordinates of the planet, in cartesian coordinates relative to the star
      the radial velocity rv(pl/star) is one of the final coordinates
      see https://www.sciencedirect.com/topics/engineering/eccentric-anomaly

    - if we assume that the atmospheric signal tracks the planet orbital velocity (ie, there is no atmospheric dynamics), then we can 
      link the planet and star masses with this measurement. Doing as few assumption as possible, the quantity we measure is 
 rv(pl/CDM_sun) = rv(pl/CDM_star) + rv(CDM_star/CDM_sun)
 rv(pl/CDM_sun) = - (2*pi*G/P)^(1/3)*(Mstar-Mpl)*sin(ip)/(sqrt(1-e^2)*(Mstar+Mp)^(1/3) )   +      rv_sys   
 
'''
def calc_pl_coord(ecc,omega_bar,aRs,inclin,ph_loc,RpRs,lambda_rad,star_params,rv_LOS=False,omega_p=None):
      
    #True anomaly
    True_anom_loc,Ecc_anom_loc,_=True_anom_calc(ecc,ph_loc,omega_bar)            

    #Orbital frequency (yr-1 to h-1) and velocity coordinates initialization 
    if rv_LOS:omega_p_h = omega_p/(365.2425*24.) 
    else:vxp_loc,vyp_loc,vzp_loc=None,None,None
            
    #Circular orbit
    #  - 1e-4 instead of 0 to avoid loosing too much time with really low tested values of ecc    
    if (ecc<1e-4):

        #Coordinates in the orbital plane oriented toward Earth
        #    - axis:
        # + X1 is the node line, in the plane of sky (ie the plane perpendicular to the LOS)
        #      obtained by rotation of Z1 in the direction of the orbital motion 
        # + Y1 is the perpendicular to the node line in the orbital plane (oriented toward the half space containing Earth)
        # + Z1 is the normal to the orbital plane   
        #      z1 is null because the planet moves in the x1,y1 plane 
        #    - coordinates in /Rstar and /Rstar/h
        X1_p = aRs*np.sin(True_anom_loc)        
        Y1_p = aRs*np.cos(True_anom_loc)
        if rv_LOS:      
            VX1_p= aRs*omega_p_h*np.cos(True_anom_loc)            
            VY1_p=-aRs*omega_p_h*np.sin(True_anom_loc)

        #Coordinates in the sky-projected stellar referential 
        #    - axis:
        # + X2 = X1         
        # + Y2 = Z1 obtained as Z2
        #      projection of the oriented normal (ie the spin axis of the planet motion along its orbit) to the orbital plane in the plane of sky (perpendicular to the LOS)  
        #      the planet always moves in front of the star from J2- to J2+ (with I2+ toward the Earth)
        #      y2 = - y1 cos(inclin) + z1 sin(inclin)                
        # + Z2 is the line of sight from star center toward Earth
        #      obtained by rotation of Y1 around X1 by inclin-90 (inclin = angle from the LOS to the normal of the orbital plane)
        #      z2 = y1 sin(inclin) + z1 cos(inclin)   
        #    - coordinates in /Rstar and /Rstar/h
        #for a circular orbit, only the inclination is necessary, since the orbital plane is oriented so that the
        #node line is the same as the Y axis of the stellar referential 
        xp_loc =  X1_p
        yp_loc = -Y1_p*np.cos(inclin)        
        zp_loc =  Y1_p*np.sin(inclin)
        if rv_LOS:
            vxp_loc=  VX1_p 
            vyp_loc= -VY1_p*np.cos(inclin)          
            vzp_loc=  VY1_p*np.sin(inclin)            
            
            
    #--------------------------------------------------------------------------------------------------------------------
    #Eccentric orbit   
    #  - by definition the ascending node is where the object goes toward the observer through the plane of sky
    #  - omega_bar is the angle between the ascending node and the periastron, in the orbital plane (>0 counterclockwise)                       
    else:

        #Planet position in the orbital plane in Rstar
        #  - origin at the focus (star)
        #  - I0 : major axis ( a*cos(E) - a*cos(e) ) oriented toward the periapsis
        #  - J0 : perpendicular to I0 ( b*sin(E) ) when rotating with the orbital motion
        #  - K0 : normal to the orbital plane 
        X0_p=aRs*(np.cos(Ecc_anom_loc)-ecc)
        Y0_p=aRs*np.sqrt(1.-ecc*ecc)*np.sin(Ecc_anom_loc)
    
        #Coordinates in the orbital plane oriented toward the Earth in Rstar
        #Omega_bar : angle between the ascending node and the periastron in the orbital plane
        #  - I1 is the node line (from K1 when rotating with the orbital motion)    
        #    Y1 is the perpendicular to the node line in the orbital plane, toward the Earth
        #  - Z1 = ( sin(omega_bar) # cos(omega_bar)  ) in (I0,J0)
        #    Y1 = ( -cos(omega_bar) # sin(omega_bar)  ) in (I0,J0)
        X1_p =  -X0_p*np.cos(omega_bar) +  Y0_p*np.sin(omega_bar)        
        Y1_p =  X0_p*np.sin(omega_bar) +  Y0_p*np.cos(omega_bar)         

        #Coordinates in the plane containing the LOS and the node line
        #   - Z : LOS toward Earth (from I1 by rotating with angle inclin)
        #   - X : node line (=K1)          
        #   - Y : complete the right handed referential
        xp_loc=X1_p            
        yp_loc=-Y1_p*np.cos(inclin)
        zp_loc= Y1_p*np.sin(inclin)


        #Velocity coordinates
        #    - in /Rstar/h 
        if rv_LOS:

            #Distance star - planet (/Rstar)
            Dplanet=aRs*(1.-ecc*np.cos(Ecc_anom_loc))
 
            #Planet velocity in the orbital plane 
            #  - vx = - a sin(E) dE/dt
            #    vy = a sqrt(1-e^2) cos(E) dE/dt
            #  - dM = omega_p_h*dt = dE ( 1 - e cos(E) ) = dE*Dplanet/a_pl  
            VX0_p =-omega_p_h*(pow(aRs,2.))*np.sin(Ecc_anom_loc)/Dplanet
            VY0_p = omega_p_h*(pow(aRs,2.))*np.sqrt(1-pow(ecc,2.))*np.cos(Ecc_anom_loc)/Dplanet    

            #Planet velocity in the orbital plane oriented toward the Earth   
            VX1_p = -VX0_p*np.cos(omega_bar) + VY0_p*np.sin(omega_bar)  
            VY1_p =  VX0_p*np.sin(omega_bar) + VY0_p*np.cos(omega_bar)           
            
            #Coordinates in the sky-projected referential 
            vxp_loc=  VX1_p
            vyp_loc= -VY1_p*np.cos(inclin)  
            vzp_loc=  VY1_p*np.sin(inclin)   

    #--------------------------------------------------------------------------------------------------------------------
    #Distance star - planet in the plane of sky
    #    - in /Rstar
    Dprojplanet=np.sqrt(xp_loc*xp_loc + yp_loc*yp_loc)

    #--------------------------------------------------------------------------------------------------------------------
    #Eclipse status  
    if RpRs is not None:ecl_loc = eclipse_def(Dprojplanet,RpRs,lambda_rad,star_params,xp_loc,yp_loc)     
    else:ecl_loc = None
          
    return xp_loc,yp_loc,zp_loc,Dprojplanet,vxp_loc,vyp_loc,vzp_loc ,ecl_loc   
    
    
    
    
    
    


    


'''
Time of the orbit at the beginning/center/end of the exposure (days)
'''    
def get_timeorbit(pl_loc,coord_dic,inst,vis,bjd_tab, PlParam, exp_time_tab):
    
    #Transit center (days)
    Tcen= coord_dic[inst][vis][pl_loc]['Tcenter'] - 2400000.  

    #Orbital phase (between +-0.5) and corresponding times (d)
    phase_temp=(bjd_tab-Tcen)/PlParam["period"]
    mid_phase_tab = (phase_temp - npint(phase_temp+np.sign(phase_temp)*0.5))
    mid_time_tab  = mid_phase_tab*PlParam["period"]
    if exp_time_tab is not None:
        
        #Half exposure time (days)
        texp_d_tab=exp_time_tab/(2.*24.*3600.)

        #Times
        phase_temp=(bjd_tab-texp_d_tab-Tcen)/PlParam["period"]
        st_phase_tab = (phase_temp - npint(phase_temp+np.sign(phase_temp)*0.5))
        st_time_tab  = st_phase_tab*PlParam["period"]
        phase_temp=(bjd_tab+texp_d_tab-Tcen)/PlParam["period"]
        end_phase_tab = (phase_temp - npint(phase_temp+np.sign(phase_temp)*0.5))
        end_time_tab  = end_phase_tab*PlParam["period"]
        
    else:st_phase_tab,end_phase_tab,st_time_tab,end_time_tab=None,None,None,None

    return st_phase_tab,mid_phase_tab,end_phase_tab,st_time_tab,mid_time_tab,end_time_tab





 

def True_anom_calc(ecc,phase,omega_bar):
    
    #Circular orbit
    #  - 1e-4 instead of 0 to avoid loosing too much time with really low tested values of ecc    
    if (ecc<1e-4):

        #Mean anomaly
        Mean_anom=2.*np.pi*phase

        #Eccentric anomaly
        Ecc_anom=Mean_anom
        
        #True anomaly
        True_anom=Mean_anom
                                
    #--------------------------------------------------------------------------------------------------------------------
    #Eccentric orbit   
    #  - by definition the ascending node is where the object goes toward the observer through the plane of sky
    #  - omega_bar is the angle between the ascending node and the periastron, in the orbital plane (>0 counterclockwise)                       
    else:

        #Mean anomaly
        #  - time origin of t_mean at the periapsis (t_mean=0 <-> M=0 <-> E=0)
        #  - M(t_mean)=M(dt_transit)+M(t_simu) 
        Mean_anom=2.*np.pi*phase+Mean_anom_TR_calc(ecc,omega_bar)
			        
        #Eccentric anomaly : 
        #  - M = E - e sin(E)
        #    - >0 counterclockwise
        #  - angle, with origin at the ellipse center, between the major axis toward the periapsis and the 
        #line crossing the circle with radius 'aRs' at its intersection with the perpendicular to
        #the major axis through the planet position
        if np.isscalar(Mean_anom):
            Ecc_anom=newton(Kepler_func,Mean_anom,args=(Mean_anom,ecc,))
        else:
            dim_input = np.shape(Mean_anom)
            if len(dim_input)==1:Ecc_anom=np.array([newton(Kepler_func,Mean_anom_loc,args=(Mean_anom_loc,ecc,)) for Mean_anom_loc in Mean_anom])
            else:
                Ecc_anom = np.zeros(dim_input,dtype=float)
                for idim in range(dim_input[0]):Ecc_anom[idim]=np.array([newton(Kepler_func,Mean_anom_loc,args=(Mean_anom_loc,ecc,)) for Mean_anom_loc in Mean_anom[idim]])

        #True anomaly of the planet at current time
        True_anom=2.*np.arctan(np.sqrt((1.+ecc)/(1.-ecc))*np.tan(Ecc_anom/2.))
                
    return True_anom,Ecc_anom,Mean_anom 

 




 
'''
Spline Interpolation of a given function
    Use scipy.interpolate.InterpolatedUnivariateSpline
    the sqrt is given as the error
    
    NB: here the shift is a translation
    
    if y is shifted like
       y[a+shift:b+shift]
    or xnew = x + shift
    or x = x - shift
    then: -shift>0 => ynew is blue shift
          -shift<0 => ynew is red shift
    NB: Be careful to the scale (not the same shift in x or y)
        
    keyword arguments:
    x -- Old x axis
    y -- Old y axis
    xnew -- New x axis
    k -- The Spline Order (1=linear, 3=cubic)

'''
def spline_inter(x, y, xnew, k=3 , ext = 0 ):
    splflux = interpolate.InterpolatedUnivariateSpline(x, y, k=k , ext = ext)
    ynew = splflux(xnew)
    errorynew = np.sqrt(np.maximum(ynew, 0.))
    return ynew, errorynew

'''
#Perform stellar Orbit Correction
#    - all profiles are currently as a function of v(M/sun)
#      we substract v(star/sun) = v(star/CDM) + v(CDM/sun)
#      if v(CDM/sun) is unknown, it can be set to 0 as input and derived later afterwards by fitting the master DI CCFs
#    - Keplerian motion of the star induced by a given planet is:
# v(star/CDM ; from pl)= K*( cos(nu+omega_bar)+ecc*cos(omega_bar) )
#      with K, nu, ecc and omega_bar specific to the planet
#      velocity is negative when planet is coming toward us, thus in the same frame as the observed velocity tables
#    - Keplerian semi-amplitude:
# + the general formula is
# K = (2*pi*G/P)^(1./3.)  *  Msec * sin(inc) * (Msec+Mprim)^(1./3.) / Mprim   * 1/sqrt(1-e^2)
#   with K in km/s, G in m3 kg-1 s-2, P in s, Mp and Mprim in kg    
#   and [ (2*!dpi*6.67300e-11)^(1./3.) * 1.9891001e+030*(1.9891001e+030)^(1./3.)]/[(24.*3600)^(1./3.) *(1.9891001e+030)]/1000. = 212.91918458020422
# K = (212.91918458020422/P_days^(1./3.)) * Msec*sin(inc)*(Msec+Mprim)^(1./3.)/Mprim*1/sqrt(1-e^2)
#   with P_days in days, Mp and Mprim in Msun
# + for a planet Msec << Mprim: 
# K ~ (2*pi*G/P)^(1/3)  *  Mp * sin(inc)/(Mstar)^2./3. * 1/sqrt(1-e^2)
#   with K in km/s, G in m3 kg-1 s-2, P in s, Mp and Mstar in kg
#   and [ (2*!dpi*6.67300e-11)^(1./3.) * 1.8986000e+027]/[(24.*3600)^(1./3.) *(1.9891001e+030)^(2./3.)]/1000. = 0.20323137    
# >> K = (0.20323137323712528/P_jours^(1./3.))*Mp*sin(inc)/Mstar^(2./3.)*1/sqrt(1-e^2) 
#   with P_days in days, Mp and Mstar in Msun
# >> K = (0.02843112300449059/P_years^(1./3.))*Mp*sin(inc)/Mstar^(2./3.)*1/sqrt(1-e^2) 
#   with P_years in years, Mp in Mjup and Mstar in Msun  
'''
def calc_Kpl(params,star_params):    #Returns Kpl in m/s
    return (2.*np.pi*G_usi/(params['period_s']))**(1./3.)*(params['Msini']*Mjup/(star_params['Mstar']*Msun)**(2./3.))*1./np.sqrt(1.-params['ecc']**2.)


def calc_orb_motion(coord_dic,inst,vis,system_param,gen_dic,bjd_exp,dur_exp,sysvel):

    #Oversampling factor of exposures
    n_ov=10.

    #-------------------------------------    
    #Keplerian motion induced by requested planets
    RV_star_stelCDM_ov=0.
    for pl_loc in gen_dic['kepl_pl']:     
        PlParam_loc=system_param[pl_loc]   
        
        #Orbital phase 
        st_phase,phase,end_phase=get_timeorbit(pl_loc,coord_dic,inst,vis,bjd_exp, PlParam_loc, dur_exp)[0:3]

        #True anomaly for start and end of exposure
        True_anom=True_anom_calc(PlParam_loc['ecc'],np.vstack((st_phase,end_phase)),PlParam_loc['omega_rad'])[0]

        #Oversampling of true anomaly over exposure
        min_TA=np.min(True_anom)
        max_TA=np.max(True_anom)
        
        #Oversampling
        dTrueAnom_ov=(max_TA-min_TA)/n_ov            
        True_anom_ov=min_TA+dTrueAnom_ov*(0.5+np.arange(n_ov,dtype=float))

        #Keplerian motion induced by the planet (km/s)
        RV_star_stelCDM_ov+=PlParam_loc['Kstar_kms']*(np.cos(True_anom_ov+PlParam_loc['omega_rad'])+PlParam_loc['ecc']*np.cos(PlParam_loc['omega_rad']) )                
    
    #-------------------------------------    

    #Average oversampled RV (km/s)
    RV_star_stelCDM=np.mean(RV_star_stelCDM_ov)  
    
    #Stellar radial velocity relative to the sun
    RV_star_solCDM=RV_star_stelCDM+sysvel

    return RV_star_stelCDM,RV_star_solCDM


































'''
Theoretical radial velocity of star relative to CDM
    - must be calculated for each visit, because the RV curve will not be the same if there are multiple planets
'''
def orb_motion_theoRV(pl_ref,system_param,kepl_pl,coord_dic,inst,vis,data_dic):

    #Phase resolution and high-resolution phase table
    #    - we extend slightly the table on both sides
    #    - we take a temporal resolution shorter than the shorter exposure in the visit
    nexp_HR=10
    t_dur_d=coord_dic[inst][vis]['t_dur_d']
    min_texpd=np.min(t_dur_d)
    dbjd_HR=min_texpd/nexp_HR
    min_bjd=np.min(coord_dic[inst][vis]['bjd']-t_dur_d)
    max_bjd=np.max(coord_dic[inst][vis]['bjd']+t_dur_d)
    nbjd_HR=int((max_bjd-min_bjd)/dbjd_HR)+1
    bjd_tab_HR=min_bjd+dbjd_HR*np.arange(nbjd_HR)   
    phase_tab_HR=np.zeros(nbjd_HR, dtype=float)
    if pl_ref not in kepl_pl:stop('Current planet not accounted for in Keplerian motion')
    
    #Radial velocity table
    RV_star_stelCDM = np.zeros(nbjd_HR, dtype=float)

    #Motion for all instrument exposures
    for i_loc,bjd_loc in enumerate(bjd_tab_HR):
        for ipl,pl_loc in enumerate(kepl_pl):
            PlParam=system_param[pl_loc]
                
            #Orbital phase 
            phase_temp=(bjd_loc - (coord_dic[inst][vis][pl_loc]['Tcenter'] - 2400000.))/PlParam["period"]
            phase = (phase_temp - npint(phase_temp+np.sign(phase_temp)*0.5)) 
            if pl_loc==pl_ref:phase_tab_HR[i_loc]=phase            
            
            #True anomaly
            ecc=PlParam['ecc']
            omega_bar=PlParam['omega_rad']          
            True_anom,Ecc_anom,_=True_anom_calc(ecc,phase,omega_bar)
  
            #Keplerian motion (km/s)
            RV_star_stelCDM[i_loc]+=PlParam['Kstar_kms']*(np.cos(True_anom+omega_bar)+ecc*np.cos(omega_bar) )

    #---- end of loop on exposures 
 
    #Keplerian curve in heliocentric rest frame (km/s)
    RV_star_solCDM=RV_star_stelCDM+data_dic['DI']['sysvel'][inst][vis]

    return phase_tab_HR,RV_star_solCDM





























'''
General routine to fit profiles and extract properties
'''
def ana_prof_func(isub_exp,iexp,inst,data_dic,vis,fit_prop_dic,gen_dic,verbose,cond_def_fit,cond_def_cont,prof_type,  
                  edge_bins,cen_bins,flux_loc,cov_loc,idx_force_det,theo_dic,star_params,fit_properties,line_fit_priors,model_choice,model_prop,data_type): 
    
    #Arguments to be passed to the fit function
    fixed_args = deepcopy(fit_properties)
    output_prop_dic={}

    #Fitted data tables 
    #    - final model is calculated on the same bins as the input data, limited to the minimum global definition range for the fit
    #    - to avoid issues with gaps, resampling/convolution, and chi2 calculation the model is calculated on the full continuous velocity table, and then limited to fitted pixels
    #    - fitted profiles are trimmed and calculated over the smallest continuous range encompassing all fitted pixels to avoid computing models over a too-wide table
    idx_def_fit = np_where1D(cond_def_fit)
    if len(idx_def_fit)==0.:stop('No bin in fitted range')
    cen_bins_fit = cen_bins[cond_def_fit]
    flux_loc_fit=flux_loc[cond_def_fit]
    idx_mod = range(idx_def_fit[0],idx_def_fit[-1]+1)
    fixed_args['idx_fit'] = np_where1D(cond_def_fit[idx_mod])    
    fixed_args['ncen_bins'] = len(idx_mod)
    fixed_args['cen_bins']= cen_bins[idx_mod]
    fixed_args['dim_exp']= [1,idx_mod]
    fixed_args['edge_bins'] = def_edge_tab(fixed_args['cen_bins'][None,:][None,:])[0,0]
    fixed_args['dcen_bins'] = fixed_args['edge_bins'][1::]-fixed_args['edge_bins'][0:-1] 
    fixed_args['x_val']=cen_bins[idx_mod]
    fixed_args['y_val']=flux_loc[idx_mod]
    fixed_args['cov_val'] = cov_loc[:,idx_mod]

    #Activation of spectral conversion and resampling 
    cond_conv_st_prof_tab(theo_dic['rv_osamp_line_mod'],fixed_args,data_type)

    #Resampled spectral table for model line profile
    if fixed_args['resamp']:resamp_model_st_prof_tab(None,None,None,fixed_args,gen_dic,1,theo_dic['rv_osamp_line_mod'])
    
    #Effective instrumental convolution
    if fixed_args['conv_model']:fixed_args['FWHM_inst'] = ref_inst_convol(inst,fixed_args,fixed_args['cen_bins'])    
    else:fixed_args['FWHM_inst'] = None

    #Effective table for model calculation
    fixed_args['args_exp'] = def_st_prof_tab(None,None,None,fixed_args)

    #------------------------------------------------------------------------------- 
    #Guess and prior values
    #------------------------------------------------------------------------------- 

    #-----------------------------------
    #Identification of CCF peak
    #-----------------------------------
    if (prof_type=='Intr'):
        
        #For intrinsic CCFs we assume the local stellar RV is bounded by +=0.5*FWHM, with FWHM an estimate given as prior, and take the RV at minimum in this range   
        if ('FWHM' in model_prop):
            cond_peak=(cen_bins_fit>-0.5*model_prop['FWHM'][inst][vis]['guess']) & (cen_bins_fit<0.5*model_prop['FWHM'][inst][vis]['guess'])
            if np.sum(cond_peak)==0:CCF_peak=0.            
            else:CCF_peak=min(flux_loc_fit[cond_peak])
        else:CCF_peak=min(flux_loc_fit)
        
    elif prof_type=='DI':
        
        #For disk-integrated CCFs we take the RV at minimum as guess 
        CCF_peak=min(flux_loc_fit)
        
    elif prof_type=='Atm':
        #For atmospheric CCFs we take the maximum as guess
        CCF_peak=max(flux_loc_fit)

    #-----------------------------------
    #Centroid velocity (km/s) 
    #-----------------------------------
    RV_guess_tab = [None,None,None]
    
    #Estimate from system properties
    if ('RV_cen' in fixed_args):RV_guess_tab[0] = fixed_args['RV_cen']
    
    #Estimate from measured CCF peak
    else:
        if CCF_peak==0.:RV_guess_tab[0]=np.mean(cen_bins_fit)
        else:RV_guess_tab[0]=cen_bins_fit[flux_loc_fit==CCF_peak][0]
        
    #Guess value from input
    #    - overwrites default values
    if ('rv' in model_prop) and (inst in model_prop['rv']) and (vis in model_prop['rv'][inst]):RV_guess_tab[0] = model_prop['rv'][inst][vis]['guess']
    
    #Prior range  
    if prof_type=='DI':
        RV_guess_tab[1] = RV_guess_tab[0]-200.
        RV_guess_tab[2] = RV_guess_tab[0]+200.       
    elif prof_type=='Intr':
        RV_guess_tab[1] = -3.*star_params['vsini']
        RV_guess_tab[2] = +3.*star_params['vsini']
    elif prof_type=='Atm':
        RV_guess_tab[1] = RV_guess_tab[0]-50.
        RV_guess_tab[2] = RV_guess_tab[0]+50.

    #Guesses for analytical models
    if fixed_args['mode']=='ana':
            
        #-----------------------------------    
        #FWHM (km/s)
        #    - sigma = FWHM / (2*sqrt(2*ln(2)))
        #    - find the approximate position of CCF, and then FWHM (for master out only, otherwise overestimate the FWHM)
        #      for intrinsic CCFs we start from the FWHM of the master out if known
        #    - if there is no signature of the intrinsic CCF, the minimum of the CCF can be found at the edges of the table
        #    - set to 10km/s for atmospheric CCFs
        #-----------------------------------
        if (prof_type in ['DI','Intr']): 
            idx_low=np.where(cen_bins_fit < RV_guess_tab[0])[0]
            if len(idx_low)==0:vlow_FWHM=cen_bins_fit[0]
            else:
                ilow_FWHM=closest(flux_loc_fit[idx_low],0.5*(CCF_peak+fixed_args['flux_cont']))
                vlow_FWHM=cen_bins_fit[idx_low][ilow_FWHM]
            idx_high=np.where(cen_bins_fit >= RV_guess_tab[0])[0]
            if len(idx_high)==0:vhigh_FWHM=cen_bins_fit[-1]
            else:
                ihigh_FWHM=closest(flux_loc_fit[idx_high],0.5*(CCF_peak+fixed_args['flux_cont']))
                vhigh_FWHM=cen_bins_fit[idx_high][ihigh_FWHM]     
            FWHM_guess=(vhigh_FWHM - vlow_FWHM)
        elif (prof_type=='Atm'):
            FWHM_guess=10.
            
        #Upper boundary
        if prof_type=='DI':
            FWHM_max = 200.
        elif prof_type=='Intr':
            FWHM_max = 100.
        elif prof_type=='Atm':
            FWHM_max = 50.
            
        #-----------------------------------
        #Guess of CCF contrast  
        #    - contrast is considered as always positive, thus we define a sign for the amplitude depending on the case
        #-----------------------------------

        ctrst_guess = -(CCF_peak-fixed_args['flux_cont'])/fixed_args['flux_cont']    
        if prof_type in ['DI','Intr']:fixed_args['amp_sign']=-1.
        elif prof_type=='Atm':fixed_args['amp_sign']=1.

    #-------------------------------------------------------------------------------    
    #Parameters and functions
    #    - lower / upper values in 'p_start' will be used as default uniform prior ranges if none are provided as inputs
    #                                                     as default walkers starting ranges (if mcmc fit)
    #-------------------------------------------------------------------------------    

    # Initialise model parameters
    p_start = Parameters()

    #-------------------------------------------------------------------------------
    #Custom model for disk-integrated profiles
    #    - obtained via numerical integration over the disk
    #    - intrinsic CCFs do not necessarily have here the same flux level as the fitted disk-integrated profile, so that the continuum level should be left as a free parameter
    if (model_choice=='custom') and (prof_type=='DI'):
        
        #Custom model initialization
        fixed_args.update({
            'mac_mode':theo_dic['mac_mode'], 
            'inst':inst,
            'vis':vis})

        #Grid initialization
        fixed_args['DI_grid'] = True
        fixed_args['conv2intr'] = False
        fixed_args,p_start = init_custom_DI_par(fixed_args,gen_dic,data_dic['DI']['system_prop'],star_params,p_start,RV_guess_tab)

        #Fit function
        fixed_args['fit_func']=MAIN_custom_DI_prof
        fixed_args['fit_func_gen']=custom_DI_prof

    #Analytical models
    else:
        
        #Continuum
        p_start.add_many(('cont',    fixed_args['flux_cont'], False,   None,None,None))      
        
        #Centroid
        p_start.add_many(('rv',      RV_guess_tab[0],      True,   RV_guess_tab[1], RV_guess_tab[2],     None)) 

        #Contrast and FWHM
        p_start.add_many(('ctrst',   ctrst_guess,   True,   0.,     1. ,        None),              
                         ('FWHM',    FWHM_guess,    True,   0.,     FWHM_max,   None))        
        
        #Simple inverted gaussian or Voigt profile
        if (model_choice in ['gauss','voigt']):
            
            #Fit parameters 
            p_start.add_many(('skewA',   0.0,           False,   -1e3,1e3,None),
                             ('kurtA',   0.0,           False,   -100.,100.,None))        

            #Continuum
            for ideg in range(1,5):p_start.add_many(('c'+str(ideg)+'_pol',          0.,              False,    None,None,None)) 
        
            #Simple inverted gaussian
            #    - the main component of the model is an inverted gaussian with constant baseline
            #    - for fast rotators or sloped continuum use skewness and slope properties for the gaussian
            if (model_choice=='gauss'):
                fixed_args['fit_func']=MAIN_gauss_herm_lin 
                fixed_args['fit_func_gen']=gauss_herm_lin 
                
            #Inverted Voigt profile
            elif (model_choice=='voigt'):
                fixed_args['fit_func'] = MAIN_voigt 
                fixed_args['fit_func_gen']=voigt  
                p_start.add_many((  'a_damp',  1., True,    0.,       1e15,  None))
    
            # #Complex prior function     -> left as an example, but now that contrast is a direct parameter a function is not required anymore
            # if (fit_dic['fit_mod']=='mcmc') and (len(fit_prop_dic['prior_func'])>0):
                
            #     def global_ln_prior_func(p_step_loc,args_loc):
            #         ln_p_loc = 0.
            
            #         #Uniform prior on contrast between 0 and 1
            #         #    - contrast is defined as C = amplitude / mean 
            #         ctrst_loc =args_loc['amp_sign']*p_step_loc['amp']/p_step_loc['cont']
            #         if (ctrst_loc<0) or (ctrst_loc>1):ln_p_loc += -np.inf
    
            #         return ln_p_loc 
                
            #     fixed_args['global_ln_prior_func']=global_ln_prior_func          
    
        #-------------------------------------------------------------------------------
        #Flat continuum with 6th-order polynomial at center, then added to an inverted gaussian
        #    - polynomial is made continuous in value and derivative, at a velocity value symmetric with respect to the centroid RV
        elif (model_choice=='gauss_poly'):
            
            #Guess of distance from centroid where lobes end and continuum starts
            #    - typically about three times the FWHM
            dRV_joint=3.*FWHM_guess
            
            #Polynomial coefficients
            #    - center of polynomial continuum is at cont + a4*dx^4, and set to the maximum of the CCF:
            # a4 = (max - cont)/dx^4
            #    - we assume here that the 6th order coefficient is null to estimate the guess
            c4_pol=(max(flux_loc_fit)- fixed_args['flux_cont'])/ dRV_joint**4.        
            c6_pol=0.
            
            #Parameters
            p_start.add_many( ( 'dRV_joint',    dRV_joint,      True,   0.,None,  None),
                              ( 'c4_pol',       c4_pol,         True,   None,None,  None),
                              ( 'c6_pol',       c6_pol,         True,   None,None,  None))
    
            #Fit function
            fixed_args['fit_func']=MAIN_gauss_poly
            fixed_args['fit_func_gen']=gauss_poly
            
        #-------------------------------------------------------------------------------
        #-------------------------------------------------------------------------------
        #Gaussian whose wings are the continuum added to an inverted gaussian
        #    - positions, widths and contrast are linked 
        elif (model_choice=='dgauss'):
    
            #Amplitude ratio 
            if 'amp_l2c' in model_prop:amp_l2c = model_prop['amp_l2c'][inst][vis]['guess']
            else:amp_l2c=0.5
            
            #FWHM ratio
            if 'FWHM_l2c' in model_prop:FWHM_l2c = model_prop['FWHM_l2c'][inst][vis]['guess']
            else:FWHM_l2c=2.
                
            #Shift between the RV centroids of the two components
            if 'rv_l2c' in model_prop:rv_l2c = model_prop['rv_l2c'][inst][vis]['guess']
            else:rv_l2c=0.
    
            #Contrast guess
            amp_core=2.*(CCF_peak-fixed_args['flux_cont'])  
            ctrst_guess = -amp_core*( 1. - amp_l2c) / fixed_args['flux_cont']
    
            #-----------------------
            
            #Parameters
            p_start.add_many(( 'rv_l2c',        rv_l2c,             True ,  None ,  None,  None), 
                             ( 'amp_l2c',       amp_l2c,            True ,  0.,None,  None),
                             ( 'FWHM_l2c',      FWHM_l2c,           True ,  0.,None,  None))


            #Continuum
            for ideg in range(1,5):p_start.add_many(('c'+str(ideg)+'_pol',          0.,              False,    None,None,None)) 
        
            #Fit function
            fixed_args['fit_func']=MAIN_dgauss
            fixed_args['fit_func_gen']=dgauss

    ######################################################################################################## 

    #Fit dictionary
    fit_dic={
        'fit_mod':fit_prop_dic['fit_mod'],
        'uf_bd':{},
        'nx_fit':len(fixed_args['y_val'])
        }
    fixed_args['fit'] = {'chi2':True,'':False,'mcmc':True}[fit_prop_dic['fit_mod']]

    #Parameter initialization
    p_start = par_formatting(p_start,model_prop,line_fit_priors,fit_dic,fixed_args,inst,vis)

    #Model initialization
    #    - must be done after the final parameter initialization
    if (model_choice=='custom') and (prof_type=='DI'):
        fixed_args = init_custom_DI_prof(fixed_args,gen_dic,data_dic['DI']['system_prop'],theo_dic,star_params,p_start)
        if (not fixed_args['fit']) or (not fixed_args['var_star_grid']):
            theo_intr2loc(fixed_args['grid_dic'],fixed_args['system_prop'],fixed_args['args_exp'],fixed_args['args_exp']['ncen_bins'],fixed_args['grid_dic']['nsub_star'])         
        
    #Fit initialization
    fit_prop_dic['progress'] = True
    fit_dic['save_dir'] = fixed_args['save_dir']+'iexp'+str(fixed_args['iexp'])+'/'
    init_fit(fit_dic,fixed_args,p_start,model_par_names(),fit_prop_dic)  

    #--------------------------------------------------------------
    #Fit by chi2 minimization
    if fit_prop_dic['fit_mod']=='chi2':
        p_final = fit_minimization(ln_prob_func_lmfit,p_start,fixed_args['x_val'],fixed_args['y_val'],fixed_args['cov_val'],fixed_args['fit_func'],verbose=verbose,fixed_args=fixed_args)[2]
     
    #--------------------------------------------------------------   
    #Fit by emcmc 
    elif fit_prop_dic['fit_mod']=='mcmc': 
        
        #Calculate HDI for error definition
        #    - automatic definition of PDF resolution is used unless histogram resolution is set
        fit_dic['HDI_dbins']= {}
        fit_dic['HDI_bwf']= {}
        for param_loc in fixed_args['var_par_list']:
            if ('HDI_dbins' in fit_prop_dic) and (param_loc in fit_prop_dic['HDI_dbins']) and (inst in fit_prop_dic['HDI_dbins'][param_loc]) and (vis in fit_prop_dic['HDI_dbins'][param_loc][inst]):
                fit_dic['HDI_dbins'][param_loc]=fit_prop_dic['HDI_dbins'][param_loc][inst][vis]
            elif ('HDI_bwf' in fit_prop_dic) and (param_loc in fit_prop_dic['HDI_bwf']) and (inst in fit_prop_dic['HDI_bwf'][param_loc]) and (vis in fit_prop_dic['HDI_bwf'][param_loc][inst]):
                fit_dic['HDI_bwf'][param_loc]=fit_prop_dic['HDI_bwf'][param_loc][inst][vis]
    
        #Store options
        for key in ['nwalkers','nsteps','nburn']:fit_dic[key] = fit_prop_dic['mcmc_set'][key][inst][vis]

        #Run MCMC
        if fit_prop_dic['mcmc_run_mode']=='use':
            walker_chains=call_MCMC(gen_dic['fit_prof_nthreads'],fixed_args,fit_dic,verbose=verbose)
                           
        #Reuse MCMC
        elif fit_prop_dic['mcmc_run_mode']=='reuse':
            walker_chains=np.load(fit_dic['save_dir']+'/raw_chains_walk'+str(fit_prop_dic['mcmc_set']['nwalkers'][inst][vis])+'_steps'+str(fit_prop_dic['mcmc_set']['nsteps'][inst][vis])+fit_dic['run_name']+'.npz')['walker_chains']  
 
        #Excluding parts of the chains
        if fit_dic['exclu_walk']:
            if gen_dic['star_name'] == 'HD106315':
                wgood=np_where1D((np.min(walker_chains[:,400::,np_where1D(fixed_args['var_par_list']=='veq')],axis=1)>8.))

            walker_chains=np.take(walker_chains,wgood,axis=0)     
            fit_dic['nwalkers']=len(wgood) 
 
    
        #Processing:
        p_final,merged_chain,par_sample_sig1,par_sample=postMCMCwrapper_1(fit_dic,fixed_args,walker_chains,gen_dic['fit_prof_nthreads'],fixed_args['par_names'],verbose=verbose)

    #--------------------------------------------------------------   
    #Fixed model
    elif fit_prop_dic['fit_mod']=='': 
        p_final = p_start
  
    #Merit values     
    p_final=fit_merit(p_final,fixed_args,fit_dic,verbose)    

    ########################################################################################################    
    #Post-processing
    ########################################################################################################   
    fixed_args['fit'] = False 

    #Store outputs
    output_prop_dic['BIC']=fit_dic['BIC']    
    output_prop_dic['red_chi2']=fit_dic['red_chi2'] 
 
    #Best-fit model to the full profile (over the minimum fit range) at the observed resolution
    output_prop_dic['edge_bins'] = fixed_args['edge_bins']
    output_prop_dic['cen_bins'] = fixed_args['cen_bins']
    fixed_args['cond_def_fit']=np.repeat(True,fixed_args['ncen_bins'])    
    output_prop_dic['cond_def'] = np.ones(len(cen_bins),dtype=bool)
    output_prop_dic['flux']=fixed_args['fit_func'](p_final,None,args=fixed_args)

    #Double gaussian model: output the two components
    if (model_choice=='dgauss'):
        output_prop_dic['gauss_core'],output_prop_dic['gauss_lobe']=fixed_args['fit_func_gen'](p_final,cen_bins)[1:3] 
        if fixed_args['FWHM_inst'] is not None:
            output_prop_dic['gauss_core'] = convol_prof(output_prop_dic['gauss_core'],cen_bins,fixed_args['FWHM_inst'])  
            output_prop_dic['gauss_lobe'] = convol_prof(output_prop_dic['gauss_lobe'],cen_bins,fixed_args['FWHM_inst']) 

    #Gaussian + polynomial model: output the two components
    elif (model_choice=='gauss_poly'):
        output_prop_dic['gauss_core'],output_prop_dic['poly_lobe']=fixed_args['fit_func_gen'](p_final,cen_bins)[1:3] 
        if fixed_args['FWHM_inst'] is not None:
            output_prop_dic['gauss_core'] = convol_prof(output_prop_dic['gauss_core'],cen_bins,fixed_args['FWHM_inst'])  
            output_prop_dic['poly_lobe'] = convol_prof(output_prop_dic['poly_lobe'],cen_bins,fixed_args['FWHM_inst']) 
            
    #Custom model: output the line without continuum
    elif (model_choice=='custom'):
        output_prop_dic['core'],output_prop_dic['core_norm']=fixed_args['fit_func_gen'](p_final,cen_bins,args=fixed_args)[1:3]
        if fixed_args['FWHM_inst'] is not None:
            output_prop_dic['core'] = convol_prof(output_prop_dic['core'],cen_bins,fixed_args['FWHM_inst'])  
            output_prop_dic['core_norm'] = convol_prof(output_prop_dic['core_norm'],cen_bins,fixed_args['FWHM_inst']) 

    #---------------------------------------------------------------------------------------------------

    #Best-fit model to the full line profile, including instrumental convolution if requested
    #    - observed tables are not used anymore and are overwritten in fixed_args
    if (inst not in fit_prop_dic['best_mod_tab']):fit_prop_dic['best_mod_tab'][inst]={}
    if 'dx' not in fit_prop_dic['best_mod_tab'][inst]:dx_mod = np.median(fixed_args['dcen_bins'])
    else:dx_mod=fit_prop_dic['best_mod_tab'][inst]['dx']
    if 'min_x' not in fit_prop_dic['best_mod_tab'][inst]:min_x = fixed_args['edge_bins'][0]
    else:min_x=fit_prop_dic['best_mod_tab'][inst]['min_x']
    if 'max_x' not in fit_prop_dic['best_mod_tab'][inst]:max_x = fixed_args['edge_bins'][-1]
    else:max_x=fit_prop_dic['best_mod_tab'][inst]['max_x']
    fixed_args['ncen_bins'] =  int((max_x-min_x)/dx_mod) 
    dx_mod = (max_x-min_x)/fixed_args['ncen_bins'] 
    fixed_args['edge_bins'] = min_x+np.arange(fixed_args['ncen_bins']+1)*dx_mod    
    fixed_args['cen_bins'] = 0.5*(fixed_args['edge_bins'][0:-1]+fixed_args['edge_bins'][1::])
    fixed_args['dim_exp']= [1,fixed_args['ncen_bins']]
    fixed_args['dcen_bins'] = np.repeat(dx_mod,fixed_args['ncen_bins'])
    fixed_args['cond_def_fit']=np.repeat(True,fixed_args['ncen_bins']) 
    
    #Deactivate resampling 
    #    - model resolution can be directly adjusted
    fixed_args['resamp'] = False
    fixed_args['args_exp'] = def_st_prof_tab(None,None,None,fixed_args)
    
    #Custom model
    if (model_choice=='custom') and (prof_type=='DI'):
        fixed_args = init_custom_DI_prof(fixed_args,gen_dic,data_dic['DI']['system_prop'],theo_dic,star_params,p_final)
        theo_intr2loc(fixed_args['grid_dic'],fixed_args['system_prop'],fixed_args['args_exp'],fixed_args['args_exp']['ncen_bins'],fixed_args['grid_dic']['nsub_star']) 
            
    #Store final model
    output_prop_dic['cen_bins_HR']=fixed_args['cen_bins']       
    output_prop_dic['flux_HR']=fixed_args['fit_func'](p_final,None,args=fixed_args)

    ########################################################################################################             
    #Derived parameters
    #    - with chi2 fit: best-fit value and error of the derived parameter are defined here 
    #      with mcmc fit: the chain of the derived parameter is defined here, and its best-fit value and error are then derived in postMCMCwrapper_2()
    ########################################################################################################  
    if (fit_prop_dic['thresh_area'] is not None) or (fit_prop_dic['thresh_amp'] is not None):fit_prop_dic['deriv_prop'] += ['amp'] 
    if len(fit_prop_dic['deriv_prop'])>0:
        
        if (model_choice in ['gauss','voigt']):
            
            #Amplitude
            if (('amp' in fit_prop_dic['deriv_prop']) or ('area' in fit_prop_dic['deriv_prop'])):
                
                #Contrast is defined as C = 1 - ( CCF minimum / mean continuum flux)
                #                       C = (mean continuum flux -  CCF minimum) / mean continuum flux)        
                #                       C = -amp / cont 
                #Amplitude is defined as amp = -C*cont  
                if fit_dic['fit_mod'] in ['chi2','']:
                    p_final['amp']=fixed_args['amp_sign']*p_final['ctrst']*p_final['cont']                
                    if fit_dic['fit_mod']=='chi2':                   
                        #d[A]  = sqrt( (err(C)/cont)^2 + (err(cont)/C)^2 )   
                        if 'ctrst' in fixed_args['var_par_list']:err_ctrst= fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='ctrst')[0]]  
                        else:err_ctrst=0.
                        if 'cont' in fixed_args['var_par_list']:err_cont= fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='cont')[0]]  
                        else:err_cont=0.
                        sig_loc=np.sqrt( (err_ctrst/p_final['cont'])**2. + (err_cont/p_final['ctrst'])**2. )  
                        fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))  
                elif fit_dic['fit_mod']=='mcmc':    
                    if 'ctrst' in fixed_args['var_par_list']:ctrst_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='ctrst')[0]]
                    else:cont_chain=p_final['ctrst']
                    if 'cont' in fixed_args['var_par_list']:cont_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='cont')[0]]
                    else:cont_chain=p_final['cont']
                    chain_loc=fixed_args['amp_sign']*ctrst_chain*cont_chain
                    merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)            
                fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'amp')
                fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],'Amp')       

            #Integral under the gaussian model
            #    - formula valid if there is no asymetry to the gaussian shape:
            # A = 0.5*amp*FWHM*sqrt(pi/ln(2))
            if ('area' in fit_prop_dic['deriv_prop']) and (model_choice=='gauss') and (p_final['skewA']==0.) and (p_final['kurtA']==0.):    
                if fit_dic['fit_mod'] in ['chi2','']:
                    p_final['area']=np.abs(p_final['amp'])*p_final['FWHM']*np.sqrt(np.pi)/(2.*np.sqrt(np.log(2))) 
                    if fit_dic['fit_mod']=='chi2':                 
                        if 'amp' in fixed_args['var_par_list']:err_amp= fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='amp')[0]]  
                        else:err_amp=0.
                        if 'FWHM' in fixed_args['var_par_list']:err_FWHM= fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='FWHM')[0]]  
                        else:err_FWHM=0.
                        #d[A]  = 0.5*sqrt( (err(amp)*FWHM)^2 + (err(FWHM)*amp)^2 )*sqrt(pi/ln(2))
                        sig_loc=np.sqrt( (err_amp*p_final['FWHM'])**2. + (err_FWHM*p_final['amp'])**2. )*np.sqrt(np.pi)/(2.*np.sqrt(np.log(2)))   
                        fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))  
                elif fit_dic['fit_mod']=='mcmc': 
                    if 'amp' in fixed_args['var_par_list']:amp_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='amp')[0]]
                    else:amp_chain=p_final['amp']
                    if 'FWHM' in fixed_args['var_par_list']:FWHM_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='FWHM')[0]]
                    else:FWHM_chain=p_final['FWHM']
                    chain_loc=np.abs(amp_chain)*FWHM_chain*np.sqrt(np.pi)/(2.*np.sqrt(np.log(2))) 
                    merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)   
                fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'area')
                fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],'Area')         
    
    
        if (prof_type in ['DI','Intr']):
    
            #FWHM of the Lorentzian and Voigt profiles
            #    - see function definition
            #    - fL = a*fG/sqrt(ln(2))
            #    - fV = 0.5436*fL+ sqrt(0.2166*fL^2+fG^2)
            # for the determination of uncertainty in chi2 mode we must take derivatives with respect to a and fG 
            #      fV = (0.5436*a*fG/sln2 )+ sqrt( 0.2166*(a*fG)^2/sln2^2  + fG^2 ) 
            # with sln2 = sqrt(ln(2))
            #      fV = fG*( (0.5436*a/sln2 )+ sqrt( 0.2166*a^2/sln2^2  + 1 ) )   
            #         = fG*( (0.5436*a)+ sqrt( 0.2166*a^2 + sln2^2 ) )/sln2  
            # we define
            #      fa = ( (0.5436*a)+ sqrt( 0.2166*a^2 + sln2^2 ) )/sln2  
            #      sfa = (0.5436 + (d[ 0.2166*a^2 + sln2^2  ]/(2*sqrt( 0.2166*a^2  + sln2^2 ))))*sa/sln2     
            #          = (0.5436+ ( 0.2166*a)/sqrt( 0.2166*a^2 + sln2^2  ))*sa/sln2         
            # with fV = fG*fa independent
            #      sfV = sqrt( (fa*sfG)^2 + (fG*sfa)^2 )    
            if (model_choice=='voigt') and (('FWHM_LOR' in fit_prop_dic['deriv_prop']) or ('FWHM_voigt' in fit_prop_dic['deriv_prop'])):
                if fit_dic['fit_mod'] in ['chi2','']:
                    sln2 = np.sqrt(np.log(2.))
                    p_final['FWHM_LOR'] = p_final['a_damp']*p_final['FWHM']/sln2
                    p_final['FWHM_voigt'] = 0.5436*p_final['FWHM_LOR']+ np.sqrt(0.2166*p_final['FWHM_LOR']**2.+p_final['FWHM']**2.)                 
                    if fit_dic['fit_mod']=='chi2': 
                        if 'FWHM' in fixed_args['var_par_list']:sfwhm_gauss= fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='FWHM')[0]] 
                        else:sfwhm_gauss=0.
                        if 'a_damp' in fixed_args['var_par_list']:sa_damp= fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='a_damp')[0]] 
                        else:sa_damp=0.
                        sfwhm_lor = p_final['FWHM_LOR']*np.sqrt( ( (sa_damp/p_final['a_damp'])**2. + (sfwhm_gauss/p_final['FWHM'])**2. ) )
                        fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sfwhm_lor],[sfwhm_lor]])) 
                        fa = (0.5436*p_final['a_damp']/sln2 )+ np.sqrt( 0.2166*p_final['a_damp']**2./sln2**2.  + 1. )
                        sfa = ( 0.5436 + (0.2166*p_final['a_damp'])/np.sqrt( 0.2166*p_final['a_damp']**2.  + sln2**2. ) )*sa_damp/sln2 
                        sig_loc = np.sqrt( (fa*sfwhm_gauss)**2. + (p_final['FWHM']*sfa)**2. ) 
                        fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]])) 
                elif fit_dic['fit_mod']=='mcmc':   
                    if 'FWHM' in fixed_args['var_par_list']:fwhm_gauss_chain = merged_chain[:,np_where1D(fixed_args['var_par_list']=='FWHM')[0]]
                    else:fwhm_gauss_chain = p_final['FWHM'] 
                    if 'a_damp' in fixed_args['var_par_list']:a_damp_chain = merged_chain[:,np_where1D(fixed_args['var_par_list']=='a_damp')[0]]
                    else:a_damp_chain = p_final['a_damp']                
                    fwhm_lor_chain = a_damp_chain*fwhm_gauss_chain/np.sqrt(np.log(2.))                
                    merged_chain=np.concatenate((merged_chain,fwhm_lor_chain[:,None]),axis=1) 
                    chain_loc=0.5436*fwhm_lor_chain+ np.sqrt(0.2166*fwhm_lor_chain**2.+fwhm_gauss_chain**2.) 
                    merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)          
                fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],['FWHM_LOR','FWHM_voigt'])
                fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],['a$_\mathrm{damp}$','FWHM$_\mathrm{Voigt}$'])                
                    
    
            #True properties of the global line
            #    - mesured on the high-resolution model, after instrumental convolution if requested
            if (model_choice in ['dgauss','custom']):
                if any([par_name in fit_prop_dic['deriv_prop'] for par_name in ['true_ctrst','true_FWHM','true_amp']]):
                    if fit_dic['fit_mod'] in ['chi2','']:
                        p_final['true_ctrst'],p_final['true_FWHM'],p_final['true_amp'] = cust_mod_true_prop(p_final,output_prop_dic['cen_bins_HR'],fixed_args)  
                        if fit_dic['fit_mod']=='chi2': 
                            sig_loc=np.nan
                            fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]])) 
                            fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]])) 
                            fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]])) 
                    elif fit_dic['fit_mod']=='mcmc':           
                        p_final_loc=deepcopy(p_final)
                        fixed_args['cen_bins_HR'] = output_prop_dic['cen_bins_HR']
                        if fixed_args['nthreads']>1:chain_loc=para_cust_mod_true_prop(proc_cust_mod_true_prop,fixed_args['nthreads'],fit_dic['nsteps_pb_all'],[merged_chain],(fixed_args,p_final_loc,))                           
                        else:chain_loc=proc_cust_mod_true_prop(merged_chain,fixed_args,p_final_loc)       
                        merged_chain=np.concatenate((merged_chain,chain_loc.T),axis=1)  
                    fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],['true_ctrst','true_FWHM','true_amp',])
                    fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],['C$_\mathrm{true}$','FWHM$_\mathrm{true}$','A$_\mathrm{true}$'])
        
                #Stellar rotational velocity
                if (model_choice=='custom') and ('vsini' in fit_prop_dic['deriv_prop']):
                    if ('veq' in fixed_args['var_par_list']):iveq=np_where1D(fixed_args['var_par_list']=='veq')[0] 
                    if ('cos_istar' in fixed_args['var_par_list']):iistar = np_where1D(fixed_args['var_par_list']=='cos_istar')[0] 
                    if fit_dic['fit_mod']=='chi2':             
                        #    - vsini = veq*sin(i)
                        #      dvsini = vsini*sqrt( (dveq/veq)^2 + (dsini/sini)^2 )
                        #      d[sini] = cos(i)*di
                        sin_istar = np.sqrt(1.-p_final['cos_istar']**2.)
                        p_final['vsini'] = p_final['veq']*sin_istar
                        if ('veq' in fixed_args['var_par_list']):sig_veq = fit_dic['sig_parfinal_err']['1s'][0,iveq]
                        else:sig_veq=0.
                        if ('cos_istar' in fixed_args['var_par_list']):sig_cos_istar = fit_dic['sig_parfinal_err']['1s'][0,iistar]
                        else:sig_cos_istar=0.
                        if (sig_veq>0.) or (sig_cos_istar>0.):
                            dsini = p_final['cos_istar']*sig_cos_istar/ sin_istar
                            sig_loc = p_final['vsini']*np.sqrt( (sig_veq/p_final['veq'])**2. + (dsini/sin_istar)**2. )  
                        else:sig_loc=np.nan
                        fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))     
                        merged_chain = None
                    elif fit_dic['fit_mod']=='mcmc':                  
                        if ('veq' in fixed_args['var_par_list']):veq_chain=merged_chain[:,iveq]  
                        else:veq_chain=deepcopy(merged_chain[:,iveq])  
                        if ('cos_istar' in fixed_args['var_par_list']):cosistar_chain=merged_chain[:,iistar]  
                        else:cosistar_chain=p_final['cos_istar']
                        chain_loc = veq_chain*np.sqrt(1.-cosistar_chain*cosistar_chain)
                        merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)                 
                    fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],'vsini')            
                    fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],'v$_\mathrm{eq}$sin i$_{*}$ (km/s)')     
        
                    #Convert cos(istar[rad]) into istar[deg]
                    conv_cosistar('conv',fixed_args,fit_dic,p_final,merged_chain)
                
                
                if (model_choice=='dgauss'):
            
                    #Amplitude and contrast from continuum
                    #   C = 1 - ( CCF minimum / mean continuum flux)  
                    # see in function for the CCF minimum 
                    #   C = 1 - ( (cont + invert_amp + amp_lobe) / cont)    
                    #   C = 1 - ( (cont + invert_amp -invert_amp*amp_l2c) / cont) 
                    #   C = invert_amp*( amp_l2c-1) / cont 
                    #   C = cont_amp / cont
                    if any([par_name in fit_prop_dic['deriv_prop'] for par_name in ['cont_amp','amp']]):
                        if fit_dic['fit_mod']=='chi2':  
                            
                            #Core component amplitude    
                            #    - Ac = C*cont/( amp_l2c - 1 )
                            #      dAc = Ac*sqrt( (d(C)/C)^2 + (d(cont)/cont)^2 + (d(amp_l2c)/(amp_l2c - 1))^2 )    
                            p_final['amp'] = p_final['ctrst']*p_final['cont']/( p_final['amp_l2c'] - 1. )
                            if 'ctrst' in fixed_args['var_par_list']:err_ctrst=fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='ctrst')[0]]  
                            else:err_ctrst=0.
                            if 'cont' in fixed_args['var_par_list']:err_cont=fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='cont')[0]]  
                            else:err_cont=0.
                            if 'amp_l2c' in fixed_args['var_par_list']:err_amp_l2c=fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='amp_l2c')[0]]  
                            else:err_amp_l2c=0.            
                            sig_Ac = np.abs(p_final['amp'])*np.sqrt( (err_ctrst/p_final['ctrst'])**2. + (err_cont/p_final['cont'])**2. + (err_amp_l2c/(p_final['amp_l2c'] - 1.))**2. )
                            fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_Ac],[sig_Ac]])) 
                
                            #Amplitude from continuum 
                            #   - Acont = Ac*( amp_l2c-1) 
                            #     d[Acont] = sqrt( (d(Ac)*(amp_l2c-1))^2 + (d(amp_l2c)*invert_amp)^2 ) 
                            p_final['cont_amp'] = -fixed_args['amp_sign']*p_final['amp']*(p_final['amp_l2c']-1.) 
                            sig_cont_amp=np.sqrt( (sig_Ac*(p_final['amp_l2c']-1.))**2. + (err_amp_l2c*p_final['amp'])**2. )
                            fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_cont_amp],[sig_cont_amp]]))  
                
                        elif fit_dic['fit_mod']=='mcmc': 
                            if 'ctrst' in fixed_args['var_par_list']:ctrst_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='ctrst')[0]] 
                            else:ctrst_chain=p_final['ctrst']            
                            if 'cont' in fixed_args['var_par_list']:cont_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='cont')[0]]
                            else:cont_chain=p_final['cont']            
                            if 'amp_l2c' in fixed_args['var_par_list']:amp_l2c_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='amp_l2c')[0]]
                            else:amp_l2c_chain=p_final['amp_l2c']         
                            amp_chain = ctrst_chain*cont_chain/( amp_l2c_chain - 1. )
                            cont_amp_chain=-fixed_args['amp_sign']*amp_chain*(amp_l2c_chain-1.)   
                            merged_chain=np.concatenate((merged_chain,cont_amp_chain[:,None],amp_chain[:,None]),axis=1)   
                    
                        fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],['cont_amp','amp'])
                        fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],['Acont','Amp'])
        
        
                    #Lobe properties
                    if ('RV_lobe' in fit_prop_dic['deriv_prop']):                      
                        irv = np_where1D(fixed_args['var_par_list']=='rv')[0] 
                        if fit_dic['fit_mod']=='chi2': 
                            if 'rv_l2c' in fixed_args['var_par_list']:err_rv_l2c=fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='rv_l2c')[0]]  
                            else:err_rv_l2c=0.
                            p_final['RV_lobe']=p_final['rv'] + p_final['rv_l2c']
                            sig_loc=np.sqrt( fit_dic['sig_parfinal_err']['1s'][0,irv]**2.  + err_rv_l2c**2.  )     
                            fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))         
                        elif fit_dic['fit_mod']=='mcmc': 
                            if 'rv_l2c' in fixed_args['var_par_list']:rv_l2c_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='rv_l2c')[0]]
                            else:rv_l2c_chain=p_final['rv_l2c']
                            chain_loc = merged_chain[:,irv] + rv_l2c_chain
                            merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)
                        fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],['RV_lobe'])
                        fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],['rv$_\mathrm{lobe}$ (km s$^{-1}$)'])                    
                    
                    if ('amp_lobe' in fit_prop_dic['deriv_prop']):            
                        if fit_dic['fit_mod']=='chi2': 
                            if 'amp_l2c' in fixed_args['var_par_list']:err_amp_l2c=fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='amp_l2c')[0]]  
                            else:err_amp_l2c=0.
                            p_final['amp_lobe']=p_final['amp']*p_final['amp_l2c']
                            sig_loc=np.sqrt((sig_Ac*p_final['amp_l2c'])**2.+(err_amp_l2c*p_final['amp'])**2.)
                            fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]])) 
                        elif fit_dic['fit_mod']=='mcmc': 
                            if 'amp_l2c' in fixed_args['var_par_list']:amp_l2c_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='amp_l2c')[0]]
                            else:amp_l2c_chain=p_final['amp_l2c']
                            chain_loc=merged_chain[:,np_where1D(fixed_args['var_par_list']=='amp')[0]]*amp_l2c_chain
                            merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)
                        fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],['amp_lobe'])
                        fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],['A$_\mathrm{lobe}$'])                    
                    
                    if ('FWHM_lobe' in fit_prop_dic['deriv_prop']):                     
                        if fit_dic['fit_mod']=='chi2': 
                            if 'FWHM_l2c' in fixed_args['var_par_list']:err_FWHM_l2c=fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='FWHM_l2c')[0]]  
                            else:err_FWHM_l2c=0.
                            if 'FWHM' in fixed_args['var_par_list']:err_FWHM=fit_dic['sig_parfinal_err']['1s'][0,np_where1D(fixed_args['var_par_list']=='FWHM')[0]]  
                            else:err_FWHM=0.     
                            p_final['FWHM_lobe']=p_final['FWHM']*p_final['FWHM_l2c']
                            sig_loc=np.sqrt((err_FWHM*p_final['FWHM_l2c'])**2.+(err_FWHM_l2c*p_final['FWHM'])**2.) 
                            fit_dic['sig_parfinal_err']['1s']= np.hstack((fit_dic['sig_parfinal_err']['1s'],[[sig_loc],[sig_loc]]))   
                        elif fit_dic['fit_mod']=='mcmc': 
                            if 'FWHM_l2c' in fixed_args['var_par_list']:FWHM_l2c_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='FWHM_l2c')[0]]
                            else:FWHM_l2c_chain=p_final['FWHM_l2c']
                            if 'FWHM' in fixed_args['var_par_list']:FWHM_chain=merged_chain[:,np_where1D(fixed_args['var_par_list']=='FWHM')[0]]
                            else:FWHM_chain=p_final['FWHM']
                            chain_loc=FWHM_chain*FWHM_l2c_chain
                            merged_chain=np.concatenate((merged_chain,chain_loc[:,None]),axis=1)
                        fixed_args['var_par_list']=np.append(fixed_args['var_par_list'],['FWHM_lobe'])
                        fixed_args['var_par_names']=np.append(fixed_args['var_par_names'],['FWHM$_\mathrm{lobe}$ (km s$^{-1}$)'])
            

         
            


    ########################################################################################################   
    #Processing and storing best-fit values and confidence interval for the final parameters  
    #    - derived properties (not used in the model definitions) must have been added above before postMCMCwrapper_2() is run 
    ########################################################################################################       
    if fit_dic['fit_mod']=='mcmc':
        
        #Add new properties to relevant dictionaries
        for param_loc in fixed_args['var_par_list']:
            if ('HDI_dbins' in fit_prop_dic) and (param_loc in fit_prop_dic['HDI_dbins']) and (inst in fit_prop_dic['HDI_dbins'][param_loc]) and (vis in fit_prop_dic['HDI_dbins'][param_loc][inst]):
                fit_dic['HDI_dbins'][param_loc]=fit_prop_dic['HDI_dbins'][param_loc][inst][vis]
            elif ('HDI_bwf' in fit_prop_dic) and (param_loc in fit_prop_dic['HDI_bwf']) and (inst in fit_prop_dic['HDI_bwf'][param_loc]) and (vis in fit_prop_dic['HDI_bwf'][param_loc][inst]):
                fit_dic['HDI_bwf'][param_loc]=fit_prop_dic['HDI_bwf'][param_loc][inst][vis]        
       
        #Process
        p_final=postMCMCwrapper_2(fit_dic,fixed_args,merged_chain)

    #Storing best-fit values and selected uncertainties in output dictionary
    for key in ['cont','rv','FWHM','amp','ctrst','area','FWHM_LOR','a_damp','FWHM_voigt','rv_l2c','amp_l2c','FWHM_l2c','cont_amp','RV_lobe','amp_lobe','FWHM_lobe','true_amp','true_ctrst','true_FWHM','FWHM','veq','cos_istar','vsini','ctrst_ord0__IS__VS_','FWHM_ord0__IS__VS_']:
        
        if key in p_final:
            output_prop_dic[key]  = p_final[key]  
            
            #Variable parameter
            if (key in fixed_args['var_par_list']):
                ipar = np_where1D(fixed_args['var_par_list']==key)[0]
                
                #Errors set to percentiles-based uncertainties
                if (fit_dic['fit_mod']=='chi2') or ((fit_dic['fit_mod']=='mcmc') & (fit_prop_dic['out_err_mode']=='quant')):
                    output_prop_dic['err_'+key]= fit_dic['sig_parfinal_err']['1s'][:,ipar]  
                
                #Errors set to HDI-based uncertainties  
                elif (fit_dic['fit_mod']=='mcmc') & (fit_prop_dic['out_err_mode']=='HDI'):
                    output_prop_dic['err_'+key]= np.array([p_final[key]-fit_dic['HDI_interv'][ipar][0][0],fit_dic['HDI_interv'][ipar][-1][1]-p_final[key]])           
                
            else:output_prop_dic['err_'+key]=[0.,0.]        

    #Save derived parameters
    save_fit_results('derived',fixed_args,fit_dic,fit_dic['fit_mod'],p_final)

    #Close save file
    fit_dic['file_save'].close()

    ######################################################################################################## 
    #Criterion for line detection
    ######################################################################################################## 
    if (fit_prop_dic['thresh_area'] is not None) or (fit_prop_dic['thresh_amp'] is not None): 
        if np.sum(cond_def_cont)==0.:stop('No bin in CCF continuum')
      
        #Continuum dispersion
        disp_cont = flux_loc[cond_def_cont].std()
        
        #Area and amplitude criterion
        #    - we ensure that 
        # abs(cont -  peak) > disp            
        #      and that 
        # abs(cont -  peak)*sqrt(FWHM) > disp*sqrt(pix)
        #      which can be seen as requesting that the area of a box equivalent to the CCF is larger than the area of a box equivalent to a bin with amplitude the dispersion                
        #    - see also Allart+2017
        if (model_choice in ['gauss','voigt']) and ('amp' in output_prop_dic):
            output_prop_dic['crit_area']=np.abs(output_prop_dic['amp'])*np.sqrt(output_prop_dic['FWHM']) / (disp_cont*np.sqrt(gen_dic['pix_size_v'][inst]))            
            output_prop_dic['crit_amp']=np.abs(output_prop_dic['amp'])/disp_cont 
        if (model_choice in ['dgauss','custom']) and ('true_amp' in output_prop_dic):
            output_prop_dic['crit_area']=np.abs(output_prop_dic['true_amp'])*np.sqrt(output_prop_dic['true_FWHM']) / (disp_cont*np.sqrt(gen_dic['pix_size_v'][inst]))
            output_prop_dic['crit_amp']=np.abs(output_prop_dic['true_amp'])/disp_cont

        #---------------------------------------------------------
    
        #Force detection
        if (idx_force_det is not None):
            output_prop_dic['forced_det']=True 
            output_prop_dic['detected']=idx_force_det
        
        #Assess detection
        #    - Amplitude >= threshold x continuum dispersion
        #      this criterion includes the check that amplitude is larger than 0
        #    - Area criterion 
        else:
            output_prop_dic['forced_det']=False
            if ('crit_amp' in output_prop_dic) and ('crit_area' in output_prop_dic):  
                output_prop_dic['detected']= (output_prop_dic['crit_amp']>fit_prop_dic['thresh_amp']) & (output_prop_dic['crit_area']>fit_prop_dic['thresh_area'])            
            else:output_prop_dic['detected'] = False

    else:
        output_prop_dic['forced_det']=False
        output_prop_dic['detected'] = ''

    ######################################################################################################## 
    #Direct measurements
    ######################################################################################################## 

    #Calculation of bissector
    if ('biss' in fit_prop_dic['meas_prop']):
        biss_prop = fit_prop_dic['meas_prop']['biss']

        #Spectrum selection   
        if biss_prop['source']=='obs':
            rv_biss = deepcopy(cen_bins)
            flux_biss = deepcopy(flux_loc)
        elif biss_prop['source']=='mod':
            rv_biss = deepcopy(output_prop_dic['cen_bins_HR'])
            flux_biss = deepcopy(output_prop_dic['flux_HR'])            

        #Normalize flux with best-fit continuum        
        flux_biss /= p_final['cont']
        
        #Calculating bissector
        output_prop_dic['RV_biss'],output_prop_dic['F_biss'],output_prop_dic['RV_biss_span'],output_prop_dic['F_biss_span'],output_prop_dic['biss_span']=calc_biss(flux_biss,rv_biss,output_prop_dic['rv'],fit_prop_dic['biss_range_frame'][:,iexp],biss_prop['dF'],gen_dic['resamp_mode'],biss_prop['Cspan'])
        output_prop_dic['F_biss']*=p_final['cont']

    #Calculation of equivalent width
    if ('EW' in fit_prop_dic['meas_prop']):

        #Normalize flux with best-fit continuum      
        flux_norm,cov_norm = bind.mul_array(flux_loc,cov_loc,1./np.repeat(p_final['cont'],len(flux_loc)))

        #Equivalent width 
        #    - defined as int( x, dx*(1 - Fnorm(x))  ), and calculated here in RV space as Delta_RV - int(x , Fnorm(x)*dw)
        bd_int = fit_prop_dic['EW_range_frame'][:,iexp]
        idx_int_sign = np_where1D((edge_bins[0:-1]>=bd_int[0]) & (edge_bins[1::]<=bd_int[1]))
        if len(idx_int_sign)==0:stop('No pixels in "EW_range"')
        width_range = bd_int[1]-bd_int[0]
        int_flux,cov_int_flux = bind.resampling(bd_int,edge_bins, flux_norm, cov =  cov_norm, kind=gen_dic['resamp_mode']) 
        output_prop_dic['EW'] = width_range - int_flux[0]
        output_prop_dic['err_EW'] = np.repeat(np.sqrt(cov_int_flux[0,0]),2)

    #Calculation of mean integrated signal
    if prof_type=='Atm': 
        if ('int_sign' in fit_prop_dic['meas_prop']):
            
            #Signal in each requested range
            int_sign = []
            var_int_sign = []
            for bd_int in fit_prop_dic['int_sign_range_frame']:
                idx_int_sign = np_where1D((edge_bins[0:-1]>=bd_int[0,iexp]) & (edge_bins[1::]<=bd_int[1,iexp]))
                if len(idx_int_sign)==0:stop('No pixels in "int_sign_range"')
                width_range = bd_int[1,iexp]-bd_int[0,iexp]
                int_sign_loc,cov_int_sign_loc = bind.resampling(bd_int,edge_bins, flux_loc, cov =  cov_loc, kind=gen_dic['resamp_mode']) 
                int_sign+=[int_sign_loc[0]/width_range]
                var_int_sign+=[cov_int_sign_loc[0,0]/width_range**2.]
                
            #Global signal
            #    - calculated as the mean of the signal over the requested ranges
            output_prop_dic['int_sign'] = np.mean(int_sign)
            output_prop_dic['err_int_sign'] = np.sqrt(np.sum(var_int_sign))
            output_prop_dic['R_sign'] = fit_dic[iexp]['int_sign']/fit_dic[iexp]['err_int_sign']
            output_prop_dic['err_int_sign'] = np.repeat(output_prop_dic['err_int_sign'],2.)

    return output_prop_dic
    

'''
Sub-function to modify parameters properties using input values
'''
def par_formatting(p_start,model_prop,priors_prop,fit_dic,fixed_args,inst,vis):
  
    #Parameters are used for fitting   
    if isinstance(p_start,lmfit.parameter.Parameters):fit_used=True
    else:fit_used=False

    #Process default / additional parameters
    fixed_args['varpar_priors']={}
    for par in np.unique( list(p_start.keys()) + list(model_prop.keys())  ):    

        #Overwrite default properties 
        if (par in model_prop):
            
            #Check property is used for current instrument
            #    - this is to avoid modifying a property for other instruments than the one(s) it is set up with in model_prop
            #    - except for planet/star properties independent of a dataset (defined as 'physical')
            if (inst in par) or ('__IS' in par) or ((inst in model_prop[par]) and (vis in model_prop[par][inst])) or (('physical' in model_prop[par]) and (model_prop[par]['physical'] is True)):
     
                #Properties are fitted
                if fit_used:
                
                    #Properties depend on instrument and/or visit or is common to all
                    if (inst in model_prop[par]):
                        if (vis in model_prop[par][inst]):model_prop_par = model_prop[par][inst][vis]
                        else:model_prop_par = model_prop[par][inst]
                    else:model_prop_par = model_prop[par] 

                    #Overwrite property fields
                    if (par in p_start):
                        p_start[par].value  = model_prop_par['guess']  
                        p_start[par].vary  = model_prop[par]['vary']                   
                    
                    #Add property
                    else:  
                        p_start.add(par, model_prop_par['guess']  ,model_prop[par]['vary']  ,None , None, None)
         
                    #Value linked to other parameter
                    if ('expr' in model_prop_par):p_start[par].expr = model_prop_par['expr']  
                    
                #Fixed properties
                else:p_start[par] = model_prop[par]               
             
        #Variable parameter
        if fit_used and (par in p_start) and p_start[par].vary:
            
            #Chi2 fit
            #    - overwrite default priors
            if (fit_dic['fit_mod']=='chi2'):
                if (par in priors_prop) and (priors_prop[par]['mod']=='uf'): 
                    p_start[par].min = priors_prop[par]['low']
                    p_start[par].max = priors_prop[par]['high']

                #Change guess value if beyond prior range
                if (not np.isinf(p_start[par].min)) and (np.isinf(p_start[par].max)) and (p_start[par].value<p_start[par].min):p_start[par].value=p_start[par].min
                if (np.isinf(p_start[par].min)) and (not np.isinf(p_start[par].max)) and (p_start[par].value>p_start[par].max):p_start[par].value=p_start[par].max
                if (not np.isinf(p_start[par].min)) and (not np.isinf(p_start[par].max)) and ((p_start[par].value<p_start[par].min) or (p_start[par].value>p_start[par].max)):p_start[par].value=0.5*(p_start[par].min+p_start[par].max)

            #MCMC fit
            elif (fit_dic['fit_mod']=='mcmc'):
                
                #Range for walkers initialization
                if (par in model_prop):fit_dic['uf_bd'][par]=model_prop_par['bd']
                else:
                    uf_bd=[-1e5,1e5]
                    if (not np.isinf(p_start[par].min)):uf_bd[0]=p_start[par].min
                    if (not np.isinf(p_start[par].max)):uf_bd[1]=p_start[par].max
                    fit_dic['uf_bd'][par]=uf_bd
                
                #Priors
                if (par in priors_prop):
                    fixed_args['varpar_priors'][par] = priors_prop[par]                          
                else:
                    varpar_priors=[-1e5,1e5]
                    if (not np.isinf(p_start[par].min)):varpar_priors[0]=p_start[par].min
                    if (not np.isinf(p_start[par].max)):varpar_priors[1]=p_start[par].max                
                    fixed_args['varpar_priors'][par]={'mod':'uf','low':varpar_priors[0],'high':varpar_priors[1]}

                #Change guess value if beyond prior range
                if ((p_start[par].value<fixed_args['varpar_priors'][par]['low']) or (p_start[par].value>fixed_args['varpar_priors'][par]['high'])):p_start[par].value=0.5*(fixed_args['varpar_priors'][par]['low']+fixed_args['varpar_priors'][par]['high'])


    #---------------------------------------------------

    #Associate the correct input names for model functions with instrument and visit dependence
    #     - properties must be defined in p_start as 'propN__ISx_VSy'
    #  + N is the degree of the coefficient (if relevant)
    #  + x is the id of the instrument
    #  + y is the id of the visit associated with the instrument
    #     - x and y can be set to '_' so that the coefficient is common to all instruments and/or visits
    fixed_args['name_prop2input'] = {}
    fixed_args['coeff_ord2name'] = {}
    fixed_args['linevar_par'] = {}

    #Retrieve all root name parameters with instrument/visit dependence, possibly with several orders (polynomial degree, or list)
    par_list=[]
    root_par_list=[]
    fixed_args['genpar_instvis']  = {}
    if 'inst_list' not in fixed_args:fixed_args['inst_list']=[inst]
    if 'inst_vis_list' not in fixed_args:fixed_args['inst_vis_list']={inst:[vis]}
    for par in p_start:
 
        #Parameter depend on instrument and visit
        if ('__IS') and ('_VS') in par:
            root_par = par.split('__IS')[0]
            inst_vis_par = par.split('__IS')[1]
            inst_par  = inst_vis_par.split('_VS')[0]
            vis_par  = inst_vis_par.split('_VS')[1]              
            if root_par not in fixed_args['genpar_instvis'] :fixed_args['genpar_instvis'][root_par] = {}          
            if inst_par not in fixed_args['genpar_instvis'][root_par]:fixed_args['genpar_instvis'][root_par][inst_par]=[]
            if vis_par not in fixed_args['genpar_instvis'][root_par][inst_par]:fixed_args['genpar_instvis'][root_par][inst_par]+=[vis_par]                  
            root_par_list+=[root_par]            
            par_list+=[par]

            #Parameter vary as polynomial of spatial stellar coordinate
            if ('_ord' in par):
                gen_root_par = par.split('_ord')[0] 
                
                #Define parameter for current instrument and visit (if specified) or all instruments and visits (if undefined) 
                if inst_par in fixed_args['inst_list']:inst_list = [inst_par]
                elif inst_par=='_':inst_list = fixed_args['inst_list'] 
                for inst_loc in inst_list:
                    if inst_loc not in fixed_args['coeff_ord2name']:fixed_args['coeff_ord2name'][inst_loc] = {}
                    if vis_par in fixed_args['inst_vis_list'][inst_loc]:vis_list = [vis_par]
                    elif vis_par=='_':vis_list = fixed_args['inst_vis_list'][inst_loc]              
                    for vis_loc in vis_list:
                        if vis_loc not in fixed_args['coeff_ord2name'][inst_loc]:fixed_args['coeff_ord2name'][inst_loc][vis_loc] = {}                
                        if gen_root_par not in fixed_args['coeff_ord2name'][inst_loc][vis_loc]:fixed_args['coeff_ord2name'][inst_loc][vis_loc][gen_root_par]={}
    
                        #Identify stellar line properties with polynomial spatial dependence 
                        if gen_root_par in ['ctrst','FWHM','amp_l2c','rv_l2c','FWHM_l2c','a_damp','rv_line']:
                            if inst_loc not in fixed_args['linevar_par']:fixed_args['linevar_par'][inst_loc]={}
                            if vis_loc not in fixed_args['linevar_par'][inst_loc]:fixed_args['linevar_par'][inst_loc][vis_loc]=[]
                            if gen_root_par not in fixed_args['linevar_par'][inst_loc][vis_loc]:fixed_args['linevar_par'][inst_loc][vis_loc]+=[gen_root_par]                     

    #Process parameters with dependence on instrument/visit
    for root_par in np.unique(root_par_list):

        #Parameter is associated with order coefficient
        if ('_ord' in root_par):
            deg_coeff=int(root_par[-1])
            gen_root_par = root_par.split('_ord')[0]  
        else:deg_coeff=None
         
        #Property common to all instruments is also common to all visits
        #    - we associate the property for current instrument and visit to the property common to all instruments and visits
        if any([root_par+'__IS__' in str_loc for str_loc in par_list]):
            for inst in fixed_args['inst_list']:
                for vis in fixed_args['inst_vis_list'][inst]:
                    par_input = root_par+'__IS'+inst+'_VS'+vis                    
                    fixed_args['name_prop2input'][par_input] = root_par+'__IS__VS_'
                    if deg_coeff is not None:
                        fixed_args['coeff_ord2name'][inst][vis][gen_root_par][deg_coeff]=root_par+'__IS__VS_' 
            
        #Property is specific to a given instrument
        else:

            #Process all fitted instruments associated with the property
            for inst in fixed_args['inst_list']:
                if (inst in fixed_args['genpar_instvis'][root_par]):

                    #Property is common to all visits of current instrument
                    #    - we associate the property for current instrument and visit to the value specific to this instrument, common to all visits
                    if any([root_par+'__IS'+inst+'_VS_' in str_loc for str_loc in par_list]): 
                        for vis in fixed_args['inst_vis_list'][inst]:
                            par_input = root_par+'__IS'+inst+'_VS'+vis 
                            fixed_args['name_prop2input'][par_input] = root_par+'__IS'+inst+'_VS_'  
                            if deg_coeff is not None:fixed_args['coeff_ord2name'][inst][vis][gen_root_par][deg_coeff]=root_par+'__IS'+inst+'_VS_'      
        
                    #Property is specific to the visit
                    #    - we associate the property for current instrument and visit to the value specific to this instrument, and this visit
                    else:
                        
                        #Process all fitted visits associated with the property
                        for vis in fixed_args['inst_vis_list'][inst]:
                            if (vis in fixed_args['genpar_instvis'][root_par][inst]):   
                                par_input = root_par+'__IS'+inst+'_VS'+vis 
                                fixed_args['name_prop2input'][par_input] = root_par+'__IS'+inst+'_VS'+vis   
                                if deg_coeff is not None:fixed_args['coeff_ord2name'][inst][vis][gen_root_par][deg_coeff]=root_par+'__IS'+inst+'_VS'+vis  

    return p_start























'''
Bissector calculation
    - for a line profile provided in RV space as input
'''
def calc_biss(Fnorm_in,RV_tab_in,RV_min,max_rv_range,dF_grid,resamp_mode,Cspan):

    #Reduction to maximum range    
    cond_range = (RV_tab_in>max_rv_range[0]) & (RV_tab_in<max_rv_range[-1])
    Fnorm = Fnorm_in[cond_range]
    RV_tab = RV_tab_in[cond_range]

    #Blue and red wings of input line
    idx_blue=np_where1D(RV_tab<RV_min)
    idx_red=np_where1D(RV_tab>=RV_min)
  
    #Calculation range
    #    - between the maxima on each side of the line
    max_blue=max(Fnorm[idx_blue])
    max_red=max(Fnorm[idx_red])
    idx_max_blue=np_where1D(Fnorm==max_blue)[0]
    idx_max_red=np_where1D(Fnorm==max_red)[0]
    idx_min = closest(RV_tab,RV_min)
    idx_biss_blue=range(idx_max_blue,idx_min+1)
    idx_biss_red=range(idx_min,idx_max_red+1)
    Fnorm_biss = Fnorm[idx_max_blue:idx_max_red+1]
     
    #RV = f(flux) in blue and red wings
    RV_biss_blue=RV_tab[idx_biss_blue]
    F_biss_blue=Fnorm[idx_biss_blue]
    idx_sort = np.argsort(F_biss_blue)
    RV_biss_blue = RV_biss_blue[idx_sort]
    F_biss_blue = F_biss_blue[idx_sort]
    Fedge_biss_blue = 0.5*(F_biss_blue[0:-1]+F_biss_blue[1:])
    Fedge_biss_blue = np.concatenate(( [F_biss_blue[0] - 0.5*(F_biss_blue[1]-F_biss_blue[0])] , Fedge_biss_blue , [F_biss_blue[-1]+ 0.5*(F_biss_blue[-1]-F_biss_blue[-2])] ))
    RV_biss_red=RV_tab[idx_biss_red]
    F_biss_red=Fnorm[idx_biss_red]
    idx_sort = np.argsort(F_biss_red)
    RV_biss_red = RV_biss_red[idx_sort]
    F_biss_red = F_biss_red[idx_sort]
    Fedge_biss_red = 0.5*(F_biss_red[0:-1]+F_biss_red[1:])
    Fedge_biss_red = np.concatenate(([ F_biss_red[0] - 0.5*(F_biss_red[1]-F_biss_red[0])] , Fedge_biss_red , [F_biss_red[-1]+ 0.5*(F_biss_red[-1]-F_biss_red[-2])] ))
    
    #Resampling RV versus flux
    minF_biss=np.min(Fnorm_biss)
    maxF_biss=np.max(Fnorm_biss)
    nF_biss=int((maxF_biss-minF_biss)/dF_grid)
    F_biss_edgeHR=minF_biss+dF_grid*np.arange(nF_biss+1)
    RV_biss_blue_HR = bind.resampling(F_biss_edgeHR,Fedge_biss_blue,RV_biss_blue,kind = resamp_mode)            
    RV_biss_red_HR = bind.resampling(F_biss_edgeHR,Fedge_biss_red,RV_biss_red,kind = resamp_mode)

    #Bissector
    RV_biss_HR=0.5*(RV_biss_blue_HR+RV_biss_red_HR)
    F_biss_HR = 0.5*(F_biss_edgeHR[0:-1]+F_biss_edgeHR[1:])
    
    #Bissector span
    #    - difference in velocity between minimum and region of larger variation in the series
    if Cspan is not None:idx_span = closest(flux_biss_HR,1. - Cspan*(1.-minF_biss) )
    else:idx_span = np.argmax(np.abs(RV_biss_HR-RV_min))
    RV_biss_span = RV_biss_HR[idx_span]
    F_biss_span = F_biss_HR[idx_span]
    bis_span = RV_biss_span - RV_min
                                  
    return RV_biss_HR,F_biss_HR,RV_biss_span,F_biss_span,bis_span



'''
General oversampling of model line profiles
    function must return several arguments (or a list of one) with the model as first argument
    model is calculated over a continuous table to allow for convolution and use of covariance matrix (fitted pixels are accounted for directly in the minimization routine)
'''
def gen_over_func(param_in,func_nam,args):
    args_exp = deepcopy(args['args_exp'])

    #In case param_in is defined as a Parameters structure, retrieve values and define dictionary
    if isinstance(param_in,lmfit.parameter.Parameters):
        param={}
        for par in param_in:param[par]=param_in[par].value
    else:param=param_in   
    
    #Line profile model
    sp_line_model = func_nam(param,args_exp['cen_bins'],args)[0]

    #Add offset if required
    if ('offset' in param):sp_line_model+=param['offset']    

    #Conversion and resampling 
    mod_prof = conv_st_prof_tab(None,None,None,args,args_exp,sp_line_model,args['FWHM_inst'])    

    return mod_prof




'''
Convolution by instrumental LSF   
    - bins must have the same size in a given table
'''
def convol_prof(prof_in,cen_bins,FWHM):

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



'''
Dispatching function
'''
def dispatch_func_prof(func_name):
    return {'gauss':gauss_intr_prof,
            'dgauss':dgauss,
            'voigt':voigt
        }[func_name]


'''
Polynomial continuum for stellar line models
'''
def pol_cont(cen_bins_ref,args,param):
    return (1. + param['c1_pol']*cen_bins_ref + param['c2_pol']*cen_bins_ref**2.  + param['c3_pol']*cen_bins_ref**3. + param['c4_pol']*cen_bins_ref**4. ) 

'''
Gaussian with hermitian polynom term for skewness/kurtosis and with linear trend
'''
def MAIN_gauss_herm_lin(param,RV,args=None):
    ymodel=gen_over_func(param,gauss_herm_lin,args)
    return ymodel

'''
Gaussian with hermitian polynom term for skewness/kurtosis and with linear trend
'''
def gauss_herm_lin(param,x,args=None):
    x_tab = (2.*np.sqrt(2.*np.log(2.)))*(x-param['rv' ])/param['FWHM'] 
    
    #Skewness and kurtosis
    skew1 = param['skewA']
    kurt1 = param['kurtA']
    factor=1.
    if skew1!=0. or kurt1!=0.:  
        c0 = np.sqrt(6.)/4.
        c1 = -np.sqrt(3.)
        c2 = -np.sqrt(6.)
        c3 = 2./np.sqrt(3.)
        c4 = np.sqrt(6.)/3.        
    if skew1!=0.:factor+=skew1*(c1*x_tab+c3*x_tab**3.)
    if kurt1!=0.:factor+=kurt1*(c0+c2*x_tab**2.+c4*x_tab**4.)   
 
    #Skewd gaussian profile
    sk_gauss = 1.- factor*param['ctrst']*np.exp(-0.5*x_tab**2.)
     
    #Continuum        
    cont_pol = param['cont']*pol_cont(x,args,param)    

    return sk_gauss*cont_pol , cont_pol

'''
Voigt profile main function
'''
def MAIN_voigt(param,RV,args=None):
    ymodel=gen_over_func(param,voigt,args)
    return ymodel

'''
Voigt profile with linear trend
    - the FWHM_gauss parameter is that of the gaussian component (factorless):
 G(rv) = exp(- (rv-rv0)^2/ ( FWHM_gauss/(2 sqrt(ln(2))) )^2  )             
 G(rv) = exp(- (rv-rv0)^2/ 2*( FWHM_gauss/(2 sqrt(2ln(2))) )^2  )  
 G(rv) = exp(- (rv-rv0)^2/ 2*sig^2  ) 
      with sig = FWHM_gauss/(2 sqrt(2ln(2)))          
    - the Voigt profile is expressed in terms of the Faddeeva function (factorless):
 V(rv) = Real(w(x))      
      with x = ((rv-rv0) + i*gamma)/(sqrt(2)*sig)
      we use the damping coefficient 
 a = gamma/(sqrt(2)*sigma) = 2*gamma*sqrt(ln(2))/FWHM_gauss 
      which can be expressed as a function of FWHM_LOR = 2*gamma, so that 
 a = sqrt(ln(2))*FWHM_LOR/FWHM_gauss 
      and
 x = (rv-rv0)/(sqrt(2)*sig) + i*a 
   = 2*sqrt(ln(2))*(rv-rv0)/FWHM_gauss + i*a
    - the full-width at half maximum of the Voigt profile approximates as:
 fwhm_voigt = (0.5436*fwhm_lor+ np.sqrt(0.2166*pow(fhwhm_lor,2.)+pow(fwhm_gauss,2.)) )
      from J.J.Olivero and R.L. Longbothum in Empirical fits to the Voigt line width: A brief review, JQSRT 17, P233
'''
def voigt(param,x,args=None):
    z_tab = 2.*np.sqrt(np.log(2.))*(x - param['rv' ])/param['FWHM'] +  1j*param['a_damp']
    
    #Voigt profile
    voigt_peak = special.wofz(1j*param['a_damp']).real
    voigt_mod = 1. - (param['ctrst']/voigt_peak)*special.wofz(z_tab).real
                 
    #Continuum        
    cont_pol = param['cont']*pol_cont(x,args,param)   

    return voigt_mod*cont_pol , cont_pol


'''
Simple gaussian profile 
'''
def gauss_intr_prof(param,rv_tab,args=None):
    return [param['cont']*(1.-param['ctrst']*np.exp(-np.power( 2.*np.sqrt(np.log(2.))*(rv_tab-param['rv'])/param['FWHM']  ,2.  )))] 


'''
Function returning the measured FWHM and contrast of intrinsic stellar profiles with gaussian shape
    - if the intrinsic line is a gaussian, its FWHM is sqrt( FWHM_intr^2 + FWHM_instr^2 )
      the area of the convolution is also equal to the product of the area of the two convolved profiles, providing an analytical value for the contrast of the convolved profile:
 + gaussian kernel : fkern(rv) = (1/( sqrt(2*pi)*sig_k ))*exp( -rv^2 / (2*sig_k^2) ) 
                     sig_k  = FWHM_inst/( 2.*np.sqrt(2*np.log(2.)) )
 + profile : fprof(rv) = A*(1/( sqrt(2*pi)*sig_p ))*exp( -rv^2 / (2*sig_p^2) )    
                     sig_p  = FWHM_exp/( 2.*np.sqrt(2*np.log(2.)) )
                     A = ctrst_exp*sqrt(pi)*FWHM_exp/( 2.*np.sqrt(np.log(2.)) )
 + amplitude of the convolution profile: A/sqrt(2*pi*( sig_k^2 + sig_p^2 ))
                                       = ctrst_exp*sqrt(pi)*FWHM_exp/(  2.*np.sqrt(np.log(2.)) * sqrt(2*pi*( sig_k^2 + sig_p^2 ))  )
                                       = ctrst_exp*FWHM_exp/(  2.*np.sqrt(2*np.log(2.)) * sqrt(sig_k^2 + sig_p^2 )  )
 sig_k^2 + sig_p^2 = (FWHM_inst^2 + FWHM_exp^2) /( 2.*np.sqrt(2*np.log(2.)) )^2
                                       = ctrst_exp*FWHM_exp/( sqrt(FWHM_inst^2 + FWHM_exp^2)  )
    - this is only valid for analytical profiles
      for numerical profiles, the model is the sum of intrinsic profiles over the planet-occulted surface and may be further broadened by the local surface rv field                            
'''
def gauss_intr_prop(ctrst_intr,FWHM_intr,FWHM_inst):
    FWHM_meas = np.sqrt(FWHM_intr**2. + FWHM_inst**2.) 
    ctrst_meas = ctrst_intr*FWHM_intr/FWHM_meas
    return ctrst_meas,FWHM_meas








'''
Gaussian with flat continuum and polynomial for sidelobes
'''
def MAIN_gauss_poly(param,RV,args=None):
    ymodel=gen_over_func(param,gauss_poly,args)
    return ymodel

def gauss_poly(param,RV,args=None):
 
    #Gaussian with baseline set to continuum value
    cen_RV = param['rv']    #center
    y_gausscore=1.-param['ctrst']*np.exp(-np.power( 2.*np.sqrt(np.log(2.))*(RV-cen_RV)/param['FWHM']  ,2.  )) 
    ymodel = y_gausscore*param['cont']
    
    #Polynomial
    #    - P(x) = a0 + a1*(x-x0) + a2*(x-x0)^2 + a3*(x-x0)^3 + a4*(x-x0)^4 + a6*(x-x0)^6
    #      P must by symmetric, and have continuity of value and derivative in x=x0+-dx
    #      this must be true in particular if x0 = 0, and we thus solve:
    # P(x) = P(-x)
    # P(dx) = cont
    # P'(dx) = 0 
    #    - the continuum is set to its constant value beyond the continuity points
    c4_pol = param['c4_pol']                          #4th order coefficient
    c6_pol = param['c6_pol']                          #6th order coefficient
    dRV_joint=param['dRV_joint']                      #continuity points
    RV_joint_high = cen_RV + dRV_joint       
    RV_joint_low  = cen_RV - dRV_joint
    cond_lobes = (RV >= RV_joint_low) & (RV <= RV_joint_high) 
    y_polylobe = np.repeat(1.,len(RV))
    y_polylobe[cond_lobes] *= c4_pol*dRV_joint**4. + 2.*c6_pol*dRV_joint**6. - dRV_joint**2.*np.power(RV[cond_lobes]-cen_RV,2.)*(2.*c4_pol + 3.*c6_pol*dRV_joint**2.) + c4_pol*np.power(RV[cond_lobes]-cen_RV,4.) + c6_pol*np.power(RV[cond_lobes]-cen_RV,6.)
    ymodel[cond_lobes]*=y_polylobe[cond_lobes]    
    
    return ymodel, param['cont']*y_gausscore, param['cont']*y_polylobe
                 





'''
Model with gaussian for continuum, and inverted gaussian for the core
    - the gaussian profiles are centered but their amplitude and width can be fixed or independant
    - the model is defined as :
 CCF = cont + amp_core*exp(f1) + amp_lobe*exp(f2)
 CCF = cont + amp_core*( exp(f1) - amp_l2c*exp(f2) ) 
      we define a contrast parameter as the relative flux between the continuum and the CCF minimum :
 C = (cont - (cont + amp_core - amp_core*amp_l2c) ) / cont 
 C = -amp_core*( 1 - amp_l2c) / cont
 amp_core = -C*cont/( 1 - amp_l2c)
 amp_lobe = -amp_l2c*amp_core = amp_l2c*C*cont/( 1 - amp_l2c)
      thus the model can be expressed as :    
 CCF = cont - C*cont*( exp(f1) - amp_l2c*exp(f2) )/( 1 - amp_l2c)
 CCF = cont*( 1 - C*( exp(f1) - amp_l2c*exp(f2) )/( 1 - amp_l2c) ) 
'''
def MAIN_dgauss(param,RV,args=None):
    ymodel=gen_over_func(param,dgauss,args)
    return ymodel

def dgauss(param,rv_tab,args=None):
 
    #Reduced contrast
    ctrst_red = param['ctrst']/( 1. - param['amp_l2c'])
     
    #Inverted gaussian core
    y_gausscore=param['cont']*(1. - ctrst_red*np.exp(-np.power(2.*np.sqrt(np.log(2.))*(rv_tab-param['rv'])/param['FWHM'],2))) 
    
    #Gaussian lobes
    cen_RV_lobes=param['rv']+param['rv_l2c']   
    FWHM_lobes = (param['FWHM'])*(param['FWHM_l2c'])   
    y_gausslobes=param['cont']*(1. + ctrst_red*param['amp_l2c']*np.exp(-np.power(2.*np.sqrt(np.log(2.))*(rv_tab-cen_RV_lobes)/FWHM_lobes,2)))     
    
    #Complete model
    ymodel = y_gausscore + y_gausslobes - param['cont']

    #Output models of each gaussian component superimposed to the continuum
    return ymodel , y_gausscore, y_gausslobes






'''
Function returning the measured FWHM and contrast of stellar profiles with custom shape
    - we return the 'true' contrast and FWM on the measured-like model
    - we calculate the model on a HR table 
    - we calculate with the derived model the contrast counted from the estimated continuum (
 + for double-gaussian profiles, the peaks of the lobes are the closest estimate to the true stellar continuum
    - we calculate the FWHM by finding the points on the blue side at half-maximum, and on the red side
'''
def cust_mod_true_prop(param,velccf_HR,args):

    #HR model
    CCF_HR_intr = args['fit_func_gen'](param,velccf_HR,args=args)[0]
    
    #Instrumental convolution 
    if args['FWHM_inst'] is not None:CCF_HR = convol_prof(CCF_HR_intr,velccf_HR,args['FWHM_inst'])  
    else:CCF_HR=CCF_HR_intr
    
    #Approximation of continuum
    max_CCF = np.max(CCF_HR)
    
    #CCF inverted peak
    min_CCF = np.min(CCF_HR)
    
    #Estimates of amplitude and contrast
    true_amp=max_CCF - min_CCF 
    true_ctrst=1. - (min_CCF/max_CCF) 
    
    #Blue side and red side of model CCF
    blueRV_HR=np_where1D(velccf_HR<=param['rv'])
    if len(blueRV_HR)==0:blueRV_HR=[0]
    redRV_HR=np_where1D(velccf_HR>=param['rv'])
    if len(redRV_HR)==0:redRV_HR=[-1]
    
    #Model pixels closest to half-maximum (mid-point between maximum and minimum of model) on each wing
    idx_blue_WHM=closest(CCF_HR[blueRV_HR],0.5*(np.max(CCF_HR[blueRV_HR])+min_CCF))  
    idx_red_WHM=closest(CCF_HR[redRV_HR],0.5*(np.max(CCF_HR[redRV_HR])+min_CCF))   
    
    #Estimate of FWHM defined as sum of HWHM in each wing
    true_FWHM = (param['rv'] - velccf_HR[blueRV_HR][idx_blue_WHM]) + (velccf_HR[redRV_HR][idx_red_WHM] - param['rv'])       
           
    return true_ctrst,true_FWHM,true_amp    

def proc_cust_mod_true_prop(merged_chain_proc,args,param_loc):
    nsamp=len(merged_chain_proc[:,0])    
    param_proc = deepcopy(param_loc)
    chain_loc_proc=np.empty([3,0],dtype=float) 
    for istep in range(nsamp): 
        for ipar,par in enumerate(args['var_par_list']):param_proc[par]=merged_chain_proc[istep,ipar]     
        true_ctrst_step,true_FWHM_step,true_amp_step=cust_mod_true_prop(param_proc,args['cen_bins_HR'],args)  
        chain_loc_proc=np.append(chain_loc_proc,[[true_ctrst_step],[true_FWHM_step],[true_amp_step]],axis=1) 
    return chain_loc_proc
    
def para_cust_mod_true_prop(func_input,nthreads,n_elem,y_inputs,common_args):   
    pool_proc = Pool(processes=nthreads)   #cannot be passed through lmfit
    ind_chunk_list=init_parallel_func(nthreads,n_elem)
    chunked_args=[(y_inputs[0][ind_chunk[0]:ind_chunk[1],:],)+common_args for ind_chunk in ind_chunk_list]	
    all_results=tuple(tab for tab in pool_proc.starmap(func_input,chunked_args))				
    y_output=np.concatenate(tuple(all_results[i] for i in range(nthreads)),axis=1)
    pool_proc.close()
    pool_proc.join() 
    return y_output

#Local macroturbulence broadening functions
#    - formulation with anisotropic gaussian from Takeda & UeNo 2017 or with radial-tangential model (Gray 1975, 2005)
def calc_macro_ker_rt(rv_mac_kernel,param,cos_th,sin_th):
    if cos_th>0:term_R=(param['A_R']/(np.sqrt(np.pi)*param['ksi_R']*cos_th))*np.exp(-rv_mac_kernel**2./  ((param['ksi_R']*cos_th)**2.)  )   #Otherwise exp yield 0
    else:term_R=0.
    if sin_th>0:term_T=(param['A_T']/(np.sqrt(np.pi)*param['ksi_T']*sin_th))*np.exp(-rv_mac_kernel**2./  ((param['ksi_T']*sin_th)**2.)  )   #Otherwise exp yield 0 
    else:term_T=0.
    macro_ker_sub=term_R+term_T    
    return macro_ker_sub

def calc_macro_ker_anigauss(rv_mac_kernel,param,cos_th,sin_th):
    macro_ker_sub=np.exp(  -rv_mac_kernel**2./  ((param['eta_R']*cos_th)**2.  +  (param['eta_T']*sin_th)**2.)  )
    return macro_ker_sub






#Cumulate local profiles from each cell of the stellar disk
def coadd_loc_line_prof(rv_shift_grid,icell_list,Fsurf_grid_spec,flux_intr_grid,mu_grid,param,args):
    flux_DI_sum=[]
    for isub,(icell,rv_shift,flux_intr_cell,mu_cell) in enumerate(zip(icell_list,rv_shift_grid,flux_intr_grid,mu_grid)):
        flux_DI_sum+=[calc_loc_line_prof(icell,rv_shift,Fsurf_grid_spec[isub],flux_intr_cell,mu_cell,args,param)]
    return flux_DI_sum

def para_coadd_loc_line_prof(func_input,nthreads,n_elem,y_inputs,common_args):  
    pool_proc = Pool(processes=nthreads)   #cannot be passed through lmfit
    ind_chunk_list=init_parallel_func(nthreads,n_elem)
    chunked_args=[tuple(y_inputs[i][ind_chunk[0]:ind_chunk[1]] for i in range(len(y_inputs)))+common_args for ind_chunk in ind_chunk_list]	
    all_results=tuple(tab for tab in pool_proc.starmap(func_input,chunked_args))	
    y_output=np.concatenate(tuple(all_results[i] for i in range(nthreads)))
    pool_proc.close()
    pool_proc.join()    
    return y_output

#Calculate local line profile
def calc_loc_line_prof(icell,rv_shift,Fsurf_cell_spec,flux_loc_cell,mu_cell,args,param):

    #Calculation of analytical intrinsic line profile
    #    - model is always calculated in RV space, and later converted back to wavelength space if relevant
    #    - the model is directly calculated over the RV table at its requested position, rather than being pre-calculated and shifted
    if args['mode']=='ana':
        input_cell = {'cont':1. , 'rv':rv_shift }
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
        
    #Shift stored intrinsic profile to RV of local stellar surface element        
    #    - see align_data for details 
    #    - models are pre-calculated and then shifted, since the line profile and the shift are independent
    elif (args['mode'] in ['theo','Intrbin']):
        if ('spec' in args['type']):edge_bins_rest = args['edge_bins_intr']*spec_dopshift(-rv_shift) 
        elif (args['type']=='CCF'):edge_bins_rest = args['edge_bins_intr'] + rv_shift
        flux_intr = bind.resampling(args['edge_bins'],edge_bins_rest,flux_loc_cell, kind=args['resamp_mode'])            
    
    #Continuum scaling into local line profiles
    #    - default continuum of intrinsic profiles is set to 1 so that it can be modulated chromatically here
    flux_loc=flux_intr*Fsurf_cell_spec
    
    return flux_loc






'''
Attribution of original / resampled spectral grid for line profile calculations
'''
def def_st_prof_tab(inst,vis,isub,args):
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

'''
Activation of spectral conversion and resampling 
'''
def cond_conv_st_prof_tab(rv_osamp_line_mod,fixed_args,data_type):
    
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
        if fixed_args['line_trans'] is None:stop('Define "line_trans" to fit spectral data with "glob_mod"')
    else:fixed_args['spec2rv'] = False
    
    return None

'''
Resampled spectral table for model line profile
    - theoretical profiles are directly calculated at the requested resolution, measured profiles are extracted at their native resolution
'''
def resamp_model_st_prof_tab(inst,vis,isub,fixed_args,gen_dic,nexp,rv_osamp_line_mod):
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

'''
Effective instrumental convolution
    - in RV space for analytical model, in wavelength space for theoretical profiles
    - disabled if measured profiles as used as proxy for the intrinsic profiles
'''
def ref_inst_convol(inst,fixed_args,cen_bins):
    
    #Reference point
    if (fixed_args['mode']=='ana') or fixed_args['spec2rv']:fixed_args['ref_conv'] = c_light
    elif fixed_args['mode']=='theo':fixed_args['ref_conv'] = cen_bins[int(len(cen_bins)/2)]    
    
    #Instrumental response 
    if (fixed_args['mode']=='Intrbin'):FWHM_inst = None
    else:FWHM_inst = return_FWHM_inst(inst,fixed_args['ref_conv'])      
    
    return FWHM_inst

'''
Convolution, spectral conversion and resampling of spectral grid for line profile calculations
'''
def conv_st_prof_tab(inst,vis,isub,args,args_exp,line_mod_in,FWHM_inst):

    #Convolve with instrumental response 
    #    - performed on table with constant bin size
    if FWHM_inst is None:line_mod_out = line_mod_in
    else:line_mod_out =  convol_prof( line_mod_in,args_exp['cen_bins'],FWHM_inst)

    #Convert table from RV to spectral space if relevant
    if args['spec2rv']:
        args_exp['edge_bins'] = args['line_trans']*spec_dopshift(-args_exp['edge_bins'])  
        args_exp['cen_bins'] = args['line_trans']*spec_dopshift(-args_exp['cen_bins'])  

    #Resample model on observed table if oversampling
    if args['resamp']:
        if inst is None:edge_bins_mod_out = args['edge_bins']
        else:edge_bins_mod_out = args['edge_bins'][inst][vis][isub]
        line_mod_out = bind.resampling(edge_bins_mod_out,args_exp['edge_bins'],line_mod_out, kind=args['resamp_mode'])       

    return line_mod_out


'''
Initialization of stellar grid properties
    - for disk-integrated and local stellar grid
    - fit parameters are initialized to default stellar properties
'''
def init_custom_DI_par(fixed_args,gen_dic,system_prop,star_params,params,RV_guess_tab):  

    #Stellar grid properties
    #    - all stellar properties are initialized to default stellar values
    #      those defined as variable properties through the settings will be overwritten in 'par_formatting'
    for key,bd_min,bd_max in zip(['veq','alpha_rot','beta_rot','c1_CB','c2_CB','c3_CB','cos_istar','f_GD','beta_GD','Tpole','A_R','ksi_R','A_T','ksi_T','eta_R','eta_T'],
                                 [0.,    None,       None,      None,   None,   None,   -1.,        0.,     0.,      0.,     0. , 0.,     0. ,  0.,     0. ,    0.],
                                 [1e4,   None,       None,      None,   None,   None,    1.,        1.,     1.,      1e5,    1.,  1e5,    1e5,  1e5,    100.,   100.]):
        if key in star_params:params.add_many((key, star_params[key],   False,    bd_min,bd_max,None))

    #Properties specific to disk-integrated profiles
    if fixed_args['DI_grid']:
        for ideg in range(1,5):params.add_many(('LD_u'+str(ideg),  system_prop['achrom']['LD_u'+str(ideg)][0],              False,    None,None,None))   

    #Line model properties
    params.add_many(('cont',      fixed_args['flux_cont'],                          False,    None,             None,               None))
    params.add_many(('rv',        RV_guess_tab[0],                                  False,    RV_guess_tab[1],  RV_guess_tab[2],    None)) 
    for ideg in range(1,5):params.add_many(('c'+str(ideg)+'_pol',          0.,              False,    None,None,None)) 

    return fixed_args,params    



'''
Initializing intrinsic / local profiles for global grid
'''
def init_custom_DI_prof(fixed_args,gen_dic,system_prop,theo_dic,star_params,param_in):   
    if isinstance(param_in,lmfit.parameter.Parameters):params={par:param_in[par].value for par in param_in}
    else:params=deepcopy(param_in) 

    #Store various properties potentially overwritten in the fitting procedure
    fixed_args['resamp_mode'] = gen_dic['resamp_mode'] 
    fixed_args['system_prop'] = deepcopy(system_prop) 
    fixed_args['star_params'] = deepcopy(star_params)  
    fixed_args['grid_dic'] = deepcopy(theo_dic)

    #------------------------------------------------------------------------
    #Identify variable stellar grid
    #    - condition is that one property controlling the stellar grid is variable OR has a value different from the one in theo_dic use to define the default stellar grid 
    #    - if condition is true, the sky-projected stellar grid and corresponding broadband emission are re-calculated at each step of the minimization for the step parameters
    #    - this option cannot be used with chromatic intensity variations
    #    - this option is not required if only veq is varying 
    #------------------------------------------------------------------------
    stargrid_prop = ['veq','alpha_rot','beta_rot','c1_CB','c2_CB','c3_CB','cos_istar','f_GD','beta_GD','Tpole','A_R','ksi_R','A_T','ksi_T','eta_R','eta_T']
    
    #Forward mode
    #    - only if properties differ from default ones
    if (not fixed_args['fit']):
        up_grid = False
        for par in params:
            if ((par in stargrid_prop) and (params[par] != star_params[par])) or ((par in ['LD_u1','LD_u2','LD_u3','LD_u4']) and (params[par] != system_prop['achrom'][par][0])):up_grid = True
        if up_grid:up_model_star(fixed_args,params)

    #Fit mode
    #    - grid will be updated in custom_DI_prof() if one of the properties vary
    else:
        
        #Update condition
        fixed_args['var_star_grid']=False
        for par in params:
            if ((par in stargrid_prop) and (params[par] != star_params[par])) or ((par in ['LD_u1','LD_u2','LD_u3','LD_u4']) and (params[par] != system_prop['achrom'][par][0])):fixed_args['var_star_grid']=True

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





'''
Routine to initialize intrinsic profile grid and properties
    - called upon initialization of the disk-integrated stellar profile grid, or in fit mode if the stellar grid or the intrinsic profile grid vary
'''
def init_st_intr_prof(args,grid_dic,param):

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
        for par_loc in args['linevar_par'][args['inst']][args['vis']]:     
            args['coeff_line'][par_loc] = polycoeff_def(param,args['coeff_ord2name'][args['inst']][args['vis']][par_loc])
            args['input_cell_all'][par_loc] = calc_polymodu(args['pol_mode'],args['coeff_line'][par_loc],grid_dic['linevar_coord_grid']) 

    #Measured intrinsic profiles
    #    - attributing to each stellar cell the measured profile with closest coordinate
    elif (args['mode']=='Intrbin'):
        ibin_to_cell = closest_arr(args['cen_dim_Intrbin'], grid_dic[args['coord_line']])
        args['flux_intr_grid'] = np.zeros([grid_dic['nsub_star'],len(args['flux_Intrbin'][0])],dtype=float)
        for icell in range(grid_dic['nsub_star']):args['flux_intr_grid'][icell]=args['flux_Intrbin'][ibin_to_cell[icell]] 
            
    return None

'''
Custom discretized profile
    - the disk-integrated profile is built upon a discretized grid of the stellar surface to allow accounting for any type of intensity and velocity field
'''
def MAIN_custom_DI_prof(param,RV,args=None):
    ymodel=gen_over_func(param,custom_DI_prof,args)
    return ymodel

'''
Function to calculate the disk-integrated stellar profile
    - separated from init_custom_DI_prof() so that custom_DI_prof() can be used in fitting routines with varying parameters
'''
def custom_DI_prof(param,x,args=None):

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
            theo_intr2loc(args['grid_dic'],args['system_prop'],args,args['ncen_bins'],args['grid_dic']['nsub_star'])     
    
        #--------------------------------------------------------------------------------        
        #Updating intrinsic line profiles
        #    - if the line properties or the stellar grid cells to which they are attributed to vary
        if args['var_line'] or args['var_star_grid']:
            init_st_intr_prof(args,args['grid_dic'],param)
            
    #--------------------------------------------------------------------------------
    #Radial velocities of the stellar surface (km/s)
    #    - an offset is allowed to account for the star/input frame velocity when the model is used on raw data 
    #--------------------------------------------------------------------------------
    rv_shift_grid = calc_RVrot(args['grid_dic']['x_st_sky'],args['grid_dic']['y_st'],args['star_params']['istar_rad'],param) + param['rv']
    cb_band = calc_CB_RV(LD_coeff_func(args['system_prop']['achrom'],0),args['system_prop']['achrom']['LD'][0],param['c1_CB'], param['c2_CB'], param['c3_CB'],param)
    if np.max(np.abs(cb_band))!=0.:rv_shift_grid += np_poly(cb_band)(args['grid_dic']['mu']).flatten()

    #--------------------------------------------------------------------------------        
    #Coadding local line profiles over stellar disk
    #--------------------------------------------------------------------------------
    icell_list = np.arange(args['grid_dic']['nsub_star'])
    
    #Multithreading
    #    - disabled with theoretical profiles, there seems to be an incompatibility with sme
    if (args['nthreads']>1) and (args['mode']!='theo'):
        flux_DI_sum=para_coadd_loc_line_prof(coadd_loc_line_prof,args['nthreads'],args['grid_dic']['nsub_star'],[rv_shift_grid,icell_list,args['Fsurf_grid_spec'],args['flux_intr_grid'],args['grid_dic']['mu']],(param,args,))                           
    
    #Direct call
    else:flux_DI_sum=coadd_loc_line_prof(rv_shift_grid,icell_list,args['Fsurf_grid_spec'],args['flux_intr_grid'],args['grid_dic']['mu'],param,args)
    
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






























'''    
Repeat boundary value in case of bad interpolation on the edge
'''
def repeat_interp(interp_prof):
    
    #In case of bad interpolation on the edge, 
    wok=np.invert(np.isnan(interp_prof))  #well-defined points
    wlowok=min(np.where(wok==True)[0])    #first well-defined bin
    if wlowok>0:
        interp_prof[0:wlowok]=interp_prof[wlowok]
    whighok=max(np.where(wok==True)[0])
    if whighok<len(interp_prof)-1:
        interp_prof[whighok:len(interp_prof)]=interp_prof[whighok]
        
    return interp_prof





'''
Fit to the standard deviation as a function of bin size
'''
def binned_stddev_fit(param,x):

    #Uncorrelated standard deviation
    sig_uncorr=(param['sig_uncorr'].value)/np.sqrt(x)

    #Correlated standard deviation
    sig_corr=(param['sig_corr'].value)

    #Global standard deviation with bin size
    sig_bin=1./np.power( np.power(1./sig_uncorr,4.) + np.power(1./sig_corr,4.) ,1./4.)

    return sig_bin
    










'''
Sub-function to calculate theoretical properties of the planet occulted-regions
    - calculatd from all and each transiting planet
'''
def calc_plocc_prop(system_param,gen_dic,theo_dic,coord_dic,inst,vis,data_dic,calc_pl_atm=False):

    print('   > Calculating properties of planet-occulted regions')    
    if (gen_dic['calc_theoPlOcc']):
        print('         Calculating data')

        #Theoretical RV of the planet occulted-regions
        #    - calculated for the nominal and broadband planet properties 
        #    - for the nominal properties we retrieve the range of some properties covered by the planet during each exposures
        #    - chromatic transit required if local profiles are in spectral mode  
        params = deepcopy(system_param['star'])
        params.update({'rv':0.,'cont':1.})
        par_list=['rv','CB_RV','mu','lat','lon','x_st','y_st','SpSstar','xp_abs','r_proj']
        key_chrom = ['achrom']
        if ('spec' in data_dic[inst][vis]['type']) and ('chrom' in data_dic['DI']['system_prop']):key_chrom+=['chrom']
        plocc_prop = sub_calc_plocc_prop(key_chrom,{},par_list,data_dic[inst][vis]['transit_pl'],system_param,theo_dic,data_dic['DI']['system_prop'],params,coord_dic[inst][vis],gen_dic[inst][vis]['idx_in'],False,out_ranges=True)

        #Save properties
        np.savez_compressed(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis,data=plocc_prop,allow_pickle=True)    

    else:
        check_data({'path':gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis})        

    return None

  

'''
Sub-function to calculate theoretical properties of spots
'''
def calc_spots_prop(gen_dic,star_params,theo_dic,inst,data_dic):
    print('   > Calculating properties of spots')    
    if star_params['f_GD']>0.:stop('Spot processing undefined for oblate stars')

    if (gen_dic['calc_theo_spots']):
        print('         Calculating data')


        
        #Process spots in each requested visit
        for vis in np.intersect1d(theo_dic['spots_prop'][inst],data_dic[inst]['visit_list']):
            for spot in theo_dic['spots_prop'][inst][vis]:
                spot_prop = theo_dic['spots_prop'][inst][vis][spot]
    

# faire une boucle d'oversampling et dedans, pour chaque expo, boucler sur les spots

               # calc_spot_region_prop(bjd_osamp,spot_prop,theo_dic['x_st'],theo_dic['y_st'],theo_dic['z_st'],theo_dic['Ssub_Sstar'],theo_dic['r2_st_sky'],par_star,LD_law,ld_coeff)
    

    
    ##boucler sur tous les spots d'abord
    ##et ensuite boucler sur oversamp 
    
    
                # #Coordinates of spot contour in 'spot rest frame'
                # #    - Xsp axis is the star-spot center axis
                # #      Ysp axis is the perpendicular in the plane parallel to the stellar equatorial plane, in the direction of the stellar rotation
                # #      Zsp completes the referential
                # nlimb = 501
                # th_limb = 2*np.pi*np.arange(nlimb)/(nlimb-1.)
                # x_sp = np.repeat(np.cos(ang_sp_rad) , nlimb)
                # y_sp = np.sin( ang_sp_rad )*np.cos(th_limb)
                # z_sp = np.sin( ang_sp_rad )*np.sin(th_limb)
    



###attention inclure ce calcul dans une sous-routine d'oversampling


        ####  je pense que la meilleure approche c de faire comme pour les contacts du transit oblate
        ####  je definis l'enveloppe HR du spot dans le ref ou il est aligne avec la LOS. Depend que de sa taille
        ## ensuite je shifte les coord avec l'offset en phase (definir pr chaque spot un T0 comme les TTV)
        ## ensuite je shifte l'offset en latitude
        ## ensuite je peux identifier quelles cellules dans le ref stellaire (spin axis dans plan ciel) sont occultees
        ## avec ca je peux utiliser mes routines deja existantes pour calculer les props des spots (RV et Ftot surtout)
        
        ## dans le custom model, je ferai pareil : dans une expo donnee je sais ou est le spot, j'identifie ses cellules, et au lieu de la CCF normale
        ## je peux y mettre une CCF * facteur de scaling du flux (qui serait une autre des props des spots)

### refechir a implementer spots avec transits: je prends les cells occultee par une planete, je checke si elles sont dans tous les spots, et si oui je change
### le flux de ces cellules 


        stop('spots')

    
    return None












'''
Calculate theoretical properties of the average stellar surface occulted during an observed or theoretical exposure
    - we normalize all quantities by the flux emitted by the occulted regions
    - all positions are in units of Rstar 
'''
def sub_calc_plocc_prop(key_chrom,args,par_list_gen,transit_pl,system_param,theo_dic,system_prop_in,param,coord_pl_in,iexp_list,cond_fit,out_ranges=False,Ftot_star=False):
    system_prop = deepcopy(system_prop_in)
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
    for subkey_chrom in key_chrom:surf_prop_dic[subkey_chrom] = {}
    if 'line_prof' in par_list_in:
        for subkey_chrom in key_chrom:surf_prop_dic[subkey_chrom]['line_prof']=np.zeros([args['ncen_bins'],n_exp],dtype=float)

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
        if Ftot_star:surf_prop_dic[subkey_chrom]['Ftot_star']=np.zeros([system_prop[subkey_chrom]['nw'],n_exp])*np.nan 
        cb_band_dic[subkey_chrom]={}  
        if ('CB_RV' in par_list) or ('c0_CB' in par_list):     
            surf_prop_dic[subkey_chrom]['c0_CB']=np.zeros(system_prop[subkey_chrom]['nw'])*np.nan
            for iband in range(system_prop[subkey_chrom]['nw']):
                cb_band_dic[subkey_chrom][iband] = calc_CB_RV(LD_coeff_func(system_prop[subkey_chrom],iband),system_prop[subkey_chrom]['LD'][iband],par_star['c1_CB'],par_star['c2_CB'],par_star['c3_CB'],par_star) 
                surf_prop_dic[subkey_chrom]['c0_CB'][iband]=cb_band_dic[subkey_chrom][iband][0] 
        else:
            for iband in range(system_prop[subkey_chrom]['nw']):cb_band_dic[subkey_chrom][iband] = None
    if 'rv' in par_list_in:par_list+=['rv']  #must be placed after all other RV contributions

    #Condition for spot calculation
    #TBD
    cond_spots = np.zeros(n_exp,dtype=bool)    
    
    range_par_list=[]
    n_osamp_exp_all = np.repeat(1,n_exp)
    if (len(theo_dic['d_oversamp'])>0) and out_ranges:range_par_list = np.intersect1d(['mu','lat','lon','x_st','y_st','xp_abs','r_proj'],par_list)
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
    cond_iexp_proc = cond_spots|cond_transit
    for i_in,(iexp,n_osamp_exp) in enumerate(zip(iexp_list,n_osamp_exp_all)):
        transit_pl_exp = np.array(transit_pl)[cond_transit_all[i_in]]
        
        #Initialize averaged and range values
        Focc_star={}
        sum_prop_dic={}
        coord_reg_dic={}
        range_dic={}
        line_occ_HP={}
        for subkey_chrom in key_chrom:
            Focc_star[subkey_chrom]=np.zeros(system_prop[subkey_chrom]['nw'],dtype=float)            
            sum_prop_dic[subkey_chrom]={}
            coord_reg_dic[subkey_chrom]={}
            range_dic[subkey_chrom]={}
            line_occ_HP[subkey_chrom]={}
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
                    
            #Line profile can be calculated over each stellar cell only in achromatic / closest-achromatic mode 
            if ('line_prof' in par_list_in):line_occ_HP[subkey_chrom] = np.repeat(theo_dic['precision'],system_prop[subkey_chrom]['nw'])
            else:line_occ_HP[subkey_chrom] = np.repeat('',system_prop[subkey_chrom]['nw'])  
            
        #Theoretical properties from regions occulted by each planet, at exposure center    
        if cond_iexp_proc[i_in]:        
            x_oversamp_pl={}
            y_oversamp_pl={}
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
               
            #Loop on oversampled exposure positions
            #    - after x_oversamp_pl has been defined for all planets
            #    - if oversampling is not active a single central position is processed
            #    - we neglect the potential chromatic variations of the planet radius and corresponding grid 
            #    - if at least one of the processed planet is transiting, or if spots are accounted for
            n_osamp_exp_eff = 0
            for iosamp in range(n_osamp_exp):
                pl_proc={subkey_chrom:{iband:[] for iband in range(system_prop[subkey_chrom]['nw'])} for subkey_chrom in key_chrom}
                for pl_loc in transit_pl_exp:   
                    
                    #Frame conversion of planet coordinates from the classical frame perpendicular to the LOS, to the 'inclined star' frame
                    x_st_sky_pos,y_st_sky_pos,_=conv_Losframe_to_inclinedStarFrame(lambda_rad_pl[pl_loc],x_oversamp_pl[pl_loc][iosamp],y_oversamp_pl[pl_loc][iosamp],None)      
    
                    #Largest possible square grid enclosing the planet shifted to current planet position     
                    x_st_sky_max = x_st_sky_pos+theo_dic['x_st_sky_grid_pl'][pl_loc]
                    y_st_sky_max = y_st_sky_pos+theo_dic['y_st_sky_grid_pl'][pl_loc]
                    
                    #Calculating properties
                    cond_occ = False
                    for subkey_chrom in key_chrom:
                        for iband in range(system_prop[subkey_chrom]['nw']):
                            Focc_star[subkey_chrom][iband],cond_occ = calc_occ_region_prop(line_occ_HP[subkey_chrom][iband],cond_occ,iband,args,system_prop[subkey_chrom],iosamp,pl_loc,pl_proc[subkey_chrom][iband],theo_dic['Ssub_Sstar_pl'][pl_loc],x_st_sky_max,y_st_sky_max,system_prop[subkey_chrom]['cond_in_RpRs'][pl_loc][iband],par_list,par_star,theo_dic['Istar_norm_'+subkey_chrom],\
                                                                                  x_oversamp_pl,y_oversamp_pl,lambda_rad_pl,par_star,sum_prop_dic[subkey_chrom][pl_loc],coord_reg_dic[subkey_chrom][pl_loc],range_dic[subkey_chrom][pl_loc],range_par_list,Focc_star[subkey_chrom][iband],line_occ_HP[subkey_chrom][iband],star_params,cb_band_dic[subkey_chrom][iband],theo_dic,cond_fit)
         
                            #Cumulate line profile from planet-occulted cells
                            #    - in high-precision mode there is a single subkey_chrom and achromatic band, but several planets may have been processed
                            if ('line_prof' in par_list_in):
                                if (theo_dic['precision']=='low'):surf_prop_dic[subkey_chrom][pl_loc]['rv_broad'][iband,i_in] = np.max([coord_reg_dic[subkey_chrom][pl_loc]['rv_broad'][iband],surf_prop_dic[subkey_chrom][pl_loc]['rv_broad'][iband,i_in]])
                                elif (theo_dic['precision']=='high'):surf_prop_dic[subkey_chrom]['line_prof'][:,i_in]+=sum_prop_dic[subkey_chrom][pl_loc]['line_prof']
                    
                #Star was effectively occulted at oversampled position
                if cond_occ:
                    n_osamp_exp_eff+=1
                    
                    #Calculate line profile from planet-occulted region 
                    if ('line_prof' in par_list_in) and  (theo_dic['precision']=='medium'):
                        idx_w = {'achrom':range(system_prop['achrom']['nw'])}
                        if ('chrom' in key_chrom):idx_w['chrom'] = range(system_prop['chrom']['nw'])
                        surf_prop_dic[key_chrom[-1]]['line_prof'][:,i_in]+=intr2plocc_prof(args,transit_pl_exp,coord_reg_dic,idx_w,system_prop,key_chrom,par_star,theo_dic)
           
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
                
                            #Total occulted surface ratio
                            #    - calculated per band so that the chromatic dependence of the radius can be accounted for also at the stellar limbs, which we cannot do with the input chromatic RpRs
                            if par_loc=='SpSstar':surf_prop_dic[subkey_chrom][pl_loc]['SpSstar'][:,i_in] = sum_prop_dic[subkey_chrom][pl_loc]['SpSstar']/n_osamp_exp_eff
                                                    
                            #Average intensity from planet-occulted region
                            elif par_loc=='Ftot':
                                surf_prop_dic[subkey_chrom][pl_loc]['Ftot'][:,i_in] = sum_prop_dic[subkey_chrom][pl_loc]['Ftot']/sum_prop_dic[subkey_chrom][pl_loc]['nocc']
                     
                            #Other surface properties
                            else:surf_prop_dic[subkey_chrom][pl_loc][par_loc][:,i_in] = sum_prop_dic[subkey_chrom][pl_loc][par_loc]/sum_prop_dic[subkey_chrom][pl_loc]['Ftot']
   
                            #Range of values covered during exposures    
                            if out_ranges and (par_loc in range_par_list):
                                surf_prop_dic[subkey_chrom][pl_loc][par_loc+'_range'][:,i_in,:] = range_dic[subkey_chrom][pl_loc][par_loc+'_range']
                                   
            #Normalized stellar flux after occultation by all planets
            #    - the intensity from each cell is calculated in the same way as that of the full pre-calculated stellar grid
            if Ftot_star:
                for subkey_chrom in key_chrom:
                    surf_prop_dic[subkey_chrom]['Ftot_star'][:,i_in] = 1.
                    if n_osamp_exp_eff>0:
                        surf_prop_dic[subkey_chrom]['Ftot_star'][:,i_in] -= Focc_star[subkey_chrom]/(n_osamp_exp_eff*theo_dic['Ftot_star_'+subkey_chrom][iband])
           
            #Planet-occulted line profile from current exposure
            if ('line_prof' in par_list_in) and (n_osamp_exp_eff>0):
         
                #Profile from averaged properties over exposures
                if (theo_dic['precision']=='low'): 
                    idx_w = {'achrom':(range(system_prop['achrom']['nw']),i_in)}
                    if ('chrom' in key_chrom):idx_w['chrom'] = (range(system_prop['chrom']['nw']),i_in)          
                    surf_prop_dic[key_chrom[-1]]['line_prof'][:,i_in]=intr2plocc_prof(args,transit_pl_exp,surf_prop_dic,idx_w,system_prop,key_chrom,par_star,theo_dic)
                    
                #Averaged profiles behind all occulted regions during exposure   
                #    - the weighing by stellar intensity is naturally included when applying flux scaling 
                elif (theo_dic['precision'] in ['medium','high']): 
                    surf_prop_dic[key_chrom[-1]]['line_prof'][:,i_in]/=n_osamp_exp_eff

                    #Normalization into intrinsic profile
                    #    - profiles used to tile the planet-occulted regions have mean unity, and are then scaled by the cell achromatic flux
                    #      we normalize by the total planet-occulted flux
                    #    - high-precision profils is achromatic
                    #    - not required for low- and medium-precision because intrinsic profiles are not scaled to local flux upon calculation in intr2plocc_prof()
                    if (theo_dic['precision']=='high') and args['conv2intr']:
                        Focc_star_achrom=Focc_star[key_chrom[-1]]/n_osamp_exp_eff
                        surf_prop_dic[key_chrom[-1]]['line_prof'][:,i_in] /=Focc_star_achrom                         

    ### end of exposure            
 
    #Output properties in chromatic mode if calculated in closest-achromatic mode
    if ('line_prof' in par_list_in) and switch_chrom:surf_prop_dic = {'chrom':surf_prop_dic['achrom']}

    return surf_prop_dic

'''
Calculation of planet-occulted line profile
    - profiles can be theoretical (calculated with a stellar atmospheric model), measured, or calculated analytically (in RV or wavelength space, but over a single line) 
    - chromatic mode: planet-to-star radius ratio and/or stellar intensy are chromatic (spectral mode only)
      closest-achromatic mode: profile width is smaller than the typical scale of chromatic variations (spectral mode only)
      achromatic mode: white values are used (default in CCF mode, optional in spectral mode)
    - high precision: intrinsic spectra are summed over each occulted cells
                      this option is only possible in chromatic / closest-achromatic  mode (intrinsic spectra cannot be summed over individual cells for each chromatic band, since different parts of the spectra are affected differently)
      medium precision: intrinsic profiles are defined for each occulted region, based on the average region properties, and cumulated
                        in chromatic mode each profile is respectively scaled and shifted using the chromatic RV and flux scaling table from the region 
      low precision: intrinsic profiles are defined for each exposure, based on the average exposure properties
                     in chromatic mode each profile is respectively scaled and shifted using the chromatic RV and flux scaling table from the exposure 
    - when several planets are transiting the properties are averaged over the complementary regions they occult, in particular the flux scaling, so that the final profile cumulated over all planets should have the flux corresponding to the summed planet-to-star surface ratios 
'''
def intr2plocc_prof(args,transit_pl,coord_dic,idx_w,system_prop,key_chrom,param,theo_dic):
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
            
                #Shift profile to average position for current planet
                #    - 'conv2intr' is True when the routine is used to calculate model intrinsic profiles (flux_sc_spec is then 1) to be compared with measured intrinsic profiles
                #       since the latter were corrected for chromatic deviations upon extraction, the present model profiles are only shifted with the achromatic planet-occulted rv 
                #    - otherwise a chromatic shift is applied
                if args['conv2intr']:dic_rv = {'achrom':{pl_loc:{'rv': np.reshape(coord_dic['achrom'][pl_loc]['rv'][idx_w['achrom']],[system_prop['achrom']['nw'],1]) }}}
                else:dic_rv = {chrom_calc:{pl_loc:{'rv': np.reshape(coord_dic[chrom_calc][pl_loc]['rv'][idx_w[chrom_calc]],[system_prop[chrom_calc]['nw'],1]) }}}
                surf_shifts,surf_shifts_edge = def_surf_shift('theo',dic_rv,0,data_loc,pl_loc,args['type'],system_prop,[1,ncen_bins_Intr],1,ncen_bins_Intr) 
                if surf_shifts_edge is not None:surf_shifts_edge*=-1.
                data_av_pl=align_data(data_loc,args['type'],1,args['dim_exp'],args['resamp_mode'],data_av_pl['cen_bins'],data_av_pl['edge_bins'],-surf_shifts,rv_shift_edge = surf_shifts_edge, nocov = True)
               
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






'''
Calculation of stellar surface properties behind the planet-occulted regions
'''
def calc_occ_region_prop(line_occ_HP_band,cond_occ,iband,args,system_prop,idx,pl_loc,pl_proc_band,Ssub_Sstar,x_st_sky_max,y_st_sky_max,cond_in_RpRs,par_list,param,Istar_norm_band,x_pos_pl,y_pos_pl,lambda_rad_pl,par_star,sum_prop_dic_pl,\
                         coord_reg_dic_pl,range_reg_pl,range_par_list,Focc_star_band,line_occ_HP,star_params,cb_band,theo_dic,cond_fit):

    #Reduce maximum square planet grid to size of planet in current band
    coord_grid = {}
    coord_grid['x_st_sky']=x_st_sky_max[cond_in_RpRs] 
    coord_grid['y_st_sky']=y_st_sky_max[cond_in_RpRs]   

    #Coordinates of stellar occulting cells in the sky-projected star rest frame
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
                x_st_sky_prev,y_st_sky_prev,_=conv_Losframe_to_inclinedStarFrame(lambda_rad_pl[pl_prev],x_pos_pl[pl_prev][idx],y_pos_pl[pl_prev][idx],None)

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
        _,_,mu_grid_star,_,Fsurf_grid_star,Ftot_star,_ = calc_Isurf_grid([iband],coord_grid['nsub_star'],system_prop,coord_grid,star_params,Ssub_Sstar,Istar_norm = Istar_norm_band)
        coord_grid['mu'] = mu_grid_star[:,0]

        #Scale continuum level
        Fsurf_grid_star*=param['cont']
        Ftot_star*=param['cont']

        #Flux and number of cells occulted from all planets, cumulated over oversampled positions
        Focc_star_band+= Ftot_star[0]   
        sum_prop_dic_pl['nocc']+=coord_grid['nsub_star']
    
        #--------------------------------

        #Co-adding properties from current region to the cumulated values over oversampled planet positions 
        sum_region_prop(line_occ_HP_band,iband,args,system_prop,par_list,Fsurf_grid_star[:,0],coord_grid,Ssub_Sstar,cb_band,range_par_list,range_reg_pl,sum_prop_dic_pl,coord_reg_dic_pl,par_star,lambda_rad_pl[pl_loc],theo_dic,param,cond_fit)

    return Focc_star_band,cond_occ

'''
Averaged values behind the planet (unormalized)
    - the flux emitted by a local element writes: 
dF[nu] =   I[nu](cos(theta)) * dA N(dA).N(LOS)
      with dF[nu] emitted in the direction of the LOS, N(LOS)
          dA = Rstar^2 sin(th)*dth*dphi the spherical surface element at the surface of the star, N(dA) its normal
      by definition N(dA).N(LOS) = cos(theta) = mu
dF[nu] =   I[nu](cos(theta)) * Rstar^2 sin(th)*dth*dphi cos(theta)
      which can also write, with dS = dxdy = dA*cos(theta) the projection of dA onto the plane of sky (where XY is the plane perpendicular to the LOS (Z))
dF[nu] =   I[nu](cos(theta)) * dS[theta]
      here the fluxes are normalized by the stellar surface, ie dS_norm[theta] = d_grid^2/(pi*Rstar^2), since d_grid is defined in units of Rstar

    - the total flux emitted by the star in the direction of the LOS is then
Fstar[nu] = int( phi:0,2pi; theta:0,pi/2 [ I[nu](cos(theta)) * sin(th)*dth*dphi cos(th) ] )/pi
        mu = cos(th) and dmu = -sin(th)dth
Fstar[nu] = 2*int( mu:0,1; [ I[nu](mu) mu*dmu  ] )
            = 2*I0[nu]*int( mu:0,1; [ LD(mu)*mu*dmu  ] )
            = 2*I0[nu]*Int0
    - if there is no limb-darkening:
Int0 = 1/2 and Fstar[nu] = I0[nu]      

    - in the non-oversampled case :
+ we add the values, even if each index is only updated once, so that the subroutine can be used directly with oversampling
+ all tables have been initialized to 0
    - in the oversampled case :
+ average values are co-added over regions oversampling the total region occulted during an exposure
+ see explanations in the non-oversampling section for the flux emitted by a surface element:
        dF(mu) =   I[nu](mu) * Ssub_Sstar = I[nu](xy) * Ssub_Sstar
            - here we must consider the flux that is emitted by a surface element during the time it is occulted by the planet:
        dF_occ(xy) = dF(xy) * Tocc(xy)
              if we assume the planet velocity is constant during the exposure, and it has no latitudinal motion, then 
        dF_occ(xy) = dF(xy) * docc(xy)/v_pl
              where docc is the distance between the planet centers from the first to the last moment it occults the surface element
        docc(xy) = sqrt( Rp^2 - y_grid^2 ) = docc(y)
+ the weighted mean of a quantity V during an exposure would thus be:
        <A> = sum( each xy_grid occulted during the exposure, V(xy)*dF_occ(xy) ) / sum( each xy_grid occulted during the exposure, dF_occ(xy) )   
        <A> = sum( xy_grid, V(xy)*I[nu](xy)*Ssub_Sstar*docc(y)/v_pl ) / sum( xy_grid, I[nu](xy)*Ssub_Sstar*docc(y)/v_pl )   
        <A> = sum( xy_grid, V(xy)*I[nu](xy)*docc(y) ) / sum( xy_grid, I[nu](xy)*docc(y) )   
+ instead of discretizing the exact surface occulted by the planet during a full exposure, we place the planet
and its grid at consecutive positions during the exposure. Between two consecutive positions separated by 'd_oversamp_exp',
the planet spent a time t_oversamp_exp = d_oversamp_exp/v_pl. If a surface element is occulted by the planet during N(xy) consecutive positions, then
we can write  the total occultation time as Tocc(xy) = N(xy)*t_oversamp_exp, and dF_occ(xy) = dF(xy)*N(xy)*d_oversamp_exp/v_pl
      the weighted mean of a quantity V during an exposure is then:
        <A> = sum( xy_grid, V(xy)*dF_occ(xy) ) / sum( xy_grid, dF_occ(xy) )              
        <A> = sum( xy_grid , V(xy)*dF(xy)*N(xy)*d_oversamp_exp/v_pl ) / sum( xy_grid , dF(xy)*N(xy)*d_oversamp_exp/v_pl )   
        <A> = sum( xy_grid , V(xy)*dF(xy)*N(xy) ) / sum( xy_grid , dF(xy)*N(xy) )  
      ie that we 'add' successively the N(xy) times a given surface element flux was occulted
      the normalization factor corresponds to Ftot_oversamp
      to ensure that this approximation is good, N(xy) must be high enough, ie t_oversamp_exp and d_oversamp_exp small enough 
+ note that Ssub_Rstar2 is normalized by Rs^2, since d_grid is defined from Rp/Rs
    - latitude and longitude (degrees)
      X and Y positions in star frame (units of Rstar)
      planet-to-star surface ratio, not limb-darkening weighted
    - in case of oversampling we update values cumulated during an exposure through every passage through the function
'''
def sum_region_prop(line_occ_HP_band,iband,args,system_prop,par_list,Fsurf_grid_band,coord_grid,Ssub_Sstar,cb_band,range_par_list,range_reg_pl,sum_prop_dic_pl,coord_reg_dic_pl,par_star,lambda_rad_pl_loc,theo_dic,param,cond_fit):
    
    #Distance from projected orbital normal in the sky plane, in absolute value
    if ('xp_abs' in par_list) or (('coord_line' in args) and (args['coord_line']=='xp_abs')):coord_grid['xp_abs'] = conv_inclinedStarFrame_to_Losframe(lambda_rad_pl_loc,coord_grid['x_st_sky'],coord_grid['y_st_sky'],coord_grid['z_st_sky'])[0]  

    #Sky-projected distance from star center
    if ('r_proj' in par_list) or (('coord_line' in args) and (args['coord_line']=='r_proj')):coord_grid['r_proj'] = np.sqrt(coord_grid['r2_st_sky'])                   

    #Processing requested properties
    for par_loc in par_list:
     
        #Occultation ratio
        #    - ratio = Sp/S* = sum(x,sp(x))/(pi*Rstar^2) = sum(x,dp(x)^2)/(pi*Rstar^2) = sum(x,d_surfloc(x)*Rstar^2)/(pi*Rstar^2) = sum(x,d_surfloc(x))/pi
        #      since we use a square grid, sum(x,d_surfloc(x)) = nx*d_surfloc
        if par_loc=='SpSstar':
            sum_prop_dic_pl[par_loc][iband]+=Ssub_Sstar*coord_grid['nsub_star']
            coord_reg_dic_pl[par_loc][iband] = Ssub_Sstar*coord_grid['nsub_star']
         
        else:
        
            #Flux level from region occulted by the planet alone
            #    - set to 1 since it is weighted by flux afterward
            if par_loc=='Ftot':coord_grid[par_loc] = 1.                    

            #Stellar latitude and longitude (degrees)
            #    - sin(lat) = Ystar / Rstar
            #    - sin(lon) = Xstar / Rstar
            elif par_loc=='lat':coord_grid[par_loc] = np.arcsin(coord_grid['y_st'])*180./np.pi    
            elif par_loc=='lon':coord_grid[par_loc] = np.arcsin(coord_grid['x_st'])*180./np.pi     
    
            #Stellar line properties with polynomial spatial dependence 
            elif (par_loc in args['linevar_par_vis']):  
                linevar_coord_grid = calc_linevar_coord_grid(args['coord_line'],coord_grid)
                coord_grid[par_loc] = calc_polymodu(args['pol_mode'],args['coeff_line'][par_loc],linevar_coord_grid) 
    
            #Stellar-rotation induced radial velocity (km/s)
            elif par_loc=='Rot_RV':coord_grid[par_loc] = calc_RVrot(coord_grid['x_st_sky'],coord_grid['y_st'],par_star['istar_rad'],par_star)
                         
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

            #Calculate average property over current occulted region  
            if par_loc=='Ftot':coord_reg_dic_pl[par_loc][iband] = coord_grid[par_loc+'_sum']/coord_grid['nsub_star']
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



'''
Calculation of stellar surface properties from spotted regions
    - we do not pre-calculate a grid to be shifted to the spot location, as with planets, because the projected area of the spot does not remain a disk
'''
def calc_spot_region_prop(bjd_sp,spot_prop,x_st_grid,y_st_grid,z_st_grid,Ssub_Sstar,r2_st_sky,par_star,LD_mod,ld_coeff,cb_band,par_list,range_par_list,range_reg,sum_prop_dic,Ftot_star_grid):

    #Stellar surface period at spot latitude (days)
    #    - accounting for differential rotation: 
    # om(lat) = om_eq*(1-alpha*sin(th_lat)^2)
    # P(lat) = 2*np.pi/(om_eq*(1-alpha*sin(th_lat)^2))
    #    - th_lat is the angular latitude of the spot, counted between the stellar equator and the star-spot center axis  
    th_lat_rad = spot_prop['th_lat']*np.pi/180.
    lat_sp = np.sin(th_lat_rad)
    P_spot = 2*np.pi/((1.-par_star['alpha_rot']*lat_sp**2.-par_star['beta_rot']*lat_sp**4.)*par_star['om_eq']*3600.*24.)

    #Spot phase
    #    - null phase corresponds to the crossing of the projected spin-axis
    Tcen_sp = spot_prop['Tcenter'] - 2400000.  
    ph_rad=(bjd_sp-Tcen_sp)/P_spot

    #Grid coordinates from star rest frame to rotated star frame
    #    - Xstar is the inclined LOS (by istar), Ystar its perpendicular in the star equatorial plane, Zstar the spin axis
    #    - rotation by spot phase angle, so that X_sp_star is the projection of the star-spot axis onto the equatorial plane
    x_sp_star =  x_st_grid*np.cos(ph_rad) + y_st_grid*np.sin(ph_rad)   
    y_sp_star = -x_st_grid*np.sin(ph_rad) + y_st_grid*np.cos(ph_rad)
    z_sp_star =  deepcopy(z_st_grid)

    #Grid coordinates in 'spot rest frame'    
    #    - Xsp axis is the star-spot center axis
    #      Ysp axis is the perpendicular in the plane parallel to the stellar equatorial plane, in the direction of the stellar rotation
    #      Zsp completes the referential
    x_sp =  x_sp_star*np.cos(th_lat_rad) + z_sp_star*np.sin(th_lat_rad) 
    y_sp =  y_sp_star
    z_sp = -x_sp_star*np.sin(th_lat_rad) + z_sp_star*np.cos(th_lat_rad) 

    #Cells within the spot
    #    - ang_sp is the (half) angular size of the spot, so that its projected radius when the spot is seen face-on is Rstar*sin(ang_sp)
    #    - condition is phi(cell) <= ang_sp, with phi counted from the star-spot center axis:
    # phi(cell) = arctan( sqrt(Ysp^2 + Zsp^2)/Xsp ) 
    phi_sp = np.arctan2(np.sqrt(y_sp**2. + z_sp**2.),x_sp)
    cond_in_sp = phi_sp <= spot_prop['ang_sp']*np.pi/180.    
    if True in cond_in_sp:
        region_prop = {}

        #Stellar flux from spot subcells
        #    - assuming a specific intensity at disk center of 1, without LD and GD contributions
        #    - Fr_sp is a flux scaling factor of the spot flux
        flux_grid = np.ones(np.sum(cond_in_sp))*Ssub_Sstar*spot_prop['Fr_sp']

        #Mu coordinate     
        region_prop['mu'] = np.sqrt(1. - r2_st_sky[cond_in_sp]) 
        
        #Limb-darkening 
        flux_grid*= LD_mu_func(LD_mod,region_prop['mu'],ld_coeff)        
        
        #Modify stellar flux from current spot
        #    - we assume the effect of spots is cumulative
        if Ftot_star_grid is not None:
            Ftot_star_grid[cond_in_sp]*=spot_prop['Fr_sp']
                         
        #Processing requested properties
        #    - obliquity is set to 0 as spots are assumed to evolve in planes parallel to the equator
        sum_region_prop(par_list,flux_grid,region_prop,Ssub_Sstar,cb_band,range_par_list,range_reg,sum_prop_dic,par_star,0.,'',cond_fit)


######## appeler avant calcul des prop planetaires
######## updater et sauver la grille stellaire pour chaque exposition
####### ensuite dans la routine planetaire, occulter la grill de l'expo, et faire gaffe a recalculer le flux total par expo aussi


    
    return Ftot_star_grid







'''
Calculation of blackbody emission with gravity-darkening 
    - formalism from Barnes+2009, accounting from gravity-darkening through temperature variation and local blackbody emission
'''
def calc_GD(x_grid_star,y_grid_star,z_grid_star,star_params,gd_band,x_st_sky_grid,y_st_sky_grid):

    #Distance to stellar rotation axis
    rproj_grid_star2 = x_grid_star*x_grid_star + z_grid_star*z_grid_star
    
    #Distance to photosphere
    r_grid_star = np.sqrt(rproj_grid_star2 + y_grid_star*y_grid_star)

    #Local surface gravity vector
    #    - Eq 10 in Barnes+2009
    #    - Vg = gr Vr + grp Vrp  
    # with gr = (-G*Mstar/R^2)   [m3 kg-1 s-2 kg m-2] = [m s-2] 
    #      gr/R = (-G*Mstar/R^3) [s-2]    
    #      grp = om_eq^2*Rp    [rad s-2 m] 
    #      grp/Rp = om_eq^2    [s-2]  
    #    - projected onto the star frame:
    #      Vgx = xstar*((gr/R) + (grp/Rp))
    #      Vgy = ystar*(gr/R) 
    #      Vgz = zstar*((gr/R) + (grp/Rp)) 
    #    - g = sqrt( Vgx^2 + Vgy^2 + Vgz^2 )
    #    - at the pole of the star: 
    # xstar=0,zstar=0,ystar = +- Req*(1-f)
    # R = ystar 
    # gpole = Vgy_pole  
    Rstar_m = star_params['Rstar_km']*1e3                                 
    gr_R = -G_usi*star_params['Mstar']*Msun/(r_grid_star*Rstar_m)**3.
    grp_Rp = star_params['om_eq']**2.
    Vgx_grid_star = x_grid_star*(gr_R + grp_Rp)
    Vgy_grid_star = y_grid_star*gr_R 
    Vgz_grid_star = z_grid_star*(gr_R + grp_Rp) 
    g_grid_star = np.sqrt( Vgx_grid_star**2. + Vgy_grid_star**2. + Vgz_grid_star**2. )

    #Mu angle
    #    - defined as mu = cos(theta), where theta is the angle between the local normal to the photosphere N and the LOS = K
    #    - the cross-product between the local normal and the LOS is vN.vK = N*K*mu
    #    - see calc_zLOS_oblate(), the LOS coordinate of the (visible) photosphere is defined in the sky-projected rest frame via :
    # z = f(x,y) = (-B(y)+sqrt(D(x,y)))/(2*A)
    #   A = (1 - si^2*(1-Rpole^2))
    #   B(y) = 2*y*ci*si*(1 - Rpole^2)
    #   C(x,y) = y^2*si^2*(1- Rpole^2) + Rpole^2*(x^2+ y^2 - 1)
    #   D(x,y) = B(y)^2 - 4*A*C(x,y)
    #      the normal to the photosphere is defined as  
    # N = (-df/dx,-df/dy,1)
    #   df/dx = d[ (-B(y)+sqrt(D(x,y)))/(2*A) ]/dx
    #         = (1/(2*A)) * d[ sqrt(D(x,y)) ]/dx    
    #         =  (1/(2*A)) * ( d[D(x,y)]/dx )/( 2*sqrt(D(x,y)) )
    #         =  (1/(2*A)) * ( - 4*A*dC(x,y)/dx )/( 2*sqrt(D(x,y)) )
    #         = - ( dC(x,y)/dx )/sqrt(D(x,y)
    #         = - ( Rpole^2*2*x )/sqrt(D(x,y)
    #   df/dy = d[ (-B(y)+sqrt(D(x,y)))/(2*A) ]/dy
    #         = (1/(2*A)) * ( -d[B(y)]/dy + d[sqrt(D(x,y))]/dy )
    #         = (1/(2*A)) * ( -(2*ci*si*(1 - Rpole^2))   + ( (d[D(x,y)]/dy )/( 2*sqrt(D(x,y)) )) )
    #         = (1/(2*A)) * ( -(2*ci*si*(1 - Rpole^2))   + ( ( d[B(y)^2]/dy - d[4*A*C(x,y)]/dy )/( 2*sqrt(D(x,y)) )) )    
    #         = (1/(2*A)) * ( -(2*ci*si*(1 - Rpole^2))   + ( ( 2*B(y)*d[B(y)]/dy - 4*A*d[C(x,y)]/dy  )/( 2*sqrt(D(x,y)) )) )      
    #         = (1/(2*A)) * ( -(2*ci*si*(1 - Rpole^2))   + ( ( 2*y*ci*si*(1 - Rpole^2)*( 2*ci*si*(1 - Rpole^2) ) - 2*A*( 2*y*si^2*(1- Rpole^2) + 2*y*Rpole^2 )  )/sqrt(D(x,y)) ) )        
    #         = (1/(2*A)) * ( -(2*ci*si*(1 - Rpole^2))   + ( ( 4*y*ci^2*si^2*(1 - Rpole^2)^2 - 4*A*y*( si^2*(1- Rpole^2) + Rpole^2 )  )/sqrt(D(x,y)) ) )    
    #         = (1/(2*A)) * ( -(2*ci*si*(1 - Rpole^2))   + 4*y*( ( ci^2*si^2*(1 - Rpole^2)^2 - (1 - si^2*(1-Rpole^2))*( si^2*(1- Rpole^2) + Rpole^2 )  )/sqrt(D(x,y)) ) )  
    #         the term inside the parenthesis simplifies as Rpole^2, so that:
    #         = (1/A) * ( -(ci*si*(1 - Rpole^2))   + 2*y*Rpole^2/sqrt(D(x,y)) ) 
    RpoleReq2 = star_params['RpoleReq']**2.
    ci = np.cos(star_params['istar_rad'])
    si = np.sin(star_params['istar_rad'])
    mRp2 = (1. - RpoleReq2)
    Aquad = 1. - si**2.*mRp2
    Bquad = 2.*y_st_sky_grid*ci*si*mRp2
    Cquad = y_st_sky_grid**2.*si**2.*mRp2 + star_params['RpoleReq']**2.*(x_st_sky_grid**2.+ y_st_sky_grid**2. - 1.) 
    det = Bquad**2.-4.*Aquad*Cquad
    Nx_st_sky_grid2 = ( RpoleReq2*2.*x_st_sky_grid )**2./np.abs(det)
    Ny_st_sky_grid2 = ((1./Aquad)*( -(ci*si*mRp2)   + 2.*y_st_sky_grid*RpoleReq2/np.sqrt(det) ))**2.    
    Nz_st_sky_grid = 1.
    N_grid_star = np.sqrt(Nx_st_sky_grid2 + Ny_st_sky_grid2 + Nz_st_sky_grid*Nz_st_sky_grid )
    mu_grid_star = Nz_st_sky_grid/N_grid_star

    #Calculate gravity-darkening
    if len(gd_band)>0:
        y_pole_star = star_params['RpoleReq']
        gr_R_pole = -G_usi*star_params['Mstar']*Msun/(y_pole_star*Rstar_m)**3.
        g_pole_star = np.abs(y_pole_star*gr_R_pole)  
    
        #Stellar cell temperature
        #    - from Maeder+2009
        temp_grid_star = star_params['Tpole']*(g_grid_star/g_pole_star)**star_params['beta_GD']
    
        #Wavelength table for the band, in A
        nw_band=int((gd_band['wmax']-gd_band['wmin'])/gd_band['dw'])
        dw_band=(gd_band['wmax']-gd_band['wmin'])/nw_band
        wav_band=gd_band['wmin']+0.5*dw_band+dw_band*np.arange(nw_band)            
        
        #Black body flux
        #    - at star surface, but absolute flux does not matter, so we scale by a constant factor to get values around 1
        gd_grid_star = np.sum(planck(wav_band[:,None],temp_grid_star)*dw_band,axis=0)/1e11            

    else:
        gd_grid_star = np.ones(mu_grid_star.shape,dtype=float)

    return gd_grid_star,mu_grid_star











'''
Calculation of convective blueshift contribution
'''
def calc_CB_RV(ld_coeff,LD_mod,c1_CB,c2_CB,c3_CB,star_params):
    
    #Convective blueshift contribution
    if (c3_CB != 0.) or (c2_CB != 0.) or (c1_CB != 0.): 
        if star_params['f_GD']>0.:stop('CB not available with GD')

        #---------------------------------------------------------------------        
        #Determination of the constant coefficient for the CB
        #    - after isolating the local stellar profile behind the planet during the transit, we shift them  
        # by the out-of-transit RV obtained from the master-out, unocculted CCFs. This RVout includes not only
        # the systemic RV but also the disk-integrated CB RV (the disk-integrated rotational contribution is null)
        #      thus, the RV we measure and fit here are RVobs = RVobs(st. rot) + RVobs(CB) - RVobs(disk CB)
        #      our model must verify:
        # int( RVmod over disk) = int( RVobs over disk)
        # int( RVmod(st. rot.) + RVmod(poly for CB) over disk) = int( RVobs(st. rot) + RVobs(CB) - RVobs(disk CB), over disk)     
        # int( RVmod(poly for CB) over disk) = int( RVobs(CB) , over disk) - RVobs(disk CB) = 0
        # int( RVmod(poly for CB) over disk) = 0
        # int( phi:0,2pi; theta:0,pi/2 [ I(th) cos(th)*RVconv(th)*Rstar^2*sin(th)*dth*dphi ])/disk_integrated_I = 0
        #        mu = cos(th) and dmu = -sin(th)dth
        # int( mu:0,1 [ I(mu) mu*RVconv(mu)dmu ]) = 0
        #       then for a given limb-darkening law we express c0 as a function of the combination of the other disk integrated coefficients
        #     - with convective blueshift defined as a polynomial in mu:
        # RVconv(mu) = c0 + c1*mu + c2*mu^2 + c3*mu^2 ...
        #       the condition becomes
        # c0*int( mu:0,1 [ I(mu)*mu*dmu ]) + sum(i>=1 , ci*int( mu:0,1 [ I(mu)*mu^(i+1)*dmu ])  = 0    
        # c0 = - sum(i>=1 , ci*Int_i)/Int_0
        #       with Int_i = int( mu:0,1 [ I(mu)*mu^(i+1)*dmu ])
        #            I(mu) = I0*LD(mu) and I0 the specific intensity at disk center disk, which is not involved in the calculations                                 
        # the constant coefficient in our model includes the true constant contribution from the CB and a correction factor due to the RV shift
        if LD_mod == 'uniform' or LD_mod == 'linear' or LD_mod == 'quadratic':
            LD_u1,LD_u2=ld_coeff[0:2]
            Int0 = 0.5 - (1./6.)*LD_u1 - (1./12.)*LD_u2           #int(0:1 , I(mu)*mu*dmu)
            Int1 = (1./3.) - (1./12.)*LD_u1 - (1./30.)*LD_u2      #int(0:1 , I(mu)*mu2*dmu)
            Int2 = (1./4.) - (1./20.)*LD_u1 - (1./60.)*LD_u2      #int(0:1 , I(mu)*mu3*dmu)
            Int3 = (1./5.) - (1./30.)*LD_u1 - (1./105.)*LD_u2     #int(0:1 , I(mu)*mu4*dmu)
        elif LD_mod == 'nonlinear':
            LD_u1,LD_u2,LD_u3,LD_u4=ld_coeff[0:5]
            Int0 = 0.5 - (1./10.)*LD_u1 - (1./6.)*LD_u2 - (3./14.)*LD_u3 - (1./4.)*LD_u4           #int(0:1 , I(mu)*mu*dmu)
            Int1 = (1./3.) - (1./21.)*LD_u1 - (1./12.)*LD_u2 - (1./9.)*LD_u3 - (2./15.)*LD_u4      #int(0:1 , I(mu)*mu2*dmu)
            Int2 = (1./4.) - (1./36.)*LD_u1 - (1./20.)*LD_u2 - (3./44.)*LD_u3 - (1./12.)*LD_u4     #int(0:1 , I(mu)*mu3*dmu)
            Int3 = (1./5.) - (1./55.)*LD_u1 - (1./30.)*LD_u2 - (3./65.)*LD_u3 - (2./35.)*LD_u4     #int(0:1 , I(mu)*mu4*dmu)
        else:stop('LD law undefined')
            
        #Solving the equation int( RVmod(poly for CB) over disk) = 0 we get (in km/s)
        c0_CB = - (c1_CB*Int1 + c2_CB*Int2 +c3_CB*Int3)/Int0

    else:
        c0_CB=0.
    cb_band = np.array([c0_CB,c1_CB,c2_CB,c3_CB])    
    return cb_band













'''
Frame conversion, from the classical frame perpendicular to the LOS, to the sky-projected (inclined) stellar frame
    - rotation around the Z axis (remains unchanged), ie the LOS 
      lambda counted >0 counterclockwise from the X' to the X axis, or the Y' to Y axis, in the XY plane
    - X' is the sky-projected stellar equator
      Y' vertical axis is the sky-projected stellar spin 
      Z' = Z 
'''
def conv_Losframe_to_inclinedStarFrame(lambd,xin,yin,zin):
    xout = xin*np.cos(lambd) - yin*np.sin(lambd)
    yout = xin*np.sin(lambd) + yin*np.cos(lambd)
    zout=deepcopy(zin)
    return xout,yout,zout

def conv_inclinedStarFrame_to_Losframe(lambd,xin,yin,zin):  
    xout = xin*np.cos(lambd)+yin*np.sin(lambd) 
    yout = -xin*np.sin(lambd)+yin*np.cos(lambd) 
    zout=deepcopy(zin)  
    return xout,yout,zout

'''
Frame conversion from the inclined star frame to the 'star' frame 
    - rotation around the X axis (remains unchanged) 
      istar counted >0 counterclockwise from the Z to the Y' axis, in the YZ plane
    - Y' is the stellar spin
      X' is the stellar equator
'''
def conv_inclinedStarFrame_to_StarFrame(xin,yin,zin,istar):
    xout=deepcopy(xin)
    yout= np.sin(istar)*yin + np.cos(istar)*zin
    zout=-np.cos(istar)*yin + np.sin(istar)*zin   
    return xout,yout,zout

def conv_StarFrame_to_inclinedStarFrame(xin,yin,zin,istar):
    xout=deepcopy(xin)
    yout= np.sin(istar)*yin - np.cos(istar)*zin
    zout= np.cos(istar)*yin + np.sin(istar)*zin   
    return xout,yout,zout

'''
Coordinate along z axis (=LOS) in sky-projected frame (=inclined star frame), for an oblate star
    - Barnes+2009, Eq. 13 and 14 (with phi = pi/2 - istar)
      BEWARE: their Eq 14 is wrong, 1-f^2 should be (1-f)^2
    - coordinates are for cells in the stellar hemisphere facing the observer
    - there is no Req in the equation because coordinates are all normalized by Rstar
'''
def calc_zLOS_oblate(x_st_sk,y_st_sk,istar_rad,RpoleReq):
    #the photosphere is defined in the star frame as (Eq 12 in Barnes 2009):
    #xstar^2 + ystar^2 / Rpole^2 + zstar^2 = 1
    #in coordinates of the sky-projected star frame:
    #xstar = x
    #ystar = y*si + z*ci
    #zstar =-y*ci + z*si
    #x^2 + (y*si + z*ci)^2 / Rpole^2 + (-y*ci + z*si)^2 = 1
    #(x*Rpole)^2 + y^2*si^2 + 2*y*z*ci*si + z^2*ci^2 + y^2*ci^2*Rpole^2 - 2*y*ci*z*si*Rpole^2 + z^2*si^2*Rpole^2 = Rpole^2
    #z^2*ci^2 + z^2*si^2*Rpole^2    +  2*y*z*ci*si - 2*y*ci*z*si*Rpole^2   + (x*Rpole)^2 + y^2*si^2 + y^2*ci^2*Rpole^2 - Rpole^2 = 0
    #z^2*(ci^2 + si^2*Rpole^2) +  2*y*z*ci*si*(1 - Rpole^2) + y^2*(si^2 + ci^2*Rpole^2) + Rpole^2*(x^2 - 1) = 0
    #z^2*(1 - si^2*(1-Rpole^2)) +  2*y*z*ci*si*(1 - Rpole^2) + y^2*si^2*(1- Rpole^2) + Rpole^2*(x^2+ y^2 - 1) = 0
    #   A = (1 - si^2*(1-Rpole^2))
    #   B = 2*y*ci*si*(1 - Rpole^2)
    #   C = y^2*si^2*(1- Rpole^2) + Rpole^2*(x^2+ y^2 - 1)
    #delta = B^2 - 4*A*C
    #z = (-B+-sqrt(delta))/(2*A)
    ci = np.cos(istar_rad)
    si = np.sin(istar_rad)
    mRp2 = (1. - RpoleReq**2.)
    Aquad = 1. - si**2.*mRp2
    Bquad = 2.*y_st_sk*ci*si*mRp2
    Cquad = y_st_sk**2.*si**2.*mRp2 + RpoleReq**2.*(x_st_sk**2.+ y_st_sk**2. - 1.) 
    det = Bquad**2.-4.*Aquad*Cquad
    cond_def = det>=0.
    z_st_sky_behind = np.zeros(len(x_st_sk),dtype=float)*np.nan
    z_st_sky_front = np.zeros(len(x_st_sk),dtype=float)*np.nan
    if True in cond_def:
        z_st_sky_behind[cond_def] = (-Bquad[cond_def]-np.sqrt(det[cond_def]))/(2.*Aquad)
        z_st_sky_front[cond_def] = (-Bquad[cond_def]+np.sqrt(det[cond_def]))/(2.*Aquad)
    return z_st_sky_behind,z_st_sky_front,cond_def


'''
Limb-Darkening coefficients
    - we use the same structure for all cases : [LD_u1,LD_u2,LD_u3,LD_u4]
'''
def LD_coeff_func(transit_prop,iband):
    LD_mod = transit_prop['LD'][iband]
    #Uniform
    if LD_mod == 'uniform':
        ld_coeff=[np.nan,np.nan,np.nan,np.nan]
    #Linear
    elif LD_mod == 'linear':
        ld_coeff=[transit_prop['LD_u1'][iband],np.nan,np.nan,np.nan]
    #Quadratic, squareroot, logarithmic, power-2, exponential
    elif LD_mod in [ 'quadratic' ,'squareroot','logarithmic', 'power2' ,'exponential']:
        ld_coeff=[transit_prop['LD_u1'][iband],transit_prop['LD_u2'][iband],np.nan,np.nan]
    #Nonlinear
    elif LD_mod == 'nonlinear':               
        ld_coeff=[transit_prop['LD_u1'][iband],transit_prop['LD_u2'][iband],transit_prop['LD_u3'][iband],transit_prop['LD_u4'][iband]]
    #Solar
    elif LD_mod == "Sun":               
        ld_coeff=[transit_prop['LD_u'+str(i)][iband] for i in range(1,7)]
    else:
        stop('Limb-darkening law not supported by ANTARESS')  
    return ld_coeff

'''
Limb-Darkening value at a given mu
'''
def LD_mu_func(LD_mod,mu,ld_coeff):
    if LD_mod == 'uniform':
        ld_val = 1.
    elif LD_mod == 'linear':
        ld_val = 1. - ld_coeff[0]*(1. -mu)
    elif LD_mod == 'quadratic':
        ld_val = 1. - ld_coeff[0]*(1. -mu) - ld_coeff[1]*np.power(1. -mu,2.)
    elif LD_mod == 'squareroot':
        ld_val = 1. - ld_coeff[0]*(1. -mu) - ld_coeff[1]*(1. - np.sqrt(mu))
    elif LD_mod == 'nonlinear':
        ld_val = 1. - ld_coeff[0]*(1. -mu**0.5) - ld_coeff[1]*(1. -mu) - ld_coeff[2]*(1. -mu**1.5) - ld_coeff[3]*(1. -mu**2.)
    elif LD_mod == 'power2':        
        ld_val = 1. - ld_coeff[0]*(1. -mu**ld_coeff[1]) 
    elif LD_mod == "Sun":        
        norm_ld = 2.*np.pi*(ld_coeff[0]/2. + ld_coeff[1]/3. - ld_coeff[2]/4. + ld_coeff[3]/5. -ld_coeff[4]/5. +ld_coeff[5]/7.)
        ld_val = (ld_coeff[0] + ld_coeff[1]*mu - ld_coeff[2]*mu**2. +ld_coeff[3]*mu**3. -ld_coeff[4]*mu**4. +ld_coeff[5]*mu**5.)/norm_ld
    return ld_val





























'''
Definition of polynomial coefficients from the parameter format
'''
def polycoeff_def(param,coeff_ord2name_polpar):

    #Polynomial coefficients 
    #    - keys in 'coeff_ord2name_polpar' are the coefficients degrees, values are their names, as defined in 'param' 
    #      they can be defined in disorder (in terms of degrees), as coefficients are forced to order from deg_max to 0 in 'coeff_grid_polpar' 
    #    - degrees can be missing
    #    - input coefficients must be given in decreasing order of degree to poly1d
    deg_max=max(coeff_ord2name_polpar.keys())
    coeff_grid_polpar=[param[coeff_ord2name_polpar[ideg]] if ideg in coeff_ord2name_polpar else 0. for ideg in range(deg_max,-1,-1)]

    return coeff_grid_polpar

#Calculation of 'absolute' or 'modulated' polynomial
#    - 'poly1d' takes coefficient array in decreasing powers
#      'coeff_pol' has been defined in this way, using input coefficient defined through their power value
#    - 'abs' : coeff_pol[0]*x^n + coeff_pol[1]*x^(n-1) .. + coeff_pol[n]*x^0
#      'modul' : (coeff_pol[0]*x^n + coeff_pol[1]*x^(n-1) .. + 1)*coeff_pol[n]
def calc_polymodu(pol_mode,coeff_pol,x_val):
    if pol_mode=='abs':
        mod= np.poly1d(coeff_pol)(x_val)         
    elif pol_mode=='modul':
        coeff_pol_modu = coeff_pol[:-1] + [1]
        mod= coeff_pol[-1]*np.poly1d(coeff_pol_modu)(x_val)  
    else:stop('Undefined polynomial mode')
    return mod

'''
Function returning the relevant coordinate for calculation of line properties variations
'''
def calc_linevar_coord_grid(dim,grid):
    if (dim in ['mu','xp_abs','r_proj','y_st']):linevar_coord_grid = grid[dim]
    elif (dim=='y_st2'):linevar_coord_grid = grid['y_st']**2.
    elif (dim=='abs_y_st'):linevar_coord_grid = np.abs(grid['y_st'])
    else:stop('Undefined line coordinate')
    return linevar_coord_grid

    
    
# Stage Théo : fusion des 2 dernières fonctions. 
def poly_prop_calc(param,fit_coord_grid,coeff_ord2name_polpar, pol_mode):

    #Polynomial coefficients 
    #    - keys in 'coeff_ord2name_polpar' are the coefficients degrees, values are their names, as defined in 'param' 
    #      they can be defined in disorder (in terms of degrees), as coefficients are forced to order from 0 to deg_max in 'coeff_pol_ctrst' 
    #    - degrees can be missing
    #     - input coefficients must be given in decreasing order of degree to poly1d
    deg_max=max(coeff_ord2name_polpar.keys())
    
    coeff_grid_polpar=[param[coeff_ord2name_polpar[ideg]] if ideg in coeff_ord2name_polpar else 0. for ideg in range(deg_max,-1,-1)]
    
    # Absolute polynomial 
    if pol_mode == 'abs' : 
        polpar_grid = np.poly1d(coeff_grid_polpar)(fit_coord_grid)
        
    # Modulated polynomial
    if pol_mode == 'modul' :
        coeff_pol_modu = coeff_grid_polpar[:-1] + [1]
        polpar_grid = np.poly1d(coeff_pol_modu)(fit_coord_grid)*coeff_grid_polpar[-1]
        
    return polpar_grid



'''
Intrinsic to local intensity scaling
    - see proc_intr_data() and loc_prof_DI_mast(): intrinsic spectra do not necessarily have the same flux as the disk-integrated profiles, but have been set to the same continuum or total flux
    - now that the intrinsic and disk-integrated profiles are equivalent in terms of flux we have:
   F_intr(w,t,vis) ~ F_DI(w,vis)
   this is valid for broadband fluxes or the continuum of CCFs (see rescale_data() )
    - the model disk-integrated profile is defined as 
   F_DI_mod(w,vis) = sum( all stellar cells , F_intr(w - w_shift , x,vis)*LD(cell,w)*dS ) / A
                   ~ sum( all stellar cells , F_DI(w - w_shift,vis)*LD(cell,w)*dS ) / A
      where the shift accounts for the stellar surface velocity field. If we neglect these shifts (for spectra) or consider the continuum range (for CCF)
   F_DI_mod(w,vis) ~ F_DI(w,vis)*sum( all stellar cells , LD(cell,w)*dS ) / A
   A(w) = sum( all stellar cells , LD(cell,w)*dS )
   where A is a spectrum
    - if the line profile is calculated directly within the fit function we store the scaling profiles
'''
def theo_intr2loc(grid_dic,system_prop,fixed_args,ncen_bins,nsub_star):

    #Grid of broadband intensity variations, at the resolution of current grid
    #    - condition is validated if data is spectral and intensity input is chromatic
    if ('spec' in fixed_args['type']) and ('chrom' in system_prop):

        #Grid of broadband flux variations, at the spectral resolution of the input chromatic properties
        Fsurf_grid_band = grid_dic['Fsurf_grid_star_chrom']
        
        #Grid of broadband flux variations, at the spectral resolution of the local profiles
        #    - interpolation over the profile table if the full range of the profile is larger than the scale of chromatic variations, otherwise closest definition point
        fixed_args['Fsurf_grid_spec'] = np.zeros([nsub_star,ncen_bins])
        if (fixed_args['edge_bins'][-1]-fixed_args['edge_bins'][0]>system_prop['chrom']['med_dw']):
            for icell in range(nsub_star):
                fixed_args['Fsurf_grid_spec'][icell,:] = np_interp(fixed_args['cen_bins'],system_prop['chrom']['w'],Fsurf_grid_band[icell],left=Fsurf_grid_band[icell,0],right=Fsurf_grid_band[icell,-1])
        else:
            iband = closest(system_prop['chrom']['w'],np.median(fixed_args['cen_bins']))
            fixed_args['Fsurf_grid_spec'] = Fsurf_grid_band[:,iband][:,None]
    else:
        fixed_args['Fsurf_grid_spec'] = np.tile(grid_dic['Fsurf_grid_star_achrom'][:,0],[ncen_bins,1]).T       

    return None


















'''
Determine the correlation length for a given dataset
    - see Bourrier et al. 2015 (tomography) and Pont et al. 2006 for the method
    - routine should be applied to residuals spread around 0 : we use out-of-transit local residual CCFs
'''
def corr_length_determination(Res_data_vis,data_vis,scr_search,inst,vis,gen_dic):

    #Processing out-of-transit data
    Res_data_vis['corr_search']={
        'meas':np.zeros([data_vis['n_out_tr'],gen_dic['scr_srch_max_binwin']],dtype=float),
        'fit':np.zeros([data_vis['n_out_tr'],gen_dic['scr_srch_max_binwin']],dtype=float),
        'sig_corr':np.zeros([data_vis['n_out_tr']],dtype=float),
        'sig_uncorr':np.zeros([data_vis['n_out_tr']],dtype=float)
        }
    for isub,iexp in enumerate(gen_dic[inst][vis]['idx_out']):    
        
        #Upload data
        data_exp = np.load(data_vis['proc_Res_data_paths']+str(iexp)+'.npz',allow_pickle=True)['data'].item()
        n_pix = len(data_exp['flux'][0]) 
        
        #Calculate dispersion on binned points for each window bin size
        for ibin,nperbin in enumerate(gen_dic['scr_srch_nperbins']):
            
            #Number of possible positions for the sliding window, and corresponding positions
            #    - the last value for i_0 is n_win-1
            #                     for i_f is n_pts-1
            n_win=n_pix-nperbin+1.
            i_0_tab=np.arange(n_win)
            i_f_tab=nperbin-1.+np.arange(n_pix-nperbin+1.)                    
    
    #       #Placing sliding window successively with no overlap
    #       n_win=int(n_pix/nperbin)
    #       i_0_tab=np.arange(n_win)*nperbin
    #       i_f_tab=i_0_tab+nperbin-1.
    
            #Table of the mean residuals within a given window
            mean_tabs=np.zeros(n_win)
    
            #Sliding window along the whole time-series
            #    - steps are by one point, smaller than the sampling interval
            for idx,(i_0,i_f) in enumerate(zip(i_0_tab,i_f_tab)): 
      
                #Binning residuals within current window
                mean_tabs[idx]=np.mean(data_exp['flux'][i_0:i_f+1])
                          
            #Standard-deviation of the mean residuals, for current bin size 
            Res_data_vis['corr_search']['meas'][isub,ibin]=mean_tabs.std()
    
        #--------------------------------------------------------------
    
        #Measured values to fit
        xtofit = gen_dic['scr_srch_nperbins']
        ytofit = Res_data_vis['corr_search']['meas'][isub]
        covtofit = np.ones([1,len(xtofit)])
    
        #Guess
        #     -  sig_bin=1./np.power( np.power(sqrt(nbin)/sig_uncorr,4.) + np.power(1./sig_corr,4.) ,1./4.)
        # + for large bin size, uncorrelated noise should dominate: sig_bin ~  sig_uncorr / sqrt(nbin)
        # + for a bin size of one, we use the guess for the uncorrelated noise
        sig_uncorr= ytofit[-1]*np.sqrt(xtofit[-1])
        sig_corr = np.power( (1./np.power(ytofit[0],4.) ) - np.power(np.sqrt(1.)/sig_uncorr,4.) , -1./4. )       
    
        # Initialise fit parameters 
        #             (    Name,    Value,  Vary,   Min,   Max,  Expr)
        p_use = Parameters()
        p_use.add_many(( 'sig_uncorr', sig_uncorr,  True,None,None,  None),
                      (  'sig_corr', sig_corr,  True,None,None,  None))
    
        #Fitting
        result,merit,p_best = fit_minimization(ln_prob_func_lmfit,p_use,xtofit,ytofit,covtofit,binned_stddev_fit,verbose=True)

        #Saving fit and its uncorrelated/correlated components 
        Res_data_vis['corr_search']['fit'][isub]=merit['fit']
        Res_data_vis['corr_search']['sig_corr'][isub]=p_best['sig_corr'].value    
        Res_data_vis['corr_search']['sig_uncorr'][isub]=p_best['sig_uncorr'].value 
    
    return None                








 




















'''
Global call to functions defining planet-occulted profiles associated with each observed exposure
    - local profiles are used to correct residual profiles from stellar contamination
    - intrinsic profiles are used to assess fit quality
'''
def def_plocc_profiles(inst,vis,gen_dic,data_dic,data_prop,coord_dic,system_param,theo_dic,glob_fit_dic):

    print('   > Building estimates for planet-occulted stellar profiles') 
    text_print={
        'DIbin':    '           Using DI master',
        'Intrbin':  '           Using binned intrinsic profiles',
        'glob_mod': '           Using global model',
        'indiv_mod':'           Using individual models',
        'rec_prof': '           Using reconstruction',
        }[data_dic['Intr']['mode_loc_data_corr']]
    print(text_print)  

    #Calculating
    if (gen_dic['calc_loc_data_corr']):
        print('         Calculating data')     
 
        #Using disk-integrated master or binned intrinsic profiles
        if data_dic['Intr']['mode_loc_data_corr'] in ['DIbin','Intrbin']:
            data_add = loc_prof_meas(data_dic['Intr']['mode_loc_data_corr'],inst,vis,data_dic['Intr'],gen_dic,data_dic,data_prop,data_dic['Atm'],coord_dic)
            
        #Using global profile model
        elif data_dic['Intr']['mode_loc_data_corr']=='glob_mod': 
            data_add = loc_prof_globmod(inst,vis,gen_dic,data_dic,data_prop,system_param,theo_dic,coord_dic,glob_fit_dic)
            
        #Using individual profile models
        elif data_dic['Intr']['mode_loc_data_corr']=='indiv_mod': 
            data_add = loc_prof_indivmod(inst,vis,gen_dic,data_dic)

        #Defining undefined pixels via a polynomial fit to defined pixels in complementary exposures, or via a 2D interpolation over complementary exposures and a narrow spectral band
        elif data_dic['Intr']['mode_loc_data_corr']=='rec_prof': 
            data_add = loc_prof_rec(inst,vis,gen_dic,data_dic,coord_dic)
            
        #Saving complementary data
        data_add['loc_data_corr_inpath'] = data_dic[inst][vis]['proc_Intr_data_paths']
        data_add['loc_data_corr_outpath'] = data_dic[inst][vis]['proc_Res_data_paths']
        data_add['rest_frame'] = data_dic['Intr'][inst][vis]['rest_frame']
        datasave_npz(gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['mode_loc_data_corr']+'/'+inst+'_'+vis+'_add',data_add)
            
    #Checking that local data has been calculated for all exposures
    else:
        idx_est_loc = np.load(gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['mode_loc_data_corr']+'/'+inst+'_'+vis+'_add.npz',allow_pickle=True)['data'].item()['idx_est_loc']
        data_paths={i_in:gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['mode_loc_data_corr']+'/'+inst+'_'+vis+'_'+str(i_in) for i_in in idx_est_loc}
        check_data(data_paths)

    return None




'''
Sub-function for local profiles using measured profiles as best estimates of intrinsic profiles
    - when binned intrinsic profiles are used:
 + we first define the range and center of the binned profiles, along the bin dimension
 + for each in-transit local profile, we find the nearest binned profile along the bin dimension (via their centers)
 + we use intrinsic profiles that have already been aligned to a null rest frame, first shifting them to the rv of the target local profile, and then binning them
 + we could use the original intrinsic profiles and shift them by their own rv minus that of the local profile, however in this way we can use directly the outputs of align_Intr(), and 
   shifting the profiles in two steps is not an issue when they are not resampled until binned
 + we finally bin the aligned profiles, and scale the binned profile to the level of the local one        
'''
def loc_prof_meas(corr_mode,inst,vis,gen_dic,data_dic,data_prop,coord_dic):
    data_inst = data_dic[inst]
    data_vis = data_inst[vis]
    opt_dic = data_dic['Intr']['opt_loc_data_corr'][corr_mode]
    
    #Initialize surface RV
    ref_pl,dic_rv,idx_aligned = init_surf_shift(gen_dic,inst,vis,data_dic,data_dic['Intr']['align_mode'])  

    #Binning mode
    #   - using current visit exposures only, or exposures from multiple visits
    if (inst in opt_dic['vis_in_bin']) and (len(opt_dic['vis_in_bin'][inst])>0):vis_to_bin = opt_dic['vis_in_bin'][inst]   
    else:vis_to_bin = [vis]           
    
    #Initialize binning
    #    - when disk-integrated profiles are used, the output tables contain a single value, associated with the single master (=binned profile) used for the extraction
    if corr_mode=='DIbin':
        in_type='DI'
        dim_bin = 'phase'
    elif corr_mode=='Intrbin':
        in_type='Intr'
        dim_bin = opt_dic['dim_bin']
        if not gen_dic['align_Intr']:stop('Intrinsic profiles must have been aligned')
    new_x_cen,_,_,x_cen_all,n_in_bin_all,idx_to_bin_all,dx_ov_all,_,idx_bin2orig,idx_bin2vis,idx_to_bin_unik = init_bin_rout(in_type,opt_dic[inst][vis],opt_dic['idx_in_bin'],dim_bin,coord_dic,inst,vis_to_bin,data_dic,gen_dic)

    #Find binned profile closest (along bin dimension ) to each processed in-transit exposure 
    if corr_mode=='Intrbin':idx_bin_closest = closest_arr(new_x_cen, x_cen_all[idx_aligned])
    
    #Processing in-transit exposures for which planet-occulted rv is known
    for isub,i_in in enumerate(idx_aligned):    
    
        #Upload spectral tables from residual or intrinsic profile of current exposure
        if data_dic['Intr']['plocc_prof_type']=='Intr':iexp_eff = i_in
        elif data_dic['Intr']['plocc_prof_type']=='Res':iexp_eff = gen_dic[inst][vis]['idx_in2exp'][i_in]
        data_loc_exp = dataload_npz(data_vis['proc_'+data_dic['Intr']['plocc_prof_type']+'_data_paths']+str(iexp_eff))

        #Index of binned profile associated with current processed exposure
        if corr_mode=='DIbin':i_bin = 0
        elif corr_mode=='Intrbin':i_bin = idx_bin_closest[isub]
                
        #Calculating binned profile associated with current processed exposure
        #    - since the shift is specific to each processed exposure, contributing profiles must be aligned and resampled for each one, either on the table common to all profiles, or on the table of current exposure, before being binned together        
        data_to_bin={}
        for iexp_off in idx_to_bin_all[i_bin]:    
        
            #Original index and visit of contributing exposure
            iexp_bin = idx_bin2orig[iexp_off]
            vis_bin = idx_bin2vis[iexp_off]
            
            #Upload latest processed disk-integrated data         
            if corr_mode=='DIbin':
                iexp_bin_glob = iexp_bin
                data_exp_bin = np.load(data_inst[vis_bin]['proc_DI_data_paths']+str(iexp_bin)+'.npz',allow_pickle=True)['data'].item() 
                if gen_dic['flux_sc']: scaled_data_paths = data_dic[inst][vis_bin]['scaled_DI_data_paths']  
                else:scaled_data_paths=None
                
                #Exclude planet-contaminated bins 
                #    - here we set to nan the flux (rather than just the defined bins) when profiles are still aligned in the star rest frame, to avoid having to shift for every exposure the planet-excluded ranges
                #      after profiles are aligned to the local surface velocity, the defined pixels will account for the exclusion and be used to define the weights in the bin routine
                if ('DI_Mast' in data_dic['Atm']['no_plrange']) and (iexp_bin in data_dic['Atm'][inst][vis_bin]['iexp_no_plrange']):
                    for iord in range(data_inst['nord']):
                        data_exp_bin['flux'][iord][excl_plrange(data_exp_bin['cond_def'][iord],data_dic['Atm'][inst][vis_bin]['exclu_range_star'],iexp_bin,data_loc_exp['edge_bins'][iord],data_vis['type'])[0]] = np.nan
                if data_dic[inst][vis_bin]['tell_sp']:data_exp_bin['tell'] = dataload_npz(data_dic[inst][vis_bin]['tell_DI_data_paths'][iexp_bin])['tell']             
                if data_dic[inst][vis_bin]['mean_gdet']:data_exp_bin['mean_gdet'] = dataload_npz(data_dic[inst][vis_bin]['mean_gdet_DI_data_paths'][iexp_bin])['mean_gdet']             
                data_ref = dataload_npz(data_dic[inst][vis_bin]['mast_DI_data_paths'][iexp_bin])
                
                
            #Upload intrinsic stellar profiles aligned in their local frame
            #    - we use intrinsic profiles already aligned in a null rest frame (still defined on their original tables unless a common table was used)     
            #    - profiles could be retrieved from their original extraction rest frame, and shifted directly to the surface rv associated with current exposure
            #      however in the ideal processing profiles were maintained on their individual tables, which were shifted without resampling to create the aligned profiles, thus doing an additional shift here
            # will not create any correlations 
            elif corr_mode=='Intrbin':
                iexp_bin_glob = gen_dic[inst][vis_bin]['idx_in2exp'][iexp_bin]
                if iexp_bin_glob not in data_dic['Intr'][inst][vis]['idx_def']:stop('Intrinsic exposure at i=',str(iexp_bin),' has not been aligned')
                data_exp_bin = dataload_npz(gen_dic['save_data_dir']+'Aligned_Intr_data/'+inst+'_'+vis_bin+'_'+str(iexp_bin))
                if data_dic[inst][vis_bin]['tell_sp']:data_exp_bin['tell'] = dataload_npz(gen_dic['save_data_dir']+'Aligned_Intr_data/'+inst+'_'+vis_bin+'_in_tell'+str(iexp_bin))['tell']             
                if data_dic[inst][vis_bin]['mean_gdet']:data_exp_bin['mean_gdet'] = dataload_npz(gen_dic['save_data_dir']+'Aligned_Intr_data/'+inst+'_'+vis_bin+'_in_mean_gdet'+str(iexp_bin))['mean_gdet']             
                data_ref = np.load(gen_dic['save_data_dir']+'Aligned_Intr_data/'+inst+'_'+vis_bin+'_in_ref'+str(iexp_bin)+'.npz',allow_pickle=True)['data'].item() 
                if gen_dic['flux_sc']: scaled_data_paths = data_dic[inst][vis_bin]['scaled_Intr_data_paths']  
                else:scaled_data_paths=None
                
            #Radial velocity shifts set to the opposite of the planet-occulted surface rv associated with current exposure
            surf_shifts,surf_shifts_edge = def_surf_shift(data_dic['Intr']['align_mode'],dic_rv,i_in,data_exp_bin,ref_pl,data_vis['type'],data_dic['DI']['system_prop'],data_dic[inst][vis_bin]['dim_exp'],data_dic[inst]['nord'],data_dic[inst][vis_bin]['nspec'])
            if surf_shifts_edge is not None:surf_shifts_edge*= -1.
                    
            #Aligning contributing profile at current exposure stellar surface rv 
            #    - aligned profiles are here resampled on the table of current exposure, which is common to all exposures if a common table is used        
            #    - complementary tables follow the same shifts
            data_to_bin[iexp_off]=align_data(data_exp_bin,data_vis['type'],data_dic[inst]['nord'],data_dic[inst][vis]['dim_exp'],gen_dic['resamp_mode'],data_loc_exp['cen_bins'],data_loc_exp['edge_bins'],-surf_shifts,rv_shift_edge = surf_shifts_edge)
                
            #Shifting weighing master at current exposure stellar surface rv 
            #    - master will be resampled on the same table as current exposure
            data_ref_align=align_data(data_ref,data_vis['type'],data_dic[inst]['nord'],data_dic[inst][vis]['dim_exp'],gen_dic['resamp_mode'],data_loc_exp['cen_bins'], data_loc_exp['edge_bins'],-surf_shifts,rv_shift_edge = surf_shifts_edge)

            #Weight profile
            data_to_bin[iexp_off]['weight'] = def_weights_spatiotemp_bin(range(data_inst['nord']),scaled_data_paths,inst,vis_bin,gen_dic['corr_Fbal'],gen_dic['corr_FbalOrd'],gen_dic['save_data_dir'],gen_dic['type'],data_inst['nord'],iexp_bin_glob,in_type,data_vis['type'],data_vis['dim_exp'],data_to_bin[iexp_off]['tell'],data_to_bin[iexp_off]['mean_gdet'],data_to_bin[iexp_off]['cen_bins'],coord_dic[inst][vis_bin]['t_dur'][iexp],data_ref_align['flux'],data_ref_align['cov'],bdband_flux_sc = gen_dic['flux_sc'])            
            
        #Calculating binned profile
        data_est_loc = calc_binned_prof(idx_to_bin_all[i_bin],data_dic[inst]['nord'],data_vis['dim_exp'],data_vis['nspec'],data_to_bin,inst,n_in_bin_all[i_bin],data_loc_exp['cen_bins'],data_loc_exp['edge_bins'],dx_ov_in = dx_ov_all[i_bin])
        
        #Rescaling measured intrinsic profile to the level of the local profile
        #    - this operation assumes that all exposures used to compute the master-out have not been rescaled with respect to the reference, ie that they all have the same flux balance as current exposure before it was rescaled
        #    - see rescale_data() and proc_intr_data() for more details
        #      a given local profile write as 
        #      F_res(w,t,vis) = MFstar(w,vis) - Fsc(w,vis,t)
        #   at low resolution, in the continuum (see rescale_data())
        #      Fsc(w,vis,t) = LC(w,t)*Fstar(w,vis_norm)*Cref(w)
        #   assuming that the rescaling light curves are constant outside of the transit, all out-of-transit profiles are equivalent between themselves and thus to the master-out
        #      MFstar(w,vis) ~ Fstar(w,vis_norm)*Cref(w)
        #   thus
        #      F_res(w,t,vis) = Fstar(w,vis_norm)*Cref(w) - LC(w,t)*Fstar(w,vis_norm)*Cref(w)
        #                     = Fstar(w,vis_norm)*Cref(w)*(1 - LC(w,t))
        #                     = MFstar(w,vis)*(1 - LC(w,t))           
        #    - if intrinsic profiles are requested no scaling is applied, since F_res(w,t,vis) = F_intr(w,t,vis)*(1 - LC(w,t))      
        #    - the scaling spectrum is defined at all pixels, and thus does not affect undefined pixels in the master (the covariance matrix cannot be sliced)
        if (data_dic['Intr']['plocc_prof_type']=='Res') and gen_dic['flux_sc']:
            data_scaling = dataload_npz(data_vis['scaled_Intr_data_paths']+str(gen_dic[inst][vis]['idx_in2exp'][i_in]))
            for iord in range(data_inst['nord']):
                loc_flux_scaling_ord = data_scaling['loc_flux_scaling'](data_est_loc['cen_bins'][iord])              
                data_est_loc['flux'][iord],data_est_loc['cov'][iord] = bind.mul_array(data_est_loc['flux'][iord],data_est_loc['cov'][iord],loc_flux_scaling_ord)
            
        #Saving estimate of local profile for current exposure                   
        np.savez_compressed(gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['mode_loc_data_corr']+'/'+inst+'_'+vis+'_'+str(i_in),data=data_est_loc,allow_pickle=True)

    #Complementary data
    data_add={'idx_est_loc':idx_aligned}
    if corr_mode=='Intrbin':data_add['idx_bin_closest']=idx_bin_closest

    return data_add




'''
Sub-function for local profiles using line profile model
    - we use the model fitted to all intrinsic lines together in fit_IntrProf_all()
      model can be based on analytical, measured, or theoretical profiles
    - flux scaling is not applied to a global intrinsic profile using the chromatic light curve, but re-calculated for each planet-occulted intrinsic line profile for more flexibility (the cumulated scaling should be equivalent)
      rv used to shift the profiles are similarly re-calculated theoretically 
'''
def loc_prof_globmod(inst,vis,gen_dic,data_dic,data_prop,system_param,theo_dic,coord_dic,glob_fit_dic):
    data_vis = data_dic[inst][vis]
    gen_opt_dic = data_dic['Intr']['opt_loc_data_corr'] 
    def_iord = gen_opt_dic['def_iord']
    opt_dic = gen_opt_dic['glob_mod']
    if (data_dic['Intr']['align_mode']=='meas'):stop('This mode cannot be used with measured RVs')

    #Retrieving selected model properties
    if ('IntrProf_prop_path' not in opt_dic):opt_dic['IntrProf_prop_path']={inst:{vis:gen_dic['save_data_dir']+'/Joined_fits/Intr_prof/chi2/Fit_results'}}
    data_prop = dataload_npz(opt_dic['IntrProf_prop_path'][inst][vis])
    params=data_prop['p_final'] 
    fixed_args={  
        'mode':opt_dic['mode'],
        'type':data_dic[inst][vis]['type'],
        'nord':data_dic[inst]['nord'],
        'nthreads': gen_opt_dic['nthreads'],
        'resamp_mode' : gen_dic['resamp_mode'], 
        'inst':inst,
        'vis':vis,  
        'fit':False,
    } 
    if fixed_args['mode']=='ana':
        fixed_args.update({  
            'mac_mode':theo_dic['mac_mode'], 
            'coeff_line':data_prop['coeff_line_dic'][inst][vis],
            'func_prof_name':data_prop['func_prof_name'][inst]
        })        
        for key in ['coeff_ord2name','pol_mode','coord_line','linevar_par']:fixed_args[key] = data_prop[key]
    if data_dic['Intr']['plocc_prof_type']=='Intr':fixed_args['conv2intr'] = True
    else:fixed_args['conv2intr'] = False
    chrom_mode = data_dic['DI']['system_prop']['chrom_mode']

    #Activation of spectral conversion and resampling 
    cond_conv_st_prof_tab(theo_dic['rv_osamp_line_mod'],fixed_args,data_dic[inst][vis]['type']) 

    #Processing in-transit exposures
    for isub,i_in in enumerate(range(data_dic[inst][vis]['n_in_tr'])):
      
        #Upload spectral tables from residual or intrinsic profile of current exposure
        if data_dic['Intr']['plocc_prof_type']=='Intr':iexp_eff = i_in
        else:iexp_eff = gen_dic[inst][vis]['idx_in2exp'][i_in]
        data_loc_exp = dataload_npz(data_vis['proc_'+data_dic['Intr']['plocc_prof_type']+'_data_paths']+str(iexp_eff))

        #Limit model table to requested definition range
        if len(gen_opt_dic['def_range'])==0:cond_calc_pix = np.arange(data_dic[inst][vis]['nspec'] ,dtype=bool)    
        else:cond_calc_pix = (data_loc_exp['edge_bins'][def_iord][0:-1]>=gen_opt_dic['def_range'][0]) & (data_loc_exp['edge_bins'][def_iord][1:]<=gen_opt_dic['def_range'][1])             
        idx_calc_pix = np_where1D(cond_calc_pix)

        #Final table for model line profile
        fixed_args['ncen_bins']=len(idx_calc_pix)
        fixed_args['dim_exp'] = [1,fixed_args['ncen_bins']] 
        fixed_args['cen_bins'] = data_loc_exp['cen_bins'][def_iord,idx_calc_pix]
        fixed_args['edge_bins']=data_loc_exp['edge_bins'][def_iord,idx_calc_pix[0]:idx_calc_pix[-1]+2]
        fixed_args['dcen_bins']=fixed_args['edge_bins'][1::] - fixed_args['edge_bins'][0:-1] 

        #Initializing stellar profiles
        #    - can be defined using the first exposure table
        if isub==0:
            fixed_args = init_custom_DI_prof(fixed_args,gen_dic,data_dic['DI']['system_prop'],theo_dic,system_param['star'],params)                  

            #Effective instrumental convolution
            fixed_args['FWHM_inst'] = ref_inst_convol(inst,fixed_args,fixed_args['cen_bins'])

        #Resampled spectral table for model line profile
        if fixed_args['resamp']:resamp_model_st_prof_tab(None,None,None,fixed_args,gen_dic,1,theo_dic['rv_osamp_line_mod'])
        
        #Table for model calculation
        args_exp = def_st_prof_tab(None,None,None,fixed_args)      

        #Define broadband scaling of intrinsic profiles into local profiles
        if data_dic['Intr']['plocc_prof_type']=='Res':
            theo_intr2loc(fixed_args['grid_dic'],fixed_args['system_prop'],args_exp,args_exp['ncen_bins'],fixed_args['grid_dic']['nsub_star']) 

        #Planet-occulted line profile 
        surf_prop_dic = sub_calc_plocc_prop([chrom_mode],args_exp,['line_prof'],data_dic[inst][vis]['transit_pl'],deepcopy(system_param),theo_dic,fixed_args['system_prop'],params,coord_dic[inst][vis],[gen_dic[inst][vis]['idx_in2exp'][i_in]],False)
        sp_line_model = surf_prop_dic[chrom_mode]['line_prof'][:,0]

        #Scaling to fitted intrinsic continuum level
        if (data_dic['Intr']['plocc_prof_type']=='Intr'):sp_line_model*=params['cont']   
 
        #Conversion and resampling 
        flux_loc = conv_st_prof_tab(None,None,None,fixed_args,args_exp,sp_line_model,fixed_args['FWHM_inst'])
        
        #Filling full table with defined reconstructed profile
        flux_full = np.zeros(data_dic[inst][vis]['nspec'],dtype=float)*np.nan
        flux_full[idx_calc_pix] = flux_loc
        cond_def_full = np.zeros(data_dic[inst][vis]['nspec'],dtype=bool)
        cond_def_full[idx_calc_pix] = True

        #Saving estimate of local profile for current exposure                   
        np.savez_compressed(gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['mode_loc_data_corr']+'/'+inst+'_'+vis+'_'+str(i_in),
                            data={'cen_bins':data_loc_exp['cen_bins'],'edge_bins':data_loc_exp['edge_bins'],'cond_def':np.array([cond_def_full]),'flux' : np.array([flux_full])},allow_pickle=True)

    #Complementary data
    data_add={'idx_est_loc':range(data_dic[inst][vis]['n_in_tr']),'cont':params['cont']}

    return data_add


'''
Sub-function for local profiles using individual CCF profile model
    - we use the models fitted, outside of the range affected by the planetary atmosphere, to each intrinsic profile
      these models correspond directly to the measured profile and needs only be rescaled to the level of the local profile
      this approach only works for exposures in which the stellar line could be fitted well after excluding the planet-contaminated range
'''
def loc_prof_indivmod(inst,vis,gen_dic,data_dic):
    data_vis = data_dic[inst][vis]
    if data_vis['type']!='CCF':stop('Method not valid for spectra')

    #Upload fit results
    data_prop=(np.load(gen_dic['save_data_dir']+'Introrig_prop/'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item())['prof_fit_dic']
    idx_aligned = np_where1D(data_prop['cond_detected'])

    #Processing in-transit exposures
    for i_in in idx_aligned:
     
        #Model intrinsic profile
        data_est_loc={'cen_bins':data_prop[i_in]['cen_bins'],'edge_bins':data_prop[i_in]['edge_bins'],'flux':data_prop[i_in]['flux'][:,None],'cond_def':np.ones(data_dic[inst][vis]['dim_exp'],dtype=bool)}

        #Rescaling model intrinsic profile to the level of the local profile
        if data_dic['Intr']['plocc_prof_type']=='Res':
            loc_flux_scaling = dataload_npz(data_vis['scaled_Intr_data_paths']+str(gen_dic[inst][vis]['idx_in2exp'][i_in]))['loc_flux_scaling']
            data_est_loc['flux'][0] *= loc_flux_scaling(data_est_loc['cen_bins'][0]) 

        #Saving estimate of local profile for current exposure                   
        np.savez_compressed(gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['mode_loc_data_corr']+'/'+inst+'_'+vis+'_'+str(i_in),data=data_est_loc,allow_pickle=True)

    #Complementary data
    data_add={'idx_est_loc':idx_aligned}
      
    return None


'''
Sub-function for local profiles defining undefined pixels via a polynomial fit to defined pixels in complementary exposures, or via a 2D interpolation over complementary exposures and a narrow spectral band
    - providing the planetary track shifts sufficiently during the transit, we can reconstruct the local spectra at all wavelengths by interpolating the surrounding spectra at the wavelengths masked for planetary absorption
      this approach allows accounting for changes in the shape of the local stellar spectra between exposures 
    - we reconstruct undefined pixels in the map of intrinsic profiles aligned in the null rest frame and resampled on the common spectral table
      for each exposure, we then shift the corresponding reconstructed profile to the stellar surface velocity associated with the exposure, and resample it on the exposure table
      ideally the map should be first aligned at the stellar surface velocity of each exposure, and then resampled on its table, but that would take too much time and the error is likely lower than that introduced by the interpolation
    - no errors are propagated onto the new pixels
    - at defined pixels, the local profile and its best estimate for the intrinsic profile will have the same values, resulting in null values in the atmospheric profiles
      there will be small differences due to the need to resample the estimate on a different table than the corresponding local profile
'''
def loc_prof_rec(inst,vis,gen_dic,data_dic,coord_dic):
    opt_dic = data_dic['Intr']['opt_loc_data_corr']['rec_prof']
    data_inst = data_dic[inst]
    data_vis = data_inst[vis]
    if not gen_dic['align_Intr']:stop('Intrinsic stellar profiles must have been aligned')

    #Upload common spectral table for processed visit
    data_com = np.load(data_inst[vis]['proc_com_data_paths']+'.npz',allow_pickle=True)['data'].item()  

    #Using current visit exposures only, or exposures from multiple visits
    if (inst in opt_dic['vis_in_rec']) and (len(opt_dic['vis_in_rec'][inst])>0):vis_in_rec = opt_dic['vis_in_rec'][inst]     
    else:vis_in_rec = [vis]  

    #Initializing reconstruction
    y_cen_all = np.zeros(0,dtype=float)
    idx_orig_vis = np.zeros(0,dtype=int)
    vis_orig=np.zeros(0,dtype='U35')
    n_in_rec = 0
    for vis_rec in vis_in_rec:

        #Original indexes of aligned intrinsic profiles in current visit, which can potentially contribute to the reconstruction
        #    - relative to in-transit tables
        ref_pl,dic_rv_rec,_ = init_surf_shift(gen_dic,inst,vis_rec,data_dic,data_dic['Intr']['align_mode'])
            
        #Limiting aligned intrinsic profiles to input selection
        idx_in_rec = data_dic['Intr'][inst][vis]['idx_def']
        if (inst in opt_dic['idx_in_rec']) and (vis_rec in opt_dic['idx_in_rec'][inst]):
            idx_in_rec = np.intersect1d(opt_dic['idx_in_rec'][inst][vis_rec],)
            if len(idx_in_rec)==0:stop('No remaining exposures after input selection')  
        n_to_rec = len(idx_in_rec)                   

        #Tables along the chosen fit/interpolation dimension
        if opt_dic['dim_bin']=='phase':    
            y_cen_vis_rec = coord_dic[inst][vis_rec]['cen_ph'][gen_dic[inst][vis_rec]['idx_in']]   
        elif opt_dic['dim_bin'] in ['xp_abs','r_proj']: 
            transit_prop_nom = (np.load(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis_rec+'.npz',allow_pickle=True)['data'].item())['achrom'][ref_pl]                           
            y_cen_vis_rec = transit_prop_nom[opt_dic['dim_bin']][0,:]
        y_cen_all=np.append(y_cen_all,y_cen_vis_rec[idx_in_rec])
        
        #Store properties for visit to reconstruct
        if vis_rec==vis:
            dic_rv=deepcopy(dic_rv_rec)
            idx_aligned=deepcopy(idx_in_rec)
            
            #Indexes of processed visit within data_for_rec tables
            isub_vis_to_rec = n_in_rec+np.arange(n_to_rec)

            #Coordinates of exposures in processed visit
            y_cen_vis = y_cen_vis_rec[idx_in_rec]        

        #Store exposure identifiers
        n_in_rec+=n_to_rec                                          #total number of contributing exposures
        idx_orig_vis = np.append(idx_orig_vis,idx_in_rec)                   #original indexes of contributing exposures in their respective in-transit tables
        vis_orig = np.append(vis_orig,np.repeat(vis_rec,len(idx_in_rec)))   #original visits of contributing exposures

    #Upload contributing and processed intrinsic profiles
    #    - data must be put in global tables to perform the fits and interpolations  
    #    - the fits cannot deal with covariance matrix along the dimension orthogonal to the spectral one, thus we only use the diagonal
    data_for_rec={}
    data_rec={}
    for key in ['flux','err']:data_for_rec[key]=np.zeros([n_in_rec]+data_vis['dim_exp'], dtype=float)*np.nan
    data_for_rec['cond_def']=np.zeros([n_in_rec]+data_vis['dim_exp'], dtype=bool)
    isub_to_rec=0
    for isub_rec,(vis_rec,iexp_rec) in enumerate(zip(vis_orig,idx_orig_vis)):
        data_exp = np.load(gen_dic['save_data_dir']+'Aligned_Intr_data/'+inst+'_'+vis_rec+'_'+str(iexp_rec)+'.npz',allow_pickle=True)['data'].item()    
    
        #Resampling aligned intrinsic profiles on the common spectral table of the processed visit
        #    - if the processed and reconstructed visits are the same and exposures do not share a common table
        #    - if the processed and reconstructed visits are not the same, and visits do not share a common table
        if ((vis_rec==vis) and (not data_vis['comm_sp_tab'])) or ((vis_rec!=vis) and (not data_inst['comm_sp_tab'])): 
            for iord in range(data_inst['nord']):
                data_for_rec['flux'][isub_rec,iord],cov_ord = bind.resampling(data_com['edge_bins'][iord],data_exp['edge_bins'][iord], data_exp['flux'][iord] , cov =  data_exp['cov'][iord], kind=gen_dic['resamp_mode']) 
                data_for_rec['err'][isub_rec,iord] = np.sqrt(cov_ord[0])
            data_for_rec['cond_def'][isub_rec] = ~np.isnan(data_for_rec['flux'][isub_rec])     

        #Aligned intrinsic profiles are already defined on a common table for all visits
        else:
            for key in ['flux','cond_def']:data_for_rec[key][isub_rec] = data_exp[key]
            for iord in range(data_inst['nord']):data_for_rec['err'][isub_rec,iord] = np.sqrt(data_exp['cov'][iord][0])
            
        #Initialize reconstructed profiles table for processed visit               
        if (vis_rec==vis):
            data_rec[isub_to_rec]={
                'cen_bins':data_com['cen_bins'],
                'edge_bins':data_com['edge_bins'],
                'flux':data_for_rec['flux'][isub_rec]}
            isub_to_rec+=1

    #Process each order 
    for iord in range(data_dic[inst]['nord']):
        
        #Identify pixels to reconstruct in at least one exposure of the processed visit
        idx_undef_pix = np_where1D(np.sum(data_for_rec['cond_def'][isub_vis_to_rec,iord,:],axis=0)<n_to_rec) 

        #Process undefined pixels
        for idx_pix in idx_undef_pix: 

            #Exposures for which the pixel is defined in contributing exposures from all visits
            cond_def_pix_exp = data_for_rec['cond_def'][:,iord,idx_pix]

            #Exposures for which the pixel is undefined in exposures of processed visit
            idx_undef_pix_vis = np_where1D( data_for_rec['cond_def'][isub_vis_to_rec,iord,idx_pix] )
            
            #------------------------------------------------------------------------------------------------------------------------
                                         
            #Pixel-per-pixel interpolation
            #    - for each undefined pixel, we perform a polynomial fit along the chosen dimension over defined pixels
            if opt_dic['rec_mode']=='pix_polfit':

                #Fit polynomial along the chosen dimension    
                #    - only if more than two pixels are defined
                if np.sum(cond_def_pix_exp)>2:
                    func_def_pix = np.poly1d(np.polyfit(y_cen_all[cond_def_pix_exp], data_for_rec['flux'][cond_def_pix_exp,iord,idx_pix],opt_dic['pol_deg'],w=1./data_for_rec['err'][cond_def_pix_exp,iord,idx_pix]))

                    #Calculate polynomial in exposures of processed visit for which the pixel is undefined
                    val_def_pix = func_def_pix(y_cen_vis[idx_undef_pix_vis])
                    for isub_to_rec,val_def_pix_loc in zip(idx_undef_pix_vis,val_def_pix):
                        data_rec[isub_to_rec]['flux'][iord,idx_pix] = val_def_pix_loc   

            #------------------------------------------------------------------------------------------------------------------------
                        
            #2D band interpolation
            elif opt_dic['rec_mode']=='band_interp':  
                
                #Pixels to interpolate
                #    - we use the pixels defined over complementary exposures in a band surrounding the current pixel
                #    - tables must be put in 1D
                idx_band = np.arange(max(0,idx_pix-opt_dic['band_pix_hw']),min(data_vis['nspec']-1,idx_pix+opt_dic['band_pix_hw'])+1)
                nx_band = len(idx_band)
                cond_def_band = data_for_rec['cond_def'][:,iord,idx_band].flatten()      #defined pixels in 1D flux table within the band
                if np.sum(cond_def_band)>0:
                    flux_band = (data_for_rec['flux'][:,iord,idx_band].flatten())[cond_def_band]                       
                    x_cen_band = np.repeat(data_com['cen_bins'][iord,idx_band],n_in_rec)[cond_def_band]
                    y_cen_band = np.tile(y_cen_all,[nx_band])[cond_def_band]

                    #Tables must have the same scaling because the interpolation is done in euclidian norm
                    xmin_sc=np.min(x_cen_band)
                    ymin_sc=np.min(y_cen_band)
                    xscale=np.max(x_cen_band)-xmin_sc
                    yscale=np.max(y_cen_band)-ymin_sc
                    if (xscale>0) and (yscale>0):

                        #Pixels to reconstruct in processed visit
                        n_undef_pix_exp_vis = len(idx_undef_pix_vis)
                        x_cen_rec = np.repeat(data_com['cen_bins'][iord,idx_pix],n_undef_pix_exp_vis)
                        y_cen_rec = y_cen_vis[idx_undef_pix_vis]
                        
                        #Scaling
                        x_cen_band_sc=(x_cen_band-xmin_sc)/xscale
                        y_cen_band_sc=(y_cen_band-ymin_sc)/yscale    
                        x_cen_rec_sc=(x_cen_rec-xmin_sc)/xscale
                        y_cen_rec_sc=(y_cen_rec-ymin_sc)/yscale     
 
                        #Interpolating
                        val_def_pix = griddata((x_cen_band_sc,y_cen_band_sc), flux_band, (x_cen_rec_sc, y_cen_rec_sc),method=opt_dic['interp_mode'])
                        for isub_to_rec,val_def_pix_loc in zip(idx_undef_pix_vis,val_def_pix):
                            data_rec[isub_to_rec]['flux'][iord,idx_pix] = val_def_pix_loc

    #------------------------------------------------------------------------------------------------------------------------

    #Processing in-transit exposures with reconstructed intrinsic profiles
    for isub,i_in in enumerate(idx_aligned):

        #Upload residual or intrinsic profile for current exposure to get its spectral tables
        if data_dic['Intr']['plocc_prof_type']=='Intr':iexp_eff = i_in
        else:iexp_eff = gen_dic[inst][vis]['idx_in2exp'][i_in]
        data_loc_exp = dataload_npz(data_vis['proc_'+data_dic['Intr']['plocc_prof_type']+'_data_paths']+str(iexp_eff))  

        #Radial velocity shifts set to the opposite of the planet-occulted surface rv associated with current exposure
        surf_shifts,surf_shifts_edge = def_surf_shift(data_dic['Intr']['align_mode'],dic_rv,i_in,data_rec[isub],ref_pl,data_vis['type'],data_dic['DI']['system_prop'],data_dic[inst][vis]['dim_exp'],data_dic[inst]['nord'],data_dic[inst][vis]['nspec'])

        #Aligning reconstructed profile at current exposure stellar surface rv 
        #    - reconstructed profile is already aligned in a null rest frame and resampled on a common table
        #    - aligned profiles are resampled on the table of current exposure, which is common to all exposures if a common table is used 
        if surf_shifts_edge is not None:surf_shifts_edge*= -1.
        data_est_loc=align_data(data_rec[isub],data_vis['type'],data_dic[inst]['nord'],data_dic[inst][vis]['dim_exp'],gen_dic['resamp_mode'],data_loc_exp['cen_bins'],data_loc_exp['edge_bins'],-surf_shifts,rv_shift_edge = surf_shifts_edge ,nocov=True)

        #Rescaling reconstructed intrinsic profile to the level of the local profile
        if data_dic['Intr']['plocc_prof_type']=='Res':
            loc_flux_scaling = dataload_npz(data_vis['scaled_Intr_data_paths']+str(gen_dic[inst][vis]['idx_in2exp'][i_in]))['loc_flux_scaling']
            for iord in range(data_dic[inst]['nord']):data_est_loc['flux'][iord] *=loc_flux_scaling(data_est_loc['cen_bins'][iord]) 

        #Saving estimate of local profile for current exposure                   
        np.savez_compressed(gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['mode_loc_data_corr']+'/'+inst+'_'+vis+'_'+str(i_in),data=data_est_loc,allow_pickle=True)


    #Complementary data
    data_add={'idx_est_loc':idx_aligned}

    return None













                                                                       







    


'''
Extract atmospheric signals
'''
def extract_pl_profiles(data_dic,inst,vis,gen_dic):

    print('   > Extracting atmospheric stellar profiles')
    data_vis = data_dic[inst][vis]
    plAtm_vis = data_dic['Atm'][inst][vis]

    #Current rest frame
    if data_dic['Res'][inst][vis]['rest_frame']!='star':print('WARNING: residual profiles must be aligned')
    data_dic['Atm'][inst][vis]['rest_frame'] = 'star'

    #Indexes of in-transit exposures with defined estimates of local stellar profiles
    idx_est_loc = (np.load(gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['mode_loc_data_corr']+'/'+inst+'_'+vis+'_add.npz',allow_pickle=True)['data'].item())['idx_est_loc']

    #Indexes of exposures with retrieved signal
    if (data_dic['Atm']['pl_atm_sign']=='Absorption'):
        #In-transit indexes
        plAtm_vis['idx_def'] = idx_est_loc
        #Corresponding global indexes
        iexp_glob = np.array(gen_dic[inst][vis]['idx_in'])[idx_est_loc]

    elif (data_dic['Atm']['pl_atm_sign']=='Emission'):
        #Global indexes
        plAtm_vis['idx_def'] = list(gen_dic[inst][vis]['idx_out']) + list(np.array(gen_dic[inst][vis]['idx_in'])[idx_est_loc])
        iexp_glob = plAtm_vis['idx_def']
         
    #Initializing path to atmospheric data
    data_vis['proc_Atm_data_paths']=gen_dic['save_data_dir']+'Atm_data/'+data_dic['Atm']['pl_atm_sign']+'/'+inst+'_'+vis+'_'
                
    #Initialize path of weighing profiles for atmospheric exposures
    #    - the weighing profiles include the master disk-integrated profile, and the best estimates of the local stellar profiles (if measured)
    #    - best estimates are only defined for in-transit profiles, and paths must be defined relative to the indexes used to call atmpospheric profiles
    #    - weighing master is defined on the common table for the visit
    if (data_dic['Atm']['pl_atm_sign']=='Absorption') or ((data_dic['Atm']['pl_atm_sign']=='Emission') and data_dic['Intr']['cov_loc_star']):
        data_vis['LocEst_Atm_data_paths'] = {}
        if (data_dic['Atm']['pl_atm_sign']=='Absorption'):iexp_paths = idx_est_loc     #in-transit indexes
        elif (data_dic['Atm']['pl_atm_sign']=='Emission'):iexp_paths = np.array(gen_dic[inst][vis]['idx_in'])[idx_est_loc]    #global indexes, limited to in-transit values
        data_vis['LocEst_Atm_data_paths'] = {iexp:gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['mode_loc_data_corr']+'/'+inst+'_'+vis+'_'+str(i_in) for iexp,i_in in zip(iexp_paths,idx_est_loc)}
    
    #Calculating
    if (gen_dic['calc_pl_atm']):
        print('         Calculating data')     

        #Data for absorption signal
        if (data_dic['Atm']['pl_atm_sign']=='Absorption'):

            #Properties of planet-occulted regions
            dic_plocc_prop = np.load(gen_dic['save_data_dir']+'Introrig_prop/PlOcc_Prop_'+inst+'_'+vis+'.npz',allow_pickle=True)['data'].item()     
                     
        #Process all exposures
        for iexp,i_in in zip(range(data_vis['n_in_visit']),gen_dic[inst][vis]['idx_exp2in']):

            #Upload local profile
            #    - the local stellar profiles defined in the star rest frame write as 
            # Fres(w,t,vis) = ( fp(w,vis)*(Sthick(band,t) + Sthin(w-wp,t) ) -  Fatm(w-wp,t) )*Cref(band)/Fr(vis) 
            #      we want to retrieve the atmospheric emission signal Fatm, and the atmospheric absorption surface Sthin
            data_loc = np.load(data_vis['proc_Res_data_paths']+str(iexp)+'.npz',allow_pickle=True)['data'].item()             
                
            #Upload estimate of local stellar profile for in-transit exposures
            #    - we distinguish between theoretical estimates and ones derived from data in the calculation of the covariance below 
            if (i_in>-1) and (i_in in idx_est_loc):data_loc_star = np.load(gen_dic['save_data_dir']+'Loc_estimates/'+data_dic['Intr']['mode_loc_data_corr']+'/'+inst+'_'+vis+'_'+str(i_in)+'.npz',allow_pickle=True)['data'].item()   
                        
            #Extraction of emission signal
            #    - out-of-transit we take the opposite of the local profiles to retrieve the emission signal
            # F_em(w,t,vis) = - Fres(w,t,vis) 
            #               = Fatm(w-wp,t)*Cref(band)/Fr(vis)  
            #    - in-transit, the estimates of local stellar profiles correspond to 
            # Fstar_loc(w,t) = fp(w,vis)*Sthick(band,t)*Cref(band)/Fr(vis) 
            #      and we define the emission signal as    
            # F_em(w,t,vis) = Fstar_loc(w,t) - Fres(w,t,vis) 
            #               = Fatm(w-wp,t)*Cref(band)/Fr(vis) - fp(w,vis)*Sthin(w-wp,t )*Cref(band)/Fr(vis)
            #      thus the emission signal will remain contaminated if there is an absorption signal
            #    - unless the exact stellar SED for each visit was used to correct for the color balance, and to set spectra to the correct overall flux level in each visit, the
            # emission signals will be known to a scaling factor Cref(band)/Fr(vis)                                                                         
            if (data_dic['Atm']['pl_atm_sign']=='Emission') and (iexp in plAtm_vis['idx_def']):
                data_em = {'cen_bins':data_loc['cen_bins'],'edge_bins':data_loc['edge_bins'],'tell':data_loc['tell']}              
                
                #Out-of-transit signal
                if (i_in == -1):
                    data_em['flux'] = -data_loc['flux']
                    for key in ['cov','cond_def']:data_em[key] = data_loc[key]

                #In-transit signal
                #    - in-transit exposures with no estimates of local stellar profiles cannot be retrieved
                elif (i_in in idx_est_loc):
                    if not data_dic['Intr']['cov_loc_star']:
                        data_em['flux'] = data_loc_star['flux']-data_loc['flux']
                        data_em['cov'] = data_loc['cov']
                    else:
                        data_em['flux'] = np.zeros(data_vis['dim_exp'], dtype=float)
                        data_em['cov'] = np.zeros(data_dic[inst]['nord'], dtype=object)
                        for iord in range(data_dic[inst]['nord']):
                            data_em['flux'][iord],data_em['cov'][iord]=bind.add(data_loc_star['flux'][iord], data_loc_star['cov'][iord],-data_loc['flux'][iord], data_loc['cov'][iord])                 
                    data_em['cond_def'] = ~np.isnan(data_em['flux'])                 

                #Saving data               
                np.savez_compressed(data_vis['proc_Atm_data_paths']+str(iexp),data=data_em,allow_pickle=True)
                  
            #------------------------------------------------------------------------------------------------------------

            #Extraction of absorption signal
            #    - the absorption signal in-transit is retrieved as
            # Abs(w,t,vis) = ( F_res(w,t,vis) - Fstar_loc(w,t,vis) ) / ( Fstar_loc(w,t,vis)/( Sthick(band,t)/Sstar ) )                    
            #      subtracting Fstar_loc(w,t,vis) removes the local stellar profile absorbed by the planetary continuum
            #      dividing by Fstar_loc(w,t,vis) then removes the contribution of the local stellar profile
            #      rescaling by Sthick(band,t)/Sstar finally replaces the scaling of the planet-occulted region surface by the full stellar surface, so that the result is comparable with classical absorption signal
            #              = ( [ ( fp(w,vis)*(Sthick(band,t) + Sthin(w-wp,t) ) -  Fatm(w-wp,t) )*Cref(band)/Fr(vis) ]  - [ fp(w,vis)*Sthick(band,t)*Cref(band)/Fr(vis) ]  ) / ( [ fp(w,vis)*Sthick(band,t)*Cref(band)/Fr(vis) ]/( Sthick(band,t)/Sstar ) )    
            #              = ( ( fp(w,vis)*(Sthick(band,t) + Sthin(w-wp,t) ) -  Fatm(w-wp,t) )  -  fp(w,vis)*Sthick(band,t)  ) / ( fp(w,vis)*Sstar )                     
            #              = ( fp(w,vis)*Sthin(w-wp,t) -  Fatm(w-wp,t) )/ ( fp(w,vis)*Sstar )                     
            #              = Sthin(w-wp,t)/Sstar -  Fatm(w-wp,t)/( fp(w,vis)*Sstar )   
            #    - the calculation sums up as:
            # Abs(w,t,vis) = [ F_res(w,t,vis)/Fstar_loc(w,t,vis)  - 1 ]*( Sthick(band,t)/Sstar )
            #    - the absorption signal can be contaminated by an emission signal, however the latter likely varies over the planet orbit (thus preventing us to compute an out-of-transit master to be corrected from 
            # in-transit exposures) and is unlikely to be visible during transit in any case as it would arise from nightside emission
            #      if no emission is present, the corrected spectra correspond to:
            # Signal(w,t) = Sthin(w-wp,t )/Sstar
            #    - we use a numerical estimate of Sthick(band,t)/Sstar with a constant Sstar, so that we extract the pure atmospheric spectral surface, normalized
            # by a constant stellar surface to be equivalent to an absorption signal, but unbiased by a spectral stellar surface
            #    - if we consider that there is an absorption signal outside of the transit defined by the input light curve, ie that a region of the planetary atmosphere is absorbing
            # in a specific line but not in the continuum:
            # F_res(w,t,vis) = ( fp(w,vis)*Sthin(w-wp,t) -  Fatm(w-wp,t) )*Cref(band)/Fr(vis) 
            if (data_dic['Atm']['pl_atm_sign']=='Absorption') and (i_in>-1) and (i_in in idx_est_loc):

                #Planet-to-star surface ratios
                SpSstar_spec = np.zeros([data_dic[inst]['nord'],data_vis['nspec']],dtype=float)   
                                          
                #Achromatic/chromatic planet-to-star radius ratio
                #    - for now a single transiting planet is considered
                if ('spec' in data_vis['type']) and ('chrom' in data_dic['DI']['system_prop']):SpSstar_chrom = True
                else:
                    if len(data_vis['transit_pl'])>1:stop()
                    SpSstar_spec[:,:] = dic_plocc_prop['achrom'][data_vis['transit_pl'][0]]['SpSstar'][0,i_in] 
                    SpSstar_chrom = False
             
                #Processing each order
                data_abs = {'cen_bins':data_loc['cen_bins'],'edge_bins':data_loc['edge_bins'],
                            'flux' : np.zeros(data_vis['dim_exp'], dtype=float),
                            'cov' : np.zeros(data_dic[inst]['nord'], dtype=object)}
                for iord in range(data_dic[inst]['nord']):
                    
                    #Chromatic planet-to-star radius ratio
                    if SpSstar_chrom: 
                        SpSstar_spec[iord] = np_interp(data_loc['cen_bins'][iord],data_dic['DI']['system_prop']['chrom']['w'],dic_plocc_prop['chrom']['SpSstar'][:,i_in],left=dic_plocc_prop['chrom']['SpSstar'][0,i_in],right=dic_plocc_prop['chrom']['SpSstar'][-1,i_in])

                    #Calculation of absorption signal                                            
                    if data_dic['Intr']['cov_loc_star']:dat_temp,cov_temp = bind.div(data_loc['flux'][iord],data_loc['cov'][iord],data_loc_star['flux'][iord], data_loc_star['cov'][iord])
                    else:dat_temp,cov_temp = bind.mul_array(data_loc['flux'][iord],data_loc['cov'][iord],1./data_loc_star['flux'][iord])
                    data_abs['flux'][iord],data_abs['cov'][iord] = bind.mul_array(dat_temp-1.,cov_temp,SpSstar_spec[iord])
                data_abs['cond_def'] = ~np.isnan(data_abs['flux'])                                          
                data_abs['SpSstar_spec'] = SpSstar_spec   
                    
                #Saving data               
                np.savez_compressed(data_vis['proc_Atm_data_paths']+str(i_in),data=data_abs,allow_pickle=True)
                
    #------------------------------------------------------------------------------------------------------------
                
    #Checking that local data has been calculated for all exposures
    else:
        data_paths={iexp:data_vis['proc_Atm_data_paths']+str(iexp) for iexp in plAtm_vis['idx_def']}        
        check_data(data_paths)

    #Path to associated tables
    #    - atmospheric profiles are extracted in the same frame as residual profiles
    #    - indexes may be limited to in-transit indexes if absorption signals are extracted
    if gen_dic['DImast_weight']:data_vis['mast_Atm_data_paths'] = {}
    if data_vis['tell_sp']:data_vis['tell_Atm_data_paths'] = {}
    if data_vis['mean_gdet']:data_vis['mean_gdet_Atm_data_paths'] = {}
    for iexp_atm,iexp in zip(plAtm_vis['idx_def'],iexp_glob):
        if gen_dic['DImast_weight']:data_vis['mast_Atm_data_paths'][iexp_atm] = data_dic[inst][vis]['mast_Res_data_paths'][iexp] 
        if data_vis['tell_sp']:data_vis['tell_Atm_data_paths'][iexp_atm] = data_vis['tell_Res_data_paths'][iexp] 
        if data_vis['mean_gdet']:data_vis['mean_gdet_Atm_data_paths'][iexp_atm] = data_vis['mean_gdet_Res_data_paths'][iexp] 

    return None    













'''
Conversion of 2D spectra into 1D spectra
    - 2D spectra are resampled over the common table and coadded
    - spectral values from different orders have to be equivalent at a given wavelength
    - conversion is applied to the latest processed data of each type
    - converted data is saved independently, but used as default data in all modules following the conversion
'''
def init_conversion(data_type_gen,gen_dic,prop_dic,inst,vis,mode,dir_save):
    txt_print = {'CCFfromSpec':'CCF','1Dfrom2D':'1D'}[mode]
    if data_type_gen in ['DI','Intr']:    
        data_type = data_type_gen
        if data_type_gen=='Intr':
            iexp_conv = list(gen_dic[inst][vis]['idx_out'])+list(gen_dic[inst][vis]['idx_in'][prop_dic[inst][vis]['idx_def']])    #Global indexes
            data_type_key = ['Res','Intr']
            print('   > Converting OT residual and intrinsic spectra into '+txt_print)
        else:iexp_conv = range(data_vis['n_in_visit'])    #Global indexes
    if data_type_gen in ['DI','Atm']:  
        if data_type_gen=='Atm': 
            data_type = prop_dic['pl_atm_sign']
            iexp_conv = prop_dic[inst][vis]['idx_def']    #Global or in-transit indexes
        data_type_key = [data_type_gen]
        print('   > Converting '+gen_dic['type_name'][data_type]+' spectra into '+txt_print)
    for key in data_type_key:dir_save[key] = gen_dic['save_data_dir']+key+'_data/'+mode+'/'+gen_dic['add_txt_path'][key]+'/'+inst+'_'+vis+'_'    
    
    return iexp_conv,data_type_key,data_type

def conv_2D_to_1D_spec(data_type_gen,inst,vis,gen_dic,data_dic,prop_dic):
    data_vis=data_dic[inst][vis]
    gen_vis=gen_dic[inst][vis]
    dir_save = {} 
    iexp_conv,data_type_key,data_type = init_conversion(data_type_gen,gen_dic,prop_dic,inst,vis,'1Dfrom2D',dir_save)

    #Paths
    proc_com_data_paths_new = gen_dic['save_data_dir']+'Processed_data/spec1D_'+inst+'_'+vis+'_com'
    
    #Calculating data
    if gen_dic['calc_spec_1D_'+data_type_gen]:
        print('         Calculating data')
        nthreads = gen_dic['nthreads_spec_1D_'+data_type_gen]
        if data_vis['type']!='spec2D':stop('Data must be 2D')
        nspec_1D = prop_dic['spec_1D_prop'][inst]['nspec']
        cen_bins_1D = prop_dic['spec_1D_prop'][inst]['cen_bins']
        edge_bins_1D = prop_dic['spec_1D_prop'][inst]['edge_bins']

        #Associated tables  
        proc_data_paths = {}
        scaling_data_paths = {} if gen_dic['flux_sc'] else None
        tell_data_paths =  {} if data_vis['tell_sp'] else None
        if gen_dic['cal_weight'] and data_vis['mean_gdet']:
            proc_weight=True
            mean_gdet_data_paths = {}
        else:
            proc_weight=False
            mean_gdet_data_paths = None
        DImast_weight_data_paths = {} if gen_dic['DImast_weight'] else None
        for key in data_type_key:   
            proc_data_paths[key] = data_vis['proc_'+key+'_data_paths']
            if scaling_data_paths is not None:scaling_data_paths[key] = data_vis['scaled_'+key+'_data_paths']
            if tell_data_paths is not None:tell_data_paths[key] = data_vis['tell_'+key+'_data_paths']
            if mean_gdet_data_paths is not None:mean_gdet_data_paths[key] = data_vis['mean_gdet_'+key+'_data_paths']
            if DImast_weight_data_paths is not None:DImast_weight_data_paths[key] = data_vis['mast_'+key+'_data_paths']
        LocEst_Atm_data_paths = data_vis['LocEst_Atm_data_paths'] if data_type_gen=='Atm' else None

        #Processing all exposures
        ifirst = iexp_conv[0]
        common_args = (data_type_gen,gen_dic['resamp_mode'],dir_save,cen_bins_1D,edge_bins_1D,nspec_1D,data_dic[inst]['nord'],ifirst,proc_com_data_paths_new,proc_weight,\
                       gen_dic[inst][vis]['idx_in2exp'],data_dic['Intr']['cov_loc_star'],proc_data_paths,tell_data_paths,scaling_data_paths,mean_gdet_data_paths,DImast_weight_data_paths,LocEst_Atm_data_paths,inst,vis,gen_dic['corr_Fbal'],gen_dic['corr_FbalOrd'],\
                       gen_dic['save_data_dir'],gen_dic['type'],data_vis['type'],data_vis['dim_exp'],gen_vis['idx_exp2in'],gen_vis['idx_in'])
        if nthreads>1: para_conv_2D_to_1D_exp(conv_2D_to_1D_exp,nthreads,len(iexp_conv),[iexp_conv],common_args)                           
        else: conv_2D_to_1D_exp(iexp_conv,*common_args)  

        #Saving complementary data
        for gen in data_type_key: 
            rest_frame = data_dic[gen][inst][vis]['rest_frame']         #Rest frame propagated from original data       
            data_add = {'rest_frame':rest_frame,'dim_exp':[1,prop_dic['spec_1D_prop'][inst]['nspec']]}
            if gen=='Intr':data_add['iexp_conv'] = prop_dic[inst][vis]['idx_def'] 
            elif gen=='Res':data_add['iexp_conv'] = gen_dic[inst][vis]['idx_out']
            else:data_add['iexp_conv'] = iexp_conv
            datasave_npz(dir_save[gen]+'add',data_add)  

    else: 
        check_data({'path':proc_com_data_paths_new})
        
    #Updating paths
    #    - scaling is defined as a function and does not need updating
    data_vis['proc_com_data_paths'] = proc_com_data_paths_new
    for gen in dir_save:
        data_vis['proc_'+gen+'_data_paths'] = dir_save[gen]  
        if data_vis['tell_sp']:data_vis['tell_'+gen+'_data_paths'] = {}        
        data_vis['mast_'+gen+'_data_paths'] = {}    
    if data_type_gen=='Atm':data_vis['LocEst_Atm_data_paths'] = {}
    for iexp in iexp_conv:
        gen = deepcopy(data_type_gen)
        iexp_eff = deepcopy(iexp) 
        if data_type_gen=='Intr':         
            if (iexp in gen_vis['idx_in']):iexp_eff = gen_vis['idx_exp2in'][iexp] 
            else:gen = 'Res'
        data_vis['mast_'+gen+'_data_paths'][iexp_eff] = dir_save[gen]+'ref_'+str(iexp_eff)
        if data_vis['tell_sp']:data_vis['tell_'+data_type_gen+'_data_paths'][iexp] = dir_save[gen]+'_tell'+str(iexp_eff)
        if data_type_gen=='Atm':data_vis['LocEst_Atm_data_paths'][iexp] = dir_save[gen]+'estloc_'+str(iexp_eff)
      
    #Convert spectral mode 
    print('         ANTARESS switched to 1D processing')
    data_vis['mean_gdet'] = False 
    data_vis['comm_sp_tab']=True
    data_vis['type']='spec1D'
    data_vis['nspec'] = prop_dic['spec_1D_prop'][inst]['nspec']
    data_dic[inst]['nord'] = 1
    data_vis['dim_all'] = [data_vis['n_in_visit'],1,data_vis['nspec']]
    data_vis['dim_exp'] = [1,data_vis['nspec']]
    data_vis['dim_sp'] = [data_vis['n_in_visit'],1]

    #Updating/correcting continuum level          
    data_dic['Intr'][inst][vis]['mean_cont']=calc_Intr_mean_cont(data_vis['n_in_tr'],data_dic[inst]['nord'],data_vis['nspec'],data_vis['proc_Intr_data_paths'],data_vis['type'],data_dic['Intr']['cont_range'],inst,data_dic['Intr']['cont_norm'])

    return None

def conv_2D_to_1D_exp(iexp_conv,data_type_gen,resamp_mode,dir_save,cen_bins_1D,edge_bins_1D,nspec_1D,nord,ifirst,proc_com_data_paths,proc_weight,\
                      idx_in2exp,cov_loc_star,proc_data_paths,tell_data_paths,scaling_data_paths,mean_gdet_data_paths,DImast_weight_data_paths,LocEst_Atm_data_paths,inst,vis,gen_corr_Fbal,gen_corr_Fbal_ord,\
                      save_data_dir,gen_type,data_mode,dim_exp,idx_exp2in,idx_in):
    
    #Processing each exposure
    #    - the conversion can be seen as the binning of all orders after they are resampled on a common table
    for iexp_sub,iexp in enumerate(iexp_conv):   
        data_type = deepcopy(data_type_gen)
        iexp_eff = deepcopy(iexp)     #Effective index (relative to global or in-transit tables)
        iexp_glob = deepcopy(iexp)    #Global index
        if data_type_gen=='DI':bdband_flux_sc = False
        else:
            bdband_flux_sc = True           
            if (data_type_gen=='Intr'):
                if (iexp in idx_in):
                    iexp_eff = idx_exp2in[iexp] 
                else:data_type = 'Res'
            elif data_type=='Absorption':iexp_glob = idx_in[iexp]
            
        #Upload spectra and associated tables in star or local frame
        flux_est_loc_exp = None
        cov_est_loc_exp = None
        SpSstar_spec = None    
        data_exp = dataload_npz(proc_data_paths[data_type]+str(iexp_eff))
        if scaling_data_paths is not None:data_scaling_exp = dataload_npz(scaling_data_paths[data_type]+str(iexp_eff))
        if tell_data_paths is not None:data_exp['tell'] = dataload_npz(tell_data_paths[data_type][iexp_eff])['tell'] 
        if proc_weight:data_exp['mean_gdet'] = dataload_npz(mean_gdet_data_paths[data_type][iexp_eff])['mean_gdet'] 
        if DImast_weight_data_paths is not None:data_ref = dataload_npz(DImast_weight_data_paths[data_type][iexp_eff])
        if data_type_gen=='Atm':
            data_est_loc=dataload_npz(LocEst_Atm_data_paths[iexp_eff])
            flux_est_loc_exp = data_est_loc['flux']
            if cov_loc_star:cov_est_loc_exp = data_est_loc['cov']                      
            if data_type=='Absorption':SpSstar_spec = data_exp['SpSstar_spec']

        #Weight definition
        #    - cannot be parallelized as functions cannot be pickled
        #    - here the binning is performed between overlapping orders of the same exposure 
        #      all profiles that are the same for overlapping orders (tellurics, disk-integrated stellar spectrum, global flux scaling, ...) are thus not used in the weighing 
        #      they are however processed in the same way as the exposure if used later on in the pipeline 
        #    - for intrinsic and atmospheric profiles we provide the broadband flux scaling, even if does not matter to the weighing, because it is otherwise set to 0 and messes up with weights definition
        data_exp['weights'] = def_weights_spatiotemp_bin(range(nord),scaling_data_paths[data_type],inst,vis,gen_corr_Fbal,gen_corr_Fbal_ord,save_data_dir,gen_type,nord,iexp_glob,data_type,data_mode,dim_exp,None,data_exp['mean_gdet'], data_exp['cen_bins'],1.,np.ones(dim_exp),np.zeros([dim_exp[0],1,dim_exp[1]]),corr_Fbal=False,bdband_flux_sc=bdband_flux_sc)

        #Resample spectra and weights on 1D table in each order, and clean weights
        flux_exp_all,cov_exp_all,cond_def_all,glob_weight_all,cond_def_binned = pre_calc_binned_prof(nord,[nspec_1D],range(nord),resamp_mode,None,data_exp,edge_bins_1D)

        #Processing each order
        flux_ord_contr=[]
        cov_ord_contr=[] 
        if scaling_data_paths is not None:loc_flux_scaling_ord_contr= np.zeros(nspec_1D, dtype=float)
        if tell_data_paths is not None:tell_ord_contr = np.zeros(nspec_1D, dtype=float)
        if DImast_weight_data_paths is not None:
            flux_ref_ord_contr=[]
            cov_ref_ord_contr=[] 
        if SpSstar_spec is not None:SpSstar_spec_ord_contr = np.zeros(nspec_1D, dtype=float)
        if flux_est_loc_exp is not None:
            flux_est_loc_ord_contr=[]
            if cov_est_loc_exp is not None:cov_est_loc_ord_contr=[]            
        for iord in range(nord):
         
            #Multiply by order weight and store
            flux_ord,cov_ord = bind.mul_array(flux_exp_all[iord] , cov_exp_all[iord] , glob_weight_all[iord])          
            flux_ord_contr+=[flux_ord]
            cov_ord_contr+=[cov_ord]
            cond_def = cond_def_all[iord]

            #Apply same steps to complementary spectra
            #    - calibration profiles are not consistent in the overlaps between orders, and are not used anymore
            #    - spectral scaling is not updated, since it is defined as a function and remains applicable to the 1D spectrum as with the 2D spectrum 
            #    - the master has followed the same shifts as the intrinsic or atmospheric profiles, but always remain either defined on the common table, or on a specific table different from the table of its associated exposure
            # if (scaling_data_paths is not None) and (not data_scaling_exp['null_loc_flux_scaling']): 
            #     loc_flux_scaling_temp = bind.resampling(edge_bins_1D,data_exp['edge_bins'][iord],data_scaling_exp['loc_flux_scaling'](data_exp['cen_bins'][iord]),kind=resamp_mode)
            #     loc_flux_scaling_temp[~cond_def] = 0.
            #     loc_flux_scaling_ord_contr+=loc_flux_scaling_temp*glob_weight_all[iord]
            if tell_data_paths is not None:
                tell_temp = bind.resampling(edge_bins_1D,data_exp['edge_bins'][iord],  data_exp['tell'][iord] ,kind=resamp_mode)
                tell_temp[~cond_def] = 0.                    
                tell_ord_contr+=tell_temp*glob_weight_all[iord]                     
            if DImast_weight_data_paths is not None:
                flux_ref_temp,cov_ref_temp = bind.resampling(edge_bins_1D, data_ref['edge_bins'][iord], data_ref['flux'][iord] , cov = data_ref['cov'][iord] , kind=resamp_mode)  
                flux_ref_temp,cov_ref_temp = bind.mul_array(flux_ref_temp,cov_ref_temp ,glob_weight_all[iord]      )
                flux_ref_temp[~cond_def] = 0.   
                flux_ref_ord_contr+=[flux_ref_temp]
                cov_ref_ord_contr+=[cov_ref_temp]
            if SpSstar_spec is not None:
                SpSstar_spec_temp = bind.resampling(edge_bins_1D,data_exp['edge_bins'][iord],SpSstar_spec[iord](data_exp['cen_bins'][iord]) ,kind=resamp_mode)
                SpSstar_spec_temp[~cond_def] = 0.
                SpSstar_spec_ord_contr+=SpSstar_spec_temp*glob_weight_all[iord]
            if flux_est_loc_exp is not None:  
                if cov_est_loc_exp is None:                    
                    flux_est_loc_temp = bind.resampling(edge_bins_1D, data_ref['edge_bins'][iord], flux_est_loc_exp[iord] , kind=resamp_mode)   
                    flux_est_loc_temp[~cond_def] = 0.   
                    flux_est_loc_ord_contr+=flux_est_loc_temp*glob_weight_all[iord]                  
                else:
                    flux_est_loc_temp,cov_est_loc_temp = bind.resampling(edge_bins_1D, data_ref['edge_bins'][iord], flux_est_loc_exp[iord] , cov = cov_est_loc_exp[iord], kind=resamp_mode)   
                    flux_est_loc_temp,cov_est_loc_temp = bind.mul_array( flux_est_loc_temp, cov_est_loc_temp , glob_weight_all[iord])
                    flux_est_loc_temp[~cond_def] = 0.   
                    flux_est_loc_ord_contr+=[flux_est_loc_temp]
                    cov_est_loc_ord_contr+=[cov_est_loc_temp]

        #Co-addition of spectra from all orders
        flux_1D,cov_1D = bind.sum(flux_ord_contr,cov_ord_contr)

        #Reset undefined pixels to nan
        flux_1D[~cond_def_binned]=np.nan        

        #Store data with artifical order for consistency with the routines
        #    - calibration profile are not used anymore afterward as there is no clear conversion from 2D to 1D for them
        #    - global flux scaling is not modified
        #      flux scaling tables are always called with global indexes
        data_exp1D = {'cen_bins':cen_bins_1D[None,:],'edge_bins':edge_bins_1D[None,:],'flux' : flux_1D[None,:],'cond_def' : cond_def_binned[None,:], 'cov' : [cov_1D]} 
        if SpSstar_spec is not None:data_exp1D['SpSstar_spec'] =  SpSstar_spec_ord_contr[None,:]
        datasave_npz(dir_save[data_type]+str(iexp_eff),data_exp1D)
        if tell_data_paths is not None:
            tell_1D = tell_ord_contr[None,:]
            datasave_npz(dir_save[data_type]+'_tell'+str(iexp_eff), {'tell':tell_1D})                 
        if DImast_weight_data_paths is not None:
            flux_ref_1D,cov_ref_1D = bind.sum(flux_ref_ord_contr,cov_ref_ord_contr)
            flux_ref_1D[~cond_def_binned]=np.nan   
            datasave_npz(dir_save[data_type]+'ref_'+str(iexp_eff),{'edge_bins':data_exp1D['edge_bins'],'flux':flux_ref_1D[None,:],'cov':[cov_1D]})           
        if flux_est_loc_exp is not None:
            if cov_est_loc_exp is None:
                flux_est_loc_1D = flux_est_loc_ord_contr
                flux_est_loc_1D[~cond_def_binned]=np.nan 
                dic_sav_estloc = {'edge_bins':data_exp1D['edge_bins'],'flux':flux_est_loc_1D[None,:]}
            else:
                flux_est_loc_1D,cov_est_loc_1D = bind.sum(flux_est_loc_ord_contr,cov_est_loc_ord_contr)
                flux_est_loc_1D[~cond_def_binned]=np.nan   
                dic_sav_estloc = {'edge_bins':data_exp1D['edge_bins'],'flux':flux_est_loc_1D[None,:],'cov':[cov_est_loc_1D]}
            datasave_npz(dir_save[data_type]+'estloc_'+str(iexp_eff),dic_sav_estloc)

        #Update common table for the visit
        #    - set to the table of first processed exposure
        if iexp==ifirst:datasave_npz(proc_com_data_paths, {'dim_exp':[1,nspec_1D],'nspec':nspec_1D,'cen_bins':np.tile(cen_bins_1D,[1,1]),'edge_bins':np.tile(edge_bins_1D,[1,1])})      
    
    return None

def para_conv_2D_to_1D_exp(func_input,nthreads,n_elem,y_inputs,common_args):
    pool_proc = Pool(processes=nthreads)   #cannot be passed through lmfit
    ind_chunk_list=init_parallel_func(nthreads,n_elem)
    chunked_args=[(y_inputs[0][ind_chunk[0]:ind_chunk[1]],)+common_args for ind_chunk in ind_chunk_list]	
    all_results=tuple(tab for tab in pool_proc.starmap(func_input,chunked_args))
    pool_proc.close()
    pool_proc.join() 	    
    return None








































"""

Function which calculates the properties of the stellar surface elements occulted by the planetary grid (x_st_sky_pl , y_st_sky_pl)

"""


def get_planet_disk_prop(spots_prop, pl_loc, grid_dic,system_prop, x_pos_pl, y_pos_pl, star_params, LD_law, ld_coeff, gd_band, cb_band, param, coeff_ord2name, func_prof_name, Rp_Rs, var_par_list, pol_mode,args) :


    # #Shift planet grid to current planet position
    # x_st_sky_pl = x_pos_pl+grid_dic['x_st_sky_grid_pl'][pl_loc][  system_prop['cond_in_RpRs'][pl_loc][0]  ]
    # y_st_sky_pl = y_pos_pl+grid_dic['y_st_sky_grid_pl'][pl_loc][  system_prop['cond_in_RpRs'][pl_loc][0]  ]

    # #Distance to star center, squared
    # r_proj2_sky_pl=x_st_sky_pl*x_st_sky_pl+y_st_sky_pl*y_st_sky_pl

    # #Identifying planet subcells occulting the star
    # #    - see model_star() for details on oblate star
    # if star_params['f_GD']==0 :cond_pl_occ = ( r_proj2_sky_pl <=1. )
    # else : z_st_sky_pl, cond_pl_occ = calc_zLOS_oblate(x_st_sky_pl, y_st_sky_pl, np.arccos(param['cos_istar']), star_params['RpoleReq'])[1:3]



    #Star is effectively occulted
    region_prop = {}

    if True in cond_pl_occ:

        ## Coordinates calculation

        # # x, y, z, r of occulted cells, in the 'inclined' star frame
        # region_prop['r_proj2_sky_pl']  = r_proj2_sky_pl[cond_pl_occ]
        # region_prop['x_st_sky_pl'] = x_st_sky_pl[cond_pl_occ]
        # region_prop['y_st_sky_pl'] = y_st_sky_pl[cond_pl_occ]
        # if star_params['f_GD']>0.:region_prop['z_st_sky_pl']=z_st_sky_pl[cond_pl_occ]
        # else:region_prop['z_st_sky_pl']=np.sqrt(1.-region_prop['r_proj2_sky_pl'])

        # #Frame conversion from the inclined star frame to the 'star' frame
        # region_prop['x_st_pl'],region_prop['y_st_pl'],region_prop['zstar_pl'] = conv_inclinedStarFrame_to_StarFrame(region_prop['x_st_sky_pl'],
        #                             region_prop['y_st_sky_pl'], region_prop['z_st_sky_pl'], np.arccos(param['cos_istar']))

        ## Flux calculation

        # # Size of disk cells
        # region_prop['flux_pl'] = np.ones(np.sum(cond_pl_occ))*grid_dic['Ssub_Sstar_pl'][pl_loc]


        # # Mu coordinate and gravity-darkening
        # #     - mu = cos(theta)), from 1 at the center of the disk to 0 at the limbs), with theta angle between LOS and local normal
        # if gd_band is not None :
        #     gd_grid,region_prop['mu_pl']=calc_GD(region_prop['x_st_pl'],
        #                                         region_prop['y_st_pl'],
        #                                         region_prop['zstar_pl'], 
        #                                         star_params,gd_band,
        #                                         region_prop['x_st_sky_pl'],
        #                                         region_prop['y_st_sky_pl'], 
        #                                         np.arccos([param['cos_istar']]))
                                                
        #     region_prop['flux_pl'] *= gd_grid   # We correct the flux from gravity darkening effect (larger flux at the pole)

        # else: region_prop['mu_pl'] = np.sqrt(1. - region_prop['r_proj2_sky_pl']  )


        # # Limb-Darkening coefficient at mu
        # region_prop['flux_pl'] *= LD_mu_func(LD_law,region_prop['mu_pl'],ld_coeff)

        # # Renormalisation to take into account that sum(Ftile) < 1 :
        # region_prop['flux_pl'] /= grid_dic['Ftot_star_achrom'][0]


        ## Spot effect on flux 
        
        # 'Base' flux level of the tiles : 1 if off-spot, and product of the spot flux if the tiles belongs to one or more spots (spot effect are assumed cumulative) :
        spot_atenuation = np.ones(np.sum(cond_pl_occ))  
        
        for spot in spots_prop :
            # Check if the spot is visible and 'close' to the planet center (in inclined star frame) : 
            x_sp_sky, y_sp_sky, ang_sp = spots_prop[spot]['x_sky_exp_center'],spots_prop[spot]['y_sky_exp_center'],spots_prop[spot]['ang_rad']
            if spots_prop[spot]['is_visible'] and (x_sp_sky - x_pos_pl)**2 + (y_sp_sky - y_pos_pl)**2 < (ang_sp + Rp_Rs)**2:
                
                spot_within_grid, spotted_tiles = calc_spotted_tiles(spots_prop[spot], region_prop['x_st_sky_pl'], region_prop['y_st_sky_pl'], region_prop['z_st_sky_pl'], {},
                                                                     star_params, param, use_grid_dic = False)
                                                                     
                # Multiply spotted tiles flux by the spot flux                                                
                if spot_within_grid :
                    for i in range(np.sum(cond_pl_occ)):
                        if spotted_tiles[i] : spot_atenuation[i] *= spots_prop[spot]['flux']  
                        
        region_prop['flux_pl'] *= spot_atenuation


        ## Radial velocity calculation

        # Vitesse de rotation
        region_prop['RV_pl'] = calc_RVrot(region_prop['x_st_sky_pl'],region_prop['y_st_pl'],np.arccos(param['cos_istar']),param)

        # Vitesse systémique
        region_prop['RV_pl'] += param['rv']

        # Convective blueshift
        CB_pl = np_poly(cb_band)(region_prop['mu_pl'])
        region_prop['RV_pl'] += CB_pl


        ## Other properties calculation : FW, ctrst, ...
        
        # We store the coordinates associated with the chosen dimension( mu, r_proj,.. add more possible coord choice ? )
        # if dim == 'mu'        : coord_prop = region_prop['mu_pl']
        # if dim == 'r_proj'    : coord_prop = np.sqrt(region_prop['r_proj2_sky_pl'])
        
        coord_prop = args['linevar_coord_grid']
        
        # Contrast and FWHM, always used
        region_prop['FWHM_pl']     = poly_prop_calc(param,coord_prop,coeff_ord2name['FWHM' ], pol_mode)
        region_prop['ctrst_pl']    = poly_prop_calc(param,coord_prop,coeff_ord2name['ctrst'], pol_mode)

        # Cas à deux gaussiennes
        if func_prof_name == 'dgauss' :
            region_prop['amp_l2c_pl'] = poly_prop_calc(param,coord_prop,coeff_ord2name['amp_l2c'], pol_mode)
            region_prop['rv_l2c_pl'] = poly_prop_calc(param,coord_prop,coeff_ord2name['rv_l2c'], pol_mode)
            region_prop['FWHM_l2c_pl'] = poly_prop_calc(param,coord_prop,coeff_ord2name['FWHM_l2c'], pol_mode)

        # Cas d'un profil voigt
        if func_prof_name == 'voigt' :
            region_prop['a_damp_pl'] = poly_prop_calc(param,coord_prop,coeff_ord2name['a_damp'], pol_mode)

        # On supprime les champs inutiles
        for key in ['x_st_sky_pl', 'y_st_sky_pl', 'z_st_sky_pl', 'x_st_pl', 'y_st_pl', 'zstar_pl'] :
            region_prop.pop(key)

    # On renvoie si le disque recouvre l'étoile, et les propriétés de la région occultée.
    return (   (True in cond_pl_occ) and np.any(region_prop['flux_pl'] > 0)   ), region_prop





"""

Fonction which decides if a spot is visible or not, based on star inclination (istar), spot coordinates in star rest frame (long, lat) and spot angular size (ang).

The fonction discretizes the spot circle and test for each point if it visible, thanks this calculation :

    Let's P = (x_st, y_st, z_st) be a point of stellar surface (star rest frame), with X axis along stellar equator and Y axis along stellar spin.

    Writing P in spherical coordinateed in star rest frame gives :

    x_st = sin(long)cos(lat)
    y_st = sin(lat)
    z_st = cos(long)cos(lat)

    We rotate this vector by angle (pi/2-istar) around X axis, moving it in the 'inclined' star frame (with Z now along the LOS, and Y the projected stellar spin):

    x_sky = x_st
    y_sky = sin(istar)y_st - cos sin(istar)z_st
    z_sky = cos(istar)y_st + sin(istar)z_st,

    with Y_sky axis along the line of sight and X_sky still along the star equator.

    The condition for P to be visible then reads z_sky > 0, yielding :

                                            cos(istar)sin(lat) + sin(istar)cos(long)cos(lat) > 0


    With gravity darkening : we replace y_st by sin(lat)*(1-f), which yields :

                                          cos(istar)sin(lat)(1-f) + sin(istar)cos(long)cos(lat) > 0

"""


def is_spot_visible(istar, long_rad, lat_rad, ang_rad, f_GD) :
    spot_visible = False
    
    for arg in np.linspace(0,2*np.pi, 20) :
        long_edge = long_rad + ang_rad * np.sin(arg)
        lat_edge  = lat_rad  + ang_rad * np.cos(arg)
        criterion = (    (np.cos(istar)*np.sin(lat_edge)*(1-f_GD)  +  np.sin(istar)*np.cos(long_edge)*np.cos(lat_edge))   >   0     )
        spot_visible |= criterion

    return spot_visible








"""

Fucntion which transforms a parameters list with spots properties (eg, lat__ISinst_VSvis_SPspot) into a more convenient dictionary of the form : 

# spot_prop = {
#   'spot1' : {lat : 0,   ang : 0,   Tcenter : 0,   flux : 0,   lat_rad_exp_center : 0,   sin_lat_exp_center : 0,    ... }
#   'spot2  : { ... }
#    }


# We assume spot parameters are never defined as common for all visit or all inst.

"""

def retrieve_spots_prop_from_param(star_params, param, inst, vis, t_bjd): 

    spots_prop = {}
    for par in param : 
        # Parameter is spot-related and linked to the right visit and instrument
        if ('_SP' in par) and ('_IS'+inst in par) and ('_VS'+vis in par) : 
            spot_name = par.split('_SP')[1]
            spot_par = par.split('__IS')[0]
            if spot_name not in spots_prop : spots_prop[spot_name] = {}
            spots_prop[spot_name][spot_par] = param[par]
            
    # Retrieve if spots are visible (at exposure center) 
    for spot in spots_prop : 
    
        # Spot lattitude
        lat_rad = spots_prop[spot]['lat']*np.pi/180.
        
        # Spot longitude
        sin_lat = np.sin(lat_rad)
        P_spot = 2*np.pi/((1.-param['alpha_rot']*sin_lat**2.-param['beta_rot']*sin_lat**4.)*star_params['om_eq']*3600.*24.)
        Tcen_sp = spots_prop[spot]['Tcenter'] - 2400000.
        long_rad = (t_bjd-Tcen_sp)/P_spot * 2*np.pi
        
        # Spot center coordinates in star rest frame
        x_st = np.sin(long_rad)*np.cos(lat_rad)
        y_st = np.sin(lat_rad)
        z_st = np.cos(long_rad)*np.cos(lat_rad)
    
        # inclined frame
        istar = np.arccos(param['cos_istar'])
        x_sky,y_sky,z_sky = conv_StarFrame_to_inclinedStarFrame(x_st,y_st,z_st,istar)
       
        # Store properties at exposure center
        spots_prop[spot]['lat_rad_exp_center'] = lat_rad
        spots_prop[spot]['sin_lat_exp_center'] = np.sin(lat_rad)
        spots_prop[spot]['cos_lat_exp_center'] = np.cos(lat_rad)
        spots_prop[spot]['long_rad_exp_center'] = long_rad
        spots_prop[spot]['sin_long_exp_center'] = np.sin(long_rad)
        spots_prop[spot]['cos_long_exp_center'] = np.cos(long_rad)
        spots_prop[spot]['x_sky_exp_center'] = x_sky
        spots_prop[spot]['y_sky_exp_center'] = y_sky
        spots_prop[spot]['z_sky_exp_center'] = z_sky
        spots_prop[spot]['ang_rad'] = spots_prop[spot]['ang'] * np.pi/180
        spots_prop[spot]['is_visible'] = is_spot_visible(istar,long_rad, lat_rad, spots_prop[spot]['ang_rad'], star_params['f_GD'])

    return spots_prop
             
            
            
            
            
            
            
            


"""

Function which calculates which tiles of the input sky grid are spotted

2 options : 

    + use_grid_dic = False : calculation will be performed on the   x_sky_grid, y_sky_grid, z_sky_grid   args (can be either a planetary or stellar grid),
                             by moving these grids from inclined star to star rest frame
    + use_grid_dic = True : calculation will be performed on the star grid contained in grid_dic['x/y/z_st'], which is already in star rest frame (no frame conversion needed). 
                            This option is used for calculating spotted stellar tiles, when istar is not fitted.
                            
                            
Calculation is straighforward : 

 - We rotate the star grid by the longitude of the spot around stellar spin : 
 
    x_sp_star =  x_st_grid*np.cos(long_rad) - z_st_grid*np.sin(long_rad)
    y_sp_star =  deepcopy(y_st_grid)
    z_sp_star =  x_st_grid*np.sin(long_rad) + z_st_grid*np.cos(long_rad)
    
    
 - We rotate the new grid by the lattitude of the spot, moving it to the spot rest frame
 
    x_sp =   deepcopy(x_sp_star)
    y_sp =   y_sp_star*np.cos(lat_rad) - z_sp_star*np.sin(long_rad)
    z_sp =   y_sp_star*np.sin(lat_rad) + z_sp_star*np.cos(long_rad)
    
    
 - We then check wich cells are within the spot by evaluing : 
 
                                            np.arctan2(np.sqrt(x_sp**2. + y_sp**2.),z_sp)     <     ang_sp
    
     
"""

def calc_spotted_tiles(spot_prop, x_sky_grid, y_sky_grid, z_sky_grid, grid_dic, star_params, param, use_grid_dic = False) :
                                                

    if use_grid_dic :
        cond_close_to_spot = (grid_dic['x_st_sky'] - spot_prop['x_sky_exp_center'])**2 + (grid_dic['y_st_sky'] - spot_prop['y_sky_exp_center'])**2 < spot_prop['ang_rad']**2
     
        x_st_grid, y_st_grid, z_st_grid = grid_dic['x_st'][cond_close_to_spot], grid_dic['y_st'][cond_close_to_spot], grid_dic['z_st'][cond_close_to_spot]
        
        
    else :  
        cond_close_to_spot = (x_sky_grid - spot_prop['x_sky_exp_center'])**2 + (y_sky_grid - spot_prop['y_sky_exp_center'])**2 < spot_prop['ang_rad']**2
    
        x_st_grid, y_st_grid, z_st_grid = conv_inclinedStarFrame_to_StarFrame(x_sky_grid[cond_close_to_spot],
                                                                                    y_sky_grid[cond_close_to_spot],
                                                                                    z_sky_grid[cond_close_to_spot],
                                                                                    np.arccos(param['cos_istar']))
        
        
    
    # Retrieve angular coordinates of spot
    cos_long, sin_long, cos_lat, sin_lat =  spot_prop['cos_long_exp_center'],  spot_prop['sin_long_exp_center'],                             spot_prop['cos_lat_exp_center' ],  spot_prop['sin_lat_exp_center' ]
    
    # Calculate coordinates in spot rest frame
    x_sp =                         x_st_grid*cos_long - z_st_grid*sin_long
    y_sp = y_st_grid*cos_lat  - (x_st_grid*sin_long + z_st_grid*cos_long)   *   sin_lat
    z_sp = y_st_grid*sin_lat  + (x_st_grid*sin_long + z_st_grid*cos_long)   *   cos_lat
   
    # Deduce which cells are within the spot
    phi_sp = np.arctan2(np.sqrt(x_sp**2. + y_sp**2.),z_sp)
    cond_in_sp = cond_close_to_spot
    cond_in_sp[cond_close_to_spot] = (phi_sp <= spot_prop['ang_rad'])
        
    # Check if at least one tile is within the spot
    spot_within_grid = (True in cond_in_sp)   
    

    return spot_within_grid, cond_in_sp







"""

Function which calculates the properties of spot-occulted stellar cells 

   + For each spot, we check if it is visible (with spots_prop[spot]['is_visible']), and if so, we use the previous function to calculate which cells of the stellar grid it occults
   + We store one global list of all spotted cells of the star (cond_in_sp), and their base flux level, calculated as the product of spot flux (flux_emitted_all_tiles_sp)
   + We then deduce the absorbed flux, as well as all the other properties of spotted tiles (RV, mu, ctrst, ...), exactly like in get_planet_disk_prop.

"""

def calc_spotted_region_prop(spots_prop, grid_dic, t_bjd, star_params, LD_law, ld_coeff, gd_band, cb_band, param, coeff_ord2name, dim, func_prof_name, var_par_list, pol_mode) :
    
    # Nombre de cases de l'étoile
    n_tiles = len(grid_dic['x_st_sky'])

    # On stocke la liste des cases stellaires occultées par au moins 1 spot, et la liste des flux occultés par les spots sur chaque case. On cummule l'occultation des spots si 2 ou plus overlappent. 


    flux_emitted_all_tiles_sp = np.ones(n_tiles, dtype = float)
    cond_in_sp = np.zeros(n_tiles, dtype = bool)
    spot_within_grid_all = False
    for spot in spots_prop :
        if spots_prop[spot]['is_visible'] : 
            if 'cos_istar' in var_par_list : use_grid_dic = False
            else : use_grid_dic = True
            spot_within_grid, cond_in_one_sp = calc_spotted_tiles(spots_prop[spot],
                                    grid_dic['x_st_sky'], grid_dic['y_st_sky'], grid_dic['z_st_sky'], grid_dic,
                                    star_params, param, use_grid_dic)
            if spot_within_grid:
                spot_within_grid_all = True
                flux_emitted_all_tiles_sp[cond_in_one_sp] *= spots_prop[spot]['flux']
                cond_in_sp |= cond_in_one_sp
                
    flux_occulted_all_tiles_sp = 1 - flux_emitted_all_tiles_sp
    
    region_prop = {}
    
    # Star is effectively affected by (at least) one spot
    if spot_within_grid_all :


        ## Coordinates calculation

        # Coordonnées x, y, z, r des régions spottées, référentiel incliné
        region_prop['r_proj2_st_sky_sp']  = grid_dic['r2_st_sky'][cond_in_sp]
        region_prop['x_st_sky_sp']        = grid_dic['x_st_sky'][cond_in_sp]
        region_prop['y_st_sky_sp']        = grid_dic['y_st_sky'][cond_in_sp]
        region_prop['z_st_sky_sp']        = grid_dic['z_st_sky'][cond_in_sp]

        #Frame conversion from the inclined star frame to the 'star' frame
        region_prop['x_st_sp'],region_prop['y_st_sp'],region_prop['z_st_sp'] = conv_inclinedStarFrame_to_StarFrame(region_prop['x_st_sky_sp'],
                                                                                                                      region_prop['y_st_sky_sp'], 
                                                                                                                      region_prop['z_st_sky_sp'], 
                                                                                                                      np.arccos(param['cos_istar']))
        


        ## Flux calculation


        # On garde uniquement les cases spottées, en gardant pour chaque case le flux du spot le plus sombre qui la recouvre (cf flux_occulted_all_tiles_sp)
        region_prop['flux_sp'] = flux_occulted_all_tiles_sp[cond_in_sp]*grid_dic['Ssub_Sstar']

        # If GD is on AND istar is fitted, then we need to recalculate mu, LD and GD
        if   (gd_band is not None)    and    ('cos_istar' in var_par_list)   :
            
            gd_sp, mu_sp = calc_GD(region_prop['x_st_sp'],
                                   region_prop['y_st_sp'],
                                   region_prop['z_st_sp'], 
                                   star_params, gd_band, 
                                   region_prop['x_st_sky_sp'],
                                   region_prop['y_st_sky_sp'], 
                                   np.arccos(param['cos_istar']))
                                   
            ld_sp = LD_mu_func(LD_law, mu_sp, ld_coeff) 
        
        # Otherwise it's ok, we can retrieve mu, LD and GD from those contained in grid_dic
        else: 
            mu_sp = grid_dic['mu_grid_star_achrom'][cond_in_sp][:,0]
            gd_sp = grid_dic['gd_grid_star_achrom'][cond_in_sp][:,0]
            ld_sp = grid_dic['ld_grid_star_achrom'][cond_in_sp][:,0]
            

        region_prop['flux_sp'] *= ld_sp
        region_prop['flux_sp'] *= gd_sp
        region_prop['mu_sp']    = mu_sp


        #Limb-Darkening coefficient at mu
        region_prop['flux_sp'] *= LD_mu_func(LD_law,region_prop['mu_sp'],ld_coeff)

        # Renormalisation to take into account that sum(Ftile) < 1 :
        region_prop['flux_sp'] /= grid_dic['Ftot_star_achrom'][0]
        

        ## Radial velocity calculation

        # Rotation speed
        region_prop['RV_sp'] = calc_RVrot(region_prop['x_st_sky_sp'],region_prop['y_st_sp'],star_params['istar_rad'],param)

        # Systemic velocity
        region_prop['RV_sp'] += param['rv']

        # Convectivd blueshift
        CB_sp = np_poly(cb_band)(region_prop['mu_sp'])
        region_prop['RV_sp'] += CB_sp


        ## Other properties calculation : FW, ctrst, ...


        # We store the coordinates associated with the chosen dimension( mu, r_proj,.. Add more possible coord choice ? )
        if dim == 'mu'        : coord_prop = region_prop['mu_sp']
        if dim == 'r_proj'    : coord_prop = np.sqrt(region_prop['r_proj2_st_sky_sp'])
        
        # FW et ctrst : always useful
        region_prop['FWHM_sp']     = poly_prop_calc(param,coord_prop,coeff_ord2name['FWHM' ], pol_mode) 
        region_prop['ctrst_sp']    = poly_prop_calc(param,coord_prop,coeff_ord2name['ctrst'], pol_mode)

        # Cas à deux gaussiennes
        if func_prof_name == 'dgauss' :
            region_prop['amp_l2c_sp'] = poly_prop_calc(param,coord_prop,coeff_ord2name['amp_l2c'], pol_mode)
            region_prop['rv_l2c_sp'] = poly_prop_calc(param,coord_prop,coeff_ord2name['rv_l2c'], pol_mode)
            region_prop['FWHM_l2c_sp'] = poly_prop_calc(param,coord_prop,coeff_ord2name['FWHM_l2c'], pol_mode)

        # Cas profil 'voigt'
        if func_prof_name == 'voigt' :
            region_prop['a_damp_pl'] = poly_prop_calc(param,coord_prop,coeff_ord2name['a_damp'], pol_mode)

        #On retire les champs inutiles dans region_prop :
        for key in ['r_proj2_st_sky_sp', 'x_st_sky_sp', 'y_st_sky_sp', 'z_st_sky_sp', 'x_st_sp', 'y_st_sp', 'z_st_sp', 'mu_sp'] :
           region_prop.pop(key)
    
    return cond_in_sp, (   spot_within_grid_all and np.any(region_prop['flux_sp'] > 0)   ), region_prop









"""

Final function which compute the deviation of an exposure from the 'normal' stellar CCF

+ We first calculate the properties of spot-occulted and planet-occulted cells 
+ We then compute the corresponding profiles, with the chosen calculation mode (see args['precision']).
+ Note that spot properties are assumed constant throughout the exposure

"""

def compute_deviation_profile(args, param, inst, vis, iexp,gen_dic,theo_dic,data_dic,coord_dic,system_param) :
    

    #-----------------------------------------------
    
    # # Retrieve spot parameters
    # spot_within_grid_all = False
    # if args['use_spots']:
    #     spots_prop = retrieve_spots_prop_from_param(args['star_params'], param, inst, vis, args['t_exp_bjd'][inst][vis][iexp])
      
    
    #     # Spots occulted tiles
        
    #     # Properties are stores in a single dic : 
        
    #     # - tab_prop_spot['rv'][i_tile]   -> RV of the spotted tile number 'itile'
    #     # - tab_prop_spot['flux'][i_tile] -> flux level of the spotted tile number 'itile'
    #     # etc...
    
    
    #     cond_sp, spot_within_grid_all, tab_prop_sp = calc_spotted_region_prop(
    #                                                     spots_prop,
    #                                                     args['grid_dic'],
    #                                                     args['t_exp_bjd'][inst][vis][iexp],
                
    #                                                     args['star_params'],
    #                                                     LD_law,
    #                                                     ld_coeff,
    #                                                     gd_band,
    #                                                     cb_band,
    #                                                     param,
    #                                                     args['coeff_ord2name'][inst][vis],
    #                                                     args['coord_line'],
    #                                                     args['func_prof_name'][inst],
    #                                                     args['var_par_list'],
    #                                                     args['pol_mode'])    
      
    # else:spots_prop={}

    #-----------------------------------------------    

    prof_deviation = np.zeros(len(args['cen_bins']))


  
  
    #Retrieve planetary coordinates
    # for pl_loc in gen_dic['studied_pl']:
        # x_oversamp_pl, y_oversamp_pl, n_disk = calc_pl_coord_sky(args, param, pl_loc, inst, vis, iexp,theo_dic)
        

        # Planet occulted tile
        
        # Properties are stores in a single dic : 
        # - tab_prop_pl['rv'][i_disk][i_tile] -> RV of the tile number 'itile' of the disk number 'idisk'
    
        # n_disk_in_transit = 0
        # tab_prop_pl = {}
        # if args['calc_pl_flux'] : 
        #     for idisk in range(n_disk) :
        
        #         disk_in_transit, region_prop_pl = get_planet_disk_prop(
        #                                         spots_prop,
        #                                         pl_loc,
        #                                         theo_dic,
        #                                         args['system_prop'],
        #                                         x_oversamp_pl[idisk],
        #                                         y_oversamp_pl[idisk],
        #                                         star_params,
        #                                         LD_law,
        #                                         ld_coeff,
        #                                         gd_band,
        #                                         cb_band,
        #                                         param,
        #                                         args['coeff_ord2name'][inst][vis],
        #                                         args['func_prof_name'],
        #                                         args['system_prop'][pl_loc][0],
        #                                         args['var_par_list'],
        #                                         args['pol_mode'],args)
        
        #         if disk_in_transit :
        #             n_disk_in_transit += 1
        #             for prop in region_prop_pl :
        #                 if prop not in tab_prop_pl : tab_prop_pl[prop] = []
        #                 tab_prop_pl[prop].append(region_prop_pl[prop])



    # We then compute the deviation profile

    # - For spots, we compute one profile per tile, we the computed properties
    # - For the planet, the calculation mode depends on args['precision']

    # # Spot contribution
    # if spots_prop != {} and spot_within_grid_all :

    #     for itile in range(len(tab_prop_sp['flux_sp'])) :

    #         input_exp = {'cont'  :  tab_prop_sp['flux_sp'] [itile],
    #                      'rv'    :  tab_prop_sp['RV_sp']   [itile],
    #                      'ctrst' :  tab_prop_sp['ctrst_sp'][itile],
    #                      'FWHM'  :  tab_prop_sp['FWHM_sp'] [itile]
    #                     }

    #         # Cas demi-gaussienne :
    #         if args['func_prof_name'] == 'dgauss' :
    #             input_exp['rv_l2c'  ] = tab_prop_sp['rv_l2c_sp']   [itile]
    #             input_exp['amp_l2c' ] = tab_prop_sp['amp_l2c']     [itile]
    #             input_exp['FWHM_l2c'] = tab_prop_sp['FWHM_l2c_sp'] [itile]


    #         # Cas voigt :
    #         if args['func_prof_name'] == 'voigt' :
    #             input_exp['a_damp'] = tab_prop_sp ['a_damp_sp'][itile]

    #         # On ajoute le profil de la case au profil de la déviation
    #         prof_deviation +=  args['func_prof'][inst]( input_exp, args['cen_bins'][inst][vis]  )[0]       # On l'ajoute au profil local du disque








    # # We store spot occulted flux, useful for the corr_spot module
    # spot_occulted_prof = deepcopy(prof_deviation)
    
    
    # # Retrieving total occulted flux (ie, expected continuum) and mean RV of planet-occulted region : 
    # occulted_flux = 0 
    # mean_prop = {'rv' : 0, 'FWHM' : 0, 'ctrst' : 0, 'mu' : 0}    #j'ai mis mu temporarieemnt
    
    # # From spots
    # if spots_prop != {} and spot_within_grid_all : occulted_flux += np.sum(tab_prop_sp['flux_sp'])
    # spot_occulted_flux = occulted_flux
    
    # # From the planet
    # if n_disk_in_transit >= 1 : 
    #     for idisk in range(n_disk_in_transit) : 
        
    #         # Flux
    #         occulted_flux += np.sum(tab_prop_pl['flux_pl'][idisk])/n_disk
            
    #         # Mean prop
    #         if args['calc_pl_mean_prop'] : 
    #             mean_prop['rv']    += np.sum(    tab_prop_pl['flux_pl'][idisk] * tab_prop_pl['RV_pl'][idisk]     /   np.sum(tab_prop_pl['flux_pl'][idisk])   )   /   n_disk
    #             mean_prop['FWHM']  += np.sum(    tab_prop_pl['flux_pl'][idisk] * tab_prop_pl['FWHM_pl'][idisk]   /   np.sum(tab_prop_pl['flux_pl'][idisk])   )   /   n_disk
    #             mean_prop['ctrst'] += np.sum(    tab_prop_pl['flux_pl'][idisk] * tab_prop_pl['ctrst_pl'][idisk]  /   np.sum(tab_prop_pl['flux_pl'][idisk])   )   /   n_disk
    #             # if args['coord_line'] == 'mu':
    #             mean_prop['mu'] += np.sum(   tab_prop_pl['flux_pl'][idisk] * tab_prop_pl['mu_pl'][idisk]     /   np.sum(tab_prop_pl['flux_pl'][idisk])   )   /   n_disk
    #             # if args['coord_line'] == 'r_proj' : 
    #             #     mean_prop['r_proj'] += np.sum(    tab_prop_pl['flux_pl'][idisk] * np.sqrt(tab_prop_pl['r_proj2_sky_pl'][idisk])  /   np.sum(tab_prop_pl['flux_pl'][idisk])   )   /   n_disk

    return prof_deviation





   
   
   












    
    
    
    
    



                
   
   
   






"""

Routine to correction DI profiles from spot contamination
Profile are assumed to be aligned in star rest frame 

"""

def corr_spot(corr_spot_dic, coord_dic,inst,vis,data_dic,data_prop,gen_dic, theo_dic,system_param) :
    
    
    print('   > Correcting DI CCF from spot contamination' )
    proc_gen_data_paths_new = gen_dic['save_data_dir']+'Spot_corr_DI_data/'+gen_dic['add_txt_path']['DI']+'/'+inst+'_'+vis+'_'

    if gen_dic['calc_correct_spots']:
        print('         Calculating data')
    
        data_vis = data_dic[inst][vis]
        data_DI_prop_vis = np.load(gen_dic['save_data_dir']+'DIorig_prop/'+inst+'_'+vis+'.npz',  allow_pickle=True)['data'].item()
        pl_loc = list(gen_dic['transit_pl'])[0]
    
    
    
        for iexp in range(data_vis['n_in_visit']):    
    
            # Load exp profile
            data_exp = np.load(data_dic[inst][vis]['proc_DI_data_paths']+str(iexp)+'.npz',allow_pickle=True)['data'].item()
    
            if corr_spot_dic['spots_prop'][inst][vis] != {} : 
            
                if iexp == 0 :
                    
                    # Initialize args for 'compute_deviation_profile' function, common to all exposures
                    fixed_args = {}
                    fixed_args['inst_list']=[inst]
                    fixed_args['inst_vis_list']={inst:[vis]}
                    fixed_args['pl_loc'] = pl_loc
                    fixed_args['cen_bins'] = {inst : {vis : data_exp['cen_bins'][0]}}
                    fixed_args['dcen_bin'] = {inst : {vis : data_exp['cen_bins'][0][1]  -  data_exp['cen_bins'][0][0]}}
                    fixed_args['system_param'] = system_param
                    fixed_args['grid_dic'] = theo_dic     
                    fixed_args['grid_dic']['cond_in_RpRs'] = {pl_loc : data_dic['DI']['system_prop']['achrom']['cond_in_RpRs'][pl_loc]}
                    fixed_args['system_prop'] = data_dic['DI']['system_prop']['achrom']
                    fixed_args['coord_line'] = corr_spot_dic['coord_line']
                    fixed_args['func_prof'] = {inst : dispatch_func_prof(corr_spot_dic['intr_prof']['func_prof_name'][inst])}
                    fixed_args['func_prof_name'] =  corr_spot_dic['intr_prof']['func_prof_name']
                    fixed_args['pol_mode'] = corr_spot_dic['intr_prof']['pol_mode']
                    fixed_args['phase']     = {inst : {vis :  [coord_dic[inst][vis][pl_loc][m] for m in ['st_ph','cen_ph','end_ph']]}}
                    fixed_args['t_exp_bjd'] = {inst : { vis : coord_dic[inst][vis]['bjd']   }}
                    fixed_args['precision'] = corr_spot_dic['precision']
                    fixed_args['var_par_list'] = []
                    fixed_args['print_exp'] = False
                    fixed_args['calc_pl_mean_prop'] = False
                    
                    # Paramètres susceptibles d'être fittés (voir gen_dic['fit_ResProf'])
                    params = {'rv' : 0}
                    for par in ['veq','alpha_rot','beta_rot','cos_istar','c1_CB','c2_CB','c3_CB'] : params[par]=system_param['star'][par]
                    params['lambda_rad__pl'+pl_loc] = system_param[pl_loc]['lambda_rad']
                    params['aRs__pl'+pl_loc] = system_param[pl_loc]['aRs']
                    params['inclin_rad__pl'+pl_loc] = system_param[pl_loc]['inclin_rad']
                        
                    # Propriétés du profils stellaire
                    params = par_formatting(params,corr_spot_dic['intr_prof']['mod_prop'],None,None,fixed_args,inst,vis) 
                    params_without_spot = deepcopy(params)
                    
                    # Propriétés des spots   
                    par_formatting(params,corr_spot_dic['spots_prop'][inst][vis],None,None,fixed_args,inst,vis)
                    params_with_spot = deepcopy(params)
                
                
                # Load exposure continuum
                cont_exp = data_DI_prop_vis[iexp]['cont']
        
                # Version with overlapping took into account
                if corr_spot_dic['overlap'] :
                    fixed_args['calc_pl_flux'] = True
                    tot_occulted_prof, tot_occulted_flux = compute_deviation_profile(fixed_args, params_with_spot,    inst, vis, iexp,star_params,gen_dic,theo_dic,data_dic,coord_dic)[0:2]
                    pl_occulted_prof , pl_occulted_flux  = compute_deviation_profile(fixed_args, params_without_spot, inst, vis, iexp,star_params,gen_dic,theo_dic,data_dic,coord_dic)[0:2]
                    spot_occulted_prof, spot_occulted_flux = tot_occulted_prof - pl_occulted_prof  ,  tot_occulted_flux - pl_occulted_flux
                
                # Version which assumes that spots and planet are always separated. 
                else : 
                    fixed_args['calc_pl_flux'] = False
                    spot_occulted_prof, spot_occulted_flux = compute_deviation_profile(fixed_args, params_with_spot,    inst, vis, iexp,star_params,gen_dic,theo_dic,data_dic,coord_dic)[2:4]
                    fixed_args['calc_pl_flux'] = True
                    pl_occulted_flux                       = compute_deviation_profile(fixed_args, params_without_spot, inst, vis, iexp,star_params,gen_dic,theo_dic,data_dic,coord_dic)[1]
                
                # Set the DI profile continuum to the expected value given planetary and spot occultation
                data_exp['flux'][0] *= (1 - spot_occulted_flux - pl_occulted_flux)  /  cont_exp
                
                # Add the profile occulted by spots, taking into account instrumental dispersion :
                data_exp['flux'][0] += convol_prof(spot_occulted_prof, fixed_args['cen_bins'][inst][vis], return_FWHM_inst(inst))
                
                # Reset the exposure continuum to the initial value (might be important if the profiles are not fitted again, especially in the Joined Residual profiles fitting). 
                data_exp['flux'][0] *= cont_exp  /  (1 - pl_occulted_flux)
            
            # Save exp data
            np.savez_compressed(proc_gen_data_paths_new+str(iexp),data=data_exp,allow_pickle=True)

        # updating path to DI data
        data_vis['proc_DI_data_paths'] = proc_gen_data_paths_new

            
        
    else :
        check_data({'path':proc_gen_data_paths_new+'_add'}) 
        data_vis['proc_DI_data_paths'] = proc_gen_data_paths_new


    return None





