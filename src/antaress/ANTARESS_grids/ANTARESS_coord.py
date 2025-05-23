#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import newton
from copy import deepcopy
from ..ANTARESS_general.constant_data import G_usi,Mjup,Msun
from ..ANTARESS_general.utils import np_where1D,npint,stop,closest

##################################################################################################    
#%% Coordinates
################################################################################################## 

def coord_expos(pl_loc,coord_dic,inst,vis,star_params,pl_params,t_bjd,exp_time,data_dic,RpRs):
    r"""**Exposure coordinates**

    Calculates temporal and spatial coordinates of planets at the start, mid, and end of an exposure.
    
    Args:
        TBD
    
    Returns:
        None
    
    """ 
    
    #Orbital phase and corresponding times (days) of current exposure relative to transit center  
    st_phases,phases,end_phases=get_timeorbit(coord_dic[inst][vis][pl_loc]['Tcenter'],t_bjd, pl_params, exp_time)[0:3]   

    #Exposure duration in phase
    ph_dur=end_phases-st_phases

    #Return values for start, mid, end exposure    
    xp_all,yp_all,zp_all,_,_,_,_,_,ecl= calc_pl_coord(pl_params['ecc'],pl_params['omega_rad'],pl_params['aRs'],pl_params['inclin_rad'],np.vstack((st_phases,phases,end_phases)),RpRs,pl_params['lambda_rad'],star_params)
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

        #Average orbital velocity over oversampled values
        #    - converted from velocity along the LOS in /Rstar/h to km/s
        rv_pl_osamp,v_pl_osamp = calc_pl_coord(pl_params['ecc'],pl_params['omega_rad'],pl_params['aRs'],pl_params['inclin_rad'],ph_osamp_loc,None,None,None,rv_LOS=True,omega_p=pl_params['omega_p'])[6:8]
        rv_pl_all=star_params['RV_conv_fact']*np.mean(rv_pl_osamp)    
        v_pl_all=star_params['RV_conv_fact']*np.mean(v_pl_osamp)   
                
    else:
        rv_pl_all=None
        v_pl_all=None
        
    return positions,st_positions,end_positions,eclipse,rv_pl_all,v_pl_all,st_phases,phases,end_phases,ph_dur


def coord_expos_ar(ar_name,t_bjd,ar_prop,star_params,exp_dur,coord_par):
    r"""**Exposure active region coordinates**

    Calculates temporal and spatial coordinates of active regions at the start, mid, and end of an exposure.
    
    Args:
        ar_name (str) : name of the active region considered.
        t_bjd (float) : timestamp of the exposure considered. Needed to calculate the active region longitude.
        ar_prop (dict) : nominal properties of the active region considered.
        star_params (dict) : star properties.
        exp_dur (float) : optional, duration of the exposure considered. If left as None, only the active region properties at the center of the exposure will be computed.
                            Otherwise, the active region properties at the start, center and end of exposure will be computed.
        coord_par (dict): coordinates to store for the active region.
    
    Returns:
        None
    
    """ 
    ar_prop_exp = {} 

    #Finding type of active region used
    if ar_prop[ar_name]['fctrst'] < 1.:contamin_type = 'spots'
    else:contamin_type = 'faculae'

    # Active region longitude - varies over time
    P = 2*np.pi/((1.-star_params['alpha_rot_'+contamin_type]*np.sin(ar_prop[ar_name]['lat_rad'])**2.-star_params['beta_rot_'+contamin_type]*np.sin(ar_prop[ar_name]['lat_rad'])**4.)*star_params['om_eq_'+contamin_type]*3600.*24.)  
    long_t_start,long_t_center,long_t_end = get_timeorbit(ar_prop[ar_name]['Tc_ar'],t_bjd, {'period':P}, exp_dur,conv_ang = True)[0:3]

    #Coordinates start, mid, end exposure 
    istar = np.arccos(ar_prop['cos_istar'])
    for key in coord_par:ar_prop_exp[key] = np.zeros(3,dtype=float)*np.nan
    ar_prop_exp['is_visible'] = np.zeros(3,dtype=bool)
    calc_ar_coord(ar_prop_exp,ar_prop[ar_name]['ang_rad'],ar_prop[ar_name]['lat_rad'],long_t_center*2*np.pi,1,istar,star_params)
    if exp_dur is not None:
        calc_ar_coord(ar_prop_exp,ar_prop[ar_name]['ang_rad'],ar_prop[ar_name]['lat_rad'],long_t_start*2*np.pi,0,istar,star_params)
        calc_ar_coord(ar_prop_exp,ar_prop[ar_name]['ang_rad'],ar_prop[ar_name]['lat_rad'],long_t_end*2*np.pi,2,istar,star_params)
  
    return ar_prop_exp




##################################################################################################    
#%%% Spatial and velocity coordinates
################################################################################################## 

def calc_Kstar(params,star_params):   
    r"""**Keplerian semi-amplitude.** 
    
    Calculates Keplerian semi-amplitude of the star induced by a given planet (in m/s).

    The general formula is
    
    .. math:: 
       K_\mathrm{prim} = (2 \pi G/P)^{1/3} M_\mathrm{sec} \sin(i_p) ((M_\mathrm{sec}+M_\mathrm{prim})^{1/3} / M_\mathrm{prim}) /\sqrt{1-e^2}
    
    With :math:`K_\mathrm{prim}` in km/s, `G` in :math:`m^3 kg^{-1} s^{-2}`, `P` in s, :math:`M_\mathrm{prim}` and :math:`M_\mathrm{sec}` in kg. Thus    

    .. math:: 
       K_\mathrm{prim} &= (212.91918458020422/P_\mathrm{days}^{1/3}) M_\mathrm{sec} \sin(i_p) ((M_\mathrm{sec}+M_\mathrm{prim})^{1/3}/M_\mathrm{prim}) / \sqrt{1-e^2}   \\
       &\mathrm{where\,} [ (2 \pi 6.67300\times10^{-11})^{1/3} 1.9891001\times10^{30} (1.9891001\times10^{30})^{1/3}]/[(24 \times 3600)^{1/3} (1.9891001\times10^{30})]/1000 = 212.91918458020422

    With :math:`P_\mathrm{days}` in days, :math:`M_\mathrm{prim}` and :math:`M_\mathrm{sec}` in :math:`M_\mathrm{Sun}`.  
    For a planet :math:`M_\mathrm{sec} = M_p << M_\mathrm{prim} = M_{\star}`, thus
    
    .. math::     
       K_{\star} \sim (2 \pi G/P)^{1/3} (M_p \sin(i_p)/M_{\star}^{2/3}) /\sqrt{1-e^2}

    With :math:`K_{\star}` in km/s, `G` in :math:`m^3 kg^{-1} s^{-2}`, `P` in s, :math:`M_p` and :math:`M_{\star}` in kg. Or
    
    .. math:: 
       K_{\star} &= (0.20323137323712528/P_\mathrm{days}^{1/3}) (M_p \sin(i_p)/M_{\star}^{2/3}) /\sqrt{1-e^2} \\
       &\mathrm{where\,} [ (2 \pi 6.67300\times10^{-11})^{1/3} 1.8986000\times10^{27}]/[(24 \times 3600)^{1/3} (1.9891001\times10^{30})^{2/3}]/1000 = 0.20323137    

    With :math:`P_\mathrm{days}` in days, :math:`M_p` and :math:`M_{\star}` in :math:`M_\mathrm{Sun}`. Or

    .. math:: 
       K_{\star} = (0.02843112300449059/P_\mathrm{years}^{1/3}) (M_p \sin(i_p)/M_{\star}^{2/3}) /\sqrt{1-e^2}
 
    With :math:`P_\mathrm{years}` in years, :math:`M_p` and :math:`M_{\star}` in :math:`M_\mathrm{Sun}`.   

    Args:
        TBD
    
    Returns:
        Kstar (float): Keplerian semi-amplitude of the star from the planet (km/s) 
    
    """    
    Kstar = ((2.*np.pi*G_usi/(params['period_s']))**(1./3.))/np.sqrt(1.-params['ecc']**2.)
    if ('Mp' in params) and (params['Mp']*Mjup/(star_params['Mstar']*Msun)>1e-2):
        print('Mp/Mstar>1e-2 : using exact Keplerian semi-amplitude formula')
        Kstar *= params['Mp']*np.sin(params['inclin_rad'])*Mjup * ((params['Mp']*Mjup + star_params['Mstar']*Msun)**(1./3.))/(star_params['Mstar']*Msun)  
    else:
        Kstar *= params['Msini']*Mjup/(star_params['Mstar']*Msun)**(2./3.)
    return Kstar


def calc_rv_star(coord_dic,inst,vis,system_param,gen_dic,bjd_exp,dur_exp,sysvel):
    r"""**Stellar rv: for observations**

    Calculates the rv of the star in the stellar and sun barycentric rest frames.

    The Keplerian rv of the star induced by a given planet is
    
    .. math::     
       rv(\mathrm{star/stellar \, CDM \,;\, from \, pl})= K_{\star} ( \cos(\nu+\omega_\mathrm{bar})+ e \cos(\omega_\mathrm{bar}) )
       
    With :math:`K_{\star}`, :math:`\nu`, `e` and :math:`\omega_\mathrm{bar}` specific to the planet.
    Radial velocity is negative when the planet is coming toward the observer, thus in the same frame as the observed rv tables.

    Args:
        TBD
    
    Returns:
        None
    
    """     

    #Oversampling factor of exposures
    n_ov=10.

    #-------------------------------------    
    #Keplerian motion induced by requested planets
    RV_star_stelCDM_ov=0.
    for pl_loc in gen_dic['kepl_pl']:     
        PlParam_loc=system_param[pl_loc]   
        
        #Orbital phase 
        st_phase,phase,end_phase=get_timeorbit(coord_dic[inst][vis][pl_loc]['Tcenter'],bjd_exp, PlParam_loc, dur_exp)[0:3]

        #True anomaly for start and end of exposure
        True_anom=calc_true_anom(PlParam_loc['ecc'],np.vstack((st_phase,end_phase)),PlParam_loc['omega_rad'])[0]

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






def calc_rv_star_HR(pl_ref,system_param,kepl_pl,coord_dic,inst,vis,data_dic):
    r"""**Stellar rv: for plots**

    Calculates the rv of the star in the sun barycentric rest frame, over a high-resolution phase table.
    
    Specific to each visit, since the Keplerian rv is not the same if there are multiple planets.

    Args:
        TBD
    
    Returns:
        None
    
    """ 
    
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
            True_anom,Ecc_anom,_=calc_true_anom(ecc,phase,omega_bar)
  
            #Keplerian motion (km/s)
            RV_star_stelCDM[i_loc]+=PlParam['Kstar_kms']*(np.cos(True_anom+omega_bar)+ecc*np.cos(omega_bar) )

    ### End of loop on exposures 
 
    #Keplerian curve in heliocentric rest frame (km/s)
    RV_star_solCDM=RV_star_stelCDM+data_dic['DI']['sysvel'][inst][vis]

    return phase_tab_HR,RV_star_solCDM




def Kepler_func(Ecc_anom,Mean_anom,ecc):
    r"""**Eccentric anomaly solver.**
    
    Finds the eccentric anomaly of the orbit using the mean anomaly and eccentricity.
    
    :math:`M = E - e \sin(E)`        
    
    Args:
        Ecc_anom (float): Eccentric anomaly.
        Mean_anom (float): Mean anomaly.
        ecc (float): Eccentricity.
    
    Returns:
        delta (float): equation to nullify 
    
    """     
    delta=Ecc_anom-ecc*np.sin(Ecc_anom)-Mean_anom
    return delta 



   
def calc_mean_anom_TR(ecc,omega_bar):  
    r"""**Mean anomany at mid-transit.**
    
    Args:
        ecc (float): Eccentricity.
        omega_bar (float): Argument of periastron.
    
    Returns:
        Mean_anom_TR (float): Mean anomany at mid-transit. 
    
    """     
    
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
    


def calc_true_anom(ecc,phase,omega_bar):
    r"""**True anomaly.**
    
    Args:
        ecc (float): Eccentricity.
        phase (1D array): orbital phase.
        omega_bar (float): Argument of periastron.
    
    Returns:
        True_anom (1D array): True anomany at given phases. 
        Ecc_anom (1D array): Eccentric anomany at given phases. 
        Mean_anom (1D array): Mean anomany at given phases. 
    
    """  
    
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
        Mean_anom=2.*np.pi*phase+calc_mean_anom_TR(ecc,omega_bar)
			        
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

 

  



def calc_pl_coord(ecc,omega_bar,aRs,inclin,ph_loc,RpRs,lambda_rad,star_params,rv_LOS=False,omega_p=None):
    r"""**Planet orbit**

    Calculates orbital planetary position and velocity, as well as eclipse condition.
    
    References:

        - `<http://www.gehirn-und-geist.de/sixcms/media.php/370/Leseprobe.406372.pdf>`_
        - `<http://www.relativitycalculator.com/pdfs/RV_Derivation.pdf>`_
        - `<https://www.sciencedirect.com/topics/engineering/eccentric-anomaly>`_

    We calculate the perifocal coordinates of the planet, in cartesian coordinates relative to the star.
    The radial velocity :math:`rv(pl/star)` (negative when the planet is coming toward us) is derived directly from these coordinates as

    .. math::      
       rv(pl/star) = - K_\mathrm{pl/star} ( \cos(\nu+\omega_\mathrm{bar})+e \cos(\omega_\mathrm{bar}) )    

    Kepler third's law gives 
     
    .. math::       
       P^2/a^3 &= 4 \pi^2 / (G (M_\mathrm{pl}+M_{\star}))    \\
       a &= ( G P^2 (M_{\star}+M_\mathrm{pl}) / (4 \pi^2) )^{1/3}
          
    Thus we can write the semi-amplitude of the planet Keplerian motion as
          
    .. math::      
       K_\mathrm{pl/star} &= 2 \pi a \sin(i_p)/(P \sqrt{1-e^2})    \\   
       K_\mathrm{pl/star} &= (2 \pi G/P)^{1/3} (M_{\star}+M_\mathrm{pl})^{1/3} \sin(i_p)/\sqrt{1-e^2}        
    
    The radial velocity of the planet in the star rest frame can also be defined as 

    .. math::  
       rv(pl/star) = rv(pl/CMD_{\star}) - rv(star/CMD_{\star})
    
    And because the barycenter is fixed in its own frame:

    .. math::          
       & M_{\star} rv(star/CMD_{\star}) +  M_\mathrm{pl} rv(pl/CMD_{\star}) = 0          \\  
       & M_{\star} rv(star/CMD_{\star}) +  M_\mathrm{pl} (rv(pl/star)+rv(star/CMD_{\star})) = 0       \\     
       & rv(star/CMD_{\star})= - rv(pl/star) M_\mathrm{pl}/(M_{\star}+M_\mathrm{pl}) 
     
    Thus 

    .. math:: 
       rv(star/CMD_{\star}) &= K[star/CMD_{\star}] ( \cos(\nu+\omega_\mathrm{bar})+e \cos(\omega_\mathrm{bar}) )     \\
       \mathrm{where\,} & K[star/CMD_{\star}] = K_\mathrm{pl/star} M_\mathrm{pl}/(M_{\star}+M_\mathrm{pl}) \\
                        & K[star/CMD_{\star}] = (2 \pi G/P)^{1/3} M_\mathrm{pl} \sin(i_p)/((M_{\star}+M_\mathrm{pl})^{2/3} \sqrt{1-e^2}) 
    
    With `K` in km/s, `G` in :math:`m^3 kg^{-1} s^{-2}`, `P` in s, :math:`M_\mathrm{pl}` and :math:`M_{\star}` in kg. Or (see `calc_Kstar()`)  

    .. math::  
       K[star/CMD_{\star}] &= (212.91918458020422/P_\mathrm{days}^{1/3}) M_\mathrm{pl,Msun} \sin(i_p)/((M_{\star,Msun} +M_\mathrm{pl,Msun})^{2/3} \sqrt{1-e^2}) \\
       K[star/CMD_{\star}] &= (0.02843112300449059/P_\mathrm{yr}^{1/3}) M_\mathrm{pl,Msun} \sin(i_p)/((M_{\star,Msun} +M_\mathrm{pl,Msun})^{2/3} \sqrt{1-e^2})           
              
    We can also write  

    .. math:: 
       rv(pl/CMD_{\star}) &=  rv(pl/star) + rv(star/CMD_{\star})  \\
                          &=  ( - K_\mathrm{pl/star} + K_\mathrm{pl/star} M_\mathrm{pl}/(M_{\star}+M_\mathrm{pl}) ) ( \cos(\nu+\omega_\mathrm{bar})+e \cos(\omega_\mathrm{bar}) )     \\  
                          &=  K[pl/CMD_{\star}] ( \cos(\nu+\omega_\mathrm{bar})+e \cos(\omega_\mathrm{bar}) )  \\
       \mathrm{where\,} & K[pl/CMD_{\star}] = K_\mathrm{pl/star} ( M_\mathrm{pl}/(M_{\star}+M_\mathrm{pl})  - 1 )  \\
                        & K[pl/CMD_{\star}] = K_\mathrm{pl/star} ( (M_\mathrm{pl}-M_{\star})/(M_{\star}+M_\mathrm{pl}) )  \\
                        & K[pl/CMD_{\star}] = - (2 \pi G/P)^{1/3} (M_{\star}-M_\mathrm{pl}) \sin(i_p)/(\sqrt{1-e^2} (M_{\star}+M_\mathrm{pl})^{1/3} )

    If we assume that the atmospheric signal tracks the planet orbital velocity (ie, there is no atmospheric dynamics), then we can 
    link the planet and star masses with this measurement. Doing as few assumptions as possible, the quantity we measure is 
     
    .. math::     
       rv(pl/CDM_\mathrm{sun}) &= rv(pl/CMD_{\star}) + rv(CMD_{\star}/CDM_\mathrm{sun})  \\
                               &= - (2 \pi G/P)^{1/3} (M_{\star}-M_\mathrm{pl}) \sin(i_p)/(\sqrt{1-e^2} (M_{\star}+M_\mathrm{pl})^{1/3} ) + rv_\mathrm{sys}   
     
    Args:
        TBD
    
    Returns:
        None
    
    """     
    c_ip = np.cos(inclin)       
    s_ip = np.sin(inclin)       
      
    #True anomaly
    True_anom_loc,Ecc_anom_loc,_=calc_true_anom(ecc,ph_loc,omega_bar)  

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
            VX1_p= omega_p_h*Y1_p    
            VY1_p=-omega_p_h*X1_p

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
        yp_loc = -Y1_p*c_ip
        zp_loc =  Y1_p*s_ip
        if rv_LOS:
            vxp_loc=  VX1_p 
            vyp_loc= -VY1_p*c_ip              
            vzp_loc=  VY1_p*s_ip          
            
            
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
        c_ecc = np.cos(Ecc_anom_loc)
        s_ecc = np.sin(Ecc_anom_loc)        
        X0_p=aRs*(c_ecc-ecc)
        Y0_p=aRs*np.sqrt(1.-ecc*ecc)*s_ecc
    
        #Coordinates in the orbital plane oriented toward the Earth in Rstar
        #Omega_bar : angle between the ascending node and the periastron in the orbital plane
        #  - I1 is the node line (from K1 when rotating with the orbital motion)    
        #    Y1 is the perpendicular to the node line in the orbital plane, toward the Earth
        #  - Z1 = ( sin(omega_bar) # cos(omega_bar)  ) in (I0,J0)
        #    Y1 = ( -cos(omega_bar) # sin(omega_bar)  ) in (I0,J0)
        c_om = np.cos(omega_bar)
        s_om = np.sin(omega_bar)
        X1_p =  -X0_p*c_om + Y0_p*s_om
        Y1_p =   X0_p*s_om + Y0_p*c_om    

        #Coordinates in the plane containing the LOS and the node line
        #   - Z : LOS toward Earth (from I1 by rotating with angle inclin)
        #   - X : node line (=K1)          
        #   - Y : complete the right handed referential
        xp_loc=X1_p            
        yp_loc=-Y1_p*c_ip
        zp_loc= Y1_p*s_ip

        #Velocity coordinates
        #    - in /Rstar/h 
        if rv_LOS:

            #Distance star - planet (/Rstar)
            Dplanet=aRs*(1.-ecc*c_ecc)
 
            #Planet velocity in the orbital plane 
            #  - vx = - a sin(E) dE/dt
            #    vy = a sqrt(1-e^2) cos(E) dE/dt
            #  - dM = omega_p_h*dt = dE ( 1 - e cos(E) ) = dE*Dplanet/a_pl  
            VX0_p =-omega_p_h*(pow(aRs,2.))*s_ecc/Dplanet
            VY0_p = omega_p_h*(pow(aRs,2.))*np.sqrt(1-pow(ecc,2.))*c_ecc/Dplanet    

            #Planet velocity in the orbital plane oriented toward the Earth   
            VX1_p = -VX0_p*c_om + VY0_p*s_om  
            VY1_p =  VX0_p*s_om + VY0_p*c_om
            
            #Coordinates in the sky-projected referential 
            vxp_loc=  VX1_p
            vyp_loc= -VY1_p*c_ip 
            vzp_loc=  VY1_p*s_ip   

    #--------------------------------------------------------------------------------------------------------------------
    #Absolute orbital velocity
    #    - in /Rstar/h 
    if rv_LOS:vp_loc=np.sqrt(VX1_p*VX1_p + VY1_p*VY1_p)
    else:vp_loc=None

    #--------------------------------------------------------------------------------------------------------------------
    #Distance star - planet in the plane of sky
    #    - in /Rstar
    Dprojplanet=np.sqrt(xp_loc*xp_loc + yp_loc*yp_loc)

    #--------------------------------------------------------------------------------------------------------------------
    #Eclipse status  
    if RpRs is not None:ecl_loc = eclipse_status(Dprojplanet,RpRs,lambda_rad,star_params,xp_loc,yp_loc,zp_loc)     
    else:ecl_loc = None
          
    return xp_loc,yp_loc,zp_loc,Dprojplanet,vxp_loc,vyp_loc,vzp_loc , vp_loc, ecl_loc   
    
    
def calc_pl_coord_plots(n_pts_orbit,pl_params,unit = 'Rs'):  
    r"""**Planet orbit: for plots.**
    
    Defines planetary orbit coordinates for plot and contact times.
    Coordinates are defined in the sky-projected orbital frame
        
    .. math:: 
       X_\mathrm{sky} &= \mathrm{node \, line \, of \, orbital \, plane}  \\
       Y_\mathrm{sky} &= \mathrm{projection \, of \, the \, orbital \, plane \, normal} \\
       Z_\mathrm{sky} &= \mathrm{LOS}

    Args:
        TBD
    
    Returns:
        coord_orbit (array): sky-projected coordinates. 
    
    """    
    
    #Orbital properties
    #    - Inclin is the inclination from the LOS to the normal to the orbital plane
    ecc=pl_params['ecc']
    inclin=pl_params['inclin_rad'] 
 
    #Circular orbit  
    if (ecc<1e-4):    
        n_pts_orbit=int(n_pts_orbit)
        
        #True anomaly
        true_anom=2.*np.pi*np.arange(n_pts_orbit+1.)/n_pts_orbit

        #Coordinates in the orbit plane with semi-major axis as X axis =  orbital plane oriented toward the Earth         
        XorbE_norm= np.cos(true_anom)
        YorbE_norm= np.sin(true_anom)
        
    #Elliptic orbit 
    else:
        
        #Time resolution of the orbit
        #    - we set a high resolution around the periastron   
        #    - see planet_coord for details  
        n_pts_horbit=int(n_pts_orbit/2.)
        phase=-(0.05)+(0.1/n_pts_horbit)*np.arange(n_pts_horbit+1)
        phase=np.append(phase,(0.05)+((1-0.1 )/n_pts_horbit)*np.arange(n_pts_horbit+1))

        #Mean anomaly    
        #    - null by definition at the periapsis: we use a higher resolution aroung the periapsis        
        Mean_anom_plot=2.*np.pi*phase
        
        #Eccentric anomaly
        Ecc_anom_plot=np.array([newton(Kepler_func,Mean_anom_plot[i],args=(Mean_anom_plot[i],ecc,)) for i in range(len(Mean_anom_plot))])
    
        #Coordinates in the orbital plane with semi-major axis as X axis    
        Xorb_norm=(np.cos(Ecc_anom_plot)-ecc)
        Yorb_norm=np.sqrt(1.-pow(ecc,2.))*np.sin(Ecc_anom_plot)

        #Conversion from orbital plane with semi-major axis as X axis to orbital plane oriented toward the Earth
        c_om=np.cos(pl_params['omega_rad'])
        s_om=np.sin(pl_params['omega_rad'])
        XorbE_norm =  Xorb_norm*s_om +  Yorb_norm*c_om
        YorbE_norm = -Xorb_norm*c_om +  Yorb_norm*s_om        

    #Conversion from orbital plane oriented toward the Earth, to the sky-projected orbital referential    
    coord_orbit_sky_p = np.array([ YorbE_norm,                  #Xsky
                                  -XorbE_norm*np.cos(inclin),   #Ysky
                                   XorbE_norm*np.sin(inclin)])  #Zsky
    
    #Conversion to units of Rstar or au
    if unit=='Rs':  coord_orbit_sky_p*=pl_params['aRs']
    elif unit=='au':coord_orbit_sky_p*=pl_params['a_planet']
    
    return coord_orbit_sky_p    
    


def calc_zLOS_oblate(x_st_sk,y_st_sk,istar_rad,RpoleReq):
    r"""**LOS coordinate for oblate photosphere.**

    Calculates the `Z` coordinate (along the LOS) of an element at `(X,Y)` over the sky-projected photosphere (inclined star frame) of an oblate star. 
    The `Z` coordinate is returned in the stellar hemispheres facing and opposite the observer.

    Based on Barnes+2009, Eq. 13 and 14 with :math:`\Phi = \pi/2 - i_{\star}`. 
    Beware that their Eq 14 is wrong, as :math:`1-f^2` should be :math:`(1-f)^2`

    Args:
        TBD
    
    Returns:
        None
    
    """    
    
    #The photosphere is defined in the star frame as (Eq 12 in Barnes 2009):
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
    #there is no Req in the equation because coordinates are all normalized by Rstar
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



def excl_plrange(cond_def,range_star_in,iexp,edge_bins,data_format):
    r"""**Planet atmospheric masking.**

    Identifies spectral pixels contaminated by the planetary atmosphere, as requested in input. 
    A common rv range is excluded for each spectral line in the requested planet atmosphere mask 
    (see `init_visit()` for the definition of these ranges in other frames).

    Args:
        TBD
    
    Returns:
        None
    
    """ 
    cond_kept = np.ones(cond_def.shape,dtype=bool)
    idx_excl_bd_ranges = []
    if data_format=='CCF':
        range_star = range_star_in['CCF']
        for pl_loc in range_star:
            idx_excl = np_where1D((edge_bins[0:-1]>=range_star[pl_loc][0,iexp]) & (edge_bins[1:]<=range_star[pl_loc][1,iexp]))
            if len(idx_excl)>0:
                idx_excl_bd_ranges+=[[idx_excl[0],idx_excl[-1]]]
                cond_kept[idx_excl] = False 

    elif 'spec' in data_format:
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


def calc_ar_coord(ar_prop_exp,ang_rad,lat_rad,long_rad,i_pos,istar,star_params):
    r"""**Active region track**

    Calculates position of active regions across the stellar surface, in the aligned star rest frame.
     
    Args:
        TBD
    
    Returns:
        None
    
    """     

    # Active regions latitude - constant in time
    clat_rad = np.cos(lat_rad)
    slat_rad = np.sin(lat_rad)

    # Active regions center coordinates in star rest frame
    #Exposure center
    x_st = np.sin(long_rad)*clat_rad
    y_st = slat_rad
    z_st = np.cos(long_rad)*clat_rad

    #Conversion to sky rest frame
    x_sky,y_sky,z_sky = frameconv_star_to_skystar(x_st,y_st,z_st,istar)

    # Store properties
    ar_prop_exp['lat_rad_exp'][i_pos] = lat_rad
    ar_prop_exp['sin_lat_exp'][i_pos] = slat_rad
    ar_prop_exp['cos_lat_exp'][i_pos] = clat_rad
    ar_prop_exp['long_rad_exp'][i_pos] = long_rad    
    ar_prop_exp['sin_long_exp'][i_pos] = np.sin(long_rad)
    ar_prop_exp['cos_long_exp'][i_pos] = np.cos(long_rad)
    ar_prop_exp['x_sky_exp'][i_pos] = x_sky
    ar_prop_exp['y_sky_exp'][i_pos] = y_sky
    ar_prop_exp['z_sky_exp'][i_pos] = z_sky
    ar_prop_exp['is_visible'][i_pos] = is_ar_visible(istar,long_rad, lat_rad,ang_rad, star_params['f_GD'], star_params['RpoleReq'])
    
    return None



##################################################################################################    
#%%% Time coordinates
################################################################################################## 
  
def get_timeorbit(Tcenter,bjd_tab, param, exp_time_tab,conv_ang=False):
    r"""**Exposure time and phase**

    Calculates time and phase coordinates of planets or active regions at the start, mid, and end of an exposure.

    Args:
        TBD
    
    Returns:
        None
    
    """ 
    
    #Transit center (days)
    Tcen= Tcenter - 2400000.  
    
    #Orbital phase (between +-0.5) and corresponding times (d)
    phase_temp=(bjd_tab-Tcen)/param["period"]
    mid_phase_tab = (phase_temp - npint(phase_temp+np.sign(phase_temp)*0.5))

    #Conversion to angle of time
    if conv_ang:mid_time_tab=mid_phase_tab*2.*np.pi
    else:mid_time_tab=mid_phase_tab*param["period"]

    #Start/end properties
    if exp_time_tab is not None:
        
        #Half exposure time (days)
        texp_d_tab=exp_time_tab/(2.*24.*3600.)

        #Times
        phase_temp=(bjd_tab-texp_d_tab-Tcen)/param["period"]
        st_phase_tab = (phase_temp - npint(phase_temp+np.sign(phase_temp)*0.5))
        st_time_tab  = st_phase_tab*param["period"]
        phase_temp=(bjd_tab+texp_d_tab-Tcen)/param["period"]
        end_phase_tab = (phase_temp - npint(phase_temp+np.sign(phase_temp)*0.5))
        end_time_tab  = end_phase_tab*param["period"]
        if conv_ang:
            st_time_tab*=2.*np.pi
            end_time_tab*=2.*np.pi
        
    else:st_phase_tab,end_phase_tab,st_time_tab,end_time_tab=None,None,None,None

    return st_phase_tab,mid_phase_tab,end_phase_tab,st_time_tab,mid_time_tab,end_time_tab


def conv_phase(coord_dic,inst,vis,system_param,ref_pl,pl_loc,phase_tab_in):
    r"""**Planet phase conversion**

    Converts orbital phase of reference planet for another planet.

    Args:
        TBD
    
    Returns:
        None
    
    """ 
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




def calc_tr_contacts(RpRs,pl_params,stend_ph,star_params):  
    r"""**Transit contacts.**

    Calculates the orbital phase of the four transit contacts for the selected planet.

    Args:
        TBD
    
    Returns:
        None
    
    """     
    ecc=pl_params['ecc']
    aRs=pl_params['aRs']
    omega_bar=pl_params['omega_rad']
    Inclin=pl_params['inclin_rad']
    c_ip = np.cos(Inclin) 
    s_ip = np.sin(Inclin) 
    
    #Defining contacts numerically over a high-resolution phase table around the transit. 
    contact_phases=np.zeros(4,dtype=float)*np.nan
    n_pts_contacts=int(5000)   
    ph_ineg=np.arcsin(1./aRs)/(2.*np.pi)   #phase approximative de l'ingress
    ph_st=-stend_ph*ph_ineg
    ph_end=stend_ph*ph_ineg
    ph_contacts=ph_st+((ph_end-ph_st)/n_pts_contacts)*np.arange(n_pts_contacts+1.)
    if (ecc > 1e-4):
        if 'Mean_anom_TR' not in pl_params:pl_params['Mean_anom_TR'] = calc_mean_anom_TR(ecc,omega_bar) 
        Mean_anom_plot=2.*np.pi*ph_contacts+pl_params['Mean_anom_TR']
        Ecc_anom_plot=np.array([newton(Kepler_func,Mean_anom_plot[i],args=(Mean_anom_plot[i],ecc,)) for i in range(len(Mean_anom_plot))])   
        X0_plot=aRs*(np.cos(Ecc_anom_plot)-ecc)
        Y0_plot=aRs*np.sqrt(1.-pow(ecc,2.))*np.sin(Ecc_anom_plot)
        xp=-X0_plot*np.cos(omega_bar) +  Y0_plot*np.sin(omega_bar)
        yp=-( X0_plot*np.sin(omega_bar) +  Y0_plot*np.cos(omega_bar) )*c_ip
        zp=( X0_plot*np.sin(omega_bar) +  Y0_plot*np.cos(omega_bar) )*s_ip
    else:
        X0_plot= aRs*np.cos(2.*np.pi*ph_contacts)
        Y0_plot= aRs*np.sin(2.*np.pi*ph_contacts)
        xp=Y0_plot
        yp=-X0_plot*c_ip
        zp=X0_plot*s_ip
        
    #Points before transit, front of the star
    w_bef=np_where1D((xp<0) & (zp>0))      

    #Points after transit, front of the star
    w_aft=np_where1D((xp>0) & (zp>0))

    #Oblate star    
    if star_params['f_GD']>0.:

        #Frame conversion of planet coordinates from the classical frame perpendicular to the LOS, to the 'inclined star' frame
        idx_front = np_where1D(zp>0)
        xp_st_sk,yp_st_sk,_=frameconv_skyorb_to_skystar(pl_params['lambda_rad'],xp[idx_front],yp[idx_front],None)              

        #Number of planet limb points within the projected stellar photosphere
        nlimb = 501
        nlimb_in_ph = pl_limb_in_oblate_star(nlimb,RpRs,xp_st_sk,yp_st_sk,star_params)

        #Check for transit status
        if nlimb_in_ph==0:
            print('  No contact points identified')
            contact_phases = None    
        else:

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
    
        #Check for transit status
        if np.sum(Dprojplanet<=(1.+RpRs))==0:
            print('  No contact points identified')
            contact_phases = None    
        else:
        
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
    if contact_phases is not None:
        if w_bef[w_first]==0:stop('ERROR : Start phase is too short for contact determination: increase "plot_dic["stend_ph"]"')
        if w_aft[w_fth]==n_pts_contacts:stop('ERROR : End phase is too short for contact determination: increase "plot_dic["stend_ph"]"')
        contact_phases[0]=ph_contacts[w_bef][w_first]      
        contact_phases[1]=ph_contacts[w_bef][w_scnd]     
        contact_phases[2]=ph_contacts[w_aft][w_thrd]     
        contact_phases[3]=ph_contacts[w_aft][w_fth]    
            
    return contact_phases   



def eclipse_status(Dprojp_all,RpRs,lambda_rad,star_params,xp_sk_all,yp_sk_all,zp_sk_all):
    r"""**Exposure eclipse status.**

    Identify planet transit and occulation status at a given orbital position.
    
     - Post-occultation/pre-transit: ecl = -1
     - Ingress of transit: ecl = +2
     - Full transit: ecl = +3
     - Egress of transit: ecl = +2
     - Post-transit/pre-occultation: ecl = +1
     - Ingress of occultation: ecl = -2
     - Full occultation: ecl = -3
     - Egress of occultation: ecl = -2

    To be conservative, an exposure is considered out of the transit/occultation, or fully in-transit/occultation if it is such during the entire exposure.
    Input planet position either correspond to the center of the exposure, or to its (start,central,end) values.
     
    Args:
        TBD
    
    Returns:
        None
    
    """ 
    dim_input = np.shape(xp_sk_all)
    if len(dim_input)==1:exact_pos = True
    else:exact_pos = False
    
    #Oblate star
    if star_params['f_GD']>0.:
        nlimb = 501
        if exact_pos:
            nexp = len(xp_sk_all)
            x_st_sky_all,y_st_sky_all,_=frameconv_skyorb_to_skystar(lambda_rad,xp_sk_all,yp_sk_all,None) 
            nlimb_in_ph = np.array([ pl_limb_in_oblate_star(nlimb,RpRs,[x_st_sky],[y_st_sky],star_params)[0] for x_st_sky,y_st_sky in zip(x_st_sky_all,y_st_sky_all) ])   
        else:
            nexp = len(xp_sk_all[0])

            #Frame conversion of planet coordinates from the classical frame perpendicular to the LOS, to the 'inclined star' frame
            start_x_st_sky_all,start_y_st_sky_all,_=frameconv_skyorb_to_skystar(lambda_rad,xp_sk_all[0],yp_sk_all[0],None)           
            end_x_st_sky_all,end_y_st_sky_all,_=frameconv_skyorb_to_skystar(lambda_rad,xp_sk_all[2],yp_sk_all[2],None) 
    
            #Number of planet limb points within the projected stellar photosphere
            nlimb_in_ph = np.zeros(nexp,dtype=float)
            for isub,(start_x_st_sky,start_y_st_sky,end_x_st_sky,end_y_st_sky) in enumerate(zip(start_x_st_sky_all,start_y_st_sky_all,end_x_st_sky_all,end_y_st_sky_all)):
                start_nlimb_in_ph = pl_limb_in_oblate_star(nlimb,RpRs,[start_x_st_sky],[start_y_st_sky],star_params)[0]
                end_nlimb_in_ph = pl_limb_in_oblate_star(nlimb,RpRs,[end_x_st_sky],[end_y_st_sky],star_params)[0]
                nlimb_in_ph[isub] = start_nlimb_in_ph+end_nlimb_in_ph
    
        #Planetary disk is outside the projected stellar photosphere
        cond_out = (nlimb_in_ph==0)

        #Planet is entirely in front of the disk  
        cond_in = (nlimb_in_ph==2*nlimb)
            
    #Spherical star    
    else:
        if exact_pos:
            nexp = len(Dprojp_all)
            cond_out = (Dprojp_all > (1.+RpRs)) 
            cond_in =  (Dprojp_all <= (1.-RpRs))  
        else:
            nexp = len(Dprojp_all[0])
            cond_out = (Dprojp_all[0] > (1.+RpRs)) & (Dprojp_all[2] > (1.+RpRs))
            cond_in = (Dprojp_all[0] <= (1.-RpRs)) & (Dprojp_all[2] <= (1.-RpRs))
       
    #Planet is in front or behind the sky-projected plane
    if exact_pos:cond_tr = (zp_sk_all>0.)
    else:cond_tr = (0.5*(zp_sk_all[0]+zp_sk_all[2])>0.)
    cond_occ = ~cond_tr
    
    #Default (ingress/egress of transit and occultation)
    eclipse = np.repeat(2.,nexp)      
    eclipse[cond_occ]*=-1.  
    
    #Planet is entirely in front of the disk (full transit) 
    eclipse[cond_in & cond_tr] = 3.

    #Planet is entirely behind the disk (full occultation) 
    eclipse[cond_in & cond_occ] = -3.    

    #Planetary disk is outside the projected stellar photosphere
    eclipse[cond_out] = 1.

    #Post-occultation and pre-transit 
    if exact_pos:eclipse[xp_sk_all<0.]*=-1.
    else:eclipse[0.5*(xp_sk_all[0]+xp_sk_all[2])<0.]*=-1.
                                    
    return eclipse


def pl_limb_in_oblate_star(nlimb,RpRs,xp_st_sk,yp_st_sk,star_params):
    r"""**Transit contacts.**

    Calculates the number of discretized planet limb points within the projected oblate stellar photosphere, for a given orbital position.

    Args:
        TBD
    
    Returns:
        None
    
    """ 
    xlimb = RpRs*np.cos(2*np.pi*np.arange(nlimb)/(nlimb-1.))
    ylimb = RpRs*np.sin(2*np.pi*np.arange(nlimb)/(nlimb-1.))
    nlimb_in_ph = np.zeros(len(xp_st_sk),dtype=int)
    for iloc,(xp_st_sk_loc,yp_st_sk_loc) in enumerate(zip(xp_st_sk,yp_st_sk)):
        x_st_sk_limb = xp_st_sk_loc+xlimb
        y_st_sk_limb = yp_st_sk_loc+ylimb
        cond_in_stphot=calc_zLOS_oblate(x_st_sk_limb,y_st_sk_limb,star_params['istar_rad'],star_params['RpoleReq'])[2] 
        nlimb_in_ph[iloc] = np.sum( cond_in_stphot )    
    return nlimb_in_ph

def is_ar_visible(istar, long_rad, lat_rad, ang_rad, f_GD, RpoleReq) :
    r"""**Active region visibility**

    Performs a rough estimation of the visibility of an active region, based on star inclination (istar), active region coordinates in star rest frame (long, lat) and active region angular size (ang).
    To do so, we discretize the active region edge and check if at least one point is visible. The visibility criterion is derived as follows. 

    Let :math:`\mathrm{P} = (x_{\mathrm{st}}, y_{\mathrm{st}}, z_{\mathrm{st}})` be a point on the stellar surface, in the star rest frame (i.e. X axis parallel to the star 
    equator and  Y axis along the stellar spin), that is on the edge of an active region.
    
    Expressing P in spherical coordinates gives

    .. math::
        & x_{\mathrm{st}} = \mathrm{sin}(long) \mathrm{cos}(lat)
        & y_{\mathrm{st}} = \mathrm{sin}(lat)
        & z_{\mathrm{st}} = \mathrm{cos}(long) \mathrm{cos}(lat)
    
    Moving P to the 'inclined' star rest frame gives 

    .. math::
        & x_{\mathrm{sky}} = x_{\mathrm{st}}
        & y_{\mathrm{sky}} = \mathrm{sin}(i_*) y_{\mathrm{st}} - \mathrm{cos}(i_*) z_{\mathrm{st}}
        & z_{\mathrm{sky}} = \mathrm{cos}(i_*) y_{\mathrm{st}} + \mathrm{sin}(i_*) z_{\mathrm{st}}
    
    The condition for P to be visible is then :math:`z_{\mathrm{sky}} > 0` or 
    
    .. math::
        & \mathrm{cos}(i_*) \mathrm{sin}(lat) + \mathrm{sin}(i_*) \mathrm{cos}(long) \mathrm{cos}(lat) > 0

    WIP : Doing this with gravity darkening

    Args:
        istar (float) : stellar inclination (in radians)
        long_rad (float) : active region longitude (in radians)
        lat_rad (float) : active region latitude (in radians)
        ang_rad (float) : active region half-angular size (in radians)
        f_GD (float) : oblateness coefficient.
        RpoleReq (float) : pole to equatoral radius ratio.
     
    Returns:
        ar_visible (bool) : active region visibility criterion.
        
    """ 
    ar_visible = False
    
    for arg in np.linspace(0,2*np.pi, 20) :
        
        #Define the edges of the active regions
        long_edge = long_rad + ang_rad * np.sin(arg)
        lat_edge  = lat_rad  + ang_rad * np.cos(arg)

        #Define the corresponding x, y, and z coordinates of the edges of the active regions in the un-inclined star rest frame
        x_st_edge = np.sin(long_edge)*np.cos(lat_edge)
        y_st_edge = np.sin(lat_edge)
        z_st_edge = np.cos(long_edge)*np.cos(lat_edge)

        #Move the x, y, and z coordinates of the active region edges into the inclined star rest frame 
        x_sky_edge, y_sky_edge, z_sky_edge = frameconv_star_to_skystar(x_st_edge, y_st_edge, z_st_edge, istar)

        ##### WIP #####
        #Checking the value of the z coordinate for each edge point to see if the point is visible
        if f_GD>0:
            #We take the modulo in case the active region is plotted over a very long time.
            long_edge_deg = (long_edge * 180/np.pi)%360
            #Checking if the active region is in the front - rough estimate that doesn't account for the star inclination or active region latitude
            additive = 90 - (istar * 180/np.pi)
            first_criterion = (-90-additive <= long_edge_deg and long_edge_deg <= 90+additive) or (270-additive <= long_edge_deg and long_edge_deg <=360) or (-360 <= long_edge_deg and long_edge_deg <= -270+additive)
            #Now that we known the active region is in front, combine with the condition "is in stellar photosphere"
            criteria = calc_zLOS_oblate(np.array([x_sky_edge]),np.array([y_sky_edge/(1-f_GD)]),istar, RpoleReq)[2]
            criterion = first_criterion and criteria
        ##########
        else:
            criterion = (z_sky_edge > 0)
        
        ar_visible |= criterion

    return ar_visible













##################################################################################################    
#%% Frame change
################################################################################################## 

def frameconv_skyorb_to_skystar(lambd,x_skorb,y_skorb,z_skorb):
    r"""**Frame: sky-projected, orbital to stellar.**

    Converts coordinates from the classical sky-projected orbital frame, to the sky-projected (inclined) stellar frame.
    Frame is rotated around the :math:`Z_\mathrm{LOS}` axis (the LOS, which remains unchanged), by angle :math:`\lambda` counted :math:`>0` counterclockwise
    from the :math:`X_\mathrm{sky,star}` to the :math:`X_\mathrm{sky}` axis, or from the :math:`Y_\mathrm{sky,star}` to the :math:`Y_\mathrm{sky}` axis, in the :math:`XY_\mathrm{sky}` plane.

    .. math:: 
       X_\mathrm{sky,star} &= \mathrm{sky-projected \, stellar \, equator}  \\
       Y_\mathrm{sky,star} &= \mathrm{sky-projected \, stellar \, spin} \\
       Z_\mathrm{sky,star} &= \mathrm{LOS}

    Args:
        TBD
    
    Returns:
        None
    
    """ 
    c_lamb=np.cos(lambd) 
    s_lamb=np.sin(lambd) 
    x_skst = x_skorb*c_lamb - y_skorb*s_lamb
    y_skst = x_skorb*s_lamb + y_skorb*c_lamb
    z_skst=deepcopy(z_skorb)
    return x_skst,y_skst,z_skst

def frameconv_skystar_to_skyorb(lambd,x_skst,y_skst,z_skst):
    r"""**Frame: sky-projected, stellar to orbital.**

    Converts coordinates from the sky-projected (inclined) stellar frame to the classical sky-projected orbital frame.

    Args:
        TBD
    
    Returns:
        None
    
    """ 
    c_lamb=np.cos(lambd) 
    s_lamb=np.sin(lambd) 
    x_skorb =  x_skst*c_lamb+y_skst*s_lamb
    y_skorb = -x_skst*s_lamb+y_skst*c_lamb
    z_skorb=deepcopy(z_skst)  
    return x_skorb,y_skorb,z_skorb



def frameconv_skystar_to_star(xin,yin,zin,istar):
    r"""**Frame: sky-projected stellar to stellar.**

    Converts coordinates from the sky-projected (inclined) stellar frame to the star frame. 
    Frame is rotated around the :math:`X_\mathrm{sky,star}` axis (which remains unchanged), by angle :math:`i_{\star}` counted :math:`>0` counted counterclockwise
    from the :math:`Z_\mathrm{sky,star}` to the :math:`Y_\star` axis in the :math:`YZ_\mathrm{sky,star}` plane.

    .. math:: 
       X_\mathrm{star} &= \mathrm{star \,equator} \\
       Y_\mathrm{star} &= \mathrm{star \, spin} \\
       Z_\mathrm{star} &= \mathrm{complements \, frame}  

    Args:
        TBD
    
    Returns:
        None
    
    """ 
    xout=deepcopy(xin)
    yout= np.sin(istar)*yin + np.cos(istar)*zin
    zout=-np.cos(istar)*yin + np.sin(istar)*zin   
    return xout,yout,zout

def frameconv_star_to_skystar(xin,yin,zin,istar):
    r"""**Frame: stellar to  sky-projected stellar.**

    Converts coordinates from the star frame to the sky-projected (inclined) stellar frame.

    Args:
        TBD
    
    Returns:
        None
    
    """ 
    xout=deepcopy(xin)
    yout= np.sin(istar)*yin - np.cos(istar)*zin
    zout= np.cos(istar)*yin + np.sin(istar)*zin   
    return xout,yout,zout





