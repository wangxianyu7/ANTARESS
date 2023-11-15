import numpy as np
from scipy.optimize import newton



def Kepler_func(Ecc_anom,Mean_anom,ecc):
    r"""**Eccentric anomaly solver.**
    
    Find the eccentric anomaly of the orbit using the mean anomaly and eccentricity
    
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



   
def Mean_anom_TR_calc(ecc,omega_bar):  
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
    



  
def def_plotorbite(n_pts_orbit,pl_params):  
    r"""**Plotted orbit.**
    
    Defines planetary orbite coordinates for plot and contact times.
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






