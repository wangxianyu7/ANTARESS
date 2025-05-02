#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from itertools import product as it_product
import lmfit
from copy import deepcopy
from ..ANTARESS_general.utils import stop,planck
from ..ANTARESS_general.constant_data import G_usi,Msun,Rsun
from ..ANTARESS_grids.ANTARESS_coord import calc_zLOS_oblate,frameconv_skystar_to_star

################################################################################################################    
#%% Radial velocity field
################################################################################################################  

def calc_RVrot(x_st_sky,y_st,istar_rad,veq,alpha_rot,beta_rot):
    r"""**Stellar rotational rv**

    Calculates radial velocity of stellar surface element from rotation (in km/s).     
    The absolute and radial velocity depend on stellar latitude in presence of differential rotation.

    .. math::  
       v &= \Omega R_\mathrm{\star} \\
         &= \Omega_\mathrm{eq} (1-\alpha_\mathrm{rot} y_\mathrm{lat}^2 - \beta_\mathrm{rot} y_\mathrm{lat}^4) R_\mathrm{\star} \\           
         &= v_\mathrm{eq} (1-\alpha_\mathrm{rot} y_\mathrm{lat}^2 - \beta_\mathrm{rot} y_\mathrm{lat}^4) 
         
    The velocity vector in the star frame is defined as

    .. math::    
       v_\mathrm{x,star} &=  v \cos(\Phi)   \\
       v_\mathrm{y,star} &= -v \sin(\Phi)   \\
       v_\mathrm{z,star} &=  0 
        
    Where :math:`\Phi` is the angle between the LOS `z` and the surface element in the `zx` plane.
    The velocity vector in the inclined star frame is then
    
    .. math::    
       v_\mathrm{x,sky star} &=  v \cos(\Phi)   \\
       v_\mathrm{y,sky star} &= -v \sin(\Phi) cos(i_\star)   \\
       v_\mathrm{z,sky star} &= -v \sin(\Phi) sin(i_\star) 

    And the radial velocity along the :math:`z_\mathrm{sky star}` axis, defined as negative toward the observer, is then
    
    .. math:: 
       rv &= - v_\mathrm{z,sky star}    \\
          &= v \sin(\Phi) sin(i_\star)    \\
          &= v x_\mathrm{norm} sin(i_\star)    \\          
          &= x_\mathrm{norm} v_\mathrm{eq} sin(i_\star) (1-\alpha_\mathrm{rot} y_\mathrm{lat}^2 - \beta_\mathrm{rot} y_\mathrm{lat}^4)  

    Args:
        TBD
    
    Returns:
        TBD
    
    """   
    Vrot = veq*(1.-alpha_rot*y_st**2.-beta_rot*y_st**4.)
    RVrot = x_st_sky*Vrot*np.sin(istar_rad) 
    return RVrot,Vrot



def calc_CB_RV(ld_coeff,LD_mod,c1_CB,c2_CB,c3_CB,star_params):
    r"""**Stellar convective blueshift**

    Calculates radial velocity of stellar surface element from convective blueshift.

    Args:
        TBD
    
    Returns:
        TBD
    
    """     
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


################################################################################################################    
#%% Intensity field
################################################################################################################  

def calc_GD(x_grid_star,y_grid_star,z_grid_star,star_params_eff,gd_band,x_st_sky_grid,y_st_sky_grid):
    r"""**Gravity-darkening intensity**

    Calculates blackbody emission with gravity-darkening, using the formalism from Barnes+2009.
    It accounts from gravity-darkening through temperature variation and local blackbody emission.

    Args:
        TBD
        star_params_eff (dict) : dictionary containing nominal or variable (overwritten from fit routines) stellar properties
    
    Returns:
        TBD
    
    """ 
    
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
    Rstar_m = star_params_eff['Rstar_km']*1e3                                 
    gr_R = -G_usi*star_params_eff['Mstar']*Msun/(r_grid_star*Rstar_m)**3.
    grp_Rp = star_params_eff['om_eq']**2.
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
    RpoleReq2 = star_params_eff['RpoleReq']**2.
    ci = np.cos(star_params_eff['istar_rad'])
    si = np.sin(star_params_eff['istar_rad'])
    mRp2 = (1. - RpoleReq2)
    Aquad = 1. - si**2.*mRp2
    Bquad = 2.*y_st_sky_grid*ci*si*mRp2
    Cquad = y_st_sky_grid**2.*si**2.*mRp2 + star_params_eff['RpoleReq']**2.*(x_st_sky_grid**2.+ y_st_sky_grid**2. - 1.) 
    det = Bquad**2.-4.*Aquad*Cquad
    Nx_st_sky_grid2 = ( RpoleReq2*2.*x_st_sky_grid )**2./np.abs(det)
    Ny_st_sky_grid2 = ((1./Aquad)*( -(ci*si*mRp2)   + 2.*y_st_sky_grid*RpoleReq2/np.sqrt(det) ))**2.    
    Nz_st_sky_grid = 1.
    N_grid_star = np.sqrt(Nx_st_sky_grid2 + Ny_st_sky_grid2 + Nz_st_sky_grid*Nz_st_sky_grid )
    mu_grid_star = Nz_st_sky_grid/N_grid_star

    #Calculate gravity-darkening
    if len(gd_band)>0:
        y_pole_star = star_params_eff['RpoleReq']
        gr_R_pole = -G_usi*star_params_eff['Mstar']*Msun/(y_pole_star*Rstar_m)**3.
        g_pole_star = np.abs(y_pole_star*gr_R_pole)  
    
        #Stellar cell temperature
        #    - from Maeder+2009
        temp_grid_star = star_params_eff['Tpole']*(g_grid_star/g_pole_star)**star_params_eff['beta_GD']
    
        #Wavelength table for the band, in A
        nw_band=int((gd_band['wmax']-gd_band['wmin'])/gd_band['dw'])
        dw_band=(gd_band['wmax']-gd_band['wmin'])/nw_band
        wav_band=gd_band['wmin']+0.5*dw_band+dw_band*np.arange(nw_band)            
        
        #Black body flux
        #    - at star surface, but absolute flux does not matter, so we scale by a constant factor to get values around 1
        gd_grid_star = np.sum(planck(wav_band[:,None],temp_grid_star)[0]*dw_band,axis=0)/1e11            

    else:
        gd_grid_star = np.ones(mu_grid_star.shape,dtype=float)

    return gd_grid_star,mu_grid_star
























def get_LD_coeff(transit_prop,iband):
    r"""**Limb-Darkening coefficients**

    Store input limb-darkening coefficients in common structure [LD_u1,LD_u2,LD_u3,LD_u4].

    Args:
        transit_prop (dict) : dictionary containing the planet/active region limb-darkening properties.
        iband (int) : index of the band considered.

    Returns:
        ld_coeff (list) : formatted list of the limb-darkening coefficients used.
    
    """     
    
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






def calc_LD(LD_mod,mu,ld_coeff):
    r"""**Limb-Darkening intensity**

    Calculates limb-Darkening value at a given :math:`\mu` (from 1 at stellar center to 0 at the limb).

    Args:
        TBD
    
    Returns:
        TBD
    
    """   
    
    #Uniform emission 
    if LD_mod == 'uniform':
        ld_val = 1.
        
   	#Linear law
   	#    - I(mu)=I0*(1.-c1*(1.-mu))	
    elif LD_mod == 'linear':
        ld_val = 1. - ld_coeff[0]*(1. -mu)
       
   	#Quadratic law
   	#    - Kopal (1950, Harvard Col. Obs. Circ., 454, 1)
   	#    - I(mu)=I0*(1.-c1*(1.-mu)-c2*pow(1.-mu,2.))    
    elif LD_mod == 'quadratic':
        ld_val = 1. - ld_coeff[0]*(1. -mu) - ld_coeff[1]*np.power(1. -mu,2.)
        
    elif LD_mod == 'squareroot':
        ld_val = 1. - ld_coeff[0]*(1. -mu) - ld_coeff[1]*(1. - np.sqrt(mu))
        
   	#Four-parameter law
   	#    - Claret, 2000, A&A, 363, 1081
   	#    - I(mu)=I0*(1.-c1*(1.-sqrt(mu))-c2*(1.-mu)-c3*(1.-pow(mu,3./2.))-c4*(1.-pow(mu,2.)))    
    elif LD_mod == 'nonlinear': 
        ld_val = 1. - ld_coeff[0]*(1. -mu**0.5) - ld_coeff[1]*(1. -mu) - ld_coeff[2]*(1. -mu**1.5) - ld_coeff[3]*(1. -mu**2.)
        
    elif LD_mod == 'power2':        
        ld_val = 1. - ld_coeff[0]*(1. -mu**ld_coeff[1]) 

   	#Three-parameter LD law 
   	#    - Sing et al., 2009, A&A, 505, 891
   	#    - I(mu)=I0*(1.-c1*(1.-mu)-c2*(1.-pow(mu,3./2.))-c3*(1.-pow(mu,2.))) 
    elif LD_mod=='LD_Sing':        
        ld_val = 1. - ld_coeff[0]*(1.-mu)-ld_coeff[1]*(1.-mu**(3./2.))-ld_coeff[2]*(1.-mu**2.)
        
    #Solar law
    elif LD_mod == "Sun":        
        norm_ld = 2.*np.pi*(ld_coeff[0]/2. + ld_coeff[1]/3. - ld_coeff[2]/4. + ld_coeff[3]/5. -ld_coeff[4]/5. +ld_coeff[5]/7.)
        ld_val = (ld_coeff[0] + ld_coeff[1]*mu - ld_coeff[2]*mu**2. +ld_coeff[3]*mu**3. -ld_coeff[4]*mu**4. +ld_coeff[5]*mu**5.)/norm_ld
        
    else:
        stop('ERROR: limb-darkening mode '+LD_mod+' not defined. Implement in ANTARESS_star_grid.py > calc_LD()')
        
    return ld_val





def calc_Isurf_grid(iband_list,ngrid_star,system_prop,coord_grid,star_params_eff,Ssub_Sstar,Istar_norm=1.,region = 'star',Ssub_Sstar_ref = None):
    r"""**Stellar intensity grid.**

    Calculates flux and intensity values over the stellar grid, from various contributions.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 

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
        if (star_params_eff['f_GD']>0.):
            if star_params_eff['GD']:gd_band = {'wmin':system_prop['GD_wmin'][iband],'wmax':system_prop['GD_wmax'][iband],'dw':system_prop['GD_dw'][iband]}
            gd_grid_star[:,isubband],mu_grid_star[:,isubband] = calc_GD(coord_grid['x_st'],coord_grid['y_st'],coord_grid['z_st'],star_params_eff,gd_band,coord_grid['x_st_sky'],coord_grid['y_st_sky'])         

        #Spherical star
        else:mu_grid_star[:,isubband] = coord_grid['z_st_sky']

        #Limb-darkening coefficients over the grid
        ld_grid_star[:,isubband]=calc_LD(system_prop['LD'][iband],mu_grid_star[:,isubband],get_LD_coeff(system_prop,iband))

    #Intensity and fluxes from cells over the grid
    #    - the chromatic intensity is normalized so that the mean flux integrated over the defined spectral bands, and summed over the full (normalized) stellar disk, is unity
    # mean(w , sum(cell,F(w,cell))) = 1 
    # mean(w , sum(cell,I(w,cell))*SpSs) = 1 
    # mean(w , sum(cell,I(w,cell))) = 1/SpSs 
    #    - specific intensity at disk center is 1 without LD and GD contributions
    #    - normalized by stellar surface, ie F = I0*Scell/Sstar = 1*dl^2/(pi*Rstar)^2 = (dl/Rstar)^2*(1/pi)
    Isurf_grid_star=ld_grid_star*gd_grid_star
    if Istar_norm==1.:     #calculation for full stellar grid
        Itot_star_chrom = np.sum(Isurf_grid_star,axis=0)
        if system_prop['nw']>1:Istar_norm = np.sum(system_prop['dw'][None,:]*Itot_star_chrom)/np.sum(system_prop['dw'])
        else:Istar_norm = np.mean(Itot_star_chrom)
    
    #Flux values from stellar cells, normalized by total stellar flux
    if region == 'star':Fsurf_grid_star = Isurf_grid_star / Istar_norm                                  #fcell = I*SpSs/Ftot, where Ftot = sum(I*SpSs) = Itot*SpSs                            
    elif region == 'local':Fsurf_grid_star = Isurf_grid_star*Ssub_Sstar / (Istar_norm*Ssub_Sstar_ref)      #fcell = I_pl*SpSs_pl/Ftot = I_pl*SpSs_pl/(Itot*SpSs)

    #Total flux over the full star in each band
    Ftot_star = np.sum(Fsurf_grid_star,axis=0)

    return ld_grid_star,gd_grid_star,mu_grid_star,Fsurf_grid_star,Ftot_star,Istar_norm

################################################################################################################    
#%% Stellar grid
################################################################################################################  

def calc_st_sky(coord_grid,star_params):
    r"""**Sky-projected stellar grid.**

    Calculates coordinates of stellar cells in the sky-projected star rest frame.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    
    #Distance of subcells to star center (squared)
    coord_grid['r2_st_sky']=coord_grid['x_st_sky']*coord_grid['x_st_sky']+coord_grid['y_st_sky']*coord_grid['y_st_sky']

    #Oblate star
    #    - the condition is that the 2nd order equation yielding zst_sky for a given (xst_sky,yst_sky) has at least one solution
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
    #    - the (x_st_sky,y_st_sky) coordinates of the planet-occulted region are the same as that of the planet, but the z_st_sky coordinates differ since the region is on the stellar surface
    #      it must thus be calculated directly and not converted from the planet st_sky coordinates
    nsub_star = np.sum(cond_in_stphot)
    if (nsub_star>0):coord_grid['x_st'],coord_grid['y_st'],coord_grid['z_st']=frameconv_skystar_to_star(coord_grid['x_st_sky'],coord_grid['y_st_sky'],coord_grid['z_st_sky'],star_params['istar_rad'])
    
    return nsub_star









def model_star(mode,grid_dic,grid_type,system_prop_in,nsub_Dstar,star_params_eff,var_stargrid_bulk,var_stargrid_I):
    r"""**Model stellar grid**

    Defines coordinates and intensity values over the stellar grid.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
 
    #Initializing velocity grid
    #    - if active regions are accounted for, the surface rv properties will be stored as cell-dependent grids to allow distinguishing between quiet and active cells
    for key in ['veq','alpha_rot','beta_rot']:grid_dic[key] = star_params_eff[key]    
     
    #Updating bulk stellar grid
    #    - this is only required if :
    # + the stellar cell size is changed from its nominal value, otherwise the same grid defined in ANTARESS_main() > init_gen() can be used
    # + the oblateness of the star is changed from its nominal value
    # + the stellar inclination is changed from its nominal value
    if var_stargrid_bulk:
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
        nsub_star = calc_st_sky(coord_grid,star_params_eff)
    
        #Storing for model fits
        if mode=='grid':
            grid_dic.update({'Ssub_Sstar' : Ssub_Sstar,'nsub_star' : nsub_star})
            for key in ['x','y','z','r2']: grid_dic[key+'_st_sky'] = coord_grid[key+'_st_sky'] 
            for key in ['x','y','z']: grid_dic[key+'_st'] = coord_grid[key+'_st'] 
            grid_dic['r_proj'] = np.sqrt(coord_grid['r2_st_sky'])  

    #Spectral grid
    #    - the grid is used to model disk-integrated profile and thus only need to be chromatic if they are not in CCF format
    #    - all tables nonetheless have the same structure in ncell x nband
    if var_stargrid_bulk or var_stargrid_I:
        grid_type_eff = ['achrom']
        if ('spec' in grid_type) and ('chrom' in system_prop_in):grid_type_eff+=['chrom']
        for key_type in grid_type_eff:
    
            #Intensity grid
            ld_grid_star,gd_grid_star,mu_grid_star,Fsurf_grid_star,Ftot_star,Istar_norm = calc_Isurf_grid(range(system_prop_in[key_type]['nw']),nsub_star,system_prop_in[key_type],coord_grid,star_params_eff,Ssub_Sstar)
        
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


def up_model_star(args,param_in):
    r"""**Update stellar grid**

    Update coordinates and intensity values over the stellar grid.
    This function is called when stellar properties differ from the nominal ones.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    if isinstance(param_in,lmfit.parameter.Parameters):param={par:param_in[par].value for par in param_in}
    else:param=deepcopy(param_in)
    
    #Initializing local stellar properties with nominal ones
    star_params_eff = args['system_param']['star']    
    
    #Updating main stellar properties with variable model parameters 
    #    - properties stored in 'var_stargrid_prop' are defined as model parameters, either with values that differ from the nominal ones, or because they are fitted
    if 'Rstar' in args['var_stargrid_prop']:Rstar_km = param['Rstar']*Rsun
    else:Rstar_km = star_params_eff['Rstar_km']
    for key in ['','_spots','_faculae']:    
        for par in args['var_stargrid_prop'+key]:
            par_ar = par+key
            if ('LD_u' in par):
                ideg = par.split('LD_u')
                args['system_prop']['achrom'].update({'LD_u'+str(ideg):[param['LD_u'+str(ideg)]]}) 
            else:
                star_params_eff[par_ar] = param[par_ar]

                #Updating rotational period or velocity
                if par in ['veq','Peq']:
                    if par=='veq':
                        star_params_eff['Peq'+key] = (2*np.pi*Rstar_km)/(param['veq'+key]*24*3600) 
                        star_params_eff['om_eq'+key]=param['veq'+key]/Rstar_km
                    elif par=='Peq':
                        star_params_eff['veq'+key] = (2*np.pi*Rstar_km)/(param['Peq'+key]*24*3600)  
                        star_params_eff['om_eq'+key]=star_params_eff['veq'+key]/Rstar_km  
            
    #Updating stellar properties derived from main properties
    if 'f_GD' in args['var_stargrid_prop']:star_params_eff['RpoleReq']=1.-star_params_eff['f_GD']
    if 'cos_istar' in args['var_stargrid_prop']:star_params_eff['istar_rad']=np.arccos(star_params_eff['cos_istar'])

    #Updating stellar grid
    #    - physical sky-projected grid and broadband intensity variations     
    #    - only if grid settings or specific stellar properties changed value       
    model_star('grid',args['grid_dic'],[args['type']],args['system_prop'],args['grid_dic']['nsub_Dstar'],star_params_eff,args['var_stargrid_bulk'],args['var_stargrid_I'])

    return None




