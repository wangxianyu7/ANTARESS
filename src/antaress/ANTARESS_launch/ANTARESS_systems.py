#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def get_system_params():
    r"""**Planetary system properties.**    

    Returns dictionary with properties of planetary systems. Format with minimum required properties is 
    
    >>> all_system_params={
        'star_name':{
            'star':{
                'Rstar': val,
                'istar': val,
                'veq':   val 
                },      
            'planet_b':{
                'period':    val,
                'TCenter':   val,
                'ecc':       val,     
                'omega_deg': val,          
                'Kstar':     val
            },           
        },
    }
      
    Additional planets can be associated with a given system. Required and optional properties are defined as follows.
    
    **Star properties**
    
     - `veq` [km/s] : equatorial rotational velocity 
     
     - `istar` [deg] : stellar inclination 
     
         + angle counted from the LOS toward the stellar spin axis
         + defined in [ 0 ; 180 ] deg
      
     - `dstar` [pc]: distance Sun-star

     - `Mstar` [Msun] : stellar mass 
        
         + only used if `Kstar` is not defined
         
     - `Rstar` [Rsun]: stellar radius
        
         + used to calculate the planet orbital rv in the star rest frame
         + used as `Req` if the star's oblateness is accounted for
         
      - `alpha_rot`, `beta_rot` [] : differential rotation coefficients 
     
          + differential rotation is defined as :math:`\Omega = \Omega_\mathrm{eq} (1-\alpha_\mathrm{rot} y_\mathrm{lat}^2-\beta_\mathrm{rot} y_\mathrm{lat}^4)`
          + the pipeline checks that :math:`\alpha_\mathrm{rot}+\beta_\mathrm{rot}<1` (assuming that the star rotates in the same direction at all latitudes)
          + defined in percentage of rotation rate
         
      - `ci` [km/s] : coefficients of the :math:`\mu`-dependent convective blueshift velocity polynomial
    
          + convective blueshift is defined as :math:`\mathrm{rv}_\mathrm{cb} = \sum_{i}{c_i \mu^i}` 
          + :math:`c_0` is degenerate with higher-order coefficients and defined independently
          
      - `V` [] : stellar magnitude 
           
           + assumed to correspond  to the achromatic band of the transit light curve
           + only used in plots to estimate errors on surface rv
          
      - `f_GD` [] : stellar oblateness
    
          + defined as :math:`(R_\mathrm{eq}-R_\mathrm{pole})/R_\mathrm{eq}`
         
      - `Tpole` [K] : stellar temperature at pole 
     
          + used to account for gravity-darkening
          + used as effective temperature for stellar atmosphere grid and mask generation
         
      - `beta_GD` [K] : gravity-darkening coefficient 
     
      - `A_R`, `ksi_R`, (`A_T`, `ksi_T`) : radial–tangential macroturbulence properties
        `eta_R`, (`eta_T`) : anisotropic faussian macroturbulence properties 
       
          + only used if macroturbulence is activated, for analytical intrinsic profiles
          + set `_T` parameters equal to `_R` for isotropic macroturbulence
         
      - `logg` [log(cgs)] : surface gravity of the star 
     
          + only used for the stellar atmosphere grid    
         
      - `vmic` and `vmac` [km/s] : micro and macroturbulence 
     
          + only used for the stellar atmosphere grid 
     
        
    **Planet properties**

      - `period` [days] : orbital period 
     
      - `TCenter` [bjd] : epoch of transit center 
     
          + calculated from `Tperi` if unknown, and orbit is eccentric
         
      - `Tperi` [bjd] : epoch of periastron 
     
      - `TLength` [days] : duration of transit 
     
          + only used to estimate the range of the phase table in the light curve plot
         
      - `ecc` [] : orbital eccentricity 
     
      - `omega_deg` [deg] : argument of periastron
     
      - `inclination` [deg] : orbital inclination
     
          + counted from the LOS to the normal of the orbital plane     
          + defined in [ 0 ; 90 ] deg      
         
      - `lambda_proj` [deg] : sky-projected obliquity
     
          + angle counted from the sky-projected stellar spin axis (or rather, from the axis :math:`Y_{\star \mathrm{sky}}`, which corresponds to the projection of :math:`Y_{\star}` when :math:`i_{\star} = i_{\star \mathrm{in}}` in [ 0 ; 180 ]) toward the sky-projected normal to the orbital plane
          + defined in [ -180 ; 180 ] deg        
         
      - `Kstar` [m/s] : rv semi-amplitude induced by the planet on the star 
     
      - `Msini` [Mjup] : Mpl*sin(inclination) 
     
          + only used to calculate `Kstar` if undefined  
         
      - `aRs` [Rs] : semi-major axis 
     
    
    Notes
    -----

      - this is a note about degeneracies on a planetary system orbital architecture.
        See the definition of the axis systems in Bourrier+2024 (ANTARESS I).
     
      - we distinguish between the values given as input here for :math:`i_\mathrm{p,in}` ([0;90]), :math:`\lambda_\mathrm{in}` (x+[0;360]), and :math:`i_{\star \mathrm{in}}` ([0;180]) and the true angles that can take any value between 0 and 360.
        Note that the angle ranges could be inverted between :math:`\lambda` and :math:`i_{\star}`.
        The reason is that degeneracies prevent distinguishing some of the true angle values, except in specific cases.
          
      - the true 3D spin-orbit angle is defined as :math:`\psi = \arccos(\sin(i_{\star}) \cos(\lambda) \sin(i_p) + \cos(i_{\star}) \cos(i_p))`. 
     
      - with solid body rotation, any :math:`i_{\star}` is equivalent, and there is a degeneracy between
     
          1. :math:`(i_\mathrm{p},\lambda) = (i_\mathrm{p,in},\lambda_\mathrm{in})` associated with :math:`\omega = \lambda_\mathrm{in}`
          2. :math:`(i_\mathrm{p},\lambda) = (\pi-i_\mathrm{p,in},-\lambda_\mathrm{in})` associated with :math:`\omega = -\lambda_\mathrm{in}`
          
        Both solutions yield the same :math:`x_{\star \mathrm{sky}}` coordinate.
        :math:`\omega` is the longitude of the ascending node of the planetary orbit (taking the ascending node of the stellar equatorial plane as reference).  
          
      - if the absolute value of :math:`\sin(i_{\star})` is known, via the combination of :math:`P_\mathrm{eq}` and :math:`v_\mathrm{eq} \sin(i_{\star})`, then the following configurations are degenerate. 
        The coordinate probed by the velocity field is :math:`x_{\star \mathrm{sky}} \sin(i_{\star})`.
        The other coordinate probed is :math:`|\sin(i_{\star})|`.
        Only combinations of angles that conserve both quantities are allowed.  
          
          1. :math:`(i_\mathrm{p},\lambda,i_{\star}) = (i_\mathrm{p,in}   , \lambda_\mathrm{in}   , i_{\star \mathrm{in}})`
          
                  default :math:`x_{\star \mathrm{sky}},y_{\star \mathrm{sky}},y_{\star}`.  
                 
                  default :math:`x_{\star \mathrm{sky}} \sin(i_{\star \mathrm{in}})` and :math:`|\sin(i_{\star \mathrm{in}})|`.  
                 
                  default :math:`\psi = \psi_\mathrm{in}`.     
                 
            :math:`(i_\mathrm{p},\lambda,i_{\star}) = (i_\mathrm{p,in}   , \pi+\lambda_\mathrm{in}   , -i_{\star \mathrm{in}})` 
            
                  same as 1. rotated around the LOS: yields :math:`-x_{\star \mathrm{sky}},-y_{\star \mathrm{sky}},y_{\star},\psi_\mathrm{in}`  
                 
            :math:`(i_\mathrm{p},\lambda,i_{\star}) = (-i_\mathrm{p,in}   , \pi+\lambda_\mathrm{in}   , \pi+i_{\star \mathrm{in}})` 
            
                  same as 1. rotated around the LOS: yields :math:`-x_{\star \mathrm{sky}},-y_{\star \mathrm{sky}},y_{\star},\pi-\psi_\mathrm{in}` 
                 
          2. :math:`(i_\mathrm{p},\lambda,i_{\star}) = (\pi-i_\mathrm{p,in}   , -\lambda_\mathrm{in}   , \pi-i_{\star \mathrm{in}})` 
         
                  yields :math:`x_{\star \mathrm{sky}},-y_{\star \mathrm{sky}}, -y_{\star},\psi_\mathrm{in}` 
                 
            :math:`(i_\mathrm{p},\lambda,i_{\star}) = (\pi+i_\mathrm{p,in}   , -\lambda_\mathrm{in}   , i_{\star \mathrm{in}})` 
            
                  yields :math:`x_{\star \mathrm{sky}},-y_{\star \mathrm{sky}}, -y_{\star},\pi-\psi_\mathrm{in}`  
                 
            :math:`(i_\mathrm{p},\lambda,i_{\star}) = (\pi-i_\mathrm{p,in}   , \pi-\lambda_\mathrm{in}   , \pi+i_{\star \mathrm{in}})` 
            
                  same as 2. rotated around the LOS: yields :math:`-x_{\star \mathrm{sky}},y_{\star \mathrm{sky}},-y_{\star},\psi_\mathrm{in}` 
                 
            :math:`(i_\mathrm{p},\lambda,i_{\star}) = (\pi+i_\mathrm{p,in}   , \pi-\lambda_\mathrm{in}   , -i_{\star \mathrm{in}})` 
            
                  same as 2. rotated around the LOS: yields :math:`-x_{\star \mathrm{sky}},y_{\star \mathrm{sky}},-y_{\star},\pi-\psi_\mathrm{in}`
                 
          3. :math:`(i_\mathrm{p},\lambda,i_{\star}) = (i_\mathrm{p,in}   , \lambda_\mathrm{in}   , \pi-i_{\star \mathrm{in}})` 
         
                  yields :math:`x_{\star \mathrm{sky}},y_{\star \mathrm{sky}}, y_{\star,2} = \sin(i_{\star \mathrm{in}}) y_{\star \mathrm{sky}} - \cos(i_{\star \mathrm{in}}) \sin(i_\mathrm{p,in}), \psi_2 = \arccos(\sin(i_{\star \mathrm{in}}) \cos(\lambda_\mathrm{in}) \sin(i_\mathrm{p,in}) - \cos(i_{\star \mathrm{in}}) \cos(i_\mathrm{p,in}))`,     
                  where :math:`(y_{\star,2},\psi_2)` are associated with another possible system configuration that cannot be related to :math:`(y_{\star},\psi)`
            
            :math:`(i_\mathrm{p},\lambda,i_{\star}) = (i_\mathrm{p,in}   , \pi+\lambda_\mathrm{in}   , \pi+i_{\star \mathrm{in}})` 
            
                  same as 3. rotated around the LOS: yields :math:`-x_{\star \mathrm{sky}},-y_{\star \mathrm{sky}},y_{\star,2},\psi_2`
            
            :math:`(i_\mathrm{p},\lambda,i_{\star}) = (-i_\mathrm{p,in}   , \pi+\lambda_\mathrm{in}   , -i_{\star \mathrm{in}})` 
            
                  same as 3. rotated around the LOS: yields :math:`-x_{\star \mathrm{sky}},-y_{\star \mathrm{sky}},y_{\star,2},\pi - \psi_2`
                 
          4. :math:`(i_\mathrm{p},\lambda,i_{\star}) = (\pi-i_\mathrm{p,in}   , -\lambda_\mathrm{in}   , i_{\star \mathrm{in}})`
         
                  yields :math:`x_{\star \mathrm{sky}},-y_{\star \mathrm{sky}},-y_{\star,2},\psi_2`
            
            :math:`(i_\mathrm{p},\lambda,i_{\star}) = (\pi+i_\mathrm{p,in}   , -\lambda_\mathrm{in}   , \pi-i_{\star \mathrm{in}})` 
            
                  yields :math:`x_{\star \mathrm{sky}},-y_{\star \mathrm{sky}},-y_{\star,2},\pi-\psi_2`
            
            :math:`(i_\mathrm{p},\lambda,i_{\star}) = (\pi-i_\mathrm{p,in}   , \pi-\lambda_\mathrm{in}   , -i_{\star \mathrm{in}})` 
            
                  same as 4. rotated around the LOS: yields :math:`-x_{\star \mathrm{sky}},y_{\star \mathrm{sky}},-y_{\star,2},\psi_2`   
            
            :math:`(i_\mathrm{p},\lambda,i_{\star}) = (\pi+i_\mathrm{p,in}   , \pi-\lambda_\mathrm{in}   , \pi+i_{\star \mathrm{in}})` 
            
                  same as 4. rotated around the LOS: yields :math:`-x_{\star \mathrm{sky}},y_{\star \mathrm{sky}},-y_{\star,2},\pi-\psi_2` 
             
      - if the absolute stellar latitude is constrained (eg with differential rotation), then there is a degeneracy between two configurations, which yield the same :math:`\psi`. 
        The coordinate probed by the velocity field is :math:`x_{\star \mathrm{sky}} \sin(i_{\star})`.
        The other coordinate probed is :math:`y_{\star}^2`
        Only combinations of angles that conserve both quantities are allowed.
        Fewer configurations than in the previous case are allowed since :math:`|y_{\star}|` is more constraining than :math:`|\sin(i_{\star})|`.
              
          1. :math:`(i_\mathrm{p},\lambda,i_{\star}) = (i_\mathrm{p,in}   , \lambda_\mathrm{in}   , i_{\star \mathrm{in}})`
         
                  conserves :math:`x_{\star \mathrm{sky}} \sin(i_{\star})` and :math:`y_{\star}^2`.
                 
                  yields :math:`\psi = \psi_\mathrm{in}`. 
                 
            :math:`(i_\mathrm{p},\lambda,i_{\star}) = (i_\mathrm{p,in}   , \pi+\lambda_\mathrm{in},-i_{\star \mathrm{in}})`
            
                  conserves :math:`-x_{\star \mathrm{sky}} \sin(-i_{\star})` and :math:`y_{\star}^2`.
                 
                  yields :math:`\psi = \psi_\mathrm{in}`. 
                 
                  this configuration is the same as 1., rotating the whole system by :math:`\pi` around the LOS (ie, yielding :math:`x_{\star \mathrm{sky}} = -x_{\star \mathrm{sky}}` and :math:`y_{\star \mathrm{sky}} = -y_{\star \mathrm{sky}}` in the sky-projected frame).
                 
          2. :math:`(i_\mathrm{p},\lambda,i_{\star}) = (\pi-i_\mathrm{p,in},-\lambda_\mathrm{in}   , \pi-i_{\star \mathrm{in}})` 
         
                  conserves :math:`x_{\star \mathrm{sky}} \sin(i_{\star})` and :math:`(-y_{\star})^2`.
                 
                  yields :math:`\psi = \psi_\mathrm{in}`
                 
            :math:`(i_\mathrm{p},\lambda,i_{\star}) = (\pi-i_\mathrm{p,in},\pi-\lambda_\mathrm{in} , \pi+i_{\star \mathrm{in}})`   
            
                  conserves :math:`-x_{\star \mathrm{sky}} (-\sin(i_{\star}))` and :math:`(-y_{\star})^2`.
                 
                  yields :math:`\psi = \psi_\mathrm{in}`.
                 
                  this configuration is the same as 2., rotating the whole system by :math:`\pi` around the LOS (ie, yielding :math:`x_{\star \mathrm{sky}} = -x_{\star \mathrm{sky}}` and :math:`y_{\star \mathrm{sky}} = y_{\star \mathrm{sky}}` in the sky-projected frame) 
                 
          3. :math:`(i_\mathrm{p},\lambda,i_{\star}) = (-i_\mathrm{p,in}  , \lambda_\mathrm{in}   , \pi-i_{\star \mathrm{in}})`  
         
                  conserves :math:`x_{\star \mathrm{sky}} \sin(i_{\star})` and :math:`y_{\star}^2`.
                 
                  yields :math:`\psi = \pi - \psi_\mathrm{in}`.
                 
            :math:`(i_\mathrm{p},\lambda,i_{\star}) = (-i_\mathrm{p,in}  , \pi+\lambda_\mathrm{in}, \pi+i_{\star \mathrm{in}})`     
            
                  conserves :math:`-x_{\star \mathrm{sky}} (-\sin(i_{\star}))` and :math:`y_{\star}^2`.
                 
                  this configuration is the same as 3., rotating the whole system by :math:`\pi` around the LOS (ie, yielding :math:`x_{\star \mathrm{sky}} = -x_{\star \mathrm{sky}}` and :math:`y_{\star \mathrm{sky}} = -y_{\star \mathrm{sky}}` in the sky-projected frame).
                 
                  yields :math:`\psi = \pi - \psi_\mathrm{in}`.
                 
            but this configuration is not authorized for a transiting planet (it would yield the transit behind the star).
            Note that if several planets are constrained, only combinations of 1. together, or 2. together, are possible since they must share the same :math:`i_{\star}`    
    
    Args:
        None
    
    Returns:
        all_system_params (dict) : dictionary with planetary system properties
    
    """        
    all_system_params={
    
        'Arda':{
            'star':{
                'Rstar':0.9,
                'istar':90.,
                'veq':15., 
                },  
     
            'Valinor':{
                'period':3.8,
                'TCenter':2457176.1,
                'ecc':0.,     
                'omega_deg':90.,          
                'Kstar':38.1,
            },
    
            'Numenor':{
                'period':8.,
                'TCenter':2457176.2,
                'ecc':0.,
                'omega_deg':90.,
                'Kstar':20.,
            },           
        },
    }

    return all_system_params




