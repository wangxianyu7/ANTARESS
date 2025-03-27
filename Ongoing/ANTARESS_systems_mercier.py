#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def get_system_params():
    r"""**Planetary system properties.**    

    Returns dictionary with properties of planetary systems. Format with minimum required properties is 
    
    >>> all_system_params={
        'star_name':{
            'star':{
                'sysvel': val,
                'Rstar':  val,
                'istar':  val,
                'veq':    val 
                },      
            'planet_b':{
                'period':      val,
                'TCenter':     val,
                'ecc':         val,     
                'omega_deg':   val,
                'inclination': val,     
                'lambda_proj': val,     
                'aRs':         val,     
                'Kstar':       val  
            },           
        },
    }
      
    Additional planets can be associated with a given system. Required and optional properties are defined as follows.
    
    **Star properties**
    
     - `sysvel` [km/s] : systemic radial velocity
     
         + relative velocity between the stellar and solar system barycenters 
         + this value is used for first-order definitions and does not need to be extremely accurate
    
     - `veq` [km/s] : equatorial rotational velocity 
     
     - `veq_spots` [km/s] : equatorial rotational velocity of active regions categorized as spots (contrast < 1)
     
     - `veq_faculae` [km/s] : equatorial rotational velocity of active regions categorized as faculae (contrast >=1)
     
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

      - `alpha_rot_spots`, `beta_rot_spots` [] : differential rotation coefficients of active regions categorized as spots (contrast < 1) 

      - `alpha_rot_faculae`, `beta_rot_faculae` [] : differential rotation coefficients of active regions categorized as faculae (contrast < 1)
         
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

      - `Tcenter` [bjd] : reference time for null stellar phase
      
          + set to 2400000 if undefined
        
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

        These degenerate configurations thus yields two possible values for :math:`\psi`, equivalent to maintaining the same sky-projected angle and orbital inclination, and considering a 
        `Northern` configuration (:math:`i_{\star}=i_{\star \mathrm{in}}` ; :math:`\Psi=\arccos(\sin(i_{\star \mathrm{in}}) \cos(\lambda_\mathrm{in}) \sin(i_\mathrm{p,in}) + \cos(i_{\star \mathrm{in}}) \cos(i_\mathrm{p,in}))`) 
        and a
        `Southern` configuration (:math:`i_{\star}=\pi-i_{\star \mathrm{in}}` ; :math:`\Psi=\arccos(\sin(i_{\star \mathrm{in}}) \cos(\lambda_\mathrm{in}) \sin(i_\mathrm{p,in}) - \cos(i_{\star \mathrm{in}}) \cos(i_\mathrm{p,in}))`)
        which are equiprobable.                

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
                'inclination':90.,
                'lambda_proj':0.,
                'aRs':0.05,
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

        #------------------------------    

        'V1298tau':{ 
            'star':{   #Provided by A. Sozetti and G. Guilluy
                'Rstar':1.345,     #David2019ApJL) 
                'istar':90.,
                'Mstar':1.101,
                'veq':23.59, #-0.26 +0.31                           fitted from prior 24 +- 1.  (Calculated from LD fit with free LD coefficient (i), see LD)
                'veq_spots':23.59,
                },

            'V1298tau_b':{   #Provided by A. Sozetti and G. Guilluy
                'period':24.1396,           #David priv. comm.
                'TCenter':2457067.0488,      #Set to visit-specific values
                'ecc':0.,
                'omega_deg':92.,
                'Kstar':40.5,
                'aRs':27.0,  #(Ggrav/(4*!dpi^2.))^(1./3.)*(Mstar)^(1./3.)*((Per)^(2./3.)/Rstar) = (2942.71377/(4*np.pi**2.))**(1./3.)*(1.101)**(1./3.)*(24.1414410**(2./3.)/1.345)    ;From Sozzetti et al. 2007
                'inclination':89.00,   # -0.24+0.46                         David2019b
                'TLength':6.42/24.,      #Feinstein2021
                'lambda_proj':4.
            }, 

        },    

        #------------------------------      

        'HD189733':{
            
            'star':{  
                # 'dstar':19.45,
                # 'Mstar':0.823,
                # 'sysvel':-2.2765,
                
                # #Limb-darkening
                # 'LD_mod':'linear',
                # 'LD_u1':0.816
    
                #Analysis for Mounzer+2023, from Krenn+2023            
                'Rstar':0.784,  #+- 0.007
                      
                #SB
                # 'istar':90.,        
                # 'veq':3.1461070814, 
                
                #DR+CB FINAL
                'alpha_rot':3.1975297175e-01,
                'istar':9.1243662412e+01,         
                'veq':3.6489555591e+00,    
                'c1_CB':-1.7377394883e-01,
                                                                     
                },
        
            'HD189733b':{
                #Old
                # 'Msini':1.135,
                # 'TDepth':0.024122,
                # 'Rpl':1.138,
                # 'sma':0.0312,
    
                #Analysis for Mounzer+2023, from Krenn+2023             
                'period':2.2185751979,
                'TCenter':2459446.49851,
                'TLength':1.81/24. ,# 1.9070525639199452/24.,
                'ecc':0.,
                'omega_deg':90.,
                'inclination':85.70539050086485,   #0.016060250342902198 deg   (b = 0.665)
                'Kstar':201.3,  
                'aRs':8.8843,
              
                # 'lambda_proj':-6.6062684702e-01 #SB
                'lambda_proj':-8.0873395849e-01       #DR+CB FINAL  
                
                
                },
    
        },

        #------------------------------    

        'TOI3884':{ 
            # 'star':{  #Libby-Roberts+2023 
            #     'Rstar':0.302,              
            #     'istar':155.,                    #unknown
            #     'Mstar':0.298,
            #     'veq':8.495,                     #unknown 
            #     'logg':4.97,   
            #     'Tpole':3180,
            #     'vmic':0.,            
            #     # 'veq_spots':23.59,
            #     },

            'star':{  #Equatorial band hypothesis - updated
                'Rstar':0.302,              
                'istar':90.,                    #unknown
                'Mstar':0.298,
                'veq':8.495,                     #unknown                
                # 'veq_spots':23.59,
                },

            # 'star':{  #Equatorial band hypothesis
            #     'Rstar':0.302,              
            #     'istar':89.,                    #unknown
            #     'Mstar':0.298,
            #     'veq':1.504,                     #unknown                
            #     # 'veq_spots':23.59,
            #     },

            # 'star':{  #Almenara+2022
            #     'Rstar':0.3043,              
            #     'istar':133.,                    #unknown
            #     'Mstar':0.2813,
            #     'veq':1.504,                     #unknown                
            #     # 'veq_spots':23.59,
            #     },

            # 'TOI3884_b':{  #Libby-Roberts+2023  
            #     'period':4.5445828,           
            #     'TCenter':2459556.51669,      
            #     'ecc':0.060,
            #     'omega_deg':-112., #very different from previous studies - re-fit?
            #     'Kstar':28.03,     #very different from previous studies
            #     'aRs':25.90,  
            #     'inclination':89.81,   
            #     'TLength':1.60/24.,      
            #     'lambda_proj':75.  #what we want to constrain so will vary
            #     }, 

            # 'TOI3884_b':{  #Equatorial band hypothesis 
            #     'period':4.5445828,           
            #     'TCenter':2459556.51669,      
            #     'ecc':0.060,
            #     'omega_deg':-112., #very different from previous studies - re-fit?
            #     'Kstar':28.03,     #very different from previous studies
            #     'aRs':25.90,  
            #     'inclination':90.4,   
            #     'TLength':1.60/24.,      
            #     'lambda_proj':30.  #what we want to constrain so will vary
            #     },

            'TOI3884_b':{  #Equatorial band hypothesis updated
                'period':4.5445697,           
                'TCenter':2459556.51669,      
                'ecc':0.059,        #Unknown
                'omega_deg':190., #very different from previous studies - re-fit?
                'Kstar':14.9,     #very different from previous studies
                'aRs':25.01,  
                'inclination':90.40,   
                'TLength':1.646/24.,      
                'lambda_proj':30.  #what we want to constrain so will vary
                }, 

            # 'TOI3884_b':{  #Almenara+2022
            #     'period':4.5445697,           
            #     'TCenter':2459642.86314,      
            #     'ecc':0.059,        #Unknown
            #     'omega_deg':190., #very different from previous studies - re-fit?
            #     'Kstar':14.9,     #very different from previous studies
            #     'aRs':25.01,  
            #     'inclination':90.10,   
            #     'TLength':1.646/24.,      
            #     'lambda_proj':50.  #what we want to constrain so will vary
            # }, 
        },      

        #------------------------------    

        'TRAPPIST1':{

            'star':{
                'Rstar':0.1192,               #solar radii #Agol+2021
                'Mstar':0.0898,               #solar mass #Agol+2021
                'logg':5.2396,                #Agol+2021
                'veq':2.04,                   #km/s #Agol+2021
                'alpha_rot':0.0,
                'istar':89,                   
                },

            'TRAPPIST1_b':{
                'period':1.510826,           #days #Agol+2021
                'TCenter':2457322.514193,    #days #Ducrot+2020
                'ecc':0.00622,               #Grimm+2018
                'omega_deg':336.86,          #degrees #Grimm+2018
                'inclination':89.728,        #degrees #Agol+2021
                'Kstar':0.150722,            #m/s #Calculated
                'TLength':0.6010/24,         #days #Agol+2021
                'aRs':20.843,                #Rstar #Agol+2021
                'lambda_proj':0.,            #degrees #Hirano+2020
            },

            'TRAPPIST1_c':{
                'period':2.421937,           #days #Agol+2021
                'TCenter':2457282.8113871,   #days #Ducrot+2020
                'ecc':0.00654,               #Grimm+2018
                'omega_deg':282.45,          #degrees #Grimm+2018
                'inclination':89.778,        #degrees #Agol+2021
                'Kstar':0.121249,            #m/s #Calculated
                'TLength':0.7005/24,         #days #Agol+2021
                'aRs':28.549,                #Rstar #Agol+2021
                'lambda_proj':0.,            #Undefined
            },

            'TRAPPIST1_d':{
                'period':4.049219,           #days #Agol+2021
                'TCenter':2457670.1463014,   #days #Ducrot+2020
                'ecc':0.00837,               #Grimm+2018
                'omega_deg':-8.73,           #degrees #Grimm+2018
                'inclination':89.896,        #degrees #Agol+2021
                'Kstar':0.0292108,           #m/s #Calculated
                'TLength':0.8145/24,         #days #Agol+2021
                'aRs':40.216,                #Rstar #Agol+2021 
                'lambda_proj':0.,            #Undefined
            },

            'TRAPPIST1_e':{
                'period':6.101013,           #days #Agol+2021
                'TCenter':2457660.3676621,   #days #Ducrot+2020
                'ecc':0.00510,               #Grimm+2018
                'omega_deg':108.37,          #degrees #Grimm+2018
                'inclination':89.793,        #degrees #Agol+2021
                'Kstar':0.0469638,           #m/s #Calculated
                'TLength':0.9293/24,         #days #Agol+2021
                'aRs':52.855,                #Rstar #Agol+2021
                'lambda_proj':0.,           #degrees #Hirano+2020
            },

            'TRAPPIST1_f':{
                'period':9.207540,           #days #Agol+2021
                'TCenter':2457671.3737299,   #days #Ducrot+2020
                'ecc':0.01007,               #Grimm+2018
                'omega_deg':368.81,          #degrees #Grimm+2018
                'inclination':89.740,        #degrees #Agol+2021
                'Kstar':0.0622481,           #m/s #Calculated
                'TLength':1.0480/24,         #days #Agol+2021
                'aRs':69.543,                #Rstar #Agol+2021
                'lambda_proj':0.,            #degrees #Hirano+2020
            },

            'TRAPPIST1_g':{
                'period':12.352446,          #days #Agol+2021
                'TCenter':2457665.3628439,   #days #Ducrot+2020
                'ecc':0.00208,               #Grimm+2018
                'omega_deg':191.34,          #degrees #Grimm+2018
                'inclination':89.742,        #degrees #Agol+20201
                'Kstar':0.0717257,           #m/s #Calculated
                'TLength':1.1370/24,         #days #Agol+2021
                'aRs':84.591,                #Rstar #Agol+2021
                'lambda_proj':0.,            #Undefined
            },

            'TRAPPIST1_h':{
                'period':18.772866,          #days #Agol+2021
                'TCenter':2457662.5741486,   #days #Ducrot+2020
                'ecc':0.00567,               #Grimm+2018
                'omega_deg':338.92,          #degrees #Grimm+2018
                'inclination':89.805,        #degrees #Agol+2021
                'Kstar':0.0151621,           #m/s #Calculated
                'TLength':1.2690/24,         #days #Agol+2021
                'aRs':111.817,               #Rstar #Agol+2021
                'lambda_proj':0.,            #Undefined
            }
        },

        #------------------------------    

        'AUMic':{

            'star':{
                'Rstar':0.75,                #+/- 0.03 #solar radii #Plavchan et al. 2020
                'Mstar':0.50,                #+/- 0.03 #solar mass #Plavchan et al. 2020
                'logg':4.39,                 #+/- 0.03 #computed from Plavchan et al. 2020 Rstar and Mstar values by Zicher et al. 2022
                'veq':7.8,                   #+/- 0.3 #km/s #Klein et al. 2021
                'istar':89,                  #unknown
                # 'mag':8.81,                  #+/- 0.10 #Johnson V magnitude (true value)
                # 'f_GD':0.1947,              #test
                # 'Tpole':8450.,              #test
                # 'beta_GD':0.190,            #test
                # 'veq_spots':7.8,
                },

            'AUMicb':{
                'period':8.463000,           #+/- 0.000002 #days #Martioli et al. 2021 
                'TCenter':2458330.39051,     #+/- 0.00015 #days #Martioli et al. 2021
                'ecc':0.04,                  #+0.045 -0.025 #Zicher et al. 2022
                'omega_deg':179,             #+128 -125 #degrees #Zicher et al. 2022
                'inclination':89.18,         #+0.53 - 0.45 #degrees #Gilbert et al. 2022
                'Kstar':5.8,                 #+/- 2.5 #m/s #Zicher et al. 2022 
                'TLength':3.55/24,           #+/- 0.08 #days #Martioli et al. 2021
                'aRs':18.5,                  #+1.3 - 1.4 #Rstar #Gilbert et al. 2022 
                'lambda_proj':-4.70,         #+6.80 -6.40 #degrees #Hirano et al. 2020
            },

            # 'AUMicc':{ -- for plotting puposes -- 
            #    'period':18.859019,          #+/- 0.000016 #days #Martioli et al. 2021
            #    'TCenter':2458330.29051,     #+/- 0.00050 #days #Martioli et al. 2021
            #    'ecc':0.041,                 #+0.047 -0.026 #Zicher et al. 2022
            #    'omega_deg':153,             #+124 -94 #degrees #Zicher et al. 2022
            #    'inclination':92,        #+0.40 -0.28 #degrees #Gilbert et al. 2022
            #    'Kstar':8.5,                 #+/- 2.5 #m/s #Zhicert et al. 2022
            #    'TLength':4.5/24,            #+/- 0.8 #days #Martioli et al. 2021
            #    'aRs':31.7,                  #+2.6 -2.7 #Rstsr #Gilbert et al. 2022
            #    'lambda_proj':0.,            #unknown

            # },
        },
    
    #------------------------------    
        # Wittrock+2023 values except for veq and Rstar, which is from Donati+2023 

        'AU_Mic':{

            'star':{
                'Rstar':0.82,                 #+/-0.02 #solar radii
                'Mstar':0.510,                #+0.028 -0.027 #solar mass
                'logg':4.404,                 #+0.026 -0.031
                'veq':8.5,                    #+/- 0.3 #km/s <- will be fitted
                'alpha_rot':0.034,
                'istar':89,                   #unknown <- will be fitted
                },

            'AU_Mic_b':{
                'period':8.46308,            #+/- 0.00006 #days
                'TCenter':2458330.39080,     #+0.00052 -0.00051 #days
                'ecc':0.00577,               #+/- 0.00101
                'omega_deg':88.43038,        #+/-0.05783 #degrees
                'inclination':89.57917,      #+/- 0.37639 #degrees
                'Kstar':0.31290,             #+/- 0.26983 #m/s
                'TLength':3.4927/24,         #+0.0074 -0.0067 #days
                'aRs':18.79,                 #+0.5 -0.59 #Rstar 
                'lambda_proj':-4.70,         #+6.80 -6.40 #degrees <- Hirano+2020 value, but will be fitted
            },
        },

    #------------------------------    
        # Wittrock+2023 values except for veq, which is from Donati+2023 

        'fakeAU_Mic':{

            'star':{
                'Rstar':0.82,                 #+/-0.02 #solar radii
                'Mstar':0.510,                #+0.028 -0.027 #solar mass
                'logg':4.404,                 #+0.026 -0.031
                'veq':8.595,                    #+/- 0.3 #km/s <- will be fitted
                'alpha_rot':0.034,
                'istar':89,                   #unknown <- will be fitted
                },

            'fakeAU_Mic_b':{
                'period':8.46308,            #+/- 0.00006 #days
                'TCenter':2458330.39080,     #+0.00052 -0.00051 #days
                'ecc':0.00577,               #+/- 0.00101
                'omega_deg':88.43038,        #+/-0.05783 #degrees
                'inclination':89.57917,      #+/- 0.37639 #degrees
                'Kstar':0.31290,             #+/- 0.26983 #m/s
                'TLength':3.4927/24,         #+0.0074 -0.0067 #days
                'aRs':18.79,                 #+0.5 -0.59 #Rstar 
                'lambda_proj':-4.70,         #+6.80 -6.40 #degrees <- Hirano+2020 value, but will be fitted
            },
        },


    #--------------------------------------------------------------------------------
    #------------------------------------  GEMS ------------------------------------- 
    #--------------------------------------------------------------------------------
      'TOI5205':{ #Kanodia+2023
        'star':{   
            'Rstar':0.394,     
            'istar':90., #unknown
            'Mstar':0.392,
            'veq':2.0,   #unknown - used vsini
            'veq_spots':2.0,
        },

        'TOI5205_b':{ #Kanodia+2023
            'period':1.630757,
            'TCenter':2459443.47179,
            'ecc':0.02,
            'omega_deg':-42.,
            'Kstar':346,
            'aRs':10.94,
            'inclination':88.21,
            'TLength':1.4/24.,
            'lambda_proj':0. #unknown
        }, 

      },    
    #------------------------------ 
      'TOI3714':{ #Hartman+2023
        'star':{   
            'Rstar':0.4958,     
            'istar':90., #unknown
            'Mstar':0.522,
            'veq':1.08,#Canas+2022
            'veq_spots':1.08,#Canas+2022
        },

        'TOI3714_b':{
            'period':2.15484802,
            'TCenter':2459687.365240,
            'ecc':0.03, #Canas+2022
            'omega_deg':100., #Canas+2022
            'Kstar':167.1,
            'aRs':11.391,
            'inclination':87.830,
            'TLength':1.626/24.,
            'lambda_proj':0. #unknown
        }, 

      },    
    #------------------------------
      'TOI5293':{ #Canas+2023
        'star':{   
            'Rstar':0.52,     
            'istar':90., #unknown
            'Mstar':0.54,
            'veq':1.28,   #derived from Peq = 20.6d
            'veq_spots':1.28,
        },

        'TOI5293_b':{ #Canas+2023
            'period':2.930289,
            'TCenter':2459448.9148,
            'ecc':0.38, #upper limit
            'omega_deg':-92.,
            'Kstar':115.6,
            'aRs':14.1,
            'inclination':88.8,
            'TLength':1.94/24.,
            'lambda_proj':65. #unknown
        }, 

      },    
    #------------------------------
      'TOI3757':{ #Kanodia+2022
        'star':{   
            'Rstar':0.62,     
            'istar':90., #unknown
            'Mstar':0.64,
            'veq':2.0,   #unknown
            'veq_spots':2.0, 
        },

        'TOI3757_b':{ #Kanodia+2022
            'period':3.438753,
            'TCenter':2458838.77148,
            'ecc':0.14,
            'omega_deg':130.,
            'Kstar':49.24,
            'aRs':13.26,
            'inclination':86.76,
            'TLength':1.92/24.,
            'lambda_proj':0. #unknown
        }, 

      },    
    #------------------------------
      'TOI3984':{ #Canas+2023
        'star':{   
            'Rstar':0.47,     
            'istar':90., #unknown
            'Mstar':0.49,
            'veq':0.476,   #derived from Peq = 50d
            'veq_spots':0.476,
        },

        'TOI3984_b':{ #Canas+2023
            'period':4.353326,
            'TCenter':2459715.02268,
            'ecc':0.23, #upper limit
            'omega_deg':-54.,
            'Kstar':27.7,
            'aRs':19.1,
            'inclination':89.5,
            'TLength':2.01/24.,
            'lambda_proj':0. #unknown
        }, 

      },    
    #------------------------------
      'HATS75':{ #Jordan+2022
        'star':{   
            'Rstar':0.5848,     
            'istar':90., #unknown
            'Mstar':0.6017,
            'veq':0.85,
            'veq_spots':0.85,
        },

        'HATS75_b':{ #Jordan+2022
            'period':2.7886556,
            'TCenter':2458611.05487,
            'ecc':0.064, #upper limit
            'omega_deg':90., #unknown
            'Kstar':99.2,
            'aRs':12.037,
            'inclination':88.07,
            'TLength':1.91256/24.,
            'lambda_proj':0. #unknown
        }, 

      },    
    #------------------------------
      'HATS6':{ #Hartman+2015
        'star':{   
            'Rstar':0.570,     
            'istar':90., #unknown
            'Mstar':0.574,
            'veq':2.0, #unknown
            'veq_spots':2.0, #unknown
        },

        'HATS6_b':{ #Hartman+2015
            'period':3.3252725,
            'TCenter':2456643.740580,
            'ecc':0.,
            'omega_deg':90.,
            'Kstar':63,
            'aRs':13.65,
            'inclination':88.210,
            'TLength':2.041/24.,
            'lambda_proj':0. #unknown
        }, 

      },    

    #------------------------------
      'TOI1224':{ #Thao+2024
        'star':{   
            'Rstar':0.404,     
            'istar':90., #unknown
            'Mstar':0.400,
            'veq':22.1, #unknown
            'veq_spots':22.1, #unknown
        },

        'TOI1224_b':{ #Thao+2024
            'period':4.1782745,
            'TCenter':2458327.70236,
            'ecc':0., #unknown
            'omega_deg':90., #unknown
            # 'Kstar':, #unknown
            'aRs':18.90,
            'inclination':89.19,
            'TLength':1.71/24.,
            'lambda_proj':0. #unknown
        }, 

      },    

    #------------------------------
      'TOI540':{ #Ment+2021
        'star':{   
            'Rstar':0.1895,     
            'istar':90., #unknown
            'Mstar':0.159,
            'veq':13.5, #unknown
            'veq_spots':13.5, #unknown
        },

        'TOI540_b':{ #Ment+2021
            'period':1.2391491,
            'TCenter':2458411.82601,
            'ecc':0.,
            'omega_deg':90.,
            # 'Kstar':, #unknown
            'aRs':13.90,
            'inclination':86.80,
            'TLength':0.48/24.,
            'lambda_proj':0. #unknown
        }, 

      },    

    #------------------------------
      'TOI122':{ #Waalkes+2021
        'star':{   
            'Rstar':0.334,     
            'istar':90., #unknown
            'Mstar':0.312,
            'veq':7.2, #unknown
            'veq_spots':7.2, #unknown
        },

        'TOI122_b':{ #Waalkes+2021
            'period':5.078030,
            'TCenter':2458425.602564,
            'ecc':0.,
            'omega_deg':90.,
            # 'Kstar':, #unknown
            'aRs':25.2,
            'inclination':88.4,
            'TLength':1.23/24.,
            'lambda_proj':0. #unknown
        }, 

      }, 

    #------------------------------
      'K233':{ #Mann+2016
        'star':{   
            'Rstar':1.05,     
            'istar':90., #unknown
            'Mstar':0.56,
            'veq':8.2, #unknown
            'veq_spots':8.2, #unknown
        },

        'K233_b':{ #Mann+2016
            'period':5.424865,
            'TCenter':2456898.69288,
            'ecc':0.,
            'omega_deg':90.,
            # 'Kstar':, #unknown
            'aRs':10.40,
            'inclination':89.1,
            'TLength':4.08/24.,
            'lambda_proj':0. #unknown
        }, 

      },  

    #------------------------------
      'TOI1227':{ #Mann+2022
        'star':{   
            'Rstar':0.56,     
            'istar':90., #unknown
            'Mstar':0.170,
            'veq':16.65, #unknown
            'veq_spots':16.65, #unknown
        },

        'TOI1227_b':{ #Mann+2022
            'period':27.36397,
            'TCenter':2458617.4621,
            'ecc':0., #unknown
            'omega_deg':90., #unknown
            # 'Kstar':, #unknown
            'aRs':34.01,
            'inclination':88.571,
            'TLength':4.831/24.,
            'lambda_proj':0. #unknown
        }, 

      },     

    #--------------------------------------------------------------------------------
    #------------------------------  MCMC run purposes ------------------------------ 
    #--------------------------------------------------------------------------------
        #Fast rotating star
        #   + veq changes
        'Capricorn':{
            'star':{
                'Rstar':0.75,                
                'Mstar':0.50,                
                'logg':4.39,                 
                'veq':20,                  
                'istar':88,                  
                'mag':8.81,                  
                },

            'Capricorn_b':{
                'period':8.463000,         
                'TCenter':2458330.39051,     
                'ecc':0.04,                  
                'omega_deg':179,             
                'inclination':89.18,         
                'Kstar':5.8,                 
                'TLength':3.55/24,           
                'aRs':18.5,                  
                'lambda_proj':-4.70,         
            },
        },
        #--------------------------------------------------------------------------------
        #Slow rotating star
        #   + veq changes
        'Cancer':{
            'star':{
                'Rstar':0.75,                
                'Mstar':0.50,                
                'logg':4.39,                 
                'veq':0.2,                   
                'istar':88,                  
                'mag':8.81,                  
                },

            'Cancer_b':{
                'period':8.463000,         
                'TCenter':2458330.39051,     
                'ecc':0.04,                  
                'omega_deg':179,             
                'inclination':89.18,         
                'Kstar':5.8,                 
                'TLength':3.55/24,           
                'aRs':18.5,                  
                'lambda_proj':-4.70,         
            },
        },
        #--------------------------------------------------------------------------------
        #Big transiting planet - Jupiter size and 1/2 mass
        #   + RpRstar changes -> 0.1339
        #   + TLength changes -> 3.85 hours
        #   + Kstar changes -> 148.3
        'Gemini':{
            'star':{
                'Rstar':0.75,                
                'Mstar':0.50,                
                'logg':4.39,                 
                'veq':7.8,                   
                'istar':88,                  
                'mag':8.81,                  
                },

            'Gemini_b':{
                'period':8.463000,         
                'TCenter':2458330.39051,     
                'ecc':0.04,                  
                'omega_deg':179,             
                'inclination':89.18,         
                'Kstar':148.3,                 
                'TLength':3.85/24,           
                'aRs':18.5,                  
                'lambda_proj':-4.70,         
            },
        },
        #--------------------------------------------------------------------------------
        #Small transiting panet - super-Earth size and mass
        #   + RpRstar changes -> 0.0122
        #   + TLength changes -> 3.41 hours
        #   + Kstar changes -> 0.934
        'Sagittarius':{
            'star':{
                'Rstar':0.75,                
                'Mstar':0.50,                
                'logg':4.39,                 
                'veq':7.8,
                'veq_spots':8.1,                   
                'istar':88,                  
                'mag':8.81,                  
                },

            'Sagittarius_b':{
                'period':8.463000,         
                'TCenter':2458330.39051,     
                'ecc':0.04,                  
                'omega_deg':179,             
                'inclination':89.18,         
                'Kstar':0.93,                 
                'TLength':3.41/24,           
                'aRs':18.5,                  
                'lambda_proj':-4.70,         
            },
        },
        #--------------------------------------------------------------------------------
        #Big spot
        'Leo':{
            'star':{
                'Rstar':0.75,           
                'Mstar':0.50,                
                'logg':4.39,                 
                'veq':7.8,                   
                'istar':88,                  
                'mag':8.81,                  
                },

            'Leo_b':{
                'period':8.463000,         
                'TCenter':2458330.39051,     
                'ecc':0.04,                  
                'omega_deg':179,             
                'inclination':89.18,         
                'Kstar':5.8,                 
                'TLength':3.55/24,           
                'aRs':18.5,                  
                'lambda_proj':-4.70,         
            },
        },
        #--------------------------------------------------------------------------------
        #Small spot
        'Aquarius':{
            'star':{
                'Rstar':0.75,                
                'Mstar':0.50,                
                'logg':4.39,                 
                'veq':7.8,                   
                'istar':88,                  
                'mag':8.81,                  
                },

            'Aquarius_b':{
                'period':8.463000,         
                'TCenter':2458330.39051,     
                'ecc':0.04,                  
                'omega_deg':179,             
                'inclination':89.18,         
                'Kstar':5.8,                 
                'TLength':3.55/24,           
                'aRs':18.5,                  
                'lambda_proj':-4.70,         
            },
        },
        #--------------------------------------------------------------------------------
        #Dark spot
        'Aries':{
            'star':{
                'Rstar':0.75,                
                'Mstar':0.50,                
                'logg':4.39,                 
                'veq':7.8,                   
                'istar':88,                  
                'mag':8.81,                  
                },

            'Aries_b':{
                'period':8.463000,         
                'TCenter':2458330.39051,     
                'ecc':0.04,                  
                'omega_deg':179,             
                'inclination':89.18,         
                'Kstar':5.8,                 
                'TLength':3.55/24,           
                'aRs':18.5,                  
                'lambda_proj':-4.70,         
            },
        },
        #--------------------------------------------------------------------------------
        #Bright spot
        'Libra':{
            'star':{
                'Rstar':0.75,                
                'Mstar':0.50,                
                'logg':4.39,                 
                'veq':7.8,                   
                'istar':88,                  
                'mag':8.81,                  
                },

            'Libra_b':{
                'period':8.463000,         
                'TCenter':2458330.39051,     
                'ecc':0.04,                  
                'omega_deg':179,             
                'inclination':89.18,         
                'Kstar':5.8,                 
                'TLength':3.55/24,           
                'aRs':18.5,                  
                'lambda_proj':-4.70,         
            },
        },
        #--------------------------------------------------------------------------------
        #Low stellar inclination
        #   + istar changes
        'Taurus':{
            'star':{
                'Rstar':0.75,                
                'Mstar':0.50,                
                'logg':4.39,                 
                'veq':7.8,                   
                'istar':10,                  
                'mag':8.81,                  
                },

            'Taurus_b':{
                'period':8.463000,         
                'TCenter':2458330.39051,     
                'ecc':0.04,                  
                'omega_deg':179,             
                'inclination':89.18,         
                'Kstar':5.8,                 
                'TLength':3.55/24,           
                'aRs':18.5,                  
                'lambda_proj':-4.70,         
            },
        },
        #--------------------------------------------------------------------------------
        #High stellar inclination
        #   + istar changes
        'Scorpio':{
            'star':{
                'Rstar':0.75,                
                'Mstar':0.50,                
                'logg':4.39,                 
                'veq':7.8,                   
                'istar':110,                  
                'mag':8.81,                  
                },

            'Scorpio_b':{
                'period':8.463000,         
                'TCenter':2458330.39051,     
                'ecc':0.04,                  
                'omega_deg':179,             
                'inclination':89.18,         
                'Kstar':5.8,                 
                'TLength':3.55/24,           
                'aRs':18.5,                  
                'lambda_proj':-4.70,         
            },
        },
        #--------------------------------------------------------------------------------
        #Low obliquity
        #   + lambda_proj changes
        'Virgo':{
            'star':{
                'Rstar':0.75,                
                'Mstar':0.50,                
                'logg':4.39,                 
                'veq':7.8,                   
                'istar':88,                  
                'mag':8.81,                  
                },

            'Virgo_b':{
                'period':8.463000,         
                'TCenter':2458330.39051,     
                'ecc':0.04,                  
                'omega_deg':179,             
                'inclination':89.18,         
                'Kstar':5.8,                 
                'TLength':3.55/24,           
                'aRs':18.5,                  
                'lambda_proj':10,         
            },
        },
        #--------------------------------------------------------------------------------
        #High obliquity
        #   + lambda_proj changes
        'Pisces':{
            'star':{
                'Rstar':0.75,                
                'Mstar':0.50,                
                'logg':4.39,                 
                'veq':7.8,                   
                'istar':88,                  
                'mag':8.81,                  
                },

            'Pisces_b':{
                'period':8.463000,         
                'TCenter':2458330.39051,     
                'ecc':0.04,                  
                'omega_deg':179,             
                'inclination':89.18,         
                'Kstar':5.8,                 
                'TLength':3.55/24,           
                'aRs':18.5,                  
                'lambda_proj':110,         
            },
        },
    } ##end of systems

    return all_system_params




