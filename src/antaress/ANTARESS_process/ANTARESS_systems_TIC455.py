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

      - `alpha_rot_faculae`, `beta_rot_faculae` [] : differential rotation coefficients of active regions categorized as faculae (contrast >= 1)
            
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
        'Star_tseries':{
            'star':{
                'sysvel':0.
                }
            },

    #------------------------------
    ##############################################################
    # Atreides
    #------------------------------


    'TOI421':{
        
        'star':{
            'Rstar':0.866,                          #± 0.006, Krenn+2024
            'Mstar':0.833,                          #+0.048−0.054, Krenn+2024
            # 'veq'  :1.8,                          # vsini + 1.8, Carleo+2020
            'veq'  :1.16319389,
            'mag'  :9.931,                          # 
            'istar':90.,                             # unknown
            #'logg' :4.486,                         # Testing for theoretical atmosphere
            'Tpole':5291.,                          #± 64    Krenn+2024
            #'vmic':0.9,
            #'vmac':2.5
            }, 

        'TOI421b':{
            'period':5.197576,                  #± 0.000005, Krenn+2024 
            'TCenter':2459189.7341,             #± 0.0005,   Krenn+2024 
            'TLength':0.0454,                   #± 0.0014,   Krenn+2024 
            'ecc':0.13,                         #± 0.05,     Krenn+2024 
            'omega_deg':140,                    #± 30,       Krenn+2024 
            'inclination':85.67222,             #-0.2111+0.2958    shared by A. Krenn
            'Kstar':2.83,                       #± 0.18,     Krenn+2024 
            'aRs':13.75985181,                  #-0.21821+0.234295    shared by A. Krenn 
            'lambda_proj':0.                    #0.180$^{+0.019}_{-0.016}$  
            },

        'TOI421c':{   
            'period':16.067541,                 #± 0.000004, Krenn+2024 
            'TCenter':2459195.30741,            #± 0.00018,  Krenn+2024      
            'TLength':0.1148,                   #± 0.0009,   Krenn+2024       
            'ecc':0.19,                         #± 0.04,     Krenn+2024       
            'omega_deg':102.,                   #± 14,       Krenn+2024      
            'inclination':88.30373,             #-0.05221+0.050453    shared by A. Krenn
            'Kstar':4.1,                        # ± 0.3     Krenn+2024     
            'aRs':29.0546178,                   #-0.40028+0.401457    shared by A. Krenn
            'lambda_proj':16.216242787          # unknown 
            },
    },

    #------------------------------

    'K2_79':{
        
        'star':{
            'Rstar':1.247,                          # Duck+2021 (1.265 +/-0.041/0.027 Bonomo+2023)
            'Mstar':1.066,                          # +/-118 Bonomo+2023
            'logg':4.2490,                          # +/-0.077/0.072 Duck+2021
            'veq':2.7,                              # vsini +/- 0.5 Nava+ 2022
            'mag':12.07,                            # Zacharias+ 2012
            'istar':90.,                            # unknown
            },

        'K2_79b':{                                  # Rp/R* = 0.02948
                'period':10.99386423,               # (10.99386423 from excel), (10.99470±0.00047 Bonomo+2023)
                'TCenter':2460214.7753992,          # (2457103.22750±0.00084 Bonomo+2023)
                'TLength':4.4856/24.,               # (4.4856 from Excel) (4.420±0.088 Crossfield+2016)
                'ecc':0.23,                         # <0.23 Bonomo+2023
                'omega_deg':90.,                    # -
                'inclination':88.44,                # ±0.44 Bonomo+2023
                'Kstar':2.63,                       # ±0.69 Bonomo+2023
                'aRs':13.22+0.36,                   # +0.36-0.38 Duck+2021 (0.0988 au, au = 215.032 Rsun --> 16.797606)
                'lambda_proj':0.                    # unknown
                },
    },

    #------------------------------

    'TOI942':{
        
        'star':{                                                                # other params: limb darkening q1/q2 = 0.264/0.419 (Carleo+2021)
            'mag':11.982,                                                       # 
            'Rstar':0.84,                                                       #
            'Mstar':0.860,                                                      #
            'veq':13.8,                                                         #vsini Excel (13.8 in Carleo+ 2021) Prot=3.39
            'istar':90.,                                                        #unknown sini=1.04 (+/- 0.09/0.10) Carleo+2021
            },

        'TOI942b':{                                                             #20231121, Rp/R* = 0.0425 , 
                'period':4.3243020,                                             #Excel
                'TCenter':2460270.7365,                                         #com.w/ Vincent,             #Excel
                'TLength':3.67/24.,                                             #3.67 -0.10 +0.12 hours(com.w/ Vincent) (Carleo+2021 -> 2.761, now produces a good match)
                'ecc':0.062,                                                    # -0.041 +0.074 (vincent)
                'omega_deg':90.,                                                # "
                'inclination':88.6,                                             # "
                # 'Kstar':0.,                                                     # "
                'Msini':0.062,                                                  # Mp : 19.8 -9.1 +11 M_Earth
                'aRs':11.732,                                                   # 	Carleo et al. 2021
                'lambda_proj':0.                                                # "
                },
        
        'TOI942c':{                                                             #20231226, Rp/R* = 0.048 ,  Mp : 29.1 -12 +22 M_Earth
                'period':10.1557327,                                            #Excel
                'TCenter':2460305.6345,                                         #-0.0014 +0.0013 (com.w/ Vicnent),              #Excel
                'TLength':4.400/24.,                                            #0.400 +/- 0.067 hours (com.w/ Vicnent) (Carleo+2021 3.723)
                'ecc':0.173,                                                    #-0.033 +0.056 (with a lower limit of 0.1, Vincent) ,,,,, (0.32, Wirth+2021), (0, Zhou+2021),  ##(0.175 Caleo+2021)
                'omega_deg':90., #268.,                                         # Zhou+2021 does not provide value, (268 +-58, Calreo+2021), (20+63-33, Wirth+2021)
                'inclination':90.,#89.54,#89.16, #89.2,                         #(89.16 Wirth+2021) (89.54 Zhou+2021),(89.2 Carleo+2021)
                # 'Kstar':0.,                                                   # "
                'Msini':0.091,                                                  # Mp : 29.1 -12 +22 M_Earth
                'aRs':20.728,                                                   # Carleo+2021 (17.88, Zhou+2021)
                'lambda_proj':0.  
            },
    },

    #------------------------------

    'K2_98':{
        
        'star':{                # other params: limb darkening q1/q2 = 0.40/0.26 (Barragan+2016)
            'Rstar':1.311,                           # Barragan+2016
            'veq':6.1,                              # Excel (vsini = 6.1)
            'mag':12.2,                             # Excel
            'istar':0.,                             # unknown
            },

        'K2_98b':{                                  # Rp/R* = 0.0304 (Livingston+2018)
                'period':10.1367349,                # 
                'TCenter':2457662.95321,            # 
                'TLength':5.03/24.,                 # 
                'ecc':0.,                           # unknown
                'omega_deg':0.,                     # "
                'inclination':89.0,                 # Barragan+2016
                'Kstar':9.1,                        # "
                'aRs':15.09,                        # Livingston+2018
                'lambda_proj':0.                    #unknown
                },
    },

    #------------------------------

    'TOI2498':{
        
        'star':{
            'Rstar':1.26,                        # +/- 0.04 (Frame+2023)
            'veq':0.,                            # vsini need to look up
            'mag':11.2,                          # Excel
            'istar':90.,                          # 
            },

        'TOI2498b':{                             # Rp/R* = 0.04 ± 0.01
                'period':3.7382452,              # Excel (Frame+2023)
                'TCenter':2459204.4167,          # Excel (Frame+2023)
                'TLength':3.957/24.,             # Excel (Frame+2023)
                'ecc':0., #.089,                     # Frame+2023
                'omega_deg':90.,#89.99,               # "
                'inclination':90., #87.12,             # "
                'Kstar':13.25,                   # "
                'aRs':8.3794,                    # "
                'lambda_proj':0.                 #unknown
                },
    },
    
    #------------------------------

    'TOI620':{
        
        'star':{
            'Rstar':0.550,                        # +/-0.017 Reefe+2022
            'veq':3.,                            # vsini <3 (Reefe+2022)
            'mag':12.262,                        # Reefe+2022
            'istar':0.,                          # unknown
            },

        'TOI620b':{                                # Rp/R* = 0.0627 Reefe+2022
                'period':5.0987782179,             # 
                'TCenter':2460042.54236282,        #
                'TLength':1.36/24.,                # 
                'ecc':0,#0.22,                        # Reefe+2022
                'omega_deg':-84.,                  # "
                'inclination':87.47,               # "
                'Kstar':8.9,                       # "
                'aRs':18.87,                       # "
                'lambda_proj':0.                   #unknown
                },
    },
    
    #------------------------------

    'K2_271':{
        
        'star':{                # 
            'Rstar':1.2641,                          # Excel
            'veq':0.,                              # Unknown
            'mag':13.845,                             # Excel
            'istar':90.,                             # unknown
            },

        'K2_271b':{                                  # 
                'period':8.562334432,                # Excel
                'TCenter':2457149.7123583,           # Excel
                'TLength':3.24/24.,                  # Excel
                'ecc':0.,                            # Unknown
                'omega_deg':0.,                      # "
                'inclination':90.0,                  # "
                'Kstar':0.,                           # "
                'aRs':20.69,                        # Livingston+2018
                'lambda_proj':0.                    #unknown
                },
    },
    #------------------------------
    
    'K2_10':{
        
        'star':{                                                                # 
            'Rstar':0.98,                                                       # Excel
            'veq':3.,                                                           # Van Eylen+2016
            'mag':12.4,                                                         # Excel F9
            'istar':90.,                                                        # unknown
            },

        'K2_10b':{                                                              # Rp/Rs = 0.037959 ±0.001013/0.000722 , Mayo+2018
                'period':19.3073375,                                            # Excel
                'TCenter':2456819.57784,                                        # Excel
                'TLength':3.8064/24.,                                           # Excel
                'ecc':0.,#31,                                                   # 0.31 from Van Eylen+2016, but large errorbars
                'omega_deg':0.,                                                 # "
                'inclination':89.5,                                             # Mayo+2018
                'Kstar':7.30,                                                   # Van Eylen+2016
                'aRs':38.59,                                                    # Mayo+2018
                'lambda_proj':0.                                                # unknown
                },
    },
    #------------------------------
    
    'K2_108':{
        
        'star':{                                                                # 
            'Rstar':1.76,                                                         # Livingston+2018
            'veq':2.,                                                           # Petigura+2017
            'mag':12.3,                                                         # Excel G6
            'istar':90.,                                                         # unknown
            },

        'K2_108b':{                                                             # 
                'period':4.733496189,                                           # Excel
                'TCenter':2457140.367782,                                       # Excel
                'TLength':3.5496/24.,                                           # Excel
                'ecc':0.18,                                                     # ±0.042 Petigura+12017
                'omega_deg':0.,                                                 # "
                'inclination':90.0,                                             # ±86.135130 2.726937/3.999993 (Mayo+2018)
                'Kstar':21.3,                                                   # Petigura+2017
                'aRs':9.86,                                                     # ±0.54/1.44 Livingston+2018
                'lambda_proj':0.                                                # unknown
                },
    },
    #------------------------------
    
    'TOI3071':{
        
        'star':{ #F6
            'Rstar':1.31,                                                       # ±0.04 Hacker+2024 
            'Mstar':1.29,                                                       #0.02 ±0.04 Hacker+2024
            'veq'  :3.14,                                                       # ±0.02 Hacker+2024
            'mag'  :12.4,                                                       # 
            'istar':90,                                                         # 
            'logg' :4.32,                                                       # ±0.02 Hacker+2024
            'Tpole':6177.,                                                      # ±62 Hacker+2024
            'vmic' :1.29,                                                       # ±0.02 Hacker+2024
            }, 

        'TOI3071b':{                                                            #
                'period':1.266938,                                              # 0.000002 Hacker+2024
                'TCenter':2458594.608,                                          # +- 0.002 Hacker+2024
                'TLength':1.85/24.,                                             # +0.08-0.07 Hacker+2024
                'ecc':0.09,                                                     #0.09> Hacker+2024 Hacker+2024
                'omega_deg':90.,                                                #--
                'inclination':82.1,                                             # +1.3-1.1 Hacker+2024
                'Kstar':33.7,                                                   # ±1.7 Hacker+2024
                'aRs':4.11,                                                     # ±0.13 Hacker+2024
                'lambda_proj':0.                                                #--
                },
    },
    'TIC455':{
        
        'star':{ #K4Ve
            'Rstar':1.179,                                                       # ±0.04 Hacker+2024
            'Mstar':0.69,
            'veq'  :10.65,                                                       # ±0.02 Hacker+2024
            'mag'  :11.15,                                                       # 
            'istar':90.,                                                         # Unknown
            'sysvel': 1.4217e+01                                                 # Educated guess for the transit night
            }, 

        'TIC455b':{                                                            #
                'period':18.7137718,                                           # -0.0000327 +0.0000326 # TB updated with Hritam new ephemeris
                'TCenter':2458600.34299,                                       # -0.0021035 +0,0022118
                'TLength':6.5961/24.,                                          #
                'ecc':0.,                                                      # 0.09> Hacker+2024 Hacker+2024
                'omega_deg':90.,                                               #
                'inclination':90.,                                             # 
                'Kstar':5.,                                                  # 
                'aRs':22.25,                                                   # my guess
                'lambda_proj':0.                                               # --
                },
    },
    #------------------------------

    ##############################################################
    # GJ9827

    #------------------------------
    
    'GJ9827':{
        
        'star':{                                                                # K7V
            'mag':10.250,                                                       # ± 0.138 Passegger+2024
            'Mstar': 0.62,                                                      # ± 0.04  Passegger+2024
            'Rstar':0.58,                                                       # ± 0.03  Passegger+2024
            'veq':1.0342352786,#1.02,                                           # vsini < 1.75. , (Passegger+2024)
            'istar':90.,                                                        # unknown
            'Tpole':4236.,                                                      # ± 12, (Passegger+2024)
            },

        'GJ9827b':{                                                             # Rp/Rs = 0.02401 (Dai+2019)
                'period':1.208974,                                              # ±0.000001, (Passegger+2024)
                'TCenter':2457738.825934,                                       # ±0.0005 , (Passegger+2024)
                'TLength':1.28/24.,                                             # ±0.022/0.020 , (Maria Rosa)
                'ecc':0.,                                                       # < 0.063 (Bonomo+2023)
                'omega_deg':90.,                                                # unknown
                'inclination':87.60,                                            # ±0.41/0.34 (Passegger+2024)
                'Kstar':3.53,                                                   # ±0.39 (Passegger+2024)
                'aRs':6.98926,                                                  # [-3.803e-01 +3.683e-01] Passegger priv. com.
                'lambda_proj':-9.3603441324                                     # unknown
                },
                
        'GJ9827c':{                                                             # Rp/Rs = 0.01887 ±0.00034/0.00037 (Rice+2023)
                'period':3.648103,                                              # ±1e-5*(1.3/1.0)(Passegger+2024)
                'TCenter':2457742.199967,                                       # ±0.0014 (Passegger+2024)
                'TLength':1.83/24.,                                             # ±0.037 , (Maria Rosa)
                'ecc':0.,                                                       # <0.094 , (Bonomo+2023)
                'omega_deg':90.,                                                # unknown
                'inclination': 89.09,                                           # ±+0.21/0.18 , (Bonomo+2023)
                'Kstar':1.06,                                                   # ±0.21 (Passegger+2024)
                'aRs':14.61056,                                                 #  [-8.066e-01 +7.758e-01] Passegger priv. com.
                'lambda_proj':-2.1811130603e+01                                 # unknown
            },
                
        'GJ9827d':{                                                             # Rp/Rs = 0.03073 ±0.00065/0.00060 (Rice+2019)
                'period':6.201812,                                              # ±0.000009 (Passegger+2024)
                'TCenter':2457740.958775,                                       # ± 1e-4*(8./9.) (Passegger+2024)
                'TLength':1.29/24.,                                             # ±0.035/0.029 (Maria Rosa)
                'ecc':0.,                                                       # <0.13  (Bonomo+2023)
                'omega_deg':90.,                                                # unknown
                'inclination':87.66,                                            # ±0.045 (Passegger+2024)
                'Kstar':1.44,                                                   # ±0.43  (Passegger+2024)
                'aRs':20.82083,                                                 # [-1.149e+00 +1.083e+00] Passegger priv. com.
                'lambda_proj':-2.4224943132e+00,                                # unknown
            },
    },
}

    return all_system_params




