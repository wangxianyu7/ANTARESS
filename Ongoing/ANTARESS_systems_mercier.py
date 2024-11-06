import numpy as np
from antaress.ANTARESS_general.constant_data import Rjup,Rearth,AU_1,Rsun,Mearth,Mjup

'''
Properties of studied planetary systems
-----------------------------------------------------------------
Star
-----------------------------------------------------------------
    - veq : true equatorial rotational velocity [km/s]
    - istar : stellar inclination [deg]
 + angle counted from the LOS toward the stellar spin axis
 + defined in 0 ; 180
    - dstar: distance sun-star [pc]
 + unused
    - Mstar: stellar mass [Msun]
 + only used if Kstar is not defined
    - SystV: systemic velocity [km/s]
 + unused
    - Rstar: stellar radius [Rsun]
 + used to calculate planet orbital RV in star rest frame
 + used as Req if star's oblateness is accounted for
    - alpha_rot, beta_rot : differential rotation coefficients 
 + DR law is defined as omega = omega_eq(1-alpha_rot*ylat^2-beta_rot*ylat^4)
 + if the star rotates in the same direction at all latitudes, we must ensure alpha_rot+beta_rot<1
 + defined in percentage of rotation rate
    - c1, c2 and c3 : coefficients of the mu-dependent CB velocity polynomial [km/s] 
 + law is RV_CB = c3*mu^3 + c2*mu^2d + c1*mu + c0
 + c0 is degenerate with higher-order coefficients and defined independently 
    - mag : magnitude in the band corresponding to the input light curve
            used only in plot routine to estimate errors on local RVs
    - f_GD : star's oblateness
 + defined as (Req-Rpole)/Req
    - Tpole : stellar temperature at pole [K]
 + used to account for gravity-darkening
 + used as effective temperature for stellar atmosphere grid and mask generation
    - beta_GD : gravity-darkening coefficient 
    - 'A_R', 'ksi_R', ('A_T', 'ksi_T'): radial–tangential macroturbulence properties
      'eta_R', ('eta_T'): anisotropic faussian macroturbulence properties 
 + only used if macroturbulence is activated, for analytical intrinsic profiles
 + set 'T' parameters equal to 'R' for isotropic macroturbulence
    - logg : surface gravity of the star in log(cgs)
    - vmic and vmac: micro and macroturbulence (km/s)
 + only used for stellar atmosphere grid 
    
-----------------------------------------------------------------
Planets
-----------------------------------------------------------------
    - period: Orbital Period [days]
    - TCenter: Epoch of Transit Center [bjd]
 + calculated from Tperi if unknown and eccentric orbit
    - Tperi: Epoch of periastron [bjd]
    - TLength: Duration of Transit [days]
 + only used to estimate the range of the phase table in the light curve plot
    - ecc: Orbital Eccentricity []
    - omega_deg: argument of Periastron [deg]
    - inclination: Orbital inclination [deg]    
 + counted from the LOS to the normal of the orbital plane     
 + defined in 0 ; 90      
    - lambda_proj : sky-projected obliquity [deg]
 + angle counted from the sky-projected stellar spin axis (or rather, from the axis Yperp, which corresponds to the projection of Ystar when istar = istar_in in 0 ; 180) toward the sky-projected normal to the orbital plane
 + defined in -180 ; 180  
    - Kstar: Velocity Semi-Amplitude induced by the planet on the star [m/s]
    - Msini: Mpl*sin(inclination) [Mjup]
 + unused, except if Kstar is not defined  
    - aRs: semi-Major Axis [Rs]
-----------------------------------------------------------------
Note on degeneracies
-----------------------------------------------------------------
    - for more details see the definition of the axis system in ANTARESS_all_routines.py and in Cegla+2016
    - we distinguish between the values given as input here for ip_in ([0;90]), lambda_in (x+[0;360]), and istar_in ([0;180]) and the true angles that can take any value between 0 and 360
      note that the angle ranges could be inverted between lambda and istar
      the reason is that degeneracies prevent distinguishing some of the true angle values, except in specific cases
    - the true 3D spin-orbit angle is defined as psi = acos(sin(istar)*cos(lamba)*sin(ip) + cos(istar)*cos(ip)) 
    - with solid body rotation, any istar is equivalent, and there is a degeneracy between
 1. (ip,lambda) = (ip_in   , lambda_in)    associated with omega = lambda_in
 2. (ip,lambda) = (pi-ip_in,-lambda_in)    associated with omega = -lambda_in
      both solutions yield the same xperp coordinate
      omega is the longitude of the ascending node of the planetary orbit (taking the ascending node of the stellar equatorial plane as reference)   
    - if the absolute value of sin(istar) is known, via the combination of Peq and veq sin(istar), the following configurations are degenerate 
      the coordinate probed by the velocity field is xperp*sin(istar)
      the other coordinate probed is |sin(istar)|
      only combinations of angles that conserve both quantities are allowed below   
 1. (ip,lambda,istar) = (ip_in   , lambda_in   , istar_in)          
         default xperp,yperp,ystar
         default xperp*sin(istar_in) and |sin(istar_in)|
         default psi = psi_in    
    (ip,lambda,istar) = (ip_in   , pi+lambda_in   , -istar_in)  
         same as 1. rotated around the LOS: yields -xperp,-yperp,ystar,psi_in  
    (ip,lambda,istar) = (-ip_in   , pi+lambda_in   , pi+istar_in) 
         same as 1. rotated around the LOS: yields -xperp,-yperp,ystar,pi-psi_in  
 2. (ip,lambda,istar) = (pi-ip_in   , -lambda_in   , pi-istar_in) 
         yields xperp,-yperp, -ystar,psi_in 
    (ip,lambda,istar) = (pi+ip_in   , -lambda_in   , istar_in) 
         yields xperp,-yperp, -ystar,pi-psi_in  
    (ip,lambda,istar) = (pi-ip_in   , pi-lambda_in   , pi+istar_in) 
         same as 2. rotated around the LOS: yields -xperp,yperp,-ystar,psi_in 
    (ip,lambda,istar) = (pi+ip_in   , pi-lambda_in   , -istar_in) 
         same as 2. rotated around the LOS: yields -xperp,yperp,-ystar,pi-psi_in
 3. (ip,lambda,istar) = (ip_in   , lambda_in   , pi-istar_in) 
         yields xperp,yperp, ystar2 = sin(istar_in)*yperp - cos(istar_in)*sin(ip_in), psi2 = acos(sin(istar_in)*cos(lamba_in)*sin(ip_in) - cos(istar_in)*cos(ip_in))    
    (ip,lambda,istar) = (ip_in   , pi+lambda_in   , pi+istar_in) 
         same as 3. rotated around the LOS: yields -xperp,-yperp,ystar2,psi2
    (ip,lambda,istar) = (-ip_in   , pi+lambda_in   , -istar_in) 
         same as 3. rotated around the LOS: yields -xperp,-yperp,ystar2,pi - psi2
 4. (ip,lambda,istar) = (pi-ip_in   , -lambda_in   , istar_in)
         yields xperp,-yperp,-ystar2,psi2
    (ip,lambda,istar) = (pi+ip_in   , -lambda_in   , pi-istar_in) 
         yields xperp,-yperp,-ystar2,pi-psi2
    (ip,lambda,istar) = (pi-ip_in   , pi-lambda_in   , -istar_in) 
         same as 4. rotated around the LOS: yields -xperp,yperp,-ystar2,psi2   
    (ip,lambda,istar) = (pi+ip_in   , pi-lambda_in   , pi+istar_in) 
         same as 4. rotated around the LOS: yields -xperp,yperp,-ystar2,pi-psi2      
    - if the absolute stellar latitude is constrained (eg with differential rotation), then there is a degeneracy between two configurations, which yield the same Psi 
      the coordinate probed by the velocity field is xperp*sin(istar)
      the other coordinate probed is ystar^2
      only combinations of angles that conserve both quantities are allowed
      fewer configurations than in the above case are allowed since |ystar| is more constraining than |sin(istar)|
 1. (ip,lambda,istar) = (ip_in   , lambda_in   , istar_in)          
         conserves xperp*sin(istar) and ystar^2
         yields psi = psi_in 
    (ip,lambda,istar) = (ip_in   , pi+lambda_in,-istar_in)          
         conserves -xperp*sin(-istar) and (ystar)^2
         yields psi = psi_in 
         this configuration is the same as 1., rotating the whole system by pi around the LOS (ie, yielding xperp' = -xperp and yperp' = -yperp in the sky-projected frame)
 2. (ip,lambda,istar) = (pi-ip_in,-lambda_in   , pi-istar_in)       
         conserves xperp*sin(istar) and (-ystar)^2
         yields psi = psi_in
    (ip,lambda,istar) = (pi-ip_in,pi-lambda_in , pi+istar_in)       
         conserves -xperp*(-sin(istar)) and (-ystar)^2
         yields psi = psi_in
         this configuration is the same as 2., rotating the whole system by pi around the LOS (ie, yielding xperp' = -xperp and yperp' = yperp in the sky-projected frame) 
 3. (ip,lambda,istar) = (-ip_in  , lambda_in   , pi-istar_in)       
         conserves xperp*sin(istar) and (ystar)^2
         yields psi = pi - psi_in
    (ip,lambda,istar) = (-ip_in  , pi+lambda_in, pi+istar_in)       
         conserves -xperp*(-sin(istar)) and (ystar)^2
         this configuration is the same as 3., rotating the whole system by pi around the LOS (ie, yielding xperp' = -xperp and yperp' = -yperp in the sky-projected frame) 
         yields psi = pi - psi_in
    but this configuration is not authorized for a transiting planet (it would yield the transit behind the star)
    note that if several planets are constrained, only combinations of 1. together, or 2. together, are possible since they must share the same istar    

'''
def get_system_params():

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

        'TOI3884':{ 
            # 'star':{  #Libby-Roberts+2023 
            #     'Rstar':0.302,              
            #     'istar':155.,                    #unknown
            #     'Mstar':0.298,
            #     'veq':8.495,                     #unknown                
            #     # 'veq_spots':23.59,
            #     },

            # 'star':{  #Equatorial band hypothesis - updated
            #     'Rstar':0.302,              
            #     'istar':90.,                    #unknown
            #     'Mstar':0.298,
            #     'veq':8.495,                     #unknown                
            #     # 'veq_spots':23.59,
            #     },

            # 'star':{  #Equatorial band hypothesis
            #     'Rstar':0.302,              
            #     'istar':89.,                    #unknown
            #     'Mstar':0.298,
            #     'veq':1.504,                     #unknown                
            #     # 'veq_spots':23.59,
            #     },

            'star':{  #Almenara+2022
                'Rstar':0.3043,              
                'istar':90.,                    #unknown
                'Mstar':0.2813,
                'veq':1.504,                     #unknown                
                # 'veq_spots':23.59,
                },

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

            # 'TOI3884_b':{  #Equatorial band hypothesis updated
            #     'period':4.5445697,           
            #     'TCenter':2459556.51669,      
            #     'ecc':0.059,        #Unknown
            #     'omega_deg':190., #very different from previous studies - re-fit?
            #     'Kstar':14.9,     #very different from previous studies
            #     'aRs':25.01,  
            #     'inclination':90.40,   
            #     'TLength':1.646/24.,      
            #     'lambda_proj':30.  #what we want to constrain so will vary
            #     }, 

            'TOI3884_b':{  #Almenara+2022
                'period':4.5445697,           
                'TCenter':2459642.86314,      
                'ecc':0.059,        #Unknown
                'omega_deg':190., #very different from previous studies - re-fit?
                'Kstar':14.9,     #very different from previous studies
                'aRs':25.01,  
                'inclination':90.10,   
                'TLength':1.646/24.,      
                'lambda_proj':50.  #what we want to constrain so will vary
                }, 
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




