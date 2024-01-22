import numpy as np
from constant_data import Rjup,Rearth,AU_1,Rsun,Mearth,Mjup

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
    - veq_spots : equatorial rotational velocity of spots [km/s]
    - alpha_rot_spots, beta_rot_spots : differential rotation coefficients for spots
+ DR law is defined in the same way as for the stellar surface

    
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

all_system_params={

    '55Cnc':{
        
        'star':{
        #Bourrier+2018
        'Rstar':0.943,
        'istar':90.,'veq':1.2563076374e+00, 
            },  
 
    #------------------------------
    '55Cnc_b':{
        #Bourrier+2018
        'period':14.6516,    #+-0.0001
        'TCenter':2455495.587,  #+0.013-0.016
        'ecc':0.,            #fixed
        'omega_deg':90.,          
        'Kstar':71.37,            #+-0.21   
    },
    #------------------------------
    '55Cnc_c':{
        #Bourrier+2018
        'period':44.3989,    #+0.0042-0.0043
        'TCenter':2455492.02,   #+0.34-0.42
        'ecc':0.03,    #+-0.02
        'omega_deg':2.4,            #+43.1-49.2
        'Kstar':9.89,             #+-0.22 
    },
    #------------------------------
    '55Cnc_d':{
        #Bourrier+2018
        'period':5574.2,     #+93.8-88.6
        'TCenter':2456669.3,    #+83.6-76.5
        'ecc':0.13,    #+-0.02
        'omega_deg':-69.1,          #+9.1-7.9
        'Kstar':38.6,             #+1.3-1.4    
    },
    #------------------------------
    '55Cnc_e':{
        'period': 0.73654627,  #+-2.2e-7  Meier-Valdes 2022
        'TCenter':2457063.2096,  # +6e-4-4e-4,  #For the old HARPS,HARPS-N,SOPHIE visits, from B18 
        # 'TCenter':2458870.692582,  #+0.000163 −0.000159   #For the ESPRESSO visits, Meier-Valdes 2022 (do not use, as T0 are set directly for the ESPRESSO transits)
        
        #Bourrier+2018
        'ecc':0.05,     #+-0.03
        'omega_deg':86.0,            #+30.7-33.4
        # 'ecc':0.,            #fixed to circular to match T14 from B18
        # 'omega_deg':90.,           
        'Kstar':6.02,              #+0.24-0.23  
        'inclination':83.59,     #+0.47-0.44          
        'aRs':3.52,             #+-0.01
        # 'TLength':0.0648,        #Sulis+2019
        'TLength':0.0634,        #B18
        # 'TLength':0.0617,        #To match eccentric orbit from B18 (difference of 2.4min with B18)

        
        'lambda_proj':-8.3196050869e+01
    },
    #------------------------------
    '55Cnc_f':{
        #Bourrier+2018  
        'period':259.88,     #+76.4-77.4ma
        'TCenter':2455491.5,    #+-4.8
        'ecc':0.08,    #+0.05-0.04
        'omega_deg':-97.6,          #+37.0-51.3
        'Kstar':5.14,             #+0.26-0.25    
    },
    #------------------------------
    '55Cnc_magc':{
        #Bourrier+2018  
        'period':3822.4,     #+76.4-77.4
        'TCenter':2455336.9,    #+45.5-50.6
        'ecc':0.17,    #+-0.04
        'omega_deg':174.7,          #+16.6-14.1
        'Kstar':15.2,             #+1.6-1.8   
    },    
    #------------------------------
    '55Cnc_g':{  
        'period':0.73654737/2.,
        'TCenter':2455733.0060-(2.5/24.),
        'ecc':0.,
        'omega_deg':90. ,
        'Kstar':3.,   
    },               
    },
        
    #------------------------------
    #------------------------------

    'HD189733':{
        
        'star':{
            # 'dstar':19.45,
            # 'Mstar':0.823,
            # 'SystV':-2.2765,
            
            # #Limb-darkening
            # 'LD_mod':'linear',
            # 'LD_u1':0.816

            #Analysis for Mounzer+2023
            'Rstar':0.784,
            'istar':90.,'veq':3.25, 
            
            },
    
        'HD189733b':{
            #Old
            # 'Msini':1.135,
            # 'TDepth':0.024122,
            # 'Rpl':1.138,
            # 'sma':0.0312,

            #Analysis for Mounzer+2023            
            'period':2.2185751979,
            'TCenter':2459446.49851,
            'TLength':1.81/24. ,# 1.9070525639199452/24.,
            'ecc':0.,
            'omega_deg':90.,
            'inclination':85.46578817370532,
            'Kstar':201.3,	
            'aRs':8.8843,
            'lambda_proj':-0.45
            
            
            },

    },

    #------------------------------
    #------------------------------

    'K2-79':{
        'star':{
            'Rstar':1.265,     #+0.041-0.027    Bonomo+2023
            'istar':90.,'veq':1., 
            },
    
        'K2-79b':{           
            'period':10.99386423,       #TO BE REVISED
            'TCenter':2460214.7753992,        #TO BE REVISED
            'TLength':4.420/24.,  #±0.088 /24.   Crossfield+2016
            'ecc':0.,                   #assumed circular   Bonomo+2023
            'omega_deg':90.,
            'inclination':88.44,   #±0.44        Bonomo+2023
            'Kstar':2.63,  #  +-0.69   	   Bonomo+2023
            'aRs':16.794606,    #0.09880*AU_1_m/(1.265*Rsun*1e3)       Bonomo+2023
            'lambda_proj':0.

            },

    },



    #------------------------------
    'Corot_9':{
        
        'star':{
    #    'dstar':460,
    #    'Mstar':0.99,
    #    'SystV':19.791,
        #Limb-darkening
        'LD_mod':'linear',
        'LD_u1':0.57
            },
        
    
    #------------------------------
    'Corot_9b':{
        'period':95.2738,
        'TCenter':2454603.3447,
        'TLength':0.3367,   
        'ecc':0.11,
        'omega_deg':37.,
        'inclination':89.99,
        'Kstar':38.,	
        'TDepth':0.0155 ,
        'Rpl':1.05,
    #    'sma':0.407,
        'aRs':93.,
    },        
    },
    #------------------------------
    'WASP_8':{
        
        'star':{
        'dstar':87.,   #+-7 pc (Queloz 2010)  not necessary
        #Limb-darkening
        'LD_mod':'quadratic',
        'LD_u1':0.51598802,
        'LD_u2':0.22373799,
        
    #    #Best fit Queloz 2010
    #    'istar':90.,
    #    'veq':1.59,
    
        #Best-fit solid-body
        'istar':90.,
        'veq':1.9,
    
#        #Best-fit DR
#        'istar':124.40181,
#        'veq':2.4745936,
#        'alpha_rot':0.33324478,
#        'beta_rot':0.,           
        
    },

            # 'RpRs':np.sqrt(0.01276),
            # 'LD_mod':'quadratic',
            # 'LD_c_u1':0.51598802,
            # 'LD_c_u2':0.22373799,
            # 'inclination':88.55}

    #------------------------------
    'WASP_8b':{
        'period':8.158715,   # +- 1.6e-5 (Knutson et al. 2014 from Queloz 2010)
        'TCenter':2454679.33393  ,  # +- 0.00047  (Knutson et al. 2014 from Queloz 2010)
        'TLength':0.1832,   # +0.003-0.0024 days (Queloz 2010)  
        'ecc':0.3044   ,  # +0.0039 -0.004 (Knutson et al. 2014)
        'omega_deg':274.215    ,  # + 0.084 – 0.082 deg (Knutson et al. 2014)
        'inclination':88.55,  # +0.15-0.17 deg (Queloz 2010)
        'Kstar':221.1,  # +-1.2 m/s-1 (Knutson et al. 2014)
        'TDepth':0.01276,   #  +- 0.00033 (Queloz 2010)
        'Rpl':1.038,   # +0.007-0.047 Rj (Queloz 2010)  not necessary
        'aRs':1./0.0549,    #Rstar/a = 0.0549 +-0.0024 (Queloz 2010)

        #    #Best fit Queloz 2010
        #    'lambda_proj':-123.
            #Best-fit solid-body
            'lambda_proj':-143, 
    #        #Best-fit DR
    #        'lambda_proj':-142.60778,            
            
            
    },

            },     
    
    #------------------------------
    'GJ436':{
        
        'star':{
        # 'dstar':10.14,  #pc (Van Leeuwen 2007)  not necessary
        # 'Rstar':0.449,
        # 'Rstar':0.438,   #+-0.013    Maxted+2021
        'Rstar':0.425,   #+-0.006    Maxted+2021, combined with literature
    
        # #Best fit SB Bourrier+2018
        # 'istar':39.,
        # 'veq':0.3300552/sin(39.*pi/180.), 
        
    #    #Aligned system
    #    'istar':39.,
    #    'veq':0.3300552/sin(39.*pi/180.),  
    
      
    #    'istar':40.4342563,
    #    'veq':0.1165299,  #0.24,
    #    'alpha_rot':-463.3848183,
    #    'beta_rot':0.,   

        # #Best fit SB Bourrier+2018
        # 'istar':90.,
        # 'veq':3.143177704e-01,             



        #Best fit ESPRESSO reloaded RM
        # 'istar':90.,
        # 'veq':2.9289247493e-01,      
        
        # # #Best fit ESPRESSO
        # 'istar':90.,
        # 'veq':3.0047120229e-01,                  

        #Best fit ESPRESSO + HARPS/HARPS-N
        # 'istar':90.,
        # 'veq':2.9351845084e-01,   
        'istar':3.5753197353e+01,                #en derivant istar
        'veq':2.9351845084e-01/np.sin(3.5753197353e+01*np.pi/180.),  
            },

    #------------------------------
    'GJ436_b':{
        # 'period':2.64389803,   # +2.7e-7 -2.5e-7  d (Lanotte et al. 2014)
        # 'TCenter':2454865.084034, #+3.5e-5 -3.4e-5  BJD_TT (Lanotte et al. 2014)    
        # 'TLength':0.04227,   # approx. d (Knutson et al. 2011)    
        # 'ecc':0.1616,  #+-0.004 (Lanotte 2014)
        # 'omega_deg':327.2    ,  # +1.8 -2.2 deg (Lanotte et al. 2014)
        # 'inclination':86.858,  # +0.049 -0.052 deg (Lanotte et al. 2014)    
        # 'Kstar':17.59,  # +-0.25 m/s-1 (Lanotte et al. 2014)
        # 'TDepth':0.006819,   #+-0.000028 (Lanotte et al. 2014)    
        # 'Rpl':4.096*Rearth/Rjup,   # (+-0.162 Re) -> Rj  (Lanotte et al. 2014) not necessary    
        # 'aRs':14.54, #+-0.14 (Lanotte et al. 2014)
         
    #      #Tests at +-1 sigma       
    #    'period':2.64389803   -2.5e-7,   #+ 2.7e-7,
    #    'TCenter':2454865.084034,  #  -3.4e-5 ,   #+3.5e-5    ,  
    #    'ecc':0.1616,  #  -0.004   ,#+0.004
    #    'omega_deg':327.2, # -2.2  ,#+1.8
    #    'inclination':86.858 +3.*0.049  ,   # -0.052  ,#  +0.049,
    #    'Kstar':17.59 , #  -0.25,   # + 0.25   ,  
    #    'TDepth':0.006819,  # - 0.000028,   #   + 0.000028,       
    #    'aRs':14.54   +0.14 ,   # -0.14,   #+ 0.14  ,

        #     'lambda_proj':72.3021645,   #Best fit SB Bourrier+2018
        # #    'lambda_proj':0.,      #Aligned system
        # #    'lambda_proj':28.3747006,  #170., #120.,  # 135., 

        #ESPRESSO analysis
        'period':2.64389803,   # +2.7e-7 -2.5e-7  d (Lanotte et al. 2014)
        'TCenter':2455475.82450,   #Maxted+2021 -> errors of 8s on mid-transit time at the epoch of the ESPRESSO transits
        'TLength':0.01593*2.64389803,   #Maxted+2021 
        'ecc':0.152,  #+-0.009  (Trifonov et al. 2018)     
        'omega_deg':325.8,   #+5.4-5.7   (Trifonov et al. 2018)    
        'inclination':86.7889,    #Maxted+2021  (a partir du sin i de la Table 11); sini = 0.99843+-0.00003 -> di = dsini/sqrt(1-sini^2) = 180.*0.00003/( sqrt(1.-0.99843^2) * np.pi ) = 0.03 deg 
        'Kstar':17.38,  # +-0.17 m/s-1 (Trifonov et al. 2018)    
        'aRs':14.46,     #+-0.08   #Maxted+2021  

        # 'lambda_proj':1.0749188731e+02,    #Reloaded, Best fit ESPRESSO 
        # 'lambda_proj':1.1407928063e+02,    #Revolutions, osamp 5, Best fit ESPRESSO 
        'lambda_proj':1.1350818542e+02,    #Revolutions, osamp 5, Best fit ESPRESSO + HARPS/HARPS-N
        

    },

    #------------------------------
    'GJ436_c':{      #added for plotting purposes, but does not contribute to the Keplerian
        'period':0.,
        'TCenter':0.,
        'aRs':3752.,
        'ecc':0.,
        'omega_deg':90. ,
        'inclination':89.8,
        'lambda':139.,
        'Kstar':0., 
    },
        
    },





    #------------------------------

    'HD3167':{ 
        
        'star':{
        # 'Rstar':0.87,         #Christiansen+2017 
        # 'Rstar':0.835,         #Gandolfi+2017    
        'Rstar':0.85,         #+-0.020;    S. Sousa's analysis from ESPRESSO data 

     #Dalal+2019
    #     'veq':1.8944367e+00,    

       #Christiansen+2017   HRRM
    #     'veq':1.9005308337e+00,    

            #Christiansen+2017 + CHEOPS   RRM 
        # 'veq':2.5083818936e+00,  

        # 'veq':0.,             #pour empecher l'alignement des CCFintr mais les binner quand meme en mode xp_abs

                
        # 'veq':2.6085795745e+00,'istar':90.,               #final HD3167c, HRRM, a & i fixed, loose priors, C1 & FWHM0
        # 'veq':2.36871718e+00,'istar':90.,                 #final revised HD3167c, HRRM, a & i fixed, loose priors, C1 & FWHM0
        # 'veq':2.4171897935e+00,'istar':90.,             #final HD3167b, HRRM, a & i fixed, loose priors, C1 & FWHM0, prior vsini 2.6
        # 'veq':2.1607816629e+00,'istar':90.,             #final revised HD3167b, HRRM, a & i fixed, loose priors, C1 & FWHM0, prior vsini 2.36


        'veq':2.6,'istar':90.,             #joint fit, istar fixed
        # 'veq':2.6494101203e+00,             #final joint fit HRRM
        # 'istar':1.1155793037e+02,        
        # 'istar':180.-1.1155793037e+02,        


        # #tests plots
        # 'istar':108.49,    #np.arccos(-3.1714065791e-01)*180./np.pi,
        # 'veq':2.3524385670e+00/np.sin(108.49*np.pi/180.)
            },

    #------------------------------
    #Christiansen+2017 + CHEOPS analysis
    'HD3167_b':{
        #'period':0.959641,  #-0.000012+0.000011     C17
        # 'period':0.959642,  #-0.000012+0.000012     CHEOPS
        'period':0.9596550144,  #-6.0559116e-7     +5.90936521e-7  d;     cf paper, derived from T0 of C17 and CHEOPS 
        # 'TCenter':2457394.37454,  #+-0.00043       C17
        # 'TCenter':2459141.90513,      # +0.00048/-0.00050              (from 4 transits: 2459081.447996,        #+ 0.001035 / - 0.001177           CHEOPS
        'TCenter':2458534.444697069,  #+-0,0006708301437  d;     cf paper, derived from T0 of C17 and CHEOPS         
        
        'ecc':0.,       #C17
        'omega_deg':90.,   #C17
        'Kstar':3.58,   #C17
        
        'inclination':83.4,      #C17      +4.6-7.7       #donne b = 0.47 (qui correspond bien au b=0.47+-0.31 donne dans C17)
        'aRs':4.082,          #C17   +0.464-0.986
        # 'inclination':87.1026805,      # + 1.6224408 / - 1.4363617  (± 1.5294013)  #CHEOPS   analyse prelim. des 4 transits, mais moins coherent avec C17 et avec analyse des 12 transits donc eviter (donne b=0.23)          
        # 'aRs':4.6081708,    #  + 0.1055471 / - 0.1062032  (± 0.1058751)     #CHEOPS
        # 'inclination':82.593416,      #+3.670999   -4.357614    #CHEOPS analyse des 12 transits. Donne b=0.51, donc proche de C17, du coup je garde C17 pour ne pas compliquer l'explication de l'utilisation de CHEOPS
        # 'aRs':3.987058,          #+0.442866   -0.590144       #CHEOPS
        
        'TLength':0.06758,           #C17
        # 'TLength':1.5652985/24.,        #  + 0.0375650 / - 0.0395368  (± 0.0385509)     #CHEOPS  
        
        # 'lambda_proj':5.5334361132e-02,   #-32+22             #final, HRRM, a & i fixed, loose priors, C1 & FWHM0, prior vsini 2.6
        # 'lambda_proj':4.6712717536e+00,  #           #final revised, HRRM, a & i fixed, loose priors, C1 & FWHM0, prior vsini 2.4
        'lambda_proj':-6.6162315243e+00,     #final joint HRRM fit
        
        # 'inclination':180.-83.4,  
        # 'lambda_proj':6.6162315243e+00,    
        
        
        
    },
    'HD3167_c':{
        # 'period':29.8454,    #+-0.0012       #C17
        # 'TCenter':2457394.9788,     #+-0.0012             #C17       
                    #Transit HARPS vers 2457663.57  (9 periodes)   -> T0_HARPS = 2457663.5874
                    #T_HARPS = T0+n*P -> s(T_HARPS) = sqrt(sT0**2. + (n*sP)**2.) = sqrt(0.0012**2. + (9.*0.0012)**2.) = 0.010866462165764899 d = 15.647705518701455 min  
        #'TCenter':2458260.526553,       #+0.00016−0.00014 BJD_TDB         Guilluy+2020, priv. comm.
                    #Transit HARPS (20 periodes)   -> T0_HARPS = 2457663.618553
                    #s(T_HARPS) = sqrt(0.00015**2. + (20.*0.0012)**2.) = 34.560674993408334 min      avec le P de C17
                    #s(T_HARPS) = sqrt(0.00015**2. + (20.*0.00098)**2.) = 28.224826518510255 min      avec le P de G17
                    #Delta avec le T0_HARPS derive de C17: 44.9 min  
        'TCenter':2457663.5966234943,            #+-4.4e-4           cf paper, derived from T0 of C17 and Guilluy+2020
        'period':29.846448616261263,             #+-1.9e-05 d        cf paper, derived from T0 of C17 and Guilluy+2020
        'ecc':0.0,
        'omega_deg':0.0,
        'Kstar':2.23,
        'inclination':89.3,   #+0.5-0.96
        'aRs':40.323,     #+5.549 -12.622
        'TLength':5.15/24.,
        
        # 'inclination':89.3-0.96,      #lower values
#        'aRs':40.323-12.622,     
#        'inclination':89.3+0.5,      #upper values
#        'aRs':40.323+5.549,   
#        'inclination':89.47,      #Shweta RM
#        'aRs':43.32,        
#        'inclination':88.94,      #Shweta Tomo
#        'aRs':36.05,          

        #'lambda_proj':-1.1254191e+02,           #Dalal+2019
        #'lambda_proj':-101.199854,          #Christiansen+2017   HRRM
        # 'lambda_proj':-9.7320544656e+01,        #Christiansen+2017 + CHEOPS   RRM
        # 'lambda_proj':-1.0067778004e+02,  #+7.2-7.6      #final, HRRM, a & i fixed, loose priors, C1 & FWHM0
        # 'lambda_proj':-1.0220090491e+02,   #-7.5456599869e+00	+8.7574427041e+00      #final revised, HRRM, a & i fixed, loose priors, C1 & FWHM0
        'lambda_proj':-1.0888254834e+02,       #final joint HRRM fit

        # 'inclination':180.-89.3,  
        # 'lambda_proj':1.0888254834e+02, 

    },
    'HD3167_d':{
        'period':8.509,
        'TCenter':2457806.07,
        'ecc':0.,
        'omega_deg':90.,
        'Kstar':2.39,
        'aRs':0.07757*AU_1/(0.86*Rsun),
        # 'inclination':90.,   #unknown
        # 'lambda_proj':0.,    #unknown
        'inclination':87.,   #to get b=1.05
        'lambda_proj':-1.0067778004e+02,    #unknown

    },

    # #------------------------------
    # #Gandolfi+2017
    # 'HD3167_b':{
    #     'period':0.959632,   #+-0.000015
    #     'TCenter':2457394.37442,   #+0.00060-0.00055
    #     'ecc':0.,
    #     'omega_deg':90.,
    #     'Kstar':4.02,
    #     'inclination':88.6,   
    #     'aRs':4.516,       
    #     'TLength':1.65/24.,   
    # },
    # 'HD3167_c':{
    #     'period':29.84622,     #+-0.00098
    #     'TCenter':2457394.97831,       +-0.00085         #G17
                    #Transit HARPS (9 periodes)   -> T0_HARPS = 2457663.59429
                    #s(T_HARPS) = sqrt(0.00085**2. + (9.*0.00098)**2.) = 12.759643280280212 min 
                    #Delta avec le T0_HARPS derive de C17: 9.9 min
                    #Delta avec le T0_HARPS derive de Guilluy+2020: 34.93 min   
    #     'ecc':0.05,
    #     'omega_deg':178.,
    #     'Kstar':1.88,
    #     'inclination':89.6,
    #     'aRs':46.5,
    #     'TLength':0.2,
    # },
    # 'HD3167_d':{
    #     'period':8.509,
    #     'TCenter':2457806.07,
    #     'ecc':0.,
    #     'omega_deg':90.,
    #     'Kstar':2.39,
    # },


        
    },

    #------------------------------
    #Analyse EULER M. Lendl, all visits but 2017
    'WASP121':{
        
        'star':{
        'Rstar':1.45817285,   
        # 'LD_mod':'quadratic',
        # 'LD_u1':0.37584753,
        # 'LD_u2':0.17604912
        
       'veq':9.7705730e+01,     
        'alpha_rot':7.9316453e-02, 
        'beta_rot':0., 
        'istar':8.1384764,
#            'istar':171.85645, 
        
#            'alpha_rot':0., 
#            'istar':90.,        
            },
    #------------------------------
    'WASP121b':{

        #Fit final
        'period':1.27492504,       #1.274924889550925,
        'TCenter':2458119.72074,     #2458119.7207479831666   ,   #BJD_TDB (donnees HARPS)        
        'ecc':0.,
        'omega_deg':90.,
        'Kstar':177.0,
        'RpRs':0.12534,    #0.12626795,   EULER
        'TLength':0.12008178002208923,
        'aRs':3.8131,    #3.8219354549605975,            #+0.00730584688945024 -0.007885253069904952  
#         'aRs':4.6,
#        'inclination':88.51157701,                   #From EULER fit
        'inclination':88.489694,    #From RM

            'lambda_proj':8.7201724e+01,

    },
        
        
    },

    #------------------------------
    #Kelt-9
    'KELT9':{
        
        'star':{
        'Rstar':2.196,   #Hoiejmakers+2019   
        'LD_mod':'quadratic',      #from EXOFAST online, V band, Teff=10170 K (-> 10000 car bug sinon), logg=4.093, Fe/H=-0.03  (Gaudi+2017)
        'LD_u1':0.22728092,
        'LD_u2':0.32660916,
        
         'istar':90.,
        'veq':91.,#111.4,  # 1.2055425799e+02, 
       
        
    },
        # main_pl_params={
        #     'RpRs':np.sqrt(0.00677),
        #     'LD_mod':'linear',
        #     'LD_c_u1':0.35,
        #     'inclination':86.79} 
            
    #------------------------------
    'Kelt9b':{
        'period':1.4811235,
        'TCenter':2457095.68572,    
        'ecc':0.,
        'omega_deg':90,
        'Kstar':276.,
        'RpRs':0.08228,
        'aRs':3.153,    
        'inclination':86.79-0.5,
        'TLength':0.16316,
        
            'lambda_proj':-84.,  #-84.8,   #-1.4948470742e+00*180/np.pi,
    },        
        
        },


    #------------------------------
    #WASP76
    'WASP76':{
        
        'star':{
        'Rstar':1.77,  #Tabernero+2021 
        'istar':90.,
        # 'veq':1.48,    #Ehrenreich+2020
        'veq':8.5678789895e-01,      #ANTARESS I
        
        #Properties for theoretical atmosphere grid
        'Tpole':6316.,
        'logg':4.13,
        'vmic':1.38,        
            },
  
    #------------------------------
    'WASP76b':{
        'period':1.80988198,    #Ehrenreich+2020
        'TCenter':2458080.626165,      #Ehrenreich+2020
        'ecc':0.,    #Ehrenreich+2020
        'omega_deg':90,   #Ehrenreich+2020
        'Kstar':116.02,   #Ehrenreich+2020   
        # 'aRs':4.08,       #Ehrenreich+2020
        # 'inclination':89.623,    #Ehrenreich+2020
        'aRs':4.0589878141,       #ANTARESS I    
        'inclination':8.9603775940e+01,    #ANTARESS I    
        'TLength':0.15818,
        # 'lambda_proj':61.28,   #Ehrenreich+2020
        'lambda_proj':-3.7082467195e+01    #ANTARESS I    
        
    },
     
    },

    #------------------------------
    #WASP127
    'WASP127':{
        
        'star':{
        'Rstar':1.33,
        'LD_mod':'quadratic',      
        'LD_u1':0.44012,
        'LD_u2':0.271056,
        
        'istar':90.,
        'veq':0.3,  
            },        
    #------------------------------
    'WASP127b':{
         'period': 4.178062, 
         'TCenter':2457808.60283, 
         'ecc':0.,
         'omega_deg':90.,
         'Kstar':22.4, 
         'aRs':7.95,
         'inclination':88.2,
         'TLength': 0.1795,
                     'lambda_proj':0.,
    },        
    },

    #------------------------------
    #HD209458
    'HD209458':{
        
        'star':{
        # 'Rstar':1.1550,    #Used in Bonomo+2017, from SILVA-VALIO A. 
        'Rstar':1.160,   #Casasayas-Barris+2021
        
        #   #SB
        # 'veq':4.2277446532,
        # 'istar':90.,

        #  #SB chi2 ANTARESS I
        # 'veq':4.2732126941e+00,
        # 'istar':90.,
 
        #Joint prof mcmc ANTARESS I
        'veq':4.2721415478,
        'istar':90.,        
 
        #DR alpha neg.
        # 'veq':4.62694508e+00,
        # 'istar':120.66761463443008,
        # 'alpha_rot':-1.63781472e-01,             

  
        # 'veq':4.62694508e+00,
        # 'istar':160.,
        # 'alpha_rot':-1., 
        
        'mag': 	7.65,
        

        # 'A_R':0.5, 
        # 'ksi_R':50.0, 
        # 'A_T':0.5,
        # 'ksi_T':0.,         
        # 'eta_R':20.0, 
        # 'eta_T': 20.0,
        
        
        #ANTARESS I, artificial view, DR 
        'veq':4.,
        'istar':50.,        
        'alpha_rot':0.5,
        # 'c1_CB':-2. 
        
        # #ANTARESS I, artificial view oblate 
        # 'Rstar':2.029,        
        # 'veq':2.*np.pi*2.029*Rsun/(8.64*3600.),    #285 km/s
        # 'Mstar':1.8,
        # 'f_GD':0.1947,
        # 'Tpole':8450.,
        # 'beta_GD':0.190,
        # 'istar':50. ,               

        # #ANTARESS I, mock dataset for comparison of precisions
        # 'veq':25.,
        # 'istar':90.,    

        # #ANTARESS I, mock dataset for multi-transits
        # 'veq':15.,
        # 'istar':90., 
        
        # #Properties for theoretical atmosphere grid
        # 'Tpole':6026.,
        # 'logg':4.307,
        # 'vmic':0.85,    
        # 'vmac':3.98  
        
            },
    #------------------------------
    'HD209458b':{      #From Casasayas Barris + 2020
        'period':3.52474859,    #Bonomo+2017
        # 'aRs':8.76,     #Torres+2008   
        # 'inclination':86.71,      #Torres+2008       donne b = 0.502734        
        # 'TCenter':2454560.80588,   #Evans+2015  (incertitude sur T0 d'aujourd'hui : 0.6505058110240061 min)
        # 'aRs':8.87  ,              #Evans+2015        
        # 'inclination':86.78  ,     #Evans+2015  
        'ecc':0.,       #Bonomo+2017
        'omega_deg':90, #Bonomo+2017
        'Kstar':84.27,    #Bonomo+2017      
        'TLength':0.12408,      #Richardson+2006

        # 'TCenter':2454560.80588  +(1.5/24./60.)   ,   #Exploration
        # 'inclination':86.71,     
        # 'aRs':8.92,

        #Derived from fit assuming SB model
        'TCenter':2454560.80588  +(1.260826849/24./60.) ,        #2454560.806755574
        'inclination':86.71,     
        'aRs':8.903744503359597,



        
        
        # 'period':3.52474859,          #Fit Monika Lendl
        # 'TCenter':2458684.75913806,   
        # 'ecc':0.,       
        # 'omega_deg':90, 
        # 'Kstar':84.27,   
        # 'RpRs':0.12300157, 
        # 'aRs':8.81000000 ,      
        # 'inclination':86.71,     
        # 'TLength':0.12819010           
 
            # 'lambda_proj':1.5875666090,    #SB nominal (ANTARESS)
            # 'lambda_proj':1.0861926459e+00   #SB chi2 (ANTARESS I)
            # 'lambda_proj':0.4158672165556291,    #DR alpha neg.
            # 'lambda_proj':0.4158672165556291,    
            'lambda_proj':1.0699092308,    #Joint prof mcmc ANTARESS I
      
        
        # #ANTARESS I, mock, oblate
        # 'lambda_proj':40.
        
        # #ANTARESS I, mock, multi-tr
        # 'lambda_proj':0.   

        },
    'HD209458c':{      #Mock planet, multi-tr
        'period':3.52474859*2.,
        'ecc':0.,       
        'omega_deg':90, 
        'Kstar':84.27,         
        'TLength':0.12408*3.,      
        'TCenter':2454560.806755574 + (1.2/24.),
        'inclination':86.,     
        'aRs':(8.903744503359597*2)**(2./3.),        
        'lambda_proj':70.         
    },
    },

    #------------------------------
    #Altair (tests GD Barnes+2009)
    'Altair':{
        
        'star':{
        'Rstar':2.029,        
        'veq':2.*np.pi*2.029*Rsun/(8.64*3600.),    #285 km/s
        'Mstar':1.8,
        'f_GD':0.1947,
        'Tpole':8450.,
        'beta_GD':0.190,
        'istar':90.  #70.
            }, 

    #------------------------------
    #Altair (tests GD Barnes+2009)
    'Altair_b':{
        'period':3.04,
        'ecc':0.,      
        'omega_deg':90, 
        'Kstar':0.,
        'TLength':4./24., 
        'TCenter':2455000.,
        'inclination':90., # 88.,     
        'aRs':0.05*AU_1/(2.029*Rsun),   
        'lambda_proj':90.   #45.,
        },

        },

    #------------------------------
    #Corot7
    'Corot7':{
        
        'star':{
            'Rstar':0.91500276,
                      'istar':90.,
        'veq':1.8944367e+00, 
            },
 

     #------------------------------
      'Corot7b':{
            'period':0.85359159,   #+-5.7e-7  d
            'TCenter':2454398.07756,  #+4.5e-4 -7.4e-4
            'ecc':0.,
            'omega_deg':90.,
            'Kstar':3.94 ,   #+- 0.57 m/s
            'aRs':4.484,   #+-0.070
            'inclination':80.78,   #+0.51−0.23 deg
            'TLength':1.059/24.,   
            'lambda_proj':0.,          
    },      
     #------------------------------
      'Corot7c':{
            'period':3.70,  #+-0.02  d
            'TCenter':2455953.54,  #+-0.07  d
            'ecc':0.12,   #+-0.06
            'omega_deg':90.,   #
            'Kstar':6.01,    #+-0.47 m/s  
    },         
        },

    #------------------------------
    #Nu 2 Lupi
    'Nu2Lupi':{
        
        'star':{'Rstar':1.058,'istar':90.,'veq':2.*np.pi*1.058*Rsun/(23.8*24.*3600.),'mag':5.65},
   
     #------------------------------
      'Nu2Lupi_b':{
            #Delrez+2020          
            'period': 11.57797,   
            'TCenter':2458944.3726,   
            'ecc':0.0,   #
            'omega_deg':90.,   
            'Kstar':1.46,   # m/s 

            # #Kane+2020
            # 'period':11.57782,
            # 'TCenter':2458631.7671,
            # 'ecc':0.075,
            # 'omega_deg':174.,
            # 'Kstar':1.40   
          },    
     #------------------------------
      'Nu2Lupi_c':{
            #Delrez+2020
            'period':27.59221,  
            'TCenter':2458954.40990,       
            'ecc':0.,   
            'omega_deg':90.,   #
            'Kstar':2.61,   #   m/s
            'aRs':34.97,   #
            'inclination': 88.571,   #
            'TLength':3.251/24. ,   
            
            # #Kane+2020
            # 'period':27.5909,
            # 'TCenter':2458650.8946,       
            # 'ecc':0.036,
            # 'omega_deg':141.,
            # 'Kstar':2.55,
            # 'aRs':37.30,
            # 'inclination':88.684,
            # 'TLength':0.1334
            'lambda_proj':0.,  
            
    },      
     #------------------------------
      'Nu2Lupi_d':{
            #Delrez+2020          
            'period':107.245,   #
            'TCenter':2459331.18759,   #revised by CHEOPS
            'ecc':0.,   #
            'omega_deg':90.,   #
            'Kstar':1.30,   
            'aRs':86.46,   #
            'inclination': 89.73,   #
            'TLength':8.87/24. ,  
            'lambda_proj':0., 
            # 'lambda_proj':90., 
            
            # #Kane+2020
            # 'period':107.63,
            # 'TCenter':2455902.7,
            # 'ecc':0.075,
            # 'omega_deg':-175.,
            # 'Kstar':1.51            
    },       
  
            },

    #------------------------------
    #GJ 9827
    'GJ9827':{
        
        'star':{'Rstar':0.579,   # ;+-0.018  KOSIAREK+2020
           'istar':90.,
        'veq':7.8004826717e-01,   },
     #------------------------------
      'GJ9827d':{
            'period':6.20183 ,   #1e−05     ;KOSIAREK+2020
            'TCenter':2457740.96114,   #+0.00045−0.00044    KOSIAREK+2020
            'ecc':0.,   
            'omega_deg':90.,   
            'Kstar':1.7,     #+-0.3    KOSIAREK+2020
            'aRs':20.61189056,   #  =0.0555*AU_1/(0.579*Rsun)    KOSIAREK+2020
            'inclination': 87.443 ,   #+0.045−0.045   ;RICE+2018
            'TLength':0.05095,   
            'lambda_proj':-4.1575566287e+01,       
    },      
     #------------------------------
      'GJ9827b':{
            'period':1.2089765,    #2.3e−06   KOSIAREK+2020
            'TCenter':2457586.5479,    #+0.003−0.0026    KOSIAREK+2020
            'ecc':0.,  
            'omega_deg':90.,  
            'Kstar':4.1,   #+-0.3     KOSIAREK+2020 
            'aRs':6.930051856,   #  =0.01866*AU_1/(0.579*Rsun)    KOSIAREK+2020
            'inclination':86.07,     #+0.41−0.34   ;RICE+2018
            'TLength':0.05270,   
            'lambda_proj':0.
    },    
     #------------------------------
      'GJ9827c':{
            'period':3.648096,     #+-2.4e-05       KOSIAREK+2020
            'TCenter':2457742.19929,    #+0.00072−0.00071     KOSIAREK+2020
            'ecc':0.,   
            'omega_deg':90.,   
            'Kstar':1.13,        # +-0.29     KOSIAREK+2020
            'lambda_proj':0.
    },   
            },

    #------------------------------
    #TOI-178
    'TOI178':{
        
        'star':{'Rstar':0.651, 
              'istar':90.,
              'veq':1.5,
              'mag':11.95},

 
     #------------------------------
      'TOI178b':{
            'period':1.914557,    
            'TCenter':2458741.6371,  
            'ecc':0.,   
            'omega_deg':90.,   
            'Kstar':1.05, 
            'aRs':8.61,
            'inclination':88.,
            'TLength':1.649/24., 
            'lambda_proj':0.,                 
    },       
      'TOI178c':{
            'period':3.238458,    
            'TCenter':2458741.4790,  
            'ecc':0.,   
            'omega_deg':90.,   
            'Kstar':2.77,  
            'aRs':12.25,
            'inclination':88.04,
            'TLength':1.89/24., 
            'lambda_proj':0.,         
    }, 
      'TOI178d':{
            'period':6.557694,    
            'TCenter':2458747.14645,  
            'ecc':0.,   
            'omega_deg':90.,   
            'Kstar':1.34,  
            'aRs':19.58,
            'inclination':88.44,
            'TLength':2.054/24., 
            'lambda_proj':0.,         
    }, 
      'TOI178e':{
            'period':9.961872,    
            'TCenter':2458751.4661,  
            'ecc':0.,   
            'omega_deg':90.,   
            'Kstar':1.62, 
            'aRs':25.83,
            'inclination':88.67,
            'TLength':2.455/24., 
            'lambda_proj':0.,          
    }, 
      'TOI178f':{
            'period':15.231930,    
            'TCenter':2458745.7177,  
            'ecc':0.,   
            'omega_deg':90.,   
            'Kstar':2.76,
            'aRs':34.44,
            'inclination':88.731,
            'TLength':2.356/24., 
            'lambda_proj':0.,           
    }, 
      'TOI178g':{
            'period':20.709460,    
            'TCenter':2458748.0293,  
            'ecc':0.,   
            'omega_deg':90.,   
            'Kstar':1.30, 
            'aRs':42.28,
            'inclination':88.833,
            'TLength':2.189/24., 
            'lambda_proj':0.,          
    },
            },


    #------------------------------
    'TOI858':{
        
        'star':{
        'Rstar':1.333,'istar':90.,
        # 'veq':6.9,   #2 visits, common prof, no osamp
        # 'veq':8.1906894494e+00,   #visit 1, no osamp
        # 'veq':3.5879186524e+00,   #visit 2, no osamp
        'veq':7.0893711287,    #2 visits, diff prof, osamp5
            },
    #------------------------------
    'TOI858b':{
        'period':3.2797176,
        'TCenter':2458386.45269,
        'ecc':0.,
        'omega_deg':90,
        'Kstar':139.4,    
        'aRs':7.18,
        'inclination':86.83,
        'TLength':0.14972,
        # 'lambda_proj':100.
        # 'lambda_proj':9.9419302844e+01  #visit 1 no osamp
        # 'lambda_proj':8.0891972284e+01  #visit 2 no osamp
        'lambda_proj':9.9294680793e+01     #2 visits, diff prof, osamp5

    }, 
    },
    
    #------------------------------    

    'Sun':{
        
        'star':{
        'Rstar':1.,'istar':90.,'veq':0.,  
            }, 
    #------------------------------      
    'Moon':{

        #'period':365.25,  #Periode utile pour le run? ###### utilisé pour régler KeyError
        #'period':371.04255265403975,  #Periode utile pour le run? ###### utilisé pour régler KeyError
        'period': 1, #  32.067115558638714+1.5, #Periode calculer à partir de transit théorique et determination de transit duration on JPL


        #For moon 2
        'TCenter': 2459198.15842591254,
          
        'ecc':0.,
        'omega_deg':0.0,
        'Kstar':0., 
        'inclination':89.99925482100697, 
        'aRs':214.93709,    
            
        'TLength':0.09528936026617885,     #Calculated with routine
        
        #With indexes removed
        'lambda_proj':-5.769234751276427,           
    },
    #------------------------------
    'Mercury':{
        
        'period': 1, 

        'TCenter':2458799.13865453,  #JDUT à mettre en BJD

        'ecc':0.,
        'omega_deg':0.0,
        'Kstar':0.,
        'inclination':89.95323420684971, 
        'aRs':145.2,       #À voir
        'TLength':0.1, #À voir    
        'lambda_proj':-101.199854, #copied from HD3167_c
        
        },
    },
   
    #------------------------------    

    'TIC61024636':{
        
        'star':{
        'Rstar':1.125811,'istar':90.,'veq':0.5,'mag':12., 
            }, 
      
    #------------------------------
    'TIC61024636b':{
        'period':50.070,
        'TCenter':2459054.3096,
        'ecc':0.71,
        'omega_deg':184.85,
        'Kstar':388.57,    
        'aRs':60.52,
        'inclination':89.839,    #  180.*np.arccos(0.17/60.52)/np.pi
        'TLength':4.988/24.,
        'lambda_proj':0.
    },  
    },
        
    #------------------------------
    #HIP 41378
    'HIP41378':{
        
        'star':{
        # 'Mstar':1.16,
        'Rstar':1.28,
        'istar':90.,'veq':6.8402247789e+00},
      
      
    #------------------------------
    'HIP41378b':{               #Santerne+2019
            'period':15.57208 ,  
            'TCenter':2457152.283,   
            'ecc':0.05,  
            'omega_deg':270.,
            'Kstar':1.6,
    },       
     #------------------------------
      'HIP41378c':{   #Santerne+2019
            'period':31.70603 ,  
            'TCenter':2457163.162, 
            'ecc':0.05,  
            'omega_deg':125.,
            'Kstar':0.8,
    },      
     #------------------------------
      'HIP41378d':{       #Santerne+2019
            'period':278.3618 , 
            # 'TCenter':2457166.261,    T0 nominal (mais presence de TTVs)
            # 'TCenter':2458836.48,  #0.02  transit de decembre 2019
            'TCenter':2458836.4318,   #T0 nominal updated
            'ecc':0.08, 
            'omega_deg':122.,
            # 'ecc':0.,        #fixed in RM analysis
            # 'omega_deg':90.,
            'Kstar':0.12,        
            'aRs':147.03,
            'inclination':89.81,  
            'TLength':12.67/24.,
            'lambda_proj':5.7108035371e+01
    },      
     #------------------------------
      'HIP41378e':{    #Santerne+2019
            'period':368. ,  
            'TCenter':2457142.0170,   
            'ecc':0.14, 
            'omega_deg':93.,
            'Kstar':0.9,
    },      
     #------------------------------
      'HIP41378f':{     #Santerne+2019
            'period':542.0797 ,  
            'TCenter':2457186.9153,  
            'ecc':0.0119,  
            'omega_deg':98.1,
            'Kstar':0.9,
    },      
     #------------------------------
      'HIP41378g':{      #Santerne+2019
            'period':62.06 ,  
            'TCenter':2457150.,  
            'ecc':0.06,
            'omega_deg':146.,
            'Kstar':1.0,
    },      
         
            },

    #------------------------------
    #HD 15337
    'HD15337':{
        
        'star':{'Rstar':0.840,'istar':90.,'veq':1.},
      
    #------------------------------
    'HD15337b':{               
            'period':4.755980 ,  
            'TCenter':2458625.48100,   
            'ecc':0.124,  
            'omega_deg':66.,
            'Kstar':3.1079 ,    #Dumusque+2019
    },       
    'HD15337c':{      
            'period':17.180701, 
            'TCenter':2458603.5375,
            'ecc':0.178, 
            'omega_deg':-30.,
            # 'omega_deg':-100.*0.,   #set t0 0 (consistent within ~1sigma with nominal value, otherwise best-fit does not correspond to the observed LC - probably 
                                    #because it was fit from RV alone. At -100deg the major axis is nearly aligned with the LOS, with the periastron away from us & closer to the star 
                                    # / the apiastron toward us & farther from the star, and because orbit is inclined the periastron is thus at a lower b / the apastron at a larger b, hence
                                    # during transit near the apastron the duration is smaller than it should be
            'Kstar':2.4819,    #Dumusque+2019        
            'aRs':31.53,
            'inclination':88.47,  
            'TLength':0.0950,   #2.28
            'lambda_proj':0.
    },     

            },

    #------------------------------    

    'TOI-3362':{
        
        'star':{
        'Rstar':1.83,'istar':90.,'veq':20.0,'mag':10.86,'Mstar':1.445,  
            },
    

    #------------------------------
    'TOI-3362b':{
        'period':18.09547,
        'TCenter':2458529.325,
        'ecc':0.815,
        'omega_deg':50.873,    
        'aRs': 0.153*AU_1/(1.83*Rsun),   #=17.978
        'inclination':89.140,
        'TLength':2.621/24.,
        'Msini':5.029*np.sin(89.140*np.pi/180.),
        'lambda_proj':0.,
        # 'lambda_proj':90.,

    }, 
    },    

    #------------------------------    

    'K2-139':{
        
        'star':{
        'Rstar':0.862 ,'istar':90.,'veq':2.8,'mag':11.653, 
            },


    #------------------------------
    'K2-139b':{
        'period':28.38236,
        'TCenter':2457325.81714,
        'ecc':0.12,
        'omega_deg':124.,    
        'aRs':44.8,
        'inclination':89.62,
        'TLength': 4.89/24.,
        'Kstar':27.7,
        # 'lambda_proj':0.,
        'lambda_proj':90.,

    }, 

    },  

    #------------------------------    

    'TIC257527578':{
        
        'star':{
        'Rstar':1.78,'istar':90.,'veq':8.0,'mag':11.23,
        # 'alpha_rot':0.2,        
            },
    #------------------------------
    'TIC257527578b':{
        'period': 54.1892 ,
        'TCenter':2458432.97890,
        'ecc':0.42,
        'omega_deg':65.38,
        'Kstar':137,    
        'aRs':37.85,
        'inclination':87.9,    
        'TLength':5.028/24.,
        'lambda_proj':90.
    },  
    },    

    #------------------------------
    'MASCARA1':{  
        
        'star':{  #Hooton+2021
        'Rstar':2.082,        
        'veq':123.4034,    #101.7/sin(55.5*np.pi/180.)
        'Mstar':1.900,
        'f_GD':0.0439,
        'Tpole':7490.,
        'beta_GD':0.199,
        
        'istar':55.5
        # 'istar':180.-55.5
            },
    #------------------------------
    'MASCARA1b':{    #Hooton+2021
        'period':2.14877381,
        'TCenter':2458833.488151,
        'ecc':0.,
        'omega_deg':90,
        'Kstar':190.,    
        'aRs':4.1676,
        'TLength':4.226/24.,
        
        # 'inclination':88.45,        
        # 'lambda_proj':-69.2 

        # 'inclination':88.45,        
        # 'lambda_proj':69.2 
        
        # 'inclination':180.-88.45,         
        # 'lambda_proj':69.2 

        'inclination':180.-88.45,         
        'lambda_proj':-69.2 



    },  


        },
    
    #------------------------------    

    'V1298tau':{ 
        
        'star':{   #Provided by A. Sozetti and G. Guilluy
        'Rstar':1.345,     #David2019ApJL) 
        'istar':90.,
        'Mstar':1.101,
        'veq':23.59, #-0.26 +0.31		         			fitted from prior 24 +- 1.  (Calculated from LD fit with free LD coefficient (i), see LD)
        'veq_spots':23.59,
            },
    #------------------------------
    'V1298tau_b':{   #Provided by A. Sozetti and G. Guilluy
        'period':24.1396,			#David priv. comm.
        'TCenter':2457067.0488,      #Set to visit-specific values
        'ecc':0.,
        'omega_deg':92.,
        'Kstar':40.5,
        'aRs':27.0,  #(Ggrav/(4*!dpi^2.))^(1./3.)*(Mstar)^(1./3.)*((Per)^(2./3.)/Rstar) = (2942.71377/(4*np.pi**2.))**(1./3.)*(1.101)**(1./3.)*(24.1414410**(2./3.)/1.345)    ;From Sozzetti et al. 2007
        'inclination':89.00,   # -0.24+0.46  						David2019b
        'TLength':6.42/24.,      #Feinstein2021
        'lambda_proj':4.
    }, 

    },      

    #------------------------------    

    'AUMic':{

        'star':{
            'Rstar':0.75,                #+/- 0.03 #solar radii #Plavchan et al. 2020
            'Mstar':0.50,                #+/- 0.03 #solar mass #Plavchan et al. 2020
            'logg':4.39,                 #+/- 0.03 #computed from Plavchan et al. 2020 Rstar and Mstar values by Zicher et al. 2022
            'veq':7.8,                   #+/- 0.3 #km/s #Klein et al. 2021
            'istar':90,                  #unknown
            # 'istar':30,                  #unknown
            #'mag':5,                     #test
            'mag':8.81,                  #+/- 0.10 #Johnson V magnitude (true value)
            # 'f_GD':0.1947,              #test
            # 'Tpole':8450.,              #test
            # 'beta_GD':0.190,            #test
            'veq_spots':7.8,
        },
        'AUMicb':{
            #'period':2.407000,           #test
            'period':8.463000,           #+/- 0.000002 #days #Martioli et al. 2021 (true value)
            'TCenter':2458330.39051,     #+/- 0.00015 #days #Martioli et al. 2021
            'ecc':0.04,                  #+0.045 -0.025 #Zicher et al. 2022
            'omega_deg':179,             #+128 -125 #degrees #Zicher et al. 2022
            'inclination':89.18,         #+0.53 - 0.45 #degrees #Gilbert et al. 2022
            #'Kstar':8.9,                 #test
            'Kstar':5.8,                 #+/- 2.5 #m/s #Zicher et al. 2022 (true value)
            'TLength':3.50/24,           #+/- 0.08 #days #Martioli et al. 2021
            #'aRs':8,                     #test
            'aRs':18.5,                  #+1.3 - 1.4 #Rstar #Gilbert et al. 2022 (true value)
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


    'WASP49':{ 
        
        'star':{   #Provided by O. Apurva
        'Rstar':0.918377,     
        'istar':90.,
        'veq':1.93,
            },
    },      
    
    #------------------------------    

    'GJ3090':{  
        
        'star':{    #Almenara+2022
        'Rstar':0.516,     
        'istar':90.,'veq':1.
            },
     'GJ3090b':{
            'period':2.853136, 
            'TCenter':2458370.41849, 
            'ecc':0.,
            'omega_deg':90.,
            'Kstar':2.38,
            'aRs':13.18,
            'inclination':87.14,
            'TLength':1.281/24.,
            'lambda_proj':0.,
    },
    'GJ3090c':{
            'period':12.729, 
            'TCenter':2458370.96, 
            'ecc':0.,
            'omega_deg':90.,
            'Kstar':6.01,
        },        
        
    },      
    
    

    #------------------------------
    #RM survey
    #    - properties from overleaf, compiled by Omar
    #------------------------------
    
    'HAT_P3':{
        'star':{ 'Rstar':0.850,
              # 'istar':90,'veq':1.2,
              #'istar':90,'veq':4.5608810292e-01,    #BEST RMR  
              'istar':15.8,'veq':4.5608810292e-01/np.sin(12.1*np.pi/180.),    #BEST RMR, istar  
        },
    'HAT_P3b':{
            'period':2.89973797, 
            'TCenter':2457237.38678, 
            'ecc':0.,
            'omega_deg':90.,
            # 'omega_deg':107.7,
            'Kstar':90.63,
            'aRs':9.8105,
            'inclination':86.31,
            'TLength':2.0808/24.,
            # 'lambda_proj':21.2,
            'lambda_proj':-2.5326984809e+01,   #BEST RMR
    },                     
        },
    'HAT_P11':{
        'star':{ 'Rstar':0.74,
              # 'istar':90,'veq':1.5,   
              # 'istar':90,'veq':6.7023224211e-01,        #BEST RMR              
              'istar':160.,'veq':6.7023224211e-01/np.sin(160.*np.pi/180.),        #BEST RMR, istar 
        }, 
    'HAT_P11b':{
            'period':4.887802443, 
            'TCenter':2454957.8132067,
            'ecc':0.264353,
            'omega_deg':342.185794,
            'Kstar':12.01,
            'aRs':16.50,
            'inclination':89.05,
            'TLength':2.3565/24.,
            'lambda_proj':1.3390417560e+02,        #BEST RMR     
        }, 
    'HAT_P11c':{
            'period':3407., 
            'TCenter':2456746., 
            'ecc':0.601,
            'omega_deg':143.7,
            'Kstar':30.9,
        },  
        },        
    'HAT_P33':{
        'star':{ 'Rstar':1.91,
              # 'veq':13.7,'istar':90,
              # 'veq':1.5651446135e+01,'c1_CB':-2.7791211923e+00,'istar':90.      #CB1   
               'veq':1.5572336137e+01,'istar':90.      #BEST RMR   
        },
    'HAT_P33b':{
            'period':3.47447773, 
            'TCenter':2456684.86508,
            'ecc':0.18,
            'omega_deg':88.,
            # 'Kstar':78.,
            # 'ecc':0.1,
            # 'omega_deg':64.,
            'Kstar':69.7,
            'aRs':5.69,
            'inclination':88.2,
            'TLength':0.18075,
            # 'lambda_proj':3.1974209641e+00,       #CB1
            'lambda_proj':-5.8688006663e+00,       #BEST RMR               
        },
        },
    'HAT_P49':{
        'star':{ 'Rstar':1.833,
              # 'istar':90,'veq':16., 
              'istar':90,'veq':1.0682424499e+01,        #BEST RMR 
        }, 
    'HAT_P49b':{
            'period':2.6915539, 
            'TCenter':2456975.61737,
            
            #Literature
            # 'ecc':0.,
            # 'omega_deg':90.,
            # 'Kstar':188.7,
            
            # #PCK
            # 'ecc':0.06,
            # 'omega_deg':64.,
            # 'Kstar':151.6,

            #PCK circular
            'ecc':0.,
            'omega_deg':90.,
            'Kstar':151.6,
            
            # #Manual search
            # 'ecc':0.,
            # 'omega_deg':90.,
            # 'Kstar':260.,    #best-fit to minimize dispersion        
            
            'aRs':5.13,
            'inclination':86.2,
            'TLength':4.1088/24.,
            'lambda_proj':-9.7740121829e+01,        #BEST RMR                 
        },             
        }, 
    'HD89345':{
        'star':{ 'Rstar':1.657,
              # 'istar':90,'veq':2.6, 
              # 'istar':90,'veq':5.8461681830e-01,        #BEST RMR                         
              'istar':45,'veq':5.8461681830e-01/np.sin(45.*np.pi/180.),        #BEST RMR, istar 
        },
    'HD89345b':{
            #'period':11.81430,      #Yu+2018
            # 'TCenter':2457913.80504,    #Yu+2018
            'period':11.8144024,           #TESS+K2  
            'TCenter':2458740.81147,       #TESS+K2

            # 'ecc':0.203,
            # 'omega_deg':-14.9,
            # 'Kstar':9.49,

            'ecc':0.208,
            'omega_deg':27.,
            'Kstar':8.6,

            'aRs':13.625,
            # 'aRs':13.93,
            'inclination':87.68,  
            'TLength':5.65/24.,  
            'lambda_proj':7.4248978717e+01,        #BEST RMR                     
        },
        },
    
    
    'Kepler25':{
        'star':{ 'Rstar':1.316,'Mstar':1.165,
              # 'istar':90,'veq':9.34,          
              # 'istar':90,'veq':8.8917947373e+00,    #BEST RMR   
              'istar':66.7,'veq':8.8917947373e+00/np.sin(66.7*np.pi/180.),     #BEST RMR, istar 
        },

    'Kepler25b':{
            'period':6.2385347882, 
            'TCenter':2458648.00807, #from A. Leuleu TTV analysis
            'ecc':0.,
            'omega_deg':90.,
            'Msini':0.0275*np.sin(92.827*np.pi/180. ),
        },
    'Kepler25c':{
            'period':12.720370495, 
            'TCenter':2458649.55482, #from A. Leuleu TTV analysis
            'ecc':0.,
            'omega_deg':90.,
            'Msini':0.0479*np.sin(92.764*np.pi/180. ),
            # 'aRs':18.44,        #Benomar2014
            # 'inclination':87.2556288190905,   #Benomar2014
            'aRs':18.336,             #Mills2019, from Kepler's law
            'inclination':180.-92.764,    #Mills2019
            # 'TLength':2.59373/24.,   #WRONG
            'TLength':2.862/24.,    #Benomar2014
            'lambda_proj':-9.4007895574e-01,   #BEST RMR       
        },
    'Kepler25d':{
            'period':122.4, 
            'TCenter':2455715., 
            'ecc':0.,
            'omega_deg':90.,
            'Msini':0.226,
        }, 
        },
    'Kepler68':{
        'star':{ 'Rstar':1.23,
              'istar':90,'veq':0.5,    #Literature 
        }, 
    'Kepler68b':{
            'period':5.3987525913, 
            'TCenter':2455006.85878000,
            'ecc':0.,
            'omega_deg':90.,
            'Kstar':2.7,
            'aRs':10.68,
            'inclination':87.60,
            'TLength':3.459/24.,
            'lambda_proj':0.,      #Non-detection        
        },
    'Kepler68c':{
            'period':9.60502738150, 
            'TCenter':2454969.38207000, 
            'ecc':0.,
            'omega_deg':90.,
            'Kstar':0.59,
        },     
    'Kepler68d':{
            'period':634.6, 
            'TCenter':2455878., 
            'ecc':0.112,
            'omega_deg':-64.7442308497,
            'Kstar':17.75,
        },     

        },

    'K2_105':{
        'star':{ 'Rstar':0.97,
              # 'veq':1.76,'istar':90, 
              'veq':2.1264145833e+00,'istar':90,      #BEST RMR   
        },  

    'K2_105b':{
            'period':8.2669897, 
            'TCenter':2458363.23873,
            'ecc':0.,
            'omega_deg':90.,
            'Kstar':0.,   
            'aRs':17.39,   
            'inclination':88.62,
            'TLength':3.43/24.,
            'lambda_proj':-8.1144006365e+01        #BEST RMR            
        },
    'K2_105c':{
            'period':5.017132695358806, 
            'TCenter':2455498.16044567601, 
            'ecc':0.,
            'omega_deg':90.,
            'Kstar':5.39800692349897,
        },               
        },

    'Kepler63':{
        'star':{ 'Rstar':0.901,
                # 'istar':90,'veq':5.6,        
                # 'istar':90,'veq':7.4734129210e+00,        #BEST RMR  
                'istar':138.,'veq':7.4734129210e+00/np.sin(1.3111278962e+02*np.pi/180.),         #BEST RMR, istar  
        }, 

    'Kepler63b':{
            'period':9.4341503479, 
            'TCenter':2455010.84340000,
            'ecc':0.,
            'omega_deg':90.,
            'Kstar':40.,
            'aRs':19.12,
            'inclination':87.806,
            'TLength':2.903/24.,
            'lambda_proj':-1.3508499878e+02,        #BEST RMR               
        },
        },
   
    'WASP47':{
        'star':{ 'Rstar':1.137,'Mstar':1.040,
              'istar':90,
              # 'veq':1.8,  
              'veq':0.,  #pour alignr sur 0   
        },

    'WASP47b':{
            'period':4.1591492, 
            'TCenter':2457007.932103, 
            'ecc':0.,
            'omega_deg':90.,
            'Kstar':140.84,
        },  

    'WASP47c':{
            'period':588.8, 
            'TCenter':2457763.1 , 
            'ecc':0.295,
            'omega_deg':112.,
            'Kstar':31.04 ,
        },  
    
    'WASP47d':{
            'period':9.03052118 , 
            'TCenter':2459426.5437,
            'ecc':0.01,
            'omega_deg':16.5,
            'Kstar':4.26,
            'aRs':16.34,
            'inclination':89.55,
            'TLength':4.288/24.,
            'lambda_proj':90.,  
        },  
    'WASP47e':{
            'period':0.7895933, 
            'TCenter':2457011.34862, 
            'ecc':0.,
            'omega_deg':90.,
            'Kstar':4.55,
            'aRs':3.20,
            'inclination':86.19,
            'TLength':1.899/24.,
            'lambda_proj':0.,  
        }, 
        },      
    'WASP107':{
        'star':{ 
              #'Rstar':0.67,
              # 'istar':90,'veq':0.45,  
              # 'istar':90,'veq':5.0781969591e-01,       #BEST RMR 
              'istar':15.1,'veq':5.0781969591e-01/np.sin(1.4848928519e+01*np.pi/180.),         #BEST RMR, istar  
              
              #Allart+
              'Rstar':0.66,
        }, 
    
    'WASP107b':{
            # 'period':5.7214742, 
            # 'TCenter':2457584.329897,
            # 'ecc':0.,
            # 'omega_deg':90.0,
            # # 'ecc':0.06,
            # # 'omega_deg':40.0,
            # 'Kstar':14.1,
            # 'aRs':18.02,
            # 'inclination':89.56,
            # 'TLength':2.7528/24.,             
            
            #Allart+
            'period': 5.72148836, 
            'TCenter':2458574.147242,
            'ecc':0.152,
            'omega_deg':43.575,
            'Kstar':14.344,
            'aRs':17.289,
            'inclination':89.61,
            'TLength':2.751/24.,
            
            # 'lambda_proj':118.1,  
            'lambda_proj':-1.5803260704e+02        #BEST RMR             
        }, 
    'WASP107c':{
            # 'period':1088., 
            # 'TCenter':2458520., 
            # 'ecc':0.28,
            # 'omega_deg':-120.,
            # 'Kstar':9.6,
 
            #Allart+
            'period':1105.56, 
            'TCenter':2458484.3757, 
            'ecc':0.24,
            'omega_deg':246.93,
            'Kstar':10.65,            
            
        }, 
        },      
    'WASP166':{
        'star':{ 'Rstar':1.22,
              # 'istar':90,'veq':5.1, 
              'istar':87.9,'veq':5.4931421947e+00,       #BEST RMR 
        }, 
    'WASP166b':{
            'period':5.4435402, 
            'TCenter':2458540.739389,
            'ecc':0.,
            'omega_deg':90.0,
            'Kstar':10.4,
            'aRs':11.14,
            'inclination':87.95,
            'TLength':3.60/24.,
            'lambda_proj':-0.7       #BEST RMR                 
        }, 
         
        },         

    'WASP156':{
        'star':{ 'Rstar':0.76,
              # 'istar':90,'veq':3.80,          
              'istar':90,'veq':3.1729579669e+00,       #BEST RMR 
        }, 
    'WASP156b':{
            'period':3.8361672, 
            'TCenter':2458644.30467,
            'ecc':0.,
            'omega_deg':90.,
            'Kstar':19.,
            'aRs':12.748,
            'inclination':88.902,
            'TLength':2.3926/24.,
            'lambda_proj':1.0568458320e+02,        
        },           
        },     
    'HD106315':{
        'star':{ 
              #'Rstar':1.269,
              'Rstar':1.281,
              # 'istar':90,'veq':13., 
              'istar':90,'veq':9.6575551290e+00,       #BEST RMR  
        }, 
    'HD106315b':{
            'period':9.55287, 
            'TCenter':2457586.5476, 
            'ecc':0.,
            'omega_deg':90.,
            'Kstar':2.88,
        }, 
    'HD106315c':{
            'period':21.05652, 
            'TCenter':2457569.01767,
            'ecc':0.,
            'omega_deg':90.,
            'Kstar':2.53,
            
            # 'aRs':25.10,    #Guilluy21 (donne b = 0.801)
            # 'inclination':88.17,

            'aRs':29.5,    #Kosiarek21 Spitzer, best match T14 (donne b = 0.571472)
            'inclination':88.89,     

            # 'aRs':26.52,    #Kosiarek21 RV (donne b = 0.513)
            # 'inclination':88.89,

            # 'aRs':25.69,    #Rodriguez 2017 (donne b = 0.681 coherent avec leur b = 0.688)
            # 'inclination':88.48,    

            
            'TLength':4.728/24., #Rodriguez 2017

            'lambda_proj':-2.6806273030e+00        #BEST RMR         
        },          
        }, 
    'HD29291':{
        'star':{ 
        'Rstar':1.77,  #Tabernero+2021 
        'istar':90.,
        'veq':1.48,    #Ehrenreich+2020
        },
    'HD29291b':{   #Mock planet
        'period':1.80988198,    
        'TCenter':2459183.74322647,      
        'ecc':0.,    
        'omega_deg':90,   
        'Kstar':116.02,   
        'aRs':4.08,      
        'inclination':89.623,    
        'TLength':0.15818,
        'lambda_proj':61.28   
    },        
        
    },


    
    #------------------------------
    #NIRPS GTO
    #------------------------------
    
    'WASP43':{
        'star':{ 
            'Rstar':0.6506,  #+-0054       Esposito+2017 (analysis more complete, with star + planet properties)
            'istar':90,
            'veq':2.26,     #+-0.54        Esposito+2017
        },
    'WASP43b':{
            #NIRPS Obs in March 2023 = 2460004.
            #Ephem from Patel & Espinoza+2022: nT0(NIRPS) = 1780; sigT0(NIRPS) = 2.3 min
            # 'period':0.8134749,       #+0.0000009-0.0000010  
            # 'TCenter':2458555.80567   #+-0.00005   
            #Ephem from Kokori et al. 2022: nT0(NIRPS) = 4864; sigT0(NIRPS) = 0.2 
            'period':0.81347414,        #+-0.00000003,       
            'TCenter':2456047.051715,   #+-0.000022             
            'ecc':0.,     #all studies
            'omega_deg':90.,
            'Kstar':551.,              #+-3.2   Esposito+2017
            'aRs':4.97,           #+-0.14    Esposito+2017
            'inclination':82.109,      #+-0.088    Esposito+2017 
            'TLength':1.16/24.,     # Esposito+2017 
            'lambda_proj':3.5,      #+-6.8     Esposito+2017  
    },                     
        },    
    
    'L98_59':{
        #Parameters from Demangeon+2021 (most complete analysis, with planet e)
        #    - no evidence from TTV
        'star':{ 
            'Rstar':0.303,     #+0.026-0.023
            'istar':90,
            'veq':1.,          #unknown
        },
    'L98_59b':{
            'period':2.2531136,  #+0.0000012-0.0000015
            'TCenter':2458366.17067,   #+0.00036-0.00033
            'ecc':0.103,    #+0.117-0.045
            'omega_deg':87.71,   #+1.16-0.44
            'Kstar':0.46,   #+0.20-0.17
        }, 
    'L98_59c':{
            #NIRPS Obs in April 2023 = 2460035.
            #nT0(NIRPS) = 451; sigT0(NIRPS) = 1.0 min
            'period':3.6906777,    #+0.0000016-0.0000026
            'TCenter':2458367.27375,    #+0.00013-0.00022
            'ecc':0.103,   #+0.045-0.058
            'omega_deg':261,   #+20-10
            'Kstar':2.19,   #+0.17-0.20
            'aRs':19.00,   #+1.20-0.80
            'inclination':88.11,   #+0.36-0.16
            'TLength':1.346/24.,
            'lambda_proj':0.,     #unknown
    }, 
    'L98_59d':{
            #NIRPS Obs in April 2023 = 2460035.
            #nT0(NIRPS) = 224; sigT0(NIRPS) = 2.7 min
            'period':7.4507245,   #+0.0000081-0.0000046
            'TCenter':2458362.73974,    #+0.00031-0.00040
            'ecc':0.0740,   #+0.0570-0.0460
            'omega_deg':180.,   #+27-50
            'Kstar':1.50,    #+0.22-0.19
            'aRs':33.7,   #+1.9-1.7
            'inclination':88.449,    #+0.058-0.111
            'TLength':0.840/24.,
            'lambda_proj':0.,     #unknown
    },     
    'L98_59e':{
            'period':12.796,   #+0.020-0.019
            'TCenter':2458439.4,    #+0.37−0.36
            'ecc':0.128,   #+0.108-0.076
            'omega_deg':165,   #+40-29
            'Kstar':2.01,   #+0.16-0.20
        },                   
        },      
    
    'GJ1214':{
        #Parameters from Cloutier+2021 (most complete analysis, with star+planet)
        'star':{ 
            'Rstar':0.215,   #0.008
            'istar':90,
            'veq':1.,    #unknown 
        },
    'GJ1214b':{
            #NIRPS Obs in April 2023 = 2460035.
            #nT0(NIRPS) = 2742; sigT0(NIRPS) = 0.5 min
            'period':1.58040433, #+-0.00000013
            'TCenter':2455701.413328,    #+0.000066-0.000059
            'ecc':0.,
            'omega_deg':90.,
            'Kstar':14.36,   #0.53 
            'aRs':14.85, #0.16 
            'inclination':88.7,   #±0.1 
            'TLength':0.8688/24.,
            'lambda_proj':0.,   #unknown   
    },                     
        },      
    
    #------------------------------
    
    'WASP189':{
        
        'star':{
            'Rstar':2.36 , # solar radii # Lendl et al. 2020
            'veq':93.1,    # km/s # Lendl et al. 2020
            'istar':90, 
            },
        'WASP189b':{
            'period':2.7240330,         # days # Lendl et al. 2020
            'TCenter':2458926.5416960,  # days # Lendl et al. 2020
            'TLength':4.3336/24 ,       # days # Lendl et al. 2020
            'ecc':0.,
            'omega_deg':90.,
            'inclination':84.03,        # deg # Lendl et al. 2020
            'Kstar':182.0,	            #  m/s # Lendl et al. 2020
            'aRs':4.6,               # R_star # a*R_star # Lendl et al. 2020
            'lambda_proj':0.
            
            },
        
    },
    
    #------------------------------
    
    'WASP69':{
        
        'star':{
            'Rstar':0.813 , #Allart+2023
            'veq':1.,    #unknown
            'istar':90, #unknown
            },
        'WASP69b':{   #No known
            'period':3.8681390,         #days #Kokori+2022
            'TCenter':2457176.17789,  # days #Kokori+2022
            'TLength':2.23/24. ,       # days  Casasayas-Barris et al. (2017).
            'ecc':0.,
            'omega_deg':90.,
            'inclination':86.71,        # deg #  Casasayas-Barris et al. (2017).
            'Kstar':38.1,	            #  m/s # Casasayas-Barris et al. (2017).
            'aRs':12.00,               # R_star #  Casasayas-Barris et al. (2017).
            'lambda_proj':0.      #unknown    
            
            },
        
    },
    
    #------------------------------

    'TOI4562':{    #Heitzmann+2023
        
        'star':{
            'Rstar':1.152,
            'veq':17.,
            'mag':12.098,
            'istar':90, #unknown
            },
        'TOI4562b':{   
            'period':225.11781,
            'TCenter':2460257.8142 ,  
            'TLength':4.32/24. , 
            'ecc':0.76,
            'omega_deg':60.,
            'inclination':89.06,    
            'Kstar':106.,	           
            'aRs':147.4,               
            'lambda_proj':0.      #unknown    
            },
        
    },
    
    #------------------------------

}






