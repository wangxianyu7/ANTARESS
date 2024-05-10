#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import pi
import numpy as np

#######################################################################
#A routine to define planet parameters and physical constants
#    - check regularly for papers with more accurate data
#######################################################################

#################################################################
#Astrophysical
#    - online catalog use the equatorial radii of Solar system planets as reference, as do probably most papers
#################################################################

AU_1=149597870.7    #(km) astronomical unit [IAU resolution 2012]
AU_1_m=AU_1*1e3     #(m)
AU_1_cm=AU_1_m*1e2  #(cm)
pc_1=648000./pi     #(au) parsec  [IAU resolution 2015]
c_light=299792.458  #(km/s) speed of light 
c_light_m=299792458.  #(m/s) speed of light
c_light_A=2.99792458e+18  #(A/s) speed of light
AUyr_kms=AU_1/(365.25*24.*3600.)
     
#Sun 
Msun=1.988415829e30 #(kg) solar mass [IAU resolution 2015, with G from CODATA 2006]  
Rsun=695700.        #(km) solar radius [IAU resolution 2015]
Lsun=382.8*1e24     #(kg⋅m2⋅s-3) solar luminosity [IAU resolution 2015]
    
#Jupiter : 
Mjup=1.898130285e27 #(kg) Jupiter mass [IAU resolution 2015, with G from CODATA 2006]  
Rjup=71492.         #(km) equatorial Jupiter radius [IAU resolution 2015] 
    
#Neptune : http://nssdc.gsfc.nasa.gov/planetary/factsheet/neptunefact.html
Mnept=102.42e24      #(kg)
Rnept=24764.         #(km) equatorial radius
    
#Earth : 
Mearth=5.972186e24 #(kg) Earth mass [IAU resolution 2015, with G from CODATA 2006]  
Rearth=6378.1      #(km) equatorial Earth radius [IAU resolution 2015] 

#################################################################
#Physical/chemical
#################################################################

G_usi=6.67428e-11        #(m3 kg-1 s-2) gravitational constant [CODATA 2006 value, see IAU resolution 2015]
sig_sb=5.670373*1e-8     #(kg s-3 K-4) Stefan-Boltzmann constant 
amu=1.660531e-27         #(kg) Atomic mass unit 
k_boltz=1.3806488*1e-23  #(m2 kg s-2 K-1) Boltzmann constant 
h_planck=6.62606957e-34  #(kg m2 s-1) Planck constant   
alpha=2.654008849418e-06 #(m2 s-1) = (pi e^2)/ (4 pi Epsilon_0 m_e c)  with e, epsilon_0, m_e, c in USI
                         #           e = 1.60217662e-19 A s; Eps_0 = 8.854187817620e−12 A^2 s^4 kg−1 m−3; m_e = 9.10938356e-31 kg 
N_avo=6.02214129e23      #(mol-1) = Avogadro constant
elec_ch=1.6021766208e-19 #(C = A s) = elementary charge (proton charge)

#Loschmidt's number (in cm-3) 
#    - number of molecules per unit of volume
#    - N_Loschmidt = N_Avo / V_mol
#           = N_Avo / (R * T / p) = p / (k_B * T) = density for a perfect gas  
#    - value at 0°C and 1 atm is (https://en.wikipedia.org/wiki/Loschmidt_constant):
# N_Loschmidt = 2.6867805e19 /cm3
#      thus value at T = 15°C and p = 1013 hPa (1 atm) is:
# N_Loschmidt[15°C] = N_Loschmidt[0] * T[0°C]/T[15°C]      
#                   = 2.6867805e19 * 273.15 / (273.15 + 15)
#                   = 2.5469168612701716e+19 /cm3 
N_Loschmidt = 2.5469168612701716e+19

#Pressure normalization coefficient
#    - P = n*k*T
# with P in Pa (kg m-1 s-2)  
#      n the number density in atoms/m3
#      k in m2 kg s-2 K-1
#      T in K
#    - conversion
# 1 Pa = 1e-5 bar -> P[bar] = P[Pa]*1e-5 
# 1 atm = 1.01325 bar
# 1 bar = 0.9869232667160128 atm -> P[atm] = P[bar]*0.9869232667160128 
#      thus:
# P[Pa]  = 1e6*n[atoms/cm3]*k[m2 kg s-2 K-1]*T[K]
# P[bar] = 1e6*n[atoms/cm3]*k[m2 kg s-2 K-1]*T[K]*1e-5
# P[atm] = 1e6*n[atoms/cm3]*k[m2 kg s-2 K-1]*T[K]*1e-5*0.9869232667160128 
# P[atm] = Patm_Ncm3_Tk*n[atoms/cm3]*T[K]
#      with Patm_Ncm3_Tk = 1e6*k[m2 kg s-2 K-1]*1e-5*0.9869232667160128
#                      = 9.869232667160128*k[m2 kg s-2 K-1]
#                      = 1.3625944238835428e-22
Patm_Ncm3_Tk=k_boltz*9.869232667160128



#################################################################
#Atomic & molecular data
#    - name is 'X_Y_Z', where
# + X is the element name
# + Y is its ionization level
# + Z is its excited level
#    - properties:
# + m_at : mass of the atom/ion/molecule (in atomic mass unit)
# masses can be retrieved from http://webbook.nist.gov/chemistry/name-ser.html
# or https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl
# + sig_coll : collision cross section between two of the species particles (in m^2) 
#                 if unknown, set to pi*(2*r)^2
# + ion_thresh: ionization threshold for the species (A)
# + q: electric charge (in elementary charges)
#################################################################

dico_all_gases={
    
    #-------------------------
    #Atoms & ion
    #-------------------------
    
    #Hydrogen
    'H_0+_g':{'nature':'gas','m_at':1.007975,'sig_coll':1e-21,'ion_thresh':911.8,'q':0.}, 
    'H_1+_g':{'nature':'gas','m_at':1.00739,'q':elec_ch}, 

    #Deuterium
    'D_0+_g':{'nature':'gas','m_at':2.0141017778,'q':0.},

    #Helium
    #    - radius = 31 pm 
    #    - collisionnal cross-section set to pi*(2.*31.*1e-12)**2.
    'He_0+_g':{'nature':'gas','m_at':4.002602,'ion_thresh':504.2055460784134,'sig_coll':1.208e-20,'q':0.}, 
    'He_0+_23S1':{'nature':'gas','m_at':4.002602,'ion_thresh':2593.01,'sig_coll':1.208e-20}, 
    
    #Carbon
    'C_0+_g':{'nature':'gas','m_at':12.0107,'ion_thresh':1101.1025202547235,'q':0.},
    'C_1+_g':{'nature':'gas','m_at':12.0107,'ion_thresh':508.5485799043555,'q':elec_ch},
    'C_2+_g':{'nature':'gas','m_at':12.0107,'q':2.*elec_ch},

    #Nitrogen
    'N_0+_g':{'nature':'gas','m_at':14.007,'ion_thresh':853.297617210474,'q':0.},
    'N_1+_g':{'nature':'gas','m_at':14.007,'ion_thresh':418.86535061041167,'q':elec_ch},
    'N_2+_g':{'nature':'gas','m_at':14.007,'ion_thresh':261.2942966926909,'q':2.*elec_ch}, 
    'N_3+_g':{'nature':'gas','m_at':14.007,'ion_thresh':160.04149190742464,'q':3.*elec_ch}, 
    'N_4+_g':{'nature':'gas','m_at':14.007,'ion_thresh':126.65659799844914,'q':4.*elec_ch}, 
    'N_5+_g':{'nature':'gas','m_at':14.007,'q':5.*elec_ch},      

    #Oxygen
    'O_0+_g':{'nature':'gas','m_at':15.9994,'q':0.},
    'O_4+_g':{'nature':'gas','m_at':15.9994,'q':4.*elec_ch},
 
    #Sodium
    #    - radius = 190 pm 
    #    - collisionnal cross-section set to pi*(2.*190.*1e-12)**2.
    'Na_0+_g':{'nature':'gas','m_at':22.98976928,'sig_coll':4.536e-19,'q':0.},
    'Na_1+_g':{'nature':'gas','m_at':22.98922072,'q':elec_ch},
    
    #Magnesium
    'Mg_0+_g':{'nature':'gas','m_at':24.3050,'ion_thresh':1621.5556340659411,'q':0.},
    'Mg_1+_g':{'nature':'gas','m_at':24.3050,'q':elec_ch},

    #Silicon
    'Si_1+_g':{'nature':'gas','m_at':28.0855,'q':elec_ch},
    'Si_2+_g':{'nature':'gas','m_at':28.0855,'q':2.*elec_ch},
    'Si_3+_g':{'nature':'gas','m_at':28.0855,'q':3.*elec_ch},    

    #Calcium
    'Ca_1+_g':{'nature':'gas','m_at':40.077,'q':elec_ch},  
    
    #Iron
    'Fe_0+_g':{'nature':'gas','m_at':55.845,'q':0.},
    'Fe_11+_g':{'nature':'gas','m_at':55.845,'q':11.*elec_ch},          
    
    
    #-------------------------
    #Molecules
    #-------------------------
    
    #Dihydrogen
    #    - m_at = 2*m(H)
    #    - radius (Van der Waals) = 202 pm
    #    - collisionnal cross-section set to pi*(2.*202.*1e-12)**2.
    'H2_0+_g':{'nature':'gas','m_at':2.01588,'sig_coll':5.13e-19}, 
    
    #Water
    #    - m_at = 2*m(H) + m(O)
    #    - radius (Van der Waals) = 275 pm
    #    - collisionnal cross-section set to pi*(2.*275.*1e-12)**2.
    'H2O_0+_g':{'nature':'gas','m_at':18.01528,'sig_coll':9.50e-19},
    
    
    
    #################################################################
    #Dust grains
    #    - properties:
    # + rho : density (g cm-3)
    # + Mg: molecular mass of the gas molecules/atoms released from dust due to sublimation (amu) 
    # + L: latent heat of sublimation, in m2 s-2
    # + Pinf: vapor pressure for infinite temperature, in kg m-1 s-2
    # + alpha: evaporation coefficient that parameterises kinetic inhibition of the sublimation process (from Van Lieshout et al. 2014)
    #          set arbitrarily to 0.1 when unknown  
    #    - see change_state for how to obtain L and Pinf from A and B in Van Lieshout+2014 
    #################################################################
 
    # Alumina (gamma alumina)
    #    - from Budaj+2015
    #    - we assume sublimation releases 2 Al (Mg = 26.981539 amu) and 3 O (15.999 amu)
    'alumina':{
        'nature':'dust',
        'rho':2.9,
        'Mg':101.96, #=2*26.981539 + 3*15.999
        'L':np.nan,
        'Pinf':np.nan,
        'alpha':0.1 #arbitrary
    }, 
    # Ammonia
    #    - from Budaj+2015
    'ammonia':{   
        'nature':'dust', 
        'rho': 0.88,
        'Mg':np.nan,
        'L':np.nan,
        'Pinf':np.nan,
        'alpha':0.1 #arbitrary
    },      
    # Carbon at T=400°C 
    #    - opacity in Budaj+2015 is calculated for T=400°C
    #    - we assume the properties for Graphite in Van Lieshout et al. 2014 are valid for this type of carbon   
    'carbon0400':{
        'nature':'dust',
        'rho':1.435,            #Budaj+2015 (T=400°C). Van Lieshout et al. 2014 gives 2.16 for graphite 
        'Mg':12.011,            #Van Lieshout et al. 2014
        'L':64825562.918477856, #=93646.*k_boltz/(12.011*amu), from Van Lieshout et al. 2014
        'Pinf':8.68175420053e14,#=0.1*np.exp(36.7), from Van Lieshout et al. 2014
        'alpha':0.1             #arbitrary
    },
    # Carbon at T=1000°C 
    #    - opacity in Budaj+2015 is calculated for T=1000°C
    'carbon1000':{
        'nature':'dust',
        'rho':1.988,            #Budaj+2015 (T=400°C). Van Lieshout et al. 2014 gives 2.16 for graphite
        'Mg':12.011,            #Van Lieshout et al. 2014
        'L':64825562.918477856, #=93646.*k_boltz/(12.011*amu), from Van Lieshout et al. 2014
        'Pinf':8.68175420053e14,#=0.1*np.exp(36.7), from Van Lieshout et al. 2014
        'alpha':0.1             #arbitrary 
    },    
    # Corundum (Al2O3 = alpha alumina)
    #    - from Van Lieshout et al. 2014
    'corundum':{
        'nature':'dust',
        'rho':2.9, #from Budaj+2015, for consistency with its opacity table (see note in their section 4.1)
        'Mg':101.961,
        'L':6308798.7788943825, #=77365.*k_boltz/(101.961*amu)
        'Pinf':1.16888864240e16,#=0.1*np.exp(39.3)
        'alpha':0.1 #arbitrary
    }, 
    # Cryst. enstatite (MgSiO3)
    #    - from Van Lieshout et al. 2014
    #    - considered as an iron-free pyroxene in Budaj+2015
    'enstatite':{
        'nature':'dust',
        'rho':3.20,
        'Mg':100.389,
        'L':5707156.068029924, #=68908.*k_boltz/(100.389*amu)
        'Pinf':3.5206249346e15,#=0.1*np.exp(38.1)
        'alpha':0.1 #arbitrary
    },   
    # Cryst. fayalite (Fe2SiO4)
    #    - from Van Lieshout et al. 2014
    #    - no associated cross-section
    'fayalite':{
        'nature':'dust',
        'rho':4.39,
        'Mg':203.774,
        'L':2463536.4452695027, #=60377.*k_boltz/(203.774*amu)
        'Pinf':2.35994546824e15,#=0.1*np.exp(37.7)
        'alpha':0.1
    }, 
    # Cryst. forsterite (Mg2SiO4)
    #    - from Van Lieshout et al. 2014
    #    - considered as an iron-free olivine in Budaj+2015
    'forsterite':{
        'nature':'dust',
        'rho':3.27,
        'Mg':140.694,
        'L':3859464.3979731086, #=65308.*k_boltz/(140.694*amu)   
        'Pinf':6.44824949651e13,#=0.1*np.exp(34.1)
        'alpha':0.1 #arbitrary
    },       
    # Iron (Fe)
    #    - from Van Lieshout et al. 2014 
    'iron_dust':{
        'nature':'dust',
        'rho':7.874, 
        'Mg':55.845,  
        'L':7199201.725729473,  #=48354.*k_boltz/(55.845*amu)
#        'L':3.534e8/55.845,    #???? 
        'Pinf':4.8017425537e11, #=0.1*np.exp(29.2)
#        'Pinf':7.801e9,   ????
        'alpha':1.
    }, 
    # Olivine
    #    - Budaj+2015 opacities are for iron-enriched olivine (Mg(1) Fe(1) SiO4) with half–half of Mg–Fe atoms 
    #      Kimura+2002 values are for (Mg(1.1) Fe(0.9) SiO(4) ) olivine     
    'olmg50':{
        'nature':'dust',
        'rho': 3.0,     #Budaj+2015 
        'Mg': 169.0809, #Kimura+2002
        'L':3.21e6,     #Kimura+2002
        'Pinf':6.72e13, #Kimura+2002
        'alpha':0.1 #arbitrary
    },       
    # Perovskite (CaTiO3)
    #    - from Budaj+2015
    'perovskite':{
        'nature':'dust',
        'rho': 4.1,
        'Mg':np.nan,
        'L':np.nan,
        'Pinf':np.nan,
        'alpha':0.1      #arbitrary
    },  
    # Pyroxene 20% iron
    #    - opacity in Budaj+2015 is calculated for Mg(0.8) Fe(0.2) SiO3, but note they correspond to SiO2 alone 
    'pyrmg80':{
        'nature':'dust',
        'rho': 3.0,      #Budaj+2015  
        'Mg':106.6955,   #=0.8*24.305+0.2*55.845+28.0855+3.*15.999     
        'L':9.61e6,      #Kimura et al. 2002   
        'Pinf':3.13e10,  #Kimura et al. 2002
        'alpha':0.1      #arbitrary
    },
    # Pyroxene 60% iron
    #    - opacity in Budaj+2015 is calculated for Mg(0.4) Fe(0.6) SiO3
    #      we assume the sublimation properties from Kimura+2002 are valid for this composition, but note they correspond to SiO2 alone 
    'pyrmg40':{
        'nature':'dust',
        'rho': 3.0,      #Budaj+2015
        'Mg':119.3115,   #=0.4*24.305+0.6*55.845+28.0855+3.*15.999
        'L':5e6,     #Kimura et al. 2002
        'Pinf':3.13e10, #Kimura et al. 2002
        'alpha':0.1     #arbitrary
    }, 
    # Quartz (SiO2)
    #    - from Van Lieshout et al. 2014
    #    - no associated cross-section
    'quartz':{
        'nature':'dust',
        'rho':2.60,
        'Mg':60.084,
        'L':9609750.740329912, #=69444.*k_boltz/(60.084*amu)
        'Pinf':2.3721784213e13,#=0.1*np.exp(33.1)
        'alpha':1.0
    }, 
    # Silicon carbide (SiC)
    #    - from Van Lieshout et al. 2014
    #    - no associated cross-section
    'SiC_dust':{
        'nature':'dust',
        'rho':3.22,
        'Mg':40.10,
        'L':16268639.420874726, #=78462.*k_boltz/(40.10*amu)
        'Pinf':2.60814309975e15,#=0.1*np.exp(37.8)
        'alpha':0.1
    }, 
    # Silicon monoxide (SiO)
    #    - from Van Lieshout et al. 2014
    #    - no associated cross-section
    'SiO_dust':{
        'nature':'dust',
        'rho':2.13,
        'Mg':44.085,
        'L':9339551.536356324,  #=49520.*k_boltz/(44.085*amu)
        'Pinf':1.30187912050e13,#=0.1*np.exp(32.5)
        'alpha':0.04
    },
    # Water Ice
    #    - from Budaj+2015
    'waterice':{
        'nature':'dust',
        'rho': 1.0 ,
        'Mg':np.nan,
        'L':np.nan,
        'Pinf':np.nan,
        'alpha':0.1 #arbitrary
    },
    # Water Liquid
    #    - from Budaj+2015
    'waterliq':{
        'nature':'dust',
        'rho': 1.0 ,
        'Mg':np.nan,
        'L':np.nan,
        'Pinf':np.nan,
        'alpha':0.1 #arbitrary
    },

} 

#Add atomic mass in kg
for elem in dico_all_gases:
    if ('m_at' in dico_all_gases[elem]):dico_all_gases[elem]['mu_kg']=dico_all_gases[elem]['m_at']*amu





#################################################################
#Radiative processes
#    - each process must be uniquely associated to a given species, as indicated in the field 'species'
#    - each process is given a field 'dependance' to indicate if it depends on the local properties of the species (temperature, velocity, density, pressure) or not
#################################################################

dico_all_processes={
    
    #################################################################
    #Electronic transitions
    #    - f_os : oscillator strentgh (for radiation pressure coefficient)
    #    - delta_damping : damping wings parameter (Gamma/4 pi) in s-1
    #    - w0 : stellar line central wavelength (w in A)
    # beware of using vacuum (w0_V) or air(w0_A) reference for the transition wavelengths, depending on whether we adjust space-borne (generally vacuum-defined) or ground-based  (generally vacuum-defined) data 
    # it is possible to convert from one reference to the other, but beware that the difference vacuum - air is wavelength-dependent)
    # it is easier to select vacuum or air mode in settings, depending on the case 
    #    - use 'd' instead of a dot in transition wavelengths to avoid name issues
    #################################################################

    #Carbon 2+
    #    - see http://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=C+III&low_wl=1174&upp_wn=&upp_wl=1177&low_wn=&unit=0&submit=Retrieve+Data&de=0&java_window=3&java_mult=&format=1&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&order_out=0&max_low_enrg=&show_av=3&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on
    'CIII_1174d9':{'type':'electrans','species':'C_2+_g','w0_V':1174.93,'w0_A':None,'f_os':1.136e-01,'delta_damping':3.293e+08/(4.*pi),'dependance':True}, 
    'CIII_1175d3':{'type':'electrans','species':'C_2+_g','w0_V':1175.26,'w0_A':None,'f_os':2.724e-01,'delta_damping':4.385e+08/(4.*pi),'dependance':True}, 
    'CIII_1175d59':{'type':'electrans','species':'C_2+_g','w0_V':1175.59,'w0_A':None,'f_os':6.810e-02,'delta_damping':3.287e+08 /(4.*pi),'dependance':True}, 
    'CIII_1175d71':{'type':'electrans','species':'C_2+_g','w0_V':1175.71,'w0_A':None,'f_os':2.042e-01,'delta_damping':9.856e+08/(4.*pi),'dependance':True}, 
    'CIII_1175d99':{'type':'electrans','species':'C_2+_g','w0_V':1175.99,'w0_A':None,'f_os':9.074e-02,'delta_damping':1.313e+09/(4.*pi),'dependance':True}, 
    'CIII_1176d37':{'type':'electrans','species':'C_2+_g','w0_V':1176.37,'w0_A':None,'f_os':6.807e-02,'delta_damping':5.468e+08/(4.*pi),'dependance':True}, 

    #Neutral deuterium  
    #    - values from atomic data   
    'DI':{'type':'electrans','species':'D_0+_g','w0_V':1215.3394,'w0_A':None,'f_os':0.416,'delta_damping':0.627e9/(4.*pi),'dependance':True}, 

    #Silicon +
    #    - see atomic data and NIST
    'SiII_1197':{'type':'electrans','species':'Si_1+_g','w0_V':1197.3938,'w0_A':None,'f_os':1.50e-01,'delta_damping':1.40e+09/(4.*pi),'dependance':True}, 

    #Silicon 2+
    #    - see https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=Si+III&limits_type=0&low_w=1205&upp_w=1216&unit=0&submit=Retrieve+Data&de=0&format=0&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on
    'SiIII_1206':{'type':'electrans','species':'Si_2+_g','w0_V':1206.5,'w0_A':None,'f_os':1.67,'delta_damping':2.55E+09 /(4.*pi),'dependance':True}, 

    #Lyman-alpha line
    #    - see https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=H+I+&limits_type=0&low_w=1210&upp_w=1220&unit=0&de=0&format=0&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data for H I in vacuum
    'Lalpha':{'type':'electrans','species':'H_0+_g','w0_V':1215.6702,'w0_A':None,'f_os':0.41641,'delta_damping':0.627e9/(4.*pi),'dependance':True},
 
    #Oxygen 4+
    #    - see https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=O+V&limits_type=0&low_w=1217&upp_w=1220&unit=0&submit=Retrieve+Data&de=0&format=0&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on
    'OV_1218':{'type':'electrans','species':'O_4+_g','w0_V':1218.344,'w0_A':None,'f_os':1.56e-06,'delta_damping':2.34e+03/(4.*pi),'dependance':True}, 
    
    #Iron 11+
    #    - see https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=Fe+XII&limits_type=0&low_w=1240&upp_w=1244&unit=0&de=0&format=0&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=3&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data
    'FeXII_1241':{'type':'electrans','species':'Fe_11+_g','w0_V':1242.,'w0_A':None,'f_os':0.,'delta_damping':3.17e2 /(4.*pi),'dependance':True}, 
        
    #Nitrogen 4+
    #    - see http://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=N+V&low_wl=1238&upp_wn=&upp_wl=1244&low_wn=&unit=0&de=0&java_window=3&java_mult=&format=0&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&order_out=0&max_low_enrg=&show_av=3&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data
    'NV_1239':{'type':'electrans','species':'N_4+_g','w0_V':1238.821,'w0_A':None,'f_os':0.156,'delta_damping':0.340E+09 /(4.*pi),'dependance':True}, 
    'NV_1243':{'type':'electrans','species':'N_4+_g','w0_V':1242.804,'w0_A':None,'f_os':7.8e-2,'delta_damping':0.337E+09/(4.*pi),'dependance':True}, 

    #Nitrogen 0+
    'NI_1243':{'type':'electrans','species':'N_0+_g','w0_V':1243.179,'w0_A':None,'f_os':7.47e-02,'delta_damping':3.22e+08/(4.*pi),'dependance':True}, 

    #Carbon +
    #    - see http://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=C+II&low_wl=1334&upp_wn=&upp_wl=1336&low_wn=&unit=0&submit=Retrieve+Data&de=0&java_window=3&java_mult=&format=0&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&order_out=0&max_low_enrg=&show_av=3&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on
    'CII_1335':{'type':'electrans','species':'C_1+_g','w0_V':1334.5323,'w0_A':None,'f_os':0.128,'delta_damping':0.240E+09 /(4.*pi),'dependance':True}, 
    'CII_1336':{'type':'electrans','species':'C_1+_g','w0_V':1335.7080,'w0_A':None,'f_os':0.115,'delta_damping':0.288E+09/(4.*pi),'dependance':True}, 
                
    #Silicon 3+
    #    - see https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=Si+IV&limits_type=0&low_w=1391&upp_w=1404&unit=0&de=0&format=0&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=1&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data
    'SiIV_1393':{'type':'electrans','species':'Si_3+_g','w0_V':1393.755,'w0_A':None,'f_os':0.513,'delta_damping':0.880E+09 /(4.*pi),'dependance':True}, 
    'SiIV_1402':{'type':'electrans','species':'Si_3+_g','w0_V':1402.770,'w0_A':None,'f_os':0.255,'delta_damping':0.863E+09/(4.*pi),'dependance':True}, 
    
    #UV doublet of ionized magnesium  
    'MgIIk':{'type':'electrans','species':'Mg_1+_g','w0_V':2796.3518,'w0_A':None,'f_os':0.615,'delta_damping':0.262e9/(4.*pi),'dependance':True}, 
    'MgIIh':{'type':'electrans','species':'Mg_1+_g','w0_V':2803.5305,'w0_A':None,'f_os':0.306,'delta_damping':0.259e9/(4.*pi),'dependance':True}, 

    #UV line of neutral magnesium      
    'MgI_2852':{'type':'electrans','species':'Mg_0+_g','w0_V':2852.9641,'w0_A':None,'f_os':1.83,'delta_damping':0.5e9/(4.*pi),'dependance':True}, 
        
    #Calcium + 
    'CaII_3933':{'type':'electrans','species':'Ca_1+_g','w0_V':3934.777,'w0_A':3933.663,'f_os':6.82e-01,'delta_damping':1.47e+08/(4.*pi),'dependance':True}, #CaIIk
    'CaII_3968':{'type':'electrans','species':'Ca_1+_g','w0_V':3969.591,'w0_A':3968.469,'f_os':3.3e-01,'delta_damping':1.4e+08/(4.*pi),'dependance':True},   #CaIIh
    
    #Visible line of neutral magnesium  
    'MgI_4572':{'type':'electrans','species':'Mg_0+_g','w0_V':4572.3767,'w0_A':None,'f_os':0.205e-5,'delta_damping':0.218e3/(4.*pi),'dependance':True}, 

    #Visible doublet of neutral sodium  
    'NaID2':{'type':'electrans','species':'Na_0+_g','w0_V':5891.583253,'w0_A':5889.95094,'f_os':0.641,'delta_damping':0.616E+08/(4.*pi),'dependance':True}, 
    'NaID1':{'type':'electrans','species':'Na_0+_g','w0_V':5897.558147,'w0_A':5895.92424,'f_os':0.320,'delta_damping':0.614E+08/(4.*pi),'dependance':True}, 
    
    #Iron lines
    #    - oscillator strengths and damping coefficient unknown are set to arbitrary values for the need of the calculation
    'FeI_5893d5':{'type':'electrans','species':'Fe_0+_g','w0_V':5893.5097,'w0_A':5891.8782,'f_os':0.1,'delta_damping':0.,'dependance':True}, 
    'FeI_5899d9':{'type':'electrans','species':'Fe_0+_g','w0_V':5899.8502,'w0_A':5898.2149,'f_os':0.1,'delta_damping':0.,'dependance':True}, 
   
    #Visible doublet of neutral potassium 
    #    - see https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=K+I&limits_type=0&low_w=7000&upp_w=8000&unit=0&de=0&java_window=3&java_mult=&format=0&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=3&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&f_out=on&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on&submit=Retrieve+Data
    'KID2':{'type':'electrans','species':'K_0+_g','w0_V':7667.008906,'w0_A':None,'f_os':0.670,'delta_damping':0.380E+08/(4.*pi),'dependance':True}, 
    'KID1':{'type':'electrans','species':'K_0+_g','w0_V':7701.083536,'w0_A':None,'f_os':0.333,'delta_damping':0.375e+08 /(4.*pi),'dependance':True}, 

    #Infrared triplet of neutral metastable Helium  
    'HeI_10832d1':{'type':'electrans','species':'He_0+_23S1','w0_V':10832.057472,'w0_A':10829.09114,'f_os':5.9902e-02,'delta_damping':1.0216e+07/(4.*pi),'dependance':True}, 
    'HeI_10833d2':{'type':'electrans','species':'He_0+_23S1','w0_V':10833.216751,'w0_A':10830.25010,'f_os':1.7974e-01,'delta_damping':1.0216e+07/(4.*pi),'dependance':True}, 
    'HeI_10833d3':{'type':'electrans','species':'He_0+_23S1','w0_V':10833.306444,'w0_A':10830.33977,'f_os':2.9958e-01,'delta_damping':1.0216e+07/(4.*pi),'dependance':True}, 
         
    #################################################################
    #Broadband processes
    #################################################################

    #Rayleigh scattering from H2
    'H2_Rayleigh':{'type':'broadband','species':'H2_0+_g','dependance':False},
    
    #Rayleigh scattering from He
    'He_Rayleigh':{'type':'broadband','species':'He_0+_g','dependance':False},

    #Hydrogen photoionization
    'H0_XUV':{'type':'broadband','species':'H_0+_g','dependance':False},


   
}  
 
