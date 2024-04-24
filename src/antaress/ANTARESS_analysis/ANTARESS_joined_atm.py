#!/usr/bin/env python3
# -*- coding: utf-8 -*-




'''
Wrap-up function to process atmospheric profiles
'''
def joined_Atm_ana(gen_dic):

    #Fitting intrinsic stellar CCFs with joined model
    if gen_dic['fit_AtmProf']:
        fit_CCFatm_all()    
        
    #Fitting stellar surface properties with a linked model
    if gen_dic['fit_AtmProp']:
        fit_CCFatm_prop()    
    

    return None




'''
Routine to fit atmospheric CCFs with a joined model
    - we use simple analytical profiles to describe the CCFs
      their shape and position are linked using polynomial laws as a function of phase
    - contrast and FWHM correspond to the atmospheric line before instrumental convolution, so that data from all instruments and visits can be fitted together
    - polynomial coefficients can nonetheless depend on the visit and their associated instrument, to account for possible variations in the line shape between visits
'''
def fit_CCFatm_all():
    print('   > Fitting atmospheric CCFs with linked model')


    return None







'''
Routine to fit atmospheric CCF properties with a common model
'''
def fit_CCFatm_prop():
    print('   > Fitting atmospheric properties with linked model')


    return None














    