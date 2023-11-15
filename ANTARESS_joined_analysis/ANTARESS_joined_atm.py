#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import np_where1D,dataload_npz
from ANTARESS_all_routines import sub_calc_plocc_prop,return_FWHM_inst,convol_prof,def_st_prof_tab,conv_st_prof_tab,cond_conv_st_prof_tab,resamp_model_st_prof_tab,gen_theo_intr_prof,\
                            compute_deviation_profile, calc_binned_prof,init_custom_DI_prof,ref_inst_convol
from copy import deepcopy
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from ANTARESS_joined_analysis.ANTARESS_joined_comm import init_joined_routines,init_joined_routines_inst,init_joined_routines_vis,init_joined_routines_vis_fit,common_fit_rout,post_proc_func,calc_plocc_coord


'''
Wrap-up function to process atmospheric profiles
'''
def fit_atm_funcs(gen_dic):

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














    