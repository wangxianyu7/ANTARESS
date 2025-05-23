#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
import bindensity as bind
from ..ANTARESS_general.utils import dup_edges

#NB: Kept in an independent file to prevent issues of circular import 
def align_data(data_exp,rout_mode,nord,dim_exp_resamp,resamp_mode,cen_bins_resamp, edge_bins_resamp,rv_shift_cen,spec_dopshift_cen,rv_shift_edge = None ,spec_dopshift_edge = None , nocov = False):
    r"""**Aligning routine.**
    
    Shifts spectral or CCF profiles, and associated tables, with input rv values. 

    Args:
        TBD
    
    Returns:
        TBD
        
    """
    #Shift wavelength tables to a common rest frame
    #    - we apply the general Doppler effect formula (see gen_specdopshift())
    # w_receiver = w_source * (1+ (rv[s/r]/c))
    # w_source = w_receiver / (1+ (rv[s/r]/c))
    #      where rv[s/r] < 0 if the source is moving toward the receiver
    #
    #    - for disk-integrated spectra the source is the star, and the receiver the solar system barycenter (we assume the motion of the Earth with respect to the solar system barycenter has already been corrected)
    #      disk-integrated spectra are aligned in the star rest frame by correcting for the systemic velocity (motion of the stellar system barycenter relative to the solar system barycenter), and
    # the Keplerian motion of the star (motion of the star relative to the stellar system barycenter)
    # w_solbar = w_star * (1+ (rv[star/star bary]/c)) * (1+ (rv[star bary/sun bary]/c))  
    #          = w_star * (1+ (rv_kep/c)) * (1+ (rv_sys/c))   
    # w_star = w_solbar / ((1+ (rv_kep/c)) * (1+ (rv_sys/c))  
    #
    #    - for intrinsic spectra the source is the photosphere, and the receiver the star (this is the frame where they are extracted)
    #      intrinsic spectra are aligned in the photosphere rest frame by correcting for the motion of the stellar surface relative to the star rest velocity
    # w_star = w_surf * (1+ rv[surf/star]/c))
    # w_surf = w_star / (1+ rv[surf/star]/c))
    #
    #    - for out-of-transit profiles or aligned intrinsic profiles used as estimates for a given local profile the source can be seen as the photosphere and the receiver the star (this is the frame where they are defined)
    #      local spectra are shifted back to the star rest frame by re-injecting the motion of the photosphere relative to the star rest velocity
    # w_surf = w_star * (1+ rv[star/surf]/c))    
    # w_star = w_surf / (1+ rv[star/surf]/c))   
    #
    #    - for atmospheric spectra the source is the planet, and the receiver the star (this is the frame where they are extracted)   
    #      atmospheric spectra are aligned in the planet rest frame by correcting for the motion of the planet relative to the star rest velocity
    # w_star = w_pl * (1 + rv[pl/star]/c))  
    # w_pl = w_star / (1 + rv[pl/star]/c)) 
    if ('spec' in rout_mode):
    
        #Achromatic shift
        if (rv_shift_edge is None):
            edge_bins_rest = data_exp['edge_bins']*spec_dopshift_cen
            cen_bins_rest = data_exp['cen_bins']*spec_dopshift_cen                         
         
        #Chromatic shift
        #    - in this case the bin edges and center are shifted using the rv calculated at their exact wavelength, to keep the new bins contiguous
        else:
            edge_bins_rest = data_exp['edge_bins']*spec_dopshift_edge  
            cen_bins_rest = data_exp['cen_bins']*spec_dopshift_cen    

    #Shift velocity tables to chosen rest frame
    #    - for disk-integrated data: RV(M/star) = RV(M/sun) - RV(CDM_star/sun) - RV(star/CDM_star)
    #    - for intrinsic data:       RV(M/region) = RV(M/star) - RV(region/star)
    #    - for master out data:      RV(M/star) = RV(M/region) + RV(region/star)
    #    - for atmospheric data:     RV(M/pl) = RV(M/star) - RV(pl/star)
    elif (rout_mode=='CCF'):
        edge_bins_rest = data_exp['edge_bins'] - rv_shift_cen
        cen_bins_rest = data_exp['cen_bins'] - rv_shift_cen        
        
    #----------------------------------------------------------------

    #Port data from previous processing
    data_align = deepcopy(data_exp) 
    
    #Data is resampled on the common table given as input
    #    - for spectra we neglect the few bins lost by resampling the data (defined on tables shifted to the star rest frame) on the common table defined in the input rest frame
    if cen_bins_resamp is not None:

        #Initialize aligned data
        data_align['edge_bins']=deepcopy(edge_bins_resamp)
        data_align['cen_bins']=deepcopy(cen_bins_resamp) 
        data_align['flux']=np.zeros(dim_exp_resamp, dtype=float)*np.nan
      
        #Aligning each order
        if nocov:
            for iord in range(nord):
                data_align['flux'][iord] = bind.resampling(data_align['edge_bins'][iord], edge_bins_rest[iord], data_exp['flux'][iord], kind=resamp_mode)            
        else:
            data_align['cov']=np.zeros(nord, dtype=object)          
            for iord in range(nord):
                data_align['flux'][iord],data_align['cov'][iord] = bind.resampling(data_align['edge_bins'][iord], edge_bins_rest[iord], data_exp['flux'][iord] , cov = data_exp['cov'][iord], kind=resamp_mode) 

        #Defined bins
        data_align['cond_def'] = ~np.isnan(data_align['flux'])
       
        #Processing each order
        key2proc = []
        if ('tell' in data_exp):        key2proc+=['tell']              #Telluric spectrum     
        if ('mean_gcal' in data_exp):   key2proc+=['mean_gcal']         #Scaling calibration profile
        if ('sing_gcal' in data_exp):   key2proc+=['sing_gcal']         #Weighing calibration profile          
        if ('sdet2' in data_exp):       key2proc+=['sdet2']             #Weighing detector noise 
        for key_var in ['EFsc2','EFdiff2','EFintr2','EFem2','EAbs2']:   #1D variance grids
            if (key_var in data_exp):   key2proc+=[key_var] 
        for key in key2proc:
            data_align[key]=np.zeros(dim_exp_resamp, dtype=float)*np.nan
            for iord in range(nord):
                
                #Shifting and resampling
                data_align[key][iord] = bind.resampling(data_align['edge_bins'][iord], edge_bins_rest[iord], data_exp[key][iord], kind=resamp_mode)            
                
                #Keeping profiles fully defined
                #    - edge values are filled with latest defined value
                data_align[key][iord] = dup_edges(data_align[key][iord])
                
    #Data remain defined on independent tables for each exposure
    #    - we do not resample the data tables of each exposure and keep them defined on their shifted spectral tables
    else:      
        data_align['cen_bins'] = cen_bins_rest   
        data_align['edge_bins'] = edge_bins_rest                

    return data_align




