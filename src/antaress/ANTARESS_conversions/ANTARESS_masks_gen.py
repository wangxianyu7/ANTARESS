#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from astropy.io import fits 
from copy import deepcopy
import os as os_system 
import bindensity as bind
from ..ANTARESS_general.utils import stop,np_where1D,dataload_npz,datasave_npz,air_index,gen_specdopshift





def def_masks(vis_mode,gen_dic,data_type_gen,inst,vis,data_dic,plot_dic,system_param,data_prop):
    r"""**CCF mask generation**

    Generates CCF binary masks from processed stellar spectrum. 
    
    2D spectra must have been aligned in the star (for disk-integrated profiles), photosphere (for intrinsic profiles), or planet (for atmospheric profiles) rest frame, converted into 1D profiles, and binned into a master spectrum.  
    This alignment in the rest frame of the line transitions is necessary to cross-match them with linelists
    
    The derived masks are thus defined in the approximate rest frame defined above   
     - Disk-integrated masks are shifted from the star rest frame to the rest frame of the input data, based on the input systemic velocity used to align the disk-integrated profiles.
       The masks can then be used to generate CCFs from disk-integrated spectra in their input rest frame (where CCFs will be centered at the approximate systemic rv + Keplerian rv)
     - Intrinsic masks are left defined in the photosphere rest frame.  
       They can be used to generate CCFs from intrinsic spectra in the star (where CCFs will be centered at the photosphere + line rv) or photosphere rest frame (where CCFs will be centered at the line rv) 
     - Atmospheric masks are left defined in the planet rest frame.  
       They can be used to generate CCFs from atmospheric spectra in the star (where CCFs will be centered at the orbital + atmospheric rv) or planet rest frame (where CCFs will be centered at the atmospheric rv) 

    Args:
        TBD

    Returns:
        None
    """    
    
    data_inst = data_dic[inst]
    print('   > Defining CCF mask for '+gen_dic['type_name'][data_type_gen]+' profiles') 
    if data_inst['type']!='spec1D':stop('Spectra must be 1D')
    
    #Using master from several visits
    if vis_mode=='multivis':vis_det='binned'

    #Using master from single visit
    elif vis_mode=='':vis_det=vis    
    
    print('         Calculating data')
    save_data_paths = gen_dic['save_data_dir']+'CCF_masks_'+data_type_gen+'/'+gen_dic['add_txt_path'][data_type_gen]+'/'+inst+'_'+vis_det+'/'
    if not os_system.path.exists(save_data_paths):os_system.makedirs(save_data_paths)  
    prop_dic = deepcopy(data_dic[data_type_gen]) 
    mask_dic = prop_dic['mask']

    #Retrieve binning information
    data_bin = dataload_npz(gen_dic['save_data_dir']+data_type_gen+'bin_data/'+gen_dic['add_txt_path'][data_type_gen]+inst+'_'+vis_det+'_'+prop_dic['dim_bin']+'_add')

    #Retrieve master spectrum
    if data_bin['n_exp']>1:stop('Bin data into a single master spectrum')
    data_mast = dataload_npz(gen_dic['save_data_dir']+data_type_gen+'bin_data/'+gen_dic['add_txt_path'][data_type_gen]+inst+'_'+vis_det+'_'+prop_dic['dim_bin']+str(0))

    #Check for alignment
    if (not gen_dic['align_'+data_type_gen]) or ((data_type_gen=='DI') and (not gen_dic['align_DI'] and data_bin['sysvel']==0.)):
        stop('Data must have been aligned in the stellar rest frame')
        
    #Check for in-transit contamination
    if (data_type_gen=='DI') and data_bin['in_inbin']:
        stop('Disk-integrated master contain in-transit profiles')
        
    #CCF masks for disk-integrated and intrinsic spectra
    dic_sav = {}
    if (data_type_gen in ['DI','Intr']):
        if data_dic['DI']['mask']['verbose']:print('           Continuum-normalization')

        #Limit master to minimum definition range
        idx_def_mast = np_where1D(data_mast['cond_def'][0])
        flux_mast = data_mast['flux'][:,idx_def_mast[0]:idx_def_mast[-1]+1]
        cen_bins_mast = data_mast['cen_bins'][:,idx_def_mast[0]:idx_def_mast[-1]+1]
        cond_def_mast = data_mast['cond_def'][:,idx_def_mast[0]:idx_def_mast[-1]+1]
        edge_bins_mast = data_mast['edge_bins'][:,idx_def_mast[0]:idx_def_mast[-1]+2]
        nspec = len(flux_mast[0])
        
        #Continuum-normalisation
        cont_func_dic = dataload_npz(gen_dic['save_data_dir']+'Stellar_cont_DI/'+inst+'_'+vis_det+'/St_cont')['cont_func_dic']
        flux_mast_norm = flux_mast[0]
        flux_mast_norm[~cond_def_mast[0]] = 0.
        flux_mast_norm[cond_def_mast[0]] /=cont_func_dic(cen_bins_mast[0,cond_def_mast[0]])

        #---------------------------------------------------------------------------------------------------------------------
        #Telluric contamination
        if data_dic['DI']['mask']['verbose']:print('           Mean telluric spectrum')
        tell_spec = np.zeros(nspec,dtype=float) if data_inst['tell_sp'] else None
        specdopshift_receiver_Earth_inbin = []
        nexp_in_bin = np.zeros(nspec,dtype=float) 
        for vis_bin in data_bin['vis_iexp_in_bin']:
            
            #Retrieve mean RV shifts used to align data
            data_align_comp = dataload_npz(gen_dic['save_data_dir']+'Aligned_DI_data/'+inst+'_'+vis_bin+'__add')

            #Define mean telluric spectrum over all exposures used in the master stellar spectrum
            #    - iexp is relative to global or in-transit indexes depending on data_type                
            for iexp in data_bin['vis_iexp_in_bin'][vis_bin]:
                if data_type_gen=='Intr':iexp_orig = gen_dic[inst][vis_bin]['idx_in'][iexp]
                else:iexp_orig = iexp

                #Align and resample telluric spectra
                #    - see gen_specdopshift():
                # w_source = w_receiver / (1+ (rv[s/r]/c))       
                #      disk-integrated spectra used in the binning were aligned in the star rest frame, so we shift their telluric spectrum back to the Earth rest frame as
                # w(tell/earth) = w(tell/star) / ( (1+ (rv[Earth/solbar]/c)) * (1+ (rv[solbar/starbar]/c)) * (1+ (rv[starbar/star]/c)) )
                #               = w(tell/star) / ( (1+ (BERV/c)) * (1- (rv_sys/c)) * (1- (rv_kep/c)) )    
                #      intrinsic spectra used in the binning were aligned in the photosphere rest frame, so we shift their telluric spectrum back to the Earth rest frame as
                # w(tell/earth) = w(tell/star) / ( (1+ (BERV/c)) * (1- (rv_sys/c)) * (1- (rv_kep/c)) * (1+ (rv[star/photo]/c) ) 
                #               = w(tell/star) / ( (1+ (BERV/c)) * (1- (rv_sys/c)) * (1- (rv_kep/c)) * (1- (rv_surf/c)) )              
                #    - the resulting master telluric spectrum is aligned in the Earth rest frame  
                specdopshift_earth_receiver = 1./(gen_specdopshift(data_prop[inst][vis_bin]['BERV'][iexp_orig])*(1.+1.55e-8)*gen_specdopshift(-data_align_comp['rv_starbar_solbar'])*gen_specdopshift(-data_align_comp['star_starbar'][iexp_orig]))
                if (data_type_gen=='Intr'):specdopshift_earth_receiver *= 1./gen_specdopshift(-data_align_comp['surf_star'][iexp])
                specdopshift_receiver_Earth_inbin+=[1./specdopshift_earth_receiver]
                if data_inst['tell_sp']:
                    
                    #Retrieve the 1D telluric spectrum associated with the exposure
                    #    - we retrieve the spectrum associated with the 1D exposure before it was binned
                    data_exp = dataload_npz(data_bin['vis_iexp_in_bin'][vis_bin][iexp]['data_path'])
                    cond_def_exp = data_exp['cond_def'][0]
                    tell_exp = dataload_npz(data_bin['vis_iexp_in_bin'][vis_bin][iexp]['tell_path'])['tell'][0]      
                    tell_exp[~cond_def_exp] = np.nan
                    
                    #Shifting telluric spectrum back from receiver (star or photosphere) back to Earth (source) rest frame
                    edge_bins_earth=data_exp['edge_bins'][0]*specdopshift_earth_receiver
                    tell_exp = bind.resampling(edge_bins_mast[0],edge_bins_earth,tell_exp, kind=gen_dic['resamp_mode'])
                
                    #Co-add
                    cond_def_tell = ~np.isnan(tell_exp)
                    tell_spec[cond_def_tell]+=tell_exp[cond_def_tell]
                    nexp_in_bin[cond_def_tell]+=1.

        #Average telluric spectrum
        if data_inst['tell_sp']:
            cond_def_tell = nexp_in_bin>0.
            tell_spec[cond_def_tell]/=nexp_in_bin[cond_def_tell]
            tell_spec[~cond_def_tell] = 1.
            
        #Min/max Doppler shift of tellurics in the rest frame of the stellar master spectrum 
        min_specdopshift_receiver_Earth = np.min(specdopshift_receiver_Earth_inbin)
        max_specdopshift_receiver_Earth = np.max(specdopshift_receiver_Earth_inbin)

        #Mask generation
        #    - defined in the stellar (for disk-integrated profiles) or surface (for intrinsic profiles) rest frames
        from ANTARESS_conversions.KitCat import KitCat_main
        mask_waves,mask_weights,mask_info = KitCat_main.kitcat_mask(mask_dic,mask_dic['fwhm_ccf'],cen_bins_mast[0],inst,edge_bins_mast[0],flux_mast_norm,gen_dic,save_data_paths,tell_spec,data_dic['DI']['sysvel'][inst][vis_bin],min_specdopshift_receiver_Earth,
                                                        max_specdopshift_receiver_Earth,dic_sav,plot_dic[data_type_gen+'mask_spectra'],plot_dic[data_type_gen+'mask_ld'],plot_dic[data_type_gen+'mask_ld_lw'],plot_dic[data_type_gen+'mask_RVdev_fit'],cont_func_dic,
                                                        data_bin['vis_iexp_in_bin'],data_type_gen,data_dic,plot_dic[data_type_gen+'mask_tellcont'],plot_dic[data_type_gen+'mask_vald_depthcorr'],plot_dic[data_type_gen+'mask_morphasym'],
                                                        plot_dic[data_type_gen+'mask_morphshape'],plot_dic[data_type_gen+'mask_RVdisp'])

        #Save mask in format readable by ESPRESSO-like DRSs
        #    - the convention for those DRS is that masks are defined in air
        if gen_dic['sp_frame']=='vacuum':
            n_refr=air_index(mask_waves, t=15., p=760.)
            DRS_waves=mask_waves/n_refr
        else:DRS_waves = mask_waves
    
        #Create data columns
        c1 = fits.Column(name='lambda', array=DRS_waves, format='1D')
        c2 = fits.Column(name='contrast', array=mask_weights, format='1E')
        t = fits.BinTableHDU.from_columns([c1, c2])
        
        #Create header columns
        hdr = fits.Header()
        hdr['HIERARCH ESO PRO CATG'] = 'MASK_TABLE'
        inst_ref = inst.split('_')[0]
        hdr['INSTRUME'] = inst_ref
        p = fits.PrimaryHDU(header=hdr)
        
        #Saving
        hdulist = fits.HDUList([p,t])
        mask_name = 'CCF_mask_'+data_type_gen+'_'+gen_dic['star_name']+'_'+inst_ref+mask_info
        hdulist.writeto(save_data_paths+mask_name+'.fits',overwrite=True,output_verify='ignore')
        np.savetxt(save_data_paths+mask_name+'_'+gen_dic['sp_frame']+'.txt', np.column_stack((mask_waves,mask_weights)),fmt=('%15.10f','%15.10f') )

    else:
        flux_mast_norm = flux_mast[0]

        stop('Code Atmospheric mask routine based on expected species in the planet')

    #Save for plotting
    if (plot_dic[data_type_gen+'mask_spectra']!='') | (plot_dic[data_type_gen+'mask_ld_lw']!='') | (plot_dic[data_type_gen+'mask_RVdev_fit']!=''):
        datasave_npz(save_data_paths+'Plot_info',dic_sav)

    return None



