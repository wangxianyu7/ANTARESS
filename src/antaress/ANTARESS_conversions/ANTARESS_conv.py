#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
import bindensity as bind
from scipy.interpolate import interp1d
from pathos.multiprocessing import Pool
from ..ANTARESS_general.constant_data import c_light
from ..ANTARESS_conversions.ANTARESS_binning import pre_calc_bin_prof,weights_bin_prof,weights_bin_prof_calc
from ..ANTARESS_grids.ANTARESS_coord import excl_plrange
from ..ANTARESS_process.ANTARESS_data_process import calc_Intr_mean_cont
from ..ANTARESS_corrections.ANTARESS_detrend import corr_length_determination
from ..ANTARESS_general.utils import stop,dataload_npz,datasave_npz,MAIN_multithread,np_where1D,init_parallel_func,gen_specdopshift,check_data,dup_edges


def init_conversion(data_type_gen,gen_dic,prop_dic,inst,vis,mode,dir_save,data_dic):
    r"""**Initialization for conversion routines.**
    
    Initializes various parameters for conversion into CCF or S1D spectra. 
    
    Args:
        TBD
    
    Returns:
        None
    
    """     
    txt_print = {'CCFfromSpec':'CCF','1Dfrom2D':'1D'}[mode]
    if data_type_gen in ['DI','Diff','Intr']:    
        data_type = data_type_gen
        if data_type_gen=='Intr':
            iexp_conv = list(gen_dic[inst][vis]['idx_out'])+list(gen_dic[inst][vis]['idx_in'][prop_dic[inst][vis]['idx_def']])    #Global indexes
            data_type_key = ['Diff','Intr']
            print('   > Converting OT differential and intrinsic spectra into '+txt_print)
        else:
            if data_type_gen=='Diff':data_type_key = ['Diff']
            iexp_conv = range(data_dic[inst][vis]['n_in_visit'])    #Global indexes
    if data_type_gen in ['DI','Atm']:  
        if data_type_gen=='Atm': 
            data_type = prop_dic['pl_atm_sign']
            iexp_conv = prop_dic[inst][vis]['idx_def']    #Global or in-transit indexes
        data_type_key = [data_type_gen]
        print('   > Converting '+gen_dic['type_name'][data_type]+' spectra into '+txt_print)
    for key in data_type_key:dir_save[key] = gen_dic['save_data_dir']+key+'_data/'+mode+'/'+gen_dic['add_txt_path'][key]+'/'+inst+'_'+vis+'_'    
    
    return iexp_conv,data_type_key,data_type


def CCF_from_spec(data_type_gen,inst,vis,data_dic,gen_dic,prop_dic):
    r"""**Wrap-up function to compute CCFs** 
    
    Prepares spectral time-series for conversion into CCF, calling `new_compute_CCF()` 
    
    Args:
        TDB
    
    Returns:
        None
    
    """    
    data_vis=data_dic[inst][vis]
    gen_vis=gen_dic[inst][vis]
    dir_save = {}
    iexp_conv,data_type_key,_ = init_conversion(data_type_gen,gen_dic,prop_dic,inst,vis,'CCFfromSpec',dir_save,data_dic)

    #New paths
    #    - intrinsic and out-of-transit differential profiles are stored separately, contrary to global tables  
    flux_sc = False
    if data_type_gen in ['Intr','Atm']:
        dir_mast={}
        for gen in dir_save:dir_mast[gen] = {iexp_eff:dir_save[gen]+'ref_'+str(iexp_eff) for iexp_eff in data_vis['mast_'+gen+'_data_paths']}
        if gen_dic['flux_sc']:flux_sc = True
    if (data_type_gen=='DI') and (data_dic['DI'][inst][vis]['rest_frame']!='star'):com_frame = ''
    else:com_frame = '_star'
    proc_com_data_paths_new = gen_dic['save_data_dir']+'Processed_data/CCFfromSpec/'+inst+'_'+vis+'_com'+com_frame

    #Calculating data
    if gen_dic['calc_'+data_type_gen+'_CCF']:
        print('         Calculating data')
        if data_vis['type']=='CCF':stop('Data must be spectral')         

        #Orders that are used to compute the CCFs
        ord_coadd = gen_dic[inst]['orders4ccf'] 
        nord_coadd = len(ord_coadd)

        #Scaling calibration profile
        #    - always defined for echelle spectra
        #    - the effective calibration profile used here will be calculated on the common table, which is defined in the input rest frame if non-aligned, disk-integrated spectra are cross-correlated or in the star rest frame otherwise
        #    - if alignment in the star rest frame was not applied, the common star table points toward the common input table
        if data_vis['type']=='spec2D':
            data_com = dataload_npz(data_vis['proc_com_star_data_paths'])
            mean_gcal_com = np.zeros([nord_coadd,data_com['dim_exp'][1]],dtype=float)
        else:gcal_ord=None
    
        #Upload data from all exposures 
        if flux_sc:data_scaling_all={}
        data_proc={}
        n_exp = len(iexp_conv) 
        for iexp_sub,iexp in enumerate(iexp_conv):
            gen = deepcopy(data_type_gen)
            
            #Retrieving data
            #    - out-of-transit differential profiles are retrieved with global indexes
            #    - in-transit intrinsic profiles are retrieved with in-transit indexes
            iexp_eff = deepcopy(iexp)     
            if (gen=='Intr'):
                if (iexp in gen_vis['idx_in']):iexp_eff = gen_vis['idx_exp2in'][iexp]    
                else:gen = 'Diff'
            data_exp = dataload_npz(data_vis['proc_'+gen+'_data_paths']+str(iexp_eff))
         
            #Check that planetary ranges were not yet excluded
            if data_type_gen in ['Diff','Intr']:
                if (iexp in gen_vis['idx_in']) and (data_type_gen in data_dic['Atm']['no_plrange']) and (iexp in data_dic['Atm'][inst][vis]['iexp_no_plrange']) and data_exp['plrange_exc']:
                    stop('    Planetary ranges excluded too soon: re-run gen_dic["'+{'Diff':'diff','Intr':'intr'}[data_type_gen]+'_data"] with gen_dic["'+data_type_gen+'_CCF"]')
                
            #Upload data
            if iexp_sub==0:
                if flux_sc:data_proc['cen_bins'] = np.zeros([n_exp,nord_coadd,data_vis['nspec']],dtype=float)
                data_proc['edge_bins'] = np.zeros([n_exp,nord_coadd,data_vis['nspec']+1],dtype=float)
                data_proc['flux'] = np.zeros([n_exp,nord_coadd,data_vis['nspec']],dtype=float)
                data_proc['cov'] = np.zeros([n_exp,nord_coadd],dtype=object)
                data_proc['cond_def'] = np.zeros([n_exp,nord_coadd,data_vis['nspec']],dtype=bool)             
            for key in ['edge_bins','flux','cov','cond_def']:
                data_proc[key][iexp_sub] = data_exp[key][ord_coadd]  

            #Upload flux scaling
            #    - we compute the equivalent CCF of the broadband scaling for the propagation of broadband spectral scaling on disk-integrated profiles into weights
            #    - global flux scaling is not modified  
            #    - scaling profiles are retrieved with global indexes
            if flux_sc:
                data_proc['cen_bins'][iexp_sub] = data_exp['cen_bins'][ord_coadd]          
                data_scaling_all[iexp]=dataload_npz(data_vis['scaled_'+gen+'_data_paths']+str(iexp)) 

            #Mean calibration profile over processed exposures
            #    - due to the various shifts of the processed spectra from the input rest frame, calibration profiles are not equivalent for a given line between exposure
            #      to maintain the relative flux balance between lines when computing CCFs, we calculate an average calibration profile common to all processes exposures
            #    - we resample the profile, fill-in undefined edges, and co-add it to the average one
            if data_vis['type']=='spec2D':
                mean_gcal_exp = dataload_npz(data_vis['mean_gcal_'+gen+'_data_paths'][iexp_eff])['mean_gcal'] 
                for isub_ord,iord in enumerate(ord_coadd):
                    mean_gcal_com[isub_ord]+=dup_edges(bind.resampling(data_com['edge_bins'][iord], data_proc['edge_bins'][iexp_sub,isub_ord],mean_gcal_exp[iord], kind=gen_dic['resamp_mode'])/n_exp) 
                    
            #Upload weighing disk-integrated master 
            #    - the master always remains either defined on the common table, or on a specific table different from the table of its associated exposure
            #    - the master is computed after DI spectra have been converted into CCFs, and thus need conversion only for later profile types
            if data_type_gen in ['Diff','Intr','Atm']:
                data_ref = dataload_npz(data_vis['mast_'+gen+'_data_paths'][iexp_eff])
                if iexp_sub==0:
                    nspec_mast = (data_ref['cen_bins'].shape)[1]
                    data_proc['edge_bins_ref'] = np.zeros([n_exp,nord_coadd,nspec_mast+1],dtype=float)
                    data_proc['flux_ref'] = np.zeros([n_exp,nord_coadd,nspec_mast],dtype=float)
                    data_proc['cov_ref'] = np.zeros([n_exp,nord_coadd],dtype=object)                
                for key in ['edge_bins','flux','cov']:
                    data_proc[key+'_ref'][iexp_sub] = data_ref[key][ord_coadd]               

        #Initialize CCF tables
        CCF_all = np.zeros([n_exp,1,data_vis['nvel']],dtype=float)
        cov_exp_ord = np.zeros([n_exp,nord_coadd],dtype=object)
        nd_cov_exp_ord=np.zeros([n_exp,nord_coadd],dtype=int) 
        if data_type_gen in ['DI','Diff','Intr']:
            CCF_mask_wav = gen_dic['CCF_mask_wav'][inst]
            CCF_mask_wgt = gen_dic['CCF_mask_wgt'][inst]
        elif data_type_gen=='Atm':
            CCF_mask_wav = data_dic['Atm']['CCF_mask_wav']
            CCF_mask_wgt = data_dic['Atm']['CCF_mask_wgt']
        if data_type_gen in ['Diff','Intr','Atm']:
            CCF_ref = np.zeros([n_exp,1,data_vis['nvel']],dtype=float)
            cov_ref_ord = np.zeros([n_exp,nord_coadd],dtype=object)            
            nd_cov_ref_ord=np.zeros([n_exp,nord_coadd],dtype=int)
        
        #Velocity tables and initializations
        #    - we create an artificial order in velocity table to keep the same structure as spectra
        if data_type_gen=='DI':   
            velccf = data_vis['velccf']
            edge_velccf = data_vis['edge_velccf']          
        elif data_type_gen in ['Diff','Intr','Atm']:
            velccf = data_vis['velccf_star']
            edge_velccf = data_vis['edge_velccf_star']
        cen_bins = np.tile(velccf,[1,1])
        edge_bins = np.tile(edge_velccf,[1,1])  
        
        #Flux scaling
        if flux_sc: 
            norm_loc_flux_scaling_CCF = np.zeros(n_exp,dtype=float)                  
            loc_flux_scaling_CCF = np.zeros([n_exp,data_vis['nvel']],dtype=float)   
            
        #Calculate CCF over requested orders in each exposure
        #    - the covariance of CCFs calculated over different orders may have different dimensions, thus we first store them independently before they can be co-added
        #    - the structure below works for both s1d and e2ds
        ord_coadd_eff = []
        for isub,iord in enumerate(ord_coadd):
 
            #Identify lines that can contribute to all exposures for current order
            idx_maskL_kept = check_CCF_mask_lines(n_exp,data_proc['edge_bins'][:,isub],data_proc['cond_def'][:,isub],CCF_mask_wav,edge_velccf)

            #Calculating CCF for current order in each exposure with contributing lines
            #    - parallelisation is disabled, as it is inefficient given the size of the tables to process
            if len(idx_maskL_kept)>0:
                ord_coadd_eff+=[isub]
                if data_vis['type']=='spec2D':
                    gcal_ord = dup_edges(bind.resampling(data_proc['edge_bins'][iexp_sub,isub],data_com['edge_bins'][iord],mean_gcal_com[isub], kind=gen_dic['resamp_mode']))
                for iexp_sub,iexp in enumerate(iexp_conv):                      
                    flux_ord,cov_ord = new_compute_CCF(data_proc['edge_bins'][iexp_sub,isub],data_proc['flux'][iexp_sub,isub],data_proc['cov'][iexp_sub,isub],gen_dic['resamp_mode'],edge_velccf,CCF_mask_wgt[idx_maskL_kept],CCF_mask_wav[idx_maskL_kept],1.,cal = gcal_ord)[0:2]
                    CCF_all[iexp_sub,0]+=flux_ord
                    cov_exp_ord[iexp_sub,isub] = cov_ord 
                    nd_cov_exp_ord[iexp_sub,isub] = np.shape(cov_ord)[0]

                    #Compute CCF of spectral scaling
                    #    - loc_flux_scaling = 1 - LC
                    #      loc_flux_scaling_CCF = X - CCF_LC
                    #       we thus calculate the CCF of 1 to get X and normalize the scaling profile
                    if flux_sc and (not data_scaling_all[iexp]['null_loc_flux_scaling']):                
                        loc_flux_scaling_CCF_exp,_,norm_loc_flux_scaling_CCF_exp = new_compute_CCF(data_proc['edge_bins'][iexp_sub,isub],data_scaling_all[iexp]['loc_flux_scaling'](data_proc['cen_bins'][iexp_sub,isub]),None,gen_dic['resamp_mode'],edge_velccf,CCF_mask_wgt[idx_maskL_kept],CCF_mask_wav[idx_maskL_kept],1.,cal = gcal_ord)
                        loc_flux_scaling_CCF[iexp_sub] +=loc_flux_scaling_CCF_exp
                        norm_loc_flux_scaling_CCF[iexp_sub] += norm_loc_flux_scaling_CCF_exp

                    #Computing CCF of master disk-integrated spectrum
                    #    - so that it can be used in the weighing profiles
                    if data_type_gen in ['Diff','Intr','Atm']:
                        flux_temp,cov_temp = new_compute_CCF(data_proc['edge_bins_ref'][iexp_sub,isub],data_proc['flux_ref'][iexp_sub,isub],data_proc['cov_ref'][iexp_sub,isub],gen_dic['resamp_mode'],edge_velccf,CCF_mask_wgt[idx_maskL_kept],CCF_mask_wav[idx_maskL_kept],1.,cal = gcal_ord)[0:2]
                        CCF_ref[iexp_sub,0]+=flux_temp
                        cov_ref_ord[iexp_sub,isub] = cov_temp 
                        nd_cov_ref_ord[iexp_sub,isub] = np.shape(cov_temp)[0]

        #Check
        if len(ord_coadd_eff)==0:stop('         No lines contribute to all exposures')

        #Computing final covariance matrix in artificial new order
        for iexp_sub,iexp in enumerate(iexp_conv):
            gen = deepcopy(data_type_gen)
            iexp_eff = deepcopy(iexp)    #out- (global) or in-transit index
            cond_def_exp = (~np.isnan(CCF_all[iexp_sub,0]))[None,:] 
            data_CCF_exp={}
            if (data_type_gen=='Intr'):
                if (iexp in gen_vis['idx_in']):
                    iexp_eff = gen_vis['idx_exp2in'][iexp] 
                    
                    #Set to nan planetary ranges in intrinsic CCFs
                    #    - this is not done for disk-integrated and differential CCFs, as planetary signals need to be kept in them to be later extracted (planetary ranges are temporarily excluded when analyzing CCFs from those profiles)
                    if ('Intr' in data_dic['Atm']['no_plrange']) and (iexp in data_dic['Atm'][inst][vis]['iexp_no_plrange']):
                        cond_in_pl = ~( np.ones(data_vis['nvel'],dtype=bool) & excl_plrange(cond_def_exp[0],data_dic['Atm'][inst][vis]['exclu_range_star'],iexp,edge_bins[0],'CCF')[0])
                        CCF_all[iexp_sub,0,cond_in_pl]=np.nan
                        cond_def_exp[0,cond_in_pl]=False  
                        data_CCF_exp['plrange_exc'] = True
                    else:data_CCF_exp['plrange_exc'] = False
                        
                else:gen = 'Diff'

            #Maximum dimension of covariance matrix in current exposure from all contributing orders
            nd_cov_exp = np.amax(nd_cov_exp_ord[iexp_sub,:])
            if data_type_gen in ['Diff','Intr','Atm']:nd_cov_exp = np.max([nd_cov_exp,np.amax(nd_cov_ref_ord[iexp_sub,:])])  
           
            #Co-adding contributions from orders
            cov_exp = np.zeros(1,dtype=object)
            cov_exp[0] = np.zeros([nd_cov_exp,data_vis['nvel']])
            if data_type_gen in ['Diff','Intr','Atm']:
                cov_ref = np.zeros(1,dtype=object)
                cov_ref[0] = np.zeros([nd_cov_exp,data_vis['nvel']])
            for isub in ord_coadd_eff:
                cov_exp[0][0:nd_cov_exp_ord[iexp_sub,isub],:] +=  cov_exp_ord[iexp_sub,isub]   
                if data_type_gen in ['Diff','Intr','Atm']:cov_ref[0][0:nd_cov_ref_ord[iexp_sub,isub],:] +=  cov_ref_ord[iexp_sub,isub]   
      
            #Saving data for each exposure
            #    - CCF are stored independently of input spectra, so that both can be retrieved
            data_CCF_exp.update({'cen_bins':cen_bins,'edge_bins':edge_bins,'flux':CCF_all[iexp_sub],'cond_def':cond_def_exp,'cov':cov_exp,'nd_cov':nd_cov_exp})              
            datasave_npz(dir_save[gen]+str(iexp_eff),data_CCF_exp)
            
            #Processing disk-integrated masters
            if data_type_gen in ['Diff','Intr','Atm']:
                datasave_npz(dir_mast[gen][iexp_eff],{'cen_bins':cen_bins,'edge_bins':edge_bins,'flux':CCF_ref[iexp_sub],'cov':cov_ref})

            #Redefine spectral scaling table
            if flux_sc:
                
                #Normalizing scaling CCF
                #    - if 'null_loc_flux_scaling' then 'loc_flux_scaling_CCF' remains set to 0
                if (not data_scaling_all[iexp]['null_loc_flux_scaling']):loc_flux_scaling_CCF[iexp_sub,:]/=norm_loc_flux_scaling_CCF[iexp_sub]
                
                #Defining the scaling CCF as a function
                #    - if scaling light curves are chromtic we propagate the chromaticity into the scaling CCF
                if not data_scaling_all[iexp]['chrom']:loc_flux_scaling_exp = np.poly1d(np.mean(loc_flux_scaling_CCF[iexp_sub]))                
                else:loc_flux_scaling_exp = interp1d(cen_bins[0],loc_flux_scaling_CCF[iexp_sub],fill_value=(loc_flux_scaling_CCF[iexp_sub,0],loc_flux_scaling_CCF[iexp_sub,-1]), bounds_error=False)
                
                #Saving
                data_scaling_all[iexp]['loc_flux_scaling'] = loc_flux_scaling_exp
                data_scaling_all[iexp]['chrom'] = False
                datasave_npz(dir_save[gen]+'_scaling_'+str(iexp),data_scaling_all[iexp])

        #Update common table
        #    - set to the input table for disk-integrated profiles not aligned in the star rest frame, set to the star frame table otherwise
        cen_bins_com = data_vis['velccf_com'+com_frame]
        edge_bins_com = data_vis['edge_velccf_com'+com_frame]                                     
        datasave_npz(proc_com_data_paths_new,{'dim_exp':[1,len(cen_bins_com)],'nspec':len(cen_bins_com),'cen_bins':np.tile(cen_bins_com,[1,1]),'edge_bins':np.tile(edge_bins_com,[1,1])})

    else:
        check_data({'path':proc_com_data_paths_new})

    #Updating path to processed data and checking it has been calculated
    #    - if data is still in the input frame we update the path to the common grid in the star frame as well, or it will later still point toward the original spectral grid
    data_vis['proc_com'+com_frame+'_data_paths'] = proc_com_data_paths_new
    if com_frame=='':data_vis['proc_com_star_data_paths'] = deepcopy(data_vis['proc_com'+com_frame+'_data_paths'])
    for gen in dir_save:
        data_vis['proc_'+gen+'_data_paths'] = dir_save[gen]  
        if flux_sc:data_vis['scaled_'+gen+'_data_paths'] = dir_save[gen]+'_scaling_'
        if gen in ['Diff','Intr','Atm']:data_vis['mast_'+gen+'_data_paths'] = dir_mast[gen]

    #Convert spectral mode 
    #    - all operations afterwards will be performed on CCFs
    #    - tellurics are not propagated to calculate weights in CCF mode 
    #    - spectral calibration is no longer required because it is only needed to calculate DI variance, which is either known after DI conversion or not required if later conversions
    print('         ANTARESS switched to CCF processing')
    data_vis['comm_sp_tab']=True
    data_vis['type']='CCF'
    data_vis['nspec'] = data_vis['nvel']
    data_dic[inst]['nord'] = 1
    data_vis['dim_all'] = [data_vis['n_in_visit'],data_dic[inst]['nord'],data_vis['nspec']]
    data_vis['dim_exp'] = [data_dic[inst]['nord'],data_vis['nspec']]
    data_vis['dim_ord'] = [data_vis['n_in_visit'],data_vis['nspec']]
    if ('chrom' in data_vis['system_prop']):
        data_vis['system_prop']['chrom_mode'] = 'achrom'
        data_vis['system_prop'].pop('chrom')

    return None    

def DI_CCF_from_spec(inst,vis,data_dic,gen_dic):
    r"""**CCF conversion: DI** 
    
    Wrap-up for conversion of disk-integrated spectra into CCFs.
    
    Args:
        TDB
    
    Returns:
        None
    
    """ 

    #Calculating CCFs
    CCF_from_spec('DI',inst,vis,data_dic,gen_dic,data_dic['DI'])

    #Saving complementary data
    datasave_npz(data_dic[inst][vis]['proc_DI_data_paths']+'add',{'rest_frame':data_dic['DI'][inst][vis]['rest_frame'],'dim_exp':data_dic[inst][vis]['dim_exp']})

    return None

def DiffIntr_CCF_from_spec(data_type_gen,inst,vis,data_dic,gen_dic):
    r"""**CCF conversion: Diff, Intr** 
    
    Wrap-up for conversion of (out-of-transit) differential and intrinsic spectra into CCFs.
    
    Args:
        TDB
    
    Returns:
        None
    
    """ 
    data_inst = data_dic[inst]
    data_vis = data_inst[vis]
    gen_vis = gen_dic[inst][vis]

    #Calculating CCFs
    #    - if there is atmospheric contamination, excluding them from spectra before the conversion of intrinsic profiles would create empty ranges that shift between exposures
    # due to the planet orbital motion, thus potentially excluding mask lines from some exposures and not others, and resulting in CCFs that are not equivalent between all exposures
    #      the exclusion is thus applied after the conversion, internally to the routine 
    #    - conversion of all differential profiles is only allowed if intrinsic spectra are not extracted
    CCF_from_spec(data_type_gen,inst,vis,data_dic,gen_dic,data_dic[data_type_gen])

    #Continuum pixels over all defined exposures
    #    - exclusion of planetary ranges is not required for intrinsic profiles if already applied to their definition, and if not already applied contamination is either negligible or neglected  
    #    - intrinsic profiles at the limbs may not be defined
    #    - the continuum is common to differential and intrinsic profiles, and defined from both their input continuum ranges, if required for error definition
    #      if only required for the calculation of the intrinsic profile continuum, it does not account for differential profile continuum
    if data_type_gen == 'Diff':iexp_conv = range(data_vis['n_in_visit'])
    elif data_type_gen == 'Intr':iexp_conv = list(gen_dic[inst][vis]['idx_out'])+list(gen_dic[inst][vis]['idx_in'][data_dic['Intr'][inst][vis]['idx_def']])   
    cond_def_cont_all  = np.zeros([len(iexp_conv)]+data_dic[inst][vis]['dim_exp'],dtype=bool)        
    for isub,iexp in enumerate(iexp_conv):         
        i_in = gen_vis['idx_exp2in'][iexp]
        if (i_in ==-1) and data_dic[data_type_gen]['disp_err']:
            gen ='Diff'
            iexp_eff = iexp
        elif i_in !=-1:
            gen='Intr'
            iexp_eff = i_in
        else:
            gen = None
        if gen is not None:
            data_exp = dataload_npz(data_vis['proc_'+gen+'_data_paths']+str(iexp_eff))
    
            #Continuum ranges
            #    - if planetary contamination is excluded from differential out-of-transit profiles, define a large enough initial continuum range if the planet has a wide velocimetric motion 
            for iord in range(data_dic[inst]['nord']):
                if iord in data_dic[gen]['cont_range'][inst]:
                    for bd_int in data_dic[gen]['cont_range'][inst][iord]:
                        cond_def_cont_all[isub,iord] |= ((data_exp['edge_bins'][iord,0:-1]>=bd_int[0]) & (data_exp['edge_bins'][iord,1:]<=bd_int[1]))        
                else:cond_def_cont_all[isub,iord] = True
                cond_def_cont_all[isub,iord] &= data_exp['cond_def'][iord]
                if (gen=='Diff') and ('Diff_prof' in data_dic['Atm']['no_plrange']) and (iexp in data_dic['Atm'][inst][vis]['iexp_no_plrange']):     
                    cond_def_cont_all[isub,iord] &= excl_plrange(cond_def_cont_all[isub,iord],data_dic['Atm'][inst][vis]['exclu_range_star'],iexp,data_exp['edge_bins'][iord],'CCF')[0]

    #Redefinition of continuum errors
    #    - attributing constant error to all points, if requested, based on the dispersion in the continuum
    if data_dic[data_type_gen]['disp_err'] is not None:
        print('         Setting errors on CCFs to continuum dispersion')
        for isub,iexp in enumerate(iexp_conv):         
            i_in = gen_vis['idx_exp2in'][iexp]
            if i_in ==-1:
                gen = 'Diff'
                iexp_eff = iexp
            else:
                gen='Intr'
                iexp_eff = i_in
            data_exp = dataload_npz(data_vis['proc_'+gen+'_data_paths']+str(iexp_eff))

            #Error set to scaled continuum dispersion
            for iord in range(data_dic[inst]['nord']):
                disp_cont=data_exp['flux'][iord,cond_def_cont_all[isub,iord]].std() 
                err_tab = np.sqrt(data_dic[data_type_gen]['disp_err'])*np.repeat(disp_cont,data_vis['nspec'])
                data_exp['cov'][iord] = (err_tab*err_tab)[None,:]

            #Overwrite exposure data
            np.savez_compressed(data_vis['proc_'+gen+'_data_paths']+str(iexp_eff),data=data_exp,allow_pickle=True)  

    #Updating/correcting continuum level
    if data_type_gen == 'Intr':
        loc_type = 'CCFIntr_from_SpecIntr'
        print('         Intrinsic continuum calculations')
        if data_dic['Intr']['calc_cont']:           
            data_dic['Intr'][inst][vis]['mean_cont'],cont_norm_flag=calc_Intr_mean_cont(data_dic['Intr'][inst][vis]['idx_def'],data_dic[inst]['nord'],data_dic[inst][vis]['nspec'],data_vis['proc_Intr_data_paths'],data_vis['type'],data_dic['Intr']['cont_range'],inst,data_dic['Intr']['cont_norm'],gen_dic['flag_err_inst'][inst],loc_type)
            np.savez_compressed(data_vis['proc_Intr_data_paths']+'_add',data={'mean_cont':data_dic['Intr'][inst][vis]['mean_cont'],'cont_norm_flag':cont_norm_flag,'type':loc_type},allow_pickle=True)
        else:
            check_flag = check_data({'0':data_vis['proc_Intr_data_paths']+'_add'},silent=True)
            if not check_flag:stop('WARNING: calculate continuum for intrinsic CCF (origin: intrinsic spectra)')
            data_add = dataload_npz(data_vis['proc_Intr_data_paths']+'_add')
            if data_add['type']!=loc_type:stop('WARNING: continuum type incompatible with data, run extraction again.')
            data_dic['Intr'][inst][vis]['mean_cont'] = data_add['mean_cont']
            if data_add['cont_norm_flag']:print('         Correcting intrinsic continuum')

    #Determine the correlation length for the visit
    if gen_dic['scr_search']:
        corr_length_determination(data_dic['Diff'][inst][vis],data_vis,gen_dic[inst][vis]['scr_search'],inst,vis,gen_dic)    

    return None


def Atm_CCF_from_spec(inst,vis,data_dic,gen_dic):
    r"""**CCF conversion: Atm** 
    
    Wrap-up for conversion of atmospheric spectra into CCFs.
    
    Args:
        TDB
    
    Returns:
        None
    
    """ 

    #Calculating CCFs
    CCF_from_spec('Atm',inst,vis,data_dic,gen_dic,data_dic['Atm'])

    #Saving complementary data
    datasave_npz(data_dic[inst][vis]['proc_Atm_data_paths']+'add',{'rest_frame':data_dic['Atm'][inst][vis]['rest_frame'],'dim_exp':data_dic[inst][vis]['dim_exp']})

    return None






def check_CCF_mask_lines(n_exp,edge_wav_all,cond_def_all,wav_mask,edge_velccf):
    r"""**CCF mask lines** 
    
    Identifies mask lines that should be used to compute CCFs.
    We keep lines for which the CCF RV table (once converted into wavelength space):
        
     - is fully within the spectral range of the input spectrum
     - contains no nan from the input spectrum  
     
    The check is made for the time-series of spectra given as input as a whole, as CCFs should be calculated from the same mask lines in all exposures. 
    Tables must have dimension [n_time x n_spec] as input.
    
    Args:
        TDB
    
    Returns:
        idx_lines_kept (1D array): indexes of lines kept in mask.
    
    """ 
    #Conversion from wavelengths in star rest rame (lines transitions) to wavelength in the spectrum rest frame
    #    - what we convert is the radial velocity table of the CCF (the edges of its bins)
    #      the new table has dimension n_lines*(n_RV+1)
    #      we later loop on each line, so that the sub-table is centered on the expected line transition, assuming the mask is centered on each successive pixel
    #    - nu_received/nu_source = (1 + v_receiver/c )/(1 - v_source/c )
    #  w_source/w_received   = (1 + v_receiver/c )/(1 - v_source/c )
    #  with v_receiver > 0 if receiver moving toward the source
    #  with v_source > 0   if source moving toward the receiver
    #    - with the receiver at rest in the frame where the spectrum was measured:
    #  v_receiver = 0.
    #  v_source = - rv_star_spframe since by definition rv < 0 when moving toward us
    #  -> wav_rstar/w_spframe = 1./(1 + rv_star_spframe/c )
    #  -> w_spframe = wav_rstar*(1 + rv_star_spframe/c ) 
    #    - dimension n_libes x n_RV
    edge_mask_lines=wav_mask[:,None]*(1.+(edge_velccf/c_light))

    #First start wavelength and last end wavelength of the RV table relative to each mask line
    #    - dimension n_lines
    wstart_mask_table = edge_mask_lines[:,0]
    wstop_mask_table = edge_mask_lines[:,-1]    
    
    #Cheking all input exposures
    n_lines = len(wav_mask)
    idx_lines_kept = np.arange(n_lines)
    for iexp in range(n_exp):

        #Check for discontinuous input table
        low_pix_wav = edge_wav_all[iexp,0:-1]
        high_pix_wav = edge_wav_all[iexp,1::]

        #Identifying the first and last original pixels overlapping with the RV table of each line in wavelength space
        #    - we exploit searchsorted(x,y) which returns i_k in x for each y[k] so that 
        # x_low[i_k-1]  <= y_low[k]  <  x_low[i_k]  (side = right)
        # x_high[i_k-1] <  y_high[k] <= x_high[i_k] (side = left)
        #    - with x the lower boundary of original pixels and y the lower boundary of new bins, we get i_k-1 the index of the first original pixel overlapping with the new bin
        #      with x the upper boundary of original pixels and y the upper boundary of new bins, we get i_k the index of the last original pixel overlapping with the new bin
        #    - only defined pixels must be given to "searchsorted"
        #    - y values beyond the first & last values in x returns indexes -1 or n
        #    - because the old and new bins must be continuous here we could use the edge tables
        #      however this would make searchsorted find the position of all new bins in the mask table, while here we only need to find the start and end pixel
        idx_first_overpix =np.searchsorted(low_pix_wav,wstart_mask_table,side='right')-1
        idx_last_overpix =np.searchsorted(high_pix_wav,wstop_mask_table,side='left')        
        
        #Keep lines whose range is fully within that of the input spectrum
        #    - condition can be invalid for lines at the edges of the spectral range and wide RV tables
        idx_keepL = np_where1D( (idx_first_overpix>=0) & (idx_last_overpix<=len(low_pix_wav)-1) )
    
        #Keep lines whose range does not contain nan values from the input spectrum
        cond_keepL_sub = np.ones(len(idx_keepL),dtype=bool)
        for iline,(idx_first_overpix_line,idx_last_overpix_line) in enumerate(zip(idx_first_overpix[idx_keepL],idx_last_overpix[idx_keepL])):
            cond_keepL_sub[iline] &= (np.sum(~cond_def_all[iexp,idx_first_overpix_line:idx_last_overpix_line+1]) == 0)
    
        #Updates indexes of lines to be kept
        idx_keepL=idx_keepL[cond_keepL_sub]           #indexes relative to reduced table
        idx_lines_kept = idx_lines_kept[idx_keepL]    #indexes relative to reduced table but containting original line indexes in mask
        
        #Updates mask tables to speed the check
        wstart_mask_table = wstart_mask_table[idx_keepL]
        wstop_mask_table = wstop_mask_table[idx_keepL]

    return idx_lines_kept






def new_compute_CCF(edge_wav,flux,cov,resamp_mode,edge_velccf,wght_mask,wav_mask,nthreads,cal = None):
    r"""**Main routine to compute CCF** 
    
    Calculates CCFs from input spectra, propagating covariance matrix
    
    Args:
        TDB
    
    Returns:
        None
    
    """ 
    #Check for discontinuous input table
    low_pix_wav = edge_wav[0:-1]
    high_pix_wav = edge_wav[1::]
    if np.any(low_pix_wav[1:]-high_pix_wav[0:-1]):
        stop('Spectral bins must be continuous')
    
    #Line wavelength at each requested RV
    #    - for each line, the rest wavelength of its transition (source, assimilated to the star rest frame) is shifted to all trial wavelengths of the spectrum rest frame (receiver) associated with the CCF RVs
    #      see gen_specdopshift():
    # w_receiver = w_source * (1+ rv[s/r]/c))
    # w_trial = w_line * (1+ rv[line/trial]/c))
    n_RV=len(edge_velccf)-1
    edge_mask_lines=wav_mask[:,None]*gen_specdopshift(edge_velccf)

    #Call to parallelized function   
    if (nthreads>1) and (nthreads<=len(wght_mask)):            
        common_args=(n_RV,edge_wav,flux,cov,cal,resamp_mode)
        chunkable_args=[edge_mask_lines,wght_mask]
        fluxCCF,covCCF_line,nd_covCCF_line,contCCF=multithread_new_compute_CCF(sub_new_compute_CCF,nthreads,len(wght_mask),chunkable_args,common_args)                           

    #Regular routine
    else:
        fluxCCF,covCCF_line,nd_covCCF_line,contCCF=sub_new_compute_CCF(edge_mask_lines,wght_mask,n_RV,edge_wav,flux,cov,cal,resamp_mode)

    #Computing final covariance matrix
    #    - maximum dimension of covariance matrix from all contributing orders
    if cov is not None:
        nd_covCCF = np.amax(nd_covCCF_line) 
        covCCF = np.zeros([nd_covCCF,n_RV],dtype=float)
        for isub,wght_mask_line in enumerate(wght_mask):
            covCCF[0:nd_covCCF_line[isub],:] +=  covCCF_line[isub]         
    else:
        covCCF = None

    return fluxCCF,covCCF,contCCF


def sub_new_compute_CCF(edge_mask,wght_mask,n_RV,edge_wav,flux,cov,cal,resamp_mode):
    r"""**CCF computation** 
    
    Args:
        TDB
    
    Returns:
        None
    
    """   
    #Loop on selected lines
    #    - for each line in the line list, we co-add the contribution to the CCF
    #    - the bins have constant spectral width (which is the same for all lines in RV space, but changes with the line in wavelength space)
    nL_kept = len(wght_mask)
    fluxCCF = np.zeros(n_RV,dtype=float)
    contCCF = 0.
    covCCF_line = np.zeros(nL_kept,dtype=object)
    nd_covCCF_line = np.zeros(nL_kept,dtype=int)
    for isub,(edge_mask_line,wght_mask_line) in enumerate(zip(edge_mask,wght_mask)):

        #Spectrum around current line brought back from extracted to raw count units
        #    - if a calibration profile is provided as input we take its mean value in the local line range and use it to scale the input spectrum
        #      this is to get the spectrum as close as possible to its original count level, so that regions of the spectrum with comparable flux levels but different count levels do not contribute in the same way to the CCF
        #      the use of a constant estimated calibration rather than the actual profile is to keep the color balance of the spectrum intact, and avoid biasing the CCF
        if cal is not None:
            idxCCF_sub = np_where1D((edge_wav>=edge_mask_line[0]) & (edge_wav<=edge_mask_line[-1]))    #indexes where spectrum falls within mask line range
            idxCCF_sub_max = min([len(edge_wav)-1,idxCCF_sub[-1]])                                     
            mean_gainCCF_sub = np.nanmean(cal[idxCCF_sub[0]:idxCCF_sub_max+1])
        else:mean_gainCCF_sub =  1.

        #Spectrum around current line resampled on the CCF table
        if cov is None:
            fluxCCF_sub = bind.resampling(edge_mask_line,edge_wav, flux, kind=resamp_mode)/mean_gainCCF_sub                        
        else:
            fluxCCF_sub,covCCF_sub = bind.resampling(edge_mask_line,edge_wav, flux/mean_gainCCF_sub , cov = cov/mean_gainCCF_sub**2., kind=resamp_mode)   
            covCCF_line[isub]=(wght_mask_line**2.)*covCCF_sub
            nd_covCCF_line[isub] = np.shape(covCCF_line[isub])[0]                      

        #Add the weighted contribution of current line to the CCF
        fluxCCF += wght_mask_line*fluxCCF_sub
        contCCF += wght_mask_line/mean_gainCCF_sub

    return fluxCCF,covCCF_line,nd_covCCF_line,contCCF


def multithread_new_compute_CCF(func_input,nthreads,n_elem,y_inputs,common_args):
    r"""**Multithreading of sub_new_compute_CCF().**
    
    Specific implementation of `MAIN_multithread()`

    Args:
        func_input (function): multi-threaded function
        nthreads (int): number of threads
        n_elem (int): number of elements to thread
        y_inputs (list): threadable function inputs 
        common_args (tuple): common function inputs
    
    Returns:
        TBD
    
    """ 
    pool_proc = Pool(processes=nthreads)  
      
    #Indexes of chunks to be processed by each core       
    ind_chunk_list=init_parallel_func(nthreads,n_elem)

    #2 arrays with dimensions n_lines x n and n_lines
    chunked_args=[(y_inputs[0][ind_chunk[0]:ind_chunk[1],:],y_inputs[1][ind_chunk[0]:ind_chunk[1]])+common_args for ind_chunk in ind_chunk_list]				
         
    #------------------------------------------------------------------------------------	     					
    #Return the results from all cores as elements of a tuple
    #    - arguments could be given whole to map(), and be automatically divided using an input 'chunksize', however for long arrays it takes 
    # less time to do the chunking before
    all_results=tuple(tab for tab in pool_proc.starmap(func_input,chunked_args))

    #------------------------------------------------------------------------------------	 			
    #Outputs:
    #    - flux table co-added between processors	
    #    - covariance matrix and dimension table are appended between processors
    fluxCCF = np.sum(tuple(all_results[i][0] for i in range(nthreads)),axis=0)
    covCCF_line = np.concatenate(tuple(all_results[i][1] for i in range(nthreads)))
    nd_covCCF_line = np.concatenate(tuple(all_results[i][2] for i in range(nthreads)))
    contCCF = np.sum(tuple(all_results[i][3] for i in range(nthreads)))

    pool_proc.close()
    pool_proc.join()

    return fluxCCF,covCCF_line,nd_covCCF_line,contCCF










def conv_2D_to_1D_spec(data_type_gen,inst,vis,gen_dic,data_dic,prop_dic,coord_dic):
    """**Wrap-up function to compute 1D spectra.**
    
    Runs ANTARESS with default or manual settings.  
    
    Args:
        TBD
    
    Returns:
        None
    
    """ 
    data_vis=data_dic[inst][vis]
    gen_vis=gen_dic[inst][vis]
    dir_save = {} 
    iexp_conv,data_type_key,data_type = init_conversion(data_type_gen,gen_dic,prop_dic,inst,vis,'1Dfrom2D',dir_save,data_dic)

    #Paths
    if (data_type_gen=='DI') and (data_dic[data_type_gen][inst][vis]['rest_frame']!='star'):com_frame = ''
    else:com_frame = '_star'
    proc_com_data_paths_new = gen_dic['save_data_dir']+'Processed_data/spec1D_'+inst+'_'+vis+'_com'+com_frame

    #Calculating data
    if gen_dic['calc_spec_1D_'+data_type_gen]:
        print('         Calculating data')
        nthreads = gen_dic['nthreads_spec_1D_'+data_type_gen]
        if data_vis['type']!='spec2D':stop('Data must be 2D')
        nspec_1D = prop_dic['spec_1D_prop'][inst]['nspec']
        cen_bins_1D = prop_dic['spec_1D_prop'][inst]['cen_bins']
        edge_bins_1D = prop_dic['spec_1D_prop'][inst]['edge_bins']
     
        #Initializing weight calculation conditions
        #    - at this stage no 2D/1D conversion has been applied
        calc_EFsc2,calc_var_ref2,calc_flux_sc_all,var_key_def = weights_bin_prof_calc(None,data_type,None,None,None,check_var = False)   
        calc_cond = (calc_EFsc2,calc_var_ref2,calc_flux_sc_all)        

        #Associated tables  
        dt_all = coord_dic[inst][vis]['t_dur']       
        proc_data_paths = {}
        if gen_dic['flux_sc'] and calc_flux_sc_all:scaled_data_paths = {}
        else:scaled_data_paths = None
        if ('spec' in data_vis['type']) and gen_dic['corr_tell'] and calc_EFsc2:tell_data_paths =  {}         
        else:tell_data_paths =  None
        if calc_EFsc2:sing_gcal_data_paths = {}
        else:sing_gcal_data_paths = None
        if (sing_gcal_data_paths is not None) and (vis in data_dic[inst]['gcal_blaze_vis']):cond_sdet2 = True
        else:cond_sdet2 = False
        if (calc_EFsc2 or calc_var_ref2):DImast_weight_data_paths = {}
        else:DImast_weight_data_paths = None
        for key in data_type_key:   
            proc_data_paths[key] = data_vis['proc_'+key+'_data_paths']
            if scaled_data_paths is not None:scaled_data_paths[key] = data_vis['scaled_'+key+'_data_paths']
            if tell_data_paths is not None:
                if ('tell_'+key+'_data_paths' not in data_vis):stop('ERROR : weighing telluric profiles undefined; make sure you activate gen_dic["calc_proc_data"] and gen_dic["calc_corr_tell"] when running this module.')  
                tell_data_paths[key] = data_vis['tell_'+key+'_data_paths']
            if sing_gcal_data_paths is not None:
                if ('sing_gcal_'+key+'_data_paths' not in data_vis):stop('ERROR : weighing calibration profiles undefined; make sure you activate gen_dic["calc_proc_data"] and gen_dic["calc_gcal"] when running this module.')  
                sing_gcal_data_paths[key] = data_vis['sing_gcal_'+key+'_data_paths']
            if DImast_weight_data_paths is not None:
                if 'mast_'+key+'_data_paths' not in data_vis:stop('ERROR : weighing DI master undefined; make sure you activate gen_dic["calc_proc_data"] and gen_dic["calc_DImast"] when running this module.')
                DImast_weight_data_paths[key] = data_vis['mast_'+key+'_data_paths']
        if (data_type_gen=='Atm') and (data_type=='Absorption') or ((data_type=='Emission') and data_dic['Intr']['cov_loc_star']): 
            LocEst_Atm_data_paths = data_vis['LocEst_Atm_data_paths']
        else:LocEst_Atm_data_paths = None

        #Processing all exposures
        ifirst = iexp_conv[0]
        common_args = (data_type_gen,data_type,gen_dic['resamp_mode'],dir_save,cen_bins_1D,edge_bins_1D,nspec_1D,data_dic[inst]['nord'],ifirst,proc_com_data_paths_new,\
                       gen_dic[inst][vis]['idx_in2exp'],data_dic['Intr']['cov_loc_star'],proc_data_paths,tell_data_paths,scaled_data_paths,sing_gcal_data_paths,DImast_weight_data_paths,LocEst_Atm_data_paths,inst,vis,gen_dic['corr_Fbal'],gen_dic['corr_FbalOrd'],\
                       gen_dic['save_data_dir'],gen_dic['type'],data_vis['type'],data_vis['dim_exp'],gen_vis['idx_exp2in'],gen_vis['idx_in'],gen_dic['type2var'],dt_all,data_vis['comm_sp_tab'],gen_dic['sequence'],calc_cond,cond_sdet2)
        if (nthreads>1) and (nthreads <=len(iexp_conv)): MAIN_multithread(conv_2D_to_1D_exp,nthreads,len(iexp_conv),[iexp_conv],common_args)                           
        else: conv_2D_to_1D_exp(iexp_conv,*common_args)  

        #Saving complementary data
        for gen in data_type_key: 
            rest_frame = data_dic[gen][inst][vis]['rest_frame']         #Rest frame propagated from original data       
            data_add = {'rest_frame':rest_frame,'dim_exp':[1,prop_dic['spec_1D_prop'][inst]['nspec']]}
            if gen=='Intr':data_add['iexp_conv'] = prop_dic[inst][vis]['idx_def'] 
            elif gen=='Diff':data_add['iexp_conv'] = gen_dic[inst][vis]['idx_out']
            else:data_add['iexp_conv'] = iexp_conv
            datasave_npz(dir_save[gen]+'add',data_add)  

    else: 
        check_data({'path':proc_com_data_paths_new})
   
    #Updating paths
    #    - scaling is defined as a function and does not need updating
    #    - calibration profiles are not used with 1D spectra and do not need to be defined
    #    - variance grids are saved under a path and with a keyword that is specific to both the present data type for which the variance is calculated here (for example EFsc2 for DI profiles) and the local data type for which they may be used later on (for example EFsc2 for Diff profiles)
    #      here the two types are by definition the same
    data_vis['proc_com'+com_frame+'_data_paths'] = proc_com_data_paths_new
    if com_frame=='':data_vis['proc_com_star_data_paths'] = deepcopy(data_vis['proc_com'+com_frame+'_data_paths'])
    for gen in dir_save:
        data_vis['proc_'+gen+'_data_paths'] = dir_save[gen]  
        if ('spec' in data_vis['type']) and gen_dic['corr_tell']:data_vis['tell_'+gen+'_data_paths'] = {}        
        data_vis['mast_'+gen+'_data_paths'] = {}   
    data_vis[gen_dic['type2var'][data_type]+'_'+data_type_gen+'_data_paths']={}
    if data_type_gen=='Atm':data_vis['LocEst_Atm_data_paths'] = {}
    for iexp in iexp_conv:
        gen = deepcopy(data_type_gen)
        iexp_eff = deepcopy(iexp) 
        if data_type_gen=='Intr':         
            if (iexp in gen_vis['idx_in']):iexp_eff = gen_vis['idx_exp2in'][iexp] 
            else:gen = 'Diff'
        data_vis['mast_'+gen+'_data_paths'][iexp_eff] = dir_save[gen]+'ref_'+str(iexp_eff)
        if ('spec' in data_vis['type']) and gen_dic['corr_tell']:data_vis['tell_'+data_type_gen+'_data_paths'][iexp] = dir_save[gen]+'_tell'+str(iexp_eff)
        data_vis[gen_dic['type2var'][data_type]+'_'+data_type_gen+'_data_paths'][iexp] = dir_save[gen]+'_'+gen_dic['type2var'][data_type]+str(iexp_eff)
        if data_type_gen=='Atm':data_vis['LocEst_Atm_data_paths'][iexp] = dir_save[gen]+'estloc_'+str(iexp_eff)
    
    #Convert spectral mode 
    print('         ANTARESS switched to 1D processing')
    data_vis['comm_sp_tab']=True
    data_vis['type']='spec1D'
    data_vis['nspec'] = prop_dic['spec_1D_prop'][inst]['nspec']
    data_dic[inst]['nord'] = 1
    data_vis['dim_all'] = [data_vis['n_in_visit'],1,data_vis['nspec']]
    data_vis['dim_exp'] = [1,data_vis['nspec']]
    data_vis['dim_sp'] = [data_vis['n_in_visit'],1]

    #Continuum level and correction
    if data_type_gen=='Intr': 
        print('         Intrinsic continuum calculations')
        loc_type = 'Spec1DIntr_from_Spec2DIntr'
        if data_dic['Intr']['calc_cont']:           
            data_dic['Intr'][inst][vis]['mean_cont'],cont_norm_flag=calc_Intr_mean_cont(data_dic['Intr'][inst][vis]['idx_def'],data_dic[inst]['nord'],data_vis['nspec'],data_vis['proc_Intr_data_paths'],data_vis['type'],data_dic['Intr']['cont_range'],inst,data_dic['Intr']['cont_norm'],gen_dic['flag_err_inst'][inst],loc_type)
            datasave_npz(data_vis['proc_Intr_data_paths']+'_add',{'mean_cont':data_dic['Intr'][inst][vis]['mean_cont'],'cont_norm_flag':cont_norm_flag,'type':loc_type})
        else:
            check_flag = check_data({'0':data_vis['proc_Intr_data_paths']+'_add'},silent=True)
            if not check_flag:stop('WARNING: calculate continuum for intrinsic spectra (origin: 2D spectra)')
            data_add = dataload_npz(data_vis['proc_Intr_data_paths']+'_add')
            if data_add['type']!=loc_type:stop('WARNING: continuum type incompatible with data, run extraction again.')
            data_dic['Intr'][inst][vis]['mean_cont'] = data_add['mean_cont']
            if data_add['cont_norm_flag']:print('         Correcting intrinsic continuum')
     
    return None

def conv_2D_to_1D_exp(iexp_conv,data_type_gen,data_type,resamp_mode,dir_save,cen_bins_1D,edge_bins_1D,nspec_1D,nord,ifirst,proc_com_data_paths,\
                      idx_in2exp,cov_loc_star,proc_data_paths,tell_data_paths,scaled_data_paths,sing_gcal_data_paths,DImast_weight_data_paths,LocEst_Atm_data_paths,inst,vis,gen_corr_Fbal,gen_corr_Fbal_ord,\
                      save_data_dir,gen_type,data_mode,dim_exp,idx_exp2in,idx_in,type2var,dt_all,comm_sp_tab,sequence,calc_cond,cond_sdet2):
    r"""**Main routine to convert 2D spectra into 1D spectra** 
    
    Calculates 1D spectra from 2D spectra, propagating covariance matrix
     - 2D spectra are resampled over the common table and coadded
     - spectral values from different orders have to be equivalent at a given wavelength
     - conversion is applied to the latest processed data of each type
     - converted data is saved independently, but used as default data in all modules following the conversion

    Args:
        TDB
    
    Returns:
        None
    
    """ 
    #Processing each exposure
    #    - the conversion can be seen as the binning of all orders after they are resampled on a common table
    for iexp_sub,iexp in enumerate(iexp_conv):   
        data_type_eff = deepcopy(data_type)
        iexp_eff = deepcopy(iexp)     #Effective index (relative to global or in-transit tables)
        iexp_glob = deepcopy(iexp)    #Global index         
        if (data_type_gen=='Intr'):
            if (iexp in idx_in):
                iexp_eff = idx_exp2in[iexp] 
            else:data_type_eff = 'Diff'
        elif data_type_eff=='Absorption':iexp_glob = idx_in[iexp]

        #Upload spectra and associated tables in star or local frame
        #    - all complementary tables are defined on the same grid as the current exposure   
        data_exp = dataload_npz(proc_data_paths[data_type_eff]+str(iexp_eff))
        if DImast_weight_data_paths is not None:
            
            #DI master
            #    - the master has been calculated on the common spectral grid for the visit
            #      if the resampling condition is not met, this grid is not the same as that of individual exposures
            data_ref = dataload_npz(DImast_weight_data_paths[data_type_eff][iexp_eff])
            if (not comm_sp_tab):
                flux_ref=np.zeros(dim_exp,dtype=float)*np.nan
                cov_ref=np.zeros(nord,dtype=object)
                for iord in range(nord): 
                    flux_ref[iord],cov_ref[iord] = bind.resampling(data_exp['edge_bins'][iord], data_ref['edge_bins'][iord], data_ref['flux'][iord] , cov = data_ref['cov'][iord], kind=resamp_mode) 
                data_ref['cen_bins'] = data_exp['cen_bins']
                data_ref['edge_bins'] = data_exp['edge_bins']
                data_ref['flux'] = flux_ref
                data_ref['cov'] = cov_ref
                data_ref['cond_def'] = ~np.isnan(flux_ref)
        if scaled_data_paths is not None:scaled_data_paths_exp = scaled_data_paths[data_type_eff]
        else:scaled_data_paths_exp = None
        if tell_data_paths is not None:data_exp['tell'] = dataload_npz(tell_data_paths[data_type_eff][iexp_eff])['tell'] 
        if sing_gcal_data_paths is not None:
            data_gcal = dataload_npz(sing_gcal_data_paths[data_type_eff][iexp_eff])
            data_exp['sing_gcal'] = data_gcal['gcal'] 
            if cond_sdet2:data_exp['sdet2'] = data_gcal['sdet2']   
        else:
            data_exp['sing_gcal']=None   
            data_exp['sdet2'] = None   
        flux_est_loc_exp = None
        cov_est_loc_exp = None
        if LocEst_Atm_data_paths is not None:    
            data_est_loc=dataload_npz(LocEst_Atm_data_paths[iexp_eff])
            flux_est_loc_exp = data_est_loc['flux']
            if cov_loc_star:cov_est_loc_exp = data_est_loc['cov']  
        if (data_type_gen=='Atm') and (data_type_eff=='Absorption'):SpSstar_spec = data_exp['SpSstar_spec']
        else:SpSstar_spec = None 

        #Weight definition
        #    - cannot be parallelized as functions cannot be pickled
        #    - here the binning is performed between overlapping orders of the same exposure 
        #      all profiles that are the same for overlapping orders (tellurics, disk-integrated stellar spectrum, global flux scaling, ...) are thus not used in the weighing 
        #      they are however processed in the same way as the exposure if used later on in the pipeline 
        #    - for intrinsic and atmospheric profiles we provide the broadband flux scaling, even if does not matter to the weighing, because it is otherwise set to 0 and messes up with weights definition
        #    - input profiles are by definition S2D, so there are no estimates of true variance already calculated
        data_exp['weights'],EFsc2_all,EFdiff2,EFintr2,EFem2,EAbs2 = weights_bin_prof(range(nord),scaled_data_paths_exp,inst,vis,gen_corr_Fbal,gen_corr_Fbal_ord,save_data_dir,nord,iexp_glob,data_type_eff,data_mode,dim_exp,data_exp['tell'],data_exp['sing_gcal'], data_exp['cen_bins'],dt_all[iexp_glob],data_ref['flux'],data_ref['cov'],calc_cond,
                                                                                     flux_est_loc_exp=flux_est_loc_exp,cov_est_loc_exp = cov_est_loc_exp, SpSstar_spec = SpSstar_spec,sdet_exp2 = data_exp['sdet2'])
        variances_1D = {'DI':EFsc2_all,'Diff':EFdiff2,'Intr':EFintr2,'Emission':EFem2,'Absorption':EAbs2}

        #-----------------------
        #Resample profiles and weights on 1D table in each order, and clean weights
        #    - at this stage, weights are still defined in S2D format
        #    - complementary tables are processed in the same way as the exposure spectrum, using the same weights but cleaning the weights according to the complementary table defined pixels    
        #    - the DI weighing master has followed the same shifts as intrinsic or atmospheric profiles, but always remain either defined on the common table, or on a specific table different from the table of its associated exposure
        #    - spectral scaling is not updated, since it is defined as a function and remains applicable to the 1D spectrum as with the 2D spectrum 
        #    - calibration profiles are not consistent in the overlaps between orders, and are not used anymore
        #-----------------------
        
        #Exposure spectrum
        flux_exp_all,cov_exp_all,cond_def_all,glob_weight_all,cond_def_binned = pre_calc_bin_prof(nord,[nspec_1D],range(nord),resamp_mode,None,data_exp,edge_bins_1D)

        #Variance
        #    - we use variances to define true errors, ie the diagonal of the covariance matrix that undergoes
        # var_ord = var_all*glob_weight_all**2.
        var_exp_all,_,cond_def_var_all,glob_weight_var_all,cond_def_var_binned = pre_calc_bin_prof(nord,[nspec_1D],range(nord),resamp_mode,None,{'edge_bins':data_exp['edge_bins'],'flux':variances_1D[data_type],'weights':data_exp['weights']**2.},edge_bins_1D,nocov=True)
        variance_ord_contr = np.zeros(nspec_1D, dtype=float) 
        
        #Complementary profiles
        if sequence not in ['st_master_tseries']:
            
            #DI weighing master
            if DImast_weight_data_paths is not None:
                data_ref['weights']= deepcopy(data_exp['weights'])
                flux_ref_all,cov_ref_all,cond_def_ref_all,glob_weight_ref_all,cond_def_ref_binned = pre_calc_bin_prof(nord,[nspec_1D],range(nord),resamp_mode,None,data_ref,edge_bins_1D)
                flux_ref_ord_contr=[]
                cov_ref_ord_contr=[] 
                
            #Telluric spectrum
            if tell_data_paths is not None:  
                tell_exp_all,_,cond_def_tell_all,glob_weight_tell_all,cond_def_tell_binned = pre_calc_bin_prof(nord,[nspec_1D],range(nord),resamp_mode,None,{'edge_bins':data_exp['edge_bins'],'flux':data_exp['tell'],'weights':data_exp['weights']},edge_bins_1D,nocov=True)
                tell_ord_contr = np.zeros(nspec_1D, dtype=float)
                
            #Surface ratio
            if SpSstar_spec is not None:  
                dic_temp = {'edge_bins':data_exp['edge_bins'],'flux':np.zeros(dim_exp,dtype=float),'weights':data_exp['weights']}
                for iord in range(nord):dic_temp['flux'][iord] = SpSstar_spec[iord](data_exp['cen_bins'][iord])
                SpSstar_exp_all,_,cond_def_SpSstar_all,glob_weight_SpSstar_all,cond_def_SpSstar_binned = pre_calc_bin_prof(nord,[nspec_1D],range(nord),resamp_mode,None,dic_temp,edge_bins_1D,nocov=True)
                SpSstar_spec_ord_contr = np.zeros(nspec_1D, dtype=float)
                
            #Planet-occulted profile
            if flux_est_loc_exp is not None:  
                if cov_est_loc_exp is None:   
                    floc_exp_all,_,cond_def_floc_all,glob_weight_floc_all,cond_def_floc_binned = pre_calc_bin_prof(nord,[nspec_1D],range(nord),resamp_mode,None,{'edge_bins':data_exp['edge_bins'],'flux':flux_est_loc_exp[iord],'weights':data_exp['weights']},edge_bins_1D,nocov=True)                                  
                else:
                    floc_exp_all,covloc_exp_all,cond_def_floc_all,glob_weight_floc_all,cond_def_floc_binned = pre_calc_bin_prof(nord,[nspec_1D],range(nord),resamp_mode,None,{'edge_bins':data_exp['edge_bins'],'flux':flux_est_loc_exp[iord],'cov':cov_est_loc_exp[iord],'weights':data_exp['weights']},edge_bins_1D)
                    cov_est_loc_ord_contr=[] 
                flux_est_loc_ord_contr=[]

        #Processing each order
        flux_ord_contr=[]
        cov_ord_contr=[] 
        for iord in range(nord):

            #Multiply by order weight and store
            flux_ord,cov_ord = bind.mul_array(flux_exp_all[iord] , cov_exp_all[iord] , glob_weight_all[iord])          
            flux_ord_contr+=[flux_ord]
            cov_ord_contr+=[cov_ord]

            #Complementary profiles
            variance_ord_contr+=dup_edges(bind.resampling(edge_bins_1D,data_exp['edge_bins'][iord],variances_1D[data_type][iord],kind=resamp_mode))*glob_weight_var_all[iord]  
            if sequence not in ['st_master_tseries']:          
                if DImast_weight_data_paths is not None:
                    flux_ref_ord,cov_ref_ord = bind.mul_array(flux_ref_all[iord] , cov_ref_all[iord] , glob_weight_ref_all[iord])          
                    flux_ref_ord_contr+=[flux_ref_ord]
                    cov_ref_ord_contr+=[cov_ref_ord]
                if tell_data_paths is not None:tell_ord_contr+=tell_exp_all[iord]*glob_weight_tell_all[iord] 
                if SpSstar_spec is not None:SpSstar_spec_ord_contr+=SpSstar_exp_all[iord]*glob_weight_SpSstar_all[iord] 
                if flux_est_loc_exp is not None:  
                    if cov_est_loc_exp is None:  
                        flux_est_loc_ord_contr+=floc_exp_all[iord]*glob_weight_tell_all[iord]                 
                    else:
                        floc_ref_ord,covloc_ref_ord = bind.mul_array(floc_exp_all[iord] , covloc_exp_all[iord] , glob_weight_ref_all[iord])          
                        flux_est_loc_ord_contr+=[floc_ref_ord]
                        cov_est_loc_ord_contr+=[covloc_ref_ord]                    

        #Co-addition of spectra from all orders
        flux_1D,cov_1D = bind.sum(flux_ord_contr,cov_ord_contr)

        #Reset undefined pixels to nan      
        flux_1D[~cond_def_binned]=np.nan        

        #Store data with artifical order for consistency with the routines
        #    - calibration profile are not used anymore afterward as there is no clear conversion from 2D to 1D for them
        #    - global flux scaling is not modified
        #      flux scaling tables are always called with global indexes
        #    - variance grids are saved for the data type that was converted (thus, only one data type in a given processing will have its 1D variance grid defined)
        data_exp1D = {'cen_bins':cen_bins_1D[None,:],'edge_bins':edge_bins_1D[None,:],'flux' : flux_1D[None,:],'cond_def' : cond_def_binned[None,:], 'cov' : [cov_1D]}   
        if sequence not in ['st_master_tseries']:
            if SpSstar_spec is not None:
                SpSstar_spec_ord_contr[~cond_def_SpSstar_binned]=0.  
                data_exp1D['SpSstar_spec'] =  SpSstar_spec_ord_contr[None,:]
            if DImast_weight_data_paths is not None:
                flux_ref_1D,cov_ref_1D = bind.sum(flux_ref_ord_contr,cov_ref_ord_contr)
                flux_ref_1D[~cond_def_ref_binned]=np.nan   
                datasave_npz(dir_save[data_type_eff]+'ref_'+str(iexp_eff),{'cen_bins':data_exp1D['cen_bins'],'edge_bins':data_exp1D['edge_bins'],'flux':flux_ref_1D[None,:],'cov':[cov_ref_1D]})          
            if tell_data_paths is not None:
                tell_ord_contr[~cond_def_tell_binned]=1.       
                datasave_npz(dir_save[data_type_eff]+'_tell'+str(iexp_eff), {'tell':tell_ord_contr[None,:]})                 
            if flux_est_loc_exp is not None:
                if cov_est_loc_exp is None:
                    flux_est_loc_ord_contr[~cond_def_floc_binned]=np.nan  
                    dic_sav_estloc = {'edge_bins':data_exp1D['edge_bins'],'flux':flux_est_loc_ord_contr[None,:]}
                else:
                    flux_est_loc_1D,cov_est_loc_1D = bind.sum(flux_est_loc_ord_contr,cov_est_loc_ord_contr)  
                    flux_est_loc_1D[~cond_def_floc_binned]=np.nan  
                    dic_sav_estloc = {'edge_bins':data_exp1D['edge_bins'],'flux':flux_est_loc_1D[None,:],'cov':[cov_est_loc_1D]}
                datasave_npz(dir_save[data_type_eff]+'estloc_'+str(iexp_eff),dic_sav_estloc)                  
        datasave_npz(dir_save[data_type_eff]+str(iexp_eff),data_exp1D)
        variance_ord_contr[~cond_def_var_binned]=np.nan   
        datasave_npz(dir_save[data_type_eff]+'_'+type2var[data_type]+str(iexp_eff), {'var':variance_ord_contr[None,:]})          

        #Update common table for the visit
        if iexp==ifirst:datasave_npz(proc_com_data_paths, {'dim_exp':[1,nspec_1D],'nspec':nspec_1D,'cen_bins':np.tile(cen_bins_1D,[1,1]),'edge_bins':np.tile(edge_bins_1D,[1,1])})      

    return None


