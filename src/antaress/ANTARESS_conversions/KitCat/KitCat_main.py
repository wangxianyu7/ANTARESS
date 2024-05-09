"""
AUTHOR : Michael Cretignier 
EDITOR : Khaled Al Moulla

KitCat - adapted to the ANTARESS workflow and upgraded by Vincent Bourrier
"""


# MODULES
# =============================================================================


import matplotlib.pylab   as     plt
import numpy              as     np
import pandas             as     pd
from   scipy.signal       import argrelextrema
import bindensity as bind
from copy import deepcopy
from pathos.multiprocessing import Pool
from ..ANTARESS_general.utils import stop,np_where1D,dataload_npz,init_parallel_func,gen_specdopshift
from ..ANTARESS_general.constant_data import c_light,c_light_m
from ..ANTARESS_conversions.KitCat import Kitcat_classes         as     myc
from ..ANTARESS_conversions.KitCat import Kitcat_functions       as     myf
from ..ANTARESS_conversions.KitCat import calculate_RV_line_by_line3 as calculate_RV_line_by_line 


def kitcat_mask(mask_dic,fwhm_ccf,cen_bins_mast,inst,edge_bins_mast,flux_mask_norm,gen_dic,save_data_paths,tell_spec,rv_sys,min_specdopshift_receiver_Earth,max_specdopshift_receiver_Earth,dic_sav,plot_spec,plot_ld,plot_ld_lw,plot_RVdev_fit,cont_func_dic,vis_iexp_in_bin,
                data_type_gen,data_dic,plot_tellcont,plot_vald_depthcorr,plot_morphasym,plot_morphshape,plot_RVdisp):
    mask_info=''
       
    #=============================================================================
    #Spectra preparation    
    #=============================================================================
    if mask_dic['verbose']:print('           Formatting spectrum')
    
    #Kernel smoothing length
    if (inst in mask_dic['kernel_smooth']):kernel_smooth = mask_dic['kernel_smooth'][inst]
    else:
        kernel_smooth={'ESPRESSO':5}
        if inst not in kernel_smooth:stop('Add '+inst+' to "kernel_smooth" in KitCat_main.py')
        kernel_smooth=kernel_smooth[inst]        
    
    #Kernel profile
    if fwhm_ccf<=15.:kernel_prof = 'savgol'
    else:kernel_prof = 'gaussian'

    #Smoothed master spectrum  
    flux_norm = myf.smooth(flux_mask_norm, kernel_smooth, shape=kernel_prof)

    #Resampling master and telluric spectrum on oversampled regular grid
    if (inst in mask_dic['dw_reg']):dw_reg = mask_dic['dw_reg'][inst]    
    else:
        dw_reg={'ESPRESSO':0.001}
        if inst not in dw_reg:stop('Add '+inst+' to "dw_reg" in KitCat_main.py')
        dw_reg=dw_reg[inst]           
    n_reg = int(np.ceil((edge_bins_mast[-1]-edge_bins_mast[0])/dw_reg))
    if n_reg<len(cen_bins_mast)*5:print('         WARNING: oversampling lower than 5')
    edge_bins_reg = np.linspace(edge_bins_mast[0],edge_bins_mast[-1],n_reg)
    cen_bins_reg = 0.5*(edge_bins_reg[0:-1]+edge_bins_reg[1::])
    dw_reg = cen_bins_reg[1]-cen_bins_reg[0]
    flux_norm_reg = bind.resampling(edge_bins_reg, edge_bins_mast, flux_norm, kind=gen_dic['resamp_mode'])   
    cond_undef_reg = np.isnan(flux_norm_reg)
    flux_norm_reg[cond_undef_reg] = 0.

    #Store for plotting
    if plot_spec:dic_sav.update({'cen_bins_reg':cen_bins_reg,'flux_norm_reg':flux_norm_reg})

    #=============================================================================
    # EXTREMA LOCALISATION
    #=============================================================================
    if mask_dic['verbose']:print('           Finding extrema')

    #Vicinity window for the local extrema algorithm
    #    - vicinity_fwhm in fraction of the FWHM (should not be too small or the algorithm finds extrema within the lines themselves)
    #      vicinity_reg in pixels of the oversampled regular grid
    if (inst in mask_dic['vicinity_fwhm']):vicinity_fwhm = mask_dic['vicinity_fwhm'][inst]    
    else:
        vicinity_fwhm={'ESPRESSO':10.}    
        if inst not in vicinity_fwhm:stop('Add '+inst+' to "vicinity_fwhm" in KitCat_main.py')
        vicinity_fwhm=vicinity_fwhm[inst]              
    vicinity_rv = fwhm_ccf/vicinity_fwhm
    vicinity_spec = vicinity_rv*cen_bins_reg/c_light
    vicinity_reg = int(np.min(vicinity_spec/dw_reg))
    if vicinity_reg<5:stop('Vicinity window = '+str(vicinity_fwhm)+' x FWHM = '+str(vicinity_reg)+' pixels < 5 pixels: decrease "vicinity_fwhm"')

    #Minima
    index_minima, f_minima = myf.local_max(-flux_norm_reg,vicinity_reg)
    index_minima = index_minima.astype('int')
    wave_minima = cen_bins_reg[index_minima]
   
    #Minima    
    index_maxima, flux_maxima = myf.local_max(flux_norm_reg,vicinity_reg)
    index_maxima = index_maxima.astype('int')
    wave_maxima = cen_bins_reg[index_maxima]
       
    #Checking edges
    if wave_minima[0]<wave_maxima[0]:
        wave_maxima = np.insert(wave_maxima,0,cen_bins_reg[0])
        flux_maxima = np.insert(flux_maxima,0,flux_norm_reg[0])
        index_maxima = np.insert(index_maxima,0,0)
    if wave_minima[-1]>wave_maxima[-1]:
        wave_maxima = np.insert(wave_maxima,-1,cen_bins_reg[-1])    
        flux_maxima = np.insert(flux_maxima,-1,flux_norm_reg[-1]) 
        index_maxima = np.insert(index_maxima,-1,len(cen_bins_reg)-1)
    
    #Extrema
    wave_extrema = np.append(wave_minima,wave_maxima)
    idx_sort = wave_extrema.argsort()
    wave_extrema = wave_extrema[idx_sort]    
    flag_extrema = np.append(-1*np.ones(len(wave_minima)),np.ones(len(wave_maxima)))[idx_sort]    
    flux_extrema = np.append(-f_minima,flux_maxima)[idx_sort]    
    index_extrema = np.append(index_minima,index_maxima)[idx_sort]     

    #Check for two consecutive maxima minima
    if len(np.unique(flag_extrema[::2]))>1:  
        trash,matrix = myf.clustering(flag_extrema,0.01,0)
        
        delete_liste = []
        for j in range(len(matrix)):
            numero = np.arange(matrix[j,0],matrix[j,1]+2)
            fluxes = flux_extrema[numero].argsort()
            if trash[j][0] == 1 :
                delete_liste.append(numero[fluxes[0:-1]])
            else:
                delete_liste.append(numero[fluxes[1:]])
        delete_liste = np.hstack(delete_liste)        
        
        flag_extrema = np.delete(flag_extrema,delete_liste)       
        flux_extrema = np.delete(flux_extrema,delete_liste)       
        wave_extrema = np.delete(wave_extrema,delete_liste)       
        index_extrema = np.delete(index_extrema,delete_liste)       
    
    #Retrieve unique minima/maxima
    #    - maxima table have one more value than minima table
    minima = np_where1D(flag_extrema==-1)
    maxima = np_where1D(flag_extrema==1)
    index_minima = index_extrema[minima] 
    index_maxima = index_extrema[maxima]
    f_minima = flux_extrema[minima] 
    flux_maxima = flux_extrema[maxima]
    wave_minima = wave_extrema[minima] 
    wave_maxima = wave_extrema[maxima]

    #Store extrema in 3 columns of (minima, maxima to the left of the minima, maxima to the right of the minima)
    matrix_wave = np.vstack([wave_minima,wave_maxima[0:-1],wave_maxima[1:]]).T
    matrix_flux = np.vstack([f_minima,flux_maxima[0:-1],flux_maxima[1:]]).T
    matrix_index = np.vstack([index_minima,index_maxima[0:-1],index_maxima[1:]]).T
    nlines = len(matrix_index[:,0])

    #=============================================================================
    #Line screening
    #=============================================================================
    if mask_dic['verbose']:print('           Selection: screening')

    #Remove lines in selected rejection ranges + telluric oxygen bands 
    #    - shifted from the Earth (source) into the star (receiver) rest frame. We neglect the Keplerian and BERV motions.
    #      see gen_specdopshift():
    # w_star ~ w_starbar = w_solbar * (1+ rv[solbar/starbar]/c)) ~ w_Earth * (1+ rv_sys/c))
    line_rej_range = np.array([[6865,6930],[7590,7705]])*gen_specdopshift(rv_sys)
    if (inst in mask_dic['line_rej_range']):line_rej_range = np.append(line_rej_range,np.array(mask_dic['line_rej_range'][inst]),axis=0)
    cond_rej = np.repeat(False,nlines)
    for bd_int in line_rej_range:  
        cond_rej |= ((matrix_wave[:,0]>bd_int[0]) & ((matrix_wave[:,0]<bd_int[1])))
    mask_line=~cond_rej
    matrix_wave = matrix_wave[mask_line]
    matrix_flux = matrix_flux[mask_line]
    matrix_index = matrix_index[mask_line] 

    #Issue warning
    if np.min(matrix_wave[:,0])<3000.:print('         WARNING: expect bad RASSINE normalisation below 3000 A')
    if np.max(matrix_wave[:,0])>10000.*gen_specdopshift(rv_sys):print('         WARNING: expect issues with telluric oxygen above 10000 A')    
    
    #Remove lines with same left and right maxima, or with left (resp. right) maxima to the right (resp. left) of their minima
    mask_line = (matrix_index[:,1]!=matrix_index[:,2]) & (matrix_index[:,2]>matrix_index[:,0]) & (matrix_index[:,1]<matrix_index[:,0])
    matrix_wave = matrix_wave[mask_line]
    matrix_flux = matrix_flux[mask_line]
    matrix_index = matrix_index[mask_line]   
    
    #Store line list as dictionary to facilitate manipulations
    wave_minima = matrix_wave[:,0] 
    w_maxima_left = matrix_wave[:,1]    
    w_maxima_right = matrix_wave[:,2] 
    f_minima = matrix_flux[:,0] 
    f_maxima_left = matrix_flux[:,1]    
    f_maxima_right = matrix_flux[:,2] 
    Dico = {'w_minima':wave_minima,'w_maxima_left':w_maxima_left,'w_maxima_right':w_maxima_right, 
            'idx_minima':matrix_index[:,0],'idx_maxima_left':matrix_index[:,1],'idx_maxima_right':matrix_index[:,2],
            'f_minima':f_minima,'line_depth':1.-f_minima,'f_maxima_left':f_maxima_left,'f_maxima_right':f_maxima_right, 'dist_continuum':(1-np.min([f_maxima_left,f_maxima_right],axis=0))/(1 - f_minima)}
    Dico = pd.DataFrame(Dico)
    Dico['left_width'] = Dico['w_minima']-Dico['w_maxima_left']
    Dico['right_width'] = Dico['w_minima']-Dico['w_maxima_right']
    Dico['left_depth'] = Dico['f_minima']-Dico['f_maxima_left']
    Dico['right_depth'] = Dico['f_minima']-Dico['f_maxima_right']    

    #Store for plotting
    dic_sav['sel0']={}
    if plot_spec:
        dic_sav['sel0'].update({'nl_mask_pre':nlines,'nl_mask_post':len(Dico['w_minima'])})                   
        for key in ['f_minima','w_maxima_left','w_maxima_right','f_maxima_left','f_maxima_right']:dic_sav['sel0'][key] = np.array(Dico[key])
        dic_sav['sel0']['w_lines'] = np.array(Dico['w_minima'])
        dic_sav['line_rej_range'] = line_rej_range    

    #=============================================================================  
    #RV weights 
    #    - weights do not include a contribution from the flux errors because they are used to compute CCFs in count-equivalent units (it would otherwise account twice for this ponderation)
    #=============================================================================
    det_noise = {'ESPRESSO':1e-3}[inst]
    flux_gradient = np.gradient(flux_norm_reg)/dw_reg
    weight_vec = cen_bins_reg**2 * flux_gradient**2/(flux_norm_reg+det_noise**2.)
    fwhm_pix = np.round(np.array(Dico['w_minima'])*(fwhm_ccf/c_light)*(1./dw_reg)).astype('int')  #maximum window for weight computation
    idx_maxima_left_max = np.array(Dico['idx_minima'])-fwhm_pix
    idx_maxima_right_min = np.array(Dico['idx_minima'])+fwhm_pix
    idx_maxima_left_eff = np.maximum(np.array(Dico['idx_maxima_left']),idx_maxima_left_max)
    idx_maxima_right_eff = np.minimum(np.array(Dico['idx_maxima_right']),idx_maxima_right_min)
    weight_rv  = []
    weight_rv_sym  = []
    for indice_left,indice_right,idx_maxima_left_max_loc,idx_maxima_right_min_loc in zip(idx_maxima_left_eff,idx_maxima_right_eff,idx_maxima_left_max,idx_maxima_right_min):
        weight_rv.append(np.sum(weight_vec[indice_left:indice_right+1]))
        weight_rv_sym.append(np.sum(weight_vec[idx_maxima_left_max_loc:idx_maxima_right_min_loc+1]))

    #Final weights
    #    - see Bouchy+2001 Eq 8, weight is defined as the squared inverse of the error on a line RV 
    # W[i] = w[i]^2* (df/dw)^2 / (f[i]+sD^2) 
    #      = weight_vec with sD = 1e-3 
    #      and
    # W[line] = sum(W[i])
    #      the weight is defined here as sqrt(W[line]) but is put to the square by pipelines before computing CCFs
    weight_rv = np.array(weight_rv)
    weight_rv /= np.nanpercentile(weight_rv,95)
    weight_rv[weight_rv<=0] = np.min(abs(weight_rv[weight_rv!=0]))
    Dico['weight_rv'] = (weight_rv)**(0.5)
    
    weight_rv_sym = np.array(weight_rv_sym)
    weight_rv_sym /= np.nanpercentile(weight_rv_sym,95)
    weight_rv_sym[weight_rv_sym<=0] = np.min(abs(weight_rv_sym[weight_rv_sym!=0]))
    Dico['weight_rv_sym'] = (weight_rv_sym)**(0.5)    
    
    
    
    #=============================================================================
    #Line depth and width selection
    #=============================================================================
    if mask_dic['verbose']:print('           Selection: depth and width')
    
    #Depth between minima and continuum
    line_depth_cont = 1.-np.array(Dico['f_minima'])
    
    #Depth between minima and average of bracketing maxima
    line_depth = abs(np.array(Dico['f_minima'])  - np.mean([np.array(Dico['f_maxima_left']),np.array(Dico['f_maxima_right'])],axis=0))      
    
    #Depth range criteria
    if (inst in mask_dic['linedepth_min']):linedepth_min = mask_dic['linedepth_min'][inst]
    else:linedepth_min = 0.01    #mainly to exclude abnormal 'positive' lines
    if (inst in mask_dic['linedepth_max']):linedepth_max = mask_dic['linedepth_max'][inst]
    else:linedepth_max = 0.99    #mainly to exclude abnormal 'negative' lines    
    if (inst in mask_dic['linedepth_cont_min']):linedepth_cont_min = mask_dic['linedepth_cont_min'][inst]
    else:
        if (fwhm_ccf<15.):linedepth_cont_min = 0.10   
        else:linedepth_cont_min = 0.03 
    if (inst in mask_dic['linedepth_cont_max']):linedepth_cont_max = mask_dic['linedepth_cont_max'][inst]
    else:
        if (fwhm_ccf<15.):linedepth_cont_max = 0.95  
        else:linedepth_cont_max = 0.40 
        
    #Depth = f(continuum depth) limit
    #    - both are correlated, so that a specific threshold can be defined
    if (inst in mask_dic['linedepth_contdepth']):
        linedepth_contdepth = mask_dic['linedepth_contdepth'][inst] 
        
        #Linear threshold
        linedepth_rel = linedepth_contdepth[0]*line_depth_cont+linedepth_contdepth[1]
        
    else:
        linedepth_contdepth = None
        linedepth_rel = 10.
     
    #Store for plotting
    if plot_ld:
        dic_sav.update({'line_depth_cont':line_depth_cont,'line_depth':line_depth,'linedepth_cont_min':linedepth_cont_min,'linedepth_cont_max':linedepth_cont_max,'linedepth_max':linedepth_max,'linedepth_min':linedepth_min,'linedepth_contdepth':linedepth_contdepth,
                        'weight_rv_ld':np.array(Dico[mask_dic['mask_weights']])})
  
    #Keep lines with continuum depth within the requested depth range, and with effective depth larger than a threshold
    mask_line = (line_depth_cont > linedepth_cont_min)&(line_depth_cont < linedepth_cont_max)&(line_depth > linedepth_min ) & (line_depth < linedepth_max) & (line_depth<linedepth_rel)
    Dico = Dico.loc[mask_line]
    Dico = Dico.reset_index(drop=True)    
    
    #------------------------------------------------------

    #Keep lines with minimum depth and width larger than requested threshold
    if (mask_dic['line_width_logmin'] is not None):line_width_logmin = mask_dic['line_width_logmin']
    else:line_width_logmin = -1.3 
    if (mask_dic['line_depth_logmin'] is not None):line_depth_logmin = mask_dic['line_depth_logmin']
    else:line_depth_logmin = -1.4 
    log10_min_line_width = np.log10(np.min(np.vstack([abs(Dico['left_width']),abs(Dico['right_width'])]),axis=0))
    log10_min_line_depth = np.log10(np.min(np.vstack([abs(Dico['left_depth']),abs(Dico['right_depth'])]),axis=0))  
    mask_line = (log10_min_line_depth>line_depth_logmin)&(log10_min_line_width>line_width_logmin)

    #Store for plotting
    if plot_ld_lw:
        dic_sav.update({'log10_min_line_depth':log10_min_line_depth,'log10_min_line_width':log10_min_line_width,'line_depth_logmin':line_depth_logmin,'line_width_logmin':line_width_logmin,'weight_rv_ld_lw':np.array(Dico[mask_dic['mask_weights']])})
        
    #Restrict
    Dico = Dico.loc[mask_line]
    Dico = Dico.reset_index(drop=True) 
    
    #Store for plotting
    dic_sav['sel1']={}
    if plot_spec:
        dic_sav['sel1'].update({'nl_mask_pre':nlines,'nl_mask_post':len(Dico['w_minima']),'linedepth_cont_min':linedepth_cont_min,'linedepth_cont_max':linedepth_cont_max})                                
        for key in ['f_minima','w_maxima_left','w_maxima_right','f_maxima_left','f_maxima_right']:dic_sav['sel1'][key] = np.array(Dico[key])
        dic_sav['sel1']['w_lines'] = np.array(Dico['w_minima'])

    #=============================================================================
    #Line properties
    #=============================================================================
    
    Dico['min_width'] = np.min(np.vstack([abs(Dico['left_width']),abs(Dico['right_width'])]),axis=0)
    Dico['min_width_signed'] = np.diag(np.vstack([Dico['left_width'],Dico['right_width']])[np.argmin(np.vstack([abs(Dico['left_width']),abs(Dico['right_width'])]),axis=0),:])
    Dico['min_depth'] = np.min(np.vstack([abs(Dico['left_depth']),abs(Dico['right_depth'])]),axis=0)
    Dico['width'] = Dico['w_maxima_right']-Dico['w_maxima_left']
    Dico['line_depth'] = 1.- Dico['f_minima']
    Dico['line_nb']=np.arange(len(Dico['w_minima']))

    #Equivalent width
    EW = []
    for idx_maxima_left_loc,idx_maxima_right_loc in zip(np.array(Dico['idx_maxima_left']),np.array(Dico['idx_maxima_right'])):
        EW.append(np.sum(abs(1.-0.5*(flux_norm_reg[idx_maxima_left_loc:idx_maxima_right_loc]+flux_norm_reg[idx_maxima_left_loc+1:idx_maxima_right_loc+1]))))
    Dico['equivalent_width'] = dw_reg*np.array(EW)

    Dico['mean_grad'] = 0.5*(abs(Dico['left_depth'])/(Dico['left_width'])+abs(Dico['right_depth'])/(-Dico['right_width']))
    factor_fwhm = np.array(Dico['w_minima']/np.min(cen_bins_reg))
    Dico['mean_grad'] *= factor_fwhm
    Dico['mean_grad'] /= np.max(Dico['mean_grad'])
    
    #=============================================================================
    # Line position selection
    #=============================================================================
    if mask_dic['verbose']:print('           Selection: position')
    
    #RV window for line position fit (km/s)
    if mask_dic['win_core_fit'] is None:win_core_fit=1.
    else:win_core_fit = mask_dic['win_core_fit']
    
    #Fitting line central wavelength
    coordinates = np.zeros((len(np.array(Dico['idx_minima'])),7))
    loop = 0
    continuum_line = 1.
    for j in np.array(Dico['idx_minima']):
        try:

            #Local line spectrum             
            win_core_pix = int((cen_bins_reg[j]*win_core_fit/c_light)*(1./np.diff(cen_bins_reg[j:j+2])))
            grid_line = cen_bins_reg[j-win_core_pix:j+win_core_pix+1]
            spectrum_line = flux_norm_reg[j-win_core_pix:j+win_core_pix+1]
            
            #Local normalization
            line = myc.tableXY(grid_line, spectrum_line/continuum_line, np.sqrt(abs(spectrum_line))/continuum_line)
            
            #Line centered around its mean in X and Y
            line.recenter()
            
            #Line fitted with parabola
            line.fit_poly(d=2)
            line.interpolate(replace=False)
            
            #Estimated wavelength of line minimum
            center = -0.5 * line.poly_coefficient[1]/line.poly_coefficient[0]
            center = center + line.xmean
            coordinates[loop,0] = center   
            
            #Estimated line minimum  
            line_minimum = np.polyval(line.poly_coefficient, center) + line.ymean
            
            #Estimated line depth              
            coordinates[loop,1] = 1. - line_minimum
            coordinates[loop,2] = 1 - (line.y.min() + line.ymean)
            coordinates[loop,3] = 1 - (line.y_interp.min() + line.ymean)
            
            #Estimated errors
            mini1 = np.argmin(spectrum_line) 
            coordinates[loop,4] = np.sqrt(abs(spectrum_line[mini1]*(1+(spectrum_line[mini1]/continuum_line)**2)))/abs(continuum_line)
            errors = np.sqrt(np.diag(line.cov))
            coordinates[loop,5] = 0.5*np.sqrt((errors[1]/line.poly_coefficient[0])**2+(errors[0]*line.poly_coefficient[1]/line.poly_coefficient[0]**2)**2)
            coordinates[loop,6] = line.chi2
        except:
           pass 
        loop+=1

        #To visualize a fit
        # plt.plot(grid_line,spectrum_line,color='red')
        # plt.plot(line.x+ center,np.polyval(line.poly_coefficient, line.x) + line.ymean,color='blue')
        # plt.show()
        # stop()


    #Threshold on relative diffence in line position
    if mask_dic['abs_RVdev_fit_max'] is None:abs_RVdev_fit_max=1500.
    else:abs_RVdev_fit_max = mask_dic['abs_RVdev_fit_max']        
        
    #Keep lines with relative difference in position (=RV deviation) with the measured minima below threshold
    #    - it makes more sense to use the relative difference, as it is equivalent to a RV difference (delta_rv = c*delta_w / w)
    #    - the same criterion cannot be applied to the depth because the fit is performed in the core of the line, locally, so that the fit continuum is much lower than the actual continuum
    Dico['w_fitted'] = coordinates[:,0]
    abs_RVdev_fit = c_light_m*abs(np.array((Dico['w_minima']-Dico['w_fitted'])/Dico['w_minima']))

    #Store for plotting
    if plot_RVdev_fit:
        dic_sav.update({'abs_RVdev_fit_max':abs_RVdev_fit_max,'abs_RVdev_fit':abs_RVdev_fit,'weight_rv_RVdev_fit':np.array(Dico[mask_dic['mask_weights']])})

    #Restrict
    Dico = Dico.loc[abs_RVdev_fit<abs_RVdev_fit_max]
    Dico = Dico.reset_index(drop=True) 
    
    #Attribute mask line positions to minima positions
    Dico['w_lines'] = Dico['w_minima']    

    #Store for plotting
    dic_sav['sel2']={}
    if plot_spec:
        dic_sav['sel2'].update({'nl_mask_pre':dic_sav['sel1']['nl_mask_post'],'nl_mask_post':len(Dico['w_lines'])})
        for key in ['w_lines','f_minima','w_maxima_left','w_maxima_right','f_maxima_left','f_maxima_right']:dic_sav['sel2'][key] = np.array(Dico[key])

    #=============================================================================
    # MORPHOLOGICAL properties
    #=============================================================================

    #Second derivative criterion
    dw_raw = edge_bins_mast[1::]-edge_bins_mast[0:-1]
    deriv_flux_norm = np.gradient(flux_norm)/dw_raw
    deriv2_flux_norm = np.gradient(deriv_flux_norm)/dw_raw
    deriv2_flux = bind.resampling(edge_bins_reg, edge_bins_mast,deriv2_flux_norm, kind=gen_dic['resamp_mode'])            
    
    #Kernel smoothing length
    if (inst in mask_dic['kernel_smooth_deriv2']):kernel_smooth_deriv2 = mask_dic['kernel_smooth_deriv2'][inst]
    else:
        if inst=='ESPRESSO':kernel_smooth_deriv2 = 15  
        else:stop('Define kernel_smooth_deriv2')     

    #Kernel profile
    kernel_prof_deriv2 = 'gaussian'
    
    #Smoothing second derivative   
    deriv2_flux = myf.smooth(deriv2_flux, kernel_smooth_deriv2, shape = kernel_prof_deriv2)
        
    dd_min = np.where((np.in1d(np.arange(len(cen_bins_reg)),argrelextrema(deriv2_flux,np.less)[0])&(deriv2_flux<0))==True)[0]    # find all dd minima with dd<0 (clue to define the windows later)
    dd_min = np.hstack([0,dd_min,len(cen_bins_reg)])    
    distance = dd_min-np.array(Dico['idx_minima'])[:,np.newaxis]
    
    pos = np.array([np.argmax(distance[j,distance[j,:]<0]) for j in range(len(distance))])
    dd_left = dd_min[pos]
    dd_right = dd_min[pos+1]
    
    #Fix the border to min width if no minima
    for j in range(len(dd_left)):
        if dd_left[j]<Dico.loc[j,'idx_maxima_left']:
            dd_left[j]=Dico.loc[j,'idx_maxima_left']
        if dd_right[j]>Dico.loc[j,'idx_maxima_right']:
            dd_right[j]=Dico.loc[j,'idx_maxima_right']

    #Maximum difference of flux between the line center and one of the maxima
    Dico['max_depth'] = np.max(np.vstack([abs(Dico['left_depth']),abs(Dico['right_depth'])]),axis=0) 
    dd_max = np.where((np.in1d(np.arange(len(cen_bins_reg)),argrelextrema(deriv2_flux,np.greater)[0])&(deriv2_flux>0))==True)[0] # find all dd maxima with dd>0 (clue to identify blends later)
    save_diff = dd_max-dd_right[:,np.newaxis]
    save_diff2 = dd_max-dd_left[:,np.newaxis]    
    
    #Number of maxima with dd>0 between the two dd minima (blend clue)
    Dico['num_dd_max'] = np.sum((save_diff*save_diff2)<0,axis=1) 
    
    
    Dico['asym_ddflux_norm'] = (flux_norm_reg[dd_left] - flux_norm_reg[dd_right])/Dico['max_depth']
    
    #Half-range covered by the line
    Dico['min_width_dd'] = np.min(np.vstack([abs(Dico['w_lines']-cen_bins_reg[dd_left]),abs(Dico['w_lines']-cen_bins_reg[dd_right])]),axis=0)
    
    #Final half-range covered by the line
    Dico['line_hrange'] = Dico[['min_width_dd','min_width']].mean(axis=1)



    #=============================================================================
    #Computation of telluric contamination
    #=============================================================================
    if tell_spec is not None:
        if mask_dic['verbose']:print('           Selection: tellurics') 
        
        #Resampling on regular grid
        spectre_t = bind.resampling(edge_bins_reg, edge_bins_mast, tell_spec, kind=gen_dic['resamp_mode']) 
        
        #Minima of telluric spectrum in the Earth rest frame
        #    - only telluric deeper than threshold are considered
        indext_minima, fluxt_minima = myf.local_max(-spectre_t,vicinity_reg)
        fluxt_minima = -fluxt_minima
        indext_minima = indext_minima.astype('int')
        if mask_dic['tell_depth_min'] is None:tell_depth_min = 0.001
        else:tell_depth_min = mask_dic['tell_depth_min']
        mask = (fluxt_minima<1.-tell_depth_min)
        wavet_minima = cen_bins_reg[indext_minima][mask]
        indext_minima = indext_minima[mask]
        fluxt_minima = fluxt_minima[mask] 
        
        #Exclusion of stellar lines overlapping with deep telluric lines
        #    - the master stellar spectrum is aligned in the star (for disk-integrated profiles) or surface (for intrinsic profiles) rest frames
        #      we thus shift telluric minima from the Earth (source) to these receiver frames as:
        # w_receiver = w_source * (1+ rv[s/r]/c))  
        # w_star = w_Earth * (1+ rv[Earth/solbar]/c)) * (1+ rv[solbar/starbar]/c)) * (1+ rv[starbar/star]/c)) 
        # w_star = w_Earth * (1+ BERV/c)) * (1 - rv_sys/c)) * (1 - rv_kepl/c))  
        #      or 
        # w_photo = w_star  * (1+ rv[star/photo]/c))  
        #    - we define the maximum and minimum shift of the telluric lines over the processed visits, accounting for a margin of 4kms to account for telluric width
        min_dopp_shift = min_specdopshift_receiver_Earth*gen_specdopshift(-4.)
        max_dopp_shift = max_specdopshift_receiver_Earth*gen_specdopshift(4.) 
        mean_dopp_shift = 0.5*(min_dopp_shift+max_dopp_shift)

        #Processing each stellar line in the mask   
        if mask_dic['tell_star_depthR_min'] is None:tell_star_depthR_min = 0.001
        else:tell_star_depthR_min = mask_dic['tell_star_depthR_min']               
        wave_tel = [] ; depth_tel = [] 
        for l_min_depth,l_center,l_win in zip(np.array(Dico['min_depth']),np.array(Dico['w_lines']),np.array(Dico['line_hrange'])):

            #Processing telluric lines deeper than current stellar line by chosen threshold
            mask2 = ((1.-fluxt_minima)/l_min_depth) > tell_star_depthR_min
            wavet_minima_l =   wavet_minima[mask2]
           
            #Min and max position of telluric line during the visit
            #    - shifted from the Earth (source) to the star (receiver) rest frame
            wmax_t = wavet_minima_l*max_dopp_shift
            wmin_t = wavet_minima_l*min_dopp_shift
            
            #Conditions for overlap 
            c1 = np.sign((l_center+l_win)-wmin_t)   
            c2 = np.sign((l_center+l_win)-wmax_t)
            c3 = np.sign((l_center-l_win)-wmin_t)
            c4 = np.sign((l_center-l_win)-wmax_t)
            if np.product((c1==c2)*(c3==c4)*(c1==c3)*(c1==c4)*(c2==c4))==1:
                wave_tel.append(np.nan)
                depth_tel.append(0)
            else:
                
                #Overlapping telluric lines
                #    - at least one of the condition must not be verified, ie:
                # + c1 != c2 : 
                #       (l_center+l_win)>wmin_t & (l_center+l_win)<wmax_t = min tell < right bound line & max tell > right bound line = tell line 'crosses' right bound line during visit
                #       (l_center+l_win)<wmin_t & (l_center+l_win)>wmax_t = not possible
                # + c3 != c4 : 
                #       (l_center-l_win)>wmin_t & (l_center-l_win)<wmax_t = min tell < left bound line & max tell > left bound line = tell line 'crosses' left bound line during visit
                #       (l_center-l_win)<wmin_t & (l_center-l_win)>wmax_t = not possible
                # + c1 != c3 : 
                #       (l_center+l_win)>wmin_t & (l_center-l_win)<wmin_t = min tell < right bound line & min tell > left bound line = tell line within line at minimum   
                #       (l_center+l_win)<wmin_t & (l_center-l_win)>wmin_t = not possible                     
                # + c2 != c4 : 
                #       (l_center+l_win)>wmax_t & (l_center-l_win)<wmax_t = max tell < right bound line & max tell > left bound line = tell line within line at maximum   
                #       (l_center+l_win)<wmax_t & (l_center-l_win)>wmax_t = not possible  
                # + c1 != c4 : 
                #       (l_center+l_win)>wmin_t & (l_center-l_win)<wmax_t = min tell < right bound line & max tell > left bound line = tell line within line during visit      
                #       (l_center+l_win)<wmin_t & (l_center-l_win)>wmax_t = not possible
                loc_tellu = np_where1D((c1==c2)*(c3==c4)*(c1==c3)*(c1==c4)*(c2==c4) == False)
                fluxt_minima_l = fluxt_minima[mask2]

                #Deepest overlapping line
                max_cont = (1.-fluxt_minima_l)[loc_tellu].argmax()
                wave_tel.append(mean_dopp_shift*wavet_minima_l[loc_tellu][max_cont])
                depth_tel.append((1.-fluxt_minima_l)[loc_tellu].max()) 
                
        #Excluding contaminated stellar lines
        #    - condition is for the ratio between the deepest overlapping telluric line and the stellar line to be larger than a threshold
        if mask_dic['tell_star_depthR_max'] is None:tell_star_depthR_max=0.2
        else:tell_star_depthR_max = mask_dic['tell_star_depthR_max']        
        wave_tel = np.array(wave_tel)
        depth_tel = np.array(depth_tel)
        Dico['rel_contam'] = depth_tel/Dico['line_depth']
  
        #Store for plotting 
        if plot_tellcont:
            dic_sav.update({'rel_contam':deepcopy(Dico['rel_contam']),'tell_star_depthR_max':tell_star_depthR_max,'weight_rv_tellcont':np.array(Dico[mask_dic['mask_weights']])})

        #Remove contaminated lines
        cond_clean = Dico['rel_contam']<tell_star_depthR_max
        Dico = Dico.loc[cond_clean]
        Dico = Dico.reset_index(drop=True)

        #-----------------------------------------------------------------------------
        #Store for plotting
        #-----------------------------------------------------------------------------       
        dic_sav['sel3']={}
        if plot_spec:
            idx_contam = np_where1D(~cond_clean)   #Contaminating telluric lines
            w_tell_contam = wave_tel[idx_contam]
            f_tell_contam = 1.-depth_tel[idx_contam]
            dic_sav['sel3'].update({'nl_mask_pre':dic_sav['sel2']['nl_mask_post'],'nl_mask_post':len(Dico['w_lines']),'spectre_t':spectre_t,'min_dopp_shift':min_dopp_shift,'max_dopp_shift':max_dopp_shift,
                                    'w_tell_contam':w_tell_contam,'f_tell_contam':f_tell_contam,'tell_depth_min':tell_depth_min})                                
            for key in ['w_lines','f_minima','w_maxima_left','w_maxima_right','f_maxima_left','f_maxima_right']:dic_sav['sel3'][key] = np.array(Dico[key])        
    


    
    #=============================================================================
    # VALD CROSS MATCHING 
    #    - the master stellar spectrum is already at rest, as the VALD spectrum
    #    - care must however be taken about retrieving the VALD spectrum in air or vacuum (as chosen in settings)
    #=============================================================================
    if mask_dic['VALD_linelist'] is not None:
        if mask_dic['verbose']:print('           Selection: VALD') 
            
        #Upload VALD linelist
        from pysme.linelist.vald import ValdFile
        vald_table = ValdFile(mask_dic['VALD_linelist'])
        
        #Reduce to spectrum range
        cond_within = (vald_table['wlcent']>=np.min(Dico['w_lines'])) & (vald_table['wlcent']<=np.max(Dico['w_lines']))
        vald_table = vald_table[cond_within]

        #Remove VALD duplicates
        idx_unique = np.unique(vald_table['wlcent'],return_index=True)[1]
        vald_table = vald_table[idx_unique]  
        if len(vald_table)==0:stop('No VALD lines in processed range')

        #Correction of VALD line depths
        vald_table=pd.DataFrame(vald_table)
        vald_table['depth_corrected'] = vald_table['depth']
        if mask_dic['VALD_depth_corr']:
 
            #Condition for VALD lines to overlap with mask stellar lines 
            vald_wave = np.array(vald_table['wlcent'])
            cond_stl_blend = (vald_wave>np.array(Dico['w_maxima_left'])[:,np.newaxis]) & (vald_wave<np.array(Dico['w_maxima_right'])[:,np.newaxis])   #dimensions n_st_lines x n_vald_lines
                    
            #Number of blended (=more than 1) VALD lines overlapping with each stellar line
            Dico['nb_blends'] = np.sum(cond_stl_blend,axis=1)-1     #dimension n_st_lines
            cond_unblended = (Dico['nb_blends']==0)
            if np.sum(cond_unblended)==0:stop('No unblended VALD lines')

            #Stellar lines in mask overlapping with a single VALD line
            #    - at a given row for stellar line i: 
            # [ ( vald_l[0] , vald_l[1], ..  ) with a single k where vald_l[k] = True = overlapping with i ] x np.arange(n_k) = [ 0, 0, .. k .. 0] 
            idx_vald2stline = np.sum( cond_stl_blend[cond_unblended] * np.arange(len(vald_wave)),axis=1) 

            #Estimated dispersion in stellar line depth
            rassine_accuracy = 0.01+0.07*np.array(Dico['w_lines']<4500.)
            line_depth_std = np.sqrt(2.*abs(Dico['line_depth']))
            Dico['line_depth_std'] = np.sqrt(line_depth_std**2. + rassine_accuracy**2) 

            #Dictionary of mask lines associated with a single VALD line
            #    - we attribute to the mask line the properties of the associated VALD line
            Dico_unblended = Dico.loc[cond_unblended]
            Dico_unblended = Dico_unblended.reset_index(drop=True)
            Dico_unblended['species'] = np.array(vald_table.loc[idx_vald2stline,'species'])     
            Dico_unblended['log_gf'] = np.array(vald_table.loc[idx_vald2stline,'gflog'])
            Dico_unblended['vald_depth'] = np.array(vald_table.loc[idx_vald2stline,'depth'])
            
            #Difference in RV position (m/s) and depth between VALD and stellar lines
            diff_vald_wave = np.array(Dico_unblended['w_lines']) - vald_wave[idx_vald2stline]
            Dico_unblended['diff_vald_RV'] = c_light_m*diff_vald_wave/Dico_unblended['w_lines']
            Dico_unblended['diff_vald_depth'] = Dico_unblended['line_depth'] - Dico_unblended['vald_depth']
            Dico_unblended['diff_vald_depth_rel'] = np.ones(len(Dico_unblended['vald_depth']),dtype=float)*1e100
            
            #Typical window of variation in RV position (m/s) between VALD and stellar lines
            med_diff_vald_RV = np.median(Dico_unblended['diff_vald_RV'])
            IQ_diff_vald_RV = 1.5*myf.IQ(Dico_unblended['diff_vald_RV'])
            typ_shift = [med_diff_vald_RV-IQ_diff_vald_RV,med_diff_vald_RV+IQ_diff_vald_RV]
           
            #Removing stellar lines deviating too far from VALD lines, or overlapping with shallow VALD lines
            Dico_unblended = Dico_unblended.loc[(np.abs(Dico_unblended['diff_vald_RV'])<1500.) & (Dico_unblended['vald_depth']>0.02)]
            Dico_unblended = Dico_unblended.reset_index(drop=True)
            
            #Number of lines for each species associated with matching VALD lines
            count_species = Dico_unblended['species'].value_counts()   
            
            #Species with more than 20 VALD lines
            species_kept = count_species.loc[count_species>=20].keys()
        
            #Correcting all lines of retained species
            if len(species_kept):
                if plot_vald_depthcorr:dic_plot_vald={}
                
                #Degree of polynomial for depth correction
                deg_depth_corr = 3
                
                for idx_plot,elem in enumerate(species_kept):
                    
                    #Stellar lines associated with current species
                    cond_mask_sp = Dico_unblended['species']==elem
                    
                    #Depth of matching VALD lines
                    vald_depth = Dico_unblended.loc[cond_mask_sp,'vald_depth']
                    line_depth =  Dico_unblended.loc[cond_mask_sp,'line_depth']
                    
                    #Difference between stellar and VALD line depths
                    line_minus_vald_depth = Dico_unblended.loc[cond_mask_sp,'diff_vald_depth']
                    
                    #Estimated error on line depth difference
                    line_minus_vald_depth_std = Dico_unblended.loc[cond_mask_sp,'line_depth_std']
             
                    #Prepare data for fit
                    #    - removing vald lines with depth difference below 3% and outliers
                    line = myc.tableXY(vald_depth,line_minus_vald_depth,line_minus_vald_depth_std)
                    line.clip(min=[0.03,None],replace=True)
                    line.rm_outliers()
                    mask = myf.rm_outliers(line.yerr,m=2, kind='inter')[0]
                    line.masked(mask)
                    
                    #Adding mock line with depth unity and no deviation to act as reference
                    line.x = np.insert(line.x,len(line.x),1) 
                    line.y = np.insert(line.y,len(line.y),0) 
                    line.yerr = np.insert(line.yerr,len(line.yerr),0.001) 
                    line.xerr = np.insert(line.xerr,len(line.xerr),0) 

                    #Fit depth difference on matching VALD lines
                    #    - errors are scaled after a preliminary fit to ensure reduced chi2 unity
                    line.fit_poly(Draw=False,d=deg_depth_corr,color='k')
                    red_chi2 = line.chi2/(len(line.x) - (deg_depth_corr+1.))
                    line.yerr = np.sqrt(red_chi2)*line.yerr
                    line.fit_poly(Draw=False,d=deg_depth_corr,color='k')  

                    #All VALD lines associated with current species
                    cond_vald_sp = vald_table['species']==elem
                    
                    #Depth correction
                    vald_sp_depth = np.array(vald_table.loc[cond_vald_sp,'depth'])
                    delta_depth = np.polyval(line.poly_coefficient,vald_sp_depth)
                    vald_table.loc[cond_vald_sp,'depth_corrected'] = vald_sp_depth + delta_depth
                    
                    #Relative variation in depth between matching VALD and stellar lines
                    Dico_unblended.loc[cond_mask_sp,'diff_vald_depth_rel'] = ((vald_depth + np.polyval(line.poly_coefficient,vald_depth)  )/line_depth) - 1.
                    
                    #Store for plotting
                    if plot_vald_depthcorr:
                        dic_plot_vald[elem] = {'vald_depth_fit':line.x,'line_minus_vald_depth_fit':line.y,'line_minus_vald_depth_err_fit':line.yerr,
                                               'deltadepth_mod_coeff':line.poly_coefficient,'vald_depth':vald_depth,'line_minus_vald_depth':line_minus_vald_depth,'w_lines':np.array(Dico_unblended.loc[cond_mask_sp,'w_lines'])}
    
            #Corrected line depth set to 0 (resp. 1) if lower than 0 (resp. larger than 1)
            vald_table.loc[vald_table['depth_corrected']<0,'depth_corrected']=0
            vald_table.loc[vald_table['depth_corrected']>1,'depth_corrected']=1

            #Typical window of variation in depth between VALD and stellar lines
            #    - defined after depth correction 
            cond_depth_corr = Dico_unblended['diff_vald_depth_rel']<1e50
            diff_vald_depth_rel=Dico_unblended['diff_vald_depth_rel'][cond_depth_corr]
            typ_depthvar_rel = [np.median(diff_vald_depth_rel)-1.5*myf.IQ(diff_vald_depth_rel),
                                np.median(diff_vald_depth_rel)+1.5*myf.IQ(diff_vald_depth_rel)]

            #Store for plotting
            if plot_vald_depthcorr:dic_sav.update({'vald_depthcorr':dic_plot_vald})

        #No adjustment of VALD lines
        else:
            typ_shift = [-2000.,2000.]
            typ_depthvar_rel = [-0.5,0.5]            

        #Condition for VALD lines to overlap with mask stellar lines 
        vald_wave = np.array(vald_table['wlcent'])
        n_maskline = len(Dico['w_lines'])
        cond_stl_blend = (vald_wave>np.array(Dico['w_lines']-Dico['line_hrange'])[:,np.newaxis])*(vald_wave<np.array(Dico['w_lines']+Dico['line_hrange'])[:,np.newaxis])
        Dico['nb_blends'] = np.sum(cond_stl_blend,axis=1)-1
        
        #Processing all stellar lines overlapping with VALD lines
        vald_depth = np.array(vald_table['depth_corrected'])  
        matching_VALD_line_index = []
        contam = []
        delta_RV = []
        delta_depth_rel = []
        for j in range(n_maskline):
   
            #Index of VALD lines overlapping with current stellar lines
            loc_blend = np_where1D(cond_stl_blend[j])
            
            #Difference in position between the stellar line and each VALD line
            diff_wave = Dico.loc[j,'w_lines']-vald_wave[loc_blend]
            diff_RV = c_light_m*diff_wave/Dico.loc[j,'w_lines']
            
            #Relative difference in depth between the stellar line and each VALD line
            diff_vald_depth_rel = (vald_depth[loc_blend]/Dico.loc[j,'line_depth'])-1.

            #Main VALD lines overlapping with stellar line (position within window around stellar line)
            matching_VALD_line = (diff_RV>typ_shift[0])&(diff_RV<typ_shift[1]) & (diff_vald_depth_rel>typ_depthvar_rel[0])&(diff_vald_depth_rel<typ_depthvar_rel[1]) 
            if sum(matching_VALD_line):

                #VALD line most consistent with current stellar line (in position and depth)
                final_matching_VALD_line = np.argmin(np.sqrt(diff_RV[matching_VALD_line]**2+diff_vald_depth_rel[matching_VALD_line]**2))
                
                #Storing global VALD line index
                matching_VALD_line_index.append(loc_blend[matching_VALD_line][final_matching_VALD_line])
                
                #Storing delta-position between stellar and VALD line
                delta_RV.append(diff_RV[matching_VALD_line][final_matching_VALD_line])
                delta_depth_rel.append(diff_vald_depth_rel[matching_VALD_line][final_matching_VALD_line])
              
                #Flagging blended VALD line (ie, VALD lines other than the main one overlapping with current stellar line)
                if len(loc_blend)>1:
                    
                    #Setting to False main overlapping VALD line and True all others 
                    blend_line = np.ones(len(loc_blend)).astype('bool')
                    blend_line[ np.arange(len(loc_blend))[matching_VALD_line][final_matching_VALD_line] ] = False

                    #Maximum depth of blended VALD lines
                    highest_blend = np.max(vald_depth[loc_blend][blend_line])
                    
                    #Ratio between deepest blended VALD line and stellar line
                    contam.append(highest_blend/Dico.loc[j,'line_depth'])
      
                else:
                    contam.append(0)

            #No VALD line consistent with stellar line
            else:
                matching_VALD_line_index.append(-99.9)
                contam.append(-99.9)
                delta_RV.append(np.nan)
                delta_depth_rel.append(np.nan)
      
        #Flag stellar lines deviating from matching VALD line distribution
        delta_RV = np.array(delta_RV)
        delta_depth_rel = np.array(delta_depth_rel)
        not_outliers = myf.rm_outliers(delta_RV,m=2,kind='inter')[0] & myf.rm_outliers(delta_depth_rel,m=2,kind='inter')[0]
        
        #Indexes of matching VALD lines in original VALD linelist
        #    - set to -99.9 if associated with an outlying stellar line (ie, not consistent with VALD lines)
        matching_VALD_line_index = np.array(matching_VALD_line_index)
        matching_VALD_line_index[np.arange(n_maskline)[~not_outliers]] = -99.9
       
        #Exclusion of matching VALD lines with blends
        #    - VALD line is discarded if there is at least one other VALD line overlapping with the matched stellar line, and the deepest one (amongst the blends) is deeper than the stellar line by the threshold
        matching_VALD_line_index[np.array(contam)>0.1] = -99.9
       
        #Associate VALD line properties with mask lines
        kw      = ['wave_vald','depth_vald','depth_vald_corrected','species','atomic_number','log_gf','E_up','J_low','J_up','lande_low','lande_up','lande_mean','damp_rad','damp_stark','damp_waals','ionisation_energy']
        kw_vald = ['wlcent'   ,'depth'     ,'depth_corrected',     'species','atom_number'  ,'gflog','e_upp','j_lo','j_up','lande_lower','lande_upper','lande','gamrad','gamqst','gamvw','ionization']
        for keyword in kw:
            Dico[keyword] = np.nan
        i=-1
        for j in matching_VALD_line_index:
            i+=1
            if j !=-99.9:
                for keyword,keyword_vald in zip(kw,kw_vald):
                    Dico.loc[i,keyword] = vald_table.loc[j,keyword_vald]

        #Store for plotting     
        if plot_spec:
            dic_sav.update({'wave_vald':np.array(Dico['wave_vald']),'depth_vald_corr':np.array(Dico['depth_vald_corrected']),'species':np.array(Dico['species'])})                                

    
    #=============================================================================
    # MORPHOLOGICAL CLIPPING 1 
    #=============================================================================
    if mask_dic['verbose']:print('           Selection: morphological asymmetry') 
    
    #Threshold
    if (mask_dic['diff_cont_rel_max'] is not None):diff_cont_rel_max = mask_dic['diff_cont_rel_max']
    else:diff_cont_rel_max = 1. 
    if (mask_dic['asym_ddflux_max'] is not None):asym_ddflux_max = mask_dic['asym_ddflux_max']
    else:asym_ddflux_max = 0.25
    
    #Absolute flux difference between the two maxima normalized by the maximum flux difference between line center and one of the maxima
    diff_continuum = abs(Dico['f_maxima_left']-Dico['f_maxima_right'])/Dico['max_depth'] 
    
    #Absolute flux difference between line center and mean maxima
    Dico['diff_depth'] = abs(Dico['f_minima']-np.mean([Dico['f_maxima_left'],Dico['f_maxima_right']],axis=0))
    
    #Ratio of normalized continuum difference and relative depth
    diff_continuum_rel = deepcopy(diff_continuum/Dico['diff_depth'])   

    #Remove lines with morphological properties beyond threshold
    asym_ddflux_norm = deepcopy(abs(Dico['asym_ddflux_norm']))
    mask = (diff_continuum_rel<diff_cont_rel_max) & (asym_ddflux_norm<asym_ddflux_max)
    
    #Store for plotting    
    if plot_morphasym:
        dic_sav.update({'diff_continuum_rel':diff_continuum_rel,'abs_asym_ddflux_norm':asym_ddflux_norm,
                        'diff_cont_rel_max':diff_cont_rel_max,'asym_ddflux_max':asym_ddflux_max,'weight_rv_morphasym':np.array(Dico[mask_dic['mask_weights']])})
    
    #Remove lines with blends
    mask &= Dico['num_dd_max']==1
    Dico = Dico.loc[mask]
    Dico = Dico.reset_index(drop=True)

    #Store for plotting    
    dic_sav['sel4']={}
    if plot_spec:
        dic_sav['sel4'].update({'nl_mask_pre':dic_sav['sel3']['nl_mask_post'],'nl_mask_post':len(Dico['w_lines'])})                                
        for key in ['w_lines','f_minima','w_maxima_left','w_maxima_right','f_maxima_left','f_maxima_right']:dic_sav['sel4'][key] = np.array(Dico[key])        



    
    #=============================================================================
    # MORPHOLOGICAL CLIPPING 2 
    #=============================================================================
    if mask_dic['verbose']:print('           Selection: morphological shape')
        
    #Thresholds
    if (mask_dic['diff_depth_min'] is not None):diff_depth_min = mask_dic['diff_depth_min']
    else:diff_depth_min = 0.15 
    if (mask_dic['width_max'] is not None):width_max = mask_dic['width_max']
    else:width_max = 13.0
    
    #Line width (km/s)
    Dico['width_kms'] = Dico['line_hrange']*c_light/Dico['w_lines']

    #Store for plotting
    if plot_morphshape:
        dic_sav.update({'diff_depth':Dico['diff_depth'],'width_kms':Dico['width_kms'],'diff_depth_min':diff_depth_min,'width_max':width_max,'weight_rv_morphshape':np.array(Dico[mask_dic['mask_weights']])})

    #Remove lines with morphological properties beyond threshold  
    #    - absolute flux difference between line center and mean maxima, line width (km/s)
    mask = (Dico['diff_depth']>diff_depth_min)&(Dico['width_kms']<width_max)
    Dico = Dico.loc[mask]
    Dico = Dico.reset_index(drop=True)
    
    #-----------------------------------------------------------------------------
    #Store for plotting
    #-----------------------------------------------------------------------------       
    dic_sav['sel5']={}
    if plot_spec:
        dic_sav['sel5'].update({'nl_mask_pre':dic_sav['sel4']['nl_mask_post'],'nl_mask_post':len(Dico['w_lines'])})                                
        for key in ['w_lines','f_minima','w_maxima_left','w_maxima_right','f_maxima_left','f_maxima_right']:dic_sav['sel5'][key] = np.array(Dico[key])        




    


    #=============================================================================
    # Selection based on RV dispersion
    #=============================================================================
    if mask_dic['RV_disp_sel']:
        if mask_dic['verbose']:print('           Selection: RV dispersion')
        
        #Number of iterations on the line fit
        #    - ensures better convergence
        Niter = 3
        
        #Size of window to match line position 
        #    - full size is 2*win_size*(half-line range)
        win_size = 2.

        #Isolate ranges for each line in master spectrum
        nlines = len(np.array(Dico['w_lines']))
        wav_mast_lineranges = []
        flux_mast_lineranges = []
        cond_def_lines = np.ones(nlines,dtype=bool)
        for iline,(wline_loc,line_hrange) in enumerate(zip(np.array(Dico['w_lines']),np.array(Dico['line_hrange']))):
        
            #Range of the line
            #    - the master window must be larger to account for the shift applied to the spectrum while searching for its RV
            cond_line_in_mast = (cen_bins_reg>  wline_loc - 2.*win_size*line_hrange) & (cen_bins_reg< wline_loc + 2.*win_size*line_hrange)
            
            #Check for nans
            if (np.sum(cond_line_in_mast)>0) and (np.sum(flux_norm_reg[cond_line_in_mast]==0.)==0):
                wav_mast_lineranges+=[cen_bins_reg[cond_line_in_mast]]
                flux_mast_lineranges+=[flux_norm_reg[cond_line_in_mast]]                
            else:cond_def_lines[iline] = False
                
        #Exclude mask lines that cannot be assessed
        Dico = Dico.loc[cond_def_lines]
        Dico = Dico.reset_index(drop=True)      

        #Process requested exposures
        nlines = len(np.array(Dico['w_lines']))
        RV_tab = np.empty([2,nlines,0],dtype=float)
        for vis in vis_iexp_in_bin:
            if (inst in mask_dic['idx_RV_disp_sel']) and (vis in mask_dic['idx_RV_disp_sel'][inst]) and len(mask_dic['idx_RV_disp_sel'][inst][vis])>0:idx_RV_disp_sel = mask_dic['idx_RV_disp_sel'][inst][vis] 
            else:
                if data_type_gen=='DI':idx_RV_disp_sel = gen_dic[inst][vis]['idx_out'] 
                elif data_type_gen=='Intr':idx_RV_disp_sel = data_dic['Intr'][inst][vis]['idx_def']    
            idx_RV_disp_sel = np.intersect1d(idx_RV_disp_sel,list(vis_iexp_in_bin[vis].keys()))

            #Processing all exposures
            common_args = (Niter,win_size,nlines,data_dic[inst][vis]['proc_'+data_type_gen+'_data_paths'],cont_func_dic,np.array(Dico['w_lines']),np.array(Dico['line_hrange']),wav_mast_lineranges,flux_mast_lineranges)
            if mask_dic['nthreads']>1: RV_tab_vis = para_RV_LBL(RV_LBL,mask_dic['nthreads'],len(idx_RV_disp_sel),[idx_RV_disp_sel],common_args)                           
            else: RV_tab_vis = RV_LBL(idx_RV_disp_sel,*common_args)  
            RV_tab = np.append(RV_tab,RV_tab_vis,axis=2)   
                 
        #Calculate weighted average of RV, dispersion, and dispersion/mean error, for each line
        #    - beware in cases where the number of out-of-transit spectra, and thus of RV measurements for each line, is too low to analyze the RV distribution and errors of a single line 
        av_RV_lines = np.ones(nlines,dtype=float)*1e10
        disp_RV_lines = np.ones(nlines,dtype=float)*1e10
        disp_err_RV_lines = np.ones(nlines,dtype=float)*1e10
        for iline in range(nlines):
            cond_def = (~np.isnan(RV_tab[0,iline,:])) & (~np.isnan(RV_tab[1,iline,:])) & (~np.isinf(RV_tab[0,iline,:])) & (~np.isinf(RV_tab[1,iline,:])) 
            if np.sum(cond_def)>0:
            
                #Weighted average (m/s)
                wRV_line = 1./RV_tab[1,iline,cond_def]**2.
                av_RV_lines[iline] = np.sum(RV_tab[0,iline,cond_def]*wRV_line)/np.sum(wRV_line)
            
                #Dispersion to error ratio
                disp_RV_lines[iline] = np.std(RV_tab[0,iline,cond_def])
                mean_eRV = np.mean(RV_tab[1,iline,cond_def])
                disp_err_RV_lines[iline] = disp_RV_lines[iline]/mean_eRV

        #Dispersion thresholds
        if (inst in mask_dic['absRV_max']):absRV_max = mask_dic['absRV_max'][inst]
        else:absRV_max={'ESPRESSO':50}[inst]
        if (inst in mask_dic['RVdisp_max']):RVdisp_max = mask_dic['RVdisp_max'][inst]
        else:
            RVdisp_max={'ESPRESSO':100.}
            if inst not in RVdisp_max:stop('Add '+inst+' to "RVdisp_max" in KitCat_main.py')
            RVdisp_max=RVdisp_max[inst]              
        if (inst in mask_dic['RVdisp2err_max']):RVdisp2err_max = mask_dic['RVdisp2err_max'][inst]
        else:RVdisp2err_max=10.     
        
        #Store for plotting
        abs_av_RV_lines = np.abs(av_RV_lines)
        if plot_RVdisp:
            dic_sav.update({'nexp4lineRV':RV_tab.shape[2],'abs_av_RV_lines':abs_av_RV_lines,'disp_err_RV_lines':disp_err_RV_lines,'disp_RV_lines':disp_RV_lines,'absRV_max':absRV_max,'RVdisp2err_max':RVdisp2err_max,'RVdisp_max':RVdisp_max,'weight_rv_RVdisp':np.array(Dico[mask_dic['mask_weights']])})        
        
        #Exclude from mask lines with absolute RV and RV dispersion/error beyond threshold
        #    - RV should be well spread around 0, since exposures and master are aligned in the star frame, and have dispersion and error comparable over the time series
        mask = (abs_av_RV_lines<absRV_max) & (disp_err_RV_lines<RVdisp2err_max) & (disp_RV_lines<RVdisp_max)
        Dico = Dico.loc[mask]
        Dico = Dico.reset_index(drop=True)

    #Final exclusion of stellar lines contaminated by tellurics
    if tell_spec is not None:
        if mask_dic['tell_star_depthR_max_final'] is None:tell_star_depthR_max_final=0.03
        else:tell_star_depthR_max_final = mask_dic['tell_star_depthR_max_final']          
        if plot_tellcont:
            dic_sav.update({'rel_contam_final':deepcopy(Dico['rel_contam']),'tell_star_depthR_max_final':tell_star_depthR_max_final,'weight_rv_tellcont_final':np.array(Dico[mask_dic['mask_weights']])})
        cond_clean = Dico['rel_contam']<tell_star_depthR_max_final
        mask_info+='_t'+"{0:.1f}".format(100.*tell_star_depthR_max_final)
        Dico = Dico.loc[cond_clean]
        Dico = Dico.reset_index(drop=True) 

    #=============================================================================    
    #Final binary mask
    #    - defined in the stellar (for disk-integrated profiles) or surface (for intrinsic profiles) rest frames
    #    - ANTARESS and ESPRESSO-like DRS use the square of the line weights as effective weights, so that we normalize them accordingly 
    #    - setting the line positions to minima positions in the oversampled specrum grid 
    #=============================================================================
    if mask_dic['verbose']:print('           Computing mask')
    mask_waves = np.array(Dico['w_lines'])
    line_weights = np.array(Dico[mask_dic['mask_weights']])
    mask_weights = line_weights/np.sqrt(np.mean(line_weights**2.))

    #-----------------------------------------------------------------------------
    #Store for plotting
    #-----------------------------------------------------------------------------       
    dic_sav['sel6']={}
    if plot_spec:
        dic_sav['sel6'].update({'nl_mask_pre':dic_sav['sel5']['nl_mask_post'],'nl_mask_post':len(Dico['w_lines']),'line_hrange':np.array(Dico['line_hrange'])})                                
        for key in ['w_lines','f_minima','w_maxima_left','w_maxima_right','f_maxima_left','f_maxima_right']:dic_sav['sel6'][key] = np.array(Dico[key])        

    return mask_waves,mask_weights,mask_info
    
    
def para_RV_LBL(func_input,nthreads,n_elem,y_inputs,common_args):
    pool_proc = Pool(processes=nthreads)   #cannot be passed through lmfit
    ind_chunk_list=init_parallel_func(nthreads,n_elem)
    chunked_args=[(y_inputs[0][ind_chunk[0]:ind_chunk[1]],)+common_args for ind_chunk in ind_chunk_list]	
    all_results=tuple(tab for tab in pool_proc.starmap(func_input,chunked_args))
    y_output=np.concatenate(tuple(all_results[i] for i in range(nthreads)),axis=2)
    pool_proc.close()
    pool_proc.join() 	    
    return y_output    
    
def RV_LBL(idx_RV_disp_sel,Niter,win_size,nlines,data_vis_paths,cont_func_dic,w_lines,line_hrange,wav_mast_lineranges,flux_mast_lineranges):
    RV_tab = np.empty([2,nlines,0],dtype=float)
    for iexp in idx_RV_disp_sel:

        #Retrieve spectra 
        #    - must be aligned in the star rest frame
        data_exp = dataload_npz(data_vis_paths+str(iexp))
    
        #Continuum-normalisation
        wav_exp  = data_exp['cen_bins'][0]
        cont_exp= 1./cont_func_dic(wav_exp)
        flux_exp_norm,cov_exp_norm = bind.mul_array(data_exp['flux'][0],data_exp['cov'][0],cont_exp)
        err_exp_norm = np.sqrt(cov_exp_norm[0])
    
        #Process line in mask  
        RV_tab_exp  = np.zeros([2,nlines])*np.nan
        for iline,(wline_loc,line_hrange_loc) in enumerate(zip(w_lines,line_hrange)):
        
            #Range of the line
            cond_line_in_spec = (wav_exp>  wline_loc - win_size*line_hrange_loc) & (wav_exp< wline_loc + win_size*line_hrange_loc)
            if (np.sum(cond_line_in_spec)>0) and (np.sum(np.isnan(flux_exp_norm[cond_line_in_spec]))==0):
                
                #Measure RV of current line in current exposure through template matching with the master
                #    - in m/s
                RV_line = 0.
                for it in range(Niter):
                    _, RV_line, RV_err_line, *_ = calculate_RV_line_by_line.get_RV_line_by_line2(wav_exp[cond_line_in_spec], err_exp_norm[cond_line_in_spec], flux_exp_norm[cond_line_in_spec],wav_mast_lineranges[iline], flux_mast_lineranges[iline], 0., RV_line, wline_loc)
                
                #Store line with physical / useful values
                if (np.abs(RV_line)<1000.) & (np.abs(RV_err_line)<1000.):RV_tab_exp[:,iline] = [RV_line,RV_err_line]

        RV_tab = np.append(RV_tab,RV_tab_exp[:,:,None],axis=2)   

    return RV_tab


    