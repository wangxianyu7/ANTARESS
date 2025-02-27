#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Romain Allart - Original code to compute resolution maps for ESPRESSO/NIRPS
@author: Sara Tavella - Updates to compute resolutions maps for HARPS and HARPS-N 

"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import os
from astropy.io import fits as pyfits
import matplotlib as mpl
import time
from lmfit import Model
from scipy.stats import median_abs_deviation as mad
import pandas as pd


'''
REGRESSION MODELS FOR RESOLUTION MAPS
Here I define different regression models I want to test to interpolate the thar-lines resolution and reproduce the resolution maps 
'''

# 1d fitting curves
def polynomial_regression_2d_1D(t, a, b, c):
    return a * t + b * t**2. + c

def polynomial_regression_3d_1D(t, a, b, c, d):
    return a * t + b * t**2. + c * t**3. + d

def polynomial_regression_4d_1D(t, a, b, c, d, e):
    return a * t + b * t**2. + c * t**3. + d * t**4 + e

def polynomial_regression_5d_1D(t, a, b, c, d, e, f):
    return a * t + b * t**2. + c * t**3. + d * t**4 + e * t**5 + f


# 2d fitting curves 
def polynomial_regression_2d(xy, a, b, c, d, e, f):
    x = xy[0]
    y = xy[1]
    return a * x + b * x**2 + c * y + d * y**2 + e * x * y + f

def polynomial_regression_3d(xy, a, b, c, d, e, f, g, h, i, l):
    x = xy[0]
    y = xy[1]
    return a * x + b * x**2. + c * x**3. + d * y + e * y**2 + f * y**3. + g * x * y + h * x**2. * y + i * y**2. * x + l

def polynomial_regression_4d(xy, a, b, c, d, e, f, g, h, i, l, m, n, o, p, q):
    x = xy[0]
    y = xy[1]
    return a * x + b * x**2. + c * x**3. + d * x**4 + e * y + f * y**2 + g * y**3. + h * y**4 + i * x * y + l * x**2. * y + m * y**2. * x + n * x ** 3. * y + o * y ** 3. * x + p * x ** 2. * y ** 2. + q

def polynomial_regression_espresso(xy, a, b, c, d, e, f, g):
    # espression used to create the resolution map in ESPRESSO (@romain)
    x = xy[0]
    y = xy[1]
    return a * x + b * x**2. + c * y + d * y**2. + e * x * y + f * x ** 2. * y ** 2. + g



def chi_2(ydata1,ydata2,err):

    if len(ydata1) == len(ydata2):
        return np.sum((ydata2 - ydata1)**2/(err**2))
    else: 
        return "error: ydata1 and ydata2 have different dimensions"

def BIC_calc(ydata,yerr,model,k):
    
    ydata = np.array(ydata)
    model = np.array(model)
    n = ydata.size
    if ydata.size == model.size:
        bic = chi_2(ydata,model,yerr) + k * np.log(n)
        return bic
    else: 
        return "error: ydata and model have different dimensions"



#####################################################################################################
#####################################################################################################
#####################################################################################################
####################                               ##################################################
####################  CALCULATING RESOLUTION MAPS  ##################################################
####################                               ##################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################

'''
FIRST STEP - CREATING THE RIGHT SHAPE FOR THE RESOLUTION MAPS 
The following functions reproduce the hard structure of the detector, accounting for the reading binning allowed for each mode. 

For ESPRESSO we have many detector mode that output different average resolution
- high resolution (UHR and HR) implies detector reading pixel by pixel, because otherwise some information is lost 
- low resolution allows for a read-binning of 4 to 8 pixel 

For HARPS and HARPS-N the reading of the detectors is done pixel per pixel for every mode of the instruments. 
- At first approximation there is no need of shaping the detector differently according to differen epochs.
- Depending on the quality of the count sampling, I might want to remove some pixels at the edges later on.
'''

############
# ESPRESSO #
############
# Some rows at the edges of the detector need to be excluded a priori 
# This depends on the epoch of the instrument as well 
# We work with 2 fibers on the VLT, each of them feeding the detector independetely
def shape_s2d_ESPRESSO(time_science,ins_mode,bin_x):
    if (ins_mode == 'SINGLEUHR') or (ins_mode == 'SINGLEHR'):
        if time_science < 58421.5:
            index_pixel_blue = np.arange(1361., 7936. + 1, 1)
            index_pixel_red = np.arange(1., 9211. + 1, 1)
        elif (time_science > 58421.5) and (time_science < 58653.1):
            index_pixel_blue = np.arange(1361., 7901. + 1, 1)
            index_pixel_red = np.arange(1., 9111. + 1, 1)
        elif time_science > 58653.1:
            index_pixel_blue = np.arange(1361., 7901. + 1, 1)
            index_pixel_red = np.arange(1., 9111. + 1, 1)

    # MULTIMR means using the 4 VLT telescopes together on the detector 
    elif (ins_mode == 'MULTIMR') and (bin_x == 4): #bin_x -> binning for det. reading
        if time_science < 58421.5:
            index_pixel_blue = np.arange(591., 4040. + 1, 1)
            index_pixel_red = np.arange(1., 4590. + 1, 1)
        elif (time_science > 58421.5) and (time_science < 58653.1):
            index_pixel_blue = np.arange(691., 3930. + 1, 1)
            index_pixel_red = np.arange(1., 4545. + 1, 1)
        elif time_science > 58653.1:
            index_pixel_blue = np.arange(691., 3930. + 1, 1)
            index_pixel_red = np.arange(1., 4545. + 1, 1)

    elif (ins_mode == 'MULTIMR') and (bin_x == 8):
        if time_science < 58421.5:
            index_pixel_blue = np.arange(296., 2020. + 1, 1)
            index_pixel_red = np.arange(1., 2295. + 1, 1)
        elif (time_science > 58421.5) and (time_science < 58653.1):
            index_pixel_blue = np.arange(346., 1965. + 1, 1)
            index_pixel_red = np.arange(1., 2275. + 1, 1)
        elif time_science > 58653.1:
            index_pixel_blue = np.arange(346., 1965. + 1, 1)
            index_pixel_red = np.arange(1., 2275. + 1, 1)

    return index_pixel_blue, index_pixel_red

############
### HARPS ##
############
# 2 detectors of 4096x4096 pixel 
def shape_s2d_HARPS(): 
    n_pixel = 4096.
    index_pixel_blue = np.arange(1., n_pixel + 1., 1)
    index_pixel_red = np.arange(1., n_pixel + 1., 1)
    return index_pixel_blue, index_pixel_red

############
## HARPS-N #
############
# 1 detector of 4096x4096 pixel 
def shape_s2d_HARPS_N(): 
    n_pixel = 4096.
    index_pixel = np.arange(1., n_pixel + 1., 1)
    return index_pixel



'''
PLOTTING FUNCTION for resolution maps
'''
# HARPS and HARPS-N
# author: Sara 
def plot_resolution_map_HARPS(excl_lines,pixels_blue,orders_blue,resolution_blue,pixels_blue_hr,matrix_resolution_blue_hr_theo,pixels_red,
                               orders_red,resolution_red,pixels_red_hr,matrix_resolution_red_hr_theo,order_sep,n_ord,save_path,res_min,res_max,
                               ins_mode,mjd,degree,Doit=False):
    n_pixel = 4096.
    if Doit==True:
        fig,ax = plt.subplots(figsize=(20, 20))
        ax.scatter(pixels_blue, orders_blue, c=resolution_blue, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        a = ax.pcolormesh(np.insert(pixels_blue_hr,0,0.), np.arange(1., order_sep+1.01, 1),matrix_resolution_blue_hr_theo, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
        ax.pcolormesh(np.insert(pixels_blue_hr,0,0.), np.arange(1., order_sep+1.01, 1),matrix_resolution_blue_hr_theo, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
        ax.scatter(pixels_red, orders_red, c=resolution_red, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        ax.pcolormesh(np.insert(pixels_red_hr,0,order_sep), np.arange(order_sep+1., n_ord+1.01, 1), matrix_resolution_red_hr_theo, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
        ax.scatter(excl_lines[0], excl_lines[1], color = 'black', label = 'excluded lines ' + str(len(excl_lines[1])))
        ax.hlines(46, xmin = 0, xmax = n_pixel, color = 'black', linestyle = '-.')
        ax.set_xlabel('Pixels', size=20)
        ax.set_ylabel('Orders', size=20)
        try:
            ax.set_title('HARPS - ' + ins_mode + ' - regression ' + degree, size=30)
        except:
            ax.set_title('HARPS - ' + ins_mode + ' - regression blue: ' + degree[0] + ', red: ' + degree[1], size=30)

        ax.legend()
        ax.tick_params(axis='both',which='both', direction='out', width=2, length=6)
        ax.set_ylim(1, n_ord+1)
        ax.set_xlim(0, n_pixel)
        cbar_ax = fig.add_axes([0.93, 0.11, 0.014, 0.873])
        axcb = plt.colorbar(mappable=a,cax=cbar_ax, format='%0.0f')
        axcb.ax.tick_params(labelsize=20)
        axcb.set_label('Resolution', size=30)
        fig.tight_layout(w_pad=0.1, h_pad=2, rect=(0.0005, 0.0001, 0.935, 0.999))
        try:
            fig.savefig(save_path + ins_mode + '_regr_' + degree + '_resolution_map_' + str(int(mjd)) + '.png', dpi=100, format='png', bbox_inches='tight')
        except:
            fig.savefig(save_path + ins_mode + '_regr_' + degree[0] + '_' + degree[1] + '_resolution_map_' + str(int(mjd)) + '.png', dpi=100, format='png', bbox_inches='tight')
    
    return

def plot_resolution_map_HARPN(excl_lines, pixels,orders,resolution,pixels_hr,matrix_resolution_hr_theo,n_ord,save_path,res_min,res_max,ins_mode,mjd,degree,Doit=False):

    n_pixel = 4096.
    if Doit==True:
        fig,ax = plt.subplots(figsize=(20, 20))
        ax.scatter(pixels, orders, c=resolution, s=50, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        a = ax.pcolormesh(np.insert(pixels_hr,0,0.), np.arange(1., n_ord+1.01, 1),matrix_resolution_hr_theo, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
        ax.pcolormesh(np.insert(pixels_hr,0,0.), np.arange(1., n_ord+1.01, 1),matrix_resolution_hr_theo, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
        ax.scatter(excl_lines[0], excl_lines[1], color = 'black', label = 'excluded lines ' + str(len(excl_lines[0])))
        ax.set_xlabel('Pixels', size=55)
        ax.set_ylabel('Orders', size=55)
        ax.set_title('HARPN - ' + ins_mode + ' - regression ' + degree, size=55)
        ax.legend()
        ax.tick_params(axis='both',which='both', direction='out', width=3, length=6, labelsize = 40)
        ax.set_ylim(1, n_ord+1)
        ax.set_xlim(0, n_pixel)
        cbar_ax = fig.add_axes([0.93, 0.11, 0.014, 0.873])
        axcb = plt.colorbar(mappable=a,cax=cbar_ax, format='%0.0f')
        axcb.ax.tick_params(labelsize=20)
        axcb.set_label('Resolution', size=30)
        fig.tight_layout(w_pad=0.1, h_pad=2, rect=(0.0005, 0.0001, 0.935, 0.999))
        fig.savefig(save_path + ins_mode + '_regr_' + degree + '_resolution_map_' + str(int(mjd)) + '.png', dpi=100, format='png', bbox_inches='tight')
    
    return

def f_plot_harpn_thar_line_distribution_rainbow(harpn_thar_line, ins_mode, file_path, simple_plot=False, xmedian_pixel=None, xmedian_res = None, ymedian_res=None):
    file_path = '/Users/saratavella/Desktop/ANTARESS/resolution_maps/plot_presentation/'
    
    # access thar line table
    qual = harpn_thar_line[1].data['qc']
    ords_all = np.array(harpn_thar_line[1].data['order'][np.where(qual != 0)])
    pxs_all = np.array(harpn_thar_line[1].data['x0'][np.where(qual != 0)])
    res_all = np.array(harpn_thar_line[1].data['resolution'][np.where(qual != 0)])

    # line exclusion
    mean_res = np.mean(res_all) #np.median(res_all)
    sigma_res = np.std(res_all) #mad(res_all)
    n_sigma = 3

    ords = np.array([ ords_all[i] for i in np.arange(ords_all.size) if ( res_all[i] < (mean_res + n_sigma * sigma_res) and res_all[i] > (mean_res - n_sigma * sigma_res)) ])
    pxs = np.array([ pxs_all[i] for i in np.arange(pxs_all.size) if ( res_all[i] < (mean_res + n_sigma * sigma_res) and res_all[i] > (mean_res - n_sigma * sigma_res)) ])
    res = np.array([ res_all[i] for i in np.arange(res_all.size) if ( res_all[i] < (mean_res + n_sigma * sigma_res) and res_all[i] > (mean_res - n_sigma * sigma_res)) ]) 


    # plot - excluded lines map 
    if simple_plot:
        fig, ax = plt.subplots(figsize=(12, 12))

        # excluded lines
        excl_pxs = np.array([ pxs_all[i] for i in np.arange(pxs_all.size) if ( res_all[i] >= (mean_res + n_sigma * sigma_res) or res_all[i] < (mean_res - n_sigma * sigma_res)) ])
        excl_ords = np.array([ ords_all[i] for i in np.arange(ords_all.size) if ( res_all[i] >= (mean_res + n_sigma * sigma_res) or res_all[i] < (mean_res - n_sigma * sigma_res)) ])
        
        ax.scatter(excl_pxs, excl_ords, color='black', label = 'excluded lines ' + str(excl_pxs.size))
        a = ax.scatter(pxs, ords, c=res, cmap=mpl.cm.rainbow, vmin=min(res),vmax=max(res), edgecolors='white', zorder=100)
        ax.scatter(pxs, ords, c=res, cmap=mpl.cm.rainbow, vmin=min(res),vmax=max(res), edgecolors='white', zorder=100)
        ax.set_title(ins_mode + ' - line map')
        
        ax.set_xlabel('Pixels', size=20)
        ax.set_ylabel('Orders', size=20)
        ax.tick_params(axis='both',which='both', direction='out', width=2, length=6)
        ax.set_ylim(1, 70)
        ax.set_xlim(0, 4096)
        cbar_ax = fig.add_axes([0.93, 0.11, 0.014, 0.873])
        axcb = plt.colorbar(mappable=a,cax=cbar_ax, format='%0.0f')
        axcb.ax.tick_params(labelsize=20)
        axcb.set_label('Resolution', size=30)
        fig.tight_layout(w_pad=0.1, h_pad=2, rect=(0.0005, 0.0001, 0.935, 0.999))
        ax.legend()
        fig.savefig(file_path + ins_mode + '_map_lines_excluded_simple.png', dpi=100, format='png', bbox_inches='tight')

    else: 
        fig, ax = plt.subplots(figsize=(20, 20))
        gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                            left=0.1, right=0.9, bottom=0.1, top=0.9,
                            wspace=0.05, hspace=0.05)

        ax = fig.add_subplot(gs[1, 0])
        ax_cutx = fig.add_subplot(gs[0, 0])
        ax_cuty = fig.add_subplot(gs[1, 1])
        ax_cutx.plot(xmedian_pixel[100:-100],xmedian_res[100:-100], alpha=0.5, color = 'black')
        ax_cutx.scatter(xmedian_pixel[100:-100],xmedian_res[100:-100], color = 'black')
        ax_cutx.set_ylabel('Resolution', size=20)
        ax_cuty.plot(ymedian_res, np.arange(1,70), alpha=0.5, color = 'black')
        ax_cuty.scatter(ymedian_res, np.arange(1,70), color = 'black')
        ax_cuty.set_xlabel('Resolution', size=20)
        #ax_cuty.set_xscale('log')

        # excluded lines
        excl_pxs = np.array([ pxs_all[i] for i in np.arange(pxs_all.size) if ( res_all[i] >= (mean_res + n_sigma * sigma_res) or res_all[i] < (mean_res - n_sigma * sigma_res)) ])
        excl_ords = np.array([ ords_all[i] for i in np.arange(ords_all.size) if ( res_all[i] >= (mean_res + n_sigma * sigma_res) or res_all[i] < (mean_res - n_sigma * sigma_res)) ])
        
        ax.scatter(excl_pxs, excl_ords, color='black', label = 'excluded lines ' + str(excl_pxs.size))
        a = ax.scatter(pxs, ords, c=res, cmap=mpl.cm.rainbow, vmin=min(res),vmax=max(res), edgecolors='white', zorder=100)
        ax.scatter(pxs, ords, c=res, cmap=mpl.cm.rainbow, vmin=min(res),vmax=max(res), edgecolors='white', zorder=100)
        #ax.set_title(ins_mode + ' - line map')
        
        ax.set_xlabel('Pixels', size=20)
        ax.set_ylabel('Orders', size=20)
        ax.tick_params(axis='both',which='both', direction='out', width=2, length=6)
        ax.set_ylim(1, 70)
        ax.set_xlim(0, 4096)
        cbar_ax = fig.add_axes([0.93, 0.11, 0.014, 0.873])
        axcb = plt.colorbar(mappable=a,cax=cbar_ax, format='%0.0f')
        axcb.ax.tick_params(labelsize=20)
        axcb.set_label('Resolution', size=30)
        #fig.tight_layout(w_pad=0.1, h_pad=2, rect=(0.0005, 0.0001, 0.935, 0.999))
        ax.legend()
        fig.savefig(file_path + ins_mode + '_map_lines_excluded.png', dpi=100, format='png', bbox_inches='tight')


    return


# ESPRESSO and NIRPS 
# Romain
def plot_resolution_map_UHR_HR(pixels_blue_slice1,orders_blue_slice1,resolution_blue_slice1,pixels_blue_hr,matrix_resolution_blue_hr_theo_slice1,pixels_red_slice1,
                               orders_red_slice1,resolution_red_slice1,pixels_red_hr,matrix_resolution_red_hr_theo_slice1,res_min,res_max,
                               pixels_blue_slice2,orders_blue_slice2,resolution_blue_slice2,matrix_resolution_blue_hr_theo_slice2,pixels_red_slice2,
                               orders_red_slice2,resolution_red_slice2,matrix_resolution_red_hr_theo_slice2,ins_mode,binx,mjd,Doit=False,resolution_map_path='/Users/saratavella/Desktop/phd/ANTARESS/resolution_maps/'):
    if Doit==True:
        fig = pl.figure(figsize=(20, 20))
        pl.scatter(pixels_blue_slice1, orders_blue_slice1, c=resolution_blue_slice1, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_blue_hr, np.arange(1, 90.01, 1), matrix_resolution_blue_hr_theo_slice1, cmap=mpl.cm.rainbow,shading='gouraud', vmin=res_min, vmax=res_max)
        pl.scatter(pixels_red_slice1, orders_red_slice1, c=resolution_red_slice1, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_red_hr, np.arange(91, 170.01, 1), matrix_resolution_red_hr_theo_slice1, cmap=mpl.cm.rainbow,shading='gouraud', vmin=res_min, vmax=res_max)
        pl.scatter(pixels_blue_slice2, orders_blue_slice2, c=resolution_blue_slice2, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_blue_hr, np.arange(1, 90.01, 1), matrix_resolution_blue_hr_theo_slice2, cmap=mpl.cm.rainbow,shading='gouraud', vmin=res_min, vmax=res_max)
        pl.scatter(pixels_red_slice2, orders_red_slice2, c=resolution_red_slice2, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_red_hr, np.arange(91, 170.01, 1), matrix_resolution_red_hr_theo_slice2, cmap=mpl.cm.rainbow,shading='gouraud', vmin=res_min, vmax=res_max)
        pl.xlabel('Pixels', size=30)
        pl.ylabel('Orders', size=30)
        pl.xticks(size=20)
        pl.yticks(size=20)
        pl.ylim(0, 170)
        pl.xlim(0, 9211)
        cbar_ax = fig.add_axes([0.93, 0.11, 0.014, 0.873])
        axcb = pl.colorbar(cax=cbar_ax, format='%0.0f')
        axcb.ax.tick_params(labelsize=20)
        axcb.set_label('Resolution', size=30)
        pl.tight_layout(w_pad=0.1, h_pad=2, rect=(0.0005, 0.0001, 0.935, 0.999))
        pl.savefig(resolution_map_path+'Map_resolution_' + ins_mode + '_' + str(binx) + '_' + str(int(mjd)) + '.png', dpi=100, format='png', bbox_inches='tight')
    return

def plot_resolution_map_slice_UHR_HR(pixels_blue_slice1,orders_blue_slice1,resolution_blue_slice1,pixels_blue_hr,matrix_resolution_blue_hr_theo_slice1,pixels_red_slice1,
                                     orders_red_slice1,resolution_red_slice1,pixels_red_hr,matrix_resolution_red_hr_theo_slice1,res_min,res_max,
                                     pixels_blue_slice2,orders_blue_slice2,resolution_blue_slice2,matrix_resolution_blue_hr_theo_slice2,pixels_red_slice2,
                                     orders_red_slice2,resolution_red_slice2,matrix_resolution_red_hr_theo_slice2,ins_mode,binx,mjd,Doit=False,resolution_map_path='/Users/saratavella/Desktop/phd/ANTARESS/resolution_maps/'):
    if Doit == True:
        fig = pl.figure(figsize=(20, 20))
        pl.subplot(1, 2, 1)
        pl.scatter(pixels_blue_slice1, orders_blue_slice1, c=resolution_blue_slice1, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(np.insert(pixels_blue_hr,0,0.), np.arange(1, 91.01, 1), matrix_resolution_blue_hr_theo_slice1, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
        pl.scatter(pixels_red_slice1, orders_red_slice1, c=resolution_red_slice1, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(np.insert(pixels_red_hr,0,90.), np.arange(91, 171.01, 1), matrix_resolution_red_hr_theo_slice1, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
        pl.xlabel('Pixels', size=30)
        pl.ylabel('Orders', size=30)
        pl.xticks(size=20)
        pl.yticks(size=20)
        pl.ylim(0, 170)
        pl.xlim(0, 9211)
        cbar_ax = fig.add_axes([0.93, 0.092, 0.014, 0.89])
        axcb = pl.colorbar(cax=cbar_ax, format='%0.2f')
        axcb.set_label('Resolution', size=30)
        pl.tight_layout(w_pad=0.1, h_pad=2, rect=(0.0005, 0.0001, 0.935, 0.999))
        pl.subplot(1, 2, 2)
        pl.scatter(pixels_blue_slice2, orders_blue_slice2, c=resolution_blue_slice2, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_blue_hr, np.arange(1, 90.01, 1), matrix_resolution_blue_hr_theo_slice2, cmap=mpl.cm.rainbow,shading='gouraud', vmin=res_min, vmax=res_max)
        pl.scatter(pixels_red_slice2, orders_red_slice2, c=resolution_red_slice2, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_red_hr, np.arange(91, 170.01, 1), matrix_resolution_red_hr_theo_slice2, cmap=mpl.cm.rainbow,shading='gouraud', vmin=res_min, vmax=res_max)
        pl.xlabel('Pixels', size=30)
        pl.ylabel('Orders', size=30)
        pl.xticks(size=20)
        pl.yticks(size=20)
        pl.ylim(0, 170)
        pl.xlim(0, 9211)
        cbar_ax = fig.add_axes([0.93, 0.092, 0.014, 0.89])
        axcb = pl.colorbar(cax=cbar_ax, format='%0.0f')
        axcb.ax.tick_params(labelsize=20)
        axcb.set_label('Resolution', size=30)
        pl.tight_layout(w_pad=0.1, h_pad=2, rect=(0.0005, 0.0001, 0.935, 0.999))
        pl.savefig(resolution_map_path+'Map_resolution_slices_' + ins_mode + '_' + str(binx) + '_' + str(int(mjd)) + '.png', dpi=100, format='png', bbox_inches='tight')
    return

def plot_resolution_map_MR(pixels_blue_slice1,orders_blue_slice1,resolution_blue_slice1,pixels_blue_hr,matrix_resolution_blue_hr_theo_slice1,pixels_red_slice1,
                           orders_red_slice1,resolution_red_slice1,pixels_red_hr,matrix_resolution_red_hr_theo_slice1,res_min,res_max,
                           ins_mode,binx,mjd,Doit=False,resolution_map_path='/Users/saratavella/Desktop/phd/ANTARESS/resolution_maps/'):
    if Doit == True:
        fig = pl.figure(figsize=(20, 20))
        pl.scatter(pixels_blue_slice1, orders_blue_slice1, c=resolution_blue_slice1, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_blue_hr, np.arange(1, 45.01, 1), matrix_resolution_blue_hr_theo_slice1, cmap=mpl.cm.rainbow,shading='gouraud', vmin=res_min, vmax=res_max)
        pl.scatter(pixels_red_slice1, orders_red_slice1, c=resolution_red_slice1, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_red_hr, np.arange(46, 85.01, 1), matrix_resolution_red_hr_theo_slice1, cmap=mpl.cm.rainbow,shading='gouraud', vmin=res_min, vmax=res_max)
        pl.xlabel('Pixels', size=30)
        pl.ylabel('Orders', size=30)
        pl.xticks(size=20)
        pl.yticks(size=20)
        pl.ylim(0, 85)
        if binx ==8:
            pl.xlim(0, 2295)            
        else:
            pl.xlim(0, 4590)
        cbar_ax = fig.add_axes([0.93, 0.11, 0.014, 0.873])
        axcb = pl.colorbar(cax=cbar_ax, format='%0.0f')
        axcb.ax.tick_params(labelsize=20)
        axcb.set_label('Resolution', size=30)
        pl.tight_layout(w_pad=0.1, h_pad=2, rect=(0.0005, 0.0001, 0.935, 0.999))
        pl.savefig(resolution_map_path+'Map_resolution_' + ins_mode + '_' + str(binx) + '_' + str(int(mjd)) + '.png', dpi=100, format='png', bbox_inches='tight')
    return




'''
SECOND STEP - CREATING RESOLUTION MAPS
The following functions create the resolution maps 
- Built differently for the 2 different instruments 
- Nominal resolution for HARPS and HARPS-N: 115000
- We expect the average resolution not to vary a lot over time. Yet, it might vary more at certain position on the detector.

- Depending on the mode (resolution) of the instrument we need different line lists:
    HARPS-N only has 1 mode (HR) 
    HARPS - HAM is the standard mode for the insturment
          - EGGS if the second mode

- For the moment, we consider three different epochs of the instrument for both HARPS and HARPS-N. 
  For each epoch, we treat the detector as a completely diverse instrument, as follows:

    HARPS 
    - installed in 2003 :> 03
    - change of the fibers in 2015 :> 15 
    - change of the cryostat in 2023 :> 23 

    HARPSN
    - installed in 2012 :> 12
    - change of the focus in 2014 :> 14 
    - change of the cryostat in 2021 :> 21 

    
### NAMES OF THE OUTPUT FILES 
- Only one map for each instrument/version is needed at the moment
  (this is true in first approximation, even though there might be some resolution changes unrelated to instrument interventions as well 
  
    HARPS_HARPS03_RESOLUTION_MAP.fits
    HARPS_HARPS15_RESOLUTION_MAP.fits
    HARPS_HARPS23_RESOLUTION_MAP.fits

    HARPS_EGGS03_RESOLUTION_MAP.fits
    HARPS_EGGS15_RESOLUTION_MAP.fits
    HARPS_EGGS23_RESOLUTION_MAP.fits

    HARPN_HARPN12_RESOLUTION_MAP.fits
    HARPN_HARPN14_RESOLUTION_MAP.fits
    HARPN_HARPN21_RESOLUTION_MAP.fits


'''


############
### HARPS ##
############
# author: Sara 
# inst_mode = 'EGGS03', 'EGGS15', 'EGGS23', 'HARPS03', 'HARPS15', 'HARPS23'
def Instrumental_resolution_HARPS(thar_file_path,THAR_LINE_TABLE_THAR_FP_A, inst_mode, function_map_resolution, degree):
    
    if 'HARPS' in inst_mode: 
        res_min=70000
        res_max=180000
    if 'EGGS' in inst_mode: 
        res_min=55000
        res_max=130000
    
    # number of orders 
    n_ord = 71
    # index of the order where the detectors' gap falls 
    order_sep = 45

    #header file 
    header = THAR_LINE_TABLE_THAR_FP_A[0].header
    obs_time = header['MJD-OBS']

    # access to the thar-line table
    # quality flag:
    qual = THAR_LINE_TABLE_THAR_FP_A[1].data['qc']  
    # 2D resolution table for the thar lines:
    ords = THAR_LINE_TABLE_THAR_FP_A[1].data['order'][np.where(qual != 0)]
    pxs = THAR_LINE_TABLE_THAR_FP_A[1].data['x0'][np.where(qual != 0)]
    res = THAR_LINE_TABLE_THAR_FP_A[1].data['resolution'][np.where(qual != 0)]

    # shaping of the detectors 
    # HARPS works with 1 fiber only (as the MR mode in espresso)
    shape_s2d = shape_s2d_HARPS()
    
    orders_blue_hr = np.arange(1, order_sep + 1, 1, dtype=int)
    pixels_blue_hr = shape_s2d[0]

    orders_red_hr = np.arange(order_sep + 1, n_ord + 1, 1, dtype=int)
    pixels_red_hr = shape_s2d[1]

    matrix_resolution_blue_hr_theo = np.zeros((orders_blue_hr.size, len(pixels_blue_hr)))
    matrix_resolution_red_hr_theo = np.zeros((orders_red_hr.size, len(pixels_red_hr)))

    #################
    # BLUE DETECTOR #
    #################
    orders_blue_all = ords[(ords <= order_sep)]
    pixels_blue_all = pxs[(ords <= order_sep)]
    resolution_blue_all = res[(ords <= order_sep)]

    orders_blue_all = np.array(orders_blue_all)
    pixels_blue_all = np.array(pixels_blue_all)
    resolution_blue_all = np.array(resolution_blue_all)
    
    # lines exclusion based on line resolution distribution (exclusion of outliers wrt to the mean)
    mean_res_blue = resolution_blue_all.mean()
    sigma_res_blue = resolution_blue_all.std()
    n_sigma_blue = 3
    n_thar_lines_blue = resolution_blue_all.size

    keep_condition = lambda i : ( resolution_blue_all[i] < (mean_res_blue + n_sigma_blue * sigma_res_blue) and resolution_blue_all[i] > (mean_res_blue - n_sigma_blue * sigma_res_blue) ) 
    excl_condition = lambda i : ( resolution_blue_all[i] > (mean_res_blue + n_sigma_blue * sigma_res_blue) or resolution_blue_all[i] < (mean_res_blue - n_sigma_blue * sigma_res_blue) ) 
    
    orders_blue = [ orders_blue_all[i] for i in np.arange(n_thar_lines_blue) if keep_condition(i) ]
    excl_orders_blue = [ orders_blue_all[i] for i in np.arange(n_thar_lines_blue) if excl_condition(i) ]
    pixels_blue = [ pixels_blue_all[i] for i in np.arange(n_thar_lines_blue) if keep_condition(i) ]
    excl_pixels_blue = [ pixels_blue_all[i] for i in np.arange(n_thar_lines_blue) if excl_condition(i) ]
    resolution_blue = [ resolution_blue_all[i] for i in np.arange(n_thar_lines_blue) if keep_condition(i) ] 

    ################
    # RED DETECTOR #
    ################
    orders_red_all = ords[(ords > order_sep)]
    pixels_red_all = pxs[(ords > order_sep)]
    resolution_red_all = res[(ords > order_sep)]

    orders_red_all = np.array(orders_red_all)
    pixels_red_all = np.array(pixels_red_all)
    resolution_red_all = np.array(resolution_red_all)
    
    mean_res_red = resolution_red_all.mean()
    sigma_res_red = resolution_red_all.std()
    n_sigma_red = 3
    n_thar_lines_red = resolution_red_all.size
    
    keep_condition = lambda i : ( resolution_red_all[i] < (mean_res_red + n_sigma_red * sigma_res_red) and resolution_red_all[i] > (mean_res_red - n_sigma_red * sigma_res_red) ) 
    excl_condition = lambda i : ( resolution_red_all[i] > (mean_res_red + n_sigma_red * sigma_res_red) or resolution_red_all[i] < (mean_res_red - n_sigma_red * sigma_res_red) ) 
    
    orders_red = [orders_red_all[i] for i in np.arange(n_thar_lines_red) if keep_condition(i) ]
    excl_orders_red = [orders_red_all[i] for i in np.arange(n_thar_lines_red) if excl_condition(i) ]
    pixels_red = [pixels_red_all[i] for i in np.arange(n_thar_lines_red) if keep_condition(i) ]
    excl_pixels_red = [pixels_red_all[i] for i in np.arange(n_thar_lines_red) if excl_condition(i) ]
    resolution_red = [resolution_red_all[i] for i in np.arange(n_thar_lines_red) if keep_condition(i) ] 
    
    ####################################
    # AVERAGE PROFILES ON THE DETECTOR #
    ####################################

    # create the average res. profile along x and y direction on the detector - to find initial condition for the model
    ymedian_res_blue, ymedian_res_blue_error, xmedian_res_blue, xmedian_res_blue_error, xmedian_pixel_blue = fit_average_resolution_profile_blue(orders_blue,resolution_blue,pixels_blue, order_sep, n_ord)[0]
    ymedian_res_red, ymedian_res_red_error, xmedian_res_red, xmedian_res_red_error, xmedian_pixel_red = fit_average_resolution_profile_red(orders_red,resolution_red,pixels_red, order_sep, n_ord)[0]

    fig, axs = plt.subplots(2,2, figsize=(12,10))
    axs = axs.flatten()
    axs[0].plot(ymedian_res_red, np.arange(order_sep, n_ord), alpha=0.5)
    axs[0].scatter(ymedian_res_red, np.arange(order_sep, n_ord))
    axs[0].set_title(inst_mode + ' - mean resolution order per order - red')
    axs[1].plot(xmedian_pixel_red[100:-100],xmedian_res_red[100:-100], alpha=0.5)
    axs[1].scatter(xmedian_pixel_red[100:-100],xmedian_res_red[100:-100])
    axs[1].set_title(inst_mode + ' - overall resolution trend over pixels - red')
    axs[2].plot(ymedian_res_blue, np.arange(order_sep), alpha=0.5)
    axs[2].scatter(ymedian_res_blue, np.arange(order_sep))
    axs[2].set_title(inst_mode + ' - mean resolution order per order - blue')
    axs[3].plot(xmedian_pixel_blue[100:-100],xmedian_res_blue[100:-100], alpha=0.5)
    axs[3].scatter(xmedian_pixel_blue[100:-100],xmedian_res_blue[100:-100])
    axs[3].set_title(inst_mode + ' - overall resolution trend over pixels - blue')
    fig.savefig(thar_file_path + inst_mode + '_detector_average_profiles.png', dpi=100, format='png', bbox_inches='tight')

    # fit the average profiles 
    fit_y_blue,BIC_y_blue,fit_x_blue,BIC_x_blue  = fit_average_resolution_profile_blue(orders_blue,resolution_blue,pixels_blue, order_sep, n_ord)[1]
    fit_y_red,BIC_y_red,fit_x_red,BIC_x_red =fit_average_resolution_profile_red(orders_red,resolution_red,pixels_red, order_sep, n_ord)[1]

    #plot BICs
    fig, axs = plt.subplots(2,2, figsize = (15,15))
    axs = axs.flatten()
    axs[0].plot(np.arange(4), BIC_x_red)
    axs[0].set_xticks(np.arange(4),['2d','3d','4d','5d'])
    axs[0].set_title(inst_mode + ' - BICs x profile - red')
    axs[1].plot(np.arange(4), BIC_y_red)
    axs[1].set_xticks(np.arange(4),['2d','3d','4d','5d'])
    axs[1].set_title(inst_mode + ' - BICs y profile - red')
    axs[2].plot(np.arange(4), BIC_x_blue)
    axs[2].set_xticks(np.arange(4),['2d','3d','4d','5d'])
    axs[2].set_title(inst_mode + ' - BICs x profile - blue')
    axs[3].plot(np.arange(4), BIC_y_blue)
    axs[3].set_xticks(np.arange(4),['2d','3d','4d','5d'])
    axs[3].set_title(inst_mode + ' - BICs y profile - blue')
    fig.savefig(thar_file_path + inst_mode + '_detector_average_profiles_BIC.png', dpi=100, format='png', bbox_inches='tight')

    # plot best fit model
    fit_x_blue[BIC_x_blue.index(min(BIC_x_blue))].plot()
    plt.title(inst_mode + ' - best fit model for x profile - blue')
    plt.show()
    fit_y_blue[BIC_y_blue.index(min(BIC_y_blue))].plot()
    plt.title(inst_mode + ' - best fit model for y profile - blue')
    plt.show()
    fit_x_red[BIC_x_red.index(min(BIC_x_red))].plot()
    plt.title(inst_mode + ' - best fit model for x profile - red')
    plt.show()
    fit_y_red[BIC_y_red.index(min(BIC_y_red))].plot()
    plt.title(inst_mode + ' - best fit model for y profile - red')
    plt.show()
    
    
    ################
    # model creation
    ################
    
    if function_map_resolution==None:
        func_list = [polynomial_regression_2d,polynomial_regression_3d,polynomial_regression_4d,polynomial_regression_espresso]
        BIC = {'blue':[],'red':[]}
        rchi2 = {'blue':[],'red':[]}
        fit = {'blue':[],'red':[]}

        for function_map_resolution in func_list:
            if function_map_resolution == polynomial_regression_2d: degree = '2d'
            elif function_map_resolution == polynomial_regression_3d: degree = '3d'
            elif function_map_resolution == polynomial_regression_4d: degree = '4d'
            elif function_map_resolution == polynomial_regression_espresso: degree = 'espresso'

            gmodel = Model(function_map_resolution)
            param_list = gmodel.param_names
            k = len(param_list)
            for par in param_list:
                gmodel.set_param_hint(par, value=1., vary=True)
            
            gmodel.set_param_hint(param_list[-1], value=np.median(resolution_blue), vary=True)
            pars = gmodel.make_params()
            result_poly_reg_2d_blue = gmodel.fit(resolution_blue,xy=np.array([pixels_blue, orders_blue]), params=pars)
            BIC['blue'].append(result_poly_reg_2d_blue.bic)
            rchi2['blue'].append(result_poly_reg_2d_blue.redchi)
            fit['blue'].append(result_poly_reg_2d_blue)

            gmodel.set_param_hint(param_list[-1], value=np.median(resolution_red), vary=True)
            pars = gmodel.make_params()
            result_poly_reg_2d_red = gmodel.fit(resolution_red, xy=np.array([pixels_red, orders_red]), params=pars)
            BIC['red'].append(result_poly_reg_2d_red.bic)
            rchi2['red'].append(result_poly_reg_2d_red.redchi)
            fit['red'].append(result_poly_reg_2d_red)

            # interpolation of thar lines
            for id in orders_blue_hr:
                i = id - 1
                if k == 6: matrix_resolution_blue_hr_theo[i, :] = function_map_resolution(np.array([pixels_blue_hr, i], dtype=object),result_poly_reg_2d_blue.values['a'],result_poly_reg_2d_blue.values['b'],result_poly_reg_2d_blue.values['c'],result_poly_reg_2d_blue.values['d'],result_poly_reg_2d_blue.values['e'],result_poly_reg_2d_blue.values['f'])
                elif k == 7: matrix_resolution_blue_hr_theo[i, :] = function_map_resolution(np.array([pixels_blue_hr, i], dtype=object),result_poly_reg_2d_blue.values['a'],result_poly_reg_2d_blue.values['b'],result_poly_reg_2d_blue.values['c'],result_poly_reg_2d_blue.values['d'],result_poly_reg_2d_blue.values['e'],result_poly_reg_2d_blue.values['f'],result_poly_reg_2d_blue.values['g'])
                elif k == 10:matrix_resolution_blue_hr_theo[i, :] = function_map_resolution(np.array([pixels_blue_hr, i], dtype=object),result_poly_reg_2d_blue.values['a'],result_poly_reg_2d_blue.values['b'],result_poly_reg_2d_blue.values['c'],result_poly_reg_2d_blue.values['d'],result_poly_reg_2d_blue.values['e'],result_poly_reg_2d_blue.values['f'],result_poly_reg_2d_blue.values['g'],result_poly_reg_2d_blue.values['h'],result_poly_reg_2d_blue.values['i'],result_poly_reg_2d_blue.values['l'])
                elif k == 15: matrix_resolution_blue_hr_theo[i, :] = function_map_resolution(np.array([pixels_blue_hr, i], dtype=object),result_poly_reg_2d_blue.values['a'],result_poly_reg_2d_blue.values['b'],result_poly_reg_2d_blue.values['c'],result_poly_reg_2d_blue.values['d'],result_poly_reg_2d_blue.values['e'],result_poly_reg_2d_blue.values['f'],result_poly_reg_2d_blue.values['g'],result_poly_reg_2d_blue.values['h'],result_poly_reg_2d_blue.values['i'],result_poly_reg_2d_blue.values['l'],result_poly_reg_2d_blue.values['m'],result_poly_reg_2d_blue.values['n'],result_poly_reg_2d_blue.values['o'],result_poly_reg_2d_blue.values['p'],result_poly_reg_2d_blue.values['q'])
            
            for id in orders_red_hr:
                i = id - 1        
                if k == 6: matrix_resolution_red_hr_theo[i - order_sep, :] = function_map_resolution(np.array([pixels_red_hr, i], dtype=object),result_poly_reg_2d_red.values['a'],result_poly_reg_2d_red.values['b'],result_poly_reg_2d_red.values['c'],result_poly_reg_2d_red.values['d'],result_poly_reg_2d_red.values['e'],result_poly_reg_2d_red.values['f'])
                elif k == 7: matrix_resolution_red_hr_theo[i - order_sep, :] = function_map_resolution(np.array([pixels_red_hr, i], dtype=object),result_poly_reg_2d_red.values['a'],result_poly_reg_2d_red.values['b'],result_poly_reg_2d_red.values['c'],result_poly_reg_2d_red.values['d'],result_poly_reg_2d_red.values['e'],result_poly_reg_2d_red.values['f'],result_poly_reg_2d_red.values['g'])
                elif k == 10: matrix_resolution_red_hr_theo[i - order_sep, :] = function_map_resolution(np.array([pixels_red_hr, i], dtype=object),result_poly_reg_2d_red.values['a'],result_poly_reg_2d_red.values['b'],result_poly_reg_2d_red.values['c'],result_poly_reg_2d_red.values['d'],result_poly_reg_2d_red.values['e'],result_poly_reg_2d_red.values['f'],result_poly_reg_2d_red.values['g'],result_poly_reg_2d_red.values['h'],result_poly_reg_2d_red.values['i'],result_poly_reg_2d_red.values['l'])
                elif k == 15: matrix_resolution_red_hr_theo[i - order_sep, :] = function_map_resolution(np.array([pixels_red_hr, i], dtype=object),result_poly_reg_2d_red.values['a'],result_poly_reg_2d_red.values['b'],result_poly_reg_2d_red.values['c'],result_poly_reg_2d_red.values['d'],result_poly_reg_2d_red.values['e'],result_poly_reg_2d_red.values['f'],result_poly_reg_2d_red.values['g'],result_poly_reg_2d_red.values['h'],result_poly_reg_2d_red.values['i'],result_poly_reg_2d_red.values['l'],result_poly_reg_2d_red.values['m'],result_poly_reg_2d_red.values['n'],result_poly_reg_2d_red.values['o'],result_poly_reg_2d_red.values['p'],result_poly_reg_2d_red.values['q'])
            
            matrix_resolution_blue_hr_theo = np.where(matrix_resolution_blue_hr_theo == 0., np.nan,matrix_resolution_blue_hr_theo)
            matrix_resolution_red_hr_theo = np.where(matrix_resolution_red_hr_theo == 0., np.nan,matrix_resolution_red_hr_theo)

            # plot resolution maps
            excl_pxs = excl_pixels_blue + excl_pixels_red
            excl_ords = excl_orders_blue + excl_orders_red

            plot_resolution_map_HARPS([excl_pxs, excl_ords],pixels_blue,orders_blue,resolution_blue,pixels_blue_hr,matrix_resolution_blue_hr_theo,pixels_red,orders_red,resolution_red,pixels_red_hr,matrix_resolution_red_hr_theo,order_sep,n_ord,thar_file_path,res_min,res_max,inst_mode,obs_time,degree, Doit=True)

            #######################################
            ## store resolution maps in a fits file
            #######################################
            resolution_map = np.zeros((n_ord, len(shape_s2d[1])))

            # the numbering of the order starts at 1: 
            # order 45 is index 44 
            # order 46 is index 45 
            resolution_map[:order_sep, int(pixels_blue_hr[0])-1:int(pixels_blue_hr[-1]) + 1] = matrix_resolution_blue_hr_theo
            resolution_map[order_sep:, :] = matrix_resolution_red_hr_theo
            resolution_map = np.where(resolution_map == 0., np.nan, resolution_map)

            prihdu = pyfits.PrimaryHDU(header=header)
            prihdu.writeto(thar_file_path + inst_mode + '_regr_' + degree + '_RESOLUTION_MAP.fits', overwrite=True)
            pyfits.append(thar_file_path + inst_mode + '_regr_' + degree + '_RESOLUTION_MAP.fits', resolution_map)

            print(inst_mode + ' - BLUE - resolution map - MAD =', int(mad(resolution_map[:order_sep, :].flatten())) )
            print(inst_mode + ' - RED  - resolution map - MAD =', int(mad(resolution_map[order_sep:, :].flatten())) )
            
        #plot BICs
        fig, axs = plt.subplots(1,2, figsize = (15,8))
        axs = axs.flatten()
        axs[0].plot(np.arange(4), BIC['blue'])
        axs[0].set_xticks(np.arange(4),['2d','3d','4d','espresso'])
        axs[0].set_title(inst_mode + ' - BICs 2D map - blue')
        axs[1].plot(np.arange(4), BIC['red'])
        axs[1].set_xticks(np.arange(4),['2d','3d','4d','espresso'])
        axs[1].set_title(inst_mode + ' - BICs 2D map - red')
        fig.savefig(thar_file_path + inst_mode + '_resolution_model_BIC.png', dpi=100, format='png', bbox_inches='tight')
    
    else: 
        function_map_resolution_blue = function_map_resolution[0]
        function_map_resolution_red = function_map_resolution[1]
        degree_blue = degree[0]
        degree_red = degree[1]

        # blue detector
        gmodel = Model(function_map_resolution_blue)
        param_list = gmodel.param_names
        k = len(param_list)
        for par in param_list:
            gmodel.set_param_hint(par, value=1., vary=True)
        
        gmodel.set_param_hint(param_list[-1], value=np.median(resolution_blue), vary=True)
        pars = gmodel.make_params()
        result_poly_reg_2d_blue = gmodel.fit(resolution_blue,xy=np.array([pixels_blue, orders_blue]), params=pars)
        
        # interpolation of thar lines
        for id in orders_blue_hr:
            i = id - 1
            if k == 6: matrix_resolution_blue_hr_theo[i, :] = function_map_resolution_blue(np.array([pixels_blue_hr, i], dtype=object),result_poly_reg_2d_blue.values['a'],result_poly_reg_2d_blue.values['b'],result_poly_reg_2d_blue.values['c'],result_poly_reg_2d_blue.values['d'],result_poly_reg_2d_blue.values['e'],result_poly_reg_2d_blue.values['f'])
            elif k == 7: matrix_resolution_blue_hr_theo[i, :] = function_map_resolution_blue(np.array([pixels_blue_hr, i], dtype=object),result_poly_reg_2d_blue.values['a'],result_poly_reg_2d_blue.values['b'],result_poly_reg_2d_blue.values['c'],result_poly_reg_2d_blue.values['d'],result_poly_reg_2d_blue.values['e'],result_poly_reg_2d_blue.values['f'],result_poly_reg_2d_blue.values['g'])
            elif k == 10:matrix_resolution_blue_hr_theo[i, :] = function_map_resolution_blue(np.array([pixels_blue_hr, i], dtype=object),result_poly_reg_2d_blue.values['a'],result_poly_reg_2d_blue.values['b'],result_poly_reg_2d_blue.values['c'],result_poly_reg_2d_blue.values['d'],result_poly_reg_2d_blue.values['e'],result_poly_reg_2d_blue.values['f'],result_poly_reg_2d_blue.values['g'],result_poly_reg_2d_blue.values['h'],result_poly_reg_2d_blue.values['i'],result_poly_reg_2d_blue.values['l'])
            elif k == 15: matrix_resolution_blue_hr_theo[i, :] = function_map_resolution_blue(np.array([pixels_blue_hr, i], dtype=object),result_poly_reg_2d_blue.values['a'],result_poly_reg_2d_blue.values['b'],result_poly_reg_2d_blue.values['c'],result_poly_reg_2d_blue.values['d'],result_poly_reg_2d_blue.values['e'],result_poly_reg_2d_blue.values['f'],result_poly_reg_2d_blue.values['g'],result_poly_reg_2d_blue.values['h'],result_poly_reg_2d_blue.values['i'],result_poly_reg_2d_blue.values['l'],result_poly_reg_2d_blue.values['m'],result_poly_reg_2d_blue.values['n'],result_poly_reg_2d_blue.values['o'],result_poly_reg_2d_blue.values['p'],result_poly_reg_2d_blue.values['q'])
        

        # red detector
        gmodel = Model(function_map_resolution_red)
        param_list = gmodel.param_names
        k = len(param_list)
        for par in param_list:
            gmodel.set_param_hint(par, value=1., vary=True) 

        gmodel.set_param_hint(param_list[-1], value=np.median(resolution_red), vary=True)
        pars = gmodel.make_params()
        result_poly_reg_2d_red = gmodel.fit(resolution_red, xy=np.array([pixels_red, orders_red]), params=pars)
        
        # interpolation of thar lines
        for id in orders_red_hr:
            i = id - 1        
            if k == 6: matrix_resolution_red_hr_theo[i - order_sep, :] = function_map_resolution_red(np.array([pixels_red_hr, i], dtype=object),result_poly_reg_2d_red.values['a'],result_poly_reg_2d_red.values['b'],result_poly_reg_2d_red.values['c'],result_poly_reg_2d_red.values['d'],result_poly_reg_2d_red.values['e'],result_poly_reg_2d_red.values['f'])
            elif k == 7: matrix_resolution_red_hr_theo[i - order_sep, :] = function_map_resolution_red(np.array([pixels_red_hr, i], dtype=object),result_poly_reg_2d_red.values['a'],result_poly_reg_2d_red.values['b'],result_poly_reg_2d_red.values['c'],result_poly_reg_2d_red.values['d'],result_poly_reg_2d_red.values['e'],result_poly_reg_2d_red.values['f'],result_poly_reg_2d_red.values['g'])
            elif k == 10: matrix_resolution_red_hr_theo[i - order_sep, :] = function_map_resolution_red(np.array([pixels_red_hr, i], dtype=object),result_poly_reg_2d_red.values['a'],result_poly_reg_2d_red.values['b'],result_poly_reg_2d_red.values['c'],result_poly_reg_2d_red.values['d'],result_poly_reg_2d_red.values['e'],result_poly_reg_2d_red.values['f'],result_poly_reg_2d_red.values['g'],result_poly_reg_2d_red.values['h'],result_poly_reg_2d_red.values['i'],result_poly_reg_2d_red.values['l'])
            elif k == 15: matrix_resolution_red_hr_theo[i - order_sep, :] = function_map_resolution_red(np.array([pixels_red_hr, i], dtype=object),result_poly_reg_2d_red.values['a'],result_poly_reg_2d_red.values['b'],result_poly_reg_2d_red.values['c'],result_poly_reg_2d_red.values['d'],result_poly_reg_2d_red.values['e'],result_poly_reg_2d_red.values['f'],result_poly_reg_2d_red.values['g'],result_poly_reg_2d_red.values['h'],result_poly_reg_2d_red.values['i'],result_poly_reg_2d_red.values['l'],result_poly_reg_2d_red.values['m'],result_poly_reg_2d_red.values['n'],result_poly_reg_2d_red.values['o'],result_poly_reg_2d_red.values['p'],result_poly_reg_2d_red.values['q'])
        

        matrix_resolution_blue_hr_theo = np.where(matrix_resolution_blue_hr_theo == 0., np.nan,matrix_resolution_blue_hr_theo)
        matrix_resolution_red_hr_theo = np.where(matrix_resolution_red_hr_theo == 0., np.nan,matrix_resolution_red_hr_theo)

        # plot resolution maps
        excl_pxs = excl_pixels_blue + excl_pixels_red
        excl_ords = excl_orders_blue + excl_orders_red

        plot_resolution_map_HARPS([excl_pxs, excl_ords],pixels_blue,orders_blue,resolution_blue,pixels_blue_hr,matrix_resolution_blue_hr_theo,pixels_red,orders_red,resolution_red,pixels_red_hr,matrix_resolution_red_hr_theo,order_sep,n_ord,thar_file_path,res_min,res_max,inst_mode,obs_time,degree, Doit=True)

        #######################################
        ## store resolution maps in a fits file
        #######################################
        resolution_map = np.zeros((n_ord, len(shape_s2d[1])))

        # the numbering of the order starts at 1: 
        # order 45 is index 44 
        # order 46 is index 45 
        resolution_map[:order_sep, int(pixels_blue_hr[0])-1:int(pixels_blue_hr[-1]) + 1] = matrix_resolution_blue_hr_theo
        resolution_map[order_sep:, :] = matrix_resolution_red_hr_theo
        resolution_map = np.where(resolution_map == 0., np.nan, resolution_map)

        prihdu = pyfits.PrimaryHDU(header=header)
        prihdu.writeto(thar_file_path + inst_mode + '_regr_' + degree_blue + '_' + degree_red + '_RESOLUTION_MAP.fits', overwrite=True)
        pyfits.append(thar_file_path + inst_mode + '_regr_' + degree_blue + '_' + degree_red + '_RESOLUTION_MAP.fits', resolution_map)

        print(inst_mode + ' - BLUE - resolution map - MAD =', int(mad(resolution_map[:order_sep, :].flatten())) )
        print(inst_mode + ' - RED  - resolution map - MAD =', int(mad(resolution_map[order_sep:, :].flatten())) )
        
    excluded_pixels = [[excl_orders_blue,excl_pixels_blue],[excl_orders_red,excl_pixels_red]]
    
    return excluded_pixels, [xmedian_pixel_red,xmedian_res_red,ymedian_res_red], [xmedian_pixel_blue,xmedian_res_blue,ymedian_res_blue]

def fit_average_resolution_profile_blue(orders,resolution,pixels, order_sep, n_ord): 
    '''fit a cumulative trend for x and y direction on the detector to find initial condition for the model'''
    
    # create average x and y profiles
    ymedian_res2 = []
    for id in np.arange(order_sep):
        id += 1 
        mask_id = [orders[i] == id for i in np.arange(len(orders))]
        ymedian_res2.append(np.median(np.array(resolution)[mask_id]))
    ymedian_res = pd.Series(ymedian_res2.copy()).rolling(15, min_periods=1).mean()
    ymedian_res_error = pd.Series(ymedian_res2.copy()).rolling(15, min_periods=1).apply(mad)

    df_xmedian_res = pd.DataFrame({'pixel': pixels, 'res': resolution})
    df_xmedian_res = df_xmedian_res.sort_values('pixel')
    xmedian_pixel = df_xmedian_res['pixel']
    xmedian_res = df_xmedian_res['res'].copy().rolling(400, min_periods=1).mean()
    xmedian_res_error = df_xmedian_res['res'].copy().rolling(400, min_periods=1).apply(mad)

    # fit average x and y profiles
    func_list = [polynomial_regression_2d_1D,polynomial_regression_3d_1D,polynomial_regression_4d_1D,polynomial_regression_5d_1D]
    BIC_x = []
    BIC_y = []
    rchi2_x = []
    rchi2_y = []
    fit_x = []
    fit_y = []

    for func in func_list: 
        gmodel = Model(func)
        param_list = gmodel.param_names
        k = len(param_list)
        for par in param_list:
            gmodel.set_param_hint(par, value=1., vary=True)

        ## fit x profile 
        gmodel.set_param_hint(param_list[-1], value=np.median(xmedian_res), vary=True)
        pars = gmodel.make_params()
        fit_medianx = gmodel.fit(xmedian_res[100:-100], t=df_xmedian_res['pixel'][100:-100], params=pars, weights = 1/xmedian_res_error[100:-100])
        #BIC_list.append(fit_median.bic)
        BIC_x.append(BIC_calc(xmedian_res[100:-100],xmedian_res_error[100:-100],fit_medianx.best_fit,k))
        rchi2_x.append(fit_medianx.redchi)
        fit_x.append(fit_medianx)

        ## fit the y profile for initial conditions 
        gmodel.set_param_hint(param_list[-1], value=np.median(ymedian_res), vary=True)
        pars = gmodel.make_params()
        fit_mediany = gmodel.fit(ymedian_res[5:-5], t=np.arange(1, order_sep + 1)[5:-5], params=pars, weights = 1/ymedian_res_error[5:-5])
        BIC_y.append(BIC_calc(ymedian_res[5:-5],ymedian_res_error[5:-5],fit_mediany.best_fit,k))
        #BIC_list.append(fit_median.bic)
        rchi2_y.append(fit_mediany.redchi)
        fit_y.append(fit_mediany)
    
    result = [[ymedian_res, ymedian_res_error, xmedian_res, xmedian_res_error, xmedian_pixel],[fit_y,BIC_y,fit_x,BIC_x]]
    
    return result

def fit_average_resolution_profile_red(orders,resolution,pixels, order_sep, n_ord):
    '''fit a cumulative trend for x and y direction on the detector to find initial condition for the model'''
    
    ymedian_res2 = []

    for id in np.arange(order_sep+1,n_ord+1):
        mask_id = [orders[i] == id for i in np.arange(len(orders))]
        ymedian_res2.append(np.median(np.array(resolution)[mask_id]))
    ymedian_res = pd.Series(ymedian_res2.copy()).rolling(10, min_periods=1).mean()
    ymedian_res_error = pd.Series(ymedian_res2.copy()).rolling(10, min_periods=1).apply(mad)

    df_xmedian_res = pd.DataFrame({'pixel': pixels, 'res': resolution})
    df_xmedian_res = df_xmedian_res.sort_values('pixel')
    xmedian_pixel = df_xmedian_res['pixel']
    xmedian_res = df_xmedian_res['res'].copy().rolling(400, min_periods=1).mean()
    xmedian_res_error = df_xmedian_res['res'].copy().rolling(400, min_periods=1).apply(mad)

    # fit average x and y profiles
    func_list = [polynomial_regression_2d_1D,polynomial_regression_3d_1D,polynomial_regression_4d_1D,polynomial_regression_5d_1D]    
    BIC_x = []
    BIC_y = []
    rchi2_x = []
    rchi2_y = []
    fit_x = []
    fit_y = []

    for func in func_list: 
        gmodel = Model(func)
        param_list = gmodel.param_names
        k = len(param_list)
        for par in param_list:
            gmodel.set_param_hint(par, value=1., vary=True)

        ## fit x profile 
        gmodel.set_param_hint(param_list[-1], value=np.median(xmedian_res), vary=True)
        pars = gmodel.make_params()
        fit_medianx = gmodel.fit(xmedian_res[100:-100], t=df_xmedian_res['pixel'][100:-100], params=pars, weights = 1/xmedian_res_error[100:-100])
        #BIC_list.append(fit_median.bic)
        BIC_x.append(BIC_calc(xmedian_res[100:-100],xmedian_res_error[100:-100],fit_medianx.best_fit,k))
        rchi2_x.append(fit_medianx.redchi)
        fit_x.append(fit_medianx)

        ## fit the y profile for initial conditions 
        gmodel.set_param_hint(param_list[-1], value=np.median(ymedian_res), vary=True)
        pars = gmodel.make_params()
        fit_mediany = gmodel.fit(ymedian_res[1:-1], t=np.arange(order_sep + 1, n_ord + 1)[1:-1], params=pars, weights = 1/ymedian_res_error[1:-1])
        BIC_y.append(BIC_calc(ymedian_res[1:-1],ymedian_res_error[1:-1],fit_mediany.best_fit,k))
        #BIC_list.append(fit_median.bic)
        rchi2_y.append(fit_mediany.redchi)
        fit_y.append(fit_mediany)
    
    result = [[ymedian_res, ymedian_res_error, xmedian_res, xmedian_res_error, xmedian_pixel],[fit_y,BIC_y,fit_x,BIC_x]]
        
    return result



############
## HARPS-N #
############
# author: Sara 
# inst_mode = 'HARPN12', 'HARPN14', 'HARPN21' 
def Instrumental_resolution_HARPN(thar_file_path,THAR_LINE_TABLE_THAR_FP_A, inst_mode, function_map_resolution, degree):

    res_min=90000
    res_max=148000
    
    # number of orders 
    n_ord = 69
    
    # header file
    header = THAR_LINE_TABLE_THAR_FP_A[0].header
    obs_time = header['MJD-OBS']

    # access to the thar-line table
    # quality flag:
    qual = THAR_LINE_TABLE_THAR_FP_A[1].data['qc']
    # 2D resolution table for the thar lines:
    ords = THAR_LINE_TABLE_THAR_FP_A[1].data['order'][np.where(qual != 0)]
    pxs = THAR_LINE_TABLE_THAR_FP_A[1].data['x0'][np.where(qual != 0)]
    res = THAR_LINE_TABLE_THAR_FP_A[1].data['resolution'][np.where(qual != 0)]

    # shaping the detector 
    # HARPS-N works with 1 fiber only (as the MR mode in espresso)
    shape_s2d = shape_s2d_HARPS_N()

    orders_hr = np.arange(1, n_ord + 1, 1, dtype=int)
    pixels_hr = shape_s2d

    matrix_resolution_hr_theo = np.zeros((orders_hr.size, pixels_hr.size))

    # lines exclusion based on line resolution distribution (exclusion of outliers wrt to the mean)
    res_all = np.array(res)
    ords_all = np.array(ords)
    pxs_all = np.array(pxs)

    mean_res = res.mean()
    sigma_res = res.std()
    n_sigma = 3
    n_thar_lines = res_all.size

    keep_condition = lambda i : ( res[i] < (mean_res + n_sigma * sigma_res) and res[i] > (mean_res - n_sigma * sigma_res) ) 
    excl_condition = lambda i : ( res[i] > (mean_res + n_sigma * sigma_res) or res[i] < (mean_res - n_sigma * sigma_res) ) 

    orders = [ords_all[i] for i in np.arange(n_thar_lines) if keep_condition(i) ]
    excl_orders = [ords_all[i] for i in np.arange(n_thar_lines) if excl_condition(i) ]
    pixels = [pxs_all[i] for i in np.arange(n_thar_lines) if keep_condition(i) ]
    excl_pixels = [pxs_all[i] for i in np.arange(n_thar_lines) if excl_condition(i) ]
    resolution = [res_all[i] for i in np.arange(n_thar_lines) if keep_condition(i) ] 
    
    ####################################
    # AVERAGE PROFILES ON THE DETECTOR #
    ####################################

    # create the average res. profile along x and y direction on the detector - to find initial condition for the model
    ymedian_res, ymedian_res_error, xmedian_res, xmedian_res_error, xmedian_pixel = fit_average_resolution_profile(orders,resolution,pixels, n_ord)[0]
    
    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs = axs.flatten()
    axs[0].plot(xmedian_pixel[100:-100],xmedian_res[100:-100], alpha=0.5)
    axs[0].scatter(xmedian_pixel[100:-100],xmedian_res[100:-100])
    axs[0].set_title(inst_mode + ' - overall resolution trend over pixels - blue')
    axs[1].plot(ymedian_res, np.arange(n_ord), alpha=0.5)
    axs[1].scatter(ymedian_res, np.arange(n_ord))
    axs[1].set_title(inst_mode + ' - mean resolution order per order - blue')
    fig.savefig(thar_file_path + inst_mode + '_detector_average_profiles.png', dpi=100, format='png', bbox_inches='tight')

    # fit the average profiles 
    fit_y,BIC_y,fit_x,BIC_x  = fit_average_resolution_profile(orders,resolution,pixels, n_ord)[1]
    
    #plot BICs
    fig, axs = plt.subplots(1,2, figsize = (12,6))
    axs = axs.flatten()
    axs[0].plot(np.arange(4), BIC_x)
    axs[0].set_xticks(np.arange(4),['2d','3d','4d','5d'])
    axs[0].set_title(inst_mode + ' - BICs x profile - blue')
    axs[1].plot(np.arange(4), BIC_y)
    axs[1].set_xticks(np.arange(4),['2d','3d','4d','5d'])
    axs[1].set_title(inst_mode + ' - BICs y profile - blue')
    fig.savefig(thar_file_path + inst_mode + '_detector_average_profiles_BIC.png', dpi=100, format='png', bbox_inches='tight')

    # plot best fit model
    fit_x[BIC_x.index(min(BIC_x))].plot()
    plt.title(inst_mode + ' - best fit model for x profile - blue')
    plt.show()
    fit_y[BIC_y.index(min(BIC_y))].plot()
    plt.title(inst_mode + ' - best fit model for y profile - blue')
    plt.show()
    
    # plot rainbow plot
    f_plot_harpn_thar_line_distribution_rainbow(THAR_LINE_TABLE_THAR_FP_A, inst_mode, thar_file_path, simple_plot=False, xmedian_pixel=xmedian_pixel, xmedian_res = xmedian_res, ymedian_res=ymedian_res)
    

    ################
    # model creation
    ################

    if function_map_resolution == None:
        func_list = [polynomial_regression_2d,polynomial_regression_3d,polynomial_regression_4d,polynomial_regression_espresso]
        BIC = []
        rchi2 = []
        fit = []

        for function_map_resolution in func_list:
            if function_map_resolution == polynomial_regression_2d: degree = '2d'
            elif function_map_resolution == polynomial_regression_3d: degree = '3d'
            elif function_map_resolution == polynomial_regression_4d: degree = '4d'
            elif function_map_resolution == polynomial_regression_espresso: degree = 'espresso'

            gmodel = Model(function_map_resolution)
            param_list = gmodel.param_names
            k = len(param_list)
            for par in param_list:
                gmodel.set_param_hint(par, value=1., vary=True)

            gmodel.set_param_hint(param_list[-1], value=np.median(resolution), vary=True)
            pars = gmodel.make_params()
            result_poly_reg_2d = gmodel.fit(resolution,xy=np.array([pixels, orders]), params=pars)
            BIC.append(result_poly_reg_2d.bic)
            rchi2.append(result_poly_reg_2d.redchi)
            fit.append(result_poly_reg_2d)
            
            # interpolation of thar lines
            for id in orders_hr:
                i = id - 1
                if k == 6: matrix_resolution_hr_theo[i, :] = function_map_resolution(np.array([pixels_hr, i], dtype=object),result_poly_reg_2d.values['a'],result_poly_reg_2d.values['b'],result_poly_reg_2d.values['c'],result_poly_reg_2d.values['d'],result_poly_reg_2d.values['e'],result_poly_reg_2d.values['f'])
                elif k == 7: matrix_resolution_hr_theo[i, :] = function_map_resolution(np.array([pixels_hr, i], dtype=object),result_poly_reg_2d.values['a'],result_poly_reg_2d.values['b'],result_poly_reg_2d.values['c'],result_poly_reg_2d.values['d'],result_poly_reg_2d.values['e'],result_poly_reg_2d.values['f'],result_poly_reg_2d.values['g'])
                elif k == 10:matrix_resolution_hr_theo[i, :] = function_map_resolution(np.array([pixels_hr, i], dtype=object),result_poly_reg_2d.values['a'],result_poly_reg_2d.values['b'],result_poly_reg_2d.values['c'],result_poly_reg_2d.values['d'],result_poly_reg_2d.values['e'],result_poly_reg_2d.values['f'],result_poly_reg_2d.values['g'],result_poly_reg_2d.values['h'],result_poly_reg_2d.values['i'],result_poly_reg_2d.values['l'])
                elif k == 15: matrix_resolution_hr_theo[i, :] = function_map_resolution(np.array([pixels_hr, i], dtype=object),result_poly_reg_2d.values['a'],result_poly_reg_2d.values['b'],result_poly_reg_2d.values['c'],result_poly_reg_2d.values['d'],result_poly_reg_2d.values['e'],result_poly_reg_2d.values['f'],result_poly_reg_2d.values['g'],result_poly_reg_2d.values['h'],result_poly_reg_2d.values['i'],result_poly_reg_2d.values['l'],result_poly_reg_2d.values['m'],result_poly_reg_2d.values['n'],result_poly_reg_2d.values['o'],result_poly_reg_2d.values['p'],result_poly_reg_2d.values['q'])
            
            matrix_resolution_hr_theo = np.where(matrix_resolution_hr_theo == 0., np.nan,matrix_resolution_hr_theo)

            # plot resolution maps
            plot_resolution_map_HARPN([excl_pixels, excl_orders],pixels,orders,resolution,pixels_hr,matrix_resolution_hr_theo,n_ord,thar_file_path,res_min,res_max,inst_mode,obs_time,degree,Doit=True)

            #######################################
            ## store resolution maps in a fits file
            #######################################
            resolution_map = np.zeros((n_ord, shape_s2d.size))

            # the numbering of the order starts at 1: 
            resolution_map[:n_ord,:] = matrix_resolution_hr_theo
            resolution_map = np.where(resolution_map == 0., np.nan, resolution_map)


            prihdu = pyfits.PrimaryHDU(header=header)
            prihdu.writeto(thar_file_path + inst_mode + '_regr_' + degree + '_RESOLUTION_MAP.fits', overwrite=True)
            pyfits.append(thar_file_path + inst_mode + '_regr_' + degree + '_RESOLUTION_MAP.fits', resolution_map)

            print(inst_mode + ' - resolution map - MAD =', int(mad(resolution_map.flatten())) )

        #plot BICs
        fig, ax = plt.subplots(figsize = (8,8))
        ax.plot(np.arange(4), BIC)
        ax.set_xticks(np.arange(4),['2d','3d','4d','espresso'])
        ax.set_title(inst_mode + ' - BICs 2D map - blue')
        fig.savefig(thar_file_path + inst_mode + '_resolution_model_BIC.png', dpi=100, format='png', bbox_inches='tight')
        
    else: 
        gmodel = Model(function_map_resolution)
        param_list = gmodel.param_names
        k = len(param_list)
        for par in param_list:
            gmodel.set_param_hint(par, value=1., vary=True)

        gmodel.set_param_hint(param_list[-1], value=np.median(resolution), vary=True)
        pars = gmodel.make_params()
        result_poly_reg_2d = gmodel.fit(resolution,xy=np.array([pixels, orders]), params=pars)
        
        # interpolation of thar lines
        for id in orders_hr:
            i = id - 1
            if k == 6: matrix_resolution_hr_theo[i, :] = function_map_resolution(np.array([pixels_hr, i], dtype=object),result_poly_reg_2d.values['a'],result_poly_reg_2d.values['b'],result_poly_reg_2d.values['c'],result_poly_reg_2d.values['d'],result_poly_reg_2d.values['e'],result_poly_reg_2d.values['f'])
            elif k == 7: matrix_resolution_hr_theo[i, :] = function_map_resolution(np.array([pixels_hr, i], dtype=object),result_poly_reg_2d.values['a'],result_poly_reg_2d.values['b'],result_poly_reg_2d.values['c'],result_poly_reg_2d.values['d'],result_poly_reg_2d.values['e'],result_poly_reg_2d.values['f'],result_poly_reg_2d.values['g'])
            elif k == 10:matrix_resolution_hr_theo[i, :] = function_map_resolution(np.array([pixels_hr, i], dtype=object),result_poly_reg_2d.values['a'],result_poly_reg_2d.values['b'],result_poly_reg_2d.values['c'],result_poly_reg_2d.values['d'],result_poly_reg_2d.values['e'],result_poly_reg_2d.values['f'],result_poly_reg_2d.values['g'],result_poly_reg_2d.values['h'],result_poly_reg_2d.values['i'],result_poly_reg_2d.values['l'])
            elif k == 15: matrix_resolution_hr_theo[i, :] = function_map_resolution(np.array([pixels_hr, i], dtype=object),result_poly_reg_2d.values['a'],result_poly_reg_2d.values['b'],result_poly_reg_2d.values['c'],result_poly_reg_2d.values['d'],result_poly_reg_2d.values['e'],result_poly_reg_2d.values['f'],result_poly_reg_2d.values['g'],result_poly_reg_2d.values['h'],result_poly_reg_2d.values['i'],result_poly_reg_2d.values['l'],result_poly_reg_2d.values['m'],result_poly_reg_2d.values['n'],result_poly_reg_2d.values['o'],result_poly_reg_2d.values['p'],result_poly_reg_2d.values['q'])
        
        matrix_resolution_hr_theo = np.where(matrix_resolution_hr_theo == 0., np.nan,matrix_resolution_hr_theo)

        # plot resolution maps
        plot_resolution_map_HARPN([excl_pixels, excl_orders],pixels,orders,resolution,pixels_hr,matrix_resolution_hr_theo,n_ord,thar_file_path,res_min,res_max,inst_mode,obs_time,degree,Doit=True)

        #######################################
        ## store resolution maps in a fits file
        #######################################
        resolution_map = np.zeros((n_ord, shape_s2d.size))

        # the numbering of the order starts at 1: 
        resolution_map[:n_ord,:] = matrix_resolution_hr_theo
        resolution_map = np.where(resolution_map == 0., np.nan, resolution_map)


        prihdu = pyfits.PrimaryHDU(header=header)
        prihdu.writeto(thar_file_path + inst_mode + '_regr_' + degree + '_RESOLUTION_MAP.fits', overwrite=True)
        pyfits.append(thar_file_path + inst_mode + '_regr_' + degree + '_RESOLUTION_MAP.fits', resolution_map)

        print(inst_mode + ' - resolution map - MAD =', int(mad(resolution_map.flatten())) )



    return [excl_orders,excl_pixels],[xmedian_pixel,xmedian_res,ymedian_res]

def fit_average_resolution_profile(orders,resolution,pixels,n_ord): 
    '''fit a cumulative trend for x and y direction on the detector to find initial condition for the model'''
    # create average x and y profiles
    ymedian_res2 = []
    for id in np.arange(n_ord):
        id += 1 
        mask_id = [orders[i] == id for i in np.arange(len(orders))]
        ymedian_res2.append(np.median(np.array(resolution)[mask_id]))
    ymedian_res = pd.Series(ymedian_res2.copy()).rolling(15, min_periods=1).mean()
    ymedian_res_error = pd.Series(ymedian_res2.copy()).rolling(15, min_periods=1).apply(mad)

    df_xmedian_res = pd.DataFrame({'pixel': pixels, 'res': resolution})
    df_xmedian_res = df_xmedian_res.sort_values('pixel')
    xmedian_pixel = df_xmedian_res['pixel']
    xmedian_res = df_xmedian_res['res'].copy().rolling(400, min_periods=1).mean()
    xmedian_res_error = df_xmedian_res['res'].copy().rolling(400, min_periods=1).apply(mad)

    # fit average x and y profiles
    func_list = [polynomial_regression_2d_1D,polynomial_regression_3d_1D,polynomial_regression_4d_1D,polynomial_regression_5d_1D]    
    BIC_x = []
    BIC_y = []
    rchi2_x = []
    rchi2_y = []
    fit_x = []
    fit_y = []

    for func in func_list: 
        gmodel = Model(func)
        param_list = gmodel.param_names
        k = len(param_list)
        for par in param_list:
            gmodel.set_param_hint(par, value=1., vary=True)

        ## fit x profile 
        gmodel.set_param_hint(param_list[-1], value=np.median(xmedian_res), vary=True)
        pars = gmodel.make_params()
        fit_medianx = gmodel.fit(xmedian_res[100:-100], t=df_xmedian_res['pixel'][100:-100], params=pars, weights = 1/xmedian_res_error[100:-100])
        #BIC_list.append(fit_median.bic)
        BIC_x.append(BIC_calc(xmedian_res[100:-100],xmedian_res_error[100:-100],fit_medianx.best_fit,k))
        rchi2_x.append(fit_medianx.redchi)
        fit_x.append(fit_medianx)

        ## fit the y profile for initial conditions 
        gmodel.set_param_hint(param_list[-1], value=np.median(ymedian_res), vary=True)
        pars = gmodel.make_params()
        fit_mediany = gmodel.fit(ymedian_res[5:-5], t=np.arange(1, n_ord + 1)[5:-5], params=pars, weights = 1/ymedian_res_error[5:-5])
        BIC_y.append(BIC_calc(ymedian_res[5:-5],ymedian_res_error[5:-5],fit_mediany.best_fit,k))
        #BIC_list.append(fit_median.bic)
        rchi2_y.append(fit_mediany.redchi)
        fit_y.append(fit_mediany)
    
    result = [[ymedian_res, ymedian_res_error, xmedian_res, xmedian_res_error, xmedian_pixel],[fit_y,BIC_y,fit_x,BIC_x]]
    
    return result




########################
## ESPRESSO and NIRPS ##
########################
# author: Romain
def Instrumental_resolution_ESPRESSO(THAR_LINE_TABLE_THAR_FP_A, function_map_resolution, resolution_map_path='/Users/saratavella/Desktop/phd/ANTARESS/resolution_maps/'):
    res_min=100000#120000
    res_max=170000#240000

    header = THAR_LINE_TABLE_THAR_FP_A[0].header

    Time = header['MJD-OBS']

    qual = THAR_LINE_TABLE_THAR_FP_A[1].data['qc']
    ords = THAR_LINE_TABLE_THAR_FP_A[1].data['order'][np.where(qual != 0)]
    pxs = THAR_LINE_TABLE_THAR_FP_A[1].data['x0'][np.where(qual != 0)]
    res = THAR_LINE_TABLE_THAR_FP_A[1].data['resolution'][np.where(qual != 0)]

    if header['HIERARCH ESO INS MODE'] == 'MULTIMR':
        shape_s2d=shape_s2d_ESPRESSO(time_science=Time,ins_mode = header['HIERARCH ESO INS MODE'],bin_x=header['HIERARCH ESO DET BINX'])
        shape_s2d=shape_s2d_ESPRESSO(time_science=Time,ins_mode = header['HIERARCH ESO INS MODE'],bin_x=header['HIERARCH ESO DET BINX'])
        
        orders_blue_hr_slice1 = np.arange(1, 45.01, 1, dtype=int)
        pixels_blue_hr = shape_s2d[0]

        orders_red_hr_slice1 = np.arange(46, 85.01, 1, dtype=int)
        pixels_red_hr = shape_s2d[1]

        matrix_resolution_blue_hr_theo_slice1 = np.zeros((len(np.arange(1, 45.01, 1)), len(pixels_blue_hr)))
        matrix_resolution_red_hr_theo_slice1 = np.zeros((len(np.arange(46, 85.01, 1)), len(pixels_red_hr)))

        orders_blue_slice1 = ords[(ords <= 45)]
        pixels_blue_slice1 = pxs[(ords <= 45)]
        resolution_blue_slice1 = res[(ords <= 45)]

        orders_red_slice1 = ords[(ords > 45)]
        pixels_red_slice1 = pxs[(ords > 45)]
        resolution_red_slice1 = res[(ords > 45)]

        gmodel = Model(function_map_resolution)
        gmodel.set_param_hint('a', value=1., vary=True)
        gmodel.set_param_hint('b', value=1., vary=True)
        gmodel.set_param_hint('c', value=1., vary=True)
        gmodel.set_param_hint('d', value=1., vary=True)
        gmodel.set_param_hint('e', value=1., vary=True)
        gmodel.set_param_hint('f', value=1., vary=True)
        gmodel.set_param_hint('g', value=np.median(resolution_blue_slice1), vary=True)
        pars = gmodel.make_params()
        result_poly_reg_2d_blue_slice1 = gmodel.fit(resolution_blue_slice1,xy=np.array([pixels_blue_slice1, orders_blue_slice1]), params=pars)

        gmodel.set_param_hint('g', value=np.median(resolution_red_slice1), vary=True)
        pars = gmodel.make_params()
        result_poly_reg_2d_red_slice1 = gmodel.fit(resolution_red_slice1, xy=np.array([pixels_red_slice1, orders_red_slice1]), params=pars)

        for i in orders_blue_hr_slice1:
            matrix_resolution_blue_hr_theo_slice1[i - 1, :] = function_map_resolution(np.array([pixels_blue_hr, i-1]),result_poly_reg_2d_blue_slice1.values['a'],result_poly_reg_2d_blue_slice1.values['b'],result_poly_reg_2d_blue_slice1.values['c'],result_poly_reg_2d_blue_slice1.values['d'],result_poly_reg_2d_blue_slice1.values['e'],result_poly_reg_2d_blue_slice1.values['f'],result_poly_reg_2d_blue_slice1.values['g'])

        for i in orders_red_hr_slice1:
            matrix_resolution_red_hr_theo_slice1[i - 46, :] = function_map_resolution(np.array([pixels_red_hr, i-1]),result_poly_reg_2d_red_slice1.values['a'],result_poly_reg_2d_red_slice1.values['b'],result_poly_reg_2d_red_slice1.values['c'],result_poly_reg_2d_red_slice1.values['d'],result_poly_reg_2d_red_slice1.values['e'],result_poly_reg_2d_red_slice1.values['f'],result_poly_reg_2d_red_slice1.values['g'])

        matrix_resolution_blue_hr_theo_slice1 = np.where(matrix_resolution_blue_hr_theo_slice1 == 0., np.nan,matrix_resolution_blue_hr_theo_slice1)
        matrix_resolution_red_hr_theo_slice1 = np.where(matrix_resolution_red_hr_theo_slice1 == 0., np.nan,matrix_resolution_red_hr_theo_slice1)

        plot_resolution_map_MR(pixels_blue_slice1,orders_blue_slice1,resolution_blue_slice1,pixels_blue_hr,matrix_resolution_blue_hr_theo_slice1,
                               pixels_red_slice1,orders_red_slice1,resolution_red_slice1,pixels_red_hr,matrix_resolution_red_hr_theo_slice1,
                               res_min,res_max,header['HIERARCH ESO INS MODE'],header['HIERARCH ESO DET BINX'],header['MJD-OBS'],
                               Doit=True,resolution_map_path=resolution_map_path)

        resolution_map = np.zeros((85, len(shape_s2d[1])))

        resolution_map[:45, int(pixels_blue_hr[0]):int(pixels_blue_hr[-1]) + 1] = matrix_resolution_blue_hr_theo_slice1
        resolution_map[45:, :] = matrix_resolution_red_hr_theo_slice1
        resolution_map = np.where(resolution_map == 0., np.nan, resolution_map)

    else:
        shape_s2d=shape_s2d_ESPRESSO(time_science=Time,ins_mode = header['HIERARCH ESO INS MODE'],bin_x=header['HIERARCH ESO DET BINX'])
        orders_blue_hr_slice1 = np.arange(1, 90.01, 1, dtype=int)[::2]
        orders_blue_hr_slice2 = np.arange(1, 90.01, 1, dtype=int)[1::2]
        pixels_blue_hr = shape_s2d[0]

        orders_red_hr_slice1 = np.arange(91, 170.01, 1, dtype=int)[::2]
        orders_red_hr_slice2 = np.arange(91, 170.01, 1, dtype=int)[1::2]
        pixels_red_hr = shape_s2d[1]

        matrix_resolution_blue_hr_theo_slice1 = np.zeros((len(np.arange(1, 90.01, 1)), len(pixels_blue_hr)))
        matrix_resolution_blue_hr_theo_slice2 = np.zeros((len(np.arange(1, 90.01, 1)), len(pixels_blue_hr)))
        matrix_resolution_red_hr_theo_slice1 = np.zeros((len(np.arange(91, 170.01, 1)), len(pixels_red_hr)))
        matrix_resolution_red_hr_theo_slice2 = np.zeros((len(np.arange(91, 170.01, 1)), len(pixels_red_hr)))

        orders_blue_slice1 = ords[(ords <= 90) & (ords % 2 != 0)]
        pixels_blue_slice1 = pxs[(ords <= 90) & (ords % 2 != 0)]
        resolution_blue_slice1 = res[(ords <= 90) & (ords % 2 != 0)]

        orders_blue_slice2 = ords[(ords <= 90) & (ords % 2 == 0)]
        pixels_blue_slice2 = pxs[(ords <= 90) & (ords % 2 == 0)]
        resolution_blue_slice2 = res[(ords <= 90) & (ords % 2 == 0)]

        orders_red_slice1 = ords[(ords > 90) & (ords % 2 != 0)]
        pixels_red_slice1 = pxs[(ords > 90) & (ords % 2 != 0)]
        resolution_red_slice1 = res[(ords > 90) & (ords % 2 != 0)]

        orders_red_slice2 = ords[(ords > 90) & (ords % 2 == 0)]
        pixels_red_slice2 = pxs[(ords > 90) & (ords % 2 == 0)]
        resolution_red_slice2 = res[(ords > 90) & (ords % 2 == 0)]

        gmodel = Model(function_map_resolution)
        gmodel.set_param_hint('a', value=1., vary=True)
        gmodel.set_param_hint('b', value=1., vary=True)
        gmodel.set_param_hint('c', value=1., vary=True)
        gmodel.set_param_hint('d', value=1., vary=True)
        gmodel.set_param_hint('e', value=1., vary=True)
        gmodel.set_param_hint('f', value=1., vary=True)
        gmodel.set_param_hint('g', value=np.median(resolution_blue_slice1), vary=True)
        pars = gmodel.make_params()
        result_poly_reg_2d_blue_slice1 = gmodel.fit(resolution_blue_slice1,xy=np.array([pixels_blue_slice1, orders_blue_slice1]), params=pars)

        gmodel.set_param_hint('g', value=np.median(resolution_blue_slice2), vary=True)
        pars = gmodel.make_params()
        result_poly_reg_2d_blue_slice2 = gmodel.fit(resolution_blue_slice2,xy=np.array([pixels_blue_slice2, orders_blue_slice2]), params=pars)

        gmodel.set_param_hint('g', value=np.median(resolution_red_slice1), vary=True)
        pars = gmodel.make_params()
        result_poly_reg_2d_red_slice1 = gmodel.fit(resolution_red_slice1,xy=np.array([pixels_red_slice1, orders_red_slice1]), params=pars)

        gmodel.set_param_hint('g', value=np.median(resolution_red_slice2), vary=True)
        pars = gmodel.make_params()
        result_poly_reg_2d_red_slice2 = gmodel.fit(resolution_red_slice2,xy=np.array([pixels_red_slice2, orders_red_slice2]), params=pars)

        for i in orders_blue_hr_slice1:
            matrix_resolution_blue_hr_theo_slice1[i - 1, :] = function_map_resolution(np.array([pixels_blue_hr, i-1]),result_poly_reg_2d_blue_slice1.values['a'],result_poly_reg_2d_blue_slice1.values['b'],result_poly_reg_2d_blue_slice1.values['c'],result_poly_reg_2d_blue_slice1.values['d'],result_poly_reg_2d_blue_slice1.values['e'],result_poly_reg_2d_blue_slice1.values['f'],result_poly_reg_2d_blue_slice1.values['g'])

        for i in orders_blue_hr_slice2:
            matrix_resolution_blue_hr_theo_slice2[i - 1, :] = function_map_resolution(np.array([pixels_blue_hr, i-1]),result_poly_reg_2d_blue_slice2.values['a'],result_poly_reg_2d_blue_slice2.values['b'],result_poly_reg_2d_blue_slice2.values['c'],result_poly_reg_2d_blue_slice2.values['d'],result_poly_reg_2d_blue_slice2.values['e'],result_poly_reg_2d_blue_slice2.values['f'],result_poly_reg_2d_blue_slice2.values['g'])

        for i in orders_red_hr_slice1:
            matrix_resolution_red_hr_theo_slice1[i - 91, :] = function_map_resolution(np.array([pixels_red_hr, i-1]),result_poly_reg_2d_red_slice1.values['a'],result_poly_reg_2d_red_slice1.values['b'],result_poly_reg_2d_red_slice1.values['c'],result_poly_reg_2d_red_slice1.values['d'],result_poly_reg_2d_red_slice1.values['e'],result_poly_reg_2d_red_slice1.values['f'],result_poly_reg_2d_red_slice1.values['g'])

        for i in orders_red_hr_slice2:
            matrix_resolution_red_hr_theo_slice2[i - 91, :] = function_map_resolution(np.array([pixels_red_hr, i-1]),result_poly_reg_2d_red_slice2.values['a'],result_poly_reg_2d_red_slice2.values['b'],result_poly_reg_2d_red_slice2.values['c'],result_poly_reg_2d_red_slice2.values['d'],result_poly_reg_2d_red_slice2.values['e'],result_poly_reg_2d_red_slice2.values['f'],result_poly_reg_2d_red_slice2.values['g'])

        matrix_resolution_blue_hr_theo_slice1 = np.where(matrix_resolution_blue_hr_theo_slice1 == 0., np.nan,matrix_resolution_blue_hr_theo_slice1)
        matrix_resolution_blue_hr_theo_slice2 = np.where(matrix_resolution_blue_hr_theo_slice2 == 0., np.nan,matrix_resolution_blue_hr_theo_slice2)
        matrix_resolution_red_hr_theo_slice1 = np.where(matrix_resolution_red_hr_theo_slice1 == 0., np.nan,matrix_resolution_red_hr_theo_slice1)
        matrix_resolution_red_hr_theo_slice2 = np.where(matrix_resolution_red_hr_theo_slice2 == 0., np.nan,matrix_resolution_red_hr_theo_slice2)

        plot_resolution_map_slice_UHR_HR(pixels_blue_slice1,orders_blue_slice1,resolution_blue_slice1,pixels_blue_hr,matrix_resolution_blue_hr_theo_slice1,pixels_red_slice1,
                                         orders_red_slice1,resolution_red_slice1,pixels_red_hr,matrix_resolution_red_hr_theo_slice1,res_min,res_max,
                                         pixels_blue_slice2,orders_blue_slice2,resolution_blue_slice2,matrix_resolution_blue_hr_theo_slice2,pixels_red_slice2,
                                         orders_red_slice2,resolution_red_slice2,matrix_resolution_red_hr_theo_slice2,header['HIERARCH ESO INS MODE'],header['HIERARCH ESO DET BINX'],header['MJD-OBS'],
                                         Doit=True,resolution_map_path=resolution_map_path)
        plot_resolution_map_UHR_HR(pixels_blue_slice1,orders_blue_slice1,resolution_blue_slice1,pixels_blue_hr,matrix_resolution_blue_hr_theo_slice1,pixels_red_slice1,
                                         orders_red_slice1,resolution_red_slice1,pixels_red_hr,matrix_resolution_red_hr_theo_slice1,res_min,res_max,
                                         pixels_blue_slice2,orders_blue_slice2,resolution_blue_slice2,matrix_resolution_blue_hr_theo_slice2,pixels_red_slice2,
                                         orders_red_slice2,resolution_red_slice2,matrix_resolution_red_hr_theo_slice2,header['HIERARCH ESO INS MODE'],header['HIERARCH ESO DET BINX'],header['MJD-OBS'],
                                         Doit=True,resolution_map_path=resolution_map_path)

        resolution_map = np.zeros((170, len(shape_s2d[1])))

        resolution_map[:90, int(pixels_blue_hr[0]):int(pixels_blue_hr[-1]) + 1] = np.where(
            np.isnan(matrix_resolution_blue_hr_theo_slice1), matrix_resolution_blue_hr_theo_slice2,
            matrix_resolution_blue_hr_theo_slice1)
        resolution_map[90:, :] = np.where(np.isnan(matrix_resolution_red_hr_theo_slice1),
                                          matrix_resolution_red_hr_theo_slice2, matrix_resolution_red_hr_theo_slice1)
        resolution_map = np.where(resolution_map == 0., np.nan, resolution_map)
    prihdu = pyfits.PrimaryHDU(header=THAR_LINE_TABLE_THAR_FP_A[0].header)
    prihdu.writeto(resolution_map_path+'r.' + header['ARCFILE'][:-5] + '_RESOLUTION_MAP.fits', overwrite=True)
    pyfits.append(resolution_map_path+'r.' + header['ARCFILE'][:-5] + '_RESOLUTION_MAP.fits', resolution_map)
    return





