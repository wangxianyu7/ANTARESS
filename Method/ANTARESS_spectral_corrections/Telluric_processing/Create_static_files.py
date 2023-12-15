#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 20:26:21 2018

@author: allartromain
"""

import numpy as np
import pylab as pl
import os
from astropy.io import fits 
import matplotlib as mpl
import time
from lmfit import Model


def find_nearest(array, values):
    array = np.asarray(array)

    # the last dim must be 1 to broadcast in (array - values) below.
    values = np.expand_dims(values, axis=-1) 

    indices = np.abs(array - values).argmin(axis=-1)

    return indices

def read_hitran2012_parfile(filename):
    '''
    Given a HITRAN2012-format text file, read in the parameters of the molecular absorption features.
    Parameters
    ----------
    filename : str
        The filename to read in.
    Return
    ------
    data : dict
        The dictionary of HITRAN data for the molecule.
    '''

    if not os.path.exists:
        raise ImportError('The input filename"' + filename + '" does not exist.')

    if filename.endswith('.zip'):
        import zipfile
        zip = zipfile.ZipFile(filename, 'r')
        (object_name, ext) = os.path.splitext(os.path.basename(filename))
        print((object_name, ext))
        filehandle = zip.read(object_name).splitlines()
    else:
        filehandle = open(filename, 'r')

    data = {'M': [],  ## molecule identification number
            'I': [],  ## isotope number
            'linecenter': [],  ## line center wavenumber (in cm^{-1})
            'S': [],  ## line strength, in cm^{-1} / (molecule m^{-2})
            'Acoeff': [],  ## Einstein A coefficient (in s^{-1})
            'gamma-air': [],  ## line HWHM for air-broadening
            'gamma-self': [],  ## line HWHM for self-emission-broadening
            'Epp': [],  ## energy of lower transition level (in cm^{-1})
            'N': [],  ## temperature-dependent exponent for "gamma-air"
            'delta': [],  ## air-pressure shift, in cm^{-1} / atm
            'Vp': [],  ## upper-state "global" quanta index
            'Vpp': [],  ## lower-state "global" quanta index
            'Qp': [],  ## upper-state "local" quanta index
            'Qpp': [],  ## lower-state "local" quanta index
            'Ierr': [],  ## uncertainty indices
            'Iref': [],  ## reference indices
            'flag': [],  ## flag
            'gp': [],  ## statistical weight of the upper state
            'gpp': []}  ## statistical weight of the lower state

    print(('Reading "' + filename + '" ...'))

    for line in filehandle:
        if (len(line) < 160):
            raise ImportError(
                'The imported file ("' + filename + '") does not appear to be a HITRAN2012-format data file.')

        data['M'].append(np.uint(line[0:2]))
        data['I'].append(np.uint(line[2]))
        data['linecenter'].append(np.float64(line[3:15]))
        data['S'].append(np.float64(line[15:25]))
        data['Acoeff'].append(np.float64(line[25:35]))
        data['gamma-air'].append(np.float64(line[35:40]))
        data['gamma-self'].append(np.float64(line[40:45]))
        data['Epp'].append(np.float64(line[45:55]))
        data['N'].append(np.float64(line[55:59]))
        data['delta'].append(np.float64(line[59:67]))
        data['Vp'].append(line[67:82])
        data['Vpp'].append(line[82:97])
        data['Qp'].append(line[97:112])
        data['Qpp'].append(line[112:127])
        data['Ierr'].append(line[127:133])
        data['Iref'].append(line[133:145])
        data['flag'].append(line[145])
        data['gp'].append(line[146:153])
        data['gpp'].append(line[153:160])

    if filename.endswith('.zip'):
        zip.close()
    else:
        filehandle.close()

    for key in data:
        data[key] = np.array(data[key])

    return (data)

"""
def static_hitran_file(hitran_file, QT_file, species, S_cut=10 ** -26,instrument='ESPRESSO'):
    hitran_database = read_hitran2012_parfile(hitran_file)
    qt_file = np.genfromtxt(QT_file)
    
    wave_number = hitran_database['linecenter'][hitran_database['S'] > S_cut]
    Intensity   = hitran_database['S'][hitran_database['S'] > S_cut]
    gamma_air   = hitran_database['gamma-air'][hitran_database['S'] > S_cut]
    gamma_self  = hitran_database['gamma-self'][hitran_database['S'] > S_cut]
    N           = hitran_database['N'][hitran_database['S'] > S_cut]
    delta       = hitran_database['delta'][hitran_database['S'] > S_cut]
    Epp         = hitran_database['Epp'][hitran_database['S'] > S_cut]
    
    removed_lines = [13570.9,13704.525,13708.60,13708.625,13766.465,13941.252,13947.17,13928.5,15437.85,15437.89,15437.9,
                     13701.8,13910.16]
    
    
    col1 = fits.Column(name='wave_number', format='1D', array=np.delete(wave_number,find_nearest(wave_number, removed_lines)))
    col2 = fits.Column(name='Intensity',   format='1D', array=np.delete(Intensity,find_nearest(wave_number, removed_lines)))
    col3 = fits.Column(name='gamma_air',   format='1D', array=np.delete(gamma_air,find_nearest(wave_number, removed_lines)))
    col3b = fits.Column(name='gamma_self', format='1D', array=np.delete(gamma_self,find_nearest(wave_number, removed_lines)))
    col4 = fits.Column(name='N',           format='1D', array=np.delete(N,find_nearest(wave_number, removed_lines)))
    col5 = fits.Column(name='delta',       format='1D', array=np.delete(delta,find_nearest(wave_number, removed_lines)))
    col6 = fits.Column(name='Epp',         format='1D', array=np.delete(Epp,find_nearest(wave_number, removed_lines)))

    col7 = fits.Column(name='Temperature', format='1D', array=qt_file[:, 0])
    col8 = fits.Column(name='Qt', format='1D', array=qt_file[:, 1])

    cols0 = fits.ColDefs([col1, col2, col3, col3b, col4, col5, col6])
    tbhdu0 = fits.BinTableHDU.from_columns(cols0)

    cols1 = fits.ColDefs([col7, col8])
    tbhdu1 = fits.BinTableHDU.from_columns(cols1)
    prihdr = fits.Header()
    prihdr['Intensity Cut'] = str(S_cut)
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu0, tbhdu1])
    thdulist.writeto('Static_model/'+instrument+'/Static_hitran_qt_'+species+'.fits',
                     overwrite=True)
    return
"""
#(tavella)
def static_hitran_file(hitran_file, QT_file, species, S_cut=10 ** -26,instrument='HARPS_N'):
    hitran_database = read_hitran2012_parfile(hitran_file)
    qt_file = np.genfromtxt(QT_file)
    
    wave_number = hitran_database['linecenter'][hitran_database['S'] > S_cut]
    Intensity   = hitran_database['S'][hitran_database['S'] > S_cut]
    gamma_air   = hitran_database['gamma-air'][hitran_database['S'] > S_cut]
    gamma_self  = hitran_database['gamma-self'][hitran_database['S'] > S_cut]
    N           = hitran_database['N'][hitran_database['S'] > S_cut]
    delta       = hitran_database['delta'][hitran_database['S'] > S_cut]
    Epp         = hitran_database['Epp'][hitran_database['S'] > S_cut]
    
    removed_lines = [13570.9,13704.525,13708.60,13708.625,13766.465,13941.252,13947.17,13928.5,15437.85,15437.89,15437.9,
                     13701.8,13910.16]
    
    
    col1 = fits.Column(name='wave_number', format='1D', array=np.delete(wave_number,find_nearest(wave_number, removed_lines)))
    col2 = fits.Column(name='Intensity',   format='1D', array=np.delete(Intensity,find_nearest(wave_number, removed_lines)))
    col3 = fits.Column(name='gamma_air',   format='1D', array=np.delete(gamma_air,find_nearest(wave_number, removed_lines)))
    col3b = fits.Column(name='gamma_self', format='1D', array=np.delete(gamma_self,find_nearest(wave_number, removed_lines)))
    col4 = fits.Column(name='N',           format='1D', array=np.delete(N,find_nearest(wave_number, removed_lines)))
    col5 = fits.Column(name='delta',       format='1D', array=np.delete(delta,find_nearest(wave_number, removed_lines)))
    col6 = fits.Column(name='Epp',         format='1D', array=np.delete(Epp,find_nearest(wave_number, removed_lines)))

    col7 = fits.Column(name='Temperature', format='1D', array=qt_file[:, 0])
    col8 = fits.Column(name='Qt', format='1D', array=qt_file[:, 1])

    cols0 = fits.ColDefs([col1, col2, col3, col3b, col4, col5, col6])
    tbhdu0 = fits.BinTableHDU.from_columns(cols0)

    cols1 = fits.ColDefs([col7, col8])
    tbhdu1 = fits.BinTableHDU.from_columns(cols1)
    prihdr = fits.Header()
    prihdr['Intensity Cut'] = str(S_cut)
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu0, tbhdu1])
    thdulist.writeto('Static_model/'+instrument+'/Static_hitran_qt_'+species+'.fits',
                     overwrite=True)
    return


"""

def lines_to_fit(input_file,species,wavenumber_born=np.array([10**8/7000,10**8/8000]),number_of_lines=10,region_excluded=np.array([10**8/12680,10**8/12710]),lines_excluded=np.array([1.39149595e+04,1.38726811e+04,1.38184074e+04]),instrument='ESPRESSO'):
    "wave_born, region_excluded and lines_excluded are in wavenumber in cm-1"
    hitran_database = fits.open(input_file)[1].data
    
    wave_number_hitran_rest_temp = hitran_database['wave_number'][(wavenumber_born[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wavenumber_born[0])]
    intensity_hitran_temp        = hitran_database['Intensity'][(wavenumber_born[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wavenumber_born[0])]
    gamma_air_temp               = hitran_database['gamma_air'][(wavenumber_born[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wavenumber_born[0])]
    n_air_temp                   = hitran_database['N'][(wavenumber_born[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wavenumber_born[0])]
    delta_air_temp               = hitran_database['delta'][(wavenumber_born[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wavenumber_born[0])]
    Epp_temp                     = hitran_database['Epp'][(wavenumber_born[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wavenumber_born[0])]
        
   	##############################################
   	####    Selection of the strongest lines  ####
   	##############################################

    wave_number_hitran_rest_Nstrong          = []
    wave_number_hitran_rest_temp_order       = wave_number_hitran_rest_temp[np.argsort(intensity_hitran_temp)]
    wave_number_hitran_rest_excluded         = []


    for sw in range(len(wave_number_hitran_rest_temp_order)):
        idx = -1-sw
        #if (np.round(wave_number_hitran_rest_temp_order[idx],4) not in np.round(lines_excluded,4)) & ((wave_number_hitran_rest_temp_order[idx] > region_excluded[0]) | (wave_number_hitran_rest_temp_order[idx] < region_excluded[1])):
        if (np.sum((wave_number_hitran_rest_temp_order[idx] < 10**8/(10**8/lines_excluded + 2*40./300000*np.mean(10**8/lines_excluded))) | (wave_number_hitran_rest_temp_order[idx] > 10**8/(10**8/lines_excluded - 2*40./300000*np.mean(10**8/lines_excluded))))==len(lines_excluded)) & ((wave_number_hitran_rest_temp_order[idx] > region_excluded[0]) | (wave_number_hitran_rest_temp_order[idx] < region_excluded[1])):
            if (len(wave_number_hitran_rest_Nstrong) == 0):
                wave_number_hitran_rest_Nstrong = np.concatenate((wave_number_hitran_rest_Nstrong,[wave_number_hitran_rest_temp_order[idx]]))
            elif (len(wave_number_hitran_rest_Nstrong) != 0)  :
                rejection_criteria=0
                for swl in range(len(wave_number_hitran_rest_Nstrong)):
                    if (wave_number_hitran_rest_temp_order[idx] > 10**8/(10**8/wave_number_hitran_rest_Nstrong[swl] + 2*40./300000*np.mean(10**8/wavenumber_born))) and (wave_number_hitran_rest_temp_order[idx] < 10**8/(10**8/wave_number_hitran_rest_Nstrong[swl] - 2*40./300000*np.mean(10**8/wavenumber_born)))  :  #avoid two strong telluric line blended
                        rejection_criteria+=1
                if (rejection_criteria ==0) :
                    wave_number_hitran_rest_Nstrong = np.concatenate((wave_number_hitran_rest_Nstrong,[wave_number_hitran_rest_temp_order[idx]]))
                else:
                    wave_number_hitran_rest_excluded = np.concatenate((wave_number_hitran_rest_excluded,[wave_number_hitran_rest_temp_order[idx]]))
            if len(wave_number_hitran_rest_Nstrong) >=number_of_lines:
                break
        else:
            wave_number_hitran_rest_excluded = np.concatenate((wave_number_hitran_rest_excluded,[wave_number_hitran_rest_temp_order[idx]]))

    wave_number_hitran_rest_Nstrong  = np.array(wave_number_hitran_rest_Nstrong)
    wave_number_hitran_rest_excluded = np.array(wave_number_hitran_rest_excluded)


    wave_number_hitran_rest_Nstrong_below = 1./((1./(wave_number_hitran_rest_Nstrong*10**-8) - 40./300000*np.mean(10**8/wavenumber_born)) * 10**-8 )
    wave_number_hitran_rest_Nstrong_upper = 1./((1./(wave_number_hitran_rest_Nstrong*10**-8) + 40./300000*np.mean(10**8/wavenumber_born)) * 10**-8 )

   	##################################################################
   	####    Selection of all the lines around the strongest ones  ####
   	##################################################################

    wave_number_hitran_rest = []
    intensity_hitran        = []
    gamma_air               = []
    n_air                   = []
    delta_air               = []
    Epp                     = []

    for wn in range(len(wave_number_hitran_rest_Nstrong)): # select all the lines around the N strongest
        wave_number_hitran_rest = np.concatenate(( wave_number_hitran_rest, wave_number_hitran_rest_temp[(wave_number_hitran_rest_Nstrong_below[wn] > wave_number_hitran_rest_temp) &  (wave_number_hitran_rest_temp > wave_number_hitran_rest_Nstrong_upper[wn])] ))
        intensity_hitran        = np.concatenate(( intensity_hitran,        intensity_hitran_temp[(wave_number_hitran_rest_Nstrong_below[wn] > wave_number_hitran_rest_temp) &  (wave_number_hitran_rest_temp > wave_number_hitran_rest_Nstrong_upper[wn])] ))
        gamma_air               = np.concatenate(( gamma_air,               gamma_air_temp[(wave_number_hitran_rest_Nstrong_below[wn] > wave_number_hitran_rest_temp) &  (wave_number_hitran_rest_temp > wave_number_hitran_rest_Nstrong_upper[wn])] ))
        n_air                   = np.concatenate(( n_air,                   n_air_temp[(wave_number_hitran_rest_Nstrong_below[wn] > wave_number_hitran_rest_temp) &  (wave_number_hitran_rest_temp > wave_number_hitran_rest_Nstrong_upper[wn])] ))
        delta_air               = np.concatenate(( delta_air,               delta_air_temp[(wave_number_hitran_rest_Nstrong_below[wn] > wave_number_hitran_rest_temp) &  (wave_number_hitran_rest_temp > wave_number_hitran_rest_Nstrong_upper[wn])] ))
        Epp                     = np.concatenate(( Epp,                     Epp_temp[(wave_number_hitran_rest_Nstrong_below[wn] > wave_number_hitran_rest_temp) &  (wave_number_hitran_rest_temp > wave_number_hitran_rest_Nstrong_upper[wn])] ))



    col1 = fits.Column(name='wave_number', format='1D', array=wave_number_hitran_rest)
    col2 = fits.Column(name='Intensity',   format='1D', array=intensity_hitran)
    col3 = fits.Column(name='gamma_air',   format='1D', array=gamma_air)
    col4 = fits.Column(name='N',           format='1D', array=n_air)
    col5 = fits.Column(name='delta',       format='1D', array=delta_air)
    col6 = fits.Column(name='Epp',         format='1D', array=Epp)

    col7 = fits.Column(name='CCF_lines_position_wavenumber', format='1D', array=wave_number_hitran_rest_Nstrong)
    col8 = fits.Column(name='CCF_lines_position_wavelength', format='1D', array=10**8/wave_number_hitran_rest_Nstrong)

    col9 = fits.Column(name='Excluded_lines_position_wavenumber', format='1D', array=wave_number_hitran_rest_excluded)
    col10 = fits.Column(name='Excluded_lines_position_wavelength', format='1D', array=10**8/wave_number_hitran_rest_excluded)

    cols0 = fits.ColDefs([col1, col2, col3, col4, col5, col6])
    tbhdu0 = fits.BinTableHDU.from_columns(cols0)

    cols1 = fits.ColDefs([col7, col8])
    tbhdu1 = fits.BinTableHDU.from_columns(cols1)

    cols2 = fits.ColDefs([col9, col10])
    tbhdu2 = fits.BinTableHDU.from_columns(cols2)    
    
    prihdr = fits.Header()
    prihdr['Number of lines'] = str(number_of_lines)
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu0, tbhdu1, tbhdu2])
    thdulist.writeto('Static_model/'+instrument+'/Static_hitran_strongest_lines_'+species+'.fits', overwrite=True)

    return

"""
#(tavella)
def lines_to_fit(input_file,species,wavenumber_born=np.array([10**8/7000,10**8/8000]),number_of_lines=10,region_excluded=np.array([10**8/12680,10**8/12710]),lines_excluded=np.array([1.39149595e+04,1.38726811e+04,1.38184074e+04]),instrument='HARPS_N'):
    "wave_born, region_excluded and lines_excluded are in wavenumber in cm-1"
    hitran_database = fits.open(input_file)[1].data
    
    wave_number_hitran_rest_temp = hitran_database['wave_number'][(wavenumber_born[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wavenumber_born[0])]
    intensity_hitran_temp        = hitran_database['Intensity'][(wavenumber_born[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wavenumber_born[0])]
    gamma_air_temp               = hitran_database['gamma_air'][(wavenumber_born[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wavenumber_born[0])]
    n_air_temp                   = hitran_database['N'][(wavenumber_born[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wavenumber_born[0])]
    delta_air_temp               = hitran_database['delta'][(wavenumber_born[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wavenumber_born[0])]
    Epp_temp                     = hitran_database['Epp'][(wavenumber_born[-1] < hitran_database['wave_number']) &  (hitran_database['wave_number'] < wavenumber_born[0])]
        
   	##############################################
   	####    Selection of the strongest lines  ####
   	##############################################

    wave_number_hitran_rest_Nstrong          = []
    delta_air_temp_Nstrong                   = []
    wave_number_hitran_rest_temp_order       = wave_number_hitran_rest_temp[np.argsort(intensity_hitran_temp)]
    delta_air_temp_order                     = delta_air_temp[np.argsort(intensity_hitran_temp)]
    wave_number_hitran_rest_excluded         = []


    for sw in range(len(wave_number_hitran_rest_temp_order)):
        idx = -1-sw
        #if (np.round(wave_number_hitran_rest_temp_order[idx],4) not in np.round(lines_excluded,4)) & ((wave_number_hitran_rest_temp_order[idx] > region_excluded[0]) | (wave_number_hitran_rest_temp_order[idx] < region_excluded[1])):
        if (np.sum((wave_number_hitran_rest_temp_order[idx] < 10**8/(10**8/lines_excluded + 2*40./300000*np.mean(10**8/lines_excluded))) | (wave_number_hitran_rest_temp_order[idx] > 10**8/(10**8/lines_excluded - 2*40./300000*np.mean(10**8/lines_excluded))))==len(lines_excluded)) & ((wave_number_hitran_rest_temp_order[idx] > region_excluded[0]) | (wave_number_hitran_rest_temp_order[idx] < region_excluded[1])):
            if (len(wave_number_hitran_rest_Nstrong) == 0):
                wave_number_hitran_rest_Nstrong = np.concatenate((wave_number_hitran_rest_Nstrong,[wave_number_hitran_rest_temp_order[idx]]))
                delta_air_temp_Nstrong = np.concatenate((delta_air_temp_Nstrong,[delta_air_temp_order[idx]]))
            elif (len(wave_number_hitran_rest_Nstrong) != 0)  :
                rejection_criteria=0
                for swl in range(len(wave_number_hitran_rest_Nstrong)):
                    if (wave_number_hitran_rest_temp_order[idx] > 10**8/(10**8/wave_number_hitran_rest_Nstrong[swl] + 2*40./300000*np.mean(10**8/wavenumber_born))) and (wave_number_hitran_rest_temp_order[idx] < 10**8/(10**8/wave_number_hitran_rest_Nstrong[swl] - 2*40./300000*np.mean(10**8/wavenumber_born)))  :  #avoid two strong telluric line blended
                        rejection_criteria+=1
                if (rejection_criteria ==0) :
                    wave_number_hitran_rest_Nstrong = np.concatenate((wave_number_hitran_rest_Nstrong,[wave_number_hitran_rest_temp_order[idx]]))
                    delta_air_temp_Nstrong = np.concatenate((delta_air_temp_Nstrong,[delta_air_temp_order[idx]]))
                else:
                    wave_number_hitran_rest_excluded = np.concatenate((wave_number_hitran_rest_excluded,[wave_number_hitran_rest_temp_order[idx]]))
            if len(wave_number_hitran_rest_Nstrong) >=number_of_lines:
                break
        else:
            wave_number_hitran_rest_excluded = np.concatenate((wave_number_hitran_rest_excluded,[wave_number_hitran_rest_temp_order[idx]]))

    wave_number_hitran_rest_Nstrong  = np.array(wave_number_hitran_rest_Nstrong)
    delta_air_temp_Nstrong  = np.array(delta_air_temp_Nstrong)
    wave_number_hitran_rest_excluded = np.array(wave_number_hitran_rest_excluded)


    wave_number_hitran_rest_Nstrong_below = 1./((1./(wave_number_hitran_rest_Nstrong*10**-8) - 40./300000*np.mean(10**8/wavenumber_born)) * 10**-8 )
    wave_number_hitran_rest_Nstrong_upper = 1./((1./(wave_number_hitran_rest_Nstrong*10**-8) + 40./300000*np.mean(10**8/wavenumber_born)) * 10**-8 )

   	##################################################################
   	####    Selection of all the lines around the strongest ones  ####
   	##################################################################

    wave_number_hitran_rest = []
    intensity_hitran        = []
    gamma_air               = []
    n_air                   = []
    delta_air               = []
    Epp                     = []

    for wn in range(len(wave_number_hitran_rest_Nstrong)): # select all the lines around the N strongest
        wave_number_hitran_rest = np.concatenate(( wave_number_hitran_rest, wave_number_hitran_rest_temp[(wave_number_hitran_rest_Nstrong_below[wn] > wave_number_hitran_rest_temp) &  (wave_number_hitran_rest_temp > wave_number_hitran_rest_Nstrong_upper[wn])] ))
        intensity_hitran        = np.concatenate(( intensity_hitran,        intensity_hitran_temp[(wave_number_hitran_rest_Nstrong_below[wn] > wave_number_hitran_rest_temp) &  (wave_number_hitran_rest_temp > wave_number_hitran_rest_Nstrong_upper[wn])] ))
        gamma_air               = np.concatenate(( gamma_air,               gamma_air_temp[(wave_number_hitran_rest_Nstrong_below[wn] > wave_number_hitran_rest_temp) &  (wave_number_hitran_rest_temp > wave_number_hitran_rest_Nstrong_upper[wn])] ))
        n_air                   = np.concatenate(( n_air,                   n_air_temp[(wave_number_hitran_rest_Nstrong_below[wn] > wave_number_hitran_rest_temp) &  (wave_number_hitran_rest_temp > wave_number_hitran_rest_Nstrong_upper[wn])] ))
        delta_air               = np.concatenate(( delta_air,               delta_air_temp[(wave_number_hitran_rest_Nstrong_below[wn] > wave_number_hitran_rest_temp) &  (wave_number_hitran_rest_temp > wave_number_hitran_rest_Nstrong_upper[wn])] ))
        Epp                     = np.concatenate(( Epp,                     Epp_temp[(wave_number_hitran_rest_Nstrong_below[wn] > wave_number_hitran_rest_temp) &  (wave_number_hitran_rest_temp > wave_number_hitran_rest_Nstrong_upper[wn])] ))



    col1 = fits.Column(name='wave_number', format='1D', array=wave_number_hitran_rest)
    col2 = fits.Column(name='Intensity',   format='1D', array=intensity_hitran)
    col3 = fits.Column(name='gamma_air',   format='1D', array=gamma_air)
    col4 = fits.Column(name='N',           format='1D', array=n_air)
    col5 = fits.Column(name='delta',       format='1D', array=delta_air)
    col6 = fits.Column(name='Epp',         format='1D', array=Epp)

    col7 = fits.Column(name='CCF_lines_position_wavenumber', format='1D', array=wave_number_hitran_rest_Nstrong)
    col8 = fits.Column(name='CCF_lines_position_wavelength', format='1D', array=10**8/wave_number_hitran_rest_Nstrong)
    col8bis = fits.Column(name='CCF_lines_delta', format='1D', array=delta_air_temp_Nstrong)

    col9 = fits.Column(name='Excluded_lines_position_wavenumber', format='1D', array=wave_number_hitran_rest_excluded)
    col10 = fits.Column(name='Excluded_lines_position_wavelength', format='1D', array=10**8/wave_number_hitran_rest_excluded)

    cols0 = fits.ColDefs([col1, col2, col3, col4, col5, col6])
    tbhdu0 = fits.BinTableHDU.from_columns(cols0)

    cols1 = fits.ColDefs([col7, col8 , col8bis])
    tbhdu1 = fits.BinTableHDU.from_columns(cols1)

    cols2 = fits.ColDefs([col9, col10])
    tbhdu2 = fits.BinTableHDU.from_columns(cols2)    
    
    prihdr = fits.Header()
    prihdr['Number of lines'] = str(number_of_lines)
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu0, tbhdu1, tbhdu2])
    thdulist.writeto('Static_model/'+instrument+'/Static_hitran_strongest_lines_'+species+'.fits', overwrite=True)

    return



def polynomial_regression_2d(xy, a, b, c, d, e, f, g):
    x = xy[0]
    y = xy[1]
    return a * x + b * y + c * x * y + d * x ** 2. + e * y ** 2. + f * x ** 2. * y ** 2. + g

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

    elif (ins_mode == 'MULTIMR') and (bin_x == 4):
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


def plot_resolution_map_UHR_HR(pixels_blue_slice1,orders_blue_slice1,resolution_blue_slice1,pixels_blue_hr,matrix_resolution_blue_hr_theo_slice1,pixels_red_slice1,
                               orders_red_slice1,resolution_red_slice1,pixels_red_hr,matrix_resolution_red_hr_theo_slice1,res_min,res_max,
                               pixels_blue_slice2,orders_blue_slice2,resolution_blue_slice2,matrix_resolution_blue_hr_theo_slice2,pixels_red_slice2,
                               orders_red_slice2,resolution_red_slice2,matrix_resolution_red_hr_theo_slice2,ins_mode,binx,mjd,Doit=False):
    if Doit==True:
        fig = pl.figure(figsize=(20, 20))
        pl.scatter(pixels_blue_slice1, orders_blue_slice1, c=resolution_blue_slice1, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_blue_hr, np.arange(1, 90.01, 1), matrix_resolution_blue_hr_theo_slice1, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
        pl.scatter(pixels_red_slice1, orders_red_slice1, c=resolution_red_slice1, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_red_hr, np.arange(91, 170.01, 1), matrix_resolution_red_hr_theo_slice1, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
        pl.scatter(pixels_blue_slice2, orders_blue_slice2, c=resolution_blue_slice2, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_blue_hr, np.arange(1, 90.01, 1), matrix_resolution_blue_hr_theo_slice2, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
        pl.scatter(pixels_red_slice2, orders_red_slice2, c=resolution_red_slice2, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_red_hr, np.arange(91, 170.01, 1), matrix_resolution_red_hr_theo_slice2, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
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
        pl.savefig('Output/Resolution_map/ESPRESSO/Map_resolution_' + ins_mode + '_' + str(binx) + '_' + str(int(mjd)) + '.png', dpi=100, format='png', bbox_inches='tight')
    return

def plot_resolution_map_slice_UHR_HR(pixels_blue_slice1,orders_blue_slice1,resolution_blue_slice1,pixels_blue_hr,matrix_resolution_blue_hr_theo_slice1,pixels_red_slice1,
                                     orders_red_slice1,resolution_red_slice1,pixels_red_hr,matrix_resolution_red_hr_theo_slice1,res_min,res_max,
                                     pixels_blue_slice2,orders_blue_slice2,resolution_blue_slice2,matrix_resolution_blue_hr_theo_slice2,pixels_red_slice2,
                                     orders_red_slice2,resolution_red_slice2,matrix_resolution_red_hr_theo_slice2,ins_mode,binx,mjd,Doit=False):
    if Doit == True:
        fig = pl.figure(figsize=(20, 20))
        pl.subplot(1, 2, 1)
        pl.scatter(pixels_blue_slice1, orders_blue_slice1, c=resolution_blue_slice1, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_blue_hr, np.arange(1, 90.01, 1), matrix_resolution_blue_hr_theo_slice1, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
        pl.scatter(pixels_red_slice1, orders_red_slice1, c=resolution_red_slice1, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_red_hr, np.arange(91, 170.01, 1), matrix_resolution_red_hr_theo_slice1, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
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
        pl.pcolormesh(pixels_blue_hr, np.arange(1, 90.01, 1), matrix_resolution_blue_hr_theo_slice2, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
        pl.scatter(pixels_red_slice2, orders_red_slice2, c=resolution_red_slice2, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_red_hr, np.arange(91, 170.01, 1), matrix_resolution_red_hr_theo_slice2, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
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
        pl.savefig('Output/Resolution_map/ESPRESSO/Map_resolution_slices_' + ins_mode + '_' + str(binx) + '_' + str(int(mjd)) + '.png', dpi=100, format='png', bbox_inches='tight')
    return

def plot_resolution_map_MR(pixels_blue_slice1,orders_blue_slice1,resolution_blue_slice1,pixels_blue_hr,matrix_resolution_blue_hr_theo_slice1,pixels_red_slice1,
                           orders_red_slice1,resolution_red_slice1,pixels_red_hr,matrix_resolution_red_hr_theo_slice1,res_min,res_max,
                           ins_mode,binx,mjd,Doit=False):
    if Doit == True:
        fig = pl.figure(figsize=(20, 20))
        pl.scatter(pixels_blue_slice1, orders_blue_slice1, c=resolution_blue_slice1, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_blue_hr, np.arange(1, 45.01, 1), matrix_resolution_blue_hr_theo_slice1, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
        pl.scatter(pixels_red_slice1, orders_red_slice1, c=resolution_red_slice1, cmap=mpl.cm.rainbow, vmin=res_min,vmax=res_max, edgecolors='white', zorder=100)
        pl.pcolormesh(pixels_red_hr, np.arange(46, 85.01, 1), matrix_resolution_red_hr_theo_slice1, cmap=mpl.cm.rainbow,shading='flat', vmin=res_min, vmax=res_max)
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
        pl.savefig('Output/Resolution_map/ESPRESSO/Map_resolution_' + ins_mode + '_' + str(binx) + '_' + str(int(mjd)) + '.png', dpi=100, format='png', bbox_inches='tight')
    return

def Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A, function_map_resolution):
    res_min=120000
    res_max=240000

    header = THAR_LINE_TABLE_THAR_FP_A[0].header

    Time = header['MJD-OBS']

    qual = THAR_LINE_TABLE_THAR_FP_A[1].data['qc']
    ords = THAR_LINE_TABLE_THAR_FP_A[1].data['order'][np.where(qual != 0)]
    pxs = THAR_LINE_TABLE_THAR_FP_A[1].data['x0'][np.where(qual != 0)]
    res = THAR_LINE_TABLE_THAR_FP_A[1].data['resolution'][np.where(qual != 0)]

    if header['HIERARCH ESO INS MODE'] == 'MULTIMR':
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
                               Doit=True)

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
                                         Doit=True)
        plot_resolution_map_UHR_HR(pixels_blue_slice1,orders_blue_slice1,resolution_blue_slice1,pixels_blue_hr,matrix_resolution_blue_hr_theo_slice1,pixels_red_slice1,
                                         orders_red_slice1,resolution_red_slice1,pixels_red_hr,matrix_resolution_red_hr_theo_slice1,res_min,res_max,
                                         pixels_blue_slice2,orders_blue_slice2,resolution_blue_slice2,matrix_resolution_blue_hr_theo_slice2,pixels_red_slice2,
                                         orders_red_slice2,resolution_red_slice2,matrix_resolution_red_hr_theo_slice2,header['HIERARCH ESO INS MODE'],header['HIERARCH ESO DET BINX'],header['MJD-OBS'],
                                         Doit=True)

        resolution_map = np.zeros((170, len(shape_s2d[1])))

        resolution_map[:90, int(pixels_blue_hr[0]):int(pixels_blue_hr[-1]) + 1] = np.where(
            np.isnan(matrix_resolution_blue_hr_theo_slice1), matrix_resolution_blue_hr_theo_slice2,
            matrix_resolution_blue_hr_theo_slice1)
        resolution_map[90:, :] = np.where(np.isnan(matrix_resolution_red_hr_theo_slice1),
                                          matrix_resolution_red_hr_theo_slice2, matrix_resolution_red_hr_theo_slice1)
        resolution_map = np.where(resolution_map == 0., np.nan, resolution_map)
    prihdu = fits.PrimaryHDU(header=THAR_LINE_TABLE_THAR_FP_A[0].header)
    prihdu.writeto('Static_resolution/ESPRESSO/r.' + header['ARCFILE'][:-5] + '_RESOLUTION_MAP.fits', overwrite=True)
    fits.append('Static_resolution/ESPRESSO/r.' + header['ARCFILE'][:-5] + '_RESOLUTION_MAP.fits', resolution_map)
    return




########################
### files to compute ###
########################



# #UHR
# file_to_map_resolution = fits.open('Input_files_statics/Resolution_table/ESPRESSO/UHR/r.ESPRE.2018-09-09T12:18:49.369_THAR_LINE_TABLE_THAR_FP_A.fits')
# Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A=file_to_map_resolution,function_map_resolution=polynomial_regression_2d)
# file_to_map_resolution = fits.open('Input_files_statics/Resolution_table/ESPRESSO/UHR/r.ESPRE.2019-04-06T11:52:27.477_THAR_LINE_TABLE_THAR_FP_A.fits')
# Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A=file_to_map_resolution,function_map_resolution=polynomial_regression_2d)
# file_to_map_resolution = fits.open('Input_files_statics/Resolution_table/ESPRESSO/UHR/r.ESPRE.2019-11-06T11:06:36.913_THAR_LINE_TABLE_THAR_FP_A.fits')
# Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A=file_to_map_resolution,function_map_resolution=polynomial_regression_2d)
#file_to_map_resolution = fits.open('Input_files_statics/Resolution_table/ESPRESSO/UHR/r.ESPRE.2019-11-26T15:27:21.138_THAR_LINE_TABLE_FP_THAR_B.fits') # fiber B
#Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A=file_to_map_resolution,function_map_resolution=polynomial_regression_2d)
# #HR 1x1
# file_to_map_resolution = fits.open('Input_files_statics/Resolution_table/ESPRESSO/HR/1x1/r.ESPRE.2018-07-04T21:28:53.759_THAR_LINE_TABLE_THAR_FP_A.fits')
# Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A=file_to_map_resolution,function_map_resolution=polynomial_regression_2d)
# file_to_map_resolution = fits.open('Input_files_statics/Resolution_table/ESPRESSO/HR/1x1/r.ESPRE.2019-01-05T19:53:55.501_THAR_LINE_TABLE_THAR_FP_A.fits')
# Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A=file_to_map_resolution,function_map_resolution=polynomial_regression_2d)
# file_to_map_resolution = fits.open('Input_files_statics/Resolution_table/ESPRESSO/HR/1x1/r.ESPRE.2019-11-19T10:10:12.384_THAR_LINE_TABLE_THAR_FP_A.fits')
# Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A=file_to_map_resolution,function_map_resolution=polynomial_regression_2d)

# #HR 2x1
# file_to_map_resolution = fits.open('Input_files_statics/Resolution_table/ESPRESSO/HR/2x1/r.ESPRE.2018-09-05T14:01:58.063_THAR_LINE_TABLE_THAR_FP_A.fits')
# Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A=file_to_map_resolution,function_map_resolution=polynomial_regression_2d)
# file_to_map_resolution = fits.open('Input_files_statics/Resolution_table/ESPRESSO/HR/2x1/r.ESPRE.2019-03-30T21:28:32.060_THAR_LINE_TABLE_THAR_FP_A.fits')
# Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A=file_to_map_resolution,function_map_resolution=polynomial_regression_2d)
# file_to_map_resolution = fits.open('Input_files_statics/Resolution_table/ESPRESSO/HR/2x1/r.ESPRE.2019-09-26T11:06:01.271_THAR_LINE_TABLE_THAR_FP_A.fits')
# Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A=file_to_map_resolution,function_map_resolution=polynomial_regression_2d)

# #MR4x2
# file_to_map_resolution = fits.open('Input_files_statics/Resolution_table/ESPRESSO/4x2/r.ESPRE.2018-07-08T14:51:53.873_THAR_LINE_TABLE_THAR_FP_A.fits')
# Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A=file_to_map_resolution,function_map_resolution=polynomial_regression_2d)
# file_to_map_resolution = fits.open('Input_files_statics/Resolution_table/ESPRESSO/4x2/r.ESPRE.2018-12-01T11:25:27.377_THAR_LINE_TABLE_THAR_FP_A.fits')
# Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A=file_to_map_resolution,function_map_resolution=polynomial_regression_2d)
# file_to_map_resolution = fits.open('Input_files_statics/Resolution_table/ESPRESSO/4x2/r.ESPRE.2019-11-05T12:23:45.139_THAR_LINE_TABLE_THAR_FP_A.fits')
# Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A=file_to_map_resolution,function_map_resolution=polynomial_regression_2d)

# #MR8x4
# file_to_map_resolution = fits.open('Input_files_statics/Resolution_table/ESPRESSO/8x4/r.ESPRE.2018-07-06T11:48:22.862_THAR_LINE_TABLE_THAR_FP_A.fits')
# Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A=file_to_map_resolution,function_map_resolution=polynomial_regression_2d)
# file_to_map_resolution = fits.open('Input_files_statics/Resolution_table/ESPRESSO/8x4/r.ESPRE.2018-10-24T14:16:52.394_THAR_LINE_TABLE_THAR_FP_A.fits')
# Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A=file_to_map_resolution,function_map_resolution=polynomial_regression_2d)
# file_to_map_resolution = fits.open('Input_files_statics/Resolution_table/ESPRESSO/8x4/r.ESPRE.2019-11-05T12:59:57.138_THAR_LINE_TABLE_THAR_FP_A.fits')
# Instrumental_resolution(THAR_LINE_TABLE_THAR_FP_A=file_to_map_resolution,function_map_resolution=polynomial_regression_2d)


"""



static_hitran_file('Input_files_statics/Molecules/CH4/NIRPS/12C_H4__9000_18000.txt',
                    'Input_files_statics/Molecules/CH4/NIRPS/QT.txt','CH4',
                    S_cut=10 ** -24,instrument='NIRPS')

# static_hitran_file('Input_files_statics/Molecules/CO/NIRPS/12C_16O__9000_18000.txt',
#                     'Input_files_statics/Molecules/CO/NIRPS/QT.txt','CO',
#                     S_cut=10 ** -26,instrument='NIRPS')

static_hitran_file('Input_files_statics/Molecules/CO2/NIRPS/12C_16O2__9000_18000.txt',
                    'Input_files_statics/Molecules/CO2/NIRPS/QT.txt','CO2',
                    S_cut=10 ** -26,instrument='NIRPS')

static_hitran_file('Input_files_statics/Molecules/H2O/NIRPS/H2_16O__9000_18000.txt',
                    'Input_files_statics/Molecules/H2O/NIRPS/QT.txt','H2O',
                    S_cut=10 ** -26,instrument='NIRPS')

static_hitran_file('Input_files_statics/Molecules/O2/NIRPS/16O2__9000_18000.txt',
                    'Input_files_statics/Molecules/O2/NIRPS/QT.txt','O2',
                    S_cut=10 ** -28,instrument='NIRPS')


lines_to_fit('Static_model/NIRPS/Static_hitran_qt_H2O.fits',
             'H2O',wavenumber_born=np.array([10**8/12850,10**8/15170]),number_of_lines=50,
             region_excluded=np.array([10**8/13250,10**8/15000]),instrument='NIRPS')

lines_to_fit('Static_model/NIRPS/Static_hitran_qt_O2.fits',
             'O2',wavenumber_born=np.array([10**8/12450,10**8/12950]),number_of_lines=30,
             region_excluded=np.array([10**8/12955,10**8/12960]),instrument='NIRPS')

lines_to_fit('Static_model/NIRPS/Static_hitran_qt_CH4.fits',
             'CH4',wavenumber_born=np.array([10**8/16200,10**8/16900]),number_of_lines=16,instrument='NIRPS')

lines_to_fit('Static_model/NIRPS/Static_hitran_qt_CO2.fits',
             'CO2',wavenumber_born=np.array([10**8/15150,10**8/16200]),number_of_lines=50,
             region_excluded=np.array([10**8/12680,10**8/12710]),instrument='NIRPS')

"""
"""

static_hitran_file('Input_files_statics/Molecules/H2O/ESPRESSO/H2_16O__3700_8000.txt',
                  'Input_files_statics/Molecules/H2O/ESPRESSO/QT.txt','H2O',
                  S_cut=10 ** -26,instrument='ESPRESSO')

static_hitran_file('Input_files_statics/Molecules/O2/ESPRESSO/16O2__3700_8000.txt',
                  'Input_files_statics/Molecules/O2/ESPRESSO/QT.txt','O2',
                  S_cut=10 ** -28,instrument='ESPRESSO')


lines_to_fit('Static_model/ESPRESSO/Static_hitran_qt_H2O.fits',
             'H2O',wavenumber_born=np.array([10**8/7150,10**8/7350]),number_of_lines=20,
             lines_excluded=np.array([10**8/7186.51031647,10**8/7208.41193416,10**8/7236.72396574]),instrument='ESPRESSO')

lines_to_fit('Static_model/ESPRESSO/Static_hitran_qt_O2.fits',
             'O2',wavenumber_born=np.array([10**8/6000,10**8/6400]),number_of_lines=12,
             region_excluded=np.array([10**8/6855,10**8/6918]),instrument='ESPRESSO')

"""


#(TAVELLA)

#HARPS_N

#H2O
static_hitran_file('Input_files_statics/Molecules/H2O/HARPS_N/H2_16O__3750_6950.txt',
                  'Input_files_statics/Molecules/H2O/HARPS_N/QT.txt','H2O',
                  S_cut=10 ** -26,instrument='HARPS_N')

lines_to_fit('Static_model/HARPS_N/Static_hitran_qt_H2O.fits',
             'H2O',wavenumber_born=np.array([10**8/6450,10**8/6600]),number_of_lines=20, instrument='HARPS_N')

#O2
static_hitran_file('Input_files_statics/Molecules/O2/HARPS_N/16O2__3750_6950.txt',
                  'Input_files_statics/Molecules/O2/HARPS_N/QT.txt','O2',
                  S_cut=10 ** -28,instrument='HARPS_N')

lines_to_fit('Static_model/HARPS_N/Static_hitran_qt_O2.fits',
             'O2',wavenumber_born=np.array([10**8/6250,10**8/6350]),number_of_lines=12,
             region_excluded=np.array([10**8/6855,10**8/6918]),instrument='HARPS_N')

#HARPS

#H2O
static_hitran_file('Input_files_statics/Molecules/H2O/HARPS/H2_16O__3750_6950.txt',
                  'Input_files_statics/Molecules/H2O/HARPS/QT.txt','H2O',
                  S_cut=10 ** -26,instrument='HARPS')

lines_to_fit('Static_model/HARPS/Static_hitran_qt_H2O.fits',
             'H2O',wavenumber_born=np.array([10**8/6450,10**8/6600]),number_of_lines=20, instrument='HARPS')

#O2
static_hitran_file('Input_files_statics/Molecules/O2/HARPS/16O2__3750_6950.txt',
                  'Input_files_statics/Molecules/O2/HARPS/QT.txt','O2',
                  S_cut=10 ** -28,instrument='HARPS')

lines_to_fit('Static_model/HARPS/Static_hitran_qt_O2.fits',
             'O2',wavenumber_born=np.array([10**8/6250,10**8/6350]),number_of_lines=12,
             region_excluded=np.array([10**8/6855,10**8/6918]),instrument='HARPS')

#CARMENES_VIS

#H2O
static_hitran_file('Input_files_statics/Molecules/H2O/CARMENES_VIS/H2_16O__5000_9800.txt',
                  'Input_files_statics/Molecules/H2O/CARMENES_VIS/QT.txt','H2O',
                  S_cut=10 ** -26,instrument='CARMENES_VIS')

lines_to_fit('Static_model/CARMENES_VIS/Static_hitran_qt_H2O.fits',
             'H2O',wavenumber_born=np.array([10**8/7150,10**8/7350]),number_of_lines=20,
             lines_excluded=np.array([10**8/7186.51031647,10**8/7208.41193416,10**8/7236.72396574]), instrument='CARMENES_VIS')

#O2
static_hitran_file('Input_files_statics/Molecules/O2/CARMENES_VIS/16O2__5000_9800.txt',
                  'Input_files_statics/Molecules/O2/CARMENES_VIS/QT.txt','O2',
                  S_cut=10 ** -28,instrument='CARMENES_VIS') 

lines_to_fit('Static_model/CARMENES_VIS/Static_hitran_qt_O2.fits',
             'O2',wavenumber_born=np.array([10**8/6250,10**8/6350]),number_of_lines=12,
             region_excluded=np.array([10**8/6855,10**8/6918]),instrument='CARMENES_VIS')

#different options for restricted fit range + excluded regions
#lines_to_fit('Static_model/CARMENS_VIS/Static_hitran_qt_O2.fits',
            # 'O2',wavenumber_born=np.array([10**8/6250,10**8/6350]),number_of_lines=12, # np.array([10**8/6920,10**8/7000]) or np.array([10**8/7690,10**8/7750])
            # region_excluded=np.array([10**8/6855,10**8/6918], [10**8/7590,10**8/7690]),instrument='CARMENS_VIS')



"""

lines_to_fit('Static_model/HARPS_N/Static_hitran_qt_H2O.fits',
             'H2O',wavenumber_born=np.array([10**8/7150,10**8/7350]),number_of_lines=20,
             lines_excluded=np.array([10**8/7186.51031647,10**8/7208.41193416,10**8/7236.72396574]),instrument='HARPS_N')

lines_to_fit('Static_model/HARPS_N/Static_hitran_qt_O2.fits',
             'O2',wavenumber_born=np.array([10**8/6250,10**8/6350]),number_of_lines=12,
             region_excluded=np.array([10**8/6855,10**8/6918]),instrument='HARPS_N')






"""











