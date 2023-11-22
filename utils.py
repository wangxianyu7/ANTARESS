# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 11:51:09 2014

@author: vincent
"""
import numpy as np
from math import pi
from copy import deepcopy
from pathos.multiprocessing import Pool
from mpmath import fp    
import astropy.io.fits as fits
from scipy.interpolate import InterpolatedUnivariateSpline
# from constant_data import c_light


'''
#Redefinition of usual functions
'''    
np_append=np.append
np_arange=np.arange
np_arctan2=np.arctan2
np_array=np.array
np_column_stack=np.column_stack
np_cos=np.cos
np_empty=np.empty
np_exp=np.exp
np_interp=np.interp
np_invert=np.invert
np_isnan=np.isnan
np_log=np.log
np_mean=np.mean
np_nan=np.nan
np_ones=np.ones
np_outer=np.outer
np_power=np.power
np_repeat=np.repeat
np_reshape=np.reshape
np_rint=np.rint
np_savetxt=np.savetxt
np_sqrt=np.sqrt
np_sum=np.sum
np_tile=np.tile
np_unique=np.unique
np_vstack=np.vstack
np_zeros=np.zeros
np_poly = np.polynomial.Polynomial

def dataload_npz(path):
    return np.load(path+'.npz',allow_pickle=True)['data'].item()
def datasave_npz(path,data):
    return np.savez_compressed(path,data=data,allow_pickle=True)

def np_where1D(cond):
    return np.where(cond)[0]

def is_odd(num):
   return bool(num % 2)


'''
Default function returning 1
'''
def default_func(*x):return np.array([1.])




    
'''
Factorial of value
'''
def factrial(val):
    return np.math.factorial(val)    

'''
Closest modulo function
    given x, p, y, returns the value of x +- n*p closest to y 
    equivalent to finding the largest n so that [ (x - y)/p ] +- n close to 0
'''
def closest_mod(x,p,y):
    return x - np.round((x-y)/p)*p
    

'''
Routine to calculate the convolution of a tabulated function with a continuously-defined kernel
    - f_conv(x) = int( f(x-t)*k_norm(t)*dt), with f_conv and f in the same units, and thus k_norm in units of [dt-1]                
 with k_norm(t) = k(t)/int(k(u)*du), so that the integral of k_norm is 1
    - expressed in the discrete domain, the convolution writes as (see http://www.exelisvis.com/docs/CONVOL.html):
 F_conv(x) = sum( i=0:m-1 ; F( x+i-hm ) * k(i) ) if hm <= x <= n-hm-1
           = F(0) * sum(i=0,hm-x-1 ; k(i) ) + sum( i=hm-x:m-1 ; F( x+i-hm ) * k(i) ) if x < hm 
           = sum( i=0:n-1+hm-x ; F( x+i-hm ) * k(i) ) + F(n-1) * sum(i=n+hm-x,m-1 ; k(i) )  if x > n-hm-1    
 with 'F' the function to be convolved (dimension 'n'), and 'm' the kernel (dimension 'k'). This can also be written
 with (j=x+i-hm ->  i=j-x+hm) and (m - hm = hm) as:   
 F_conv(x) = sum( j=x-hm:x+hm ; F( j ) * k(j-x+hm) ) if hm <= x <= n-hm-1      
           = F(0) * sum(j=x-hm,-1 ; k(j-x+hm) ) + sum( j=0:x+hm ; F( j ) * k(j-x+hm) ) if x < hm 
           = sum( j=x-hm:n-1 ; F( j ) * k(j-x+hm) ) + F(n-1) * sum(j=n,x+hm-1 ; k(j-x+hm) )  if x > n-hm-1     
 here we use extended edges, ie we repeat the values of F at the edge to compute their convolved values      
    - here we take into account the (possibly variable) resolution of F, and use a function for the kernel to be able to calculate
 its value at any position. We define:
 + k[wcen](w) as the value of the kernel centered on wcen, at wavelength w. 
 + 'hw_kern' (ie hm) the interval between the kernel center and the wavelength point where the kernel falls below 1% of its max, ie where its 
 contribution to the convolution becomes negligible (~ 0.1% difference with the full kernel range) 
 + 'dw(j)' the width of the pixel at index j in the table of F
    - we separate again three different cases:    
 + regular values: w(0)+hw_kern <= w(x) <= w(n-1)-hw_kern         
       F_conv(x) = sum( j=js,je ; F( j ) * k[w(j)]( w(x) ) * dw(j) ) 
   with js lowest index that gives   w(js) >= w(x) - hw_kern
        je highest index that gives  w(je) <= w(x) + hw_kern
   we cannot define js,je in terms of indexes because the resolution of F can be variable
 + lower edge: w(x) < w(0)+hw_kern 
       F_conv(x) = F(0) * dw(0) * sum(j= -1 - int(hw_kern/dw(0)),-1 ; k[w(0)+dw(0)*j]( w(x) ) ) +
                   sum( j=0:je ; F( j ) * k[w(j)]( w(x) ) * dw(j) )
   with je highest index that gives w(je) <= w(x) + hw_kern 
 in that case, because hw_kern/dw(0) -1 < int(hw_kern/dw(0)) <= hw_kern/dw(0) the first pixel on which the kernel is centered on at
 w_jmin = w(0) - dw(0) - dw(0)*int( hw_kern/dw(0) ) verifies hw_kern<w(0)-w_jmin<=hw_kern+dw
 thus this is the farthest phantom pixel, when extending the table with pixels at the resolution of the first pixel, for which the kernel
 contributes significantly to the convolution (since the next extended pixel would be at more than hw_kern from the first pixel at w(0)) 
 + upper edge: w(x) > w(n-1)-hw_kern
       F_conv(x) = sum( j=js:n-1 ; F( j ) * k[w(j)]( w(x) ) * dw(j) ) 
                   + F(n-1) * dw(n-1) * sum(j=1,1+int(hw_kern/dw(n-1)) ; k[w(n-1)+dw(n-1)*j]( w(x) ) )  
   with js lowest index that gives   w(js) >= w(x) - hw_kern
 in that case, because hw_kern/dw(n-1) -1 < int(hw_kern/dw(n-1)) <= hw_kern/dw(n-1) the last pixel on which the kernel is centered on at
 w_jmax = w(n-1) + dw(n-1) + dw(n-1)*int( hw_kern/dw(n-1) ) verifies hw_kern < w_jmax - w(n-1)  <= hw_kern +dw(n-1)   
'''
def convol_contin(wave_tab,dwave_tab,F_raw,kern,hw_kern):
 
    #First/last wavelength pixels
    n_pix=len(F_raw) 
    wav0=wave_tab[0]
    dwav0=dwave_tab[0]
    dwavf=dwave_tab[n_pix-1]    
    wavf=wave_tab[n_pix-1]
    if 2.*hw_kern>(wavf-wav0):stop('Kernel too large compared to spectrum')
   
    #The convolved spectrum is defined on the same wavelength table as the original spectrum
    F_conv=np.zeros(n_pix)
 
    #We use directly the product spectrum x resolution 'F_dw'
    F_dw=F_raw*dwave_tab

    #--------------------------------------------------------------------------------------------
    #Regular values: w(0)+hw_kern <= w(x) <= w(n-1)-hw_kern         
    #       F_conv(x) = sum( j=js,je ; F( j ) * k[w(j)]( w(x) ) * dw(j) ) 
    #   with js lowest index that gives   w(js) >= w(x) - hw_kern
    #        je highest index that gives  w(je) <= w(x) + hw_kern
    id_reg=np.where( (wave_tab >= wav0 + hw_kern) & (wave_tab <= wavf - hw_kern))[0]
    for ix in id_reg:
        wavx=wave_tab[ix]
        j_tab_reg=np.arange(( np.where(wave_tab >= wavx - hw_kern)[0] )[0],( np.where(wave_tab <= wavx + hw_kern)[0] )[-1]+1)
        F_conv[ix]+=np.sum(F_dw[j_tab_reg]*kern(wavx,wave_tab[j_tab_reg]))

    #--------------------------------------------------------------------------------------------
    #Lower edge: w(x) < w(0)+hw_kern 
    #       F_conv(x) = F(0) * dw(0) * sum(j= -1 - int(hw_kern/dw(0)),-1 ; k[w(0)+dw(0)*j]( w(x) ) ) +
    #                   sum( j=0:je ; F( j ) * k[w(j)]( w(x) ) * dw(j) )
    #   with je highest index that gives w(je) <= w(x) + hw_kern 
    id_low=np.where( (wave_tab < wav0 + hw_kern))[0]
    for ix in id_low:
        wavx=wave_tab[ix]        
        j_tab_low=np.arange(-1-int(hw_kern/dwav0),0)
        j_tab_reg=np.arange(0,( np.where(wave_tab <= wavx + hw_kern)[0] )[-1]+1)          
        F_conv[ix]+=F_dw[0]*np.sum(kern(wavx,wav0+dwav0*j_tab_low))+\
                    np.sum(F_dw[j_tab_reg]*kern(wavx,wave_tab[j_tab_reg]))

    #--------------------------------------------------------------------------------------------
    #Upper edge: w(x) > w(n-1)-hw_kern
    #       F_conv(x) = sum( j=js:n-1 ; F( j ) * k[w(j)]( w(x) ) * dw(j) ) 
    #                   + F(n-1) * dw(n-1) * sum(j=1,1+int(hw_kern/dw(n-1)) ; k[w(n-1)+dw(n-1)*j]( w(x) ) )
    #   with js lowest index that gives w(js) >= w(x) - hw_kern  
    id_high=np.where( (wave_tab > wavf - hw_kern))[0]
    for ix in id_high:
        wavx=wave_tab[ix]        
        j_tab_reg=np.arange(( np.where(wave_tab >= wavx - hw_kern)[0] )[0],n_pix)
        j_tab_high=np.arange(1,1+int(hw_kern/dwavf)+1)          
        F_conv[ix]+=np.sum(F_dw[j_tab_reg]*kern(wavx,wave_tab[j_tab_reg]))+\
                    F_dw[n_pix-1]*np.sum(kern(wavx,wavf+dwavf*j_tab_high))    
    
    return F_conv

'''
Return all points between start and stop separated by step
    - start + i*step <= stop
'''
def step_range(start,stop,step):
    nsteps=int(round((stop-start)/step))+1		
    return start+np.arange(nsteps)*step

'''
Return index of closest value in an array
'''
def closest(array,value):	
    return (np.abs(array-value)).argmin()

def closest_Ndim(array,value):	
    return np.unravel_index(np.argmin(np.abs(array-value), axis=None), array.shape)
 
'''
Return index of closest value (can be an array) in an array  
    - array must be sorted
''' 
def closest_arr(array, in_value):

    #Returns index 0 for single value array	
    if (len(array)==1):
	    idx=np.repeat(0,len(in_value))
    else:					
	    #Return indexes in 'array' that would maintain order if 'in_value' values were inserted into 'array' after the 'array' value corresponding to index		
	    #This means that idx-1 corresponds to the value in array immediately lower than the 'in_value' value, except if the value is lower than the min of array (then idx=0)
	    idx = array.searchsorted(in_value)	
					
	    #Set indexes equal to 0 (ie values lower than the min in array) to 1
	    #Set indexes equal to len(array) (ie values higher than the max in array) to len(array)-1							
	    idx = np.clip(idx, 1, len(array)-1) 
	
	    #Return 'array' values immediately lower/higher than each 'in_value' values, if they are within 'array' min/max values
	    #    if value is lower than the min, idx has been set to 1 and left returns the min (and first) value of array (right the second value in array)
	    #    if value is higher than the max, idx has been set to len(array)-1 and right returns the max (and last) value of array (left the next to last value in array)
	    left = array[idx-1]
	    right = array[idx]
					
	    #Return True (=1) if value is closer to its lower limit in array (left), otherwise return False (=0)
	    #    in the 1st case, idx-1 corresponds to the value in array immediately lower than the 'in_value' value
	    #    in the 2nd case, idx   corresponds to the value in array immediately higher than the 'in_value' value	
	    #
	    # if value is lower than the min, (in_value - left) is <0 and always lower than (right - in_value) positive
	    #    thus idx=1 is reduced by 1 and returns the min of the array  				
	    # if value is higher than the max, (in_value - left) is >0 and always higher than (right - in_value) negative
	    #    thus idx=len(array)-1 is left untouched and returns the max of the array  
	    idx -= in_value - left < right - in_value

    return idx











'''
Importing tabulated values
    - importing from 'path' if mode=0
    - first column must be altitude in au, second column an atmospheric properties in the relevant unit for EVE
'''
def import_tabulated(dico_prop):
    if (dico_prop['prop']==0):
        with open(dico_prop['path'], 'r') as fl:
            all_lines = fl.readlines()
        rval_tabulated=np.empty([2,0])
        for lines in all_lines:
            rval_tabulated=np.append(rval_tabulated,[[float(lines.split()[0])],[float(lines.split()[1])]],1) 
    else:
        rval_tabulated=None     
    return rval_tabulated  
 
#TODO: generaliser ou faire plusieurs fonctions selon coordonnees associees au profil (r, th, phi)
   
   
def stop(message=None):
    r'''**Stop routine**
    
    Stop process with optional message
    
    Args:
        message (str): stop message

    Returns:
        None    
    ''' 
    str_message=' : '+message if message is not None else ''
    print('Stop'+str_message)
    raise SystemExit
    return None
		
'''
Find string between 2 strings
    - works only if strings are unique
'''    		
def find_between(s, start, end):
    return (s.split(start))[1].split(end)[0]
				
"""
To convert an array to integers
"""
def npint(array):
    return array.astype(int) 
				
####################################	##################	##################	##################	
#Grid functions
				
"""
Function to adjust a grid resolution
    - inner/outer grid boundaries are fixed 
round() allows to have the resolution closest to its initial value  
"""   
def adjust_res(d_cells,min_cells,max_cells):
        
    #Closest number of cells between the boundaries at the initial resolution
    n_cells=int(round((max_cells-min_cells)/d_cells))
    if n_cells==0:n_cells+=1   
				
    #Adjusted resolution
    d_cells=(max_cells-min_cells)/n_cells     
        
    return n_cells,d_cells  

				
"""
Function to adjust the grid boundaries
    - either both boundaries are adjusted, or one of them and the other is fixed: in both cases only one fixed position is
needed to build the tables ('start_bound'), either as an intermediate position, or as a lower/higher boundary
    - low_mode=1 to adjust lower boundary (otherwise start_bound is used)
      high_mode=1 to adjust higher boundary (otherwise start_bound is used)
"""        
def adjust_bound(val,d_cells,start_bound,low_mode,high_mode):

            #----------------------------------------------------------------
            #Number of cells between the starting value and the farthest particle, which must be contained in the farthest cell
            #    - n_cells=int(abs(bound-start_bound)/d_cells) +1 and bound = start_bound +- d_cells*n_cells
            # int() and +1 allows to add one cell which contains the farthest particles                
            #    - abs are necessary if minimum (maximum) value is above (below) starting point   
            #    - in that case, one cell must be taken out, and the boundary is on the other side of starting point, and the total number of cells
            #in the table is decreased      

            #---------------------

            #Lower boundary is adjusted using initial resolution and a fixed higher (high_mode=0) or intermediate (high_mode=1) boundary            
            if (low_mode==1):    
                nlow_temp =int(abs(start_bound-min(val))/d_cells) +1
                lowsign=np.sign(start_bound-min(val))
                if lowsign<0.:nlow_temp=nlow_temp-1
                min_cells=start_bound-lowsign*d_cells*nlow_temp
                nlow_temp=lowsign*nlow_temp
                
            #Lower boundary is fixed to its input value, and higher boundary is adjusted  
            else:    
                min_cells=start_bound
                nlow_temp=0

            #---------------------

            #Higher boundary is adjusted using initial resolution and a fixed lower (low_mode=0) or intermediate (low_mode=1) boundary 
            if (high_mode==1):               
                nhigh_temp=int(abs(max(val)-start_bound)/d_cells) +1    
                highsign=np.sign(max(val)-start_bound)
                if highsign<0.:nhigh_temp=nhigh_temp-1
                max_cells=start_bound+highsign*d_cells*nhigh_temp 
                nhigh_temp=highsign*nhigh_temp
                
            #Higher boundary is fixed to its input value, and lower boundary is adjusted                
            else:    
                max_cells=start_bound
                nhigh_temp=0

            #----------------------------------------------------------------
            
            #Number of cells 
            n_cells=int(nlow_temp+nhigh_temp)     

            return n_cells,min_cells,max_cells 
 				
"""
Convert time in float to string elements in h,mn,s
"""        
def time_str(t_float,time_type=None):	

    #Default type: hour
    if time_type=='y':time_h=t_float*24.*365.25
    if time_type=='d':time_h=t_float*24.
    if time_type is None:time_type='h'
    if time_type=='mn':time_h=t_float/60.
    if time_type=='s':time_h=t_float/3600.

    #Before/after transit
    str_sign='-' if (time_h < 0.) else'+'

    #Hour component				
    t_h=int(abs(time_h))
    str_h=	"{0}".format(t_h)+' h '

    #Minute component				
    t_mn=int((abs(time_h)- t_h)*60.)
    str_mn=	"{0}".format(t_mn)+' mn '				

    #Second component				
    t_s=int(((abs(time_h)- t_h)*60.- t_mn)*60.)
    str_s=	"{0}".format(t_s)+' s '					
	
    return str_sign,str_h,str_mn,str_s								
								
'''				
Routine to allow the return of a given value for missing keys in a dictionary			
'''
def dicval(dic,key,dval):
	return [dic.get(key) if dic.get(key) is not None else dval][0]			

'''	
Generic function to append the velocity/resolution tables for the Lorentzian/Voigt profiles
'''
def append_HRtable(rvtab_in,dvtab_in,dv_loc,va_loc,vb_loc):
	nv_loc,dv_loc=adjust_res(dv_loc,va_loc,vb_loc)			
	rv_loc=va_loc+dv_loc*(0.5+np.arange(nv_loc))
	rvtab_out=np.concatenate((rvtab_in,rv_loc))
	dvtab_out=np.concatenate((dvtab_in,np.repeat(dv_loc,nv_loc)))				
	return rvtab_out,dvtab_out						

''' 
Fonction for log conversion
    - the mode needs not be given as input
'''
def conv_log(var_tab,log_mod):
    if log_mod==True:
        return np.log10(var_tab)
    else:
        return var_tab
    
'''
Hypergeometric mpmath function for Voigt profile integration, with broadcasting
    - original function cannot be used with arrays    
'''
def np_hyp2f2(z):
    np_hyp2f2_temp = np.frompyfunc(fp.hyp2f2, 5, 1)
    return np_hyp2f2_temp(1.,1.,2.,3./2.,-z*z)



'''
Black body
    - wavelength table in A
    - temperature in K
    - returns the black body flux in erg/cm2/s/A at the radius of the star
    - to get the black body flux at distance d from the star, do
 planck(wav_tab,Teff)*(Rstar/d)^2
'''
def planck(wav_tab,Teff):

    #Constants (cgs units)
    h_planck=6.62606957e-27  #(g cm2 s-1 = erg s) Planck constant   
    k_boltz=1.3806488*1e-16  #(g cm2 s-2 K-1 = erg K-1) Boltzmann constant     
    c_light_cm=29979245800.  #(cm s-1) speed of light        
    
    #Wavelengths in cm
    wav_tab_cm=wav_tab*1e-8    
    
    #Coefficients
    a = 2.0*h_planck*(c_light_cm**2)                  #in g cm4 s-3 = erg cm2 s-1
    b = h_planck*c_light_cm/(k_boltz*wav_tab_cm*Teff) #in nothing
    
    #Black body intensity : bbint_w
    #    - bbint_nu = (2*h*nu^3/c^2)*(1/(exp(h*nu/(k*T))-1)   
    # in erg s-1 cm-2 Hz-1 steradian-1 
    #    - bbint_w = bbint_nu*(dnu/dlambda) 
    #              = bbint_nu*(c*dlambda/lambda^2)/dlambda 
    #              = bbint_nu*(c/lambda^2) 
    #              = (2*h*c^2/lambda^5)*(1/(exp(h*c/(lambda*k*T))-1) 
    # in erg s-1 cm-3 steradian-1
    bbint_w = a/((np.power(wav_tab_cm,5.))*(np.exp(b)-1.0))

    #Black body flux  
    #    - integral of intensity over all outgoing angles from the star surface, along the LOS visible by the observer
    #    - flux_w = int(phi in 0:2pi, theta in 0:pi/2 , bbint_w * d_om)
    #      flux_w = pi*bbint_w 
    # in erg s-1 cm-2 cm-1, at the star radius
    #      thus flux_w*1e-8 is in erg s-1 cm-2 A-1
    bbflux=pi*bbint_w*1e-8
    
    return bbflux



def dichotomy(low, high, f, x):

	'''
	Returns the value v between low and high such that f(v)=x using a dichotomic search.
	If v is not in [low, high], returns -1.
	
	Assumes that f is an increasing function (i.e. f(a)>f(b) for a>b).
	If f is a decreasing function, use -f.
	'''

	if f(low) > x or f(high) < x: return -1

	mid = .5*(high + low)

	if f(mid) == x:
		return mid
	elif f(mid) > x:
		return dichotomy(low, mid, f, x)
	else:
		return dichotomy(mid, high, f, x)


def MAIN_multithread(func_input,nthreads,n_elem,y_inputs,common_args,output = False):  
    r"""**Wrap-up multithreading routine.**

    Args:
        func_input (function): multi-threaded function
        nthreads (int): number of threads
        n_elem (int): number of elements to thread
        y_inputs (list): threadable function inputs 
        common_args (tuple): common function inputs
        output (bool): set to True to return function outputs
    
    Returns:
        y_output (None or specific to func_input): function outputs 
    
    """
    pool_proc = Pool(processes=nthreads)   #cannot be passed through lmfit
    ind_chunk_list=init_parallel_func(nthreads,n_elem)
    chunked_args=[tuple(y_inputs[i][ind_chunk[0]:ind_chunk[1]] for i in range(len(y_inputs)))+common_args for ind_chunk in ind_chunk_list]	
    all_results=tuple(tab for tab in pool_proc.starmap(func_input,chunked_args))
    if output:y_output=np.concatenate(tuple(all_results[i] for i in range(nthreads)))
    else:y_output = None     
    pool_proc.close()
    pool_proc.join() 				
    return y_output



'''
Routines to execute on different core a function with several arguments
    - pool_proc: open workers
    - func_input: parallelized function
    - nthreads: number of used cores
    - n_elem: total number of elements to be assigned to the different workers
    - y_inputs: list of tables, with one dimension of length n_elem, to be divided between the different workers 
    - common_args: all parameters common to all workers (not to be divided)
    - we create different routines to avoid time loss with 'if' statements, and because the function generally requires
specific treatments for the inputs/outputs				
'''
def init_parallel_func(nthreads,n_elem):

    if (nthreads<=n_elem):

        #Size of chunks to be processed by each core        
        npercore=int(n_elem/nthreads)
        
        #Indexes of each chunk       
        ind_chunk_list=[(iproc*npercore,(iproc+1)*npercore) for iproc in range(nthreads-1)]
                
        #Last core has to process the remaining elements    
        ind_chunk_list+=[((nthreads-1)*npercore,int(n_elem))]
        
    else:
        stop('Too many threads ('+str(nthreads)+') for the number of input elements ('+str(n_elem)+')')
   
    return ind_chunk_list



def findgen(n,int=False):
    """This is basically IDL's findgen function.
    a = findgen(5) will return an array with 5 elements from 0 to 4:
    [0,1,2,3,4]
    """
    import numpy as np
    if int:
        return np.linspace(0,n-1,n).astype(int)
    else:
        return np.linspace(0,n-1,n)

def path(dp):
    """This makes sure that a string that points to a folder (as a filepath)
    has a slash at the end."""
    if dp[-1] != '/':
        dp=dp+'/'
    return(dp)

def statusbar(i,x):
    if type(x) == int:
        print('  '+f"{i/(float(x)-1)*100:.1f} %", end="\r")
    else:
        print('  '+f"{i/(len(x)-1)*100:.1f} %", end="\r")#Statusbar.

def start():
    import time
    return(time.time())

def end(start,id=''):
    import time
    end=time.time()
    print('Elapsed %s: %s' % ('on timer '+id,end-start))
    return end-start

def nantest(varname,var):
    import numpy as np
    if np.isnan(var).any()  == True:
        raise Exception("NaN error: %s contains NaNs." % varname)
    if np.isinf(var).any()  == True:
        raise Exception("Finite error: %s contains in-finite values." % varname)

def postest(a,varname=''):
    """This function tests whether a number/array is strictly positive."""
    import numpy as np
    if np.min(a) <= 0:
        raise Exception('POSTEST ERROR: Variable %s should be strictly positive' % varname)

def notnegativetest(a,varname=''):
    """This function tests whether a number/array is strictly positive."""
    import numpy as np
    if np.min(a) < 0:
        raise Exception('POSTEST ERROR: Variable %s should not be negative' % varname)

def typetest(varname,var,vartype):
    """This program tests the type of var which has the name varname against
    the type vartype, and raises an exception if either varname is not a string,
    or if type(var) is not equal to vartype.

    Example:
    a = 'ohai'
    utils.typtest('a',a,str)"""
    if isinstance(varname,str) != True:
        raise Exception("Input error in typetest: varname should be of type string.")
    if isinstance(var,vartype) != True:
        raise Exception("Type error: %s should be %s." % (varname,vartype))

def typetest_array(varname,var,vartype):
    """This program tests the type of the elements in the array or list var which has the
    name varname, against the type vartype, and raises an exception if either
    varname is not a string, type(var) is not equal to numpy.array or list, or the elements of
    var are not ALL of a type equal to vartype.

    Example:
    a = ['alef','lam','mim']
    utils.typetest_array('teststring',a,str)"""
    #NEED TO FIX: MAKE SURE THAT A RANGE OF TYPES CAN BE TESTED FOR, SUCH AS
    #float, np.float32, np.float64... should all pass as a float.
    import numpy as np
    if isinstance(varname,str) != True:
        raise Exception("Input error in typetest: varname should be of type string.")
    if (isinstance(var,list) != True) and (isinstance(var,np.ndarray) != True):
        raise Exception("Input error in typetest_array: %s should be of class list or numpy array." % varname)
    for i in range(0,len(var)):
        typetest('element %s of %s' % (i,varname),var[i],vartype)


def dimtest(var,sizes):

    """This program tests the dimensions and shape of the input array var.
    Sizes is the number of elements on each axis.
    The program uses the above type tests to make sure that the input is ok.
    If an element in sizes is set to zero, that dimension is not checked against.
    Example:
    import numpy as np
    a=[[1,2,3],[4,3,9]]
    b=np.array(a)
    dimtest(a,2,[2,3])
    dimtest(a,2,[3,10])
    """
    import numpy as np
    typetest_array('sizes',sizes,int)
    ndim=len(sizes)

    if np.ndim(var) != ndim:
        raise Exception("Dimension error in vartest:  ndim = %s but was required to be %s." % (np.ndim(var),ndim))

    sizes_var=np.shape(var)

    for i in range(0,len(sizes)):
        if sizes[i] < 0:
            raise Exception("Sizes was not set correctly. It contains negative values. (%s)" % sizes(i))
        if sizes[i] > 0:
            if sizes[i] != sizes_var[i]:
                raise Exception("Dimension error in vartest: Axis %s contains %s elements, but %s were required." % (i,sizes_var[i],sizes[i]))

def save_stack(filename,list_of_2D_frames):
    """This code saves a stack of fits-files to a 3D cube, that you can play
    through in DS9. For diagnostic purposes."""
    import astropy.io.fits as fits
    import numpy as np
    base = np.shape(list_of_2D_frames[0])
    N = len(list_of_2D_frames)
    out = np.zeros((base[0],base[1],N))
    for i in range(N):
        out[:,:,i] = list_of_2D_frames[i]
    fits.writeto(filename,np.swapaxes(np.swapaxes(out,2,0),1,2),overwrite=True)

def writefits(filename,array):
    """This is a fast wrapper for fits.writeto, with overwrite enabled.... ! >_<"""
    fits.writeto(filename,array,overwrite=True)
    

def selmax(y_in,p,s=0.0):
    """This program returns the p (fraction btw 0 and 1) highest points in y,
    ignoring the very top s % (default zero, i.e. no points ignored), for the
    purpose of outlier rejection."""
    postest(p)
    y=deepcopy(y_in)#Copy itself, or there are pointer-related troubles...
    if s < 0.0:
        raise Exception("ERROR in selmax: s should be zero or positive.")
    if p >= 1.0:
        raise Exception("ERROR in selmax: p should be strictly between 0.0 and 1.0.")
    if s >= 1.0:
        raise Exception("ERROR in selmax: s should be strictly less than 1.0.")
    postest(-1.0*p+1.0)
    #nantest('y in selmax',y)
    dimtest(y,[0])#Test that it is one-dimensional.
    y[np.isnan(y)]=np.nanmin(y)#set all nans to the minimum value such that they will not be selected.
    y_sorting = np.flipud(np.argsort(y))#These are the indices in descending order (thats why it gets a flip)
    N=len(y)
    if s == 0.0:
        max_index = np.max([int(round(N*p)),1])#Max because if the fraction is 0 elements, then at least it should contain 1.0
        return y_sorting[0:max_index]

    if s > 0.0:
        min_index = np.max([int(round(N*s)),1])#If 0, then at least it should be 1.0
        max_index = np.max([int(round(N*(p+s))),2]) #If 0, then at least it should be 1+1.
        return y_sorting[min_index:max_index]

"""
Upsampling array that does not have to be equidistant (by integer factor)
"""
def upsample(x, sample=2):
    d=np.diff(x)/float(sample)
    r = d.repeat(sample)
    s=np.concatenate([[x[0]],np.cumsum(r)])
    return s




'''    
Repeat boundary value in case of bad interpolation on the edge
'''
def repeat_interp(interp_prof):
    
    #In case of bad interpolation on the edge, 
    wok=np.invert(np.isnan(interp_prof))  #well-defined points
    wlowok=min(np.where(wok==True)[0])    #first well-defined bin
    if wlowok>0:
        interp_prof[0:wlowok]=interp_prof[wlowok]
    whighok=max(np.where(wok==True)[0])
    if whighok<len(interp_prof)-1:
        interp_prof[whighok:len(interp_prof)]=interp_prof[whighok]
        
    return interp_prof





 
'''
Spline Interpolation of a given function
    Use scipy.interpolate.InterpolatedUnivariateSpline
    the sqrt is given as the error
    
    NB: here the shift is a translation
    
    if y is shifted like
       y[a+shift:b+shift]
    or xnew = x + shift
    or x = x - shift
    then: -shift>0 => ynew is blue shift
          -shift<0 => ynew is red shift
    NB: Be careful to the scale (not the same shift in x or y)
        
    keyword arguments:
    x -- Old x axis
    y -- Old y axis
    xnew -- New x axis
    k -- The Spline Order (1=linear, 3=cubic)

'''
def spline_inter(x, y, xnew, k=3 , ext = 0 ):
    splflux = InterpolatedUnivariateSpline(x, y, k=k , ext = ext)
    ynew = splflux(xnew)
    errorynew = np.sqrt(np.maximum(ynew, 0.))
    return ynew, errorynew




def air_index(l, t=15., p=760.):
    """Compute the refraction index n of the air 
    
    Author: C.Lovis
    
    wl_air = wl_vacuum/n
    n_vacuum=1.
    
    Parameters
    ----------
    :l (float, array): Wavelength in Angström
    :t (float, array): Air temperature in Celsius
    :p (float, array): Air pressure in millimeter of mercury
    
    Returns
    -------
    :n (float, array): Refraction index.
    
    """
    n = 1e-6 * p * (1 + (1.049-0.0157*t)*1e-6*p) / 720.883 / (1 + 0.003661*t) * (64.328 + 29498.1/(146-(1e4/l)**2) + 255.4/(41-(1e4/l)**2))
    n = n + 1
    return n


def spec_dopshift(rv_shift):
    return np.sqrt(1. - (rv_shift/c_light) )/np.sqrt(1. + (rv_shift/c_light) )

