#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import special
from copy import deepcopy
import lmfit
import bindensity as bind
from pathos.multiprocessing import Pool
from ..ANTARESS_analysis.ANTARESS_inst_resp import convol_prof,conv_st_prof_tab
from ..ANTARESS_general.utils import closest,init_parallel_func,np_where1D,stop

##################################################################################################    
#%%% Model line profiles
##################################################################################################

def gen_fit_prof(param_in,x_tab,args=None):
    r"""**Generic profile fit function**
    
    Function called by minimization algorithms, returning the chosen model profile after applyying (if requested) convolution, spectral conversion and resampling.
    
    The profile model function must return several arguments (or a list of one) with the model as first argument.
    The profile is calculated over a continuous table to allow for convolution and use of covariance matrix (fitted pixels are accounted for directly in the minimization routine).

    Args:
        TBD
    
    Returns:
        TBD
    
    """      
    args_exp = deepcopy(args['args_exp'])

    #In case param_in is defined as a Parameters structure, retrieve values and define dictionary
    if isinstance(param_in,lmfit.parameter.Parameters):
        param={}
        for par in param_in:param[par]=param_in[par].value
    else:param=param_in   
   
   	#Profile model
    sp_line_model = args['func_nam'](param,args_exp['cen_bins'],args)[0]
   
   	#Convolution and resampling 
    mod_prof = conv_st_prof_tab(None,None,None,args,args_exp,sp_line_model,args['FWHM_inst'])  

    return mod_prof




def dispatch_func_prof(func_name):
    r"""**Line profile model dispatching**
    
    Returns the chosen line profile function.

    Args:
        func_name (str): name of the chosen function
    
    Returns:
        func (function): chosen function
    
    """  
    func = {'voigt':voigt,
            'gauss':gauss_intr_prof,
            'cgauss':gauss_herm_lin,
            'pgauss':gauss_poly,
            'dgauss':dgauss,
        }[func_name]
    return func





def voigt(param,x,args=None):
    r"""**Line model: Voigt**
    
    Calculates Voigt profile, expressed in terms of the Faddeeva function (factorless)

    .. math::     
       V(rv) = Real(w(x))     
           
    As a function of 
    
    .. math:: 
       x &= ((rv-rv_0) + i \gamma)/(\sqrt{2} \sigma)   \\
         &= (rv-rv_0)/(\sqrt{2} \sigma) + i a       \\
         &= 2 \sqrt{\ln{2}} (rv-rv_0)/\mathrm{FWHM}_\mathrm{gauss} + i a
       
    Where `a` is the damping coefficient 
    
    .. math:: 
       a &= \gamma/(\sqrt{2} \sigma)            \\
         &= 2 \gamma \sqrt{\ln{2}}/\mathrm{FWHM}_\mathrm{gauss}       \\
         &= \sqrt{\ln{2}} \mathrm{FWHM}_\mathrm{lor}/\mathrm{FWHM}_\mathrm{gauss}       
       
    The :math:`\mathrm{FWHM}_\mathrm{gauss}` parameter is that of the Gaussian component (factorless)
    
    .. math::
       G(rv) &= \exp(- (rv-rv_0)^2/ ( \mathrm{FWHM}_\mathrm{gauss}/(2 \sqrt{\ln{2}}) )^2  )            \\     
             &= \exp(- (rv-rv_0)^2/ 2 ( \mathrm{FWHM}_\mathrm{gauss}/(2 \sqrt{2 \ln{2}}) )^2  )         \\
             &= \exp(- (rv-rv_0)^2/ 2 \sigma^2  )

    With :math:`\sigma = \mathrm{FWHM}_\mathrm{gauss}/(2 \sqrt{\ln{2}})` and :math:`\mathrm{FWHM}_\mathrm{lor} = 2 \gamma`
   
    The full-width at half maximum of the Voigt profile approximates as

    .. math::    
       \mathrm{FWHM}_\mathrm{voigt} = 0.5436 \mathrm{FWHM}_\mathrm{lor}+ \sqrt{0.2166 \mathrm{FWHM}_\mathrm{lor}^2 + \mathrm{FWHM}_\mathrm{gauss}^2} 
    
    From J.J.Olivero and R.L. Longbothum in Empirical fits to the Voigt line width: A brief review, JQSRT 17, P233    

    Args:
        TBD
    
    Returns:
        TBD
    
    """  
    z_tab = 2.*np.sqrt(np.log(2.))*(x - param['rv' ])/param['FWHM'] +  1j*param['a_damp']
    
    #Voigt profile
    voigt_peak = special.wofz(1j*param['a_damp']).real
    voigt_mod = 1. - (param['ctrst']/voigt_peak)*special.wofz(z_tab).real
  
    #Continuum        
    cont_pol = param['cont']*pol_cont(x,args,param)   

    return voigt_mod*cont_pol , cont_pol



def gauss_intr_prof(param,rv_tab,args=None):
    r"""**Line model: Gaussian**
    
    Calculates Gaussian profile.
    
    Args:
        TBD
    
    Returns:
        TBD
        
    """ 
    ymodel = [param['cont']*(1.-param['ctrst']*np.exp(-np.power( 2.*np.sqrt(np.log(2.))*(rv_tab-param['rv'])/param['FWHM']  ,2.  )))] 
    return ymodel





def gauss_herm_lin(param,x,args=None):
    r"""**Line model: Skewed Gaussian**
    
    Calculates Gaussian profile with hermitian polynom term for skewness/kurtosis.
    
    Args:
        TBD
    
    Returns:
        TBD
        
    """    
    x_tab = (2.*np.sqrt(2.*np.log(2.)))*(x-param['rv' ])/param['FWHM'] 
    
    #Skewness and kurtosis
    skew1 = param['skewA']
    kurt1 = param['kurtA']
    factor=1.
    if skew1!=0. or kurt1!=0.:  
        c0 = np.sqrt(6.)/4.
        c1 = -np.sqrt(3.)
        c2 = -np.sqrt(6.)
        c3 = 2./np.sqrt(3.)
        c4 = np.sqrt(6.)/3.        
    if skew1!=0.:factor+=skew1*(c1*x_tab+c3*x_tab**3.)
    if kurt1!=0.:factor+=kurt1*(c0+c2*x_tab**2.+c4*x_tab**4.)   
 
    #Skewd gaussian profile
    sk_gauss = 1.- factor*param['ctrst']*np.exp(-0.5*x_tab**2.)
     
    #Continuum        
    cont_pol = param['cont']*pol_cont(x,args,param)    

    return sk_gauss*cont_pol , cont_pol



def gauss_poly(param,RV,args=None):
    r"""**Line model: Sidelobed Gaussian**
    
    Calculates Gaussian profile with flat continuum and polynomial for sidelobes.
    
    Args:
        TBD
    
    Returns:
        TBD
        
    """ 
 
    #Gaussian with baseline set to continuum value
    cen_RV = param['rv']    #center
    y_gausscore=1.-param['ctrst']*np.exp(-np.power( 2.*np.sqrt(np.log(2.))*(RV-cen_RV)/param['FWHM']  ,2.  )) 
    ymodel = y_gausscore*param['cont']
    
    #Polynomial
    #    - P(x) = a0 + a1*(x-x0) + a2*(x-x0)^2 + a3*(x-x0)^3 + a4*(x-x0)^4 + a6*(x-x0)^6
    #      P must by symmetric, and have continuity of value and derivative in x=x0+-dx
    #      this must be true in particular if x0 = 0, and we thus solve:
    # P(x) = P(-x)
    # P(dx) = cont
    # P'(dx) = 0 
    #    - the continuum is set to its constant value beyond the continuity points
    c4_pol = param['c4_pol']                          #4th order coefficient
    c6_pol = param['c6_pol']                          #6th order coefficient
    dRV_joint=param['dRV_joint']                      #continuity points
    RV_joint_high = cen_RV + dRV_joint       
    RV_joint_low  = cen_RV - dRV_joint
    cond_lobes = (RV >= RV_joint_low) & (RV <= RV_joint_high) 
    y_polylobe = np.repeat(1.,len(RV))
    y_polylobe[cond_lobes] *= c4_pol*dRV_joint**4. + 2.*c6_pol*dRV_joint**6. - dRV_joint**2.*np.power(RV[cond_lobes]-cen_RV,2.)*(2.*c4_pol + 3.*c6_pol*dRV_joint**2.) + c4_pol*np.power(RV[cond_lobes]-cen_RV,4.) + c6_pol*np.power(RV[cond_lobes]-cen_RV,6.)
    ymodel[cond_lobes]*=y_polylobe[cond_lobes]    
    
    return ymodel, param['cont']*y_gausscore, param['cont']*y_polylobe
                 


def dgauss(param,rv_tab,args=None):
    r"""**Line model: Double Gaussian**
    
    Calculates Double-Gaussian profile, with Gaussian in absorption co-added with a Gaussian in emission at the core.
    
    The Gaussian profiles are centered but their amplitude and width can be fixed or independant. The model is defined as

    .. math::    
       F &= F_\mathrm{cont} + A_\mathrm{core} \exp(f_1) + A_\mathrm{lobe} \exp(f_2)       \\
       F &= F_\mathrm{cont} + A_\mathrm{core} ( \exp(f_1) - A_\mathrm{l2c} \exp(f_2) ) 
       
    We define a contrast parameter as the relative flux between the continuum and the CCF minimum 

    .. math::    
       C &= (F_\mathrm{cont} - (F_\mathrm{cont} + A_\mathrm{core} - A_\mathrm{core} A_\mathrm{l2c}) ) / F_\mathrm{cont}          \\
       C &= -A_\mathrm{core} ( 1 - A_\mathrm{l2c}) / F_\mathrm{cont}              \\
       A_\mathrm{core} &= -C F_\mathrm{cont}/( 1 - A_\mathrm{l2c})             \\
       A_\mathrm{lobe} &= -A_\mathrm{l2c} A_\mathrm{core} = A_\mathrm{l2c} C F_\mathrm{cont}/( 1 - A_\mathrm{l2c})
      
    Thus the model can be expressed as   

    .. math::     
       F &= F_\mathrm{cont} - C F_\mathrm{cont} ( \exp(f_1) - A_\mathrm{l2c} \exp(f_2) )/( 1 - A_\mathrm{l2c})             \\
       F &= F_\mathrm{cont} ( 1 - C ( \exp(f_1) - A_\mathrm{l2c} \exp(f_2) )/( 1 - A_\mathrm{l2c}) )     
    
    Args:
        TBD
    
    Returns:
        TBD
        
    """ 
 
    #Reduced contrast
    ctrst_red = param['ctrst']/( 1. - param['amp_l2c'])
     
    #Inverted gaussian core
    y_gausscore=param['cont']*(1. - ctrst_red*np.exp(-np.power(2.*np.sqrt(np.log(2.))*(rv_tab-param['rv'])/param['FWHM'],2))) 
    
    #Gaussian lobes
    cen_RV_lobes=param['rv']+param['rv_l2c']   
    FWHM_lobes = (param['FWHM'])*(param['FWHM_l2c'])   
    y_gausslobes=param['cont']*(1. + ctrst_red*param['amp_l2c']*np.exp(-np.power(2.*np.sqrt(np.log(2.))*(rv_tab-cen_RV_lobes)/FWHM_lobes,2)))     
    
    #Complete model
    ymodel = y_gausscore + y_gausslobes - param['cont']

    #Output models of each gaussian component superimposed to the continuum
    return ymodel , y_gausscore, y_gausslobes






def pol_cont(cen_bins_ref,args,param):
    r"""**Line polynomial continuum**
    
    Calculates polynomial modulated around 1.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    return (1. + param['c1_pol']*cen_bins_ref + param['c2_pol']*cen_bins_ref**2.  + param['c3_pol']*cen_bins_ref**3. + param['c4_pol']*cen_bins_ref**4. ) 



def calc_macro_ker_rt(rv_mac_kernel,param,cos_th,sin_th):
    r"""**Macroturbulence broadening: anisotropic Gaussian**
    
    Calculates broadening kernel for local macroturbulence, based on formulation with anisotropic gaussian (Takeda \& UeNo 2017)

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    
    if cos_th>0:term_R=(param['A_R']/(np.sqrt(np.pi)*param['ksi_R']*cos_th))*np.exp(-rv_mac_kernel**2./  ((param['ksi_R']*cos_th)**2.)  )   #Otherwise exp yield 0
    else:term_R=0.
    if sin_th>0:term_T=(param['A_T']/(np.sqrt(np.pi)*param['ksi_T']*sin_th))*np.exp(-rv_mac_kernel**2./  ((param['ksi_T']*sin_th)**2.)  )   #Otherwise exp yield 0 
    else:term_T=0.
    macro_ker_sub=term_R+term_T    
    return macro_ker_sub

def calc_macro_ker_anigauss(rv_mac_kernel,param,cos_th,sin_th):
    r"""**Macroturbulence broadening: radial-tangential**
    
    Calculates broadening kernel for local macroturbulence, based on radial-tangential model (Gray 1975, 2005)

    Args:
        TBD
    
    Returns:
        TBD
    
    """     
    macro_ker_sub=np.exp(  -rv_mac_kernel**2./  ((param['eta_R']*cos_th)**2.  +  (param['eta_T']*sin_th)**2.)  )
    return macro_ker_sub





##################################################################################################    
#%%% Model line properties
##################################################################################################

def calc_linevar_coord_grid(dim,grid):
    r"""**Line profile coordinate dispatching**
    
    Returns the relevant coordinate for calculation of line properties variations.

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 
    if (dim in ['mu','xp_abs','r_proj','y_st']):linevar_coord_grid = grid[dim]
    elif (dim=='y_st2'):linevar_coord_grid = grid['y_st']**2.
    elif (dim=='abs_y_st'):linevar_coord_grid = np.abs(grid['y_st'])
    else:stop('Undefined line coordinate')
    return linevar_coord_grid


def calc_polymodu(pol_mode,coeff_pol,x_val):
    r"""**Line profile coordinate models**
    
    Calculates absolute or modulated polynomial
    
    The list of polynomial coefficient 'coeff_pol' has been defined in decreasing powers, as expected by `poly1d`, using input coefficient defined through their power value (see `polycoeff_def()`)

    Args:
        TBD
    
    Returns:
        TBD
    
    """     
    
    #Absolute polynomial
    #    - coeff_pol[0]*x^n + coeff_pol[1]*x^(n-1) .. + coeff_pol[n]*x^0
    if pol_mode=='abs':
        mod= np.poly1d(coeff_pol)(x_val)         

    #Modulated polynomial
    #    - (coeff_pol[0]*x^n + coeff_pol[1]*x^(n-1) .. + 1)*coeff_pol[n] 
    elif pol_mode=='modul':
        coeff_pol_modu = coeff_pol[:-1] + [1]
        mod= coeff_pol[-1]*np.poly1d(coeff_pol_modu)(x_val)  
    else:stop('Undefined polynomial mode')
    return mod





def polycoeff_def(param,coeff_ord2name_polpar):
    r"""**Line profile coordinate coefficients**
    
    Defines polynomial coefficients from the `Parameter()` format through their power value

    Args:
        TBD
    
    Returns:
        TBD
    
    """ 

    #Polynomial coefficients 
    #    - keys in 'coeff_ord2name_polpar' are the coefficients degrees, values are their names, as defined in 'param' 
    #      they can be defined in disorder (in terms of degrees), as coefficients are forced to order from deg_max to 0 in 'coeff_grid_polpar' 
    #    - degrees can be missing
    #    - input coefficients must be given in decreasing order of degree to poly1d
    deg_max=max(coeff_ord2name_polpar.keys())
    coeff_grid_polpar=[param[coeff_ord2name_polpar[ideg]] if ideg in coeff_ord2name_polpar else 0. for ideg in range(deg_max,-1,-1)]

    return coeff_grid_polpar





# Stage Théo : fusion des 2 dernières fonctions. 
def poly_prop_calc(param,fit_coord_grid,coeff_ord2name_polpar, pol_mode):

    #Polynomial coefficients 
    #    - keys in 'coeff_ord2name_polpar' are the coefficients degrees, values are their names, as defined in 'param' 
    #      they can be defined in disorder (in terms of degrees), as coefficients are forced to order from 0 to deg_max in 'coeff_pol_ctrst' 
    #    - degrees can be missing
    #     - input coefficients must be given in decreasing order of degree to poly1d
    deg_max=max(coeff_ord2name_polpar.keys())
    
    coeff_grid_polpar=[param[coeff_ord2name_polpar[ideg]] if ideg in coeff_ord2name_polpar else 0. for ideg in range(deg_max,-1,-1)]
    
    # Absolute polynomial 
    if pol_mode == 'abs' : 
        polpar_grid = np.poly1d(coeff_grid_polpar)(fit_coord_grid)
        
    # Modulated polynomial
    if pol_mode == 'modul' :
        coeff_pol_modu = coeff_grid_polpar[:-1] + [1]
        polpar_grid = np.poly1d(coeff_pol_modu)(fit_coord_grid)*coeff_grid_polpar[-1]
        
    return polpar_grid









##################################################################################################    
#%%% Model line analysis
##################################################################################################

def calc_biss(Fnorm_in,RV_tab_in,RV_min,max_rv_range,dF_grid,resamp_mode,Cspan):
    r"""**Bissector.**
    
    Calculates bissector for a line profile provided in RV space as input.
    
    Args:
        TBD
    
    Returns:
        TBD
        
    """ 
    #Reduction to maximum range    
    cond_range = (RV_tab_in>max_rv_range[0]) & (RV_tab_in<max_rv_range[-1])
    Fnorm = Fnorm_in[cond_range]
    RV_tab = RV_tab_in[cond_range]

    #Blue and red wings of input line
    idx_blue=np_where1D(RV_tab<RV_min)
    idx_red=np_where1D(RV_tab>=RV_min)
  
    #Calculation range
    #    - between the maxima on each side of the line
    max_blue=max(Fnorm[idx_blue])
    max_red=max(Fnorm[idx_red])
    idx_max_blue=np_where1D(Fnorm==max_blue)[0]
    idx_max_red=np_where1D(Fnorm==max_red)[0]
    idx_min = closest(RV_tab,RV_min)
    idx_biss_blue=range(idx_max_blue,idx_min+1)
    idx_biss_red=range(idx_min,idx_max_red+1)
    Fnorm_biss = Fnorm[idx_max_blue:idx_max_red+1]
     
    #RV = f(flux) in blue and red wings
    RV_biss_blue=RV_tab[idx_biss_blue]
    F_biss_blue=Fnorm[idx_biss_blue]
    idx_sort = np.argsort(F_biss_blue)
    RV_biss_blue = RV_biss_blue[idx_sort]
    F_biss_blue = F_biss_blue[idx_sort]
    Fedge_biss_blue = 0.5*(F_biss_blue[0:-1]+F_biss_blue[1:])
    Fedge_biss_blue = np.concatenate(( [F_biss_blue[0] - 0.5*(F_biss_blue[1]-F_biss_blue[0])] , Fedge_biss_blue , [F_biss_blue[-1]+ 0.5*(F_biss_blue[-1]-F_biss_blue[-2])] ))
    RV_biss_red=RV_tab[idx_biss_red]
    F_biss_red=Fnorm[idx_biss_red]
    idx_sort = np.argsort(F_biss_red)
    RV_biss_red = RV_biss_red[idx_sort]
    F_biss_red = F_biss_red[idx_sort]
    Fedge_biss_red = 0.5*(F_biss_red[0:-1]+F_biss_red[1:])
    Fedge_biss_red = np.concatenate(([ F_biss_red[0] - 0.5*(F_biss_red[1]-F_biss_red[0])] , Fedge_biss_red , [F_biss_red[-1]+ 0.5*(F_biss_red[-1]-F_biss_red[-2])] ))
    
    #Resampling RV versus flux
    minF_biss=np.min(Fnorm_biss)
    maxF_biss=np.max(Fnorm_biss)
    nF_biss=int((maxF_biss-minF_biss)/dF_grid)
    F_biss_edgeHR=minF_biss+dF_grid*np.arange(nF_biss+1)
    RV_biss_blue_HR = bind.resampling(F_biss_edgeHR,Fedge_biss_blue,RV_biss_blue,kind = resamp_mode)            
    RV_biss_red_HR = bind.resampling(F_biss_edgeHR,Fedge_biss_red,RV_biss_red,kind = resamp_mode)

    #Bissector
    RV_biss_HR=0.5*(RV_biss_blue_HR+RV_biss_red_HR)
    F_biss_HR = 0.5*(F_biss_edgeHR[0:-1]+F_biss_edgeHR[1:])
    
    #Bissector span
    #    - difference in velocity between minimum and region of larger variation in the series
    if Cspan is not None:idx_span = closest(F_biss_HR,1. - Cspan*(1.-minF_biss) )
    else:idx_span = np.argmax(np.abs(RV_biss_HR-RV_min))
    RV_biss_span = RV_biss_HR[idx_span]
    F_biss_span = F_biss_HR[idx_span]
    bis_span = RV_biss_span - RV_min
                                  
    return RV_biss_HR,F_biss_HR,RV_biss_span,F_biss_span,bis_span



def gauss_intr_prop(ctrst_intr,FWHM_intr,FWHM_inst):
    r"""**Intrinsic > Measured Gaussian properties.**
    
    Estimates the FWHM and contrast of a measured-like Gaussian profile, defined as the convolution of a Gaussian intrinsic model profile and a Gaussian instrumental response.

    The FWHM can be expressed as
    
    .. math:: 
        \mathrm{FWHM} &= \sqrt{\mathrm{FWHM}_\mathrm{intr}^2 + \mathrm{FWHM}_\mathrm{inst}^2 } \\
        \mathrm{with}\,\sigma &= \frac{\mathrm{FWHM}}{2 \sqrt{\ln{2}}} = \sqrt{\sigma_\mathrm{intr}^2 + \sigma_\mathrm{inst}^2 }
        
    The area of the measured convolution profile is equal to the product of the area of the intrinsic and instrumental convolved profiles, providing an analytical expression for the contrast.  
    The instrumental kernel, with area = 1, is expressed as

    .. math::     
        f_\mathrm{kern}(rv) &= \frac{1}{\sqrt{2 \pi} \sigma_\mathrm{inst} } \exp^{ -\frac{rv^2}{2 \sigma_\mathrm{inst}^2} } \\
        \mathrm{where}\,\sigma_\mathrm{inst} &= \frac{\mathrm{FWHM}_\mathrm{inst}}{ 2 \sqrt{2 \log{2}} }
            
    The intrinsic profile, with area = :math:`\mathrm{A}_\mathrm{intr}`, is expressed as
    
    .. math::     
        f_\mathrm{intr}(rv) &= \frac{\mathrm{A}_\mathrm{intr}}{\sqrt{2 \pi} \sigma_\mathrm{intr} } \exp^{ -\frac{rv^2}{2 \sigma_\mathrm{intr}^2} } \\
        \mathrm{where}\,\sigma_\mathrm{intr} &= \frac{\mathrm{FWHM}_\mathrm{intr}}{ 2 \sqrt{2 \ln{2}} }    
    
    The contrast `C` of the measured profile relates to its area `A` as
    
    .. math::     
        C &= \frac{A}{\sqrt{2 \pi} \sigma }  \\
          &= \frac{1 \times \mathrm{A}_\mathrm{intr} }{ \sqrt{2 \pi (\sigma_\mathrm{intr}^2 + \sigma_\mathrm{inst}^2 ) }}  \\
          &= \frac{\mathrm{C}_\mathrm{intr} \sqrt{2 \pi} \sigma_\mathrm{intr} }{\sqrt{2 \pi (\sigma_\mathrm{intr}^2 + \sigma_\mathrm{inst}^2 ) }}   \\        
          &= \frac{\mathrm{C}_\mathrm{intr} \sigma_\mathrm{intr} }{\sqrt{\sigma_\mathrm{intr}^2 + \sigma_\mathrm{inst}^2}}   \\          
          &= \frac{\mathrm{C}_\mathrm{intr} \mathrm{FWHM}_\mathrm{intr} }{\sqrt{\mathrm{FWHM}_\mathrm{intr}^2 + \mathrm{FWHM}_\mathrm{inst}^2}}    
                                     
    This is only valid for analytical profiles. For numerical profiles, the model is the sum of intrinsic profiles over the planet-occulted surface that may have non-Gaussian shape, and which
    are further broadened by the local surface `rv` field. In that case, use `cust_mod_true_prop()`. 
    
    Args:
        TBD
    
    Returns:
        TBD
        
    """ 
    FWHM_meas = np.sqrt(FWHM_intr**2. + FWHM_inst**2.) 
    ctrst_meas = ctrst_intr*FWHM_intr/FWHM_meas
    return ctrst_meas,FWHM_meas







def cust_mod_true_prop(param,velccf_HR,args):
    r"""**Intrinsic > Measured line properties.**
    
    Estimates the FWHM and contrast of a measured-like line profile with custom shape.
    This is done numerically, calculating the model profile at high resolution.
    
    Contrast is counted from the estimated continuum.
    For double-gaussian profiles, the peaks of the lobes are considered as the closest estimate to the true stellar continuum.
    
    FWHM is calculated by finding the points on the blue side at half-maximum, and on the red side.
    
    Args:
        TBD
    
    Returns:
        TBD
        
    """ 
    #HR model
    CCF_HR_intr = args['fit_func_gen'](param,velccf_HR,args=args)[0]
    
    #Instrumental convolution 
    if args['FWHM_inst'] is not None:CCF_HR = convol_prof(CCF_HR_intr,velccf_HR,args['FWHM_inst'])  
    else:CCF_HR=CCF_HR_intr
    
    #Approximation of continuum
    max_CCF = np.max(CCF_HR)
    
    #CCF inverted peak
    min_CCF = np.min(CCF_HR)
    
    #Estimates of amplitude and contrast
    true_amp=max_CCF - min_CCF 
    true_ctrst=1. - (min_CCF/max_CCF) 
    
    #Blue side and red side of model CCF
    blueRV_HR=np_where1D(velccf_HR<=param['rv'])
    if len(blueRV_HR)==0:blueRV_HR=[0]
    redRV_HR=np_where1D(velccf_HR>=param['rv'])
    if len(redRV_HR)==0:redRV_HR=[-1]
    
    #Model pixels closest to half-maximum (mid-point between maximum and minimum of model) on each wing
    idx_blue_WHM=closest(CCF_HR[blueRV_HR],0.5*(np.max(CCF_HR[blueRV_HR])+min_CCF))  
    idx_red_WHM=closest(CCF_HR[redRV_HR],0.5*(np.max(CCF_HR[redRV_HR])+min_CCF))   
    
    #Estimate of FWHM defined as sum of HWHM in each wing
    true_FWHM = (param['rv'] - velccf_HR[blueRV_HR][idx_blue_WHM]) + (velccf_HR[redRV_HR][idx_red_WHM] - param['rv'])       
           
    return true_ctrst,true_FWHM,true_amp    

def proc_cust_mod_true_prop(merged_chain_proc,args,param_loc):
    r"""**Wrap-up of cust_mod_true_prop().**
    
    Estimates the FWHM and contrast of a measured-like line profile with custom shape, for a sample of properties.
    
    Args:
        TBD
    
    Returns:
        TBD
        
    """     
    nsamp=len(merged_chain_proc[:,0])    
    param_proc = deepcopy(param_loc)
    chain_loc_proc=np.empty([3,0],dtype=float) 
    for istep in range(nsamp): 
        for ipar,par in enumerate(args['var_par_list']):param_proc[par]=merged_chain_proc[istep,ipar]     
        true_ctrst_step,true_FWHM_step,true_amp_step=cust_mod_true_prop(param_proc,args['cen_bins_HR'],args)  
        chain_loc_proc=np.append(chain_loc_proc,[[true_ctrst_step],[true_FWHM_step],[true_amp_step]],axis=1) 
    return chain_loc_proc
    
def para_cust_mod_true_prop(func_input,nthreads,n_elem,y_inputs,common_args):  
    r"""**Multithreading routine for proc_cust_mod_true_prop().**

    Args:
        func_input (function): multi-threaded function
        nthreads (int): number of threads
        n_elem (int): number of elements to thread
        y_inputs (list): threadable function inputs 
        common_args (tuple): common function inputs
    
    Returns:
        y_output (None or specific to func_input): function outputs 
    
    """        
    pool_proc = Pool(processes=nthreads)   #cannot be passed through lmfit
    ind_chunk_list=init_parallel_func(nthreads,n_elem)
    chunked_args=[(y_inputs[0][ind_chunk[0]:ind_chunk[1],:],)+common_args for ind_chunk in ind_chunk_list]	
    all_results=tuple(tab for tab in pool_proc.starmap(func_input,chunked_args))				
    y_output=np.concatenate(tuple(all_results[i] for i in range(nthreads)),axis=1)
    pool_proc.close()
    pool_proc.join() 
    return y_output
































