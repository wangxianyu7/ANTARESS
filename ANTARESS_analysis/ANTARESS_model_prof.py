import numpy as np
from utils import closest,init_parallel_func,np_where1D
from scipy import special
from copy import deepcopy
import lmfit
import bindensity as bind
from pathos.multiprocessing import Pool
from ANTARESS_analysis.ANTARESS_inst_resp import convol_prof,conv_st_prof_tab



'''
Sub-function to modify parameters properties using input values
'''
def par_formatting(p_start,model_prop,priors_prop,fit_dic,fixed_args,inst,vis):
  
    #Parameters are used for fitting   
    if isinstance(p_start,lmfit.parameter.Parameters):fit_used=True
    else:fit_used=False

    #Process default / additional parameters
    fixed_args['varpar_priors']={}
    for par in np.unique( list(p_start.keys()) + list(model_prop.keys())  ):    

        #Overwrite default properties 
        if (par in model_prop):
            
            #Check property is used for current instrument
            #    - this is to avoid modifying a property for other instruments than the one(s) it is set up with in model_prop
            #    - except for planet/star properties independent of a dataset (defined as 'physical')
            if (inst in par) or ('__IS' in par) or ((inst in model_prop[par]) and (vis in model_prop[par][inst])) or (('physical' in model_prop[par]) and (model_prop[par]['physical'] is True)):
     
                #Properties are fitted
                if fit_used:
                
                    #Properties depend on instrument and/or visit or is common to all
                    if (inst in model_prop[par]):
                        if (vis in model_prop[par][inst]):model_prop_par = model_prop[par][inst][vis]
                        else:model_prop_par = model_prop[par][inst]
                    else:model_prop_par = model_prop[par] 

                    #Overwrite property fields
                    if (par in p_start):
                        p_start[par].value  = model_prop_par['guess']  
                        p_start[par].vary  = model_prop[par]['vary']                   
                    
                    #Add property
                    else:  
                        p_start.add(par, model_prop_par['guess']  ,model_prop[par]['vary']  ,None , None, None)
         
                    #Value linked to other parameter
                    if ('expr' in model_prop_par):p_start[par].expr = model_prop_par['expr']  
                    
                #Fixed properties
                else:p_start[par] = model_prop[par]               
             
        #Variable parameter
        if fit_used and (par in p_start) and p_start[par].vary:
            
            #Chi2 fit
            #    - overwrite default priors
            if (fit_dic['fit_mod']=='chi2'):
                if (par in priors_prop) and (priors_prop[par]['mod']=='uf'): 
                    p_start[par].min = priors_prop[par]['low']
                    p_start[par].max = priors_prop[par]['high']

                #Change guess value if beyond prior range
                if (not np.isinf(p_start[par].min)) and (np.isinf(p_start[par].max)) and (p_start[par].value<p_start[par].min):p_start[par].value=p_start[par].min
                if (np.isinf(p_start[par].min)) and (not np.isinf(p_start[par].max)) and (p_start[par].value>p_start[par].max):p_start[par].value=p_start[par].max
                if (not np.isinf(p_start[par].min)) and (not np.isinf(p_start[par].max)) and ((p_start[par].value<p_start[par].min) or (p_start[par].value>p_start[par].max)):p_start[par].value=0.5*(p_start[par].min+p_start[par].max)

            #MCMC fit
            elif (fit_dic['fit_mod']=='mcmc'):
                
                #Range for walkers initialization
                if (par in model_prop):fit_dic['uf_bd'][par]=model_prop_par['bd']
                else:
                    uf_bd=[-1e5,1e5]
                    if (not np.isinf(p_start[par].min)):uf_bd[0]=p_start[par].min
                    if (not np.isinf(p_start[par].max)):uf_bd[1]=p_start[par].max
                    fit_dic['uf_bd'][par]=uf_bd
                
                #Priors
                if (par in priors_prop):
                    fixed_args['varpar_priors'][par] = priors_prop[par]                          
                else:
                    varpar_priors=[-1e5,1e5]
                    if (not np.isinf(p_start[par].min)):varpar_priors[0]=p_start[par].min
                    if (not np.isinf(p_start[par].max)):varpar_priors[1]=p_start[par].max                
                    fixed_args['varpar_priors'][par]={'mod':'uf','low':varpar_priors[0],'high':varpar_priors[1]}

                #Change guess value if beyond prior range
                if ((p_start[par].value<fixed_args['varpar_priors'][par]['low']) or (p_start[par].value>fixed_args['varpar_priors'][par]['high'])):p_start[par].value=0.5*(fixed_args['varpar_priors'][par]['low']+fixed_args['varpar_priors'][par]['high'])


    #---------------------------------------------------

    #Associate the correct input names for model functions with instrument and visit dependence
    #     - properties must be defined in p_start as 'propN__ISx_VSy'
    #  + N is the degree of the coefficient (if relevant)
    #  + x is the id of the instrument
    #  + y is the id of the visit associated with the instrument
    #     - x and y can be set to '_' so that the coefficient is common to all instruments and/or visits
    fixed_args['name_prop2input'] = {}
    fixed_args['coeff_ord2name'] = {}
    fixed_args['linevar_par'] = {}

    #Retrieve all root name parameters with instrument/visit dependence, possibly with several orders (polynomial degree, or list)
    par_list=[]
    root_par_list=[]
    fixed_args['genpar_instvis']  = {}
    if 'inst_list' not in fixed_args:fixed_args['inst_list']=[inst]
    if 'inst_vis_list' not in fixed_args:fixed_args['inst_vis_list']={inst:[vis]}
    for par in p_start:
 
        #Parameter depend on instrument and visit
        if ('__IS') and ('_VS') in par:
            root_par = par.split('__IS')[0]
            inst_vis_par = par.split('__IS')[1]
            inst_par  = inst_vis_par.split('_VS')[0]
            vis_par  = inst_vis_par.split('_VS')[1]              
            if root_par not in fixed_args['genpar_instvis'] :fixed_args['genpar_instvis'][root_par] = {}          
            if inst_par not in fixed_args['genpar_instvis'][root_par]:fixed_args['genpar_instvis'][root_par][inst_par]=[]
            if vis_par not in fixed_args['genpar_instvis'][root_par][inst_par]:fixed_args['genpar_instvis'][root_par][inst_par]+=[vis_par]                  
            root_par_list+=[root_par]            
            par_list+=[par]

            #Parameter vary as polynomial of spatial stellar coordinate
            if ('_ord' in par):
                gen_root_par = par.split('_ord')[0] 
                
                #Define parameter for current instrument and visit (if specified) or all instruments and visits (if undefined) 
                if inst_par in fixed_args['inst_list']:inst_list = [inst_par]
                elif inst_par=='_':inst_list = fixed_args['inst_list'] 
                for inst_loc in inst_list:
                    if inst_loc not in fixed_args['coeff_ord2name']:fixed_args['coeff_ord2name'][inst_loc] = {}
                    if vis_par in fixed_args['inst_vis_list'][inst_loc]:vis_list = [vis_par]
                    elif vis_par=='_':vis_list = fixed_args['inst_vis_list'][inst_loc]              
                    for vis_loc in vis_list:
                        if vis_loc not in fixed_args['coeff_ord2name'][inst_loc]:fixed_args['coeff_ord2name'][inst_loc][vis_loc] = {}                
                        if gen_root_par not in fixed_args['coeff_ord2name'][inst_loc][vis_loc]:fixed_args['coeff_ord2name'][inst_loc][vis_loc][gen_root_par]={}
    
                        #Identify stellar line properties with polynomial spatial dependence 
                        if gen_root_par in ['ctrst','FWHM','amp_l2c','rv_l2c','FWHM_l2c','a_damp','rv_line']:
                            if inst_loc not in fixed_args['linevar_par']:fixed_args['linevar_par'][inst_loc]={}
                            if vis_loc not in fixed_args['linevar_par'][inst_loc]:fixed_args['linevar_par'][inst_loc][vis_loc]=[]
                            if gen_root_par not in fixed_args['linevar_par'][inst_loc][vis_loc]:fixed_args['linevar_par'][inst_loc][vis_loc]+=[gen_root_par]                     

    #Process parameters with dependence on instrument/visit
    for root_par in np.unique(root_par_list):

        #Parameter is associated with order coefficient
        if ('_ord' in root_par):
            deg_coeff=int(root_par[-1])
            gen_root_par = root_par.split('_ord')[0]  
        else:deg_coeff=None
         
        #Property common to all instruments is also common to all visits
        #    - we associate the property for current instrument and visit to the property common to all instruments and visits
        if any([root_par+'__IS__' in str_loc for str_loc in par_list]):
            for inst in fixed_args['inst_list']:
                for vis in fixed_args['inst_vis_list'][inst]:
                    par_input = root_par+'__IS'+inst+'_VS'+vis                    
                    fixed_args['name_prop2input'][par_input] = root_par+'__IS__VS_'
                    if deg_coeff is not None:
                        fixed_args['coeff_ord2name'][inst][vis][gen_root_par][deg_coeff]=root_par+'__IS__VS_' 
            
        #Property is specific to a given instrument
        else:

            #Process all fitted instruments associated with the property
            for inst in fixed_args['inst_list']:
                if (inst in fixed_args['genpar_instvis'][root_par]):

                    #Property is common to all visits of current instrument
                    #    - we associate the property for current instrument and visit to the value specific to this instrument, common to all visits
                    if any([root_par+'__IS'+inst+'_VS_' in str_loc for str_loc in par_list]): 
                        for vis in fixed_args['inst_vis_list'][inst]:
                            par_input = root_par+'__IS'+inst+'_VS'+vis 
                            fixed_args['name_prop2input'][par_input] = root_par+'__IS'+inst+'_VS_'  
                            if deg_coeff is not None:fixed_args['coeff_ord2name'][inst][vis][gen_root_par][deg_coeff]=root_par+'__IS'+inst+'_VS_'      
        
                    #Property is specific to the visit
                    #    - we associate the property for current instrument and visit to the value specific to this instrument, and this visit
                    else:
                        
                        #Process all fitted visits associated with the property
                        for vis in fixed_args['inst_vis_list'][inst]:
                            if (vis in fixed_args['genpar_instvis'][root_par][inst]):   
                                par_input = root_par+'__IS'+inst+'_VS'+vis 
                                fixed_args['name_prop2input'][par_input] = root_par+'__IS'+inst+'_VS'+vis   
                                if deg_coeff is not None:fixed_args['coeff_ord2name'][inst][vis][gen_root_par][deg_coeff]=root_par+'__IS'+inst+'_VS'+vis  
                    
    return p_start























'''
Bissector calculation
    - for a line profile provided in RV space as input
'''
def calc_biss(Fnorm_in,RV_tab_in,RV_min,max_rv_range,dF_grid,resamp_mode,Cspan):

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



'''
General oversampling of model line profiles
    function must return several arguments (or a list of one) with the model as first argument
    model is calculated over a continuous table to allow for convolution and use of covariance matrix (fitted pixels are accounted for directly in the minimization routine)
'''
def gen_over_func(param_in,func_nam,args):
    args_exp = deepcopy(args['args_exp'])

    #In case param_in is defined as a Parameters structure, retrieve values and define dictionary
    if isinstance(param_in,lmfit.parameter.Parameters):
        param={}
        for par in param_in:param[par]=param_in[par].value
    else:param=param_in   
    
    #Line profile model
    sp_line_model = func_nam(param,args_exp['cen_bins'],args)[0]

    #Add offset if required
    if ('offset' in param):sp_line_model+=param['offset']    

    #Conversion and resampling 
    mod_prof = conv_st_prof_tab(None,None,None,args,args_exp,sp_line_model,args['FWHM_inst'])    

    return mod_prof






'''
Dispatching function
'''
def dispatch_func_prof(func_name):
    return {'gauss':gauss_intr_prof,
            'dgauss':dgauss,
            'voigt':voigt
        }[func_name]


'''
Polynomial continuum for stellar line models
'''
def pol_cont(cen_bins_ref,args,param):
    return (1. + param['c1_pol']*cen_bins_ref + param['c2_pol']*cen_bins_ref**2.  + param['c3_pol']*cen_bins_ref**3. + param['c4_pol']*cen_bins_ref**4. ) 

'''
Gaussian with hermitian polynom term for skewness/kurtosis and with linear trend
'''
def MAIN_gauss_herm_lin(param,RV,args=None):
    ymodel=gen_over_func(param,gauss_herm_lin,args)
    return ymodel

'''
Gaussian with hermitian polynom term for skewness/kurtosis and with linear trend
'''
def gauss_herm_lin(param,x,args=None):
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

'''
Voigt profile main function
'''
def MAIN_voigt(param,RV,args=None):
    ymodel=gen_over_func(param,voigt,args)
    return ymodel

'''
Voigt profile with linear trend
    - the FWHM_gauss parameter is that of the gaussian component (factorless):
 G(rv) = exp(- (rv-rv0)^2/ ( FWHM_gauss/(2 sqrt(ln(2))) )^2  )             
 G(rv) = exp(- (rv-rv0)^2/ 2*( FWHM_gauss/(2 sqrt(2ln(2))) )^2  )  
 G(rv) = exp(- (rv-rv0)^2/ 2*sig^2  ) 
      with sig = FWHM_gauss/(2 sqrt(2ln(2)))          
    - the Voigt profile is expressed in terms of the Faddeeva function (factorless):
 V(rv) = Real(w(x))      
      with x = ((rv-rv0) + i*gamma)/(sqrt(2)*sig)
      we use the damping coefficient 
 a = gamma/(sqrt(2)*sigma) = 2*gamma*sqrt(ln(2))/FWHM_gauss 
      which can be expressed as a function of FWHM_LOR = 2*gamma, so that 
 a = sqrt(ln(2))*FWHM_LOR/FWHM_gauss 
      and
 x = (rv-rv0)/(sqrt(2)*sig) + i*a 
   = 2*sqrt(ln(2))*(rv-rv0)/FWHM_gauss + i*a
    - the full-width at half maximum of the Voigt profile approximates as:
 fwhm_voigt = (0.5436*fwhm_lor+ np.sqrt(0.2166*pow(fhwhm_lor,2.)+pow(fwhm_gauss,2.)) )
      from J.J.Olivero and R.L. Longbothum in Empirical fits to the Voigt line width: A brief review, JQSRT 17, P233
'''
def voigt(param,x,args=None):
    z_tab = 2.*np.sqrt(np.log(2.))*(x - param['rv' ])/param['FWHM'] +  1j*param['a_damp']
    
    #Voigt profile
    voigt_peak = special.wofz(1j*param['a_damp']).real
    voigt_mod = 1. - (param['ctrst']/voigt_peak)*special.wofz(z_tab).real
                 
    #Continuum        
    cont_pol = param['cont']*pol_cont(x,args,param)   

    return voigt_mod*cont_pol , cont_pol


'''
Simple gaussian profile 
'''
def gauss_intr_prof(param,rv_tab,args=None):
    return [param['cont']*(1.-param['ctrst']*np.exp(-np.power( 2.*np.sqrt(np.log(2.))*(rv_tab-param['rv'])/param['FWHM']  ,2.  )))] 


'''
Function returning the measured FWHM and contrast of intrinsic stellar profiles with gaussian shape
    - if the intrinsic line is a gaussian, its FWHM is sqrt( FWHM_intr^2 + FWHM_instr^2 )
      the area of the convolution is also equal to the product of the area of the two convolved profiles, providing an analytical value for the contrast of the convolved profile:
 + gaussian kernel : fkern(rv) = (1/( sqrt(2*pi)*sig_k ))*exp( -rv^2 / (2*sig_k^2) ) 
                     sig_k  = FWHM_inst/( 2.*np.sqrt(2*np.log(2.)) )
 + profile : fprof(rv) = A*(1/( sqrt(2*pi)*sig_p ))*exp( -rv^2 / (2*sig_p^2) )    
                     sig_p  = FWHM_exp/( 2.*np.sqrt(2*np.log(2.)) )
                     A = ctrst_exp*sqrt(pi)*FWHM_exp/( 2.*np.sqrt(np.log(2.)) )
 + amplitude of the convolution profile: A/sqrt(2*pi*( sig_k^2 + sig_p^2 ))
                                       = ctrst_exp*sqrt(pi)*FWHM_exp/(  2.*np.sqrt(np.log(2.)) * sqrt(2*pi*( sig_k^2 + sig_p^2 ))  )
                                       = ctrst_exp*FWHM_exp/(  2.*np.sqrt(2*np.log(2.)) * sqrt(sig_k^2 + sig_p^2 )  )
 sig_k^2 + sig_p^2 = (FWHM_inst^2 + FWHM_exp^2) /( 2.*np.sqrt(2*np.log(2.)) )^2
                                       = ctrst_exp*FWHM_exp/( sqrt(FWHM_inst^2 + FWHM_exp^2)  )
    - this is only valid for analytical profiles
      for numerical profiles, the model is the sum of intrinsic profiles over the planet-occulted surface and may be further broadened by the local surface rv field                            
'''
def gauss_intr_prop(ctrst_intr,FWHM_intr,FWHM_inst):
    FWHM_meas = np.sqrt(FWHM_intr**2. + FWHM_inst**2.) 
    ctrst_meas = ctrst_intr*FWHM_intr/FWHM_meas
    return ctrst_meas,FWHM_meas








'''
Gaussian with flat continuum and polynomial for sidelobes
'''
def MAIN_gauss_poly(param,RV,args=None):
    ymodel=gen_over_func(param,gauss_poly,args)
    return ymodel

def gauss_poly(param,RV,args=None):
 
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
                 





'''
Model with gaussian for continuum, and inverted gaussian for the core
    - the gaussian profiles are centered but their amplitude and width can be fixed or independant
    - the model is defined as :
 CCF = cont + amp_core*exp(f1) + amp_lobe*exp(f2)
 CCF = cont + amp_core*( exp(f1) - amp_l2c*exp(f2) ) 
      we define a contrast parameter as the relative flux between the continuum and the CCF minimum :
 C = (cont - (cont + amp_core - amp_core*amp_l2c) ) / cont 
 C = -amp_core*( 1 - amp_l2c) / cont
 amp_core = -C*cont/( 1 - amp_l2c)
 amp_lobe = -amp_l2c*amp_core = amp_l2c*C*cont/( 1 - amp_l2c)
      thus the model can be expressed as :    
 CCF = cont - C*cont*( exp(f1) - amp_l2c*exp(f2) )/( 1 - amp_l2c)
 CCF = cont*( 1 - C*( exp(f1) - amp_l2c*exp(f2) )/( 1 - amp_l2c) ) 
'''
def MAIN_dgauss(param,RV,args=None):
    ymodel=gen_over_func(param,dgauss,args)
    return ymodel

def dgauss(param,rv_tab,args=None):
 
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






'''
Function returning the measured FWHM and contrast of stellar profiles with custom shape
    - we return the 'true' contrast and FWM on the measured-like model
    - we calculate the model on a HR table 
    - we calculate with the derived model the contrast counted from the estimated continuum (
 + for double-gaussian profiles, the peaks of the lobes are the closest estimate to the true stellar continuum
    - we calculate the FWHM by finding the points on the blue side at half-maximum, and on the red side
'''
def cust_mod_true_prop(param,velccf_HR,args):

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
    nsamp=len(merged_chain_proc[:,0])    
    param_proc = deepcopy(param_loc)
    chain_loc_proc=np.empty([3,0],dtype=float) 
    for istep in range(nsamp): 
        for ipar,par in enumerate(args['var_par_list']):param_proc[par]=merged_chain_proc[istep,ipar]     
        true_ctrst_step,true_FWHM_step,true_amp_step=cust_mod_true_prop(param_proc,args['cen_bins_HR'],args)  
        chain_loc_proc=np.append(chain_loc_proc,[[true_ctrst_step],[true_FWHM_step],[true_amp_step]],axis=1) 
    return chain_loc_proc
    
def para_cust_mod_true_prop(func_input,nthreads,n_elem,y_inputs,common_args):   
    pool_proc = Pool(processes=nthreads)   #cannot be passed through lmfit
    ind_chunk_list=init_parallel_func(nthreads,n_elem)
    chunked_args=[(y_inputs[0][ind_chunk[0]:ind_chunk[1],:],)+common_args for ind_chunk in ind_chunk_list]	
    all_results=tuple(tab for tab in pool_proc.starmap(func_input,chunked_args))				
    y_output=np.concatenate(tuple(all_results[i] for i in range(nthreads)),axis=1)
    pool_proc.close()
    pool_proc.join() 
    return y_output

#Local macroturbulence broadening functions
#    - formulation with anisotropic gaussian from Takeda & UeNo 2017 or with radial-tangential model (Gray 1975, 2005)
def calc_macro_ker_rt(rv_mac_kernel,param,cos_th,sin_th):
    if cos_th>0:term_R=(param['A_R']/(np.sqrt(np.pi)*param['ksi_R']*cos_th))*np.exp(-rv_mac_kernel**2./  ((param['ksi_R']*cos_th)**2.)  )   #Otherwise exp yield 0
    else:term_R=0.
    if sin_th>0:term_T=(param['A_T']/(np.sqrt(np.pi)*param['ksi_T']*sin_th))*np.exp(-rv_mac_kernel**2./  ((param['ksi_T']*sin_th)**2.)  )   #Otherwise exp yield 0 
    else:term_T=0.
    macro_ker_sub=term_R+term_T    
    return macro_ker_sub

def calc_macro_ker_anigauss(rv_mac_kernel,param,cos_th,sin_th):
    macro_ker_sub=np.exp(  -rv_mac_kernel**2./  ((param['eta_R']*cos_th)**2.  +  (param['eta_T']*sin_th)**2.)  )
    return macro_ker_sub











