import numpy as np
from utils import stop


'''
Definition of polynomial coefficients from the parameter format
'''
def polycoeff_def(param,coeff_ord2name_polpar):

    #Polynomial coefficients 
    #    - keys in 'coeff_ord2name_polpar' are the coefficients degrees, values are their names, as defined in 'param' 
    #      they can be defined in disorder (in terms of degrees), as coefficients are forced to order from deg_max to 0 in 'coeff_grid_polpar' 
    #    - degrees can be missing
    #    - input coefficients must be given in decreasing order of degree to poly1d
    deg_max=max(coeff_ord2name_polpar.keys())
    coeff_grid_polpar=[param[coeff_ord2name_polpar[ideg]] if ideg in coeff_ord2name_polpar else 0. for ideg in range(deg_max,-1,-1)]

    return coeff_grid_polpar

#Calculation of 'absolute' or 'modulated' polynomial
#    - 'poly1d' takes coefficient array in decreasing powers
#      'coeff_pol' has been defined in this way, using input coefficient defined through their power value
#    - 'abs' : coeff_pol[0]*x^n + coeff_pol[1]*x^(n-1) .. + coeff_pol[n]*x^0
#      'modul' : (coeff_pol[0]*x^n + coeff_pol[1]*x^(n-1) .. + 1)*coeff_pol[n]
def calc_polymodu(pol_mode,coeff_pol,x_val):
    if pol_mode=='abs':
        mod= np.poly1d(coeff_pol)(x_val)         
    elif pol_mode=='modul':
        coeff_pol_modu = coeff_pol[:-1] + [1]
        mod= coeff_pol[-1]*np.poly1d(coeff_pol_modu)(x_val)  
    else:stop('Undefined polynomial mode')
    return mod

'''
Function returning the relevant coordinate for calculation of line properties variations
'''
def calc_linevar_coord_grid(dim,grid):
    if (dim in ['mu','xp_abs','r_proj','y_st']):linevar_coord_grid = grid[dim]
    elif (dim=='y_st2'):linevar_coord_grid = grid['y_st']**2.
    elif (dim=='abs_y_st'):linevar_coord_grid = np.abs(grid['y_st'])
    else:stop('Undefined line coordinate')
    return linevar_coord_grid

    
    
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

