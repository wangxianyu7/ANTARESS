#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double* C_coadd_loc_gauss_prof(double* rv_surf_star_grid, double* ctrst_grid, double* FWHM_grid, double* args_cen_bins, double* Fsurf_grid_spec, int args_ncen_bins, int Fsurf_grid_spec_shape_0) 
{
    double* gauss = (double*)malloc(Fsurf_grid_spec_shape_0 * args_ncen_bins * sizeof(double));
    double X = 2.0 * sqrt(log(2.0));

    // Make grid of profiles
    for (int i = 0; i < Fsurf_grid_spec_shape_0; i++) {
        double A = Fsurf_grid_spec[i];
        double B = rv_surf_star_grid[i];
        double C = ctrst_grid[i];
        double D = FWHM_grid[i];
        for (int j = 0; j < args_ncen_bins; j++) {
            gauss[i * args_ncen_bins + j] = A * (1.0 - C * exp(-(X * (args_cen_bins[j] - B) / D) * (X * (args_cen_bins[j] - B) / D)));
        }
    }
    return gauss;
}

void free_gaussian_line_grid(double* ptr) {
    free(ptr);
}



