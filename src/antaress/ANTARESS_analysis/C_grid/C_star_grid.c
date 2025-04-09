#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cerf.h>
#include <complex.h>

void C_coadd_loc_gauss_prof(double* rv_surf_star_grid, double* ctrst_grid, double* FWHM_grid, double* args_cen_bins, double* Fsurf_grid_spec, int args_ncen_bins, int Fsurf_grid_spec_shape_0, double* gauss_grid) 
{
    double X = 2.0 * sqrt(log(2.0));

    // Make grid of profiles
    for (int i = 0; i < Fsurf_grid_spec_shape_0; i++) {
        double A = Fsurf_grid_spec[i];
        double B = rv_surf_star_grid[i];
        double C = ctrst_grid[i];
        double D = FWHM_grid[i];
        for (int j = 0; j < args_ncen_bins; j++) {
            gauss_grid[i * args_ncen_bins + j] = A * (1.0 - C * exp(-(X * (args_cen_bins[j] - B) / D) * (X * (args_cen_bins[j] - B) / D)));
        }
    }
}

void C_coadd_loc_cgauss_prof(double* rv_surf_star_grid, double* ctrst_grid, double* FWHM_grid, float skew, float kurt, double* args_cen_bins, double* Fsurf_grid_spec, int args_ncen_bins, int Fsurf_grid_spec_shape_0, double* cgauss_grid) 
{
    double X = 2.0 * sqrt(2.0 * log(2.0));

    double c[5] = {sqrt(6.) / 4., -sqrt(3.), -sqrt(6.), 2. / sqrt(3.), sqrt(6.) / 3.};

    // Make grid of profiles
    for (int i = 0; i < Fsurf_grid_spec_shape_0; i++) {
        double A = Fsurf_grid_spec[i];
        double B = rv_surf_star_grid[i];
        double C = ctrst_grid[i];
        double D = FWHM_grid[i];
        for (int j = 0; j < args_ncen_bins; j++) {

            double G = 1;

            // Apply skewness if present
            if (skew != 0.0) {
                G += skew * (c[1] * (X * (args_cen_bins[j] - B) / D) + c[3] * (X * (args_cen_bins[j] - B) / D) * (X * (args_cen_bins[j] - B) / D) * (X * (args_cen_bins[j] - B) / D));
            }

            // Apply kurtosis if present
            if (kurt != 0.0) {
                G += kurt * (c[0] + c[2] * (X * (args_cen_bins[j] - B) / D) * (X * (args_cen_bins[j] - B) / D) + c[4] * (X * (args_cen_bins[j] - B) / D) * (X * (args_cen_bins[j] - B) / D) * (X * (args_cen_bins[j] - B) / D) * (X * (args_cen_bins[j] - B) / D));
            }

            // Compute the final profile
            cgauss_grid[i * args_ncen_bins + j] = A * (1.0 - G * C * exp(-0.5 * (X * (args_cen_bins[j] - B) / D) * (X * (args_cen_bins[j] - B) / D)));
        }
    }
}

void C_coadd_loc_dgauss_prof(double* rv_surf_star_grid, double* ctrst_grid, double* FWHM_grid, double* rv_l2c_grid, double* FWHM_l2c_grid, double* amp_l2c_grid, double* args_cen_bins, double* Fsurf_grid_spec, int args_ncen_bins, int Fsurf_grid_spec_shape_0, double* dgauss_grid) 
{
    double X = 2.0 * sqrt(log(2.0));

    // Make grid of profiles
    for (int i = 0; i < Fsurf_grid_spec_shape_0; i++) {
        double A = Fsurf_grid_spec[i];
        double B = rv_surf_star_grid[i];
        double C = ctrst_grid[i];
        double D = FWHM_grid[i];
        double E = rv_l2c_grid[i];
        double F = FWHM_l2c_grid[i];
        double G = amp_l2c_grid[i];
        double H = C / (1.0 - G);
        for (int j = 0; j < args_ncen_bins; j++) {
            dgauss_grid[i * args_ncen_bins + j] = A * (1.0 - H * exp(-(X * (args_cen_bins[j] - B) / D) * (X * (args_cen_bins[j] - B) / D)) + H * G * exp(-(X * (args_cen_bins[j] - B - E) / (D * F)) * (X * (args_cen_bins[j] - B - E) / (D * F))));
        }
    }
}

void C_coadd_loc_pgauss_prof(double* rv_surf_star_grid, double* ctrst_grid, double* FWHM_grid, double* c4_pol_grid, double* c6_pol_grid, double* dRV_joint_grid, double* args_cen_bins, double* Fsurf_grid_spec, int args_ncen_bins, int Fsurf_grid_spec_shape_0, double* pgauss_grid) 
{
    
    double X = 2.0 * sqrt(log(2.0));

    for (int i = 0; i < Fsurf_grid_spec_shape_0; i++) {
        double A = Fsurf_grid_spec[i];
        double B = rv_surf_star_grid[i];
        double C = ctrst_grid[i];
        double D = FWHM_grid[i];
        double E = c4_pol_grid[i];
        double F = c6_pol_grid[i];
        double G = dRV_joint_grid[i];

    // Gaussian with baseline set to continuum value
        for (int j = 0; j < args_ncen_bins; j++) {
            pgauss_grid[i * args_ncen_bins + j] = A * (1.0 - C * exp(-(X * (args_cen_bins[j] - B) / D) * (X * (args_cen_bins[j] - B) / D)));
            if ((args_cen_bins[j] >= (B - G)) && (args_cen_bins[j] <= (B + G))) {
                pgauss_grid[i * args_ncen_bins + j] *= (E * G * G * G * G) + (2.0 * F * G * G * G * G * G * G) - (G * G * (args_cen_bins[j] - B) * (args_cen_bins[j] - B) * (2.0 * E + 3.0 * F * G * G)) + (E * (args_cen_bins[j] - B) * (args_cen_bins[j] - B) * (args_cen_bins[j] - B) * (args_cen_bins[j] - B)) + (F * (args_cen_bins[j] - B) * (args_cen_bins[j] - B) * (args_cen_bins[j] - B) * (args_cen_bins[j] - B) * (args_cen_bins[j] - B) * (args_cen_bins[j] - B));
            }
        }
    }
}

void C_coadd_loc_voigt_prof(double* rv_surf_star_grid, double* ctrst_grid, double* FWHM_grid, double* a_damp_grid, double* args_cen_bins, double* Fsurf_grid_spec, int args_ncen_bins, int Fsurf_grid_spec_shape_0, double* voigt_grid) 
{
    
    double X = 2.0 * sqrt(log(2.0));

    for (int i = 0; i < Fsurf_grid_spec_shape_0; i++) {
        double A = Fsurf_grid_spec[i];
        double B = rv_surf_star_grid[i];
        double C = ctrst_grid[i];
        double D = FWHM_grid[i];
        double E = a_damp_grid[i];

    // Gaussian with baseline set to continuum value
        for (int j = 0; j < args_ncen_bins; j++) {
            voigt_grid[i * args_ncen_bins + j] = A * (1.0 - (C/creal(w_of_z(E * I))) * creal(w_of_z((X * (args_cen_bins[j] - B) / D) + E * I)));
        }
    }
}






