/***********************************************
			 Helper functions
***********************************************/
double clip(double a, double b, double c);


/***********************************************
			Keplerian equations
***********************************************/
double getEccentricAnomaly(double M, double e);

/***********************************************
			Radial velocity
***********************************************/
void rv(double * time, double * RV1, double * RV2, int n_elements, double t_zero, double period, double K1, double K2, double fs, double fc, double V0, double dV0, int nthreads, int CPUorGPU);
void rv_batch(double * time, double * RV1, double * RV2, int  n_elements, int n_batch, double * t_zero, double * period, double * K1, double * K2, double * fs, double * fc, double * V0, double * dV0);


/***********************************************
			      Limb-darkening
***********************************************/
double get_intensity_from_limb_darkening_law (int ld_law, double * ldc, double mu_i, int offset);

/***********************************************
		    Analytical power-2 law
***********************************************/

double Flux_drop_analytical_power_2(double d_radius, double k, double c, double a, double f, double eps);


/***********************************************
		    Analytical uniform
***********************************************/

double Flux_drop_analytical_uniform(double d_radius, double k, double SBR, double f);

/***********************************************
		      Eker quadratic match
***********************************************/
void ld_quad_match(int ld_law, double * ldc, double * I_ret);


/***********************************************
    Eker model for spot reduction of flux
***********************************************/



/***********************************************
                   Lightcurve
***********************************************/
void lc(double * time, double * LC, int n_elements, double t_zero, double period, double radius_1, double k, double fs, double fc, double incl, double SBR, int ld_law_1, double * ldc_1, int ld_law_2, double * ldc_2, double third_light, int nthreads);
double lc_loglike(double * time, double * lightcurve, double * lightcurve_err, int n_elements, double t_zero, double period, double radius_1, double k, double fs, double fc, double incl, double SBR, int ld_law_1, double * ldc_1, int ld_law_2, double * ldc_2, double third_light, int nthreads);


void d_lc_batch(double * time, double * LC, int n_elements, int n_batch, double * t_zero, double * period, double * radius_1, double * k, double * fs, double * fc, double * incl, double * SBR, int ld_law_1, double * ldc_1, int ld_law_2, double * ldc_2, double * third_light, int offset_1, int offset_2);
void lc_batch(double * time, double * LC, int n_elements, int n_batch, double * t_zero, double * period, double * radius_1, double * k, double * fs, double * fc, double * incl, double * SBR, int ld_law_1, double * ldc_1, int ld_law_2, double * ldc_2, double * third_light, int offset_1, int offset_2);

void lc_batch_loglike(double * time, double * flux, double * flux_err, double * LC_loglike, int n_elements, int n_batch, double * t_zero, double * period, double * radius_1, double * k, double * fs, double * fc, double * incl, double * SBR, int ld_law_1, double * ldc_1, int ld_law_2, double * ldc_2, double * third_light, int offset_1, int offset_2);

void dwt(double * data, int n);
void fft(double * data, int n);
void fft_convolve(double * data, int n_data, double * kernel, int n_kernel);




double Flux_drop_annulus(double d_radius, double k, double SBR, double f, int ld_law, double * ldc, int n_annulus, int primary, int offset);