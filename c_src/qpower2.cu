/*
   Copyright (C) 2018  Samuel Gill, Keele University

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.

   Some parts of this code, particulary relating to solving Keplers
   equations are extracted from the BATMAN exoplanet transit model.
   This code is subject to the following copyright:
			   ----------------------------------
			   Copyright (C) 2015 Laura Kreidberg
			   ----------------------------------
   Functions subject to this copyright include, but not limited to:
		double area()
		double Flux_drop_analytical_uniform()
		void lc() **
	** The lc function uses batman formulaism to solve Keplers equations. 
*/


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION 
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_wavelet.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>


#define M_PI 3.14159265358979323846


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



/***********************************************
			 Make definitions to use
***********************************************/
// First, check for openMP support
#if defined (_OPENMP)
#  include <omp.h>
#endif



/***********************************************
			 Helper functions
***********************************************/
#ifdef __CUDACC__
__host__ __device__ 
#endif
double clip(double a, double b, double c)
{
	if (a < b)
		return b;
	else if (a > c)
		return c;
	else
		return a;
}

#ifdef __CUDACC__
__host__ __device__ 
#endif
double area(double z, double r1, double r2)
{
	//
	// Returns area of overlapping circles with radii x and R; separated by a distance d
	//

	double arg1 = clip((z*z + r1*r1 - r2*r2)/(2.*z*r1),-1,1);
	double arg2 = clip((z*z + r2*r2 - r1*r1)/(2.*z*r2),-1,1);
	double arg3 = clip(max((-z + r1 + r2)*(z + r1 - r2)*(z - r1 + r2)*(z + r1 + r2), 0.),-1,1);

	if   (r1 <= r2 - z) return M_PI*r1*r1;							                              // planet completely overlaps stellar circle
	else if (r1 >= r2 + z) return M_PI*r2*r2;						                                  // stellar circle completely overlaps planet
	else                return r1*r1*acosf(arg1) + r2*r2*acosf(arg2) - 0.5*sqrtf(arg3);          // partial overlap
}



/***********************************************
			Keplerian equations
***********************************************/
/*
double getEccentricAnomaly(double M, double e)	//calculates the eccentric anomaly (see Seager Exoplanets book:  Murray & Correia eqn. 5 -- see section 3)
{
	double E = M, eps = 1.0e-5;
	double fe, fs;

	// modification from LK 05/07/2017:
	// add fmod to ensure convergence for diabolical inputs (following Eastman et al. 2013; Section 3.1)
	while(fmod(fabs(E - e*sin(E) - M), 2.*M_PI) > eps)
	{
		fe = fmodf(E - e*sinf(E) - M, 2.*M_PI);
		fs = fmodf(1 - e*cosf(E), 2.*M_PI);
		E = E - fe/fs;
	}
	return E;
}
*/
#ifdef __CUDACC__
__host__ __device__ 
#endif
double getEccentricAnomaly(double M, double e)	//calculates the eccentric anomaly (see Seager Exoplanets book:  Murray & Correia eqn. 5 -- see section 3)
{
	if (e == 0.0)
		return M;

	double m = fmodf(M , (2*M_PI));
	int flip;
	if (m > M_PI)
	{
		m = 2*M_PI - m;
		flip = 1;
	}
	else
		flip = 0;

	double alpha = (3*M_PI + 1.6*(M_PI-fabsf(m))/(1+e) )/(M_PI - 6/M_PI);
	double d = 3*(1 - e) + alpha*e;
	double r = 3*alpha*d * (d-1+e)*m + m*m*m;
	double q = 2*alpha*d*(1-e) - m*m;
	double w = powf((fabsf(r) + sqrtf(q*q*q + r*r)),(2/3));
	double E = (2*r*w/(w*w + w*q + q*q) + m) / d;
	double f_0 = E - e*sinf(E) - m;
	double f_1 = 1 - e*cosf(E);
	double f_2 = e*sinf(E);
	double f_3 = 1-f_1;
	double d_3 = -f_0/(f_1 - 0.5*f_0*f_2/f_1);
	double d_4 = -f_0/(f_1 + 0.5*d_3*f_2 + (d_3*d_3*d_3)*f_3/6);
	E = E -f_0/(f_1 + 0.5*d_4*f_2 + d_4*d_4*f_3/6 - d_4*d_4*d_4*f_2/24);
	if (flip==1)
		E =  2*M_PI - E;
	return E;
}



/***********************************************
			Radial velocity
***********************************************/

__global__ void d_rv(double * d_time, double * d_RV1, double * d_RV2, int n_elements, double t_zero, double period, double K1, double K2, double e, double w, double V0, double dV0, double n )
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<n_elements)
	{
			double nu = M_PI/2. - w;                                                   //true anomaly corresponding to time of primary transit center
			double E = 2.*atanf(sqrtf((1. - e)/(1. + e))*tanf(nu/2.));				    //corresponding eccentric anomaly
			double M = E - e*sin(E);                                                   // Mean anomaly
			double tp = t_zero - period*M/2./M_PI;							            //time of periastron


			if(e < 1.0e-5)
			{
				nu = ((d_time[i] - tp)/period - (int)(((d_time[i] - tp)/period)))*2.*M_PI;			// calculates true anomaly for a circular orbit
			}
			else
			{
				M = n*(d_time[i] - tp);
				E = getEccentricAnomaly(M, e);
				nu = 2.*atanf(sqrtf((1.+e)/(1.-e))*tanf(E/2.));                                 // calculates true anomaly for a eccentric orbit
			}


			if(e < 1.0e-5)
			{
				d_RV1[i] = K1*cosf(nu + w)                              + V0  + dV0*(d_time[i] - t_zero);
				d_RV2[i] = K2*cosf(nu + w + M_PI)                       + V0  + dV0*(d_time[i] - t_zero);
			}
			else
			{
				d_RV1[i] = K1*( e*cosf(w) + cosf(nu + w ))              + V0  + dV0*(d_time[i] - t_zero);
				d_RV2[i] = K2*( e*cosf(w) + cosf(nu + w + M_PI))        + V0  + dV0*(d_time[i] - t_zero);
			}
	}
}

void rv(double * time, double * RV1, double * RV2, int n_elements, double t_zero, double period, double K1, double K2, double fs, double fc, double V0, double dV0, int nthreads, int CPUorGPU)
{

	// Set the number of threads if openMP is setup
	#if defined (_OPENMP) && CPUorGPU==0
	omp_set_num_threads(nthreads);	//specifies number of threads (if OpenMP is supported)
	#endif

	// Unpack argumant of periastron "w" and eccentricity "e"
	double w;
	if (fc != 0 && fs != 0)
		w = atan2(fs,fc);
	else
		w=0.0;
	double e = fs*fs + fc*fc; // orbital eccentricity


	// Definitions for keplerian solution
	double nu, M, E, tp;
	double n = 2.*M_PI/period;	// mean motion

	// Now begin the main loop
	if (CPUorGPU==0)
	{
		////////////////////////////////////////////
		// If we are here, we will run on the CPU //
		////////////////////////////////////////////
		int i;
		#if defined (_OPENMP)
		#pragma omp parallel for
		#endif
		for (i=0; i < n_elements; i++)
		{
			nu = M_PI/2. - w;                                                   //true anomaly corresponding to time of primary transit center
			E = 2.*atanf(sqrtf((1. - e)/(1. + e))*tanf(nu/2.));				    //corresponding eccentric anomaly
			M = E - e*sin(E);                                                   // Mean anomaly
			tp = t_zero - period*M/2./M_PI;							            //time of periastron


			if(e < 1.0e-5)
			{
				nu = ((time[i] - tp)/period - (int)(((time[i] - tp)/period)))*2.*M_PI;			// calculates true anomaly for a circular orbit
			}
			else
			{
				M = n*(time[i] - tp);
				E = getEccentricAnomaly(M, e);
				nu = 2.*atanf(sqrtf((1.+e)/(1.-e))*tanf(E/2.));                                 // calculates true anomaly for a eccentric orbit
			}


			if(e < 1.0e-5)
			{
				RV1[i] = K1*cosf(nu + w)              + V0  + dV0*(time[i] - t_zero);
				RV2[i] = K2*cosf(nu + w + M_PI)       + V0  + dV0*(time[i] - t_zero);
			}
			else
			{
				RV1[i] = K1*( e*cosf(w) + cosf(nu + w ))              + V0  + dV0*(time[i] - t_zero);
				RV2[i] = K2*( e*cosf(w) + cosf(nu + w + M_PI))        + V0  + dV0*(time[i] - t_zero);
			}
		}
	}
	else if (CPUorGPU==1)
	{
		////////////////////////////////////////////
		// If we are here, we will run on the GPU //
		////////////////////////////////////////////

		// First create the pointers
		double * d_time, *d_RV1, *d_RV2;

		// Now allocate the device storage
		cudaMalloc(&d_time, n_elements*sizeof(double)); 
		cudaMalloc(&d_RV1,  n_elements*sizeof(double)); 
		cudaMalloc(&d_RV2,  n_elements*sizeof(double)); 

		// Now copy the time axis to the device
		cudaMemcpy(d_time, time, n_elements*sizeof(double), cudaMemcpyHostToDevice);
	
		// Now execute the kernel
		d_rv<<<ceil(n_elements / 512.0), 512>>>(d_time, d_RV1, d_RV2, n_elements, t_zero, period, K1, K2, e, w, V0, dV0, n );


		// Now copy RV1 and RV2 back to host
		cudaMemcpy(RV1, d_RV1, n_elements*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(RV2, d_RV2, n_elements*sizeof(double), cudaMemcpyDeviceToHost);

		// Now free up memory on the device
		cudaFree(d_time);
		cudaFree(d_RV1);
		cudaFree(d_RV2);

	}
	else 
		printf("\nCPUorGPU has been set incorrectly. ");
}





__global__ void d_rv_batch(double * time, double * RV1, double * RV2, int  n_elements, int n_batch, double * t_zero, double * period, double * K1, double * K2, double * fs, double * fc, double * V0, double * dV0)
{

	int j = blockDim.x * blockIdx.x + threadIdx.x;
	if (j<n_batch)
	{


		// Unpack argumant of periastron "w" and eccentricity "e"
		double w;
		if (fc[j] != 0 && fs[j] != 0)
			w = atan2(fs[j],fc[j]);
		else
			w=0.0;
		double e = fs[j]*fs[j] + fc[j]*fc[j]; // orbital eccentricity



		// Definitions for keplerian solution
		double nu, M, E, tp;
		double n = 2.*M_PI/period[j];	// mean motion

		// Now begin the main loop
		int i;
		for (i=0; i < n_elements; i++)
		{
			nu = M_PI/2. - w;                                                   //true anomaly corresponding to time of primary transit center
			E = 2.*atanf(sqrtf((1. - e)/(1. + e))*tanf(nu/2.));				    //corresponding eccentric anomaly
			M = E - e*sinf(E);                                                   // Mean anomaly
			tp = t_zero[j] - period[j]*M/2./M_PI;							            //time of periastron


			if(e < 1.0e-5)
			{
				nu = ((time[i] - tp)/period[j] - (int)(((time[i] - tp)/period[j])))*2.*M_PI;			// calculates true anomaly for a circular orbit
			}
			else
			{
				M = n*(time[i] - tp);
				E = getEccentricAnomaly(M, e);
				nu = 2.*atanf(sqrtf((1.+e)/(1.-e))*tanf(E/2.));                                 // calculates true anomaly for a eccentric orbit
			}


			if(e < 1.0e-5)
			{
				RV1[j*n_elements + i] = K1[j]*cosf(nu + w)              + V0[j]  + dV0[j]*(time[i] - t_zero[j]);
				RV2[j*n_elements + i] = K2[j]*cosf(nu + w + M_PI)       + V0[j]  + dV0[j]*(time[i] - t_zero[j]);
			}
			else
			{
				RV1[j*n_elements + i] = K1[j]*( e*cosf(w) + cosf(nu + w ))              + V0[j]  + dV0[j]*(time[i] - t_zero[j]);
				RV2[j*n_elements + i] = K2[j]*( e*cosf(w) + cosf(nu + w + M_PI))        + V0[j]  + dV0[j]*(time[i] - t_zero[j]);
			}
			//printf("\nModel %d  : time %.6f   RV1 : %.6f   RV2 : %.6f   nu = %.6f   E = %.6f tp = %.6f  P = %.6f", j, time[i], RV1[n_elements*j + i], RV2[n_elements*j + i], nu, E, tp, period[j]);
		}
	}
}


void rv_batch(double * time, double * RV1, double * RV2, int  n_elements, int n_batch, double * t_zero, double * period, double * K1, double * K2, double * fs, double * fc, double * V0, double * dV0)
{
	// DEVICE PART
	double *d_time, *d_RV1, *d_RV2, *d_t_zero, *d_period, *d_K1, *d_K2, *d_fs, *d_fc, *d_V0, *d_dV0;

	// Malloc the arrays
	cudaMalloc(&d_time, n_elements*sizeof(double)); 
	cudaMalloc(&d_RV1, n_elements*n_batch*sizeof(double)); 
	cudaMalloc(&d_RV2, n_elements*n_batch*sizeof(double)); 
	cudaMalloc(&d_t_zero, n_batch*sizeof(double)); 
	cudaMalloc(&d_period, n_batch*sizeof(double)); 
	cudaMalloc(&d_K1, n_batch*sizeof(double)); 
	cudaMalloc(&d_K2, n_batch*sizeof(double)); 
	cudaMalloc(&d_fs, n_batch*sizeof(double)); 
	cudaMalloc(&d_fc, n_batch*sizeof(double)); 
	cudaMalloc(&d_V0, n_batch*sizeof(double)); 
	cudaMalloc(&d_dV0, n_batch*sizeof(double)); 

	// Copy data to gpu
	cudaMemcpy(d_time,   time,   n_elements*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_t_zero, t_zero, n_batch*sizeof(double),   cudaMemcpyHostToDevice);
	cudaMemcpy(d_period, period, n_batch*sizeof(double),   cudaMemcpyHostToDevice);
	cudaMemcpy(d_K1,     K1,     n_batch*sizeof(double),   cudaMemcpyHostToDevice);
	cudaMemcpy(d_K2,     K2,     n_batch*sizeof(double),   cudaMemcpyHostToDevice);
	cudaMemcpy(d_fs,     fs,     n_batch*sizeof(double),   cudaMemcpyHostToDevice);
	cudaMemcpy(d_fc,     fc,     n_batch*sizeof(double),   cudaMemcpyHostToDevice);
	cudaMemcpy(d_V0,     V0,     n_batch*sizeof(double),   cudaMemcpyHostToDevice);
	cudaMemcpy(d_dV0,    dV0,    n_batch*sizeof(double),   cudaMemcpyHostToDevice);

	d_rv_batch<<<ceil(n_batch / 512.0), 512>>>(d_time, d_RV1, d_RV2, n_elements, n_batch, d_t_zero, d_period, d_K1, d_K2, d_fs, d_fc, d_V0, d_dV0);

	// Now copy RV data back to host
	cudaMemcpy(RV1, d_RV1, n_elements*n_batch*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(RV2, d_RV2, n_elements*n_batch*sizeof(double), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();
	
	// Now free everything
	cudaFree(d_time);
	cudaFree(d_RV1);
	cudaFree(d_RV2);
	cudaFree(d_t_zero);
	cudaFree(d_period);
	cudaFree(d_K1);
	cudaFree(d_K2);
	cudaFree(d_fs);
	cudaFree(d_fc);
	cudaFree(d_V0);
	cudaFree(d_dV0);	
}



/***********************************************
			      Limb-darkening
***********************************************/
#ifdef __CUDACC__
__host__ __device__ 
#endif
double get_intensity_from_limb_darkening_law (int ld_law, double * ldc, double mu_i, int offset)
{
	/*
	Calculte limb-darkening for a variety of laws e.t.c.
	[0] linear (Schwarzschild (1906, Nachrichten von der Königlichen Gesellschaft der Wissenschaften zu Göttingen. Mathematisch-Physikalische Klasse, p. 43)
	[1] Quadratic Kopal (1950, Harvard Col. Obs. Circ., 454, 1)
	[2] Square-root (Díaz-Cordovés & Giménez, 1992, A&A, 259, 227) 
	[3] Logarithmic (Klinglesmith & Sobieski, 1970, AJ, 75, 175)
	[4] Exponential LD law (Claret & Hauschildt, 2003, A&A, 412, 241)
	[5] Sing three-parameter law (Sing et al., 2009, A&A, 505, 891)
	[6] Claret four-parameter law (Claret, 2000, A&A, 363, 1081)
	[7] Power-2 law (Maxted 2018 in prep)
	*/
	if (ld_law == 0) 
		return 1 - ldc[offset]*(1 - mu_i);
	if (ld_law == 1) 
		return 1 - ldc[offset]*(1 - mu_i) - ldc[offset+1] * powf((1 - mu_i),2)  ;
	if (ld_law == 2) 
		return 1 -  ldc[offset]*(1 - mu_i) - ldc[offset+1]*(1 - powf(mu_i,2) ) ;
	if (ld_law == 3) 
		return 1 -  ldc[offset]*(1 - mu_i) - ldc[offset+1]*mu_i*logf(mu_i); 
	if (ld_law == 4) 
		return 1 -  ldc[offset]*(1 - mu_i) - ldc[offset+1]/(1-expf(mu_i));  
	if (ld_law == 5) 
		return 1 -  ldc[offset]*(1 - mu_i) - ldc[offset+1]*(1 - powf(mu_i,1.5)) - ldc[offset+2]*(1 - powf(mu_i,2));
	if (ld_law == 6) 
		return 1 - ldc[offset]*(1 - powf(mu_i,0.5)) -  ldc[offset+1]*(1 - mu_i) - ldc[offset+2]*(1 - powf(mu_i,1.5))  - ldc[offset+3]*(1 - powf(mu_i,2));
	if (ld_law == 7) 
		return 1 - ldc[offset]*(1 - powf(mu_i,ldc[offset+1]));	
	else
		return 1 - ldc[offset]*(1 - mu_i);
}



/***********************************************
		    Analytical power-2 law
***********************************************/
#ifdef __CUDACC__
__host__ __device__ 
#endif
double q1(double z, double p, double c, double a, double g, double I_0)
{
	double zt = clip(abs(z), 0,1-p);
	double s = 1-zt*zt;
	double c0 = (1-c+c*pow(s,g));
	double c2 = 0.5*a*c*pow(s,(g-2))*((a-1)*zt*zt-1);
	return 1-I_0*M_PI*p*p*(c0 + 0.25*p*p*c2 - 0.125*a*c*p*p*pow(s,(g-1)));
}


#ifdef __CUDACC__
__host__ __device__ 
#endif
double q2(double z, double p, double c, double a, double g, double I_0, double eps)
{
	double zt = clip(abs(z), 1-p,1+p);
	double d = clip((zt*zt - p*p + 1)/(2*zt),0,1);
	double ra = 0.5*(zt-p+d);
	double rb = 0.5*(1+d);
	double sa = clip(1-ra*ra,eps,1);
	double sb = clip(1-rb*rb,eps,1);
	double q = clip((zt-d)/p,-1,1);
	double w2 = p*p-(d-zt)*(d-zt);
	double w = sqrt(clip(w2,eps,1));
	double c0 = 1 - c + c*pow(sa,g);
	double c1 = -a*c*ra*pow(sa,(g-1));
	double c2 = 0.5*a*c*pow(sa,(g-2))*((a-1)*ra*ra-1);
	double a0 = c0 + c1*(zt-ra) + c2*(zt-ra)*(zt-ra);
	double a1 = c1+2*c2*(zt-ra);
	double aq = acos(q);
	double J1 =  (a0*(d-zt)-(2./3.)*a1*w2 + 0.25*c2*(d-zt)*(2.0*(d-zt)*(d-zt)-p*p))*w + (a0*p*p + 0.25*c2*pow(p,4))*aq ;
	double J2 = a*c*pow(sa,(g-1))*pow(p,4)*(0.125*aq + (1./12.)*q*(q*q-2.5)*sqrt(clip(1-q*q,0.0,1.0)) );
	double d0 = 1 - c + c*pow(sb,g);
	double d1 = -a*c*rb*pow(sb,(g-1));
	double K1 = (d0-rb*d1)*acos(d) + ((rb*d+(2./3.)*(1-d*d))*d1 - d*d0)*sqrt(clip(1-d*d,0.0,1.0));
	double K2 = (1/3)*c*a*pow(sb,(g+0.5))*(1-d);
	return 1 - I_0*(J1 - J2 + K1 - K2);
}





#ifdef __CUDACC__
__host__ __device__ 
#endif
double Flux_drop_analytical_power_2(double d_radius, double k, double c, double a, double f, double eps)
{
	/*
	Calculate the analytical flux drop por the power-2 law.
	
	Parameters
	d_radius : double
		Projected seperation of centers in units of stellar radii.
	k : double
		Ratio of the radii.
	c : double
		The first power-2 coefficient.
	a : double
		The second power-2 coefficient.
	f : double
		The flux from which to drop light from.
	eps : double
		Factor (1e-9)
	*/
	double I_0 = (a+2)/(M_PI*(a-c*a+2));
	double g = 0.5*a;

	if (d_radius < 1-k) return q1(d_radius, k, c, a, g, I_0);
	else if (abs(d_radius-1) < k) return q2(d_radius, k, c, a, g, I_0, 1e-9);
	else return 1.0;
}







/***********************************************
		    Analytical uniform
***********************************************/
#ifdef __CUDACC__
__host__ __device__ 
#endif
double Flux_drop_analytical_uniform(double d_radius, double k, double SBR, double f)
{
		
		if(d_radius >= 1. + k)
			return f;		//no overlap
		if(d_radius >= 1. && d_radius <= k - 1.) 
			return 0.0;     //total eclipse of the star
		else if(d_radius <= 1. - k) 
		{
			if (SBR !=-99) return f - SBR*k*k;	//planet is fully in transit
			else  return f - k*k;	//planet is fully in transit
		}
		else						//planet is crossing the limb
		{
			double kap1=acos(fmin((1. - k*k + d_radius*d_radius)/2./d_radius, 1.));
			double kap0=acos(fmin((k*k + d_radius*d_radius - 1.)/2./k/d_radius, 1.));
			if (SBR != -99) return f - SBR*  (k*k*kap0 + kap1 - 0.5*sqrt(fmax(4.*d_radius*d_radius - powf(1. + d_radius*d_radius - k*k, 2.), 0.)))/M_PI;
			else
				return f - (k*k*kap0 + kap1 - 0.5*sqrt(fmax(4.*d_radius*d_radius - powf(1. + d_radius*d_radius - k*k, 2.), 0.)))/M_PI;

		}
}






/***********************************************
		      Eker quadratic match
***********************************************/
#ifdef __CUDACC__
__host__ __device__ 
#endif
void ld_quad_match(int ld_law, double * ldc, double * I_ret)
{
	// Set coefficients of a quadratic limb darkening law so that the intensity
	// profile matches at mu= 0, 0.5, 1.0
	// N.B.  these are the coefficients on the quadratic limb darkening law as used
	// in eker, i.e.,  I_0[1 - u_1.mu + u_2.mu^2], so u_2 is the negative of the
	// normal quadratic limb darkening coefficient.

	if (ld_law==-99)
	{
		I_ret[0] = 0.0;
		I_ret[1] = 0.0;
	}
	else if (ld_law==0)
	{
		I_ret[0] = ldc[0];
		I_ret[1] = 0.0;
	}
	else if (ld_law==1)
	{
		I_ret[0] = ldc[0];
		I_ret[1] = -ldc[1];		
	}
	else
	{
		double x0 = get_intensity_from_limb_darkening_law(ld_law,  ldc, 0.0, 0.0);
		double x1 = get_intensity_from_limb_darkening_law(ld_law,  ldc, 0.5, 0.0);
		I_ret[0] = 3.0 - 4.0*x1 + x0;
		I_ret[1] = -4.0*x1 + 2.0*x0 + 2.0;	
	}
}



/***********************************************
    Eker model for spot reduction of flux
***********************************************/

/***********************************************
             Annulus integration
***********************************************/
#ifdef __CUDACC__
__host__ __device__ 
#endif
double Flux_drop_annulus(double d_radius, double k, double SBR, double f, int ld_law, double * ldc, int n_annulus, int primary, int offset)
{

	double dr = 1.0/n_annulus;

	int ss;
	double r_ss, mu_ss, ra, rb, I_ss, F_ss, fp,A_ra_rc , A_rb_rc, A_annuli_covered, A_annuli, Flux_total, Flux_occulted;
	Flux_total = 0.0;
	Flux_occulted = 0.0;

	for (ss=0; ss < n_annulus;ss++)
	{
		// Calculate r_ss
		r_ss = (ss + 0.5)/n_annulus;

		ra = r_ss + 0.5/n_annulus;
		rb = r_ss - 0.5/n_annulus;

		// Calculate mu_ss
		mu_ss = sqrt(1 - r_ss*r_ss);

		// Calculate upper (ra) and lower extent (rb)
		if (primary==0)
		{
			// Get intensity from ss
			I_ss = get_intensity_from_limb_darkening_law(ld_law, ldc, mu_ss, offset);

			// Get flux at mu_ss
			F_ss = I_ss*2*M_PI*r_ss*dr;

			if ((ra + k) < d_radius) fp = 0;
			else if (rb >= (d_radius-k) & ra <= (d_radius + k))
			{
				// Calculate intersection between the circle of star 2
				// and the outer radius of the annuli, (ra)
				A_ra_rc = area(d_radius, k, ra);

				// Calculate intersection between the circle of star 2
				// and the inner radius of the annuli, (rb)
				A_rb_rc = area(d_radius, k, rb);

				// So now the area of the of the anuuli covered by star 2 
				// is the difference between these two areas...
				A_annuli_covered = A_ra_rc - A_rb_rc;

				// Great, now we need the area of the annuli itself...
				A_annuli = M_PI*(ra*ra - rb*rb);

				// So now we can calculate fp, 
				fp = A_annuli_covered/A_annuli;	
			}
			else
				fp = 0.0;

		}
		else
		{
			// Get intensity at mu_ss
			I_ss = get_intensity_from_limb_darkening_law(ld_law, ldc, mu_ss,offset);

			// Get Flux at mu_ss
			F_ss = I_ss*2*M_PI*r_ss*dr;

			if   ((d_radius + k) <= 1.0)  fp = 1;   // all the flux from star 2 is occulted as the 
												    // annulus sits behind star 1
			else if ((d_radius - k) >= 1.0)  fp = 0;  // All the flux from star 2 is visible
			else if ((d_radius + ra) <= 1.0)  fp = 1; // check that the annulus is not entirely behind star 1
			else if ((d_radius - ra) >= 1.0)  fp = 0; // check that the annulus is not entirely outside star 1
			else
			{
				// Calculate intersection between the circle of star 2
				// and the outer radius of the annuli, (ra)
				A_ra_rc = area(d_radius, 1.0, ra*k);

				// Calculate intersection between the circle of star 2
				// and the inner radius of the annuli, (rb)
				A_rb_rc = area(d_radius, 1.0, rb*k);


				// So now the area of the of the anuuli covered by star 2 
				// is the difference between these two areas...
				A_annuli_covered = A_ra_rc - A_rb_rc;

				// Great, now we need the area of the annuli itself...
				A_annuli = M_PI*((ra*k)*(ra*k) - (rb*k)*(rb*k));

				// So now we can calculate fp, 
				fp = A_annuli_covered/A_annuli;
			}

		}

		// Now we can calculate the occulted flux...
		Flux_total =  Flux_total + F_ss;
		Flux_occulted =  Flux_occulted + F_ss*fp;
	}


	if (primary==0) return f - Flux_occulted/Flux_total;
	else
		return f - k*k*SBR*Flux_occulted/Flux_total;

}

/***********************************************
                   Lightcurve
***********************************************/
void lc(double * time, double * LC, int n_elements, double t_zero, double period, double radius_1, double k, double fs, double fc, double incl, double SBR, int ld_law_1, double * ldc_1, int ld_law_2, double * ldc_2, double third_light, int nthreads)
{
	// Set the number of threads if openMP is setup
	//#ifdef _OPENMP 
	omp_set_num_threads(nthreads);	//specifies number of threads (if OpenMP is supported)
	//#endif

	// Unpack argumant of periastron "w" and eccentricity "e"
	double w;
	if (fc != 0 && fs != 0)
		w = atan2(fs,fc);
	else
		w=0.0;
	double e = fs*fs + fc*fc; // orbital eccentricity

	// Change inclination to radians
	incl = M_PI*incl/180;

	// Definitions for keplerian solution
	double nu, M, E, tp;
	double n = 2.*M_PI/period;	// mean motion

	// Definition for transit parameters
	double d_radius;

	// Now begin the main loop
	int i;
	double f;

	#if defined (_OPENMP)
	#pragma omp parallel for schedule(static,8)
	#endif
	for (i=0; i < n_elements; i++)
	{

		nu = M_PI/2. - w;                                                   //true anomaly corresponding to time of primary transit center
		E = 2.*atanf(sqrtf((1. - e)/(1. + e))*tanf(nu/2.));				    //corresponding eccentric anomaly
		M = E - e*sin(E);                                                   // Mean anomaly
		tp = t_zero - period*M/2./M_PI;							            //time of periastron


		if(e < 1.0e-5)
		{
			nu = ((time[i] - tp)/period - (int)(((time[i] - tp)/period)))*2.*M_PI;			// calculates true anomaly for a circular orbit
		}
		else
		{
			M = n*(time[i] - tp);
			E = getEccentricAnomaly(M, e);
			nu = 2.*atanf(sqrtf((1.+e)/(1.-e))*tanf(E/2.));                                 // calculates true anomaly for a eccentric orbit
		}

		// Place holder for flux, this will eventually be replacess with a flux drop rom spots
		LC[i] = 1.0;

		// Now calculate the projected seperation of centers
		d_radius = (1-e*e) * sqrt( 1.0 - sinf(incl)*sinf(incl)  *  sinf(nu + w)*sinf(nu + w)) / (1 + e*sinf(nu)) /radius_1;


		// At this point, we might check if the distance between
		if (d_radius > (1.0+ k)) continue;

		// Now test if star 2 is in front of star 1 (f>0) or vice0versal (f<0)
		f = sinf(nu + w)*sinf(incl);

		if (f>0)
		{
			// Primary eclipse
			// First check for analytical expressions
			if (ld_law_1==7)      LC[i] =  Flux_drop_analytical_power_2(d_radius, k, ldc_1[0], ldc_1[1], 1.0, 1e-8); // analytical expression for the power-2 law
			else if (ld_law_1==9) LC[i] =  Flux_drop_analytical_uniform(d_radius, k, -99, 1.0); // analytical expression for a secondary transit 
			else                  LC[i] = Flux_drop_annulus(d_radius, k, 1.0, 1.0, ld_law_1, ldc_1, 4000, 0, 0);

		}
		else if (SBR>0)
		{
			// Secondary eclipse
			// First check analytical expressions
			if (ld_law_2==9)     LC[i] =  Flux_drop_analytical_uniform(d_radius, k, SBR, 1.0); // analytical expression for a secondary transit 
			else                 LC[i] = Flux_drop_annulus(d_radius, k, SBR, 1.0, ld_law_2, ldc_2, 4000, 1, 0);

		}

		// Now account for third light
		LC[i] = LC[i]/(1 + third_light) + (1-1.0/(1 + third_light));
	}
}


double lc_loglike(double * time, double * lightcurve, double * lightcurve_err, int n_elements, double t_zero, double period, double radius_1, double k, double fs, double fc, double incl, double SBR, int ld_law_1, double * ldc_1, int ld_law_2, double * ldc_2, double third_light, int nthreads)
{
	// Set the number of threads if openMP is setup
	//#ifdef _OPENMP 
	omp_set_num_threads(nthreads);	//specifies number of threads (if OpenMP is supported)
	//#endif

	// Unpack argumant of periastron "w" and eccentricity "e"
	double w;
	if (fc != 0 && fs != 0)
		w = atan2(fs,fc);
	else
		w=0.0;
	double e = fs*fs + fc*fc; // orbital eccentricity

	// Change inclination to radians
	incl = M_PI*incl/180;

	// Definitions for keplerian solution
	double nu, M, E, tp;
	double n = 2.*M_PI/period;	// mean motion

	// Definition for transit parameters
	double d_radius;

	// Now begin the main loop
	int i;
	double f, LC;
	double loglike = 0.0;


	for (i=0; i < n_elements; i++)
	{

		nu = M_PI/2. - w;                                                   //true anomaly corresponding to time of primary transit center
		E = 2.*atanf(sqrtf((1. - e)/(1. + e))*tanf(nu/2.));				    //corresponding eccentric anomaly
		M = E - e*sin(E);                                                   // Mean anomaly
		tp = t_zero - period*M/2./M_PI;							            //time of periastron


		if(e < 1.0e-5)
		{
			nu = ((time[i] - tp)/period - (int)(((time[i] - tp)/period)))*2.*M_PI;			// calculates true anomaly for a circular orbit
		}
		else
		{
			M = n*(time[i] - tp);
			E = getEccentricAnomaly(M, e);
			nu = 2.*atanf(sqrtf((1.+e)/(1.-e))*tanf(E/2.));                                 // calculates true anomaly for a eccentric orbit
		}

		// Place holder for flux, this will eventually be replacess with a flux drop rom spots
		LC = 1.0;

		// Now calculate the projected seperation of centers
		d_radius = (1-e*e) * sqrt( 1.0 - sinf(incl)*sinf(incl)  *  sinf(nu + w)*sinf(nu + w)) / (1 + e*sinf(nu)) /radius_1;


		// At this point, we might check if the distance between
		if (d_radius > (1.0+ k)) continue;

		// Now test if star 2 is in front of star 1 (f>0) or vice0versal (f<0)
		f = sinf(nu + w)*sinf(incl);

		if (f>0)
		{
			// Primary eclipse
			// First check for analytical expressions
			if (ld_law_1==7)      LC =  Flux_drop_analytical_power_2(d_radius, k, ldc_1[0], ldc_1[1], 1.0, 1e-8); // analytical expression for the power-2 law
			else if (ld_law_1==9) LC =  Flux_drop_analytical_uniform(d_radius, k, -99, 1.0); // analytical expression for a secondary transit 
			else                  LC = Flux_drop_annulus(d_radius, k, 1.0, 1.0, ld_law_1, ldc_1, 4000, 0, 0);

		}
		else if (SBR>0)
		{
			// Secondary eclipse
			// First check analytical expressions
			if (ld_law_2==9)     LC =  Flux_drop_analytical_uniform(d_radius, k, SBR, 1.0); // analytical expression for a secondary transit 
			else                 LC = Flux_drop_annulus(d_radius, k, SBR, 1.0, ld_law_2, ldc_2, 4000, 1, 0);

		}

		// Now account for third light
		LC = LC/(1 + third_light) + (1-1.0/(1 + third_light));

		// Now add to loklike
		loglike = loglike -0.5*pow(lightcurve[i] - LC,2) / pow(lightcurve_err[i],2);
	}

	// Now return loglike
	return loglike;
}




__global__ void d_lc_batch(double * time, double * LC, int n_elements, int n_batch, double * t_zero, double * period, double * radius_1, double * k, double * fs, double * fc, double * incl, double * SBR, int ld_law_1, double * ldc_1, int ld_law_2, double * ldc_2, double * third_light, int offset_1, int offset_2)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	if (j<n_batch)
	{

		// Unpack argumant of periastron "w" and eccentricity "e"
		double w;
		if (fc[j] != 0 && fs[j] != 0)
			w = atan2(fs[j],fc[j]);
		else
			w=0.0;
		double e = fs[j]*fs[j] + fc[j]*fc[j]; // orbital eccentricity

		// Change inclination to radians
		double incl_ = M_PI*incl[j]/180;

		// Definitions for keplerian solution
		double nu, M, E, tp;
		double n = 2.*M_PI/period[j];	// mean motion

		// Definition for transit parameters
		double d_radius;

		// Now begin the main loop
		int i;
		double f;

		for (i=0; i < n_elements; i++)
		{
			nu = M_PI/2. - w;                                                   //true anomaly corresponding to time of primary transit center
			E = 2.*atanf(sqrtf((1. - e)/(1. + e))*tanf(nu/2.));				    //corresponding eccentric anomaly
			M = E - e*sin(E);                                                   // Mean anomaly
			tp = t_zero[j] - period[j]*M/2./M_PI;							            //time of periastron


			if(e < 1.0e-5)
			{
				nu = ((time[i] - tp)/period[j] - (int)(((time[i] - tp)/period[j])))*2.*M_PI;			// calculates true anomaly for a circular orbit
			}
			else
			{
				M = n*(time[i] - tp);
				E = getEccentricAnomaly(M, e);
				nu = 2.*atanf(sqrtf((1.+e)/(1.-e))*tanf(E/2.));                                 // calculates true anomaly for a eccentric orbit
			}


			// Place holder for flux, this will eventually be replacess with a flux drop rom spots
			LC[j*n_elements + i] = 1.0;

			// Now calculate the projected seperation of centers
			d_radius = (1-e*e) * sqrt( 1.0 - sinf(incl_)*sinf(incl_)  *  sinf(nu + w)*sinf(nu + w)) / (1 + e*sinf(nu)) /radius_1[j];

			// At this point, we might check if the distance between
			if (d_radius > (1.0+ k[j])) continue;

			// Now test if star 2 is in front of star 1 (f>0) or vice0versal (f<0)
			f = sinf(nu + w)*sinf(incl_);

		

			if (f>0)
			{
				// Primary eclipse
				// First check for analytical expressions
				if (ld_law_1==7)      LC[j*n_elements + i] =  Flux_drop_analytical_power_2(d_radius, k[j], ldc_1[j*offset_1], ldc_1[j*offset_1+1], 1.0, 1e-8); // analytical expression for the power-2 law
				else if (ld_law_1==9) LC[j*n_elements + i] =  Flux_drop_analytical_uniform(d_radius, k[j], -99, 1.0); // analytical expression for a secondary transit 
				else                  LC[j*n_elements + i] = Flux_drop_annulus(d_radius, k[j], 1.0, 1.0, ld_law_1, ldc_1, 4000, 0, j*offset_1);

			}
			else if (SBR[j]>0) 
			{
				// Secondary eclipse
				// First check analytical expressions
				if (ld_law_2==9)     LC[j*n_elements + i] =  Flux_drop_analytical_uniform(d_radius, k[j], SBR[j], 1.0); // analytical expression for a secondary transit 
				else                 LC[j*n_elements + i] = Flux_drop_annulus(d_radius, k[j], SBR[j], 1.0, ld_law_2, ldc_2, 4000, 1, j*offset_2);

			}
				

			// Now account for third light
			//LC[j*n_elements + i] =  LC[j*n_elements + i]/(1 + third_light[j]) + (1-1.0/(1 + third_light[j]));
			
		}
	}
}



void lc_batch(double * time, double * LC, int n_elements, int n_batch, double * t_zero, double * period, double * radius_1, double * k, double * fs, double * fc, double * incl, double * SBR, int ld_law_1, double * ldc_1, int ld_law_2, double * ldc_2, double * third_light, int offset_1, int offset_2)
{

	// DEVICE PART
	double *d_time, *d_LC, *d_t_zero, *d_period, *d_radius_1, *d_k, *d_fs, *d_fc, *d_incl, *d_SBR, *d_ldc_1, *d_ldc_2, *d_third_light;

	// Malloc the arrays
	cudaMalloc(&d_time,        n_elements*sizeof(double)); 
	cudaMalloc(&d_LC,          n_elements*n_batch*sizeof(double)); 
	cudaMalloc(&d_t_zero,      n_batch*sizeof(double)); 
	cudaMalloc(&d_period,      n_batch*sizeof(double)); 
	cudaMalloc(&d_radius_1,    n_batch*sizeof(double)); 
	cudaMalloc(&d_k,           n_batch*sizeof(double)); 
	cudaMalloc(&d_fs,          n_batch*sizeof(double)); 
	cudaMalloc(&d_fc,          n_batch*sizeof(double)); 
	cudaMalloc(&d_incl,        n_batch*sizeof(double)); 
	cudaMalloc(&d_SBR,         n_batch*sizeof(double)); 
	cudaMalloc(&d_third_light, n_batch*sizeof(double)); 

	cudaMalloc(&d_ldc_1,       (offset_1+1)*n_batch*sizeof(double)); 
	cudaMalloc(&d_ldc_2,       (offset_2+1)*n_batch*sizeof(double)); 


	// Copy data to gpu
	cudaMemcpy(d_time,        time,        n_elements*sizeof(double),         cudaMemcpyHostToDevice);
	cudaMemcpy(d_t_zero,      t_zero,      n_batch*sizeof(double),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_period,      period,      n_batch*sizeof(double),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_radius_1,    radius_1,    n_batch*sizeof(double),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_k,           k,           n_batch*sizeof(double),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_fs,          fs,          n_batch*sizeof(double),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_fc,          fc,          n_batch*sizeof(double),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_incl,        incl,        n_batch*sizeof(double),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_SBR,         SBR,         n_batch*sizeof(double),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_third_light, third_light, n_batch*sizeof(double),            cudaMemcpyHostToDevice);

	cudaMemcpy(d_ldc_1,       ldc_1,       n_batch*(offset_1+1)*sizeof(double),   cudaMemcpyHostToDevice);
	cudaMemcpy(d_ldc_2,       ldc_2,       n_batch*(offset_2+1)*sizeof(double),   cudaMemcpyHostToDevice);


	d_lc_batch<<<ceil(n_batch / 256.0), 256>>>(d_time, d_LC, n_elements, n_batch, d_t_zero, d_period, d_radius_1, d_k, d_fs, d_fc, d_incl, d_SBR, ld_law_1, d_ldc_1, ld_law_2, d_ldc_2, d_third_light, offset_1, offset_2);

    // check if cuda kernel executed correctly
    gpuErrchk(cudaPeekAtLastError())

    // make sure kernel execution has ended
    gpuErrchk(cudaDeviceSynchronize())

	// Now copy LC data back to host

	cudaMemcpy(LC, d_LC, n_elements*n_batch*sizeof(double), cudaMemcpyDeviceToHost);


	// Now free everything
	cudaFree(d_time);
	cudaFree(d_LC);
	cudaFree(d_t_zero);
	cudaFree(d_period);
	cudaFree(d_radius_1);
	cudaFree(d_k);
	cudaFree(d_fs);
	cudaFree(d_fc);
	cudaFree(d_incl);
	cudaFree(d_SBR);	
	cudaFree(d_ldc_1);	
	cudaFree(d_ldc_2);	
	cudaFree(d_third_light);	
}





__global__ void d_lc_batch_loglike(double * time, double * flux, double * flux_err, double * LC_loglike, int n_elements, int n_batch, double * t_zero, double * period, double * radius_1, double * k, double * fs, double * fc, double * incl, double * SBR, int ld_law_1, double * ldc_1, int ld_law_2, double * ldc_2, double * third_light, int offset_1, int offset_2)
{
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	if (j<n_batch)
	{

		// Unpack argumant of periastron "w" and eccentricity "e"
		double w;
		if (fc[j] != 0 && fs[j] != 0)
			w = atan2(fs[j],fc[j]);
		else
			w=0.0;
		double e = fs[j]*fs[j] + fc[j]*fc[j]; // orbital eccentricity

		// Change inclination to radians
		double incl_ = M_PI*incl[j]/180;

		// Definitions for keplerian solution
		double nu, M, E, tp;
		double n = 2.*M_PI/period[j];	// mean motion

		// Definition for transit parameters
		double d_radius;

		// Now begin the main loop
		int i;
		double f, flux_point;

		for (i=0; i < n_elements; i++)
		{
			nu = M_PI/2. - w;                                                   //true anomaly corresponding to time of primary transit center
			E = 2.*atanf(sqrtf((1. - e)/(1. + e))*tanf(nu/2.));				    //corresponding eccentric anomaly
			M = E - e*sin(E);                                                   // Mean anomaly
			tp = t_zero[j] - period[j]*M/2./M_PI;							            //time of periastron


			if(e < 1.0e-5)
			{
				nu = ((time[i] - tp)/period[j] - (int)(((time[i] - tp)/period[j])))*2.*M_PI;			// calculates true anomaly for a circular orbit
			}
			else
			{
				M = n*(time[i] - tp);
				E = getEccentricAnomaly(M, e);
				nu = 2.*atanf(sqrtf((1.+e)/(1.-e))*tanf(E/2.));                                 // calculates true anomaly for a eccentric orbit
			}


			// Place holder for flux, this will eventually be replacess with a flux drop rom spots

			// Now calculate the projected seperation of centers
			d_radius = (1-e*e) * sqrt( 1.0 - sinf(incl_)*sinf(incl_)  *  sinf(nu + w)*sinf(nu + w)) / (1 + e*sinf(nu)) /radius_1[j];

			// At this point, we might check if the distance between
			if (d_radius > (1.0+ k[j])) continue;

			// Now test if star 2 is in front of star 1 (f>0) or vice0versal (f<0)
			f = sinf(nu + w)*sinf(incl_);

		

			if (f>0)
			{
				// Primary eclipse
				// First check for analytical expressions
				if (ld_law_1==7)      flux_point =  Flux_drop_analytical_power_2(d_radius, k[j], ldc_1[j*offset_1], ldc_1[j*offset_1+1], 1.0, 1e-8); // analytical expression for the power-2 law
				else if (ld_law_1==9) flux_point =  Flux_drop_analytical_uniform(d_radius, k[j], -99, 1.0); // analytical expression for a secondary transit 
				else                  flux_point = Flux_drop_annulus(d_radius, k[j], 1.0, 1.0, ld_law_1, ldc_1, 4000, 0, j*offset_1);

			}
			else if (SBR[j]>0) 
			{
				// Secondary eclipse
				// First check analytical expressions
				if (ld_law_2==9)     flux_point =  Flux_drop_analytical_uniform(d_radius, k[j], SBR[j], 1.0); // analytical expression for a secondary transit 
				else                 flux_point = Flux_drop_annulus(d_radius, k[j], SBR[j], 1.0, ld_law_2, ldc_2, 4000, 1, j*offset_2);
			}	
			
			LC_loglike[j] -= 0.5*powf(flux_point - flux[i], 2)/powf(flux_err[i], 2);
			//printf("\nFlux_point = %f", flux_point);

		}
	}


}




void lc_batch_loglike(double * time, double * flux, double * flux_err, double * LC_loglike, int n_elements, int n_batch, double * t_zero, double * period, double * radius_1, double * k, double * fs, double * fc, double * incl, double * SBR, int ld_law_1, double * ldc_1, int ld_law_2, double * ldc_2, double * third_light, int offset_1, int offset_2)
{

	// DEVICE PART
	double *d_time, *d_flux, *d_flux_err, *d_LC_loglike, *d_t_zero, *d_period, *d_radius_1, *d_k, *d_fs, *d_fc, *d_incl, *d_SBR, *d_ldc_1, *d_ldc_2, *d_third_light;

	// Malloc the arrays
	cudaMalloc(&d_time,        n_elements*sizeof(double)); 
	cudaMalloc(&d_flux,        n_elements*sizeof(double)); 
	cudaMalloc(&d_flux_err,    n_elements*sizeof(double)); 
	cudaMalloc(&d_LC_loglike,  n_batch*sizeof(double)); 
	cudaMalloc(&d_t_zero,      n_batch*sizeof(double)); 
	cudaMalloc(&d_period,      n_batch*sizeof(double)); 
	cudaMalloc(&d_radius_1,    n_batch*sizeof(double)); 
	cudaMalloc(&d_k,           n_batch*sizeof(double)); 
	cudaMalloc(&d_fs,          n_batch*sizeof(double)); 
	cudaMalloc(&d_fc,          n_batch*sizeof(double)); 
	cudaMalloc(&d_incl,        n_batch*sizeof(double)); 
	cudaMalloc(&d_SBR,         n_batch*sizeof(double)); 
	cudaMalloc(&d_third_light, n_batch*sizeof(double)); 

	cudaMalloc(&d_ldc_1,       (offset_1+1)*n_batch*sizeof(double)); 
	cudaMalloc(&d_ldc_2,       (offset_2+1)*n_batch*sizeof(double)); 


	// Copy data to gpu
	cudaMemcpy(d_time,        time,        n_elements*sizeof(double),         cudaMemcpyHostToDevice);
	cudaMemcpy(d_flux,        flux,        n_elements*sizeof(double),         cudaMemcpyHostToDevice);
	cudaMemcpy(d_flux_err,    flux_err,    n_elements*sizeof(double),         cudaMemcpyHostToDevice);
	cudaMemcpy(d_t_zero,      t_zero,      n_batch*sizeof(double),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_period,      period,      n_batch*sizeof(double),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_radius_1,    radius_1,    n_batch*sizeof(double),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_k,           k,           n_batch*sizeof(double),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_fs,          fs,          n_batch*sizeof(double),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_fc,          fc,          n_batch*sizeof(double),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_incl,        incl,        n_batch*sizeof(double),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_SBR,         SBR,         n_batch*sizeof(double),            cudaMemcpyHostToDevice);
	cudaMemcpy(d_third_light, third_light, n_batch*sizeof(double),            cudaMemcpyHostToDevice);

	cudaMemcpy(d_ldc_1,       ldc_1,       n_batch*(offset_1+1)*sizeof(double),   cudaMemcpyHostToDevice);
	cudaMemcpy(d_ldc_2,       ldc_2,       n_batch*(offset_2+1)*sizeof(double),   cudaMemcpyHostToDevice);


	d_lc_batch_loglike<<<ceil(n_batch / 256.0), 256>>>(d_time, d_flux, d_flux_err, d_LC_loglike, n_elements, n_batch, d_t_zero, d_period, d_radius_1, d_k, d_fs, d_fc, d_incl, d_SBR, ld_law_1, d_ldc_1, ld_law_2, d_ldc_2, d_third_light, offset_1, offset_2 );

	//d_lc_batch<<<ceil(n_batch / 256.0), 256>>>(d_time, d_flux, d_flux_err, d_LC_loglike, n_elements, n_batch, d_t_zero, d_period, d_radius_1, d_k, d_fs, d_fc, d_incl, d_SBR, ld_law_1, d_ldc_1, ld_law_2, d_ldc_2, d_third_light, offset_1, offset_2);

    // check if cuda kernel executed correctly
    gpuErrchk(cudaPeekAtLastError())

    // make sure kernel execution has ended
    gpuErrchk(cudaDeviceSynchronize())

	// Now copy LC data back to host
	cudaMemcpy(LC_loglike, d_LC_loglike, n_batch*sizeof(double), cudaMemcpyDeviceToHost);


	// Now free everything
	cudaFree(d_time);
	cudaFree(d_flux);
	cudaFree(d_flux_err);
	cudaFree(d_LC_loglike);
	cudaFree(d_t_zero);
	cudaFree(d_period);
	cudaFree(d_radius_1);
	cudaFree(d_k);
	cudaFree(d_fs);
	cudaFree(d_fc);
	cudaFree(d_incl);
	cudaFree(d_SBR);	
	cudaFree(d_ldc_1);	
	cudaFree(d_ldc_2);	
	cudaFree(d_third_light);	
}





void dwt(double * data, int n)
{
	gsl_wavelet *w;
	gsl_wavelet_workspace *work;

	w = gsl_wavelet_alloc (gsl_wavelet_daubechies, 4);
	work = gsl_wavelet_workspace_alloc (n);	
	
	gsl_wavelet_transform_forward(w, data, 1, n, work);

	gsl_wavelet_free (w);
	gsl_wavelet_workspace_free (work);
}

/*
int main()
{
	double data[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

	dwt(data, 16);
	return 0;
}
*/

void fft(double * data, int n)
{
	gsl_fft_real_wavetable * real;
	//gsl_fft_halfcomplex_wavetable * hc;
	gsl_fft_real_workspace * work;

	work = gsl_fft_real_workspace_alloc (n);
	real = gsl_fft_real_wavetable_alloc (n);

	gsl_fft_real_transform (data, 1, n, real, work);

	gsl_fft_real_wavetable_free (real);
	//gsl_fft_halfcomplex_wavetable_free (hc);
	gsl_fft_real_workspace_free (work);

}







