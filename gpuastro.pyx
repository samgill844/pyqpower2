from gpuastro cimport clip as c_clip
from gpuastro cimport getEccentricAnomaly as c_getEccentricAnomaly
from gpuastro cimport rv as c_rv
from gpuastro cimport rv_batch as c_rv_batch
from gpuastro cimport get_intensity_from_limb_darkening_law as c_get_intensity_from_limb_darkening_law
#from gpuastro cimport Flux_drop_analytical_power_2 as c_Flux_drop_analytical_power_2
#from gpuastro cimport Flux_drop_analytical_uniform as c_Flux_drop_analytical_uniform
from gpuastro cimport ld_quad_match as c_ld_quad_match
from gpuastro cimport lc as c_lc
from gpuastro cimport dwt as c_dwt
from gpuastro cimport fft as c_fft
from gpuastro cimport lc_loglike as c_lc_loglike
from gpuastro cimport lc_batch_loglike as c_lc_batch_loglike


from gpuastro cimport lc_batch as c_lc_batch
from gpuastro cimport fft_convolve as c_fft_convolve

import ctypes
from cpython cimport array
import numpy as np
cimport numpy as np

__all__ = ['clip', 'getE', 'rv', 'get_intensity_from_limb_darkening_law', 'Flux_drop_analytical_power_2', 'lc_loglike', 'fft_convolve']

#/***********************************************
#			 Helper functions
#***********************************************/
def clip(a,b,c) : return c_clip(a,b,c)



#/***********************************************
#			Keplerian equations
#***********************************************/
def getEccentricAnomaly(M=0.0,e=0.0): 
	'''
	Calculate the eccentric anamoly from the eccentricity and mean anomaly.

	Parameters
	----------
	M : double
	    The mean anomaly in radians.
	e : double
	    The eccentricity of the orbit.

	Returns
	-------
	E : double
		The eccentric anomaly.

	An example use would be:
	>> from gpuastro import getEccentricAnomaly
	>> E = getEccentricAnomaly(0.2,0.4)
	>> print(E)
	[out] 0.32938429848384687
	'''
	return c_getEccentricAnomaly(M,e)




#/***********************************************
#			Radial velocity
#***********************************************/
def rv(np.ndarray[double, ndim=1, mode="c"] time , t_zero=0.0, period=1.0, K1 = 10.0, K2 = 10.0, fs=0.0, fc=0.0, V0 = 0.0, dV0=0.0, nthreads=1, CPUorGPU=0):
	'''
	Calculate the radial velocity for given time measurements.

	Parameters
	----------
	time : numpy array (np.float64)
	    The time axis with a dtype of np.float64.
	t_zero : double
	    The time of central eclipse where star 1 is occulted by star 2.
	period : double
		The period of the system.
	K1 : double
		The semi-amplitude of star 1 [km/s].
	K2 : double
		The semi-amplitude of star 2 [km/s].
	fs : double
		The decorrelation parameter sqrt(e)*sin(w), where "e" is the 
		eccentricity and "w" is the argument of periastron (radians). 
	fc : double
		The decorrelation parameter sqrt(e)*cos(w), where "e" is the 
		eccentricity and "w" is the argument of periastron (radians).
	V0 : double
		The systematic velocity of ther system [km/s].
	dV0 : double
		The drift insystematic velocity of ther system [km/s per day].
	nthreads : int
		The number of threads which OpenMP requests to be used.
	CPUorGPU : int
		Use either the CPU (0) or the GPU (1).

	Returns
	-------
	RV1 : numpy array (np.float64)
		Radial velocity measuremnts for star 1.
	RV2 : numpy array (np.float64)
		Radial velocity measuremnts for star 1.


	An example use would be:
	# Define a time axis
	>> import numpy as np
	>> t = np.linspace(0,1,1000, dtype=np.float64) # remember to conver to doubles
	>>
	>> from gpuastro import rv
	>> RV1, RV2 = rv(t, fs=0.2,fc=0.4)
	>> print(RV1[:5])
	[out] array([1.78884796, 1.70958892, 1.63054226, 1.55171285, 1.47310431])
	>> print(RV2[:5])
	[out] array([1.78885925, 1.86811829, 1.94716733, 2.02599675, 2.10460646])
	'''
	# see https://stackoverflow.com/questions/17014379/cython-cant-convert-python-object-to-double to pass arrays
	#cdef array.array _time = array.array('f', time)
	#	fib.rv(_time.data.as_floats)

	cdef np.ndarray[double, ndim=1, mode="c"] RV1 = np.ascontiguousarray(np.zeros(len(time),dtype=np.float64 ))
	cdef np.ndarray[double, ndim=1, mode="c"] RV2 = np.ascontiguousarray(np.zeros(len(time),dtype=np.float64))

	# Method 2
	# https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC
	# and error with numpy import 
	# https://stackoverflow.com/questions/14657375/cython-fatal-error-numpy-arrayobject-h-no-such-file-or-directory

	c_rv(&time[0], &RV1[0], &RV2[0], len(time), t_zero, period, K1, K2, fs, fc, V0, dV0, nthreads,CPUorGPU)
	return RV1, RV2

def rv_batch(np.ndarray[double, ndim=1, mode="c"] time = np.linspace(0,1,1000,dtype=np.float64), np.ndarray[double, ndim=1, mode="c"] t_zero = np.zeros(2560,dtype=np.float64), np.ndarray[double, ndim=1, mode="c"] period = np.ones(2560,dtype=np.float64), np.ndarray[double, ndim=1, mode="c"] K1 = np.random.normal(1, 1e-1, 2560).astype(np.float64), np.ndarray[double, ndim=1, mode="c"] K2 = np.random.normal(1, 1e-2, 2560).astype(np.float64), np.ndarray[double, ndim=1, mode="c"] fs= np.zeros(2560,dtype=np.float64), np.ndarray[double, ndim=1, mode="c"] fc= np.zeros(2560,dtype=np.float64), np.ndarray[double, ndim=1, mode="c"] V0= np.zeros(2560,dtype=np.float64), np.ndarray[double, ndim=1, mode="c"] dV0= np.zeros(2560,dtype=np.float64), n_batch = 2560):
	'''
	Batch calculate the radial velocity for given time measurements.

	Parameters
	----------
	time : numpy array (np.float64)
	    The time axis with a dtype of np.float64.
	t_zero : numpy array (np.float64)
	    The time of central eclipse where star 1 is occulted by star 2.
	period : numpy array (np.float64)
		The period of the system.
	K1 : numpy array (np.float64)
		The semi-amplitude of star 1 [km/s].
	K2 : numpy array (np.float64)
		The semi-amplitude of star 2 [km/s].
	fs : numpy array (np.float64)
		The decorrelation parameter sqrt(e)*sin(w), where "e" is the 
		eccentricity and "w" is the argument of periastron (radians). 
	fc : numpy array (np.float64)
		The decorrelation parameter sqrt(e)*cos(w), where "e" is the 
		eccentricity and "w" is the argument of periastron (radians).
	V0 : numpy array (np.float64)
		The systematic velocity of ther system [km/s].
	dV0 : numpy array (np.float64)
		The drift insystematic velocity of ther system [km/s per day].
	n_batch : int
		The number of radial velocity models to calculate.

	Returns
	-------
	RV1 : numpy array (np.float64)
		Radial velocity measuremnts for star 1 for each model. The array will
		have shape (n_batch, len(time)).
	RV2 : numpy array (np.float64)
		Radial velocity measuremnts for star 2 for each model. The array will
		have shape (n_batch, len(time)).

	An example use would be:
	# Define a time axis
	>> import numpy as np
	>> t = np.linspace(0,1,1000, dtype=np.float64) # remember to conver to doubles
	>>
	>> from gpuastro_gpu import rv_batch
	>> RV1, RV2 = rv_batch(t)
	>>
	>> print(RV1.shape)
	[out] (2560, 1000)
	>> print(RV2.shape)
	[out] (2560, 1000)
	>>
	>> improt matplotlib.pyplot as plt
	>> for i in range(int(RV1.shape[0]/2)):
    ...:     plt.plot(t, RV1[i], 'k', alpha = 0.01)
    >> plt.xlabel('Phase');plt.ylabel('RV [km/s]')
    >> plt.show()
	'''

	cdef np.ndarray[double, ndim=1, mode="c"] RV1 = np.ascontiguousarray(np.zeros(len(time)*n_batch,dtype=np.float64 ))
	cdef np.ndarray[double, ndim=1, mode="c"] RV2 = np.ascontiguousarray(np.zeros(len(time)*n_batch,dtype=np.float64 ))


	c_rv_batch(&time[0], &RV1[0], &RV2[0], len(time), n_batch, &t_zero[0], &period[0], &K1[0], &K2[0], &fs[0], &fc[0], &V0[0], &dV0[0])

	return RV1.reshape((n_batch, len(time))) ,    RV2.reshape((n_batch, len(time)))



#/***********************************************
#			      Limb-darkening
#***********************************************/
def get_intensity_from_limb_darkening_law(ld_law=0, np.ndarray[double, ndim=1, mode="c"] ldc = np.array([0.6], dtype = np.float64), mu_i=0.5):
	'''
	Calculate the intensity profile across the stellar disk for a given limb-darkening
	law.

	Parameters
	----------
	ld_law : int
	    The limb-darknening law to use. The coiches are:
	    	[0] linear (Schwarzschild (1906, Nachrichten von der Königlichen Gesellschaft der Wissenschaften zu Göttingen. Mathematisch-Physikalische Klasse, p. 43)
			[1] Quadratic Kopal (1950, Harvard Col. Obs. Circ., 454, 1)
			[2] Square-root (Díaz-Cordovés & Giménez, 1992, A&A, 259, 227) 
			[3] Logarithmic (Klinglesmith & Sobieski, 1970, AJ, 75, 175)
			[4] Exponential LD law (Claret & Hauschildt, 2003, A&A, 412, 241)
			[5] Sing three-parameter law (Sing et al., 2009, A&A, 505, 891)
			[6] Claret four-parameter law (Claret, 2000, A&A, 363, 1081)
			[7] Power-2 law (Maxted 2018 in prep)
			[8] Uniform limb darkening
	ldc : numpy array (np.float64)
	    The limb-darkening coeffiiencts. Note this MUST be a double array (np.floaot64) 
	    with number of array elements to match the chose limb-darkening law.
	mu_i : double
		mu_i = cos γ where γ is the angle between a line normal to the stellar surface and the line of sight of the observer.

	Returns
	-------
	I : double
		The intensity of the stellar disk.


	An example use would be:
	# Define a time axis
	>> import numpy as np
	>> t = np.linspace(0,1,1000, dtype=np.float64) # remember to conver to doubles
	>>
	>> from gpuastro import rv
	>> RV1, RV2 = rv(t, fs=0.2,fc=0.4)
	>> print(RV1[:5])
	[out] array([1.78884796, 1.70958892, 1.63054226, 1.55171285, 1.47310431])
	>> print(RV2[:5])
	[out] array([1.78885925, 1.86811829, 1.94716733, 2.02599675, 2.10460646])
	'''
	return c_get_intensity_from_limb_darkening_law(ld_law, &ldc[0], mu_i, 0)



#/***********************************************
#		    Analytical power-2 law
#***********************************************/

def Flux_drop_analytical_power_2(d_radius = 0.0, k = 0.2, c = 0.8, a = 0.8, f=1.0, eps = 1e-8):
	'''
	Calculate thee analytical flux drop por the power-2 law.
	
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

	Returns
		f : double
			The modified flux from a primary transit.

	An example use would be:
	# Define a "time axis" which is actually a seperation axis [in terms of stellar radii].
	>> import numpy as np
	>> t = np.linspace(-2,2, 1000, dtype=np.float64) # remember to conver to doubles
	>>
	>> from gpuastro import Flux_drop_analytical_power_2
	>> lightcurve=[]
	>> for i in np.abs(t) : lc.append(Flux_drop_analytical_power_2(i))
	>>
	>> import matplotlib.pyplot as plt
	>> plt.plot(t, lc)
	>> plt.xlabel('Time')
	>> plt.show()
	'''
	return #c_Flux_drop_analytical_power_2(d_radius, k , c, a, f, eps)

#/***********************************************
#		   Power-2 law coefficient transform
#     			     Maxted 2018
#***********************************************/

def power_2_h_to_coeffs(h1, h2) : return 1-h1+h2, np.log2((1-h1+h2)/h2)
def power_2_coeffs_to_h(c,alpha) : return 1 - c*(1-2**(-alpha)), c*2**(-alpha)



#/***********************************************
#		    Analytical uniform
#***********************************************/

def Flux_drop_analytical_uniform(d_radius=0.0, k=0.2, SBR = 0.0, f=1.0):
	'''
	Calculate thee analytical flux drop for a uniform disk.
	
	Parameters
		d_radius : double
			Projected seperation of centers in units of stellar radii.
		k : double
			Ratio of the radii.
		f : double
			The flux from which to drop light from.
	Returns
		f : double
			The modified flux from a primary transit.

	An example use would be:
	# Define a "time axis" which is actually a seperation axis [in terms of stellar radii].
	>> import numpy as np
	>> t = np.linspace(-2,2, 1000, dtype=np.float64) # remember to conver to doubles
	>>
	>> from gpuastro import Flux_drop_analytical_power_2
	>> lightcurve=[]
	>> for i in np.abs(t) : lc.append(Flux_drop_analytical_power_2(i))
	>>
	>> import matplotlib.pyplot as plt
	>> plt.plot(t, lc)
	>> plt.xlabel('Time')
	>> plt.show()
	'''
	return #c_Flux_drop_analytical_uniform(d_radius, k, SBR, f)





#/***********************************************
#		      Eker quadratic match
#***********************************************/
def ld_quad_match(ld_law = 1, np.ndarray[double, ndim=1, mode="c"] ldc = np.array([0.6], dtype = np.float64)):
	'''
	Match quadratic limb-darkening coefficients used in the Eker spot model
	with any choice in limb-darkening law. 

	Parameters
		ld_law : int
			The limb-darkening law to use
		ldc : numpy array (np.float64)
			The limb-darkening law for the chosen coefficients. 
	Returns
		Quadratic coefficients : numpy array (np.float64)
			A length 2 array containing (u1, -u2)

	An example use would be:
	>> from gpuastro import ld_quad_match
	>> import numpy as np
	>>
	>> ld_quad_match(7, np.array([0.8,0.8], dtype = np.float64))
	[out] array([ 0.56208267, -0.23791733])
	'''

	cdef np.ndarray[double, ndim=1, mode="c"] I_ret = np.ascontiguousarray(np.zeros(len(ldc),dtype=np.float64 ))
	c_ld_quad_match(ld_law, &ldc[0], &I_ret[0])

	return I_ret




#/***********************************************
#    Eker model for spot reduction of flux
#***********************************************/


#/***********************************************
#                   Lightcurve
#***********************************************/
def lc(np.ndarray[double, ndim=1, mode="c"] time, t_zero = 0.0, period = 1.0, radius_1 = 0.2, k = 0.2, fs = 0.0, fc = 0.0, incl = 90.0, SBR = 0.0, ld_law_1 = 7, np.ndarray[double, ndim=1, mode="c"] ldc_1 = np.array([0.8,0.8], dtype = np.float64), ld_law_2 = 7, np.ndarray[double, ndim=1, mode="c"] ldc_2 = np.array([0.8,0.8], dtype = np.float64), third_light = 0.0, nthreads=1 ):

	cdef np.ndarray[double, ndim=1, mode="c"] LC = np.ascontiguousarray(np.zeros(len(time),dtype=np.float64 ))

	c_lc(&time[0], &LC[0], len(time), t_zero, period, radius_1, k, fs, fc, incl, SBR, ld_law_1, &ldc_1[0], ld_law_2, &ldc_2[0], third_light, nthreads)

	return LC

def lc_loglike(np.ndarray[double, ndim=1, mode="c"] time, np.ndarray[double, ndim=1, mode="c"] lightcurve, np.ndarray[double, ndim=1, mode="c"] lightcurve_err, t_zero = 0.0, period = 1.0, radius_1 = 0.2, k = 0.2, fs = 0.0, fc = 0.0, incl = 90.0, SBR = 0.0, ld_law_1 = 7, np.ndarray[double, ndim=1, mode="c"] ldc_1 = np.array([0.8,0.8], dtype = np.float64), ld_law_2 = 7, np.ndarray[double, ndim=1, mode="c"] ldc_2 = np.array([0.8,0.8], dtype = np.float64), third_light = 0.0, nthreads=1 ):
	return c_lc_loglike(&time[0], &lightcurve[0], &lightcurve_err[0], len(time), t_zero, period, radius_1, k, fs, fc, incl, SBR, ld_law_1, &ldc_1[0], ld_law_2, &ldc_2[0], third_light, nthreads)




def lc_batch(np.ndarray[double, ndim=1, mode="c"] time = np.linspace(-0.1,0.1,1000,dtype=np.float64), 
	np.ndarray[double, ndim=1, mode="c"] t_zero = np.zeros(2560*4,dtype=np.float64), 
	np.ndarray[double, ndim=1, mode="c"] period = np.ones(2560*4,dtype=np.float64),
	np.ndarray[double, ndim=1, mode="c"] radius_1 = 0.2*np.ones( 2560*4).astype(np.float64), 
	np.ndarray[double, ndim=1, mode="c"] k = 0.2*np.ones( 2560*4).astype(np.float64), 
	np.ndarray[double, ndim=1, mode="c"] fs= np.zeros(2560*4,dtype=np.float64), 
	np.ndarray[double, ndim=1, mode="c"] fc= np.zeros(2560*4,dtype=np.float64), 
	np.ndarray[double, ndim=1, mode="c"] incl= 90*np.ones(2560*4,dtype=np.float64), 
	np.ndarray[double, ndim=1, mode="c"] SBR= np.zeros(2560*4,dtype=np.float64), 
	ld_law_1 = 7, 
	np.ndarray[double, ndim=1, mode="c"] ldc_1= 0.8*np.ones(2560*2*4,dtype=np.float64), 
	ld_law_2 = 9, 
	np.ndarray[double, ndim=1, mode="c"] ldc_2 = 0.8*np.ones(2560*2*4,dtype=np.float64), 
	np.ndarray[double, ndim=1, mode="c"] third_light = np.zeros(2560*4,dtype=np.float64)):

	cdef np.ndarray[double, ndim=1, mode="c"] LC = np.ascontiguousarray(np.zeros(len(time)*len(t_zero),dtype=np.float64 ))

	if (ld_law_1 == 0)       : offset_1 = 0
	if (ld_law_1 == 1)       : offset_1 = 1
	if (ld_law_1 == 2)       : offset_1 = 1
	if (ld_law_1 == 3)       : offset_1 = 1 
	if (ld_law_1 == 4)       : offset_1 = 1 
	if (ld_law_1 == 5)       : offset_1 = 2 
	if (ld_law_1 == 6)       : offset_1 = 3 
	if (ld_law_1 == 7)       : offset_1 = 1
	else                     : offset_1 = 0

	if (ld_law_2 == 0)       : offset_2 = 0
	if (ld_law_2 == 1)       : offset_2 = 1
	if (ld_law_2 == 2)       : offset_2 = 1
	if (ld_law_2 == 3)       : offset_2 = 1 
	if (ld_law_2 == 4)       : offset_2 = 1 
	if (ld_law_2 == 5)       : offset_2 = 2 
	if (ld_law_2 == 6)       : offset_2 = 3 
	if (ld_law_2 == 7)       : offset_2 = 1
	else                     : offset_2 = 0

	c_lc_batch(&time[0],  &LC[0], len(time), len(t_zero), &t_zero[0],  &period[0], &radius_1[0], &k[0],  &fs[0], &fc[0],  &incl[0],  &SBR[0], ld_law_1, &ldc_1[0],  ld_law_2, &ldc_2[0], &third_light[0],  offset_1, offset_2)

	return LC.reshape((len(t_zero), len(time)))


def lc_batch_loglike( 
	np.ndarray[double, ndim=1, mode="c"] time, 
	np.ndarray[double, ndim=1, mode="c"] flux, 
	np.ndarray[double, ndim=1, mode="c"] flux_err, 

	np.ndarray[double, ndim=1, mode="c"] t_zero = np.zeros(2560*4,dtype=np.float64), 
	np.ndarray[double, ndim=1, mode="c"] period = np.ones(2560*4,dtype=np.float64),
	np.ndarray[double, ndim=1, mode="c"] radius_1 = 0.2*np.ones( 2560*4).astype(np.float64), 
	np.ndarray[double, ndim=1, mode="c"] k = 0.2*np.ones( 2560*4).astype(np.float64), 
	np.ndarray[double, ndim=1, mode="c"] fs= np.zeros(2560*4,dtype=np.float64), 
	np.ndarray[double, ndim=1, mode="c"] fc= np.zeros(2560*4,dtype=np.float64), 
	np.ndarray[double, ndim=1, mode="c"] incl= 90*np.ones(2560*4,dtype=np.float64), 
	np.ndarray[double, ndim=1, mode="c"] SBR= np.zeros(2560*4,dtype=np.float64), 
	ld_law_1 = 7, 
	np.ndarray[double, ndim=1, mode="c"] ldc_1= 0.8*np.ones(2560*2*4,dtype=np.float64), 
	ld_law_2 = 9, 
	np.ndarray[double, ndim=1, mode="c"] ldc_2 = 0.8*np.ones(2560*2*4,dtype=np.float64), 
	np.ndarray[double, ndim=1, mode="c"] third_light = np.zeros(2560*4,dtype=np.float64)):

	cdef np.ndarray[double, ndim=1, mode="c"] LC_loglike = np.ascontiguousarray(np.zeros(len(t_zero),dtype=np.float64 ))

	if (ld_law_1 == 0)       : offset_1 = 0
	if (ld_law_1 == 1)       : offset_1 = 1
	if (ld_law_1 == 2)       : offset_1 = 1
	if (ld_law_1 == 3)       : offset_1 = 1 
	if (ld_law_1 == 4)       : offset_1 = 1 
	if (ld_law_1 == 5)       : offset_1 = 2 
	if (ld_law_1 == 6)       : offset_1 = 3 
	if (ld_law_1 == 7)       : offset_1 = 1
	else                     : offset_1 = 0

	if (ld_law_2 == 0)       : offset_2 = 0
	if (ld_law_2 == 1)       : offset_2 = 1
	if (ld_law_2 == 2)       : offset_2 = 1
	if (ld_law_2 == 3)       : offset_2 = 1 
	if (ld_law_2 == 4)       : offset_2 = 1 
	if (ld_law_2 == 5)       : offset_2 = 2 
	if (ld_law_2 == 6)       : offset_2 = 3 
	if (ld_law_2 == 7)       : offset_2 = 1
	else                     : offset_2 = 0

	c_lc_batch_loglike(&time[0], &flux[0], &flux_err[0], &LC_loglike[0], len(time), len(t_zero), &t_zero[0],  &period[0], &radius_1[0], &k[0],  &fs[0], &fc[0],  &incl[0],  &SBR[0], ld_law_1, &ldc_1[0],  ld_law_2, &ldc_2[0], &third_light[0],  offset_1, offset_2)

	return LC_loglike





#/***********************************************
#                  Helpers
#***********************************************/

def get_all_dics():
	return dict(t_zero = 0.0,
				period = 1.0,

				radius_1=0.2,
				k = 0.2,
				incl = 0.0,
				SBR = 0.0,
				third_light = 0.0,

				fs = 0.0,
				fc = 0.0,
				K1 = 10.0,
				K2 = 10.0,
				V0 = 0.0,
				dV0 = 0.0,

				ld_law_1 = 7,
				ld_law_2 = 9,
				ldc_1 = np.array([0.8,0.8], dtype=np.float64),
				ldc_2 = np.array([0.8,0.8], dtype=np.float64),

				h1 = 0.6, h2 = 0.4,

				nthreads=1), dict(), dict()



def dwt(np.ndarray[double, ndim=1, mode="c"] data):
	cdef np.ndarray[double, ndim=1, mode="c"] data_= data.copy().astype(np.float64)
	c_dwt(&data_[0], len(data_))
	return data_

def fft(np.ndarray[double, ndim=1, mode="c"] data):
	cdef np.ndarray[double, ndim=1, mode="c"] data_= data.copy().astype(np.float64)
	c_fft(&data_[0], len(data_))
	return data_


def fft_convolve(np.ndarray[double, ndim=1, mode="c"] data, np.ndarray[double, ndim=1, mode="c"] kernel):
	cdef np.ndarray[double, ndim=1, mode="c"] data_= data.copy().astype(np.float64)
	c_fft_convolve(&data_[0], len(data_), &kernel[0], len(kernel))
	return data_



def TEST1():
	import ellc
	import numpy as np
	import matplotlib.pyplot as plt


	radius_1 = 0.2
	k = 0.2
	radius_2 = radius_1*k
	SBR = 0.25
	incl = 90.0

	ld_law_1 = 7
	ldc_1 = np.array([0.6], dtype = np.float64)
	ld_law_2 = 7
	ldc_2 = np.array([0.6], dtype = np.float64)



	# Define time
	t = np.linspace(-0.2,0.8, 2000, dtype=np.float64)

	# Now get ellc
	flux_ellc = ellc.lc(t,radius_1=radius_1,radius_2=radius_2,sbratio=SBR, incl=incl,q=0.001,ld_1='lin',ldc_1=ldc_1,ld_2='lin',ldc_2=ldc_2)	
	plt.plot(t, flux_ellc, 'r')

	flux_gpuastro = lc(t, radius_1 = radius_1, k=k, SBR=SBR, incl = incl, ld_law_1 = 0, ldc_1 = ldc_1, ld_law_2 = 0, ldc_2 = ldc_2)
	plt.plot(t, flux_gpuastro, 'b')

	plt.figure()
	plt.plot(t, 1e6*(flux_ellc - flux_gpuastro))

	plt.show()



def ellc_test_primary():
	from ellc import lc as elc
	import matplotlib.pyplot as plt
	import numpy as np
	import matplotlib
	matplotlib.rcParams.update({'font.size': 30})

	plt.rc('font', family='serif')
	plt.rc('xtick', labelsize='x-small')
	plt.rc('ytick', labelsize='x-small')

	fig = plt.figure(figsize=(4, 3))
	ax = fig.add_subplot(1, 1, 1)

	p = 0.1
	r_star = 0.1
	c = 0.85
	a = 0.80
	mu = np.linspace(0,1,5001)
	I_0 = (a+2)/(np.pi*(a-c*a+2))
	r = np.sqrt(1-mu**2)
	mugrid =  I_0*(1 - c + c*(1-r**2)**(a/2))

	t14 = np.arcsin(r_star*np.sqrt( ((1+p)**2)  ))/np.pi
	t = np.linspace(-0.51*t14,0.51*t14,1001)
	fellc = elc(t, radius_1=r_star, radius_2=p*r_star,sbratio=0, incl=90,
	       ld_1='mugrid', ldc_1=mugrid,
	       grid_1='very_fine', grid_2='very_fine')

	F_gpu_astro_power2_integral = lc(t, t_zero=0.0, period=1.0, radius_1 = r_star, k = p, fs=0, fc=0, incl=90, ld_law_1=7, ldc_1=np.array([c,a],dtype=np.float64))
	plt.plot(t, (F_gpu_astro_power2_integral-fellc)*1e6, 'k', label='Power-2')
	plt.xlabel(r'Time')
	plt.ylabel(r'O-C [ppm]')

	fig = plt.figure(figsize=(4, 3))
	ax = fig.add_subplot(1, 1, 1)
	plt.plot(t, fellc, 'r')
	plt.plot(t, F_gpu_astro_power2_integral, 'b')

def ellc_test_secondary():
	from ellc import lc as elc
	import matplotlib.pyplot as plt
	import numpy as np

	p = 0.1
	r_star = 0.1
	c = 0.85
	a = 0.80
	mu = np.linspace(0,1,5001)
	I_0 = (a+2)/(np.pi*(a-c*a+2))
	r = np.sqrt(1-mu**2)
	mugrid =  I_0*(1 - c + c*(1-r**2)**(a/2))

	t14 = np.arcsin(r_star*np.sqrt( ((1+p)**2)  ))/np.pi
	t = np.linspace(-0.51*t14,0.51*t14,1001) + 0.5
	fellc = elc(t, radius_1=r_star, radius_2=p*r_star,sbratio=0.5, incl=90,
	       ld_1='mugrid', ldc_1=mugrid,
	       grid_1='very_fine', grid_2='very_fine', ld_2 = None, ldc_2 = 0.6)

	F_gpu_astro_power2_integral = lc(t, t_zero=0.0, period=1.0, radius_1 = r_star, k = p, fs=0, fc=0, incl=90, ld_law_1=7, ldc_1=np.array([c,a],dtype=np.float64), SBR = 0.5, ld_law_2 = 9, ldc_2 = np.array([0.6], dtype=np.float64))

	plt.plot(t, (F_gpu_astro_power2_integral-fellc)*1e6, label='Power-2')
	plt.xlabel(r'Time')
	plt.ylabel(r'O-C [ppm]')


	plt.figure()
	plt.plot(t, fellc, 'r')
	plt.plot(t, F_gpu_astro_power2_integral, 'b')



def ellc_test_primary():
	from ellc import lc as elc
	import matplotlib.pyplot as plt
	import numpy as np

	p = 0.1
	r_star = 0.1
	c = 0.85
	a = 0.80
	mu = np.linspace(0,1,5001)
	I_0 = (a+2)/(np.pi*(a-c*a+2))
	r = np.sqrt(1-mu**2)
	mugrid =  I_0*(1 - c + c*(1-r**2)**(a/2))

	t14 = np.arcsin(r_star*np.sqrt( ((1+p)**2)  ))/np.pi
	t = np.linspace(-0.51*t14,0.51*t14,1001)
	fellc = elc(t, radius_1=r_star, radius_2=p*r_star,sbratio=0, incl=90,
	       ld_1='mugrid', ldc_1=mugrid,
	       grid_1='very_fine', grid_2='very_fine')

	F_gpu_astro_power2_integral = lc(t, t_zero=0.0, period=1.0, radius_1 = r_star, k = p, fs=0, fc=0, incl=90, ld_law_1=7, ldc_1=np.array([c,a],dtype=np.float64))
	plt.plot(t, (F_gpu_astro_power2_integral-fellc)*1e6, label='Power-2')
	plt.xlabel(r'Time')
	plt.ylabel(r'O-C [ppm]')

	plt.figure()
	plt.plot(t, fellc, 'r')
	plt.plot(t, F_gpu_astro_power2_integral, 'b')


def ellc_test_primary_lin():
	from ellc import lc as elc
	import matplotlib.pyplot as plt
	import numpy as np

	p = 0.1
	r_star = 0.1

	t14 = np.arcsin(r_star*np.sqrt( ((1+p)**2)  ))/np.pi

	t = np.linspace(-0.51*t14,0.51*t14,1001)
	fellc = elc(t, radius_1=r_star, radius_2=p*r_star,sbratio=0, incl=90,
	       ld_1='lin', ldc_1=0.6,
	       grid_1='very_fine', grid_2='very_fine')

	F_gpu_astro_power2_integral = lc(t, t_zero=0.0, period=1.0, radius_1 = r_star, k = p, fs=0, fc=0, incl=90, ld_law_1=0, ldc_1=np.array([0.6],dtype=np.float64))

	plt.figure()
	plt.plot(t, fellc, 'r')
	plt.plot(t, F_gpu_astro_power2_integral, 'b')
	plt.figure()
	plt.plot(t, fellc-F_gpu_astro_power2_integral, 'r')


def test_batch_test():
	import matplotlib.pyplot as plt
	import numpy as np

	t = np.linspace(-0.05, 0.05, 1000, dtype=np.float64)

	LC = lc_batch(t)

	for i in LC : plt.plot(t,i, 'k', alpha=0.03)
	plt.xlabel('Phase')
	plt.ylabel('Flux')
	plt.show()
