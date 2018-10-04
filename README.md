# qpower2

A blazingly fast lightcurve model which works with OpenMP and CUDA. 

## Requirements
1. The cuda toolkit, which can be downloaded
   and installed here: https://developer.nvidia.com/cuda-toolkit
       
   Note: you don't have to install the graphics driver, just the toolkit!
         NVCC works even without a GPU, as is required for this project. 
         
         
         
## Installation
Installation should be as simple as 
```python
python setup.py install
```
qpower2 autamatically looks for openMP and nvcc to compile eveything. However,
this was built and tested under Ubuntu 18.04, but should work just fine under 
Mac. 

## Example use

In this example, we will demonstrate the power of qpower2 to generate lightcurve
models. First, start with the imports,
```python
import qpower2, numpy as np, matplotlib.pyplot as plt, emcee, corner
```
Now lets mock a K2 campaign lightcurve with a 30 minute cadence, slight eccentricity
and 8.3 day period
```python
t = np.arange(0,80,1/48)
flux = np.random.normal(qpower2.lc(t, fs=0.1,fc=0.1, period=8.3),0.001)
plt.plot(t,flux);plt.xlabel('Time [d]'); plt.ylabel('Flux')
plt.show()
```
![alt text](https://github.com/samgill844/pyqpower2/blob/master/images/Figure_1.png)


Alright, now we obviously want to generate some models to fit this. First lets time it
to see how many models we can do per second:

```python
%timeit qpower2.lc(t, fs = 0.1, fc = 0.1, period = 8.3)
915 µs ± 7.52 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```
Not bad, but we can do beter if we have openMP. We can specify how many processors 
to use with the "nthreads" argument...
```python
%timeit qpower2.lc(t, fs = 0.1, fc = 0.1, period = 8.3, nthreads=8)
159 µs ± 1.76 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
```
Not bad, thats a gain of around 5.7 in speed using 8 processors. However, we might
not want to use nthreads when using a bayesian analysis package which is capable of
multiprocessing (e.g. emcee). 

## emcee example
              
```python
def lnlike(theta):
     # Unpack
     radius_1,k,t_zero,period = theta
     
     # Nounds checking
     if (radius_1<0) or (radius_1 > 0.5) : return -np.inf
     if (k<0) or (k>0.5) : return -np.inf
     if (period < 0) or (period > 15) : return -np.inf
     if (t_zero < -0.1) or (t_zero > 0.1) : return -np.inf

     
     # Model call
     flux_model = qpower2.lc(t, radius_1 = radius_1, k = k, t_zero = t_zero, period = period)
     
     # Return
     return -0.5*np.sum((flux - flux_model)**2/(0.005**2))
```
Now doing an initial guess
```python
initial = [0.2,0.2,0,8.3]
print('Initial guess : ', lnlike(initial))
```

Now setup some emcee stuff
```python
ndim = len(initial)
nwalkers = 4*ndim
p0 = np.array([np.random.normal(initial, 1e-5) for i in range(nwalkers) ])
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, threads=8)
sampler.run_mcmc(p0, 10000)
```

On my system, I can achieve about 500 steps/s - that means it takes 20 seconds to run the above example.

Lets check the results...
```python
samples = sampler.chain[:, 5000:, :].reshape((-1, ndim))
fig = corner.corner(samples, labels = [r'$R_1/a$', 'k', r'$T_0 [BJD]$', 'P [d]'], truths = [0.2,0.2,0.0,8.3])
plt.show()
```
![alt text](https://github.com/samgill844/pyqpower2/blob/master/images/Figure_2.png)

## Batch lightcurves

We can go even faster usignt the GPU. It is inefficient to use the GPU to create a single model (the memory
copying slows things down to the point which the cpu is faster). Instead, we will generate 10,240 models at once. 
The first thing we need to do is tweak emcee. The reason for this is that emcee calls lnlike for each trial model 
(e.g. in our case, an array of [r1, k, t_zero, period] is passed to lnlike. Instead, we'd like to pass all the trial models
at once and get an array of log-likliehoods back.

I reccomed cloning your version of emcee in site packages (anaconda distributions) and rename the folder from "emcee (copy)"
to "emceegpu". That way when you run
```python
import emceegpu
```
it actually just import a copied distro of emcee.

Next go to line ~382 and find
```python
results = list(M(self.lnprobfn, [p[i] for i in range(len(p))]))
```
and change it to
```python
results = self.lnprobfn(p) # Her p is a list of models
```
try to import emceegpu to check everything is OK,
```python 
import emceegpu as emcee
```


Now we need to create a modified lnlike function which has a different style of bounds checking. 

```python
def lnlike(theta):
    theta = theta.T # transpose parameter array

    # Bounds checking
    mask = (theta[0]<0.0) | (theta[0] > 0.5) | (theta[1]<0.0) | (theta[1] > 0.5) | (theta[2]<-0.1) | (theta[2] > 0.1) | (theta[3]<2) | (theta[3] > 15)

    # create the loglike array
    loglike = np.full(theta.shape[1], -np.inf)      


    # make the call
    batch_lc = qpower2.lc_batch(t, radius_1 = np.ascontiguousarray(theta[0]), k = np.ascontiguousarray(theta[1]),
     t_zero = np.ascontiguousarray(theta[2]), period = np.ascontiguousarray(theta[3]))

    # calculate the loglikes
    loglike[~mask] = np.sum(-0.5*(np.subtract(flux,batch_lc)**2/0.001), axis=1)[~mask]

    return loglike
```

Don't forget to update the number of walkers and create new starting positions,
```python
   nwalkers = 10240
   p0 = np.array([np.random.normal(initial, 1e-5) for i in range(nwalkers) ])
```
and 
```python
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike)
```



We're just about ready to make the call now. But how many should we make? Bare in mind that 
with each step in emcee we are creating and storing 10240 trial steps. We should see convergence
quickly in, say, 20 steps.

```python
sampler.run_mcmc(p0, 100)
```
On my system, that took 8 seconds and generated 204800 models (40&mu per model). Lets inspect the results:
```python
samples = sampler.chain[:, 98:, :].reshape((-1, ndim)) # this is 10240*2 values
fig = corner.corner(samples, labels = [r'$R_1/a$', 'k', r'$T_0 [BJD]$', 'P [d]'], truths = [0.2,0.2,0.0,8.3])
plt.show()
```
![alt text](https://github.com/samgill844/qpower2/blob/master/images/Figure_3.png)

Brilliant!


## The seriously fast way

We havent yet exploted the GPU as well as we could. The loss of efficieny comes from 
memory transfers between the device and the host to get 10,240 lightcurves back. In reality,
we only need the log-likliehood for each model back, not the model itself. So instead of copying an array of shape 
t.shape[0] * n_walkers back from the GPU, we only copy an array of shape n_walkers (the loglikliehood array). 
We can achieve this using the lc_batch_loglike function. We re-write out loglikliehood function:

```python
def lnlike(theta):
   theta = theta.T # transpose parameter array
     
   # Bounds checking
   mask = (theta[0]<0.0) | (theta[0] > 0.5) | (theta[1]<0.0) | (theta[1] > 0.5) | (theta[2]<-0.1) | (theta[2] > 0.1) | (theta[3]<2) | (theta[3] > 15)

   # create the loglike array
   loglike = np.full(theta.shape[1], -np.inf)      

   # make the call
   return qpower2.lc_batch_loglike(t, flux, flux_err, radius_1 = np.ascontiguousarray(theta[0]), k = np.ascontiguousarray(theta[1]),
   det_zero = np.ascontiguousarray(theta[2]), period = np.ascontiguousarray(theta[3]))[~mask]
   
   
flux_err = 0.005*np.ones(len(t))
```
No we redefine our sampler and run again!
```python
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike)
sampler.run_mcmc(p0, 100)
samples = sampler.chain[:, 98:, :].reshape((-1, ndim)) # this is 10240*2 values
fig = corner.corner(samples, labels = [r'$R_1/a$', 'k', r'$T_0 [BJD]$', 'P [d]'], truths = [0.2,0.2,0.0,8.3])
plt.show()
```
![alt text](https://github.com/samgill844/pyqpower2/blob/master/images/Figure_4.png)

Much faster, I can hit over 30 iterations a second which translates to 307,200 models per second! In total, this took 3 seconds to run and converge. 

## C implmenetation
TBD




