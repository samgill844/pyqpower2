import numpy as np
import matplotlib.pyplot as plt 
import gpuastro as g 


t = np.linspace(0,1,20, dtype=np.float64)
RV1, _ = g.rv(t, fs=0.2,fc = -0.12, V0 = 14, K1 = 35.5)
RV1_err = np.random.uniform(0.8,1.4,len(RV1))
RV1 = np.random.normal(RV1, RV1_err)




N_models, N_dim = 6, 4

# Constant parameters
t_zero = np.zeros(N_models**N_dim,dtype=np.float64)
period = np.ones(N_models**N_dim,dtype=np.float64)
K2 = np.ones(N_models**N_dim,dtype=np.float64)
dV0= np.zeros(N_models**N_dim,dtype=np.float64)


K1_low,K1_high = 30,40
fs_low, fs_high = -0.5,0.5
fc_low, fc_high = -0.5,0.5
V0_low, V0_high = 10,20


f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.set_xlabel('fs')
ax2.set_xlabel('fc')
ax3.set_xlabel('K1')
ax4.set_xlabel('V0')

print('{:>15} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15}'.format('K1 [km/s]', 'fs', 'fc', 'V0', 'Iteration', 'Chi_red', '# of models tested'))
itt = 1
best_old = np.ones(4)*100

while True:
	# Fitting parameters
	K1 = np.linspace(K1_low,K1_high,N_models,dtype=np.float64)
	fs = np.linspace(fs_low, fs_high,N_models,dtype=np.float64)
	fc = np.linspace(fc_low, fc_high,N_models,dtype=np.float64)
	V0 = np.linspace(V0_low, V0_high,N_models,dtype=np.float64)

	models=np.zeros((N_models**N_dim, N_dim))
	count=0
	for i in K1:
		for j in fs:
			for k in fc:
				for y in V0:
					models[count] = [i,j,k,y]
					count +=1 

	K1, fs, fc, V0 = models.T
	K1 = np.ascontiguousarray(K1)
	fs = np.ascontiguousarray(fs)
	fc = np.ascontiguousarray(fc)
	V0 = np.ascontiguousarray(V0)

	RV1_batch, _ =  g.rv_batch(t,  t_zero , period , K1 , K2 , fs, fc,  V0, dV0, n_batch = N_models**N_dim)
	#RV1_batch[np.isnan(RV1_batch)] = np.inf
	loglike = -0.5*np.sum((RV1 - RV1_batch)**2, axis=1)

	best = models[np.argmax(loglike)]
	print('{:>15.6f} {:>15.6f} {:>15.6f} {:>15.6f} {:>15d} {:>15.6f} {:>15d}'.format(best[0], best[1], best[2],best[3], itt, np.max(loglike)/len(t),  itt*N_models**N_dim))

	ax1.scatter(fs, np.log(-2*loglike), s=10, c='k', alpha = 0.2)
	ax2.scatter(fc, np.log(-2*loglike), s=10, c='k', alpha = 0.2)
	ax3.scatter(K1, np.log(-2*loglike), s=10, c='k', alpha = 0.2)
	ax4.scatter(V0, np.log(-2*loglike), s=10, c='k', alpha = 0.2)

	if np.sum((np.abs((best-best_old)) < 1e-6))==4 : break 

	K1_low,K1_high = K1_low + 0.5*(best[0] - K1_low), K1_high - 0.5*( K1_high-best[0])
	fs_low, fs_high = fs_low + 0.5*(best[1] - fs_low), fs_high - 0.5*( fs_high-best[1])
	fc_low, fc_high = fc_low + 0.5*(best[2] - fc_low), fc_high - 0.5*( fc_high-best[2])
	V0_low, V0_high = V0_low + 0.5*(best[3] - V0_low), V0_high - 0.5*( V0_high-best[3])
	itt += 1
	best_old = best

plt.figure()
plt.errorbar(t,RV1, yerr=RV1_err, fmt='ko', alpha = 0.2)
plt.plot(np.linspace(0,1,1000), g.rv(np.linspace(0,1,1000),fs=best[1],fc = best[2], V0 = best[3], K1 = best[0] )[0], 'r')
plt.errorbar(t,-30+RV1 - g.rv(t,fs=best[1],fc = best[2], V0 = best[3], K1 = best[0] )[0], yerr = RV1_err, fmt='ko', alpha = 0.2)
plt.xlabel('Phase')
plt.ylabel('RV [km/s]')


plt.show()