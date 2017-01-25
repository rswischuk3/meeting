import numpy as np
import matplotlib.pyplot as plt
import h5py as hf

def fil(avgGroundSpeed, q = 1e-3, r = .1**2):
	dset = hf.File("Tail_687_1_4120922.h5", "r")

	n = len(avgGroundSpeed)
	sz = (n,)
	#uncomment if you uncomment the plots below
	#x = dset['TRUE AIRSPEED LSP (KNOTS)/data'][0]
	z = avgGroundSpeed[:]
	
	Q = q
	
	xhat = np.zeros(sz)
	P = np.zeros(sz)
	xhatminus = np.zeros(sz)
	Pminus = np.zeros(sz)
	K = np.zeros(sz)
	u = np.zeros(sz)
	
	
	R = r
	
	xhat[0] = avgGroundSpeed[0]
	P[0] = 1
	
	for k in range(1,n):
		xhatminus[k] = xhat[k-1] 
		Pminus[k] = P[k-1]+Q
		
		K[k] = Pminus[k]/(Pminus[k]+R)
		xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
		P[k] = (1-K[k])*Pminus[k]
	return xhat
	#plt.plot(z,'k+', label = 'noisy measurements')
	#plt.plot(xhat, label = 'a posteri estimate')
	#plt.plot(x, color = 'g', label = 'truth value')
	#plt.legend()
	#plt.show()