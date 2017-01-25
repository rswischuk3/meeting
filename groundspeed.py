import numpy as np
import matplotlib.pyplot as plt
import h5py as hf
from numpy import math 
from kalman import fil
def direction(dset):
	lat = dset['LATITUDE POSITION LSP (DEG)/data'][0]
	lon = dset['LONGITUDE POSITION LSP (DEG)/data'][0]

	n = len(lat)
	nn = n/10
	degree = np.zeros(n-1)

	for i in range(1,nn):
		if lat[i] == 0.0:
			continue
		if lon[i] ==0.0:
			continue
		else:
			lat1 = lat[i*10-10]*math.pi/180
			lon1 = lon[i*10-10]*math.pi/180
			lat2 = lat[i*10]*math.pi/180
			lon2 = lon[i*10]*math.pi/180
		
			dlat = lat2-lat1
			dlon = lon2-lon1
			
			y = math.sin(dlon)*math.cos(lat2)
			x = math.cos(lat1)*math.sin(lat2)-math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
			
			bearing = math.atan2(y,x)*180/math.pi
			
			if (bearing < 0):
				bearing = 360-abs(bearing)
			
			degree[i-1] = bearing
	return degree
	
def speed():
	'''Working'''
	dset = hf.File("Tail_687_1_4120922.h5", "r")
	n = len(dset['LATITUDE POSITION LSP (DEG)/data'][0])
	m = len(dset['GROUND SPEED LSP (KNOTS)/data'][0])
	time = np.zeros(n)
	GroundSpeed = np.zeros(n)
	for i in range(1,n):
		'''Degrees to radians'''
		lat1 = dset['LATITUDE POSITION LSP (DEG)/data'][0,i-1]*np.math.pi/180
		lat2 = dset['LATITUDE POSITION LSP (DEG)/data'][0,i]*np.math.pi/180
		
		lon1 = dset['LONGITUDE POSITION LSP (DEG)/data'][0,i-1]*np.math.pi/180
		lon2 = dset['LONGITUDE POSITION LSP (DEG)/data'][0,i]*np.math.pi/180
		
		'''Radius of the Earth (meters)'''
		r = 6378100
		
		dlon = lon2 - lon1
		dlat = lat2 - lat1
		
		a = (math.sin(dlat/2))**2+math.cos(lat1)*math.cos(lat2)*(math.sin(dlon/2))**2
		c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
		D = r*c
		
		GroundSpeed[i-1] = D*1.94384
	for k in range(n):
		time[k] = 4*k
		
	'''This works well'''
	
	GroundSpeed = fil(GroundSpeed, 1e-5, .1**2)

	return time[:4660],GroundSpeed[:4660]

def estimate(AirSpeed, AirAngle, WindSpeed, WindAngle):
	'''Works better'''
	#dset = hf.File("Tail_687_1_4120922.h5", "r")
	#GroundSpeedTrue = dset['GROUND SPEED LSP (KNOTS)/data'][0]
	#GroundAngle = dset['TRUE HEADING LSP (DEG)/dat'][0]
	
	
	#AirSpeed = dset['TRUE AIRSPEED LSP (KNOTS)/data'][0]
	#AirAngle = dset['TRUE HEADING LSP (DEG)/data'][0]*math.pi/180
	
	#WindSpeed = dset['WIND SPEED (KNOTS)/data'][0]
	#WindAngle = dset['WIND DIRECTION TRUE (DEG)/data'][0]*math.pi/180
	n = len(AirSpeed)
	Vground = np.zeros(n)
	
	for i in range(n):
		Vground[i] = math.sqrt((AirSpeed[i]**2)+(2*AirSpeed[i]*WindSpeed[i]*math.cos(-AirAngle[i]-WindAngle[i]))+WindSpeed[i]**2)
	Vground = fil(Vground, q = 1e-3, r = 2)

	return Vground

	# plt.plot(GroundSpeedTrue, label = 'True')
	# plt.plot(Vground, label = 'pitot est')
	# plt.plot(time,GroundSpeedEst, label = 'GPS est')
	# plt.legend()
	# plt.subplot(2,1,2)
	# plt.plot((GroundSpeed[::2]-Vground) , label = 'error')
	# plt.plot(np.zeros(n))
	# plt.legend()
	
	#plt.show()

#estimate()
	
	
	
	
	
	
	
	
	
	