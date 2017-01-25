import numpy as np
import matplotlib.pyplot as plt
import h5py as hf
from numpy import math 
from kalman import fil
#from groundspeed import speed

dset = hf.File("Tail_687_1_4120922.h5", "r") #Cincinatti to detroit
WindSpeed = dset['WIND SPEED (KNOTS)/data'][0][::2]
n = len(WindSpeed)
PS = dset['STATIC PRESSURE LSP (IN)/data'][0]*33.8639  # 1 IN Mercury =  33PA
#Alt = (1-((PS)/1013.25)**.190284)*145366.45*.3048	#Altitude in meters
#print len(Alt)
Lat = dset['LATITUDE POSITION LSP (DEG)/data'][0]*math.pi/180
Lon = dset['LONGITUDE POSITION LSP (DEG)/data'][0]*math.pi/180
#print len(Lat)
Pitch = dset['PITCH ANGLE LSP (DEG)/data'][0]*math.pi/180
Roll = dset['ROLL ANGLE LSP (DEG)/data'][0]*math.pi/180
Yaw = dset['TRUE HEADING LSP (DEG)/data'][0]*math.pi/180
print len(Lat)
#print len(PA)
#time, groundspeed = speed()

PA = dset['PRESSURE ALTITUDE LSP (FEET)/data'][0]
GroundSpeedTrue = dset['GROUND SPEED LSP (KNOTS)/data'][0]
PA_edit = PA[::4]

m = 4732
r = np.zeros(m)
x = np.zeros(m)
y = np.zeros(m)
z = np.zeros(m)
gs = np.zeros(m)
time = np.zeros(m)
for k in range(m):
		time[k] = 4*k
for k in range(m):
	r[k] = 6378137 + PA_edit[k]
	x[k] = r[k]*math.cos(Lat[k])*math.cos(Lon[k])
	y[k] = r[k]*math.cos(Lat[k])*math.sin(Lon[k])
	z[k] = r[k]*math.sin(Lat[k])*(1-1/298.257223563)
for p in range(m-1):
	a = x[p+1]-x[p]
	b = y[p+1]-y[p]
	c = z[p+1]-z[p]
	gs[p] = math.sqrt(a**2+b**2+c**2)*1.94384
gs[4660:] = np.zeros(m-4660)
gs = fil(gs,q = 1e-3, r = 2)

ws = np.zeros(m)
