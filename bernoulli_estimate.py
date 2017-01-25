import numpy as np
import matplotlib.pyplot as plt
import h5py as hf
from kalman import fil
from numpy import math
import pickle

dset = hf.File("Tail_687_1_4120922.h5", "r")
# '''Altitude'''
alt = dset['PRESSURE ALTITUDE LSP (FEET)/data'][0]


# '''pitot tube'''
PT = dset['TOTAL PRESSURE LSP (MB)/data'][0]*100 # 1MB = 100PA

# '''static port'''
PS = dset['STATIC PRESSURE LSP (IN)/data'][0]*3386.39  # 1 IN Mercury = 33MB , 1MB = 100PA 

# '''Dynamic Pressure'''
PD = np.abs(PT-PS) #Pascals 

# '''True Airspeed'''
TAS = dset['TRUE AIRSPEED LSP (KNOTS)/data'][0][::2]
n = PD.shape[0]

# '''Total Air Temperature'''
TAT = dset['TOTAL AIR TEMPERATURE (DEG)/data'][0]+273.15 #Kelvin
temp = np.zeros(n)
for k in range(n/2-1):
	temp[2*k] = TAT[k]
	temp[2*k+1] = (TAT[k]+TAT[k+1])/2

# b_speed = np.zeros(n)
# air_density = np.zeros(n)
# for i in range(n):
	# air_density[i] = PT[i]*.92/(temp[i]*287.05)
	# rad = 2*PD[i]/air_density[i]
	# b_speed[i] = math.sqrt(rad)*1.94284
#plt.plot(b_speed, label = 'bernoulli')
#plt.plot(TAS, label = 'TAS')
#plt.legend()
#plt.show()

def create_error_library(tas, ps, pt, t, alt):
	'''tas should the the True Air speed for one entire flight
	   ps pd density and temp should also be for one entire flight'''
	   
	'''This function will simulate each of the three blocks they will each last 100 seconds'''
	n = tas.shape[0]   
	takeoff_start = np.int(n*.18) 
	cruise_start1 = np.int(n*.25)
	cruise_start2 = np.int(n*.4)
	cruise_start3 = np.int(n*.55)
	landing_start = np.int(n*.75)
	landing_2 = np.int(n*.85)
	
	temp = np.zeros(n)
	for k in range(n/2-1):
		temp[2*k] = t[k]
		temp[2*k+1] = (t[k]+t[k+1])/2
	'''Pitot tube block CLASS 1'''

	pitot_tube_block = np.zeros((n,2))
	pitot_tube_block[:,0] = tas

	for k in range(10):
		pitot_tube_block[takeoff_start+k,:] = pitot_tube_block[takeoff_start+k-1,0]*(1/10),1
		pitot_tube_block[cruise_start1+k,:] = pitot_tube_block[cruise_start1+k-1,0]*(1/10),1
		pitot_tube_block[cruise_start2+k,:] = pitot_tube_block[cruise_start2+k-1,0]*(1/10),1
		pitot_tube_block[cruise_start3+k,:] = pitot_tube_block[cruise_start3+k-1,0]*(1/10),1
		pitot_tube_block[landing_start+k,:] = pitot_tube_block[landing_start+k-1,0]*(1/10),1
		pitot_tube_block[landing_2+k,:] = pitot_tube_block[landing_2+k-1,0]*(1/10),1


	for i in range(10,90):
		'''speed erring'''
		pitot_tube_block[takeoff_start+i,:] = 0,1
		pitot_tube_block[cruise_start1+i,:] = 0,1
		pitot_tube_block[cruise_start2+i,:] = 0,1
		pitot_tube_block[cruise_start3+i,:] = 0,1
		pitot_tube_block[landing_start+i,:] = 0,1
		pitot_tube_block[landing_2+i,:] = 0,1

	
	
	for j in range(90,100): 
		pitot_tube_block[takeoff_start+j,:] = pitot_tube_block[takeoff_start+j-1,0]*(10),1
		pitot_tube_block[cruise_start1+j,:] = pitot_tube_block[cruise_start1+j-1,0]*(10),1
		pitot_tube_block[cruise_start2+j,:] = pitot_tube_block[cruise_start2+j-1,0]*(10),1
		pitot_tube_block[cruise_start3+j,:] = pitot_tube_block[cruise_start3+j-1,0]*(10),1
		pitot_tube_block[landing_start+j,:] = pitot_tube_block[landing_start+j-1,0]*(10),1
		pitot_tube_block[landing_2+j,:] = pitot_tube_block[landing_2+j-1,0]*(10),1

	#plt.plot(pitot_tube_block)
	#plt.show()
	'''Pitot tube + drain block CLASS 2'''
	pitot_drain_block = np.zeros((n,2))
	pitot_drain_block[:,0] = tas
	pt[takeoff_start:takeoff_start+100] = pt[takeoff_start]
	pt[cruise_start1:cruise_start1+100] = pt[cruise_start1]
	pt[cruise_start2:cruise_start2+100] = pt[cruise_start2]
	pt[cruise_start3:cruise_start3+100] = pt[cruise_start3]
	pt[landing_start:landing_start+100] = pt[landing_start]
	pt[landing_2:landing_2+100] = pt[landing_2]

	
	pd = pt-ps
	air_density = np.zeros(n)
	for i in range(100):
		air_density[takeoff_start+i] = pt[takeoff_start+i]*.92/(temp[takeoff_start+i]*287.05)
		rad = 2*pd[takeoff_start+i]/air_density[takeoff_start+i]
		pitot_drain_block[takeoff_start+i, :] = math.sqrt(rad)*1.94284,2
			
		air_density[cruise_start1+i] = pt[cruise_start1+i]*.92/(temp[cruise_start1+i]*287.05)
		rad = 2*pd[cruise_start1+i]/air_density[cruise_start1+i]
		pitot_drain_block[cruise_start1+i, :] = math.sqrt(rad)*1.94284,2
		
		air_density[cruise_start2+i] = pt[cruise_start2+i]*.92/(temp[cruise_start2+i]*287.05)
		rad = 2*pd[cruise_start2+i]/air_density[cruise_start2+i]
		pitot_drain_block[cruise_start2+i, :] = math.sqrt(rad)*1.94284,2
		
		air_density[cruise_start3+i] = pt[cruise_start3+i]*.92/(temp[cruise_start3+i]*287.05)
		rad = 2*pd[cruise_start3+i]/air_density[cruise_start3+i]
		pitot_drain_block[cruise_start3+i, :] = math.sqrt(rad)*1.94284,2
		
		air_density[landing_start+i] = pt[landing_start+i]*.92/(temp[landing_start+i]*287.05)
		rad = 2*pd[landing_start+i]/air_density[landing_start+i]
		pitot_drain_block[landing_start+i, :] = math.sqrt(rad)*1.94284,2
		
		air_density[landing_2+i] = pt[landing_2+i]*.92/(temp[landing_2+i]*287.05)
		rad = 2*pd[landing_2+i]/air_density[landing_2+i]
		pitot_drain_block[landing_2+i, :] = math.sqrt(rad)*1.94284,2

	#plt.plot(pitot_drain_block)
	#plt.show()
	'''Static Port Block CLASS 3'''
	
	static_block = np.zeros((n,2))
	static_block[:,0] = tas
	ps[takeoff_start:takeoff_start+100] = ps[takeoff_start]
	ps[cruise_start1:cruise_start1+100] = ps[cruise_start1]
	ps[cruise_start2:cruise_start2+100] = ps[cruise_start2]
	ps[cruise_start3:cruise_start3+100] = ps[cruise_start3]
	ps[landing_start:landing_start+100] = ps[landing_start]
	ps[landing_2:landing_2+100] = ps[landing_2]

	air_density = np.zeros(n)
	pd = pt-ps
	
	for i in range(100):
		air_density[takeoff_start+i] = pt[takeoff_start+i]*.92/(temp[takeoff_start+i]*287.05)
		rad = 2*pd[takeoff_start+i]/air_density[takeoff_start+i]
		static_block[takeoff_start+i, :] = math.sqrt(rad)*1.94284,3
		
		air_density[cruise_start1+i] = pt[cruise_start1+i]*.92/(temp[cruise_start1+i]*287.05)
		rad = 2*pd[cruise_start1+i]/air_density[cruise_start1+i]
		static_block[cruise_start1+i, :] = math.sqrt(rad)*1.94284,3
		
		air_density[cruise_start2+i] = pt[cruise_start2+i]*.92/(temp[cruise_start2+i]*287.05)
		rad = 2*pd[cruise_start2+i]/air_density[cruise_start2+i]
		static_block[cruise_start2+i, :] = math.sqrt(rad)*1.94284,3
		
		air_density[cruise_start3+i] = pt[cruise_start3+i]*.92/(temp[cruise_start3+i]*287.05)
		rad = 2*pd[cruise_start3+i]/air_density[cruise_start3+i]
		static_block[cruise_start3+i, :] = math.sqrt(rad)*1.94284,3
		
		air_density[landing_start+i] = pt[landing_start+i]*.92/(temp[landing_start+i]*287.05)
		rad = 2*pd[landing_start+i]/air_density[landing_start+i]
		static_block[landing_start+i, :] = math.sqrt(rad)*1.94284,3
		
		air_density[landing_2+i] = pt[landing_2+i]*.92/(temp[landing_2+i]*287.05)
		rad = 2*pd[landing_2+i]/air_density[landing_2+i]
		static_block[landing_2+i, :] = math.sqrt(rad)*1.94284,3
	total = np.concatenate((pitot_drain_block, np.concatenate((pitot_tube_block, static_block))))
	return total
#create_error_library(TAS, PS, PT,  temp, alt)


	
	   
	   
	
	   
	   
	   
	   
	   
	   
	