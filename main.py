import numpy as np
import pickle
import matplotlib.pyplot as plt
import groundspeed
import build_library
import bernoulli_estimate

tas_t = pickle.load(open('data/tas_3', 'r'))
yaw_t = pickle.load(open('data/yaw_3', 'r'))
roll_t = pickle.load(open('data/roll_3', 'r'))
pitch_t = pickle.load(open('data/pitch_3', 'r'))
gs = pickle.load(open('data/gs_3', 'r'))
wd = pickle.load(open('data/wd_2', 'r'))
ws = pickle.load(open('data/ws_2', 'r'))
alt_t = pickle.load(open('data/alt_3','r'))
pt_t = pickle.load(open('data/pt','r'))
ps_t = pickle.load(open('data/ps','r'))
tat_t = pickle.load(open('data/tat','r'))

k = len(roll_t)
roll = [None]*k
pitch = [None]*k
ps = [None]*k
pt = [None]*k
tat = [None]*k
tas = [None]*k
alt = [None]*k
yaw = [None]*k
x = 7 #i did 3,4,5,6,7
for i in range(k):
	roll[i] = roll_t[i][::4]
	pitch[i] = pitch_t[i][::4]
	ps[i] = ps_t[i]*3386.39
	pt[i] = pt_t[i]*100
	tat[i] = tat_t[i]+273.15
	tas[i] = tas_t[i][::2]
	alt[i] = alt_t[i][::2]
	yaw[i] = yaw_t[i][::2]
print pt_t[x].shape
print ps_t[x].shape
print tat[x].shape
print tas[x].shape
print alt[x].shape
print roll[x].shape
print pitch[x].shape
print yaw[x].shape
error = bernoulli_estimate.create_error_library(tas[x], ps[x], pt[x], tat[x], alt[x])
y = error.shape[0]
s = y/3
plt.subplot(3,1,1)
plt.plot(error[:s,0])
plt.plot(tas[x])
plt.subplot(3,1,2)
plt.plot(error[s:s*2,0])
plt.plot(tas[x])

plt.subplot(3,1,3)
plt.plot(error[s*2:s*3,0])
plt.plot(tas[x])

plt.show()

pickle.dump(error, open('errorFlight5', 'w'))