import numpy as np
import matplotlib.pyplot as plt
import h5py as hf
from numpy import math 
import pickle
import operator
from sklearn import cluster
from numpy import random
with open('tas', 'r') as fp:
	tas = pickle.load(fp)
with open('yaw', 'r') as f2:
	trident = pickle.load(f2)
with open('roll', 'r') as f3:
	roll = pickle.load(f3)
with open('pitch', 'r') as f4:
	pitch = pickle.load(f4)
with open('gs', 'r') as f4:
	gs = pickle.load(f4)

psi = []
theta = []
for r in range(10):
	psi.append(roll[r][::2])
	theta.append(pitch[r][::2])
	
shape = psi[0].shape[0]+psi[1].shape[0]+psi[2].shape[0]

train = np.zeros((shape,6))
train[:,0] = np.append(trident[0],np.append(trident[1], trident[2]))
train[:,1] = np.append(theta[0], np.append(theta[1], theta[2]))
train[:,2] = np.append(psi[0], np.append(psi[1],psi[2]))
train[:,3] = np.append(gs[0], np.append(gs[1], gs[2]))
train[:,4] = np.append(tas[0], np.append(tas[1], tas[2]))
	

n_clusters = 15
min_size = shape/n_clusters - n_clusters*100
cluster_centers = np.zeros((n_clusters, 4))
for i in range(n_clusters):
	center_index = random.randint(0,shape)
	cluster_centers[i, :] = train[center_index, :4]
distance = np.zeros(n_clusters)
for k in range(shape):
	for x in range(n_clusters):
		distance[x] = math.sqrt((train[k,0]-cluster_centers[x,0])**2+(train[k,1]-cluster_centers[x,1])**2+(train[k,2]-cluster_centers[x,2])**2+(train[k,3]-cluster_centers[x,3])**2)
	train[]