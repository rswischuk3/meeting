import numpy as np
import matplotlib.pyplot as plt
import h5py as hf
from numpy import math 
import pickle
import operator
from sklearn import cluster
from cluster_analysis import cluster_info

with open('data/tas', 'r') as fp:
	tas = pickle.load(fp)
with open('data/yaw', 'r') as f2:
	yaw = pickle.load(f2)
with open('data/roll', 'r') as f3:
	r = pickle.load(f3)
with open('data/pitch', 'r') as f4:
	p = pickle.load(f4)
with open('data/gs', 'r') as f4:
	gs = pickle.load(f4)

roll = []
pitch = []
for c in range(10):
	roll.append(r[c][::2])
	pitch.append(p[c][::2])
total = 458016


shape = roll[0].shape[0]+roll[1].shape[0]+roll[2].shape[0]

train = np.zeros((shape,5))
train[:,0] = np.append(yaw[0],np.append(yaw[1], yaw[2]))
train[:,1] = np.append(pitch[0], np.append(pitch[1], pitch[2]))
train[:,2] = np.append(roll[0], np.append(roll[1], roll[2]))
train[:,3] = np.append(gs[0], np.append(gs[1], gs[2]))
train[:,4] = np.append(tas[0], np.append(tas[1], tas[2]))


def euclidean(x1,x2,length = 4):
	distance = 0
	for i in range(length):
		if i==3:
			distance += 10*(x1[i]-x2[i])**2
		else:
			distance += (x1[i]-x2[i])**2
	return math.sqrt(distance)
def kNeighbors(trainSet, test, k):
	distances = []
	length = len(test) # attributes: yaw, pitch, roll, ground speed
	labels = np.zeros(k)
	for x in range(len(trainSet)):
		dist = euclidean(test, trainSet[x], length)
		distances.append((trainSet[x], dist, x))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append((distances[x][0]))
		labels[x] = distances[x][2]
	return neighbors, labels #return the row (1x4) of the closest point
	
n_clusters = 15
est = np.zeros(10000)
#labelIndex, centers = cluster_info(n_clusters)


def predict_tas(train, yaw, pitch, roll, gs):
	n = len(yaw[0])
	labelIndex, centers = cluster(train)
	for y in range(n):
		test = [yaw[5][y], pitch[5][y], roll[5][y], gs[5][y]]
		temp,labels = kNeighbors(centers, test,3)
		pull = np.append(labelIndex[np.int(labels[0])], np.append(labelIndex[np.int(labels[1])],labelIndex[np.int(labels[2])]))

		neighbors, l = kNeighbors(train[pull, :], test, 4)
		avg = (neighbors[0][4]+neighbors[1][4]+neighbors[2][4]+neighbors[3][4])/4
		est[y] = avg
	return est
	# plt.plot(est)
	# plt.plot(tas[5][2000:12000], label = 'true')
	# plt.legend()
	# plt.show()

def cluster(train):
	clusters = cluster.KMeans(n_clusters = 15)
	clusters.fit(train[:,:4])
	labels = clusters.labels_
	labels_unique = np.unique(labels)
	n_clusters = len(labels_unique)
	centers = clusters.cluster_centers_
	labelIDX, centers = cluster_info(15, labels, centers)
	return labelIDX, centers
	
def cluster_info(n_clusters, labels, centers):
	#labels = np.loadtxt('cluster_labels.txt')

	label_indices = [None]*n_clusters

	for i in range(n_clusters):
		label_indices[i] = [x for x,v in enumerate(labels) if v == i]

	#centers = np.loadtxt('cluster_centers.txt')
	return label_indices, centers

# np.savetxt('cluster_labels.txt', labels)
# np.savetxt('cluster_centers.txt', centers)
