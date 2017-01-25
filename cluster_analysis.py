import numpy as np
import matplotlib.pyplot as plt
import h5py as hf
from numpy import math

def cluster_info(n_clusters):
	labels = np.loadtxt('cluster_labels.txt')

	label_indices = [None]*n_clusters

	for i in range(n_clusters):
		label_indices[i] = [x for x,v in enumerate(labels) if v == i]

	centers = np.loadtxt('cluster_centers.txt')
	return label_indices, centers