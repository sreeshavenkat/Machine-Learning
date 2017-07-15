import scipy.io
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plot

train_mat = scipy.io.loadmat('images.mat')
train_images = train_mat['images'].T
train_images_flat = []
for i in range(len(train_images)):
    train_images_flat.append(train_images[i].T.ravel())
train_images_flat = np.array(train_images_flat)

k = 10
k_classes = [[] for i in range(k)]
mu = [np.random.rand(784) for i in range(k)]
not_converged = True
converged = [True for i in range(k)]
while not_converged:
	for image in train_images_flat:
		index, best  = 0, 1000000000000000
		i = 0
		for curr in mu:
			norm = np.linalg.norm(curr - image)
			if norm < best:
				index, best = i, norm
			i += 1
		k_classes[index].append(image)
	for i in range(len(mu)):
		mean = np.mean(k_classes[i], axis=0)
		converged[i] = (mean == mu[i])
		mu[i] = mean
	for c in converged:
		if c is False:
			not_converged = True
			break
		not_converged = False