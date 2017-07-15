import scipy.io
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt

ranks = [i for i in range (1, 101)]
MSE = []
for r in ranks:
	print(r)
	data = imread("face.jpg")
	U, sigma, V = np.linalg.svd(data, full_matrices = False)
	for i in range(sigma.shape[0]):
		if i >= r:
			sigma[i] = 0
	img = np.matmul(U, np.multiply(V.T, sigma).T)
	dist = np.matrix(np.square(data - img))
	MSE.append(dist.sum())
plt.figure()
plt.plot(ranks, MSE)
plt.xlabel("Ranks")
plt.ylabel("MSE")
plt.grid()
plt.show()
