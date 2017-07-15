"""
Q3: Eigenvectors of the Gaussian Covariance Matrix
"""
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math

from scipy.stats import multivariate_normal
from numpy import linalg as LA

X1 = np.random.normal(3, 3, 100)
X2 = (.5*X1) + np.random.normal(4, 2, 100)
X = np.vstack((X1, X2)).T
mean = np.mean(X, axis = 0)
print("Mean: ")
print(mean)

covariance = np.cov(X1, X2)
print("Covariance: ")
print(covariance)

eigenvalues, eigenvectors = LA.eig(covariance)
print("Eigenvalues: ")
print(eigenvalues)
print("Eigenvectors: ")
print(eigenvectors)

ax = plt.gca()
ax.arrow(mean[0], mean[1], eigenvectors[0][0] * eigenvalues[0], eigenvectors[1][0] * eigenvalues[0], 
	head_width = 0.7, head_length = 1.1, fc = 'k', ec = 'k')
ax.arrow(mean[0], mean[1], eigenvectors[0][1] * eigenvalues[1], eigenvectors[1][1] * eigenvalues[1], 
	head_width = 0.7, head_length = 1.1, fc = 'k', ec = 'k')
plt.draw()
plt.xlabel('X1')
plt.xlim(-15, 15)
plt.ylabel('X2')
plt.ylim(-15, 15)
plt.scatter(X1, X2)
plt.show()

sub_mean = X - np.mean(X, axis = 0)
sub_mean_eig = eigenvectors.T.dot(sub_mean.T).T

plt.scatter(sub_mean_eig[:, 0], sub_mean_eig[:, 1])
plt.draw()
plt.xlabel('X1')
plt.xlim(-15, 15)
plt.ylabel('X2')
plt.ylim(-15, 15)
plt.show()
