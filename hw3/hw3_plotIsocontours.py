"""
Illustrate simple contour plotting, contours on an image with
a colorbar for the contours, and labelled contours.

See also contour_image.py.
"""
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math

from scipy.stats import multivariate_normal

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
x = np.arange(-4.0, 4.0, delta)
y = np.arange(-4.0, 4.0, delta)
sigmax = math.sqrt(2.0)
sigmay = math.sqrt(1.0)
sigmaxy = 0.0
mux = 1.0
muy = 1.0
X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, sigmax, sigmay, mux, muy, sigmaxy)

b_sigmax = math.sqrt(2.0)
b_sigmay = math.sqrt(2.0)
b_sigmaxy = 1.0
b_mux = -1.0
b_muy = -1.0
Z2 = mlab.bivariate_normal(X, Y, b_sigmax, b_sigmay, b_mux, b_muy, b_sigmaxy)
Z = 10.0 * (Z2 - Z1)
plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Question 2e')

print("break")
plt.show()