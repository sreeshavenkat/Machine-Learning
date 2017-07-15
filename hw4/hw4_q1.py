import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
import scipy.io as sio
import random 
import csv

from scipy.stats import multivariate_normal
from sklearn import preprocessing

w = np.matrix([[-2],[1],[0]])
s = np.matrix([[.9526],[.7311],[.7311],[.2689]])
x = np.matrix([[0, 3, 1], [1, 3, 1], [0, 1, 1], [1, 1, 1]])
omega = np.matrix([[(s[0]*(1-s[0])), 0, 0, 0],
				  [0, (s[1]*(1-s[1])), 0, 0],
				  [0, 0, (s[2]*(1-s[2])), 0],
				  [0, 0, 0, (s[3]*(1-s[3]))]
				  ])

num_1 = np.multiply(.14, w)
num_2 = np.multiply(.0474, x.T)
numerator = num_1 - num_2

temp = np.dot(x.T, omega)
denominator = .14 - np.dot(temp, x)
div = numerator/denominator

print(w - div)