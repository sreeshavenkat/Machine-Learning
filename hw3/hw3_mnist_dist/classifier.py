"""
Q6 Notes:
- Each label at the end of each row corresponds to class value
"""
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

linear = True
mnistTrainData = sio.loadmat('train.mat')["trainX"]
data = mnistTrainData[:,:-1]
labels = mnistTrainData[:,-1]
data = data[10000:10100]
labels = labels[10000:10100]
data, labels = np.array(data), np.array(labels)

mnistValidationData = sio.loadmat('validation.mat')
validation_data = mnistValidationData["validation"]
validation_labels = mnistValidationData["labels"]
validation_data, validation_labels = np.array(validation_data), np.array(validation_labels)

mnistTestData = sio.loadmat('test.mat')
test_data = mnistTestData["testX"]
test_data = np.array(test_data)

all_train = np.append(preprocessing.normalize(data).T, [labels], axis=0).T
validation_data = preprocessing.normalize(validation_data, norm='l2', axis=1, copy=True)
test_data = preprocessing.normalize(test_data, norm='l2', axis=1, copy=True)

mnistTrainData = all_train.astype(float)
mnistTrainData = mnistTrainData[mnistTrainData[:,-1].argsort()]
indices = np.where(np.diff(mnistTrainData[:,-1]))[0]
indices = np.insert(indices, 0, -1)
classes = []
for i in range (0, indices.shape[0] - 1):
	start = indices[i] + 1
	end = indices[i + 1] + 1
	classes.append(mnistTrainData[start:end])
classes.append(mnistTrainData[indices[-1] + 1:])

class_matrix = []
for i in range(len(classes)):
	class_matrix.append(classes[i][:,:-1])

mean_vector = []
covariance_vector = []
for arr in class_matrix:
	mean_vector.append(np.mean(arr, axis=0))
	covariance_vector.append(np.cov(arr, rowvar=False))
mean_vector, covariance_vector = np.array(mean_vector), np.array(covariance_vector)
average_cov = np.mean(covariance_vector, axis=0)

identity = np.identity(784)
identity = np.multiply(identity, 10**-3.6)
var = []
j = 0
while j < 10:
 	var.append(multivariate_normal(mean_vector[j], covariance_vector[j] + identity))
 	j += 1
plt.pcolor(covariance_vector[0])
plt.colorbar()
plt.yticks(np.arange(0.5, 10.5), range(0, 10))
plt.xticks(np.arange(0.5, 10.5), range(0, 10))
plt.show()

# success = 0
# failure = 0
# total = 0

# inv_precompute = []
# determinant_pre = []
# for i in range(10):	
# 	if linear:
# 		inv_precompute.append(np.matrix(np.linalg.inv(average_cov + identity)))
# 		eig = np.linalg.eig(average_cov + identity)
# 	else:
# 		inv_precompute.append(np.matrix(np.linalg.inv(covariance_vector[i] + identity)))
# 		eig = np.linalg.eig(covariance_vector[i] + identity)
# 	log_eig = np.log(eig[0])
# 	determinant = float(np.sum(log_eig))
# 	determinant_pre.append(determinant)

# for d in range(len(validation_data)):
# 	correct_class = 0
# 	data = validation_data[d]
# 	max_prob = -100000
# 	print(d)
# 	for i in range(10):
# 		determinant = determinant_pre[i]
# 		mean_transpose = np.matrix(np.subtract(data[:784], mean_vector[i]))
# 		mean_regular = mean_transpose.T
# 		inverse = inv_precompute[i]
# 		right_val = np.matmul(mean_transpose, np.matmul(inverse, mean_regular))
# 		left_val = np.multiply(-0.5, determinant)
# 		log_pdf = left_val + (-0.5 * right_val)
# 		if log_pdf > max_prob:
# 			max_prob = log_pdf
# 			correct_class = i
# 	if validation_labels[0][d] == correct_class:
# 		success += 1
# 	else:
# 		failure += 1
# 	total += 1
# print(failure/total)


