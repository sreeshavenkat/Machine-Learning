import matplotlib
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plot
import math
import scipy.io as sio
import scipy.special as special
import random 
import sklearn
import csv
from sklearn import preprocessing

trainingSet = sio.loadmat('data.mat')
data = trainingSet["X"]
labels = trainingSet["y"]
test = trainingSet["X_test"]
data = sklearn.preprocessing.normalize(data, norm='l2', axis=1, copy=True)
test = sklearn.preprocessing.normalize(test, norm='l2', axis=1, copy=True)
shuffled_data = []

validation_data = data[:1200]
training_data = data[1200:]
validation_labels = labels[:1200]
training_labels = labels[1200:]

#learning_rate = 0.0023
regularization = 0.0023

def cost_fn(w, X, y):
	logistic_regression_sum = 0
	for i in range(len(X)):
		first_term = y[i] * np.log(special.expit(np.dot(X[i], w)))
		X_remainder = np.log(1 - special.expit(np.dot(X[i], w)))
		y_remainder = 1 - y[i]
		second_term = X_remainder * y_remainder
		logistic_regression_sum += (first_term + second_term)
	return logistic_regression_sum

def l2_cost_fn(w, X, y):
	l2_regularization = np.multiply(regularization, np.square(np.linalg.norm(w, 2)))
	cost = cost_fn(w, X, y)
	l2_cost = l2_regularization - cost
	return l2_cost[0]

def update(w, X, y):
	sigmoid_subtraction = np.subtract(y, special.expit(np.matmul(X, w)))
	sigmoid_mul = np.matmul(X.T, sigmoid_subtraction)
	reg = regularization * 2 * w
	update = w + (learning_rate * (reg + sigmoid_mul))
	return update

X = np.array(training_data)
y = np.array(training_labels)
w = np.zeros((len(X[0]), 1))
iterations = [i for i in range(1000)]
cost = []

for i in iterations:
	print(i)
	learning_rate = 1/(1+i)
	x_array = np.array([X[i]])
	y_array = np.array([y[i]])
	w = update(w, x_array, y_array)
	iter_cost = l2_cost_fn(w, X, y)
	cost.append(iter_cost)

plot.plot(iterations, cost, label = "Decreasing Learning Rate, Stochastic Gradient Descent")
plot.legend()
plot.grid()
plot.xlabel("Iterations")
plot.ylabel("Cost")
plot.savefig("Learning Rate Decreasing, Stochastic Gradient Descent Cost")

X = np.array(test)
y = np.array(training_labels)

# total = 0
# error = 0
# count = 0
# for i in range(len(X)):
# 	classification = special.expit(np.dot(w.T, X[i]))
# 	if classification < .5:
# 		value = 0
# 	else:
# 		value = 1
# 	if value != y[i]:
# 		error += 1
# 	total += 1
# 	count += 1

# print(cost)
# print(error/total)