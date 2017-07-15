import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plot

from sklearn.svm import SVC

mat_contents = sio.loadmat('validation.mat')
oct_a = mat_contents['validation']
val_labels = oct_a[:,-1:].ravel()
val_matrix = oct_a[:,:-1]

mat_contents_b = sio.loadmat('train2.mat')
oct_b = mat_contents_b['training']
train_labels = oct_b[:,-1:].ravel()
train_matrix = oct_b[:,:-1]

x_values = [100, 200, 500, 1000, 2000, 5000]
y_values = []
train_yvalues = []

for v in x_values:
	clf = SVC(kernel = 'linear')
	clf.fit(train_matrix[:v], train_labels[:v])
	val_score = clf.score(val_matrix, val_labels)
	y_values.append(val_score)
	train_score = clf.score(train_matrix[:v], train_labels[:v])
	train_yvalues.append(train_score)

plot.plot(x_values, y_values, label = "CIFAR Validation")
plot.plot(x_values, train_yvalues, label = "CIFAR Training")
plot.legend()
plot.grid()
plot.xlabel("Training Examples")
plot.ylabel("Accuracy")
plot.savefig("CIFAR_Output")