import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plot

from sklearn.svm import SVC

mat_contents = sio.loadmat('validation.mat')
oct_a = mat_contents['validation']
val_labels = []
for x in oct_a:
	val_labels.append(x[784])
val_matrix = oct_a[:,:-1]

mat_contents_b = sio.loadmat('train.mat')
oct_b = mat_contents_b['training']
train_labels = []
for y in oct_b:
	train_labels.append(y[784])
train_matrix = oct_b[:,:-1]

#x_values = [100, 200, 500, 1000, 2000, 5000]
c_value = .0000001
y_values = []
train_yvalues = []
while c_value < 1:
	clf = SVC(C = c_value, kernel = 'linear')
	clf.fit(train_matrix[:10000], train_labels[:10000])
	y_values.append(clf.score(val_matrix, val_labels))
	train_yvalues.append(clf.score(train_matrix[:1000], train_labels[:1000]))
	c_value *= 10

plot.plot(x_values, y_values, label = "MNIST Validation")
plot.plot(x_values, train_yvalues, label = "MNIST Training")
plot.legend()
plot.grid()
plot.xlabel("Training Examples")
plot.ylabel("Accuracy")
plot.savefig("Improving C")