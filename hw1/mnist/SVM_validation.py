import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plot
import csv

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

mat_contents_c = sio.loadmat('test.mat')
oct_c = mat_contents_c['testX']
test_matrix = oct_c

x_values = [100, 200, 500, 1000, 2000, 5000, 10000]
#y_values = []
#y_values = [["id", "category"]]
train_yvalues = []

# for v in x_values:
# 	clf = SVC(kernel = 'linear')
# 	clf.fit(train_matrix[:v], train_labels[:v])
# 	val_score = clf.score(val_matrix, val_labels)
# 	y_values.append(val_score)
# 	train_score = clf.score(train_matrix[:v], train_labels[:v])
# 	train_yvalues.append(train_score)

# clf = SVC(C = .000001, kernel = 'linear')
# clf.fit(train_matrix, train_labels)
# predict = clf.predict(test_matrix)
# i = 0
# for v in predict:
# 	y_values += [[i, v]]
# 	i += 1

#train_yvalues.append(clf.score(train_matrix[:v], train_labels[:v]))

# with open('mnist_test.csv', 'w') as f:
# 	writer = csv.writer(f)
# 	writer.writerows(y_values)

# plot.plot(x_values, y_values, label = "MNIST Validation")
# plot.plot(x_values, train_yvalues, label = "MNIST Training")
# plot.legend()
# plot.grid()
# plot.xlabel("Training Examples")
# plot.ylabel("Accuracy")
# plot.savefig("Improving C")