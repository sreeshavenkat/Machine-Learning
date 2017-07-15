import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plot
import csv

from sklearn.svm import SVC

mat_contents = sio.loadmat('validation.mat')
oct_a = mat_contents['validation']
oct_v_labels = mat_contents['labels']
val_matrix = oct_a
val_labels = oct_v_labels.ravel()

mat_contents_b = sio.loadmat('train.mat')
oct_b = mat_contents_b['training']
oct_t_labels = mat_contents_b['labels']
train_matrix = oct_b
train_labels = oct_t_labels.ravel()

mat_contents_c = sio.loadmat('spam_data.mat')
oct_c = mat_contents_c['test_data']
test_matrix = oct_c
print(test_matrix.shape)

#x_values = [100, 200, 500, 1000, 2000, 4137]
#y_values = []
#train_yvalues = []

#for v in x_values:
clf = SVC(C = 1, kernel = 'linear')
clf.fit(train_matrix, train_labels)
#y_values.append(clf.score(val_matrix, val_labels))
predict = clf.predict(test_matrix)
y_values = [["id", "category"]]
i = 0
for	v in predict:
	y_values += [[i,v]]
	i += 1

with open('spam_test.csv', 'w') as f:
	writer = csv.writer(f)
	#for arr in y_values:
	writer.writerows(y_values)
#lstrain_yvalues.append(clf.score(train_matrix[:v], train_labels[:v]))
	
# plot.plot(x_values, y_values, label = "SPAM Validation")
# plot.plot(x_values, train_yvalues, label = "SPAM Training")
# plot.legend()
# plot.grid()
# plot.xlabel("Training Examples")
# plot.ylabel("Accuracy")
# plot.savefig("SPAM_Output")