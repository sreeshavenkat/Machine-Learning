import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plot

from sklearn.svm import SVC

k_dict = dict()

mat_contents = sio.loadmat('k1.mat')
oct_a = mat_contents['k1']
k1val_labels = oct_a[:,-1:].ravel()
k1val_matrix = oct_a[:,:-1]
k_dict[1] = (k1val_matrix, k1val_labels)

mat_contents_b = sio.loadmat('k2.mat')
oct_b = mat_contents_b['k2']
k2val_labels = oct_b[:,-1:].ravel()
k2val_matrix = oct_b[:,:-1]
k_dict[2] = (k2val_matrix, k2val_labels)

mat_contents_c = sio.loadmat('k3.mat')
oct_c = mat_contents_c['k3']
k3val_labels = oct_c[:,-1:].ravel()
k3val_matrix = oct_c[:,:-1]
k_dict[3] = (k3val_matrix, k3val_labels)

mat_contents_d = sio.loadmat('k4.mat')
oct_d = mat_contents_d['k4']
k4val_labels = oct_d[:,-1:].ravel()
k4val_matrix = oct_d[:,:-1]
k_dict[4] = (k4val_matrix, k4val_labels)

mat_contents_e = sio.loadmat('k5.mat')
oct_e = mat_contents_e['k5']
k5val_labels = oct_e[:,-1:].ravel()
k5val_matrix = oct_e[:,:-1]
k_dict[5] = (k5val_matrix, k1val_labels)

c_value = .0000001
y_values
while c_value < 1000000:
	z = 0
	while z < 5:
		train_matrix = []
		train_labels = []
		for k in k_dict:
			if k != z:
				train_matrix.append(k_dict[k][0])
				train_labels.append(k_dict[k][1])
		clf = SVC(C = c_value, kernel = 'linear')
		clf.fit(train_matrix[:1000], train_labels[:1000])
		y_values.append(clf.score(k_dict[z][0], k_dict[z][1]))
		z += 1 
		c_value *= 10

print(y_values)
