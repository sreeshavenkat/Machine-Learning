import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plot

from sklearn.svm import SVC

k_dict = dict()

mat_contents = sio.loadmat('k1.mat')
oct_a = mat_contents['k1']
k1_a = mat_contents['labels']
k1val_labels = k1_a.ravel()
k1val_matrix = oct_a[:,:-1]
k_dict[1] = (k1val_matrix, k1val_labels)

mat_contents_b = sio.loadmat('k2.mat')
oct_b = mat_contents_b['k2']
k2_b = mat_contents_b['labels']
k2val_labels = k2_b.ravel()
k2val_matrix = oct_b[:,:-1]
k_dict[2] = (k2val_matrix, k2val_labels)

mat_contents_c = sio.loadmat('k3.mat')
oct_c = mat_contents_c['k3']
k3_c = mat_contents_c['labels']
k3val_labels = k3_c.ravel()
k3val_matrix = oct_c[:,:-1]
k_dict[3] = (k3val_matrix, k3val_labels)

mat_contents_d = sio.loadmat('k4.mat')
oct_d = mat_contents_d['k4']
k4_d = mat_contents_d['labels']
k4val_labels = k4_d.ravel()
k4val_matrix = oct_d[:,:-1]
k_dict[4] = (k4val_matrix, k4val_labels)

mat_contents_e = sio.loadmat('k5.mat')
oct_e = mat_contents_e['k5']
k5_e = mat_contents_e['labels']
k5val_labels = k5_e.ravel()
k5val_matrix = oct_e[:,:-1]
k_dict[5] = (k5val_matrix, k5val_labels)

c_value = 1
fin_val = []
while c_value < 100:
	y_values = []
	z = 1
	while z < 6:
		train_matrix = []
		train_labels = []
		for k in k_dict:
			if k != z:
				train_matrix.append(k_dict[k][0])
				train_labels.append(k_dict[k][1])
		tuple(train_matrix)
		tuple(train_labels)
		concat_matrix = np.concatenate(train_matrix)
		concat_labels = np.concatenate(train_labels)
		clf = SVC(C = c_value, kernel = 'linear')
		clf.fit(concat_matrix, concat_labels)		
		y_values.append(clf.score(k_dict[z][0], k_dict[z][1]))
		z += 1 
	c_value += 10
	fin_val.append(y_values)

print(y_values)
