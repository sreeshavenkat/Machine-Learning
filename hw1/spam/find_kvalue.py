import scipy.io as sio
import numpy as np
import random

mat_contents = sio.loadmat('spam_data.mat')
oct_a = mat_contents['training_data']
np.random.shuffle(np.array(oct_a))
split = np.array_split(oct_a, 5)
sio.savemat('k1.mat', {'k1':split[0]})
sio.savemat('k2.mat', {'k2':split[1]})
sio.savemat('k3.mat', {'k3':split[2]})
sio.savemat('k4.mat', {'k4':split[3]})
sio.savemat('k5.mat', {'k5':split[4]})


mat_contents = sio.loadmat('spam_data.mat')
oct_a = mat_contents['training_data']
oct_b = mat_contents['training_labels'][0]
zipped = list(zip(oct_a, oct_b))
random.shuffle(zipped)
oct_a, oct_b = zip(*zipped)
oct_a, oct_b = np.array(oct_a), np.array(oct_b)
split_data = np.array_split(oct_a, 5)
split_labels = np.array_split(oct_b, 5)
sio.savemat('k1.mat', {'k1':split_data[0], 'labels': split_labels[0]})
sio.savemat('k2.mat', {'k2':split_data[1], 'labels': split_labels[1]})
sio.savemat('k3.mat', {'k3':split_data[2], 'labels': split_labels[2]})
sio.savemat('k4.mat', {'k4':split_data[3], 'labels': split_labels[3]})
sio.savemat('k5.mat', {'k5':split_data[4], 'labels': split_labels[4]})