import scipy.io as sio
import numpy as np
import random

mnistTrainData = sio.loadmat('train.mat')["trainX"]
data = mnistTrainData[:,:-1]
labels = mnistTrainData[:,-1]
# zipped = list(zip(data, labels))
# random.shuffle(zipped)
# data, labels = zip(*zipped)
data, labels = np.array(data), np.array(labels)
validation = data[:10000]
v_labels = labels[:10000]
training = data[10000:]
t_labels = labels[10000:]
sio.savemat('validation.mat', {'validation':validation, 'labels':v_labels})
sio.savemat('partition_train.mat', {'training':training, 'labels': t_labels})
