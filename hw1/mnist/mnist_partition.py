import scipy.io as sio
import numpy as np

mat_contents = sio.loadmat('train.mat')
oct_a = mat_contents['trainX']
np.random.shuffle(oct_a)
validation = oct_a[:10000]
training = oct_a[10000:]
sio.savemat('validation.mat', {'validation':validation})
sio.savemat('train.mat', {'training':training})
