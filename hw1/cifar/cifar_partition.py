import scipy.io as sio
import numpy as np

mat_contents = sio.loadmat('train.mat')
oct_a = mat_contents['trainX']
print(oct_a.shape)
np.random.shuffle(oct_a)
validation = oct_a[:5000]
training = oct_a[5000:]
sio.savemat('validation.mat', {'validation':validation})
sio.savemat('train2.mat', {'training':training})
