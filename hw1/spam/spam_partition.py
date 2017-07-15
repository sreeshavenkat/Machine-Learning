pytimport scipy.io as sio
import numpy as np
import random

mat_contents = sio.loadmat('spam_data.mat')
oct_a = mat_contents['training_data']
oct_b = mat_contents['training_labels'][0]
zipped = list(zip(oct_a, oct_b))
random.shuffle(zipped)
oct_a, oct_b = zip(*zipped)
oct_a, oct_b = np.array(oct_a), np.array(oct_b)
validation = oct_a[:1035]
v_labels = oct_b[:1035]
training = oct_a[1035:]
t_labels = oct_b[1035:]
sio.savemat('validation.mat', {'validation':validation, 'labels':v_labels})
sio.savemat('train.mat', {'training':training, 'labels': t_labels})
