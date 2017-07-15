import csv
import scipy.io as sio
import numpy as np
import random
import matplotlib.pyplot as plot
import sklearn.preprocessing
from sklearn.utils import shuffle
from scipy.special import expit as sigmoid

def train(images, labels, iterations):
	weight_V = np.random.normal(0, 0.01, (200, 785))
	weight_W = np.random.normal(0, 0.01, (26, 201))
	learning_rate = .01
	x_values = []
	y_values = []
	#for count in range(1):
	for i in range(50000):
		rand = i
		#rand = np.random.randint(0,99)
		r_image = images[rand]
		r_label = labels[rand]
		y, z, h = forward_pass(r_image, r_label, weight_V, weight_W)
		dLdV, dLdW = backward_pass(y, z, h, r_image, weight_V, weight_W)
		weight_V, weight_W = stochastic_gradient_descent(weight_W, weight_V, dLdW, dLdV, learning_rate)
		if i % 1000 == 0:
			x_values.append(i)
			y_values.append(loss(y, z))
	learning_rate *= 0.7
	return weight_V, weight_W

def stochastic_gradient_descent(W, V, dLdW, dLdV, learning_rate):
	W = W - learning_rate * dLdW
	W = np.asarray(W)
	V = V - learning_rate * dLdV
	V = np.asarray(V)
	V = V * (1 - .000001)
	W = W * (1 - .000001)
	return V, W

def forward_pass(image, label, V, W):
	img_array = np.asarray(image).T
	z1 = np.matmul(V, img_array) # z1 is (200, 1)
	activation = np.tanh(z1)
	z1 = np.append(activation, 1) # z1 is (201, 1)
	z1 = np.matrix(z1).T
	activation = z1
	z1 = np.matmul(W, z1) # z1 is (26, 1)
	z = sigmoid(z1)
	y = np.zeros((26,1))
	y[label - 1] = 1
	return y, z, activation

def backward_pass(y, z, h, image, V, W):
	lst = []
	sig_V = sigmoid(2 * np.dot(V, image))
	for i in range(200):
		grad_h = 4 * (sig_V[i] * (1 - sig_V[i]))
		transpose = np.matrix(W.T[i])
		lst.append(np.ndarray.tolist(transpose * grad_h)[0])
	image = np.matrix(image)
	second_term = np.matmul(z - y, image)
	dLdV = np.matmul(np.array(lst), second_term)
	dLdW = np.matmul(z - y, h.T)
	return dLdV, dLdW

def loss(y, z):
	loss = sum( -(np.multiply(y, np.log(z)) + np.multiply(1.0 - y, np.log(1.0 - z))))
	return loss.item(0)

def predict(images, V, W):
	predicted_labels = []
	for img in images:
		z1 = np.dot(V, img.T)
		activation = np.tanh(z1)
		z1 = np.append(activation, 1)
		z1 = np.reshape(z1, (len(z1), 1))
		z1 = np.dot(W, z1)
		z = sigmoid(z1)
		predicted_labels.append(np.argmax(z) + 1)
	return predicted_labels

def error_rate(labels, predictions):
	i = 0
	total = 0
	error = 0
	for v in labels:
		if v != predictions[i]:
			print(i)
			error += 1
		total += 1
		i += 1
	return error/total

def write_to_csv(predictions):
	predictions_csv = [["id", "category"]]
	i = 1
	for v in predictions:
		predictions_csv += [[i, v]]
		i += 1
	with open('letters_data_predictions.csv', 'w') as f:
		writer = csv.writer(f)
		writer.writerows(predictions_csv)

def main():	
	letters_data = sio.loadmat("letters_data.mat")
	test_data = letters_data['test_x']
	test_data = (test_data - np.mean(test_data)) / np.std(test_data)
	test_data = sklearn.preprocessing.scale(test_data)
	test_data = np.hstack((test_data, np.ones((20800, 1))))
	data = letters_data['train_x']
	labels = letters_data['train_y']
	all_letters_data = np.hstack((data, labels))
	np.random.shuffle(all_letters_data)
	data = all_letters_data[:,:-1]
	data = sklearn.preprocessing.scale(data)
	data = np.hstack((data, np.ones((124800, 1))))
	labels = []
	for row in all_letters_data:
		labels.append(row[784])
	labels = np.asarray(labels)
	training_data = data
	training_labels = labels
	training_data = data[:99840]
	training_labels = labels[:99840]
	# validation_data = data[99840:]
	# validation_labels = labels[99840:]
	V, W = train(training_data, training_labels, 100)
	exit()
	predictions = predict(test_data, V, W)
	write_to_csv(predictions)
	error_rate(validation_labels, predictions)

if __name__ == "__main__":
	main()