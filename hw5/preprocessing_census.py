import pandas as pd
import csv
import scipy.io as sio
import numpy as np
import random
from Decision_Tree import DecisionTree
import matplotlib.pyplot as plot
from sklearn.utils import shuffle

def replace_missing_values(input_file, categorical=dict()):
	for row in input_file:
		for key, value in row.items():
			if value == "?":
				row[key] = None
			elif key not in categorical:
				row[key] = float(value)

def mean_and_mode(input_file):
	d = pd.DataFrame(input_file)
	mean = d.mean()
	mode = d.mode()
	return (mean, mode)

def impute(input_file, mean, mode, categorical=dict()):
	for row in input_file:
		for key, value in row.items():
			if value == None:
				if key in categorical:
					row[key] = mode[key][0]
				else:
					row[key] = mean[key]

def rf_train(data, forest_size, tree_depth):
	print("rf train")
	classifiers = []
	for i in range(forest_size):
		tree = DecisionTree()
		tree.root = tree.train(data, tree_depth, True)
		classifiers.append(tree)
	return classifiers

def rf_predict(classifiers, data):
	all_labels = []
	i = 0
	for c in classifiers:
		all_labels.append(c.predict(data))
		i += 1
	print("INDIVIDUAL PREDICTIONS")
	for a in all_labels:
		print(a[:50])
	zip(*all_labels)
	return [average(n) for n in zip(*all_labels)]

def average(nums, default=float('nan')):
	return np.round(int(sum(nums) / float(len(nums))) if nums else default)

def census():
	input_file = list(csv.DictReader(open("train_data.csv")))
	test_input_file = list(csv.DictReader(open("test_data.csv")))
	categorical = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
	return (input_file, test_input_file, 6545, 'label', categorical)

def titanic():
	input_file = list(csv.DictReader(open("../hw5_titanic_dist/titanic_training.csv")))
	test_input_file = list(csv.DictReader(open("../hw5_titanic_dist/titanic_testing_data.csv")))
	for titanic_dict in input_file:
		del titanic_dict["cabin"]
		del titanic_dict["ticket"]
	categorical = ["embarked", "sex"]
	return (input_file, test_input_file, 200, 'survived', categorical)

def spam():
	trainingSet = sio.loadmat("../dist/spam_data.mat")
	spam_training_data = trainingSet['training_data']
	spam_training_labels = trainingSet['training_labels']
	spam_test_data = trainingSet['test_data']
	test_input_file = spam_test_data
	data = pd.DataFrame(spam_training_data)
	labels = pd.Series(spam_training_labels[0])
	data['label'] = labels
	input_file = data
	return (data, test_input_file, int(len(input_file)*.2), 'label', dict())

def main():	
	# MAKE SURE ALL OF THESE SWITCHES ARE SET CORRECTLY!!!!
	random_forest_switch = False
	error_rate_switch = True
	test_switch = False
	spam_val = False

	input_file, test_input_file, index, label, categorical = census()

	if not spam_val:
		replace_missing_values(input_file, categorical)
		mean, mode = mean_and_mode(input_file)
		impute(input_file, mean, mode, categorical)
		random.shuffle(input_file)
	else:
		input_file = shuffle(input_file)
	train = input_file[index:]
	validation = input_file[:index]
	train_df = pd.DataFrame(train)
	train_d = pd.get_dummies(train_df)
	validation_df = pd.DataFrame(validation)
	validation_d = pd.get_dummies(validation_df)

	if test_switch:
		replace_missing_values(test_input_file)
		test_mean, test_mode = mean_and_mode(test_input_file)
		impute(test_input_file, test_mean, test_mode)
		test_df = pd.DataFrame(test_input_file)
		test_d = pd.get_dummies(test_df)

	if random_forest_switch:
		all_classifiers = rf_train(train_d, 30, 18)
		predict = rf_predict(all_classifiers, validation_d)
		print("RF PREDICT")
		print(predict[:50])
	else:
		x_values = []
		y_values = []
		for i in range(21):
			if i == 0:
				continue
			x_values.append(i)
			classifier = DecisionTree()
			classifier.root = classifier.train(train_d, i)
			predict = classifier.predict(validation_d)
			total = 0
			error = 0
			validation_labels = validation_d[[label]]
			a = validation_labels[label]
			i = 0
			for v in validation_labels[label]:
				if v != predict[i]:
					error += 1
				total += 1
				i += 1
			print("ERROR RATE:")
			print(1 - (error/total))
			y_values.append(1 - (error/total))
		plot.plot(x_values, y_values, label = "census graphs")
		plot.legend()
		plot.grid()
		plot.xlabel("Depth")
		plot.ylabel("Accuracy")
		plot.savefig("census_tests")
	if error_rate_switch:
		# CALCULATE ERROR RATE
		total = 0
		error = 0
		validation_labels = validation_d[[label]]
		a = validation_labels[label]
		i = 0
		for v in validation_labels[label]:
			if v != predict[i]:
				error += 1
			total += 1
			i += 1
		print("ERROR RATE:")
		print(error/total)
	else:
		# WRITE TO CSV
		predictions = [["id", "category"]]
		i = 0
		for v in predict:
			predictions += [[i, v]]
			i += 1
		with open('spam_test_predictions.csv', 'w') as f:
			writer = csv.writer(f)
			writer.writerows(predictions)

if __name__ == "__main__":
	main()