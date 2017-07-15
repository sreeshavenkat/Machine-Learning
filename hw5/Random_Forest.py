from Decision_Tree import DecisionTree
import numpy as np

def train(self, data, forest_size, tree_depth):
	classifiers = []
	for i in range(forest_size):
		forest_data = data
		tree = DecisionTree()
		tree.root = tree.train(forest_data, depth, True)
		classifiers.append()
	return classifiers

def predict(self, classifiers, data):
	predicted_label = 0
	for c in classifiers:
		predicted_label += int(c.predict([data, ])[0])
	return np.round(predicted_label / len(classifiers))