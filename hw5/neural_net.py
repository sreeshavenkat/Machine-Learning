import pandas as pd
import csv
import scipy.io as sio
import numpy as np
import random
from Decision_Tree import DecisionTree
import matplotlib.pyplot as plot
from sklearn.utils import shuffle

def main():	
	letters_data = sio.loadmat("letters_data.mat")
	print(letters_data)
if __name__ == "__main__":
	main()