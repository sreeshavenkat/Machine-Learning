import numpy
from scipy import io

def impurity(left_label_hist, right_label_hist):
	left = sum(left_label_hist)
	right = sum(right_label_hist)
	l_0_label = left_label_hist[0]/float(left)
	l_1_label = left_label_hist[1]/float(left)
	r_0_label = right_label_hist[0]/float(right)
	r_1_label = right_label_hist[1]/float(right)
	