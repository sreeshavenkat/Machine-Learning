import numpy as np

a = np.array([1,2,3,4,5,6])
print(a.shape)
print(a.ndim)
print(a)
a = np.matrix(a)
print(a.shape)
b = np.transpose(a)
print(np.multiply(a,b))

# var = []
# while j < 10:
#  	var.append(multivariate_normal(mean_vector[j], covariance_vector[j] + identity))
# plt.pcolor(covariance_vector[j])
# plt.colorbar()
# plt.yticks(np.arange(0.5, 10.5), range(0, 10))
# plt.xticks(np.arange(0.5, 10.5), range(0, 10))
# plt.show()