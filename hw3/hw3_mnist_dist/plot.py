# 100 0.2212
# 200 0.1755
# 500 0.1034
# 1000 .0814
# 2000 .0661
# 5000 .047
# 10000 .0446
# 30000 .0408
# 50000 .0406

import matplotlib.pyplot as plot

# x_values = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
# y_values = [0.2212, 0.1755, 0.1034, 0.0814, 0.0661, 0.047, 0.0446, 0.0408, 0.0406]
x_values = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
y_values = [0.3073, 0.2423, 0.1736, 0.1528, 0.1327, 0.119, 0.1143, 0.115, 0.1143]
plot.plot(x_values, y_values, label = "MNIST Training LDA")
plot.legend()
plot.grid()
plot.xlabel("Training Points")
plot.ylabel("Error Rate")
plot.savefig("MNIST_Output")