import pandas
import numpy as np
import torch
import matplotlib.pyplot as plot
import sklearn.neighbors

data = pandas.read_csv('data/kernal_1.csv', index_col=False)
n = data.shape[0]
d = data.shape[1]
def kernal_function(u):
    if (np.abs(u) <= 1/2):
        return 1
    else:
        return 0

def p_kernal_simple(x, h):
    return (1 / n) * ((1 / h**d) * ((x - data) / h).applymap(kernal_function)).sum()

x_values = np.arange(0, 14, 1)
y_values = list(map(p_kernal_simple, x_values, np.full(15, 5)))

#plot.plot(x_values, y_values)

#use scikit learn for kernal, both top-hat (boxed) and gaussian
boxed_kernel = sklearn.neighbors.KernelDensity(kernel='tophat', bandwidth=5).fit(data['value'].to_numpy().reshape(-1, 1))
y_values = boxed_kernel.score_samples(x_values.reshape(-1, 1))
y_values = np.exp(y_values)

plot.plot(x_values, y_values)
#plot.show()

gaussian_kernel = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=5).fit(data['value'].to_numpy().reshape(-1, 1))
y_values = gaussian_kernel.score_samples(x_values.reshape(-1, 1))
y_values = np.exp(y_values)

plot.plot(x_values, y_values)
plot.show()
