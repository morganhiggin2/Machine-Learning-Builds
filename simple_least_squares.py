import numpy
import pandas
import torch
import scipy
import sklearn.neighbors
import matplotlib.pyplot as plot

def sin_function(args, x, t):
    return args[0] * numpy.sin(args[1] * x) + args[2] - t

#import data
data = pandas.read_csv('data/sin_1.csv', index_col=False)

#get numpy array form data
x_input = data.index.to_numpy()
y_train = data['value'].to_numpy()

#train data
inital_params = numpy.array([1, 1, 0])
solution = scipy.optimize.least_squares(sin_function, inital_params, args=(x_input, y_train))
residual = solution.x

#plot data
y_sin = sin_function(residual, x_input, 0)

plot.plot(x_input, y_sin)
plot.plot(x_input, y_train, 'o')
plot.show()
