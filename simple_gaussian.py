# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:12:01 2022

@author: morgan
"""

#import numpy as np

import os
import sys

import matplotlib.pyplot as plt
import numpy
import math

#-----------------------------------------

#data points
x = numpy.array([1.0, 2.0, 2.0, 5.0, 4.0, 3.0, 3.0, 3.0, 5.0, 3.0, 7.0, 3.0, 3.0, 2.0, 2.0, 5.0, 4.0])

#most likelyhood mean
u_ml = numpy.sum(x) / x.shape[0]

#standard deviation
std_ml = numpy.std(x, axis=0)

#gaussian function
def gaus(mean, std, x):
    return (numpy.sqrt(1.0 / (2.0 * math.pi * numpy.power(std, 2))) * numpy.exp((-1 / (2 * std)) * numpy.power(x - mean, 2)))

#get range of x values for plotting data
plot_x_values = numpy.arange(0, 10, 0.1, dtype=numpy.float32)

#plot data
plt.plot(plot_x_values, gaus(u_ml, std_ml, plot_x_values))
plt.show()