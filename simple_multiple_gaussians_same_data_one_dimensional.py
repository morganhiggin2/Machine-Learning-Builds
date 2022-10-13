# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 23:20:09 2022

@author: morgan
"""

#import numpy as np

import os
import sys

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import math

#-----------------------------------------

#data points
x = tf.constant([1.0, 2.0, 2.0, 5.0, 4.0, 3.0, 3.0, 3.0, 5.0, 3.0, 7.0, 3.0, 3.0, 2.0, 2.0, 5.0, 4.0])
y = tf.constant([5.0, 7.0, 3.0, 6.0, 3.0, 4.0, 6.0, 8.0, 5.0, 5.0, 5.0])

#gaussian function
def gaus(mean, std, x):
    return (tf.math.sqrt(1.0 / (2.0 * math.pi * tf.math.pow(std, 2))) * tf.math.exp((-1 / (2 * std)) * tf.math.pow(x - mean, 2)))



#p(x)
#most likelyhood mean
x_u_ml = tf.reduce_sum(x, axis=0) / tf.cast(tf.shape(x)[0], tf.float32)

#standard deviation
x_std_ml = tf.math.reduce_std(x, axis=0)



#p(y)
#most likelyhood mean
y_u_ml = tf.reduce_sum(y, axis=0) / tf.cast(tf.shape(y)[0], tf.float32)

#standard deviation
y_std_ml = tf.math.reduce_std(y, axis=0)


#p(x+y)

x_y_concat = tf.concat([x, y], axis=0)

x_y_u_ml = tf.reduce_sum(x_y_concat, axis=0) / tf.cast(tf.shape(x_y_concat)[0], tf.float32)

#standard deviation
x_y_std_ml = tf.math.reduce_std(x_y_concat, axis=0)



unique_x, x_indicies, x_counts = tf.unique_with_counts(x)
unique_y, y_indicies, y_counts = tf.unique_with_counts(x)
unique_x_y, x_y_indicies, x_y_counts = tf.unique_with_counts(x)

#plot
#get range of x values for plotting data
plot_values = tf.range(0, limit=10, delta=0.1)

#plot data
plt.plot(plot_values, gaus(x_u_ml, x_std_ml, plot_values), label='p(x)', color='blue')
plt.plot(plot_values, gaus(y_u_ml, y_std_ml, plot_values), label='p(y)', color='orange')
plt.plot(plot_values, gaus(x_y_u_ml, x_y_std_ml, plot_values), label='p(x+y)', color='green')
plt.scatter(unique_x, tf.cast(x_counts, tf.float32) / tf.cast(tf.shape(x)[0], tf.float32), color='blue')
plt.scatter(unique_y, tf.cast(y_counts, tf.float32) / tf.cast(tf.shape(y)[0], tf.float32), color='orange')
plt.scatter(unique_x_y, tf.cast(x_y_counts, tf.float32) / tf.cast(tf.shape(x_y_concat)[0], tf.float32), color='green')
plt.show()

print(unique_x)
print(x_indicies)
print(x_counts)