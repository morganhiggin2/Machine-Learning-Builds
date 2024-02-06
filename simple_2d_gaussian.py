# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 19:28:30 2022

@author: morga
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:12:01 2022

@author: morgan
"""

#rewrite as numpy

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
x1 = tf.constant([1.0, 4.0, 3.0, 5.0, 3.0, 5.0, 1.0, 2.0])
x2 = tf.constant([9.0, 5.0, 9.0, 7.0, 11.0, 10.0, 9.0, 8.0])

x = tf.stack([x1, x2], axis=1)


#most likelyhood mean
u_ml = tf.reduce_sum(x, axis=0) / tf.cast(tf.shape(x)[0], tf.float32)

#standard deviation
std_ml = tf.math.reduce_std(x, axis=0)

#covariance matrix
cov_ml = tfp.stats.covariance(x);
precission_ml= tf.linalg.inv(cov_ml)

#gaussian function for 1d
def gaus1d(mean, std, x):
    return (tf.math.sqrt(1.0 / (2.0 * math.pi * tf.math.pow(std, 2))) * tf.math.exp((-1 / (2 * std)) * tf.math.pow(x - mean, 2)))

#gaussian function for 2d
def gaus2d(mean, precission, x):
    return ((tf.math.sqrt(tf.linalg.det(precission)) / (tf.pow(2.0 * math.pi, tf.cast(tf.shape(x)[0], tf.float32) / 2.0))) * tf.math.exp(-0.5 * tf.matmul(tf.matmul((x - mean), precission), (x - mean), transpose_b=True)))

#grid dimensions
width = 2
height = 2

#make grid points for 2d height map
x_points = tf.range(0, limit=width, delta=1)
y_points = tf.range(0, limit=height, delta=1)

#initialize empty matrix with dimensions
grid_points = tf.reshape(tf.convert_to_tensor(()), (0, 2))
gaus_results = tf.zeros((tf.keras.backend.get_value(tf.cast(tf.shape(x_points)[0], tf.int32)), tf.keras.backend.get_value(tf.cast(tf.shape(y_points)[0], tf.int32)))) #tf.reshape(tf.convert_to_tensor(()), (tf.keras.backend.get_value(tf.shape(x_points)[0]), tf.keras.backend.get_value(tf.shape(y_points)[0])))

for x in x_points:
    for y in y_points:
        grid_points = tf.concat([grid_points, [[x, y]]], axis=0)
        
for x in tf.range(0.0, tf.keras.backend.get_value(tf.cast(tf.shape(x_points)[0], tf.int32))):
    for y in tf.range(0.0, tf.keras.backend.get_value(tf.cast(tf.shape(y_points)[0], tf.int32))):
        gaus_results[x][y] = gaus2d(u_ml, precission_ml, [[x_points[x], y_points[y]]])

print(grid_points)
print(gaus2d(u_ml, precission_ml, grid_points))

#initialize empty matrix with dimensions
#gaus_results = tf.reshape(tf.convert_to_tensor(()), (0, 2))

#get range of x values for plotting data
#plot_x_values = tf.tensordot(tf.range(0, limit=10, delta=0.1), tf.range(0, limit=10, delta=0.1))
#plot_x_values = tf.tensordot(tf.range(0, limit=10, delta=1), tf.range(0, limit=10, delta=1), axes=1)
#print(plot_x_values)

#map = plt.pcolormesh(x_points, y_points, gaus2d(u_ml, precission_ml, grid_points), vmin=0.0, vmax=1.0, cmap="RdBu_r")
#map = plt.pcolormesh(x_points, y_points, tf.constant([[0.5, 0.0], [0.2, 0.8]]), vmin=0.0, vmax=1.0, cmap="RdBu_r", shading="nearest")
#plt.colorbar(map, )
#plt.show()

#plot data
#plt.plot(plot_x_values, gaus1d(u_ml[0], std_ml[0], plot_x_values))
#plt.plot(plot_x_values, gaus1d(u_ml[1], std_ml[1], plot_x_values))
#plt.plot(plot_x_values, gaus1d(u_ml, std_ml, plot_x_values))
#plt.show()
