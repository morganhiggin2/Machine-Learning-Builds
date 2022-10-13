# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:12:01 2022

@author: morgan
"""

#import numpy as np

import os
import sys

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
import matplotlib.pyplot as plt
import math

#-----------------------------------------

#data points
x = tf.constant([1.0, 2.0, 2.0, 5.0, 4.0, 3.0, 3.0, 3.0, 5.0, 3.0, 7.0, 3.0, 3.0, 2.0, 2.0, 5.0, 4.0])

#most likelihood mean
u_ml = tf.reduce_sum(x, axis=0) / tf.cast(tf.shape(x)[0], tf.float32)

#standard deviation
std_ml = tf.math.reduce_std(x, axis=0)

#compute expectation value of x
exp_val_x = tf.reduce_sum(x) / tf.cast(tf.shape(x)[0], dtype=tf.float32)

#compute expectation value of x^2
exp_val_x_2 = tf.reduce_sum(tf.pow(x, 2)) / tf.cast(tf.shape(x)[0], dtype=tf.float32)

#compute var[x]
var_x = exp_val_x_2 - tf.pow(exp_val_x, 2)
