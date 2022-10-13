# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 10:10:52 2022

@author: morga
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy
import math

#-----------------------------------

#function for distrubiton
def binary(u, x):
    return numpy.power(u, x) * numpy.power(1 - u, 1 - x)

#probility of variable occurance
u = 0.49

#data points
x_values = numpy.arange(1, 11, 1)

#print distrubution
plt.plot(x_values, binary(u, x_values))
plt.show();