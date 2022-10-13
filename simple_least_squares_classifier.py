from sklearn.kernel_ridge import KernelRidge
import numpy as np 
import pandas 

data = pandas.read_csv('data/simple_classifier_2d.py', index_col=False)

kernel_ridge = KernelRidge(alpha=1.0)
kernel_ridge.fit(data[['x', 'y']].to_numpy(), data['t'].to_numpy())

print(kernel_ridge.predict([[9,8]]))

#this is a kenel regression model for fitting a model though lines, fits more to chapter 6
