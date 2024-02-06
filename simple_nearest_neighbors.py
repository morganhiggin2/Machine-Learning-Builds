import numpy
import pandas
import sklearn.neighbors

#import data
data = pandas.read_csv('data/classed_data_1.csv', index_col=False)

#get numpy array
values = data['value'].to_numpy().reshape(-1, 1)
classes = data['class'].to_numpy()

#fit data to nearest neighbors
neighbors = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
neighbors.fit(values, classes)

print(neighbors.predict([[3.5]]))
