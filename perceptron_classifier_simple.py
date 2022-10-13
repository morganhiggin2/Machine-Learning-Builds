import numpy as nu  
import pandas
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plot 

data = pandas.read_csv('data/simple_classifier_2d.py', index_col=False)
column_names = data.columns

'''datasets = train_test_split(data, 
                            labels,
                            test_size=0.2)

train_data, test_data, train_labels, test_labels = datasets'''

model = Perceptron(random_state=11, max_iter=20, penalty='l1', alpha=0.001)
model.fit(data[column_names[0:2]], data[column_names[2]])

predicted_targets = model.predict(data[column_names[0:2]].to_numpy())

available_colors = ['red', 'blue', 'green', 'orange', 'pink', 'black']

def colors(targets):
    n = targets.shape[0]
    colors = [''] * n 

    for i in range(len(targets)):
        colors[i] = available_colors[targets[i] % n] 

    return colors

figure, axs = plot.subplots(2)
axs[0].scatter(data[column_names[0]], data[column_names[1]], c=colors(data[column_names[-1]]))
axs[1].scatter(data[column_names[0]], data[column_names[1]], c=colors(predicted_targets))

print(model.score(data[column_names[0:2]], data[column_names[2]]))

plot.show()
