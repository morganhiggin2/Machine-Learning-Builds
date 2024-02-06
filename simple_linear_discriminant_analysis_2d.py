from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  
import numpy as np  
import pandas     

data = pandas.read_csv('data/simple_classifier_2d.py', index_col=False)  

lda_machine = LinearDiscriminantAnalysis()                      
lda_machine.fit(data[['x', 'y']].to_numpy(), data['t'].to_numpy())                                                                               
print(lda_machine.predict([[3,1]]))
