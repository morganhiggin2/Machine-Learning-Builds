import numpy 
from numpy.random import shuffle as numpy_shuffle
from math import floor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import log_loss

def get_data():
    raw_data = numpy.genfromtxt("../data/heart_attack_prediction/heart.csv", delimiter=",", skip_header=1)
    numpy_shuffle(raw_data) 
    input = raw_data[:, list(range(0, raw_data.shape[1] - 1))]
    input =  raw_data.astype(numpy.float32)

    #turn into 2d with class labels
    output = raw_data[:, raw_data.shape[1] - 1]
    output = output.astype(numpy.float32)

    return input, output

classifier = GaussianProcessClassifier()
X, Y = get_data()

deciding_index = int(floor(len(X) * 0.8))
X_train, X_test = numpy.split(X, [deciding_index])
Y_train, Y_test = numpy.split(Y, [deciding_index])

classifier.fit(X_train, Y_train)

Y_predictions = classifier.predict_proba(X_test)
avg_loss = log_loss(Y_test, Y_predictions)

print(avg_loss)

#TODO for sure determine default weights for torch's cross entropy loss