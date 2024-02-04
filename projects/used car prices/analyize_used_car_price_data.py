'''
ideas:

classify cheap vs expensive cars into 1? 2? 3? classes? compare intra class variance. minimze variance amoung these
    - better way to analyze variance? trade off for lower variance vs class? loss function?

year vs average car price
 - kernel ridge regression on gaussian kernel vs guassian process for regression

decission trees for classification of cars?...

group average used car price per year, and make marcov chain with transision matrix,
each transition going between nodex (latent variable), with x, output, being the price (the actual year is irrelevent, since we are only concern with one year progression)
  z1 -> z2 -> z3
   |     |     |
   $     $     $
 - derive the transition matrix for this marcov chain


'''

import numpy
import matplotlib.pyplot as plot
import matplotlib.colors
from load_used_car_price_data import get_used_car_prices_dataset

def price_classification(dataset): 
    '''
    lets find if there are 3 classes of cars based on selling price and mileage 
    '''

    from sklearn.cluster import KMeans

    #get numpy arrays from data
    X = dataset[['Kms_Driven', 'Selling_Price']].to_numpy()

    #k means clustering
    k_means_cluster = KMeans(n_clusters=3, random_state=0, n_init="auto", init="random").fit(X)

    Y_predictions = k_means_cluster.predict(X)
    color_dictionary = {0: 'green', 1: 'blue', 2: 'yellow'}
    Y_prediction_color_labels = numpy.vectorize(color_dictionary.get)(Y_predictions)

    plot.style.use('ggplot')
    matplotlib.use('TkAgg')
    plot.scatter(X[:,0], X[:,1], c=Y_predictions.tolist(), cmap=matplotlib.colors.ListedColormap(['blue', 'green', 'yellow']))
    plot.xlabel("Selling Price")
    plot.ylabel("Mileage (Kilometers)")
    plot.show()
    #Graph out class prediction for each training point, graph

dataset = get_used_car_prices_dataset()
price_classification(dataset)