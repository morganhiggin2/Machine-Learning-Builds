import pyarrow
import pandas
from scipy.stats import vonmises 
import math
import numpy

TESTING = False 

def get_weather_dataset_pyarrow():
    dtype = {'date': 'string[pyarrow]', 'ind.0': 'int8[pyarrow]', 'rain': 'float32[pyarrow]', 'ind.1': 'int8[pyarrow]', 'temp': 'float32[pyarrow]', 'ind.2': 'int8[pyarrow]', 'wetb': 'float32[pyarrow]', 'dewpt': 'float32[pyarrow]', 'vappr': 'float32[pyarrow]', 'rhum': 'int8[pyarrow]', 'msl': 'float32[pyarrow]'}
    read_csv_args = {'filepath_or_buffer': '../data/weather_data/hly175.csv', 'skiprows': 15, 'dtype': dtype, 'on_bad_lines': 'skip', 'skip_blank_lines': True, 'na_values': [" "], 'encoding_errors': 'ignore', 'nrows': 10000}

    if TESTING:
        read_csv_args['nrows'] = 10

    dataset = pandas.read_csv(**read_csv_args)

    dataset = dataset.dropna()

    #convert indicator columns to bools
    dataset['ind'] = dataset['ind'].astype('bool[pyarrow]')
    dataset['ind.1'] = dataset['ind.1'].astype('bool[pyarrow]')
    dataset['ind.2'] = dataset['ind.2'].astype('bool[pyarrow]')

    #convert date column from string[pyarrow] to date64[pyarrow] type
    #dataset['date_converted'] = pandas.Series(pyarrow.compute.strptime(dataset['date'].values,  '%Y-%b-%m %H%M', 's'), dtype='timestamp[s][pyarrow]')
    pandas.to_datetime(dataset['date'], format='%d-%b-%Y %H:%M')
    dataset['date_converted'] = pandas.to_datetime(dataset['date'], format='%d-%b-%Y %H:%M').astype(dtype=pandas.ArrowDtype(pyarrow.timestamp('s')))

    #get timestamp of the first day of the year for each timestamp
    date_year_converted = pandas.to_datetime('01-jan-' + dataset['date_converted'].dt.year.astype('string[pyarrow]') + ' 00:00', format='%d-%b-%Y %H:%M').astype(dtype=pandas.ArrowDtype(pyarrow.timestamp('s')))

    time_delta = dataset['date_converted'] - date_year_converted
    dataset['given_year_delta'] = time_delta

    return dataset

def yearly_weather_regression():
    dataset = get_weather_dataset_pyarrow()


    theta = (dataset['given_year_delta']/dataset['given_year_delta'].max() * 2 * math.pi).to_numpy()
    temperature = dataset['temp'].to_numpy()

def rainy_classification():
    '''
    based on rainy days and clear days, create a model that can predict based on time of year and tempurature 
    weither or not it will rain 
    '''
    from sklearn.linear_model import RidgeClassifier

    dataset = get_weather_dataset_pyarrow() 
    # get rid of rows that don't have values (where rain indicator is 0)
    dataset = dataset[dataset['ind'] == False]

    # weither the day was rainy or not
    dataset['percipitation_classifier'] = dataset['rain'].apply(lambda val: True if val > 0.0 else False).astype('bool[pyarrow]')

    # number of rainy and clear days
    rainy_days = dataset[dataset['percipitation_classifier'] == True].shape[0]
    clear_days = dataset.shape[0] - rainy_days 
    total_days = rainy_days + clear_days # or could have been total shape

    # weights to each class, class 1 being rainy days, class 2 being clear days
    classification_weights = {0: rainy_days / total_days, 1: clear_days / total_days}

    # we want to know the probability of it raining based on a. the time of year and b. the temperature outside (although the two are somewhat correlted, we will use both to be comprehensive)
    #dataset['given_year_delta'].astype('int64[pyarrow]')
    given_year_delta_seconds = dataset['given_year_delta'].astype('int64[pyarrow]')
    temp_rounded_int = dataset['temp'].apply(pyarrow.compute.floor).astype('int64[pyarrow]')

    #vectorize as numpy
    X = pandas.DataFrame(data={'given_year_delta_seconds': given_year_delta_seconds, 'temp_rounded_int': temp_rounded_int})
    x = X.to_numpy(dtype='i4')#(dtype=[('given_year_delta_seconds', '<i8'), ('temp_rounded_int', '<i8')])
    
    # rainy, clear
    # False, True
    # ...
    y = pandas.DataFrame(data={'rainy': dataset['percipitation_classifier'], 'clear': ~dataset['percipitation_classifier']}, dtype='bool[pyarrow]')

    y = y.to_numpy(dtype='?')
    
    #TODO add class weights
    clf = RidgeClassifier(max_iter=100, solver='svd').fit(x, y)

    print(clf.predict(numpy.array([(19616400, 9)])))

def rainy_classification_logistic():
    '''
    based on rainy days and clear days, create a model that can predict based on time of year and tempurature 
    weither or not it will rain.
    similar to rainly_classification but logistic. 
    '''
    from sklearn.linear_model import LogisticRegression 

    dataset = get_weather_dataset_pyarrow() 
    # get rid of rows that don't have values (where rain indicator is 0)
    dataset = dataset[dataset['ind'] == False]

    # weither the day was rainy or not
    dataset['percipitation_classifier'] = dataset['rain'].apply(lambda val: True if val > 0.0 else False).astype('bool[pyarrow]')

    # number of rainy and clear days
    rainy_days = dataset[dataset['percipitation_classifier'] == True].shape[0]
    clear_days = dataset.shape[0] - rainy_days 
    total_days = rainy_days + clear_days # or could have been total shape

    # weights to each class, class 1 being rainy days, class 2 being clear days
    classification_weights = {0: rainy_days / total_days, 1: clear_days / total_days}

    # we want to know the probability of it raining based on a. the time of year and b. the temperature outside (although the two are somewhat correlted, we will use both to be comprehensive)
    #dataset['given_year_delta'].astype('int64[pyarrow]')
    given_year_delta_seconds = dataset['given_year_delta'].astype('int64[pyarrow]')
    temp_rounded_int = dataset['temp'].apply(pyarrow.compute.floor).astype('int64[pyarrow]')

    #vectorize as numpy
    X = pandas.DataFrame(data={'given_year_delta_seconds': given_year_delta_seconds, 'temp_rounded_int': temp_rounded_int})
    x = X.to_numpy(dtype='i4')#(dtype=[('given_year_delta_seconds', '<i8'), ('temp_rounded_int', '<i8')])
    
    # rainy, clear
    # False, True
    # ...
    y = pandas.DataFrame(data={'rainy': dataset['percipitation_classifier']}, dtype='bool[pyarrow]')

    y = y.to_numpy(dtype='?')
    y = numpy.ravel(y)
    
    #how to encode target values in mutlinomial? one hot encoded
    clf = LogisticRegression(penalty=None, max_iter=100, solver='lbfgs', multi_class='multinomial').fit(x, y)

    print(clf.classes_)
    print(clf.predict(numpy.array([(19616400, 9)])))
    print(clf.predict_proba(numpy.array([(19616400, 9)])))

def get_rainy_clear_clustering():
    '''
    determine the classification crateria on weither or not it will rain 
    '''
    from sklearn.mixture import GaussianMixture

    dataset = get_weather_dataset_pyarrow()

    # get rid of rows that don't have values (where rain indicator is 0)
    dataset = dataset[dataset['ind'] == False]

    # get rain data for a day 
    x = dataset['rain'].to_numpy(dtype='f4').reshape(-1, 1)

    cluster_model = GaussianMixture(n_components=2).fit(x)

    # lets get classification for a few test values
    test_values = [0.0, 0.01, 0.2, 4.5]

    for test_value in test_values:
        x_sample = numpy.array([[test_value]])
        classification = cluster_model.predict(x_sample)
        classification_prob = cluster_model.predict_proba(x_sample)[0][classification]
        print(f"classification for percipitation amount {test_value} is {classification} with probability of {numpy.round(classification_prob * 100, 2)}%")
    
    # get classification mark by determining the points where both classifiers are equally likely

def bold_classifier_svm():
    '''
    use all the dimensions of the data to predict weither it will be rainy or not 
    '''
 
    dataset = get_weather_dataset_pyarrow()

    # get rid of rows that don't have values (where rain indicator is 0)
    dataset = dataset[dataset['ind'] == False]
    dataset = dataset[dataset['ind.1'] == False]
    dataset = dataset[dataset['ind.2'] == False]

    # weither the day was rainy or not
    dataset['percipitation_classifier'] = dataset['rain'].apply(lambda val: True if val > 0.0 else False).astype('bool[pyarrow]')

    #only use 50 random data points
    n_points = 100
    n_predictions = 5

    import random

    rand_indicies = list(random.randint(0, dataset.shape[0]) for n in range(0, n_points))
    subset_dataset = dataset[dataset.index.isin(rand_indicies)]


    from sklearn.svm import SVC

    svm = SVC(kernel='linear', class_weight='balanced', max_iter=50000)

    # get rain data for a day 
    x = subset_dataset[['rain', 'temp', 'wetb','dewpt', 'vappr', 'rhum', 'msl']].to_numpy(dtype='f')#.reshape(-1, 1)

    # rainy, clear
    # False, True
    # ...
    y = pandas.DataFrame(data={'rainy': subset_dataset['percipitation_classifier']}, dtype='bool[pyarrow]')

    y = y.to_numpy(dtype='?')
    y = numpy.ravel(y)

    classifier = svm.fit(x, y)

    #get other random rows from the dataset

    #use these to predict values
    rand_indicies = list(random.randint(0, dataset.shape[0]) for n in range(0, n_predictions))
    predict_dataset = dataset[dataset.index.isin(rand_indicies)]

    print(rand_indicies)

    pred_x = predict_dataset[['rain', 'temp', 'wetb', 'dewpt', 'vappr', 'rhum', 'msl']].to_numpy(dtype='f')#.reshape(-1, 1)

    print(str(['rain', 'temp', 'wetb', 'dewpt', 'vappr', 'rhum', 'msl']))
    print(pred_x)
    print()
    print(classifier.predict(pred_x))

    #TODO can we see correlations factors? what contributed the most?

#TODO do multiclass regression just like rainy_classification(), but with LogisticRegression

bold_classifier_svm()

#get_rainy_clear_clustering()

#dtypes = [pandas.ArrowDType(pyarrow.string())]