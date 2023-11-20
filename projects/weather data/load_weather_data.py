import pyarrow
import pandas
from scipy.stats import vonmises 
import math

def get_weather_dataset_pyarrow():
    dtype = {'date': 'string[pyarrow]', 'ind.0': 'bool[pyarrow]', 'rain': 'float32[pyarrow]', 'ind.1': 'bool[pyarrow]', 'temp': 'float32[pyarrow]', 'ind.2': 'bool[pyarrow]', 'wetb': 'float32[pyarrow]', 'dewpt': 'float32[pyarrow]', 'vappr': 'float32[pyarrow]', 'rhum': 'int8[pyarrow]', 'msl': 'float32[pyarrow]'}
    dataset = pandas.read_csv('../data/weather_data/hly175.csv', skiprows=15, nrows=10, dtype=dtype)

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
    classification_weights = {'rainy': rainy_days / total_days, 'clear': clear_days / total_days}

    # we want to know the probability of it raining based on a. the time of year and b. the temperature outside (although the two are somewhat correlted, we will use both to be comprehensive)
    #dataset['given_year_delta'].astype('int64[pyarrow]')
    given_year_delta_seconds = dataset['given_year_delta'].astype('int64[pyarrow]')
    temp_rounded_int = dataset['temp'].floor().astype('int64[pyarrow]')

    #vectorize as numpy
    X = pandas.Dataset(data=[given_year_dalta_seconds, temp_rounded_int], columns=['given_year_delta_seconds', 'temp_rounded_int'])

    x = X.to_numpy()
    y = dataset['percipitation_classifier'].to_numpy()

    clf = RidgeClassifier(max_iter=100, class_weight=classification_weights, solver='svd').fit(x, y)


rainy_classification()

#dtypes = [pandas.ArrowDType(pyarrow.string())]
