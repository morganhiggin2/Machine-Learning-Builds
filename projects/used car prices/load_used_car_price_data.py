import pandas
import numpy
import pyarrow.csv

def get_used_car_prices_dataset():
    dataset = pandas.read_csv('../data/used_car_prices/cardekho_data.csv')#pyarrow.csv.read_csv('../data/used_car_prices/cardekho_data.csv')

    #encode Fuel_Type,Seller_Type,Transmission
    fuel_type_class_labels = {label: i[0] for i, label in numpy.ndenumerate(dataset['Fuel_Type'].unique())}
    dataset['Fuel_Type_Encoding'] = dataset['Fuel_Type'].apply(lambda label: fuel_type_class_labels[label])
    seller_type_class_labels = {label: i[0] for i, label in numpy.ndenumerate(dataset['Seller_Type'].unique())}
    dataset['Seller_Type_Encoding'] = dataset['Seller_Type'].apply(lambda label: seller_type_class_labels[label])
    transmission_class_labels = {label: i[0] for i, label in numpy.ndenumerate(dataset['Transmission'].unique())}
    dataset['Transmission_Encoding'] = dataset['Transmission'].apply(lambda label: transmission_class_labels[label])

    return dataset

get_used_car_prices_dataset()