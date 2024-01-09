import pandas
import pyarrow
import pyarrow.csv

def get_used_car_prices_dataset():
    raw_data = pyarrow.csv.read_csv('../data/used_car_prices/cardekho_data.csv')

    #convert pyarrow table to pandas, maps to existing data 
    dataset = raw_data.to_pandas(split_blocks=True)

    print(dataset) 
    print(dataset.dtypes)

get_used_car_prices_dataset()