import pickle

import pandas as pd

from rating_model import PICKLE_PROTOCOL

def dump_pickle(file_path, obj):
    with open(file_path, "wb") as dump_file:
        pickle.dump(obj, dump_file, protocol=PICKLE_PROTOCOL)
        
def load_pickle(file_path):
    with open(file_path, "rb") as dump_file:
        return pickle.load(dump_file)
    
def optimize_dataframe_numeric_dtypes(dataframe):
    numeric_columns = dataframe.select_dtypes("number").columns
    dataframe[numeric_columns] = dataframe[numeric_columns].apply(lambda x: pd.to_numeric(x, downcast='integer'))