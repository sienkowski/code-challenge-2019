import os

import pandas as pd
import numpy as np

from utils import processing


# Get datatypes of each column.
datatypes = {
    "Unnamed: 0": "int64",
    "country": "object",
    "description": "object",
    "designation": "object",
    "points": "int64",
    "price": "float64",
    "province": "object",
    "region_1": "object",
    "region_2": "object",
    "taster_name": "object",
    "taster_twitter_handle": "object",
    "title": "object",
    "variety": "object",
    "winery": "object"
}

# read in dataframe
df = pd.read_csv('../data_root/raw/wine_dataset.csv', dtype=datatypes)

# get description and column fields out of dataframe
df_keep = df[['description', 'points']].loc[:]

# transform dataframe
df_keep = processing(df_keep, col="description")

# save dataframe in feather format
os.makedirs('tmp', exist_ok=True)
df_keep.to_feather('tmp/df_keep')
