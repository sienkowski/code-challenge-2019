import pandas as pd
import numpy as np

import joblib
import click

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion


from sklearn.linear_model import LogisticRegression
from utils import TextSelector, NumberSelector, processing


pipeline = joblib.load('tmp/lr_model.pkl')


@click.command()
@click.option('--text', default="fruity", help='text describing wine')
def wine_class(text, pipeline=pipeline):
    input = {
        'description': [text],
        'points': [np.nan]
    }
    df = pd.DataFrame(data=input)

    df_keep = processing(df, col="description")

    engineered_features = [
        f for f in df_keep.columns.values if f not in ['description', 'points']]

    # print(df_keep[engineered_features].head())
    df_keep = df_keep[engineered_features].loc[:]

    cat = ''.join(pipeline.predict(df_keep))

    print(f"This will be a(an) `{cat}` wine")


#X_train = processing(X_train)

if __name__ == "__main__":
    wine_class()
