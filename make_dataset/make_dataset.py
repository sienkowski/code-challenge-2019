from pathlib import Path

import click
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def _save_datasets(X_train, X_test, y_train, y_test, outdir):
    """Save data sets into nice directory structure and write SUCCESS flag."""
    outdir = Path(outdir)
    out_X_train = outdir / 'X_train.csv'
    out_X_test = outdir / 'X_test.csv'
    out_y_train = outdir / 'y_train.csv'
    out_y_test = outdir / 'y_test.csv'
    flag = outdir / '.SUCCESS'
    y_train.columns = ['rating']
    y_test.columns =['rating']
    X_train.to_csv(out_X_train, index=False)
    X_test.to_csv(out_X_test, index=False)
    y_train.to_csv(out_y_train, index=False, header=['rating'])
    y_test.to_csv(out_y_test, index=False, header=['rating'] )
    
    flag.touch()


@click.command()
@click.option('--in-csv')
@click.option('--in-dir')
@click.option('--out-dir')
def make_datasets(in_csv, in_dir, out_dir):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_dir = Path(in_dir) / in_csv

    # load data as a dask Dataframe if you have trouble with dask
    # please fall back to pandas or numpy
    df = pd.read_csv(str(file_dir))

    # we set the index so we can properly execute loc below
    # df = df.set_index('Unnamed: 0')

    df = df.drop(['Unnamed: 0', 'description', 'designation', 'taster_twitter_handle', 'title'], axis=1)
    df['country'] = df.country.where(df.country=='', 'US')

    df['price'] = df.price.where(df.price=='', df['price'].mean())
    df['region_1'] = df. region_1.where(df.region_1=='', 'Napa Valley')
    df['region_2'] = df.region_2.where(df.region_2=='', 'Central Coast')
    df['province'] = df.province.where(df.province=='', 'California')
    df['taster_name'] = df.taster_name.where(df.taster_name=='', 'Roger Voss')
    
    df['rating'] = 0
    
    df.loc[(df['points']<86), 'rating'] = 1
    df.loc[(df['points']>=86) & (df['points']<91), 'rating'] = 2
    df.loc[(df['points']>=91) & (df['points']<96), 'rating'] = 3
    df.loc[(df['points']>=96), 'rating'] = 4

    # cols = list(df.columns)
    # a, b = cols.index('points'), cols.index('winery')
    # cols[b], cols[a] = cols[a], cols[b]
    # df = df[cols]

    X = df[:].drop(['rating'], axis=1)
    X= pd.get_dummies(X)
    y= df['rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    _save_datasets(X_train, X_test, y_train, y_test, out_dir)
    

if __name__ == '__main__':
    make_datasets()
