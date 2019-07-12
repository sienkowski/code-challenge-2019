import click
import dask.dataframe as dd
import numpy as np
from distributed import Client
from pathlib import Path
import pandas as pd


def _save_datasets(train, test, outdir: Path):
    """Save data sets into nice directory structure and write SUCCESS flag."""
    out_train = outdir / 'train.parquet/'
    out_test = outdir / 'test.parquet/'
    flag = outdir / '.SUCCESS'

    train.to_parquet(str(out_train))
    test.to_parquet(str(out_test))

    flag.touch()


@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
def make_datasets(in_csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Connect to the dask cluster
    #c = Client('dask-scheduler:8786')

    # load data as a dask Dataframe if you have trouble with dask
    # please fall back to pandas or numpy
    df = pd.read_csv(in_csv, dtype=datatypes)

    # we set the index so we can properly execute loc below
    df_keep = df[['description', 'points']].loc[:]

    # trigger computation
    n_samples = len(df_keep)

    # TODO: implement proper dataset creation here
    # http://docs.dask.org/en/latest/dataframe-api.html

    # split dataset into train test feel free to adjust test percentage
    idx = np.arange(n_samples)
    test_idx = idx[:n_samples // 10]
    test = df_keep.loc[test_idx]

    train_idx = idx[n_samples // 10:]
    train = df_keep.loc[train_idx]

    _save_datasets(train, test, out_dir)


if __name__ == '__main__':
    make_datasets()
