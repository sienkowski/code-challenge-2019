from pathlib import Path

import click
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@click.command()
@click.option('--x-train')
@click.option('--y-train')
@click.option('--in-dir')
@click.option('--out-dir')
def train_data(x_train, y_train, in_dir, out_dir):

    in_dir = Path(in_dir)
    X_train = pd.read_csv(str(in_dir / x_train))
    y_train = pd.read_csv(str(in_dir / y_train))
    
    out_dir = Path(out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_full = out_dir / 'model.joblib'

    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train)

    dump(rfc, out_dir_full)


if __name__ == "__main__":
    train_data()
