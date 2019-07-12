from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix


@click.command()
@click.option('--x-test')
@click.option('--y-test')
@click.option('--model-path')
@click.option('--in-dir')
@click.option('--out-dir')
def evaluate_model(x_test, y_test, model_path, in_dir, out_dir):
    in_dir = Path(in_dir)
    
    X_test = pd.read_csv(in_dir / x_test)
    y_test = pd.read_csv(in_dir / y_test)
    model = Path(model_path)
    model = load(model)

    out_dir =Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_full = out_dir / 'confusion_matrix.png'
    
    prediction = model.predict(X_test)
    score = accuracy_score(prediction, y_test)
    
    data = confusion_matrix(y_test, prediction)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    df_cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    
    plt.figure(figsize = (10,7))
    plt.title('Accuracy Score is: {} %'.format(round(score*100, 2)))
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})
    plt.savefig(out_dir_full)


if __name__ == "__main__":
    evaluate_model()
