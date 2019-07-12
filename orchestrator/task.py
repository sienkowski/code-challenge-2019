import logging

import luigi
import os
from pathlib import Path

from util import DockerTask

VERSION = os.getenv('PIPELINE_VERSION', '0.1')


class Debug(DockerTask):
    """Use this task with appropriate image to debug things."""

    @property
    def image(self):
        return 'code-challenge/download-data:0.1'

    @property
    def command(self):
        return [
            'sleep', '3600'
        ]


class DownloadData(DockerTask):
    """Initial pipeline task downloads dataset."""

    fname = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/raw/')
    url = luigi.Parameter(
        default='https://github.com/datarevenue-berlin/code-challenge-2019/'
                'releases/download/0.1.0/dataset_sampled.csv'
    )

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'download_data.py',
            '--name', self.fname,
            '--url', self.url,
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(out_dir/f'{self.fname}.csv')
        )


class MakeDatasets(DockerTask):

    in_csv = luigi.Parameter(default='wine_dataset.csv')
    in_dir = luigi.Parameter(default='/usr/share/data/raw/')
    out_dir = luigi.Parameter(default='/usr/share/data/split_data')

    @property
    def image(self):
        return f'code-challenge/make-dataset:{VERSION}'

    def requires(self):
        return DownloadData()

    @property
    def command(self):
        # TODO: Try to get the input path from self.requires() ;)
        return [
            'python', 'make_dataset.py',
            '--in-csv', self.in_csv,
            '--in-dir', self.in_dir,
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(out_dir / '.SUCCESS')
        )


class TrainModel(DockerTask):

    X_train = luigi.Parameter(default='X_train.csv')
    y_train = luigi.Parameter(default='y_train.csv')
    in_dir = luigi.Parameter(default='/usr/share/data/split_data')
    out_dir = luigi.Parameter(default='/usr/share/data/model')

    @property
    def image(self):
        return f'code-challenge/train-model:{VERSION}'
    
    def requires(self):
        return MakeDatasets()
    
    @property
    def command(self):
        return [
            'python', 'train_model.py',
            '--x-train', self.X_train,
            '--y-train', self.y_train,
            '--in-dir', self.in_dir,
            '--out-dir', self.out_dir
        ]
    
    # TODO: Fix the output
    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(out_dir / 'model.joblib')
        )


class EvaluateModel(DockerTask):

    X_test = luigi.Parameter(default='X_test.csv')
    y_test = luigi.Parameter(default='y_test.csv')
    model_path = luigi.Parameter(default='/usr/share/data/model/model.joblib')
    in_dir = luigi.Parameter(default='/usr/share/data/split_data')
    out_dir = luigi.Parameter(default='/usr/share/data/evaluate_model')

    @property
    def image(self):
        return f'code-challenge/evaluate-model:{VERSION}'
    
    def requires(self):
        return TrainModel()
    
    @property
    def command(self):
        return [
            'python', 'evaluate.py',
            '--x-test', self.X_test,
            '--y-test', self.y_test,
            '--model-path', self.model_path,
            '--in-dir', self.in_dir,
            '--out-dir', self.out_dir
        ]
    
    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(out_dir / 'confusion_matrix.png')
        )
