import re

import pandas as pd
import numpy as np
import seaborn as sns

import nltk
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin

stopWords = set(stopwords.words('english'))


class TextSelector(BaseEstimator, TransformerMixin):
    """Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data

    Parameters
    ---------
    key: Name of text column on which transformation is done
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class NumberSelector(BaseEstimator, TransformerMixin):
    """Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data

    Parameters
    ---------
    key: Name of numeric column on which transformation is done
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]


def percentage_empty(df):
    '''Takes a dataframe and calculates the percentage of null values
    in each column

    Parameters
    ----------
    df: Dataframe

    Example of Usage
    ----------------
    >>> percentage_empty(df)

    '''

    return df.isnull().sum().sort_index()/len(df) * 100


def dependent_column_to_categorical(df, column):
    '''Converts the points column into 4 classes. The division is 
    done using information from https://www.wine-searcher.com/wine-scores

    Parameters
    ----------
    df: Dataframe to be converted 
    column: Dependent feature, in our case points

    Example of Usage
    ---------------
    >>> df[column] = df.apply(lambda df:dependent_column_to_categorical(df,column),
                                          axis = 1)
    '''

    if (df[column] >= 80) & (df[column] <= 84):
        return "Good"
    elif (df[column] >= 85) & (df[column] <= 89):
        return "Very good"
    elif (df[column] >= 90) & (df[column] <= 94):
        return "Outstanding"
    elif (df[column] >= 95) & (df[column] <= 100):
        return "Classic"


def plot_var(df, feat):
    '''This plots the distribution of values or features in a column

    Parameters
    ----------
    df: Dataframe to be converted 
    feat: Column whose categories are to be plotted

    Example of usage
    ----------------
    >>> plot_var(df,feat='points')
    '''

    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    sns.countplot(x=feat, data=df)
    return df.loc[:, feat].value_counts()


def processing(df, col):
    '''Chain of transformations for initially processing dataframe

        1. Change dependent variable to categorical

        2. Convert description column to lowercase and remove punctuations

        3. Get total length of sentences from 2

        4. Get total number of words

        5. Get total number of non-stop words

        6. Get the average word length

       Parameters
       ----------
       df: dataframe on which transformation is done
       col: The independent column. In our case, a column with text

       Example of Usage
       ----------------
       >>> df_keep = processing(df_keep,col="description")
    '''

    if df['points'].dtype == 'int64':
        df["points"] = df.apply(lambda df: dependent_column_to_categorical(df, "points"),
                                axis=1)

    df[f'{col}_processed'] = df[col].apply(
        lambda x: re.sub(r'[^\w\s]', '', x.lower()))

    df['length'] = df[f'{col}_processed'].apply(lambda x: len(x))

    df['words'] = df[f'{col}_processed'].apply(lambda x: len(x.split(' ')))

    df['words_not_stopword'] = df[f'{col}_processed'].apply(
        lambda x: len([t for t in x.split(' ') if t not in stopWords]))

    df['avg_word_length'] = df[f'{col}_processed'].apply(lambda x: np.mean([len(t) for t in x.split(
        ' ') if t not in stopWords]) if len([len(t) for t in x.split(' ') if t not in stopWords]) > 0 else 0)

    return(df)
