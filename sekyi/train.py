import pandas as pd
import numpy as np

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion


from sklearn.linear_model import LogisticRegression
from utils import TextSelector, NumberSelector


# Get saved feather dataframe
df_keep = pd.read_feather('tmp/df_keep')


dependent = 'points'
engineered_features = [
    f for f in df_keep.columns.values if f not in ['description', 'points']]
numeric_features = [numeric for numeric in df_keep.columns.values if numeric not in [
    'description', 'points', 'description_processed']]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df_keep[engineered_features], df_keep[dependent], test_size=0.2, random_state=42)

# sklearn pipelines
description = Pipeline([
    ('selector', TextSelector(key='description_processed')),
    ('tfidf', TfidfVectorizer(stop_words='english'))
])

length = Pipeline([
    ('selector', NumberSelector(key='length')),
    ('standard', StandardScaler())
])

words = Pipeline([
    ('selector', NumberSelector(key='words')),
    ('standard', StandardScaler())
])

words_not_stopword = Pipeline([
    ('selector', NumberSelector(key='words_not_stopword')),
    ('standard', StandardScaler())
])

avg_word_length = Pipeline([
    ('selector', NumberSelector(key='avg_word_length')),
    ('standard', StandardScaler())
])

feats = FeatureUnion([('description', description),
                      ('length', length),
                      ('words', words),
                      ('words_not_stopword', words_not_stopword),
                      ('avg_word_length', avg_word_length)])

feature_processing = Pipeline([('feats', feats)])
feature_processing.fit_transform(X_train)

# fitting logistic regression algo
lr_pipeline = Pipeline([
    ('features', feats),
    ('classifier', LogisticRegression(
        random_state=42, solver='lbfgs', multi_class='auto', max_iter=1000)),
])

lr_pipeline.fit(X_train, y_train)
preds = lr_pipeline.predict(X_test)


# serialize the entire pipeline(includes models and preprocessing steps)
joblib.dump(lr_pipeline, 'tmp/lr_model.pkl')

print("****Training completed****")
print("****Model serialized successfully into the `tmp` folder****")
print(f"Model accuracy is {lr_pipeline.score(X_test, y_test)}")
