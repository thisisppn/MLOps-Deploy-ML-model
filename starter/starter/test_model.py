import pickle

import numpy
import pandas as pd
import sklearn.naive_bayes
from numpy import float64
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from starter.ml.data import process_data

# Add code to load in the data.
from starter.ml.model import train_model, inference, compute_model_metrics

data = pd.read_csv('starter/data/census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.

train, test = train_test_split(data, test_size=0.20)
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train model
clf = train_model(X_train, y_train)

# NOTE Unlike what was suggested in the previous review, I am actually testing the pre trained/saved model itself
# by loading it in the following lines. I am not testing the newly trained model.

# Test model with saved model and encoder
model = pickle.load(open("starter/model/naive_model.pkl", 'rb'))
encoder = pickle.load(open("starter/model/encoder.pkl", 'rb'))

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

pred = inference(model, X_test)


p, r, fbeta = compute_model_metrics(y_test, pred)


def test_train_model():
    assert type(clf) == sklearn.naive_bayes.GaussianNB


def test_inference():
    assert type(pred) == numpy.ndarray


def test_model_metrics():
    assert type(p) == float64 and type(r) == float64 and type(fbeta) == float64
