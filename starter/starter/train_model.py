# Script to train machine learning model.
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data

# Add code to load in the data.
from ml.model import train_model, compute_model_metrics, inference

data = pd.read_csv('../data/census.csv')

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

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
clf = train_model(X_train, y_train)

y_pred = inference(clf, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

with open("slice_outpt.txt", "w") as file:
    file.write("Precision " + str(precision) + '\n')
    file.write("Recall " + str(recall) + '\n')
    file.write("F-beta score " + str(fbeta))

pickle.dump(clf, open('../model/naive_model.pkl', 'wb'))
pickle.dump(encoder, open('../model/encoder.pkl', 'wb'))