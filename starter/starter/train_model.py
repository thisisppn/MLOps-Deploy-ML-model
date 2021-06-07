# Script to train machine learning model.
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable

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

pickle.dump(clf, open('../model/naive_model.pkl', 'wb'))
pickle.dump(encoder, open('../model/encoder.pkl', 'wb'))

print("Overall metrics: ")
print("""Precision: {}
   Recall: {}
    fbeta: {}
    """.format(precision, recall, fbeta))


# NOTE: Solution for Rubric #5.
# Fix a feature, and output metrics for each of it's unique value.

fix_variable = "education"

x = PrettyTable()
x.title = 'Metrics with fixed categorical variable: "{}"'.format(fix_variable)
x.field_names = ['Value', 'Precision', 'Recall', 'F-Beta']

for cat_value in data[fix_variable].unique():
    sub_data = data[data[fix_variable] == cat_value]

    # Process the test data with the process_data function.
    X_test_subset, y_test_subset, encoder, lb = process_data(
        sub_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    y_pred_subset = clf.predict(X_test_subset)
    precision, recall, fbeta = compute_model_metrics(y_test_subset, y_pred_subset)

    x.add_row([cat_value, precision, recall, fbeta])

print(x)

with open('slice_output.txt', 'w') as f:
    f.write(str(x))
