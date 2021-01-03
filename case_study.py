#!/usr/bin/env python

from sklearn.metrics import accuracy_score

from src.loader import load_dataset
from src.models import SRR
from src.preprocessing import processing_pipeline, one_hot_encode


# Load the data
X, y = load_dataset(name="german_credit")

# Apply the processing pipeline
X_train, X_test, y_train, y_test = processing_pipeline(X, y)

# Construct and train Select-Regress-Round model
model = SRR(k=3, M=3)
model.fit(X_train, y_train, verbose=True)

# Show model
print("SRR model:", model.df, "", sep='\n\n')

# Show statistics of the model
train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
test_acc = accuracy_score(y_test, model.predict(X_test)) * 100
baseline = max(1-y.mean(), y.mean()) * 100
print("With M={}, training accuracy of {:.1f} % and test accuracy of {:.1f} % (baseline {:.1f} %)".format(model.M, train_acc, test_acc, baseline))

