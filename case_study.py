#!/usr/bin/env python

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score

from src.loader import load_dataset
from src.models import SRR
from src.preprocessing import processing_pipeline, one_hot_encode


# Load the data
X, y = load_dataset(name="german_credit")

# Apply the processing pipeline
X_train_bin, X_test_bin, y_train, y_test = processing_pipeline(X, y)

# One-hot encode the categorical variables
X_train_one_hot = (X_train_bin)
X_test_one_hot = one_hot_encode(X_test_bin)

# Construct and train Select-Regress-Round model
model = SRR(k=3, M=3)
model.fit(one_hot_encode(X_train_bin), y_train, verbose=True)

# Show model
print("SRR model:", model.df, "", sep='\n\n')

# Show statistics of the model
train_acc = accuracy_score(y_train, model.predict(X_train_one_hot)) * 100
test_acc = accuracy_score(y_test, model.predict(X_test_one_hot)) * 100
baseline = max(1-y.mean(), y.mean()) * 100
print("With M={}, training accuracy of {:.1f} % and test accuracy of {:.1f} % (baseline {:.1f} %)".format(model.M, train_acc, test_acc, baseline))

