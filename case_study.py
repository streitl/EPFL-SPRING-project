#!/home/lua/Anaconda3/envs/spring/bin/python

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.loader import load_dataset
from src.models import SRR
from src.preprocessing import bin_features, one_hot_encode


# Load the data
X, y = load_dataset(name="german_credit")

# Split data into train and test sets
X_train, X_test = train_test_split(X, train_size=0.9, random_state=42)
y_train, y_test = train_test_split(y, train_size=0.9, random_state=42)

# Bin numerical features
X_train, X_test = bin_features(X_train, X_test, nbins=3)

# One-hot encode the categorical variables
X_train = one_hot_encode(X_train)
X_test = one_hot_encode(X_test)

# Construct and train Select-Regress-Round model
model = SRR(k=3, Ms=range(1, 10+1))
model.fit(X_train, y_train, verbose=True)

# Show model
print("SRR model:", model.df, "", sep='\n\n')

# Show statistics of the model
M = 5
train_acc = accuracy_score(y_train, model.predict(X_train, M)) * 100
test_acc = accuracy_score(y_test, model.predict(X_test, M)) * 100
baseline = np.concatenate([y_train, y_test]).mean() * 100
print("With M=%d, training accuracy of %.1f %% and test accuracy of %.1f %% (baseline %.1f %%)" % (M, train_acc, test_acc, baseline))

