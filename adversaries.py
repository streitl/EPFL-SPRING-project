#!/usr/bin/env python
# coding: utf-8
import sys

import pandas as pd

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from src.loader import load_dataset
from src.models import SRR
from src.preprocessing import processing_pipeline, one_hot_encode
from src.attacks import find_adversarial_examples


# Check that the right number of arguments was given
if len(sys.argv) < 3:
    raise ValueError("Wrong number of arguments. Usage: ./adversaries.py dataset_name [modifiable_features]")

# Retrieve dataset name from command line
dataset_name = sys.argv[1]
# And the list of features which can be modified
modifiable_features = sys.argv[2:]

# Loading and processing data
X, y = load_dataset(dataset_name)
    
# Apply the processing pipeline
X_train_bin, X_test_bin, y_train, y_test = processing_pipeline(X, y)


# To fight class imbalance
if dataset_name in ['texas', 'ieeecis']:
    X_train_bin_subset = pd.concat([
        X_train_bin[y_train == 1].sample(n=3000, random_state=15),
        X_train_bin[y_train == 0].sample(n=3000, random_state=15)
    ])
    y_train_subset = y_train.loc[X_train_bin_subset.index]
else:
    X_train_bin_subset = X_train_bin
    y_train_subset = y_train

# Fitting model
model = SRR(k=3, M=3)
model.fit(one_hot_encode(X_train_bin_subset), y_train_subset, verbose=True)

print("SRR model:", model.df, "", sep="\n\n")


# Finding adversarial examples
adversarial_results = find_adversarial_examples(model, X_train_bin_subset, y_train_subset, 
                                                can_change=modifiable_features)
n_adversaries = adversarial_results.shape[0]
print(f"Found {n_adversaries} adversarial examples for {dataset_name} by changing only {modifiable_features}:\n")
print(adversarial_results)

