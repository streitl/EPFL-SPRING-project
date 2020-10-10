#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from src.loader import load_dataset
from src.models import SRR
from src.preprocessing import bin_features, one_hot_encode, clean
from src.attacks import find_adversarial_examples


# Loading and processing data
X, y = load_dataset("german_credit")
    
X = clean(X)

X_train, X_test = train_test_split(X, train_size=0.9, random_state=15)
y_train, y_test = train_test_split(y, train_size=0.9, random_state=15)

X_train, X_test = bin_features(X_train, X_test, nbins=3)


# Fitting model
model = SRR(k=3, M=3)
model.fit(one_hot_encode(X_train), y_train, verbose=True)

print("SRR model:", model.df, "", sep="\n\n")


# Finding adversarial examples
adversarial_results = find_adversarial_examples(model, X_train, y_train, 
                                                can_change=['Duration_in_months'])

print("Found", adversarial_results.shape[0], 
      "adversarial examples for 'german_credit' by changing only 'Duration_in_months':\n\n",
      adversarial_results)

