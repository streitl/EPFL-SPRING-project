#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from src.loader import load_dataset
from src.models import SRR
from src.preprocessing import processing_pipeline, one_hot_encode
from src.attacks import find_adversarial_examples


# Loading and processing data
X, y = load_dataset("german_credit")
    
# Apply the processing pipeline
X_train_bin, X_test_bin, y_train, y_test = processing_pipeline(X, y)


# Fitting model
model = SRR(k=3, M=3)
model.fit(one_hot_encode(X_train_bin), y_train, verbose=True)

print("SRR model:", model.df, "", sep="\n\n")


# Finding adversarial examples
adversarial_results = find_adversarial_examples(model, X_train_bin, y_train, 
                                                can_change=['Duration_in_months'])

print("Found", adversarial_results.shape[0], 
      "adversarial examples for 'german_credit' by changing only 'Duration_in_months':\n\n",
      adversarial_results)

