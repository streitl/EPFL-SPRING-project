#!/usr/bin/env python
# coding: utf-8
import pandas as pd

import argparse

from tqdm import tqdm

from src.loader import load_dataset
from src.models import SRR
from src.preprocessing import processing_pipeline, one_hot_encode
from src.vulnerabilities import find_adversarial_examples


# Instantiate the parser
parser = argparse.ArgumentParser(description='Adversarial examples finder')

# Add all required arguments
parser.add_argument('--k', type=int, nargs='?', default=3,
                    help='k is the number of features to be selected by the model')

parser.add_argument('--M', type=int, nargs='?', default=3,
                    help='M is the amplitude of the weights of the model')

parser.add_argument('dataset', type=str,
                    help='Name of the dataset without the .csv extension')

parser.add_argument('modifiable_features', type=str, nargs ='+',
                    help='List of feature names to be changed in order to find adversaries')

args = parser.parse_args()


# Loading and processing data
X, y = load_dataset(args.dataset)
    
# Apply the processing pipeline
X_train_bin, X_test_bin, y_train, y_test = processing_pipeline(X, y)


# To fight class imbalance on texas and ieeecis
if args.dataset in ['texas', 'ieeecis']:
    X_train_bin_subset = pd.concat([
        X_train_bin[y_train == 1].sample(n=3000, random_state=15),
        X_train_bin[y_train == 0].sample(n=3000, random_state=15)
    ])
    y_train_subset = y_train.loc[X_train_bin_subset.index]
else:
    X_train_bin_subset = X_train_bin
    y_train_subset = y_train


# Try to load model from disk, and if it doesn't exist train it
try:
    model = SRR.load(args.dataset, k=args.k, M=args.M)
except:
    # Fit model
    model = SRR(k=args.k, M=args.M)
    model.fit(one_hot_encode(X_train_bin_subset), y_train_subset, verbose=True)
    # Save it to disk
    model.save(args.dataset)

print("SRR model:", model.df, "", sep="\n\n")


# Finding adversarial examples
adversarial_results = find_adversarial_examples(model, X_train_bin_subset, y_train_subset, 
                                                can_change=args.modifiable_features)
n_adversaries = adversarial_results.shape[0]
print(f"Found {n_adversaries} adversarial examples for {args.dataset} by changing only {args.modifiable_features}:\n")
print(adversarial_results)

