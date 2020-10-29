#!/usr/bin/env python

import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm

from src.loader import load_dataset
from src.models import SRR
from src.preprocessing import processing_pipeline, one_hot_encode
from src.vulnerabilities import binned_features_pass_monotonicity


# Instantiate the parser
parser = argparse.ArgumentParser(description='Monotonicity checker')

# Add all required arguments
parser.add_argument('--k', type=int, nargs='?', default=3,
                    help='k is the number of features to be selected by the model')

parser.add_argument('--M', type=int, nargs='?', default=3,
                    help='M is the amplitude of the weights of the model')

parser.add_argument('--ntests', type=int, nargs='?', default=10,
                    help='The number of tests to do, so the number of models to train')

parser.add_argument('dataset', type=str,
                    help='Name of the dataset without the .csv extension')

args = parser.parse_args()


X, y = load_dataset(args.dataset)

passed = 0

print(f"Verifying monotonicity for {args.ntests} different models:")
for nfold in tqdm(range(args.ntests)):
    
    X_train_bin, X_test_bin, y_train, y_test = processing_pipeline(X, y, seed=nfold)

    model = SRR(k=args.k, M=args.M)
    model.fit(one_hot_encode(X_train_bin), y_train, verbose=False)
    
    passed += int(binned_features_pass_monotonicity(model, X_train_bin, y_train))

print("{}: {:.1f} % passed monotonicity check".format(args.dataset, 100 * passed / args.ntests))

