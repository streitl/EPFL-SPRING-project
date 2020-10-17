#!/usr/bin/env python

import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm

from src.loader import load_dataset
from src.models import SRR
from src.preprocessing import processing_pipeline, one_hot_encode
from src.vulnerabilities import verifies_monotonicity


# Instantiate the parser
parser = argparse.ArgumentParser(description='Monotonicity checker')

# Add all required arguments
parser.add_argument('dataset', type=str,
                    help='Name of the dataset without the .csv extension')

args = parser.parse_args()


X, y = load_dataset(args.dataset)

passed = 0
n_tests = 10

for nfold in tqdm(range(n_tests)):
    
    X_train_bin, X_test_bin, y_train, y_test = processing_pipeline(X, y, seed=nfold)

    model = SRR(k=3, M=3)
    model.fit(one_hot_encode(X_train_bin), y_train, verbose=False)
    
    passed += int(verifies_monotonicity(model))

print("{}: {:.1f} % passed monotonicity check".format(args.dataset, 100 * passed / n_tests))

