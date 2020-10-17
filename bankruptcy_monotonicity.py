#!/usr/bin/env python

import pandas as pd
import numpy as np

from tqdm import tqdm

from src.loader import load_dataset
from src.models import SRR
from src.preprocessing import processing_pipeline, one_hot_encode


def bankruptcy_verifies_monotonicity(model):
    for feature in model.df.index.levels[0]:
        if feature != 'bias':
            N = model.df.loc['competitiveness'].loc['N', 'original']
            A = model.df.loc['competitiveness'].loc['A', 'original']
            P = model.df.loc['competitiveness'].loc['P', 'original']
            if not((N <= A and A <= P) or (N >= A and A >= P)):
                print("Monotonicity violated for feature", feature)
                return False
    
    return True


X, y = load_dataset('bankruptcy')

passed = 0
n_tests = 500

for nfold in tqdm(range(n_tests)):
    
    X_train_bin, X_test_bin, y_train, y_test = processing_pipeline(X, y, seed=nfold)

    model = SRR(k=3, M=3)
    model.fit(one_hot_encode(X_train_bin), y_train, verbose=False)
    
    passed += int(bankruptcy_verifies_monotonicity(model))

print("{:.1f} % passed monotonicity check".format(100 * passed / n_tests))

