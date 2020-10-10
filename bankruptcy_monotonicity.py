#!/usr/bin/env python

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from src.loader import load_dataset
from src.models import SRR
from src.preprocessing import bin_features, one_hot_encode


def verifies_monotonicity(model):
    for feature in model.df.index.levels[0]:
        if feature != 'bias':
            N = model.df.loc['competitiveness'].loc['N', 'original']
            A = model.df.loc['competitiveness'].loc['A', 'original']
            P = model.df.loc['competitiveness'].loc['P', 'original']
            if not((N <= A and A <= P) or (N >= A and A >= P)):
                print("Monotonicity violated for feature", feature)
                return False
    
    return True


results = []

X, y = load_dataset('bankruptcy')

passed = 0
n_tests = 500

for nfold in tqdm(range(n_tests)):
    X_train, X_test = train_test_split(X, train_size=0.9, random_state=nfold)
    y_train, y_test = train_test_split(y, train_size=0.9, random_state=nfold)

    X_train, X_test = bin_features(X_train, X_test, nbins=3)

    X_train = one_hot_encode(X_train)
    X_test = one_hot_encode(X_test)

    model = SRR(k=3, Ms=range(1, 10+1))
    model.fit(X_train, y_train, verbose=False)
    
    passed += int(verifies_monotonicity(model))

print("%.2f %% passed monotonicity check" % (100 * passed / n_tests))

