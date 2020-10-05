#!/home/lua/Anaconda3/envs/spring/bin/python

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
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

baseline = y.mean() * 100
baseline = max(baseline, 100 - baseline)

for nfold in tqdm(range(500)):
    X_train, X_test = train_test_split(X, train_size=0.9, random_state=nfold)
    y_train, y_test = train_test_split(y, train_size=0.9, random_state=nfold)

    X_train, X_test = bin_features(X_train, X_test, nbins=3)

    X_train = one_hot_encode(X_train)
    X_test = one_hot_encode(X_test)

    model = SRR(k=3, Ms=range(1, 10+1))
    model.fit(X_train, y_train, verbose=False)

    train_acc = accuracy_score(y_train, model.predict(X_train, M=5)) * 100
    test_acc = accuracy_score(y_test, model.predict(X_test, M=5)) * 100
    
    results.append(verifies_monotonicity(model))

print(np.array(results).sum(), "passed monotonicity check")

