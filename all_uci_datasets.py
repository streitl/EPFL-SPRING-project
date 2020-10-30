#!/usr/bin/env python

import pandas as pd

from sklearn.metrics import accuracy_score

from src.loader import load_dataset
from src.models import SRR
from src.preprocessing import processing_pipeline, one_hot_encode


uci_datasets = ['adult', 'annealing', 'audiology-std', 'bank', 'bankruptcy', 'car',
                'chess-krvk', 'chess-krvkp', 'congress-voting', 'contrac', 'credit-approval',
                'ctg', 'cylinder-bands', 'dermatology', 'german_credit', 'heart-cleveland',
                'ilpd', 'mammo', 'mushroom', 'wine', 'wine_qual']

results = pd.DataFrame(columns=["dataset", "train accuracy", "test accuracy", "baseline"])

for dataname in uci_datasets:
    print(f"-> {dataname} dataset")
    # Load the data
    X, y = load_dataset(name=dataname)
    
    # Apply the processing pipeline
    X_train_bin, X_test_bin, y_train, y_test = processing_pipeline(X, y)
    
    # One-hot encode the categorical variables
    X_train_one_hot = one_hot_encode(X_train_bin)
    X_test_one_hot = one_hot_encode(X_test_bin)
    
    # Construct and train Select-Regress-Round model
    model = SRR(k=1, M=5, cv=2)
    model.fit(X_train_one_hot, y_train, verbose=True)
    
    # Show statistics of the model
    train_acc = accuracy_score(y_train, model.predict(X_train_one_hot)) * 100
    test_acc = accuracy_score(y_test, model.predict(X_test_one_hot)) * 100
    baseline = max(1-y.mean(), y.mean()) * 100
    print("Training accuracy of {:.1f} % and test accuracy of {:.1f} % (baseline {:.1f} %)\n".format(train_acc, test_acc, baseline))
    
    results = results.append(
        {'dataset': dataname, 
        'train accuracy': train_acc,
        'test accuracy': test_acc,
        'baseline': baseline},
        ignore_index=True)

print(results)

