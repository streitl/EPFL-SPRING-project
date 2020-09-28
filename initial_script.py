#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import rpy2
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from rpy2.robjects.packages import importr
from rpy2.robjects import r, globalenv, pandas2ri

# Suppresses R warning output
import logging
rpy2_logger.setLevel(logging.ERROR)

base = importr("base")
pandas2ri.activate()

r("set.seed(0)")
r("setwd('src')")

# Runs the files with the helper functions
utils = r.source("utils.R")
subfuncs = r.source("subfuncs.R")


class SRR():
    """
    A python wrapper for the SRR model as returned in R.
    
    The constructor builds a dataframe whose rows are the selected features,
    and whose columns are different M values for which the rounding was done (by default, from 1 to 10).
    As such, in theory this class contains 10 different SRR models.
    
    Allows the prediction of the class for new data points.
    """
    def __init__(self, r_res):
        self.df = pd.DataFrame(r_res.rx2('values'),
                               index=r_res.rx2('rownames'),
                               columns=pd.to_numeric(r_res.rx2('colnames')),
                               dtype=int)
    
    def predict(self, x, M=3):
        """
        
        x: DataFrame to predict
        M: Amplitude of the weights (acts as a model selector here)
        
        TODO add a threshold here somehow to have a binary prediction vector.
        """
        preds = pd.Series(np.zeros(len(x.index)), dtype=int)
        for col in x.columns:
            if col in self.df.index:
                preds += self.df.loc[col][M] * x[col]
        
        return preds
    
    
def accuracy(y_pred, y):
    """
    Computes the mean accuracy of the prediction vector in a binary classification context.
    """
    acc = (y_pred == y).astype(int).mean()
    return acc


# Calls the function train_srr in utils.R
r_res = r['train_srr'](dataname="german_credit", k=5)

# Builds python wrapper for the srr model
model = SRR(r_res)

# Load train and test data from the R object
train = pd.get_dummies(r_res.rx2('train_data'), prefix_sep='').reset_index()
test = pd.get_dummies(r_res.rx2('test_data'), prefix_sep='').reset_index()

# Model predictions
train_preds = model.predict(train, M=5)
test_preds = model.predict(test, M=5)
# TODO replace 0 by baseline
train_acc = accuracy(train_preds >= 0, train.label)
test_acc = accuracy(test_preds >= 0, test.label)

# Show results
baseline = pd.concat([train.label, test.label]).mean()
print("SRR model with training accuracy %.1f %% and test accuracy %.1f %% (baseline %.1f %%)" % (train_acc * 100, test_acc * 100, baseline * 100))



