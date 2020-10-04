import pandas as pd
import numpy as np


def bin_features(X_train, X_test, nbins):
    """
    Bins the numerical features of the input DataFrames, by first defining quantiles
    based on the training set, and then using quantiles to split the features into groups.
    
    Arguments:
    - X_train: DataFrame to be binned, used to compute the quantiles
    - X_test:  DataFrame to be binned
    - nbins:   # of bins to be defined
    
    Returns:
    - train: DataFrame whose numerical features are binned
    - test:  DataFrame whose numerical features are binned
    """
    # Copy the data
    train = X_train.copy()
    test = X_test.copy()
    
    for col in X_train.columns:
        # Bin only numerical features
        if not np.issubdtype(X_train[col].dtype, np.number):
            continue
            
        # Define the bins using X_train only
        breaks = train[col].quantile([i/nbins for i in range(1, nbins)]).unique().tolist()
        breaks = [float("-inf")] + breaks + [float("inf")]
        
        # Use the bins to bin X_train and X_test
        train[col] = pd.cut(train[col], breaks)
        test[col] = pd.cut(test[col], breaks)
        
        # Sometimes a bin is empty, so we remove it
        train[col].cat.remove_unused_categories(inplace=True)
        test[col].cat.remove_unused_categories(inplace=True)
    return train, test


def one_hot_encode(df, sep='~'):
    """
    One-hot encodes the given dataset, by transforming each categorical feature column into
    a set of binary features, and grouping them together in a pandas MultiIndex.
    
    Arguments:
    - df:  DataFrame to one-hot encode
    - sep: String to be used for the construction of the MultiIndex, can't belong to the columns
    
    Returns:
    - new_df: DataFrame with a MultiIndex column index, containing only binary features
    """
    new_df = pd.get_dummies(df, prefix_sep=sep)
    new_df.columns = pd.MultiIndex.from_tuples([c.split(sep) for c in new_df.columns])
    return new_df

