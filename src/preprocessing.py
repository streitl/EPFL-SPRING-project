import pandas as pd
import numpy as np


def clean(X):
    """
    Does some pre-processing cleaning to the dataset.
    Includes dropping columns with no information, and converting numerical
    columns to their rightful datatype when there are missing values.
    
    Arguments:
    - X: DataFrame to clean
    
    Returns:
    - X_new: The cleaned version of the original DataFrame
    """
    X_new = X.copy()
    
    for col in X.columns:
        # Drop columns that are useless
        if len(X[col].unique()) == 1:
            X_new.drop(columns=col, inplace=True)
        # Try to change missing values into NaNs to get numerical columns
        elif '?' in set(X[col]):
            # Maybe the column is numerical once the '?' are removed
            try:
                temp = X[col].replace('?', np.nan).astype(float)
                X_new[col] = X[col].replace('?', np.nan).astype(float)
            except:
                pass
    
    return X_new


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
        breaks = train[col].dropna().quantile([i/nbins for i in range(1, nbins)]).unique().tolist()
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
    # Construct a multi index where the first level is the original column name,
    # and the second level is the column value
    new_df = pd.get_dummies(df, prefix_sep=sep, dummy_na=True)
    new_df.columns = pd.MultiIndex.from_tuples([c.split(sep) for c in new_df.columns])
    
    # Remove columns that have no information
    for col in new_df.columns:
        if len(new_df[col].unique()) == 1:
            new_df.drop(columns=col, inplace=True)
    
    return new_df

