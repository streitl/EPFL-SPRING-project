import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def drop_useless_columns(X):
    """
    Drops columns of the given dataframe that have no information (always same value).
    
    Arguments:
    - X: DataFrame to clean
    
    Returns:
    - X_new: A copy the original DataFrame with potentially fewer columns
    """
    X_new = X.copy()
    for col in X.columns:
        # Drop columns that are useless
        if len(X[col].unique()) == 1:
            X_new.drop(columns=col, inplace=True)
    
    return X_new


def find_boundaries(series, nbins=3):
    """
    Takes a numerical pandas Series as input, as well as a number of bins to be defined,
    and tries find a list of boundaries that splits the series into this number of groups,
    where groups are numerical intervals.
    
    To do so, we try to find appropriate breaks in two passes: one on the index-ordered
    histogram of series, and another on the same histogram but in descending order.
    
    Arguments:
    - series: A pandas Series of numerical values
    - nbins : The # of bins to be defined with the boundaries
    """
    assert int(nbins) == nbins, "the number of bins must be an int"
    assert 2 <= nbins, "the number of bins must be at least 2"
    
    # Create 'histogram' of values for the series (dropping NaN's)
    counts = series.value_counts()
    assert len(counts) >= nbins, "there must be at most as many bins as values"
    
    # Compute the total number of values in series (minus NaNs)
    total = counts.sum()
    
    def find_best_breaks(ascending=True):
        """
        Small subfunction that approximates finding the quantiles of the data,
         but takes into account that some values occur very often.
        
        It works like this:
        1. We define 'n' to be the current quantile number that we want to find.
        2. We define 'goal' to be the current cumulative sum objective, that is,
            as soon as the cumulative sum is bigger than 'goal', we know that we
            have found a break (that approximates a quantile defined by n).
        3. Whenever a break is found, we decrease n by 1, and we define goal to
            be the 'quantile' of the remaining data points, those which have not
            yet been covered by the sum. We also reset the cumulative sum to 0.
        4. We stop when n is 1 (since there is no 1-th quantile).
        
        Arguments:
        - ascending: Boolean value indicating whether to iterate over
                      the counts in ascending or descending order
        
        Returns:
        - breaks: A sorted list of float values whose size is at most nbins-1
        """
        # This represents which quantile we want to find
        n = nbins
        # This represents the objective sum
        goal = total / n
        # This is the cumulative sum
        current_sum = 0
        
        sorted_counts = counts.sort_index(ascending=ascending)
    
        breaks = []
        for i, (value, count) in enumerate(sorted_counts.iteritems()):
            current_sum += count

            if current_sum >= goal:
                n -= 1
                # There is a special case for when ascending=False.
                # pd.cut needs the intervals to be left-excluding, right-including,
                #  so when we find a potential break in inverse order, we set the
                #  NEXT element (smaller than the current) to be the break instead.
                if ascending or i+1 == len(sorted_counts):
                    breaks.append(value)
                else:
                    breaks.append(sorted_counts.index[i+1])
                if n == 1: break
                goal = (total - current_sum) / n
                current_sum = 0
        
        if ascending:
            return breaks
        else: 
            return sorted(breaks)
    
    # Find the best breaks both in an ascending and in a descending pass
    forward = find_best_breaks(ascending=True)
    backward = find_best_breaks(ascending=False)
    
    # If both passes yielded the same number of breaks, average them
    if len(forward) == len(backward):
        boundaries = np.array([forward, backward]).mean(axis=0).tolist()
    
    # Otherwise keep the longest list of breaks
    elif len(forward) > len(backward):
        boundaries = forward
    else:
        boundaries = backward
    
    # Set -inf and +inf as extreme boundaries so that any value can be binned    
    return [(float('-inf'))] + boundaries + [(float('inf'))]


def bin_features(X_train, X_test, nbins):
    """
    Bins the numerical features of the input DataFrames, by first defining quantiles
     based on the training set, and then using quantiles to split the features into groups.
    
    Arguments:
    - X_train: DataFrame to be binned, used to compute the quantiles
    - X_test : DataFrame to be binned
    - nbins  : # of bins to be defined
    
    Returns:
    - X_train_bin: DataFrame whose numerical features are binned
    - X_test_bin : DataFrame whose numerical features are binned
    """
    # Copy the data
    X_train_bin = X_train.copy()
    X_test_bin = X_test.copy()
    
    # Iterate over numerical features
    for col in X_train.select_dtypes(include=np.number).columns:
        # If the data already has at most nbins distinct values, just convert the column to a category
        if len(X_train[col].unique()) <= nbins:
            X_train_bin[col] = X_train[col].astype('category')
            X_test_bin[col] = X_test[col].astype('category')
            continue
        
        # Define the bins using X_train only    
        breaks = find_boundaries(X_train[col], nbins=nbins)
        
        # Use the bins to bin X_train and X_test
        X_train_bin[col] = pd.cut(X_train[col], breaks)
        X_test_bin[col] = pd.cut(X_test[col], breaks)
        
        # Sometimes a bin is empty, so we remove it
        X_train_bin[col].cat.remove_unused_categories(inplace=True)
        X_test_bin[col].cat.remove_unused_categories(inplace=True)
    
    return X_train_bin, X_test_bin


def one_hot_encode(df, sep='~'):
    """
    One-hot encodes the given dataset, by transforming each categorical feature column into
     a set of binary features, and grouping them together in a pandas MultiIndex.
    
    Arguments:
    - df : DataFrame to one-hot encode
    - sep: String to be used for the construction of the MultiIndex, can't belong to the columns
    
    Returns:
    - new_df: DataFrame with a MultiIndex column index, containing only binary features
    """
    # Construct a multi index where the first level is the original column name,
    #  and the second level is the column value (if there are nan values, create a nan entry as well)
    new_df = pd.get_dummies(df, prefix_sep=sep, dummy_na=df.isnull().values.any())
    new_df.columns = pd.MultiIndex.from_tuples([c.split(sep) for c in new_df.columns])
    
    return new_df


def processing_pipeline(X, y, train_size=0.9, seed=100, nbins=3):
    """
    Applies the whole processing pipeline (except 1-hot encoding).
    
    Arguments:
    - X         : DataFrame with the unprocessed features
    - y         : DataFrame with the labels
    - train_size: Proportion of data that goes towards training
    - seed      : Random seed to be used when splitting the data intro train and test
    - nbins     : Number of bins to partition numerical data into
    
    Returns:
    - X_train: DataFrame with the train features, cleaned and processed
    - X_test : DataFrame with the test features, cleaned and processed
    - y_train: DataFrame with the train labels, values are only 0 and 1
    - y_test : DataFrame with the test labels, values are only 0 and 1
    """
    # Removes columns with no information
    X_clean = drop_useless_columns(X)
    
    # Splits data into train and test in a reproducible manner
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, train_size=train_size,
                                                        random_state=seed, stratify=y)

    # Bin the numerical features into groups
    X_train_bin, X_test_bin = bin_features(X_train, X_test, nbins=nbins)
    
    return X_train_bin, X_test_bin, y_train, y_test

