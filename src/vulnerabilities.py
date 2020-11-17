import sys

import pandas as pd
import numpy as np

from tqdm import tqdm
from itertools import product

from .preprocessing import one_hot_encode
from .models import SRR, RoundedLogisticRegression, SRRWithoutCrossValidation


def find_adversarial_examples(srr_model, X, y, can_change, unit_changes=False, allow_nan=True):
    """
    Given an SRR model, data points, and the features which can be changed, produces a list of 
    adversarial examples that have changed the model prediction label.
    
    Args:
        srr_model   : Trained SRR model
        X           : DataFrame with categorical features, preprocessed but not 1-hot encoded
        y           : Series with the labels
        can_change  : List with features in X that we can modify
        unit_changes: Boolean indicating whether to change a single feature at a time
        allow_nan   : Boolean indicating whether nans are allowed changes or not
    
    Returns:
        adversaries_and_originals: Dataframe with adversarial examples
                                 and the corresponding original datapoints
    """
    # Keep track of which features can be modified
    modifiable_features = set(srr_model.features).intersection(set(can_change))
    assert len(modifiable_features) > 0, "No features can be modified with the given parameters"
    
    # Create model predictions
    y_pred = srr_model.predict(one_hot_encode(X))
    
    # Only keep data points whose label was correctly predicted,
    # and only keep the features selected by the SRR model
    correctly_classified = X.loc[y_pred == y, srr_model.features]
    
    # Append label column to datapoints
    correctly_classified['label~original'] = y[y_pred == y]
    
    # Create a mapping from features to possible categories
    if allow_nan:
        feat_to_cats = {feat: list(X[feat].unique()) for feat in modifiable_features}
    else:
        feat_to_cats = {feat: list(X[feat].dropna().unique()) for feat in modifiable_features}
    
    # Create a list with tuples of possible feature changes
    if unit_changes:
        possible_changes = [(k, v) for k, l in feat_to_cats.items() for v in l]
    else:
        possible_changes = [(modifiable_features, feat) for feat in (product(*feat_to_cats.values()))]
    
    ## Construct a list of potential adversarial examples by deforming each correctly classified sample
    potential_adversaries = pd.DataFrame(columns=correctly_classified.columns)
    
    # Iterate over correctly classified points
    for index, data in tqdm(correctly_classified.iterrows(), total=correctly_classified.shape[0]):
        
        # Go through precomputed tuples of feature changes
        for change_cols, change_vals in possible_changes:
            
            # Modify the original data point and add it to the list
            deformed = data.copy()
            deformed.loc[change_cols] = change_vals
            potential_adversaries = potential_adversaries.append(deformed)
    
    # Get the model prediction for the adversaries
    potential_adversaries['label~adversarial'] = srr_model.predict(
        one_hot_encode(potential_adversaries.drop(columns=['label~original']))
    )
    
    # Only keep those which changed the prediction label
    true_adversaries = potential_adversaries.loc[
         potential_adversaries['label~adversarial'] != potential_adversaries['label~original']
    ]
    
    # Join the true adversaries with the corresponding datapoints to see what changed
    adversaries_and_originals = pd.merge(true_adversaries, X[srr_model.features],
         left_index=True, right_index=True,
         suffixes=['~adversarial', '~original'])
    
    # Organise the results in a multi index to easily retrieve original and adversarial info
    adversaries_and_originals.columns = pd.MultiIndex.from_tuples(
        [col.split("~")[::-1] for col in adversaries_and_originals.columns]
    )
    
    # Sort the column names to get a more compact view
    return adversaries_and_originals.reindex(sorted(adversaries_and_originals.columns), axis=1)



def binned_features_pass_monotonicity(srr_model, X, y):
    """
    Takes a trained SRR model, and for each selected feature whose categories are intervals (and possibly nan),
    checks that the weights corresponding to the intervals are either in increasing or decreasing order, and see if
    this non-monotonicity allows to find adversarial examples.
    
    Args:
        srr_model: Trained SRR model
        X        : DataFrame with the features
        y        : DataFrame with the label
    
    Returns:
        Boolean indicating whether the binned features of the model pass monotonicity check
    """
    # Initialize set of non-monotonic features
    non_monotonic_features = set()
    
    # Iterate over all features that the model uses
    for feature in srr_model.df.index.levels[0]:
        
        # Retrieve the categories corresponding to this feature
        categories = srr_model.df.loc[feature].index
        
        # Intervals are of the form '(left, right]'
        if categories.str.startswith("(").any():
            
            # Keep only non-na categories
            non_na_categories = categories[categories != 'nan']
            
            # Retrieve original weights of the model (as a Series)
            weights = srr_model.df.loc[feature].loc[non_na_categories, srr_model.M]
            
            # Indicators of whether the weights have already increased/decreased so far
            increased, decreased = False, False
            
            # Go trough pairs of current and previous weights
            for current, previous in zip(weights.iloc[1:], weights):
                if current > previous:
                    increased = True
                elif current < previous:
                    decreased = True
                
                # If both are true, then the feature is monotonic
                if increased and decreased:
                    non_monotonic_features.add(feature)
    
    # If all features are monotonic then the test is passed
    if len(non_monotonic_features) == 0:
        return True
        
    # Look for adversarial examples by changing only the features that are non-monotonic, one at a time
    adversarial_examples = find_adversarial_examples(srr_model, X, y, can_change=non_monotonic_features,
                                                     unit_changes=True, allow_nan=False)
    
    if adversarial_examples.shape[0] > 0:
        return False
    return True


def poisoning_attack(original_srr, X_train, y_train, feature, category=None, goal='nullify', col='normal'):
    """
    Performs a poisoning attack by iteratively removing points from the training set of the given model.

    There are 3 possible goals, defined by the 'goal' parameter:
        'flip_sign'     : flip the sign of the weight of the model corresponding to 'feature', 'category'
        'remove_feature': remove 'feature' from the set of features used by the model
        'nullify'       : make the weight of the model corresponding to 'feature', 'category' go to zero

    To achieve this goal, we try to maximize/minimize (if goal flip_sign) or nullify (if goal nullify)
    some weight of the model corresponding to the given feature and category.
    In the case of 'remove_feature', we try to nullify the weights of all categories for 'feature',
    in order to make the feature irrelevant.

    Many different kinds of weights can be used, determined by the 'col' param:
        'original': from the inner logistic regression model (recommended for 'nullify' and 'remove_feature')
        'relative': from the SRR model right before multiplication by M and rounding
                        (not recommended, if weight is the biggest/smallest, then likely no changes)
        'normal'  : from the inner logistic regression model but normalized (recommended for 'flip_sign')
        'M'       : from the SRR model (not recommended, likely no changes)

    Args:
        original_srr: pre-trained SRR model on the given dataset
        X_train     : training set DataFrame with the features, binned but not 1-hot encoded
        y_train     : training set Series with the labels
        feature     : feature of the model to poison
        category    : optional (only defined if goal is 'flip_sign' or 'nullify')
        goal        : the goal of the poisoning attack
        col         : which kind of weight to use

    Returns:
        removals: List with the indices of the data points that were removed
    """
    # Validation of parameters
    assert goal in ['flip_sign', 'remove_feature', 'nullify'], \
        "goal must be either 'flip_sign', 'remove_feature', or 'nullify'"
    assert col in ['original', 'relative', 'normal', 'M'], \
            "col must be either 'original', 'relative', 'normal', or 'M'"
    assert (category is not None) ^ (goal == 'remove_feature'), \
        "category can only be None if goal is 'remove_feature'"

    if col == 'M':
        col = original_srr.M

    if goal in ['flip_sign', 'nullify']:
        # We approximate SRR by a version without feature selection and cross-validation in this case
        model_kind = RoundedLogisticRegression
        # This is the SRR weight that we want to change
        original_weight = original_srr.get_weight(feature, category)

        # Check if weight is defined
        if np.isnan(original_weight):
            raise ValueError('The weight is does not exist in the model.')
        # Check whether the original weight is already 0
        if original_weight == 0 and goal == 'flip_sign':
            raise ValueError('Cannot flip weight sign of feature with weight 0.')
        if original_weight == 0 and goal == 'nullify':
            raise ValueError('The weight is already null and cannot be nullified. Stopping.')

        if original_weight > 0:
            print(f'Original weight is positive ({original_weight:.0f}), so we want it to decrease.')
        else:
            print(f'Original weight is negative ({original_weight:.0f}), so we want it to increase.')
        sys.stdout.flush()
    else: # goal == 'remove_feature'
        # We approximate SRR by a version without cross-validation in this case
        model_kind = SRRWithoutCrossValidation

    # Define the base set from which we remove one feature at a time.
    # Initially it's a copy of the given training set.
    X_base = X_train.copy()
    y_base = y_train.copy()

    # The list with the points we remove from the training set
    removals = []

    srr = original_srr
    iteration = 0
    # Stop when we remove half the training points (or if the goal is achieved inside the while loop)
    while iteration < X_train.shape[0] / 2:
        # Instantiate DataFrame to put results in
        if goal in ['flip_sign', 'nullify']:
            res = pd.DataFrame(dtype=float, columns=[col, 'M'])
        else:
            res = pd.DataFrame(dtype=float, columns=[col, 'M'])

        # Iterate over all points over the base set (which we want to remove)
        for ith_point, _ in tqdm(X_base.iterrows(), total=X_base.shape[0], leave=True, position=0):

            # Create dataset without i-th point
            X_no_i = X_base.drop(index=ith_point)
            y_no_i = y_base.drop(index=ith_point)

            # Fit alternative model (with same parameters as srr) to reduced dataset
            model = model_kind.from_srr(srr)
            try:
                model.fit(one_hot_encode(X_no_i), y_no_i)
            except ValueError:
                # This error happens when there are no samples of some class in the training set
                # In this case, we break the loop
                print("Could not achieve goal...")
                return removals

            # Try to retrieve values of interest
            if goal in ['flip_sign', 'nullify']:
                try:
                    res.loc[ith_point] = model.df.loc[(feature, category)][[col, model.M]].values
                except KeyError:
                    res.loc[ith_point] = (np.nan, np.nan)
            elif goal == 'remove_feature':
                if feature in model.features:
                    res.loc[ith_point] = model.df.loc[feature, [col, model.M]].abs().mean().values
                else:
                    # The feature is not present in the alternate model so we notify this with a NaN
                    res.loc[ith_point] = (np.nan, np.nan)

        # If we could not get any weights for the model in these two cases, abandon
        if goal in ['flip_sign', 'nullify'] and res.isna().all().all():
            raise ValueError(f'Attack failed, feature was not present in any model. Tried removals: {removals}')

        # Find which point brings us closer to our goal or achieves it (on the alternative model)
        if goal == 'flip_sign':
            # In this case we want the point which brings us closer to the opposite sign of the original weight
            # To do so, we keep the points with the best rounded weights, and then take the best according to 'col'
            if original_weight > 0:
                best_point = res.loc[res.M == res.M.min(), col].idxmin()
            else:
                best_point = res.loc[res.M == res.M.max(), col].idxmax()
        elif goal == 'nullify':
            # In this case we are looking for the point that brings us closer to 0
            # We first get the rounded weights with smallest abs, then the best according to 'col'
            best_point = res.loc[res.M.abs() == res.M.abs().min(), col].abs().idxmin()
        else: # goal == 'remove_feature':
            # If some data point caused a feature removal
            if res[col].isna().any():
                # Keep the first point with nan (arbitrarily)
                best_point = res[res[col].isna()].index[0]
            else:
                # Otherwise we keep the point which brings us closer to zero
                # Again, we first get the rounded weights with smallest abs, then the best according to 'col'
                best_point = res.loc[res.M.abs() == res.M.abs().min(), col].abs().idxmin()

        # Add the best point to our list of removals
        removals.append(best_point)
        iteration += 1

        # Remove the current best point from the base set
        X_base.drop(index=best_point, inplace=True)
        y_base.drop(index=best_point, inplace=True)

        # Train SRR model on the new reduced dataset to see if the goal was achieved
        srr = SRR.copy_params(original_srr)
        srr.fit(one_hot_encode(X_base), y_base)

        # Check if we have achieved our goal, and if so, return list of removals
        if (goal == 'flip_sign' and original_weight * srr.get_weight(feature, category) < 0) \
                or (goal == 'nullify' and srr.get_weight(feature, category) == 0) \
                or (goal == 'remove_feature' and feature not in srr.features):
            print('\nAttack successful! :D')
            return removals

        if goal in ['flip_sign', 'nullify']:
            try:
                s = srr.df.loc[(feature, category), [col, srr.M]]
                print(f'Iteration {iteration}: removed {best_point}, got weights: {s.index.values}, {s.values}', end='')
            except KeyError:
                print(f'Iteration {iteration}: removed {best_point}, got no weights (feature not in model)', end='')
        else:
            try:
                s = srr.df.loc[feature, [col, srr.M]].abs().mean()
                print(f'Iteration {iteration}: removed {best_point}, got weights: {s.index.values}, {s.values}', end='')
            except KeyError:
                print(f'Iteration {iteration}: removed {best_point}, got no weights (feature not in model)', end='')
        sys.stdout.flush()


    raise ValueError(f'Attack failed, removed too many points. Tried removals: {removals}')
