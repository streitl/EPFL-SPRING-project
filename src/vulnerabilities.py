import sys

import pandas as pd
import numpy as np

from tqdm import tqdm
from itertools import product, combinations, chain

from .preprocessing import one_hot_encode, processing_pipeline
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
    potential_adversaries['label~new'] = srr_model.predict(
        one_hot_encode(potential_adversaries.drop(columns=['label~original']))
    )
    
    # Only keep those which changed the prediction label
    true_adversaries = potential_adversaries.loc[
         potential_adversaries['label~new'] != potential_adversaries['label~original']
    ]
    
    # Join the true adversaries with the corresponding datapoints to see what changed
    adversaries_and_originals = pd.merge(true_adversaries, X[srr_model.features],
         left_index=True, right_index=True,
         suffixes=['~new', '~original'])
    
    # Organise the results in a multi index to easily retrieve original and adversarial info
    adversaries_and_originals.columns = pd.MultiIndex.from_tuples(
        [col.split("~")[::-1] for col in adversaries_and_originals.columns]
    )
    
    # Sort the column names to get a more compact view
    return adversaries_and_originals.reindex(sorted(adversaries_and_originals.columns), axis=1)



def binned_features_pass_monotonicity(srr, X, y):
    """
    Takes a trained SRR model, and for each selected feature whose categories are intervals (and possibly nan),
    checks that the weights corresponding to the intervals are either in increasing or decreasing order, and see if
    this non-monotonicity allows to find adversarial examples.
    
    Args:
        srr: Trained SRR model
        X  : DataFrame with the features
        y  : DataFrame with the label
    
    Returns:
        Boolean indicating whether the binned features of the model pass monotonicity check
    """
    # Initialize set of non-monotonic features
    non_monotonic_features = set()
    
    # Iterate over all features that the model uses
    for feature in srr.df.index.levels[0]:
        
        # Retrieve the categories corresponding to this feature
        categories = srr.df.loc[feature].index
        
        # Intervals are of the form '(left, right]'
        if categories.str.startswith("(").any():
            
            # Keep only non-na categories
            non_na_categories = categories[categories != 'nan']
            
            # Retrieve original weights of the model (as a Series)
            weights = srr.df.loc[feature].loc[non_na_categories, srr.M]
            
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
    adversarial_examples = find_adversarial_examples(srr, X, y, can_change=non_monotonic_features,
                                                     unit_changes=True, allow_nan=False)
    
    if adversarial_examples.shape[0] > 0:
        return False
    return True


def poisoning_attack_point_removal(original_srr, X_train, y_train,
                                   feature, category=None,
                                   goal='nullify', col='normal', use_stats=False):
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
        use_stats   : Boolean indicating whether to use dataset statistics to improve the attack, default is False.
                        Cannot be set for goal 'remove_feature' (for now)

    Returns:
        removals: List with the indices of the data points that were removed
    """
    # Validation of parameters
    assert goal in ['flip_sign', 'remove_feature', 'nullify'], \
        "goal must be either 'flip_sign', 'remove_feature', or 'nullify'"
    assert col in ['original', 'relative', 'normal', 'M'], \
            "col must be either 'original', 'relative', 'normal', or 'M'"
    assert (category is not None) ^ (goal == 'remove_feature'), \
        "either goal is 'remove_feature' or category is not None"
    assert not (use_stats and goal == 'remove_feature'), \
        "cannot use statistics if goal is 'remove_feature'"

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

    else: # goal == 'remove_feature'
        # We approximate SRR by a version without cross-validation in this case
        model_kind = SRRWithoutCrossValidation

        if feature not in original_srr:
            raise ValueError('The given feature is already not in the model. Stopping.')


    # Define the set of candidate points to remove
    if use_stats:
        # Only consider points which have the specified (feature, category) and are associated to the label
        # This means that if the original weight is negative, we will only try to remove points that have label 0
        #  in order to push the classifier to believe that the category is not associated to a negative label
        # This can work to bring the feature to 0 or to make it flip sign
        candidates = list(y_train[(X_train[feature] == category) & (y_train == int(original_weight > 0))].index)
    else:
        # If we don't use stats, the set of candidate points is the whole training set
        candidates = list(y_train.index)

    # Define the base set from which we remove one feature at a time.
    # Initially it's a copy of the given training set.
    X_base = X_train.copy()
    y_base = y_train.copy()

    # The list with the points we remove from the training set
    removals = []

    srr = original_srr
    iteration = 0
    # Stop when we remove half the training points (or if the goal is achieved inside the while loop)
    while iteration < X_train.shape[0] * 0.75 and len(candidates) > 0:
        # Instantiate DataFrame to put results in
        res = pd.DataFrame(dtype=float, columns=[col, 'M'])

        # Iterate over all points over the candidate set (which we want to remove)
        for ith_point in tqdm(candidates, leave=True, position=0):

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

        # Add the best point to our list of removals and remove it from set of candidates
        removals.append(best_point)
        candidates.remove(best_point)
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

    raise ValueError(f'Attack failed, removed too many points. Tried removals: {removals}')


def poisoning_attack_hyperparameters(original_srr, X, y,
                                     feature, category=None,
                                     goal='remove_feature',
                                     train_size_list=None, seed_list=None, nbins_list=None, cv_list=None,
                                     Cs_list=None, max_iter_list=None, random_state_list=None):
    """
    Tries all possible tuples of hyper-parameters given the argument lists, and returns a dictionary with the
    first such tuple that achieved the goal, or None if the goal was never achieved

    Args:
        original_srr     : pre-trained SRR model on the given dataset
        X                : DataFrame with features, before any processing
        y                : Series with labels
        feature          : feature of the model to poison
        category         : optional (only defined if goal is 'flip_sign' or 'nullify')
        goal             : the goal of the poisoning attack
        train_size_list  : list of possible values for the train split size (fraction of entire data)
        seed_list        : list of possible values for the train/test split seed
        nbins_list       : list of possible values for the number of bins to discretize numerical features into
        cv_list          : list of possible values for the parameter cv (number of cross-validation folds)
        Cs_list          : list of possible values for the parameter Cs (number of regularization values to try)
        max_iter_list    : list of possible values for the parameter max_iter
        random_state_list: list of possible values for the parameter random_state
    Returns:
        A dictionary
    """
    assert goal in ['flip_sign', 'remove_feature', 'nullify'], \
        "goal must be either 'flip_sign', 'remove_feature', or 'nullify'"

    assert (category is not None) ^ (goal == 'remove_feature'), \
        "category can only be None if and only if goal is 'remove_feature'"

    assert train_size_list is not None, 'train_size_list must be defined'
    assert seed_list is not None, 'seed_list must be defined'
    assert nbins_list is not None, 'nbins_list must be defined'
    assert cv_list is not None, 'cv_list must be defined'
    assert Cs_list is not None, 'Cs_list must be defined'
    assert max_iter_list is not None, 'max_iter_list must be defined'
    assert random_state_list is not None, 'random_state_list must be defined'

    assert len(train_size_list) > 0, 'train_size_list must have at least one element'
    assert len(seed_list) > 0, 'seed_list must have at least one element'
    assert len(nbins_list) > 0, 'n_bins_list must have at least one element'
    assert len(cv_list) > 0, 'cv_list must have at least one element'
    assert len(Cs_list) > 0, 'Cs_list must have at least one element'
    assert len(max_iter_list) > 0, 'max_iter_list must have at least one element'
    assert len(random_state_list) > 0, 'random_state_list must have at least one element'

    for train_size, seed, nbins in tqdm(product(train_size_list, seed_list, nbins_list),
                                        total=len(train_size_list) * len(seed_list) * len(nbins_list)):

        X_train, _, y_train, _ = processing_pipeline(X, y, train_size=train_size, seed=seed, nbins=nbins)

        for cv, Cs, m_i, r_s in product(cv_list, Cs_list, max_iter_list, random_state_list):
            srr = SRR(k=original_srr.k, M=original_srr.M, cv=cv, Cs=Cs, max_iter=m_i, random_state=r_s)
            srr.fit(one_hot_encode(X_train), y_train)

            if (goal == 'flip_sign' and srr.get_weight(feature, category)
                                        * original_srr.get_weight(feature, category) < 0) \
                    or (goal == 'remove_feature' and feature not in srr) \
                    or (goal == 'nullify' and srr.get_weight(feature, category) == 0):

                print(f'Achieved goal! Resulting model:\n{srr}')
                return {'k': srr.k, 'M': srr.M, 'train_size': train_size, 'seed': seed, 'nbins': nbins,
                        'cv': cv, 'Cs': Cs, 'max_iter': m_i, 'random_state': r_s}

    return None


def poisoning_attack_drop_columns(original_srr, X_train, y_train,
                                  feature, category=None,
                                  goal='remove_feature',
                                  col='normal',
                                  greedy=True,
                                  assume_selection=True):
    """
    An attack consisting of removing columns from the training set of the model in order to achieve some goal.
    Can be greedy, in which case we remove one feature at a time, the one that brings us closer to our goal
    (very similar to the point removal attack), or can be extensive, in each case we try each possible
    subset of columns and train a model on them.

    Many different kinds of weights can be used, determined by the 'col' param:
        'original': from the inner logistic regression model (recommended for 'nullify' and 'remove_feature')
        'relative': from the SRR model right before multiplication by M and rounding
                        (not recommended, if weight is the biggest/smallest, then likely no changes)
        'normal'  : from the inner logistic regression model but normalized (recommended for 'flip_sign')
        'M'       : from the SRR model (not recommended, likely no changes)

    Args:
        original_srr    : pre-trained SRR model on the given dataset
        X_train         : training set DataFrame with the features, binned but not 1-hot encoded
        y_train         : training set Series with the labels
        feature         : feature of the model to poison
        category        : optional (only defined if goal is 'flip_sign' or 'nullify')
        col             : which kind of weight to use
        goal            : the goal of the poisoning attack
        greedy          : boolean saying if we should remove columns greedily (one at a time) or using the power-set
        assume_selection: boolean indicating whether we believe that removing a feature not chosen by the
                            model will have an influence (False) or not (True) on the new selection (only works in
                            non-greedy attack)
    Returns:
        removals: a list with the name of the columns that we removed
    """
    assert goal in ['flip_sign', 'remove_feature', 'nullify'], \
        "goal must be either 'flip_sign', 'remove_feature', or 'nullify'"
    assert col in ['original', 'relative', 'normal', 'M'], \
            "col must be either 'original', 'relative', 'normal', or 'M'"
    assert (category is not None) ^ (goal == 'remove_feature'), \
        "either goal is 'remove_feature' or category is not None"

    k = original_srr.k
    n = len(X_train.columns)

    if greedy:
        if goal in ['flip_sign', 'nullify']:
            # This is the SRR weight that we want to change
            original_weight = original_srr.get_weight(feature, category)

            # Check if weight is defined
            if np.isnan(original_weight):
                raise ValueError('The weight is does not exist in the model.')
            # Check whether the original weight is already 0
            if original_weight == 0 and goal == 'flip_sign':
                raise ValueError('Cannot flip weight sign of feature with weight 0.')
            if original_weight == 0 and goal == 'nullify':
                raise ValueError('The weight is already null and cannot be nullified.')

        else: # goal == 'remove_feature'
            if feature not in original_srr:
                raise ValueError('The given feature is already not in the model.')


        # The list with the columns we remove from the training set
        removals = []

        # Stop when there are only k features remaining, or if we achieved the goal before
        while len(removals) < n - k - 1:
            # Instantiate DataFrame to put results in
            res = pd.DataFrame(dtype=float, columns=[col, 'M'])

            for column in tqdm(X_train.drop(columns=removals+[feature]).columns, leave=False):
                X_reduced = X_train.drop(columns=removals+[column])

                # Fit SRR on the training set without the new colums
                srr = SRR.copy_params(original_srr)
                srr.fit(one_hot_encode(X_reduced), y_train)

                # If the goal is achieved, stop
                if (goal == 'flip_sign' and srr.get_weight(feature, category)
                    * original_srr.get_weight(feature, category) < 0) \
                        or (goal == 'remove_feature' and feature not in srr) \
                        or (goal == 'nullify' and srr.get_weight(feature, category) == 0):
                    removals.append(column)
                    print(f'Attack successful! Removals:\n{removals}\n\nResulting model:\n{srr}')
                    return removals


                if goal in ['flip_sign', 'nullify']:
                    try:
                        res.loc[column] = srr.df.loc[(feature, category)][[col, srr.M]].values
                    except KeyError:
                        res.loc[column] = (np.nan, np.nan)
                elif goal == 'remove_feature':
                    res.loc[column] = srr.df.loc[feature, [col, srr.M]].abs().mean().values

            # Pick the best column and add it to the list of removals
            if goal == 'flip_sign':
                # In this case we want the column which brings us closer to the opposite sign of the original weight
                # To do so, we keep the columns with the best rounded weights, and then take the best according to 'col'
                if original_weight > 0:
                    best_col = res.loc[res.M == res.M.min(), col].idxmin()
                else:
                    best_col = res.loc[res.M == res.M.max(), col].idxmax()
            elif goal == 'nullify':
                # In this case we are looking for the column that brings us closer to 0
                # We first get the rounded weights with smallest abs, then the best according to 'col'
                best_col = res.loc[res.M.abs() == res.M.abs().min(), col].abs().idxmin()
            else:  # goal == 'remove_feature':
                # We keep the point which brings us closer to zero
                # Again, we first get the rounded weights with smallest abs, then the best according to 'col'
                best_col = res.loc[res.M.abs() == res.M.abs().min(), col].abs().idxmin()

            removals.append(best_col)

        print(f'Attack failed, removed too many columns. Tried removals:\n{removals}')
        return None

    else: # not greedy, iterate over all possible
        def powerset(it, max_length=None):
            """Small function that computes the power-set of an iterable minus the empty set, and in reverse order"""
            if max_length is None:
                return chain.from_iterable(combinations(it, r) for r in range(len(it), 0, -1))
            else:
                return chain.from_iterable(combinations(it, r) for r in range(max_length, 0, -1))


        # In this case, always remove at least one feature in the model (other than the one we are attacking)
        if assume_selection:
            it = tqdm(
                list(chain.from_iterable(
                    product(combinations(pd.Index(original_srr.features).drop(feature), i),
                            powerset(X_train.columns.drop(original_srr.features), max_length=n-k-i))
                    for i in range(1, k)
                    )
                )
            )
        else:
            it = tqdm(list(powerset(X_train.columns.drop(feature), max_length=n-k)))

        # We define this function once to be able to flatten the removals (which are given in a weird format)
        flatten = lambda x: [e for l in x for e in l]

        for removals in it:
            if assume_selection:
                removals = flatten(removals)
            else:
                removals = list(removals)

            srr = SRR.copy_params(original_srr)
            srr.fit(one_hot_encode(X_train.drop(columns=removals)), y_train)

            if (goal == 'flip_sign' and srr.get_weight(feature, category)
                * original_srr.get_weight(feature, category) < 0) \
                    or (goal == 'remove_feature' and feature not in srr) \
                    or (goal == 'nullify' and srr.get_weight(feature, category) == 0):
                print(f'Achieved goal! Resulting model:\n{srr}')
                return removals

    return None
