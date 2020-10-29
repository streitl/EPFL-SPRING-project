import pandas as pd

from itertools import product

from tqdm import tqdm

from .preprocessing import one_hot_encode


def find_adversarial_examples(srr_model, X, y, can_change, unit_changes=False, allow_nan=True):
    """
    Given an SRR model, data points, and the features which can be changed, produces a list of 
    adversarial examples that have changed the model prediction label.
    
    Arguments:
    - srr_model   : Trained SRR model
    - X           : DataFrame with categorical features, preprocessed but not 1-hot encoded
    - y           : Series with the labels
    - can_change  : List with features in X that we can modify
    - unit_changes: Boolean indicating whether to change a single feature at a time
    - allow_nan   : Boolean indicating whether nans are allowed changes or not
    
    Returns:
    - adversaries_and_originals: Dataframe with adversarial examples
                                 and the corresponding original datapoints
    """
    # Keep track of which features can be modified
    modifiable_features = set(srr_model.selected_features).intersection(set(can_change))
    assert len(modifiable_features) > 0, "No features can be modified with the given parameters"
    
    # Create model predictions
    y_pred = srr_model.predict(one_hot_encode(X))
    
    # Only keep data points whose label was correctly predicted,
    # and only keep the features selected by the SRR model
    correctly_classified = X.loc[y_pred == y, srr_model.selected_features]
    
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
    adversaries_and_originals = pd.merge(true_adversaries, X[srr_model.selected_features],
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
    
    Arguments:
    - model: Trained SRR model
    
    Returns:
    - Boolean indicating whether the binned features of the model pass monotonicity check
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
    
    print(f"Found the following adversaries:\n{adversarial_examples}")
    
    if adversarial_examples.shape[0] > 0:
        return False
    return True

