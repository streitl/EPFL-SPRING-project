import pandas as pd

from itertools import product

from tqdm import tqdm

from .preprocessing import one_hot_encode


def find_adversarial_examples(srr_model, X, y, can_change):
    """
    Given an SRR model, data points, and the features which can be changed, produces a list of 
    adversarial examples that have changed the model prediction label.
    
    Arguments:
    - srr_model : Trained SRR model
    - X         : DataFrame with categorical features, preprocessed but not 1-hot encoded
    - y         : Series with the labels
    - can_change: List with features in X that we can modify
    
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
    feat_to_cats = {feat: list(X[feat].unique()) for feat in X.columns}
    
    # Create a list with tuples of possible feature changes
    possible_changes = list(product(*[feat_to_cats[feat] for feat in modifiable_features]))
    
    
    ## Construct a list of potential adversarial examples by deforming each correctly classified sample
    potential_adversaries = pd.DataFrame(columns=correctly_classified.columns)
    
    # Iterate over correctly classified points
    for index, data in tqdm(correctly_classified.iterrows(), total=correctly_classified.shape[0]):
        
        # Go through precomputed tuples of feature changes
        for change in possible_changes:
            
            # Modify the original data point and add it to the list
            deformed = data.copy()
            deformed.loc[modifiable_features] = change
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



def verifies_monotonicity(model):
    """
    Takes a trained SRR model, and for each selected feature whose categories are intervals (and possibly nan),
    checks that the weights corresponding to the intervals are either in increasing or decreasing order.
    
    Arguments:
    - model: Trained SRR model
    
    Returns:
    - Boolean indicating whether the model verifies monotonicity
    """
    # Iterate over all features that the model uses
    for feature in model.df.index.levels[0]:
        
        # Retrieve the categories corresponding to this feature
        categories = model.df.loc[feature].index
        
        # Intervals are of the form (left, right]
        if categories.str.startswith("(").any():
            
            # Keep only non-na categories
            non_na_categories = categories[categories != 'nan']
            
            # Retrieve original weights of the model (as a Series)
            weights = model.df.loc[feature].loc[non_na_categories, model.M]
            
            # Indicators of whether the weights have already increased/decreased so far
            increased, decreased = False, False
            
            # Go trough pairs of current and previous weights
            for current, previous in zip(weights.iloc[1:], weights):
                if current > previous:
                    increased = True
                elif current < previous:
                    decreased = True
                
                # If both are true, then monotonicity is not verified for this feature
                if increased and decreased:
                    print("Monotonicity check failed for:", model.df.loc[[feature]][model.M], sep='\n')
                    return False
    
    # If no feature broke monotonicity, then the model verifies monotonicity
    return True

