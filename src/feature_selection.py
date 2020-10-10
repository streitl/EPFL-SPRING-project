import pandas as pd
import statsmodels.api as sm


def forward_stepwise_regression(X, y, k, criteria="AIC"):
    """
    Performs forward stepwise regression.
    Iteratively selects the feature column whose logistic model has
    the lowest criteria score and adds it to the set of features.
    
    Arguments:
    - X:        DataFrame with the features, one-hot encoded and with a two-level column index
    - y:        DataFrame with the target
    - k:        Number of features to be selected
    - criteria: String indicating which criteria to use to score a model
    
    Returns:
    - selected_features: A list of k features from X
    """
    # The number of features to be selected must be smaller than the number of columns
    assert k <= len(X.columns.levels[0]), "the given dataset has less than k features"
    assert criteria in ["AIC", "BIC"], "the criteria must be AIC or BIC"
    
    # We keep track of all features already added to the model and all that can be added
    candidate_features = list(X.columns.levels[0])
    selected_features = []
    
    # Only stop when we have k features
    while len(selected_features) < k:
        # Simple initialization to get the best candidate
        best_candidate = None
        best_score = None
        
        # Keep candidate that results in the best (smallest) score
        for candidate in candidate_features:
            # Fit a logistic regression statistical model
            regr = sm.GLS(endog=y,
                          exog=sm.add_constant(X[selected_features + [candidate]]),
                          family=sm.families.Binomial()).fit()
            # Get score
            if criteria == "AIC": score = regr.aic
            elif criteria == "BIC": score = regr.bic
            else: score = 0
            
            if best_candidate is None or score < best_score:
                best_candidate = candidate
                best_score = score
        
        # Move the best candidate feature to the list of selected features
        candidate_features.remove(best_candidate)
        selected_features.append(best_candidate)
        
    return selected_features

