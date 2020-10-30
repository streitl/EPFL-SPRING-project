import numpy as np

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss

def get_score(sklearn_model, X, y, criterion="AIC"):
    """
    Unused function for now, kept for posteriority.
    Computes the AIC or BIC score for an sklearn model by training it on the given data,
    and using the log likelihood of the fit.

    sklearn_model could be something like LogisticRegression(penalty='l1', solver='saga', max_iter=500, C=1e20)

    Arguments:
        sklearn_model: The sklearn model to compute the score for (ex
        X            : The predictive features
        y            : The predicted value
        criterion    : Either AIC or BIC

    Returns:
        the AIC or BIC score (a float)
    """
    sklearn_model.fit(X, y)

    # Properties of the model to be used for AIC/BIC computation
    n_feats = len(X.columns) + 1
    n_obs = len(y)

    # Get the log likelihood from the model
    y_pred = sklearn_model.predict(X)
    log_likelihood = -log_loss(y, y_pred, normalize=False)

    # Get score using AIC or BIC formula
    return (2 if criterion == "AIC" else np.log(n_obs)) * n_feats - 2 * log_likelihood


def forward_stepwise_regression(X, y, k, criterion="AIC", kind="linear", verbose=False):
    """
    Performs forward stepwise regression.
    Iteratively selects the feature column whose regression model has
    the lowest criteria score and adds it to the set of features.
    
    Arguments:
    - X        : DataFrame with the features, one-hot encoded and with a two-level column index
    - y        : DataFrame with the target
    - k        : Number of features to be selected
    - criterion: String indicating which criteria to use to score a model
    - kind     : Which kind of regression to perform, linear or logistic
    - verbose  : Boolean indicating whether to print intermediate results
    
    Returns:
    - selected_features: A list of k features from X
    """
    # The number of features to be selected must be smaller than the number of columns
    assert k <= len(X.columns.levels[0]), "the given dataset has less than k features"
    assert criterion in ["AIC", "BIC"], "the criterion must be AIC or BIC"
    assert kind in ["linear", "logistic"], "the model can only be linear or logistic"
    
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

            if kind == "linear":
                # Fit a linear regression model
                lm = sm.GLS(endog=y,
                            exog=sm.add_constant(X[selected_features + [candidate]]))\
                    .fit()

                score = lm.aic if criterion == 'AIC' else lm.bic
            else:
                # Fit a logistic regression model
                logit = LogisticRegression(penalty='l1', solver='saga', max_iter=500, C=1e20)

                score = get_score(logit, X=X[selected_features + [candidate]], y=y, criterion=criterion)
            
            if verbose: print(f"{candidate} {score:.1f}", end="\n")
            
            if best_candidate is None or score < best_score:
                best_candidate = candidate
                best_score = score
        
        if verbose: print(f"\n--> Adding {best_candidate} {best_score:.1f} to {selected_features}\n\n")
        
        # Move the best candidate feature to the list of selected features
        candidate_features.remove(best_candidate)
        selected_features.append(best_candidate)
        
    return selected_features
