import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def load_and_process_german():
    # Load the data from the csv file
    df = pd.read_csv("data/german.data", sep=" ", header=None)
    
    # The last column is the label
    df = df.rename(columns={20: "y"})
    
    # We want 0 to be the label "bad" and 1 to be "good"
    df.y = 2 - df.y
    
    # Bin some numerical features into 5 quantiles
    features_to_bin = [1, 4, 12]
    for idx in features_to_bin:
        df[idx] = pd.qcut(df[idx], q=5, duplicates="drop")
    
    
    # One-hot encode all the categorical features except the label y
    X = pd.DataFrame()
    
    for col in df.columns:
        # Don't add label to X
        if col == "y":
            continue
            
        # If the column was originally numerical,
        # add a prefix with the original column number to the new column name
        prefix = None
        numerical_features = [1, 4, 7, 10, 12, 15, 17]
        if col in numerical_features:
            prefix = "A" + str(col + 1)
        
        # One hot encode a single column
        one_hot_df = pd.get_dummies(df[col], prefix=prefix)
        
        # Rename columns A11, A30 to A1_1, A3_0 to improve legibility
        for c in one_hot_df.columns:
            if c.startswith("A") and "_" not in c:
                new_name = c[:-1] + "_" + c[-1:]
                one_hot_df = one_hot_df.rename(columns={c: new_name})
                
        # Append each boolean feature to the final data set (at the end)
        for c in one_hot_df.columns:                
            X.insert(loc=len(X.columns), column=c, value=one_hot_df[c])
    
    y = df.y
    
    return X, y


def forward_stepwise_regression(X, y, k):
    """
    X: DataFrame with the features
    y: DataFrame with the labels
    k: Number of features to be selected
    """
    # The number of features to be selected must be smaller than the number of columns
    assert(k < len(X.columns))
    
    # We keep track of all features already added to the model and all that can be added
    candidate_features = list(X.columns)
    selected_features = []
    
    # Only stop when we have k features
    while len(selected_features) < k:
        # Simple initialization to get the best candidate
        best_candidate = None
        best_score = -1
        
        # Create a logistic regression model,
        # fit it on the selected features and the current candidate,
        # compute log loss and compare it to the best
        for candidate in candidate_features:
            # Model creation and fitting
            model = LogisticRegression(penalty="none")
            model.fit(X[selected_features + [candidate]], y)
            
            # Log loss of the created model
            score = log_loss(y, model.predict(X[selected_features + [candidate]]))
            
            if score > best_score:
                best_candidate = candidate
                best_score = score
        
        # Move the best candidate feature to the list of selected features
        candidate_features.remove(best_candidate)
        selected_features.append(best_candidate)
        
    return selected_features


def select_regress_round(X, y, k, M):
    """
    k: Number of features to be selected
    M: Magnitude of the weights
    """
    # Do forward stepwise regression to select only k features
    selected_features = forward_stepwise_regression(X, y, k)
    
    # Train L1-regularized logistic regression model
    model = LogisticRegression(penalty="l1", solver="saga", C=1000)
    model.fit(X[selected_features], y)
    weights = model.coef_
    
    # Round the weights using M
    w_max = np.abs(weights).max()
    assert(w_max > 0)
    final_weights = (weights * M / w_max).round().tolist()[0]
    
    # Combine features and feature weights to output model
    return dict(zip(selected_features, final_weights))


X, y = load_and_process_german()

model = select_regress_round(X, y, k=5, M=10)

print(model)