import pandas as pd
import sklearn as sk
import numpy as np

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score


def load_and_process_german():
    """
    Loads the data from the german dataset, and processes it by 
    binning numerical features, and then 1-hot encoding all the features.
    
    Returns X, the dataframe with boolean features only, including a constant bias,
    and y, the dataframe with the class label.
    """
    # Load the data from the csv file
    df = pd.read_csv("data/german.data", sep=" ", header=None)
    # Make indices start at 1 for R compatibility
    df.index += 1
    
    # The last column is the label
    df = df.rename(columns={20: "y"})
    
    # We want 0 to be the label "bad" and 1 to be "good"
    df.y = 2 - df.y
    
    # Divide some of the numerical features into bins
    features_to_bin = [1, 4, 12]
    for idx in features_to_bin:
        df[idx] = pd.qcut(df[idx], q=3)
    
    
    # Create the dataframe of features
    X = pd.DataFrame()
    
    # One-hot encode all the categorical features except the label y
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
    
    # Group boolean attributes corresponding to the same original column together
    X.columns = pd.MultiIndex.from_tuples([c.split("_") for c in X.columns])
    
    y = df.y
    
    return X, y


def forward_stepwise_regression(X, y, k):
    """
    Performs forward stepwise regression.
    Iteratively selects the feature with the lowest AIC score and adds it to the set of features.
    
    X: DataFrame with the features
    y: DataFrame with the labels
    k: Number of features to be selected
    
    Returns a list of k features.
    """
    # The number of features to be selected must be smaller than the number of columns
    assert(k < len(X.columns))
    
    # We keep track of all features already added to the model and all that can be added
    candidate_features = list(X.columns.levels[0])
    selected_features = []
    
    # Only stop when we have k features
    while len(selected_features) < k:
        # Simple initialization to get the best candidate
        best_candidate = None
        best_score = None
        
        # Keep candidate that results in the best (smallest) AIC score
        for candidate in candidate_features:
            # Get AIC score
            regr = OLS(y, add_constant(X[selected_features + [candidate]])).fit()
            score = regr.aic
            
            if best_candidate is None or score < best_score:
                best_candidate = candidate
                best_score = score
        
        # Move the best candidate feature to the list of selected features
        candidate_features.remove(best_candidate)
        selected_features.append(best_candidate)
        
    return selected_features


class SRR():
    """
    Wrapper class for a select-regress-round model.
    
    It contains the attributes:
    - weights, a dictionary of feature name to integer weight
    - threshold, the float decision threshold
    
    
    """
    
    def __init__(self, feature_names, logistic_model, k, M):
        """
        Constructs a select-regress-round model with the given parameters.
        
        feature_names : List containing the name of the features that the model uses
        logistic_model: Trained scikit-learn logistic regression model with lasso regularization
        k             : Number of features used in the model
        M             : Amplitude of the weights
        """
        # Keep this information to show later
        self.k = k
        self.M = M
        
        # Extract the weight and bias from the logistic model
        feature_weights = logistic_model.coef_[0]
        bias = logistic_model.intercept_.item()
        
        # Rescale and round the weights
        w_max = np.abs(feature_weights).max()
        rounded_weights = (feature_weights * M / w_max).round().astype(int).tolist()
        
        # Store the model parameters
        self.weights = dict(zip(feature_names, rounded_weights))
        self.threshold = np.ceil(-bias)
    
    
    def predict(self, X):
        """
        Predicts the label of each input sample.

        X: DataFrame with the input samples

        Returns a numpy array with the binary predictions.
        """
        # Initialize a numpy array of zeros with size the number of samples in X
        n_rows = len(X.index)
        predictions = np.zeros(n_rows, dtype=int)

        # Add the weight of each feature
        for feature, weight in self.weights.items():
            predictions += X[feature] * weight

        # Decision threshold
        predictions[predictions >= self.threshold] = 1
        predictions[predictions < 0] = 0

        return predictions
    
    
    def show_accuracy(self, X, y):
        """
        Computes the model prediction and prints its accuracy, along with the baseline.
        
        X: DataFrame with the features
        y: DataFrame with the labels
        """
        # Model prediction and accuracy
        y_pred = self.predict(X)
        accuracy = (y == y_pred).astype(int).mean()
        
        # The baseline is the proportion of the largest class
        baseline = max(1 - y.mean(), y.mean())
        
        print("Training accuracy of %.2f (baseline %.2f)" % (accuracy, baseline))
    

    def __str__(self):
        """
        Returns the string representation of the model, including the weights and decision threshold.
        """
        # We create a dataframe for pretty printing
        mux = pd.MultiIndex.from_tuples(self.weights.keys(), names=["Attribute", "Category"])
        df = pd.DataFrame(self.weights.values(), index=mux, columns=["Weights"])

        s = "SRR model with k=%d and M=%d:" % (self.k, self.M)
        s += "\n\n%s\n\n" % (df.to_string())
        s += "Predict class 1 if sum >= %d, and 0 otherwise." % (self.threshold)
        
        return s


def select_regress_round(X, y, k, M, verbose=False):
    """
    Trains and returns a select-regress-round model on the data with the given parameters.
    
    X: DataFrame with the features
    y: DataFrame with the labels
    k: Number of features to be selected
    M: Magnitude of the weights
    
    Returns an object of class SRR.
    """
    # Do forward stepwise regression to select only k features
    if verbose: print("Selecting %d features..." % k)
    selected_features = forward_stepwise_regression(X, y, k)
    if verbose: print("Selected features", selected_features, "\n")
    
    # Train L1-regularized logistic regression model
    if verbose: print("Running cross-validation for logistic regression...")
    clf = LogisticRegressionCV(cv=5, penalty="l1", Cs=1000, solver="saga")
    clf.fit(X[selected_features], y)
    accuracy = accuracy_score(y, clf.predict(X[selected_features]))
    if verbose: print("Logistic regression training accuracy: %.2f" % (accuracy))
    
    # Construct the model
    srr_model = SRR(feature_names=X[selected_features],
                     logistic_model=clf,
                     k=k,
                     M=M)
    
    return srr_model


X, y = load_and_process_german()

model = select_regress_round(X, y, k=5, M=3, verbose=True)

model.show_accuracy(X, y)

print(model)
