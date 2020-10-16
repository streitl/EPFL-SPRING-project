import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from .feature_selection import forward_stepwise_regression


class SRR(BaseEstimator, ClassifierMixin):
    """
    An sklearn BaseEstimator implementing the Select-Regress-Round model.
    """
    
    def __init__(self, k, M, cv=5, Cs=1000, n_jobs=-1, max_iter=150):
        """
        The SRR class constructor.
        
        Arguments:
        - k       : # of features to be used in the model
        - M       : Amplitude of the weights of the model
        - cv      : # of cross-validation folds to perform when training the logistic regression model
        - Cs      : # of regularization values to try for the logistic regression model
        - n_jobs  : # of jobs that can run in parallel during cross validation
        - max_iter: # of iterations for the logistic regression model optimization
        """
        assert int(k) == k, "k must be an integer"
        assert k > 0, "k must be positive"
        
        assert int(M) == M, "M must be an integer"
        assert M > 0, "M must be positive"
        assert M <= 10, "M must be reasonably small (10 or less)"
        
        # Calls the parent's constructor, not sure if needed
        super().__init__()
        
        # Store parameters
        self.k = k
        self.M = M
        self.cv = cv
        self.Cs = Cs
        self.n_jobs = n_jobs
        self.max_iter = max_iter
    
    
    def fit(self, X, y, verbose=False):
        """
        Performs the Select-Regress-Round training procedure:
            1. Using forward stepwise logistic regression, selects the best k features
            2. Given the k features, fit a logistic regression model with l1 regularization
            3. Given the logistic model, rescale and round the weights
        At the end, the trained model is stored and represented in self.df
        
        Arguments:
        - X      : DataFrame with the features, one-hot encoded and with a two-level column index
        - y      : DataFrame with the target
        - verbose: Boolean value indicating whether to print intermediary results
        """
        assert self.k <= len(X.columns.levels[0]), "the given dataset has less than k features"
        
        ## Step 1. Select k features
        if verbose: print("Selecting", self.k, "features...")
        selected_features = forward_stepwise_regression(X, y, self.k)
        if verbose: print("Selected features", ', '.join(selected_features))
        
        # Store the selected features in the model
        self.selected_features = selected_features
        
        
        ## Step 2. Train L1-regularized logistic regression model
        logistic_model = LogisticRegressionCV(
            cv=self.cv, penalty="l1", 
            Cs=self.Cs, solver="saga", 
            fit_intercept=True,
            n_jobs=self.n_jobs,
            max_iter=self.max_iter
        )
        
        if verbose: print("Cross-validating the logistic regression model...")
        logistic_model.fit(X[selected_features], y)
        if verbose:
            acc = accuracy_score(y, logistic_model.predict(X[selected_features])) * 100
            baseline = max(y.mean(), 1-y.mean()) * 100
            print("Logistic model accuracy of {:.1f} % on the training set (baseline {:.1f} %)".format(acc, baseline))
        
        
        ## Step 3. Retrieve the model weights
        # Accessing the sklearn params
        feature_weights = logistic_model.coef_[0]
        bias = logistic_model.intercept_.item()
        
        # Constructing the model dataframe
        mux = pd.MultiIndex.from_tuples(list(X[selected_features].columns) + [('bias', '')])
        self.df = pd.DataFrame(np.append(feature_weights, bias),
                               index=mux,
                               columns=['original'])
        self.df.index.names = ["Feature", "Category"]
        self.df.columns.names = ["M"]
        
        ## Rescaling and rounding the weights (including bias)
        w_max = np.abs(self.df['original']).max()
        # We do the rounding for multiple M-values since it allows us to have many models in one training
        for M in range(1, 10+1):
            # It might be that all weights are 0 because of L1-regularization
            if w_max != 0:
                self.df[M] = pd.Series((self.df['original'] * M / w_max).round(), dtype=int)
            else:
                self.df[M] = 0
        
        if verbose: print("Done!")
    
    
    def predict(self, X, M=None):
        """
        Predicts the label of each input sample.
        
        Arguments:
        - X: DataFrame with the features, one-hot encoded and with a two-level column index
        - M: Amplitude of the weights, acts as a selector for the corresponding column in the model.
             If None, the value given to the constructor of the model is used.

        Returns:
        - prediction: a numpy array with the binary predictions
        """
        if M is None:
            M = self.M
        assert int(M) == M, "M must be an integer"
        assert M > 0, "M must be positive"
        assert M <= 10, "M must be reasonably small"
        
        # Initialize a numpy array of zeros with size the number of samples in X
        n_rows = len(X.index)
        predictions = np.zeros(n_rows)
        
        # Add the weight of each feature
        for feature in self.df.index.intersection(X.columns):
            predictions += X[feature] * self.df.loc[feature, M]
        # Add the bias
        predictions += self.df.loc[('bias', ''), M]
        
        # Apply the decision threshold
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = 0
        
        return predictions.astype(int)

