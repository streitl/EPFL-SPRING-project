import pandas as pd
import numpy as np

import pickle


from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

from .feature_selection import forward_stepwise_regression


class RoundedWeightClassifier(BaseEstimator, ClassifierMixin):
    """
    An sklearn estimator with rounded weights.
    Used as a super class that the SRR model and its approximations can extend from.
    """
    def __init__(self, M):
        """
        Initializes M, and also creates some attributes needed for all subclasses.
        Args:
            M: Maximum amplitude of the rounded weights
        """
        assert int(M) == M, "M must be an integer"
        assert M > 0, "M must be positive"
        assert M <= 10, "M must be reasonably small (10 or less)"

        self.M = M

        # Initialize some attributes
        self.inner_model = None
        self.features = None
        self.df = None


    def build_df_of_rounded_weights(self, X):
        """
        Retrieves the inner model weights, rescales and rounds them, and saves the result in self.df
        Args:
            X: DataFrame used to train the model, we just need it to retrieve the categories of the selected features
        """
        # Retrieve weights from model
        weights = self.inner_model.coef_[0]
        bias = self.inner_model.intercept_.item()

        # Create DataFrame with MultiIndex in the columns (feature, category)
        mux = pd.MultiIndex.from_tuples(list(X[self.features].columns) + [('bias', '')])
        self.df = pd.DataFrame(np.append(weights, bias), index=mux, columns=['original'])
        self.df.index.names = ["Feature", "Category"]
        self.df.columns.names = ["Weight"]

        # Rescaling and rounding the weights (including bias)
        max_weight = np.abs(self.df['original']).max()
        # If all weights are 0 then it does not make sense to set the model weights to anything
        if max_weight == 0:
            raise ValueError("All weights of the internal logistic regression model are 0.")

        # This is useful for poisoning attacks
        self.df['relative'] = self.df['original'] / max_weight
        self.df['normal'] = (self.df['original'] - self.df['original'].mean()) / self.df['original'].std()

        # We do the rounding for multiple M-values since it allows us to have many models in one training
        for M in range(1, 10 + 1):
            self.df[M] = (self.df['relative'] * M).round().astype(int)


    def fit(self, X, y):
        raise NotImplemented


    def predict(self, X, M=None):
        """
        Predicts the label of each input sample.

        Args:
            X: DataFrame with the features, one-hot encoded and with a two-level column index
            M: Amplitude of the weights, acts as a selector for the corresponding column in the model.
                If None, the value given to the constructor of the model is used.

        Returns:
            predictions: Numpy array with the binary predictions
        """
        if M is None:
            M = self.M
        else:
            assert int(M) == M, "M must be an integer"
            assert M > 0, "M must be positive"
            assert M <= 10, "M must be reasonably small"

        # Initialize a numpy array of zeros with size the number of samples in X
        n_rows = len(X.index)
        predictions = np.zeros(n_rows)

        # Add the weight of each feature present both in the model and in X
        for feature in self.df.index.intersection(X.columns):
            predictions += X[feature] * self.df.loc[feature, M]
        # Add the bias
        predictions += self.df.loc[('bias', ''), M]

        # Apply the decision threshold
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = 0

        return predictions.astype(int)


    def get_weight(self, feature, category, column='M'):
        """
        Getter for the model's weight for the given (feature, category) pair.
        The 'column' parameter determines whether to get the raw weight from the inner logistic model,
            the relative weight from the inner logistic model, the normalized weight from the inner logistic model,
            or the rounded weight from the SRR model.

        Args:
            feature : Name of the feature we want to get
            category: Name of the category we want to get
            column  : Determines which weight to get

        Returns:
            The weight of the feature corresponding to feature, category, column
        """
        assert column in ['original', 'relative', 'normal', 'M'], \
            "column must be either 'original', 'relative', 'normal', or 'M'"
        try:
            if column == 'M':
                return self.df.loc[(feature, category), self.M]
            else:
                return self.df.loc[(feature, category), column]
        except:
            raise ValueError(f'The model does not have weights for ({feature}, {category}) in column {column}')


    def __contains__(self, feature):
        """
        Re-definition of the built-in function to allow checking if a feature is part of the model or not
        Args:
            feature: string name of the feature to check

        Returns:
            True if feature contained in the model, otherwise false
        """
        return feature in self.features



class SRR(RoundedWeightClassifier):
    """
    An sklearn BaseEstimator implementing the Select-Regress-Round model.

    Extends from RoundedWeightClassifier, which defines how to build the DataFrame with the features,
    and how to do the prediction.
    """

    def __init__(self, k, M, cv=5, Cs=20, n_jobs=-1, max_iter=150, random_state=42):
        """
        The SRR class constructor.
        
        Args:
            k           : # of features (columns in a non 1-hot encoded matrix) to be used in the model
            M           : Amplitude of the weights of the model
            cv          : Cross-validation folds to perform when training the logistic regression model
            Cs          : Regularization values to try for the logistic regression model
            n_jobs      : # of jobs that can run in parallel during cross validation
            max_iter    : Iterations for the logistic regression model optimization
            random_state: Int to be used for reproducibility
        """
        # Calls the parent's constructor
        super().__init__(M)

        assert int(k) == k, "k must be an integer"
        assert k > 0, "k must be positive"

        # Store parameters
        self.k = k
        self.cv = cv
        self.Cs = Cs
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.random_state = random_state

        self.inner_model = LogisticRegressionCV(
            cv=self.cv, penalty="l1",
            Cs=self.Cs, solver="saga",
            fit_intercept=True,
            n_jobs=self.n_jobs,
            max_iter=self.max_iter,
            random_state=self.random_state
        )


    @staticmethod
    def copy_params(model):
        """
        An SRR constructor which uses the parameters of the given model.

        Args:
            model: The model whose parameters we want to copy

        Returns:
            a new SRR model with exactly the same parameters as the given model
        """
        return SRR(k=model.k, M=model.M, cv=model.cv, Cs=model.Cs,
                   n_jobs=model.n_jobs, max_iter=model.max_iter, random_state=model.random_state)


    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X, y, kind="linear", verbose=0):
        """
        Performs the Select-Regress-Round training procedure:
            1. Using forward stepwise logistic regression, selects the best k features
            2. Given the k features, fit a logistic regression model with l1 regularization
            3. Given the logistic model, rescale and round the weights
        At the end, the trained model is stored and represented in self.df
        
        Args:
            X      : DataFrame with the features, one-hot encoded and with a two-level column index
            y      : DataFrame with the target
            kind   : Regression to be used for the feature selection, either linear or logistic
            verbose: Integer level of verbosity
        """
        assert self.k <= len(X.columns.levels[0]), "the given dataset has less than k features"

        ## Step 1. Select k features
        if verbose >= 2: print("Selecting", self.k, "features...")
        self.features = forward_stepwise_regression(X, y, self.k, verbose=verbose, kind=kind)
        if verbose: print("Selected features", ', '.join(self.features))


        ## Step 2. Train L1-regularized logistic regression model
        if verbose >= 2: print("Cross-validating the logistic regression model...")
        # Fit inner logistic regression model with cross-validation and L1-regularization
        self.inner_model.fit(X[self.features], y)
        if verbose:
            acc = accuracy_score(y, self.inner_model.predict(X[self.features])) * 100
            baseline = max(y.mean(), 1-y.mean()) * 100
            print(f"Logistic model accuracy of {acc:.1f} % on the training set (baseline {baseline:.1f} %)")


        ## Step 3. Rescale and round the model weights
        self.build_df_of_rounded_weights(X)
        if verbose >= 2: print("Done!")


    def save(self, dataset_name):
        """
        Saves the model into a file using pickle.
        The file name is a combination of the dataset name, and of the k, M values of the model.
        
        Args:
            dataset_name: String with the name of the dataset that was used to train the model.
        """
        model_path = f"models/srr_{dataset_name}_k_{self.k}_M_{self.M}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

        print(f"Saved SRR model to {model_path}")


    @staticmethod
    def load(dataset_name, k, M):
        """
        Loads an SRR model with the specified properties.
        The file name is a combination of the dataset name, and of the k, M values of the model.

        Args:
            dataset_name: String with the name of the dataset that was used to train the model that we want to load
            k           : The number of features that the wanted model has selected
            M           : The amplitude of the weights of the wanted model

        Returns:
            model: The SRR model, loaded from a file
        """
        model_path = f"models/srr_{dataset_name}_k_{k}_M_{M}.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        print(f"Loaded SRR model from {model_path}")
        return model



class RoundedLogisticRegression(RoundedWeightClassifier):
    """
    This class is a simplified version of SRR without feature selection and cross-validation (the features to use and
    the L1-penalty are given in the constructor).

    It should produce predictions very similar to that of SRR, even when removing some points from the training set,
    but the advantage is that the training time of this class is a lot smaller.
    """

    def __init__(self, M, features, C, max_iter, random_state):
        """
        Constructor, similar to SRR, but does not take k and Cs, and instead takes features and C as arguments.

        Args:
            M           : Maximum amplitude of the rounded weights
            features    : List of strings corresponding to the features to be used by the model
            C           : Inverse regularization strength to be used by the logistic regression
            max_iter    : Maximum number of iterations of the logistic regression
            random_state: Given to LogisticRegression to get reproducible results
        """
        # Calls the parent's constructor
        super().__init__(M)

        self.features = features
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

        # The inner model is LogisticRegression as opposed to SRR's LogisticRegressionCV
        self.inner_model = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state
        )


    @staticmethod
    def from_srr(srr):
        """Copies the selected features and the L1-penalty from an already trained SRR model."""
        return RoundedLogisticRegression(
            M=srr.M,
            features=srr.features,
            C=srr.inner_model.C_.item(),
            max_iter=srr.max_iter,
            random_state=srr.random_state
        )


    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X, y):
        """
        Similar to SRR's training procedure but without feature selection and cross-validation.

        Args:
            X: DataFrame with the features, one-hot encoded and with a two-level column index
            y: DataFrame with the target
        """
        # Step 1. Fit Logistic Regression model (no cross-validation)
        self.inner_model.fit(X[self.features], y)
        # Step 2. Rescale and round the weights
        self.build_df_of_rounded_weights(X)



class SRRWithoutCrossValidation(RoundedWeightClassifier):
    """
    This class is a simplified version of SRR without cross-validation (the L1-penalty is given in the constructor).

    It should produce predictions very similar to that of SRR, even when removing some points from the training set,
    but the advantage is that the training time of this class is a lot smaller.
    """

    def __init__(self, k, M, C, max_iter, random_state):
        """
        Constructor, similar to SRR, but does not take Cs, and instead takes and C as argument.

        Args:
            k           : Number of features to be selected
            M           : Maximum amplitude of the rounded weights
            C           : Inverse regularization strength to be used by the logistic regression
            max_iter    : Maximum number of iterations of the logistic regression
            random_state: Given to LogisticRegression to get reproducible results
        """
        # Calls the parent's constructor
        super().__init__(M)

        # Store the arguments
        self.k = k
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

        # The inner model is LogisticRegression as opposed to SRR's LogisticRegressionCV
        self.inner_model = LogisticRegression(
            penalty='l1',
            solver='saga',
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state
        )

    @staticmethod
    def from_srr(srr):
        return SRRWithoutCrossValidation(
            k=srr.k,
            M=srr.M,
            C=srr.inner_model.C_[0],
            max_iter=srr.max_iter,
            random_state=srr.random_state
        )


    @ignore_warnings(category=ConvergenceWarning)
    def fit(self, X, y):
        """
        Similar to SRR's training procedure but without cross-validation.

        Args:
            X: DataFrame with the features, one-hot encoded and with a two-level column index
            y: DataFrame with the target
        """
        # Step 1. Select k features with forward stepwise regression
        self.features = forward_stepwise_regression(X, y, self.k)
        # Step 2. Fit logistic regression model
        self.inner_model.fit(X[self.features], y)
        # Step 3. Rescale and round the inner model weights
        self.build_df_of_rounded_weights(X)
