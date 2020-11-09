import unittest

import pandas as pd
import numpy as np

import src.preprocessing as preprocessing

class TestPreprocessing(unittest.TestCase):

    def test_drop_useless_columns(self):
        X = pd.DataFrame(columns=['useful1', 'useful2', 'useless'])
        X['useful1'] = np.random.random(size=100)
        X['useful2'] = np.random.random(size=100)
        X['useless'] = 3

        X_new = preprocessing.drop_useless_columns(X)

        # Check whether the columns of the original df were unchanged
        self.assertTrue((X.columns == ['useful1', 'useful2', 'useless']).all())
        # Check whether the new df has one fewer column (the useless)
        self.assertTrue((X_new.columns == ['useful1', 'useful2']).all())
        # Check whether the two dfs have the same number of points
        self.assertEqual(X.shape[0], X_new.shape[0])
        # Check whether the useful columns stayed the same
        self.assertTrue((X[['useful1', 'useful2']] == X_new[['useful1', 'useful2']]).all().all())


    def test_find_boundaries_nans(self):
        series = pd.Series([1, 34, 2, 1.2, np.nan, 0, 4, 23, np.nan])
        boundaries = preprocessing.find_boundaries(series, nbins=3)
        # Check whether there are nbins+1 breaks, and the first and last are - and + inf
        self.assertTrue(len(boundaries) == 4)
        self.assertEqual(boundaries[0], float('-inf'))
        self.assertEqual(boundaries[-1], float('inf'))


    def test_find_boundaries_no_nans(self):
        series = pd.Series([1, 34, 2, 1.2, 16, 0, 4, 23, 42])
        boundaries = preprocessing.find_boundaries(series, nbins=3)
        # Check whether there are nbins+1 breaks, and the first and last are - and + inf
        self.assertTrue(len(boundaries) == 4)
        self.assertEqual(boundaries[0], float('-inf'))
        self.assertEqual(boundaries[-1], float('inf'))


    def test_bin_features(self):
        X_train = pd.DataFrame(columns=['cat1', 'cat2', 'num1', 'num2'])
        X_train['cat1'] = pd.Series(['A', 'B', 'C', 'A', 'C'])
        X_train['cat2'] = pd.Series(['O', 'O', 'A', 'A', 'B'])
        X_train['num1'] = np.random.random(size=5)
        X_train['num2'] = np.arange(1, 1 + 5)

        X_test = pd.DataFrame(columns=['cat1', 'cat2', 'num1', 'num2'])
        X_test['cat1'] = pd.Series(['C', 'B', 'A', 'C', 'B'])
        X_test['cat2'] = pd.Series(['2', '1', 'A', 'O', 'B'])
        X_test['num1'] = np.random.random(size=5)
        X_test['num2'] = np.arange(3, 3 + 5)

        X_train_new, X_test_new = preprocessing.bin_features(X_train, X_test, nbins=3)

        # Check whether the numerical features were binned
        self.assertTrue(X_train_new['num1'].unique().size <= 3)
        self.assertTrue(X_train_new['num2'].unique().size <= 3)
        self.assertTrue(X_test_new['num1'].unique().size <= 3)
        self.assertTrue(X_test_new['num2'].unique().size <= 3)
        # Check whether the columns were not changed
        self.assertTrue((X_train.columns == X_train_new.columns).all())
        self.assertTrue((X_test.columns == X_test_new.columns).all())
        # Check whether there is the same amount of data
        self.assertEqual(X_train.shape, X_train_new.shape)
        self.assertEqual(X_test.shape, X_test_new.shape)
        # Check whether non-numerical features were not modified
        self.assertTrue((X_train[['cat1', 'cat2']] == X_train_new[['cat1', 'cat2']]).all().all())
        self.assertTrue((X_test[['cat1', 'cat2']] == X_test_new[['cat1', 'cat2']]).all().all())


    def test_one_hot_encode(self):
        X = pd.DataFrame(columns=['only_nan', 'one_nan', 'no_nan'])
        X['only_nan'] = pd.Series([np.nan] * 5)
        X['one_nan'] = pd.Series(['A', np.nan, 'A', 'B', 'B'])
        X['no_nan'] = pd.Series(['+', '+', '-', '-', '.'])

        X_new = preprocessing.one_hot_encode(X)

        # Check whether the columns of the two dfs correspond
        self.assertTrue((set(X.columns) == set(X_new.columns.levels[0])))
        # Check whether both dfs have the same number of points
        self.assertEqual(X.shape[0], X_new.shape[0])

        # Check that the column with only nans is properly encoded
        self.assertTrue((X_new[('only_nan', 'nan')] == 1).all())
        # Check that the column with one nan is properly encoded
        self.assertTrue((set(X_new['one_nan'].columns) == {'A', 'B', 'nan'}))
        self.assertTrue((X_new[('one_nan', 'nan')] == 1).any())
        # Check that the column with no nans is properly encoded
        self.assertTrue((set(X_new['no_nan'].columns) == {'+', '-', '.'}))

        # Check that each column's categories sum up to 1
        for col in X.columns:
            self.assertTrue((X_new[col].sum(axis=1) == 1).all())


if __name__ == '__main__':
    unittest.main()
